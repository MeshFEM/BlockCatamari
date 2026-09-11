/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_IMPL_H_

#include <algorithm>
#include <stdexcept>

#include <MeshFEMCore/GlobalBenchmark.hh>
#include <MeshFEMCore/Types.hh>
#include <catamari/dense_basic_linear_algebra-impl.hpp>
#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"
#include <MeshFEMCore/Parallelism.hh>

#include "trs_kernels.hpp"
#include "solve_kernels.hpp"

// Avoid repeated memory allocation/deallocation when applying permutations
// (at the cost of `right_hand_sides` worth of memory).
#define SOLVE_PERMUTE_SCRATCH 1

#define SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE 0

// // Whether to use the "Schur complement" buffers as the "work_right_hand_sides"
// // storage for out-of-place upper triangular solves.
// // The motivation is that accessing entries from the parent's buffer will be
// // less scattered/cache friendlier than going back to the full RHS vector;
// // see [Duff, Erisman, and Reid: Direct Methods for Sparse Matrices, Section 14.3].
// // This requires using num_child_diag_indices/child_rel_indices in addition to `child_indices`.
// // TODO: experiment with implementing this variant!
// #define USE_SCHUR_COMPLEMENT_STORAGE_FOR_OOP_LOWER_TRANSPOSE_SOLVE 1

namespace catamari {
namespace supernodal_ldl {

// Accumulate each serial scheduling subtree into its owned RHS and a private
// root boundary. Cache only external maps; internal rows use native block indices.
template <class Field>
void Factorization<Field>::PrepareSolveAccumulationCache() const {
  const Int count = ordering_.supernode_sizes.Size();
  if (solve_accumulation_ready_) return;
  // BENCHMARK_SCOPED_TIMER_SECTION timer("CacheSolveAccumulation");
  // A failed allocation/construction may be retried; publish readiness last.
  solve_accumulation_groups_.clear();
  solve_accumulation_external_.clear();
  solve_accumulation_group_for_root_.Resize(count);
  std::fill(solve_accumulation_group_for_root_.begin(), solve_accumulation_group_for_root_.end(), Int(-1));
  solve_accumulation_owned_.Resize(count);
  solve_accumulation_offsets_.Resize(count);
  const auto &af = ordering_.assembly_forest;
  const Int bs = lower_factor_->IndexBlockSize();
  struct Pending { Int node; int depth; };
  std::vector<Pending> pending;
  for (Int r : af.roots) pending.push_back({r, 0});
  while (!pending.empty()) {
    const auto item = pending.back(); pending.pop_back();
    const Int root = item.node;
    if (work_estimates_[root] < solve_kernels::forward_work || item.depth > solve_kernels::forward_max_depth) {
      solve_accumulation_group_for_root_[root] = solve_accumulation_groups_.size();
      solve_accumulation_groups_.push_back({root, {}});
    } else {
      for (Int ci = af.child_offsets[root]; ci < af.child_offsets[root + 1]; ++ci)
        pending.push_back({af.children[ci], item.depth + 1});
    }
  }
  {
    const Int num_groups = solve_accumulation_groups_.size();
    std::vector<Int> group_counts(num_groups), group_offsets(num_groups), max_degrees(num_groups);
    {
      // BENCHMARK_SCOPED_TIMER_SECTION timer("AccumulationCountRows");
      tbb::parallel_for(Int(0), num_groups, [&](Int g) {
        const Int root = solve_accumulation_groups_[g].root;
        const Int end = ordering_.supernode_offsets[root] + ordering_.supernode_sizes[root];
        const bool empty_boundary = lower_factor_->blocks[root].height == 0;
        Int entries = 0, max_degree = 0;
        for (Int i = solve_subtree_begin_[root]; i < solve_subtree_end_[root]; ++i) {
          const Int s = solve_postorder_[i];
          const Int blocks = lower_factor_->blocks[s].height / bs;
          Int owned;
          if (empty_boundary) owned = blocks;
          else if (s == root) owned = 0;
          else if (!blocks) owned = 0;
          else {
            const Int *beg = lower_factor_->StructureBeg(s);
            if (beg[blocks - 1] < end) owned = blocks;
            else owned = std::lower_bound(beg, beg + blocks, end) - beg;
          }
          solve_accumulation_owned_[s] = owned;
          solve_accumulation_offsets_[s] = entries; // Relative to this group's packed slice.
          entries += blocks - owned;
          max_degree = std::max(max_degree, lower_factor_->blocks[s].height);
        }
        group_counts[g] = entries;
        max_degrees[g] = max_degree;
      });
    }
    {
      // BENCHMARK_SCOPED_TIMER_SECTION timer("AccumulationAllocate");
      Int entries = 0;
      for (Int g = 0; g < num_groups; ++g) {
        group_offsets[g] = entries;
        entries += group_counts[g];
      }
      solve_accumulation_external_.resize(entries);
    }
    {
      // BENCHMARK_SCOPED_TIMER_SECTION timer("AccumulationFillMaps");
      tbb::parallel_for(Int(0), num_groups, [&](Int g) {
        auto &group = solve_accumulation_groups_[g];
        const Int root = group.root;
        const Int *boundary_beg = lower_factor_->StructureBeg(root), *boundary_end = lower_factor_->StructureEnd(root);
        for (Int i = solve_subtree_begin_[root]; i < solve_subtree_end_[root]; ++i) {
          const Int s = solve_postorder_[i];
          Int &offset = solve_accumulation_offsets_[s];
          offset += group_offsets[g];
          const Int owned = solve_accumulation_owned_[s];
          const Int length = lower_factor_->blocks[s].height / bs - owned;
          if (!length) continue;
          const Int *external = lower_factor_->StructureBeg(s) + owned;
          Int *out = solve_accumulation_external_.data() + offset;
          if (s == root) {
            for (Int j = 0; j < length; ++j) out[j] = bs * j;
            continue;
          }
          const Int *cursor = boundary_beg;
          for (Int j = 0; j < length; ++j) {
            const Int *q;
            if (j > 0) {
              // Sorted lists: advance only between the first and last match.
              while (cursor != boundary_end && *cursor < external[j]) ++cursor;
              q = cursor;
            } else {
              q = std::lower_bound(cursor, boundary_end, external[j]);
            }
            if (q == boundary_end || *q != external[j])
              throw std::logic_error("Invalid accumulation boundary map");
            out[j] = bs * (q - boundary_beg);
            cursor = q + 1;
          }
        }
        // Each group owns its scratch allocation; no shared vector grows here.
        group.scratch.resize(max_degrees[g]);
      });
    }
  }
  solve_accumulation_ready_ = true;
}

template <class Field>
void Factorization<Field>::Solve(
    BlasMatrixView<Field>* right_hand_sides, Int block_size, bool already_permuted) const {
  if (block_size != lower_factor_->IndexBlockSize())
    throw std::runtime_error("Solve kernel block size must match index block size");
  if (solve_profile_.enabled) ++solve_profile_.solve;
  SolveProfile::Scope solve_profile_scope(solve_profile_, -1, FineGrainedTimersSolve::SolvePhase);
  const bool needs_permutation = !(ordering_.permutation.Empty() || already_permuted);
  // Reorder the input into the permutation of the factorization.

  BlasMatrixView<Field> permuted_right_hand_sides = *right_hand_sides;
  if (needs_permutation) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Permute");
#if SOLVE_PERMUTE_SCRATCH
    const Int size = right_hand_sides->width * right_hand_sides->height;
    if (permute_scratch_.Size() < size)
        permute_scratch_.Resize(size);
    permuted_right_hand_sides.data = permute_scratch_.Data();
    InversePermute(block_size, ordering_.inverse_permutation, *right_hand_sides, &permuted_right_hand_sides);
#else
    Permute(ordering_.permutation, right_hand_sides);
#endif
  }

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  SolveSharedState &shared_state = solve_shared_state_;
#if CATAMARI_FINEGRAINED_TIMERS
    if (shared_state.finegrained_timers.supernodeCount() != num_supernodes)
        shared_state.finegrained_timers.allocate(num_supernodes);
#endif  // ifdef CATAMARI_FINEGRAINED_TIMERS


  const Int max_threads = get_max_num_tbb_threads();
  if (max_threads > 1) {
    if (solve_postorder_.Size() != num_supernodes) {
      // BENCHMARK_SCOPED_TIMER_SECTION timer("CacheSolveTraversal");
      solve_postorder_.Resize(num_supernodes);
      solve_subtree_begin_.Resize(num_supernodes);
      solve_subtree_end_.Resize(num_supernodes);
      const auto &af = ordering_.assembly_forest;
      std::vector<Int> pending(af.roots.begin(), af.roots.end());
      Int next = num_supernodes;
      while (!pending.empty()) {
        const Int s = pending.back();
        pending.pop_back();
        solve_postorder_[--next] = s;
        for (Int ci = af.child_offsets[s]; ci < af.child_offsets[s + 1]; ++ci)
          pending.push_back(af.children[ci]);
      }
      assert(next == 0);
      for (Int i = 0; i < num_supernodes; ++i) {
        const Int s = solve_postorder_[i];
        Int begin = i;
        for (Int ci = af.child_offsets[s]; ci < af.child_offsets[s + 1]; ++ci)
          begin = std::min(begin, solve_subtree_begin_[af.children[ci]]);
        solve_subtree_begin_[s] = begin;
        solve_subtree_end_[s] = i + 1;
      }
    }
    // Avoid thread oversubscription (in case we're not linked against sequential BLAS)
    BlasSingleThreadingObserver blas_single_threading_observer;

    // Set up the shared state holding the "supernode rhs" arrays.
    // In order to allow the number of rhs to change without updating
    // the offsets, we use a "column major" storage  where all
    // supernodes' data for the first rhs column comes first, followed
    // by the data for the second column (if any), and so on.

    {
#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE

        if (shared_state.schur_complements.Size() != num_supernodes) {
            shared_state.schur_complements.Resize(num_supernodes);
            shared_state.schur_complement_storage.Resize(num_supernodes);
        }
#else !SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
        // BENCHMARK_SCOPED_TIMER_SECTION timer("Allocate");
        const Int num_rhs = right_hand_sides->width;

        auto &scb = shared_state.schur_complement_buffers;
        Int total_degree;
        if (scb.Size() != 1) {
            // First time allocating
            scb.Resize(1);

            total_degree = 0;
            for (Int supernode = 0; supernode < num_supernodes; ++supernode)
                total_degree += lower_factor_->blocks[supernode].height;

            shared_state.schur_complements.Resize(num_supernodes);
            for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
                auto &supernode_rhs = shared_state.schur_complements[supernode];
                supernode_rhs.height = lower_factor_->blocks[supernode].height;
                supernode_rhs.leading_dim = total_degree;
            }
        }
        else {
            // The leading dimension of each schur_complements matrix view
            // is the total degree...
            if (shared_state.schur_complements.Size() != num_supernodes) throw std::runtime_error("Unexpected size change");
            total_degree = shared_state.schur_complements[0].leading_dim;
        }

        Int total_size = total_degree * num_rhs;
        Buffer<Field> &workspace_buffer = scb[0];
        bool realloc = (total_size > workspace_buffer.Size());
        if (realloc) workspace_buffer.Resize(total_size);
        bool num_rhs_changed = shared_state.schur_complements[0].width != num_rhs;

        if (realloc) {
            // std::cout << "Allocated solve workspace buffer of size "
            //           << total_size * sizeof(Field) / (1024. * 1024)
            //           << "MB" << std::endl;
            // std::cout << "This is " << total_size << " entries vs rhs size of " << right_hand_sides->width * right_hand_sides->height << std::endl;

            Int offset = 0;
            for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
                const Int degree = lower_factor_->blocks[supernode].height;
                auto &supernode_rhs = shared_state.schur_complements[supernode];
                supernode_rhs.width = num_rhs; // num_rhs must also have changed to trigger a realloc!
                supernode_rhs.data = workspace_buffer.Data() + offset;
                offset += degree;
            }
        }
        else if (num_rhs_changed) {
            // num_rhs has shrunk, meaning we just must update each supernode_rhs.width
            for (Int supernode = 0; supernode < num_supernodes; ++supernode)
                shared_state.schur_complements[supernode].width = num_rhs;
        }
#endif // SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
    }

    // Compute flop-count estimates (which usually was already filled by the factorization).
    // TODO: compute actual solve work estimates instead of reusing
    // factorization estimates? However, they should be simliar enough
    // (denoting a subtree's factorization flop count as f, the
    // corresponding solve flop count should be BigTheta(f^{2/3})).
    Buffer<double> &work_estimates = const_cast<Buffer<double> &>(work_estimates_);
    double &total_work = const_cast<double &>(total_work_);
    if (work_estimates.Size() != num_supernodes) {
        work_estimates.Resize(num_supernodes);
        // Any postorder will do...
        const auto &af = ordering_.assembly_forest;
        for (Int i = 0; i < num_supernodes; ++i) {
            const Int child_beg = af.child_offsets[i];
            const Int child_end = af.child_offsets[i + 1];

            double subtree_work = 0;
            for (Int child_index = child_beg; child_index < child_end; ++child_index)
                subtree_work += work_estimates[af.children[child_index]];
            work_estimates[i] = subtree_work + IntraNodeWorkEstimate(i, *lower_factor_);
        }

        total_work = 0;
        for (const Int& root : ordering_.assembly_forest.roots)
            total_work += work_estimates[root];
    }

#if !SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
    if (right_hand_sides->width == 1 && !control_.supernodal_pivoting &&
        control_.factorization_type == kCholeskyFactorization)
      PrepareSolveAccumulationCache();
#endif

    {
        if (block_size == 3) {
            OpenMPLowerTriangularSolve<3>(&permuted_right_hand_sides, &shared_state);
            OpenMPDiagonalSolve(&permuted_right_hand_sides);
            OpenMPLowerTransposeTriangularSolve<3>(&permuted_right_hand_sides, &shared_state);
        }
        else if (block_size == 2) {
            OpenMPLowerTriangularSolve<2>(&permuted_right_hand_sides, &shared_state);
            OpenMPDiagonalSolve(&permuted_right_hand_sides);
            OpenMPLowerTransposeTriangularSolve<2>(&permuted_right_hand_sides, &shared_state);
        }
        else {
            OpenMPLowerTriangularSolve<1>(&permuted_right_hand_sides, &shared_state);
            OpenMPDiagonalSolve(&permuted_right_hand_sides);
            OpenMPLowerTransposeTriangularSolve<1>(&permuted_right_hand_sides, &shared_state);
        }
    }

  } else {
      if (block_size == 3) {
          LowerTriangularSolve<3>(&permuted_right_hand_sides);
          DiagonalSolve(&permuted_right_hand_sides);
          LowerTransposeTriangularSolve<3>(&permuted_right_hand_sides);
      }
      else if (block_size == 2) {
          LowerTriangularSolve<2>(&permuted_right_hand_sides);
          DiagonalSolve(&permuted_right_hand_sides);
          LowerTransposeTriangularSolve<2>(&permuted_right_hand_sides);
      }
      else {
          LowerTriangularSolve<1>(&permuted_right_hand_sides);
          DiagonalSolve(&permuted_right_hand_sides);
          LowerTransposeTriangularSolve<1>(&permuted_right_hand_sides);
      }
  }

  // Reverse the factorization permutation.
  if (needs_permutation) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPermute");
#if SOLVE_PERMUTE_SCRATCH
    InversePermute(block_size, ordering_.permutation, permuted_right_hand_sides, right_hand_sides);
#else
    Permute(ordering_.inverse_permutation, right_hand_sides);
#endif
  }
}

template <class Field>
template<Int BLOCK_SIZE>
void Factorization<Field>::LowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* workspace) const {
  SolveProfile::Scope node_profile(solve_profile_, supernode, FineGrainedTimersSolve::ForwardNode);
  // Eliminate this supernode.
  const Int num_rhs = right_hand_sides->width;
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  const ConstBlasMatrixView<Field> diag_block = diagonal_factor_->blocks[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  SOLVE_START_TIMER(supernode, ForwardSolveDiag);

  // Solve against the diagonal block of the supernode.
  if (control_.supernodal_pivoting) {
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    InversePermute(permutation, &right_hand_sides_supernode);
  }
  if (is_cholesky) {
    if (right_hand_sides_supernode.width > 1)
        LeftLowerTriangularSolves(diag_block, &right_hand_sides_supernode);
    else {
        if (supernode_size < 24)
          trs_kernels::SolveLowerTri<Field, BLOCK_SIZE>::run(supernode_size, diag_block.data, diag_block.leading_dim, right_hand_sides_supernode.data);
        else TriangularSolveLeftLower(diag_block, right_hand_sides_supernode.Data());
    }
  } else {
    LeftLowerUnitTriangularSolves(diag_block, &right_hand_sides_supernode);
  }

  SOLVE_STOP_TIMER(supernode, ForwardSolveDiag);

  const ConstBlasMatrixView<Field> subdiagonal = lower_factor_->blocks[supernode].ToConst();
  if (!subdiagonal.height) {
    return;
  }

  // Handle the external updates for this supernode.
  // Note: it seems that the out-of-place update is always faster than
  // using Accelerate BLAS on Apple Silicon and always slower than
  // MKL on x86. So we select based on platform rather than
  // using the original threshold rule:
  //        if (supernode_size >= control_.forward_solve_out_of_place_supernode_threshold) {
  const Int* indices = lower_factor_->StructureBeg(supernode);
  const bool out_of_place = supernode_size >= control_.forward_solve_out_of_place_supernode_threshold;

  if (out_of_place) {
    SOLVE_START_TIMER(supernode, OutOfPlaceForwardsubUpdate);
    // Perform an out-of-place GEMM.
    BlasMatrixView<Field> work_right_hand_sides;
    work_right_hand_sides.height = subdiagonal.height;
    work_right_hand_sides.width = num_rhs;
    work_right_hand_sides.leading_dim = subdiagonal.height;
    work_right_hand_sides.data = workspace->Data();

#if 1
    // Store the updates in the workspace.
    solve_kernels::multiply<BLOCK_SIZE>(false, Field{1}, subdiagonal,
                               right_hand_sides_supernode.ToConst(), Field{0},
                               &work_right_hand_sides);
#else
    MatrixVectorProduct(Field{1}, subdiagonal, right_hand_sides_supernode.Data(), work_right_hand_sides.data);
#endif

    // Accumulate the workspace into the solution right_hand_sides.
    for (Int j = 0; j < num_rhs; ++j) {
            Field * rhs_ptr = right_hand_sides->Pointer(0, j);
      const Field *wrhs_ptr = work_right_hand_sides.Pointer(0, j);
      for (Int i = 0; i < subdiagonal.height; i += BLOCK_SIZE) {
        using Vec = VecN_T<Field, BLOCK_SIZE>; // TODO: evaluate add_strip version with restrict pointer, not using Eigen.
        using  VMap = Eigen::Map<      Vec, (BLOCK_SIZE == 2 && std::is_same<Field, double>::value) ? Eigen::Aligned16 : Eigen::Unaligned>;
        using CVMap = Eigen::Map<const Vec, (BLOCK_SIZE == 2 && std::is_same<Field, double>::value) ? Eigen::Aligned16 : Eigen::Unaligned>;
        VMap(rhs_ptr + indices[i / BLOCK_SIZE]) -= CVMap(wrhs_ptr + i);
      }
    }
    SOLVE_STOP_TIMER(supernode, OutOfPlaceForwardsubUpdate);
  } else {
    SOLVE_START_TIMER(supernode, InPlaceForwardsubUpdate);
    trs_kernels::MultiplyLowerBlock<Field, BLOCK_SIZE>::run(
        indices, supernode_start, supernode_size, subdiagonal.height,
        subdiagonal.data, subdiagonal.leading_dim, num_rhs,
        right_hand_sides->data, right_hand_sides->leading_dim);
    SOLVE_STOP_TIMER(supernode, InPlaceForwardsubUpdate);
  }
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::LowerTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* workspace) const {
  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    LowerTriangularSolveRecursion<BLOCK_SIZE>(child, right_hand_sides, workspace);
  }

  // Perform this supernode's trapezoidal solve.
  LowerSupernodalTrapezoidalSolve(supernode, right_hand_sides, workspace);
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  BENCHMARK_SCOPED_TIMER_SECTION timer("LowerTriangularSolve<" + std::to_string(BLOCK_SIZE) + ">");
  SolveProfile::Scope phase_profile(solve_profile_, -1, FineGrainedTimersSolve::ForwardPhase);

  // Allocate the workspace.
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Field> workspace(workspace_size, Field{0});

#if 0
  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LowerTriangularSolveRecursion<BLOCK_SIZE>(root, right_hand_sides, &workspace);
  }
#else
  // Any postorder will do...
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int s = 0; s < num_supernodes; ++s) {
    LowerSupernodalTrapezoidalSolve<BLOCK_SIZE>(s, right_hand_sides, &workspace);
  }
#endif

}

template <class Field>
void Factorization<Field>::DiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rhs = right_hand_sides->width;
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  if (is_cholesky) {
    // D is the identity.
    return;
  }

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrixView<Field> diagonal_right_hand_sides =
        diagonal_factor_->blocks[supernode];

    const Int supernode_size = ordering_.supernode_sizes[supernode];
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    BlasMatrixView<Field> right_hand_sides_supernode =
        right_hand_sides->Submatrix(supernode_start, 0, supernode_size,
                                    num_rhs);

    // Handle the diagonal-block portion of the supernode.
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < supernode_size; ++i) {
        right_hand_sides_supernode(i, j) /= diagonal_right_hand_sides(i, i);
      }
    }
  }
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::LowerTransposeSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* packed_input_buf) const {
  const ConstBlasMatrixView<Field> subdiagonal = lower_factor_->blocks[supernode];
  const Int num_rhs = right_hand_sides->width;

  BlasMatrixView<Field> work_right_hand_sides;
  work_right_hand_sides.height = subdiagonal.height;
  work_right_hand_sides.width = num_rhs;
  work_right_hand_sides.leading_dim = subdiagonal.height;
  work_right_hand_sides.data = packed_input_buf->Data();

  LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(supernode, right_hand_sides, work_right_hand_sides);
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::LowerTransposeSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    BlasMatrixView<Field> &work_right_hand_sides) const {
  SolveProfile::Scope node_profile(solve_profile_, supernode, FineGrainedTimersSolve::BackwardNode);
  const Int num_rhs = right_hand_sides->width;
  const bool is_selfadjoint =
      control_.factorization_type != kLDLTransposeFactorization;
  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int* indices = lower_factor_->StructureBeg(supernode);

  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  const ConstBlasMatrixView<Field> subdiagonal = lower_factor_->blocks[supernode].ToConst();
  const Int degree = subdiagonal.height;
  if constexpr (std::is_same<Field, double>::value || std::is_same<Field, float>::value) {
    if (supernode_size <= solve_kernels::Policy<BLOCK_SIZE>::fused_backward_max_size &&
        degree <= solve_kernels::Policy<BLOCK_SIZE>::fused_backward_max_degree && num_rhs == 1 &&
        control_.factorization_type == kCholeskyFactorization && !control_.supernodal_pivoting) {
      SOLVE_START_TIMER(supernode, FusedBackward);
      solve_kernels::fused_backward<BLOCK_SIZE>(diagonal_factor_->blocks[supernode].ToConst(),
          subdiagonal, right_hand_sides_supernode.data, right_hand_sides->data, indices);
      SOLVE_STOP_TIMER(supernode, FusedBackward);
      return;
    }
  }
  if (degree) {
    using Policy = solve_kernels::Policy<BLOCK_SIZE>;
    const Int size_threshold = control_.backward_solve_out_of_place_supernode_threshold < 0
        ? Policy::backward_size : control_.backward_solve_out_of_place_supernode_threshold;
    const Int degree_threshold = control_.backward_solve_out_of_place_degree_threshold < 0
        ? Policy::backward_degree : control_.backward_solve_out_of_place_degree_threshold;
    const bool out_of_place = supernode_size >= size_threshold || degree >= degree_threshold;
    if (out_of_place) {
      SOLVE_START_TIMER(supernode, OutOfPlaceBacksubUpdate);
      // Fill the work right_hand_sides.
      for (Int j = 0; j < num_rhs; ++j) {
        const Field * const  rhs_ptr =      right_hand_sides->Pointer(0, j);
              Field *       wrhs_ptr = work_right_hand_sides. Pointer(0, j);
        using   Vec = VecN_T<Field, BLOCK_SIZE>; // TODO: evaluate add_strip version with restrict pointer, not using Eigen.
        using  VMap = Eigen::Map<      Vec, (BLOCK_SIZE == 2 && std::is_same<Field, double>::value) ? Eigen::Aligned16 : Eigen::Unaligned>;
        using CVMap = Eigen::Map<const Vec, (BLOCK_SIZE == 2 && std::is_same<Field, double>::value) ? Eigen::Aligned16 : Eigen::Unaligned>;
        for (Int i = 0; i < degree; i += BLOCK_SIZE) {
            // (VMap(wrhs_ptr)) = CVMap(rhs_ptr + indices[i / BLOCK_SIZE]);
            // wrhs_ptr += BLOCK_SIZE;
            const Field *src = rhs_ptr + indices[i / BLOCK_SIZE];
            for (Int c = 0; c < BLOCK_SIZE; ++c)
              *(wrhs_ptr++) = *(src++);
        }
      }

      if (is_selfadjoint) {
        solve_kernels::multiply<BLOCK_SIZE>(true, Field{-1}, subdiagonal,
                                    work_right_hand_sides.ToConst(), Field{1},
                                    &right_hand_sides_supernode);
      } else {
        MatrixMultiplyTransposeNormal(Field{-1}, subdiagonal,
                                      work_right_hand_sides.ToConst(), Field{1},
                                      &right_hand_sides_supernode);
      }
      SOLVE_STOP_TIMER(supernode, OutOfPlaceBacksubUpdate);
    } else {
      SOLVE_START_TIMER(supernode, InPlaceBacksubUpdate);
      trs_kernels::MultiplyLowerBlockAdjoint<Field, BLOCK_SIZE>::run(
          is_selfadjoint, indices, supernode_start, supernode_size, subdiagonal.height,
          subdiagonal.data, subdiagonal.leading_dim,
          num_rhs, right_hand_sides->data, right_hand_sides->leading_dim);
      SOLVE_STOP_TIMER(supernode, InPlaceBacksubUpdate);
    }
  }

  SOLVE_START_TIMER(supernode, BackwardSolveDiag);

  // Solve against the diagonal block of this supernode.
  const ConstBlasMatrixView<Field> diag_block = diagonal_factor_->blocks[supernode];
  if (control_.factorization_type == kCholeskyFactorization) {
    if (right_hand_sides_supernode.width > 1) {
        LeftLowerAdjointTriangularSolves(diag_block, &right_hand_sides_supernode);
    }
    else {
        if (supernode_size < 24)
          trs_kernels::SolveLowerTriAdjoint<Field, BLOCK_SIZE>::run(supernode_size, diag_block.data, diag_block.leading_dim, right_hand_sides_supernode.data);
        else TriangularSolveLeftLowerAdjoint(diag_block, right_hand_sides_supernode.Data());
    }
  } else if (control_.factorization_type == kLDLAdjointFactorization) {
    LeftLowerAdjointUnitTriangularSolves(diag_block, &right_hand_sides_supernode);
  } else {
    LeftLowerTransposeUnitTriangularSolves(diag_block, &right_hand_sides_supernode);
  }
  if (control_.supernodal_pivoting) {
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    Permute(permutation, &right_hand_sides_supernode);
  }

  SOLVE_STOP_TIMER(supernode, BackwardSolveDiag);
}

template <class Field>
void Factorization<Field>::LowerTransposeTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* packed_input_buf) const {
  // Perform this supernode's trapezoidal solve.
  LowerTransposeSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                           packed_input_buf);

  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    LowerTransposeTriangularSolveRecursion(child, right_hand_sides,
                                           packed_input_buf);
  }
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  BENCHMARK_SCOPED_TIMER_SECTION timer("LowerTransposeTriangularSolve<" + std::to_string(BLOCK_SIZE) + ">");
  SolveProfile::Scope phase_profile(solve_profile_, -1, FineGrainedTimersSolve::BackwardPhase);

  // Allocate the workspace.
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Field> packed_input_buf(workspace_size);

#if 0
  // Recurse from each root of the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LowerTransposeTriangularSolveRecursion(root, right_hand_sides,
                                           &packed_input_buf);
  }
#else

#if 1
  // Any pre-order will do
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int s = num_supernodes - 1; s >= 0; --s)
      LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(s, right_hand_sides, &packed_input_buf);
#else
  std::stack<std::pair<Int, Int>> stack;
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    Int s = ordering_.assembly_forest.roots[root_index];
    LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(s, right_hand_sides, &packed_input_buf);
    stack.push({s, ordering_.assembly_forest.child_offsets[s]});
  }

  while (!stack.empty()) {
      auto &t = stack.top();
      Int s = t.first;
      Int &ci = t.second;

      if (ci < ordering_.assembly_forest.child_offsets[s + 1]) {
          const Int child = ordering_.assembly_forest.children[ci];
          LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(child, right_hand_sides, &packed_input_buf);
          stack.push({child, ordering_.assembly_forest.child_offsets[child]}); // descend
          ++ci;
      }
      else stack.pop();
  };
#endif
#endif
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_IMPL_H_
