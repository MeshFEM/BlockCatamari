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

template <class Field>
void Factorization<Field>::Solve(
    BlasMatrixView<Field>* right_hand_sides, Int block_size, bool already_permuted) const {
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
    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

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

    SetNumBlasThreads(old_max_threads);

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
  // Eliminate this supernode.
  const Int num_rhs = right_hand_sides->width;
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  const ConstBlasMatrixView<Field> diag_block = diagonal_factor_->blocks[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, SolveDiag);

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
#if 1
        if (supernode_size < 24)
          trs_kernels::SolveLowerTri<Field, BLOCK_SIZE>::run(supernode_size, diag_block.data, diag_block.leading_dim, right_hand_sides_supernode.data);
        else TriangularSolveLeftLower(diag_block, right_hand_sides_supernode.Data());
#else
        TriangularSolveLeftLower(diag_block, right_hand_sides_supernode.Data());
#endif
    }
  } else {
    LeftLowerUnitTriangularSolves(diag_block, &right_hand_sides_supernode);
  }

  FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, SolveDiag);

  const ConstBlasMatrixView<Field> subdiagonal =
      lower_factor_->blocks[supernode];
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
    FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, OutOfPlaceForwardsubUpdate);
    // Perform an out-of-place GEMM.
    BlasMatrixView<Field> work_right_hand_sides;
    work_right_hand_sides.height = subdiagonal.height;
    work_right_hand_sides.width = num_rhs;
    work_right_hand_sides.leading_dim = subdiagonal.height;
    work_right_hand_sides.data = workspace->Data();

#if 1
    // Store the updates in the workspace.
    MatrixMultiplyNormalNormal(Field{1}, subdiagonal,
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
        using  VMap = Eigen::Map<      Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
        using CVMap = Eigen::Map<const Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
        VMap(rhs_ptr + indices[i]) -= CVMap(wrhs_ptr + i);
      }
    }
    FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, OutOfPlaceForwardsubUpdate);
  } else {
    FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, InPlaceForwardsubUpdate);
    trs_kernels::MultiplyLowerBlock<Field, BLOCK_SIZE>::run(
        indices, supernode_start, supernode_size, subdiagonal.height,
        subdiagonal.data, subdiagonal.leading_dim, num_rhs,
        right_hand_sides->data, right_hand_sides->leading_dim);
    FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, InPlaceForwardsubUpdate);
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
  const Int num_rhs = right_hand_sides->width;
  const bool is_selfadjoint =
      control_.factorization_type != kLDLTransposeFactorization;
  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int* indices = lower_factor_->StructureBeg(supernode);

  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  const ConstBlasMatrixView<Field> & subdiagonal =
      lower_factor_->blocks[supernode];
  const Int degree = subdiagonal.height;
  if (degree) {
    const bool out_of_place = (supernode_size >= control_.backward_solve_out_of_place_supernode_threshold) || (degree >= 100);
    if (out_of_place) {
      FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, OutOfPlaceBacksubUpdate);
      // Fill the work right_hand_sides.
      for (Int j = 0; j < num_rhs; ++j) {
        const Field * const  rhs_ptr =      right_hand_sides->Pointer(0, j);
              Field *       wrhs_ptr = work_right_hand_sides. Pointer(0, j);
        using   Vec = VecN_T<Field, BLOCK_SIZE>; // TODO: evaluate add_strip version with restrict pointer, not using Eigen.
        using  VMap = Eigen::Map<      Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
        using CVMap = Eigen::Map<const Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
        for (Int i = 0; i < degree; i += BLOCK_SIZE) {
            // (VMap(wrhs_ptr)) = CVMap(rhs_ptr + indices[i]);
            // wrhs_ptr += BLOCK_SIZE;
            const Field *src = rhs_ptr + indices[i];
            for (Int c = 0; c < BLOCK_SIZE; ++c)
              *(wrhs_ptr++) = *(src++);
        }
      }

      if (is_selfadjoint) {
        MatrixMultiplyAdjointNormal(Field{-1}, subdiagonal,
                                    work_right_hand_sides.ToConst(), Field{1},
                                    &right_hand_sides_supernode);
      } else {
        MatrixMultiplyTransposeNormal(Field{-1}, subdiagonal,
                                      work_right_hand_sides.ToConst(), Field{1},
                                      &right_hand_sides_supernode);
      }
      FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, OutOfPlaceBacksubUpdate);
    } else {
      FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, InPlaceBacksubUpdate);
      trs_kernels::MultiplyLowerBlockAdjoint<Field, BLOCK_SIZE>::run(
              is_selfadjoint, indices, supernode_start, supernode_size, subdiagonal.height,
              subdiagonal.data, subdiagonal.leading_dim,
              num_rhs, right_hand_sides->data, right_hand_sides->leading_dim);
      FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, InPlaceBacksubUpdate);
    }
  }

  FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, SolveDiag);

  // Solve against the diagonal block of this supernode.
  const ConstBlasMatrixView<Field> diag_block = diagonal_factor_->blocks[supernode];
  if (control_.factorization_type == kCholeskyFactorization) {
    if (right_hand_sides_supernode.width > 1) {
        LeftLowerAdjointTriangularSolves(diag_block, &right_hand_sides_supernode);
    }
    else {
#if 1
        if (supernode_size < 24)
          trs_kernels::SolveLowerTriAdjoint<Field, BLOCK_SIZE>::run(supernode_size, diag_block.data, diag_block.leading_dim, right_hand_sides_supernode.data);
        else TriangularSolveLeftLowerAdjoint(diag_block, right_hand_sides_supernode.Data());
#else
        TriangularSolveLeftLowerAdjoint(diag_block, right_hand_sides_supernode.Data());
#endif
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

  FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, SolveDiag);
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
