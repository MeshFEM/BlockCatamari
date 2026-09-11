/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_

#include <algorithm>
#include <catamari/dense_basic_linear_algebra-impl.hpp>
#include <queue>
#include <bitset>
#include <stdexcept>

#include <tbb/task_group.h>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"
#include "catamari/sparse_ldl/supernodal/supernode_utils-impl.hpp"

#include <MeshFEMCore/GlobalBenchmark.hh>

// USE_TLS_SCHUR_RHS: whether to use a separate thread-local workspace buffer
// for storing the "schur complement" updates in LowerTransposeSupernodalTrapezoidalSolve
// or to use the per-supernode storage buffers; theoretically this can help cache
// locality/avoid false sharing--at the cost of additional memory allocation.
#define USE_TLS_SCHUR_RHS 0

#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE && !USE_TLS_SCHUR_RHS
#error "USE_TLS_SCHUR_RHS must be set when SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE is enabled to avoid memory bugs!"
#endif

#define USE_ORIGINAL_LOWER_TRI_MERGE_SOLVE 1

namespace catamari {
namespace supernodal_ldl {

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::OpenMPLowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    BlasMatrixView<Field>* supernode_schur_complement, bool initialize_output) const {
  SolveProfile::Scope node_profile(solve_profile_, supernode, FineGrainedTimersSolve::ForwardNode);
  const Int num_rhs = right_hand_sides->width;
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  const ConstBlasMatrixView<Field> diag_block = diagonal_factor_->blocks[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);


  if (is_cholesky && !control_.supernodal_pivoting && num_rhs == 1 &&
      supernode_size <= solve_kernels::Policy<BLOCK_SIZE>::fused_forward_max_size) {
    SOLVE_START_TIMER(supernode, FusedForward);
    solve_kernels::fused_forward(diag_block, lower_factor_->blocks[supernode].ToConst(),
        right_hand_sides_supernode.data, supernode_schur_complement->data, initialize_output);
    SOLVE_STOP_TIMER(supernode, FusedForward);
    return;
  }

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

  // Store the updates in the workspace.
  SOLVE_START_TIMER(supernode, MultiplySubdiagonal);
  solve_kernels::multiply<BLOCK_SIZE>(false, Field{-1}, subdiagonal,
                             right_hand_sides_supernode.ToConst(), initialize_output ? Field{0} : Field{1},
                             supernode_schur_complement);
  SOLVE_STOP_TIMER(supernode, MultiplySubdiagonal);
}

template <class Field>
template<Int BLOCK_SIZE>
void Factorization<Field>::SolveAccumulationGroupForward(
    Int group_index, BlasMatrixView<Field>* rhs, SolveSharedState* shared_state) const {
  auto &group = solve_accumulation_groups_[group_index];
  auto &boundary = shared_state->schur_complements[group.root];
  SOLVE_START_TIMER(group.root, InitializeSchur);
  if (boundary.height) std::fill(boundary.data, boundary.data + boundary.height, Field(0));
  SOLVE_STOP_TIMER(group.root, InitializeSchur);
  for (Int i = solve_subtree_begin_[group.root]; i < solve_subtree_end_[group.root]; ++i) {
    const Int s = solve_postorder_[i];
    const auto &lower = lower_factor_->blocks[s].ToConst();
    const Int owned = solve_accumulation_owned_[s];
    const Int *indices = lower_factor_->StructureBeg(s);
    const Int *external = solve_accumulation_external_.empty() ? nullptr : solve_accumulation_external_.data() + solve_accumulation_offsets_[s];
    if (ordering_.supernode_sizes[s] <= solve_kernels::Policy<BLOCK_SIZE>::fused_forward_max_size) {
      SolveProfile::Scope node_profile(solve_profile_, s, FineGrainedTimersSolve::ForwardNode);
      SOLVE_START_TIMER(s, FusedForward);
      solve_kernels::fused_forward_scatter<BLOCK_SIZE>(diagonal_factor_->blocks[s].ToConst(), lower,
          rhs->data + ordering_.supernode_offsets[s], rhs->data, boundary.data, indices, owned, external);
      SOLVE_STOP_TIMER(s, FusedForward);
    } else {
      BlasMatrixView<Field> scratch;
      scratch.data = group.scratch.data(); scratch.height = lower.height;
      scratch.width = 1; scratch.leading_dim = lower.height;
      OpenMPLowerSupernodalTrapezoidalSolve<BLOCK_SIZE>(s, rhs, &scratch, true);
      SOLVE_START_TIMER(s, MergeChildContributions);
      using Vec = Eigen::Matrix<Field, BLOCK_SIZE, 1>;
      for (Int j = 0; j < owned; ++j)
        Eigen::Map<Vec>(rhs->data + indices[j]) += Eigen::Map<const Vec>(scratch.data + BLOCK_SIZE * j);
      for (Int j = owned; j < lower.height / BLOCK_SIZE; ++j)
        Eigen::Map<Vec>(boundary.data + external[j - owned]) += Eigen::Map<const Vec>(scratch.data + BLOCK_SIZE * j);
      SOLVE_STOP_TIMER(s, MergeChildContributions);
    }
  }
}

template <class Field>
template<Int BLOCK_SIZE>
void Factorization<Field>::OpenMPLowerTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    SolveSharedState* shared_state, int level) const {

  auto processChild = [&, shared_state, right_hand_sides, level](Int child_index) {
      const Int child = ordering_.assembly_forest.children[child_index];
      OpenMPLowerTriangularSolveRecursion<BLOCK_SIZE>(child, right_hand_sides, shared_state, level + 1);
  };

  // Merge the child rhs contributions into the parent.
  const Int num_rhs = right_hand_sides->width;

  SchurComplementStorage<Field, /* VectorOnly = */ true> *subtreeStorage = nullptr;

  auto prepare_schur_complement_rhs = [this, shared_state, num_rhs, subtreeStorage](Int s, BlasMatrixView<Field> &result) {
#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
    if (num_rhs > 1) throw std::runtime_error("Multi-rhs not supported in this mode yet");
    const Int degree = lower_factor_->blocks[s].height;
    if (subtreeStorage) result = subtreeStorage->push(degree);
    else                result = shared_state->schur_complement_storage[s].allocateSingleMatrixForDegree(degree);
#endif

    SOLVE_START_TIMER(s, InitializeSchur);
    Field *rhs_col = result.data;
    for (Int j = 0; j < num_rhs; ++j) {
      std::fill(rhs_col, rhs_col + result.height, Field{0});
      rhs_col += result.leading_dim;
    }
    SOLVE_STOP_TIMER(s, InitializeSchur);
  };

  auto mergeChild = [this, shared_state, right_hand_sides, num_rhs, subtreeStorage](const Int parent, const Int child_index, BlasMatrixView<Field> &main_right_hand_sides) {
    SOLVE_START_TIMER(parent, MergeChildContributions);
    const Int child = ordering_.assembly_forest.children[child_index];

    const Int* child_indices = lower_factor_->StructureBeg(child);
    BlasMatrixView<Field>& child_right_hand_sides = shared_state->schur_complements[child];
    const Int child_degree = child_right_hand_sides.height;
    assert(child_degree == BLOCK_SIZE * (ordering_.assembly_forest.child_rel_indices_offsets[child + 1] - ordering_.assembly_forest.child_rel_indices_offsets[child]));

    const Int supernode_size = ordering_.supernode_sizes[parent];
    const Int num_child_diag_indices = ordering_.assembly_forest.num_child_diag_indices[child];

    using   Vec = VecN_T<Field, BLOCK_SIZE>;
    using  VMap = Eigen::Map<      Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
    using CVMap = Eigen::Map<const Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
    const Int *child_rel_indices = ordering_.assembly_forest.child_rel_indices.Data() + ordering_.assembly_forest.child_rel_indices_offsets[child];
    for (Int j = 0; j < num_rhs; ++j) {
        const Field* CATAMARI_RESTRICT crhs_col = child_right_hand_sides.Pointer(0, j);
        Field*       CATAMARI_RESTRICT  rhs_col = right_hand_sides->Pointer(0, j);
        Field*       CATAMARI_RESTRICT mrhs_col = main_right_hand_sides.Pointer(-supernode_size, j);

        for (Int i = 0; i < num_child_diag_indices; i += BLOCK_SIZE)
            VMap(rhs_col + child_indices[i / BLOCK_SIZE]) += CVMap(crhs_col + i);

        for (Int i = num_child_diag_indices; i < child_degree; i += BLOCK_SIZE)
            VMap(mrhs_col + child_rel_indices[i / BLOCK_SIZE]) += CVMap(crhs_col + i);
    }

#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
    // Pop the child Schur complement from the stack.
    // Note: this will not deallocate the stack itself; that is done by the
    // parent of the serial subtree root (if run in parallel), or the
    // top-level loop over roots.
    if (subtreeStorage) subtreeStorage->free(child_right_hand_sides);
    else shared_state->schur_complement_storage[child].deallocate();
    child_right_hand_sides.data = nullptr;
#endif

    SOLVE_STOP_TIMER(parent, MergeChildContributions);
  };

  // Recurse on this supernode's children.
  const auto &af = ordering_.assembly_forest;
  const Int child_beg = af.child_offsets[supernode];
  const Int child_end = af.child_offsets[supernode + 1];
  auto finishNode = [&](Int s, BlasMatrixView<Field> &scrhs) {
      OpenMPLowerSupernodalTrapezoidalSolve<BLOCK_SIZE>(s, right_hand_sides, &scrhs);
  };

  const bool serialSubtree = (work_estimates_[supernode] < solve_kernels::forward_work) || (level > solve_kernels::forward_max_depth);

  if ((child_end - child_beg) > 1 && !serialSubtree) {
      tbb::task_group group;
      for (Int child_index = child_beg; child_index < child_end - 1; ++child_index) {
          group.run([&processChild, child_index]() { processChild(child_index); });
      }
      processChild(child_end - 1);
      group.wait();

      BlasMatrixView<Field> &scrhs = shared_state->schur_complements[supernode];
      prepare_schur_complement_rhs(supernode, scrhs);
      for (Int child_index = child_beg; child_index < child_end; ++child_index)
          mergeChild(supernode, child_index, scrhs);
  }
  else if (!serialSubtree) {
    assert(child_end - child_beg <= 1);
    // one or no children
    if (child_end > child_beg) processChild(child_beg);
    BlasMatrixView<Field> &scrhs = shared_state->schur_complements[supernode];
    prepare_schur_complement_rhs(supernode, scrhs);
    if (child_end > child_beg) mergeChild(supernode, child_beg, scrhs);
  }
  else {
#if !SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
    if (num_rhs == 1 && !control_.supernodal_pivoting &&
        control_.factorization_type == kCholeskyFactorization) {
      const Int group = solve_accumulation_group_for_root_[supernode];
      assert(group >= 0 && solve_accumulation_groups_[group].root == supernode);
      SolveAccumulationGroupForward<BLOCK_SIZE>(group, right_hand_sides, shared_state);
      return;
    }
    // Multiple RHS and LDL retain the original child-merge order.
    for (Int i = solve_subtree_begin_[supernode]; i < solve_subtree_end_[supernode]; ++i) {
      const Int s = solve_postorder_[i];
      auto &scrhs = shared_state->schur_complements[s];
      prepare_schur_complement_rhs(s, scrhs);
      for (Int ci = af.child_offsets[s]; ci < af.child_offsets[s + 1]; ++ci)
        mergeChild(s, ci, scrhs);
      finishNode(s, scrhs);
    }
    return;
#else
#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
    // Process this subtree serially. We use a stack-based DFS to avoid
    // passing more parameters through recursion.
    subtreeStorage = &(shared_state->schur_complement_storage[supernode]);
    subtreeStorage->reallocate(subtreeStorage->getStoragedNeeded(supernode, ordering_.assembly_forest, *lower_factor_));
#endif

    std::stack<std::pair<Int, Int>> stack;
    stack.push({supernode, child_beg});
    while (!stack.empty()) {
        auto &t = stack.top();
        Int s = t.first;
        Int &ci = t.second;

        const Int cb = af.child_offsets[s];
        const Int ce = af.child_offsets[s + 1];

        BlasMatrixView<Field> &scrhs = shared_state->schur_complements[s];
        if (ci > cb) { // a child has just finished processing
          if (ci == cb + 1) prepare_schur_complement_rhs(s, scrhs); // first child
          mergeChild(s, ci - 1, scrhs);
        }

        if (ci < ce) {
            const Int child = af.children[ci];
            stack.push({child, af.child_offsets[child]}); // descend to process next child
            ++ci;
        }
        else { // last child was processed
          if (ci == cb) prepare_schur_complement_rhs(s, scrhs); // there were no children...
          finishNode(s, scrhs);
          stack.pop();
        }
    };

    // Note: subtreeStorage will be deallocated by the caller...

    return; // we already did all work for this supernode!
#endif
  }

  // Perform this supernode's trapezoidal solve.
  BlasMatrixView<Field> &scrhs = shared_state->schur_complements[supernode];
  finishNode(supernode, scrhs);
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::OpenMPLowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides,
    SolveSharedState* shared_state) const {
  BENCHMARK_SCOPED_TIMER_SECTION timer("ParallelLowerTriangularSolve<" + std::to_string(BLOCK_SIZE) + ">");
    SolveProfile::Scope phase_profile(solve_profile_, -1, FineGrainedTimersSolve::ForwardPhase);

  const Int num_roots = ordering_.assembly_forest.roots.Size();

#if USE_ORIGINAL_LOWER_TRI_MERGE_SOLVE
  // Construct the map from child structures to parent fronts (in case it wasn't populated during the factorization (e.g., for left-looking))
  constructChildToParentMap(ordering_, lower_factor_.get());

  // Recurse on each tree in the elimination forest.
  tbb::task_group tg;
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
      tg.run([right_hand_sides, shared_state, root_index, &tg, this]() {
         const Int root = ordering_.assembly_forest.roots[root_index];
         OpenMPLowerTriangularSolveRecursion<BLOCK_SIZE>(root, right_hand_sides, shared_state, 0);
#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
         shared_state->schur_complement_storage[root].deallocate();
#endif
      });
  }
  tg.wait();
#else // !USE_ORIGINAL_LOWER_TRI_MERGE_SOLVE
  if (right_hand_sides->width > 1) throw std::runtime_error("ParallelLowerTriangularSolveThreadLocalRHSRecursion does not yet support multiple RHS");

  const Int nt = tbb::this_task_arena::max_concurrency();
  thread_local_solve_data.clear();
  thread_local_solve_data.resize(nt);

  // Allocate the workspaces
  Buffer<Buffer<Field>> thread_workspaces(nt);

  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    std::bitset<MAX_THREADS> contributingSubtreeThreads;
	ParallelLowerTriangularSolveThreadLocalRHSRecursion<BLOCK_SIZE>(root, right_hand_sides, &thread_workspaces, contributingSubtreeThreads, /* level = */ 0);
  }
#endif // USE_STACK_DFS_INSTEAD_OF_SERIAL_RECURSION
}

template <class Field>
void Factorization<Field>::OpenMPDiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control_.factorization_type == kCholeskyFactorization) {
    // D is the identity.
    return;
  }

  const SymmetricOrdering* ordering_ptr = &ordering_;
  const DiagonalFactor<Field>* diagonal_factor_ptr = diagonal_factor_.get();

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  // TODO(JP): re-parallelize
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    {
      const ConstBlasMatrixView<Field> diagonal_right_hand_sides =
          diagonal_factor_ptr->blocks[supernode];

      const Int num_rhs = right_hand_sides->width;
      const Int supernode_size = ordering_ptr->supernode_sizes[supernode];
      const Int supernode_start = ordering_ptr->supernode_offsets[supernode];
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
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::OpenMPLowerTransposeTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    SolveSharedState* shared_state, int level, tbb::task_group &tg) const {
    // Perform this supernode's trapezoidal solve.
#if USE_TLS_SCHUR_RHS
    Buffer<Field> &workspace_buffer = thread_local_solve_data[tbb::this_task_arena::current_thread_index()];
    LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(supernode, right_hand_sides, &workspace_buffer);
#else
    LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(supernode, right_hand_sides, shared_state->schur_complements[supernode]);
#endif

    auto processChild = [right_hand_sides, shared_state, level, &tg, this](Int child_index) {
        const Int child = ordering_.assembly_forest.children[child_index];
        OpenMPLowerTransposeTriangularSolveRecursion<BLOCK_SIZE>(child, right_hand_sides, shared_state, level + 1, tg);
    };

    const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
    const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
    const Int numChildren = child_end - child_beg;
    if (numChildren <= 1) {
        if (numChildren == 1) processChild(child_beg);
        return;
    }
  const bool serialSubtree = (work_estimates_[supernode] < solve_kernels::backward_work) || (level > solve_kernels::backward_max_depth);
    if (serialSubtree) {
        // The root has already been solved. Reverse postorder visits each
        // descendant after its parent; sibling solves are independent.
        const Int count = solve_subtree_end_[supernode] - solve_subtree_begin_[supernode] - 1;
        for (Int offset = 0; offset < count; ++offset) {
            const Int child = solve_postorder_[solve_subtree_end_[supernode] - 2 - offset];
#if USE_TLS_SCHUR_RHS
            LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(child, right_hand_sides, &workspace_buffer);
#else
            LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(child, right_hand_sides, shared_state->schur_complements[child]);
#endif
        }
        return;
    }

    // Parallel tail recursion
    for (Int child_index = child_beg; child_index < child_end - 1; ++child_index)
        tg.run([processChild, child_index]() { processChild(child_index); });
    processChild(child_end - 1);
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::OpenMPLowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides,
    SolveSharedState* shared_state) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("ParallelTransposeTriangularSolve<" + std::to_string(BLOCK_SIZE) + ">");
    SolveProfile::Scope phase_profile(solve_profile_, -1, FineGrainedTimersSolve::BackwardPhase);

#if USE_TLS_SCHUR_RHS
    const Int nt = tbb::this_task_arena::max_concurrency();
    thread_local_solve_data.resize(nt);
    for (Int t = 0; t < nt; ++t)
        thread_local_solve_data[t].Resize(max_degree_);
#endif

    const Int num_roots = ordering_.assembly_forest.roots.Size();
    if (num_roots == 0) return;

    // Tail recurse from each root of the elimination forest.
    tbb::task_group tg;
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
        tg.run([right_hand_sides, shared_state, root_index, &tg, this]() {
            OpenMPLowerTransposeTriangularSolveRecursion<BLOCK_SIZE>(ordering_.assembly_forest.roots[root_index], right_hand_sides, shared_state, 0, tg);
        });
    }
    tg.wait();
}

template <class Field>
template<Int BLOCK_SIZE>
void Factorization<Field>::ParallelLowerTriangularSolveThreadLocalRHSRecursion(
	Int supernode, BlasMatrixView<Field>* right_hand_sides, Buffer<Buffer<Field>> *thread_workspaces,
    std::bitset<MAX_THREADS> &contributingSubtreeThreads, int level) const {
  // Set up/access thread-local RHS and workspace vectors.
  const Int nt = thread_workspaces->Size();
  Int thread = tbb::this_task_arena::current_thread_index();
  Buffer<Field> &rhs_copy = thread_local_solve_data[thread];
  if (rhs_copy.Size() == 0) rhs_copy.Resize(right_hand_sides->height, Field{0});
  Buffer<Field> &workspace = (*thread_workspaces)[thread];
  if (workspace.Size() == 0) workspace.Resize(max_degree_);

  BlasMatrixView<Field> rhs_copy_bmv;
  rhs_copy_bmv.height = right_hand_sides->height;
  rhs_copy_bmv.width = 1;
  rhs_copy_bmv.leading_dim = right_hand_sides->height;
  rhs_copy_bmv.data = rhs_copy.Data();

  auto solve_supernode = [this, &rhs_copy_bmv, right_hand_sides, &workspace, nt, thread, &contributingSubtreeThreads](Int s, bool serial) {
    const Int supernode_offset = ordering_.supernode_offsets[s];
    const Int supernode_size   = ordering_.supernode_sizes[s];

    Eigen::Map<VecX_T<Field>>(rhs_copy_bmv.data + supernode_offset, supernode_size) +=
      Eigen::Map<const VecX_T<Field>>(right_hand_sides->data + supernode_offset, supernode_size);

    if (!serial) {
      // Merge in contributions from the other threads.
      for (Int other_thread = 0; other_thread < nt; ++other_thread) {
        if ((thread == other_thread) || !contributingSubtreeThreads.test(other_thread)) continue;
        Eigen::Map<VecX_T<Field>>(rhs_copy_bmv.data + supernode_offset, supernode_size) +=
          Eigen::Map<const VecX_T<Field>>(thread_local_solve_data[other_thread].Data() + supernode_offset, supernode_size);
      }
    }

    LowerSupernodalTrapezoidalSolve<BLOCK_SIZE>(s, &rhs_copy_bmv, &workspace);

    // Finalize supernode's result.
    Eigen::Map<VecX_T<Field>>(right_hand_sides->data + supernode_offset, supernode_size) =
      Eigen::Map<const VecX_T<Field>>(rhs_copy_bmv.data + supernode_offset, supernode_size);
  };

  contributingSubtreeThreads.set(thread);

  const auto &af = ordering_.assembly_forest;
  const Int child_beg = af.child_offsets[supernode];
  const Int child_end = af.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  const bool serial = level > 6;
  if (!serial && (num_children > 1)) {
    Buffer<std::bitset<MAX_THREADS>> childContributingSubtreeThreads(num_children);
    tbb::task_group tg;
    for (Int child_index = 0; child_index < num_children; ++child_index) {
        const Int child = af.children[child_beg + child_index];
        std::bitset<MAX_THREADS> &childContributingSubtreeThread = childContributingSubtreeThreads[child_index];
        tg.run([this, child, right_hand_sides, thread_workspaces, &childContributingSubtreeThread, level]() {
          ParallelLowerTriangularSolveThreadLocalRHSRecursion<BLOCK_SIZE>(child, right_hand_sides, thread_workspaces, childContributingSubtreeThread, level + 1);
        });
    }
    tg.wait();

    for (Int child_index = 0; child_index < num_children; ++child_index)
      contributingSubtreeThreads |= childContributingSubtreeThreads[child_index];
  }
  else if (!serial && (num_children > 0)) {
      const Int child = af.children[child_beg];
      ParallelLowerTriangularSolveThreadLocalRHSRecursion<BLOCK_SIZE>(child, right_hand_sides, thread_workspaces, contributingSubtreeThreads, level + 1);
  }
  else {
    std::stack<std::pair<Int, Int>> stack;
    stack.push({supernode, child_beg});
    while (!stack.empty()) {
      auto &t = stack.top();
      Int s = t.first;
      Int &ci = t.second;

      const Int cb = af.child_offsets[s];
      const Int ce = af.child_offsets[s + 1];

      if (ci < ce) {
        const Int child = af.children[ci];
        stack.push({child, af.child_offsets[child]}); // descend to process next child
        ++ci;
      }
      else { // last child was processed
        solve_supernode(s, serial);
        stack.pop();
      }
    };
    return; // subtree root has been processed!
  }

  solve_supernode(supernode, serial);
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
