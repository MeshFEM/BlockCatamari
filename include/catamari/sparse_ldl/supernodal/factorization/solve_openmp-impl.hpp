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
#include <stdexcept>

#include <tbb/task_group.h>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"
#include "catamari/sparse_ldl/supernodal/supernode_utils-impl.hpp"

#include "../../../../../../../src/lib/MeshFEM/GlobalBenchmark.hh"

#define USE_STACK_DFS_INSTEAD_OF_SERIAL_RECURSION 1
// USE_TLS_SCHUR_RHS: whether to use a separate thread-local workspace buffer
// for storing the "schur complement" updates in LowerTransposeSupernodalTrapezoidalSolve
// or to use the per-supernode storage buffers; theoretically this can help cache
// locality/avoid false sharing--at the cost of additional memory allocation.
#define USE_TLS_SCHUR_RHS 1

#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE && !USE_TLS_SCHUR_RHS
#error "USE_TLS_SCHUR_RHS must be set when SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE is enabled to avoid memory bugs!"
#endif

namespace catamari {
namespace supernodal_ldl {

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::OpenMPLowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    BlasMatrixView<Field>* supernode_schur_complement) const {
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
    else
        if (supernode_size < 24)
          trs_kernels::SolveLowerTri<Field, BLOCK_SIZE>::run(supernode_size, diag_block.data, diag_block.leading_dim, right_hand_sides_supernode.data);
    else TriangularSolveLeftLower(diag_block, right_hand_sides_supernode.Data());
  } else {
    LeftLowerUnitTriangularSolves(diag_block, &right_hand_sides_supernode);
  }

  FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, SolveDiag);

  const ConstBlasMatrixView<Field> subdiagonal = lower_factor_->blocks[supernode];
  if (!subdiagonal.height) {
    return;
  }

  // Store the updates in the workspace.
  FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, MultiplySubdiagonal);
  MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                             right_hand_sides_supernode.ToConst(), Field{1},
                             supernode_schur_complement);
  FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, MultiplySubdiagonal);
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

    Field *rhs_col = result.data;
    for (Int j = 0; j < num_rhs; ++j) {
      std::fill(rhs_col, rhs_col + result.height, Field{0});
      rhs_col += result.leading_dim;
    }
  };

  auto mergeChild = [this, shared_state, right_hand_sides, num_rhs, subtreeStorage](const Int parent, const Int child_index, BlasMatrixView<Field> &main_right_hand_sides) {
    FG_START_TIMER(solve_shared_state_.finegrained_timers, parent, MergeChildContributions);
    const Int child = ordering_.assembly_forest.children[child_index];

    const Int* child_indices = lower_factor_->StructureBeg(child);
    BlasMatrixView<Field>& child_right_hand_sides = shared_state->schur_complements[child];
    const Int child_degree = child_right_hand_sides.height;
    assert(child_degree == ordering_.assembly_forest.child_rel_indices_offsets[child + 1] - ordering_.assembly_forest.child_rel_indices_offsets[child]);

    const Int supernode_size = ordering_.supernode_sizes[parent];
    const Int num_child_diag_indices = ordering_.assembly_forest.num_child_diag_indices[child];

    using   Vec = VecN_T<Field, BLOCK_SIZE>;
    using  VMap = Eigen::Map<      Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
    using CVMap = Eigen::Map<const Vec, (BLOCK_SIZE == 2) ? Eigen::Aligned16 : Eigen::Unaligned>;
#if 1
    const Int *child_rel_indices = ordering_.assembly_forest.child_rel_indices.Data() + ordering_.assembly_forest.child_rel_indices_offsets[child];
    for (Int j = 0; j < num_rhs; ++j) {
        const Field* __restrict__ crhs_col = child_right_hand_sides.Pointer(0, j);
        Field*       __restrict__  rhs_col = right_hand_sides->Pointer(0, j);
        Field*       __restrict__ mrhs_col = main_right_hand_sides.Pointer(-supernode_size, j);

        for (Int i = 0; i < num_child_diag_indices; i += BLOCK_SIZE)
            VMap(rhs_col + child_indices[i]) += CVMap(crhs_col + i);

        for (Int i = num_child_diag_indices; i < child_degree; i += BLOCK_SIZE)
            VMap(mrhs_col + child_rel_indices[i]) += CVMap(crhs_col + i);
    }
#else
    for (Int j = 0; j < num_rhs; ++j) {
        Field* __restrict__  rhs_col = right_hand_sides->Pointer(0, j);
        Field* __restrict__ mrhs_col = main_right_hand_sides.Pointer(-supernode_size, j);
        const Field* __restrict__ crhs_col = child_right_hand_sides.Pointer(0, j);

        for (Int i = 0; i < num_child_diag_indices; i += BLOCK_SIZE) {
            const Int row = child_indices[i];
            VMap(rhs_col) += CVMap(crhs_col + i);
        }
        for (Int i = num_child_diag_indices, main_i = 0; i < child_degree; i += BLOCK_SIZE) {
            const Int row = child_indices[i];
            while (main_indices[main_i] != row) main_i += BLOCK_SIZE;
            VMap(mrhs_col + main_i) += CVMap(crhs_col + i);
        }
    }
#endif

#if SOLVE_USE_DYNAMIC_SCHUR_COMPLEMENT_STORAGE
    // Pop the child Schur complement from the stack.
    // Note: this will not deallocate the stack itself; that is done by the
    // parent of the serial subtree root (if run in parallel), or the
    // top-level loop over roots.
    if (subtreeStorage) subtreeStorage->free(child_right_hand_sides);
    else shared_state->schur_complement_storage[child].deallocate();
    child_right_hand_sides.data = nullptr;
#endif

    FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, MergeChildContributions);
  };

  // Recurse on this supernode's children.
  const auto &af = ordering_.assembly_forest;
  const Int child_beg = af.child_offsets[supernode];
  const Int child_end = af.child_offsets[supernode + 1];

#if 1
  const bool serialSubtree = (work_estimates_[supernode] < 1e6) || (level > 8); // Avoid excessive task scheduling overhead/use larger serial subtrees
#else
  const bool serialSubtree = level > 8;
#endif

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
    if (child_end > child_beg) mergeChild(supernode, child_beg,  scrhs);
  }
  else {
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
          OpenMPLowerSupernodalTrapezoidalSolve<BLOCK_SIZE>(s, right_hand_sides, &scrhs);
          stack.pop();
        }
    };

    // Note: subtreeStorage will be deallocated by the caller...

    return; // we already did all work for this supernode!
  }

  // Perform this supernode's trapezoidal solve.
  BlasMatrixView<Field> &scrhs = shared_state->schur_complements[supernode];
  OpenMPLowerSupernodalTrapezoidalSolve<BLOCK_SIZE>(supernode, right_hand_sides, &scrhs);
}

template <class Field>
template <Int BLOCK_SIZE>
void Factorization<Field>::OpenMPLowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides,
    SolveSharedState* shared_state) const {
  BENCHMARK_SCOPED_TIMER_SECTION timer("ParallelLowerTriangularSolve<" + std::to_string(BLOCK_SIZE) + ">");

  // Construct the map from child structures to parent fronts (in case it wasn't populated during the factorization (e.g., for left-looking))
  constructChildToParentMap(ordering_, lower_factor_.get());

  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
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
#if 1
  const bool serialSubtree = (work_estimates_[supernode] < 1e5); // Avoid excessive task scheduling overhead/use larger serial subtrees
#else
    const bool serialSubtree = level > 8; // Avoid excessive task scheduling overhead
#endif
    if (serialSubtree) {
#if USE_STACK_DFS_INSTEAD_OF_SERIAL_RECURSION
        std::stack<std::pair<Int, Int>> stack;
        stack.push({supernode, ordering_.assembly_forest.child_offsets[supernode]});
        while (!stack.empty()) {
            auto &t = stack.top();
            Int s = t.first;
            Int &ci = t.second;

            if (ci < ordering_.assembly_forest.child_offsets[s + 1]) {
                const Int child = ordering_.assembly_forest.children[ci];
#if USE_TLS_SCHUR_RHS
                LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(child, right_hand_sides, &workspace_buffer);
#else
                LowerTransposeSupernodalTrapezoidalSolve<BLOCK_SIZE>(child, right_hand_sides, shared_state->schur_complements[child]);
#endif
                stack.push({child, ordering_.assembly_forest.child_offsets[child]}); // descend
                ++ci;
            }
            else stack.pop();
        };
#else // !USE_STACK_DFS_INSTEAD_OF_SERIAL_RECURSION
        for (Int child_index = child_beg; child_index < child_end; ++child_index)
            processChild(child_index);
#endif // USE_STACK_DFS_INSTEAD_OF_SERIAL_RECURSION
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

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
