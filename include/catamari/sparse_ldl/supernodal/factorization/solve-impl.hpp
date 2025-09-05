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

#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/Types.hh>
#include <catamari/dense_basic_linear_algebra-impl.hpp>
#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"
#include <MeshFEM/Parallelism.hh>

// Avoid repeated memory allocation/deallocation when applying permutations
// (at the cost of `right_hand_sides` worth of memory).
#define SOLVE_PERMUTE_SCRATCH 1

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::Solve(
    BlasMatrixView<Field>* right_hand_sides, bool already_permuted) const {
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
    InversePermute(ordering_.inverse_permutation, *right_hand_sides, &permuted_right_hand_sides);
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
    }

    {
        OpenMPLowerTriangularSolve(&permuted_right_hand_sides, &shared_state);
        OpenMPDiagonalSolve(&permuted_right_hand_sides);
        OpenMPLowerTransposeTriangularSolve(&permuted_right_hand_sides, &shared_state);
    }

    SetNumBlasThreads(old_max_threads);

  } else {
      LowerTriangularSolve(&permuted_right_hand_sides);
      DiagonalSolve(&permuted_right_hand_sides);
      LowerTransposeTriangularSolve(&permuted_right_hand_sides);
  }

  // Reverse the factorization permutation.
  if (needs_permutation) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPermute");
#if SOLVE_PERMUTE_SCRATCH
    InversePermute(ordering_.permutation, permuted_right_hand_sides, right_hand_sides);
#else
    Permute(ordering_.inverse_permutation, right_hand_sides);
#endif
  }
}

template <class Field>
void Factorization<Field>::LowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* workspace) const {
  // Eliminate this supernode.
  const Int num_rhs = right_hand_sides->width;
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  const ConstBlasMatrixView<Field> triangular_right_hand_sides =
      diagonal_factor_->blocks[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  // Solve against the diagonal block of the supernode.
  if (control_.supernodal_pivoting) {
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    InversePermute(permutation, &right_hand_sides_supernode);
  }
  if (is_cholesky) {
    if (right_hand_sides_supernode.width > 1)
        LeftLowerTriangularSolves(triangular_right_hand_sides,
                                 &right_hand_sides_supernode);
    else
        TriangularSolveLeftLower(triangular_right_hand_sides,
                                 right_hand_sides_supernode.Data());
  } else {
    LeftLowerUnitTriangularSolves(triangular_right_hand_sides,
                                  &right_hand_sides_supernode);
  }

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
#if defined(__APPLE__)
  static constexpr bool out_of_place = false;
#else
  static constexpr bool out_of_place = true;
#endif
  if (out_of_place) {
    // Perform an out-of-place GEMM.
    BlasMatrixView<Field> work_right_hand_sides;
    work_right_hand_sides.height = subdiagonal.height;
    work_right_hand_sides.width = num_rhs;
    work_right_hand_sides.leading_dim = subdiagonal.height;
    work_right_hand_sides.data = workspace->Data();

    // Store the updates in the workspace.
    MatrixMultiplyNormalNormal(Field{1}, subdiagonal,
                               right_hand_sides_supernode.ToConst(), Field{0},
                               &work_right_hand_sides);

    // Accumulate the workspace into the solution right_hand_sides.
    for (Int j = 0; j < num_rhs; ++j) {
            Field * rhs_ptr = right_hand_sides->Pointer(0, j);
      const Field *wrhs_ptr = work_right_hand_sides.Pointer(0, j);
      for (Int i = 0; i < subdiagonal.height; ++i) {
        rhs_ptr[indices[i]] -= wrhs_ptr[i];
      }
    }
  } else {
    for (Int j = 0; j < num_rhs; ++j) {
      const Field *srhs_ptr = right_hand_sides_supernode.Pointer(0, j);
            Field * rhs_ptr = right_hand_sides->         Pointer(0, j);
#if 0
      for (Int k = 0; k < supernode_size; ++k) {
        const Field eta = srhs_ptr[k];
        const Field *subdiag_ptr = subdiagonal.Pointer(0, k);
        for (Int i = 0; i < subdiagonal.height; ++i)
          rhs_ptr[indices[i]] -= subdiag_ptr[i] * eta;
      }
#else
      // Julian Panetta: this ordering is measurably faster...
      //  for (Int i = 0; i < subdiagonal.height; ++i) {
      //     Field val = 0;
      //     for (Int k = 0; k < supernode_size; ++k)
      //       val += subdiagonal(i, k) * srhs_ptr[k];
      //     rhs_ptr[indices[i]] -= val;
      //  }

#if 1
      constexpr Int CHUNK_SIZE = 4;
      using Vec = VecN_T<Field, CHUNK_SIZE>;
      Int i;
      for (i = 0; i <= subdiagonal.height - CHUNK_SIZE; i += CHUNK_SIZE) {
         Vec val = Eigen::Map<const Vec>(subdiagonal.Pointer(i, 0)) * srhs_ptr[0];
         for (Int k = 1; k < supernode_size; ++k)
             val += Eigen::Map<const Vec>(subdiagonal.Pointer(i, k)) * srhs_ptr[k];
         rhs_ptr[indices[i + 0]] -= val[0];
         rhs_ptr[indices[i + 1]] -= val[1];
         rhs_ptr[indices[i + 2]] -= val[2];
         rhs_ptr[indices[i + 3]] -= val[3];
         // rhs_ptr[indices[i + 4]] -= val[4];
         // rhs_ptr[indices[i + 5]] -= val[5];
         // rhs_ptr[indices[i + 6]] -= val[6];
         // rhs_ptr[indices[i + 7]] -= val[7];
      }
      for (; i < subdiagonal.height; ++i) {
         Field val = subdiagonal(i, 0) * srhs_ptr[0];
         for (Int k = 1; k < supernode_size; ++k)
             val += subdiagonal(i, k) * srhs_ptr[k];
         rhs_ptr[indices[i]] -= val;
      }
#else
      const Field *subd_ptr_base = subdiagonal.data;
      for (Int i = 0; i < subdiagonal.height; ++i) {
        const Field *subd_ptr = subd_ptr_base;
        Field val = (*subd_ptr) * srhs_ptr[0];
        for (Int k = 1; k < supernode_size; ++k) {
          subd_ptr += subdiagonal.leading_dim;
          val += (*subd_ptr) * srhs_ptr[k];
        }
        rhs_ptr[indices[i]] -= val;
        ++subd_ptr_base;
      }
#endif
#endif
    }
  }
}

template <class Field>
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
    LowerTriangularSolveRecursion(child, right_hand_sides, workspace);
  }

  // Perform this supernode's trapezoidal solve.
  LowerSupernodalTrapezoidalSolve(supernode, right_hand_sides, workspace);
}

template <class Field>
void Factorization<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  BENCHMARK_SCOPED_TIMER_SECTION timer("LowerTriangularSolve");

  // Allocate the workspace.
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Field> workspace(workspace_size, Field{0});

#if 0
  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LowerTriangularSolveRecursion(root, right_hand_sides, &workspace);
  }
#else
  // Any postorder will do...
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int s = 0; s < num_supernodes; ++s) {
    LowerSupernodalTrapezoidalSolve(s, right_hand_sides, &workspace);
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

  LowerTransposeSupernodalTrapezoidalSolve(supernode, right_hand_sides, work_right_hand_sides);
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
  if (subdiagonal.height) {

    // Handle the external updates for this supernode.
    // Note: it seems that the out-of-place update is always faster than
    // using Accelerate BLAS on Apple Silicon and always slower than
    // MKL on x86. So we select based on platform rather than
    // using the original threshold rule:
    //      if (supernode_size >= control_.backward_solve_out_of_place_supernode_threshold) {
#if defined(__APPLE__)
      static constexpr bool out_of_place = false;
      // bool out_of_place = supernode_size >= control_.backward_solve_out_of_place_supernode_threshold;
#else
      static constexpr bool out_of_place = true;
#endif
    if (out_of_place) {
      FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, OutOfPlaceBacksubUpdate);
      // Fill the work right_hand_sides.
      for (Int j = 0; j < num_rhs; ++j) {
        const Field * const  rhs_ptr =      right_hand_sides->Pointer(0, j);
              Field *       wrhs_ptr = work_right_hand_sides. Pointer(0, j);
        using VecBlock = VecN_T<Field, BLOCK_SIZE>;
        for (Int i = 0; i < subdiagonal.height; i += BLOCK_SIZE) {
          const Field * src = rhs_ptr + indices[i];
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
      for (Int j = 0; j < num_rhs; ++j) {
        const Field * const  rhs_ptr = right_hand_sides         ->Pointer(0, j);
              Field * const srhs_ptr = right_hand_sides_supernode.Pointer(0, j);
#if 1
        if (is_selfadjoint) {
            constexpr Int CHUNK_SIZE = 6;
            using Vec = VecN_T<Field, CHUNK_SIZE>;
            Int k;
            using VecBlock = VecN_T<Field, BLOCK_SIZE>;
            using Block = Eigen::Matrix<Field, BLOCK_SIZE, CHUNK_SIZE>;
            // Eigen::Stride Gotcha: for single-row matrices, even in column
            // major format, it is the **inner stride** that is used to
            // determine the pointer increment between consecutive entries.
            // However, with 2 or more rows, it is the outer stride, as one
            // would expect.
            // See also: https://gitlab.com/libeigen/eigen/-/issues/416#note_709598886
            //           https://eigen.tuxfamily.org/bz/show_bug.cgi?id=416#c9
            using Stride = std::conditional_t<BLOCK_SIZE == 1, Eigen::InnerStride<>, Eigen::OuterStride<>>;
            using BMap = Eigen::Map<const Block, 0, Stride>;
            for (k = 0; k <= supernode_size - CHUNK_SIZE; k += CHUNK_SIZE) {
              Vec val = Vec::Zero();
              for (Int i = 0; i < subdiagonal.height; i += BLOCK_SIZE) {
                // for (int c = 0; c < BLOCK_SIZE; ++c)
                //     assert(indices[i + c] == indices[i] + c);
                val += BMap(subdiagonal.Pointer(i, k), BLOCK_SIZE, CHUNK_SIZE, Stride(subdiagonal.leading_dim)).adjoint()
                            * Eigen::Map<const VecBlock>(rhs_ptr + indices[i]);
              }
              Eigen::Map<Vec>(srhs_ptr + k) -= val;
            }
            for (; k < supernode_size; ++k) {
              Field val = 0;
              for (Int i = 0; i < subdiagonal.height; i += BLOCK_SIZE) {
                val += Eigen::Map<const VecBlock>(subdiagonal.Pointer(i, k)).dot(Eigen::Map<const VecBlock>(rhs_ptr + indices[i]));
              }
              srhs_ptr[k] -= val;
            }
        }
        else {
            constexpr Int CHUNK_SIZE = 6;
            using Vec = VecN_T<Field, CHUNK_SIZE>;
            Int k;
            for (k = 0; k <= supernode_size - CHUNK_SIZE; k += CHUNK_SIZE) {
              Vec val = rhs_ptr[indices[0]] * Eigen::Map<const Vec, 0, Eigen::InnerStride<>>(subdiagonal.Pointer(0, k), Eigen::InnerStride<>(subdiagonal.leading_dim));
              for (Int i = 1; i < subdiagonal.height; ++i) {
                Vec c = Eigen::Map<const Vec, 0, Eigen::InnerStride<>>(subdiagonal.Pointer(i, k), Eigen::InnerStride<>(subdiagonal.leading_dim));
                val += rhs_ptr[indices[i]] * c;
              }
              Eigen::Map<Vec>(srhs_ptr + k) -= val;
            }
            for (; k < supernode_size; ++k) {
              Field val = 0;
              for (Int i = 0; i < subdiagonal.height; ++i) {
                val += subdiagonal(i, k) * rhs_ptr[indices[i]];
              }
              srhs_ptr[k] -= val;
            }
        }
#else
        for (Int k = 0; k < supernode_size; ++k) {
          const Field *subdiagonal_ptr = subdiagonal.Pointer(0, k);
          Field val = 0;
          for (Int i = 0; i < subdiagonal.height; i += BLOCK_SIZE) {
            const Field *sdp = subdiagonal_ptr + i;
            const Field *rhsp = rhs_ptr + indices[i];
            for (Int c = 0; c < BLOCK_SIZE; ++c)
                val += *(sdp++) * *(rhsp++);
          }
          srhs_ptr[k] -= val;
        }
#endif
      }
      FG_STOP_TIMER(solve_shared_state_.finegrained_timers, supernode, InPlaceBacksubUpdate);
    }
  }

  FG_START_TIMER(solve_shared_state_.finegrained_timers, supernode, SolveDiag);

  // Solve against the diagonal block of this supernode.
  const ConstBlasMatrixView<Field> triangular_right_hand_sides =
      diagonal_factor_->blocks[supernode];
  if (control_.factorization_type == kCholeskyFactorization) {
    if (right_hand_sides_supernode.width > 1) {
        LeftLowerAdjointTriangularSolves(triangular_right_hand_sides, &right_hand_sides_supernode);
    }
    else {
        TriangularSolveLeftLowerAdjoint(triangular_right_hand_sides, right_hand_sides_supernode.Data());
    }
  } else if (control_.factorization_type == kLDLAdjointFactorization) {
    LeftLowerAdjointUnitTriangularSolves(triangular_right_hand_sides,
                                         &right_hand_sides_supernode);
  } else {
    LeftLowerTransposeUnitTriangularSolves(triangular_right_hand_sides,
                                           &right_hand_sides_supernode);
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
void Factorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  BENCHMARK_SCOPED_TIMER_SECTION timer("LowerTransposeTriangularSolve");

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
  // Any pre-order will do
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int s = num_supernodes - 1; s >= 0; --s) {
      LowerTransposeSupernodalTrapezoidalSolve(s, right_hand_sides,
                                               &packed_input_buf);
  }
#endif
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_IMPL_H_
