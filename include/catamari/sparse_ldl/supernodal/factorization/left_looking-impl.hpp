/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::LeftLookingSupernodeUpdate(
    Int supernode,
    LeftLookingSharedState* shared_state, PrivateState<Field>* private_state) {
  CATAMARI_START_TIMER(profile.left_looking_update);
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int supernode_size = lower_block.width;
  const Int supernode_degree = lower_block.height;
  const Int supernode_offset = ordering_.supernode_offsets[supernode];

  Int* local_index_for_L_row = private_state->pattern_flags.Data();
  Int* rel_ind = private_state->relative_indices.Data();

  // Scatter the pattern of this supernode into pattern_flags.
  // TODO(Jack Poulson): Switch away from pointers to Int members.
  const Int* structure = lower_factor_->StructureBeg(supernode);
  for (Int i = 0; i < supernode_degree; ++i) {
    local_index_for_L_row[structure[i]] = i;
  }

  shared_state->rel_rows[supernode] = 0;
  shared_state->intersect_ptrs[supernode] =
      lower_factor_->IntersectionSizesBeg(supernode);

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  const Int head = shared_state->descendants.heads[supernode];
  for (Int next_descendant = head; next_descendant >= 0;) {
    const Int descendant = next_descendant;
    CATAMARI_ASSERT(descendant < supernode, "Looking into upper triangle");
    const ConstBlasMatrixView<Field>& descendant_lower_block =
        lower_factor_->blocks[descendant];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_size = descendant_lower_block.width;

    const Int descendant_main_rel_row = shared_state->rel_rows[descendant];
    const Int intersect_size = *shared_state->intersect_ptrs[descendant]; // How much of descendant's structure falls within this supernode
    CATAMARI_ASSERT(intersect_size > 0, "Non-positive intersection size.");

    const Int* descendant_structure =
        lower_factor_->StructureBeg(descendant) + descendant_main_rel_row;
    CATAMARI_ASSERT(
        descendant_structure < lower_factor_->StructureEnd(descendant),
        "Relative row exceeded end of structure.");
    CATAMARI_ASSERT(
        supernode_member_to_index_[*descendant_structure] == supernode,
        "First member of structure was not intersecting supernode.");
    CATAMARI_ASSERT(
        supernode_member_to_index_[descendant_structure[intersect_size - 1]] ==
            supernode,
        "Last member of structure was not intersecting supernode.");

    const ConstBlasMatrixView<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         intersect_size, descendant_size);

    BlasMatrixView<Field> scaled_transpose;
    if (control_.factorization_type != kCholeskyFactorization) {
      scaled_transpose.height = descendant_size;
      scaled_transpose.width = intersect_size;
      scaled_transpose.leading_dim = descendant_size;
      scaled_transpose.data = private_state->scaled_transpose_buffer.Data();
      FormScaledTranspose(control_.factorization_type,
                          diagonal_factor_->blocks[descendant].ToConst(),
                          descendant_main_matrix, &scaled_transpose);
    }

    const Int descendant_below_main_rel_row =
        shared_state->rel_rows[descendant] + intersect_size;
    const Int descendant_main_degree =
        descendant_degree - descendant_main_rel_row;
    const Int descendant_degree_remaining =
        descendant_degree - descendant_below_main_rel_row;

    // Construct mapping of descendant structure to supernode structure.
    const bool inplace_diag_update = intersect_size == supernode_size;
    const bool inplace_subdiag_update = inplace_diag_update && descendant_degree_remaining == supernode_degree;
    if (!inplace_subdiag_update) {
      // Store the relative indices of the diagonal block.
      for (Int i_rel = 0; i_rel < intersect_size; ++i_rel) {
        const Int i = descendant_structure[i_rel];
        CATAMARI_ASSERT(
            i >= supernode_offset && i < supernode_offset + supernode_size,
            "Invalid relative diagonal block index.");
        rel_ind[i_rel] = i - supernode_offset; // Index of local decendent row within this supernode's diagonal block.
      }
    }

    // Update the diagonal block.
    BlasMatrixView<Field> workspace_matrix;
    workspace_matrix.height = intersect_size;
    workspace_matrix.width = intersect_size;
    workspace_matrix.leading_dim = intersect_size;
    workspace_matrix.data = private_state->workspace_buffer.Data();
    if (inplace_diag_update) {
      // Apply the diagonal block update in-place.
      CATAMARI_START_TIMER(profile.herk);
      if (control_.factorization_type == kCholeskyFactorization) {
        LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix,
                                         Real{1}, &diagonal_block);
      } else {
        MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                        scaled_transpose.ToConst(), Field{1},
                                        &diagonal_block);
      }
      CATAMARI_STOP_TIMER(profile.herk);
    } else {
      // Form the diagonal block update out-of-place.
      CATAMARI_START_TIMER(profile.herk);
      if (control_.factorization_type == kCholeskyFactorization) {
        LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix,
                                         Real{0}, &workspace_matrix);
      } else {
        MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                        scaled_transpose.ToConst(), Field{0},
                                        &workspace_matrix);
      }
      CATAMARI_STOP_TIMER(profile.herk);

      // Apply the diagonal block update.
      CATAMARI_START_TIMER(profile.herk_unpack);
      for (Int j_rel = 0; j_rel < intersect_size; ++j_rel) {
        const Int j = rel_ind[j_rel];
        Field* diag_col = diagonal_block.Pointer(0, j);
        const Field* workspace_col = workspace_matrix.Pointer(0, j_rel);
        for (Int i_rel = j_rel; i_rel < intersect_size; ++i_rel) {
          const Int i = rel_ind[i_rel];
          diag_col[i] += workspace_col[i_rel];
        }
      }
      CATAMARI_STOP_TIMER(profile.herk_unpack);
    }
#ifdef CATAMARI_ENABLE_TIMERS
    profile.herk_gflops +=
        intersect_size * (intersect_size + 1.) * descendant_size / 1.e9;
#endif  // ifdefCATAMARI_ENABLE_TIMERS

    shared_state->intersect_ptrs[descendant]++;
    shared_state->rel_rows[descendant] = descendant_below_main_rel_row;

    next_descendant = shared_state->descendants.lists[descendant];
    if (descendant_degree_remaining > 0) {
      const ConstBlasMatrixView<Field> descendant_below_main_matrix =
          descendant_lower_block.Submatrix(descendant_below_main_rel_row, 0,
                                           descendant_degree_remaining,
                                           descendant_size);

      // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
      //                = L(KNext:n, J) * Z(J, K).
      if (inplace_subdiag_update) {
        // Apply the subdiagonal block update in-place.
        CATAMARI_START_TIMER(profile.gemm);
        if (control_.factorization_type == kCholeskyFactorization) {
          MatrixMultiplyNormalAdjoint(Field{-1}, descendant_below_main_matrix,
                                      descendant_main_matrix, Field{1},
                                      &lower_block);
        } else {
          MatrixMultiplyNormalNormal(Field{-1}, descendant_below_main_matrix,
                                     scaled_transpose.ToConst(), Field{1},
                                     &lower_block);
        }
        CATAMARI_STOP_TIMER(profile.gemm);
      } else {
        // Form the subdiagonal block update out-of-place.
        workspace_matrix.height = descendant_degree_remaining;
        workspace_matrix.width = intersect_size;
        workspace_matrix.leading_dim = descendant_degree_remaining;
        CATAMARI_START_TIMER(profile.gemm);
        if (control_.factorization_type == kCholeskyFactorization) {
          MatrixMultiplyNormalAdjoint(Field{-1}, descendant_below_main_matrix,
                                      descendant_main_matrix, Field{0},
                                      &workspace_matrix);
        } else {
          MatrixMultiplyNormalNormal(Field{-1}, descendant_below_main_matrix,
                                     scaled_transpose.ToConst(), Field{0},
                                     &workspace_matrix);
        }
        CATAMARI_STOP_TIMER(profile.gemm);

        // Store the relative indices on the lower structure.
        for (Int i_rel = intersect_size; i_rel < descendant_main_degree;
             ++i_rel) {
          const Int i = descendant_structure[i_rel];
          rel_ind[i_rel] = local_index_for_L_row[i];
          CATAMARI_ASSERT(
              rel_ind[i_rel] >= 0 && rel_ind[i_rel] < supernode_degree,
              "Invalid subdiagonal relative index.");
        }

        // Apply the subdiagonal block update.
        CATAMARI_START_TIMER(profile.gemm_unpack);
        for (Int j_rel = 0; j_rel < intersect_size; ++j_rel) {
          const Int j = rel_ind[j_rel];
          CATAMARI_ASSERT(j >= 0 && j < supernode_size,
                          "Invalid unpacked column index.");
          CATAMARI_ASSERT(j + ordering_.supernode_offsets[supernode] ==
                              descendant_structure[j_rel],
                          "Mismatched unpacked column structure.");

          Field* lower_col = lower_block.Pointer(0, j);
          const Field* workspace_col = workspace_matrix.Pointer(0, j_rel);
          for (Int i_rel = 0; i_rel < descendant_degree_remaining; ++i_rel) {
            const Int i = rel_ind[i_rel + intersect_size];
            CATAMARI_ASSERT(i >= 0 && i < supernode_degree,
                            "Invalid row index.");
            CATAMARI_ASSERT(
                structure[i] == descendant_structure[i_rel + intersect_size],
                "Mismatched row structure.");
            lower_col[i] += workspace_col[i_rel];
          }
        }
        CATAMARI_STOP_TIMER(profile.gemm_unpack);
      }
#ifdef CATAMARI_ENABLE_TIMERS
      profile.gemm_gflops += 2. * intersect_size * descendant_size *
                             descendant_degree_remaining / 1.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

      // Insert the descendant supernode into the list of its next ancestor.
      // NOTE: We would need a lock for this in a multithreaded setting.
      const Int next_ancestor =
          supernode_member_to_index_[descendant_structure[intersect_size]];
      shared_state->descendants.Insert(next_ancestor, descendant);
    }
  }

  if (supernode_degree > 0) {
    // Insert the supernode into the list of its parent.
    // NOTE: We would need a lock for this in a multithreaded setting.
    const Int parent = supernode_member_to_index_[structure[0]];
    shared_state->descendants.Insert(parent, supernode);
  }

  // Clear the descendant list for this node.
  shared_state->descendants.heads[supernode] = -1;
  CATAMARI_STOP_TIMER(profile.left_looking_update);
}

template <class Field>
bool Factorization<Field>::LeftLookingSupernodeFinalize(
    Int supernode, const DynamicRegularizationParams<Field>& dynamic_reg_params,
    SparseLDLResult<Field>* result) {
  CATAMARI_START_TIMER(profile.left_looking_finalize);
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  CATAMARI_START_TIMER(profile.cholesky);
  Int num_supernode_pivots;
  if (control_.supernodal_pivoting) {
    // TODO(Jack Poulson): Add support for dynamic regularization into the
    // dynamically pivoted version.
    BlasMatrixView<Int> permutation = SupernodePermutation(supernode);
    num_supernode_pivots = PivotedFactorDiagonalBlock(
        control_.block_size, control_.factorization_type, &diagonal_block,
        &permutation);
  } else {
    num_supernode_pivots = FactorDiagonalBlock(
        control_.block_size, control_.factorization_type, dynamic_reg_params,
        &diagonal_block, &result->dynamic_regularization);
  }
  CATAMARI_STOP_TIMER(profile.cholesky);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < supernode_size) {
    CATAMARI_STOP_TIMER(profile.left_looking_finalize);
    return false;
  }
#ifdef CATAMARI_ENABLE_TIMERS
  profile.cholesky_gflops +=
      supernode_size *
      (supernode_size * (supernode_size / 3.) + supernode_size / 2. - 5. / 6.) /
      1.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);
  if (!degree) {
    CATAMARI_STOP_TIMER(profile.left_looking_finalize);
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  CATAMARI_START_TIMER(profile.trsm);
  if (control_.supernodal_pivoting) {
    // Solve against P^T from the right, which is the same as applying P
    // from the right, which is the same as applying P^T to each row.
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    InversePermuteColumns(permutation, &lower_block);
  }
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            diagonal_block.ToConst(), &lower_block);
  CATAMARI_STOP_TIMER(profile.trsm);
#ifdef CATAMARI_ENABLE_TIMERS
  profile.trsm_gflops += std::pow(1. * supernode_size, 2.) * degree / 1.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  CATAMARI_STOP_TIMER(profile.left_looking_finalize);
  return true;
}

// We no longer support OpenMP in the left-looking factorization.
template <class Field>
SparseLDLResult<Field> Factorization<Field>::LeftLooking(
    const CoordinateMatrix<Field>& /* matrix */) { // matrix data is accessed via the ConversionPlan!
  typedef ComplexBase<Field> Real;
  CATAMARI_START_TIMER(profile.left_looking);
  const Int num_supernodes = ordering_.supernode_sizes.Size();

  BENCHMARK_SCOPED_TIMER_SECTION timer("LeftLooking");

  CATAMARI_START_TIMER(profile.left_looking_allocate);
  LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);
  shared_state.descendants.Initialize(num_supernodes);

  PrivateState<Field> private_state;
  Int numRows = ordering_.supernode_offsets[num_supernodes];
  private_state.pattern_flags.Resize(numRows);
  private_state.relative_indices.Resize(numRows);
  if (control_.factorization_type != kCholeskyFactorization) {
    private_state.scaled_transpose_buffer.Resize(
        left_looking_scaled_transpose_size_, Field{0});
  }
  private_state.workspace_buffer.Resize(left_looking_workspace_size_, Field{0});
  CATAMARI_STOP_TIMER(profile.left_looking_allocate);

  // Set up the base value of the dynamic regularization parameters, and
  // update the offset for each supernode.
  static const Real kEpsilon = std::numeric_limits<Real>::epsilon();
  DynamicRegularizationParams<Field> dynamic_reg_params;
  dynamic_reg_params.enabled = control_.dynamic_regularization.enabled;
  if (dynamic_reg_params.enabled) {
      throw std::runtime_error("Dynamic regularization suppport is disabled");
      dynamic_reg_params.positive_threshold = std::pow(
          kEpsilon, control_.dynamic_regularization.positive_threshold_exponent);
      dynamic_reg_params.negative_threshold = std::pow(
          kEpsilon, control_.dynamic_regularization.negative_threshold_exponent);
      if (control_.dynamic_regularization.relative) {
        const Real matrix_max_norm = 0; // MaxNorm(matrix); // TODO: get from ConversionPlan
        dynamic_reg_params.positive_threshold *= matrix_max_norm;
        dynamic_reg_params.negative_threshold *= matrix_max_norm;
      }
      dynamic_reg_params.signatures = &control_.dynamic_regularization.signatures;
      dynamic_reg_params.inverse_permutation = ordering_.inverse_permutation.Empty()
                                                   ? nullptr
                                                   : &ordering_.inverse_permutation;
  }

  // Note that any postordering of the supernodal elimination forest suffices.
  SparseLDLResult<Field> result;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    CATAMARI_START_TIMER(profile.left_looking_update);

    CATAMARI_START_TIMER(profile.initialize_columns);
    const Int sno = ordering_.supernode_offsets[supernode];
    const Int supernode_size = ordering_.supernode_sizes[supernode];
    BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
#if 0
    for (Int j = 0, cj = 0; j < supernode_size; ++j)
        InitializeFactorColumn(sno + j, j, diagonal_block);
#else
    Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, 1>>(
        diagonal_block.data, diagonal_block.LeadingDimension() * diagonal_block.Width()).setZero();
    m_inputData.injectEntries(sno, sno + supernode_size, factor_values_.Data());
    if (m_inputData.sigma != 0) {
        for (Int j = 0, cj = 0; j < supernode_size; ++j)
            diagonal_block(j, j) += m_inputData.sigma;
    }
#endif

    CATAMARI_STOP_TIMER(profile.initialize_columns);

    LeftLookingSupernodeUpdate(supernode, &shared_state, &private_state);

    dynamic_reg_params.offset = ordering_.supernode_offsets[supernode];
    const bool succeeded =
        LeftLookingSupernodeFinalize(supernode, dynamic_reg_params, &result);
    if (!succeeded) {
      CATAMARI_STOP_TIMER(profile.left_looking);
      return result;
    }
  }

#ifdef CATAMARI_DEBUG
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    CATAMARI_ASSERT(shared_state.rel_rows[supernode] ==
                        lower_factor_->blocks[supernode].height,
                    "Did not properly handle relative row offsets.");
  }
#endif  // ifdef CATAMARI_DEBUG
  CATAMARI_STOP_TIMER(profile.left_looking);
  return result;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_IMPL_H_
