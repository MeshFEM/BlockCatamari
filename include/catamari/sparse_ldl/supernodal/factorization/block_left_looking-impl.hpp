////////////////////////////////////////////////////////////////////////////////
// block_left_looking-impl.hpp
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Block-accelerated version of Jack Poulson's left-looking supernodal code.
//  This is designed to be compatible with the block right-looking code
//  (for use in subtrees that are not run in parallel).
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/21/2025 12:35:27
*///////////////////////////////////////////////////////////////////////////////
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_BLOCK_LEFT_LOOKING_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_BLOCK_LEFT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template<Int BlockSize, class Field>
void accumulateBlock(const Field *__restrict src, const Int src_leading_dim, Field *__restrict dst, Int dst_leading_dim) {
    for (Int cj = 0; cj < BlockSize; ++cj) {
        for (Int ci = 0; ci < BlockSize; ++ci)
            dst[ci] += src[ci];
        src += src_leading_dim;
        dst += dst_leading_dim;
    }
}

template <class Field>
template <Int BlockSize>
void Factorization<Field>::BlockLeftLookingSupernodeUpdate(
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
    for (Int i = 0; i < supernode_degree; i += BlockSize) {
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
        const ConstBlasMatrixView<Field>& descendant_lower_block = lower_factor_->blocks[descendant];
        const Int descendant_degree = descendant_lower_block.height;
        const Int descendant_size = descendant_lower_block.width;

        const Int descendant_main_rel_row = shared_state->rel_rows[descendant];
        const Int intersect_size = *shared_state->intersect_ptrs[descendant]; // How much of descendant's structure falls within this supernode
        const Int intersect_blocks = intersect_size / BlockSize;
        assert(intersect_blocks * BlockSize == intersect_size && "Intersect size must be a multiple of BlockSize");

        const Int* descendant_structure = lower_factor_->StructureBeg(descendant) + descendant_main_rel_row;

        const ConstBlasMatrixView<Field> descendant_main_matrix = descendant_lower_block.Submatrix(descendant_main_rel_row, 0, intersect_size, descendant_size);

        BlasMatrixView<Field> scaled_transpose;

        const Int descendant_below_main_rel_row = shared_state->rel_rows[descendant] + intersect_size;
        const Int descendant_main_degree        = descendant_degree - descendant_main_rel_row;
        const Int descendant_degree_remaining   = descendant_degree - descendant_below_main_rel_row;

        // Construct mapping of descendant structure to supernode structure.
        const bool inplace_diag_update = intersect_size == supernode_size;
        const bool inplace_subdiag_update = inplace_diag_update && descendant_degree_remaining == supernode_degree;
        if (!inplace_subdiag_update) {
            // Store the relative indices of the diagonal block.
            for (Int i_rel = 0; i_rel < intersect_blocks; ++i_rel) {
                const Int i = descendant_structure[BlockSize * i_rel];
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
            LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix,
                    Real{1}, &diagonal_block);
            CATAMARI_STOP_TIMER(profile.herk);
        } else {
            // Form the diagonal block update out-of-place.
            CATAMARI_START_TIMER(profile.herk);
            LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix, Real{0}, &workspace_matrix);
            CATAMARI_STOP_TIMER(profile.herk);

            // Apply the diagonal block update.
            CATAMARI_START_TIMER(profile.herk_unpack);
            const Field* workspace_col = workspace_matrix.Data();

            for (Int j_rel = 0; j_rel < intersect_blocks; ++j_rel) {
                Field* diag_col = diagonal_block.Pointer(0, rel_ind[j_rel]);
                for (Int i_rel = j_rel; i_rel < intersect_blocks; ++i_rel) {
                    accumulateBlock<BlockSize>(workspace_col + BlockSize * i_rel, workspace_matrix.leading_dim, // src
                                               diag_col + rel_ind[i_rel], diagonal_block.leading_dim);          // dst
                }
                workspace_col += BlockSize * workspace_matrix.leading_dim;
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
            const ConstBlasMatrixView<Field> descendant_below_main_matrix = descendant_lower_block.Submatrix(descendant_below_main_rel_row, 0, descendant_degree_remaining, descendant_size);

            // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
            //                = L(KNext:n, J) * Z(J, K).
            if (inplace_subdiag_update) {
                // Apply the subdiagonal block update in-place.
                CATAMARI_START_TIMER(profile.gemm);
                MatrixMultiplyNormalAdjoint(Field{-1}, descendant_below_main_matrix, descendant_main_matrix, Field{1}, &lower_block);
                CATAMARI_STOP_TIMER(profile.gemm);
            } else {
                // Form the subdiagonal block update out-of-place.
                workspace_matrix.height      = descendant_degree_remaining;
                workspace_matrix.width       = intersect_size;
                workspace_matrix.leading_dim = descendant_degree_remaining;
                CATAMARI_START_TIMER(profile.gemm);
                MatrixMultiplyNormalAdjoint(Field{-1}, descendant_below_main_matrix, descendant_main_matrix, Field{0}, &workspace_matrix);
                CATAMARI_STOP_TIMER(profile.gemm);

                // Store the relative indices on the lower structure.
                const Int descendant_main_block_degree = descendant_main_degree / BlockSize;
                const Int descendant_block_degree_remaining = descendant_degree_remaining / BlockSize;
                for (Int i_rel = intersect_blocks; i_rel < descendant_main_block_degree; ++i_rel) {
                    const Int i = descendant_structure[BlockSize * i_rel];
                    rel_ind[i_rel] = local_index_for_L_row[i];
                }

                // Apply the subdiagonal block update.
                CATAMARI_START_TIMER(profile.gemm_unpack);
                const Field* workspace_ptr = workspace_matrix.Data();
                for (Int j_rel = 0; j_rel < intersect_blocks; ++j_rel) {
                    Field* lower_col = lower_block.Pointer(0, rel_ind[j_rel]);
                    for (Int i_rel = 0; i_rel < descendant_block_degree_remaining; ++i_rel) {
                        accumulateBlock<BlockSize>(workspace_ptr, workspace_matrix.leading_dim, // src
                                                   lower_col + rel_ind[i_rel + intersect_blocks], lower_block.leading_dim); // dst
                        workspace_ptr += BlockSize;
                    }
                    workspace_ptr += (BlockSize - 1) * workspace_matrix.leading_dim; // Note: `workspace_ptr` has iterated through the full workspace matrix's height via the innner loop!
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
template <Int BlockSize>
SparseLDLResult<Field> Factorization<Field>::BlockLeftLooking() { // matrix data is accessed via the ConversionPlan!
    DynamicRegularizationParams<Field> dynamic_reg_params;
    dynamic_reg_params.enabled = control_.dynamic_regularization.enabled;
    if (dynamic_reg_params.enabled) throw std::runtime_error("Dynamic regularization suppport is disabled");
    if (control_.factorization_type != kCholeskyFactorization) throw std::runtime_error("Only Cholesky factorization is currently supported for simplicity");

    typedef ComplexBase<Field> Real;
    CATAMARI_START_TIMER(profile.left_looking);
    const Int num_supernodes = ordering_.supernode_sizes.Size();

    BENCHMARK_SCOPED_TIMER_SECTION timer("BlockLeftLooking<" + std::to_string(BlockSize) + ">");

    CATAMARI_START_TIMER(profile.left_looking_allocate);
    LeftLookingSharedState shared_state;
    shared_state.rel_rows.Resize(num_supernodes);
    shared_state.intersect_ptrs.Resize(num_supernodes);
    shared_state.descendants.Initialize(num_supernodes);

    PrivateState<Field> private_state;
    Int numRows = ordering_.supernode_offsets[num_supernodes];
    private_state.pattern_flags.Resize(numRows);
    private_state.relative_indices.Resize(numRows);
    private_state.workspace_buffer.Resize(left_looking_workspace_size_, Field{0});
    CATAMARI_STOP_TIMER(profile.left_looking_allocate);

    // Note that any postordering of the supernodal elimination forest suffices.
    SparseLDLResult<Field> result;
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
        CATAMARI_START_TIMER(profile.left_looking_update);

        CATAMARI_START_TIMER(profile.initialize_columns);
        // InitializeBlockColumn(supernode, matrix);
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

        BlockLeftLookingSupernodeUpdate<BlockSize>(supernode, &shared_state, &private_state);

        const bool succeeded = LeftLookingSupernodeFinalize(supernode, dynamic_reg_params, &result); // Use regular non-block version (no indexing overhead)
        if (!succeeded) break;
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

#endif  // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_BLOCK_LEFT_LOOKING_IMPL_H_
