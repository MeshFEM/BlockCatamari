////////////////////////////////////////////////////////////////////////////////
// block_left_looking-impl.hpp
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Block-accelerated version of Jack Poulson's right-looking multifrontal code.
//  This version also uses TBB instead of OpenMP for parallelism.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/21/2025 12:35:27
*///////////////////////////////////////////////////////////////////////////////
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_BLOCK_RIGHT_LOOKING_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_BLOCK_RIGHT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

#include "../../../../../../../src/lib/MeshFEM/GlobalBenchmark.hh"
#include "../../../../../../../src/lib/MeshFEM/ParallelVectorOps.hh"
#include "SchurComplementStorage.hpp"
#include <cassert>
#include "../catamari_config.hh"

#define REUSE_CHOLESKY_FLOWGRAPHS 0

namespace catamari {
namespace supernodal_ldl {

template <class Field>
template <Int BlockSize>
bool Factorization<Field>::BlockRightLookingSupernodeFinalize(
                Int supernode, const DynamicRegularizationParams<Field>& dynamic_reg_params,
                RightLookingSharedState<Field>* shared_state,
                SparseLDLResult<Field>* result)
{
    typedef ComplexBase<Field> Real;
    BlasMatrixView<Field> diagonal_block = diagonal_factor_->blocks[supernode];
    BlasMatrixView<Field> lower_block = lower_factor_->blocks[supernode];
    const Int degree = lower_block.height;
    const Int supernode_size = lower_block.width;
    const bool has_children = ordering_.assembly_forest.child_offsets[supernode + 1] > ordering_.assembly_forest.child_offsets[supernode];

    Int num_supernode_pivots;
    const bool single_thread = shared_state->num_tbb_threads < 2;
    if (control_.supernodal_pivoting) {
        BlasMatrixView<Int> permutation = SupernodePermutation(supernode);
        num_supernode_pivots = PivotedFactorDiagonalBlock(
                control_.block_size, control_.factorization_type, &diagonal_block,
                &permutation);
        result->num_successful_pivots += num_supernode_pivots;
    } else {
        FG_START_TIMER(shared_state->finegrained_timers, supernode, FactorDiag);
#if 1
        if ((diagonal_block.height > 3 * control_.factor_tile_size) && !single_thread) {
#if REUSE_CHOLESKY_FLOWGRAPHS
            auto &fg = shared_state->cholesky_flowgraphs[supernode];
            if (!fg) {
                fg = std::make_unique<CholeskyFlowgraph<Field>>(*(shared_state->tbb_ctx), diagonal_block, control_.block_size, control_.factor_tile_size);
                fg->failureCallback = [shared_state]() { shared_state->setFailed(); };
            }
            num_supernode_pivots = fg->run(diagonal_block);
#else
            CholeskyFlowgraph<Field> fg(*(shared_state->tbb_ctx), diagonal_block, control_.block_size, control_.factor_tile_size);
            fg.failureCallback = [shared_state]() { shared_state->setFailed(); };
            num_supernode_pivots = fg.run(diagonal_block);
#endif
        } else {
            num_supernode_pivots = LowerCholeskyFactorizationDynamicBLASDispatch(control_.block_size, &diagonal_block);
        }
#else
        num_supernode_pivots = FactorDiagonalBlock(
                control_.block_size,
                control_.factorization_type, dynamic_reg_params, &diagonal_block,
                &result->dynamic_regularization);
#endif
        FG_STOP_TIMER(shared_state->finegrained_timers, supernode, FactorDiag);
        result->num_successful_pivots += num_supernode_pivots;
    }
    if (num_supernode_pivots < supernode_size) {
        if (shared_state->tbb_ctx) shared_state->tbb_ctx->cancel_group_execution();
        return false;
    }

    IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);

    if (!degree)
        return true; // We can early exit.

    if (shared_state->hasFailed()) return false; // Stop immediately if another thread encountered a failure!

    CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
    if (control_.supernodal_pivoting) {
        // Solve against P^T from the right, which is the same as applying P
        // from the right, which is the same as applying P^T to each row.
        const ConstBlasMatrixView<Int> permutation = SupernodePermutation(supernode);
        InversePermuteColumns(permutation, &lower_block);
    }
    // TODO: implement entire `Finalize` routine as one big flowgraph that is very similar to `CholeskyFlowgraph`;
    // just includes operates also on the lower_block and schur_complement parts of the frontal matrix.

    FG_START_TIMER(shared_state->finegrained_timers, supernode, SolveDiag);
    Int tile_size = control_.factor_tile_size;
#if 1
    if ((lower_block.height > 2 * tile_size) && !single_thread) {
        // First determine the number of tiles based on the target tile size.
        Int num_tiles = (lower_block.height + tile_size - 1) / tile_size;
        // Then produce as even-sized tiles as possible.
        Int trsm_tile_size = (lower_block.height + num_tiles - 1) / num_tiles;
        tbb::parallel_for(tbb::blocked_range<Int>(0, num_tiles, 1), [&lower_block, &diagonal_block, trsm_tile_size](const tbb::blocked_range<Int> &r) {
            for (Int i_tile = r.begin(); i_tile < r.end(); ++i_tile) {
                Int i = i_tile * trsm_tile_size;
                const Int tsize = std::min(lower_block.height - i, trsm_tile_size);
                BlasMatrixView<Field> tile = lower_block.Submatrix(i, 0, tsize, lower_block.width);
                RightLowerAdjointTriangularSolves(diagonal_block.ToConst(), &tile);
            }
        }, *(shared_state->tbb_ctx));
    }
    else
        RightLowerAdjointTriangularSolves(diagonal_block.ToConst(), &lower_block);
#else
    SolveAgainstDiagonalBlock(control_.factorization_type, diagonal_block.ToConst(), &lower_block);
#endif
    FG_STOP_TIMER(shared_state->finegrained_timers, supernode, SolveDiag);

    if (shared_state->hasFailed()) return false; // Stop immediately if another thread encountered a failure!

    FG_START_TIMER(shared_state->finegrained_timers, supernode, OuterProduct);
    BlasMatrixView<Field>& schur_complement = shared_state->schur_complements[supernode];

#if 1
    if ((schur_complement.height > 1.5 * tile_size) && !single_thread)
        TBBLowerNormalHermitianOuterProduct(*(shared_state->tbb_ctx), tile_size, Real{-1}, lower_block.ToConst(), has_children ? Real{1} : Real{0}, &schur_complement);
    else
        LowerNormalHermitianOuterProduct(Real{-1}, lower_block.ToConst(), has_children ? Real{1} : Real{0}, &schur_complement);
#else
    LowerNormalHermitianOuterProduct( Real{-1}, lower_block.ToConst(), has_children ? Real{1} : Real{0}, &schur_complement);
#endif

    FG_STOP_TIMER(shared_state->finegrained_timers, supernode, OuterProduct);

    return true;
}

template <Int BlockSize, class Field>
void BlockMergeChildSchurComplement(Int supernode, Int child,
                                    const SymmetricOrdering& ordering,
                                    const LowerFactor<Field> *lower_factor,
                                    const BlasMatrixView<Field> &child_schur_complement,
                                    BlasMatrixView<Field> lower_block,
                                    BlasMatrixView<Field> diagonal_block,
                                    BlasMatrixView<Field> schur_complement,
                                    Factorization<Field> &ldl, RightLookingSharedState<Field> &shared_state,
                                    bool first_merge) {
    const Int child_degree = child_schur_complement.height;
    const Int sno = ordering.supernode_offsets[supernode];
    assert(child_degree == lower_factor->blocks[child].height && "Incorrectly sized child schur complement. Perhaps child was not processed?");

    // Number of child rows/cols that map to the parent's diagonal block.
    const Int num_child_diag_indices = ordering.assembly_forest.num_child_diag_indices[child];

    // Locations of child's rows/cols relative to the parent front's upper-left corner
    const Int *child_rel_indices = ordering.assembly_forest.child_rel_indices.Data() + ordering.assembly_forest.child_rel_indices_offsets[child];

    const Int supernode_size = ordering.supernode_sizes[supernode];

    if (first_merge) {
        FG_START_TIMER(shared_state.finegrained_timers, supernode, InitializeColumns); // This is not entirely accurate since it includes part of the first child merge time :(

        // Initialize each of the supernode's columns of the factor
        // and merge in the first child's Schur complement.
        {
            Int back_j = 0;
            for (Int cj = 0; cj < num_child_diag_indices; cj += BlockSize) {
                Int dst_j = child_rel_indices[cj]; // Parent block column into which the child block column is merging
                Int jnext = dst_j + BlockSize;
                ldl.template BlockCPlanInitializeFactorColumns<BlockSize>(sno, back_j, jnext, diagonal_block); // Initialize up to the right edge of the destination block column

                const Field* child_column = child_schur_complement.Pointer(0, cj);
                Field *     factor_column = diagonal_block.Pointer(0, dst_j);
                for (Int i = cj; i < child_degree; i += BlockSize) {
                    accumulateBlock<BlockSize>(child_column  + i, child_schur_complement.LeadingDimension(), // src
                                               factor_column + child_rel_indices[i], lower_block.LeadingDimension()); // dst
                }
                back_j = jnext; // Advance to the next uninitialized parent column block.
            }
            if (back_j < supernode_size) ldl.template BlockCPlanInitializeFactorColumns<BlockSize>(sno, back_j, supernode_size, diagonal_block); // Initialize remaining columns not intersected by the child
        }
        FG_STOP_TIMER(shared_state.finegrained_timers, supernode, InitializeColumns);

        FG_START_TIMER(shared_state.finegrained_timers, supernode, MergeSchur);
        FillZerosLowerTriangular(schur_complement.data, schur_complement.width, schur_complement.height);
        for (Int j = num_child_diag_indices; j < child_degree; j += BlockSize) {
            Int dst_j = child_rel_indices[j] - supernode_size; // Parent block column *within the schur complement* into which the child block column is merging

            const Field* child_column = child_schur_complement.Pointer(0, j);
            // Get pointer to the (conceptual) full parent front column, of which schur_complement is the bottom part.
            // Note: parent front's upper-left corner is (-supernode_size, -supernode_size) relative to this block...
            Field* schur_column = schur_complement.Pointer(-supernode_size, dst_j);
            for (Int i = j; i < child_degree; i += BlockSize) {
                accumulateBlock<BlockSize>(child_column + i,              child_schur_complement.LeadingDimension(),  // src
                                           schur_column + child_rel_indices[i], schur_complement.LeadingDimension()); // dst
            }
        }
        FG_STOP_TIMER(shared_state.finegrained_timers, supernode, MergeSchur);
    }
    else {
        FG_START_TIMER(shared_state.finegrained_timers, supernode, MergeSchur);
        // Add the child Schur complement into this supernode's front.
        for (Int j = 0; j < num_child_diag_indices; j += BlockSize) {
            const Field* child_column = child_schur_complement.Pointer(0, j);
            Field* factor_column = diagonal_block.Pointer(0, child_rel_indices[j]);
            for (Int i = j; i < child_degree; i += BlockSize) {
                accumulateBlock<BlockSize>(child_column + i, child_schur_complement.LeadingDimension(), // src
                                           factor_column + child_rel_indices[i], diagonal_block.LeadingDimension()); // dst
            }
        }

        // Contribute into the bottom-right block of the front.
        for (Int j = num_child_diag_indices; j < child_degree; j += BlockSize) {
            const Field* child_column = child_schur_complement.Pointer(0, j);
            Field* schur_column = schur_complement.Pointer(-supernode_size, child_rel_indices[j] - supernode_size);
            for (Int i = j; i < child_degree; i += BlockSize) {
                accumulateBlock<BlockSize>(child_column + i,              child_schur_complement.LeadingDimension(),  // src
                                           schur_column + child_rel_indices[i], schur_complement.LeadingDimension()); // dst
            }
        }
        FG_STOP_TIMER(shared_state.finegrained_timers, supernode, MergeSchur);
    }
}

template <Int BlockSize, class Field>
void BlockMergeChildSchurComplements(Int supernode, Factorization<Field> &ldl,
                                const Buffer<BlasMatrixView<Field>> &schur_complements) {
    const auto &o = ldl.ordering_;
    const auto &af = o.assembly_forest;
    const Int child_beg = af.child_offsets[supernode];
    const Int num_children = af.child_offsets[supernode + 1] - child_beg;
    const Int sno = o.supernode_offsets[supernode];

    // Output destination buffers
    BlasMatrixView<Field> lower_block      = ldl.lower_factor_->blocks[supernode];
    BlasMatrixView<Field> diagonal_block   = ldl.diagonal_factor_->blocks[supernode];
    BlasMatrixView<Field> schur_complement = schur_complements[supernode];

    const Int supernode_size = o.supernode_sizes[supernode];
    std::vector<Int> child_j(num_children); // pointer into the child columns
    const Int factor_height = diagonal_block.Height() + lower_block.Height();

    for (Int j = 0; j < supernode_size; j += BlockSize) {
        ldl.template BlockCPlanInitializeFactorBlockColumn<BlockSize>(sno + j, j, diagonal_block);

        Field* factor_column = diagonal_block.Pointer(0, j);
        for (Int ci = 0; ci < num_children; ++ci) {
            Int cj = child_j[ci];

            const Int child = af.children[child_beg + ci];
            const Int num_child_diag_indices = af.num_child_diag_indices[child];
            const Int child_degree = af.child_rel_indices_offsets[child + 1] - af.child_rel_indices_offsets[child];
            const Int *child_rel_indices = af.child_rel_indices.Data() + af.child_rel_indices_offsets[child];

            if (cj >= child_degree || child_rel_indices[cj] != j) continue;

            const BlasMatrixView<Field> &child_schur_complement = schur_complements[child];
            const Field* child_column = child_schur_complement.Pointer(0, cj);

            for (Int i = cj; i < child_degree; i += BlockSize) {
                accumulateBlock<BlockSize>(child_column + i, child_schur_complement.LeadingDimension(), // src
                                           factor_column + child_rel_indices[i], lower_block.LeadingDimension()); // dst
            }

            child_j[ci] = cj + BlockSize;
        }
    }

    const Int sc_size = schur_complement.width;
    for (Int j = 0; j < sc_size; j += BlockSize) {
        FillZerosLowerTriangularMiddleCols(schur_complement.data, j, j + BlockSize, schur_complement.height);
        Int front_j = j + supernode_size;
        Field *schur_column = schur_complement.Pointer(-supernode_size, j);

        for (Int ci = 0; ci < num_children; ++ci) {
            Int cj = child_j[ci];

            const Int child = af.children[child_beg + ci];
            const Int child_degree = af.child_rel_indices_offsets[child + 1] - af.child_rel_indices_offsets[child];
            const Int *child_rel_indices = af.child_rel_indices.Data() + af.child_rel_indices_offsets[child];

            if (cj >= child_degree || child_rel_indices[cj] != front_j) continue;

            const BlasMatrixView<Field> &child_schur_complement = schur_complements[child];

            const Field* child_column = child_schur_complement.Pointer(0, cj);
            for (Int i = cj; i < child_degree; i += BlockSize) {
                accumulateBlock<BlockSize>(child_column + i,              child_schur_complement.LeadingDimension(), // src
                                           schur_column + child_rel_indices[i], schur_complement.LeadingDimension()); // dst
            }

            child_j[ci] = cj + BlockSize;
        }
    }
}

template <class Field>
template <Int BlockSize>
bool Factorization<Field>::BlockRightLookingSubtree(
        Int supernode,
        const DynamicRegularizationParams<Field>& dynamic_reg_params,
        const Buffer<double>& work_estimates, double min_parallel_work,
        RightLookingSharedState<Field>* shared_state,
        SparseLDLResult<Field>* result,
        SchurComplementStorage<Field> *subtreeStorage) {

#if CATAMARI_FINEGRAINED_TIMERS
    shared_state->finegrained_timers.assigned_thread[supernode] = tbb::this_task_arena::current_thread_index();
#endif

    const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
    const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
    const Int num_children = child_end - child_beg;

    const double work_estimate = work_estimates[supernode];
    const bool parallel = (work_estimate >= min_parallel_work) && (num_children > 1);

    auto process_child = [&, supernode, min_parallel_work, shared_state](Int child, SparseLDLResult<Field> *resultContrib, SchurComplementStorage<Field> *stack) {
        const Int child_offset = ordering_.supernode_offsets[child];
        DynamicRegularizationParams<Field> subparams = dynamic_reg_params;
        subparams.offset = child_offset;
        if (shared_state->hasFailed()) return; // Stop immediately if another thread encountered a failure!
        bool success = BlockRightLookingSubtree<BlockSize>(
                child, subparams, work_estimates, min_parallel_work,
                shared_state, resultContrib, stack);
        if (!success) {
            shared_state->setFailed();
            if (shared_state->tbb_ctx) shared_state->tbb_ctx->cancel_group_execution();
        }
    };

    // Allocate this supernode's Schur complement either in an existing subtree
    // storage stack, or in a standalone `SchurComplementStorage` object.
    auto allocate_schur_complement = [&]() {
        const Int degree = lower_factor_->blocks[supernode].height;
        BlasMatrixView<Field> &sc = shared_state->schur_complements[supernode];
        if (subtreeStorage == nullptr)
            sc = shared_state->schur_complement_storage[supernode].allocateSingleMatrixForDegree(degree);
        else
            sc = subtreeStorage->push(degree);
    };

    if (!parallel) {
        // Output destination buffers
        BlasMatrixView<Field> lower_block    = lower_factor_->blocks[supernode];
        BlasMatrixView<Field> diagonal_block = diagonal_factor_->blocks[supernode];
        if (shared_state->hasFailed()) return false; // Stop immediately if another thread encountered a failure!

        // Construct a stack for holding the child schur complements of the serial subtree rooted at `supernode` (if it doesn't exist already)
        const bool serialSubtree = (work_estimate < min_parallel_work); // make sure we're actually in a serial subtree rather than simply having a single child...
        if (serialSubtree && (subtreeStorage == nullptr)) { // Root of the serial subtree.
            // FG_START_TIMER(shared_state->finegrained_timers, supernode, Allocation); // Allocation time is apparently pretty negligible, especially if we do multiple numeric factorizations in a row.
            subtreeStorage = &(shared_state->schur_complement_storage[supernode]);
            subtreeStorage->reallocate(subtreeStorage->getStoragedNeeded(supernode, ordering_.assembly_forest, *lower_factor_));
            // FG_STOP_TIMER(shared_state->finegrained_timers, supernode, Allocation);
        }

        if (shared_state->hasFailed()) return false; // Stop immediately if another thread encountered a failure!
        if (num_children == 0)
            BlockCPlanInitializeFactorSupernodeColumns<BlockSize>(ordering_.supernode_offsets[supernode], lower_block.width, diagonal_block);

        allocate_schur_complement(); // TODO: move this after `process_child` if/when we implement an expand-in-place strategy
        for (Int child_index = 0; child_index < num_children; ++child_index) {
            const Int child = ordering_.assembly_forest.children[child_beg + child_index]; // sorted_children[child_index];

            SparseLDLResult<Field> resultContrib;

            // FG_START_TIMER(shared_state->finegrained_timers, supernode, Recurse);
            process_child(child, &resultContrib, subtreeStorage);
            // FG_STOP_TIMER(shared_state->finegrained_timers, supernode, Recurse);

            MergeContribution(resultContrib, result);
            if (dynamic_reg_params.enabled) assert(false); /* MergeDynamicRegularizations(result_contributions, result); */

            // Stop immediately if this child failed to finalize (or if another thread encountered a failure)
            if (shared_state->hasFailed()) return false;

            auto &sc_child = shared_state->schur_complements[child];
            BlockMergeChildSchurComplement<BlockSize>(supernode, child, ordering_,
                    lower_factor_.get(), sc_child,
                    lower_block, diagonal_block, shared_state->schur_complements[supernode], *this, *shared_state, /* first_merge = */ child_index == 0);

            // Pop the child Schur complement from the stack.
            // Note: this will not deallocate the stack itself; that is done by the
            // parent of the serial subtree root (if run in parallel), or the
            // top-level BlockRightLooking loop.
            if (serialSubtree) subtreeStorage->free(sc_child);
            else shared_state->schur_complement_storage[child].deallocate();
        }
    }
    else {
        // FG_START_TIMER(shared_state->finegrained_timers, supernode, Recurse);
        Buffer<SparseLDLResult<Field>> result_contributions(num_children);

        tbb::task_group tg(*(shared_state->tbb_ctx));
        for (Int child_index = 0; child_index < num_children; ++child_index) {
            const Int child = ordering_.assembly_forest.children[child_beg + child_index]; // sorted_children[child_index];
            tg.run([&process_child, &result_contributions, child, child_index, shared_state]() {
                    process_child(child, &result_contributions[child_index], nullptr);
            });
        }
        // process_child(ordering_.assembly_forest.children[child_end - 1], &result_contributions[num_children - 1], nullptr);
        auto status = tg.wait();

        // FG_STOP_TIMER(shared_state->finegrained_timers, supernode, Recurse);

        if (status != tbb::task_group_status::complete)
            shared_state->setFailed();

        if (!shared_state->hasFailed()) {
            // FG_START_TIMER(shared_state->finegrained_timers, supernode, Allocation);
            allocate_schur_complement();
            // FG_STOP_TIMER(shared_state->finegrained_timers, supernode, Allocation);

#if 0
            BlasMatrixView<Field> lower_block      = lower_factor_->blocks[supernode];
            BlasMatrixView<Field> diagonal_block   = diagonal_factor_->blocks[supernode];
            for (Int child_index = 0; child_index < num_children; ++child_index) {
                const Int child = ordering_.assembly_forest.children[child_beg + child_index]; // sorted_children[child_index];
                BlockMergeChildSchurComplement<BlockSize>(supernode, child, ordering_,
                        lower_factor_.get(), shared_state->schur_complements[child],
                        lower_block, diagonal_block, shared_state->schur_complements[supernode], *this, *shared_state, /* first_merge = */ child_index == 0);
            }
#else
            FG_START_TIMER(shared_state->finegrained_timers, supernode, MergeSchurInPara);
            BlockMergeChildSchurComplements<BlockSize>(supernode, *this, shared_state->schur_complements);
            FG_STOP_TIMER(shared_state->finegrained_timers, supernode, MergeSchurInPara);
#endif
        }

        FG_START_TIMER(shared_state->finegrained_timers, supernode, Deallocation);
        // Clear out all storage used by descendants' fronts.
        for (Int child_index = 0; child_index < num_children; ++child_index) {
            const Int child = ordering_.assembly_forest.children[child_beg + child_index];
            auto &sc = shared_state->schur_complements[child];
            sc.width = sc.height = 0;
            sc.data = nullptr;
            shared_state->schur_complement_storage[child].deallocate();
        }
        FG_STOP_TIMER(shared_state->finegrained_timers, supernode, Deallocation);

        for (Int child_index = 0; child_index < num_children; ++child_index)
            MergeContribution(result_contributions[child_index], result);
        if (dynamic_reg_params.enabled) MergeDynamicRegularizations(result_contributions, result);
    }

    if (shared_state->hasFailed()) return false;

    return BlockRightLookingSupernodeFinalize<BlockSize>(supernode, dynamic_reg_params, shared_state, result);
}

template <class Field>
template <Int BlockSize>
SparseLDLResult<Field> Factorization<Field>::BlockRightLooking() {
    const Int num_supernodes = ordering_.supernode_sizes.Size();
    const Int num_roots = ordering_.assembly_forest.roots.Size();

    DynamicRegularizationParams<Field> dynamic_reg_params;
    dynamic_reg_params.enabled = control_.dynamic_regularization.enabled;
    if (dynamic_reg_params.enabled) throw std::runtime_error("Dynamic regularization suppport is disabled");
    if (control_.factorization_type != kCholeskyFactorization) throw std::runtime_error("Only Cholesky factorization is currently supported for simplicity");

#if 0
    {
        const auto &af = ordering_.assembly_forest;
        const auto &lf = *lower_factor_;
        std::cout << "Num roots: " << num_roots << std::endl;
        std::cout << "Serial memory required: " << SchurComplementStorage<Field>::storageNeeded(af.roots[0], af, lf) << std::endl;
        std::cout << "Serial memory required with expand-in-place: " << SchurComplementStorage<Field>::storageNeededExpandInPlace(af.roots[0], af, lf) << std::endl;
        std::cout << "Serial memory required with expand-in-place (optimal ordering): " << SchurComplementStorage<Field>::storageNeededExpandInPlaceOptimal(af.roots[0], af, lf) << std::endl;
    }
#endif

    BENCHMARK_SCOPED_TIMER_SECTION timer("BlockRightLooking<" + std::to_string(BlockSize) + ">");
    typedef ComplexBase<Field> Real;

    const Int max_threads = get_max_num_tbb_threads();
    shared_state_.num_tbb_threads = max_threads;

    // Compute flop-count estimates
    Buffer<double> &work_estimates = work_estimates_;
    double &total_work = total_work_;
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
    const double min_parallel_ratio_work = (total_work * control_.parallel_ratio_threshold) / max_threads;
    const double min_parallel_work = std::max(std::max(control_.min_parallel_threshold, min_parallel_ratio_work),
            max_threads < 2 ? std::numeric_limits<double>::infinity() : 0); // Forbid parallel execution
#if 0
    std::cout << "parallel_ratio_threshold: " << control_.parallel_ratio_threshold
              << ", min_parallel_threshold: " << control_.min_parallel_threshold << std::endl;
    std::cout << "Total work: " << total_work << ", min_parallel_work: " << min_parallel_work
              << ", max_threads: " << max_threads << std::endl;

    {
        Int parallel_supernode_count = 0;
        for (Int i = 0; i < num_supernodes; ++i) {
            if (work_estimates[i] >= min_parallel_work)
                ++parallel_supernode_count;
        }
        std::cout << "Parallel supernodes: " << parallel_supernode_count << std::endl;
    }
#endif

    // Construct the map from child structures to parent fronts.
    constructChildToParentMap(ordering_, lower_factor_.get());

    RightLookingSharedState<Field> &shared_state = shared_state_;
    if (shared_state.schur_complements.Size() != num_supernodes) {
        shared_state.schur_complements.Resize(num_supernodes);
        shared_state.schur_complement_storage.Resize(num_supernodes);
    }
#if REUSE_CHOLESKY_FLOWGRAPHS
    if (shared_state.cholesky_flowgraphs.size() != num_supernodes) {
        shared_state.cholesky_flowgraphs.clear();
        shared_state.cholesky_flowgraphs.resize(num_supernodes); // .assign(num_supernodes, nullptr) tries to copy a unique_ptr...
    }
#endif

#if 0
    {
        // Determine the amount of memory needed to store the Schur
        // complements in one contiguous buffer (without freeing).
        Int storage_size = 0;
        for (Int s = 0; s < num_supernodes; ++s) {
            const bool parallel = (work_estimates[s] >= min_parallel_work) && (max_threads > 1);
            Int degree = lower_factor_->blocks[s].height;
            if (parallel)
                storage_size += degree * degree;
            else {
                // Subtree storage stacks are allocated only for the roots of
                // serial trees.
                const bool serial_subtree_root = (ordering_.assembly_forest.parents[s] < 0) || (work_estimates[ordering_.assembly_forest.parents[s]] >= min_parallel_work) && (max_threads > 1);
                if (serial_subtree_root)
                    storage_size += shared_state_.schur_complement_storage[s].getStoragedNeeded(s, ordering_.assembly_forest, *lower_factor_);
            }
        }

        // std::cout << "Total Schur complement storage: " << storage_size * sizeof(Field) / 1024 / 1024 << " MB." << std::endl;
        // std::cout << "    As fraction of factor size: "
        //           << (double(storage_size) / factor_values_.data.Size()) << std::endl;
    }
#endif

#if CATAMARI_FINEGRAINED_TIMERS
    shared_state.finegrained_timers.allocate(num_supernodes);
#endif  // ifdef CATAMARI_FINEGRAINED_TIMERS

    SparseLDLResult<Field> result;

    shared_state.unsetFailed();

    Buffer<SparseLDLResult<Field>> result_contributions(num_roots);

    auto process_root = [&, min_parallel_work](Int root_index) {
        const Int root = ordering_.assembly_forest.roots[root_index];
        DynamicRegularizationParams<Field> subparams = dynamic_reg_params;
        subparams.offset = ordering_.supernode_offsets[root];
        bool success = BlockRightLookingSubtree<BlockSize>(
                root, subparams, work_estimates, min_parallel_work,
                &shared_state, &result_contributions[root_index]);
        FG_START_TIMER(shared_state.finegrained_timers, root, Deallocation);
        shared_state.schur_complement_storage[root].deallocate();
        FG_STOP_TIMER(shared_state.finegrained_timers, root, Deallocation);
        if (!success) shared_state.setFailed();
    };

    // const int old_max_threads = GetMaxBlasThreads();
    const bool parallel = (max_threads > 1) && (total_work >= min_parallel_work);

    if (parallel) {
        if (shared_state.tbb_ctx) shared_state.tbb_ctx->reset();
        else shared_state.tbb_ctx = std::make_unique<tbb::task_group_context>();
    }
    else shared_state.tbb_ctx.reset();

    // Recurse on each tree in the elimination forest.
    if (!parallel || num_roots <= 1) {
        for (Int root_index = 0; root_index < num_roots; ++root_index) {
            process_root(root_index);
            if (shared_state.hasFailed()) break;
        }
    }
    else {
        tbb::task_group tg(*(shared_state.tbb_ctx));
        for (Int root_index = 0; root_index < num_roots - 1; ++root_index) {
            tg.run([&process_root, root_index]() { process_root(root_index); });
        }
        process_root(num_roots - 1);
        tg.wait();
    }

    bool succeeded = !shared_state.hasFailed();
    if (succeeded) {
        for (Int index = 0; index < num_roots; ++index)
            MergeContribution(result_contributions[index], &result);
        if (dynamic_reg_params.enabled)
            MergeDynamicRegularizations(result_contributions, &result);
    }

    return result;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // CATAMARI_SUPERNODAL_LDL_OPENMP_H_
