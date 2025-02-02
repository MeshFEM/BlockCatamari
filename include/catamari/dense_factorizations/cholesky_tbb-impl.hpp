////////////////////////////////////////////////////////////////////////////////
// cholesky_tbb-impl.hpp
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Parallel dense Cholesky factorization implemented using a TBB flow graph.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  02/01/2025 15:37:24
*///////////////////////////////////////////////////////////////////////////////

#ifndef CATAMARI_DENSE_FACTORIZATIONS_CHOLESKY_TBB_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_CHOLESKY_TBB_IMPL_H_
#include "catamari/dense_basic_linear_algebra.hpp"
#include "cholesky-impl.hpp"
#include <tbb/tbb.h>
#include <Eigen/Dense>

namespace catamari {

template <class Field>
struct CholeskyFlowgraph {
    using Node = tbb::flow::continue_node<tbb::flow::continue_msg>;
    using NodePtr = std::shared_ptr<Node>;
    using Real = ComplexBase<Field>;

    CholeskyFlowgraph(const BlasMatrixView<double> &matrix_, Int tile_size, Int block_size_, bool force_serial = false)
        : block_size(block_size_), matrix(matrix_)
    {
        Int height = matrix.height;
        serial = force_serial || (height < 3 * tile_size) || (get_max_num_tbb_threads() < 2);
        if (serial) return;

        Int num_tiles = (height + tile_size - 1) / tile_size; // Number of tiles along width and height
        Eigen::Matrix<Node *, Eigen::Dynamic, Eigen::Dynamic> last_update;
        last_update.setConstant(num_tiles, num_tiles, nullptr);

        num_pivots = 0;

        for (Int j = 0; j < num_tiles; ++j) {
            Int tstart_j = j * tile_size;
            Int tsize_j = std::min(tile_size, height - tstart_j);

            // Cholesky factorization of diagonal block (j, j)
            nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tsize_j](const tbb::flow::continue_msg &msg) {
                BlasMatrixView<Field> block_j_j = matrix.Submatrix(tstart_j, tstart_j, tsize_j, tsize_j);
                const Int p = LowerCholeskyFactorization(block_size, &block_j_j);
                num_pivots += p; // Note: diagonal blocks are factorized sequentially due to dependencies, so this pivot count accumulation need not be atomic!
                if (p < block_j_j.height) {
                    // Stop early on failed factorization
                    tbb::task_group_context *ctx = tbb::task::current_context();
                    if (ctx) ctx->cancel_group_execution();
                }
                return msg;
            }));

            Node *factor_j_j = nodes.back().get();
            // The factorization of block (j, j) can only start when it has received its final update.
            if (last_update(j, j) != nullptr)
                tbb::flow::make_edge(*last_update(j, j), *factor_j_j);

            int solve_jp1_j_offset = nodes.size(); // index within `nodes` of the node responsible for the (j + 1, j) solve.
            for (Int i = j + 1; i < num_tiles; ++i) {
                Int tstart_i = i * tile_size;
                Int tsize_i = std::min(tile_size, height - tstart_i);

                // Solve for subdiagonal block (i, j)
                nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tstart_i, tsize_j, tsize_i](const tbb::flow::continue_msg &msg) {
                    BlasMatrixView<Field> block_j_j = matrix.Submatrix(tstart_j, tstart_j, tsize_j, tsize_j);
                    BlasMatrixView<Field> block_i_j = matrix.Submatrix(tstart_i, tstart_j, tsize_i, tsize_j);
                    RightLowerAdjointTriangularSolves(block_j_j.ToConst(), &block_i_j);
                    return msg;
                }));

                Node *solve_i_j = nodes.back().get();
                tbb::flow::make_edge(*factor_j_j, *solve_i_j);
                if (last_update(i, j) != nullptr) // Make sure the (i, j) block has received all its updates.
                    tbb::flow::make_edge(*last_update(i, j), *solve_i_j);
            }

            for (Int i = j + 1; i < num_tiles; ++i) {
                Int tstart_i = i * tile_size;
                Int tsize_i = std::min(tile_size, height - tstart_i);

                Node *solve_i_j = nodes[solve_jp1_j_offset + i - (j + 1)].get();

                for (Int j2 = j + 1; j2 < i; ++j2) {
                    Int tstart_j2 = j2 * tile_size;
                    Int tsize_j2 = std::min(tile_size, height - tstart_j2);
                    // Low-rank update of block (i, j2) for j2 < i
                    nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tstart_i, tstart_j2, tsize_j, tsize_i, tsize_j2](const tbb::flow::continue_msg &msg) {
                        BlasMatrixView<Field> block_i_j  = matrix.Submatrix(tstart_i , tstart_j , tsize_i , tsize_j );
                        BlasMatrixView<Field> block_j2_j = matrix.Submatrix(tstart_j2, tstart_j , tsize_j2, tsize_j );
                        BlasMatrixView<Field> block_i_j2 = matrix.Submatrix(tstart_i , tstart_j2, tsize_i , tsize_j2);

                        MatrixMultiplyNormalAdjoint(Field{-1}, block_i_j.ToConst(), block_j2_j.ToConst(),
                                                    Field{ 1}, &block_i_j2);
                        return msg;
                    }));
                    Node *solve_j2_j = nodes[solve_jp1_j_offset + j2 - (j + 1)].get();
                    tbb::flow::make_edge(*solve_i_j , *nodes.back());
                    tbb::flow::make_edge(*solve_j2_j, *nodes.back());
                    if (last_update(i, j2) != nullptr) // Make sure the (i, j2) block has received all its updates.
                        tbb::flow::make_edge(*last_update(i, j2), *nodes.back());
                    last_update(i, j2) = nodes.back().get();
                }

                // Low-rank update of the diagonal block (i, i)
                nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tstart_i, tsize_j, tsize_i](const tbb::flow::continue_msg &msg) {
                    BlasMatrixView<Field> block_i_i = matrix.Submatrix(tstart_i, tstart_i, tsize_i, tsize_i);
                    BlasMatrixView<Field> block_i_j = matrix.Submatrix(tstart_i, tstart_j, tsize_i, tsize_j);
                    LowerNormalHermitianOuterProduct(Real{-1}, block_i_j.ToConst(), Real{1}, &block_i_i);
                    return msg;
                }));
                tbb::flow::make_edge(*solve_i_j, *nodes.back());
                if (last_update(i, i) != nullptr) // Make sure the (i, i) block has received all its updates.
                    tbb::flow::make_edge(*last_update(i, i), *nodes.back());
                last_update(i, i) = nodes.back().get();
            }
        }
    }

    // Run on the matrix stored within this flowgraph, returning the number of successful pivots.
    Int run() {
        if (serial) return LowerCholeskyFactorizationDynamicBLASDispatch(block_size, &matrix);

        num_pivots = 0;
        nodes[0]->try_put(tbb::flow::continue_msg());
        g.wait_for_all();
        if (num_pivots < matrix.height) g.reset(); // Graph must be reset after it is cancelled.
        return num_pivots;
    }

    // Run on the passed matrix (whose dimensions must be the same as the
    // matrix originally passed to the constructor).
    Int run(BlasMatrixView<Field> &mat) {
        if (serial) return LowerCholeskyFactorizationDynamicBLASDispatch(block_size, &mat);

        if (mat.height != matrix.height || mat.width != matrix.width)
            throw std::runtime_error("CholeskyFlowgraph::run: input matrix dimensions do not those of this flowgraph.");

        matrix = mat;
        return run();
    }

    bool serial = false;
    Int block_size;
    BlasMatrixView<Field> matrix;
    tbb::flow::graph g;
    std::vector<NodePtr> nodes;
    Int num_pivots;
};

} // namespace catamari

#endif // CATAMARI_DENSE_FACTORIZATIONS_CHOLESKY_TBB_IMPL_H_
