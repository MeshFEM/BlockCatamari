#ifndef CATAMARI_SUPERNODAL_SOLVE_KERNELS_HPP
#define CATAMARI_SUPERNODAL_SOLVE_KERNELS_HPP

#include <algorithm>
#include <type_traits>
#include <Eigen/Core>
#include <MeshFEMCore/GlobalBenchmark.hh>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include "catamari/dense_basic_linear_algebra-impl.hpp"

namespace catamari { namespace supernodal_ldl { namespace solve_kernels {

// Scalar dimensions. Fusion and parallel dense updates have only been measured
// on Apple Silicon. Elsewhere retain the existing kernels (including handwritten
// AVX when available) and BLAS crossovers until we have target-specific profiles.
template<Int BS> struct Policy {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    static constexpr Int fused_forward_max_size = BS == 2 ? 384 : 192;
    static constexpr Int fused_backward_max_size = 48;
    static constexpr Int fused_backward_max_degree = 384;
    static constexpr Int backward_size = BS == 1 ? 64 : 96;
    static constexpr Int backward_degree = BS == 1 ? 100 : 384;
    static constexpr Int parallel_min_entries = BS == 3 ? 32768 : 0;
#else
    // Negative cutoffs disable fusion, including for empty supernodes.
    static constexpr Int fused_forward_max_size = -1;
    static constexpr Int fused_backward_max_size = -1;
    static constexpr Int fused_backward_max_degree = -1;
#if defined(__APPLE__)
    static constexpr Int backward_size = 64;
#else
    static constexpr Int backward_size = 18;
#endif
    static constexpr Int backward_degree = 100;
    static constexpr Int parallel_min_entries = 0;
#endif
};

constexpr double forward_work = 1e6, backward_work = 2e6;
constexpr int forward_max_depth = 4, backward_max_depth = 10;

// Process two solved columns at a time, updating both the remaining diagonal
// RHS and the contiguous Schur RHS while these factor columns are in cache.
// Unlike a GEMV, this revisits the Schur RHS for each column pair: intended for
// small nodes whose RHS fits in cache, not as a replacement for large BLAS calls.
template<class Field>
void fused_forward(const ConstBlasMatrixView<Field> &diag,
                   const ConstBlasMatrixView<Field> &lower, Field *rhs, Field *schur,
                   bool initialize) {
    using Vec = Eigen::Matrix<Field, 2, 1>;
    const Int n = diag.width;
    for (Int c = 0; c < n; c += 2) {
        const bool pair = c + 1 < n;
        const Field a = rhs[c] / diag(c, c);
        const Field b = pair ? (rhs[c + 1] - diag(c + 1, c) * a) / diag(c + 1, c + 1) : Field(0);
        rhs[c] = a;
        if (pair) rhs[c + 1] = b;
        auto update = [&](const ConstBlasMatrixView<Field> &matrix, Int first, Int end, Field *out, bool assign) {
            const Field *p = matrix.data + c * matrix.leading_dim;
            const Field *q = pair ? p + matrix.leading_dim : p;
            Int i = first;
            for (; i + 1 < end; i += 2) {
                const Vec value = Eigen::Map<const Vec>(p + i) * a + Eigen::Map<const Vec>(q + i) * b;
                if (assign) Eigen::Map<Vec>(out + i) = -value;
                else Eigen::Map<Vec>(out + i) -= value;
            }
            for (; i < end; ++i) {
                const Field value = p[i] * a + q[i] * b;
                if (assign) out[i] = -value;
                else out[i] -= value;
            }
        };
        update(diag, c + (pair ? 2 : 1), n, rhs, false);
        if (lower.height) update(lower, 0, lower.height, schur, initialize && c == 0);
    }
}

// Pull solved ancestor values directly using native block indices, then solve
// the diagonal transpose in the same descending two-column traversal. Real
// Cholesky only: the caller retains the existing complex/LDL/multi-RHS paths.
template<Int BS, class Field>
void fused_backward(const ConstBlasMatrixView<Field> &diag,
                    const ConstBlasMatrixView<Field> &lower, Field *rhs,
                    const Field *global, const Int *indices) {
    using Block = Eigen::Matrix<Field, BS, 1>;
    using Pair = Eigen::Matrix<Field, 2, 1>;
    using Vector = Eigen::Matrix<Field, Eigen::Dynamic, 1>;
    const Int n = diag.width;
    Int end = n;
    if (n % 2) {
        const Int c = --end;
        Field sum = 0;
        for (Int i = 0; i < lower.height / BS; ++i)
            sum += Eigen::Map<const Block>(lower.data + c * lower.leading_dim + BS * i)
                .dot(Eigen::Map<const Block>(global + indices[i]));
        rhs[c] = (rhs[c] - sum) / diag(c, c);
    }
    for (; end > 0; end -= 2) {
        const Int c = end - 2;
        Pair sum = Pair::Zero();
        using Columns = Eigen::Matrix<Field, BS, 2>;
        using Stride = std::conditional_t<BS == 1, Eigen::InnerStride<>, Eigen::OuterStride<>>;
        using Map = Eigen::Map<const Columns, 0, Stride>;
        for (Int i = 0; i < lower.height / BS; ++i)
            sum += Map(lower.data + c * lower.leading_dim + BS * i, BS, 2, Stride(lower.leading_dim)).transpose()
                * Eigen::Map<const Block>(global + indices[i]);
        const auto tail = Eigen::Map<const Vector>(rhs + end, n - end);
        sum[0] += Eigen::Map<const Vector>(diag.data + c * diag.leading_dim + end, n - end).dot(tail);
        sum[1] += Eigen::Map<const Vector>(diag.data + (c + 1) * diag.leading_dim + end, n - end).dot(tail);
        const Field b = (rhs[c + 1] - sum[1]) / diag(c + 1, c + 1);
        const Field a = (rhs[c] - sum[0] - diag(c + 1, c) * b) / diag(c, c);
        rhs[c + 1] = b;
        rhs[c] = a;
    }
}

// The same two-column triangular recurrence, with block-indexed output updates.
// Internal rows are owned by this serial group; external rows use its private boundary.
template<Int BS, class Field>
void fused_forward_scatter(const ConstBlasMatrixView<Field> &diag,
                           const ConstBlasMatrixView<Field> &lower, Field *rhs,
                           Field *global, Field *boundary, const Int *indices,
                           Int owned_blocks, const Int *external) {
    using Pair = Eigen::Matrix<Field, 2, 1>;
    using Block = Eigen::Matrix<Field, BS, 1>;
    const Int n = diag.width;
    for (Int c = 0; c < n; c += 2) {
        const bool pair = c + 1 < n;
        const Field a = rhs[c] / diag(c, c);
        const Field b = pair ? (rhs[c + 1] - diag(c + 1, c) * a) / diag(c + 1, c + 1) : Field(0);
        rhs[c] = a;
        if (pair) rhs[c + 1] = b;
        const Field *p = diag.data + c * diag.leading_dim;
        const Field *q = pair ? p + diag.leading_dim : p;
        Int r = c + (pair ? 2 : 1);
        for (; r + 1 < n; r += 2)
            Eigen::Map<Pair>(rhs + r) -= Eigen::Map<const Pair>(p + r) * a + Eigen::Map<const Pair>(q + r) * b;
        for (; r < n; ++r) rhs[r] -= p[r] * a + q[r] * b;
        p = lower.data + c * lower.leading_dim;
        q = pair ? p + lower.leading_dim : p;
        for (Int i = 0; i < owned_blocks; ++i)
            Eigen::Map<Block>(global + indices[i]) -= Eigen::Map<const Block>(p + BS * i) * a + Eigen::Map<const Block>(q + BS * i) * b;
        for (Int i = owned_blocks; i < lower.height / BS; ++i)
            Eigen::Map<Block>(boundary + external[i - owned_blocks]) -= Eigen::Map<const Block>(p + BS * i) * a + Eigen::Map<const Block>(q + BS * i) * b;
    }
}

template<class Field>
void multiply_serial(bool transpose, Field alpha, const ConstBlasMatrixView<Field> &a,
                     const ConstBlasMatrixView<Field> &x, Field beta, BlasMatrixView<Field> *y) {
    if (transpose) MatrixMultiplyAdjointNormal(alpha, a, x, beta, y);
    else MatrixMultiplyNormalNormal(alpha, a, x, beta, y);
}

template<Int BS, class Field>
void multiply(bool transpose, Field alpha, const ConstBlasMatrixView<Field> &a,
              const ConstBlasMatrixView<Field> &x, Field beta, BlasMatrixView<Field> *y) {
    const int threshold = Policy<BS>::parallel_min_entries;
    const Int outputs = transpose ? a.width : a.height;
    if (threshold > 0 && x.width == 1 && double(a.height) * a.width >= 2.0 * threshold) {
        // Split independent output entries. Each tile retains the factor's
        // column stride and invokes single-threaded BLAS; no output reductions.
        const Int chunks = std::min<Int>(tbb::this_task_arena::max_concurrency(),
            std::min<Int>(outputs / 64, Int(double(a.height) * a.width / threshold)));
        if (chunks > 1) {
            const Int grain = ((outputs + chunks - 1) / chunks + 7) / 8 * 8;
            tbb::parallel_for(Int(0), chunks, [&](Int chunk) {
                const Int first = chunk * grain, count = std::min(grain, outputs - first);
                if (count <= 0) return;
                auto block = transpose ? a.Submatrix(0, first, a.height, count) : a.Submatrix(first, 0, count, a.width);
                auto output = y->Submatrix(first, 0, count, 1);
                multiply_serial(transpose, alpha, block, x, beta, &output);
            });
            return;
        }
    }
    multiply_serial(transpose, alpha, a, x, beta, y);
}

}}} // namespace catamari::supernodal_ldl::solve_kernels
#endif
