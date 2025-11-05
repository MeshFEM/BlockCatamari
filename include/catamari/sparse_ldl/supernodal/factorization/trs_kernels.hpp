////////////////////////////////////////////////////////////////////////////////
// trs_kernels.hpp
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Experiment with optimized kernels for the low-level solve operations
//  executed on the frontal matrix and right-hand side vector(s).
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  11/02/2025 12:55:14
*///////////////////////////////////////////////////////////////////////////////
#ifndef TRS_KERNELS_HPP
#define TRS_KERNELS_HPP

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {
namespace trs_kernels {

////////////////////////////////////////////////////////////////////////////////
// MultiplyLowerBlockAdjoint
////////////////////////////////////////////////////////////////////////////////
// Execute `B[start:end, :] -= A^T B[I, :]`, where `I` is the supernode's lower block
// row index sequence and `A` is the supernode's rectangular lower block matrix
// (of shape `degree x size`).
// Here, `start := supernode_start` and `end := start + supernode_size`.
////////////////////////////////////////////////////////////////////////////////
template<class Field, Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjoint { // `catamari_legacy` implementation
    static void run(const bool conjugate,
             const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const Field *A_data, const Int A_leading_dim,
             const Int num_rhs, Field *B_data, const Int B_leading_dim) {
        const Field *a_col = A_data;
        for (Int k = 0; k < supernode_size; ++k) {
            for (Int i = 0; i < degree; ++i) {
                Field a = conjugate ? Conjugate(a_col[i]) : a_col[i];
                const Int row = I[i];
                Field *b_col = B_data;
                for (Int j = 0; j < num_rhs; ++j) {
                    b_col[k + supernode_start] -= a * b_col[row];
                    b_col += B_leading_dim;
                }
            }
            a_col += A_leading_dim;
        }
    }
};

template<Int BLOCK_SIZE> struct MultiplyLowerBlockAdjointEigenChunked;
template<Int BLOCK_SIZE> struct MultiplyLowerBlockAdjointSmall;

template<Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjoint<double, BLOCK_SIZE> { // Optimized kernel for double
    static void run(const bool /* conjugate */,
             const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
#if 1
            MultiplyLowerBlockAdjointEigenChunked<BLOCK_SIZE>::run(
                I, supernode_start, supernode_size, degree,
                A_data, A_leading_dim,
                num_rhs, B_data, B_leading_dim);
#else
            MultiplyLowerBlockAdjointSmall<BLOCK_SIZE>::run(
                I, supernode_start, supernode_size, degree,
                A_data, A_leading_dim,
                num_rhs, B_data, B_leading_dim);
#endif
    }
};

template<Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjointEigenChunked {
    static void run(
             const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
        const double * __restrict__  rhs_ptr = B_data;
              double * __restrict__ srhs_ptr = B_data + supernode_start;

        for (Int j = 0; j < num_rhs; ++j) {
            constexpr Int CHUNK_SIZE = (BLOCK_SIZE == 1) ? 4 : 2; // Currently tuned for M4 Pro...
            using Vec = VecN_T<double, CHUNK_SIZE>;
            using VecBlock = VecN_T<double, BLOCK_SIZE>;
            using Block = Eigen::Matrix<double, BLOCK_SIZE, CHUNK_SIZE>;
            // Eigen::Stride Gotcha: for single-row matrices, even in column
            // major format, it is the **inner stride** that is used to
            // determine the pointer increment between consecutive entries.
            // However, with 2 or more rows, it is the outer stride, as one
            // would expect.
            // See also: https://gitlab.com/libeigen/eigen/-/issues/416#note_709598886
            //           https://eigen.tuxfamily.org/bz/show_bug.cgi?id=416#c9
            using Stride = std::conditional_t<BLOCK_SIZE == 1, Eigen::InnerStride<>, Eigen::OuterStride<>>;
            using BMap = Eigen::Map<const Block, 0, Stride>;
            Int k;
            for (k = 0; k <= supernode_size - CHUNK_SIZE; k += CHUNK_SIZE) {
                Vec val = Vec::Zero();
                for (Int i = 0; i < degree; i += BLOCK_SIZE) {
                    // for (int c = 0; c < BLOCK_SIZE; ++c)
                    //     assert(I[i + c] == I[i] + c);
                    val += BMap(A_data + i + k * A_leading_dim, BLOCK_SIZE, CHUNK_SIZE, Stride(A_leading_dim)).transpose()
                        * Eigen::Map<const VecBlock>(rhs_ptr + I[i]);
                }
                Eigen::Map<Vec>(srhs_ptr + k) -= val;
            }
#if 1
            for (; k < supernode_size; ++k) {
                double val = 0;
                for (Int i = 0; i < degree; i += BLOCK_SIZE)
                    val += Eigen::Map<const VecBlock>(A_data + i + k * A_leading_dim).dot(Eigen::Map<const VecBlock>(rhs_ptr + I[i]));
                srhs_ptr[k] -= val;
            }
#else
            const Int k_start = k;
            const double * __restrict__ a_ptr_start = A_data + k_start * A_leading_dim;
            for (Int i = 0; i < degree; i += BLOCK_SIZE) {
                VecBlock b_entries = Eigen::Map<const VecBlock>(rhs_ptr + I[i]);
                const double * __restrict__ a_ptr = a_ptr_start + i;
                for (k = k_start; k < supernode_size; ++k) {
                    srhs_ptr[k] -= Eigen::Map<const VecBlock>(a_ptr).dot(b_entries);
                    a_ptr += A_leading_dim;
                }
            }
#endif
             rhs_ptr += B_leading_dim;
            srhs_ptr += B_leading_dim;
        }
    }
};

template<Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjointSmall { // Optimized kernel for double
    static void run(
             const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
        double * __restrict__ b_col = B_data;
        for (Int j = 0; j < num_rhs; ++j) {
            double * __restrict__ b_col_supernode = b_col + supernode_start;
            for (Int i = 0; i < degree; ++i) {
                const double * __restrict__ a_ptr = A_data + i;
                const double b_entry = b_col[I[i]];
                for (Int k = 0; k < supernode_size; ++k) {
                    b_col_supernode[k] -= (*a_ptr) * b_entry;
                    a_ptr += A_leading_dim;
                }
            }
            b_col += B_leading_dim;
        }
    }
};

template<>
struct MultiplyLowerBlockAdjointSmall<2> {
    static void run(
             const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
        double * __restrict__ b_col = B_data;
        for (Int j = 0; j < num_rhs; ++j) {
            double * __restrict__ b_col_supernode = b_col + supernode_start;
#if 0
            Eigen::Map<Eigen::VectorXd> b_map(b_col_supernode, supernode_size);
            for (Int i = 0; i < degree; i += 2) {
                const double * __restrict__ a_ptr = A_data + i;
                const Vec2_T<double> b_strip = Eigen::Map<const Vec2_T<double>>(b_col + I[i]);
                Eigen::Map<const Eigen::Matrix<double, 2, Eigen::Dynamic>, 0, Eigen::OuterStride<>> A_map(a_ptr, 2, supernode_size, Eigen::OuterStride<>(A_leading_dim));
                b_map -= A_map.transpose() * b_strip;
            }
#else
            for (Int i = 0; i < degree; i += 2) {
                const double * __restrict__ a_ptr = A_data + i;
                const Vec2_T<double> b_strip = Eigen::Map<const Vec2_T<double>>(b_col + I[i]);

                for (Int k = 0; k < supernode_size; k += 2) {
                    Mat2_T<double> A_block;
                    A_block(0, 0) = a_ptr[0];
                    A_block(1, 0) = a_ptr[1];
                    A_block(0, 1) = a_ptr[A_leading_dim];
                    A_block(1, 1) = a_ptr[A_leading_dim + 1];
                    Eigen::Map<Vec2_T<double>>(b_col_supernode + k) -= A_block.transpose() * b_strip;
                    a_ptr += 2 * A_leading_dim;
                }
            }
#endif
            b_col += B_leading_dim;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// MultiplyLowerBlock
////////////////////////////////////////////////////////////////////////////////
// Execute `B[:, :] -= A B[start:end, :]`, where `I` is the supernode's lower block
// row index sequence and `A` is the supernode's rectangular lower block matrix
// (of shape `degree x size`).
// Here, `start := supernode_start` and `end := start + supernode_size`.
////////////////////////////////////////////////////////////////////////////////
template<class Field, Int BLOCK_SIZE>
struct MultiplyLowerBlock { // `catamari_legacy` implementation
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const Field * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, Field * __restrict__ B_data, const Int B_leading_dim) {
        for (Int j = 0; j < num_rhs; ++j) {
            Field *b_col = B_data;
            for (Int k = 0; k < supernode_size; ++k) {
                const Field eta = b_col[k + supernode_start];
                const Field *A_col = A_data + k * A_leading_dim;
                for (Int i = 0; i < degree; ++i)
                    b_col[I[i]] -= A_col[i] * eta;
            }
            b_col += B_leading_dim;
        }
    }
};

#if 1
template<Int BLOCK_SIZE>
struct MultiplyLowerBlock<double, BLOCK_SIZE> { // Optimized kernel for double
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
              double * __restrict__  rhs_ptr = B_data;
        const double * __restrict__ srhs_ptr = B_data + supernode_start;

        for (Int j = 0; j < num_rhs; ++j) {
#if 1
          constexpr Int CHUNK_SIZE = 6;
          using Vec = VecN_T<double, CHUNK_SIZE>;
          Int i;
          for (i = 0; i <= degree - CHUNK_SIZE; i += CHUNK_SIZE) {
              const double * __restrict__ a_row = A_data + i;

              Vec val;
              if constexpr (BLOCK_SIZE == 1) {
                  val = Eigen::Map<const Vec>(a_row) * srhs_ptr[0];
                  for (Int k = 1; k < supernode_size; ++k) {
                      a_row += A_leading_dim;
                      val += Eigen::Map<const Vec>(a_row) * srhs_ptr[k];
                  }
              }
              else {
                  using MMap = Eigen::Map<const Eigen::Matrix<double, CHUNK_SIZE, BLOCK_SIZE>, 0, Eigen::OuterStride<>>;
                  using CVMap = Eigen::Map<const VecN_T<double, BLOCK_SIZE>>;
                  val = MMap(a_row, CHUNK_SIZE, BLOCK_SIZE, Eigen::OuterStride<>(A_leading_dim)) * CVMap(srhs_ptr);
                  for (Int k = BLOCK_SIZE; k < supernode_size; k += BLOCK_SIZE) {
                      a_row += BLOCK_SIZE * A_leading_dim;
                      val += MMap(a_row, CHUNK_SIZE, BLOCK_SIZE, Eigen::OuterStride<>(A_leading_dim)) * CVMap(srhs_ptr + k);
                  }
              }

              rhs_ptr[I[i + 0]] -= val[0];
              rhs_ptr[I[i + 1]] -= val[1];
              rhs_ptr[I[i + 2]] -= val[2];
              rhs_ptr[I[i + 3]] -= val[3];
              rhs_ptr[I[i + 4]] -= val[4];
              rhs_ptr[I[i + 5]] -= val[5];
          }
          for (; i < degree; ++i) {
              const double * __restrict__ a_row = A_data + i;
              double val;
              if constexpr (BLOCK_SIZE == 1) {
                  val = (*a_row) * srhs_ptr[0];
                  for (Int k = 1; k < supernode_size; ++k) {
                      a_row += A_leading_dim;
                      val += (*a_row) * srhs_ptr[k];
                  }
              }
              else {
                  using MMap = Eigen::Map<const Eigen::Matrix<double, 1, BLOCK_SIZE>, 0, Eigen::InnerStride<>>;
                  using CVMap = Eigen::Map<const VecN_T<double, BLOCK_SIZE>>;
                  val = MMap(a_row, 1, BLOCK_SIZE, Eigen::InnerStride<>(A_leading_dim)).transpose().dot(CVMap(srhs_ptr));
                  for (Int k = BLOCK_SIZE; k < supernode_size; k += BLOCK_SIZE) {
                      a_row += BLOCK_SIZE * A_leading_dim;
                      val += MMap(a_row, 1, BLOCK_SIZE, Eigen::InnerStride<>(A_leading_dim)).transpose().dot(CVMap(srhs_ptr + k));
                  }
              }
              rhs_ptr[I[i]] -= val;
          }
#else
          using Vec = VecN_T<double, BLOCK_SIZE>;
          using Mat = MatN_T<double, BLOCK_SIZE>;
          using Stride = std::conditional_t<BLOCK_SIZE == 1, Eigen::InnerStride<>, Eigen::OuterStride<>>;
          using CVMap = Eigen::Map<const Vec>;
          using MMap = Eigen::Map<const Mat, 0, Stride>;
          for (Int i = 0; i < degree; i += BLOCK_SIZE) {
             // Eigen::Map<const Eigen::Matrix<double, BLOCK_SIZE, Eigen::Dynamic>, 0, Stride> A_block(A_data + i, BLOCK_SIZE, supernode_size, Stride(A_leading_dim));
             // Eigen::Map<Vec>(rhs_ptr + I[i]) -= A_block * Eigen::Map<const Eigen::VectorXd>(srhs_ptr, supernode_size);

             const double * __restrict__ a_row = A_data + i;
             Vec val = MMap(a_row, BLOCK_SIZE, BLOCK_SIZE, Stride(A_leading_dim)) * CVMap(srhs_ptr);
             for (Int k = BLOCK_SIZE; k < supernode_size; k += BLOCK_SIZE) {
                 a_row += BLOCK_SIZE * A_leading_dim;
                 val += MMap(a_row, BLOCK_SIZE, BLOCK_SIZE, Stride(A_leading_dim)) * CVMap(srhs_ptr + k);
             }
             Eigen::Map<Vec>(rhs_ptr + I[i]) -= val;
          }
#endif
          rhs_ptr += B_leading_dim;
          srhs_ptr += B_leading_dim;
        }
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////
// SolveLowerTri
////////////////////////////////////////////////////////////////////////////////
// Execute `b = L \ b`
////////////////////////////////////////////////////////////////////////////////
template<class Field, Int BLOCK_SIZE>
struct SolveLowerTri {
    static void run(const Int supernode_size,
             const Field * __restrict__ L_data, const Int L_leading_dim,
             Field * __restrict__ b) {
        const Field *__restrict__ L_col = L_data;
        for (Int j = 0; j < supernode_size; ++j) {
            b[j] /= L_col[j];
            const Field eta = b[j];
            for (Int i = j + 1; i < supernode_size; ++i)
                b[i] -= L_col[i] * eta;
            L_col += L_leading_dim;
        }
    }
};

template<>
struct SolveLowerTri<double, 2> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
             double * __restrict__ b) {
        const double *__restrict__ L_col_a = L_data;
        const double *__restrict__ L_col_b = L_data + L_leading_dim;
        using V2d = Vec2_T<double>;
        using  VMap = Eigen::Map<      V2d, Eigen::Aligned16>;
        using CVMap = Eigen::Map<const V2d, Eigen::Aligned16>;

        for (Int j = 0; j < supernode_size; j += 2) {
            VMap eta(b + j);
            CVMap L_col_strip_a(L_col_a + j);

            eta[0] /= L_col_strip_a[0];
            eta[1] -= eta[0] * L_col_strip_a[1];
            eta[1] /= L_col_b[j + 1];

            for (Int i = j + 2; i < supernode_size; i += 2)
                VMap(b + i) -= CVMap(L_col_a + i) * eta[0] + CVMap(L_col_b + i) * eta[1];
            L_col_a += L_leading_dim * 2;
            L_col_b += L_leading_dim * 2;
        }
    }
};

template<>
struct SolveLowerTri<double, 3> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
             double * __restrict__ b) {
        const double *__restrict__ L_col_a = L_data;
        const double *__restrict__ L_col_b = L_col_a + L_leading_dim;
        const double *__restrict__ L_col_c = L_col_b + L_leading_dim;
        using V3d = Vec3_T<double>;
        using  VMap = Eigen::Map<      V3d>;
        using CVMap = Eigen::Map<const V3d>;

        for (Int j = 0; j < supernode_size; j += 3) {
            VMap eta(b + j);
            CVMap L_col_strip_a(L_col_a + j);

            eta[0] /= L_col_strip_a[0];
            eta[1] -= eta[0] * L_col_strip_a[1];
            eta[1] /= L_col_b[j + 1];
            eta[2] -= eta[0] * L_col_strip_a[2] + eta[1] * L_col_b[j + 2];
            eta[2] /= L_col_c[j + 2];

            for (Int i = j + 3; i < supernode_size; i += 3)
                VMap(b + i) -= CVMap(L_col_a + i) * eta[0] + CVMap(L_col_b + i) * eta[1] + CVMap(L_col_c + i) * eta[2];;
            L_col_a += L_leading_dim * 3;
            L_col_b += L_leading_dim * 3;
            L_col_c += L_leading_dim * 3;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// SolveLowerTriAdjoint
////////////////////////////////////////////////////////////////////////////////
// Execute `b = L^H \ b`
////////////////////////////////////////////////////////////////////////////////
template<class Field, Int BLOCK_SIZE>
struct SolveLowerTriAdjoint {
    static void run(const Int supernode_size,
            const Field * __restrict__ L_data, const Int L_leading_dim,
            Field * __restrict__ b) {
        const Field * __restrict__ L_col = L_data + (supernode_size - 1) * L_leading_dim;
        for (Int j = supernode_size - 1; j >= 0; --j) {
            Field eta = b[j];
            for (Int i = j + 1; i < supernode_size; ++i)
                eta -= Conjugate(L_col[i]) * b[i];
            eta /= Conjugate(L_col[j]);
            b[j] = eta;
            L_col -= L_leading_dim;
        }
    }
};

template<>
struct SolveLowerTriAdjoint<double, 2> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
             double * __restrict__ b) {
        const double *__restrict__ L_col_a = L_data + L_leading_dim * (supernode_size - 2);
        const double *__restrict__ L_col_b = L_col_a + L_leading_dim;
        using V2d = Vec2_T<double>;
        using  VMap = Eigen::Map<      V2d, Eigen::Aligned16>;
        using CVMap = Eigen::Map<const V2d, Eigen::Aligned16>;

        for (Int j = supernode_size - 2; j >= 0; j -= 2) {
            V2d eta = VMap(b + j);

            for (Int i = j + 2; i < supernode_size; i += 2) {
                V2d b_strip = CVMap(b + i);
                eta[0] -= CVMap(L_col_a + i).dot(b_strip);
                eta[1] -= CVMap(L_col_b + i).dot(b_strip);
            }

            CVMap L_col_strip_a(L_col_a + j);
            eta[1] /= L_col_b[j + 1];
            eta[0] -= eta[1] * L_col_strip_a[1];
            eta[0] /= L_col_strip_a[0];

            VMap(b + j) = eta;

            L_col_a -= L_leading_dim * 2;
            L_col_b -= L_leading_dim * 2;
        }
    }
};

template<>
struct SolveLowerTriAdjoint<double, 3> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
             double * __restrict__ b) {
        const double *__restrict__ L_col_a = L_data  + L_leading_dim * (supernode_size - 3);
        const double *__restrict__ L_col_b = L_col_a + L_leading_dim;
        const double *__restrict__ L_col_c = L_col_b + L_leading_dim;
        using V3d = Vec3_T<double>;
        using  VMap = Eigen::Map<      V3d>;
        using CVMap = Eigen::Map<const V3d>;

        for (Int j = supernode_size - 3; j >= 0; j -= 3) {
            V3d eta = VMap(b + j);

            for (Int i = j + 3; i < supernode_size; i += 3) {
                V3d b_strip = CVMap(b + i);
                eta[0] -= CVMap(L_col_a + i).dot(b_strip);
                eta[1] -= CVMap(L_col_b + i).dot(b_strip);
                eta[2] -= CVMap(L_col_c + i).dot(b_strip);
            }

            CVMap L_col_strip_a(L_col_a + j);
            eta[2] /= L_col_c[j + 2];
            eta[1] -= eta[2] * L_col_b[j + 2];
            eta[1] /= L_col_b[j + 1];
            eta[0] -= eta[1] * L_col_strip_a[1] + eta[2] * L_col_strip_a[2];
            eta[0] /= L_col_strip_a[0];

            VMap(b + j) = eta;

            L_col_a -= L_leading_dim * 3;
            L_col_b -= L_leading_dim * 3;
            L_col_c -= L_leading_dim * 3;
        }
    }
};

}
}
}

#endif /* end of include guard: TRS_KERNELS_HPP */
