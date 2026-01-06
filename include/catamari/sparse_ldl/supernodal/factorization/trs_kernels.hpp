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

template<Int BLOCK_SIZE> struct MultiplyLowerBlockAdjointEigenUnchunked;
template<Int BLOCK_SIZE> struct MultiplyLowerBlockAdjointEigenChunked;
template<Int BLOCK_SIZE> struct MultiplyLowerBlockAdjointEigenChunkedAlternate;
template<Int BLOCK_SIZE> struct MultiplyLowerBlockAdjointEigenChunkedOuterInner;
template<Int BLOCK_SIZE> struct MultiplyLowerBlockAdjointSmall;

template<Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjoint<double, BLOCK_SIZE> { // Optimized kernel for double
    static void run(const bool /* conjugate */,
             const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
#if defined(__APPLE__)
            MultiplyLowerBlockAdjointEigenChunked<BLOCK_SIZE>::run(
#else
            MultiplyLowerBlockAdjointEigenUnchunked<BLOCK_SIZE>::run( // Seems to be our fastest SSE/AVX kernel
#endif
                I, supernode_start, supernode_size, degree,
                A_data, A_leading_dim,
                num_rhs, B_data, B_leading_dim);
    }
};

template<Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjointEigenUnchunked {
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
        const double * __restrict__  rhs_ptr = B_data;
              double * __restrict__ srhs_ptr = B_data + supernode_start;

        for (Int j = 0; j < num_rhs; ++j) {
            for (Int k = 0; k < supernode_size; ++k) {
                const double * __restrict__ a_ptr = A_data + k * A_leading_dim;
                double val = 0;
                for (Int i = 0; i < degree; i += BLOCK_SIZE) {
                    using VecBlock = VecN_T<double, BLOCK_SIZE>;
                    VecBlock rhs_strip = Eigen::Map<const VecBlock>(rhs_ptr + I[i]);
                    val += Eigen::Map<const VecBlock>(a_ptr).dot(rhs_strip);
                    a_ptr += BLOCK_SIZE;
                }
                srhs_ptr[k] -= val;
            }
            srhs_ptr += B_leading_dim;
             rhs_ptr += B_leading_dim;
        }
    }
};

#if !defined(__APPLE__)
template<>
struct MultiplyLowerBlockAdjointEigenUnchunked<2> {
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim)
    {
        const double * __restrict__ rhs_ptr  = B_data;
              double * __restrict__ srhs_ptr = B_data + supernode_start;

        for (Int j = 0; j < num_rhs; ++j) {
            for (Int k = 0; k < supernode_size; k += 2) {
                const double * __restrict__ a0 = A_data + k * A_leading_dim;
                const double * __restrict__ a1 = a0 + A_leading_dim;
                __m128d acc0 = _mm_setzero_pd();
                __m128d acc1 = _mm_setzero_pd();
                for (Int i = 0; i < degree; i += 2) {
                    __m128d b = _mm_load_pd(rhs_ptr + I[i]);
                    __m128d a0v = _mm_load_pd(a0);
                    __m128d a1v = _mm_load_pd(a1);
                    acc0 = _mm_fmadd_pd(a0v, b, acc0);
                    acc1 = _mm_fmadd_pd(a1v, b, acc1);
                    a0 += 2;
                    a1 += 2;
                }

                __m128d vals = _mm_hadd_pd(acc0, acc1);
                __m128d srhs = _mm_load_pd(srhs_ptr + k);
                srhs = _mm_sub_pd(srhs, vals);
                _mm_store_pd(srhs_ptr + k, srhs);
            }
            srhs_ptr += B_leading_dim;
            rhs_ptr  += B_leading_dim;
        }
    }
};
#endif

#if !defined(__APPLE__)
template<>
struct MultiplyLowerBlockAdjointEigenUnchunked<3> {
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim)
    {
        const double * __restrict__ rhs_ptr  = B_data;
              double * __restrict__ srhs_ptr = B_data + supernode_start;

        for (Int j = 0; j < num_rhs; ++j) {
            for (Int k = 0; k < supernode_size; k += 3) {
                const double * __restrict__ a0 = A_data + k * A_leading_dim;
                const double * __restrict__ a1 = a0 + A_leading_dim;
                const double * __restrict__ a2 = a1 + A_leading_dim;
                // Accumulators `acc*` accumulate cwise products of each column of a 3x3 block of A:
                //     +---+---+---+
                //     | 0 | 1 | 2 |
                //     +---+---+---+
                //     | 0 | 1 | 2 |   <-- accumulator var indices
                //     +---+---+---+
                //     | 3 | 3 | 4 |
                //     +---+---+---+
                // with corresponding 3-strip of b.
                __m128d acc0 = _mm_setzero_pd();
                __m128d acc1 = _mm_setzero_pd();
                __m128d acc2 = _mm_setzero_pd();
                __m128d acc3 = _mm_setzero_pd();
                double acc4 = 0;
                for (Int i = 0; i < degree; i += 3) {
                    const double * __restrict__ b_ptr = rhs_ptr + I[i];
                    __m128d b = _mm_loadu_pd(b_ptr);

                    __m128d a0v = _mm_loadu_pd(a0);
                    __m128d a1v = _mm_loadu_pd(a1);
                    __m128d a2v = _mm_loadu_pd(a2);

                    acc0 = _mm_fmadd_pd(a0v, b, acc0);
                    acc1 = _mm_fmadd_pd(a1v, b, acc1);
                    acc2 = _mm_fmadd_pd(a2v, b, acc2);

                    __m128d a3v = _mm_set_pd(a1[2], a0[2]);
                    __m128d s = _mm_set_pd1(b_ptr[2]);
                    acc3 = _mm_fmadd_pd(a3v, s, acc3);

                    acc4 += a2[2] * b_ptr[2];

                    a0 += 3;
                    a1 += 3;
                    a2 += 3;
                }

                __m128d srhs = _mm_loadu_pd(srhs_ptr + k);
                srhs = _mm_sub_pd(srhs, _mm_hadd_pd(acc0, acc1));
                srhs = _mm_sub_pd(srhs, acc3);
                _mm_storeu_pd(srhs_ptr + k, srhs);

                acc2 = _mm_hadd_pd(acc2, acc2);
                srhs_ptr[k + 2] -= _mm_cvtsd_f64(acc2) + acc4;
            }
            srhs_ptr += B_leading_dim;
            rhs_ptr  += B_leading_dim;
        }
    }
};
#endif

#if !defined(__APPLE__)
template<>
struct MultiplyLowerBlockAdjointEigenChunkedAlternate<2> {
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim)
    {
        double * __restrict__ b_col = B_data;
        for (Int j = 0; j < num_rhs; ++j) {
            double* __restrict__ srhs_ptr = B_data + supernode_start + j * B_leading_dim;

            Int i = 0;
            for (; i + 8 <= degree; i += 8) {
                __m128d b0 = _mm_load_pd(b_col + I[i]);
                __m128d b1 = _mm_load_pd(b_col + I[i + 2]);
                __m128d b2 = _mm_load_pd(b_col + I[i + 4]);
                __m128d b3 = _mm_load_pd(b_col + I[i + 6]);

                const double * __restrict__ a0 = A_data + i;
                const double * __restrict__ a1 = a0 + A_leading_dim;

                Int k = 0;
                for (; k < supernode_size; k += 2) {
                    __m128d acc0 = _mm_mul_pd(_mm_load_pd(a0), b0);
                    __m128d acc1 = _mm_mul_pd(_mm_load_pd(a1), b0);

                    acc0 = _mm_fmadd_pd(_mm_load_pd(a0 + 2), b1, acc0);
                    acc1 = _mm_fmadd_pd(_mm_load_pd(a1 + 2), b1, acc1);

                    acc0 = _mm_fmadd_pd(_mm_load_pd(a0 + 4), b2, acc0);
                    acc1 = _mm_fmadd_pd(_mm_load_pd(a1 + 4), b2, acc1);

                    acc0 = _mm_fmadd_pd(_mm_load_pd(a0 + 6), b3, acc0);
                    acc1 = _mm_fmadd_pd(_mm_load_pd(a1 + 6), b3, acc1);

                    _mm_store_pd(srhs_ptr + k, _mm_sub_pd(_mm_load_pd(srhs_ptr + k), _mm_hadd_pd(acc0, acc1)));

                    a0 += 2 * A_leading_dim;
                    a1 += 2 * A_leading_dim;
                }
            }

            for (; i + 4 <= degree; i += 4) {
                __m128d b0 = _mm_load_pd(b_col + I[i]);
                __m128d b1 = _mm_load_pd(b_col + I[i + 2]);

                const double * __restrict__ a0 = A_data + i;
                const double * __restrict__ a1 = a0 + A_leading_dim;

                Int k = 0;
                for (; k < supernode_size; k += 2) {
                    __m128d acc0 = _mm_mul_pd(_mm_load_pd(a0), b0);
                    __m128d acc1 = _mm_mul_pd(_mm_load_pd(a1), b0);

                    acc0 = _mm_fmadd_pd(_mm_load_pd(a0 + 2), b1, acc0);
                    acc1 = _mm_fmadd_pd(_mm_load_pd(a1 + 2), b1, acc1);

                    _mm_store_pd(srhs_ptr + k, _mm_sub_pd(_mm_load_pd(srhs_ptr + k), _mm_hadd_pd(acc0, acc1)));

                    a0 += 2 * A_leading_dim;
                    a1 += 2 * A_leading_dim;
                }
            }

            for (; i < degree; i += 2) {
                __m128d b0 = _mm_load_pd(b_col + I[i]);

                const double * __restrict__ a0 = A_data + i;
                const double * __restrict__ a1 = a0 + A_leading_dim;

                Int k = 0;
                for (; k < supernode_size; k += 2) {
                    __m128d acc0 = _mm_mul_pd(_mm_load_pd(a0), b0);
                    __m128d acc1 = _mm_mul_pd(_mm_load_pd(a1), b0);

                    __m128d vals = _mm_hadd_pd(acc0, acc1);
                    __m128d srhs = _mm_load_pd(srhs_ptr + k);
                    srhs = _mm_sub_pd(srhs, vals);
                    _mm_store_pd(srhs_ptr + k, srhs);

                    a0 += 2 * A_leading_dim;
                    a1 += 2 * A_leading_dim;
                }
            }

            b_col += B_leading_dim;
        }
    }
};
#endif

template<Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjointEigenChunked {
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
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
struct MultiplyLowerBlockAdjointEigenChunkedAlternate { // Transposed loop/chunking order from above (to match MultiplyLowerBlock kernel)
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
        const double * __restrict__  rhs_ptr = B_data;
              double * __restrict__ srhs_ptr = B_data + supernode_start;

        for (Int j = 0; j < num_rhs; ++j) {
          constexpr Int CHUNK_SIZE = 6;
          using Vec = VecN_T<double, CHUNK_SIZE>;
          Int i;
          for (i = 0; i <= degree - CHUNK_SIZE; i += CHUNK_SIZE) {
              const double * __restrict__ a_row = A_data + i;

              Vec rhs_strip;
              rhs_strip[0] = rhs_ptr[I[i + 0]];
              rhs_strip[1] = rhs_ptr[I[i + 1]];
              rhs_strip[2] = rhs_ptr[I[i + 2]];
              rhs_strip[3] = rhs_ptr[I[i + 3]];
              rhs_strip[4] = rhs_ptr[I[i + 4]];
              rhs_strip[5] = rhs_ptr[I[i + 5]];

              if constexpr (BLOCK_SIZE == 1) {
                  for (Int k = 0; k < supernode_size; ++k) {
                      srhs_ptr[k] -= Eigen::Map<const Vec>(a_row).dot(rhs_strip);
                      a_row += A_leading_dim;
                  }
              }
              else {
                  using MMap = Eigen::Map<const Eigen::Matrix<double, CHUNK_SIZE, BLOCK_SIZE>, 0, Eigen::OuterStride<>>;
                  using VMap = Eigen::Map<VecN_T<double, BLOCK_SIZE>>;
                  // Not sure how helpful this blocking is...
                  for (Int k = 0; k < supernode_size; k += BLOCK_SIZE) {
                      VMap(srhs_ptr + k) -= MMap(a_row, CHUNK_SIZE, BLOCK_SIZE, Eigen::OuterStride<>(A_leading_dim)).transpose() * rhs_strip;
                      a_row += BLOCK_SIZE * A_leading_dim;
                  }
              }
          }
          if constexpr ((BLOCK_SIZE == 2) || (BLOCK_SIZE == 3)) { // must divide CHUNK_SIZE!
              for (; i < degree; i += BLOCK_SIZE) {
                  const double * __restrict__ a_row = A_data + i;
                  using VBlock = VecN_T<double, BLOCK_SIZE>;
                  VBlock rhs_strip = Eigen::Map<const VBlock>(rhs_ptr + I[i]);
                  using MMap = Eigen::Map<const Eigen::Matrix<double, BLOCK_SIZE, BLOCK_SIZE>, 0, Eigen::OuterStride<>>;
                  for (Int k = 0; k < supernode_size; k += BLOCK_SIZE) {
                      Eigen::Map<VBlock>(srhs_ptr + k) -= MMap(a_row, BLOCK_SIZE, BLOCK_SIZE, Eigen::OuterStride<>(A_leading_dim)).transpose() * rhs_strip;
                      a_row += BLOCK_SIZE * A_leading_dim;
                  }
                  // Eigen::Map<Eigen::VectorXd>(srhs_ptr, supernode_size) -= Eigen::Map<const Eigen::Matrix<double, BLOCK_SIZE, Eigen::Dynamic>, 0, Eigen::OuterStride<>>(a_row, BLOCK_SIZE, supernode_size, Eigen::OuterStride<>(A_leading_dim)).transpose() * rhs_strip;
              }
          }
          else {
              for (; i < degree; ++i) {
                  const double * __restrict__ a_row = A_data + i;
                  const double rhs_val = rhs_ptr[I[i]];
                  for (Int k = 0; k < supernode_size; ++k) {
                      srhs_ptr[k] -= (*a_row) * rhs_val;
                      a_row += A_leading_dim;
                  }
              }
          }

          rhs_ptr += B_leading_dim;
          srhs_ptr += B_leading_dim;
        }
    }
};

template<Int BLOCK_SIZE>
struct MultiplyLowerBlockAdjointEigenChunkedOuterInner { // Transposed loop/chunking order from above (to match MultiplyLowerBlock kernel)
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
        const double * __restrict__  rhs_ptr = B_data;
              double * __restrict__ srhs_ptr = B_data + supernode_start;

        for (Int j = 0; j < num_rhs; ++j) {
          constexpr Int CHUNK_SIZE = 4;
          using Vec = VecN_T<double, CHUNK_SIZE>;
          Int i;
          for (i = 0; i <= degree - CHUNK_SIZE; i += CHUNK_SIZE) {
              Vec rhs_strip;
              rhs_strip[0] = rhs_ptr[I[i + 0]];
              rhs_strip[1] = rhs_ptr[I[i + 1]];
              rhs_strip[2] = rhs_ptr[I[i + 2]];
              rhs_strip[3] = rhs_ptr[I[i + 3]];
              // rhs_strip[4] = rhs_ptr[I[i + 4]];
              // rhs_strip[5] = rhs_ptr[I[i + 5]];

              const double * __restrict__ a_row = A_data + i;
              Int k = 0;
              {
                  constexpr Int INNER_CHUNK_SIZE = 4;
                  using MMap = Eigen::Map<const Eigen::Matrix<double, CHUNK_SIZE, INNER_CHUNK_SIZE>, 0, Eigen::OuterStride<>>;
                  using VMap = Eigen::Map<VecN_T<double, INNER_CHUNK_SIZE>>;
                  for (; k <= supernode_size - INNER_CHUNK_SIZE; k += INNER_CHUNK_SIZE) {
                      VMap(srhs_ptr + k) -= MMap(a_row, CHUNK_SIZE, INNER_CHUNK_SIZE, Eigen::OuterStride<>(A_leading_dim)).transpose() * rhs_strip;
                      a_row += INNER_CHUNK_SIZE * A_leading_dim;
                  }
              }

              for (; k < supernode_size; ++k) {
                  srhs_ptr[k] -= Eigen::Map<const Vec>(a_row).dot(rhs_strip);
                  a_row += A_leading_dim;
              }
          }

          // Remaining rows
          if constexpr ((BLOCK_SIZE == 2) || (BLOCK_SIZE == 3)) { // must divide CHUNK_SIZE!
              for (; i < degree; i += BLOCK_SIZE) {
                  const double * __restrict__ a_row = A_data + i;
                  using VBlock = VecN_T<double, BLOCK_SIZE>;
                  VBlock rhs_strip = Eigen::Map<const VBlock>(rhs_ptr + I[i]);
                  using MMap = Eigen::Map<const Eigen::Matrix<double, BLOCK_SIZE, BLOCK_SIZE>, 0, Eigen::OuterStride<>>;
                  for (Int k = 0; k < supernode_size; k += BLOCK_SIZE) {
                      Eigen::Map<VBlock>(srhs_ptr + k) -= MMap(a_row, BLOCK_SIZE, BLOCK_SIZE, Eigen::OuterStride<>(A_leading_dim)).transpose() * rhs_strip;
                      a_row += BLOCK_SIZE * A_leading_dim;
                  }
              }
          }
          else {
              for (; i < degree; ++i) {
                  const double * __restrict__ a_row = A_data + i;
                  const double rhs_val = rhs_ptr[I[i]];
                  for (Int k = 0; k < supernode_size; ++k) {
                      srhs_ptr[k] -= (*a_row) * rhs_val;
                      a_row += A_leading_dim;
                  }
              }
          }

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

#if !defined(__APPLE__)
template<>
struct MultiplyLowerBlock<double, 2> { // Optimized x86 kernel for double
    static void run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
              double * __restrict__  rhs_ptr = B_data;
        const double * __restrict__ srhs_ptr = B_data + supernode_start;
        Int i = 0;
        for (; i + 8 <= degree; i += 8) {
            __m256d acc_0 = _mm256_setzero_pd();
            __m256d acc_1 = _mm256_setzero_pd();

            const double* __restrict__ a_ptr = A_data + i;  // points to A(i,0)

            for (Int k = 0; k < supernode_size; ++k) {
                __m256d L_pair_0 = _mm256_loadu_pd(a_ptr);
                __m256d L_pair_1 = _mm256_loadu_pd(a_ptr + 4);
                __m256d etak = _mm256_set1_pd(srhs_ptr[k]);

                acc_0 = _mm256_fmadd_pd(L_pair_0, etak, acc_0);
                acc_1 = _mm256_fmadd_pd(L_pair_1, etak, acc_1);

                a_ptr += A_leading_dim;
            }

            __m128d hi = _mm256_extractf128_pd(acc_0, 1);
            __m128d lo = _mm256_castpd256_pd128(acc_0);

            double* dst_0 = rhs_ptr + I[i];
            double* dst_1 = rhs_ptr + I[i + 2];
            __m128d b_0 = _mm_load_pd(dst_0);
            __m128d b_1 = _mm_load_pd(dst_1);

            b_0 = _mm_sub_pd(b_0, lo);
            _mm_store_pd(dst_0, b_0);

            b_1 = _mm_sub_pd(b_1, hi);
            _mm_store_pd(dst_1, b_1);

            __m128d hi_1 = _mm256_extractf128_pd(acc_1, 1);
            __m128d lo_1 = _mm256_castpd256_pd128(acc_1);

            double* dst_0_1 = rhs_ptr + I[i + 4];
            double* dst_1_1 = rhs_ptr + I[i + 6];
            __m128d b_0_1 = _mm_load_pd(dst_0_1);
            __m128d b_1_1 = _mm_load_pd(dst_1_1);

            b_0_1 = _mm_sub_pd(b_0_1, lo_1);
            _mm_store_pd(dst_0_1, b_0_1);

            b_1_1 = _mm_sub_pd(b_1_1, hi_1);
            _mm_store_pd(dst_1_1, b_1_1);
        }
        for (; i + 4 <= degree; i += 4) {
            __m256d acc = _mm256_setzero_pd();

            const double* __restrict__ a_ptr = A_data + i;  // points to A(i,0)

            for (Int k = 0; k < supernode_size; ++k) {
                __m256d a4   = _mm256_loadu_pd(a_ptr);
                __m256d etak = _mm256_set1_pd(srhs_ptr[k]);

                acc = _mm256_fmadd_pd(a4, etak, acc);

                a_ptr += A_leading_dim;
            }

            __m128d hi = _mm256_extractf128_pd(acc, 1);
            __m128d lo = _mm256_castpd256_pd128(acc);

            double* dst_0 = rhs_ptr + I[i];
            double* dst_1 = rhs_ptr + I[i + 2];
            __m128d b_0 = _mm_load_pd(dst_0);
            __m128d b_1 = _mm_load_pd(dst_1);

            b_0 = _mm_sub_pd(b_0, lo);
            _mm_store_pd(dst_0, b_0);

            b_1 = _mm_sub_pd(b_1, hi);
            _mm_store_pd(dst_1, b_1);
        }

        if (i < degree) {
            const double* __restrict__ a_ptr = A_data + i;

            // accumulate contribution for rows i and i+1
            __m128d acc_0 = _mm_setzero_pd();
            __m128d acc_1 = _mm_setzero_pd();

            for (Int k = 0; k < supernode_size; k += 2) {
                __m128d Lpair_0 = _mm_load_pd(a_ptr); a_ptr += A_leading_dim;
                __m128d Lpair_1 = _mm_load_pd(a_ptr); a_ptr += A_leading_dim;

                __m128d eta_v_0 = _mm_set1_pd(srhs_ptr[k]);
                __m128d eta_v_1 = _mm_set1_pd(srhs_ptr[k + 1]);

                acc_0 = _mm_fmadd_pd(Lpair_0, eta_v_0, acc_0);
                acc_1 = _mm_fmadd_pd(Lpair_1, eta_v_1, acc_1);
            }

            double* dst = rhs_ptr + I[i];
            __m128d bpair = _mm_load_pd(dst);
            bpair = _mm_sub_pd(bpair, acc_0);
            bpair = _mm_sub_pd(bpair, acc_1);
            _mm_store_pd(dst, bpair);
        }
    }
};
#endif

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

#if 1 // We can't seem to do better than the Eigen-vectorized implementation with intrinsics...
template<>
struct SolveLowerTri<double, 2> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
             double * __restrict__ b) {
        // using MMap = Eigen::Map<const Eigen::MatrixXd, Eigen::Aligned16, Eigen::OuterStride<>>;
        // using VMap = Eigen::Map<Eigen::VectorXd, Eigen::Aligned16>;
        // MMap(L_data, supernode_size, supernode_size, Eigen::OuterStride<>(L_leading_dim)).triangularView<Eigen::Lower>().solveInPlace(VMap(b, supernode_size));

        const double *__restrict__ L_col_a = L_data;
        const double *__restrict__ L_col_b = L_col_a + L_leading_dim;
        const double *__restrict__ L_col_c = L_col_b + L_leading_dim;
        const double *__restrict__ L_col_d = L_col_c + L_leading_dim;
        using V2d = Vec2_T<double>;
        using  VMap = Eigen::Map<      V2d, Eigen::Aligned16>;
        using CVMap = Eigen::Map<const V2d, Eigen::Aligned16>;

#if 0
        Int j = 0;
        for (; j + 4 <= supernode_size; j += 4) {
            VMap eta(b + j);
            CVMap L_col_strip_a(L_col_a + j);

            eta[0] /= L_col_strip_a[0];
            eta[1] -= eta[0] * L_col_strip_a[1];
            eta[1] /= L_col_b[j + 1];

            VMap eta_1(b + j + 2);
            eta_1 -= CVMap(L_col_a + j + 2) * eta[0] + CVMap(L_col_b + j + 2) * eta[1];
            CVMap L_col_strip_c(L_col_c + j + 2);
            eta_1[0] /= L_col_strip_c[0];
            eta_1[1] -= eta_1[0] * L_col_strip_c[1];
            eta_1[1] /= L_col_d[j + 3];

            for (Int i = j + 4; i < supernode_size; i += 2)
                VMap(b + i) -= CVMap(L_col_a + i) * eta[0] + CVMap(L_col_b + i) * eta[1] + CVMap(L_col_c + i) * eta_1[0] + CVMap(L_col_d + i) * eta_1[1];
            L_col_a += L_leading_dim * 4;
            L_col_b += L_leading_dim * 4;
            L_col_c += L_leading_dim * 4;
            L_col_d += L_leading_dim * 4;
        }
        if (j < supernode_size) {
#else
        for (Int j = 0; j < supernode_size; j += 2) {
#endif
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
#else
template<>
struct SolveLowerTri<double, 2> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
             double * __restrict__ b) {
    const double *__restrict__ L_col_a = L_data;
    const double *__restrict__ L_col_b = L_data + L_leading_dim;

    for (Int j = 0; j < supernode_size; j += 2) {
        const double *col_a = L_col_a;
        const double *col_b = L_col_b;

        double *bj = b + j;

        double eta0 = bj[0];
        double eta1 = bj[1];

        eta0 /= col_a[j];
        eta1 -= eta0 * col_a[j + 1];
        eta1 /= col_b[j + 1];

        __m128d eta_vec = _mm_set_pd(eta1, eta0);
        _mm_store_pd(bj, eta_vec);

        __m128d eta0v = _mm_set1_pd(eta0);
        __m128d eta1v = _mm_set1_pd(eta1);

#pragma unroll 4
        for (Int i = j + 2; i < supernode_size; i += 2) {
            __m128d la = _mm_load_pd(col_a + i);
            __m128d lb = _mm_load_pd(col_b + i);

            __m128d bi = _mm_load_pd(b + i);
            bi = _mm_fnmadd_pd(la, eta0v, bi);
            bi = _mm_fnmadd_pd(lb, eta1v, bi);
            _mm_store_pd(b + i, bi);
        }

        // advance to columns j+2, j+3
        L_col_a += 2 * L_leading_dim;
        L_col_b += 2 * L_leading_dim;
    }
}
};
#endif

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

#if defined(__APPLE__)
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
#else // x86 intrinsics version is faster where available.
template<>
struct SolveLowerTriAdjoint<double, 2> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
                 double * __restrict__ b) {
        // start from last 2 columns in the supernode
        const double* __restrict__ L_col_a = L_data + L_leading_dim * (supernode_size - 2);
        const double* __restrict__ L_col_b = L_col_a + L_leading_dim;

        for (Int j = supernode_size - 2; j >= 0; j -= 2) {
            __m128d acc0 = _mm_setzero_pd();
            __m128d acc1 = _mm_setzero_pd();
            for (Int i = j + 2; i < supernode_size; i += 2) {
                __m128d b_strip = _mm_load_pd(b + i);

                __m128d la = _mm_load_pd(L_col_a + i);
                __m128d lb = _mm_load_pd(L_col_b + i);

                acc0 = _mm_fmadd_pd(la, b_strip, acc0);
                acc1 = _mm_fmadd_pd(lb, b_strip, acc1);
            }
            __m128d sums = _mm_hadd_pd(acc0, acc1); // Horizontal add hoisted out of the loop; this is a big part of how we beat Eigen.

            __m128d eta = _mm_load_pd(b + j);
            eta = _mm_sub_pd(eta, sums);

            double eta0 = _mm_cvtsd_f64(eta);
            double eta1 = _mm_cvtsd_f64(_mm_unpackhi_pd(eta, eta));

            eta1 /= L_col_b[j + 1];
            eta0 -= eta1 * L_col_a[j + 1];
            eta0 /= L_col_a[j];

            __m128d eta_out = _mm_set_pd(eta1, eta0);
            _mm_store_pd(b + j, eta_out);

            L_col_a -= 2 * L_leading_dim;
            L_col_b -= 2 * L_leading_dim;
        }
    }
};
#endif

#if defined(__APPLE__)
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
#else // x86 intrinsics version is faster where available.
template<>
struct SolveLowerTriAdjoint<double, 3> {
    static void run(const Int supernode_size,
             const double * __restrict__ L_data, const Int L_leading_dim,
                 double * __restrict__ b) {
        const double *__restrict__ L_col_a = L_data  + L_leading_dim * (supernode_size - 3);
        const double *__restrict__ L_col_b = L_col_a + L_leading_dim;
        const double *__restrict__ L_col_c = L_col_b + L_leading_dim;

        for (Int j = supernode_size - 3; j >= 0; j -= 3) {
            __m128d acc0 = _mm_setzero_pd();
            __m128d acc1 = _mm_setzero_pd();
            __m128d acc2 = _mm_setzero_pd();
            __m128d acc3 = _mm_setzero_pd();
            double acc4 = 0;

            for (Int i = j + 3; i < supernode_size; i += 3) {
                __m128d b_strip = _mm_loadu_pd(b + i);

                __m128d la = _mm_loadu_pd(L_col_a + i);
                __m128d lb = _mm_loadu_pd(L_col_b + i);
                __m128d lc = _mm_loadu_pd(L_col_c + i);

                acc0 = _mm_fmadd_pd(la, b_strip, acc0);
                acc1 = _mm_fmadd_pd(lb, b_strip, acc1);
                acc2 = _mm_fmadd_pd(lc, b_strip, acc2);

                __m128d a3v = _mm_set_pd(L_col_b[i + 2], L_col_a[i + 2]);
                __m128d s = _mm_set_pd1(b[i + 2]);
                acc3 = _mm_fmadd_pd(a3v, s, acc3);
                acc4 += L_col_c[i + 2] * b[i + 2];
            }

            __m128d etav = _mm_loadu_pd(b + j);
            __m128d sums = _mm_hadd_pd(acc0, acc1);

            etav = _mm_sub_pd(etav, sums);
            acc2 = _mm_hadd_pd(acc2, acc2);
            etav = _mm_sub_pd(etav, acc3);

            double eta0 = _mm_cvtsd_f64(etav);
            double eta1 = _mm_cvtsd_f64(_mm_unpackhi_pd(etav, etav));
            double eta2 = b[j + 2] - _mm_cvtsd_f64(acc2) - acc4;

            eta2 /= L_col_c[j + 2]; eta1 -= L_col_b[j + 2] * eta2;
            eta1 /= L_col_b[j + 1]; eta0 -= L_col_a[j + 2] * eta2 + L_col_a[j + 1] * eta1;
            eta0 /= L_col_a[j    ];

            b[j    ] = eta0;
            b[j + 1] = eta1;
            b[j + 2] = eta2;

            L_col_a -= 3 * L_leading_dim;
            L_col_b -= 3 * L_leading_dim;
            L_col_c -= 3 * L_leading_dim;
        }
    }
};
#endif

}
}
}

#endif /* end of include guard: TRS_KERNELS_HPP */
