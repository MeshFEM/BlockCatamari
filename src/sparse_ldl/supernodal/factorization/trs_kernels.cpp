#include "catamari/sparse_ldl/supernodal/factorization/trs_kernels.hpp"

#ifdef CATAMARI_SOLVE_AVX_KERNELS
#include <immintrin.h>

namespace catamari {
namespace supernodal_ldl {
namespace trs_kernels {

void MultiplyLowerBlockAdjointEigenUnchunked<2>::run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
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

void MultiplyLowerBlockAdjointEigenUnchunked<3>::run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
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

void MultiplyLowerBlockAdjointEigenChunkedAlternate<2>::run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
             const double * __restrict__ A_data, const Int A_leading_dim,
             const Int num_rhs, double * __restrict__ B_data, const Int B_leading_dim) {
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

void MultiplyLowerBlock<double, 2>::run(const Int *I, const Int supernode_start, const Int supernode_size, const Int degree,
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

void SolveLowerTriAdjoint<double, 2>::run(const Int supernode_size,
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

void SolveLowerTriAdjoint<double, 3>::run(const Int supernode_size,
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

} // namespace trs_kernels
} // namespace supernodal_ldl
} // namespace catamari

#endif // CATAMARI_SOLVE_AVX_KERNELS
