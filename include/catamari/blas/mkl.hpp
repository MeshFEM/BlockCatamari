/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MKL_H_
#define CATAMARI_BLAS_MKL_H_

#include "catamari/complex.hpp"

#include <mkl.h>

#define CATAMARI_HAVE_BLAS_PROTOS
#define CATAMARI_HAVE_LAPACK_PROTOS

#define BLAS_SYMBOL(name) name

// TODO(Jack Poulson): Decide when to avoid enabling this function. It seems to
// be slightly slower than doing twice as much work by running Gemm then
// setting the strictly upper triangle of the result to zero.
#define CATAMARI_USE_GEMMT

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef MKL_INT BlasInt;
typedef MKL_Complex8 BlasComplexFloat;
typedef MKL_Complex16 BlasComplexDouble;

static_assert(sizeof(catamari::Complex<float>) == sizeof(MKL_Complex8),
              "catamari::Complex<float> must be ABI-compatible with MKL_Complex8");
static_assert(sizeof(catamari::Complex<double>) == sizeof(MKL_Complex16),
              "catamari::Complex<double> must be ABI-compatible with MKL_Complex16");
static_assert(alignof(catamari::Complex<float>) == alignof(MKL_Complex8),
              "catamari::Complex<float> must have the same alignment as MKL_Complex8");
static_assert(alignof(catamari::Complex<double>) == alignof(MKL_Complex16),
              "catamari::Complex<double> must have the same alignment as MKL_Complex16");

#endif  // ifndef CATAMARI_BLAS_MKL_H_
