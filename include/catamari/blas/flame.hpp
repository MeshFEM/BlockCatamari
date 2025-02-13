#ifndef CATAMARI_BLAS_FLAME_H_
#define CATAMARI_BLAS_FLAME_H_

#include <complex>

#define CATAMARI_HAVE_BLAS_PROTOS
#define CATAMARI_HAVE_LAPACK_PROTOS

#define BLAS_SYMBOL(name) name

typedef int BlasInt;
typedef std::complex<float> BlasComplexFloat;
typedef std::complex<double> BlasComplexDouble;

#include <FLAME.h>

#endif  // ifndef CATAMARI_BLAS_FLAME_H_
