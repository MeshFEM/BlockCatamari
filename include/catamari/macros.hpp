/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_MACROS_H_
#define CATAMARI_MACROS_H_

#include "quotient/macros.hpp"
#include "sparse_ldl/supernodal/catamari_config.hh"

// Note: for universal builds on macOS, the `CATAMARI_HAVE_XMMINTRIN` flag
// is passed independently of which architecture is currently being compiled.
// We therefore check the architecture here to ensure `xmmintrin.h` is only
// included when compiling for x86_64 or i386.
#if defined(CATAMARI_HAVE_XMMINTRIN) && !(defined(__x86_64__) || defined(__i386__))
#undef CATAMARI_HAVE_XMMINTRIN
#endif

// The same goes for AVX kernels, which build only for x86_64 or i386 architectures.
#if defined(CATAMARI_SOLVE_AVX_KERNELS) && !(defined(__x86_64__) || defined(__i386__))
#warning "CATAMARI_SOLVE_AVX_KERNELS is defined, but the current architecture is not x86_64 or i386. AVX kernels will not be used."
#undef CATAMARI_SOLVE_AVX_KERNELS
#endif

#define CATAMARI_ASSERT(condition, msg) QUOTIENT_ASSERT(condition, msg)

#define CATAMARI_NOEXCEPT QUOTIENT_NOEXCEPT

#define CATAMARI_UNUSED QUOTIENT_UNUSED

#if defined(_MSC_VER)
#define CATAMARI_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define CATAMARI_RESTRICT __restrict__
#else
#define CATAMARI_RESTRICT
#endif

#ifdef CATAMARI_ENABLE_TIMERS
#define CATAMARI_START_TIMER(timer) timer.Start()
#define CATAMARI_STOP_TIMER(timer) timer.Stop()
#else
#define CATAMARI_START_TIMER(timer)
#define CATAMARI_STOP_TIMER(timer)
#endif  // ifdef CATAMARI_ENABLE_TIMERS

#endif  // ifndef CATAMARI_MACROS_H_
