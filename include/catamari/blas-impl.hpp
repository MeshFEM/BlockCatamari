/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_IMPL_H_
#define CATAMARI_BLAS_IMPL_H_

#include "catamari/blas.hpp"
#include <tbb/task_scheduler_observer.h>

#if defined(CATAMARI_HAVE_ACCELERATE) && __has_include(<vecLib/thread_api.h>)
#include <vecLib/thread_api.h>
#define CATAMARI_HAVE_ACCELERATE_THREAD_API
#endif

namespace catamari {

inline int GetMaxBlasThreads() {
#ifdef CATAMARI_HAVE_MKL
  return mkl_get_max_threads();
#elif defined(CATAMARI_HAVE_OPENBLAS)
  return openblas_get_num_threads();
#elif defined(CATAMARI_HAVE_ACCELERATE_THREAD_API)
  return BLASGetThreading() == BLAS_THREADING_SINGLE_THREADED ? 1 : 2; // Accelerate does not expose an exact thread count; indicate "multi-threaded" with 2.
#else
  return 1;
#endif  // ifdef CATAMARI_HAVE_MKL
}

inline void SetNumBlasThreads(int num_threads) {
#ifdef CATAMARI_HAVE_MKL
  mkl_set_num_threads(num_threads);
#elif defined(CATAMARI_HAVE_OPENBLAS)
  openblas_set_num_threads(num_threads);
#elif defined(CATAMARI_HAVE_ACCELERATE_THREAD_API)
  // Accelerate does not expose an exact thread count:
  //   1  -> force single-threaded
  //   >1 -> allow Accelerate to choose its own threading
  // Also note that this is a thread-local setting, and
  // global control does not appear to be available!
  //    https://developer.apple.com/documentation/accelerate/blassetthreading(_:)#:~:text=This%20setting%20is%20per%20thread%2C%20and%20Accelerate%20saves%20it%20in%20a%20thread%2Dlocal%20variable.
  BLASSetThreading(
      num_threads == 1
          ? BLAS_THREADING_SINGLE_THREADED
          : BLAS_THREADING_MULTI_THREADED);
#endif
}

inline int SetNumLocalBlasThreads(int num_threads) {
#ifdef CATAMARI_HAVE_MKL
  return mkl_set_num_threads_local(num_threads);
#elif defined(CATAMARI_HAVE_OPENBLAS)
  const int old_num_threads = openblas_get_num_threads();
  openblas_set_num_threads(num_threads);
  return old_num_threads;
#elif defined(CATAMARI_HAVE_ACCELERATE_THREAD_API)
  const int old_num_threads = GetMaxBlasThreads();
  // See discussion above. `BLASSetThreading` is appropriately thread-local.
  BLASSetThreading(
      num_threads == 1
          ? BLAS_THREADING_SINGLE_THREADED
          : BLAS_THREADING_MULTI_THREADED);
  return old_num_threads;
#else
  return 1;
#endif  // ifdef CATAMARI_HAVE_MKL
}

class BlasSingleThreadingObserver : public tbb::task_scheduler_observer {
 public:
  BlasSingleThreadingObserver() {
    old_max_threads_ = SetNumLocalBlasThreads(1);
    observe(true);
  }

  ~BlasSingleThreadingObserver() {
    observe(false);

    // Reset at least the main thread's maximum number of BLAS threads to its
    // original value (Note that, at least with Accelerate, this only is done
    // for the main thread; TBB worker threads will retain their single-threaded
    // BLAS setting, but this is probably fine for the user. Doing it also for
    // the worker threads would involve a more complicated mechanism since it
    // seems a `task_scheduler_observer::on_scheduler_exit` may not always be
    // called.)
    SetNumLocalBlasThreads(old_max_threads_);
  }

  void on_scheduler_entry(bool) override {
    SetNumLocalBlasThreads(1);
  }

 private:
  int old_max_threads_;
};

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_IMPL_H_
