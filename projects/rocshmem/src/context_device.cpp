/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "backend_bc.hpp"
#include "context_incl.hpp"
#include "util.hpp"

namespace rocshmem {

__device__ Context::Context(Backend* handle)
    : num_pes(handle->getNumPEs()),
      my_pe(handle->getMyPE()),
      btype(handle->type) {
  /*
   * Device-side context constructor is a work-group collective, so make
   * sure all the members have their default values before returning.
   *
   * Each thread is essentially initializing the same thing right over the
   * top of each other for all the default values in context.hh (and the
   * initializer list). It's not incorrect, but it is weird and probably
   * wasteful.
   *
   * TODO: Might consider refactoring so that constructor is always called
   * from a single thread, and the parallel portion of initialization can be
   * a separate function. This requires reworking all the derived classes
   * since their constructors actually make use of all the threads to boost
   * performance.
   */
  __syncthreads();
}

/******************************************************************************
 ********************** CONTEXT DISPATCH IMPLEMENTATIONS **********************
 *****************************************************************************/

__device__ void Context::threadfence_system() {
  DISPATCH(threadfence_system());
}

__device__ void Context::ctx_create() {
  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_CREATE);
  }

  DISPATCH(ctx_create());
}

__device__ void Context::ctx_destroy() {
  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_FINALIZE);
    device_backend_proxy->globalStats.accumulateStats(ctxStats);
  }

  DISPATCH(ctx_destroy());
}

__device__ void Context::putmem(void* dest, const void* source, size_t nelems,
                                int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT);

  DISPATCH(putmem(dest, source, nelems, pe));
}

__device__ void Context::getmem(void* dest, const void* source, size_t nelems,
                                int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET);

  DISPATCH(getmem(dest, source, nelems, pe));
}

__device__ void Context::putmem_nbi(void* dest, const void* source,
                                    size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI);

  DISPATCH(putmem_nbi(dest, source, nelems, pe));
}

__device__ void Context::getmem_nbi(void* dest, const void* source, size_t size,
                                    int pe) {
  if (size == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI);

  DISPATCH(getmem_nbi(dest, source, size, pe));
}

__device__ void Context::fence() {
  ctxStats.incStat(NUM_FENCE);

  DISPATCH(fence());
}

__device__ void Context::fence(int pe) {
  ctxStats.incStat(NUM_FENCE);

  DISPATCH(fence(pe));
}

__device__ void Context::quiet() {
  ctxStats.incStat(NUM_QUIET);

  DISPATCH(quiet());
}

__device__ void Context::pe_quiet(size_t pe) {
  ctxStats.incStat(NUM_PE_QUIET);

  DISPATCH(pe_quiet(pe));
}

__device__ void* Context::shmem_ptr(const void* dest, int pe) {
  ctxStats.incStat(NUM_SHMEM_PTR);

  DISPATCH_RET_PTR(shmem_ptr(dest, pe));
}

__device__ void Context::barrier_all() {
  ctxStats.incStat(NUM_BARRIER_ALL);

  DISPATCH(barrier_all());
}

__device__ void Context::barrier_all_wave() {
  ctxStats.incStat(NUM_BARRIER_ALL_WAVE);

  DISPATCH(barrier_all_wave());
}

__device__ void Context::barrier_all_wg() {
  ctxStats.incStat(NUM_BARRIER_ALL_WG);

  DISPATCH(barrier_all_wg());
}

__device__ void Context::barrier(rocshmem_team_t team) {
  ctxStats.incStat(NUM_BARRIER);

  DISPATCH(barrier(team));
}

__device__ void Context::barrier_wave(rocshmem_team_t team) {
  ctxStats.incStat(NUM_BARRIER_WAVE);

  DISPATCH(barrier_wave(team));
}

__device__ void Context::barrier_wg(rocshmem_team_t team) {
  ctxStats.incStat(NUM_BARRIER_WG);

  DISPATCH(barrier_wg(team));
}

__device__ void Context::sync_all() {
  ctxStats.incStat(NUM_SYNC_ALL);

  DISPATCH(sync_all());
}

__device__ void Context::sync_all_wave() {
  ctxStats.incStat(NUM_SYNC_ALL_WAVE);

  DISPATCH(sync_all_wave());
}

__device__ void Context::sync_all_wg() {
  ctxStats.incStat(NUM_SYNC_ALL_WG);

  DISPATCH(sync_all_wg());
}

__device__ void Context::sync(rocshmem_team_t team) {
  ctxStats.incStat(NUM_SYNC);

  DISPATCH(sync(team));
}

__device__ void Context::sync_wave(rocshmem_team_t team) {
  ctxStats.incStat(NUM_SYNC_WAVE);

  DISPATCH(sync_wave(team));
}

__device__ void Context::sync_wg(rocshmem_team_t team) {
  ctxStats.incStat(NUM_SYNC_WG);

  DISPATCH(sync_wg(team));
}

__device__ void Context::putmem_wg(void* dest, const void* source,
                                   size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_WG);

  DISPATCH(putmem_wg(dest, source, nelems, pe));
}

__device__ void Context::getmem_wg(void* dest, const void* source,
                                   size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_WG);

  DISPATCH(getmem_wg(dest, source, nelems, pe));
}

__device__ void Context::putmem_nbi_wg(void* dest, const void* source,
                                       size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI_WG);

  DISPATCH(putmem_nbi_wg(dest, source, nelems, pe));
}

__device__ void Context::getmem_nbi_wg(void* dest, const void* source,
                                       size_t size, int pe) {
  if (size == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI_WG);

  DISPATCH(getmem_nbi_wg(dest, source, size, pe));
}

__device__ void Context::putmem_wave(void* dest, const void* source,
                                     size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_WAVE);

  DISPATCH(putmem_wave(dest, source, nelems, pe));
}

__device__ void Context::getmem_wave(void* dest, const void* source,
                                     size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_WAVE);

  DISPATCH(getmem_wave(dest, source, nelems, pe));
}

__device__ void Context::putmem_nbi_wave(void* dest, const void* source,
                                         size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI_WAVE);

  DISPATCH(putmem_nbi_wave(dest, source, nelems, pe));
}

__device__ void Context::getmem_nbi_wave(void* dest, const void* source,
                                         size_t size, int pe) {
  if (size == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI_WAVE);

  DISPATCH(getmem_nbi_wave(dest, source, size, pe));
}

#define CONTEXT_PUTMEM_SIGNAL_DEF(SUFFIX, STATS_SUFFIX)                                           \
  __device__ void Context::putmem_signal##SUFFIX(void *dest, const void *source, size_t nelems,   \
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                 int pe) {                                        \
    if (nelems == 0) {                                                                            \
      return;                                                                                     \
    }                                                                                             \
                                                                                                  \
    ctxStats.incStat(NUM_PUT_SIGNAL##STATS_SUFFIX);                                               \
                                                                                                  \
    DISPATCH(putmem_signal##SUFFIX(dest, source, nelems, sig_addr, signal, sig_op, pe));          \
  }

CONTEXT_PUTMEM_SIGNAL_DEF(,)
CONTEXT_PUTMEM_SIGNAL_DEF(_wg, _WG)
CONTEXT_PUTMEM_SIGNAL_DEF(_wave, _WAVE)
CONTEXT_PUTMEM_SIGNAL_DEF(_nbi, _NBI)
CONTEXT_PUTMEM_SIGNAL_DEF(_nbi_wg, _NBI_WG)
CONTEXT_PUTMEM_SIGNAL_DEF(_nbi_wave, _NBI_WAVE)

#define CONTEXT_SIGNAL_FETCH_DEF(SUFFIX)                                    \
__device__ uint64_t Context::signal_fetch##SUFFIX(const uint64_t *sig_addr) \
{                                                                           \
    DISPATCH_RET(signal_fetch##SUFFIX(sig_addr));                           \
}

CONTEXT_SIGNAL_FETCH_DEF()
CONTEXT_SIGNAL_FETCH_DEF(_wg)
CONTEXT_SIGNAL_FETCH_DEF(_wave)

}  // namespace rocshmem
