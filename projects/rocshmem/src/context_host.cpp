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

namespace rocshmem {

__host__ Context::Context(Backend* handle, bool shareable)
    : num_pes(handle->getNumPEs()),
      my_pe(handle->getMyPE()),
      fence_(shareable),
      btype(handle->type) {
}

__host__ Context::~Context() {
}

/******************************************************************************
 ********************** CONTEXT DISPATCH IMPLEMENTATIONS **********************
 *****************************************************************************/

__host__ void Context::putmem(void* dest, const void* source, size_t nelems,
                              int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_PUT);

  HOST_DISPATCH(putmem(dest, source, nelems, pe));
}

__host__ void Context::getmem(void* dest, const void* source, size_t nelems,
                              int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_GET);

  HOST_DISPATCH(getmem(dest, source, nelems, pe));
}

__host__ void Context::putmem_nbi(void* dest, const void* source, size_t nelems,
                                  int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_PUT_NBI);

  HOST_DISPATCH(putmem_nbi(dest, source, nelems, pe));
}

__host__ void Context::getmem_nbi(void* dest, const void* source, size_t nelems,
                                  int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_GET_NBI);

  HOST_DISPATCH(getmem_nbi(dest, source, nelems, pe));
}

__host__ void Context::fence() {
  ctxHostStats.incStat(NUM_HOST_FENCE);

  HOST_DISPATCH(fence());
}

__host__ void Context::quiet() {
  ctxHostStats.incStat(NUM_HOST_QUIET);

  HOST_DISPATCH(quiet());
}

__host__ void* Context::shmem_ptr(const void* dest, int pe) {
  ctxHostStats.incStat(NUM_HOST_SHMEM_PTR);

  HOST_DISPATCH_RET_PTR(shmem_ptr(dest, pe));
}

__host__ void Context::sync_all() {
  ctxHostStats.incStat(NUM_HOST_SYNC_ALL);

  HOST_DISPATCH(sync_all());
}

__host__ void Context::barrier_all() {
  ctxHostStats.incStat(NUM_HOST_BARRIER_ALL);

  HOST_DISPATCH(barrier_all());
}

__host__ void Context::barrier_all_on_stream(hipStream_t stream) {
  ctxHostStats.incStat(NUM_HOST_BARRIER_ALL);

  HOST_DISPATCH(barrier_all_on_stream(stream));
}

}  // namespace rocshmem
