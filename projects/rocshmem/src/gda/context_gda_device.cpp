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

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_device_functions.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "backend_gda.hpp"
#include "context_gda_device.hpp"
#include "context_gda_tmpl_device.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

__host__ GDAContext::GDAContext(Backend *b, unsigned int ctx_id, int gda_provider)
    : Context(b) {
  GDABackend *backend{static_cast<GDABackend *>(b)};
  base_heap = backend->heap.get_heap_bases().data();

  barrier_sync = backend->barrier_sync;
  wrk_sync_pool_bases_ = backend->get_wrk_sync_bases();

  CHECK_HIP(hipMalloc(&qps, sizeof(QueuePair) * num_pes));
  CHECK_HIP(hipMemset(qps, 0, sizeof(QueuePair) * num_pes));
  for (int i = 0; i < num_pes; i++) {
    int offset = num_pes * ctx_id + i;
    CHECK_HIP(hipMemcpy(&qps[i], &backend->gpu_qps[offset], sizeof(QueuePair), hipMemcpyDefault));
    qps[i].base_heap = base_heap;
  }

  ipcImpl_.ipc_bases = backend->ipcImpl.ipc_bases;
  ipcImpl_.shm_size = backend->ipcImpl.shm_size;
  ipcImpl_.shm_rank = backend->ipcImpl.shm_rank;
  ipcImpl_.pes_with_ipc_avail = backend->ipcImpl.pes_with_ipc_avail;

  ctx_id_ = ctx_id;
  gda_provider_ = gda_provider;
}

__host__ GDAContext::~GDAContext() {
  CHECK_HIP(hipFree(qps));
}

__device__ void GDAContext::ctx_create() {
}

__device__ void GDAContext::ctx_destroy(){
}

__device__ void GDAContext::putmem(void *dest, const void *source, size_t nelems,
                                   int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
  qps[pe].quiet();
}

__device__ void GDAContext::getmem(void *dest, const void *source, size_t nelems,
                                   int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  qps[pe].get_nbi(dest, base_heap[pe] + L_offset, nelems, pe);
  qps[pe].quiet();
}

__device__ void GDAContext::putmem_nbi(void *dest, const void *source,
                                       size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
}

__device__ void GDAContext::getmem_nbi(void *dest, const void *source,
                                       size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  qps[pe].get_nbi(dest, base_heap[pe] + L_offset, nelems, pe);
}

__device__ void GDAContext::fence() { //TODO: optimize
  for (int i = 0; i < num_pes; i++) {
    qps[i].quiet();
  }
  __threadfence_system();
}

__device__ void GDAContext::fence(int pe) {
  fence(); //TODO: optimize
}

__device__ void GDAContext::quiet() {
  for (int i = 0; i < num_pes; i++) {
    qps[i].quiet();
  }
}

__device__ void GDAContext::quiet_wave() {
  for (int i = 0; i < num_pes; i++) {
    qps[i].quiet(QueuePair::WAVE);
  }
}

__device__ void GDAContext::pe_quiet(size_t pe) {
  qps[pe].quiet();
}

__device__ void GDAContext::pe_quiet_single(size_t pe) {
  qps[pe].quiet_single();
}

__device__ void *GDAContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    void *dst = const_cast<void *>(dest);
    uint64_t L_offset = reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ret = ipcImpl_.ipc_bases[local_pe] + L_offset;
  }
  return ret;
}

__device__ void GDAContext::putmem_wg(void *dest, const void *source,
                                      size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  if (is_wave_zero_in_block()) {
    qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe, QueuePair::WAVE);
    qps[pe].quiet();
  }
}

__device__ void GDAContext::getmem_wg(void *dest, const void *source,
                                      size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  if (is_wave_zero_in_block()) {
    qps[pe].get_nbi(dest, base_heap[pe] + L_offset, nelems, pe, QueuePair::WAVE);
    qps[pe].quiet();
  }
}

__device__ void GDAContext::putmem_nbi_wg(void *dest, const void *source,
                                          size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  if (is_wave_zero_in_block()) {
    qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe, QueuePair::WAVE);
  }
}

__device__ void GDAContext::getmem_nbi_wg(void *dest, const void *source,
                                          size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  if (is_wave_zero_in_block()) {
    qps[pe].get_nbi(dest, base_heap[pe] + L_offset, nelems, pe, QueuePair::WAVE);
  }
}

__device__ void GDAContext::putmem_wave(void *dest, const void *source,
                                        size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe, QueuePair::WAVE);
  qps[pe].quiet();
}

__device__ void GDAContext::getmem_wave(void *dest, const void *source,
                                        size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  qps[pe].get_nbi(dest, base_heap[pe] + L_offset, nelems, pe, QueuePair::WAVE);
  qps[pe].quiet();
}

__device__ void GDAContext::putmem_nbi_wave(void *dest, const void *source,
                                            size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe, QueuePair::WAVE);
}

__device__ void GDAContext::getmem_nbi_wave(void *dest, const void *source,
                                            size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  qps[pe].get_nbi(dest, base_heap[pe] + L_offset, nelems, pe, QueuePair::WAVE);
}


//TODO: copied from IPC, needs review
__device__ void GDAContext::putmem_signal(void *dest, const void *source, size_t nelems,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op,
                                          int pe) {
  putmem(dest, source, nelems, pe);
  fence();

  switch (sig_op) {
  case ROCSHMEM_SIGNAL_SET:
    amo_set<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
    break;
  case ROCSHMEM_SIGNAL_ADD:
    amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
    break;
  default:
    DPRINTF("[%s] Invalid sig_op value (%d)\n", __func__, sig_op);
    break;
  }
  //TODO: missing quiet_pe?
}

__device__ void GDAContext::putmem_signal_wg(void *dest, const void *source, size_t nelems,
                                             uint64_t *sig_addr, uint64_t signal, int sig_op,
                                             int pe) {
  putmem_wg(dest, source, nelems, pe);
  fence();

  if (is_thread_zero_in_block()) {
    switch (sig_op) {
    case ROCSHMEM_SIGNAL_SET:
      amo_set<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    case ROCSHMEM_SIGNAL_ADD:
      amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    default:
      DPRINTF("[%s] Invalid sig_op value (%d)\n", __func__, sig_op);
      break;
    }
    //TODO: missing quiet_pe?
  }
}

__device__ void GDAContext::putmem_signal_wave(void *dest, const void *source, size_t nelems,
                                               uint64_t *sig_addr, uint64_t signal, int sig_op,
                                               int pe) {
  putmem_wave(dest, source, nelems, pe);
  fence();

  if (is_thread_zero_in_wave()) {
    switch (sig_op) {
    case ROCSHMEM_SIGNAL_SET:
      amo_set<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    case ROCSHMEM_SIGNAL_ADD:
      amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    default:
      DPRINTF("[%s] Invalid sig_op value (%d)\n", __func__, sig_op);
      break;
    }
    //TODO: missing quiet_pe?
  }
}

__device__ void GDAContext::putmem_signal_nbi(void *dest, const void *source, size_t nelems,
                                              uint64_t *sig_addr, uint64_t signal, int sig_op,
                                              int pe) {
  putmem_signal(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ void GDAContext::putmem_signal_nbi_wg(void *dest, const void *source, size_t nelems,
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                 int pe) {
  putmem_signal_wg(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ void GDAContext::putmem_signal_nbi_wave(void *dest, const void *source, size_t nelems,
                                                   uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                   int pe) {
  putmem_signal_wave(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ uint64_t GDAContext::signal_fetch(const uint64_t *sig_addr) {
  uint64_t *dst = const_cast<uint64_t*>(sig_addr);
  return amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
}

__device__ uint64_t GDAContext::signal_fetch_wg(const uint64_t *sig_addr) {
  __shared__ uint64_t value;
  if (is_thread_zero_in_block()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  return value;
}

__device__ uint64_t GDAContext::signal_fetch_wave(const uint64_t *sig_addr) {
  uint64_t value;
  if (is_thread_zero_in_wave()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  value = __shfl(value, 0);
  return value;
}

}  // namespace rocshmem
