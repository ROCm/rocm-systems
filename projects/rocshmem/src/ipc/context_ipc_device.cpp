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
#include "backend_ipc.hpp"
#include "context_ipc_device.hpp"
#include "context_ipc_tmpl_device.hpp"

namespace rocshmem {

__host__ IPCContext::IPCContext(Backend *b, unsigned int ctx_id)
    : Context(b) {
  IPCBackend *backend{static_cast<IPCBackend *>(b)};
  ipcImpl_.ipc_bases = b->ipcImpl.ipc_bases;
  ipcImpl_.shm_size = b->ipcImpl.shm_size;

  barrier_sync = backend->barrier_sync;
  fence_pool = backend->fence_pool;
  wrk_sync_pool_bases_ = backend->get_wrk_sync_bases();
  ctx_id_ = ctx_id;

  orders_.store = detail::atomic::rocshmem_memory_order::memory_order_seq_cst;
}

__device__ void IPCContext::threadfence_system() {
  __threadfence_system();
}

__device__ void IPCContext::ctx_create() {
}

__device__ void IPCContext::ctx_destroy(){
}

__device__ void IPCContext::putmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset, const_cast<void *>(source), nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::getmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[pe] + L_offset, nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::putmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  putmem(dest, source, nelems, pe);
}

__device__ void IPCContext::getmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  getmem(dest, source, nelems, pe);
}

__device__ void IPCContext::fence() {
  for (int i{0}, j{tinfo->pe_start}; i < tinfo->size; i++, j += tinfo->stride) {
    detail::atomic::store<int, detail::atomic::memory_scope_system>(&fence_pool[j], 1, orders_);
  }
}

__device__ void IPCContext::fence(int pe) {
  detail::atomic::store<int, detail::atomic::memory_scope_system>(&fence_pool[pe], 1, orders_);
}

__device__ void IPCContext::quiet() {
  fence();
}

__device__ void IPCContext::pe_quiet(size_t pe) {
  fence(pe);
}

__device__ void *IPCContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  void *dst = const_cast<void *>(dest);
  uint64_t L_offset = reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[my_pe];
  ret = ipcImpl_.ipc_bases[pe] + L_offset;
  return ret;
}

__device__ void IPCContext::putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[pe] + L_offset, const_cast<void *>(source), nelems);
  __syncthreads();
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[pe] + L_offset, nelems);
  __syncthreads();
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::putmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  putmem_wg(dest, source, nelems, pe);
}

__device__ void IPCContext::getmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  getmem_wg(dest, source, nelems, pe);
}

__device__ void IPCContext::putmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[pe] + L_offset, const_cast<void *>(source), nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::getmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[pe] + L_offset, nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::putmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  putmem_wave(dest, source, nelems, pe);
}

__device__ void IPCContext::getmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  getmem_wave(dest, source, nelems, pe);
}

__device__ void IPCContext::internal_putmem(void *dest, const void *source,
                                            size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - wrk_sync_pool_bases_[my_pe];
  memcpy_lane(wrk_sync_pool_bases_[pe] + L_offset, const_cast<void *>(source), nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::internal_getmem(void *dest, const void *source,
                                            size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - wrk_sync_pool_bases_[my_pe];
  memcpy_lane(dest, wrk_sync_pool_bases_[pe] + L_offset, nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::internal_putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - wrk_sync_pool_bases_[my_pe];
  memcpy_wg(wrk_sync_pool_bases_[pe] + L_offset, const_cast<void *>(source), nelems);
  __syncthreads();
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::internal_getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - wrk_sync_pool_bases_[my_pe];
  memcpy_wg(dest, wrk_sync_pool_bases_[pe] + L_offset, nelems);
  __syncthreads();
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::internal_putmem_wave(void *dest,
                        const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - wrk_sync_pool_bases_[my_pe];
  memcpy_wave(wrk_sync_pool_bases_[pe] + L_offset, const_cast<void *>(source), nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::internal_getmem_wave(void *dest,
                        const void *source, size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - wrk_sync_pool_bases_[my_pe];
  memcpy_wave(dest, wrk_sync_pool_bases_[pe] + L_offset, nelems);
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::putmem_signal(void *dest, const void *source, size_t nelems,
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
}

__device__ void IPCContext::putmem_signal_wg(void *dest, const void *source, size_t nelems,
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
  }
}

__device__ void IPCContext::putmem_signal_wave(void *dest, const void *source, size_t nelems,
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
  }
}

__device__ void IPCContext::putmem_signal_nbi(void *dest, const void *source, size_t nelems,
                                              uint64_t *sig_addr, uint64_t signal, int sig_op,
                                              int pe) {
  putmem_signal(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__device__ void IPCContext::putmem_signal_nbi_wg(void *dest, const void *source, size_t nelems,
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                 int pe) {
  putmem_signal_wg(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__device__ void IPCContext::putmem_signal_nbi_wave(void *dest, const void *source, size_t nelems,
                                                   uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                   int pe) {
  putmem_signal_wave(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__device__ uint64_t IPCContext::signal_fetch(const uint64_t *sig_addr) {
  uint64_t *dst = const_cast<uint64_t*>(sig_addr);
  return amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
}

__device__ uint64_t IPCContext::signal_fetch_wg(const uint64_t *sig_addr) {
  __shared__ uint64_t value;
  if (is_thread_zero_in_block()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  return value;
}

__device__ uint64_t IPCContext::signal_fetch_wave(const uint64_t *sig_addr) {
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
