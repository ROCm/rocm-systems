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
#include "ipc_user_buf.hpp"
#include "log.hpp"
#include "context_ipc_device.hpp"
#include "context_ipc_tmpl_device.hpp"

namespace rocshmem {

// Constant memory declared in ipc_user_buf.cpp
extern __constant__ ipc_user_buf_entry_t ipc_user_buf_table[];
extern __constant__ int                  ipc_user_buf_count;

// Resolve dest address to a remote pointer for the given PE.
// Checks user-registered buffers first (constant memory, no HBM load),
// falls back to symmetric heap bases.
__device__ static inline char* ipc_resolve_remote(void *dest, int pe,
                                                   char **heap_bases, int my_pe) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(dest);
  for (int b = 0; b < ipc_user_buf_count; b++) {
    uintptr_t base = ipc_user_buf_table[b].local_base;
    if (addr >= base && addr < base + ipc_user_buf_table[b].length) {
      uint64_t offset = addr - base;
      return reinterpret_cast<char*>(ipc_user_buf_table[b].remote_bases[pe] + offset);
    }
  }
  // Symmetric heap fallback for addresses on the rocshmem symmetric heap.
  char* my_base = heap_bases ? heap_bases[my_pe] : nullptr;
  char* pe_base = heap_bases ? heap_bases[pe] : nullptr;
  uint64_t offset = reinterpret_cast<char*>(dest) - my_base;
  return pe_base + offset;
}

__host__ IPCContext::IPCContext(Backend *b, unsigned int ctx_id)
    : Context(b) {
  IPCBackend *backend{static_cast<IPCBackend *>(b)};
  ipcImpl_.initFrom(b->ipcImpl);
  ipcImpl_.assignSdmaChannel(ctx_id);

  barrier_sync = backend->barrier_sync;
  fence_pool = backend->fence_pool;
  wrk_sync_pool_bases_ = backend->get_wrk_sync_bases();
  ctx_id_ = ctx_id;

  orders_.store = detail::atomic::rocshmem_memory_order::memory_order_seq_cst;
}

__device__ void IPCContext::putmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  char *remote = ipc_resolve_remote(dest, pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy<MemcpyKind::PutBlocking>(
      remote, const_cast<void *>(source), nelems, pe);
}

__device__ void IPCContext::getmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy<MemcpyKind::GetBlocking>(
      dest, remote, nelems, pe);
}

__device__ void IPCContext::putmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy<MemcpyKind::Put>(remote, const_cast<void *>(source), nelems, pe);
}

__device__ void IPCContext::getmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy<MemcpyKind::Get>(dest, remote, nelems, pe);
}

__device__ void IPCContext::fence() {
  ipcImpl_.ipcFence();
}

__device__ void IPCContext::fence(int pe) {
  ipcImpl_.ipcFence<detail::atomic::memory_scope_system,
                    detail::atomic::memory_order_release>(pe);
}

__device__ void IPCContext::quiet() {
  ipcImpl_.ipcQuiet();
}

__device__ void IPCContext::pe_quiet(size_t pe) {
  ipcImpl_.ipcQuiet(pe);
}

__device__ void *IPCContext::shmem_ptr(const void *dest, int pe) {
  return ipc_resolve_remote(const_cast<void*>(dest), pe, ipcImpl_.ipc_bases, my_pe);
}

__device__ void IPCContext::putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wg<MemcpyKind::PutBlocking>(
      remote, const_cast<void *>(source), nelems, pe);
  __builtin_amdgcn_s_barrier();
}

__device__ void IPCContext::getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wg<MemcpyKind::GetBlocking>(
      dest, remote, nelems, pe);
  __builtin_amdgcn_s_barrier();
}

__device__ void IPCContext::putmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wg<MemcpyKind::Put>(remote, const_cast<void *>(source), nelems, pe);
}

__device__ void IPCContext::getmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wg<MemcpyKind::Get>(dest, remote, nelems, pe);
}

__device__ void IPCContext::putmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wave<MemcpyKind::PutBlocking>(
      remote, const_cast<void *>(source), nelems, pe);
}

__device__ void IPCContext::getmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wave<MemcpyKind::GetBlocking>(
      dest, remote, nelems, pe);
}

__device__ void IPCContext::putmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wave<MemcpyKind::Put>(remote, const_cast<void *>(source), nelems, pe);
}

__device__ void IPCContext::getmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, ipcImpl_.ipc_bases, my_pe);
  ipcImpl_.ipcCopy_wave<MemcpyKind::Get>(dest, remote, nelems, pe);
}

__device__ void IPCContext::internal_putmem(void *dest, const void *source,
                                            size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, wrk_sync_pool_bases_, my_pe);
  memcpy_lane<MemcpyKind::Put>(remote, const_cast<void *>(source), nelems);
}

__device__ void IPCContext::internal_getmem(void *dest, const void *source,
                                            size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, wrk_sync_pool_bases_, my_pe);
  memcpy_lane<MemcpyKind::Get>(dest, remote, nelems);
}

__device__ void IPCContext::internal_putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, wrk_sync_pool_bases_, my_pe);
  memcpy_wg<MemcpyKind::Put>(remote, const_cast<void *>(source), nelems);
  __builtin_amdgcn_s_barrier();
}

__device__ void IPCContext::internal_getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, wrk_sync_pool_bases_, my_pe);
  memcpy_wg<MemcpyKind::Get>(dest, remote, nelems);
  __builtin_amdgcn_s_barrier();
}

__device__ void IPCContext::internal_putmem_wave(void *dest,
                        const void *source, size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(dest, pe, wrk_sync_pool_bases_, my_pe);
  memcpy_wave<MemcpyKind::Put>(remote, const_cast<void *>(source), nelems);
}

__device__ void IPCContext::internal_getmem_wave(void *dest,
                        const void *source, size_t nelems, int pe) {
  char *remote = ipc_resolve_remote(const_cast<void*>(source), pe, wrk_sync_pool_bases_, my_pe);
  memcpy_wave<MemcpyKind::Get>(dest, remote, nelems);
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
    LOGD_WARN("[%s] Invalid sig_op value (%d)", __func__, sig_op);
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
      LOGD_WARN("[%s] Invalid sig_op value (%d)", __func__, sig_op);
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
      LOGD_WARN("[%s] Invalid sig_op value (%d)", __func__, sig_op);
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
  if (is_thread_zero_in_block()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    wg_signal_scratch = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __syncthreads();
  return wg_signal_scratch;
}

__device__ uint64_t IPCContext::signal_fetch_wave(const uint64_t *sig_addr) {
  uint64_t value{0};
  if (is_thread_zero_in_wave()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  value = __shfl(value, 0);
  return value;
}

/******************************************************************************
 ******************** TILE API STUB IMPLEMENTATION ****************************
 *****************************************************************************/

__device__ int IPCContext::tile_collective_wait([[maybe_unused]] rocshmem_team_t team,
                                                [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for IPC backend");
  return ROCSHMEM_ERROR;
}

}  // namespace rocshmem
