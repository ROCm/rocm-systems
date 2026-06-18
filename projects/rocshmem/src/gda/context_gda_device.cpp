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
#include "log.hpp"
#include "context_gda_device.hpp"
#include "context_gda_tmpl_device.hpp"

namespace rocshmem {

__host__ GDAContext::GDAContext(Backend *b, unsigned int ctx_id)
    : Context(b) {
  GDABackend *backend{static_cast<GDABackend *>(b)};
  base_heap = backend->heap.get_heap_bases().data();

  barrier_sync = backend->barrier_sync;
  wrk_sync_pool_bases_ = backend->get_wrk_sync_bases();

  ctx_id_ = ctx_id;
  num_qps_per_pe = ctx_id_?
      backend->qps_per_pe_usr_ctx_ :
      backend->qps_per_pe_default_ctx_;

  num_qps = num_qps_per_pe * num_pes;

  // Calculate offset into the backend's GPU QP array
  int offset = (ctx_id_ > 0) *
    (backend->qps_per_pe_default_ctx_ +
     backend->qps_per_pe_usr_ctx_ * (ctx_id_ - 1));
  offset *= num_pes;

  CHECK_HIP(hipMalloc(&qp_counter, sizeof(uint32_t) * num_pes));
  CHECK_HIP(hipMemset(qp_counter, 0, sizeof(uint32_t) * num_pes));
  CHECK_HIP(hipMalloc(&qps, sizeof(QueuePair) * num_qps));
  CHECK_HIP(hipMemset(qps, 0, sizeof(QueuePair) * num_qps));

  CHECK_HIP(hipMemcpy(qps, &backend->gpu_qps[offset],
                      num_qps * sizeof(QueuePair),
                      hipMemcpyDefault));

  ipcImpl_.ipc_bases = backend->ipcImpl.ipc_bases;
  ipcImpl_.shm_size = backend->ipcImpl.shm_size;
  ipcImpl_.shm_rank = backend->ipcImpl.shm_rank;
  ipcImpl_.pes_with_ipc_avail = backend->ipcImpl.pes_with_ipc_avail;
  ipcImpl_.ipc_first_pe = backend->ipcImpl.ipc_first_pe;
  ipcImpl_.ipc_stride = backend->ipcImpl.ipc_stride;
}

__host__ GDAContext::~GDAContext() {
  CHECK_HIP(hipFree(qp_counter));
  CHECK_HIP(hipFree(qps));
}

__device__ void GDAContext::putmem(void *dest, const void *source, size_t nelems,
                                   int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::PutBlocking>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
  qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
  qps[qp_index].quiet(wf_info);
}

__device__ void GDAContext::getmem(void *dest, const void *source, size_t nelems,
                                   int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::GetBlocking>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
  qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
  qps[qp_index].quiet(wf_info);
}

__device__ void GDAContext::putmem_nbi(void *dest, const void *source,
                                       size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
  qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
}

__device__ void GDAContext::getmem_nbi(void *dest, const void *source,
                                       size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::Get>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
  qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
}

__device__ void GDAContext::fence() {
  /**
   * Operations issued by this context may use two backends: RDMA QPs
   * for remote PEs and the IPC fast path, when enabled, for shm-local
   * peers. The fence must order writes across both paths.
   *
   * RDMA: A single QP already orders its own traffic through in-order
   * delivery; only the multi-QP case requires an explicit per-QP quiet.
   */
  if (num_qps_per_pe > 1) {
    ActiveWFInfo wf_info(ctx_id_);
    for (uint32_t i = 0; i < num_qps; i++) {
      qps[i].quiet(wf_info);
    }
  }

  /**
   * IPC: Skip when there are no shm-local peers. Otherwise, issue a
   * system-scope release fence to ensure prior IPC writes are visible
   * to peer ranks.
   */
  if (constmem.ipc_shm_size != 0) {
    ipcImpl_.ipcFence();
  }
}

__device__ void GDAContext::fence(int pe) {
  /**
   * Operations targeting `pe` may use two backends: RDMA QPs for remote
   * PEs and the IPC fast path, when enabled, for shm-local peers. The
   * fence must order writes to `pe` across both paths.
   *
   * RDMA: A single QP per PE already orders its own traffic through
   * in-order delivery; only the multi-QP-per-PE case requires an explicit
   * quiet on each QP associated with `pe`.
   */
  if (num_qps_per_pe > 1) {
    ActiveWFInfo wf_info(ctx_id_);
    for(uint32_t i = 0; i < num_qps_per_pe; i++) {
      int qp_index = i * constmem.num_pes + pe;
      qps[qp_index].quiet(wf_info);
    }
  }

  /**
   * IPC: Skip when `pe` is not shm-local. Otherwise, issue a
   * system-scope release fence so prior IPC writes to `pe` are visible
   * to that peer rank. Passing `local_pe` lets the SDMA-enabled policy
   * quiet only the channels associated with `pe` instead of falling
   * back to sdmaQuietAll().
   */
  int local_pe;
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ipcImpl_.ipcFence(local_pe);
  }
}

__device__ void GDAContext::quiet() {
  ActiveWFInfo wf_info(ctx_id_);
  internal_quiet(wf_info);
}

__device__ void GDAContext::internal_quiet(ActiveWFInfo &wf_info) {
  for (uint32_t i = 0; i < num_qps; i++) {
    qps[i].quiet(wf_info);
  }
  ipcImpl_.ipcQuiet();
}

__device__ void GDAContext::pe_quiet(size_t pe) {
  ActiveWFInfo wf_info(ctx_id_);
  for(uint32_t i = 0; i < num_qps_per_pe; i++) {
    int qp_index = i * constmem.num_pes + pe;
    qps[qp_index].quiet(wf_info);
  }
  ipcImpl_.ipcQuiet();
}

__device__ void *GDAContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    void *dst = const_cast<void *>(dest);
    uint64_t L_offset = reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ret = ipcImpl_.ipc_bases[local_pe] + L_offset;
  }
  return ret;
}

__device__ void GDAContext::putmem_wg(void *dest, const void *source,
                                      size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::PutBlocking>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_block()) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    ActiveWFInfo wf_info(pe, ThreadScope::wg);
    int qp_index = get_qp_index(pe, wf_info);
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::getmem_wg(void *dest, const void *source,
                                      size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::GetBlocking>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_block()) {
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    ActiveWFInfo wf_info(pe, ThreadScope::wg);
    int qp_index = get_qp_index(pe, wf_info);
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::putmem_nbi_wg(void *dest, const void *source,
                                          size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_block()) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    ActiveWFInfo wf_info(pe, ThreadScope::wg);
    int qp_index = get_qp_index(pe, wf_info);
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
  }
}

__device__ void GDAContext::getmem_nbi_wg(void *dest, const void *source,
                                          size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::Get>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_block()) {
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    ActiveWFInfo wf_info(pe, ThreadScope::wg);
    int qp_index = get_qp_index(pe, wf_info);
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
  }
}

__device__ void GDAContext::putmem_wave(void *dest, const void *source,
                                        size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::PutBlocking>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    ActiveWFInfo wf_info(pe, ThreadScope::wave);
    int qp_index = get_qp_index(pe, wf_info);
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::getmem_wave(void *dest, const void *source,
                                        size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::GetBlocking>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    ActiveWFInfo wf_info(pe, ThreadScope::wave);
    int qp_index = get_qp_index(pe, wf_info);
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::putmem_nbi_wave(void *dest, const void *source,
                                            size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    ActiveWFInfo wf_info(pe, ThreadScope::wave);
    int qp_index = get_qp_index(pe, wf_info);
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
  }
}

__device__ void GDAContext::getmem_nbi_wave(void *dest, const void *source,
                                            size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::Get>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    ActiveWFInfo wf_info(pe, ThreadScope::wave);
    int qp_index = get_qp_index(pe, wf_info);
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
  }
}


//TODO: copied from IPC, needs review
__device__ void GDAContext::putmem_signal(void *dest, const void *source,
    size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  internal_putmem(dest, source, nelems, pe, qp_index, wf_info);
  qps[qp_index].quiet(wf_info);

  switch (sig_op) {
  case ROCSHMEM_SIGNAL_SET:
    internal_amo_swap<uint64_t>(static_cast<void*>(sig_addr), signal, pe,
      qp_index, wf_info);
    break;
  case ROCSHMEM_SIGNAL_ADD:
    internal_amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe,
      qp_index, wf_info);
    break;
  default:
    LOGD_WARN("[%s] Invalid sig_op value (%d)", __func__, sig_op);
    break;
  }
  //TODO: missing quiet_pe?
}

__device__ void GDAContext::putmem_signal_wg(void *dest, const void *source,
      size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  if (is_thread_zero_in_block()) {
    ActiveWFInfo wf_info(pe, ThreadScope::wg);
    int qp_index = get_qp_index(pe, wf_info);
    internal_putmem_wg(dest, source, nelems, pe, qp_index, wf_info);
    qps[qp_index].quiet(wf_info);
    switch (sig_op) {
    case ROCSHMEM_SIGNAL_SET:
      internal_amo_swap<uint64_t>(static_cast<void*>(sig_addr), signal, pe,
        qp_index, wf_info);
      break;
    case ROCSHMEM_SIGNAL_ADD:
      internal_amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe,
        qp_index, wf_info);
      break;
    default:
      LOGD_WARN("[%s] Invalid sig_op value (%d)", __func__, sig_op);
      break;
    }
    //TODO: missing quiet_pe?
  }
}

__device__ void GDAContext::putmem_signal_wave(void *dest, const void *source,
      size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  if (is_thread_zero_in_wave()) {
    ActiveWFInfo wf_info(pe, ThreadScope::wave);
    int qp_index = get_qp_index(pe, wf_info);
    internal_putmem_wave(dest, source, nelems, pe, qp_index, wf_info);
    qps[qp_index].quiet(wf_info);
    switch (sig_op) {
    case ROCSHMEM_SIGNAL_SET:
      internal_amo_swap<uint64_t>(static_cast<void*>(sig_addr), signal, pe,
        qp_index, wf_info);
      break;
    case ROCSHMEM_SIGNAL_ADD:
      internal_amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe,
        qp_index, wf_info);
      break;
    default:
      LOGD_WARN("[%s] Invalid sig_op value (%d)", __func__, sig_op);
      break;
    }
    //TODO: missing quiet_pe?
  }
}

__device__ void GDAContext::putmem_signal_nbi(void *dest, const void *source,
    size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  putmem_signal(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ void GDAContext::putmem_signal_nbi_wg(void *dest,
    const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,
    int sig_op, int pe) {
  putmem_signal_wg(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ void GDAContext::putmem_signal_nbi_wave(void *dest,
    const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,
    int sig_op, int pe) {
  putmem_signal_wave(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ uint64_t GDAContext::signal_fetch(const uint64_t *sig_addr) {
  ActiveWFInfo wf_info(constmem.my_pe);
  int qp_index = get_qp_index(constmem.my_pe, wf_info);
  uint64_t *dst = const_cast<uint64_t*>(sig_addr);
  return internal_amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, constmem.my_pe,
           qp_index, wf_info);
}

__device__ uint64_t GDAContext::signal_fetch_wg(const uint64_t *sig_addr) {
  if (is_thread_zero_in_block()) {
    ActiveWFInfo wf_info(constmem.my_pe, ThreadScope::wg);
    int qp_index = get_qp_index(constmem.my_pe, wf_info);
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    wg_signal_scratch = internal_amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0,
                                                         constmem.my_pe, qp_index, wf_info);
  }
  __syncthreads();
  return wg_signal_scratch;
}

__device__ uint64_t GDAContext::signal_fetch_wave(const uint64_t *sig_addr) {
  uint64_t value = 0;
  if (is_thread_zero_in_wave()) {
    ActiveWFInfo wf_info(constmem.my_pe, ThreadScope::wave);
    int qp_index = get_qp_index(constmem.my_pe, wf_info);
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = internal_amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, constmem.my_pe,
              qp_index, wf_info);
  }
  __threadfence_block();
  value = __shfl(value, 0);
  return value;
}

// internal functions used by collective operations
__device__ void GDAContext::internal_putmem(void *dest, const void *source, size_t nelems,
    int pe, int qp_index, ActiveWFInfo &wf_info) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::PutBlocking>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
  qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
  qps[qp_index].quiet(wf_info);
}

__device__ void GDAContext::internal_getmem(void *dest, const void *source, size_t nelems,
    int pe, int qp_index, ActiveWFInfo &wf_info) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::GetBlocking>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
  qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
  qps[qp_index].quiet(wf_info);
}

__device__ void GDAContext::internal_putmem_wg(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::PutBlocking>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_block()) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::internal_getmem_wg(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::GetBlocking>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_wave_zero_in_block()) {
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::internal_putmem_wave(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::PutBlocking>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::internal_getmem_wave(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::GetBlocking>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
    qps[qp_index].quiet(wf_info);
  }
}

__device__ void GDAContext::internal_putmem_nbi(void *dest, const void *source, size_t nelems,
    int pe, int qp_index, ActiveWFInfo &wf_info) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
  qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
}

__device__ void GDAContext::internal_getmem_nbi(void *dest, const void *source, size_t nelems,
    int pe, int qp_index, ActiveWFInfo &wf_info) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy<MemcpyKind::Get>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
  qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
}

__device__ void GDAContext::internal_putmem_nbi_wg(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_wave_zero_in_block()) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
  }
}

__device__ void GDAContext::internal_getmem_nbi_wg(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::Get>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_wave_zero_in_block()) {
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
  }
}

__device__ void GDAContext::internal_putmem_nbi_wave(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, const_cast<void *>(source), nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[constmem.my_pe];
    qps[qp_index].put_nbi(base_heap[pe] + L_offset, source, nelems, wf_info);
  }
}

__device__ void GDAContext::internal_getmem_nbi_wave(void *dest, const void *source,
    size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    uint64_t L_offset = const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave<MemcpyKind::Get>(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems, local_pe);
    return;
  }
  if (is_thread_zero_in_wave()) {
    uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[constmem.my_pe];
    qps[qp_index].get_nbi(dest, base_heap[pe] + L_offset, nelems, wf_info);
  }
}

/******************************************************************************
 **************** TILE API STUB IMPLEMENTATION (NOT IMPLEMENTED) **************
 *****************************************************************************/

__device__ int GDAContext::tile_collective_wait([[maybe_unused]] rocshmem_team_t team,
                                                 [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

}  // namespace rocshmem
