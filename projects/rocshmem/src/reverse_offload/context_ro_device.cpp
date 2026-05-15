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

#include "context_ro_device.hpp"
#include "context_ro_tmpl_device.hpp"

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_device_functions.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "backend_type.hpp"
#include "hdp_policy.hpp"
#include "backend_proxy.hpp"
#include "backend_ro.hpp"
#include "ro_net_team.hpp"
#include "sync/abql_block_mutex.hpp"

namespace rocshmem {

__host__ ROContext::ROContext(Backend *b,
                              size_t block_id,
                              bool default_ctx)
    : Context(b),
      is_default_ctx{default_ctx} {

  ROBackend *backend{static_cast<ROBackend *>(b)};
  if (is_default_ctx) {
    block_handle = backend->default_block_handle_proxy_.get();
  } else {
    auto block_base{backend->block_handle_proxy_.get()};
    block_handle = &block_base[block_id];
  }
  ro_net_win_id = block_id % backend->ro_window_proxy_->get_num_MPI_windows();

  ipcImpl_.ipc_bases = b->ipcImpl.ipc_bases;
  ipcImpl_.shm_size = b->ipcImpl.shm_size;
  ipcImpl_.shm_rank = b->ipcImpl.shm_rank;
  ipcImpl_.pes_with_ipc_avail = b->ipcImpl.pes_with_ipc_avail;
}

__device__ void ROContext::putmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset,
                     const_cast<void *>(source), nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_PUT, dest, const_cast<void *>(source), nelems,
                        pe, 0, 0, 0, nullptr, nullptr, NULL,
                        ro_net_win_id, block_handle, true, get_status_flag(),
                        is_default_ctx);
  }
}

__device__ void ROContext::getmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_GET, dest, const_cast<void *>(source), nelems,
                        pe, 0, 0, 0, nullptr, nullptr, NULL,
                        ro_net_win_id, block_handle, true, get_status_flag(),
                        is_default_ctx);
  }
}

__device__ void ROContext::putmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset,
                     const_cast<void *>(source), nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_PUT_NBI, dest, const_cast<void *>(source),
                        nelems, pe, 0, 0, 0, nullptr, nullptr, NULL,
                        ro_net_win_id, block_handle, false);
  }
}

__device__ void ROContext::getmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_GET_NBI, dest, const_cast<void *>(source),
                        nelems, pe, 0, 0, 0, nullptr, nullptr, NULL,
                        ro_net_win_id, block_handle, false);
  }
}

__device__ void ROContext::fence() {
  build_queue_element(RO_NET_FENCE, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, NULL, ro_net_win_id, block_handle,
                      true, get_status_flag(), is_default_ctx);
}

__device__ void ROContext::fence(int pe) {
  // TODO(khamidou): need to check if per pe has any special handling
  build_queue_element(RO_NET_FENCE, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, NULL, ro_net_win_id, block_handle,
                      true, get_status_flag(), is_default_ctx);
}

__device__ void ROContext::quiet() {
  build_queue_element(RO_NET_QUIET, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, NULL, ro_net_win_id, block_handle,
                      true, get_status_flag(), is_default_ctx);
}

__device__ void ROContext::pe_quiet(size_t pe) {
  // TODO: Optimize
  quiet();
}

__device__ void *ROContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    void *dst = const_cast<void *>(dest);
    uint64_t L_offset =
        reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ret = ipcImpl_.ipc_bases[local_pe] + L_offset;
  }
  return ret;
}

__device__ void ROContext::barrier_all() {
  build_queue_element(RO_NET_BARRIER, nullptr, nullptr, 0, 0, 0, 0, 0,
                      nullptr, nullptr, NULL, ro_net_win_id,
                      block_handle, true, get_status_flag(), is_default_ctx);
}

__device__ void ROContext::barrier_all_wave() {
  if (is_thread_zero_in_wave()) {
    build_queue_element(RO_NET_BARRIER, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, NULL, ro_net_win_id,
                        block_handle, true, get_status_flag(), is_default_ctx);
  }
}

__device__ void ROContext::barrier_all_wg() {
  if (is_thread_zero_in_block()) {
    build_queue_element(RO_NET_BARRIER, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, NULL, ro_net_win_id,
                        block_handle, true, get_status_flag(), is_default_ctx);
  }
  __syncthreads();
}

__device__ void ROContext::barrier(rocshmem_team_t team) {
  ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
  build_queue_element(RO_NET_BARRIER, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, (intptr_t)team_obj->mpi_comm, ro_net_win_id, block_handle,
                      true, get_status_flag(), is_default_ctx);
}

__device__ void ROContext::barrier_wave(rocshmem_team_t team) {
  ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
  if (is_thread_zero_in_wave()) {
    build_queue_element(RO_NET_BARRIER, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                        nullptr, (intptr_t)team_obj->mpi_comm, ro_net_win_id, block_handle,
                        true, get_status_flag(), is_default_ctx);
  }
}

__device__ void ROContext::barrier_wg(rocshmem_team_t team) {
  ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
  if (is_thread_zero_in_block()) {
    build_queue_element(RO_NET_BARRIER, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                        nullptr, (intptr_t)team_obj->mpi_comm, ro_net_win_id, block_handle,
                        true, get_status_flag(), is_default_ctx);
  }
  __syncthreads();
}

__device__ void ROContext::sync_all() {
  build_queue_element(RO_NET_SYNC, nullptr, nullptr, 0, 0, 0, 0, 0,
                      nullptr, nullptr, NULL, ro_net_win_id,
                      block_handle, true, get_status_flag(), is_default_ctx);
}

__device__ void ROContext::sync_all_wave() {
  if (is_thread_zero_in_wave()) {
    build_queue_element(RO_NET_SYNC, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, NULL, ro_net_win_id,
                        block_handle, true, get_status_flag(), is_default_ctx);
  }
}

__device__ void ROContext::sync_all_wg() {
  if (is_thread_zero_in_block()) {
    build_queue_element(RO_NET_SYNC, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, NULL, ro_net_win_id,
                        block_handle, true, get_status_flag(), is_default_ctx);
  }
  __syncthreads();
}

__device__ void ROContext::sync(rocshmem_team_t team) {
  ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
  build_queue_element(RO_NET_SYNC, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, (intptr_t)team_obj->mpi_comm, ro_net_win_id, block_handle,
                      true, get_status_flag(), is_default_ctx);
}

__device__ void ROContext::sync_wave(rocshmem_team_t team) {
  ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
  if (is_thread_zero_in_wave()) {
    build_queue_element(RO_NET_SYNC, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                        nullptr, (intptr_t)team_obj->mpi_comm, ro_net_win_id, block_handle,
                        true, get_status_flag(), is_default_ctx);
  }
}

__device__ void ROContext::sync_wg(rocshmem_team_t team) {
  ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
  if (is_thread_zero_in_block()) {
    build_queue_element(RO_NET_SYNC, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                        nullptr, (intptr_t)team_obj->mpi_comm, ro_net_win_id, block_handle,
                        true, get_status_flag(), is_default_ctx);
  }
  __syncthreads();
}

__device__ void ROContext::ctx_destroy() {
  if (is_thread_zero_in_block()) {
    ROBackend *backend{static_cast<ROBackend *>(device_backend_proxy)};
    BackendProxyT &backend_proxy{backend->backend_proxy};
    auto *proxy{backend_proxy.get()};

    build_queue_element(RO_NET_FINALIZE, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, NULL, ro_net_win_id,
                        block_handle, true, get_status_flag(), is_default_ctx);

    int buffer_id = ro_net_win_id;
    backend->queue_.descriptor(buffer_id)->write_index = block_handle->write_index;

    ROStats &global_handle = proxy->profiler[buffer_id];
    global_handle.accumulateStats(block_handle->profiler);
  }

  __syncthreads();
}

__device__ void ROContext::putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset,
                        const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_PUT, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, true, get_status_flag(),
                          is_default_ctx);
    }
  }
  __syncthreads();
}

__device__ void ROContext::getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_GET, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, true, get_status_flag(),
                          is_default_ctx);
    }
  }
  __syncthreads();
}

__device__ void ROContext::putmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset,
                        const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_PUT_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
  __syncthreads();
}

__device__ void ROContext::getmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_GET_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
  __syncthreads();
}

__device__ void ROContext::putmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset,
                          const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_PUT, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, true, get_status_flag(),
                          is_default_ctx);
    }
  }
}

__device__ void ROContext::getmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset,
                          nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_GET, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, true, get_status_flag(),
                          is_default_ctx);
    }
  }
}

__device__ void ROContext::putmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset,
                          const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_PUT_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
}

__device__ void ROContext::getmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset,
                          nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_GET_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
}

__device__ void ROContext::putmem_signal(void *dest, const void *source, size_t nelems,
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

__device__ void ROContext::putmem_signal_wg(void *dest, const void *source, size_t nelems,
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

__device__ void ROContext::putmem_signal_wave(void *dest, const void *source, size_t nelems,
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

__device__ void ROContext::putmem_signal_nbi(void *dest, const void *source, size_t nelems,
                                             uint64_t *sig_addr, uint64_t signal, int sig_op,
                                             int pe) {
  putmem_signal(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__device__ void ROContext::putmem_signal_nbi_wg(void *dest, const void *source, size_t nelems,
                                                uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                int pe) {
  putmem_signal_wg(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__device__ void ROContext::putmem_signal_nbi_wave(void *dest, const void *source, size_t nelems,
                                                  uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                  int pe) {
  putmem_signal_wave(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__device__ uint64_t ROContext::signal_fetch(const uint64_t *sig_addr) {
  uint64_t *dst = const_cast<uint64_t*>(sig_addr);
  return amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
}

__device__ uint64_t ROContext::signal_fetch_wg(const uint64_t *sig_addr) {
  __shared__ uint64_t value;
  if (is_thread_zero_in_block()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  return value;
}

__device__ uint64_t ROContext::signal_fetch_wave(const uint64_t *sig_addr) {
  uint64_t value;
  if (is_thread_zero_in_wave()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  value = __shfl(value, 0);
  return value;
}

__device__ uint64_t number_active_lanes() {
  return __popcll(__ballot(1));
}

__device__ uint64_t active_logical_lane_id() {
  uint64_t ballot{__ballot(1)};
  uint64_t my_physical_lane_id{__lane_id()};
  uint64_t all_ones_mask = -1;
  uint64_t lane_mask{all_ones_mask << my_physical_lane_id};
  uint64_t inverted_mask{~lane_mask};
  uint64_t lower_active_lanes{ballot & inverted_mask};
  uint64_t my_logical_lane_id{__popcll(lower_active_lanes)};
  return my_logical_lane_id;
}

__device__ uint64_t broadcast_lds(bool lowest_active, uint64_t value) {
  constexpr size_t SIZE = 1024 / WF_SIZE;
  __shared__ uint64_t value_per_warp[SIZE];
  auto wavefront_number {get_flat_block_id() / WF_SIZE};
  if (lowest_active) {
    value_per_warp[wavefront_number] = value;
    __threadfence_block();
  }
  return value_per_warp[wavefront_number];
}

__device__ uint64_t broadcast_shfl_up(uint64_t value) {
  for (unsigned i{0}; i < WF_SIZE; i++) {
    uint64_t temp{__shfl_up(value, i)};
    if (temp) {
      value = temp;
    }
  }
  return value;
}

__device__ uint64_t broadcast(bool lowest_active, uint64_t value) {
  return broadcast_lds(lowest_active, value);
}

__device__ bool enough_space(BlockHandle *h, uint64_t required) {
  return (h->queue_size - (h->write_index - h->read_index)) >= required;
}

__device__ void acquire_lock(BlockHandle *handle) {
  while(atomicCAS((uint64_t *)&handle->lock, 0, 1) == 1) ;
}

__device__ void release_lock(BlockHandle *handle) {
  handle->lock = 0;
  __threadfence();
}

__device__ void wait_until_space_available(BlockHandle *handle, uint64_t required) {
  while (!enough_space(handle, required)) {
    refresh_volatile_dwordx2(&handle->read_index, handle->host_read_index);
  }
}

__device__ uint64_t next_write_slot_o_o_o(BlockHandle *handle) {
  uint64_t write_slot{0};
  wait_until_space_available(handle, 1);
  write_slot = handle->write_index;
  handle->write_index += 1;
  __threadfence();
  return write_slot % handle->queue_size;
}

__device__ uint64_t next_write_slot_o_o_m(BlockHandle *handle) {
  auto num_active_lanes{number_active_lanes()};
  uint64_t write_slot{0};
  auto my_active_lane_id {active_logical_lane_id()};
  bool is_lowest_active_lane {my_active_lane_id == 0};
  if (is_lowest_active_lane) {
    wait_until_space_available(handle, num_active_lanes);
    write_slot = handle->write_index;
    handle->write_index += num_active_lanes;
    __threadfence();
  }
  write_slot = broadcast(is_lowest_active_lane, write_slot);
  write_slot += my_active_lane_id;
  return write_slot % handle->queue_size;
}

__device__ uint64_t next_write_slot_o_m_o(BlockHandle *handle) {
  uint64_t write_slot{0};
  acquire_lock(handle);
  wait_until_space_available(handle, 1);
  write_slot = handle->write_index;
  handle->write_index += 1;
  __threadfence();
  release_lock(handle);
  return write_slot % handle->queue_size;
}

__device__ uint64_t next_write_slot_o_m_m(BlockHandle *handle) {
  auto num_active_lanes{number_active_lanes()};
  uint64_t write_slot{0};
  auto my_active_lane_id {active_logical_lane_id()};
  bool is_lowest_active_lane {my_active_lane_id == 0};
  if (is_lowest_active_lane) {
    acquire_lock(handle);
    wait_until_space_available(handle, num_active_lanes);
    write_slot = handle->write_index;
    handle->write_index += num_active_lanes;
    __threadfence();
    release_lock(handle);
  }
  write_slot = broadcast(is_lowest_active_lane, write_slot);
  write_slot += my_active_lane_id;
  return write_slot % handle->queue_size;
}

__device__ uint64_t next_write_slot(BlockHandle *handle) {
//  return next_write_slot_o_o_o(handle);
//  return next_write_slot_o_o_m(handle);
//  return next_write_slot_o_m_o(handle);
  return next_write_slot_o_m_m(handle);
}

__device__ void build_queue_element(
    ro_net_cmds type, void *dst, void *src, size_t size, int pe,
    int logPE_stride, int PE_size, int PE_root, void *pWrk, long *pSync,
    intptr_t team_comm, int ro_net_win_id, BlockHandle *handle,
    bool blocking, volatile char *status, bool default_ctx, ROCSHMEM_OP op,
    ro_net_types datatype) {
  auto write_slot{next_write_slot(handle)};
  auto queue_element = &handle->queue[write_slot];

  queue_element->type = type;
  queue_element->PE = pe;
  queue_element->ol1.size = size;
  queue_element->dst = dst;
  queue_element->ro_net_win_id = ro_net_win_id;
  if(blocking) {
    queue_element->status = status;
  }

  if (type == RO_NET_P) {
    memcpy(&queue_element->src, src, size);
  } else {
    queue_element->src = src;
  }

  if (type == RO_NET_AMO_FOP) {
    queue_element->op = op;
    queue_element->datatype = datatype;
  }
  if (type == RO_NET_AMO_FCAS) {
    queue_element->ol2.pWrk = pWrk;
    queue_element->datatype = datatype;
  }
  if (type == RO_NET_TEAM_REDUCE) {
    queue_element->op = op;
    queue_element->datatype = datatype;
    queue_element->team_comm = team_comm;
  }
  if (type == RO_NET_TEAM_BROADCAST) {
    queue_element->PE_root = PE_root;
    queue_element->datatype = datatype;
    queue_element->team_comm = team_comm;
  }
  if (type == RO_NET_ALLTOALL) {
    queue_element->datatype = datatype;
    queue_element->team_comm = team_comm;
    queue_element->ol2.pWrk = pWrk;
  }
  if (type == RO_NET_FCOLLECT) {
    queue_element->datatype = datatype;
    queue_element->team_comm = team_comm;
    queue_element->ol2.pWrk = pWrk;
  }
  if (type == RO_NET_SYNC) {
    queue_element->team_comm = team_comm;
  }

  // Make sure queue element data is visible to CPU
  __threadfence();

  // Make data as ready and make visible to CPU
  queue_element->notify_cpu.valid = 1;
  __threadfence();

  // Blocking requires the CPU to complete the operation.
  if (blocking) {
    int network_status{0};
    do {
      refresh_volatile_sbyte(&network_status, queue_element->status);
    } while (network_status == 0);

    *(queue_element->status) = 0;
    __threadfence();
  }
  if (default_ctx) {
    handle->default_ctx_status->enqueue(status);
  }
}

__device__ uint64_t *ROContext::get_atomic_ret_buf() {
  uint64_t *buf_addr{nullptr};
  if(is_default_ctx) {
    buf_addr = block_handle->default_ctx_atomic_ret->dequeue();
  }
  else {
    uint64_t *atomic_base_ptr{
      reinterpret_cast<uint64_t*>(block_handle->atomic_ret)};
    int thread_id{get_flat_block_id()};
    buf_addr = &atomic_base_ptr[thread_id];
  }
  return buf_addr;
}

__device__ uint64_t *ROContext::get_g_ret_buf() {
  uint64_t *buf_addr{nullptr};
  if(is_default_ctx) {
    buf_addr = block_handle->default_ctx_g_ret->dequeue();
  }
  else {
    uint64_t *g_ret{reinterpret_cast<uint64_t*>(block_handle->g_ret)};
    int thread_id{get_flat_block_id()};
    buf_addr = &g_ret[thread_id];
  }
  return buf_addr;
}

__device__ volatile char *ROContext::get_status_flag() {
  volatile char *status_addr{nullptr};
  if(is_default_ctx) {
    status_addr = block_handle->default_ctx_status->dequeue();
  }
  else {
    volatile char* status{block_handle->status};
    int thread_id{get_flat_block_id()};
    status_addr = &status[thread_id];
  }
  return status_addr;
}

}  // namespace rocshmem
