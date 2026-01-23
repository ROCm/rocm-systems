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

#include "host.hpp"

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem_SIG_OP.hpp"
#include "envvar.hpp"
#include "host_helpers.hpp"
#include "memory/window_info.hpp"
#include "util.hpp"

#include <cassert>

namespace rocshmem {

__host__ HostContextWindowInfo::HostContextWindowInfo(MPI_Comm comm_world,
                                                      SymmetricHeap* heap) {
  window_info_ =
      new WindowInfoMPI(comm_world, heap->get_local_heap_base(), heap->get_size());
}

__host__ HostContextWindowInfo::HostContextWindowInfo(SymmetricHeap* heap) {
  window_info_ =
      new WindowInfo(heap->get_local_heap_base(), heap->get_size());
}

__host__ HostContextWindowInfo::~HostContextWindowInfo() {
  delete window_info_;
}

WindowInfo* HostInterface::acquire_window_context() {
  auto index{find_avail_pool_entry()};
  /* Entry should have been available; consider this as an error. */
  assert(index >= 0);

  HostContextWindowInfo* acquired_win_info = host_window_context_pool_[index];

  acquired_win_info->mark_unavail();

  return acquired_win_info->get();
}

__host__ void HostInterface::release_window_context(WindowInfo* window_info) {
  auto index{find_win_info_in_pool(window_info)};
  /* Entry should have been present; consider this as an error. */
  assert(index >= 0);

  host_window_context_pool_[index]->mark_avail();
}

int HostInterface::find_avail_pool_entry() {
  for (size_t i = 0; i < envvar::max_num_host_contexts; i++) {
    if (host_window_context_pool_[i]->is_avail()) {
      return i;
    }
  }
  return -1;
}

int HostInterface::find_win_info_in_pool(WindowInfo* window_info) {
  for (size_t i = 0; i < envvar::max_num_host_contexts; i++) {
    if (host_window_context_pool_[i]->is_avail()) {
      continue;
    }
    if (window_info == host_window_context_pool_[i]->get()) {
      return i;
    }
  }
  return -1;
}

__host__ HostInterface::HostInterface(HdpPolicy* hdp_policy,
                                      MPI_Comm rocshmem_comm,
                                      SymmetricHeap* heap) {
  /*
   * Duplicate a communicator from roc_shem's comm
   * world for the host interface
   */
  mpilib_ftable_.Comm_dup(rocshmem_comm, &host_comm_world_);
  mpilib_ftable_.Comm_rank(host_comm_world_, &my_pe_);
  mpilib_ftable_.Comm_size(host_comm_world_, &num_pes_);

  /*
   * Create an MPI window on the HDP so that it can be flushed
   * by remote PEs for host-facing functions
   */
  hdp_policy_ = hdp_policy;

  /*
   * Allocate and initialize pool of windows for contexts
   */
  size_t pool_size = envvar::max_num_host_contexts * sizeof(HostContextWindowInfo*);
  host_window_context_pool_ =
      reinterpret_cast<HostContextWindowInfo**>(malloc(pool_size));

  for (size_t ctx_i = 0; ctx_i < envvar::max_num_host_contexts; ctx_i++) {
    host_window_context_pool_[ctx_i] =
        new HostContextWindowInfo(host_comm_world_, heap);
  }

#if defined(USE_HDP_FLUSH) && !defined(USE_SINGLE_NODE)
  // The single node implementation needs a different path since
  // the HDP flush pointers are allocated on the symmetric heap
  // and we need to wait for other initialization to happen before
  // calling `get_hdp_flush_ptr`.
  create_hdp_window();
#endif  // defined(USE_HDP_FLUSH) && !defined(USE_SINGLE_NODE)
}

#if defined USE_HDP_FLUSH
__host__ void HostInterface::create_hdp_window() {
  mpilib_ftable_.Win_create(hdp_policy_->get_hdp_flush_ptr(),
                            sizeof(unsigned int), /* size of window */
                            sizeof(unsigned int), /* displacement */
                            MPI_INFO_NULL, host_comm_world_, &hdp_win);
  
  /*
   * Start a shared access epoch on windows of all ranks,
   * and let the library there is no need to check for
   * lock exclusivity during operations on this window
   * (MPI_MODE_NOCHECK).
   */
  mpilib_ftable_.Win_lock_all(MPI_MODE_NOCHECK, hdp_win);
}
#endif  // USE_HDP_FLUSH

__host__ HostInterface::HostInterface(HdpPolicy* hdp_policy,
                                      TcpBootstrap *bootstr,
                                      SymmetricHeap* heap) {
  host_bootstrap_ = bootstr;
  my_pe_ = bootstr->getRank();
  num_pes_ = bootstr->getNranks();

  /*
   * Not sure we need this.
   */
  hdp_policy_ = hdp_policy;

  /*
   * Allocate and initialize pool of windows for contexts
   */
  size_t pool_size = envvar::max_num_host_contexts * sizeof(HostContextWindowInfo*);
  host_window_context_pool_ =
      reinterpret_cast<HostContextWindowInfo**>(malloc(pool_size));

  for (size_t ctx_i = 0; ctx_i < envvar::max_num_host_contexts; ctx_i++) {
    host_window_context_pool_[ctx_i] =
        new HostContextWindowInfo(heap);
  }

#if defined USE_HDP_FLUSH &&  not defined USE_SINGLE_NODE
  printf("Non-mpi use-cases only supported with coherent heap at the moment. Aborting.\n");
  abort();
#endif
}

__host__ HostInterface::~HostInterface() {
#if defined USE_HDP_FLUSH
  mpilib_ftable_.Win_unlock_all(hdp_win);

  mpilib_ftable_.Win_free(&hdp_win);
#endif  // USE_HDP_FLUSH

  /* Detroy the pool of contexts */

  if (host_window_context_pool_ != nullptr) {
    for (size_t ctx_i = 0; ctx_i < envvar::max_num_host_contexts; ctx_i++) {
      delete host_window_context_pool_[ctx_i];
    }
    free(host_window_context_pool_);
  }

  if (host_comm_world_ != MPI_COMM_NULL) {
    mpilib_ftable_.Comm_free(&host_comm_world_);
  }
}

__host__ void HostInterface::putmem_nbi(void* dest, const void* source,
                                        size_t nelems, int pe,
                                        WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  initiate_put(dest, source, nelems, pe, window_info_mpi);
}

__host__ void HostInterface::getmem_nbi(void* dest, const void* source,
                                        size_t nelems, int pe,
                                        WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  initiate_get(dest, source, nelems, pe, window_info_mpi);
}

__host__ void HostInterface::putmem(void* dest, const void* source,
                                    size_t nelems, int pe,
                                    WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  initiate_put(dest, source, nelems, pe, window_info_mpi);

  mpilib_ftable_.Win_flush_local(pe, window_info_mpi->get_win());
}

__host__ void HostInterface::getmem(void* dest, const void* source,
                                    size_t nelems, int pe,
                                    WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  initiate_get(dest, source, nelems, pe, window_info_mpi);

  mpilib_ftable_.Win_flush_local(pe, window_info_mpi->get_win());

  /*
   * Flush local HDP to ensure that the NIC's write
   * of the fetched data is visible in device memory
   */
  hdp_policy_->hdp_flush();
}

__host__ void HostInterface::fence(WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  complete_all(window_info_mpi->get_win());

  /*
   * Flush my HDP and the HDPs of remote GPUs.
   * The HDP is a write-combining (WC) write-through
   * cache. But, even after the WC buffer is full and
   * the data is passed to the Data Fabric (DF), DF
   * can still reorder the writes. A flush ensures
   * that writes after the flush are written only
   * after those before the flush.
   */
  hdp_policy_->hdp_flush();
  flush_remote_hdps();

  return;
}

__host__ void HostInterface::quiet(WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  complete_all(window_info_mpi->get_win());

  /* Same explanation as in fence */
  hdp_policy_->hdp_flush();
  flush_remote_hdps();

  return;
}

__host__ void HostInterface::sync_all(WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (window_info_mpi) {
    mpilib_ftable_.Win_sync(window_info_mpi->get_win());

    hdp_policy_->hdp_flush();
    /*
     * No need to flush remote
     * HDPs here since all PEs are
     * participating.
     */

    mpilib_ftable_.Barrier(host_comm_world_);
  } else {
    hdp_policy_->hdp_flush();
    host_bootstrap_->barrier();
  }

  return;
}

__host__ void HostInterface::barrier_all(WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (window_info_mpi) {
    complete_all(window_info_mpi->get_win());

    /*
     * Flush my HDP cache so remote NICs will
     * see the latest values in device memory
     */
    hdp_policy_->hdp_flush();

    mpilib_ftable_.Barrier(host_comm_world_);
  } else {
    // Probably not required
    hdp_policy_->hdp_flush();
    host_bootstrap_->barrier();
  }

  return;
}

__host__ void HostInterface::barrier_all_on_stream(hipStream_t stream) {
  // Launch kernel to do barrier with given stream
  rocshmem_barrier_all_kernel<<<1, 1, 0, stream>>>();
}

__host__ void HostInterface::alltoallmem_on_stream(rocshmem_team_t team,
                                                   void *dest,
                                                   const void *source,
                                                   size_t size,
                                                   hipStream_t stream) {
  // Use dynamic block size determination:
  // - Query optimal block size using occupancy API
  // - Limit block size to size (number of bytes) to avoid over-subscription
  // - Always use 1 block (single workgroup collective)
  int optimal_block_size = 0;
  int grid_size = 0;
  CHECK_HIP(hipOccupancyMaxPotentialBlockSize(&grid_size, &optimal_block_size,
                                              rocshmem_alltoallmem_kernel, 0,
                                              0));

  // Limit block size to size (bytes) to avoid over-subscription
  int num_threads_per_block = (optimal_block_size > static_cast<int>(size))
                                  ? static_cast<int>(size)
                                  : optimal_block_size;

  // Launch kernel to do alltoall with given stream                                  
  dim3 gridSize(1);
  dim3 blockSize(num_threads_per_block);
  rocshmem_alltoallmem_kernel<<<gridSize, blockSize, 0, stream>>>(team, dest,
                                                                  source, size);
}

__host__ void HostInterface::broadcastmem_on_stream(rocshmem_team_t team,
                                                    void *dest,
                                                    const void *source,
                                                    size_t nelems, int pe_root,
                                                    hipStream_t stream) {
  // Use dynamic block size determination:
  // - Query optimal block size using occupancy API
  // - Limit block size to nelems (number of bytes) to avoid over-subscription
  // - Always use 1 block (single workgroup collective)
  int optimal_block_size = 0;
  int grid_size = 0;
  CHECK_HIP(hipOccupancyMaxPotentialBlockSize(&grid_size,
                                              &optimal_block_size,
                                              rocshmem_broadcastmem_kernel,
                                              0,
                                              0));

  // Limit block size to nelems (bytes) to avoid over-subscription
  int num_threads_per_block = (optimal_block_size > static_cast<int>(nelems))
                                  ? static_cast<int>(nelems)
                                  : optimal_block_size;

  // Launch kernel to do broadcast with given stream
  dim3 gridSize(1);
  dim3 blockSize(num_threads_per_block);
  rocshmem_broadcastmem_kernel<<<gridSize, blockSize, 0, stream>>>(team,
                                                                   dest,
                                                                   source,
                                                                   nelems,
                                                                   pe_root);
}

__host__ void HostInterface::getmem_on_stream(void *dest, const void *source,
                                              size_t nelems, int pe,
                                              hipStream_t stream) {
  int optimal_block_size = 0;
  int grid_size = 0;
  CHECK_HIP(hipOccupancyMaxPotentialBlockSize(&grid_size, &optimal_block_size,
                                              rocshmem_getmem_kernel, 0, 0));

  // Limit block size to nelems to avoid over-subscription
  int num_threads_per_block = (optimal_block_size > static_cast<int>(nelems))
                                  ? static_cast<int>(nelems)
                                  : optimal_block_size;

  // Launch kernel to do getmem with given stream
  dim3 gridSize(1);
  dim3 blockSize(num_threads_per_block);
  rocshmem_getmem_kernel<<<gridSize, blockSize, 0, stream>>>(dest, source,
                                                             nelems, pe);
}

__host__ void HostInterface::putmem_on_stream(void *dest, const void *source,
                                              size_t nelems, int pe,
                                              hipStream_t stream) {
  int optimal_block_size = 0;
  int grid_size = 0;
  CHECK_HIP(hipOccupancyMaxPotentialBlockSize(&grid_size, &optimal_block_size,
                                              rocshmem_putmem_kernel, 0, 0));

  // Limit block size to nelems to avoid over-subscription
  int num_threads_per_block = (optimal_block_size > static_cast<int>(nelems))
                                  ? static_cast<int>(nelems)
                                  : optimal_block_size;

  // Launch kernel to do putmem with given stream
  dim3 gridSize(1);
  dim3 blockSize(num_threads_per_block);
  rocshmem_putmem_kernel<<<gridSize, blockSize, 0, stream>>>(dest, source,
                                                             nelems, pe);
}

__host__ void HostInterface::putmem_signal_on_stream(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe, hipStream_t stream) {
  int optimal_block_size = 0;
  int grid_size = 0;
  CHECK_HIP(hipOccupancyMaxPotentialBlockSize(
      &grid_size, &optimal_block_size, rocshmem_putmem_signal_kernel, 0, 0));

  // Limit block size to nelems to avoid over-subscription
  int num_threads_per_block = (optimal_block_size > static_cast<int>(nelems))
                                  ? static_cast<int>(nelems)
                                  : optimal_block_size;

  // Launch kernel to do putmem_signal with given stream
  dim3 gridSize(1);
  dim3 blockSize(num_threads_per_block);
  rocshmem_putmem_signal_kernel<<<gridSize, blockSize, 0, stream>>>(
      dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__host__ void HostInterface::signal_wait_until_on_stream(uint64_t *sig_addr,
                                                         int cmp,
                                                         uint64_t cmp_value,
                                                         hipStream_t stream) {
  // Use a single thread to wait on the signal
  dim3 gridSize(1);
  dim3 blockSize(1);
  rocshmem_signal_wait_until_kernel<<<gridSize, blockSize, 0, stream>>>(
      sig_addr, cmp, cmp_value);
}

__host__ void HostInterface::barrier_for_sync() {
  if (host_comm_world_ != MPI_COMM_NULL) {
    mpilib_ftable_.Barrier(host_comm_world_);
  } else {
    host_bootstrap_->barrier();
  }
}

}  // namespace rocshmem
