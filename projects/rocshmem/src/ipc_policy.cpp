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

#include "ipc_policy.hpp"

#include <algorithm>
#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "memory/default_allocator.hpp"
#include "backend_bc.hpp"
#include "context_incl.hpp"
#include "envvar.hpp"
#include "util.hpp"
#include "log.hpp"
#include "memfabric/pod_detection.hpp"

namespace rocshmem {

__host__ void IpcOnImpl::ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                                     MPI_Comm thread_comm) {
  MPI_Comm shmcomm;
  std::vector<int> ipc_ranks;

  // Check if we should use pod-based detection (for VMM Fabric allocator)
  HIPAllocator *allocator = get_default_allocator();
  bool use_pod_detection = (allocator->type == AllocatorTypeVMMFabric);

  if (use_pod_detection) {
    // Use pod-based detection
    PodIds localPodIds = detectLocalPodIds();
    if (IS_PODIDS_ZERO(localPodIds)) {
      printf("Could not detect local Pod ID. Please use a different heap allocator\n");
      abort();
    }

    // Get communicator info
    int all_ranks;
    mpilib_ftable_.Comm_size(thread_comm, &all_ranks);
    int my_rank;
    mpilib_ftable_.Comm_rank(thread_comm, &my_rank);

    // AllGather pod IDs across all ranks
    std::vector<PodIds> allPodIds(all_ranks);
    mpilib_ftable_.Allgather(&localPodIds, sizeof(PodIds), MPI_CHAR,
                            allPodIds.data(), sizeof(PodIds), MPI_CHAR, thread_comm);

    // Match IPC-capable ranks
    ipc_ranks = matchIpcCapableRanks(my_rank, allPodIds);
    shm_size = ipc_ranks.size();
    shm_rank = std::find(ipc_ranks.begin(), ipc_ranks.end(), my_rank) - ipc_ranks.begin();

    // Create a group and communicator from IPC-capable ranks
    MPI_Group thread_grp, ipc_grp;
    mpilib_ftable_.Comm_group(thread_comm, &thread_grp);
    mpilib_ftable_.Group_incl(thread_grp, ipc_ranks.size(), ipc_ranks.data(), &ipc_grp);
    mpilib_ftable_.Comm_create_group(thread_comm, ipc_grp, 0, &shmcomm);
    mpilib_ftable_.Group_free(&ipc_grp);
    mpilib_ftable_.Group_free(&thread_grp);
  } else {
    // Fallback to MPI_COMM_TYPE_SHARED (original implementation)
    mpilib_ftable_.Comm_split_type(thread_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                   &shmcomm);

    /*
     * Figure out how many local process there are.
     */
    int Shm_size;
    mpilib_ftable_.Comm_size(shmcomm, &Shm_size);
    shm_size = Shm_size;

    /*
     * Figure out how this process' rank among local processes.
     */
    mpilib_ftable_.Comm_rank(shmcomm, &shm_rank);
  }

  /*
   * Allocate a host-side c-array to hold the IPC handles.
   */
  HIPIpcHandleVec *vec_ipc_handle = allocator->AllocateIpcHandleVec(shm_size);

  /*
   * Call into the hip runtime to get an IPC handle for my symmetric
   * heap and store that IPC handle into the host-side c-array which was
   * just allocated.
   */
  char *base_heap = heap_bases[my_pe];
  CHECK_HIP(allocator->GetIpcHandle(base_heap, vec_ipc_handle->GetHandleVecElem(shm_rank)));

  /*
   * Do an all-to-all exchange with each local processing element to
   * share the symmetric heap IPC handles.
   */
  size_t ipc_handle_size = allocator->GetIpcHandleSize();
  mpilib_ftable_.Allgather(MPI_IN_PLACE, ipc_handle_size, MPI_CHAR,
                           vec_ipc_handle->GetHandleVecElem(0), ipc_handle_size, MPI_CHAR, shmcomm);

  /*
   * Allocate device-side array to hold the IPC symmetric heap base
   * addresses.
   */
  char **ipc_base;
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&ipc_base),
                      shm_size * sizeof(char **)));

  /*
   * For all local processing elements, initialize the device-side array
   * with the IPC symmetric heap base addresses.
   */
  for (int i = 0; i < shm_size; i++) {
    if (i != shm_rank) {
      void **ipc_base_uncast = reinterpret_cast<void **>(&ipc_base[i]);
      CHECK_HIP(allocator->OpenIpcHandle(ipc_base_uncast,
                                         vec_ipc_handle->GetHandleVecElem(i)));
    } else {
      ipc_base[i] = base_heap;
    }
  }

  /*
   * Set member variables used by subsequent method calls.
   */
  ipc_bases = ipc_base;

  /*
   * Free the host-side memory used to exchange the symmetric heap base
   * addresses.
   */
  delete vec_ipc_handle;

  if (envvar::ro::disable_ipc || envvar::disable_ipc) {
    if (0 == my_pe) {
      LOG_WARN("ROCSHMEM_RO_DISABLE_IPC and RO_DISABLE_IPC environment variables have been deprecated."
               "Please use ROCSHMEM_DISABLE_MIXED_IPC as a replacement.");
    }
  }
  auto disable_ipc = envvar::disable_mixed_ipc || envvar::ro::disable_ipc || envvar::disable_ipc;
  if (!disable_ipc) {
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&pes_with_ipc_avail), shm_size * sizeof(int)));

    if (use_pod_detection) {
      // In pod detection path, ipc_ranks already contains global ranks
      CHECK_HIP(hipMemcpy(pes_with_ipc_avail, ipc_ranks.data(), shm_size * sizeof(int), hipMemcpyHostToDevice));
    } else {
      // In fallback path, need to translate from shmcomm ranks to thread_comm ranks
      MPI_Group thread_grp;
      MPI_Group shm_grp;
      int *host_pes_with_ipc_avail = new int[shm_size];

      mpilib_ftable_.Comm_group(thread_comm, &thread_grp);
      mpilib_ftable_.Comm_group(shmcomm, &shm_grp);
      int *seqranks = new int[shm_size];
      for(int i = 0; i < shm_size; i++)
        seqranks[i] = i;
      mpilib_ftable_.Group_translate_ranks(shm_grp, shm_size, seqranks, thread_grp, host_pes_with_ipc_avail);
      CHECK_HIP(hipMemcpy(pes_with_ipc_avail, host_pes_with_ipc_avail, shm_size * sizeof(int), hipMemcpyHostToDevice));
      // since we delete host_pes_with_ipc_avail, want to make sure the data transfer is complete
      CHECK_HIP(hipStreamSynchronize(0));

      delete [] host_pes_with_ipc_avail;
      delete [] seqranks;
      mpilib_ftable_.Group_free(&shm_grp);
      mpilib_ftable_.Group_free(&thread_grp);
    }
  }
}

__host__ void IpcOnImpl::ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                                     TcpBootstrap *bootstr) {
  // Check if we should use pod-based detection (for VMM Fabric allocator)
  HIPAllocator *allocator = get_default_allocator();
  bool use_pod_detection = (allocator->type == AllocatorTypeVMMFabric);

  // For VMM_FABRIC, use pod-based IPC capability detection; otherwise use local ranks
  auto shm_ranks = use_pod_detection ? bootstr->getIpcCapableRanks() : bootstr->getLocalRanks();
  shm_size = shm_ranks.size();
  if (shm_size == 0) {
    printf("Error in detecting IPC / shared memory rank, shm_size is 0\n");
    abort();
  }
  shm_rank = std::find(shm_ranks.begin(), shm_ranks.end(), my_pe) - shm_ranks.begin();

  /*
   * Allocate a host-side c-array to hold the IPC handles.
   */
  HIPIpcHandleVec *vec_ipc_handle = allocator->AllocateIpcHandleVec(shm_size);

  /*
   * Call into the hip runtime to get an IPC handle for my symmetric
   * heap and store that IPC handle into the host-side c-array which was
   * just allocated.
   */
  char *base_heap = heap_bases[my_pe];
  CHECK_HIP(allocator->GetIpcHandle(base_heap, vec_ipc_handle->GetHandleVecElem(shm_rank)));

  /*
   * Do an all-to-all exchange with each local processing element to
   * share the symmetric heap IPC handles.
   */
  size_t ipc_handle_size = allocator->GetIpcHandleSize();
  bootstr->groupAllGather(vec_ipc_handle->GetHandleVecElem(0), ipc_handle_size, shm_ranks);

  /*
   * Allocate device-side array to hold the IPC symmetric heap base
   * addresses.
   */
  char **ipc_base;
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&ipc_base),
                      shm_size * sizeof(char **)));

  /*
   * For all local processing elements, initialize the device-side array
   * with the IPC symmetric heap base addresses.
   */
  for (int i = 0; i < shm_size; i++) {
    if (i != shm_rank) {
      void **ipc_base_uncast = reinterpret_cast<void **>(&ipc_base[i]);
      CHECK_HIP(allocator->OpenIpcHandle(ipc_base_uncast,
                                         vec_ipc_handle->GetHandleVecElem(i)));
    } else {
      ipc_base[i] = base_heap;
    }
  }

  /*
   * Set member variables used by subsequent method calls.
   */
  ipc_bases = ipc_base;

  /*
   * Free the host-side memory used to exchange the symmetric heap base
   * addresses.
   */
  delete vec_ipc_handle;

  if (envvar::ro::disable_ipc || envvar::disable_ipc) {
    if (0 == my_pe) {
      LOG_WARN("ROCSHMEM_RO_DISABLE_IPC and RO_DISABLE_IPC environment variables have been deprecated."
               "Please use ROCSHMEM_DISABLE_MIXED_IPC as a replacement.");
    }
  }
  auto disable_ipc = envvar::disable_mixed_ipc || envvar::ro::disable_ipc || envvar::disable_ipc;
  if (!disable_ipc) {
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&pes_with_ipc_avail), shm_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(pes_with_ipc_avail, shm_ranks.data(), shm_size * sizeof(int), hipMemcpyHostToDevice));
  }
}

__host__ void IpcOnImpl::ipcHostStop() {
  HIPAllocator *allocator = get_default_allocator();

  for (int i = 0; i < shm_size; i++) {
    if (i != shm_rank) {
      CHECK_HIP(allocator->CloseIpcHandle(ipc_bases[i]));
    }
  }
  CHECK_HIP(hipFree(ipc_bases));

  if (nullptr != pes_with_ipc_avail) {
    CHECK_HIP(hipFree(pes_with_ipc_avail));
  }
}

__device__ void IpcOnImpl::ipcCopy(void *dst, void *src, size_t size) {
  memcpy_lane(dst, src, size);
}

__device__ void IpcOnImpl::ipcCopy_wave(void *dst, void *src, size_t size) {
  memcpy_wave(dst, src, size);
}

__device__ void IpcOnImpl::ipcCopy_wg(void *dst, void *src, size_t size) {
  memcpy_wg(dst, src, size);
}

}  // namespace rocshmem
