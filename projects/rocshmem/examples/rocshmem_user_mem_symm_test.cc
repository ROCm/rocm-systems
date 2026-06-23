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

/*
 * Symmetric user-buffer registration example.
 *
 * This example exercises the symmetric user-buffer registration APIs:
 *
 *   rocshmem_buffer_register_symmetric(addr, length)
 *   rocshmem_buffer_unregister_symmetric(addr)
 *
 * Registering a user buffer symmetrically lets it be used as the remote target
 * of RMA and collective operations. The call is collective and must be passed
 * the same length on every PE. It returns the address to use as the symmetric
 * object for subsequent operations, which is also the address passed back to
 * rocshmem_buffer_unregister_symmetric().
 *
 * The buffer handed to the symmetric API must be HIP VMM device memory, so the
 * example skips cleanly when either prerequisite is not met:
 *   - The GPU supports HIP virtual memory management (queried at runtime via
 *     hipDeviceAttributeVirtualMemoryManagementSupported).
 *   - rocshmem_buffer_register_symmetric() returns a non-null address. A null
 *     return simply means the current rocSHMEM build/backend does not support
 *     symmetric registration yet.
 *
 * The VMM allocation below uses hipMemHandleTypePosixFileDescriptor for the
 * requested handle type, which is the common single-node configuration.
 *
 * HIP VMM symmetric heaps are not compatible with MPI-*based* rocSHMEM
 * initialization, so this example initializes via rocshmem_init_attr() over
 * the UNIQUEID path (see rocshmem_init_attr_test.cc). MPI is used only to
 * bootstrap and broadcast the unique id, not to initialize rocSHMEM.
 *
 * To run (single node):
 *   mpirun -np 2 -x ROCSHMEM_MAX_NUM_CONTEXTS=2 ./rocshmem_user_mem_symm_test
 */

#include <rocshmem/rocshmem.hpp>
#include <mpi.h>

#include <vector>

#include "util.h"

using namespace rocshmem;

__global__ void user_mem_symm_test(int *source, int *dest, size_t nelem,
                                   rocshmem_team_t team) {
  __shared__ rocshmem_ctx_t ctx;
  int64_t ctx_type = 0;

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  rocshmem_ctx_int_alltoall_wg(ctx, team, dest, source, nelem);

  rocshmem_ctx_quiet(ctx);
  __syncthreads();

  rocshmem_wg_ctx_destroy(&ctx);
}

static void init_sendbuf(int *source, int nelem, int my_pe, int npes) {
  for (int pe = 0; pe < npes; pe++) {
    for (int i = 0; i < nelem; i++) {
      int idx = (pe * nelem) + i;
      source[idx] = my_pe + pe;
    }
  }
}

static bool check_recvbuf(int *dest, int nelem, int my_pe, int npes) {
  bool res = true;

  for (int pe = 0; pe < npes; pe++) {
    for (int i = 0; i < nelem; i++) {
      int idx = (pe * nelem) + i;
      int result = my_pe + pe;
      if (dest[idx] != result) {
        res = false;
#ifdef VERBOSE
        std::cout << "recvbuf[" << i << "] = " << dest[i] << " expected "
                  << result << std::endl;
#endif
      }
    }
  }

  return res;
}

/* Query whether the GPU supports HIP virtual memory management. */
static bool device_supports_vmm(int device_id) {
  int supported = 0;
  hipError_t err = hipDeviceGetAttribute(
      &supported, hipDeviceAttributeVirtualMemoryManagementSupported,
      device_id);
  return (err == hipSuccess) && (supported != 0);
}

/*
 * Allocate HIP VMM device memory suitable for symmetric registration.
 *
 * Performs the standard VMM sequence: create a physical handle, reserve a
 * virtual address range, map the handle, and grant device + host access. The
 * memory is device-resident, RDMA-capable, and uses a POSIX-fd handle type;
 * the requested size is rounded up to the allocation granularity.
 *
 * The granularity-aligned size is returned via *aligned_size, since both the
 * symmetric registration and the later unmap must use that exact length.
 */
static bool vmm_alloc(void **ptr, hipMemGenericAllocationHandle_t *handle,
                      size_t size, size_t *aligned_size, int device_id) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_id;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.allocFlags.gpuDirectRDMACapable = 1;

  size_t granularity = 0;
  if (hipMemGetAllocationGranularity(&granularity, &prop,
                                     hipMemAllocationGranularityMinimum) !=
          hipSuccess ||
      granularity == 0) {
    return false;
  }

  size_t alloc_size = ((size + granularity - 1) / granularity) * granularity;

  if (hipMemCreate(handle, alloc_size, &prop, 0) != hipSuccess) {
    return false;
  }

  void *va = nullptr;
  if (hipMemAddressReserve(&va, alloc_size, 0, 0, 0) != hipSuccess) {
    (void)hipMemRelease(*handle);
    return false;
  }

  if (hipMemMap(va, alloc_size, 0, *handle, 0) != hipSuccess) {
    (void)hipMemAddressFree(va, alloc_size);
    (void)hipMemRelease(*handle);
    return false;
  }

  hipMemAccessDesc access_desc[2];
  access_desc[0].location.type = hipMemLocationTypeDevice;
  access_desc[0].location.id = device_id;
  access_desc[0].flags = hipMemAccessFlagsProtReadWrite;
  access_desc[1].location.type = hipMemLocationTypeHost;
  access_desc[1].location.id = 0;
  access_desc[1].flags = hipMemAccessFlagsProtReadWrite;

  if (hipMemSetAccess(va, alloc_size, access_desc, 2) != hipSuccess) {
    (void)hipMemUnmap(va, alloc_size);
    (void)hipMemAddressFree(va, alloc_size);
    (void)hipMemRelease(*handle);
    return false;
  }

  *ptr = va;
  *aligned_size = alloc_size;
  return true;
}

static void vmm_free(void *ptr, hipMemGenericAllocationHandle_t handle,
                     size_t size) {
  (void)hipMemUnmap(ptr, size);
  (void)hipMemAddressFree(ptr, size);
  (void)hipMemRelease(handle);
}

/*
 * Register dest_buf symmetrically, run an all-to-all into the returned address,
 * verify the result, and unregister. If registration is not supported for the
 * given memory, the case is reported and skipped so the example stays portable
 * across backends.
 */
static bool run_symm_variant(const char *label, void *dest_buf,
                             size_t dest_size, int *source, int nelem,
                             int my_pe, int npes, rocshmem_team_t team,
                             const char *prog) {
  int *dest = reinterpret_cast<int *>(
      rocshmem_buffer_register_symmetric(dest_buf, dest_size));
  if (dest == nullptr) {
    if (my_pe == 0) {
      std::cout << label
                << ": rocshmem_buffer_register_symmetric is not supported for "
                   "this memory in the current configuration; skipping"
                << std::endl;
    }
    return true;
  }

  size_t nbytes = static_cast<size_t>(nelem) * npes * sizeof(int);

  /* Initialize the destination to -1 (0xff bytes) before any remote writes. */
  CHECK_HIP(hipMemset(dest, 0xff, nbytes));
  rocshmem_barrier_all();

  user_mem_symm_test<<<dim3(1), dim3(256), 0, 0>>>(source, dest, nelem, team);
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<int> host(static_cast<size_t>(nelem) * npes);
  CHECK_HIP(hipMemcpy(host.data(), dest, nbytes, hipMemcpyDeviceToHost));
  bool pass = check_recvbuf(host.data(), nelem, my_pe, npes);

  std::cout << "[" << my_pe << "] Test " << prog << " (" << label
            << ") \t nelem " << nelem << " " << (pass ? "[PASS]" : "[FAIL]")
            << std::endl;

  int ret = rocshmem_buffer_unregister_symmetric(dest);
  if (ROCSHMEM_SUCCESS != ret) {
    std::cout << label << ": Error unregistering symmetric user buffer"
              << std::endl;
    rocshmem_global_exit(1);
  }

  return pass;
}

#define MAX_ELEM 256

int main(int argc, char **argv) {
  int nelem = MAX_ELEM;

  if (argc > 1) {
    nelem = atoi(argv[1]);
  }

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI_THREAD_MULTIPLE support disabled.\n";
  }

  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  CHECK_HIP(hipSetDevice(get_launcher_local_rank()));

  rocshmem_uniqueid_t uid;
  rocshmem_init_attr_t attr;
  int ret;

  if (rank == 0) {
    ret = rocshmem_get_uniqueid(&uid);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_get_uniqueid. Aborting.\n";
      MPI_Abort(MPI_COMM_WORLD, ret);
    }
  }

  MPI_Bcast(&uid, sizeof(rocshmem_uniqueid_t), MPI_BYTE, 0, MPI_COMM_WORLD);

  ret = rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  if (ret != ROCSHMEM_SUCCESS) {
    std::cout << rank
              << ": Error in rocshmem_set_attr_uniqueid_args. Aborting.\n";
    MPI_Abort(MPI_COMM_WORLD, ret);
  }

  ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
  if (ret != ROCSHMEM_SUCCESS) {
    std::cout << rank << ": Error in rocshmem_init_attr. Aborting.\n";
    MPI_Abort(MPI_COMM_WORLD, ret);
  }

  int my_pe = rocshmem_my_pe();
  int npes = rocshmem_n_pes();

  int device_id = 0;
  CHECK_HIP(hipGetDevice(&device_id));

  size_t buffer_size = nelem * sizeof(int) * npes;

  /* Shared all-to-all source, used by every variant. */
  int *source = reinterpret_cast<int *>(rocshmem_malloc(buffer_size));
  if (source == nullptr) {
    std::cout << "Error allocating source from symmetric heap" << std::endl;
    rocshmem_global_exit(1);
  }
  init_sendbuf(source, nelem, my_pe, npes);

  rocshmem_team_t team_reduce_world_dup = ROCSHMEM_TEAM_INVALID;
  rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0,
                              &team_reduce_world_dup);

  /* Variant 1: plain hipMalloc device memory. */
  int *dest_hip = nullptr;
  CHECK_HIP(hipMalloc(&dest_hip, buffer_size));
  run_symm_variant("hipMalloc", dest_hip, buffer_size, source, nelem, my_pe,
                   npes, team_reduce_world_dup, argv[0]);
  CHECK_HIP(hipFree(dest_hip));

  rocshmem_barrier_all();

  /* Variant 2: HIP VMM device memory (only attempted if the GPU supports it). */
  if (device_supports_vmm(device_id)) {
    void *dest_vmm = nullptr;
    hipMemGenericAllocationHandle_t dest_handle{};
    size_t dest_size = 0;
    if (vmm_alloc(&dest_vmm, &dest_handle, buffer_size, &dest_size,
                  device_id)) {
      run_symm_variant("VMM", dest_vmm, dest_size, source, nelem, my_pe, npes,
                       team_reduce_world_dup, argv[0]);
      vmm_free(dest_vmm, dest_handle, dest_size);
    } else if (my_pe == 0) {
      std::cout << "VMM: HIP VMM allocation unavailable; skipping" << std::endl;
    }
  } else if (my_pe == 0) {
    std::cout << "VMM: GPU does not support HIP virtual memory management; "
                 "skipping"
              << std::endl;
  }

  rocshmem_free(source);

  rocshmem_finalize();
  MPI_Finalize();
  return 0;
}
