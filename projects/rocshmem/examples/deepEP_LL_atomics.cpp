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
#include <rocshmem/rocshmem.hpp>
#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include "util.h"

static void init_pmix(int *rank, int *nranks);
static void pmix_finalize();
static void pmix_bcast(void *buf, size_t nbytes, char *key, int root);

using namespace rocshmem;

#define NUM_TIMEOUT_CYCLES 20000000000ll // 200G cycles ~= 10s

template <typename T>
__host__ __device__ T cell_div(T a, T b) {
  return (a + b - 1) / b;
}

void print_usage(const char* prog_name)
{
  std::cerr
    << "Usage: " << prog_name << " [options]\n"
    << "Options:\n"
    << "  -u                 Print this usage message and exit\n"
    << "  -e <num_experts>   Number of experts (default: 288)\n"
    << "  -i <iterations>    Number of iterations (default: 10)\n";
}

struct atomics_buffer {
  int num_elems {0};
  int64_t* atomic_buffer_ptr {nullptr};

  int64_t* atomic_buffers[2] {nullptr, nullptr};

  int buffer_idx {0};

  atomics_buffer(int num_elems_, int64_t* atomic_buffer_ptr_)
      : num_elems(num_elems_), atomic_buffer_ptr(atomic_buffer_ptr_) {
    atomic_buffers[0] = atomic_buffer_ptr;
    atomic_buffers[1] = atomic_buffer_ptr + num_elems;
    // print_debug_info();
  }

  void print_debug_info() {
    std::cout << "num_elems: " << num_elems
              << ", atomic_buffer_ptr: " << atomic_buffer_ptr
              << ", atomic_buffers[0]: " << atomic_buffers[0]
              << ", atomic_buffers[1]: " << atomic_buffers[1]
              << ", buffer_idx: " << buffer_idx
              << ", ptr difference: " << atomic_buffers[1] - atomic_buffers[0]
              << std::endl;
  }
};

size_t get_atomic_buffer_size(int num_elems) {
  return num_elems * sizeof(int64_t) * 2;
}

/**
* Grid barrier implementation using a global counter.
* All the work-groups must be co-resident on the GPU for this to work
* correctly.
*/
__forceinline__ __device__ void grid_barrier(int* global_counter,
    int num_wgs) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(&global_counter[0], 1,
                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(global_counter,
                            __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT) != num_wgs);
  }
  __syncthreads();
}

/**
* kernel_1:
* - Atomics pattern of deepEP_LL dispatch kernel
* - Each work group is responsible for some destination experts
*/
template <int kNumWavesPerGroup, int kNumWaveGroups>
__global__ void kernel_1(int64_t* atomic_buffer, int num_experts,
    int64_t* next_atomic_buffer, int num_pes, int pe, int* grid_sync_buffer,
    int iter) {
  const int wg_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int wave_id = static_cast<int>(thread_id / warpSize);
  const int num_wgs = static_cast<int>(gridDim.x);
  const int lane_id = static_cast<int>(thread_id % warpSize);
  constexpr int num_waves = kNumWaveGroups * kNumWavesPerGroup;
  const int num_local_experts = num_experts / num_pes;
  const int wave_group_id = wave_id / kNumWavesPerGroup;
  const int sub_wave_id = wave_id % kNumWavesPerGroup;
  const int responsible_expert_id = wg_id * kNumWaveGroups + wave_group_id;

  // unique value for each expert (unique to each PE)
  const int unique_value = pe * num_experts + responsible_expert_id;

  // clear next_atomic_buffer
  if (wg_id == 0 && wave_id == 0) {
    for (int i = lane_id; i < num_experts; i += warpSize) {
      next_atomic_buffer[i] = 0;
    }
  }

  // // Issue atomics using unique value
  if (responsible_expert_id < num_experts && sub_wave_id == 0 && lane_id == 0) {
    const int dst_pe = responsible_expert_id / num_local_experts;
    const int dst_expert_local_idx = responsible_expert_id % num_local_experts;
    // atomic buffer dimension: [num_local_experts][num_pes]
    int64_t* const dst_ptr = atomic_buffer + dst_expert_local_idx *
                                   num_pes + pe;
    if (dst_pe != pe) {
      rocshmem_long_atomic_add(dst_ptr, -unique_value - 1, dst_pe);
      rocshmem_quiet();
    } else {
      __hip_atomic_store(dst_ptr, -unique_value - 1, __ATOMIC_RELEASE,
        __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  // Grid barrier
  // grid_barrier(grid_sync_buffer, num_wgs);

  if (responsible_expert_id < num_experts) {
    const int src_pe = responsible_expert_id / num_local_experts;
    const int src_expert_local_idx = responsible_expert_id % num_local_experts;
    int64_t* const src_ptr = atomic_buffer + src_expert_local_idx *
                                   num_pes + src_pe;
    // expected unique_value for this slot: owner (src_pe) uses
    // responsible_expert_id in the sender's side = pe * num_local_experts + src_expert_local_idx
    // expected_value = src_pe * num_experts + pe * num_local_experts + src_expert_local_idx
    int expected_value = src_pe * num_experts + pe * num_local_experts +
                         src_expert_local_idx;
    expected_value = -expected_value - 1;

    // Wait until the destination ptr is updated
    if (sub_wave_id == 0 && lane_id == 0) {
      long long int start_time = wall_clock64();
      while (__hip_atomic_load(src_ptr, __ATOMIC_ACQUIRE,
             __HIP_MEMORY_SCOPE_AGENT) != expected_value) {
        if (wall_clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          // Print the information of the PE, workgroup, and wavegroup responsible for this slot
          int src_responsible_expert_id = pe * num_local_experts + src_expert_local_idx;
          int src_wg_id = src_responsible_expert_id / kNumWaveGroups;
          int src_wave_group_id = src_responsible_expert_id % kNumWaveGroups;
          printf("[%s] PE: %02d, Iter: %d, Timeout waiting for destination ptr to be updated by: "
                 "src {PE: %d, WG ID: %d, Wave Group ID: %d, Responsible Expert ID: %d}, "
                 "Expected Value: %d, Actual Value: %ld\n",
                 __func__, pe, iter, src_pe, src_wg_id, src_wave_group_id,
                 src_responsible_expert_id, expected_value, *src_ptr);
          // reset the start time
          start_time = wall_clock64();
        }
      }
    }
    // Synchronize sub-warps in the warp group
    __syncthreads();
  }
}

template <int kNumWavesPerGroup, int kNumWaveGroups>
__global__ void kernel_2(int64_t* atomic_buffer, int num_experts,
    int64_t* next_atomic_buffer, int num_pes, int pe, int* grid_sync_buffer,
    int iter) {
  const int wg_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int wave_id = static_cast<int>(thread_id / warpSize);
  const int num_wgs = static_cast<int>(gridDim.x);
  const int num_threads = static_cast<int>(blockDim.x);
  const int lane_id = static_cast<int>(thread_id % warpSize);
  constexpr int num_waves = kNumWaveGroups * kNumWavesPerGroup;
  const int num_local_experts = num_experts / num_pes;
  const int wave_group_id = wave_id / kNumWavesPerGroup;
  const int sub_wave_id = wave_id % kNumWavesPerGroup;
  const int responsible_expert_id = wg_id * kNumWaveGroups + wave_group_id;

  // unique value for each expert (unique to each PE)
  const int unique_value = pe * num_experts + responsible_expert_id;

  // clear next_atomic_buffer
  if (wg_id == 0 && wave_id == 0) {
    for (int i = lane_id; i < num_experts; i += warpSize) {
      next_atomic_buffer[i] = 0;
    }
  }

  if (responsible_expert_id < num_experts && sub_wave_id == 0 && lane_id == 0) {
    const int dst_pe = responsible_expert_id / num_local_experts;
    const int dst_expert_local_idx = responsible_expert_id % num_local_experts;
    const int global_expert_idx = pe * num_local_experts + dst_expert_local_idx;
    // atomic buffer dimension: [num_experts]
    int64_t* const dst_ptr = atomic_buffer + global_expert_idx;

    if (dst_pe != pe) {
      rocshmem_long_atomic_add(dst_ptr, -unique_value - 1, dst_pe);
      rocshmem_quiet();
    } else {
      __hip_atomic_store(dst_ptr, -unique_value - 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  // Wait until the destination ptr is updated
  if (responsible_expert_id < num_experts && sub_wave_id == 0 && lane_id == 0) {
    int64_t* const dst_ptr = atomic_buffer + responsible_expert_id;
    const int src_expert_local_idx = responsible_expert_id % num_local_experts;
    // expected unique_value for this slot: owner (src_pe) uses
    // responsible_expert_id in the sender's side = pe * num_local_experts + src_expert_local_idx
    // expected_value = src_pe * num_experts + pe * num_local_experts + src_expert_local_idx
    const int src_pe = responsible_expert_id / num_local_experts;
    int expected_value = src_pe * num_experts + pe * num_local_experts +
                         src_expert_local_idx;
    expected_value = -expected_value - 1;
    long long int start_time = wall_clock64();
    while (__hip_atomic_load(dst_ptr, __ATOMIC_ACQUIRE,
           __HIP_MEMORY_SCOPE_AGENT) != expected_value) {
      if (wall_clock64() - start_time > NUM_TIMEOUT_CYCLES) {
        // Print the information of the PE, workgroup, and wavegroup responsible for this slot
        int src_responsible_expert_id = pe * num_local_experts + src_expert_local_idx;
        int src_wg_id = src_responsible_expert_id / kNumWaveGroups;
        int src_wave_group_id = src_responsible_expert_id % kNumWaveGroups;
        printf("[%s] PE: %02d, Iter: %d, Timeout waiting for destination ptr to be updated by: "
               "src {PE: %d, WG ID: %d, Wave Group ID: %d, Responsible Expert ID: %d}, "
               "Expected Value: %d, Actual Value: %ld\n",
               __func__, pe, iter, src_pe, wg_id, wave_group_id, responsible_expert_id,
               expected_value, *dst_ptr);
        // reset the start time
        start_time = wall_clock64();
      }
    }
  }

  // Synchronize sub-warps in the warp group
  __syncthreads();
}

int main (int argc, char **argv) {
  int num_experts {16};
  int num_iterations {1};

  int opt {0};
  while ((opt = getopt(argc, argv, "e:i:u")) != -1) {
    switch (opt) {
      case 'e':
        num_experts = atoi(optarg);
        break;
      case 'i':
        num_iterations = atoi(optarg);
        break;
      case 'u':
        print_usage(argv[0]);
        return EXIT_SUCCESS;
      case '?':
        if (optopt == 'e' || optopt == 'i') {
          std::cerr << "Option -" << static_cast<char>(optopt)
                    << " requires an argument." << std::endl;
        } else {
          std::cerr << "Unknown option -"
                    << static_cast<char>(optopt) << std::endl;
        }
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
  }
  CHECK_HIP(hipSetDevice(get_launcher_local_rank()));

#ifdef HAVE_PMIX
  int test_uuid = 0;
  char *rocshmem_test_uuid = getenv("ROCSHMEM_TEST_UUID");
  if (rocshmem_test_uuid != nullptr) {
    test_uuid = atoi(rocshmem_test_uuid);
  }

  if (test_uuid) {
    int ret;
    int rank, nranks;
    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;

    init_pmix(&rank, &nranks);
    if (rank == 0) {
      ret = rocshmem_get_uniqueid (&uid);
      if (ret != ROCSHMEM_SUCCESS) {
        std::cout << rank << ": Error in rocshmem_get_uniqueid. Aborting.\n";
        abort();
      }
    }

    char key[] = "rocshmem-uuid";
    pmix_bcast(&uid, sizeof(rocshmem_uniqueid_t), key, 0);

    // Close PMIx before potentially doing MPI_Init inside rocshmem_init
    pmix_finalize();

    ret = rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_set_attr_uniqueid_args. Aborting.\n";
      abort();
    }

    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_init_attr. Aborting.\n";
      abort();
    }
  } else //intentional spillover
#endif
  rocshmem_init();

  int my_pe {rocshmem_my_pe()};
  int n_pes {rocshmem_n_pes()};
  int device_id {-1};

  CHECK_HIP(hipGetDevice(&device_id));

  // Create HIP stream after setting the device
  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  // Get the device properties
  hipDeviceProp_t device_prop;
  CHECK_HIP(hipGetDeviceProperties(&device_prop, device_id));
  // Wavefront size
  const int wf_size = device_prop.warpSize;
  std::cout << "my_pe: " << my_pe
            << ", npes: " << n_pes
            << ", num_experts: " << num_experts
            << ", num_iterations: " << num_iterations
            << ", get_launcher_local_rank(): " << get_launcher_local_rank()
            << ", device: " << device_id
            << std::endl;

  CHECK_HIP(hipDeviceSynchronize());

  size_t atomic_buffer_size = get_atomic_buffer_size(num_experts);
  void* atomic_buffer_ptr {nullptr};

  // Allocate rocSHMEM buffer
  atomic_buffer_ptr = rocshmem_malloc(atomic_buffer_size);
  if (atomic_buffer_ptr == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    rocshmem_finalize();
    return EXIT_FAILURE;
  }

  atomics_buffer amo_buffers(num_experts, reinterpret_cast<int64_t*>(atomic_buffer_ptr));
  int* grid_sync_buffer {nullptr};
  CHECK_HIP(hipMalloc(&grid_sync_buffer, sizeof(int)));

  CHECK_HIP(hipMemsetAsync(atomic_buffer_ptr, 0, atomic_buffer_size, stream));

  for (int iter = 0; iter < num_iterations; iter++) {
    constexpr int kNumWavesPerGroup = 4;
    constexpr int kNumWaveGroups = 4;

    const auto num_waves   = kNumWaveGroups * kNumWavesPerGroup;
    const auto num_wgs     = cell_div(num_experts, kNumWaveGroups);
    const auto num_threads = num_waves * wf_size;

    dim3 grid(num_wgs);
    dim3 block(num_threads);

    // print debug info
    // std::cout << "iter: " << iter
    //           << ", grid: {" << grid.x << ", " << grid.y << ", " << grid.z << "}"
    //           << ", block: {" << block.x << ", " << block.y << ", " << block.z << "}"
    //           << ", kNumWavesPerGroup: " << kNumWavesPerGroup
    //           << ", kNumWaveGroups: " << kNumWaveGroups
    //           << ", kWaveSize: " << kWaveSize
    //           << ", num_waves: " << num_waves
    //           << ", num_wgs: " << num_wgs
    //           << ", num_threads: " << num_threads
    //           << ", num_experts: " << num_experts
    //           << std::endl;
    CHECK_HIP(hipMemsetAsync(grid_sync_buffer, 0, sizeof(int), stream));
    kernel_1<kNumWavesPerGroup, kNumWaveGroups>
      <<<grid, block, 0, stream>>>(amo_buffers.atomic_buffers[amo_buffers.buffer_idx],
        amo_buffers.num_elems, amo_buffers.atomic_buffers[amo_buffers.buffer_idx ^= 1],
        n_pes, my_pe, grid_sync_buffer, iter);

    // Synchronize the stream
    CHECK_HIP(hipStreamSynchronize(stream));

    // Reset the grid sync buffer
    CHECK_HIP(hipMemsetAsync(grid_sync_buffer, 0, sizeof(int), stream));
    kernel_2<kNumWavesPerGroup, kNumWaveGroups>
      <<<grid, block, 0, stream>>>(amo_buffers.atomic_buffers[amo_buffers.buffer_idx],
        amo_buffers.num_elems, amo_buffers.atomic_buffers[amo_buffers.buffer_idx ^= 1],
        n_pes, my_pe, grid_sync_buffer, iter);

    // Synchronize the stream
    CHECK_HIP(hipStreamSynchronize(stream));
  }

  // Synchronize all the PEs
  rocshmem_barrier_all();

  // Finalize
  CHECK_HIP(hipFree(grid_sync_buffer));
  rocshmem_free(atomic_buffer_ptr);
  CHECK_HIP(hipStreamDestroy(stream));
  rocshmem_finalize();
  return 0;
}

#if defined(HAVE_PMIX)
#include <pmix.h>

static pmix_proc_t pmix_myproc;
static pmix_proc_t pmix_proc;

static void init_pmix(int *rank, int *nranks)
{
    pmix_status_t rc;
    pmix_value_t *val;

    if (PMIX_SUCCESS != (rc = PMIx_Init(&pmix_myproc, NULL, 0))) {
      std::cerr << "Rank " << pmix_myproc.rank << " PMIx_Init failed: " << rc << std::endl;
      abort();
    }
#ifdef VERBOSE
    printf("Client ns %s rank %d: Running\n", pmix_myproc.nspace, pmix_myproc.rank);
#endif
    PMIX_PROC_CONSTRUCT(&pmix_proc);
    PMIX_LOAD_PROCID(&pmix_proc, pmix_myproc.nspace, PMIX_RANK_WILDCARD);

    /* get our job size */
    if (PMIX_SUCCESS != (rc = PMIx_Get(&pmix_proc, PMIX_JOB_SIZE, NULL, 0, &val))) {
      std::cerr << "Rank " << pmix_myproc.rank << " PMIx_Get universe size failed: "
                <<  rc << std::endl;
        abort();
    }

    *nranks = val->data.uint32;
    *rank   = pmix_myproc.rank;

    PMIX_VALUE_RELEASE(val);
    return;
}

static void pmix_finalize() {
  PMIx_Finalize(NULL, 0);
}

static void pmix_bcast(void *buf, size_t nbytes, char *key, int root)
{
    pmix_status_t rc;
    pmix_value_t value;
    pmix_value_t *val;
    pmix_info_t *info;
    bool flag;

    if (pmix_myproc.rank == root) {
      value.type = PMIX_BYTE_OBJECT;
      value.data.bo.bytes = (char *) (buf);
      value.data.bo.size = nbytes;

      rc = PMIx_Put(PMIX_GLOBAL, key, &value);
      if (PMIX_SUCCESS != rc) {
        std::cerr << "Rank " << pmix_myproc.rank << " PMIx_Put failed: " << rc << std::endl;
        abort();
      }

      /* push the data to our PMIx server */
      if (PMIX_SUCCESS != (rc = PMIx_Commit())) {
        std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Commit failed: " << rc << std::endl;
        abort();
      }
    }

    /* call fence to synchronize with our peers - instruct
     * the fence operation to collect and return all "put"
     * data from our peers */
    PMIX_INFO_CREATE(info, 1);
    flag = true;
        PMIX_INFO_LOAD(info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);
    if (PMIX_SUCCESS != (rc = PMIx_Fence(&pmix_proc, 1, info, 1))) {
      std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Fence failed: " << rc << std::endl;
      abort();
    }
    PMIX_INFO_FREE(info, 1);

    pmix_proc.rank = 0;
    if (PMIX_SUCCESS != (rc = PMIx_Get(&pmix_proc, key, NULL, 0, &val))) {
      std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Get failed: " << rc << std::endl;
      abort();
    }
    if (PMIX_BYTE_OBJECT != val->type) {
      std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Get returned wrong type: " << val->type  << std::endl;
      PMIX_VALUE_RELEASE(val);
      abort();
    }

    if (pmix_myproc.rank != root) {
      if (NULL == val->data.bo.bytes) {
        std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Get %d returned NULL pointer\n";
        PMIX_VALUE_RELEASE(val);
        abort();
      }
      memcpy (buf, val->data.bo.bytes, val->data.bo.size);
    }
    PMIX_VALUE_RELEASE(val);

    return;
}
#endif

