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

namespace deepep_config {
  using CounterType = size_t;
  // 512 MiB
  constexpr size_t kNumSenderWarps = 4;
  constexpr size_t kNumSenderThreads = kNumSenderWarps * warpSize;
  constexpr size_t kThreadsInBlock = 512;
  constexpr size_t kSendCount = 8 * warpSize;
  constexpr size_t kBufferCount = kNumSenderWarps * kSendCount * 8 * 1024;
  constexpr size_t kWorldSize = 2;
  constexpr size_t kNumGpus = 8;
  constexpr uint64_t kFullWarpMask = 0xFFFF'FFFF'FFFF'FFFF;

  static_assert(kThreadsInBlock % warpSize == 0);
  static_assert(kBufferCount % kNumSenderThreads == 0);
  static_assert(kBufferCount % kSendCount == 0);
  static_assert(kSendCount % warpSize == 0);

  enum class WarpRole : int {
    Sender = 0,
    Receiver = 1,
  };
}

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__device__ void deepep_pattern(rocshmem_ctx_t ctx,
                               const T *send, T *recv,
                               CounterType *num_received_data,
                               const size_t num_elements, const int rank_to_send) {
  assert(num_elements % kSendCount == 0);
  assert(num_elements % kNumSenderThreads == 0);

  const auto thread_id = threadIdx.x;
  const auto wave_id = thread_id / warpSize;
  const auto lane_id = thread_id % warpSize;
  const auto global_thread_id = thread_id + blockIdx.x * blockDim.x;
  const auto warp_role = global_thread_id < kNumSenderThreads
                             ? WarpRole::Sender
                             : WarpRole::Receiver;
  const auto my_pe = rocshmem_my_pe();

  if(0 == global_thread_id) rocshmem_barrier_all();
  __syncthreads();

  if (warp_role == WarpRole::Sender) {
#if VERBOSE
    if (global_thread_id == 0 || global_thread_id == kNumSenderThreads-1) printf("[%02d] %02d:%02d (%02d:%02d) role is sender\n", my_pe, blockIdx.x, threadIdx.x, wave_id, lane_id);
#endif
    ulong num_sent_elements = 0;
    for (size_t idx = global_thread_id / warpSize * kSendCount; idx < num_elements;
         idx += kSendCount * kNumSenderWarps) {
#if VERBOSE
      if (0==lane_id) printf("[%02d] %02d:%02d (%02d:%02d) put_nbi_wave(recv_off=%zd, send_off=%zd, size=%zd, to=%d)\n", my_pe, blockIdx.x, threadIdx.x, wave_id, lane_id,
          idx, idx, kSendCount, rank_to_send);
#endif
      rocshmem_ctx_put_nbi_wave(ctx, recv + idx, send + idx, kSendCount, rank_to_send);
      num_sent_elements += kSendCount;
      rocshmem_ctx_fence(ctx);
    }
    // __syncwarp()
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

    if (0 != num_sent_elements) {
#if VERBOSE
      printf("[%02d] %02d:%02d (%02d:%02d) ulong_atomic_add(0, addval=%zd, to=%d)\n", my_pe, blockIdx.x, threadIdx.x, wave_id, lane_id,
          num_sent_elements, rank_to_send);
#endif
      rocshmem_ctx_ulong_atomic_add(ctx, &num_received_data[0], num_sent_elements / warpSize,
                                    rank_to_send);
      rocshmem_ctx_ulong_atomic_add(ctx, &num_received_data[1], num_sent_elements / warpSize,
                                    my_pe);
      rocshmem_ctx_fence(ctx);
    }
  } else {
    bool ready = false;
    const int offset = global_thread_id - kNumSenderThreads;
#if VERBOSE
    if (0 == lane_id) printf("[%02d] %02d:%02d (%02d:%02d) Start recvrole (offset=%d)\n", my_pe, blockIdx.x, threadIdx.x, wave_id, lane_id,
        offset);
#endif
    // __any_sync(kFullWarpMask, ready)
    while (~__ballot(ready) & kFullWarpMask) {
      auto val = (offset < num_elements)? __builtin_nontemporal_load(recv + offset): offset;
      ready = val == offset;
    }
#if VERBOSE
    if (0 == lane_id) printf("[%02d] %02d:%02d (%02d:%02d) End   recvrole (offset=%d)\n", my_pe, blockIdx.x, threadIdx.x, wave_id, lane_id,
        offset);
#endif
  }

#if VERBOSE
  if (lane_id == 0) printf("[%02d] %02d:%02d (%02d:%02d) Start amoballot\n", my_pe, blockIdx.x, threadIdx.x, wave_id, lane_id);
#endif
  bool ready = false;
  // __any_sync(kFullWarpMask, ready)
  while (~__ballot(ready) & kFullWarpMask) {
    ready = (__builtin_nontemporal_load(&num_received_data[0]) == num_elements);
  }
  ready = false;
  // __any_sync(kFullWarpMask, ready)
  while (~__ballot(ready) & kFullWarpMask) {
    ready = (__builtin_nontemporal_load(&num_received_data[1]) == num_elements);
  }
#if VERBOSE
  if (lane_id == 0) printf("[%02d] %02d:%02d (%02d:%02d) End   amoballot (num_received=%zd)\n", my_pe, blockIdx.x, threadIdx.x, wave_id, lane_id, num_received_data[1]);
#endif
}

template <typename T>
__global__ void DeepepPatternTest(int loop, int skip, long long int *start_time, long long int *end_time,
                                 const T *source, T *dest, CounterType  *num_received_data, int num_elems,
                                 ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_create_ctx(ctx_type, &ctx);

  int n_pes = rocshmem_ctx_n_pes(ctx);

  __syncthreads();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && hipThreadIdx_x == 0) {
      start_time[wg_id] = wall_clock64();
    }
    deepep_pattern(ctx, source, dest, num_received_data, num_elems, rank_to_send);
  }

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
DeepepPatternTester::DeepepPatternTester(TesterArguments args)
    : Tester(args){
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  // Number of elements per work group
  int num_elems_wg = (args.max_msg_size / sizeof(T)) * n_pes;
  // Total number of elements in the GPU kernel
  int total_elems = num_elems_wg * args.num_wgs;
  int buff_size = kBufferCount * sizeof(T);
    //total_elems * sizeof(T);

  sourcef = (T *)rocshmem_malloc(buff_size);
  destf   = (T *)rocshmem_malloc(buff_size);
  CounterType *num_received_data = (CounterType *)rocshmem_malloc(sizeof(CounterType) * 2);

  if (sourcef == nullptr || destf == nullptr || num_received_data == nullptr) {
    std::cout << "Error allocating memory from symmetric heap" << std::endl;
    std::cout << "source: " << source
              << ", target: " << dest
              << ", countr: " << num_received_data
              << std::endl;
    rocshmem_global_exit(1);
  }
}

template <typename T>
DeepepPatternTester<T>::~TeamAlltoallTester() {
  rocshmem_free(sourcef);
  rocshmem_free(destf);
}

template <typename T>
void DeepepPatternTester<T>::launchKernel(dim3 gridSize, dim3 blockSize,
                                          int loop, size_t size) {
  size_t shared_bytes = 0;

  int num_elems = size / sizeof(T);

  hipLaunchKernelGGL(DeepepPatternTest<T>, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     source, dest, num_elems, _shmem_context,
                     team_alltoall_world_dup);
#if 0
  test_send<<<(kNumSenderThreads + kBufferCount + kThreadsInBlock - 1) / kThreadsInBlock, kThreadsInBlock>>>
           (source, target, num_received_data, kBufferCount, rank_to_send);
#endif

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

template <typename T>
void DeepepPatternTester<T>::resetBuffers(size_t size) {
#if 0
  num_received_data[0] = num_received_data[1] = 0;
  for (size_t i = 0; i < kBufferCount; ++i) {
    source[i] = i;
  }
#endif
  int num_elems = size / sizeof(T);
  int buff_size = num_elems * sizeof(T) * args.num_wgs * n_pes;
  int idx = 0;

  for(int wg_id = 0; wg_id < args.num_wgs; wg_id++) {
    for(int pe = 0; pe < n_pes; pe++) {
      for(int i = 0; i < num_elems; i++) {
        idx = (wg_id * n_pes + pe) * num_elems + i;
        if constexpr (std::is_same<T, char>::value ||
                      std::is_same<T, signed char>::value ||
                      std::is_same<T, unsigned char>::value) {
          source[idx] = static_cast<T>('a' + my_pe + pe + wg_id);
        }
        else if constexpr (std::is_floating_point<T>::value) {
          source[idx] = static_cast<T>(3.14 + my_pe + pe + wg_id);
        }
        else if constexpr (std::is_integral<T>::value) {
          source[idx] = static_cast<T>(my_pe + pe + wg_id);
        }
      }
    }
  }

  memset(dest, -1, buff_size);
}

template <typename T>
void DeepepPatternTester<T>::verifyResults(size_t size) {
  int num_elems = size / sizeof(T);
  int idx = 0;

  for(int wg_id = 0; wg_id < args.num_wgs; wg_id++) {
    for(int pe = 0; pe < n_pes; pe++) {
      for(int i = 0; i < num_elems; i++) {
        idx = (wg_id * n_pes + pe) * num_elems + i;
        if (dest[idx] != source[idx]) {
          std::cerr << "Data validation error at idx " << idx << std::endl;
          std::cerr << "PE " << my_pe << " Got " << dest[idx]
          << ", Expected " << source[idx] << std::endl;
          exit(-1);
        }
      }
    }
  }
}
