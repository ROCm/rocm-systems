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

template <typename T>
__device__ void wave_broadcast([[maybe_unused]] rocshmem_ctx_t ctx, [[maybe_unused]] rocshmem_team_t team,
                                  [[maybe_unused]] T *dest, [[maybe_unused]] const T *source, [[maybe_unused]] int nelems,
                                  [[maybe_unused]] int pe_root) {
  return;
}

/* Define templates to call ROCSHMEM */
#define BROADCAST_WAVE_DEF_GEN(T, TNAME)                                      \
  template <>                                                                 \
  __device__ void wave_broadcast<T>(                                          \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,    \
      int nelems, int pe_root) {                                              \
    rocshmem_ctx_##TNAME##_broadcast_wave(ctx, team, dest, source, nelems,    \
                                         pe_root);                            \
  }

BROADCAST_WAVE_DEF_GEN(float, float)
BROADCAST_WAVE_DEF_GEN(double, double)
BROADCAST_WAVE_DEF_GEN(char, char)
// BROADCAST_WAVE_DEF_GEN(long double, longdouble)
BROADCAST_WAVE_DEF_GEN(signed char, schar)
BROADCAST_WAVE_DEF_GEN(short, short)
BROADCAST_WAVE_DEF_GEN(int, int)
BROADCAST_WAVE_DEF_GEN(long, long)
BROADCAST_WAVE_DEF_GEN(long long, longlong)
BROADCAST_WAVE_DEF_GEN(unsigned char, uchar)
BROADCAST_WAVE_DEF_GEN(unsigned short, ushort)
BROADCAST_WAVE_DEF_GEN(unsigned int, uint)
BROADCAST_WAVE_DEF_GEN(unsigned long, ulong)
BROADCAST_WAVE_DEF_GEN(unsigned long long, ulonglong)
BROADCAST_WAVE_DEF_GEN(__half, half)
BROADCAST_WAVE_DEF_GEN(__hip_bfloat16, bfloat16)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1>
__global__ void BroadcastWaveTestKernel(int loop, int skip, long long int *start_time,
                                  long long int *end_time, T1 *source_buf,
                                  T1 *dest_buf, int size,
                                  ShmemContextType ctx_type,
                                  rocshmem_team_t *teams, int wf_size,
                                  int num_waves_per_wg) {
  extern __shared__ rocshmem_ctx_t ctx_array[];

  int t_id       = get_flat_block_id();
  int wg_id      = get_flat_grid_id();
  int wf_id      = t_id / wf_size;
  int wg_offset  = wg_id * num_waves_per_wg;
  int flat_wf_id = wg_offset + wf_id;

  // All threads in the WG collectively create one context per wave.
  // Each iteration is WG-collective so __syncthreads() guards each call.
  for (int wf_i = 0; wf_i < num_waves_per_wg; wf_i++) {
    rocshmem_wg_team_create_ctx(teams[wg_offset + wf_i], ctx_type,
                                &ctx_array[wf_i]);
    __syncthreads();
  }

  // Each wave uses its own buffer slice
  T1 *my_source = source_buf + flat_wf_id * size;
  T1 *my_dest   = dest_buf   + flat_wf_id * size;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && t_id % wf_size == 0) {
      start_time[flat_wf_id] = wall_clock64();
    }
    wave_broadcast<T1>(ctx_array[wf_id], teams[flat_wf_id], my_dest, my_source,
                       size, 0);
  }

  __syncthreads();

  if (t_id % wf_size == 0) {
    end_time[flat_wf_id] = wall_clock64();
  }

  // Destroy all contexts — WG-collective, same order as creation
  for (int wf_i = 0; wf_i < num_waves_per_wg; wf_i++) {
    rocshmem_wg_ctx_destroy(&ctx_array[wf_i]);
    __syncthreads();
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T1>
BroadcastWaveTester<T1>::BroadcastWaveTester(TesterArguments args)
    : Tester(args){
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  // One buffer slice per wave (num_wgs * num_warps total waves)
  int total_elems = (max_msg_size / sizeof(T1)) * args.num_wgs * num_warps;
  int buff_size = total_elems * sizeof(T1);

  source_buf = (T1 *)alloc_test_buffer(buff_size, args.local_buf_type);
  dest_buf = (T1 *)alloc_test_buffer(buff_size);

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  // Need one team per wave
  int total_waves = num_warps * args.num_wgs;
  if (num_teams < total_waves){
    printf("not enough teams for each wavefront, try increasing ROCSHMEM_MAX_NUM_TEAMS\n");
    exit(0);
  }

  CHECK_HIP(hipMalloc(&bcast_wave_world_dup,
                      sizeof(rocshmem_team_t) * total_waves));
}

template <typename T1>
BroadcastWaveTester<T1>::~BroadcastWaveTester() {
  free_test_buffer(source_buf, args.local_buf_type);
  free_test_buffer(dest_buf);
  CHECK_HIP(hipFree(bcast_wave_world_dup));
}

template <typename T1>
void BroadcastWaveTester<T1>::preLaunchKernel() {
  bw_factor = n_pes;

  int total_waves = num_warps * args.num_wgs;
  for (int team_i = 0; team_i < total_waves; team_i++) {
    bcast_wave_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                 &bcast_wave_world_dup[team_i]);
    if (bcast_wave_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      printf("Team %d is invalid!\n", team_i);
      abort();
    }
  }
}

template <typename T1>
void BroadcastWaveTester<T1>::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, size_t size) {
  size_t shared_bytes = num_warps * sizeof(rocshmem_ctx_t);

  int num_elems = size / sizeof(T1);

  hipLaunchKernelGGL(BroadcastWaveTestKernel<T1>, gridSize, blockSize,
                     shared_bytes, stream, loop, args.skip,
                     start_time, end_time, source_buf, dest_buf,
                     num_elems, _shmem_context, bcast_wave_world_dup, wf_size,
                     num_warps);

  num_msgs = (loop + args.skip) * gridSize.x * num_warps;
  num_timed_msgs = loop * gridSize.x * num_warps;
}

template <typename T1>
void BroadcastWaveTester<T1>::postLaunchKernel() {
  int total_waves = num_warps * args.num_wgs;
  for (int team_i = 0; team_i < total_waves; team_i++) {
    rocshmem_team_destroy(bcast_wave_world_dup[team_i]);
  }
}

template <typename T1>
void BroadcastWaveTester<T1>::resetBuffers(size_t size) {
  int num_elems = size / sizeof(T1);
  int total_waves = args.num_wgs * num_warps;

  for (int wave_id = 0; wave_id < total_waves; wave_id++) {
    for (int i = 0; i < num_elems; i++) {
      int idx = wave_id * num_elems + i;
      if constexpr (std::is_same<T1, __half>::value ||
                    std::is_same<T1, __hip_bfloat16>::value) {
        source_buf[idx] = static_cast<T1>(3.14f + n_pes + wave_id);
        dest_buf[idx]   = static_cast<T1>(3.14f + wave_id);
      } else if constexpr (std::is_floating_point<T1>::value) {
        source_buf[idx] = static_cast<T1>(3.14 + n_pes + wave_id);
        dest_buf[idx]   = static_cast<T1>(3.14 + wave_id);
      } else {
        source_buf[idx] = static_cast<T1>('a' + n_pes + wave_id);
        dest_buf[idx]   = static_cast<T1>('a' + wave_id);
      }
    }
  }
}

template <typename T1>
void BroadcastWaveTester<T1>::verifyResults(size_t size) {
  int num_elems = size / sizeof(T1);
  int total_waves = args.num_wgs * num_warps;

  for (int wave_id = 0; wave_id < total_waves; wave_id++) {
    for (int i = 0; i < num_elems; i++) {
      int idx = wave_id * num_elems + i;
      T1 expected;
      if constexpr (std::is_same<T1, __half>::value ||
                    std::is_same<T1, __hip_bfloat16>::value) {
        expected = static_cast<T1>(3.14f + wave_id + n_pes);
      } else if constexpr (std::is_floating_point<T1>::value) {
        expected = static_cast<T1>(3.14 + wave_id + n_pes);
      } else {
        expected = static_cast<T1>('a' + wave_id + n_pes);
      }
      if (dest_buf[idx] != expected) {
        std::cerr << "Data validation error at wave " << wave_id
                  << " idx " << i << std::endl;
        if constexpr (std::is_same<T1, __half>::value ||
                      std::is_same<T1, __hip_bfloat16>::value) {
          std::cerr << "PE " << my_pe << " Got " << static_cast<float>(dest_buf[idx])
                    << ", Expected " << static_cast<float>(expected) << std::endl;
        } else {
          std::cerr << "PE " << my_pe << " Got " << dest_buf[idx]
                    << ", Expected " << expected << std::endl;
        }
        exit(-1);
      }
    }
  }
}
