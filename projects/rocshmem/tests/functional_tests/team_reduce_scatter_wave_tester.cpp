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

using namespace rocshmem;

/*
 * reduce_scatter_wave(team, dest, source, nreduce):
 *   - source has n_pes * nreduce elements per PE
 *   - PE i receives the element-wise reduction of source[i*nreduce..(i+1)*nreduce-1]
 *     from all PEs into dest[0..nreduce-1]
 *
 * Only wave 0 (wf_id == 0) within each workgroup calls the API.
 * All threads sync afterward so timing is measured across the full WG.
 */

/* Generic stub — specializations below call the actual rocSHMEM API */
template <typename T, ROCSHMEM_OP Op>
__device__ int wave_team_reduce_scatter([[maybe_unused]] rocshmem_ctx_t ctx,
                                        [[maybe_unused]] rocshmem_team_t team,
                                        [[maybe_unused]] T *dest,
                                        [[maybe_unused]] const T *source,
                                        [[maybe_unused]] int nreduce) {
  return ROCSHMEM_SUCCESS;
}

#define TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, Op_API, Op)                    \
  template <>                                                                      \
  __device__ int wave_team_reduce_scatter<T, Op>(rocshmem_ctx_t ctx,              \
                                                 rocshmem_team_t team, T *dest,   \
                                                 const T *source, int nreduce) {  \
    return rocshmem_ctx_##TNAME##_##Op_API##_reduce_scatter_wave(ctx, team, dest, \
                                                                  source, nreduce);\
  }

#define TEAM_ARITH_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME)              \
  TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM)       \
  TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN)       \
  TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX)       \
  TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)

#define TEAM_BITWISE_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME)            \
  TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)         \
  TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, and, ROCSHMEM_AND)       \
  TEAM_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)

#define TEAM_INT_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME)  \
  TEAM_ARITH_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME)      \
  TEAM_BITWISE_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME)

#define TEAM_FLOAT_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME) \
  TEAM_ARITH_REDUCE_SCATTER_WAVE_DEF_GEN(T, TNAME)

TEAM_INT_REDUCE_SCATTER_WAVE_DEF_GEN(int, int)
TEAM_INT_REDUCE_SCATTER_WAVE_DEF_GEN(short, short)
TEAM_INT_REDUCE_SCATTER_WAVE_DEF_GEN(long, long)
TEAM_INT_REDUCE_SCATTER_WAVE_DEF_GEN(long long, longlong)
TEAM_FLOAT_REDUCE_SCATTER_WAVE_DEF_GEN(float, float)
TEAM_FLOAT_REDUCE_SCATTER_WAVE_DEF_GEN(double, double)
TEAM_FLOAT_REDUCE_SCATTER_WAVE_DEF_GEN(__half, half)
TEAM_FLOAT_REDUCE_SCATTER_WAVE_DEF_GEN(__hip_bfloat16, bfloat16)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1, ROCSHMEM_OP T2>
__global__ void TeamReduceScatterWaveTest(int loop, int skip,
                                          long long int *start_time,
                                          long long int *end_time, T1 *s_buf,
                                          T1 *r_buf, size_t size,
                                          [[maybe_unused]] TestType type,
                                          ShmemContextType ctx_type,
                                          rocshmem_team_t *teams,
                                          int num_waves_per_wg) {
  extern __shared__ rocshmem_ctx_t ctx_array[];

  int t_id       = get_flat_block_id();
  int wg_id      = get_flat_grid_id();
  int wf_id      = t_id / WF_SIZE;
  int wg_offset  = wg_id * num_waves_per_wg;
  int flat_wf_id = wg_offset + wf_id;

  // All threads in the WG collectively create one context per wave.
  // Each iteration is WG-collective so __syncthreads() guards each call.
  for (int wf_i = 0; wf_i < num_waves_per_wg; wf_i++) {
    rocshmem_wg_team_create_ctx(teams[wg_offset + wf_i], ctx_type,
                                &ctx_array[wf_i]);
    __syncthreads();
  }

  int n_pes = rocshmem_ctx_n_pes(ctx_array[wf_id]);

  // Each wave uses its own buffer slice
  T1 *my_s_buf = s_buf + flat_wf_id * n_pes * size;
  T1 *my_r_buf = r_buf + flat_wf_id * size;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && t_id % WF_SIZE == 0) {
      start_time[flat_wf_id] = wall_clock64();
    }
    wave_team_reduce_scatter<T1, T2>(ctx_array[wf_id], teams[flat_wf_id],
                                     my_r_buf, my_s_buf, size);
  }

  __syncthreads();

  if (t_id % WF_SIZE == 0) {
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
template <typename T1, ROCSHMEM_OP T2>
TeamReduceScatterWaveTester<T1, T2>::TeamReduceScatterWaveTester(
    TesterArguments args, std::function<void(T1 &, T1 &)> f1,
    std::function<std::pair<bool, std::string>(const T1 &, const T1 &)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2} {
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  int total_waves = args.num_wgs * num_warps;

  // One source slice (n_pes * max_msg_size) and one dest slice (max_msg_size)
  // per wave across all WGs
  s_buf = (T1 *)alloc_test_buffer(n_pes * max_msg_size * sizeof(T1) * total_waves,
                                  args.local_buf_type);
  r_buf = (T1 *)alloc_test_buffer(max_msg_size * sizeof(T1) * total_waves);

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  // Need one team per wave
  if (num_teams < total_waves) {
    printf("not enough teams for each wavefront, try increasing ROCSHMEM_MAX_NUM_TEAMS\n");
    exit(0);
  }

  CHECK_HIP(hipMalloc(&team_reduce_scatter_wave_world_dup,
                      sizeof(rocshmem_team_t) * total_waves));
}

template <typename T1, ROCSHMEM_OP T2>
TeamReduceScatterWaveTester<T1, T2>::~TeamReduceScatterWaveTester() {
  free_test_buffer(s_buf, args.local_buf_type);
  free_test_buffer(r_buf);
  CHECK_HIP(hipFree(team_reduce_scatter_wave_world_dup));
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterWaveTester<T1, T2>::preLaunchKernel() {
  bw_factor = n_pes;

  int total_waves = args.num_wgs * num_warps;
  for (int team_i = 0; team_i < total_waves; team_i++) {
    team_reduce_scatter_wave_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &team_reduce_scatter_wave_world_dup[team_i]);
    if (team_reduce_scatter_wave_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      printf("Team %d is invalid!\n", team_i);
      abort();
    }
  }
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterWaveTester<T1, T2>::launchKernel(dim3 gridSize,
                                                        dim3 blockSize,
                                                        int loop,
                                                        uint64_t size) {
  size_t shared_bytes = num_warps * sizeof(rocshmem_ctx_t);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(TeamReduceScatterWaveTest<T1, T2>),
                     gridSize, blockSize, shared_bytes, stream, loop, args.skip,
                     start_time, end_time, s_buf, r_buf, size, _type,
                     _shmem_context, team_reduce_scatter_wave_world_dup,
                     num_warps);

  num_msgs = (loop + args.skip) * gridSize.x * num_warps;
  num_timed_msgs = loop * gridSize.x * num_warps;
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterWaveTester<T1, T2>::postLaunchKernel() {
  int total_waves = args.num_wgs * num_warps;
  for (int team_i = 0; team_i < total_waves; team_i++) {
    rocshmem_team_destroy(team_reduce_scatter_wave_world_dup[team_i]);
  }
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterWaveTester<T1, T2>::resetBuffers(
    [[maybe_unused]] uint64_t size) {
  int total_waves = args.num_wgs * num_warps;

  for (uint64_t i = 0; i < (uint64_t)n_pes * max_msg_size * total_waves; i++) {
    s_buf[i] = static_cast<T1>(1);
  }
  for (uint64_t i = 0; i < (uint64_t)max_msg_size * total_waves; i++) {
    r_buf[i] = static_cast<T1>(0);
  }
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterWaveTester<T1, T2>::verifyResults(uint64_t size) {
  int total_waves = args.num_wgs * num_warps;

  for (int wave_id = 0; wave_id < total_waves; wave_id++) {
    T1 *wave_r_buf = r_buf + wave_id * size;
    for (uint64_t i = 0; i < size; i++) {
      auto r = verify_buf(wave_r_buf[i], (T1)n_pes);
      if (r.first == false) {
        fprintf(stderr, "Data validation error at wave %d idx %lu\n", wave_id, i);
        fprintf(stderr, "%s.\n", r.second.c_str());
        exit(-1);
      }
    }
  }
}
