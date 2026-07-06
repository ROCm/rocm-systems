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

/* Declare the template with a generic implementation */
template <typename T, ROCSHMEM_OP Op>
__device__ int wave_team_reduce([[maybe_unused]] rocshmem_ctx_t ctx,
                                [[maybe_unused]] rocshmem_team_t team,
                                [[maybe_unused]] T *dest,
                                [[maybe_unused]] const T *source,
                                [[maybe_unused]] int nreduce) {
  return ROCSHMEM_SUCCESS;
}

/* Define templates to call rocSHMEM */
#define TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, Op_API, Op)                        \
  template <>                                                                     \
  __device__ int wave_team_reduce<T, Op>(rocshmem_ctx_t ctx,                     \
                                         rocshmem_team_t team, T *dest,           \
                                         const T *source, int nreduce) {          \
    return rocshmem_ctx_##TNAME##_##Op_API##_reduce_wave(ctx, team, dest,         \
                                                          source, nreduce);       \
  }

#define TEAM_ARITH_REDUCTION_WAVE_DEF_GEN(T, TNAME)              \
  TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM)       \
  TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN)       \
  TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX)       \
  TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)

#define TEAM_BITWISE_REDUCTION_WAVE_DEF_GEN(T, TNAME)            \
  TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)         \
  TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, and, ROCSHMEM_AND)       \
  TEAM_REDUCTION_WAVE_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)

#define TEAM_INT_REDUCTION_WAVE_DEF_GEN(T, TNAME)  \
  TEAM_ARITH_REDUCTION_WAVE_DEF_GEN(T, TNAME)      \
  TEAM_BITWISE_REDUCTION_WAVE_DEF_GEN(T, TNAME)

#define TEAM_FLOAT_REDUCTION_WAVE_DEF_GEN(T, TNAME) \
  TEAM_ARITH_REDUCTION_WAVE_DEF_GEN(T, TNAME)

TEAM_INT_REDUCTION_WAVE_DEF_GEN(int, int)
TEAM_INT_REDUCTION_WAVE_DEF_GEN(short, short)
TEAM_INT_REDUCTION_WAVE_DEF_GEN(long, long)
TEAM_INT_REDUCTION_WAVE_DEF_GEN(long long, longlong)
TEAM_FLOAT_REDUCTION_WAVE_DEF_GEN(float, float)
TEAM_FLOAT_REDUCTION_WAVE_DEF_GEN(double, double)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1, ROCSHMEM_OP T2>
__global__ void ReduceWaveTest(int loop, int skip, long long int *start_time,
                               long long int *end_time, T1 *s_buf, T1 *r_buf,
                               size_t size, [[maybe_unused]] TestType type,
                               ShmemContextType ctx_type, int wf_size,
                               rocshmem_team_t *teams) {
  int t_id  = get_flat_block_id();
  int wg_id = get_flat_grid_id();
  int wf_id = t_id / wf_size;
  int wf_per_wg = hipBlockDim_x / wf_size;

  int flat_wf_id = wg_id * wf_per_wg + wf_id;

  __shared__ rocshmem_ctx_t ctx;

  rocshmem_wg_team_create_ctx(teams[flat_wf_id], ctx_type, &ctx);

  __syncthreads();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && hipThreadIdx_x == 0) {
      start_time[flat_wf_id] = wall_clock64();
    }
    if (wf_id == 0) {
      wave_team_reduce<T1, T2>(ctx, teams[flat_wf_id],
                             r_buf + flat_wf_id * size,
                             s_buf + flat_wf_id * size, size);
    }
    __syncthreads();
  }

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[flat_wf_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T1, ROCSHMEM_OP T2>
ReduceWaveTester<T1, T2>::ReduceWaveTester(
    TesterArguments args, std::function<void(T1 &, T1 &)> f1,
    std::function<std::pair<bool, std::string>(const T1 &, const T1 &)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2} {
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  s_buf = (T1 *)alloc_test_buffer(max_msg_size * sizeof(T1), args.local_buf_type);
  r_buf = (T1 *)alloc_test_buffer(max_msg_size * sizeof(T1));

  char *value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  if (num_teams < num_warps){
    printf("not enough teams for each wavefront, try increasing ROCSHMEM_MAX_NUM_TEAMS\n");
    abort();
  }

  CHECK_HIP(hipMalloc(&reduce_wave_world_dup,
                      sizeof(rocshmem_team_t) * num_teams));
}

template <typename T1, ROCSHMEM_OP T2>
ReduceWaveTester<T1, T2>::~ReduceWaveTester() {
  free_test_buffer(s_buf, args.local_buf_type);
  free_test_buffer(r_buf);
  CHECK_HIP(hipFree(reduce_wave_world_dup));
}

template <typename T1, ROCSHMEM_OP T2>
void ReduceWaveTester<T1, T2>::preLaunchKernel() {
  bw_factor = n_pes;

  for (int team_i = 0; team_i < num_teams; team_i++) {
    reduce_wave_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &reduce_wave_world_dup[team_i]);
    if (reduce_wave_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      std::cout << "Team " << team_i << " is invalid!" << std::endl;
      abort();
    }
  }
}

template <typename T1, ROCSHMEM_OP T2>
void ReduceWaveTester<T1, T2>::launchKernel(dim3 gridSize, dim3 blockSize,
                                            int loop, size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(ReduceWaveTest<T1, T2>), gridSize,
                     blockSize, shared_bytes, stream, loop, args.skip,
                     start_time, end_time, s_buf, r_buf, size, _type,
                     _shmem_context, wf_size, reduce_wave_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

template <typename T1, ROCSHMEM_OP T2>
void ReduceWaveTester<T1, T2>::postLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(reduce_wave_world_dup[team_i]);
  }
}

template <typename T1, ROCSHMEM_OP T2>
void ReduceWaveTester<T1, T2>::resetBuffers([[maybe_unused]] size_t size) {
  for (uint64_t i = 0; i < max_msg_size; i++) {
    init_buf(s_buf[i], r_buf[i]);
  }
}

template <typename T1, ROCSHMEM_OP T2>
void ReduceWaveTester<T1, T2>::verifyResults(size_t size) {
  for (uint64_t i = 0; i < size; i++) {
    auto r = verify_buf(r_buf[i], (T1)n_pes);
    if (r.first == false) {
      fprintf(stderr, "Data validation error at idx %lu\n", i);
      fprintf(stderr, "%s.\n", r.second.c_str());
      exit(-1);
    }
  }
}
