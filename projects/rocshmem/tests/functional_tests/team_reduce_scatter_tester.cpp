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
 * reduce_scatter(team, dest, source, nreduce):
 *   - source has n_pes * nreduce elements per PE
 *   - PE i receives the element-wise reduction of source[i*nreduce..(i+1)*nreduce-1]
 *     from all PEs into dest[0..nreduce-1]
 */

/* Generic stub — specializations below call the actual rocSHMEM API */
template <typename T, ROCSHMEM_OP Op>
__device__ int wg_team_reduce_scatter([[maybe_unused]] rocshmem_ctx_t ctx,
                                      [[maybe_unused]] rocshmem_team_t team,
                                      [[maybe_unused]] T *dest,
                                      [[maybe_unused]] const T *source,
                                      [[maybe_unused]] int nreduce) {
  return ROCSHMEM_SUCCESS;
}

#define TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, Op_API, Op)                       \
  template <>                                                                    \
  __device__ int wg_team_reduce_scatter<T, Op>(rocshmem_ctx_t ctx,              \
                                               rocshmem_team_t team, T *dest,   \
                                               const T *source, int nreduce) {  \
    return rocshmem_ctx_##TNAME##_##Op_API##_reduce_scatter_wg(ctx, team, dest, \
                                                                source, nreduce);\
  }

#define TEAM_ARITH_REDUCE_SCATTER_DEF_GEN(T, TNAME)              \
  TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM)       \
  TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN)       \
  TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX)       \
  TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)

#define TEAM_BITWISE_REDUCE_SCATTER_DEF_GEN(T, TNAME)            \
  TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)         \
  TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, and, ROCSHMEM_AND)       \
  TEAM_REDUCE_SCATTER_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)

#define TEAM_INT_REDUCE_SCATTER_DEF_GEN(T, TNAME)  \
  TEAM_ARITH_REDUCE_SCATTER_DEF_GEN(T, TNAME)      \
  TEAM_BITWISE_REDUCE_SCATTER_DEF_GEN(T, TNAME)

#define TEAM_FLOAT_REDUCE_SCATTER_DEF_GEN(T, TNAME) \
  TEAM_ARITH_REDUCE_SCATTER_DEF_GEN(T, TNAME)

TEAM_INT_REDUCE_SCATTER_DEF_GEN(int, int)
TEAM_INT_REDUCE_SCATTER_DEF_GEN(short, short)
TEAM_INT_REDUCE_SCATTER_DEF_GEN(long, long)
TEAM_INT_REDUCE_SCATTER_DEF_GEN(long long, longlong)
TEAM_FLOAT_REDUCE_SCATTER_DEF_GEN(float, float)
TEAM_FLOAT_REDUCE_SCATTER_DEF_GEN(double, double)

rocshmem_team_t team_reduce_scatter_world_dup;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1, ROCSHMEM_OP T2>
__global__ void TeamReduceScatterTest(int loop, int skip,
                                      long long int *start_time,
                                      long long int *end_time, T1 *s_buf,
                                      T1 *r_buf, size_t size,
                                      [[maybe_unused]] TestType type,
                                      ShmemContextType ctx_type,
                                      rocshmem_team_t team) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  __syncthreads();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && hipThreadIdx_x == 0) {
      start_time[wg_id] = wall_clock64();
    }
    wg_team_reduce_scatter<T1, T2>(ctx, team, r_buf, s_buf, size);
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
template <typename T1, ROCSHMEM_OP T2>
TeamReduceScatterTester<T1, T2>::TeamReduceScatterTester(
    TesterArguments args, std::function<void(T1 &, T1 &)> f1,
    std::function<std::pair<bool, std::string>(const T1 &, const T1 &)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2} {
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  // source buffer: n_pes * max_msg_size elements (on symmetric heap)
  s_buf = (T1 *)alloc_test_buffer(n_pes * max_msg_size * sizeof(T1),
                                  args.local_buf_type);
  // dest buffer: max_msg_size elements (on symmetric heap)
  r_buf = (T1 *)alloc_test_buffer(max_msg_size * sizeof(T1));
}

template <typename T1, ROCSHMEM_OP T2>
TeamReduceScatterTester<T1, T2>::~TeamReduceScatterTester() {
  free_test_buffer(s_buf, args.local_buf_type);
  free_test_buffer(r_buf);
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterTester<T1, T2>::preLaunchKernel() {
  bw_factor = n_pes;

  team_reduce_scatter_world_dup = ROCSHMEM_TEAM_INVALID;
  rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                               &team_reduce_scatter_world_dup);
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterTester<T1, T2>::launchKernel(dim3 gridSize,
                                                    dim3 blockSize, int loop,
                                                    uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(TeamReduceScatterTest<T1, T2>), gridSize,
                     blockSize, shared_bytes, stream, loop, args.skip,
                     start_time, end_time, s_buf, r_buf, size, _type,
                     _shmem_context, team_reduce_scatter_world_dup);

  num_msgs = loop + args.skip;
  num_timed_msgs = loop;
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterTester<T1, T2>::postLaunchKernel() {
  rocshmem_team_destroy(team_reduce_scatter_world_dup);
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterTester<T1, T2>::resetBuffers([[maybe_unused]] uint64_t size) {
  // Each PE sets its entire source buffer to 1 so that a SUM
  // over n_pes PEs yields n_pes in every element of dest.
  for (uint64_t i = 0; i < (uint64_t)n_pes * max_msg_size; i++) {
    s_buf[i] = static_cast<T1>(1);
  }
  for (uint64_t i = 0; i < max_msg_size; i++) {
    r_buf[i] = static_cast<T1>(0);
  }
}

template <typename T1, ROCSHMEM_OP T2>
void TeamReduceScatterTester<T1, T2>::verifyResults(uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    auto r = verify_buf(r_buf[i], (T1)n_pes);
    if (r.first == false) {
      fprintf(stderr, "Data validation error at idx %lu\n", i);
      fprintf(stderr, "%s.\n", r.second.c_str());
      exit(-1);
    }
  }
}
