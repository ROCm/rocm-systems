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
template <typename T>
__device__ void alltoall_wave([[maybe_unused]] rocshmem_ctx_t ctx, [[maybe_unused]] rocshmem_team_t team,
                              [[maybe_unused]] T *dest, [[maybe_unused]] const T *source, [[maybe_unused]] int nelems) {
  return;
}

/* Define templates to call rocSHMEM */
#define ALLTOALLWAVEGEN(T, TNAME)                                              \
  template <>                                                                  \
  __device__ void alltoall_wave<T>(rocshmem_ctx_t ctx, rocshmem_team_t team,   \
                                 T * dest, const T *source, int nelems) {      \
    rocshmem_ctx_##TNAME##_alltoall_wave(ctx, team, dest, source, nelems);}

ALLTOALLWAVEGEN(float, float)
ALLTOALLWAVEGEN(double, double)

ALLTOALLWAVEGEN(short, short)
ALLTOALLWAVEGEN(int, int)
ALLTOALLWAVEGEN(long, long)
ALLTOALLWAVEGEN(long long, longlong)
ALLTOALLWAVEGEN(char, char)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1>
__global__ void TeamAlltoallWaveTest(int loop, int skip, long long int *start_time,
                                 long long int *end_time, T1 *source_buf,
                                 T1 *dest_buf, int num_elems,
                                 ShmemContextType ctx_type,
                                 rocshmem_team_t *teams, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int t_id = get_flat_block_id();
  int wf_id = t_id / wf_size;
  int wf_per_wg = (get_flat_block_size() - 1) / wf_size + 1;
  int flat_wf_id = wf_id + wg_id * wf_per_wg;

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int n_pes = rocshmem_ctx_n_pes(ctx);

  source_buf += flat_wf_id * n_pes * num_elems;
  dest_buf += flat_wf_id * n_pes * num_elems;

  __syncthreads();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && t_id % wf_size == 0) {
      start_time[flat_wf_id] = wall_clock64();
    }
    alltoall_wave<T1>(ctx, teams[flat_wf_id], dest_buf, source_buf, num_elems);
  }

  __syncthreads();

  if (t_id % wf_size == 0) {
    end_time[flat_wf_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T1>
AlltoallWaveTester<T1>::AlltoallWaveTester(TesterArguments args)
    : Tester(args){
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  bw_factor = n_pes;
  size_factor = n_pes;

  // Number of elements per work group
  size_t num_elems_wg = size_factor * (max_msg_size / sizeof(T1));
  // Total number of elements in the GPU kernel
  size_t total_elems = num_elems_wg * args.num_wgs * num_warps;
  size_t buff_size = total_elems * sizeof(T1);

  source_buf = (T1 *)alloc_test_buffer(buff_size, args.local_buf_type);
  dest_buf = (T1 *)alloc_test_buffer(buff_size);

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  if (num_teams < num_warps * args.num_wgs){
    printf("not enough teams for each wavefront, try increasing ROCSHMEM_MAX_NUM_TEAMS\n");
    exit(0);
  }

  CHECK_HIP(hipMalloc(&alltoall_wave_world_dup,
                      sizeof(rocshmem_team_t) * num_teams));
}

template <typename T1>
AlltoallWaveTester<T1>::~AlltoallWaveTester() {
  free_test_buffer(source_buf, args.local_buf_type);
  free_test_buffer(dest_buf);
  CHECK_HIP(hipFree(alltoall_wave_world_dup));
}

template <typename T1>
void AlltoallWaveTester<T1>::preLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    alltoall_wave_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                 &alltoall_wave_world_dup[team_i]);
    if (alltoall_wave_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      std::cout << "Team " << team_i << " is invalid!" << std::endl;
      abort();
    }
  }
}

template <typename T1>
void AlltoallWaveTester<T1>::launchKernel(dim3 gridSize, dim3 blockSize,
                                          int loop, size_t size) {
  size_t shared_bytes = 0;

  int num_elems = size / sizeof(T1);

  hipLaunchKernelGGL(TeamAlltoallWaveTest<T1>, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     source_buf, dest_buf, num_elems, _shmem_context,
                     alltoall_wave_world_dup, wf_size);

  num_msgs = (loop + args.skip) * gridSize.x * num_warps;
  num_timed_msgs = loop * gridSize.x * num_warps;
}

template <typename T1>
void AlltoallWaveTester<T1>::postLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(alltoall_wave_world_dup[team_i]);
  }
}

template <typename T1>
void AlltoallWaveTester<T1>::resetBuffers(size_t size) {

  int num_elems = size / sizeof(T1);
  int buff_size = num_elems * sizeof(T1) * args.num_wgs * n_pes;
  int idx = 0;
  int total_wfs = args.num_wgs * num_warps;

  for(unsigned int wf_id = 0; wf_id < total_wfs; wf_id++) {
    for(int pe = 0; pe < n_pes; pe++) {
      for(unsigned int i = 0; i < static_cast<unsigned int>(num_elems); i++) {
        idx = (wf_id * n_pes + pe) * num_elems + i;
        if constexpr (std::is_floating_point<T1>::value) {
          source_buf[idx] = static_cast<T1>(3.14 + my_pe + pe + wf_id);
        }
        else {
          source_buf[idx] = static_cast<T1>('a' + my_pe + pe + wf_id);
        }
      }
    }
  }

  memset(dest_buf, -1, buff_size);
}

template <typename T1>
void AlltoallWaveTester<T1>::verifyResults(size_t size) {
  int num_elems = size / sizeof(T1);
  int idx = 0;
  int total_wfs = args.num_wgs * num_warps;

  for(int wf_id = 0; wf_id < total_wfs; wf_id++) {
    for(int pe = 0; pe < n_pes; pe++) {
      for(int i = 0; i < num_elems; i++) {
        idx = (wf_id * n_pes + pe) * num_elems + i;
        if (dest_buf[idx] != source_buf[idx]) {
          std::cerr << "Data validation error at idx " << idx << std::endl;
          std::cerr << "PE " << my_pe << " Got " << dest_buf[idx]
          << ", Expected " << source_buf[idx] << std::endl;
          exit(-1);
        }
      }
    }
  }
}
