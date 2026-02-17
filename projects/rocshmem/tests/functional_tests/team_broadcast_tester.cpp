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
__device__ void wg_team_broadcast(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                  T *dest, const T *source, int nelem,
                                  int pe_root) {
  return;
}

/* Define templates to call ROCSHMEM */
#define TEAM_BROADCAST_DEF_GEN(T, TNAME)                                      \
  template <>                                                                 \
  __device__ void wg_team_broadcast<T>(                                       \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,    \
      int nelem, int pe_root) {                                               \
    rocshmem_ctx_##TNAME##_broadcast_wg(ctx, team, dest, source, nelem,       \
                                         pe_root);                            \
  }

TEAM_BROADCAST_DEF_GEN(float, float)
TEAM_BROADCAST_DEF_GEN(double, double)
TEAM_BROADCAST_DEF_GEN(char, char)
// TEAM_BROADCAST_DEF_GEN(long double, longdouble)
TEAM_BROADCAST_DEF_GEN(signed char, schar)
TEAM_BROADCAST_DEF_GEN(short, short)
TEAM_BROADCAST_DEF_GEN(int, int)
TEAM_BROADCAST_DEF_GEN(long, long)
TEAM_BROADCAST_DEF_GEN(long long, longlong)
TEAM_BROADCAST_DEF_GEN(unsigned char, uchar)
TEAM_BROADCAST_DEF_GEN(unsigned short, ushort)
TEAM_BROADCAST_DEF_GEN(unsigned int, uint)
TEAM_BROADCAST_DEF_GEN(unsigned long, ulong)
TEAM_BROADCAST_DEF_GEN(unsigned long long, ulonglong)

rocshmem_team_t team_bcast_world_dup;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1>
__global__ void TeamBroadcastTest(int loop, int skip, long long int *start_time,
                                  long long int *end_time, T1 *source_buf,
                                  T1 *dest_buf, int size,
                                  ShmemContextType ctx_type,
                                  rocshmem_team_t *teams) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_team_create_ctx(teams[wg_id], ctx_type, &ctx);

  int n_pes = rocshmem_ctx_n_pes(ctx);
  source_buf += wg_id * size;
  dest_buf += wg_id * size;

  __syncthreads();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && hipThreadIdx_x == 0) {
      start_time[wg_id] = wall_clock64();
    }

    wg_team_broadcast<T1>(ctx, teams[wg_id],
                          dest_buf,         // T* dest
                          source_buf,       // const T* source
                          size,             // int nelement
                          0);               // int PE_root
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
template <typename T1>
TeamBroadcastTester<T1>::TeamBroadcastTester(TesterArguments args)
    : Tester(args){
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  // Total number of elements in src buffer
  int total_elems = (max_msg_size / sizeof(T1)) * args.num_wgs ;
  int buff_size = total_elems * sizeof(T1);

  source_buf = (T1 *)rocshmem_malloc(buff_size);
  dest_buf = (T1 *)rocshmem_malloc(buff_size);

  if (source_buf == nullptr || dest_buf == nullptr) {
    std::cout << "Error allocating memory from symmetric heap" << std::endl;
    std::cout << "source: " << source_buf << ", dest: " << dest_buf << std::endl;
    rocshmem_global_exit(1);
  }

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  CHECK_HIP(hipMalloc(&team_bcast_world_dup,
                      sizeof(rocshmem_team_t) * num_teams));
}

template <typename T1>
TeamBroadcastTester<T1>::~TeamBroadcastTester() {
  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
  CHECK_HIP(hipFree(team_bcast_world_dup));
}

template <typename T1>
void TeamBroadcastTester<T1>::preLaunchKernel() {
  bw_factor = n_pes;

  for (int team_i = 0; team_i < num_teams; team_i++) {
    team_bcast_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                 &team_bcast_world_dup[team_i]);
    if (team_bcast_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      printf("Team %d is invalid!\n", team_i);
      abort();
    }
  }
}

template <typename T1>
void TeamBroadcastTester<T1>::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, size_t size) {
  size_t shared_bytes = 0;

  int num_elems = size / sizeof(T1);

  hipLaunchKernelGGL(TeamBroadcastTest<T1>, gridSize, blockSize,
                     shared_bytes, stream, loop, args.skip,
                     start_time, end_time, source_buf, dest_buf,
                     num_elems, _shmem_context, team_bcast_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

template <typename T1>
void TeamBroadcastTester<T1>::postLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(team_bcast_world_dup[team_i]);
  }
}

template <typename T1>
void TeamBroadcastTester<T1>::resetBuffers(size_t size) {

  int num_elems = size / sizeof(T1);
  int buff_size = num_elems * sizeof(T1) * args.num_wgs;
  int idx = 0;

  for (int wg_id = 0; wg_id < args.num_wgs; wg_id++) {
    for (int i = 0; i < num_elems; i++) {
      idx = wg_id * num_elems + i;
      if constexpr (std::is_same<T1, char>::value ||
                    std::is_same<T1, signed char>::value ||
                    std::is_same<T1, unsigned char>::value) {
        source_buf[idx] = static_cast<T1>('a' + n_pes + wg_id);
        dest_buf[idx] = static_cast<T1>('a' + wg_id);
      }
      else if constexpr (std::is_floating_point<T1>::value) {
        source_buf[idx] = static_cast<T1>(3.14 + n_pes + wg_id);
        dest_buf[idx] = static_cast<T1>(3.14 + wg_id);
      }
      else if constexpr (std::is_integral<T1>::value) {
        source_buf[idx] = static_cast<T1>(n_pes + wg_id);
        dest_buf[idx] = static_cast<T1>(wg_id);
      }
    }
  }
}

template <typename T1>
void TeamBroadcastTester<T1>::verifyResults(size_t size) {

  int num_elems = size / sizeof(T1);
  int idx = 0;
  T1 expected;

  /**
   * The verification routine here requires that the
   * PE_root value is 0 which denotes that the
   * sending processing element is rank 0.
   *
   * The difference in expected values arises from
   * the specification for broadcast where the
   * PE_root processing element does not copy the
   * contents from its own source to dest during
   * the broadcast.
   */
  for (int wg_id = 0; wg_id < args.num_wgs; wg_id++) {
    for (int i = 0; i < num_elems; i++) {
      idx = wg_id * num_elems + i;
      if constexpr (std::is_same<T1, char>::value ||
                    std::is_same<T1, signed char>::value ||
                    std::is_same<T1, unsigned char>::value) {
        expected = static_cast<T1>('a' + wg_id + (my_pe ? n_pes : 0));
      }
      else if constexpr (std::is_floating_point<T1>::value) {
        expected = static_cast<T1>(3.14 + wg_id + (my_pe ? n_pes : 0));
      }
      else if constexpr (std::is_integral<T1>::value) {
        expected = static_cast<T1>(wg_id + (my_pe ? n_pes : 0));
      }
      if (dest_buf[idx] != expected) {
        std::cerr << "Data validation error at idx " << idx << std::endl;
        std::cerr << "PE " << my_pe << " Got " << dest_buf[idx]
        << ", Expected " << expected << std::endl;
        exit(-1);
      }
    }
  }
}
