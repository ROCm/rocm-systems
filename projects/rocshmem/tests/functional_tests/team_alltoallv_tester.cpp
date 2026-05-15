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

#include "team_alltoallv_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* Declare the template with a generic implementation */
template <typename T>
__device__ void wg_team_alltoallv(rocshmem_team_t team,
                                  T *dest,
                                  const size_t dest_nelems[],
                                  const size_t dest_displs[],
                                  T *source,
                                  const size_t source_nelems[],
                                  const size_t source_displs[]) {
  return;
}

/* Define templates to call rocSHMEM */
#define TEAM_ALLTOALLV_DEF_GEN(T, TNAME)                                   \
  template <>                                                              \
  __device__ void wg_team_alltoallv(rocshmem_team_t team,                  \
                                    T *dest,                               \
                                    const size_t dest_nelems[],            \
                                    const size_t dest_displs[],            \
                                    T *source,                             \
                                    const size_t source_nelems[],          \
                                    const size_t source_displs[]) {        \
    rocshmem_##TNAME##_alltoallv_wg(team,                                  \
                                    dest, dest_nelems, dest_displs,        \
                                    source, source_nelems, source_displs); \
}

TEAM_ALLTOALLV_DEF_GEN(float, float)
TEAM_ALLTOALLV_DEF_GEN(double, double)
TEAM_ALLTOALLV_DEF_GEN(char, char)
TEAM_ALLTOALLV_DEF_GEN(signed char, schar)
TEAM_ALLTOALLV_DEF_GEN(short, short)
TEAM_ALLTOALLV_DEF_GEN(int, int)
TEAM_ALLTOALLV_DEF_GEN(long, long)
TEAM_ALLTOALLV_DEF_GEN(long long, longlong)
TEAM_ALLTOALLV_DEF_GEN(unsigned char, uchar)
TEAM_ALLTOALLV_DEF_GEN(unsigned short, ushort)
TEAM_ALLTOALLV_DEF_GEN(unsigned int, uint)
TEAM_ALLTOALLV_DEF_GEN(unsigned long, ulong)
TEAM_ALLTOALLV_DEF_GEN(unsigned long long, ulonglong)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/

template <typename T1>
__global__ void TeamAlltoallvTest(int loop, int skip,
                                  long long int *start_time,
                                  long long int *end_time,
                                  T1 *dest,
                                  const size_t dest_nelems[],
                                  const size_t dest_displs[],
                                  T1 *source,
                                  const size_t source_nelems[],
                                  const size_t source_displs[],
                                  ShmemContextType ctx_type,
                                  rocshmem_team_t *teams) {

  __syncthreads();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && hipThreadIdx_x == 0) {
      start_time[0] = wall_clock64();
    }
    wg_team_alltoallv<T1>(teams[0],
                          dest, dest_nelems, dest_displs,
                          source, source_nelems, source_displs);
  }

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[0] = wall_clock64();
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T1>
TeamAlltoallvTester<T1>::TeamAlltoallvTester(TesterArguments args)
    : Tester(args){
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  if (args.num_wgs > 1) {
    printf("Alltoallv only supports a single workgroup.\n");
    rocshmem_global_exit(1);
  }

  // Number of elements per work group
  int num_elems_wg = (args.max_msg_size / sizeof(T1)) * n_pes;

  // Total number of elements in the GPU kernel
  int total_elems = num_elems_wg;
  int buff_size   = total_elems * sizeof(T1);

  source_buf = (T1 *)rocshmem_malloc(buff_size);
  dest_buf   = (T1 *)rocshmem_malloc(buff_size);

  CHECK_HIP(hipMalloc(&source_displs, n_pes * sizeof(size_t)));
  CHECK_HIP(hipMalloc(&dest_displs  , n_pes * sizeof(size_t)));
  CHECK_HIP(hipMalloc(&source_nelems, n_pes * sizeof(size_t)));
  CHECK_HIP(hipMalloc(&dest_nelems  , n_pes * sizeof(size_t)));

  if (source_buf == nullptr    ||
      dest_buf == nullptr      ||
      source_displs == nullptr ||
      dest_displs == nullptr   ||
      source_nelems == nullptr ||
      dest_nelems == nullptr) {

    printf("Error allocating memory from symmetric heap.\n"
           "Source %p, Source Displacements %p, Source Elems %p\n"
           "Dest %p, Dest Displacements %p, Dest Elems %p\n",
           source_buf, source_displs, source_nelems,
           dest_buf, dest_displs, dest_nelems);
    rocshmem_global_exit(1);
  }

  char* value = nullptr;

  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  CHECK_HIP(hipMalloc(&team_alltoallv_world_dup,
                      sizeof(rocshmem_team_t) * num_teams));
}

template <typename T1>
TeamAlltoallvTester<T1>::~TeamAlltoallvTester() {
  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
  CHECK_HIP(hipFree(source_displs));
  CHECK_HIP(hipFree(dest_displs));
  CHECK_HIP(hipFree(source_nelems));
  CHECK_HIP(hipFree(dest_nelems));
  CHECK_HIP(hipFree(team_alltoallv_world_dup));
}

template <typename T1>
void TeamAlltoallvTester<T1>::preLaunchKernel() {
  bw_factor = n_pes;

  for (int team_i = 0; team_i < num_teams; team_i++) {
    team_alltoallv_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                 &team_alltoallv_world_dup[team_i]);
    if (team_alltoallv_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      std::cout << "Team " << team_i << " is invalid!" << std::endl;
      abort();
    }
  }
}

template <typename T1>
void TeamAlltoallvTester<T1>::launchKernel(dim3 gridSize, dim3 blockSize,
                                          int loop, size_t size) {
  size_t shared_bytes = 0;
  int num_elems = size / sizeof(T1);
  size_t disp = 0;

  for (int i = 0; i < n_pes; i++) {
    source_nelems[i] = num_elems;
    dest_nelems[i]   = num_elems;
    source_displs[i] = disp;
    dest_displs[i]   = disp;
    disp += num_elems;
  }

  hipLaunchKernelGGL(TeamAlltoallvTest<T1>, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     dest_buf, dest_nelems, dest_displs,
                     source_buf, source_nelems, source_displs,
                     _shmem_context,
                     team_alltoallv_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

template <typename T1>
void TeamAlltoallvTester<T1>::postLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(team_alltoallv_world_dup[team_i]);
  }
}

template <typename T1>
void TeamAlltoallvTester<T1>::resetBuffers(size_t size) {
  int num_elems = size / sizeof(T1);
  int buff_size = num_elems * sizeof(T1) * n_pes;
  int idx = 0;

  for(int pe = 0; pe < n_pes; pe++) {
    for(int i = 0; i < num_elems; i++) {
      idx = pe * num_elems + i;
      if constexpr (std::is_same<T1, char>::value ||
                    std::is_same<T1, signed char>::value ||
                    std::is_same<T1, unsigned char>::value) {
        source_buf[idx] = static_cast<T1>('a' + my_pe + pe);
      }
      else if constexpr (std::is_floating_point<T1>::value) {
        source_buf[idx] = static_cast<T1>(3.14 + my_pe + pe);
      }
      else if constexpr (std::is_integral<T1>::value) {
        source_buf[idx] = static_cast<T1>(my_pe + pe);
      }
    }
  }

  memset(dest_buf, -1, buff_size);
}

template <typename T1>
void TeamAlltoallvTester<T1>::verifyResults(size_t size) {
  int num_elems = size / sizeof(T1);

  for(int pe = 0; pe < n_pes; pe++) {
    T1* dst = (T1*) ((char*)dest_buf + (dest_displs[pe] * sizeof(T1)));
    T1* src = (T1*) &source_buf[pe * num_elems];

    for(int i = 0; i < dest_nelems[pe]; i++) {
      if (dst[i] != src[i]) {
        std::cerr << "Data validation error at idx " << i << std::endl;
        std::cerr << "PE " << my_pe << " Got " << dest_buf[i]
        << ", Expected " << source_buf[i] << std::endl;
        exit(-1);
      }
    }
  }
}
