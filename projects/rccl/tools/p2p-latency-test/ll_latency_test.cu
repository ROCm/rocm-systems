/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include <cuda_runtime.h>
#include <iostream> //cerr
#include <cstring>

#define NUM_LOOPS_WARMUP 2000
#define NUM_LOOPS_RUN 10000

#define LL_MAX_THREADS 256
#define LL_MAX_LINES 1000

union LLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

#define HIP_IPC_MEM_MIN_SIZE (LL_MAX_THREADS*LL_MAX_LINES*sizeof(LLFifoLine))

__device__ void storeLL(union LLFifoLine* dst, uint64_t val, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
}

#define LL_SPINS_BEFORE_CHECK_ABORT 1000000

inline __device__ int checkAbort(int &spins, uint32_t* abortFlag) {
  uint32_t abort = 0;
  spins++;
  if (spins == LL_SPINS_BEFORE_CHECK_ABORT) {
    abort = *((volatile uint32_t*)abortFlag);
    spins = 0;
  }
  return abort;
}

__device__ uint64_t readLL(union LLFifoLine* src, uint32_t flag, uint32_t* abortFlag) {
  int spins = 0;

  union LLFifoLine i4;
  do {
      asm("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(i4.data1), "=r"(i4.flag1), "=r"(i4.data2), "=r"(i4.flag2) : "l"(&src->i4));
      if (checkAbort(spins, abortFlag)) break;
  } while ((i4.flag1 != flag) || (i4.flag2 != flag));
  uint64_t val64 = (uint64_t)(i4.data1) + (((uint64_t)i4.data2) << 32);
  return val64;
}


__global__ void PingKernel(LLFifoLine* local_flag, LLFifoLine* remote_flag, uint64_t* time_delta, uint32_t* abortFlag) {
  int tid = threadIdx.x;
  #pragma unroll
  for (uint32_t i = 1; i <= NUM_LOOPS_WARMUP; i++) {
    storeLL(remote_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, i);
    while (readLL(local_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, abortFlag) != i);
  }
  uint64_t start_time, end_time;
  if (tid == 0) start_time = clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP + 1; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    storeLL(remote_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, i);
    while (readLL(local_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, abortFlag) != i);
  }
  __syncthreads();
  if (tid == 0) end_time = clock64();
  if (tid == 0) *time_delta = end_time - start_time;
}

__global__ void PongKernel(LLFifoLine* local_flag, LLFifoLine* remote_flag, uint64_t* time_delta, uint32_t* abortFlag) {
  int tid = threadIdx.x;
  #pragma unroll
  for (uint32_t i = 1; i <= NUM_LOOPS_WARMUP; i++) {
    while (readLL(local_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, abortFlag) != i);
    storeLL(remote_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, i);
  }
  uint64_t start_time, end_time;
  if (tid == 0) start_time = clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP + 1; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    while (readLL(local_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, abortFlag) != i);
    storeLL(remote_flag+tid+(i%LL_MAX_LINES)*LL_MAX_THREADS, i, i);
  }
  __syncthreads();
  if (tid == 0) end_time = clock64();
  if (tid == 0) *time_delta = end_time - start_time;
}

#define HIPCHECK(cmd)                                                          \
do {                                                                           \
  cudaError_t error = (cmd);                                                    \
  if (error != cudaSuccess)                                                     \
  {                                                                            \
    std::cerr << "Encountered HIP error (" << error << ") at line "            \
              << __LINE__ << " in file " << __FILE__ << "\n";                  \
    exit(-1);                                                                  \
  }                                                                            \
} while (0)

int main(int argc, char** argv) {
  cudaStream_t stream[2];
  int device_id[2];
  cudaDeviceProp prop[2];

  if (argc != 3) {
    fprintf(stderr, "Usage: ./ll_latency_test ping_dev_id pong_dev_id\n");
    return -1;
  }
  device_id[0] = atoi(argv[1]);
  device_id[1] = atoi(argv[2]);

  fprintf(stdout, "Using devices %d %d\n", device_id[0], device_id[1]);

  LLFifoLine *flag[2];
  uint64_t *time_delta[2];
  uint32_t *abortFlag[2];

  HIPCHECK(cudaSetDevice(device_id[0]));
  HIPCHECK(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
  HIPCHECK(cudaDeviceEnablePeerAccess(device_id[1], 0));
  HIPCHECK(cudaGetDeviceProperties(&prop[0], device_id[0]));
  HIPCHECK(cudaMalloc((void**)&flag[0], HIP_IPC_MEM_MIN_SIZE));
  HIPCHECK(cudaHostAlloc ((void**)&time_delta[0], sizeof(uint64_t), cudaHostAllocDefault));
  HIPCHECK(cudaMalloc((void**)&abortFlag[0], sizeof(uint32_t)));
  HIPCHECK(cudaMemsetAsync(flag[0], 0, HIP_IPC_MEM_MIN_SIZE, stream[0]));
  HIPCHECK(cudaMemsetAsync(abortFlag[0], 0, sizeof(uint32_t), stream[0]));
  HIPCHECK(cudaStreamSynchronize(stream[0]));

  HIPCHECK(cudaSetDevice(device_id[1]));
  HIPCHECK(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));
  HIPCHECK(cudaDeviceEnablePeerAccess(device_id[0], 0));
  HIPCHECK(cudaGetDeviceProperties(&prop[1], device_id[1]));
  HIPCHECK(cudaMalloc((void**)&flag[1], HIP_IPC_MEM_MIN_SIZE));
  HIPCHECK(cudaHostAlloc((void**)&time_delta[1], sizeof(uint64_t), cudaHostAllocDefault));
  HIPCHECK(cudaMalloc((void**)&abortFlag[1], sizeof(uint32_t)));
  HIPCHECK(cudaMemsetAsync(flag[1], 0, HIP_IPC_MEM_MIN_SIZE, stream[1]));
  HIPCHECK(cudaMemsetAsync(abortFlag[1], 0, sizeof(uint32_t), stream[0]));
  HIPCHECK(cudaStreamSynchronize(stream[1]));

  HIPCHECK(cudaSetDevice(device_id[0]));
  PingKernel<<<1, LL_MAX_THREADS, 0, stream[0]>>>(flag[0], flag[1], time_delta[0], abortFlag[0]);

  HIPCHECK(cudaSetDevice(device_id[1]));
  PongKernel<<<1, LL_MAX_THREADS, 0, stream[1]>>>(flag[1], flag[0], time_delta[1], abortFlag[1]);

  double gpu_rtc_freq;

  HIPCHECK(cudaStreamSynchronize(stream[0]));
  gpu_rtc_freq = prop[0].clockRate*1.0E3;
  fprintf(stdout, "One-way latency in us: %g\n", double(*time_delta[0]) * 1e6 / NUM_LOOPS_RUN / gpu_rtc_freq / 2);

  HIPCHECK(cudaStreamSynchronize(stream[1]));
  gpu_rtc_freq = prop[1].clockRate*1.0E3;
  fprintf(stdout, "One-way latency in us: %g\n", double(*time_delta[1]) * 1e6 / NUM_LOOPS_RUN / gpu_rtc_freq / 2);

  HIPCHECK(cudaFree(flag[0]));
  HIPCHECK(cudaFreeHost(time_delta[0]));
  HIPCHECK(cudaFree(abortFlag[0]));
  HIPCHECK(cudaFree(flag[1]));
  HIPCHECK(cudaFreeHost(time_delta[1]));
  HIPCHECK(cudaFree(abortFlag[1]));
  return 0;
}
