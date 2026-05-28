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
#include <hip/hip_runtime.h>
#include <iostream> //cerr
#include <cstring>

#define HIP_IPC_MEM_MIN_SIZE 2097152UL

#define NUM_LOOPS_WARMUP 2000
#define NUM_LOOPS_RUN 10000

#define PING_MODE 0
#define PONG_MODE 1

__global__ void PingKernel(uint64_t* local_flag, uint64_t* remote_flag, uint64_t* time_delta) {
  #pragma unroll
  for (uint32_t i = 1; i <= NUM_LOOPS_WARMUP; i++) {
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
  }
  uint64_t start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP + 1; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
  }
  uint64_t end_time = wall_clock64();
  *time_delta = end_time - start_time;
}

__global__ void PongKernel(uint64_t* local_flag, uint64_t* remote_flag, uint64_t* time_delta) {
  #pragma unroll
  for (uint32_t i = 1; i <= NUM_LOOPS_WARMUP; i++) {
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
  }
  uint64_t start_time = wall_clock64();
  #pragma unroll
  for (uint32_t i = NUM_LOOPS_WARMUP + 1; i <= NUM_LOOPS_WARMUP + NUM_LOOPS_RUN; i++) {
    while (__atomic_load_n(local_flag, __ATOMIC_RELAXED) != i);
    __atomic_store_n(remote_flag, i, __ATOMIC_RELAXED);
  }
  uint64_t end_time = wall_clock64();
  *time_delta = end_time - start_time;
}

#define HIPCHECK(cmd)                                                          \
do {                                                                           \
  hipError_t error = (cmd);                                                    \
  if (error != hipSuccess)                                                     \
  {                                                                            \
    std::cerr << "Encountered HIP error (" << error << ") at line "            \
              << __LINE__ << " in file " << __FILE__ << "\n";                  \
    exit(-1);                                                                  \
  }                                                                            \
} while (0)

int main(int argc, char** argv) {
  hipStream_t stream[2];
  hipError_t err = hipSuccess;
  int device_id[2];
  hipDeviceProp_t prop[2];

  if (argc != 3) {
    fprintf(stderr, "Usage: ./p2p_latency_test ping_dev_id pong_dev_id\n");
    return -1;
  }
  device_id[0] = atoi(argv[1]);
  device_id[1] = atoi(argv[2]);

  fprintf(stdout, "Using devices %d %d\n", device_id[0], device_id[1]);

  uint64_t *flag[2];
  uint64_t *time_delta[2];

  HIPCHECK(hipSetDevice(device_id[0]));
  HIPCHECK(hipStreamCreateWithFlags(&stream[0], hipStreamNonBlocking));
  HIPCHECK(hipDeviceEnablePeerAccess(device_id[1], 0));
  HIPCHECK(hipGetDeviceProperties(&prop[0], device_id[0]));
  HIPCHECK(hipExtMallocWithFlags((void**)&flag[0], HIP_IPC_MEM_MIN_SIZE, (strncmp(prop[0].gcnArchName, "gfx942", 6) == 0 || strncmp(prop[0].gcnArchName, "gfx950", 6) == 0) ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
  HIPCHECK(hipMalloc((void**)&time_delta[0], HIP_IPC_MEM_MIN_SIZE));
  HIPCHECK(hipMemsetAsync(flag[0], 0, HIP_IPC_MEM_MIN_SIZE, stream[0]));
  HIPCHECK(hipStreamSynchronize(stream[0]));

  HIPCHECK(hipSetDevice(device_id[1]));
  HIPCHECK(hipStreamCreateWithFlags(&stream[1], hipStreamNonBlocking));
  HIPCHECK(hipDeviceEnablePeerAccess(device_id[0], 0));
  HIPCHECK(hipGetDeviceProperties(&prop[1], device_id[1]));
  HIPCHECK(hipExtMallocWithFlags((void**)&flag[1], HIP_IPC_MEM_MIN_SIZE, (strncmp(prop[1].gcnArchName, "gfx942", 6) == 0 || strncmp(prop[1].gcnArchName, "gfx950", 6) == 0) ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
  HIPCHECK(hipMalloc((void**)&time_delta[1], HIP_IPC_MEM_MIN_SIZE));
  HIPCHECK(hipMemsetAsync(flag[1], 0, HIP_IPC_MEM_MIN_SIZE, stream[1]));
  HIPCHECK(hipStreamSynchronize(stream[1]));

  HIPCHECK(hipSetDevice(device_id[0]));
  PingKernel<<<1, 1, 0, stream[0]>>>(flag[0], flag[1], time_delta[0]);

  HIPCHECK(hipSetDevice(device_id[1]));
  PongKernel<<<1, 1, 0, stream[1]>>>(flag[1], flag[0], time_delta[1]);

  double vega_gpu_rtc_freq;

  HIPCHECK(hipStreamSynchronize(stream[0]));
  vega_gpu_rtc_freq = (strncmp(prop[0].gcnArchName, "gfx942", 6) == 0 || strncmp(prop[0].gcnArchName, "gfx950", 6) == 0) ? 1.0E8 : 2.5E7;
  fprintf(stdout, "One-way latency in us: %g\n", double(*time_delta[0]) * 1e6 / NUM_LOOPS_RUN / vega_gpu_rtc_freq / 2);

  HIPCHECK(hipStreamSynchronize(stream[1]));
  vega_gpu_rtc_freq = (strncmp(prop[1].gcnArchName, "gfx942", 6) == 0 || strncmp(prop[1].gcnArchName, "gfx950", 6) == 0) ? 1.0E8 : 2.5E7;
  fprintf(stdout, "One-way latency in us: %g\n", double(*time_delta[1]) * 1e6 / NUM_LOOPS_RUN / vega_gpu_rtc_freq / 2);

  HIPCHECK(hipFree(flag[0]));
  HIPCHECK(hipFree(time_delta[0]));
  HIPCHECK(hipFree(flag[1]));
  HIPCHECK(hipFree(time_delta[1]));
  return 0;
}
