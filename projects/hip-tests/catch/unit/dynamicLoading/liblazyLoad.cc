/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>  // std::min

#define CHECK_RET_VAL(cmd)                                                                         \
  {                                                                                                \
    hipError_t error = cmd;                                                                        \
    if (error != hipSuccess) {                                                                     \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__,     \
              __LINE__);                                                                           \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  }

__global__ static void addition(float* C, float* A, float* B, size_t N) {
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  size_t stride = hipBlockDim_x * hipGridDim_x;

  for (size_t i = offset; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

static int vectorAddKernelTest() {
  int testResult = 1;
  float *A_d, *B_d, *C_d;
  float *A_h, *B_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);
  static int device = 0;

  CHECK_RET_VAL(hipSetDevice(device));
  hipDeviceProp_t props;
  CHECK_RET_VAL(hipGetDeviceProperties(&props, device));
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  CHECK_RET_VAL(A_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  B_h = reinterpret_cast<float*>(malloc(Nbytes));
  CHECK_RET_VAL(B_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  CHECK_RET_VAL(C_h == nullptr ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
    B_h[i] = 1.618f + i;
  }

  CHECK_RET_VAL(hipMalloc(&A_d, Nbytes));
  CHECK_RET_VAL(hipMalloc(&B_d, Nbytes));
  CHECK_RET_VAL(hipMalloc(&C_d, Nbytes));
  CHECK_RET_VAL(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  CHECK_RET_VAL(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;
  hipLaunchKernelGGL(addition, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, B_d, N);
  CHECK_RET_VAL(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != (A_h[i] + B_h[i])) {
      testResult = 0;
      break;
    }
  }

  CHECK_RET_VAL(hipFree(A_d));
  CHECK_RET_VAL(hipFree(B_d));
  CHECK_RET_VAL(hipFree(C_d));

  free(A_h);
  free(B_h);
  free(C_h);
  return testResult;
}

#include "hip/hip_cooperative_groups.h"

namespace cg = cooperative_groups;

static const uint BufferSizeInDwords = 448 * 1024 * 1024;

__global__ static void test_gws(uint* buf, uint bufSize, int64_t* tmpBuf, int64_t* result) {
  extern __shared__ int64_t tmp[];
  uint offset = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = gridDim.x * blockDim.x;
  cg::grid_group gg = cg::this_grid();

  int64_t sum = 0;

  for (uint i = offset; i < bufSize; i += stride) {
    sum += buf[i];
  }

  tmp[threadIdx.x] = sum;
  __syncthreads();

  if (threadIdx.x == 0) {
    sum = 0;
    for (uint i = 0; i < blockDim.x; i++) {
      sum += tmp[i];
    }
    tmpBuf[blockIdx.x] = sum;
  }

  gg.sync();

  if (offset == 0) {
    for (uint i = 1; i < gridDim.x; ++i) {
      sum += tmpBuf[i];
    }
    *result = sum;
  }
}

static int cooperativeKernelTest() {
  int testResult = 1;
  uint* dA;
  int64_t* dB;
  int64_t* dC;
  int64_t* Ah;

  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);

  if (!deviceProp.cooperativeLaunch) {
    return testResult;
  }

  uint32_t* init = new uint32_t[BufferSizeInDwords];

  for (uint32_t i = 0; i < BufferSizeInDwords; ++i) {
    init[i] = i;
  }
  size_t SIZE = BufferSizeInDwords * sizeof(uint);

  CHECK_RET_VAL(hipMalloc(reinterpret_cast<void**>(&dA), SIZE));
  CHECK_RET_VAL(hipMalloc(reinterpret_cast<void**>(&dC), sizeof(int64_t)));
  CHECK_RET_VAL(hipMemcpy(dA, init, SIZE, hipMemcpyHostToDevice));
  Ah = reinterpret_cast<int64_t*>(malloc(sizeof(int64_t)));

  hipStream_t stream;
  CHECK_RET_VAL(hipStreamCreate(&stream));

  dim3 dimBlock = dim3(1);
  dim3 dimGrid = dim3(1);

  int numBlocks = 0;
  uint workgroups[4] = {32, 64, 128, 256};

  for (uint i = 0; i < 4; ++i) {
    dimBlock.x = workgroups[i];
    /* Calculate the device occupancy to know how many blocks can be
       run concurrently */
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, test_gws, dimBlock.x * dimBlock.y * dimBlock.z, dimBlock.x * sizeof(int64_t));
    dimGrid.x = deviceProp.multiProcessorCount * std::min(numBlocks, 32);
    CHECK_RET_VAL(hipMalloc(reinterpret_cast<void**>(&dB), dimGrid.x * sizeof(int64_t)));

    void* params[4];
    params[0] = reinterpret_cast<void*>(&dA);
    params[1] = (void*)(&BufferSizeInDwords);  // NOLINT
    params[2] = reinterpret_cast<void*>(&dB);
    params[3] = reinterpret_cast<void*>(&dC);

    CHECK_RET_VAL(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_gws), dimGrid, dimBlock,
                                             params, dimBlock.x * sizeof(int64_t), stream));

    CHECK_RET_VAL(hipMemcpy(Ah, dC, sizeof(int64_t), hipMemcpyDeviceToHost));

    if (*Ah != (((int64_t)(BufferSizeInDwords) * (BufferSizeInDwords - 1)) / 2)) {
      CHECK_RET_VAL(hipFree(dB));
      testResult = 0;
      break;

    } else {
      CHECK_RET_VAL(hipFree(dB));
    }
  }
  CHECK_RET_VAL(hipStreamDestroy(stream));
  CHECK_RET_VAL(hipFree(dC));
  CHECK_RET_VAL(hipFree(dA));
  delete[] init;
  free(Ah);
  return testResult;
}

extern "C" int lazyLoad() { return vectorAddKernelTest() & cooperativeKernelTest(); }
