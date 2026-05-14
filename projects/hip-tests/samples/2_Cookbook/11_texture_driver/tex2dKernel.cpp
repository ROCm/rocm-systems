/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip/hip_runtime.h"

extern "C" __global__ void tex2dKernelChar(char* outputData, hipTextureObject_t texObj, int width,
                                           int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<char>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelShort(short* outputData, hipTextureObject_t texObj, int width,
                                            int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<short>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelInt(int* outputData, hipTextureObject_t texObj, int width,
                                          int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<int>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelFloat(float* outputData, hipTextureObject_t texObj, int width,
                                            int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<float>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelChar4(char4* outputData, hipTextureObject_t texObj, int width,
                                            int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<char4>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelShort4(short4* outputData, hipTextureObject_t texObj,
                                             int width, int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<short4>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelInt4(int4* outputData, hipTextureObject_t texObj, int width,
                                           int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<int4>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelFloat4(float4* outputData, hipTextureObject_t texObj,
                                             int width, int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<float4>(texObj, x, y);
#endif
}
