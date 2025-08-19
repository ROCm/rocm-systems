/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "hip/hip_runtime.h"

extern "C" __global__ void tex2dKernelChar(char* outputData,hipTextureObject_t texObj, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<char>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelShort(short* outputData,hipTextureObject_t texObj, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<short>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelInt(int* outputData,hipTextureObject_t texObj ,int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<int>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelFloat(float* outputData,hipTextureObject_t texObj, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<float>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelChar4(char4* outputData,hipTextureObject_t texObj, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<char4>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelShort4(short4* outputData,hipTextureObject_t texObj, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<short4>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelInt4(int4* outputData,hipTextureObject_t texObj, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<int4>(texObj, x, y);
#endif
}

extern "C" __global__ void tex2dKernelFloat4(float4* outputData,hipTextureObject_t texObj, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    outputData[y * width + x] = tex2D<float4>(texObj, x, y);
#endif
}
