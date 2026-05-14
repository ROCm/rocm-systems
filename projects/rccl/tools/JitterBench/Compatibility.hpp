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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if defined(__NVCC__)

#include <cuda_runtime.h>

// ROCm specific
#define wall_clock64                                       clock64
#define gcnArchName                                        name

// Datatypes
#define hipDeviceProp_t                                    cudaDeviceProp
#define hipError_t                                         cudaError_t
#define hipEvent_t                                         cudaEvent_t
#define hipStream_t                                        cudaStream_t

// Enumerations
#define hipDeviceAttributeClockRate                        cudaDevAttrClockRate
#define hipDeviceAttributeMaxSharedMemoryPerMultiprocessor cudaDevAttrMaxSharedMemoryPerMultiprocessor
#define hipDeviceAttributeMultiprocessorCount              cudaDevAttrMultiProcessorCount
#define hipErrorPeerAccessAlreadyEnabled                   cudaErrorPeerAccessAlreadyEnabled
#define hipFuncCachePreferShared                           cudaFuncCachePreferShared
#define hipMemcpyDefault                                   cudaMemcpyDefault
#define hipMemcpyDeviceToHost                              cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice                              cudaMemcpyHostToDevice
#define hipSuccess                                         cudaSuccess

// Functions
#define hipDeviceCanAccessPeer                             cudaDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess                          cudaDeviceEnablePeerAccess
#define hipDeviceGetAttribute                              cudaDeviceGetAttribute
#define hipDeviceGetPCIBusId                               cudaDeviceGetPCIBusId
#define hipDeviceSetCacheConfig                            cudaDeviceSetCacheConfig
#define hipDeviceSynchronize                               cudaDeviceSynchronize
#define hipEventCreate                                     cudaEventCreate
#define hipEventDestroy                                    cudaEventDestroy
#define hipEventElapsedTime                                cudaEventElapsedTime
#define hipEventRecord                                     cudaEventRecord
#define hipFree                                            cudaFree
#define hipGetDeviceCount                                  cudaGetDeviceCount
#define hipGetDeviceProperties                             cudaGetDeviceProperties
#define hipGetErrorString                                  cudaGetErrorString
#define hipHostFree                                        cudaFreeHost
#define hipHostMalloc                                      cudaMallocHost
#define hipMalloc                                          cudaMalloc
#define hipMemcpy                                          cudaMemcpy
#define hipMemcpyAsync                                     cudaMemcpyAsync
#define hipMemset                                          cudaMemset
#define hipMemsetAsync                                     cudaMemsetAsync
#define hipSetDevice                                       cudaSetDevice
#define hipStreamCreate                                    cudaStreamCreate
#define hipStreamDestroy                                   cudaStreamDestroy
#define hipStreamSynchronize                               cudaStreamSynchronize

#else

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#endif
