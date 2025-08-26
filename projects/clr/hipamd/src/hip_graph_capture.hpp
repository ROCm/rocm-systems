/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once
namespace hip {
// forward declaration of capture methods
hipError_t capturehipLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                  dim3& blockDim, void**& args, size_t& sharedMemBytes);

hipError_t capturehipExtModuleLaunchKernel(hipStream_t& stream, hipFunction_t& f,
                                           uint32_t& globalWorkSizeX, uint32_t& globalWorkSizeY,
                                           uint32_t& globalWorkSizeZ, uint32_t& localWorkSizeX,
                                           uint32_t& localWorkSizeY, uint32_t& localWorkSizeZ,
                                           size_t& sharedMemBytes, void**& kernelParams,
                                           void**& extra, hipEvent_t& startEvent,
                                           hipEvent_t& stopEvent, uint32_t& flags);

hipError_t capturehipExtLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                     dim3& blockDim, void**& args, size_t& sharedMemBytes,
                                     hipEvent_t& startEvent, hipEvent_t& stopEvent, int& flags);

hipError_t capturehipModuleLaunchKernel(hipStream_t& stream, hipFunction_t& f, uint32_t& gridDimX,
                                        uint32_t& gridDimY, uint32_t& gridDimZ, uint32_t& blockDimX,
                                        uint32_t& blockDimY, uint32_t& blockDimZ,
                                        uint32_t& sharedMemBytes, void**& kernelParams,
                                        void**& extra);

hipError_t capturehipModuleLaunchCooperativeKernel(hipStream_t& stream, hipFunction_t& f,
                                                   uint32_t& gridDimX, uint32_t& gridDimY,
                                                   uint32_t& gridDimZ, uint32_t& blockDimX,
                                                   uint32_t& blockDimY, uint32_t& blockDimZ,
                                                   uint32_t& sharedMemBytes, void**& kernelParams);

hipError_t capturehipLaunchByPtr(hipStream_t& stream, hipFunction_t func, dim3 blockDim,
                                 dim3 gridDim, unsigned int sharedMemBytes, void** extra);

hipError_t capturehipLaunchCooperativeKernel(hipStream_t& stream, const void*& f, dim3& gridDim,
                                             dim3& blockDim, void**& kernelParams,
                                             uint32_t& sharedMemBytes);

hipError_t capturehipMemcpy2DAsync(hipStream_t& stream, void*& dst, size_t& dpitch,
                                   const void*& src, size_t& spitch, size_t& width, size_t& height,
                                   hipMemcpyKind& kind);

hipError_t capturehipMemcpyParam2DAsync(hipStream_t& stream, const hip_Memcpy2D*& pCopy);

hipError_t capturehipMemcpy2DFromArrayAsync(hipStream_t& stream, void*& dst, size_t& dpitch,
                                            hipArray_const_t& src, size_t& wOffsetSrc,
                                            size_t& hOffsetSrc, size_t& width, size_t& height,
                                            hipMemcpyKind& kind);

hipError_t capturehipMemcpy2DToArrayAsync(hipStream_t& stream, hipArray_t& dst, size_t& wOffset,
                                          size_t& hOffset, const void*& src, size_t& spitch,
                                          size_t& width, size_t& height, hipMemcpyKind& kind);

hipError_t capturehipMemcpyAtoHAsync(hipStream_t& stream, void*& dstHost, hipArray_t& srcArray,
                                     size_t& srcOffset, size_t& ByteCount);

hipError_t capturehipMemcpyHtoAAsync(hipStream_t& stream, hipArray_t& dstArray, size_t& dstOffset,
                                     const void*& srcHost, size_t& ByteCount);

hipError_t capturehipMemcpy3DAsync(hipStream_t& stream, const hipMemcpy3DParms*& p);

hipError_t capturehipMemcpyAsync(hipStream_t& stream, void*& dst, const void*& src,
                                 size_t& sizeBytes, hipMemcpyKind& kind);

hipError_t capturehipMemcpyHtoDAsync(hipStream_t& stream, hipDeviceptr_t& dstDevice,
                                     const void*& srcHost, size_t& ByteCount, hipMemcpyKind& kind);

hipError_t capturehipMemcpyDtoDAsync(hipStream_t& stream, hipDeviceptr_t& dstDevice,
                                     hipDeviceptr_t& srcDevice, size_t& ByteCount,
                                     hipMemcpyKind& kind);

hipError_t capturehipMemcpyDtoHAsync(hipStream_t& stream, void*& dstHost, hipDeviceptr_t& srcDevice,
                                     size_t& ByteCount, hipMemcpyKind& kind);

hipError_t capturehipMemcpyFromSymbolAsync(hipStream_t& stream, void*& dst, const void*& symbol,
                                           size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind);

hipError_t capturehipMemcpyToSymbolAsync(hipStream_t& stream, const void*& symbol, const void*& src,
                                         size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind);

hipError_t capturehipMemsetAsync(hipStream_t& stream, void*& dst, int& value, size_t& valueSize,
                                 size_t& sizeBytes);

hipError_t capturehipMemset2DAsync(hipStream_t& stream, void*& dst, size_t& pitch, int& value,
                                   size_t& width, size_t& height);

hipError_t capturehipMemset3DAsync(hipStream_t& stream, hipPitchedPtr& pitchedDevPtr, int& value,
                                   hipExtent& extent);

hipError_t capturehipLaunchHostFunc(hipStream_t& stream, hipHostFn_t& fn, void*& userData);

hipError_t capturehipMallocAsync(hipStream_t stream, hipMemPool_t mem_pool, size_t size,
                                 void** dev_ptr);

hipError_t capturehipFreeAsync(hipStream_t stream, void* dev_ptr);
}  // namespace hip