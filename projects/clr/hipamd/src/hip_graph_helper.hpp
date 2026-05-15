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

#include "hip_conversions.hpp"

namespace hip {
hipError_t ihipMemcpy3D_validate(const hipMemcpy3DParms* p);

hipError_t ihipDrvMemcpy3D_validate(const HIP_MEMCPY3D* pCopy);

hipError_t hipMemcpy2DValidateArray(hipArray_const_t arr, size_t wOffset, size_t hOffset,
                                    size_t width, size_t height);

hipError_t hipMemcpy2DValidateBuffer(const void* buf, size_t pitch, size_t width);

hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

hipError_t ihipMemcpyCommand(amd::Command*& command, void* dst, const void* src, size_t sizeBytes,
                             hipMemcpyKind kind, hip::Stream& stream, bool isAsync = true);

void ihipHtoHMemcpy(void* dst, const void* src, size_t sizeBytes, hip::Stream& stream);

bool IsHtoHMemcpy(void* dst, const void* src);

hipError_t ihipLaunchKernel_validate(hipFunction_t f, const amd::LaunchParams& launch_params,
                                     void** kernelParams, void** extra, int deviceId,
                                     uint32_t params);

hipError_t ihipMemset_validate(void* dst, int64_t value, size_t valueSize, size_t sizeBytes);

hipError_t ihipMemset3D_validate(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                 size_t sizeBytes);

hipError_t ihipLaunchKernelCommand(amd::Command*& command, hipFunction_t f,
                                   amd::LaunchParams& launch_params, hip::Stream* stream,
                                   void** kernelParams, void** extra, hipEvent_t startEvent,
                                   hipEvent_t stopEvent, uint32_t flags, uint32_t params,
                                   uint32_t gridId, uint32_t numGrids, uint64_t prevGridSum,
                                   uint64_t allGridSum, uint32_t firstDevice);

hipError_t ihipMemcpy3DCommand(amd::Command*& command, const hipMemcpy3DParms* p,
                               hip::Stream* stream);

hipError_t ihipGetMemcpyParam3DCommand(amd::Command*& command, const HIP_MEMCPY3D* pCopy,
                                       hip::Stream* stream);

hipError_t ihipMemsetCommand(std::vector<amd::Command*>& commands, void* dst, int64_t value,
                             size_t valueSize, size_t sizeBytes, hip::Stream* stream);

hipError_t ihipMemset3DCommand(std::vector<amd::Command*>& commands, hipPitchedPtr pitchedDevPtr,
                               int value, hipExtent extent, hip::Stream* stream,
                               size_t elementSize = 1);

hipError_t ihipMemcpySymbol_validate(const void* symbol, size_t sizeBytes, size_t offset,
                                     size_t& sym_size, hipDeviceptr_t& device_ptr);

hipError_t ihipMemcpyAtoDValidate(hipArray_t srcArray, void* dstDevice, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t dstRowPitch, size_t dstSlicePitch, amd::Memory*& dstMemory,
                                  amd::Image*& srcImage, amd::BufferRect& srcRect,
                                  amd::BufferRect& dstRect);

hipError_t ihipMemcpyDtoAValidate(void* srcDevice, hipArray_t dstArray, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, amd::Image*& dstImage,
                                  amd::Memory*& srcMemory, amd::BufferRect& dstRect,
                                  amd::BufferRect& srcRect);

hipError_t ihipMemcpyDtoDValidate(void* srcDevice, void* dstDevice, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, size_t dstRowPitch,
                                  size_t dstSlicePitch, amd::Memory*& srcMemory,
                                  amd::Memory*& dstMemory, amd::BufferRect& srcRect,
                                  amd::BufferRect& dstRect);


hipError_t ihipMemcpyDtoHValidate(void* srcDevice, void* dstHost, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, size_t dstRowPitch,
                                  size_t dstSlicePitch, amd::Memory*& srcMemory,
                                  amd::BufferRect& srcRect, amd::BufferRect& dstRect);

hipError_t ihipMemcpyHtoDValidate(const void* srcHost, void* dstDevice, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, size_t dstRowPitch,
                                  size_t dstSlicePitch, amd::Memory*& dstMemory,
                                  amd::BufferRect& srcRect, amd::BufferRect& dstRect);


hipError_t ihipMemcpyAtoAValidate(hipArray_t srcArray, hipArray_t dstArray, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  amd::Image*& srcImage, amd::Image*& dstImage);


hipError_t ihipMemcpyHtoAValidate(const void* srcHost, hipArray_t dstArray, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, amd::Image*& dstImage,
                                  size_t& start);

hipError_t ihipMemcpyAtoHValidate(hipArray_t srcArray, void* dstHost, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t dstRowPitch, size_t dstSlicePitch, amd::Image*& srcImage,
                                  size_t& start);

hipError_t ihipGraphMemsetParams_validate(const hipMemsetParams* pNodeParams);

hip::MemcpyType ihipGetMemcpyType(const void* src, void* dst, hipMemcpyKind kind);
}  // namespace hip
