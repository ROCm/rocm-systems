/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/amd_detail/hip_api_trace.hpp>
namespace hip {
const HipDispatchTable* GetHipDispatchTable();
const HipCompilerDispatchTable* GetHipCompilerDispatchTable();
const HipToolsDispatchTable* GetHipToolsDispatchTable();
template <typename T> T HandleException();
template <> hipError_t HandleException<hipError_t>();
}  // namespace hip

#ifdef _WIN32
#define DllExport extern "C" __declspec(dllexport)
#else  // !_WIN32
#define DllExport extern "C"
#endif  // !_WIN32

#define TRY try {
#define CATCH } catch(...) { return hip::HandleException<hipError_t>(); }

DllExport hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                              uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                              uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                              uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                              hipStream_t hStream, void** kernelParams,
                                              void** extra, hipEvent_t startEvent,
                                              hipEvent_t stopEvent, uint32_t flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipExtModuleLaunchKernel_fn(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags);
  CATCH;
}
DllExport hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                              uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                              uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                              uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                              hipStream_t hStream, void** kernelParams,
                                              void** extra, hipEvent_t startEvent,
                                              hipEvent_t stopEvent) {
  TRY;
  return hip::GetHipDispatchTable()->hipHccModuleLaunchKernel_fn(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent);
  CATCH;
}
