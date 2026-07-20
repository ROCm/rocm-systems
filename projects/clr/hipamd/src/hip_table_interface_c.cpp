/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/amd_detail/hip_api_trace.hpp>
#include "hip_internal.hpp"
#include "lttng/rocm_trace_emit.h"
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
#define CATCH                                                                                      \
  }                                                                                                \
  catch (...) {                                                                                    \
    HIP_RETURN(hip::HandleException<hipError_t>());                                                \
  }

/* See hip_table_interface.cpp for full doc on these macros (schema v1). Both
 * wrappers here are STATUS (hipError_t) and all-IN, so only the _NOARGS
 * variant is used; the captured form is provided for completeness. */
#define ROCM_TRACE_RET_STATUS_CURATED(api, expr, ...)                                              \
  do {                                                                                             \
    const hipError_t __rocm_status = (expr);                                                       \
    rocm_trace_emit_##api##_exit(__VA_ARGS__, __rocm_status);                                      \
    return __rocm_status;                                                                          \
  } while (0)

#define ROCM_TRACE_RET_STATUS_CURATED_NOARGS(api, expr)                                            \
  do {                                                                                             \
    const hipError_t __rocm_status = (expr);                                                       \
    rocm_trace_emit_##api##_exit(__rocm_status);                                                   \
    return __rocm_status;                                                                          \
  } while (0)

DllExport hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                              uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                              uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                              uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                              hipStream_t hStream, void** kernelParams,
                                              void** extra, hipEvent_t startEvent,
                                              hipEvent_t stopEvent, uint32_t flags) {
  auto const __rocm_in_f = f;
  auto const __rocm_in_globalWorkSizeX = globalWorkSizeX;
  auto const __rocm_in_globalWorkSizeY = globalWorkSizeY;
  auto const __rocm_in_globalWorkSizeZ = globalWorkSizeZ;
  auto const __rocm_in_localWorkSizeX = localWorkSizeX;
  auto const __rocm_in_localWorkSizeY = localWorkSizeY;
  auto const __rocm_in_localWorkSizeZ = localWorkSizeZ;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_hStream = hStream;
  auto const __rocm_in_kernelParams = kernelParams;
  auto const __rocm_in_extra = extra;
  auto const __rocm_in_startEvent = startEvent;
  auto const __rocm_in_stopEvent = stopEvent;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipExtModuleLaunchKernel_enter(
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_globalWorkSizeX),
      (__rocm_in_globalWorkSizeY),
      (__rocm_in_globalWorkSizeZ),
      (__rocm_in_localWorkSizeX),
      (__rocm_in_localWorkSizeY),
      (__rocm_in_localWorkSizeZ),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_hStream),
      (const void*)(uintptr_t)(__rocm_in_kernelParams),
      (const void*)(uintptr_t)(__rocm_in_extra),
      (uint64_t)(uintptr_t)(__rocm_in_startEvent),
      (uint64_t)(uintptr_t)(__rocm_in_stopEvent),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipExtModuleLaunchKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExtModuleLaunchKernel, hip::GetHipDispatchTable()->hipExtModuleLaunchKernel_fn(
          f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
          localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent,
          flags));
  CATCH;
}
DllExport hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                              uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                              uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                              uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                              hipStream_t hStream, void** kernelParams,
                                              void** extra, hipEvent_t startEvent,
                                              hipEvent_t stopEvent) {
  auto const __rocm_in_f = f;
  auto const __rocm_in_globalWorkSizeX = globalWorkSizeX;
  auto const __rocm_in_globalWorkSizeY = globalWorkSizeY;
  auto const __rocm_in_globalWorkSizeZ = globalWorkSizeZ;
  auto const __rocm_in_localWorkSizeX = localWorkSizeX;
  auto const __rocm_in_localWorkSizeY = localWorkSizeY;
  auto const __rocm_in_localWorkSizeZ = localWorkSizeZ;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_hStream = hStream;
  auto const __rocm_in_kernelParams = kernelParams;
  auto const __rocm_in_extra = extra;
  auto const __rocm_in_startEvent = startEvent;
  auto const __rocm_in_stopEvent = stopEvent;
  rocm_trace_emit_hipHccModuleLaunchKernel_enter(
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_globalWorkSizeX),
      (__rocm_in_globalWorkSizeY),
      (__rocm_in_globalWorkSizeZ),
      (__rocm_in_localWorkSizeX),
      (__rocm_in_localWorkSizeY),
      (__rocm_in_localWorkSizeZ),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_hStream),
      (const void*)(uintptr_t)(__rocm_in_kernelParams),
      (const void*)(uintptr_t)(__rocm_in_extra),
      (uint64_t)(uintptr_t)(__rocm_in_startEvent),
      (uint64_t)(uintptr_t)(__rocm_in_stopEvent)); /* __ROCM_CURATED__: hipHccModuleLaunchKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipHccModuleLaunchKernel, hip::GetHipDispatchTable()->hipHccModuleLaunchKernel_fn(
          f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
          localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent));
  CATCH;
}
