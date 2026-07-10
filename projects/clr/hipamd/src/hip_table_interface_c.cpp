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

/* See hip_table_interface.cpp for full doc on these macros. Both wrappers
 * here are STATUS (hipError_t), so only that variant is needed. */
#define ROCM_TRACE_RET_STATUS(EXPR)                                                                \
  do {                                                                                             \
    const auto __rocm_rv = (EXPR);                                                                 \
    rocm_trace_emit_hip_api_exit_status(__func__, static_cast<int32_t>(__rocm_rv));                \
    return __rocm_rv;                                                                              \
  } while (0)

DllExport hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                              uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                              uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                              uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                              hipStream_t hStream, void** kernelParams,
                                              void** extra, hipEvent_t startEvent,
                                              hipEvent_t stopEvent, uint32_t flags) {
  rocm_trace_emit_hip_api_enter(__func__);
  TRY;
  ROCM_TRACE_RET_STATUS(hip::GetHipDispatchTable()->hipExtModuleLaunchKernel_fn(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags));
  CATCH;
}
DllExport hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                              uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                              uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                              uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                              hipStream_t hStream, void** kernelParams,
                                              void** extra, hipEvent_t startEvent,
                                              hipEvent_t stopEvent) {
  rocm_trace_emit_hip_api_enter(__func__);
  TRY;
  ROCM_TRACE_RET_STATUS(hip::GetHipDispatchTable()->hipHccModuleLaunchKernel_fn(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent));
  CATCH;
}
