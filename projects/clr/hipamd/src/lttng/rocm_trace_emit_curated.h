/* AUTO-GENERATED from curated_apis.yaml. Do not edit by hand.
 * SHA256(curated_apis.yaml) at generation:
 * 644fc23cd752aec46ae309d5b3f117f4bac5067a8a169b5d6fe8674028c5df45
 *
 * Per-API typed emit helpers for curated parameter capture. Every helper
 * takes (uint64_t corr_id, <captured-args...>, <status_type> status);
 * status is the call's success result, used to gate OUT-param deref.
 * All-IN APIs accept it but mark it unused.
 */
#ifndef ROCM_HIP_TRACE_EMIT_CURATED_H_
#define ROCM_HIP_TRACE_EMIT_CURATED_H_

// NOTE: When the codegen is regenerated (via PR2's lttng_curated_codegen.py),
// ensure "rocm_trace_tid.h" is emitted INSIDE the *_ENABLE_LTTNG_UST guard.
// The unconditional include breaks Windows builds where rocprofiler-register
// is not on the include path. See PR #5475 CI fix history.

#include <stdint.h>
#include <stddef.h>
/* Force the AMD platform define so the host-only HIP runtime header is
 * self-contained. rocclr internal TUs that pull in rocm_trace_emit.h
 * (e.g. device/rocm/rocvirtual.cpp) don't set this themselves. This
 * file is built only into libamdhip64; there is no NVIDIA path. */
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>
#include "rocm_dim3_pack.h"

#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

#include "rocm_trace_tid.h"
#include <atomic>
#include "rocm_hip_curated_tp.h"

extern std::atomic<bool> rocm_hip_trace_g_disabled;
#ifndef ROCM_TRACE_DISABLED_DEFINED
#define ROCM_TRACE_DISABLED_DEFINED
static inline bool rocm_trace_disabled(void) {
  return rocm_hip_trace_g_disabled.load(std::memory_order_relaxed);
}
#endif

static inline void rocm_trace_emit_hipStreamCreate_args(uint64_t corr_id,
                                                        hipStream_t* stream_out_ptr,
                                                        hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamCreate_args)) {
    const uint64_t stream_val = (status == hipSuccess && stream_out_ptr != NULL)
                                    ? (uint64_t)((uint64_t)(uintptr_t)(*stream_out_ptr))
                                    : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipStreamCreate_args, corr_id, stream_val);
  }
}

static inline void rocm_trace_emit_hipStreamCreateWithFlags_args(uint64_t corr_id,
                                                                 hipStream_t* stream_out_ptr,
                                                                 uint32_t flags,
                                                                 hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamCreateWithFlags_args)) {
    const uint64_t stream_val = (status == hipSuccess && stream_out_ptr != NULL)
                                    ? (uint64_t)((uint64_t)(uintptr_t)(*stream_out_ptr))
                                    : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipStreamCreateWithFlags_args, corr_id, stream_val,
                            (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipStreamCreateWithPriority_args(uint64_t corr_id,
                                                                    hipStream_t* stream_out_ptr,
                                                                    uint32_t flags,
                                                                    int32_t priority,
                                                                    hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamCreateWithPriority_args)) {
    const uint64_t stream_val = (status == hipSuccess && stream_out_ptr != NULL)
                                    ? (uint64_t)((uint64_t)(uintptr_t)(*stream_out_ptr))
                                    : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipStreamCreateWithPriority_args, corr_id, stream_val,
                            (uint32_t)(flags), (int32_t)(priority));
  }
}

static inline void rocm_trace_emit_hipStreamDestroy_args(
    uint64_t corr_id, uint64_t stream, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamDestroy_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipStreamDestroy_args, corr_id,
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipStreamGetFlags_args(uint64_t corr_id, uint64_t stream,
                                                          uint32_t* flags_out_ptr,
                                                          hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamGetFlags_args)) {
    const auto flags_val = (status == hipSuccess && flags_out_ptr != NULL) ? *flags_out_ptr : 0;
    lttng_ust_do_tracepoint(rocm_hip, hipStreamGetFlags_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), (uint32_t)(flags_val));
  }
}

static inline void rocm_trace_emit_hipStreamGetPriority_args(uint64_t corr_id, uint64_t stream,
                                                             int32_t* priority_out_ptr,
                                                             hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamGetPriority_args)) {
    const auto priority_val =
        (status == hipSuccess && priority_out_ptr != NULL) ? *priority_out_ptr : 0;
    lttng_ust_do_tracepoint(rocm_hip, hipStreamGetPriority_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), (int32_t)(priority_val));
  }
}

static inline void rocm_trace_emit_hipStreamSynchronize_args(
    uint64_t corr_id, uint64_t stream, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamSynchronize_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipStreamSynchronize_args, corr_id,
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipStreamWaitEvent_args(
    uint64_t corr_id, uint64_t stream, uint64_t event, uint32_t flags,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamWaitEvent_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipStreamWaitEvent_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), (uint64_t)(uintptr_t)(event),
                            (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipStreamQuery_args(
    uint64_t corr_id, uint64_t stream, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamQuery_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipStreamQuery_args, corr_id, (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipStreamAddCallback_args(
    uint64_t corr_id, uint64_t stream, const void* callback, const void* userData, uint32_t flags,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamAddCallback_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipStreamAddCallback_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), (uint64_t)(uintptr_t)(callback),
                            (uint64_t)(uintptr_t)(userData), (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipDeviceSynchronize_args(
    uint64_t corr_id, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipDeviceSynchronize_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipDeviceSynchronize_args, corr_id);
  }
}

static inline void rocm_trace_emit_hipStreamAttachMemAsync_args(
    uint64_t corr_id, uint64_t stream, const void* dev_ptr, size_t length, uint32_t flags,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamAttachMemAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipStreamAttachMemAsync_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), (uint64_t)(uintptr_t)(dev_ptr),
                            (uint64_t)(length), (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipEventCreate_args(uint64_t corr_id, hipEvent_t* event_out_ptr,
                                                       hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipEventCreate_args)) {
    const uint64_t event_val = (status == hipSuccess && event_out_ptr != NULL)
                                   ? (uint64_t)((uint64_t)(uintptr_t)(*event_out_ptr))
                                   : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipEventCreate_args, corr_id, event_val);
  }
}

static inline void rocm_trace_emit_hipEventCreateWithFlags_args(uint64_t corr_id,
                                                                hipEvent_t* event_out_ptr,
                                                                uint32_t flags, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipEventCreateWithFlags_args)) {
    const uint64_t event_val = (status == hipSuccess && event_out_ptr != NULL)
                                   ? (uint64_t)((uint64_t)(uintptr_t)(*event_out_ptr))
                                   : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipEventCreateWithFlags_args, corr_id, event_val,
                            (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipEventDestroy_args(
    uint64_t corr_id, uint64_t event, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipEventDestroy_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipEventDestroy_args, corr_id, (uint64_t)(uintptr_t)(event));
  }
}

static inline void rocm_trace_emit_hipEventRecord_args(
    uint64_t corr_id, uint64_t event, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipEventRecord_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipEventRecord_args, corr_id, (uint64_t)(uintptr_t)(event),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipEventSynchronize_args(
    uint64_t corr_id, uint64_t event, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipEventSynchronize_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipEventSynchronize_args, corr_id,
                            (uint64_t)(uintptr_t)(event));
  }
}

static inline void rocm_trace_emit_hipEventQuery_args(
    uint64_t corr_id, uint64_t event, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipEventQuery_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipEventQuery_args, corr_id, (uint64_t)(uintptr_t)(event));
  }
}

static inline void rocm_trace_emit_hipEventElapsedTime_args(uint64_t corr_id, float* ms_out_ptr,
                                                            uint64_t start, uint64_t stop,
                                                            hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipEventElapsedTime_args)) {
    const auto ms_val = (status == hipSuccess && ms_out_ptr != NULL) ? *ms_out_ptr : 0;
    lttng_ust_do_tracepoint(rocm_hip, hipEventElapsedTime_args, corr_id, (float)(ms_val),
                            (uint64_t)(uintptr_t)(start), (uint64_t)(uintptr_t)(stop));
  }
}

static inline void rocm_trace_emit_hipLaunchKernel_args(
    uint64_t corr_id, const void* function_address, dim3 numBlocks, dim3 dimBlocks,
    const void* args, size_t sharedMemBytes, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipLaunchKernel_args)) {
    const uint64_t numBlocks_packed = ROCM_DIM3_PACK(numBlocks);
    const uint64_t dimBlocks_packed = ROCM_DIM3_PACK(dimBlocks);
    lttng_ust_do_tracepoint(rocm_hip, hipLaunchKernel_args, corr_id,
                            (uint64_t)(uintptr_t)(function_address), numBlocks_packed,
                            dimBlocks_packed, (uint64_t)(uintptr_t)(args),
                            (uint64_t)(sharedMemBytes), (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipLaunchCooperativeKernel_args(
    uint64_t corr_id, const void* f, dim3 gridDim, dim3 blockDimX, const void* kernelParams,
    uint32_t sharedMemBytes, uint64_t stream, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipLaunchCooperativeKernel_args)) {
    const uint64_t gridDim_packed = ROCM_DIM3_PACK(gridDim);
    const uint64_t blockDimX_packed = ROCM_DIM3_PACK(blockDimX);
    lttng_ust_do_tracepoint(rocm_hip, hipLaunchCooperativeKernel_args, corr_id,
                            (uint64_t)(uintptr_t)(f), gridDim_packed, blockDimX_packed,
                            (uint64_t)(uintptr_t)(kernelParams), (uint32_t)(sharedMemBytes),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipLaunchCooperativeKernelMultiDevice_args(
    uint64_t corr_id, const void* launchParamsList, int32_t numDevices, uint32_t flags,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipLaunchCooperativeKernelMultiDevice_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipLaunchCooperativeKernelMultiDevice_args, corr_id,
                            (uint64_t)(uintptr_t)(launchParamsList), (int32_t)(numDevices),
                            (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipModuleLaunchKernel_args(
    uint64_t corr_id, uint64_t f, uint32_t sharedMemBytes, uint64_t stream,
    const void* kernelParams, const void* extra, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipModuleLaunchKernel_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipModuleLaunchKernel_args, corr_id, (uint64_t)(uintptr_t)(f),
                            (uint32_t)(sharedMemBytes), (uint64_t)(uintptr_t)(stream),
                            (uint64_t)(uintptr_t)(kernelParams), (uint64_t)(uintptr_t)(extra));
  }
}

static inline void rocm_trace_emit_hipExtLaunchKernel_args(
    uint64_t corr_id, const void* function_address, dim3 numBlocks, dim3 dimBlocks,
    const void* args, size_t sharedMemBytes, uint64_t stream, uint64_t startEvent,
    uint64_t stopEvent, int32_t flags, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipExtLaunchKernel_args)) {
    const uint64_t numBlocks_packed = ROCM_DIM3_PACK(numBlocks);
    const uint64_t dimBlocks_packed = ROCM_DIM3_PACK(dimBlocks);
    lttng_ust_do_tracepoint(
        rocm_hip, hipExtLaunchKernel_args, corr_id, (uint64_t)(uintptr_t)(function_address),
        numBlocks_packed, dimBlocks_packed, (uint64_t)(uintptr_t)(args), (uint64_t)(sharedMemBytes),
        (uint64_t)(uintptr_t)(stream), (uint64_t)(uintptr_t)(startEvent),
        (uint64_t)(uintptr_t)(stopEvent), (int32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipExtLaunchMultiKernelMultiDevice_args(
    uint64_t corr_id, const void* launchParamsList, int32_t numDevices, uint32_t flags,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipExtLaunchMultiKernelMultiDevice_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipExtLaunchMultiKernelMultiDevice_args, corr_id,
                            (uint64_t)(uintptr_t)(launchParamsList), (int32_t)(numDevices),
                            (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipModuleGetFunction_args(uint64_t corr_id,
                                                             hipFunction_t* function_out_ptr,
                                                             uint64_t module, const char* kname,
                                                             hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipModuleGetFunction_args)) {
    const uint64_t function_val = (status == hipSuccess && function_out_ptr != NULL)
                                      ? (uint64_t)((uint64_t)(uintptr_t)(*function_out_ptr))
                                      : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipModuleGetFunction_args, corr_id, function_val,
                            (uint64_t)(uintptr_t)(module), (kname ? kname : ""));
  }
}

static inline void rocm_trace_emit_hipModuleLoadData_args(uint64_t corr_id,
                                                          hipModule_t* module_out_ptr,
                                                          const void* image, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipModuleLoadData_args)) {
    const uint64_t module_val = (status == hipSuccess && module_out_ptr != NULL)
                                    ? (uint64_t)((uint64_t)(uintptr_t)(*module_out_ptr))
                                    : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipModuleLoadData_args, corr_id, module_val,
                            (uint64_t)(uintptr_t)(image));
  }
}

static inline void rocm_trace_emit_hipModuleLoadDataEx_args(
    uint64_t corr_id, hipModule_t* module_out_ptr, const void* image, uint32_t numOptions,
    const void* options, const void* optionValues, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipModuleLoadDataEx_args)) {
    const uint64_t module_val = (status == hipSuccess && module_out_ptr != NULL)
                                    ? (uint64_t)((uint64_t)(uintptr_t)(*module_out_ptr))
                                    : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipModuleLoadDataEx_args, corr_id, module_val,
                            (uint64_t)(uintptr_t)(image), (uint32_t)(numOptions),
                            (uint64_t)(uintptr_t)(options), (uint64_t)(uintptr_t)(optionValues));
  }
}

static inline void rocm_trace_emit_hipModuleUnload_args(
    uint64_t corr_id, uint64_t module, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipModuleUnload_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipModuleUnload_args, corr_id, (uint64_t)(uintptr_t)(module));
  }
}

static inline void rocm_trace_emit_hipMemcpy_args(uint64_t corr_id, const void* dst,
                                                  const void* src, size_t sizeBytes, int32_t kind,
                                                  hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpy_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpy_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (uint64_t)(uintptr_t)(src), (uint64_t)(sizeBytes), (int32_t)(kind));
  }
}

static inline void rocm_trace_emit_hipMemcpyDtoH_args(
    uint64_t corr_id, const void* dst, uint64_t src, size_t sizeBytes,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyDtoH_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyDtoH_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (uint64_t)(src), (uint64_t)(sizeBytes));
  }
}

static inline void rocm_trace_emit_hipMemcpyHtoD_args(
    uint64_t corr_id, uint64_t dst, const void* src, size_t sizeBytes,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyHtoD_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyHtoD_args, corr_id, (uint64_t)(dst),
                            (uint64_t)(uintptr_t)(src), (uint64_t)(sizeBytes));
  }
}

static inline void rocm_trace_emit_hipMemcpyDtoD_args(
    uint64_t corr_id, uint64_t dst, uint64_t src, size_t sizeBytes,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyDtoD_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyDtoD_args, corr_id, (uint64_t)(dst), (uint64_t)(src),
                            (uint64_t)(sizeBytes));
  }
}

static inline void rocm_trace_emit_hipMemcpyPeer_args(
    uint64_t corr_id, const void* dst, int32_t dstDeviceId, const void* src, int32_t srcDeviceId,
    size_t sizeBytes, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyPeer_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyPeer_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (int32_t)(dstDeviceId), (uint64_t)(uintptr_t)(src),
                            (int32_t)(srcDeviceId), (uint64_t)(sizeBytes));
  }
}

static inline void rocm_trace_emit_hipMemcpyAsync_args(
    uint64_t corr_id, const void* dst, const void* src, size_t sizeBytes, int32_t kind,
    uint64_t stream, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyAsync_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (uint64_t)(uintptr_t)(src), (uint64_t)(sizeBytes), (int32_t)(kind),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemcpyDtoHAsync_args(
    uint64_t corr_id, const void* dst, uint64_t src, size_t sizeBytes, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyDtoHAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyDtoHAsync_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (uint64_t)(src), (uint64_t)(sizeBytes), (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemcpyHtoDAsync_args(
    uint64_t corr_id, uint64_t dst, const void* src, size_t sizeBytes, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyHtoDAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyHtoDAsync_args, corr_id, (uint64_t)(dst),
                            (uint64_t)(uintptr_t)(src), (uint64_t)(sizeBytes),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemcpyDtoDAsync_args(
    uint64_t corr_id, uint64_t dst, uint64_t src, size_t sizeBytes, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyDtoDAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyDtoDAsync_args, corr_id, (uint64_t)(dst),
                            (uint64_t)(src), (uint64_t)(sizeBytes), (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemcpyPeerAsync_args(
    uint64_t corr_id, const void* dst, int32_t dstDeviceId, const void* src, int32_t srcDevice,
    size_t sizeBytes, uint64_t stream, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyPeerAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpyPeerAsync_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (int32_t)(dstDeviceId), (uint64_t)(uintptr_t)(src),
                            (int32_t)(srcDevice), (uint64_t)(sizeBytes),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemcpy2DAsync_args(
    uint64_t corr_id, const void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
    size_t height, int32_t kind, uint64_t stream, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpy2DAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpy2DAsync_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (uint64_t)(dpitch), (uint64_t)(uintptr_t)(src), (uint64_t)(spitch),
                            (uint64_t)(width), (uint64_t)(height), (int32_t)(kind),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemcpy3DAsync_args(
    uint64_t corr_id, const void* p, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpy3DAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemcpy3DAsync_args, corr_id, (uint64_t)(uintptr_t)(p),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMalloc_args(uint64_t corr_id, void** ptr_out_ptr, size_t size,
                                                  hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMalloc_args)) {
    const uint64_t ptr_val = (status == hipSuccess && ptr_out_ptr != NULL)
                                 ? (uint64_t)((uint64_t)(uintptr_t)(*ptr_out_ptr))
                                 : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipMalloc_args, corr_id, ptr_val, (uint64_t)(size));
  }
}

static inline void rocm_trace_emit_hipMallocAsync_args(uint64_t corr_id, void** dev_ptr_out_ptr,
                                                       size_t size, uint64_t stream,
                                                       hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMallocAsync_args)) {
    const uint64_t dev_ptr_val = (status == hipSuccess && dev_ptr_out_ptr != NULL)
                                     ? (uint64_t)((uint64_t)(uintptr_t)(*dev_ptr_out_ptr))
                                     : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipMallocAsync_args, corr_id, dev_ptr_val, (uint64_t)(size),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMallocHost_args(uint64_t corr_id, void** ptr_out_ptr,
                                                      size_t size, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMallocHost_args)) {
    const uint64_t ptr_val = (status == hipSuccess && ptr_out_ptr != NULL)
                                 ? (uint64_t)((uint64_t)(uintptr_t)(*ptr_out_ptr))
                                 : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipMallocHost_args, corr_id, ptr_val, (uint64_t)(size));
  }
}

static inline void rocm_trace_emit_hipHostMalloc_args(uint64_t corr_id, void** ptr_out_ptr,
                                                      size_t size, uint32_t flags,
                                                      hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipHostMalloc_args)) {
    const uint64_t ptr_val = (status == hipSuccess && ptr_out_ptr != NULL)
                                 ? (uint64_t)((uint64_t)(uintptr_t)(*ptr_out_ptr))
                                 : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipHostMalloc_args, corr_id, ptr_val, (uint64_t)(size),
                            (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipMallocManaged_args(uint64_t corr_id, void** dev_ptr_out_ptr,
                                                         size_t size, uint32_t flags,
                                                         hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMallocManaged_args)) {
    const uint64_t dev_ptr_val = (status == hipSuccess && dev_ptr_out_ptr != NULL)
                                     ? (uint64_t)((uint64_t)(uintptr_t)(*dev_ptr_out_ptr))
                                     : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipMallocManaged_args, corr_id, dev_ptr_val, (uint64_t)(size),
                            (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipFree_args(uint64_t corr_id, const void* ptr,
                                                hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipFree_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipFree_args, corr_id, (uint64_t)(uintptr_t)(ptr));
  }
}

static inline void rocm_trace_emit_hipFreeAsync_args(
    uint64_t corr_id, const void* dev_ptr, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipFreeAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipFreeAsync_args, corr_id, (uint64_t)(uintptr_t)(dev_ptr),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipFreeHost_args(
    uint64_t corr_id, const void* ptr, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipFreeHost_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipFreeHost_args, corr_id, (uint64_t)(uintptr_t)(ptr));
  }
}

static inline void rocm_trace_emit_hipHostFree_args(
    uint64_t corr_id, const void* ptr, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipHostFree_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipHostFree_args, corr_id, (uint64_t)(uintptr_t)(ptr));
  }
}

static inline void rocm_trace_emit_hipMemset_args(uint64_t corr_id, const void* dst, int32_t value,
                                                  size_t sizeBytes,
                                                  hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemset_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemset_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (int32_t)(value), (uint64_t)(sizeBytes));
  }
}

static inline void rocm_trace_emit_hipMemsetAsync_args(
    uint64_t corr_id, const void* dst, int32_t value, size_t sizeBytes, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemsetAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemsetAsync_args, corr_id, (uint64_t)(uintptr_t)(dst),
                            (int32_t)(value), (uint64_t)(sizeBytes), (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemsetD8_args(
    uint64_t corr_id, uint64_t dest, uint32_t value, size_t count,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemsetD8_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemsetD8_args, corr_id, (uint64_t)(dest),
                            (uint32_t)(value), (uint64_t)(count));
  }
}

static inline void rocm_trace_emit_hipMemsetD16_args(
    uint64_t corr_id, uint64_t dest, uint32_t value, size_t count,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemsetD16_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemsetD16_args, corr_id, (uint64_t)(dest),
                            (uint32_t)(value), (uint64_t)(count));
  }
}

static inline void rocm_trace_emit_hipMemsetD32_args(
    uint64_t corr_id, uint64_t dest, int32_t value, size_t count,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemsetD32_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemsetD32_args, corr_id, (uint64_t)(dest),
                            (int32_t)(value), (uint64_t)(count));
  }
}

static inline void rocm_trace_emit_hipMemPrefetchAsync_args(
    uint64_t corr_id, const void* dev_ptr, size_t count, int32_t device, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemPrefetchAsync_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemPrefetchAsync_args, corr_id,
                            (uint64_t)(uintptr_t)(dev_ptr), (uint64_t)(count), (int32_t)(device),
                            (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipMemAdvise_args(
    uint64_t corr_id, const void* dev_ptr, size_t count, int32_t advice, int32_t device,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemAdvise_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipMemAdvise_args, corr_id, (uint64_t)(uintptr_t)(dev_ptr),
                            (uint64_t)(count), (int32_t)(advice), (int32_t)(device));
  }
}

static inline void rocm_trace_emit_hipGraphCreate_args(uint64_t corr_id, hipGraph_t* pGraph_out_ptr,
                                                       uint32_t flags, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphCreate_args)) {
    const uint64_t pGraph_val = (status == hipSuccess && pGraph_out_ptr != NULL)
                                    ? (uint64_t)((uint64_t)(uintptr_t)(*pGraph_out_ptr))
                                    : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipGraphCreate_args, corr_id, pGraph_val, (uint32_t)(flags));
  }
}

static inline void rocm_trace_emit_hipGraphDestroy_args(
    uint64_t corr_id, uint64_t graph, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphDestroy_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipGraphDestroy_args, corr_id, (uint64_t)(uintptr_t)(graph));
  }
}

static inline void rocm_trace_emit_hipGraphInstantiate_args(uint64_t corr_id,
                                                            hipGraphExec_t* pGraphExec_out_ptr,
                                                            uint64_t graph, const void* pErrorNode,
                                                            const void* pLogBuffer,
                                                            size_t bufferSize, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphInstantiate_args)) {
    const uint64_t pGraphExec_val = (status == hipSuccess && pGraphExec_out_ptr != NULL)
                                        ? (uint64_t)((uint64_t)(uintptr_t)(*pGraphExec_out_ptr))
                                        : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipGraphInstantiate_args, corr_id, pGraphExec_val,
                            (uint64_t)(uintptr_t)(graph), (uint64_t)(uintptr_t)(pErrorNode),
                            (uint64_t)(uintptr_t)(pLogBuffer), (uint64_t)(bufferSize));
  }
}

static inline void rocm_trace_emit_hipGraphExecDestroy_args(
    uint64_t corr_id, uint64_t graphExec, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphExecDestroy_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipGraphExecDestroy_args, corr_id,
                            (uint64_t)(uintptr_t)(graphExec));
  }
}

static inline void rocm_trace_emit_hipStreamBeginCapture_args(
    uint64_t corr_id, uint64_t stream, int32_t mode,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamBeginCapture_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipStreamBeginCapture_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), (int32_t)(mode));
  }
}

static inline void rocm_trace_emit_hipStreamEndCapture_args(uint64_t corr_id, uint64_t stream,
                                                            hipGraph_t* pGraph_out_ptr,
                                                            hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamEndCapture_args)) {
    const uint64_t pGraph_val = (status == hipSuccess && pGraph_out_ptr != NULL)
                                    ? (uint64_t)((uint64_t)(uintptr_t)(*pGraph_out_ptr))
                                    : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipStreamEndCapture_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), pGraph_val);
  }
}

static inline void rocm_trace_emit_hipStreamIsCapturing_args(uint64_t corr_id, uint64_t stream,
                                                             int32_t* pCaptureStatus_out_ptr,
                                                             hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipStreamIsCapturing_args)) {
    const auto pCaptureStatus_val =
        (status == hipSuccess && pCaptureStatus_out_ptr != NULL) ? *pCaptureStatus_out_ptr : 0;
    lttng_ust_do_tracepoint(rocm_hip, hipStreamIsCapturing_args, corr_id,
                            (uint64_t)(uintptr_t)(stream), (int32_t)(pCaptureStatus_val));
  }
}

static inline void rocm_trace_emit_hipGraphAddKernelNode_args(
    uint64_t corr_id, hipGraphNode_t* pGraphNode_out_ptr, uint64_t graph, const void* pDependencies,
    size_t numDependencies, const void* pNodeParams, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphAddKernelNode_args)) {
    const uint64_t pGraphNode_val = (status == hipSuccess && pGraphNode_out_ptr != NULL)
                                        ? (uint64_t)((uint64_t)(uintptr_t)(*pGraphNode_out_ptr))
                                        : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipGraphAddKernelNode_args, corr_id, pGraphNode_val,
                            (uint64_t)(uintptr_t)(graph), (uint64_t)(uintptr_t)(pDependencies),
                            (uint64_t)(numDependencies), (uint64_t)(uintptr_t)(pNodeParams));
  }
}

static inline void rocm_trace_emit_hipGraphAddMemcpyNode_args(
    uint64_t corr_id, hipGraphNode_t* pGraphNode_out_ptr, uint64_t graph, const void* pDependencies,
    size_t numDependencies, const void* pCopyParams, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphAddMemcpyNode_args)) {
    const uint64_t pGraphNode_val = (status == hipSuccess && pGraphNode_out_ptr != NULL)
                                        ? (uint64_t)((uint64_t)(uintptr_t)(*pGraphNode_out_ptr))
                                        : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipGraphAddMemcpyNode_args, corr_id, pGraphNode_val,
                            (uint64_t)(uintptr_t)(graph), (uint64_t)(uintptr_t)(pDependencies),
                            (uint64_t)(numDependencies), (uint64_t)(uintptr_t)(pCopyParams));
  }
}

static inline void rocm_trace_emit_hipGraphAddMemsetNode_args(
    uint64_t corr_id, hipGraphNode_t* pGraphNode_out_ptr, uint64_t graph, const void* pDependencies,
    size_t numDependencies, const void* pMemsetParams, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphAddMemsetNode_args)) {
    const uint64_t pGraphNode_val = (status == hipSuccess && pGraphNode_out_ptr != NULL)
                                        ? (uint64_t)((uint64_t)(uintptr_t)(*pGraphNode_out_ptr))
                                        : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipGraphAddMemsetNode_args, corr_id, pGraphNode_val,
                            (uint64_t)(uintptr_t)(graph), (uint64_t)(uintptr_t)(pDependencies),
                            (uint64_t)(numDependencies), (uint64_t)(uintptr_t)(pMemsetParams));
  }
}

static inline void rocm_trace_emit_hipGraphAddEventRecordNode_args(
    uint64_t corr_id, hipGraphNode_t* pGraphNode_out_ptr, uint64_t graph, const void* pDependencies,
    size_t numDependencies, uint64_t event, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphAddEventRecordNode_args)) {
    const uint64_t pGraphNode_val = (status == hipSuccess && pGraphNode_out_ptr != NULL)
                                        ? (uint64_t)((uint64_t)(uintptr_t)(*pGraphNode_out_ptr))
                                        : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipGraphAddEventRecordNode_args, corr_id, pGraphNode_val,
                            (uint64_t)(uintptr_t)(graph), (uint64_t)(uintptr_t)(pDependencies),
                            (uint64_t)(numDependencies), (uint64_t)(uintptr_t)(event));
  }
}

static inline void rocm_trace_emit_hipGraphAddEventWaitNode_args(
    uint64_t corr_id, hipGraphNode_t* pGraphNode_out_ptr, uint64_t graph, const void* pDependencies,
    size_t numDependencies, uint64_t event, hipError_t status) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphAddEventWaitNode_args)) {
    const uint64_t pGraphNode_val = (status == hipSuccess && pGraphNode_out_ptr != NULL)
                                        ? (uint64_t)((uint64_t)(uintptr_t)(*pGraphNode_out_ptr))
                                        : 0ULL;
    lttng_ust_do_tracepoint(rocm_hip, hipGraphAddEventWaitNode_args, corr_id, pGraphNode_val,
                            (uint64_t)(uintptr_t)(graph), (uint64_t)(uintptr_t)(pDependencies),
                            (uint64_t)(numDependencies), (uint64_t)(uintptr_t)(event));
  }
}

static inline void rocm_trace_emit_hipGraphAddDependencies_args(
    uint64_t corr_id, uint64_t graph, const void* from, const void* to, size_t numDependencies,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphAddDependencies_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipGraphAddDependencies_args, corr_id,
                            (uint64_t)(uintptr_t)(graph), (uint64_t)(uintptr_t)(from),
                            (uint64_t)(uintptr_t)(to), (uint64_t)(numDependencies));
  }
}

static inline void rocm_trace_emit_hipGraphLaunch_args(
    uint64_t corr_id, uint64_t graphExec, uint64_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphLaunch_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipGraphLaunch_args, corr_id,
                            (uint64_t)(uintptr_t)(graphExec), (uint64_t)(uintptr_t)(stream));
  }
}

static inline void rocm_trace_emit_hipGraphExecKernelNodeSetParams_args(
    uint64_t corr_id, uint64_t hGraphExec, uint64_t node, const void* pNodeParams,
    hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphExecKernelNodeSetParams_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipGraphExecKernelNodeSetParams_args, corr_id,
                            (uint64_t)(uintptr_t)(hGraphExec), (uint64_t)(uintptr_t)(node),
                            (uint64_t)(uintptr_t)(pNodeParams));
  }
}

static inline void rocm_trace_emit_hipGraphExecMemcpyNodeSetParams1D_args(
    uint64_t corr_id, uint64_t hGraphExec, uint64_t node, const void* dst, const void* src,
    size_t count, int32_t kind, hipError_t /*status*/ /* unused: all-IN API */) {
  if (rocm_trace_disabled()) return;
  if (lttng_ust_tracepoint_enabled(rocm_hip, hipGraphExecMemcpyNodeSetParams1D_args)) {
    lttng_ust_do_tracepoint(rocm_hip, hipGraphExecMemcpyNodeSetParams1D_args, corr_id,
                            (uint64_t)(uintptr_t)(hGraphExec), (uint64_t)(uintptr_t)(node),
                            (uint64_t)(uintptr_t)(dst), (uint64_t)(uintptr_t)(src),
                            (uint64_t)(count), (int32_t)(kind));
  }
}


#else /* HIP_ENABLE_LTTNG_UST not defined — all helpers are no-ops */

static inline void rocm_trace_emit_hipStreamCreate_args(uint64_t, hipStream_t*, hipError_t) {}
static inline void rocm_trace_emit_hipStreamCreateWithFlags_args(uint64_t, hipStream_t*, uint32_t,
                                                                 hipError_t) {}
static inline void rocm_trace_emit_hipStreamCreateWithPriority_args(uint64_t, hipStream_t*,
                                                                    uint32_t, int32_t, hipError_t) {
}
static inline void rocm_trace_emit_hipStreamDestroy_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipStreamGetFlags_args(uint64_t, uint64_t, uint32_t*,
                                                          hipError_t) {}
static inline void rocm_trace_emit_hipStreamGetPriority_args(uint64_t, uint64_t, int32_t*,
                                                             hipError_t) {}
static inline void rocm_trace_emit_hipStreamSynchronize_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipStreamWaitEvent_args(uint64_t, uint64_t, uint64_t, uint32_t,
                                                           hipError_t) {}
static inline void rocm_trace_emit_hipStreamQuery_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipStreamAddCallback_args(uint64_t, uint64_t, const void*,
                                                             const void*, uint32_t, hipError_t) {}
static inline void rocm_trace_emit_hipDeviceSynchronize_args(uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipStreamAttachMemAsync_args(uint64_t, uint64_t, const void*,
                                                                size_t, uint32_t, hipError_t) {}
static inline void rocm_trace_emit_hipEventCreate_args(uint64_t, hipEvent_t*, hipError_t) {}
static inline void rocm_trace_emit_hipEventCreateWithFlags_args(uint64_t, hipEvent_t*, uint32_t,
                                                                hipError_t) {}
static inline void rocm_trace_emit_hipEventDestroy_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipEventRecord_args(uint64_t, uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipEventSynchronize_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipEventQuery_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipEventElapsedTime_args(uint64_t, float*, uint64_t, uint64_t,
                                                            hipError_t) {}
static inline void rocm_trace_emit_hipLaunchKernel_args(uint64_t, const void*, dim3, dim3,
                                                        const void*, size_t, uint64_t, hipError_t) {
}
static inline void rocm_trace_emit_hipLaunchCooperativeKernel_args(uint64_t, const void*, dim3,
                                                                   dim3, const void*, uint32_t,
                                                                   uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipLaunchCooperativeKernelMultiDevice_args(uint64_t, const void*,
                                                                              int32_t, uint32_t,
                                                                              hipError_t) {}
static inline void rocm_trace_emit_hipModuleLaunchKernel_args(uint64_t, uint64_t, uint32_t,
                                                              uint64_t, const void*, const void*,
                                                              hipError_t) {}
static inline void rocm_trace_emit_hipExtLaunchKernel_args(uint64_t, const void*, dim3, dim3,
                                                           const void*, size_t, uint64_t, uint64_t,
                                                           uint64_t, int32_t, hipError_t) {}
static inline void rocm_trace_emit_hipExtLaunchMultiKernelMultiDevice_args(uint64_t, const void*,
                                                                           int32_t, uint32_t,
                                                                           hipError_t) {}
static inline void rocm_trace_emit_hipModuleGetFunction_args(uint64_t, hipFunction_t*, uint64_t,
                                                             const char*, hipError_t) {}
static inline void rocm_trace_emit_hipModuleLoadData_args(uint64_t, hipModule_t*, const void*,
                                                          hipError_t) {}
static inline void rocm_trace_emit_hipModuleLoadDataEx_args(uint64_t, hipModule_t*, const void*,
                                                            uint32_t, const void*, const void*,
                                                            hipError_t) {}
static inline void rocm_trace_emit_hipModuleUnload_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemcpy_args(uint64_t, const void*, const void*, size_t,
                                                  int32_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyDtoH_args(uint64_t, const void*, uint64_t, size_t,
                                                      hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyHtoD_args(uint64_t, uint64_t, const void*, size_t,
                                                      hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyDtoD_args(uint64_t, uint64_t, uint64_t, size_t,
                                                      hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyPeer_args(uint64_t, const void*, int32_t, const void*,
                                                      int32_t, size_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyAsync_args(uint64_t, const void*, const void*, size_t,
                                                       int32_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyDtoHAsync_args(uint64_t, const void*, uint64_t, size_t,
                                                           uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyHtoDAsync_args(uint64_t, uint64_t, const void*, size_t,
                                                           uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyDtoDAsync_args(uint64_t, uint64_t, uint64_t, size_t,
                                                           uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemcpyPeerAsync_args(uint64_t, const void*, int32_t,
                                                           const void*, int32_t, size_t, uint64_t,
                                                           hipError_t) {}
static inline void rocm_trace_emit_hipMemcpy2DAsync_args(uint64_t, const void*, size_t, const void*,
                                                         size_t, size_t, size_t, int32_t, uint64_t,
                                                         hipError_t) {}
static inline void rocm_trace_emit_hipMemcpy3DAsync_args(uint64_t, const void*, uint64_t,
                                                         hipError_t) {}
static inline void rocm_trace_emit_hipMalloc_args(uint64_t, void**, size_t, hipError_t) {}
static inline void rocm_trace_emit_hipMallocAsync_args(uint64_t, void**, size_t, uint64_t,
                                                       hipError_t) {}
static inline void rocm_trace_emit_hipMallocHost_args(uint64_t, void**, size_t, hipError_t) {}
static inline void rocm_trace_emit_hipHostMalloc_args(uint64_t, void**, size_t, uint32_t,
                                                      hipError_t) {}
static inline void rocm_trace_emit_hipMallocManaged_args(uint64_t, void**, size_t, uint32_t,
                                                         hipError_t) {}
static inline void rocm_trace_emit_hipFree_args(uint64_t, const void*, hipError_t) {}
static inline void rocm_trace_emit_hipFreeAsync_args(uint64_t, const void*, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipFreeHost_args(uint64_t, const void*, hipError_t) {}
static inline void rocm_trace_emit_hipHostFree_args(uint64_t, const void*, hipError_t) {}
static inline void rocm_trace_emit_hipMemset_args(uint64_t, const void*, int32_t, size_t,
                                                  hipError_t) {}
static inline void rocm_trace_emit_hipMemsetAsync_args(uint64_t, const void*, int32_t, size_t,
                                                       uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemsetD8_args(uint64_t, uint64_t, uint32_t, size_t,
                                                    hipError_t) {}
static inline void rocm_trace_emit_hipMemsetD16_args(uint64_t, uint64_t, uint32_t, size_t,
                                                     hipError_t) {}
static inline void rocm_trace_emit_hipMemsetD32_args(uint64_t, uint64_t, int32_t, size_t,
                                                     hipError_t) {}
static inline void rocm_trace_emit_hipMemPrefetchAsync_args(uint64_t, const void*, size_t, int32_t,
                                                            uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipMemAdvise_args(uint64_t, const void*, size_t, int32_t,
                                                     int32_t, hipError_t) {}
static inline void rocm_trace_emit_hipGraphCreate_args(uint64_t, hipGraph_t*, uint32_t,
                                                       hipError_t) {}
static inline void rocm_trace_emit_hipGraphDestroy_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipGraphInstantiate_args(uint64_t, hipGraphExec_t*, uint64_t,
                                                            const void*, const void*, size_t,
                                                            hipError_t) {}
static inline void rocm_trace_emit_hipGraphExecDestroy_args(uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipStreamBeginCapture_args(uint64_t, uint64_t, int32_t,
                                                              hipError_t) {}
static inline void rocm_trace_emit_hipStreamEndCapture_args(uint64_t, uint64_t, hipGraph_t*,
                                                            hipError_t) {}
static inline void rocm_trace_emit_hipStreamIsCapturing_args(uint64_t, uint64_t, int32_t*,
                                                             hipError_t) {}
static inline void rocm_trace_emit_hipGraphAddKernelNode_args(uint64_t, hipGraphNode_t*, uint64_t,
                                                              const void*, size_t, const void*,
                                                              hipError_t) {}
static inline void rocm_trace_emit_hipGraphAddMemcpyNode_args(uint64_t, hipGraphNode_t*, uint64_t,
                                                              const void*, size_t, const void*,
                                                              hipError_t) {}
static inline void rocm_trace_emit_hipGraphAddMemsetNode_args(uint64_t, hipGraphNode_t*, uint64_t,
                                                              const void*, size_t, const void*,
                                                              hipError_t) {}
static inline void rocm_trace_emit_hipGraphAddEventRecordNode_args(uint64_t, hipGraphNode_t*,
                                                                   uint64_t, const void*, size_t,
                                                                   uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipGraphAddEventWaitNode_args(uint64_t, hipGraphNode_t*,
                                                                 uint64_t, const void*, size_t,
                                                                 uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipGraphAddDependencies_args(uint64_t, uint64_t, const void*,
                                                                const void*, size_t, hipError_t) {}
static inline void rocm_trace_emit_hipGraphLaunch_args(uint64_t, uint64_t, uint64_t, hipError_t) {}
static inline void rocm_trace_emit_hipGraphExecKernelNodeSetParams_args(uint64_t, uint64_t,
                                                                        uint64_t, const void*,
                                                                        hipError_t) {}
static inline void rocm_trace_emit_hipGraphExecMemcpyNodeSetParams1D_args(uint64_t, uint64_t,
                                                                          uint64_t, const void*,
                                                                          const void*, size_t,
                                                                          int32_t, hipError_t) {}

#endif /* HIP_ENABLE_LTTNG_UST */

#endif /* ROCM_HIP_TRACE_EMIT_CURATED_H_ */
