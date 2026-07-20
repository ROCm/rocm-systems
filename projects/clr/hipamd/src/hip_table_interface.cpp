/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/amd_detail/hip_api_trace.hpp>
#include "hip_internal.hpp"
#include "utils/flags.hpp"
#include "utils/debug.hpp"
#include "lttng/rocm_trace_emit.h"
#include <exception>
#include <tuple>

#if defined(__GNUC__)
#define __forceinline __inline__ __attribute__((always_inline))
#endif

namespace hip {
const HipDispatchTable* GetHipDispatchTable();
const HipCompilerDispatchTable* GetHipCompilerDispatchTable();
const HipToolsDispatchTable* GetHipToolsDispatchTable();

// ================================================================================================
// Generic exception handler - returns default value for non-hipError_t types
template <typename T> __forceinline T HandleException() {
  std::ignore = HandleException<hipError_t>();
  return T();
}

// ================================================================================================
// Specialization for hipError_t with full exception handling
template <> hipError_t HandleException<hipError_t>() {
  try {
    throw;
  } catch (const std::bad_alloc&) {
    LogPrintfError("HIP Exception: C++ BadAlloc\n");
    return hipErrorOutOfMemory;
  } catch (const std::nested_exception& e) {
    LogPrintfError("HIP Exception: C++ Callback Threw, forwarding.\n");
    e.rethrow_nested();
  } catch (const std::exception& e) {
    LogPrintfError("HIP Exception: C++ Exception: %s\n", e.what());
    return hipErrorUnknown;
  } catch (...) {
    assert(false && "HIP Exception: Unhandled.");
    return hipErrorUnknown;
  }
}
}  // namespace hip

#define TRY try {
#define CATCH                                                                                      \
  }                                                                                                \
  catch (...) {                                                                                    \
    HIP_RETURN(hip::HandleException<hipError_t>());                                                \
  }
#define CATCHRET(RETURN_TYPE)                                                                      \
  }                                                                                                \
  catch (...) {                                                                                    \
    return hip::HandleException<RETURN_TYPE>();                                                    \
  }

/* ---------- Curated combined-event return macros (schema v1) ----------
 * Each curated API is ONE combined LTTng event fired twice per call. The
 * wrapper emits the ENTER record up front via rocm_trace_emit_<api>_enter(...)
 * (IN args); these return macros emit the matching EXIT record via
 * rocm_trace_emit_<api>_exit(<OUT args...>, status) after the real call, then
 * return. STATUS/PTR/VOID/I32 select the return-field encoding; each has a
 * captured-OUT-args form and a _NOARGS form (all-IN wrappers, whose EXIT
 * record carries only phase + the return field).
 */

/* Captured-OUT-args variants. __VA_ARGS__ carries the wrapper's OUT pointers
 * (non-empty by construction; all-IN wrappers use the _NOARGS variants). */
#define ROCM_TRACE_RET_STATUS_CURATED(api, expr, ...)                                              \
  do {                                                                                             \
    const hipError_t __rocm_status = (expr);                                                       \
    rocm_trace_emit_##api##_exit(__VA_ARGS__, __rocm_status);                                      \
    return __rocm_status;                                                                          \
  } while (0)

/* PTR: the combined event's return field is the actual returned pointer
 * (retptr), captured as a uint64_t hex value. PTR-returning curated APIs have
 * no OUT args, so the exit helper takes only the return value. */
#define ROCM_TRACE_RET_PTR_CURATED(api, ptr_type, expr, ...)                                       \
  do {                                                                                             \
    ptr_type const __rocm_ptr = (expr);                                                            \
    rocm_trace_emit_##api##_exit(__VA_ARGS__, (uint64_t)(uintptr_t)__rocm_ptr);                    \
    return __rocm_ptr;                                                                             \
  } while (0)

#define ROCM_TRACE_RET_VOID_CURATED(api, expr, ...)                                                \
  do {                                                                                             \
    (expr);                                                                                        \
    rocm_trace_emit_##api##_exit(__VA_ARGS__, hipSuccess);                                         \
    return;                                                                                        \
  } while (0)

#define ROCM_TRACE_RET_I32_CURATED(api, expr, ...)                                                 \
  do {                                                                                             \
    const int __rocm_rv = (expr);                                                                  \
    rocm_trace_emit_##api##_exit(__VA_ARGS__, (int32_t)__rocm_rv);                                 \
    return __rocm_rv;                                                                              \
  } while (0)

/* Zero-captured-args variants. Separate macros to avoid empty
 * __VA_ARGS__ expansion in the captured-args macros above. */
#define ROCM_TRACE_RET_STATUS_CURATED_NOARGS(api, expr)                                            \
  do {                                                                                             \
    const hipError_t __rocm_status = (expr);                                                       \
    rocm_trace_emit_##api##_exit(__rocm_status);                                                   \
    return __rocm_status;                                                                          \
  } while (0)

#define ROCM_TRACE_RET_PTR_CURATED_NOARGS(api, ptr_type, expr)                                     \
  do {                                                                                             \
    ptr_type const __rocm_ptr = (expr);                                                            \
    rocm_trace_emit_##api##_exit((uint64_t)(uintptr_t)__rocm_ptr);                                 \
    return __rocm_ptr;                                                                             \
  } while (0)

/* VOID has no return field: the all-IN EXIT record carries only phase, so the
 * exit helper takes no argument at all. */
#define ROCM_TRACE_RET_VOID_CURATED_NOARGS(api, expr)                                              \
  do {                                                                                             \
    (expr);                                                                                        \
    rocm_trace_emit_##api##_exit();                                                                \
    return;                                                                                        \
  } while (0)

#define ROCM_TRACE_RET_I32_CURATED_NOARGS(api, expr)                                               \
  do {                                                                                             \
    const int __rocm_rv = (expr);                                                                  \
    rocm_trace_emit_##api##_exit((int32_t)__rocm_rv);                                              \
    return __rocm_rv;                                                                              \
  } while (0)

extern "C" hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                                hipStream_t* stream) {
  auto const __rocm_in_gridDim = gridDim;
  auto const __rocm_in_blockDim = blockDim;
  auto const __rocm_in_sharedMem = sharedMem;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit___hipPopCallConfiguration_enter(
      (const void*)(uintptr_t)(__rocm_in_gridDim),
      (const void*)(uintptr_t)(__rocm_in_blockDim),
      (const void*)(uintptr_t)(__rocm_in_sharedMem),
      (const void*)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: __hipPopCallConfiguration */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(__hipPopCallConfiguration, hip::GetHipCompilerDispatchTable()->__hipPopCallConfiguration_fn(gridDim, blockDim, sharedMem,
                                                                       stream));
  CATCH;
}
extern "C" hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                                 hipStream_t stream) {
  auto const __rocm_in_gridDim = gridDim;
  auto const __rocm_in_blockDim = blockDim;
  auto const __rocm_in_sharedMem = sharedMem;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit___hipPushCallConfiguration_enter(
      (__rocm_in_gridDim),
      (__rocm_in_blockDim),
      (__rocm_in_sharedMem),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: __hipPushCallConfiguration */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(__hipPushCallConfiguration, hip::GetHipCompilerDispatchTable()->__hipPushCallConfiguration_fn(
                                    gridDim, blockDim, sharedMem, stream));
  CATCH;
}
extern "C" void** __hipRegisterFatBinary(const void* data) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipRegisterFatBinary_fn(data);
  CATCHRET(void**);
}
extern "C" void __hipRegisterFunction(void** modules, const void* hostFunction,
                                      char* deviceFunction, const char* deviceName,
                                      unsigned int threadLimit, uint3* tid, uint3* bid,
                                      dim3* blockDim, dim3* gridDim, int* wSize) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipRegisterFunction_fn(
      modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim,
      wSize);
  CATCHRET(void);
}
extern "C" void __hipRegisterManagedVar(void* hipModule, void** pointer, void* init_value,
                                        const char* name, size_t size, unsigned align) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipRegisterManagedVar_fn(
      hipModule, pointer, init_value, name, size, align);
  CATCHRET(void);
}
extern "C" void __hipRegisterSurface(void** modules, void* var, char* hostVar, char* deviceVar,
                                     int type, int ext) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipRegisterSurface_fn(modules, var, hostVar,
                                                                     deviceVar, type, ext);
  CATCHRET(void);
}
extern "C" void __hipRegisterTexture(void** modules, void* var, char* hostVar, char* deviceVar,
                                     int type, int norm, int ext) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipRegisterTexture_fn(modules, var, hostVar,
                                                                     deviceVar, type, norm, ext);
  CATCHRET(void);
}
extern "C" void __hipRegisterVar(void** modules, void* var, char* hostVar, char* deviceVar, int ext,
                                 size_t size, int constant, int global) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipRegisterVar_fn(modules, var, hostVar, deviceVar,
                                                                 ext, size, constant, global);
  CATCHRET(void);
}
extern "C" void __hipUnregisterFatBinary(void** modules) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipUnregisterFatBinary_fn(modules);
  CATCHRET(void);
}
extern "C" const char* hipApiName(uint32_t id) {
  TRY;
  return hip::GetHipDispatchTable()->hipApiName_fn(id);
  CATCHRET(const char*);
}
hipError_t hipArray3DCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  auto const __rocm_in_array = array;
  auto const __rocm_in_pAllocateArray = pAllocateArray;
  rocm_trace_emit_hipArray3DCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_array),
      (const void*)(uintptr_t)(__rocm_in_pAllocateArray)); /* __ROCM_CURATED__: hipArray3DCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipArray3DCreate, hip::GetHipDispatchTable()->hipArray3DCreate_fn(array, pAllocateArray));
  CATCH;
}
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  auto const __rocm_in_pArrayDescriptor = pArrayDescriptor;
  auto const __rocm_in_array = array;
  rocm_trace_emit_hipArray3DGetDescriptor_enter(
      (const void*)(uintptr_t)(__rocm_in_pArrayDescriptor),
      (const void*)(uintptr_t)(__rocm_in_array)); /* __ROCM_CURATED__: hipArray3DGetDescriptor */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipArray3DGetDescriptor, hip::GetHipDispatchTable()->hipArray3DGetDescriptor_fn(pArrayDescriptor, array));
  CATCH;
}
hipError_t hipArrayCreate(hipArray_t* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  auto const __rocm_in_pHandle = pHandle;
  auto const __rocm_in_pAllocateArray = pAllocateArray;
  rocm_trace_emit_hipArrayCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_pHandle),
      (const void*)(uintptr_t)(__rocm_in_pAllocateArray)); /* __ROCM_CURATED__: hipArrayCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipArrayCreate, hip::GetHipDispatchTable()->hipArrayCreate_fn(pHandle, pAllocateArray));
  CATCH;
}
hipError_t hipArrayDestroy(hipArray_t array) {
  auto const __rocm_in_array = array;
  rocm_trace_emit_hipArrayDestroy_enter(
      (const void*)(uintptr_t)(__rocm_in_array)); /* __ROCM_CURATED__: hipArrayDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipArrayDestroy, hip::GetHipDispatchTable()->hipArrayDestroy_fn(array));
  CATCH;
}
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  auto const __rocm_in_pArrayDescriptor = pArrayDescriptor;
  auto const __rocm_in_array = array;
  rocm_trace_emit_hipArrayGetDescriptor_enter(
      (const void*)(uintptr_t)(__rocm_in_pArrayDescriptor),
      (const void*)(uintptr_t)(__rocm_in_array)); /* __ROCM_CURATED__: hipArrayGetDescriptor */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipArrayGetDescriptor, hip::GetHipDispatchTable()->hipArrayGetDescriptor_fn(pArrayDescriptor, array));
  CATCH;
}
hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags,
                           hipArray_t array) {
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_extent = extent;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_array = array;
  rocm_trace_emit_hipArrayGetInfo_enter(
      (const void*)(uintptr_t)(__rocm_in_desc),
      (const void*)(uintptr_t)(__rocm_in_extent),
      (const void*)(uintptr_t)(__rocm_in_flags),
      (const void*)(uintptr_t)(__rocm_in_array)); /* __ROCM_CURATED__: hipArrayGetInfo */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipArrayGetInfo, hip::GetHipDispatchTable()->hipArrayGetInfo_fn(desc, extent, flags, array));
  CATCH;
}
extern "C" hipError_t hipBindTexture(size_t* offset, const textureReference* tex,
                                     const void* devPtr, const hipChannelFormatDesc* desc,
                                     size_t size) {
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_tex = tex;
  auto const __rocm_in_devPtr = devPtr;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_size = size;
  rocm_trace_emit_hipBindTexture_enter(
      (const void*)(uintptr_t)(__rocm_in_offset),
      (const void*)(uintptr_t)(__rocm_in_tex),
      (const void*)(uintptr_t)(__rocm_in_devPtr),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (__rocm_in_size)); /* __ROCM_CURATED__: hipBindTexture */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipBindTexture, hip::GetHipDispatchTable()->hipBindTexture_fn(offset, tex, devPtr, desc, size));
  CATCH;
}
hipError_t hipBindTexture2D(size_t* offset, const textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch) {
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_tex = tex;
  auto const __rocm_in_devPtr = devPtr;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_pitch = pitch;
  rocm_trace_emit_hipBindTexture2D_enter(
      (const void*)(uintptr_t)(__rocm_in_offset),
      (const void*)(uintptr_t)(__rocm_in_tex),
      (const void*)(uintptr_t)(__rocm_in_devPtr),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (__rocm_in_width),
      (__rocm_in_height),
      (__rocm_in_pitch)); /* __ROCM_CURATED__: hipBindTexture2D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipBindTexture2D, hip::GetHipDispatchTable()->hipBindTexture2D_fn(offset, tex, devPtr, desc, width, height,
                                                      pitch));
  CATCH;
}
hipError_t hipBindTextureToArray(const textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
  auto const __rocm_in_tex = tex;
  auto const __rocm_in_array = array;
  auto const __rocm_in_desc = desc;
  rocm_trace_emit_hipBindTextureToArray_enter(
      (const void*)(uintptr_t)(__rocm_in_tex),
      (const void*)(uintptr_t)(__rocm_in_array),
      (const void*)(uintptr_t)(__rocm_in_desc)); /* __ROCM_CURATED__: hipBindTextureToArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipBindTextureToArray, hip::GetHipDispatchTable()->hipBindTextureToArray_fn(tex, array, desc));
  CATCH;
}
hipError_t hipBindTextureToMipmappedArray(const textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
  auto const __rocm_in_tex = tex;
  auto const __rocm_in_mipmappedArray = mipmappedArray;
  auto const __rocm_in_desc = desc;
  rocm_trace_emit_hipBindTextureToMipmappedArray_enter(
      (const void*)(uintptr_t)(__rocm_in_tex),
      (const void*)(uintptr_t)(__rocm_in_mipmappedArray),
      (const void*)(uintptr_t)(__rocm_in_desc)); /* __ROCM_CURATED__: hipBindTextureToMipmappedArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipBindTextureToMipmappedArray, hip::GetHipDispatchTable()->hipBindTextureToMipmappedArray_fn(tex, mipmappedArray, desc));
  CATCH;
}
extern "C" hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
  TRY;
  return hip::GetHipDispatchTable()->hipChooseDevice_fn(device, prop);
  CATCH;
}
extern "C" hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* properties) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_properties = properties;
  rocm_trace_emit_hipChooseDeviceR0000_enter(
      (const void*)(uintptr_t)(__rocm_in_device),
      (const void*)(uintptr_t)(__rocm_in_properties)); /* __ROCM_CURATED__: hipChooseDeviceR0000 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipChooseDeviceR0000, hip::GetHipDispatchTable()->hipChooseDeviceR0000_fn(device, properties));
  CATCH;
}
extern "C" hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                       hipStream_t stream) {
  auto const __rocm_in_gridDim = gridDim;
  auto const __rocm_in_blockDim = blockDim;
  auto const __rocm_in_sharedMem = sharedMem;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipConfigureCall_enter(
      (__rocm_in_gridDim),
      (__rocm_in_blockDim),
      (__rocm_in_sharedMem),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipConfigureCall */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipConfigureCall, hip::GetHipDispatchTable()->hipConfigureCall_fn(gridDim, blockDim, sharedMem, stream));
  CATCH;
}
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                  const hipResourceDesc* pResDesc) {
  auto const __rocm_in_pSurfObject = pSurfObject;
  auto const __rocm_in_pResDesc = pResDesc;
  rocm_trace_emit_hipCreateSurfaceObject_enter(
      (const void*)(uintptr_t)(__rocm_in_pSurfObject),
      (const void*)(uintptr_t)(__rocm_in_pResDesc)); /* __ROCM_CURATED__: hipCreateSurfaceObject */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCreateSurfaceObject, hip::GetHipDispatchTable()->hipCreateSurfaceObject_fn(pSurfObject, pResDesc));
  CATCH;
}
hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const struct hipResourceViewDesc* pResViewDesc) {
  auto const __rocm_in_pTexObject = pTexObject;
  auto const __rocm_in_pResDesc = pResDesc;
  auto const __rocm_in_pTexDesc = pTexDesc;
  auto const __rocm_in_pResViewDesc = pResViewDesc;
  rocm_trace_emit_hipCreateTextureObject_enter(
      (const void*)(uintptr_t)(__rocm_in_pTexObject),
      (const void*)(uintptr_t)(__rocm_in_pResDesc),
      (const void*)(uintptr_t)(__rocm_in_pTexDesc),
      (const void*)(uintptr_t)(__rocm_in_pResViewDesc)); /* __ROCM_CURATED__: hipCreateTextureObject */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCreateTextureObject, hip::GetHipDispatchTable()->hipCreateTextureObject_fn(
                                    pTexObject, pResDesc, pTexDesc, pResViewDesc));
  CATCH;
}
extern "C" hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipCtxCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (__rocm_in_flags),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipCtxCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxCreate, hip::GetHipDispatchTable()->hipCtxCreate_fn(ctx, flags, device));
  CATCH;
}
extern "C" hipError_t hipCtxDestroy(hipCtx_t ctx) {
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipCtxDestroy_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipCtxDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxDestroy, hip::GetHipDispatchTable()->hipCtxDestroy_fn(ctx));
  CATCH;
}
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
  auto const __rocm_in_peerCtx = peerCtx;
  rocm_trace_emit_hipCtxDisablePeerAccess_enter(
      (const void*)(uintptr_t)(__rocm_in_peerCtx)); /* __ROCM_CURATED__: hipCtxDisablePeerAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxDisablePeerAccess, hip::GetHipDispatchTable()->hipCtxDisablePeerAccess_fn(peerCtx));
  CATCH;
}
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
  auto const __rocm_in_peerCtx = peerCtx;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipCtxEnablePeerAccess_enter(
      (const void*)(uintptr_t)(__rocm_in_peerCtx),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipCtxEnablePeerAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxEnablePeerAccess, hip::GetHipDispatchTable()->hipCtxEnablePeerAccess_fn(peerCtx, flags));
  CATCH;
}
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int* apiVersion) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_apiVersion = apiVersion;
  rocm_trace_emit_hipCtxGetApiVersion_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (const void*)(uintptr_t)(__rocm_in_apiVersion)); /* __ROCM_CURATED__: hipCtxGetApiVersion */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxGetApiVersion, hip::GetHipDispatchTable()->hipCtxGetApiVersion_fn(ctx, apiVersion));
  CATCH;
}
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
  auto const __rocm_in_cacheConfig = cacheConfig;
  rocm_trace_emit_hipCtxGetCacheConfig_enter(
      (const void*)(uintptr_t)(__rocm_in_cacheConfig)); /* __ROCM_CURATED__: hipCtxGetCacheConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxGetCacheConfig, hip::GetHipDispatchTable()->hipCtxGetCacheConfig_fn(cacheConfig));
  CATCH;
}
hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipCtxGetCurrent_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipCtxGetCurrent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxGetCurrent, hip::GetHipDispatchTable()->hipCtxGetCurrent_fn(ctx));
  CATCH;
}
hipError_t hipCtxGetDevice(hipDevice_t* device) {
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipCtxGetDevice_enter(
      (const void*)(uintptr_t)(__rocm_in_device)); /* __ROCM_CURATED__: hipCtxGetDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxGetDevice, hip::GetHipDispatchTable()->hipCtxGetDevice_fn(device));
  CATCH;
}
hipError_t hipCtxGetFlags(unsigned int* flags) {
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipCtxGetFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_flags)); /* __ROCM_CURATED__: hipCtxGetFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxGetFlags, hip::GetHipDispatchTable()->hipCtxGetFlags_fn(flags));
  CATCH;
}
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  auto const __rocm_in_pConfig = pConfig;
  rocm_trace_emit_hipCtxGetSharedMemConfig_enter(
      (const void*)(uintptr_t)(__rocm_in_pConfig)); /* __ROCM_CURATED__: hipCtxGetSharedMemConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxGetSharedMemConfig, hip::GetHipDispatchTable()->hipCtxGetSharedMemConfig_fn(pConfig));
  CATCH;
}
hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipCtxPopCurrent_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipCtxPopCurrent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxPopCurrent, hip::GetHipDispatchTable()->hipCtxPopCurrent_fn(ctx));
  CATCH;
}
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipCtxPushCurrent_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipCtxPushCurrent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxPushCurrent, hip::GetHipDispatchTable()->hipCtxPushCurrent_fn(ctx));
  CATCH;
}
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  auto const __rocm_in_cacheConfig = cacheConfig;
  rocm_trace_emit_hipCtxSetCacheConfig_enter(
      (int32_t)(__rocm_in_cacheConfig)); /* __ROCM_CURATED__: hipCtxSetCacheConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxSetCacheConfig, hip::GetHipDispatchTable()->hipCtxSetCacheConfig_fn(cacheConfig));
  CATCH;
}
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipCtxSetCurrent_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipCtxSetCurrent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxSetCurrent, hip::GetHipDispatchTable()->hipCtxSetCurrent_fn(ctx));
  CATCH;
}
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  auto const __rocm_in_config = config;
  rocm_trace_emit_hipCtxSetSharedMemConfig_enter(
      (int32_t)(__rocm_in_config)); /* __ROCM_CURATED__: hipCtxSetSharedMemConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxSetSharedMemConfig, hip::GetHipDispatchTable()->hipCtxSetSharedMemConfig_fn(config));
  CATCH;
}
hipError_t hipCtxSynchronize(void) {
  rocm_trace_emit_hipCtxSynchronize_enter(); /* __ROCM_CURATED__: hipCtxSynchronize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipCtxSynchronize, hip::GetHipDispatchTable()->hipCtxSynchronize_fn());
  CATCH;
}
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
  TRY;
  return hip::GetHipDispatchTable()->hipDestroyExternalMemory_fn(extMem);
  CATCH;
}
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
  TRY;
  return hip::GetHipDispatchTable()->hipDestroyExternalSemaphore_fn(extSem);
  CATCH;
}
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
  auto const __rocm_in_surfaceObject = surfaceObject;
  rocm_trace_emit_hipDestroySurfaceObject_enter(
      (const void*)(uintptr_t)(__rocm_in_surfaceObject)); /* __ROCM_CURATED__: hipDestroySurfaceObject */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDestroySurfaceObject, hip::GetHipDispatchTable()->hipDestroySurfaceObject_fn(surfaceObject));
  CATCH;
}
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  auto const __rocm_in_textureObject = textureObject;
  rocm_trace_emit_hipDestroyTextureObject_enter(
      (const void*)(uintptr_t)(__rocm_in_textureObject)); /* __ROCM_CURATED__: hipDestroyTextureObject */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDestroyTextureObject, hip::GetHipDispatchTable()->hipDestroyTextureObject_fn(textureObject));
  CATCH;
}
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  auto const __rocm_in_canAccessPeer = canAccessPeer;
  auto const __rocm_in_deviceId = deviceId;
  auto const __rocm_in_peerDeviceId = peerDeviceId;
  rocm_trace_emit_hipDeviceCanAccessPeer_enter(
      (const void*)(uintptr_t)(__rocm_in_canAccessPeer),
      (__rocm_in_deviceId),
      (__rocm_in_peerDeviceId)); /* __ROCM_CURATED__: hipDeviceCanAccessPeer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceCanAccessPeer, hip::GetHipDispatchTable()->hipDeviceCanAccessPeer_fn(canAccessPeer, deviceId, peerDeviceId));
  CATCH;
}
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
  auto const __rocm_in_major = major;
  auto const __rocm_in_minor = minor;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceComputeCapability_enter(
      (const void*)(uintptr_t)(__rocm_in_major),
      (const void*)(uintptr_t)(__rocm_in_minor),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceComputeCapability */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceComputeCapability, hip::GetHipDispatchTable()->hipDeviceComputeCapability_fn(major, minor, device));
  CATCH;
}
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  auto const __rocm_in_peerDeviceId = peerDeviceId;
  rocm_trace_emit_hipDeviceDisablePeerAccess_enter(
      (__rocm_in_peerDeviceId)); /* __ROCM_CURATED__: hipDeviceDisablePeerAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceDisablePeerAccess, hip::GetHipDispatchTable()->hipDeviceDisablePeerAccess_fn(peerDeviceId));
  CATCH;
}
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  auto const __rocm_in_peerDeviceId = peerDeviceId;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipDeviceEnablePeerAccess_enter(
      (__rocm_in_peerDeviceId),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipDeviceEnablePeerAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceEnablePeerAccess, hip::GetHipDispatchTable()->hipDeviceEnablePeerAccess_fn(peerDeviceId, flags));
  CATCH;
}
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_ordinal = ordinal;
  rocm_trace_emit_hipDeviceGet_enter(
      (const void*)(uintptr_t)(__rocm_in_device),
      (__rocm_in_ordinal)); /* __ROCM_CURATED__: hipDeviceGet */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGet, hip::GetHipDispatchTable()->hipDeviceGet_fn(device, ordinal));
  CATCH;
}
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId) {
  auto const __rocm_in_pi = pi;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_deviceId = deviceId;
  rocm_trace_emit_hipDeviceGetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_pi),
      (int32_t)(__rocm_in_attr),
      (__rocm_in_deviceId)); /* __ROCM_CURATED__: hipDeviceGetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetAttribute, hip::GetHipDispatchTable()->hipDeviceGetAttribute_fn(pi, attr, deviceId));
  CATCH;
}
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_pciBusId = pciBusId;
  rocm_trace_emit_hipDeviceGetByPCIBusId_enter(
      (const void*)(uintptr_t)(__rocm_in_device),
      (const char*)(__rocm_in_pciBusId)); /* __ROCM_CURATED__: hipDeviceGetByPCIBusId */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetByPCIBusId, hip::GetHipDispatchTable()->hipDeviceGetByPCIBusId_fn(device, pciBusId));
  CATCH;
}
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig) {
  auto const __rocm_in_cacheConfig = cacheConfig;
  rocm_trace_emit_hipDeviceGetCacheConfig_enter(
      (const void*)(uintptr_t)(__rocm_in_cacheConfig)); /* __ROCM_CURATED__: hipDeviceGetCacheConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetCacheConfig, hip::GetHipDispatchTable()->hipDeviceGetCacheConfig_fn(cacheConfig));
  CATCH;
}
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGetDefaultMemPool_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGetDefaultMemPool */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetDefaultMemPool, hip::GetHipDispatchTable()->hipDeviceGetDefaultMemPool_fn(mem_pool, device));
  CATCH;
}
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipDeviceGetGraphMemAttribute_enter(
      (__rocm_in_device),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipDeviceGetGraphMemAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetGraphMemAttribute, hip::GetHipDispatchTable()->hipDeviceGetGraphMemAttribute_fn(device, attr, value));
  CATCH;
}
hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit) {
  auto const __rocm_in_pValue = pValue;
  auto const __rocm_in_limit = limit;
  rocm_trace_emit_hipDeviceGetLimit_enter(
      (const void*)(uintptr_t)(__rocm_in_pValue),
      (int32_t)(__rocm_in_limit)); /* __ROCM_CURATED__: hipDeviceGetLimit */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetLimit, hip::GetHipDispatchTable()->hipDeviceGetLimit_fn(pValue, limit));
  CATCH;
}
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGetMemPool_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGetMemPool */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetMemPool, hip::GetHipDispatchTable()->hipDeviceGetMemPool_fn(mem_pool, device));
  CATCH;
}
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
  auto const __rocm_in_name = name;
  auto const __rocm_in_len = len;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGetName_enter(
      (const char*)(__rocm_in_name),
      (__rocm_in_len),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGetName */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetName, hip::GetHipDispatchTable()->hipDeviceGetName_fn(name, len, device));
  CATCH;
}
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice,
                                    int dstDevice) {
  auto const __rocm_in_value = value;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_srcDevice = srcDevice;
  auto const __rocm_in_dstDevice = dstDevice;
  rocm_trace_emit_hipDeviceGetP2PAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_value),
      (int32_t)(__rocm_in_attr),
      (__rocm_in_srcDevice),
      (__rocm_in_dstDevice)); /* __ROCM_CURATED__: hipDeviceGetP2PAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetP2PAttribute, hip::GetHipDispatchTable()->hipDeviceGetP2PAttribute_fn(value, attr, srcDevice, dstDevice));
  CATCH;
}
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  auto const __rocm_in_pciBusId = pciBusId;
  auto const __rocm_in_len = len;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGetPCIBusId_enter(
      (const char*)(__rocm_in_pciBusId),
      (__rocm_in_len),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGetPCIBusId */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetPCIBusId, hip::GetHipDispatchTable()->hipDeviceGetPCIBusId_fn(pciBusId, len, device));
  CATCH;
}
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  auto const __rocm_in_pConfig = pConfig;
  rocm_trace_emit_hipDeviceGetSharedMemConfig_enter(
      (const void*)(uintptr_t)(__rocm_in_pConfig)); /* __ROCM_CURATED__: hipDeviceGetSharedMemConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetSharedMemConfig, hip::GetHipDispatchTable()->hipDeviceGetSharedMemConfig_fn(pConfig));
  CATCH;
}
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  auto const __rocm_in_leastPriority = leastPriority;
  auto const __rocm_in_greatestPriority = greatestPriority;
  rocm_trace_emit_hipDeviceGetStreamPriorityRange_enter(
      (const void*)(uintptr_t)(__rocm_in_leastPriority),
      (const void*)(uintptr_t)(__rocm_in_greatestPriority)); /* __ROCM_CURATED__: hipDeviceGetStreamPriorityRange */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetStreamPriorityRange, hip::GetHipDispatchTable()->hipDeviceGetStreamPriorityRange_fn(
                                    leastPriority, greatestPriority));
  CATCH;
}
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device) {
  auto const __rocm_in_uuid = uuid;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGetUuid_enter(
      (const void*)(uintptr_t)(__rocm_in_uuid),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGetUuid */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetUuid, hip::GetHipDispatchTable()->hipDeviceGetUuid_fn(uuid, device));
  CATCH;
}
hipError_t hipDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, hipDevice_t device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetLuid_fn(luid, deviceNodeMask, device);
  CATCH;
}
hipError_t hipDeviceGraphMemTrim(int device) {
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGraphMemTrim_enter(
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGraphMemTrim */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGraphMemTrim, hip::GetHipDispatchTable()->hipDeviceGraphMemTrim_fn(device));
  CATCH;
}
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
  auto const __rocm_in_dev = dev;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_active = active;
  rocm_trace_emit_hipDevicePrimaryCtxGetState_enter(
      (__rocm_in_dev),
      (const void*)(uintptr_t)(__rocm_in_flags),
      (const void*)(uintptr_t)(__rocm_in_active)); /* __ROCM_CURATED__: hipDevicePrimaryCtxGetState */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevicePrimaryCtxGetState, hip::GetHipDispatchTable()->hipDevicePrimaryCtxGetState_fn(dev, flags, active));
  CATCH;
}
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  auto const __rocm_in_dev = dev;
  rocm_trace_emit_hipDevicePrimaryCtxRelease_enter(
      (__rocm_in_dev)); /* __ROCM_CURATED__: hipDevicePrimaryCtxRelease */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevicePrimaryCtxRelease, hip::GetHipDispatchTable()->hipDevicePrimaryCtxRelease_fn(dev));
  CATCH;
}
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  auto const __rocm_in_dev = dev;
  rocm_trace_emit_hipDevicePrimaryCtxReset_enter(
      (__rocm_in_dev)); /* __ROCM_CURATED__: hipDevicePrimaryCtxReset */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevicePrimaryCtxReset, hip::GetHipDispatchTable()->hipDevicePrimaryCtxReset_fn(dev));
  CATCH;
}
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  auto const __rocm_in_pctx = pctx;
  auto const __rocm_in_dev = dev;
  rocm_trace_emit_hipDevicePrimaryCtxRetain_enter(
      (const void*)(uintptr_t)(__rocm_in_pctx),
      (__rocm_in_dev)); /* __ROCM_CURATED__: hipDevicePrimaryCtxRetain */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevicePrimaryCtxRetain, hip::GetHipDispatchTable()->hipDevicePrimaryCtxRetain_fn(pctx, dev));
  CATCH;
}
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  auto const __rocm_in_dev = dev;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipDevicePrimaryCtxSetFlags_enter(
      (__rocm_in_dev),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipDevicePrimaryCtxSetFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevicePrimaryCtxSetFlags, hip::GetHipDispatchTable()->hipDevicePrimaryCtxSetFlags_fn(dev, flags));
  CATCH;
}
hipError_t hipDeviceReset(void) {
  rocm_trace_emit_hipDeviceReset_enter(); /* __ROCM_CURATED__: hipDeviceReset */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceReset, hip::GetHipDispatchTable()->hipDeviceReset_fn());
  CATCH;
}
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  auto const __rocm_in_cacheConfig = cacheConfig;
  rocm_trace_emit_hipDeviceSetCacheConfig_enter(
      (int32_t)(__rocm_in_cacheConfig)); /* __ROCM_CURATED__: hipDeviceSetCacheConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSetCacheConfig, hip::GetHipDispatchTable()->hipDeviceSetCacheConfig_fn(cacheConfig));
  CATCH;
}
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipDeviceSetGraphMemAttribute_enter(
      (__rocm_in_device),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipDeviceSetGraphMemAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSetGraphMemAttribute, hip::GetHipDispatchTable()->hipDeviceSetGraphMemAttribute_fn(device, attr, value));
  CATCH;
}
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
  auto const __rocm_in_limit = limit;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipDeviceSetLimit_enter(
      (int32_t)(__rocm_in_limit),
      (__rocm_in_value)); /* __ROCM_CURATED__: hipDeviceSetLimit */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSetLimit, hip::GetHipDispatchTable()->hipDeviceSetLimit_fn(limit, value));
  CATCH;
}
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_mem_pool = mem_pool;
  rocm_trace_emit_hipDeviceSetMemPool_enter(
      (__rocm_in_device),
      (const void*)(uintptr_t)(__rocm_in_mem_pool)); /* __ROCM_CURATED__: hipDeviceSetMemPool */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSetMemPool, hip::GetHipDispatchTable()->hipDeviceSetMemPool_fn(device, mem_pool));
  CATCH;
}
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
  auto const __rocm_in_config = config;
  rocm_trace_emit_hipDeviceSetSharedMemConfig_enter(
      (int32_t)(__rocm_in_config)); /* __ROCM_CURATED__: hipDeviceSetSharedMemConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSetSharedMemConfig, hip::GetHipDispatchTable()->hipDeviceSetSharedMemConfig_fn(config));
  CATCH;
}
hipError_t hipDeviceSynchronize(void) {
  rocm_trace_emit_hipDeviceSynchronize_enter(); /* __ROCM_CURATED__: hipDeviceSynchronize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSynchronize, hip::GetHipDispatchTable()->hipDeviceSynchronize_fn());
  CATCH;
}
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
  auto const __rocm_in_bytes = bytes;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceTotalMem_enter(
      (const void*)(uintptr_t)(__rocm_in_bytes),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceTotalMem */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceTotalMem, hip::GetHipDispatchTable()->hipDeviceTotalMem_fn(bytes, device));
  CATCH;
}
hipError_t hipDriverGetVersion(int* driverVersion) {
  TRY;
  return hip::GetHipDispatchTable()->hipDriverGetVersion_fn(driverVersion);
  CATCH;
}
hipError_t hipDrvGetErrorName(hipError_t hipError, const char** errorString) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGetErrorName_fn(hipError, errorString);
  CATCH;
}
hipError_t hipDrvGetErrorString(hipError_t hipError, const char** errorString) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGetErrorString_fn(hipError, errorString);
  CATCH;
}
hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  auto const __rocm_in_phGraphNode = phGraphNode;
  auto const __rocm_in_hGraph = hGraph;
  auto const __rocm_in_dependencies = dependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_copyParams = copyParams;
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipDrvGraphAddMemcpyNode_enter(
      (const void*)(uintptr_t)(__rocm_in_phGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_hGraph),
      (const void*)(uintptr_t)(__rocm_in_dependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_copyParams),
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipDrvGraphAddMemcpyNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvGraphAddMemcpyNode, hip::GetHipDispatchTable()->hipDrvGraphAddMemcpyNode_fn(phGraphNode, hGraph, dependencies,
                                                              numDependencies, copyParams, ctx));
  CATCH;
}
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
  auto const __rocm_in_pCopy = pCopy;
  rocm_trace_emit_hipDrvMemcpy2DUnaligned_enter(
      (const void*)(uintptr_t)(__rocm_in_pCopy)); /* __ROCM_CURATED__: hipDrvMemcpy2DUnaligned */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvMemcpy2DUnaligned, hip::GetHipDispatchTable()->hipDrvMemcpy2DUnaligned_fn(pCopy));
  CATCH;
}
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
  auto const __rocm_in_pCopy = pCopy;
  rocm_trace_emit_hipDrvMemcpy3D_enter(
      (const void*)(uintptr_t)(__rocm_in_pCopy)); /* __ROCM_CURATED__: hipDrvMemcpy3D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvMemcpy3D, hip::GetHipDispatchTable()->hipDrvMemcpy3D_fn(pCopy));
  CATCH;
}
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
  auto const __rocm_in_pCopy = pCopy;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipDrvMemcpy3DAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_pCopy),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipDrvMemcpy3DAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvMemcpy3DAsync, hip::GetHipDispatchTable()->hipDrvMemcpy3DAsync_fn(pCopy, stream));
  CATCH;
}
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes,
                                      void** data, hipDeviceptr_t ptr) {
  auto const __rocm_in_numAttributes = numAttributes;
  auto const __rocm_in_attributes = attributes;
  auto const __rocm_in_data = data;
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipDrvPointerGetAttributes_enter(
      (__rocm_in_numAttributes),
      (const void*)(uintptr_t)(__rocm_in_attributes),
      (const void*)(uintptr_t)(__rocm_in_data),
      (uint64_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipDrvPointerGetAttributes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvPointerGetAttributes, hip::GetHipDispatchTable()->hipDrvPointerGetAttributes_fn(numAttributes, attributes, data,
                                                                ptr));
  CATCH;
}
hipError_t hipEventCreate(hipEvent_t* event) {
  rocm_trace_emit_hipEventCreate_enter(); /* __ROCM_CURATED__: hipEventCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipEventCreate, hip::GetHipDispatchTable()->hipEventCreate_fn(event), event);
  CATCH;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipEventCreateWithFlags_enter(
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipEventCreateWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipEventCreateWithFlags, hip::GetHipDispatchTable()->hipEventCreateWithFlags_fn(event, flags), event);
  CATCH;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipEventDestroy_enter(
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipEventDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipEventDestroy, hip::GetHipDispatchTable()->hipEventDestroy_fn(event));
  CATCH;
}
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  auto const __rocm_in_start = start;
  auto const __rocm_in_stop = stop;
  rocm_trace_emit_hipEventElapsedTime_enter(
      (uint64_t)(uintptr_t)(__rocm_in_start),
      (uint64_t)(uintptr_t)(__rocm_in_stop)); /* __ROCM_CURATED__: hipEventElapsedTime */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipEventElapsedTime, hip::GetHipDispatchTable()->hipEventElapsedTime_fn(ms, start, stop), ms);
  CATCH;
}
hipError_t hipEventQuery(hipEvent_t event) {
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipEventQuery_enter(
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipEventQuery */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipEventQuery, hip::GetHipDispatchTable()->hipEventQuery_fn(event));
  CATCH;
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  auto const __rocm_in_event = event;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipEventRecord_enter(
      (uint64_t)(uintptr_t)(__rocm_in_event),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipEventRecord */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipEventRecord, hip::GetHipDispatchTable()->hipEventRecord_fn(event, stream));
  CATCH;
}
hipError_t hipEventSynchronize(hipEvent_t event) {
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipEventSynchronize_enter(
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipEventSynchronize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipEventSynchronize, hip::GetHipDispatchTable()->hipEventSynchronize_fn(event));
  CATCH;
}
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype,
                                        uint32_t* hopcount) {
  TRY;
  return hip::GetHipDispatchTable()->hipExtGetLinkTypeAndHopCount_fn(device1, device2, linktype,
                                                                     hopcount);
  CATCH;
}
extern "C" hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks,
                                         dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                         hipStream_t stream, hipEvent_t startEvent,
                                         hipEvent_t stopEvent, int flags) {
  auto const __rocm_in_function_address = function_address;
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_dimBlocks = dimBlocks;
  auto const __rocm_in_args = args;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_startEvent = startEvent;
  auto const __rocm_in_stopEvent = stopEvent;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipExtLaunchKernel_enter(
      (const void*)(uintptr_t)(__rocm_in_function_address),
      (__rocm_in_numBlocks),
      (__rocm_in_dimBlocks),
      (const void*)(uintptr_t)(__rocm_in_args),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (uint64_t)(uintptr_t)(__rocm_in_startEvent),
      (uint64_t)(uintptr_t)(__rocm_in_stopEvent),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipExtLaunchKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExtLaunchKernel, hip::GetHipDispatchTable()->hipExtLaunchKernel_fn(function_address, numBlocks, dimBlocks,
                                                        args, sharedMemBytes, stream, startEvent,
                                                        stopEvent, flags));
  CATCH;
}
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                              unsigned int flags) {
  auto const __rocm_in_launchParamsList = launchParamsList;
  auto const __rocm_in_numDevices = numDevices;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipExtLaunchMultiKernelMultiDevice_enter(
      (const void*)(uintptr_t)(__rocm_in_launchParamsList),
      (__rocm_in_numDevices),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipExtLaunchMultiKernelMultiDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExtLaunchMultiKernelMultiDevice, hip::GetHipDispatchTable()->hipExtLaunchMultiKernelMultiDevice_fn(
                                    launchParamsList, numDevices, flags));
  CATCH;
}
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipExtMallocWithFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_sizeBytes),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipExtMallocWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExtMallocWithFlags, hip::GetHipDispatchTable()->hipExtMallocWithFlags_fn(ptr, sizeBytes, flags));
  CATCH;
}
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_cuMaskSize = cuMaskSize;
  auto const __rocm_in_cuMask = cuMask;
  rocm_trace_emit_hipExtStreamCreateWithCUMask_enter(
      (const void*)(uintptr_t)(__rocm_in_stream),
      (__rocm_in_cuMaskSize),
      (const void*)(uintptr_t)(__rocm_in_cuMask)); /* __ROCM_CURATED__: hipExtStreamCreateWithCUMask */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExtStreamCreateWithCUMask, hip::GetHipDispatchTable()->hipExtStreamCreateWithCUMask_fn(stream, cuMaskSize, cuMask));
  CATCH;
}
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_cuMaskSize = cuMaskSize;
  auto const __rocm_in_cuMask = cuMask;
  rocm_trace_emit_hipExtStreamGetCUMask_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (__rocm_in_cuMaskSize),
      (const void*)(uintptr_t)(__rocm_in_cuMask)); /* __ROCM_CURATED__: hipExtStreamGetCUMask */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExtStreamGetCUMask, hip::GetHipDispatchTable()->hipExtStreamGetCUMask_fn(stream, cuMaskSize, cuMask));
  CATCH;
}
hipError_t hipExternalMemoryGetMappedBuffer(void** devPtr, hipExternalMemory_t extMem,
                                            const hipExternalMemoryBufferDesc* bufferDesc) {
  TRY;
  return hip::GetHipDispatchTable()->hipExternalMemoryGetMappedBuffer_fn(devPtr, extMem,
                                                                         bufferDesc);
  CATCH;
}
hipError_t hipFree(void* ptr) {
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipFree_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipFree */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFree, hip::GetHipDispatchTable()->hipFree_fn(ptr));
  CATCH;
}
hipError_t hipFreeArray(hipArray_t array) {
  auto const __rocm_in_array = array;
  rocm_trace_emit_hipFreeArray_enter(
      (const void*)(uintptr_t)(__rocm_in_array)); /* __ROCM_CURATED__: hipFreeArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFreeArray, hip::GetHipDispatchTable()->hipFreeArray_fn(array));
  CATCH;
}
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipFreeAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipFreeAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFreeAsync, hip::GetHipDispatchTable()->hipFreeAsync_fn(dev_ptr, stream));
  CATCH;
}
hipError_t hipFreeHost(void* ptr) {
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipFreeHost_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipFreeHost */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFreeHost, hip::GetHipDispatchTable()->hipFreeHost_fn(ptr));
  CATCH;
}
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  auto const __rocm_in_mipmappedArray = mipmappedArray;
  rocm_trace_emit_hipFreeMipmappedArray_enter(
      (const void*)(uintptr_t)(__rocm_in_mipmappedArray)); /* __ROCM_CURATED__: hipFreeMipmappedArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFreeMipmappedArray, hip::GetHipDispatchTable()->hipFreeMipmappedArray_fn(mipmappedArray));
  CATCH;
}
hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
  auto const __rocm_in_value = value;
  auto const __rocm_in_attrib = attrib;
  auto const __rocm_in_hfunc = hfunc;
  rocm_trace_emit_hipFuncGetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_value),
      (int32_t)(__rocm_in_attrib),
      (uint64_t)(uintptr_t)(__rocm_in_hfunc)); /* __ROCM_CURATED__: hipFuncGetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFuncGetAttribute, hip::GetHipDispatchTable()->hipFuncGetAttribute_fn(value, attrib, hfunc));
  CATCH;
}
hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func) {
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_func = func;
  rocm_trace_emit_hipFuncGetAttributes_enter(
      (const void*)(uintptr_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_func)); /* __ROCM_CURATED__: hipFuncGetAttributes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFuncGetAttributes, hip::GetHipDispatchTable()->hipFuncGetAttributes_fn(attr, func));
  CATCH;
}
hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value) {
  auto const __rocm_in_func = func;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipFuncSetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_func),
      (int32_t)(__rocm_in_attr),
      (__rocm_in_value)); /* __ROCM_CURATED__: hipFuncSetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFuncSetAttribute, hip::GetHipDispatchTable()->hipFuncSetAttribute_fn(func, attr, value));
  CATCH;
}
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config) {
  auto const __rocm_in_func = func;
  auto const __rocm_in_config = config;
  rocm_trace_emit_hipFuncSetCacheConfig_enter(
      (const void*)(uintptr_t)(__rocm_in_func),
      (int32_t)(__rocm_in_config)); /* __ROCM_CURATED__: hipFuncSetCacheConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFuncSetCacheConfig, hip::GetHipDispatchTable()->hipFuncSetCacheConfig_fn(func, config));
  CATCH;
}
hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config) {
  auto const __rocm_in_func = func;
  auto const __rocm_in_config = config;
  rocm_trace_emit_hipFuncSetSharedMemConfig_enter(
      (const void*)(uintptr_t)(__rocm_in_func),
      (int32_t)(__rocm_in_config)); /* __ROCM_CURATED__: hipFuncSetSharedMemConfig */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipFuncSetSharedMemConfig, hip::GetHipDispatchTable()->hipFuncSetSharedMemConfig_fn(func, config));
  CATCH;
}
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
  TRY;
  return hip::GetHipDispatchTable()->hipGLGetDevices_fn(pHipDeviceCount, pHipDevices,
                                                        hipDeviceCount, deviceList);
  CATCH;
}
hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array) {
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_array = array;
  rocm_trace_emit_hipGetChannelDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_desc),
      (const void*)(uintptr_t)(__rocm_in_array)); /* __ROCM_CURATED__: hipGetChannelDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetChannelDesc, hip::GetHipDispatchTable()->hipGetChannelDesc_fn(desc, array));
  CATCH;
}
hipError_t hipGetDevice(int* deviceId) {
  auto const __rocm_in_deviceId = deviceId;
  rocm_trace_emit_hipGetDevice_enter(
      (const void*)(uintptr_t)(__rocm_in_deviceId)); /* __ROCM_CURATED__: hipGetDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetDevice, hip::GetHipDispatchTable()->hipGetDevice_fn(deviceId));
  CATCH;
}
hipError_t hipGetDeviceCount(int* count) {
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipGetDeviceCount_enter(
      (const void*)(uintptr_t)(__rocm_in_count)); /* __ROCM_CURATED__: hipGetDeviceCount */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetDeviceCount, hip::GetHipDispatchTable()->hipGetDeviceCount_fn(count));
  CATCH;
}
hipError_t hipGetDeviceFlags(unsigned int* flags) {
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGetDeviceFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_flags)); /* __ROCM_CURATED__: hipGetDeviceFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetDeviceFlags, hip::GetHipDispatchTable()->hipGetDeviceFlags_fn(flags));
  CATCH;
}
extern "C" hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int deviceId) {
  auto const __rocm_in_prop = prop;
  auto const __rocm_in_deviceId = deviceId;
  rocm_trace_emit_hipGetDevicePropertiesR0600_enter(
      (const void*)(uintptr_t)(__rocm_in_prop),
      (__rocm_in_deviceId)); /* __ROCM_CURATED__: hipGetDevicePropertiesR0600 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetDevicePropertiesR0600, hip::GetHipDispatchTable()->hipGetDevicePropertiesR0600_fn(prop, deviceId));
  CATCH;
}
extern "C" hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device) {
  auto const __rocm_in_prop = prop;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipGetDevicePropertiesR0000_enter(
      (const void*)(uintptr_t)(__rocm_in_prop),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipGetDevicePropertiesR0000 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetDevicePropertiesR0000, hip::GetHipDispatchTable()->hipGetDevicePropertiesR0000_fn(prop, device));
  CATCH;
}
hipError_t hipGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                  hipDriverEntryPointQueryResult* status) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetDriverEntryPoint_fn(symbol, funcPtr, flags, status);
  CATCH;
}
hipError_t hipGetDriverEntryPoint_spt(const char* symbol, void** funcPtr, unsigned long long flags,
                                      hipDriverEntryPointQueryResult* status) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetDriverEntryPoint_spt_fn(symbol, funcPtr, flags, status);
  CATCH;
}
const char* hipGetErrorName(hipError_t hip_error) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetErrorName_fn(hip_error);
  CATCHRET(const char*);
}
const char* hipGetErrorString(hipError_t hipError) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetErrorString_fn(hipError);
  CATCHRET(const char*);
}
hipError_t hipGetLastError(void) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetLastError_fn();
  CATCH;
}
hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,
                                     hipMipmappedArray_const_t mipmappedArray, unsigned int level) {
  auto const __rocm_in_levelArray = levelArray;
  auto const __rocm_in_mipmappedArray = mipmappedArray;
  auto const __rocm_in_level = level;
  rocm_trace_emit_hipGetMipmappedArrayLevel_enter(
      (const void*)(uintptr_t)(__rocm_in_levelArray),
      (const void*)(uintptr_t)(__rocm_in_mipmappedArray),
      (__rocm_in_level)); /* __ROCM_CURATED__: hipGetMipmappedArrayLevel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetMipmappedArrayLevel, hip::GetHipDispatchTable()->hipGetMipmappedArrayLevel_fn(levelArray, mipmappedArray, level));
  CATCH;
}
hipError_t hipExternalMemoryGetMappedMipmappedArray(
    hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem,
    const hipExternalMemoryMipmappedArrayDesc* mipmapDesc) {
  TRY;
  return hip::GetHipDispatchTable()->hipExternalMemoryGetMappedMipmappedArray_fn(mipmap, extMem,
                                                                                 mipmapDesc);
  CATCH;
}
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetSymbolAddress_fn(devPtr, symbol);
  CATCH;
}
hipError_t hipGetSymbolSize(size_t* size, const void* symbol) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetSymbolSize_fn(size, symbol);
  CATCH;
}
hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* texref) {
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_texref = texref;
  rocm_trace_emit_hipGetTextureAlignmentOffset_enter(
      (const void*)(uintptr_t)(__rocm_in_offset),
      (const void*)(uintptr_t)(__rocm_in_texref)); /* __ROCM_CURATED__: hipGetTextureAlignmentOffset */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetTextureAlignmentOffset, hip::GetHipDispatchTable()->hipGetTextureAlignmentOffset_fn(offset, texref));
  CATCH;
}
hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
  auto const __rocm_in_pResDesc = pResDesc;
  auto const __rocm_in_textureObject = textureObject;
  rocm_trace_emit_hipGetTextureObjectResourceDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_pResDesc),
      (const void*)(uintptr_t)(__rocm_in_textureObject)); /* __ROCM_CURATED__: hipGetTextureObjectResourceDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetTextureObjectResourceDesc, hip::GetHipDispatchTable()->hipGetTextureObjectResourceDesc_fn(pResDesc, textureObject));
  CATCH;
}
hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject) {
  auto const __rocm_in_pResViewDesc = pResViewDesc;
  auto const __rocm_in_textureObject = textureObject;
  rocm_trace_emit_hipGetTextureObjectResourceViewDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_pResViewDesc),
      (const void*)(uintptr_t)(__rocm_in_textureObject)); /* __ROCM_CURATED__: hipGetTextureObjectResourceViewDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetTextureObjectResourceViewDesc, hip::GetHipDispatchTable()->hipGetTextureObjectResourceViewDesc_fn(
                                    pResViewDesc, textureObject));
  CATCH;
}
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject) {
  auto const __rocm_in_pTexDesc = pTexDesc;
  auto const __rocm_in_textureObject = textureObject;
  rocm_trace_emit_hipGetTextureObjectTextureDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_pTexDesc),
      (const void*)(uintptr_t)(__rocm_in_textureObject)); /* __ROCM_CURATED__: hipGetTextureObjectTextureDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetTextureObjectTextureDesc, hip::GetHipDispatchTable()->hipGetTextureObjectTextureDesc_fn(pTexDesc, textureObject));
  CATCH;
}
hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol) {
  auto const __rocm_in_texref = texref;
  auto const __rocm_in_symbol = symbol;
  rocm_trace_emit_hipGetTextureReference_enter(
      (const void*)(uintptr_t)(__rocm_in_texref),
      (const void*)(uintptr_t)(__rocm_in_symbol)); /* __ROCM_CURATED__: hipGetTextureReference */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGetTextureReference, hip::GetHipDispatchTable()->hipGetTextureReference_fn(texref, symbol));
  CATCH;
}
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_childGraph = childGraph;
  rocm_trace_emit_hipGraphAddChildGraphNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (uint64_t)(uintptr_t)(__rocm_in_childGraph)); /* __ROCM_CURATED__: hipGraphAddChildGraphNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddChildGraphNode, hip::GetHipDispatchTable()->hipGraphAddChildGraphNode_fn(pGraphNode, graph, pDependencies,
                                                               numDependencies, childGraph));
  CATCH;
}
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_from = from;
  auto const __rocm_in_to = to;
  auto const __rocm_in_numDependencies = numDependencies;
  rocm_trace_emit_hipGraphAddDependencies_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_from),
      (const void*)(uintptr_t)(__rocm_in_to),
      (__rocm_in_numDependencies)); /* __ROCM_CURATED__: hipGraphAddDependencies */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddDependencies, hip::GetHipDispatchTable()->hipGraphAddDependencies_fn(graph, from, to, numDependencies));
  CATCH;
}
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  rocm_trace_emit_hipGraphAddEmptyNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies)); /* __ROCM_CURATED__: hipGraphAddEmptyNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddEmptyNode, hip::GetHipDispatchTable()->hipGraphAddEmptyNode_fn(pGraphNode, graph, pDependencies,
                                                          numDependencies));
  CATCH;
}
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipGraphAddEventRecordNode_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipGraphAddEventRecordNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipGraphAddEventRecordNode, hip::GetHipDispatchTable()->hipGraphAddEventRecordNode_fn(
                                    pGraphNode, graph, pDependencies, numDependencies, event), pGraphNode);
  CATCH;
}
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipGraphAddEventWaitNode_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipGraphAddEventWaitNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipGraphAddEventWaitNode, hip::GetHipDispatchTable()->hipGraphAddEventWaitNode_fn(
                                    pGraphNode, graph, pDependencies, numDependencies, event), pGraphNode);
  CATCH;
}
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphAddHostNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphAddHostNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddHostNode, hip::GetHipDispatchTable()->hipGraphAddHostNode_fn(pGraphNode, graph, pDependencies,
                                                         numDependencies, pNodeParams));
  CATCH;
}
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphAddKernelNode_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphAddKernelNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipGraphAddKernelNode, hip::GetHipDispatchTable()->hipGraphAddKernelNode_fn(
                                    pGraphNode, graph, pDependencies, numDependencies, pNodeParams), pGraphNode);
  CATCH;
}
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   hipMemAllocNodeParams* pNodeParams) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphAddMemAllocNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphAddMemAllocNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddMemAllocNode, hip::GetHipDispatchTable()->hipGraphAddMemAllocNode_fn(pGraphNode, graph, pDependencies,
                                                             numDependencies, pNodeParams));
  CATCH;
}
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  void* dev_ptr) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_dev_ptr = dev_ptr;
  rocm_trace_emit_hipGraphAddMemFreeNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_dev_ptr)); /* __ROCM_CURATED__: hipGraphAddMemFreeNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddMemFreeNode, hip::GetHipDispatchTable()->hipGraphAddMemFreeNode_fn(pGraphNode, graph, pDependencies,
                                                            numDependencies, dev_ptr));
  CATCH;
}
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_pCopyParams = pCopyParams;
  rocm_trace_emit_hipGraphAddMemcpyNode_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_pCopyParams)); /* __ROCM_CURATED__: hipGraphAddMemcpyNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipGraphAddMemcpyNode, hip::GetHipDispatchTable()->hipGraphAddMemcpyNode_fn(
                                    pGraphNode, graph, pDependencies, numDependencies, pCopyParams), pGraphNode);
  CATCH;
}
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_count = count;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphAddMemcpyNode1D_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_count),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphAddMemcpyNode1D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddMemcpyNode1D, hip::GetHipDispatchTable()->hipGraphAddMemcpyNode1D_fn(
          pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind));
  CATCH;
}
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_count = count;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphAddMemcpyNodeFromSymbol_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (__rocm_in_count),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphAddMemcpyNodeFromSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddMemcpyNodeFromSymbol, hip::GetHipDispatchTable()->hipGraphAddMemcpyNodeFromSymbol_fn(
          pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind));
  CATCH;
}
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_src = src;
  auto const __rocm_in_count = count;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphAddMemcpyNodeToSymbol_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_count),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphAddMemcpyNodeToSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddMemcpyNodeToSymbol, hip::GetHipDispatchTable()->hipGraphAddMemcpyNodeToSymbol_fn(
          pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind));
  CATCH;
}
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_pMemsetParams = pMemsetParams;
  rocm_trace_emit_hipGraphAddMemsetNode_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_pMemsetParams)); /* __ROCM_CURATED__: hipGraphAddMemsetNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipGraphAddMemsetNode, hip::GetHipDispatchTable()->hipGraphAddMemsetNode_fn(pGraphNode, graph, pDependencies,
                                                           numDependencies, pMemsetParams), pGraphNode);
  CATCH;
}
hipError_t hipGraphAddNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                           const hipGraphNode_t* pDependencies, size_t numDependencies,
                           hipGraphNodeParams* nodeParams) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphAddNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphAddNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddNode, hip::GetHipDispatchTable()->hipGraphAddNode_fn(pGraphNode, graph, pDependencies,
                                                     numDependencies, nodeParams));
  CATCH;
}
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pGraph = pGraph;
  rocm_trace_emit_hipGraphChildGraphNodeGetGraph_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pGraph)); /* __ROCM_CURATED__: hipGraphChildGraphNodeGetGraph */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphChildGraphNodeGetGraph, hip::GetHipDispatchTable()->hipGraphChildGraphNodeGetGraph_fn(node, pGraph));
  CATCH;
}
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  auto const __rocm_in_pGraphClone = pGraphClone;
  auto const __rocm_in_originalGraph = originalGraph;
  rocm_trace_emit_hipGraphClone_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphClone),
      (uint64_t)(uintptr_t)(__rocm_in_originalGraph)); /* __ROCM_CURATED__: hipGraphClone */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphClone, hip::GetHipDispatchTable()->hipGraphClone_fn(pGraphClone, originalGraph));
  CATCH;
}
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGraphCreate_enter(
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipGraphCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipGraphCreate, hip::GetHipDispatchTable()->hipGraphCreate_fn(pGraph, flags), pGraph);
  CATCH;
}
hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_path = path;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGraphDebugDotPrint_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const char*)(__rocm_in_path),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipGraphDebugDotPrint */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphDebugDotPrint, hip::GetHipDispatchTable()->hipGraphDebugDotPrint_fn(graph, path, flags));
  CATCH;
}
hipError_t hipGraphDestroy(hipGraph_t graph) {
  auto const __rocm_in_graph = graph;
  rocm_trace_emit_hipGraphDestroy_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph)); /* __ROCM_CURATED__: hipGraphDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphDestroy, hip::GetHipDispatchTable()->hipGraphDestroy_fn(graph));
  CATCH;
}
hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  auto const __rocm_in_node = node;
  rocm_trace_emit_hipGraphDestroyNode_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node)); /* __ROCM_CURATED__: hipGraphDestroyNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphDestroyNode, hip::GetHipDispatchTable()->hipGraphDestroyNode_fn(node));
  CATCH;
}
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_event_out = event_out;
  rocm_trace_emit_hipGraphEventRecordNodeGetEvent_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_event_out)); /* __ROCM_CURATED__: hipGraphEventRecordNodeGetEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphEventRecordNodeGetEvent, hip::GetHipDispatchTable()->hipGraphEventRecordNodeGetEvent_fn(node, event_out));
  CATCH;
}
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipGraphEventRecordNodeSetEvent_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipGraphEventRecordNodeSetEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphEventRecordNodeSetEvent, hip::GetHipDispatchTable()->hipGraphEventRecordNodeSetEvent_fn(node, event));
  CATCH;
}
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_event_out = event_out;
  rocm_trace_emit_hipGraphEventWaitNodeGetEvent_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_event_out)); /* __ROCM_CURATED__: hipGraphEventWaitNodeGetEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphEventWaitNodeGetEvent, hip::GetHipDispatchTable()->hipGraphEventWaitNodeGetEvent_fn(node, event_out));
  CATCH;
}
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipGraphEventWaitNodeSetEvent_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipGraphEventWaitNodeSetEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphEventWaitNodeSetEvent, hip::GetHipDispatchTable()->hipGraphEventWaitNodeSetEvent_fn(node, event));
  CATCH;
}
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_childGraph = childGraph;
  rocm_trace_emit_hipGraphExecChildGraphNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (uint64_t)(uintptr_t)(__rocm_in_childGraph)); /* __ROCM_CURATED__: hipGraphExecChildGraphNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecChildGraphNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecChildGraphNodeSetParams_fn(
                                    hGraphExec, node, childGraph));
  CATCH;
}
hipError_t hipGraphExecNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                     hipGraphNodeParams* nodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphExecNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphExecNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecNodeSetParams_fn(hGraphExec, node, nodeParams));
  CATCH;
}
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
  auto const __rocm_in_graphExec = graphExec;
  rocm_trace_emit_hipGraphExecDestroy_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graphExec)); /* __ROCM_CURATED__: hipGraphExecDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecDestroy, hip::GetHipDispatchTable()->hipGraphExecDestroy_fn(graphExec));
  CATCH;
}
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipGraphExecEventRecordNodeSetEvent_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipGraphExecEventRecordNodeSetEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecEventRecordNodeSetEvent, hip::GetHipDispatchTable()->hipGraphExecEventRecordNodeSetEvent_fn(hGraphExec, hNode, event));
  CATCH;
}
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipGraphExecEventWaitNodeSetEvent_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipGraphExecEventWaitNodeSetEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecEventWaitNodeSetEvent, hip::GetHipDispatchTable()->hipGraphExecEventWaitNodeSetEvent_fn(hGraphExec, hNode, event));
  CATCH;
}
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphExecHostNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphExecHostNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecHostNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecHostNodeSetParams_fn(hGraphExec, node, pNodeParams));
  CATCH;
}
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphExecKernelNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphExecKernelNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecKernelNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecKernelNodeSetParams_fn(hGraphExec, node, pNodeParams));
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphExecMemcpyNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphExecMemcpyNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecMemcpyNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParams_fn(hGraphExec, node, pNodeParams));
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_count = count;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphExecMemcpyNodeSetParams1D_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_count),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphExecMemcpyNodeSetParams1D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecMemcpyNodeSetParams1D, hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParams1D_fn(hGraphExec, node, dst, src,
                                                                       count, kind));
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_count = count;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphExecMemcpyNodeSetParamsFromSymbol_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (__rocm_in_count),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphExecMemcpyNodeSetParamsFromSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecMemcpyNodeSetParamsFromSymbol, hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParamsFromSymbol_fn(
          hGraphExec, node, dst, symbol, count, offset, kind));
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset,
                                                   hipMemcpyKind kind) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_src = src;
  auto const __rocm_in_count = count;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphExecMemcpyNodeSetParamsToSymbol_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_count),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphExecMemcpyNodeSetParamsToSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecMemcpyNodeSetParamsToSymbol, hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParamsToSymbol_fn(
          hGraphExec, node, symbol, src, count, offset, kind));
  CATCH;
}
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphExecMemsetNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphExecMemsetNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecMemsetNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecMemsetNodeSetParams_fn(hGraphExec, node, pNodeParams));
  CATCH;
}
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hGraph = hGraph;
  auto const __rocm_in_hErrorNode_out = hErrorNode_out;
  auto const __rocm_in_updateResult_out = updateResult_out;
  rocm_trace_emit_hipGraphExecUpdate_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hGraph),
      (const void*)(uintptr_t)(__rocm_in_hErrorNode_out),
      (const void*)(uintptr_t)(__rocm_in_updateResult_out)); /* __ROCM_CURATED__: hipGraphExecUpdate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecUpdate, hip::GetHipDispatchTable()->hipGraphExecUpdate_fn(
                                    hGraphExec, hGraph, hErrorNode_out, updateResult_out));
  CATCH;
}
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_from = from;
  auto const __rocm_in_to = to;
  auto const __rocm_in_numEdges = numEdges;
  rocm_trace_emit_hipGraphGetEdges_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_from),
      (const void*)(uintptr_t)(__rocm_in_to),
      (const void*)(uintptr_t)(__rocm_in_numEdges)); /* __ROCM_CURATED__: hipGraphGetEdges */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphGetEdges, hip::GetHipDispatchTable()->hipGraphGetEdges_fn(graph, from, to, numEdges));
  CATCH;
}
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_nodes = nodes;
  auto const __rocm_in_numNodes = numNodes;
  rocm_trace_emit_hipGraphGetNodes_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_nodes),
      (const void*)(uintptr_t)(__rocm_in_numNodes)); /* __ROCM_CURATED__: hipGraphGetNodes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphGetNodes, hip::GetHipDispatchTable()->hipGraphGetNodes_fn(graph, nodes, numNodes));
  CATCH;
}
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pRootNodes = pRootNodes;
  auto const __rocm_in_pNumRootNodes = pNumRootNodes;
  rocm_trace_emit_hipGraphGetRootNodes_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pRootNodes),
      (const void*)(uintptr_t)(__rocm_in_pNumRootNodes)); /* __ROCM_CURATED__: hipGraphGetRootNodes */
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphGetRootNodes, hip::GetHipDispatchTable()->hipGraphGetRootNodes_fn(graph, pRootNodes, pNumRootNodes));
}
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphHostNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphHostNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphHostNodeGetParams, hip::GetHipDispatchTable()->hipGraphHostNodeGetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphHostNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphHostNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphHostNodeSetParams, hip::GetHipDispatchTable()->hipGraphHostNodeSetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pErrorNode = pErrorNode;
  auto const __rocm_in_pLogBuffer = pLogBuffer;
  auto const __rocm_in_bufferSize = bufferSize;
  rocm_trace_emit_hipGraphInstantiate_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pErrorNode),
      (const void*)(uintptr_t)(__rocm_in_pLogBuffer),
      (__rocm_in_bufferSize)); /* __ROCM_CURATED__: hipGraphInstantiate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipGraphInstantiate, hip::GetHipDispatchTable()->hipGraphInstantiate_fn(
                                    pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize), pGraphExec);
  CATCH;
}
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags) {
  auto const __rocm_in_pGraphExec = pGraphExec;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGraphInstantiateWithFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipGraphInstantiateWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphInstantiateWithFlags, hip::GetHipDispatchTable()->hipGraphInstantiateWithFlags_fn(pGraphExec, graph, flags));
  CATCH;
}
hipError_t hipGraphInstantiateWithParams(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                         hipGraphInstantiateParams* instantiateParams) {
  auto const __rocm_in_pGraphExec = pGraphExec;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_instantiateParams = instantiateParams;
  rocm_trace_emit_hipGraphInstantiateWithParams_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_instantiateParams)); /* __ROCM_CURATED__: hipGraphInstantiateWithParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphInstantiateWithParams, hip::GetHipDispatchTable()->hipGraphInstantiateWithParams_fn(
                                    pGraphExec, graph, instantiateParams));
  CATCH;
}
hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst) {
  auto const __rocm_in_hSrc = hSrc;
  auto const __rocm_in_hDst = hDst;
  rocm_trace_emit_hipGraphKernelNodeCopyAttributes_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hSrc),
      (uint64_t)(uintptr_t)(__rocm_in_hDst)); /* __ROCM_CURATED__: hipGraphKernelNodeCopyAttributes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphKernelNodeCopyAttributes, hip::GetHipDispatchTable()->hipGraphKernelNodeCopyAttributes_fn(hSrc, hDst));
  CATCH;
}
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipGraphKernelNodeGetAttribute_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipGraphKernelNodeGetAttribute */
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphKernelNodeGetAttribute, hip::GetHipDispatchTable()->hipGraphKernelNodeGetAttribute_fn(hNode, attr, value));
}
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphKernelNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphKernelNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphKernelNodeGetParams, hip::GetHipDispatchTable()->hipGraphKernelNodeGetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipGraphKernelNodeSetAttribute_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipGraphKernelNodeSetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphKernelNodeSetAttribute, hip::GetHipDispatchTable()->hipGraphKernelNodeSetAttribute_fn(hNode, attr, value));
  CATCH;
}
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                       const hipKernelNodeParams* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphKernelNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphKernelNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphKernelNodeSetParams, hip::GetHipDispatchTable()->hipGraphKernelNodeSetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  auto const __rocm_in_graphExec = graphExec;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipGraphLaunch_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graphExec),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipGraphLaunch */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphLaunch, hip::GetHipDispatchTable()->hipGraphLaunch_fn(graphExec, stream));
  CATCH;
}
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphMemAllocNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphMemAllocNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemAllocNodeGetParams, hip::GetHipDispatchTable()->hipGraphMemAllocNodeGetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_dev_ptr = dev_ptr;
  rocm_trace_emit_hipGraphMemFreeNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_dev_ptr)); /* __ROCM_CURATED__: hipGraphMemFreeNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemFreeNodeGetParams, hip::GetHipDispatchTable()->hipGraphMemFreeNodeGetParams_fn(node, dev_ptr));
  CATCH;
}
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphMemcpyNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphMemcpyNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemcpyNodeGetParams, hip::GetHipDispatchTable()->hipGraphMemcpyNodeGetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphMemcpyNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphMemcpyNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemcpyNodeSetParams, hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_count = count;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphMemcpyNodeSetParams1D_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_count),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphMemcpyNodeSetParams1D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemcpyNodeSetParams1D, hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParams1D_fn(node, dst, src, count, kind));
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_count = count;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphMemcpyNodeSetParamsFromSymbol_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (__rocm_in_count),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphMemcpyNodeSetParamsFromSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemcpyNodeSetParamsFromSymbol, hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParamsFromSymbol_fn(node, dst, symbol, count,
                                                                           offset, kind));
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_src = src;
  auto const __rocm_in_count = count;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipGraphMemcpyNodeSetParamsToSymbol_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_count),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipGraphMemcpyNodeSetParamsToSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemcpyNodeSetParamsToSymbol, hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParamsToSymbol_fn(
                                    node, symbol, src, count, offset, kind));
  CATCH;
}
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphMemsetNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphMemsetNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemsetNodeGetParams, hip::GetHipDispatchTable()->hipGraphMemsetNodeGetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pNodeParams = pNodeParams;
  rocm_trace_emit_hipGraphMemsetNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pNodeParams)); /* __ROCM_CURATED__: hipGraphMemsetNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphMemsetNodeSetParams, hip::GetHipDispatchTable()->hipGraphMemsetNodeSetParams_fn(node, pNodeParams));
  CATCH;
}
hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph) {
  auto const __rocm_in_pNode = pNode;
  auto const __rocm_in_originalNode = originalNode;
  auto const __rocm_in_clonedGraph = clonedGraph;
  rocm_trace_emit_hipGraphNodeFindInClone_enter(
      (const void*)(uintptr_t)(__rocm_in_pNode),
      (uint64_t)(uintptr_t)(__rocm_in_originalNode),
      (uint64_t)(uintptr_t)(__rocm_in_clonedGraph)); /* __ROCM_CURATED__: hipGraphNodeFindInClone */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphNodeFindInClone, hip::GetHipDispatchTable()->hipGraphNodeFindInClone_fn(pNode, originalNode, clonedGraph));
  CATCH;
}
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_pNumDependencies = pNumDependencies;
  rocm_trace_emit_hipGraphNodeGetDependencies_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (const void*)(uintptr_t)(__rocm_in_pNumDependencies)); /* __ROCM_CURATED__: hipGraphNodeGetDependencies */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphNodeGetDependencies, hip::GetHipDispatchTable()->hipGraphNodeGetDependencies_fn(
                                    node, pDependencies, pNumDependencies));
  CATCH;
}
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pDependentNodes = pDependentNodes;
  auto const __rocm_in_pNumDependentNodes = pNumDependentNodes;
  rocm_trace_emit_hipGraphNodeGetDependentNodes_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pDependentNodes),
      (const void*)(uintptr_t)(__rocm_in_pNumDependentNodes)); /* __ROCM_CURATED__: hipGraphNodeGetDependentNodes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphNodeGetDependentNodes, hip::GetHipDispatchTable()->hipGraphNodeGetDependentNodes_fn(
                                    node, pDependentNodes, pNumDependentNodes));
  CATCH;
}
hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_isEnabled = isEnabled;
  rocm_trace_emit_hipGraphNodeGetEnabled_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_isEnabled)); /* __ROCM_CURATED__: hipGraphNodeGetEnabled */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphNodeGetEnabled, hip::GetHipDispatchTable()->hipGraphNodeGetEnabled_fn(hGraphExec, hNode, isEnabled));
  CATCH;
}
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_pType = pType;
  rocm_trace_emit_hipGraphNodeGetType_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_pType)); /* __ROCM_CURATED__: hipGraphNodeGetType */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphNodeGetType, hip::GetHipDispatchTable()->hipGraphNodeGetType_fn(node, pType));
  CATCH;
}
hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_isEnabled = isEnabled;
  rocm_trace_emit_hipGraphNodeSetEnabled_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (__rocm_in_isEnabled)); /* __ROCM_CURATED__: hipGraphNodeSetEnabled */
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphNodeSetEnabled, hip::GetHipDispatchTable()->hipGraphNodeSetEnabled_fn(hGraphExec, hNode, isEnabled));
}
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_object = object;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipGraphReleaseUserObject_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_object),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipGraphReleaseUserObject */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphReleaseUserObject, hip::GetHipDispatchTable()->hipGraphReleaseUserObject_fn(graph, object, count));
  CATCH;
}
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_from = from;
  auto const __rocm_in_to = to;
  auto const __rocm_in_numDependencies = numDependencies;
  rocm_trace_emit_hipGraphRemoveDependencies_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_from),
      (const void*)(uintptr_t)(__rocm_in_to),
      (__rocm_in_numDependencies)); /* __ROCM_CURATED__: hipGraphRemoveDependencies */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphRemoveDependencies, hip::GetHipDispatchTable()->hipGraphRemoveDependencies_fn(graph, from, to, numDependencies));
  CATCH;
}
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count,
                                    unsigned int flags) {
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_object = object;
  auto const __rocm_in_count = count;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGraphRetainUserObject_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_object),
      (__rocm_in_count),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipGraphRetainUserObject */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphRetainUserObject, hip::GetHipDispatchTable()->hipGraphRetainUserObject_fn(graph, object, count, flags));
  CATCH;
}
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  auto const __rocm_in_graphExec = graphExec;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipGraphUpload_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graphExec),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipGraphUpload */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphUpload, hip::GetHipDispatchTable()->hipGraphUpload_fn(graphExec, stream));
  CATCH;
}
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags) {
  auto const __rocm_in_resource = resource;
  auto const __rocm_in_buffer = buffer;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGraphicsGLRegisterBuffer_enter(
      (const void*)(uintptr_t)(__rocm_in_resource),
      (__rocm_in_buffer),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipGraphicsGLRegisterBuffer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphicsGLRegisterBuffer, hip::GetHipDispatchTable()->hipGraphicsGLRegisterBuffer_fn(resource, buffer, flags));
  CATCH;
}
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags) {
  auto const __rocm_in_resource = resource;
  auto const __rocm_in_image = image;
  auto const __rocm_in_target = target;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGraphicsGLRegisterImage_enter(
      (const void*)(uintptr_t)(__rocm_in_resource),
      (__rocm_in_image),
      (__rocm_in_target),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipGraphicsGLRegisterImage */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphicsGLRegisterImage, hip::GetHipDispatchTable()->hipGraphicsGLRegisterImage_fn(resource, image, target, flags));
  CATCH;
}
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream) {
  auto const __rocm_in_count = count;
  auto const __rocm_in_resources = resources;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipGraphicsMapResources_enter(
      (__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_resources),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipGraphicsMapResources */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphicsMapResources, hip::GetHipDispatchTable()->hipGraphicsMapResources_fn(count, resources, stream));
  CATCH;
}
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource) {
  auto const __rocm_in_devPtr = devPtr;
  auto const __rocm_in_size = size;
  auto const __rocm_in_resource = resource;
  rocm_trace_emit_hipGraphicsResourceGetMappedPointer_enter(
      (const void*)(uintptr_t)(__rocm_in_devPtr),
      (const void*)(uintptr_t)(__rocm_in_size),
      (const void*)(uintptr_t)(__rocm_in_resource)); /* __ROCM_CURATED__: hipGraphicsResourceGetMappedPointer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphicsResourceGetMappedPointer, hip::GetHipDispatchTable()->hipGraphicsResourceGetMappedPointer_fn(devPtr, size, resource));
  CATCH;
}
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource,
                                                unsigned int arrayIndex, unsigned int mipLevel) {
  auto const __rocm_in_array = array;
  auto const __rocm_in_resource = resource;
  auto const __rocm_in_arrayIndex = arrayIndex;
  auto const __rocm_in_mipLevel = mipLevel;
  rocm_trace_emit_hipGraphicsSubResourceGetMappedArray_enter(
      (const void*)(uintptr_t)(__rocm_in_array),
      (const void*)(uintptr_t)(__rocm_in_resource),
      (__rocm_in_arrayIndex),
      (__rocm_in_mipLevel)); /* __ROCM_CURATED__: hipGraphicsSubResourceGetMappedArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphicsSubResourceGetMappedArray, hip::GetHipDispatchTable()->hipGraphicsSubResourceGetMappedArray_fn(
                                    array, resource, arrayIndex, mipLevel));
  CATCH;
}
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream) {
  auto const __rocm_in_count = count;
  auto const __rocm_in_resources = resources;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipGraphicsUnmapResources_enter(
      (__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_resources),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipGraphicsUnmapResources */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphicsUnmapResources, hip::GetHipDispatchTable()->hipGraphicsUnmapResources_fn(count, resources, stream));
  CATCH;
}
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
  auto const __rocm_in_resource = resource;
  rocm_trace_emit_hipGraphicsUnregisterResource_enter(
      (const void*)(uintptr_t)(__rocm_in_resource)); /* __ROCM_CURATED__: hipGraphicsUnregisterResource */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphicsUnregisterResource, hip::GetHipDispatchTable()->hipGraphicsUnregisterResource_fn(resource));
  CATCH;
}
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_size = size;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipHostAlloc_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_size),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipHostAlloc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipHostAlloc, hip::GetHipDispatchTable()->hipHostAlloc_fn(ptr, size, flags));
  CATCH;
}
hipError_t hipHostFree(void* ptr) {
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipHostFree_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipHostFree */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipHostFree, hip::GetHipDispatchTable()->hipHostFree_fn(ptr));
  CATCH;
}
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
  auto const __rocm_in_devPtr = devPtr;
  auto const __rocm_in_hstPtr = hstPtr;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipHostGetDevicePointer_enter(
      (const void*)(uintptr_t)(__rocm_in_devPtr),
      (const void*)(uintptr_t)(__rocm_in_hstPtr),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipHostGetDevicePointer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipHostGetDevicePointer, hip::GetHipDispatchTable()->hipHostGetDevicePointer_fn(devPtr, hstPtr, flags));
  CATCH;
}
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  auto const __rocm_in_flagsPtr = flagsPtr;
  auto const __rocm_in_hostPtr = hostPtr;
  rocm_trace_emit_hipHostGetFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_flagsPtr),
      (const void*)(uintptr_t)(__rocm_in_hostPtr)); /* __ROCM_CURATED__: hipHostGetFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipHostGetFlags, hip::GetHipDispatchTable()->hipHostGetFlags_fn(flagsPtr, hostPtr));
  CATCH;
}
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  auto const __rocm_in_size = size;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipHostMalloc_enter(
      (__rocm_in_size),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipHostMalloc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipHostMalloc, hip::GetHipDispatchTable()->hipHostMalloc_fn(ptr, size, flags), ptr);
  CATCH;
}
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  auto const __rocm_in_hostPtr = hostPtr;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipHostRegister_enter(
      (const void*)(uintptr_t)(__rocm_in_hostPtr),
      (__rocm_in_sizeBytes),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipHostRegister */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipHostRegister, hip::GetHipDispatchTable()->hipHostRegister_fn(hostPtr, sizeBytes, flags));
  CATCH;
}
hipError_t hipHostUnregister(void* hostPtr) {
  auto const __rocm_in_hostPtr = hostPtr;
  rocm_trace_emit_hipHostUnregister_enter(
      (const void*)(uintptr_t)(__rocm_in_hostPtr)); /* __ROCM_CURATED__: hipHostUnregister */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipHostUnregister, hip::GetHipDispatchTable()->hipHostUnregister_fn(hostPtr));
  CATCH;
}
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out,
                                   const hipExternalMemoryHandleDesc* memHandleDesc) {
  TRY;
  return hip::GetHipDispatchTable()->hipImportExternalMemory_fn(extMem_out, memHandleDesc);
  CATCH;
}
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc) {
  TRY;
  return hip::GetHipDispatchTable()->hipImportExternalSemaphore_fn(extSem_out, semHandleDesc);
  CATCH;
}
hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  auto const __rocm_in_phGraphNode = phGraphNode;
  auto const __rocm_in_hGraph = hGraph;
  auto const __rocm_in_dependencies = dependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_memsetParams = memsetParams;
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipDrvGraphAddMemsetNode_enter(
      (const void*)(uintptr_t)(__rocm_in_phGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_hGraph),
      (const void*)(uintptr_t)(__rocm_in_dependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_memsetParams),
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipDrvGraphAddMemsetNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvGraphAddMemsetNode, hip::GetHipDispatchTable()->hipDrvGraphAddMemsetNode_fn(phGraphNode, hGraph, dependencies,
                                                              numDependencies, memsetParams, ctx));
  CATCH;
}
hipError_t hipInit(unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipInit_fn(flags);
  CATCH;
}
hipError_t hipInitDevice(int device, unsigned int deviceFlags, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipInitDevice_fn(device, deviceFlags, flags);
  CATCH;
}
hipError_t hipIpcCloseMemHandle(void* devPtr) {
  auto const __rocm_in_devPtr = devPtr;
  rocm_trace_emit_hipIpcCloseMemHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_devPtr)); /* __ROCM_CURATED__: hipIpcCloseMemHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipIpcCloseMemHandle, hip::GetHipDispatchTable()->hipIpcCloseMemHandle_fn(devPtr));
  CATCH;
}
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipIpcGetEventHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_handle),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipIpcGetEventHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipIpcGetEventHandle, hip::GetHipDispatchTable()->hipIpcGetEventHandle_fn(handle, event));
  CATCH;
}
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_devPtr = devPtr;
  rocm_trace_emit_hipIpcGetMemHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_handle),
      (const void*)(uintptr_t)(__rocm_in_devPtr)); /* __ROCM_CURATED__: hipIpcGetMemHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipIpcGetMemHandle, hip::GetHipDispatchTable()->hipIpcGetMemHandle_fn(handle, devPtr));
  CATCH;
}
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipIpcOpenEventHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipIpcOpenEventHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipIpcOpenEventHandle, hip::GetHipDispatchTable()->hipIpcOpenEventHandle_fn(event, handle));
  CATCH;
}
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
  auto const __rocm_in_devPtr = devPtr;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipIpcOpenMemHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_devPtr),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipIpcOpenMemHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipIpcOpenMemHandle, hip::GetHipDispatchTable()->hipIpcOpenMemHandle_fn(devPtr, handle, flags));
  CATCH;
}
extern "C" const char* hipKernelNameRef(const hipFunction_t f) {
  auto const __rocm_in_f = f;
  rocm_trace_emit_hipKernelNameRef_enter(
      (uint64_t)(uintptr_t)(__rocm_in_f)); /* __ROCM_CURATED__: hipKernelNameRef */
  TRY;
  ROCM_TRACE_RET_PTR_CURATED_NOARGS(hipKernelNameRef, auto, hip::GetHipDispatchTable()->hipKernelNameRef_fn(f));
  CATCHRET(const char*);
}
extern "C" const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream) {
  auto const __rocm_in_hostFunction = hostFunction;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipKernelNameRefByPtr_enter(
      (const void*)(uintptr_t)(__rocm_in_hostFunction),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipKernelNameRefByPtr */
  TRY;
  ROCM_TRACE_RET_PTR_CURATED_NOARGS(hipKernelNameRefByPtr, auto, hip::GetHipDispatchTable()->hipKernelNameRefByPtr_fn(hostFunction, stream));
  CATCHRET(const char*);
}
extern "C" hipError_t hipLaunchByPtr(const void* func) {
  auto const __rocm_in_func = func;
  rocm_trace_emit_hipLaunchByPtr_enter(
      (const void*)(uintptr_t)(__rocm_in_func)); /* __ROCM_CURATED__: hipLaunchByPtr */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchByPtr, hip::GetHipDispatchTable()->hipLaunchByPtr_fn(func));
  CATCH;
}
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream) {
  auto const __rocm_in_f = f;
  auto const __rocm_in_gridDim = gridDim;
  auto const __rocm_in_blockDimX = blockDimX;
  auto const __rocm_in_kernelParams = kernelParams;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipLaunchCooperativeKernel_enter(
      (const void*)(uintptr_t)(__rocm_in_f),
      (__rocm_in_gridDim),
      (__rocm_in_blockDimX),
      (const void*)(uintptr_t)(__rocm_in_kernelParams),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipLaunchCooperativeKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchCooperativeKernel, hip::GetHipDispatchTable()->hipLaunchCooperativeKernel_fn(f, gridDim, blockDimX, kernelParams,
                                                                sharedMemBytes, stream));
  CATCH;
}
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                                 unsigned int flags) {
  auto const __rocm_in_launchParamsList = launchParamsList;
  auto const __rocm_in_numDevices = numDevices;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipLaunchCooperativeKernelMultiDevice_enter(
      (const void*)(uintptr_t)(__rocm_in_launchParamsList),
      (__rocm_in_numDevices),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipLaunchCooperativeKernelMultiDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchCooperativeKernelMultiDevice, hip::GetHipDispatchTable()->hipLaunchCooperativeKernelMultiDevice_fn(launchParamsList,
                                                                           numDevices, flags));
  CATCH;
}
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_fn = fn;
  auto const __rocm_in_userData = userData;
  rocm_trace_emit_hipLaunchHostFunc_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_fn),
      (const void*)(uintptr_t)(__rocm_in_userData)); /* __ROCM_CURATED__: hipLaunchHostFunc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchHostFunc, hip::GetHipDispatchTable()->hipLaunchHostFunc_fn(stream, fn, userData));
  CATCH;
}
extern "C" hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                                      void** args, size_t sharedMemBytes, hipStream_t stream) {
  auto const __rocm_in_function_address = function_address;
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_dimBlocks = dimBlocks;
  auto const __rocm_in_args = args;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipLaunchKernel_enter(
      (const void*)(uintptr_t)(__rocm_in_function_address),
      (__rocm_in_numBlocks),
      (__rocm_in_dimBlocks),
      (const void*)(uintptr_t)(__rocm_in_args),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipLaunchKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchKernel, hip::GetHipDispatchTable()->hipLaunchKernel_fn(function_address, numBlocks, dimBlocks, args,
                                                     sharedMemBytes, stream));
  CATCH;
}
hipError_t hipMalloc(void** ptr, size_t size) {
  auto const __rocm_in_size = size;
  rocm_trace_emit_hipMalloc_enter(
      (__rocm_in_size)); /* __ROCM_CURATED__: hipMalloc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipMalloc, hip::GetHipDispatchTable()->hipMalloc_fn(ptr, size), ptr);
  CATCH;
}
hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  auto const __rocm_in_pitchedDevPtr = pitchedDevPtr;
  rocm_trace_emit_hipMalloc3D_enter(
      (const void*)(uintptr_t)(__rocm_in_pitchedDevPtr)); /* __ROCM_CURATED__: hipMalloc3D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMalloc3D, hip::GetHipDispatchTable()->hipMalloc3D_fn(pitchedDevPtr, extent));
  CATCH;
}
extern "C" hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                                       struct hipExtent extent, unsigned int flags) {
  auto const __rocm_in_array = array;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMalloc3DArray_enter(
      (const void*)(uintptr_t)(__rocm_in_array),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMalloc3DArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMalloc3DArray, hip::GetHipDispatchTable()->hipMalloc3DArray_fn(array, desc, extent, flags));
  CATCH;
}
extern "C" hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc,
                                     size_t width, size_t height, unsigned int flags) {
  auto const __rocm_in_array = array;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMallocArray_enter(
      (const void*)(uintptr_t)(__rocm_in_array),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (__rocm_in_width),
      (__rocm_in_height),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMallocArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMallocArray, hip::GetHipDispatchTable()->hipMallocArray_fn(array, desc, width, height, flags));
  CATCH;
}
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream) {
  auto const __rocm_in_size = size;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMallocAsync_enter(
      (__rocm_in_size),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMallocAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipMallocAsync, hip::GetHipDispatchTable()->hipMallocAsync_fn(dev_ptr, size, stream), dev_ptr);
  CATCH;
}
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_size = size;
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMallocFromPoolAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_size),
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMallocFromPoolAsync */
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMallocFromPoolAsync, hip::GetHipDispatchTable()->hipMallocFromPoolAsync_fn(dev_ptr, size, mem_pool, stream));
}
hipError_t hipMallocHost(void** ptr, size_t size) {
  auto const __rocm_in_size = size;
  rocm_trace_emit_hipMallocHost_enter(
      (__rocm_in_size)); /* __ROCM_CURATED__: hipMallocHost */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipMallocHost, hip::GetHipDispatchTable()->hipMallocHost_fn(ptr, size), ptr);
  CATCH;
}
hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
  auto const __rocm_in_size = size;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMallocManaged_enter(
      (__rocm_in_size),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMallocManaged */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipMallocManaged, hip::GetHipDispatchTable()->hipMallocManaged_fn(dev_ptr, size, flags), dev_ptr);
  CATCH;
}
extern "C" hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,
                                              const struct hipChannelFormatDesc* desc,
                                              struct hipExtent extent, unsigned int numLevels,
                                              unsigned int flags) {
  auto const __rocm_in_mipmappedArray = mipmappedArray;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_numLevels = numLevels;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMallocMipmappedArray_enter(
      (const void*)(uintptr_t)(__rocm_in_mipmappedArray),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (__rocm_in_numLevels),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMallocMipmappedArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMallocMipmappedArray, hip::GetHipDispatchTable()->hipMallocMipmappedArray_fn(
                                    mipmappedArray, desc, extent, numLevels, flags));
  CATCH;
}
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_pitch = pitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  rocm_trace_emit_hipMallocPitch_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (const void*)(uintptr_t)(__rocm_in_pitch),
      (__rocm_in_width),
      (__rocm_in_height)); /* __ROCM_CURATED__: hipMallocPitch */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMallocPitch, hip::GetHipDispatchTable()->hipMallocPitch_fn(ptr, pitch, width, height));
  CATCH;
}
hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  auto const __rocm_in_devPtr = devPtr;
  auto const __rocm_in_size = size;
  rocm_trace_emit_hipMemAddressFree_enter(
      (const void*)(uintptr_t)(__rocm_in_devPtr),
      (__rocm_in_size)); /* __ROCM_CURATED__: hipMemAddressFree */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemAddressFree, hip::GetHipDispatchTable()->hipMemAddressFree_fn(devPtr, size));
  CATCH;
}
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_size = size;
  auto const __rocm_in_alignment = alignment;
  auto const __rocm_in_addr = addr;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMemAddressReserve_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_size),
      (__rocm_in_alignment),
      (const void*)(uintptr_t)(__rocm_in_addr),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMemAddressReserve */
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemAddressReserve, hip::GetHipDispatchTable()->hipMemAddressReserve_fn(ptr, size, alignment, addr, flags));
}
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_count = count;
  auto const __rocm_in_advice = advice;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipMemAdvise_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_count),
      (int32_t)(__rocm_in_advice),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipMemAdvise */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemAdvise, hip::GetHipDispatchTable()->hipMemAdvise_fn(dev_ptr, count, advice, device));
  CATCH;
}
hipError_t hipMemAdvise_v2(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                           hipMemLocation location) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_count = count;
  auto const __rocm_in_advice = advice;
  rocm_trace_emit_hipMemAdvise_v2_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_count),
      (int32_t)(__rocm_in_advice)); /* __ROCM_CURATED__: hipMemAdvise_v2 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemAdvise_v2, hip::GetHipDispatchTable()->hipMemAdvise_v2_fn(dev_ptr, count, advice, location));
  CATCH;
}
hipError_t hipMemAllocHost(void** ptr, size_t size) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_size = size;
  rocm_trace_emit_hipMemAllocHost_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_size)); /* __ROCM_CURATED__: hipMemAllocHost */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemAllocHost, hip::GetHipDispatchTable()->hipMemAllocHost_fn(ptr, size));
  CATCH;
}
hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes) {
  auto const __rocm_in_dptr = dptr;
  auto const __rocm_in_pitch = pitch;
  auto const __rocm_in_widthInBytes = widthInBytes;
  auto const __rocm_in_height = height;
  auto const __rocm_in_elementSizeBytes = elementSizeBytes;
  rocm_trace_emit_hipMemAllocPitch_enter(
      (const void*)(uintptr_t)(__rocm_in_dptr),
      (const void*)(uintptr_t)(__rocm_in_pitch),
      (__rocm_in_widthInBytes),
      (__rocm_in_height),
      (__rocm_in_elementSizeBytes)); /* __ROCM_CURATED__: hipMemAllocPitch */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemAllocPitch, hip::GetHipDispatchTable()->hipMemAllocPitch_fn(
                                    dptr, pitch, widthInBytes, height, elementSizeBytes));
  CATCH;
}
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags) {
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_size = size;
  auto const __rocm_in_prop = prop;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMemCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_handle),
      (__rocm_in_size),
      (const void*)(uintptr_t)(__rocm_in_prop),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMemCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemCreate, hip::GetHipDispatchTable()->hipMemCreate_fn(handle, size, prop, flags));
  CATCH;
}
hipError_t hipMemExportToShareableHandle(void* shareableHandle,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType handleType,
                                         unsigned long long flags) {
  auto const __rocm_in_shareableHandle = shareableHandle;
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_handleType = handleType;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMemExportToShareableHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_shareableHandle),
      (const void*)(uintptr_t)(__rocm_in_handle),
      (int32_t)(__rocm_in_handleType),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMemExportToShareableHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemExportToShareableHandle, hip::GetHipDispatchTable()->hipMemExportToShareableHandle_fn(
                                    shareableHandle, handle, handleType, flags));
  CATCH;
}
hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr) {
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_location = location;
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipMemGetAccess_enter(
      (const void*)(uintptr_t)(__rocm_in_flags),
      (const void*)(uintptr_t)(__rocm_in_location),
      (const void*)(uintptr_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipMemGetAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemGetAccess, hip::GetHipDispatchTable()->hipMemGetAccess_fn(flags, location, ptr));
  CATCH;
}
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  auto const __rocm_in_pbase = pbase;
  auto const __rocm_in_psize = psize;
  auto const __rocm_in_dptr = dptr;
  rocm_trace_emit_hipMemGetAddressRange_enter(
      (const void*)(uintptr_t)(__rocm_in_pbase),
      (const void*)(uintptr_t)(__rocm_in_psize),
      (uint64_t)(__rocm_in_dptr)); /* __ROCM_CURATED__: hipMemGetAddressRange */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemGetAddressRange, hip::GetHipDispatchTable()->hipMemGetAddressRange_fn(pbase, psize, dptr));
  CATCH;
}
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option) {
  auto const __rocm_in_granularity = granularity;
  auto const __rocm_in_prop = prop;
  auto const __rocm_in_option = option;
  rocm_trace_emit_hipMemGetAllocationGranularity_enter(
      (const void*)(uintptr_t)(__rocm_in_granularity),
      (const void*)(uintptr_t)(__rocm_in_prop),
      (int32_t)(__rocm_in_option)); /* __ROCM_CURATED__: hipMemGetAllocationGranularity */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemGetAllocationGranularity, hip::GetHipDispatchTable()->hipMemGetAllocationGranularity_fn(granularity, prop, option));
  CATCH;
}
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                   hipMemGenericAllocationHandle_t handle) {
  auto const __rocm_in_prop = prop;
  auto const __rocm_in_handle = handle;
  rocm_trace_emit_hipMemGetAllocationPropertiesFromHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_prop),
      (const void*)(uintptr_t)(__rocm_in_handle)); /* __ROCM_CURATED__: hipMemGetAllocationPropertiesFromHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemGetAllocationPropertiesFromHandle, hip::GetHipDispatchTable()->hipMemGetAllocationPropertiesFromHandle_fn(prop, handle));
  CATCH;
}
hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  auto const __rocm_in_free = free;
  auto const __rocm_in_total = total;
  rocm_trace_emit_hipMemGetInfo_enter(
      (const void*)(uintptr_t)(__rocm_in_free),
      (const void*)(uintptr_t)(__rocm_in_total)); /* __ROCM_CURATED__: hipMemGetInfo */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemGetInfo, hip::GetHipDispatchTable()->hipMemGetInfo_fn(free, total));
  CATCH;
}
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle,
                                           hipMemAllocationHandleType shHandleType) {
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_osHandle = osHandle;
  auto const __rocm_in_shHandleType = shHandleType;
  rocm_trace_emit_hipMemImportFromShareableHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_handle),
      (const void*)(uintptr_t)(__rocm_in_osHandle),
      (int32_t)(__rocm_in_shHandleType)); /* __ROCM_CURATED__: hipMemImportFromShareableHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemImportFromShareableHandle, hip::GetHipDispatchTable()->hipMemImportFromShareableHandle_fn(
                                    handle, osHandle, shHandleType));
  CATCH;
}
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_size = size;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMemMap_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_size),
      (__rocm_in_offset),
      (const void*)(uintptr_t)(__rocm_in_handle),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMemMap */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemMap, hip::GetHipDispatchTable()->hipMemMap_fn(ptr, size, offset, handle, flags));
  CATCH;
}
hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count,
                               hipStream_t stream) {
  auto const __rocm_in_mapInfoList = mapInfoList;
  auto const __rocm_in_count = count;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemMapArrayAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_mapInfoList),
      (__rocm_in_count),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemMapArrayAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemMapArrayAsync, hip::GetHipDispatchTable()->hipMemMapArrayAsync_fn(mapInfoList, count, stream));
  CATCH;
}
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_pool_props = pool_props;
  rocm_trace_emit_hipMemPoolCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (const void*)(uintptr_t)(__rocm_in_pool_props)); /* __ROCM_CURATED__: hipMemPoolCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolCreate, hip::GetHipDispatchTable()->hipMemPoolCreate_fn(mem_pool, pool_props));
  CATCH;
}
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
  auto const __rocm_in_mem_pool = mem_pool;
  rocm_trace_emit_hipMemPoolDestroy_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool)); /* __ROCM_CURATED__: hipMemPoolDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolDestroy, hip::GetHipDispatchTable()->hipMemPoolDestroy_fn(mem_pool));
  CATCH;
}
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr) {
  auto const __rocm_in_export_data = export_data;
  auto const __rocm_in_dev_ptr = dev_ptr;
  rocm_trace_emit_hipMemPoolExportPointer_enter(
      (const void*)(uintptr_t)(__rocm_in_export_data),
      (const void*)(uintptr_t)(__rocm_in_dev_ptr)); /* __ROCM_CURATED__: hipMemPoolExportPointer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolExportPointer, hip::GetHipDispatchTable()->hipMemPoolExportPointer_fn(export_data, dev_ptr));
  CATCH;
}
hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool,
                                             hipMemAllocationHandleType handle_type,
                                             unsigned int flags) {
  auto const __rocm_in_shared_handle = shared_handle;
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_handle_type = handle_type;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMemPoolExportToShareableHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_shared_handle),
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (int32_t)(__rocm_in_handle_type),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMemPoolExportToShareableHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolExportToShareableHandle, hip::GetHipDispatchTable()->hipMemPoolExportToShareableHandle_fn(
                                    shared_handle, mem_pool, handle_type, flags));
  CATCH;
}
hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                               hipMemLocation* location) {
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_location = location;
  rocm_trace_emit_hipMemPoolGetAccess_enter(
      (const void*)(uintptr_t)(__rocm_in_flags),
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (const void*)(uintptr_t)(__rocm_in_location)); /* __ROCM_CURATED__: hipMemPoolGetAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolGetAccess, hip::GetHipDispatchTable()->hipMemPoolGetAccess_fn(flags, mem_pool, location));
  CATCH;
}
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipMemPoolGetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipMemPoolGetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolGetAttribute, hip::GetHipDispatchTable()->hipMemPoolGetAttribute_fn(mem_pool, attr, value));
  CATCH;
}
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle,
                                               hipMemAllocationHandleType handle_type,
                                               unsigned int flags) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_shared_handle = shared_handle;
  auto const __rocm_in_handle_type = handle_type;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMemPoolImportFromShareableHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (const void*)(uintptr_t)(__rocm_in_shared_handle),
      (int32_t)(__rocm_in_handle_type),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMemPoolImportFromShareableHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolImportFromShareableHandle, hip::GetHipDispatchTable()->hipMemPoolImportFromShareableHandle_fn(
                                    mem_pool, shared_handle, handle_type, flags));
  CATCH;
}
hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool,
                                   hipMemPoolPtrExportData* export_data) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_export_data = export_data;
  rocm_trace_emit_hipMemPoolImportPointer_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (const void*)(uintptr_t)(__rocm_in_export_data)); /* __ROCM_CURATED__: hipMemPoolImportPointer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolImportPointer, hip::GetHipDispatchTable()->hipMemPoolImportPointer_fn(dev_ptr, mem_pool, export_data));
  CATCH;
}
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list,
                               size_t count) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_desc_list = desc_list;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemPoolSetAccess_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (const void*)(uintptr_t)(__rocm_in_desc_list),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemPoolSetAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolSetAccess, hip::GetHipDispatchTable()->hipMemPoolSetAccess_fn(mem_pool, desc_list, count));
  CATCH;
}
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipMemPoolSetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipMemPoolSetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolSetAttribute, hip::GetHipDispatchTable()->hipMemPoolSetAttribute_fn(mem_pool, attr, value));
  CATCH;
}
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  auto const __rocm_in_mem_pool = mem_pool;
  auto const __rocm_in_min_bytes_to_hold = min_bytes_to_hold;
  rocm_trace_emit_hipMemPoolTrimTo_enter(
      (const void*)(uintptr_t)(__rocm_in_mem_pool),
      (__rocm_in_min_bytes_to_hold)); /* __ROCM_CURATED__: hipMemPoolTrimTo */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPoolTrimTo, hip::GetHipDispatchTable()->hipMemPoolTrimTo_fn(mem_pool, min_bytes_to_hold));
  CATCH;
}
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_count = count;
  auto const __rocm_in_device = device;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemPrefetchAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_count),
      (__rocm_in_device),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemPrefetchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPrefetchAsync, hip::GetHipDispatchTable()->hipMemPrefetchAsync_fn(dev_ptr, count, device, stream));
  CATCH;
}
hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count, hipMemLocation location,
                                  unsigned int flags, hipStream_t stream) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_count = count;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemPrefetchAsync_v2_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_count),
      (__rocm_in_flags),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemPrefetchAsync_v2 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPrefetchAsync_v2, hip::GetHipDispatchTable()->hipMemPrefetchAsync_v2_fn(
                                    dev_ptr, count, location, flags, stream));
  CATCH;
}
hipError_t hipMemPrefetchBatchAsync(void** dev_ptrs, size_t* sizes, size_t count,
                                    hipMemLocation* prefetch_locs, size_t* prefetch_loc_idxs,
                                    size_t num_prefetch_locs, unsigned long long flags,
                                    hipStream_t stream) {
  auto const __rocm_in_dev_ptrs = dev_ptrs;
  auto const __rocm_in_sizes = sizes;
  auto const __rocm_in_count = count;
  auto const __rocm_in_prefetch_locs = prefetch_locs;
  auto const __rocm_in_prefetch_loc_idxs = prefetch_loc_idxs;
  auto const __rocm_in_num_prefetch_locs = num_prefetch_locs;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemPrefetchBatchAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptrs),
      (const void*)(uintptr_t)(__rocm_in_sizes),
      (__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_prefetch_locs),
      (const void*)(uintptr_t)(__rocm_in_prefetch_loc_idxs),
      (__rocm_in_num_prefetch_locs),
      (__rocm_in_flags),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemPrefetchBatchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPrefetchBatchAsync, hip::GetHipDispatchTable()->hipMemPrefetchBatchAsync_fn(dev_ptrs, sizes, count, prefetch_locs,
                                                              prefetch_loc_idxs, num_prefetch_locs,
                                                              flags, stream));
  CATCH;
}
hipError_t hipMemDiscardBatchAsync(void** dev_ptrs, size_t* sizes, size_t count,
                                   unsigned long long flags, hipStream_t stream) {
  auto const __rocm_in_dev_ptrs = dev_ptrs;
  auto const __rocm_in_sizes = sizes;
  auto const __rocm_in_count = count;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemDiscardBatchAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptrs),
      (const void*)(uintptr_t)(__rocm_in_sizes),
      (__rocm_in_count),
      (__rocm_in_flags),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemDiscardBatchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemDiscardBatchAsync, hip::GetHipDispatchTable()->hipMemDiscardBatchAsync_fn(dev_ptrs, sizes, count, flags, stream));
  CATCH;
}
hipError_t hipDrvMemDiscardBatchAsync(hipDeviceptr_t* dptrs, size_t* sizes, size_t count,
                                      unsigned long long flags, hipStream_t stream) {
  auto const __rocm_in_dptrs = dptrs;
  auto const __rocm_in_sizes = sizes;
  auto const __rocm_in_count = count;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipDrvMemDiscardBatchAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dptrs),
      (const void*)(uintptr_t)(__rocm_in_sizes),
      (__rocm_in_count),
      (__rocm_in_flags),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipDrvMemDiscardBatchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvMemDiscardBatchAsync, hip::GetHipDispatchTable()->hipDrvMemDiscardBatchAsync_fn(dptrs, sizes, count, flags, stream));
  CATCH;
}
hipError_t hipMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count,
                                              hipMemLocation* prefetchLocs, size_t* prefetchLocIdxs,
                                              size_t numPrefetchLocs, unsigned long long flags,
                                              hipStream_t stream) {
  auto const __rocm_in_dptrs = dptrs;
  auto const __rocm_in_sizes = sizes;
  auto const __rocm_in_count = count;
  auto const __rocm_in_prefetchLocs = prefetchLocs;
  auto const __rocm_in_prefetchLocIdxs = prefetchLocIdxs;
  auto const __rocm_in_numPrefetchLocs = numPrefetchLocs;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemDiscardAndPrefetchBatchAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dptrs),
      (const void*)(uintptr_t)(__rocm_in_sizes),
      (__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_prefetchLocs),
      (const void*)(uintptr_t)(__rocm_in_prefetchLocIdxs),
      (__rocm_in_numPrefetchLocs),
      (__rocm_in_flags),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemDiscardAndPrefetchBatchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemDiscardAndPrefetchBatchAsync, hip::GetHipDispatchTable()->hipMemDiscardAndPrefetchBatchAsync_fn(
          dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream));
  CATCH;
}
hipError_t hipDrvMemDiscardAndPrefetchBatchAsync(hipDeviceptr_t* dptrs, size_t* sizes, size_t count,
                                                 hipMemLocation* prefetchLocs,
                                                 size_t* prefetchLocIdxs, size_t numPrefetchLocs,
                                                 unsigned long long flags, hipStream_t stream) {
  auto const __rocm_in_dptrs = dptrs;
  auto const __rocm_in_sizes = sizes;
  auto const __rocm_in_count = count;
  auto const __rocm_in_prefetchLocs = prefetchLocs;
  auto const __rocm_in_prefetchLocIdxs = prefetchLocIdxs;
  auto const __rocm_in_numPrefetchLocs = numPrefetchLocs;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipDrvMemDiscardAndPrefetchBatchAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dptrs),
      (const void*)(uintptr_t)(__rocm_in_sizes),
      (__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_prefetchLocs),
      (const void*)(uintptr_t)(__rocm_in_prefetchLocIdxs),
      (__rocm_in_numPrefetchLocs),
      (__rocm_in_flags),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipDrvMemDiscardAndPrefetchBatchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvMemDiscardAndPrefetchBatchAsync, hip::GetHipDispatchTable()->hipDrvMemDiscardAndPrefetchBatchAsync_fn(
          dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream));
  CATCH;
}
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_size = size;
  rocm_trace_emit_hipMemPtrGetInfo_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (const void*)(uintptr_t)(__rocm_in_size)); /* __ROCM_CURATED__: hipMemPtrGetInfo */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemPtrGetInfo, hip::GetHipDispatchTable()->hipMemPtrGetInfo_fn(ptr, size));
  CATCH;
}
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count) {
  auto const __rocm_in_data = data;
  auto const __rocm_in_data_size = data_size;
  auto const __rocm_in_attribute = attribute;
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemRangeGetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_data),
      (__rocm_in_data_size),
      (int32_t)(__rocm_in_attribute),
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemRangeGetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemRangeGetAttribute, hip::GetHipDispatchTable()->hipMemRangeGetAttribute_fn(
                                    data, data_size, attribute, dev_ptr, count));
  CATCH;
}
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                    hipMemRangeAttribute* attributes, size_t num_attributes,
                                    const void* dev_ptr, size_t count) {
  auto const __rocm_in_data = data;
  auto const __rocm_in_data_sizes = data_sizes;
  auto const __rocm_in_attributes = attributes;
  auto const __rocm_in_num_attributes = num_attributes;
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemRangeGetAttributes_enter(
      (const void*)(uintptr_t)(__rocm_in_data),
      (const void*)(uintptr_t)(__rocm_in_data_sizes),
      (const void*)(uintptr_t)(__rocm_in_attributes),
      (__rocm_in_num_attributes),
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemRangeGetAttributes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemRangeGetAttributes, hip::GetHipDispatchTable()->hipMemRangeGetAttributes_fn(data, data_sizes, attributes,
                                                              num_attributes, dev_ptr, count));
  CATCH;
}
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
  auto const __rocm_in_handle = handle;
  rocm_trace_emit_hipMemRelease_enter(
      (const void*)(uintptr_t)(__rocm_in_handle)); /* __ROCM_CURATED__: hipMemRelease */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemRelease, hip::GetHipDispatchTable()->hipMemRelease_fn(handle));
  CATCH;
}
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr) {
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_addr = addr;
  rocm_trace_emit_hipMemRetainAllocationHandle_enter(
      (const void*)(uintptr_t)(__rocm_in_handle),
      (const void*)(uintptr_t)(__rocm_in_addr)); /* __ROCM_CURATED__: hipMemRetainAllocationHandle */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemRetainAllocationHandle, hip::GetHipDispatchTable()->hipMemRetainAllocationHandle_fn(handle, addr));
  CATCH;
}
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_size = size;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemSetAccess_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_size),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemSetAccess */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemSetAccess, hip::GetHipDispatchTable()->hipMemSetAccess_fn(ptr, size, desc, count));
  CATCH;
}
hipError_t hipMemUnmap(void* ptr, size_t size) {
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_size = size;
  rocm_trace_emit_hipMemUnmap_enter(
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_size)); /* __ROCM_CURATED__: hipMemUnmap */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemUnmap, hip::GetHipDispatchTable()->hipMemUnmap_fn(ptr, size));
  CATCH;
}
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy, hip::GetHipDispatchTable()->hipMemcpy_fn(dst, src, sizeBytes, kind));
  CATCH;
}
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy2D_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy2D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2D, hip::GetHipDispatchTable()->hipMemcpy2D_fn(dst, dpitch, src, spitch, width, height, kind));
  CATCH;
}
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy2DAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy2DAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DAsync, hip::GetHipDispatchTable()->hipMemcpy2DAsync_fn(
                                    dst, dpitch, src, spitch, width, height, kind, stream));
  CATCH;
}
hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy2DFromArray_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy2DFromArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DFromArray, hip::GetHipDispatchTable()->hipMemcpy2DFromArray_fn(
                                    dst, dpitch, src, wOffset, hOffset, width, height, kind));
  CATCH;
}
hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                     size_t hOffset, size_t width, size_t height,
                                     hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy2DFromArrayAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy2DFromArrayAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DFromArrayAsync, hip::GetHipDispatchTable()->hipMemcpy2DFromArrayAsync_fn(dst, dpitch, src, wOffset, hOffset,
                                                               width, height, kind, stream));
  CATCH;
}
hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy2DToArray_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy2DToArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DToArray, hip::GetHipDispatchTable()->hipMemcpy2DToArray_fn(
                                    dst, wOffset, hOffset, src, spitch, width, height, kind));
  CATCH;
}
hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy2DToArrayAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy2DToArrayAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DToArrayAsync, hip::GetHipDispatchTable()->hipMemcpy2DToArrayAsync_fn(dst, wOffset, hOffset, src, spitch,
                                                             width, height, kind, stream));
  CATCH;
}
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
  auto const __rocm_in_p = p;
  rocm_trace_emit_hipMemcpy3D_enter(
      (const void*)(uintptr_t)(__rocm_in_p)); /* __ROCM_CURATED__: hipMemcpy3D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy3D, hip::GetHipDispatchTable()->hipMemcpy3D_fn(p));
  CATCH;
}
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  auto const __rocm_in_p = p;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy3DAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_p),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy3DAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy3DAsync, hip::GetHipDispatchTable()->hipMemcpy3DAsync_fn(p, stream));
  CATCH;
}
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyAsync, hip::GetHipDispatchTable()->hipMemcpyAsync_fn(dst, src, sizeBytes, kind, stream));
  CATCH;
}
hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray, size_t srcOffset, size_t count) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_srcArray = srcArray;
  auto const __rocm_in_srcOffset = srcOffset;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemcpyAtoH_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_srcArray),
      (__rocm_in_srcOffset),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemcpyAtoH */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyAtoH, hip::GetHipDispatchTable()->hipMemcpyAtoH_fn(dst, srcArray, srcOffset, count));
  CATCH;
}
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  rocm_trace_emit_hipMemcpyDtoD_enter(
      (uint64_t)(__rocm_in_dst),
      (uint64_t)(__rocm_in_src),
      (__rocm_in_sizeBytes)); /* __ROCM_CURATED__: hipMemcpyDtoD */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyDtoD, hip::GetHipDispatchTable()->hipMemcpyDtoD_fn(dst, src, sizeBytes));
  CATCH;
}
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyDtoDAsync_enter(
      (uint64_t)(__rocm_in_dst),
      (uint64_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyDtoDAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyDtoDAsync, hip::GetHipDispatchTable()->hipMemcpyDtoDAsync_fn(dst, src, sizeBytes, stream));
  CATCH;
}
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  rocm_trace_emit_hipMemcpyDtoH_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (uint64_t)(__rocm_in_src),
      (__rocm_in_sizeBytes)); /* __ROCM_CURATED__: hipMemcpyDtoH */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyDtoH, hip::GetHipDispatchTable()->hipMemcpyDtoH_fn(dst, src, sizeBytes));
  CATCH;
}
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyDtoHAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (uint64_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyDtoHAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyDtoHAsync, hip::GetHipDispatchTable()->hipMemcpyDtoHAsync_fn(dst, src, sizeBytes, stream));
  CATCH;
}
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_srcArray = srcArray;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_count = count;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpyFromArray_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_srcArray),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (__rocm_in_count),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpyFromArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyFromArray, hip::GetHipDispatchTable()->hipMemcpyFromArray_fn(
                                    dst, srcArray, wOffset, hOffset, count, kind));
  CATCH;
}
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                               hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpyFromSymbol_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpyFromSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyFromSymbol, hip::GetHipDispatchTable()->hipMemcpyFromSymbol_fn(dst, symbol, sizeBytes, offset, kind));
  CATCH;
}
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyFromSymbolAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyFromSymbolAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyFromSymbolAsync, hip::GetHipDispatchTable()->hipMemcpyFromSymbolAsync_fn(
                                    dst, symbol, sizeBytes, offset, kind, stream));
  CATCH;
}
hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost, size_t count) {
  auto const __rocm_in_dstArray = dstArray;
  auto const __rocm_in_dstOffset = dstOffset;
  auto const __rocm_in_srcHost = srcHost;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemcpyHtoA_enter(
      (const void*)(uintptr_t)(__rocm_in_dstArray),
      (__rocm_in_dstOffset),
      (const void*)(uintptr_t)(__rocm_in_srcHost),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemcpyHtoA */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyHtoA, hip::GetHipDispatchTable()->hipMemcpyHtoA_fn(dstArray, dstOffset, srcHost, count));
  CATCH;
}
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  rocm_trace_emit_hipMemcpyHtoD_enter(
      (uint64_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes)); /* __ROCM_CURATED__: hipMemcpyHtoD */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyHtoD, hip::GetHipDispatchTable()->hipMemcpyHtoD_fn(dst, src, sizeBytes));
  CATCH;
}
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void* src, size_t sizeBytes,
                              hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyHtoDAsync_enter(
      (uint64_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyHtoDAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyHtoDAsync, hip::GetHipDispatchTable()->hipMemcpyHtoDAsync_fn(dst, src, sizeBytes, stream));
  CATCH;
}
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  auto const __rocm_in_pCopy = pCopy;
  rocm_trace_emit_hipMemcpyParam2D_enter(
      (const void*)(uintptr_t)(__rocm_in_pCopy)); /* __ROCM_CURATED__: hipMemcpyParam2D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyParam2D, hip::GetHipDispatchTable()->hipMemcpyParam2D_fn(pCopy));
  CATCH;
}
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream) {
  auto const __rocm_in_pCopy = pCopy;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyParam2DAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_pCopy),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyParam2DAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyParam2DAsync, hip::GetHipDispatchTable()->hipMemcpyParam2DAsync_fn(pCopy, stream));
  CATCH;
}
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                         size_t sizeBytes) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstDeviceId = dstDeviceId;
  auto const __rocm_in_src = src;
  auto const __rocm_in_srcDeviceId = srcDeviceId;
  auto const __rocm_in_sizeBytes = sizeBytes;
  rocm_trace_emit_hipMemcpyPeer_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dstDeviceId),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_srcDeviceId),
      (__rocm_in_sizeBytes)); /* __ROCM_CURATED__: hipMemcpyPeer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyPeer, hip::GetHipDispatchTable()->hipMemcpyPeer_fn(dst, dstDeviceId, src, srcDeviceId, sizeBytes));
  CATCH;
}
hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstDeviceId = dstDeviceId;
  auto const __rocm_in_src = src;
  auto const __rocm_in_srcDevice = srcDevice;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyPeerAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dstDeviceId),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_srcDevice),
      (__rocm_in_sizeBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyPeerAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyPeerAsync, hip::GetHipDispatchTable()->hipMemcpyPeerAsync_fn(
                                    dst, dstDeviceId, src, srcDevice, sizeBytes, stream));
  CATCH;
}
hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_src = src;
  auto const __rocm_in_count = count;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpyToArray_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_count),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpyToArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyToArray, hip::GetHipDispatchTable()->hipMemcpyToArray_fn(dst, wOffset, hOffset, src, count, kind));
  CATCH;
}
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind) {
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpyToSymbol_enter(
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpyToSymbol */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyToSymbol, hip::GetHipDispatchTable()->hipMemcpyToSymbol_fn(symbol, src, sizeBytes, offset, kind));
  CATCH;
}
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyToSymbolAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyToSymbolAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyToSymbolAsync, hip::GetHipDispatchTable()->hipMemcpyToSymbolAsync_fn(
                                    symbol, src, sizeBytes, offset, kind, stream));
  CATCH;
}
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyWithStream_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyWithStream */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyWithStream, hip::GetHipDispatchTable()->hipMemcpyWithStream_fn(dst, src, sizeBytes, kind, stream));
  CATCH;
}
hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_value = value;
  auto const __rocm_in_sizeBytes = sizeBytes;
  rocm_trace_emit_hipMemset_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_value),
      (__rocm_in_sizeBytes)); /* __ROCM_CURATED__: hipMemset */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset, hip::GetHipDispatchTable()->hipMemset_fn(dst, value, sizeBytes));
  CATCH;
}
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_pitch = pitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  rocm_trace_emit_hipMemset2D_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_pitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height)); /* __ROCM_CURATED__: hipMemset2D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset2D, hip::GetHipDispatchTable()->hipMemset2D_fn(dst, pitch, value, width, height));
  CATCH;
}
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,
                            hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_pitch = pitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemset2DAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_pitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemset2DAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset2DAsync, hip::GetHipDispatchTable()->hipMemset2DAsync_fn(dst, pitch, value, width, height, stream));
  CATCH;
}
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipMemset3D_enter(
      (__rocm_in_value)); /* __ROCM_CURATED__: hipMemset3D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset3D, hip::GetHipDispatchTable()->hipMemset3D_fn(pitchedDevPtr, value, extent));
  CATCH;
}
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                            hipStream_t stream) {
  auto const __rocm_in_value = value;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemset3DAsync_enter(
      (__rocm_in_value),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemset3DAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset3DAsync, hip::GetHipDispatchTable()->hipMemset3DAsync_fn(pitchedDevPtr, value, extent, stream));
  CATCH;
}
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_value = value;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_value),
      (__rocm_in_sizeBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetAsync, hip::GetHipDispatchTable()->hipMemsetAsync_fn(dst, value, sizeBytes, stream));
  CATCH;
}
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count) {
  auto const __rocm_in_dest = dest;
  auto const __rocm_in_value = value;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemsetD16_enter(
      (uint64_t)(__rocm_in_dest),
      (__rocm_in_value),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemsetD16 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD16, hip::GetHipDispatchTable()->hipMemsetD16_fn(dest, value, count));
  CATCH;
}
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count,
                             hipStream_t stream) {
  auto const __rocm_in_dest = dest;
  auto const __rocm_in_value = value;
  auto const __rocm_in_count = count;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetD16Async_enter(
      (uint64_t)(__rocm_in_dest),
      (__rocm_in_value),
      (__rocm_in_count),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetD16Async */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD16Async, hip::GetHipDispatchTable()->hipMemsetD16Async_fn(dest, value, count, stream));
  CATCH;
}
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
  auto const __rocm_in_dest = dest;
  auto const __rocm_in_value = value;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemsetD32_enter(
      (uint64_t)(__rocm_in_dest),
      (__rocm_in_value),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemsetD32 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD32, hip::GetHipDispatchTable()->hipMemsetD32_fn(dest, value, count));
  CATCH;
}
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_value = value;
  auto const __rocm_in_count = count;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetD32Async_enter(
      (uint64_t)(__rocm_in_dst),
      (__rocm_in_value),
      (__rocm_in_count),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetD32Async */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD32Async, hip::GetHipDispatchTable()->hipMemsetD32Async_fn(dst, value, count, stream));
  CATCH;
}
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
  auto const __rocm_in_dest = dest;
  auto const __rocm_in_value = value;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipMemsetD8_enter(
      (uint64_t)(__rocm_in_dest),
      (__rocm_in_value),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipMemsetD8 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD8, hip::GetHipDispatchTable()->hipMemsetD8_fn(dest, value, count));
  CATCH;
}
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count,
                            hipStream_t stream) {
  auto const __rocm_in_dest = dest;
  auto const __rocm_in_value = value;
  auto const __rocm_in_count = count;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetD8Async_enter(
      (uint64_t)(__rocm_in_dest),
      (__rocm_in_value),
      (__rocm_in_count),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetD8Async */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD8Async, hip::GetHipDispatchTable()->hipMemsetD8Async_fn(dest, value, count, stream));
  CATCH;
}
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels) {
  auto const __rocm_in_pHandle = pHandle;
  auto const __rocm_in_pMipmappedArrayDesc = pMipmappedArrayDesc;
  auto const __rocm_in_numMipmapLevels = numMipmapLevels;
  rocm_trace_emit_hipMipmappedArrayCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_pHandle),
      (const void*)(uintptr_t)(__rocm_in_pMipmappedArrayDesc),
      (__rocm_in_numMipmapLevels)); /* __ROCM_CURATED__: hipMipmappedArrayCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMipmappedArrayCreate, hip::GetHipDispatchTable()->hipMipmappedArrayCreate_fn(
                                    pHandle, pMipmappedArrayDesc, numMipmapLevels));
  CATCH;
}
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
  auto const __rocm_in_hMipmappedArray = hMipmappedArray;
  rocm_trace_emit_hipMipmappedArrayDestroy_enter(
      (const void*)(uintptr_t)(__rocm_in_hMipmappedArray)); /* __ROCM_CURATED__: hipMipmappedArrayDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMipmappedArrayDestroy, hip::GetHipDispatchTable()->hipMipmappedArrayDestroy_fn(hMipmappedArray));
  CATCH;
}
hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level) {
  auto const __rocm_in_pLevelArray = pLevelArray;
  auto const __rocm_in_hMipMappedArray = hMipMappedArray;
  auto const __rocm_in_level = level;
  rocm_trace_emit_hipMipmappedArrayGetLevel_enter(
      (const void*)(uintptr_t)(__rocm_in_pLevelArray),
      (const void*)(uintptr_t)(__rocm_in_hMipMappedArray),
      (__rocm_in_level)); /* __ROCM_CURATED__: hipMipmappedArrayGetLevel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMipmappedArrayGetLevel, hip::GetHipDispatchTable()->hipMipmappedArrayGetLevel_fn(pLevelArray, hMipMappedArray, level));
  CATCH;
}
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
  auto const __rocm_in_module = module;
  auto const __rocm_in_kname = kname;
  rocm_trace_emit_hipModuleGetFunction_enter(
      (uint64_t)(uintptr_t)(__rocm_in_module),
      (const char*)(__rocm_in_kname)); /* __ROCM_CURATED__: hipModuleGetFunction */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipModuleGetFunction, hip::GetHipDispatchTable()->hipModuleGetFunction_fn(function, module, kname), function);
  CATCH;
}
hipError_t hipModuleGetFunctionCount(unsigned int* count, hipModule_t mod) {
  auto const __rocm_in_count = count;
  auto const __rocm_in_mod = mod;
  rocm_trace_emit_hipModuleGetFunctionCount_enter(
      (const void*)(uintptr_t)(__rocm_in_count),
      (uint64_t)(uintptr_t)(__rocm_in_mod)); /* __ROCM_CURATED__: hipModuleGetFunctionCount */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleGetFunctionCount, hip::GetHipDispatchTable()->hipModuleGetFunctionCount_fn(count, mod));
  CATCH;
}
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name) {
  auto const __rocm_in_dptr = dptr;
  auto const __rocm_in_bytes = bytes;
  auto const __rocm_in_hmod = hmod;
  auto const __rocm_in_name = name;
  rocm_trace_emit_hipModuleGetGlobal_enter(
      (const void*)(uintptr_t)(__rocm_in_dptr),
      (const void*)(uintptr_t)(__rocm_in_bytes),
      (uint64_t)(uintptr_t)(__rocm_in_hmod),
      (const char*)(__rocm_in_name)); /* __ROCM_CURATED__: hipModuleGetGlobal */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleGetGlobal, hip::GetHipDispatchTable()->hipModuleGetGlobal_fn(dptr, bytes, hmod, name));
  CATCH;
}
hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_hmod = hmod;
  auto const __rocm_in_name = name;
  rocm_trace_emit_hipModuleGetTexRef_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (uint64_t)(uintptr_t)(__rocm_in_hmod),
      (const char*)(__rocm_in_name)); /* __ROCM_CURATED__: hipModuleGetTexRef */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleGetTexRef, hip::GetHipDispatchTable()->hipModuleGetTexRef_fn(texRef, hmod, name));
  CATCH;
}
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams) {
  auto const __rocm_in_f = f;
  auto const __rocm_in_gridDimX = gridDimX;
  auto const __rocm_in_gridDimY = gridDimY;
  auto const __rocm_in_gridDimZ = gridDimZ;
  auto const __rocm_in_blockDimX = blockDimX;
  auto const __rocm_in_blockDimY = blockDimY;
  auto const __rocm_in_blockDimZ = blockDimZ;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_kernelParams = kernelParams;
  rocm_trace_emit_hipModuleLaunchCooperativeKernel_enter(
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_gridDimX),
      (__rocm_in_gridDimY),
      (__rocm_in_gridDimZ),
      (__rocm_in_blockDimX),
      (__rocm_in_blockDimY),
      (__rocm_in_blockDimZ),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_kernelParams)); /* __ROCM_CURATED__: hipModuleLaunchCooperativeKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleLaunchCooperativeKernel, hip::GetHipDispatchTable()->hipModuleLaunchCooperativeKernel_fn(
                                    f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                    blockDimZ, sharedMemBytes, stream, kernelParams));
  CATCH;
}
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices,
                                                       unsigned int flags) {
  auto const __rocm_in_launchParamsList = launchParamsList;
  auto const __rocm_in_numDevices = numDevices;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipModuleLaunchCooperativeKernelMultiDevice_enter(
      (const void*)(uintptr_t)(__rocm_in_launchParamsList),
      (__rocm_in_numDevices),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipModuleLaunchCooperativeKernelMultiDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleLaunchCooperativeKernelMultiDevice, hip::GetHipDispatchTable()->hipModuleLaunchCooperativeKernelMultiDevice_fn(launchParamsList,
                                                                                 numDevices, flags));
  CATCH;
}
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra) {
  auto const __rocm_in_f = f;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_kernelParams = kernelParams;
  auto const __rocm_in_extra = extra;
  rocm_trace_emit_hipModuleLaunchKernel_enter(
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_kernelParams),
      (const void*)(uintptr_t)(__rocm_in_extra)); /* __ROCM_CURATED__: hipModuleLaunchKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleLaunchKernel, hip::GetHipDispatchTable()->hipModuleLaunchKernel_fn(
                                    f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                    blockDimZ, sharedMemBytes, stream, kernelParams, extra));
  CATCH;
}
hipError_t hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin) {
  auto const __rocm_in_module = module;
  auto const __rocm_in_fatbin = fatbin;
  rocm_trace_emit_hipModuleLoadFatBinary_enter(
      (const void*)(uintptr_t)(__rocm_in_module),
      (const void*)(uintptr_t)(__rocm_in_fatbin)); /* __ROCM_CURATED__: hipModuleLoadFatBinary */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleLoadFatBinary, hip::GetHipDispatchTable()->hipModuleLoadFatBinary_fn(module, fatbin));
  CATCH;
}
hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  auto const __rocm_in_module = module;
  auto const __rocm_in_fname = fname;
  rocm_trace_emit_hipModuleLoad_enter(
      (const void*)(uintptr_t)(__rocm_in_module),
      (const char*)(__rocm_in_fname)); /* __ROCM_CURATED__: hipModuleLoad */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleLoad, hip::GetHipDispatchTable()->hipModuleLoad_fn(module, fname));
  CATCH;
}
hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  auto const __rocm_in_image = image;
  rocm_trace_emit_hipModuleLoadData_enter(
      (const void*)(uintptr_t)(__rocm_in_image)); /* __ROCM_CURATED__: hipModuleLoadData */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipModuleLoadData, hip::GetHipDispatchTable()->hipModuleLoadData_fn(module, image), module);
  CATCH;
}
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues) {
  auto const __rocm_in_image = image;
  auto const __rocm_in_numOptions = numOptions;
  auto const __rocm_in_options = options;
  auto const __rocm_in_optionValues = optionValues;
  rocm_trace_emit_hipModuleLoadDataEx_enter(
      (const void*)(uintptr_t)(__rocm_in_image),
      (__rocm_in_numOptions),
      (const void*)(uintptr_t)(__rocm_in_options),
      (const void*)(uintptr_t)(__rocm_in_optionValues)); /* __ROCM_CURATED__: hipModuleLoadDataEx */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipModuleLoadDataEx, hip::GetHipDispatchTable()->hipModuleLoadDataEx_fn(
                                    module, image, numOptions, options, optionValues), module);
  CATCH;
}

hipError_t hipLinkAddData(hipLinkState_t state, hipJitInputType type, void* data, size_t size,
                          const char* name, unsigned int numOptions, hipJitOption* options,
                          void** optionValues) {
  auto const __rocm_in_state = state;
  auto const __rocm_in_type = type;
  auto const __rocm_in_data = data;
  auto const __rocm_in_size = size;
  auto const __rocm_in_name = name;
  auto const __rocm_in_numOptions = numOptions;
  auto const __rocm_in_options = options;
  auto const __rocm_in_optionValues = optionValues;
  rocm_trace_emit_hipLinkAddData_enter(
      (const void*)(uintptr_t)(__rocm_in_state),
      (int32_t)(__rocm_in_type),
      (const void*)(uintptr_t)(__rocm_in_data),
      (__rocm_in_size),
      (const char*)(__rocm_in_name),
      (__rocm_in_numOptions),
      (const void*)(uintptr_t)(__rocm_in_options),
      (const void*)(uintptr_t)(__rocm_in_optionValues)); /* __ROCM_CURATED__: hipLinkAddData */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLinkAddData, hip::GetHipDispatchTable()->hipLinkAddData_fn(state, type, data, size, name, numOptions,
                                                    options, optionValues));
  CATCH;
}

hipError_t hipLinkAddFile(hipLinkState_t state, hipJitInputType type, const char* path,
                          unsigned int numOptions, hipJitOption* options, void** optionValues) {
  auto const __rocm_in_state = state;
  auto const __rocm_in_type = type;
  auto const __rocm_in_path = path;
  auto const __rocm_in_numOptions = numOptions;
  auto const __rocm_in_options = options;
  auto const __rocm_in_optionValues = optionValues;
  rocm_trace_emit_hipLinkAddFile_enter(
      (const void*)(uintptr_t)(__rocm_in_state),
      (int32_t)(__rocm_in_type),
      (const char*)(__rocm_in_path),
      (__rocm_in_numOptions),
      (const void*)(uintptr_t)(__rocm_in_options),
      (const void*)(uintptr_t)(__rocm_in_optionValues)); /* __ROCM_CURATED__: hipLinkAddFile */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLinkAddFile, hip::GetHipDispatchTable()->hipLinkAddFile_fn(
                                    state, type, path, numOptions, options, optionValues));
  CATCH;
}

hipError_t hipLinkComplete(hipLinkState_t state, void** hipBinOut, size_t* sizeOut) {
  auto const __rocm_in_state = state;
  auto const __rocm_in_hipBinOut = hipBinOut;
  auto const __rocm_in_sizeOut = sizeOut;
  rocm_trace_emit_hipLinkComplete_enter(
      (const void*)(uintptr_t)(__rocm_in_state),
      (const void*)(uintptr_t)(__rocm_in_hipBinOut),
      (const void*)(uintptr_t)(__rocm_in_sizeOut)); /* __ROCM_CURATED__: hipLinkComplete */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLinkComplete, hip::GetHipDispatchTable()->hipLinkComplete_fn(state, hipBinOut, sizeOut));
  CATCH;
}

hipError_t hipLinkCreate(unsigned int numOptions, hipJitOption* options, void** optionValues,
                         hipLinkState_t* stateOut) {
  auto const __rocm_in_numOptions = numOptions;
  auto const __rocm_in_options = options;
  auto const __rocm_in_optionValues = optionValues;
  auto const __rocm_in_stateOut = stateOut;
  rocm_trace_emit_hipLinkCreate_enter(
      (__rocm_in_numOptions),
      (const void*)(uintptr_t)(__rocm_in_options),
      (const void*)(uintptr_t)(__rocm_in_optionValues),
      (const void*)(uintptr_t)(__rocm_in_stateOut)); /* __ROCM_CURATED__: hipLinkCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLinkCreate, hip::GetHipDispatchTable()->hipLinkCreate_fn(numOptions, options, optionValues, stateOut));
  CATCH;
}

hipError_t hipLinkDestroy(hipLinkState_t state) {
  auto const __rocm_in_state = state;
  rocm_trace_emit_hipLinkDestroy_enter(
      (const void*)(uintptr_t)(__rocm_in_state)); /* __ROCM_CURATED__: hipLinkDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLinkDestroy, hip::GetHipDispatchTable()->hipLinkDestroy_fn(state));
  CATCH;
}

extern "C" hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_f = f;
  auto const __rocm_in_blockSize = blockSize;
  auto const __rocm_in_dynSharedMemPerBlk = dynSharedMemPerBlk;
  rocm_trace_emit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_enter(
      (const void*)(uintptr_t)(__rocm_in_numBlocks),
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_blockSize),
      (__rocm_in_dynSharedMemPerBlk)); /* __ROCM_CURATED__: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, hip::GetHipDispatchTable()->hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn(
          numBlocks, f, blockSize, dynSharedMemPerBlk));
  CATCH;
}
extern "C" hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_f = f;
  auto const __rocm_in_blockSize = blockSize;
  auto const __rocm_in_dynSharedMemPerBlk = dynSharedMemPerBlk;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_numBlocks),
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_blockSize),
      (__rocm_in_dynSharedMemPerBlk),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, hip::GetHipDispatchTable()->hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn(
          numBlocks, f, blockSize, dynSharedMemPerBlk, flags));
  CATCH;
}
extern "C" hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                              hipFunction_t f,
                                                              size_t dynSharedMemPerBlk,
                                                              int blockSizeLimit) {
  auto const __rocm_in_gridSize = gridSize;
  auto const __rocm_in_blockSize = blockSize;
  auto const __rocm_in_f = f;
  auto const __rocm_in_dynSharedMemPerBlk = dynSharedMemPerBlk;
  auto const __rocm_in_blockSizeLimit = blockSizeLimit;
  rocm_trace_emit_hipModuleOccupancyMaxPotentialBlockSize_enter(
      (const void*)(uintptr_t)(__rocm_in_gridSize),
      (const void*)(uintptr_t)(__rocm_in_blockSize),
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_dynSharedMemPerBlk),
      (__rocm_in_blockSizeLimit)); /* __ROCM_CURATED__: hipModuleOccupancyMaxPotentialBlockSize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleOccupancyMaxPotentialBlockSize, hip::GetHipDispatchTable()->hipModuleOccupancyMaxPotentialBlockSize_fn(
          gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit));
  CATCH;
}
extern "C" hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit,
    unsigned int flags) {
  auto const __rocm_in_gridSize = gridSize;
  auto const __rocm_in_blockSize = blockSize;
  auto const __rocm_in_f = f;
  auto const __rocm_in_dynSharedMemPerBlk = dynSharedMemPerBlk;
  auto const __rocm_in_blockSizeLimit = blockSizeLimit;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_gridSize),
      (const void*)(uintptr_t)(__rocm_in_blockSize),
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (__rocm_in_dynSharedMemPerBlk),
      (__rocm_in_blockSizeLimit),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipModuleOccupancyMaxPotentialBlockSizeWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleOccupancyMaxPotentialBlockSizeWithFlags, hip::GetHipDispatchTable()->hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn(
          gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags));
  CATCH;
}
hipError_t hipModuleUnload(hipModule_t module) {
  auto const __rocm_in_module = module;
  rocm_trace_emit_hipModuleUnload_enter(
      (uint64_t)(uintptr_t)(__rocm_in_module)); /* __ROCM_CURATED__: hipModuleUnload */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipModuleUnload, hip::GetHipDispatchTable()->hipModuleUnload_fn(module));
  CATCH;
}
extern "C" hipError_t hipOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize,
                                                               const void* f, int numBlocks,
                                                               int blockSize) {
  auto const __rocm_in_dynamicSmemSize = dynamicSmemSize;
  auto const __rocm_in_f = f;
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_blockSize = blockSize;
  rocm_trace_emit_hipOccupancyAvailableDynamicSMemPerBlock_enter(
      (const void*)(uintptr_t)(__rocm_in_dynamicSmemSize),
      (const void*)(uintptr_t)(__rocm_in_f),
      (__rocm_in_numBlocks),
      (__rocm_in_blockSize)); /* __ROCM_CURATED__: hipOccupancyAvailableDynamicSMemPerBlock */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipOccupancyAvailableDynamicSMemPerBlock, hip::GetHipDispatchTable()->hipOccupancyAvailableDynamicSMemPerBlock_fn(dynamicSmemSize, f,
                                                                              numBlocks, blockSize));
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                                   int blockSize,
                                                                   size_t dynSharedMemPerBlk) {
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_f = f;
  auto const __rocm_in_blockSize = blockSize;
  auto const __rocm_in_dynSharedMemPerBlk = dynSharedMemPerBlk;
  rocm_trace_emit_hipOccupancyMaxActiveBlocksPerMultiprocessor_enter(
      (const void*)(uintptr_t)(__rocm_in_numBlocks),
      (const void*)(uintptr_t)(__rocm_in_f),
      (__rocm_in_blockSize),
      (__rocm_in_dynSharedMemPerBlk)); /* __ROCM_CURATED__: hipOccupancyMaxActiveBlocksPerMultiprocessor */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipOccupancyMaxActiveBlocksPerMultiprocessor, hip::GetHipDispatchTable()->hipOccupancyMaxActiveBlocksPerMultiprocessor_fn(
          numBlocks, f, blockSize, dynSharedMemPerBlk));
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_f = f;
  auto const __rocm_in_blockSize = blockSize;
  auto const __rocm_in_dynSharedMemPerBlk = dynSharedMemPerBlk;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_numBlocks),
      (const void*)(uintptr_t)(__rocm_in_f),
      (__rocm_in_blockSize),
      (__rocm_in_dynSharedMemPerBlk),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, hip::GetHipDispatchTable()->hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn(
          numBlocks, f, blockSize, dynSharedMemPerBlk, flags));
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                        const void* f, size_t dynSharedMemPerBlk,
                                                        int blockSizeLimit) {
  auto const __rocm_in_gridSize = gridSize;
  auto const __rocm_in_blockSize = blockSize;
  auto const __rocm_in_f = f;
  auto const __rocm_in_dynSharedMemPerBlk = dynSharedMemPerBlk;
  auto const __rocm_in_blockSizeLimit = blockSizeLimit;
  rocm_trace_emit_hipOccupancyMaxPotentialBlockSize_enter(
      (const void*)(uintptr_t)(__rocm_in_gridSize),
      (const void*)(uintptr_t)(__rocm_in_blockSize),
      (const void*)(uintptr_t)(__rocm_in_f),
      (__rocm_in_dynSharedMemPerBlk),
      (__rocm_in_blockSizeLimit)); /* __ROCM_CURATED__: hipOccupancyMaxPotentialBlockSize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipOccupancyMaxPotentialBlockSize, hip::GetHipDispatchTable()->hipOccupancyMaxPotentialBlockSize_fn(
                                    gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit));
  CATCH;
}
hipError_t hipPeekAtLastError(void) {
  TRY;
  return hip::GetHipDispatchTable()->hipPeekAtLastError_fn();
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxActiveClusters(int* numClusters, const void* func,
                                                    const hipLaunchConfig_t* launchConfig) {
  auto const __rocm_in_numClusters = numClusters;
  auto const __rocm_in_func = func;
  auto const __rocm_in_launchConfig = launchConfig;
  rocm_trace_emit_hipOccupancyMaxActiveClusters_enter(
      (const void*)(uintptr_t)(__rocm_in_numClusters),
      (const void*)(uintptr_t)(__rocm_in_func),
      (const void*)(uintptr_t)(__rocm_in_launchConfig)); /* __ROCM_CURATED__: hipOccupancyMaxActiveClusters */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipOccupancyMaxActiveClusters, hip::GetHipDispatchTable()->hipOccupancyMaxActiveClusters_fn(numClusters, func, launchConfig));
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxPotentialClusterSize(int* clusterSize, const void* func,
                                                          const hipLaunchConfig_t* config) {
  auto const __rocm_in_clusterSize = clusterSize;
  auto const __rocm_in_func = func;
  auto const __rocm_in_config = config;
  rocm_trace_emit_hipOccupancyMaxPotentialClusterSize_enter(
      (const void*)(uintptr_t)(__rocm_in_clusterSize),
      (const void*)(uintptr_t)(__rocm_in_func),
      (const void*)(uintptr_t)(__rocm_in_config)); /* __ROCM_CURATED__: hipOccupancyMaxPotentialClusterSize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipOccupancyMaxPotentialClusterSize, hip::GetHipDispatchTable()->hipOccupancyMaxPotentialClusterSize_fn(clusterSize, func, config));
  CATCH;
}
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
  auto const __rocm_in_data = data;
  auto const __rocm_in_attribute = attribute;
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipPointerGetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_data),
      (int32_t)(__rocm_in_attribute),
      (uint64_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipPointerGetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipPointerGetAttribute, hip::GetHipDispatchTable()->hipPointerGetAttribute_fn(data, attribute, ptr));
  CATCH;
}
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  auto const __rocm_in_attributes = attributes;
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipPointerGetAttributes_enter(
      (const void*)(uintptr_t)(__rocm_in_attributes),
      (const void*)(uintptr_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipPointerGetAttributes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipPointerGetAttributes, hip::GetHipDispatchTable()->hipPointerGetAttributes_fn(attributes, ptr));
  CATCH;
}
hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr) {
  auto const __rocm_in_value = value;
  auto const __rocm_in_attribute = attribute;
  auto const __rocm_in_ptr = ptr;
  rocm_trace_emit_hipPointerSetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_value),
      (int32_t)(__rocm_in_attribute),
      (uint64_t)(__rocm_in_ptr)); /* __ROCM_CURATED__: hipPointerSetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipPointerSetAttribute, hip::GetHipDispatchTable()->hipPointerSetAttribute_fn(value, attribute, ptr));
  CATCH;
}
hipError_t hipProfilerStart() {
  TRY;
  return hip::GetHipDispatchTable()->hipProfilerStart_fn();
  CATCH;
}
hipError_t hipProfilerStop() {
  TRY;
  return hip::GetHipDispatchTable()->hipProfilerStop_fn();
  CATCH;
}
hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
  TRY;
  return hip::GetHipDispatchTable()->hipRuntimeGetVersion_fn(runtimeVersion);
  CATCH;
}
hipError_t hipSetDevice(int deviceId) {
  auto const __rocm_in_deviceId = deviceId;
  rocm_trace_emit_hipSetDevice_enter(
      (__rocm_in_deviceId)); /* __ROCM_CURATED__: hipSetDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipSetDevice, hip::GetHipDispatchTable()->hipSetDevice_fn(deviceId));
  CATCH;
}
hipError_t hipSetDeviceFlags(unsigned flags) {
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipSetDeviceFlags_enter(
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipSetDeviceFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipSetDeviceFlags, hip::GetHipDispatchTable()->hipSetDeviceFlags_fn(flags));
  CATCH;
}
extern "C" hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  auto const __rocm_in_arg = arg;
  auto const __rocm_in_size = size;
  auto const __rocm_in_offset = offset;
  rocm_trace_emit_hipSetupArgument_enter(
      (const void*)(uintptr_t)(__rocm_in_arg),
      (__rocm_in_size),
      (__rocm_in_offset)); /* __ROCM_CURATED__: hipSetupArgument */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipSetupArgument, hip::GetHipDispatchTable()->hipSetupArgument_fn(arg, size, offset));
  CATCH;
}
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipSignalExternalSemaphoresAsync_fn(extSemArray, paramsArray,
                                                                         numExtSems, stream);
  CATCH;
}
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_callback = callback;
  auto const __rocm_in_userData = userData;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamAddCallback_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_callback),
      (const void*)(uintptr_t)(__rocm_in_userData),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamAddCallback */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamAddCallback, hip::GetHipDispatchTable()->hipStreamAddCallback_fn(stream, callback, userData, flags));
  CATCH;
}
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length,
                                   unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_length = length;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamAttachMemAsync_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (__rocm_in_length),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamAttachMemAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamAttachMemAsync, hip::GetHipDispatchTable()->hipStreamAttachMemAsync_fn(stream, dev_ptr, length, flags));
  CATCH;
}
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_mode = mode;
  rocm_trace_emit_hipStreamBeginCapture_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (int32_t)(__rocm_in_mode)); /* __ROCM_CURATED__: hipStreamBeginCapture */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamBeginCapture, hip::GetHipDispatchTable()->hipStreamBeginCapture_fn(stream, mode));
  CATCH;
}
hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  rocm_trace_emit_hipStreamCopyAttributes_enter(
      (uint64_t)(uintptr_t)(__rocm_in_dst),
      (uint64_t)(uintptr_t)(__rocm_in_src)); /* __ROCM_CURATED__: hipStreamCopyAttributes */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamCopyAttributes, hip::GetHipDispatchTable()->hipStreamCopyAttributes_fn(dst, src));
  CATCH;
}
hipError_t hipStreamCreate(hipStream_t* stream) {
  rocm_trace_emit_hipStreamCreate_enter(); /* __ROCM_CURATED__: hipStreamCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipStreamCreate, hip::GetHipDispatchTable()->hipStreamCreate_fn(stream), stream);
  CATCH;
}
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamCreateWithFlags_enter(
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamCreateWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipStreamCreateWithFlags, hip::GetHipDispatchTable()->hipStreamCreateWithFlags_fn(stream, flags), stream);
  CATCH;
}
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_priority = priority;
  rocm_trace_emit_hipStreamCreateWithPriority_enter(
      (__rocm_in_flags),
      (__rocm_in_priority)); /* __ROCM_CURATED__: hipStreamCreateWithPriority */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipStreamCreateWithPriority, hip::GetHipDispatchTable()->hipStreamCreateWithPriority_fn(stream, flags, priority), stream);
  CATCH;
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamDestroy_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamDestroy, hip::GetHipDispatchTable()->hipStreamDestroy_fn(stream));
  CATCH;
}
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamEndCapture_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamEndCapture */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipStreamEndCapture, hip::GetHipDispatchTable()->hipStreamEndCapture_fn(stream, pGraph), pGraph);
  CATCH;
}
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_pCaptureStatus = pCaptureStatus;
  auto const __rocm_in_pId = pId;
  rocm_trace_emit_hipStreamGetCaptureInfo_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_pCaptureStatus),
      (const void*)(uintptr_t)(__rocm_in_pId)); /* __ROCM_CURATED__: hipStreamGetCaptureInfo */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetCaptureInfo, hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_fn(stream, pCaptureStatus, pId));
  CATCH;
}
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_captureStatus_out = captureStatus_out;
  auto const __rocm_in_id_out = id_out;
  auto const __rocm_in_graph_out = graph_out;
  auto const __rocm_in_dependencies_out = dependencies_out;
  auto const __rocm_in_numDependencies_out = numDependencies_out;
  rocm_trace_emit_hipStreamGetCaptureInfo_v2_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_captureStatus_out),
      (const void*)(uintptr_t)(__rocm_in_id_out),
      (const void*)(uintptr_t)(__rocm_in_graph_out),
      (const void*)(uintptr_t)(__rocm_in_dependencies_out),
      (const void*)(uintptr_t)(__rocm_in_numDependencies_out)); /* __ROCM_CURATED__: hipStreamGetCaptureInfo_v2 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetCaptureInfo_v2, hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_v2_fn(
          stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out));
  CATCH;
}
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipStreamGetDevice_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_device)); /* __ROCM_CURATED__: hipStreamGetDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetDevice, hip::GetHipDispatchTable()->hipStreamGetDevice_fn(stream, device));
  CATCH;
}
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamGetFlags_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamGetFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipStreamGetFlags, hip::GetHipDispatchTable()->hipStreamGetFlags_fn(stream, flags), flags);
  CATCH;
}
hipError_t hipStreamGetId(hipStream_t stream, unsigned long long* streamId) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_streamId = streamId;
  rocm_trace_emit_hipStreamGetId_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_streamId)); /* __ROCM_CURATED__: hipStreamGetId */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetId, hip::GetHipDispatchTable()->hipStreamGetId_fn(stream, streamId));
  CATCH;
}
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamGetPriority_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamGetPriority */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipStreamGetPriority, hip::GetHipDispatchTable()->hipStreamGetPriority_fn(stream, priority), priority);
  CATCH;
}
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamIsCapturing_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamIsCapturing */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED(hipStreamIsCapturing, hip::GetHipDispatchTable()->hipStreamIsCapturing_fn(stream, pCaptureStatus), (int32_t*)(pCaptureStatus));
  CATCH;
}
hipError_t hipStreamQuery(hipStream_t stream) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamQuery_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamQuery */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamQuery, hip::GetHipDispatchTable()->hipStreamQuery_fn(stream));
  CATCH;
}
hipError_t hipStreamSynchronize(hipStream_t stream) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamSynchronize_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamSynchronize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamSynchronize, hip::GetHipDispatchTable()->hipStreamSynchronize_fn(stream));
  CATCH;
}
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_dependencies = dependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamUpdateCaptureDependencies_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_dependencies),
      (__rocm_in_numDependencies),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamUpdateCaptureDependencies */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamUpdateCaptureDependencies, hip::GetHipDispatchTable()->hipStreamUpdateCaptureDependencies_fn(
                                    stream, dependencies, numDependencies, flags));
  CATCH;
}
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_event = event;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamWaitEvent_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (uint64_t)(uintptr_t)(__rocm_in_event),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamWaitEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamWaitEvent, hip::GetHipDispatchTable()->hipStreamWaitEvent_fn(stream, event, flags));
  CATCH;
}
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_value = value;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_mask = mask;
  rocm_trace_emit_hipStreamWaitValue32_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_value),
      (__rocm_in_flags),
      (__rocm_in_mask)); /* __ROCM_CURATED__: hipStreamWaitValue32 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamWaitValue32, hip::GetHipDispatchTable()->hipStreamWaitValue32_fn(stream, ptr, value, flags, mask));
  CATCH;
}
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_value = value;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_mask = mask;
  rocm_trace_emit_hipStreamWaitValue64_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_value),
      (__rocm_in_flags),
      (__rocm_in_mask)); /* __ROCM_CURATED__: hipStreamWaitValue64 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamWaitValue64, hip::GetHipDispatchTable()->hipStreamWaitValue64_fn(stream, ptr, value, flags, mask));
  CATCH;
}
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value,
                                 unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_value = value;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamWriteValue32_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_value),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamWriteValue32 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamWriteValue32, hip::GetHipDispatchTable()->hipStreamWriteValue32_fn(stream, ptr, value, flags));
  CATCH;
}
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value,
                                 unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_value = value;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamWriteValue64_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (__rocm_in_value),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamWriteValue64 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamWriteValue64, hip::GetHipDispatchTable()->hipStreamWriteValue64_fn(stream, ptr, value, flags));
  CATCH;
}
hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_count = count;
  auto const __rocm_in_paramArray = paramArray;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamBatchMemOp_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_paramArray),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamBatchMemOp */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamBatchMemOp, hip::GetHipDispatchTable()->hipStreamBatchMemOp_fn(stream, count, paramArray, flags));
  CATCH;
}
hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject, const HIP_RESOURCE_DESC* pResDesc,
                              const HIP_TEXTURE_DESC* pTexDesc,
                              const HIP_RESOURCE_VIEW_DESC* pResViewDesc) {
  auto const __rocm_in_pTexObject = pTexObject;
  auto const __rocm_in_pResDesc = pResDesc;
  auto const __rocm_in_pTexDesc = pTexDesc;
  auto const __rocm_in_pResViewDesc = pResViewDesc;
  rocm_trace_emit_hipTexObjectCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_pTexObject),
      (const void*)(uintptr_t)(__rocm_in_pResDesc),
      (const void*)(uintptr_t)(__rocm_in_pTexDesc),
      (const void*)(uintptr_t)(__rocm_in_pResViewDesc)); /* __ROCM_CURATED__: hipTexObjectCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexObjectCreate, hip::GetHipDispatchTable()->hipTexObjectCreate_fn(
                                    pTexObject, pResDesc, pTexDesc, pResViewDesc));
  CATCH;
}
hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
  auto const __rocm_in_texObject = texObject;
  rocm_trace_emit_hipTexObjectDestroy_enter(
      (const void*)(uintptr_t)(__rocm_in_texObject)); /* __ROCM_CURATED__: hipTexObjectDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexObjectDestroy, hip::GetHipDispatchTable()->hipTexObjectDestroy_fn(texObject));
  CATCH;
}
hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, hipTextureObject_t texObject) {
  auto const __rocm_in_pResDesc = pResDesc;
  auto const __rocm_in_texObject = texObject;
  rocm_trace_emit_hipTexObjectGetResourceDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_pResDesc),
      (const void*)(uintptr_t)(__rocm_in_texObject)); /* __ROCM_CURATED__: hipTexObjectGetResourceDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexObjectGetResourceDesc, hip::GetHipDispatchTable()->hipTexObjectGetResourceDesc_fn(pResDesc, texObject));
  CATCH;
}
hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                           hipTextureObject_t texObject) {
  auto const __rocm_in_pResViewDesc = pResViewDesc;
  auto const __rocm_in_texObject = texObject;
  rocm_trace_emit_hipTexObjectGetResourceViewDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_pResViewDesc),
      (const void*)(uintptr_t)(__rocm_in_texObject)); /* __ROCM_CURATED__: hipTexObjectGetResourceViewDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexObjectGetResourceViewDesc, hip::GetHipDispatchTable()->hipTexObjectGetResourceViewDesc_fn(pResViewDesc, texObject));
  CATCH;
}
hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, hipTextureObject_t texObject) {
  auto const __rocm_in_pTexDesc = pTexDesc;
  auto const __rocm_in_texObject = texObject;
  rocm_trace_emit_hipTexObjectGetTextureDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_pTexDesc),
      (const void*)(uintptr_t)(__rocm_in_texObject)); /* __ROCM_CURATED__: hipTexObjectGetTextureDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexObjectGetTextureDesc, hip::GetHipDispatchTable()->hipTexObjectGetTextureDesc_fn(pTexDesc, texObject));
  CATCH;
}
hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef) {
  auto const __rocm_in_dev_ptr = dev_ptr;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetAddress_enter(
      (const void*)(uintptr_t)(__rocm_in_dev_ptr),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetAddress */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetAddress, hip::GetHipDispatchTable()->hipTexRefGetAddress_fn(dev_ptr, texRef));
  CATCH;
}
hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode* pam, const textureReference* texRef,
                                   int dim) {
  auto const __rocm_in_pam = pam;
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_dim = dim;
  rocm_trace_emit_hipTexRefGetAddressMode_enter(
      (const void*)(uintptr_t)(__rocm_in_pam),
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (__rocm_in_dim)); /* __ROCM_CURATED__: hipTexRefGetAddressMode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetAddressMode, hip::GetHipDispatchTable()->hipTexRefGetAddressMode_fn(pam, texRef, dim));
  CATCH;
}
hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode* pfm, const textureReference* texRef) {
  auto const __rocm_in_pfm = pfm;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetFilterMode_enter(
      (const void*)(uintptr_t)(__rocm_in_pfm),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetFilterMode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetFilterMode, hip::GetHipDispatchTable()->hipTexRefGetFilterMode_fn(pfm, texRef));
  CATCH;
}
hipError_t hipTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef) {
  auto const __rocm_in_pFlags = pFlags;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_pFlags),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetFlags, hip::GetHipDispatchTable()->hipTexRefGetFlags_fn(pFlags, texRef));
  CATCH;
}
hipError_t hipTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels,
                              const textureReference* texRef) {
  auto const __rocm_in_pFormat = pFormat;
  auto const __rocm_in_pNumChannels = pNumChannels;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetFormat_enter(
      (const void*)(uintptr_t)(__rocm_in_pFormat),
      (const void*)(uintptr_t)(__rocm_in_pNumChannels),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetFormat */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetFormat, hip::GetHipDispatchTable()->hipTexRefGetFormat_fn(pFormat, pNumChannels, texRef));
  CATCH;
}
hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef) {
  auto const __rocm_in_pmaxAnsio = pmaxAnsio;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetMaxAnisotropy_enter(
      (const void*)(uintptr_t)(__rocm_in_pmaxAnsio),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetMaxAnisotropy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetMaxAnisotropy, hip::GetHipDispatchTable()->hipTexRefGetMaxAnisotropy_fn(pmaxAnsio, texRef));
  CATCH;
}
extern "C" hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,
                                                 const textureReference* texRef) {
  auto const __rocm_in_pArray = pArray;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetMipMappedArray_enter(
      (const void*)(uintptr_t)(__rocm_in_pArray),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetMipMappedArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetMipMappedArray, hip::GetHipDispatchTable()->hipTexRefGetMipMappedArray_fn(pArray, texRef));
  CATCH;
}
hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode* pfm,
                                        const textureReference* texRef) {
  auto const __rocm_in_pfm = pfm;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetMipmapFilterMode_enter(
      (const void*)(uintptr_t)(__rocm_in_pfm),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetMipmapFilterMode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetMipmapFilterMode, hip::GetHipDispatchTable()->hipTexRefGetMipmapFilterMode_fn(pfm, texRef));
  CATCH;
}
hipError_t hipTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef) {
  auto const __rocm_in_pbias = pbias;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetMipmapLevelBias_enter(
      (const void*)(uintptr_t)(__rocm_in_pbias),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetMipmapLevelBias */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetMipmapLevelBias, hip::GetHipDispatchTable()->hipTexRefGetMipmapLevelBias_fn(pbias, texRef));
  CATCH;
}
hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp,
                                        const textureReference* texRef) {
  auto const __rocm_in_pminMipmapLevelClamp = pminMipmapLevelClamp;
  auto const __rocm_in_pmaxMipmapLevelClamp = pmaxMipmapLevelClamp;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetMipmapLevelClamp_enter(
      (const void*)(uintptr_t)(__rocm_in_pminMipmapLevelClamp),
      (const void*)(uintptr_t)(__rocm_in_pmaxMipmapLevelClamp),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetMipmapLevelClamp */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetMipmapLevelClamp, hip::GetHipDispatchTable()->hipTexRefGetMipmapLevelClamp_fn(
                                    pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef));
  CATCH;
}
hipError_t hipTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr,
                               size_t bytes) {
  auto const __rocm_in_ByteOffset = ByteOffset;
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_dptr = dptr;
  auto const __rocm_in_bytes = bytes;
  rocm_trace_emit_hipTexRefSetAddress_enter(
      (const void*)(uintptr_t)(__rocm_in_ByteOffset),
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (uint64_t)(__rocm_in_dptr),
      (__rocm_in_bytes)); /* __ROCM_CURATED__: hipTexRefSetAddress */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetAddress, hip::GetHipDispatchTable()->hipTexRefSetAddress_fn(ByteOffset, texRef, dptr, bytes));
  CATCH;
}
hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr, size_t Pitch) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_dptr = dptr;
  auto const __rocm_in_Pitch = Pitch;
  rocm_trace_emit_hipTexRefSetAddress2D_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (uint64_t)(__rocm_in_dptr),
      (__rocm_in_Pitch)); /* __ROCM_CURATED__: hipTexRefSetAddress2D */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetAddress2D, hip::GetHipDispatchTable()->hipTexRefSetAddress2D_fn(texRef, desc, dptr, Pitch));
  CATCH;
}
hipError_t hipTexRefSetAddressMode(textureReference* texRef, int dim,
                                   enum hipTextureAddressMode am) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_dim = dim;
  auto const __rocm_in_am = am;
  rocm_trace_emit_hipTexRefSetAddressMode_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (__rocm_in_dim),
      (int32_t)(__rocm_in_am)); /* __ROCM_CURATED__: hipTexRefSetAddressMode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetAddressMode, hip::GetHipDispatchTable()->hipTexRefSetAddressMode_fn(texRef, dim, am));
  CATCH;
}
hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags) {
  auto const __rocm_in_tex = tex;
  auto const __rocm_in_array = array;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipTexRefSetArray_enter(
      (const void*)(uintptr_t)(__rocm_in_tex),
      (const void*)(uintptr_t)(__rocm_in_array),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipTexRefSetArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetArray, hip::GetHipDispatchTable()->hipTexRefSetArray_fn(tex, array, flags));
  CATCH;
}
hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_pBorderColor = pBorderColor;
  rocm_trace_emit_hipTexRefSetBorderColor_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (const void*)(uintptr_t)(__rocm_in_pBorderColor)); /* __ROCM_CURATED__: hipTexRefSetBorderColor */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetBorderColor, hip::GetHipDispatchTable()->hipTexRefSetBorderColor_fn(texRef, pBorderColor));
  CATCH;
}
hipError_t hipTexRefSetFilterMode(textureReference* texRef, enum hipTextureFilterMode fm) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_fm = fm;
  rocm_trace_emit_hipTexRefSetFilterMode_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (int32_t)(__rocm_in_fm)); /* __ROCM_CURATED__: hipTexRefSetFilterMode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetFilterMode, hip::GetHipDispatchTable()->hipTexRefSetFilterMode_fn(texRef, fm));
  CATCH;
}
hipError_t hipTexRefSetFlags(textureReference* texRef, unsigned int Flags) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_Flags = Flags;
  rocm_trace_emit_hipTexRefSetFlags_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (__rocm_in_Flags)); /* __ROCM_CURATED__: hipTexRefSetFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetFlags, hip::GetHipDispatchTable()->hipTexRefSetFlags_fn(texRef, Flags));
  CATCH;
}
hipError_t hipTexRefSetFormat(textureReference* texRef, hipArray_Format fmt,
                              int NumPackedComponents) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_fmt = fmt;
  auto const __rocm_in_NumPackedComponents = NumPackedComponents;
  rocm_trace_emit_hipTexRefSetFormat_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (int32_t)(__rocm_in_fmt),
      (__rocm_in_NumPackedComponents)); /* __ROCM_CURATED__: hipTexRefSetFormat */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetFormat, hip::GetHipDispatchTable()->hipTexRefSetFormat_fn(texRef, fmt, NumPackedComponents));
  CATCH;
}
hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_maxAniso = maxAniso;
  rocm_trace_emit_hipTexRefSetMaxAnisotropy_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (__rocm_in_maxAniso)); /* __ROCM_CURATED__: hipTexRefSetMaxAnisotropy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetMaxAnisotropy, hip::GetHipDispatchTable()->hipTexRefSetMaxAnisotropy_fn(texRef, maxAniso));
  CATCH;
}
hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef, enum hipTextureFilterMode fm) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_fm = fm;
  rocm_trace_emit_hipTexRefSetMipmapFilterMode_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (int32_t)(__rocm_in_fm)); /* __ROCM_CURATED__: hipTexRefSetMipmapFilterMode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetMipmapFilterMode, hip::GetHipDispatchTable()->hipTexRefSetMipmapFilterMode_fn(texRef, fm));
  CATCH;
}
hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_bias = bias;
  rocm_trace_emit_hipTexRefSetMipmapLevelBias_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (__rocm_in_bias)); /* __ROCM_CURATED__: hipTexRefSetMipmapLevelBias */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetMipmapLevelBias, hip::GetHipDispatchTable()->hipTexRefSetMipmapLevelBias_fn(texRef, bias));
  CATCH;
}
hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp,
                                        float maxMipMapLevelClamp) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_minMipMapLevelClamp = minMipMapLevelClamp;
  auto const __rocm_in_maxMipMapLevelClamp = maxMipMapLevelClamp;
  rocm_trace_emit_hipTexRefSetMipmapLevelClamp_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (__rocm_in_minMipMapLevelClamp),
      (__rocm_in_maxMipMapLevelClamp)); /* __ROCM_CURATED__: hipTexRefSetMipmapLevelClamp */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetMipmapLevelClamp, hip::GetHipDispatchTable()->hipTexRefSetMipmapLevelClamp_fn(
                                    texRef, minMipMapLevelClamp, maxMipMapLevelClamp));
  CATCH;
}
hipError_t hipTexRefSetMipmappedArray(textureReference* texRef,
                                      struct hipMipmappedArray* mipmappedArray,
                                      unsigned int Flags) {
  auto const __rocm_in_texRef = texRef;
  auto const __rocm_in_mipmappedArray = mipmappedArray;
  auto const __rocm_in_Flags = Flags;
  rocm_trace_emit_hipTexRefSetMipmappedArray_enter(
      (const void*)(uintptr_t)(__rocm_in_texRef),
      (const void*)(uintptr_t)(__rocm_in_mipmappedArray),
      (__rocm_in_Flags)); /* __ROCM_CURATED__: hipTexRefSetMipmappedArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefSetMipmappedArray, hip::GetHipDispatchTable()->hipTexRefSetMipmappedArray_fn(texRef, mipmappedArray, Flags));
  CATCH;
}
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  auto const __rocm_in_mode = mode;
  rocm_trace_emit_hipThreadExchangeStreamCaptureMode_enter(
      (const void*)(uintptr_t)(__rocm_in_mode)); /* __ROCM_CURATED__: hipThreadExchangeStreamCaptureMode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipThreadExchangeStreamCaptureMode, hip::GetHipDispatchTable()->hipThreadExchangeStreamCaptureMode_fn(mode));
  CATCH;
}
extern "C" hipError_t hipUnbindTexture(const textureReference* tex) {
  auto const __rocm_in_tex = tex;
  rocm_trace_emit_hipUnbindTexture_enter(
      (const void*)(uintptr_t)(__rocm_in_tex)); /* __ROCM_CURATED__: hipUnbindTexture */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipUnbindTexture, hip::GetHipDispatchTable()->hipUnbindTexture_fn(tex));
  CATCH;
}
hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                               unsigned int initialRefcount, unsigned int flags) {
  auto const __rocm_in_object_out = object_out;
  auto const __rocm_in_ptr = ptr;
  auto const __rocm_in_destroy = destroy;
  auto const __rocm_in_initialRefcount = initialRefcount;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipUserObjectCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_object_out),
      (const void*)(uintptr_t)(__rocm_in_ptr),
      (const void*)(uintptr_t)(__rocm_in_destroy),
      (__rocm_in_initialRefcount),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipUserObjectCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipUserObjectCreate, hip::GetHipDispatchTable()->hipUserObjectCreate_fn(object_out, ptr, destroy, initialRefcount,
                                                         flags));
  CATCH;
}
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
  auto const __rocm_in_object = object;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipUserObjectRelease_enter(
      (const void*)(uintptr_t)(__rocm_in_object),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipUserObjectRelease */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipUserObjectRelease, hip::GetHipDispatchTable()->hipUserObjectRelease_fn(object, count));
  CATCH;
}
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
  auto const __rocm_in_object = object;
  auto const __rocm_in_count = count;
  rocm_trace_emit_hipUserObjectRetain_enter(
      (const void*)(uintptr_t)(__rocm_in_object),
      (__rocm_in_count)); /* __ROCM_CURATED__: hipUserObjectRetain */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipUserObjectRetain, hip::GetHipDispatchTable()->hipUserObjectRetain_fn(object, count));
  CATCH;
}
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                          const hipExternalSemaphoreWaitParams* paramsArray,
                                          unsigned int numExtSems, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipWaitExternalSemaphoresAsync_fn(extSemArray, paramsArray,
                                                                       numExtSems, stream);
  CATCH;
}
extern "C" hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                     hipChannelFormatKind f) {
  auto const __rocm_in_x = x;
  auto const __rocm_in_y = y;
  auto const __rocm_in_z = z;
  auto const __rocm_in_w = w;
  auto const __rocm_in_f = f;
  rocm_trace_emit_hipCreateChannelDesc_enter(
      (__rocm_in_x),
      (__rocm_in_y),
      (__rocm_in_z),
      (__rocm_in_w),
      (int32_t)(__rocm_in_f)); /* __ROCM_CURATED__: hipCreateChannelDesc */
  TRY;
  do {
    auto __rocm_rv = (hip::GetHipDispatchTable()->hipCreateChannelDesc_fn(x, y, z, w, f));
    rocm_trace_emit_hipCreateChannelDesc_exit();
    return __rocm_rv;
  } while (0);
  CATCHRET(hipChannelFormatDesc)
}

#ifdef _WIN32
#define DllExport __declspec(dllexport)
#else  // !_WIN32
#define DllExport
#endif  // !_WIN32

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

hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy_spt, hip::GetHipDispatchTable()->hipMemcpy_spt_fn(dst, src, sizeBytes, kind));
  CATCH;
}
hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                                 size_t offset, hipMemcpyKind kind) {
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpyToSymbol_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpyToSymbol_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyToSymbol_spt, hip::GetHipDispatchTable()->hipMemcpyToSymbol_spt_fn(symbol, src, sizeBytes, offset, kind));
  CATCH;
}
hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                   hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpyFromSymbol_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpyFromSymbol_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyFromSymbol_spt, hip::GetHipDispatchTable()->hipMemcpyFromSymbol_spt_fn(dst, symbol, sizeBytes, offset, kind));
  CATCH;
}
hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                           size_t height, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy2D_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy2D_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2D_spt, hip::GetHipDispatchTable()->hipMemcpy2D_spt_fn(dst, dpitch, src, spitch, width, height, kind));
  CATCH;
}
hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                    size_t hOffset, size_t width, size_t height,
                                    hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy2DFromArray_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy2DFromArray_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DFromArray_spt, hip::GetHipDispatchTable()->hipMemcpy2DFromArray_spt_fn(
                                    dst, dpitch, src, wOffset, hOffset, width, height, kind));
  CATCH;
}
hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p) {
  auto const __rocm_in_p = p;
  rocm_trace_emit_hipMemcpy3D_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_p)); /* __ROCM_CURATED__: hipMemcpy3D_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy3D_spt, hip::GetHipDispatchTable()->hipMemcpy3D_spt_fn(p));
  CATCH;
}
hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_value = value;
  auto const __rocm_in_sizeBytes = sizeBytes;
  rocm_trace_emit_hipMemset_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_value),
      (__rocm_in_sizeBytes)); /* __ROCM_CURATED__: hipMemset_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset_spt, hip::GetHipDispatchTable()->hipMemset_spt_fn(dst, value, sizeBytes));
  CATCH;
}
hipError_t hipMemsetAsync_spt(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_value = value;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_value),
      (__rocm_in_sizeBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetAsync_spt, hip::GetHipDispatchTable()->hipMemsetAsync_spt_fn(dst, value, sizeBytes, stream));
  CATCH;
}
hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_pitch = pitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  rocm_trace_emit_hipMemset2D_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_pitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height)); /* __ROCM_CURATED__: hipMemset2D_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset2D_spt, hip::GetHipDispatchTable()->hipMemset2D_spt_fn(dst, pitch, value, width, height));
  CATCH;
}
hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value, size_t width, size_t height,
                                hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_pitch = pitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemset2DAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_pitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemset2DAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset2DAsync_spt, hip::GetHipDispatchTable()->hipMemset2DAsync_spt_fn(dst, pitch, value, width, height, stream));
  CATCH;
}
hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                hipStream_t stream) {
  auto const __rocm_in_value = value;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemset3DAsync_spt_enter(
      (__rocm_in_value),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemset3DAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset3DAsync_spt, hip::GetHipDispatchTable()->hipMemset3DAsync_spt_fn(pitchedDevPtr, value, extent, stream));
  CATCH;
}
hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipMemset3D_spt_enter(
      (__rocm_in_value)); /* __ROCM_CURATED__: hipMemset3D_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemset3D_spt, hip::GetHipDispatchTable()->hipMemset3D_spt_fn(pitchedDevPtr, value, extent));
  CATCH;
}
hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                              hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyAsync_spt, hip::GetHipDispatchTable()->hipMemcpyAsync_spt_fn(dst, src, sizeBytes, kind, stream));
  CATCH;
}
hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms* p, hipStream_t stream) {
  auto const __rocm_in_p = p;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy3DAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_p),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy3DAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy3DAsync_spt, hip::GetHipDispatchTable()->hipMemcpy3DAsync_spt_fn(p, stream));
  CATCH;
}
hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, hipMemcpyKind kind,
                                hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy2DAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy2DAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DAsync_spt, hip::GetHipDispatchTable()->hipMemcpy2DAsync_spt_fn(
                                    dst, dpitch, src, spitch, width, height, kind, stream));
  CATCH;
}
hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t sizeBytes,
                                        size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyFromSymbolAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyFromSymbolAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyFromSymbolAsync_spt, hip::GetHipDispatchTable()->hipMemcpyFromSymbolAsync_spt_fn(
                                    dst, symbol, sizeBytes, offset, kind, stream));
  CATCH;
}
hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t sizeBytes,
                                      size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_symbol = symbol;
  auto const __rocm_in_src = src;
  auto const __rocm_in_sizeBytes = sizeBytes;
  auto const __rocm_in_offset = offset;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyToSymbolAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_symbol),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_sizeBytes),
      (__rocm_in_offset),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyToSymbolAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyToSymbolAsync_spt, hip::GetHipDispatchTable()->hipMemcpyToSymbolAsync_spt_fn(
                                    symbol, src, sizeBytes, offset, kind, stream));
  CATCH;
}
hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                  size_t hOffset, size_t count, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_wOffsetSrc = wOffsetSrc;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_count = count;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpyFromArray_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_wOffsetSrc),
      (__rocm_in_hOffset),
      (__rocm_in_count),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpyFromArray_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyFromArray_spt, hip::GetHipDispatchTable()->hipMemcpyFromArray_spt_fn(
                                    dst, src, wOffsetSrc, hOffset, count, kind));
  CATCH;
}
hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy2DToArray_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy2DToArray_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DToArray_spt, hip::GetHipDispatchTable()->hipMemcpy2DToArray_spt_fn(
                                    dst, wOffset, hOffset, src, spitch, width, height, kind));
  CATCH;
}
hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                         size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                         size_t height, hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dpitch = dpitch;
  auto const __rocm_in_src = src;
  auto const __rocm_in_wOffsetSrc = wOffsetSrc;
  auto const __rocm_in_hOffsetSrc = hOffsetSrc;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy2DFromArrayAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_dpitch),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_wOffsetSrc),
      (__rocm_in_hOffsetSrc),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy2DFromArrayAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DFromArrayAsync_spt, hip::GetHipDispatchTable()->hipMemcpy2DFromArrayAsync_spt_fn(
          dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height, kind, stream));
  CATCH;
}
hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset,
                                       const void* src, size_t spitch, size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_wOffset = wOffset;
  auto const __rocm_in_hOffset = hOffset;
  auto const __rocm_in_src = src;
  auto const __rocm_in_spitch = spitch;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy2DToArrayAsync_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_wOffset),
      (__rocm_in_hOffset),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_spitch),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy2DToArrayAsync_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DToArrayAsync_spt, hip::GetHipDispatchTable()->hipMemcpy2DToArrayAsync_spt_fn(dst, wOffset, hOffset, src, spitch,
                                                                 width, height, kind, stream));
  CATCH;
}
hipError_t hipStreamQuery_spt(hipStream_t stream) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamQuery_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamQuery_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamQuery_spt, hip::GetHipDispatchTable()->hipStreamQuery_spt_fn(stream));
  CATCH;
}
hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipStreamSynchronize_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipStreamSynchronize_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamSynchronize_spt, hip::GetHipDispatchTable()->hipStreamSynchronize_spt_fn(stream));
  CATCH;
}
hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_priority = priority;
  rocm_trace_emit_hipStreamGetPriority_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_priority)); /* __ROCM_CURATED__: hipStreamGetPriority_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetPriority_spt, hip::GetHipDispatchTable()->hipStreamGetPriority_spt_fn(stream, priority));
  CATCH;
}
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_event = event;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamWaitEvent_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (uint64_t)(uintptr_t)(__rocm_in_event),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamWaitEvent_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamWaitEvent_spt, hip::GetHipDispatchTable()->hipStreamWaitEvent_spt_fn(stream, event, flags));
  CATCH;
}
hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamGetFlags_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamGetFlags_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetFlags_spt, hip::GetHipDispatchTable()->hipStreamGetFlags_spt_fn(stream, flags));
  CATCH;
}
hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback,
                                    void* userData, unsigned int flags) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_callback = callback;
  auto const __rocm_in_userData = userData;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipStreamAddCallback_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_callback),
      (const void*)(uintptr_t)(__rocm_in_userData),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipStreamAddCallback_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamAddCallback_spt, hip::GetHipDispatchTable()->hipStreamAddCallback_spt_fn(stream, callback, userData, flags));
  CATCH;
}
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream) {
  auto const __rocm_in_event = event;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipEventRecord_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_event),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipEventRecord_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipEventRecord_spt, hip::GetHipDispatchTable()->hipEventRecord_spt_fn(event, stream));
  CATCH;
}
hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim, dim3 blockDim,
                                          void** kernelParams, uint32_t sharedMemBytes,
                                          hipStream_t hStream) {
  auto const __rocm_in_f = f;
  auto const __rocm_in_gridDim = gridDim;
  auto const __rocm_in_blockDim = blockDim;
  auto const __rocm_in_kernelParams = kernelParams;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_hStream = hStream;
  rocm_trace_emit_hipLaunchCooperativeKernel_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_f),
      (__rocm_in_gridDim),
      (__rocm_in_blockDim),
      (const void*)(uintptr_t)(__rocm_in_kernelParams),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_hStream)); /* __ROCM_CURATED__: hipLaunchCooperativeKernel_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchCooperativeKernel_spt, hip::GetHipDispatchTable()->hipLaunchCooperativeKernel_spt_fn(
          f, gridDim, blockDim, kernelParams, sharedMemBytes, hStream));
  CATCH;
}

extern "C" hipError_t hipLaunchKernel_spt(const void* function_address, dim3 numBlocks,
                                          dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                          hipStream_t stream) {
  auto const __rocm_in_function_address = function_address;
  auto const __rocm_in_numBlocks = numBlocks;
  auto const __rocm_in_dimBlocks = dimBlocks;
  auto const __rocm_in_args = args;
  auto const __rocm_in_sharedMemBytes = sharedMemBytes;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipLaunchKernel_spt_enter(
      (const void*)(uintptr_t)(__rocm_in_function_address),
      (__rocm_in_numBlocks),
      (__rocm_in_dimBlocks),
      (const void*)(uintptr_t)(__rocm_in_args),
      (__rocm_in_sharedMemBytes),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipLaunchKernel_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchKernel_spt, hip::GetHipDispatchTable()->hipLaunchKernel_spt_fn(function_address, numBlocks, dimBlocks,
                                                         args, sharedMemBytes, stream));
  CATCH;
}

hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream) {
  auto const __rocm_in_graphExec = graphExec;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipGraphLaunch_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graphExec),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipGraphLaunch_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphLaunch_spt, hip::GetHipDispatchTable()->hipGraphLaunch_spt_fn(graphExec, stream));
  CATCH;
}
hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_mode = mode;
  rocm_trace_emit_hipStreamBeginCapture_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (int32_t)(__rocm_in_mode)); /* __ROCM_CURATED__: hipStreamBeginCapture_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamBeginCapture_spt, hip::GetHipDispatchTable()->hipStreamBeginCapture_spt_fn(stream, mode));
  CATCH;
}
hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_pGraph = pGraph;
  rocm_trace_emit_hipStreamEndCapture_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_pGraph)); /* __ROCM_CURATED__: hipStreamEndCapture_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamEndCapture_spt, hip::GetHipDispatchTable()->hipStreamEndCapture_spt_fn(stream, pGraph));
  CATCH;
}
hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_pCaptureStatus = pCaptureStatus;
  rocm_trace_emit_hipStreamIsCapturing_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_pCaptureStatus)); /* __ROCM_CURATED__: hipStreamIsCapturing_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamIsCapturing_spt, hip::GetHipDispatchTable()->hipStreamIsCapturing_spt_fn(stream, pCaptureStatus));
  CATCH;
}
hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                       unsigned long long* pId) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_pCaptureStatus = pCaptureStatus;
  auto const __rocm_in_pId = pId;
  rocm_trace_emit_hipStreamGetCaptureInfo_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_pCaptureStatus),
      (const void*)(uintptr_t)(__rocm_in_pId)); /* __ROCM_CURATED__: hipStreamGetCaptureInfo_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetCaptureInfo_spt, hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_spt_fn(stream, pCaptureStatus, pId));
  CATCH;
}
hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,
                                          hipStreamCaptureStatus* captureStatus_out,
                                          unsigned long long* id_out, hipGraph_t* graph_out,
                                          const hipGraphNode_t** dependencies_out,
                                          size_t* numDependencies_out) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_captureStatus_out = captureStatus_out;
  auto const __rocm_in_id_out = id_out;
  auto const __rocm_in_graph_out = graph_out;
  auto const __rocm_in_dependencies_out = dependencies_out;
  auto const __rocm_in_numDependencies_out = numDependencies_out;
  rocm_trace_emit_hipStreamGetCaptureInfo_v2_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_captureStatus_out),
      (const void*)(uintptr_t)(__rocm_in_id_out),
      (const void*)(uintptr_t)(__rocm_in_graph_out),
      (const void*)(uintptr_t)(__rocm_in_dependencies_out),
      (const void*)(uintptr_t)(__rocm_in_numDependencies_out)); /* __ROCM_CURATED__: hipStreamGetCaptureInfo_v2_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetCaptureInfo_v2_spt, hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_v2_spt_fn(
          stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out));
  CATCH;
}
hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void* userData) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_fn = fn;
  auto const __rocm_in_userData = userData;
  rocm_trace_emit_hipLaunchHostFunc_spt_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_fn),
      (const void*)(uintptr_t)(__rocm_in_userData)); /* __ROCM_CURATED__: hipLaunchHostFunc_spt */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchHostFunc_spt, hip::GetHipDispatchTable()->hipLaunchHostFunc_spt_fn(stream, fn, userData));
  CATCH;
}
extern "C" int hipGetStreamDeviceId(hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetStreamDeviceId_fn(stream);
  CATCHRET(int)
}
hipError_t hipExtGetLastError() {
  TRY;
  return hip::GetHipDispatchTable()->hipExtGetLastError_fn();
  CATCH;
}
hipError_t hipTexRefGetBorderColor(float* pBorderColor, const textureReference* texRef) {
  auto const __rocm_in_pBorderColor = pBorderColor;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetBorderColor_enter(
      (const void*)(uintptr_t)(__rocm_in_pBorderColor),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetBorderColor */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetBorderColor, hip::GetHipDispatchTable()->hipTexRefGetBorderColor_fn(pBorderColor, texRef));
  CATCH;
}
hipError_t hipTexRefGetArray(hipArray_t* pArray, const textureReference* texRef) {
  auto const __rocm_in_pArray = pArray;
  auto const __rocm_in_texRef = texRef;
  rocm_trace_emit_hipTexRefGetArray_enter(
      (const void*)(uintptr_t)(__rocm_in_pArray),
      (const void*)(uintptr_t)(__rocm_in_texRef)); /* __ROCM_CURATED__: hipTexRefGetArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipTexRefGetArray, hip::GetHipDispatchTable()->hipTexRefGetArray_fn(pArray, texRef));
  CATCH;
}
extern "C" hipError_t hipGetProcAddress(const char* symbol, void** pfn, int hipVersion,
                                        uint64_t flags,
                                        hipDriverProcAddressQueryResult* symbolStatus) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetProcAddress_fn(symbol, pfn, hipVersion, flags,
                                                          symbolStatus);
  CATCH;
}
extern "C" hipError_t hipGetProcAddress_spt(const char* symbol, void** pfn, int  hipVersion,
                                            uint64_t flags,
                                            hipDriverProcAddressQueryResult* symbolStatus) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetProcAddress_spt_fn(symbol, pfn, hipVersion, flags,
                                                              symbolStatus);
  CATCH;
}
hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
                                        const hipGraphNode_t* dependencies,
                                        const hipGraphEdgeData* dependencyData,
                                        size_t numDependencies, hipStreamCaptureMode mode) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_dependencies = dependencies;
  auto const __rocm_in_dependencyData = dependencyData;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_mode = mode;
  rocm_trace_emit_hipStreamBeginCaptureToGraph_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_dependencies),
      (const void*)(uintptr_t)(__rocm_in_dependencyData),
      (__rocm_in_numDependencies),
      (int32_t)(__rocm_in_mode)); /* __ROCM_CURATED__: hipStreamBeginCaptureToGraph */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamBeginCaptureToGraph, hip::GetHipDispatchTable()->hipStreamBeginCaptureToGraph_fn(
          stream, graph, dependencies, dependencyData, numDependencies, mode));
  CATCH;
}
hipError_t hipGetFuncBySymbol(hipFunction_t* functionPtr, const void* symbolPtr) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetFuncBySymbol_fn(functionPtr, symbolPtr);
  CATCH;
}
hipError_t hipDrvGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_memsetParams = memsetParams;
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipDrvGraphExecMemsetNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_memsetParams),
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipDrvGraphExecMemsetNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvGraphExecMemsetNodeSetParams, hip::GetHipDispatchTable()->hipDrvGraphExecMemsetNodeSetParams_fn(hGraphExec, hNode,
                                                                        memsetParams, ctx));
  CATCH;
}
hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec, unsigned long long* flags) {
  auto const __rocm_in_graphExec = graphExec;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGraphExecGetFlags_enter(
      (uint64_t)(uintptr_t)(__rocm_in_graphExec),
      (const void*)(uintptr_t)(__rocm_in_flags)); /* __ROCM_CURATED__: hipGraphExecGetFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecGetFlags, hip::GetHipDispatchTable()->hipGraphExecGetFlags_fn(graphExec, flags));
  CATCH;
}
hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     hipDeviceptr_t dptr) {
  auto const __rocm_in_phGraphNode = phGraphNode;
  auto const __rocm_in_hGraph = hGraph;
  auto const __rocm_in_dependencies = dependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_dptr = dptr;
  rocm_trace_emit_hipDrvGraphAddMemFreeNode_enter(
      (const void*)(uintptr_t)(__rocm_in_phGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_hGraph),
      (const void*)(uintptr_t)(__rocm_in_dependencies),
      (__rocm_in_numDependencies),
      (uint64_t)(__rocm_in_dptr)); /* __ROCM_CURATED__: hipDrvGraphAddMemFreeNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvGraphAddMemFreeNode, hip::GetHipDispatchTable()->hipDrvGraphAddMemFreeNode_fn(
                                    phGraphNode, hGraph, dependencies, numDependencies, dptr));
  CATCH;
}
hipError_t hipDrvGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_copyParams = copyParams;
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipDrvGraphExecMemcpyNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_copyParams),
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipDrvGraphExecMemcpyNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvGraphExecMemcpyNodeSetParams, hip::GetHipDispatchTable()->hipDrvGraphExecMemcpyNodeSetParams_fn(hGraphExec, hNode,
                                                                        copyParams, ctx));
  CATCH;
}
hipError_t hipSetValidDevices(int* device_arr, int len) {
  auto const __rocm_in_device_arr = device_arr;
  auto const __rocm_in_len = len;
  rocm_trace_emit_hipSetValidDevices_enter(
      (const void*)(uintptr_t)(__rocm_in_device_arr),
      (__rocm_in_len)); /* __ROCM_CURATED__: hipSetValidDevices */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipSetValidDevices, hip::GetHipDispatchTable()->hipSetValidDevices_fn(device_arr, len));
  CATCH;
}
hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset,
                         size_t ByteCount) {
  auto const __rocm_in_dstDevice = dstDevice;
  auto const __rocm_in_srcArray = srcArray;
  auto const __rocm_in_srcOffset = srcOffset;
  auto const __rocm_in_ByteCount = ByteCount;
  rocm_trace_emit_hipMemcpyAtoD_enter(
      (uint64_t)(__rocm_in_dstDevice),
      (const void*)(uintptr_t)(__rocm_in_srcArray),
      (__rocm_in_srcOffset),
      (__rocm_in_ByteCount)); /* __ROCM_CURATED__: hipMemcpyAtoD */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyAtoD, hip::GetHipDispatchTable()->hipMemcpyAtoD_fn(dstDevice, srcArray, srcOffset, ByteCount));
  CATCH;
}
hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice,
                         size_t ByteCount) {
  auto const __rocm_in_dstArray = dstArray;
  auto const __rocm_in_dstOffset = dstOffset;
  auto const __rocm_in_srcDevice = srcDevice;
  auto const __rocm_in_ByteCount = ByteCount;
  rocm_trace_emit_hipMemcpyDtoA_enter(
      (const void*)(uintptr_t)(__rocm_in_dstArray),
      (__rocm_in_dstOffset),
      (uint64_t)(__rocm_in_srcDevice),
      (__rocm_in_ByteCount)); /* __ROCM_CURATED__: hipMemcpyDtoA */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyDtoA, hip::GetHipDispatchTable()->hipMemcpyDtoA_fn(dstArray, dstOffset, srcDevice, ByteCount));
  CATCH;
}
hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray,
                         size_t srcOffset, size_t ByteCount) {
  auto const __rocm_in_dstArray = dstArray;
  auto const __rocm_in_dstOffset = dstOffset;
  auto const __rocm_in_srcArray = srcArray;
  auto const __rocm_in_srcOffset = srcOffset;
  auto const __rocm_in_ByteCount = ByteCount;
  rocm_trace_emit_hipMemcpyAtoA_enter(
      (const void*)(uintptr_t)(__rocm_in_dstArray),
      (__rocm_in_dstOffset),
      (const void*)(uintptr_t)(__rocm_in_srcArray),
      (__rocm_in_srcOffset),
      (__rocm_in_ByteCount)); /* __ROCM_CURATED__: hipMemcpyAtoA */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyAtoA, hip::GetHipDispatchTable()->hipMemcpyAtoA_fn(
                                    dstArray, dstOffset, srcArray, srcOffset, ByteCount));
  CATCH;
}
hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray, size_t srcOffset,
                              size_t ByteCount, hipStream_t stream) {
  auto const __rocm_in_dstHost = dstHost;
  auto const __rocm_in_srcArray = srcArray;
  auto const __rocm_in_srcOffset = srcOffset;
  auto const __rocm_in_ByteCount = ByteCount;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyAtoHAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dstHost),
      (const void*)(uintptr_t)(__rocm_in_srcArray),
      (__rocm_in_srcOffset),
      (__rocm_in_ByteCount),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyAtoHAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyAtoHAsync, hip::GetHipDispatchTable()->hipMemcpyAtoHAsync_fn(
                                    dstHost, srcArray, srcOffset, ByteCount, stream));
  CATCH;
}
hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                              size_t ByteCount, hipStream_t stream) {
  auto const __rocm_in_dstArray = dstArray;
  auto const __rocm_in_dstOffset = dstOffset;
  auto const __rocm_in_srcHost = srcHost;
  auto const __rocm_in_ByteCount = ByteCount;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyHtoAAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dstArray),
      (__rocm_in_dstOffset),
      (const void*)(uintptr_t)(__rocm_in_srcHost),
      (__rocm_in_ByteCount),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyHtoAAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyHtoAAsync, hip::GetHipDispatchTable()->hipMemcpyHtoAAsync_fn(
                                    dstArray, dstOffset, srcHost, ByteCount, stream));
  CATCH;
}
hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                   hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t width, size_t height, hipMemcpyKind kind) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_wOffsetDst = wOffsetDst;
  auto const __rocm_in_hOffsetDst = hOffsetDst;
  auto const __rocm_in_src = src;
  auto const __rocm_in_wOffsetSrc = wOffsetSrc;
  auto const __rocm_in_hOffsetSrc = hOffsetSrc;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_kind = kind;
  rocm_trace_emit_hipMemcpy2DArrayToArray_enter(
      (const void*)(uintptr_t)(__rocm_in_dst),
      (__rocm_in_wOffsetDst),
      (__rocm_in_hOffsetDst),
      (const void*)(uintptr_t)(__rocm_in_src),
      (__rocm_in_wOffsetSrc),
      (__rocm_in_hOffsetSrc),
      (__rocm_in_width),
      (__rocm_in_height),
      (int32_t)(__rocm_in_kind)); /* __ROCM_CURATED__: hipMemcpy2DArrayToArray */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy2DArrayToArray, hip::GetHipDispatchTable()->hipMemcpy2DArrayToArray_fn(
          dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind));
  CATCH;
}
hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D* nodeParams) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipDrvGraphMemcpyNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipDrvGraphMemcpyNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvGraphMemcpyNodeGetParams, hip::GetHipDispatchTable()->hipDrvGraphMemcpyNodeGetParams_fn(hNode, nodeParams));
  CATCH;
}
hipError_t hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t hNode, const HIP_MEMCPY3D* nodeParams) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipDrvGraphMemcpyNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipDrvGraphMemcpyNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvGraphMemcpyNodeSetParams, hip::GetHipDispatchTable()->hipDrvGraphMemcpyNodeSetParams_fn(hNode, nodeParams));
  CATCH;
}
hipError_t hipGraphNodeSetParams(hipGraphNode_t node, hipGraphNodeParams* nodeParams) {
  auto const __rocm_in_node = node;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_node),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphNodeSetParams, hip::GetHipDispatchTable()->hipGraphNodeSetParams_fn(node, nodeParams));
  CATCH;
}
hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     const hipBatchMemOpNodeParams* nodeParams) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_dependencies = dependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphAddBatchMemOpNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_dependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphAddBatchMemOpNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddBatchMemOpNode, hip::GetHipDispatchTable()->hipGraphAddBatchMemOpNode_fn(pGraphNode, graph, dependencies,
                                                               numDependencies, nodeParams));
  CATCH;
}
hipError_t hipGraphBatchMemOpNodeGetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams_out) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams_out = nodeParams_out;
  rocm_trace_emit_hipGraphBatchMemOpNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams_out)); /* __ROCM_CURATED__: hipGraphBatchMemOpNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphBatchMemOpNodeGetParams, hip::GetHipDispatchTable()->hipGraphBatchMemOpNodeGetParams_fn(hNode, nodeParams_out));
  CATCH;
}
hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphBatchMemOpNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphBatchMemOpNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphBatchMemOpNodeSetParams, hip::GetHipDispatchTable()->hipGraphBatchMemOpNodeSetParams_fn(hNode, nodeParams));
  CATCH;
}
hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               const hipBatchMemOpNodeParams* nodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphExecBatchMemOpNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphExecBatchMemOpNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecBatchMemOpNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecBatchMemOpNodeSetParams_fn(
                                    hGraphExec, hNode, nodeParams));
  CATCH;
}
hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags) {
  auto const __rocm_in_event = event;
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipEventRecordWithFlags_enter(
      (uint64_t)(uintptr_t)(__rocm_in_event),
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipEventRecordWithFlags */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipEventRecordWithFlags, hip::GetHipDispatchTable()->hipEventRecordWithFlags_fn(event, stream, flags));
  CATCH;
}

hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, const void* fPtr, void** args) {
  auto const __rocm_in_config = config;
  auto const __rocm_in_fPtr = fPtr;
  auto const __rocm_in_args = args;
  rocm_trace_emit_hipLaunchKernelExC_enter(
      (const void*)(uintptr_t)(__rocm_in_config),
      (const void*)(uintptr_t)(__rocm_in_fPtr),
      (const void*)(uintptr_t)(__rocm_in_args)); /* __ROCM_CURATED__: hipLaunchKernelExC */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLaunchKernelExC, hip::GetHipDispatchTable()->hipLaunchKernelExC_fn(config, fPtr, args));
  CATCH;
}

hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, hipFunction_t f, void** kernel,
                                void** extra) {
  auto const __rocm_in_config = config;
  auto const __rocm_in_f = f;
  auto const __rocm_in_kernel = kernel;
  auto const __rocm_in_extra = extra;
  rocm_trace_emit_hipDrvLaunchKernelEx_enter(
      (const void*)(uintptr_t)(__rocm_in_config),
      (uint64_t)(uintptr_t)(__rocm_in_f),
      (const void*)(uintptr_t)(__rocm_in_kernel),
      (const void*)(uintptr_t)(__rocm_in_extra)); /* __ROCM_CURATED__: hipDrvLaunchKernelEx */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDrvLaunchKernelEx, hip::GetHipDispatchTable()->hipDrvLaunchKernelEx_fn(config, f, kernel, extra));
  CATCH;
}

hipError_t hipMemGetHandleForAddressRange(void* handle, hipDeviceptr_t dptr, size_t size,
                                          hipMemRangeHandleType handleType,
                                          unsigned long long flags) {
  auto const __rocm_in_handle = handle;
  auto const __rocm_in_dptr = dptr;
  auto const __rocm_in_size = size;
  auto const __rocm_in_handleType = handleType;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipMemGetHandleForAddressRange_enter(
      (const void*)(uintptr_t)(__rocm_in_handle),
      (uint64_t)(__rocm_in_dptr),
      (__rocm_in_size),
      (int32_t)(__rocm_in_handleType),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipMemGetHandleForAddressRange */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemGetHandleForAddressRange, hip::GetHipDispatchTable()->hipMemGetHandleForAddressRange_fn(
                                    handle, dptr, size, handleType, flags));
  CATCH;
}
hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch, unsigned char value, size_t width,
                         size_t height) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstPitch = dstPitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  rocm_trace_emit_hipMemsetD2D8_enter(
      (uint64_t)(__rocm_in_dst),
      (__rocm_in_dstPitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height)); /* __ROCM_CURATED__: hipMemsetD2D8 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD2D8, hip::GetHipDispatchTable()->hipMemsetD2D8_fn(dst, dstPitch, value, width, height));
  CATCH;
}
hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
                              size_t width, size_t height, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstPitch = dstPitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetD2D8Async_enter(
      (uint64_t)(__rocm_in_dst),
      (__rocm_in_dstPitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetD2D8Async */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD2D8Async, hip::GetHipDispatchTable()->hipMemsetD2D8Async_fn(
                                    dst, dstPitch, value, width, height, stream));
  CATCH;
}
hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch, unsigned short value, size_t width,
                          size_t height) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstPitch = dstPitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  rocm_trace_emit_hipMemsetD2D16_enter(
      (uint64_t)(__rocm_in_dst),
      (__rocm_in_dstPitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height)); /* __ROCM_CURATED__: hipMemsetD2D16 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD2D16, hip::GetHipDispatchTable()->hipMemsetD2D16_fn(dst, dstPitch, value, width, height));
  CATCH;
}
hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
                               size_t width, size_t height, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstPitch = dstPitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetD2D16Async_enter(
      (uint64_t)(__rocm_in_dst),
      (__rocm_in_dstPitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetD2D16Async */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD2D16Async, hip::GetHipDispatchTable()->hipMemsetD2D16Async_fn(
                                    dst, dstPitch, value, width, height, stream));
  CATCH;
}
hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width,
                          size_t height) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstPitch = dstPitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  rocm_trace_emit_hipMemsetD2D32_enter(
      (uint64_t)(__rocm_in_dst),
      (__rocm_in_dstPitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height)); /* __ROCM_CURATED__: hipMemsetD2D32 */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD2D32, hip::GetHipDispatchTable()->hipMemsetD2D32_fn(dst, dstPitch, value, width, height));
  CATCH;
}
hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch, unsigned int value,
                               size_t width, size_t height, hipStream_t stream) {
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dstPitch = dstPitch;
  auto const __rocm_in_value = value;
  auto const __rocm_in_width = width;
  auto const __rocm_in_height = height;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemsetD2D32Async_enter(
      (uint64_t)(__rocm_in_dst),
      (__rocm_in_dstPitch),
      (__rocm_in_value),
      (__rocm_in_width),
      (__rocm_in_height),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemsetD2D32Async */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemsetD2D32Async, hip::GetHipDispatchTable()->hipMemsetD2D32Async_fn(
                                    dst, dstPitch, value, width, height, stream));
  CATCH;
}
hipError_t hipStreamSetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 const hipStreamAttrValue* value) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipStreamSetAttribute_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipStreamSetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamSetAttribute, hip::GetHipDispatchTable()->hipStreamSetAttribute_fn(stream, attr, value));
  CATCH;
}
hipError_t hipStreamGetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 hipStreamAttrValue* value) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_attr = attr;
  auto const __rocm_in_value = value;
  rocm_trace_emit_hipStreamGetAttribute_enter(
      (uint64_t)(uintptr_t)(__rocm_in_stream),
      (int32_t)(__rocm_in_attr),
      (const void*)(uintptr_t)(__rocm_in_value)); /* __ROCM_CURATED__: hipStreamGetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetAttribute, hip::GetHipDispatchTable()->hipStreamGetAttribute_fn(stream, attr, value));
  CATCH;
}
hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count,
                               hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                               size_t* failIdx, hipStream_t stream) {
  auto const __rocm_in_dsts = dsts;
  auto const __rocm_in_srcs = srcs;
  auto const __rocm_in_sizes = sizes;
  auto const __rocm_in_count = count;
  auto const __rocm_in_attrs = attrs;
  auto const __rocm_in_attrsIdxs = attrsIdxs;
  auto const __rocm_in_numAttrs = numAttrs;
  auto const __rocm_in_failIdx = failIdx;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpyBatchAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_dsts),
      (const void*)(uintptr_t)(__rocm_in_srcs),
      (const void*)(uintptr_t)(__rocm_in_sizes),
      (__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_attrs),
      (const void*)(uintptr_t)(__rocm_in_attrsIdxs),
      (__rocm_in_numAttrs),
      (const void*)(uintptr_t)(__rocm_in_failIdx),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpyBatchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpyBatchAsync, hip::GetHipDispatchTable()->hipMemcpyBatchAsync_fn(dsts, srcs, sizes, count, attrs, attrsIdxs,
                                                         numAttrs, failIdx, stream));
  CATCH;
}
hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp* opList, size_t* failIdx,
                                 unsigned long long flags, hipStream_t stream) {
  auto const __rocm_in_numOps = numOps;
  auto const __rocm_in_opList = opList;
  auto const __rocm_in_failIdx = failIdx;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy3DBatchAsync_enter(
      (__rocm_in_numOps),
      (const void*)(uintptr_t)(__rocm_in_opList),
      (const void*)(uintptr_t)(__rocm_in_failIdx),
      (__rocm_in_flags),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy3DBatchAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy3DBatchAsync, hip::GetHipDispatchTable()->hipMemcpy3DBatchAsync_fn(numOps, opList, failIdx, flags, stream));
  CATCH;
}
hipError_t hipMemcpy3DPeer(hipMemcpy3DPeerParms* p) {
  auto const __rocm_in_p = p;
  rocm_trace_emit_hipMemcpy3DPeer_enter(
      (const void*)(uintptr_t)(__rocm_in_p)); /* __ROCM_CURATED__: hipMemcpy3DPeer */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy3DPeer, hip::GetHipDispatchTable()->hipMemcpy3DPeer_fn(p));
  CATCH;
}
hipError_t hipMemcpy3DPeerAsync(hipMemcpy3DPeerParms* p, hipStream_t stream) {
  auto const __rocm_in_p = p;
  auto const __rocm_in_stream = stream;
  rocm_trace_emit_hipMemcpy3DPeerAsync_enter(
      (const void*)(uintptr_t)(__rocm_in_p),
      (uint64_t)(uintptr_t)(__rocm_in_stream)); /* __ROCM_CURATED__: hipMemcpy3DPeerAsync */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemcpy3DPeerAsync, hip::GetHipDispatchTable()->hipMemcpy3DPeerAsync_fn(p, stream));
  CATCH;
}
hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements,
                                               const hipChannelFormatDesc* fmtDesc, int device) {
  auto const __rocm_in_maxWidthInElements = maxWidthInElements;
  auto const __rocm_in_fmtDesc = fmtDesc;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGetTexture1DLinearMaxWidth_enter(
      (const void*)(uintptr_t)(__rocm_in_maxWidthInElements),
      (const void*)(uintptr_t)(__rocm_in_fmtDesc),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGetTexture1DLinearMaxWidth */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetTexture1DLinearMaxWidth, hip::GetHipDispatchTable()->hipDeviceGetTexture1DLinearMaxWidth_fn(
                                    maxWidthInElements, fmtDesc, device));
  CATCH;
}
hipError_t hipGraphAddExternalSemaphoresSignalNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphAddExternalSemaphoresSignalNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphAddExternalSemaphoresSignalNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddExternalSemaphoresSignalNode, hip::GetHipDispatchTable()->hipGraphAddExternalSemaphoresSignalNode_fn(
          pGraphNode, graph, pDependencies, numDependencies, nodeParams));
  CATCH;
}
hipError_t hipGraphAddExternalSemaphoresWaitNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  auto const __rocm_in_pGraphNode = pGraphNode;
  auto const __rocm_in_graph = graph;
  auto const __rocm_in_pDependencies = pDependencies;
  auto const __rocm_in_numDependencies = numDependencies;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphAddExternalSemaphoresWaitNode_enter(
      (const void*)(uintptr_t)(__rocm_in_pGraphNode),
      (uint64_t)(uintptr_t)(__rocm_in_graph),
      (const void*)(uintptr_t)(__rocm_in_pDependencies),
      (__rocm_in_numDependencies),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphAddExternalSemaphoresWaitNode */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphAddExternalSemaphoresWaitNode, hip::GetHipDispatchTable()->hipGraphAddExternalSemaphoresWaitNode_fn(
          pGraphNode, graph, pDependencies, numDependencies, nodeParams));
  CATCH;
}
hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphExternalSemaphoresSignalNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphExternalSemaphoresSignalNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExternalSemaphoresSignalNodeSetParams, hip::GetHipDispatchTable()->hipGraphExternalSemaphoresSignalNodeSetParams_fn(hNode,
                                                                                   nodeParams));
  CATCH;
}
hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_params_out = params_out;
  rocm_trace_emit_hipGraphExternalSemaphoresSignalNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_params_out)); /* __ROCM_CURATED__: hipGraphExternalSemaphoresSignalNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExternalSemaphoresSignalNodeGetParams, hip::GetHipDispatchTable()->hipGraphExternalSemaphoresSignalNodeGetParams_fn(hNode,
                                                                                   params_out));
  CATCH;
}
hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_params_out = params_out;
  rocm_trace_emit_hipGraphExternalSemaphoresWaitNodeGetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_params_out)); /* __ROCM_CURATED__: hipGraphExternalSemaphoresWaitNodeGetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExternalSemaphoresWaitNodeGetParams, hip::GetHipDispatchTable()->hipGraphExternalSemaphoresWaitNodeGetParams_fn(hNode, params_out));
  CATCH;
}
hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphExternalSemaphoresWaitNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphExternalSemaphoresWaitNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExternalSemaphoresWaitNodeSetParams, hip::GetHipDispatchTable()->hipGraphExternalSemaphoresWaitNodeSetParams_fn(hNode, nodeParams));
  CATCH;
}
hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphExecExternalSemaphoresSignalNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphExecExternalSemaphoresSignalNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecExternalSemaphoresSignalNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecExternalSemaphoresSignalNodeSetParams_fn(
          hGraphExec, hNode, nodeParams));
  CATCH;
}
hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  auto const __rocm_in_hGraphExec = hGraphExec;
  auto const __rocm_in_hNode = hNode;
  auto const __rocm_in_nodeParams = nodeParams;
  rocm_trace_emit_hipGraphExecExternalSemaphoresWaitNodeSetParams_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hGraphExec),
      (uint64_t)(uintptr_t)(__rocm_in_hNode),
      (const void*)(uintptr_t)(__rocm_in_nodeParams)); /* __ROCM_CURATED__: hipGraphExecExternalSemaphoresWaitNodeSetParams */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGraphExecExternalSemaphoresWaitNodeSetParams, hip::GetHipDispatchTable()->hipGraphExecExternalSemaphoresWaitNodeSetParams_fn(
          hGraphExec, hNode, nodeParams));
  CATCH;
}
hipError_t hipLibraryLoadData(hipLibrary_t* library, const void* code, hipJitOption* jitOptions,
                              void** jitOptionsValues, unsigned int numJitOptions,
                              hipLibraryOption* libraryOptions, void** libraryOptionValues,
                              unsigned int numLibraryOptions) {
  auto const __rocm_in_library = library;
  auto const __rocm_in_code = code;
  auto const __rocm_in_jitOptions = jitOptions;
  auto const __rocm_in_jitOptionsValues = jitOptionsValues;
  auto const __rocm_in_numJitOptions = numJitOptions;
  auto const __rocm_in_libraryOptions = libraryOptions;
  auto const __rocm_in_libraryOptionValues = libraryOptionValues;
  auto const __rocm_in_numLibraryOptions = numLibraryOptions;
  rocm_trace_emit_hipLibraryLoadData_enter(
      (const void*)(uintptr_t)(__rocm_in_library),
      (const void*)(uintptr_t)(__rocm_in_code),
      (const void*)(uintptr_t)(__rocm_in_jitOptions),
      (const void*)(uintptr_t)(__rocm_in_jitOptionsValues),
      (__rocm_in_numJitOptions),
      (const void*)(uintptr_t)(__rocm_in_libraryOptions),
      (const void*)(uintptr_t)(__rocm_in_libraryOptionValues),
      (__rocm_in_numLibraryOptions)); /* __ROCM_CURATED__: hipLibraryLoadData */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryLoadData, hip::GetHipDispatchTable()->hipLibraryLoadData_fn(library, code, jitOptions, jitOptionsValues,
                                                        numJitOptions, libraryOptions,
                                                        libraryOptionValues, numLibraryOptions));
  CATCH;
}
hipError_t hipLibraryLoadFromFile(hipLibrary_t* library, const char* fileName,
                                  hipJitOption* jitOptions, void** jitOptionsValues,
                                  unsigned int numJitOptions, hipLibraryOption* libraryOptions,
                                  void** libraryOptionValues, unsigned int numLibraryOptions) {
  auto const __rocm_in_library = library;
  auto const __rocm_in_fileName = fileName;
  auto const __rocm_in_jitOptions = jitOptions;
  auto const __rocm_in_jitOptionsValues = jitOptionsValues;
  auto const __rocm_in_numJitOptions = numJitOptions;
  auto const __rocm_in_libraryOptions = libraryOptions;
  auto const __rocm_in_libraryOptionValues = libraryOptionValues;
  auto const __rocm_in_numLibraryOptions = numLibraryOptions;
  rocm_trace_emit_hipLibraryLoadFromFile_enter(
      (const void*)(uintptr_t)(__rocm_in_library),
      (const char*)(__rocm_in_fileName),
      (const void*)(uintptr_t)(__rocm_in_jitOptions),
      (const void*)(uintptr_t)(__rocm_in_jitOptionsValues),
      (__rocm_in_numJitOptions),
      (const void*)(uintptr_t)(__rocm_in_libraryOptions),
      (const void*)(uintptr_t)(__rocm_in_libraryOptionValues),
      (__rocm_in_numLibraryOptions)); /* __ROCM_CURATED__: hipLibraryLoadFromFile */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryLoadFromFile, hip::GetHipDispatchTable()->hipLibraryLoadFromFile_fn(
          library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
          libraryOptionValues, numLibraryOptions));
  CATCH;
}
hipError_t hipLibraryUnload(hipLibrary_t library) {
  auto const __rocm_in_library = library;
  rocm_trace_emit_hipLibraryUnload_enter(
      (const void*)(uintptr_t)(__rocm_in_library)); /* __ROCM_CURATED__: hipLibraryUnload */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryUnload, hip::GetHipDispatchTable()->hipLibraryUnload_fn(library));
  CATCH;
}
hipError_t hipLibraryGetKernel(hipKernel_t* pKernel, hipLibrary_t library, const char* name) {
  auto const __rocm_in_pKernel = pKernel;
  auto const __rocm_in_library = library;
  auto const __rocm_in_name = name;
  rocm_trace_emit_hipLibraryGetKernel_enter(
      (const void*)(uintptr_t)(__rocm_in_pKernel),
      (const void*)(uintptr_t)(__rocm_in_library),
      (const char*)(__rocm_in_name)); /* __ROCM_CURATED__: hipLibraryGetKernel */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryGetKernel, hip::GetHipDispatchTable()->hipLibraryGetKernel_fn(pKernel, library, name));
  CATCH;
}
hipError_t hipLibraryGetKernelCount(unsigned int* count, hipLibrary_t library) {
  auto const __rocm_in_count = count;
  auto const __rocm_in_library = library;
  rocm_trace_emit_hipLibraryGetKernelCount_enter(
      (const void*)(uintptr_t)(__rocm_in_count),
      (const void*)(uintptr_t)(__rocm_in_library)); /* __ROCM_CURATED__: hipLibraryGetKernelCount */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryGetKernelCount, hip::GetHipDispatchTable()->hipLibraryGetKernelCount_fn(count, library));
  CATCH;
}
hipError_t hipLibraryGetGlobal(void** dptr, size_t* bytes, hipLibrary_t library, const char* name) {
  auto const __rocm_in_dptr = dptr;
  auto const __rocm_in_bytes = bytes;
  auto const __rocm_in_library = library;
  auto const __rocm_in_name = name;
  rocm_trace_emit_hipLibraryGetGlobal_enter(
      (const void*)(uintptr_t)(__rocm_in_dptr),
      (const void*)(uintptr_t)(__rocm_in_bytes),
      (const void*)(uintptr_t)(__rocm_in_library),
      (const char*)(__rocm_in_name)); /* __ROCM_CURATED__: hipLibraryGetGlobal */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryGetGlobal, hip::GetHipDispatchTable()->hipLibraryGetGlobal_fn(dptr, bytes, library, name));
  CATCH;
}
hipError_t hipLibraryGetManaged(void** dptr, size_t* bytes, hipLibrary_t library,
                                const char* name) {
  auto const __rocm_in_dptr = dptr;
  auto const __rocm_in_bytes = bytes;
  auto const __rocm_in_library = library;
  auto const __rocm_in_name = name;
  rocm_trace_emit_hipLibraryGetManaged_enter(
      (const void*)(uintptr_t)(__rocm_in_dptr),
      (const void*)(uintptr_t)(__rocm_in_bytes),
      (const void*)(uintptr_t)(__rocm_in_library),
      (const char*)(__rocm_in_name)); /* __ROCM_CURATED__: hipLibraryGetManaged */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryGetManaged, hip::GetHipDispatchTable()->hipLibraryGetManaged_fn(dptr, bytes, library, name));
  CATCH;
}
hipError_t hipKernelGetAttribute(int* pi, hipFunction_attribute attrib, hipKernel_t kernel,
                                 hipDevice_t dev) {
  auto const __rocm_in_pi = pi;
  auto const __rocm_in_attrib = attrib;
  auto const __rocm_in_kernel = kernel;
  auto const __rocm_in_dev = dev;
  rocm_trace_emit_hipKernelGetAttribute_enter(
      (const void*)(uintptr_t)(__rocm_in_pi),
      (int32_t)(__rocm_in_attrib),
      (const void*)(uintptr_t)(__rocm_in_kernel),
      (__rocm_in_dev)); /* __ROCM_CURATED__: hipKernelGetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipKernelGetAttribute, hip::GetHipDispatchTable()->hipKernelGetAttribute_fn(pi, attrib, kernel, dev));
  CATCH;
}
hipError_t hipLibraryEnumerateKernels(hipKernel_t* kernels, unsigned int numKernels,
                                      hipLibrary_t library) {
  auto const __rocm_in_kernels = kernels;
  auto const __rocm_in_numKernels = numKernels;
  auto const __rocm_in_library = library;
  rocm_trace_emit_hipLibraryEnumerateKernels_enter(
      (const void*)(uintptr_t)(__rocm_in_kernels),
      (__rocm_in_numKernels),
      (const void*)(uintptr_t)(__rocm_in_library)); /* __ROCM_CURATED__: hipLibraryEnumerateKernels */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipLibraryEnumerateKernels, hip::GetHipDispatchTable()->hipLibraryEnumerateKernels_fn(kernels, numKernels, library));
  CATCH;
}
hipError_t hipKernelGetLibrary(hipLibrary_t* library, hipKernel_t kernel) {
  auto const __rocm_in_library = library;
  auto const __rocm_in_kernel = kernel;
  rocm_trace_emit_hipKernelGetLibrary_enter(
      (const void*)(uintptr_t)(__rocm_in_library),
      (const void*)(uintptr_t)(__rocm_in_kernel)); /* __ROCM_CURATED__: hipKernelGetLibrary */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipKernelGetLibrary, hip::GetHipDispatchTable()->hipKernelGetLibrary_fn(library, kernel));
  CATCH;
}
hipError_t hipKernelGetName(const char** name, hipKernel_t kernel) {
  auto const __rocm_in_name = name;
  auto const __rocm_in_kernel = kernel;
  rocm_trace_emit_hipKernelGetName_enter(
      (const void*)(uintptr_t)(__rocm_in_name),
      (const void*)(uintptr_t)(__rocm_in_kernel)); /* __ROCM_CURATED__: hipKernelGetName */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipKernelGetName, hip::GetHipDispatchTable()->hipKernelGetName_fn(name, kernel));
  CATCH;
}
hipError_t hipKernelGetParamInfo(hipKernel_t kernel, size_t paramIndex, size_t* paramOffset,
                                 size_t* paramSize) {
  auto const __rocm_in_kernel = kernel;
  auto const __rocm_in_paramIndex = paramIndex;
  auto const __rocm_in_paramOffset = paramOffset;
  auto const __rocm_in_paramSize = paramSize;
  rocm_trace_emit_hipKernelGetParamInfo_enter(
      (const void*)(uintptr_t)(__rocm_in_kernel),
      (__rocm_in_paramIndex),
      (const void*)(uintptr_t)(__rocm_in_paramOffset),
      (const void*)(uintptr_t)(__rocm_in_paramSize)); /* __ROCM_CURATED__: hipKernelGetParamInfo */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipKernelGetParamInfo, hip::GetHipDispatchTable()->hipKernelGetParamInfo_fn(
                                    kernel, paramIndex, paramOffset, paramSize));
  CATCH;
}
hipError_t hipExtEnableLogging() {
  TRY;
  return hip::GetHipDispatchTable()->hipExtEnableLogging_fn();
  CATCH;
}
hipError_t hipExtDisableLogging() {
  TRY;
  return hip::GetHipDispatchTable()->hipExtDisableLogging_fn();
  CATCH;
}
hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size, size_t log_mask) {
  TRY;
  return hip::GetHipDispatchTable()->hipExtSetLoggingParams_fn(log_level, log_size, log_mask);
  CATCH;
}
hipError_t hipMemSetMemPool(hipMemLocation* location, hipMemAllocationType type,
                            hipMemPool_t pool) {
  auto const __rocm_in_location = location;
  auto const __rocm_in_type = type;
  auto const __rocm_in_pool = pool;
  rocm_trace_emit_hipMemSetMemPool_enter(
      (const void*)(uintptr_t)(__rocm_in_location),
      (int32_t)(__rocm_in_type),
      (const void*)(uintptr_t)(__rocm_in_pool)); /* __ROCM_CURATED__: hipMemSetMemPool */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemSetMemPool, hip::GetHipDispatchTable()->hipMemSetMemPool_fn(location, type, pool));
  CATCH;
}
hipError_t hipMemGetMemPool(hipMemPool_t* pool, hipMemLocation* location,
                            hipMemAllocationType type) {
  auto const __rocm_in_pool = pool;
  auto const __rocm_in_location = location;
  auto const __rocm_in_type = type;
  rocm_trace_emit_hipMemGetMemPool_enter(
      (const void*)(uintptr_t)(__rocm_in_pool),
      (const void*)(uintptr_t)(__rocm_in_location),
      (int32_t)(__rocm_in_type)); /* __ROCM_CURATED__: hipMemGetMemPool */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMemGetMemPool, hip::GetHipDispatchTable()->hipMemGetMemPool_fn(pool, location, type));
  CATCH;
}
hipError_t hipMipmappedArrayGetMemoryRequirements(hipArrayMemoryRequirements* memoryRequirements,
                                                  hipMipmappedArray_t mipmap, hipDevice_t device) {
  auto const __rocm_in_memoryRequirements = memoryRequirements;
  auto const __rocm_in_mipmap = mipmap;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipMipmappedArrayGetMemoryRequirements_enter(
      (const void*)(uintptr_t)(__rocm_in_memoryRequirements),
      (const void*)(uintptr_t)(__rocm_in_mipmap),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipMipmappedArrayGetMemoryRequirements */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipMipmappedArrayGetMemoryRequirements, hip::GetHipDispatchTable()->hipMipmappedArrayGetMemoryRequirements_fn(memoryRequirements,
                                                                            mipmap, device));
  CATCH;
}
hipError_t hipKernelSetAttribute(hipFunction_attribute attrib, int value, hipKernel_t kernel,
                                 hipDevice_t dev) {
  auto const __rocm_in_attrib = attrib;
  auto const __rocm_in_value = value;
  auto const __rocm_in_kernel = kernel;
  auto const __rocm_in_dev = dev;
  rocm_trace_emit_hipKernelSetAttribute_enter(
      (int32_t)(__rocm_in_attrib),
      (__rocm_in_value),
      (const void*)(uintptr_t)(__rocm_in_kernel),
      (__rocm_in_dev)); /* __ROCM_CURATED__: hipKernelSetAttribute */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipKernelSetAttribute, hip::GetHipDispatchTable()->hipKernelSetAttribute_fn(attrib, value, kernel, dev));
  CATCH;
}
hipError_t hipKernelGetFunction(hipFunction_t* pFunc, hipKernel_t kernel) {
  auto const __rocm_in_pFunc = pFunc;
  auto const __rocm_in_kernel = kernel;
  rocm_trace_emit_hipKernelGetFunction_enter(
      (const void*)(uintptr_t)(__rocm_in_pFunc),
      (const void*)(uintptr_t)(__rocm_in_kernel)); /* __ROCM_CURATED__: hipKernelGetFunction */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipKernelGetFunction, hip::GetHipDispatchTable()->hipKernelGetFunction_fn(pFunc, kernel));
  CATCH;
}
hipError_t hipGreenCtxCreate(hipExecutionCtx_t* ctx, hipDevResourceDesc_t desc, int device,
                             unsigned int flags) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_desc = desc;
  auto const __rocm_in_device = device;
  auto const __rocm_in_flags = flags;
  rocm_trace_emit_hipGreenCtxCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (const void*)(uintptr_t)(__rocm_in_desc),
      (__rocm_in_device),
      (__rocm_in_flags)); /* __ROCM_CURATED__: hipGreenCtxCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipGreenCtxCreate, hip::GetHipDispatchTable()->hipGreenCtxCreate_fn(ctx, desc, device, flags));
  CATCH;
}
hipError_t hipExecutionCtxDestroy(hipExecutionCtx_t ctx) {
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipExecutionCtxDestroy_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipExecutionCtxDestroy */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxDestroy, hip::GetHipDispatchTable()->hipExecutionCtxDestroy_fn(ctx));
  CATCH;
}
hipError_t hipExecutionCtxStreamCreate(hipStream_t* stream, hipExecutionCtx_t greenctx,
                                       unsigned int flags, int priority) {
  auto const __rocm_in_stream = stream;
  auto const __rocm_in_greenctx = greenctx;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_priority = priority;
  rocm_trace_emit_hipExecutionCtxStreamCreate_enter(
      (const void*)(uintptr_t)(__rocm_in_stream),
      (const void*)(uintptr_t)(__rocm_in_greenctx),
      (__rocm_in_flags),
      (__rocm_in_priority)); /* __ROCM_CURATED__: hipExecutionCtxStreamCreate */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxStreamCreate, hip::GetHipDispatchTable()->hipExecutionCtxStreamCreate_fn(stream, greenctx, flags, priority));
  CATCH;
}
hipError_t hipDeviceGetDevResource(hipDevice_t device, hipDevResource* resource,
                                   hipDevResourceType type) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_resource = resource;
  auto const __rocm_in_type = type;
  rocm_trace_emit_hipDeviceGetDevResource_enter(
      (__rocm_in_device),
      (const void*)(uintptr_t)(__rocm_in_resource),
      (int32_t)(__rocm_in_type)); /* __ROCM_CURATED__: hipDeviceGetDevResource */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetDevResource, hip::GetHipDispatchTable()->hipDeviceGetDevResource_fn(device, resource, type));
  CATCH;
}
hipError_t hipDevSmResourceSplitByCount(hipDevResource* result, unsigned int* nbGroups,
                                        const hipDevResource* input, hipDevResource* remainder,
                                        unsigned int flags, unsigned int minCount) {
  auto const __rocm_in_result = result;
  auto const __rocm_in_nbGroups = nbGroups;
  auto const __rocm_in_input = input;
  auto const __rocm_in_remainder = remainder;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_minCount = minCount;
  rocm_trace_emit_hipDevSmResourceSplitByCount_enter(
      (const void*)(uintptr_t)(__rocm_in_result),
      (const void*)(uintptr_t)(__rocm_in_nbGroups),
      (const void*)(uintptr_t)(__rocm_in_input),
      (const void*)(uintptr_t)(__rocm_in_remainder),
      (__rocm_in_flags),
      (__rocm_in_minCount)); /* __ROCM_CURATED__: hipDevSmResourceSplitByCount */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevSmResourceSplitByCount, hip::GetHipDispatchTable()->hipDevSmResourceSplitByCount_fn(result, nbGroups, input,
                                                                  remainder, flags, minCount));
  CATCH;
}
hipError_t hipDevSmResourceSplit(hipDevResource* result, unsigned int nbGroups,
                                 const hipDevResource* input, hipDevResource* remainder,
                                 unsigned int flags, hipDevSmResourceGroupParams* groupParams) {
  auto const __rocm_in_result = result;
  auto const __rocm_in_nbGroups = nbGroups;
  auto const __rocm_in_input = input;
  auto const __rocm_in_remainder = remainder;
  auto const __rocm_in_flags = flags;
  auto const __rocm_in_groupParams = groupParams;
  rocm_trace_emit_hipDevSmResourceSplit_enter(
      (const void*)(uintptr_t)(__rocm_in_result),
      (__rocm_in_nbGroups),
      (const void*)(uintptr_t)(__rocm_in_input),
      (const void*)(uintptr_t)(__rocm_in_remainder),
      (__rocm_in_flags),
      (const void*)(uintptr_t)(__rocm_in_groupParams)); /* __ROCM_CURATED__: hipDevSmResourceSplit */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevSmResourceSplit, hip::GetHipDispatchTable()->hipDevSmResourceSplit_fn(
                                    result, nbGroups, input, remainder, flags, groupParams));
  CATCH;
}
hipError_t hipDevResourceGenerateDesc(hipDevResourceDesc_t* phDesc, hipDevResource* resources,
                                      unsigned int nbResources) {
  auto const __rocm_in_phDesc = phDesc;
  auto const __rocm_in_resources = resources;
  auto const __rocm_in_nbResources = nbResources;
  rocm_trace_emit_hipDevResourceGenerateDesc_enter(
      (const void*)(uintptr_t)(__rocm_in_phDesc),
      (const void*)(uintptr_t)(__rocm_in_resources),
      (__rocm_in_nbResources)); /* __ROCM_CURATED__: hipDevResourceGenerateDesc */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDevResourceGenerateDesc, hip::GetHipDispatchTable()->hipDevResourceGenerateDesc_fn(phDesc, resources, nbResources));
  CATCH;
}
hipError_t hipDeviceGetExecutionCtx(hipExecutionCtx_t* ctx, int device) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_device = device;
  rocm_trace_emit_hipDeviceGetExecutionCtx_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (__rocm_in_device)); /* __ROCM_CURATED__: hipDeviceGetExecutionCtx */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceGetExecutionCtx, hip::GetHipDispatchTable()->hipDeviceGetExecutionCtx_fn(ctx, device));
  CATCH;
}
hipError_t hipExecutionCtxGetDevResource(hipExecutionCtx_t ctx, hipDevResource* resource,
                                         hipDevResourceType type) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_resource = resource;
  auto const __rocm_in_type = type;
  rocm_trace_emit_hipExecutionCtxGetDevResource_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (const void*)(uintptr_t)(__rocm_in_resource),
      (int32_t)(__rocm_in_type)); /* __ROCM_CURATED__: hipExecutionCtxGetDevResource */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxGetDevResource, hip::GetHipDispatchTable()->hipExecutionCtxGetDevResource_fn(ctx, resource, type));
  CATCH;
}
hipError_t hipExecutionCtxGetDevice(int* device, hipExecutionCtx_t ctx) {
  auto const __rocm_in_device = device;
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipExecutionCtxGetDevice_enter(
      (const void*)(uintptr_t)(__rocm_in_device),
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipExecutionCtxGetDevice */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxGetDevice, hip::GetHipDispatchTable()->hipExecutionCtxGetDevice_fn(device, ctx));
  CATCH;
}
hipError_t hipExecutionCtxGetId(hipExecutionCtx_t ctx, unsigned long long* ctxId) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_ctxId = ctxId;
  rocm_trace_emit_hipExecutionCtxGetId_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (const void*)(uintptr_t)(__rocm_in_ctxId)); /* __ROCM_CURATED__: hipExecutionCtxGetId */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxGetId, hip::GetHipDispatchTable()->hipExecutionCtxGetId_fn(ctx, ctxId));
  CATCH;
}
hipError_t hipStreamGetDevResource(hipStream_t hStream, hipDevResource* resource,
                                   hipDevResourceType type) {
  auto const __rocm_in_hStream = hStream;
  auto const __rocm_in_resource = resource;
  auto const __rocm_in_type = type;
  rocm_trace_emit_hipStreamGetDevResource_enter(
      (uint64_t)(uintptr_t)(__rocm_in_hStream),
      (const void*)(uintptr_t)(__rocm_in_resource),
      (int32_t)(__rocm_in_type)); /* __ROCM_CURATED__: hipStreamGetDevResource */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipStreamGetDevResource, hip::GetHipDispatchTable()->hipStreamGetDevResource_fn(hStream, resource, type));
  CATCH;
}
hipError_t hipExecutionCtxRecordEvent(hipExecutionCtx_t ctx, hipEvent_t event) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipExecutionCtxRecordEvent_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipExecutionCtxRecordEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxRecordEvent, hip::GetHipDispatchTable()->hipExecutionCtxRecordEvent_fn(ctx, event));
  CATCH;
}
hipError_t hipExecutionCtxSynchronize(hipExecutionCtx_t ctx) {
  auto const __rocm_in_ctx = ctx;
  rocm_trace_emit_hipExecutionCtxSynchronize_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx)); /* __ROCM_CURATED__: hipExecutionCtxSynchronize */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxSynchronize, hip::GetHipDispatchTable()->hipExecutionCtxSynchronize_fn(ctx));
  CATCH;
}
hipError_t hipExecutionCtxWaitEvent(hipExecutionCtx_t ctx, hipEvent_t event) {
  auto const __rocm_in_ctx = ctx;
  auto const __rocm_in_event = event;
  rocm_trace_emit_hipExecutionCtxWaitEvent_enter(
      (const void*)(uintptr_t)(__rocm_in_ctx),
      (uint64_t)(uintptr_t)(__rocm_in_event)); /* __ROCM_CURATED__: hipExecutionCtxWaitEvent */
  TRY;
  ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipExecutionCtxWaitEvent, hip::GetHipDispatchTable()->hipExecutionCtxWaitEvent_fn(ctx, event));
  CATCH;
}
hipError_t hipMemGetDefaultMemPool(hipMemPool_t* memPool, hipMemLocation* location,
                                   hipMemAllocationType type) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetDefaultMemPool_fn(memPool, location, type);
  CATCH;
}
