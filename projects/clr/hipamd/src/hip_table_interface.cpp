/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// This is the entry point of all APIs we need to support. This file needs to
// guarantee ABI stability so we need to see all versions.
#include <hip/amd_detail/hip_api_trace.hpp>
#include <hip/hip_deprecated.h> // For hipDeviceProp_tR0000
#include "hip_internal.hpp"
#include "hip_dispatch_load.hpp"
#include "utils/flags.hpp"
#include "utils/debug.hpp"
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
template <typename T>
__forceinline T HandleException() {
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
#define CATCH } catch(...) { HIP_RETURN(hip::HandleException<hipError_t>()); }
#define CATCHRET(RETURN_TYPE) } catch(...) { return hip::HandleException<RETURN_TYPE>(); }

// Tools write the live dispatch tables while other threads call through them, so every
// entry point loads its slot atomically (see hip_dispatch_load.hpp).
#define HIP_LOAD_DISPATCH_TABLE_FN(fn) hip::LoadDispatchSlot(hip::GetHipDispatchTable()->fn)
#define HIP_LOAD_COMPILER_TABLE_FN(fn) hip::LoadDispatchSlot(hip::GetHipCompilerDispatchTable()->fn)

HIP_PUBLIC_API hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                                hipStream_t* stream) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipPopCallConfiguration_fn)(gridDim, blockDim, sharedMem,
                                                                  stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                                 hipStream_t stream) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipPushCallConfiguration_fn)(gridDim, blockDim, sharedMem,
                                                                   stream);
  CATCH;
}
HIP_PUBLIC_API void** __hipRegisterFatBinary(const void* data) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipRegisterFatBinary_fn)(data);
  CATCHRET(void**);
}
HIP_PUBLIC_API void __hipRegisterFunction(void** modules, const void* hostFunction,
                                      char* deviceFunction, const char* deviceName,
                                      unsigned int threadLimit, uint3* tid, uint3* bid,
                                      dim3* blockDim, dim3* gridDim, int* wSize) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipRegisterFunction_fn)(
      modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim,
      wSize);
  CATCHRET(void);
}
HIP_PUBLIC_API void __hipRegisterManagedVar(void* hipModule, void** pointer, void* init_value,
                                        const char* name, size_t size, unsigned align) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipRegisterManagedVar_fn)(
      hipModule, pointer, init_value, name, size, align);
  CATCHRET(void);
}
HIP_PUBLIC_API void __hipRegisterSurface(void** modules, void* var, char* hostVar, char* deviceVar,
                                     int type, int ext) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipRegisterSurface_fn)(modules, var, hostVar, deviceVar,
                                                             type, ext);
  CATCHRET(void);
}
HIP_PUBLIC_API void __hipRegisterTexture(void** modules, void* var, char* hostVar, char* deviceVar,
                                     int type, int norm, int ext) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipRegisterTexture_fn)(modules, var, hostVar, deviceVar,
                                                             type, norm, ext);
  CATCHRET(void);
}
HIP_PUBLIC_API void __hipRegisterVar(void** modules, void* var, char* hostVar, char* deviceVar, int ext,
                                 size_t size, int constant, int global) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipRegisterVar_fn)(modules, var, hostVar, deviceVar, ext,
                                                         size, constant, global);
  CATCHRET(void);
}
HIP_PUBLIC_API void __hipUnregisterFatBinary(void** modules) {
  TRY;
  return HIP_LOAD_COMPILER_TABLE_FN(__hipUnregisterFatBinary_fn)(modules);
  CATCHRET(void);
}
HIP_PUBLIC_API const char* hipApiName(uint32_t id) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipApiName_fn)(id);
  CATCHRET(const char*);
}
HIP_PUBLIC_API hipError_t hipArray3DCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipArray3DCreate_fn)(array, pAllocateArray);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipArray3DGetDescriptor_fn)(pArrayDescriptor, array);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipArrayCreate(hipArray_t* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipArrayCreate_fn)(pHandle, pAllocateArray);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipArrayDestroy(hipArray_t array) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipArrayDestroy_fn)(array);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipArrayGetDescriptor_fn)(pArrayDescriptor, array);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags,
                           hipArray_t array) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipArrayGetInfo_fn)(desc, extent, flags, array);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipBindTexture(size_t* offset, const textureReference* tex,
                                     const void* devPtr, const hipChannelFormatDesc* desc,
                                     size_t size) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipBindTexture_fn)(offset, tex, devPtr, desc, size);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipBindTexture2D(size_t* offset, const textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipBindTexture2D_fn)(offset, tex, devPtr, desc, width, height,
                                                         pitch);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipBindTextureToArray(const textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipBindTextureToArray_fn)(tex, array, desc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipBindTextureToMipmappedArray(const textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipBindTextureToMipmappedArray_fn)(tex, mipmappedArray, desc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipChooseDeviceR0600(int* device, const hipDeviceProp_t* prop) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipChooseDevice_fn)(device, prop);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* properties) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipChooseDeviceR0000_fn)(device, properties);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                       hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipConfigureCall_fn)(gridDim, blockDim, sharedMem, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                  const hipResourceDesc* pResDesc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCreateSurfaceObject_fn)(pSurfObject, pResDesc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const struct hipResourceViewDesc* pResViewDesc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCreateTextureObject_fn)(pTexObject, pResDesc, pTexDesc,
                                                               pResViewDesc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxCreate_fn)(ctx, flags, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxDestroy(hipCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxDestroy_fn)(ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxDisablePeerAccess_fn)(peerCtx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxEnablePeerAccess_fn)(peerCtx, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int* apiVersion) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxGetApiVersion_fn)(ctx, apiVersion);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxGetCacheConfig_fn)(cacheConfig);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxGetCurrent_fn)(ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxGetDevice(hipDevice_t* device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxGetDevice_fn)(device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxGetFlags(unsigned int* flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxGetFlags_fn)(flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxGetSharedMemConfig_fn)(pConfig);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxPopCurrent_fn)(ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxPushCurrent_fn)(ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxSetCacheConfig_fn)(cacheConfig);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxSetCurrent_fn)(ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxSetSharedMemConfig_fn)(config);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipCtxSynchronize(void) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCtxSynchronize_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDestroyExternalMemory_fn)(extMem);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDestroyExternalSemaphore_fn)(extSem);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDestroySurfaceObject_fn)(surfaceObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDestroyTextureObject_fn)(textureObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceCanAccessPeer_fn)(canAccessPeer, deviceId,
                                                               peerDeviceId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceComputeCapability_fn)(major, minor, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceDisablePeerAccess_fn)(peerDeviceId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceEnablePeerAccess_fn)(peerDeviceId, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGet_fn)(device, ordinal);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetAttribute_fn)(pi, attr, deviceId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetByPCIBusId_fn)(device, pciBusId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetCacheConfig_fn)(cacheConfig);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetDefaultMemPool_fn)(mem_pool, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetGraphMemAttribute_fn)(device, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetLimit_fn)(pValue, limit);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetMemPool_fn)(mem_pool, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetName_fn)(name, len, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice,
                                    int dstDevice) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetP2PAttribute_fn)(value, attr, srcDevice, dstDevice);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetPCIBusId_fn)(pciBusId, len, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetSharedMemConfig_fn)(pConfig);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetStreamPriorityRange_fn)(leastPriority,
                                                                        greatestPriority);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetUuid_fn)(uuid, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, hipDevice_t device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetLuid_fn)(luid, deviceNodeMask, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGraphMemTrim(int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGraphMemTrim_fn)(device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevicePrimaryCtxGetState_fn)(dev, flags, active);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevicePrimaryCtxRelease_fn)(dev);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevicePrimaryCtxReset_fn)(dev);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevicePrimaryCtxRetain_fn)(pctx, dev);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevicePrimaryCtxSetFlags_fn)(dev, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceReset(void) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceReset_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceSetCacheConfig_fn)(cacheConfig);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceSetGraphMemAttribute_fn)(device, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceSetLimit_fn)(limit, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceSetMemPool_fn)(device, mem_pool);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceSetSharedMemConfig_fn)(config);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceSynchronize(void) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceSynchronize_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceTotalMem_fn)(bytes, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDriverGetVersion(int* driverVersion) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDriverGetVersion_fn)(driverVersion);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGetErrorName(hipError_t hipError, const char** errorString) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGetErrorName_fn)(hipError, errorString);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGetErrorString(hipError_t hipError, const char** errorString) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGetErrorString_fn)(hipError, errorString);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGraphAddMemcpyNode_fn)(phGraphNode, hGraph, dependencies,
                                                                 numDependencies, copyParams, ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvMemcpy2DUnaligned_fn)(pCopy);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvMemcpy3D_fn)(pCopy);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvMemcpy3DAsync_fn)(pCopy, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes,
                                      void** data, hipDeviceptr_t ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvPointerGetAttributes_fn)(numAttributes, attributes, data,
                                                                   ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventCreate(hipEvent_t* event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventCreate_fn)(event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventCreateWithFlags_fn)(event, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventDestroy(hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventDestroy_fn)(event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventElapsedTime_fn)(ms, start, stop);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventQuery(hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventQuery_fn)(event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventRecord_fn)(event, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventSynchronize(hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventSynchronize_fn)(event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype,
                                        uint32_t* hopcount) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtGetLinkTypeAndHopCount_fn)(device1, device2, linktype,
                                                                     hopcount);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks,
                                         dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                         hipStream_t stream, hipEvent_t startEvent,
                                         hipEvent_t stopEvent, int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtLaunchKernel_fn)(function_address, numBlocks, dimBlocks,
                                                           args, sharedMemBytes, stream, startEvent,
                                                           stopEvent, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                              unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtLaunchMultiKernelMultiDevice_fn)(launchParamsList,
                                                                           numDevices, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtMallocWithFlags_fn)(ptr, sizeBytes, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtStreamCreateWithCUMask_fn)(stream, cuMaskSize, cuMask);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtStreamGetCUMask_fn)(stream, cuMaskSize, cuMask);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExternalMemoryGetMappedBuffer(void** devPtr, hipExternalMemory_t extMem,
                                            const hipExternalMemoryBufferDesc* bufferDesc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExternalMemoryGetMappedBuffer_fn)(devPtr, extMem,
                                                                         bufferDesc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFree(void* ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFree_fn)(ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFreeArray(hipArray_t array) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFreeArray_fn)(array);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFreeAsync_fn)(dev_ptr, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFreeHost(void* ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFreeHost_fn)(ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFreeMipmappedArray_fn)(mipmappedArray);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFuncGetAttribute_fn)(value, attrib, hfunc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFuncGetAttributes_fn)(attr, func);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFuncSetAttribute_fn)(func, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFuncSetCacheConfig_fn)(func, config);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipFuncSetSharedMemConfig_fn)(func, config);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGLGetDevices_fn)(pHipDeviceCount, pHipDevices,
                                                        hipDeviceCount, deviceList);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetChannelDesc_fn)(desc, array);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetDevice(int* deviceId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetDevice_fn)(deviceId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetDeviceCount(int* count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetDeviceCount_fn)(count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetDeviceFlags(unsigned int* flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetDeviceFlags_fn)(flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int deviceId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetDevicePropertiesR0600_fn)(prop, deviceId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetDevicePropertiesR0000_fn)(prop, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                  hipDriverEntryPointQueryResult* status) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetDriverEntryPoint_fn)(symbol, funcPtr, flags, status);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetDriverEntryPoint_spt(const char* symbol, void** funcPtr, unsigned long long flags,
                                      hipDriverEntryPointQueryResult* status) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetDriverEntryPoint_spt_fn)(symbol, funcPtr, flags, status);
  CATCH;
}
const char* hipGetErrorName(hipError_t hip_error) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetErrorName_fn)(hip_error);
  CATCHRET(const char*);
}
const char* hipGetErrorString(hipError_t hipError) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetErrorString_fn)(hipError);
  CATCHRET(const char*);
}
HIP_PUBLIC_API hipError_t hipGetLastError(void) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetLastError_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,
                                     hipMipmappedArray_const_t mipmappedArray, unsigned int level) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetMipmappedArrayLevel_fn)(levelArray, mipmappedArray,
                                                                  level);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExternalMemoryGetMappedMipmappedArray(
    hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem,
    const hipExternalMemoryMipmappedArrayDesc* mipmapDesc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExternalMemoryGetMappedMipmappedArray_fn)(mipmap, extMem,
                                                                                 mipmapDesc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetSymbolAddress_fn)(devPtr, symbol);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetSymbolSize(size_t* size, const void* symbol) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetSymbolSize_fn)(size, symbol);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* texref) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetTextureAlignmentOffset_fn)(offset, texref);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetTextureObjectResourceDesc_fn)(pResDesc, textureObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetTextureObjectResourceViewDesc_fn)(pResViewDesc,
                                                                            textureObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetTextureObjectTextureDesc_fn)(pTexDesc, textureObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetTextureReference_fn)(texref, symbol);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddChildGraphNode_fn)(pGraphNode, graph, pDependencies,
                                                                  numDependencies, childGraph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddDependencies_fn)(graph, from, to, numDependencies);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddEmptyNode_fn)(pGraphNode, graph, pDependencies,
                                                             numDependencies);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddEventRecordNode_fn)(pGraphNode, graph, pDependencies,
                                                                   numDependencies, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddEventWaitNode_fn)(pGraphNode, graph, pDependencies,
                                                                 numDependencies, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddHostNode_fn)(pGraphNode, graph, pDependencies,
                                                            numDependencies, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddKernelNode_fn)(pGraphNode, graph, pDependencies,
                                                              numDependencies, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   hipMemAllocNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddMemAllocNode_fn)(pGraphNode, graph, pDependencies,
                                                                numDependencies, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  void* dev_ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddMemFreeNode_fn)(pGraphNode, graph, pDependencies,
                                                               numDependencies, dev_ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddMemcpyNode_fn)(pGraphNode, graph, pDependencies,
                                                              numDependencies, pCopyParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddMemcpyNode1D_fn)(
      pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddMemcpyNodeFromSymbol_fn)(
      pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddMemcpyNodeToSymbol_fn)(
      pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddMemsetNode_fn)(pGraphNode, graph, pDependencies,
                                                              numDependencies, pMemsetParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                           const hipGraphNode_t* pDependencies, size_t numDependencies,
                           hipGraphNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddNode_fn)(pGraphNode, graph, pDependencies,
                                                        numDependencies, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphChildGraphNodeGetGraph_fn)(node, pGraph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphClone_fn)(pGraphClone, originalGraph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphCreate_fn)(pGraph, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphDebugDotPrint_fn)(graph, path, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphDestroy(hipGraph_t graph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphDestroy_fn)(graph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphDestroyNode_fn)(node);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphEventRecordNodeGetEvent_fn)(node, event_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphEventRecordNodeSetEvent_fn)(node, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphEventWaitNodeGetEvent_fn)(node, event_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphEventWaitNodeSetEvent_fn)(node, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecChildGraphNodeSetParams_fn)(hGraphExec, node,
                                                                            childGraph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                     hipGraphNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecNodeSetParams_fn)(hGraphExec, node, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecDestroy_fn)(graphExec);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecEventRecordNodeSetEvent_fn)(hGraphExec, hNode,
                                                                            event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecEventWaitNodeSetEvent_fn)(hGraphExec, hNode, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecHostNodeSetParams_fn)(hGraphExec, node,
                                                                      pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecKernelNodeSetParams_fn)(hGraphExec, node,
                                                                        pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecMemcpyNodeSetParams_fn)(hGraphExec, node,
                                                                        pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecMemcpyNodeSetParams1D_fn)(hGraphExec, node, dst,
                                                                          src, count, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecMemcpyNodeSetParamsFromSymbol_fn)(
      hGraphExec, node, dst, symbol, count, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset,
                                                   hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecMemcpyNodeSetParamsToSymbol_fn)(
      hGraphExec, node, symbol, src, count, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecMemsetNodeSetParams_fn)(hGraphExec, node,
                                                                        pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecUpdate_fn)(hGraphExec, hGraph, hErrorNode_out,
                                                           updateResult_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphGetEdges_fn)(graph, from, to, numEdges);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphGetNodes_fn)(graph, nodes, numNodes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes) {
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphGetRootNodes_fn)(graph, pRootNodes, pNumRootNodes);
}
HIP_PUBLIC_API hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphHostNodeGetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphHostNodeSetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphInstantiate_fn)(pGraphExec, graph, pErrorNode,
                                                            pLogBuffer, bufferSize);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphInstantiateWithFlags_fn)(pGraphExec, graph, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphInstantiateWithParams(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                         hipGraphInstantiateParams* instantiateParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphInstantiateWithParams_fn)(pGraphExec, graph,
                                                                      instantiateParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphKernelNodeCopyAttributes_fn)(hSrc, hDst);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value) {
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphKernelNodeGetAttribute_fn)(hNode, attr, value);
}
HIP_PUBLIC_API hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphKernelNodeGetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphKernelNodeSetAttribute_fn)(hNode, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                       const hipKernelNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphKernelNodeSetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphLaunch_fn)(graphExec, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemAllocNodeGetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemFreeNodeGetParams_fn)(node, dev_ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemcpyNodeGetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemcpyNodeSetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemcpyNodeSetParams1D_fn)(node, dst, src, count, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemcpyNodeSetParamsFromSymbol_fn)(node, dst, symbol,
                                                                              count, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemcpyNodeSetParamsToSymbol_fn)(node, symbol, src,
                                                                            count, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemsetNodeGetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphMemsetNodeSetParams_fn)(node, pNodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphNodeFindInClone_fn)(pNode, originalNode, clonedGraph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphNodeGetDependencies_fn)(node, pDependencies,
                                                                    pNumDependencies);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphNodeGetDependentNodes_fn)(node, pDependentNodes,
                                                                      pNumDependentNodes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphNodeGetEnabled_fn)(hGraphExec, hNode, isEnabled);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphNodeGetType_fn)(node, pType);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled) {
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphNodeSetEnabled_fn)(hGraphExec, hNode, isEnabled);
}
HIP_PUBLIC_API hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphReleaseUserObject_fn)(graph, object, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphRemoveDependencies_fn)(graph, from, to,
                                                                   numDependencies);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count,
                                    unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphRetainUserObject_fn)(graph, object, count, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphUpload_fn)(graphExec, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphicsGLRegisterBuffer_fn)(resource, buffer, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphicsGLRegisterImage_fn)(resource, image, target, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphicsMapResources_fn)(count, resources, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphicsResourceGetMappedPointer_fn)(devPtr, size, resource);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource,
                                                unsigned int arrayIndex, unsigned int mipLevel) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphicsSubResourceGetMappedArray_fn)(array, resource,
                                                                             arrayIndex, mipLevel);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphicsUnmapResources_fn)(count, resources, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphicsUnregisterResource_fn)(resource);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHostAlloc_fn)(ptr, size, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipHostFree(void* ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHostFree_fn)(ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHostGetDevicePointer_fn)(devPtr, hstPtr, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHostGetFlags_fn)(flagsPtr, hostPtr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHostMalloc_fn)(ptr, size, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHostRegister_fn)(hostPtr, sizeBytes, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipHostUnregister(void* hostPtr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHostUnregister_fn)(hostPtr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out,
                                   const hipExternalMemoryHandleDesc* memHandleDesc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipImportExternalMemory_fn)(extMem_out, memHandleDesc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipImportExternalSemaphore_fn)(extSem_out, semHandleDesc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGraphAddMemsetNode_fn)(
      phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipInit(unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipInit_fn)(flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipInitDevice(int device, unsigned int deviceFlags, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipInitDevice_fn)(device, deviceFlags, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipIpcCloseMemHandle(void* devPtr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipIpcCloseMemHandle_fn)(devPtr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipIpcGetEventHandle_fn)(handle, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipIpcGetMemHandle_fn)(handle, devPtr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipIpcOpenEventHandle_fn)(event, handle);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipIpcOpenMemHandle_fn)(devPtr, handle, flags);
  CATCH;
}
HIP_PUBLIC_API const char* hipKernelNameRef(const hipFunction_t f) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelNameRef_fn)(f);
  CATCHRET(const char*);
}
HIP_PUBLIC_API const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelNameRefByPtr_fn)(hostFunction, stream);
  CATCHRET(const char*);
}
HIP_PUBLIC_API hipError_t hipLaunchByPtr(const void* func) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchByPtr_fn)(func);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchCooperativeKernel_fn)(
      f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                                 unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchCooperativeKernelMultiDevice_fn)(launchParamsList,
                                                                              numDevices, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchHostFunc_fn)(stream, fn, userData);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                                      void** args, size_t sharedMemBytes, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchKernel_fn)(function_address, numBlocks, dimBlocks,
                                                        args, sharedMemBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMalloc(void** ptr, size_t size) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMalloc_fn)(ptr, size);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMalloc3D_fn)(pitchedDevPtr, extent);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                                       struct hipExtent extent, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMalloc3DArray_fn)(array, desc, extent, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc,
                                     size_t width, size_t height, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMallocArray_fn)(array, desc, width, height, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMallocAsync_fn)(dev_ptr, size, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream) {
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMallocFromPoolAsync_fn)(dev_ptr, size, mem_pool, stream);
}
HIP_PUBLIC_API hipError_t hipMallocHost(void** ptr, size_t size) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMallocHost_fn)(ptr, size);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMallocManaged_fn)(dev_ptr, size, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,
                                              const struct hipChannelFormatDesc* desc,
                                              struct hipExtent extent, unsigned int numLevels,
                                              unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMallocMipmappedArray_fn)(mipmappedArray, desc, extent,
                                                                numLevels, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMallocPitch_fn)(ptr, pitch, width, height);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemAddressFree_fn)(devPtr, size);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags) {
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemAddressReserve_fn)(ptr, size, alignment, addr, flags);
}
HIP_PUBLIC_API hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemAdvise_fn)(dev_ptr, count, advice, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemAdvise_v2(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                           hipMemLocation location) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemAdvise_v2_fn)(dev_ptr, count, advice, location);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemAllocHost(void** ptr, size_t size) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemAllocHost_fn)(ptr, size);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemAllocPitch_fn)(dptr, pitch, widthInBytes, height,
                                                         elementSizeBytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemCreate_fn)(handle, size, prop, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemExportToShareableHandle(void* shareableHandle,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType handleType,
                                         unsigned long long flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemExportToShareableHandle_fn)(shareableHandle, handle,
                                                                      handleType, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetAccess_fn)(flags, location, ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetAddressRange_fn)(pbase, psize, dptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetAllocationGranularity_fn)(granularity, prop, option);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                   hipMemGenericAllocationHandle_t handle) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetAllocationPropertiesFromHandle_fn)(prop, handle);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetInfo_fn)(free, total);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle,
                                           hipMemAllocationHandleType shHandleType) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemImportFromShareableHandle_fn)(handle, osHandle,
                                                                        shHandleType);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemMap_fn)(ptr, size, offset, handle, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count,
                               hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemMapArrayAsync_fn)(mapInfoList, count, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolCreate_fn)(mem_pool, pool_props);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolDestroy_fn)(mem_pool);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolExportPointer_fn)(export_data, dev_ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool,
                                             hipMemAllocationHandleType handle_type,
                                             unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolExportToShareableHandle_fn)(shared_handle, mem_pool,
                                                                          handle_type, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                               hipMemLocation* location) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolGetAccess_fn)(flags, mem_pool, location);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolGetAttribute_fn)(mem_pool, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle,
                                               hipMemAllocationHandleType handle_type,
                                               unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolImportFromShareableHandle_fn)(mem_pool, shared_handle,
                                                                            handle_type, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool,
                                   hipMemPoolPtrExportData* export_data) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolImportPointer_fn)(dev_ptr, mem_pool, export_data);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list,
                               size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolSetAccess_fn)(mem_pool, desc_list, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolSetAttribute_fn)(mem_pool, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPoolTrimTo_fn)(mem_pool, min_bytes_to_hold);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPrefetchAsync_fn)(dev_ptr, count, device, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count, hipMemLocation location,
                                  unsigned int flags, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPrefetchAsync_v2_fn)(dev_ptr, count, location, flags,
                                                               stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPrefetchBatchAsync(void** dev_ptrs, size_t* sizes, size_t count,
                                    hipMemLocation* prefetch_locs, size_t* prefetch_loc_idxs,
                                    size_t num_prefetch_locs, unsigned long long flags,
                                    hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPrefetchBatchAsync_fn)(
      dev_ptrs, sizes, count, prefetch_locs, prefetch_loc_idxs, num_prefetch_locs, flags, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemDiscardBatchAsync(void** dev_ptrs, size_t* sizes, size_t count,
                                   unsigned long long flags, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemDiscardBatchAsync_fn)(
      dev_ptrs, sizes, count, flags, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvMemDiscardBatchAsync(hipDeviceptr_t* dptrs, size_t* sizes, size_t count,
                                      unsigned long long flags, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvMemDiscardBatchAsync_fn)(
      dptrs, sizes, count, flags, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count,
                                              hipMemLocation* prefetchLocs,
                                              size_t* prefetchLocIdxs,
                                              size_t numPrefetchLocs,
                                              unsigned long long flags, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemDiscardAndPrefetchBatchAsync_fn)(
      dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvMemDiscardAndPrefetchBatchAsync(hipDeviceptr_t* dptrs, size_t* sizes, size_t count,
                                                 hipMemLocation* prefetchLocs,
                                                 size_t* prefetchLocIdxs,
                                                 size_t numPrefetchLocs,
                                                 unsigned long long flags, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvMemDiscardAndPrefetchBatchAsync_fn)(
      dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemPtrGetInfo_fn)(ptr, size);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemRangeGetAttribute_fn)(data, data_size, attribute, dev_ptr,
                                                                count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                    hipMemRangeAttribute* attributes, size_t num_attributes,
                                    const void* dev_ptr, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemRangeGetAttributes_fn)(data, data_sizes, attributes,
                                                                 num_attributes, dev_ptr, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemRelease_fn)(handle);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemRetainAllocationHandle_fn)(handle, addr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemSetAccess_fn)(ptr, size, desc, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemUnmap(void* ptr, size_t size) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemUnmap_fn)(ptr, size);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy_fn)(dst, src, sizeBytes, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2D_fn)(dst, dpitch, src, spitch, width, height, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DAsync_fn)(dst, dpitch, src, spitch, width, height,
                                                         kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DFromArray_fn)(dst, dpitch, src, wOffset, hOffset,
                                                             width, height, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                     size_t hOffset, size_t width, size_t height,
                                     hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DFromArrayAsync_fn)(
      dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DToArray_fn)(dst, wOffset, hOffset, src, spitch,
                                                           width, height, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DToArrayAsync_fn)(dst, wOffset, hOffset, src, spitch,
                                                                width, height, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy3D_fn)(p);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy3DAsync_fn)(p, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyAsync_fn)(dst, src, sizeBytes, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray, size_t srcOffset, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyAtoH_fn)(dst, srcArray, srcOffset, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyDtoD_fn)(dst, src, sizeBytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyDtoDAsync_fn)(dst, src, sizeBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyDtoH_fn)(dst, src, sizeBytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyDtoHAsync_fn)(dst, src, sizeBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyFromArray_fn)(dst, srcArray, wOffset, hOffset, count,
                                                           kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                               hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyFromSymbol_fn)(dst, symbol, sizeBytes, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyFromSymbolAsync_fn)(dst, symbol, sizeBytes, offset,
                                                                 kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyHtoA_fn)(dstArray, dstOffset, srcHost, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyHtoD_fn)(dst, src, sizeBytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void* src, size_t sizeBytes,
                              hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyHtoDAsync_fn)(dst, src, sizeBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyParam2D_fn)(pCopy);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyParam2DAsync_fn)(pCopy, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                         size_t sizeBytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyPeer_fn)(dst, dstDeviceId, src, srcDeviceId,
                                                      sizeBytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyPeerAsync_fn)(dst, dstDeviceId, src, srcDevice,
                                                           sizeBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyToArray_fn)(dst, wOffset, hOffset, src, count, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyToSymbol_fn)(symbol, src, sizeBytes, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyToSymbolAsync_fn)(symbol, src, sizeBytes, offset, kind,
                                                               stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyWithStream_fn)(dst, src, sizeBytes, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset_fn)(dst, value, sizeBytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset2D_fn)(dst, pitch, value, width, height);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,
                            hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset2DAsync_fn)(dst, pitch, value, width, height, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset3D_fn)(pitchedDevPtr, value, extent);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                            hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset3DAsync_fn)(pitchedDevPtr, value, extent, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetAsync_fn)(dst, value, sizeBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD16_fn)(dest, value, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count,
                             hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD16Async_fn)(dest, value, count, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD32_fn)(dest, value, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD32Async_fn)(dst, value, count, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD8_fn)(dest, value, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count,
                            hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD8Async_fn)(dest, value, count, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMipmappedArrayCreate_fn)(pHandle, pMipmappedArrayDesc,
                                                                numMipmapLevels);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMipmappedArrayDestroy_fn)(hMipmappedArray);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMipmappedArrayGetLevel_fn)(pLevelArray, hMipMappedArray,
                                                                  level);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleGetFunction_fn)(function, module, kname);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleGetFunctionCount(unsigned int* count, hipModule_t mod) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleGetFunctionCount_fn)(count, mod);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleGetGlobal_fn)(dptr, bytes, hmod, name);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleGetTexRef_fn)(texRef, hmod, name);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleLaunchCooperativeKernel_fn)(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
      kernelParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices,
                                                       unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleLaunchCooperativeKernelMultiDevice_fn)(
      launchParamsList, numDevices, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleLaunchKernel_fn)(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
      kernelParams, extra);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleLoadFatBinary_fn)(module, fatbin);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleLoad_fn)(module, fname);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleLoadData_fn)(module, image);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleLoadDataEx_fn)(module, image, numOptions, options,
                                                            optionValues);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipLinkAddData(hipLinkState_t state, hipJitInputType type, void* data, size_t size,
                          const char* name, unsigned int numOptions, hipJitOption* options,
                          void** optionValues) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLinkAddData_fn)(state, type, data, size, name, numOptions,
                                                       options, optionValues);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipLinkAddFile(hipLinkState_t state, hipJitInputType type, const char* path,
                          unsigned int numOptions, hipJitOption* options, void** optionValues) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLinkAddFile_fn)(state, type, path, numOptions, options,
                                                       optionValues);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipLinkComplete(hipLinkState_t state, void** hipBinOut, size_t* sizeOut) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLinkComplete_fn)(state, hipBinOut, sizeOut);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipLinkCreate(unsigned int numOptions, hipJitOption* options, void** optionValues,
                         hipLinkState_t* stateOut) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLinkCreate_fn)(numOptions, options, optionValues, stateOut);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipLinkDestroy(hipLinkState_t state) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLinkDestroy_fn)(state);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn)(
      numBlocks, f, blockSize, dynSharedMemPerBlk);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn)(
      numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                              hipFunction_t f,
                                                              size_t dynSharedMemPerBlk,
                                                              int blockSizeLimit) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleOccupancyMaxPotentialBlockSize_fn)(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit,
    unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn)(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipModuleUnload(hipModule_t module) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleUnload_fn)(module);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* f,
                                                               int numBlocks, int blockSize) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipOccupancyAvailableDynamicSMemPerBlock_fn)(dynamicSmemSize,
                                                                                 f, numBlocks,
                                                                                 blockSize);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                                   int blockSize,
                                                                   size_t dynSharedMemPerBlk) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipOccupancyMaxActiveBlocksPerMultiprocessor_fn)(
      numBlocks, f, blockSize, dynSharedMemPerBlk);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn)(
      numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                        const void* f, size_t dynSharedMemPerBlk,
                                                        int blockSizeLimit) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipOccupancyMaxPotentialBlockSize_fn)(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipPeekAtLastError(void) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipPeekAtLastError_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipOccupancyMaxActiveClusters(int* numClusters, const void* func,
                                                    const hipLaunchConfig_t* launchConfig) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipOccupancyMaxActiveClusters_fn)(numClusters, func,
                                                                      launchConfig);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipOccupancyMaxPotentialClusterSize(int* clusterSize, const void* func,
                                                          const hipLaunchConfig_t* config) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipOccupancyMaxPotentialClusterSize_fn)(clusterSize, func,
                                                                            config);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipPointerGetAttribute_fn)(data, attribute, ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipPointerGetAttributes_fn)(attributes, ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipPointerSetAttribute_fn)(value, attribute, ptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipProfilerStart() {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipProfilerStart_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipProfilerStop() {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipProfilerStop_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipRuntimeGetVersion_fn)(runtimeVersion);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipSetDevice(int deviceId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipSetDevice_fn)(deviceId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipSetDeviceFlags(unsigned flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipSetDeviceFlags_fn)(flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipSetupArgument_fn)(arg, size, offset);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipSignalExternalSemaphoresAsync_fn)(extSemArray, paramsArray,
                                                                         numExtSems, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamAddCallback_fn)(stream, callback, userData, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length,
                                   unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamAttachMemAsync_fn)(stream, dev_ptr, length, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamBeginCapture_fn)(stream, mode);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamCopyAttributes_fn)(dst, src);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamCreate(hipStream_t* stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamCreate_fn)(stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamCreateWithFlags_fn)(stream, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamCreateWithPriority_fn)(stream, flags, priority);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamDestroy(hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamDestroy_fn)(stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamEndCapture_fn)(stream, pGraph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetCaptureInfo_fn)(stream, pCaptureStatus, pId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetCaptureInfo_v2_fn)(
      stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetDevice_fn)(stream, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetFlags_fn)(stream, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetId(hipStream_t stream, unsigned long long* streamId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetId_fn)(stream, streamId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetPriority_fn)(stream, priority);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamIsCapturing_fn)(stream, pCaptureStatus);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamQuery(hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamQuery_fn)(stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamSynchronize(hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamSynchronize_fn)(stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamUpdateCaptureDependencies_fn)(stream, dependencies,
                                                                           numDependencies, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamWaitEvent_fn)(stream, event, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamWaitValue32_fn)(stream, ptr, value, flags, mask);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamWaitValue64_fn)(stream, ptr, value, flags, mask);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value,
                                 unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamWriteValue32_fn)(stream, ptr, value, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value,
                                 unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamWriteValue64_fn)(stream, ptr, value, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamBatchMemOp_fn)(stream, count, paramArray, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject, const HIP_RESOURCE_DESC* pResDesc,
                              const HIP_TEXTURE_DESC* pTexDesc,
                              const HIP_RESOURCE_VIEW_DESC* pResViewDesc) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexObjectCreate_fn)(pTexObject, pResDesc, pTexDesc,
                                                           pResViewDesc);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexObjectDestroy_fn)(texObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, hipTextureObject_t texObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexObjectGetResourceDesc_fn)(pResDesc, texObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                           hipTextureObject_t texObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexObjectGetResourceViewDesc_fn)(pResViewDesc, texObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, hipTextureObject_t texObject) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexObjectGetTextureDesc_fn)(pTexDesc, texObject);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetAddress_fn)(dev_ptr, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode* pam, const textureReference* texRef,
                                   int dim) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetAddressMode_fn)(pam, texRef, dim);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode* pfm, const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetFilterMode_fn)(pfm, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetFlags_fn)(pFlags, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels,
                              const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetFormat_fn)(pFormat, pNumChannels, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetMaxAnisotropy_fn)(pmaxAnsio, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,
                                                 const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetMipMappedArray_fn)(pArray, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode* pfm,
                                        const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetMipmapFilterMode_fn)(pfm, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetMipmapLevelBias_fn)(pbias, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp,
                                        const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetMipmapLevelClamp_fn)(pminMipmapLevelClamp,
                                                                     pmaxMipmapLevelClamp, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr,
                               size_t bytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetAddress_fn)(ByteOffset, texRef, dptr, bytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr, size_t Pitch) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetAddress2D_fn)(texRef, desc, dptr, Pitch);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetAddressMode(textureReference* texRef, int dim,
                                   enum hipTextureAddressMode am) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetAddressMode_fn)(texRef, dim, am);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetArray_fn)(tex, array, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetBorderColor_fn)(texRef, pBorderColor);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetFilterMode(textureReference* texRef, enum hipTextureFilterMode fm) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetFilterMode_fn)(texRef, fm);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetFlags(textureReference* texRef, unsigned int Flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetFlags_fn)(texRef, Flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetFormat(textureReference* texRef, hipArray_Format fmt,
                              int NumPackedComponents) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetFormat_fn)(texRef, fmt, NumPackedComponents);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetMaxAnisotropy_fn)(texRef, maxAniso);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef, enum hipTextureFilterMode fm) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetMipmapFilterMode_fn)(texRef, fm);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetMipmapLevelBias_fn)(texRef, bias);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp,
                                        float maxMipMapLevelClamp) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetMipmapLevelClamp_fn)(texRef, minMipMapLevelClamp,
                                                                     maxMipMapLevelClamp);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefSetMipmappedArray(textureReference* texRef,
                                      struct hipMipmappedArray* mipmappedArray,
                                      unsigned int Flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefSetMipmappedArray_fn)(texRef, mipmappedArray, Flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipThreadExchangeStreamCaptureMode_fn)(mode);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipUnbindTexture(const textureReference* tex) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipUnbindTexture_fn)(tex);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                               unsigned int initialRefcount, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipUserObjectCreate_fn)(object_out, ptr, destroy,
                                                            initialRefcount, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipUserObjectRelease_fn)(object, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipUserObjectRetain_fn)(object, count);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                          const hipExternalSemaphoreWaitParams* paramsArray,
                                          unsigned int numExtSems, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipWaitExternalSemaphoresAsync_fn)(extSemArray, paramsArray,
                                                                       numExtSems, stream);
  CATCH;
}
HIP_PUBLIC_API hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                     hipChannelFormatKind f) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipCreateChannelDesc_fn)(x, y, z, w, f);
  CATCHRET(hipChannelFormatDesc)
}

HIP_PUBLIC_API hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                               uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                               uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                               uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                               hipStream_t hStream, void** kernelParams,
                                               void** extra, hipEvent_t startEvent,
                                               hipEvent_t stopEvent, uint32_t flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtModuleLaunchKernel_fn)(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                               uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                               uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                               uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                               hipStream_t hStream, void** kernelParams,
                                               void** extra, hipEvent_t startEvent,
                                               hipEvent_t stopEvent) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipHccModuleLaunchKernel_fn)(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy_spt_fn)(dst, src, sizeBytes, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                                 size_t offset, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyToSymbol_spt_fn)(symbol, src, sizeBytes, offset, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                   hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyFromSymbol_spt_fn)(dst, symbol, sizeBytes, offset,
                                                                kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                           size_t height, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2D_spt_fn)(dst, dpitch, src, spitch, width, height,
                                                        kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                    size_t hOffset, size_t width, size_t height,
                                    hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DFromArray_spt_fn)(dst, dpitch, src, wOffset, hOffset,
                                                                 width, height, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy3D_spt_fn)(p);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset_spt_fn)(dst, value, sizeBytes);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetAsync_spt(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetAsync_spt_fn)(dst, value, sizeBytes, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset2D_spt_fn)(dst, pitch, value, width, height);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value, size_t width, size_t height,
                                hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset2DAsync_spt_fn)(dst, pitch, value, width, height,
                                                             stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset3DAsync_spt_fn)(pitchedDevPtr, value, extent, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemset3D_spt_fn)(pitchedDevPtr, value, extent);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                              hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyAsync_spt_fn)(dst, src, sizeBytes, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms* p, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy3DAsync_spt_fn)(p, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, hipMemcpyKind kind,
                                hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DAsync_spt_fn)(dst, dpitch, src, spitch, width,
                                                             height, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t sizeBytes,
                                        size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyFromSymbolAsync_spt_fn)(dst, symbol, sizeBytes, offset,
                                                                     kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t sizeBytes,
                                      size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyToSymbolAsync_spt_fn)(symbol, src, sizeBytes, offset,
                                                                   kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                  size_t hOffset, size_t count, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyFromArray_spt_fn)(dst, src, wOffsetSrc, hOffset, count,
                                                               kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DToArray_spt_fn)(dst, wOffset, hOffset, src, spitch,
                                                               width, height, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                         size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                         size_t height, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DFromArrayAsync_spt_fn)(
      dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset,
                                       const void* src, size_t spitch, size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DToArrayAsync_spt_fn)(
      dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamQuery_spt(hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamQuery_spt_fn)(stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamSynchronize_spt_fn)(stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetPriority_spt_fn)(stream, priority);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamWaitEvent_spt_fn)(stream, event, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetFlags_spt_fn)(stream, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback,
                                    void* userData, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamAddCallback_spt_fn)(stream, callback, userData, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventRecord_spt_fn)(event, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim, dim3 blockDim,
                                          void** kernelParams, uint32_t sharedMemBytes,
                                          hipStream_t hStream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchCooperativeKernel_spt_fn)(
      f, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipLaunchKernel_spt(const void* function_address, dim3 numBlocks,
                                          dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                          hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchKernel_spt_fn)(function_address, numBlocks, dimBlocks,
                                                            args, sharedMemBytes, stream);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphLaunch_spt_fn)(graphExec, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamBeginCapture_spt_fn)(stream, mode);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamEndCapture_spt_fn)(stream, pGraph);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamIsCapturing_spt_fn)(stream, pCaptureStatus);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                       unsigned long long* pId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetCaptureInfo_spt_fn)(stream, pCaptureStatus, pId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,
                                          hipStreamCaptureStatus* captureStatus_out,
                                          unsigned long long* id_out, hipGraph_t* graph_out,
                                          const hipGraphNode_t** dependencies_out,
                                          size_t* numDependencies_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetCaptureInfo_v2_spt_fn)(
      stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void* userData) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchHostFunc_spt_fn)(stream, fn, userData);
  CATCH;
}
HIP_PUBLIC_API int hipGetStreamDeviceId(hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetStreamDeviceId_fn)(stream);
  CATCHRET(int)
}
HIP_PUBLIC_API hipError_t hipExtGetLastError() {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtGetLastError_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetBorderColor(float* pBorderColor, const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetBorderColor_fn)(pBorderColor, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipTexRefGetArray(hipArray_t* pArray, const textureReference* texRef) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipTexRefGetArray_fn)(pArray, texRef);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetProcAddress(const char* symbol, void** pfn, int hipVersion,
                                        uint64_t flags,
                                        hipDriverProcAddressQueryResult* symbolStatus) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetProcAddress_fn)(symbol, pfn, hipVersion, flags,
                                                          symbolStatus);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetProcAddress_spt(const char* symbol, void** pfn, int  hipVersion,
                                            uint64_t flags,
                                            hipDriverProcAddressQueryResult* symbolStatus) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetProcAddress_spt_fn)(symbol, pfn, hipVersion, flags,
                                                              symbolStatus);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
                                        const hipGraphNode_t* dependencies,
                                        const hipGraphEdgeData* dependencyData,
                                        size_t numDependencies, hipStreamCaptureMode mode) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamBeginCaptureToGraph_fn)(
      stream, graph, dependencies, dependencyData, numDependencies, mode);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGetFuncBySymbol(hipFunction_t* functionPtr, const void* symbolPtr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGetFuncBySymbol_fn)(functionPtr, symbolPtr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGraphExecMemsetNodeSetParams_fn)(hGraphExec, hNode,
                                                                           memsetParams, ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec, unsigned long long* flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecGetFlags_fn)(graphExec, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     hipDeviceptr_t dptr) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGraphAddMemFreeNode_fn)(phGraphNode, hGraph, dependencies,
                                                                  numDependencies, dptr);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGraphExecMemcpyNodeSetParams_fn)(hGraphExec, hNode,
                                                                           copyParams, ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipSetValidDevices(int* device_arr, int len) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipSetValidDevices_fn)(device_arr, len);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset,
                         size_t ByteCount) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyAtoD_fn)(dstDevice, srcArray, srcOffset, ByteCount);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice,
                         size_t ByteCount) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyDtoA_fn)(dstArray, dstOffset, srcDevice, ByteCount);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray,
                         size_t srcOffset, size_t ByteCount) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyAtoA_fn)(dstArray, dstOffset, srcArray, srcOffset,
                                                      ByteCount);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray, size_t srcOffset,
                              size_t ByteCount, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyAtoHAsync_fn)(dstHost, srcArray, srcOffset, ByteCount,
                                                           stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                              size_t ByteCount, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyHtoAAsync_fn)(dstArray, dstOffset, srcHost, ByteCount,
                                                           stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                   hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy2DArrayToArray_fn)(
      dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGraphMemcpyNodeGetParams_fn)(hNode, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t hNode, const HIP_MEMCPY3D* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvGraphMemcpyNodeSetParams_fn)(hNode, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphNodeSetParams(hipGraphNode_t node, hipGraphNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphNodeSetParams_fn)(node, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     const hipBatchMemOpNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddBatchMemOpNode_fn)(pGraphNode, graph, dependencies,
                                                                  numDependencies, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphBatchMemOpNodeGetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphBatchMemOpNodeGetParams_fn)(hNode, nodeParams_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphBatchMemOpNodeSetParams_fn)(hNode, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               const hipBatchMemOpNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecBatchMemOpNodeSetParams_fn)(hGraphExec, hNode,
                                                                            nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipEventRecordWithFlags_fn)(event, stream, flags);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, const void* fPtr, void** args) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLaunchKernelExC_fn)(config, fPtr, args);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, hipFunction_t f, void** kernel,
                                void** extra) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDrvLaunchKernelEx_fn)(config, f, kernel, extra);
  CATCH;
}

HIP_PUBLIC_API hipError_t hipMemGetHandleForAddressRange(void* handle, hipDeviceptr_t dptr, size_t size,
                                          hipMemRangeHandleType handleType,
                                          unsigned long long flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetHandleForAddressRange_fn)(handle, dptr, size,
                                                                       handleType, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch, unsigned char value, size_t width,
                         size_t height) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD2D8_fn)(dst, dstPitch, value, width, height);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
                              size_t width, size_t height, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD2D8Async_fn)(dst, dstPitch, value, width, height,
                                                           stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch, unsigned short value, size_t width,
                          size_t height) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD2D16_fn)(dst, dstPitch, value, width, height);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
                               size_t width, size_t height, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD2D16Async_fn)(dst, dstPitch, value, width, height,
                                                            stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width,
                          size_t height) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD2D32_fn)(dst, dstPitch, value, width, height);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch, unsigned int value,
                               size_t width, size_t height, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemsetD2D32Async_fn)(dst, dstPitch, value, width, height,
                                                            stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamSetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 const hipStreamAttrValue* value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamSetAttribute_fn)(stream, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 hipStreamAttrValue* value) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetAttribute_fn)(stream, attr, value);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count,
                               hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                               size_t* failIdx, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpyBatchAsync_fn)(dsts, srcs, sizes, count, attrs,
                                                            attrsIdxs, numAttrs, failIdx, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp* opList, size_t* failIdx,
                                 unsigned long long flags, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy3DBatchAsync_fn)(numOps, opList, failIdx, flags,
                                                              stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy3DPeer(hipMemcpy3DPeerParms* p) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy3DPeer_fn)(p);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemcpy3DPeerAsync(hipMemcpy3DPeerParms* p, hipStream_t stream) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemcpy3DPeerAsync_fn)(p, stream);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements,
                                               const hipChannelFormatDesc* fmtDesc, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetTexture1DLinearMaxWidth_fn)(maxWidthInElements,
                                                                            fmtDesc, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddExternalSemaphoresSignalNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddExternalSemaphoresSignalNode_fn)(
      pGraphNode, graph, pDependencies, numDependencies, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphAddExternalSemaphoresWaitNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphAddExternalSemaphoresWaitNode_fn)(
      pGraphNode, graph, pDependencies, numDependencies, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExternalSemaphoresSignalNodeSetParams_fn)(hNode,
                                                                                      nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExternalSemaphoresSignalNodeGetParams_fn)(hNode,
                                                                                      params_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExternalSemaphoresWaitNodeGetParams_fn)(hNode,
                                                                                    params_out);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExternalSemaphoresWaitNodeSetParams_fn)(hNode,
                                                                                    nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecExternalSemaphoresSignalNodeSetParams_fn)(
      hGraphExec, hNode, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGraphExecExternalSemaphoresWaitNodeSetParams_fn)(
      hGraphExec, hNode, nodeParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryLoadData(hipLibrary_t* library, const void* code, hipJitOption* jitOptions,
                              void** jitOptionsValues, unsigned int numJitOptions,
                              hipLibraryOption* libraryOptions, void** libraryOptionValues,
                              unsigned int numLibraryOptions) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryLoadData_fn)(
      library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
      libraryOptionValues, numLibraryOptions);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryLoadFromFile(hipLibrary_t* library, const char* fileName,
                                  hipJitOption* jitOptions, void** jitOptionsValues,
                                  unsigned int numJitOptions, hipLibraryOption* libraryOptions,
                                  void** libraryOptionValues, unsigned int numLibraryOptions) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryLoadFromFile_fn)(
      library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
      libraryOptionValues, numLibraryOptions);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryUnload(hipLibrary_t library) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryUnload_fn)(library);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryGetKernel(hipKernel_t* pKernel, hipLibrary_t library, const char* name)  {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryGetKernel_fn)(pKernel, library,
                                                            name);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryGetKernelCount(unsigned int *count, hipLibrary_t library) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryGetKernelCount_fn)(count,
                                                                 library);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryGetGlobal(void** dptr, size_t* bytes, hipLibrary_t library,
                               const char* name) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryGetGlobal_fn)(dptr, bytes, library, name);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryGetManaged(void** dptr, size_t* bytes, hipLibrary_t library,
                                const char* name) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryGetManaged_fn)(dptr, bytes, library, name);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryGetModule(hipModule_t* pMod, hipLibrary_t library) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryGetModule_fn)(pMod, library);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipKernelGetAttribute(int* pi, hipFunction_attribute attrib, hipKernel_t kernel,
                                 hipDevice_t dev) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelGetAttribute_fn)(pi, attrib, kernel, dev);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipLibraryEnumerateKernels(hipKernel_t* kernels, unsigned int numKernels,
                                      hipLibrary_t library) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipLibraryEnumerateKernels_fn)(kernels, numKernels, library);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipKernelGetLibrary(hipLibrary_t* library, hipKernel_t kernel) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelGetLibrary_fn)(library, kernel);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipKernelGetName(const char** name, hipKernel_t kernel) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelGetName_fn)(name, kernel);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipKernelGetParamInfo(hipKernel_t kernel, size_t paramIndex, size_t* paramOffset,
                                 size_t* paramSize) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelGetParamInfo_fn)(kernel, paramIndex, paramOffset,
                                                              paramSize);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtEnableLogging() {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtEnableLogging_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtDisableLogging() {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtDisableLogging_fn)();
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size, size_t log_mask) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExtSetLoggingParams_fn)(log_level, log_size, log_mask);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemSetMemPool(hipMemLocation* location, hipMemAllocationType type,
                            hipMemPool_t pool) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemSetMemPool_fn)(location, type, pool);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemGetMemPool(hipMemPool_t* pool, hipMemLocation* location,
                            hipMemAllocationType type) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetMemPool_fn)(pool, location, type);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMipmappedArrayGetMemoryRequirements(hipArrayMemoryRequirements* memoryRequirements,
                                                  hipMipmappedArray_t mipmap, hipDevice_t device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMipmappedArrayGetMemoryRequirements_fn)(memoryRequirements,
                                                                               mipmap, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipKernelSetAttribute(hipFunction_attribute attrib, int value, hipKernel_t kernel, hipDevice_t dev) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelSetAttribute_fn)(attrib, value, kernel, dev);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipKernelSetAttributeForDevice(hipKernel_t kernel, hipFuncAttribute attr,
                                                         int value, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelSetAttributeForDevice_fn)(kernel, attr, value, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipKernelGetFunction(hipFunction_t* pFunc, hipKernel_t kernel) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipKernelGetFunction_fn)(pFunc, kernel) ;
  CATCH;
}
HIP_PUBLIC_API hipError_t hipGreenCtxCreate(hipExecutionCtx_t* ctx, hipDevResourceDesc_t desc, int device,
                             unsigned int flags) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipGreenCtxCreate_fn)(ctx, desc, device, flags);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxDestroy(hipExecutionCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxDestroy_fn)(ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxStreamCreate(hipStream_t* stream, hipExecutionCtx_t greenctx,
                                        unsigned int flags, int priority) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxStreamCreate_fn)(stream, greenctx, flags,
                                                                     priority);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetDevResource(hipDevice_t device, hipDevResource* resource,
                                   hipDevResourceType type) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetDevResource_fn)(device, resource, type);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevSmResourceSplitByCount(hipDevResource* result, unsigned int* nbGroups,
                                        const hipDevResource* input, hipDevResource* remainder,
                                        unsigned int flags, unsigned int minCount) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevSmResourceSplitByCount_fn)(result, nbGroups, input,
                                                                      remainder, flags, minCount);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevSmResourceSplit(hipDevResource* result, unsigned int nbGroups,
                                 const hipDevResource* input, hipDevResource* remainder,
                                 unsigned int flags,
                                 hipDevSmResourceGroupParams* groupParams) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevSmResourceSplit_fn)(result, nbGroups, input, remainder,
                                                              flags, groupParams);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDevResourceGenerateDesc(hipDevResourceDesc_t* phDesc, hipDevResource* resources,
                                       unsigned int nbResources) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDevResourceGenerateDesc_fn)(phDesc, resources, nbResources);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipDeviceGetExecutionCtx(hipExecutionCtx_t* ctx, int device) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceGetExecutionCtx_fn)(ctx, device);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxGetDevResource(hipExecutionCtx_t ctx, hipDevResource* resource,
                                          hipDevResourceType type) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxGetDevResource_fn)(ctx, resource, type);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxGetDevice(int* device, hipExecutionCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxGetDevice_fn)(device, ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxGetId(hipExecutionCtx_t ctx, unsigned long long* ctxId) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxGetId_fn)(ctx, ctxId);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipStreamGetDevResource(hipStream_t hStream, hipDevResource* resource,
                                    hipDevResourceType type) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipStreamGetDevResource_fn)(hStream, resource, type);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxRecordEvent(hipExecutionCtx_t ctx, hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxRecordEvent_fn)(ctx, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxSynchronize(hipExecutionCtx_t ctx) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxSynchronize_fn)(ctx);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipExecutionCtxWaitEvent(hipExecutionCtx_t ctx, hipEvent_t event) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipExecutionCtxWaitEvent_fn)(ctx, event);
  CATCH;
}
HIP_PUBLIC_API hipError_t hipMemGetDefaultMemPool(hipMemPool_t* memPool, hipMemLocation* location,
                                   hipMemAllocationType type) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipMemGetDefaultMemPool_fn)(memPool, location, type);
  CATCH;
}
hipError_t hipModuleEnumerateFunctions(hipFunction_t* functions, unsigned int numFunctions,
                                       hipModule_t mod) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipModuleEnumerateFunctions_fn)(functions, numFunctions, mod);
  CATCH;
}

hipError_t hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTarget target,
                                             hipFlushGPUDirectRDMAWritesScope scope) {
  TRY;
  return HIP_LOAD_DISPATCH_TABLE_FN(hipDeviceFlushGPUDirectRDMAWrites_fn)(target, scope);
  CATCH;
}
