/*
    Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/amd_detail/hip_api_trace.hpp>
#include "hip_internal.hpp"
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
#define CATCH } catch(...) { return hip::HandleException<hipError_t>(); }
#define CATCHRET(RETURN_TYPE) } catch(...) { return hip::HandleException<RETURN_TYPE>(); }

extern "C" hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                                hipStream_t* stream) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipPopCallConfiguration_fn(gridDim, blockDim,
                                                                          sharedMem, stream);
  CATCH;
}
extern "C" hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                                 hipStream_t stream) {
  TRY;
  return hip::GetHipCompilerDispatchTable()->__hipPushCallConfiguration_fn(gridDim, blockDim,
                                                                           sharedMem, stream);
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
  TRY;
  return hip::GetHipDispatchTable()->hipArray3DCreate_fn(array, pAllocateArray);
  CATCH;
}
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  TRY;
  return hip::GetHipDispatchTable()->hipArray3DGetDescriptor_fn(pArrayDescriptor, array);
  CATCH;
}
hipError_t hipArrayCreate(hipArray_t* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  TRY;
  return hip::GetHipDispatchTable()->hipArrayCreate_fn(pHandle, pAllocateArray);
  CATCH;
}
hipError_t hipArrayDestroy(hipArray_t array) {
  TRY;
  return hip::GetHipDispatchTable()->hipArrayDestroy_fn(array);
  CATCH;
}
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  TRY;
  return hip::GetHipDispatchTable()->hipArrayGetDescriptor_fn(pArrayDescriptor, array);
  CATCH;
}
hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags,
                           hipArray_t array) {
  TRY;
  return hip::GetHipDispatchTable()->hipArrayGetInfo_fn(desc, extent, flags, array);
  CATCH;
}
extern "C" hipError_t hipBindTexture(size_t* offset, const textureReference* tex,
                                     const void* devPtr, const hipChannelFormatDesc* desc,
                                     size_t size) {
  TRY;
  return hip::GetHipDispatchTable()->hipBindTexture_fn(offset, tex, devPtr, desc, size);
  CATCH;
}
hipError_t hipBindTexture2D(size_t* offset, const textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch) {
  TRY;
  return hip::GetHipDispatchTable()->hipBindTexture2D_fn(offset, tex, devPtr, desc, width, height,
                                                         pitch);
  CATCH;
}
hipError_t hipBindTextureToArray(const textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
  TRY;
  return hip::GetHipDispatchTable()->hipBindTextureToArray_fn(tex, array, desc);
  CATCH;
}
hipError_t hipBindTextureToMipmappedArray(const textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
  TRY;
  return hip::GetHipDispatchTable()->hipBindTextureToMipmappedArray_fn(tex, mipmappedArray, desc);
  CATCH;
}
extern "C" hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
  TRY;
  return hip::GetHipDispatchTable()->hipChooseDevice_fn(device, prop);
  CATCH;
}
extern "C" hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* properties) {
  TRY;
  return hip::GetHipDispatchTable()->hipChooseDeviceR0000_fn(device, properties);
  CATCH;
}
extern "C" hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                       hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipConfigureCall_fn(gridDim, blockDim, sharedMem, stream);
  CATCH;
}
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                  const hipResourceDesc* pResDesc) {
  TRY;
  return hip::GetHipDispatchTable()->hipCreateSurfaceObject_fn(pSurfObject, pResDesc);
  CATCH;
}
hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const struct hipResourceViewDesc* pResViewDesc) {
  TRY;
  return hip::GetHipDispatchTable()->hipCreateTextureObject_fn(pTexObject, pResDesc, pTexDesc,
                                                               pResViewDesc);
  CATCH;
}
extern "C" hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxCreate_fn(ctx, flags, device);
  CATCH;
}
extern "C" hipError_t hipCtxDestroy(hipCtx_t ctx) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxDestroy_fn(ctx);
  CATCH;
}
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxDisablePeerAccess_fn(peerCtx);
  CATCH;
}
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxEnablePeerAccess_fn(peerCtx, flags);
  CATCH;
}
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int* apiVersion) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxGetApiVersion_fn(ctx, apiVersion);
  CATCH;
}
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxGetCacheConfig_fn(cacheConfig);
  CATCH;
}
hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxGetCurrent_fn(ctx);
  CATCH;
}
hipError_t hipCtxGetDevice(hipDevice_t* device) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxGetDevice_fn(device);
  CATCH;
}
hipError_t hipCtxGetFlags(unsigned int* flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxGetFlags_fn(flags);
  CATCH;
}
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxGetSharedMemConfig_fn(pConfig);
  CATCH;
}
hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxPopCurrent_fn(ctx);
  CATCH;
}
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxPushCurrent_fn(ctx);
  CATCH;
}
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxSetCacheConfig_fn(cacheConfig);
  CATCH;
}
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxSetCurrent_fn(ctx);
  CATCH;
}
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  TRY;
  return hip::GetHipDispatchTable()->hipCtxSetSharedMemConfig_fn(config);
  CATCH;
}
hipError_t hipCtxSynchronize(void) { 
  TRY;
  return hip::GetHipDispatchTable()->hipCtxSynchronize_fn(); 
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
  TRY;
  return hip::GetHipDispatchTable()->hipDestroySurfaceObject_fn(surfaceObject);
  CATCH;
}
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipDestroyTextureObject_fn(textureObject);
  CATCH;
}
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceCanAccessPeer_fn(canAccessPeer, deviceId,
                                                               peerDeviceId);
  CATCH;
}
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceComputeCapability_fn(major, minor, device);
  CATCH;
}
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceDisablePeerAccess_fn(peerDeviceId);
  CATCH;
}
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceEnablePeerAccess_fn(peerDeviceId, flags);
  CATCH;
}
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGet_fn(device, ordinal);
  CATCH;
}
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetAttribute_fn(pi, attr, deviceId);
  CATCH;
}
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetByPCIBusId_fn(device, pciBusId);
  CATCH;
}
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetCacheConfig_fn(cacheConfig);
  CATCH;
}
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetDefaultMemPool_fn(mem_pool, device);
  CATCH;
}
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetGraphMemAttribute_fn(device, attr, value);
  CATCH;
}
hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetLimit_fn(pValue, limit);
  CATCH;
}
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetMemPool_fn(mem_pool, device);
  CATCH;
}
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetName_fn(name, len, device);
  CATCH;
}
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice,
                                    int dstDevice) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetP2PAttribute_fn(value, attr, srcDevice, dstDevice);
  CATCH;
}
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetPCIBusId_fn(pciBusId, len, device);
  CATCH;
}
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetSharedMemConfig_fn(pConfig);
  CATCH;
}
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetStreamPriorityRange_fn(leastPriority,
                                                                        greatestPriority);
  CATCH;
}
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetUuid_fn(uuid, device);
  CATCH;
}
hipError_t hipDeviceGraphMemTrim(int device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGraphMemTrim_fn(device);
  CATCH;
}
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
  TRY;
  return hip::GetHipDispatchTable()->hipDevicePrimaryCtxGetState_fn(dev, flags, active);
  CATCH;
}
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  TRY;
  return hip::GetHipDispatchTable()->hipDevicePrimaryCtxRelease_fn(dev);
  CATCH;
}
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  TRY;
  return hip::GetHipDispatchTable()->hipDevicePrimaryCtxReset_fn(dev);
  CATCH;
}
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  TRY;
  return hip::GetHipDispatchTable()->hipDevicePrimaryCtxRetain_fn(pctx, dev);
  CATCH;
}
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipDevicePrimaryCtxSetFlags_fn(dev, flags);
  CATCH;
}
hipError_t hipDeviceReset(void) { 
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceReset_fn(); 
  CATCH;
}
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceSetCacheConfig_fn(cacheConfig);
  CATCH;
}
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceSetGraphMemAttribute_fn(device, attr, value);
  CATCH;
}
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceSetLimit_fn(limit, value);
  CATCH;
}
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceSetMemPool_fn(device, mem_pool);
  CATCH;
}
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceSetSharedMemConfig_fn(config);
  CATCH;
}
hipError_t hipDeviceSynchronize(void) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceSynchronize_fn();
  CATCH;
}
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceTotalMem_fn(bytes, device);
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
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGraphAddMemcpyNode_fn(phGraphNode, hGraph, dependencies,
                                                                 numDependencies, copyParams, ctx);
  CATCH;
}
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvMemcpy2DUnaligned_fn(pCopy);
  CATCH;
}
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvMemcpy3D_fn(pCopy);
  CATCH;
}
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvMemcpy3DAsync_fn(pCopy, stream);
  CATCH;
}
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes,
                                      void** data, hipDeviceptr_t ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvPointerGetAttributes_fn(numAttributes, attributes, data,
                                                                   ptr);
  CATCH;
}
hipError_t hipEventCreate(hipEvent_t* event) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventCreate_fn(event);
  CATCH;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventCreateWithFlags_fn(event, flags);
  CATCH;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventDestroy_fn(event);
  CATCH;
}
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventElapsedTime_fn(ms, start, stop);
  CATCH;
}
hipError_t hipEventQuery(hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventQuery_fn(event);
  CATCH;
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventRecord_fn(event, stream);
  CATCH;
}
hipError_t hipEventSynchronize(hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventSynchronize_fn(event);
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
  TRY;
  return hip::GetHipDispatchTable()->hipExtLaunchKernel_fn(function_address, numBlocks, dimBlocks,
                                                           args, sharedMemBytes, stream, startEvent,
                                                           stopEvent, flags);
  CATCH;
}
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                              unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipExtLaunchMultiKernelMultiDevice_fn(launchParamsList,
                                                                           numDevices, flags);
  CATCH;
}
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipExtMallocWithFlags_fn(ptr, sizeBytes, flags);
  CATCH;
}
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask) {
  TRY;
  return hip::GetHipDispatchTable()->hipExtStreamCreateWithCUMask_fn(stream, cuMaskSize, cuMask);
  CATCH;
}
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask) {
  TRY;
  return hip::GetHipDispatchTable()->hipExtStreamGetCUMask_fn(stream, cuMaskSize, cuMask);
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
  TRY;
  return hip::GetHipDispatchTable()->hipFree_fn(ptr); 
  CATCH;
}
hipError_t hipFreeArray(hipArray_t array) {
  TRY;
  return hip::GetHipDispatchTable()->hipFreeArray_fn(array);
  CATCH;
}
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipFreeAsync_fn(dev_ptr, stream);
  CATCH;
}
hipError_t hipFreeHost(void* ptr) { 
  TRY;
  return hip::GetHipDispatchTable()->hipFreeHost_fn(ptr); 
  CATCH;
}
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  TRY;
  return hip::GetHipDispatchTable()->hipFreeMipmappedArray_fn(mipmappedArray);
  CATCH;
}
hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
  TRY;
  return hip::GetHipDispatchTable()->hipFuncGetAttribute_fn(value, attrib, hfunc);
  CATCH;
}
hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func) {
  TRY;
  return hip::GetHipDispatchTable()->hipFuncGetAttributes_fn(attr, func);
  CATCH;
}
hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value) {
  TRY;
  return hip::GetHipDispatchTable()->hipFuncSetAttribute_fn(func, attr, value);
  CATCH;
}
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config) {
  TRY;
  return hip::GetHipDispatchTable()->hipFuncSetCacheConfig_fn(func, config);
  CATCH;
}
hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config) {
  TRY;
  return hip::GetHipDispatchTable()->hipFuncSetSharedMemConfig_fn(func, config);
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
  TRY;
  return hip::GetHipDispatchTable()->hipGetChannelDesc_fn(desc, array);
  CATCH;
}
hipError_t hipGetDevice(int* deviceId) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetDevice_fn(deviceId);
  CATCH;
}
hipError_t hipGetDeviceCount(int* count) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetDeviceCount_fn(count);
  CATCH;
}
hipError_t hipGetDeviceFlags(unsigned int* flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetDeviceFlags_fn(flags);
  CATCH;
}
extern "C" hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int deviceId) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetDevicePropertiesR0600_fn(prop, deviceId);
  CATCH;
}
extern "C" hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetDevicePropertiesR0000_fn(prop, device);
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
  TRY;
  return hip::GetHipDispatchTable()->hipGetMipmappedArrayLevel_fn(levelArray, mipmappedArray,
                                                                  level);
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
  TRY;
  return hip::GetHipDispatchTable()->hipGetTextureAlignmentOffset_fn(offset, texref);
  CATCH;
}
hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetTextureObjectResourceDesc_fn(pResDesc, textureObject);
  CATCH;
}
hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetTextureObjectResourceViewDesc_fn(pResViewDesc,
                                                                            textureObject);
  CATCH;
}
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetTextureObjectTextureDesc_fn(pTexDesc, textureObject);
  CATCH;
}
hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetTextureReference_fn(texref, symbol);
  CATCH;
}
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddChildGraphNode_fn(pGraphNode, graph, pDependencies,
                                                                  numDependencies, childGraph);
  CATCH;
}
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddDependencies_fn(graph, from, to, numDependencies);
  CATCH;
}
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddEmptyNode_fn(pGraphNode, graph, pDependencies,
                                                             numDependencies);
  CATCH;
}
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddEventRecordNode_fn(pGraphNode, graph, pDependencies,
                                                                   numDependencies, event);
  CATCH;
}
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddEventWaitNode_fn(pGraphNode, graph, pDependencies,
                                                                 numDependencies, event);
  CATCH;
}
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddHostNode_fn(pGraphNode, graph, pDependencies,
                                                            numDependencies, pNodeParams);
  CATCH;
}
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddKernelNode_fn(pGraphNode, graph, pDependencies,
                                                              numDependencies, pNodeParams);
  CATCH;
}
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   hipMemAllocNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddMemAllocNode_fn(pGraphNode, graph, pDependencies,
                                                                numDependencies, pNodeParams);
  CATCH;
}
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  void* dev_ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddMemFreeNode_fn(pGraphNode, graph, pDependencies,
                                                               numDependencies, dev_ptr);
  CATCH;
}
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddMemcpyNode_fn(pGraphNode, graph, pDependencies,
                                                              numDependencies, pCopyParams);
  CATCH;
}
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddMemcpyNode1D_fn(
      pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
  CATCH;
}
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddMemcpyNodeFromSymbol_fn(
      pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
  CATCH;
}
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddMemcpyNodeToSymbol_fn(
      pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
  CATCH;
}
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddMemsetNode_fn(pGraphNode, graph, pDependencies,
                                                              numDependencies, pMemsetParams);
  CATCH;
}
hipError_t hipGraphAddNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                           const hipGraphNode_t* pDependencies, size_t numDependencies,
                           hipGraphNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddNode_fn(pGraphNode, graph, pDependencies,
                                                        numDependencies, nodeParams);
  CATCH;
}
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphChildGraphNodeGetGraph_fn(node, pGraph);
  CATCH;
}
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphClone_fn(pGraphClone, originalGraph);
  CATCH;
}
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphCreate_fn(pGraph, flags);
  CATCH;
}
hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphDebugDotPrint_fn(graph, path, flags);
  CATCH;
}
hipError_t hipGraphDestroy(hipGraph_t graph) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphDestroy_fn(graph);
  CATCH;
}
hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphDestroyNode_fn(node);
  CATCH;
}
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphEventRecordNodeGetEvent_fn(node, event_out);
  CATCH;
}
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphEventRecordNodeSetEvent_fn(node, event);
  CATCH;
}
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphEventWaitNodeGetEvent_fn(node, event_out);
  CATCH;
}
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphEventWaitNodeSetEvent_fn(node, event);
  CATCH;
}
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecChildGraphNodeSetParams_fn(hGraphExec, node,
                                                                            childGraph);
  CATCH;
}
hipError_t hipGraphExecNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                     hipGraphNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecNodeSetParams_fn(hGraphExec, node, nodeParams);
  CATCH;
}
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecDestroy_fn(graphExec);
  CATCH;
}
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecEventRecordNodeSetEvent_fn(hGraphExec, hNode,
                                                                            event);
  CATCH;
}
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecEventWaitNodeSetEvent_fn(hGraphExec, hNode, event);
  CATCH;
}
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecHostNodeSetParams_fn(hGraphExec, node,
                                                                      pNodeParams);
  CATCH;
}
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecKernelNodeSetParams_fn(hGraphExec, node,
                                                                        pNodeParams);
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParams_fn(hGraphExec, node,
                                                                        pNodeParams);
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParams1D_fn(hGraphExec, node, dst,
                                                                          src, count, kind);
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParamsFromSymbol_fn(
      hGraphExec, node, dst, symbol, count, offset, kind);
  CATCH;
}
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset,
                                                   hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecMemcpyNodeSetParamsToSymbol_fn(
      hGraphExec, node, symbol, src, count, offset, kind);
  CATCH;
}
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecMemsetNodeSetParams_fn(hGraphExec, node,
                                                                        pNodeParams);
  CATCH;
}
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecUpdate_fn(hGraphExec, hGraph, hErrorNode_out,
                                                           updateResult_out);
  CATCH;
}
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphGetEdges_fn(graph, from, to, numEdges);
  CATCH;
}
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphGetNodes_fn(graph, nodes, numNodes);
  CATCH;
}
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes) {
  return hip::GetHipDispatchTable()->hipGraphGetRootNodes_fn(graph, pRootNodes, pNumRootNodes);
}
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphHostNodeGetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphHostNodeSetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphInstantiate_fn(pGraphExec, graph, pErrorNode,
                                                            pLogBuffer, bufferSize);
  CATCH;
}
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphInstantiateWithFlags_fn(pGraphExec, graph, flags);
  CATCH;
}
hipError_t hipGraphInstantiateWithParams(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                         hipGraphInstantiateParams* instantiateParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphInstantiateWithParams_fn(pGraphExec, graph,
                                                                      instantiateParams);
  CATCH;
}
hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphKernelNodeCopyAttributes_fn(hSrc, hDst);
  CATCH;
}
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value) {
  return hip::GetHipDispatchTable()->hipGraphKernelNodeGetAttribute_fn(hNode, attr, value);
}
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphKernelNodeGetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphKernelNodeSetAttribute_fn(hNode, attr, value);
  CATCH;
}
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                       const hipKernelNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphKernelNodeSetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphLaunch_fn(graphExec, stream);
  CATCH;
}
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemAllocNodeGetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemFreeNodeGetParams_fn(node, dev_ptr);
  CATCH;
}
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemcpyNodeGetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParams1D_fn(node, dst, src, count, kind);
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParamsFromSymbol_fn(node, dst, symbol,
                                                                              count, offset, kind);
  CATCH;
}
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemcpyNodeSetParamsToSymbol_fn(node, symbol, src,
                                                                            count, offset, kind);
  CATCH;
}
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemsetNodeGetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphMemsetNodeSetParams_fn(node, pNodeParams);
  CATCH;
}
hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphNodeFindInClone_fn(pNode, originalNode, clonedGraph);
  CATCH;
}
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphNodeGetDependencies_fn(node, pDependencies,
                                                                    pNumDependencies);
  CATCH;
}
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphNodeGetDependentNodes_fn(node, pDependentNodes,
                                                                      pNumDependentNodes);
  CATCH;
}
hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphNodeGetEnabled_fn(hGraphExec, hNode, isEnabled);
  CATCH;
}
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphNodeGetType_fn(node, pType);
  CATCH;
}
hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled) {
  return hip::GetHipDispatchTable()->hipGraphNodeSetEnabled_fn(hGraphExec, hNode, isEnabled);
}
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphReleaseUserObject_fn(graph, object, count);
  CATCH;
}
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphRemoveDependencies_fn(graph, from, to,
                                                                   numDependencies);
  CATCH;
}
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count,
                                    unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphRetainUserObject_fn(graph, object, count, flags);
  CATCH;
}
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphUpload_fn(graphExec, stream);
  CATCH;
}
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphicsGLRegisterBuffer_fn(resource, buffer, flags);
  CATCH;
}
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphicsGLRegisterImage_fn(resource, image, target, flags);
  CATCH;
}
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphicsMapResources_fn(count, resources, stream);
  CATCH;
}
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphicsResourceGetMappedPointer_fn(devPtr, size, resource);
  CATCH;
}
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource,
                                                unsigned int arrayIndex, unsigned int mipLevel) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphicsSubResourceGetMappedArray_fn(array, resource,
                                                                             arrayIndex, mipLevel);
  CATCH;
}
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphicsUnmapResources_fn(count, resources, stream);
  CATCH;
}
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphicsUnregisterResource_fn(resource);
  CATCH;
}
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipHostAlloc_fn(ptr, size, flags);
  CATCH;
}
hipError_t hipHostFree(void* ptr) { 
  TRY;
  return hip::GetHipDispatchTable()->hipHostFree_fn(ptr); 
  CATCH;
}
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipHostGetDevicePointer_fn(devPtr, hstPtr, flags);
  CATCH;
}
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  TRY;
  return hip::GetHipDispatchTable()->hipHostGetFlags_fn(flagsPtr, hostPtr);
  CATCH;
}
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipHostMalloc_fn(ptr, size, flags);
  CATCH;
}
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipHostRegister_fn(hostPtr, sizeBytes, flags);
  CATCH;
}
hipError_t hipHostUnregister(void* hostPtr) {
  TRY;
  return hip::GetHipDispatchTable()->hipHostUnregister_fn(hostPtr);
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
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGraphAddMemsetNode_fn(
      phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
  CATCH;
}
hipError_t hipInit(unsigned int flags) { 
  TRY;
  return hip::GetHipDispatchTable()->hipInit_fn(flags); 
  CATCH;
}
hipError_t hipIpcCloseMemHandle(void* devPtr) {
  TRY;
  return hip::GetHipDispatchTable()->hipIpcCloseMemHandle_fn(devPtr);
  CATCH;
}
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  TRY;
  return hip::GetHipDispatchTable()->hipIpcGetEventHandle_fn(handle, event);
  CATCH;
}
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
  TRY;
  return hip::GetHipDispatchTable()->hipIpcGetMemHandle_fn(handle, devPtr);
  CATCH;
}
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  TRY;
  return hip::GetHipDispatchTable()->hipIpcOpenEventHandle_fn(event, handle);
  CATCH;
}
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipIpcOpenMemHandle_fn(devPtr, handle, flags);
  CATCH;
}
extern "C" const char* hipKernelNameRef(const hipFunction_t f) {
  TRY;
  return hip::GetHipDispatchTable()->hipKernelNameRef_fn(f);
  CATCHRET(const char*);
}
extern "C" const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipKernelNameRefByPtr_fn(hostFunction, stream);
  CATCHRET(const char*);
}
extern "C" hipError_t hipLaunchByPtr(const void* func) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchByPtr_fn(func);
  CATCH;
}
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchCooperativeKernel_fn(
      f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream);
  CATCH;
}
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                                 unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchCooperativeKernelMultiDevice_fn(launchParamsList,
                                                                              numDevices, flags);
  CATCH;
}
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchHostFunc_fn(stream, fn, userData);
  CATCH;
}
extern "C" hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                                      void** args, size_t sharedMemBytes, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchKernel_fn(function_address, numBlocks, dimBlocks,
                                                        args, sharedMemBytes, stream);
  CATCH;
}
hipError_t hipMalloc(void** ptr, size_t size) {
  TRY;
  return hip::GetHipDispatchTable()->hipMalloc_fn(ptr, size);
  CATCH;
}
hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  TRY;
  return hip::GetHipDispatchTable()->hipMalloc3D_fn(pitchedDevPtr, extent);
  CATCH;
}
extern "C" hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                                       struct hipExtent extent, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMalloc3DArray_fn(array, desc, extent, flags);
  CATCH;
}
extern "C" hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc,
                                     size_t width, size_t height, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMallocArray_fn(array, desc, width, height, flags);
  CATCH;
}
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMallocAsync_fn(dev_ptr, size, stream);
  CATCH;
}
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream) {
  return hip::GetHipDispatchTable()->hipMallocFromPoolAsync_fn(dev_ptr, size, mem_pool, stream);
}
hipError_t hipMallocHost(void** ptr, size_t size) {
  TRY;
  return hip::GetHipDispatchTable()->hipMallocHost_fn(ptr, size);
  CATCH;
}
hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMallocManaged_fn(dev_ptr, size, flags);
  CATCH;
}
extern "C" hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,
                                              const struct hipChannelFormatDesc* desc,
                                              struct hipExtent extent, unsigned int numLevels,
                                              unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMallocMipmappedArray_fn(mipmappedArray, desc, extent,
                                                                numLevels, flags);
  CATCH;
}
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  TRY;
  return hip::GetHipDispatchTable()->hipMallocPitch_fn(ptr, pitch, width, height);
  CATCH;
}
hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemAddressFree_fn(devPtr, size);
  CATCH;
}
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags) {
  return hip::GetHipDispatchTable()->hipMemAddressReserve_fn(ptr, size, alignment, addr, flags);
}
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemAdvise_fn(dev_ptr, count, advice, device);
  CATCH;
}
hipError_t hipMemAdvise_v2(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                           hipMemLocation location) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemAdvise_v2_fn(dev_ptr, count, advice, location);
  CATCH;
}
hipError_t hipMemAllocHost(void** ptr, size_t size) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemAllocHost_fn(ptr, size);
  CATCH;
}
hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemAllocPitch_fn(dptr, pitch, widthInBytes, height,
                                                         elementSizeBytes);
  CATCH;
}
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemCreate_fn(handle, size, prop, flags);
  CATCH;
}
hipError_t hipMemExportToShareableHandle(void* shareableHandle,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType handleType,
                                         unsigned long long flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemExportToShareableHandle_fn(shareableHandle, handle,
                                                                      handleType, flags);
  CATCH;
}
hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetAccess_fn(flags, location, ptr);
  CATCH;
}
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetAddressRange_fn(pbase, psize, dptr);
  CATCH;
}
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetAllocationGranularity_fn(granularity, prop, option);
  CATCH;
}
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                   hipMemGenericAllocationHandle_t handle) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetAllocationPropertiesFromHandle_fn(prop, handle);
  CATCH;
}
hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetInfo_fn(free, total);
  CATCH;
}
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle,
                                           hipMemAllocationHandleType shHandleType) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemImportFromShareableHandle_fn(handle, osHandle,
                                                                        shHandleType);
  CATCH;
}
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemMap_fn(ptr, size, offset, handle, flags);
  CATCH;
}
hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count,
                               hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemMapArrayAsync_fn(mapInfoList, count, stream);
  CATCH;
}
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolCreate_fn(mem_pool, pool_props);
  CATCH;
}
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolDestroy_fn(mem_pool);
  CATCH;
}
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolExportPointer_fn(export_data, dev_ptr);
  CATCH;
}
hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool,
                                             hipMemAllocationHandleType handle_type,
                                             unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolExportToShareableHandle_fn(shared_handle, mem_pool,
                                                                          handle_type, flags);
  CATCH;
}
hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                               hipMemLocation* location) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolGetAccess_fn(flags, mem_pool, location);
  CATCH;
}
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolGetAttribute_fn(mem_pool, attr, value);
  CATCH;
}
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle,
                                               hipMemAllocationHandleType handle_type,
                                               unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolImportFromShareableHandle_fn(mem_pool, shared_handle,
                                                                            handle_type, flags);
  CATCH;
}
hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool,
                                   hipMemPoolPtrExportData* export_data) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolImportPointer_fn(dev_ptr, mem_pool, export_data);
  CATCH;
}
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list,
                               size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolSetAccess_fn(mem_pool, desc_list, count);
  CATCH;
}
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolSetAttribute_fn(mem_pool, attr, value);
  CATCH;
}
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPoolTrimTo_fn(mem_pool, min_bytes_to_hold);
  CATCH;
}
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPrefetchAsync_fn(dev_ptr, count, device, stream);
  CATCH;
}
hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count, hipMemLocation location,
                                  unsigned int flags, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPrefetchAsync_v2_fn(dev_ptr, count, location, flags,
                                                               stream);
  CATCH;
}
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemPtrGetInfo_fn(ptr, size);
  CATCH;
}
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemRangeGetAttribute_fn(data, data_size, attribute, dev_ptr,
                                                                count);
  CATCH;
}
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                    hipMemRangeAttribute* attributes, size_t num_attributes,
                                    const void* dev_ptr, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemRangeGetAttributes_fn(data, data_sizes, attributes,
                                                                 num_attributes, dev_ptr, count);
  CATCH;
}
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemRelease_fn(handle);
  CATCH;
}
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemRetainAllocationHandle_fn(handle, addr);
  CATCH;
}
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemSetAccess_fn(ptr, size, desc, count);
  CATCH;
}
hipError_t hipMemUnmap(void* ptr, size_t size) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemUnmap_fn(ptr, size);
  CATCH;
}
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy_fn(dst, src, sizeBytes, kind);
  CATCH;
}
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2D_fn(dst, dpitch, src, spitch, width, height, kind);
  CATCH;
}
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DAsync_fn(dst, dpitch, src, spitch, width, height,
                                                         kind, stream);
  CATCH;
}
hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DFromArray_fn(dst, dpitch, src, wOffset, hOffset,
                                                             width, height, kind);
  CATCH;
}
hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                     size_t hOffset, size_t width, size_t height,
                                     hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DFromArrayAsync_fn(
      dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
  CATCH;
}
hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DToArray_fn(dst, wOffset, hOffset, src, spitch,
                                                           width, height, kind);
  CATCH;
}
hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DToArrayAsync_fn(dst, wOffset, hOffset, src, spitch,
                                                                width, height, kind, stream);
  CATCH;
}
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy3D_fn(p);
  CATCH;
}
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy3DAsync_fn(p, stream);
  CATCH;
}
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyAsync_fn(dst, src, sizeBytes, kind, stream);
  CATCH;
}
hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray, size_t srcOffset, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyAtoH_fn(dst, srcArray, srcOffset, count);
  CATCH;
}
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyDtoD_fn(dst, src, sizeBytes);
  CATCH;
}
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyDtoDAsync_fn(dst, src, sizeBytes, stream);
  CATCH;
}
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyDtoH_fn(dst, src, sizeBytes);
  CATCH;
}
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyDtoHAsync_fn(dst, src, sizeBytes, stream);
  CATCH;
}
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyFromArray_fn(dst, srcArray, wOffset, hOffset, count,
                                                           kind);
  CATCH;
}
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                               hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyFromSymbol_fn(dst, symbol, sizeBytes, offset, kind);
  CATCH;
}
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyFromSymbolAsync_fn(dst, symbol, sizeBytes, offset,
                                                                 kind, stream);
  CATCH;
}
hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyHtoA_fn(dstArray, dstOffset, srcHost, count);
  CATCH;
}
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyHtoD_fn(dst, src, sizeBytes);
  CATCH;
}
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void* src, size_t sizeBytes,
                              hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyHtoDAsync_fn(dst, src, sizeBytes, stream);
  CATCH;
}
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyParam2D_fn(pCopy);
  CATCH;
}
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyParam2DAsync_fn(pCopy, stream);
  CATCH;
}
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                         size_t sizeBytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyPeer_fn(dst, dstDeviceId, src, srcDeviceId,
                                                      sizeBytes);
  CATCH;
}
hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyPeerAsync_fn(dst, dstDeviceId, src, srcDevice,
                                                           sizeBytes, stream);
  CATCH;
}
hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyToArray_fn(dst, wOffset, hOffset, src, count, kind);
  CATCH;
}
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyToSymbol_fn(symbol, src, sizeBytes, offset, kind);
  CATCH;
}
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyToSymbolAsync_fn(symbol, src, sizeBytes, offset, kind,
                                                               stream);
  CATCH;
}
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyWithStream_fn(dst, src, sizeBytes, kind, stream);
  CATCH;
}
hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset_fn(dst, value, sizeBytes);
  CATCH;
}
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset2D_fn(dst, pitch, value, width, height);
  CATCH;
}
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,
                            hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset2DAsync_fn(dst, pitch, value, width, height, stream);
  CATCH;
}
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset3D_fn(pitchedDevPtr, value, extent);
  CATCH;
}
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                            hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset3DAsync_fn(pitchedDevPtr, value, extent, stream);
  CATCH;
}
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetAsync_fn(dst, value, sizeBytes, stream);
  CATCH;
}
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD16_fn(dest, value, count);
  CATCH;
}
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count,
                             hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD16Async_fn(dest, value, count, stream);
  CATCH;
}
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD32_fn(dest, value, count);
  CATCH;
}
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD32Async_fn(dst, value, count, stream);
  CATCH;
}
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD8_fn(dest, value, count);
  CATCH;
}
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count,
                            hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD8Async_fn(dest, value, count, stream);
  CATCH;
}
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels) {
  TRY;
  return hip::GetHipDispatchTable()->hipMipmappedArrayCreate_fn(pHandle, pMipmappedArrayDesc,
                                                                numMipmapLevels);
  CATCH;
}
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
  TRY;
  return hip::GetHipDispatchTable()->hipMipmappedArrayDestroy_fn(hMipmappedArray);
  CATCH;
}
hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level) {
  TRY;
  return hip::GetHipDispatchTable()->hipMipmappedArrayGetLevel_fn(pLevelArray, hMipMappedArray,
                                                                  level);
  CATCH;
}
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleGetFunction_fn(function, module, kname);
  CATCH;
}
hipError_t hipModuleGetFunctionCount(unsigned int* count, hipModule_t mod) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleGetFunctionCount_fn(count, mod);
  CATCH;
}
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleGetGlobal_fn(dptr, bytes, hmod, name);
  CATCH;
}
hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleGetTexRef_fn(texRef, hmod, name);
  CATCH;
}
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleLaunchCooperativeKernel_fn(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
      kernelParams);
  CATCH;
}
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices,
                                                       unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleLaunchCooperativeKernelMultiDevice_fn(
      launchParamsList, numDevices, flags);
  CATCH;
}
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleLaunchKernel_fn(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
      kernelParams, extra);
  CATCH;
}
hipError_t hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleLoadFatBinary_fn(module, fatbin);
  CATCH;
}
hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleLoad_fn(module, fname);
  CATCH;
}
hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleLoadData_fn(module, image);
  CATCH;
}
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleLoadDataEx_fn(module, image, numOptions, options,
                                                            optionValues);
  CATCH;
}

hipError_t hipLinkAddData(hipLinkState_t state, hipJitInputType type, void* data, size_t size,
                          const char* name, unsigned int numOptions, hipJitOption* options,
                          void** optionValues) {
  TRY;
  return hip::GetHipDispatchTable()->hipLinkAddData_fn(state, type, data, size, name, numOptions,
                                                       options, optionValues);
  CATCH;
}

hipError_t hipLinkAddFile(hipLinkState_t state, hipJitInputType type, const char* path,
                          unsigned int numOptions, hipJitOption* options, void** optionValues) {
  TRY;
  return hip::GetHipDispatchTable()->hipLinkAddFile_fn(state, type, path, numOptions, options,
                                                       optionValues);
  CATCH;
}

hipError_t hipLinkComplete(hipLinkState_t state, void** hipBinOut, size_t* sizeOut) {
  TRY;
  return hip::GetHipDispatchTable()->hipLinkComplete_fn(state, hipBinOut, sizeOut);
  CATCH;
}

hipError_t hipLinkCreate(unsigned int numOptions, hipJitOption* options, void** optionValues,
                         hipLinkState_t* stateOut) {
  TRY;
  return hip::GetHipDispatchTable()->hipLinkCreate_fn(numOptions, options, optionValues, stateOut);
  CATCH;
}

hipError_t hipLinkDestroy(hipLinkState_t state) {
  TRY;
  return hip::GetHipDispatchTable()->hipLinkDestroy_fn(state);
  CATCH;
}

extern "C" hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn(
      numBlocks, f, blockSize, dynSharedMemPerBlk);
  CATCH;
}
extern "C" hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn(
      numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
  CATCH;
}
extern "C" hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                              hipFunction_t f,
                                                              size_t dynSharedMemPerBlk,
                                                              int blockSizeLimit) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleOccupancyMaxPotentialBlockSize_fn(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  CATCH;
}
extern "C" hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit,
    unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags);
  CATCH;
}
hipError_t hipModuleUnload(hipModule_t module) {
  TRY;
  return hip::GetHipDispatchTable()->hipModuleUnload_fn(module);
  CATCH;
}
extern "C" hipError_t hipOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* f,
                                                               int numBlocks, int blockSize) {
  TRY;
  return hip::GetHipDispatchTable()->hipOccupancyAvailableDynamicSMemPerBlock_fn(dynamicSmemSize,
                                                                                 f, numBlocks,
                                                                                 blockSize);
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                                   int blockSize,
                                                                   size_t dynSharedMemPerBlk) {
  TRY;
  return hip::GetHipDispatchTable()->hipOccupancyMaxActiveBlocksPerMultiprocessor_fn(
      numBlocks, f, blockSize, dynSharedMemPerBlk);
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn(
      numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
  CATCH;
}
extern "C" hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                        const void* f, size_t dynSharedMemPerBlk,
                                                        int blockSizeLimit) {
  TRY;
  return hip::GetHipDispatchTable()->hipOccupancyMaxPotentialBlockSize_fn(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  CATCH;
}
hipError_t hipPeekAtLastError(void) { 
  TRY;
  return hip::GetHipDispatchTable()->hipPeekAtLastError_fn(); 
  CATCH;
}
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipPointerGetAttribute_fn(data, attribute, ptr);
  CATCH;
}
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipPointerGetAttributes_fn(attributes, ptr);
  CATCH;
}
hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipPointerSetAttribute_fn(value, attribute, ptr);
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
  TRY;
  return hip::GetHipDispatchTable()->hipSetDevice_fn(deviceId);
  CATCH;
}
hipError_t hipSetDeviceFlags(unsigned flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipSetDeviceFlags_fn(flags);
  CATCH;
}
extern "C" hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  TRY;
  return hip::GetHipDispatchTable()->hipSetupArgument_fn(arg, size, offset);
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
  TRY;
  return hip::GetHipDispatchTable()->hipStreamAddCallback_fn(stream, callback, userData, flags);
  CATCH;
}
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length,
                                   unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamAttachMemAsync_fn(stream, dev_ptr, length, flags);
  CATCH;
}
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamBeginCapture_fn(stream, mode);
  CATCH;
}
hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamCopyAttributes_fn(dst, src);
  CATCH;
}
hipError_t hipStreamCreate(hipStream_t* stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamCreate_fn(stream);
  CATCH;
}
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamCreateWithFlags_fn(stream, flags);
  CATCH;
}
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamCreateWithPriority_fn(stream, flags, priority);
  CATCH;
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamDestroy_fn(stream);
  CATCH;
}
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamEndCapture_fn(stream, pGraph);
  CATCH;
}
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_fn(stream, pCaptureStatus, pId);
  CATCH;
}
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_v2_fn(
      stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  CATCH;
}
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetDevice_fn(stream, device);
  CATCH;
}
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetFlags_fn(stream, flags);
  CATCH;
}
hipError_t hipStreamGetId(hipStream_t stream, unsigned long long* streamId) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetId_fn(stream, streamId);
  CATCH;
}
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetPriority_fn(stream, priority);
  CATCH;
}
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamIsCapturing_fn(stream, pCaptureStatus);
  CATCH;
}
hipError_t hipStreamQuery(hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamQuery_fn(stream);
  CATCH;
}
hipError_t hipStreamSynchronize(hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamSynchronize_fn(stream);
  CATCH;
}
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamUpdateCaptureDependencies_fn(stream, dependencies,
                                                                           numDependencies, flags);
  CATCH;
}
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamWaitEvent_fn(stream, event, flags);
  CATCH;
}
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamWaitValue32_fn(stream, ptr, value, flags, mask);
  CATCH;
}
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamWaitValue64_fn(stream, ptr, value, flags, mask);
  CATCH;
}
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value,
                                 unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamWriteValue32_fn(stream, ptr, value, flags);
  CATCH;
}
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value,
                                 unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamWriteValue64_fn(stream, ptr, value, flags);
  CATCH;
}
hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamBatchMemOp_fn(stream, count, paramArray, flags);
  CATCH;
}
hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject, const HIP_RESOURCE_DESC* pResDesc,
                              const HIP_TEXTURE_DESC* pTexDesc,
                              const HIP_RESOURCE_VIEW_DESC* pResViewDesc) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexObjectCreate_fn(pTexObject, pResDesc, pTexDesc,
                                                           pResViewDesc);
  CATCH;
}
hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexObjectDestroy_fn(texObject);
  CATCH;
}
hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, hipTextureObject_t texObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexObjectGetResourceDesc_fn(pResDesc, texObject);
  CATCH;
}
hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                           hipTextureObject_t texObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexObjectGetResourceViewDesc_fn(pResViewDesc, texObject);
  CATCH;
}
hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, hipTextureObject_t texObject) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexObjectGetTextureDesc_fn(pTexDesc, texObject);
  CATCH;
}
hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetAddress_fn(dev_ptr, texRef);
  CATCH;
}
hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode* pam, const textureReference* texRef,
                                   int dim) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetAddressMode_fn(pam, texRef, dim);
  CATCH;
}
hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode* pfm, const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetFilterMode_fn(pfm, texRef);
  CATCH;
}
hipError_t hipTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetFlags_fn(pFlags, texRef);
  CATCH;
}
hipError_t hipTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels,
                              const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetFormat_fn(pFormat, pNumChannels, texRef);
  CATCH;
}
hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetMaxAnisotropy_fn(pmaxAnsio, texRef);
  CATCH;
}
extern "C" hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,
                                                 const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetMipMappedArray_fn(pArray, texRef);
  CATCH;
}
hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode* pfm,
                                        const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetMipmapFilterMode_fn(pfm, texRef);
  CATCH;
}
hipError_t hipTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetMipmapLevelBias_fn(pbias, texRef);
  CATCH;
}
hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp,
                                        const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetMipmapLevelClamp_fn(pminMipmapLevelClamp,
                                                                     pmaxMipmapLevelClamp, texRef);
  CATCH;
}
hipError_t hipTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr,
                               size_t bytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetAddress_fn(ByteOffset, texRef, dptr, bytes);
  CATCH;
}
hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr, size_t Pitch) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetAddress2D_fn(texRef, desc, dptr, Pitch);
  CATCH;
}
hipError_t hipTexRefSetAddressMode(textureReference* texRef, int dim,
                                   enum hipTextureAddressMode am) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetAddressMode_fn(texRef, dim, am);
  CATCH;
}
hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetArray_fn(tex, array, flags);
  CATCH;
}
hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetBorderColor_fn(texRef, pBorderColor);
  CATCH;
}
hipError_t hipTexRefSetFilterMode(textureReference* texRef, enum hipTextureFilterMode fm) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetFilterMode_fn(texRef, fm);
  CATCH;
}
hipError_t hipTexRefSetFlags(textureReference* texRef, unsigned int Flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetFlags_fn(texRef, Flags);
  CATCH;
}
hipError_t hipTexRefSetFormat(textureReference* texRef, hipArray_Format fmt,
                              int NumPackedComponents) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetFormat_fn(texRef, fmt, NumPackedComponents);
  CATCH;
}
hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetMaxAnisotropy_fn(texRef, maxAniso);
  CATCH;
}
hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef, enum hipTextureFilterMode fm) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetMipmapFilterMode_fn(texRef, fm);
  CATCH;
}
hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetMipmapLevelBias_fn(texRef, bias);
  CATCH;
}
hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp,
                                        float maxMipMapLevelClamp) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetMipmapLevelClamp_fn(texRef, minMipMapLevelClamp,
                                                                     maxMipMapLevelClamp);
  CATCH;
}
hipError_t hipTexRefSetMipmappedArray(textureReference* texRef,
                                      struct hipMipmappedArray* mipmappedArray,
                                      unsigned int Flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefSetMipmappedArray_fn(texRef, mipmappedArray, Flags);
  CATCH;
}
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  TRY;
  return hip::GetHipDispatchTable()->hipThreadExchangeStreamCaptureMode_fn(mode);
  CATCH;
}
extern "C" hipError_t hipUnbindTexture(const textureReference* tex) {
  TRY;
  return hip::GetHipDispatchTable()->hipUnbindTexture_fn(tex);
  CATCH;
}
hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                               unsigned int initialRefcount, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipUserObjectCreate_fn(object_out, ptr, destroy,
                                                            initialRefcount, flags);
  CATCH;
}
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
  TRY;
  return hip::GetHipDispatchTable()->hipUserObjectRelease_fn(object, count);
  CATCH;
}
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
  TRY;
  return hip::GetHipDispatchTable()->hipUserObjectRetain_fn(object, count);
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
  TRY;
  return hip::GetHipDispatchTable()->hipCreateChannelDesc_fn(x, y, z, w, f);
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

hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy_spt_fn(dst, src, sizeBytes, kind);
  CATCH;
}
hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                                 size_t offset, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyToSymbol_spt_fn(symbol, src, sizeBytes, offset, kind);
  CATCH;
}
hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                   hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyFromSymbol_spt_fn(dst, symbol, sizeBytes, offset,
                                                                kind);
  CATCH;
}
hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                           size_t height, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2D_spt_fn(dst, dpitch, src, spitch, width, height,
                                                        kind);
  CATCH;
}
hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                    size_t hOffset, size_t width, size_t height,
                                    hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DFromArray_spt_fn(dst, dpitch, src, wOffset, hOffset,
                                                                 width, height, kind);
  CATCH;
}
hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy3D_spt_fn(p);
  CATCH;
}
hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset_spt_fn(dst, value, sizeBytes);
  CATCH;
}
hipError_t hipMemsetAsync_spt(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetAsync_spt_fn(dst, value, sizeBytes, stream);
  CATCH;
}
hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset2D_spt_fn(dst, pitch, value, width, height);
  CATCH;
}
hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value, size_t width, size_t height,
                                hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset2DAsync_spt_fn(dst, pitch, value, width, height,
                                                             stream);
  CATCH;
}
hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset3DAsync_spt_fn(pitchedDevPtr, value, extent, stream);
  CATCH;
}
hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemset3D_spt_fn(pitchedDevPtr, value, extent);
  CATCH;
}
hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                              hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyAsync_spt_fn(dst, src, sizeBytes, kind, stream);
  CATCH;
}
hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms* p, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy3DAsync_spt_fn(p, stream);
  CATCH;
}
hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, hipMemcpyKind kind,
                                hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DAsync_spt_fn(dst, dpitch, src, spitch, width,
                                                             height, kind, stream);
  CATCH;
}
hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t sizeBytes,
                                        size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyFromSymbolAsync_spt_fn(dst, symbol, sizeBytes, offset,
                                                                     kind, stream);
  CATCH;
}
hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t sizeBytes,
                                      size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyToSymbolAsync_spt_fn(symbol, src, sizeBytes, offset,
                                                                   kind, stream);
  CATCH;
}
hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                  size_t hOffset, size_t count, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyFromArray_spt_fn(dst, src, wOffsetSrc, hOffset, count,
                                                               kind);
  CATCH;
}
hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DToArray_spt_fn(dst, wOffset, hOffset, src, spitch,
                                                               width, height, kind);
  CATCH;
}
hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                         size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                         size_t height, hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DFromArrayAsync_spt_fn(
      dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height, kind, stream);
  CATCH;
}
hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset,
                                       const void* src, size_t spitch, size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DToArrayAsync_spt_fn(
      dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
  CATCH;
}
hipError_t hipStreamQuery_spt(hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamQuery_spt_fn(stream);
  CATCH;
}
hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamSynchronize_spt_fn(stream);
  CATCH;
}
hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetPriority_spt_fn(stream, priority);
  CATCH;
}
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamWaitEvent_spt_fn(stream, event, flags);
  CATCH;
}
hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetFlags_spt_fn(stream, flags);
  CATCH;
}
hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback,
                                    void* userData, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamAddCallback_spt_fn(stream, callback, userData, flags);
  CATCH;
}
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventRecord_spt_fn(event, stream);
  CATCH;
}
hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim, dim3 blockDim,
                                          void** kernelParams, uint32_t sharedMemBytes,
                                          hipStream_t hStream) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchCooperativeKernel_spt_fn(
      f, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);
  CATCH;
}

extern "C" hipError_t hipLaunchKernel_spt(const void* function_address, dim3 numBlocks,
                                          dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                          hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchKernel_spt_fn(function_address, numBlocks, dimBlocks,
                                                            args, sharedMemBytes, stream);
  CATCH;
}

hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphLaunch_spt_fn(graphExec, stream);
  CATCH;
}
hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamBeginCapture_spt_fn(stream, mode);
  CATCH;
}
hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamEndCapture_spt_fn(stream, pGraph);
  CATCH;
}
hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamIsCapturing_spt_fn(stream, pCaptureStatus);
  CATCH;
}
hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                       unsigned long long* pId) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_spt_fn(stream, pCaptureStatus, pId);
  CATCH;
}
hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,
                                          hipStreamCaptureStatus* captureStatus_out,
                                          unsigned long long* id_out, hipGraph_t* graph_out,
                                          const hipGraphNode_t** dependencies_out,
                                          size_t* numDependencies_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetCaptureInfo_v2_spt_fn(
      stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  CATCH;
}
hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void* userData) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchHostFunc_spt_fn(stream, fn, userData);
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
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetBorderColor_fn(pBorderColor, texRef);
  CATCH;
}
hipError_t hipTexRefGetArray(hipArray_t* pArray, const textureReference* texRef) {
  TRY;
  return hip::GetHipDispatchTable()->hipTexRefGetArray_fn(pArray, texRef);
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
  TRY;
  return hip::GetHipDispatchTable()->hipStreamBeginCaptureToGraph_fn(
      stream, graph, dependencies, dependencyData, numDependencies, mode);
  CATCH;
}
hipError_t hipGetFuncBySymbol(hipFunction_t* functionPtr, const void* symbolPtr) {
  TRY;
  return hip::GetHipDispatchTable()->hipGetFuncBySymbol_fn(functionPtr, symbolPtr);
  CATCH;
}
hipError_t hipDrvGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGraphExecMemsetNodeSetParams_fn(hGraphExec, hNode,
                                                                           memsetParams, ctx);
  CATCH;
}
hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec, unsigned long long* flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecGetFlags_fn(graphExec, flags);
  CATCH;
}
hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     hipDeviceptr_t dptr) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGraphAddMemFreeNode_fn(phGraphNode, hGraph, dependencies,
                                                                  numDependencies, dptr);
  CATCH;
}
hipError_t hipDrvGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGraphExecMemcpyNodeSetParams_fn(hGraphExec, hNode,
                                                                           copyParams, ctx);
  CATCH;
}
hipError_t hipSetValidDevices(int* device_arr, int len) {
  TRY;
  return hip::GetHipDispatchTable()->hipSetValidDevices_fn(device_arr, len);
  CATCH;
}
hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset,
                         size_t ByteCount) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyAtoD_fn(dstDevice, srcArray, srcOffset, ByteCount);
  CATCH;
}
hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice,
                         size_t ByteCount) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyDtoA_fn(dstArray, dstOffset, srcDevice, ByteCount);
  CATCH;
}
hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray,
                         size_t srcOffset, size_t ByteCount) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyAtoA_fn(dstArray, dstOffset, srcArray, srcOffset,
                                                      ByteCount);
  CATCH;
}
hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray, size_t srcOffset,
                              size_t ByteCount, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyAtoHAsync_fn(dstHost, srcArray, srcOffset, ByteCount,
                                                           stream);
  CATCH;
}
hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                              size_t ByteCount, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyHtoAAsync_fn(dstArray, dstOffset, srcHost, ByteCount,
                                                           stream);
  CATCH;
}
hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                   hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t width, size_t height, hipMemcpyKind kind) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy2DArrayToArray_fn(
      dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
  CATCH;
}
hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGraphMemcpyNodeGetParams_fn(hNode, nodeParams);
  CATCH;
}
hipError_t hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t hNode, const HIP_MEMCPY3D* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvGraphMemcpyNodeSetParams_fn(hNode, nodeParams);
  CATCH;
}
hipError_t hipGraphNodeSetParams(hipGraphNode_t node, hipGraphNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphNodeSetParams_fn(node, nodeParams);
  CATCH;
}
hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     const hipBatchMemOpNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddBatchMemOpNode_fn(pGraphNode, graph, dependencies,
                                                                  numDependencies, nodeParams);
  CATCH;
}
hipError_t hipGraphBatchMemOpNodeGetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphBatchMemOpNodeGetParams_fn(hNode, nodeParams_out);
  CATCH;
}
hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphBatchMemOpNodeSetParams_fn(hNode, nodeParams);
  CATCH;
}
hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               const hipBatchMemOpNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecBatchMemOpNodeSetParams_fn(hGraphExec, hNode,
                                                                            nodeParams);
  CATCH;
}
hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipEventRecordWithFlags_fn(event, stream, flags);
  CATCH;
}

hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, const void* fPtr, void** args) {
  TRY;
  return hip::GetHipDispatchTable()->hipLaunchKernelExC_fn(config, fPtr, args);
  CATCH;
}

hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, hipFunction_t f, void** kernel,
                                void** extra) {
  TRY;
  return hip::GetHipDispatchTable()->hipDrvLaunchKernelEx_fn(config, f, kernel, extra);
  CATCH;
}

hipError_t hipMemGetHandleForAddressRange(void* handle, hipDeviceptr_t dptr, size_t size,
                                          hipMemRangeHandleType handleType,
                                          unsigned long long flags) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetHandleForAddressRange_fn(handle, dptr, size,
                                                                       handleType, flags);
  CATCH;
}
hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch, unsigned char value, size_t width,
                         size_t height) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD2D8_fn(dst, dstPitch, value, width, height);
  CATCH;
}
hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
                              size_t width, size_t height, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD2D8Async_fn(dst, dstPitch, value, width, height,
                                                           stream);
  CATCH;
}
hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch, unsigned short value, size_t width,
                          size_t height) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD2D16_fn(dst, dstPitch, value, width, height);
  CATCH;
}
hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
                               size_t width, size_t height, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD2D16Async_fn(dst, dstPitch, value, width, height,
                                                            stream);
  CATCH;
}
hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width,
                          size_t height) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD2D32_fn(dst, dstPitch, value, width, height);
  CATCH;
}
hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch, unsigned int value,
                               size_t width, size_t height, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemsetD2D32Async_fn(dst, dstPitch, value, width, height,
                                                            stream);
  CATCH;
}
hipError_t hipStreamSetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 const hipStreamAttrValue* value) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamSetAttribute_fn(stream, attr, value);
  CATCH;
}
hipError_t hipStreamGetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 hipStreamAttrValue* value) {
  TRY;
  return hip::GetHipDispatchTable()->hipStreamGetAttribute_fn(stream, attr, value);
  CATCH;
}
hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count,
                               hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                               size_t* failIdx, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpyBatchAsync_fn(dsts, srcs, sizes, count, attrs,
                                                            attrsIdxs, numAttrs, failIdx, stream);
  CATCH;
}
hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp* opList, size_t* failIdx,
                                 unsigned long long flags, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy3DBatchAsync_fn(numOps, opList, failIdx, flags,
                                                              stream);
  CATCH;
}
hipError_t hipMemcpy3DPeer(hipMemcpy3DPeerParms* p) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy3DPeer_fn(p);
  CATCH;
}
hipError_t hipMemcpy3DPeerAsync(hipMemcpy3DPeerParms* p, hipStream_t stream) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemcpy3DPeerAsync_fn(p, stream);
  CATCH;
}
hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements,
                                               const hipChannelFormatDesc* fmtDesc, int device) {
  TRY;
  return hip::GetHipDispatchTable()->hipDeviceGetTexture1DLinearMaxWidth_fn(maxWidthInElements,
                                                                            fmtDesc, device);
  CATCH;
}
hipError_t hipGraphAddExternalSemaphoresSignalNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddExternalSemaphoresSignalNode_fn(
      pGraphNode, graph, pDependencies, numDependencies, nodeParams);
  CATCH;
}
hipError_t hipGraphAddExternalSemaphoresWaitNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphAddExternalSemaphoresWaitNode_fn(
      pGraphNode, graph, pDependencies, numDependencies, nodeParams);
  CATCH;
}
hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExternalSemaphoresSignalNodeSetParams_fn(hNode,
                                                                                      nodeParams);
  CATCH;
}
hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExternalSemaphoresSignalNodeGetParams_fn(hNode,
                                                                                      params_out);
  CATCH;
}
hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExternalSemaphoresWaitNodeGetParams_fn(hNode,
                                                                                    params_out);
  CATCH;
}
hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExternalSemaphoresWaitNodeSetParams_fn(hNode,
                                                                                    nodeParams);
  CATCH;
}
hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecExternalSemaphoresSignalNodeSetParams_fn(
      hGraphExec, hNode, nodeParams);
  CATCH;
}
hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  TRY;
  return hip::GetHipDispatchTable()->hipGraphExecExternalSemaphoresWaitNodeSetParams_fn(
      hGraphExec, hNode, nodeParams);
  CATCH;
}
hipError_t hipLibraryLoadData(hipLibrary_t* library, const void* code, hipJitOption* jitOptions,
                              void** jitOptionsValues, unsigned int numJitOptions,
                              hipLibraryOption* libraryOptions, void** libraryOptionValues,
                              unsigned int numLibraryOptions) {
  TRY;
  return hip::GetHipDispatchTable()->hipLibraryLoadData_fn(
      library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
      libraryOptionValues, numLibraryOptions);
  CATCH;
}
hipError_t hipLibraryLoadFromFile(hipLibrary_t* library, const char* fileName,
                                  hipJitOption* jitOptions, void** jitOptionsValues,
                                  unsigned int numJitOptions, hipLibraryOption* libraryOptions,
                                  void** libraryOptionValues, unsigned int numLibraryOptions) {
  TRY;
  return hip::GetHipDispatchTable()->hipLibraryLoadFromFile_fn(
      library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
      libraryOptionValues, numLibraryOptions);
  CATCH;
}
hipError_t hipLibraryUnload(hipLibrary_t library) {
  TRY;
  return hip::GetHipDispatchTable()->hipLibraryUnload_fn(library);
  CATCH;
}
hipError_t hipLibraryGetKernel(hipKernel_t* pKernel, hipLibrary_t library, const char* name)  {
  TRY;
  return hip::GetHipDispatchTable()->hipLibraryGetKernel_fn(pKernel, library,
                                                            name);
  CATCH;
}
hipError_t hipLibraryGetKernelCount(unsigned int *count, hipLibrary_t library) {
  TRY;
  return hip::GetHipDispatchTable()->hipLibraryGetKernelCount_fn(count,
                                                                 library);
  CATCH;
}
hipError_t hipLibraryEnumerateKernels(hipKernel_t* kernels, unsigned int numKernels,
                                      hipLibrary_t library) {
  TRY;
  return hip::GetHipDispatchTable()->hipLibraryEnumerateKernels_fn(kernels, numKernels, library);
  CATCH;
}
hipError_t hipKernelGetLibrary(hipLibrary_t* library, hipKernel_t kernel) {
  TRY;
  return hip::GetHipDispatchTable()->hipKernelGetLibrary_fn(library, kernel);
  CATCH;
}
hipError_t hipKernelGetName(const char** name, hipKernel_t kernel) {
  TRY;
  return hip::GetHipDispatchTable()->hipKernelGetName_fn(name, kernel);
  CATCH;
}
hipError_t hipKernelGetParamInfo(hipKernel_t kernel, size_t paramIndex, size_t* paramOffset,
                                 size_t* paramSize) {
  TRY;
  return hip::GetHipDispatchTable()->hipKernelGetParamInfo_fn(kernel, paramIndex, paramOffset,
                                                              paramSize);
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
  TRY;
  return hip::GetHipDispatchTable()->hipMemSetMemPool_fn(location, type, pool);
  CATCH;
}
hipError_t hipMemGetMemPool(hipMemPool_t* pool, hipMemLocation* location,
                            hipMemAllocationType type) {
  TRY;
  return hip::GetHipDispatchTable()->hipMemGetMemPool_fn(pool, location, type);
  CATCH;
}
hipError_t hipMipmappedArrayGetMemoryRequirements(hipArrayMemoryRequirements* memoryRequirements,
                                                  hipMipmappedArray_t mipmap, hipDevice_t device) {
  TRY;
  return hip::GetHipDispatchTable()->hipMipmappedArrayGetMemoryRequirements_fn(memoryRequirements,
                                                                               mipmap, device);
  CATCH;
}