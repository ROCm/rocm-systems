/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * Dispatch table wrappers for the HIP CLR built-in profiling layer.
 * Pattern mirrors the reference hip_tracer.cpp:
 *   auto* record = HipGetActiveRecordExt(api_id);  // allocs slot, sets correlation_id TLS
 *   [call real function]
 *   if (record) record->end_ns = NowNs();
 */

#include "hip/amd_detail/hip_api_trace.hpp"
#include "hip_clr_profiler.hpp"
#include "rocclr/os/os.hpp"
#include "../hip_global.hpp"
#include "../hip_graph_internal.hpp"

#include <atomic>
#include <vector>

static inline uint64_t NowNs() { return amd::Os::timeNanos(); }

// Saved original dispatch table (the "next layer").
static HipDispatchTable g_next{};

// Pre-built wrapper table — populated once by HipProfilerBuildWrapperTableExt.
static HipDispatchTable g_wrapper_tbl{};

// Idempotency guard — true while wrappers are installed.
static std::atomic<bool> g_wrapped{false};

// Thread-local dims forwarded from __hipPushCallConfiguration → hipLaunchByPtr.
// hipLaunchKernelGGL via <<<>>> compiles to __hipPushCallConfiguration + hipLaunchByPtr;
// we capture grid/block from push so hipLaunchByPtr can stamp them on its record.
static thread_local dim3 g_pushed_grid{};
static thread_local dim3 g_pushed_block{};

// Parse the `extra` array from hipModuleLaunchKernel / hipExtModuleLaunchKernel.
// Format: { HIP_LAUNCH_PARAM_BUFFER_POINTER, ptr, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, END }
// Returns true and sets out_ptr/out_size if the buffer is valid; false otherwise.
static bool ParseKernelExtra(void** extra, const void*& out_ptr, size_t& out_size) {
  if (!extra) return false;
  if (extra[0] != HIP_LAUNCH_PARAM_BUFFER_POINTER) return false;
  if (extra[2] != HIP_LAUNCH_PARAM_BUFFER_SIZE) return false;
  if (extra[4] != HIP_LAUNCH_PARAM_END) return false;
  out_ptr = extra[1];
  out_size = *reinterpret_cast<size_t*>(extra[3]);
  return out_ptr != nullptr && out_size > 0;
}


// ── Compiler dispatch table hooks ────────────────────────────────────────────
// __hipPushCallConfiguration / __hipPopCallConfiguration live in HipCompilerDispatchTable,
// not HipDispatchTable.  We keep a saved copy and a wrapper copy for this table too.
static HipCompilerDispatchTable g_compiler_next{};
static HipCompilerDispatchTable g_compiler_wrapper_tbl{};
// Idempotency guard — true while compiler wrappers are installed.
static std::atomic<bool> g_compiler_wrapped{false};

// __hipPushCallConfiguration — called by the compiler for every <<<>>> launch.
// Saves grid/block in TLS so hipLaunchByPtrLayer can stamp them on its record.
static hipError_t __hipPushCallConfigurationLayer(dim3 gridDim, dim3 blockDim,
                                                   size_t sharedMem, hipStream_t stream) {
  // kHipApiNamesCountExt is out-of-range → api_name gets "unknown"; override below.
  auto* _rec = HipGetActiveRecordExt(kHipApiNamesCountExt);
  _rec->api_name = "__hipPushCallConfiguration";
  g_pushed_grid  = gridDim;
  g_pushed_block = blockDim;
  auto _r = g_compiler_next.__hipPushCallConfiguration_fn(gridDim, blockDim, sharedMem, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// __hipPopCallConfiguration — completes the <<<>>> sequence; no dims needed here.
static hipError_t __hipPopCallConfigurationLayer(dim3* gridDim, dim3* blockDim,
                                                  size_t* sharedMem, hipStream_t* stream) {
  auto* _rec = HipGetActiveRecordExt(kHipApiNamesCountExt);
  _rec->api_name = "__hipPopCallConfiguration";
  auto _r = g_compiler_next.__hipPopCallConfiguration_fn(gridDim, blockDim, sharedMem, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 0
static const char* hipApiNameLayer(uint32_t id) {
  auto* _rec = HipGetActiveRecordExt(0u);
  auto _r = g_next.hipApiName_fn(id);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 1
static hipError_t hipArray3DCreateLayer(hipArray_t* array,
                                         const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  auto* _rec = HipGetActiveRecordExt(1u);
  auto _r = g_next.hipArray3DCreate_fn(array, pAllocateArray);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 2
static hipError_t hipArray3DGetDescriptorLayer(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor,
                                                hipArray_t array) {
  auto* _rec = HipGetActiveRecordExt(2u);
  auto _r = g_next.hipArray3DGetDescriptor_fn(pArrayDescriptor, array);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 3
static hipError_t hipArrayCreateLayer(hipArray_t* pHandle,
                                       const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  auto* _rec = HipGetActiveRecordExt(3u);
  auto _r = g_next.hipArrayCreate_fn(pHandle, pAllocateArray);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 4
static hipError_t hipArrayDestroyLayer(hipArray_t array) {
  auto* _rec = HipGetActiveRecordExt(4u);
  auto _r = g_next.hipArrayDestroy_fn(array);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 5
static hipError_t hipArrayGetDescriptorLayer(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                              hipArray_t array) {
  auto* _rec = HipGetActiveRecordExt(5u);
  auto _r = g_next.hipArrayGetDescriptor_fn(pArrayDescriptor, array);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 6
static hipError_t hipArrayGetInfoLayer(hipChannelFormatDesc* desc, hipExtent* extent,
                                        unsigned int* flags, hipArray_t array) {
  auto* _rec = HipGetActiveRecordExt(6u);
  auto _r = g_next.hipArrayGetInfo_fn(desc, extent, flags, array);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 7
static hipError_t hipBindTextureLayer(size_t* offset, const textureReference* tex,
                                       const void* devPtr, const hipChannelFormatDesc* desc,
                                       size_t size) {
  auto* _rec = HipGetActiveRecordExt(7u);
  auto _r = g_next.hipBindTexture_fn(offset, tex, devPtr, desc, size);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 8
static hipError_t hipBindTexture2DLayer(size_t* offset, const textureReference* tex,
                                         const void* devPtr, const hipChannelFormatDesc* desc,
                                         size_t width, size_t height, size_t pitch) {
  auto* _rec = HipGetActiveRecordExt(8u);
  auto _r = g_next.hipBindTexture2D_fn(offset, tex, devPtr, desc, width, height, pitch);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 9
static hipError_t hipBindTextureToArrayLayer(const textureReference* tex, hipArray_const_t array,
                                              const hipChannelFormatDesc* desc) {
  auto* _rec = HipGetActiveRecordExt(9u);
  auto _r = g_next.hipBindTextureToArray_fn(tex, array, desc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 10
static hipError_t hipBindTextureToMipmappedArrayLayer(const textureReference* tex,
                                                       hipMipmappedArray_const_t mipmappedArray,
                                                       const hipChannelFormatDesc* desc) {
  auto* _rec = HipGetActiveRecordExt(10u);
  auto _r = g_next.hipBindTextureToMipmappedArray_fn(tex, mipmappedArray, desc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 11
static hipError_t hipChooseDeviceLayer(int* device, const hipDeviceProp_t* prop) {
  auto* _rec = HipGetActiveRecordExt(11u);
  auto _r = g_next.hipChooseDevice_fn(device, prop);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 12
static hipError_t hipChooseDeviceR0000Layer(int* device, const hipDeviceProp_tR0000* properties) {
  auto* _rec = HipGetActiveRecordExt(12u);
  auto _r = g_next.hipChooseDeviceR0000_fn(device, properties);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 13
static hipError_t hipConfigureCallLayer(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                         hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(13u);
  _rec->stream = stream;
  auto _r = g_next.hipConfigureCall_fn(gridDim, blockDim, sharedMem, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 14
static hipError_t hipCreateSurfaceObjectLayer(hipSurfaceObject_t* pSurfObject,
                                               const hipResourceDesc* pResDesc) {
  auto* _rec = HipGetActiveRecordExt(14u);
  auto _r = g_next.hipCreateSurfaceObject_fn(pSurfObject, pResDesc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 15
static hipError_t hipCreateTextureObjectLayer(hipTextureObject_t* pTexObject,
                                               const hipResourceDesc* pResDesc,
                                               const hipTextureDesc* pTexDesc,
                                               const struct hipResourceViewDesc* pResViewDesc) {
  auto* _rec = HipGetActiveRecordExt(15u);
  auto _r = g_next.hipCreateTextureObject_fn(pTexObject, pResDesc, pTexDesc, pResViewDesc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 16
static hipError_t hipCtxCreateLayer(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
  auto* _rec = HipGetActiveRecordExt(16u);
  auto _r = g_next.hipCtxCreate_fn(ctx, flags, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 17
static hipError_t hipCtxDestroyLayer(hipCtx_t ctx) {
  auto* _rec = HipGetActiveRecordExt(17u);
  auto _r = g_next.hipCtxDestroy_fn(ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 18
static hipError_t hipCtxDisablePeerAccessLayer(hipCtx_t peerCtx) {
  auto* _rec = HipGetActiveRecordExt(18u);
  auto _r = g_next.hipCtxDisablePeerAccess_fn(peerCtx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 19
static hipError_t hipCtxEnablePeerAccessLayer(hipCtx_t peerCtx, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(19u);
  auto _r = g_next.hipCtxEnablePeerAccess_fn(peerCtx, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 20
static hipError_t hipCtxGetApiVersionLayer(hipCtx_t ctx, unsigned int* apiVersion) {
  auto* _rec = HipGetActiveRecordExt(20u);
  auto _r = g_next.hipCtxGetApiVersion_fn(ctx, apiVersion);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 21
static hipError_t hipCtxGetCacheConfigLayer(hipFuncCache_t* cacheConfig) {
  auto* _rec = HipGetActiveRecordExt(21u);
  auto _r = g_next.hipCtxGetCacheConfig_fn(cacheConfig);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 22
static hipError_t hipCtxGetCurrentLayer(hipCtx_t* ctx) {
  auto* _rec = HipGetActiveRecordExt(22u);
  auto _r = g_next.hipCtxGetCurrent_fn(ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 23
static hipError_t hipCtxGetDeviceLayer(hipDevice_t* device) {
  auto* _rec = HipGetActiveRecordExt(23u);
  auto _r = g_next.hipCtxGetDevice_fn(device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 24
static hipError_t hipCtxGetFlagsLayer(unsigned int* flags) {
  auto* _rec = HipGetActiveRecordExt(24u);
  auto _r = g_next.hipCtxGetFlags_fn(flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 25
static hipError_t hipCtxGetSharedMemConfigLayer(hipSharedMemConfig* pConfig) {
  auto* _rec = HipGetActiveRecordExt(25u);
  auto _r = g_next.hipCtxGetSharedMemConfig_fn(pConfig);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 26
static hipError_t hipCtxPopCurrentLayer(hipCtx_t* ctx) {
  auto* _rec = HipGetActiveRecordExt(26u);
  auto _r = g_next.hipCtxPopCurrent_fn(ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 27
static hipError_t hipCtxPushCurrentLayer(hipCtx_t ctx) {
  auto* _rec = HipGetActiveRecordExt(27u);
  auto _r = g_next.hipCtxPushCurrent_fn(ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 28
static hipError_t hipCtxSetCacheConfigLayer(hipFuncCache_t cacheConfig) {
  auto* _rec = HipGetActiveRecordExt(28u);
  auto _r = g_next.hipCtxSetCacheConfig_fn(cacheConfig);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 29
static hipError_t hipCtxSetCurrentLayer(hipCtx_t ctx) {
  auto* _rec = HipGetActiveRecordExt(29u);
  auto _r = g_next.hipCtxSetCurrent_fn(ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 30
static hipError_t hipCtxSetSharedMemConfigLayer(hipSharedMemConfig config) {
  auto* _rec = HipGetActiveRecordExt(30u);
  auto _r = g_next.hipCtxSetSharedMemConfig_fn(config);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 31
static hipError_t hipCtxSynchronizeLayer(void) {
  auto* _rec = HipGetActiveRecordExt(31u);
  auto _r = g_next.hipCtxSynchronize_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 32
static hipError_t hipDestroyExternalMemoryLayer(hipExternalMemory_t extMem) {
  auto* _rec = HipGetActiveRecordExt(32u);
  auto _r = g_next.hipDestroyExternalMemory_fn(extMem);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 33
static hipError_t hipDestroyExternalSemaphoreLayer(hipExternalSemaphore_t extSem) {
  auto* _rec = HipGetActiveRecordExt(33u);
  auto _r = g_next.hipDestroyExternalSemaphore_fn(extSem);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 34
static hipError_t hipDestroySurfaceObjectLayer(hipSurfaceObject_t surfaceObject) {
  auto* _rec = HipGetActiveRecordExt(34u);
  auto _r = g_next.hipDestroySurfaceObject_fn(surfaceObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 35
static hipError_t hipDestroyTextureObjectLayer(hipTextureObject_t textureObject) {
  auto* _rec = HipGetActiveRecordExt(35u);
  auto _r = g_next.hipDestroyTextureObject_fn(textureObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 36
static hipError_t hipDeviceCanAccessPeerLayer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  auto* _rec = HipGetActiveRecordExt(36u);
  auto _r = g_next.hipDeviceCanAccessPeer_fn(canAccessPeer, deviceId, peerDeviceId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 37
static hipError_t hipDeviceComputeCapabilityLayer(int* major, int* minor, hipDevice_t device) {
  auto* _rec = HipGetActiveRecordExt(37u);
  auto _r = g_next.hipDeviceComputeCapability_fn(major, minor, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 38
static hipError_t hipDeviceDisablePeerAccessLayer(int peerDeviceId) {
  auto* _rec = HipGetActiveRecordExt(38u);
  auto _r = g_next.hipDeviceDisablePeerAccess_fn(peerDeviceId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 39
static hipError_t hipDeviceEnablePeerAccessLayer(int peerDeviceId, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(39u);
  auto _r = g_next.hipDeviceEnablePeerAccess_fn(peerDeviceId, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 40
static hipError_t hipDeviceGetLayer(hipDevice_t* device, int ordinal) {
  auto* _rec = HipGetActiveRecordExt(40u);
  auto _r = g_next.hipDeviceGet_fn(device, ordinal);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 41
static hipError_t hipDeviceGetAttributeLayer(int* pi, hipDeviceAttribute_t attr, int deviceId) {
  auto* _rec = HipGetActiveRecordExt(41u);
  auto _r = g_next.hipDeviceGetAttribute_fn(pi, attr, deviceId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 42
static hipError_t hipDeviceGetByPCIBusIdLayer(int* device, const char* pciBusId) {
  auto* _rec = HipGetActiveRecordExt(42u);
  auto _r = g_next.hipDeviceGetByPCIBusId_fn(device, pciBusId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 43
static hipError_t hipDeviceGetCacheConfigLayer(hipFuncCache_t* cacheConfig) {
  auto* _rec = HipGetActiveRecordExt(43u);
  auto _r = g_next.hipDeviceGetCacheConfig_fn(cacheConfig);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 44
static hipError_t hipDeviceGetDefaultMemPoolLayer(hipMemPool_t* mem_pool, int device) {
  auto* _rec = HipGetActiveRecordExt(44u);
  auto _r = g_next.hipDeviceGetDefaultMemPool_fn(mem_pool, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 45
static hipError_t hipDeviceGetGraphMemAttributeLayer(int device, hipGraphMemAttributeType attr,
                                                      void* value) {
  auto* _rec = HipGetActiveRecordExt(45u);
  auto _r = g_next.hipDeviceGetGraphMemAttribute_fn(device, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 46
static hipError_t hipDeviceGetLimitLayer(size_t* pValue, enum hipLimit_t limit) {
  auto* _rec = HipGetActiveRecordExt(46u);
  auto _r = g_next.hipDeviceGetLimit_fn(pValue, limit);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 47
static hipError_t hipDeviceGetMemPoolLayer(hipMemPool_t* mem_pool, int device) {
  auto* _rec = HipGetActiveRecordExt(47u);
  auto _r = g_next.hipDeviceGetMemPool_fn(mem_pool, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 48
static hipError_t hipDeviceGetNameLayer(char* name, int len, hipDevice_t device) {
  auto* _rec = HipGetActiveRecordExt(48u);
  auto _r = g_next.hipDeviceGetName_fn(name, len, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 49
static hipError_t hipDeviceGetP2PAttributeLayer(int* value, hipDeviceP2PAttr attr, int srcDevice,
                                                 int dstDevice) {
  auto* _rec = HipGetActiveRecordExt(49u);
  auto _r = g_next.hipDeviceGetP2PAttribute_fn(value, attr, srcDevice, dstDevice);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 50
static hipError_t hipDeviceGetPCIBusIdLayer(char* pciBusId, int len, int device) {
  auto* _rec = HipGetActiveRecordExt(50u);
  auto _r = g_next.hipDeviceGetPCIBusId_fn(pciBusId, len, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 51
static hipError_t hipDeviceGetSharedMemConfigLayer(hipSharedMemConfig* pConfig) {
  auto* _rec = HipGetActiveRecordExt(51u);
  auto _r = g_next.hipDeviceGetSharedMemConfig_fn(pConfig);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 52
static hipError_t hipDeviceGetStreamPriorityRangeLayer(int* leastPriority, int* greatestPriority) {
  auto* _rec = HipGetActiveRecordExt(52u);
  auto _r = g_next.hipDeviceGetStreamPriorityRange_fn(leastPriority, greatestPriority);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 53
static hipError_t hipDeviceGetTexture1DLinearMaxWidthLayer(size_t* maxWidthInElements,
                                                            const hipChannelFormatDesc* fmtDesc,
                                                            int device) {
  auto* _rec = HipGetActiveRecordExt(53u);
  auto _r = g_next.hipDeviceGetTexture1DLinearMaxWidth_fn(maxWidthInElements, fmtDesc, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 54
static hipError_t hipDeviceGetUuidLayer(hipUUID* uuid, hipDevice_t device) {
  auto* _rec = HipGetActiveRecordExt(54u);
  auto _r = g_next.hipDeviceGetUuid_fn(uuid, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 55
static hipError_t hipDeviceGraphMemTrimLayer(int device) {
  auto* _rec = HipGetActiveRecordExt(55u);
  auto _r = g_next.hipDeviceGraphMemTrim_fn(device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 56
static hipError_t hipDevicePrimaryCtxGetStateLayer(hipDevice_t dev, unsigned int* flags,
                                                    int* active) {
  auto* _rec = HipGetActiveRecordExt(56u);
  auto _r = g_next.hipDevicePrimaryCtxGetState_fn(dev, flags, active);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 57
static hipError_t hipDevicePrimaryCtxReleaseLayer(hipDevice_t dev) {
  auto* _rec = HipGetActiveRecordExt(57u);
  auto _r = g_next.hipDevicePrimaryCtxRelease_fn(dev);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 58
static hipError_t hipDevicePrimaryCtxResetLayer(hipDevice_t dev) {
  auto* _rec = HipGetActiveRecordExt(58u);
  auto _r = g_next.hipDevicePrimaryCtxReset_fn(dev);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 59
static hipError_t hipDevicePrimaryCtxRetainLayer(hipCtx_t* pctx, hipDevice_t dev) {
  auto* _rec = HipGetActiveRecordExt(59u);
  auto _r = g_next.hipDevicePrimaryCtxRetain_fn(pctx, dev);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 60
static hipError_t hipDevicePrimaryCtxSetFlagsLayer(hipDevice_t dev, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(60u);
  auto _r = g_next.hipDevicePrimaryCtxSetFlags_fn(dev, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 61
static hipError_t hipDeviceResetLayer(void) {
  auto* _rec = HipGetActiveRecordExt(61u);
  auto _r = g_next.hipDeviceReset_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 62
static hipError_t hipDeviceSetCacheConfigLayer(hipFuncCache_t cacheConfig) {
  auto* _rec = HipGetActiveRecordExt(62u);
  auto _r = g_next.hipDeviceSetCacheConfig_fn(cacheConfig);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 63
static hipError_t hipDeviceSetGraphMemAttributeLayer(int device, hipGraphMemAttributeType attr,
                                                      void* value) {
  auto* _rec = HipGetActiveRecordExt(63u);
  auto _r = g_next.hipDeviceSetGraphMemAttribute_fn(device, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 64
static hipError_t hipDeviceSetLimitLayer(enum hipLimit_t limit, size_t value) {
  auto* _rec = HipGetActiveRecordExt(64u);
  auto _r = g_next.hipDeviceSetLimit_fn(limit, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 65
static hipError_t hipDeviceSetMemPoolLayer(int device, hipMemPool_t mem_pool) {
  auto* _rec = HipGetActiveRecordExt(65u);
  auto _r = g_next.hipDeviceSetMemPool_fn(device, mem_pool);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 66
static hipError_t hipDeviceSetSharedMemConfigLayer(hipSharedMemConfig config) {
  auto* _rec = HipGetActiveRecordExt(66u);
  auto _r = g_next.hipDeviceSetSharedMemConfig_fn(config);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 67
static hipError_t hipDeviceSynchronizeLayer(void) {
  auto* _rec = HipGetActiveRecordExt(67u);
  auto _r = g_next.hipDeviceSynchronize_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 68
static hipError_t hipDeviceTotalMemLayer(size_t* bytes, hipDevice_t device) {
  auto* _rec = HipGetActiveRecordExt(68u);
  auto _r = g_next.hipDeviceTotalMem_fn(bytes, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 69
static hipError_t hipDriverGetVersionLayer(int* driverVersion) {
  auto* _rec = HipGetActiveRecordExt(69u);
  auto _r = g_next.hipDriverGetVersion_fn(driverVersion);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 70
static hipError_t hipDrvGetErrorNameLayer(hipError_t hipError, const char** errorString) {
  auto* _rec = HipGetActiveRecordExt(70u);
  auto _r = g_next.hipDrvGetErrorName_fn(hipError, errorString);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 71
static hipError_t hipDrvGetErrorStringLayer(hipError_t hipError, const char** errorString) {
  auto* _rec = HipGetActiveRecordExt(71u);
  auto _r = g_next.hipDrvGetErrorString_fn(hipError, errorString);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 72
static hipError_t hipDrvGraphAddMemcpyNodeLayer(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                                 const hipGraphNode_t* dependencies,
                                                 size_t numDependencies,
                                                 const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  auto* _rec = HipGetActiveRecordExt(72u);
  auto _r = g_next.hipDrvGraphAddMemcpyNode_fn(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 73
static hipError_t hipDrvMemcpy2DUnalignedLayer(const hip_Memcpy2D* pCopy) {
  auto* _rec = HipGetActiveRecordExt(73u);
  auto _r = g_next.hipDrvMemcpy2DUnaligned_fn(pCopy);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 74
static hipError_t hipDrvMemcpy3DLayer(const HIP_MEMCPY3D* pCopy) {
  auto* _rec = HipGetActiveRecordExt(74u);
  auto _r = g_next.hipDrvMemcpy3D_fn(pCopy);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 75
static hipError_t hipDrvMemcpy3DAsyncLayer(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(75u);
  _rec->stream = stream;
  auto _r = g_next.hipDrvMemcpy3DAsync_fn(pCopy, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 76
static hipError_t hipDrvPointerGetAttributesLayer(unsigned int numAttributes,
                                                   hipPointer_attribute* attributes, void** data,
                                                   hipDeviceptr_t ptr) {
  auto* _rec = HipGetActiveRecordExt(76u);
  auto _r = g_next.hipDrvPointerGetAttributes_fn(numAttributes, attributes, data, ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 77
static hipError_t hipEventCreateLayer(hipEvent_t* event) {
  auto* _rec = HipGetActiveRecordExt(77u);
  auto _r = g_next.hipEventCreate_fn(event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 78
static hipError_t hipEventCreateWithFlagsLayer(hipEvent_t* event, unsigned flags) {
  auto* _rec = HipGetActiveRecordExt(78u);
  auto _r = g_next.hipEventCreateWithFlags_fn(event, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 79
static hipError_t hipEventDestroyLayer(hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(79u);
  auto _r = g_next.hipEventDestroy_fn(event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 80
static hipError_t hipEventElapsedTimeLayer(float* ms, hipEvent_t start, hipEvent_t stop) {
  auto* _rec = HipGetActiveRecordExt(80u);
  auto _r = g_next.hipEventElapsedTime_fn(ms, start, stop);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 81
static hipError_t hipEventQueryLayer(hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(81u);
  auto _r = g_next.hipEventQuery_fn(event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 82
static hipError_t hipEventRecordLayer(hipEvent_t event, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(82u);
  _rec->stream = stream;
  auto _r = g_next.hipEventRecord_fn(event, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 83
static hipError_t hipEventSynchronizeLayer(hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(83u);
  auto _r = g_next.hipEventSynchronize_fn(event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 84
static hipError_t hipExtGetLinkTypeAndHopCountLayer(int device1, int device2, uint32_t* linktype,
                                                     uint32_t* hopcount) {
  auto* _rec = HipGetActiveRecordExt(84u);
  auto _r = g_next.hipExtGetLinkTypeAndHopCount_fn(device1, device2, linktype, hopcount);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 85
static hipError_t hipExtLaunchKernelLayer(const void* function_address, dim3 numBlocks,
                                           dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                           hipStream_t stream, hipEvent_t startEvent,
                                           hipEvent_t stopEvent, int flags) {
  auto* _rec = HipGetActiveRecordExt(85u);
  _rec->stream = stream;
  _rec->gpu.grid_x = numBlocks.x; _rec->gpu.grid_y = numBlocks.y; _rec->gpu.grid_z = numBlocks.z;
  _rec->gpu.block_x = dimBlocks.x; _rec->gpu.block_y = dimBlocks.y; _rec->gpu.block_z = dimBlocks.z;
  if (args) {
    hipFunction_t hfunc = nullptr;
    if (hip::ihipGetFuncBySymbol(&hfunc, function_address) == hipSuccess)
      HipCaptureKernelArgsExt(&_rec->gpu, hfunc, args);
  }
  auto _r = g_next.hipExtLaunchKernel_fn(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 86
static hipError_t hipExtLaunchMultiKernelMultiDeviceLayer(hipLaunchParams* launchParamsList,
                                                           int numDevices, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(86u);
  auto _r = g_next.hipExtLaunchMultiKernelMultiDevice_fn(launchParamsList, numDevices, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 87
static hipError_t hipExtMallocWithFlagsLayer(void** ptr, size_t sizeBytes, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(87u);
  auto _r = g_next.hipExtMallocWithFlags_fn(ptr, sizeBytes, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 88
static hipError_t hipExtStreamCreateWithCUMaskLayer(hipStream_t* stream, uint32_t cuMaskSize,
                                                     const uint32_t* cuMask) {
  auto* _rec = HipGetActiveRecordExt(88u);
  auto _r = g_next.hipExtStreamCreateWithCUMask_fn(stream, cuMaskSize, cuMask);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 89
static hipError_t hipExtStreamGetCUMaskLayer(hipStream_t stream, uint32_t cuMaskSize,
                                              uint32_t* cuMask) {
  auto* _rec = HipGetActiveRecordExt(89u);
  _rec->stream = stream;
  auto _r = g_next.hipExtStreamGetCUMask_fn(stream, cuMaskSize, cuMask);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 90
static hipError_t hipExternalMemoryGetMappedBufferLayer(void** devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc* bufferDesc) {
  auto* _rec = HipGetActiveRecordExt(90u);
  auto _r = g_next.hipExternalMemoryGetMappedBuffer_fn(devPtr, extMem, bufferDesc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 91
static hipError_t hipFreeLayer(void* ptr) {
  auto* _rec = HipGetActiveRecordExt(91u);
  _rec->memory1 = ptr;
  auto _r = g_next.hipFree_fn(ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 92
static hipError_t hipFreeArrayLayer(hipArray_t array) {
  auto* _rec = HipGetActiveRecordExt(92u);
  auto _r = g_next.hipFreeArray_fn(array);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 93
static hipError_t hipFreeAsyncLayer(void* dev_ptr, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(93u);
  _rec->stream = stream;
  _rec->memory1 = dev_ptr;
  auto _r = g_next.hipFreeAsync_fn(dev_ptr, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 94
static hipError_t hipFreeHostLayer(void* ptr) {
  auto* _rec = HipGetActiveRecordExt(94u);
  _rec->memory1 = ptr;
  auto _r = g_next.hipFreeHost_fn(ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 95
static hipError_t hipFreeMipmappedArrayLayer(hipMipmappedArray_t mipmappedArray) {
  auto* _rec = HipGetActiveRecordExt(95u);
  auto _r = g_next.hipFreeMipmappedArray_fn(mipmappedArray);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 96
static hipError_t hipFuncGetAttributeLayer(int* value, hipFunction_attribute attrib,
                                            hipFunction_t hfunc) {
  auto* _rec = HipGetActiveRecordExt(96u);
  auto _r = g_next.hipFuncGetAttribute_fn(value, attrib, hfunc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 97
static hipError_t hipFuncGetAttributesLayer(struct hipFuncAttributes* attr, const void* func) {
  auto* _rec = HipGetActiveRecordExt(97u);
  auto _r = g_next.hipFuncGetAttributes_fn(attr, func);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 98
static hipError_t hipFuncSetAttributeLayer(const void* func, hipFuncAttribute attr, int value) {
  auto* _rec = HipGetActiveRecordExt(98u);
  auto _r = g_next.hipFuncSetAttribute_fn(func, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 99
static hipError_t hipFuncSetCacheConfigLayer(const void* func, hipFuncCache_t config) {
  auto* _rec = HipGetActiveRecordExt(99u);
  auto _r = g_next.hipFuncSetCacheConfig_fn(func, config);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 100
static hipError_t hipFuncSetSharedMemConfigLayer(const void* func, hipSharedMemConfig config) {
  auto* _rec = HipGetActiveRecordExt(100u);
  auto _r = g_next.hipFuncSetSharedMemConfig_fn(func, config);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 101
static hipError_t hipGLGetDevicesLayer(unsigned int* pHipDeviceCount, int* pHipDevices,
                                        unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
  auto* _rec = HipGetActiveRecordExt(101u);
  auto _r = g_next.hipGLGetDevices_fn(pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 102
static hipError_t hipGetChannelDescLayer(hipChannelFormatDesc* desc, hipArray_const_t array) {
  auto* _rec = HipGetActiveRecordExt(102u);
  auto _r = g_next.hipGetChannelDesc_fn(desc, array);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 103
static hipError_t hipGetDeviceLayer(int* deviceId) {
  auto* _rec = HipGetActiveRecordExt(103u);
  auto _r = g_next.hipGetDevice_fn(deviceId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 104
static hipError_t hipGetDeviceCountLayer(int* count) {
  auto* _rec = HipGetActiveRecordExt(104u);
  auto _r = g_next.hipGetDeviceCount_fn(count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 105
static hipError_t hipGetDeviceFlagsLayer(unsigned int* flags) {
  auto* _rec = HipGetActiveRecordExt(105u);
  auto _r = g_next.hipGetDeviceFlags_fn(flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 106
static hipError_t hipGetDevicePropertiesR0600Layer(hipDeviceProp_tR0600* prop, int device) {
  auto* _rec = HipGetActiveRecordExt(106u);
  auto _r = g_next.hipGetDevicePropertiesR0600_fn(prop, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 107
static const char* hipGetErrorNameLayer(hipError_t hip_error) {
  auto* _rec = HipGetActiveRecordExt(107u);
  auto _r = g_next.hipGetErrorName_fn(hip_error);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 108
static const char* hipGetErrorStringLayer(hipError_t hipError) {
  auto* _rec = HipGetActiveRecordExt(108u);
  auto _r = g_next.hipGetErrorString_fn(hipError);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 109
static hipError_t hipGetLastErrorLayer(void) {
  auto* _rec = HipGetActiveRecordExt(109u);
  auto _r = g_next.hipGetLastError_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 110
static hipError_t hipGetMipmappedArrayLevelLayer(hipArray_t* levelArray,
                                                  hipMipmappedArray_const_t mipmappedArray,
                                                  unsigned int level) {
  auto* _rec = HipGetActiveRecordExt(110u);
  auto _r = g_next.hipGetMipmappedArrayLevel_fn(levelArray, mipmappedArray, level);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 111
static hipError_t hipGetSymbolAddressLayer(void** devPtr, const void* symbol) {
  auto* _rec = HipGetActiveRecordExt(111u);
  auto _r = g_next.hipGetSymbolAddress_fn(devPtr, symbol);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 112
static hipError_t hipGetSymbolSizeLayer(size_t* size, const void* symbol) {
  auto* _rec = HipGetActiveRecordExt(112u);
  auto _r = g_next.hipGetSymbolSize_fn(size, symbol);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 113
static hipError_t hipGetTextureAlignmentOffsetLayer(size_t* offset,
                                                     const textureReference* texref) {
  auto* _rec = HipGetActiveRecordExt(113u);
  auto _r = g_next.hipGetTextureAlignmentOffset_fn(offset, texref);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 114
static hipError_t hipGetTextureObjectResourceDescLayer(hipResourceDesc* pResDesc,
                                                        hipTextureObject_t textureObject) {
  auto* _rec = HipGetActiveRecordExt(114u);
  auto _r = g_next.hipGetTextureObjectResourceDesc_fn(pResDesc, textureObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 115
static hipError_t hipGetTextureObjectResourceViewDescLayer(struct hipResourceViewDesc* pResViewDesc, hipTextureObject_t textureObject) {
  auto* _rec = HipGetActiveRecordExt(115u);
  auto _r = g_next.hipGetTextureObjectResourceViewDesc_fn(pResViewDesc, textureObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 116
static hipError_t hipGetTextureObjectTextureDescLayer(hipTextureDesc* pTexDesc,
                                                       hipTextureObject_t textureObject) {
  auto* _rec = HipGetActiveRecordExt(116u);
  auto _r = g_next.hipGetTextureObjectTextureDesc_fn(pTexDesc, textureObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 117
static hipError_t hipGetTextureReferenceLayer(const textureReference** texref, const void* symbol) {
  auto* _rec = HipGetActiveRecordExt(117u);
  auto _r = g_next.hipGetTextureReference_fn(texref, symbol);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 118
static hipError_t hipGraphAddChildGraphNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                  const hipGraphNode_t* pDependencies,
                                                  size_t numDependencies, hipGraph_t childGraph) {
  auto* _rec = HipGetActiveRecordExt(118u);
  auto _r = g_next.hipGraphAddChildGraphNode_fn(pGraphNode, graph, pDependencies, numDependencies, childGraph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 119
static hipError_t hipGraphAddDependenciesLayer(hipGraph_t graph, const hipGraphNode_t* from,
                                                const hipGraphNode_t* to, size_t numDependencies) {
  auto* _rec = HipGetActiveRecordExt(119u);
  auto _r = g_next.hipGraphAddDependencies_fn(graph, from, to, numDependencies);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 120
static hipError_t hipGraphAddEmptyNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                             const hipGraphNode_t* pDependencies,
                                             size_t numDependencies) {
  auto* _rec = HipGetActiveRecordExt(120u);
  auto _r = g_next.hipGraphAddEmptyNode_fn(pGraphNode, graph, pDependencies, numDependencies);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 121
static hipError_t hipGraphAddEventRecordNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                   const hipGraphNode_t* pDependencies,
                                                   size_t numDependencies, hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(121u);
  auto _r = g_next.hipGraphAddEventRecordNode_fn(pGraphNode, graph, pDependencies, numDependencies, event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 122
static hipError_t hipGraphAddEventWaitNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                 const hipGraphNode_t* pDependencies,
                                                 size_t numDependencies, hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(122u);
  auto _r = g_next.hipGraphAddEventWaitNode_fn(pGraphNode, graph, pDependencies, numDependencies, event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 123
static hipError_t hipGraphAddHostNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                            const hipGraphNode_t* pDependencies,
                                            size_t numDependencies,
                                            const hipHostNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(123u);
  auto _r = g_next.hipGraphAddHostNode_fn(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 124
static hipError_t hipGraphAddKernelNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                              const hipGraphNode_t* pDependencies,
                                              size_t numDependencies,
                                              const hipKernelNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(124u);
  auto _r = g_next.hipGraphAddKernelNode_fn(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 125
static hipError_t hipGraphAddMemAllocNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                const hipGraphNode_t* pDependencies,
                                                size_t numDependencies,
                                                hipMemAllocNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(125u);
  auto _r = g_next.hipGraphAddMemAllocNode_fn(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 126
static hipError_t hipGraphAddMemFreeNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                               const hipGraphNode_t* pDependencies,
                                               size_t numDependencies, void* dev_ptr) {
  auto* _rec = HipGetActiveRecordExt(126u);
  auto _r = g_next.hipGraphAddMemFreeNode_fn(pGraphNode, graph, pDependencies, numDependencies, dev_ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 127
static hipError_t hipGraphAddMemcpyNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                              const hipGraphNode_t* pDependencies,
                                              size_t numDependencies,
                                              const hipMemcpy3DParms* pCopyParams) {
  auto* _rec = HipGetActiveRecordExt(127u);
  auto _r = g_next.hipGraphAddMemcpyNode_fn(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 128
static hipError_t hipGraphAddMemcpyNode1DLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                const hipGraphNode_t* pDependencies,
                                                size_t numDependencies, void* dst, const void* src,
                                                size_t count, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(128u);
  auto _r = g_next.hipGraphAddMemcpyNode1D_fn(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 129
static hipError_t hipGraphAddMemcpyNodeFromSymbolLayer(hipGraphNode_t* pGraphNode,
                                                        hipGraph_t graph,
                                                        const hipGraphNode_t* pDependencies,
                                                        size_t numDependencies, void* dst,
                                                        const void* symbol, size_t count,
                                                        size_t offset, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(129u);
  auto _r = g_next.hipGraphAddMemcpyNodeFromSymbol_fn(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 130
static hipError_t hipGraphAddMemcpyNodeToSymbolLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                      const hipGraphNode_t* pDependencies,
                                                      size_t numDependencies, const void* symbol,
                                                      const void* src, size_t count, size_t offset,
                                                      hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(130u);
  auto _r = g_next.hipGraphAddMemcpyNodeToSymbol_fn(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 131
static hipError_t hipGraphAddMemsetNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                              const hipGraphNode_t* pDependencies,
                                              size_t numDependencies,
                                              const hipMemsetParams* pMemsetParams) {
  auto* _rec = HipGetActiveRecordExt(131u);
  auto _r = g_next.hipGraphAddMemsetNode_fn(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 132
static hipError_t hipGraphAddNodeLayer(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies, size_t numDependencies,
                                        hipGraphNodeParams* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(132u);
  auto _r = g_next.hipGraphAddNode_fn(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 133
static hipError_t hipGraphChildGraphNodeGetGraphLayer(hipGraphNode_t node, hipGraph_t* pGraph) {
  auto* _rec = HipGetActiveRecordExt(133u);
  auto _r = g_next.hipGraphChildGraphNodeGetGraph_fn(node, pGraph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 134
static hipError_t hipGraphCloneLayer(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  auto* _rec = HipGetActiveRecordExt(134u);
  auto _r = g_next.hipGraphClone_fn(pGraphClone, originalGraph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 135
static hipError_t hipGraphCreateLayer(hipGraph_t* pGraph, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(135u);
  auto _r = g_next.hipGraphCreate_fn(pGraph, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 136
static hipError_t hipGraphDebugDotPrintLayer(hipGraph_t graph, const char* path,
                                              unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(136u);
  auto _r = g_next.hipGraphDebugDotPrint_fn(graph, path, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 137
static hipError_t hipGraphDestroyLayer(hipGraph_t graph) {
  auto* _rec = HipGetActiveRecordExt(137u);
  auto _r = g_next.hipGraphDestroy_fn(graph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 138
static hipError_t hipGraphDestroyNodeLayer(hipGraphNode_t node) {
  auto* _rec = HipGetActiveRecordExt(138u);
  auto _r = g_next.hipGraphDestroyNode_fn(node);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 139
static hipError_t hipGraphEventRecordNodeGetEventLayer(hipGraphNode_t node, hipEvent_t* event_out) {
  auto* _rec = HipGetActiveRecordExt(139u);
  auto _r = g_next.hipGraphEventRecordNodeGetEvent_fn(node, event_out);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 140
static hipError_t hipGraphEventRecordNodeSetEventLayer(hipGraphNode_t node, hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(140u);
  auto _r = g_next.hipGraphEventRecordNodeSetEvent_fn(node, event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 141
static hipError_t hipGraphEventWaitNodeGetEventLayer(hipGraphNode_t node, hipEvent_t* event_out) {
  auto* _rec = HipGetActiveRecordExt(141u);
  auto _r = g_next.hipGraphEventWaitNodeGetEvent_fn(node, event_out);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 142
static hipError_t hipGraphEventWaitNodeSetEventLayer(hipGraphNode_t node, hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(142u);
  auto _r = g_next.hipGraphEventWaitNodeSetEvent_fn(node, event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 143
static hipError_t hipGraphExecChildGraphNodeSetParamsLayer(hipGraphExec_t hGraphExec,
                                                            hipGraphNode_t node,
                                                            hipGraph_t childGraph) {
  auto* _rec = HipGetActiveRecordExt(143u);
  auto _r = g_next.hipGraphExecChildGraphNodeSetParams_fn(hGraphExec, node, childGraph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 144
static hipError_t hipGraphExecDestroyLayer(hipGraphExec_t graphExec) {
  auto* _rec = HipGetActiveRecordExt(144u);
  HipEraseGraphExecNodesExt(graphExec);
  auto _r = g_next.hipGraphExecDestroy_fn(graphExec);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 145
static hipError_t hipGraphExecEventRecordNodeSetEventLayer(hipGraphExec_t hGraphExec,
                                                            hipGraphNode_t hNode, hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(145u);
  auto _r = g_next.hipGraphExecEventRecordNodeSetEvent_fn(hGraphExec, hNode, event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 146
static hipError_t hipGraphExecEventWaitNodeSetEventLayer(hipGraphExec_t hGraphExec,
                                                          hipGraphNode_t hNode, hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(146u);
  auto _r = g_next.hipGraphExecEventWaitNodeSetEvent_fn(hGraphExec, hNode, event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 147
static hipError_t hipGraphExecHostNodeSetParamsLayer(hipGraphExec_t hGraphExec,
                                                      hipGraphNode_t node,
                                                      const hipHostNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(147u);
  auto _r = g_next.hipGraphExecHostNodeSetParams_fn(hGraphExec, node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 148
static hipError_t hipGraphExecKernelNodeSetParamsLayer(hipGraphExec_t hGraphExec,
                                                        hipGraphNode_t node,
                                                        const hipKernelNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(148u);
  auto _r = g_next.hipGraphExecKernelNodeSetParams_fn(hGraphExec, node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 149
static hipError_t hipGraphExecMemcpyNodeSetParamsLayer(hipGraphExec_t hGraphExec,
                                                        hipGraphNode_t node,
                                                        hipMemcpy3DParms* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(149u);
  auto _r = g_next.hipGraphExecMemcpyNodeSetParams_fn(hGraphExec, node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 150
static hipError_t hipGraphExecMemcpyNodeSetParams1DLayer(hipGraphExec_t hGraphExec,
                                                          hipGraphNode_t node, void* dst,
                                                          const void* src, size_t count,
                                                          hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(150u);
  auto _r = g_next.hipGraphExecMemcpyNodeSetParams1D_fn(hGraphExec, node, dst, src, count, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 151
static hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbolLayer(hipGraphExec_t hGraphExec,
                                                                  hipGraphNode_t node, void* dst,
                                                                  const void* symbol, size_t count,
                                                                  size_t offset,
                                                                  hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(151u);
  auto _r = g_next.hipGraphExecMemcpyNodeSetParamsFromSymbol_fn(hGraphExec, node, dst, symbol, count, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 152
static hipError_t hipGraphExecMemcpyNodeSetParamsToSymbolLayer(hipGraphExec_t hGraphExec,
                                                                hipGraphNode_t node,
                                                                const void* symbol, const void* src,
                                                                size_t count, size_t offset,
                                                                hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(152u);
  auto _r = g_next.hipGraphExecMemcpyNodeSetParamsToSymbol_fn(hGraphExec, node, symbol, src, count, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 153
static hipError_t hipGraphExecMemsetNodeSetParamsLayer(hipGraphExec_t hGraphExec,
                                                        hipGraphNode_t node,
                                                        const hipMemsetParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(153u);
  auto _r = g_next.hipGraphExecMemsetNodeSetParams_fn(hGraphExec, node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 154
static hipError_t hipGraphExecUpdateLayer(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                                           hipGraphNode_t* hErrorNode_out,
                                           hipGraphExecUpdateResult* updateResult_out) {
  auto* _rec = HipGetActiveRecordExt(154u);
  auto _r = g_next.hipGraphExecUpdate_fn(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 155
static hipError_t hipGraphGetEdgesLayer(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                                         size_t* numEdges) {
  auto* _rec = HipGetActiveRecordExt(155u);
  auto _r = g_next.hipGraphGetEdges_fn(graph, from, to, numEdges);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 156
static hipError_t hipGraphGetNodesLayer(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
  auto* _rec = HipGetActiveRecordExt(156u);
  auto _r = g_next.hipGraphGetNodes_fn(graph, nodes, numNodes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 157
static hipError_t hipGraphGetRootNodesLayer(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                             size_t* pNumRootNodes) {
  auto* _rec = HipGetActiveRecordExt(157u);
  auto _r = g_next.hipGraphGetRootNodes_fn(graph, pRootNodes, pNumRootNodes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 158
static hipError_t hipGraphHostNodeGetParamsLayer(hipGraphNode_t node,
                                                  hipHostNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(158u);
  auto _r = g_next.hipGraphHostNodeGetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 159
static hipError_t hipGraphHostNodeSetParamsLayer(hipGraphNode_t node,
                                                  const hipHostNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(159u);
  auto _r = g_next.hipGraphHostNodeSetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// Map hipMemcpyKind + array/rect flags to HipCopyKindExt.
static HipCopyKindExt GraphMemcpyKindToExt(hipMemcpyKind kind,
                                            bool src_is_array, bool dst_is_array,
                                            bool is_rect) {
  switch (kind) {
    case hipMemcpyHostToDevice:
      return dst_is_array ? HIP_COPY_KIND_H2D_IMAGE_EXT
           : is_rect      ? HIP_COPY_KIND_H2D_RECT_EXT
                          : HIP_COPY_KIND_H2D_EXT;
    case hipMemcpyDeviceToHost:
      return src_is_array ? HIP_COPY_KIND_D2H_IMAGE_EXT
           : is_rect      ? HIP_COPY_KIND_D2H_RECT_EXT
                          : HIP_COPY_KIND_D2H_EXT;
    case hipMemcpyDeviceToDevice:
      if (src_is_array && dst_is_array) return HIP_COPY_KIND_D2D_IMAGE_EXT;
      if (dst_is_array) return HIP_COPY_KIND_BUFFER_TO_IMAGE_EXT;
      if (src_is_array) return HIP_COPY_KIND_IMAGE_TO_BUFFER_EXT;
      return is_rect ? HIP_COPY_KIND_D2D_RECT_EXT : HIP_COPY_KIND_D2D_EXT;
    default:
      return HIP_COPY_KIND_UNKNOWN_EXT;
  }
}

// Capture kernel and memcpy node info from a graph, store per graphExec.
// Called after a successful hipGraphInstantiate* — pGraphExec is already set.
// Uses internal C++ graph APIs directly to avoid calling public HIP functions
// from within a wrapper (which would clobber hipGetLastError and risk reentrancy).
static void CaptureGraphExecNodes(hipGraphExec_t exec, hipGraph_t graph) {
  if (!exec || !graph) return;
  auto* g = reinterpret_cast<hip::Graph*>(graph);
  if (!hip::Graph::isGraphValid(g)) return;
  const auto& nodes = g->GetNodes();
  if (nodes.empty()) return;

  std::vector<HipGraphNodeInfoExt> infos;
  for (auto* node : nodes) {
    if (!node) continue;
    switch (node->GetType()) {
      case hipGraphNodeTypeKernel: {
        auto* knode = static_cast<hip::GraphKernelNode*>(node);
        hipKernelNodeParams kp{};
        knode->GetParams(&kp);
        if (!kp.func) continue;
        hipFunction_t hfunc = nullptr;
        if (hip::ihipGetFuncBySymbol(&hfunc, kp.func) != hipSuccess || !hfunc) {
          // Stream-captured graphs store a hipFunction_t (not a host symbol pointer) in kp.func.
          // ihipGetFuncBySymbol fails for these — fall back to casting directly, mirroring
          // GraphKernelNode::getFunc().
          hfunc = static_cast<hipFunction_t>(const_cast<void*>(kp.func));
          if (!hip::asKernel(hfunc)) continue;
        }
        HipGraphNodeInfoExt info;
        info.gpu.op      = HIP_OP_DISPATCH_EXT;
        info.gpu.grid_x  = kp.gridDim.x;
        info.gpu.grid_y  = kp.gridDim.y;
        info.gpu.grid_z  = kp.gridDim.z;
        info.gpu.block_x = kp.blockDim.x;
        info.gpu.block_y = kp.blockDim.y;
        info.gpu.block_z = kp.blockDim.z;
        HipCaptureGraphNodeArgsExt(&info, hfunc, kp.kernelParams);
        // For stream-captured graphs kp.kernelParams is NULL but kp.extra holds
        // the packed buffer.  Try the packed path as a fallback.
        if (info.gpu.kernel_args == nullptr && kp.extra) {
          const void* kbuf; size_t ksz;
          if (ParseKernelExtra(kp.extra, kbuf, ksz))
            HipCaptureKernelArgsPackedExt(&info.gpu, hfunc, kbuf, ksz);
        }
        if (info.gpu.kernel_name != nullptr) {
          infos.push_back(std::move(info));
        }
        break;
      }
      case hipGraphNodeTypeMemcpy: {
        auto* mnode = static_cast<hip::GraphMemcpyNode*>(node);
        hipMemcpy3DParms mp{};
        mnode->GetParams(&mp);
        bool src_arr = (mp.srcArray != nullptr);
        bool dst_arr = (mp.dstArray != nullptr);
        bool is_rect = (mp.extent.height > 1 || mp.extent.depth > 1);
        HipGraphNodeInfoExt info;
        info.gpu.op        = HIP_OP_COPY_EXT;
        info.gpu.src       = src_arr ? nullptr : mp.srcPtr.ptr;
        info.gpu.dst       = dst_arr ? nullptr : mp.dstPtr.ptr;
        info.gpu.bytes     = mp.extent.width * mp.extent.height * mp.extent.depth;
        info.gpu.copy_kind = GraphMemcpyKindToExt(mp.kind, src_arr, dst_arr, is_rect);
        infos.push_back(std::move(info));
        break;
      }
      default: break;
    }
  }
  if (!infos.empty()) {
    HipStoreGraphExecNodesExt(exec, std::move(infos));
  }
}

// api_id = 160
static hipError_t hipGraphInstantiateLayer(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                            hipGraphNode_t* pErrorNode, char* pLogBuffer,
                                            size_t bufferSize) {
  auto* _rec = HipGetActiveRecordExt(160u);
  auto _r = g_next.hipGraphInstantiate_fn(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
  _rec->end_ns = NowNs();
  if (_r == hipSuccess && pGraphExec && *pGraphExec)
    CaptureGraphExecNodes(*pGraphExec, graph);
  return _r;
}

// api_id = 161
static hipError_t hipGraphInstantiateWithFlagsLayer(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                                     unsigned long long flags) {
  auto* _rec = HipGetActiveRecordExt(161u);
  auto _r = g_next.hipGraphInstantiateWithFlags_fn(pGraphExec, graph, flags);
  _rec->end_ns = NowNs();
  if (_r == hipSuccess && pGraphExec && *pGraphExec)
    CaptureGraphExecNodes(*pGraphExec, graph);
  return _r;
}

// api_id = 162
static hipError_t hipGraphInstantiateWithParamsLayer(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                                      hipGraphInstantiateParams* instantiateParams) {
  auto* _rec = HipGetActiveRecordExt(162u);
  auto _r = g_next.hipGraphInstantiateWithParams_fn(pGraphExec, graph, instantiateParams);
  _rec->end_ns = NowNs();
  if (_r == hipSuccess && pGraphExec && *pGraphExec)
    CaptureGraphExecNodes(*pGraphExec, graph);
  return _r;
}

// api_id = 163
static hipError_t hipGraphKernelNodeCopyAttributesLayer(hipGraphNode_t hSrc, hipGraphNode_t hDst) {
  auto* _rec = HipGetActiveRecordExt(163u);
  auto _r = g_next.hipGraphKernelNodeCopyAttributes_fn(hSrc, hDst);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 164
static hipError_t hipGraphKernelNodeGetAttributeLayer(hipGraphNode_t hNode,
                                                       hipKernelNodeAttrID attr,
                                                       hipKernelNodeAttrValue* value) {
  auto* _rec = HipGetActiveRecordExt(164u);
  auto _r = g_next.hipGraphKernelNodeGetAttribute_fn(hNode, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 165
static hipError_t hipGraphKernelNodeGetParamsLayer(hipGraphNode_t node,
                                                    hipKernelNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(165u);
  auto _r = g_next.hipGraphKernelNodeGetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 166
static hipError_t hipGraphKernelNodeSetAttributeLayer(hipGraphNode_t hNode,
                                                       hipKernelNodeAttrID attr,
                                                       const hipKernelNodeAttrValue* value) {
  auto* _rec = HipGetActiveRecordExt(166u);
  auto _r = g_next.hipGraphKernelNodeSetAttribute_fn(hNode, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 167
static hipError_t hipGraphKernelNodeSetParamsLayer(hipGraphNode_t node,
                                                    const hipKernelNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(167u);
  auto _r = g_next.hipGraphKernelNodeSetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 168
static hipError_t hipGraphLaunchLayer(hipGraphExec_t graphExec, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(168u);
  _rec->stream   = stream;
  _rec->memory1  = reinterpret_cast<void*>(graphExec);  // stash exec for callback lookup
  auto _r = g_next.hipGraphLaunch_fn(graphExec, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 169
static hipError_t hipGraphMemAllocNodeGetParamsLayer(hipGraphNode_t node,
                                                      hipMemAllocNodeParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(169u);
  auto _r = g_next.hipGraphMemAllocNodeGetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 170
static hipError_t hipGraphMemFreeNodeGetParamsLayer(hipGraphNode_t node, void* dev_ptr) {
  auto* _rec = HipGetActiveRecordExt(170u);
  auto _r = g_next.hipGraphMemFreeNodeGetParams_fn(node, dev_ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 171
static hipError_t hipGraphMemcpyNodeGetParamsLayer(hipGraphNode_t node,
                                                    hipMemcpy3DParms* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(171u);
  auto _r = g_next.hipGraphMemcpyNodeGetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 172
static hipError_t hipGraphMemcpyNodeSetParamsLayer(hipGraphNode_t node,
                                                    const hipMemcpy3DParms* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(172u);
  auto _r = g_next.hipGraphMemcpyNodeSetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 173
static hipError_t hipGraphMemcpyNodeSetParams1DLayer(hipGraphNode_t node, void* dst,
                                                      const void* src, size_t count,
                                                      hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(173u);
  auto _r = g_next.hipGraphMemcpyNodeSetParams1D_fn(node, dst, src, count, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 174
static hipError_t hipGraphMemcpyNodeSetParamsFromSymbolLayer(hipGraphNode_t node, void* dst,
                                                              const void* symbol, size_t count,
                                                              size_t offset, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(174u);
  auto _r = g_next.hipGraphMemcpyNodeSetParamsFromSymbol_fn(node, dst, symbol, count, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 175
static hipError_t hipGraphMemcpyNodeSetParamsToSymbolLayer(hipGraphNode_t node, const void* symbol,
                                                            const void* src, size_t count,
                                                            size_t offset, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(175u);
  auto _r = g_next.hipGraphMemcpyNodeSetParamsToSymbol_fn(node, symbol, src, count, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 176
static hipError_t hipGraphMemsetNodeGetParamsLayer(hipGraphNode_t node,
                                                    hipMemsetParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(176u);
  auto _r = g_next.hipGraphMemsetNodeGetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 177
static hipError_t hipGraphMemsetNodeSetParamsLayer(hipGraphNode_t node,
                                                    const hipMemsetParams* pNodeParams) {
  auto* _rec = HipGetActiveRecordExt(177u);
  auto _r = g_next.hipGraphMemsetNodeSetParams_fn(node, pNodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 178
static hipError_t hipGraphNodeFindInCloneLayer(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                                hipGraph_t clonedGraph) {
  auto* _rec = HipGetActiveRecordExt(178u);
  auto _r = g_next.hipGraphNodeFindInClone_fn(pNode, originalNode, clonedGraph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 179
static hipError_t hipGraphNodeGetDependenciesLayer(hipGraphNode_t node,
                                                    hipGraphNode_t* pDependencies,
                                                    size_t* pNumDependencies) {
  auto* _rec = HipGetActiveRecordExt(179u);
  auto _r = g_next.hipGraphNodeGetDependencies_fn(node, pDependencies, pNumDependencies);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 180
static hipError_t hipGraphNodeGetDependentNodesLayer(hipGraphNode_t node,
                                                      hipGraphNode_t* pDependentNodes,
                                                      size_t* pNumDependentNodes) {
  auto* _rec = HipGetActiveRecordExt(180u);
  auto _r = g_next.hipGraphNodeGetDependentNodes_fn(node, pDependentNodes, pNumDependentNodes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 181
static hipError_t hipGraphNodeGetEnabledLayer(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               unsigned int* isEnabled) {
  auto* _rec = HipGetActiveRecordExt(181u);
  auto _r = g_next.hipGraphNodeGetEnabled_fn(hGraphExec, hNode, isEnabled);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 182
static hipError_t hipGraphNodeGetTypeLayer(hipGraphNode_t node, hipGraphNodeType* pType) {
  auto* _rec = HipGetActiveRecordExt(182u);
  auto _r = g_next.hipGraphNodeGetType_fn(node, pType);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 183
static hipError_t hipGraphNodeSetEnabledLayer(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               unsigned int isEnabled) {
  auto* _rec = HipGetActiveRecordExt(183u);
  auto _r = g_next.hipGraphNodeSetEnabled_fn(hGraphExec, hNode, isEnabled);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 184
static hipError_t hipGraphReleaseUserObjectLayer(hipGraph_t graph, hipUserObject_t object,
                                                  unsigned int count) {
  auto* _rec = HipGetActiveRecordExt(184u);
  auto _r = g_next.hipGraphReleaseUserObject_fn(graph, object, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 185
static hipError_t hipGraphRemoveDependenciesLayer(hipGraph_t graph, const hipGraphNode_t* from,
                                                   const hipGraphNode_t* to,
                                                   size_t numDependencies) {
  auto* _rec = HipGetActiveRecordExt(185u);
  auto _r = g_next.hipGraphRemoveDependencies_fn(graph, from, to, numDependencies);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 186
static hipError_t hipGraphRetainUserObjectLayer(hipGraph_t graph, hipUserObject_t object,
                                                 unsigned int count, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(186u);
  auto _r = g_next.hipGraphRetainUserObject_fn(graph, object, count, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 187
static hipError_t hipGraphUploadLayer(hipGraphExec_t graphExec, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(187u);
  _rec->stream = stream;
  auto _r = g_next.hipGraphUpload_fn(graphExec, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 188
static hipError_t hipGraphicsGLRegisterBufferLayer(hipGraphicsResource** resource, GLuint buffer,
                                                    unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(188u);
  auto _r = g_next.hipGraphicsGLRegisterBuffer_fn(resource, buffer, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 189
static hipError_t hipGraphicsGLRegisterImageLayer(hipGraphicsResource** resource, GLuint image,
                                                   GLenum target, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(189u);
  auto _r = g_next.hipGraphicsGLRegisterImage_fn(resource, image, target, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 190
static hipError_t hipGraphicsMapResourcesLayer(int count, hipGraphicsResource_t* resources,
                                                hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(190u);
  _rec->stream = stream;
  auto _r = g_next.hipGraphicsMapResources_fn(count, resources, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 191
static hipError_t hipGraphicsResourceGetMappedPointerLayer(void** devPtr, size_t* size,
                                                            hipGraphicsResource_t resource) {
  auto* _rec = HipGetActiveRecordExt(191u);
  auto _r = g_next.hipGraphicsResourceGetMappedPointer_fn(devPtr, size, resource);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 192
static hipError_t hipGraphicsSubResourceGetMappedArrayLayer(hipArray_t* array,
                                                             hipGraphicsResource_t resource,
                                                             unsigned int arrayIndex,
                                                             unsigned int mipLevel) {
  auto* _rec = HipGetActiveRecordExt(192u);
  auto _r = g_next.hipGraphicsSubResourceGetMappedArray_fn(array, resource, arrayIndex, mipLevel);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 193
static hipError_t hipGraphicsUnmapResourcesLayer(int count, hipGraphicsResource_t* resources,
                                                  hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(193u);
  _rec->stream = stream;
  auto _r = g_next.hipGraphicsUnmapResources_fn(count, resources, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 194
static hipError_t hipGraphicsUnregisterResourceLayer(hipGraphicsResource_t resource) {
  auto* _rec = HipGetActiveRecordExt(194u);
  auto _r = g_next.hipGraphicsUnregisterResource_fn(resource);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 195
static hipError_t hipHostAllocLayer(void** ptr, size_t size, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(195u);
  auto _r = g_next.hipHostAlloc_fn(ptr, size, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 196
static hipError_t hipHostFreeLayer(void* ptr) {
  auto* _rec = HipGetActiveRecordExt(196u);
  auto _r = g_next.hipHostFree_fn(ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 197
static hipError_t hipHostGetDevicePointerLayer(void** devPtr, void* hstPtr, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(197u);
  auto _r = g_next.hipHostGetDevicePointer_fn(devPtr, hstPtr, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 198
static hipError_t hipHostGetFlagsLayer(unsigned int* flagsPtr, void* hostPtr) {
  auto* _rec = HipGetActiveRecordExt(198u);
  auto _r = g_next.hipHostGetFlags_fn(flagsPtr, hostPtr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 199
static hipError_t hipHostMallocLayer(void** ptr, size_t size, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(199u);
  auto _r = g_next.hipHostMalloc_fn(ptr, size, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 200
static hipError_t hipExtHostAllocLayer(void** ptr, size_t size, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(200u);
  auto _r = g_next.hipExtHostAlloc_fn(ptr, size, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 201
static hipError_t hipHostRegisterLayer(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(201u);
  auto _r = g_next.hipHostRegister_fn(hostPtr, sizeBytes, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 202
static hipError_t hipHostUnregisterLayer(void* hostPtr) {
  auto* _rec = HipGetActiveRecordExt(202u);
  auto _r = g_next.hipHostUnregister_fn(hostPtr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 203
static hipError_t hipImportExternalMemoryLayer(hipExternalMemory_t* extMem_out,
                                                const hipExternalMemoryHandleDesc* memHandleDesc) {
  auto* _rec = HipGetActiveRecordExt(203u);
  auto _r = g_next.hipImportExternalMemory_fn(extMem_out, memHandleDesc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 204
static hipError_t hipImportExternalSemaphoreLayer(hipExternalSemaphore_t* extSem_out, const hipExternalSemaphoreHandleDesc* semHandleDesc) {
  auto* _rec = HipGetActiveRecordExt(204u);
  auto _r = g_next.hipImportExternalSemaphore_fn(extSem_out, semHandleDesc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 205
static hipError_t hipInitLayer(unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(205u);
  auto _r = g_next.hipInit_fn(flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 206
static hipError_t hipIpcCloseMemHandleLayer(void* devPtr) {
  auto* _rec = HipGetActiveRecordExt(206u);
  auto _r = g_next.hipIpcCloseMemHandle_fn(devPtr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 207
static hipError_t hipIpcGetEventHandleLayer(hipIpcEventHandle_t* handle, hipEvent_t event) {
  auto* _rec = HipGetActiveRecordExt(207u);
  auto _r = g_next.hipIpcGetEventHandle_fn(handle, event);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 208
static hipError_t hipIpcGetMemHandleLayer(hipIpcMemHandle_t* handle, void* devPtr) {
  auto* _rec = HipGetActiveRecordExt(208u);
  auto _r = g_next.hipIpcGetMemHandle_fn(handle, devPtr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 209
static hipError_t hipIpcOpenEventHandleLayer(hipEvent_t* event, hipIpcEventHandle_t handle) {
  auto* _rec = HipGetActiveRecordExt(209u);
  auto _r = g_next.hipIpcOpenEventHandle_fn(event, handle);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 210
static hipError_t hipIpcOpenMemHandleLayer(void** devPtr, hipIpcMemHandle_t handle,
                                            unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(210u);
  auto _r = g_next.hipIpcOpenMemHandle_fn(devPtr, handle, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 211
static const char* hipKernelNameRefLayer(const hipFunction_t f) {
  auto* _rec = HipGetActiveRecordExt(211u);
  auto _r = g_next.hipKernelNameRef_fn(f);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 212
static const char* hipKernelNameRefByPtrLayer(const void* hostFunction, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(212u);
  _rec->stream = stream;
  auto _r = g_next.hipKernelNameRefByPtr_fn(hostFunction, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 213
static hipError_t hipLaunchByPtrLayer(const void* func) {
  auto* _rec = HipGetActiveRecordExt(213u);
  // Grid/block were stashed by __hipPushCallConfigurationLayer (<<<>>> lowering).
  _rec->gpu.grid_x = g_pushed_grid.x;  _rec->gpu.grid_y = g_pushed_grid.y;  _rec->gpu.grid_z = g_pushed_grid.z;
  _rec->gpu.block_x = g_pushed_block.x; _rec->gpu.block_y = g_pushed_block.y; _rec->gpu.block_z = g_pushed_block.z;
  auto _r = g_next.hipLaunchByPtr_fn(func);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 214
static hipError_t hipLaunchCooperativeKernelLayer(const void* f, dim3 gridDim, dim3 blockDimX,
                                                   void** kernelParams, unsigned int sharedMemBytes,
                                                   hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(214u);
  _rec->stream = stream;
  _rec->gpu.grid_x = gridDim.x;  _rec->gpu.grid_y = gridDim.y;  _rec->gpu.grid_z = gridDim.z;
  _rec->gpu.block_x = blockDimX.x; _rec->gpu.block_y = blockDimX.y; _rec->gpu.block_z = blockDimX.z;
  if (kernelParams) {
    hipFunction_t hfunc = nullptr;
    if (hip::ihipGetFuncBySymbol(&hfunc, f) == hipSuccess)
      HipCaptureKernelArgsExt(&_rec->gpu, hfunc, kernelParams);
  }
  auto _r = g_next.hipLaunchCooperativeKernel_fn(f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 215
static hipError_t hipLaunchCooperativeKernelMultiDeviceLayer(hipLaunchParams* launchParamsList,
                                                              int numDevices, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(215u);
  auto _r = g_next.hipLaunchCooperativeKernelMultiDevice_fn(launchParamsList, numDevices, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 216
static hipError_t hipLaunchHostFuncLayer(hipStream_t stream, hipHostFn_t fn, void* userData) {
  auto* _rec = HipGetActiveRecordExt(216u);
  _rec->stream = stream;
  auto _r = g_next.hipLaunchHostFunc_fn(stream, fn, userData);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 217
static hipError_t hipLaunchKernelLayer(const void* function_address, dim3 numBlocks,
                                        dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                        hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(217u);
  _rec->stream = stream;
  _rec->gpu.grid_x = numBlocks.x; _rec->gpu.grid_y = numBlocks.y; _rec->gpu.grid_z = numBlocks.z;
  _rec->gpu.block_x = dimBlocks.x; _rec->gpu.block_y = dimBlocks.y; _rec->gpu.block_z = dimBlocks.z;
  if (args) {
    hipFunction_t hfunc = nullptr;
    if (hip::ihipGetFuncBySymbol(&hfunc, function_address) == hipSuccess)
      HipCaptureKernelArgsExt(&_rec->gpu, hfunc, args);
  }
  auto _r = g_next.hipLaunchKernel_fn(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 218
static hipError_t hipMallocLayer(void** ptr, size_t size) {
  auto* _rec = HipGetActiveRecordExt(218u);
  _rec->size = size;
  auto _r = g_next.hipMalloc_fn(ptr, size);
  if (ptr) _rec->memory1 = *ptr;
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 219
static hipError_t hipMalloc3DLayer(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  auto* _rec = HipGetActiveRecordExt(219u);
  auto _r = g_next.hipMalloc3D_fn(pitchedDevPtr, extent);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 220
static hipError_t hipMalloc3DArrayLayer(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                                         struct hipExtent extent, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(220u);
  auto _r = g_next.hipMalloc3DArray_fn(array, desc, extent, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 221
static hipError_t hipMallocArrayLayer(hipArray_t* array, const hipChannelFormatDesc* desc,
                                       size_t width, size_t height, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(221u);
  auto _r = g_next.hipMallocArray_fn(array, desc, width, height, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 222
static hipError_t hipMallocAsyncLayer(void** dev_ptr, size_t size, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(222u);
  _rec->stream = stream;
  _rec->size = size;
  auto _r = g_next.hipMallocAsync_fn(dev_ptr, size, stream);
  if (dev_ptr) _rec->memory1 = *dev_ptr;
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 223
static hipError_t hipMallocFromPoolAsyncLayer(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                               hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(223u);
  _rec->stream = stream;
  _rec->size = size;
  auto _r = g_next.hipMallocFromPoolAsync_fn(dev_ptr, size, mem_pool, stream);
  if (dev_ptr) _rec->memory1 = *dev_ptr;
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 224
static hipError_t hipMallocHostLayer(void** ptr, size_t size) {
  auto* _rec = HipGetActiveRecordExt(224u);
  _rec->size = size;
  auto _r = g_next.hipMallocHost_fn(ptr, size);
  if (ptr) _rec->memory1 = *ptr;
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 225
static hipError_t hipMallocManagedLayer(void** dev_ptr, size_t size, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(225u);
  _rec->size = size;
  auto _r = g_next.hipMallocManaged_fn(dev_ptr, size, flags);
  if (dev_ptr) _rec->memory1 = *dev_ptr;
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 226
static hipError_t hipMallocMipmappedArrayLayer(hipMipmappedArray_t* mipmappedArray,
                                                const struct hipChannelFormatDesc* desc,
                                                struct hipExtent extent, unsigned int numLevels,
                                                unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(226u);
  auto _r = g_next.hipMallocMipmappedArray_fn(mipmappedArray, desc, extent, numLevels, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 227
static hipError_t hipMallocPitchLayer(void** ptr, size_t* pitch, size_t width, size_t height) {
  auto* _rec = HipGetActiveRecordExt(227u);
  auto _r = g_next.hipMallocPitch_fn(ptr, pitch, width, height);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 228
static hipError_t hipMemAddressFreeLayer(void* devPtr, size_t size) {
  auto* _rec = HipGetActiveRecordExt(228u);
  auto _r = g_next.hipMemAddressFree_fn(devPtr, size);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 229
static hipError_t hipMemAddressReserveLayer(void** ptr, size_t size, size_t alignment, void* addr,
                                             unsigned long long flags) {
  auto* _rec = HipGetActiveRecordExt(229u);
  auto _r = g_next.hipMemAddressReserve_fn(ptr, size, alignment, addr, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 230
static hipError_t hipMemAdviseLayer(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                                     int device) {
  auto* _rec = HipGetActiveRecordExt(230u);
  auto _r = g_next.hipMemAdvise_fn(dev_ptr, count, advice, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 231
static hipError_t hipMemAdvise_v2Layer(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                                        hipMemLocation device) {
  auto* _rec = HipGetActiveRecordExt(231u);
  auto _r = g_next.hipMemAdvise_v2_fn(dev_ptr, count, advice, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 232
static hipError_t hipMemAllocHostLayer(void** ptr, size_t size) {
  auto* _rec = HipGetActiveRecordExt(232u);
  auto _r = g_next.hipMemAllocHost_fn(ptr, size);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 233
static hipError_t hipMemAllocPitchLayer(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes,
                                         size_t height, unsigned int elementSizeBytes) {
  auto* _rec = HipGetActiveRecordExt(233u);
  auto _r = g_next.hipMemAllocPitch_fn(dptr, pitch, widthInBytes, height, elementSizeBytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 234
static hipError_t hipMemCreateLayer(hipMemGenericAllocationHandle_t* handle, size_t size,
                                     const hipMemAllocationProp* prop, unsigned long long flags) {
  auto* _rec = HipGetActiveRecordExt(234u);
  auto _r = g_next.hipMemCreate_fn(handle, size, prop, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 235
static hipError_t hipMemExportToShareableHandleLayer(void* shareableHandle,
                                                      hipMemGenericAllocationHandle_t handle,
                                                      hipMemAllocationHandleType handleType,
                                                      unsigned long long flags) {
  auto* _rec = HipGetActiveRecordExt(235u);
  auto _r = g_next.hipMemExportToShareableHandle_fn(shareableHandle, handle, handleType, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 236
static hipError_t hipMemGetAccessLayer(unsigned long long* flags, const hipMemLocation* location,
                                        void* ptr) {
  auto* _rec = HipGetActiveRecordExt(236u);
  auto _r = g_next.hipMemGetAccess_fn(flags, location, ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 237
static hipError_t hipMemGetAddressRangeLayer(hipDeviceptr_t* pbase, size_t* psize,
                                              hipDeviceptr_t dptr) {
  auto* _rec = HipGetActiveRecordExt(237u);
  auto _r = g_next.hipMemGetAddressRange_fn(pbase, psize, dptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 238
static hipError_t hipMemGetAllocationGranularityLayer(size_t* granularity,
                                                       const hipMemAllocationProp* prop,
                                                       hipMemAllocationGranularity_flags option) {
  auto* _rec = HipGetActiveRecordExt(238u);
  auto _r = g_next.hipMemGetAllocationGranularity_fn(granularity, prop, option);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 239
static hipError_t hipMemGetAllocationPropertiesFromHandleLayer(hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle) {
  auto* _rec = HipGetActiveRecordExt(239u);
  auto _r = g_next.hipMemGetAllocationPropertiesFromHandle_fn(prop, handle);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 240
static hipError_t hipMemGetInfoLayer(size_t* free, size_t* total) {
  auto* _rec = HipGetActiveRecordExt(240u);
  auto _r = g_next.hipMemGetInfo_fn(free, total);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 241
static hipError_t hipMemImportFromShareableHandleLayer(hipMemGenericAllocationHandle_t* handle,
                                                        void* osHandle,
                                                        hipMemAllocationHandleType shHandleType) {
  auto* _rec = HipGetActiveRecordExt(241u);
  auto _r = g_next.hipMemImportFromShareableHandle_fn(handle, osHandle, shHandleType);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 242
static hipError_t hipMemMapLayer(void* ptr, size_t size, size_t offset,
                                  hipMemGenericAllocationHandle_t handle, unsigned long long flags) {
  auto* _rec = HipGetActiveRecordExt(242u);
  auto _r = g_next.hipMemMap_fn(ptr, size, offset, handle, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 243
static hipError_t hipMemMapArrayAsyncLayer(hipArrayMapInfo* mapInfoList, unsigned int count,
                                            hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(243u);
  _rec->stream = stream;
  auto _r = g_next.hipMemMapArrayAsync_fn(mapInfoList, count, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 244
static hipError_t hipMemPoolCreateLayer(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  auto* _rec = HipGetActiveRecordExt(244u);
  auto _r = g_next.hipMemPoolCreate_fn(mem_pool, pool_props);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 245
static hipError_t hipMemPoolDestroyLayer(hipMemPool_t mem_pool) {
  auto* _rec = HipGetActiveRecordExt(245u);
  auto _r = g_next.hipMemPoolDestroy_fn(mem_pool);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 246
static hipError_t hipMemPoolExportPointerLayer(hipMemPoolPtrExportData* export_data,
                                                void* dev_ptr) {
  auto* _rec = HipGetActiveRecordExt(246u);
  auto _r = g_next.hipMemPoolExportPointer_fn(export_data, dev_ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 247
static hipError_t hipMemPoolExportToShareableHandleLayer(void* shared_handle,
                                                          hipMemPool_t mem_pool,
                                                          hipMemAllocationHandleType handle_type,
                                                          unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(247u);
  auto _r = g_next.hipMemPoolExportToShareableHandle_fn(shared_handle, mem_pool, handle_type, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 248
static hipError_t hipMemPoolGetAccessLayer(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                                            hipMemLocation* location) {
  auto* _rec = HipGetActiveRecordExt(248u);
  auto _r = g_next.hipMemPoolGetAccess_fn(flags, mem_pool, location);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 249
static hipError_t hipMemPoolGetAttributeLayer(hipMemPool_t mem_pool, hipMemPoolAttr attr,
                                               void* value) {
  auto* _rec = HipGetActiveRecordExt(249u);
  auto _r = g_next.hipMemPoolGetAttribute_fn(mem_pool, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 250
static hipError_t hipMemPoolImportFromShareableHandleLayer(hipMemPool_t* mem_pool,
                                                            void* shared_handle,
                                                            hipMemAllocationHandleType handle_type,
                                                            unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(250u);
  auto _r = g_next.hipMemPoolImportFromShareableHandle_fn(mem_pool, shared_handle, handle_type, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 251
static hipError_t hipMemPoolImportPointerLayer(void** dev_ptr, hipMemPool_t mem_pool,
                                                hipMemPoolPtrExportData* export_data) {
  auto* _rec = HipGetActiveRecordExt(251u);
  auto _r = g_next.hipMemPoolImportPointer_fn(dev_ptr, mem_pool, export_data);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 252
static hipError_t hipMemPoolSetAccessLayer(hipMemPool_t mem_pool,
                                            const hipMemAccessDesc* desc_list, size_t count) {
  auto* _rec = HipGetActiveRecordExt(252u);
  auto _r = g_next.hipMemPoolSetAccess_fn(mem_pool, desc_list, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 253
static hipError_t hipMemPoolSetAttributeLayer(hipMemPool_t mem_pool, hipMemPoolAttr attr,
                                               void* value) {
  auto* _rec = HipGetActiveRecordExt(253u);
  auto _r = g_next.hipMemPoolSetAttribute_fn(mem_pool, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 254
static hipError_t hipMemPoolTrimToLayer(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  auto* _rec = HipGetActiveRecordExt(254u);
  auto _r = g_next.hipMemPoolTrimTo_fn(mem_pool, min_bytes_to_hold);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 255
static hipError_t hipMemPrefetchAsyncLayer(const void* dev_ptr, size_t count, int device,
                                            hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(255u);
  _rec->stream = stream;
  auto _r = g_next.hipMemPrefetchAsync_fn(dev_ptr, count, device, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 256
static hipError_t hipMemPrefetchAsync_v2Layer(const void* dev_ptr, size_t count,
                                               hipMemLocation location, unsigned int flags,
                                               hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(256u);
  _rec->stream = stream;
  auto _r = g_next.hipMemPrefetchAsync_v2_fn(dev_ptr, count, location, flags, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 257
static hipError_t hipMemPrefetchBatchAsyncLayer(void** dev_ptrs, size_t* sizes, size_t count,
                                                hipMemLocation* prefetch_locs, size_t* prefetch_loc_idxs,
                                                size_t num_prefetch_locs, unsigned long long flags,
                                                hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(257u);
  _rec->stream = stream;
  auto _r = g_next.hipMemPrefetchBatchAsync_fn(dev_ptrs, sizes, count, prefetch_locs, prefetch_loc_idxs, num_prefetch_locs, flags, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 258
static hipError_t hipMemPtrGetInfoLayer(void* ptr, size_t* size) {
  auto* _rec = HipGetActiveRecordExt(258u);
  auto _r = g_next.hipMemPtrGetInfo_fn(ptr, size);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 259
static hipError_t hipMemRangeGetAttributeLayer(void* data, size_t data_size,
                                                hipMemRangeAttribute attribute, const void* dev_ptr,
                                                size_t count) {
  auto* _rec = HipGetActiveRecordExt(259u);
  auto _r = g_next.hipMemRangeGetAttribute_fn(data, data_size, attribute, dev_ptr, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 260
static hipError_t hipMemRangeGetAttributesLayer(void** data, size_t* data_sizes,
                                                 hipMemRangeAttribute* attributes,
                                                 size_t num_attributes, const void* dev_ptr,
                                                 size_t count) {
  auto* _rec = HipGetActiveRecordExt(260u);
  auto _r = g_next.hipMemRangeGetAttributes_fn(data, data_sizes, attributes, num_attributes, dev_ptr, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 261
static hipError_t hipMemReleaseLayer(hipMemGenericAllocationHandle_t handle) {
  auto* _rec = HipGetActiveRecordExt(261u);
  auto _r = g_next.hipMemRelease_fn(handle);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 262
static hipError_t hipMemRetainAllocationHandleLayer(hipMemGenericAllocationHandle_t* handle,
                                                     void* addr) {
  auto* _rec = HipGetActiveRecordExt(262u);
  auto _r = g_next.hipMemRetainAllocationHandle_fn(handle, addr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 263
static hipError_t hipMemSetAccessLayer(void* ptr, size_t size, const hipMemAccessDesc* desc,
                                        size_t count) {
  auto* _rec = HipGetActiveRecordExt(263u);
  auto _r = g_next.hipMemSetAccess_fn(ptr, size, desc, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 264
static hipError_t hipMemUnmapLayer(void* ptr, size_t size) {
  auto* _rec = HipGetActiveRecordExt(264u);
  auto _r = g_next.hipMemUnmap_fn(ptr, size);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 265
static hipError_t hipMemcpyLayer(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(265u);
  _rec->memory1 = dst;
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpy_fn(dst, src, sizeBytes, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 266
static hipError_t hipMemcpy2DLayer(void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(266u);
  _rec->memory1 = dst;
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = width * height;
  auto _r = g_next.hipMemcpy2D_fn(dst, dpitch, src, spitch, width, height, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 267
static hipError_t hipMemcpy2DAsyncLayer(void* dst, size_t dpitch, const void* src, size_t spitch,
                                         size_t width, size_t height, hipMemcpyKind kind,
                                         hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(267u);
  _rec->stream = stream;
  _rec->memory1 = dst;
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = width * height;
  auto _r = g_next.hipMemcpy2DAsync_fn(dst, dpitch, src, spitch, width, height, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 268
static hipError_t hipMemcpy2DFromArrayLayer(void* dst, size_t dpitch, hipArray_const_t src,
                                             size_t wOffset, size_t hOffset, size_t width,
                                             size_t height, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(268u);
  auto _r = g_next.hipMemcpy2DFromArray_fn(dst, dpitch, src, wOffset, hOffset, width, height, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 269
static hipError_t hipMemcpy2DFromArrayAsyncLayer(void* dst, size_t dpitch, hipArray_const_t src,
                                                  size_t wOffset, size_t hOffset, size_t width,
                                                  size_t height, hipMemcpyKind kind,
                                                  hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(269u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy2DFromArrayAsync_fn(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 270
static hipError_t hipMemcpy2DToArrayLayer(hipArray_t dst, size_t wOffset, size_t hOffset,
                                           const void* src, size_t spitch, size_t width,
                                           size_t height, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(270u);
  auto _r = g_next.hipMemcpy2DToArray_fn(dst, wOffset, hOffset, src, spitch, width, height, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 271
static hipError_t hipMemcpy2DToArrayAsyncLayer(hipArray_t dst, size_t wOffset, size_t hOffset,
                                                const void* src, size_t spitch, size_t width,
                                                size_t height, hipMemcpyKind kind,
                                                hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(271u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy2DToArrayAsync_fn(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 272
static hipError_t hipMemcpy3DLayer(const struct hipMemcpy3DParms* p) {
  auto* _rec = HipGetActiveRecordExt(272u);
  auto _r = g_next.hipMemcpy3D_fn(p);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 273
static hipError_t hipMemcpy3DAsyncLayer(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(273u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy3DAsync_fn(p, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 274
static hipError_t hipMemcpyAsyncLayer(void* dst, const void* src, size_t sizeBytes,
                                       hipMemcpyKind kind, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(274u);
  _rec->stream = stream;
  _rec->memory1 = dst;
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyAsync_fn(dst, src, sizeBytes, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 275
static hipError_t hipMemcpyAtoHLayer(void* dst, hipArray_t srcArray, size_t srcOffset,
                                      size_t count) {
  auto* _rec = HipGetActiveRecordExt(275u);
  auto _r = g_next.hipMemcpyAtoH_fn(dst, srcArray, srcOffset, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 276
static hipError_t hipMemcpyDtoDLayer(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  auto* _rec = HipGetActiveRecordExt(276u);
  _rec->memory1 = reinterpret_cast<void*>(dst);
  _rec->memory2 = reinterpret_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyDtoD_fn(dst, src, sizeBytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 277
static hipError_t hipMemcpyDtoDAsyncLayer(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                                           hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(277u);
  _rec->stream = stream;
  _rec->memory1 = reinterpret_cast<void*>(dst);
  _rec->memory2 = reinterpret_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyDtoDAsync_fn(dst, src, sizeBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 278
static hipError_t hipMemcpyDtoHLayer(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  auto* _rec = HipGetActiveRecordExt(278u);
  _rec->memory1 = dst;
  _rec->memory2 = reinterpret_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyDtoH_fn(dst, src, sizeBytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 279
static hipError_t hipMemcpyDtoHAsyncLayer(void* dst, hipDeviceptr_t src, size_t sizeBytes,
                                           hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(279u);
  _rec->stream = stream;
  _rec->memory1 = dst;
  _rec->memory2 = reinterpret_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyDtoHAsync_fn(dst, src, sizeBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 280
static hipError_t hipMemcpyFromArrayLayer(void* dst, hipArray_const_t srcArray, size_t wOffset,
                                           size_t hOffset, size_t count, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(280u);
  auto _r = g_next.hipMemcpyFromArray_fn(dst, srcArray, wOffset, hOffset, count, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 281
static hipError_t hipMemcpyFromSymbolLayer(void* dst, const void* symbol, size_t sizeBytes,
                                            size_t offset, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(281u);
  auto _r = g_next.hipMemcpyFromSymbol_fn(dst, symbol, sizeBytes, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 282
static hipError_t hipMemcpyFromSymbolAsyncLayer(void* dst, const void* symbol, size_t sizeBytes,
                                                 size_t offset, hipMemcpyKind kind,
                                                 hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(282u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyFromSymbolAsync_fn(dst, symbol, sizeBytes, offset, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 283
static hipError_t hipMemcpyHtoALayer(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                                      size_t count) {
  auto* _rec = HipGetActiveRecordExt(283u);
  auto _r = g_next.hipMemcpyHtoA_fn(dstArray, dstOffset, srcHost, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 284
static hipError_t hipMemcpyHtoDLayer(hipDeviceptr_t dst, const void* src, size_t sizeBytes) {
  auto* _rec = HipGetActiveRecordExt(284u);
  _rec->memory1 = reinterpret_cast<void*>(dst);
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyHtoD_fn(dst, src, sizeBytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 285
static hipError_t hipMemcpyHtoDAsyncLayer(hipDeviceptr_t dst, const void* src, size_t sizeBytes,
                                           hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(285u);
  _rec->stream = stream;
  _rec->memory1 = reinterpret_cast<void*>(dst);
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyHtoDAsync_fn(dst, src, sizeBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 286
static hipError_t hipMemcpyParam2DLayer(const hip_Memcpy2D* pCopy) {
  auto* _rec = HipGetActiveRecordExt(286u);
  auto _r = g_next.hipMemcpyParam2D_fn(pCopy);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 287
static hipError_t hipMemcpyParam2DAsyncLayer(const hip_Memcpy2D* pCopy, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(287u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyParam2DAsync_fn(pCopy, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 288
static hipError_t hipMemcpyPeerLayer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                                      size_t sizeBytes) {
  auto* _rec = HipGetActiveRecordExt(288u);
  _rec->memory1 = dst;
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyPeer_fn(dst, dstDeviceId, src, srcDeviceId, sizeBytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 289
static hipError_t hipMemcpyPeerAsyncLayer(void* dst, int dstDeviceId, const void* src,
                                           int srcDevice, size_t sizeBytes, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(289u);
  _rec->stream = stream;
  _rec->memory1 = dst;
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyPeerAsync_fn(dst, dstDeviceId, src, srcDevice, sizeBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 290
static hipError_t hipMemcpyToArrayLayer(hipArray_t dst, size_t wOffset, size_t hOffset,
                                         const void* src, size_t count, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(290u);
  auto _r = g_next.hipMemcpyToArray_fn(dst, wOffset, hOffset, src, count, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 291
static hipError_t hipMemcpyToSymbolLayer(const void* symbol, const void* src, size_t sizeBytes,
                                          size_t offset, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(291u);
  auto _r = g_next.hipMemcpyToSymbol_fn(symbol, src, sizeBytes, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 292
static hipError_t hipMemcpyToSymbolAsyncLayer(const void* symbol, const void* src,
                                               size_t sizeBytes, size_t offset, hipMemcpyKind kind,
                                               hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(292u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyToSymbolAsync_fn(symbol, src, sizeBytes, offset, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 293
static hipError_t hipMemcpyWithStreamLayer(void* dst, const void* src, size_t sizeBytes,
                                            hipMemcpyKind kind, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(293u);
  _rec->stream = stream;
  _rec->memory1 = dst;
  _rec->memory2 = const_cast<void*>(src);
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemcpyWithStream_fn(dst, src, sizeBytes, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 294
static hipError_t hipMemsetLayer(void* dst, int value, size_t sizeBytes) {
  auto* _rec = HipGetActiveRecordExt(294u);
  _rec->memory1 = dst;
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemset_fn(dst, value, sizeBytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 295
static hipError_t hipMemset2DLayer(void* dst, size_t pitch, int value, size_t width,
                                    size_t height) {
  auto* _rec = HipGetActiveRecordExt(295u);
  _rec->memory1 = dst;
  _rec->size = width * height;
  auto _r = g_next.hipMemset2D_fn(dst, pitch, value, width, height);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 296
static hipError_t hipMemset2DAsyncLayer(void* dst, size_t pitch, int value, size_t width,
                                         size_t height, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(296u);
  _rec->stream = stream;
  _rec->memory1 = dst;
  _rec->size = width * height;
  auto _r = g_next.hipMemset2DAsync_fn(dst, pitch, value, width, height, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 297
static hipError_t hipMemset3DLayer(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  auto* _rec = HipGetActiveRecordExt(297u);
  auto _r = g_next.hipMemset3D_fn(pitchedDevPtr, value, extent);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 298
static hipError_t hipMemset3DAsyncLayer(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                         hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(298u);
  _rec->stream = stream;
  auto _r = g_next.hipMemset3DAsync_fn(pitchedDevPtr, value, extent, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 299
static hipError_t hipMemsetAsyncLayer(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(299u);
  _rec->stream = stream;
  _rec->memory1 = dst;
  _rec->size = sizeBytes;
  auto _r = g_next.hipMemsetAsync_fn(dst, value, sizeBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 300
static hipError_t hipMemsetD16Layer(hipDeviceptr_t dest, unsigned short value, size_t count) {
  auto* _rec = HipGetActiveRecordExt(300u);
  _rec->memory1 = reinterpret_cast<void*>(dest);
  _rec->size = count * 2;
  auto _r = g_next.hipMemsetD16_fn(dest, value, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 301
static hipError_t hipMemsetD16AsyncLayer(hipDeviceptr_t dest, unsigned short value, size_t count,
                                          hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(301u);
  _rec->stream = stream;
  _rec->memory1 = reinterpret_cast<void*>(dest);
  _rec->size = count * 2;
  auto _r = g_next.hipMemsetD16Async_fn(dest, value, count, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 302
static hipError_t hipMemsetD32Layer(hipDeviceptr_t dest, int value, size_t count) {
  auto* _rec = HipGetActiveRecordExt(302u);
  _rec->memory1 = reinterpret_cast<void*>(dest);
  _rec->size = count * 4;
  auto _r = g_next.hipMemsetD32_fn(dest, value, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 303
static hipError_t hipMemsetD32AsyncLayer(hipDeviceptr_t dst, int value, size_t count,
                                          hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(303u);
  _rec->stream = stream;
  _rec->memory1 = reinterpret_cast<void*>(dst);
  _rec->size = count * 4;
  auto _r = g_next.hipMemsetD32Async_fn(dst, value, count, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 304
static hipError_t hipMemsetD8Layer(hipDeviceptr_t dest, unsigned char value, size_t count) {
  auto* _rec = HipGetActiveRecordExt(304u);
  _rec->memory1 = reinterpret_cast<void*>(dest);
  _rec->size = count;
  auto _r = g_next.hipMemsetD8_fn(dest, value, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 305
static hipError_t hipMemsetD8AsyncLayer(hipDeviceptr_t dest, unsigned char value, size_t count,
                                         hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(305u);
  _rec->stream = stream;
  _rec->memory1 = reinterpret_cast<void*>(dest);
  _rec->size = count;
  auto _r = g_next.hipMemsetD8Async_fn(dest, value, count, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 306
static hipError_t hipMipmappedArrayCreateLayer(hipMipmappedArray_t* pHandle,
                                                HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                                unsigned int numMipmapLevels) {
  auto* _rec = HipGetActiveRecordExt(306u);
  auto _r = g_next.hipMipmappedArrayCreate_fn(pHandle, pMipmappedArrayDesc, numMipmapLevels);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 307
static hipError_t hipMipmappedArrayDestroyLayer(hipMipmappedArray_t hMipmappedArray) {
  auto* _rec = HipGetActiveRecordExt(307u);
  auto _r = g_next.hipMipmappedArrayDestroy_fn(hMipmappedArray);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 308
static hipError_t hipMipmappedArrayGetMemoryRequirementsLayer(hipArrayMemoryRequirements* memoryRequirements, hipMipmappedArray_t mipmap, hipDevice_t device) {
  auto* _rec = HipGetActiveRecordExt(308u);
  auto _r = g_next.hipMipmappedArrayGetMemoryRequirements_fn(memoryRequirements, mipmap, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 309
static hipError_t hipMipmappedArrayGetLevelLayer(hipArray_t* pLevelArray,
                                                  hipMipmappedArray_t hMipMappedArray,
                                                  unsigned int level) {
  auto* _rec = HipGetActiveRecordExt(309u);
  auto _r = g_next.hipMipmappedArrayGetLevel_fn(pLevelArray, hMipMappedArray, level);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 310
static hipError_t hipModuleGetFunctionLayer(hipFunction_t* function, hipModule_t module,
                                             const char* kname) {
  auto* _rec = HipGetActiveRecordExt(310u);
  auto _r = g_next.hipModuleGetFunction_fn(function, module, kname);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 311
static hipError_t hipModuleGetFunctionCountLayer(unsigned int* count, hipModule_t module) {
  auto* _rec = HipGetActiveRecordExt(311u);
  auto _r = g_next.hipModuleGetFunctionCount_fn(count, module);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 312
static hipError_t hipModuleGetGlobalLayer(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                           const char* name) {
  auto* _rec = HipGetActiveRecordExt(312u);
  auto _r = g_next.hipModuleGetGlobal_fn(dptr, bytes, hmod, name);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 313
static hipError_t hipModuleGetTexRefLayer(textureReference** texRef, hipModule_t hmod,
                                           const char* name) {
  auto* _rec = HipGetActiveRecordExt(313u);
  auto _r = g_next.hipModuleGetTexRef_fn(texRef, hmod, name);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 314
static hipError_t hipModuleLaunchCooperativeKernelLayer(hipFunction_t f,
                                                        unsigned int gridDimX,
                                                        unsigned int gridDimY,
                                                        unsigned int gridDimZ,
                                                        unsigned int blockDimX,
                                                        unsigned int blockDimY,
                                                        unsigned int blockDimZ,
                                                        unsigned int sharedMemBytes,
                                                        hipStream_t stream,
                                                        void** kernelParams) {
  auto* _rec = HipGetActiveRecordExt(314u);
  _rec->stream = stream;
  _rec->gpu.grid_x  = gridDimX;
  _rec->gpu.grid_y  = gridDimY;
  _rec->gpu.grid_z  = gridDimZ;
  _rec->gpu.block_x = blockDimX;
  _rec->gpu.block_y = blockDimY;
  _rec->gpu.block_z = blockDimZ;
  if (kernelParams) {
    HipCaptureKernelArgsExt(&_rec->gpu, f, kernelParams);
  }
  auto _r = g_next.hipModuleLaunchCooperativeKernel_fn(f,
                                                       gridDimX, gridDimY, gridDimZ,
                                                       blockDimX, blockDimY, blockDimZ,
                                                       sharedMemBytes, stream, kernelParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 315
static hipError_t hipModuleLaunchCooperativeKernelMultiDeviceLayer(hipFunctionLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(315u);
  auto _r = g_next.hipModuleLaunchCooperativeKernelMultiDevice_fn(launchParamsList, numDevices, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 316
static hipError_t hipModuleLaunchKernelLayer(hipFunction_t f, unsigned int gridDimX,
                                             unsigned int gridDimY, unsigned int gridDimZ,
                                             unsigned int blockDimX, unsigned int blockDimY,
                                             unsigned int blockDimZ,
                                             unsigned int sharedMemBytes,
                                             hipStream_t stream, void** kernelParams,
                                             void** extra) {
  auto* _rec = HipGetActiveRecordExt(316u);
  _rec->stream = stream;
  _rec->gpu.grid_x  = gridDimX;
  _rec->gpu.grid_y  = gridDimY;
  _rec->gpu.grid_z  = gridDimZ;
  _rec->gpu.block_x = blockDimX;
  _rec->gpu.block_y = blockDimY;
  _rec->gpu.block_z = blockDimZ;
  if (kernelParams) {
    HipCaptureKernelArgsExt(&_rec->gpu, f, kernelParams);
  } else {
    const void* kbuf;
    size_t ksz;
    if (ParseKernelExtra(extra, kbuf, ksz)) {
      HipCaptureKernelArgsPackedExt(&_rec->gpu, f, kbuf, ksz);
    }
  }
  auto _r = g_next.hipModuleLaunchKernel_fn(f,
                                            gridDimX, gridDimY, gridDimZ,
                                            blockDimX, blockDimY, blockDimZ,
                                            sharedMemBytes, stream, kernelParams, extra);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 317
static hipError_t hipModuleLoadFatBinaryLayer(hipModule_t* module, const void* fatbin) {
  auto* _rec = HipGetActiveRecordExt(317u);
  auto _r = g_next.hipModuleLoadFatBinary_fn(module, fatbin);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 318
static hipError_t hipModuleLoadLayer(hipModule_t* module, const char* fname) {
  auto* _rec = HipGetActiveRecordExt(318u);
  auto _r = g_next.hipModuleLoad_fn(module, fname);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 319
static hipError_t hipModuleLoadDataLayer(hipModule_t* module, const void* image) {
  auto* _rec = HipGetActiveRecordExt(319u);
  auto _r = g_next.hipModuleLoadData_fn(module, image);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 320
static hipError_t hipModuleLoadDataExLayer(hipModule_t* module, const void* image,
                                            unsigned int numOptions, hipJitOption* options,
                                            void** optionValues) {
  auto* _rec = HipGetActiveRecordExt(320u);
  auto _r = g_next.hipModuleLoadDataEx_fn(module, image, numOptions, options, optionValues);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 321
static hipError_t hipLinkAddDataLayer(hipLinkState_t state, hipJitInputType type, void* data,
                                       size_t size, const char* name, unsigned int numOptions,
                                       hipJitOption* options, void** optionValues) {
  auto* _rec = HipGetActiveRecordExt(321u);
  auto _r = g_next.hipLinkAddData_fn(state, type, data, size, name, numOptions, options, optionValues);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 322
static hipError_t hipLinkAddFileLayer(hipLinkState_t state, hipJitInputType type, const char* path,
                                       unsigned int numOptions, hipJitOption* options,
                                       void** optionValues) {
  auto* _rec = HipGetActiveRecordExt(322u);
  auto _r = g_next.hipLinkAddFile_fn(state, type, path, numOptions, options, optionValues);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 323
static hipError_t hipLinkCompleteLayer(hipLinkState_t state, void** hipBinOut, size_t* sizeOut) {
  auto* _rec = HipGetActiveRecordExt(323u);
  auto _r = g_next.hipLinkComplete_fn(state, hipBinOut, sizeOut);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 324
static hipError_t hipLinkCreateLayer(unsigned int numOptions, hipJitOption* options,
                                      void** optionValues, hipLinkState_t* stateOut) {
  auto* _rec = HipGetActiveRecordExt(324u);
  auto _r = g_next.hipLinkCreate_fn(numOptions, options, optionValues, stateOut);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 325
static hipError_t hipLinkDestroyLayer(hipLinkState_t state) {
  auto* _rec = HipGetActiveRecordExt(325u);
  auto _r = g_next.hipLinkDestroy_fn(state);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 326
static hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorLayer(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
  auto* _rec = HipGetActiveRecordExt(326u);
  auto _r = g_next.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn(numBlocks, f, blockSize, dynSharedMemPerBlk);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 327
static hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsLayer(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(327u);
  auto _r = g_next.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn(numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 328
static hipError_t hipModuleOccupancyMaxPotentialBlockSizeLayer(int* gridSize, int* blockSize,
                                                                hipFunction_t f,
                                                                size_t dynSharedMemPerBlk,
                                                                int blockSizeLimit) {
  auto* _rec = HipGetActiveRecordExt(328u);
  auto _r = g_next.hipModuleOccupancyMaxPotentialBlockSize_fn(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 329
static hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlagsLayer(int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit,
    unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(329u);
  auto _r = g_next.hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 330
static hipError_t hipModuleUnloadLayer(hipModule_t module) {
  auto* _rec = HipGetActiveRecordExt(330u);
  auto _r = g_next.hipModuleUnload_fn(module);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 331
static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorLayer(int* numBlocks, const void* f,
                                                                     int blockSize,
                                                                     size_t dynSharedMemPerBlk) {
  auto* _rec = HipGetActiveRecordExt(331u);
  auto _r = g_next.hipOccupancyMaxActiveBlocksPerMultiprocessor_fn(numBlocks, f, blockSize, dynSharedMemPerBlk);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 332
static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsLayer(int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(332u);
  auto _r = g_next.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn(numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 333
static hipError_t hipOccupancyMaxPotentialBlockSizeLayer(int* gridSize, int* blockSize,
                                                          const void* f, size_t dynSharedMemPerBlk,
                                                          int blockSizeLimit) {
  auto* _rec = HipGetActiveRecordExt(333u);
  auto _r = g_next.hipOccupancyMaxPotentialBlockSize_fn(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 334
static hipError_t hipOccupancyMaxActiveClustersLayer(int* numClusters, const void* f,
                                                      const hipLaunchConfig_t* launchConfig) {
  auto* _rec = HipGetActiveRecordExt(334u);
  auto _r = g_next.hipOccupancyMaxActiveClusters_fn(numClusters, f, launchConfig);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 335
static hipError_t hipOccupancyMaxPotentialClusterSizeLayer(int* clusterSize, const void* f,
                                                            const hipLaunchConfig_t* config) {
  auto* _rec = HipGetActiveRecordExt(335u);
  auto _r = g_next.hipOccupancyMaxPotentialClusterSize_fn(clusterSize, f, config);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 336
static hipError_t hipPeekAtLastErrorLayer(void) {
  auto* _rec = HipGetActiveRecordExt(336u);
  auto _r = g_next.hipPeekAtLastError_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 337
static hipError_t hipPointerGetAttributeLayer(void* data, hipPointer_attribute attribute,
                                               hipDeviceptr_t ptr) {
  auto* _rec = HipGetActiveRecordExt(337u);
  auto _r = g_next.hipPointerGetAttribute_fn(data, attribute, ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 338
static hipError_t hipPointerGetAttributesLayer(hipPointerAttribute_t* attributes, const void* ptr) {
  auto* _rec = HipGetActiveRecordExt(338u);
  auto _r = g_next.hipPointerGetAttributes_fn(attributes, ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 339
static hipError_t hipPointerSetAttributeLayer(const void* value, hipPointer_attribute attribute,
                                               hipDeviceptr_t ptr) {
  auto* _rec = HipGetActiveRecordExt(339u);
  auto _r = g_next.hipPointerSetAttribute_fn(value, attribute, ptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 340
static hipError_t hipProfilerStartLayer(void) {
  auto* _rec = HipGetActiveRecordExt(340u);
  auto _r = g_next.hipProfilerStart_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 341
static hipError_t hipProfilerStopLayer(void) {
  auto* _rec = HipGetActiveRecordExt(341u);
  auto _r = g_next.hipProfilerStop_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 342
static hipError_t hipRuntimeGetVersionLayer(int* runtimeVersion) {
  auto* _rec = HipGetActiveRecordExt(342u);
  auto _r = g_next.hipRuntimeGetVersion_fn(runtimeVersion);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 343
static hipError_t hipSetDeviceLayer(int deviceId) {
  auto* _rec = HipGetActiveRecordExt(343u);
  auto _r = g_next.hipSetDevice_fn(deviceId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 344
static hipError_t hipSetDeviceFlagsLayer(unsigned flags) {
  auto* _rec = HipGetActiveRecordExt(344u);
  auto _r = g_next.hipSetDeviceFlags_fn(flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 345
static hipError_t hipSetupArgumentLayer(const void* arg, size_t size, size_t offset) {
  auto* _rec = HipGetActiveRecordExt(345u);
  auto _r = g_next.hipSetupArgument_fn(arg, size, offset);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 346
static hipError_t hipSignalExternalSemaphoresAsyncLayer(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreSignalParams* paramsArray,
    unsigned int numExtSems, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(346u);
  _rec->stream = stream;
  auto _r = g_next.hipSignalExternalSemaphoresAsync_fn(extSemArray, paramsArray, numExtSems, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 347
static hipError_t hipStreamAddCallbackLayer(hipStream_t stream, hipStreamCallback_t callback,
                                             void* userData, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(347u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamAddCallback_fn(stream, callback, userData, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 348
static hipError_t hipStreamAttachMemAsyncLayer(hipStream_t stream, void* dev_ptr, size_t length,
                                                unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(348u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamAttachMemAsync_fn(stream, dev_ptr, length, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 349
static hipError_t hipStreamBeginCaptureLayer(hipStream_t stream, hipStreamCaptureMode mode) {
  auto* _rec = HipGetActiveRecordExt(349u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamBeginCapture_fn(stream, mode);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 350
static hipError_t hipStreamCopyAttributesLayer(hipStream_t dst, hipStream_t src) {
  auto* _rec = HipGetActiveRecordExt(350u);
  _rec->stream = dst;
  auto _r = g_next.hipStreamCopyAttributes_fn(dst, src);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 351
static hipError_t hipStreamCreateLayer(hipStream_t* stream) {
  auto* _rec = HipGetActiveRecordExt(351u);
  auto _r = g_next.hipStreamCreate_fn(stream);
  if (_r == hipSuccess && stream) _rec->memory1 = reinterpret_cast<void*>(*stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 352
static hipError_t hipStreamCreateWithFlagsLayer(hipStream_t* stream, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(352u);
  auto _r = g_next.hipStreamCreateWithFlags_fn(stream, flags);
  if (_r == hipSuccess && stream) _rec->memory1 = reinterpret_cast<void*>(*stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 353
static hipError_t hipStreamCreateWithPriorityLayer(hipStream_t* stream, unsigned int flags,
                                                    int priority) {
  auto* _rec = HipGetActiveRecordExt(353u);
  auto _r = g_next.hipStreamCreateWithPriority_fn(stream, flags, priority);
  if (_r == hipSuccess && stream) _rec->memory1 = reinterpret_cast<void*>(*stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 354
static hipError_t hipStreamDestroyLayer(hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(354u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamDestroy_fn(stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 355
static hipError_t hipStreamEndCaptureLayer(hipStream_t stream, hipGraph_t* pGraph) {
  auto* _rec = HipGetActiveRecordExt(355u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamEndCapture_fn(stream, pGraph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 356
static hipError_t hipStreamGetCaptureInfoLayer(hipStream_t stream,
                                                hipStreamCaptureStatus* pCaptureStatus,
                                                unsigned long long* pId) {
  auto* _rec = HipGetActiveRecordExt(356u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetCaptureInfo_fn(stream, pCaptureStatus, pId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 357
static hipError_t hipStreamGetCaptureInfo_v2Layer(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out,
    hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  auto* _rec = HipGetActiveRecordExt(357u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetCaptureInfo_v2_fn(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 358
static hipError_t hipStreamGetDeviceLayer(hipStream_t stream, hipDevice_t* device) {
  auto* _rec = HipGetActiveRecordExt(358u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetDevice_fn(stream, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 359
static hipError_t hipStreamGetFlagsLayer(hipStream_t stream, unsigned int* flags) {
  auto* _rec = HipGetActiveRecordExt(359u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetFlags_fn(stream, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 360
static hipError_t hipStreamGetIdLayer(hipStream_t stream, unsigned long long* streamId) {
  auto* _rec = HipGetActiveRecordExt(360u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetId_fn(stream, streamId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 361
static hipError_t hipStreamGetPriorityLayer(hipStream_t stream, int* priority) {
  auto* _rec = HipGetActiveRecordExt(361u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetPriority_fn(stream, priority);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 362
static hipError_t hipStreamIsCapturingLayer(hipStream_t stream,
                                             hipStreamCaptureStatus* pCaptureStatus) {
  auto* _rec = HipGetActiveRecordExt(362u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamIsCapturing_fn(stream, pCaptureStatus);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 363
static hipError_t hipStreamQueryLayer(hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(363u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamQuery_fn(stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 364
static hipError_t hipStreamSynchronizeLayer(hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(364u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamSynchronize_fn(stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 365
static hipError_t hipStreamUpdateCaptureDependenciesLayer(hipStream_t stream,
                                                           hipGraphNode_t* dependencies,
                                                           size_t numDependencies,
                                                           unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(365u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamUpdateCaptureDependencies_fn(stream, dependencies, numDependencies, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 366
static hipError_t hipStreamWaitEventLayer(hipStream_t stream, hipEvent_t event,
                                           unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(366u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamWaitEvent_fn(stream, event, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 367
static hipError_t hipStreamWaitValue32Layer(hipStream_t stream, void* ptr, uint32_t value,
                                             unsigned int flags, uint32_t mask) {
  auto* _rec = HipGetActiveRecordExt(367u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamWaitValue32_fn(stream, ptr, value, flags, mask);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 368
static hipError_t hipStreamWaitValue64Layer(hipStream_t stream, void* ptr, uint64_t value,
                                             unsigned int flags, uint64_t mask) {
  auto* _rec = HipGetActiveRecordExt(368u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamWaitValue64_fn(stream, ptr, value, flags, mask);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 369
static hipError_t hipStreamWriteValue32Layer(hipStream_t stream, void* ptr, uint32_t value,
                                              unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(369u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamWriteValue32_fn(stream, ptr, value, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 370
static hipError_t hipStreamWriteValue64Layer(hipStream_t stream, void* ptr, uint64_t value,
                                              unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(370u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamWriteValue64_fn(stream, ptr, value, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 371
static hipError_t hipStreamBatchMemOpLayer(hipStream_t stream, unsigned int count,
                                            hipStreamBatchMemOpParams* paramArray,
                                            unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(371u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamBatchMemOp_fn(stream, count, paramArray, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 372
static hipError_t hipTexObjectCreateLayer(hipTextureObject_t* pTexObject,
                                           const HIP_RESOURCE_DESC* pResDesc,
                                           const HIP_TEXTURE_DESC* pTexDesc,
                                           const HIP_RESOURCE_VIEW_DESC* pResViewDesc) {
  auto* _rec = HipGetActiveRecordExt(372u);
  auto _r = g_next.hipTexObjectCreate_fn(pTexObject, pResDesc, pTexDesc, pResViewDesc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 373
static hipError_t hipTexObjectDestroyLayer(hipTextureObject_t texObject) {
  auto* _rec = HipGetActiveRecordExt(373u);
  auto _r = g_next.hipTexObjectDestroy_fn(texObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 374
static hipError_t hipTexObjectGetResourceDescLayer(HIP_RESOURCE_DESC* pResDesc,
                                                    hipTextureObject_t texObject) {
  auto* _rec = HipGetActiveRecordExt(374u);
  auto _r = g_next.hipTexObjectGetResourceDesc_fn(pResDesc, texObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 375
static hipError_t hipTexObjectGetResourceViewDescLayer(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                                        hipTextureObject_t texObject) {
  auto* _rec = HipGetActiveRecordExt(375u);
  auto _r = g_next.hipTexObjectGetResourceViewDesc_fn(pResViewDesc, texObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 376
static hipError_t hipTexObjectGetTextureDescLayer(HIP_TEXTURE_DESC* pTexDesc,
                                                   hipTextureObject_t texObject) {
  auto* _rec = HipGetActiveRecordExt(376u);
  auto _r = g_next.hipTexObjectGetTextureDesc_fn(pTexDesc, texObject);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 377
static hipError_t hipTexRefGetAddressLayer(hipDeviceptr_t* dev_ptr,
                                            const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(377u);
  auto _r = g_next.hipTexRefGetAddress_fn(dev_ptr, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 378
static hipError_t hipTexRefGetAddressModeLayer(enum hipTextureAddressMode* pam,
                                                const textureReference* texRef, int dim) {
  auto* _rec = HipGetActiveRecordExt(378u);
  auto _r = g_next.hipTexRefGetAddressMode_fn(pam, texRef, dim);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 379
static hipError_t hipTexRefGetFilterModeLayer(enum hipTextureFilterMode* pfm,
                                               const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(379u);
  auto _r = g_next.hipTexRefGetFilterMode_fn(pfm, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 380
static hipError_t hipTexRefGetFlagsLayer(unsigned int* pFlags, const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(380u);
  auto _r = g_next.hipTexRefGetFlags_fn(pFlags, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 381
static hipError_t hipTexRefGetFormatLayer(hipArray_Format* pFormat, int* pNumChannels,
                                           const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(381u);
  auto _r = g_next.hipTexRefGetFormat_fn(pFormat, pNumChannels, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 382
static hipError_t hipTexRefGetMaxAnisotropyLayer(int* pmaxAnsio, const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(382u);
  auto _r = g_next.hipTexRefGetMaxAnisotropy_fn(pmaxAnsio, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 383
static hipError_t hipTexRefGetMipMappedArrayLayer(hipMipmappedArray_t* pArray,
                                                   const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(383u);
  auto _r = g_next.hipTexRefGetMipMappedArray_fn(pArray, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 384
static hipError_t hipTexRefGetMipmapFilterModeLayer(enum hipTextureFilterMode* pfm,
                                                     const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(384u);
  auto _r = g_next.hipTexRefGetMipmapFilterMode_fn(pfm, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 385
static hipError_t hipTexRefGetMipmapLevelBiasLayer(float* pbias, const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(385u);
  auto _r = g_next.hipTexRefGetMipmapLevelBias_fn(pbias, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 386
static hipError_t hipTexRefGetMipmapLevelClampLayer(float* pminMipmapLevelClamp,
                                                     float* pmaxMipmapLevelClamp,
                                                     const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(386u);
  auto _r = g_next.hipTexRefGetMipmapLevelClamp_fn(pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 387
static hipError_t hipTexRefSetAddressLayer(size_t* ByteOffset, textureReference* texRef,
                                            hipDeviceptr_t dptr, size_t bytes) {
  auto* _rec = HipGetActiveRecordExt(387u);
  auto _r = g_next.hipTexRefSetAddress_fn(ByteOffset, texRef, dptr, bytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 388
static hipError_t hipTexRefSetAddress2DLayer(textureReference* texRef,
                                              const HIP_ARRAY_DESCRIPTOR* desc, hipDeviceptr_t dptr,
                                              size_t Pitch) {
  auto* _rec = HipGetActiveRecordExt(388u);
  auto _r = g_next.hipTexRefSetAddress2D_fn(texRef, desc, dptr, Pitch);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 389
static hipError_t hipTexRefSetAddressModeLayer(textureReference* texRef, int dim,
                                                enum hipTextureAddressMode am) {
  auto* _rec = HipGetActiveRecordExt(389u);
  auto _r = g_next.hipTexRefSetAddressMode_fn(texRef, dim, am);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 390
static hipError_t hipTexRefSetArrayLayer(textureReference* tex, hipArray_const_t array,
                                          unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(390u);
  auto _r = g_next.hipTexRefSetArray_fn(tex, array, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 391
static hipError_t hipTexRefSetBorderColorLayer(textureReference* texRef, float* pBorderColor) {
  auto* _rec = HipGetActiveRecordExt(391u);
  auto _r = g_next.hipTexRefSetBorderColor_fn(texRef, pBorderColor);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 392
static hipError_t hipTexRefSetFilterModeLayer(textureReference* texRef,
                                               enum hipTextureFilterMode fm) {
  auto* _rec = HipGetActiveRecordExt(392u);
  auto _r = g_next.hipTexRefSetFilterMode_fn(texRef, fm);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 393
static hipError_t hipTexRefSetFlagsLayer(textureReference* texRef, unsigned int Flags) {
  auto* _rec = HipGetActiveRecordExt(393u);
  auto _r = g_next.hipTexRefSetFlags_fn(texRef, Flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 394
static hipError_t hipTexRefSetFormatLayer(textureReference* texRef, hipArray_Format fmt,
                                           int NumPackedComponents) {
  auto* _rec = HipGetActiveRecordExt(394u);
  auto _r = g_next.hipTexRefSetFormat_fn(texRef, fmt, NumPackedComponents);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 395
static hipError_t hipTexRefSetMaxAnisotropyLayer(textureReference* texRef, unsigned int maxAniso) {
  auto* _rec = HipGetActiveRecordExt(395u);
  auto _r = g_next.hipTexRefSetMaxAnisotropy_fn(texRef, maxAniso);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 396
static hipError_t hipTexRefSetMipmapFilterModeLayer(textureReference* texRef,
                                                     enum hipTextureFilterMode fm) {
  auto* _rec = HipGetActiveRecordExt(396u);
  auto _r = g_next.hipTexRefSetMipmapFilterMode_fn(texRef, fm);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 397
static hipError_t hipTexRefSetMipmapLevelBiasLayer(textureReference* texRef, float bias) {
  auto* _rec = HipGetActiveRecordExt(397u);
  auto _r = g_next.hipTexRefSetMipmapLevelBias_fn(texRef, bias);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 398
static hipError_t hipTexRefSetMipmapLevelClampLayer(textureReference* texRef,
                                                     float minMipMapLevelClamp,
                                                     float maxMipMapLevelClamp) {
  auto* _rec = HipGetActiveRecordExt(398u);
  auto _r = g_next.hipTexRefSetMipmapLevelClamp_fn(texRef, minMipMapLevelClamp, maxMipMapLevelClamp);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 399
static hipError_t hipTexRefSetMipmappedArrayLayer(textureReference* texRef,
                                                   struct hipMipmappedArray* mipmappedArray,
                                                   unsigned int Flags) {
  auto* _rec = HipGetActiveRecordExt(399u);
  auto _r = g_next.hipTexRefSetMipmappedArray_fn(texRef, mipmappedArray, Flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 400
static hipError_t hipThreadExchangeStreamCaptureModeLayer(hipStreamCaptureMode* mode) {
  auto* _rec = HipGetActiveRecordExt(400u);
  auto _r = g_next.hipThreadExchangeStreamCaptureMode_fn(mode);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 401
static hipError_t hipUnbindTextureLayer(const textureReference* tex) {
  auto* _rec = HipGetActiveRecordExt(401u);
  auto _r = g_next.hipUnbindTexture_fn(tex);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 402
static hipError_t hipUserObjectCreateLayer(hipUserObject_t* object_out, void* ptr,
                                            hipHostFn_t destroy, unsigned int initialRefcount,
                                            unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(402u);
  auto _r = g_next.hipUserObjectCreate_fn(object_out, ptr, destroy, initialRefcount, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 403
static hipError_t hipUserObjectReleaseLayer(hipUserObject_t object, unsigned int count) {
  auto* _rec = HipGetActiveRecordExt(403u);
  auto _r = g_next.hipUserObjectRelease_fn(object, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 404
static hipError_t hipUserObjectRetainLayer(hipUserObject_t object, unsigned int count) {
  auto* _rec = HipGetActiveRecordExt(404u);
  auto _r = g_next.hipUserObjectRetain_fn(object, count);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 405
static hipError_t hipWaitExternalSemaphoresAsyncLayer(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreWaitParams* paramsArray,
    unsigned int numExtSems, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(405u);
  _rec->stream = stream;
  auto _r = g_next.hipWaitExternalSemaphoresAsync_fn(extSemArray, paramsArray, numExtSems, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 406
static hipChannelFormatDesc hipCreateChannelDescLayer(int x, int y, int z, int w,
                                                       hipChannelFormatKind f) {
  auto* _rec = HipGetActiveRecordExt(406u);
  auto _r = g_next.hipCreateChannelDesc_fn(x, y, z, w, f);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 407
static hipError_t hipExtModuleLaunchKernelLayer(hipFunction_t f,
                                                uint32_t globalWorkSizeX,
                                                uint32_t globalWorkSizeY,
                                                uint32_t globalWorkSizeZ,
                                                uint32_t localWorkSizeX,
                                                uint32_t localWorkSizeY,
                                                uint32_t localWorkSizeZ,
                                                size_t sharedMemBytes,
                                                hipStream_t hStream,
                                                void** kernelParams,
                                                void** extra,
                                                hipEvent_t startEvent,
                                                hipEvent_t stopEvent,
                                                uint32_t flags) {
  auto* _rec = HipGetActiveRecordExt(407u);
  _rec->stream = hStream;
  // globalWorkSize = grid * localWorkSize (OpenCL convention)
  _rec->gpu.block_x = localWorkSizeX;
  _rec->gpu.block_y = localWorkSizeY;
  _rec->gpu.block_z = localWorkSizeZ;
  _rec->gpu.grid_x  = localWorkSizeX ? globalWorkSizeX / localWorkSizeX : 0;
  _rec->gpu.grid_y  = localWorkSizeY ? globalWorkSizeY / localWorkSizeY : 0;
  _rec->gpu.grid_z  = localWorkSizeZ ? globalWorkSizeZ / localWorkSizeZ : 0;
  if (kernelParams) {
    HipCaptureKernelArgsExt(&_rec->gpu, f, kernelParams);
  } else {
    const void* kbuf;
    size_t ksz;
    if (ParseKernelExtra(extra, kbuf, ksz)) {
      HipCaptureKernelArgsPackedExt(&_rec->gpu, f, kbuf, ksz);
    }
  }
  auto _r = g_next.hipExtModuleLaunchKernel_fn(f,
                                               globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                                               localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                               sharedMemBytes, hStream, kernelParams, extra,
                                               startEvent, stopEvent, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 408
static hipError_t hipHccModuleLaunchKernelLayer(hipFunction_t f,
                                                uint32_t globalWorkSizeX,
                                                uint32_t globalWorkSizeY,
                                                uint32_t globalWorkSizeZ,
                                                uint32_t localWorkSizeX,
                                                uint32_t localWorkSizeY,
                                                uint32_t localWorkSizeZ,
                                                size_t sharedMemBytes,
                                                hipStream_t hStream,
                                                void** kernelParams,
                                                void** extra,
                                                hipEvent_t startEvent,
                                                hipEvent_t stopEvent) {
  auto* _rec = HipGetActiveRecordExt(408u);
  _rec->stream = hStream;
  _rec->gpu.block_x = localWorkSizeX;
  _rec->gpu.block_y = localWorkSizeY;
  _rec->gpu.block_z = localWorkSizeZ;
  _rec->gpu.grid_x  = localWorkSizeX ? globalWorkSizeX / localWorkSizeX : 0;
  _rec->gpu.grid_y  = localWorkSizeY ? globalWorkSizeY / localWorkSizeY : 0;
  _rec->gpu.grid_z  = localWorkSizeZ ? globalWorkSizeZ / localWorkSizeZ : 0;
  if (kernelParams) {
    HipCaptureKernelArgsExt(&_rec->gpu, f, kernelParams);
  } else {
    const void* kbuf;
    size_t ksz;
    if (ParseKernelExtra(extra, kbuf, ksz)) {
      HipCaptureKernelArgsPackedExt(&_rec->gpu, f, kbuf, ksz);
    }
  }
  auto _r = g_next.hipHccModuleLaunchKernel_fn(f,
                                               globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                                               localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                               sharedMemBytes, hStream, kernelParams, extra,
                                               startEvent, stopEvent);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 409
static hipError_t hipMemcpy_sptLayer(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(409u);
  auto _r = g_next.hipMemcpy_spt_fn(dst, src, sizeBytes, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 410
static hipError_t hipMemcpyToSymbol_sptLayer(const void* symbol, const void* src, size_t sizeBytes,
                                              size_t offset, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(410u);
  auto _r = g_next.hipMemcpyToSymbol_spt_fn(symbol, src, sizeBytes, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 411
static hipError_t hipMemcpyFromSymbol_sptLayer(void* dst, const void* symbol, size_t sizeBytes,
                                                size_t offset, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(411u);
  auto _r = g_next.hipMemcpyFromSymbol_spt_fn(dst, symbol, sizeBytes, offset, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 412
static hipError_t hipMemcpy2D_sptLayer(void* dst, size_t dpitch, const void* src, size_t spitch,
                                        size_t width, size_t height, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(412u);
  auto _r = g_next.hipMemcpy2D_spt_fn(dst, dpitch, src, spitch, width, height, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 413
static hipError_t hipMemcpy2DFromArray_sptLayer(void* dst, size_t dpitch, hipArray_const_t src,
                                                 size_t wOffset, size_t hOffset, size_t width,
                                                 size_t height, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(413u);
  auto _r = g_next.hipMemcpy2DFromArray_spt_fn(dst, dpitch, src, wOffset, hOffset, width, height, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 414
static hipError_t hipMemcpy3D_sptLayer(const struct hipMemcpy3DParms* p) {
  auto* _rec = HipGetActiveRecordExt(414u);
  auto _r = g_next.hipMemcpy3D_spt_fn(p);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 415
static hipError_t hipMemset_sptLayer(void* dst, int value, size_t sizeBytes) {
  auto* _rec = HipGetActiveRecordExt(415u);
  auto _r = g_next.hipMemset_spt_fn(dst, value, sizeBytes);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 416
static hipError_t hipMemsetAsync_sptLayer(void* dst, int value, size_t sizeBytes,
                                           hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(416u);
  _rec->stream = stream;
  auto _r = g_next.hipMemsetAsync_spt_fn(dst, value, sizeBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 417
static hipError_t hipMemset2D_sptLayer(void* dst, size_t pitch, int value, size_t width,
                                        size_t height) {
  auto* _rec = HipGetActiveRecordExt(417u);
  auto _r = g_next.hipMemset2D_spt_fn(dst, pitch, value, width, height);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 418
static hipError_t hipMemset2DAsync_sptLayer(void* dst, size_t pitch, int value, size_t width,
                                             size_t height, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(418u);
  _rec->stream = stream;
  auto _r = g_next.hipMemset2DAsync_spt_fn(dst, pitch, value, width, height, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 419
static hipError_t hipMemset3DAsync_sptLayer(hipPitchedPtr pitchedDevPtr, int value,
                                             hipExtent extent, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(419u);
  _rec->stream = stream;
  auto _r = g_next.hipMemset3DAsync_spt_fn(pitchedDevPtr, value, extent, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 420
static hipError_t hipMemset3D_sptLayer(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  auto* _rec = HipGetActiveRecordExt(420u);
  auto _r = g_next.hipMemset3D_spt_fn(pitchedDevPtr, value, extent);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 421
static hipError_t hipMemcpyAsync_sptLayer(void* dst, const void* src, size_t sizeBytes,
                                           hipMemcpyKind kind, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(421u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyAsync_spt_fn(dst, src, sizeBytes, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 422
static hipError_t hipMemcpy3DAsync_sptLayer(const hipMemcpy3DParms* p, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(422u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy3DAsync_spt_fn(p, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 423
static hipError_t hipMemcpy2DAsync_sptLayer(void* dst, size_t dpitch, const void* src,
                                             size_t spitch, size_t width, size_t height,
                                             hipMemcpyKind kind, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(423u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy2DAsync_spt_fn(dst, dpitch, src, spitch, width, height, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 424
static hipError_t hipMemcpyFromSymbolAsync_sptLayer(void* dst, const void* symbol,
                                                     size_t sizeBytes, size_t offset,
                                                     hipMemcpyKind kind, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(424u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyFromSymbolAsync_spt_fn(dst, symbol, sizeBytes, offset, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 425
static hipError_t hipMemcpyToSymbolAsync_sptLayer(const void* symbol, const void* src,
                                                   size_t sizeBytes, size_t offset,
                                                   hipMemcpyKind kind, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(425u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyToSymbolAsync_spt_fn(symbol, src, sizeBytes, offset, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 426
static hipError_t hipMemcpyFromArray_sptLayer(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                               size_t hOffset, size_t count, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(426u);
  auto _r = g_next.hipMemcpyFromArray_spt_fn(dst, src, wOffsetSrc, hOffset, count, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 427
static hipError_t hipMemcpy2DToArray_sptLayer(hipArray_t dst, size_t wOffset, size_t hOffset,
                                               const void* src, size_t spitch, size_t width,
                                               size_t height, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(427u);
  auto _r = g_next.hipMemcpy2DToArray_spt_fn(dst, wOffset, hOffset, src, spitch, width, height, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 428
static hipError_t hipMemcpy2DFromArrayAsync_sptLayer(void* dst, size_t dpitch,
                                                      hipArray_const_t src, size_t wOffsetSrc,
                                                      size_t hOffsetSrc, size_t width,
                                                      size_t height, hipMemcpyKind kind,
                                                      hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(428u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy2DFromArrayAsync_spt_fn(dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 429
static hipError_t hipMemcpy2DToArrayAsync_sptLayer(hipArray_t dst, size_t wOffset, size_t hOffset,
                                                    const void* src, size_t spitch, size_t width,
                                                    size_t height, hipMemcpyKind kind,
                                                    hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(429u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy2DToArrayAsync_spt_fn(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 430
static hipError_t hipStreamQuery_sptLayer(hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(430u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamQuery_spt_fn(stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 431
static hipError_t hipStreamSynchronize_sptLayer(hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(431u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamSynchronize_spt_fn(stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 432
static hipError_t hipStreamGetPriority_sptLayer(hipStream_t stream, int* priority) {
  auto* _rec = HipGetActiveRecordExt(432u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetPriority_spt_fn(stream, priority);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 433
static hipError_t hipStreamWaitEvent_sptLayer(hipStream_t stream, hipEvent_t event,
                                               unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(433u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamWaitEvent_spt_fn(stream, event, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 434
static hipError_t hipStreamGetFlags_sptLayer(hipStream_t stream, unsigned int* flags) {
  auto* _rec = HipGetActiveRecordExt(434u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetFlags_spt_fn(stream, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 435
static hipError_t hipStreamAddCallback_sptLayer(hipStream_t stream, hipStreamCallback_t callback,
                                                 void* userData, unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(435u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamAddCallback_spt_fn(stream, callback, userData, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 436
static hipError_t hipEventRecord_sptLayer(hipEvent_t event, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(436u);
  _rec->stream = stream;
  auto _r = g_next.hipEventRecord_spt_fn(event, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 437
static hipError_t hipLaunchCooperativeKernel_sptLayer(const void* f, dim3 gridDim, dim3 blockDim,
                                                       void** kernelParams, uint32_t sharedMemBytes,
                                                       hipStream_t hStream) {
  auto* _rec = HipGetActiveRecordExt(437u);
  _rec->stream = hStream;
  auto _r = g_next.hipLaunchCooperativeKernel_spt_fn(f, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 438
static hipError_t hipLaunchKernel_sptLayer(const void* function_address, dim3 numBlocks,
                                            dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                            hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(438u);
  _rec->stream = stream;
  auto _r = g_next.hipLaunchKernel_spt_fn(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 439
static hipError_t hipGraphLaunch_sptLayer(hipGraphExec_t graphExec, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(439u);
  _rec->stream  = stream;
  _rec->memory1 = reinterpret_cast<void*>(graphExec);
  auto _r = g_next.hipGraphLaunch_spt_fn(graphExec, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 440
static hipError_t hipStreamBeginCapture_sptLayer(hipStream_t stream, hipStreamCaptureMode mode) {
  auto* _rec = HipGetActiveRecordExt(440u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamBeginCapture_spt_fn(stream, mode);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 441
static hipError_t hipStreamEndCapture_sptLayer(hipStream_t stream, hipGraph_t* pGraph) {
  auto* _rec = HipGetActiveRecordExt(441u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamEndCapture_spt_fn(stream, pGraph);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 442
static hipError_t hipStreamIsCapturing_sptLayer(hipStream_t stream,
                                                 hipStreamCaptureStatus* pCaptureStatus) {
  auto* _rec = HipGetActiveRecordExt(442u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamIsCapturing_spt_fn(stream, pCaptureStatus);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 443
static hipError_t hipStreamGetCaptureInfo_sptLayer(hipStream_t stream,
                                                    hipStreamCaptureStatus* pCaptureStatus,
                                                    unsigned long long* pId) {
  auto* _rec = HipGetActiveRecordExt(443u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetCaptureInfo_spt_fn(stream, pCaptureStatus, pId);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 444
static hipError_t hipStreamGetCaptureInfo_v2_sptLayer(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out,
    hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  auto* _rec = HipGetActiveRecordExt(444u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetCaptureInfo_v2_spt_fn(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 445
static hipError_t hipLaunchHostFunc_sptLayer(hipStream_t stream, hipHostFn_t fn, void* userData) {
  auto* _rec = HipGetActiveRecordExt(445u);
  _rec->stream = stream;
  auto _r = g_next.hipLaunchHostFunc_spt_fn(stream, fn, userData);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 446
static int hipGetStreamDeviceIdLayer(hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(446u);
  _rec->stream = stream;
  auto _r = g_next.hipGetStreamDeviceId_fn(stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 447
static hipError_t hipDrvGraphAddMemsetNodeLayer(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                                 const hipGraphNode_t* dependencies,
                                                 size_t numDependencies,
                                                 const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  auto* _rec = HipGetActiveRecordExt(447u);
  auto _r = g_next.hipDrvGraphAddMemsetNode_fn(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 448
static hipError_t hipGetDevicePropertiesR0000Layer(hipDeviceProp_tR0000* prop, int device) {
  auto* _rec = HipGetActiveRecordExt(448u);
  auto _r = g_next.hipGetDevicePropertiesR0000_fn(prop, device);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 449
static hipError_t hipGetDriverEntryPointLayer(const char* symbol, void** funcPtr,
                                               unsigned long long flags,
                                               hipDriverEntryPointQueryResult* status) {
  auto* _rec = HipGetActiveRecordExt(449u);
  auto _r = g_next.hipGetDriverEntryPoint_fn(symbol, funcPtr, flags, status);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 450
static hipError_t hipGetDriverEntryPoint_sptLayer(const char* symbol, void** funcPtr,
                                                   unsigned long long flags,
                                                   hipDriverEntryPointQueryResult* status) {
  auto* _rec = HipGetActiveRecordExt(450u);
  auto _r = g_next.hipGetDriverEntryPoint_spt_fn(symbol, funcPtr, flags, status);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 451
static hipError_t hipExtGetLastErrorLayer(void) {
  auto* _rec = HipGetActiveRecordExt(451u);
  auto _r = g_next.hipExtGetLastError_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 452
static hipError_t hipTexRefGetBorderColorLayer(float* pBorderColor,
                                                const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(452u);
  auto _r = g_next.hipTexRefGetBorderColor_fn(pBorderColor, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 453
static hipError_t hipTexRefGetArrayLayer(hipArray_t* pArray, const textureReference* texRef) {
  auto* _rec = HipGetActiveRecordExt(453u);
  auto _r = g_next.hipTexRefGetArray_fn(pArray, texRef);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 454
static hipError_t hipGetProcAddressLayer(const char* symbol, void** pfn, int hipVersion,
                                          uint64_t flags,
                                          hipDriverProcAddressQueryResult* symbolStatus) {
  auto* _rec = HipGetActiveRecordExt(454u);
  auto _r = g_next.hipGetProcAddress_fn(symbol, pfn, hipVersion, flags, symbolStatus);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 455
static hipError_t hipGetProcAddress_sptLayer(const char* symbol, void** pfn, int hipVersion, uint64_t flags,
                                              hipDriverProcAddressQueryResult* symbolStatus) {
  auto* _rec = HipGetActiveRecordExt(455u);
  auto _r = g_next.hipGetProcAddress_spt_fn(symbol, pfn, hipVersion, flags, symbolStatus);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 456
static hipError_t hipStreamBeginCaptureToGraphLayer(hipStream_t stream, hipGraph_t graph,
                                                     const hipGraphNode_t* dependencies,
                                                     const hipGraphEdgeData* dependencyData,
                                                     size_t numDependencies,
                                                     hipStreamCaptureMode mode) {
  auto* _rec = HipGetActiveRecordExt(456u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamBeginCaptureToGraph_fn(stream, graph, dependencies, dependencyData, numDependencies, mode);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 457
static hipError_t hipGetFuncBySymbolLayer(hipFunction_t* functionPtr, const void* symbolPtr) {
  auto* _rec = HipGetActiveRecordExt(457u);
  auto _r = g_next.hipGetFuncBySymbol_fn(functionPtr, symbolPtr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 458
static hipError_t hipSetValidDevicesLayer(int* device_arr, int len) {
  auto* _rec = HipGetActiveRecordExt(458u);
  auto _r = g_next.hipSetValidDevices_fn(device_arr, len);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 459
static hipError_t hipMemcpyAtoDLayer(hipDeviceptr_t dstDevice, hipArray_t srcArray,
                                      size_t srcOffset, size_t ByteCount) {
  auto* _rec = HipGetActiveRecordExt(459u);
  auto _r = g_next.hipMemcpyAtoD_fn(dstDevice, srcArray, srcOffset, ByteCount);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 460
static hipError_t hipMemcpyDtoALayer(hipArray_t dstArray, size_t dstOffset,
                                      hipDeviceptr_t srcDevice, size_t ByteCount) {
  auto* _rec = HipGetActiveRecordExt(460u);
  auto _r = g_next.hipMemcpyDtoA_fn(dstArray, dstOffset, srcDevice, ByteCount);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 461
static hipError_t hipMemcpyAtoALayer(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray,
                                      size_t srcOffset, size_t ByteCount) {
  auto* _rec = HipGetActiveRecordExt(461u);
  auto _r = g_next.hipMemcpyAtoA_fn(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 462
static hipError_t hipMemcpyAtoHAsyncLayer(void* dstHost, hipArray_t srcArray, size_t srcOffset,
                                           size_t ByteCount, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(462u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyAtoHAsync_fn(dstHost, srcArray, srcOffset, ByteCount, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 463
static hipError_t hipMemcpyHtoAAsyncLayer(hipArray_t dstArray, size_t dstOffset,
                                           const void* srcHost, size_t ByteCount,
                                           hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(463u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyHtoAAsync_fn(dstArray, dstOffset, srcHost, ByteCount, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 464
static hipError_t hipMemcpy2DArrayToArrayLayer(hipArray_t dst, size_t wOffsetDst,
                                                size_t hOffsetDst, hipArray_const_t src,
                                                size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                                size_t height, hipMemcpyKind kind) {
  auto* _rec = HipGetActiveRecordExt(464u);
  auto _r = g_next.hipMemcpy2DArrayToArray_fn(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 465
static hipError_t hipDrvGraphAddMemFreeNodeLayer(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                                  const hipGraphNode_t* dependencies,
                                                  size_t numDependencies, hipDeviceptr_t dptr) {
  auto* _rec = HipGetActiveRecordExt(465u);
  auto _r = g_next.hipDrvGraphAddMemFreeNode_fn(phGraphNode, hGraph, dependencies, numDependencies, dptr);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 466
static hipError_t hipDrvGraphExecMemcpyNodeSetParamsLayer(hipGraphExec_t hGraphExec,
                                                           hipGraphNode_t hNode,
                                                           const HIP_MEMCPY3D* copyParams,
                                                           hipCtx_t ctx) {
  auto* _rec = HipGetActiveRecordExt(466u);
  auto _r = g_next.hipDrvGraphExecMemcpyNodeSetParams_fn(hGraphExec, hNode, copyParams, ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 467
static hipError_t hipDrvGraphExecMemsetNodeSetParamsLayer(hipGraphExec_t hGraphExec,
                                                           hipGraphNode_t hNode,
                                                           const hipMemsetParams* memsetParams,
                                                           hipCtx_t ctx) {
  auto* _rec = HipGetActiveRecordExt(467u);
  auto _r = g_next.hipDrvGraphExecMemsetNodeSetParams_fn(hGraphExec, hNode, memsetParams, ctx);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 468
static hipError_t hipGraphExecGetFlagsLayer(hipGraphExec_t graphExec, unsigned long long* flags) {
  auto* _rec = HipGetActiveRecordExt(468u);
  auto _r = g_next.hipGraphExecGetFlags_fn(graphExec, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 469
static hipError_t hipGraphNodeSetParamsLayer(hipGraphNode_t node, hipGraphNodeParams* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(469u);
  auto _r = g_next.hipGraphNodeSetParams_fn(node, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 470
static hipError_t hipGraphExecNodeSetParamsLayer(hipGraphExec_t graphExec, hipGraphNode_t node,
                                                  hipGraphNodeParams* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(470u);
  auto _r = g_next.hipGraphExecNodeSetParams_fn(graphExec, node, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 471
static hipError_t hipExternalMemoryGetMappedMipmappedArrayLayer(hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem,
    const hipExternalMemoryMipmappedArrayDesc* mipmapDesc) {
  auto* _rec = HipGetActiveRecordExt(471u);
  auto _r = g_next.hipExternalMemoryGetMappedMipmappedArray_fn(mipmap, extMem, mipmapDesc);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 472
static hipError_t hipDrvGraphMemcpyNodeGetParamsLayer(hipGraphNode_t hNode,
                                                       HIP_MEMCPY3D* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(472u);
  auto _r = g_next.hipDrvGraphMemcpyNodeGetParams_fn(hNode, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 473
static hipError_t hipDrvGraphMemcpyNodeSetParamsLayer(hipGraphNode_t hNode,
                                                       const HIP_MEMCPY3D* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(473u);
  auto _r = g_next.hipDrvGraphMemcpyNodeSetParams_fn(hNode, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 474
static hipError_t hipGraphAddBatchMemOpNodeLayer(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                                  const hipGraphNode_t* dependencies,
                                                  size_t numDependencies,
                                                  const hipBatchMemOpNodeParams* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(474u);
  auto _r = g_next.hipGraphAddBatchMemOpNode_fn(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 475
static hipError_t hipGraphBatchMemOpNodeGetParamsLayer(hipGraphNode_t hNode,
                                                        hipBatchMemOpNodeParams* nodeParams_out) {
  auto* _rec = HipGetActiveRecordExt(475u);
  auto _r = g_next.hipGraphBatchMemOpNodeGetParams_fn(hNode, nodeParams_out);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 476
static hipError_t hipGraphBatchMemOpNodeSetParamsLayer(hipGraphNode_t hNode,
                                                        hipBatchMemOpNodeParams* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(476u);
  auto _r = g_next.hipGraphBatchMemOpNodeSetParams_fn(hNode, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 477
static hipError_t hipGraphExecBatchMemOpNodeSetParamsLayer(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipBatchMemOpNodeParams* nodeParams) {
  auto* _rec = HipGetActiveRecordExt(477u);
  auto _r = g_next.hipGraphExecBatchMemOpNodeSetParams_fn(hGraphExec, hNode, nodeParams);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 478
static hipError_t hipEventRecordWithFlagsLayer(hipEvent_t event, hipStream_t stream,
                                                unsigned int flags) {
  auto* _rec = HipGetActiveRecordExt(478u);
  _rec->stream = stream;
  auto _r = g_next.hipEventRecordWithFlags_fn(event, stream, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 479
static hipError_t hipLaunchKernelExCLayer(const hipLaunchConfig_t* config, const void* fPtr,
                                           void** args) {
  auto* _rec = HipGetActiveRecordExt(479u);
  if (config) {
    _rec->stream = config->stream;
    _rec->gpu.grid_x = config->gridDim.x; _rec->gpu.grid_y = config->gridDim.y; _rec->gpu.grid_z = config->gridDim.z;
    _rec->gpu.block_x = config->blockDim.x; _rec->gpu.block_y = config->blockDim.y; _rec->gpu.block_z = config->blockDim.z;
  }
  auto _r = g_next.hipLaunchKernelExC_fn(config, fPtr, args);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 480
static hipError_t hipDrvLaunchKernelExLayer(const HIP_LAUNCH_CONFIG* config, hipFunction_t f,
                                             void** params, void** extra) {
  auto* _rec = HipGetActiveRecordExt(480u);
  if (config) {
    _rec->stream = config->hStream;
    _rec->gpu.grid_x = config->gridDimX; _rec->gpu.grid_y = config->gridDimY; _rec->gpu.grid_z = config->gridDimZ;
    _rec->gpu.block_x = config->blockDimX; _rec->gpu.block_y = config->blockDimY; _rec->gpu.block_z = config->blockDimZ;
  }
  auto _r = g_next.hipDrvLaunchKernelEx_fn(config, f, params, extra);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 481
static hipError_t hipMemGetHandleForAddressRangeLayer(void* handle, hipDeviceptr_t dptr,
                                                       size_t size,
                                                       hipMemRangeHandleType handleType,
                                                       unsigned long long flags) {
  auto* _rec = HipGetActiveRecordExt(481u);
  auto _r = g_next.hipMemGetHandleForAddressRange_fn(handle, dptr, size, handleType, flags);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 482
static hipError_t hipMemsetD2D8Layer(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
                                      size_t width, size_t height) {
  auto* _rec = HipGetActiveRecordExt(482u);
  auto _r = g_next.hipMemsetD2D8_fn(dst, dstPitch, value, width, height);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 483
static hipError_t hipMemsetD2D8AsyncLayer(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
                                           size_t width, size_t height, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(483u);
  _rec->stream = stream;
  auto _r = g_next.hipMemsetD2D8Async_fn(dst, dstPitch, value, width, height, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 484
static hipError_t hipMemsetD2D16Layer(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
                                       size_t width, size_t height) {
  auto* _rec = HipGetActiveRecordExt(484u);
  auto _r = g_next.hipMemsetD2D16_fn(dst, dstPitch, value, width, height);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 485
static hipError_t hipMemsetD2D16AsyncLayer(hipDeviceptr_t dst, size_t dstPitch,
                                            unsigned short value, size_t width, size_t height,
                                            hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(485u);
  _rec->stream = stream;
  auto _r = g_next.hipMemsetD2D16Async_fn(dst, dstPitch, value, width, height, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 486
static hipError_t hipMemsetD2D32Layer(hipDeviceptr_t dst, size_t dstPitch, unsigned int value,
                                       size_t width, size_t height) {
  auto* _rec = HipGetActiveRecordExt(486u);
  auto _r = g_next.hipMemsetD2D32_fn(dst, dstPitch, value, width, height);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 487
static hipError_t hipMemsetD2D32AsyncLayer(hipDeviceptr_t dst, size_t dstPitch, unsigned int value,
                                            size_t width, size_t height, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(487u);
  _rec->stream = stream;
  auto _r = g_next.hipMemsetD2D32Async_fn(dst, dstPitch, value, width, height, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 488
static hipError_t hipStreamGetAttributeLayer(hipStream_t stream, hipStreamAttrID attr,
                                              hipStreamAttrValue* value_out) {
  auto* _rec = HipGetActiveRecordExt(488u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamGetAttribute_fn(stream, attr, value_out);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 489
static hipError_t hipStreamSetAttributeLayer(hipStream_t stream, hipStreamAttrID attr,
                                              const hipStreamAttrValue* value) {
  auto* _rec = HipGetActiveRecordExt(489u);
  _rec->stream = stream;
  auto _r = g_next.hipStreamSetAttribute_fn(stream, attr, value);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 490
static hipError_t hipMemcpyBatchAsyncLayer(void** dsts, void** srcs, size_t* sizes, size_t count,
                                            hipMemcpyAttributes* attrs, size_t* attrsIdxs,
                                            size_t numAttrs, size_t* failIdx, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(490u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpyBatchAsync_fn(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 491
static hipError_t hipMemcpy3DBatchAsyncLayer(size_t numOps, struct hipMemcpy3DBatchOp* opList,
                                              size_t* failIdx, unsigned long long flags,
                                              hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(491u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy3DBatchAsync_fn(numOps, opList, failIdx, flags, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 492
static hipError_t hipMemcpy3DPeerLayer(hipMemcpy3DPeerParms* p) {
  auto* _rec = HipGetActiveRecordExt(492u);
  auto _r = g_next.hipMemcpy3DPeer_fn(p);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 493
static hipError_t hipMemcpy3DPeerAsyncLayer(hipMemcpy3DPeerParms* p, hipStream_t stream) {
  auto* _rec = HipGetActiveRecordExt(493u);
  _rec->stream = stream;
  auto _r = g_next.hipMemcpy3DPeerAsync_fn(p, stream);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 494
static hipError_t hipLibraryLoadDataLayer(hipLibrary_t* library, const void* code,
                                           hipJitOption* jitOptions, void** jitOptionsValues,
                                           unsigned int numJitOptions,
                                           hipLibraryOption* libraryOptions,
                                           void** libraryOptionValues,
                                           unsigned int numLibraryOptions) {
  auto* _rec = HipGetActiveRecordExt(494u);
  auto _r = g_next.hipLibraryLoadData_fn(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 495
static hipError_t hipLibraryLoadFromFileLayer(hipLibrary_t* library, const char* fileName,
                                               hipJitOption* jitOptions, void** jitOptionsValues,
                                               unsigned int numJitOptions,
                                               hipLibraryOption* libraryOptions,
                                               void** libraryOptionValues,
                                               unsigned int numLibraryOptions) {
  auto* _rec = HipGetActiveRecordExt(495u);
  auto _r = g_next.hipLibraryLoadFromFile_fn(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 496
static hipError_t hipLibraryUnloadLayer(hipLibrary_t library) {
  auto* _rec = HipGetActiveRecordExt(496u);
  auto _r = g_next.hipLibraryUnload_fn(library);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 497
static hipError_t hipLibraryGetKernelLayer(hipKernel_t* pKernel, hipLibrary_t library,
                                            const char* name) {
  auto* _rec = HipGetActiveRecordExt(497u);
  auto _r = g_next.hipLibraryGetKernel_fn(pKernel, library, name);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 498
static hipError_t hipLibraryGetKernelCountLayer(unsigned int *count,
                                                 hipLibrary_t library) {
  auto* _rec = HipGetActiveRecordExt(498u);
  auto _r = g_next.hipLibraryGetKernelCount_fn(count, library);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 499
static hipError_t hipLibraryEnumerateKernelsLayer(hipKernel_t* kernels, unsigned int numKernels,
                                                   hipLibrary_t library) {
  auto* _rec = HipGetActiveRecordExt(499u);
  auto _r = g_next.hipLibraryEnumerateKernels_fn(kernels, numKernels, library);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 500
static hipError_t hipKernelGetLibraryLayer(hipLibrary_t* library, hipKernel_t kernel) {
  auto* _rec = HipGetActiveRecordExt(500u);
  auto _r = g_next.hipKernelGetLibrary_fn(library, kernel);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 501
static hipError_t hipKernelGetNameLayer(const char** name, hipKernel_t kernel) {
  auto* _rec = HipGetActiveRecordExt(501u);
  auto _r = g_next.hipKernelGetName_fn(name, kernel);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 502
static hipError_t hipOccupancyAvailableDynamicSMemPerBlockLayer(size_t* dynamicSmemSize, const void* f,
                                                                 int numBlocks, int blockSize) {
  auto* _rec = HipGetActiveRecordExt(502u);
  auto _r = g_next.hipOccupancyAvailableDynamicSMemPerBlock_fn(dynamicSmemSize, f, numBlocks, blockSize);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 503
static hipError_t hipKernelGetParamInfoLayer(hipKernel_t kernel, size_t paramIndex,
                                              size_t* paramOffset, size_t* paramSize) {
  auto* _rec = HipGetActiveRecordExt(503u);
  auto _r = g_next.hipKernelGetParamInfo_fn(kernel, paramIndex, paramOffset, paramSize);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 504
static hipError_t hipExtDisableLoggingLayer(void) {
  auto* _rec = HipGetActiveRecordExt(504u);
  auto _r = g_next.hipExtDisableLogging_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 505
static hipError_t hipExtEnableLoggingLayer(void) {
  auto* _rec = HipGetActiveRecordExt(505u);
  auto _r = g_next.hipExtEnableLogging_fn();
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 506
static hipError_t hipExtSetLoggingParamsLayer(size_t log_level, size_t log_size, size_t log_mask) {
  auto* _rec = HipGetActiveRecordExt(506u);
  auto _r = g_next.hipExtSetLoggingParams_fn(log_level, log_size, log_mask);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 507
static hipError_t hipMemSetMemPoolLayer(hipMemLocation* location, hipMemAllocationType type,
                                         hipMemPool_t pool) {
  auto* _rec = HipGetActiveRecordExt(507u);
  auto _r = g_next.hipMemSetMemPool_fn(location, type, pool);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 508
static hipError_t hipMemGetMemPoolLayer(hipMemPool_t* pool, hipMemLocation* location,
                                         hipMemAllocationType type) {
  auto* _rec = HipGetActiveRecordExt(508u);
  auto _r = g_next.hipMemGetMemPool_fn(pool, location, type);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 509
static hipError_t hipKernelGetAttributeLayer(int* pi, hipFunction_attribute attrib, hipKernel_t kernel,
                                              hipDevice_t dev) {
  auto* _rec = HipGetActiveRecordExt(509u);
  auto _r = g_next.hipKernelGetAttribute_fn(pi, attrib, kernel, dev);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 510
static hipError_t hipKernelSetAttributeLayer(hipFunction_attribute attrib,
                                         int value, hipKernel_t kernel, hipDevice_t dev) {
  auto* _rec = HipGetActiveRecordExt(510u);
  auto _r = g_next.hipKernelSetAttribute_fn(attrib, value, kernel, dev);
  _rec->end_ns = NowNs();
  return _r;
}

// api_id = 511
static hipError_t hipKernelGetFunctionLayer(hipFunction_t* pFunc, hipKernel_t kernel) {
  auto* _rec = HipGetActiveRecordExt(511u);
  auto _r = g_next.hipKernelGetFunction_fn(pFunc, kernel);
  _rec->end_ns = NowNs();
  return _r;
}

// API name table — indexed by api_id (same order as UpdateDispatchTable).
const char* const kHipApiNamesExt[] = {
  "hipApiName",
  "hipArray3DCreate",
  "hipArray3DGetDescriptor",
  "hipArrayCreate",
  "hipArrayDestroy",
  "hipArrayGetDescriptor",
  "hipArrayGetInfo",
  "hipBindTexture",
  "hipBindTexture2D",
  "hipBindTextureToArray",
  "hipBindTextureToMipmappedArray",
  "hipChooseDevice",
  "hipChooseDeviceR0000",
  "hipConfigureCall",
  "hipCreateSurfaceObject",
  "hipCreateTextureObject",
  "hipCtxCreate",
  "hipCtxDestroy",
  "hipCtxDisablePeerAccess",
  "hipCtxEnablePeerAccess",
  "hipCtxGetApiVersion",
  "hipCtxGetCacheConfig",
  "hipCtxGetCurrent",
  "hipCtxGetDevice",
  "hipCtxGetFlags",
  "hipCtxGetSharedMemConfig",
  "hipCtxPopCurrent",
  "hipCtxPushCurrent",
  "hipCtxSetCacheConfig",
  "hipCtxSetCurrent",
  "hipCtxSetSharedMemConfig",
  "hipCtxSynchronize",
  "hipDestroyExternalMemory",
  "hipDestroyExternalSemaphore",
  "hipDestroySurfaceObject",
  "hipDestroyTextureObject",
  "hipDeviceCanAccessPeer",
  "hipDeviceComputeCapability",
  "hipDeviceDisablePeerAccess",
  "hipDeviceEnablePeerAccess",
  "hipDeviceGet",
  "hipDeviceGetAttribute",
  "hipDeviceGetByPCIBusId",
  "hipDeviceGetCacheConfig",
  "hipDeviceGetDefaultMemPool",
  "hipDeviceGetGraphMemAttribute",
  "hipDeviceGetLimit",
  "hipDeviceGetMemPool",
  "hipDeviceGetName",
  "hipDeviceGetP2PAttribute",
  "hipDeviceGetPCIBusId",
  "hipDeviceGetSharedMemConfig",
  "hipDeviceGetStreamPriorityRange",
  "hipDeviceGetTexture1DLinearMaxWidth",
  "hipDeviceGetUuid",
  "hipDeviceGraphMemTrim",
  "hipDevicePrimaryCtxGetState",
  "hipDevicePrimaryCtxRelease",
  "hipDevicePrimaryCtxReset",
  "hipDevicePrimaryCtxRetain",
  "hipDevicePrimaryCtxSetFlags",
  "hipDeviceReset",
  "hipDeviceSetCacheConfig",
  "hipDeviceSetGraphMemAttribute",
  "hipDeviceSetLimit",
  "hipDeviceSetMemPool",
  "hipDeviceSetSharedMemConfig",
  "hipDeviceSynchronize",
  "hipDeviceTotalMem",
  "hipDriverGetVersion",
  "hipDrvGetErrorName",
  "hipDrvGetErrorString",
  "hipDrvGraphAddMemcpyNode",
  "hipDrvMemcpy2DUnaligned",
  "hipDrvMemcpy3D",
  "hipDrvMemcpy3DAsync",
  "hipDrvPointerGetAttributes",
  "hipEventCreate",
  "hipEventCreateWithFlags",
  "hipEventDestroy",
  "hipEventElapsedTime",
  "hipEventQuery",
  "hipEventRecord",
  "hipEventSynchronize",
  "hipExtGetLinkTypeAndHopCount",
  "hipExtLaunchKernel",
  "hipExtLaunchMultiKernelMultiDevice",
  "hipExtMallocWithFlags",
  "hipExtStreamCreateWithCUMask",
  "hipExtStreamGetCUMask",
  "hipExternalMemoryGetMappedBuffer",
  "hipFree",
  "hipFreeArray",
  "hipFreeAsync",
  "hipFreeHost",
  "hipFreeMipmappedArray",
  "hipFuncGetAttribute",
  "hipFuncGetAttributes",
  "hipFuncSetAttribute",
  "hipFuncSetCacheConfig",
  "hipFuncSetSharedMemConfig",
  "hipGLGetDevices",
  "hipGetChannelDesc",
  "hipGetDevice",
  "hipGetDeviceCount",
  "hipGetDeviceFlags",
  "hipGetDevicePropertiesR0600",
  "hipGetErrorName",
  "hipGetErrorString",
  "hipGetLastError",
  "hipGetMipmappedArrayLevel",
  "hipGetSymbolAddress",
  "hipGetSymbolSize",
  "hipGetTextureAlignmentOffset",
  "hipGetTextureObjectResourceDesc",
  "hipGetTextureObjectResourceViewDesc",
  "hipGetTextureObjectTextureDesc",
  "hipGetTextureReference",
  "hipGraphAddChildGraphNode",
  "hipGraphAddDependencies",
  "hipGraphAddEmptyNode",
  "hipGraphAddEventRecordNode",
  "hipGraphAddEventWaitNode",
  "hipGraphAddHostNode",
  "hipGraphAddKernelNode",
  "hipGraphAddMemAllocNode",
  "hipGraphAddMemFreeNode",
  "hipGraphAddMemcpyNode",
  "hipGraphAddMemcpyNode1D",
  "hipGraphAddMemcpyNodeFromSymbol",
  "hipGraphAddMemcpyNodeToSymbol",
  "hipGraphAddMemsetNode",
  "hipGraphAddNode",
  "hipGraphChildGraphNodeGetGraph",
  "hipGraphClone",
  "hipGraphCreate",
  "hipGraphDebugDotPrint",
  "hipGraphDestroy",
  "hipGraphDestroyNode",
  "hipGraphEventRecordNodeGetEvent",
  "hipGraphEventRecordNodeSetEvent",
  "hipGraphEventWaitNodeGetEvent",
  "hipGraphEventWaitNodeSetEvent",
  "hipGraphExecChildGraphNodeSetParams",
  "hipGraphExecDestroy",
  "hipGraphExecEventRecordNodeSetEvent",
  "hipGraphExecEventWaitNodeSetEvent",
  "hipGraphExecHostNodeSetParams",
  "hipGraphExecKernelNodeSetParams",
  "hipGraphExecMemcpyNodeSetParams",
  "hipGraphExecMemcpyNodeSetParams1D",
  "hipGraphExecMemcpyNodeSetParamsFromSymbol",
  "hipGraphExecMemcpyNodeSetParamsToSymbol",
  "hipGraphExecMemsetNodeSetParams",
  "hipGraphExecUpdate",
  "hipGraphGetEdges",
  "hipGraphGetNodes",
  "hipGraphGetRootNodes",
  "hipGraphHostNodeGetParams",
  "hipGraphHostNodeSetParams",
  "hipGraphInstantiate",
  "hipGraphInstantiateWithFlags",
  "hipGraphInstantiateWithParams",
  "hipGraphKernelNodeCopyAttributes",
  "hipGraphKernelNodeGetAttribute",
  "hipGraphKernelNodeGetParams",
  "hipGraphKernelNodeSetAttribute",
  "hipGraphKernelNodeSetParams",
  "hipGraphLaunch",
  "hipGraphMemAllocNodeGetParams",
  "hipGraphMemFreeNodeGetParams",
  "hipGraphMemcpyNodeGetParams",
  "hipGraphMemcpyNodeSetParams",
  "hipGraphMemcpyNodeSetParams1D",
  "hipGraphMemcpyNodeSetParamsFromSymbol",
  "hipGraphMemcpyNodeSetParamsToSymbol",
  "hipGraphMemsetNodeGetParams",
  "hipGraphMemsetNodeSetParams",
  "hipGraphNodeFindInClone",
  "hipGraphNodeGetDependencies",
  "hipGraphNodeGetDependentNodes",
  "hipGraphNodeGetEnabled",
  "hipGraphNodeGetType",
  "hipGraphNodeSetEnabled",
  "hipGraphReleaseUserObject",
  "hipGraphRemoveDependencies",
  "hipGraphRetainUserObject",
  "hipGraphUpload",
  "hipGraphicsGLRegisterBuffer",
  "hipGraphicsGLRegisterImage",
  "hipGraphicsMapResources",
  "hipGraphicsResourceGetMappedPointer",
  "hipGraphicsSubResourceGetMappedArray",
  "hipGraphicsUnmapResources",
  "hipGraphicsUnregisterResource",
  "hipHostAlloc",
  "hipHostFree",
  "hipHostGetDevicePointer",
  "hipHostGetFlags",
  "hipHostMalloc",
  "hipExtHostAlloc",
  "hipHostRegister",
  "hipHostUnregister",
  "hipImportExternalMemory",
  "hipImportExternalSemaphore",
  "hipInit",
  "hipIpcCloseMemHandle",
  "hipIpcGetEventHandle",
  "hipIpcGetMemHandle",
  "hipIpcOpenEventHandle",
  "hipIpcOpenMemHandle",
  "hipKernelNameRef",
  "hipKernelNameRefByPtr",
  "hipLaunchByPtr",
  "hipLaunchCooperativeKernel",
  "hipLaunchCooperativeKernelMultiDevice",
  "hipLaunchHostFunc",
  "hipLaunchKernel",
  "hipMalloc",
  "hipMalloc3D",
  "hipMalloc3DArray",
  "hipMallocArray",
  "hipMallocAsync",
  "hipMallocFromPoolAsync",
  "hipMallocHost",
  "hipMallocManaged",
  "hipMallocMipmappedArray",
  "hipMallocPitch",
  "hipMemAddressFree",
  "hipMemAddressReserve",
  "hipMemAdvise",
  "hipMemAdvise_v2",
  "hipMemAllocHost",
  "hipMemAllocPitch",
  "hipMemCreate",
  "hipMemExportToShareableHandle",
  "hipMemGetAccess",
  "hipMemGetAddressRange",
  "hipMemGetAllocationGranularity",
  "hipMemGetAllocationPropertiesFromHandle",
  "hipMemGetInfo",
  "hipMemImportFromShareableHandle",
  "hipMemMap",
  "hipMemMapArrayAsync",
  "hipMemPoolCreate",
  "hipMemPoolDestroy",
  "hipMemPoolExportPointer",
  "hipMemPoolExportToShareableHandle",
  "hipMemPoolGetAccess",
  "hipMemPoolGetAttribute",
  "hipMemPoolImportFromShareableHandle",
  "hipMemPoolImportPointer",
  "hipMemPoolSetAccess",
  "hipMemPoolSetAttribute",
  "hipMemPoolTrimTo",
  "hipMemPrefetchAsync",
  "hipMemPrefetchAsync_v2",
  "hipMemPrefetchBatchAsync",
  "hipMemPtrGetInfo",
  "hipMemRangeGetAttribute",
  "hipMemRangeGetAttributes",
  "hipMemRelease",
  "hipMemRetainAllocationHandle",
  "hipMemSetAccess",
  "hipMemUnmap",
  "hipMemcpy",
  "hipMemcpy2D",
  "hipMemcpy2DAsync",
  "hipMemcpy2DFromArray",
  "hipMemcpy2DFromArrayAsync",
  "hipMemcpy2DToArray",
  "hipMemcpy2DToArrayAsync",
  "hipMemcpy3D",
  "hipMemcpy3DAsync",
  "hipMemcpyAsync",
  "hipMemcpyAtoH",
  "hipMemcpyDtoD",
  "hipMemcpyDtoDAsync",
  "hipMemcpyDtoH",
  "hipMemcpyDtoHAsync",
  "hipMemcpyFromArray",
  "hipMemcpyFromSymbol",
  "hipMemcpyFromSymbolAsync",
  "hipMemcpyHtoA",
  "hipMemcpyHtoD",
  "hipMemcpyHtoDAsync",
  "hipMemcpyParam2D",
  "hipMemcpyParam2DAsync",
  "hipMemcpyPeer",
  "hipMemcpyPeerAsync",
  "hipMemcpyToArray",
  "hipMemcpyToSymbol",
  "hipMemcpyToSymbolAsync",
  "hipMemcpyWithStream",
  "hipMemset",
  "hipMemset2D",
  "hipMemset2DAsync",
  "hipMemset3D",
  "hipMemset3DAsync",
  "hipMemsetAsync",
  "hipMemsetD16",
  "hipMemsetD16Async",
  "hipMemsetD32",
  "hipMemsetD32Async",
  "hipMemsetD8",
  "hipMemsetD8Async",
  "hipMipmappedArrayCreate",
  "hipMipmappedArrayDestroy",
  "hipMipmappedArrayGetMemoryRequirements",
  "hipMipmappedArrayGetLevel",
  "hipModuleGetFunction",
  "hipModuleGetFunctionCount",
  "hipModuleGetGlobal",
  "hipModuleGetTexRef",
  "hipModuleLaunchCooperativeKernel",
  "hipModuleLaunchCooperativeKernelMultiDevice",
  "hipModuleLaunchKernel",
  "hipModuleLoadFatBinary",
  "hipModuleLoad",
  "hipModuleLoadData",
  "hipModuleLoadDataEx",
  "hipLinkAddData",
  "hipLinkAddFile",
  "hipLinkComplete",
  "hipLinkCreate",
  "hipLinkDestroy",
  "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
  "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "hipModuleOccupancyMaxPotentialBlockSize",
  "hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
  "hipModuleUnload",
  "hipOccupancyMaxActiveBlocksPerMultiprocessor",
  "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "hipOccupancyMaxPotentialBlockSize",
  "hipOccupancyMaxActiveClusters",
  "hipOccupancyMaxPotentialClusterSize",
  "hipPeekAtLastError",
  "hipPointerGetAttribute",
  "hipPointerGetAttributes",
  "hipPointerSetAttribute",
  "hipProfilerStart",
  "hipProfilerStop",
  "hipRuntimeGetVersion",
  "hipSetDevice",
  "hipSetDeviceFlags",
  "hipSetupArgument",
  "hipSignalExternalSemaphoresAsync",
  "hipStreamAddCallback",
  "hipStreamAttachMemAsync",
  "hipStreamBeginCapture",
  "hipStreamCopyAttributes",
  "hipStreamCreate",
  "hipStreamCreateWithFlags",
  "hipStreamCreateWithPriority",
  "hipStreamDestroy",
  "hipStreamEndCapture",
  "hipStreamGetCaptureInfo",
  "hipStreamGetCaptureInfo_v2",
  "hipStreamGetDevice",
  "hipStreamGetFlags",
  "hipStreamGetId",
  "hipStreamGetPriority",
  "hipStreamIsCapturing",
  "hipStreamQuery",
  "hipStreamSynchronize",
  "hipStreamUpdateCaptureDependencies",
  "hipStreamWaitEvent",
  "hipStreamWaitValue32",
  "hipStreamWaitValue64",
  "hipStreamWriteValue32",
  "hipStreamWriteValue64",
  "hipStreamBatchMemOp",
  "hipTexObjectCreate",
  "hipTexObjectDestroy",
  "hipTexObjectGetResourceDesc",
  "hipTexObjectGetResourceViewDesc",
  "hipTexObjectGetTextureDesc",
  "hipTexRefGetAddress",
  "hipTexRefGetAddressMode",
  "hipTexRefGetFilterMode",
  "hipTexRefGetFlags",
  "hipTexRefGetFormat",
  "hipTexRefGetMaxAnisotropy",
  "hipTexRefGetMipMappedArray",
  "hipTexRefGetMipmapFilterMode",
  "hipTexRefGetMipmapLevelBias",
  "hipTexRefGetMipmapLevelClamp",
  "hipTexRefSetAddress",
  "hipTexRefSetAddress2D",
  "hipTexRefSetAddressMode",
  "hipTexRefSetArray",
  "hipTexRefSetBorderColor",
  "hipTexRefSetFilterMode",
  "hipTexRefSetFlags",
  "hipTexRefSetFormat",
  "hipTexRefSetMaxAnisotropy",
  "hipTexRefSetMipmapFilterMode",
  "hipTexRefSetMipmapLevelBias",
  "hipTexRefSetMipmapLevelClamp",
  "hipTexRefSetMipmappedArray",
  "hipThreadExchangeStreamCaptureMode",
  "hipUnbindTexture",
  "hipUserObjectCreate",
  "hipUserObjectRelease",
  "hipUserObjectRetain",
  "hipWaitExternalSemaphoresAsync",
  "hipCreateChannelDesc",
  "hipExtModuleLaunchKernel",
  "hipHccModuleLaunchKernel",
  "hipMemcpy_spt",
  "hipMemcpyToSymbol_spt",
  "hipMemcpyFromSymbol_spt",
  "hipMemcpy2D_spt",
  "hipMemcpy2DFromArray_spt",
  "hipMemcpy3D_spt",
  "hipMemset_spt",
  "hipMemsetAsync_spt",
  "hipMemset2D_spt",
  "hipMemset2DAsync_spt",
  "hipMemset3DAsync_spt",
  "hipMemset3D_spt",
  "hipMemcpyAsync_spt",
  "hipMemcpy3DAsync_spt",
  "hipMemcpy2DAsync_spt",
  "hipMemcpyFromSymbolAsync_spt",
  "hipMemcpyToSymbolAsync_spt",
  "hipMemcpyFromArray_spt",
  "hipMemcpy2DToArray_spt",
  "hipMemcpy2DFromArrayAsync_spt",
  "hipMemcpy2DToArrayAsync_spt",
  "hipStreamQuery_spt",
  "hipStreamSynchronize_spt",
  "hipStreamGetPriority_spt",
  "hipStreamWaitEvent_spt",
  "hipStreamGetFlags_spt",
  "hipStreamAddCallback_spt",
  "hipEventRecord_spt",
  "hipLaunchCooperativeKernel_spt",
  "hipLaunchKernel_spt",
  "hipGraphLaunch_spt",
  "hipStreamBeginCapture_spt",
  "hipStreamEndCapture_spt",
  "hipStreamIsCapturing_spt",
  "hipStreamGetCaptureInfo_spt",
  "hipStreamGetCaptureInfo_v2_spt",
  "hipLaunchHostFunc_spt",
  "hipGetStreamDeviceId",
  "hipDrvGraphAddMemsetNode",
  "hipGetDevicePropertiesR0000",
  "hipGetDriverEntryPoint",
  "hipGetDriverEntryPoint_spt",
  "hipExtGetLastError",
  "hipTexRefGetBorderColor",
  "hipTexRefGetArray",
  "hipGetProcAddress",
  "hipGetProcAddress_spt",
  "hipStreamBeginCaptureToGraph",
  "hipGetFuncBySymbol",
  "hipSetValidDevices",
  "hipMemcpyAtoD",
  "hipMemcpyDtoA",
  "hipMemcpyAtoA",
  "hipMemcpyAtoHAsync",
  "hipMemcpyHtoAAsync",
  "hipMemcpy2DArrayToArray",
  "hipDrvGraphAddMemFreeNode",
  "hipDrvGraphExecMemcpyNodeSetParams",
  "hipDrvGraphExecMemsetNodeSetParams",
  "hipGraphExecGetFlags",
  "hipGraphNodeSetParams",
  "hipGraphExecNodeSetParams",
  "hipExternalMemoryGetMappedMipmappedArray",
  "hipDrvGraphMemcpyNodeGetParams",
  "hipDrvGraphMemcpyNodeSetParams",
  "hipGraphAddBatchMemOpNode",
  "hipGraphBatchMemOpNodeGetParams",
  "hipGraphBatchMemOpNodeSetParams",
  "hipGraphExecBatchMemOpNodeSetParams",
  "hipEventRecordWithFlags",
  "hipLaunchKernelExC",
  "hipDrvLaunchKernelEx",
  "hipMemGetHandleForAddressRange",
  "hipMemsetD2D8",
  "hipMemsetD2D8Async",
  "hipMemsetD2D16",
  "hipMemsetD2D16Async",
  "hipMemsetD2D32",
  "hipMemsetD2D32Async",
  "hipStreamGetAttribute",
  "hipStreamSetAttribute",
  "hipMemcpyBatchAsync",
  "hipMemcpy3DBatchAsync",
  "hipMemcpy3DPeer",
  "hipMemcpy3DPeerAsync",
  "hipLibraryLoadData",
  "hipLibraryLoadFromFile",
  "hipLibraryUnload",
  "hipLibraryGetKernel",
  "hipLibraryGetKernelCount",
  "hipLibraryEnumerateKernels",
  "hipKernelGetLibrary",
  "hipKernelGetName",
  "hipOccupancyAvailableDynamicSMemPerBlock",
  "hipKernelGetParamInfo",
  "hipExtDisableLogging",
  "hipExtEnableLogging",
  "hipExtSetLoggingParams",
  "hipMemSetMemPool",
  "hipMemGetMemPool",
  "hipKernelGetAttribute",
  "hipKernelSetAttribute",
  "hipKernelGetFunction",
};
const size_t kHipApiNamesCountExt = 512;

#include <cstring>

void HipProfilerBuildWrapperTableExt(HipDispatchTable* tbl) {
  g_next = *tbl;
  g_wrapper_tbl = g_next;  // start from a full valid copy so any unhooked fields pass through
  g_wrapper_tbl.hipApiName_fn = hipApiNameLayer;
  g_wrapper_tbl.hipArray3DCreate_fn = hipArray3DCreateLayer;
  g_wrapper_tbl.hipArray3DGetDescriptor_fn = hipArray3DGetDescriptorLayer;
  g_wrapper_tbl.hipArrayCreate_fn = hipArrayCreateLayer;
  g_wrapper_tbl.hipArrayDestroy_fn = hipArrayDestroyLayer;
  g_wrapper_tbl.hipArrayGetDescriptor_fn = hipArrayGetDescriptorLayer;
  g_wrapper_tbl.hipArrayGetInfo_fn = hipArrayGetInfoLayer;
  g_wrapper_tbl.hipBindTexture_fn = hipBindTextureLayer;
  g_wrapper_tbl.hipBindTexture2D_fn = hipBindTexture2DLayer;
  g_wrapper_tbl.hipBindTextureToArray_fn = hipBindTextureToArrayLayer;
  g_wrapper_tbl.hipBindTextureToMipmappedArray_fn = hipBindTextureToMipmappedArrayLayer;
  g_wrapper_tbl.hipChooseDevice_fn = hipChooseDeviceLayer;
  g_wrapper_tbl.hipChooseDeviceR0000_fn = hipChooseDeviceR0000Layer;
  g_wrapper_tbl.hipConfigureCall_fn = hipConfigureCallLayer;
  g_wrapper_tbl.hipCreateSurfaceObject_fn = hipCreateSurfaceObjectLayer;
  g_wrapper_tbl.hipCreateTextureObject_fn = hipCreateTextureObjectLayer;
  g_wrapper_tbl.hipCtxCreate_fn = hipCtxCreateLayer;
  g_wrapper_tbl.hipCtxDestroy_fn = hipCtxDestroyLayer;
  g_wrapper_tbl.hipCtxDisablePeerAccess_fn = hipCtxDisablePeerAccessLayer;
  g_wrapper_tbl.hipCtxEnablePeerAccess_fn = hipCtxEnablePeerAccessLayer;
  g_wrapper_tbl.hipCtxGetApiVersion_fn = hipCtxGetApiVersionLayer;
  g_wrapper_tbl.hipCtxGetCacheConfig_fn = hipCtxGetCacheConfigLayer;
  g_wrapper_tbl.hipCtxGetCurrent_fn = hipCtxGetCurrentLayer;
  g_wrapper_tbl.hipCtxGetDevice_fn = hipCtxGetDeviceLayer;
  g_wrapper_tbl.hipCtxGetFlags_fn = hipCtxGetFlagsLayer;
  g_wrapper_tbl.hipCtxGetSharedMemConfig_fn = hipCtxGetSharedMemConfigLayer;
  g_wrapper_tbl.hipCtxPopCurrent_fn = hipCtxPopCurrentLayer;
  g_wrapper_tbl.hipCtxPushCurrent_fn = hipCtxPushCurrentLayer;
  g_wrapper_tbl.hipCtxSetCacheConfig_fn = hipCtxSetCacheConfigLayer;
  g_wrapper_tbl.hipCtxSetCurrent_fn = hipCtxSetCurrentLayer;
  g_wrapper_tbl.hipCtxSetSharedMemConfig_fn = hipCtxSetSharedMemConfigLayer;
  g_wrapper_tbl.hipCtxSynchronize_fn = hipCtxSynchronizeLayer;
  g_wrapper_tbl.hipDestroyExternalMemory_fn = hipDestroyExternalMemoryLayer;
  g_wrapper_tbl.hipDestroyExternalSemaphore_fn = hipDestroyExternalSemaphoreLayer;
  g_wrapper_tbl.hipDestroySurfaceObject_fn = hipDestroySurfaceObjectLayer;
  g_wrapper_tbl.hipDestroyTextureObject_fn = hipDestroyTextureObjectLayer;
  g_wrapper_tbl.hipDeviceCanAccessPeer_fn = hipDeviceCanAccessPeerLayer;
  g_wrapper_tbl.hipDeviceComputeCapability_fn = hipDeviceComputeCapabilityLayer;
  g_wrapper_tbl.hipDeviceDisablePeerAccess_fn = hipDeviceDisablePeerAccessLayer;
  g_wrapper_tbl.hipDeviceEnablePeerAccess_fn = hipDeviceEnablePeerAccessLayer;
  g_wrapper_tbl.hipDeviceGet_fn = hipDeviceGetLayer;
  g_wrapper_tbl.hipDeviceGetAttribute_fn = hipDeviceGetAttributeLayer;
  g_wrapper_tbl.hipDeviceGetByPCIBusId_fn = hipDeviceGetByPCIBusIdLayer;
  g_wrapper_tbl.hipDeviceGetCacheConfig_fn = hipDeviceGetCacheConfigLayer;
  g_wrapper_tbl.hipDeviceGetDefaultMemPool_fn = hipDeviceGetDefaultMemPoolLayer;
  g_wrapper_tbl.hipDeviceGetGraphMemAttribute_fn = hipDeviceGetGraphMemAttributeLayer;
  g_wrapper_tbl.hipDeviceGetLimit_fn = hipDeviceGetLimitLayer;
  g_wrapper_tbl.hipDeviceGetMemPool_fn = hipDeviceGetMemPoolLayer;
  g_wrapper_tbl.hipDeviceGetName_fn = hipDeviceGetNameLayer;
  g_wrapper_tbl.hipDeviceGetP2PAttribute_fn = hipDeviceGetP2PAttributeLayer;
  g_wrapper_tbl.hipDeviceGetPCIBusId_fn = hipDeviceGetPCIBusIdLayer;
  g_wrapper_tbl.hipDeviceGetSharedMemConfig_fn = hipDeviceGetSharedMemConfigLayer;
  g_wrapper_tbl.hipDeviceGetStreamPriorityRange_fn = hipDeviceGetStreamPriorityRangeLayer;
  g_wrapper_tbl.hipDeviceGetTexture1DLinearMaxWidth_fn = hipDeviceGetTexture1DLinearMaxWidthLayer;
  g_wrapper_tbl.hipDeviceGetUuid_fn = hipDeviceGetUuidLayer;
  g_wrapper_tbl.hipDeviceGraphMemTrim_fn = hipDeviceGraphMemTrimLayer;
  g_wrapper_tbl.hipDevicePrimaryCtxGetState_fn = hipDevicePrimaryCtxGetStateLayer;
  g_wrapper_tbl.hipDevicePrimaryCtxRelease_fn = hipDevicePrimaryCtxReleaseLayer;
  g_wrapper_tbl.hipDevicePrimaryCtxReset_fn = hipDevicePrimaryCtxResetLayer;
  g_wrapper_tbl.hipDevicePrimaryCtxRetain_fn = hipDevicePrimaryCtxRetainLayer;
  g_wrapper_tbl.hipDevicePrimaryCtxSetFlags_fn = hipDevicePrimaryCtxSetFlagsLayer;
  g_wrapper_tbl.hipDeviceReset_fn = hipDeviceResetLayer;
  g_wrapper_tbl.hipDeviceSetCacheConfig_fn = hipDeviceSetCacheConfigLayer;
  g_wrapper_tbl.hipDeviceSetGraphMemAttribute_fn = hipDeviceSetGraphMemAttributeLayer;
  g_wrapper_tbl.hipDeviceSetLimit_fn = hipDeviceSetLimitLayer;
  g_wrapper_tbl.hipDeviceSetMemPool_fn = hipDeviceSetMemPoolLayer;
  g_wrapper_tbl.hipDeviceSetSharedMemConfig_fn = hipDeviceSetSharedMemConfigLayer;
  g_wrapper_tbl.hipDeviceSynchronize_fn = hipDeviceSynchronizeLayer;
  g_wrapper_tbl.hipDeviceTotalMem_fn = hipDeviceTotalMemLayer;
  g_wrapper_tbl.hipDriverGetVersion_fn = hipDriverGetVersionLayer;
  g_wrapper_tbl.hipDrvGetErrorName_fn = hipDrvGetErrorNameLayer;
  g_wrapper_tbl.hipDrvGetErrorString_fn = hipDrvGetErrorStringLayer;
  g_wrapper_tbl.hipDrvGraphAddMemcpyNode_fn = hipDrvGraphAddMemcpyNodeLayer;
  g_wrapper_tbl.hipDrvMemcpy2DUnaligned_fn = hipDrvMemcpy2DUnalignedLayer;
  g_wrapper_tbl.hipDrvMemcpy3D_fn = hipDrvMemcpy3DLayer;
  g_wrapper_tbl.hipDrvMemcpy3DAsync_fn = hipDrvMemcpy3DAsyncLayer;
  g_wrapper_tbl.hipDrvPointerGetAttributes_fn = hipDrvPointerGetAttributesLayer;
  g_wrapper_tbl.hipEventCreate_fn = hipEventCreateLayer;
  g_wrapper_tbl.hipEventCreateWithFlags_fn = hipEventCreateWithFlagsLayer;
  g_wrapper_tbl.hipEventDestroy_fn = hipEventDestroyLayer;
  g_wrapper_tbl.hipEventElapsedTime_fn = hipEventElapsedTimeLayer;
  g_wrapper_tbl.hipEventQuery_fn = hipEventQueryLayer;
  g_wrapper_tbl.hipEventRecord_fn = hipEventRecordLayer;
  g_wrapper_tbl.hipEventSynchronize_fn = hipEventSynchronizeLayer;
  g_wrapper_tbl.hipExtGetLinkTypeAndHopCount_fn = hipExtGetLinkTypeAndHopCountLayer;
  g_wrapper_tbl.hipExtLaunchKernel_fn = hipExtLaunchKernelLayer;
  g_wrapper_tbl.hipExtLaunchMultiKernelMultiDevice_fn = hipExtLaunchMultiKernelMultiDeviceLayer;
  g_wrapper_tbl.hipExtMallocWithFlags_fn = hipExtMallocWithFlagsLayer;
  g_wrapper_tbl.hipExtStreamCreateWithCUMask_fn = hipExtStreamCreateWithCUMaskLayer;
  g_wrapper_tbl.hipExtStreamGetCUMask_fn = hipExtStreamGetCUMaskLayer;
  g_wrapper_tbl.hipExternalMemoryGetMappedBuffer_fn = hipExternalMemoryGetMappedBufferLayer;
  g_wrapper_tbl.hipFree_fn = hipFreeLayer;
  g_wrapper_tbl.hipFreeArray_fn = hipFreeArrayLayer;
  g_wrapper_tbl.hipFreeAsync_fn = hipFreeAsyncLayer;
  g_wrapper_tbl.hipFreeHost_fn = hipFreeHostLayer;
  g_wrapper_tbl.hipFreeMipmappedArray_fn = hipFreeMipmappedArrayLayer;
  g_wrapper_tbl.hipFuncGetAttribute_fn = hipFuncGetAttributeLayer;
  g_wrapper_tbl.hipFuncGetAttributes_fn = hipFuncGetAttributesLayer;
  g_wrapper_tbl.hipFuncSetAttribute_fn = hipFuncSetAttributeLayer;
  g_wrapper_tbl.hipFuncSetCacheConfig_fn = hipFuncSetCacheConfigLayer;
  g_wrapper_tbl.hipFuncSetSharedMemConfig_fn = hipFuncSetSharedMemConfigLayer;
  g_wrapper_tbl.hipGLGetDevices_fn = hipGLGetDevicesLayer;
  g_wrapper_tbl.hipGetChannelDesc_fn = hipGetChannelDescLayer;
  g_wrapper_tbl.hipGetDevice_fn = hipGetDeviceLayer;
  g_wrapper_tbl.hipGetDeviceCount_fn = hipGetDeviceCountLayer;
  g_wrapper_tbl.hipGetDeviceFlags_fn = hipGetDeviceFlagsLayer;
  g_wrapper_tbl.hipGetDevicePropertiesR0600_fn = hipGetDevicePropertiesR0600Layer;
  g_wrapper_tbl.hipGetErrorName_fn = hipGetErrorNameLayer;
  g_wrapper_tbl.hipGetErrorString_fn = hipGetErrorStringLayer;
  g_wrapper_tbl.hipGetLastError_fn = hipGetLastErrorLayer;
  g_wrapper_tbl.hipGetMipmappedArrayLevel_fn = hipGetMipmappedArrayLevelLayer;
  g_wrapper_tbl.hipGetSymbolAddress_fn = hipGetSymbolAddressLayer;
  g_wrapper_tbl.hipGetSymbolSize_fn = hipGetSymbolSizeLayer;
  g_wrapper_tbl.hipGetTextureAlignmentOffset_fn = hipGetTextureAlignmentOffsetLayer;
  g_wrapper_tbl.hipGetTextureObjectResourceDesc_fn = hipGetTextureObjectResourceDescLayer;
  g_wrapper_tbl.hipGetTextureObjectResourceViewDesc_fn = hipGetTextureObjectResourceViewDescLayer;
  g_wrapper_tbl.hipGetTextureObjectTextureDesc_fn = hipGetTextureObjectTextureDescLayer;
  g_wrapper_tbl.hipGetTextureReference_fn = hipGetTextureReferenceLayer;
  g_wrapper_tbl.hipGraphAddChildGraphNode_fn = hipGraphAddChildGraphNodeLayer;
  g_wrapper_tbl.hipGraphAddDependencies_fn = hipGraphAddDependenciesLayer;
  g_wrapper_tbl.hipGraphAddEmptyNode_fn = hipGraphAddEmptyNodeLayer;
  g_wrapper_tbl.hipGraphAddEventRecordNode_fn = hipGraphAddEventRecordNodeLayer;
  g_wrapper_tbl.hipGraphAddEventWaitNode_fn = hipGraphAddEventWaitNodeLayer;
  g_wrapper_tbl.hipGraphAddHostNode_fn = hipGraphAddHostNodeLayer;
  g_wrapper_tbl.hipGraphAddKernelNode_fn = hipGraphAddKernelNodeLayer;
  g_wrapper_tbl.hipGraphAddMemAllocNode_fn = hipGraphAddMemAllocNodeLayer;
  g_wrapper_tbl.hipGraphAddMemFreeNode_fn = hipGraphAddMemFreeNodeLayer;
  g_wrapper_tbl.hipGraphAddMemcpyNode_fn = hipGraphAddMemcpyNodeLayer;
  g_wrapper_tbl.hipGraphAddMemcpyNode1D_fn = hipGraphAddMemcpyNode1DLayer;
  g_wrapper_tbl.hipGraphAddMemcpyNodeFromSymbol_fn = hipGraphAddMemcpyNodeFromSymbolLayer;
  g_wrapper_tbl.hipGraphAddMemcpyNodeToSymbol_fn = hipGraphAddMemcpyNodeToSymbolLayer;
  g_wrapper_tbl.hipGraphAddMemsetNode_fn = hipGraphAddMemsetNodeLayer;
  g_wrapper_tbl.hipGraphAddNode_fn = hipGraphAddNodeLayer;
  g_wrapper_tbl.hipGraphChildGraphNodeGetGraph_fn = hipGraphChildGraphNodeGetGraphLayer;
  g_wrapper_tbl.hipGraphClone_fn = hipGraphCloneLayer;
  g_wrapper_tbl.hipGraphCreate_fn = hipGraphCreateLayer;
  g_wrapper_tbl.hipGraphDebugDotPrint_fn = hipGraphDebugDotPrintLayer;
  g_wrapper_tbl.hipGraphDestroy_fn = hipGraphDestroyLayer;
  g_wrapper_tbl.hipGraphDestroyNode_fn = hipGraphDestroyNodeLayer;
  g_wrapper_tbl.hipGraphEventRecordNodeGetEvent_fn = hipGraphEventRecordNodeGetEventLayer;
  g_wrapper_tbl.hipGraphEventRecordNodeSetEvent_fn = hipGraphEventRecordNodeSetEventLayer;
  g_wrapper_tbl.hipGraphEventWaitNodeGetEvent_fn = hipGraphEventWaitNodeGetEventLayer;
  g_wrapper_tbl.hipGraphEventWaitNodeSetEvent_fn = hipGraphEventWaitNodeSetEventLayer;
  g_wrapper_tbl.hipGraphExecChildGraphNodeSetParams_fn = hipGraphExecChildGraphNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecDestroy_fn = hipGraphExecDestroyLayer;
  g_wrapper_tbl.hipGraphExecEventRecordNodeSetEvent_fn = hipGraphExecEventRecordNodeSetEventLayer;
  g_wrapper_tbl.hipGraphExecEventWaitNodeSetEvent_fn = hipGraphExecEventWaitNodeSetEventLayer;
  g_wrapper_tbl.hipGraphExecHostNodeSetParams_fn = hipGraphExecHostNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecKernelNodeSetParams_fn = hipGraphExecKernelNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecMemcpyNodeSetParams_fn = hipGraphExecMemcpyNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecMemcpyNodeSetParams1D_fn = hipGraphExecMemcpyNodeSetParams1DLayer;
  g_wrapper_tbl.hipGraphExecMemcpyNodeSetParamsFromSymbol_fn = hipGraphExecMemcpyNodeSetParamsFromSymbolLayer;
  g_wrapper_tbl.hipGraphExecMemcpyNodeSetParamsToSymbol_fn = hipGraphExecMemcpyNodeSetParamsToSymbolLayer;
  g_wrapper_tbl.hipGraphExecMemsetNodeSetParams_fn = hipGraphExecMemsetNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecUpdate_fn = hipGraphExecUpdateLayer;
  g_wrapper_tbl.hipGraphGetEdges_fn = hipGraphGetEdgesLayer;
  g_wrapper_tbl.hipGraphGetNodes_fn = hipGraphGetNodesLayer;
  g_wrapper_tbl.hipGraphGetRootNodes_fn = hipGraphGetRootNodesLayer;
  g_wrapper_tbl.hipGraphHostNodeGetParams_fn = hipGraphHostNodeGetParamsLayer;
  g_wrapper_tbl.hipGraphHostNodeSetParams_fn = hipGraphHostNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphInstantiate_fn = hipGraphInstantiateLayer;
  g_wrapper_tbl.hipGraphInstantiateWithFlags_fn = hipGraphInstantiateWithFlagsLayer;
  g_wrapper_tbl.hipGraphInstantiateWithParams_fn = hipGraphInstantiateWithParamsLayer;
  g_wrapper_tbl.hipGraphKernelNodeCopyAttributes_fn = hipGraphKernelNodeCopyAttributesLayer;
  g_wrapper_tbl.hipGraphKernelNodeGetAttribute_fn = hipGraphKernelNodeGetAttributeLayer;
  g_wrapper_tbl.hipGraphKernelNodeGetParams_fn = hipGraphKernelNodeGetParamsLayer;
  g_wrapper_tbl.hipGraphKernelNodeSetAttribute_fn = hipGraphKernelNodeSetAttributeLayer;
  g_wrapper_tbl.hipGraphKernelNodeSetParams_fn = hipGraphKernelNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphLaunch_fn = hipGraphLaunchLayer;
  g_wrapper_tbl.hipGraphMemAllocNodeGetParams_fn = hipGraphMemAllocNodeGetParamsLayer;
  g_wrapper_tbl.hipGraphMemFreeNodeGetParams_fn = hipGraphMemFreeNodeGetParamsLayer;
  g_wrapper_tbl.hipGraphMemcpyNodeGetParams_fn = hipGraphMemcpyNodeGetParamsLayer;
  g_wrapper_tbl.hipGraphMemcpyNodeSetParams_fn = hipGraphMemcpyNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphMemcpyNodeSetParams1D_fn = hipGraphMemcpyNodeSetParams1DLayer;
  g_wrapper_tbl.hipGraphMemcpyNodeSetParamsFromSymbol_fn = hipGraphMemcpyNodeSetParamsFromSymbolLayer;
  g_wrapper_tbl.hipGraphMemcpyNodeSetParamsToSymbol_fn = hipGraphMemcpyNodeSetParamsToSymbolLayer;
  g_wrapper_tbl.hipGraphMemsetNodeGetParams_fn = hipGraphMemsetNodeGetParamsLayer;
  g_wrapper_tbl.hipGraphMemsetNodeSetParams_fn = hipGraphMemsetNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphNodeFindInClone_fn = hipGraphNodeFindInCloneLayer;
  g_wrapper_tbl.hipGraphNodeGetDependencies_fn = hipGraphNodeGetDependenciesLayer;
  g_wrapper_tbl.hipGraphNodeGetDependentNodes_fn = hipGraphNodeGetDependentNodesLayer;
  g_wrapper_tbl.hipGraphNodeGetEnabled_fn = hipGraphNodeGetEnabledLayer;
  g_wrapper_tbl.hipGraphNodeGetType_fn = hipGraphNodeGetTypeLayer;
  g_wrapper_tbl.hipGraphNodeSetEnabled_fn = hipGraphNodeSetEnabledLayer;
  g_wrapper_tbl.hipGraphReleaseUserObject_fn = hipGraphReleaseUserObjectLayer;
  g_wrapper_tbl.hipGraphRemoveDependencies_fn = hipGraphRemoveDependenciesLayer;
  g_wrapper_tbl.hipGraphRetainUserObject_fn = hipGraphRetainUserObjectLayer;
  g_wrapper_tbl.hipGraphUpload_fn = hipGraphUploadLayer;
  g_wrapper_tbl.hipGraphicsGLRegisterBuffer_fn = hipGraphicsGLRegisterBufferLayer;
  g_wrapper_tbl.hipGraphicsGLRegisterImage_fn = hipGraphicsGLRegisterImageLayer;
  g_wrapper_tbl.hipGraphicsMapResources_fn = hipGraphicsMapResourcesLayer;
  g_wrapper_tbl.hipGraphicsResourceGetMappedPointer_fn = hipGraphicsResourceGetMappedPointerLayer;
  g_wrapper_tbl.hipGraphicsSubResourceGetMappedArray_fn = hipGraphicsSubResourceGetMappedArrayLayer;
  g_wrapper_tbl.hipGraphicsUnmapResources_fn = hipGraphicsUnmapResourcesLayer;
  g_wrapper_tbl.hipGraphicsUnregisterResource_fn = hipGraphicsUnregisterResourceLayer;
  g_wrapper_tbl.hipHostAlloc_fn = hipHostAllocLayer;
  g_wrapper_tbl.hipHostFree_fn = hipHostFreeLayer;
  g_wrapper_tbl.hipHostGetDevicePointer_fn = hipHostGetDevicePointerLayer;
  g_wrapper_tbl.hipHostGetFlags_fn = hipHostGetFlagsLayer;
  g_wrapper_tbl.hipHostMalloc_fn = hipHostMallocLayer;
  g_wrapper_tbl.hipExtHostAlloc_fn = hipExtHostAllocLayer;
  g_wrapper_tbl.hipHostRegister_fn = hipHostRegisterLayer;
  g_wrapper_tbl.hipHostUnregister_fn = hipHostUnregisterLayer;
  g_wrapper_tbl.hipImportExternalMemory_fn = hipImportExternalMemoryLayer;
  g_wrapper_tbl.hipImportExternalSemaphore_fn = hipImportExternalSemaphoreLayer;
  g_wrapper_tbl.hipInit_fn = hipInitLayer;
  g_wrapper_tbl.hipIpcCloseMemHandle_fn = hipIpcCloseMemHandleLayer;
  g_wrapper_tbl.hipIpcGetEventHandle_fn = hipIpcGetEventHandleLayer;
  g_wrapper_tbl.hipIpcGetMemHandle_fn = hipIpcGetMemHandleLayer;
  g_wrapper_tbl.hipIpcOpenEventHandle_fn = hipIpcOpenEventHandleLayer;
  g_wrapper_tbl.hipIpcOpenMemHandle_fn = hipIpcOpenMemHandleLayer;
  g_wrapper_tbl.hipKernelNameRef_fn = hipKernelNameRefLayer;
  g_wrapper_tbl.hipKernelNameRefByPtr_fn = hipKernelNameRefByPtrLayer;
  g_wrapper_tbl.hipLaunchByPtr_fn = hipLaunchByPtrLayer;
  g_wrapper_tbl.hipLaunchCooperativeKernel_fn = hipLaunchCooperativeKernelLayer;
  g_wrapper_tbl.hipLaunchCooperativeKernelMultiDevice_fn = hipLaunchCooperativeKernelMultiDeviceLayer;
  g_wrapper_tbl.hipLaunchHostFunc_fn = hipLaunchHostFuncLayer;
  g_wrapper_tbl.hipLaunchKernel_fn = hipLaunchKernelLayer;
  g_wrapper_tbl.hipMalloc_fn = hipMallocLayer;
  g_wrapper_tbl.hipMalloc3D_fn = hipMalloc3DLayer;
  g_wrapper_tbl.hipMalloc3DArray_fn = hipMalloc3DArrayLayer;
  g_wrapper_tbl.hipMallocArray_fn = hipMallocArrayLayer;
  g_wrapper_tbl.hipMallocAsync_fn = hipMallocAsyncLayer;
  g_wrapper_tbl.hipMallocFromPoolAsync_fn = hipMallocFromPoolAsyncLayer;
  g_wrapper_tbl.hipMallocHost_fn = hipMallocHostLayer;
  g_wrapper_tbl.hipMallocManaged_fn = hipMallocManagedLayer;
  g_wrapper_tbl.hipMallocMipmappedArray_fn = hipMallocMipmappedArrayLayer;
  g_wrapper_tbl.hipMallocPitch_fn = hipMallocPitchLayer;
  g_wrapper_tbl.hipMemAddressFree_fn = hipMemAddressFreeLayer;
  g_wrapper_tbl.hipMemAddressReserve_fn = hipMemAddressReserveLayer;
  g_wrapper_tbl.hipMemAdvise_fn = hipMemAdviseLayer;
  g_wrapper_tbl.hipMemAdvise_v2_fn = hipMemAdvise_v2Layer;
  g_wrapper_tbl.hipMemAllocHost_fn = hipMemAllocHostLayer;
  g_wrapper_tbl.hipMemAllocPitch_fn = hipMemAllocPitchLayer;
  g_wrapper_tbl.hipMemCreate_fn = hipMemCreateLayer;
  g_wrapper_tbl.hipMemExportToShareableHandle_fn = hipMemExportToShareableHandleLayer;
  g_wrapper_tbl.hipMemGetAccess_fn = hipMemGetAccessLayer;
  g_wrapper_tbl.hipMemGetAddressRange_fn = hipMemGetAddressRangeLayer;
  g_wrapper_tbl.hipMemGetAllocationGranularity_fn = hipMemGetAllocationGranularityLayer;
  g_wrapper_tbl.hipMemGetAllocationPropertiesFromHandle_fn = hipMemGetAllocationPropertiesFromHandleLayer;
  g_wrapper_tbl.hipMemGetInfo_fn = hipMemGetInfoLayer;
  g_wrapper_tbl.hipMemImportFromShareableHandle_fn = hipMemImportFromShareableHandleLayer;
  g_wrapper_tbl.hipMemMap_fn = hipMemMapLayer;
  g_wrapper_tbl.hipMemMapArrayAsync_fn = hipMemMapArrayAsyncLayer;
  g_wrapper_tbl.hipMemPoolCreate_fn = hipMemPoolCreateLayer;
  g_wrapper_tbl.hipMemPoolDestroy_fn = hipMemPoolDestroyLayer;
  g_wrapper_tbl.hipMemPoolExportPointer_fn = hipMemPoolExportPointerLayer;
  g_wrapper_tbl.hipMemPoolExportToShareableHandle_fn = hipMemPoolExportToShareableHandleLayer;
  g_wrapper_tbl.hipMemPoolGetAccess_fn = hipMemPoolGetAccessLayer;
  g_wrapper_tbl.hipMemPoolGetAttribute_fn = hipMemPoolGetAttributeLayer;
  g_wrapper_tbl.hipMemPoolImportFromShareableHandle_fn = hipMemPoolImportFromShareableHandleLayer;
  g_wrapper_tbl.hipMemPoolImportPointer_fn = hipMemPoolImportPointerLayer;
  g_wrapper_tbl.hipMemPoolSetAccess_fn = hipMemPoolSetAccessLayer;
  g_wrapper_tbl.hipMemPoolSetAttribute_fn = hipMemPoolSetAttributeLayer;
  g_wrapper_tbl.hipMemPoolTrimTo_fn = hipMemPoolTrimToLayer;
  g_wrapper_tbl.hipMemPrefetchAsync_fn = hipMemPrefetchAsyncLayer;
  g_wrapper_tbl.hipMemPrefetchAsync_v2_fn = hipMemPrefetchAsync_v2Layer;
  g_wrapper_tbl.hipMemPrefetchBatchAsync_fn = hipMemPrefetchBatchAsyncLayer;
  g_wrapper_tbl.hipMemPtrGetInfo_fn = hipMemPtrGetInfoLayer;
  g_wrapper_tbl.hipMemRangeGetAttribute_fn = hipMemRangeGetAttributeLayer;
  g_wrapper_tbl.hipMemRangeGetAttributes_fn = hipMemRangeGetAttributesLayer;
  g_wrapper_tbl.hipMemRelease_fn = hipMemReleaseLayer;
  g_wrapper_tbl.hipMemRetainAllocationHandle_fn = hipMemRetainAllocationHandleLayer;
  g_wrapper_tbl.hipMemSetAccess_fn = hipMemSetAccessLayer;
  g_wrapper_tbl.hipMemUnmap_fn = hipMemUnmapLayer;
  g_wrapper_tbl.hipMemcpy_fn = hipMemcpyLayer;
  g_wrapper_tbl.hipMemcpy2D_fn = hipMemcpy2DLayer;
  g_wrapper_tbl.hipMemcpy2DAsync_fn = hipMemcpy2DAsyncLayer;
  g_wrapper_tbl.hipMemcpy2DFromArray_fn = hipMemcpy2DFromArrayLayer;
  g_wrapper_tbl.hipMemcpy2DFromArrayAsync_fn = hipMemcpy2DFromArrayAsyncLayer;
  g_wrapper_tbl.hipMemcpy2DToArray_fn = hipMemcpy2DToArrayLayer;
  g_wrapper_tbl.hipMemcpy2DToArrayAsync_fn = hipMemcpy2DToArrayAsyncLayer;
  g_wrapper_tbl.hipMemcpy3D_fn = hipMemcpy3DLayer;
  g_wrapper_tbl.hipMemcpy3DAsync_fn = hipMemcpy3DAsyncLayer;
  g_wrapper_tbl.hipMemcpyAsync_fn = hipMemcpyAsyncLayer;
  g_wrapper_tbl.hipMemcpyAtoH_fn = hipMemcpyAtoHLayer;
  g_wrapper_tbl.hipMemcpyDtoD_fn = hipMemcpyDtoDLayer;
  g_wrapper_tbl.hipMemcpyDtoDAsync_fn = hipMemcpyDtoDAsyncLayer;
  g_wrapper_tbl.hipMemcpyDtoH_fn = hipMemcpyDtoHLayer;
  g_wrapper_tbl.hipMemcpyDtoHAsync_fn = hipMemcpyDtoHAsyncLayer;
  g_wrapper_tbl.hipMemcpyFromArray_fn = hipMemcpyFromArrayLayer;
  g_wrapper_tbl.hipMemcpyFromSymbol_fn = hipMemcpyFromSymbolLayer;
  g_wrapper_tbl.hipMemcpyFromSymbolAsync_fn = hipMemcpyFromSymbolAsyncLayer;
  g_wrapper_tbl.hipMemcpyHtoA_fn = hipMemcpyHtoALayer;
  g_wrapper_tbl.hipMemcpyHtoD_fn = hipMemcpyHtoDLayer;
  g_wrapper_tbl.hipMemcpyHtoDAsync_fn = hipMemcpyHtoDAsyncLayer;
  g_wrapper_tbl.hipMemcpyParam2D_fn = hipMemcpyParam2DLayer;
  g_wrapper_tbl.hipMemcpyParam2DAsync_fn = hipMemcpyParam2DAsyncLayer;
  g_wrapper_tbl.hipMemcpyPeer_fn = hipMemcpyPeerLayer;
  g_wrapper_tbl.hipMemcpyPeerAsync_fn = hipMemcpyPeerAsyncLayer;
  g_wrapper_tbl.hipMemcpyToArray_fn = hipMemcpyToArrayLayer;
  g_wrapper_tbl.hipMemcpyToSymbol_fn = hipMemcpyToSymbolLayer;
  g_wrapper_tbl.hipMemcpyToSymbolAsync_fn = hipMemcpyToSymbolAsyncLayer;
  g_wrapper_tbl.hipMemcpyWithStream_fn = hipMemcpyWithStreamLayer;
  g_wrapper_tbl.hipMemset_fn = hipMemsetLayer;
  g_wrapper_tbl.hipMemset2D_fn = hipMemset2DLayer;
  g_wrapper_tbl.hipMemset2DAsync_fn = hipMemset2DAsyncLayer;
  g_wrapper_tbl.hipMemset3D_fn = hipMemset3DLayer;
  g_wrapper_tbl.hipMemset3DAsync_fn = hipMemset3DAsyncLayer;
  g_wrapper_tbl.hipMemsetAsync_fn = hipMemsetAsyncLayer;
  g_wrapper_tbl.hipMemsetD16_fn = hipMemsetD16Layer;
  g_wrapper_tbl.hipMemsetD16Async_fn = hipMemsetD16AsyncLayer;
  g_wrapper_tbl.hipMemsetD32_fn = hipMemsetD32Layer;
  g_wrapper_tbl.hipMemsetD32Async_fn = hipMemsetD32AsyncLayer;
  g_wrapper_tbl.hipMemsetD8_fn = hipMemsetD8Layer;
  g_wrapper_tbl.hipMemsetD8Async_fn = hipMemsetD8AsyncLayer;
  g_wrapper_tbl.hipMipmappedArrayCreate_fn = hipMipmappedArrayCreateLayer;
  g_wrapper_tbl.hipMipmappedArrayDestroy_fn = hipMipmappedArrayDestroyLayer;
  g_wrapper_tbl.hipMipmappedArrayGetMemoryRequirements_fn = hipMipmappedArrayGetMemoryRequirementsLayer;
  g_wrapper_tbl.hipMipmappedArrayGetLevel_fn = hipMipmappedArrayGetLevelLayer;
  g_wrapper_tbl.hipModuleGetFunction_fn = hipModuleGetFunctionLayer;
  g_wrapper_tbl.hipModuleGetFunctionCount_fn = hipModuleGetFunctionCountLayer;
  g_wrapper_tbl.hipModuleGetGlobal_fn = hipModuleGetGlobalLayer;
  g_wrapper_tbl.hipModuleGetTexRef_fn = hipModuleGetTexRefLayer;
  g_wrapper_tbl.hipModuleLaunchCooperativeKernel_fn = hipModuleLaunchCooperativeKernelLayer;
  g_wrapper_tbl.hipModuleLaunchCooperativeKernelMultiDevice_fn = hipModuleLaunchCooperativeKernelMultiDeviceLayer;
  g_wrapper_tbl.hipModuleLaunchKernel_fn = hipModuleLaunchKernelLayer;
  g_wrapper_tbl.hipModuleLoadFatBinary_fn = hipModuleLoadFatBinaryLayer;
  g_wrapper_tbl.hipModuleLoad_fn = hipModuleLoadLayer;
  g_wrapper_tbl.hipModuleLoadData_fn = hipModuleLoadDataLayer;
  g_wrapper_tbl.hipModuleLoadDataEx_fn = hipModuleLoadDataExLayer;
  g_wrapper_tbl.hipLinkAddData_fn = hipLinkAddDataLayer;
  g_wrapper_tbl.hipLinkAddFile_fn = hipLinkAddFileLayer;
  g_wrapper_tbl.hipLinkComplete_fn = hipLinkCompleteLayer;
  g_wrapper_tbl.hipLinkCreate_fn = hipLinkCreateLayer;
  g_wrapper_tbl.hipLinkDestroy_fn = hipLinkDestroyLayer;
  g_wrapper_tbl.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn = hipModuleOccupancyMaxActiveBlocksPerMultiprocessorLayer;
  g_wrapper_tbl.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn = hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsLayer;
  g_wrapper_tbl.hipModuleOccupancyMaxPotentialBlockSize_fn = hipModuleOccupancyMaxPotentialBlockSizeLayer;
  g_wrapper_tbl.hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn = hipModuleOccupancyMaxPotentialBlockSizeWithFlagsLayer;
  g_wrapper_tbl.hipModuleUnload_fn = hipModuleUnloadLayer;
  g_wrapper_tbl.hipOccupancyMaxActiveBlocksPerMultiprocessor_fn = hipOccupancyMaxActiveBlocksPerMultiprocessorLayer;
  g_wrapper_tbl.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsLayer;
  g_wrapper_tbl.hipOccupancyMaxPotentialBlockSize_fn = hipOccupancyMaxPotentialBlockSizeLayer;
  g_wrapper_tbl.hipOccupancyMaxActiveClusters_fn = hipOccupancyMaxActiveClustersLayer;
  g_wrapper_tbl.hipOccupancyMaxPotentialClusterSize_fn = hipOccupancyMaxPotentialClusterSizeLayer;
  g_wrapper_tbl.hipPeekAtLastError_fn = hipPeekAtLastErrorLayer;
  g_wrapper_tbl.hipPointerGetAttribute_fn = hipPointerGetAttributeLayer;
  g_wrapper_tbl.hipPointerGetAttributes_fn = hipPointerGetAttributesLayer;
  g_wrapper_tbl.hipPointerSetAttribute_fn = hipPointerSetAttributeLayer;
  g_wrapper_tbl.hipProfilerStart_fn = hipProfilerStartLayer;
  g_wrapper_tbl.hipProfilerStop_fn = hipProfilerStopLayer;
  g_wrapper_tbl.hipRuntimeGetVersion_fn = hipRuntimeGetVersionLayer;
  g_wrapper_tbl.hipSetDevice_fn = hipSetDeviceLayer;
  g_wrapper_tbl.hipSetDeviceFlags_fn = hipSetDeviceFlagsLayer;
  g_wrapper_tbl.hipSetupArgument_fn = hipSetupArgumentLayer;
  g_wrapper_tbl.hipSignalExternalSemaphoresAsync_fn = hipSignalExternalSemaphoresAsyncLayer;
  g_wrapper_tbl.hipStreamAddCallback_fn = hipStreamAddCallbackLayer;
  g_wrapper_tbl.hipStreamAttachMemAsync_fn = hipStreamAttachMemAsyncLayer;
  g_wrapper_tbl.hipStreamBeginCapture_fn = hipStreamBeginCaptureLayer;
  g_wrapper_tbl.hipStreamCopyAttributes_fn = hipStreamCopyAttributesLayer;
  g_wrapper_tbl.hipStreamCreate_fn = hipStreamCreateLayer;
  g_wrapper_tbl.hipStreamCreateWithFlags_fn = hipStreamCreateWithFlagsLayer;
  g_wrapper_tbl.hipStreamCreateWithPriority_fn = hipStreamCreateWithPriorityLayer;
  g_wrapper_tbl.hipStreamDestroy_fn = hipStreamDestroyLayer;
  g_wrapper_tbl.hipStreamEndCapture_fn = hipStreamEndCaptureLayer;
  g_wrapper_tbl.hipStreamGetCaptureInfo_fn = hipStreamGetCaptureInfoLayer;
  g_wrapper_tbl.hipStreamGetCaptureInfo_v2_fn = hipStreamGetCaptureInfo_v2Layer;
  g_wrapper_tbl.hipStreamGetDevice_fn = hipStreamGetDeviceLayer;
  g_wrapper_tbl.hipStreamGetFlags_fn = hipStreamGetFlagsLayer;
  g_wrapper_tbl.hipStreamGetId_fn = hipStreamGetIdLayer;
  g_wrapper_tbl.hipStreamGetPriority_fn = hipStreamGetPriorityLayer;
  g_wrapper_tbl.hipStreamIsCapturing_fn = hipStreamIsCapturingLayer;
  g_wrapper_tbl.hipStreamQuery_fn = hipStreamQueryLayer;
  g_wrapper_tbl.hipStreamSynchronize_fn = hipStreamSynchronizeLayer;
  g_wrapper_tbl.hipStreamUpdateCaptureDependencies_fn = hipStreamUpdateCaptureDependenciesLayer;
  g_wrapper_tbl.hipStreamWaitEvent_fn = hipStreamWaitEventLayer;
  g_wrapper_tbl.hipStreamWaitValue32_fn = hipStreamWaitValue32Layer;
  g_wrapper_tbl.hipStreamWaitValue64_fn = hipStreamWaitValue64Layer;
  g_wrapper_tbl.hipStreamWriteValue32_fn = hipStreamWriteValue32Layer;
  g_wrapper_tbl.hipStreamWriteValue64_fn = hipStreamWriteValue64Layer;
  g_wrapper_tbl.hipStreamBatchMemOp_fn = hipStreamBatchMemOpLayer;
  g_wrapper_tbl.hipTexObjectCreate_fn = hipTexObjectCreateLayer;
  g_wrapper_tbl.hipTexObjectDestroy_fn = hipTexObjectDestroyLayer;
  g_wrapper_tbl.hipTexObjectGetResourceDesc_fn = hipTexObjectGetResourceDescLayer;
  g_wrapper_tbl.hipTexObjectGetResourceViewDesc_fn = hipTexObjectGetResourceViewDescLayer;
  g_wrapper_tbl.hipTexObjectGetTextureDesc_fn = hipTexObjectGetTextureDescLayer;
  g_wrapper_tbl.hipTexRefGetAddress_fn = hipTexRefGetAddressLayer;
  g_wrapper_tbl.hipTexRefGetAddressMode_fn = hipTexRefGetAddressModeLayer;
  g_wrapper_tbl.hipTexRefGetFilterMode_fn = hipTexRefGetFilterModeLayer;
  g_wrapper_tbl.hipTexRefGetFlags_fn = hipTexRefGetFlagsLayer;
  g_wrapper_tbl.hipTexRefGetFormat_fn = hipTexRefGetFormatLayer;
  g_wrapper_tbl.hipTexRefGetMaxAnisotropy_fn = hipTexRefGetMaxAnisotropyLayer;
  g_wrapper_tbl.hipTexRefGetMipMappedArray_fn = hipTexRefGetMipMappedArrayLayer;
  g_wrapper_tbl.hipTexRefGetMipmapFilterMode_fn = hipTexRefGetMipmapFilterModeLayer;
  g_wrapper_tbl.hipTexRefGetMipmapLevelBias_fn = hipTexRefGetMipmapLevelBiasLayer;
  g_wrapper_tbl.hipTexRefGetMipmapLevelClamp_fn = hipTexRefGetMipmapLevelClampLayer;
  g_wrapper_tbl.hipTexRefSetAddress_fn = hipTexRefSetAddressLayer;
  g_wrapper_tbl.hipTexRefSetAddress2D_fn = hipTexRefSetAddress2DLayer;
  g_wrapper_tbl.hipTexRefSetAddressMode_fn = hipTexRefSetAddressModeLayer;
  g_wrapper_tbl.hipTexRefSetArray_fn = hipTexRefSetArrayLayer;
  g_wrapper_tbl.hipTexRefSetBorderColor_fn = hipTexRefSetBorderColorLayer;
  g_wrapper_tbl.hipTexRefSetFilterMode_fn = hipTexRefSetFilterModeLayer;
  g_wrapper_tbl.hipTexRefSetFlags_fn = hipTexRefSetFlagsLayer;
  g_wrapper_tbl.hipTexRefSetFormat_fn = hipTexRefSetFormatLayer;
  g_wrapper_tbl.hipTexRefSetMaxAnisotropy_fn = hipTexRefSetMaxAnisotropyLayer;
  g_wrapper_tbl.hipTexRefSetMipmapFilterMode_fn = hipTexRefSetMipmapFilterModeLayer;
  g_wrapper_tbl.hipTexRefSetMipmapLevelBias_fn = hipTexRefSetMipmapLevelBiasLayer;
  g_wrapper_tbl.hipTexRefSetMipmapLevelClamp_fn = hipTexRefSetMipmapLevelClampLayer;
  g_wrapper_tbl.hipTexRefSetMipmappedArray_fn = hipTexRefSetMipmappedArrayLayer;
  g_wrapper_tbl.hipThreadExchangeStreamCaptureMode_fn = hipThreadExchangeStreamCaptureModeLayer;
  g_wrapper_tbl.hipUnbindTexture_fn = hipUnbindTextureLayer;
  g_wrapper_tbl.hipUserObjectCreate_fn = hipUserObjectCreateLayer;
  g_wrapper_tbl.hipUserObjectRelease_fn = hipUserObjectReleaseLayer;
  g_wrapper_tbl.hipUserObjectRetain_fn = hipUserObjectRetainLayer;
  g_wrapper_tbl.hipWaitExternalSemaphoresAsync_fn = hipWaitExternalSemaphoresAsyncLayer;
  g_wrapper_tbl.hipCreateChannelDesc_fn = hipCreateChannelDescLayer;
  g_wrapper_tbl.hipExtModuleLaunchKernel_fn = hipExtModuleLaunchKernelLayer;
  g_wrapper_tbl.hipHccModuleLaunchKernel_fn = hipHccModuleLaunchKernelLayer;
  g_wrapper_tbl.hipMemcpy_spt_fn = hipMemcpy_sptLayer;
  g_wrapper_tbl.hipMemcpyToSymbol_spt_fn = hipMemcpyToSymbol_sptLayer;
  g_wrapper_tbl.hipMemcpyFromSymbol_spt_fn = hipMemcpyFromSymbol_sptLayer;
  g_wrapper_tbl.hipMemcpy2D_spt_fn = hipMemcpy2D_sptLayer;
  g_wrapper_tbl.hipMemcpy2DFromArray_spt_fn = hipMemcpy2DFromArray_sptLayer;
  g_wrapper_tbl.hipMemcpy3D_spt_fn = hipMemcpy3D_sptLayer;
  g_wrapper_tbl.hipMemset_spt_fn = hipMemset_sptLayer;
  g_wrapper_tbl.hipMemsetAsync_spt_fn = hipMemsetAsync_sptLayer;
  g_wrapper_tbl.hipMemset2D_spt_fn = hipMemset2D_sptLayer;
  g_wrapper_tbl.hipMemset2DAsync_spt_fn = hipMemset2DAsync_sptLayer;
  g_wrapper_tbl.hipMemset3DAsync_spt_fn = hipMemset3DAsync_sptLayer;
  g_wrapper_tbl.hipMemset3D_spt_fn = hipMemset3D_sptLayer;
  g_wrapper_tbl.hipMemcpyAsync_spt_fn = hipMemcpyAsync_sptLayer;
  g_wrapper_tbl.hipMemcpy3DAsync_spt_fn = hipMemcpy3DAsync_sptLayer;
  g_wrapper_tbl.hipMemcpy2DAsync_spt_fn = hipMemcpy2DAsync_sptLayer;
  g_wrapper_tbl.hipMemcpyFromSymbolAsync_spt_fn = hipMemcpyFromSymbolAsync_sptLayer;
  g_wrapper_tbl.hipMemcpyToSymbolAsync_spt_fn = hipMemcpyToSymbolAsync_sptLayer;
  g_wrapper_tbl.hipMemcpyFromArray_spt_fn = hipMemcpyFromArray_sptLayer;
  g_wrapper_tbl.hipMemcpy2DToArray_spt_fn = hipMemcpy2DToArray_sptLayer;
  g_wrapper_tbl.hipMemcpy2DFromArrayAsync_spt_fn = hipMemcpy2DFromArrayAsync_sptLayer;
  g_wrapper_tbl.hipMemcpy2DToArrayAsync_spt_fn = hipMemcpy2DToArrayAsync_sptLayer;
  g_wrapper_tbl.hipStreamQuery_spt_fn = hipStreamQuery_sptLayer;
  g_wrapper_tbl.hipStreamSynchronize_spt_fn = hipStreamSynchronize_sptLayer;
  g_wrapper_tbl.hipStreamGetPriority_spt_fn = hipStreamGetPriority_sptLayer;
  g_wrapper_tbl.hipStreamWaitEvent_spt_fn = hipStreamWaitEvent_sptLayer;
  g_wrapper_tbl.hipStreamGetFlags_spt_fn = hipStreamGetFlags_sptLayer;
  g_wrapper_tbl.hipStreamAddCallback_spt_fn = hipStreamAddCallback_sptLayer;
  g_wrapper_tbl.hipEventRecord_spt_fn = hipEventRecord_sptLayer;
  g_wrapper_tbl.hipLaunchCooperativeKernel_spt_fn = hipLaunchCooperativeKernel_sptLayer;
  g_wrapper_tbl.hipLaunchKernel_spt_fn = hipLaunchKernel_sptLayer;
  g_wrapper_tbl.hipGraphLaunch_spt_fn = hipGraphLaunch_sptLayer;
  g_wrapper_tbl.hipStreamBeginCapture_spt_fn = hipStreamBeginCapture_sptLayer;
  g_wrapper_tbl.hipStreamEndCapture_spt_fn = hipStreamEndCapture_sptLayer;
  g_wrapper_tbl.hipStreamIsCapturing_spt_fn = hipStreamIsCapturing_sptLayer;
  g_wrapper_tbl.hipStreamGetCaptureInfo_spt_fn = hipStreamGetCaptureInfo_sptLayer;
  g_wrapper_tbl.hipStreamGetCaptureInfo_v2_spt_fn = hipStreamGetCaptureInfo_v2_sptLayer;
  g_wrapper_tbl.hipLaunchHostFunc_spt_fn = hipLaunchHostFunc_sptLayer;
  g_wrapper_tbl.hipGetStreamDeviceId_fn = hipGetStreamDeviceIdLayer;
  g_wrapper_tbl.hipDrvGraphAddMemsetNode_fn = hipDrvGraphAddMemsetNodeLayer;
  g_wrapper_tbl.hipGetDevicePropertiesR0000_fn = hipGetDevicePropertiesR0000Layer;
  g_wrapper_tbl.hipGetDriverEntryPoint_fn = hipGetDriverEntryPointLayer;
  g_wrapper_tbl.hipGetDriverEntryPoint_spt_fn = hipGetDriverEntryPoint_sptLayer;
  g_wrapper_tbl.hipExtGetLastError_fn = hipExtGetLastErrorLayer;
  g_wrapper_tbl.hipTexRefGetBorderColor_fn = hipTexRefGetBorderColorLayer;
  g_wrapper_tbl.hipTexRefGetArray_fn = hipTexRefGetArrayLayer;
  g_wrapper_tbl.hipGetProcAddress_fn = hipGetProcAddressLayer;
  g_wrapper_tbl.hipGetProcAddress_spt_fn = hipGetProcAddress_sptLayer;
  g_wrapper_tbl.hipStreamBeginCaptureToGraph_fn = hipStreamBeginCaptureToGraphLayer;
  g_wrapper_tbl.hipGetFuncBySymbol_fn = hipGetFuncBySymbolLayer;
  g_wrapper_tbl.hipSetValidDevices_fn = hipSetValidDevicesLayer;
  g_wrapper_tbl.hipMemcpyAtoD_fn = hipMemcpyAtoDLayer;
  g_wrapper_tbl.hipMemcpyDtoA_fn = hipMemcpyDtoALayer;
  g_wrapper_tbl.hipMemcpyAtoA_fn = hipMemcpyAtoALayer;
  g_wrapper_tbl.hipMemcpyAtoHAsync_fn = hipMemcpyAtoHAsyncLayer;
  g_wrapper_tbl.hipMemcpyHtoAAsync_fn = hipMemcpyHtoAAsyncLayer;
  g_wrapper_tbl.hipMemcpy2DArrayToArray_fn = hipMemcpy2DArrayToArrayLayer;
  g_wrapper_tbl.hipDrvGraphAddMemFreeNode_fn = hipDrvGraphAddMemFreeNodeLayer;
  g_wrapper_tbl.hipDrvGraphExecMemcpyNodeSetParams_fn = hipDrvGraphExecMemcpyNodeSetParamsLayer;
  g_wrapper_tbl.hipDrvGraphExecMemsetNodeSetParams_fn = hipDrvGraphExecMemsetNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecGetFlags_fn = hipGraphExecGetFlagsLayer;
  g_wrapper_tbl.hipGraphNodeSetParams_fn = hipGraphNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecNodeSetParams_fn = hipGraphExecNodeSetParamsLayer;
  g_wrapper_tbl.hipExternalMemoryGetMappedMipmappedArray_fn = hipExternalMemoryGetMappedMipmappedArrayLayer;
  g_wrapper_tbl.hipDrvGraphMemcpyNodeGetParams_fn = hipDrvGraphMemcpyNodeGetParamsLayer;
  g_wrapper_tbl.hipDrvGraphMemcpyNodeSetParams_fn = hipDrvGraphMemcpyNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphAddBatchMemOpNode_fn = hipGraphAddBatchMemOpNodeLayer;
  g_wrapper_tbl.hipGraphBatchMemOpNodeGetParams_fn = hipGraphBatchMemOpNodeGetParamsLayer;
  g_wrapper_tbl.hipGraphBatchMemOpNodeSetParams_fn = hipGraphBatchMemOpNodeSetParamsLayer;
  g_wrapper_tbl.hipGraphExecBatchMemOpNodeSetParams_fn = hipGraphExecBatchMemOpNodeSetParamsLayer;
  g_wrapper_tbl.hipEventRecordWithFlags_fn = hipEventRecordWithFlagsLayer;
  g_wrapper_tbl.hipLaunchKernelExC_fn = hipLaunchKernelExCLayer;
  g_wrapper_tbl.hipDrvLaunchKernelEx_fn = hipDrvLaunchKernelExLayer;
  g_wrapper_tbl.hipMemGetHandleForAddressRange_fn = hipMemGetHandleForAddressRangeLayer;
  g_wrapper_tbl.hipMemsetD2D8_fn = hipMemsetD2D8Layer;
  g_wrapper_tbl.hipMemsetD2D8Async_fn = hipMemsetD2D8AsyncLayer;
  g_wrapper_tbl.hipMemsetD2D16_fn = hipMemsetD2D16Layer;
  g_wrapper_tbl.hipMemsetD2D16Async_fn = hipMemsetD2D16AsyncLayer;
  g_wrapper_tbl.hipMemsetD2D32_fn = hipMemsetD2D32Layer;
  g_wrapper_tbl.hipMemsetD2D32Async_fn = hipMemsetD2D32AsyncLayer;
  g_wrapper_tbl.hipStreamGetAttribute_fn = hipStreamGetAttributeLayer;
  g_wrapper_tbl.hipStreamSetAttribute_fn = hipStreamSetAttributeLayer;
  g_wrapper_tbl.hipMemcpyBatchAsync_fn = hipMemcpyBatchAsyncLayer;
  g_wrapper_tbl.hipMemcpy3DBatchAsync_fn = hipMemcpy3DBatchAsyncLayer;
  g_wrapper_tbl.hipMemcpy3DPeer_fn = hipMemcpy3DPeerLayer;
  g_wrapper_tbl.hipMemcpy3DPeerAsync_fn = hipMemcpy3DPeerAsyncLayer;
  g_wrapper_tbl.hipLibraryLoadData_fn = hipLibraryLoadDataLayer;
  g_wrapper_tbl.hipLibraryLoadFromFile_fn = hipLibraryLoadFromFileLayer;
  g_wrapper_tbl.hipLibraryUnload_fn = hipLibraryUnloadLayer;
  g_wrapper_tbl.hipLibraryGetKernel_fn = hipLibraryGetKernelLayer;
  g_wrapper_tbl.hipLibraryGetKernelCount_fn = hipLibraryGetKernelCountLayer;
  g_wrapper_tbl.hipLibraryEnumerateKernels_fn = hipLibraryEnumerateKernelsLayer;
  g_wrapper_tbl.hipKernelGetLibrary_fn = hipKernelGetLibraryLayer;
  g_wrapper_tbl.hipKernelGetName_fn = hipKernelGetNameLayer;
  g_wrapper_tbl.hipOccupancyAvailableDynamicSMemPerBlock_fn = hipOccupancyAvailableDynamicSMemPerBlockLayer;
  g_wrapper_tbl.hipKernelGetParamInfo_fn = hipKernelGetParamInfoLayer;
  g_wrapper_tbl.hipExtDisableLogging_fn = hipExtDisableLoggingLayer;
  g_wrapper_tbl.hipExtEnableLogging_fn = hipExtEnableLoggingLayer;
  g_wrapper_tbl.hipExtSetLoggingParams_fn = hipExtSetLoggingParamsLayer;
  g_wrapper_tbl.hipMemSetMemPool_fn = hipMemSetMemPoolLayer;
  g_wrapper_tbl.hipMemGetMemPool_fn = hipMemGetMemPoolLayer;
  g_wrapper_tbl.hipKernelGetAttribute_fn = hipKernelGetAttributeLayer;
  g_wrapper_tbl.hipKernelSetAttribute_fn = hipKernelSetAttributeLayer;
  g_wrapper_tbl.hipKernelGetFunction_fn = hipKernelGetFunctionLayer;
  // g_wrapper_tbl is fully written before any Install call copies it in.
}

void HipProfilerInstallWrappersExt(HipDispatchTable* tbl) {
  if (g_wrapped.exchange(true)) return;
  // x86-64 guarantees 8-byte aligned stores are atomic, so individual pointer
  // reads by concurrent HIP callers are never torn — each thread sees either
  // the old function pointer or the new one, never a half-written value.
  std::memcpy(tbl, &g_wrapper_tbl, sizeof(HipDispatchTable));
}

void HipProfilerRemoveWrappersExt(HipDispatchTable* tbl) {
  if (!g_wrapped.exchange(false)) return;
  std::memcpy(tbl, &g_next, sizeof(HipDispatchTable));
}

void HipProfilerInstallCompilerWrappersExt(HipCompilerDispatchTable* tbl) {
  if (g_compiler_wrapped.exchange(true)) return;
  g_compiler_next = *tbl;
  g_compiler_wrapper_tbl = g_compiler_next;
  g_compiler_wrapper_tbl.__hipPushCallConfiguration_fn = __hipPushCallConfigurationLayer;
  g_compiler_wrapper_tbl.__hipPopCallConfiguration_fn  = __hipPopCallConfigurationLayer;
  std::memcpy(tbl, &g_compiler_wrapper_tbl, sizeof(HipCompilerDispatchTable));
}

void HipProfilerRemoveCompilerWrappersExt(HipCompilerDispatchTable* tbl) {
  if (!g_compiler_wrapped.exchange(false)) return;
  std::memcpy(tbl, &g_compiler_next, sizeof(HipCompilerDispatchTable));
}
