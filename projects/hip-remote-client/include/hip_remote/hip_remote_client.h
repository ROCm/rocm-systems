/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hip_remote_client.h
 * @brief Internal client API for remote HIP execution
 *
 * This header provides the internal API for managing connections to
 * the remote HIP worker service and sending/receiving protocol messages.
 */

#ifndef HIP_REMOTE_CLIENT_H
#define HIP_REMOTE_CLIENT_H

#include "hip_remote_protocol.h"
#include "hip_remote_platform.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * HIP Type Definitions
 *
 * These definitions are copied directly from the official HIP headers at:
 *   rocm-systems/projects/hip/include/hip/hip_runtime_api.h
 *
 * We cannot simply #include the HIP headers because:
 * 1. This library builds on macOS where HIP is not installed
 * 2. The HIP headers have many nested platform-specific dependencies
 *    (amd_detail/host_defines.h, driver_types.h, etc.)
 * 3. The remote client only needs the type definitions, not the full HIP API
 *
 * These values must stay synchronized with the HIP headers.
 * Last synced with: HIP 6.3 (ROCm 6.3)
 * ============================================================================ */

typedef enum {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorMemoryAllocation = 2,  /* Deprecated alias */
    hipErrorNotInitialized = 3,
    hipErrorInitializationError = 3,  /* Deprecated alias */
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    hipErrorInvalidPitchValue = 12,
    hipErrorInvalidSymbol = 13,
    hipErrorInvalidDevicePointer = 17,
    hipErrorInvalidMemcpyDirection = 21,
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorNoDevice = 100,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidImage = 200,
    hipErrorInvalidContext = 201,
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    hipErrorMapBufferObjectFailed = 205,  /* Deprecated alias */
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    hipErrorOperatingSystem = 304,
    hipErrorInvalidHandle = 400,
    hipErrorInvalidResourceHandle = 400,  /* Deprecated alias */
    hipErrorIllegalState = 401,
    hipErrorNotFound = 500,
    hipErrorNotReady = 600,
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,
    hipErrorLaunchTimeOut = 702,
    hipErrorPeerAccessAlreadyEnabled = 704,
    hipErrorPeerAccessNotEnabled = 705,
    hipErrorSetOnActiveProcess = 708,
    hipErrorContextIsDestroyed = 709,
    hipErrorAssert = 710,
    hipErrorHostMemoryAlreadyRegistered = 712,
    hipErrorHostMemoryNotRegistered = 713,
    hipErrorLaunchFailure = 719,
    hipErrorCooperativeLaunchTooLarge = 720,
    hipErrorNotSupported = 801,
    hipErrorStreamCaptureUnsupported = 900,
    hipErrorStreamCaptureInvalidated = 901,
    hipErrorStreamCaptureMerge = 902,
    hipErrorStreamCaptureUnmatched = 903,
    hipErrorStreamCaptureUnjoined = 904,
    hipErrorStreamCaptureIsolation = 905,
    hipErrorStreamCaptureImplicit = 906,
    hipErrorCapturedEvent = 907,
    hipErrorStreamCaptureWrongThread = 908,
    hipErrorGraphExecUpdateFailure = 910,
    hipErrorInvalidChannelDescriptor = 911,
    hipErrorInvalidTexture = 912,
    hipErrorUnknown = 999,
    hipErrorRuntimeMemory = 1052,
    hipErrorRuntimeOther = 1053,
    hipErrorTbd = 1054
} hipError_t;

/* Memory copy direction - from hip_runtime_api.h */
typedef enum {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

/* Memory type - from hip_runtime_api.h */
typedef enum {
    hipMemoryTypeUnregistered = 0,
    hipMemoryTypeHost = 1,
    hipMemoryTypeDevice = 2,
    hipMemoryTypeManaged = 3,
    hipMemoryTypeArray = 10,
    hipMemoryTypeUnified = 11
} hipMemoryType;

/* Pointer attributes - from hip_runtime_api.h */
typedef struct {
    hipMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags;
} hipPointerAttribute_t;

/* Opaque stream handle */
typedef void* hipStream_t;

/* Opaque event handle */
typedef void* hipEvent_t;

/* Opaque device handle */
typedef int hipDevice_t;

/* Opaque context handle [Deprecated] */
typedef struct ihipCtx_t* hipCtx_t;

/* UUID structure */
typedef struct {
    char bytes[16];
} hipUUID;

/* Function cache config */
typedef enum {
    hipFuncCachePreferNone = 0,
    hipFuncCachePreferShared = 1,
    hipFuncCachePreferL1 = 2,
    hipFuncCachePreferEqual = 3
} hipFuncCache_t;

/* Shared memory config */
typedef enum {
    hipSharedMemBankSizeDefault = 0,
    hipSharedMemBankSizeFourByte = 1,
    hipSharedMemBankSizeEightByte = 2
} hipSharedMemConfig;

/* P2P attributes */
typedef enum {
    hipDevP2PAttrPerformanceRank = 0,
    hipDevP2PAttrAccessSupported = 1,
    hipDevP2PAttrNativeAtomicSupported = 2,
    hipDevP2PAttrHipArrayAccessSupported = 3
} hipDeviceP2PAttr;

/* Stream flags */
#define hipStreamDefault        0x00
#define hipStreamNonBlocking    0x01

/* Event flags */
#define hipEventDefault         0x00
#define hipEventBlockingSync    0x01
#define hipEventDisableTiming   0x02
#define hipEventInterprocess    0x04

/* Host registration flags */
#define hipHostRegisterDefault  0x00
#define hipHostRegisterPortable 0x01
#define hipHostRegisterMapped   0x02
#define hipHostRegisterIoMemory 0x04
#define hipHostRegisterReadOnly 0x08

/* Memory advise values */
typedef enum {
    hipMemAdviseSetReadMostly = 1,
    hipMemAdviseUnsetReadMostly = 2,
    hipMemAdviseSetPreferredLocation = 3,
    hipMemAdviseUnsetPreferredLocation = 4,
    hipMemAdviseSetAccessedBy = 5,
    hipMemAdviseUnsetAccessedBy = 6
} hipMemoryAdvise;

/* Memory range attributes */
typedef enum {
    hipMemRangeAttributeReadMostly = 1,
    hipMemRangeAttributePreferredLocation = 2,
    hipMemRangeAttributeAccessedBy = 3,
    hipMemRangeAttributeLastPrefetchLocation = 4,
    hipMemRangeAttributeCoherencyMode = 100
} hipMemRangeAttribute;

/* Pointer attributes */
typedef enum {
    hipPointerAttributeContext = 1,
    hipPointerAttributeMemoryType = 2,
    hipPointerAttributeDevicePointer = 3,
    hipPointerAttributeHostPointer = 4,
    hipPointerAttributeDeviceOrdinal = 6,
    hipPointerAttributeSyncMemops = 7,
    hipPointerAttributeBufferId = 8,
    hipPointerAttributeIsManaged = 9,
    hipPointerAttributeMappedSize = 10,
    hipPointerAttributeAllocationFlags = 11,
    hipPointerAttributeRange = 12
} hipPointer_attribute;

/* ============================================================================
 * Client State
 * ============================================================================ */

/**
 * Client connection state
 */
typedef struct {
    hip_socket_t socket_fd;         /**< Socket file descriptor */
    hip_mutex_t lock;               /**< Mutex for thread safety */
    uint32_t next_request_id;       /**< Next request ID */
    bool connected;                 /**< Connection status */
    bool debug_enabled;             /**< Debug logging enabled */
    char worker_host[256];          /**< Worker hostname */
    int worker_port;                /**< Worker port */
    int connect_timeout_sec;        /**< Connection timeout (seconds) */
    int io_timeout_sec;             /**< I/O timeout (seconds) */
    hipError_t last_error;          /**< Last error code */
} HipRemoteClientState;

/**
 * Get the global client state.
 * Thread-safe after first call.
 */
HipRemoteClientState* hip_remote_get_client_state(void);

/* ============================================================================
 * Connection Management
 * ============================================================================ */

/**
 * Ensure client is connected to worker.
 * Automatically connects on first call or after disconnect.
 *
 * @return 0 on success, -1 on failure
 */
int hip_remote_ensure_connected(void);

/**
 * Disconnect from worker.
 */
void hip_remote_disconnect(void);

/**
 * Check if client is connected.
 */
bool hip_remote_is_connected(void);

/* ============================================================================
 * Message I/O
 * ============================================================================ */

/**
 * Send a request and receive a response (synchronous).
 *
 * @param op_code Operation code
 * @param request Request payload (may be NULL)
 * @param request_size Request payload size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size
);

/**
 * Send a request with inline data and receive response.
 *
 * @param op_code Operation code
 * @param request Request payload
 * @param request_size Request payload size
 * @param data Inline data to send
 * @param data_size Inline data size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request_with_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size,
    void* response,
    size_t response_size
);

/**
 * Send a request and receive response with inline data.
 *
 * @param op_code Operation code
 * @param request Request payload
 * @param request_size Request payload size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @param data_out Buffer for received inline data
 * @param data_size Size of data to receive
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request_receive_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size,
    void* data_out,
    size_t data_size
);

/**
 * Fire-and-forget request — sends the request but does NOT wait for a response.
 * Used for async GPU operations (kernel launches, memset, etc.) to eliminate
 * round-trip latency. Errors are deferred to the next sync operation.
 */
hipError_t hip_remote_request_fire_and_forget(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size
);

/**
 * Fire-and-forget request with inline data (e.g. H2D memcpy).
 */
hipError_t hip_remote_request_with_data_fire_and_forget(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size
);

/* ============================================================================
 * Logging
 * ============================================================================ */

/**
 * Log a debug message (only if debug enabled).
 */
void hip_remote_log_debug(const char* fmt, ...);

/**
 * Log an error message.
 */
void hip_remote_log_error(const char* fmt, ...);

/* ============================================================================
 * HIP API Functions
 * These functions implement the HIP runtime API by forwarding to the worker.
 * ============================================================================ */

/* Device Management */
hipError_t hipGetDeviceCount(int* count);
hipError_t hipSetDevice(int deviceId);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceProperties(void* prop, int deviceId);
hipError_t hipDeviceGetAttribute(int* value, int attr, int deviceId);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceReset(void);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipRuntimeGetVersion(int* runtimeVersion);

/* Device Limits */
typedef enum {
    hipLimitStackSize = 0x00,
    hipLimitPrintfFifoSize = 0x01,
    hipLimitMallocHeapSize = 0x02,
    hipLimitRange = 0x03
} hipLimit_t;

hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit);
hipError_t hipDeviceSetLimit(hipLimit_t limit, size_t value);

/* Peer Access */
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);

/* Device Driver APIs */
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device);

/* Device Cache/Config APIs */
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
hipError_t hipGetDeviceFlags(unsigned int* flags);
hipError_t hipSetDeviceFlags(unsigned int flags);
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice);
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
hipError_t hipSetValidDevices(int* device_arr, int len);
hipError_t hipChooseDevice(int* device, const void* prop);

/* Context Management [Deprecated] */
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
hipError_t hipCtxDestroy(hipCtx_t ctx);
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
hipError_t hipCtxGetCurrent(hipCtx_t* ctx);
hipError_t hipCtxPushCurrent(hipCtx_t ctx);
hipError_t hipCtxPopCurrent(hipCtx_t* ctx);
hipError_t hipCtxGetDevice(hipDevice_t* device);
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig);
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
hipError_t hipCtxSynchronize(void);
hipError_t hipCtxGetFlags(unsigned int* flags);
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);

/* Error Handling */
const char* hipGetErrorString(hipError_t error);
const char* hipGetErrorName(hipError_t error);
hipError_t hipGetLastError(void);
hipError_t hipPeekAtLastError(void);

/* Memory Management */
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMallocHost(void** ptr, size_t size);
hipError_t hipFreeHost(void* ptr);
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, void* stream);
hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoH(void* dst, const void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoD(void* dst, const void* src, size_t sizeBytes);
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, void* stream);
hipError_t hipMemGetInfo(size_t* free, size_t* total);
hipError_t hipMallocAsync(void** ptr, size_t size, void* stream);
hipError_t hipFreeAsync(void* ptr, void* stream);
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height, hipMemcpyKind kind, void* stream);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, const void* ptr);
hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t sizeBytes);
hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                              size_t sizeBytes, void* stream);

/* 3D Memory Copy */
typedef struct {
    size_t width;
    size_t height;
    size_t depth;
} hipExtent;

typedef struct {
    void* ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
} hipPitchedPtr;

typedef struct {
    size_t x;
    size_t y;
    size_t z;
} hipPos;

typedef struct {
    hipPitchedPtr srcPtr;
    hipPos srcPos;
    hipPitchedPtr dstPtr;
    hipPos dstPos;
    hipExtent extent;
    hipMemcpyKind kind;
} hipMemcpy3DParms;

hipError_t hipMemcpy3D(const hipMemcpy3DParms* p);
hipError_t hipMemcpy3DAsync(const hipMemcpy3DParms* p, void* stream);

/* Peer Memory Copy */
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes);
hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                               size_t sizeBytes, void* stream);

/* Host Memory Registration */
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
hipError_t hipHostUnregister(void* hostPtr);
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void* ptr);

/* Pitched Memory Allocation */
hipError_t hipMemAllocPitch(void** dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes);

/* Unified Memory Management */
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device);
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, void* stream);
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count);
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes, hipMemRangeAttribute* attributes,
                                    size_t num_attributes, const void* dev_ptr, size_t count);

/* Stream Management */
hipError_t hipStreamCreate(void** stream);
hipError_t hipStreamCreateWithFlags(void** stream, unsigned int flags);
hipError_t hipStreamCreateWithPriority(void** stream, unsigned int flags, int priority);
hipError_t hipStreamDestroy(void* stream);
hipError_t hipStreamSynchronize(void* stream);
hipError_t hipStreamWaitEvent(void* stream, void* event, unsigned int flags);
hipError_t hipStreamQuery(void* stream);
hipError_t hipStreamGetFlags(void* stream, unsigned int* flags);
hipError_t hipStreamGetPriority(void* stream, int* priority);
hipError_t hipStreamGetCaptureInfo(void* stream, int* captureStatus, unsigned long long* id);
hipError_t hipStreamUpdateCaptureDependencies(void* stream, void** dependencies,
                                              size_t numDependencies, unsigned int flags);

/* Event Management */
hipError_t hipEventCreate(void** event);
hipError_t hipEventCreateWithFlags(void** event, unsigned int flags);
hipError_t hipEventDestroy(void* event);
hipError_t hipEventRecord(void* event, void* stream);
hipError_t hipEventSynchronize(void* event);
hipError_t hipEventQuery(void* event);
hipError_t hipEventElapsedTime(float* ms, void* start, void* stop);

/* Module Management */
#if !(defined(HIP_REMOTE_USE_HIP_HEADERS) && HIP_REMOTE_USE_HIP_HEADERS)
/* Fallback definitions when HIP headers are not available */
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef enum {
    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock,
    hipJitOptionInfoLogBuffer,
    hipJitOptionInfoLogBufferSizeBytes,
    hipJitOptionErrorLogBuffer,
    hipJitOptionErrorLogBufferSizeBytes,
    hipJitOptionOptimizationLevel,
    hipJitOptionTargetFromContext,
    hipJitOptionTarget,
    hipJitOptionFallbackStrategy,
    hipJitOptionGenerateDebugInfo,
    hipJitOptionLogVerbose,
    hipJitOptionGenerateLineInfo,
    hipJitOptionCacheMode,
    hipJitOptionNumOptions
} hipJitOption;

/* Kernel Launch */
typedef struct {
    unsigned int x, y, z;
} dim3;

/* Function launch parameters for multi-device cooperative launch */
typedef struct {
    hipFunction_t function;
    dim3 gridDim;
    dim3 blockDim;
    void** kernelParams;
    unsigned int sharedMemBytes;
    hipStream_t stream;
} hipFunctionLaunchParams;

#endif /* !HIP_REMOTE_USE_HIP_HEADERS */

hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                unsigned int numOptions, hipJitOption* options,
                                void** optionValues);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                 const char* kname);

/**
 * Launch a kernel function.
 *
 * The remote HIP client queries kernel metadata from the worker to determine
 * the number of arguments (requires ROCm 7.2+ on the worker). For older ROCm
 * versions, the kernelParams array must be NULL-terminated:
 *
 *     void* args[] = { &d_a, &d_b, &d_c, &N, NULL };
 *     hipModuleLaunchKernel(function, gridX, 1, 1, blockX, 1, 1, 0, stream, args, NULL);
 *
 * Note: All arguments are currently assumed to be pointer-sized (8 bytes).
 * The 'extra' parameter is not supported in remote mode.
 */
hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                  unsigned int gridDimX,
                                  unsigned int gridDimY,
                                  unsigned int gridDimZ,
                                  unsigned int blockDimX,
                                  unsigned int blockDimY,
                                  unsigned int blockDimZ,
                                  unsigned int sharedMemBytes,
                                  hipStream_t stream,
                                  void** kernelParams,
                                  void** extra);

hipError_t hipLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                 int numDevices,
                                                 unsigned int flags);

hipError_t hipLaunchKernel(const void* function_address,
                            dim3 numBlocks,
                            dim3 dimBlocks,
                            void** args,
                            size_t sharedMemBytes,
                            hipStream_t stream);

hipError_t hipLaunchCooperativeKernel(const void* f,
                                       dim3 gridDim,
                                       dim3 blockDim,
                                       void** kernelParams,
                                       unsigned int sharedMemBytes,
                                       hipStream_t stream);

/* Function Attributes */
typedef struct {
    int sharedSizeBytes;
    int constSizeBytes;
    int localSizeBytes;
    int numRegs;
    int maxThreadsPerBlock;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
} hipFuncAttributes;

typedef enum {
    hipFuncAttributeMaxDynamicSharedMemorySize = 8,
    hipFuncAttributePreferredSharedMemoryCarveout = 9,
    hipFuncAttributeMax
} hipFunction_attribute;

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func);
hipError_t hipFuncSetAttribute(const void* func, hipFunction_attribute attr, int value);
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig);

/* Occupancy */
hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                              hipFunction_t f, size_t dynSharedMemPerBlk,
                                              int blockSizeLimit);
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f,
                                                         int blockSize, size_t dynSharedMemPerBlk);

/* Graph APIs */
typedef void* hipGraph_t;
typedef void* hipGraphExec_t;
typedef void* hipGraphNode_t;

typedef enum {
    hipStreamCaptureStatusNone = 0,
    hipStreamCaptureStatusActive = 1,
    hipStreamCaptureStatusInvalidated = 2
} hipStreamCaptureStatus;

typedef enum {
    hipStreamCaptureModeGlobal = 0,
    hipStreamCaptureModeThreadLocal = 1,
    hipStreamCaptureModeRelaxed = 2
} hipStreamCaptureMode;

typedef enum {
    hipGraphExecUpdateSuccess = 0,
    hipGraphExecUpdateError = 1,
    hipGraphExecUpdateErrorTopologyChanged = 2,
    hipGraphExecUpdateErrorNodeTypeChanged = 3,
    hipGraphExecUpdateErrorFunctionChanged = 4,
    hipGraphExecUpdateErrorParametersChanged = 5,
    hipGraphExecUpdateErrorNotSupported = 6,
    hipGraphExecUpdateErrorUnsupportedFunctionChange = 7
} hipGraphExecUpdateResult;

hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);
hipError_t hipGraphDestroy(hipGraph_t graph);
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                hipGraphNode_t* pErrorNode, char* pLogBuffer,
                                size_t bufferSize);
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);

/* Graph Node Types */
typedef enum {
    hipGraphNodeTypeKernel = 0,
    hipGraphNodeTypeMemcpy = 1,
    hipGraphNodeTypeMemset = 2,
    hipGraphNodeTypeHost = 3,
    hipGraphNodeTypeGraph = 4,
    hipGraphNodeTypeEmpty = 5,
    hipGraphNodeTypeWaitEvent = 6,
    hipGraphNodeTypeEventRecord = 7,
    hipGraphNodeTypeExtSemaphoreSignal = 8,
    hipGraphNodeTypeExtSemaphoreWait = 9,
    hipGraphNodeTypeMemAlloc = 10,
    hipGraphNodeTypeMemFree = 11,
    hipGraphNodeTypeCount = 12
} hipGraphNodeType;

/* Memset parameters for graph nodes */
typedef struct {
    void* dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
} hipMemsetParams;

/* Kernel node parameters */
typedef struct {
    dim3 blockDim;
    void** extra;
    void* func;
    dim3 gridDim;
    void** kernelParams;
    unsigned int sharedMemBytes;
} hipKernelNodeParams;

/* Host node parameters (limited support in remote mode) */
typedef struct {
    void* fn;
    void* userData;
} hipHostNodeParams;

/**
 * Add a memcpy node to a graph.
 */
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemcpy3DParms* pCopyParams);

/**
 * Add a 1D memcpy node to a graph.
 */
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    void* dst, const void* src, size_t count, hipMemcpyKind kind);

/**
 * Add a memset node to a graph.
 */
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemsetParams* pMemsetParams);

/**
 * Add a kernel launch node to a graph.
 */
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipKernelNodeParams* pNodeParams);

/**
 * Add dependencies between nodes.
 */
hipError_t hipGraphAddDependencies(hipGraph_t graph,
                                    const hipGraphNode_t* from,
                                    const hipGraphNode_t* to,
                                    size_t numDependencies);

/**
 * Add an empty node to a graph.
 */
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies);

/**
 * Get all nodes in a graph.
 */
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);

/**
 * Get root nodes of a graph.
 */
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes, size_t* pNumRootNodes);

/**
 * Get edges of a graph.
 */
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to, size_t* numEdges);

/**
 * Get type of a graph node.
 */
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType);

/**
 * Destroy a graph node.
 */
hipError_t hipGraphDestroyNode(hipGraphNode_t node);

/**
 * Clone an existing graph.
 */
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph);

/**
 * Get dependency nodes of a graph node.
 */
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                        size_t* pNumDependencies);

/**
 * Get dependent nodes of a graph node (nodes that depend on this node).
 */
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                          size_t* pNumDependentNodes);

/**
 * Update an instantiated graph with new parameters from the original graph.
 */
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                               hipGraphNode_t* hErrorNode_out,
                               hipGraphExecUpdateResult* updateResult_out);

/**
 * Update kernel node parameters in an executable graph.
 */
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                            const hipKernelNodeParams* pNodeParams);

/**
 * Add an event record node to a graph.
 */
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                       const hipGraphNode_t* pDependencies, size_t numDependencies,
                                       hipEvent_t event);

/**
 * Add an event wait node to a graph.
 */
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipEvent_t event);

/* Stream Callback (Note: callbacks execute on worker, not client) */
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback,
                                 void* userData, unsigned int flags);

/* ============================================================================
 * IPC (Inter-Process Communication) APIs
 * These enable sharing GPU memory and events between processes.
 * ============================================================================ */

/** IPC memory handle - 64 bytes, matches CUDA/HIP */
typedef struct {
    char reserved[64];
} hipIpcMemHandle_t;

/** IPC event handle - 64 bytes, matches CUDA/HIP */
typedef struct {
    char reserved[64];
} hipIpcEventHandle_t;

/** Flag for hipIpcOpenMemHandle */
#define hipIpcMemLazyEnablePeerAccess 0x01

/**
 * Get an IPC handle for an existing device memory allocation.
 * The handle can be passed to other processes to enable sharing.
 *
 * @param handle Pointer to receive the IPC handle
 * @param devPtr Device pointer allocated via hipMalloc
 * @return hipSuccess on success
 */
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);

/**
 * Open an IPC memory handle from another process.
 * Returns a device pointer valid in this process.
 *
 * @param devPtr Pointer to receive the device pointer
 * @param handle IPC handle from hipIpcGetMemHandle
 * @param flags Flags (hipIpcMemLazyEnablePeerAccess)
 * @return hipSuccess on success
 */
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);

/**
 * Close an IPC memory handle opened with hipIpcOpenMemHandle.
 *
 * @param devPtr Device pointer from hipIpcOpenMemHandle
 * @return hipSuccess on success
 */
hipError_t hipIpcCloseMemHandle(void* devPtr);

/**
 * Get an IPC handle for an event.
 * The event must have been created with hipEventInterprocess flag.
 *
 * @param handle Pointer to receive the IPC handle
 * @param event Event handle
 * @return hipSuccess on success
 */
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);

/**
 * Open an IPC event handle from another process.
 *
 * @param event Pointer to receive the event handle
 * @param handle IPC handle from hipIpcGetEventHandle
 * @return hipSuccess on success
 */
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);

/* ============================================================================
 * Memory Pool APIs
 * High-performance memory allocation with pooling.
 * ============================================================================ */

/** Memory pool handle */
typedef void* hipMemPool_t;

/** Memory allocation type */
typedef enum {
    hipMemAllocationTypeInvalid = 0,
    hipMemAllocationTypePinned = 1,
    hipMemAllocationTypeMax = 2
} hipMemAllocationType;

/** Memory location type */
typedef enum {
    hipMemLocationTypeInvalid = 0,
    hipMemLocationTypeDevice = 1,
    hipMemLocationTypeMax = 2
} hipMemLocationType;

/** Memory handle type for sharing */
typedef enum {
    hipMemHandleTypeNone = 0,
    hipMemHandleTypePosixFileDescriptor = 1,
    hipMemHandleTypeWin32 = 2,
    hipMemHandleTypeWin32Kmt = 4
} hipMemHandleType;

/** Memory pool attributes */
typedef enum {
    hipMemPoolReuseFollowEventDependencies = 1,
    hipMemPoolReuseAllowOpportunistic = 2,
    hipMemPoolReuseAllowInternalDependencies = 3,
    hipMemPoolAttrReleaseThreshold = 4,
    hipMemPoolAttrReservedMemCurrent = 5,
    hipMemPoolAttrReservedMemHigh = 6,
    hipMemPoolAttrUsedMemCurrent = 7,
    hipMemPoolAttrUsedMemHigh = 8
} hipMemPoolAttr;

/** Memory location descriptor */
typedef struct {
    hipMemLocationType type;
    int id;
} hipMemLocation;

/** Memory pool properties */
typedef struct {
    hipMemAllocationType allocType;
    hipMemHandleType handleTypes;
    hipMemLocation location;
    void* win32SecurityAttributes;
    size_t maxSize;
    unsigned char reserved[56];
} hipMemPoolProps;

/**
 * Create a memory pool.
 *
 * @param memPool Pointer to receive the memory pool handle
 * @param poolProps Pool properties
 * @return hipSuccess on success
 */
hipError_t hipMemPoolCreate(hipMemPool_t* memPool, const hipMemPoolProps* poolProps);

/**
 * Destroy a memory pool.
 *
 * @param memPool Memory pool to destroy
 * @return hipSuccess on success
 */
hipError_t hipMemPoolDestroy(hipMemPool_t memPool);

/**
 * Set a memory pool attribute.
 *
 * @param memPool Memory pool
 * @param attr Attribute to set
 * @param value Pointer to attribute value
 * @return hipSuccess on success
 */
hipError_t hipMemPoolSetAttribute(hipMemPool_t memPool, hipMemPoolAttr attr, void* value);

/**
 * Get a memory pool attribute.
 *
 * @param memPool Memory pool
 * @param attr Attribute to get
 * @param value Pointer to receive attribute value
 * @return hipSuccess on success
 */
hipError_t hipMemPoolGetAttribute(hipMemPool_t memPool, hipMemPoolAttr attr, void* value);

/**
 * Allocate memory from a pool asynchronously.
 *
 * @param devPtr Pointer to receive device pointer
 * @param size Allocation size in bytes
 * @param memPool Memory pool to allocate from
 * @param stream Stream for the async operation
 * @return hipSuccess on success
 */
hipError_t hipMallocFromPoolAsync(void** devPtr, size_t size, hipMemPool_t memPool, hipStream_t stream);

/**
 * Trim a memory pool to a minimum size.
 *
 * @param memPool Memory pool
 * @param minBytesToKeep Minimum bytes to retain
 * @return hipSuccess on success
 */
hipError_t hipMemPoolTrimTo(hipMemPool_t memPool, size_t minBytesToKeep);

/**
 * Get the default memory pool for a device.
 *
 * @param memPool Pointer to receive the memory pool handle
 * @param device Device ID
 * @return hipSuccess on success
 */
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* memPool, int device);

/**
 * Set the current memory pool for a device.
 *
 * @param device Device ID
 * @param memPool Memory pool to set
 * @return hipSuccess on success
 */
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t memPool);

/**
 * Get the current memory pool for a device.
 *
 * @param memPool Pointer to receive the memory pool handle
 * @param device Device ID
 * @return hipSuccess on success
 */
hipError_t hipDeviceGetMemPool(hipMemPool_t* memPool, int device);

#ifdef __cplusplus
}
#endif

#endif /* HIP_REMOTE_CLIENT_H */
