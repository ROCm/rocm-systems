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

/* Stream flags */
#define hipStreamDefault        0x00
#define hipStreamNonBlocking    0x01

/* Event flags */
#define hipEventDefault         0x00
#define hipEventBlockingSync    0x01
#define hipEventDisableTiming   0x02
#define hipEventInterprocess    0x04

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

/**
 * Flush the client-side write buffer. Must be called before any synchronous
 * operation that reads from the GPU (D2H memcpy, event query, synchronize).
 * Automatically called by hip_remote_request() and related synchronous calls.
 */
void hip_remote_flush(void);

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

/* Batched allocation: allocate N device buffers in a single round-trip */
hipError_t hipMallocBatch(void** ptrs, const size_t* sizes, uint32_t count);

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

/* Stream Callback (Note: callbacks execute on worker, not client) */
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback,
                                 void* userData, unsigned int flags);

#ifdef __cplusplus
}
#endif

#endif /* HIP_REMOTE_CLIENT_H */
