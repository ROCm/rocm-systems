/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "error_handling_common.hh"

const char* ErrorName(hipError_t enumerator) {
  switch (enumerator) {
#if HT_AMD
    case hipSuccess:
      return "hipSuccess";
    case hipErrorInvalidValue:
      return "hipErrorInvalidValue";
    case hipErrorOutOfMemory:
      return "hipErrorOutOfMemory";
    case hipErrorNotInitialized:
      return "hipErrorNotInitialized";
    case hipErrorDeinitialized:
      return "hipErrorDeinitialized";
    case hipErrorProfilerDisabled:
      return "hipErrorProfilerDisabled";
    case hipErrorProfilerNotInitialized:
      return "hipErrorProfilerNotInitialized";
    case hipErrorProfilerAlreadyStarted:
      return "hipErrorProfilerAlreadyStarted";
    case hipErrorProfilerAlreadyStopped:
      return "hipErrorProfilerAlreadyStopped";
    case hipErrorInvalidConfiguration:
      return "hipErrorInvalidConfiguration";
    case hipErrorInvalidSymbol:
      return "hipErrorInvalidSymbol";
    case hipErrorInvalidDevicePointer:
      return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection:
      return "hipErrorInvalidMemcpyDirection";
    case hipErrorInsufficientDriver:
      return "hipErrorInsufficientDriver";
    case hipErrorMissingConfiguration:
      return "hipErrorMissingConfiguration";
    case hipErrorPriorLaunchFailure:
      return "hipErrorPriorLaunchFailure";
    case hipErrorInvalidDeviceFunction:
      return "hipErrorInvalidDeviceFunction";
    case hipErrorNoDevice:
      return "hipErrorNoDevice";
    case hipErrorInvalidDevice:
      return "hipErrorInvalidDevice";
    case hipErrorInvalidPitchValue:
      return "hipErrorInvalidPitchValue";
    case hipErrorInvalidImage:
      return "hipErrorInvalidImage";
    case hipErrorInvalidContext:
      return "hipErrorInvalidContext";
    case hipErrorContextAlreadyCurrent:
      return "hipErrorContextAlreadyCurrent";
    case hipErrorMapFailed:
      return "hipErrorMapFailed";
    case hipErrorUnmapFailed:
      return "hipErrorUnmapFailed";
    case hipErrorArrayIsMapped:
      return "hipErrorArrayIsMapped";
    case hipErrorAlreadyMapped:
      return "hipErrorAlreadyMapped";
    case hipErrorNoBinaryForGpu:
      return "hipErrorNoBinaryForGpu";
    case hipErrorAlreadyAcquired:
      return "hipErrorAlreadyAcquired";
    case hipErrorNotMapped:
      return "hipErrorNotMapped";
    case hipErrorNotMappedAsArray:
      return "hipErrorNotMappedAsArray";
    case hipErrorNotMappedAsPointer:
      return "hipErrorNotMappedAsPointer";
    case hipErrorECCNotCorrectable:
      return "hipErrorECCNotCorrectable";
    case hipErrorUnsupportedLimit:
      return "hipErrorUnsupportedLimit";
    case hipErrorContextAlreadyInUse:
      return "hipErrorContextAlreadyInUse";
    case hipErrorPeerAccessUnsupported:
      return "hipErrorPeerAccessUnsupported";
    case hipErrorInvalidKernelFile:
      return "hipErrorInvalidKernelFile";
    case hipErrorInvalidGraphicsContext:
      return "hipErrorInvalidGraphicsContext";
    case hipErrorInvalidSource:
      return "hipErrorInvalidSource";
    case hipErrorFileNotFound:
      return "hipErrorFileNotFound";
    case hipErrorSharedObjectSymbolNotFound:
      return "hipErrorSharedObjectSymbolNotFound";
    case hipErrorSharedObjectInitFailed:
      return "hipErrorSharedObjectInitFailed";
    case hipErrorOperatingSystem:
      return "hipErrorOperatingSystem";
    case hipErrorInvalidHandle:
      return "hipErrorInvalidHandle";
    case hipErrorIllegalState:
      return "hipErrorIllegalState";
    case hipErrorNotFound:
      return "hipErrorNotFound";
    case hipErrorNotReady:
      return "hipErrorNotReady";
    case hipErrorIllegalAddress:
      return "hipErrorIllegalAddress";
    case hipErrorLaunchOutOfResources:
      return "hipErrorLaunchOutOfResources";
    case hipErrorLaunchTimeOut:
      return "hipErrorLaunchTimeOut";
    case hipErrorPeerAccessAlreadyEnabled:
      return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled:
      return "hipErrorPeerAccessNotEnabled";
    case hipErrorSetOnActiveProcess:
      return "hipErrorSetOnActiveProcess";
    case hipErrorContextIsDestroyed:
      return "hipErrorContextIsDestroyed";
    case hipErrorAssert:
      return "hipErrorAssert";
    case hipErrorHostMemoryAlreadyRegistered:
      return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
      return "hipErrorHostMemoryNotRegistered";
    case hipErrorLaunchFailure:
      return "hipErrorLaunchFailure";
    case hipErrorNotSupported:
      return "hipErrorNotSupported";
    case hipErrorUnknown:
      return "hipErrorUnknown";
    case hipErrorRuntimeMemory:
      return "hipErrorRuntimeMemory";
    case hipErrorRuntimeOther:
      return "hipErrorRuntimeOther";
    case hipErrorCooperativeLaunchTooLarge:
      return "hipErrorCooperativeLaunchTooLarge";
    case hipErrorStreamCaptureUnsupported:
      return "hipErrorStreamCaptureUnsupported";
    case hipErrorStreamCaptureInvalidated:
      return "hipErrorStreamCaptureInvalidated";
    case hipErrorStreamCaptureMerge:
      return "hipErrorStreamCaptureMerge";
    case hipErrorStreamCaptureUnmatched:
      return "hipErrorStreamCaptureUnmatched";
    case hipErrorStreamCaptureUnjoined:
      return "hipErrorStreamCaptureUnjoined";
    case hipErrorStreamCaptureIsolation:
      return "hipErrorStreamCaptureIsolation";
    case hipErrorStreamCaptureImplicit:
      return "hipErrorStreamCaptureImplicit";
    case hipErrorCapturedEvent:
      return "hipErrorCapturedEvent";
    case hipErrorStreamCaptureWrongThread:
      return "hipErrorStreamCaptureWrongThread";
    case hipErrorGraphExecUpdateFailure:
      return "hipErrorGraphExecUpdateFailure";
    case hipErrorTbd:
      return "hipErrorTbd";
    default:
      return "hipErrorUnknown";
#else
    case hipSuccess:
      return "CUDA_SUCCESS";
    case hipErrorInvalidValue:
      return "CUDA_ERROR_INVALID_VALUE";
    case hipErrorOutOfMemory:
      return "CUDA_ERROR_OUT_OF_MEMORY";
    case hipErrorNotInitialized:
      return "CUDA_ERROR_NOT_INITIALIZED";
    case hipErrorDeinitialized:
      return "CUDA_ERROR_DEINITIALIZED";
    case hipErrorProfilerDisabled:
      return "CUDA_ERROR_PROFILER_DISABLED";
    case hipErrorProfilerNotInitialized:
      return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case hipErrorProfilerAlreadyStarted:
      return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case hipErrorProfilerAlreadyStopped:
      return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    case hipErrorInvalidConfiguration:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidSymbol:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidDevicePointer:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidMemcpyDirection:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorInsufficientDriver:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorMissingConfiguration:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorPriorLaunchFailure:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidDeviceFunction:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorNoDevice:
      return "CUDA_ERROR_NO_DEVICE";
    case hipErrorInvalidDevice:
      return "CUDA_ERROR_INVALID_DEVICE";
    case hipErrorInvalidPitchValue:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidImage:
      return "CUDA_ERROR_INVALID_IMAGE";
    case hipErrorInvalidContext:
      return "CUDA_ERROR_INVALID_CONTEXT";
    case hipErrorContextAlreadyCurrent:
      return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case hipErrorMapFailed:
      return "CUDA_ERROR_MAP_FAILED";
    case hipErrorUnmapFailed:
      return "CUDA_ERROR_UNMAP_FAILED";
    case hipErrorArrayIsMapped:
      return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case hipErrorAlreadyMapped:
      return "CUDA_ERROR_ALREADY_MAPPED";
    case hipErrorNoBinaryForGpu:
      return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case hipErrorAlreadyAcquired:
      return "CUDA_ERROR_ALREADY_ACQUIRED";
    case hipErrorNotMapped:
      return "CUDA_ERROR_NOT_MAPPED";
    case hipErrorNotMappedAsArray:
      return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case hipErrorNotMappedAsPointer:
      return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case hipErrorECCNotCorrectable:
      return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case hipErrorUnsupportedLimit:
      return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case hipErrorContextAlreadyInUse:
      return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case hipErrorPeerAccessUnsupported:
      return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case hipErrorInvalidKernelFile:
      return "CUDA_ERROR_INVALID_PTX";
    case hipErrorInvalidGraphicsContext:
      return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    case hipErrorInvalidSource:
      return "CUDA_ERROR_INVALID_SOURCE";
    case hipErrorFileNotFound:
      return "CUDA_ERROR_FILE_NOT_FOUND";
    case hipErrorSharedObjectSymbolNotFound:
      return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case hipErrorSharedObjectInitFailed:
      return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case hipErrorOperatingSystem:
      return "CUDA_ERROR_OPERATING_SYSTEM";
    case hipErrorInvalidHandle:
      return "CUDA_ERROR_INVALID_HANDLE";
    case hipErrorIllegalState:
      return "CUDA_ERROR_ILLEGAL_STATE";
    case hipErrorNotFound:
      return "CUDA_ERROR_NOT_FOUND";
    case hipErrorNotReady:
      return "CUDA_ERROR_NOT_READY";
    case hipErrorIllegalAddress:
      return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case hipErrorLaunchOutOfResources:
      return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case hipErrorLaunchTimeOut:
      return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case hipErrorPeerAccessAlreadyEnabled:
      return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case hipErrorPeerAccessNotEnabled:
      return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case hipErrorSetOnActiveProcess:
      return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case hipErrorContextIsDestroyed:
      return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    case hipErrorAssert:
      return "CUDA_ERROR_ASSERT";
    case hipErrorHostMemoryAlreadyRegistered:
      return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case hipErrorHostMemoryNotRegistered:
      return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    case hipErrorLaunchFailure:
      return "CUDA_ERROR_LAUNCH_FAILED";
    case hipErrorNotSupported:
      return "CUDA_ERROR_NOT_SUPPORTED";
    case hipErrorUnknown:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorRuntimeMemory:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorRuntimeOther:
      return "CUDA_ERROR_UNKNOWN";
    case hipErrorCooperativeLaunchTooLarge:
      return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
    case hipErrorStreamCaptureUnsupported:
      return "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
    case hipErrorStreamCaptureInvalidated:
      return "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
    case hipErrorStreamCaptureMerge:
      return "CUDA_ERROR_STREAM_CAPTURE_MERGE";
    case hipErrorStreamCaptureUnmatched:
      return "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";
    case hipErrorStreamCaptureUnjoined:
      return "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";
    case hipErrorStreamCaptureIsolation:
      return "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";
    case hipErrorStreamCaptureImplicit:
      return "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";
    case hipErrorCapturedEvent:
      return "CUDA_ERROR_CAPTURED_EVENT";
    case hipErrorStreamCaptureWrongThread:
      return "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";
    case hipErrorGraphExecUpdateFailure:
      return "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
    default:
      return "CUDA_ERROR_UNKNOWN";
#endif
  }
}

const char* ErrorString(hipError_t enumerator) {
  switch (enumerator) {
    case hipSuccess:
      return "no error";
    case hipErrorInvalidValue:
      return "invalid argument";
    case hipErrorOutOfMemory:
      return "out of memory";
    case hipErrorNotInitialized:
      return "initialization error";
    case hipErrorDeinitialized:
      return "driver shutting down";
    case hipErrorProfilerDisabled:
      return "profiler disabled while using external profiling tool";
    case hipErrorProfilerNotInitialized:
#if HT_AMD
      return "profiler is not initialized";
#elif HT_NVIDIA
      return "profiler not initialized: call cudaProfilerInitialize()";
#endif
    case hipErrorProfilerAlreadyStarted:
      return "profiler already started";
    case hipErrorProfilerAlreadyStopped:
      return "profiler already stopped";
    case hipErrorInvalidConfiguration:
#if HT_AMD
      return "invalid configuration argument";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorInvalidPitchValue:
#if HT_AMD
      return "invalid pitch argument";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorInvalidSymbol:
#if HT_AMD
      return "invalid device symbol";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorInvalidDevicePointer:
#if HT_AMD
      return "invalid device pointer";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorInvalidMemcpyDirection:
#if HT_AMD
      return "invalid copy direction for memcpy";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorInsufficientDriver:
#if HT_AMD
      return "driver version is insufficient for runtime version";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorMissingConfiguration:
#if HT_AMD
      return "__global__ function call is not configured";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorPriorLaunchFailure:
#if HT_AMD
      return "unspecified launch failure in prior launch";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorInvalidDeviceFunction:
#if HT_AMD
      return "invalid device function";
#elif HT_NVIDIA
      return "unknown error";
#endif
    case hipErrorNoDevice:
#if HT_AMD
      return "no ROCm-capable device is detected";
#elif HT_NVIDIA
      return "no CUDA-capable device is detected";
#endif
    case hipErrorInvalidDevice:
      return "invalid device ordinal";
    case hipErrorInvalidImage:
      return "device kernel image is invalid";
    case hipErrorInvalidContext:
      return "invalid device context";
    case hipErrorContextAlreadyCurrent:
#if HT_AMD
      return "context is already current context";
#elif HT_NVIDIA
      return "context already current";
#endif
    case hipErrorMapFailed:
      return "mapping of buffer object failed";
    case hipErrorUnmapFailed:
      return "unmapping of buffer object failed";
    case hipErrorArrayIsMapped:
      return "array is mapped";
    case hipErrorAlreadyMapped:
      return "resource already mapped";
    case hipErrorNoBinaryForGpu:
      return "no kernel image is available for execution on the device";
    case hipErrorAlreadyAcquired:
      return "resource already acquired";
    case hipErrorNotMapped:
      return "resource not mapped";
    case hipErrorNotMappedAsArray:
      return "resource not mapped as array";
    case hipErrorNotMappedAsPointer:
      return "resource not mapped as pointer";
    case hipErrorECCNotCorrectable:
      return "uncorrectable ECC error encountered";
    case hipErrorUnsupportedLimit:
      return "limit is not supported on this architecture";
    case hipErrorContextAlreadyInUse:
      return "exclusive-thread device already in use by a different thread";
    case hipErrorPeerAccessUnsupported:
      return "peer access is not supported between these two devices";
    case hipErrorInvalidKernelFile:
#if HT_AMD
      return "invalid kernel file";
#elif HT_NVIDIA
      return "a PTX JIT compilation failed";
#endif
    case hipErrorInvalidGraphicsContext:
      return "invalid OpenGL or DirectX context";
    case hipErrorInvalidSource:
      return "device kernel image is invalid";
    case hipErrorFileNotFound:
      return "file not found";
    case hipErrorSharedObjectSymbolNotFound:
      return "shared object symbol not found";
    case hipErrorSharedObjectInitFailed:
      return "shared object initialization failed";
    case hipErrorOperatingSystem:
      return "OS call failed or operation not supported on this OS";
    case hipErrorInvalidHandle:
      return "invalid resource handle";
    case hipErrorIllegalState:
      return "the operation cannot be performed in the present state";
    case hipErrorNotFound:
      return "named symbol not found";
    case hipErrorNotReady:
      return "device not ready";
    case hipErrorIllegalAddress:
      return "an illegal memory access was encountered";
    case hipErrorLaunchOutOfResources:
      return "too many resources requested for launch";
    case hipErrorLaunchTimeOut:
      return "the launch timed out and was terminated";
    case hipErrorPeerAccessAlreadyEnabled:
      return "peer access is already enabled";
    case hipErrorPeerAccessNotEnabled:
      return "peer access has not been enabled";
    case hipErrorSetOnActiveProcess:
      return "cannot set while device is active in this process";
    case hipErrorContextIsDestroyed:
      return "context is destroyed";
    case hipErrorAssert:
      return "device-side assert triggered";
    case hipErrorHostMemoryAlreadyRegistered:
      return "part or all of the requested memory range is already mapped";
    case hipErrorHostMemoryNotRegistered:
      return "pointer does not correspond to a registered memory region";
    case hipErrorLaunchFailure:
      return "unspecified launch failure";
    case hipErrorCooperativeLaunchTooLarge:
      return "too many blocks in cooperative launch";
    case hipErrorNotSupported:
      return "operation not supported";
    case hipErrorStreamCaptureUnsupported:
      return "operation not permitted when stream is capturing";
    case hipErrorStreamCaptureInvalidated:
      return "operation failed due to a previous error during capture";
    case hipErrorStreamCaptureMerge:
      return "operation would result in a merge of separate capture sequences";
    case hipErrorStreamCaptureUnmatched:
      return "capture was not ended in the same stream as it began";
    case hipErrorStreamCaptureUnjoined:
      return "capturing stream has unjoined work";
    case hipErrorStreamCaptureIsolation:
      return "dependency created on uncaptured work in another stream";
    case hipErrorStreamCaptureImplicit:
      return "operation would make the legacy stream depend on a capturing blocking stream";  // NOLINT
    case hipErrorCapturedEvent:
      return "operation not permitted on an event last recorded in a capturing stream";  // NOLINT
    case hipErrorStreamCaptureWrongThread:
      return "attempt to terminate a thread-local capture sequence from another thread";  // NOLINT
    case hipErrorGraphExecUpdateFailure:
      return "the graph update was not performed because it included changes which violated "
             "constraints specific to instantiated graph update";  // NOLINT
    case hipErrorRuntimeMemory:
      return "runtime memory call returned error";
    case hipErrorRuntimeOther:
      return "runtime call other than memory returned error";
    case hipErrorUnknown:
    default:
      return "unknown error";
  }
}