/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"

namespace hip {
hipError_t hipExtGetLastError() {
  HIP_INIT_API(hipExtGetLastError);
  hipError_t err = hip::tls.last_command_error_;
  hip::tls.last_command_error_ = hipSuccess;
  return err;
}

hipError_t hipGetLastError() {
  HIP_INIT_API(hipGetLastError);
  hipError_t err = hip::tls.last_error_;
  hip::tls.last_error_ = hipSuccess;
  return err;
}

hipError_t hipPeekAtLastError() {
  HIP_INIT_API(hipPeekAtLastError);
  hipError_t err = hip::tls.last_error_;
  HIP_RETURN(err);
}

const char* ihipGetErrorName(hipError_t hip_error) {
  switch (hip_error) {
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
  };
}

const char* ihipGetErrorString(hipError_t hip_error) {
  switch (hip_error) {
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
      return "profiler is not initialized";
    case hipErrorProfilerAlreadyStarted:
      return "profiler already started";
    case hipErrorProfilerAlreadyStopped:
      return "profiler already stopped";
    case hipErrorInvalidConfiguration:
      return "invalid configuration argument";
    case hipErrorInvalidPitchValue:
      return "invalid pitch argument";
    case hipErrorInvalidSymbol:
      return "invalid device symbol";
    case hipErrorInvalidDevicePointer:
      return "invalid device pointer";
    case hipErrorInvalidMemcpyDirection:
      return "invalid copy direction for memcpy";
    case hipErrorInsufficientDriver:
      return "driver version is insufficient for runtime version";
    case hipErrorMissingConfiguration:
      return "__global__ function call is not configured";
    case hipErrorPriorLaunchFailure:
      return "unspecified launch failure in prior launch";
    case hipErrorInvalidDeviceFunction:
      return "invalid device function";
    case hipErrorNoDevice:
      return "no ROCm-capable device is detected";
    case hipErrorInvalidDevice:
      return "invalid device ordinal";
    case hipErrorInvalidImage:
      return "device kernel image is invalid";
    case hipErrorInvalidContext:
      return "invalid device context";
    case hipErrorContextAlreadyCurrent:
      return "context is already current context";
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
      return "invalid kernel file";
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
      return "operation would make the legacy stream depend on a capturing blocking stream";
    case hipErrorCapturedEvent:
      return "operation not permitted on an event last recorded in a capturing stream";
    case hipErrorStreamCaptureWrongThread:
      return "attempt to terminate a thread-local capture sequence from another thread";
    case hipErrorGraphExecUpdateFailure:
      return "the graph update was not performed because it included changes which violated "
             "constraints specific to instantiated graph update";
    case hipErrorRuntimeMemory:
      return "runtime memory call returned error";
    case hipErrorRuntimeOther:
      return "runtime call other than memory returned error";
    case hipErrorUnknown:
    default:
      return "unknown error";
  }
}

const char* hipGetErrorName(hipError_t hip_error) { return ihipGetErrorName(hip_error); }

const char* hipGetErrorString(hipError_t hip_error) { return ihipGetErrorString(hip_error); }

hipError_t hipDrvGetErrorName(hipError_t hip_error, const char** errStr) {
  if (errStr == nullptr) {
    return hipErrorInvalidValue;
  }
  *errStr = ihipGetErrorName(hip_error);
  if (hip_error == hipErrorUnknown || strcmp(*errStr, "hipErrorUnknown") != 0) {
    return hipSuccess;
  } else {
    return hipErrorInvalidValue;
  }
}

hipError_t hipDrvGetErrorString(hipError_t hip_error, const char** errStr) {
  if (errStr == nullptr) {
    return hipErrorInvalidValue;
  }
  *errStr = ihipGetErrorString(hip_error);
  if (hip_error == hipErrorUnknown || strcmp(*errStr, "unknown error") != 0) {
    return hipSuccess;
  } else {
    return hipErrorInvalidValue;
  }
}
}  // namespace hip
