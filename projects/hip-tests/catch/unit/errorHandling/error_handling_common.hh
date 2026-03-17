/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime_api.h>
#include <hip_test_context.hh>

constexpr hipError_t kErrorEnumerators[] = {hipSuccess,
                                            hipErrorInvalidValue,
                                            hipErrorOutOfMemory,
                                            hipErrorNotInitialized,
                                            hipErrorDeinitialized,
                                            hipErrorProfilerDisabled,
                                            hipErrorProfilerNotInitialized,
                                            hipErrorProfilerAlreadyStarted,
                                            hipErrorProfilerAlreadyStopped,
#if HT_AMD
                                            hipErrorInvalidConfiguration,
                                            hipErrorInvalidPitchValue,
                                            hipErrorInvalidSymbol,
                                            hipErrorInvalidDevicePointer,
                                            hipErrorInvalidMemcpyDirection,
                                            hipErrorInsufficientDriver,
                                            hipErrorMissingConfiguration,
                                            hipErrorPriorLaunchFailure,
                                            hipErrorInvalidDeviceFunction,
#endif
                                            hipErrorNoDevice,
                                            hipErrorInvalidDevice,
                                            hipErrorInvalidImage,
                                            hipErrorInvalidContext,
                                            hipErrorContextAlreadyCurrent,
                                            hipErrorMapFailed,
                                            hipErrorUnmapFailed,
                                            hipErrorArrayIsMapped,
                                            hipErrorAlreadyMapped,
                                            hipErrorNoBinaryForGpu,
                                            hipErrorAlreadyAcquired,
                                            hipErrorNotMapped,
                                            hipErrorNotMappedAsArray,
                                            hipErrorNotMappedAsPointer,
                                            hipErrorECCNotCorrectable,
                                            hipErrorUnsupportedLimit,
                                            hipErrorContextAlreadyInUse,
                                            hipErrorPeerAccessUnsupported,
                                            hipErrorInvalidKernelFile,
                                            hipErrorInvalidGraphicsContext,
                                            hipErrorInvalidSource,
                                            hipErrorFileNotFound,
                                            hipErrorSharedObjectSymbolNotFound,
                                            hipErrorSharedObjectInitFailed,
                                            hipErrorOperatingSystem,
                                            hipErrorInvalidHandle,
                                            hipErrorIllegalState,
                                            hipErrorNotFound,
                                            hipErrorNotReady,
                                            hipErrorIllegalAddress,
                                            hipErrorLaunchOutOfResources,
                                            hipErrorLaunchTimeOut,
                                            hipErrorPeerAccessAlreadyEnabled,
                                            hipErrorPeerAccessNotEnabled,
                                            hipErrorSetOnActiveProcess,
                                            hipErrorContextIsDestroyed,
                                            hipErrorAssert,
                                            hipErrorHostMemoryAlreadyRegistered,
                                            hipErrorHostMemoryNotRegistered,
                                            hipErrorLaunchFailure,
                                            hipErrorCooperativeLaunchTooLarge,
                                            hipErrorNotSupported,
                                            hipErrorStreamCaptureUnsupported,
                                            hipErrorStreamCaptureInvalidated,
                                            hipErrorStreamCaptureMerge,
                                            hipErrorStreamCaptureUnmatched,
                                            hipErrorStreamCaptureUnjoined,
                                            hipErrorStreamCaptureIsolation,
                                            hipErrorStreamCaptureImplicit,
                                            hipErrorCapturedEvent,
                                            hipErrorStreamCaptureWrongThread,
                                            hipErrorGraphExecUpdateFailure,
                                            hipErrorUnknown,
#if HT_AMD
                                            hipErrorRuntimeMemory,
                                            hipErrorRuntimeOther
#endif
};

const char* ErrorName(hipError_t enumerator);

const char* ErrorString(hipError_t enumerator);
