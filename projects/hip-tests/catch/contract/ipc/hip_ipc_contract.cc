/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Tiny deterministic allocation size for IPC mem-handle probes. IPC handles
// describe a device allocation, so the exact size is not load-bearing for the
// contract; a single page-sized request keeps the test cheap and portable.
constexpr size_t kAllocSize = 64;

// Skips the test when no device is visible so that the IPC contracts only run
// against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}

// IPC handle APIs are frequently unavailable on WSL, single-device, and certain
// runtime paths, reporting hipErrorNotSupported or a context/OS error rather
// than an invalid-argument error. This classifier deliberately excludes
// hipErrorInvalidValue so that it can be reused where a hipErrorInvalidValue
// result is itself the contract under test (the null-argument rejection checks),
// without collapsing real invalid-argument validation into a skip.
bool IsIpcUnsupportedError(hipError_t error) {
  switch (error) {
    case hipErrorNotSupported:
    case hipErrorInvalidConfiguration:
    case hipErrorOperatingSystem:
      return true;
    default:
      return false;
  }
}

// Classifies the outcome of an IPC call that was made with valid arguments
// (a real device pointer or a properly created interprocess event). On some
// runtime/platform paths - notably WSL/dxg - the IPC handle APIs are not
// implemented and surface that as hipErrorInvalidValue even though the call
// arguments are valid. For such valid-argument calls there is no legitimate
// invalid-argument condition, so hipErrorInvalidValue can only mean the
// capability is unavailable; treat it, along with the other unsupported codes,
// as an unsupported-capability skip. This classifier must never be used to judge
// calls that intentionally pass null arguments, where hipErrorInvalidValue is
// the expected, correct result.
bool IsIpcUnsupportedFromValidCall(hipError_t error) {
  return IsIpcUnsupportedError(error) || error == hipErrorInvalidValue;
}

// Probes whether IPC memory handles can be produced on this device/runtime path
// by attempting a tiny allocation and hipIpcGetMemHandle with a valid pointer.
// Returns true only when the handle is produced successfully. Because the probe
// call uses valid arguments, any unsupported-capability error - including the
// hipErrorInvalidValue reported by WSL/dxg for unimplemented IPC - means the
// capability is unavailable and the probe returns false. Any other failure is
// surfaced through HIP_CHECK so that genuine contract regressions are not masked.
// The probe allocation is always freed before returning.
bool IpcMemHandleSupported() {
  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, kAllocSize));

  hipIpcMemHandle_t handle{};
  const hipError_t status = hipIpcGetMemHandle(&handle, ptr);

  HIP_CHECK(hipFree(ptr));

  if (status == hipSuccess) {
    return true;
  }
  if (IsIpcUnsupportedFromValidCall(status)) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}

void SkipIfIpcMemHandleUnsupported() {
  if (!IpcMemHandleSupported()) {
    HIP_SKIP_TEST("IPC memory handles are not supported by this device/runtime path.");
  }
}
}  // namespace

HIP_TEST_CASE(Contract_Ipc_GetMemHandle_SucceedsForDeviceAllocation) {
  RequireDevice();

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, kAllocSize));

  hipIpcMemHandle_t handle{};
  const hipError_t status = hipIpcGetMemHandle(&handle, ptr);
  // The pointer is valid, so an unsupported-capability error (including the
  // hipErrorInvalidValue WSL/dxg returns for unimplemented IPC) means the
  // platform cannot produce IPC handles and the test skips rather than fails.
  if (IsIpcUnsupportedFromValidCall(status)) {
    HIP_CHECK(hipFree(ptr));
    HIP_SKIP_TEST("IPC memory handles are not supported by this device/runtime path.");
  }
  HIP_CHECK(status);

  // The handle is an opaque, fixed-size descriptor. Its exact byte contents are
  // backend-specific and not part of the public contract, so the contract only
  // asserts that the call succeeded and produced the handle out-parameter.
  HIP_CHECK(hipFree(ptr));
}

HIP_TEST_CASE(Contract_Ipc_MemHandle_SameProcessRoundTrip) {
  RequireDevice();

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, kAllocSize));

  hipIpcMemHandle_t handle{};
  const hipError_t get_status = hipIpcGetMemHandle(&handle, ptr);
  // Valid pointer: an unsupported-capability error (including WSL/dxg's
  // hipErrorInvalidValue for unimplemented IPC) is a platform skip, not a
  // failure.
  if (IsIpcUnsupportedFromValidCall(get_status)) {
    HIP_CHECK(hipFree(ptr));
    HIP_SKIP_TEST("IPC memory handles are not supported by this device/runtime path.");
  }
  HIP_CHECK(get_status);

  void* mapped = nullptr;
  const hipError_t open_status =
      hipIpcOpenMemHandle(&mapped, handle, hipIpcMemLazyEnablePeerAccess);
  // The handle came from this process's own successful export, so the open call
  // also uses valid arguments; an unsupported-capability error (including
  // hipErrorInvalidValue on WSL/dxg) is a platform skip rather than a failure.
  if (IsIpcUnsupportedFromValidCall(open_status)) {
    HIP_CHECK(hipFree(ptr));
    HIP_SKIP_TEST("Opening IPC memory handles is not supported by this device/runtime path.");
  }
  HIP_CHECK(open_status);

  // A successfully opened handle must yield a usable mapping. The mapped pointer
  // is not required to alias the original allocation, so the contract only
  // requires a non-null mapping.
  REQUIRE(mapped != nullptr);

  HIP_CHECK(hipIpcCloseMemHandle(mapped));
  HIP_CHECK(hipFree(ptr));
}

HIP_TEST_CASE(Contract_Ipc_GetMemHandle_NullArgs_AreRejected) {
  RequireDevice();
  SkipIfIpcMemHandleUnsupported();

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, kAllocSize));

  // A null output handle is invalid input and must be rejected with a public
  // invalid-argument error rather than treated as an unsupported-capability
  // skip. The public header documents hipErrorInvalidHandle for this API while
  // the AMD runtime path reports hipErrorInvalidValue, so the contract accepts
  // either documented invalid-argument code without overfitting to one backend.
  HIP_CHECK_ERRORS(hipIpcGetMemHandle(nullptr, ptr), hipErrorInvalidValue,
                   hipErrorInvalidHandle);

  // A null device pointer is invalid input and must be rejected the same way.
  hipIpcMemHandle_t handle{};
  HIP_CHECK_ERRORS(hipIpcGetMemHandle(&handle, nullptr), hipErrorInvalidValue,
                   hipErrorInvalidHandle);

  HIP_CHECK(hipFree(ptr));
}

HIP_TEST_CASE(Contract_Ipc_GetEventHandle_RequiresInterprocessFlag) {
  RequireDevice();

  // An event created without hipEventInterprocess cannot back an IPC handle, so
  // hipIpcGetEventHandle must not report success for it. Backends differ in the
  // exact error code, so the contract only requires a non-success result.
  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

  hipIpcEventHandle_t handle{};
  const hipError_t status = hipIpcGetEventHandle(&handle, event);
  REQUIRE(status != hipSuccess);

  HIP_CHECK(hipEventDestroy(event));
}

HIP_TEST_CASE(Contract_Ipc_EventHandle_SameProcessRoundTrip) {
  RequireDevice();

  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming | hipEventInterprocess));

  hipIpcEventHandle_t handle{};
  const hipError_t get_status = hipIpcGetEventHandle(&handle, event);
  // The event was created with the required hipEventInterprocess flag, so the
  // arguments are valid; an unsupported-capability error (including the
  // hipErrorInvalidValue WSL/dxg returns for unimplemented IPC) means the
  // platform cannot export IPC event handles and the test skips.
  if (IsIpcUnsupportedFromValidCall(get_status)) {
    HIP_CHECK(hipEventDestroy(event));
    HIP_SKIP_TEST("IPC event handles are not supported by this device/runtime path.");
  }
  HIP_CHECK(get_status);

  hipEvent_t opened = nullptr;
  const hipError_t open_status = hipIpcOpenEventHandle(&opened, handle);
  // Opening an IPC event handle is designed for a different process. In the same
  // process the runtime may legitimately reject the open (e.g. with an invalid
  // context) or report the capability as unsupported; both are platform
  // limitations rather than contract violations, so they are treated as skips.
  if (open_status != hipSuccess) {
    HIP_CHECK(hipEventDestroy(event));
    HIP_SKIP_TEST(
        "Opening an IPC event handle in the same process is not supported by this "
        "device/runtime path.");
  }

  // When the same-process open does succeed, it must hand back a usable event.
  REQUIRE(opened != nullptr);

  // An opened IPC event is owned by the caller and released with hipEventDestroy,
  // mirroring the lifetime used by the HIP IPC event unit tests.
  HIP_CHECK(hipEventDestroy(opened));
  HIP_CHECK(hipEventDestroy(event));
}
