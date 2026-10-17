/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_EXT_COPY_ENGINE_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_EXT_COPY_ENGINE_H

#if __has_include(<abce_device.h>)
#include <abce_device.h>
#else
#include <abce/abce_device.h>
#endif

static_assert(sizeof(hipExtCopyEngineDeviceHandle) == 16, "CopyEngine device handle ABI changed");
static_assert(sizeof(hipExtCopyEngineToken) == 16, "CopyEngine token ABI changed");
static_assert(HIP_EXT_COPY_ENGINE_AGENT_HOST == ABCE_DEVICE_AGENT_HOST &&
                  HIP_EXT_COPY_ENGINE_AGENT_UNKNOWN == ABCE_DEVICE_AGENT_UNKNOWN,
              "CopyEngine agent id conventions changed");

namespace hip {
namespace detail {

__device__ __forceinline__ abce_device_handle_t CopyEngineHandle(
    const hipExtCopyEngineDeviceHandle* handle) {
  abce_device_handle_t internal{};
  if (handle != nullptr) {
    internal.engine_set = reinterpret_cast<abce_device_engine_set_t*>(handle->stateAddress);
    internal.abi_version = handle->deviceAbiVersion;
  }
  return internal;
}

__device__ __forceinline__ abce_device_token_t CopyEngineToken(const hipExtCopyEngineToken& token) {
  abce_device_token_t internal{};
  internal.signal = reinterpret_cast<uint64_t*>(token.signalAddress);
  internal.reference = token.reference;
  return internal;
}

__device__ __forceinline__ hipError_t CopyEngineStatus(abce_status_t status) {
  switch (status) {
    case ABCE_STATUS_OK:
      return hipSuccess;
    // This device cannot run it: no SDMA completion mechanism, no ring a kernel
    // may use, or every such ring reserved for other traffic.
    case ABCE_STATUS_UNIMPLEMENTED:
    case ABCE_STATUS_NO_ENGINE:
    case ABCE_STATUS_NO_LEGAL_ENGINE:
      return hipErrorNotSupported;
    default:
      return hipErrorInvalidValue;
  }
}

}  // namespace detail
}  // namespace hip

/// Agent id naming @p deviceId, for tagging a copy's endpoints. Returns
/// HIP_EXT_COPY_ENGINE_AGENT_UNKNOWN for a device the runtime did not publish,
/// which still submits and simply places the copy by ring availability.
static __device__ __forceinline__ int8_t
hipExtCopyEngineAgentFor(const hipExtCopyEngineDeviceHandle* handle, int deviceId) {
  if (deviceId == HIP_EXT_COPY_ENGINE_AGENT_HOST) return HIP_EXT_COPY_ENGINE_AGENT_HOST;
  if (deviceId < 0) return HIP_EXT_COPY_ENGINE_AGENT_UNKNOWN;
  const abce_device_handle_t internal = hip::detail::CopyEngineHandle(handle);
  return abce_device_agent_for(&internal, static_cast<uint32_t>(deviceId));
}

/// Agent id of the device the kernel is running on, which a kernel cannot name
/// on its own. This is the local end of any copy submitted through @p handle.
static __device__ __forceinline__ int8_t
hipExtCopyEngineLocalAgent(const hipExtCopyEngineDeviceHandle* handle) {
  const abce_device_handle_t internal = hip::detail::CopyEngineHandle(handle);
  return abce_device_local_agent(&internal);
}

/// Submit one linear copy of @p size bytes from @p src to @p dst that retires
/// @p completion, after the copy behind @p dependency when that is not null.
/// Kernel completion does not imply copy completion unless the kernel waits
/// before returning.
///
/// @p completion names an aligned 64-bit word in system-visible memory, such as
/// memory allocated with hipMallocSignalMemory, and the value the copy lands it
/// on. The call arms the word itself; poll it with hipExtCopyEngineTokenQuery or
/// hipExtCopyEngineTokenWait, and do not reuse the word until the copy retires.
///
/// @p dependency is an earlier copy's completion. The copy engine holds this
/// copy until that one has retired; the calling thread does not wait.
///
/// @p srcAgent and @p dstAgent only steer which copy engine serves the copy —
/// see hipExtCopyEngineAgentFor and hipExtCopyEngineLocalAgent — so leaving them
/// unknown is always valid.
///
/// Any number of lanes may submit at once, each its own copy; lanes of one wave
/// take turns inside the call.
///
/// This call is a system-scope release point: stores the calling thread made to
/// the source and destination before it are ordered ahead of the copy, so a
/// caller does not need to — and should not — release those buffers itself, as a
/// second release only pays the cache writeback twice. Stores made by other
/// threads are covered once the caller has synchronised with them beforehand.
static __device__ __forceinline__ hipError_t hipExtCopyEngineSubmitLinear(
    const hipExtCopyEngineDeviceHandle* handle, void* dst, const void* src, uint64_t size,
    const hipExtCopyEngineToken* completion, const hipExtCopyEngineToken* dependency = nullptr,
    int8_t srcAgent = HIP_EXT_COPY_ENGINE_AGENT_UNKNOWN,
    int8_t dstAgent = HIP_EXT_COPY_ENGINE_AGENT_UNKNOWN) {
  if (completion == nullptr) return hipErrorInvalidValue;
  const abce_device_handle_t internal = hip::detail::CopyEngineHandle(handle);
  if (!abce_device_handle_valid(&internal)) return hipErrorInvalidHandle;
  const abce_device_token_t internal_completion = hip::detail::CopyEngineToken(*completion);
  abce_device_token_t internal_dependency{};
  if (dependency != nullptr) internal_dependency = hip::detail::CopyEngineToken(*dependency);
  const abce_status_t status = abce_device_submit_linear(
      &internal, dst, src, size, srcAgent, dstAgent, ABCE_DEVICE_AUTO_ENGINE,
      dependency != nullptr ? &internal_dependency : nullptr, &internal_completion);
  return hip::detail::CopyEngineStatus(status);
}

/// Whether @p token's copy has retired. hipErrorNotReady is the ordinary answer
/// for a copy still in flight.
static __device__ __forceinline__ hipError_t
hipExtCopyEngineTokenQuery(const hipExtCopyEngineToken* token) {
  if (token == nullptr || token->signalAddress == 0) return hipErrorInvalidValue;
  const abce_device_token_t internal = hip::detail::CopyEngineToken(*token);
  return abce_device_token_retired(&internal) ? hipSuccess : hipErrorNotReady;
}

/// Wait for token completion and acquire visibility of copy-engine writes.
/// Returns hipErrorNotReady when maxSpins elapses before the copy completes.
static __device__ __forceinline__ hipError_t
hipExtCopyEngineTokenWait(const hipExtCopyEngineToken* token, uint64_t maxSpins) {
  if (token == nullptr || token->signalAddress == 0) return hipErrorInvalidValue;
  const abce_device_token_t internal = hip::detail::CopyEngineToken(*token);
  return abce_device_token_wait(&internal, maxSpins) ? hipSuccess : hipErrorNotReady;
}

#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_EXT_COPY_ENGINE_H
