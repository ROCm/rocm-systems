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
static_assert(sizeof(hipExtCopyEngineSignal) == 24, "CopyEngine signal ABI changed");
static_assert(sizeof(hipExtCopyEngineCondition) == 24, "CopyEngine condition ABI changed");
static_assert(static_cast<uint32_t>(hipExtCopyEngineSignalSet) == ABCE_DEVICE_SIGNAL_SET &&
                  static_cast<uint32_t>(hipExtCopyEngineSignalAdd) == ABCE_DEVICE_SIGNAL_ADD &&
                  static_cast<uint32_t>(hipExtCopyEngineSignalSub) == ABCE_DEVICE_SIGNAL_SUB,
              "CopyEngine signal operations changed");
static_assert(HIP_EXT_COPY_ENGINE_SIGNAL_UNORDERED == ABCE_DEVICE_SIGNAL_UNORDERED,
              "CopyEngine signal flags changed");
static_assert(HIP_EXT_COPY_ENGINE_SUBMIT_NONBLOCKING == ABCE_DEVICE_SUBMIT_NONBLOCKING,
              "CopyEngine submit flags changed");
static_assert(static_cast<uint32_t>(hipExtCopyEngineCompareGreaterEqual) == ABCE_DEVICE_WAIT_GEQ &&
                  static_cast<uint32_t>(hipExtCopyEngineCompareLessEqual) == ABCE_DEVICE_WAIT_LEQ,
              "CopyEngine compares changed");

namespace hip {
namespace detail {

__device__ __forceinline__ abce_device_handle_t CopyEngineHandle(
    const hipExtCopyEngineDeviceHandle* handle) {
  abce_device_handle_t internal{};
  if (handle != nullptr) {
    // Through the global address space: a pointer made from an integer is
    // generic, which turns every engine load into a flat (VGPR) load.
    internal.engine = (abce_device_engine_t*)(
        (__attribute__((address_space(1))) abce_device_engine_t*)handle->stateAddress);
    internal.abi_version = handle->deviceAbiVersion;
  }
  return internal;
}

__device__ __forceinline__ abce_device_signal_t CopyEngineSignal(
    const hipExtCopyEngineSignal& signal) {
  abce_device_signal_t internal{};
  internal.address = reinterpret_cast<uint64_t*>(signal.signalAddress);
  internal.value = signal.value;
  internal.op = signal.op;
  internal.flags = signal.flags;
  return internal;
}

__device__ __forceinline__ abce_device_wait_t CopyEngineCondition(
    const hipExtCopyEngineCondition& condition) {
  abce_device_wait_t internal{};
  internal.address = reinterpret_cast<uint64_t*>(condition.signalAddress);
  internal.value = condition.value;
  internal.compare = condition.compare;
  return internal;
}

// A submit's optional arguments, converted to ABCE's types.
struct CopyEngineSubmitOptions {
  abce_device_signal_t completion;
  abce_device_wait_t dependency;
  bool has_completion;
  bool has_dependency;

  __device__ __forceinline__ CopyEngineSubmitOptions(const hipExtCopyEngineSignal* completion_in,
                                                     const hipExtCopyEngineCondition* dependency_in)
      : completion{}, dependency{},
        has_completion(completion_in != nullptr), has_dependency(dependency_in != nullptr) {
    if (has_completion) completion = CopyEngineSignal(*completion_in);
    if (has_dependency) dependency = CopyEngineCondition(*dependency_in);
  }

  __device__ __forceinline__ const abce_device_signal_t* Completion() const {
    return has_completion ? &completion : nullptr;
  }
  __device__ __forceinline__ const abce_device_wait_t* Dependency() const {
    return has_dependency ? &dependency : nullptr;
  }
};

__device__ __forceinline__ hipError_t CopyEngineStatus(abce_status_t status) {
  switch (status) {
    case ABCE_STATUS_OK:
      return hipSuccess;
    // The engine has no SDMA atomics for a signal that needs them.
    case ABCE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case ABCE_STATUS_NOT_READY:
      return hipErrorNotReady;
    default:
      return hipErrorInvalidValue;
  }
}

template <bool OneLane>
__device__ __forceinline__ hipError_t CopyEngineSubmitLinear(
    const hipExtCopyEngineDeviceHandle* handle, void* dst, const void* src, uint64_t size,
    const hipExtCopyEngineSignal* completion, const hipExtCopyEngineCondition* dependency,
    uint32_t flags) {
  const abce_device_handle_t internal = CopyEngineHandle(handle);
  if (!abce_device_handle_valid(&internal)) return hipErrorInvalidHandle;
  const CopyEngineSubmitOptions options(completion, dependency);
  return CopyEngineStatus(
      OneLane ? abce_device_submit_linear_one_lane(&internal, dst, src, size, options.Dependency(),
                                                   options.Completion(), flags)
              : abce_device_submit_linear(&internal, dst, src, size, options.Dependency(),
                                          options.Completion(), flags));
}

template <bool OneLane>
__device__ __forceinline__ hipError_t CopyEngineSubmitWrite(
    const hipExtCopyEngineDeviceHandle* handle, void* dst, uint64_t value, uint32_t size,
    const hipExtCopyEngineSignal* completion, const hipExtCopyEngineCondition* dependency,
    uint32_t flags) {
  const abce_device_handle_t internal = CopyEngineHandle(handle);
  if (!abce_device_handle_valid(&internal)) return hipErrorInvalidHandle;
  const CopyEngineSubmitOptions options(completion, dependency);
  return CopyEngineStatus(
      OneLane ? abce_device_submit_write_one_lane(&internal, dst, value, size,
                                                  options.Dependency(), options.Completion(),
                                                  flags)
              : abce_device_submit_write(&internal, dst, value, size, options.Dependency(),
                                         options.Completion(), flags));
}

template <bool OneLane>
__device__ __forceinline__ hipError_t CopyEngineSubmitSignal(
    const hipExtCopyEngineDeviceHandle* handle, const hipExtCopyEngineSignal* completion,
    const hipExtCopyEngineCondition* dependency, uint32_t flags) {
  if (completion == nullptr) return hipErrorInvalidValue;
  const abce_device_handle_t internal = CopyEngineHandle(handle);
  if (!abce_device_handle_valid(&internal)) return hipErrorInvalidHandle;
  const CopyEngineSubmitOptions options(completion, dependency);
  return CopyEngineStatus(
      OneLane ? abce_device_submit_signal_one_lane(&internal, options.Completion(),
                                                  options.Dependency(), flags)
              : abce_device_submit_signal(&internal, options.Completion(), options.Dependency(),
                                         flags));
}

}  // namespace detail
}  // namespace hip

/// Submit one linear copy of @p size bytes from @p src to @p dst on the handle's
/// engine. Returns once the copy is queued, not done. Operations on one handle
/// start in order but may run concurrently; only signals say what has finished.
///
/// @p completion (optional): signal applied once the copy's writes land, ordered
///   as HIP_EXT_COPY_ENGINE_SIGNAL_UNORDERED describes. Add and subtract, and
///   before gfx125 set (one 64-bit atomic swap), need SDMA atomics
///   (hipErrorNotSupported otherwise).
/// @p dependency (optional): the copy engine waits for it. Before gfx125 only the
///   low 32 bits are compared: the value must fit in 32 bits.
/// @p flags: HIP_EXT_COPY_ENGINE_SUBMIT_* flags.
///
/// Memory visibility is the caller's: the call releases nothing. Before
/// submitting, release at system scope the source, prior destination stores
/// and the signal's initial value. For a tile the workgroup wrote, that is
/// __syncthreads() and then __threadfence_system() from the submitting lane;
/// data other workgroups wrote needs their own release (a fence writes back
/// only its own XCC's L2). To consume what the engine wrote, wait until the
/// completion's word meets its condition (relaxed loads suffice while
/// spinning), then acquire at system scope, e.g.
/// __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""), before reading; that covers
/// the calling thread's CU and XCC only.
///
/// Liveness: a blocking submit waits while the ring is full, and a dependency
/// holds back the engine and everything queued after it until it is met. Never
/// make a submit depend, directly or through the ring filling up, on work that
/// can only run after it: a signal from a later operation on the same engine,
/// from a workgroup that may not be resident yet, or from a GPU that waits on
/// this one. Use HIP_EXT_COPY_ENGINE_SUBMIT_NONBLOCKING where that cannot be
/// ruled out. Kernel completion and stream synchronization do not wait for
/// submitted copies: before reusing a source or reading a destination, wait on a
/// signal, from a kernel or with hipStreamWaitValue64(stream, signal address,
/// value, hipStreamWaitValueGte).
///
/// Lanes of a wave take turns (see hipExtCopyEngineSubmitLinearOneLane).
/// hipErrorNotReady only with HIP_EXT_COPY_ENGINE_SUBMIT_NONBLOCKING.
static __device__ __forceinline__ hipError_t hipExtCopyEngineSubmitLinear(
    const hipExtCopyEngineDeviceHandle* handle, void* dst, const void* src, uint64_t size,
    const hipExtCopyEngineSignal* completion = nullptr,
    const hipExtCopyEngineCondition* dependency = nullptr, uint32_t flags = 0) {
  return hip::detail::CopyEngineSubmitLinear<false>(handle, dst, src, size, completion,
                                                    dependency, flags);
}

/// hipExtCopyEngineSubmitLinear for a caller that submits from exactly one lane
/// of its wave (other waves and workgroups may submit to the same engine at
/// once): no turn loop, about half the registers. hipErrorInvalidValue if more
/// lanes of a wave call it.
static __device__ __forceinline__ hipError_t hipExtCopyEngineSubmitLinearOneLane(
    const hipExtCopyEngineDeviceHandle* handle, void* dst, const void* src, uint64_t size,
    const hipExtCopyEngineSignal* completion = nullptr,
    const hipExtCopyEngineCondition* dependency = nullptr, uint32_t flags = 0) {
  return hip::detail::CopyEngineSubmitLinear<true>(handle, dst, src, size, completion,
                                                   dependency, flags);
}

/// Write the low @p size (1, 2, 4 or 8) bytes of @p value, carried in the
/// command, to @p dst aligned to min(size, 4); neighbouring bytes are untouched.
/// It is data, not a completion: on gfx125+ it may land before earlier copies,
/// so mark readiness with a signal. Otherwise as hipExtCopyEngineSubmitLinear.
static __device__ __forceinline__ hipError_t hipExtCopyEngineSubmitWrite(
    const hipExtCopyEngineDeviceHandle* handle, void* dst, uint64_t value, uint32_t size,
    const hipExtCopyEngineSignal* completion = nullptr,
    const hipExtCopyEngineCondition* dependency = nullptr, uint32_t flags = 0) {
  return hip::detail::CopyEngineSubmitWrite<false>(handle, dst, value, size, completion,
                                                   dependency, flags);
}

/// hipExtCopyEngineSubmitWrite for exactly one calling lane per wave.
static __device__ __forceinline__ hipError_t hipExtCopyEngineSubmitWriteOneLane(
    const hipExtCopyEngineDeviceHandle* handle, void* dst, uint64_t value, uint32_t size,
    const hipExtCopyEngineSignal* completion = nullptr,
    const hipExtCopyEngineCondition* dependency = nullptr, uint32_t flags = 0) {
  return hip::detail::CopyEngineSubmitWrite<true>(handle, dst, value, size, completion,
                                                  dependency, flags);
}

/// Queue @p completion (required) alone: it lands once every earlier operation
/// on the handle's engine has, so waiting on it drains the engine. Otherwise as
/// hipExtCopyEngineSubmitLinear.
static __device__ __forceinline__ hipError_t hipExtCopyEngineSubmitSignal(
    const hipExtCopyEngineDeviceHandle* handle, const hipExtCopyEngineSignal* completion,
    const hipExtCopyEngineCondition* dependency = nullptr, uint32_t flags = 0) {
  return hip::detail::CopyEngineSubmitSignal<false>(handle, completion, dependency, flags);
}

/// hipExtCopyEngineSubmitSignal for exactly one calling lane per wave.
static __device__ __forceinline__ hipError_t hipExtCopyEngineSubmitSignalOneLane(
    const hipExtCopyEngineDeviceHandle* handle, const hipExtCopyEngineSignal* completion,
    const hipExtCopyEngineCondition* dependency = nullptr, uint32_t flags = 0) {
  return hip::detail::CopyEngineSubmitSignal<true>(handle, completion, dependency, flags);
}

#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_EXT_COPY_ENGINE_H
