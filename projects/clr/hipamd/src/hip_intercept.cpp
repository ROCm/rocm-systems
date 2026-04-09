/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "hip_prof_api.h"
#include "platform/clr_prof_event_bus.hpp"
#include "platform/clr_prof_interface.h"

// HIP API callback/activity
namespace hip {

extern const std::string& FunctionName(const hipFunction_t f);

int hipGetStreamDeviceId(hipStream_t stream) {
  if (!hip::isValid(stream)) {
    return -1;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  return (s != nullptr) ? s->DeviceId() : ihipGetDevice();
}

const char* hipKernelNameRef(const hipFunction_t function) {
  return (function != nullptr) ? FunctionName(function).c_str() : nullptr;
}

const char* hipKernelNameRefByPtr(const void* host_function, hipStream_t stream) {
  [](auto&&...) {}(stream);
  return (host_function != nullptr) ? PlatformState::Instance().StatCO().GetFuncName(host_function)
                                    : nullptr;
}
const char* hipApiName(uint32_t id) { return hip_api_name(id); }

}  // namespace hip

// ---------------------------------------------------------------------------
// Legacy compatibility: hipRegisterTracerCallback
//
// Kept so that roctracer (and any other tool using the old single-callback
// protocol) continues to work unchanged.  Internally this now routes through
// the EventBus legacy shim, which bridges the old protocol to the new typed
// event stream.  This means both old (roctracer) and new (clr_prof_subscribe)
// subscribers can coexist in the same process.
// ---------------------------------------------------------------------------
extern "C" void hipRegisterTracerCallback(int (*function)(activity_domain_t domain,
                                                          uint32_t operation_id, void* data)) {
  // Keep the raw pointer for the legacy emit path in activity.cpp so that
  // roctracer's pool/buffer machinery (CommitRecord sentinel, etc.) still works.
  amd::activity_prof::report_activity.store(function, std::memory_order_release);
  // Also wire it through the EventBus so CLR-level IsEnabled() can use the
  // subscriber-count fast path even for legacy tools.
  amd::clr_prof::EventBus::instance().set_legacy_callback(function);
}

// ---------------------------------------------------------------------------
// clr_prof C API — forwarded to the EventBus implementation in
// clr_prof_event_bus.cpp (compiled into rocclr, linked into libamdhip64).
// These symbols are exported in hip_hcc.map.in / amdhip.def.
// ---------------------------------------------------------------------------
extern "C" {

clr_prof_subscriber_t clr_prof_subscribe(const clr_prof_callbacks_t*  callbacks,
                                          const clr_prof_api_filter_t* filter) {
  return amd::clr_prof::EventBus::instance().subscribe(callbacks, filter);
}

void clr_prof_unsubscribe(clr_prof_subscriber_t subscriber) {
  amd::clr_prof::EventBus::instance().unsubscribe(subscriber);
}

uint64_t clr_prof_get_correlation_id(void) {
  return amd::activity_prof::correlation_id;
}

const char* clr_prof_api_name(uint32_t api_id) {
  return hip_api_name(api_id);
}

const char* clr_prof_gpu_op_name(clr_prof_gpu_op_t op) {
  switch (op) {
    case CLR_PROF_OP_KERNEL_DISPATCH: return "KernelDispatch";
    case CLR_PROF_OP_MEMCPY:          return "MemCopy";
    case CLR_PROF_OP_BARRIER:         return "Barrier";
    default:                          return "Unknown";
  }
}

void clr_prof_version(uint32_t* major, uint32_t* minor) {
  if (major) *major = CLR_PROF_VERSION_MAJOR;
  if (minor) *minor = CLR_PROF_VERSION_MINOR;
}

}  // extern "C"
