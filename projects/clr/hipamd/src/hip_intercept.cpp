/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "hip_prof_api.h"

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

extern "C" void hipRegisterTracerCallback(int (*function)(activity_domain_t domain,
                                                          uint32_t operation_id, void* data)) {
  amd::activity_prof::report_activity.store(function, std::memory_order_relaxed);
}
