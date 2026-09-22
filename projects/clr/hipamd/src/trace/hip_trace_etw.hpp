/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if defined(HIP_TRACE_BACKEND_TRACELOGGING)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// Keep <windows.h> above <TraceLoggingProvider.h>; the latter is not self-contained.
#include <TraceLoggingProvider.h>

#include <cstdint>
#include <type_traits>

#include <hip/amd_detail/hip_prof_str.h>

#include "hip_trace_init.hpp"

TRACELOGGING_DECLARE_PROVIDER(rocm_hip_tlg);

namespace hip {
namespace trace {

// Mirrors the split between HipDispatchTable and HipCompilerDispatchTable in
// hip_table_interface.cpp so a consumer can tell the two apart.
enum domain_t : uint32_t {
  kDomainRuntimeApi = 0,
  kDomainCompilerApi = 1,
};

// Reported as the exit retval when the wrapped call escaped via an exception.
constexpr int32_t kRetvalException = -1;

uint64_t emit_api_enter(uint32_t domain, uint32_t op_id, const char* op_name);
void emit_api_exit(uint32_t domain, uint32_t op_id, const char* op_name, uint64_t correlation_id,
                   int32_t retval);

inline int32_t to_retval(hipError_t value) { return static_cast<int32_t>(value); }

template <typename T> inline int32_t to_retval(const T&) { return 0; }

// Trailing return type rather than deduced auto so that a reference-returning EXPR
// keeps its reference-ness and stays type-identical to the untraced arm of the macro.
template <typename Fn>
auto invoke(uint32_t domain, uint32_t op_id, const char* op_name, Fn&& fn) -> decltype(fn()) {
  using return_type = decltype(fn());

  const uint64_t correlation_id = emit_api_enter(domain, op_id, op_name);

  if constexpr (std::is_void_v<return_type>) {
    try {
      fn();
    } catch (...) {
      emit_api_exit(domain, op_id, op_name, correlation_id, kRetvalException);
      throw;
    }
    emit_api_exit(domain, op_id, op_name, correlation_id, 0);
  } else {
    try {
      return_type retval = fn();
      emit_api_exit(domain, op_id, op_name, correlation_id, to_retval(retval));
      return retval;
    } catch (...) {
      emit_api_exit(domain, op_id, op_name, correlation_id, kRetvalException);
      throw;
    }
  }
}

}  // namespace trace
}  // namespace hip

// Wraps the return expression of a HIP_PUBLIC_API entry point. When no ETW session
// has enabled the provider this is a single inlined load plus a not-taken branch.
#define HIP_TRACE_API(NAME, EXPR)                                                                  \
  (TraceLoggingProviderEnabled(rocm_hip_tlg, 0, 0)                                                 \
       ? ::hip::trace::invoke(::hip::trace::kDomainRuntimeApi, HIP_API_ID_##NAME, #NAME,           \
                              [&]() -> decltype(EXPR) { return (EXPR); })                          \
       : (EXPR))

#else  // !HIP_TRACE_BACKEND_TRACELOGGING

// Token-identical to the unwrapped expression, so non-Windows object code is unchanged.
#define HIP_TRACE_API(NAME, EXPR) (EXPR)

#endif  // HIP_TRACE_BACKEND_TRACELOGGING
