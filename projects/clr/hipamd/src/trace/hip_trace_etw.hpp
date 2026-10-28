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

// Reported as the retval when the wrapped call escaped via an exception.
constexpr int32_t kRetvalException = -1;

// One event per call, emitted on return. An ETW event costs ~370ns on the measurement
// box almost irrespective of its payload, so the event count is the only lever that
// materially moves per-call cost; a start/end pair cost twice this for no extra
// information, since rocprofiler_buffer_tracing_hip_api_record_t wants both timestamps
// in a single record anyway.
//
// start_qpc is raw QueryPerformanceCounter ticks. The consumer takes the matching end
// from EVENT_HEADER::TimeStamp, which is QPC because the session sets
// Wnode.ClientContext = 1, so both ends share a clock domain.
void emit_api(uint32_t domain, uint32_t op_id, const char* op_name, uint64_t start_qpc,
              int32_t retval);

inline uint64_t now_qpc() {
  LARGE_INTEGER value;
  QueryPerformanceCounter(&value);
  return static_cast<uint64_t>(value.QuadPart);
}

inline int32_t to_retval(hipError_t value) { return static_cast<int32_t>(value); }

template <typename T> inline int32_t to_retval(const T&) { return 0; }

// Trailing return type rather than deduced auto so that a reference-returning EXPR
// keeps its reference-ness and stays type-identical to the untraced arm of the macro.
template <typename Fn>
auto invoke(uint32_t domain, uint32_t op_id, const char* op_name, Fn&& fn) -> decltype(fn()) {
  using return_type = decltype(fn());

  const uint64_t start_qpc = now_qpc();

  if constexpr (std::is_void_v<return_type>) {
    try {
      fn();
    } catch (...) {
      emit_api(domain, op_id, op_name, start_qpc, kRetvalException);
      throw;
    }
    emit_api(domain, op_id, op_name, start_qpc, 0);
  } else {
    try {
      return_type retval = fn();
      emit_api(domain, op_id, op_name, start_qpc, to_retval(retval));
      return retval;
    } catch (...) {
      emit_api(domain, op_id, op_name, start_qpc, kRetvalException);
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
