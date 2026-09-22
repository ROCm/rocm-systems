/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_trace_etw.hpp"

#include <atomic>

// rocm_hip_tlg — ETW provider for HIP API tracepoints.
// GUID: 726f636d-6869-7000-746c-670000000002
TRACELOGGING_DEFINE_PROVIDER(rocm_hip_tlg, "rocm_hip_tlg",
                             (0x726f636d, 0x6869, 0x7000, 0x74, 0x6c, 0x67, 0x00, 0x00, 0x00, 0x00,
                              0x02));

namespace hip {
namespace trace {
namespace {

// A LoadLibrary/FreeLibrary cycle must be able to re-register, so this is a state
// machine rather than a once_flag.
enum class provider_state : int {
  kUnregistered = 0,
  kRegistering = 1,
  kRegistered = 2,
};

std::atomic<provider_state> g_state{provider_state::kUnregistered};
std::atomic<uint64_t> g_correlation_id{0};

}  // namespace

void initialize() {
  auto expected = provider_state::kUnregistered;
  if (!g_state.compare_exchange_strong(expected, provider_state::kRegistering)) {
    return;
  }
  TraceLoggingRegister(rocm_hip_tlg);
  g_state.store(provider_state::kRegistered);
}

void finalize() {
  auto expected = provider_state::kRegistered;
  if (!g_state.compare_exchange_strong(expected, provider_state::kUnregistered)) {
    return;
  }
  TraceLoggingUnregister(rocm_hip_tlg);
}

uint64_t emit_api_enter(uint32_t domain, uint32_t op_id, const char* op_name) {
  const uint64_t correlation_id = g_correlation_id.fetch_add(1, std::memory_order_relaxed) + 1;

  TraceLoggingWrite(rocm_hip_tlg, "hip_api_enter", TraceLoggingUInt32(domain, "domain"),
                    TraceLoggingUInt32(op_id, "op_id"),
                    TraceLoggingString(op_name != nullptr ? op_name : "", "op_name"),
                    TraceLoggingUInt64(correlation_id, "correlation_id"));

  return correlation_id;
}

void emit_api_exit(uint32_t domain, uint32_t op_id, const char* op_name, uint64_t correlation_id,
                   int32_t retval) {
  TraceLoggingWrite(rocm_hip_tlg, "hip_api_exit", TraceLoggingUInt32(domain, "domain"),
                    TraceLoggingUInt32(op_id, "op_id"),
                    TraceLoggingString(op_name != nullptr ? op_name : "", "op_name"),
                    TraceLoggingUInt64(correlation_id, "correlation_id"),
                    TraceLoggingInt32(retval, "retval"));
}

}  // namespace trace
}  // namespace hip
