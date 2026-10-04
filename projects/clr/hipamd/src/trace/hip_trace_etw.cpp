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

void emit_api(uint32_t domain, uint32_t op_id, const char* op_name, uint64_t start_qpc,
              int32_t retval) {
  TraceLoggingWrite(rocm_hip_tlg, "hip_api", TraceLoggingUInt32(domain, "domain"),
                    TraceLoggingUInt32(op_id, "op_id"),
                    TraceLoggingString(op_name != nullptr ? op_name : "", "op_name"),
                    TraceLoggingUInt64(start_qpc, "start_qpc"),
                    TraceLoggingInt32(retval, "retval"));
}

}  // namespace trace
}  // namespace hip
