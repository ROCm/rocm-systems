// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AQLMON_RUNTIME_CONTRACT_PRIVATE_HPP
#define AQLMON_RUNTIME_CONTRACT_PRIVATE_HPP

#include "aqlmon/runtime_contract.h"

#if defined(__GNUC__)
#define AQLMON_RUNTIME_CONTRACT_CPP_EXPORT __attribute__((visibility("default")))
#else
#define AQLMON_RUNTIME_CONTRACT_CPP_EXPORT
#endif

namespace aqlmon::runtime_contract {

struct NegotiationSnapshot {
  bool negotiated = false;
  aqlmon_status_t status = AQLMON_STATUS_SUCCESS;
  aqlmon_completion_signal_mode_t selected_mode =
      AQLMON_COMPLETION_SIGNAL_MODE_MONITOR_PROVIDED;
  uint32_t granted_capabilities = 0;
  uint32_t abi_version = AQLMON_RUNTIME_NEGOTIATION_ABI_VERSION;
};

AQLMON_RUNTIME_CONTRACT_CPP_EXPORT
aqlmon_completion_signal_mode_t effective_completion_signal_mode();

}  // namespace aqlmon::runtime_contract

#undef AQLMON_RUNTIME_CONTRACT_CPP_EXPORT

#endif
