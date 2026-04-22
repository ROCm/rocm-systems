// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AQLMON_RUNTIME_CONTRACT_H
#define AQLMON_RUNTIME_CONTRACT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AQLMON_RUNTIME_CONTRACT_VERSION 1u
#define AQLMON_RUNTIME_NEGOTIATION_ABI_VERSION 1u

typedef enum aqlmon_completion_signal_mode_t {
  AQLMON_COMPLETION_SIGNAL_MODE_RUNTIME_PROVIDED = 0,
  AQLMON_COMPLETION_SIGNAL_MODE_MONITOR_PROVIDED = 1
} aqlmon_completion_signal_mode_t;

typedef enum aqlmon_status_t {
  AQLMON_STATUS_SUCCESS = 0,
  AQLMON_STATUS_DENIED = 1,
  AQLMON_STATUS_ERROR_INVALID_ARGUMENT = 2,
  AQLMON_STATUS_ERROR_VERSION_MISMATCH = 3
} aqlmon_status_t;

typedef enum aqlmon_completion_signal_capability_t {
  AQLMON_COMPLETION_SIGNAL_CAP_NONE = 0,
  AQLMON_COMPLETION_SIGNAL_CAP_KERNEL_DISPATCH_SIGNALS = 1u << 0
} aqlmon_completion_signal_capability_t;

typedef struct aqlmon_runtime_negotiation_request_t {
  uint64_t size;
  uint32_t abi_version;
  uint32_t reserved;
  aqlmon_completion_signal_mode_t proposed_mode;
  uint32_t proposed_capabilities;
  uint64_t reserved1[4];
} aqlmon_runtime_negotiation_request_t;

typedef struct aqlmon_runtime_negotiation_response_t {
  uint64_t size;
  uint32_t abi_version;
  uint32_t reserved;
  aqlmon_completion_signal_mode_t selected_mode;
  uint32_t granted_capabilities;
} aqlmon_runtime_negotiation_response_t;

// POC runtime-facing contract:
// - rocprofiler-register activates the runtime
// - the runtime calls negotiate once
// - aqlmon either accepts runtime-provided kernel completion signals or denies the request
// - if the runtime never negotiates, aqlmon falls back to MONITOR_PROVIDED

uint32_t aqlmon_runtime_contract_version(void);

aqlmon_status_t aqlmon_runtime_negotiate(
    const aqlmon_runtime_negotiation_request_t* request,
    aqlmon_runtime_negotiation_response_t* response);

#ifdef __cplusplus
}
#endif

#endif
