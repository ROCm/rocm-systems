// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AQLMON_RUNTIME_CONTRACT_H
#define AQLMON_RUNTIME_CONTRACT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runtime-facing contract version. Bump for any public API evolution.
#define AQLMON_RUNTIME_CONTRACT_API_VERSION 3u
// Runtime negotiation ABI/layout version. Bump for incompatible struct or calling changes.
#define AQLMON_RUNTIME_CONTRACT_ABI_VERSION 2u

// Backward-compatible aliases for the initial POC naming.
// `AQLMON_RUNTIME_CONTRACT_VERSION` aliases the runtime-facing API version, not the ABI/layout.
#define AQLMON_RUNTIME_CONTRACT_VERSION AQLMON_RUNTIME_CONTRACT_API_VERSION
#define AQLMON_RUNTIME_NEGOTIATION_ABI_VERSION AQLMON_RUNTIME_CONTRACT_ABI_VERSION

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
  // Serialized aqlmon_completion_signal_mode_t value.
  uint32_t proposed_mode;
  uint32_t proposed_capabilities;
  uint64_t reserved1[4];
} aqlmon_runtime_negotiation_request_t;

typedef struct aqlmon_runtime_negotiation_response_t {
  uint64_t size;
  uint32_t abi_version;
  uint32_t reserved;
  uint32_t api_version;
  // Serialized aqlmon_completion_signal_mode_t value.
  uint32_t selected_mode;
  uint32_t granted_capabilities;
  uint64_t reserved1[3];
} aqlmon_runtime_negotiation_response_t;

// POC runtime-facing contract:
// - rocprofiler-register activates the runtime
// - the runtime calls negotiate once per process before relying on the result
// - negotiation is process-global today: the first successful result becomes the process-wide mode
// - request->size/request->abi_version and response->size/response->abi_version must be set
// - response->api_version reports the supported runtime-facing API version
// - response->selected_mode is the canonical ownership result
// - AQLMON_STATUS_SUCCESS means the selected mode is active
// - AQLMON_STATUS_DENIED means the proposed runtime-owned mode was declined and
//   response->selected_mode reports the fallback mode
// - AQLMON_STATUS_ERROR_VERSION_MISMATCH still returns the supported ABI/API versions in
//   the response when response->size is large enough
// - if the runtime never negotiates, aqlmon falls back to MONITOR_PROVIDED

uint32_t aqlmon_runtime_contract_version(void);

uint32_t aqlmon_runtime_contract_api_version(void);

uint32_t aqlmon_runtime_contract_abi_version(void);

aqlmon_completion_signal_mode_t aqlmon_runtime_completion_signal_mode(void);

uint32_t aqlmon_runtime_completion_signal_capabilities(void);

aqlmon_status_t aqlmon_runtime_negotiate(
    const aqlmon_runtime_negotiation_request_t* request,
    aqlmon_runtime_negotiation_response_t* response);

#ifdef __cplusplus
}
#endif

#endif
