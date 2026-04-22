// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "runtime_contract.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <type_traits>

namespace {

constexpr uint32_t kKnownCapabilities =
    AQLMON_COMPLETION_SIGNAL_CAP_KERNEL_DISPATCH_SIGNALS;

constexpr uint32_t kRuntimeProvidedMode =
    static_cast<uint32_t>(AQLMON_COMPLETION_SIGNAL_MODE_RUNTIME_PROVIDED);
constexpr uint32_t kMonitorProvidedMode =
    static_cast<uint32_t>(AQLMON_COMPLETION_SIGNAL_MODE_MONITOR_PROVIDED);

bool has_struct_bytes(uint64_t size, size_t offset, size_t field_size) {
  return size >= (offset + field_size);
}

bool request_has_required_fields(const aqlmon_runtime_negotiation_request_t& request) {
  return has_struct_bytes(request.size, offsetof(aqlmon_runtime_negotiation_request_t, proposed_mode),
                          sizeof(request.proposed_mode)) &&
         has_struct_bytes(request.size,
                          offsetof(aqlmon_runtime_negotiation_request_t, proposed_capabilities),
                          sizeof(request.proposed_capabilities));
}

bool response_has_required_fields(const aqlmon_runtime_negotiation_response_t& response) {
  return has_struct_bytes(
             response.size, offsetof(aqlmon_runtime_negotiation_response_t, selected_mode),
             sizeof(response.selected_mode)) &&
         has_struct_bytes(
             response.size, offsetof(aqlmon_runtime_negotiation_response_t, granted_capabilities),
             sizeof(response.granted_capabilities));
}

bool is_valid_completion_signal_mode(uint32_t mode) {
  return mode == kRuntimeProvidedMode || mode == kMonitorProvidedMode;
}

aqlmon_completion_signal_mode_t to_completion_signal_mode(uint32_t mode) {
  return (mode == kRuntimeProvidedMode) ? AQLMON_COMPLETION_SIGNAL_MODE_RUNTIME_PROVIDED
                                        : AQLMON_COMPLETION_SIGNAL_MODE_MONITOR_PROVIDED;
}

enum class CompletionSignalPolicy {
  kMonitorProvided,
  kRuntimeProvided
};

CompletionSignalPolicy completion_signal_policy() {
  static const CompletionSignalPolicy policy = []() {
    const char* value = getenv("AQLMONITOR_COMPLETION_SIGNAL_POLICY");
    if(value != nullptr && *value != '\0') {
      if(strcasecmp(value, "runtime") == 0 ||
         strcasecmp(value, "runtime-provided") == 0 ||
         strcasecmp(value, "runtime_provided") == 0) {
        return CompletionSignalPolicy::kRuntimeProvided;
      }
    }

    return CompletionSignalPolicy::kMonitorProvided;
  }();
  return policy;
}

std::atomic<bool>& completion_signal_mode_negotiated() {
  static std::atomic<bool> value{false};
  return value;
}

std::atomic<uint32_t>& selected_completion_signal_mode() {
  static std::atomic<uint32_t> value{AQLMON_COMPLETION_SIGNAL_MODE_MONITOR_PROVIDED};
  return value;
}

std::atomic<uint32_t>& granted_completion_signal_capabilities() {
  static std::atomic<uint32_t> value{0};
  return value;
}

std::atomic<uint32_t>& negotiation_status() {
  static std::atomic<uint32_t> value{AQLMON_STATUS_SUCCESS};
  return value;
}

aqlmon_completion_signal_mode_t load_completion_signal_mode(
    const std::atomic<uint32_t>& value) {
  const uint32_t raw = value.load(std::memory_order_acquire);
  return to_completion_signal_mode(raw);
}

aqlmon::runtime_contract::NegotiationSnapshot snapshot_from_state() {
  aqlmon::runtime_contract::NegotiationSnapshot snapshot = {};
  snapshot.negotiated = completion_signal_mode_negotiated().load(std::memory_order_acquire);
  snapshot.status = static_cast<aqlmon_status_t>(
      negotiation_status().load(std::memory_order_acquire));
  snapshot.selected_mode = load_completion_signal_mode(selected_completion_signal_mode());
  snapshot.granted_capabilities =
      granted_completion_signal_capabilities().load(std::memory_order_acquire);
  snapshot.api_version = AQLMON_RUNTIME_CONTRACT_API_VERSION;
  return snapshot;
}

void store_snapshot(const aqlmon::runtime_contract::NegotiationSnapshot& snapshot) {
  selected_completion_signal_mode().store(static_cast<uint32_t>(snapshot.selected_mode),
                                          std::memory_order_release);
  granted_completion_signal_capabilities().store(snapshot.granted_capabilities,
                                                 std::memory_order_release);
  negotiation_status().store(static_cast<uint32_t>(snapshot.status), std::memory_order_release);
}

void write_response(
    aqlmon_runtime_negotiation_response_t* response,
    const aqlmon::runtime_contract::NegotiationSnapshot& snapshot) {
  if(response == nullptr) return;

  if(has_struct_bytes(response->size, offsetof(aqlmon_runtime_negotiation_response_t, abi_version),
                      sizeof(response->abi_version))) {
    response->abi_version = snapshot.abi_version;
  }
  if(has_struct_bytes(response->size, offsetof(aqlmon_runtime_negotiation_response_t, reserved),
                      sizeof(response->reserved))) {
    response->reserved = 0;
  }
  if(has_struct_bytes(response->size, offsetof(aqlmon_runtime_negotiation_response_t, api_version),
                      sizeof(response->api_version))) {
    response->api_version = snapshot.api_version;
  }
  if(has_struct_bytes(response->size,
                      offsetof(aqlmon_runtime_negotiation_response_t, selected_mode),
                      sizeof(response->selected_mode))) {
    response->selected_mode = static_cast<uint32_t>(snapshot.selected_mode);
  }
  if(has_struct_bytes(response->size,
                      offsetof(aqlmon_runtime_negotiation_response_t, granted_capabilities),
                      sizeof(response->granted_capabilities))) {
    response->granted_capabilities = snapshot.granted_capabilities;
  }
  if(has_struct_bytes(response->size, offsetof(aqlmon_runtime_negotiation_response_t, reserved1),
                      sizeof(response->reserved1))) {
    memset(response->reserved1, 0, sizeof(response->reserved1));
  }
}

aqlmon::runtime_contract::NegotiationSnapshot select_negotiation_result(
    const aqlmon_runtime_negotiation_request_t& request) {
  aqlmon::runtime_contract::NegotiationSnapshot snapshot = {};
  snapshot.negotiated = true;
  snapshot.selected_mode = AQLMON_COMPLETION_SIGNAL_MODE_MONITOR_PROVIDED;
  snapshot.granted_capabilities = 0;
  snapshot.status = AQLMON_STATUS_SUCCESS;

  if(request.proposed_mode == kMonitorProvidedMode) {
    return snapshot;
  }

  snapshot.status = AQLMON_STATUS_DENIED;
  if(completion_signal_policy() != CompletionSignalPolicy::kRuntimeProvided) {
    return snapshot;
  }

  const uint32_t requested_capabilities = request.proposed_capabilities & kKnownCapabilities;
  if((requested_capabilities & AQLMON_COMPLETION_SIGNAL_CAP_KERNEL_DISPATCH_SIGNALS) == 0) {
    return snapshot;
  }

  snapshot.selected_mode = AQLMON_COMPLETION_SIGNAL_MODE_RUNTIME_PROVIDED;
  snapshot.granted_capabilities = requested_capabilities;
  snapshot.status = AQLMON_STATUS_SUCCESS;
  return snapshot;
}

aqlmon_status_t status_against_request(
    const aqlmon_runtime_negotiation_request_t& request,
    const aqlmon::runtime_contract::NegotiationSnapshot& snapshot) {
  if(static_cast<uint32_t>(snapshot.selected_mode) != request.proposed_mode) {
    return AQLMON_STATUS_DENIED;
  }

  if(snapshot.selected_mode == AQLMON_COMPLETION_SIGNAL_MODE_RUNTIME_PROVIDED) {
    const uint32_t requested_capabilities = request.proposed_capabilities & kKnownCapabilities;
    if(snapshot.granted_capabilities != requested_capabilities) {
      return AQLMON_STATUS_DENIED;
    }
  }

  return snapshot.status;
}

}  // namespace

namespace aqlmon::runtime_contract {

static_assert(std::is_standard_layout_v<aqlmon_runtime_negotiation_request_t>);
static_assert(std::is_standard_layout_v<aqlmon_runtime_negotiation_response_t>);
static_assert(offsetof(aqlmon_runtime_negotiation_request_t, proposed_mode) % alignof(uint32_t) ==
              0);
static_assert(offsetof(aqlmon_runtime_negotiation_response_t, selected_mode) % alignof(uint32_t) ==
              0);

aqlmon_completion_signal_mode_t effective_completion_signal_mode() {
  return load_completion_signal_mode(selected_completion_signal_mode());
}

uint32_t effective_completion_signal_capabilities() {
  return granted_completion_signal_capabilities().load(std::memory_order_acquire);
}

}  // namespace aqlmon::runtime_contract

#if defined(__GNUC__)
#define AQLMON_RUNTIME_CONTRACT_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define AQLMON_RUNTIME_CONTRACT_EXPORT extern "C"
#endif

AQLMON_RUNTIME_CONTRACT_EXPORT uint32_t aqlmon_runtime_contract_version(void) {
  return AQLMON_RUNTIME_CONTRACT_API_VERSION;
}

AQLMON_RUNTIME_CONTRACT_EXPORT uint32_t aqlmon_runtime_contract_api_version(void) {
  return AQLMON_RUNTIME_CONTRACT_API_VERSION;
}

AQLMON_RUNTIME_CONTRACT_EXPORT uint32_t aqlmon_runtime_contract_abi_version(void) {
  return AQLMON_RUNTIME_CONTRACT_ABI_VERSION;
}

AQLMON_RUNTIME_CONTRACT_EXPORT aqlmon_completion_signal_mode_t
aqlmon_runtime_completion_signal_mode(void) {
  return aqlmon::runtime_contract::effective_completion_signal_mode();
}

AQLMON_RUNTIME_CONTRACT_EXPORT uint32_t aqlmon_runtime_completion_signal_capabilities(void) {
  return aqlmon::runtime_contract::effective_completion_signal_capabilities();
}

AQLMON_RUNTIME_CONTRACT_EXPORT aqlmon_status_t aqlmon_runtime_negotiate(
    const aqlmon_runtime_negotiation_request_t* request,
    aqlmon_runtime_negotiation_response_t* response) {
  if(request == nullptr || response == nullptr) return AQLMON_STATUS_ERROR_INVALID_ARGUMENT;
  const auto supported_snapshot = snapshot_from_state();
  if(!has_struct_bytes(request->size, offsetof(aqlmon_runtime_negotiation_request_t, abi_version),
                       sizeof(request->abi_version)) ||
     !has_struct_bytes(response->size, offsetof(aqlmon_runtime_negotiation_response_t, abi_version),
                       sizeof(response->abi_version))) {
    write_response(response, supported_snapshot);
    return AQLMON_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if(request->abi_version != AQLMON_RUNTIME_CONTRACT_ABI_VERSION) {
    write_response(response, supported_snapshot);
    return AQLMON_STATUS_ERROR_VERSION_MISMATCH;
  }
  if(!request_has_required_fields(*request) || !response_has_required_fields(*response)) {
    write_response(response, supported_snapshot);
    return AQLMON_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if(!is_valid_completion_signal_mode(request->proposed_mode)) {
    write_response(response, supported_snapshot);
    return AQLMON_STATUS_ERROR_INVALID_ARGUMENT;
  }

  bool expected = false;
  if(completion_signal_mode_negotiated().compare_exchange_strong(expected, true,
                                                                 std::memory_order_acq_rel,
                                                                 std::memory_order_acquire)) {
    const auto snapshot = select_negotiation_result(*request);
    store_snapshot(snapshot);
    write_response(response, snapshot);
    return snapshot.status;
  }

  const auto snapshot = snapshot_from_state();
  write_response(response, snapshot);
  return status_against_request(*request, snapshot);
}
