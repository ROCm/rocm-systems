/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "top.hpp"
#include "utils/flags.hpp"
#include "utils/debug.hpp"
#include "rocsignal.hpp"
#include "device/rocm/rocrctx.hpp"

namespace amd::roc {

Signal::~Signal() { Hsa::signal_destroy(signal_); }

bool Signal::Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) {
  hsa_status_t status = Hsa::signal_create(init, 0, nullptr, &signal_);
  if (status != HSA_STATUS_SUCCESS) {
    return false;
  }

  ws_ = ws;
  ClPrint(amd::LOG_DEBUG, amd::LOG_AQL, "Initialize Hostcall signal=0x%zx", signal_);
  return true;
}

uint64_t Signal::Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) {
  return Hsa::signal_wait_scacquire(signal_, static_cast<hsa_signal_condition_t>(c), value, timeout,
                                    static_cast<hsa_wait_state_t>(ws_));
}

void Signal::Reset(uint64_t value) { Hsa::signal_store_screlease(signal_, value); }

// ================================================================================================
// IpcSignal implementation
// ================================================================================================

IpcSignal::~IpcSignal() {
  if (signal_.handle != 0) {
    Hsa::signal_destroy(signal_);
  }
}

bool IpcSignal::Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) {
  hsa_status_t status = Hsa::signal_create(init, 0, nullptr, HSA_AMD_SIGNAL_IPC, &signal_);
  if (status != HSA_STATUS_SUCCESS) {
    return false;
  }
  ws_ = ws;
  return true;
}

uint64_t IpcSignal::Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) {
  return Hsa::signal_wait_scacquire(signal_, static_cast<hsa_signal_condition_t>(c), value,
                                    timeout, static_cast<hsa_wait_state_t>(ws_));
}

void IpcSignal::Reset(uint64_t value) { Hsa::signal_store_screlease(signal_, value); }

uint64_t IpcSignal::Load() {
  return Hsa::signal_load_relaxed(signal_);
}

bool IpcSignal::IpcExport(void* handle, size_t handle_size) {
  if (handle_size < sizeof(hsa_amd_ipc_signal_t)) {
    return false;
  }
  hsa_amd_ipc_signal_t ipc_handle;
  hsa_status_t status = Hsa::ipc_signal_create(signal_, &ipc_handle);
  if (status != HSA_STATUS_SUCCESS) {
    return false;
  }
  memcpy(handle, &ipc_handle, sizeof(ipc_handle));
  return true;
}

bool IpcSignal::IpcImport(const void* handle, size_t handle_size) {
  if (handle_size < sizeof(hsa_amd_ipc_signal_t)) {
    return false;
  }
  hsa_amd_ipc_signal_t ipc_handle;
  memcpy(&ipc_handle, handle, sizeof(ipc_handle));

  hsa_status_t status = Hsa::ipc_signal_attach(&ipc_handle, &signal_);
  if (status != HSA_STATUS_SUCCESS) {
    return false;
  }

  ws_ = WaitState::Active;
  return true;
}

};  // namespace amd::roc