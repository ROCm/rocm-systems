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
  // hsa_amd_signal_value_pointer() succeeds only for a busy-wait signal (BusyWaitSignal::IsType
  // check in rocr hsa_ext_amd.cpp) and refuses an interrupt signal. A default signal (no
  // attributes) is a busy-wait signal only when the runtime has no interrupt signals (WSL with AQL
  // emulation, HSA_ENABLE_INTERRUPT=0, ...); a wait on it then spins for its whole timeout
  // whatever the wait hint, so the caller has to poll it; Load() then reads the value directly.
  // Not on Windows (ROCr through the DXG thunk has no interrupt signals either): the listener's
  // short naps there would round up to a timer tick (~15.6 ms), so it keeps the blocking wait.
#if !defined(_WIN32)
  volatile hsa_signal_value_t* value_ptr = nullptr;
  if (Hsa::signal_value_pointer(signal_, &value_ptr) == HSA_STATUS_SUCCESS) {
    value_ptr_ = value_ptr;
  }
#endif
  ClPrint(amd::LOG_DEBUG, amd::LOG_AQL, "Initialize Hostcall signal=0x%zx poll_only=%d",
          signal_.handle, IsPollOnly());
  return true;
}

uint64_t Signal::Load() {
  return (value_ptr_ != nullptr) ? *value_ptr_ : Hsa::signal_load_relaxed(signal_);
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