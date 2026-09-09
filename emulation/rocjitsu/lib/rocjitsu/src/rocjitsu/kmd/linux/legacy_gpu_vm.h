// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legacy_gpu_vm.h
/// @brief KFD compatibility adapter for the frontend-neutral GPU VM service.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/legacy_address_space.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace rocjitsu::amdgpu {

/// @brief Owns legacy KFD bindings supplied to a generic GpuVm.
///
/// @details Each registration owns an independent LegacyAddressSpace and is
/// shared by the translator, backing, and fault callback installed in GpuVm.
/// Revoking the handle removes future lookup immediately, while an older
/// GpuVmAccess snapshot keeps the complete compatibility binding alive.
class LegacyGpuVmAdapter {
public:
  /// @brief Bind memory whose lifetime is externally bounded by the adapter.
  LegacyGpuVmAdapter(GpuVm &gpu_vm, GpuMemory *memory);
  /// @brief Bind shared memory retained by every access snapshot.
  LegacyGpuVmAdapter(GpuVm &gpu_vm, std::shared_ptr<GpuMemory> memory);
  ~LegacyGpuVmAdapter();

  /// @brief Select the sparse backing used by subsequent registrations.
  /// @details Existing registered bindings must be revoked first. Snapshots of
  /// revoked bindings continue to retain their prior compatibility service.
  [[nodiscard]] bool set_memory(GpuMemory *memory);
  [[nodiscard]] bool set_memory(std::shared_ptr<GpuMemory> memory);

  [[nodiscard]] AddressSpaceHandle
  register_address_space(uint32_t vmid, LegacyAddressSpaceRegistration registration,
                         std::shared_ptr<void> frontend_lifetime = {});
  [[nodiscard]] AddressSpaceHandle
  register_address_space(uint32_t vmid, LegacyPageTable *page_table,
                         std::shared_mutex *page_table_mutex,
                         const uint64_t *page_table_generation = nullptr,
                         std::shared_ptr<std::shared_mutex> request_mutex = {},
                         std::shared_ptr<void> frontend_lifetime = {});
  [[nodiscard]] bool unregister_address_space(AddressSpaceHandle handle);
  [[nodiscard]] bool unregister_vmid(uint32_t vmid);

  [[nodiscard]] bool set_client_pid(AddressSpaceHandle handle, pid_t client_pid);
  [[nodiscard]] bool set_client_mem_fd(AddressSpaceHandle handle, int mem_fd);
  [[nodiscard]] bool set_passthrough(AddressSpaceHandle handle, bool passthrough);
  [[nodiscard]] bool set_fault_reporter(AddressSpaceHandle handle, MemoryFaultReporter *reporter);

  /// @brief Return the compatibility service for a currently registered VMID.
  [[nodiscard]] LegacyAddressSpace *address_space(uint32_t vmid);
  [[nodiscard]] const LegacyAddressSpace *address_space(uint32_t vmid) const;

private:
  class Binding;

  class Entry {
  public:
    AddressSpaceHandle handle;
    std::shared_ptr<Binding> binding;
  };

  static void revoke_binding(const std::shared_ptr<Binding> &binding);
  void remove_stale_locked();
  [[nodiscard]] Entry *find_locked(AddressSpaceHandle handle);
  [[nodiscard]] const Entry *find_locked(AddressSpaceHandle handle) const;

  GpuVm *gpu_vm_ = nullptr;
  std::shared_ptr<GpuMemory> memory_;
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, Entry> bindings_;
};

} // namespace rocjitsu::amdgpu
