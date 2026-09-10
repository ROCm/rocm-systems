// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <functional>
#include <span>
#include <unordered_map>

#if defined(__SANITIZE_ADDRESS__)
#define RJ_LEGACY_GPU_MEMORY_FIXTURE_WITH_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RJ_LEGACY_GPU_MEMORY_FIXTURE_WITH_ASAN 1
#endif
#endif

namespace rocjitsu::amdgpu {

/// @brief Test-only access to legacy compatibility implementation details.
class LegacyAddressSpaceTestAccess {
public:
  static std::mutex *backing_atomic_mutex_for(const void *address) {
    return &LegacyAddressSpace::backing_atomic_mutex(reinterpret_cast<uintptr_t>(address));
  }

  static uint64_t rejected_identity_accesses(const LegacyAddressSpace &address_space) {
    return address_space.rejected_identity_accesses_.load(std::memory_order_relaxed);
  }

  static PageWritability page_writability(const uint8_t *page) {
    return LegacyAddressSpace::host_page_writability(page);
  }

  static void set_default_passthrough(LegacyAddressSpace &address_space, bool passthrough) {
    address_space.set_default_passthrough(passthrough);
  }

#if defined(RJ_LEGACY_GPU_MEMORY_FIXTURE_WITH_ASAN)
  static void set_page_table_unlocked_hook(LegacyAddressSpace &address_space,
                                           std::function<void()> *hook) {
    address_space.asan_page_table_unlocked_hook_.store(hook, std::memory_order_release);
  }

  static constexpr size_t metadata_retry_limit() { return LegacyAddressSpace::kMaxMetadataRetries; }
#endif
};

} // namespace rocjitsu::amdgpu

namespace rocjitsu::test {

/// @brief Test-only owner that keeps legacy mappings under a real GpuVm.
class LegacyGpuMemoryFixture : public amdgpu::GpuMemory {
public:
  explicit LegacyGpuMemoryFixture(std::string name)
      : GpuMemory(std::move(name)), direct_address_space_(*this), legacy_vm_(gpu_vm_, this) {}

  using GpuMemory::read32;
  using GpuMemory::write32;

  void register_process(uint32_t vmid, amdgpu::LegacyPageTable *page_table,
                        std::shared_mutex *page_table_mutex,
                        const uint64_t *page_table_generation = nullptr,
                        std::shared_ptr<std::shared_mutex> request_mutex = {}) {
    unregister_process(vmid);
    const amdgpu::AddressSpaceHandle handle =
        legacy_vm_.register_address_space(vmid, {.page_table = page_table,
                                                 .page_table_mutex = page_table_mutex,
                                                 .page_table_generation = page_table_generation,
                                                 .request_mutex = std::move(request_mutex),
                                                 .client_pid = client_pids_[vmid],
                                                 .client_mem_fd = -1,
                                                 .passthrough = passthrough_,
                                                 .fault_reporter = fault_reporter_});
    if (handle) {
      handles_[vmid] = handle;
      if (amdgpu::LegacyAddressSpace *address_space = legacy_vm_.address_space(vmid))
        amdgpu::LegacyAddressSpaceTestAccess::set_default_passthrough(*address_space, passthrough_);
    }
  }

  void unregister_process(uint32_t vmid) {
    auto handle = handles_.find(vmid);
    if (handle == handles_.end())
      return;
    (void)legacy_vm_.unregister_address_space(handle->second);
    handles_.erase(handle);
  }

  void set_process_client_pid(uint32_t vmid, pid_t client_pid) {
    client_pids_[vmid] = client_pid;
    auto handle = handles_.find(vmid);
    if (handle != handles_.end())
      (void)legacy_vm_.set_client_pid(handle->second, client_pid);
  }

  void set_process_mem_fd(uint32_t vmid, int mem_fd) {
    auto handle = handles_.find(vmid);
    if (handle != handles_.end())
      (void)legacy_vm_.set_client_mem_fd(handle->second, mem_fd);
  }

  void set_passthrough(bool passthrough) {
    passthrough_ = passthrough;
    amdgpu::LegacyAddressSpaceTestAccess::set_default_passthrough(direct_address_space_,
                                                                  passthrough);
    for (const auto &[vmid, handle] : handles_) {
      (void)legacy_vm_.set_passthrough(handle, passthrough);
      if (amdgpu::LegacyAddressSpace *address_space = legacy_vm_.address_space(vmid))
        amdgpu::LegacyAddressSpaceTestAccess::set_default_passthrough(*address_space, passthrough);
    }
  }

  [[nodiscard]] bool passthrough() const { return passthrough_; }

  void set_memory_fault_reporter(amdgpu::MemoryFaultReporter *reporter) {
    fault_reporter_ = reporter;
    for (const auto &[vmid, handle] : handles_) {
      (void)vmid;
      (void)legacy_vm_.set_fault_reporter(handle, reporter);
    }
  }

  amdgpu::GpuVm &gpu_vm() { return gpu_vm_; }

  amdgpu::LegacyAddressSpace &legacy_address_space(uint32_t vmid = 0) {
    if (amdgpu::LegacyAddressSpace *address_space = legacy_vm_.address_space(vmid))
      return *address_space;
    if (vmid == 0 && !handles_.empty())
      return *legacy_vm_.address_space(handles_.begin()->first);
    return direct_address_space_;
  }
  const amdgpu::LegacyAddressSpace &legacy_address_space(uint32_t vmid = 0) const {
    if (const amdgpu::LegacyAddressSpace *address_space = legacy_vm_.address_space(vmid))
      return *address_space;
    if (vmid == 0 && !handles_.empty())
      return *legacy_vm_.address_space(handles_.begin()->first);
    return direct_address_space_;
  }

  operator amdgpu::LegacyAddressSpace &() { return legacy_address_space(); }
  operator const amdgpu::LegacyAddressSpace &() const { return legacy_address_space(); }

  uint8_t *resolve_host_ptr(uint64_t addr, uint32_t vmid = 0, size_t size = 1) const {
    return legacy_address_space(vmid).resolve_host_ptr(addr, vmid, size);
  }

  uint8_t *translate_debug(uint64_t addr, uint32_t vmid, size_t size = 1) const {
    return legacy_address_space(vmid).translate_debug(addr, vmid, size);
  }

  std::pair<uint64_t, uint64_t> find_host_range(uint64_t addr, uint32_t vmid) const {
    return legacy_address_space(vmid).find_host_range(addr, vmid);
  }

  amdgpu::Mtype pte_mtype(uint64_t addr, uint32_t vmid = 0) const {
    return legacy_address_space(vmid).pte_mtype(addr, vmid);
  }

  bool has_page_mapping(uint64_t addr, uint32_t vmid = 0) const {
    return legacy_address_space(vmid).has_page_mapping(addr, vmid);
  }

  bool has_range_mapping(uint64_t addr, size_t size, uint32_t vmid = 0) const {
    return legacy_address_space(vmid).has_range_mapping(addr, size, vmid);
  }

  bool is_mapped(uint64_t addr, uint32_t vmid = 0) const {
    return legacy_address_space(vmid).is_mapped(addr, vmid);
  }

  bool is_range_mapped(uint64_t addr, size_t size, uint32_t vmid = 0) const {
    return legacy_address_space(vmid).is_range_mapped(addr, size, vmid);
  }

  bool is_fetchable(uint64_t addr, uint32_t vmid = 0) const {
    return legacy_address_space(vmid).is_fetchable(addr, vmid);
  }

  amdgpu::AccessOutcome check_range(uint64_t addr, size_t size, uint32_t vmid) {
    return legacy_address_space(vmid).check_range(addr, size, vmid);
  }

  uint8_t read8(uint64_t addr, uint32_t vmid) const {
    uint8_t value = 0;
    (void)legacy_address_space(vmid).read_block(addr, std::span<uint8_t>(&value, 1), vmid);
    return value;
  }

  uint32_t read32(uint64_t addr, uint32_t vmid) const {
    uint32_t value = 0;
    (void)legacy_address_space(vmid).read_block(
        addr, std::span<uint8_t>(reinterpret_cast<uint8_t *>(&value), sizeof(value)), vmid);
    return value;
  }

  uint64_t read64(uint64_t addr, uint32_t vmid) const {
    uint64_t value = 0;
    (void)legacy_address_space(vmid).read_block(
        addr, std::span<uint8_t>(reinterpret_cast<uint8_t *>(&value), sizeof(value)), vmid);
    return value;
  }

  void write8(uint64_t addr, uint8_t value, uint32_t vmid) {
    (void)legacy_address_space(vmid).write_block(addr, std::span<const uint8_t>(&value, 1), vmid);
  }

  void write32(uint64_t addr, uint32_t value, uint32_t vmid) {
    (void)legacy_address_space(vmid).write_block(
        addr, std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&value), sizeof(value)),
        vmid);
  }

  void write64(uint64_t addr, uint64_t value, uint32_t vmid) {
    (void)legacy_address_space(vmid).write_block(
        addr, std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&value), sizeof(value)),
        vmid);
  }

  amdgpu::AccessOutcome read_block(uint64_t addr, std::span<uint8_t> bytes, uint32_t vmid) const {
    return legacy_address_space(vmid).read_block(addr, bytes, vmid);
  }

  amdgpu::AccessOutcome write_block(uint64_t addr, std::span<const uint8_t> bytes, uint32_t vmid) {
    return legacy_address_space(vmid).write_block(addr, bytes, vmid);
  }

  amdgpu::CopyOutcome copy_block(uint64_t destination, uint64_t source, size_t size,
                                 uint32_t vmid = 0) {
    return legacy_address_space(vmid).copy_block(destination, source, size, vmid);
  }

  template <typename F>
  void atomic_rmw(uint64_t addr, uint32_t size, F &&mutation, uint32_t vmid = 0) {
    legacy_address_space(vmid).atomic_rmw(addr, size, std::forward<F>(mutation), vmid);
  }

  amdgpu::AccessOutcome atomic_fetch_add64(uint64_t addr, uint64_t amount, uint32_t vmid) {
    return legacy_address_space(vmid).atomic_fetch_add64(addr, amount, vmid);
  }

  amdgpu::AccessOutcome atomic_fetch_sub64(uint64_t addr, int64_t amount, uint32_t vmid) {
    return legacy_address_space(vmid).atomic_fetch_sub64(addr, amount, vmid);
  }

  amdgpu::CopyOutcome atomic_load(uint64_t addr, uint32_t size, uint64_t &value,
                                  uint32_t vmid) const {
    return legacy_address_space(vmid).atomic_load(addr, size, value, vmid);
  }

  amdgpu::AccessOutcome atomic_store(uint64_t addr, uint32_t size, uint64_t value, uint32_t vmid) {
    return legacy_address_space(vmid).atomic_store(addr, size, value, vmid);
  }

  amdgpu::AccessOutcome atomic_compare_exchange(uint64_t addr, uint32_t size, uint64_t expected,
                                                uint64_t desired, uint64_t &observed,
                                                bool &exchanged, uint32_t vmid) {
    return legacy_address_space(vmid).atomic_compare_exchange(addr, size, expected, desired,
                                                              observed, exchanged, vmid);
  }

  [[nodiscard]] uint64_t rejected_identity_accesses() const {
    return amdgpu::LegacyAddressSpaceTestAccess::rejected_identity_accesses(legacy_address_space());
  }

#if defined(RJ_LEGACY_GPU_MEMORY_FIXTURE_WITH_ASAN)
  void set_page_table_unlocked_hook(std::function<void()> *hook) {
    amdgpu::LegacyAddressSpaceTestAccess::set_page_table_unlocked_hook(legacy_address_space(),
                                                                       hook);
  }

  static constexpr size_t metadata_retry_limit() {
    return amdgpu::LegacyAddressSpaceTestAccess::metadata_retry_limit();
  }
#endif

private:
  amdgpu::GpuVm gpu_vm_;
  amdgpu::LegacyAddressSpace direct_address_space_;
  amdgpu::LegacyGpuVmAdapter legacy_vm_;
  std::unordered_map<uint32_t, amdgpu::AddressSpaceHandle> handles_;
  std::unordered_map<uint32_t, pid_t> client_pids_;
  bool passthrough_ = false;
  amdgpu::MemoryFaultReporter *fault_reporter_ = nullptr;
};

} // namespace rocjitsu::test

#if defined(RJ_LEGACY_GPU_MEMORY_FIXTURE_WITH_ASAN)
#undef RJ_LEGACY_GPU_MEMORY_FIXTURE_WITH_ASAN
#endif
