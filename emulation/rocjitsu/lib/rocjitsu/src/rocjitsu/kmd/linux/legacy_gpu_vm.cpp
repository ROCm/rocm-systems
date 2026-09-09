// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

constexpr VmAccessOutcome vm_access_outcome(CopyOutcome outcome) {
  switch (outcome) {
  case CopyOutcome::Complete:
    return VmAccessOutcome::Complete;
  case CopyOutcome::Unavailable:
    return VmAccessOutcome::Unavailable;
  case CopyOutcome::Faulted:
    return VmAccessOutcome::Faulted;
  }
  return VmAccessOutcome::Malformed;
}

} // namespace

class LegacyGpuVmAdapter::Binding final : public AddressSpaceTranslator,
                                          public PhysicalMemoryAccess {
public:
  Binding(std::shared_ptr<GpuMemory> memory, uint32_t vmid,
          LegacyAddressSpaceRegistration registration, std::shared_ptr<void> frontend_lifetime)
      : memory_(std::move(memory)), address_space_(std::make_shared<LegacyAddressSpace>(*memory_)),
        vmid_(vmid), frontend_lifetime_(std::move(frontend_lifetime)) {
    address_space_->register_process(vmid_, registration.page_table, registration.page_table_mutex,
                                     registration.page_table_generation,
                                     std::move(registration.request_mutex));
    address_space_->set_process_client_pid(vmid_, registration.client_pid);
    address_space_->set_process_mem_fd(vmid_, registration.client_mem_fd);
    address_space_->set_process_passthrough(vmid_, registration.passthrough);
    address_space_->set_process_fault_reporter(vmid_, registration.fault_reporter);
  }

  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind /*access*/) const override {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    constexpr uint64_t kPageBytes = 4096;
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation =
            {
                .domain = VmMemoryDomain::Compatibility,
                .address = address,
                .contiguous_bytes = kPageBytes - (address & (kPageBytes - 1)),
                .mtype = address_space_->pte_mtype(address, vmid_),
                .permissions = {.readable = true, .writable = true, .executable = true},
            },
    };
  }

  [[nodiscard]] VmTranslationResult probe_translation(uint64_t address, std::size_t size,
                                                      VmAccessKind access) const override {
    const bool accessible = access == VmAccessKind::Execute
                                ? address_space_->is_fetchable(address, vmid_)
                                : address_space_->is_range_mapped(address, size, vmid_);
    if (!accessible)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return translate(address, size, access);
  }

  [[nodiscard]] VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                                     std::span<std::byte> bytes) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    return vm_access_outcome(address_space_->read_block_strict(
        address, std::span<uint8_t>(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size()),
        vmid_));
  }

  [[nodiscard]] VmAccessOutcome read_for_access(VmMemoryDomain domain, uint64_t address,
                                                std::span<std::byte> bytes,
                                                VmAccessKind access) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    if (access != VmAccessKind::Execute)
      return read(domain, address, bytes);
    return address_space_->read_block(
               address, std::span<uint8_t>(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size()),
               vmid_) == AccessOutcome::Complete
               ? VmAccessOutcome::Complete
               : VmAccessOutcome::Faulted;
  }

  [[nodiscard]] VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                                      std::span<const std::byte> bytes) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    return vm_access_outcome(address_space_->write_block_strict(
        address,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()),
        vmid_));
  }

  [[nodiscard]] AtomicLoadResult atomic_load(VmMemoryDomain domain, uint64_t address,
                                             uint32_t width) override {
    if (domain != VmMemoryDomain::Compatibility)
      return {.outcome = VmAccessOutcome::Malformed, .value = 0};
    uint64_t value = 0;
    const VmAccessOutcome outcome =
        vm_access_outcome(address_space_->atomic_load(address, width, value, vmid_));
    return {.outcome = outcome, .value = outcome == VmAccessOutcome::Complete ? value : 0};
  }

  [[nodiscard]] VmAccessOutcome atomic_store(VmMemoryDomain domain, uint64_t address,
                                             uint32_t width, uint64_t value) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    return address_space_->atomic_store(address, width, value, vmid_) == AccessOutcome::Complete
               ? VmAccessOutcome::Complete
               : VmAccessOutcome::Faulted;
  }

  [[nodiscard]] AtomicCompareExchangeResult compare_exchange(VmMemoryDomain domain,
                                                             uint64_t address, uint32_t width,
                                                             uint64_t expected,
                                                             uint64_t desired) override {
    if (domain != VmMemoryDomain::Compatibility)
      return {.outcome = VmAccessOutcome::Malformed};
    uint64_t observed = 0;
    bool exchanged = false;
    const AccessOutcome outcome = address_space_->atomic_compare_exchange(
        address, width, expected, desired, observed, exchanged, vmid_);
    return {.outcome = outcome == AccessOutcome::Complete ? VmAccessOutcome::Complete
                                                          : VmAccessOutcome::Faulted,
            .observed = observed,
            .exchanged = exchanged};
  }

  [[nodiscard]] VmAccessOutcome atomic_modify(VmMemoryDomain domain, uint64_t address,
                                              uint32_t width,
                                              const AtomicMutation &mutation) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    const AccessOutcome outcome = address_space_->atomic_modify_strict(
        address, width,
        [&](uint8_t *bytes) {
          mutation(std::span<std::byte>(reinterpret_cast<std::byte *>(bytes), width));
        },
        vmid_);
    return outcome == AccessOutcome::Complete ? VmAccessOutcome::Complete
                                              : VmAccessOutcome::Faulted;
  }

  [[nodiscard]] std::byte *resolve_host_pointer(VmMemoryDomain domain, uint64_t address,
                                                std::size_t size) const override {
    if (domain != VmMemoryDomain::Compatibility)
      return nullptr;
    return reinterpret_cast<std::byte *>(address_space_->resolve_host_ptr(address, vmid_, size));
  }

  [[nodiscard]] std::pair<uint64_t, uint64_t> host_range(VmMemoryDomain domain,
                                                         uint64_t address) const override {
    if (domain != VmMemoryDomain::Compatibility)
      return {0, 0};
    return address_space_->find_host_range(address, vmid_);
  }

  void report_fault(uint64_t address) const { address_space_->report_vm_fault(vmid_, address); }
  void set_client_pid(pid_t client_pid) {
    address_space_->set_process_client_pid(vmid_, client_pid);
  }
  void set_client_mem_fd(int mem_fd) { address_space_->set_process_mem_fd(vmid_, mem_fd); }
  void set_passthrough(bool passthrough) {
    address_space_->set_process_passthrough(vmid_, passthrough);
  }
  void set_fault_reporter(MemoryFaultReporter *reporter) {
    address_space_->set_process_fault_reporter(vmid_, reporter);
  }
  LegacyAddressSpace &address_space() { return *address_space_; }
  const LegacyAddressSpace &address_space() const { return *address_space_; }

private:
  std::shared_ptr<GpuMemory> memory_;
  std::shared_ptr<LegacyAddressSpace> address_space_;
  uint32_t vmid_ = 0;
  std::shared_ptr<void> frontend_lifetime_;
};

LegacyGpuVmAdapter::LegacyGpuVmAdapter(GpuVm &gpu_vm, GpuMemory *memory)
    : LegacyGpuVmAdapter(gpu_vm, std::shared_ptr<GpuMemory>(memory, [](GpuMemory *) {
                           // The topology owns this backing and outlives its KFD adapter.
                         })) {}

LegacyGpuVmAdapter::LegacyGpuVmAdapter(GpuVm &gpu_vm, std::shared_ptr<GpuMemory> memory)
    : gpu_vm_(&gpu_vm), memory_(std::move(memory)) {}

LegacyGpuVmAdapter::~LegacyGpuVmAdapter() {
  std::lock_guard lock(mutex_);
  for (const auto &[vmid, entry] : bindings_) {
    (void)vmid;
    (void)gpu_vm_->unregister_address_space(entry.handle);
    revoke_binding(entry.binding);
  }
}

bool LegacyGpuVmAdapter::set_memory(GpuMemory *memory) {
  return set_memory(std::shared_ptr<GpuMemory>(memory, [](GpuMemory *) {
    // The topology owns this backing and outlives its KFD adapter.
  }));
}

bool LegacyGpuVmAdapter::set_memory(std::shared_ptr<GpuMemory> memory) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  if (!bindings_.empty())
    return false;
  memory_ = memory;
  return true;
}

AddressSpaceHandle
LegacyGpuVmAdapter::register_address_space(uint32_t vmid,
                                           LegacyAddressSpaceRegistration registration,
                                           std::shared_ptr<void> frontend_lifetime) {
  if (registration.page_table == nullptr || registration.page_table_mutex == nullptr)
    return {};
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  if (memory_ == nullptr || bindings_.contains(vmid))
    return {};
  auto binding = std::make_shared<Binding>(memory_, vmid, std::move(registration),
                                           std::move(frontend_lifetime));
  const AddressSpaceHandle handle = gpu_vm_->register_address_space(
      vmid, binding, binding,
      [binding](uint64_t address, VmAccessKind) { binding->report_fault(address); }, true);
  if (!handle)
    return {};
  bindings_.emplace(vmid, Entry{.handle = handle, .binding = std::move(binding)});
  return handle;
}

AddressSpaceHandle LegacyGpuVmAdapter::register_address_space(
    uint32_t vmid, LegacyPageTable *page_table, std::shared_mutex *page_table_mutex,
    const uint64_t *page_table_generation, std::shared_ptr<std::shared_mutex> request_mutex,
    std::shared_ptr<void> frontend_lifetime) {
  return register_address_space(vmid,
                                {.page_table = page_table,
                                 .page_table_mutex = page_table_mutex,
                                 .page_table_generation = page_table_generation,
                                 .request_mutex = std::move(request_mutex)},
                                std::move(frontend_lifetime));
}

bool LegacyGpuVmAdapter::unregister_address_space(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  const auto found = std::ranges::find_if(
      bindings_, [handle](const auto &entry) { return entry.second.handle == handle; });
  if (found == bindings_.end() || !gpu_vm_->unregister_address_space(handle))
    return false;
  revoke_binding(found->second.binding);
  bindings_.erase(found);
  return true;
}

bool LegacyGpuVmAdapter::unregister_vmid(uint32_t vmid) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  const auto found = bindings_.find(vmid);
  if (found == bindings_.end() || !gpu_vm_->unregister_address_space(found->second.handle))
    return false;
  revoke_binding(found->second.binding);
  bindings_.erase(found);
  return true;
}

bool LegacyGpuVmAdapter::set_client_pid(AddressSpaceHandle handle, pid_t client_pid) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_client_pid(client_pid);
  return true;
}

bool LegacyGpuVmAdapter::set_client_mem_fd(AddressSpaceHandle handle, int mem_fd) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_client_mem_fd(mem_fd);
  return true;
}

bool LegacyGpuVmAdapter::set_passthrough(AddressSpaceHandle handle, bool passthrough) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_passthrough(passthrough);
  return true;
}

bool LegacyGpuVmAdapter::set_fault_reporter(AddressSpaceHandle handle,
                                            MemoryFaultReporter *reporter) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_fault_reporter(reporter);
  return true;
}

LegacyAddressSpace *LegacyGpuVmAdapter::address_space(uint32_t vmid) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  const auto found = bindings_.find(vmid);
  return found == bindings_.end() ? nullptr : &found->second.binding->address_space();
}

const LegacyAddressSpace *LegacyGpuVmAdapter::address_space(uint32_t vmid) const {
  std::lock_guard lock(mutex_);
  const auto found = bindings_.find(vmid);
  return found == bindings_.end() || !gpu_vm_->lookup(found->second.handle).has_value()
             ? nullptr
             : &found->second.binding->address_space();
}

void LegacyGpuVmAdapter::revoke_binding(const std::shared_ptr<Binding> &binding) {
  if (binding != nullptr)
    binding->set_fault_reporter(nullptr);
}

void LegacyGpuVmAdapter::remove_stale_locked() {
  for (auto entry = bindings_.begin(); entry != bindings_.end();) {
    if (gpu_vm_->lookup(entry->second.handle).has_value()) {
      ++entry;
      continue;
    }
    revoke_binding(entry->second.binding);
    entry = bindings_.erase(entry);
  }
}

LegacyGpuVmAdapter::Entry *LegacyGpuVmAdapter::find_locked(AddressSpaceHandle handle) {
  for (auto &[vmid, entry] : bindings_) {
    (void)vmid;
    if (entry.handle == handle)
      return &entry;
  }
  return nullptr;
}

const LegacyGpuVmAdapter::Entry *LegacyGpuVmAdapter::find_locked(AddressSpaceHandle handle) const {
  for (const auto &[vmid, entry] : bindings_) {
    (void)vmid;
    if (entry.handle == handle)
      return &entry;
  }
  return nullptr;
}

} // namespace rocjitsu::amdgpu
