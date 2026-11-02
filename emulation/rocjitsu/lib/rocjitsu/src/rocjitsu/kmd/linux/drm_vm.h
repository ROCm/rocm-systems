// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/kmd/linux/kfd_address_space.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace rocjitsu {

/// @brief One DRM file's translation state and registration on a GPU.
/// @details The runtime's GpuVm and GpuMemory must outlive this object. File cleanup
/// retires queues before releasing the VM, and runtime teardown drains files
/// before destroying the GPU services.
class DrmVm {
public:
  DrmVm(amdgpu::GpuVm &gpu_vm, amdgpu::GpuMemory *backing, uint32_t ordinal, uint32_t vmid,
        bool passthrough, std::shared_ptr<amdgpu::MemoryFaultReporter> reporter = {})
      : gpu_ordinal(ordinal), gpu_vm_(&gpu_vm), reporter_(std::move(reporter)),
        adapter_(gpu_vm, backing) {
    // Snapshots retain the table and reporter without retaining this VM, which
    // would create a VM -> adapter -> binding -> VM ownership cycle.
    struct BindingLifetime {
      std::shared_ptr<KfdAddressSpace> memory;
      std::shared_ptr<amdgpu::MemoryFaultReporter> reporter;
    };
    auto lifetime = std::make_shared<BindingLifetime>(BindingLifetime{memory, reporter_});
    address_space_ =
        adapter_.register_address_space(vmid,
                                        {.page_table = &memory->page_table_,
                                         .page_table_mutex = &memory->page_table_mutex_,
                                         .page_table_generation = memory->page_table_generation(),
                                         .request_mutex = memory->page_table_request_mutex(),
                                         .mutation_epoch = memory->page_table_mutation_epoch(),
                                         .page_table_cache_state = memory->page_table_cache_state(),
                                         .passthrough = passthrough,
                                         .fault_reporter = lifetime->reporter.get()},
                                        lifetime);
  }

  // Declared before the adapter so its destructor revokes the registration
  // before the VM releases its own translation state.
  const std::shared_ptr<KfdAddressSpace> memory = std::make_shared<KfdAddressSpace>();
  const uint32_t gpu_ordinal;

  amdgpu::AddressSpaceHandle address_space() const { return address_space_; }
  bool belongs_to(const amdgpu::GpuVm &gpu_vm) const {
    return address_space_ && gpu_vm_ == &gpu_vm;
  }
  amdgpu::LegacyGpuVmAdapter &adapter() { return adapter_; }
  amdgpu::MemoryFaultReporter *fault_reporter() const { return reporter_.get(); }

private:
  amdgpu::GpuVm *gpu_vm_;
  const std::shared_ptr<amdgpu::MemoryFaultReporter> reporter_;
  amdgpu::LegacyGpuVmAdapter adapter_;
  amdgpu::AddressSpaceHandle address_space_;
};

} // namespace rocjitsu
