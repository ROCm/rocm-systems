// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/physical_memory_access.h"

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "simdojo/components/pci_device.h"

namespace rocjitsu {

namespace {

amdgpu::VmAccessOutcome to_vm_outcome(simdojo::DmaAccessOutcome outcome) {
  switch (outcome) {
  case simdojo::DmaAccessOutcome::Complete:
    return amdgpu::VmAccessOutcome::Complete;
  case simdojo::DmaAccessOutcome::Unavailable:
    return amdgpu::VmAccessOutcome::Unavailable;
  case simdojo::DmaAccessOutcome::Faulted:
    return amdgpu::VmAccessOutcome::Faulted;
  case simdojo::DmaAccessOutcome::Malformed:
    return amdgpu::VmAccessOutcome::Malformed;
  }
  return amdgpu::VmAccessOutcome::Malformed;
}

} // namespace

PciPhysicalMemoryAccess::PciPhysicalMemoryAccess(PciMemoryAccess &memory)
    : memory_(&memory), session_(memory.capture_transport_session()) {}

amdgpu::VmAccessOutcome PciPhysicalMemoryAccess::read(amdgpu::VmMemoryDomain domain,
                                                      uint64_t address,
                                                      std::span<std::byte> bytes) {
  if (session_ == nullptr)
    return amdgpu::VmAccessOutcome::Unavailable;
  simdojo::PciTransportSession::OperationLease lease = session_->acquire();
  if (!lease)
    return amdgpu::VmAccessOutcome::Unavailable;

  switch (domain) {
  case amdgpu::VmMemoryDomain::System:
    return to_vm_outcome(lease.read(address, bytes));
  case amdgpu::VmMemoryDomain::Local:
    return memory_->read_vram(address, bytes) ? amdgpu::VmAccessOutcome::Complete
                                              : amdgpu::VmAccessOutcome::Faulted;
  case amdgpu::VmMemoryDomain::Compatibility:
    return amdgpu::VmAccessOutcome::Malformed;
  }
  return amdgpu::VmAccessOutcome::Malformed;
}

amdgpu::VmAccessOutcome PciPhysicalMemoryAccess::write(amdgpu::VmMemoryDomain domain,
                                                       uint64_t address,
                                                       std::span<const std::byte> bytes) {
  if (session_ == nullptr)
    return amdgpu::VmAccessOutcome::Unavailable;
  simdojo::PciTransportSession::OperationLease lease = session_->acquire();
  if (!lease)
    return amdgpu::VmAccessOutcome::Unavailable;

  switch (domain) {
  case amdgpu::VmMemoryDomain::System:
    return to_vm_outcome(lease.write(address, bytes));
  case amdgpu::VmMemoryDomain::Local:
    return memory_->write_vram(address, bytes) ? amdgpu::VmAccessOutcome::Complete
                                               : amdgpu::VmAccessOutcome::Faulted;
  case amdgpu::VmMemoryDomain::Compatibility:
    return amdgpu::VmAccessOutcome::Malformed;
  }
  return amdgpu::VmAccessOutcome::Malformed;
}

amdgpu::AtomicLoadResult PciPhysicalMemoryAccess::atomic_load(amdgpu::VmMemoryDomain domain,
                                                              uint64_t address, uint32_t width) {
  if (session_ == nullptr)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
  simdojo::PciTransportSession::OperationLease lease = session_->acquire();
  if (!lease)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
  if (domain == amdgpu::VmMemoryDomain::Compatibility ||
      (width != sizeof(uint32_t) && width != sizeof(uint64_t)) || address % width != 0) {
    return {.outcome = amdgpu::VmAccessOutcome::Malformed};
  }
  if (domain == amdgpu::VmMemoryDomain::Local)
    return memory_->atomic_load_vram(address, width);

  const simdojo::DmaAtomicLoadResult result = lease.atomic_load(address, width);
  return {.outcome = to_vm_outcome(result.outcome), .value = result.value};
}

amdgpu::VmAccessOutcome PciPhysicalMemoryAccess::atomic_store(amdgpu::VmMemoryDomain domain,
                                                              uint64_t address, uint32_t width,
                                                              uint64_t value) {
  if (session_ == nullptr)
    return amdgpu::VmAccessOutcome::Unavailable;
  simdojo::PciTransportSession::OperationLease lease = session_->acquire();
  if (!lease)
    return amdgpu::VmAccessOutcome::Unavailable;
  if (domain == amdgpu::VmMemoryDomain::Compatibility ||
      (width != sizeof(uint32_t) && width != sizeof(uint64_t)) || address % width != 0) {
    return amdgpu::VmAccessOutcome::Malformed;
  }
  if (domain == amdgpu::VmMemoryDomain::Local)
    return memory_->atomic_store_vram(address, width, value);
  return to_vm_outcome(lease.atomic_store(address, width, value));
}

amdgpu::AtomicCompareExchangeResult
PciPhysicalMemoryAccess::compare_exchange(amdgpu::VmMemoryDomain domain, uint64_t address,
                                          uint32_t width, uint64_t expected, uint64_t desired) {
  if (session_ == nullptr)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
  simdojo::PciTransportSession::OperationLease lease = session_->acquire();
  if (!lease)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable};

  if (domain == amdgpu::VmMemoryDomain::Compatibility)
    return {.outcome = amdgpu::VmAccessOutcome::Malformed};
  if (width != sizeof(uint32_t) && width != sizeof(uint64_t))
    return {.outcome = amdgpu::VmAccessOutcome::Malformed};
  if (address % width != 0)
    return {.outcome = amdgpu::VmAccessOutcome::Malformed};

  if (domain == amdgpu::VmMemoryDomain::Local)
    return memory_->compare_exchange_vram(address, width, expected, desired);

  const simdojo::DmaAtomicCompareExchangeResult result =
      lease.compare_exchange(address, width, expected, desired);
  return {.outcome = to_vm_outcome(result.outcome),
          .observed = result.observed,
          .exchanged = result.exchanged};
}

} // namespace rocjitsu
