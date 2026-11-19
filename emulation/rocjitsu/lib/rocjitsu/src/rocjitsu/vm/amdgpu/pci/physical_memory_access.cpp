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
    : memory_(&memory), session_(memory.capture_transport_session()) {
  local_cache_coherent_ = memory.supports_translated_local_cache();
  if (session_ != nullptr) {
    auto lease = session_->acquire();
    peer_enabled_ = lease && dynamic_cast<PciPeerDataProvider *>(lease.dma()) != nullptr;
  }
}

amdgpu::MemoryIngressAdmission PciPhysicalMemoryAccess::admit_pt_rewalk() const {
  if (session_ == nullptr)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
  simdojo::PciTransportSession::OperationLease session_lease = session_->acquire();
  if (!session_lease)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
  const amdgpu::MemoryPtEpoch epoch{.generation = session_->generation(),
                                    .session_identity = session_};
  // MemoryIngressPermit uses a copyable release callback. Retain the move-only
  // transport lease until the walk has finished.
  auto retained =
      std::make_shared<simdojo::PciTransportSession::OperationLease>(std::move(session_lease));
  return {.permit = amdgpu::MemoryIngressPermit([retained = std::move(retained)] {}, epoch)};
}

amdgpu::MemoryIngressAdmission
PciPhysicalMemoryAccess::admit_pt_rewalk(const amdgpu::MemoryPtEpoch &expected) const {
  if (session_ == nullptr || expected.generation != session_->generation() ||
      expected.session_identity != session_)
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  simdojo::PciTransportSession::OperationLease session_lease = session_->acquire();
  if (!session_lease)
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  auto retained =
      std::make_shared<simdojo::PciTransportSession::OperationLease>(std::move(session_lease));
  return {.permit = amdgpu::MemoryIngressPermit([retained = std::move(retained)] {}, expected)};
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_deferred_system_read(
    uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate) {
  return prepare_peer_read(address, length, access, gate, nullptr, 0, nullptr, {});
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_deferred_system_read(
    uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
    const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
    const amdgpu::VmTranslation &selected_translation) {
  return prepare_peer_read(address, length, access, gate, &executing_access, gpu_va,
                           &selected_translation, {});
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_deferred_system_read(
    uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
    const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
    const amdgpu::VmTranslation &selected_translation,
    const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) {
  return prepare_peer_read(address, length, access, gate, &executing_access, gpu_va,
                           &selected_translation, pt_epoch);
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_peer_read(
    uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
    const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
    const amdgpu::VmTranslation *selected_translation,
    const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) {
  if (!peer_enabled_ || access != amdgpu::VmAccessKind::Read || session_ == nullptr)
    return std::nullopt;
  auto lease = session_->acquire();
  if (!lease)
    return std::nullopt;
  auto *provider = dynamic_cast<PciPeerDataProvider *>(lease.dma());
  return provider != nullptr
             ? provider->prepare_peer_read(session_->generation(), address, length, gate,
                                           executing_access, gpu_va, selected_translation, pt_epoch)
             : std::nullopt;
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_deferred_system_write(
    uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate) {
  return prepare_peer_write(address, bytes, gate, nullptr, 0, nullptr, {});
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_deferred_system_write(
    uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
    const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
    const amdgpu::VmTranslation &selected_translation) {
  return prepare_peer_write(address, bytes, gate, &executing_access, gpu_va, &selected_translation,
                            {});
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_deferred_system_write(
    uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
    const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
    const amdgpu::VmTranslation &selected_translation,
    const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) {
  return prepare_peer_write(address, bytes, gate, &executing_access, gpu_va, &selected_translation,
                            pt_epoch);
}

std::optional<amdgpu::VmTransferToken> PciPhysicalMemoryAccess::prepare_peer_write(
    uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
    const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
    const amdgpu::VmTranslation *selected_translation,
    const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) {
  if (!peer_enabled_ || session_ == nullptr)
    return std::nullopt;
  auto lease = session_->acquire();
  if (!lease)
    return std::nullopt;
  auto *provider = dynamic_cast<PciPeerDataProvider *>(lease.dma());
  return provider != nullptr ? provider->prepare_peer_write(session_->generation(), address, bytes,
                                                            gate, executing_access, gpu_va,
                                                            selected_translation, pt_epoch)
                             : std::nullopt;
}

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
      (width != sizeof(uint16_t) && width != sizeof(uint32_t) && width != sizeof(uint64_t)) ||
      address % width != 0) {
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
