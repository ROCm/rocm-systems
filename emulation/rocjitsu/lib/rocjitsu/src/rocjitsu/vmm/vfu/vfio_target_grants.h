// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfio_target_grants.h
/// @brief Track target-side VFIO grant admission and backing access.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "simdojo/components/pci_bar_address_decoder.h"
#include "simdojo/components/pci_device.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {

class VfioDeviceHost;

/// The event descriptor stays valid while an effect-release callback holds a
/// wake reference, independently of the VfioDeviceHost's lifetime.
class VfioQuiesceWake final {
public:
  VfioQuiesceWake();
  ~VfioQuiesceWake();
  VfioQuiesceWake(const VfioQuiesceWake &) = delete;
  VfioQuiesceWake &operator=(const VfioQuiesceWake &) = delete;
  [[nodiscard]] int fd() const noexcept { return fd_; }
  void notify() const noexcept;

private:
  int fd_ = -1;
};

/// A server-local grant for a descriptorless peer BAR DMA map. The local PCI
/// decoder obtains the BAR owner from ordinary configuration writes; the grant
/// binds that address to both server-owned transport generations.
class VfioTargetGrants final {
public:
  struct MapRequest {
    uint64_t iova = 0;
    uint64_t size = 0;
    uint64_t map_id = 0;
    uint64_t owner_offset = 0;
    uint32_t owner_gpu_ordinal = 0;
    uint32_t owner_bar = 0;
    simdojo::DmaAccess rights = simdojo::DmaAccess::None;
    uint64_t decoded_owner_generation = 0;
  };

  explicit VfioTargetGrants(std::size_t host_count);
  ~VfioTargetGrants();

  /// Bind before any host is built or served. The hosts must outlive this use.
  [[nodiscard]] bool bind(uint32_t ordinal, VfioDeviceHost &host, uint32_t gpu_id = 0);

  /// Publish a BAR window captured after a standard PCI configuration access.
  /// A changed window revokes its old grants before this call returns.
  void publish_bar_window(uint32_t ordinal,
                          std::optional<simdojo::PciBarAddressDecoder::Window> window);

  /// Bind a standard descriptorless DMA map to an unambiguous live peer BAR.
  [[nodiscard]] std::optional<uint64_t> register_standard_map(uint32_t requester,
                                                              const simdojo::DmaRegion &region);
  void unregister_standard_map(uint32_t requester, uint64_t requester_generation, uint64_t iova,
                               uint64_t size);

  void revoke_session(uint32_t ordinal, uint64_t generation);

  /// Pause both incoming and outgoing grants for one command-blind VFIO
  /// quiesce. The pause is reversible; only unregister/reset retires a grant.
  using QuiesceGates = std::vector<std::shared_ptr<amdgpu::VmTransferEffectGate>>;
  [[nodiscard]] QuiesceGates begin_quiesce(uint32_t ordinal);
  [[nodiscard]] static bool drained(const QuiesceGates &gates) noexcept;
  void end_quiesce(uint32_t ordinal);

  struct PeerSnapshot {
    uint64_t map_iova = 0;
    uint64_t map_size = 0;
    uint64_t iova = 0;
    uint64_t map_id = 0;
    uint64_t requester_generation = 0;
    uint64_t owner_generation = 0;
    uint64_t owner_offset = 0;   ///< BAR-relative offset for the target grant.
    uint64_t backing_offset = 0; ///< File-relative canonical offset for the fabric.
    uint64_t backing_id = 0;     ///< Stable owner backing identity, if available.
    uint32_t requester_ordinal = 0;
    uint32_t owner_ordinal = 0;
    uint32_t owner_gpu_id = 0;
    std::size_t length = 0;
    std::shared_ptr<std::byte> mapped_bytes;
    std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate;
  };

  enum class PeerKind : uint8_t { OrdinaryRam, Ready, Rejected };
  struct PeerPreparation {
    PeerKind kind = PeerKind::OrdinaryRam;
    simdojo::DmaAccessOutcome outcome = simdojo::DmaAccessOutcome::Complete;
    PeerSnapshot snapshot;
  };

  /// Classify an entire physical step before any effect. Exactly one live BAR
  /// grant must cover a peer request; partial or overlapping spans are denied.
  [[nodiscard]] PeerPreparation prepare_peer(uint32_t requester, uint64_t requester_generation,
                                             uint64_t iova, std::size_t length, bool to_guest);
  /// Retain every live target grant sharing a canonical backing line with the
  /// exact snapshot. A zero backing ID retains only that snapshot's grant.
  [[nodiscard]] amdgpu::MemoryIngressAdmission admit_peer(const PeerSnapshot &snapshot);
  /// A queued fabric token holds only this fail-closed weak admission handle.
  [[nodiscard]] std::function<amdgpu::MemoryIngressAdmission()>
  capture_admission(PeerSnapshot snapshot) const;

  /// One target interval whose owner's reset is held off through the transfer.
  class PreparedSegment {
  public:
    uint64_t iova = 0;
    std::size_t length = 0;
    std::shared_ptr<std::byte> mapped_bytes;
    simdojo::PciTransportSession::OperationLease owner_lease;
    amdgpu::MemoryIngressPermit effect_permit;

    void copy(std::span<std::byte> bytes, bool to_guest) const;
  };

  struct PrepareResult {
    simdojo::DmaAccessOutcome outcome = simdojo::DmaAccessOutcome::Complete;
    std::vector<PreparedSegment> segments;
  };

  /// Snapshot every overlapping target interval and acquire all owner leases
  /// before the host validates ordinary gaps or moves any bytes.
  [[nodiscard]] PrepareResult prepare(uint32_t requester, uint64_t requester_generation,
                                      uint64_t iova, std::size_t length, bool to_guest);

private:
  /// Insert a grant only after libvfio-user has inserted the exact DMA region.
  /// Returns the requester's generation for its matching unregister callback.
  [[nodiscard]] std::optional<uint64_t> register_map(uint32_t requester, const MapRequest &request);
  void unregister_map(uint32_t requester, uint64_t requester_generation, const MapRequest &request);

  struct Grant {
    MapRequest map;
    uint64_t requester_generation = 0;
    uint64_t owner_generation = 0;
    // Keep the canonical coordinate separate from the BAR-relative map.
    uint64_t backing_offset = 0;
    uint64_t backing_id = 0;
    std::shared_ptr<std::byte> mapped_bytes;
    std::weak_ptr<simdojo::PciTransportSession> requester_session;
    std::weak_ptr<simdojo::PciTransportSession> owner_session;
    std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate;
  };

  struct AdmissionLifetime {
    std::mutex mutex;
    VfioTargetGrants *owner = nullptr;
  };

  [[nodiscard]] amdgpu::MemoryIngressPermit
  retain_effect(std::optional<amdgpu::VmTransferEffectGate::Permit> permit, uint32_t requester,
                uint32_t owner);

  using IovaKey = std::pair<uint32_t, uint64_t>;
  using SessionKey = std::pair<uint32_t, uint64_t>;

  mutable std::mutex mutex_;
  std::vector<VfioDeviceHost *> hosts_;
  std::vector<std::weak_ptr<VfioQuiesceWake>> wakes_;
  std::vector<uint32_t> gpu_ids_;
  std::vector<bool> quiescing_;
  std::map<IovaKey, Grant> grants_;
  simdojo::PciBarAddressDecoder address_decoder_;
  uint64_t next_standard_map_id_ = 1;
  // A requester's IDs increase throughout its session, including after UNMAP.
  // Retaining one high-water mark per session bounds replay protection.
  std::map<SessionKey, uint64_t> highest_map_ids_;
  // An unmapped target IOVA remains known for its requester's session. A stale
  // GPU PTE must not reinterpret it as ordinary RAM after same-IOVA remapping.
  std::map<SessionKey, std::map<uint64_t, uint64_t>> known_peer_ranges_;
  std::shared_ptr<AdmissionLifetime> admission_lifetime_;
};

} // namespace rocjitsu
