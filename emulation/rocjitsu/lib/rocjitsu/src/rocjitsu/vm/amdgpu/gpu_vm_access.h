// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_vm_access.h
/// @brief Retained GPU VM access, transfer results and physical memory operations.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/prt_residency.h"
#include "rocjitsu/vm/amdgpu/vm_mutation_epoch.h"
#include "util/intrusive_shared_ptr.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace rocjitsu::amdgpu {

class GpuVm;
class GpuVmAccess;
class GpuVmAccessState;

/// @brief Cache immutable VM snapshots for one bounded execution quantum.
/// @details Nested guards share the outer cache. Borrowed objects remain alive
/// until the outer guard exits, but each operation independently checks and
/// leases its access state. Revoked entries are skipped on subsequent lookups.
class GpuVmAccessBatchGuard {
public:
  GpuVmAccessBatchGuard();
  ~GpuVmAccessBatchGuard();

  [[nodiscard]] static bool active();

  GpuVmAccessBatchGuard(const GpuVmAccessBatchGuard &) = delete;
  GpuVmAccessBatchGuard &operator=(const GpuVmAccessBatchGuard &) = delete;
  GpuVmAccessBatchGuard(GpuVmAccessBatchGuard &&) = delete;
  GpuVmAccessBatchGuard &operator=(GpuVmAccessBatchGuard &&) = delete;

private:
  friend class GpuVm;

  [[nodiscard]] static const GpuVmAccess *find_snapshot(const GpuVm *vm, AddressSpaceHandle handle);
  [[nodiscard]] static const GpuVmAccess *find_snapshot_vmid(const GpuVm *vm, uint32_t vmid);
  static void retain_snapshot(const GpuVm *vm, AddressSpaceHandle handle, const GpuVmAccess &access,
                              bool routed);
};

/// @brief Operation whose permissions must be checked by an address-space walk.
enum class VmAccessKind : uint8_t { Read, Write, Execute, Atomic };

/// @brief Typed result shared by translation and physical-memory operations.
enum class VmAccessOutcome : uint8_t {
  Complete,    ///< The requested operation completed.
  Unavailable, ///< The transport/backing is temporarily unavailable; retry may succeed.
  Faulted,     ///< The mapping is absent or rejects the requested access.
  Malformed,   ///< The address, page-table encoding, or request is invalid.
  Revoked,     ///< This snapshot was retired; the owner must recapture or terminate.
};

/// @brief Memory domain selected by a translation.
enum class VmMemoryDomain : uint8_t {
  System,
  Local,
  /// Transitional process-scoped address consumed only by a compatibility
  /// backing. It is not canonical physical identity and must not be shared
  /// across address spaces by a physical cache.
  Compatibility,
};

/// @brief Access permissions carried by a terminal translation.
class VmPermissions {
public:
  bool readable = false;
  bool writable = false;
  bool executable = false;

  [[nodiscard]] bool allows(VmAccessKind access) const {
    switch (access) {
    case VmAccessKind::Read:
      return readable;
    case VmAccessKind::Write:
      return writable;
    case VmAccessKind::Execute:
      return executable;
    case VmAccessKind::Atomic:
      return readable && writable;
    }
    return false;
  }
};

/// @brief One bounded virtual-to-physical translation.
class VmTranslation {
public:
  VmMemoryDomain domain = VmMemoryDomain::Local;
  uint64_t address = 0;
  uint64_t contiguous_bytes = 0;
  Mtype mtype = Mtype::RW;
  VmPermissions permissions;
};

/// @brief Outcome and address metadata produced by one translation attempt.
class VmTranslationResult {
public:
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  VmTranslation translation;

  explicit operator bool() const { return outcome == VmAccessOutcome::Complete; }
};

/// @brief Outcome of one complete physical request within a virtual transfer.
/// @details Failure advances no bytes and retains the physical transport's
/// no-effect guarantee. Only translation failures need the outer VM reporter;
/// a backing transport remains responsible for its own fault publication.
struct VmTransferStep {
  std::size_t completed_bytes = 0;
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  bool report_translation_fault = false;
};

/// @brief Result of one strong compare/exchange against physical backing.
///
/// @details The operation is indivisible with respect to every accessor of the
/// same backing.  It never fails spuriously: @ref exchanged is true exactly
/// when the observed value equalled the requested expected value and the
/// desired value was published.  Only naturally aligned 4- and 8-byte accesses
/// are valid.
struct AtomicCompareExchangeResult {
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  uint64_t observed = 0;
  bool exchanged = false;
};

/// @brief Result of one indivisible 2-, 4-, or 8-byte backing load.
struct AtomicLoadResult {
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  uint64_t value = 0;
};

/// @brief One ordered DWORD store from caller-owned, non-guest source storage.
struct VmRamDwordStore {
  static constexpr std::size_t kMaxBatch = 32;
  uint64_t address;
  const uint8_t *source;
};

/// @brief Stable physical-memory transport used after address translation.
class PhysicalMemoryAccess {
public:
  using AtomicMutation = std::function<void(std::span<std::byte>)>;
  using AtomicLaneMutation = std::function<void(uint32_t, std::span<std::byte>)>;

  virtual ~PhysicalMemoryAccess() = default;

  /// @brief Read one physical span as an indivisible transport request.
  /// @details A non-Complete result must leave @p bytes unmodified. Larger GPU
  /// virtual transfers may span several such requests and carry explicit
  /// progress in @ref GpuVmAccess so completed requests are never replayed.
  [[nodiscard]] virtual VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                                             std::span<std::byte> bytes) = 0;

  /// @brief Read one physical span for a specific virtual-memory operation.
  /// @details Ordinary physical backings do not distinguish instruction fetch
  /// from data reads. Compatibility adapters may preserve a legacy fetch-only
  /// backing policy here without moving translation or VMID ownership into the
  /// backing store.
  [[nodiscard]] virtual VmAccessOutcome read_for_access(VmMemoryDomain domain, uint64_t address,
                                                        std::span<std::byte> bytes,
                                                        VmAccessKind access) {
    (void)access;
    return read(domain, address, bytes);
  }
  /// @brief Write one physical span as an indivisible transport request.
  /// @details A non-Complete result must not modify the backing store.
  [[nodiscard]] virtual VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                                              std::span<const std::byte> bytes) = 0;

  /// @brief Attempt a batched copy with all-or-nothing refusal.
  /// @details A false result leaves the destination unchanged and reports no
  /// guest fault. Backings may decline spans they cannot service with this
  /// guarantee; callers retry smaller accesses. A successful copy is not atomic
  /// with respect to concurrent data accesses.
  [[nodiscard]] virtual bool try_read_contiguous(VmMemoryDomain domain, uint64_t address,
                                                 std::span<std::byte> bytes) {
    (void)domain;
    (void)address;
    (void)bytes;
    return false;
  }
  [[nodiscard]] virtual bool try_write_contiguous(VmMemoryDomain domain, uint64_t address,
                                                  std::span<const std::byte> bytes) {
    (void)domain;
    (void)address;
    (void)bytes;
    return false;
  }

  /// @brief Perform one acquire load from naturally aligned backing storage.
  /// @details Loads may be 2, 4, or 8 bytes. The 2-byte form is required by
  /// the AQL packet-header publication protocol.
  [[nodiscard]] virtual AtomicLoadResult atomic_load(VmMemoryDomain domain, uint64_t address,
                                                     uint32_t width) {
    (void)domain;
    (void)address;
    (void)width;
    return {};
  }

  /// @brief Perform one release store to naturally aligned backing storage.
  [[nodiscard]] virtual VmAccessOutcome atomic_store(VmMemoryDomain domain, uint64_t address,
                                                     uint32_t width, uint64_t value) {
    (void)domain;
    (void)address;
    (void)width;
    (void)value;
    return VmAccessOutcome::Faulted;
  }

  /// @brief Perform one non-spurious atomic compare/exchange on backing memory.
  ///
  /// @details Backings must not implement this as an unlocked read followed by
  /// a write.  The default rejects the operation so legacy backings remain
  /// source-compatible without silently claiming atomicity they cannot provide.
  [[nodiscard]] virtual AtomicCompareExchangeResult
  compare_exchange(VmMemoryDomain domain, uint64_t address, uint32_t width, uint64_t expected,
                   uint64_t desired) {
    (void)domain;
    (void)address;
    (void)width;
    (void)expected;
    (void)desired;
    return {};
  }

  /// @brief Apply one indivisible mutation to 4 or 8 bytes of backing storage.
  /// @details The callback is invoked exactly once only when the target can be
  /// serviced. Backings that cannot provide that guarantee reject the request.
  [[nodiscard]] virtual VmAccessOutcome atomic_modify(VmMemoryDomain domain, uint64_t address,
                                                      uint32_t width,
                                                      const AtomicMutation &mutation) {
    (void)domain;
    (void)address;
    (void)width;
    (void)mutation;
    return VmAccessOutcome::Faulted;
  }

  /// @brief Apply an indexed batch of independent atomic_modify() calls.
  /// Service the set bits of @p lane_mask from @p next_lane upward.
  /// @details Lane order and per-lane atomicity are those of atomic_modify(); an
  /// implementation may share lookups and admissions across lanes but must not
  /// retain any of them past the call. @p next_lane is the first unapplied lane on
  /// return: the failing lane, or the address count when every lane completed.
  /// @param addresses At most 64 translated addresses, all in @p domain.
  /// @param mutation Invoked as mutation(lane, bytes) once per serviced lane.
  [[nodiscard]] virtual VmAccessOutcome atomic_modify_lanes(VmMemoryDomain domain,
                                                            std::span<const uint64_t> addresses,
                                                            uint64_t lane_mask, uint32_t width,
                                                            const AtomicLaneMutation &mutation,
                                                            uint32_t &next_lane) {
    if (addresses.size() > 64)
      return VmAccessOutcome::Malformed;
    uint32_t lane = 0;
    const AtomicMutation lane_mutation = [&](std::span<std::byte> bytes) { mutation(lane, bytes); };
    for (lane = next_lane; lane < addresses.size(); ++lane) {
      if (!(lane_mask & (1ULL << lane)))
        continue;
      if (!mutation) {
        next_lane = lane;
        return VmAccessOutcome::Malformed;
      }
      const VmAccessOutcome outcome = atomic_modify(domain, addresses[lane], width, lane_mutation);
      if (outcome != VmAccessOutcome::Complete) {
        next_lane = lane;
        return outcome;
      }
    }
    next_lane = static_cast<uint32_t>(addresses.size());
    return VmAccessOutcome::Complete;
  }

  /// @brief Resolve a translated backing range to borrowed host storage.
  /// @details This optional debug/introspection capability returns nullptr when
  /// the backing is not directly host-addressable.
  [[nodiscard]] virtual std::byte *resolve_host_pointer(VmMemoryDomain domain, uint64_t address,
                                                        std::size_t size) const {
    (void)domain;
    (void)address;
    (void)size;
    return nullptr;
  }

  /// @brief Find the directly host-addressable range containing an address.
  /// @returns The host base and byte size, or {0, 0} when unavailable.
  [[nodiscard]] virtual std::pair<uint64_t, uint64_t> host_range(VmMemoryDomain domain,
                                                                 uint64_t address) const {
    (void)domain;
    (void)address;
    return {0, 0};
  }
};

/// @brief Copied page policy with an optional retained mutation token.
/// @details A translator may cache a range only when it invalidates the token
/// before every policy mutation. Capture the policy and token under the same
/// lease that excludes those mutations. No backing pointer is retained.
struct VmMtypeSnapshot {
  std::optional<Mtype> mtype;
  // A negative-only optimization hint. True still requires a live private-RAM
  // proof before copying; it never grants access or retains backing storage.
  bool may_batch_private_uc = false;
  bool may_batch_private_ram = false;
  uint64_t begin = 0;
  uint64_t size = 0;
  util::IntrusiveSharedPtr<const VmMutationEpoch> mutation_epoch{};
  uint64_t captured_epoch = 0;

  [[nodiscard]] bool unchanged(uint64_t address) const {
    return address >= begin && address - begin < size && mutation_epoch &&
           captured_epoch == mutation_epoch->value.load(std::memory_order_acquire);
  }
};

/// @brief Policy cache that does not retain the binding's backing.
/// @details Like an L1 cache, it requires externally serialized access.
class VmMtypeCache {
public:
  VmMtypeCache() = default;
  VmMtypeCache(const VmMtypeCache &other)
      : access_state_(other.access_state_),
        snapshots_(other.snapshots_ ? std::make_unique<Snapshots>(*other.snapshots_) : nullptr) {}
  VmMtypeCache &operator=(const VmMtypeCache &other) {
    if (this != &other) {
      VmMtypeCache copy(other);
      *this = std::move(copy);
    }
    return *this;
  }
  VmMtypeCache(VmMtypeCache &&) noexcept = default;
  VmMtypeCache &operator=(VmMtypeCache &&) noexcept = default;

  /// @brief Drop copied policies and their mutation tokens.
  /// @details Unused CUs retain neither a snapshot table nor work to clear
  /// during VM setup and teardown.
  void clear() noexcept {
    access_state_.reset();
    snapshots_.reset();
  }

private:
  friend class GpuVmAccess;
  static constexpr std::size_t kSnapshotCount = 128;
  using Snapshots = std::array<VmMtypeSnapshot, kSnapshotCount>;
  // Interleaved lanes revisit short page runs in widely separated rows.
  // Offset each 128-page block by eight slots to separate those runs without
  // mirroring adjacent pages at block boundaries; adding page >> 10 spreads
  // 4 MiB strides. Runs of up to 119 pages are conflict-free, but pages about
  // 120 apart can alias. Range and mutation-token checks validate every hit.
  [[nodiscard]] static std::size_t snapshot_index(uint64_t address) {
    const uint64_t page = address >> 12;
    return (page + ((page >> 7) << 3) + (page >> 10)) % kSnapshotCount;
  }
  std::weak_ptr<GpuVmAccessState> access_state_;
  std::unique_ptr<Snapshots> snapshots_;
};

/// @brief Stable metadata snapshot for one registered address space.
struct AddressSpaceInfo {
  uint32_t vmid = 0;
  uint64_t translation_epoch = 0;
  uint32_t queue_references = 0;
  /// True when the binding may use the compatibility virtual-address cache path.
  bool legacy_cache_compatible = false;
  /// True when a translator and physical backing are currently available.
  bool ready = false;
};

class GpuVmBindingState;

/// @brief Move-only ownership pin for a registered address-space binding.
/// @details Queue execution owners retain this lease for as long as they may
/// create new operation snapshots. Unregistration and reset fail while any
/// lease exists, closing the race between queue admission and VM teardown.
class GpuVmBindingLease {
public:
  GpuVmBindingLease() = default;
  GpuVmBindingLease(const GpuVmBindingLease &) = delete;
  GpuVmBindingLease &operator=(const GpuVmBindingLease &) = delete;
  GpuVmBindingLease(GpuVmBindingLease &&other) noexcept;
  GpuVmBindingLease &operator=(GpuVmBindingLease &&other) noexcept;
  ~GpuVmBindingLease();

  explicit operator bool() const { return state_ != nullptr; }
  [[nodiscard]] AddressSpaceHandle handle() const { return handle_; }
  [[nodiscard]] AddressSpaceInfo info() const { return info_; }

private:
  friend class GpuVm;

  GpuVmBindingLease(std::shared_ptr<GpuVmBindingState> state, AddressSpaceHandle handle,
                    AddressSpaceInfo info)
      : state_(std::move(state)), handle_(handle), info_(info) {}
  void release() noexcept;

  std::shared_ptr<GpuVmBindingState> state_;
  AddressSpaceHandle handle_;
  AddressSpaceInfo info_;
};

/// @brief Lifetime-safe namespace for virtually indexed cache entries.
///
/// @details A numeric VMID is routing metadata and may be reused.  The address-space
/// generation distinguishes successive owners of one slot, while the
/// translation epoch distinguishes root replacement and invalidation within
/// one address-space lifetime.
class VmCacheNamespace {
public:
  AddressSpaceHandle address_space;
  uint64_t translation_epoch = 0;

  explicit operator bool() const { return address_space && translation_epoch != 0; }
  friend bool operator==(const VmCacheNamespace &, const VmCacheNamespace &) = default;
};

/// @brief Immutable view of one GPU address-space binding.
///
/// @details The registry lock selects an immutable translator/backing generation.
/// Ordinary snapshots share that generation's retirement state; each pinned
/// snapshot has independent retirement state and survives root replacement.
/// Access methods hold a shared revocation lease, allowing an in-flight access
/// to finish without mixing roots. Invalidation and unregistration revoke both
/// kinds of snapshot. An admitted kernel dispatch retains a pinned snapshot;
/// functional instruction fetch without one reuses an ordinary snapshot within
/// a quantum, checking is_current() before reuse.
/// Operations on a revoked snapshot report @ref VmAccessOutcome::Revoked,
/// never the retryable Unavailable outcome.
class GpuVmAccess {
public:
  /// @brief Whether this snapshot's access state is still valid.
  /// @details Replacement, invalidation, and unregistration retire ordinary snapshots.
  /// Pinned snapshots remain valid across replacement until invalidation or unregistration.
  /// This check does not prevent concurrent retirement; access methods still validate the state.
  [[nodiscard]] bool is_current() const;

  [[nodiscard]] AddressSpaceInfo info() const { return info_; }
  /// @brief Whether invalidation, unregister, or reset permanently retired this snapshot.
  [[nodiscard]] bool revoked() const;
  [[nodiscard]] VmCacheNamespace cache_namespace() const {
    return {.address_space = address_space_, .translation_epoch = info_.translation_epoch};
  }

  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind access) const;
  [[nodiscard]] std::shared_ptr<const PrtRanges> nonresident_prt_ranges() const;
  /// @brief Query cache policy without exposing physical backing metadata.
  [[nodiscard]] std::optional<Mtype> query_mtype(uint64_t address) const;
  /// @brief Reuse a copied policy while its access state and mutation token are valid.
  [[nodiscard]] std::optional<Mtype> query_mtype(uint64_t address, VmMtypeCache &cache) const;
  /// Read a copied optimization hint without refreshing policy or invoking a
  /// translator. Uncertain/stale policy declines; true never grants access.
  [[nodiscard]] bool cached_private_uc_hint(uint64_t address, const VmMtypeCache &cache) const;
  /// @brief Query whether a complete virtual range currently permits an access.
  /// @details This side-effect-free query is for provisioning and routing decisions
  /// where an absent mapping is expected and must not be delivered as a GPU fault.
  /// It walks the same translation spans as probe() without accessing backing memory.
  [[nodiscard]] VmAccessOutcome query_access(uint64_t address, std::size_t size,
                                             VmAccessKind access) const;
  /// @brief Validate every translation span touched by a virtual range.
  /// @details Unlike translate(), this walks through page or segment boundaries
  /// and succeeds only when the complete range permits @p access. A terminal
  /// failure is reported to the address-space fault sink because this API models
  /// an attempted GPU access. It does not access the physical backing.
  [[nodiscard]] VmAccessOutcome probe(uint64_t address, std::size_t size,
                                      VmAccessKind access) const;
  [[nodiscard]] VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes,
                                     VmAccessKind access = VmAccessKind::Read) const;
  /// @brief Resume a translated read at @p completed_bytes.
  /// @details Progress advances only after one physical request completes. The
  /// caller retains it across Unavailable results to avoid replaying prior pages.
  [[nodiscard]] VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes,
                                     std::size_t &completed_bytes,
                                     VmAccessKind access = VmAccessKind::Read) const;
  [[nodiscard]] VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes) const;
  /// @brief Resume a translated write at @p completed_bytes.
  [[nodiscard]] VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes,
                                      std::size_t &completed_bytes) const;
  /// @brief Try one translated span with all-or-nothing refusal.
  /// @details Refusal leaves both bytes and backing untouched. It does not
  /// report a guest fault; the caller must issue its original smaller accesses.
  /// A successful copy is not atomic with respect to concurrent data accesses.
  [[nodiscard]] bool try_read_contiguous(uint64_t address, std::span<std::byte> bytes) const;
  /// Optional fault-free read from private UC RAM, with no allocation or callbacks.
  /// Refusal leaves the destination unchanged; issue original accesses afterward.
  [[nodiscard]] bool try_read_uncached_ram(uint64_t address, std::span<std::byte> bytes) const;
  /// No-effect refusal; sources must not overlap the admitted guest destination.
  [[nodiscard]] bool try_write_private_dwords(std::span<const VmRamDwordStore> stores,
                                              Mtype instruction_mtype, Mtype expected_mtype) const;
  /// Copied policy only; never refreshes metadata or authorizes a transfer.
  [[nodiscard]] std::optional<Mtype> cached_private_ram_mtype(uint64_t address,
                                                              const VmMtypeCache &cache) const;
  [[nodiscard]] bool try_write_contiguous(uint64_t address, std::span<const std::byte> bytes) const;
  [[nodiscard]] AtomicLoadResult atomic_load(uint64_t address, uint32_t width) const;
  [[nodiscard]] VmAccessOutcome atomic_store(uint64_t address, uint32_t width,
                                             uint64_t value) const;
  [[nodiscard]] AtomicCompareExchangeResult
  compare_exchange(uint64_t address, uint32_t width, uint64_t expected, uint64_t desired) const;
  [[nodiscard]] VmAccessOutcome
  atomic_modify(uint64_t address, uint32_t width,
                const PhysicalMemoryAccess::AtomicMutation &mutation) const;
  /// @brief Apply one atomic mutation per active lane under one revocation lease.
  /// @details Equivalent to calling atomic_modify() for each set bit of
  /// @p lane_mask from @p next_lane upward, in lane order, except that the
  /// shared revocation lease is taken once and one translated span is reused
  /// by every lane whose operand it covers. Every lane's backing mutation is
  /// still an independent atomic, and translation failures are reported per
  /// lane exactly as atomic_modify() reports them. On return @p next_lane is
  /// the first lane not yet applied: the address count when all completed, or
  /// the failing lane, so a retry resumes there. No translation outlives the call.
  /// @param lane_addresses At most 64 per-lane virtual operand addresses.
  /// @param mutation Invoked as mutation(lane, bytes) once per serviced lane.
  [[nodiscard]] VmAccessOutcome
  atomic_modify_lanes(std::span<const uint64_t> lane_addresses, uint64_t lane_mask, uint32_t width,
                      const PhysicalMemoryAccess::AtomicLaneMutation &mutation,
                      uint32_t &next_lane) const;
  [[nodiscard]] std::byte *resolve_host_pointer(uint64_t address, std::size_t size = 1) const;
  [[nodiscard]] std::pair<uint64_t, uint64_t> host_range(uint64_t address) const;

private:
  friend class GpuVm;

  GpuVmAccess(AddressSpaceHandle address_space, AddressSpaceInfo info,
              std::shared_ptr<GpuVmAccessState> access_state)
      : address_space_(address_space), info_(info), access_state_(std::move(access_state)) {}

  void report_terminal_fault(uint64_t address, VmAccessKind access, VmAccessOutcome outcome) const;
  [[nodiscard]] VmAccessOutcome probe_impl(uint64_t address, std::size_t size, VmAccessKind access,
                                           bool report_fault) const;

  AddressSpaceHandle address_space_;
  AddressSpaceInfo info_;
  std::shared_ptr<GpuVmAccessState> access_state_;
};

/// @brief Shared snapshot storage with metadata isolated from reference-count updates.
using GpuVmAccessSnapshot = util::RefCountedCacheAligned<GpuVmAccess>;

} // namespace rocjitsu::amdgpu
