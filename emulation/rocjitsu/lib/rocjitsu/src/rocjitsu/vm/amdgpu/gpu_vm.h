// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_vm.h
/// @brief GPU address-space identity and lifetime management.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/canonical_line_freshness.h"
#include "util/distributed_shared_mutex.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {

class GpuVm;
class GpuVmAccess;
class GpuVmAccessState;
class DeviceCacheMaintenanceLease;
class VmTransferEffectGate;

/// @brief Acquire a move-only cache lease for one tokenized data step.
/// @details Acquisition runs without the VM state lock. A local backing step
/// keeps the lease through its effect; a deferred request releases it before
/// submission or polling.
using VmTransferMaintenance = std::function<DeviceCacheMaintenanceLease()>;

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

/// @brief Whether a data request can use the local fast path.
enum class VmDataRoute : uint8_t { Local, Deferred, Faulted };

/// @brief Typed result shared by translation and physical-memory operations.
enum class VmAccessOutcome : uint8_t {
  Complete,    ///< The requested operation completed.
  Unavailable, ///< The transport/backing is temporarily unavailable; retry may succeed.
  Faulted,     ///< The mapping is absent or rejects the requested access.
  Malformed,   ///< The address, page-table encoding, or request is invalid.
  Revoked,     ///< This snapshot was retired; the owner must recapture or terminate.
};

/// @brief Identity of the transport session that admitted a page-table walk.
/// @details A deferred owner must reacquire a lease on this exact session
/// before revalidating a walk. This proves lifetime, not page-table publication
/// or cache coherence.
class MemoryPtEpoch {
public:
  uint64_t generation = 0;
  std::shared_ptr<const void> session_identity;

  [[nodiscard]] explicit operator bool() const { return static_cast<bool>(session_identity); }
  friend bool operator==(const MemoryPtEpoch &, const MemoryPtEpoch &) = default;
};

/// @brief Lifetime-safe namespace for virtually indexed cache entries.
/// @details A numeric VMID is routing metadata and may be reused. The
/// address-space generation and translation epoch distinguish later owners
/// and page-table roots.
class VmCacheNamespace {
public:
  AddressSpaceHandle address_space;
  uint64_t translation_epoch = 0;

  explicit operator bool() const { return address_space && translation_epoch != 0; }
  friend bool operator==(const VmCacheNamespace &, const VmCacheNamespace &) = default;
};

/// @brief Move-only lease that releases one memory-ingress count on every exit path.
class MemoryIngressPermit {
public:
  MemoryIngressPermit() = default;
  explicit MemoryIngressPermit(std::function<void()> release,
                               std::optional<MemoryPtEpoch> pt_epoch = {})
      : release_(std::move(release)) {
    if (pt_epoch) {
      pt_epoch_ = std::move(*pt_epoch);
      has_pt_epoch_ = true;
    }
  }
  MemoryIngressPermit(const MemoryIngressPermit &) = delete;
  MemoryIngressPermit &operator=(const MemoryIngressPermit &) = delete;
  MemoryIngressPermit(MemoryIngressPermit &&other) noexcept
      : release_(std::exchange(other.release_, {})), pt_epoch_(std::move(other.pt_epoch_)),
        has_pt_epoch_(std::exchange(other.has_pt_epoch_, false)) {}
  MemoryIngressPermit &operator=(MemoryIngressPermit &&other) noexcept {
    if (this != &other) {
      reset();
      release_ = std::exchange(other.release_, {});
      pt_epoch_ = std::move(other.pt_epoch_);
      has_pt_epoch_ = std::exchange(other.has_pt_epoch_, false);
    }
    return *this;
  }
  ~MemoryIngressPermit() { reset(); }

  void reset() {
    if (auto release = std::exchange(release_, {}))
      release();
    pt_epoch_ = {};
    has_pt_epoch_ = false;
  }
  explicit operator bool() const { return static_cast<bool>(release_); }
  [[nodiscard]] std::optional<MemoryPtEpoch> pt_epoch() const {
    return has_pt_epoch_ ? std::optional<MemoryPtEpoch>{pt_epoch_} : std::nullopt;
  }

private:
  std::function<void()> release_;
  // Keep the shared_ptr constructed when the epoch is absent: GCC 13 at -O3
  // warns about an inactive optional payload during admission destruction.
  MemoryPtEpoch pt_epoch_{};
  bool has_pt_epoch_ = false;
};

/// @brief Result of admitting one local memory effect.
/// @details outcome decides whether the effect may proceed. A returned permit
/// keeps that admission alive through the effect; an empty permit does not
/// itself mean the request was refused.
struct MemoryIngressAdmission {
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  MemoryIngressPermit permit;
};

/// One virtual span admitted before a local GPU memory effect.
struct MemoryIngressSpan {
  uint64_t address = 0;
  size_t size = 0;
  VmAccessKind access = VmAccessKind::Read;
};

/// @brief Admit the remaining virtual span before one local backing step.
/// @details The span is a conservative upper bound on the local step. A
/// deferred route does not call this hook, and the permit is released before
/// any deferred request is submitted or polled.
using VmTransferIngress =
    std::function<MemoryIngressAdmission(uint64_t, std::size_t, VmAccessKind)>;

/// @brief Authority for one KFD SDMA COPY source read from private UC GTT.
/// @details A Complete result with no permit uses ordinary caller maintenance.
/// A permit certifies the entire requested local source span and is held only
/// through its synchronous read step. A closing source returns Unavailable or
/// Revoked so the caller cannot silently fall back to a stale host view.
struct VmPrivateUcGttCopySourceAdmission {
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  MemoryIngressPermit permit;
};
using VmPrivateUcGttCopySource =
    std::function<VmPrivateUcGttCopySourceAdmission(uint64_t, std::size_t)>;

/// @brief Live authorization for an architectural cache or backing request.
/// @details A nonzero generation distinguishes cache lines filled under an
/// earlier frontend mapping state, even when the VMID and virtual address match.
class VmAccessAdmission {
public:
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  uint64_t generation = 0;
  // A multi-grant request has no single generation suitable for a cache tag.
  bool cacheable = true;
  // Cache admission may succeed while a raw host-backing mutation is forbidden.
  bool direct_backing_write_allowed = true;
  explicit operator bool() const { return outcome == VmAccessOutcome::Complete; }
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
  friend bool operator==(const VmPermissions &, const VmPermissions &) = default;
};

/// @brief Exact memory-type field of a terminal GFX12 PTE.
/// @details The cache model currently treats NC and RW alike, but a peer
/// admission must still distinguish their raw encodings across a rewalk.
enum class Gfx12PteMtype : uint8_t { NC = 0, CC = 1, RW = 2, UC = 3 };

/// @brief One bounded virtual-to-physical translation.
class VmTranslation {
public:
  VmMemoryDomain domain = VmMemoryDomain::Local;
  uint64_t address = 0;
  uint64_t contiguous_bytes = 0;
  Mtype mtype = Mtype::RW;
  std::optional<Gfx12PteMtype> gfx12_pte_mtype = std::nullopt;
  VmPermissions permissions;
};

/// @brief One completed read stamp and whether the read used a VFIO peer route.
struct VmCompletedRead {
  CanonicalLineFreshness::ReadStamp stamp;
  bool peer_origin = false;
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

/// @brief Per-generation permit for a deferred backing effect.
/// @details An owner event enters immediately before its bounded memory effect.
/// Revocation closes entry and waits for effects that entered earlier to finish.
/// Submission and polling never hold this permit. Revocation drains permits
/// while the VM registry mutex is held, so an entered effect body must use
/// owned backing and must not synchronously call back into GpuVm.
class VmTransferEffectGate {
public:
  /// @brief One admitted effect that keeps revocation from finishing.
  /// @details Entry is nonblocking. A batch acquires every distinct gate
  /// before changing backing bytes and releases them after the whole batch.
  class Permit {
  public:
    Permit(const Permit &) = delete;
    Permit &operator=(const Permit &) = delete;
    Permit(Permit &&other) noexcept : gate_(std::exchange(other.gate_, nullptr)) {}
    Permit &operator=(Permit &&) = delete;
    ~Permit();

  private:
    friend class VmTransferEffectGate;
    explicit Permit(VmTransferEffectGate *gate) : gate_(gate) {}
    VmTransferEffectGate *gate_ = nullptr;
  };

  [[nodiscard]] std::optional<Permit> try_acquire() noexcept;

  template <typename Effect> [[nodiscard]] bool commit(Effect &&effect) {
    std::optional<Permit> permit = try_acquire();
    if (!permit)
      return false;
    std::invoke(std::forward<Effect>(effect));
    return true;
  }

  /// Close admission without waiting. A VFIO callback uses this while the
  /// protocol library holds its lock; the serving loop drains outside it.
  void close() noexcept;
  [[nodiscard]] bool drained() const noexcept;
  void revoke();

private:
  [[nodiscard]] bool enter() noexcept;
  void leave() noexcept;

  static constexpr uint64_t kRevoked = uint64_t{1} << 63;
  std::atomic<uint64_t> state_{0};
};

/// @brief One owned request handed to an execution owner's event loop.
/// @details Preparation may retain backing leases and copy write data, but
/// must not send or affect memory. submit() queues owned work for the owner
/// event without waiting or executing the backing itself, and is called once
/// after the VM lock is released. poll() is nonblocking and may copy a completed
/// read into its supplied staging span; it leaves that span unchanged on other
/// outcomes.
/// reconcile_revocation() is a nonblocking proof query after the VM generation
/// is retired. Complete includes a retained read result even when the reply is
/// still in flight; Revoked proves the owner never entered the effect; and
/// Unavailable means the entered effect has not published its outcome yet.
/// Terminal deferred faults are published by GpuVmAccess at the retained
/// request address when this operation is retired.
/// cancel() is nonblocking, including when an owner event is still queued.
class VmDeferredTransfer {
public:
  virtual ~VmDeferredTransfer() = default;
  virtual void submit() = 0;
  [[nodiscard]] virtual VmAccessOutcome poll(std::span<std::byte> read_result) = 0;
  [[nodiscard]] virtual VmAccessOutcome reconcile_revocation(std::span<std::byte> read_result) = 0;
  /// Owned canonical line versions for a completed deferred read, when the
  /// provider can prove them. Valid only after poll or reconciliation completes.
  [[nodiscard]] virtual std::optional<CanonicalLineFreshness::ReadStamp>
  completed_read_stamp() const {
    return std::nullopt;
  }
  virtual void cancel() noexcept = 0;
};

/// @brief Owner of cache maintenance for a deferred VM write.
/// @details A controlled-local endpoint prepares and proves selected alias
/// maintenance at effect time. Every other transfer retains caller maintenance
/// before submission.
enum class VmDeferredWriteMaintenance : uint8_t { Caller, ControlledLocalEndpoint };

/// @brief Move-only identity and storage for one outstanding VM request.
/// @details An owner keeps this token across Unavailable retries. The token
/// stages read results and advances transfer progress exactly once. Destroying
/// or resetting it cancels queued work without waiting for the owner event.
class VmTransferToken {
public:
  VmTransferToken() = default;
  VmTransferToken(std::unique_ptr<VmDeferredTransfer> operation, std::size_t bytes,
                  VmDeferredWriteMaintenance write_maintenance = VmDeferredWriteMaintenance::Caller,
                  bool peer_origin = false);
  ~VmTransferToken();
  VmTransferToken(const VmTransferToken &) = delete;
  VmTransferToken &operator=(const VmTransferToken &) = delete;
  VmTransferToken(VmTransferToken &&other) noexcept;
  VmTransferToken &operator=(VmTransferToken &&other) noexcept;

  [[nodiscard]] bool pending() const noexcept { return operation_ != nullptr; }
  /// Take the stamp and peer-route marker of the most recently completed read
  /// step after its bytes have been copied into the caller's result span.
  [[nodiscard]] std::optional<VmCompletedRead> take_completed_read() noexcept {
    return std::exchange(completed_read_, std::nullopt);
  }
  void cancel() noexcept;

private:
  friend class GpuVmAccess;
  std::unique_ptr<VmDeferredTransfer> operation_;
  bool peer_origin_ = false;
  std::optional<VmCompletedRead> completed_read_;
  std::vector<std::byte> read_result_;
  std::size_t bytes_ = 0;
  std::size_t total_bytes_ = 0;
  std::size_t offset_ = 0;
  uint64_t address_ = 0;
  VmAccessKind kind_ = VmAccessKind::Read;
  std::shared_ptr<GpuVmAccessState> state_;
  VmDeferredWriteMaintenance write_maintenance_ = VmDeferredWriteMaintenance::Caller;
  bool submitted_ = false;
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

/// @brief Complete VMID-0 aperture state published by a memory-hub invalidation.
struct GartConfig {
  /// Encoded GFX12 page-directory root: physical address plus its low PDE flags.
  uint64_t page_table_base = 0;
  uint64_t aperture_start = 0;
  uint64_t aperture_end = 0;
};

/// @brief Physical-address width selected by a GFX12 product profile.
enum class Gfx12PhysicalAddressWidth : uint8_t {
  Bits48 = 48,
  Bits52 = 52,
};

/// @brief Virtual-address width selected by a GFX12 product profile.
enum class Gfx12VirtualAddressWidth : uint8_t {
  Bits48 = 48,
  Bits57 = 57,
};

/// @brief Address-encoding policy shared by GFX12 page-table translators.
///
/// @details GFX 12.0 products expose a 48-bit virtual address space through a
/// four-level walk and 48 physical-address bits in PTEs. GFX 12.1 products use
/// a 57-bit virtual address space, a five-level walk, and 52 physical-address
/// bits. Keeping those product choices together in the VM layer prevents PCI
/// transports and backing stores from interpreting page-table encodings.
class Gfx12VmConfig {
public:
  Gfx12VirtualAddressWidth virtual_address_width = Gfx12VirtualAddressWidth::Bits57;
  Gfx12PhysicalAddressWidth physical_address_width = Gfx12PhysicalAddressWidth::Bits52;

  [[nodiscard]] constexpr uint8_t page_table_levels() const {
    return virtual_address_width == Gfx12VirtualAddressWidth::Bits48 ? 4 : 5;
  }

  [[nodiscard]] static constexpr Gfx12VmConfig gfx12_0() {
    return {.virtual_address_width = Gfx12VirtualAddressWidth::Bits48,
            .physical_address_width = Gfx12PhysicalAddressWidth::Bits48};
  }

  [[nodiscard]] static constexpr Gfx12VmConfig gfx12_1() {
    return {.virtual_address_width = Gfx12VirtualAddressWidth::Bits57,
            .physical_address_width = Gfx12PhysicalAddressWidth::Bits52};
  }
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

  /// A local cache may retain clean translated lines only when every ordinary
  /// local write is reported to the device coherence domain and host BAR writes
  /// have a modeled flush boundary. Other transports keep the direct VM path.
  [[nodiscard]] virtual bool supports_translated_local_cache() const { return false; }

  /// @brief Admit a live page-table walk before taking a VM state lease.
  /// @details Ordinary backings have no separate page-table lifetime lease. A
  /// PCI backing retains its transport session until the walk completes.
  [[nodiscard]] virtual MemoryIngressAdmission admit_pt_rewalk() const { return {}; }
  /// @brief Reenter the exact page-table epoch selected by deferred prepare.
  /// @details An unqualified backing cannot prove a supplied epoch and must
  /// reject it rather than silently admitting work in a different generation.
  [[nodiscard]] virtual MemoryIngressAdmission admit_pt_rewalk(const MemoryPtEpoch &) const {
    return {.outcome = VmAccessOutcome::Revoked, .permit = {}};
  }

  /// Prepare one translated System data step without copying backing bytes.
  /// Nullopt means ordinary synchronous storage. A rejected peer returns a
  /// terminal token, so it cannot fall through to the ordinary RAM path.
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_system_read(uint64_t, std::size_t, VmAccessKind,
                               const std::shared_ptr<VmTransferEffectGate> &) {
    return std::nullopt;
  }
  /// The tokenized GFX12 path supplies its exact executing snapshot, GPU VA,
  /// and the policy selected by the translation that produced this IOVA.
  /// Backings that do not use guest VM identity retain the original behavior.
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_system_read(uint64_t iova, std::size_t size, VmAccessKind access,
                               const std::shared_ptr<VmTransferEffectGate> &gate,
                               const GpuVmAccess &executing_access, uint64_t gpu_va,
                               const VmTranslation &selected_translation) {
    (void)executing_access;
    (void)gpu_va;
    (void)selected_translation;
    return prepare_deferred_system_read(iova, size, access, gate);
  }
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_system_read(uint64_t iova, std::size_t size, VmAccessKind access,
                               const std::shared_ptr<VmTransferEffectGate> &gate,
                               const GpuVmAccess &executing_access, uint64_t gpu_va,
                               const VmTranslation &selected_translation,
                               const std::optional<MemoryPtEpoch> &pt_epoch) {
    (void)pt_epoch;
    return prepare_deferred_system_read(iova, size, access, gate, executing_access, gpu_va,
                                        selected_translation);
  }
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_system_write(uint64_t, std::span<const std::byte>,
                                const std::shared_ptr<VmTransferEffectGate> &) {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_system_write(uint64_t iova, std::span<const std::byte> bytes,
                                const std::shared_ptr<VmTransferEffectGate> &gate,
                                const GpuVmAccess &executing_access, uint64_t gpu_va,
                                const VmTranslation &selected_translation) {
    (void)executing_access;
    (void)gpu_va;
    (void)selected_translation;
    return prepare_deferred_system_write(iova, bytes, gate);
  }
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_system_write(uint64_t iova, std::span<const std::byte> bytes,
                                const std::shared_ptr<VmTransferEffectGate> &gate,
                                const GpuVmAccess &executing_access, uint64_t gpu_va,
                                const VmTranslation &selected_translation,
                                const std::optional<MemoryPtEpoch> &pt_epoch) {
    (void)pt_epoch;
    return prepare_deferred_system_write(iova, bytes, gate, executing_access, gpu_va,
                                         selected_translation);
  }

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
  std::shared_ptr<const std::atomic<uint64_t>> mutation_epoch{};
  uint64_t captured_epoch = 0;

  [[nodiscard]] bool unchanged(uint64_t address) const {
    return address >= begin && address - begin < size && mutation_epoch &&
           captured_epoch == mutation_epoch->load(std::memory_order_acquire);
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

/// @brief Address-space policy separated from the physical backing transport.
class AddressSpaceTranslator {
public:
  /// @brief Classify a data span before a CU cache or local backing path.
  /// @details Implementations must make this a bounded, nonblocking policy
  /// query with no backing effect. Deferred requires tokenized access, while
  /// Faulted must not be served from a previously filled local cache line.
  [[nodiscard]] virtual VmDataRoute classify_data_route(uint64_t, std::size_t, VmAccessKind) const {
    return VmDataRoute::Local;
  }

  /// @brief Check frontend access policy before cache hits or backing effects.
  /// @details The default has no additional policy. Implementations must make
  /// this a bounded, nonblocking query because caches may use it in hot paths.
  [[nodiscard]] virtual VmAccessAdmission admit_access(uint64_t, std::size_t, VmAccessKind) const {
    return {};
  }
  /// @brief Read a live write grant for cache-tag freshness without authorizing
  /// a backing write. The returned direct-write flag remains advisory here.
  [[nodiscard]] virtual VmAccessAdmission query_cache_write_grant(uint64_t address,
                                                                  std::size_t size) const {
    return admit_access(address, size, VmAccessKind::Write);
  }
  /// @brief Optional ordered same-page DWORD stores. False has no guest effects.
  /// @details Implementations must prove every request under the same admission held
  /// through all copies, without allocation, callbacks or fault publication.
  [[nodiscard]] virtual bool try_write_private_dwords(PhysicalMemoryAccess &,
                                                      std::span<const VmRamDwordStore>, Mtype,
                                                      Mtype) const {
    return false;
  }

  /// @brief Optional read from private, stable UC RAM.
  /// @details Implementations must refuse unknown transports before touching guest
  /// data and must not allocate, report faults, or invoke foreign callbacks.
  /// False leaves bytes unchanged.
  [[nodiscard]] virtual bool try_read_uncached_ram(PhysicalMemoryAccess &, uint64_t,
                                                   std::span<std::byte>) const {
    return false;
  }

  virtual ~AddressSpaceTranslator() = default;

  [[nodiscard]] virtual VmTranslationResult translate(uint64_t address, std::size_t size,
                                                      VmAccessKind access) const = 0;

  /// @brief Optionally prepare one deferred data request under the VM lease.
  /// @details Return nullopt for a synchronous span. A returned token must own
  /// every request and backing lease it needs after the lease is released.
  /// The preparation must have no send or memory effect. The write form must
  /// copy its source bytes before returning. Neither form may defer Execute.
  /// An owner request must retain the supplied gate and enter it around every
  /// backing effect. The entered effect must not reenter the GpuVm registry.
  /// Untokenized read/write callers reject a returned token without submitting
  /// it, so a deferred route cannot silently fall back to synchronous access.
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &, uint64_t, std::size_t, VmAccessKind,
                        const std::shared_ptr<VmTransferEffectGate> &) const {
    return std::nullopt;
  }
  /// The tokenized path supplies the identity of its executing snapshot. A
  /// pinned snapshot can have a different cache identity from a fresh lookup.
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access, const std::shared_ptr<VmTransferEffectGate> &gate,
                        const GpuVmAccess &executing_access) const {
    (void)executing_access;
    return prepare_deferred_read(memory, address, size, access, gate);
  }
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access, const std::shared_ptr<VmTransferEffectGate> &gate,
                        const GpuVmAccess &executing_access,
                        const std::optional<MemoryPtEpoch> &pt_epoch) const {
    (void)pt_epoch;
    return prepare_deferred_read(memory, address, size, access, gate, executing_access);
  }
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &, uint64_t, std::span<const std::byte>,
                         const std::shared_ptr<VmTransferEffectGate> &) const {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &memory, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &gate,
                         const GpuVmAccess &executing_access) const {
    (void)executing_access;
    return prepare_deferred_write(memory, address, bytes, gate);
  }
  [[nodiscard]] virtual std::optional<VmTransferToken> prepare_deferred_write(
      PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
      const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access,
      const std::optional<MemoryPtEpoch> &pt_epoch) const {
    (void)pt_epoch;
    return prepare_deferred_write(memory, address, bytes, gate, executing_access);
  }

  /// @brief Execute the next ordinary translated physical request.
  /// @details The default translates one span and invokes the supplied transport.
  /// A combined implementation must preserve that span's policy, exact byte
  /// effects and fault publication, and qualify the supplied transport's identity.
  /// No translation pointer or admission may escape the call.
  [[nodiscard]] virtual VmTransferStep read_step(PhysicalMemoryAccess &, uint64_t address,
                                                 std::span<std::byte> remaining,
                                                 VmAccessKind access) const;
  [[nodiscard]] virtual VmTransferStep write_step(PhysicalMemoryAccess &, uint64_t address,
                                                  std::span<const std::byte> remaining) const;

  /// @brief Read only the page policy needed to resolve an effective MTYPE.
  /// @details The default uses an ordinary read translation. Compatibility
  /// translators override this so cache policy lookup does not query allocator
  /// metadata or otherwise expose physical backing.
  [[nodiscard]] virtual std::optional<Mtype> query_mtype(uint64_t address) const {
    const VmTranslationResult translated = translate(address, 1, VmAccessKind::Read);
    return translated ? std::optional<Mtype>(translated.translation.mtype) : std::nullopt;
  }

  /// @brief Optionally retain a mutation-checked copy of page policy.
  /// @details Cacheable snapshots capture the policy and token under the same
  /// lease and invalidate that token before every policy mutation. Retain only
  /// copied policy, never a backing pointer. The default remains uncached.
  [[nodiscard]] virtual VmMtypeSnapshot snapshot_mtype(uint64_t address) const {
    return {.mtype = query_mtype(address)};
  }

  /// @brief Translate one span for a non-mutating address-validity probe.
  /// @details Most page-table implementations use the ordinary translation
  /// result. Compatibility address spaces may override this when an operation
  /// can retry through backing-store fallbacks that must not make an absent GPU
  /// mapping appear resident to debugger and fault-detection clients.
  [[nodiscard]] virtual VmTranslationResult probe_translation(uint64_t address, std::size_t size,
                                                              VmAccessKind access) const {
    return translate(address, size, access);
  }
};

/// @brief Flat internal address space that maps GPU virtual addresses to the
/// same offsets in local physical memory.
///
/// @details This translator gives standalone and legacy model queues the same
/// GpuVm access path as frontend-managed queues without teaching GpuMemory
/// about virtual addressing. Translations are bounded to one 4 KiB page so
/// operation progress and atomic-span validation retain the same shape as a
/// page-table-backed address space.
class IdentityAddressSpaceTranslator final : public AddressSpaceTranslator {
public:
  static constexpr uint64_t kPageSize = 4096;

  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind access) const override;
};

/// @brief Execute a translated read without binding it to a registered address space.
[[nodiscard]] VmAccessOutcome read_translated(const AddressSpaceTranslator &translator,
                                              PhysicalMemoryAccess &memory, uint64_t address,
                                              std::span<std::byte> bytes);

/// @brief Execute a translated write without binding it to a registered address space.
[[nodiscard]] VmAccessOutcome write_translated(const AddressSpaceTranslator &translator,
                                               PhysicalMemoryAccess &memory, uint64_t address,
                                               std::span<const std::byte> bytes);

/// @brief Profile-specific GFX12 page-table translator used by PCI/VFIO queues.
class Gfx12PageTableTranslator final : public AddressSpaceTranslator {
public:
  Gfx12PageTableTranslator(std::shared_ptr<PhysicalMemoryAccess> memory, uint64_t page_table_base,
                           Gfx12VmConfig config = Gfx12VmConfig::gfx12_1());

  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind access) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access,
                        const std::shared_ptr<VmTransferEffectGate> &gate) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &memory, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &gate) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access, const std::shared_ptr<VmTransferEffectGate> &gate,
                        const GpuVmAccess &executing_access) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access, const std::shared_ptr<VmTransferEffectGate> &gate,
                        const GpuVmAccess &executing_access,
                        const std::optional<MemoryPtEpoch> &pt_epoch) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &memory, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &gate,
                         const GpuVmAccess &executing_access) const override;
  [[nodiscard]] std::optional<VmTransferToken> prepare_deferred_write(
      PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
      const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access,
      const std::optional<MemoryPtEpoch> &pt_epoch) const override;

private:
  std::shared_ptr<PhysicalMemoryAccess> memory_;
  uint64_t page_table_base_ = 0;
  Gfx12VmConfig config_;
};

/// @brief GFX12 VMID-0 aperture translator used for driver-owned GART buffers.
///
/// @details Addresses within the configured aperture use its linear PTE array.
/// Addresses outside it are local-memory physical offsets and bypass the table.
class Gfx12GartTranslator final : public AddressSpaceTranslator {
public:
  /// @brief Borrow a backing for a translator whose lifetime is externally bounded.
  Gfx12GartTranslator(PhysicalMemoryAccess &memory, uint64_t page_table_base,
                      uint64_t aperture_start, uint64_t aperture_end,
                      Gfx12VmConfig config = Gfx12VmConfig::gfx12_1());
  Gfx12GartTranslator(std::shared_ptr<PhysicalMemoryAccess> memory, uint64_t page_table_base,
                      uint64_t aperture_start, uint64_t aperture_end,
                      Gfx12VmConfig config = Gfx12VmConfig::gfx12_1());

  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind access) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access,
                        const std::shared_ptr<VmTransferEffectGate> &gate) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &memory, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &gate) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access, const std::shared_ptr<VmTransferEffectGate> &gate,
                        const GpuVmAccess &executing_access) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &memory, uint64_t address, std::size_t size,
                        VmAccessKind access, const std::shared_ptr<VmTransferEffectGate> &gate,
                        const GpuVmAccess &executing_access,
                        const std::optional<MemoryPtEpoch> &pt_epoch) const override;
  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &memory, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &gate,
                         const GpuVmAccess &executing_access) const override;
  [[nodiscard]] std::optional<VmTransferToken> prepare_deferred_write(
      PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
      const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access,
      const std::optional<MemoryPtEpoch> &pt_epoch) const override;

private:
  std::shared_ptr<PhysicalMemoryAccess> retained_memory_;
  PhysicalMemoryAccess *memory_ = nullptr;
  uint64_t page_table_base_ = 0;
  uint64_t aperture_start_ = 0;
  uint64_t aperture_end_ = 0;
  Gfx12VmConfig config_;
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

class GpuVm;
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
// Keep immutable metadata read by every CU off the cache line containing a
// coallocated shared_ptr reference count, which wave admission and retirement update.
class alignas(64) GpuVmAccess {
public:
  /// @brief Whether this snapshot's access state is still valid.
  /// @details Replacement, invalidation, and unregistration retire ordinary snapshots.
  /// Pinned snapshots remain valid across replacement until invalidation or unregistration.
  /// This check does not prevent concurrent retirement; access methods still validate the state.
  [[nodiscard]] bool is_current() const;

  [[nodiscard]] AddressSpaceInfo info() const { return info_; }
  /// @brief Whether invalidation, unregister, or reset permanently retired this snapshot.
  [[nodiscard]] bool revoked() const;
  /// @brief Process-unique identity of this immutable backing generation.
  /// @details Ordinary snapshots of one live generation share the identity;
  /// replacement, invalidation, or VMID reuse receives a different one.
  [[nodiscard]] uint64_t cache_identity() const;
  [[nodiscard]] bool supports_translated_local_cache() const;
  /// Saturating count of admitted cache writes and atomics under this root.
  /// Invalidation and pinned snapshots share the count; explicit root
  /// replacement starts a new count. Saturation is unsafe for recovery.
  [[nodiscard]] uint64_t cache_write_sequence() const;
  /// Whether two snapshots descend from one explicit VM root binding.
  /// Invalidation preserves the lineage; root replacement starts a new one.
  [[nodiscard]] bool same_root_lineage(const GpuVmAccess &other) const;
  [[nodiscard]] VmCacheNamespace cache_namespace() const {
    return {.address_space = address_space_, .translation_epoch = info_.translation_epoch};
  }
  /// @brief Retain the exact generation gate for an already prepared effect.
  /// @details A caller must establish its own backing and frontend authority
  /// before taking cache locks. Gate entry then excludes VM retirement during
  /// the bounded effect without calling back into the VM registry.
  [[nodiscard]] std::shared_ptr<VmTransferEffectGate> retain_effect_gate() const;

  /// @brief Retain permission for one live page-table rewalk.
  /// @details Admission is acquired before the VM state lease; a closed PCI
  /// gate returns retryable Unavailable without reading the first PTE.
  [[nodiscard]] MemoryIngressAdmission admit_pt_rewalk() const;
  /// @brief Reenter the exact gate captured by a deferred PCI request.
  [[nodiscard]] MemoryIngressAdmission admit_pt_rewalk(const MemoryPtEpoch &expected) const;

  /// @brief Inspect one translated span without publishing an architectural fault.
  /// @details Intended for optional routing probes; a successful result does
  /// not authorize a cache hit, fill, or backing effect.
  [[nodiscard]] VmTranslationResult query_translation(uint64_t address, std::size_t size,
                                                      VmAccessKind access) const;
  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind access) const;
  /// @brief Recheck a queued System step without publishing a second GPU fault.
  /// @details This validates the retained snapshot, selected memory type and
  /// all permission bits, IOVA, and complete span immediately before a deferred
  /// provider grants its effect. The VFIO owner must hold the PT permit it
  /// acquired before effect entry across this validation and the effect;
  /// acquiring again here could reject an already admitted entry after BEGIN.
  [[nodiscard]] VmAccessOutcome validate_deferred_system_span(uint64_t gpu_va, uint64_t iova,
                                                              std::size_t size, VmAccessKind access,
                                                              const VmTranslation &selected) const;
  /// @brief Classify a data route, optionally suppressing architectural fault
  /// reporting for maintenance probes that must fail closed and retry.
  [[nodiscard]] VmDataRoute classify_data_route(uint64_t address, std::size_t size,
                                                VmAccessKind access,
                                                bool report_fault = true) const;
  /// @brief Check live frontend policy without translating or touching backing.
  /// @details Terminal denial is reported to this binding's GPU fault sink.
  [[nodiscard]] VmAccessAdmission admit_access(uint64_t address, std::size_t size,
                                               VmAccessKind access, bool report_fault = true) const;
  /// @brief Compare dirty cache tags with the current write grant only.
  /// @details This query does not authorize a backing effect, report a GPU
  /// fault, or advance the cache-write sequence.
  [[nodiscard]] VmAccessAdmission query_cache_write_grant(uint64_t address, std::size_t size) const;
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
  /// @brief Resume a data read with a retained deferred-request identity.
  /// @details Setting @p allow_deferred false still permits synchronous spans,
  /// but rejects a deferred route before submission. If supplied, maintenance
  /// is acquired outside the VM lock for each local span and released before
  /// any deferred request is submitted or polled.
  [[nodiscard]] VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes,
                                     std::size_t &completed_bytes, VmTransferToken &token,
                                     VmAccessKind access = VmAccessKind::Read,
                                     bool allow_deferred = true) const;
  [[nodiscard]] VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes,
                                     std::size_t &completed_bytes, VmTransferToken &token,
                                     VmAccessKind access, bool allow_deferred,
                                     const VmTransferMaintenance &maintenance,
                                     const VmTransferIngress &ingress = {},
                                     const VmPrivateUcGttCopySource &copy_source = {}) const;
  [[nodiscard]] VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes) const;
  /// @brief Resume a translated write at @p completed_bytes.
  [[nodiscard]] VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes,
                                      std::size_t &completed_bytes) const;
  [[nodiscard]] VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes,
                                      std::size_t &completed_bytes, VmTransferToken &token,
                                      bool allow_deferred = true) const;
  [[nodiscard]] VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes,
                                      std::size_t &completed_bytes, VmTransferToken &token,
                                      bool allow_deferred, const VmTransferMaintenance &maintenance,
                                      const VmTransferIngress &ingress = {}) const;
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
  template <typename Span>
  [[nodiscard]] VmAccessOutcome
  transfer_tokenized(uint64_t address, Span bytes, std::size_t &completed_bytes,
                     VmTransferToken &token, VmAccessKind access, bool allow_deferred,
                     const VmTransferMaintenance &maintenance, const VmTransferIngress &ingress,
                     const VmPrivateUcGttCopySource &copy_source) const;

  AddressSpaceHandle address_space_;
  AddressSpaceInfo info_;
  std::shared_ptr<GpuVmAccessState> access_state_;
};

/// @brief Owns GPU address-space identities independently of their front end.
///
/// @details Legacy KFD and PCI/VFIO front ends register address spaces here and
/// receive generation-checked handles. Numeric VMIDs and PASIDs remain routing
/// metadata; they are not lifetime-safe identities. Queue references describe
/// frontend-owned queues, while binding leases prevent revocation as long as an
/// execution owner may create another operation snapshot.
class GpuVm {
public:
  explicit GpuVm(Gfx12VmConfig gfx12_config = Gfx12VmConfig::gfx12_1());

  /// @brief Publish a controlled canonical backing range before its first GPU grant.
  /// @details Cache sinks consult this copy-on-write policy without a registry
  /// or KFD callback while holding cache locks. The binding identity prevents
  /// a retired VMID's policy from affecting a later owner of the same number.
  [[nodiscard]] bool protect_controlled_cache_backing(AddressSpaceHandle handle, uint64_t address,
                                                      uint64_t size);
  /// @brief Retire the policy only after the backing's grants and PTEs are retired.
  /// @details This operation does not allocate, so completed KFD retirement can use it.
  void retire_controlled_cache_backing(AddressSpaceHandle handle, uint64_t address,
                                       uint64_t size) noexcept;
  /// @brief True when a direct cache sink must refuse a write to this virtual range.
  /// @details The query uses only an immutable policy snapshot and binding lifetime bit.
  [[nodiscard]] bool controlled_cache_backing_overlaps(uint32_t vmid, uint64_t address,
                                                       uint64_t size) const noexcept;

  /// Short registry authority proving that a fill still targets the current
  /// address-space generation. Release it on the acquiring thread before any
  /// VM callback or before destroying the GpuVm.
  class CurrentNamespaceLease {
  public:
    CurrentNamespaceLease() = default;
    CurrentNamespaceLease(CurrentNamespaceLease &&) noexcept = default;
    CurrentNamespaceLease &operator=(CurrentNamespaceLease &&) noexcept = default;
    CurrentNamespaceLease(const CurrentNamespaceLease &) = delete;
    CurrentNamespaceLease &operator=(const CurrentNamespaceLease &) = delete;

    [[nodiscard]] explicit operator bool() const noexcept { return lock_.owns_lock(); }

  private:
    friend class GpuVm;
    explicit CurrentNamespaceLease(std::shared_lock<util::DistributedSharedMutex> lock) noexcept
        : lock_(std::move(lock)) {}

    std::shared_lock<util::DistributedSharedMutex> lock_;
  };

  /// Nonblocking check of the live binding's handle and translation epoch.
  /// Hold this lease through only a cache fill; root replacement and VMID
  /// reuse then cannot publish a different namespace until it is released.
  [[nodiscard]] CurrentNamespaceLease
  try_acquire_current_namespace(AddressSpaceHandle handle,
                                VmCacheNamespace expected) const noexcept;

  /// @brief Register one complete frontend-provided address-space binding.
  /// @details The VM retains the shared translator and backing in every access
  /// snapshot. @p legacy_cache_compatible describes cache-addressing behavior,
  /// not the frontend that supplied the binding.
  /// The fault reporter is shared across snapshots and generations. Invocations
  /// are serialized per registration. Same-thread callback reentry is allowed
  /// only through independent access states, such as a fresh snapshot_pinned().
  /// Nested access through the reporting state is forbidden: its revocation
  /// lease is not recursive, and ordinary snapshot()/snapshot_vmid() calls
  /// share that state. A reporter must not wait for another thread to fault
  /// through the same registration. Accesses can hold a revocation lease while
  /// reporting, so a reporter must defer invalidating, replacing, or unregistering
  /// that binding.
  [[nodiscard]] AddressSpaceHandle
  register_address_space(uint32_t vmid, std::shared_ptr<AddressSpaceTranslator> translator,
                         std::shared_ptr<PhysicalMemoryAccess> physical_memory,
                         std::function<void(uint64_t, VmAccessKind)> fault_reporter = {},
                         bool legacy_cache_compatible = false);
  /// @brief Register a handle-only binding that does not claim numeric VMID routing.
  ///
  /// @details Internal model queues can retain and snapshot this explicit
  /// handle while a frontend-owned address space with the same numeric VMID
  /// remains discoverable through find_vmid() and snapshot_vmid().
  /// The fault reporter follows the same contract as register_address_space().
  [[nodiscard]] AddressSpaceHandle
  register_unrouted_address_space(uint32_t vmid, std::shared_ptr<AddressSpaceTranslator> translator,
                                  std::shared_ptr<PhysicalMemoryAccess> physical_memory,
                                  std::function<void(uint64_t, VmAccessKind)> fault_reporter = {},
                                  bool legacy_cache_compatible = false);
  [[nodiscard]] AddressSpaceHandle
  register_translated(uint32_t vmid, std::shared_ptr<AddressSpaceTranslator> translator,
                      std::shared_ptr<PhysicalMemoryAccess> memory);
  /// @brief Register a GFX12 process root using this VM's product translation policy.
  [[nodiscard]] AddressSpaceHandle
  register_gfx12_address_space(uint32_t vmid, uint64_t page_table_base,
                               std::shared_ptr<PhysicalMemoryAccess> memory);
  /// @brief Register a handle-only GFX12 root without claiming numeric VMID routing.
  [[nodiscard]] AddressSpaceHandle register_unrouted_gfx12_address_space(
      uint32_t vmid, uint64_t page_table_base, std::shared_ptr<PhysicalMemoryAccess> memory,
      std::function<void(uint64_t, VmAccessKind)> fault_reporter = {});
  [[nodiscard]] bool replace_translated(AddressSpaceHandle handle,
                                        std::shared_ptr<AddressSpaceTranslator> translator,
                                        std::shared_ptr<PhysicalMemoryAccess> memory);
  /// @brief Replace a GFX12 process root without changing its lifetime-safe identity.
  [[nodiscard]] bool replace_gfx12_address_space_root(AddressSpaceHandle handle,
                                                      uint64_t page_table_base,
                                                      std::shared_ptr<PhysicalMemoryAccess> memory);
  /// @brief Create the device-global VMID-0 identity before its first publication.
  ///
  /// @details Idempotent for the one device attached to this VM. Until a hub
  /// invalidation publishes a configuration, accesses through the returned
  /// handle report @ref VmAccessOutcome::Unavailable.
  [[nodiscard]] AddressSpaceHandle initialize_gart_address_space();
  /// @brief Atomically publish a new VMID-0 translator and advance its epoch.
  [[nodiscard]] bool publish_gart(const GartConfig &config,
                                  std::shared_ptr<PhysicalMemoryAccess> memory);
  /// @brief Remove the VMID-0 translator and backing without revoking its identity.
  /// @details Frontend teardown calls this after releasing every queue and
  /// operation that can retain the old binding. The stable handle can then be
  /// republished by a replacement PCI frontend without disturbing unrelated
  /// legacy or translated process address spaces.
  [[nodiscard]] bool clear_gart_binding();
  /// @brief Current device-global VMID-0 identity, or an invalid handle.
  [[nodiscard]] AddressSpaceHandle gart_address_space() const;
  /// @brief Advance the translation version after a page-table invalidation.
  [[nodiscard]] bool invalidate(AddressSpaceHandle handle);
  /// @brief Retain an address space for a queue and return its routing metadata atomically.
  [[nodiscard]] std::optional<AddressSpaceInfo>
  retain_queue_address_space(AddressSpaceHandle handle);
  [[nodiscard]] bool retain_queue(AddressSpaceHandle handle);
  [[nodiscard]] bool release_queue(AddressSpaceHandle handle);
  /// @brief Retain a binding for an execution owner that will take future snapshots.
  [[nodiscard]] std::optional<GpuVmBindingLease> retain_binding(AddressSpaceHandle handle);
  [[nodiscard]] bool unregister_address_space(AddressSpaceHandle handle);

  /// @brief Capture current metadata and one coherent translator/backing/epoch tuple.
  [[nodiscard]] std::optional<GpuVmAccess> snapshot(AddressSpaceHandle handle) const;
  /// @brief Capture a transaction snapshot that survives root replacement.
  /// @details Packet execution and guest-visible publication may retry after
  /// their semantic effects have committed. A pinned snapshot keeps those
  /// retries on the root that admitted the transaction. Invalidation,
  /// unregister, and reset still revoke it synchronously.
  [[nodiscard]] std::optional<GpuVmAccess> snapshot_pinned(AddressSpaceHandle handle);
  /// @brief Capture the current binding selected by a numeric VMID.
  /// @details The lookup and snapshot occur under one lock, so unregister and
  /// VMID reuse cannot substitute a different generation between them.
  [[nodiscard]] std::optional<GpuVmAccess> snapshot_vmid(uint32_t vmid) const;
  /// @brief Borrow a cached snapshot for the active functional quantum.
  /// @details These forms avoid copying snapshot ownership on every emulated
  /// instruction. The returned pointer remains valid until the outermost
  /// GpuVmAccessBatchGuard on this thread exits. Calling without an active
  /// batch is a programming error. Metadata reflects the cached capture; use
  /// snapshot() or snapshot_vmid() when current metadata is required.
  [[nodiscard]] const GpuVmAccess *borrow_snapshot(AddressSpaceHandle handle) const;
  [[nodiscard]] const GpuVmAccess *borrow_snapshot_vmid(uint32_t vmid) const;

  /// @brief Run one operation against the current VMID snapshot.
  /// @details Functional execution borrows the snapshot retained by the
  /// active quantum. Other callers receive an owning operation snapshot, so
  /// access-state invalidation is checked in both paths. The callback must not
  /// retain snapshot pointers or references beyond the owning operation's
  /// lifetime, or beyond the outer quantum for a borrowed snapshot.
  template <typename Operation>
  decltype(auto) with_vmid_snapshot(uint32_t vmid, Operation &&operation) const {
    if (GpuVmAccessBatchGuard::active())
      return std::invoke(std::forward<Operation>(operation), borrow_snapshot_vmid(vmid));
    const std::optional<GpuVmAccess> access = snapshot_vmid(vmid);
    return std::invoke(std::forward<Operation>(operation), access ? &*access : nullptr);
  }

  [[nodiscard]] VmTranslationResult translate(AddressSpaceHandle handle, uint64_t address,
                                              std::size_t size, VmAccessKind access) const;
  [[nodiscard]] VmAccessOutcome probe(AddressSpaceHandle handle, uint64_t address, std::size_t size,
                                      VmAccessKind access) const;
  [[nodiscard]] VmAccessOutcome read(AddressSpaceHandle handle, uint64_t address,
                                     std::span<std::byte> bytes) const;
  [[nodiscard]] VmAccessOutcome write(AddressSpaceHandle handle, uint64_t address,
                                      std::span<const std::byte> bytes);
  [[nodiscard]] AtomicLoadResult atomic_load(AddressSpaceHandle handle, uint64_t address,
                                             uint32_t width) const;
  [[nodiscard]] VmAccessOutcome atomic_store(AddressSpaceHandle handle, uint64_t address,
                                             uint32_t width, uint64_t value);
  [[nodiscard]] AtomicCompareExchangeResult compare_exchange(AddressSpaceHandle handle,
                                                             uint64_t address, uint32_t width,
                                                             uint64_t expected, uint64_t desired);

  [[nodiscard]] std::optional<AddressSpaceInfo> lookup(AddressSpaceHandle handle) const;
  [[nodiscard]] std::optional<AddressSpaceHandle> find_vmid(uint32_t vmid) const;
  [[nodiscard]] std::size_t active_address_spaces() const;
  [[nodiscard]] uint64_t reset_epoch() const;

  /// @brief Revoke every address space after all queues have been removed.
  /// @retval false At least one queue still retained an address space.
  [[nodiscard]] bool reset();

private:
  friend class GpuVmAccessBatchGuard;

  struct ControlledCacheBacking {
    AddressSpaceHandle handle;
    uint32_t vmid;
    uint64_t first;
    uint64_t last;
    std::weak_ptr<GpuVmBindingState> binding;
    std::shared_ptr<std::atomic<bool>> active;
  };

  class Binding {
  public:
    uint32_t vmid = 0;
    uint64_t translation_epoch = 1;
    uint32_t queue_references = 0;
    std::shared_ptr<GpuVmBindingState> binding_state;
    std::shared_ptr<GpuVmAccessState> access_state;
    std::vector<std::weak_ptr<GpuVmAccessState>> pinned_access_states;
    bool legacy_cache_compatible = false;
    bool device_gart = false;
  };

  class Slot {
  public:
    uint64_t generation = 1;
    std::optional<Binding> binding;
  };

  [[nodiscard]] AddressSpaceHandle allocate_locked(Binding binding);
  [[nodiscard]] Binding *find_locked(AddressSpaceHandle handle);
  [[nodiscard]] const Binding *find_locked(AddressSpaceHandle handle) const;
  void advance_access_state_locked(Binding &binding, std::shared_ptr<GpuVmAccessState> replacement);
  void revoke_access_state_locked(Binding &binding);

  mutable util::DistributedSharedMutex mutex_;
  std::mutex controlled_backing_mutex_;
  std::atomic<std::shared_ptr<const std::vector<ControlledCacheBacking>>> controlled_backings_;
  std::vector<Slot> slots_;
  std::vector<uint32_t> free_slots_;
  std::unordered_map<uint32_t, AddressSpaceHandle> vmid_handles_;
  AddressSpaceHandle gart_address_space_;
  uint64_t reset_epoch_ = 1;
  const uint64_t instance_id_;
  Gfx12VmConfig gfx12_config_;
};

} // namespace rocjitsu::amdgpu
