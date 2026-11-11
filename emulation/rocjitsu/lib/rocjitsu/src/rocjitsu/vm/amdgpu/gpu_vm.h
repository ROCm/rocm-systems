// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_vm.h
/// @brief GPU address-space translation, registration and lifetime management.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm_access.h"
#include "util/distributed_shared_mutex.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {

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

/// @brief Address-space policy separated from the physical backing transport.
class AddressSpaceTranslator {
public:
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

private:
  std::shared_ptr<PhysicalMemoryAccess> retained_memory_;
  PhysicalMemoryAccess *memory_ = nullptr;
  uint64_t page_table_base_ = 0;
  uint64_t aperture_start_ = 0;
  uint64_t aperture_end_ = 0;
  Gfx12VmConfig config_;
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
  [[nodiscard]] AddressSpaceHandle
  register_unrouted_gfx12_address_space(uint32_t vmid, uint64_t page_table_base,
                                        std::shared_ptr<PhysicalMemoryAccess> memory);
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
  std::vector<Slot> slots_;
  std::vector<uint32_t> free_slots_;
  std::unordered_map<uint32_t, AddressSpaceHandle> vmid_handles_;
  AddressSpaceHandle gart_address_space_;
  uint64_t reset_epoch_ = 1;
  const uint64_t instance_id_;
  Gfx12VmConfig gfx12_config_;
};

} // namespace rocjitsu::amdgpu
