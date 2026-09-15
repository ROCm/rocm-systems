// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_vm.h
/// @brief GPU address-space identity and lifetime management.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {

class GpuMemory;

/// @brief Operation whose permissions must be checked by an address-space walk.
enum class VmAccessKind : uint8_t { Read, Write, Execute, Atomic };

/// @brief Typed result shared by translation and physical-memory operations.
enum class VmAccessOutcome : uint8_t {
  Complete,    ///< The requested operation completed.
  Unavailable, ///< The transport/backing is temporarily unavailable; retry may succeed.
  Faulted,     ///< The mapping is absent or rejects the requested access.
  Malformed,   ///< The address, page-table encoding, or request is invalid.
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
struct VmPermissions {
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
struct VmTranslation {
  VmMemoryDomain domain = VmMemoryDomain::Local;
  uint64_t address = 0;
  uint64_t contiguous_bytes = 0;
  Mtype mtype = Mtype::RW;
  VmPermissions permissions;
};

struct VmTranslationResult {
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  VmTranslation translation;

  explicit operator bool() const { return outcome == VmAccessOutcome::Complete; }
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

/// @brief Result of one indivisible 4- or 8-byte backing load.
struct AtomicLoadResult {
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  uint64_t value = 0;
};

/// @brief Complete VMID-0 aperture state published by a memory-hub invalidation.
struct GartConfig {
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
struct Gfx12VmConfig {
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

/// @brief Stable physical-memory transport used after address translation.
class PhysicalMemoryAccess {
public:
  virtual ~PhysicalMemoryAccess() = default;

  /// @brief Read one physical span as an indivisible transport request.
  /// @details A non-Complete result must leave @p bytes unmodified. Larger GPU
  /// virtual transfers may span several such requests and carry explicit
  /// progress in @ref GpuVmAccess so completed requests are never replayed.
  [[nodiscard]] virtual VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                                             std::span<std::byte> bytes) = 0;
  /// @brief Write one physical span as an indivisible transport request.
  /// @details A non-Complete result must not modify the backing store.
  [[nodiscard]] virtual VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                                              std::span<const std::byte> bytes) = 0;

  /// @brief Perform one acquire load from naturally aligned backing storage.
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
};

/// @brief Address-space policy separated from the physical backing transport.
class AddressSpaceTranslator {
public:
  virtual ~AddressSpaceTranslator() = default;

  [[nodiscard]] virtual VmTranslationResult translate(uint64_t address, std::size_t size,
                                                      VmAccessKind access) const = 0;
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

/// @brief Stable metadata snapshot for one registered address space.
struct AddressSpaceInfo {
  uint32_t vmid = 0;
  uint64_t translation_epoch = 0;
  uint32_t queue_references = 0;
  /// True for translated address spaces, including one awaiting first publication.
  bool external = false;
  /// True when a translator and physical backing are currently available.
  bool ready = false;
};

/// @brief Lifetime-safe namespace for virtually indexed cache entries.
///
/// A numeric VMID is routing metadata and may be reused.  The address-space
/// generation distinguishes successive owners of one slot, while the
/// translation epoch distinguishes root replacement and invalidation within
/// one address-space lifetime.
struct VmCacheNamespace {
  AddressSpaceHandle address_space;
  uint64_t translation_epoch = 0;

  explicit operator bool() const { return address_space && translation_epoch != 0; }
  friend bool operator==(const VmCacheNamespace &, const VmCacheNamespace &) = default;
};

/// @brief Immutable, operation-scoped view of one GPU address-space binding.
///
/// The snapshot retains the translator and physical backing selected under the
/// GpuVm lock.  A multi-page access therefore cannot observe half of an old
/// root and half of its replacement.  Replacement or invalidation advances the
/// cache namespace used by subsequent snapshots; an already-started operation
/// is allowed to finish against the binding it captured.
class GpuVmAccess {
public:
  [[nodiscard]] AddressSpaceInfo info() const { return info_; }
  [[nodiscard]] VmCacheNamespace cache_namespace() const {
    return {.address_space = address_space_, .translation_epoch = info_.translation_epoch};
  }

  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind access) const;
  /// @brief Validate every translation span touched by a virtual range.
  /// @details Unlike translate(), this walks through page or segment boundaries
  /// and succeeds only when the complete range permits @p access. It does not
  /// access the physical backing.
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
  [[nodiscard]] AtomicLoadResult atomic_load(uint64_t address, uint32_t width) const;
  [[nodiscard]] VmAccessOutcome atomic_store(uint64_t address, uint32_t width,
                                             uint64_t value) const;
  [[nodiscard]] AtomicCompareExchangeResult
  compare_exchange(uint64_t address, uint32_t width, uint64_t expected, uint64_t desired) const;

private:
  friend class GpuVm;

  GpuVmAccess(AddressSpaceHandle address_space, AddressSpaceInfo info,
              std::shared_ptr<AddressSpaceTranslator> translator,
              std::shared_ptr<PhysicalMemoryAccess> physical_memory,
              std::function<void(uint64_t, VmAccessKind)> fault_reporter)
      : address_space_(address_space), info_(info), translator_(std::move(translator)),
        physical_memory_(std::move(physical_memory)), fault_reporter_(std::move(fault_reporter)) {}

  void report_terminal_fault(uint64_t address, VmAccessKind access, VmAccessOutcome outcome) const;

  AddressSpaceHandle address_space_;
  AddressSpaceInfo info_;
  std::shared_ptr<AddressSpaceTranslator> translator_;
  std::shared_ptr<PhysicalMemoryAccess> physical_memory_;
  std::function<void(uint64_t, VmAccessKind)> fault_reporter_;
};

/// @brief Owns GPU address-space identities independently of their front end.
///
/// @details Legacy KFD and PCI/VFIO front ends register address spaces here and
/// receive generation-checked handles. Numeric VMIDs and PASIDs remain routing
/// metadata; they are not lifetime-safe identities. Queue references prevent a
/// binding from being revoked while a command processor can still use it.
class GpuVm {
public:
  explicit GpuVm(GpuMemory *memory = nullptr, Gfx12VmConfig gfx12_config = Gfx12VmConfig::gfx12_1())
      : memory_(memory), gfx12_config_(gfx12_config) {}

  void set_memory(GpuMemory *memory);

  [[nodiscard]] AddressSpaceHandle register_legacy(uint32_t vmid);
  [[nodiscard]] AddressSpaceHandle
  register_translated(uint32_t vmid, std::shared_ptr<AddressSpaceTranslator> translator,
                      std::shared_ptr<PhysicalMemoryAccess> memory);
  /// @brief Register a GFX12 process root using this VM's product translation policy.
  [[nodiscard]] AddressSpaceHandle
  register_gfx12_address_space(uint32_t vmid, uint64_t page_table_base,
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
  [[nodiscard]] bool unregister_address_space(AddressSpaceHandle handle);

  /// @brief Capture one coherent translator/backing/epoch tuple for an operation.
  [[nodiscard]] std::optional<GpuVmAccess> snapshot(AddressSpaceHandle handle) const;

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
  struct Binding {
    uint32_t vmid = 0;
    uint64_t translation_epoch = 1;
    uint32_t queue_references = 0;
    std::shared_ptr<AddressSpaceTranslator> translator;
    std::shared_ptr<PhysicalMemoryAccess> physical_memory;
    std::function<void(uint64_t, VmAccessKind)> fault_reporter;
    bool legacy_compatibility = false;
    bool device_gart = false;
  };

  struct Slot {
    uint64_t generation = 1;
    std::optional<Binding> binding;
  };

  [[nodiscard]] AddressSpaceHandle allocate_locked(Binding binding);
  [[nodiscard]] Binding *find_locked(AddressSpaceHandle handle);
  [[nodiscard]] const Binding *find_locked(AddressSpaceHandle handle) const;

  mutable std::mutex mutex_;
  GpuMemory *memory_ = nullptr;
  std::vector<Slot> slots_;
  std::vector<uint32_t> free_slots_;
  std::unordered_map<uint32_t, AddressSpaceHandle> vmid_handles_;
  AddressSpaceHandle gart_address_space_;
  uint64_t reset_epoch_ = 1;
  Gfx12VmConfig gfx12_config_;
};

} // namespace rocjitsu::amdgpu
