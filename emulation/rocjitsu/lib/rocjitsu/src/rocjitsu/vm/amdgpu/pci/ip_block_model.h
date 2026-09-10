// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file ip_block_model.h
/// @brief The behaviour of one published IP block, and the window it answers
///        registers through.
///
/// @details The device is three layers, and this is the middle one. A discovery
/// profile is pure data: which blocks a part publishes, at which versions, with
/// which register segments. A model is behaviour, keyed on the block *version*
/// rather than on the part. The device is the part-independent shell that owns
/// both.
///
/// Behaviour binds to the version because the driver binds it that way.
/// `amdgpu_discovery.c` selects every block implementation by
/// `IP_VERSION(major, minor, revision)`, so two parts that publish the same
/// version get the same implementation and two versions of one block get
/// different ones. Modelling it any other way would let a profile for one
/// version be answered with another version's register addresses, which is not
/// approximately right -- it is describing different hardware.
///
/// A model declares what it answers as claims relative to *its own block's*
/// segments, the way that block's offset header spells them, and never sees an
/// absolute address. Resolving a claim needs the segment list the discovery
/// table published for that particular block instance, which only the device
/// has; a model that resolved its own would be hardcoding one profile's layout
/// into behaviour that is supposed to be profile-independent.
///
/// A model is a plain owned object rather than a @c simdojo::Component.
/// Making it a component would give the device tree a second component base and
/// conscript every block into simulation partitioning, for a model whose whole
/// job is to answer register reads on the thread that services them.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/pci/interrupt_ring.h"
#include "rocjitsu/vm/amdgpu/pci/register_aperture.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simdojo {
class PciTransportSession;
}

namespace rocjitsu {

/// @brief Memory paths an IP block may use after a guest submits work.
class PciMemoryAccess {
public:
  virtual ~PciMemoryAccess() = default;

  /// @brief Capture the exact PCI peer generation backing retained VM state.
  /// @details The returned session never retargets when a later peer attaches.
  [[nodiscard]] virtual std::shared_ptr<simdojo::PciTransportSession>
  capture_transport_session() const = 0;

  [[nodiscard]] virtual bool read_vram(uint64_t offset, std::span<std::byte> bytes) = 0;
  [[nodiscard]] virtual bool write_vram(uint64_t offset, std::span<const std::byte> bytes) = 0;
  [[nodiscard]] virtual amdgpu::AtomicLoadResult atomic_load_vram(uint64_t offset,
                                                                  uint32_t width) = 0;
  [[nodiscard]] virtual amdgpu::VmAccessOutcome atomic_store_vram(uint64_t offset, uint32_t width,
                                                                  uint64_t value) = 0;
  [[nodiscard]] virtual amdgpu::AtomicCompareExchangeResult
  compare_exchange_vram(uint64_t offset, uint32_t width, uint64_t expected, uint64_t desired) = 0;
  [[nodiscard]] virtual bool read_register(uint64_t byte_offset, uint32_t &value) = 0;
  [[nodiscard]] virtual bool write_register(uint64_t byte_offset, uint32_t value) = 0;
  [[nodiscard]] virtual bool deliver_interrupt(const InterruptEntry &entry) = 0;
};

/// @brief Registers a model answers, named the way its block's headers name them.
///
/// @details Every register the driver touches carries a `_BASE_IDX` naming which
/// of the block's segments it lives in, and `SOC15_REG_OFFSET` resolves it as
/// `base[ip][instance][BASE_IDX] + reg`. A claim is that pair, plus how many
/// consecutive registers it covers, so a block of eighteen invalidation engines
/// is one claim rather than eighteen.
struct RegisterClaim {
  uint32_t segment_index = 0; ///< Which of the block's published bases, its `_BASE_IDX`.
  uint32_t first_dword = 0;   ///< Dwords from that base to the first register.
  uint32_t count = 1;         ///< Consecutive registers claimed.
};

/// @brief What a block did with a deferred doorbell notification.
enum class DoorbellDisposition : uint8_t {
  Ignored,  ///< The doorbell does not belong to this block.
  Complete, ///< The block consumed the notification without deferred work.
  Retry,    ///< A transient dependency prevented completion; retry this block only.
  Faulted,  ///< The notification or its retained work failed terminally.
};

/// @brief One published block instance's registers, addressed as its own
///        headers address them.
///
/// @details Cheap to copy and non-owning: it refers to the device's register
/// storage and to the segment list of one block record, both of which outlive
/// every model. It exists so a model can say "segment 0, register 0x1645"
/// and have somebody else decide what byte that is.
class IpRegisterWindow final {
public:
  /// @brief Address @p registers through @p segments.
  /// @param[in] registers Storage to answer out of; must outlive this window.
  /// @param[in] segments The block record's register bases; likewise.
  /// @param[in] owner Owner's name, for diagnostics.
  IpRegisterWindow(RegisterAperture &registers, std::span<const uint64_t> segments,
                   std::string owner);

  /// @brief Byte offset of a register, when the aperture can reach it.
  ///
  /// @details Refuses a segment that would wrap when scaled to a byte address,
  /// which is not hypothetical: a segment near the top of the range multiplies
  /// into a small offset that passes an aperture bounds test and lands on an
  /// unrelated register. Checked here rather than by each caller, because every
  /// caller computes the same `(segment + register) * 4`.
  ///
  /// @param[in] segment_index Which of the block's bases.
  /// @param[in] dword Register index from that base.
  /// @returns Its byte offset in the aperture, or nothing when the block does
  ///          not publish that segment or the register falls outside.
  [[nodiscard]] std::optional<uint64_t> resolve(uint32_t segment_index, uint32_t dword) const;

  /// @brief Model a register at its power-on value.
  /// @copydetails resolve
  /// @param[in] value What the register starts at.
  /// @retval false The register is not reachable, and has been reported.
  bool define(uint32_t segment_index, uint32_t dword, uint32_t value);

  /// @brief Model a register that answers reads and ignores writes.
  /// @copydoc define
  bool define_read_only(uint32_t segment_index, uint32_t dword, uint32_t value);

  /// @brief What a register holds.
  /// @returns Its value, or zero when it is not reachable. Zero rather than an
  ///          optional because every caller is decoding a field the driver has
  ///          not necessarily written, and an unreachable register and an
  ///          unwritten one both mean "the driver has not said".
  [[nodiscard]] uint32_t read(uint32_t segment_index, uint32_t dword) const;

  /// @brief Store into a register the model owns.
  /// @retval false The register is not reachable or is not modelled.
  bool write(uint32_t segment_index, uint32_t dword, uint32_t value);

  /// @brief Owner's name, for diagnostics.
  [[nodiscard]] const std::string &owner() const { return owner_; }

private:
  RegisterAperture *registers_;
  std::span<const uint64_t> segments_;
  std::string owner_;
};

/// @brief The behaviour of one published IP block.
///
/// @details Owned by the device, one per block record it can model. Register
/// notifications may originate on a transport callback thread, while doorbell
/// notifications run on the PCI component's simulation-owner thread. Shared
/// backing such as @ref RegisterAperture therefore provides its own narrow
/// synchronization; models must not hold model-state locks across DMA.
class IpBlockModel {
public:
  virtual ~IpBlockModel() = default;

  // Holds a window onto storage the device owns, so it is neither copied nor
  // moved out from under that storage.
  IpBlockModel(const IpBlockModel &) = delete;
  IpBlockModel &operator=(const IpBlockModel &) = delete;
  IpBlockModel(IpBlockModel &&) = delete;
  IpBlockModel &operator=(IpBlockModel &&) = delete;

  /// @brief Every register this model answers, relative to its block's segments.
  ///
  /// @details Declared rather than discovered, so the device can resolve them
  /// all once and refuse a profile in which two models claim the same register.
  [[nodiscard]] virtual std::vector<RegisterClaim> claims() const = 0;

  /// @brief Return every claimed register to its power-on value.
  /// @retval false Some register could not be modelled; it has been reported.
  virtual bool reset() = 0;

  /// @brief Observe a write that has already landed in the doorbell aperture.
  ///
  /// @details Most blocks have no doorbell-side behaviour. Blocks that do can
  /// decode the offset against registers the driver programmed in their own
  /// window, without teaching the PCI-device shell a version-specific layout.
  ///
  /// @param[in] byte_offset Byte offset within the doorbell aperture.
  /// @param[in] value The 32-bit value the guest wrote.
  [[nodiscard]] virtual DoorbellDisposition observe_doorbell_write(uint64_t byte_offset,
                                                                   uint64_t value,
                                                                   std::size_t width,
                                                                   PciMemoryAccess &memory) {
    (void)byte_offset;
    (void)value;
    (void)width;
    (void)memory;
    return DoorbellDisposition::Ignored;
  }

  /// @brief Observe a modelled register write after it has reached storage.
  virtual void observe_register_write(uint64_t byte_offset, PciMemoryAccess &memory) {
    (void)byte_offset;
    (void)memory;
  }

  /// @brief What this model is, for a diagnostic.
  [[nodiscard]] const std::string &what() const { return what_; }

protected:
  /// @param[in] what What this model is, for diagnostics.
  /// @param[in] registers Window onto its block's registers.
  IpBlockModel(std::string what, IpRegisterWindow registers);

  IpRegisterWindow registers_; ///< This block's registers, as its headers name them.

private:
  std::string what_;
};

/// @brief A range of the register aperture one model has claimed.
class ClaimedRegisters {
public:
  std::string owner;        ///< What claimed it, for the diagnostic.
  uint64_t first_dword = 0; ///< First register of the range, absolute.
  uint64_t count = 0;       ///< How many it covers.
};

/// @brief Why two claims cannot both be answered, if they cannot.
///
/// @details Two models answering one register is a construction-time bug rather
/// than a runtime surprise: whichever defined it last wins, silently, and the
/// block that lost stalls the driver on a register that reads as somebody
/// else's. Worth being loud about, and worth checking rather than assuming,
/// because blocks legitimately share segments -- GC and SDMA0 publish identical
/// bases on this family, as do the two management processors -- and stay
/// disjoint only in the offsets they claim within them. An absolute-dword map
/// is therefore well defined, and an overlap in it is always a mistake.
///
/// @param[in] claims Every resolved claim, in any order.
/// @returns An empty string when they are disjoint, or what overlaps what.
[[nodiscard]] std::string overlapping_claim(std::vector<ClaimedRegisters> claims);

} // namespace rocjitsu
