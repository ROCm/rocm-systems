// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interrupt_block_model.h
/// @brief The OSSSYS interrupt-handler block: its ring registers, and delivery
///        through them.
///
/// @details Keyed on the OSSSYS version rather than on the part, because the
/// driver keys it that way and keys it sharply:
/// `amdgpu_discovery_set_ih_ip_blocks` switches on the version and binds an
/// entirely different implementation per arm -- OSSSYS 7.1 gets `ih_v7_0` while
/// 4.4 gets `vega20_ih`, with its own register layout. A device answering one
/// version's addresses for another is not modelling the hardware slightly
/// wrong; it is programming registers nothing ever reads, which presents as a
/// GPU that accepts an interrupt ring and never reports anything through it.
///
/// A version with no known layout is therefore refused rather than guessed at.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/interrupt_ring.h"
#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "simdojo/components/pci_device.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {

/// @brief One interrupt block's ring registers, and the version they came from.
///
/// @details One row per layout read out of an offset header and checked.
/// `regIH_RB_{CNTL,RPTR,WPTR,BASE,BASE_HI,WPTR_ADDR_HI,WPTR_ADDR_LO}` and
/// `regIH_DOORBELL_RPTR` of `osssys_7_1_0_offset.h`, all `_BASE_IDX 0`. The last
/// matters more than it looks: the ring is created with doorbells
/// unconditionally on, so the driver acknowledges entries by writing a doorbell
/// rather than the read-pointer register.
struct InterruptRingLayout {
  uint16_t major = 0;                      ///< OSSSYS major version this was read from.
  uint16_t minor = 0;                      ///< OSSSYS minor version, likewise.
  uint16_t revision = 0;                   ///< OSSSYS revision, likewise.
  uint32_t control = 0;                    ///< Ring control, in dwords from segment 0.
  uint32_t read_pointer = 0;               ///< Ring read pointer, likewise.
  uint32_t write_pointer = 0;              ///< Ring write pointer, likewise.
  uint32_t base = 0;                       ///< Ring base, low half.
  uint32_t base_high = 0;                  ///< Ring base, high half.
  uint32_t write_pointer_address_high = 0; ///< Where the write pointer is published, high half.
  uint32_t write_pointer_address_low = 0;  ///< The same, low half.
  uint32_t doorbell = 0;                   ///< Ring doorbell control.
};

/// @brief The interrupt-handler block of one published OSSSYS record.
class InterruptBlockModel final : public IpBlockModel {
public:
  /// @brief Model @p block, when its version has a known register layout.
  ///
  /// @param[in] block The published OSSSYS record.
  /// @param[in] registers Window onto that record's registers.
  /// @returns The model, or nullptr for a version with no known layout, which
  ///          has been reported.
  [[nodiscard]] static std::unique_ptr<InterruptBlockModel> create(const IpBlock &block,
                                                                   IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;

  /// @brief Read the ring out of the registers the driver wrote.
  ///
  /// @details Read back rather than tracked as the registers are written,
  /// because the driver writes them in its own order and rewrites them on reset.
  [[nodiscard]] InterruptRing ring() const;

  /// @brief How far the driver has read the ring, when it has said.
  ///
  /// @details The driver acknowledges entries by writing a doorbell rather than
  /// the read-pointer register, so the value lives in the doorbell page and is
  /// reached through the index @c IH_DOORBELL_RPTR names -- qualified by that
  /// register's enable bit, since an index the driver has not asked to be used
  /// names whatever happens to be there.
  ///
  /// @param[in] doorbells The doorbell aperture's backing.
  /// @param[in] ring The ring, whose size bounds a legal pointer.
  /// @returns The byte offset the driver has consumed to, or nothing when it
  ///          has not said or said something the ring cannot hold.
  [[nodiscard]] std::optional<uint32_t> read_pointer(std::span<const std::byte> doorbells,
                                                     const InterruptRing &ring) const;

  /// @brief Put one entry in the ring, publish the pointer, and raise a message.
  ///
  /// @details The whole delivery, because the parts are only meaningful
  /// together: an entry the driver never sees, a write pointer naming an entry
  /// that is not there, or a message with nothing behind it are each worse than
  /// doing nothing.
  ///
  /// @param[in] entry What to report.
  /// @param[in] programmed Stable snapshot of the programmed ring.
  /// @param[in] consumed Stable snapshot of the driver's read pointer, when valid.
  /// @param[in] dma How to reach guest memory.
  /// @param[in] irq How to raise the message.
  /// @retval false Nothing was delivered, or not all of it was.
  [[nodiscard]] bool deliver(const InterruptEntry &entry, const InterruptRing &programmed,
                             std::optional<uint32_t> consumed, simdojo::DmaEngine &dma,
                             simdojo::IrqSink &irq);

  /// @brief Note a guest write that has already landed, and report a ring being
  ///        switched on the first time it is.
  ///
  /// @details Said once, when it happens, rather than left to be read back
  /// later: a reset clears these registers, so anything asked afterwards finds
  /// a device that was never told.
  ///
  /// @param[in] byte_offset Where the guest wrote, in the register aperture.
  void observe_write(uint64_t byte_offset);

private:
  InterruptBlockModel(IpRegisterWindow registers, const InterruptRingLayout &layout);

  /// @brief Read one of this block's registers, by its `_BASE_IDX 0` offset.
  [[nodiscard]] uint32_t value(uint32_t dword) const;

  const InterruptRingLayout *layout_;

  /// @brief Whether the ring being switched on has already been reported.
  std::atomic<bool> announced_{false};

  /// @brief Whether an overflow remains unacknowledged by the driver.
  std::atomic<bool> overflow_latched_{false};
};

} // namespace rocjitsu
