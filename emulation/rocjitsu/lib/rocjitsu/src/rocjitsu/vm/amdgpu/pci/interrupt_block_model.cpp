// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/interrupt_block_model.h"

#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "util/log.h"

#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <utility>

namespace rocjitsu {
namespace {

/// @details OSSSYS 4.4.x is deliberately absent rather than guessed at: it binds
/// `vega20_ih`, a different implementation with its own layout, and a row
/// invented for it would be indistinguishable from hardware whose ring never
/// fills.
constexpr InterruptRingLayout kInterruptRingLayouts[] = {
    {.major = 7,
     .minor = 1,
     .revision = 0,
     .control = 0x0080,
     .read_pointer = 0x0081,
     .write_pointer = 0x0082,
     .base = 0x0083,
     .base_high = 0x0084,
     .write_pointer_address_high = 0x0085,
     .write_pointer_address_low = 0x0086,
     .doorbell = 0x0087},
};

/// @brief The segment every register above is measured from, its `_BASE_IDX`.
constexpr uint32_t kRingSegment = 0;

/// @brief Bit the device sets in the published write pointer to report that
/// entries were lost.
///
/// @details `IH_RB_WPTR__RB_OVERFLOW_MASK`. This is how real hardware says the
/// ring wrapped onto entries the driver had not read: the driver checks it in
/// the copy it reads from memory first, confirms against the register, warns,
/// and resumes from `wptr + 32` having accepted the loss
/// (`ih_v7_0_get_wptr`). Reporting it is therefore not a courtesy -- it is the
/// only signal that distinguishes a ring that lost entries from one that never
/// received them, and the driver already knows what to do with it.
constexpr uint32_t kIhWritePointerOverflow = 0x1;

/// @brief Enables the doorbell the driver acknowledges entries through.
///
/// @details `IH_DOORBELL_RPTR__ENABLE_MASK`. Qualifies the offset beside it: an
/// offset read without checking this is a doorbell index the driver has not
/// asked to be used.
constexpr uint32_t kIhDoorbellEnable = 0x10000000;

/// @brief The doorbell index within @c IH_DOORBELL_RPTR.
///
/// @details `IH_DOORBELL_RPTR__OFFSET_MASK`.
constexpr uint32_t kIhDoorbellOffsetMask = 0x03ffffff;

/// @brief Bits the driver shifts the ring's address down by before writing it.
///
/// @details The base register holds bits 39:8, so the low eight are implied
/// zero and the ring is at least 256-byte aligned. The address itself does not
/// stop at 39: the high register below continues it from bit 40, and the two
/// together carry 48 bits.
constexpr unsigned kIhRingBaseShift = 8;

/// @brief Address bit at which the high half of the base register continues.
constexpr unsigned kIhRingBaseHighShift = 40;

/// @brief Bits of the write-pointer address carried by its high register.
constexpr uint64_t kIhWritePointerAddressHighMask = 0xffff;

/// @brief Bit selecting the ring within the control register.
constexpr uint32_t kIhRingEnableMask = 1U << 0;

/// @brief Bit asking for an interrupt per entry.
constexpr uint32_t kIhRingInterruptEnableMask = 1U << 17;

/// @brief Control-register pulse that acknowledges a reported overflow.
///
/// @details `IH_RB_CNTL__WPTR_OVERFLOW_CLEAR_MASK`. The driver writes one and
/// then zero after accepting the lost entries; advancing the read-pointer
/// doorbell does not acknowledge the overflow.
constexpr uint32_t kIhWritePointerOverflowClear = 1U << 31;

/// @brief Where the size field sits within the control register.
///
/// @details The driver stores the base-two logarithm of the ring's size in
/// dwords, so a size of `4 << field` bytes.
constexpr unsigned kIhRingSizeShift = 1;

/// @brief Its width, once shifted down: the field mask is `0x3e`.
constexpr uint32_t kIhRingSizeMask = 0x1f;

/// @brief Where the address-space field sits within the control register.
constexpr unsigned kIhRingSpaceShift = 28;

/// @brief Its width, once shifted down: the field mask is `0x70000000`.
constexpr uint32_t kIhRingSpaceMask = 0x7;

/// @brief Bits of the ring's base carried by its high register.
///
/// @details Narrower than the register's own field, which is seventeen bits,
/// because the driver only ever writes eight of them.
constexpr uint64_t kIhRingBaseHighMask = 0xff;

/// @brief Dwords one interrupt entry occupies, and the bytes that comes to.
///
/// @details The driver advances its read pointer by this much per entry and
/// decodes exactly this many dwords, so an entry of any other size would put
/// every later entry at an offset it does not look at.
constexpr uint32_t kInterruptEntryDwords = 8;
constexpr uint32_t kInterruptEntryBytes = kInterruptEntryDwords * 4;

/// @brief Bits of the write-pointer register that carry the offset.
///
/// @details `IH_RB_WPTR__OFFSET_MASK`. The offset occupies bits 17:2; bits 1:0
/// are the overflow flag, so the register does not carry the low two bits of an
/// address at all. Anything above the field is not part of one either.
constexpr uint32_t kIhWritePointerOffsetMask = 0x0003fffc;

/// @brief Largest ring whose every entry the write-pointer register can name.
///
/// @details The offset field is sixteen bits wide, so it addresses exactly one
/// 256 KiB ring. A larger one would have entries the device could never point
/// at, and worse, the device would wrap at the field width while the driver
/// wrapped at the ring size -- the two would disagree about which entry a
/// pointer names, and the driver would decode the never-written remainder as
/// entries. The size field is five bits and guest-writable, so it can ask for
/// far more than this.
constexpr uint64_t kLargestAddressableRingBytes =
    (uint64_t{kIhWritePointerOffsetMask} & ~uint64_t{kInterruptEntryBytes - 1}) +
    kInterruptEntryBytes;

/// @brief The message vector an entry is announced on.
///
/// @details The device advertises one, so this is it. A second would need the
/// capability to advertise it before anything could be delivered on it.
constexpr uint32_t kInterruptVector = 0;

} // namespace

std::string describe(const InterruptRing &ring) {
  // Initialised to the fallback and switched without a default: a fourth space
  // added later is a compiler warning here, while a value cast in from outside
  // the enumerators still renders as something rather than as a null pointer.
  const char *space = "an unstated address space";
  switch (ring.space) {
  case InterruptRingSpace::BusAddress:
    space = "a bus address";
    break;
  case InterruptRingSpace::GpuVirtual:
    space = "a translated address";
    break;
  case InterruptRingSpace::Unset:
    space = "an unstated address space";
    break;
  }
  // Whether it was switched on is part of the state, not a detail: a sized and
  // addressed ring that the driver never enabled is a different situation from
  // a running one, and reading the two the same way sends whoever is
  // diagnosing a silent guest looking in the wrong place.
  return std::format("its interrupt ring at {:#x}, {} bytes, {}, publishing its write pointer to "
                     "{:#x}; it {} an interrupt per entry, and it is {}",
                     ring.base, ring.bytes, space, ring.wptr_address,
                     ring.raises_messages ? "asks for" : "does not ask for",
                     ring.enabled ? "enabled" : "not yet enabled");
}

InterruptBlockModel::InterruptBlockModel(IpRegisterWindow registers,
                                         const InterruptRingLayout &layout)
    : IpBlockModel("the interrupt handler", std::move(registers)), layout_(&layout) {}

std::unique_ptr<InterruptBlockModel> InterruptBlockModel::create(const IpBlock &block,
                                                                 IpRegisterWindow registers) {
  for (const InterruptRingLayout &candidate : kInterruptRingLayouts) {
    if (candidate.major == block.major && candidate.minor == block.minor &&
        candidate.revision == block.revision) {
      return std::unique_ptr<InterruptBlockModel>(
          new InterruptBlockModel(std::move(registers), candidate));
    }
  }
  // Refused rather than answered with another version's addresses. A wrong
  // offset here is not a lesser interrupt path; it is registers the driver
  // programs and this device never reads, which presents as hardware that takes
  // an interrupt and never reports one.
  util::Logger::warn(std::format(
      "{}: no interrupt-ring layout is known for OSSSYS {}.{}.{}, so the ring this device would "
      "write to is not the one the driver programmed",
      registers.owner(), block.major, block.minor, block.revision));
  return nullptr;
}

std::vector<RegisterClaim> InterruptBlockModel::claims() const {
  std::vector<RegisterClaim> claimed;
  for (const uint32_t reg : {layout_->control, layout_->read_pointer, layout_->write_pointer,
                             layout_->base, layout_->base_high, layout_->write_pointer_address_high,
                             layout_->write_pointer_address_low, layout_->doorbell}) {
    claimed.push_back({.segment_index = kRingSegment, .first_dword = reg, .count = 1});
  }
  return claimed;
}

bool InterruptBlockModel::reset() {
  // The ring's registers answer as plain storage: the driver writes where it
  // put the ring and reads its own pointers back, so anything the device
  // invented here would be a fact the driver did not state.
  bool defined = true;
  for (const RegisterClaim &claim : claims()) {
    for (uint32_t i = 0; i < claim.count; ++i) {
      defined = registers_.define(claim.segment_index, claim.first_dword + i, 0) && defined;
    }
  }
  // Cleared with the registers it describes: a reset discards what the driver
  // said, so the next programming has to be reported as freshly as the first.
  announced_.store(false, std::memory_order_relaxed);
  overflow_latched_.store(false, std::memory_order_relaxed);
  return defined;
}

uint32_t InterruptBlockModel::value(uint32_t dword) const {
  return registers_.read(kRingSegment, dword);
}

InterruptRing InterruptBlockModel::ring() const {
  InterruptRing ring;
  ring.base = (static_cast<uint64_t>(value(layout_->base)) << kIhRingBaseShift) |
              (static_cast<uint64_t>(value(layout_->base_high) & kIhRingBaseHighMask)
               << kIhRingBaseHighShift);
  ring.wptr_address = static_cast<uint64_t>(value(layout_->write_pointer_address_low)) |
                      ((static_cast<uint64_t>(value(layout_->write_pointer_address_high)) &
                        kIhWritePointerAddressHighMask)
                       << 32);

  const uint32_t control = value(layout_->control);
  ring.enabled = (control & kIhRingEnableMask) != 0;
  ring.raises_messages = (control & kIhRingInterruptEnableMask) != 0;
  // The field beside them saying what the addresses above are addresses in.
  // Anything the driver has not written yet reads as zero, which is neither of
  // the two values it uses and is reported as such rather than guessed at.
  switch ((control >> kIhRingSpaceShift) & kIhRingSpaceMask) {
  case static_cast<uint32_t>(InterruptRingSpace::BusAddress):
    ring.space = InterruptRingSpace::BusAddress;
    break;
  case static_cast<uint32_t>(InterruptRingSpace::GpuVirtual):
    ring.space = InterruptRingSpace::GpuVirtual;
    break;
  default:
    ring.space = InterruptRingSpace::Unset;
    break;
  }
  // The field is the logarithm of the size in dwords, so the size is four bytes
  // shifted up by it. A ring the driver has not sized yet reads as zero rather
  // than as the four bytes that shift would otherwise imply.
  const uint32_t size_log = (control >> kIhRingSizeShift) & kIhRingSizeMask;
  ring.bytes = size_log == 0 ? 0 : uint64_t{4} << size_log;
  return ring;
}

std::optional<uint32_t> InterruptBlockModel::read_pointer(std::span<const std::byte> doorbells,
                                                          const InterruptRing &ring) const {
  const uint32_t control = value(layout_->doorbell);
  // The enable bit qualifies the offset beside it. Reading the offset without it
  // would take a doorbell index the driver has not asked to be used and treat
  // whatever happens to be at it as a read pointer -- which is worse than not
  // knowing, because it would look like an answer.
  if ((control & kIhDoorbellEnable) == 0) {
    return std::nullopt;
  }
  const uint64_t at = byte_offset_of_dword(control & kIhDoorbellOffsetMask);
  if (at + kRegisterBytes > doorbells.size()) {
    return std::nullopt;
  }
  uint32_t published = 0;
  std::memcpy(&published, doorbells.data() + at, sizeof(published));
  // The driver writes a byte offset into the ring; anything outside it, or not
  // on an entry boundary, is not a pointer this device can compare against.
  if (ring.bytes == 0 || published >= ring.bytes || (published % kInterruptEntryBytes) != 0) {
    return std::nullopt;
  }
  return published;
}

bool InterruptBlockModel::deliver(const InterruptEntry &entry, const InterruptRing &programmed,
                                  std::optional<uint32_t> consumed, simdojo::DmaEngine &dma,
                                  simdojo::IrqSink &irq) {
  // Resolved once: the write pointer is the one register this writes back, and
  // a block whose segments the aperture cannot reach has nothing to write to.
  if (!registers_.resolve(kRingSegment, layout_->write_pointer).has_value()) {
    return false;
  }

  if (!programmed.enabled || programmed.bytes < kInterruptEntryBytes ||
      programmed.bytes > kLargestAddressableRingBytes) {
    return false;
  }
  // An address in a space this device cannot resolve would be written to
  // whatever guest page happens to sit at that number. That is worse than
  // declining: the driver would carry on waiting while memory it owns was
  // quietly changed underneath it.
  if (programmed.space != InterruptRingSpace::BusAddress) {
    return false;
  }
  // Publishing to nowhere is not publishing: the driver reads the pointer from
  // memory, so a ring switched on before that address was given has no way to
  // be told anything, and guest-physical zero is a real page to write over.
  //
  // The ring's own base is the same question and the same answer. A driver may
  // set the enable bit before it has written the base -- the registers are
  // separate and it writes them in its own order -- and zero there means unset,
  // not "the ring is at address zero".
  if (programmed.base == 0 || programmed.wptr_address == 0) {
    return false;
  }

  // The pointers are byte offsets into the ring and wrap with it. Taking the
  // current one from the register rather than from a member keeps the device's
  // idea of it and the driver's the same object rather than two that drift --
  // every driver-side re-init writes this register back to zero.
  //
  // It is a *guest-writable* register, though, and this is where its value
  // becomes an address. So it is put through the field mask the hardware
  // defines and then aligned down to an entry, because the driver's own write-
  // and read-pointer shadows sit immediately after the ring: an entry placed
  // at an unaligned offset would run off the end and overwrite exactly the two
  // words the interrupt protocol depends on.
  const uint64_t stated = value(layout_->write_pointer) & kIhWritePointerOffsetMask;
  const auto at =
      static_cast<uint32_t>((stated & ~uint64_t{kInterruptEntryBytes - 1}) % programmed.bytes);

  // How far the driver has consumed, when it has said. The driver acknowledges
  // by writing a doorbell rather than the read-pointer register, so the value
  // lives in the doorbell page it asked us to watch.
  const auto next = static_cast<uint32_t>((uint64_t{at} + kInterruptEntryBytes) % programmed.bytes);
  // Full means the write pointer would land on the read pointer: the ring holds
  // one entry fewer than its size, because the two pointers being equal is how
  // empty is spelled and the state cannot mean both.
  const bool newly_overflowed = consumed.has_value() && next == *consumed;
  if (newly_overflowed) {
    util::Logger::warn(std::format(
        "{}: the interrupt ring is full at byte {:#x} and the entry being delivered overwrites one "
        "the driver has not read; reporting an overflow",
        registers_.owner(), at));
  }

  std::array<uint32_t, kInterruptEntryDwords> words = {};
  words[0] = static_cast<uint32_t>(entry.client_id) |
             (static_cast<uint32_t>(entry.source_id) << 8) |
             (static_cast<uint32_t>(entry.ring_id) << 16) |
             (static_cast<uint32_t>(entry.vmid & 0xf) << 24);
  words[3] = static_cast<uint32_t>(entry.pasid) | (static_cast<uint32_t>(entry.node_id) << 16);
  words[4] = entry.data[0];
  words[5] = entry.data[1];
  words[6] = entry.data[2];
  words[7] = entry.data[3];

  std::array<std::byte, kInterruptEntryBytes> raw = {};
  std::memcpy(raw.data(), words.data(), raw.size());
  if (!dma.write(programmed.base + at, raw)) {
    return false;
  }

  // Only now is there something to point at. Publishing the pointer first would
  // invite the driver to read an entry that is not yet there -- the driver orders
  // its read of the ring against its read of this pointer with a barrier, so the
  // device owes it the matching store order.
  //
  // That order is DmaEngine::write()'s to keep, not this function's: it returns
  // only once the bytes are guest-visible, and writes become visible in the order
  // they return. A fence here would not have been enough anyway, since it cannot
  // order stores an implementation has only queued.
  //
  // The overflow bit rides on the published pointer because that is where the
  // driver looks for it first, and it is set in the register too because the
  // driver confirms against the register before believing the memory copy.
  const bool report_overflow =
      overflow_latched_.load(std::memory_order_acquire) || newly_overflowed;
  const uint32_t reported = report_overflow ? (next | kIhWritePointerOverflow) : next;
  const auto published = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(reported);
  if (!dma.write(programmed.wptr_address, published)) {
    return false;
  }
  if (newly_overflowed)
    overflow_latched_.store(true, std::memory_order_release);
  (void)registers_.write(kRingSegment, layout_->write_pointer, reported);

  // The ring can be switched on while messages for it are switched off, in
  // which case entries accumulate and the driver finds them when it next looks.
  // The driver moves the two bits together, so this is a state it never asks
  // for -- but the field is decoded, and decoding a field and then ignoring it
  // is how a model starts disagreeing with itself.
  if (!programmed.raises_messages) {
    return true;
  }
  return irq.trigger(kInterruptVector);
}

void InterruptBlockModel::observe_write(uint64_t byte_offset) {
  // Resolved rather than compared against a remembered address: the model never
  // holds an absolute offset of its own, and asking the window is what keeps the
  // one it compares against and the one it defines the same answer.
  const std::optional<uint64_t> control = registers_.resolve(kRingSegment, layout_->control);
  if (!control.has_value() || byte_offset != *control) {
    return;
  }

  // Overflow is sticky until this pulse. The driver may have enabled and
  // announced the ring long before it acknowledges a later loss, so this side
  // effect must be processed on every control-register write rather than only
  // the first one. Hardware clears the register indication immediately; the
  // writeback copy is replaced by the next pointer publication.
  if ((value(layout_->control) & kIhWritePointerOverflowClear) != 0) {
    overflow_latched_.store(false, std::memory_order_release);
    const uint32_t write_pointer = value(layout_->write_pointer) & ~kIhWritePointerOverflow;
    (void)registers_.write(kRingSegment, layout_->write_pointer, write_pointer);
  }

  const InterruptRing programmed = ring();
  if (!programmed.enabled) {
    return;
  }
  if (announced_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  util::Logger::warn(
      std::format("{}: the driver enabled {}", registers_.owner(), describe(programmed)));
  // An address the device cannot translate is worth saying plainly now rather
  // than leaving for whoever wonders why no interrupt arrived.
  if (programmed.space == InterruptRingSpace::GpuVirtual) {
    util::Logger::warn(
        std::format("{}: that ring is behind translation tables this device does not walk, so it "
                    "cannot be reached; the driver places it there whenever firmware is loaded "
                    "through the security processor",
                    registers_.owner()));
  } else if (programmed.space != InterruptRingSpace::BusAddress) {
    util::Logger::warn(
        std::format("{}: that ring names no address space, so where it is cannot be acted on",
                    registers_.owner()));
  }
}

} // namespace rocjitsu
