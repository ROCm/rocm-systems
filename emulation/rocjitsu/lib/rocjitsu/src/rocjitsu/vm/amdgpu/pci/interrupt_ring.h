// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interrupt_ring.h
/// @brief The interrupt ring a guest driver programs, and what travels in it.
///
/// @details These describe the OSSSYS interrupt-handler block rather than the
/// PCI function that happens to answer its registers today. They are separate
/// from the device for that reason: the register model is being split into one
/// unit per IP block, and this is the state the interrupt block will own. A
/// consumer that wants to know what the driver said about its ring should not
/// have to include the whole device to find out.

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace rocjitsu {

/// @brief Which address space an interrupt ring's addresses are in.
///
/// @details The driver puts this in the same register as the ring's size and
/// its enable bit, and it decides what the addresses beside it *mean*. It
/// follows how firmware is loaded: a driver loading firmware through the
/// security processor allocates the ring through the translation tables and
/// programs virtual addresses, and every other way of loading gets a bus
/// address. So identical register values denote different memory depending on
/// a module parameter, and an address used in the wrong space does not fail --
/// it points somewhere real and wrong.
enum class InterruptRingSpace : uint8_t {
  /// @brief Not one of the two values the driver writes; in practice, a ring
  /// it has not programmed yet.
  Unset = 0,
  BusAddress = 2, ///< Guest physical, reachable through a shared window.
  GpuVirtual = 4  ///< Behind translation tables this device does not walk.
};

/// @brief The interrupt ring as the driver has programmed it so far.
///
/// @details The ring is the device's side of the interrupt path: the device
/// writes an entry into it, publishes a write pointer, and raises a message.
/// All of that needs to know where the driver put the ring, which the driver
/// says only by writing these registers -- so this is read back out of them
/// rather than tracked as they are written, because the driver writes them in
/// its own order and rewrites them on reset.
struct InterruptRing {
  /// @brief Address of the ring in @ref space, or zero if it is not set.
  uint64_t base = 0;
  uint64_t bytes = 0;        ///< Its size, decoded from the size field.
  uint64_t wptr_address = 0; ///< Where the device publishes the write pointer.
  /// @brief What @ref base and @ref wptr_address are addresses in.
  InterruptRingSpace space = InterruptRingSpace::Unset;
  bool enabled = false;         ///< Whether the driver has switched the ring on.
  bool raises_messages = false; ///< Whether it wants an interrupt per entry.

  /// @brief Whether the driver has said anything at all about the ring.
  ///
  /// @details Every field, rather than the address and the enable bit alone:
  /// a driver that sized a ring and named a write-pointer address but had not
  /// switched it on yet has said a great deal, and reporting that as nothing
  /// programmed would hide exactly the partial state worth seeing. These read
  /// back as their defaults only while the registers are untouched.
  /// @retval false Every field still holds what a reset left.
  [[nodiscard]] bool programmed() const {
    return base != 0 || bytes != 0 || wptr_address != 0 || space != InterruptRingSpace::Unset ||
           enabled || raises_messages;
  }
};

/// @brief One interrupt, as the ring carries it.
///
/// @details The driver looks up a handler by the pair of identifiers and
/// passes the rest to it. Everything else an entry can carry -- timestamps,
/// process and node identifiers -- describes work this device does not run,
/// so it is left zero rather than invented.
struct InterruptEntry {
  uint8_t client_id = 0;             ///< Which block is reporting.
  uint8_t source_id = 0;             ///< What it is reporting.
  uint8_t ring_id = 0;               ///< Which engine and queue produced it.
  uint8_t vmid = 0;                  ///< Hardware VMID carried in the interrupt cookie.
  uint16_t pasid = 0;                ///< Process address-space ID carried in the cookie.
  uint8_t node_id = 0;               ///< Which interrupt-hierarchy node produced it.
  std::array<uint32_t, 4> data = {}; ///< Payload words the handler receives.
};

/// @brief Render a ring for a diagnostic.
///
/// @details A sentence fragment beginning "its interrupt ring at ...", meant
/// to follow a subject and verb naming who did what -- "the driver enabled",
/// "the driver left". Logged on its own it reads as though it lost a word.
///
/// @param[in] ring The ring to describe.
/// @returns Where it is, how big, and in which address space.
[[nodiscard]] std::string describe(const InterruptRing &ring);

} // namespace rocjitsu
