// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_aperture.h
/// @brief The dense register storage behind a device's register BAR.
///
/// @details Split out of the device because it is what an IP block model has to
/// reach: a model answers registers, and it can only do that against storage
/// somebody else owns. Keeping the storage here rather than as three parallel
/// vectors inside the device means a window onto it can be handed to a model
/// without handing over the device.
///
/// Three facts per dword, kept apart on purpose. The value alone cannot say
/// whether a register is modelled -- zero is both a legitimate value and what
/// absent hardware reads as -- and telling those apart is the whole point of the
/// unmodelled-register report. Whether a register ignores writes is a third
/// fact, because what a semaphore reports is a property of the hardware rather
/// than state the driver owns.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief One dword per register in a device's register aperture.
/// @details Individual operations are synchronized because transport MMIO and
/// deferred device events can access the aperture on different threads.
class RegisterAperture final {
public:
  /// @brief Back an aperture of @p bytes, with nothing modelled yet.
  /// @param[in] owner Owner's name, for diagnostics.
  /// @param[in] bytes Size of the register BAR.
  RegisterAperture(std::string owner, uint64_t bytes);

  /// @brief Size of the aperture, in bytes.
  [[nodiscard]] uint64_t bytes() const { return bytes_; }

  /// @brief Registers the aperture holds.
  [[nodiscard]] uint64_t dwords() const { return values_.size(); }

  /// @brief Whether @p byte_offset names a register inside the aperture.
  [[nodiscard]] bool holds(uint64_t byte_offset) const;

  /// @brief Model the register at @p byte_offset, giving it a starting value.
  ///
  /// @details Refuses an offset outside the aperture rather than trusting its
  /// callers. Most offsets are constants the validated minimum aperture covers,
  /// but block registers are derived from segment addresses a discovery profile
  /// supplies, so a profile is enough to aim this past the end of the storage.
  /// Out of range it reports and defines nothing, which reads back as absent
  /// hardware -- the same outcome as never having named it, and diagnosable.
  ///
  /// @param[in] byte_offset Aperture-relative byte address of the register.
  /// @param[in] value Value the register starts at and returns to on reset.
  /// @retval false The offset lies outside the aperture or is unaligned.
  bool define(uint64_t byte_offset, uint32_t value);

  /// @brief Define a register that answers reads and ignores writes.
  ///
  /// @details For registers whose value is a property of the hardware rather
  /// than state the driver owns. A semaphore the device always grants is the
  /// case in point: the driver releases it by writing zero, and a register that
  /// stored that write would grant the acquire once and then stall forever.
  ///
  /// @copydoc define
  bool define_read_only(uint64_t byte_offset, uint32_t value);

  /// @brief Whether the device models the register at @p byte_offset.
  [[nodiscard]] bool modelled(uint64_t byte_offset) const;

  /// @brief Whether that register ignores writes.
  [[nodiscard]] bool read_only(uint64_t byte_offset) const;

  /// @brief What the register at @p byte_offset holds.
  /// @returns Its value, or zero when the offset is outside the aperture.
  [[nodiscard]] uint32_t value(uint64_t byte_offset) const;

  /// @brief Store @p value in the register at @p byte_offset.
  ///
  /// @details Stores whatever it is given, including into a read-only register:
  /// this is the model writing, not the guest. Deciding whether a guest write
  /// is allowed to land belongs to whoever services the access.
  ///
  /// @retval false The offset lies outside the aperture, or names a register
  ///               nothing has modelled.
  bool store(uint64_t byte_offset, uint32_t value);

private:
  /// @brief Index of @p byte_offset, or the dword count when it is not one.
  [[nodiscard]] std::size_t index_of(uint64_t byte_offset) const;

  std::string owner_;
  uint64_t bytes_ = 0;
  mutable std::unique_ptr<std::mutex> mutex_ = std::make_unique<std::mutex>();
  std::vector<uint32_t> values_;

  /// @brief Which registers the device actually models.
  std::vector<bool> modelled_;

  /// @brief Which of the modelled registers ignore writes.
  std::vector<bool> read_only_;
};

} // namespace rocjitsu
