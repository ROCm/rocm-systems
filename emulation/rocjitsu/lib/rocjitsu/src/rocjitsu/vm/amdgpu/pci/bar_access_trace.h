// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file bar_access_trace.h
/// @brief Diagnostics for guest register accesses against an emulated device.
///
/// @details Two failure modes dominate when an unmodified kernel driver is
/// brought up against an emulated device, and neither is visible from the guest
/// side.
///
/// The first is a register the device does not model. The driver reads it, gets
/// a default value, and either misbehaves later or gives up. Every such access
/// is counted here so that one boot produces a ranked list of what to implement
/// next, rather than a single error at a time.
///
/// The second is a driver spinning on a status bit that the device will never
/// set. The guest simply stops making progress; because the driver is polling
/// rather than faulting, nothing is logged on either side and the boot appears
/// to hang. Repeated reads of one register with no intervening write are the
/// signature of that state, so they raise a warning that names the register.
///
/// A @ref rocjitsu::BarAccessTrace is fed from the device's BAR access
/// dispatcher and is safe to feed from the transport thread while another
/// thread reads a report.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rocjitsu {

/// @brief Records guest register accesses and flags the two silent failures.
class BarAccessTrace {
public:
  /// @brief Tuning for the spin detector.
  struct Config {
    /// @brief Consecutive reads of one register, with no intervening write,
    /// that are treated as a driver spinning on a status bit.
    ///
    /// @details Legitimate polling loops do exist, so this must be far above a
    /// plausible loop count; the value only needs to be reached before a human
    /// gives up waiting for the boot.
    uint32_t spin_threshold = 4096;
  };

  /// @brief Construct a trace with the default spin detector tuning.
  /// @param[in] symbols Register names; must outlive this object.
  explicit BarAccessTrace(const RegisterSymbols &symbols);

  /// @brief Construct a trace that symbolizes through @p symbols.
  /// @param[in] symbols Register names; must outlive this object.
  /// @param[in] config Spin detector tuning.
  BarAccessTrace(const RegisterSymbols &symbols, Config config);

  /// @brief Record one guest access to a BAR.
  /// @param[in] bar BAR index the access targeted.
  /// @param[in] offset Byte offset within the BAR.
  /// @param[in] width Access width in bytes.
  /// @param[in] write True for a write access, false for a read.
  /// @param[in] modeled False if the device has no model for this register and
  ///                    answered with a default.
  void record(int bar, uint64_t offset, std::size_t width, bool write, bool modeled);

  /// @brief Record an access the device refused outright.
  /// @details Distinct from an unmodeled register, which is a *successful*
  /// access answered with a default and therefore names hardware still to be
  /// built. A refusal -- a bad width, a foreign BAR, an offset past the end --
  /// names no hardware at all, and folding it into the burndown report invents
  /// work that does not exist.
  /// @param[in] bar BAR index the access targeted.
  /// @param[in] offset Byte offset within the BAR.
  /// @param[in] width Access width in bytes.
  /// @param[in] write True for a write access, false for a read.
  void record_rejected(int bar, uint64_t offset, std::size_t width, bool write);

  /// @brief Render the refused accesses, most frequent first.
  /// @returns A multi-line report, or an empty string if none were refused.
  [[nodiscard]] std::string rejected_report() const;

  /// @brief Render the unmodeled registers, most frequently accessed first.
  /// @returns A multi-line report, or an empty string if every access was modeled.
  [[nodiscard]] std::string unmodeled_report() const;

  /// @brief Return how many times the spin detector has fired.
  /// @details One episode of spinning warns once, however long it lasts.
  [[nodiscard]] uint64_t spin_warnings() const;

private:
  struct Site {
    uint64_t reads = 0;
    uint64_t writes = 0;
    int bar = 0;
    uint64_t offset = 0;
    std::size_t width = 0;
  };

  static uint64_t make_key(int bar, uint64_t offset);

  [[nodiscard]] std::string render(const std::unordered_map<uint64_t, Site> &from,
                                   std::string_view heading) const;

  const RegisterSymbols &symbols_;
  const Config config_;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, Site> unmodeled_;
  std::unordered_map<uint64_t, Site> rejected_;
  uint64_t repeated_read_key_ = 0;
  uint32_t repeated_read_count_ = 0;
  bool repeated_read_warned_ = false;
  uint64_t spin_warnings_ = 0;
};

} // namespace rocjitsu
