// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_dispatch_prologue_effect.h
/// @brief Typed dispatch-ID entry-prologue ABI effect.

#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace rocjitsu {

/// AMDHSA initializes at most 16 user SGPRs on the targets using this path.
/// Queue-pointer/dispatch-ID insertion is allowed only when all original
/// preloads and following system SGPRs can be mapped without truncation or
/// register overflow.
enum class ConSanMoiDispatchIdPreloadSupport : uint8_t {
  SupportedAlreadyEnabled,
  SupportedInsert,
  InvalidDispatchPosition,
  UserSgprInitializationLimit,
  SgprAllocationLimit,
};

/// Complete target-neutral description of an AMDHSA dispatch-ID preload
/// transformation. Construction belongs to the private preload planner;
/// committed prologue proof and independent validation retain this value.
struct ConSanMoiDispatchIdPreloadPlan {
  ConSanMoiDispatchIdPreloadSupport support =
      ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
  uint16_t dispatch_id_sgpr = 0;
  /// AMDHSA queue-pointer source used to distinguish equal queue-local IDs.
  std::optional<uint16_t> identity_salt_sgpr;
  bool queue_ptr_was_enabled = false;
  bool dispatch_id_was_enabled = false;
  /// Exact source for each original guest SGPR that moved. This supplements
  /// the legacy uniform dispatch-only shift for queue+dispatch insertion.
  std::array<uint16_t, 32> guest_restore_destinations{};
  std::array<uint16_t, 32> guest_restore_sources{};
  uint16_t guest_restore_count = 0;
  uint16_t original_user_sgpr_count = 0;
  uint16_t expanded_user_sgpr_count = 0;
  uint16_t system_sgpr_count = 0;
  uint16_t first_shifted_guest_sgpr = 0;
  uint16_t shifted_guest_sgpr_count = 0;
  /// Full user-SGPR windows can make room for queue and dispatch identity by
  /// shortening the hardware kernarg preload and reloading up to its final
  /// four dwords in software.
  uint16_t kernarg_reload_sgpr = 0;
  uint16_t kernarg_reload_base_sgpr = 0;
  uint16_t kernarg_reload_offset_dwords = 0;
  uint16_t kernarg_reload_count = 0;
  uint16_t shifted_system_sgpr_count = 0;
  uint16_t system_sgpr_shift = 0;
  uint16_t original_kernarg_preload_length = 0;
  uint16_t replacement_kernarg_preload_length = 0;
  uint16_t required_sgpr_count = 0;

  [[nodiscard]] constexpr bool supported() const {
    return support == ConSanMoiDispatchIdPreloadSupport::SupportedAlreadyEnabled ||
           support == ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  }
  [[nodiscard]] constexpr bool descriptor_change_required() const {
    return support == ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  }
  [[nodiscard]] constexpr bool requires_kernarg_reload() const { return kernarg_reload_count != 0; }

  bool operator==(const ConSanMoiDispatchIdPreloadPlan &) const = default;
};

/// Unique persistent representation receiving a dispatch ID at kernel entry.
/// The scalar and vector forms are alternatives, never simultaneous.
class ConSanMoiDispatchIdCapture {
public:
  ConSanMoiDispatchIdCapture() = default;

  [[nodiscard]] static ConSanMoiDispatchIdCapture in_sgprs(uint16_t base) {
    return ConSanMoiDispatchIdCapture{Kind::Sgpr, base};
  }
  [[nodiscard]] static ConSanMoiDispatchIdCapture in_vgprs(uint16_t base) {
    return ConSanMoiDispatchIdCapture{Kind::Vgpr, base};
  }

  [[nodiscard]] std::optional<uint16_t> sgpr() const {
    return kind_ == Kind::Sgpr ? std::optional{base_} : std::nullopt;
  }
  [[nodiscard]] std::optional<uint16_t> vgpr() const {
    return kind_ == Kind::Vgpr ? std::optional{base_} : std::nullopt;
  }
  [[nodiscard]] bool present() const { return kind_ != Kind::None; }

  bool operator==(const ConSanMoiDispatchIdCapture &) const = default;

private:
  enum class Kind : uint8_t { None, Sgpr, Vgpr };

  ConSanMoiDispatchIdCapture(Kind kind, uint16_t base) : kind_(kind), base_(base) {}

  Kind kind_ = Kind::None;
  uint16_t base_ = 0;
};

/// Complete dispatch-ID ABI effect proved by one emitted entry prologue.
struct ConSanMoiDispatchIdPrologueEffect {
  ConSanMoiDispatchIdPreloadPlan preload;
  ConSanMoiDispatchIdCapture capture;

  [[nodiscard]] uint16_t required_sgpr_count() const {
    const std::optional<uint16_t> scalar = capture.sgpr();
    const uint32_t capture_end = scalar ? static_cast<uint32_t>(*scalar) + 2u : 0u;
    return static_cast<uint16_t>(
        capture_end > preload.required_sgpr_count ? capture_end : preload.required_sgpr_count);
  }

  bool operator==(const ConSanMoiDispatchIdPrologueEffect &) const = default;
};

} // namespace rocjitsu
