// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_

#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// Shared scalar-selector landmarks used by every supported AMDGPU ISA.
/// The generator validates these values against each ISA's OPR_SSRC table.
inline constexpr uint32_t kScalarSgprSelectorLast = 105;
inline constexpr uint32_t kFlatScratchSelectorFirst = 102;
inline constexpr uint32_t kFlatScratchSelectorLast = 103;
inline constexpr uint32_t kVccSelectorFirst = 106;
inline constexpr uint32_t kVccSelectorLast = 107;
inline constexpr uint32_t kTtmpSelectorFirst = 108;
inline constexpr uint32_t kTtmpSelectorLast = 123;
inline constexpr uint32_t kTtmpRegisterCount = kTtmpSelectorLast - kTtmpSelectorFirst + 1;
inline constexpr uint32_t kLegacyM0Selector = 124;
inline constexpr uint32_t kModernNullSelector = 124;
inline constexpr uint32_t kModernM0Selector = 125;
inline constexpr uint32_t kGfx10NullSelector = 125;
inline constexpr uint32_t kExecSelectorFirst = 126;
inline constexpr uint32_t kExecSelectorLast = 127;
inline constexpr uint32_t kFlatScratchBaseSelectorFirst = 230;
inline constexpr uint32_t kFlatScratchBaseSelectorLast = 231;

/// Backing selected by a decoded scalar-register operand.
enum class ScalarRegisterStorage : uint8_t {
  SGPR,
  FLAT_SCRATCH,
  VCC,
  TTMP,
  M0,
  EXEC,
  DISCARD,
};

/// A complete, homogeneous scalar-register range.
///
/// This is resolved once while an instruction is issued and can then be
/// carried through deferred memory completion. `index` is relative to the
/// selected storage class; `width` is measured in dwords. DISCARD represents
/// the architectural NULL destination and ignores index.
struct ScalarRegisterRange {
  ScalarRegisterStorage storage = ScalarRegisterStorage::DISCARD;
  uint16_t index = 0;
  uint8_t width = 0;

  [[nodiscard]] std::optional<RegisterRef> register_ref() const {
    switch (storage) {
    case ScalarRegisterStorage::SGPR:
      return RegisterRef{RegClass::SGPR, index, width};
    case ScalarRegisterStorage::FLAT_SCRATCH:
      return RegisterRef{RegClass::FLAT_SCRATCH, index, width};
    case ScalarRegisterStorage::VCC:
      return RegisterRef{RegClass::VCC, index, width};
    case ScalarRegisterStorage::TTMP:
      return RegisterRef{RegClass::TTMP, index, width};
    case ScalarRegisterStorage::M0:
      return RegisterRef{RegClass::M0, index, width};
    case ScalarRegisterStorage::EXEC:
      return RegisterRef{RegClass::EXEC, index, width};
    case ScalarRegisterStorage::DISCARD:
      return std::nullopt;
    }
    return std::nullopt;
  }
};

/// @brief Return whether a scalar source selector names the low word of a
/// 64-bit register pair.
///
/// @details This is the register-backed subset of resolve_src_scalar64(). It
/// includes ordinary SGPR pairs, architecture-specific aliases in that range,
/// VCC, TTMP/TBA/TMA pairs, EXEC, and the GFX11+ FLAT_SCRATCH_BASE selector.
/// Single-word sources such as M0 and inline constants are deliberately
/// excluded. The amdisa generator validates these shared values against every
/// ISA's OPR_SSRC table.
[[nodiscard]] inline constexpr bool is_src_scalar_register_pair(int ev) {
  return (ev >= 0 && ev <= static_cast<int>(kVccSelectorFirst)) ||
         (ev >= static_cast<int>(kTtmpSelectorFirst) && ev < static_cast<int>(kTtmpSelectorLast)) ||
         ev == static_cast<int>(kExecSelectorFirst) ||
         ev == static_cast<int>(kFlatScratchBaseSelectorFirst);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_
