// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_SELECTORS_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <limits>
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

[[nodiscard]] inline constexpr bool scalar_selector_is_null(rj_code_arch_t arch,
                                                            uint32_t selector) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return selector == kGfx10NullSelector;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
    return selector == kModernNullSelector;
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  default:
    return false;
  }
}

[[nodiscard]] inline constexpr uint32_t scalar_m0_selector(rj_code_arch_t arch) {
  return scalar_selector_is_null(arch, kModernNullSelector) ? kModernM0Selector : kLegacyM0Selector;
}

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

[[nodiscard]] inline std::optional<ScalarRegisterRange>
scalar_register_for_selector(rj_code_arch_t arch, uint32_t selector) {
  if (arch_uses_legacy_flat_scratch_sgprs(arch) && selector >= kFlatScratchSelectorFirst &&
      selector <= kFlatScratchSelectorLast)
    return ScalarRegisterRange{ScalarRegisterStorage::FLAT_SCRATCH,
                               static_cast<uint16_t>(selector - kFlatScratchSelectorFirst), 1};
  if (selector <= kScalarSgprSelectorLast)
    return ScalarRegisterRange{ScalarRegisterStorage::SGPR, static_cast<uint16_t>(selector), 1};
  if (selector <= kVccSelectorLast)
    return ScalarRegisterRange{ScalarRegisterStorage::VCC,
                               static_cast<uint16_t>(selector - kVccSelectorFirst), 1};
  if (selector <= kTtmpSelectorLast)
    return ScalarRegisterRange{ScalarRegisterStorage::TTMP,
                               static_cast<uint16_t>(selector - kTtmpSelectorFirst), 1};
  if (scalar_selector_is_null(arch, selector))
    return ScalarRegisterRange{ScalarRegisterStorage::DISCARD, 0, 1};
  if (selector == scalar_m0_selector(arch))
    return ScalarRegisterRange{ScalarRegisterStorage::M0, 0, 1};
  if (selector >= kExecSelectorFirst && selector <= kExecSelectorLast)
    return ScalarRegisterRange{ScalarRegisterStorage::EXEC,
                               static_cast<uint16_t>(selector - kExecSelectorFirst), 1};
  return std::nullopt;
}

/// @brief Resolve and validate one complete scalar-register range.
/// @details Every dword must name consecutive storage in one architectural
/// register file. NULL is a single selector that discards or supplies the
/// complete operand independently of its width.
[[nodiscard]] inline std::optional<ScalarRegisterRange>
resolve_scalar_storage_range(rj_code_arch_t arch, uint32_t selector, uint32_t count) {
  if (count == 0 || count > std::numeric_limits<uint8_t>::max() ||
      selector > UINT32_MAX - (count - 1))
    return std::nullopt;

  auto first = scalar_register_for_selector(arch, selector);
  if (!first)
    return std::nullopt;
  first->width = static_cast<uint8_t>(count);
  if (first->storage == ScalarRegisterStorage::DISCARD)
    return first;

  for (uint32_t offset = 1; offset < count; ++offset) {
    auto current = scalar_register_for_selector(arch, selector + offset);
    if (!current || current->storage != first->storage || current->index != first->index + offset)
      return std::nullopt;
  }

  return first;
}

/// Resolve MOVREL within its original SGPR/VCC or TTMP region and allocation.
/// Missing sources use SGPR0; missing destinations suppress the instruction.
/// B64 alignment follows RDNA3/3.5 and measured RDNA4 behavior. CDNA requires
/// even indices; its odd-index handling is extrapolated from RDNA. The shared
/// SGPR/VCC region and CDNA FLAT_SCRATCH exclusion are also extrapolations;
/// CDNA manuals do not explicitly specify the RDNA SGPR-to-VCC exception.
[[nodiscard]] inline std::optional<uint32_t>
relative_scalar_selector(rj_code_arch_t arch, uint32_t selector, uint32_t m0, uint32_t width,
                         bool destination, bool split) {
  uint32_t offset = split ? (m0 >> (destination ? 16 : 0)) & 0x3ffu : m0;
  if (width == 2)
    offset &= ~1u;
  const uint64_t indexed = uint64_t{selector} + offset;
  const uint32_t last = selector <= kVccSelectorLast ? kVccSelectorLast : kTtmpSelectorLast;
  if (selector > kTtmpSelectorLast || indexed + width > uint64_t{last} + 1)
    return std::nullopt;
  const auto base = scalar_register_for_selector(arch, selector);
  const auto range = resolve_scalar_storage_range(arch, static_cast<uint32_t>(indexed), width);
  if (!base || !range)
    return std::nullopt;
  const auto general = [](ScalarRegisterStorage storage) {
    return storage == ScalarRegisterStorage::SGPR || storage == ScalarRegisterStorage::VCC;
  };
  if (base->storage != range->storage && !(general(base->storage) && general(range->storage)))
    return std::nullopt;
  return static_cast<uint32_t>(indexed);
}

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
