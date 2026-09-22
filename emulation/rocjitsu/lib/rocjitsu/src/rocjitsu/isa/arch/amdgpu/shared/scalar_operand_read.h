// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scalar_operand_read.h
/// @brief Read a scalar register by its encoded operand selector.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_OPERAND_READ_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_OPERAND_READ_H_

#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu {
namespace amdgpu {

[[nodiscard]] inline bool scalar_selector_is_null(rj_code_arch_t arch, uint32_t selector) {
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

[[nodiscard]] inline uint32_t scalar_m0_selector(rj_code_arch_t arch) {
  return scalar_selector_is_null(arch, kModernNullSelector) ? kModernM0Selector : kLegacyM0Selector;
}

[[nodiscard]] inline std::optional<ScalarRegisterRange>
scalar_register_for_selector(const Wavefront &wf, uint32_t selector) {
  const rj_code_arch_t arch = wf.cu().arch();
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
resolve_scalar_register_range(const Wavefront &wf, uint32_t selector, uint32_t count) {
  if (count == 0 || count > std::numeric_limits<uint8_t>::max() ||
      selector > UINT32_MAX - (count - 1))
    return std::nullopt;

  auto first = scalar_register_for_selector(wf, selector);
  if (!first)
    return std::nullopt;
  first->width = static_cast<uint8_t>(count);
  if (first->storage == ScalarRegisterStorage::DISCARD)
    return first;

  for (uint32_t offset = 1; offset < count; ++offset) {
    auto current = scalar_register_for_selector(wf, selector + offset);
    if (!current || current->storage != first->storage || current->index != first->index + offset)
      return std::nullopt;
  }

  if (first->storage == ScalarRegisterStorage::SGPR &&
      !RegisterAccess(wf).owns_sgpr_range(wf.sgpr_alloc().base + first->index, count))
    return std::nullopt;
  return first;
}

/// @brief Check that a complete selector range has backing owned by this wave.
[[nodiscard]] inline bool scalar_selector_range_is_backed(const Wavefront &wf, uint32_t selector,
                                                          uint32_t count) {
  return resolve_scalar_register_range(wf, selector, count).has_value();
}

/// @brief Read one dword from a previously validated scalar-register range.
[[nodiscard]] inline uint32_t
read_scalar_register(const Wavefront &wf, const ScalarRegisterRange &range, uint32_t offset = 0) {
  if (offset >= range.width)
    return 0;
  const uint32_t index = range.index + offset;
  switch (range.storage) {
  case ScalarRegisterStorage::SGPR:
    return RegisterAccess(wf).read_sgpr(wf.sgpr_alloc().base + index);
  case ScalarRegisterStorage::FLAT_SCRATCH:
    return index == 0 ? static_cast<uint32_t>(wf.scratch_base())
                      : static_cast<uint32_t>(wf.scratch_base() >> 32);
  case ScalarRegisterStorage::VCC:
    return index == 0 ? static_cast<uint32_t>(wf.vcc()) : static_cast<uint32_t>(wf.vcc() >> 32);
  case ScalarRegisterStorage::TTMP:
    return RegisterAccess(wf).read_ttmp(index);
  case ScalarRegisterStorage::M0:
    return wf.m0();
  case ScalarRegisterStorage::EXEC:
    return index == 0 ? static_cast<uint32_t>(wf.exec())
                      : static_cast<uint32_t>(wf.exec_raw() >> 32);
  case ScalarRegisterStorage::DISCARD:
    return 0;
  }
  return 0;
}

/// @brief Read the first two dwords of a validated scalar-register range.
[[nodiscard]] inline uint64_t read_scalar_register64(const Wavefront &wf,
                                                     const ScalarRegisterRange &range) {
  if (range.width < 2)
    return 0;
  if (range.storage == ScalarRegisterStorage::SGPR)
    return RegisterAccess(wf).read_sgpr64(wf.sgpr_alloc().base + range.index);
  if (range.storage == ScalarRegisterStorage::TTMP)
    return RegisterAccess(wf).read_ttmp64(range.index);
  return static_cast<uint64_t>(read_scalar_register(wf, range)) |
         (static_cast<uint64_t>(read_scalar_register(wf, range, 1)) << 32);
}

/// @brief Read the scalar register named by an encoded operand selector.
///
/// @details Address-computation helpers take the raw selector out of an
/// instruction field (SBASE, SOFFSET, SADDR, SRSRC) and would otherwise index
/// it straight into the wave's SGPR allocation. Selectors 108..123 do not live
/// there: they name the trap-temporary file, which the decoder routes to
/// Wavefront::ttmp(). A trap handler that loads through a TTMP-held pointer --
/// which the ROCr handler does -- must read the TTMP, not whatever SGPR happens
/// to sit at that offset in the allocation.
[[nodiscard]] inline uint32_t read_scalar_selector(Wavefront &wf, uint32_t selector) {
  auto range = resolve_scalar_register_range(wf, selector, 1);
  return range ? read_scalar_register(wf, *range) : 0;
}

/// @brief Read a 64-bit scalar pair named by an encoded operand selector.
/// @details The complete pair must resolve to one architectural register file.
[[nodiscard]] inline uint64_t read_scalar_selector64(Wavefront &wf, uint32_t selector) {
  auto range = resolve_scalar_register_range(wf, selector, 2);
  if (!range)
    return 0;
  return read_scalar_register64(wf, *range);
}

[[nodiscard]] inline std::optional<uint32_t> try_read_scalar_selector(Wavefront &wf,
                                                                      uint32_t selector) {
  auto range = resolve_scalar_register_range(wf, selector, 1);
  if (!range)
    return std::nullopt;
  return read_scalar_register(wf, *range);
}

[[nodiscard]] inline std::optional<uint64_t> try_read_scalar_selector64(Wavefront &wf,
                                                                        uint32_t selector) {
  auto range = resolve_scalar_register_range(wf, selector, 2);
  if (!range)
    return std::nullopt;
  return read_scalar_register64(wf, *range);
}

/// @brief Write one scalar register selected by an encoded operand.
/// @details This is an instruction-visible write. Deferred memory completion
/// deliberately uses its separate, unobserved storage path.
inline void write_scalar_selector(Wavefront &wf, uint32_t selector, uint32_t value) {
  auto range = resolve_scalar_register_range(wf, selector, 1);
  if (!range)
    return;
  if (range->storage == ScalarRegisterStorage::SGPR) {
    RegisterAccess(wf).write_sgpr(wf.sgpr_alloc().base + range->index, value);
    return;
  }
  if (range->storage == ScalarRegisterStorage::TTMP) {
    RegisterAccess(wf).write_ttmp(range->index, value);
    return;
  }
  RegisterAccess(wf).write_scalar_unobserved(*range, 0, value);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_OPERAND_READ_H_
