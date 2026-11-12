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
resolve_scalar_storage_range(const Wavefront &wf, uint32_t selector, uint32_t count) {
  if (count == 0 || count > std::numeric_limits<uint8_t>::max() ||
      selector > UINT32_MAX - (count - 1))
    return std::nullopt;

  auto first = scalar_register_for_selector(wf, selector);
  if (!first)
    return std::nullopt;
  first->width = static_cast<uint8_t>(count);
  if (first->storage == ScalarRegisterStorage::SGPR &&
      wf.exceeds_rounded_sgpr_allocation(first->index, count))
    return std::nullopt;
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

/// Capture TTMP permission at issue for accesses that can complete after a trap transition.
[[nodiscard]] inline std::optional<ScalarRegisterRange>
resolve_scalar_register_range(const Wavefront &wf, uint32_t selector, uint32_t count) {
  auto range = resolve_scalar_storage_range(wf, selector, count);
  if (range && range->storage == ScalarRegisterStorage::TTMP &&
      !RegisterAccess::shader_can_access_ttmp(wf))
    return ScalarRegisterRange{ScalarRegisterStorage::DISCARD, 0, range->width};
  return range;
}

/// Resolve MOVREL within its original SGPR/VCC or TTMP region and allocation.
/// Missing sources use SGPR0; missing destinations suppress the instruction.
/// B64 alignment follows RDNA3/3.5 and measured RDNA4 behavior. CDNA requires
/// even indices; its odd-index handling is extrapolated from RDNA. The shared
/// SGPR/VCC region and CDNA FLAT_SCRATCH exclusion are also extrapolations;
/// CDNA manuals do not explicitly specify the RDNA SGPR-to-VCC exception.
[[nodiscard]] inline std::optional<uint32_t>
relative_scalar_selector(const Wavefront &wf, uint32_t selector, uint32_t m0, uint32_t width,
                         bool destination, bool split) {
  uint32_t offset = split ? (m0 >> (destination ? 16 : 0)) & 0x3ffu : m0;
  if (width == 2)
    offset &= ~1u;
  const uint64_t indexed = uint64_t{selector} + offset;
  const uint32_t last = selector <= kVccSelectorLast ? kVccSelectorLast : kTtmpSelectorLast;
  if (selector > kTtmpSelectorLast || indexed + width > uint64_t{last} + 1)
    return std::nullopt;
  const auto base = scalar_register_for_selector(wf, selector);
  const auto range = resolve_scalar_storage_range(wf, static_cast<uint32_t>(indexed), width);
  if (!base || !range)
    return std::nullopt;
  const auto general = [](ScalarRegisterStorage storage) {
    return storage == ScalarRegisterStorage::SGPR || storage == ScalarRegisterStorage::VCC;
  };
  if (base->storage != range->storage && !(general(base->storage) && general(range->storage)))
    return std::nullopt;
  return static_cast<uint32_t>(indexed);
}

/// Capture write permission without changing an atomic's separately resolved source.
[[nodiscard]] inline ScalarRegisterRange scalar_write_range(const Wavefront &wf,
                                                            ScalarRegisterRange range) {
  if (range.storage == ScalarRegisterStorage::TTMP && !wf.shader_can_write_ttmp())
    range.storage = ScalarRegisterStorage::DISCARD;
  return range;
}

/// Resolve the write destination at issue, before a deferred load can change privilege levels.
[[nodiscard]] inline std::optional<ScalarRegisterRange>
resolve_scalar_destination_range(const Wavefront &wf, uint32_t selector, uint32_t count) {
  auto range = resolve_scalar_register_range(wf, selector, count);
  if (!range)
    return std::nullopt;
  return scalar_write_range(wf, *range);
}

/// @brief Check that a complete selector range has backing owned by this wave.
[[nodiscard]] inline bool scalar_selector_range_is_backed(const Wavefront &wf, uint32_t selector,
                                                          uint32_t count) {
  return resolve_scalar_register_range(wf, selector, count).has_value();
}

/// Named VCC and FLAT_SCRATCH accessors deliberately bypass the SGPR callback.
inline void observe_scalar_selector_alias(const Wavefront &wf, int selector, uint8_t width,
                                          bool write) {
  if (!wf.cu().observes_register_access())
    return;
  if (selector == 251 && !write) { // VCCZ reads the whole active-lane mask.
    RegisterAccess(wf).observe_scalar_alias(
        {RegClass::VCC, 0, static_cast<uint8_t>(wf.wf_size() / 32)}, false);
    return;
  }
  if (selector < static_cast<int>(kFlatScratchSelectorFirst) ||
      selector > static_cast<int>(kVccSelectorLast))
    return;
  const auto range = scalar_register_for_selector(wf, selector);
  if (range &&
      (range->storage == ScalarRegisterStorage::VCC ||
       range->storage == ScalarRegisterStorage::FLAT_SCRATCH) &&
      range->index + width <= 2) {
    auto reg = *range->register_ref();
    reg.width = width;
    RegisterAccess(wf).observe_scalar_alias(reg, write);
  }
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
    RegisterAccess(wf).observe_scalar_alias(
        {RegClass::FLAT_SCRATCH, static_cast<uint16_t>(index), 1}, false);
    return index == 0 ? static_cast<uint32_t>(wf.scratch_base())
                      : static_cast<uint32_t>(wf.scratch_base() >> 32);
  case ScalarRegisterStorage::VCC:
    RegisterAccess(wf).observe_scalar_alias({RegClass::VCC, static_cast<uint16_t>(index), 1},
                                            false);
    return index == 0 ? static_cast<uint32_t>(wf.vcc()) : static_cast<uint32_t>(wf.vcc() >> 32);
  case ScalarRegisterStorage::TTMP:
    return RegisterAccess(wf).read_ttmp(index);
  case ScalarRegisterStorage::M0:
    return wf.m0();
  case ScalarRegisterStorage::EXEC:
    return index == 0 ? static_cast<uint32_t>(wf.exec_raw())
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
  if (range.storage == ScalarRegisterStorage::FLAT_SCRATCH ||
      range.storage == ScalarRegisterStorage::VCC) {
    RegisterAccess(wf).observe_scalar_alias({range.register_ref()->cls, range.index, 2}, false);
    return range.storage == ScalarRegisterStorage::FLAT_SCRATCH ? wf.scratch_base() : wf.vcc();
  }
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
    if (RegisterAccess(wf).write_sgpr(wf.sgpr_alloc().base + range->index, value).failed())
      return;
    return;
  }
  if (range->storage == ScalarRegisterStorage::TTMP) {
    RegisterAccess(wf).write_ttmp(range->index, value);
    return;
  }
  observe_scalar_selector_alias(wf, selector, 1, true);
  if (range->storage == ScalarRegisterStorage::EXEC) {
    if (range->index == 0)
      wf.write_exec_lo(value);
    else
      wf.write_exec_hi(value);
    return;
  }
  RegisterAccess(wf).write_scalar_unobserved(*range, 0, value);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_OPERAND_READ_H_
