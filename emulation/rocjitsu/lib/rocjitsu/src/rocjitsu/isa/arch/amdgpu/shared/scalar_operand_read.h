// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scalar_operand_read.h
/// @brief Read a scalar register by its encoded operand selector.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_OPERAND_READ_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_OPERAND_READ_H_

#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief First scalar selector that addresses a trap-temporary register.
inline constexpr uint32_t kTtmpSelectorFirst = 108;
/// @brief Last scalar selector that addresses a trap-temporary register.
inline constexpr uint32_t kTtmpSelectorLast = 123;

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
  if (selector >= kTtmpSelectorFirst && selector <= kTtmpSelectorLast)
    return wf.ttmp(selector - kTtmpSelectorFirst);
  return RegisterAccess(wf.cu()).read_sgpr(wf.sgpr_alloc().base + selector);
}

/// @brief Read a 64-bit scalar pair named by an encoded operand selector.
/// @details The two halves are resolved independently so a pair that straddles
/// the TTMP boundary still reads each half from the file that owns it.
[[nodiscard]] inline uint64_t read_scalar_selector64(Wavefront &wf, uint32_t selector) {
  return (static_cast<uint64_t>(read_scalar_selector(wf, selector + 1)) << 32) |
         read_scalar_selector(wf, selector);
}

/// @brief Write the scalar register named by an encoded operand selector.
/// @details Counterpart to read_scalar_selector(): a destination in 108..123
/// must land in the trap-temporary file. Writing it through the SGPR
/// allocation would both lose the value (every read comes from the TTMP file)
/// and scribble past the end of this wave's allocation into a neighbour's.
inline void write_scalar_selector(Wavefront &wf, uint32_t selector, uint32_t value) {
  if (selector >= kTtmpSelectorFirst && selector <= kTtmpSelectorLast) {
    wf.set_ttmp(selector - kTtmpSelectorFirst, value);
    return;
  }
  RegisterAccess(wf.cu()).write_sgpr(wf.sgpr_alloc().base + selector, value);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_OPERAND_READ_H_
