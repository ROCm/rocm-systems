// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_rsrc3.h
/// @brief COMPUTE_PGM_RSRC3 layout families, shared by DBT and DBI.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {

/// @brief COMPUTE_PGM_RSRC3 layout families, per the compute_pgm_rsrc3 tables
/// in llvm/docs/AMDGPUUsage.rst.
///
/// @details RSRC3 is not one register. The families disagree on bits the others
/// reserve, so a source word may only be carried to a target in the same family:
///
/// - Reserved: CDNA1 requires the entire word to be zero.
/// - Gfx9Accum: ACCUM_OFFSET in bits 5:0.
/// - Gfx10: SHARED_VGPR_COUNT in bits 3:0; bits 11:4 and bit 31 reserved. There
///   is no INST_PREF_SIZE here at all -- it is a GFX11+ field.
/// - Gfx11: SHARED_VGPR_COUNT in bits 3:0, a 6-bit INST_PREF_SIZE at 9:4,
///   TRAP_ON_START and TRAP_ON_END at bits 10 and 11, IMAGE_OP at bit 31.
/// - Gfx120: bits 3:0 reserved, INST_PREF_SIZE widened to 8 bits at 11:4
///   (absorbing both trap bits), GLG_EN at 13; bits 21:14 reserved.
/// - Gfx125: as Gfx120, but bits 21:14 carry NAMED_BAR_CNT,
///   ENABLE_DYNAMIC_VGPR, TCP_SPLIT and ENABLE_DIDT_THROTTLE.
enum class Rsrc3Layout : uint8_t {
  Incompatible,
  Reserved,
  Gfx9Accum,
  Gfx10,
  Gfx11,
  Gfx120,
  Gfx125
};

/// @brief RSRC3 layout family for an architecture.
[[nodiscard]] Rsrc3Layout rsrc3_layout(rj_code_arch_t arch);

/// @brief True for the layouts that encode SHARED_VGPR_COUNT in bits 3:0.
[[nodiscard]] bool rsrc3_layout_has_shared_vgpr_count(Rsrc3Layout layout);

/// @brief Bytes in one INST_PREF_SIZE unit: an instruction-cache line on GFX11
///        and later.
inline constexpr uint64_t kInstPrefLineBytes = 128;

/// @brief INST_PREF_SIZE in @p rsrc3 for @p arch: the instruction-cache lines a
///        wave prefetches from its entry at launch.
///
/// @details The field is 6 bits at 9:4 on GFX11 and 8 bits at 11:4 on GFX12 and
/// later.
///
/// @returns nullopt for a layout without the field.
[[nodiscard]] std::optional<uint32_t> inst_pref_size(uint32_t rsrc3, rj_code_arch_t arch);

/// @brief Write @p lines into @p rsrc3's INST_PREF_SIZE for @p arch, which must
///        have the field, leaving every other bit unchanged. @p lines must fit
///        the field.
void set_inst_pref_size(uint32_t &rsrc3, rj_code_arch_t arch, uint32_t lines);

} // namespace rocjitsu
