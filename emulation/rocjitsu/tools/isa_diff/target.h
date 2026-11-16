// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target.h
/// @brief The GPU a comparison runs on: rocjitsu ISA family, wave size and the traits the
/// analysis and the runner depend on.
///
/// @details rj_isa_diff executes whatever RocJITsu can execute: CDNA1-5 and RDNA1-4.
/// A target is identified by its GPU name (gfx950, gfx1100, ...) or by the AMDGPU ELF
/// machine number of a code object. GCN targets before CDNA (gfx600-gfx906, gfx909, gfx90c)
/// and GFX13 have no RocJITsu ISA and are not targets.
#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocjitsu::isa_diff {

struct Target {
  std::string name = "gfx950";
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4;
  unsigned wave_size = 64;
  /// GFX generation: 9 (GFX9 and CDNA1-4), 10 (RDNA1-2), 11 (RDNA3/3.5), 12 (RDNA4, CDNA5).
  unsigned major = 9;
  /// gfx90a and later CDNA: one register file that ACCUM_OFFSET splits into VGPRs and AGPRs.
  bool unified_vgprs = true;

  [[nodiscard]] bool wave32() const { return wave_size == 32; }
  /// Addressable SGPRs: s0-s101 on GFX9, s0-s105 from GFX10 (s102-s105 are untracked).
  [[nodiscard]] unsigned sgprs() const { return major >= 10 ? 106 : 102; }
  /// SGPRs a runner wave allocates (GFX9 keeps two above s101 for the frame).
  [[nodiscard]] unsigned wave_sgprs() const { return major >= 10 ? 106 : 104; }
  /// Scalar source selectors that differ by generation.
  [[nodiscard]] int m0_selector() const { return major >= 11 ? 125 : 124; }
  [[nodiscard]] int null_selector() const { return major >= 11 ? 124 : major == 10 ? 125 : -1; }
  /// MODE has DX10_CLAMP and IEEE (removed in GFX12).
  [[nodiscard]] bool has_ieee_dx10() const { return major < 12; }
  /// "gfx1100 wave32".
  [[nodiscard]] std::string label() const { return name + " wave" + std::to_string(wave_size); }
  bool operator==(const Target &) const = default;
};

/// @brief A RocJITsu-executable target by GPU name, with its default wave size.
[[nodiscard]] std::optional<Target> target_by_name(std::string_view name);
/// @brief The target of an AMDGPU ELF machine number (EF_AMDGPU_MACH_*).
[[nodiscard]] std::optional<Target> target_by_elf_mach(uint32_t mach);
/// @brief Apply a wave size; returns nullopt when the target cannot run at it (GFX9/CDNA
/// are wave64 only, gfx1250/gfx1251 wave32 only).
[[nodiscard]] std::optional<Target> with_wave_size(Target t, unsigned wave_size);
/// @brief Parse "gfx1100", "gfx1100:w64" or "gfx1100:wave64".
[[nodiscard]] std::optional<Target> parse_target(std::string_view text);
/// @brief Comma-separated names of every supported target.
[[nodiscard]] std::string supported_targets();

} // namespace rocjitsu::isa_diff
