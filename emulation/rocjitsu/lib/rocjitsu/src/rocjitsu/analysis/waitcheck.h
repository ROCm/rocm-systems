// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcheck.h
/// @brief Static waitcnt dependency checker for decoded AMDGPU code.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

class CodeObject;

/// @brief GFX12 split wait counters tracked by waitcheck.
enum class WaitCounterKind : uint8_t {
  Load = 0,
  Store,
  Ds,
  Km,
  Sample,
  Bvh,
  Exp,
  VmVsrc,
  VaVdst,
  Depctr,
  Count,
};

/// @brief How a later instruction conflicts with an outstanding event.
enum class WaitcheckAccessKind : uint8_t {
  Use,
  Def,
  MemoryOrder,
  ProgramEnd,
};

/// @brief One static waitcnt hazard diagnostic.
struct WaitcheckDiagnostic {
  WaitCounterKind counter = WaitCounterKind::Load;
  WaitcheckAccessKind access = WaitcheckAccessKind::Use;
  RegisterRef reg{RegClass::VGPR, 0, 1};
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;
  uint64_t producer_section_offset = 0;
  uint64_t producer_file_offset = 0;
  std::string producer_instruction;
  uint32_t required_count = 0;
  std::string message;
};

/// @brief Controls for waitcheck analysis.
struct WaitcheckOptions {
  /// @brief Maximum diagnostics to retain. Analysis still reports failure after
  /// the first hazard when this is zero.
  size_t max_diagnostics = std::numeric_limits<size_t>::max();
  /// @brief Stop checking the current input after the first observed diagnostic.
  ///
  /// @details This is intended for large corpus sweeps where only hazard
  /// presence is needed. Report counts become lower bounds when enabled.
  bool stop_after_first_diagnostic = false;
};

/// @brief Result of one waitcheck analysis run.
struct WaitcheckReport {
  bool supported = true;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  size_t instructions_analyzed = 0;
  size_t memory_events_tracked = 0;
  size_t diagnostics_observed = 0;
  bool diagnostics_truncated = false;
  bool stopped_early = false;
  std::string analysis_error;
  std::vector<WaitcheckDiagnostic> diagnostics;

  [[nodiscard]] bool passed() const { return diagnostics_observed == 0; }
};

/// @brief Human-readable split counter name, e.g. "loadcnt".
[[nodiscard]] std::string_view wait_counter_name(WaitCounterKind counter);

/// @brief Return the RocJITsu ISA arch supported by waitcheck for @p target.
///
/// @details This keeps the current prototype intentionally scoped to gfx12
/// targets. Unsupported targets return ROCJITSU_CODE_ARCH_INVALID.
[[nodiscard]] rj_code_arch_t waitcheck_arch_for_target(rj_code_target_id_t target);

/// @brief Analyze one stream of 32-bit instruction words.
///
/// @details The initial implementation is intentionally scoped to gfx12.
/// Other architectures return a report with supported=false.
[[nodiscard]] WaitcheckReport analyze_waitcnts(std::span<const uint32_t> words, rj_code_arch_t arch,
                                               WaitcheckOptions options = {});

/// @brief Analyze all executable sections in a code object.
///
/// @details Offsets in diagnostics are relative to each section and to the ELF
/// file offset when the section exposes one.
[[nodiscard]] WaitcheckReport analyze_waitcnts(const CodeObject &code_object, rj_code_arch_t arch,
                                               WaitcheckOptions options = {});

} // namespace rocjitsu
