// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file effects.h
/// @brief Per-instruction effect summary over LocSet and the P0 executable subset.
///
/// @details The summary is derived from the decoded operands only; no semantics are
/// re-implemented. It exists because the in-tree InstDefUse does not surface SCC,
/// VCC halves, or M0 on gfx950 (see docs/implementation_map.md). Every operand is
/// classified as a tracked location, state outside the tracked universe (VGPRs,
/// TTMP, memory, ...), a constant, or unknown. Unknown operands make the whole
/// instruction's effect unknown, which callers must treat conservatively.
#pragma once

#include "location.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
class Instruction;
}

namespace rocjitsu::isa_diff {

struct Effects {
  LocSet reads;  ///< Tracked locations read (a conditional write also reads its destination).
  LocSet writes; ///< Tracked locations written.
  /// Low location of each 64-bit source operand (input generation draws pair values).
  LocSet wide_reads;
  VSet vreads;            ///< VGPRs read (whole registers; any lane).
  VSet vwrites;           ///< VGPRs written (lanes under EXEC, or all for lane writes).
  VSet vwide_reads;       ///< First register of each 64-bit VGPR source.
  bool valu = false;      ///< VALU instruction (reads EXEC; region-executable when allowed).
  bool mem_read = false;  ///< Region-executable load (global_load_*, s_load_dword*).
  bool mem_write = false; ///< Region-executable store (global_store_*).
  /// True when the instruction is in the P0 allowlist and every operand is a
  /// tracked location or a constant, so a region may execute it.
  bool p0 = false;
  /// Why the instruction is not P0 (empty when p0 is true).
  std::string reject_reason;
  /// Reads the program counter (s_getpc_b64, s_swappc, ...): its results are PC-derived.
  bool reads_pc = false;
  /// Touches tracked state in a way the operand model cannot name (indirect SGPR
  /// access, implicit-effect hooks, unrecognised selectors). Liveness must treat it
  /// as reading everything.
  bool unknown_effect = false;
  /// Has the SALU instruction flag (which also covers waits, branches and s_endpgm).
  bool salu = false;
  /// Outside P0, liveness assumes the instruction may also read EXEC, VCC and M0 without
  /// naming them as operands. False only for the short list of SALU control
  /// instructions whose condition register (if any) is a classified operand.
  bool implicit_special_reads = true;
  /// The specials (EXEC/VCC/M0) an instruction outside the executable subset reads beyond
  /// its operands: none for SMEM and the short SALU list, EXEC for vector memory (plus M0
  /// for its LDS forms), EXEC and M0 for LDS, all three otherwise.
  LocSet implicit_reads;
  /// Is a direct or indirect control-flow transfer or program terminator.
  bool control_flow = false;
};

/// @brief Classify one decoded instruction.
[[nodiscard]] Effects classify(const Instruction &inst);

/// One explicit source operand: the tracked locations it reads, or its constant value.
struct SourceOperand {
  LocSet locs;                      ///< Empty for constants and untracked state.
  VSet vlocs;                       ///< VGPR source registers.
  std::optional<uint32_t> constant; ///< Inline constant or literal (low 32 bits).
  bool fieldless = false;           ///< Implicit operand such as SCC carry-in.
};

/// @brief Source operands in encoding order (src0, src1, ...).
[[nodiscard]] std::vector<SourceOperand> source_operands(const Instruction &inst);

/// @brief True if the mnemonic is in the P0 allowlist (operands are checked separately).
[[nodiscard]] bool is_p0_mnemonic(std::string_view mnemonic);
/// A VALU instruction the region runner executes: not MFMA, memory, interpolation,
/// indirect or exception-state access.
[[nodiscard]] bool is_region_valu(std::string_view mnemonic, uint64_t flags);
/// A memory instruction the region runner executes: global loads and stores (not LDS-DMA,
/// transpose or atomic forms), scalar loads, and s_waitcnt. Memory is an input of the
/// region (equal on both sides) and stored bytes are part of its output.
[[nodiscard]] bool is_region_memory(std::string_view mnemonic);

} // namespace rocjitsu::isa_diff
