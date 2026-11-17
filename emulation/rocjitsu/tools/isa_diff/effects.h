// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file effects.h
/// @brief Per-instruction effect summary over LocSet and the P0 executable subset.
///
/// @details The summary is derived from the decoded operands only; no semantics are
/// re-implemented. It exists because the in-tree InstDefUse does not surface SCC,
/// VCC halves, or M0 (see docs/implementation_map.md). Operands are read for one target:
/// its scalar selectors (M0 and NULL move between generations) and its wave size (a wave
/// mask is one SGPR in wave32). Every operand is
/// classified as a tracked location, state outside the tracked universe (VGPRs,
/// TTMP, memory, ...), a constant, or unknown. Unknown operands make the whole
/// instruction's effect unknown, which callers must treat conservatively.
#pragma once

#include "location.h"
#include "target.h"

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
  bool mem_read = false;  ///< Region-executable load or atomic (region_memory_access).
  bool mem_write = false; ///< Region-executable store or atomic.
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
  /// A direct branch a region may contain (s_branch, s_cbranch_scc0/1, vccz/vccnz,
  /// execz/execnz): the runner follows it; leaving the region is an exit.
  bool branch = false;
  /// A workgroup barrier a region may contain (is_region_barrier).
  bool barrier = false;
};

/// @brief Classify one decoded instruction of @p target.
[[nodiscard]] Effects classify(const Instruction &inst, const Target &target);

/// One 32-bit piece of an inline-asm register operand: a tracked scalar location or a VGPR.
struct AsmSlot {
  bool vgpr = false;
  unsigned index = 0; ///< Loc for scalars, register number for VGPRs.
  bool operator==(const AsmSlot &) const = default;
};

/// A comment-only inline-asm block (LLVM `;;#ASMSTART` ... `;;#ASMEND` around lines such as
/// "; def v[0:1]" or "; use s4"). It assembles to nothing; LLVM tests use it as an opaque
/// producer (`def`) or consumer (`use`) of register values.
struct InlineAsmEffect {
  std::vector<AsmSlot> uses; ///< In operand order, one per 32-bit piece.
  std::vector<AsmSlot> defs;
  bool opaque = false;    ///< Text not understood: its effects are unknown.
  bool untracked = false; ///< Names state outside the tracked universe (AGPRs, ...).
  std::string reason;     ///< Why it is opaque or untracked.
};

/// @brief Parse a comment-only inline-asm block; lines are separated by '\n'. Lines are
/// "; def R, R, ..." or "; use R, ..."; any other line that names no register is ignored.
[[nodiscard]] InlineAsmEffect parse_inline_asm(std::string_view text);

/// @brief Effect summary of a comment-only inline-asm block. A `use` of a VGPR also reads
/// EXEC (the lanes it observes). Opaque text has unknown effects; AGPR operands are untracked.
[[nodiscard]] Effects classify_inline_asm(std::string_view text);

/// Floating-point provenance of a 32-bit value (bit flags, one byte per VGPR lane): which
/// parts of it a floating-point instruction produced, and in which format. LLVM leaves
/// the payload, sign and quietness of a NaN such an instruction produces unspecified, so
/// two NaNs with the same provenance on both sides compare equal (unless --strict-nan).
enum FpTag : uint8_t {
  kFpF16Lo = 1u << 0,
  kFpF16Hi = 1u << 1,
  kFpBF16Lo = 1u << 2,
  kFpBF16Hi = 1u << 3,
  kFpF32 = 1u << 4,
  kFpF64Lo = 1u << 5, ///< Low dword of an f64 result (the high dword is the next register).
  kFpF64Hi = 1u << 6,
};

[[nodiscard]] constexpr bool is_nan_f32(uint32_t v) {
  return (v & 0x7f800000u) == 0x7f800000u && (v & 0x7fffffu) != 0;
}
[[nodiscard]] constexpr bool is_nan_f16(uint32_t h) {
  return (h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0;
}
[[nodiscard]] constexpr bool is_nan_bf16(uint32_t h) {
  return (h & 0x7f80u) == 0x7f80u && (h & 0x7fu) != 0;
}
[[nodiscard]] constexpr bool is_nan_f64(uint64_t v) {
  return (v & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (v & 0xfffffffffffffull) != 0;
}

/// @brief Bits of a ^ b that remain after NaN equivalence: an f32, or an f16/bf16 half,
/// that both tags mark as a floating-point result of that format and that is a NaN on both
/// sides does not differ. f64 needs the register pair (see the comparison).
[[nodiscard]] uint32_t nan_residual(uint32_t a, uint32_t b, uint8_t tag_a, uint8_t tag_b);

/// How a VALU instruction's result relates to floating point.
enum class FpResult : uint8_t {
  None,   ///< Not a floating-point result (integer, bitwise, compare): tags cleared.
  F32,    ///< f32 in every destination register.
  F16Lo,  ///< f16 in the low half (high half untouched or zeroed).
  F16Hi,  ///< f16 in the high half (op_sel / dst_sel:WORD_1).
  F16x2,  ///< packed f16.
  BF16Lo, ///< bf16 in the low half.
  BF16Hi,
  BF16x2,
  F64,           ///< f64 in a register pair.
  Move,          ///< v_mov_b32 / v_mov_b64: tags follow the source register.
  CndMask,       ///< v_cndmask_b32: tags follow the selected source, lane by lane.
  PkMov,         ///< v_pk_mov_b32: each destination dword follows its op_sel-selected source dword.
  Select,        ///< s_cselect: SCC picks the first or the second source's tags.
  CMov,          ///< s_cmov: SCC picks the source's tags, else the destination keeps its own.
  ReadFirstLane, ///< v_readfirstlane_b32: the first active lane's tag.
  ReadLane,      ///< v_readlane_b32: the selected lane's tag.
};

/// @brief Classify a VALU instruction's result by mnemonic and disassembly (op_sel/dst_sel).
[[nodiscard]] FpResult fp_result_kind(std::string_view mnemonic, std::string_view text);

/// @brief How an instruction's scalar (SGPR/VCC/M0) results relate to floating point:
/// SALU float (GFX11.5+) results, moves, selects and VGPR lane reads; None otherwise.
[[nodiscard]] FpResult fp_scalar_result_kind(std::string_view mnemonic);

/// @brief Format of a floating-point instruction's sources (F32/F16x2/F64/...), None if it
/// has none (integer and bitwise operations, conversions from integers).
[[nodiscard]] FpResult fp_source_kind(std::string_view mnemonic);

/// Machine analogues of LLVM IR poison/UB conditions (--poison-filter). The tool runs no
/// IR; a filter is enabled when the function's IR contains the corresponding construct, and
/// cases whose region instructions meet the condition are excluded as poison inputs.
enum PoisonKind : uint32_t {
  kPoisonShift = 1u << 0,    ///< shl/lshr/ashr: amount >= bit width
  kPoisonFpToInt = 1u << 1,  ///< fptosi/fptoui: NaN, Inf, or out of the integer range
  kPoisonCtz = 1u << 2,      ///< ctlz/cttz with is_zero_poison: zero input
  kPoisonNsw = 1u << 3,      ///< add/sub/mul nsw: signed 32-bit overflow
  kPoisonNuw = 1u << 4,      ///< add/sub/mul nuw: unsigned 32-bit overflow
  kPoisonDisjoint = 1u << 5, ///< or disjoint: common set bits
  kPoisonExact = 1u << 6,    ///< lshr/ashr exact: non-zero bits shifted out
  kPoisonNnan = 1u << 7,     ///< nnan / nofpclass(nan): a NaN input or result
  kPoisonNinf = 1u << 8,     ///< ninf / nofpclass(inf): an infinite input or result
};

/// @brief Parse "shift,fptoint,..." (or "all"); nullopt on an unknown name.
[[nodiscard]] std::optional<uint32_t> parse_poison_kinds(std::string_view text);
/// @brief Comma-separated names of a kind mask.
[[nodiscard]] std::string format_poison_kinds(uint32_t kinds);

/// @brief The poison condition a machine instruction can witness (0 for none).
[[nodiscard]] uint32_t poison_check_kind(std::string_view mnemonic);

/// @brief The block with each register operand reduced to its class and width
/// ("; def v[0:1]" -> "def v2"): equal on both sides of a correctly matched block.
[[nodiscard]] std::string inline_asm_shape(std::string_view text);

/// One explicit source operand: the tracked locations it reads, or its constant value.
struct SourceOperand {
  LocSet locs;                      ///< Empty for constants and untracked state.
  VSet vlocs;                       ///< VGPR source registers.
  std::optional<uint32_t> constant; ///< Inline constant or literal (low 32 bits).
  bool fieldless = false;           ///< Implicit operand such as SCC carry-in.
};

/// @brief Source operands in encoding order (src0, src1, ...).
[[nodiscard]] std::vector<SourceOperand> source_operands(const Instruction &inst,
                                                         const Target &target);

/// @brief True if the mnemonic is in the P0 allowlist (operands are checked separately).
[[nodiscard]] bool is_p0_mnemonic(std::string_view mnemonic);
/// A VALU instruction the region runner executes: not MFMA, memory, interpolation,
/// indirect or exception-state access.
[[nodiscard]] bool is_region_valu(std::string_view mnemonic, uint64_t flags);
/// What a region-executable memory instruction does with memory.
struct MemoryAccess {
  bool reads = false;
  bool writes = false;
  bool atomic = false; ///< A read-modify-write.
};

/// A memory instruction the region runner executes: loads, stores and atomics of global,
/// flat, scratch and buffer memory and of LDS (not LDS-DMA, transpose, block, addtid,
/// typed-format, GWS or cross-lane DS forms), scalar loads, and the counter waits of every
/// generation (s_waitcnt*, s_wait_*cnt). Memory is an input of the region (equal on both
/// sides) and stored bytes are part of its output.
[[nodiscard]] bool is_region_memory(std::string_view mnemonic);
/// How a memory instruction accesses memory; nullopt for waits and anything outside.
[[nodiscard]] std::optional<MemoryAccess> region_memory_access(std::string_view mnemonic);
/// A private access in stack-frame form, from its disassembly: no VGPR address, so through
/// the stack or frame pointer (buffer `v1, off, s[0:3], s32 offset:12`; scratch
/// `off, v1, s32 offset:12`).
[[nodiscard]] bool is_frame_access(std::string_view mnemonic, std::string_view text);
/// An instruction that may read private memory (stack frames): buffer, scratch and flat
/// loads and atomics of any form.
[[nodiscard]] bool may_read_private(std::string_view mnemonic);
/// A direct branch the region runner follows (taken edges inside the region continue
/// there; others leave it).
[[nodiscard]] bool is_region_branch(std::string_view mnemonic);
/// An instruction without functional effect in a region: scheduling hints and dependency
/// waits (s_nop, s_delay_alu, s_clause, s_wait_alu, s_waitcnt_depctr, prefetch hints).
[[nodiscard]] bool is_region_noop(std::string_view mnemonic);
/// A workgroup barrier a region may contain (s_barrier, GFX9-GFX11). The runner executes
/// one wave, so it waits for no other wave; it starts a new barrier epoch, and the shared
/// memory each side accesses per epoch is compared (RegionRunner::barrier). Not a hint: a
/// hunk that adds, removes or moves one is a semantic change.
[[nodiscard]] bool is_region_barrier(std::string_view mnemonic);
/// A lane permute through the LDS crossbar a region may contain (ds_bpermute_b32,
/// ds_permute_b32): no LDS memory access, lane-crossing, counted on the LDS counter. The
/// runner issues it like an LDS instruction (step()), so the counters stay the hardware's.
[[nodiscard]] bool is_region_lane_permute(std::string_view mnemonic);
/// An LDS transpose load (ds_read_b64_tr_b16, ds_read_b64_tr_b8, ds_read_b64_tr_b4,
/// ds_read_b96_tr_b6): a load whose elements the LDS unit redistributes across the lanes.
/// It needs a full EXEC, and the B64 forms eight-byte addresses.
[[nodiscard]] bool is_lds_transpose(std::string_view mnemonic);

} // namespace rocjitsu::isa_diff
