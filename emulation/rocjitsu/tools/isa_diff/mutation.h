// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mutation.h
/// @brief Synthetic error injection: a validator self-test, never an LLVM finding.
///
/// @details A mutation XORs a mask into one recorded instruction word on one side of a
/// region; typical uses are a wrong opcode, a wrong immediate or a wrong operand. The
/// mutated word must still decode to the same size and to a P0 instruction. That side's
/// declared writes are re-derived from the decode. The entry and observable sets are
/// kept from the real analysis, so the kernel context is unchanged. Every resulting
/// region, result and counterexample is labelled synthetic.
#pragma once

#include "case_io.h"
#include "region.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::isa_diff {

struct MutationSpec {
  bool new_side = true;
  size_t inst = 0;
  size_t word = 0;
  uint32_t xor_mask = 0;
  std::string label = "mutation";
};

[[nodiscard]] std::optional<RegionPair>
make_synthetic_mutation(const RegionPair &base, const MutationSpec &spec, std::string &error);

/// One single-bit mutant of a region's new side (a would-be miscompile).
struct Mutant {
  MutationSpec spec;
  std::string original; ///< Disassembly of the instruction before the flip.
  std::string mutated;  ///< And after.
  /// What the flip changed in the disassembly: opcode, register, constant, modifier or
  /// other (classify_mutation).
  std::string operator_class;
  std::string unit; ///< salu, valu or memory: the kind of instruction mutated.
};

/// @brief Every distinct single-bit mutant of @p base's new side.
/// @details Each bit of each word of each instruction that is not a hint, a branch or
/// an inline-asm block is flipped; a flip is kept when the word still decodes to a
/// region-executable instruction of the same size whose text differs, and whose VGPRs
/// lie in the region's tracked range. Mutants with the same text keep the first.
[[nodiscard]] std::vector<Mutant> enumerate_mutants(const RegionPair &base);

/// @brief Up to @p max of @p all: round-robin over operator classes, each class in an
/// order fixed by @p seed, so every class is represented before any repeats.
[[nodiscard]] std::vector<Mutant> sample_mutants(std::vector<Mutant> all, size_t max,
                                                 uint64_t seed);

/// @brief Locations a region instruction reads or writes while a load into them is
/// still outstanding (no wait has retired it), in program order with nothing outstanding
/// at the start. The planner's pending-memory analysis makes such a region's context
/// unresolved (CONFLICT), so it is never a pass: a mutant that adds a conflict, such as
/// a weakened s_waitcnt, is caught statically (the isolated run completes loads at once).
[[nodiscard]] std::pair<LocSet, VSet> region_pending_conflicts(const RegionSide &side,
                                                               const Target &target);

/// @brief How mutant @p mutant differs from the new side of @p base it was made from, run
/// on the same inputs as the test (the boundary sweep, then @p cases random samples):
/// "no_effect" (every final state identical: an ignored modifier, a masked shift amount),
/// "dead_effect" (differences only in locations not live at the exit: equivalent under
/// the contract), "undefined_effect" (a live-out location that is undefined at the entry
/// and passed through by the original now holds another value: any value refines an
/// undefined one) or "observable_effect" (a live-out location, memory, the exit or the
/// frame differs: the comparison with old tolerated it, e.g. a NaN payload or bits a
/// narrow return value does not have, or the test is blind to it). "unknown" when the
/// sides cannot run. @p first_case receives the first case with an observable effect.
/// Narrow return values (exit masks) and VGPRs compared on active lanes only are
/// compared as the test compares them. @p detail names the first observable difference.
[[nodiscard]] std::string survivor_effect(const RegionPair &base, const RegionPair &mutant,
                                          uint64_t seed, uint64_t cases,
                                          uint64_t *first_case = nullptr,
                                          std::string *detail = nullptr);

struct MutantRunOptions {
  /// Budget and seed of each mutant's test (the driver's budget measures its strength);
  /// reduction is off and a mutant stops at its first mismatch.
  TestOptions test;
  /// A survivor is re-run with this many random cases (0: not re-run). Killed then, it
  /// shows a budget weakness; never killed, it is equivalent or a blind spot.
  uint64_t confirm_cases = 20000;
  /// Random cases survivor_effect runs a survivor and its original on (0: not run).
  uint64_t effect_cases = 2000;
  size_t max_per_region = 48;
  std::string region; ///< Only this region (empty: every READY region).
};

/// @brief Mutation analysis of a case: each READY region that passes its own test gets
/// up to max_per_region single-bit mutants of its new side (sample_mutants), each tested
/// like the region. Returns rj_isa_diff.mutants.v1 JSON: per region the baseline status,
/// candidate and tested counts, and per mutant its class, unit, texts and verdict
/// (killed, killed_late, survived, invalid) with the phase and case of the first kill.
[[nodiscard]] Json run_mutants(const CaseData &data, const MutantRunOptions &options);

/// @brief What changed between two disassemblies of one instruction: "opcode" (the
/// mnemonic), "register", "constant" (inline constant, literal or offset), "modifier"
/// (neg, abs, op_sel, clamp, omod and other flags) or "other".
[[nodiscard]] std::string classify_mutation(const std::string &original,
                                            const std::string &mutated);

} // namespace rocjitsu::isa_diff
