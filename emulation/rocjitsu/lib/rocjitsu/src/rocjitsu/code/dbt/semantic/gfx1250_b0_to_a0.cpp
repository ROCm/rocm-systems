// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_b0_to_a0.cpp
/// @brief Handwritten semantic expansions for gfx1250 B0-to-A0 translation.

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/dbt/semantic/gfx1250_flat_scratch_base.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/semantic_scratch.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

[[nodiscard]] bool always_residual(const Instruction &) { return true; }
[[nodiscard]] bool scale16_residual(const Instruction &inst);
[[nodiscard]] bool cvt_f32_fp8_e5m3_residual(const Instruction &inst);
[[nodiscard]] bool cvt_pk_fp8_f32_e5m3_residual(const Instruction &inst);
[[nodiscard]] bool cvt_sr_fp8_f32_e5m3_residual(const Instruction &inst);

[[nodiscard]] constexpr RewriteDischarge checked_discharge(ResidualExpandFn check) {
  return RewriteDischarge::checked(check, RewriteDischargeContext::Instruction);
}

[[nodiscard]] constexpr RewriteDischarge block_checked_discharge(ResidualExpandFn check) {
  return RewriteDischarge::checked(check, RewriteDischargeContext::BasicBlock);
}

[[nodiscard]] constexpr RewriteDischarge no_success_discharge(const char *rationale) {
  return RewriteDischarge::no_success(rationale);
}

/// @brief Internal-linkage adapters keep registry callback addresses usable in
/// compile-time validation under GCC sanitizer builds.
[[nodiscard]] bool flat_scratch_base_rewrite_applies(const Instruction &inst) {
  return gfx1250_reads_flat_scratch_base_64bit(inst);
}

[[nodiscard]] ExpandResult lower_flat_scratch_base_rewrite(const Instruction &inst, uint64_t offset,
                                                           std::span<const uint8_t> source_text,
                                                           const LivenessAnalysis &liveness,
                                                           TranslationContext &context) {
  return gfx1250_lower_flat_scratch_base_source(inst, offset, source_text, liveness, context);
}

[[nodiscard]] bool flat_scratch_base_rewrite_residual(const Instruction &inst) {
  return gfx1250_flat_scratch_base_residual(inst);
}

/// @brief gfx1250 special-scalar operand encodings.
/// @details CRITICAL: on gfx1250 these are the INVERSE of CDNA — M0 = 125 and
/// NULL = 124, whereas CDNA encodes M0 = 124. Every hand-written encoding below
/// must use kGfx1250M0 where the machine reads/writes M0 and kGfx1250Null only
/// where a discarded/zero NULL operand is genuinely intended. See
/// gfx1250/operand_types.h (OPR_SRC_NULL = 124, OPR_SRC_M0 = 125).
constexpr uint8_t kGfx1250Null = 124;
constexpr uint8_t kGfx1250M0 = 125;

/// @brief VOP3P opcode of the WMMA-scale prefix half of a scaled-WMMA pair.
/// @details The scale prefix that fuses with a following WMMA is not a standalone
/// named VOP3P op in the generated opcode table (it is the first half of a
/// structural VOP3PX2 instruction), so its opcode is named here rather than
/// pulled from gfx1250 opcodes.
/// kWmmaScaleSrc2PrefixOp is the VOP3PX2 scale-src2 prefix; kWmmaScale16PrefixOp
/// is the VOP3PX2 Scale16 prefix.
constexpr uint16_t kWmmaScaleSrc2PrefixOp = 0x35;
constexpr uint16_t kWmmaScale16PrefixOp = 0x3a;
constexpr uint16_t kGfx1250InlineZero = 128;
constexpr uint32_t kGfx1250ScratchMaxDwordOffset = 0x7ffffcu;
constexpr uint16_t kGfx1250ModeFp16OvflHwreg = 1u | (23u << 6);
constexpr uint16_t kGfx1250WmmaCompletionWaitImmediate = 0x0f9f;

/// @brief Diagnose control fields that are invalid for floating-point WMMA.
///
/// @details CM in the matrix body and SCL_CM in a scale prefix are both
/// required to be zero. Keep this validation at every floating-point WMMA
/// entry point so malformed encodings cannot be copied or expanded.
[[nodiscard]] const char *
gfx1250_floating_wmma_control_error(const cdna5::Vop3pMachineInst &matrix,
                                    const cdna5::Vop3pMachineInst *scale = nullptr) {
  if (scale != nullptr && scale->clamp != 0)
    return "Input is malformed, SCL_CM must be set to zero for scaled floating-point WMMA";
  if (matrix.clamp != 0) {
    return "Input is malformed, CLAMP \"must be set to zero\" for WMMA/SWMMAC producing "
           "floating-point results";
  }
  return nullptr;
}

/// @brief Append a generated instruction's words to one replacement sequence.
template <size_t N>
void append_words(std::vector<uint32_t> &output, const std::array<uint32_t, N> &words) {
  output.insert(output.end(), words.begin(), words.end());
}

/// @brief Change the gfx1250 VGPR-bank mode while preserving trap recovery state.
///
/// @details SIMM16[7:0] selects the new SRC0/SRC1/SRC2/DST banks. The gfx1250
/// trap convention stores the immediately preceding mode in SIMM16[15:8]. If
/// this is the first instruction in a generated sequence, conservatively place
/// an S_NOP in front of it: the source-stream predecessor is outside the
/// expansion and may be an S_SETREG* write to MODE, which must not immediately
/// precede S_SET_VGPR_MSB. Once an expansion has emitted any instruction, that
/// instruction already provides the required separation.
///
/// An S_WAIT_XCNT 0 is emitted immediately before each S_SET_VGPR_MSB: changing
/// the VGPR-bank selection while cross-lane/memory work (XCNT) is still
/// outstanding could let instruction replay observe a different VGPR mapping.
/// Draining XCNT first makes the bank change observable to a consistent register
/// view. The S_WAIT_XCNT precedes the S_SET_VGPR_MSB and does not affect the
/// S_NOP separation from a preceding MODE write.
void append_gfx1250_vgpr_msb_transition(std::vector<uint32_t> &words, uint8_t &current_mode,
                                        uint8_t new_mode) {
  if (current_mode == new_mode)
    return;

  if (words.empty())
    append_words(words, cdna5::build_sopp(cdna5::kSNopSopp, {.simm16 = 0}));

  append_words(words, cdna5::build_sopp(cdna5::kSWaitXcntSopp, {.simm16 = 0}));
  const uint16_t immediate =
      static_cast<uint16_t>(new_mode) | (static_cast<uint16_t>(current_mode) << 8);
  append_words(words, cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = immediate}));
  current_mode = new_mode;
}

[[nodiscard]] std::optional<uint8_t> gfx1250_vgpr_mode_before(const Instruction &inst,
                                                              const LivenessAnalysis &liveness) {
  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank)
    return std::nullopt;
  return static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) |
                              (*dst_bank << 6));
}

/// @brief Save or restore one spill-backed low-bank VGPR lease through scratch ST mode.
///
/// @details VSCRATCH ST mode uses NULL SADDR, SVE=0, and a signed 24-bit
/// immediate. Semantic spill storage is non-negative and dword aligned, so the
/// largest usable dword offset is 0x7ffffc. The caller selects VGPR-MSB mode
/// zero before using this helper.
[[nodiscard]] bool append_gfx1250_scratch_preservation(std::vector<uint32_t> &words,
                                                       const SemanticScratchLease &lease,
                                                       bool restore) {
  if (!lease.spilled)
    return true;
  if (lease.reg_class != RegClass::VGPR || lease.count == 0 ||
      static_cast<uint32_t>(lease.base) + lease.count > 256u ||
      lease.spill_offset + (static_cast<uint32_t>(lease.count) - 1u) * sizeof(uint32_t) >
          kGfx1250ScratchMaxDwordOffset) {
    return false;
  }

  for (uint16_t i = 0; i < lease.count; ++i) {
    const uint8_t vgpr = static_cast<uint8_t>(lease.base + i);
    const uint32_t byte_offset = lease.spill_offset + static_cast<uint32_t>(i) * sizeof(uint32_t);
    append_words(words, cdna5::build_vscratch(restore ? cdna5::kScratchLoadB32Vscratch
                                                      : cdna5::kScratchStoreB32Vscratch,
                                              {.saddr = kGfx1250Null,
                                               .vdst = restore ? vgpr : uint8_t{0},
                                               .vsrc = restore ? uint8_t{0} : vgpr,
                                               .ioffset = byte_offset}));
  }
  append_words(words,
               cdna5::build_sopp(restore ? cdna5::kSWaitLoadcntSopp : cdna5::kSWaitStorecntSopp,
                                 {.simm16 = 0}));
  return true;
}

/// @brief Drain outstanding source operations before generated scratch writes.
///
/// @details Liveness does not model asynchronous completion. A register can be
/// selected as scratch while an earlier memory operation still owns a pending
/// write to it, allowing a late completion to corrupt the replacement.
///
/// TODO: Split SGPR-only paths to KMCNT and VGPR paths to their producer
/// counters. Per-register producer tracking is still needed to preserve
/// nonzero guest wait counts instead of draining each selected counter to zero.
void append_gfx1250_scratch_dependency_barrier(std::vector<uint32_t> &words) {
  append_words(words, cdna5::build_sopp(cdna5::kSWaitIdleSopp));
}

/// @brief How a live SGPR window is carried through low-bank VGPRs.
enum class Gfx1250SgprCarrierMode : uint8_t {
  None,       ///< Borrow only dead SGPRs; do not fall back to a carrier.
  ExecMasked, ///< Broadcast through active lanes and allow spill-backed carriers.
  LaneZero,   ///< Use EXEC-independent lane-zero operations without carrier spills.
};

/// @brief Ordinary SGPRs borrowed by one replacement sequence.
///
/// @details When the SGPR window is live, `carrier` holds its wave-uniform
/// values in low-bank VGPRs. The ordinary semantic scratch allocator may in
/// turn preserve EXEC-masked carrier VGPRs through private memory.
struct Gfx1250SgprScratchLease {
  uint16_t base = 0;
  uint16_t count = 0;
  std::optional<SemanticScratchLease> carrier;
  Gfx1250SgprCarrierMode carrier_mode = Gfx1250SgprCarrierMode::None;

  [[nodiscard]] bool has_carrier() const { return carrier.has_value(); }
};

/// @brief Constraints for one gfx1250 SGPR scratch allocation.
///
/// @details Current users request independent Wave32 masks or individual SGPRs,
/// so none requires tuple alignment. Add an explicit alignment constraint if a
/// future rule borrows an architectural SGPR tuple.
struct Gfx1250SgprScratchRequest {
  uint16_t count = 0;
  RegisterSet forbidden;
  Gfx1250SgprCarrierMode carrier_mode = Gfx1250SgprCarrierMode::None;
};

constexpr uint16_t kGfx1250MaxScratchSgprs =
    static_cast<uint16_t>(amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF);
constexpr uint16_t kGfx1250ScratchBaseSgprBegin = 102;
constexpr uint16_t kGfx1250ScratchBaseSgprEnd = 104;

/// @brief Registers whose source value or destination result must survive a rule.
///
/// @details InstDefUse reports encoded VGPR indices without VGPR-MSB state.
/// Every allocator receiving this set is capped at the low 256-register bank
/// and generated carrier code explicitly selects that bank, so an encoded
/// high-bank operand conservatively forbids its low-bank alias. Supporting
/// high-bank scratch requires physicalizing this set first.
[[nodiscard]] RegisterSet gfx1250_instruction_registers(const Instruction &inst) {
  const InstDefUse def_use(inst);
  return def_use.uses | def_use.defs;
}

/// @brief Whether an ordinary SGPR window satisfies one scratch request.
///
/// @details gfx1250 provides 106 ordinary SGPRs, so this target-specific
/// allocator deliberately extends beyond the generic 102-SGPR ceiling and can
/// use s104/s105. It excludes s102/s103 because a write to either register
/// requires a dependency wait before a later `src_flat_scratch_base` read,
/// which a local replacement cannot place in downstream guest code.
[[nodiscard]] bool gfx1250_sgpr_window_allowed(uint16_t base,
                                               const Gfx1250SgprScratchRequest &request) {
  const uint32_t end = static_cast<uint32_t>(base) + request.count;
  if (request.count == 0 || end > kGfx1250MaxScratchSgprs) {
    return false;
  }
  if (base < kGfx1250ScratchBaseSgprEnd && end > kGfx1250ScratchBaseSgprBegin) {
    return false;
  }

  RegisterSet candidate;
  candidate.expand({RegClass::SGPR, base, static_cast<uint8_t>(request.count)});
  return !candidate.intersects(request.forbidden);
}

/// @brief Prefer dead SGPRs, then preserve a live SGPR window through VGPRs.
///
/// @details SGPR values cannot be preserved directly by
/// SemanticScratchAllocator, whose leases are low-bank VGPRs. When
/// `carrier_mode` is not None, `allocator` supplies those VGPR carriers;
/// passing a null allocator deliberately disables carrier fallback.
[[nodiscard]] std::optional<Gfx1250SgprScratchLease>
acquire_gfx1250_sgprs(const Instruction &inst, const LivenessAnalysis &liveness,
                      TranslationContext &context, SemanticScratchAllocator *allocator,
                      const Gfx1250SgprScratchRequest &request) {
  if (request.count == 0 || request.count > kGfx1250MaxScratchSgprs ||
      !liveness.has_live_before(inst))
    return std::nullopt;

  const RegisterSet &live = liveness.live_before(inst);
  for (uint16_t base = 0; static_cast<uint32_t>(base) + request.count <= kGfx1250MaxScratchSgprs;
       ++base) {
    if (!gfx1250_sgpr_window_allowed(base, request))
      continue;
    RegisterSet candidate;
    candidate.expand({RegClass::SGPR, base, static_cast<uint8_t>(request.count)});
    if (!candidate.intersects(live)) {
      context.require_sgprs(static_cast<uint32_t>(base) + request.count);
      return Gfx1250SgprScratchLease{.base = base,
                                     .count = request.count,
                                     .carrier = std::nullopt,
                                     .carrier_mode = Gfx1250SgprCarrierMode::None};
    }
  }

  if (request.carrier_mode == Gfx1250SgprCarrierMode::None || allocator == nullptr)
    return std::nullopt;

  std::optional<uint16_t> victim;
  for (uint16_t base = 0; static_cast<uint32_t>(base) + request.count <= kGfx1250MaxScratchSgprs;
       ++base) {
    if (gfx1250_sgpr_window_allowed(base, request)) {
      victim = base;
      break;
    }
  }
  if (!victim)
    return std::nullopt;

  SemanticScratchRequest carrier_request;
  carrier_request.count = request.count;
  carrier_request.forbidden = request.forbidden;
  // Lane-zero carriers must remain EXEC-independent, while private-memory
  // spill/fill instructions are EXEC-masked.
  // TODO: Support simultaneous SGPR/VGPR pressure without suppressing the
  // guest instruction's memory-counter contribution.
  carrier_request.allow_spill = request.carrier_mode == Gfx1250SgprCarrierMode::ExecMasked;
  const SemanticScratchResult carrier = allocator->acquire_vgprs(carrier_request);
  // TODO: Return the carrier failure alongside the lease so a rejected
  // dynamic-stack spill can retain its actionable rule-specific diagnostic.
  if (!carrier)
    return std::nullopt;

  context.require_sgprs(static_cast<uint32_t>(*victim) + request.count);
  return Gfx1250SgprScratchLease{.base = *victim,
                                 .count = request.count,
                                 .carrier = *carrier.lease,
                                 .carrier_mode = request.carrier_mode};
}

/// @brief Save or restore a live SGPR lease through its low-bank VGPR carriers.
///
/// @details The caller must select VGPR-MSB mode zero. ExecMasked carriers
/// broadcast through active lanes and require a forward EXEC-zero guard around
/// the complete replacement. LaneZero carriers use EXEC-independent lane
/// operations and cannot themselves be spill-backed.
[[nodiscard]] bool append_gfx1250_sgpr_preservation(std::vector<uint32_t> &words,
                                                    const Gfx1250SgprScratchLease &lease,
                                                    bool restore) {
  if (!lease.has_carrier())
    return true;
  const SemanticScratchLease &carrier = *lease.carrier;
  if (carrier.reg_class != RegClass::VGPR || carrier.count != lease.count ||
      static_cast<uint32_t>(carrier.base) + carrier.count > 256u)
    return false;

  if (lease.carrier_mode == Gfx1250SgprCarrierMode::LaneZero) {
    if (carrier.spilled)
      return false;
    for (uint16_t i = 0; i < lease.count; ++i) {
      if (restore) {
        append_words(words,
                     cdna5::build_vop3(cdna5::kVReadlaneB32Vop3,
                                       {.vdst = static_cast<uint8_t>(lease.base + i),
                                        .src0 = static_cast<uint16_t>(256u + carrier.base + i),
                                        .src1 = kGfx1250InlineZero}));
      } else {
        append_words(words, cdna5::build_vop3(cdna5::kVWritelaneB32Vop3,
                                              {.vdst = static_cast<uint8_t>(carrier.base + i),
                                               .src0 = static_cast<uint16_t>(lease.base + i),
                                               .src1 = kGfx1250InlineZero}));
      }
    }
    return true;
  }
  if (lease.carrier_mode != Gfx1250SgprCarrierMode::ExecMasked)
    return false;

  if (!restore) {
    if (!append_gfx1250_scratch_preservation(words, carrier, false))
      return false;
    for (uint16_t i = 0; i < lease.count; ++i) {
      append_words(words, cdna5::build_vop1(cdna5::kVMovB32Vop1,
                                            {.src0 = static_cast<uint16_t>(lease.base + i),
                                             .vdst = static_cast<uint8_t>(carrier.base + i)}));
    }
    return true;
  }

  for (uint16_t i = 0; i < lease.count; ++i) {
    append_words(words, cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1,
                                          {.src0 = static_cast<uint16_t>(256u + carrier.base + i),
                                           .vdst = static_cast<uint8_t>(lease.base + i)}));
  }
  return append_gfx1250_scratch_preservation(words, carrier, true);
}

/// @brief Skip an EXEC-masked vector replacement when EXEC has no active lane.
///
/// @details The caller must prove that the guest instruction and every generated
/// effect are inactive under EXEC=0. Call only after the replacement word list
/// is final so its PC-relative target remains the first following instruction.
///
/// TODO: Share the branch-distance check with the CDNA EXECZ guards after the
/// architecture-specific SOPP builders have a common callback-based emitter.
[[nodiscard]] bool prepend_gfx1250_execz_guard_for_masked_replacement(std::vector<uint32_t> &words,
                                                                      bool required) {
  if (!required)
    return true;
  if (words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max()))
    return false;
  words.insert(words.begin(),
               cdna5::build_sopp(cdna5::kSCbranchExeczSopp,
                                 {.simm16 = static_cast<uint16_t>(words.size())})[0]);
  return true;
}

/// @brief Conservatively remove one hard-clause scheduling directive.
///
/// @details A legal S_CLAUSE has no architectural data result; it only groups
/// following instructions for issue. DBT transformations can change clause
/// membership and placement, and rocjitsu does not currently revalidate those
/// constraints. Replacing every clause with a same-size S_NOP is functionally
/// conservative. A future performance pass may retain clauses after proving
/// they remain valid in the translated control flow.
ExpandResult expand_gfx1250_s_clause(const Instruction &inst, uint32_t, uint64_t,
                                     std::span<const uint8_t>, const LivenessAnalysis &,
                                     TranslationContext &, const LaneLayout *, const LaneLayout *) {
  if (inst.mnemonic() != "s_clause" || inst.size() != static_cast<int>(sizeof(uint32_t)))
    return ExpandResult::failed("gfx1250 S_CLAUSE rule received an unsupported instruction");

  const auto nop = cdna5::build_sopp(cdna5::kSNopSopp, {.simm16 = 0});
  return ExpandResult::success(std::vector<uint32_t>(nop.begin(), nop.end()));
}

/// @brief Count canonical V_NOPs adjacent to @p inst inside its own basic block.
///
/// @details @p step picks the side to count. `previous_instruction` counts words
/// before @p inst, `next_instruction` counts words after it. Both stop at the
/// block boundary, and that is what makes the count usable: a counted word is in
/// the same block and therefore stays adjacent to @p inst after layout, whereas
/// a branch landing between the two would have started a new block and ended the
/// walk. Counting stops at @p limit, and a noncanonical NOP ends it, so credit is
/// never overstated.
[[nodiscard]] int count_adjacent_canonical_v_nops(const Instruction &inst, int limit,
                                                  const Instruction *(Instruction::*step)() const) {
  const uint32_t v_nop = cdna5::build_vop1(cdna5::kVNopVop1)[0];
  int counted = 0;
  for (const Instruction *at = (inst.*step)();
       counted < limit && at != nullptr && at->size() == static_cast<int>(sizeof(uint32_t)) &&
       at->raw_encoding() != nullptr && at->raw_encoding()[0] == v_nop;
       at = (at->*step)())
    ++counted;
  return counted;
}

/// @brief Give a MODE-register write its required leading V_NOP separation.
///
/// @details On this target an `s_setreg*` naming the MODE register requires two
/// V_NOPs immediately before it to be ordered against the instructions that
/// precede it. Compiler output does not supply that separation. The write is
/// commonly the first instruction of a kernel or device function, and sometimes
/// follows a short prefetch prologue.
///
/// The filler is V_NOP because the requirement names it, and because the
/// separation is required in the VALU pipeline, so it has to occupy VALU issue
/// slots. A scalar wait does not -- `s_nop 1` inserts two wait states in the
/// scalar path and orders nothing in the vector one -- so it is not a substitute
/// despite executing unconditionally. An arbitrary independent VALU instruction
/// is worse than V_NOP rather than better, being skipped outright under an empty
/// mask.
///
/// A V_NOP has no architectural result, so emitting the missing ones cannot
/// change what the program computes; it supplies the ordering distance and
/// nothing else.
///
/// Under EXEC==0 the filler contributes no VALU spacing, and the setreg still
/// executes -- but ordinary VALU is skipped under an empty mask, so there is
/// correspondingly little VALU work in flight. The separation is therefore
/// effective where EXEC != 0 and best effort in fully inactive control flow, and
/// no filler this translator may emit improves on that without touching EXEC,
/// which it must not do. The IU8 spacing rule below rests on the same reasoning.
///
/// Every MODE write gets the separation rather than some subset, because the
/// requirement is stated for the register and not for individual fields within
/// it.
///
/// The scope is the `s_setreg*` instruction, not the MODE register as a
/// location, so the other instructions that write MODE state are deliberately
/// excluded: `s_set_vgpr_msb`, whose banks this profile models as MODE fields,
/// and the SOPP writers `s_round_mode` and `s_denorm_mode`. What has to be
/// separated is the setreg's own execution against the instructions ahead of it;
/// a different opcode reaching the same fields is a different case and is not
/// covered by widening this rule. Writes naming any other hardware register are
/// declined and copied through unchanged.
///
/// V_NOPs already immediately before the write are counted and only the missing
/// slots are emitted, so a second translation is a fixed point. A counted V_NOP
/// reaches the target unchanged and with nothing appended to it: it matches no
/// expansion rule, takes no legalization entry, and needs no completion wait, so
/// it can only take the verbatim copy path.
///
/// TODO: Count separation an adjacent rule is about to emit, not just what the
/// source already holds. A dense IU8 WMMA immediately followed by a MODE write
/// yields eleven V_NOPs -- nine from the spacing rule, then two here -- where
/// nine already separate the write. It is a fixed point and costs only two issue
/// slots, and no corpus object has that adjacency. Fixing it needs the emitted
/// stream rather than the source, which is the same whole-kernel pass the IU8
/// rule's own TODO asks for; do both together.
[[nodiscard]] bool setreg_mode_ordering_residual(const Instruction &inst) {
  constexpr int kRequiredLeadingVNops = 2;

  if (inst.mnemonic() != "s_setreg_b32" && inst.mnemonic() != "s_setreg_imm32_b32")
    return false;
  const int size_bytes = inst.size();
  if (size_bytes < static_cast<int>(sizeof(uint32_t)) ||
      size_bytes % static_cast<int>(sizeof(uint32_t)) != 0 || inst.raw_encoding() == nullptr)
    return true;

  const uint16_t simm16 = static_cast<uint16_t>(inst.raw_encoding()[0] & 0xffffu);
  if (amdgpu::decode_vgpr_msb_hwreg(simm16).id != amdgpu::MODE_HWREG)
    return false;

  return count_adjacent_canonical_v_nops(inst, kRequiredLeadingVNops,
                                         &Instruction::previous_instruction) <
         kRequiredLeadingVNops;
}

ExpandResult expand_gfx1250_setreg_mode_ordering(const Instruction &inst, uint32_t, uint64_t,
                                                 std::span<const uint8_t>, const LivenessAnalysis &,
                                                 TranslationContext &, const LaneLayout *,
                                                 const LaneLayout *) {
  constexpr int kRequiredLeadingVNops = 2;

  if (inst.mnemonic() != "s_setreg_b32" && inst.mnemonic() != "s_setreg_imm32_b32")
    return ExpandResult::failed("gfx1250 MODE setreg rule received an unsupported instruction");
  const int size_bytes = inst.size();
  if (size_bytes < static_cast<int>(sizeof(uint32_t)) ||
      size_bytes % static_cast<int>(sizeof(uint32_t)) != 0 || inst.raw_encoding() == nullptr)
    return ExpandResult::failed("gfx1250 MODE setreg rule received an unsupported encoding");

  // SOPK carries the hardware-register selector in SIMM16. Only MODE requires
  // the separation, so the mnemonic alone cannot decide this.
  const uint16_t simm16 = static_cast<uint16_t>(inst.raw_encoding()[0] & 0xffffu);
  if (amdgpu::decode_vgpr_msb_hwreg(simm16).id != amdgpu::MODE_HWREG)
    return ExpandResult::not_handled();

  const int existing_slots = count_adjacent_canonical_v_nops(inst, kRequiredLeadingVNops,
                                                             &Instruction::previous_instruction);
  if (existing_slots == kRequiredLeadingVNops)
    return ExpandResult::not_handled();

  const size_t words_in_encoding = static_cast<size_t>(size_bytes) / sizeof(uint32_t);
  std::vector<uint32_t> words;
  words.reserve(words_in_encoding + kRequiredLeadingVNops);
  words.insert(words.end(), static_cast<size_t>(kRequiredLeadingVNops - existing_slots),
               cdna5::build_vop1(cdna5::kVNopVop1)[0]);
  words.insert(words.end(), inst.raw_encoding(), inst.raw_encoding() + words_in_encoding);
  return ExpandResult::success(std::move(words));
}

/// @brief Decline the one barrier id this profile excludes.
///
/// @details Barrier id -3 is the only one this instruction may not name; every
/// other id stays on the copy path.
///
/// The decision reads the raw SSRC0 field rather than the decoded operand
/// value. This operand takes an inline constant or M0 and nothing else, so the
/// excluded id has exactly one spelling the encoding can state: inline selector
/// 195. Comparing decoded values instead would not separate that spelling from
/// a register-supplied id.
///
/// The M0 form (selector 125) is copied through, and that is not a hole in the
/// static check: it takes the id from a zero-extended low field, so it cannot
/// produce a negative id and therefore cannot name the excluded one at run
/// time. Were a register-held id able to reach that value, copying would not be
/// fail-closed and this rule would have to refuse the dynamic form instead.
ExpandResult expand_gfx1250_barrier_signal_isfirst(const Instruction &inst, uint32_t,
                                                   uint64_t offset,
                                                   std::span<const uint8_t> source_text,
                                                   const LivenessAnalysis &, TranslationContext &,
                                                   const LaneLayout *, const LaneLayout *) {
  constexpr uint32_t kExcludedBarrierIdInline = 195;

  const size_t size = static_cast<size_t>(inst.size());
  if (size < sizeof(uint32_t) || offset + size > source_text.size())
    return ExpandResult::failed("gfx1250 barrier-signal rule received an unsupported instruction");

  uint32_t word0 = 0;
  std::memcpy(&word0, source_text.data() + offset, sizeof(word0));
  if ((word0 & 0xffu) != kExcludedBarrierIdInline)
    return ExpandResult::not_handled();

  return ExpandResult::failed(
      "gfx1250 s_barrier_signal_isfirst cannot name barrier id -3 (inline selector 195)",
      {"Use a different barrier id, or signal it without the first-signal form."});
}

struct Gfx1250Ds2Shape {
  uint16_t replacement_opcode = 0;
  uint8_t element_dwords = 0;
  bool stride64 = false;
  enum class Kind : uint8_t { Load, Store, StoreExchange } kind = Kind::Load;
};

/// @brief Describe one B0 DS2 opcode and its A0 single-address replacement.
[[nodiscard]] Gfx1250Ds2Shape gfx1250_ds2_shape(uint16_t opcode) {
  using Kind = Gfx1250Ds2Shape::Kind;
  switch (opcode) {
  case cdna5::kDsLoad2addrB32Vds:
    return {cdna5::kDsLoadB32Vds, 1, false, Kind::Load};
  case cdna5::kDsLoad2addrStride64B32Vds:
    return {cdna5::kDsLoadB32Vds, 1, true, Kind::Load};
  case cdna5::kDsStore2addrB32Vds:
    return {cdna5::kDsStoreB32Vds, 1, false, Kind::Store};
  case cdna5::kDsStore2addrStride64B32Vds:
    return {cdna5::kDsStoreB32Vds, 1, true, Kind::Store};
  case cdna5::kDsStorexchg2addrRtnB32Vds:
    return {cdna5::kDsStorexchgRtnB32Vds, 1, false, Kind::StoreExchange};
  case cdna5::kDsStorexchg2addrStride64RtnB32Vds:
    return {cdna5::kDsStorexchgRtnB32Vds, 1, true, Kind::StoreExchange};
  case cdna5::kDsLoad2addrB64Vds:
    return {cdna5::kDsLoadB64Vds, 2, false, Kind::Load};
  case cdna5::kDsLoad2addrStride64B64Vds:
    return {cdna5::kDsLoadB64Vds, 2, true, Kind::Load};
  case cdna5::kDsStore2addrB64Vds:
    return {cdna5::kDsStoreB64Vds, 2, false, Kind::Store};
  case cdna5::kDsStore2addrStride64B64Vds:
    return {cdna5::kDsStoreB64Vds, 2, true, Kind::Store};
  case cdna5::kDsStorexchg2addrRtnB64Vds:
    return {cdna5::kDsStorexchgRtnB64Vds, 2, false, Kind::StoreExchange};
  case cdna5::kDsStorexchg2addrStride64RtnB64Vds:
    return {cdna5::kDsStorexchgRtnB64Vds, 2, true, Kind::StoreExchange};
  default:
    return {};
  }
}

/// @brief Build one single-address DS instruction from a DS2 operand half.
[[nodiscard]] std::array<uint32_t, 2> build_gfx1250_ds2_half(const cdna5::VdsMachineInst &source,
                                                             const Gfx1250Ds2Shape &shape,
                                                             uint16_t byte_offset,
                                                             bool second_half) {
  const uint8_t tuple_delta = second_half ? shape.element_dwords : 0;
  // Plain DS stores have no destination operand, and their reserved VDST field
  // must remain zero. Loads and returning exchanges use consecutive VDST
  // tuples for the two halves.
  const uint8_t vdst = shape.kind == Gfx1250Ds2Shape::Kind::Store
                           ? 0
                           : static_cast<uint8_t>(source.vdst + tuple_delta);
  return cdna5::build_vds(shape.replacement_opcode,
                          {.offset0 = static_cast<uint8_t>(byte_offset),
                           .offset1 = static_cast<uint8_t>(byte_offset >> 8),
                           .addr = static_cast<uint8_t>(source.addr),
                           // A single-address store/exchange consumes DATA0. The second DS2 data
                           // operand therefore moves from the source DATA1 field into DATA0.
                           .data0 = static_cast<uint8_t>(second_half ? source.data1 : source.data0),
                           .data1 = 0,
                           .vdst = vdst});
}

/// @brief Expand a gfx1250 B0 two-address DS operation for A0.
///
/// @details A0 and B0 use different DS2 offset alignment rules. The B0-to-A0
/// profile translates the operation into two ordinary DS operations with byte
/// offsets. A local DSCNT drain preserves the completion semantics of the
/// original instruction without rewriting downstream wait counts.
ExpandResult expand_gfx1250_ds2(const Instruction &inst, uint32_t, uint64_t,
                                std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                TranslationContext &, const LaneLayout *, const LaneLayout *) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(cdna5::VdsMachineInst)) {
    return ExpandResult::failed("gfx1250 DS2 instruction has no complete VDS encoding",
                                {"Decode the complete eight-byte VDS instruction."});
  }

  cdna5::VdsMachineInst source{};
  std::memcpy(&source, raw, sizeof(source));
  const Gfx1250Ds2Shape shape = gfx1250_ds2_shape(inst.opcode());
  if (shape.element_dwords == 0) {
    return ExpandResult::failed("gfx1250 DS2 semantic rule received an unsupported opcode");
  }

  // DS2 immediates are element indices. Stride64 forms add another factor of
  // 64; ordinary single-address DS instructions instead encode a 16-bit byte
  // offset directly.
  const uint32_t byte_scale =
      static_cast<uint32_t>(shape.element_dwords) * sizeof(uint32_t) * (shape.stride64 ? 64u : 1u);
  const uint32_t offset0 = static_cast<uint32_t>(source.offset0) * byte_scale;
  const uint32_t offset1 = static_cast<uint32_t>(source.offset1) * byte_scale;
  constexpr uint32_t kSingleAddressOffsetMax = 0xffff;
  if (offset0 > kSingleAddressOffsetMax || offset1 > kSingleAddressOffsetMax) {
    return ExpandResult::failed(
        "gfx1250 DS2 scaled offset exceeds the single-address 16-bit field",
        {"Use a scratch-address lowering for DS2 offsets larger than 65535 bytes."});
  }

  const auto first = build_gfx1250_ds2_half(source, shape, static_cast<uint16_t>(offset0), false);
  const auto second = build_gfx1250_ds2_half(source, shape, static_cast<uint16_t>(offset1), true);
  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed(
        "gfx1250 DS2 lowering cannot prove the VGPR-MSB mode",
        {"Make the VGPR-MSB fields known on every CFG path reaching this instruction."});
  }
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));

  const auto physical = [](uint8_t selector, uint8_t bank) {
    return static_cast<uint16_t>(static_cast<uint16_t>(bank) * 256u + selector);
  };
  const auto physical_overlap = [](uint16_t lhs, uint8_t lhs_width, uint16_t rhs,
                                   uint8_t rhs_width) {
    return lhs < static_cast<uint32_t>(rhs) + rhs_width &&
           rhs < static_cast<uint32_t>(lhs) + lhs_width;
  };

  const uint16_t first_dst = physical(static_cast<uint8_t>(source.vdst), *dst_bank);
  const uint32_t second_dst_wide = static_cast<uint32_t>(first_dst) + shape.element_dwords;
  if ((shape.kind == Gfx1250Ds2Shape::Kind::Load ||
       shape.kind == Gfx1250Ds2Shape::Kind::StoreExchange) &&
      second_dst_wide + shape.element_dwords > 1024u) {
    return ExpandResult::failed("gfx1250 DS2 destination tuple exceeds the VGPR address space");
  }
  const uint16_t second_dst = static_cast<uint16_t>(second_dst_wide);
  const uint8_t second_dst_bank = static_cast<uint8_t>(second_dst / 256u);

  // DATA1 is a SRC2 operand in DS2 but becomes DATA0/SRC1 in the second
  // single-address instruction. Select its original bank for the new role.
  const uint8_t second_src1_bank =
      shape.kind == Gfx1250Ds2Shape::Kind::Load ? *src1_bank : *src2_bank;
  const uint8_t second_mode = static_cast<uint8_t>(*src0_bank | (second_src1_bank << 2) |
                                                   (*src2_bank << 4) | (second_dst_bank << 6));
  bool second_first = false;

  if (shape.kind == Gfx1250Ds2Shape::Kind::Load) {
    // The compound load captures ADDR before writing either destination half.
    // If the first half aliases ADDR, issue the independent second load first.
    second_first = physical_overlap(first_dst, shape.element_dwords,
                                    physical(static_cast<uint8_t>(source.addr), *src0_bank), 1);
  } else if (shape.kind == Gfx1250Ds2Shape::Kind::StoreExchange) {
    // Each exchange writes one destination half while the other still needs
    // ADDR and its input data. Pick a safe direction. A dependency in both
    // directions needs scratch storage and must fail closed for now.
    const bool first_clobbers_second =
        physical_overlap(first_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.addr), *src0_bank), 1) ||
        physical_overlap(first_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.data1), *src2_bank),
                         shape.element_dwords);
    const bool second_clobbers_first =
        physical_overlap(second_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.addr), *src0_bank), 1) ||
        physical_overlap(second_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.data0), *src1_bank),
                         shape.element_dwords);
    if (first_clobbers_second && second_clobbers_first) {
      return ExpandResult::failed("gfx1250 DS2 exchange has cyclic destination/source overlap",
                                  {"Add a scratch-VGPR DS2 exchange lowering for cyclic overlap."});
    }
    second_first = first_clobbers_second;
  }

  std::vector<uint32_t> words;
  words.reserve(9);
  uint8_t current_mode = original_mode;
  const auto set_mode = [&](uint8_t mode) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, mode);
  };
  if (second_first) {
    if (second_mode != original_mode)
      set_mode(second_mode);
    append_words(words, second);
    if (second_mode != original_mode)
      set_mode(original_mode);
    append_words(words, first);
  } else {
    append_words(words, first);
    if (second_mode != original_mode)
      set_mode(second_mode);
    append_words(words, second);
    if (second_mode != original_mode)
      set_mode(original_mode);
  }
  append_words(words, cdna5::build_sopp(cdna5::kSWaitDscntSopp, {.simm16 = 0}));
  return ExpandResult::success(std::move(words));
}

/// @brief Canonical save/clear/restore words emitted around one tensor load.
struct TensorMaskWrapper {
  uint32_t save = 0;
  uint32_t clear = 0;
  uint32_t restore = 0;
};

/// @brief Build the canonical descriptor-mask clear word.
[[nodiscard]] uint32_t build_tensor_mask_clear(uint8_t descriptor_base) {
  return cdna5::build_sop2(
      cdna5::kSPackHhB32B16Sop2,
      {.ssrc0 = kGfx1250InlineZero, .ssrc1 = descriptor_base, .sdst = descriptor_base})[0];
}

/// @brief Build the canonical save/clear/restore words around one tensor load.
[[nodiscard]] TensorMaskWrapper build_tensor_mask_wrapper(uint8_t descriptor_base,
                                                          uint8_t scratch) {
  return {
      .save =
          cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = descriptor_base, .sdst = scratch})[0],
      .clear = build_tensor_mask_clear(descriptor_base),
      .restore =
          cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = scratch, .sdst = descriptor_base})[0],
  };
}

/// @brief Check for one exact, contiguous predecessor in the same basic block.
[[nodiscard]] bool has_canonical_predecessor(const Instruction &inst, uint32_t expected_word) {
  const Instruction *previous = inst.previous_instruction();
  return previous != nullptr && previous->size() == static_cast<int>(sizeof(uint32_t)) &&
         previous->src_loc() + sizeof(uint32_t) == inst.src_loc() &&
         previous->raw_encoding() != nullptr && previous->raw_encoding()[0] == expected_word;
}

/// @brief Check whether every path to a tensor load executes the canonical mask clear.
[[nodiscard]] bool has_tensor_mask_clear(const Instruction &inst, uint8_t descriptor_base) {
  return has_canonical_predecessor(inst, build_tensor_mask_clear(descriptor_base));
}

/// @brief Whether the tensor-load expansion still needs to add its canonical prefix.
[[nodiscard]] bool tensor_load_residual(const Instruction &inst) {
  if (inst.mnemonic() != "tensor_load_to_lds" ||
      inst.size() != static_cast<int>(sizeof(cdna5::VimageMachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return true;
  }

  cdna5::VimageMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint8_t kLastOrdinarySgpr = 105;
  const uint8_t descriptor_base = static_cast<uint8_t>(source.vaddr1);
  if (descriptor_base == kGfx1250Null || descriptor_base > kLastOrdinarySgpr - 7u)
    return true;
  return !has_tensor_mask_clear(inst, descriptor_base);
}

/// @brief Disable Tensor-DMA multicast for one A0 tensor load.
///
/// @details TENSOR_LOAD_TO_LDS does not encode multicast in the instruction.
/// Descriptor group 1 bits [15:0], held in the first SGPR named by VADDR1,
/// select the workgroups which receive a multicast load. On A0 those bits must
/// therefore be cleared for every tensor load; inspecting only the instruction
/// cannot prove that the runtime descriptor mask is zero. Preserve the guest
/// descriptor value around the load because later tensor instructions commonly
/// reuse and update the same descriptor.
///
/// A second translation preserves the load when its immediately preceding
/// decoded instruction in the same basic block is the canonical mask clear.
/// This proves that no control-flow edge can bypass the clear. The clear
/// instruction currently has no B0-to-A0 semantic rule and is copied unchanged;
/// this reuse condition must be revisited if such a rule is added.
ExpandResult expand_gfx1250_tensor_load_to_lds(const Instruction &inst, uint32_t, uint64_t,
                                               std::span<const uint8_t>,
                                               const LivenessAnalysis &liveness,
                                               TranslationContext &context, const LaneLayout *,
                                               const LaneLayout *) {
  if (inst.mnemonic() != "tensor_load_to_lds" ||
      inst.size() != static_cast<int>(sizeof(cdna5::VimageMachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 tensor-load mask rule received an unsupported instruction");
  }
  cdna5::VimageMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint8_t kLastOrdinarySgpr = 105;
  const uint8_t descriptor_base = static_cast<uint8_t>(source.vaddr1);
  if (descriptor_base == kGfx1250Null || descriptor_base > kLastOrdinarySgpr - 7u) {
    return ExpandResult::failed(
        "gfx1250 tensor-load group-1 descriptor is not a valid eight-SGPR tuple",
        {"Provide TENSOR_LOAD_TO_LDS VADDR1 as an ordinary eight-SGPR descriptor."});
  }

  if (!tensor_load_residual(inst)) {
    return ExpandResult::success(std::vector<uint32_t>(
        inst.raw_encoding(),
        inst.raw_encoding() + sizeof(cdna5::VimageMachineInst) / sizeof(uint32_t)));
  }

  Gfx1250SgprScratchRequest scratch_request;
  scratch_request.count = 1;
  scratch_request.forbidden = gfx1250_instruction_registers(inst);
  const auto scratch = acquire_gfx1250_sgprs(inst, liveness, context, nullptr, scratch_request);
  if (!scratch || scratch->base > kLastOrdinarySgpr) {
    return ExpandResult::failed("gfx1250 tensor-load mask rule could not allocate scalar scratch",
                                {"Provide one dead ordinary SGPR."});
  }

  std::vector<uint32_t> words;
  words.reserve(6);
  append_gfx1250_scratch_dependency_barrier(words);

  const TensorMaskWrapper wrapper =
      build_tensor_mask_wrapper(descriptor_base, static_cast<uint8_t>(scratch->base));
  words.push_back(wrapper.save);
  // PACK_HH forms {SRC1[31:16], SRC0[31:16]}. Inline zero as SRC0 clears
  // D1[15:0] while preserving all descriptor fields in D1[31:16].
  words.push_back(wrapper.clear);
  words.insert(words.end(), inst.raw_encoding(),
               inst.raw_encoding() + sizeof(cdna5::VimageMachineInst) / sizeof(uint32_t));
  words.push_back(wrapper.restore);

  return ExpandResult::success(std::move(words));
}

/// @brief Replace one bit field in a 32-bit instruction word.
void set_word_field(uint32_t &word, uint32_t value, uint32_t shift, uint32_t width) {
  const uint32_t mask = ((uint32_t{1} << width) - 1) << shift;
  word = (word & ~mask) | ((value << shift) & mask);
}

/// @brief Whether a regular-Scale compound still needs normalization or splitting.
[[nodiscard]] bool regular_scale_residual(const Instruction &inst) {
  if (!inst.mnemonic().starts_with("v_wmma_scale_f32_") ||
      inst.size() != 4 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return true;
  }

  cdna5::Vop3pMachineInst scale{};
  cdna5::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (gfx1250_floating_wmma_control_error(matrix, &scale) != nullptr)
    return true;
  return matrix.op == cdna5::kVWmmaF3232x16x128F4Vop3p || scale.src2 != 0x100;
}

/// @brief Emit two A0 M=16 FP4 operations for one M=32 FP4 matrix operation.
///
/// @details A0 has no M=32 FP4 matrix opcode. Rows 0..15 and 16..31 are
/// independent, so each half slices eight dwords from A, C, and D while sharing
/// matrix B. @p prefix supplies the scale-prefix words used by both halves with
/// their scale sources already selected; this helper sets the per-half
/// SCL_OPSEL, clears the reuse promises that the split invalidates (each half
/// names a different A/D range), and points the architecturally unused scale
/// SRC2 at VGPR0. Both replacement matrix-format fields are forced to FP4.
///
/// @p shared_low_bank_inputs names further low-bank registers that the second
/// half still reads, so the first half's destination must not overwrite them.
/// @p context prefixes the diagnostics with the caller's source form.
[[nodiscard]] ExpandResult
gfx1250_split_m32_fp4(const Instruction &inst, const LivenessAnalysis &liveness,
                      const cdna5::Vop3pMachineInst &matrix, std::array<uint32_t, 2> prefix,
                      std::span<const uint16_t> shared_low_bank_inputs, std::string_view context) {
  constexpr uint16_t kVgprEncoding = 256;
  constexpr uint16_t kHalfDwords = 8;

  const bool src2_is_vgpr = matrix.src2 >= kVgprEncoding;
  const uint16_t src0 = static_cast<uint16_t>(matrix.src0 - kVgprEncoding);
  const uint16_t src1 = static_cast<uint16_t>(matrix.src1 - kVgprEncoding);
  const uint16_t src2 = src2_is_vgpr ? static_cast<uint16_t>(matrix.src2 - kVgprEncoding) : 0;

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed(std::string(context) + " FP4 split cannot prove the VGPR-MSB mode");
  }

  const auto physical = [](uint8_t bank, uint16_t reg) {
    return static_cast<uint16_t>(bank * 256u + reg);
  };
  const auto overlaps = [](uint16_t lhs, uint16_t lhs_count, uint16_t rhs, uint16_t rhs_count) {
    return lhs < static_cast<uint32_t>(rhs) + rhs_count &&
           rhs < static_cast<uint32_t>(lhs) + lhs_count;
  };
  const uint16_t first_dst = physical(*dst_bank, matrix.vdst);
  const uint16_t upper_a = physical(*src0_bank, static_cast<uint16_t>(src0 + kHalfDwords));
  const uint16_t shared_b = physical(*src1_bank, src1);
  bool clobbers_shared_input =
      overlaps(first_dst, kHalfDwords, upper_a, kHalfDwords) ||
      overlaps(first_dst, kHalfDwords, shared_b, kHalfDwords) ||
      (src2_is_vgpr &&
       overlaps(first_dst, kHalfDwords,
                physical(*src2_bank, static_cast<uint16_t>(src2 + kHalfDwords)), kHalfDwords));
  for (uint16_t shared : shared_low_bank_inputs)
    clobbers_shared_input |= overlaps(first_dst, kHalfDwords, physical(0, shared), 1);
  if (clobbers_shared_input) {
    return ExpandResult::failed(std::string(context) +
                                " lower destination overlaps an input needed by the upper half");
  }

  const bool dst_crosses = static_cast<uint32_t>(matrix.vdst) + kHalfDwords > 0xffu;
  const bool src0_crosses = static_cast<uint32_t>(src0) + kHalfDwords > 0xffu;
  const bool src2_crosses = src2_is_vgpr && static_cast<uint32_t>(src2) + kHalfDwords > 0xffu;
  if ((src0_crosses && *src0_bank == 3) || (dst_crosses && *dst_bank == 3) ||
      (src2_crosses && *src2_bank == 3)) {
    return ExpandResult::failed(std::string(context) + " FP4 split exceeds the VGPR address space");
  }

  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
  std::vector<uint32_t> words;
  words.reserve(12);
  uint8_t current_mode = original_mode;
  for (uint16_t half = 0; half < 2; ++half) {
    if (half != 0 && (src0_crosses || dst_crosses || src2_crosses)) {
      const uint8_t upper_mode =
          static_cast<uint8_t>((*src0_bank + (src0_crosses ? 1u : 0u)) | (*src1_bank << 2) |
                               ((*src2_bank + (src2_crosses ? 1u : 0u)) << 4) |
                               ((*dst_bank + (dst_crosses ? 1u : 0u)) << 6));
      append_gfx1250_vgpr_msb_transition(words, current_mode, upper_mode);
    }

    std::array<uint32_t, 2> half_prefix = prefix;
    half_prefix[0] &= ~((uint32_t{1} << 13) | (uint32_t{1} << 14));
    set_word_field(half_prefix[0], half, 11, 1);          // SCL_OPSEL: select this M half.
    set_word_field(half_prefix[1], kVgprEncoding, 18, 9); // unused scale_src2 = v0.
    append_words(words, half_prefix);

    const uint16_t delta = static_cast<uint16_t>(half * kHalfDwords);
    auto replacement = cdna5::build_vop3p(
        cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
        {.vdst = static_cast<uint8_t>(matrix.vdst + delta),
         .neg_hi = static_cast<uint8_t>(matrix.neg_hi),
         .opsel = 4, // matrix A: FP4
         .clamp = static_cast<uint8_t>(matrix.clamp),
         .src0 = static_cast<uint16_t>(kVgprEncoding + ((src0 + delta) & 0xffu)),
         .src1 = static_cast<uint16_t>(kVgprEncoding + src1),
         .src2 = src2_is_vgpr ? static_cast<uint16_t>(kVgprEncoding + ((src2 + delta) & 0xffu))
                              : static_cast<uint16_t>(matrix.src2),
         .opsel_hi = 0,
         .neg = static_cast<uint8_t>(matrix.neg)});
    replacement[0] |= uint32_t{1} << 14; // matrix B: FP4
    append_words(words, replacement);
  }
  if (src0_crosses || dst_crosses || src2_crosses)
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  return ExpandResult::success(std::move(words));
}

/// @brief Apply the B0-to-A0 regular-scale translation, including the M=32 split.
///
/// @details The translation profile encodes VGPR0 (0x100) in the VOP3PX2
/// `scale_src2` field for the M=16 form.
///
/// gfx1250 B0 additionally introduces the M=32 FP4 form, which A0 lacks. The
/// scale layout assigns M=0..15 and M=16..31 to lanes 0..15 and 16..31 of the
/// same A-scale VGPR, so the split reuses the source prefix and lets SCL_OPSEL
/// select the matching lane half. Matrix B and its scale are shared.
ExpandResult expand_gfx1250_wmma_scale_src2(const Instruction &inst, uint32_t, uint64_t,
                                            std::span<const uint8_t>,
                                            const LivenessAnalysis &liveness, TranslationContext &,
                                            const LaneLayout *, const LaneLayout *) {
  if (!inst.mnemonic().starts_with("v_wmma_scale_f32_") || inst.size() != 4 * sizeof(uint32_t) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 scaled-WMMA SRC2 rule received an unsupported VOP3PX2 instruction");
  }

  cdna5::Vop3pMachineInst scale{};
  cdna5::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (const char *error = gfx1250_floating_wmma_control_error(matrix, &scale))
    return ExpandResult::failed(error);
  if (matrix.op != cdna5::kVWmmaF3232x16x128F4Vop3p) {
    std::vector<uint32_t> words(inst.raw_encoding(), inst.raw_encoding() + 4);
    if (regular_scale_residual(inst)) {
      // Instruction bits [58:50] occupy word 1 bits [26:18].
      set_word_field(words[1], 0x100, 18, 9);
    }
    return ExpandResult::success(std::move(words));
  }

  constexpr uint16_t kVgprEncoding = 256;
  if (matrix.src0 < kVgprEncoding || matrix.src1 < kVgprEncoding) {
    return ExpandResult::failed(
        "gfx1250 regular-Scale 32x16 FP4 matrix inputs are not VGPR ranges");
  }

  // A scale operand held in the vector file stays live across the split, so the
  // first half's destination must not overwrite it. One that is not a vector
  // register names no range and cannot be clobbered.
  std::array<uint16_t, 2> scale_inputs{};
  size_t scale_input_count = 0;
  for (const uint16_t encoded :
       {static_cast<uint16_t>(scale.src0), static_cast<uint16_t>(scale.src1)}) {
    if (encoded >= kVgprEncoding)
      scale_inputs[scale_input_count++] = static_cast<uint16_t>(encoded - kVgprEncoding);
  }

  const std::array<uint32_t, 2> prefix = {inst.raw_encoding()[0], inst.raw_encoding()[1]};
  return gfx1250_split_m32_fp4(inst, liveness, matrix, prefix,
                               std::span<const uint16_t>(scale_inputs.data(), scale_input_count),
                               "gfx1250 regular-Scale 32x16");
}

/// @brief Translate the standalone B0 32x16 FP4 WMMA for A0.
///
/// @details A0 has neither this M=32 opcode nor an unprefixed low-precision
/// matrix operation that trap recovery can resume, so the lowering is the same
/// M=16 split as the scaled form wrapped in neutral scale prefixes. In a scale
/// source, inline integer zero reads as the E8M0 value for 1.0 and applies to
/// every matrix element, so the per-half lane selector carries no meaning here.
ExpandResult expand_gfx1250_wmma_32x16_f4(const Instruction &inst, uint32_t, uint64_t,
                                          std::span<const uint8_t>,
                                          const LivenessAnalysis &liveness, TranslationContext &,
                                          const LaneLayout *, const LaneLayout *) {
  if (inst.mnemonic() != "v_wmma_f32_32x16x128_f4" ||
      inst.opcode() != cdna5::kVWmmaF3232x16x128F4Vop3p ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 32x16 FP4 WMMA rule received an unsupported instruction");
  }

  cdna5::Vop3pMachineInst matrix{};
  std::memcpy(&matrix, inst.raw_encoding(), sizeof(matrix));
  // This form produces floating-point results, so it carries the same control
  // requirement as every other floating-point matrix entry point.
  if (const char *error = gfx1250_floating_wmma_control_error(matrix))
    return ExpandResult::failed(error);
  constexpr uint16_t kVgprEncoding = 256;
  if (matrix.src0 < kVgprEncoding || matrix.src1 < kVgprEncoding)
    return ExpandResult::failed("gfx1250 32x16 FP4 operands are not VGPR ranges");

  const auto prefix = cdna5::build_vop3p(
      kWmmaScaleSrc2PrefixOp,
      {.src0 = kGfx1250InlineZero, .src1 = kGfx1250InlineZero, .src2 = kVgprEncoding});
  return gfx1250_split_m32_fp4(inst, liveness, matrix, prefix, {}, "gfx1250 32x16");
}

/// @brief Encode an inline non-negative integer accepted by a VALU source.
[[nodiscard]] constexpr uint16_t gfx1250_inline_u32(uint16_t value) {
  return static_cast<uint16_t>(128u + value);
}

/// @brief Encode one low-bank VGPR as a generic VALU source operand.
[[nodiscard]] constexpr uint16_t gfx1250_vgpr_src(uint16_t vgpr) {
  return static_cast<uint16_t>(256u + vgpr);
}

/// @brief Preserve M=16 Scale16 and split the B0 M=32 FP4 form across M for A0.
///
/// @details The M=16 Scale16 instruction is available on both revisions, so its
/// four-DWORD encoding is retained except for the required VGPR0 encoding in
/// the scale prefix's unused SRC2 field.
///
/// The M=32 FP4 form is represented by two native M=16 Scale16 instructions.
/// Matrix A, C, and D are sliced by eight dwords; matrix B and both Scale16
/// sources are shared. The first operation covers rows 0..15 using A-scale
/// lanes 0..15, and the second covers rows 16..31 using A-scale lanes 16..31.
/// This partitions independent output rows without introducing a new
/// accumulation boundary along K.
ExpandResult expand_gfx1250_wmma_scale16(const Instruction &inst, uint32_t, uint64_t,
                                         std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                         TranslationContext &, const LaneLayout *,
                                         const LaneLayout *) {
  // The prefix opcode shares its structural lookup key with ordinary VOP3P
  // instructions. Decline those collisions so their own legalization can
  // diagnose them instead of reporting a misleading Scale16 error.
  if (!inst.mnemonic().starts_with("v_wmma_scale16_f32_"))
    return ExpandResult::not_handled();
  if (inst.size() != 4 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA rule received an unsupported VOP3PX2 instruction");
  }

  cdna5::Vop3pMachineInst scale{};
  cdna5::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (scale.op != kWmmaScale16PrefixOp || (matrix.op != cdna5::kVWmmaF3216x16x128F8f6f4Vop3p &&
                                           matrix.op != cdna5::kVWmmaF3232x16x128F4Vop3p)) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA rule received an unsupported base opcode");
  }
  if (const char *error = gfx1250_floating_wmma_control_error(matrix, &scale))
    return ExpandResult::failed(error);

  constexpr uint16_t kVgprEncoding = 256;
  const std::array<uint16_t, 2> encoded_scales = {static_cast<uint16_t>(scale.src0),
                                                  static_cast<uint16_t>(scale.src1)};
  for (const uint16_t encoded_scale : encoded_scales) {
    if (encoded_scale < kVgprEncoding)
      continue;
    const uint16_t base = static_cast<uint16_t>(encoded_scale - kVgprEncoding);
    if ((base & 1u) != 0 || base > 254u) {
      return ExpandResult::failed(
          "gfx1250 Scale16 VGPR scale sources must be even-aligned pairs in v0:v255");
    }
  }
  if (matrix.op == cdna5::kVWmmaF3216x16x128F8f6f4Vop3p) {
    std::vector<uint32_t> words(inst.raw_encoding(), inst.raw_encoding() + 4);
    set_word_field(words[1], kVgprEncoding, 18, 9);
    return ExpandResult::success(std::move(words));
  }

  if (matrix.src0 < kVgprEncoding || matrix.src1 < kVgprEncoding)
    return ExpandResult::failed("gfx1250 Scale16 32x16 FP4 matrix inputs are not VGPR ranges");

  constexpr uint16_t kHalfDwords = 8;
  const uint16_t src0 = static_cast<uint16_t>(matrix.src0 - kVgprEncoding);
  const uint16_t src1 = static_cast<uint16_t>(matrix.src1 - kVgprEncoding);
  const bool src2_is_vgpr = matrix.src2 >= kVgprEncoding;
  const uint16_t src2 = src2_is_vgpr ? static_cast<uint16_t>(matrix.src2 - kVgprEncoding) : 0;

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 Scale16 32x16 FP4 split cannot prove the VGPR-MSB mode");
  }

  const auto physical = [](uint8_t bank, uint16_t reg) {
    return static_cast<uint16_t>(bank * 256u + reg);
  };
  const auto overlaps = [](uint16_t lhs, uint16_t lhs_count, uint16_t rhs, uint16_t rhs_count) {
    return lhs < static_cast<uint32_t>(rhs) + rhs_count &&
           rhs < static_cast<uint32_t>(lhs) + lhs_count;
  };
  const uint16_t first_dst = physical(*dst_bank, matrix.vdst);
  const uint16_t upper_a = physical(*src0_bank, static_cast<uint16_t>(src0 + kHalfDwords));
  const uint16_t shared_b = physical(*src1_bank, src1);
  bool first_dst_overlaps_scale = false;
  for (const uint16_t encoded_scale : encoded_scales) {
    if (encoded_scale >= kVgprEncoding) {
      const uint16_t scale_base = static_cast<uint16_t>(encoded_scale - kVgprEncoding);
      first_dst_overlaps_scale |= overlaps(first_dst, kHalfDwords, scale_base, 2);
    }
  }
  if (overlaps(first_dst, kHalfDwords, upper_a, kHalfDwords) ||
      overlaps(first_dst, kHalfDwords, shared_b, kHalfDwords) || first_dst_overlaps_scale ||
      (src2_is_vgpr &&
       overlaps(first_dst, kHalfDwords,
                physical(*src2_bank, static_cast<uint16_t>(src2 + kHalfDwords)), kHalfDwords))) {
    return ExpandResult::failed(
        "gfx1250 Scale16 32x16 lower destination overlaps an input needed by the upper half");
  }

  const bool dst_crosses = static_cast<uint32_t>(matrix.vdst) + kHalfDwords > 0xffu;
  const bool src0_crosses = static_cast<uint32_t>(src0) + kHalfDwords > 0xffu;
  const bool src2_crosses = src2_is_vgpr && static_cast<uint32_t>(src2) + kHalfDwords > 0xffu;
  if ((src0_crosses && *src0_bank == 3) || (dst_crosses && *dst_bank == 3) ||
      (src2_crosses && *src2_bank == 3)) {
    return ExpandResult::failed("gfx1250 Scale16 32x16 FP4 split exceeds the VGPR address space");
  }

  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
  std::vector<uint32_t> words;
  words.reserve(12);
  uint8_t current_mode = original_mode;
  for (uint16_t half = 0; half < 2; ++half) {
    if (half != 0 && (src0_crosses || dst_crosses || src2_crosses)) {
      const uint8_t upper_mode =
          static_cast<uint8_t>((*src0_bank + (src0_crosses ? 1u : 0u)) | (*src1_bank << 2) |
                               ((*src2_bank + (src2_crosses ? 1u : 0u)) << 4) |
                               ((*dst_bank + (dst_crosses ? 1u : 0u)) << 6));
      append_gfx1250_vgpr_msb_transition(words, current_mode, upper_mode);
    }

    std::array<uint32_t, 2> prefix = {inst.raw_encoding()[0], inst.raw_encoding()[1]};
    prefix[0] &= ~((uint32_t{1} << 13) | (uint32_t{1} << 14));
    set_word_field(prefix[0], half, 11, 1);
    set_word_field(prefix[1], kVgprEncoding, 18, 9);
    append_words(words, prefix);

    const uint16_t delta = static_cast<uint16_t>(half * kHalfDwords);
    auto replacement = cdna5::build_vop3p(
        cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
        {.vdst = static_cast<uint8_t>((matrix.vdst + delta) & 0xffu),
         .neg_hi = static_cast<uint8_t>(matrix.neg_hi),
         .opsel = 4,
         .clamp = static_cast<uint8_t>(matrix.clamp),
         .src0 = static_cast<uint16_t>(kVgprEncoding + ((src0 + delta) & 0xffu)),
         .src1 = static_cast<uint16_t>(kVgprEncoding + src1),
         .src2 = src2_is_vgpr ? static_cast<uint16_t>(kVgprEncoding + ((src2 + delta) & 0xffu))
                              : static_cast<uint16_t>(matrix.src2),
         .neg = static_cast<uint8_t>(matrix.neg)});
    replacement[0] |= uint32_t{1} << 14; // matrix B: FP4
    append_words(words, replacement);
  }
  if (src0_crosses || dst_crosses || src2_crosses)
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  return ExpandResult::success(std::move(words));
}

/// @brief Whether the shared structural key denotes an implemented Scale16 expansion.
[[nodiscard]] bool scale16_residual(const Instruction &inst) {
  if (!inst.mnemonic().starts_with("v_wmma_scale16_f32_"))
    return false;
  if (inst.size() != 4 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr)
    return true;

  cdna5::Vop3pMachineInst scale{};
  cdna5::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (scale.op != kWmmaScale16PrefixOp ||
      (matrix.op != cdna5::kVWmmaF3216x16x128F8f6f4Vop3p &&
       matrix.op != cdna5::kVWmmaF3232x16x128F4Vop3p) ||
      gfx1250_floating_wmma_control_error(matrix, &scale) != nullptr) {
    return true;
  }

  constexpr uint16_t kVgprEncoding = 256;
  for (const uint16_t encoded_scale :
       {static_cast<uint16_t>(scale.src0), static_cast<uint16_t>(scale.src1)}) {
    if (encoded_scale < kVgprEncoding)
      continue;
    const uint16_t base = static_cast<uint16_t>(encoded_scale - kVgprEncoding);
    if ((base & 1u) != 0 || base > 254u)
      return true;
  }
  return matrix.op == cdna5::kVWmmaF3232x16x128F4Vop3p || scale.src2 != 0x100;
}

/// @brief Conservatively separate B0 integer WMMA from its A0 successor.
///
/// @details gfx1250 requires nine separating V_NOPs when dense IU8 WMMA feeds a
/// following XDL matrix input, and five for sparse IU8 SWMMAC. These bounds also
/// cover their shorter WMMA-to-VALU hazard windows. Count exact canonical V_NOPs
/// already following the instruction in the same basic block and append only
/// the missing slots. Limiting credit to the block guarantees that every
/// credited word remains adjacent after layout. Noncanonical NOPs and following
/// control-flow successors conservatively receive no credit.
///
/// The slots are VALU issue slots, so this shares the MODE separation rule's
/// filler reasoning: an ordinary VALU instruction is skipped when EXEC==0, and a
/// V_NOP still issues but occupies no slot there, which makes V_NOP the best
/// available filler rather than a guarantee. The exposure is narrower here than
/// at the MODE write, because the producer is itself EXEC-masked: a wholly
/// inactive shadow ran no WMMA and so has nothing to separate. What remains is
/// an EXEC clear between the WMMA and its shadow, which is the case the TODO
/// below has to rule out before it can count anything but a V_NOP.
[[nodiscard]] int required_iu8_spacing_slots(const Instruction &inst) {
  if (inst.mnemonic() == "v_wmma_i32_16x16x64_iu8")
    return 9;
  if (inst.mnemonic() == "v_swmmac_i32_16x16x128_iu8")
    return 5;
  return 0;
}

[[nodiscard]] int existing_iu8_spacing_slots(const Instruction &inst, int required_slots) {
  return count_adjacent_canonical_v_nops(inst, required_slots, &Instruction::next_instruction);
}

[[nodiscard]] bool iu8_spacing_residual(const Instruction &inst) {
  const int required_slots = required_iu8_spacing_slots(inst);
  if (required_slots == 0)
    return false;
  if (inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr)
    return true;
  return existing_iu8_spacing_slots(inst, required_slots) < required_slots;
}

ExpandResult expand_gfx1250_wmma_iu8_spacing(const Instruction &inst, uint32_t, uint64_t,
                                             std::span<const uint8_t>, const LivenessAnalysis &,
                                             TranslationContext &, const LaneLayout *,
                                             const LaneLayout *) {
  const int required_slots = required_iu8_spacing_slots(inst);
  if (required_slots == 0)
    return ExpandResult::not_handled();
  if (inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr)
    return ExpandResult::failed("gfx1250 IU8 WMMA rule received an unsupported VOP3P instruction");

  std::vector<uint32_t> words(inst.raw_encoding(),
                              inst.raw_encoding() + inst.size() / sizeof(uint32_t));
  const int existing_slots = existing_iu8_spacing_slots(inst, required_slots);

  // TODO: Replace canonical V_NOP counting with whole-kernel scheduling that
  // can also credit independent VALU in each reachable successor. Crediting real
  // VALU needs EXEC != 0 proved on the credited path first; without that proof
  // only V_NOP counts, because ordinary VALU is skipped under an empty mask.
  words.insert(words.end(), static_cast<size_t>(required_slots - existing_slots),
               cdna5::build_vop1(cdna5::kVNopVop1)[0]);
  return ExpandResult::success(std::move(words));
}

/// @brief True if @p opcode is a gfx1250 cluster-load form this rule covers
/// (both the plain and async-to-LDS families, all widths).
[[nodiscard]] bool is_gfx1250_cluster_load(uint16_t opcode) {
  switch (opcode) {
  case cdna5::kClusterLoadB32Vglobal:
  case cdna5::kClusterLoadB64Vglobal:
  case cdna5::kClusterLoadB128Vglobal:
  case cdna5::kClusterLoadAsyncToLdsB8Vglobal:
  case cdna5::kClusterLoadAsyncToLdsB32Vglobal:
  case cdna5::kClusterLoadAsyncToLdsB64Vglobal:
  case cdna5::kClusterLoadAsyncToLdsB128Vglobal:
    return true;
  default:
    return false;
  }
}

/// @brief Build the canonical M0 = 0 word emitted before a cluster load.
[[nodiscard]] constexpr uint32_t build_cluster_m0_clear() {
  return cdna5::build_sop1(cdna5::kSMovB32Sop1,
                           {.ssrc0 = kGfx1250InlineZero, .sdst = kGfx1250M0})[0];
}

[[nodiscard]] bool cluster_load_residual(const Instruction &inst) {
  if (!is_gfx1250_cluster_load(inst.opcode()))
    return false;
  if (inst.size() != 3 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr)
    return true;
  return !has_canonical_predecessor(inst, build_cluster_m0_clear());
}

/// @brief Rewrite a gfx1250 cluster load to run with M0 = 0.
///
/// @details Every cluster-load form (both SADDR and off/NULL-saddr, all widths)
/// is left as a cluster load and wrapped so it executes with M0 forced to zero:
/// save M0 to a scratch SGPR, set M0 = 0, run the load, then restore M0. The opcode
/// is not changed.
///
/// Under full SGPR pressure, EXEC-independent lane-zero operations carry one
/// live SGPR through a dead low-bank VGPR. This keeps the guest cluster load
/// itself unconditional and therefore preserves its wait-counter contribution.
///
/// A second translation preserves the load when its immediately preceding
/// decoded instruction in the same basic block is the canonical M0 clear. This
/// proves that no control-flow edge can bypass the clear. The clear instruction
/// currently has no B0-to-A0 semantic rule and is copied unchanged; this reuse
/// condition must be revisited if such a rule is added.
ExpandResult expand_gfx1250_cluster_load(const Instruction &inst, uint32_t, uint64_t,
                                         std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                         TranslationContext &context, const LaneLayout *,
                                         const LaneLayout *) {
  if (!is_gfx1250_cluster_load(inst.opcode()) ||
      inst.size() != 3 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 cluster-load rule received an unsupported instruction");
  }

  if (!cluster_load_residual(inst)) {
    return ExpandResult::success(
        std::vector<uint32_t>(inst.raw_encoding(), inst.raw_encoding() + 3));
  }

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256,
                            .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
  Gfx1250SgprScratchRequest scratch_request;
  scratch_request.count = 1;
  scratch_request.forbidden = gfx1250_instruction_registers(inst);
  scratch_request.carrier_mode = Gfx1250SgprCarrierMode::LaneZero;
  const auto scratch = acquire_gfx1250_sgprs(inst, liveness, context, &allocator, scratch_request);
  if (!scratch || scratch->base > 105) {
    return ExpandResult::failed(
        "gfx1250 cluster load could not allocate scalar scratch for M0 preservation");
  }

  // Save M0 to scratch, set M0 = 0, run the load, then restore M0. A binary
  // translator cannot prove M0 is dead after the load, so it saves and restores
  // the original value around it.
  //
  // Inline constant 0 encodes as 128 in a scalar source. Every M0 reference here
  // MUST use kGfx1250M0 (125): on gfx1250 M0 encodes as 125 and NULL as 124 (the
  // inverse of CDNA), so a write to 124 would be a discarded NULL write.
  std::vector<uint32_t> words;
  words.reserve(18);
  append_gfx1250_scratch_dependency_barrier(words);
  uint8_t current_mode = 0;
  std::optional<uint8_t> original_mode;
  if (scratch->has_carrier()) {
    original_mode = gfx1250_vgpr_mode_before(inst, liveness);
    if (!original_mode)
      return ExpandResult::failed(
          "gfx1250 cluster-load SGPR carrier cannot prove the VGPR-MSB mode");
    current_mode = *original_mode;
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_sgpr_preservation(words, *scratch, false))
      return ExpandResult::failed("gfx1250 cluster load could not preserve scalar scratch");
    append_gfx1250_vgpr_msb_transition(words, current_mode, *original_mode);
  }
  append_words(
      words, cdna5::build_sop1(cdna5::kSMovB32Sop1,
                               {.ssrc0 = kGfx1250M0, .sdst = static_cast<uint8_t>(scratch->base)}));
  words.push_back(build_cluster_m0_clear());
  words.insert(words.end(), inst.raw_encoding(), inst.raw_encoding() + 3);
  append_words(
      words, cdna5::build_sop1(cdna5::kSMovB32Sop1,
                               {.ssrc0 = static_cast<uint8_t>(scratch->base), .sdst = kGfx1250M0}));
  if (scratch->has_carrier()) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_sgpr_preservation(words, *scratch, true))
      return ExpandResult::failed("gfx1250 cluster load could not restore scalar scratch");
    append_gfx1250_vgpr_msb_transition(words, current_mode, *original_mode);
  }
  return ExpandResult::success(std::move(words));
}

/// @brief Materialize the B0 DS ADDTID address and issue an ordinary A0 DS op.
///
/// @details ds_*_addtid_b32 computes its LDS byte address on-chip as
/// `(M0 + tid*4) & 0xfffff`, where `tid` is the wave-local thread id and M0
/// carries the LDS base. A0 does not honor that addressing for these opcodes, so
/// the address is materialized explicitly and an ordinary ds_load_b32/ds_store_b32
/// is issued against it. The emitted sequence reproduces the formula term by term:
///   1. v_mbcnt_lo/hi_u32_b32 with mask -1 -> tid (population count of lanes below).
///   2. v_lshlrev_b32 by 2 -> tid*4.
///   3. v_add_nc_u32 with SRC0 = M0 (gfx1250 M0 encodes as 125) -> M0 + tid*4.
///   4. v_bfe_u32 offset 0 width 20 -> (M0 + tid*4) & 0xfffff.
/// This is the A0-stepping contract, not the revision-neutral simulator's
/// documented ADDTID model; the test pins the emitted operands rather than
/// executing against that model, which would test the B0 contract instead.
ExpandResult expand_gfx1250_ds_addtid(const Instruction &inst, uint32_t, uint64_t,
                                      std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                      TranslationContext &context, const LaneLayout *,
                                      const LaneLayout *) {
  const bool is_load = inst.opcode() == cdna5::kDsLoadAddtidB32Vds;
  const bool is_store = inst.opcode() == cdna5::kDsStoreAddtidB32Vds;
  if ((!is_load && !is_store) || inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 ADDTID rule received an unsupported instruction");
  }

  cdna5::VdsMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 ADDTID lowering cannot prove the VGPR-MSB mode");
  }
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));

  uint16_t temp = source.vdst;
  std::optional<SemanticScratchLease> store_scratch;
  if (is_store) {
    SemanticScratchAllocator allocator(
        inst, liveness, context,
        SemanticScratchPolicy{.max_vgprs = 256,
                              .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
    SemanticScratchRequest request;
    request.count = 1;
    request.forbidden = gfx1250_instruction_registers(inst);
    request.allow_spill = true;
    const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
    if (!scratch) {
      if (scratch.failure == SemanticScratchFailure::DynamicStackUnsupported) {
        return ExpandResult::failed(
            "gfx1250 DS store ADDTID cannot use private-memory spills in a dynamic-stack kernel");
      }
      return ExpandResult::failed("gfx1250 DS store ADDTID could not allocate low-bank scratch");
    }
    store_scratch = *scratch.lease;
    temp = store_scratch->base;
  }
  const uint8_t compute_bank = is_load ? *dst_bank : 0;
  const uint8_t compute_mode = static_cast<uint8_t>(compute_bank | (compute_bank << 2) |
                                                    (compute_bank << 4) | (compute_bank << 6));

  std::vector<uint32_t> words;
  words.reserve(26);
  append_gfx1250_scratch_dependency_barrier(words);
  uint8_t current_mode = original_mode;
  append_gfx1250_vgpr_msb_transition(words, current_mode, compute_mode);
  if (store_scratch && store_scratch->spilled &&
      !append_gfx1250_scratch_preservation(words, *store_scratch, false)) {
    return ExpandResult::failed("gfx1250 DS store ADDTID could not preserve low-bank scratch");
  }
  append_words(words,
               cdna5::build_vop3(cdna5::kVMbcntLoU32B32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                              .src0 = 193, // inline -1
                                                              .src1 = gfx1250_inline_u32(0)}));
  append_words(words,
               cdna5::build_vop3(cdna5::kVMbcntHiU32B32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                              .src0 = 193,
                                                              .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               cdna5::build_vop3(cdna5::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                           .src0 = gfx1250_inline_u32(2),
                                                           .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words, cdna5::build_vop3(cdna5::kVAddNcU32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                                .src0 = kGfx1250M0,
                                                                .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words, cdna5::build_vop3(cdna5::kVBfeU32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                              .src0 = gfx1250_vgpr_src(temp),
                                                              .src1 = gfx1250_inline_u32(0),
                                                              .src2 = gfx1250_inline_u32(20)}));

  if (is_store) {
    // The emitted ds_store_b32 keeps the original store-data VGPR in data0, and
    // data0 is a Src1-role operand in both ds_store_addtid_b32 and ds_store_b32,
    // so its high bank is src1_bank. The address VGPR is a fresh low-bank
    // scratch, so only the Src1 field needs the original store-data bank.
    const uint8_t ds_mode = static_cast<uint8_t>(*src1_bank << 2);
    append_gfx1250_vgpr_msb_transition(words, current_mode, ds_mode);
    append_words(words, cdna5::build_vds(cdna5::kDsStoreB32Vds,
                                         {.offset0 = static_cast<uint8_t>(source.offset0),
                                          .offset1 = static_cast<uint8_t>(source.offset1),
                                          .addr = static_cast<uint8_t>(temp),
                                          .data0 = static_cast<uint8_t>(source.data0)}));
    if (store_scratch && store_scratch->spilled) {
      append_words(words, cdna5::build_sopp(cdna5::kSWaitDscntSopp, {.simm16 = 0}));
      append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
      if (!append_gfx1250_scratch_preservation(words, *store_scratch, true)) {
        return ExpandResult::failed("gfx1250 DS store ADDTID could not restore low-bank scratch");
      }
    }
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  } else {
    append_words(words, cdna5::build_vds(cdna5::kDsLoadB32Vds,
                                         {.offset0 = static_cast<uint8_t>(source.offset0),
                                          .offset1 = static_cast<uint8_t>(source.offset1),
                                          .addr = static_cast<uint8_t>(temp),
                                          .vdst = static_cast<uint8_t>(temp)}));
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  }
  return ExpandResult::success(std::move(words));
}

/// @brief Emulate one RNE F32-to-UE5M3 conversion into a low byte.
void append_gfx1250_f32_to_e5m3(std::vector<uint32_t> &words, uint16_t source, uint8_t source_bank,
                                uint16_t out, uint16_t temp, uint16_t top_byte, uint16_t nan_mask,
                                uint16_t subnormal_mask, uint16_t overflow_mask, uint16_t fp16_ovfl,
                                uint8_t &current_mode) {
  const auto append_literal = [&](uint16_t opcode, cdna5::Vop3BuilderFields fields,
                                  uint32_t literal) {
    append_words(words, cdna5::build_vop3(opcode, fields));
    words.push_back(literal);
  };
  const auto append_compare_literal = [&](uint16_t opcode, uint16_t mask, uint16_t src1,
                                          uint32_t literal) {
    append_words(
        words,
        cdna5::build_vop3(opcode, {.vdst = static_cast<uint8_t>(mask), .src0 = 255, .src1 = src1}));
    words.push_back(literal);
  };
  const uint8_t source_mode = source >= 256u ? static_cast<uint8_t>(source_bank << 2) : 0;

  // E5M3 ignores the source sign bit, so convert the magnitude. Classify NaNs
  // after clearing the sign so both NaN signs use the terminal encoding.
  append_gfx1250_vgpr_msb_transition(words, current_mode, source_mode);
  append_literal(cdna5::kVAndB32Vop3,
                 {.vdst = static_cast<uint8_t>(out), .src0 = 255, .src1 = source}, 0x7fffffffu);
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
  append_compare_literal(cdna5::kVCmpLtU32Vop3, nan_mask, gfx1250_vgpr_src(out), 0x7f800000u);
  append_compare_literal(cdna5::kVCmpGtU32Vop3, subnormal_mask, gfx1250_vgpr_src(out), 0x38800000u);

  // For normal E5M3 values, round the F32 bit pattern at bit 20 and then
  // remove the exponent-bias delta: (127 - 15) * 8 == 0x380.
  append_words(words,
               cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                           .src0 = gfx1250_inline_u32(20),
                                                           .src1 = gfx1250_vgpr_src(out)}));
  append_literal(cdna5::kVAndB32Vop3,
                 {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = gfx1250_vgpr_src(temp)},
                 1u);
  append_literal(
      cdna5::kVAddNcU32Vop3,
      {.vdst = static_cast<uint8_t>(top_byte), .src0 = 255, .src1 = gfx1250_vgpr_src(out)},
      0x7ffffu);
  append_words(words,
               cdna5::build_vop3(cdna5::kVAddNcU32Vop3, {.vdst = static_cast<uint8_t>(top_byte),
                                                         .src0 = gfx1250_vgpr_src(top_byte),
                                                         .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(top_byte),
                                                           .src0 = gfx1250_inline_u32(20),
                                                           .src1 = gfx1250_vgpr_src(top_byte)}));
  append_literal(
      cdna5::kVSubNcU32Vop3,
      {.vdst = static_cast<uint8_t>(top_byte), .src0 = gfx1250_vgpr_src(top_byte), .src1 = 255},
      0x380u);

  // E5M3 subnormals have a constant 2^-17 quantum. Scaling by 2^17 is exact
  // in F32, so a direct nearest-integer conversion implements RNE without an
  // intermediate-format rounding step.
  append_literal(cdna5::kVMulF32Vop3,
                 {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = gfx1250_vgpr_src(out)},
                 0x48000000u);
  append_words(words,
               cdna5::build_vop3(cdna5::kVCvtNearestI32F32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                                 .src0 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                           .src0 = gfx1250_vgpr_src(top_byte),
                                                           .src1 = gfx1250_vgpr_src(temp),
                                                           .src2 = subnormal_mask}));

  append_compare_literal(cdna5::kVCmpLtU32Vop3, overflow_mask, gfx1250_vgpr_src(out), 0xfeu);
  // MODE.FP16_OVFL is either zero or one. XOR maps it branchlessly to the
  // required terminal encoding: mode 0 -> 0xff, mode 1 -> 0xfe.
  append_literal(cdna5::kVXorB32Vop3,
                 {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = fp16_ovfl}, 0xffu);
  append_words(words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(out),
                                                                  .src1 = gfx1250_vgpr_src(temp),
                                                                  .src2 = overflow_mask}));
  append_literal(cdna5::kVMovB32Vop3, {.vdst = static_cast<uint8_t>(temp), .src0 = 255}, 0xffu);
  append_words(words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(out),
                                                                  .src1 = gfx1250_vgpr_src(temp),
                                                                  .src2 = nan_mask}));
}

/// @brief Emulate B0 CLAMP=1 packed F32-to-UE5M3 conversion on A0.
ExpandResult expand_gfx1250_cvt_pk_fp8_f32_e5m3(const Instruction &inst, uint32_t, uint64_t,
                                                std::span<const uint8_t>,
                                                const LivenessAnalysis &liveness,
                                                TranslationContext &context, const LaneLayout *,
                                                const LaneLayout *) {
  if (inst.mnemonic() != "v_cvt_pk_fp8_f32" || inst.raw_encoding() == nullptr ||
      inst.size() < static_cast<int>(sizeof(cdna5::Vop3MachineInst))) {
    return ExpandResult::failed("gfx1250 E5M3 pack rule received an unsupported instruction");
  }
  cdna5::Vop3MachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  if (source.clamp == 0)
    return ExpandResult::not_handled();
  // ABS, NEG, and OMOD are unsupported for this conversion and do not affect
  // its result, so their encoded values are intentionally ignored.
  if (source.src0 == 233u || source.src0 == 234u || source.src0 == 250u)
    return ExpandResult::failed("gfx1250 E5M3 pack does not support DPP");
  // SRC_LITERAL64 carries a two-word payload, and the expansion has no free
  // literal slot to re-encode it: the generated helpers already spend selector
  // 255 on their own mask literals.
  if (source.src0 == 254u || source.src1 == 254u)
    return ExpandResult::failed("gfx1250 E5M3 pack does not support SRC_LITERAL64");
  const bool has_literal = source.src0 == 255u || source.src1 == 255u;
  if (has_literal && inst.size() < 3 * static_cast<int>(sizeof(uint32_t)))
    return ExpandResult::failed("gfx1250 E5M3 pack literal word is missing");
  uint32_t literal = 0;
  if (has_literal)
    std::memcpy(&literal, inst.raw_encoding() + 2, sizeof(literal));

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256,
                            .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
  SemanticScratchRequest request;
  request.count = 4;
  request.forbidden = gfx1250_instruction_registers(inst);
  request.allow_spill = true;
  const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
  if (!scratch) {
    if (scratch.failure == SemanticScratchFailure::DynamicStackUnsupported) {
      return ExpandResult::failed(
          "gfx1250 E5M3 pack cannot use private-memory spills in a dynamic-stack kernel");
    }
    return ExpandResult::failed("gfx1250 E5M3 pack could not allocate four scratch VGPRs");
  }
  const uint16_t out0 = scratch.lease->base;
  const uint16_t out1 = static_cast<uint16_t>(out0 + 1u);
  const uint16_t temp = static_cast<uint16_t>(out0 + 2u);
  const uint16_t top_byte = static_cast<uint16_t>(out0 + 3u);

  Gfx1250SgprScratchRequest mask_request;
  mask_request.count = 4;
  mask_request.forbidden = request.forbidden;
  mask_request.forbidden.expand(scratch.lease->registers());
  mask_request.carrier_mode = Gfx1250SgprCarrierMode::ExecMasked;
  const auto masks = acquire_gfx1250_sgprs(inst, liveness, context, &allocator, mask_request);
  if (!masks)
    return ExpandResult::failed("gfx1250 E5M3 pack could not allocate four scratch SGPRs");

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank)
    return ExpandResult::failed("gfx1250 E5M3 pack cannot prove the VGPR-MSB mode");
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));

  std::vector<uint32_t> words;
  words.reserve(160);
  append_gfx1250_scratch_dependency_barrier(words);
  uint8_t current_mode = original_mode;
  if (scratch.lease->spilled || masks->has_carrier()) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_scratch_preservation(words, *scratch.lease, false) ||
        !append_gfx1250_sgpr_preservation(words, *masks, false)) {
      return ExpandResult::failed("gfx1250 E5M3 pack could not preserve scratch registers");
    }
  }
  const uint16_t fp16_ovfl = static_cast<uint16_t>(masks->base + 3u);
  append_words(
      words, cdna5::build_sopk(cdna5::kSGetregB32Sopk, {.simm16 = kGfx1250ModeFp16OvflHwreg,
                                                        .sdst = static_cast<uint8_t>(fp16_ovfl)}));

  uint16_t effective_src0 = source.src0;
  // The bank travels with the operand, not with the instruction. Once a literal
  // has been materialized into out0 the effective operand is that scratch VGPR,
  // which is allocated in the low bank -- forwarding the guest operand's bank
  // would make the helper address a different physical register under a nonzero
  // VGPR-MSB mode.
  uint8_t effective_src0_bank = *src0_bank;
  if (source.src0 == 255u) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    append_words(words, cdna5::build_vop3(cdna5::kVMovB32Vop3,
                                          {.vdst = static_cast<uint8_t>(out0), .src0 = 255}));
    words.push_back(literal);
    effective_src0 = gfx1250_vgpr_src(out0);
    effective_src0_bank = 0;
  }
  append_gfx1250_f32_to_e5m3(words, effective_src0, effective_src0_bank, out0, temp, top_byte,
                             masks->base, static_cast<uint16_t>(masks->base + 1u),
                             static_cast<uint16_t>(masks->base + 2u), fp16_ovfl, current_mode);
  uint16_t effective_src1 = source.src1;
  uint8_t effective_src1_bank = *src1_bank;
  if (source.src1 == 255u) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    append_words(words, cdna5::build_vop3(cdna5::kVMovB32Vop3,
                                          {.vdst = static_cast<uint8_t>(out1), .src0 = 255}));
    words.push_back(literal);
    effective_src1 = gfx1250_vgpr_src(out1);
    effective_src1_bank = 0;
  }
  append_gfx1250_f32_to_e5m3(words, effective_src1, effective_src1_bank, out1, temp, top_byte,
                             masks->base, static_cast<uint16_t>(masks->base + 1u),
                             static_cast<uint16_t>(masks->base + 2u), fp16_ovfl, current_mode);
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
  append_words(words, cdna5::build_vop3(cdna5::kVLshlOrB32Vop3, {.vdst = static_cast<uint8_t>(out0),
                                                                 .src0 = gfx1250_vgpr_src(out1),
                                                                 .src1 = gfx1250_inline_u32(8),
                                                                 .src2 = gfx1250_vgpr_src(out0)}));
  const bool write_high = (source.opsel & 8u) != 0;
  if (write_high) {
    append_words(words,
                 cdna5::build_vop3(cdna5::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(out0),
                                                             .src0 = gfx1250_inline_u32(16),
                                                             .src1 = gfx1250_vgpr_src(out0)}));
  }
  const uint8_t merge_mode = static_cast<uint8_t>((*dst_bank << 4) | (*dst_bank << 6));
  append_gfx1250_vgpr_msb_transition(words, current_mode, merge_mode);
  append_words(words,
               cdna5::build_vop3(cdna5::kVBfiB32Vop3, {.vdst = static_cast<uint8_t>(source.vdst),
                                                       .src0 = 255,
                                                       .src1 = gfx1250_vgpr_src(out0),
                                                       .src2 = gfx1250_vgpr_src(source.vdst)}));
  words.push_back(write_high ? 0xffff0000u : 0x0000ffffu);

  if (scratch.lease->spilled || masks->has_carrier()) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_sgpr_preservation(words, *masks, true) ||
        !append_gfx1250_scratch_preservation(words, *scratch.lease, true)) {
      return ExpandResult::failed("gfx1250 E5M3 pack could not restore scratch registers");
    }
  }
  append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  if (!prepend_gfx1250_execz_guard_for_masked_replacement(words, masks->has_carrier()))
    return ExpandResult::failed("gfx1250 E5M3 pack SGPR-carrier guard is too large");
  return ExpandResult::success(std::move(words));
}

/// @brief Emulate B0 CLAMP=1 stochastic F32-to-UE5M3 conversion on A0.
ExpandResult expand_gfx1250_cvt_sr_fp8_f32_e5m3(const Instruction &inst, uint32_t, uint64_t,
                                                std::span<const uint8_t>,
                                                const LivenessAnalysis &liveness,
                                                TranslationContext &context, const LaneLayout *,
                                                const LaneLayout *) {
  if (inst.mnemonic() != "v_cvt_sr_fp8_f32" || inst.raw_encoding() == nullptr ||
      inst.size() < static_cast<int>(sizeof(cdna5::Vop3MachineInst))) {
    return ExpandResult::failed("gfx1250 stochastic E5M3 rule received an unsupported instruction");
  }
  cdna5::Vop3MachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  if (source.clamp == 0)
    return ExpandResult::not_handled();
  // ABS, NEG, and OMOD are unsupported for this conversion and do not affect
  // its result, so their encoded values are intentionally ignored.
  if (source.src0 == 233u || source.src0 == 234u || source.src0 == 250u)
    return ExpandResult::failed("gfx1250 stochastic E5M3 does not support DPP");
  if (source.src0 == 254u || source.src1 == 254u)
    return ExpandResult::failed("gfx1250 stochastic E5M3 does not support SRC_LITERAL64");
  const bool has_literal = source.src0 == 255u || source.src1 == 255u;
  if (has_literal && inst.size() < 3 * static_cast<int>(sizeof(uint32_t)))
    return ExpandResult::failed("gfx1250 stochastic E5M3 literal word is missing");
  uint32_t literal = 0;
  if (has_literal)
    std::memcpy(&literal, inst.raw_encoding() + 2, sizeof(literal));

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256,
                            .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
  SemanticScratchRequest request;
  request.count = 5;
  request.forbidden = gfx1250_instruction_registers(inst);
  request.allow_spill = true;
  const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
  if (!scratch) {
    if (scratch.failure == SemanticScratchFailure::DynamicStackUnsupported) {
      return ExpandResult::failed(
          "gfx1250 stochastic E5M3 cannot use private-memory spills in a dynamic-stack kernel");
    }
    return ExpandResult::failed("gfx1250 stochastic E5M3 could not allocate five scratch VGPRs");
  }
  const uint16_t out = scratch.lease->base;
  const uint16_t temp = static_cast<uint16_t>(out + 1u);
  const uint16_t normal = static_cast<uint16_t>(out + 2u);
  const uint16_t aux = static_cast<uint16_t>(out + 3u);
  const uint16_t shift = static_cast<uint16_t>(out + 4u);

  Gfx1250SgprScratchRequest mask_request;
  mask_request.count = 5;
  mask_request.forbidden = request.forbidden;
  mask_request.forbidden.expand(scratch.lease->registers());
  mask_request.carrier_mode = Gfx1250SgprCarrierMode::ExecMasked;
  const auto masks = acquire_gfx1250_sgprs(inst, liveness, context, &allocator, mask_request);
  if (!masks)
    return ExpandResult::failed("gfx1250 stochastic E5M3 could not allocate five scratch SGPRs");
  const uint16_t nan_mask = masks->base;
  const uint16_t subnormal_mask = static_cast<uint16_t>(masks->base + 1u);
  const uint16_t overflow_mask = static_cast<uint16_t>(masks->base + 2u);
  const uint16_t tiny_mask = static_cast<uint16_t>(masks->base + 3u);
  const uint16_t fp16_ovfl = static_cast<uint16_t>(masks->base + 4u);

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank)
    return ExpandResult::failed("gfx1250 stochastic E5M3 cannot prove the VGPR-MSB mode");
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));

  const auto append_literal = [](std::vector<uint32_t> &output, uint16_t opcode,
                                 cdna5::Vop3BuilderFields fields, uint32_t value) {
    append_words(output, cdna5::build_vop3(opcode, fields));
    output.push_back(value);
  };
  const auto append_compare_literal = [](std::vector<uint32_t> &output, uint16_t opcode,
                                         uint16_t mask, uint16_t src1, uint32_t value) {
    append_words(
        output,
        cdna5::build_vop3(opcode, {.vdst = static_cast<uint8_t>(mask), .src0 = 255, .src1 = src1}));
    output.push_back(value);
  };

  std::vector<uint32_t> words;
  words.reserve(96);
  append_gfx1250_scratch_dependency_barrier(words);
  uint8_t current_mode = original_mode;
  if (scratch.lease->spilled || masks->has_carrier()) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_scratch_preservation(words, *scratch.lease, false) ||
        !append_gfx1250_sgpr_preservation(words, *masks, false)) {
      return ExpandResult::failed("gfx1250 stochastic E5M3 could not preserve scratch registers");
    }
  }
  append_words(
      words, cdna5::build_sopk(cdna5::kSGetregB32Sopk, {.simm16 = kGfx1250ModeFp16OvflHwreg,
                                                        .sdst = static_cast<uint8_t>(fp16_ovfl)}));

  uint16_t value_source = source.src0;
  if (source.src0 == 255u) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    append_literal(words, cdna5::kVMovB32Vop3, {.vdst = static_cast<uint8_t>(out), .src0 = 255},
                   literal);
    value_source = gfx1250_vgpr_src(out);
  }
  uint16_t noise_source = source.src1;
  const uint8_t value_mode =
      value_source >= 256u && source.src0 != 255u ? static_cast<uint8_t>(*src0_bank << 2) : 0;
  append_gfx1250_vgpr_msb_transition(words, current_mode, value_mode);
  append_literal(words, cdna5::kVAndB32Vop3,
                 {.vdst = static_cast<uint8_t>(out), .src0 = 255, .src1 = value_source},
                 0x7fffffffu);
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
  append_compare_literal(words, cdna5::kVCmpLtU32Vop3, nan_mask, gfx1250_vgpr_src(out),
                         0x7f800000u);
  append_compare_literal(words, cdna5::kVCmpGtU32Vop3, subnormal_mask, gfx1250_vgpr_src(out),
                         0x38800000u);
  append_compare_literal(words, cdna5::kVCmpGtU32Vop3, tiny_mask, gfx1250_vgpr_src(out),
                         0x36800000u);

  const uint8_t noise_mode =
      noise_source >= 256u && source.src1 != 255u ? static_cast<uint8_t>(*src1_bank << 2) : 0;
  append_gfx1250_vgpr_msb_transition(words, current_mode, noise_mode);
  if (source.src1 == 255u) {
    append_literal(
        words, cdna5::kVLshrrevB32Vop3,
        {.vdst = static_cast<uint8_t>(temp), .src0 = gfx1250_inline_u32(12), .src1 = 255}, literal);
  } else {
    append_words(words,
                 cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                             .src0 = gfx1250_inline_u32(12),
                                                             .src1 = noise_source}));
  }
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
  append_words(words,
               cdna5::build_vop3(cdna5::kVAddNcU32Vop3, {.vdst = static_cast<uint8_t>(normal),
                                                         .src0 = gfx1250_vgpr_src(out),
                                                         .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(normal),
                                                           .src0 = gfx1250_inline_u32(20),
                                                           .src1 = gfx1250_vgpr_src(normal)}));
  append_literal(
      words, cdna5::kVSubNcU32Vop3,
      {.vdst = static_cast<uint8_t>(normal), .src0 = gfx1250_vgpr_src(normal), .src1 = 255},
      0x380u);
  // Subnormal stochastic rounding follows the execution model exactly. The
  // discarded significand and the high `shift` seed bits are added as
  // integers; shifting their sum yields the stochastic carry.
  append_literal(words, cdna5::kVAndB32Vop3,
                 {.vdst = static_cast<uint8_t>(aux), .src0 = 255, .src1 = gfx1250_vgpr_src(out)},
                 0x7fffffu);
  append_literal(words, cdna5::kVOrB32Vop3,
                 {.vdst = static_cast<uint8_t>(aux), .src0 = 255, .src1 = gfx1250_vgpr_src(aux)},
                 0x800000u);
  append_words(words,
               cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(shift),
                                                           .src0 = gfx1250_inline_u32(23),
                                                           .src1 = gfx1250_vgpr_src(out)}));
  append_literal(
      words, cdna5::kVSubNcU32Vop3,
      {.vdst = static_cast<uint8_t>(shift), .src0 = 255, .src1 = gfx1250_vgpr_src(shift)}, 133u);
  append_words(words, cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(shift),
                                                                  .src1 = gfx1250_vgpr_src(aux)}));
  append_words(words, cdna5::build_vop3(cdna5::kVBfeU32Vop3, {.vdst = static_cast<uint8_t>(aux),
                                                              .src0 = gfx1250_vgpr_src(aux),
                                                              .src1 = gfx1250_inline_u32(0),
                                                              .src2 = gfx1250_vgpr_src(shift)}));
  append_literal(words, cdna5::kVSubNcU32Vop3,
                 {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = gfx1250_vgpr_src(shift)},
                 32u);
  append_gfx1250_vgpr_msb_transition(words, current_mode, noise_mode);
  if (source.src1 == 255u) {
    append_literal(
        words, cdna5::kVLshrrevB32Vop3,
        {.vdst = static_cast<uint8_t>(temp), .src0 = gfx1250_vgpr_src(temp), .src1 = 255}, literal);
  } else {
    append_words(words,
                 cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                             .src0 = gfx1250_vgpr_src(temp),
                                                             .src1 = noise_source}));
  }
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
  append_words(words, cdna5::build_vop3(cdna5::kVAddNcU32Vop3, {.vdst = static_cast<uint8_t>(aux),
                                                                .src0 = gfx1250_vgpr_src(aux),
                                                                .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words, cdna5::build_vop3(cdna5::kVLshrrevB32Vop3, {.vdst = static_cast<uint8_t>(aux),
                                                                  .src0 = gfx1250_vgpr_src(shift),
                                                                  .src1 = gfx1250_vgpr_src(aux)}));
  append_words(words, cdna5::build_vop3(cdna5::kVAddNcU32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                .src0 = gfx1250_vgpr_src(out),
                                                                .src1 = gfx1250_vgpr_src(aux)}));
  append_words(words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(out),
                                                                  .src1 = gfx1250_inline_u32(0),
                                                                  .src2 = tiny_mask}));
  append_words(words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(normal),
                                                                  .src1 = gfx1250_vgpr_src(out),
                                                                  .src2 = subnormal_mask}));
  append_compare_literal(words, cdna5::kVCmpLtU32Vop3, overflow_mask, gfx1250_vgpr_src(out), 0xfeu);
  append_literal(words, cdna5::kVXorB32Vop3,
                 {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = fp16_ovfl}, 0xffu);
  append_words(words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(out),
                                                                  .src1 = gfx1250_vgpr_src(temp),
                                                                  .src2 = overflow_mask}));
  append_literal(words, cdna5::kVMovB32Vop3, {.vdst = static_cast<uint8_t>(temp), .src0 = 255},
                 0xffu);
  append_words(words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(out),
                                                                  .src1 = gfx1250_vgpr_src(temp),
                                                                  .src2 = nan_mask}));
  const uint8_t byte_sel = static_cast<uint8_t>((source.opsel >> 2u) & 3u);
  if (byte_sel != 0) {
    append_words(words, cdna5::build_vop3(cdna5::kVLshlrevB32Vop3,
                                          {.vdst = static_cast<uint8_t>(out),
                                           .src0 = gfx1250_inline_u32(byte_sel * 8u),
                                           .src1 = gfx1250_vgpr_src(out)}));
  }
  constexpr std::array<uint32_t, 4> kByteMasks = {0x000000ffu, 0x0000ff00u, 0x00ff0000u,
                                                  0xff000000u};
  const uint8_t merge_mode = static_cast<uint8_t>((*dst_bank << 4) | (*dst_bank << 6));
  append_gfx1250_vgpr_msb_transition(words, current_mode, merge_mode);
  append_literal(words, cdna5::kVBfiB32Vop3,
                 {.vdst = static_cast<uint8_t>(source.vdst),
                  .src0 = 255,
                  .src1 = gfx1250_vgpr_src(out),
                  .src2 = gfx1250_vgpr_src(source.vdst)},
                 kByteMasks[byte_sel]);

  if (scratch.lease->spilled || masks->has_carrier()) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_sgpr_preservation(words, *masks, true) ||
        !append_gfx1250_scratch_preservation(words, *scratch.lease, true)) {
      return ExpandResult::failed("gfx1250 stochastic E5M3 could not restore scratch registers");
    }
  }
  append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  if (!prepend_gfx1250_execz_guard_for_masked_replacement(words, masks->has_carrier()))
    return ExpandResult::failed("gfx1250 stochastic E5M3 SGPR-carrier guard is too large");
  return ExpandResult::success(std::move(words));
}

/// @brief Emulate B0 CLAMP=1 UE5M3 unpack on A0.
ExpandResult expand_gfx1250_cvt_f32_fp8_e5m3(const Instruction &inst, uint32_t, uint64_t,
                                             std::span<const uint8_t>,
                                             const LivenessAnalysis &liveness,
                                             TranslationContext &context, const LaneLayout *,
                                             const LaneLayout *) {
  if (!inst.mnemonic().starts_with("v_cvt_f32_fp8") ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 E5M3 unpack rule received an unsupported instruction");
  }
  cdna5::Vop3MachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint16_t kVgprEncoding = 256;
  if (!cvt_f32_fp8_e5m3_residual(inst))
    return ExpandResult::not_handled();
  // ABS, NEG, and OMOD are unsupported for this conversion and do not affect
  // its result, so their encoded values are intentionally ignored.
  if (source.src0 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 E5M3 unpack source is not a VGPR");
  }

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256,
                            .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
  SemanticScratchRequest request;
  request.count = 2;
  request.forbidden = gfx1250_instruction_registers(inst);
  request.allow_spill = true;
  const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
  if (!scratch) {
    if (scratch.failure == SemanticScratchFailure::DynamicStackUnsupported) {
      return ExpandResult::failed(
          "gfx1250 E5M3 unpack cannot use private-memory spills in a dynamic-stack kernel");
    }
    return ExpandResult::failed("gfx1250 E5M3 unpack could not allocate two scratch VGPRs");
  }
  const uint16_t out = scratch.lease->base;
  const uint16_t temp = static_cast<uint16_t>(out + 1u);

  Gfx1250SgprScratchRequest mask_request;
  mask_request.count = 2;
  mask_request.forbidden = request.forbidden;
  mask_request.forbidden.expand(scratch.lease->registers());
  // The conversion, both generated compares, and scratch memory are inactive
  // under EXEC=0. The compare masks and their carrier save/restore are skipped
  // as one region, so the borrowed SGPR window remains untouched.
  mask_request.carrier_mode = Gfx1250SgprCarrierMode::ExecMasked;
  const auto masks = acquire_gfx1250_sgprs(inst, liveness, context, &allocator, mask_request);
  if (!masks) {
    return ExpandResult::failed("gfx1250 E5M3 unpack could not allocate two SGPR masks");
  }
  const uint16_t nan_mask = masks->base;
  const uint16_t exp31_mask = static_cast<uint16_t>(masks->base + 1u);

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 E5M3 unpack cannot prove the VGPR-MSB mode");
  }
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
  const uint8_t extract_mode = *src0_bank;

  const auto append_vop3_literal = [](std::vector<uint32_t> &words, uint16_t opcode,
                                      cdna5::Vop3BuilderFields fields, uint32_t literal) {
    append_words(words, cdna5::build_vop3(opcode, fields));
    words.push_back(literal);
  };
  const auto append_compare_literal = [](std::vector<uint32_t> &words, uint16_t opcode,
                                         uint8_t sdst, uint16_t src1, uint32_t literal) {
    // gfx1250 VOP3 compares encode their scalar mask destination in the ordinary
    // VOP3 vdst field. Vop3SdstEnc is a different format whose sdst bits overlap
    // modifiers here; using it leaves vdst=0 and corrupts live s0.
    append_words(words, cdna5::build_vop3(opcode, {.vdst = sdst, .src0 = 255, .src1 = src1}));
    words.push_back(literal);
  };

  // The VOP3 encoding swaps the two byte-select bits for this opcode.
  const uint8_t byte_sel =
      static_cast<uint8_t>(((source.opsel & 1u) << 1u) | ((source.opsel & 2u) >> 1u));
  std::vector<uint32_t> words;
  words.reserve(64);
  append_gfx1250_scratch_dependency_barrier(words);
  uint8_t current_mode = original_mode;
  if (scratch.lease->spilled || masks->has_carrier()) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_scratch_preservation(words, *scratch.lease, false)) {
      return ExpandResult::failed("gfx1250 E5M3 unpack could not preserve its scratch VGPRs");
    }
    if (!append_gfx1250_sgpr_preservation(words, *masks, false)) {
      return ExpandResult::failed("gfx1250 E5M3 unpack could not preserve its SGPR masks");
    }
  }
  append_gfx1250_vgpr_msb_transition(words, current_mode, extract_mode);
  append_words(words,
               cdna5::build_vop3(cdna5::kVBfeU32Vop3,
                                 {.vdst = static_cast<uint8_t>(out),
                                  .src0 = static_cast<uint16_t>(source.src0),
                                  .src1 = gfx1250_inline_u32(static_cast<uint16_t>(byte_sel * 8u)),
                                  .src2 = gfx1250_inline_u32(8)}));
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);

  append_compare_literal(words, cdna5::kVCmpEqU32Vop3, static_cast<uint8_t>(nan_mask),
                         gfx1250_vgpr_src(out), 0xffu);
  append_compare_literal(words, cdna5::kVCmpLtU32Vop3, static_cast<uint8_t>(exp31_mask),
                         gfx1250_vgpr_src(out), 0xf7u);
  append_words(words, cdna5::build_vop3(cdna5::kVAndB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                              .src0 = gfx1250_inline_u32(7),
                                                              .src1 = gfx1250_vgpr_src(out)}));
  append_words(words,
               cdna5::build_vop3(cdna5::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                           .src0 = gfx1250_inline_u32(20),
                                                           .src1 = gfx1250_vgpr_src(temp)}));
  append_vop3_literal(
      words, cdna5::kVOrB32Vop3,
      {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = gfx1250_vgpr_src(temp)},
      0x47800000u);
  append_words(words, cdna5::build_vop3(cdna5::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_inline_u32(7),
                                                                  .src1 = gfx1250_vgpr_src(out)}));
  append_words(words, cdna5::build_vop3(cdna5::kVCvtF32F16Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                 .src0 = gfx1250_vgpr_src(out)}));
  append_words(words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                                  .src0 = gfx1250_vgpr_src(out),
                                                                  .src1 = gfx1250_vgpr_src(temp),
                                                                  .src2 = exp31_mask}));
  append_vop3_literal(words, cdna5::kVMovB32Vop3, {.vdst = static_cast<uint8_t>(temp), .src0 = 255},
                      0x7fa3d000u);

  const uint8_t final_mode = static_cast<uint8_t>(*dst_bank << 6);
  append_gfx1250_vgpr_msb_transition(words, current_mode, final_mode);
  append_words(
      words, cdna5::build_vop3(cdna5::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(source.vdst),
                                                         .src0 = gfx1250_vgpr_src(out),
                                                         .src1 = gfx1250_vgpr_src(temp),
                                                         .src2 = nan_mask}));
  if (scratch.lease->spilled || masks->has_carrier()) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    if (!append_gfx1250_sgpr_preservation(words, *masks, true)) {
      return ExpandResult::failed("gfx1250 E5M3 unpack could not restore its SGPR masks");
    }
    if (!append_gfx1250_scratch_preservation(words, *scratch.lease, true)) {
      return ExpandResult::failed("gfx1250 E5M3 unpack could not restore its scratch VGPRs");
    }
  }
  append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  if (!prepend_gfx1250_execz_guard_for_masked_replacement(words, masks->has_carrier()))
    return ExpandResult::failed("gfx1250 E5M3 unpack SGPR-carrier guard is too large");
  return ExpandResult::success(std::move(words));
}

[[nodiscard]] bool cvt_f32_fp8_e5m3_residual(const Instruction &inst) {
  if (!inst.mnemonic().starts_with("v_cvt_f32_fp8") ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return true;
  }
  cdna5::Vop3MachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  return source.clamp != 0;
}

[[nodiscard]] bool cvt_fp8_f32_e5m3_residual(const Instruction &inst, std::string_view mnemonic) {
  if (inst.mnemonic() != mnemonic ||
      inst.size() < static_cast<int>(sizeof(cdna5::Vop3MachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return true;
  }
  cdna5::Vop3MachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  return source.clamp != 0;
}

[[nodiscard]] bool cvt_pk_fp8_f32_e5m3_residual(const Instruction &inst) {
  return cvt_fp8_f32_e5m3_residual(inst, "v_cvt_pk_fp8_f32");
}

[[nodiscard]] bool cvt_sr_fp8_f32_e5m3_residual(const Instruction &inst) {
  return cvt_fp8_f32_e5m3_residual(inst, "v_cvt_sr_fp8_f32");
}

/// @brief Wrap a standalone low-precision WMMA in an A0-safe neutral scale prefix.
///
/// @details gfx1250 A0 cannot safely expose the bare F8F6F4 matrix instruction to
/// trap/CWSR recovery. The documented A0 form is the regular-Scale four-DWORD
/// instruction. In the scale-source context, inline integer zero selects the
/// neutral E8M0 scale. Keep the original two-DWORD matrix body byte-for-byte so its
/// formats, accumulator, modifiers, and register operands retain their source
/// semantics. The prefix's otherwise-unused SRC2 must encode VGPR0 to avoid the
/// documented false scalar dependency.
ExpandResult expand_gfx1250_bare_f8f6f4_wmma(const Instruction &inst, uint32_t, uint64_t,
                                             std::span<const uint8_t>, const LivenessAnalysis &,
                                             TranslationContext &, const LaneLayout *,
                                             const LaneLayout *) {
  if (inst.mnemonic() != "v_wmma_f32_16x16x128_f8f6f4" ||
      inst.opcode() != cdna5::kVWmmaF3216x16x128F8f6f4Vop3p ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 bare F8F6F4 WMMA rule received an unsupported instruction");
  }

  cdna5::Vop3pMachineInst matrix{};
  std::memcpy(&matrix, inst.raw_encoding(), sizeof(matrix));
  if (const char *error = gfx1250_floating_wmma_control_error(matrix))
    return ExpandResult::failed(error);

  std::vector<uint32_t> words;
  words.reserve(4);
  constexpr uint16_t kVgprEncoding = 256;
  append_words(words, cdna5::build_vop3p(kWmmaScaleSrc2PrefixOp, {.src0 = kGfx1250InlineZero,
                                                                  .src1 = kGfx1250InlineZero,
                                                                  .src2 = kVgprEncoding}));
  words.insert(words.end(), inst.raw_encoding(), inst.raw_encoding() + 2);
  return ExpandResult::success(std::move(words));
}

/// @brief Return the mixed-format selections for one f32 K=128 FP8/BF8 WMMA.
[[nodiscard]] bool gfx1250_k128_wmma_formats(uint16_t opcode, uint8_t &matrix_a_fmt,
                                             uint8_t &matrix_b_fmt) {
  matrix_a_fmt = 0;
  matrix_b_fmt = 0;
  switch (opcode) {
  case cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p:
    return true;
  case cdna5::kVWmmaF3216x16x128Fp8Bf8Vop3p:
    matrix_b_fmt = 1;
    return true;
  case cdna5::kVWmmaF3216x16x128Bf8Fp8Vop3p:
    matrix_a_fmt = 1;
    return true;
  case cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p:
    matrix_a_fmt = 1;
    matrix_b_fmt = 1;
    return true;
  default:
    return false;
  }
}

/// @brief Return whether an opcode is a B0 packed-f16 K=128 WMMA.
[[nodiscard]] bool gfx1250_is_f16_k128_wmma(uint16_t opcode) {
  switch (opcode) {
  case cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p:
  case cdna5::kVWmmaF1616x16x128Fp8Bf8Vop3p:
  case cdna5::kVWmmaF1616x16x128Bf8Fp8Vop3p:
  case cdna5::kVWmmaF1616x16x128Bf8Bf8Vop3p:
    return true;
  default:
    return false;
  }
}

/// @brief Lower B0 K=128 FP8/BF8 WMMAs to A0-supported forms.
///
/// @details Packed-f16 results use the f32 K=128 mixed-format operation in
/// scratch registers and round once when packing the final result. F32 results
/// emit one regular-Scale F8F6F4 operation with
/// FP8/BF8 matrix-format selectors and neutral inline E8M0 scales. It retains
/// the K=128 accumulation topology and requires no partial destination. Source
/// reuse hints are cleared: the target instruction family differs, and a
/// preceding source instruction may itself expand to more than one operation.
///
/// The source opcode does not encode matrix formats in OPSEL/OPSEL_HI. Rebuild
/// the target format selectors from the source opcode rather than copying those
/// fields. The defined matrix reuse hints are deliberately cleared because the
/// target belongs to a different instruction family. Only the defined C
/// absolute and negate bits are transferred.
ExpandResult expand_gfx1250_k128_wmma(const Instruction &inst, uint32_t, uint64_t,
                                      std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                      TranslationContext &context, const LaneLayout *,
                                      const LaneLayout *) {
  uint8_t matrix_a_fmt = 0;
  uint8_t matrix_b_fmt = 0;
  if (inst.size() != static_cast<int>(sizeof(cdna5::Vop3pMachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 K=128 WMMA has no complete source encoding");
  }
  cdna5::Vop3pMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  if (const char *error = gfx1250_floating_wmma_control_error(source))
    return ExpandResult::failed(error);

  constexpr uint16_t kVgprEncoding = 256;
  if (source.src0 < kVgprEncoding || source.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 K=128 WMMA matrix operands are not ordinary VGPR ranges");
  }
  const bool packed_f16 = gfx1250_is_f16_k128_wmma(inst.opcode());
  if (!gfx1250_k128_wmma_formats(inst.opcode(), matrix_a_fmt, matrix_b_fmt)) {
    if (packed_f16) {
      switch (inst.opcode()) {
      case cdna5::kVWmmaF1616x16x128Fp8Bf8Vop3p:
        matrix_b_fmt = 1;
        break;
      case cdna5::kVWmmaF1616x16x128Bf8Fp8Vop3p:
        matrix_a_fmt = 1;
        break;
      case cdna5::kVWmmaF1616x16x128Bf8Bf8Vop3p:
        matrix_a_fmt = 1;
        matrix_b_fmt = 1;
        break;
      default:
        break;
      }
    } else {
      return ExpandResult::failed("gfx1250 K=128 WMMA rule received an unsupported opcode");
    }
  }

  if (packed_f16) {
    // These operand restrictions belong to the packed-f16 lowering alone: it
    // addresses the accumulator and destination one dword at a time and needs
    // an f32 source it can materialize. The f32 path re-encodes VDST, SRC0,
    // SRC1, and SRC2 unchanged, so it separately rejects accumulator selectors
    // that the scaled form cannot represent.
    if ((source.src0 & 1u) != 0 || (source.src1 & 1u) != 0 || (source.vdst & 1u) != 0) {
      return ExpandResult::failed(
          "gfx1250 f16 K=128 WMMA matrix operands and destination are not even VGPR ranges");
    }
    if (source.src2 < kVgprEncoding && source.src2 != kGfx1250InlineZero) {
      return ExpandResult::failed(
          "gfx1250 f16 K=128 WMMA accumulator is not a VGPR range or inline zero");
    }
    if (source.src2 >= kVgprEncoding && (source.src2 & 1u) != 0)
      return ExpandResult::failed("gfx1250 f16 K=128 WMMA accumulator is not an even VGPR range");

    SemanticScratchAllocator allocator(
        inst, liveness, context,
        SemanticScratchPolicy{.max_vgprs = 256,
                              .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
    SemanticScratchRequest request;
    request.count = 9;
    request.alignment = 2;
    request.forbidden = gfx1250_instruction_registers(inst);
    request.allow_spill = true;
    const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
    if (!scratch)
      return ExpandResult::failed(
          "gfx1250 f16 K=128 WMMA could not allocate f32 accumulator scratch");

    Gfx1250SgprScratchRequest mode_request;
    mode_request.count = 1;
    mode_request.forbidden = request.forbidden;
    mode_request.forbidden.expand(scratch.lease->registers());
    mode_request.carrier_mode = Gfx1250SgprCarrierMode::ExecMasked;
    const auto mode_scratch =
        acquire_gfx1250_sgprs(inst, liveness, context, &allocator, mode_request);
    if (!mode_scratch)
      return ExpandResult::failed("gfx1250 f16 K=128 WMMA could not read FP16 overflow mode");

    const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
    const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
    const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
    const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
    if (!src0_bank || !src1_bank || !src2_bank || !dst_bank)
      return ExpandResult::failed("gfx1250 f16 K=128 WMMA cannot prove the VGPR-MSB mode");
    const uint16_t physical_dst = static_cast<uint16_t>(*dst_bank * 256u + source.vdst);
    if (physical_dst + 4u > 1024u)
      return ExpandResult::failed("gfx1250 f16 K=128 WMMA destination exceeds the VGPR file");
    uint16_t physical_accumulator = 0;
    if (source.src2 >= kVgprEncoding) {
      physical_accumulator =
          static_cast<uint16_t>(*src2_bank * 256u + (source.src2 - kVgprEncoding));
      if (physical_accumulator + 4u > 1024u)
        return ExpandResult::failed("gfx1250 f16 K=128 WMMA accumulator exceeds the VGPR file");
    }
    const uint8_t original_mode =
        static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));

    std::vector<uint32_t> words;
    words.reserve(96);
    append_gfx1250_scratch_dependency_barrier(words);
    uint8_t current_mode = original_mode;
    if (scratch.lease->spilled || mode_scratch->has_carrier()) {
      append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
      if (!append_gfx1250_scratch_preservation(words, *scratch.lease, false) ||
          !append_gfx1250_sgpr_preservation(words, *mode_scratch, false)) {
        return ExpandResult::failed("gfx1250 f16 K=128 WMMA could not preserve scratch registers");
      }
    }
    append_words(words, cdna5::build_sopk(cdna5::kSGetregB32Sopk,
                                          {.simm16 = kGfx1250ModeFp16OvflHwreg,
                                           .sdst = static_cast<uint8_t>(mode_scratch->base)}));
    const uint16_t scratch_src = gfx1250_vgpr_src(scratch.lease->base);
    uint16_t f32_accumulator = source.src2;
    if (source.src2 >= kVgprEncoding) {
      for (uint16_t element = 0; element < 8; ++element) {
        // The generated conversion consumes the source accumulator as SRC0.
        // Select SRC0's bank per dword because a legal sequential tuple can
        // cross the v255/v256 boundary.
        const uint16_t source_register = static_cast<uint16_t>(physical_accumulator + element / 2u);
        const uint8_t unpack_mode = static_cast<uint8_t>(source_register / 256u);
        append_gfx1250_vgpr_msb_transition(words, current_mode, unpack_mode);
        append_words(words,
                     cdna5::build_vop3(
                         cdna5::kVCvtF32F16Vop3,
                         {.vdst = static_cast<uint8_t>(scratch.lease->base + element),
                          .opsel = static_cast<uint8_t>(element & 1u),
                          .src0 = static_cast<uint16_t>(kVgprEncoding + source_register % 256u)}));
      }
      f32_accumulator = scratch_src;
      append_words(words, cdna5::build_sopp(cdna5::kSWaitAluSopp,
                                            {.simm16 = kGfx1250WmmaCompletionWaitImmediate}));
    }

    const uint8_t matrix_mode = static_cast<uint8_t>(*src0_bank | (*src1_bank << 2));
    append_gfx1250_vgpr_msb_transition(words, current_mode, matrix_mode);
    append_words(words, cdna5::build_vop3p(kWmmaScaleSrc2PrefixOp, {.src0 = kGfx1250InlineZero,
                                                                    .src1 = kGfx1250InlineZero,
                                                                    .src2 = kVgprEncoding}));
    append_words(words, cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                           {.vdst = static_cast<uint8_t>(scratch.lease->base),
                                            .neg_hi = static_cast<uint8_t>(source.neg_hi & 0x4u),
                                            .opsel = matrix_a_fmt,
                                            .src0 = static_cast<uint16_t>(source.src0),
                                            .src1 = static_cast<uint16_t>(source.src1),
                                            .src2 = f32_accumulator,
                                            .opsel_hi = matrix_b_fmt,
                                            .neg = static_cast<uint8_t>(source.neg & 0x4u)}));
    append_words(words, cdna5::build_sopp(cdna5::kSWaitAluSopp,
                                          {.simm16 = kGfx1250WmmaCompletionWaitImmediate}));
    for (uint16_t slot = 0; slot < 16; ++slot)
      append_words(words, cdna5::build_vop1(cdna5::kVNopVop1));
    // MI400 Shader Programming 4.6.12: a WMMA or SWMMAC result of 16 bits or
    // fewer becomes +/-MAX rather than +/-infinity when MODE.FP16_OVFL is set,
    // and the MODE table adds that WMMA saturates INF on this generation while
    // every other opcode preserves it. V_CVT_PK_F16_F32 is one of those other
    // opcodes, so the pack below reproduces only the finite-overflow half of
    // the contract; clamp the infinities here. The clamp bound is
    // MODE.FP16_OVFL scaled to 0x38002000 and subtracted from f32 +infinity,
    // giving +infinity when the mode is clear and 65504.0 when it is set.
    // Do this in the low scratch bank before selecting each destination bank.
    //
    // V_MAXIMUM_F32 and V_MINIMUM_F32 return the canonical quiet NaN, so a NaN
    // result keeps its NaN class but loses the sign and payload the source
    // instruction would have produced. IEEE 754 leaves both uninterpreted and
    // the ISA promises only that a NaN input yields a NaN output.
    append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
    const uint16_t limit = static_cast<uint16_t>(scratch.lease->base + 8u);
    append_words(words, cdna5::build_vop3(cdna5::kVMulLoU32Vop3,
                                          {.vdst = static_cast<uint8_t>(limit),
                                           .src0 = 255,
                                           .src1 = static_cast<uint16_t>(mode_scratch->base)}));
    words.push_back(0x38002000u);
    append_words(words,
                 cdna5::build_vop3(cdna5::kVSubNcU32Vop3, {.vdst = static_cast<uint8_t>(limit),
                                                           .src0 = 255,
                                                           .src1 = gfx1250_vgpr_src(limit)}));
    words.push_back(0x7f800000u);
    for (uint16_t element = 0; element < 8; ++element) {
      const uint16_t value = static_cast<uint16_t>(scratch.lease->base + element);
      append_words(words,
                   cdna5::build_vop3(cdna5::kVMaximumF32Vop3, {.vdst = static_cast<uint8_t>(value),
                                                               .src0 = gfx1250_vgpr_src(value),
                                                               .src1 = gfx1250_vgpr_src(limit),
                                                               .neg = 2}));
      append_words(words,
                   cdna5::build_vop3(cdna5::kVMinimumF32Vop3, {.vdst = static_cast<uint8_t>(value),
                                                               .src0 = gfx1250_vgpr_src(value),
                                                               .src1 = gfx1250_vgpr_src(limit)}));
    }
    for (uint16_t pair = 0; pair < 4; ++pair) {
      const uint16_t destination_register = static_cast<uint16_t>(physical_dst + pair);
      const uint8_t pack_mode = static_cast<uint8_t>((destination_register / 256u) << 6);
      append_gfx1250_vgpr_msb_transition(words, current_mode, pack_mode);
      append_words(
          words, cdna5::build_vop3(cdna5::kVCvtPkF16F32Vop3,
                                   {.vdst = static_cast<uint8_t>(destination_register % 256u),
                                    .src0 = static_cast<uint16_t>(scratch_src + pair * 2u),
                                    .src1 = static_cast<uint16_t>(scratch_src + pair * 2u + 1u)}));
    }
    if (scratch.lease->spilled || mode_scratch->has_carrier()) {
      append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
      if (!append_gfx1250_sgpr_preservation(words, *mode_scratch, true) ||
          !append_gfx1250_scratch_preservation(words, *scratch.lease, true)) {
        return ExpandResult::failed("gfx1250 f16 K=128 WMMA could not restore scratch registers");
      }
    }
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
    if (!prepend_gfx1250_execz_guard_for_masked_replacement(words, mode_scratch->has_carrier())) {
      return ExpandResult::failed("gfx1250 f16 K=128 WMMA SGPR-carrier guard is too large");
    }
    return ExpandResult::success(std::move(words));
  }

  if (source.src2 < kGfx1250InlineZero) {
    return ExpandResult::failed(
        "gfx1250 K=128 f32 WMMA scalar accumulator cannot be represented by the scaled form");
  }

  if (!gfx1250_k128_wmma_formats(inst.opcode(), matrix_a_fmt, matrix_b_fmt))
    return ExpandResult::failed("gfx1250 K=128 WMMA rule received an unsupported opcode");

  std::vector<uint32_t> words;
  words.reserve(4);
  append_words(words, cdna5::build_vop3p(kWmmaScaleSrc2PrefixOp, {.src0 = kGfx1250InlineZero,
                                                                  .src1 = kGfx1250InlineZero,
                                                                  .src2 = kVgprEncoding}));
  append_words(words, cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                         {.vdst = static_cast<uint8_t>(source.vdst),
                                          .neg_hi = static_cast<uint8_t>(source.neg_hi & 0x4u),
                                          .opsel = matrix_a_fmt,
                                          .src0 = static_cast<uint16_t>(source.src0),
                                          .src1 = static_cast<uint16_t>(source.src1),
                                          .src2 = static_cast<uint16_t>(source.src2),
                                          .opsel_hi = matrix_b_fmt,
                                          .neg = static_cast<uint8_t>(source.neg & 0x4u)}));
  return ExpandResult::success(std::move(words));
}

// The semantic translator binary-searches this table, so entries must stay
// sorted by the full encoding ID and then opcode; the static_assert after the
// table enforces that. VDS encoding IDs include the high opcode bits, hence the
// four consecutive kVdsOpHi* groups below.
// An encoding id is the top nine bits of the first word, so a SOPK id is the
// SOPK base plus its opcode. The generated header names only the ids the ISA
// description needed, which is why one of these two has no constant to use; the
// assertions below pin the derivation against the one that does.
constexpr uint16_t kSetregB32EncodingId = cdna5::encoding::kSopk + cdna5::kSSetregB32Sopk;
constexpr uint16_t kSetregImm32B32EncodingId = cdna5::encoding::kSopk + cdna5::kSSetregImm32B32Sopk;
static_assert(kSetregB32EncodingId == cdna5::encoding::kSopkOpHi18,
              "SOPK encoding ids must remain the SOPK base plus the opcode");
static_assert(kSetregImm32B32EncodingId == kSetregB32EncodingId + 1,
              "consecutive SOPK opcodes must yield consecutive encoding ids");

inline constexpr std::array<TranslationRule, 43> kGfx1250B0ToA0ExpandRules = {{
    {kSetregB32EncodingId, cdna5::kSSetregB32Sopk, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_setreg_mode_ordering, nullptr, nullptr, false,
     block_checked_discharge(setreg_mode_ordering_residual)},
    {kSetregImm32B32EncodingId, cdna5::kSSetregImm32B32Sopk, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_setreg_mode_ordering, nullptr, nullptr, false,
     block_checked_discharge(setreg_mode_ordering_residual)},
    {cdna5::encoding::kSop1, cdna5::kSBarrierSignalIsfirstSop1, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_barrier_signal_isfirst, nullptr, nullptr, false,
     no_success_discharge("the rule only rejects one unsupported source encoding and never emits")},
    {cdna5::encoding::kSopp, cdna5::kSClauseSopp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_s_clause, nullptr, nullptr, false, checked_discharge(always_residual)},
    {cdna5::encoding::kVop3p, cdna5::kVWmmaF3216x16x128F8f6f4Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_bare_f8f6f4_wmma, nullptr, nullptr, false,
     checked_discharge(always_residual)},
    {cdna5::encoding::kVop3p, kWmmaScaleSrc2PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale_src2, nullptr, nullptr, true,
     checked_discharge(regular_scale_residual)},
    {cdna5::encoding::kVop3p, kWmmaScale16PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale16, nullptr, nullptr, true, checked_discharge(scale16_residual)},
    {cdna5::encoding::kVop3p, cdna5::kVWmmaI3216x16x64Iu8Vop3p, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr, false,
     block_checked_discharge(iu8_spacing_residual)},
    {cdna5::encoding::kVop3p, cdna5::kVSwmmacI3216x16x128Iu8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr, false,
     block_checked_discharge(iu8_spacing_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false,
     checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF3216x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false,
     checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF3216x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false,
     checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false,
     checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF1616x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF1616x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF1616x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVop3pOpHi1, cdna5::kVWmmaF3232x16x128F4Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_32x16_f4, nullptr, nullptr, true,
     checked_discharge(always_residual)},
    {cdna5::encoding::kVimage, cdna5::kTensorLoadToLdsVimage, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_tensor_load_to_lds, nullptr, nullptr, true,
     block_checked_discharge(tensor_load_residual)},
    {cdna5::encoding::kVop3OpHi3, cdna5::kVCvtF32Fp8Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cvt_f32_fp8_e5m3, nullptr, nullptr, true,
     checked_discharge(cvt_f32_fp8_e5m3_residual)},
    {cdna5::encoding::kVop3OpHi6, cdna5::kVCvtPkFp8F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cvt_pk_fp8_f32_e5m3, nullptr, nullptr, true,
     checked_discharge(cvt_pk_fp8_f32_e5m3_residual)},
    {cdna5::encoding::kVop3OpHi6, cdna5::kVCvtSrFp8F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cvt_sr_fp8_f32_e5m3, nullptr, nullptr, true,
     checked_discharge(cvt_sr_fp8_f32_e5m3_residual)},
    {cdna5::encoding::kVds, cdna5::kDsStore2addrB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVds, cdna5::kDsStore2addrStride64B32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi1, cdna5::kDsStorexchg2addrRtnB32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi1, cdna5::kDsStorexchg2addrStride64RtnB32Vds, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi1, cdna5::kDsLoad2addrB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi1, cdna5::kDsLoad2addrStride64B32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi2, cdna5::kDsStore2addrB64Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi2, cdna5::kDsStore2addrStride64B64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi3, cdna5::kDsStorexchg2addrRtnB64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi3, cdna5::kDsStorexchg2addrStride64RtnB64Vds, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi3, cdna5::kDsLoad2addrB64Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi3, cdna5::kDsLoad2addrStride64B64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi5, cdna5::kDsStoreAddtidB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds_addtid, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVdsOpHi5, cdna5::kDsLoadAddtidB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds_addtid, nullptr, nullptr, true, checked_discharge(always_residual)},
    {cdna5::encoding::kVglobal, cdna5::kClusterLoadB32Vglobal, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cluster_load, nullptr, nullptr, true,
     block_checked_discharge(cluster_load_residual)},
    {cdna5::encoding::kVglobal, cdna5::kClusterLoadB64Vglobal, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cluster_load, nullptr, nullptr, true,
     block_checked_discharge(cluster_load_residual)},
    {cdna5::encoding::kVglobal, cdna5::kClusterLoadB128Vglobal, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cluster_load, nullptr, nullptr, true,
     block_checked_discharge(cluster_load_residual)},
    {cdna5::encoding::kVglobal, cdna5::kClusterLoadAsyncToLdsB8Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr, true,
     block_checked_discharge(cluster_load_residual)},
    {cdna5::encoding::kVglobal, cdna5::kClusterLoadAsyncToLdsB32Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr, true,
     block_checked_discharge(cluster_load_residual)},
    {cdna5::encoding::kVglobal, cdna5::kClusterLoadAsyncToLdsB64Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr, true,
     block_checked_discharge(cluster_load_residual)},
    {cdna5::encoding::kVglobal, cdna5::kClusterLoadAsyncToLdsB128Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr, true,
     block_checked_discharge(cluster_load_residual)},
}};

static_assert(translation_rules_sorted(kGfx1250B0ToA0ExpandRules),
              "the gfx1250 B0-to-A0 rule table must stay sorted by (encoding id, opcode)");

inline constexpr std::array<RegisteredInstructionRewrite, 1> kGfx1250B0ToA0InstructionRewriteRules =
    {{
        {"flat-scratch-base-64bit-source", flat_scratch_base_rewrite_applies,
         lower_flat_scratch_base_rewrite, true,
         checked_discharge(flat_scratch_base_rewrite_residual)},
    }};

inline constexpr RewriteRegistry kGfx1250B0ToA0RewriteRegistry = {
    kGfx1250B0ToA0ExpandRules,
    kGfx1250B0ToA0InstructionRewriteRules,
};

static_assert(kGfx1250B0ToA0RewriteRegistry.has_complete_discharge());

} // namespace

std::span<const TranslationRule> semantic_expand_rules_gfx1250_b0_to_a0() {
  return kGfx1250B0ToA0ExpandRules;
}

RewriteRegistry rewrite_registry_gfx1250_b0_to_a0() { return kGfx1250B0ToA0RewriteRegistry; }

} // namespace rocjitsu
