// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_b0_to_a0.cpp
/// @brief Handwritten semantic expansions for gfx1250 B0-to-A0 translation.

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/gfx1250_rewrite_hazards.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/semantic_scratch.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

namespace {

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

/// @brief Diagnose control fields that are invalid for floating-point WMMA.
///
/// @details CM in the matrix body and SCL_CM in a scale prefix are both
/// required to be zero. Keep this validation at every floating-point WMMA
/// entry point so malformed encodings cannot be copied or expanded.
[[nodiscard]] const char *
gfx1250_floating_wmma_control_error(const gfx1250::Vop3pMachineInst &matrix,
                                    const gfx1250::Vop3pMachineInst *scale = nullptr) {
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

[[nodiscard]] Gfx1250GeneratedValuHazard
generated_valu_hazard_before(const TranslationContext &context, const Instruction &inst) {
  if (context.gfx1250_rewrite_hazards == nullptr)
    return {};
  return context.gfx1250_rewrite_hazards->before_generated_valu(inst);
}

void append_generated_valu_hazard_nops(std::vector<uint32_t> &words,
                                       const Gfx1250GeneratedValuHazard &hazard,
                                       const TranslationContext &context, const RegisterSet &defs,
                                       const RegisterSet &uses = {}) {
  const uint8_t required = hazard.required_nops(defs, uses);
  const uint8_t additional = required > context.leading_valu_hazard_nops
                                 ? static_cast<uint8_t>(required - context.leading_valu_hazard_nops)
                                 : 0;
  for (uint8_t slot = 0; slot < additional; ++slot)
    append_words(words, gfx1250::build_vop1(gfx1250::kVNopVop1));
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
    append_words(words, gfx1250::build_sopp(gfx1250::kSNopSopp, {.simm16 = 0}));

  append_words(words, gfx1250::build_sopp(gfx1250::kSWaitXcntSopp, {.simm16 = 0}));
  const uint16_t immediate =
      static_cast<uint16_t>(new_mode) | (static_cast<uint16_t>(current_mode) << 8);
  append_words(words, gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = immediate}));
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
    append_words(words, gfx1250::build_vscratch(restore ? gfx1250::kScratchLoadB32Vscratch
                                                        : gfx1250::kScratchStoreB32Vscratch,
                                                {.saddr = kGfx1250Null,
                                                 .vdst = restore ? vgpr : uint8_t{0},
                                                 .vsrc = restore ? uint8_t{0} : vgpr,
                                                 .ioffset = byte_offset}));
  }
  append_words(
      words, gfx1250::build_sopp(restore ? gfx1250::kSWaitLoadcntSopp : gfx1250::kSWaitStorecntSopp,
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
  append_words(words, gfx1250::build_sopp(gfx1250::kSWaitIdleSopp));
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
  RegisterSet avoid;
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
  carrier_request.avoid = request.avoid;
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
                     gfx1250::build_vop3(gfx1250::kVReadlaneB32Vop3,
                                         {.vdst = static_cast<uint8_t>(lease.base + i),
                                          .src0 = static_cast<uint16_t>(256u + carrier.base + i),
                                          .src1 = kGfx1250InlineZero}));
      } else {
        append_words(words, gfx1250::build_vop3(gfx1250::kVWritelaneB32Vop3,
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
      append_words(words, gfx1250::build_vop1(gfx1250::kVMovB32Vop1,
                                              {.src0 = static_cast<uint16_t>(lease.base + i),
                                               .vdst = static_cast<uint8_t>(carrier.base + i)}));
    }
    return true;
  }

  for (uint16_t i = 0; i < lease.count; ++i) {
    append_words(words, gfx1250::build_vop1(gfx1250::kVReadfirstlaneB32Vop1,
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
               gfx1250::build_sopp(gfx1250::kSCbranchExeczSopp,
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

  const auto nop = gfx1250::build_sopp(gfx1250::kSNopSopp, {.simm16 = 0});
  return ExpandResult::success(std::vector<uint32_t>(nop.begin(), nop.end()));
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
  case gfx1250::kDsLoad2addrB32Vds:
    return {gfx1250::kDsLoadB32Vds, 1, false, Kind::Load};
  case gfx1250::kDsLoad2addrStride64B32Vds:
    return {gfx1250::kDsLoadB32Vds, 1, true, Kind::Load};
  case gfx1250::kDsStore2addrB32Vds:
    return {gfx1250::kDsStoreB32Vds, 1, false, Kind::Store};
  case gfx1250::kDsStore2addrStride64B32Vds:
    return {gfx1250::kDsStoreB32Vds, 1, true, Kind::Store};
  case gfx1250::kDsStorexchg2addrRtnB32Vds:
    return {gfx1250::kDsStorexchgRtnB32Vds, 1, false, Kind::StoreExchange};
  case gfx1250::kDsStorexchg2addrStride64RtnB32Vds:
    return {gfx1250::kDsStorexchgRtnB32Vds, 1, true, Kind::StoreExchange};
  case gfx1250::kDsLoad2addrB64Vds:
    return {gfx1250::kDsLoadB64Vds, 2, false, Kind::Load};
  case gfx1250::kDsLoad2addrStride64B64Vds:
    return {gfx1250::kDsLoadB64Vds, 2, true, Kind::Load};
  case gfx1250::kDsStore2addrB64Vds:
    return {gfx1250::kDsStoreB64Vds, 2, false, Kind::Store};
  case gfx1250::kDsStore2addrStride64B64Vds:
    return {gfx1250::kDsStoreB64Vds, 2, true, Kind::Store};
  case gfx1250::kDsStorexchg2addrRtnB64Vds:
    return {gfx1250::kDsStorexchgRtnB64Vds, 2, false, Kind::StoreExchange};
  case gfx1250::kDsStorexchg2addrStride64RtnB64Vds:
    return {gfx1250::kDsStorexchgRtnB64Vds, 2, true, Kind::StoreExchange};
  default:
    return {};
  }
}

/// @brief Build one single-address DS instruction from a DS2 operand half.
[[nodiscard]] std::array<uint32_t, 2> build_gfx1250_ds2_half(const gfx1250::VdsMachineInst &source,
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
  return gfx1250::build_vds(
      shape.replacement_opcode,
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
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VdsMachineInst)) {
    return ExpandResult::failed("gfx1250 DS2 instruction has no complete VDS encoding",
                                {"Decode the complete eight-byte VDS instruction."});
  }

  gfx1250::VdsMachineInst source{};
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
  append_words(words, gfx1250::build_sopp(gfx1250::kSWaitDscntSopp, {.simm16 = 0}));
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
  return gfx1250::build_sop2(
      gfx1250::kSPackHhB32B16Sop2,
      {.ssrc0 = kGfx1250InlineZero, .ssrc1 = descriptor_base, .sdst = descriptor_base})[0];
}

/// @brief Build the canonical save/clear/restore words around one tensor load.
[[nodiscard]] TensorMaskWrapper build_tensor_mask_wrapper(uint8_t descriptor_base,
                                                          uint8_t scratch) {
  return {
      .save = gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                  {.ssrc0 = descriptor_base, .sdst = scratch})[0],
      .clear = build_tensor_mask_clear(descriptor_base),
      .restore = gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                     {.ssrc0 = scratch, .sdst = descriptor_base})[0],
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
      inst.size() != static_cast<int>(sizeof(gfx1250::VimageMachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 tensor-load mask rule received an unsupported instruction");
  }
  gfx1250::VimageMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint8_t kLastOrdinarySgpr = 105;
  const uint8_t descriptor_base = static_cast<uint8_t>(source.vaddr1);
  if (descriptor_base == kGfx1250Null || descriptor_base > kLastOrdinarySgpr - 7u) {
    return ExpandResult::failed(
        "gfx1250 tensor-load group-1 descriptor is not a valid eight-SGPR tuple",
        {"Provide TENSOR_LOAD_TO_LDS VADDR1 as an ordinary eight-SGPR descriptor."});
  }

  if (has_tensor_mask_clear(inst, descriptor_base)) {
    return ExpandResult::success(std::vector<uint32_t>(
        inst.raw_encoding(),
        inst.raw_encoding() + sizeof(gfx1250::VimageMachineInst) / sizeof(uint32_t)));
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
               inst.raw_encoding() + sizeof(gfx1250::VimageMachineInst) / sizeof(uint32_t));
  words.push_back(wrapper.restore);

  return ExpandResult::success(std::move(words));
}

/// @brief Replace one bit field in a 32-bit instruction word.
void set_word_field(uint32_t &word, uint32_t value, uint32_t shift, uint32_t width) {
  const uint32_t mask = ((uint32_t{1} << width) - 1) << shift;
  word = (word & ~mask) | ((value << shift) & mask);
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
                      const gfx1250::Vop3pMachineInst &matrix, std::array<uint32_t, 2> prefix,
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
    auto replacement = gfx1250::build_vop3p(
        gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
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

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (const char *error = gfx1250_floating_wmma_control_error(matrix, &scale))
    return ExpandResult::failed(error);
  if (matrix.op != gfx1250::kVWmmaF3232x16x128F4Vop3p) {
    std::vector<uint32_t> words(inst.raw_encoding(), inst.raw_encoding() + 4);
    // Instruction bits [58:50] occupy word 1 bits [26:18].
    set_word_field(words[1], 0x100, 18, 9);
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
      inst.opcode() != gfx1250::kVWmmaF3232x16x128F4Vop3p ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 32x16 FP4 WMMA rule received an unsupported instruction");
  }

  gfx1250::Vop3pMachineInst matrix{};
  std::memcpy(&matrix, inst.raw_encoding(), sizeof(matrix));
  // This form produces floating-point results, so it carries the same control
  // requirement as every other floating-point matrix entry point.
  if (const char *error = gfx1250_floating_wmma_control_error(matrix))
    return ExpandResult::failed(error);
  constexpr uint16_t kVgprEncoding = 256;
  if (matrix.src0 < kVgprEncoding || matrix.src1 < kVgprEncoding)
    return ExpandResult::failed("gfx1250 32x16 FP4 operands are not VGPR ranges");

  const auto prefix = gfx1250::build_vop3p(
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

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (scale.op != kWmmaScale16PrefixOp || (matrix.op != gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p &&
                                           matrix.op != gfx1250::kVWmmaF3232x16x128F4Vop3p)) {
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
  if (matrix.op == gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p) {
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
    auto replacement = gfx1250::build_vop3p(
        gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
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

/// @brief Conservatively separate B0 integer WMMA from its A0 successor.
///
/// @details gfx1250 requires nine separating V_NOPs when dense IU8 WMMA feeds a
/// following XDL matrix input, and five for sparse IU8 SWMMAC. These bounds also
/// cover their shorter WMMA-to-VALU hazard windows. Count exact canonical V_NOPs
/// already following the instruction in the same basic block and append only
/// the missing slots. Limiting credit to the block guarantees that every
/// credited word remains adjacent after layout. Noncanonical NOPs and following
/// control-flow successors conservatively receive no credit.
ExpandResult expand_gfx1250_wmma_iu8_spacing(const Instruction &inst, uint32_t, uint64_t,
                                             std::span<const uint8_t>, const LivenessAnalysis &,
                                             TranslationContext &, const LaneLayout *,
                                             const LaneLayout *) {
  if (inst.mnemonic() != "v_wmma_i32_16x16x64_iu8" &&
      inst.mnemonic() != "v_swmmac_i32_16x16x128_iu8")
    return ExpandResult::not_handled();
  if (inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr)
    return ExpandResult::failed("gfx1250 IU8 WMMA rule received an unsupported VOP3P instruction");

  std::vector<uint32_t> words(inst.raw_encoding(),
                              inst.raw_encoding() + inst.size() / sizeof(uint32_t));
  const int required_slots = inst.mnemonic() == "v_wmma_i32_16x16x64_iu8" ? 9 : 5;
  const uint32_t v_nop = gfx1250::build_vop1(gfx1250::kVNopVop1)[0];
  int existing_slots = 0;
  const Instruction *next = inst.next_instruction();
  while (existing_slots < required_slots && next != nullptr &&
         next->size() == static_cast<int>(sizeof(uint32_t)) && next->raw_encoding() != nullptr &&
         next->raw_encoding()[0] == v_nop) {
    ++existing_slots;
    next = next->next_instruction();
  }

  // TODO: Replace canonical V_NOP counting with whole-kernel scheduling that
  // can also credit independent VALU in each reachable successor.
  for (int slot = existing_slots; slot < required_slots; ++slot)
    words.push_back(v_nop);
  return ExpandResult::success(std::move(words));
}

/// @brief True if @p opcode is a gfx1250 cluster-load form this rule covers
/// (both the plain and async-to-LDS families, all widths).
[[nodiscard]] bool is_gfx1250_cluster_load(uint16_t opcode) {
  switch (opcode) {
  case gfx1250::kClusterLoadB32Vglobal:
  case gfx1250::kClusterLoadB64Vglobal:
  case gfx1250::kClusterLoadB128Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB8Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB32Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB64Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB128Vglobal:
    return true;
  default:
    return false;
  }
}

/// @brief Build the canonical M0 = 0 word emitted before a cluster load.
[[nodiscard]] constexpr uint32_t build_cluster_m0_clear() {
  return gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                             {.ssrc0 = kGfx1250InlineZero, .sdst = kGfx1250M0})[0];
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

  if (has_canonical_predecessor(inst, build_cluster_m0_clear())) {
    return ExpandResult::success(
        std::vector<uint32_t>(inst.raw_encoding(), inst.raw_encoding() + 3));
  }

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256,
                            .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
  const Gfx1250GeneratedValuHazard leading_hazard = generated_valu_hazard_before(context, inst);
  Gfx1250SgprScratchRequest scratch_request;
  scratch_request.count = 1;
  scratch_request.forbidden = gfx1250_instruction_registers(inst);
  scratch_request.avoid = leading_hazard.avoid_destination_vgprs_in_bank(0);
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
    RegisterSet generated_carrier_defs;
    generated_carrier_defs.expand(scratch->carrier->registers());
    append_generated_valu_hazard_nops(words, leading_hazard, context, generated_carrier_defs);
    if (!append_gfx1250_sgpr_preservation(words, *scratch, false))
      return ExpandResult::failed("gfx1250 cluster load could not preserve scalar scratch");
    append_gfx1250_vgpr_msb_transition(words, current_mode, *original_mode);
  }
  append_words(words, gfx1250::build_sop1(
                          gfx1250::kSMovB32Sop1,
                          {.ssrc0 = kGfx1250M0, .sdst = static_cast<uint8_t>(scratch->base)}));
  words.push_back(build_cluster_m0_clear());
  words.insert(words.end(), inst.raw_encoding(), inst.raw_encoding() + 3);
  append_words(words, gfx1250::build_sop1(
                          gfx1250::kSMovB32Sop1,
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
  const bool is_load = inst.opcode() == gfx1250::kDsLoadAddtidB32Vds;
  const bool is_store = inst.opcode() == gfx1250::kDsStoreAddtidB32Vds;
  if ((!is_load && !is_store) || inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 ADDTID rule received an unsupported instruction");
  }

  gfx1250::VdsMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  const Gfx1250GeneratedValuHazard leading_hazard = generated_valu_hazard_before(context, inst);

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
    request.avoid = leading_hazard.avoid_destination_vgprs_in_bank(0);
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
  RegisterSet generated_temp;
  generated_temp.expand(
      RegisterRef{RegClass::VGPR, static_cast<uint16_t>(temp + compute_bank * 256u), 1});
  append_generated_valu_hazard_nops(words, leading_hazard, context, generated_temp);
  if (store_scratch && store_scratch->spilled &&
      !append_gfx1250_scratch_preservation(words, *store_scratch, false)) {
    return ExpandResult::failed("gfx1250 DS store ADDTID could not preserve low-bank scratch");
  }
  append_words(
      words, gfx1250::build_vop3(gfx1250::kVMbcntLoU32B32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                                .src0 = 193, // inline -1
                                                                .src1 = gfx1250_inline_u32(0)}));
  append_words(
      words, gfx1250::build_vop3(gfx1250::kVMbcntHiU32B32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                                .src0 = 193,
                                                                .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                               .src0 = gfx1250_inline_u32(2),
                                                               .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVAddNcU32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                             .src0 = kGfx1250M0,
                                                             .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVBfeU32Vop3, {.vdst = static_cast<uint8_t>(temp),
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
    append_words(words, gfx1250::build_vds(gfx1250::kDsStoreB32Vds,
                                           {.offset0 = static_cast<uint8_t>(source.offset0),
                                            .offset1 = static_cast<uint8_t>(source.offset1),
                                            .addr = static_cast<uint8_t>(temp),
                                            .data0 = static_cast<uint8_t>(source.data0)}));
    if (store_scratch && store_scratch->spilled) {
      append_words(words, gfx1250::build_sopp(gfx1250::kSWaitDscntSopp, {.simm16 = 0}));
      append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
      if (!append_gfx1250_scratch_preservation(words, *store_scratch, true)) {
        return ExpandResult::failed("gfx1250 DS store ADDTID could not restore low-bank scratch");
      }
    }
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  } else {
    append_words(words, gfx1250::build_vds(gfx1250::kDsLoadB32Vds,
                                           {.offset0 = static_cast<uint8_t>(source.offset0),
                                            .offset1 = static_cast<uint8_t>(source.offset1),
                                            .addr = static_cast<uint8_t>(temp),
                                            .vdst = static_cast<uint8_t>(temp)}));
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  }
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
  gfx1250::Vop3MachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint16_t kVgprEncoding = 256;
  if (source.clamp == 0)
    return ExpandResult::not_handled();
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
  const Gfx1250GeneratedValuHazard leading_hazard = generated_valu_hazard_before(context, inst);
  request.avoid = leading_hazard.avoid_destination_vgprs_in_bank(0);
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
  mask_request.avoid = leading_hazard.avoid_destination_vgprs_in_bank(0);
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
                                      gfx1250::Vop3BuilderFields fields, uint32_t literal) {
    append_words(words, gfx1250::build_vop3(opcode, fields));
    words.push_back(literal);
  };
  const auto append_compare_literal = [](std::vector<uint32_t> &words, uint16_t opcode,
                                         uint8_t sdst, uint16_t src1, uint32_t literal) {
    // gfx1250 VOP3 compares encode their scalar mask destination in the ordinary
    // VOP3 vdst field. Vop3SdstEnc is a different format whose sdst bits overlap
    // modifiers here; using it leaves vdst=0 and corrupts live s0.
    append_words(words, gfx1250::build_vop3(opcode, {.vdst = sdst, .src0 = 255, .src1 = src1}));
    words.push_back(literal);
  };

  // The VOP3 encoding swaps the two byte-select bits for this opcode.
  const uint8_t byte_sel =
      static_cast<uint8_t>(((source.opsel & 1u) << 1u) | ((source.opsel & 2u) >> 1u));
  std::vector<uint32_t> words;
  words.reserve(64);
  append_gfx1250_scratch_dependency_barrier(words);
  RegisterSet generated_defs;
  generated_defs.expand(scratch.lease->registers());
  if (masks->carrier)
    generated_defs.expand(masks->carrier->registers());
  RegisterSet generated_uses;
  generated_uses.expand(RegisterRef{
      RegClass::VGPR, static_cast<uint16_t>(source.src0 - kVgprEncoding + *src0_bank * 256u), 1});
  append_generated_valu_hazard_nops(words, leading_hazard, context, generated_defs, generated_uses);
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
  append_words(
      words, gfx1250::build_vop3(gfx1250::kVBfeU32Vop3,
                                 {.vdst = static_cast<uint8_t>(out),
                                  .src0 = static_cast<uint16_t>(source.src0),
                                  .src1 = gfx1250_inline_u32(static_cast<uint16_t>(byte_sel * 8u)),
                                  .src2 = gfx1250_inline_u32(8)}));
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);

  append_compare_literal(words, gfx1250::kVCmpEqU32Vop3, static_cast<uint8_t>(nan_mask),
                         gfx1250_vgpr_src(out), 0xffu);
  append_compare_literal(words, gfx1250::kVCmpLtU32Vop3, static_cast<uint8_t>(exp31_mask),
                         gfx1250_vgpr_src(out), 0xf7u);
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVAndB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                           .src0 = gfx1250_inline_u32(7),
                                                           .src1 = gfx1250_vgpr_src(out)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                               .src0 = gfx1250_inline_u32(20),
                                                               .src1 = gfx1250_vgpr_src(temp)}));
  append_vop3_literal(
      words, gfx1250::kVOrB32Vop3,
      {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = gfx1250_vgpr_src(temp)},
      0x47800000u);
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                               .src0 = gfx1250_inline_u32(7),
                                                               .src1 = gfx1250_vgpr_src(out)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVCvtF32F16Vop3, {.vdst = static_cast<uint8_t>(out),
                                                              .src0 = gfx1250_vgpr_src(out)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                               .src0 = gfx1250_vgpr_src(out),
                                                               .src1 = gfx1250_vgpr_src(temp),
                                                               .src2 = exp31_mask}));
  append_vop3_literal(words, gfx1250::kVMovB32Vop3,
                      {.vdst = static_cast<uint8_t>(temp), .src0 = 255}, 0x7fa3d000u);

  const uint8_t final_mode = static_cast<uint8_t>(*dst_bank << 6);
  append_gfx1250_vgpr_msb_transition(words, current_mode, final_mode);
  append_words(words, gfx1250::build_vop3(gfx1250::kVCndmaskB32Vop3,
                                          {.vdst = static_cast<uint8_t>(source.vdst),
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
      inst.opcode() != gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 bare F8F6F4 WMMA rule received an unsupported instruction");
  }

  gfx1250::Vop3pMachineInst matrix{};
  std::memcpy(&matrix, inst.raw_encoding(), sizeof(matrix));
  if (const char *error = gfx1250_floating_wmma_control_error(matrix))
    return ExpandResult::failed(error);

  std::vector<uint32_t> words;
  words.reserve(4);
  constexpr uint16_t kVgprEncoding = 256;
  append_words(words, gfx1250::build_vop3p(kWmmaScaleSrc2PrefixOp, {.src0 = kGfx1250InlineZero,
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
  case gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p:
    return true;
  case gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p:
    matrix_b_fmt = 1;
    return true;
  case gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p:
    matrix_a_fmt = 1;
    return true;
  case gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p:
    matrix_a_fmt = 1;
    matrix_b_fmt = 1;
    return true;
  default:
    return false;
  }
}

/// @brief Lower a B0 K=128 FP8/BF8 WMMA to an A0 K=128 mixed-format WMMA.
///
/// @details The preferred path emits one regular-Scale F8F6F4 operation with
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
                                      std::span<const uint8_t>, const LivenessAnalysis &,
                                      TranslationContext &, const LaneLayout *,
                                      const LaneLayout *) {
  switch (inst.opcode()) {
  case gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p:
  case gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p:
  case gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p:
  case gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p:
    return ExpandResult::failed(
        "gfx1250 f16 K=128 WMMA A0 lowering is not yet implemented",
        {"Provide an exact packed-f16 accumulator and single-rounding lowering."});
  default:
    break;
  }

  uint8_t matrix_a_fmt = 0;
  uint8_t matrix_b_fmt = 0;
  if (inst.size() != static_cast<int>(sizeof(gfx1250::Vop3pMachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 K=128 WMMA has no complete source encoding");
  }
  if (!gfx1250_k128_wmma_formats(inst.opcode(), matrix_a_fmt, matrix_b_fmt))
    return ExpandResult::failed("gfx1250 K=128 WMMA rule received an unsupported opcode");

  gfx1250::Vop3pMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  if (const char *error = gfx1250_floating_wmma_control_error(source))
    return ExpandResult::failed(error);

  constexpr uint16_t kVgprEncoding = 256;
  if (source.src0 < kVgprEncoding || source.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 K=128 WMMA matrix operands are not ordinary VGPR ranges");
  }

  std::vector<uint32_t> words;
  words.reserve(4);
  append_words(words, gfx1250::build_vop3p(kWmmaScaleSrc2PrefixOp, {.src0 = kGfx1250InlineZero,
                                                                    .src1 = kGfx1250InlineZero,
                                                                    .src2 = kVgprEncoding}));
  append_words(words, gfx1250::build_vop3p(gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
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
// sorted by the full encoding ID and then opcode. VDS encoding IDs include the
// high opcode bits, hence the four consecutive kVdsOpHi* groups below.
inline constexpr std::array<TranslationRule, 38> kGfx1250B0ToA0ExpandRules = {{
    {gfx1250::encoding::kSopp, gfx1250::kSClauseSopp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_s_clause, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3p, gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_bare_f8f6f4_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3p, kWmmaScaleSrc2PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale_src2, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, kWmmaScale16PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale16, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, gfx1250::kVWmmaI3216x16x64Iu8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3p, gfx1250::kVSwmmacI3216x16x128Iu8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3232x16x128F4Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_32x16_f4, nullptr, nullptr},
    {gfx1250::encoding::kVimage, gfx1250::kTensorLoadToLdsVimage, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_tensor_load_to_lds, nullptr, nullptr},
    {gfx1250::encoding::kVop3OpHi3, gfx1250::kVCvtF32Fp8Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cvt_f32_fp8_e5m3, nullptr, nullptr},
    {gfx1250::encoding::kVds, gfx1250::kDsStore2addrB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVds, gfx1250::kDsStore2addrStride64B32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsStorexchg2addrRtnB32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsStorexchg2addrStride64RtnB32Vds, RuleAction::Expand,
     0, 0, nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsLoad2addrB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsLoad2addrStride64B32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi2, gfx1250::kDsStore2addrB64Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi2, gfx1250::kDsStore2addrStride64B64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsStorexchg2addrRtnB64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsStorexchg2addrStride64RtnB64Vds, RuleAction::Expand,
     0, 0, nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsLoad2addrB64Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsLoad2addrStride64B64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi5, gfx1250::kDsStoreAddtidB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds_addtid, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi5, gfx1250::kDsLoadAddtidB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds_addtid, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadB32Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadB64Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadB128Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB8Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB32Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB64Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB128Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
}};

} // namespace

std::span<const TranslationRule> semantic_expand_rules_gfx1250_b0_to_a0() {
  return kGfx1250B0ToA0ExpandRules;
}

} // namespace rocjitsu
