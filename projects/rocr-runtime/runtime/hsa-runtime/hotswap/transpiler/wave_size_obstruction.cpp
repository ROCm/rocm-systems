#include "wave_size_obstruction.hpp"

#include "decoded_inst.hpp"
#include "isa_profile.hpp"
#include "mc_state.hpp"
#include "semop.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::OpName
#include "Utils/AMDGPUBaseInfo.h"             // AMDGPU::getNamedOperandIdx
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// SIInstrFlags::DPP / SDWA bits live in the AMDGPU target's SIDefines
// header. We only use the flag constant; no other target dependency.
#include "SIDefines.h"

namespace transpiler {

using namespace llvm;

// ----------------------------------------------------------------------------
// Taxonomy rendering. The text in each branch is the label that
// surfaces in the classifier trace and that lit tests assert on. The
// SPE_DESIGN.md §3 Class number is included parenthetically so
// operators reading the trace can cross-reference the design doc
// without mental translation.
// ----------------------------------------------------------------------------

const char *obstructionKindName(ObstructionKind k) {
  switch (k) {
  case ObstructionKind::None:
    return "None";
  case ObstructionKind::MbcntHiLaneIdLeak:
    return "MbcntHiLaneIdLeak (\u00a73 Class 1: absolute lane-ID leak via v_mbcnt_hi)";
  case ObstructionKind::OutOfRangeLaneOperand:
    return "OutOfRangeLaneOperand (\u00a73 Class 1: readlane/writelane operand >= W_s)";
  case ObstructionKind::FullWaveRotate:
    return "FullWaveRotate (\u00a73 Class 2: unrewritable v_permlane64)";
  case ObstructionKind::LaneGroupShuffle:
    return "LaneGroupShuffle (\u00a73 Class 2: permlane16 / permlanex16 / permlane*_swap)";
  case ObstructionKind::DsSwizzle:
    return "DsSwizzle (\u00a73 Class 2: ds_swizzle_b32)";
  case ObstructionKind::DppCrossLane:
    return "DppCrossLane (\u00a73 Class 2: DPP modifier)";
  case ObstructionKind::DsBpermuteGather:
    return "DsBpermuteGather (\u00a73 Class 2: ds_bpermute_b32)";
  case ObstructionKind::NonCommutativeAtomic:
    return "NonCommutativeAtomic (\u00a73 Class 3: cmpswap/swap/xchg, replica race)";
  case ObstructionKind::CmpxFromLaneId:
    return "CmpxFromLaneId (\u00a73 Class 4: lane-predicated v_cmpx)";
  case ObstructionKind::SaveExecFromLaneId:
    return "SaveExecFromLaneId (\u00a73 Class 4: lane-predicated s_*_saveexec_b32)";
  }
  return "UnknownObstructionKind";
}

const char *rewriteIdName(RewriteId r) {
  switch (r) {
  case RewriteId::None:
    return "none";
  case RewriteId::P1_DsBpermute:
    return "P1 (llvm.amdgcn.ds.bpermute)";
  case RewriteId::P2_PermLane16:
    return "P2 (llvm.amdgcn.permlane16)";
  case RewriteId::P3_PermLane64:
    return "P3 (reserved: v_permlane64 has no rewrite)";
  case RewriteId::P4_PermLaneSwap:
    return "P4 (permlane*_swap via LDS round-trip or permlane16 pair)";
  case RewriteId::P5_DppModifier:
    return "P5 (llvm.amdgcn.update.dpp)";
  case RewriteId::P6_DsSwizzle:
    return "P6 (llvm.amdgcn.ds.swizzle)";
  case RewriteId::LaneOpBoundsValidator:
    return "raise-time readlane/writelane bounds validator";
  }
  return "UnknownRewriteId";
}

// ----------------------------------------------------------------------------
// ObstructionReport queries.
// ----------------------------------------------------------------------------

bool ObstructionReport::hasUnrewritable() const {
  for (const auto &s : sites)
    if (s.rewrite == RewriteId::None)
      return true;
  return false;
}

bool ObstructionReport::hasPendingRewrite() const {
  for (const auto &s : sites)
    if (s.rewrite != RewriteId::None && !s.rewriteImplemented)
      return true;
  return false;
}

bool ObstructionReport::isOblivious() const {
  for (const auto &s : sites)
    if (!s.rewriteImplemented)
      return false;
  return true;
}

const ObstructionSite *ObstructionReport::firstUnrewritable() const {
  for (const auto &s : sites)
    if (s.rewrite == RewriteId::None)
      return &s;
  return nullptr;
}

const ObstructionSite *ObstructionReport::firstPending() const {
  for (const auto &s : sites)
    if (s.rewrite != RewriteId::None && !s.rewriteImplemented)
      return &s;
  return nullptr;
}

// ----------------------------------------------------------------------------
// Classification primitives.
// ----------------------------------------------------------------------------

namespace {

// Extract the lane operand of a v_readlane / v_writelane instruction.
// Returns std::nullopt when the operand is register-typed (dynamic) or
// when LLVM's named-operand index lookup fails (e.g. on a future LLVM
// version that renames `src1`).
//
// We use `AMDGPU::getNamedOperandIdx(opcode, AMDGPU::OpName::src1)`
// which is the authoritative way to find a named MCInst operand and
// is robust to operand-layout reordering between LLVM versions.
// Implementation note: both v_readlane_b32 (dst, src0, src1) and
// v_writelane_b32 (vdst, src0, src1, vdst_in) name the lane operand
// `src1`, so a single getNamedOperandIdx call covers both.
std::optional<int64_t> extractLaneOperandImm(const DecodedInst &di) {
  const MCInst &inst = di.inst;
  int idx = AMDGPU::getNamedOperandIdx(inst.getOpcode(), AMDGPU::OpName::src1);
  if (idx < 0 || static_cast<unsigned>(idx) >= inst.getNumOperands())
    return std::nullopt;
  const MCOperand &op = inst.getOperand(idx);
  if (op.isImm())
    return op.getImm();
  return std::nullopt; // dynamic SGPR operand — cannot statically prove range
}

// Decide whether a `ds_swizzle_b32` immediate encodes a swizzle mode
// that is *structurally* wave-size-oblivious under modulo-replication
// (SPE_DESIGN.md §7's coverage-ladder rung 1).
//
// The 16-bit imm encodes one of seven modes (SIDefines.h
// `Swizzle::Id`). Per AMDGPU SIDefines.h `Swizzle::EncBits`:
//
//   QUAD_PERM_ENC    == 0x8000, QUAD_PERM_ENC_MASK    == 0xFF00
//   BITMASK_PERM_ENC == 0x0000, BITMASK_PERM_ENC_MASK == 0x8000
//   FFT_MODE_ENC     == 0xE000  (FFT_ROTATE_MODE_MASK  == 0xF000)
//   ROTATE_MODE_ENC  == 0xC000  (FFT_ROTATE_MODE_MASK  == 0xF000)
//
// Within each envelope, only specific sub-encodings are *valid*:
//
//   QUAD_PERM     — bits 0..7  encode four 2-bit lane selectors.
//                   All 256 imms in [0x8000, 0x80FF] are valid.
//
//   BITMASK_PERM  — bits 0..4  AND mask (0..31)
//                   bits 5..9  OR mask  (0..31)
//                   bits 10..14 XOR mask (0..31)
//                   bit 15      = 0 (envelope discriminator)
//                   All 32K imms in [0x0000, 0x7FFF] are valid.
//
//   FFT_MODE      — bits 0..4   FFT_SWIZZLE_MASK (0..31)
//                   bits 5..11  reserved, MUST be 0
//                   bits 12..15 = 0xE (envelope discriminator)
//                   Valid imms: exactly the 32 in [0xE000, 0xE01F].
//
//   ROTATE_MODE   — bits 0..4   reserved, MUST be 0
//                   bits 5..9   ROTATE_SIZE_MASK (0..31)
//                   bit  10     ROTATE_DIR_MASK  (0|1)
//                   bit  11     reserved, MUST be 0
//                   bits 12..15 = 0xC (envelope discriminator)
//                   Valid imms: exactly the 64 with bits 0..4=0,
//                   bit 11=0, top nibble=0xC.
//
// All four valid-encoding sub-spaces are wave-size-oblivious under
// modulo-replication. The argument is the same for all four: wave64
// ds_swizzle hardware preserves bit 5 of the lane id, so each
// 32-lane half independently performs the same permutation.
// Preservation verified empirically on gfx942 (CDNA3) wave64 across
// the four envelopes (see the MODREP block in handle_ds.cpp for the
// per-envelope probe results and the LLVM-upstream-test cross-
// citation that lifts the property to the wave64 GPU family).
//
// REFUSED imms (any of):
//   * RESERVED top-nibble envelopes (top nibble in
//     {0x9, 0xA, 0xB, 0xD, 0xF}) — `Swizzle::EncBits` assigns no
//     semantics; hardware behavior is undefined.
//   * FFT_MODE imms with reserved bits 5..11 set — within the FFT
//     envelope but outside the valid sub-encoding space; hardware
//     behavior likewise undefined.
//   * ROTATE_MODE imms with reserved bits 0..4 or 11 set — same
//     argument.
//
// Refusing these (rather than silently passing them through to
// `llvm.amdgcn.ds.swizzle`, which would emit `ds_swizzle_b32
// offset:<imm>` and let the wave64 hardware do whatever it does
// for an undefined imm) matches the no-fallback contract.
//
// Check ordering note: the four valid-encoding checks are pairwise
// disjoint on the discriminator bits (bit 15 separates QUAD_PERM /
// BITMASK_PERM; the top nibble separates FFT_MODE / ROTATE_MODE
// from QUAD_PERM and from each other), so the order between them
// is irrelevant for correctness. The implicit "no match → return
// false" branch correctly catches all the REFUSED categories.
bool dsSwizzleSafeForModRep(uint16_t imm) {
  using namespace AMDGPU::Swizzle;
  if ((imm & QUAD_PERM_ENC_MASK) == QUAD_PERM_ENC)
    return true;
  if ((imm & BITMASK_PERM_ENC_MASK) == BITMASK_PERM_ENC)
    return true;
  // FFT_MODE: discriminator (top nibble = 0xE) AND every reserved
  // bit (5..11) clear. Equivalent to: imm & ~FFT_SWIZZLE_MASK ==
  // FFT_MODE_ENC, with the mask cast to uint16_t to keep the
  // bitwise-not within the 16-bit envelope (otherwise `~uint16_t`
  // promotes to int and would set bits 16..31 of the comparison
  // operand).
  if ((imm & static_cast<uint16_t>(~FFT_SWIZZLE_MASK)) == FFT_MODE_ENC)
    return true;
  // ROTATE_MODE: discriminator (top nibble = 0xC) AND only the size
  // (bits 5..9) and direction (bit 10) bits set. Variable-bit mask
  // is (ROTATE_SIZE_MASK << ROTATE_SIZE_SHIFT) | (ROTATE_DIR_MASK
  // << ROTATE_DIR_SHIFT) = 0x7E0; everything else (including
  // reserved bits 0..4 and 11) must match ROTATE_MODE_ENC exactly.
  constexpr uint16_t ROTATE_VAR_MASK =
      (ROTATE_SIZE_MASK << ROTATE_SIZE_SHIFT) |
      (ROTATE_DIR_MASK << ROTATE_DIR_SHIFT);
  if ((imm & static_cast<uint16_t>(~ROTATE_VAR_MASK)) == ROTATE_MODE_ENC)
    return true;
  // RESERVED top-nibble envelope, FFT/ROTATE with reserved bits
  // set, or any other non-valid encoding: refuse.
  return false;
}

} // namespace

// ----------------------------------------------------------------------------
// buildObstructionReport — the main walk.
// ----------------------------------------------------------------------------

ObstructionReport buildObstructionReport(ArrayRef<DecodedInst> insts,
                                          const MCState & /*mc*/,
                                          const ISAProfile &src,
                                          const ISAProfile &tgt) {
  ObstructionReport report;
  if (src.waveSize == tgt.waveSize)
    return report;

  // First pass: tag the self-contained obstruction kinds (lane-id
  // leaks, cross-lane shuffles, replica races). Also collect
  // co-occurrence state for the lane-predicated EXEC check below.
  //
  // The walk matches purely on `SemOp` and `MCInstrDesc` TSFlags
  // bits; there is no string matching on `rawMnemonic`. New
  // obstruction triggers should be added by extending semop.hpp +
  // opcode_map.cpp (so the lookup is a single enum compare here),
  // not by adding `raw.contains(...)` substring tests.
  bool haveMbcnt = false;
  struct PendingExecSite {
    const DecodedInst *inst;
    ObstructionKind kind; // CmpxFromLaneId or SaveExecFromLaneId.
  };
  llvm::SmallVector<PendingExecSite> pendingExecWriters;

  for (const DecodedInst &di : insts) {
    const SemOp sop = di.semOp;

    // --- §3 Class 1: absolute lane-ID leaks --------------------------
    if (sop == SemOp::V_MBCNT_HI_U32_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::MbcntHiLaneIdLeak;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail = "v_mbcnt_hi reads target exec_hi — no wave32 semantics "
                    "to preserve under modulo-replication";
      report.sites.push_back(std::move(site));
      haveMbcnt = true;
      continue;
    }
    if (sop == SemOp::V_MBCNT_LO_U32_B32) {
      // v_mbcnt_lo alone is not a leak by itself (wave32 sources use
      // it as the canonical lane-id probe and it's lane-position-
      // independent inside its wave). We track its presence only
      // for the lane-predicated-EXEC co-occurrence heuristic below.
      haveMbcnt = true;
      continue;
    }
    if (sop == SemOp::V_READLANE_B32 || sop == SemOp::V_WRITELANE_B32) {
      auto imm = extractLaneOperandImm(di);
      // Negative-value guard: an int64_t imm cast to uint64_t for the
      // bounds compare wraps around to a value > waveSize for any
      // negative value, which is the correct logical answer (negative
      // lane indices are never in [0, W_s)) but is implicit in the
      // cast. Make it explicit so the intent survives a refactor.
      if (imm.has_value() &&
          (*imm < 0 ||
           static_cast<uint64_t>(*imm) >= src.waveSize)) {
        // Static constant operand provably out of source wave range.
        // No rewrite preserves the semantics on a wider target wave.
        ObstructionSite site;
        site.inst = &di;
        site.kind = ObstructionKind::OutOfRangeLaneOperand;
        site.rewrite = RewriteId::None;
        site.rewriteImplemented = false;
        std::string det;
        raw_string_ostream os(det);
        os << "operand value " << *imm << " out of [0, W_s=" << src.waveSize
           << ")";
        site.detail = os.str();
        report.sites.push_back(std::move(site));
      }
      // Otherwise (static imm < W_s, or dynamic operand): do not
      // emit a site.
      //
      // - Static imm < W_s: provably in-bounds, safe by construction.
      // - Dynamic operand (SGPR): we cannot statically prove the
      //   runtime value is < W_s, BUT we also cannot prove it is out
      //   of bounds. Triton's softmax / matmul patterns (see
      //   gpt-oss-derisking.md §7.1) use `v_writelane_b32` with
      //   dynamic lane operands that are in-bounds at runtime but not
      //   statically provable. Flagging those as refusal would
      //   collapse coverage on every Gfx1250Gpu.* test that uses
      //   them.
      //
      // TODO(dataflow-upgrade): graduate dynamic operands from "not
      // flagged" to "proved via LLVM uniformity / value-range
      // analysis on the raised IR" once the post-raise dataflow
      // analysis lands. Today this is a sound-not-complete choice
      // toward false negatives on readlane/writelane specifically —
      // tracked in wave_size_obstruction.hpp's TODO block.
      continue;
    }

    // --- §3 Class 2: wave-width-specific cross-lane shuffles --------
    if (sop == SemOp::V_PERMLANE64_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::FullWaveRotate;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail = "v_permlane64 has no wave32 analogue";
      report.sites.push_back(std::move(site));
      continue;
    }
    if (sop == SemOp::V_PERMLANE16_B32 ||
        sop == SemOp::V_PERMLANEX16_B32 ||
        sop == SemOp::V_PERMLANE16_SWAP_B32 ||
        sop == SemOp::V_PERMLANE32_SWAP_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::LaneGroupShuffle;
      // P2 covers base permlane16/permlanex16; P4 covers the swap
      // variants. `rewriteImplemented` is set per-SemOp so the
      // decider emits a precise "cross-wave-shuffle-rewrite-pending"
      // diagnostic for the specific P-item still missing.
      //
      // P2 (base permlane16/permlanex16) landed:
      // `handle_valu_cross_lane.cpp` emits
      // `llvm.amdgcn.permlane16` / `permlanex16` with fi / bc
      // extracted from the VOP3 src{0,1}_modifiers' OP_SEL_0 bit.
      // P4 (permlane*_swap) is still pending — the swap variants
      // need an LDS-round-trip lowering or a permlane16-pair
      // selector, per CROSS_LANE_SURVEY.md P4's design sketch.
      if (sop == SemOp::V_PERMLANE16_SWAP_B32 ||
          sop == SemOp::V_PERMLANE32_SWAP_B32) {
        site.rewrite = RewriteId::P4_PermLaneSwap;
        site.rewriteImplemented = false;
      } else {
        site.rewrite = RewriteId::P2_PermLane16;
        site.rewriteImplemented = true;
      }
      report.sites.push_back(std::move(site));
      continue;
    }
    if (sop == SemOp::DS_SWIZZLE_B32) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::DsSwizzle;
      site.rewrite = RewriteId::P6_DsSwizzle;
      // P6 (CROSS_LANE_SURVEY.md item 6) landed: the handler in
      // handle_ds.cpp emits `llvm.amdgcn.ds.swizzle(value, offset)`
      // with the 16-bit immediate plumbed through. The lift is only
      // wave-size-oblivious for the QUAD_PERM and BITMASK_PERM
      // sub-modes (see `dsSwizzleSafeForModRep` above for the
      // structural argument); FFT_MODE / ROTATE_MODE / unknown-high-
      // nibble imms remain pending and refuse loudly via the same
      // CrossWaveShuffleRewritePending channel as before.
      //
      // The 16-bit imm is extracted at decode time into
      // `di.dsSwizzleImm` (see `decode.cpp::decodeDsSwizzleImm`).
      // `!di.hasDsSwizzleImm` here means the decoder rejected the
      // operand (missing, non-immediate, or out of 16-bit range);
      // we mirror that rejection as a P6-pending refusal with a
      // malformed-disassembly diagnostic so the kernel fails loudly
      // rather than the handler inventing a value.
      if (!di.hasDsSwizzleImm) {
        site.rewriteImplemented = false;
        site.detail = "ds_swizzle_b32 missing/invalid OpName::offset "
                      "immediate operand — disassembly malformed or "
                      "outside 16-bit range";
      } else {
        site.rewriteImplemented = dsSwizzleSafeForModRep(di.dsSwizzleImm);
        if (!site.rewriteImplemented) {
          std::string det;
          raw_string_ostream os(det);
          os << "ds_swizzle_b32 imm 0x"
             << format_hex_no_prefix(di.dsSwizzleImm, 4)
             << " is not a valid swizzle encoding (not QUAD_PERM, "
                "BITMASK_PERM, valid FFT_MODE, or valid ROTATE_MODE) "
                "— RESERVED top-nibble or FFT/ROTATE reserved bits "
                "set; AMDGPU hardware semantics undefined";
          site.detail = os.str();
        }
      }
      report.sites.push_back(std::move(site));
      continue;
    }
    // DPP detection via the MCInstrDesc TSFlags bit. opcode_map.cpp
    // canonicalises DPP variants down to their base SemOp, so the
    // SemOp alone cannot identify them, but `di.tsFlags` is captured
    // from the *original* MCInstrDesc (see decode.cpp) so the DPP
    // bit is still visible. Same for SDWA — though SDWA is same-lane
    // and not a cross-wave concern, so we don't flag it.
    if (di.tsFlags & SIInstrFlags::DPP) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::DppCrossLane;
      site.rewrite = RewriteId::P5_DppModifier;
      // P5 (DPP16 lift via `llvm.amdgcn.update.dpp`) landed for the
      // DPP16 encoding family. `decodeDppModifiers` sets
      // `di.hasDpp = true` only after successfully extracting every
      // DPP16 modifier operand (dpp_ctrl / row_mask / bank_mask /
      // bound_ctrl); for DPP8 instructions it early-returns and leaves
      // `hasDpp` false. Gate `rewriteImplemented` on `hasDpp` so DPP16
      // accepts (outcome b) while DPP8 refuses loudly (outcome c
      // pending a P5 extension to `llvm.amdgcn.mov.dpp8`). The
      // `tsFlags & SIInstrFlags::DPP` check still fires for both
      // forms — both are a Class-2 cross-lane site by SPE_DESIGN.md
      // taxonomy; the flipped-by-form `rewriteImplemented` bit is
      // what separates "handled" from "pending" without mucking with
      // the taxonomy.
      site.rewriteImplemented = di.hasDpp;
      if (!di.hasDpp)
        site.detail =
            "DPP8 lane-permutation form — P5 currently covers only the "
            "DPP16 encoding family via llvm.amdgcn.update.dpp; extending "
            "to DPP8 requires an llvm.amdgcn.mov.dpp8 lift path.";
      report.sites.push_back(std::move(site));
      continue;
    }
    if (sop == SemOp::DS_BPERMUTE_B32) {
      // P1 is IMPLEMENTED in handle_ds.cpp (see lit_tests/ds_bpermute_b32).
      // Record the site so the trace shows it, but mark as
      // `rewriteImplemented = true` so the decider treats it as
      // outcome (a)/(b) rather than refusal.
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::DsBpermuteGather;
      site.rewrite = RewriteId::P1_DsBpermute;
      site.rewriteImplemented = true;
      report.sites.push_back(std::move(site));
      continue;
    }

    // --- §3 Class 3: replica races on shared state ------------------
    // The SemOp set here is the complete enumeration of
    // non-commutative atomics modeled in semop.hpp today. New
    // non-commutative atomic encodings (e.g. SCRATCH_ATOMIC_SWAP if
    // we ever model it) should be added by extending the enum +
    // opcode_map.cpp + semop.cpp, not by adding a substring check
    // here. Atomics not yet modeled refuse via the existing Phase 5
    // unsupportedOpcode path.
    if (sop == SemOp::GLOBAL_ATOMIC_SWAP ||
        sop == SemOp::GLOBAL_ATOMIC_CMPSWAP ||
        sop == SemOp::FLAT_ATOMIC_SWAP ||
        sop == SemOp::FLAT_ATOMIC_CMPSWAP ||
        sop == SemOp::BUFFER_ATOMIC_SWAP ||
        sop == SemOp::BUFFER_ATOMIC_CMPSWAP ||
        sop == SemOp::S_ATOMIC_SWAP) {
      ObstructionSite site;
      site.inst = &di;
      site.kind = ObstructionKind::NonCommutativeAtomic;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail = "non-commutative atomic races target lanes "
                    "i and i+W_s under modulo-replication";
      report.sites.push_back(std::move(site));
      continue;
    }

    // --- §3 Class 4: lane-predicated EXEC writers -------------------
    // The principled check is "does the EXEC-writer's source operand
    // chain contain a value derived from mbcnt?". The syntactic
    // approximation: collect EXEC writers now, decide after the walk
    // based on kernel-level mbcnt presence.
    if (sop == SemOp::V_CMPX) {
      pendingExecWriters.push_back({&di, ObstructionKind::CmpxFromLaneId});
      continue;
    }
    if (sop == SemOp::S_AND_SAVEEXEC_B32 ||
        sop == SemOp::S_OR_SAVEEXEC_B32 ||
        sop == SemOp::S_XOR_SAVEEXEC_B32 ||
        sop == SemOp::S_ANDN2_SAVEEXEC_B32 ||
        sop == SemOp::S_ORN2_SAVEEXEC_B32) {
      pendingExecWriters.push_back({&di, ObstructionKind::SaveExecFromLaneId});
      continue;
    }
  }

  // Second pass: for each pending EXEC writer, apply the syntactic
  // co-occurrence heuristic. If the kernel contains ANY mbcnt (lo or
  // hi), treat the writer as lane-predicated and unrewritable.
  // Otherwise it is a lane-position-INDEPENDENT bounds-check style
  // writer (the overwhelming common case per gpt-oss-derisking.md
  // §7.6) and does NOT produce a site — the outer Phase 1.4 legacy
  // diagnostic (now LLVM_DEBUG-only) still logs it for auditability.
  //
  // TODO(dataflow-upgrade): replace this co-occurrence heuristic with
  // a precise dataflow check once the classifier runs post-raise on
  // the LLVM IR. See wave_size_obstruction.hpp's TODO block.
  if (haveMbcnt) {
    for (const auto &pw : pendingExecWriters) {
      ObstructionSite site;
      site.inst = pw.inst;
      site.kind = pw.kind;
      site.rewrite = RewriteId::None;
      site.rewriteImplemented = false;
      site.detail =
          "v_cmpx / saveexec co-occurs with v_mbcnt_* in the same kernel — "
          "syntactic over-approximation of 'gating expression flows from an "
          "absolute-lane-id value'. Dataflow upgrade may reclassify this as "
          "wave-size-oblivious (see TODO(dataflow-upgrade)).";
      report.sites.push_back(std::move(site));
    }
  }

  return report;
}

// ----------------------------------------------------------------------------
// Rendering — stable-enough-for-lit trace format.
// ----------------------------------------------------------------------------

std::string renderObstructionTrace(const ObstructionReport &report,
                                    StringRef kernelName, StringRef srcIsa,
                                    StringRef tgtIsa, unsigned srcWaveSize,
                                    unsigned tgtWaveSize) {
  std::string out;
  raw_string_ostream os(out);

  os << "transpiler: projection decision for kernel '" << kernelName << "':\n";
  os << "  source: " << srcIsa << " (wave" << srcWaveSize
     << ") -> target: " << tgtIsa << " (wave" << tgtWaveSize << "), R="
     << (srcWaveSize > 0 ? tgtWaveSize / srcWaveSize : 0) << "\n";

  if (report.sites.empty()) {
    os << "  obstructions found: none\n"
       << "  outcome: (a) wave-size-oblivious — emit modulo-replication\n";
    return out;
  }

  os << "  obstructions found:\n";
  for (const ObstructionSite &s : report.sites) {
    os << "    " << obstructionKindName(s.kind);
    if (s.inst) {
      os << " @ 0x" << format_hex_no_prefix(s.inst->offset, 4) << ": "
         << s.inst->rawMnemonic;
    }
    os << "\n      rewrite: " << rewriteIdName(s.rewrite);
    if (s.rewrite != RewriteId::None)
      os << " [" << (s.rewriteImplemented ? "implemented" : "pending") << "]";
    if (!s.detail.empty())
      os << "\n      detail: " << s.detail;
    os << "\n";
  }

  if (report.hasUnrewritable()) {
    os << "  outcome: (c) refuse — at least one obstruction has no rewrite "
          "in SPE_DESIGN.md \u00a74's rewrite table\n";
  } else if (report.hasPendingRewrite()) {
    os << "  outcome: (c) refuse — rewrite(s) exist on paper but the "
          "matching handler(s) have not yet landed (CROSS_LANE_SURVEY.md)\n";
  } else {
    os << "  outcome: (b) rewrite-then-emit — all obstruction sites have "
          "an implemented rewrite; emit modulo-replication\n";
  }
  return out;
}

// ----------------------------------------------------------------------------
// Failure selection — pick the first refusal-worthy site and package
// it as a RaiseFailure for raiser.cpp to propagate.
// ----------------------------------------------------------------------------

RaiseFailure selectFailureFromReport(const ObstructionReport &report) {
  // Prefer unrewritable over pending — the caller should see the
  // strongest refusal reason first. Ties broken by decoded order (the
  // `sites` vector is in decoded order, so `firstUnrewritable` /
  // `firstPending` both return the earliest match).
  //
  // Twine lifetime: each `Twine(...) + ... + ...` chain is built and
  // consumed in the SAME full-expression as the factory call below.
  // This is the LLVM-supported lifetime contract — Twine concat
  // results are temporaries that hold references into their operands
  // and *must not* be bound to a named variable (`const Twine x = a +
  // b + c` would leave `x` referencing temporaries that are destroyed
  // at the end of that statement). See the LLVM Programmer's Manual
  // on Twine.
  if (const ObstructionSite *site = report.firstUnrewritable()) {
    switch (site->kind) {
    case ObstructionKind::MbcntHiLaneIdLeak:
    case ObstructionKind::OutOfRangeLaneOperand:
      return RaiseFailure::crossWaveLaneIdLeak(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    case ObstructionKind::FullWaveRotate:
      return RaiseFailure::crossWaveUnrewritableShuffle(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    case ObstructionKind::NonCommutativeAtomic:
      return RaiseFailure::crossWaveReplicaRace(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    case ObstructionKind::CmpxFromLaneId:
    case ObstructionKind::SaveExecFromLaneId:
      return RaiseFailure::crossWaveLanePredicatedExec(
          *site->inst,
          Twine(obstructionKindName(site->kind)) + " [" + site->detail + "]");
    // The kinds below never set `rewrite = None` under
    // buildObstructionReport, so they cannot reach firstUnrewritable().
    // If they do, our state is inconsistent — fail loudly rather than
    // silently fall through to the empty-RaiseFailure return below.
    case ObstructionKind::LaneGroupShuffle:
    case ObstructionKind::DsSwizzle:
    case ObstructionKind::DppCrossLane:
    case ObstructionKind::DsBpermuteGather:
    case ObstructionKind::None:
      llvm_unreachable("ObstructionKind classified as unrewritable but "
                       "buildObstructionReport never tags it that way");
    }
    llvm_unreachable("unhandled ObstructionKind in selectFailureFromReport "
                     "(unrewritable branch)");
  }
  if (const ObstructionSite *site = report.firstPending()) {
    return RaiseFailure::crossWaveShuffleRewritePending(
        *site->inst,
        Twine(obstructionKindName(site->kind)) + " [rewrite " +
            rewriteIdName(site->rewrite) + " pending]");
  }
  // Oblivious / fully-rewritten: no failure.
  return RaiseFailure();
}

} // namespace transpiler
