#include "handle_valu_internal.hpp"

#include "semop.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::OpName
#include "SIDefines.h"                        // SISrcMods::OP_SEL_0
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

// Cross-lane VALU primitives — the subset of VALU opcodes whose result
// in lane L depends on values held by lane L' != L. Isolated from the
// rest of handleVALU because this is exactly the surface the cross-
// wave strategy (SPE_DESIGN.md §4 / CROSS_LANE_SURVEY.md) keeps
// iterating on: every rewrite from the "wave-size-baked cross-lane"
// rewrite table lands in this file, not scattered through the VALU
// arithmetic sections.
//
// Each branch MUST use a genuine cross-lane LLVM intrinsic
// (`llvm.amdgcn.readlane`, `writelane`, `readfirstlane`, `mbcnt.{lo,
// hi}`, etc.). A "same-lane" stub that ignores the source-lane
// selector is a silent miscompile for any kernel that feeds divergent
// operands into the primitive. Several permlane variants here are
// known broken (see CROSS_LANE_SURVEY.md items P2..P5); they stay
// same-lane for now but any new cross-lane SemOp must be modelled
// correctly before landing.
HandlerResult handleVALU_CrossLane(RaiseContext &ctx, const DecodedInst &di,
                                    OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  switch (sop) {

  // ---- v_permlane16_b32 / v_permlanex16_b32 ----
  // CROSS_LANE_SURVEY.md P2 lowering. Target constraint: `v_permlane16`
  // and `v_permlanex16` are RDNA/gfx10+ instructions and DO NOT exist
  // on CDNA (gfx9/gfx94x). Emitting `llvm.amdgcn.permlane16` or
  // `permlanex16` directly fails isel on gfx942 with "Cannot select:
  // intrinsic %llvm.amdgcn.permlanex16". We therefore emulate both
  // via `ds_bpermute_b32`, which IS available on every AMDGPU
  // generation with LDS (gfx8+), so this lowering is target-
  // independent — it works for gfx1250 → gfx942, gfx1250 → gfx1250,
  // and any future target with ds_bpermute.
  //
  // MCInst operand layout (from VOP3_PERMLANE_Profile's InsVOP3OpSel):
  //
  //   [0] vdst (output)             [5] src2_modifiers (always 0)
  //   [1] src0_modifiers  <-- fi    [6] src2 (SSrc_b32)  = selector_2
  //   [2] src0 (VRegSrc_32) = val   [7] vdst_in (VGPR, tied) = %old
  //   [3] src1_modifiers  <-- bc    [8] op_sel (VOP3OpSel imm, unused
  //   [4] src1 (SSrc_b32)  = sel_1                                here)
  //
  // Selector encoding: src1 and src2 are each 32-bit scalar values
  // containing 8 × 4-bit per-lane selectors. src1 covers within-
  // group lanes 0..7, src2 covers within-group lanes 8..15. Each
  // 4-bit nibble selects a source lane within the 16-lane group.
  //
  // Per-lane emulation, L = `mbcnt`-derived absolute lane id (0..W_t):
  //
  //   group_base  = L & ~0xF           // 16, 32, 48 boundaries
  //   within      = L & 0xF            // 0..15
  //   within_lo   = within & 7         // 0..7 (nibble index)
  //   sel_word    = within < 8 ? src1 : src2
  //   nibble      = (sel_word >> (within_lo * 4)) & 0xF
  //
  //   permlane16  : src_group = group_base
  //   permlanex16 : src_group = group_base ^ 0x10  (swap adjacent groups)
  //
  //   src_lane_abs = src_group | nibble
  //   byte_addr    = src_lane_abs << 2
  //   result       = ds_bpermute(byte_addr, src0)
  //
  // Wave-width correctness under modulo-replication (SPE_DESIGN.md
  // §2): the source gfx1250 kernel is wave32 so its selector values
  // encode a shuffle pattern over 2 × 16-lane groups. On wave64
  // target each modrep replica occupies 2 × 16-lane groups (R=2),
  // and the `group ^ 0x10` swap stays within a replica (0↔1 within
  // replica 0, 2↔3 within replica 1), so the modrep invariant is
  // preserved for permlanex16. permlane16 keeps every lane within
  // its own group, trivially within-replica.
  //
  // Handling of `fi` (fetch-invalid) and `bc` (bound_ctrl) — the two
  // i1 immediates encoded via `opsel_i1timm` in PermlanePat
  // (`SISrcMods::OP_SEL_0` bit of src0_modifiers / src1_modifiers):
  //
  //   - `fi=1`: on an EXEC-inactive source lane, the kernel still
  //     fetches that lane's VGPR value (possibly stale). This is
  //     exactly how `llvm.amdgcn.ds.bpermute` behaves naturally
  //     (the LDS-backed path reads the VGPR alloca regardless of
  //     EXEC), so `fi=1` is supported directly.
  //   - `bc=0`: on an "out-of-range" source lane, the target lane
  //     retains %old. For permlane16 the 4-bit selector nibble is
  //     always in [0, 16) so the source lane is always in-group;
  //     `bc=0` is the only case the emulation needs to support.
  //     Under SPE, `writeReg32`'s `emitUnderExec` already retains
  //     prior VDST values on EXEC-masked target lanes, covering the
  //     "target lane inactive" direction of `bc=0`.
  //   - `fi=0` and `bc=1` diverge from the above in ways the
  //     emulation does not model. Every GPT-OSS / softmax /
  //     bitmatrix disassembly we have examined uses `op_sel:[1, 0]`
  //     (fi=1, bc=0); refusing the other combinations keeps the
  //     classifier-gate's "no silent miscompile" invariant intact
  //     rather than emitting ds_bpermute with fi=0 semantics it
  //     does not provide.
  //
  // Future optimisation: on targets that DO support native
  // permlane16 (gfx10+), emit the intrinsic directly for lower
  // latency. Left as a profitability refinement — correctness-first
  // lands the ds_bpermute emulation.
  case SemOp::V_PERMLANE16_B32:
  case SemOp::V_PERMLANEX16_B32: {
    const bool isPermlaneX16 = (sop == SemOp::V_PERMLANEX16_B32);
    const bool fi = (op.srcMod(0) & SISrcMods::OP_SEL_0) != 0;
    const bool bc = (op.srcMod(1) & SISrcMods::OP_SEL_0) != 0;
    if (!fi || bc) {
      // Empirically the GPT-OSS / softmax / bitmatrix corpora emit
      // `op_sel:[1, 0]` exclusively (fi=1, bc=0). Refuse any other
      // encoding loudly so a future corpus kernel's extended
      // fi/bc use surfaces during classifier verification rather
      // than producing an approximation silently. Re-narrowing this
      // gate is the right place to extend the emulation.
      std::string detail;
      raw_string_ostream os(detail);
      os << "permlane16 / permlanex16 emulation supports only "
            "op_sel:[1,0] (fi=1, bc=0); saw fi="
         << (fi ? 1 : 0) << ", bc=" << (bc ? 1 : 0);
      hr.failure = RaiseFailure::unsupportedShape(di, "VALU", detail);
      return hr;
    }
    Value *src0 = op.src(0);
    Value *sel1 = op.src(1);
    Value *sel2 = op.src(2);

    // Target-hardware lane id, wave-width-aware via emitLaneIdx.
    Value *laneId = ctx.projection.emitLaneIdx(ctx.B);

    // Group base (lane & ~0xF) and within-group index (lane & 0xF).
    Value *groupBase = ctx.B.CreateAnd(laneId, ctx.B.getInt32(~0xF), "pl_group");
    Value *within = ctx.B.CreateAnd(laneId, ctx.B.getInt32(0xF), "pl_within");

    // Pick the right 32-bit selector word based on within's high bit.
    Value *isHiHalf = ctx.B.CreateICmpUGE(within, ctx.B.getInt32(8), "pl_hi");
    Value *selWord = ctx.B.CreateSelect(isHiHalf, sel2, sel1, "pl_sel");

    // Extract the 4-bit nibble at position (within & 7) * 4.
    Value *withinLo = ctx.B.CreateAnd(within, ctx.B.getInt32(7), "pl_lo");
    Value *shiftAmt = ctx.B.CreateShl(withinLo, ctx.B.getInt32(2), "pl_shift");
    Value *shifted = ctx.B.CreateLShr(selWord, shiftAmt, "pl_shifted");
    Value *nibble = ctx.B.CreateAnd(shifted, ctx.B.getInt32(0xF), "pl_nibble");

    // For permlanex16, XOR the group base by 0x10 to swap adjacent groups.
    Value *srcGroup = isPermlaneX16
        ? ctx.B.CreateXor(groupBase, ctx.B.getInt32(0x10), "plx_group")
        : groupBase;
    Value *srcLaneAbs = ctx.B.CreateOr(srcGroup, nibble, "pl_src_lane");
    Value *byteAddr = ctx.B.CreateShl(srcLaneAbs, ctx.B.getInt32(2), "pl_addr");

    // Convergent: emit the bpermute outside any emitUnderExec diamond
    // so all hardware lanes participate. writeReg32 below wraps the
    // store for EXEC masking.
    Function *bperm = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_ds_bpermute);
    Value *result = ctx.B.CreateCall(
        bperm, {byteAddr, src0},
        isPermlaneX16 ? "permlanex16_emu" : "permlane16_emu");
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }

  // ---- v_permlane64_b32 ----
  // KNOWN LIMITATION (CROSS_LANE_SURVEY P3): no wave32 analogue, so
  // the Phase 1.4.5 classifier refuses this op in any cross-wave
  // lift (it is taxonomised as FullWaveRotate / unrewritable). The
  // same-lane fallback here only runs in same-wave (wave64 → wave64)
  // translation, where a gfx1250 binary would not contain the op
  // anyway (gfx942 and earlier do not emit it). Keeping the stub
  // prevents a silent raise failure on the theoretical case.
  case SemOp::V_PERMLANE64_B32: {
    if (di.numDefs >= 1 && di.numSrcs >= 1)
      ctx.writeReg32(op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }

  // ---- gfx950 lane-swap: v_permlane{16,32}_swap_b32 ----
  // Exchange vdst and src0 across lanes 0..15↔16..31 (or 0..31↔32..63).
  // Two defs (vdst, src0_out) and two uses (vdst_in, src0). Scalar
  // model swaps the two VGPR values same-lane — KNOWN LIMITATION
  // (CROSS_LANE_SURVEY P4): the cross-lane exchange is dropped.
  case SemOp::V_PERMLANE16_SWAP_B32:
  case SemOp::V_PERMLANE32_SWAP_B32: {
    ParsedReg dstReg = op.dst();
    Value *oldDst = ctx.regs.readReg32(ctx.B, dstReg);
    Value *oldSrc = op.src(0);
    ctx.writeReg32(dstReg, oldSrc);
    int src0OutIdx = AMDGPU::getNamedOperandIdx(
        di.inst.getOpcode(), AMDGPU::OpName::src0_out);
    if (src0OutIdx >= 0 && di.isReg(src0OutIdx)) {
      ParsedReg src0Out = ctx.parseReg(di.getReg(src0OutIdx), src0OutIdx);
      ctx.writeReg32(src0Out, oldDst);
    }
    hr.handled = true;
    return hr;
  }

  // ---- v_readfirstlane_b32 sDST, vSRC ----
  // Broadcast the value of vSRC from the lowest-numbered active lane
  // (or lane 0 if EXEC==0) to sDST. A plain per-lane move would leave
  // each lane with its own vSRC, breaking the uniformity invariant
  // that downstream consumers (s_mov, s_load base, branch condition)
  // rely on. Emit the native `llvm.amdgcn.readfirstlane.i32`; the
  // SGPR dst is NOT wrapped in emitUnderExec (see
  // `RaiseContext::writeReg32` — SGPR writes bypass predication), so
  // the broadcast lands on all lanes.
  case SemOp::V_READFIRSTLANE_B32: {
    Function *rfl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_readfirstlane, {ctx.i32Ty});
    Value *src = ctx.B.CreateZExtOrTrunc(op.src(0), ctx.i32Ty, "rfl_src");
    Value *val = ctx.B.CreateCall(rfl, {src}, "readfirstlane");
    ctx.writeReg32(op.dst(), val);
    hr.handled = true;
    return hr;
  }

  // ---- v_writelane_b32 ----
  // Write `val` into lane `lane` of vDst. Cross-lane: cannot be
  // emulated via per-thread private scratch nor via a single scalar
  // SSA value. `llvm.amdgcn.writelane(val, lane, old)` lowers to the
  // hardware primitive; the intrinsic returns the new per-lane scalar
  // (either `val` when lane_id==lane, else `old`), so the VGPR's
  // SSA slot carries the correct value for whichever lane we are.
  //
  // First-write pattern: if writelane is the first assignment to
  // vDst, non-selected lanes legitimately hold whatever vDst
  // contained before (hardware semantics). `readReg32` on the
  // never-stored alloca returns LLVM `undef`, which is the right
  // "unobservable" encoding — any downstream use of those lanes
  // before they are written is itself undefined on hardware.
  case SemOp::V_WRITELANE_B32: {
    ParsedReg dst = op.dst();
    Value *val = op.src(0);
    Value *lane = op.src(1);
    lane = ctx.B.CreateZExtOrTrunc(lane, ctx.i32Ty, "wrlane_idx");
    Value *oldVal = ctx.regs.readReg32(ctx.B, dst);
    Function *wl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_writelane, {ctx.i32Ty});
    Value *newVal = ctx.B.CreateCall(wl, {val, lane, oldVal}, "writelane");
    ctx.writeReg32(dst, newVal);
    hr.handled = true;
    return hr;
  }

  // ---- v_readlane_b32 sDST, vSRC, lane ----
  // Read a specific lane of vSRC into an SGPR. Reverse of writelane;
  // cross-lane so must use the native intrinsic.
  case SemOp::V_READLANE_B32: {
    ParsedReg srcReg = op.srcReg(0);
    Value *lane = op.src(1);
    lane = ctx.B.CreateZExtOrTrunc(lane, ctx.i32Ty, "rdlane_idx");
    Value *src = ctx.regs.readReg32(ctx.B, srcReg);
    Function *rl = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_readlane, {ctx.i32Ty});
    Value *val = ctx.B.CreateCall(rl, {src, lane}, "readlane");
    ctx.writeReg32(op.dst(), val);
    hr.handled = true;
    return hr;
  }

  // ---- v_mbcnt_lo_u32_b32 / v_mbcnt_hi_u32_b32 ----
  // Count set bits in src0 below the current lane. Wave-size-aware
  // (mbcnt.hi only meaningful on wave64). These are the building
  // blocks for lane-id derivation that `WaveProjection::emitLaneIdx`
  // also uses; the raw intrinsics must be passed through exactly.
  case SemOp::V_MBCNT_LO_U32_B32: {
    Function *mbcnt = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_mbcnt_lo, {});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateCall(mbcnt, {op.src(0), op.src(1)},
                                    "mbcnt_lo"));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_MBCNT_HI_U32_B32: {
    Function *mbcnt = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_mbcnt_hi, {});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateCall(mbcnt, {op.src(0), op.src(1)},
                                    "mbcnt_hi"));
    hr.handled = true;
    return hr;
  }

  default:
    break;
  }
  return hr;
}

} // namespace transpiler
