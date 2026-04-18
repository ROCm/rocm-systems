#include "handle_valu_internal.hpp"

#include "semop.hpp"

#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

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

  // ---- v_permlane16 / v_permlanex16 / v_permlane64 ----
  // KNOWN LIMITATION (CROSS_LANE_SURVEY P2/P3): these lower to plain
  // same-lane moves today. The select1/select2/op_sel lane-remap is
  // dropped. Safe only for kernels that happen to feed uniform
  // operands (so the result is lane-invariant anyway).
  case SemOp::V_PERMLANE16_B32:
  case SemOp::V_PERMLANEX16_B32:
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
