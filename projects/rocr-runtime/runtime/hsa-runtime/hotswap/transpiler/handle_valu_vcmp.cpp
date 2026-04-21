#include "handle_valu_internal.hpp"

#include "opcode_map.hpp"
#include "sem_op_attrs.hpp"
#include "semop.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

// SPE attribute registrations. V_CMPX is a compare-and-AND-into-EXEC;
// this handler routes the EXEC mutation through `regs.storeExec` after
// folding the ballot through `WaveProjection::ballotI1ToWidth` — see
// the V_CMPX branch below. Audit before adding more entries here.
ArrayRef<SemOpAttrSpec> getHandlerVALU_VcmpAttrs() {
  static constexpr SemOpAttrSpec kAttrs[] = {
      {SemOp::V_CMPX, {/*routesExecThroughStoreExec=*/true}},
  };
  return kAttrs;
}

HandlerResult handleVALU_Vcmp(RaiseContext &ctx, const DecodedInst &di,
                               OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;
  switch (sop) {
  case SemOp::V_CMP:
  case SemOp::V_CMPX:
    break;
  default:
    return hr;
  }

  StringRef mn(di.mnemonic);

  // The ~100 V_CMP_* and V_CMPX_* MC opcodes collapse onto two SemOps
  // (V_CMP, V_CMPX); the per-opcode metadata (ICmp/FCmp predicate,
  // element width, int/float kind) is looked up via `di.vcmp`,
  // populated at decode time by OpcodeMap. That keeps this handler
  // linear in the number of abstract shapes (2) rather than in the
  // number of AMDGPU opcodes.
  const VCmpMeta *m = di.vcmp;
  if (!m) {
    errs() << "transpiler: " << mn
           << ": V_CMP/V_CMPX reached handler without VCmpMeta "
              "(OpcodeMap::build should have populated it)\n";
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VALU", "V_CMP/V_CMPX reached handler without VCmpMeta");
    return hr;
  }

  // ---- v_cmp_class_f<bits> / v_cmpx_class_f<bits> ----
  // Special-cased before the generic predicate-compare dispatch
  // because the second operand is an i32 mask of FP classes, NOT a
  // value to compare against. Lifts to `llvm.amdgcn.class.f<bits>`,
  // which yields one i1 per active lane (the wave-mask plumbing is
  // shared with the predicate-compare path below).
  Value *cmp = nullptr;
  if (m->isClass) {
    Type *fTy = nullptr;
    Value *src0 = nullptr;
    if (m->bits == 16) {
      auto *i16Ty = Type::getInt16Ty(ctx.C);
      fTy = Type::getHalfTy(ctx.C);
      Value *raw0 = op.srcF(0);
      if (raw0->getType() != fTy) {
        if (raw0->getType() != i16Ty)
          raw0 = ctx.B.CreateTrunc(raw0, i16Ty, "vclassf16_lo");
        raw0 = ctx.B.CreateBitCast(raw0, fTy, "vclassf16");
      }
      src0 = raw0;
    } else if (m->bits == 32) {
      fTy = ctx.f32Ty;
      Value *raw0 = op.srcF(0);
      if (raw0->getType() != fTy)
        raw0 = ctx.B.CreateBitCast(raw0, fTy, "vclassf32");
      src0 = raw0;
    } else {
      fTy = Type::getDoubleTy(ctx.C);
      src0 = ctx.B.CreateBitCast(op.src64(0), fTy, "vclassf64");
    }
    Value *mask = op.src(1);
    Function *classFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_class, {fTy});
    cmp = ctx.B.CreateCall(classFn, {src0, mask}, "vclass");
    // fall through to the wave-mask write-back / EXEC-AND logic
    // below by reusing the same `cmp`-driven tail.
  }

  // Fetch operands at the correct width. For 64-bit integer compares we
  // read as i64; for 64-bit float compares we read as i64 and bitcast.
  // For 32-bit float compares we read as i32 and bitcast to f32. For
  // 16-bit float compares we read as i32, truncate to i16, and bitcast
  // to half — required because srcF returns the raw 32-bit operand
  // (e.g. an inline integer immediate -1 = 0xFFFFFFFF for `v_cmpx_lt_
  // f16 vcc, -1, vN`) and CreateFCmp asserts on non-FP operand types.
  Value *s0 = nullptr, *s1 = nullptr;
  if (m->isClass) {
    // Already lifted above; skip the predicate-operand fetch.
  } else if (m->isFloat) {
    if (m->bits == 64) {
      auto *f64Ty = Type::getDoubleTy(ctx.C);
      s0 = ctx.B.CreateBitCast(op.src64(0), f64Ty);
      s1 = ctx.B.CreateBitCast(op.src64(1), f64Ty);
    } else if (m->bits == 32) {
      s0 = op.srcF(0);
      s1 = op.srcF(1);
      if (s0->getType() != ctx.f32Ty)
        s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
      if (s1->getType() != ctx.f32Ty)
        s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    } else {
      auto *i16Ty = Type::getInt16Ty(ctx.C);
      auto *f16Ty = Type::getHalfTy(ctx.C);
      s0 = op.srcF(0);
      s1 = op.srcF(1);
      if (s0->getType() != f16Ty) {
        if (s0->getType() != i16Ty)
          s0 = ctx.B.CreateTrunc(s0, i16Ty, "vcmpf16_lo0");
        s0 = ctx.B.CreateBitCast(s0, f16Ty, "vcmpf16_a");
      }
      if (s1->getType() != f16Ty) {
        if (s1->getType() != i16Ty)
          s1 = ctx.B.CreateTrunc(s1, i16Ty, "vcmpf16_lo1");
        s1 = ctx.B.CreateBitCast(s1, f16Ty, "vcmpf16_b");
      }
    }
  } else {
    if (m->bits == 64) {
      s0 = op.src64(0);
      s1 = op.src64(1);
    } else {
      s0 = op.src(0);
      s1 = op.src(1);
    }
  }
  if (!m->isClass && (!s0 || !s1)) {
    errs() << "transpiler: " << mn << ": missing operand\n";
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VALU", "V_CMP/V_CMPX missing operand");
    return hr;
  }

  if (!m->isClass)
    cmp = m->isFloat ? ctx.B.CreateFCmp(m->pred, s0, s1, "vcmpf")
                     : ctx.B.CreateICmp(m->pred, s0, s1, "vcmp");

  if (sop == SemOp::V_CMPX) {
    // Compare-and-exec: result ANDs into EXEC.
    //
    // The mask MUST be materialised as a wave-level ballot, not a
    // per-lane `sext i1`. `sext` on a divergent `cmp` produces a
    // divergent SSA value: each target lane writes its own private
    // all-ones-or-zero into the EXEC slot, and every subsequent read
    // (notably `emitLaneActiveBit`'s `lshr %exec, %lane_mod`) sees a
    // per-lane "EXEC" instead of the single wave-level mask the SPE
    // model requires. The backend then lowers the SPE diamond as a
    // divergent branch on a per-lane value, narrowing hardware EXEC
    // based on the wrong bit entirely — which surfaces as stores
    // going missing on half the wave in cross-wave lifts (gfx1250
    // wave32 → gfx942 wave64). Routing through `ballotI1ToWidth`
    // matches the VCC read path (`readVCCAsWaveMask`) and keeps EXEC
    // wave-uniform.
    //
    // Width choice. The ballot result feeds directly into the EXEC
    // alloca via AND, so we request it at the EXEC *storage* width
    // (`execTy`). Under modulo-replication `execTy` equals the
    // source wave-mask width and the projection truncates the
    // hardware ballot to match. Under wave-native cross-widening
    // (wave32 source → wave64 target) `execTy` equals the full
    // hardware wave mask (i64) and no truncation occurs — which is
    // what allows a data-dependent `v_cmpx` to preserve its per-
    // target-lane answer on lanes 32..63. See
    // `lit_tests/v_cmpx_ballot` for the pinned IR shape (MODREP)
    // and `lit_tests/v_cmpx_wave_native` for the wave-native shape.
    Value *mask = ctx.projection.ballotI1ToWidth(ctx.B, cmp,
                                                  ctx.regs.execTy,
                                                  "cmpx_ballot");
    Value *curExec = ctx.regs.loadExec(ctx.B);
    ctx.regs.storeExec(ctx.B, ctx.B.CreateAnd(curExec, mask, "cmpx_exec"));
  } else {
    // Vanilla V_CMP: write to SGPR destination (e64 with sdst) or
    // VCC (e32, or e64 whose sdst is VCC).
    if (di.numDefs >= 1) {
      ParsedReg d = op.dst();
      if (d.kind == ParsedReg::SGPR) {
        // Same ballot discipline as V_CMPX: the SGPR destination
        // carries a wave-level mask, not a per-lane predicate. `sext`
        // here would make every downstream consumer that reads the
        // SGPR as a wave mask (`s_and_b64`, `s_mov_b64 exec, …`,
        // `v_cndmask_b32`'s mask input via `readVCCAsWaveMask`) see
        // divergent SSA and silently miscompile.
        //
        // Width choice. The destination is a single SGPR (wave32
        // source) or an SGPR pair (wave64 source) — i.e. *source*
        // wave-mask width, not EXEC storage width. Under modulo-
        // replication these match; under wave-native cross-
        // widening they diverge (execTy=i64 vs sourceWaveMaskTy=
        // i32), and the SGPR physically cannot hold the 64-bit
        // hardware ballot, so we ask the projection for the
        // narrower width explicitly. That takes the trunc-to-
        // source-width branch in
        // `WaveNativeProjection::ballotI1ToWidth`, a documented
        // residual lossy path whose in-BB correctness is restored
        // by the V_CMP -> V_CNDMASK per-lane-i1 shadow recorded
        // below (see `ctx.recordSgprWaveMaskI1`), and whose out-of-
        // BB / scalar-interleaved / other-consumer cases remain the
        // obstruction classifier's responsibility to refuse
        // (wave_size_obstruction.cpp).
        Type *sourceWidth = ctx.projection.sourceWaveMaskTy();
        Value *mask = ctx.projection.ballotI1ToWidth(
            ctx.B, cmp, sourceWidth, "vcmp_ballot");
        ctx.writeRegExecWidth(d, mask);

        // Cache the per-lane `i1` alongside the narrow wave-mask
        // store. The V_CNDMASK_B32 SGPR-source arm in
        // handle_valu_vop3p.cpp looks this up by baseIdx and
        // bypasses the lossy `extractLaneBitFromWaveMask` round-
        // trip when a consumer in the same BB reads the SGPR before
        // any intervening scalar write clobbers it (the latter
        // invalidates via `AllocaRegFile::onSgprWritten` ->
        // `ctx.invalidateSgprWaveMaskI1`). The low-level
        // `storeSGPR*` inside `writeRegExecWidth` above already
        // fired the invalidation hook for this baseIdx; this call
        // restores the fresh `i1` SSA value in the same step. See
        // sgpr-wave-mask-translation.md section 3.1 for the full
        // invariants.
        //
        // Covers BOTH predicate compares (the asin / libdevice-math
        // branch shape) AND class compares
        // (v_cmp_class_f{16,32,64}). The `cmp` value is the same
        // per-lane i1 shape in both arms of this handler — `fcmp`
        // for the predicate-compare path, `llvm.amdgcn.class.f*`
        // for the class path — so caching is sound either way.
        // Gating only the predicate arm would leave class compares
        // under cross-widening miscompiling through the lossy
        // extract fallback for no reason.
        //
        // `isPair` is derived from the destination's ParsedReg
        // width: wave64-source V_CMP_e64 writes an SGPR pair
        // (d.width == 2); wave32-source writes a single SGPR
        // (d.width == 1). The flag is consulted by
        // `ctx.invalidateSgprWaveMaskI1` to decide whether a
        // subsequent write to baseIdx+1 clobbers the pair's high
        // half and should invalidate this entry.
        ctx.recordSgprWaveMaskI1(d.baseIdx, cmp, /*isPair=*/d.width >= 2);
      } else {
        ctx.regs.storeVCC(ctx.B, cmp);
      }
    } else {
      ctx.regs.storeVCC(ctx.B, cmp);
    }
  }
  hr.handled = true;
  return hr;
}

} // namespace transpiler
