#include "handle_valu_internal.hpp"

#include "opcode_map.hpp"
#include "sem_op_attrs.hpp"
#include "semop.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
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

  // Fetch operands at the correct width. For 64-bit integer compares we
  // read as i64; for 64-bit float compares we read as i64 and bitcast.
  // 16- and 32-bit values both come through the 32-bit reader; we only
  // bitcast the 32-bit float case to f32 (matches prior behaviour: F16
  // is left as whatever `srcF` returns so the low 16 bits drive the
  // compare).
  Value *s0 = nullptr, *s1 = nullptr;
  if (m->isFloat) {
    if (m->bits == 64) {
      auto *f64Ty = Type::getDoubleTy(ctx.C);
      s0 = ctx.B.CreateBitCast(op.src64(0), f64Ty);
      s1 = ctx.B.CreateBitCast(op.src64(1), f64Ty);
    } else {
      s0 = op.srcF(0);
      s1 = op.srcF(1);
      if (m->bits == 32) {
        if (s0->getType() != ctx.f32Ty)
          s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
        if (s1->getType() != ctx.f32Ty)
          s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
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
  if (!s0 || !s1) {
    errs() << "transpiler: " << mn << ": missing operand\n";
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VALU", "V_CMP/V_CMPX missing operand");
    return hr;
  }

  Value *cmp = m->isFloat ? ctx.B.CreateFCmp(m->pred, s0, s1, "vcmpf")
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
    // based on the wrong bit entirely — which surfaces as stores going
    // missing on half the wave in cross-wave lifts (gfx1250 wave32 →
    // gfx942 wave64). Routing through `ballotI1ToWidth` matches the
    // VCC read path (`readVCCAsWaveMask`) and keeps EXEC wave-uniform.
    //
    // MODREP: cross-wave (wave32 → wave64) takes the truncation path
    // inside `WaveProjection::ballotI1ToWidth`, which picks lanes
    // 0..31 of the target ballot under modulo-replication. Valid only
    // while the target-lane-K / target-lane-K+sourceBits predicates
    // agree — the precondition enforced by the Phase-1.4 cross-wave
    // gate in `raiser.cpp`. If the gate policy changes, revisit.
    Value *mask = ctx.projection.ballotI1ToWidth(ctx.B, cmp,
                                                  ctx.regs.execTy,
                                                  "cmpx_ballot");
    Value *curExec = ctx.regs.loadExec(ctx.B);
    ctx.regs.storeExec(ctx.B, ctx.B.CreateAnd(curExec, mask, "cmpx_exec"));
  } else {
    // Vanilla V_CMP: write to SGPR-pair destination (e64 with sdst) or
    // VCC (e32, or e64 whose sdst is VCC).
    if (di.numDefs >= 1) {
      ParsedReg d = op.dst();
      if (d.kind == ParsedReg::SGPR) {
        // Same ballot discipline as V_CMPX: the SGPR-pair destination
        // carries a wave-level mask, not a per-lane predicate. `sext`
        // here would make every downstream consumer that reads the
        // SGPR pair as a wave mask (`s_and_b64`, `s_mov_b64 exec, …`,
        // `v_cndmask_b32`'s mask input via `readVCCAsWaveMask`) see
        // divergent SSA and silently miscompile.
        //
        // MODREP: same modulo-replication contract as the V_CMPX
        // branch above. See `wave_projection.hpp::ballotI1ToWidth`
        // for the policy; grep for MODREP when revisiting cross-wave.
        Value *mask = ctx.projection.ballotI1ToWidth(
            ctx.B, cmp, ctx.regs.execTy, "vcmp_ballot");
        ctx.writeRegExecWidth(d, mask);
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
