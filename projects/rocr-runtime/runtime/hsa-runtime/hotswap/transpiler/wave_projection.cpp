#include "wave_projection.hpp"

#include "decoded_inst.hpp"
#include "mc_state.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, EXEC_LO, EXEC_HI
#include "Utils/AMDGPUBaseInfo.h"            // AMDGPU::mc2PseudoReg

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace transpiler {

// ----------------------------------------------------------------------------
// WaveProjection base: lane-id derivation is shared across every projection
// that keeps each target lane mapped 1:1 to a hardware lane. Subclasses that
// change that mapping (e.g. a future thread-loop projection) would override.
// ----------------------------------------------------------------------------

Value *WaveProjection::emitLaneIdx(IRBuilder<> &B) const {
  Module *M = B.GetInsertBlock()->getModule();
  Type *i32Ty = B.getInt32Ty();
  Function *mbcntLo = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_mbcnt_lo);
  Value *allOnes = ConstantInt::getSigned(i32Ty, -1);
  Value *zero32 = ConstantInt::get(i32Ty, 0);
  Value *laneId = B.CreateCall(mbcntLo, {allOnes, zero32}, "lane_lo");
  if (waveMaskTy_ != i32Ty) {
    Function *mbcntHi = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::amdgcn_mbcnt_hi);
    laneId = B.CreateCall(mbcntHi, {allOnes, laneId}, "lane_id");
  }
  return laneId;
}

Value *WaveProjection::emitInitialExec(IRBuilder<> &B) const {
  // Default: the architectural boot state of a dispatched wave is
  // "every source lane active", i.e. all-ones in the source-width
  // EXEC storage. Projections that need to DECOUPLE the modeled
  // EXEC from the hardware EXEC (e.g. `WaveNativeProjection` below,
  // which forces hardware EXEC = -1 at entry via
  // `@llvm.amdgcn.init_whole_wave` and stores the captured original
  // per-lane active bit into the alloca) override this hook.
  return ConstantInt::getSigned(execStorageTy(), -1);
}

// ----------------------------------------------------------------------------
// ModuloReplicationProjection.
// ----------------------------------------------------------------------------

Value *
ModuloReplicationProjection::emitLaneActiveBit(IRBuilder<> &B,
                                                Value *execVal) const {
  // Project the target-lane id onto the source EXEC mask under
  // modulo-replication: target lane L is active iff bit `L mod W_src` of
  // the source EXEC mask is set. Same-wave and narrowing cases collapse
  // to the identity because `lane_id < source_wave_bits` already; the
  // modulo is a no-op and the shift happens at source width.
  //
  // Shifting at source width also sidesteps the LLVM-IR poison rule that
  // `lshr iN, M` is poison for M >= N: the pre-modulo clamps the shift
  // into [0, execBits).
  Value *laneId = emitLaneIdx(B);
  Type *execTy = execVal->getType();
  unsigned execBits = execTy->getPrimitiveSizeInBits();
  Value *laneIdInExec = B.CreateZExtOrTrunc(laneId, execTy, "spe_lane_idx");
  // execBits is a power of two (32 or 64), so modulo is bitwise AND.
  Value *laneMod = B.CreateAnd(
      laneIdInExec, ConstantInt::get(execTy, execBits - 1), "spe_lane_mod");
  Value *shifted = B.CreateLShr(execVal, laneMod, "spe_exec_at_lane");
  Value *bit = B.CreateAnd(shifted, ConstantInt::get(execTy, 1),
                            "spe_exec_bit");
  return B.CreateICmpNE(bit, ConstantInt::get(execTy, 0), "spe_lane_active");
}

Value *ModuloReplicationProjection::ballotI1ToWidth(
    IRBuilder<> &B, Value *pred, Type *resultTy, const Twine &name) const {
  assert(pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy_});
  Value *waveMask = B.CreateCall(ballot, {pred}, name);
  unsigned wantedBits = resultTy->getPrimitiveSizeInBits();
  unsigned waveBits = waveMaskTy_->getPrimitiveSizeInBits();
  if (wantedBits == waveBits)
    return waveMask;
  if (wantedBits < waveBits)
    // MODREP: trunc is the modulo-replication projection of the target
    // ballot onto the source wave width.
    return B.CreateTrunc(waveMask, resultTy, name + "_trunc");
  // `wantedBits > waveBits`: wave64 source on wave32 target. No correct
  // modulo-replication projection exists (the wider source wave has
  // lanes that do not exist in the narrower target), so zero-extending
  // would invent bits. Bail so a future wave64→wave32 lift lands here
  // rather than silently miscompiles.
  report_fatal_error(
      "ballotI1ToWidth: wantedBits > waveBits (wave64 source on wave32 "
      "target) has no modulo-replication projection; this direction needs "
      "an explicit policy decision before use");
}

Value *ModuloReplicationProjection::extractLaneBitFromWaveMask(
    IRBuilder<> &B, Value *v) const {
  if (v->getType() == B.getInt1Ty())
    return v;
  Type *i64Ty = B.getInt64Ty();
  if (v->getType()->isPointerTy())
    v = B.CreatePtrToInt(v, i64Ty);
  Type *targetTy = waveMaskTy_;
  unsigned srcBits = v->getType()->getPrimitiveSizeInBits();
  unsigned dstBits = targetTy->getPrimitiveSizeInBits();
  if (srcBits < dstBits) {
    // Cross-widening case: a narrow source-wave-width mask (e.g. the
    // 32-bit result of `ballotI1ToWidth(..., i32)` or a saved VCC-lo
    // read via `loadSGPR32`) has to be widened to the target wave-mask
    // width before the per-lane shift extracts a single bit.  A plain
    // `zext` zeros the upper `dstBits - srcBits` positions, which
    // under wave32 → wave64 makes target lanes 32..63 always read a
    // zero (their `lane_id` shift lands in the zero-padded upper
    // half), and downstream every narrow-mask-guarded `v_cndmask_b32`
    // unconditionally picks its FALSE branch on those lanes.  In the
    // Triton SwiGLU shape (`corpus_swiglu_fp32`) that FALSE branch is
    // the 0x80000000 OOB-sentinel offset used to neutralise masked-
    // out buffer accesses, so all target-wave-upper-half stores land
    // out-of-bounds and the SRD bounds check silently drops them —
    // the observed "half of every target wave64's outputs stay at
    // their zero-initialised value" miscompile.  MODREP's contract
    // (`wave-size-translation.md` §6 / class-"modulo-replication"
    // policy) says target lane L reads bit `L mod W_src` of the source
    // wave's mask, so the right widening *replicates* the narrow mask
    // into the upper half rather than zero-extending.  That matches
    // the `WaveNativeProjection::extractLaneBitFromWaveMask`
    // widen-by-replication path and makes a narrow-mask round-trip on
    // the consumer side symmetric with what both projections'
    // narrow-EXEC writers already do on the producer side; the full-
    // lane-id shift below then correctly selects the replicated bit
    // for every target lane.
    Value *zext = B.CreateZExt(v, targetTy);
    Value *shifted = B.CreateShl(
        zext, ConstantInt::get(targetTy, srcBits), "mask_widen_shl");
    v = B.CreateOr(zext, shifted, "mask_widen_replicate");
  } else if (srcBits > dstBits) {
    v = B.CreateTrunc(v, targetTy);
  } else if (v->getType() != targetTy) {
    v = B.CreateBitCast(v, targetTy);
  }
  Value *laneIdx = emitLaneIdx(B);
  // Twine names are neutral (`mask_*`) rather than `vcc_*`: the helper
  // is called from every consumer that reads a wave mask as a per-lane
  // predicate — the VCC consumer path via `readVCCAsWaveMask` AND the
  // SGPR-source `V_CNDMASK_B32_e64` consumer path in
  // `handle_valu_vop3p.cpp`. Keeping the old `vcc_` prefix would make
  // raised-IR dumps for e.g. the corpus_asin_fp32 kernel print
  // `%vcc_lane_idx` for reads of `s6`, which misleads.
  Value *laneIdxExt = B.CreateZExtOrTrunc(laneIdx, targetTy, "mask_lane_idx");
  Value *shifted = B.CreateLShr(v, laneIdxExt, "mask_at_lane");
  Value *bit = B.CreateAnd(shifted, ConstantInt::get(targetTy, 1),
                            "mask_lane_bit");
  return B.CreateICmpNE(bit, ConstantInt::get(targetTy, 0), "mask_lane_i1");
}

// ----------------------------------------------------------------------------
// WaveNativeProjection — cross-widening (wave32 → wave64).
//
// The base `waveMaskTy_` is already `tgtIsa.isWave32() ? i32 : i64`,
// which on the only supported direction (wave32 source → wave64
// target) is `i64`. We reuse it directly for both the EXEC alloca
// storage and the ballot/lane-active arithmetic so the widths line up
// without any extra casting.
// ----------------------------------------------------------------------------

WaveNativeProjection::WaveNativeProjection(const ISAProfile &srcIsa,
                                             const ISAProfile &tgtIsa,
                                             Type *i32Ty, Type *i64Ty)
    : WaveProjection(srcIsa, tgtIsa, i32Ty, i64Ty) {
  // Restrict to the one translation direction where the wave-native
  // projection's extra invariants are well-defined. Same-wave paths
  // don't need a widened EXEC (ModRep already collapses to identity
  // there), and narrowing (wave64 source → wave32 target) loses lanes
  // regardless of policy — `ModuloReplicationProjection` documents the
  // narrowing bail via `report_fatal_error` in `ballotI1ToWidth`; the
  // wave-native projection is not a second answer for that direction.
  if (!(srcIsa.isWave32() && !tgtIsa.isWave32()))
    report_fatal_error(
        "WaveNativeProjection is defined only for wave32 source → "
        "wave64 target cross-widening; other directions must use "
        "ModuloReplicationProjection (same-wave / narrowing) or a "
        "future ThreadLoopProjection implementation. See hotswap/"
        "docs/wave-size-translation.md \u00a72.2 for the projection "
        "ladder.");
}

Value *WaveNativeProjection::emitInitialExec(IRBuilder<> &B) const {
  // Wave32 → Wave64 cross-widening decouples the hardware EXEC (what
  // the target gfx942 wavefront actually applies) from the modeled
  // source EXEC (what the transpiler's `emitUnderExec` diamonds read
  // through the alloca). At kernel entry we call
  // `@llvm.amdgcn.init_whole_wave`, which:
  //
  //   (1) sets hardware EXEC = -1 (all 64 Wave64 lanes active), and
  //   (2) returns a per-lane i1 whose true-bits form the ORIGINAL
  //       hardware EXEC mask at dispatch time.
  //
  // We ballot (1) back into a wave-width i64 and return that as the
  // value to seed the EXEC alloca with. From this point on the
  // `emitUnderExec` diamonds guard every VGPR write, memory store,
  // LDS op, and atomic through an IR-level `br i1 %lane_active`
  // derived from the alloca — the backend lowers those divergent
  // branches by setting hardware EXEC to the ballot of the
  // per-lane predicate inside each `do` block and restoring to
  // `EXEC = -1` afterwards, so no inactive source lane ever
  // commits a side effect. Between `emitUnderExec` diamonds the
  // hardware EXEC is -1, which is exactly what the WMMA → MFMA
  // cross-lane pipeline in `wmma_lowering.cpp` needs to produce
  // correct per-lane output on all 64 Wave64 lanes.
  //
  // This replaces the prior per-MFMA-output `@llvm.amdgcn.strict.wwm`
  // strategy. See the long comment block on
  // `WaveProjection::emitInitialExec` for the register-allocator
  // pressure argument (`SIPreAllocateWWMRegs` requires dedicated
  // physregs for every vreg inside a WWM bracket, which a 128×128
  // matmul tile cannot satisfy).
  Module *M = B.GetInsertBlock()->getModule();
  Function *initWW = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_init_whole_wave);
  Value *originalActive = B.CreateCall(initWW, {}, "orig_active");
  // Ballot the per-lane i1 back into a wave-width mask. Reuses the
  // projection's own ballot emission so the width selection matches
  // `waveMaskTy_` (i64 on Wave64 target) and the single result-type
  // overload of `llvm.amdgcn.ballot` selected is the backend-
  // supported one for this subtarget.
  return ballotI1ToWidth(B, originalActive, waveMaskTy_, "saved_exec");
}

Value *WaveNativeProjection::emitLaneActiveBit(IRBuilder<> &B,
                                                 Value *execVal) const {
  // Target lane L is active iff bit L of the widened EXEC is set. The
  // widened EXEC storage is `waveMaskTy_` (i64 on wave64 target), so
  // the shift index is the full target lane id (0..63) with no modulo
  // fold; that is the whole point of the wave-native projection
  // relative to `ModuloReplicationProjection::emitLaneActiveBit`,
  // which folds the target lane id into `lane_id mod W_src` and
  // thereby collapses target lanes 0..31 with 32..63.
  Value *laneId = emitLaneIdx(B);
  Type *execTy = execVal->getType();
  assert(execTy == waveMaskTy_ &&
         "WaveNativeProjection requires EXEC storage to match the "
         "target wave mask width; caller must size the alloca via "
         "execStorageTy()");
  Value *laneIdInExec = B.CreateZExtOrTrunc(laneId, execTy, "wn_lane_idx");
  Value *shifted = B.CreateLShr(execVal, laneIdInExec, "wn_exec_at_lane");
  Value *bit = B.CreateAnd(shifted, ConstantInt::get(execTy, 1),
                            "wn_exec_bit");
  return B.CreateICmpNE(bit, ConstantInt::get(execTy, 0), "wn_lane_active");
}

Value *WaveNativeProjection::ballotI1ToWidth(IRBuilder<> &B, Value *pred,
                                              Type *resultTy,
                                              const Twine &name) const {
  assert(pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy_});
  Value *waveMask = B.CreateCall(ballot, {pred}, name);
  unsigned wantedBits = resultTy->getPrimitiveSizeInBits();
  unsigned waveBits = waveMaskTy_->getPrimitiveSizeInBits();
  if (wantedBits == waveBits)
    return waveMask;
  if (wantedBits < waveBits)
    // Narrowing the full target ballot to a source-width scalar loses
    // the upper half (target lanes 32..63). This is the one residual
    // truncation the wave-native projection accepts: source-ISA
    // instructions that name a single 32-bit SGPR destination (e.g.
    // `v_cmp_lt_u32_e64 s4, ...` on wave32) cannot hold a 64-bit
    // mask. The `handle_valu_vcmp.cpp` V_CMPX branch asks for
    // `resultTy = execStorageTy() = waveMaskTy_` and stays at full
    // width; only the V_CMP→SGPR branch asks for the narrower source
    // width and takes this trunc. Kernels that consume the truncated
    // mask as a per-lane wave mask downstream can still miscompile,
    // and such patterns remain the obstruction classifier's
    // responsibility to refuse (see `wave_size_obstruction.cpp`).
    return B.CreateTrunc(waveMask, resultTy, name + "_trunc");
  // `wantedBits > waveBits`: wave32 target hardware ballot requested
  // wider than its native mask. This direction only arises on same-
  // target-wave lifts (not our wave32→wave64 cross-widening), so
  // reaching it under WaveNativeProjection is a raiser bug.
  report_fatal_error(
      "WaveNativeProjection::ballotI1ToWidth: wantedBits > waveBits "
      "is not defined for wave32 source → wave64 target cross-"
      "widening; caller must request resultTy ≤ waveMaskTy");
}

Value *WaveNativeProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                         Value *v) const {
  if (v->getType() == B.getInt1Ty())
    return v;
  Type *i64Ty = B.getInt64Ty();
  if (v->getType()->isPointerTy())
    v = B.CreatePtrToInt(v, i64Ty);
  Type *targetTy = waveMaskTy_;
  unsigned srcBits = v->getType()->getPrimitiveSizeInBits();
  unsigned dstBits = targetTy->getPrimitiveSizeInBits();
  if (srcBits < dstBits) {
    // Source-width wave mask (e.g. a 32-bit SGPR that caught the
    // output of `ballotI1ToWidth(..., i32, ...)` above) is widened
    // back to target width by *replication* so target lane K and
    // K+W_src read the same bit. Under wave-native this is the
    // conservative choice — it matches what the V_CMP→SGPR trunc
    // already implicitly assumed when it picked lanes 0..W_src-1 as
    // canonical — and it keeps `v_cndmask_b32` / `s_and_b64 exec,
    // ..., sN` rounds trips behaving like modulo-replication for
    // the residual save/restore pattern. Replacing replication with
    // a zero-extend would silently deactivate target lanes 32..63
    // whenever the kernel restores EXEC through a 32-bit SGPR; the
    // replication choice is the one that keeps the `v_cmpx →
    // predicated store → s_mov_b32 exec_lo, -1` shape working.
    Value *zext = B.CreateZExt(v, targetTy);
    Value *shifted = B.CreateShl(zext, srcBits);
    v = B.CreateOr(zext, shifted, "wn_mask_widen");
  } else if (srcBits > dstBits) {
    v = B.CreateTrunc(v, targetTy);
  } else if (v->getType() != targetTy) {
    v = B.CreateBitCast(v, targetTy);
  }
  Value *laneIdx = emitLaneIdx(B);
  // Neutral `mask_*` naming parity with the ModRep variant above —
  // same two-caller story (VCC consumer + SGPR-source V_CNDMASK_B32
  // consumer), same reason to avoid the old `wn_vcc_*` identifiers
  // surfacing in raised-IR dumps for kernels whose mask source is a
  // plain SGPR.
  Value *laneIdxExt = B.CreateZExtOrTrunc(laneIdx, targetTy, "wn_mask_lane_idx");
  Value *shifted = B.CreateLShr(v, laneIdxExt, "wn_mask_at_lane");
  Value *bit = B.CreateAnd(shifted, ConstantInt::get(targetTy, 1),
                            "wn_mask_lane_bit");
  return B.CreateICmpNE(bit, ConstantInt::get(targetTy, 0), "wn_mask_lane_i1");
}

// ----------------------------------------------------------------------------
// ThreadLoopProjection — second rung of the coverage ladder described
// in hotswap/docs/wave-size-translation.md §2.2, not yet implemented.
// Every virtual override report_fatal_errors so a build that silently
// instantiates it (e.g. a bad decider branch) surfaces as a loud
// runtime abort rather than wrong code.
// ----------------------------------------------------------------------------

ThreadLoopProjection::ThreadLoopProjection(const ISAProfile &srcIsa,
                                            const ISAProfile &tgtIsa,
                                            Type *i32Ty, Type *i64Ty)
    : WaveProjection(srcIsa, tgtIsa, i32Ty, i64Ty) {
  report_fatal_error(
      "ThreadLoopProjection is a placeholder for the second rung of the "
      "coverage ladder described in hotswap/docs/wave-size-translation.md "
      "\u00a72.2; its emission semantics are not yet implemented. See "
      "ThreadLoopProjection's header comment for the MAINTENANCE protocol "
      "that lands the implementation.");
}

Value *ThreadLoopProjection::emitLaneActiveBit(IRBuilder<> & /*B*/,
                                                Value * /*execVal*/) const {
  report_fatal_error("ThreadLoopProjection::emitLaneActiveBit unimplemented");
}

Value *ThreadLoopProjection::ballotI1ToWidth(IRBuilder<> & /*B*/,
                                              Value * /*pred*/,
                                              Type * /*resultTy*/,
                                              const Twine & /*name*/) const {
  report_fatal_error("ThreadLoopProjection::ballotI1ToWidth unimplemented");
}

Value *ThreadLoopProjection::extractLaneBitFromWaveMask(
    IRBuilder<> & /*B*/, Value * /*v*/) const {
  report_fatal_error(
      "ThreadLoopProjection::extractLaneBitFromWaveMask unimplemented");
}

// ----------------------------------------------------------------------------
// EXEC-writer detection.
// ----------------------------------------------------------------------------

bool instructionWritesEXEC(const DecodedInst &di, const MCState &mc) {
  if (di.defsEXEC)
    return true;
  const MCInstrDesc &desc = mc.instrInfo->get(di.inst.getOpcode());
  for (unsigned i = 0; i < desc.getNumDefs() && i < di.inst.getNumOperands();
       ++i) {
    const MCOperand &mop = di.inst.getOperand(i);
    if (!mop.isReg() || !mop.getReg())
      continue;
    MCRegister reg = AMDGPU::mc2PseudoReg(mop.getReg());
    if (reg == AMDGPU::EXEC || reg == AMDGPU::EXEC_LO ||
        reg == AMDGPU::EXEC_HI)
      return true;
  }
  return false;
}

// ----------------------------------------------------------------------------
// Phase 1.4 cross-wave warning.
// ----------------------------------------------------------------------------

bool emitCrossWaveWarning(const WaveProjection &proj, const MCState &mc,
                          ArrayRef<DecodedInst> insts, StringRef sourceISA,
                          StringRef targetISA) {
  if (proj.sourceIsa().waveSize == proj.targetIsa().waveSize)
    return false;

  const DecodedInst *firstEXECWriter = nullptr;
  for (const DecodedInst &di : insts) {
    if (instructionWritesEXEC(di, mc)) {
      firstEXECWriter = &di;
      break;
    }
  }
  if (!firstEXECWriter)
    return false;

  // Route the legacy warn-only diagnostic through LLVM_DEBUG now that
  // the Phase 1.4.5 classifier (see `wave_size_obstruction.{hpp,cpp}`)
  // owns the gate decision. The structured decider in raiser.cpp emits
  // a precise per-obstruction trace via the same DEBUG_TYPE; this
  // legacy diagnostic remains only as a fallback that surfaces under
  // `-debug-only=wave-projection` when the classifier's trace is not
  // enough context. Enable via `raise_cli -debug-only=wave-projection`
  // or `llvm-opt -debug-only=wave-projection`.
  LLVM_DEBUG({
    dbgs() << "transpiler: WARNING: cross-wave translation of an "
              "EXEC-manipulating kernel relies on modulo-replication, "
              "which is not provably correct in general.\n"
           << "  source ISA wave size: " << proj.sourceIsa().waveSize << " ("
           << sourceISA << ")\n"
           << "  target ISA wave size: " << proj.targetIsa().waveSize << " ("
           << (targetISA.empty() ? sourceISA : targetISA) << ")\n"
           << "  first EXEC-writer: " << firstEXECWriter->rawMnemonic
           << " at offset 0x"
           << format_hex_no_prefix(firstEXECWriter->offset, 4) << "\n"
           << "  rationale: the kernel manipulates EXEC; replicating it "
              "across wave halves will double per-lane side effects in a "
              "way the source author did not specify. Empirically this is "
              "correct for kernels whose EXEC writers are lane-position-"
              "independent (pointwise ops with bounds checks against a "
              "uniform >= target_wave_bits). The Phase 1.4.5 classifier "
              "(wave_size_obstruction.cpp) is the principled path for "
              "deciding between outcome (a)/(b)/(c) per hotswap/docs/"
              "wave-size-translation.md \u00a77.\n";
  });
  return true;
}

} // namespace transpiler
