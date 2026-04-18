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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

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
  if (srcBits < dstBits)
    v = B.CreateZExt(v, targetTy);
  else if (srcBits > dstBits)
    v = B.CreateTrunc(v, targetTy);
  else if (v->getType() != targetTy)
    v = B.CreateBitCast(v, targetTy);
  Value *laneIdx = emitLaneIdx(B);
  Value *laneIdxExt = B.CreateZExtOrTrunc(laneIdx, targetTy, "vcc_lane_idx");
  Value *shifted = B.CreateLShr(v, laneIdxExt, "vcc_at_lane");
  Value *bit = B.CreateAnd(shifted, ConstantInt::get(targetTy, 1),
                            "vcc_lane_bit");
  return B.CreateICmpNE(bit, ConstantInt::get(targetTy, 0), "vcc_i1");
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

  errs() << "transpiler: WARNING: cross-wave translation of an "
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
            "uniform >= target_wave_bits). Same-wave translation is the "
            "principled path for any other EXEC-divergent kernel.\n";
  return true;
}

} // namespace transpiler
