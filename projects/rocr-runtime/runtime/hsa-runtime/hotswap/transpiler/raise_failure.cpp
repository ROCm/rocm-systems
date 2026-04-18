#include "raise_failure.hpp"

#include "decoded_inst.hpp"

namespace transpiler {

const char *reasonString(RaiseFailureReason r) {
  switch (r) {
  case RaiseFailureReason::None:                    return "None";
  case RaiseFailureReason::UnsupportedOpcode:       return "UnsupportedOpcode";
  case RaiseFailureReason::UnsupportedShape:        return "UnsupportedShape";
  case RaiseFailureReason::SPEUnsafeExecWriter:
    return "SPE-unmodeled-EXEC-writer";
  case RaiseFailureReason::SMEMKernargMiss:         return "SMEMKernargMiss";
  case RaiseFailureReason::TargetMachineCreationFailed:
    return "TargetMachineCreationFailed";
  case RaiseFailureReason::IRVerificationFailed:
    return "IRVerificationFailed";
  case RaiseFailureReason::CrossWaveLaneIdLeak:
    return "cross-wave-lane-id-leak";
  case RaiseFailureReason::CrossWaveUnrewritableShuffle:
    return "cross-wave-unrewritable-shuffle";
  case RaiseFailureReason::CrossWaveShuffleRewritePending:
    return "cross-wave-shuffle-rewrite-pending";
  case RaiseFailureReason::CrossWaveReplicaRace:
    return "cross-wave-replica-race";
  case RaiseFailureReason::CrossWaveLanePredicatedExec:
    return "cross-wave-lane-predicated-exec";
  }
  return "UnknownRaiseFailureReason";
}

RaiseFailure RaiseFailure::unsupportedShape(const DecodedInst &di,
                                             llvm::StringRef format,
                                             const llvm::Twine &detail) {
  RaiseFailure f;
  f.reason = RaiseFailureReason::UnsupportedShape;
  f.mnemonic = di.mnemonic;
  f.format = format.str();
  f.offset = di.offset;
  f.detail = detail.str();
  return f;
}

RaiseFailure RaiseFailure::smemKernargMiss(const DecodedInst &di) {
  RaiseFailure f;
  f.reason = RaiseFailureReason::SMEMKernargMiss;
  f.mnemonic = di.mnemonic;
  f.format = "SMEM";
  f.offset = di.offset;
  return f;
}

RaiseFailure RaiseFailure::unsupportedOpcode(const DecodedInst &di,
                                              llvm::StringRef format) {
  RaiseFailure f;
  f.reason = RaiseFailureReason::UnsupportedOpcode;
  f.mnemonic = di.mnemonic;
  f.format = format.str();
  f.offset = di.offset;
  return f;
}

RaiseFailure RaiseFailure::speUnsafeExecWriter(const DecodedInst &di) {
  RaiseFailure f;
  f.reason = RaiseFailureReason::SPEUnsafeExecWriter;
  f.mnemonic = di.mnemonic;
  f.format = "SPE-unmodeled-EXEC-writer";
  f.offset = di.offset;
  return f;
}

RaiseFailure RaiseFailure::targetMachineCreationFailed() {
  RaiseFailure f;
  f.reason = RaiseFailureReason::TargetMachineCreationFailed;
  f.format = reasonString(RaiseFailureReason::TargetMachineCreationFailed);
  f.detail = "createTargetMachine returned null";
  return f;
}

RaiseFailure RaiseFailure::irVerificationFailed(const llvm::Twine &err) {
  RaiseFailure f;
  f.reason = RaiseFailureReason::IRVerificationFailed;
  f.format = reasonString(RaiseFailureReason::IRVerificationFailed);
  f.detail = err.str();
  return f;
}

// ----------------------------------------------------------------------------
// Phase 1.4.5 wave-size-obstruction factories. All share the same
// structure: take the refused instruction for mnemonic / offset, and
// a kind-specific detail string for the `detail` field.
// ----------------------------------------------------------------------------

namespace {

RaiseFailure makeCrossWaveFailure(RaiseFailureReason reason,
                                   const DecodedInst &di,
                                   const llvm::Twine &kindDetail) {
  RaiseFailure f;
  f.reason = reason;
  f.mnemonic = di.mnemonic;
  f.format = reasonString(reason);
  f.offset = di.offset;
  f.detail = kindDetail.str();
  return f;
}

} // namespace

RaiseFailure RaiseFailure::crossWaveLaneIdLeak(const DecodedInst &di,
                                                const llvm::Twine &kindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveLaneIdLeak, di,
                               kindDetail);
}

RaiseFailure RaiseFailure::crossWaveUnrewritableShuffle(
    const DecodedInst &di, const llvm::Twine &kindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveUnrewritableShuffle, di, kindDetail);
}

RaiseFailure RaiseFailure::crossWaveShuffleRewritePending(
    const DecodedInst &di, const llvm::Twine &kindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveShuffleRewritePending, di, kindDetail);
}

RaiseFailure RaiseFailure::crossWaveReplicaRace(const DecodedInst &di,
                                                 const llvm::Twine &kindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveReplicaRace, di,
                               kindDetail);
}

RaiseFailure RaiseFailure::crossWaveLanePredicatedExec(
    const DecodedInst &di, const llvm::Twine &kindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveLanePredicatedExec, di, kindDetail);
}

} // namespace transpiler
