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

} // namespace transpiler
