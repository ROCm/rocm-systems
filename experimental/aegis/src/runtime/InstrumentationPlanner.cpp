#include "aegisbit/InstrumentationPlan.h"

#include "llvm/Support/Error.h"

namespace aegisbit {

llvm::Expected<InstrumentationPlan>
buildInstrumentationPlan(const InstrumentationPlanningFacts &Facts) {
  InstrumentationPlan Plan;
  Plan.Instrumented = Facts.Instrumented;
  Plan.SupportsGPUAtomics = Facts.SupportsGPUAtomics;
  Plan.Payload = Facts.RequestedPayload;

  if (!Facts.Instrumented) {
    Plan.Jump = JumpStrategy::Direct;
    return Plan;
  }

  Plan.Register = Facts.SGPROverflow ? RegisterMode::ZeroSGPR
                                     : RegisterMode::StandardScratch;

  if (Plan.Register == RegisterMode::StandardScratch &&
      Plan.Payload == PayloadStrategy::OnGpuReduce) {
    Plan.Jump = JumpStrategy::SharedBody;
  } else if (Plan.Register == RegisterMode::ZeroSGPR) {
    // Current zero-SGPR behavior still decides direct vs relay per site.
    Plan.Jump = JumpStrategy::Adaptive;
  } else {
    Plan.Jump = JumpStrategy::Direct;
  }

  return validateInstrumentationPlan(Plan);
}

llvm::Expected<InstrumentationPlan>
validateInstrumentationPlan(InstrumentationPlan Plan) {
  if (!Plan.Instrumented)
    return Plan;

  if (Plan.Register == RegisterMode::ZeroSGPR &&
      Plan.Payload != PayloadStrategy::OnGpuReduce) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Zero-SGPR trampoline only supports OnGpuReduce strategy");
  }

  if (Plan.Register == RegisterMode::ZeroSGPR && !Plan.SupportsGPUAtomics) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Zero-SGPR trampoline requires SupportsGPUAtomics (fine-grained memory)");
  }

  return Plan;
}

const char *toString(RegisterMode Mode) {
  switch (Mode) {
  case RegisterMode::StandardScratch:
    return "standard";
  case RegisterMode::ZeroSGPR:
    return "zero-sgpr";
  }
  return "unknown";
}

const char *toString(JumpStrategy Strategy) {
  switch (Strategy) {
  case JumpStrategy::Direct:
    return "direct";
  case JumpStrategy::Relay:
    return "relay";
  case JumpStrategy::SharedBody:
    return "shared-body";
  case JumpStrategy::SwapPCSharedBody:
    return "swappc-shared-body";
  case JumpStrategy::Adaptive:
    return "adaptive";
  }
  return "unknown";
}

} // namespace aegisbit
