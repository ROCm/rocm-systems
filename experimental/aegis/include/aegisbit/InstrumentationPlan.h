#ifndef AEGISBIT_INSTRUMENTATION_PLAN_H
#define AEGISBIT_INSTRUMENTATION_PLAN_H

#include "aegisbit/Types.h"
#include "llvm/Support/Error.h"

namespace aegisbit {

/// Register resources available to the trampoline implementation.
enum class RegisterMode {
  StandardScratch,
  ZeroSGPR,
};

/// High-level control-flow shape used to reach trampoline code.
enum class JumpStrategy {
  Direct,
  Relay,
  SharedBody,
  SwapPCSharedBody,

  /// Planner defers the final per-site choice to lower layers.
  Adaptive,
};

/// Facts gathered before the trampoline is built.
struct InstrumentationPlanningFacts {
  bool Instrumented = false;
  bool SGPROverflow = false;
  bool SupportsGPUAtomics = false;
  PayloadStrategy RequestedPayload = PayloadStrategy::OnGpuReduce;
};

/// Typed summary of how a kernel should be instrumented.
struct InstrumentationPlan {
  bool Instrumented = false;
  RegisterMode Register = RegisterMode::StandardScratch;
  PayloadStrategy Payload = PayloadStrategy::OnGpuReduce;
  JumpStrategy Jump = JumpStrategy::Direct;
  bool SupportsGPUAtomics = false;
};

llvm::Expected<InstrumentationPlan>
buildInstrumentationPlan(const InstrumentationPlanningFacts &Facts);

llvm::Expected<InstrumentationPlan>
validateInstrumentationPlan(InstrumentationPlan Plan);

const char *toString(RegisterMode Mode);
const char *toString(JumpStrategy Strategy);

} // namespace aegisbit

#endif // AEGISBIT_INSTRUMENTATION_PLAN_H
