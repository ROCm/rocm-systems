#include "aegisbit/InstrumentationPlan.h"
#include "gtest/gtest.h"
#include "llvm/Support/Error.h"

using namespace aegisbit;

TEST(InstrumentationPlannerGTest, SelectsSharedBodyForStandardOnGpuReduce) {
  InstrumentationPlanningFacts Facts;
  Facts.Instrumented = true;
  Facts.SGPROverflow = false;
  Facts.SupportsGPUAtomics = true;
  Facts.RequestedPayload = PayloadStrategy::OnGpuReduce;

  auto PlanOrErr = buildInstrumentationPlan(Facts);
  ASSERT_TRUE(static_cast<bool>(PlanOrErr));
  EXPECT_EQ(PlanOrErr->Register, RegisterMode::StandardScratch);
  EXPECT_EQ(PlanOrErr->Jump, JumpStrategy::SharedBody);
}

TEST(InstrumentationPlannerGTest, SelectsAdaptiveForZeroSgprInstrumented) {
  InstrumentationPlanningFacts Facts;
  Facts.Instrumented = true;
  Facts.SGPROverflow = true;
  Facts.SupportsGPUAtomics = true;
  Facts.RequestedPayload = PayloadStrategy::OnGpuReduce;

  auto PlanOrErr = buildInstrumentationPlan(Facts);
  ASSERT_TRUE(static_cast<bool>(PlanOrErr));
  EXPECT_EQ(PlanOrErr->Register, RegisterMode::ZeroSGPR);
  EXPECT_EQ(PlanOrErr->Jump, JumpStrategy::Adaptive);
}

TEST(InstrumentationPlannerGTest, RejectsIllegalZeroSgprFullCapture) {
  InstrumentationPlan Plan;
  Plan.Instrumented = true;
  Plan.Register = RegisterMode::ZeroSGPR;
  Plan.Payload = PayloadStrategy::FullCapture;
  Plan.Jump = JumpStrategy::Adaptive;
  Plan.SupportsGPUAtomics = true;

  auto PlanOrErr = validateInstrumentationPlan(Plan);
  EXPECT_FALSE(static_cast<bool>(PlanOrErr));
  if (!PlanOrErr)
    llvm::consumeError(PlanOrErr.takeError());
}
