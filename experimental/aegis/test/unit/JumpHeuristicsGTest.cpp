#include "aegisbit/JumpHeuristics.h"
#include "aegisbit/RuntimeConfig.h"
#include "gtest/gtest.h"

#include <cstdlib>

using namespace aegisbit;

namespace {

InstrumentationPlan makeSharedBodyPlan() {
  InstrumentationPlan Plan;
  Plan.Instrumented = true;
  Plan.Register = RegisterMode::StandardScratch;
  Plan.Payload = PayloadStrategy::OnGpuReduce;
  Plan.Jump = JumpStrategy::SharedBody;
  Plan.SupportsGPUAtomics = true;
  return Plan;
}

} // namespace

TEST(JumpHeuristicsGTest, ForcesRelayForLargeZeroSgprIslandEstimate) {
  InstrumentationPlan Plan;
  Plan.Instrumented = true;
  Plan.Register = RegisterMode::ZeroSGPR;
  Plan.Payload = PayloadStrategy::OnGpuReduce;
  Plan.Jump = JumpStrategy::Adaptive;
  Plan.SupportsGPUAtomics = true;

  EXPECT_TRUE(shouldForceAllRelay(Plan, 300, 0, 4096));
}

TEST(JumpHeuristicsGTest, DoesNotForceRelayForStandardScratch) {
  InstrumentationPlan Plan;
  Plan.Instrumented = true;
  Plan.Register = RegisterMode::StandardScratch;
  Plan.Payload = PayloadStrategy::OnGpuReduce;
  Plan.Jump = JumpStrategy::SharedBody;
  Plan.SupportsGPUAtomics = true;

  EXPECT_FALSE(shouldForceAllRelay(Plan, 300, 0, 4096));
}

TEST(JumpHeuristicsGTest, ComputesPreKernelPaddingOnlyForLargeZeroSgprKernels) {
  InstrumentationPlan Plan;
  Plan.Instrumented = true;
  Plan.Register = RegisterMode::ZeroSGPR;
  Plan.Payload = PayloadStrategy::OnGpuReduce;
  Plan.Jump = JumpStrategy::Adaptive;
  Plan.SupportsGPUAtomics = true;

  EXPECT_GT(computePreKernelPadSize(Plan, 4096, 150 * 1024), 0u);
  EXPECT_EQ(computePreKernelPadSize(Plan, 300 * 1024, 150 * 1024), 0u);
}

// Regression: a small kernel sitting near offset 0 in a dense multi-kernel
// .text must still be upgraded to SwapPC, because the island will land near
// the end of .text -- far beyond s_call_b64's ±128 KB range.
TEST(JumpHeuristicsGTest, SwapPCUpgradeTriggeredBySpanNotKernelSize) {
  unsetenv("AEGISBIT_FORCE_SWAPPC");
  RuntimeConfig::initialize();

  InstrumentationPlan Plan = makeSharedBodyPlan();

  // 20 KB kernel, but .text is 1 MB (many neighboring kernels). BaseAddr=0,
  // TextSectionSize=1 MB: span from kernel start to island end is ~1 MB.
  EXPECT_TRUE(shouldUseSwapPCSharedBody(Plan,
                                        /*TextSectionSize=*/1 * 1024 * 1024,
                                        /*BaseAddr=*/0,
                                        /*SiteCount=*/100));
}

// Counter-case: a small kernel in an isolated (small) code object stays on
// the cheaper SharedBody path -- no gratuitous SwapPC upgrade.
TEST(JumpHeuristicsGTest, NoSwapPCUpgradeForIsolatedSmallKernel) {
  unsetenv("AEGISBIT_FORCE_SWAPPC");
  RuntimeConfig::initialize();

  InstrumentationPlan Plan = makeSharedBodyPlan();

  // 20 KB kernel starting at offset 0 in a 30 KB .text: span fits easily
  // under the 120 KB threshold.
  EXPECT_FALSE(shouldUseSwapPCSharedBody(Plan,
                                         /*TextSectionSize=*/30 * 1024,
                                         /*BaseAddr=*/0,
                                         /*SiteCount=*/100));
}

// ForceSwapPC env var short-circuits regardless of span.
TEST(JumpHeuristicsGTest, ForceSwapPCOverridesSpanCheck) {
  setenv("AEGISBIT_FORCE_SWAPPC", "1", /*overwrite=*/1);
  RuntimeConfig::initialize();

  InstrumentationPlan Plan = makeSharedBodyPlan();

  EXPECT_TRUE(shouldUseSwapPCSharedBody(Plan,
                                        /*TextSectionSize=*/4096,
                                        /*BaseAddr=*/0,
                                        /*SiteCount=*/1));

  unsetenv("AEGISBIT_FORCE_SWAPPC");
  RuntimeConfig::initialize();
}
