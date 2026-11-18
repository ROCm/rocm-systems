/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Plain C++ (no hipcc) build of nccl_device/utility.h: covers the host-only branches a HIP TU never compiles.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>  // utility.h needs __forceinline__ when hipcc is the CXX compiler.
#endif
#include "nccl_device/utility.h"

namespace RcclUnitTesting {
namespace {

#if NCCL_DEVICE_COMPILE
// Opt-in CI strictness: when set, a hipcc-built binary must fail here instead of skipping all its coverage.
constexpr const char* kRequirePlainCxxEnvVar = "UT_UTILITY_REQUIRE_PLAIN_CXX";

// A hipcc-configured build compiles .cpp as HIP, where the plain-C++ testAbort overload does not exist.
TEST(NcclUtilityHostOnly, RequiresPlainCxxCompiler) {
  if (std::getenv(kRequirePlainCxxEnvVar) != nullptr) {
    FAIL() << kRequirePlainCxxEnvVar << " is set, but this binary was built with hipcc as CXX, so it tests nothing.";
  }
  GTEST_SKIP() << "built with hipcc as CXX; the !NCCL_DEVICE_COMPILE branches need a plain C++ compiler";
}
#else

namespace nu = nccl::utility;

constexpr uint32_t kMaxSteps = 10000;  // utility.h testAbort's poll interval.

// Calls testAbort until it returns true or kMaxSteps * 3 calls pass; returns the 1-based call index or 0.
uint32_t callsUntilAbort(uint32_t* flag, uint32_t& steps) {
  for (uint32_t call = 1; call <= kMaxSteps * 3; ++call) {
    if (nu::testAbort(flag, steps)) {
      return call;
    }
  }
  return 0;
}

TEST(NcclUtilityHostOnly, TestAbortPollsFlagOncePerInterval) {
  uint32_t flag = 1;
  uint32_t steps = 0;
  // The flag is already set, yet the first kMaxSteps - 1 calls only count.
  for (uint32_t call = 1; call < kMaxSteps; ++call) {
    ASSERT_FALSE(nu::testAbort(&flag, steps)) << "call " << call;
    ASSERT_EQ(steps, call);
  }
  EXPECT_TRUE(nu::testAbort(&flag, steps));
  EXPECT_EQ(steps, 0u);
  // The counter restarts, so the next abort is again a full interval away.
  EXPECT_EQ(callsUntilAbort(&flag, steps), kMaxSteps);
}

TEST(NcclUtilityHostOnly, TestAbortClearOrNullFlagResetsSteps) {
  uint32_t flag = 0;
  uint32_t steps = kMaxSteps - 1;
  EXPECT_FALSE(nu::testAbort(&flag, steps));
  EXPECT_EQ(steps, 0u);
  steps = kMaxSteps - 1;
  EXPECT_FALSE(nu::testAbort(nullptr, steps));
  EXPECT_EQ(steps, 0u);
  EXPECT_EQ(callsUntilAbort(nullptr, steps), 0u);
  // Any non-zero flag value aborts, not only 1.
  flag = 0x80000000u;
  steps = kMaxSteps - 1;
  EXPECT_TRUE(nu::testAbort(&flag, steps));
  // A caller-supplied count at or past the interval polls on the very next call.
  steps = kMaxSteps + 5;
  EXPECT_TRUE(nu::testAbort(&flag, steps));
  EXPECT_EQ(steps, 0u);
  // UINT32_MAX wraps to 0 on increment and so starts a fresh interval.
  steps = 0xFFFFFFFFu;
  EXPECT_FALSE(nu::testAbort(&flag, steps));
  EXPECT_EQ(steps, 0u);
}

// The same integer helpers compile in this mode too; mul32hi/mul64hi take their non-device branch here as well.
TEST(NcclUtilityHostOnly, MathHelpersAvailableWithoutHip) {
  EXPECT_EQ(nu::mul32hi(0xFFFFFFFFu, 0xFFFFFFFFu), 0xFFFFFFFEu);
  EXPECT_EQ(nu::mul64hi(~uint64_t{0}, ~uint64_t{0}), ~uint64_t{0} - 1);
  EXPECT_EQ(nu::idivFast32(1000, 7, nu::idivRcp32(7)), 142u);
  EXPECT_EQ(nu::imodFast64(1000, 7, nu::idivRcp64(7)), 6u);
  EXPECT_EQ(nu::alignUp(13, 8), 16);
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(0xFFFFFFFFu, 0));
}

#endif  // NCCL_DEVICE_COMPILE

}  // namespace
}  // namespace RcclUnitTesting
