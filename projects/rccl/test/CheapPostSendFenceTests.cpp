/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for rcclComputeCheapPostSendFenceOff(), the host-side decision
// for comm->cheapPostSendFenceOff. The return value gates whether the LL128 /
// simple post-send path issues the full __threadfence_system() (return 1, cheap
// fence OFF) or the cheap signal-only fence (return 0, cheap fence ON).
//
// Behavior under test:
//   * cheap fence is only safe when cache-bypassing builtins are available
//     (uncachedMemSupported); otherwise it must always be OFF.
//   * RCCL_CHEAP_POST_SEND_FENCE_OFF param: 0 = arch-tuned auto, 1 = force off,
//     2 = force on (override auto).
//   * auto (0) enables the cheap fence on gfx942/gfx1250 only; gfx950 and every
//     other arch default to the full fence.

#include "gtest/gtest.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

// Numeric device archs (comm->cudaArch = 100*major + 10*minor).
static constexpr int kGfx942 = 940;
static constexpr int kGfx950 = 950;
static constexpr int kGfx1250 = 1250;
static constexpr int kGfx90a = 900;

// Param values.
static constexpr int64_t kAuto = 0;
static constexpr int64_t kForceOff = 1;
static constexpr int64_t kForceOn = 2;

TEST(CheapPostSendFenceTests, UncachedMemUnsupportedAlwaysOff)
{
    // Without cache-bypass builtins the cheap fence is unsafe: always OFF (1),
    // regardless of arch or param (including the force-on override).
    for (int64_t param : {kAuto, kForceOff, kForceOn, (int64_t)3}) {
        EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx942, param, /*uncached=*/false), 1);
        EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx950, param, /*uncached=*/false), 1);
        EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx1250, param, /*uncached=*/false), 1);
    }
}

TEST(CheapPostSendFenceTests, AutoModeArchTuned)
{
    // Auto: cheap fence ON (0) for gfx942/gfx1250, OFF (1) for gfx950 and others.
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx942, kAuto, true), 0);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx1250, kAuto, true), 0);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx950, kAuto, true), 1);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx90a, kAuto, true), 1);
}

TEST(CheapPostSendFenceTests, ForceOffOverridesEveryArch)
{
    // param == 1 forces the full fence even where auto would enable the cheap one.
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx942, kForceOff, true), 1);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx950, kForceOff, true), 1);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx1250, kForceOff, true), 1);
}

TEST(CheapPostSendFenceTests, ForceOnOverridesEveryArch)
{
    // param == 2 forces the cheap fence ON regardless of arch, notably
    // re-enabling it on gfx950 where auto disables it.
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx942, kForceOn, true), 0);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx950, kForceOn, true), 0);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx1250, kForceOn, true), 0);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx90a, kForceOn, true), 0);
}

TEST(CheapPostSendFenceTests, OtherNonZeroParamForcesOff)
{
    // Any non-zero value other than 2 behaves like force-off (1).
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx942, /*param=*/3, true), 1);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(kGfx1250, /*param=*/-1, true), 1);
}

TEST(CheapPostSendFenceTests, UnknownArchDefaultsOff)
{
    // An unrecognized arch must default to the full fence under auto/force-off,
    // while force-on still returns ON (0).
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(/*cudaArch=*/0, kAuto, true), 1);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(/*cudaArch=*/0, kForceOff, true), 1);
    EXPECT_EQ(rcclComputeCheapPostSendFenceOff(/*cudaArch=*/0, kForceOn, true), 0);
}

} // namespace RcclUnitTesting
