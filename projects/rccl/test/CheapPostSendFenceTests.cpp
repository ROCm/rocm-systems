/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for rcclComputeCheapPostSendFenceOff(), the host-side decision
// for comm->cheapPostSendFenceOff. The return value gates whether the Simple
// post-send path issues the full __threadfence_system() (return 1, cheap fence
// OFF) or the cheap signal-only fence (return 0, cheap fence ON). LL128 honors it
// only where FIFO stores are system scope (gfx1250) and otherwise always uses the
// full fence.
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
#include "rccl_ptr.h"

namespace RcclUnitTesting
{

// This TU is host code, so rccl_ptr.h takes the no-builtins arm. That is the
// arm prims_simple.h compiles when DWORDX4_INTRINSICS is OFF.
static_assert(RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS == 0,
              "host test must compile the no-builtins post-send fence rule");

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

// Stand-in for the ncclDevWorkColl fields rcclSimplePostSendRegisteredOrP2p reads.
// The fixture TU cannot include device.h. Dropping regUsed or netRegUsed from
// the helper fails the test below.
struct SimplePostSendRegFlags {
    int regUsed;
    int netRegUsed;
};

// RCCL_USE_CHEAP_POST_SEND_FENCE is what prims_simple.h assigns to skip_fence,
// and rcclSimplePostSendRegisteredOrP2p is the predicate it passes. Host
// compilation takes the no-builtins arm (the macro is 0 outside
// __HIP_DEVICE_COMPILE__), which is the arm this change adds.
TEST(CheapPostSendFenceTests, NoBuiltins_CheapFenceOn_UnregisteredCollectiveSkipsFullFence)
{
    const int off = rcclComputeCheapPostSendFenceOff(kGfx942, kAuto, true);
    EXPECT_EQ(off, 0);
    const SimplePostSendRegFlags unregistered{0, 0};
    EXPECT_FALSE(rcclSimplePostSendRegisteredOrP2p(&unregistered));
    EXPECT_TRUE(RCCL_USE_CHEAP_POST_SEND_FENCE(off, rcclSimplePostSendRegisteredOrP2p(&unregistered)));
}

TEST(CheapPostSendFenceTests, NoBuiltins_RegisteredOrP2pKeepsFullFence)
{
    const int off = rcclComputeCheapPostSendFenceOff(kGfx942, kAuto, true);
    EXPECT_EQ(off, 0);
    const SimplePostSendRegFlags regUsed{1, 0};
    const SimplePostSendRegFlags netRegUsed{0, 1};
    EXPECT_TRUE(rcclSimplePostSendRegisteredOrP2p<SimplePostSendRegFlags>(nullptr));
    EXPECT_TRUE(rcclSimplePostSendRegisteredOrP2p(&regUsed));
    EXPECT_TRUE(rcclSimplePostSendRegisteredOrP2p(&netRegUsed));
    EXPECT_FALSE(RCCL_USE_CHEAP_POST_SEND_FENCE(off, rcclSimplePostSendRegisteredOrP2p<SimplePostSendRegFlags>(nullptr)));
    EXPECT_FALSE(RCCL_USE_CHEAP_POST_SEND_FENCE(off, rcclSimplePostSendRegisteredOrP2p(&regUsed)));
    EXPECT_FALSE(RCCL_USE_CHEAP_POST_SEND_FENCE(off, rcclSimplePostSendRegisteredOrP2p(&netRegUsed)));
}

TEST(CheapPostSendFenceTests, NoBuiltins_CheapFenceDisabled_KeepsFullFence)
{
    // gfx90a auto and an explicit force-off both leave the cheap fence disabled,
    // so skip_fence stays false even for an unregistered collective.
    const int gfx90a = rcclComputeCheapPostSendFenceOff(kGfx90a, kAuto, true);
    EXPECT_EQ(gfx90a, 1);
    const SimplePostSendRegFlags unregistered{0, 0};
    EXPECT_FALSE(RCCL_USE_CHEAP_POST_SEND_FENCE(gfx90a, rcclSimplePostSendRegisteredOrP2p(&unregistered)));
    EXPECT_FALSE(RCCL_USE_CHEAP_POST_SEND_FENCE(kForceOff, rcclSimplePostSendRegisteredOrP2p(&unregistered)));
}

} // namespace RcclUnitTesting
