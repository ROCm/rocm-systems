/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the bootstrap bidirectional AllGather gating contract.
//
// Socket bidir is on by default (NCCL_BOOTSTRAP_BIDIR_ALLGATHER=1); net OOB and
// net bidir are off by default. The helper `bootstrapBidirEnabled(nranks, kind)`
// in src/bootstrap.cc is the single source of truth that bootstrapInit,
// bootstrapAllGather and bootstrapClose all consult; if its contract drifts the
// runtime quietly diverges between setup and teardown, hence these tests pin it.
//
// Each case runs in an isolated process (via RUN_ISOLATED_TEST_WITH_ENV) because
// NCCL_PARAM caches the env value on first read for the lifetime of the process,
// so the only way to re-evaluate is to fork a fresh child.

#include <gtest/gtest.h>
#include "common/ProcessIsolatedTestRunner.hpp"
#include "bootstrap.h"  // bootstrapBidirEnabled — single source of truth for the
                        // signature; if it drifts we get a compile error here, not
                        // a deferred linker error against librccl.so.

namespace RcclUnitTesting {

constexpr int kKindSocket = 0;
constexpr int kKindNet    = 1;

// -----------------------------------------------------------------------------
// Tiny-comm short-circuit: nranks<3 is below the bidir threshold (one ring step
// is degenerate at N==2 and undefined at N<2). Independent of any env var.
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, TinyNranks_ReturnsFalse_Default)
{
    RUN_ISOLATED_TEST(
        "BootstrapBidir.TinyNranks_ReturnsFalse_Default",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(0, kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(1, kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(2, kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(0, kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(1, kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(2, kKindNet));
        }
    );
}

TEST(BootstrapBidir, TinyNranks_ReturnsFalse_EvenWithBidirForcedOn)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.TinyNranks_ReturnsFalse_EvenWithBidirForcedOn",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(0, kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(2, kKindSocket));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_ALLGATHER", "1"},
         {"NCCL_OOB_NET_ENABLE", "0"}}
    );
}

// -----------------------------------------------------------------------------
// Branch 1 default: with no env vars set, the socket bidir path is selected for
// any nranks>=3. This is the headline behavioural change the branch introduces.
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, SocketBidir_OnByDefault)
{
    RUN_ISOLATED_TEST(
        "BootstrapBidir.SocketBidir_OnByDefault",
        []()
        {
            // Span the realistic nranks range: small (3, 8), 2-node 8ppn (16),
            // mid-scale (128), large (1024) - all should resolve to "on".
            EXPECT_TRUE(bootstrapBidirEnabled(3,    kKindSocket));
            EXPECT_TRUE(bootstrapBidirEnabled(8,    kKindSocket));
            EXPECT_TRUE(bootstrapBidirEnabled(16,   kKindSocket));
            EXPECT_TRUE(bootstrapBidirEnabled(128,  kKindSocket));
            EXPECT_TRUE(bootstrapBidirEnabled(1024, kKindSocket));
        }
    );
}

TEST(BootstrapBidir, NetBidir_OffByDefault)
{
    RUN_ISOLATED_TEST(
        "BootstrapBidir.NetBidir_OffByDefault",
        []()
        {
            // Net OOB is off by default, so net bidir is unconditionally off
            // regardless of nranks (the !netOn guard fires first).
            EXPECT_FALSE(bootstrapBidirEnabled(8,    kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(128,  kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(1024, kKindNet));
        }
    );
}

// -----------------------------------------------------------------------------
// Off-switch: BIDIR_ALLGATHER=0 must produce the legacy unidir socket path
// regardless of scale (and regardless of the threshold knob).
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, SocketBidir_ExplicitOff_DisablesEverywhere)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.SocketBidir_ExplicitOff_DisablesEverywhere",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(8,    kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(128,  kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(1024, kKindSocket));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_ALLGATHER", "0"}}
    );
}

// Threshold knob applies only to the net path; for socket it must have no effect
// (the helper short-circuits on the on/off semantics before consulting threshold).
TEST(BootstrapBidir, SocketBidir_ThresholdKnob_HasNoEffect)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.SocketBidir_ThresholdKnob_HasNoEffect",
        []()
        {
            // Threshold absurdly above any realistic nranks: still on.
            EXPECT_TRUE(bootstrapBidirEnabled(8,   kKindSocket));
            EXPECT_TRUE(bootstrapBidirEnabled(128, kKindSocket));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_THRESHOLD", "1000000"}}
    );
}

// -----------------------------------------------------------------------------
// Net OOB shadow: enabling net OOB transport must immediately disable the socket
// bidir path (the dispatcher in bootstrapAllGather picks the net path instead).
// This is the !netOn clause in bootstrapBidirEnabled(., kind=0).
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, NetOob_ShadowsSocketBidir)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetOob_ShadowsSocketBidir",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(8,   kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(128, kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(1024, kKindSocket));
        },
        {{"NCCL_OOB_NET_ENABLE", "1"},
         {"NCCL_BOOTSTRAP_BIDIR_ALLGATHER", "1"}}
    );
}

// -----------------------------------------------------------------------------
// Net bidir requires net OOB. With OOB_NET_ENABLE=0, BIDIR_NET=1 must still
// resolve to false: there is no second QP pair to drive Ring1 over.
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, NetBidir_WithoutNetOob_ReturnsFalse)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetBidir_WithoutNetOob_ReturnsFalse",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(128,  kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(1024, kKindNet));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_NET", "1"},
         {"NCCL_OOB_NET_ENABLE", "0"}}
    );
}

// -----------------------------------------------------------------------------
// Net bidir on (explicit): both knobs must be set; nranks threshold is bypassed
// when the user has explicitly opted in (BIDIR_NET=1 vs the auto value -1).
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, NetBidir_ExplicitOn_BypassesThreshold)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetBidir_ExplicitOn_BypassesThreshold",
        []()
        {
            // Threshold absurdly above the asked nranks; explicit =1 must still
            // win because it short-circuits before threshold logic.
            EXPECT_TRUE(bootstrapBidirEnabled(8,   kKindNet));
            EXPECT_TRUE(bootstrapBidirEnabled(128, kKindNet));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_NET", "1"},
         {"NCCL_OOB_NET_ENABLE", "1"},
         {"NCCL_BOOTSTRAP_BIDIR_THRESHOLD", "1000000"}}
    );
}

// -----------------------------------------------------------------------------
// Net bidir auto-threshold: BIDIR_NET=-1 (auto) + OOB_NET_ENABLE=1 with the
// default threshold 128 must (a) be off below threshold, (b) on at and above it.
// This protects the regression matrix: small comms shouldn't pay net-bidir setup
// cost, but >=128 ranks should benefit.
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, NetBidir_AutoThreshold_BelowAndAbove)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetBidir_AutoThreshold_BelowAndAbove",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(8,   kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(64,  kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(127, kKindNet));
            EXPECT_TRUE (bootstrapBidirEnabled(128, kKindNet));
            EXPECT_TRUE (bootstrapBidirEnabled(1024, kKindNet));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_NET", "-1"},
         {"NCCL_OOB_NET_ENABLE", "1"},
         {"NCCL_BOOTSTRAP_BIDIR_THRESHOLD", "128"}}
    );
}

TEST(BootstrapBidir, NetBidir_AutoThreshold_CustomValue)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetBidir_AutoThreshold_CustomValue",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(63, kKindNet));
            EXPECT_TRUE (bootstrapBidirEnabled(64, kKindNet));
            EXPECT_TRUE (bootstrapBidirEnabled(65, kKindNet));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_NET", "-1"},
         {"NCCL_OOB_NET_ENABLE", "1"},
         {"NCCL_BOOTSTRAP_BIDIR_THRESHOLD", "64"}}
    );
}

// Threshold==0 disables the auto path entirely (never crosses the gate).
TEST(BootstrapBidir, NetBidir_AutoThreshold_Zero_DisablesAuto)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetBidir_AutoThreshold_Zero_DisablesAuto",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(128,  kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(1024, kKindNet));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_NET", "-1"},
         {"NCCL_OOB_NET_ENABLE", "1"},
         {"NCCL_BOOTSTRAP_BIDIR_THRESHOLD", "0"}}
    );
}

// -----------------------------------------------------------------------------
// BIDIR_NET explicit-zero: even when net OOB is enabled, BIDIR_NET=0 must
// short-circuit (the `v == 0` branch in bootstrapBidirEnabled's kind==1 arm).
// Other tests only cover BIDIR_NET=1 / -1.
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, NetBidir_ExplicitZero_With_NetOob_DisablesNet)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetBidir_ExplicitZero_With_NetOob_DisablesNet",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(128,  kKindNet));
            EXPECT_FALSE(bootstrapBidirEnabled(1024, kKindNet));
        },
        {{"NCCL_BOOTSTRAP_BIDIR_NET", "0"},
         {"NCCL_OOB_NET_ENABLE", "1"}}
    );
}

// -----------------------------------------------------------------------------
// Net OOB auto path: NCCL_OOB_NET_ENABLE=-1 + threshold gates the underlying
// bootstrapNetEnabledEffective helper. Without this case the auto (`v == -1`)
// branch and the `thr > 0 && nranks >= thr` short-circuit inside
// bootstrapNetEnabledEffective stay at zero hits in coverage.
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, NetOob_Auto_Threshold_BelowAndAbove_GatesSocketShadow)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetOob_Auto_Threshold_BelowAndAbove_GatesSocketShadow",
        []()
        {
            // Below threshold: net OOB is auto-off, so socket bidir is allowed.
            EXPECT_TRUE (bootstrapBidirEnabled(64,  kKindSocket));
            // At/above threshold: net OOB auto-on, which shadows socket bidir.
            EXPECT_FALSE(bootstrapBidirEnabled(128, kKindSocket));
            EXPECT_FALSE(bootstrapBidirEnabled(256, kKindSocket));
        },
        {{"NCCL_OOB_NET_ENABLE", "-1"},
         {"NCCL_BOOTSTRAP_BIDIR_THRESHOLD", "128"}}
    );
}

TEST(BootstrapBidir, NetOob_Auto_Threshold_Zero_KeepsNetOff)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "BootstrapBidir.NetOob_Auto_Threshold_Zero_KeepsNetOff",
        []()
        {
            // thr==0 makes the auto-net-on path fail (never crosses), so socket
            // bidir is allowed unconditionally — covers the false branch of the
            // `thr > 0 && ...` short-circuit in bootstrapNetEnabledEffective.
            EXPECT_TRUE(bootstrapBidirEnabled(8,    kKindSocket));
            EXPECT_TRUE(bootstrapBidirEnabled(1024, kKindSocket));
        },
        {{"NCCL_OOB_NET_ENABLE", "-1"},
         {"NCCL_BOOTSTRAP_BIDIR_THRESHOLD", "0"}}
    );
}

// -----------------------------------------------------------------------------
// Unknown 'kind' values must return false (defensive default for any future
// caller that adds a third transport without updating this gate).
// -----------------------------------------------------------------------------
TEST(BootstrapBidir, UnknownKind_ReturnsFalse)
{
    RUN_ISOLATED_TEST(
        "BootstrapBidir.UnknownKind_ReturnsFalse",
        []()
        {
            EXPECT_FALSE(bootstrapBidirEnabled(128, 2));
            EXPECT_FALSE(bootstrapBidirEnabled(128, -1));
            EXPECT_FALSE(bootstrapBidirEnabled(128, 99));
        }
    );
}

} // namespace RcclUnitTesting
