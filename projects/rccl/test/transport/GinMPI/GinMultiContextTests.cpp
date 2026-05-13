/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the GIN multi-context features added in
// NCCL 2.29.2/2.29.7:
//
//   - Multi-context support including exclusive contexts (2.29.2)
//   - VA-based GIN signals (2.29.2)
//   - Strict window ordering on individual GIN signals (2.29.7)
//   - Cross-rail GIN connectivity for both proxy and gdaki (2.29.7)
//   - Decoupling of GIN from NET plugin and topology choices (2.29.7)
//
// The basic GIN MPI smoke tests live in GinMPITests.cpp; this file is the
// home for richer multi-context scenarios that the basic harness does not
// yet cover. All tests skip cleanly when GIN is not available on the host.

#include "GinMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_GIN_IB_PROXY

namespace {

class GinMultiContextTest : public GinMPITestBase {};

// Exercise creation of a comm with ginContextCount > 1 and
// ginExclusiveContexts = true. The contract is that:
//   1. Initialization succeeds.
//   2. Per-context signals are independent: posting to context A's signal
//      does not wake a waiter on context B's signal.
TEST_F(GinMultiContextTest, ExclusiveContexts_SignalsAreIndependent_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: drive a 2-context comm with ginExclusiveContexts "
                    "= true and assert per-context signal isolation "
                    "(NCCL 2.29.2 multi-context, NCCL 2.29.7 strict "
                    "window ordering).";
}

// Verify that GIN signals are addressed via virtual addresses now that
// the 2.29.2 work routes signal handles through ncclSymPtr.
TEST_F(GinMultiContextTest, VABasedSignals_RoundTrip_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: validate that ncclGinSignal handles round-trip "
                    "as VAs (NCCL 2.29.2 VA-based GIN signals).";
}

// 2.29.7: cross-rail GIN connectivity for both proxy and gdaki. Verify that
// the comm reports ginConnectionType = NCCL_GIN_CONNECTION_RAIL when
// configured.
TEST_F(GinMultiContextTest, CrossRailConnectivity_Proxy_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: create a comm with NCCL_GIN_TYPE_PROXY + "
                    "ginConnectionType = NCCL_GIN_CONNECTION_RAIL and "
                    "exercise a cross-rail put (NCCL 2.29.7).";
}

TEST_F(GinMultiContextTest, CrossRailConnectivity_Gdaki_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: same as above for NCCL_GIN_TYPE_GDAKI; requires "
                    "gdaki-capable transport (NCCL 2.29.7).";
}

// 2.29.7: GIN is now decoupled from the NET plugin / topology choices.
// Verify that NCCL_GIN_FORCE_ENABLE works regardless of the NET plugin
// the rest of the system selects.
TEST_F(GinMultiContextTest, ForceEnable_IndependentFromNetPlugin_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: assert GIN can be enabled even when the NET "
                    "plugin is set to a non-GIN choice (NCCL 2.29.7 "
                    "decoupling).";
}

} // namespace

#endif // RCCL_HAS_GIN_IB_PROXY
#endif // MPI_TESTS_ENABLED
