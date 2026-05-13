/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the advanced GIN queue-control features added in
// NCCL 2.29.7:
//
//   - Advanced queue control on dynamic ports
//   - GIN barrier API that supports different signal types
//   - nLsaTeams reporting via ncclCommQueryProperties
//
// All tests skip cleanly when GIN is not available; they live alongside
// the existing GinMPITests.cpp scaffolding.

#include "GinMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_GIN_IB_PROXY

namespace {

class GinQueueControlTest : public GinMPITestBase {};

// Validate that the ginQueueDepth requirement is honored by ncclDevCommCreate.
TEST_F(GinQueueControlTest, QueueDepth_Honored_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: assert ncclDevCommCreate observes the "
                    "ginQueueDepth value passed in "
                    "ncclDevCommRequirements (NCCL 2.29.7).";
}

// 2.29.7: GIN barrier with multiple signal types. Verify that issuing
// barriers using a counter signal and a flag signal both work and don't
// stomp on each other.
TEST_F(GinQueueControlTest, BarrierAcrossSignalTypes_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: drive ncclGinBarrier (or equivalent) with two "
                    "different signal types and assert independent "
                    "completion (NCCL 2.29.7).";
}

// 2.29.7: ncclCommQueryProperties now reports nLsaTeams. The single-rank
// case is covered by CommQueryPropertiesTests.cpp; this test asserts that
// nLsaTeams scales sensibly with comm size in a multi-process setup.
TEST_F(GinQueueControlTest, NLsaTeams_MultiRank_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/4)) {
        GTEST_SKIP() << "Need at least 4 processes for a meaningful "
                        "nLsaTeams check";
    }
    GTEST_SKIP() << "TODO: assert nLsaTeams is reported and scales with "
                    "comm size when GIN is configured (NCCL 2.29.7).";
}

} // namespace

#endif // RCCL_HAS_GIN_IB_PROXY
#endif // MPI_TESTS_ENABLED
