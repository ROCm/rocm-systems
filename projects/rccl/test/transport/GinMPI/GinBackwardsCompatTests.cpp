/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the preliminary backwards-compatibility behavior
// for LSA kernels noted in the NCCL 2.29.7 release notes:
//
//   "Preliminary backwards compatibility for LSA kernels"
//
// The contract being validated:
//   1. A comm built with an older LSA team layout continues to operate
//      against the new symmetric/LSA codepaths.
//   2. Mixed-version peers (one rank using the new LSA path, one using the
//      legacy path) interoperate when the comm config requests
//      backwards-compat mode.
//
// All tests skip cleanly when GIN is not available; the contracts are
// driven through GinMPITestBase to share the existing comm lifecycle.

#include "GinMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_GIN_IB_PROXY

namespace {

class GinBackwardsCompatTest : public GinMPITestBase {};

// Validate that a comm initialized without the new lsaMultimem hint still
// runs collectives correctly (i.e., the new code path treats the old
// configuration as a valid input).
TEST_F(GinBackwardsCompatTest, OldStyleLsa_NewBuildRuns_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: build a comm with the pre-2.29.7 LSA team layout "
                    "and run an AllReduce; assert correctness (NCCL 2.29.7 "
                    "preliminary backwards compatibility).";
}

// Validate the reverse: a comm built with the new lsaMultimem layout runs
// against a peer using the older path.
TEST_F(GinBackwardsCompatTest, NewBuildAgainstOldPeer_TODO)
{
    if (!validateTestPrerequisites(/*min_processes=*/2)) {
        GTEST_SKIP() << "Need at least 2 processes";
    }
    GTEST_SKIP() << "TODO: drive a 2-rank job where one side uses the new "
                    "LSA path and the other uses the legacy one; assert "
                    "they interoperate.";
}

} // namespace

#endif // RCCL_HAS_GIN_IB_PROXY
#endif // MPI_TESTS_ENABLED
