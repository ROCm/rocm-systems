/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <chrono>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "comm.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

TEST(HierarchicalCollectives, LazyInitGuards)
{
    auto registerCase = [](const char* name, auto test)
    {
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(name, test)
                .withTimeout(std::chrono::seconds(60))
        );
    };

    registerCase("EnsureNotEligibleIsNoOp", []()
    {
        ncclComm comm{};
        EXPECT_EQ(ncclSuccess, rcclEnsureHierarchicalComms(&comm));
        EXPECT_FALSE(comm.hierarchicalInitAttempted);
        EXPECT_FALSE(comm.hierarchicalCommsInitialized);
    });

    registerCase("EnsureInitializedIsNoOp", []()
    {
        ncclComm comm{};
        comm.hierarchicalEligible = true;
        comm.hierarchicalCommsInitialized = true;
        EXPECT_EQ(ncclSuccess, rcclEnsureHierarchicalComms(&comm));
        EXPECT_FALSE(comm.hierarchicalInitAttempted);
    });

    registerCase("EnsureReturnsPriorFailureWithoutRetry", []()
    {
        ncclComm comm{};
        comm.hierarchicalEligible = true;
        comm.hierarchicalInitAttempted = true;
        comm.hierarchicalInitResult = ncclSystemError;

        EXPECT_EQ(ncclSystemError, rcclEnsureHierarchicalComms(&comm));
        EXPECT_FALSE(comm.hierarchicalCommsInitialized);
    });

    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging = true;
    EXPECT_TRUE(ProcessIsolatedTestRunner::executeAllTests(options));
}

} // namespace RcclUnitTesting
