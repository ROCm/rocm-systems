/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <unordered_map>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "comm.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

TEST(HierarchicalCollectives, LazyInitGuards)
{
    const std::unordered_map<std::string, std::string> env = {
        {"RCCL_HIERARCHICAL_ALLGATHER", "1"},
        {"RCCL_HIERARCHICAL_ALLGATHER_MIN_BYTES_PER_RANK", "1024"},
    };

    auto registerCase = [&env](const char* name, auto test)
    {
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(name, test)
                .withEnvironment(env)
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
