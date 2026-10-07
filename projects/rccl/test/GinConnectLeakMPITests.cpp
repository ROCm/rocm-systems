/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclGinConnectOnce must close the comms it already opened when a later connect fails.

#include "MPITestBase.hpp"
#include "SymmetricMemPrereq.hpp"
#include "TestChecks.hpp"

#include <gtest/gtest.h>
#include <nccl_device.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;

namespace RcclUnitTesting
{

namespace
{

// The stub reports GPI, which no in-tree backend claims, so it becomes the active GIN backend.
constexpr int kGinTypeGpi = NCCL_NET_DEVICE_GIN_GPI;

// Fail the second connect: the first has to have been opened, and released.
constexpr int kConnections   = 2;
constexpr int kFailConnectAt = 2;

// The stub is built and installed beside the test binary, in the build tree and in an install alike.
std::string ginFaultPluginPath()
{
    std::error_code       ec;
    std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if(ec)
        return {};
    return (exe.parent_path() / RCCL_TEST_GIN_FAULT_PLUGIN_NAME).string();
}

bool readCounter(const std::string& path, const char* key, int* out)
{
    FILE* f = fopen(path.c_str(), "r");
    if(f == nullptr)
        return false;

    char name[64];
    int  value = 0;
    bool found = false;
    while(fscanf(f, "%63s %d", name, &value) == 2)
    {
        if(strcmp(name, key) == 0)
        {
            *out  = value;
            found = true;
        }
    }
    fclose(f);
    return found;
}

} // namespace

class GinConnectLeakMPITest : public MPITestBase
{
protected:
    std::string                                      counterPath_;
    std::vector<std::pair<std::string, std::string>> savedEnv_;
    std::vector<std::string>                         unsetEnv_;

    // Records the previous value so TearDown can hand later tests an untouched environment.
    int setTestEnv(const char* name, const std::string& value)
    {
        if(const char* prev = getenv(name))
            savedEnv_.emplace_back(name, prev);
        else
            unsetEnv_.emplace_back(name);
        return setenv(name, value.c_str(), 1);
    }

    void SetUp() override
    {
        // Skip on all ranks together: a rank skipping alone would hang its peer.
        const std::string plugin = ginFaultPluginPath();
        if(auto reason = mpiCoordinatedSkipReason(plugin.empty() || access(plugin.c_str(), F_OK) != 0,
                                                  "GIN fault plugin stub not found beside the test binary");
           !reason.empty())
        {
            GTEST_SKIP() << reason;
        }

        // Recorded before the collective decision, so TearDown removes it even if another rank skips.
        char      counterTemplate[] = "/tmp/rccl_gin_counters.XXXXXX";
        const int fd                = mkstemp(counterTemplate);
        if(fd >= 0)
        {
            close(fd);
            counterPath_ = counterTemplate;
        }
        if(auto reason = mpiCoordinatedSkipReason(fd < 0, "Could not create a counter file in /tmp");
           !reason.empty())
        {
            GTEST_SKIP() << reason;
        }

        // Absolute path: an in-process LD_LIBRARY_PATH change never reaches dlopen.
        ASSERT_EQ(0, setTestEnv("NCCL_GIN_PLUGIN", plugin));
        ASSERT_EQ(0, setTestEnv("NCCL_GIN_ENABLE", "1"));
        // Without cuMem there is no symmetric support, and ncclGinConnectOnce never runs.
        ASSERT_EQ(0, setTestEnv("NCCL_CUMEM_ENABLE", "1"));
        ASSERT_EQ(0, setTestEnv("NCCL_GIN_TYPE", std::to_string(kGinTypeGpi)));
        ASSERT_EQ(0, setTestEnv("NCCL_GIN_NCONNECTIONS", std::to_string(kConnections)));
        ASSERT_EQ(0, setTestEnv("RCCL_TEST_GIN_FAULT_FAIL_CONNECT_AT", std::to_string(kFailConnectAt)));
        ASSERT_EQ(0, setTestEnv("RCCL_TEST_GIN_FAULT_COUNTER_FILE", counterPath_));

        MPITestBase::SetUp();
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        for(const auto& [name, value] : savedEnv_)
            setenv(name.c_str(), value.c_str(), 1);
        for(const auto& name : unsetEnv_)
            unsetenv(name.c_str());
        if(!counterPath_.empty())
            remove(counterPath_.c_str());
    }
};

// Regression for NVIDIA/nccl#2206.
TEST_F(GinConnectLeakMPITest, FailedConnectReleasesOpenedComms)
{
    // GIN is set up only when the communicator spans more than one node.
    SKIP_UNLESS_MPI_PREREQS(/*min_processes=*/2,
                            /*max_processes=*/2,
                            /*require_power_of_two=*/false,
                            /*min_nodes=*/2);

    const std::string cuMemSkip = RCCLTestGuards::symmetricMemEnvAndRuntimeSkipReason();
    if(auto reason = mpiCoordinatedSkipReason(!cuMemSkip.empty(), cuMemSkip.empty() ? nullptr : cuMemSkip.c_str());
       !reason.empty())
    {
        GTEST_SKIP() << reason;
    }

    // Drives ncclGinConnectOnce, which is expected to fail on the second connect.
    const bool setupFailed = createTestCommunicator() != ncclSuccess;

    int        connects = 0, closeColls = 0, closeListens = 0;
    const bool haveCounters = readCounter(counterPath_, "connect", &connects);
    readCounter(counterPath_, "closeColl", &closeColls);
    readCounter(counterPath_, "closeListen", &closeListens);
    TEST_INFO("GIN fault plugin: connect=%d closeColl=%d closeListen=%d (setupFailed=%d)",
              connects, closeColls, closeListens, setupFailed ? 1 : 0);

    // Prove the plugin reached the injected failure, or the counts below would pass vacuously.
    ASSERT_MPI_TRUE(setupFailed);
    ASSERT_MPI_TRUE(haveCounters);
    ASSERT_MPI_EQ(kFailConnectAt, connects);

    // The listen comm closes once per successful connect and once more on the failure path.
    ASSERT_MPI_EQ(kFailConnectAt - 1, closeColls);
    ASSERT_MPI_EQ(kFailConnectAt, closeListens);
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
