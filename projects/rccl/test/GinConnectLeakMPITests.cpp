/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclGinConnectOnce must close the comms it already opened when a later connect fails.

#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#include <gtest/gtest.h>
#include <nccl_device.h>
#include <climits>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;

namespace RcclUnitTesting
{

namespace
{

// External PROXY plugins lose to the built-in proxy; no in-tree backend claims GPI, so the plugin wins.
constexpr int kGinTypeGpi = NCCL_NET_DEVICE_GIN_GPI;

// Fail the second connect: the first has to have been opened, and released.
constexpr int kConnections   = 2;
constexpr int kFailConnectAt = 2;

#ifdef RCCL_GIN_EXAMPLE_PLUGIN_DIR
constexpr const char* kGinExamplePluginDir = RCCL_GIN_EXAMPLE_PLUGIN_DIR;
#else
constexpr const char* kGinExamplePluginDir = nullptr;
#endif

// Prefer the plugin installed next to this binary, else the build-tree copy.
std::string ginExamplePluginPath()
{
    char self[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if(n > 0)
    {
        self[n] = '\0';
        if(char* slash = strrchr(self, '/'))
        {
            *slash = '\0';
            std::string installed = std::string(self) + "/librccl-gin-example.so";
            if(access(installed.c_str(), F_OK) == 0)
                return installed;
        }
    }

    if(!kGinExamplePluginDir || kGinExamplePluginDir[0] == '\0')
        return {};
    return std::string(kGinExamplePluginDir) + "/librccl-gin-example.so";
}

bool fileExists(const std::string& path)
{
    return !path.empty() && access(path.c_str(), F_OK) == 0;
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
    std::string counterPath_;

    void SetUp() override
    {
        // Skip on all ranks together: a rank skipping alone would hang its peer.
        const std::string plugin = ginExamplePluginPath();
        if(auto reason = mpiCoordinatedSkipReason(
               !fileExists(plugin),
               "librccl-gin-example.so not built; rebuild with ENABLE_HOST_API_TESTS=ON");
           !reason.empty())
        {
            GTEST_SKIP() << reason;
        }

        const char* counter = getenv("RCCL_GIN_EXAMPLE_COUNTER_FILE");
        if(auto reason = mpiCoordinatedSkipReason(
               counter == nullptr || *counter == '\0',
               "Set RCCL_GIN_EXAMPLE_COUNTER_FILE to a per-node path for the plugin's close counts");
           !reason.empty())
        {
            GTEST_SKIP() << reason;
        }
        counterPath_ = counter;
        // A leftover we cannot remove (sticky /tmp) would feed stale counts to the assertions.
        remove(counterPath_.c_str());
        if(auto reason = mpiCoordinatedSkipReason(
               fileExists(counterPath_),
               "Counter file already exists and could not be removed; set "
               "RCCL_GIN_EXAMPLE_COUNTER_FILE to a path this user owns");
           !reason.empty())
        {
            GTEST_SKIP() << reason;
        }

        // Absolute path: an in-process LD_LIBRARY_PATH change never reaches dlopen.
        setenv("NCCL_GIN_PLUGIN", plugin.c_str(), 1);
        setenv("NCCL_GIN_ENABLE", "1", 1);
        // Without cuMem there is no symmetric support, and ncclGinConnectOnce never runs.
        setenv("NCCL_CUMEM_ENABLE", "1", 1);
        setenv("NCCL_GIN_TYPE", std::to_string(kGinTypeGpi).c_str(), 1);
        setenv("RCCL_GIN_EXAMPLE_DEVICE_TYPE", std::to_string(kGinTypeGpi).c_str(), 1);
        setenv("NCCL_GIN_NCONNECTIONS", std::to_string(kConnections).c_str(), 1);
        setenv("RCCL_GIN_EXAMPLE_FAIL_CONNECT_AT", std::to_string(kFailConnectAt).c_str(), 1);

        MPITestBase::SetUp();
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

    // Drives ncclGinConnectOnce, which is expected to fail on the second connect.
    const bool setupFailed = createTestCommunicator() != ncclSuccess;

    int        connects = 0, closeColls = 0, closeListens = 0;
    const bool haveCounters = readCounter(counterPath_, "connect", &connects);
    readCounter(counterPath_, "closeColl", &closeColls);
    readCounter(counterPath_, "closeListen", &closeListens);
    TEST_INFO("GIN example plugin: connect=%d closeColl=%d closeListen=%d (setupFailed=%d)",
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
