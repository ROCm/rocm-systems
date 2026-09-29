/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclGinConnectOnce establishes NCCL_GIN_NCONNECTIONS comms in a loop. When a
// later one fails, the comms already opened must be closed, not leaked.

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

// The example plugin defaults to PROXY, which ncclGinPluginAssignToComm drops in
// favour of the built-in proxy. GPI has no in-tree implementation, so claiming
// it is what lets this plugin become the active backend. Taken from the enum so
// a renumbering cannot leave the test quietly selecting some other backend.
constexpr int kGinTypeGpi = NCCL_NET_DEVICE_GIN_GPI;

// Fail the second connect: the first has to have been opened, and released.
constexpr int kConnections   = 2;
constexpr int kFailConnectAt = 2;

#ifdef RCCL_GIN_EXAMPLE_PLUGIN_DIR
constexpr const char* kGinExamplePluginDir = RCCL_GIN_EXAMPLE_PLUGIN_DIR;
#else
constexpr const char* kGinExamplePluginDir = nullptr;
#endif

// The test binary is installed but the build tree is not, so look next to the
// running executable first and fall back to the build-tree path for local runs.
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
        // Both preconditions are reduced: a rank that skips alone leaves its peer
        // blocked in the next collective until the suite is force-aborted.
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
        // /tmp is sticky, so a leftover owned by another user cannot be unlinked.
        // The plugin's open() would then fail silently and readCounter would parse
        // that file's numbers instead, letting the assertions pass on stale data
        // even with the cleanup removed. Refuse to run rather than read it.
        remove(counterPath_.c_str());
        if(auto reason = mpiCoordinatedSkipReason(
               fileExists(counterPath_),
               "Counter file already exists and could not be removed; set "
               "RCCL_GIN_EXAMPLE_COUNTER_FILE to a path this user owns");
           !reason.empty())
        {
            GTEST_SKIP() << reason;
        }

        // An absolute path, not "example": the loader dlopens the name verbatim
        // first, and an LD_LIBRARY_PATH set from inside the process never reaches
        // dlopen, since glibc resolves the search path at startup.
        setenv("NCCL_GIN_PLUGIN", plugin.c_str(), 1);
        setenv("NCCL_GIN_ENABLE", "1", 1);
        // cuMem auto-enables only on gfx1250 (rocmwrap.cc), and without it
        // symmetricSupport is false and ncclGinConnectOnce never runs, so the
        // test would skip on every other architecture unless the runner set it.
        setenv("NCCL_CUMEM_ENABLE", "1", 1);
        setenv("NCCL_GIN_TYPE", std::to_string(kGinTypeGpi).c_str(), 1);
        setenv("RCCL_GIN_EXAMPLE_DEVICE_TYPE", std::to_string(kGinTypeGpi).c_str(), 1);
        setenv("NCCL_GIN_NCONNECTIONS", std::to_string(kConnections).c_str(), 1);
        setenv("RCCL_GIN_EXAMPLE_FAIL_CONNECT_AT", std::to_string(kFailConnectAt).c_str(), 1);

        MPITestBase::SetUp();
    }
};

// Regression for the leak fixed upstream in NCCL PR #2206: the failure path of
// ncclGinConnectOnce must close the listen comm and every GIN comm it opened.
TEST_F(GinConnectLeakMPITest, FailedConnectReleasesOpenedComms)
{
    // GIN is only brought up when the LSA team is smaller than the communicator,
    // so a single-node run would never reach ncclGinConnectOnce at all.
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

    // Order matters for the diagnosis: setup failing is what the injection was
    // for, so check that first. The counter assertions below only make sense once
    // the plugin is known to have become the active backend and to have reached
    // the injected failure; otherwise they would pass vacuously.
    ASSERT_MPI_TRUE(setupFailed);
    ASSERT_MPI_TRUE(haveCounters);
    ASSERT_MPI_EQ(kFailConnectAt, connects);

    // The comms opened before the failure must have been closed, not leaked. The
    // listen comm is closed once per successful connect and once more by the
    // failure path, so an exact count is what distinguishes the two.
    ASSERT_MPI_EQ(kFailConnectAt - 1, closeColls);
    ASSERT_MPI_EQ(kFailConnectAt, closeListens);
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
