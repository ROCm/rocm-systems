/*************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Exercises the amdsmi_wrap API against the machine's real GPUs.
//
// These call into librccl rather than recompiling the wrapper, so they only
// link in Debug: src/CMakeLists.txt exports the internal amd_smi_* symbols
// under -fvisibility=default there and hides them everywhere else.
//
// Every case skips rather than fails when what it needs is absent, so the
// suite stays meaningful on a machine with no GPUs, no amd_smi library, or no
// fabric.

#include "amdsmi_wrap.h"
#include "alt_rsmi.h"
#include "common/ProcessIsolatedTestRunner.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

namespace RcclUnitTesting
{

namespace
{

using StubCounterFn = unsigned (*)();

std::string lifecycleStubLibraryPath()
{
    std::string path = AMDSMI_TEST_STUB_DIR;
    if(const char* inheritedPath = std::getenv("LD_LIBRARY_PATH"))
        path += ":" + std::string(inheritedPath);
    return path;
}

unsigned lifecycleStubCounter(const char* symbol)
{
    void* handle = dlopen(AMDSMI_TEST_STUB_SONAME, RTLD_NOW | RTLD_NOLOAD);
    if(handle == nullptr)
    {
        ADD_FAILURE() << "lifecycle stub is not loaded: " << dlerror();
        return 0;
    }

    dlerror();
    auto counter = reinterpret_cast<StubCounterFn>(dlsym(handle, symbol));
    if(const char* error = dlerror())
    {
        ADD_FAILURE() << "failed to resolve " << symbol << ": " << error;
        dlclose(handle);
        return 0;
    }

    const unsigned value = counter();
    dlclose(handle);
    return value;
}

// amd_smi_init() succeeds on the sysfs fallback path too, so a failure here
// means neither backend is usable and there is nothing to assert against.
class AmdSmiWrapTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(amd_smi_init() != ncclSuccess)
            GTEST_SKIP() << "amd_smi_init() failed: no usable SMI backend";
        initialized_ = true;

        if(amd_smi_getNumDevice(&numDevices_) != ncclSuccess)
            GTEST_SKIP() << "amd_smi_getNumDevice() failed";
    }

    void TearDown() override
    {
        if(initialized_)
            EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
    }

    void requireDevices(uint32_t count)
    {
        if(numDevices_ < count)
            GTEST_SKIP() << "needs " << count << " device(s), found " << numDevices_;
    }

    uint32_t numDevices_ = 0;
    bool     initialized_ = false;
};

} // namespace

TEST(AmdSmiWrapLifecycle, InitAndShutdownSucceed)
{
    if(amd_smi_init() != ncclSuccess)
        GTEST_SKIP() << "amd_smi_init() failed: no usable SMI backend";
    EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
}

// Re-initializing after a shutdown has to work: RCCL brings the wrapper up per
// communicator, so a one-shot init would break every rank after the first.
TEST(AmdSmiWrapLifecycle, InitIsRepeatable)
{
    if(amd_smi_init() != ncclSuccess)
        GTEST_SKIP() << "amd_smi_init() failed: no usable SMI backend";
    ASSERT_EQ(amd_smi_shutdown(), ncclSuccess);

    EXPECT_EQ(amd_smi_init(), ncclSuccess);
    EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
}

TEST(AmdSmiWrapLifecycle, ConcurrentCallersReceiveCachedFailure)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ConcurrentCallersReceiveCachedFailure",
        []() {
            constexpr size_t kThreadCount = 16;
            std::vector<ncclResult_t> results(kThreadCount, ncclSuccess);
            std::vector<std::thread>  threads;
            threads.reserve(kThreadCount);

            for(size_t i = 0; i < kThreadCount; ++i)
                threads.emplace_back([&, i]() { results[i] = amd_smi_init(); });
            for(auto& thread : threads)
                thread.join();

            for(size_t i = 0; i < results.size(); ++i)
                EXPECT_EQ(results[i], ncclInternalError) << "caller " << i;

            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_init_count"), 1u);
            EXPECT_EQ(amd_smi_init(), ncclInternalError);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_init_count"), 1u);
            EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_shutdown_count"), 0u);
        },
        {{"RCCL_USE_AMD_SMI_LIB", "1"},
         {"RCCL_AMDSMI_TEST_DELAY_INIT", "1"},
         {"RCCL_AMDSMI_TEST_FAIL_INIT", "1"},
         {"LD_LIBRARY_PATH", lifecycleStubLibraryPath()}}
    );
}

TEST(AmdSmiWrapLifecycle, ShutdownCleansUpAfterVersionFailure)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ShutdownCleansUpAfterVersionFailure",
        []() {
            EXPECT_EQ(amd_smi_init(), ncclInternalError);
            EXPECT_EQ(amd_smi_init(), ncclInternalError);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_init_count"), 1u);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_version_count"), 1u);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_shutdown_count"), 0u);

            EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_shutdown_count"), 1u);

            // Shutdown clears the cached failure and permits a fresh attempt.
            EXPECT_EQ(amd_smi_init(), ncclInternalError);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_init_count"), 2u);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_version_count"), 2u);
            EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
            EXPECT_EQ(lifecycleStubCounter("amdsmi_test_shutdown_count"), 2u);
        },
        {{"RCCL_USE_AMD_SMI_LIB", "1"},
         {"RCCL_AMDSMI_TEST_FAIL_VERSION", "1"},
         {"LD_LIBRARY_PATH", lifecycleStubLibraryPath()}}
    );
}

TEST_F(AmdSmiWrapTest, PciBusIdIsPopulatedForEveryDevice)
{
    requireDevices(1);

    for(uint32_t i = 0; i < numDevices_; i++)
    {
        char busId[32] = {0};
        ASSERT_EQ(amd_smi_getDevicePciBusIdString(i, busId, sizeof(busId)), ncclSuccess)
            << "device " << i;
        EXPECT_GT(strlen(busId), 0u) << "device " << i << " reported an empty bus ID";
    }
}

// The topology code identifies GPUs by bus ID, so the mapping has to be a
// bijection: every device's own bus ID must resolve back to that same index.
TEST_F(AmdSmiWrapTest, PciBusIdRoundTripsToDeviceIndex)
{
    requireDevices(1);

    for(uint32_t i = 0; i < numDevices_; i++)
    {
        char busId[32] = {0};
        ASSERT_EQ(amd_smi_getDevicePciBusIdString(i, busId, sizeof(busId)), ncclSuccess);

        uint32_t deviceIndex = 0xFFFFFFFFu;
        ASSERT_EQ(amd_smi_getDeviceIndexByPciBusId(busId, &deviceIndex), ncclSuccess)
            << "bus ID " << busId;
        EXPECT_EQ(deviceIndex, i) << "bus ID " << busId << " resolved to the wrong device";
    }
}

TEST_F(AmdSmiWrapTest, LinkInfoIsReportedBetweenTwoDevices)
{
    requireDevices(2);

    amdsmi_link_type_t linkType;
    int                hops  = -1;
    int                count = -1;
    ASSERT_EQ(amd_smi_getLinkInfo(0, 1, &linkType, &hops, &count), ncclSuccess);

    EXPECT_GE(hops, 0);
    EXPECT_GE(count, 0);
}

TEST_F(AmdSmiWrapTest, FirmwareVersionQuerySucceeds)
{
    requireDevices(1);

    // Reported as 0 when the backend cannot source it; only the call contract
    // is under test here.
    uint64_t fwVersion = 0;
    EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fwVersion), ncclSuccess);
}

// amd_smi_isFabricSupported() and amd_smi_getFabricBandwidth() both read the
// cache that amd_smi_ensureFabricInitialized() fills, so they must agree with
// the cached record rather than each deriving their own answer.
TEST_F(AmdSmiWrapTest, FabricViewsAgreeWithCachedDeviceInfo)
{
    requireDevices(1);
    ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);

    for(uint32_t i = 0; i < numDevices_; i++)
    {
        struct amdsmiFabricDeviceInfo devInfo;
        memset(&devInfo, 0, sizeof(devInfo));
        ASSERT_EQ(amd_smi_getFabricDeviceInfo(i, &devInfo), ncclSuccess) << "device " << i;

        bool supported = false;
        ASSERT_EQ(amd_smi_isFabricSupported(i, &supported), ncclSuccess) << "device " << i;
        EXPECT_EQ(supported, devInfo.fabricSupported) << "device " << i;

        uint32_t bandwidth = 0;
        ASSERT_EQ(amd_smi_getFabricBandwidth(i, &bandwidth), ncclSuccess) << "device " << i;

        if(!supported)
        {
            // Documented contract: 0 tells the caller to fall back to
            // arch-based defaults instead of trusting a fabric number.
            EXPECT_EQ(bandwidth, 0u) << "device " << i << " reported bandwidth without fabric";
            continue;
        }

        EXPECT_EQ(bandwidth, devInfo.bandwidth) << "device " << i;
        // A device is only marked supported when it passed this predicate, so
        // the cached type/state must still satisfy it.
        EXPECT_TRUE(amdSmiFabricStateUsable(devInfo.fabricType, devInfo.state))
            << "device " << i << " marked supported with type " << devInfo.fabricType
            << " state " << devInfo.state;
        EXPECT_GT(devInfo.vpodSize, 0u) << "device " << i;
    }
}

// amd_smi_fabricTelemIdToString dispatches on the loaded runtime's major
// version between the pre-27 return-value signature and the 27+ out-param
// signature (see amdSmiTelemIdUsesOutParam in amdsmi_wrap.h). The predicate
// itself is unit-tested in AmdSmiFabricTests; this drives the actual dispatch
// against whichever runtime is really loaded and pins the contract its callers
// depend on: for any id it returns a printable string, never nullptr. A
// regression in the reinterpret_cast legacy path (or the out-param path) would
// otherwise only surface as a crash or garbage log line on real hardware.
TEST_F(AmdSmiWrapTest, FabricTelemIdToStringNeverReturnsNull)
{
    requireDevices(1);
    if(amd_smi_ensureFabricInitialized() != ncclSuccess)
        GTEST_SKIP() << "fabric not available on this machine";

    // An id the runtime does not map still has to come back as a printable
    // string, because callers log it unconditionally. Whichever call
    // convention matched the loaded library, both paths collapse a missing
    // name to "UNKNOWN" rather than nullptr.
    const char* unknownName = amd_smi_fabricTelemIdToString(~static_cast<uint64_t>(0));
    ASSERT_NE(unknownName, nullptr);
    EXPECT_GT(std::strlen(unknownName), 0u);
}

// The regression this suite exists for: RCCL_USE_AMD_SMI_LIB selects between
// amdsmi_get_gpu_fabric_info() and the ualink sysfs nodes, and the two are
// meant to be interchangeable views of one fabric. They silently diverged when
// the library returned SUCCESS without ever writing fabric_info.version and
// RCCL rejected the payload on that basis, so the library path reported no
// fabric on hardware where sysfs plainly showed it.
//
// Runs isolated because RCCL_PARAM caches the env var in a static on first
// read, so the value has to be set before this process starts.
TEST(AmdSmiWrapFabricPaths, LibraryPathMatchesSysfs)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "LibraryPathMatchesSysfs",
        []() {
            if(amd_smi_init() != ncclSuccess)
                GTEST_SKIP() << "amd_smi_init() failed: no usable SMI backend";

            // Read sysfs first: it decides whether this machine has fabric at
            // all, and so whether there is anything to compare against.
            ASSERT_EQ(ARSMI_init(), 0);
            struct ARSMI_fabricInfo sysfsInfo;
            memset(&sysfsInfo, 0, sizeof(sysfsInfo));
            const int sysfsResult = ARSMI_get_fabric_info(0, &sysfsInfo);
            if(sysfsResult == ENODEV)
                GTEST_SKIP() << "device 0 has no ualink sysfs node: no fabric on this machine";
            ASSERT_EQ(sysfsResult, 0);

            ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
            struct amdsmiFabricDeviceInfo libInfo;
            memset(&libInfo, 0, sizeof(libInfo));
            ASSERT_EQ(amd_smi_getFabricDeviceInfo(0, &libInfo), ncclSuccess);

            EXPECT_EQ(libInfo.fabricSupported, sysfsInfo.supported == 1)
                << "amd_smi and sysfs disagree on whether device 0 has fabric";
            EXPECT_EQ(static_cast<int>(libInfo.fabricType), static_cast<int>(sysfsInfo.fabric_type));
            EXPECT_EQ(static_cast<int>(libInfo.state), static_cast<int>(sysfsInfo.accel_state));
            EXPECT_EQ(libInfo.acceleratorId, sysfsInfo.accel_id);
            EXPECT_EQ(libInfo.ppodSize, sysfsInfo.ppod_size);
            EXPECT_EQ(libInfo.cliqueId, sysfsInfo.vpod_id);
            EXPECT_EQ(libInfo.vpodSize, sysfsInfo.vpod_size);

            EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
        },
        {{"RCCL_USE_AMD_SMI_LIB", "1"}}
    );
}

} // namespace RcclUnitTesting
