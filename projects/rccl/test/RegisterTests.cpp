/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdlib>
#include <cstdio>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

// Helper to check GPU availability
static bool hasGpuAvailable() {
    int numDevices = 0;
    hipError_t err = hipGetDeviceCount(&numDevices);
    return (err == hipSuccess && numDevices >= 1);
}

// Macro to skip test if no GPU is available
#define SKIP_IF_NO_GPU() \
    do { \
        if (!hasGpuAvailable()) { \
            GTEST_SKIP() << "This test requires at least 1 GPU device."; \
            return; \
        } \
    } while(0)

// Helper to initialize a single-rank communicator
static ncclResult_t initSingleRankComm(ncclComm_t* comm) {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) return res;
    return ncclCommInitRank(comm, 1, id, 0);
}

//==============================================================================
// Test implementation functions - parameterized by registration expectation
//==============================================================================

/**
 * @brief Test basic register/deregister of a single buffer
 * @param expectNonNull If true, expect non-NULL handle (registration enabled)
 */
static void testCommRegisterDeregister(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Create buffer on device
    const size_t bufferSize = 1024 * 1024; // 1 MB
    void* deviceBuffer = nullptr;
    HIPCALL(hipMalloc(&deviceBuffer, bufferSize));
    ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate device buffer";

    // Register buffer with ncclCommRegister
    void* regHandle = nullptr;
    NCCLCHECK(ncclCommRegister(comm, deviceBuffer, bufferSize, &regHandle));

    // Verify handle based on expected behavior
    if (expectNonNull) {
        EXPECT_NE(regHandle, nullptr)
            << "Buffer registration failed: regHandle is NULL even though NCCL_LOCAL_REGISTER=1";
    } else {
        EXPECT_EQ(regHandle, nullptr)
            << "Expected NULL handle when NCCL_LOCAL_REGISTER is disabled";
    }

    // Deregister and clean up
    NCCLCHECK(ncclCommDeregister(comm, regHandle));
    HIPCALL(hipFree(deviceBuffer));
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering multiple buffers simultaneously
 * @param expectNonNull If true, expect non-NULL handles and verify uniqueness
 */
static void testMultipleBufferRegistration(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Create and register multiple buffers
    const int numBuffers = 4;
    const size_t bufferSize = 64 * 1024; // 64 KB each
    void* deviceBuffers[numBuffers] = {nullptr};
    void* regHandles[numBuffers] = {nullptr};

    for (int i = 0; i < numBuffers; i++) {
        HIPCALL(hipMalloc(&deviceBuffers[i], bufferSize));
        ASSERT_NE(deviceBuffers[i], nullptr) << "Failed to allocate buffer " << i;

        NCCLCHECK(ncclCommRegister(comm, deviceBuffers[i], bufferSize, &regHandles[i]));

        if (expectNonNull) {
            EXPECT_NE(regHandles[i], nullptr) << "Registration failed for buffer " << i;
        } else {
            EXPECT_EQ(regHandles[i], nullptr) << "Expected NULL handle for buffer " << i;
        }
    }

    // Verify all handles are unique (only when registration is enabled)
    if (expectNonNull) {
        for (int i = 0; i < numBuffers; i++) {
            for (int j = i + 1; j < numBuffers; j++) {
                if (regHandles[i] != nullptr && regHandles[j] != nullptr) {
                    EXPECT_NE(regHandles[i], regHandles[j])
                        << "Buffers " << i << " and " << j << " have the same registration handle";
                }
            }
        }
    }

    // Deregister and clean up
    for (int i = 0; i < numBuffers; i++) {
        NCCLCHECK(ncclCommDeregister(comm, regHandles[i]));
        HIPCALL(hipFree(deviceBuffers[i]));
    }
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering buffers of various sizes
 * @param expectNonNull If true, expect non-NULL handles for all sizes
 */
static void testVariableSizeBuffers(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Test various buffer sizes: 4KB, 64KB, 1MB, 4MB
    const size_t sizes[] = {4096, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < numSizes; i++) {
        void* deviceBuffer = nullptr;
        void* regHandle = nullptr;

        HIPCALL(hipMalloc(&deviceBuffer, sizes[i]));
        ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate buffer of size " << sizes[i];

        NCCLCHECK(ncclCommRegister(comm, deviceBuffer, sizes[i], &regHandle));

        if (expectNonNull) {
            EXPECT_NE(regHandle, nullptr)
                << "Registration failed for buffer size " << sizes[i] << " bytes";
        } else {
            EXPECT_EQ(regHandle, nullptr)
                << "Expected NULL handle for buffer size " << sizes[i] << " bytes";
        }

        NCCLCHECK(ncclCommDeregister(comm, regHandle));
        HIPCALL(hipFree(deviceBuffer));
    }

    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test deregistering NULL handle (should succeed as no-op)
 */
static void testDeregisterNullHandle() {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Deregister NULL handle - should be a no-op
    NCCLCHECK(ncclCommDeregister(comm, nullptr));

    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Exercises the non-symmetric window register/deregister/destroy path
 *        on a single-rank communicator.
 *
 * With cuMem disabled, symmetricSupport is false, so ncclCommWindowRegister
 * takes the non-symmetric fallback path. This initializes devrState (via
 * ncclDevrInitOnce) and then tears it down through ncclDevrFinalize when the
 * comm is destroyed. The test verifies that register, deregister and destroy
 * all complete cleanly on that path (no crash / error).
 *
 * Window register/deregister errors are tolerated (they depend on cuMem
 * availability); the comm must still register and destroy without error.
 * (Byte-level leak verification is done separately via the standalone ASan
 * reproducer, since the isolated test runner _exit()s and bypasses LSan.)
 */
static void testWindowRegisterSingleRankNonSymTeardown() {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    const size_t bufferSize = NCCL_WIN_REQUIRED_ALIGNMENT; // required alignment for window registration
    void* buf = nullptr;
    NCCLCHECK(ncclMemAlloc(&buf, bufferSize));

    ncclWindow_t win = nullptr;
    ncclResult_t reg = ncclCommWindowRegister(comm, buf, bufferSize, &win,
                                              NCCL_WIN_COLL_SYMMETRIC);
    if (reg == ncclSuccess && win != nullptr) {
        NCCLCHECK(ncclCommWindowDeregister(comm, win));
    }

    NCCLCHECK(ncclMemFree(buf));
    NCCLCHECK(ncclCommDestroy(comm));
}

//==============================================================================
// Test configuration helpers
//==============================================================================

// Environment configuration for explicitly disabled local registration
static ProcessIsolatedTestRunner::TestConfig
makeDisabledConfig(const std::string& name, std::function<void()> testFn) {
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_LOCAL_REGISTER", "0"}});
}

// Environment configuration for enabled registration
static ProcessIsolatedTestRunner::TestConfig
makeEnabledConfig(const std::string& name, std::function<void()> testFn) {
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_LOCAL_REGISTER", "1"}});
}

/**
 * @brief Test ncclCommRegister and ncclCommDeregister APIs with process isolation
 *
 * This test suite verifies that:
 * 1. A device buffer can be registered with ncclCommRegister (API returns success)
 * 2. When NCCL_LOCAL_REGISTER=1, the registration returns a valid (non-NULL) handle
 * 3. When NCCL_LOCAL_REGISTER=0, NULL handle is expected (local registration off)
 * 4. The buffer can be deregistered with ncclCommDeregister
 *
 * Note: On newer HIP, NCCL_LOCAL_REGISTER defaults to 1 when unset; the "disabled"
 * cases therefore set NCCL_LOCAL_REGISTER=0 explicitly rather than clearing the var.
 */
TEST(Register, ProcessIsolatedRegisterTests)
{
    RUN_ISOLATED_TESTS(
        // CommRegisterDeregister tests
        makeDisabledConfig("CommRegisterDeregister_Disabled",
            []() { testCommRegisterDeregister(false); }),
        makeEnabledConfig("CommRegisterDeregister_Enabled",
            []() { testCommRegisterDeregister(true); }),

        // MultipleBufferRegistration tests
        makeDisabledConfig("MultipleBufferRegistration_Disabled",
            []() { testMultipleBufferRegistration(false); }),
        makeEnabledConfig("MultipleBufferRegistration_Enabled",
            []() { testMultipleBufferRegistration(true); }),

        // VariableSizeBuffers tests
        makeDisabledConfig("VariableSizeBuffers_Disabled",
            []() { testVariableSizeBuffers(false); }),
        makeEnabledConfig("VariableSizeBuffers_Enabled",
            []() { testVariableSizeBuffers(true); }),

        // DeregisterNullHandle test (no enable/disable variants needed)
        ProcessIsolatedTestRunner::TestConfig("DeregisterNullHandle", testDeregisterNullHandle),

        // Force the non-sym path (NCCL_CUMEM_ENABLE=0 => symmetricSupport=false) so the
        // non-symmetric register/teardown is exercised; NCCL_LOCAL_REGISTER=1 ensures the register path runs.
        ProcessIsolatedTestRunner::TestConfig("WindowRegisterSingleRankNonSymTeardown", testWindowRegisterSingleRankNonSymTeardown)
            .withEnvironment({{"NCCL_CUMEM_ENABLE", "0"}, {"NCCL_LOCAL_REGISTER", "1"}})
    );
}

} // namespace RcclUnitTesting
