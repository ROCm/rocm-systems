// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/registration/late.hpp"
#include "lib/rocprofiler-sdk/tests/common.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <dlfcn.h>

namespace
{
/**
 * Test fixture for late-start tests
 */
class LateStartTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Note: These tests are designed to work in isolation without
        // requiring actual runtime initialization. They test the
        // rocprofiler-register integration layer.
    }

    void TearDown() override {}
};

}  // namespace

//------------------------------------------------------------------------------
// Test: invoke_register_propagation when rocprofiler-register not loaded
//------------------------------------------------------------------------------
TEST_F(LateStartTest, no_register_loaded)
{
    // This test verifies behavior when rocprofiler-register is not loaded.
    // In a typical scenario, if no runtimes have initialized, rocprofiler-register
    // won't be loaded either.

    // Check if rocprofiler-register is loaded
    void* handle = dlopen("librocprofiler-register.so", RTLD_NOLOAD | RTLD_LAZY);

    if(!handle)
    {
        // rocprofiler-register not loaded (expected in isolated unit tests)
        auto status = rocprofiler::registration::late::invoke_register_propagation();

        // Should return INCOMPATIBLE_REGISTER_VERSION since there is no register library
        EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_REGISTER_VERSION)
            << "Should return INCOMPATIBLE_REGISTER_VERSION when rocprofiler-register not loaded";
    }
    else
    {
        // If register is loaded (e.g., due to other tests or runtime initialization),
        // that's fine - the test environment may vary
        dlclose(handle);
        GTEST_SKIP() << "rocprofiler-register is already loaded, skipping test";
    }
}

//------------------------------------------------------------------------------
// Test: invoke_register_propagation is idempotent
//------------------------------------------------------------------------------
TEST_F(LateStartTest, multiple_calls_safe)
{
    // Calling invoke_register_propagation() multiple times should be safe
    // Even if it fails (no register loaded), repeated calls should not crash

    auto status1 = rocprofiler::registration::late::invoke_register_propagation();
    auto status2 = rocprofiler::registration::late::invoke_register_propagation();

    // Both calls should have the same result
    EXPECT_EQ(status1, status2) << "Multiple calls should return same status";

    // No crash = success
    SUCCEED() << "Multiple invocations completed without crashing";
}

//------------------------------------------------------------------------------
// Test: invoke_register_propagation with register loaded but no runtimes
//------------------------------------------------------------------------------
TEST_F(LateStartTest, register_loaded_no_runtimes)
{
    // This test checks behavior when rocprofiler-register is loaded but
    // no runtimes have registered API tables yet.

    // Try to load rocprofiler-register (without RTLD_NOLOAD, so it loads if not loaded)
    void* handle = dlopen("librocprofiler-register.so", RTLD_LAZY | RTLD_LOCAL);

    if(!handle)
    {
        GTEST_SKIP() << "rocprofiler-register library not available for testing";
        return;
    }

    // rocprofiler-register is loaded, invoke propagation
    auto status = rocprofiler::registration::late::invoke_register_propagation();

    // Expected outcomes:
    // 1. SUCCESS - if function succeeds (no runtimes registered is not an error)
    // 2. ERROR_NOT_AVAILABLE - if rocprofiler-register version is too old
    // Both are acceptable for this test since we're verifying it doesn't crash

    EXPECT_TRUE(status == ROCPROFILER_STATUS_SUCCESS ||
                status == ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE ||
                status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_REGISTER_VERSION ||
                status == ROCPROFILER_STATUS_ERROR)
        << fmt::format("Should return a valid status code. Returned: {} (name={}, description={})",
                       static_cast<int>(status),
                       rocprofiler_get_status_name(status),
                       rocprofiler_get_status_string(status));

    // Clean up
    if(handle) dlclose(handle);

    SUCCEED() << "invoke_register_propagation() completed without crashing";
}

//------------------------------------------------------------------------------
// Test: Verify function signature and linking
//------------------------------------------------------------------------------
TEST_F(LateStartTest, function_linkage)
{
    // This test verifies the function exists and can be called
    // Even if it returns an error, the function should be properly linked

    rocprofiler_status_t status = rocprofiler::registration::late::invoke_register_propagation();

    // Any status is fine - we're just verifying it links and executes
    EXPECT_TRUE(status == ROCPROFILER_STATUS_SUCCESS ||
                status == ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE ||
                status == ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_REGISTER_VERSION ||
                status == ROCPROFILER_STATUS_ERROR)
        << fmt::format("Should return a valid status code. Returned: {} (name={}, description={})",
                       static_cast<int>(status),
                       rocprofiler_get_status_name(status),
                       rocprofiler_get_status_string(status));

    SUCCEED() << "invoke_register_propagation() is properly linked and callable";
}

//------------------------------------------------------------------------------
// Integration test note:
//
// More comprehensive tests that verify actual runtime wrapping behavior are
// in the integration test suite at:
//   projects/rocprofiler-sdk/tests/late-start-tracing/
//
// Those tests:
// - Initialize HSA/HIP runtimes
// - Verify API calls are traced after late-start
// - Test with actual profiling callbacks
// - Validate end-to-end late-start functionality
//------------------------------------------------------------------------------
