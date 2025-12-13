/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @file hipDeviceMemAlloc_WithLevels.cc
 * @brief Example demonstrating level-based test parameter system
 * 
 * This file demonstrates how to use the level-based testing system where
 * test parameters (memory sizes, block sizes, etc.) are loaded from a
 * centralized configuration file based on test level tags.
 * 
 * Test Levels:
 * - level_0: Quick smoke tests (3 sizes: 1K, 1M, 10M)
 * - level_1: Standard regression tests (7 sizes: 1K, 4K, 64K, 1M, 10M, 50M, 100M)
 * 
 * Configuration File: catch/config/test_levels.txt
 * 
 * Usage:
 *   ./test "[level_0]"  # Run only level_0 tests (fast)
 *   ./test "[level_1]"  # Run only level_1 tests (comprehensive)
 *   ./test "[level]"    # Run all level tests
 */

#include <hip_test_common.hh>
#include <hip_test_params.hh>
#include <vector>

/**
 * Level 0 - Quick Smoke Test
 */
TEST_CASE("Unit_hipDeviceMemAlloc_Level0_Quick", "[level_0][device][memory]") {
    auto& params = TestParameterStore::instance();
    
    // GENERATE from level_0 memory sizes (automatically loaded from config)
    auto sizes = params.getMemorySizesForCurrentLevel();
    auto size = GENERATE_COPY(from_range(sizes));
    
    INFO("Testing Level 0 with memory size: " << size << " bytes");

    // Basic allocation test
    void* d_ptr = nullptr;
    HIP_CHECK(hipMalloc(&d_ptr, size));
    REQUIRE(d_ptr != nullptr);

    // Simple memset to verify memory is usable
    HIP_CHECK(hipMemset(d_ptr, 0xAB, size));
    
    // Verify the pattern
    std::vector<char> h_data(size);
    HIP_CHECK(hipMemcpy(h_data.data(), d_ptr, size, hipMemcpyDeviceToHost));
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_data[i] == (char)0xAB);
    }

    HIP_CHECK(hipFree(d_ptr));
}

/**
 * TEST CASE 2: Level 1 - Standard Regression Test
 * 
 * Purpose: Comprehensive testing across multiple devices and patterns
 * Runtime: ~30 seconds
 * Parameters: 7 memory sizes from test_levels.txt (1K, 4K, 64K, 1M, 10M, 50M, 100M)
 */
TEST_CASE("Unit_hipDeviceMemAlloc_Level1_Standard", "[level_1][device][memory]") {
    auto& params = TestParameterStore::instance();
    
    // Get memory sizes from level_1 configuration
    auto sizes = params.getMemorySizesForCurrentLevel();
    auto size = GENERATE_COPY(from_range(sizes));
    
    INFO("Testing Level 1 with memory size: " << size << " bytes");

    // Get device count
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    REQUIRE(numDevices > 0);

    // Test on all available devices
    auto deviceId = GENERATE_COPY(range(0, numDevices));
    HIP_CHECK(hipSetDevice(deviceId));
    
    INFO("Testing on device: " << deviceId);

    void* d_ptr = nullptr;
    HIP_CHECK(hipMalloc(&d_ptr, size));
    REQUIRE(d_ptr != nullptr);

    // Test multiple memory patterns
    SECTION("Pattern 0x00") {
        HIP_CHECK(hipMemset(d_ptr, 0x00, size));
        std::vector<char> h_data(size);
        HIP_CHECK(hipMemcpy(h_data.data(), d_ptr, size, hipMemcpyDeviceToHost));
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_data[i] == (char)0x00);
        }
    }
    
    SECTION("Pattern 0xFF") {
        HIP_CHECK(hipMemset(d_ptr, 0xFF, size));
        std::vector<char> h_data(size);
        HIP_CHECK(hipMemcpy(h_data.data(), d_ptr, size, hipMemcpyDeviceToHost));
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_data[i] == (char)0xFF);
        }
    }
    
    SECTION("Pattern 0x55") {
        HIP_CHECK(hipMemset(d_ptr, 0x55, size));
        std::vector<char> h_data(size);
        HIP_CHECK(hipMemcpy(h_data.data(), d_ptr, size, hipMemcpyDeviceToHost));
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_data[i] == (char)0x55);
        }
    }

    HIP_CHECK(hipFree(d_ptr));
}

/**
 * TEST CASE 3: Verify Level Configuration
 * 
 * Purpose: Verify that level configurations are loaded correctly
 * This is a hidden test (tag: .verify) - run explicitly for verification
 * 
 * Run with: ./test "Unit_hipDeviceMemAlloc_VerifyLevelConfig"
 */
TEST_CASE("Unit_hipDeviceMemAlloc_VerifyLevelConfig", "[.verify][config]") {
    auto& params = TestParameterStore::instance();
    
    INFO("=== Verifying Level Configurations ===");
    INFO("Current Test Level: " << params.currentTestLevel);
    
    // Verify level_0 parameters exist
    REQUIRE(params.levelMemorySizes.count("level_0") > 0);
    REQUIRE(params.levelMemorySizes["level_0"].size() == 3);  // Expect 3 sizes
    
    INFO("Level 0 Memory Sizes:");
    for (size_t s : params.levelMemorySizes["level_0"]) {
        INFO("  - " << s << " bytes");
    }
    
    // Verify level_1 parameters exist
    REQUIRE(params.levelMemorySizes.count("level_1") > 0);
    REQUIRE(params.levelMemorySizes["level_1"].size() == 7);  // Expect 7 sizes
    
    INFO("Level 1 Memory Sizes:");
    for (size_t s : params.levelMemorySizes["level_1"]) {
        INFO("  - " << s << " bytes");
    }
    
    // Verify block sizes if defined
    if (params.levelBlockSizes.count("level_0") > 0) {
        INFO("Level 0 Block Sizes:");
        for (int s : params.levelBlockSizes["level_0"]) {
            INFO("  - " << s);
        }
    }
    
    if (params.levelBlockSizes.count("level_1") > 0) {
        INFO("Level 1 Block Sizes:");
        for (int s : params.levelBlockSizes["level_1"]) {
            INFO("  - " << s);
        }
    }
}

/**
 * TEST CASE 4: Compare Level Parameters
 * 
 * Purpose: Demonstrate difference between level_0 and level_1
 * This is a hidden test - run explicitly for comparison
 * 
 * Run with: ./test "Unit_hipDeviceMemAlloc_CompareLevels"
 */
TEST_CASE("Unit_hipDeviceMemAlloc_CompareLevels", "[.compare][config]") {
    auto& params = TestParameterStore::instance();
    
    INFO("=== Comparing Level 0 vs Level 1 ===");
    
    REQUIRE(params.levelMemorySizes.count("level_0") > 0);
    REQUIRE(params.levelMemorySizes.count("level_1") > 0);
    
    size_t level0_count = params.levelMemorySizes["level_0"].size();
    size_t level1_count = params.levelMemorySizes["level_1"].size();
    
    INFO("Level 0: " << level0_count << " memory sizes");
    INFO("Level 1: " << level1_count << " memory sizes");
    
    // Level 1 should have more test cases than level 0
    REQUIRE(level1_count > level0_count);
    
    INFO("\nLevel 0 is for quick smoke tests");
    INFO("Level 1 is for comprehensive regression tests");
}

