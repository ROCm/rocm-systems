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
 * This file demonstrates how ONE test case adapts its parameters based on
 * which level you run with. The same test code runs with different intensity:
 * 
 * Usage:
 *   ./test "[level_0]"  # Quick: 3 memory sizes (1K, 1M, 10M)
 *   ./test "[level_1]"  # Standard: 7 memory sizes (1K to 100M)
 *   ./test "[level_2]"  # Comprehensive: 14 memory sizes (64B to 2G)
 * 
 * The test parameters are defined in hip_tests_config.yaml under cmd_options:
 *   level_0:
 *     memory_sizes: [1K, 1M, 10M]
 *   level_1:
 *     memory_sizes: [1K, 4K, 64K, 1M, 10M, 50M, 100M]
 *   level_2:
 *     memory_sizes: [64, 256, 1K, 4K, 16K, 64K, 256K, 1M, 10M, 50M, 100M, 500M, 1G, 2G]
 * 
 * To add a new test:
 * 1. Add entry to hip_tests_config.yaml with multiple level tags
 * 2. Use TEST_CASE(MacroName) - tags come from YAML
 * 3. Access parameters via TestParameterStore::instance().getMemorySizesForCurrentLevel()
 */

#include <hip_test_common.hh>
#include <hip_test_params.hh>
#include <hip_tests_config.hh>  // Generated macros with tags from YAML
#include <vector>

/**
 * ONE test case that adapts to the level you run with.
 * 
 * YAML configuration:
 *   Unit_hipDeviceMemAlloc_Functional:
 *     level: 0  # Minimum level this test runs at
 *     tags: [level_0, level_1, level_2, device, memory]
 * 
 * Running with different levels:
 *   ./test "[level_0]" -> Uses 3 memory sizes (quick smoke test)
 *   ./test "[level_1]" -> Uses 7 memory sizes (standard regression)
 *   ./test "[level_2]" -> Uses 14 memory sizes (comprehensive)
 */
TEST_CASE(Unit_hipDeviceMemAlloc_Functional) {
    auto& params = TestParameterStore::instance();
    
    // Parameters automatically adapt based on which level filter was used
    auto sizes = params.getMemorySizesForCurrentLevel();
    auto size = GENERATE_COPY(from_range(sizes));
    
    INFO("Testing with memory size: " << size << " bytes");
    INFO("Current level: " << params.currentTestLevel);
    INFO("Total sizes for this level: " << sizes.size());

    // Basic allocation test
    void* d_ptr = nullptr;
    HIP_CHECK(hipMalloc(&d_ptr, size));
    REQUIRE(d_ptr != nullptr);

    // Memset to verify memory is usable
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
 * Another example: Multi-device test with level-based parameters
 * 
 * This test runs on all devices with multiple memory patterns.
 * The number of memory sizes tested depends on the level.
 */
TEST_CASE(Unit_hipDeviceMemAlloc_MultiDevice) {
    auto& params = TestParameterStore::instance();
    
    // Get memory sizes based on current level
    auto sizes = params.getMemorySizesForCurrentLevel();
    auto size = GENERATE_COPY(from_range(sizes));
    
    INFO("Testing with memory size: " << size << " bytes");

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

    HIP_CHECK(hipFree(d_ptr));
}

/**
 * Verify Level Configuration (hidden test)
 * 
 * Run with: ./test "Unit_hipDeviceMemAlloc_VerifyLevelConfig"
 */
TEST_CASE(Unit_hipDeviceMemAlloc_VerifyLevelConfig) {
    auto& params = TestParameterStore::instance();
    
    INFO("=== Verifying Level Configurations ===");
    INFO("Current Test Level: " << params.currentTestLevel);
    
    // Verify level_0 parameters exist
    REQUIRE(params.levelMemorySizes.count("level_0") > 0);
    REQUIRE(params.levelMemorySizes["level_0"].size() == 3);  // Expect 3 sizes
    
    INFO("Level 0 Memory Sizes: " << params.levelMemorySizes["level_0"].size());
    
    // Verify level_1 parameters exist
    REQUIRE(params.levelMemorySizes.count("level_1") > 0);
    REQUIRE(params.levelMemorySizes["level_1"].size() == 7);  // Expect 7 sizes
    
    INFO("Level 1 Memory Sizes: " << params.levelMemorySizes["level_1"].size());
    
    // Verify level_2 parameters exist
    REQUIRE(params.levelMemorySizes.count("level_2") > 0);
    REQUIRE(params.levelMemorySizes["level_2"].size() == 14);  // Expect 14 sizes
    
    INFO("Level 2 Memory Sizes: " << params.levelMemorySizes["level_2"].size());
}
