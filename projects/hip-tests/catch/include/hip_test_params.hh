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

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cstdint>

/**
 * @brief Global parameter store for test configuration.
 * 
 * Parameters are loaded from compile-time constants generated from 
 * definitions.yaml at build time. The event listener detects the test
 * level from command-line filters and loads appropriate parameters.
 * 
 * Thread Safety:
 *   This class is designed for single-threaded test execution (Catch2 default).
 *   Do not use with parallel test execution without adding synchronization.
 * 
 * Example usage:
 * @code
 * TEST_CASE(Unit_hipMemcpy_Functional) {
 *     auto& params = TestParameterStore::instance();
 *     
 *     // Get parameters for current level (auto-detected from filter)
 *     auto sizes = params.getMemorySizesForCurrentLevel();
 *     auto size = GENERATE_COPY(from_range(sizes));
 *     
 *     // Test with level-appropriate sizes:
 *     // level_0: [1K, 1M, 10M]
 *     // level_1: [1K, 4K, 64K, 1M, 10M, 50M, 100M]
 *     // level_2: [64, 256, 1K, ... 2G]
 * }
 * @endcode
 */
class TestParameterStore {
public:
    /**
     * @brief Get singleton instance
     */
    static TestParameterStore& instance() {
        static TestParameterStore inst;
        return inst;
    }

    /**
     * @brief Initialize parameter store from generated compile-time constants
     * Called once at test startup by event listener
     */
    void initialize();

    /**
     * @brief Load parameters for a specific level
     * Called by event listener when [level_X] tag is detected
     * @param level Level name (e.g., "level_0", "level_1")
     */
    void loadLevelConfig(const std::string& level);

    /**
     * @brief Get memory sizes for current test level
     * @return Vector of memory sizes in bytes
     */
    const std::vector<size_t>& getMemorySizesForCurrentLevel() const;

    /**
     * @brief Get block sizes for current test level
     * @return Vector of block sizes
     */
    const std::vector<int>& getBlockSizesForCurrentLevel() const;

    /**
     * @brief Get iterations for current test level
     * @return Number of iterations
     */
    int getIterationsForCurrentLevel() const;

    /**
     * @brief Get warmup iterations for current test level
     * @return Number of warmup iterations
     */
    int getWarmupsForCurrentLevel() const;

    /**
     * @brief Get cooperative groups iterations for current test level
     * @return Number of iterations
     */
    int getCgIterationsForCurrentLevel() const;

    /**
     * @brief Get math accuracy iteration count for current test level
     * @return Number of math accuracy iterations
     */
    uint64_t getMathAccuracyIterationsForCurrentLevel() const;

    /**
     * @brief Get math accuracy max memory percentage for current test level
     * @return Percentage (0-100) of available memory to use for math accuracy tests
     */
    int getMathAccuracyMaxMemoryPercentageForCurrentLevel() const;

    /**
     * @brief Get maximum memory for current test level
     * @return Maximum memory in bytes
     */
    size_t getMathMaxMemoryForCurrentLevel() const;

    /**
     * @brief Get math reduction factor for current test level
     * @return Reduction factor applied to math test workloads
     */
    double getMathReductionFactorForCurrentLevel() const;

    /**
     * @brief Clear all stored data
     */
    void clear();

    /**
     * @brief Current test level (set by event listener)
     */
    std::string currentTestLevel;

    /**
     * @brief Level-specific parameters (loaded from compile-time constants)
     * Public for verification tests
     */
    std::map<std::string, std::vector<size_t>> levelMemorySizes;
    std::map<std::string, std::vector<int>> levelBlockSizes;
    std::map<std::string, int> levelIterations;
    std::map<std::string, int> levelWarmups;
    std::map<std::string, int> levelCgIterations;
    std::map<std::string, uint64_t> levelMathAccuracyIterations;
    std::map<std::string, int> levelMathAccuracyMaxMemoryPercentage;
    std::map<std::string, size_t> levelMathMaxMemory;
    std::map<std::string, double> levelMathReductionFactor;

private:
    TestParameterStore() = default;
    ~TestParameterStore() = default;
    TestParameterStore(const TestParameterStore&) = delete;
    TestParameterStore& operator=(const TestParameterStore&) = delete;
    
    /**
     * @brief Fallback parameters (if no level specified)
     */
    std::vector<size_t> defaultMemorySizes = {1024, 1048576, 10485760}; // 1K, 1M, 10M
    std::vector<int> defaultBlockSizes = {64, 256};
    int defaultIterations = 1000;
    int defaultWarmups = 100;
    int defaultCgIterations = 1;
    uint64_t defaultMathAccuracyIterations = 4294967296; // 2^32
    int defaultMathAccuracyMaxMemoryPercentage = 80;
    size_t defaultMathMaxMemory = 2147483648; // 2GB
    double defaultMathReductionFactor = 0.1;
};
