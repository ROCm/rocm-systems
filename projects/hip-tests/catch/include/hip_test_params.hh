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

#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <memory>

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
     * @brief Get maximum memory for current test level
     * @return Maximum memory in bytes
     */
    size_t getMaxMemoryForCurrentLevel() const;

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
    std::map<std::string, size_t> levelMaxMemory;

private:
    TestParameterStore() = default;
    ~TestParameterStore() = default;
    TestParameterStore(const TestParameterStore&) = delete;
    TestParameterStore& operator=(const TestParameterStore&) = delete;
    
    /**
     * @brief Fallback parameters (if no level specified)
     */
    std::vector<size_t> defaultMemorySizes;
    std::vector<int> defaultBlockSizes;
    int defaultIterations = 1000;
    int defaultWarmups = 100;
    size_t defaultMaxMemory = 2147483648; // 2GB
};

/** Parse numeric suffix from "level_N" (e.g. level_2 -> 2). Returns -1 if missing or invalid. */
inline int ParseTestLevelNumber(const std::string& level) {
  if (level.size() < 7 || level.compare(0, 6, "level_") != 0) {
    return -1;
  }
  return std::atoi(level.c_str() + 6);
}

/** Level index for the running test (from listener + tags), or -1 if unset. */
inline int CurrentTestLevelNumber() {
  return ParseTestLevelNumber(TestParameterStore::instance().currentTestLevel);
}

/**
 * How many multi_grid_group.cc test_case indices to sweep (smaller for low levels).
 * Unknown level (-1) uses the broadest sweep (same as level_2+).
 */
inline int CooperativeMultiGridTestCaseCount() {
  const int n = CurrentTestLevelNumber();
  if (n == 0) {
    return 2;
  }
  if (n == 1) {
    return 8;
  }
  return 20;
}
/** SM-scale multiplier list sizes — must match cpu_grid.h GenerateBlockDimensions*. */
inline int CooperativeBlockGridMultiplierListSize() {
  const int n = CurrentTestLevelNumber();
  if (n == 0) {
    return 1;
  }
  if (n == 1) {
    return 3;
  }
  return 4;
}

inline int CooperativeBlockGridMultiplierListSizeShuffle() {
  const int n = CurrentTestLevelNumber();
  if (n == 0) {
    return 1;
  }
  return 2;
}

/** Warp-scale multiplier list sizes — must match cpu_grid.h GenerateThreadDimensions*. */
inline int CooperativeThreadMultiplierListSizeFull() {
  const int n = CurrentTestLevelNumber();
  if (n == 0) {
    return 1;
  }
  if (n == 1) {
    return 4;
  }
  return 5;
}

inline int CooperativeThreadMultiplierListSizeShuffle() {
  const int n = CurrentTestLevelNumber();
  if (n == 0) {
    return 1;
  }
  return 3;
}

/**
 * Catch2 GENERATE_COPY unions alternatives of the same type: cpu_grid.h uses
 * (2 + 3*M) block-grid variants and (10 + 3*M) thread-block variants for multiplier list size M.
 */
inline int CooperativeBlockGridGeneratorUnionCount(int multiplier_list_size) {
  return 2 + 3 * multiplier_list_size;
}

inline int CooperativeThreadGeneratorUnionCount(int multiplier_list_size) {
  return 10 + 3 * multiplier_list_size;
}

/**
 * Two GENERATE statements in one TEST_CASE form a Cartesian product (Catch2).
 * Use this for tests that call both GenerateBlockDimensions* and GenerateThreadDimensions*.
 */
inline int CooperativeGridThreadCartesianProductCount(int multiplier_list_size_block,
                                                      int multiplier_list_size_thread) {
  return CooperativeBlockGridGeneratorUnionCount(multiplier_list_size_block) *
         CooperativeThreadGeneratorUnionCount(multiplier_list_size_thread);
}

