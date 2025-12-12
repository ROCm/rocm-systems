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

#include "hip_test_params.hh"
#include "hip_test_parameters.hh"
#include "hip_test_context.hh"

void TestParameterStore::initialize() {
    // Load all level parameters from generated compile-time constants
    auto allParams = TestParameters::initializeLevelParameters();
    
    for (const auto& [levelName, params] : allParams) {
        levelMemorySizes[levelName] = params.memory_sizes;
        levelBlockSizes[levelName] = params.block_sizes;
        levelIterations[levelName] = params.iterations;
        levelWarmups[levelName] = params.warmups;
        levelMaxMemory[levelName] = params.max_memory;
        
        LogPrintf("[TestParameterStore] %s: %zu memory sizes, %zu block sizes, %d iterations\n",
                  levelName.c_str(), params.memory_sizes.size(), 
                  params.block_sizes.size(), params.iterations);
    }
    
    // Set defaults (use level_0 as fallback if available, otherwise hardcoded)
    if (levelMemorySizes.count("level_0")) {
        defaultMemorySizes = levelMemorySizes["level_0"];
        defaultBlockSizes = levelBlockSizes["level_0"];
        defaultIterations = levelIterations["level_0"];
        defaultWarmups = levelWarmups["level_0"];
    } else {
        // Hardcoded fallback if no levels defined
        defaultMemorySizes = {1024, 1048576, 10485760};  // 1K, 1M, 10M
        defaultBlockSizes = {64, 256};
        LogPrintf("[TestParameterStore] Warning: No level_0 defined, using hardcoded defaults\n%s", "");
    }
    
    LogPrintf("[TestParameterStore] Initialization complete - %zu levels loaded\n", allParams.size());
}

void TestParameterStore::loadLevelConfig(const std::string& level) {
    currentTestLevel = level;
    
    if (levelMemorySizes.count(level)) {
        const auto& memSizes = levelMemorySizes.at(level);
        const auto& blkSizes = levelBlockSizes.at(level);
        
        LogPrintf("[TestParameterStore] Activating level: %s\n", level.c_str());
        
        // Safely print memory sizes range
        if (!memSizes.empty()) {
            LogPrintf("[TestParameterStore]   Memory sizes: %zu (%zu bytes to %zu bytes)\n",
                      memSizes.size(), memSizes.front(), memSizes.back());
        } else {
            LogPrintf("[TestParameterStore]   Memory sizes: 0 (empty)\n%s", "");
        }
        
        // Safely print block sizes range
        if (!blkSizes.empty()) {
            LogPrintf("[TestParameterStore]   Block sizes: %zu (%d to %d)\n",
                      blkSizes.size(), blkSizes.front(), blkSizes.back());
        } else {
            LogPrintf("[TestParameterStore]   Block sizes: 0 (empty)\n%s", "");
        }
        
        if (levelIterations.count(level)) {
            LogPrintf("[TestParameterStore]   Iterations: %d\n", levelIterations.at(level));
        }
    } else {
        LogPrintf("[TestParameterStore] Warning: Level '%s' not found, using defaults\n", level.c_str());
    }
}

const std::vector<size_t>& TestParameterStore::getMemorySizesForCurrentLevel() const {
    if (!currentTestLevel.empty() && levelMemorySizes.count(currentTestLevel)) {
        return levelMemorySizes.at(currentTestLevel);
    }
    return defaultMemorySizes;
}

const std::vector<int>& TestParameterStore::getBlockSizesForCurrentLevel() const {
    if (!currentTestLevel.empty() && levelBlockSizes.count(currentTestLevel)) {
        return levelBlockSizes.at(currentTestLevel);
    }
    return defaultBlockSizes;
}

int TestParameterStore::getIterationsForCurrentLevel() const {
    if (!currentTestLevel.empty() && levelIterations.count(currentTestLevel)) {
        return levelIterations.at(currentTestLevel);
    }
    return defaultIterations;
}

int TestParameterStore::getWarmupsForCurrentLevel() const {
    if (!currentTestLevel.empty() && levelWarmups.count(currentTestLevel)) {
        return levelWarmups.at(currentTestLevel);
    }
    return defaultWarmups;
}

size_t TestParameterStore::getMaxMemoryForCurrentLevel() const {
    if (!currentTestLevel.empty() && levelMaxMemory.count(currentTestLevel)) {
        return levelMaxMemory.at(currentTestLevel);
    }
    return defaultMaxMemory;
}

void TestParameterStore::clear() {
    currentTestLevel.clear();
    levelMemorySizes.clear();
    levelBlockSizes.clear();
    levelIterations.clear();
    levelWarmups.clear();
    levelMaxMemory.clear();
}
