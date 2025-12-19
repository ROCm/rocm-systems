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
#include "hip_test_parameters.hh"  // Generated header with compile-time constants
#include <iostream>

void TestParameterStore::initialize() {
    std::cout << "\n[TestParameterStore] Initializing from compile-time constants..." << std::endl;
    
    // Load all level parameters from generated compile-time constants
    auto allParams = TestParameters::initializeLevelParameters();
    
    for (const auto& [levelName, params] : allParams) {
        levelMemorySizes[levelName] = params.memory_sizes;
        levelBlockSizes[levelName] = params.block_sizes;
        levelIterations[levelName] = params.iterations;
        levelWarmups[levelName] = params.warmups;
        levelMaxMemory[levelName] = params.max_memory;
        
        std::cout << "[TestParameterStore]   " << levelName << ": "
                  << params.memory_sizes.size() << " memory sizes, "
                  << params.block_sizes.size() << " block sizes, "
                  << params.iterations << " iterations" << std::endl;
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
        std::cout << "[TestParameterStore] Warning: No level_0 defined, using hardcoded defaults" << std::endl;
    }
    
    std::cout << "[TestParameterStore] Initialization complete - "
              << allParams.size() << " levels loaded\n" << std::endl;
}

void TestParameterStore::loadLevelConfig(const std::string& level) {
    currentTestLevel = level;
    
    if (levelMemorySizes.count(level)) {
        std::cout << "[TestParameterStore] Activating level: " << level << std::endl;
        std::cout << "  Memory sizes: " << levelMemorySizes[level].size() 
                  << " (" << levelMemorySizes[level][0] << " bytes to "
                  << levelMemorySizes[level][levelMemorySizes[level].size()-1] << " bytes)" << std::endl;
        std::cout << "  Block sizes: " << levelBlockSizes[level].size() 
                  << " (" << levelBlockSizes[level][0] << " to "
                  << levelBlockSizes[level][levelBlockSizes[level].size()-1] << ")" << std::endl;
        std::cout << "  Iterations: " << levelIterations[level] << std::endl;
    } else {
        std::cout << "[TestParameterStore] Warning: Level '" << level 
                  << "' not found, using defaults" << std::endl;
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
    std::cout << "[TestParameterStore] Cleared all parameters" << std::endl;
}
