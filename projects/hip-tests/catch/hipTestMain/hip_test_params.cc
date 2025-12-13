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

#include <hip_test_params.hh>
#include <hip_test_config_loader.hh>
#include <iostream>
#include <algorithm>
#include <cstdlib>

void TestParameterStore::initialize() {
    std::cout << "[TestParameterStore] Initializing test parameters..." << std::endl;
    //detectDeviceCapabilities();
    loadEnvironmentConfig();
    loadCentralizedLevelConfig();
    populateDefaultParameters();
    printConfiguration();
}

void TestParameterStore::loadCentralizedLevelConfig() {
    std::string centralizedFile = ConfigFileLoader::findConfigFile("test_levels.txt");
    
    if (!centralizedFile.empty()) {
        std::cout << "  [Centralized Config] Loading from: " << centralizedFile << std::endl;
        
        std::map<std::string, int> levelIterations;
        std::map<std::string, int> levelWarmups;
        
        if (ConfigFileLoader::loadCentralizedLevelConfig(
                centralizedFile, 
                levelMemorySizes, 
                levelBlockSizes,
                levelIterations,
                levelWarmups)) {
            
            std::cout << "    Successfully loaded centralized level config" << std::endl;
            std::cout << "    Levels defined: ";
            for (const auto& pair : levelMemorySizes) {
                std::cout << pair.first << " ";
            }
            std::cout << std::endl;
            return;
        }
    }
    
    std::cout << "  [Centralized Config] Not found, will load per-level configs as needed" << std::endl;
}

void TestParameterStore::detectDeviceCapabilities() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "[TestParameterStore] Failed to get device count: " 
                  << hipGetErrorString(err) << std::endl;
        return;
    }
    
    std::cout << "  Detected " << deviceCount << " device(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        deviceIds.push_back(i);
        
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, i);
        if (err != hipSuccess) {
            std::cerr << "[TestParameterStore] Failed to get properties for device " 
                      << i << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }
        
        deviceArchs.push_back(prop.gcnArchName);
        deviceMemorySizes[i] = prop.totalGlobalMem;
        deviceComputeCapabilities[i] = prop.major * 10 + prop.minor;
        deviceMaxThreadsPerBlock[i] = prop.maxThreadsPerBlock;
        
        std::cout << "    Device " << i << ": " << prop.name << std::endl;
        std::cout << "      Arch: " << prop.gcnArchName << std::endl;
        std::cout << "      Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        
        if (prop.managedMemory) {
            supportedFeatures_["managed_memory"] = true;
        }
        if (prop.cooperativeLaunch) {
            supportedFeatures_["cooperative_launch"] = true;
        }
    }
    
    if (deviceCount > 1) {
        int canAccess = 0;
        err = hipDeviceCanAccessPeer(&canAccess, 0, 1);
        if (err == hipSuccess && canAccess) {
            supportedFeatures_["peer_access"] = true;
            enablePeerAccessTests = true;
            std::cout << "  Peer Access: Supported" << std::endl;
        }
        enableMultiGPUTests = true;
    }
}

void TestParameterStore::loadEnvironmentConfig() {
    const char* testModeEnv = std::getenv("HIP_TEST_MODE");
    if (testModeEnv) {
        testMode = testModeEnv;
        std::cout << "  Test Mode: " << testMode << std::endl;
    }
    
    const char* extendedEnv = std::getenv("HIP_EXTENDED_TESTS");
    if (extendedEnv && std::string(extendedEnv) == "1") {
        enableExtendedTests = true;
        std::cout << "  Extended Tests: Enabled" << std::endl;
    }
}

void TestParameterStore::populateDefaultParameters() {
    if (testMode == "quick") {
        memorySizes = {1024, 1024 * 1024, 10 * 1024 * 1024};
        blockSizes = {64, 256};
    } else if (testMode == "extended") {
        memorySizes = {1024, 64 * 1024, 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024};
        blockSizes = {32, 64, 128, 256, 512, 1024};
    } else {
        memorySizes = {1024, 64 * 1024, 1024 * 1024, 10 * 1024 * 1024};
        blockSizes = {64, 128, 256, 512};
    }
    
    gridSizes = {1, 16, 64, 256};
    streamCounts = {2, 4, 8};
}

void TestParameterStore::clear() {
    deviceIds.clear();
    deviceArchs.clear();
    deviceMemorySizes.clear();
    memorySizes.clear();
    blockSizes.clear();
    levelMemorySizes.clear();
    levelBlockSizes.clear();
}

std::vector<size_t> TestParameterStore::getMemorySizesForDevice(int deviceId) const {
    auto it = deviceMemorySizes.find(deviceId);
    if (it == deviceMemorySizes.end()) return memorySizes;
    
    size_t deviceMem = it->second;
    std::vector<size_t> suitableSizes;
    for (auto size : memorySizes) {
        if (size < deviceMem / 2) suitableSizes.push_back(size);
    }
    return suitableSizes.empty() ? memorySizes : suitableSizes;
}

bool TestParameterStore::isFeatureSupported(const std::string& feature) const {
    auto it = supportedFeatures_.find(feature);
    return it != supportedFeatures_.end() && it->second;
}

int TestParameterStore::getOptimalBlockSize(int deviceId) const {
    auto it = deviceMaxThreadsPerBlock.find(deviceId);
    if (it != deviceMaxThreadsPerBlock.end()) {
        int maxThreads = it->second;
        if (maxThreads >= 256) return 256;
        if (maxThreads >= 128) return 128;
        return 64;
    }
    return 256;
}

void TestParameterStore::printConfiguration() const {
    std::cout << "\n[TestParameterStore] Configuration Summary:" << std::endl;
    std::cout << "  Devices: " << deviceIds.size() << std::endl;
    std::cout << "  Test Mode: " << testMode << std::endl;
    std::cout << "  Multi-GPU Tests: " << (enableMultiGPUTests ? "Yes" : "No") << std::endl;
    
    if (!levelMemorySizes.empty()) {
        std::cout << "\n  Level-specific configs:" << std::endl;
        for (const auto& pair : levelMemorySizes) {
            std::cout << "    " << pair.first << ": " << pair.second.size() << " memory sizes" << std::endl;
        }
    }
    std::cout << std::endl;
}

bool TestParameterStore::loadLevelConfig(const std::string& level) {
    if (level.empty()) return false;
    
    std::cout << "  [Level Config] Loading parameters for: " << level << std::endl;
    currentTestLevel = level;
    
    if (levelMemorySizes.count(level) > 0 || levelBlockSizes.count(level) > 0) {
        std::cout << "    Using parameters from centralized config" << std::endl;
        return true;
    }
    
    return false;
}

const std::vector<size_t>& TestParameterStore::getMemorySizesForCurrentLevel() const {
    if (!currentTestLevel.empty()) {
        auto it = levelMemorySizes.find(currentTestLevel);
        if (it != levelMemorySizes.end() && !it->second.empty()) {
            return it->second;
        }
    }
    return memorySizes;
}

const std::vector<int>& TestParameterStore::getBlockSizesForCurrentLevel() const {
    if (!currentTestLevel.empty()) {
        auto it = levelBlockSizes.find(currentTestLevel);
        if (it != levelBlockSizes.end() && !it->second.empty()) {
            return it->second;
        }
    }
    return blockSizes;
}

void DeviceCapabilities::initialize(int deviceId) {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, deviceId);
    if (err != hipSuccess) {
        std::cerr << "[DeviceCapabilities] Failed to get device properties" << std::endl;
        return;
    }
    
    hasUnifiedMemory = (prop.managedMemory == 1);
    hasCooperativeGroups = (prop.cooperativeLaunch == 1);
    maxThreadsPerBlock = prop.maxThreadsPerBlock;
    totalGlobalMem = prop.totalGlobalMem;
    multiProcessorCount = prop.multiProcessorCount;
    warpSize = prop.warpSize;
    gcnArchName = prop.gcnArchName;
}

void DeviceCapabilities::print() const {
    std::cout << "[Device Capabilities]" << std::endl;
    std::cout << "  Architecture: " << gcnArchName << std::endl;
    std::cout << "  Total Memory: " << (totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Max Threads/Block: " << maxThreadsPerBlock << std::endl;
}

