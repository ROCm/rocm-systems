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
#include <hip/hip_runtime.h>

/**
 * @brief Global parameter store for test configuration and runtime-detected values.
 * 
 * Example usage:
 * @code
 * TEST_CASE("My_Test", "[level_0]") {
 *     auto& params = TestParameterStore::instance();
 *     auto size = GENERATE_COPY(from_range(params.getMemorySizesForCurrentLevel()));
 *     // size comes from test_levels.txt: level_0.memory_sizes = 1K, 1M, 10M
 * }
 * @endcode
 */
class TestParameterStore {
public:
    static TestParameterStore& instance() {
        static TestParameterStore inst;
        return inst;
    }

    // Device Information (detected at runtime)
    std::vector<int> deviceIds;                    
    std::vector<std::string> deviceArchs;          
    std::map<int, size_t> deviceMemorySizes;       
    std::map<int, int> deviceComputeCapabilities;  
    std::map<int, int> deviceMaxThreadsPerBlock;   
    
    // Test Parameters
    std::vector<size_t> memorySizes;              
    std::vector<size_t> smallMemorySizes;         
    std::vector<size_t> largeMemorySizes;         
    
    // Level-specific parameters (from test_levels.txt)
    std::map<std::string, std::vector<size_t>> levelMemorySizes;  
    std::map<std::string, std::vector<int>> levelBlockSizes;      
    
    std::vector<int> blockSizes;                   
    std::vector<int> gridSizes;                    
    std::vector<dim3> blockDims2D;                 
    std::vector<dim3> blockDims3D;                 
    
    std::vector<std::string> dataTypes;            
    std::vector<size_t> dataTypeSizes;             
    std::vector<int> streamCounts;                 
    
    // Runtime Configuration
    bool enableExtendedTests = false;              
    bool enableMultiGPUTests = false;              
    bool enablePeerAccessTests = false;            
    std::string testMode = "standard";             
    std::string currentTestLevel = "";             
    
    int defaultIterations = 1000;                  
    int defaultWarmups = 100;                      
    
    void initialize();
    void clear();
    std::vector<size_t> getMemorySizesForDevice(int deviceId) const;
    bool isFeatureSupported(const std::string& feature) const;
    int getOptimalBlockSize(int deviceId) const;
    void printConfiguration() const;
    bool loadLevelConfig(const std::string& level);
    const std::vector<size_t>& getMemorySizesForCurrentLevel() const;
    const std::vector<int>& getBlockSizesForCurrentLevel() const;

private:
    TestParameterStore() = default;
    TestParameterStore(const TestParameterStore&) = delete;
    TestParameterStore& operator=(const TestParameterStore&) = delete;
    
    std::map<std::string, bool> supportedFeatures_;
    
    void detectDeviceCapabilities();
    void loadEnvironmentConfig();
    void loadCentralizedLevelConfig();
    void populateDefaultParameters();
};

struct DeviceCapabilities {
    bool hasUnifiedMemory = false;          
    bool hasPeerAccess = false;             
    bool hasCooperativeGroups = false;      
    bool hasGraphMemory = false;            
    int maxThreadsPerBlock = 0;             
    int maxGridSize[3] = {0, 0, 0};         
    size_t sharedMemPerBlock = 0;           
    size_t totalGlobalMem = 0;              
    int multiProcessorCount = 0;            
    int warpSize = 0;                       
    std::string gcnArchName;                
    
    static DeviceCapabilities& get() {
        static DeviceCapabilities caps;
        return caps;
    }
    
    void initialize(int deviceId = 0);
    void print() const;

private:
    DeviceCapabilities() = default;
};

