#pragma once

/**
 * @file hip_test_config.hh
 * @brief Central configuration for HIP tests with 5 levels (Catch2 v2 compatible)
 * 
 * Level Definitions:
 * - Level_0 (Smoke):     Emulation-safe, Core HIP functionalities, <100ms
 * - Level_1 (Light):     Emulation-safe, Simple operations, minimal data (<1MB), <500ms
 * - Level_2 (Standard):  Typical use cases, moderate data (1-100MB), <5s
 * - Level_3 (Complex):   Advanced features, larger data, complex scenarios, <30s
 * - Level_4 (Stress):    Large data (>1GB), many iterations, edge cases, Variable
 * 
 * Usage with Catch2 v2:
 *   TEST_CASE("MyTest", "[Level_2][Memory]") {
 *       auto& cfg = HIP_TEST_CONFIG(2);  // Manually specify level
 *       size_t size = cfg.memory.allocSize;
 *   }
 */

#include <string>
#include <cstddef>
#include <algorithm>

namespace HipTest {

// ============================================================================
// Configuration Structures (Organized by Category)
// ============================================================================

struct MemoryConfig {
    size_t allocSize;
    size_t maxAllocSize;
    size_t alignmentSize;
    int    numAllocations;
    bool   testPinnedMemory;
    bool   testManagedMemory;
    bool   testLargePages;
};

struct KernelConfig {
    int    threadsPerBlock;
    int    numBlocks;
    int    sharedMemSize;
    int    maxRegisters;
    bool   testDynamicShared;
    bool   testCooperative;
};

struct StreamConfig {
    int    numStreams;
    int    operationsPerStream;
    bool   testPriorities;
    bool   testCallbacks;
    bool   testAsync;
};

struct MemcpyConfig {
    size_t copySize;
    int    numIterations;
    bool   testH2D;
    bool   testD2H;
    bool   testD2D;
    bool   testAsync;
    bool   test2D;
    bool   test3D;
};

struct EventConfig {
    int    numEvents;
    bool   testTiming;
    bool   testBlocking;
    bool   testMultiStream;
};

struct GraphConfig {
    int    numNodes;
    int    numIterations;
    bool   testCapture;
    bool   testClone;
    bool   testExec;
};

struct MultiGPUConfig {
    int    numGPUs;
    bool   testP2P;
    bool   testP2PAccess;
    bool   testP2PCopy;
};

struct IPCConfig {
    int    numProcesses;
    bool   testMemHandle;
    bool   testEventHandle;
    size_t bufferSize;
};

struct PerformanceConfig {
    int    warmupIterations;
    int    measureIterations;
    bool   reportBandwidth;
    bool   reportLatency;
    bool   reportThroughput;
};

struct TextureConfig {
    size_t width;
    size_t height;
    size_t depth;
    bool   test1D;
    bool   test2D;
    bool   test3D;
    bool   testLayered;
};

struct ModuleConfig {
    int    numModules;
    bool   testCodeObject;
    bool   testFatBinary;
};

// ============================================================================
// Main Test Configuration
// ============================================================================

struct TestConfig {
    int         level;
    std::string levelName;
    std::string useCase;
    int         maxRuntimeMs;
    
    MemoryConfig      memory;
    KernelConfig      kernel;
    StreamConfig      stream;
    MemcpyConfig      memcpy;
    EventConfig       event;
    GraphConfig       graph;
    MultiGPUConfig    multiGPU;
    IPCConfig         ipc;
    PerformanceConfig performance;
    TextureConfig     texture;
    ModuleConfig      module;
    
    int    timeoutSeconds;
    bool   enableValidation;
    bool   enableTiming;
    bool   testNegativeCases;
    bool   testEdgeCases;
    
    // Get configured instance for a specific level
    static TestConfig& getConfig(int level) {
        static TestConfig instances[5];  // For levels 0, 1, 2, 3, 4
        
        // Clamp level to valid range
        int idx = level;
        if (idx < 0 || idx > 4) idx = 0;
        
        // Configure if not already done
        if (instances[idx].level == 0 && idx != 0) {
            configureLevel(instances[idx], level);
        } else if (idx == 0 && instances[0].levelName.empty()) {
            configureLevel(instances[0], 0);
        }
        
        return instances[idx];
    }
    
private:
    TestConfig() : level(0) {}
    
    static void configureLevel(TestConfig& cfg, int level);
};

// ========================================================================
// Level 0: Smoke
// ========================================================================
inline void configureLevelSmoke(TestConfig& cfg) {
    cfg.level = 0;
    cfg.levelName = "Smoke";
    cfg.useCase = "Emulation-safe. Core HIP functionalities";
    cfg.maxRuntimeMs = 100;
    cfg.timeoutSeconds = 10;
    cfg.enableValidation = false;
    cfg.enableTiming = false;
    cfg.testNegativeCases = false;
    cfg.testEdgeCases = false;
    
    cfg.memory.allocSize = 1024;
    cfg.memory.maxAllocSize = 4096;
    cfg.memory.numAllocations = 1;
    cfg.memory.testPinnedMemory = false;
    cfg.memory.testManagedMemory = false;
    
    cfg.kernel.threadsPerBlock = 64;
    cfg.kernel.numBlocks = 1;
    cfg.kernel.sharedMemSize = 0;
    
    cfg.stream.numStreams = 1;
    cfg.stream.operationsPerStream = 5;
    cfg.stream.testAsync = false;
    
    cfg.memcpy.copySize = 1024;
    cfg.memcpy.numIterations = 5;
    cfg.memcpy.testH2D = true;
    cfg.memcpy.testD2H = true;
    cfg.memcpy.testD2D = false;
    cfg.memcpy.testAsync = false;
    cfg.memcpy.test2D = false;
    cfg.memcpy.test3D = false;
    
    cfg.event.numEvents = 2;
    cfg.event.testTiming = false;
    
    cfg.graph.numNodes = 2;
    cfg.graph.numIterations = 1;
    
    cfg.multiGPU.numGPUs = 1;
    cfg.multiGPU.testP2P = false;
    
    cfg.ipc.testMemHandle = false;
    cfg.ipc.testEventHandle = false;
    
    cfg.performance.warmupIterations = 0;
    cfg.performance.measureIterations = 1;
    
    cfg.texture.width = 64;
    cfg.texture.height = 1;
    cfg.texture.test1D = true;
    cfg.texture.test2D = false;
}

// ========================================================================
// Level 1: Light
// ========================================================================
inline void configureLevelLight(TestConfig& cfg) {
    cfg.level = 1;
    cfg.levelName = "Light";
    cfg.useCase = "Emulation-safe. Simple operations, minimal data (< 1MB)";
    cfg.maxRuntimeMs = 500;
    cfg.timeoutSeconds = 30;
    cfg.enableValidation = true;
    cfg.enableTiming = false;
    cfg.testNegativeCases = false;
    cfg.testEdgeCases = false;
    
    cfg.memory.allocSize = 256 * 1024;
    cfg.memory.maxAllocSize = 1024 * 1024;
    cfg.memory.numAllocations = 2;
    cfg.memory.testPinnedMemory = true;
    
    cfg.kernel.threadsPerBlock = 128;
    cfg.kernel.numBlocks = 4;
    cfg.kernel.sharedMemSize = 1024;
    
    cfg.stream.numStreams = 2;
    cfg.stream.operationsPerStream = 20;
    cfg.stream.testAsync = true;
    
    cfg.memcpy.copySize = 256 * 1024;
    cfg.memcpy.numIterations = 10;
    cfg.memcpy.testD2D = true;
    cfg.memcpy.testAsync = true;
    
    cfg.event.numEvents = 4;
    cfg.event.testTiming = true;
    
    cfg.graph.numNodes = 4;
    cfg.graph.numIterations = 5;
    
    cfg.performance.warmupIterations = 2;
    cfg.performance.measureIterations = 5;
    
    cfg.texture.width = 128;
    cfg.texture.height = 128;
    cfg.texture.test2D = true;
}

// ========================================================================
// Level 2: Standard
// ========================================================================
inline void configureLevelStandard(TestConfig& cfg) {
    cfg.level = 2;
    cfg.levelName = "Standard";
    cfg.useCase = "Typical use cases, moderate data (1-100MB)";
    cfg.maxRuntimeMs = 5000;
    cfg.timeoutSeconds = 60;
    cfg.enableValidation = true;
    cfg.enableTiming = true;
    cfg.testNegativeCases = true;
    cfg.testEdgeCases = false;
    
    cfg.memory.allocSize = 16 * 1024 * 1024;
    cfg.memory.maxAllocSize = 100 * 1024 * 1024;
    cfg.memory.numAllocations = 5;
    cfg.memory.testPinnedMemory = true;
    cfg.memory.testManagedMemory = true;
    
    cfg.kernel.threadsPerBlock = 256;
    cfg.kernel.numBlocks = 32;
    cfg.kernel.sharedMemSize = 8192;
    cfg.kernel.testDynamicShared = true;
    
    cfg.stream.numStreams = 4;
    cfg.stream.operationsPerStream = 50;
    cfg.stream.testPriorities = true;
    cfg.stream.testCallbacks = true;
    cfg.stream.testAsync = true;
    
    cfg.memcpy.copySize = 16 * 1024 * 1024;
    cfg.memcpy.numIterations = 50;
    cfg.memcpy.testAsync = true;
    cfg.memcpy.test2D = true;
    
    cfg.event.numEvents = 8;
    cfg.event.testMultiStream = true;
    
    cfg.graph.numNodes = 16;
    cfg.graph.numIterations = 20;
    cfg.graph.testCapture = true;
    
    cfg.multiGPU.numGPUs = 2;
    cfg.multiGPU.testP2P = true;
    
    cfg.performance.warmupIterations = 5;
    cfg.performance.measureIterations = 50;
    cfg.performance.reportBandwidth = true;
    
    cfg.texture.width = 512;
    cfg.texture.height = 512;
    cfg.texture.depth = 64;
    cfg.texture.test3D = true;
}

// ========================================================================
// Level 3: Complex
// ========================================================================
inline void configureLevelComplex(TestConfig& cfg) {
    cfg.level = 3;
    cfg.levelName = "Complex";
    cfg.useCase = "Advanced features, larger data, complex scenarios";
    cfg.maxRuntimeMs = 30000;
    cfg.timeoutSeconds = 120;
    cfg.enableValidation = true;
    cfg.enableTiming = true;
    cfg.testNegativeCases = true;
    cfg.testEdgeCases = true;
    
    cfg.memory.allocSize = 256 * 1024 * 1024;
    cfg.memory.maxAllocSize = 512 * 1024 * 1024;
    cfg.memory.numAllocations = 10;
    cfg.memory.testPinnedMemory = true;
    cfg.memory.testManagedMemory = true;
    cfg.memory.testLargePages = true;
    
    cfg.kernel.threadsPerBlock = 512;
    cfg.kernel.numBlocks = 128;
    cfg.kernel.sharedMemSize = 32768;
    cfg.kernel.testDynamicShared = true;
    cfg.kernel.testCooperative = true;
    
    cfg.stream.numStreams = 16;
    cfg.stream.operationsPerStream = 100;
    cfg.stream.testPriorities = true;
    cfg.stream.testCallbacks = true;
    cfg.stream.testAsync = true;
    
    cfg.memcpy.copySize = 256 * 1024 * 1024;
    cfg.memcpy.numIterations = 100;
    cfg.memcpy.testAsync = true;
    cfg.memcpy.test2D = true;
    cfg.memcpy.test3D = true;
    
    cfg.event.numEvents = 32;
    cfg.event.testTiming = true;
    cfg.event.testMultiStream = true;
    
    cfg.graph.numNodes = 64;
    cfg.graph.numIterations = 50;
    cfg.graph.testCapture = true;
    cfg.graph.testClone = true;
    
    cfg.multiGPU.numGPUs = 2;
    cfg.multiGPU.testP2P = true;
    cfg.multiGPU.testP2PAccess = true;
    cfg.multiGPU.testP2PCopy = true;
    
    cfg.ipc.numProcesses = 2;
    cfg.ipc.testMemHandle = true;
    cfg.ipc.testEventHandle = true;
    cfg.ipc.bufferSize = 256 * 1024 * 1024;
    
    cfg.performance.warmupIterations = 10;
    cfg.performance.measureIterations = 100;
    cfg.performance.reportBandwidth = true;
    cfg.performance.reportLatency = true;
    
    cfg.texture.width = 1024;
    cfg.texture.height = 1024;
    cfg.texture.depth = 256;
    cfg.texture.testLayered = true;
    
    cfg.module.numModules = 4;
    cfg.module.testCodeObject = true;
    cfg.module.testFatBinary = true;
}

// ========================================================================
// Level 4: Stress
// ========================================================================
inline void configureLevelStress(TestConfig& cfg) {
    cfg.level = 4;
    cfg.levelName = "Stress";
    cfg.useCase = "Stress testing. Large data (> 1GB), many iterations";
    cfg.maxRuntimeMs = -1;
    cfg.timeoutSeconds = 7200;
    cfg.enableValidation = true;
    cfg.enableTiming = true;
    cfg.testNegativeCases = true;
    cfg.testEdgeCases = true;
    
    cfg.memory.allocSize = 1024 * 1024 * 1024;
    cfg.memory.maxAllocSize = 4ULL * 1024 * 1024 * 1024;
    cfg.memory.numAllocations = 100;
    cfg.memory.testPinnedMemory = true;
    cfg.memory.testManagedMemory = true;
    cfg.memory.testLargePages = true;
    
    cfg.kernel.threadsPerBlock = 1024;
    cfg.kernel.numBlocks = 2048;
    cfg.kernel.sharedMemSize = 49152;
    cfg.kernel.testDynamicShared = true;
    cfg.kernel.testCooperative = true;
    
    cfg.stream.numStreams = 128;
    cfg.stream.operationsPerStream = 5000;
    cfg.stream.testPriorities = true;
    cfg.stream.testCallbacks = true;
    cfg.stream.testAsync = true;
    
    cfg.memcpy.copySize = 1024 * 1024 * 1024;
    cfg.memcpy.numIterations = 10000;
    cfg.memcpy.testAsync = true;
    cfg.memcpy.test2D = true;
    cfg.memcpy.test3D = true;
    
    cfg.event.numEvents = 256;
    cfg.event.testTiming = true;
    cfg.event.testMultiStream = true;
    
    cfg.graph.numNodes = 512;
    cfg.graph.numIterations = 10000;
    cfg.graph.testCapture = true;
    cfg.graph.testClone = true;
    
    cfg.multiGPU.numGPUs = 2;
    cfg.multiGPU.testP2P = true;
    cfg.multiGPU.testP2PAccess = true;
    cfg.multiGPU.testP2PCopy = true;
    
    cfg.ipc.numProcesses = 16;
    cfg.ipc.testMemHandle = true;
    cfg.ipc.testEventHandle = true;
    cfg.ipc.bufferSize = 1024 * 1024 * 1024;
    
    cfg.performance.warmupIterations = 1000;
    cfg.performance.measureIterations = 100000;
    cfg.performance.reportBandwidth = true;
    cfg.performance.reportLatency = true;
    cfg.performance.reportThroughput = true;
    
    cfg.texture.width = 4096;
    cfg.texture.height = 4096;
    cfg.texture.depth = 1024;
    cfg.texture.test3D = true;
    cfg.texture.testLayered = true;
    
    cfg.module.numModules = 16;
    cfg.module.testCodeObject = true;
    cfg.module.testFatBinary = true;
}

// ========================================================================
// Configuration dispatcher
// ========================================================================
inline void TestConfig::configureLevel(TestConfig& cfg, int level) {
    switch (level) {
        case 0: configureLevelSmoke(cfg); break;
        case 1: configureLevelLight(cfg); break;
        case 2: configureLevelStandard(cfg); break;
        case 3: configureLevelComplex(cfg); break;
        case 4: configureLevelStress(cfg); break;
        default: configureLevelSmoke(cfg); break;
    }
}

} // namespace HipTest

// Convenience macro - specify level in test
#define HIP_TEST_CONFIG(level) HipTest::TestConfig::getConfig(level)
