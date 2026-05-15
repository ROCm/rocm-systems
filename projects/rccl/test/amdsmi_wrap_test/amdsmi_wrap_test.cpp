/*************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file amdsmi_wrap_test.cpp
 * @brief Standalone test for amdsmi_wrap wrapper functions
 *
 * This test exercises the amdsmi_wrap API without requiring the full RCCL
 * build infrastructure. It tests both the AMD SMI library path and the
 * fallback path.
 *
 * Build: make
 * Run:   ./amdsmi_wrap_test
 *        RCCL_USE_AMD_SMI_LIB=1 ./amdsmi_wrap_test  # to test AMD SMI path
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>

// Include test stubs first (provides ncclResult_t, macros, etc.)
#include "test_stubs.h"

// Include the amdsmi_wrap header
#include "amdsmi_wrap.h"

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << " (line " << __LINE__ << ")" << std::endl; \
        tests_failed++; \
        return false; \
    } \
} while(0)

#define TEST_PASS(msg) do { \
    std::cout << "[PASS] " << msg << std::endl; \
    tests_passed++; \
} while(0)

// Helper to print test section headers
static void printSection(const char* section) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << section << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

// ============================================================================
// Test: amd_smi_init and amd_smi_shutdown
// ============================================================================
bool test_init_shutdown() {
    printSection("amd_smi_init / amd_smi_shutdown");

    ncclResult_t res = amd_smi_init();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_init() should return ncclSuccess");
    TEST_PASS("amd_smi_init() succeeded");

    res = amd_smi_shutdown();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_shutdown() should return ncclSuccess");
    TEST_PASS("amd_smi_shutdown() succeeded");

    return true;
}

// ============================================================================
// Test: amd_smi_getNumDevice
// ============================================================================
bool test_getNumDevice() {
    printSection("amd_smi_getNumDevice");

    ncclResult_t res = amd_smi_init();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_init() should succeed");

    uint32_t numDevs = 0;
    res = amd_smi_getNumDevice(&numDevs);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getNumDevice() should return ncclSuccess");

    std::cout << "  Number of devices detected: " << numDevs << std::endl;
    TEST_PASS("amd_smi_getNumDevice() succeeded");

    amd_smi_shutdown();
    return true;
}

// ============================================================================
// Test: amd_smi_getDevicePciBusIdString
// ============================================================================
bool test_getDevicePciBusIdString() {
    printSection("amd_smi_getDevicePciBusIdString");

    ncclResult_t res = amd_smi_init();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_init() should succeed");

    uint32_t numDevs = 0;
    res = amd_smi_getNumDevice(&numDevs);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getNumDevice() should succeed");

    if (numDevs == 0) {
        std::cout << "  No devices found, skipping PCI bus ID test" << std::endl;
        amd_smi_shutdown();
        return true;
    }

    for (uint32_t i = 0; i < numDevs; i++) {
        char busId[32] = {0};
        res = amd_smi_getDevicePciBusIdString(i, busId, sizeof(busId));
        TEST_ASSERT(res == ncclSuccess, "amd_smi_getDevicePciBusIdString() should return ncclSuccess");

        std::cout << "  Device " << i << " PCI Bus ID: " << busId << std::endl;
        TEST_ASSERT(strlen(busId) > 0, "PCI Bus ID should not be empty");
    }
    TEST_PASS("amd_smi_getDevicePciBusIdString() succeeded for all devices");

    amd_smi_shutdown();
    return true;
}

// ============================================================================
// Test: amd_smi_getDeviceIndexByPciBusId
// ============================================================================
bool test_getDeviceIndexByPciBusId() {
    printSection("amd_smi_getDeviceIndexByPciBusId");

    ncclResult_t res = amd_smi_init();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_init() should succeed");

    uint32_t numDevs = 0;
    res = amd_smi_getNumDevice(&numDevs);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getNumDevice() should succeed");

    if (numDevs == 0) {
        std::cout << "  No devices found, skipping test" << std::endl;
        amd_smi_shutdown();
        return true;
    }

    // Test for all devices
    for (uint32_t i = 0; i < numDevs; i++) {
        // Get bus ID for device i
        char busId[32] = {0};
        res = amd_smi_getDevicePciBusIdString(i, busId, sizeof(busId));
        TEST_ASSERT(res == ncclSuccess, "amd_smi_getDevicePciBusIdString() should succeed");

        // Look up device index by bus ID
        uint32_t deviceIndex = 0xFFFFFFFF;
        res = amd_smi_getDeviceIndexByPciBusId(busId, &deviceIndex);
        TEST_ASSERT(res == ncclSuccess, "amd_smi_getDeviceIndexByPciBusId() should return ncclSuccess");

        std::cout << "  Bus ID " << busId << " -> Device Index: " << deviceIndex << std::endl;
        TEST_ASSERT(deviceIndex == i, "Device index should match the original device");
    }
    TEST_PASS("amd_smi_getDeviceIndexByPciBusId() succeeded for all devices");

    amd_smi_shutdown();
    return true;
}

// ============================================================================
// Test: amd_smi_getLinkInfo
// ============================================================================
bool test_getLinkInfo() {
    printSection("amd_smi_getLinkInfo");

    ncclResult_t res = amd_smi_init();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_init() should succeed");

    uint32_t numDevs = 0;
    res = amd_smi_getNumDevice(&numDevs);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getNumDevice() should succeed");

    if (numDevs < 2) {
        std::cout << "  Less than 2 devices found (" << numDevs << "), skipping link info test" << std::endl;
        amd_smi_shutdown();
        return true;
    }

    amdsmi_link_type_t linkType;
    int hops = 0;
    int count = 0;

    res = amd_smi_getLinkInfo(0, 1, &linkType, &hops, &count);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getLinkInfo() should return ncclSuccess");

    const char* linkTypeStr = "UNKNOWN";
    switch (linkType) {
        case AMDSMI_LINK_TYPE_PCIE: linkTypeStr = "PCIE"; break;
        case AMDSMI_LINK_TYPE_XGMI: linkTypeStr = "XGMI"; break;
        default: break;
    }

    std::cout << "  Link 0->1: Type=" << linkTypeStr << ", Hops=" << hops << ", Count=" << count << std::endl;
    TEST_PASS("amd_smi_getLinkInfo() succeeded");

    amd_smi_shutdown();
    return true;
}

// ============================================================================
// Test: amd_smi_getFirmwareVersion
// ============================================================================
bool test_getFirmwareVersion() {
    printSection("amd_smi_getFirmwareVersion");

    ncclResult_t res = amd_smi_init();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_init() should succeed");

    uint32_t numDevs = 0;
    res = amd_smi_getNumDevice(&numDevs);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getNumDevice() should succeed");

    if (numDevs == 0) {
        std::cout << "  No devices found, skipping firmware version test" << std::endl;
        amd_smi_shutdown();
        return true;
    }

    uint64_t fwVersion = 0;
    res = amd_smi_getFirmwareVersion(0, &fwVersion);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getFirmwareVersion() should return ncclSuccess");

    std::cout << "  Device 0 Firmware Version: " << fwVersion;
    if (fwVersion == 0) {
        std::cout << " (unavailable - RCCL_USE_AMD_SMI_LIB may not be set)";
    }
    std::cout << std::endl;

    TEST_PASS("amd_smi_getFirmwareVersion() succeeded");

    amd_smi_shutdown();
    return true;
}

// ============================================================================
// Test: amd_smi_ensureFabricInitialized and fabric functions
// ============================================================================
bool test_fabric_functions() {
    printSection("Fabric Functions");

    ncclResult_t res = amd_smi_init();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_init() should succeed");

    // Test fabric initialization
    res = amd_smi_ensureFabricInitialized();
    TEST_ASSERT(res == ncclSuccess, "amd_smi_ensureFabricInitialized() should return ncclSuccess");
    TEST_PASS("amd_smi_ensureFabricInitialized() succeeded");

    uint32_t numDevs = 0;
    res = amd_smi_getNumDevice(&numDevs);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getNumDevice() should succeed");

    if (numDevs == 0) {
        std::cout << "  No devices found, skipping detailed fabric tests" << std::endl;
        amd_smi_shutdown();
        return true;
    }

    // Test isFabricSupported
    bool supported = false;
    res = amd_smi_isFabricSupported(0, &supported);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_isFabricSupported() should return ncclSuccess");
    std::cout << "  Device 0 Fabric Supported: " << (supported ? "Yes" : "No") << std::endl;
    TEST_PASS("amd_smi_isFabricSupported() succeeded");

    // Test getFabricBandwidth
    uint32_t bandwidth = 0;
    res = amd_smi_getFabricBandwidth(0, &bandwidth);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getFabricBandwidth() should return ncclSuccess");
    std::cout << "  Device 0 Fabric Bandwidth: " << bandwidth << " Mb/s" << std::endl;
    TEST_PASS("amd_smi_getFabricBandwidth() succeeded");

    // Test getFabricDeviceInfo
    struct amdsmiFabricDeviceInfo devInfo;
    memset(&devInfo, 0, sizeof(devInfo));
    res = amd_smi_getFabricDeviceInfo(0, &devInfo);
    TEST_ASSERT(res == ncclSuccess, "amd_smi_getFabricDeviceInfo() should return ncclSuccess");

    std::cout << "  Device 0 Fabric Info:" << std::endl;
    std::cout << "    Fabric Supported: " << (devInfo.fabricSupported ? "Yes" : "No") << std::endl;
    std::cout << "    Fabric Type: " << devInfo.fabricType << std::endl;
    std::cout << "    State: " << devInfo.state << std::endl;
    std::cout << "    Accelerator ID: " << devInfo.acceleratorId << std::endl;
    std::cout << "    Bandwidth: " << devInfo.bandwidth << " Mb/s" << std::endl;
    std::cout << "    Latency: " << devInfo.latency << " ns" << std::endl;
    std::cout << "    UUID: " << devInfo.clusterUuid << std::endl;
    std::cout << "    pPOD Size: " << devInfo.ppodSize << std::endl;
    std::cout << "    vPOD ID: " << devInfo.cliqueId << std::endl;
    std::cout << "    vPOD Size: " << devInfo.vpodSize << std::endl;
    TEST_PASS("amd_smi_getFabricDeviceInfo() succeeded");

    amd_smi_shutdown();
    return true;
}

// ============================================================================
// Main entry point
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "============================================================" << std::endl;
    std::cout << "       AMD SMI Wrapper (amdsmi_wrap) Standalone Tests       " << std::endl;
    std::cout << "============================================================" << std::endl;

    // Check environment
    const char* useAmdSmiLib = getenv("RCCL_USE_AMD_SMI_LIB");
    std::cout << "\nEnvironment:" << std::endl;
    std::cout << "  RCCL_USE_AMD_SMI_LIB: " << (useAmdSmiLib ? useAmdSmiLib : "not set") << std::endl;

    // Run all tests
    test_init_shutdown();
    test_getNumDevice();
    test_getDevicePciBusIdString();
    test_getDeviceIndexByPciBusId();
    test_getLinkInfo();
    test_getFirmwareVersion();
    test_fabric_functions();

    // Print summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                      Test Summary                          " << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "  Tests Passed: " << tests_passed << std::endl;
    std::cout << "  Tests Failed: " << tests_failed << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    if (tests_failed > 0) {
        std::cout << "\n*** SOME TESTS FAILED ***\n" << std::endl;
        return 1;
    }

    std::cout << "\n*** ALL TESTS PASSED ***\n" << std::endl;
    return 0;
}
