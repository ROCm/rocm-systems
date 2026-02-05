/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * @addtogroup hipMemSetAccess hipMemSetAccess
 * @{
 * @ingroup perfVMMTest
 * `hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count)` -
 * Set the access flags for each location specified in desc for the given virtual address range.
 *
 * This test specifically measures hipMemSetAccess latency for cross-GPU access,
 * which was identified as a critical performance bottleneck in RCCL workloads.
 *
 * Findings from RCCL performance analysis:
 *   - GOOD system: hipMemSetAccess takes ~1ms for cross-GPU access
 *   - BAD system:  hipMemSetAccess takes ~200ms for cross-GPU access (200x slower!)
 *   - This causes RCCL bus bandwidth to drop from 31 GB/s to 2.5 GB/s
 */

#include <hip_test_common.hh>
#include <chrono>
#include <vector>
#include <iomanip>

// Constants
constexpr size_t kMB = (1024 * 1024);
constexpr size_t kGB = (1024 * 1024 * 1024);
constexpr size_t kDefaultAllocSize = 1 * kGB;  // 1GB allocation

// Thresholds for pass/fail (in milliseconds)
constexpr double kSetAccessWarnThresholdMs = 10.0;    // Warning if > 10ms
constexpr double kSetAccessFailThresholdMs = 100.0;   // Fail if > 100ms

/**
 * Check if VMM is supported on the device
 */
bool CheckVMMSupported(int deviceId) {
  int value = 0;
  hipDeviceAttribute_t attr = hipDeviceAttributeVirtualMemoryManagementSupported;
  HIP_CHECK(hipDeviceGetAttribute(&value, attr, deviceId));
  return static_cast<bool>(value);
}

/**
 * Get VMM allocation granularity for a device
 */
size_t GetGranularity(int deviceId) {
  size_t granularity = 0;
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = deviceId;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                            hipMemAllocationGranularityMinimum));
  return granularity;
}

/**
 * Align size to granularity
 */
size_t AlignSize(size_t size, size_t granularity) {
  return ((size + granularity - 1) / granularity) * granularity;
}

/**
 * Structure to hold VMM timing results
 */
struct VMMTimings {
  double create_us;
  double address_reserve_us;
  double map_us;
  double set_access_us;
  double unmap_us;
  double release_us;
  double address_free_us;
};

/**
 * Test VMM operations for a specific owner/accessor GPU pair
 * This measures the latency of each VMM API call
 */
VMMTimings TestVMMPair(int ownerDevice, int accessorDevice, size_t allocSize) {
  VMMTimings timings = {};

  HIP_CHECK(hipSetDevice(ownerDevice));

  size_t granularity = GetGranularity(ownerDevice);
  size_t alignedSize = AlignSize(allocSize, granularity);

  // Setup allocation properties
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = ownerDevice;

  hipMemGenericAllocationHandle_t handle;
  hipDeviceptr_t ptr = 0;

  // 1. hipMemCreate
  auto start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemCreate(&handle, alignedSize, &prop, 0));
  auto end = std::chrono::high_resolution_clock::now();
  timings.create_us = std::chrono::duration<double, std::micro>(end - start).count();

  // 2. hipMemAddressReserve
  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemAddressReserve(&ptr, alignedSize, granularity, 0, 0));
  end = std::chrono::high_resolution_clock::now();
  timings.address_reserve_us = std::chrono::duration<double, std::micro>(end - start).count();

  // 3. hipMemMap
  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemMap(ptr, alignedSize, 0, handle, 0));
  end = std::chrono::high_resolution_clock::now();
  timings.map_us = std::chrono::duration<double, std::micro>(end - start).count();

  // 4. hipMemSetAccess - THE CRITICAL OPERATION
  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = accessorDevice;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;

  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemSetAccess(ptr, alignedSize, &accessDesc, 1));
  end = std::chrono::high_resolution_clock::now();
  timings.set_access_us = std::chrono::duration<double, std::micro>(end - start).count();

  // 5. hipMemUnmap
  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemUnmap(ptr, alignedSize));
  end = std::chrono::high_resolution_clock::now();
  timings.unmap_us = std::chrono::duration<double, std::micro>(end - start).count();

  // 6. hipMemRelease
  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemRelease(handle));
  end = std::chrono::high_resolution_clock::now();
  timings.release_us = std::chrono::duration<double, std::micro>(end - start).count();

  // 7. hipMemAddressFree
  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemAddressFree(ptr, alignedSize));
  end = std::chrono::high_resolution_clock::now();
  timings.address_free_us = std::chrono::duration<double, std::micro>(end - start).count();

  return timings;
}

/**
 * Test Description
 * ------------------------
 *  - Measures hipMemSetAccess latency for cross-GPU VMM access.
 *  - This is critical for RCCL performance as slow cross-GPU VMM setup
 *    causes severe collective communication degradation.
 * Test source
 * ------------------------
 *  - perftests/vmm/hipPerfVMMSetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 *  - Multiple GPUs recommended
 */
TEST_CASE("Perf_hipMemSetAccess_CrossGPU") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices < 1) {
    HipTest::HIP_SKIP_TEST("No devices available");
    return;
  }

  // Check VMM support on device 0
  if (!CheckVMMSupported(0)) {
    HipTest::HIP_SKIP_TEST("VMM not supported on device 0");
    return;
  }

  const int numIterations = 5;
  const size_t allocSize = kDefaultAllocSize;

  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << "  hipMemSetAccess Cross-GPU Performance Test\n";
  std::cout << "  Allocation size: " << (allocSize / kMB) << " MB\n";
  std::cout << "  Iterations: " << numIterations << "\n";
  std::cout << "================================================================\n\n";

  // Print header
  std::cout << std::setw(10) << "Owner"
            << std::setw(10) << "Accessor"
            << std::setw(15) << "Create(ms)"
            << std::setw(15) << "Reserve(ms)"
            << std::setw(15) << "Map(ms)"
            << std::setw(18) << "SetAccess(ms)"
            << std::setw(15) << "Unmap(ms)"
            << std::setw(15) << "Release(ms)"
            << "\n";
  std::cout << std::string(108, '-') << "\n";

  double maxSetAccessMs = 0.0;
  double minSetAccessMs = 1e9;
  double totalSetAccessMs = 0.0;
  int pairCount = 0;
  std::vector<std::pair<int, int>> slowPairs;

  // Test each GPU pair
  for (int owner = 0; owner < numDevices; ++owner) {
    if (!CheckVMMSupported(owner)) {
      continue;
    }

    for (int accessor = 0; accessor < numDevices; ++accessor) {
      if (!CheckVMMSupported(accessor)) {
        continue;
      }

      // Average over iterations
      VMMTimings avgTimings = {};

      for (int iter = 0; iter < numIterations; ++iter) {
        VMMTimings t = TestVMMPair(owner, accessor, allocSize);
        avgTimings.create_us += t.create_us;
        avgTimings.address_reserve_us += t.address_reserve_us;
        avgTimings.map_us += t.map_us;
        avgTimings.set_access_us += t.set_access_us;
        avgTimings.unmap_us += t.unmap_us;
        avgTimings.release_us += t.release_us;
      }

      // Compute averages
      avgTimings.create_us /= numIterations;
      avgTimings.address_reserve_us /= numIterations;
      avgTimings.map_us /= numIterations;
      avgTimings.set_access_us /= numIterations;
      avgTimings.unmap_us /= numIterations;
      avgTimings.release_us /= numIterations;

      double setAccessMs = avgTimings.set_access_us / 1000.0;

      // Track statistics
      maxSetAccessMs = std::max(maxSetAccessMs, setAccessMs);
      minSetAccessMs = std::min(minSetAccessMs, setAccessMs);
      totalSetAccessMs += setAccessMs;
      pairCount++;

      if (setAccessMs > kSetAccessWarnThresholdMs) {
        slowPairs.push_back({owner, accessor});
      }

      // Print results
      std::cout << std::setw(10) << owner
                << std::setw(10) << accessor
                << std::setw(15) << std::fixed << std::setprecision(3)
                << (avgTimings.create_us / 1000.0)
                << std::setw(15) << (avgTimings.address_reserve_us / 1000.0)
                << std::setw(15) << (avgTimings.map_us / 1000.0)
                << std::setw(18) << setAccessMs
                << std::setw(15) << (avgTimings.unmap_us / 1000.0)
                << std::setw(15) << (avgTimings.release_us / 1000.0)
                << "\n";
    }
  }

  // Print summary
  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << "  SUMMARY - hipMemSetAccess Latency\n";
  std::cout << "================================================================\n";
  std::cout << "  Minimum: " << std::fixed << std::setprecision(3) << minSetAccessMs << " ms\n";
  std::cout << "  Maximum: " << maxSetAccessMs << " ms\n";
  std::cout << "  Average: " << (totalSetAccessMs / pairCount) << " ms\n";
  std::cout << "  Spread (max/min): " << std::setprecision(1)
            << (maxSetAccessMs / minSetAccessMs) << "x\n\n";

  if (!slowPairs.empty()) {
    std::cout << "  Slow GPU pairs (> " << kSetAccessWarnThresholdMs << " ms):\n";
    for (const auto& p : slowPairs) {
      std::cout << "    GPU " << p.first << " -> GPU " << p.second << "\n";
    }
    std::cout << "\n";
  }

  // Performance assessment with REQUIRE
  if (maxSetAccessMs > kSetAccessFailThresholdMs) {
    std::cout << "  RESULT: CRITICAL - hipMemSetAccess is extremely slow!\n";
    std::cout << "          Expected: < " << kSetAccessFailThresholdMs << " ms\n";
    std::cout << "          Got: " << maxSetAccessMs << " ms\n";
    std::cout << "          This will cause severe RCCL performance degradation.\n\n";
    std::cout << "  Possible causes:\n";
    std::cout << "    1. NUMA topology issues\n";
    std::cout << "    2. IOMMU misconfiguration\n";
    std::cout << "    3. PCIe ACS (Access Control Services) settings\n";
    std::cout << "    4. GPU firmware issues\n";
    std::cout << "    5. BIOS settings affecting PCIe/XGMI\n";
  } else if (maxSetAccessMs > kSetAccessWarnThresholdMs) {
    std::cout << "  RESULT: WARNING - hipMemSetAccess latency is elevated\n";
    std::cout << "          This may impact multi-GPU performance.\n";
  } else {
    std::cout << "  RESULT: PASS - hipMemSetAccess latency is within normal range\n";
  }

  std::cout << "================================================================\n\n";

  // Assert that max latency is below critical threshold
  REQUIRE(maxSetAccessMs < kSetAccessFailThresholdMs);
}

/**
 * Test Description
 * ------------------------
 *  - Stress test: Rapid VMM allocation/deallocation with cross-GPU access.
 *  - Simulates RCCL behavior during communicator initialization.
 * Test source
 * ------------------------
 *  - perftests/vmm/hipPerfVMMSetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 *  - At least 2 GPUs
 */
TEST_CASE("Perf_hipMemSetAccess_StressTest") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices < 2) {
    HipTest::HIP_SKIP_TEST("Stress test requires at least 2 GPUs");
    return;
  }

  if (!CheckVMMSupported(0) || !CheckVMMSupported(1)) {
    HipTest::HIP_SKIP_TEST("VMM not supported on required devices");
    return;
  }

  const int numAllocations = 20;
  const size_t allocSize = 64 * kMB;  // 64MB each

  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << "  VMM Stress Test - Rapid Cross-GPU Allocations\n";
  std::cout << "  Simulates RCCL communicator initialization behavior\n";
  std::cout << "  Allocations: " << numAllocations << " x " << (allocSize / kMB) << " MB\n";
  std::cout << "================================================================\n\n";

  auto totalStart = std::chrono::high_resolution_clock::now();

  std::vector<double> setAccessTimes;

  for (int i = 0; i < numAllocations; ++i) {
    int owner = i % numDevices;
    int accessor = (i + 1) % numDevices;  // Always cross-GPU

    VMMTimings t = TestVMMPair(owner, accessor, allocSize);
    double setAccessMs = t.set_access_us / 1000.0;
    setAccessTimes.push_back(setAccessMs);

    std::cout << "  Alloc " << std::setw(2) << i
              << ": GPU " << owner << " -> " << accessor
              << "  SetAccess: " << std::fixed << std::setprecision(3)
              << setAccessMs << " ms\n";
  }

  auto totalEnd = std::chrono::high_resolution_clock::now();
  double totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

  // Calculate statistics
  double sum = 0, maxT = 0, minT = 1e9;
  for (double t : setAccessTimes) {
    sum += t;
    maxT = std::max(maxT, t);
    minT = std::min(minT, t);
  }
  double avgT = sum / setAccessTimes.size();

  std::cout << "\n";
  std::cout << "  Total time: " << std::fixed << std::setprecision(1) << totalTimeMs << " ms\n";
  std::cout << "  SetAccess - Min: " << std::setprecision(3) << minT << " ms, "
            << "Max: " << maxT << " ms, Avg: " << avgT << " ms\n";

  if (maxT > kSetAccessFailThresholdMs) {
    std::cout << "\n  RESULT: STRESS TEST FAILED - Cross-GPU VMM access is too slow!\n";
  } else {
    std::cout << "\n  RESULT: STRESS TEST PASSED\n";
  }

  std::cout << "================================================================\n\n";

  // Assert stress test passes
  REQUIRE(maxT < kSetAccessFailThresholdMs);
}

/**
 * End doxygen group perfVMMTest.
 * @}
 */
