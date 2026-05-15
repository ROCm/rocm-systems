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
 * @addtogroup hipMemCreate hipMemCreate
 * @{
 * @ingroup perfVMMTest
 * `hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
 *               const hipMemAllocationProp* prop, unsigned long long flags)` -
 * Creates a memory allocation described by the properties and size.
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfBufferCopySpeed status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopySpeed.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

// Control Variables
constexpr bool single_map = false;
constexpr bool debug_failure = false;
constexpr size_t kMB = (1024 * 1024);
constexpr size_t kGB = (1024 * 1024 * 1024);
constexpr size_t chunk_size = (64 * kMB);

bool CheckVMMSupportedOnDevice(int deviceId) {
  int value = 0;
  hipDeviceAttribute_t attr = hipDeviceAttributeVirtualMemoryManagementSupported;
  HIP_CHECK(hipDeviceGetAttribute(&value, attr, deviceId));
  return static_cast<bool>(value);
}

bool GetVMMGranularityOnDevice(int deviceId, size_t& granularity) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = deviceId;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  return true;
}

template <typename T> size_t GetSizeN(size_t total_size) {
  if (total_size % sizeof(T) != 0) {
    INFO("Size " << total_size << " is not a multiple of size of type T");
    assert(false);
  }
  return (total_size / sizeof(T));
}

bool ValidateUsingCopy(int deviceId, void* dev_ptr, size_t data_size,
                       std::chrono::microseconds& h2d_elapsed,
                       std::chrono::microseconds& d2h_elapsed) {
  // Get Host Data
  std::vector<int> A_h(data_size), B_h(data_size);
  size_t size_n = GetSizeN<int>(data_size);

  for (size_t idx = 0; idx < size_n; ++idx) {
    A_h[idx] = idx;
    B_h[idx] = 0;
  }

  HIP_CHECK(hipSetDevice(deviceId));
  auto start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemcpy(dev_ptr, A_h.data(), data_size, hipMemcpyHostToDevice));
  auto end = std::chrono::high_resolution_clock::now();
  h2d_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemcpy(B_h.data(), dev_ptr, data_size, hipMemcpyDeviceToHost));
  end = std::chrono::high_resolution_clock::now();
  d2h_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  if (debug_failure) {
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  } else {
    assert(A_h.size() == B_h.size());
    for (size_t idx = 0; idx < A_h.size(); ++idx) {
      if (A_h[idx] != B_h[idx]) {
        std::cout << "Failed at first index: " << idx << " Expected: " << A_h[idx]
                  << " Value: " << B_h[idx] << std::endl;
        break;
      }
    }
  }
  return true;
}

bool TestOnDevice(int deviceId) {
  HIP_CHECK(hipSetDevice(deviceId));
  // Check if VMM is supported
  if (!CheckVMMSupportedOnDevice(deviceId)) {
    INFO("VMM is not suppored on device: " << deviceId);
    return false;
  }
  // Get VMM granularity
  size_t granularity = 0;
  if (!GetVMMGranularityOnDevice(deviceId, granularity)) {
    INFO("Cannot get granularity on device: " << deviceId);
    return false;
  }

  // Measure CPU time of allocation taken
  size_t start_size = 1 * kGB;
  size_t max_size = 64 * kGB;
  for (size_t size_idx = start_size; size_idx <= max_size; (size_idx <<= 1)) {
    void* dev_ptr = nullptr;
    // This seems to be a completely blocking call, measuring CPU time in this test for now.
    // Create Memory Reservation
    auto start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMemAddressReserve(&dev_ptr, size_idx, granularity, nullptr, 0));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds reserve_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::vector<hipMemGenericAllocationHandle_t> physmem_handles;
    std::chrono::microseconds alloc_elapsed;
    std::chrono::microseconds map_elapsed;
    assert(size_idx % chunk_size == 0);
    size_t chunk_max = size_idx / chunk_size;
    size_t chunk_idx = 0;

    size_t freeVRAM = 0, totalVRAM = 0;
    HIP_CHECK(hipMemGetInfo(&freeVRAM, &totalVRAM));
    INFO("Available total device memory : " << totalVRAM);

    if (freeVRAM < size_idx) {
      WARN("Further free device memory unavailable, hence exiting!");
      break;
    }

    if (single_map) {
      // Create Physical memory
      hipMemAllocationProp prop{};
      prop.type = hipMemAllocationTypePinned;
      prop.location.type = hipMemLocationTypeDevice;
      prop.location.id = deviceId;
      hipMemGenericAllocationHandle_t handle;

      start = std::chrono::high_resolution_clock::now();
      HIP_CHECK(hipMemCreate(&handle, size_idx, &prop, 0));
      end = std::chrono::high_resolution_clock::now();
      alloc_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      physmem_handles.push_back(handle);

      // Map the memory
      start = std::chrono::high_resolution_clock::now();
      HIP_CHECK(hipMemMap(dev_ptr, size_idx, 0, handle, 0));
      end = std::chrono::high_resolution_clock::now();
      map_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    } else {
      start = std::chrono::high_resolution_clock::now();
      while (chunk_idx < chunk_max) {
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = deviceId;
        hipMemGenericAllocationHandle_t handle;
        HIP_CHECK(hipMemCreate(&handle, chunk_size, &prop, 0));
        physmem_handles.push_back(handle);
        ++chunk_idx;
      }
      end = std::chrono::high_resolution_clock::now();
      alloc_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      chunk_idx = 0;
      start = std::chrono::high_resolution_clock::now();
      while (chunk_idx < chunk_max) {
        // Use chunk size to map multiple maps
        uint64_t uiptr = reinterpret_cast<uint64_t>(dev_ptr);
        uiptr = uiptr + chunk_idx * chunk_size;
        HIP_CHECK(hipMemMap(reinterpret_cast<void*>(uiptr), chunk_size, 0,
                            physmem_handles[chunk_idx], 0));
        ++chunk_idx;
      }
      end = std::chrono::high_resolution_clock::now();
      map_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      chunk_idx = 0;
      while (chunk_idx < chunk_max) {
        // Set access
        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = deviceId;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        uint64_t uiptr = reinterpret_cast<uint64_t>(dev_ptr);
        uiptr = uiptr + chunk_idx * chunk_size;
        // Make the address accessible to GPU 0
        HIP_CHECK(hipMemSetAccess(reinterpret_cast<void*>(uiptr), chunk_size, &accessDesc, 1));
        ++chunk_idx;
      }
    }

    // Also measure the memcpy time elapsed
    std::chrono::microseconds h2d_elapsed;
    std::chrono::microseconds d2h_elapsed;

    // Validate using copy
    if (!ValidateUsingCopy(deviceId, dev_ptr, size_idx, h2d_elapsed, d2h_elapsed)) {
      INFO("Validation failed for size: " << size_idx);
    }

    start = std::chrono::high_resolution_clock::now();
    chunk_idx = 0;
    while (chunk_idx < chunk_max) {
      uint64_t uiptr = reinterpret_cast<uint64_t>(dev_ptr);
      uiptr = uiptr + chunk_idx * chunk_size;
      HIP_CHECK(hipMemUnmap(reinterpret_cast<void*>(uiptr), chunk_size));
      ++chunk_idx;
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds unmap_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    for (auto& physmem_handle : physmem_handles) {
      HIP_CHECK(hipMemRelease(physmem_handle));
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds release_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMemAddressFree(dev_ptr, size_idx));
    end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds free_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Print the results
    std::cout << "-------------Size: " << (size_idx / kGB) << " GB----------------" << std::endl;
    std::cout << "Time taken to reserve : " << reserve_elapsed.count()
              << " micro seconds and free: " << free_elapsed.count() << " micro seconds"
              << std::endl;
    std::cout << "Time taken to alloc : " << alloc_elapsed.count()
              << " micro seconds and release: " << release_elapsed.count() << " micro seconds"
              << std::endl;
    std::cout << "Time taken to map : " << map_elapsed.count()
              << " micro seconds and unmap: " << unmap_elapsed.count() << " micro seconds"
              << std::endl;
    std::cout << "Time taken to H2D : " << h2d_elapsed.count()
              << " micro seconds and D2H: " << d2h_elapsed.count() << " micro seconds" << std::endl;
    std::cout << "-------------------------/hipMallocPerf------------------------" << std::endl;

    void* dev_ptr_legacy = nullptr;
    start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMalloc(&dev_ptr_legacy, size_idx));
    end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds hm_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipFree(dev_ptr_legacy));
    end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds hf_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken for hipMalloc : " << hm_elapsed.count()
              << " micro seconds and hipFree: " << hf_elapsed.count() << " micro seconds"
              << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
  }

  return true;
}

TEST_CASE("Perf_hipPerfVMMAllocSpeed_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices <= 0) {
    SUCCEED(
        "Skipped testcase hipPerfBufferCopySpeed as"
        "there is no device to test.");
  } else {
    // Test on Primary Device first
    int deviceId = 0;
    TestOnDevice(deviceId);
  }
}
/**
 * End doxygen group perfVMMTest.
 * @}
 */
