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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

#include <algorithm>
#include <array>

namespace {
// Test value constant
constexpr int kTestValueBase = 100;

// Standard buffer size
constexpr size_t kTestBufferElements = 1024;
constexpr size_t kTestBufferBytes = kTestBufferElements * sizeof(int);

// Macro for tests requiring managed access support
#define REQUIRE_MANAGED_ACCESS_DEVICE(device_var)                                                  \
  int device_var = 0;                                                                              \
  HIP_CHECK(hipSetDevice(device_var));                                                             \
  if (!DeviceAttributesSupport(device_var, hipDeviceAttributeConcurrentManagedAccess)) {           \
    HipTest::HIP_SKIP_TEST("Device does not support concurrent managed access");                   \
    return;                                                                                        \
  }

}  // namespace

// Kernel to verify data integrity on device
__global__ void VerifyDataKernel(const int* data, size_t num_elements, int base_value,
                                 bool* success_flag) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    if (data[idx] != base_value) {
      *success_flag = false;
    }
  }
}

// Helper function to verify data integrity on device
static void VerifyDataOnDevice(int* data, hipStream_t stream) {
  bool* success_flag;
  HIP_CHECK(hipMallocManaged(&success_flag, sizeof(bool)));
  *success_flag = true;

  constexpr int kBlockSize = 256;
  int num_blocks = (kTestBufferElements + kBlockSize - 1) / kBlockSize;

  VerifyDataKernel<<<num_blocks, kBlockSize, 0, stream>>>(data, kTestBufferElements, kTestValueBase,
                                                          success_flag);

  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(*success_flag == true);
  HIP_CHECK(hipFree(success_flag));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test for single batch operation with single location
 *  - Allocates managed memory, writes data, prefetches to device, verifies data
 *  - Validates that prefetch actually occurred using hipMemRangeGetAttribute
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_SingleOperationSingleLocation") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  std::fill_n(managed_memory.ptr(), kTestBufferElements, kTestValueBase);

  StreamGuard stream_guard(Streams::created);

  std::array<void*, 1> managed_ptrs = {managed_memory.ptr()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};

  std::array<hipMemLocation, 1> locations;
  locations[0].type = hipMemLocationTypeDevice;
  locations[0].id = device;

  std::array<size_t, 1> location_indices = {0};
  constexpr unsigned long long flags = 0;

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     locations.data(), location_indices.data(), locations.size(),
                                     flags, stream_guard.stream()));

  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  VerifyDataOnDevice(managed_memory.ptr(), stream_guard.stream());

  int last_prefetch_location = -1;
  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, managed_memory.ptr(),
                                    kTestBufferBytes));
  REQUIRE(last_prefetch_location == device);
}

/**
 * Test Description
 * ------------------------
 *  - Tests various location distribution patterns:
 *    1. All operations to same location (numPrefetchLocs=1)
 *    2. Each operation to different location (numPrefetchLocs=count)
 *    3. Mixed grouped locations (some to location A, others to B)
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_LocationDistribution") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  enum class DistributionPattern { AllSame, EachDifferent, MixedGrouped };

  auto [pattern, num_operations, description] =
      GENERATE(table<DistributionPattern, size_t, const char*>(
          {{DistributionPattern::AllSame, 4, "all operations same location"},
           {DistributionPattern::EachDifferent, 4, "each operation different location"},
           {DistributionPattern::MixedGrouped, 6, "mixed grouped locations"}}));

  DYNAMIC_SECTION(description) {
    std::vector<void*> managed_ptrs(num_operations);
    std::vector<size_t> buffer_sizes(num_operations, kTestBufferBytes);

    for (size_t op = 0; op < num_operations; op++) {
      HIP_CHECK(hipMallocManaged(&managed_ptrs[op], kTestBufferBytes));
      std::fill_n(static_cast<int*>(managed_ptrs[op]), kTestBufferElements, kTestValueBase);
    }

    StreamGuard stream_guard(Streams::created);

    std::vector<hipMemLocation> locations;
    std::vector<size_t> location_indices;

    switch (pattern) {
      case DistributionPattern::AllSame:
        locations.resize(1);
        locations[0].type = hipMemLocationTypeDevice;
        locations[0].id = device;
        location_indices = {0};
        break;

      case DistributionPattern::EachDifferent:
        locations.resize(num_operations);
        location_indices.resize(num_operations);
        for (size_t i = 0; i < num_operations; i++) {
          if (i % 2 == 0) {
            locations[i].type = hipMemLocationTypeDevice;
            locations[i].id = device;
          } else {
            locations[i].type = hipMemLocationTypeHost;
            locations[i].id = 0;
          }
          location_indices[i] = i;
        }
        break;

      case DistributionPattern::MixedGrouped:
        locations.resize(2);
        locations[0].type = hipMemLocationTypeDevice;
        locations[0].id = device;
        locations[1].type = hipMemLocationTypeHost;
        locations[1].id = 0;
        location_indices = {0, 3};
        break;
    }

    constexpr unsigned long long flags = 0;
    HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                       locations.data(), location_indices.data(), locations.size(),
                                       flags, stream_guard.stream()));

    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

    for (size_t op = 0; op < num_operations; op++) {
      int last_prefetch_location = -1;
      HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                        hipMemRangeAttributeLastPrefetchLocation, managed_ptrs[op],
                                        kTestBufferBytes));

      int expected_device = -1;
      switch (pattern) {
        case DistributionPattern::AllSame:
          expected_device = device;
          break;
        case DistributionPattern::EachDifferent:
          expected_device = (op % 2 == 0) ? device : hipCpuDeviceId;
          break;
        case DistributionPattern::MixedGrouped:
          expected_device = (op < 3) ? device : hipCpuDeviceId;
          break;
      }
      REQUIRE(last_prefetch_location == expected_device);

      if (expected_device == device) {
        VerifyDataOnDevice(static_cast<int*>(managed_ptrs[op]), stream_guard.stream());
      } else {
        ArrayFindIfNot(static_cast<int*>(managed_ptrs[op]), kTestValueBase, kTestBufferElements);
      }
    }

    for (auto ptr : managed_ptrs) {
      HIP_CHECK(hipFree(ptr));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Prefetch Device->Host->Device and verify data integrity throughout
 *  - Tests round-trip data preservation and accessibility
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_RoundTripDataIntegrity") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);

  std::fill_n(managed_memory.ptr(), kTestBufferElements, kTestValueBase);

  StreamGuard stream_guard(Streams::created);

  std::array<void*, 1> managed_ptrs = {managed_memory.ptr()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};
  std::array<size_t, 1> location_indices = {0};
  constexpr unsigned long long flags = 0;

  std::array<hipMemLocation, 1> device_location;
  device_location[0].type = hipMemLocationTypeDevice;
  device_location[0].id = device;

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     device_location.data(), location_indices.data(),
                                     location_indices.size(), flags, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyDataOnDevice(managed_memory.ptr(), stream_guard.stream());

  int last_prefetch_location = -1;
  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, managed_memory.ptr(),
                                    kTestBufferBytes));
  REQUIRE(last_prefetch_location == device);

  std::array<hipMemLocation, 1> host_location;
  host_location[0].type = hipMemLocationTypeHost;
  host_location[0].id = 0;

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     host_location.data(), location_indices.data(),
                                     location_indices.size(), flags, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  ArrayFindIfNot(managed_memory.ptr(), kTestValueBase, kTestBufferElements);

  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, managed_memory.ptr(),
                                    kTestBufferBytes));
  REQUIRE(last_prefetch_location == hipCpuDeviceId);

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     device_location.data(), location_indices.data(),
                                     location_indices.size(), flags, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyDataOnDevice(managed_memory.ptr(), stream_guard.stream());

  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, managed_memory.ptr(),
                                    kTestBufferBytes));
  REQUIRE(last_prefetch_location == device);
}

/**
 * Test Description
 * ------------------------
 *  - Test NULL dptrs, sizes, prefetchLocs, prefetchLocIdxs arrays and freed memory pointers
 *  - Verify API returns appropriate error for invalid NULL parameters and invalid pointers
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_NullAndInvalidPointers") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  StreamGuard stream_guard(Streams::created);

  std::array<void*, 1> managed_ptrs = {managed_memory.ptr()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};

  std::array<hipMemLocation, 1> locations;
  locations[0].type = hipMemLocationTypeDevice;
  locations[0].id = device;

  std::array<size_t, 1> location_indices = {0};
  constexpr unsigned long long flags = 0;

  SECTION("NULL device pointers array") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(nullptr, buffer_sizes.data(), buffer_sizes.size(),
                                             locations.data(), location_indices.data(),
                                             locations.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("NULL sizes array") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(managed_ptrs.data(), nullptr, managed_ptrs.size(),
                                             locations.data(), location_indices.data(),
                                             locations.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("NULL prefetch locations array") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(),
                                             managed_ptrs.size(), nullptr, location_indices.data(),
                                             locations.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("NULL prefetch location indices array") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(),
                                             managed_ptrs.size(), locations.data(), nullptr,
                                             locations.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Freed memory pointer") {
    int* freed_ptr = nullptr;
    HIP_CHECK(hipMallocManaged(&freed_ptr, kTestBufferBytes));
    HIP_CHECK(hipFree(freed_ptr));

    std::array<void*, 1> freed_ptrs = {freed_ptr};
    std::array<size_t, 1> freed_sizes = {kTestBufferBytes};

    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(
                        freed_ptrs.data(), freed_sizes.data(), freed_ptrs.size(), locations.data(),
                        location_indices.data(), locations.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test various invalid prefetchLocIdxs array constraints
 *  - Verify API validates: first element must be 0, strict ordering, bounds checking
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_IndexArrayConstraints") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  constexpr size_t num_operations = 3;

  std::vector<void*> managed_ptrs(num_operations);
  std::vector<size_t> buffer_sizes(num_operations, kTestBufferBytes);

  for (size_t op = 0; op < num_operations; op++) {
    HIP_CHECK(hipMallocManaged(&managed_ptrs[op], kTestBufferBytes));
  }

  StreamGuard stream_guard(Streams::created);

  std::vector<hipMemLocation> locations(num_operations);
  locations[0].type = hipMemLocationTypeDevice;
  locations[0].id = device;
  locations[1].type = hipMemLocationTypeHost;
  locations[1].id = 0;
  locations[2].type = hipMemLocationTypeHostNumaCurrent;
  locations[2].id = 0;

  constexpr unsigned long long flags = 0;

  SECTION("First index must be zero") {
    std::vector<size_t> invalid_indices = {1, 2, 0};

    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), 1,
                                             locations.data(), invalid_indices.data(),
                                             invalid_indices.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

// CUDA has a defect in the implementation of cudaMemPrefetchBatchAsync that
// allows for non-monotonic indices.
#if HT_AMD
  SECTION("Index array must be monotonically increasing") {
    std::vector<size_t> invalid_indices = {0, 1, 0};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                 locations.data(), invalid_indices.data(), invalid_indices.size(),
                                 flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Index array must be strictly increasing") {
    std::vector<size_t> invalid_indices = {0, 1, 1};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                 locations.data(), invalid_indices.data(), invalid_indices.size(),
                                 flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }
#endif

  SECTION("Last index must be less than count") {
    std::vector<size_t> invalid_indices = {0, 2, 4};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                 locations.data(), invalid_indices.data(), invalid_indices.size(),
                                 flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  for (auto ptr : managed_ptrs) {
    HIP_CHECK(hipFree(ptr));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test invalid parameters: count, sizes, flags, stream
 *  - Verify API validates all parameter constraints
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_ParameterValidation") {
  constexpr int device = 0;
  HIP_CHECK(hipSetDevice(device));

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  StreamGuard stream_guard(Streams::created);

  std::array<void*, 2> managed_ptrs = {managed_memory.ptr(), managed_memory.ptr()};
  std::array<size_t, 2> buffer_sizes = {kTestBufferBytes, kTestBufferBytes};

  std::array<hipMemLocation, 2> locations;
  locations[0].type = hipMemLocationTypeDevice;
  locations[0].id = device;
  locations[1].type = hipMemLocationTypeHost;
  locations[1].id = 0;

  std::array<size_t, 2> location_indices = {0, 1};
  constexpr unsigned long long flags = 0;

  SECTION("Zero operation count") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), 0,
                                             locations.data(), location_indices.data(),
                                             locations.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Zero location count") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(
                        managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                        locations.data(), location_indices.data(), 0, flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("More locations than operations") {
    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                 locations.data(), location_indices.data(), locations.size() + 1,
                                 flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Size larger than allocated memory") {
    auto oversized_sizes = buffer_sizes;
    oversized_sizes[0] = kTestBufferBytes * 10;

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), oversized_sizes.data(), managed_ptrs.size(),
                                 locations.data(), location_indices.data(), locations.size(), flags,
                                 stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Zero-sized range") {
    auto zero_sizes = buffer_sizes;
    zero_sizes[0] = 0;

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), zero_sizes.data(), managed_ptrs.size(),
                                 locations.data(), location_indices.data(), locations.size(), flags,
                                 stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Non-zero flags") {
    constexpr unsigned long long invalid_flags = 1;

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                 locations.data(), location_indices.data(), locations.size(),
                                 invalid_flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("NULL stream") {
    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                 locations.data(), location_indices.data(), locations.size(), flags,
                                 nullptr),
        hipErrorInvalidValue);
  }

  SECTION("Negative location.id") {
    auto invalid_locations = locations;
    invalid_locations[0].id = -5;

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                 invalid_locations.data(), location_indices.data(),
                                 invalid_locations.size(), flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test with non-managed memory (malloc) and managed memory (hipMallocManaged)
 *  - Verify behavior based on device capabilities:
 *    - Non-managed memory requires hipDeviceAttributePageableMemoryAccess
 *    - Managed memory requires hipDeviceAttributeConcurrentManagedAccess
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_DeviceCapabilities") {
  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);

  auto alloc_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipMallocManaged);
  LinearAllocGuard<int> memory(alloc_type, kTestBufferBytes);

  std::array<void*, 1> managed_ptrs = {memory.ptr()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};

  std::array<hipMemLocation, 1> locations;
  locations[0].type = hipMemLocationTypeDevice;
  locations[0].id = device;

  std::array<size_t, 1> location_indices = {0};
  constexpr unsigned long long flags = 0;
  hipError_t result = hipMemPrefetchBatchAsync(
      managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(), locations.data(),
      location_indices.data(), locations.size(), flags, stream_guard.stream());

  auto required_attr = (alloc_type == LinearAllocs::malloc)
                           ? hipDeviceAttributePageableMemoryAccess
                           : hipDeviceAttributeConcurrentManagedAccess;

  if (!DeviceAttributesSupport(device, required_attr)) {
    REQUIRE(result == hipErrorInvalidValue);
  } else {
    REQUIRE(result == hipSuccess);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Prefetch to different devices in same batch
 *  - Skip if single GPU system
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_MultiDevice") {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));

  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Multi-device test requires at least 2 GPUs");
    return;
  }

  std::vector<int> supported_devices;
  for (int dev = 0; dev < device_count; dev++) {
    if (DeviceAttributesSupport(dev, hipDeviceAttributeConcurrentManagedAccess)) {
      supported_devices.push_back(dev);
    }
  }

  if (supported_devices.size() < 2) {
    HipTest::HIP_SKIP_TEST(
        "Multi-device test requires at least 2 GPUs with concurrent managed access");
    return;
  }

  int device = supported_devices[0];
  HIP_CHECK(hipSetDevice(device));

  const size_t num_operations = supported_devices.size();

  std::vector<void*> managed_ptrs(num_operations);
  std::vector<size_t> buffer_sizes(num_operations, kTestBufferBytes);

  for (size_t op = 0; op < num_operations; op++) {
    HIP_CHECK(hipMallocManaged(&managed_ptrs[op], kTestBufferBytes));
    std::fill_n(static_cast<int*>(managed_ptrs[op]), kTestBufferElements, kTestValueBase);
  }

  StreamGuard stream_guard(Streams::created);

  std::vector<hipMemLocation> locations(num_operations);
  std::vector<size_t> location_indices(num_operations);

  for (size_t i = 0; i < num_operations; i++) {
    locations[i].type = hipMemLocationTypeDevice;
    locations[i].id = supported_devices[i];
    location_indices[i] = i;
  }

  constexpr unsigned long long flags = 0;
  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                     locations.data(), location_indices.data(), num_operations,
                                     flags, stream_guard.stream()));

  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  for (size_t op = 0; op < num_operations; op++) {
    VerifyDataOnDevice(static_cast<int*>(managed_ptrs[op]), stream_guard.stream());
  }

  // Verify that prefetch actually occurred to the correct device for each buffer
  for (size_t op = 0; op < num_operations; op++) {
    int last_prefetch_location = -1;
    HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                      hipMemRangeAttributeLastPrefetchLocation, managed_ptrs[op],
                                      kTestBufferBytes));
    REQUIRE(last_prefetch_location == supported_devices[op]);
  }

  for (size_t op = 0; op < num_operations; op++) {
    HIP_CHECK(hipFree(managed_ptrs[op]));
  }
}
