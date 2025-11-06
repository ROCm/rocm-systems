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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

#include <algorithm>
#include <cmath>
#include <array>

namespace {
// Test value constant
constexpr int kTestValueBase = 100;

// Standard buffer size for negative/edge tests
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

/**
 * @brief Kernel to verify data integrity on device
 * @param data Pointer to data array
 * @param num_elements Number of elements to check
 * @param base_value Expected base value
 * @param success_flag Pointer to success flag (set to false if any error found)
 */
__global__ void VerifyDataKernel(const int* data, size_t num_elements, int base_value,
                                  bool* success_flag) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    if (data[idx] != base_value) {
      *success_flag = false;
    }
  }
}

/**
 * @brief Helper function to verify data integrity on device
 * @param data Managed memory pointer to verify
 * @param stream Stream to launch kernel on
 *
 */
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
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - Device supports concurrent managed access
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_SingleOperationSingleLocation") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  std::fill_n(managed_memory.ptr(), kTestBufferElements, kTestValueBase);

  StreamGuard stream_guard(Streams::created);

  std::array<void*, 1> managed_ptrs = {managed_memory.ptr()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};

  std::array<hipMemLocation, 1> prefetch_locations;
  prefetch_locations[0].type = hipMemLocationTypeDevice;
  prefetch_locations[0].id = device;

  std::array<size_t, 1> prefetch_location_indices = {0};
  constexpr unsigned long long flags = 0;

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     prefetch_locations.data(), prefetch_location_indices.data(),
                                     prefetch_locations.size(), flags, stream_guard.stream()));

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
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - Device supports concurrent managed access
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
    std::vector<LinearAllocGuard<int>> managed_buffers;
    managed_buffers.reserve(num_operations);
    for (size_t i = 0; i < num_operations; ++i) {
      managed_buffers.emplace_back(LinearAllocs::hipMallocManaged, kTestBufferBytes);
    }

    std::vector<void*> managed_ptrs(num_operations);
    std::vector<size_t> buffer_sizes(num_operations, kTestBufferBytes);

    for (size_t op = 0; op < num_operations; op++) {
      managed_ptrs[op] = managed_buffers[op].ptr();
      std::fill_n(managed_buffers[op].ptr(), kTestBufferElements, kTestValueBase);
    }

    StreamGuard stream_guard(Streams::created);

    std::vector<hipMemLocation> prefetch_locations;
    std::vector<size_t> prefetch_location_indices;

    switch (pattern) {
      case DistributionPattern::AllSame:
        prefetch_locations.resize(1);
        prefetch_locations[0].type = hipMemLocationTypeDevice;
        prefetch_locations[0].id = device;
        prefetch_location_indices = {0};
        break;

      case DistributionPattern::EachDifferent:
        prefetch_locations.resize(num_operations);
        prefetch_location_indices.resize(num_operations);
        for (size_t i = 0; i < num_operations; i++) {
          if (i % 2 == 0) {
            prefetch_locations[i].type = hipMemLocationTypeDevice;
            prefetch_locations[i].id = device;
          } else {
            prefetch_locations[i].type = hipMemLocationTypeHost;
            prefetch_locations[i].id = 0;
          }
          prefetch_location_indices[i] = i;
        }
        break;

      case DistributionPattern::MixedGrouped:
        prefetch_locations.resize(2);
        prefetch_locations[0].type = hipMemLocationTypeDevice;
        prefetch_locations[0].id = device;
        prefetch_locations[1].type = hipMemLocationTypeHost;
        prefetch_locations[1].id = 0;
        prefetch_location_indices = {0, 3};
        break;
    }

    constexpr unsigned long long flags = 0;
    HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                       prefetch_locations.data(), prefetch_location_indices.data(),
                                       prefetch_locations.size(), flags, stream_guard.stream()));

    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

    for (size_t op = 0; op < num_operations; op++) {
      int last_prefetch_location = -1;
      HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                        hipMemRangeAttributeLastPrefetchLocation,
                                        managed_buffers[op].ptr(), kTestBufferBytes));

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
        VerifyDataOnDevice(managed_buffers[op].ptr(), stream_guard.stream());
      } else {
        ArrayFindIfNot(managed_buffers[op].ptr(), kTestValueBase, kTestBufferElements);
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Prefetch Device->Host->Device and verify data integrity throughout
 *  - Tests round-trip data preservation and accessibility
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - Device supports concurrent managed access
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_RoundTripDataIntegrity") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);

  std::fill_n(managed_memory.ptr(), kTestBufferElements, kTestValueBase);

  StreamGuard stream_guard(Streams::created);

  std::array<void*, 1> managed_ptrs = {managed_memory.ptr()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};
  std::array<size_t, 1> prefetch_location_indices = {0};
  constexpr unsigned long long flags = 0;

  std::array<hipMemLocation, 1> device_location;
  device_location[0].type = hipMemLocationTypeDevice;
  device_location[0].id = device;

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     device_location.data(), prefetch_location_indices.data(),
                                     prefetch_location_indices.size(), flags,
                                     stream_guard.stream()));
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
                                     host_location.data(), prefetch_location_indices.data(),
                                     prefetch_location_indices.size(), flags,
                                     stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  ArrayFindIfNot(managed_memory.ptr(), kTestValueBase, kTestBufferElements);

  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, managed_memory.ptr(),
                                    kTestBufferBytes));
  REQUIRE(last_prefetch_location == hipCpuDeviceId);

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     device_location.data(), prefetch_location_indices.data(),
                                     prefetch_location_indices.size(), flags,
                                     stream_guard.stream()));
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
 *  - Test prefetching to different destination types in one batch
 *  - Includes device, host, and NUMA locations if supported
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - Device supports concurrent managed access
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_MixedDestinationTypes") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  constexpr size_t num_operations = 3;

  std::vector<LinearAllocGuard<int>> managed_buffers;
  managed_buffers.emplace_back(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  managed_buffers.emplace_back(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  managed_buffers.emplace_back(LinearAllocs::hipMallocManaged, kTestBufferBytes);

  std::vector<void*> managed_ptrs(num_operations);
  std::vector<size_t> buffer_sizes(num_operations);

  for (size_t op = 0; op < num_operations; op++) {
    managed_ptrs[op] = managed_buffers[op].ptr();
    buffer_sizes[op] = kTestBufferBytes;
    std::fill_n(managed_buffers[op].ptr(), kTestBufferElements, kTestValueBase);
  }

  StreamGuard stream_guard(Streams::created);

  std::vector<hipMemLocation> prefetch_locations(num_operations);
  prefetch_locations[0].type = hipMemLocationTypeDevice;
  prefetch_locations[0].id = device;
  prefetch_locations[1].type = hipMemLocationTypeHost;
  prefetch_locations[1].id = 0;
  prefetch_locations[2].type = hipMemLocationTypeHostNumaCurrent;
  prefetch_locations[2].id = 0;

  std::vector<size_t> prefetch_location_indices = {0, 1, 2};
  constexpr unsigned long long flags = 0;

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                     prefetch_locations.data(), prefetch_location_indices.data(),
                                     prefetch_locations.size(), flags, stream_guard.stream()));

  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  // Verify data: op 0 is on device, op 1 and op 2 are on host
  VerifyDataOnDevice(managed_buffers[0].ptr(), stream_guard.stream());
  ArrayFindIfNot(managed_buffers[1].ptr(), kTestValueBase, kTestBufferElements);
  ArrayFindIfNot(managed_buffers[2].ptr(), kTestValueBase, kTestBufferElements);

  int last_prefetch_location = -1;
  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation,
                                    managed_buffers[0].ptr(), kTestBufferBytes));
  REQUIRE(last_prefetch_location == device);

  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation,
                                    managed_buffers[1].ptr(), kTestBufferBytes));
  REQUIRE(last_prefetch_location == hipCpuDeviceId);

  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation,
                                    managed_buffers[2].ptr(), kTestBufferBytes));
  REQUIRE(last_prefetch_location == hipCpuDeviceId);
}

/**
 * Test Description
 * ------------------------
 *  - Test NULL dptrs, sizes, prefetchLocs, prefetchLocIdxs arrays and freed memory pointers
 *  - Verify API returns appropriate error for invalid NULL parameters and invalid pointers
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - None (negative test)
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_NullAndInvalidPointers") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  StreamGuard stream_guard(Streams::created);

  std::array<void*, 1> valid_managed_ptrs = {managed_memory.ptr()};
  std::array<size_t, 1> valid_sizes = {kTestBufferBytes};

  std::array<hipMemLocation, 1> valid_locations;
  valid_locations[0].type = hipMemLocationTypeDevice;
  valid_locations[0].id = device;

  std::array<size_t, 1> valid_indices = {0};
  constexpr unsigned long long flags = 0;

  SECTION("NULL device pointers array") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(nullptr, valid_sizes.data(), valid_sizes.size(),
                                             valid_locations.data(), valid_indices.data(),
                                             valid_locations.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("NULL sizes array") {
    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(valid_managed_ptrs.data(), nullptr, valid_managed_ptrs.size(),
                                 valid_locations.data(), valid_indices.data(),
                                 valid_locations.size(), flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("NULL prefetch locations array") {
    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(valid_managed_ptrs.data(), valid_sizes.data(),
                                 valid_managed_ptrs.size(), nullptr, valid_indices.data(),
                                 valid_locations.size(), flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("NULL prefetch location indices array") {
    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(valid_managed_ptrs.data(), valid_sizes.data(),
                                 valid_managed_ptrs.size(), valid_locations.data(), nullptr,
                                 valid_locations.size(), flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Freed memory pointer") {
    int* temp_ptr = nullptr;
    HIP_CHECK(hipMallocManaged(&temp_ptr, kTestBufferBytes));
    void* freed_ptr = temp_ptr;
    HIP_CHECK(hipFree(temp_ptr));

    std::array<void*, 1> freed_ptrs = {freed_ptr};
    std::array<size_t, 1> freed_sizes = {kTestBufferBytes};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(freed_ptrs.data(), freed_sizes.data(), 1, valid_locations.data(),
                                 valid_indices.data(), 1, flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test various invalid prefetchLocIdxs array constraints
 *  - Verify API validates: first element must be 0, monotonic ordering, bounds checking
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - None (negative test)
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_IndexArrayConstraints") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  constexpr size_t num_operations = 3;

  std::vector<LinearAllocGuard<int>> managed_buffers;
  managed_buffers.emplace_back(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  managed_buffers.emplace_back(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  managed_buffers.emplace_back(LinearAllocs::hipMallocManaged, kTestBufferBytes);

  std::vector<void*> managed_ptrs(num_operations);
  std::vector<size_t> buffer_sizes(num_operations);

  for (size_t op = 0; op < num_operations; op++) {
    managed_ptrs[op] = managed_buffers[op].ptr();
    buffer_sizes[op] = kTestBufferBytes;
  }

  StreamGuard stream_guard(Streams::created);

  std::vector<hipMemLocation> prefetch_locations(num_operations);
  prefetch_locations[0].type = hipMemLocationTypeDevice;
  prefetch_locations[0].id = device;
  prefetch_locations[1].type = hipMemLocationTypeHost;
  prefetch_locations[1].id = 0;
  prefetch_locations[2].type = hipMemLocationTypeHostNumaCurrent;
  prefetch_locations[2].id = 0;

  constexpr unsigned long long flags = 0;

  SECTION("First index must be zero") {
    std::vector<size_t> invalid_indices = {1, 2, 0};

    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), 1,
                                             prefetch_locations.data(), invalid_indices.data(),
                                             invalid_indices.size(), flags, stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Index array must be monotonically increasing") {
    std::vector<size_t> invalid_indices = {0, 1, 0};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                 prefetch_locations.data(), invalid_indices.data(),
                                 invalid_indices.size(), flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Last index must be less than count") {
    std::vector<size_t> invalid_indices = {0, 2, 4};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                 prefetch_locations.data(), invalid_indices.data(),
                                 invalid_indices.size(), flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test invalid parameters: count, sizes, flags, stream
 *  - Verify API validates all parameter constraints
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - None (negative test)
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_ParameterValidation") {
  constexpr int device = 0;
  HIP_CHECK(hipSetDevice(device));

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, kTestBufferBytes);
  StreamGuard stream_guard(Streams::created);

  std::array<void*, 2> valid_managed_ptrs = {managed_memory.ptr(), managed_memory.ptr()};
  std::array<size_t, 2> valid_sizes = {kTestBufferBytes, kTestBufferBytes};

  std::array<hipMemLocation, 2> valid_locations;
  valid_locations[0].type = hipMemLocationTypeDevice;
  valid_locations[0].id = device;
  valid_locations[1].type = hipMemLocationTypeHost;
  valid_locations[1].id = 0;

  std::array<size_t, 2> valid_indices = {0, 1};
  constexpr unsigned long long flags = 0;

  SECTION("Zero operation count") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(valid_managed_ptrs.data(), valid_sizes.data(), 0,
                                             valid_locations.data(), valid_indices.data(), 1, flags,
                                             stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Zero location count") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(valid_managed_ptrs.data(), valid_sizes.data(), 2,
                                             valid_locations.data(), valid_indices.data(), 0, flags,
                                             stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("More locations than operations") {
    HIP_CHECK_ERROR(hipMemPrefetchBatchAsync(valid_managed_ptrs.data(), valid_sizes.data(), 2,
                                             valid_locations.data(), valid_indices.data(), 3, flags,
                                             stream_guard.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Size larger than allocated memory") {
    std::array<void*, 1> single_ptr = {managed_memory.ptr()};
    std::array<size_t, 1> oversized = {kTestBufferBytes * 10};
    std::array<hipMemLocation, 1> single_location;
    single_location[0].type = hipMemLocationTypeDevice;
    single_location[0].id = device;
    std::array<size_t, 1> single_index = {0};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(single_ptr.data(), oversized.data(), 1, single_location.data(),
                                 single_index.data(), 1, flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Zero-sized range") {
    std::array<void*, 1> single_ptr = {managed_memory.ptr()};
    std::array<size_t, 1> zero_size = {0};
    std::array<hipMemLocation, 1> single_location;
    single_location[0].type = hipMemLocationTypeDevice;
    single_location[0].id = device;
    std::array<size_t, 1> single_index = {0};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(single_ptr.data(), zero_size.data(), 1, single_location.data(),
                                 single_index.data(), 1, flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("Non-zero flags") {
    std::array<void*, 1> single_ptr = {managed_memory.ptr()};
    std::array<size_t, 1> size_arr = {kTestBufferBytes};
    std::array<hipMemLocation, 1> single_location;
    single_location[0].type = hipMemLocationTypeDevice;
    single_location[0].id = device;
    std::array<size_t, 1> single_index = {0};
    constexpr unsigned long long invalid_flags = 1;

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(single_ptr.data(), size_arr.data(), 1, single_location.data(),
                                 single_index.data(), 1, invalid_flags, stream_guard.stream()),
        hipErrorInvalidValue);
  }

  SECTION("NULL stream") {
    std::array<void*, 1> single_ptr = {managed_memory.ptr()};
    std::array<size_t, 1> size_arr = {kTestBufferBytes};
    std::array<hipMemLocation, 1> single_location;
    single_location[0].type = hipMemLocationTypeDevice;
    single_location[0].id = device;
    std::array<size_t, 1> single_index = {0};

    HIP_CHECK_ERROR(
        hipMemPrefetchBatchAsync(single_ptr.data(), size_arr.data(), 1, single_location.data(),
                                 single_index.data(), 1, flags, nullptr),
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
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - None (negative test)
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_Negative_DeviceCapabilities") {
  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  StreamGuard stream_guard(Streams::created);

  auto alloc_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipMallocManaged);

  LinearAllocGuard<int> memory(alloc_type, kTestBufferBytes);

  std::array<void*, 1> device_ptrs = {memory.ptr()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};
  size_t operation_count = 1;

  std::array<hipMemLocation, 1> prefetch_locations;
  prefetch_locations[0].type = hipMemLocationTypeDevice;
  prefetch_locations[0].id = device;

  std::array<size_t, 1> prefetch_location_indices = {0};
  size_t num_prefetch_locations = 1;
  unsigned long long flags = 0;

  hipError_t result = hipMemPrefetchBatchAsync(
      device_ptrs.data(), buffer_sizes.data(), operation_count, prefetch_locations.data(),
      prefetch_location_indices.data(), num_prefetch_locations, flags, stream_guard.stream());

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
 *  - Test misaligned addresses (not aligned to page boundaries)
 *  - Verify API accepts misaligned addresses and preserves data integrity
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - Device supports concurrent managed access
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_EdgeCase_MisalignedAddresses") {
  REQUIRE_MANAGED_ACCESS_DEVICE(device);

  constexpr size_t buffer_size_bytes = 4096 * sizeof(int);
  constexpr size_t num_elements = buffer_size_bytes / sizeof(int);

  LinearAllocGuard<int> managed_memory(LinearAllocs::hipMallocManaged, buffer_size_bytes);

  std::fill_n(managed_memory.ptr(), num_elements, kTestValueBase);

  StreamGuard stream_guard(Streams::created);

  void* misaligned_ptr = reinterpret_cast<char*>(managed_memory.ptr()) + 128;
  std::array<void*, 1> managed_ptrs = {misaligned_ptr};
  std::array<size_t, 1> buffer_sizes = {1024};

  std::array<hipMemLocation, 1> prefetch_locations;
  prefetch_locations[0].type = hipMemLocationTypeDevice;
  prefetch_locations[0].id = device;

  std::array<size_t, 1> prefetch_location_indices = {0};
  constexpr unsigned long long flags = 0;

  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), managed_ptrs.size(),
                                     prefetch_locations.data(), prefetch_location_indices.data(),
                                     prefetch_locations.size(), flags, stream_guard.stream()));

  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  VerifyDataOnDevice(managed_memory.ptr(), stream_guard.stream());

  // Query with the actual misaligned pointer that was prefetched
  int last_prefetch_location = -1;
  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, misaligned_ptr,
                                    buffer_sizes[0]));
  REQUIRE(last_prefetch_location == device);

  // Query with the original aligned pointer and full buffer size
  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, managed_memory.ptr(),
                                    buffer_size_bytes));

  REQUIRE(last_prefetch_location != device);
}

/**
 * Test Description
 * ------------------------
 *  - System-allocated pageable memory (if supported)
 *  - Skip if hipDeviceAttributePageableMemoryAccess not supported
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - Device supports hipDeviceAttributePageableMemoryAccess
 */
TEST_CASE("Unit_hipMemPrefetchBatchAsync_PageableMemory") {
  constexpr int device = 0;
  HIP_CHECK(hipSetDevice(device));

  if (!DeviceAttributesSupport(device, hipDeviceAttributePageableMemoryAccess)) {
    HipTest::HIP_SKIP_TEST("Device does not support pageable memory access");
    return;
  }

  auto pageable_memory = std::make_unique<int[]>(kTestBufferElements);
  std::fill_n(pageable_memory.get(), kTestBufferElements, kTestValueBase);

  StreamGuard stream_guard(Streams::created);

  std::array<void*, 1> pageable_ptrs = {pageable_memory.get()};
  std::array<size_t, 1> buffer_sizes = {kTestBufferBytes};

  std::array<hipMemLocation, 1> prefetch_locations;
  prefetch_locations[0].type = hipMemLocationTypeDevice;
  prefetch_locations[0].id = device;

  std::array<size_t, 1> prefetch_location_indices = {0};
  constexpr unsigned long long flags = 0;

  HIP_CHECK(hipMemPrefetchBatchAsync(
      pageable_ptrs.data(), buffer_sizes.data(), pageable_ptrs.size(), prefetch_locations.data(),
      prefetch_location_indices.data(), prefetch_locations.size(), flags, stream_guard.stream()));

  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  VerifyDataOnDevice(pageable_memory.get(), stream_guard.stream());

  // Verify that prefetch actually occurred for pageable memory
  int last_prefetch_location = -1;
  HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                    hipMemRangeAttributeLastPrefetchLocation, pageable_memory.get(),
                                    kTestBufferBytes));
  REQUIRE(last_prefetch_location == device);
}

/**
 * Test Description
 * ------------------------
 *  - Prefetch to different devices in same batch
 *  - Skip if single GPU system
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-GPU system with concurrent managed access
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
  std::vector<int*> host_ptrs(num_operations);
  std::vector<size_t> buffer_sizes(num_operations, kTestBufferBytes);

  for (size_t op = 0; op < num_operations; op++) {
    HIP_CHECK(hipMallocManaged(&host_ptrs[op], kTestBufferBytes));
    managed_ptrs[op] = host_ptrs[op];

    std::fill_n(host_ptrs[op], kTestBufferElements, kTestValueBase);
  }

  StreamGuard stream_guard(Streams::created);

  std::vector<hipMemLocation> prefetch_locations(num_operations);
  std::vector<size_t> prefetch_location_indices(num_operations);

  for (size_t i = 0; i < num_operations; i++) {
    prefetch_locations[i].type = hipMemLocationTypeDevice;
    prefetch_locations[i].id = supported_devices[i];
    prefetch_location_indices[i] = i;
  }

  unsigned long long flags = 0;
  HIP_CHECK(hipMemPrefetchBatchAsync(managed_ptrs.data(), buffer_sizes.data(), num_operations,
                                     prefetch_locations.data(), prefetch_location_indices.data(),
                                     num_operations, flags, stream_guard.stream()));

  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  for (size_t op = 0; op < num_operations; op++) {
    VerifyDataOnDevice(host_ptrs[op], stream_guard.stream());
  }

  // Verify that prefetch actually occurred to the correct device for each buffer
  for (size_t op = 0; op < num_operations; op++) {
    int last_prefetch_location = -1;
    HIP_CHECK(hipMemRangeGetAttribute(&last_prefetch_location, sizeof(int),
                                      hipMemRangeAttributeLastPrefetchLocation, host_ptrs[op],
                                      kTestBufferBytes));
    REQUIRE(last_prefetch_location == supported_devices[op]);
  }

  for (size_t op = 0; op < num_operations; op++) {
    HIP_CHECK(hipFree(host_ptrs[op]));
  }
}
