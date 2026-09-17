/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

/**
 * @addtogroup hipMemsetAsync hipMemsetAsync
 * @{
 * @ingroup MemoryTest
 * A fill dispatched to the device costs the same for a one, two or four byte pattern over
 * the same byte count, because a single kernel writes the whole range either way. A fill
 * carried out on the host costs one memcpy per pattern element, so over the same byte
 * count the one byte pattern costs about four times the four byte pattern. These tests
 * assert the dispatched behaviour for the targets a runtime is most likely to map for
 * direct host access, which are the ones that can reach a host fill without any special
 * memory pressure on the machine running the test.
 */

namespace {

constexpr size_t kSize = 64 * 1024 * 1024;
constexpr unsigned char kPatternByte = 0x5Au;
constexpr unsigned short kPatternHalf = 0x5A5Au;
constexpr int kPatternWord = 0x5A5A5A5A;
constexpr int kIterations = 5;

// One kernel dispatch covers every pattern size, so the ratio is about 1. One memcpy per
// pattern element makes it about 4. The bound sits a factor of two away from both.
constexpr double kMaxRatio = 2.0;

// steady_clock around the call and the stream synchronize, the same bracket CpuTimer uses
// in catch/include/performance_common.hh. Device events cannot be used here: a fill the
// runtime performs on the host is not device work, so event timestamps do not cover it and
// a host fill would be reported as the fast one.
template <typename F> double BestMilliseconds(F fill, const hipStream_t stream) {
  HIP_CHECK(fill());
  HIP_CHECK(hipStreamSynchronize(stream));

  double best = std::numeric_limits<double>::max();
  for (int i = 0; i < kIterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    HIP_CHECK(fill());
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    best = std::min(best, elapsed.count());
  }
  return best;
}

// Reads the target back through the runtime so a memset that never wrote anything cannot
// pass on timing alone.
void RequirePatternWritten(const void* ptr) {
  std::vector<unsigned char> readback(kSize, 0u);
  HIP_CHECK(hipMemcpy(readback.data(), ptr, kSize, hipMemcpyDefault));
  REQUIRE(std::all_of(readback.cbegin(), readback.cend(),
                      [](unsigned char b) { return b == kPatternByte; }));
}

void RequireFillCostIndependentOfPatternSize(const LinearAllocs allocation_type,
                                             const unsigned int flags) {
  LinearAllocGuard<unsigned char> alloc(allocation_type, kSize, flags);
  StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  void* const ptr = static_cast<void*>(alloc.ptr());
  const auto dev_ptr = reinterpret_cast<hipDeviceptr_t>(alloc.ptr());

  const double one_byte =
      BestMilliseconds([&]() { return hipMemsetAsync(ptr, kPatternByte, kSize, stream); }, stream);
  const double two_byte = BestMilliseconds(
      [&]() {
        return hipMemsetD16Async(dev_ptr, kPatternHalf, kSize / sizeof(unsigned short), stream);
      },
      stream);
  const double four_byte = BestMilliseconds(
      [&]() {
        return hipMemsetD32Async(dev_ptr, kPatternWord, kSize / sizeof(unsigned int), stream);
      },
      stream);

  RequirePatternWritten(ptr);

  INFO("one byte " << one_byte << " ms, two byte " << two_byte << " ms, four byte " << four_byte
                   << " ms, ratio " << (one_byte / four_byte));
  REQUIRE(one_byte < kMaxRatio * four_byte);
}

}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 *  - Fills the same buffer with a one, two and four byte pattern over the same byte count
 * and requires that the one byte fill does not cost more than twice the four byte fill.
 * The targets are write combined host memory and registered host memory, because those
 * are mapped for direct host access on an ordinary machine, whereas a device allocation
 * only becomes host mapped once the device local heap is exhausted.
 * Test source
 * ------------------------
 *  - catch/unit/memory/hipMemsetAsyncPatternScaling.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemsetAsync_PatternSizeScaling) {
  SECTION("write combined host memory") {
    RequireFillCostIndependentOfPatternSize(LinearAllocs::hipHostMalloc,
                                            hipHostMallocWriteCombined);
  }

  SECTION("host memory registered with hipHostRegister") {
    RequireFillCostIndependentOfPatternSize(LinearAllocs::mallocAndRegister,
                                            hipHostRegisterMapped);
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
