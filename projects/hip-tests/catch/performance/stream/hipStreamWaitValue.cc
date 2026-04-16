/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <performance_common.hh>

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

static int IsStreamWaitValueSupported(int device_id) {
  int wait_value_supported = 0;
#if HT_AMD
  HIP_CHECK(hipDeviceGetAttribute(&wait_value_supported, hipDeviceAttributeCanUseStreamWaitValue,
                                  device_id));
#else
  cuDeviceGetAttribute(&wait_value_supported, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                       device_id);
#endif
  return wait_value_supported;
}

class StreamWaitValue32Benchmark : public Benchmark<StreamWaitValue32Benchmark> {
 public:
  void operator()(const size_t array_size, unsigned int flag) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    uint32_t* value_ptr;
    uint32_t value{0};
    if (flag == hipStreamWaitValueAnd) {
      value = 1;
    }
    HIP_CHECK(hipMalloc(&value_ptr, sizeof(uint32_t) * array_size));
    HIP_CHECK(hipMemset(value_ptr, value, sizeof(uint32_t) * array_size));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWaitValue32(stream, value_ptr, value, flag));
    }
    HIP_CHECK(hipFree(value_ptr));
  }
};

class StreamWaitValue64Benchmark : public Benchmark<StreamWaitValue64Benchmark> {
 public:
  void operator()(const size_t array_size, unsigned int flag) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    uint64_t* value_ptr;
    uint64_t value{0};
    if (flag == hipStreamWaitValueAnd) {
      value = 1;
    }
    HIP_CHECK(hipMalloc(&value_ptr, sizeof(uint64_t) * array_size));
    HIP_CHECK(hipMemset(value_ptr, value, sizeof(uint64_t) * array_size));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWaitValue64(stream, value_ptr, value, flag));
    }
    HIP_CHECK(hipFree(value_ptr));
  }
};

template <typename WaitValueBenchmark>
static void RunBenchmark(const size_t array_size, unsigned int flag) {
  WaitValueBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  switch (flag) {
    case hipStreamWaitValueGte:
      benchmark.AddSectionName("greater than or equal");
      break;
    case hipStreamWaitValueEq:
      benchmark.AddSectionName("equal");
      break;
    case hipStreamWaitValueAnd:
      benchmark.AddSectionName("logical and");
      break;
    case hipStreamWaitValueNor:
      benchmark.AddSectionName("logical nor");
      break;
    default:
      benchmark.AddSectionName("unknown flag");
  }
  benchmark.Run(array_size, flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamWaitValue32` for different array sizes:
 *    -# 4 KB
 *    -# 4 MB
 *    -# 16 MB
 *  - Uses different flag types for wait criteria:
 *    -# Greater than or equal
 *    -# Equal
 *    -# Logical AND
 *    -# Logical OR
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamWaitValue.cc
 * Test requirements
 * ------------------------
 *  - Device supports Stream Wait Value operations
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipStreamWaitValue32) {
#if HT_AMD
  if (!IsStreamWaitValueSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipStreamWaitValue32() function. "
        "Hence skipping the testing with Pass result.\n");
    return;
  }

  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  unsigned int flag = GENERATE(hipStreamWaitValueGte, hipStreamWaitValueEq, hipStreamWaitValueAnd,
                               hipStreamWaitValueNor);
  RunBenchmark<StreamWaitValue32Benchmark>(array_size, flag);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamWaitValue64`:
 *    -# Allocation size:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 *    -# Wait type:
 *      - Greater than or equal
 *      - Equal
 *      - Logical AND
 *      - Logical OR
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamWaitValue.cc
 * Test requirements
 * ------------------------
 *  - Device supports Stream Wait Value operations
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipStreamWaitValue64) {
  if (!IsStreamWaitValueSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipStreamWaitValue64() function. "
        "Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  unsigned int flag = GENERATE(hipStreamWaitValueGte, hipStreamWaitValueEq, hipStreamWaitValueAnd,
                               hipStreamWaitValueNor);
  RunBenchmark<StreamWaitValue64Benchmark>(array_size, flag);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
