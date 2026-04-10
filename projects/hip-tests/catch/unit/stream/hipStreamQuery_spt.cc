/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <utils.hh>
#include "streamCommon.hh"
/**
 * @addtogroup hipStreamQuery_spt hipStreamQuery_spt
 * @{
 * @ingroup StreamTest
 * `hipStreamQuery_spt(hipStream_t stream)` -
 * Returns status of a stream
 */

/**
 * Test Description
 * ------------------------
 * - Test to query a stream with no work returns hipSuccess or not.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamQuery_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipStreamQuery_spt_WithNoWork) {
  hipStream_t stream{nullptr};

  SECTION("Null Stream") { HIP_CHECK(hipStreamQuery_spt(stream)); }

  SECTION("Created Stream") {
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamQuery_spt(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}
/**
 * Test Description
 * ------------------------
 * - Test to query a stream with finished work returns hipSuccess or not.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamQuery_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipStreamQuery_spt_WithFinishedWork) {
  hipStream_t stream{nullptr};

  SECTION("Null Stream") {
    hip::stream::empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipStreamQuery_spt(stream));
  }

  SECTION("Created Stream") {
    HIP_CHECK(hipStreamCreate(&stream));
    hip::stream::empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipStreamQuery_spt(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}
/**
 * Test Description
 * ------------------------
 * - Negative cases to query a stream.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamQuery_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
#if HT_AMD
HIP_TEST_CASE(Unit_hipStreamQuery_spt_NegativeCases) {
  SECTION("Submit Work On Stream And Query Null Stream") {
    hipStream_t ValidStream;
    HIP_CHECK(hipStreamCreate(&ValidStream));

    HIP_CHECK(hipStreamQuery_spt(hip::nullStream));
    LaunchDelayKernel(std::chrono::milliseconds(500), ValidStream);
    HIP_CHECK_ERROR(hipStreamQuery_spt(hip::nullStream), hipSuccess);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipStreamDestroy(ValidStream));
  }
  SECTION("Submit work to the nullStream") {
    HIP_CHECK(hipStreamQuery_spt(hip::nullStream));
    LaunchDelayKernel(std::chrono::milliseconds(500), hip::nullStream);
    HIP_CHECK_ERROR(hipStreamQuery_spt(hip::nullStream), hipSuccess);
    HIP_CHECK(hipStreamSynchronize(hip::nullStream));
  }
}
/**
 * Test Description
 * ------------------------
 * - Test to querying a stream with pending work returns hipErrorNotReady or not.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamQuery_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipStreamQuery_spt_WithPendingWork) {
  hipStream_t waitingStream{nullptr};
  HIP_CHECK(hipStreamCreate(&waitingStream));
  LaunchDelayKernel(std::chrono::milliseconds(500), waitingStream);
  HIP_CHECK_ERROR(hipStreamQuery_spt(waitingStream), hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(waitingStream));
  HIP_CHECK(hipStreamQuery_spt(waitingStream));
  HIP_CHECK(hipStreamDestroy(waitingStream));
}
#endif

/**
 * End doxygen group StreamTest.
 * @}
 */
