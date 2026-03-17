/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "streamCommon.hh"
#include <utils.hh>
/**
 * @brief Check that querying a stream with no work returns hipSuccess
 *
 **/
TEST_CASE(Unit_hipStreamQuery_WithNoWork) {
  hipStream_t stream{nullptr};

  SECTION("Null Stream") { HIP_CHECK(hipStreamQuery(stream)); }

  SECTION("Created Stream") {
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamQuery(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * @brief Check that querying a stream with finished work returns hipSuccess
 *
 **/
TEST_CASE(Unit_hipStreamQuery_WithFinishedWork) {
  hipStream_t stream{nullptr};

  SECTION("Null Stream") {
    hip::stream::empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipStreamQuery(stream));
  }

  SECTION("Created Stream") {
    HIP_CHECK(hipStreamCreate(&stream));
    hip::stream::empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipStreamQuery(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */

/**
 * @brief Check that submitting work to a stream sets the status of the nullStream to
 * hipErrorNotReady
 *
 */
TEST_CASE(Unit_hipStreamQuery_SubmitWorkOnStreamAndQueryNullStream) {
  {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipStreamQuery(hip::nullStream));
    LaunchDelayKernel(std::chrono::milliseconds(500), stream);
    HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * @brief Check that submitting work to the nullStream properly sets its status as
 * hipErrorNotReady.
 *
 */
TEST_CASE(Unit_hipStreamQuery_NullStreamQuery) {
  HIP_CHECK(hipStreamQuery(hip::nullStream));
  LaunchDelayKernel(std::chrono::milliseconds(500), hip::nullStream);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize(hip::nullStream));
}

/**
 * @brief Check that querying a stream with pending work returns hipErrorNotReady
 *
 **/
TEST_CASE(Unit_hipStreamQuery_WithPendingWork) {
  hipStream_t waitingStream{nullptr};
  HIP_CHECK(hipStreamCreate(&waitingStream));

  LaunchDelayKernel(std::chrono::milliseconds(500), waitingStream);
  HIP_CHECK_ERROR(hipStreamQuery(waitingStream), hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(waitingStream));
  HIP_CHECK(hipStreamQuery(waitingStream));

  HIP_CHECK(hipStreamDestroy(waitingStream));
}
#endif
