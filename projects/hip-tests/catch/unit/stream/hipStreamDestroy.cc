/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <hip_test_common.hh>
#include <utils.hh>

namespace hipStreamDestroyTests {

HIP_TEST_CASE(Unit_hipStreamDestroy_Default) {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Unit_hipStreamDestroy_Negative_NullStream) {
  HIP_CHECK_ERROR(hipStreamDestroy(nullptr), hipErrorInvalidResourceHandle);
}

template <size_t numDataPoints> void checkDataSet(int* deviceData) {
  HIP_CHECK(hipStreamSynchronize(nullptr));
  std::array<int, numDataPoints> hostData{};
  HIP_CHECK(
      hipMemcpy(hostData.data(), deviceData, sizeof(int) * numDataPoints, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(std::begin(hostData), std::end(hostData), [](int x) { return x == 1; }));
}

__global__ void setToOne(int* x, size_t size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x[idx] = 1;
  }
}

HIP_TEST_CASE(Unit_hipStreamDestroy_WithFinishedWork) {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr int numDataPoints = 10;
  int* deviceData{};
  HIP_CHECK(hipMalloc(&deviceData, sizeof(int) * numDataPoints));
  HIP_CHECK(hipMemset(deviceData, 0, sizeof(int) * numDataPoints));

  setToOne<<<1, numDataPoints, 0, stream>>>(deviceData, numDataPoints);
  checkDataSet<numDataPoints>(deviceData);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(deviceData));
}

// hipStreamDestroy should return immediately then clean up the resources when the stream is empty
// of work
#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */
HIP_TEST_CASE(Unit_hipStreamDestroy_WithPendingWork) {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr int numDataPoints = 10;
  int* deviceData{};
  HIP_CHECK(hipMalloc(&deviceData, sizeof(int) * numDataPoints));
  HIP_CHECK(hipMemset(deviceData, 0, sizeof(int) * numDataPoints));

  LaunchDelayKernel(std::chrono::milliseconds(500), stream);
  setToOne<<<1, numDataPoints, 0, stream>>>(deviceData, numDataPoints);
  SECTION("Without stream query") { fprintf(stderr, "Without stream query\n"); }
  SECTION("With stream query") {
    fprintf(stderr, "With stream query\n");
    HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  }
  HIP_CHECK(hipStreamDestroy(stream));
  checkDataSet<numDataPoints>(deviceData);
  HIP_CHECK(hipFree(deviceData));
}
#endif
}  // namespace hipStreamDestroyTests
