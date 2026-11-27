/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 Functional test for Memset3D and Memset3DAsync
 */


#include <hip_test_common.hh>


/**
 * Basic Functional test of hipMemset3D
 */
HIP_TEST_CASE(Unit_hipMemset3D_BasicFunctional) {

  constexpr int memsetval = 0x22;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  constexpr size_t depth = 10;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW * numH * depth;
  char* A_h;

  hipExtent extent = make_hipExtent(width, numH, depth);
  hipPitchedPtr devPitchedPtr;

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }
  HIP_CHECK(hipMemset3D(devPitchedPtr, memsetval, extent));
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif
  HIP_CHECK(hipMemcpy3D(&myparms));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset3D mismatch at index:" << i << " computed:" << A_h[i]
                                         << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}

/**
 * Basic Functional test of hipMemset3DAsync
 */
HIP_TEST_CASE(Unit_hipMemset3DAsync_BasicFunctional) {

  constexpr int memsetval = 0x22;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  constexpr size_t depth = 10;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW * numH * depth;
  hipExtent extent = make_hipExtent(width, numH, depth);
  hipPitchedPtr devPitchedPtr;
  char* A_h;

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif
  HIP_CHECK(hipMemcpy3D(&myparms));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset3DAsync mismatch at index:" << i << " computed:" << A_h[i]
                                              << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *  - Basic scenario to trigger capturehipMemset3DAsync internal
 *  api for improved code coverage
 * Test source
 * ------------------------
 *  - unit/memory/hipMemset3D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipMemset3DAsync_capturehipMemset3DAsync) {
  char* A_h;
  hipPitchedPtr A_d;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  int row, col, dep;
  row = isQuickLevel() ? GENERATE(3, 100) : GENERATE(3, 4, 100);
  col = isQuickLevel() ? GENERATE(3, 100) : GENERATE(3, 4, 100);
  dep = isQuickLevel() ? GENERATE(3, 100) : GENERATE(3, 4, 100);
  hipStream_t stream;

  A_h = reinterpret_cast<char*>(malloc(sizeof(char) * row * col * dep));
  hipExtent extent = make_hipExtent(col, row, dep);
  HIP_CHECK(hipMalloc3D(&A_d, extent));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemset3DAsync(A_d, 'a', extent, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  hipMemcpy3DParms params{};
  params.srcPos = make_hipPos(0, 0, 0);
  params.dstPos = make_hipPos(0, 0, 0);
  params.dstPtr = make_hipPitchedPtr(A_h, col, col, row);
  params.srcPtr = A_d;
  params.extent = extent;
  params.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&params));
  for (int i = 0; i < (row * col * dep); i++) {
    REQUIRE(A_h[i] == 'a');
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d.ptr));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *    - Test hipMemset3D while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/memory/hipMemset3D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipMemset3D_Capture) {

  constexpr int memsetval = 0x22;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  constexpr size_t depth = 10;
  size_t width = numW * sizeof(char);

  hipExtent extent = make_hipExtent(width, numH, depth);
  hipPitchedPtr devPitchedPtr;

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(hipMemset3D(devPitchedPtr, memsetval, extent), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);

  HIP_CHECK(hipFree(devPitchedPtr.ptr));
}

/*
 * hipMemset3D and hipMemset3DAsync on device 0 targeting a hipMalloc buffer
 * on device 1 must succeed without hipDeviceEnablePeerAccess.  The buffer is
 * represented with pitch == extent.width (no row padding, linear layout) so
 * the runtime treats it as a contiguous fill.
 */
HIP_TEST_CASE(Unit_hipMemset3D_PeerBufferFromOtherDevice) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
  if (!can_access_peer) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  // Ensure peer access 0->1 is off so a prior test cannot make this one pass.
  HIP_CHECK(hipSetDevice(0));
  { hipError_t e = hipDeviceDisablePeerAccess(1);
    if (e != hipSuccess && e != hipErrorPeerAccessNotEnabled) HIP_CHECK(e);
    static_cast<void>(hipGetLastError()); }
  // Restore device 0 on scope exit (also fires if a REQUIRE inside a SECTION throws).
  struct DevGuard { ~DevGuard() { static_cast<void>(hipSetDevice(0)); } } dg;

  // Sizes chosen so that pitch == width (no row padding).
  constexpr size_t kWidth = 64;    // bytes per row
  constexpr size_t kHeight = 4;
  constexpr size_t kDepth = 2;
  constexpr size_t kTotal = kWidth * kHeight * kDepth;

  // Allocate a flat buffer on device 1 and represent it as a pitched pointer
  // with pitch == width so hipMemset3D treats the whole allocation as linear.
  HIP_CHECK(hipSetDevice(1));
  uint8_t* dev1_ptr = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev1_ptr), kTotal));
  hipPitchedPtr dev1_pitched = make_hipPitchedPtr(dev1_ptr, kWidth, kWidth, kHeight);
  hipExtent extent = make_hipExtent(kWidth, kHeight, kDepth);

  HIP_CHECK(hipSetDevice(0));

  SECTION("hipMemset3D") {
    // Synchronous 3D memset from device 0 into device-1 buffer.
    HIP_CHECK(hipMemset3D(dev1_pitched, 0x7F, extent));

    std::vector<uint8_t> host(kTotal, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_ptr, kTotal, hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));
    for (size_t i = 0; i < kTotal; ++i) {
      INFO("byte " << i);
      REQUIRE(host[i] == 0x7Fu);
    }
  }

  SECTION("hipMemset3DAsync") {
    // Asynchronous 3D memset from device 0 into device-1 buffer.
    hipStream_t stream0 = nullptr;
    HIP_CHECK(hipStreamCreate(&stream0));
    HIP_CHECK(hipMemset3DAsync(dev1_pitched, 0x3C, extent, stream0));
    HIP_CHECK(hipStreamSynchronize(stream0));

    std::vector<uint8_t> host(kTotal, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_ptr, kTotal, hipMemcpyDeviceToHost));
    HIP_CHECK(hipSetDevice(0));
    for (size_t i = 0; i < kTotal; ++i) {
      INFO("byte " << i);
      REQUIRE(host[i] == 0x3Cu);
    }
    HIP_CHECK(hipStreamDestroy(stream0));
  }

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(dev1_ptr));
  HIP_CHECK(hipSetDevice(0));
}
