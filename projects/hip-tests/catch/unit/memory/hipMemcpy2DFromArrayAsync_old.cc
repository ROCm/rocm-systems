/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

static constexpr auto NUM_W{10};
static constexpr auto NUM_H{10};

/*
 * This Scenario Verifies hipMemcpy2DFromArrayAsync API by copying the
 * data from pinned host memory to device from Peer GPU.
 * Device Memory is allocated in GPU 0 and the API is trigerred from GPU1
 * INPUT:  Initialize data, A_h --> A_d device variable
 *         whose memory is allocated in GPU 0
           then A_d-->E_h  in GPU1
 * OUTPUT: validating the result by comparing A_h and E_h
 */
HIP_TEST_CASE(Unit_hipMemcpy2DFromArrayAsync_multiDevicePinnedHostMem) {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  constexpr auto def_val{10};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(0));
      hipArray_t A_d{nullptr};
      size_t width{sizeof(float) * NUM_W};
      float *A_h{nullptr}, *E_h{nullptr};
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      // Initialization of variables
      HipTest::initArrays<float>(nullptr, nullptr, nullptr, &A_h, nullptr, nullptr, width * NUM_H,
                                 false);
      hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
      HIP_CHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
      HipTest::setDefaultData<float>(width * NUM_H, A_h, nullptr, nullptr);
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&E_h), width * NUM_H));
      for (int i = 0; i < NUM_W * NUM_H; i++) {
        E_h[i] = def_val + i;
      }

      HIP_CHECK(hipMemcpy2DToArray(A_d, 0, 0, A_h, width, width, NUM_H, hipMemcpyHostToDevice));
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipMemcpy2DFromArrayAsync(E_h, width, A_d, 0, 0, width, NUM_H,
                                          hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      REQUIRE(HipTest::checkArray(A_h, E_h, NUM_W, NUM_H) == true);

      // Cleaning the memory
      HIP_CHECK(hipFreeArray(A_d));
      HIP_CHECK(hipHostFree(E_h));
      HIP_CHECK(hipStreamDestroy(stream));
      HipTest::freeArrays<float>(nullptr, nullptr, nullptr, A_h, nullptr, nullptr, false);
    } else {
      SUCCEED("Device Does not have P2P capability");
    }
  } else {
    SUCCEED("Number of devices are < 2");
  }
}

/*
 * This scenario verifies the hipMemcpy2DFromArrayAsync API in case of device
 * context change.
 * Memory is allocated in GPU-0 and the API is triggered from GPU-1
 * INPUT:  Copying Host variable hData(Initial value Phi)
 *         --> A_d device variable
 *         whose memory is allocated in GPU 0
 * OUTPUT: For validating the result,Copying A_d device variable
 *         --> A_h host variable
 *         and verifying A_h with Phi
 * */
HIP_TEST_CASE(Unit_hipMemcpy2DFromArrayAsync_multiDeviceContextChange) {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(0));
      hipArray_t A_d{nullptr};
      size_t width{sizeof(float) * NUM_W};
      float *A_h{nullptr}, *hData{nullptr};
      hipStream_t stream;

      // Initialization of variables
      HipTest::initArrays<float>(nullptr, nullptr, nullptr, &A_h, &hData, nullptr, width * NUM_H,
                                 false);
      hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
      HIP_CHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
      HipTest::setDefaultData<float>(width * NUM_H, A_h, hData, nullptr);

      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipStreamCreate(&stream));
      HIP_CHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, width, width, NUM_H, hipMemcpyHostToDevice));

      HIP_CHECK(hipMemcpy2DFromArrayAsync(A_h, width, A_d, 0, 0, width, NUM_H,
                                          hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      REQUIRE(HipTest::checkArray(A_h, hData, NUM_W, NUM_H) == true);

      // Cleaning the memory
      HIP_CHECK(hipFreeArray(A_d));
      HIP_CHECK(hipStreamDestroy(stream));
      HipTest::freeArrays<float>(nullptr, nullptr, nullptr, A_h, hData, nullptr, false);
    } else {
      SUCCEED("Device Does not have P2P capability");
    }
  } else {
    SUCCEED("Number of devices are < 2");
  }
}

/**
* Test Description
* ------------------------
*  - This testcase copies the data from host to device and launches
*  hipMemcpy2DFromArrayAsync within the graph to trigger
*  capturehipMemcpy2DFromArrayAsync internal api and verifies data in host.
* Test source
* ------------------------
*  - unit/memory/hipMemcpy2DFromArrayAsync_old.cc
* Test requirements
* ------------------------
*  - HIP_VERSION >= 6.0
*/
HIP_TEST_CASE(Unit_hipMemcpy2DFromArrayAsync_Capture) {
  CHECK_IMAGE_SUPPORT

  constexpr int kTestSizes[] = {3, 4, 100};
  int num_rows = GENERATE_REF(from_range(std::begin(kTestSizes), std::end(kTestSizes)));
  int num_cols = GENERATE_REF(from_range(std::begin(kTestSizes), std::end(kTestSizes)));

  auto host_src = std::make_unique<int[]>(num_rows * num_cols);
  auto host_dst = std::make_unique<int[]>(num_rows * num_cols);

  for (int row = 0; row < num_rows; ++row) {
    for (int col = 0; col < num_cols; ++col) {
      host_src[row * num_cols + col] = row * num_cols + col;
    }
  }

  hipArray_t device_array = nullptr;
  hipChannelFormatDesc channel_desc = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&device_array, &channel_desc, num_cols, num_rows, hipArrayDefault));
  HIP_CHECK(hipMemcpy2DToArray(device_array, 0, 0, host_src.get(), num_cols * sizeof(int),
                               num_cols * sizeof(int), num_rows, hipMemcpyHostToDevice));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemcpy2DFromArrayAsync(host_dst.get(), sizeof(int) * num_cols, device_array, 0, 0,
                                      sizeof(int) * num_cols, num_rows, hipMemcpyDeviceToHost,
                                      stream));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamSynchronize(stream));

  for (int row = 0; row < num_rows; ++row) {
    for (int col = 0; col < num_cols; ++col) {
      REQUIRE(host_dst[row * num_cols + col] == (row * num_cols + col));
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFreeArray(device_array));
}
