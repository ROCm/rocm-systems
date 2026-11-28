/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "MemUtils.hh"
#include <vector>

/*
 * This testcase verifies that asynchronous memset functions are asynchronous with respect to the
 * host except when the target is pinned host memory or a Unified Memory region
 */

constexpr int testValue1 = 97;
constexpr int testValue2 = 98;


using namespace mem_utils;

// Helper function to run tests for hipMemset allocation types
template <typename T> void runAsyncTests(hipStream_t stream, allocType type, memType memType,
                                         MultiDData data1, MultiDData data2) {
  std::pair<T*, T*> aPtr{};
  MultiDData totalRange;
  totalRange.width = data1.width + data2.width;
  totalRange.height = data1.height + data2.height;
  totalRange.depth = data1.depth + data2.depth;
  aPtr = initMemory<T>(type, memType, totalRange);
  data1.pitch = totalRange.pitch;
  data2.pitch = totalRange.pitch;

  memsetCheck(aPtr.first, testValue1, memType, data1, stream);
  memsetCheck(aPtr.first, testValue2, memType, data2, stream);

  HIP_CHECK(hipStreamSynchronize(stream));
  verifyData(aPtr.first, testValue1, data1, type, memType);
  verifyData(aPtr.first, testValue2, data2, type, memType);


  if (type == allocType::devRegistered) {
    freeStuff(aPtr.second, type);
  } else {
    freeStuff(aPtr.first, type);
  }
}

template <typename T> static void doMemsetTest(allocType mallocType, memType memset_type,
                                               MultiDData data1, MultiDData data2) {
  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));

  runAsyncTests<T>(stream, mallocType, memset_type, data1, data2);

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}

/*
 * test 2 async hipMemset's on the same memory at different offsets
 */

HIP_TEST_CASE(Unit_hipMemsetASyncMulti) {
  allocType mallocType = GENERATE(allocType::hostMalloc, allocType::deviceMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memType mem_type = memType::hipMemsetD8;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  MultiDData data2;
  data2.width = data1.width;

  data2.offset = data1.width;
  doMemsetTest<char>(mallocType, mem_type, data1, data2);
}

/*
 * test 2 async hipMemsetD[8,16,32]'s on the same memory at different offsets
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipMemsetDASyncMulti, int8_t, int16_t, uint32_t) {
  allocType mallocType = GENERATE(allocType::hostRegisted, allocType::deviceMalloc,
                                  allocType::hostMalloc, allocType::devRegistered);
  memType memset_type;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  MultiDData data2;
  data2.width = data1.width;
  data2.offset = data1.width;

  if (std::is_same<int8_t, TestType>::value) {
    memset_type = memType::hipMemsetD8;
  } else if (std::is_same<int16_t, TestType>::value) {
    memset_type = memType::hipMemsetD16;
  } else if (std::is_same<uint32_t, TestType>::value) {
    memset_type = memType::hipMemsetD32;
  }
  doMemsetTest<TestType>(mallocType, memset_type, data1, data2);
}

/*
 * test 2 async hipMemset2D's on the same memory at different offsets
 */
HIP_TEST_CASE(Unit_hipMemset2DASyncMulti) {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memType memset_type = memType::hipMem2D;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  data1.height = data1.width;
  MultiDData data2;
  data2.width = data1.width;
  data2.height = data1.height;
  data2.offset = data1.width;

  doMemsetTest<char>(mallocType, memset_type, data1, data2);
}
/*
 * test 2 async hipMemset3D's on the same memory at different offsets
 */
HIP_TEST_CASE(Unit_hipMemset3DASyncMulti) {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memType memset_type = memType::hipMem3D;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  data1.height = data1.width;
  data1.depth = data1.width;
  MultiDData data2;
  data2.width = data1.width;
  data2.height = data1.width;
  data2.depth = data1.width;
  data2.offset = data1.width;

  doMemsetTest<char>(mallocType, memset_type, data1, data2);
}

/*
 * A memset of another device's buffer from this device's stream works without
 * hipDeviceEnablePeerAccess: the fill grants the buffer peer access, as a copy does.
 * hipMemsetAsync, hipMemsetD32Async and hipMemset2DAsync with pitch == width.
 */
HIP_TEST_CASE(Unit_hipMemsetAsync_PeerBufferFromOtherDeviceStream) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
  if (!can_access_peer) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  constexpr size_t kN = 1024;  // elements

  // Allocate the target buffer on device 1.
  HIP_CHECK(hipSetDevice(1));
  uint8_t* dev1_buf = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev1_buf), kN * sizeof(uint8_t)));

  // All operations issued on a stream belonging to device 0.
  HIP_CHECK(hipSetDevice(0));
  hipStream_t stream0 = nullptr;
  HIP_CHECK(hipStreamCreate(&stream0));

  SECTION("hipMemsetAsync") {
    // Fill device-1 buffer from device-0 stream; the runtime must grant peer access.
    HIP_CHECK(hipMemsetAsync(dev1_buf, 0x5A, kN * sizeof(uint8_t), stream0));
    HIP_CHECK(hipStreamSynchronize(stream0));

    // Read back via device 1 to verify every byte.
    std::vector<uint8_t> host(kN, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_buf, kN * sizeof(uint8_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kN; ++i) {
      INFO("byte " << i);
      REQUIRE(host[i] == 0x5Au);
    }
    HIP_CHECK(hipSetDevice(0));
  }

  SECTION("hipMemsetD32Async") {
    constexpr uint32_t kFill32 = 0xDEADBEEFu;
    HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dev1_buf), kFill32,
                                kN / sizeof(uint32_t), stream0));
    HIP_CHECK(hipStreamSynchronize(stream0));

    std::vector<uint32_t> host(kN / sizeof(uint32_t), 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_buf,
                        (kN / sizeof(uint32_t)) * sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
      INFO("element " << i);
      REQUIRE(host[i] == kFill32);
    }
    HIP_CHECK(hipSetDevice(0));
  }

  SECTION("hipMemset2DAsync pitch==width") {
    // 2D memset with pitch == width takes the flat fill path that grants peer access.
    constexpr size_t kWidth = 64;  // bytes
    constexpr size_t kHeight = kN / kWidth;
    HIP_CHECK(hipMemset2DAsync(dev1_buf, kWidth, 0xAB, kWidth, kHeight, stream0));
    HIP_CHECK(hipStreamSynchronize(stream0));

    std::vector<uint8_t> host(kWidth * kHeight, 0);
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(host.data(), dev1_buf, kWidth * kHeight, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
      INFO("byte " << i);
      REQUIRE(host[i] == 0xABu);
    }
    HIP_CHECK(hipSetDevice(0));
  }

  HIP_CHECK(hipStreamDestroy(stream0));
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(dev1_buf));
  HIP_CHECK(hipSetDevice(0));
}
