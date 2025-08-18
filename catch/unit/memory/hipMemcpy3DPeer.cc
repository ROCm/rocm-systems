/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <hip_test_common.hh>
TEST_CASE("Unit_hipMemcpy3DPeer_BasicFunctional") {
  int numW = 16;
  int numH = 16;
  int depth = 4;
  // size_t width = numW * sizeof(char);
  size_t volume = numW * numH * depth * sizeof(char);
  hipExtent extent = make_hipExtent(numW, numH, depth);
  const auto device_count = HipTest::getDeviceCount();
  const auto src_device = GENERATE_COPY(range(0, device_count));
  const auto dst_device = GENERATE_COPY(range(0, device_count));
  // std::cout<<"Src device: " << src_device << ", Dst device: " << dst_device<<std::endl;
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);
  HIP_CHECK(hipSetDevice(src_device));
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (!can_access_peer) {
    std::string msg = "Skipped as peer access cannot be enabled between devices " +
        std::to_string(src_device) + " " + std::to_string(dst_device);
    HipTest::HIP_SKIP_TEST(msg.c_str());
  }
  // Array-1 Memory allocation
  hipChannelFormatDesc channelDesc_1 = hipCreateChannelDesc<char>();
  hipArray_t array_1;
  HIP_CHECK(hipMalloc3DArray(&array_1, &channelDesc_1, extent, 0));

  // Set the array memory
  char* tmpHost = (char*)malloc(volume);
  memset(tmpHost, 'b', volume);
  hipMemcpy3DParms fillParms{};
  fillParms.srcPos = make_hipPos(0, 0, 0);
  fillParms.dstPos = make_hipPos(0, 0, 0);
  fillParms.srcPtr = make_hipPitchedPtr(tmpHost, numW, numW, numH);
  fillParms.dstArray = array_1;
  fillParms.extent = extent;
  fillParms.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fillParms));
  free(tmpHost);

  // Array-2 Memory allocation
  HIP_CHECK(hipSetDevice(dst_device));

  hipChannelFormatDesc channelDesc_2 = hipCreateChannelDesc<char>();
  hipArray_t array_2;
  HIP_CHECK(hipMalloc3DArray(&array_2, &channelDesc_2, extent, 0));

  HIP_CHECK(hipSetDevice(src_device));
  // Copy data to peer-peer device
  hipMemcpy3DPeerParms peerCopyParams{};
  peerCopyParams.dstArray = array_2;
  peerCopyParams.dstDevice = dst_device;
  peerCopyParams.dstPos = make_hipPos(0, 0, 0);
  peerCopyParams.dstPtr = make_hipPitchedPtr(0, 0, 0, 0);
  peerCopyParams.srcArray = array_1;
  peerCopyParams.srcDevice = src_device;
  peerCopyParams.srcPos = make_hipPos(0, 0, 0);
  peerCopyParams.srcPtr = make_hipPitchedPtr(0, 0, 0, 0);
  peerCopyParams.extent = extent;

  HIP_CHECK(hipMemcpy3DPeer(&peerCopyParams));
  // Copy data from Device Array- host ptr
  char* hostBuf = (char*)malloc(volume);
  hipMemcpy3DParms copyParms{};
  copyParms.srcArray = array_2;
  copyParms.dstPtr = make_hipPitchedPtr(hostBuf, numW, numW, numH);
  copyParms.extent = extent;
  copyParms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&copyParms));

  // Validation
  for (size_t i = 0; i < volume; ++i) {
    if (hostBuf[i] != 'b') {
      printf("Array FAILURE : i %zu, val : %x\n", i, hostBuf[i]);
    }
  }
  free(hostBuf);
  HIP_CHECK(hipFreeArray(array_1));
  HIP_CHECK(hipFreeArray(array_2));
}
TEST_CASE("Unit_hipMemcpy3DPeer_NegativeTsts") {
  SECTION("Memcpy3DPeer Params struct as nullptr") {
    HIP_CHECK_ERROR(hipMemcpy3D(nullptr), hipErrorInvalidValue);
  }
  int numW = 16;
  int numH = 16;
  int depth = 4;
  // size_t width = numW * sizeof(char);
  // size_t volume = numW * numH * depth * sizeof(char);
  hipExtent extent = make_hipExtent(numW, numH, depth);
  const auto device_count = HipTest::getDeviceCount();
  const auto src_device = GENERATE_COPY(range(0, device_count));
  const auto dst_device = GENERATE_COPY(range(0, device_count));
  HIP_CHECK(hipSetDevice(src_device));

  // Array-2 Memory allocation
  hipArray_t array_1;
  hipChannelFormatDesc channelDesc_1 = hipCreateChannelDesc<char>();
  HIP_CHECK(hipMalloc3DArray(&array_1, &channelDesc_1, extent, 0));

  // Array-2 Memory allocation
  HIP_CHECK(hipSetDevice(dst_device));
  hipArray_t array_2;
  hipChannelFormatDesc channelDesc_2 = hipCreateChannelDesc<char>();
  HIP_CHECK(hipMalloc3DArray(&array_2, &channelDesc_2, extent, 0));
  SECTION("Invalid Memcpy3DPeer Params") {
    hipMemcpy3DPeerParms peerCopyParams{};
    peerCopyParams.dstArray = array_2;
    peerCopyParams.dstPos = make_hipPos(0, 0, 0);
    peerCopyParams.dstPtr = make_hipPitchedPtr(0, 0, 0, 0);
    peerCopyParams.srcArray = array_1;
    peerCopyParams.srcPos = make_hipPos(0, 0, 0);
    peerCopyParams.srcPtr = make_hipPitchedPtr(0, 0, 0, 0);
    peerCopyParams.extent = extent;
    SECTION("Max Destination device id") {
      peerCopyParams.dstDevice = device_count + 1;
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidDevice);
    }
    SECTION("Max Source device id") {
      peerCopyParams.srcDevice = device_count + 1;
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidDevice);
    }
    SECTION("Neg Source device id") {
      peerCopyParams.srcDevice = -1;
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidDevice);
    }
    SECTION("Neg Destination device id") {
      peerCopyParams.dstDevice = -1;
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidDevice);
    }
    SECTION("Source Array as Null") {
      peerCopyParams.srcArray = nullptr;
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidValue);
    }
    SECTION("Destination Array as Null") {
      peerCopyParams.dstArray = nullptr;
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidValue);
    }
    /*SECTION("Both Source and Desination Array same") {
      peerCopyParams.dstArray = array_1;
      peerCopyParams.srcArray = array_1;
       HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorLaunchFailure);
    }*/
    SECTION("Passing Max value to extent") {
      peerCopyParams.extent =
          make_hipExtent(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidValue);
    }
    SECTION("Passing width > max width size in extent") {
      peerCopyParams.extent = make_hipExtent(numW + 1, numH, depth);
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidValue);
    }
    SECTION("Passing height > max height size in extent") {
      peerCopyParams.extent = make_hipExtent(numW, numH + 1, depth);
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidValue);
    }
    SECTION("Passing depth > max depth size in extent") {
      peerCopyParams.extent = make_hipExtent(numW, numH, depth + 1);
      HIP_CHECK_ERROR(hipMemcpy3DPeer(&peerCopyParams), hipErrorInvalidValue);
    }
  }
}
