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
#include <hip_test_defgroups.hh>
#include<vector>
/**
 * @addtogroup hipMemcpy3DBatchAsync hipMemcpy3DBatchAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp* opList, size_t*
 failIdx, unsigned long long flags, hipStream_t stream __dparm(0))` -
 * Perform Batch of 3D copies.
 */
// Helper to check array content
void checkArrayContent(hipArray_t array, size_t width, size_t height, size_t depth, char expected) {
  std::vector<char>hostBuf(width * height * depth);
  hipMemcpy3DParms copyParms{};
  copyParms.srcArray = array;
  copyParms.dstPtr = make_hipPitchedPtr(hostBuf.data(), width, width, height);
  copyParms.extent = make_hipExtent(width, height, depth);
  copyParms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&copyParms));
  for (size_t i = 0; i < width * height * depth; ++i) {
    INFO("Array FAILURE at Index: "<< i << "\nval : " <<hostBuf[i]);
    REQUIRE(hostBuf[i] == expected);
  }
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the Asynchronus 3D batch memory copy.
 * 1. Allocate device memory for two pointers (srcptr, dstptr). This is for Pointer to pointer copy.
 * 2. Fill the srcPtr with some data with memset api.
 * 3. Allocate devie memory for two arrays(srcArray, dstArray). This is for Array to Array copy.
 * 4. Fill the srcArray with some data via hipMemcpy3D.
 * 5. Prepare hipMemcpy3DBatchOp Array with appropriate data for both ptr-ptr copy and array-ptr
 *    copy.
 * 6. Create Stream.
 * 7. Launch the hipMemcpy3DBatchAsync with appropriate fields.
 * 8. Copy the data from device to host and verify the data.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemcpy3DBatchAsync_BasicFunctional") {
  CHECK_IMAGE_SUPPORT
  constexpr int numOps = 2;
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));
  hipExtent extent = make_hipExtent(16, 16, 4);
  size_t elements_3d = extent.width * extent.height * extent.depth * sizeof(char);

  // Allocate buffers for pointer-ptr copy
  void *srcPtr0, *dstPtr0;
  HIP_CHECK(hipMalloc(&srcPtr0, elements_3d));
  HIP_CHECK(hipMalloc(&dstPtr0, elements_3d));
  HIP_CHECK(hipMemset(srcPtr0, 'a', elements_3d));  // Fill with value

  // Allocate and fill hip array for array-to-ptr copy
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<char>();
  hipArray_t srcArray1, dstArray1;
  HIP_CHECK(hipMalloc3DArray(&srcArray1, &channelDesc, extent, 0));
  HIP_CHECK(hipMalloc3DArray(&dstArray1, &channelDesc, extent, 0));

  // Fill srcArray1 with 'b'
  std::vector<char>tmpHost(elements_3d);
  memset(tmpHost.data(), 'b', elements_3d);
  hipMemcpy3DParms fillParms{};
  fillParms.srcPtr = make_hipPitchedPtr(tmpHost.data(), extent.width, extent.width, extent.height);
  fillParms.dstArray = srcArray1;
  fillParms.extent = extent;
  fillParms.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fillParms));

  // Prepare batch ops array
  hipMemcpy3DBatchOp ops[numOps];

  // Op 0: device pointer -> device pointer
  ops[0].src.type = hipMemcpyOperandTypePointer;
  ops[0].src.op.ptr.ptr = srcPtr0;
  ops[0].src.op.ptr.rowLength = extent.width;
  ops[0].src.op.ptr.layerHeight = extent.height;
  ops[0].src.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[0].src.op.ptr.locHint.id = 0;
  ops[0].dst.type = hipMemcpyOperandTypePointer;
  ops[0].dst.op.ptr.ptr = dstPtr0;
  ops[0].dst.op.ptr.rowLength = extent.width;
  ops[0].dst.op.ptr.layerHeight = extent.height;
  ops[0].dst.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[0].dst.op.ptr.locHint.id = 0;
  ops[0].extent = extent;
  ops[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[0].flags = hipMemcpyFlagDefault;

  // Op 1: hip array -> hip array
  ops[1].src.type = hipMemcpyOperandTypeArray;
  ops[1].src.op.array.array = srcArray1;
  ops[1].src.op.array.offset = {0, 0, 0};
  ops[1].dst.type = hipMemcpyOperandTypeArray;
  ops[1].dst.op.array.array = dstArray1;
  ops[1].dst.op.array.offset = {0, 0, 0};
  ops[1].extent = extent;
  ops[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[1].flags = hipMemcpyFlagDefault;

  // Launch the batch
  size_t failIdx;
  unsigned long long flags = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(numOps, ops, &failIdx, flags, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  //  Validate pointer-ptr copy (op 0)
  std::vector<char>hostBuf(elements_3d);
  HIP_CHECK(hipMemcpy(hostBuf.data(), dstPtr0, elements_3d, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < elements_3d; ++i) {
    INFO("Array FAILURE at Index: "<< i << "\nval : " <<hostBuf[i]);
    REQUIRE(hostBuf[i] == 'a');
  }
  // Validate array-array copy (op 1)
  checkArrayContent(dstArray1, extent.width, extent.height, extent.depth, 'b');

  // Cleanup
  HIP_CHECK(hipFree(srcPtr0));
  HIP_CHECK(hipFree(dstPtr0));
  HIP_CHECK(hipFreeArray(srcArray1));
  HIP_CHECK(hipFreeArray(dstArray1));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the Asynchronus 3D batch memory copy.
 * 1. Allocate Host memory for two pointers (srcptr, dstptr). This is for Pointer to pointer copy.
 * 2. Fill the srcPtr with some data.
 * 3. Allocate devie memory for 1 array(srcArray). This is for Array to pointer copy.
 * 4. Fill the srcArray with some data via hipMemcpy3D.
 * 5. Prepare hipMemcpy3DBatchOp Array with appropriate data for both ptr-ptr copy and array-ptr
 *    copy.
 * 6. Create Stream.
 * 7. Launch the hipMemcpy3DBatchAsync with appropriate fields.
 * 8. Copy the data from device to host and verify the data.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemcpy3DBatchAsync_PtrH2H_Array2ptr") {
  CHECK_IMAGE_SUPPORT
  constexpr int numOps = 2;
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));
  hipExtent extent = make_hipExtent(16, 16, 4);
  size_t elements_3d = extent.width * extent.height * extent.depth * sizeof(char);

  // Allocate buffers for pointer-ptr copy
  std::vector<char>srcPtr(elements_3d, 'a');
  std::vector<char>dstPtr(elements_3d, 'b');

  // Allocate and fill hip array for array-to-ptr copy
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<char>();
  hipArray_t srcArray;
  HIP_CHECK(hipMalloc3DArray(&srcArray, &channelDesc, extent, 0));
  std::vector<char>dstPtr2(elements_3d, 'a');

  // Fill srcArray with 'b'
  std::vector<char>tmpHost(elements_3d);
  memset(tmpHost.data(), 'b', elements_3d);
  hipMemcpy3DParms fillParms{};
  fillParms.srcPtr = make_hipPitchedPtr(tmpHost.data(), extent.width, extent.width, extent.height);
  fillParms.dstArray = srcArray;
  fillParms.extent = extent;
  fillParms.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fillParms));

  // Prepare batch ops array
  hipMemcpy3DBatchOp ops[numOps];

  // Op 0: Host pointer -> Host pointer
  ops[0].src.type = hipMemcpyOperandTypePointer;
  ops[0].src.op.ptr.ptr = srcPtr.data();
  ops[0].src.op.ptr.rowLength = extent.width;
  ops[0].src.op.ptr.layerHeight = extent.height;
  ops[0].src.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[0].src.op.ptr.locHint.id = 0;
  ops[0].dst.type = hipMemcpyOperandTypePointer;
  ops[0].dst.op.ptr.ptr = dstPtr.data();
  ops[0].dst.op.ptr.rowLength = extent.width;
  ops[0].dst.op.ptr.layerHeight = extent.height;
  ops[0].dst.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[0].dst.op.ptr.locHint.id = 0;
  ops[0].extent = extent;
  ops[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[0].flags = hipMemcpyFlagDefault;

  // Op 1: device array -> host ptr
  ops[1].src.type = hipMemcpyOperandTypeArray;
  ops[1].src.op.array.array = srcArray;
  ops[1].src.op.array.offset = {0, 0, 0};
  ops[1].dst.type = hipMemcpyOperandTypePointer;
  ops[1].dst.op.ptr.ptr = dstPtr2.data();
  ops[1].dst.op.ptr.rowLength = extent.width;
  ops[1].dst.op.ptr.layerHeight = extent.height;
  ops[1].dst.op.ptr.locHint.type = hipMemLocationTypeHost;
  ops[1].dst.op.ptr.locHint.id = 0;
  ops[1].extent = extent;
  ops[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[1].flags = hipMemcpyFlagDefault;

  // Launch the batch
  size_t failIdx;
  unsigned long long flags = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(numOps, ops, &failIdx, flags, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Validate pointer-ptr copy (op 0)
  for (size_t i = 0; i < elements_3d; ++i) {
    INFO("Pointer Copy Failure at Index: "<< i << "\nval : " <<dstPtr[i]);
    REQUIRE(dstPtr[i] == 'a');
  }
  // Validate array-ptr copy (op 1)
  for (size_t i = 0; i < elements_3d; ++i) {
    INFO("Pointer Copy Failure at Index: "<< i << "\nval : " <<dstPtr2[i]);
    REQUIRE(dstPtr2[i] == 'b');
  }
  // Cleanup
  HIP_CHECK(hipFreeArray(srcArray));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Test case to verify the negative cases of hipMemcpy3DBatchAsync.
 * 1. Num of Operations as 0.
 * 2. Non Zero flag.
 * 3. Ops array as nullptr
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemcpy3DBatchAsync_NegativeTests") {
  CHECK_IMAGE_SUPPORT
  const int numOps = 2;
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));
  size_t failIdx;
  unsigned long long flags = 0;
  hipMemcpy3DBatchOp ops[numOps];
  SECTION("Zero Operations") {
    HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(0, ops, &failIdx, flags, stream), hipErrorInvalidValue);
  }
  SECTION("Non Zero flag") {
    HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(numOps, ops, &failIdx, 2, stream), hipErrorInvalidValue);
  }
  SECTION("Ops array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(numOps, nullptr, &failIdx, flags, stream),
                    hipErrorInvalidValue);
  }
  // Cleanup
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
