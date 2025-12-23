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
#include <vector>
/**
 * @addtogroup hipMemcpy3DBatchAsync hipMemcpy3DBatchAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp*
 opList, size_t* failIdx, unsigned long long flags, hipStream_t stream
 __dparm(0))` -
 * Perform Batch of 3D copies.
 */
// Helper to check array content
template <typename T>
void checkArrayContent(hipArray_t array, size_t width, size_t height,
                       size_t depth, T expected) {
  std::vector<T> hostBuf(width * height * depth, 0);
  hipMemcpy3DParms copyParms{};
  copyParms.srcArray = array;
  copyParms.dstPtr =
      make_hipPitchedPtr(hostBuf.data(), width * sizeof(T), width, height);
  copyParms.extent = make_hipExtent(width, height, depth);
  copyParms.kind = hipMemcpyDeviceToHost;
  copyParms.srcPos = make_hipPos(0, 0, 0);
  copyParms.dstPos = make_hipPos(0, 0, 0);
  HIP_CHECK(hipMemcpy3D(&copyParms));
  for (size_t i = 0; i < width * height * depth; ++i) {
    INFO("Array FAILURE at Index: " << i << "\nval : " << hostBuf[i]
                                    << " expected:" << expected);
    REQUIRE(hostBuf[i] == expected);
  }
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the Asynchronus 3D batch memory copy.
 * 1. Test case verifies below batch pointer to pointer mem copy operations.
 * 2. Op1: Host -> Device Copy
 * 3. Op2: Device -> Device Copy
 * 4. Op3: Device -> Host Copy
 * 5. Op4: Host -> Host Copy
 * 6. Prepare hipMemcpy3DBatchOp Array with appropriate data for ptr-ptr copy.
 * 7. Create Stream.
 * 8. Launch the hipMemcpy3DBatchAsync with appropriate fields.
 * 9. Validate the data.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEMPLATE_TEST_CASE("Unit_hipMemcpy3DBatchAsync_Ptr2PtrBatchOps", "", char, int,
                   float) {
  constexpr auto kfloatval1 = -1.5f;
  constexpr auto kfloatval2 = 2.25f;
  constexpr auto kfloatval3 = -0.75f;
  const TestType val1 = std::is_floating_point_v<TestType> ? kfloatval1
                        : std::is_integral_v<TestType>     ? 10
                                                           : 'a';
  const TestType val2 = std::is_floating_point_v<TestType> ? kfloatval2
                        : std::is_integral_v<TestType>     ? 7
                                                           : 'b';
  const TestType val3 = std::is_floating_point_v<TestType> ? kfloatval3
                        : std::is_integral_v<TestType>     ? 3
                                                           : 'c';

  constexpr int numOps = 4;
  constexpr int numW = 16;
  constexpr int numH = 16;
  constexpr int depth = 10;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipExtent extent = make_hipExtent(numW * sizeof(TestType), numH, depth);
  size_t elements_3d = numW * numH * depth;

  // Source Pointers
  std::vector<TestType> srcPtr1(elements_3d, val1);
  std::vector<TestType> srcPtr2(elements_3d, val2);
  std::vector<TestType> srcPtr3(elements_3d, val3);

  // Device Pointers
  void *dstPtr1, *dstPtr2;
  HIP_CHECK(hipMalloc(&dstPtr1, elements_3d * sizeof(TestType)));
  HIP_CHECK(hipMalloc(&dstPtr2, elements_3d * sizeof(TestType)));

  // Prepare batch ops array
  hipMemcpy3DBatchOp ops[numOps];

  // Op 1: Host pointer -> Device pointer
  ops[0].src.type = hipMemcpyOperandTypePointer;
  ops[0].src.op.ptr.ptr = srcPtr1.data();
  ops[0].src.op.ptr.rowLength = extent.width;
  ops[0].src.op.ptr.layerHeight = extent.height;
  ops[0].src.op.ptr.locHint.type = hipMemLocationTypeHost;
  ops[0].src.op.ptr.locHint.id = 0;
  ops[0].dst.type = hipMemcpyOperandTypePointer;
  ops[0].dst.op.ptr.ptr = dstPtr1;
  ops[0].dst.op.ptr.rowLength = extent.width;
  ops[0].dst.op.ptr.layerHeight = extent.height;
  ops[0].dst.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[0].dst.op.ptr.locHint.id = 0;
  ops[0].extent = extent;
  ops[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[0].flags = hipMemcpyFlagDefault;

  // Op 2: device pointer -> device pointer
  ops[1].src.type = hipMemcpyOperandTypePointer;
  ops[1].src.op.ptr.ptr = dstPtr1;
  ops[1].src.op.ptr.rowLength = extent.width;
  ops[1].src.op.ptr.layerHeight = extent.height;
  ops[1].src.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[1].src.op.ptr.locHint.id = 0;
  ops[1].dst.type = hipMemcpyOperandTypePointer;
  ops[1].dst.op.ptr.ptr = dstPtr2;
  ops[1].dst.op.ptr.rowLength = extent.width;
  ops[1].dst.op.ptr.layerHeight = extent.height;
  ops[1].dst.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[1].dst.op.ptr.locHint.id = 0;
  ops[1].extent = extent;
  ops[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[1].flags = hipMemcpyFlagDefault;

  // Op 3: Device pointer -> Host pointer
  ops[2].src.type = hipMemcpyOperandTypePointer;
  ops[2].src.op.ptr.ptr = dstPtr2;
  ops[2].src.op.ptr.rowLength = extent.width;
  ops[2].src.op.ptr.layerHeight = extent.height;
  ops[2].src.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[2].src.op.ptr.locHint.id = 0;
  ops[2].dst.type = hipMemcpyOperandTypePointer;
  ops[2].dst.op.ptr.ptr = srcPtr2.data();
  ops[2].dst.op.ptr.rowLength = extent.width;
  ops[2].dst.op.ptr.layerHeight = extent.height;
  ops[2].dst.op.ptr.locHint.type = hipMemLocationTypeHost;
  ops[2].dst.op.ptr.locHint.id = 0;
  ops[2].extent = extent;
  ops[2].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[2].flags = hipMemcpyFlagDefault;

  // Op 4: Host pointer -> Host pointer
  ops[3].src.type = hipMemcpyOperandTypePointer;
  ops[3].src.op.ptr.ptr = srcPtr2.data();
  ops[3].src.op.ptr.rowLength = extent.width;
  ops[3].src.op.ptr.layerHeight = extent.height;
  ops[3].src.op.ptr.locHint.type = hipMemLocationTypeHost;
  ops[3].src.op.ptr.locHint.id = 0;
  ops[3].dst.type = hipMemcpyOperandTypePointer;
  ops[3].dst.op.ptr.ptr = srcPtr3.data();
  ops[3].dst.op.ptr.rowLength = extent.width;
  ops[3].dst.op.ptr.layerHeight = extent.height;
  ops[3].dst.op.ptr.locHint.type = hipMemLocationTypeHost;
  ops[3].dst.op.ptr.locHint.id = 0;
  ops[3].extent = extent;
  ops[3].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[3].flags = hipMemcpyFlagDefault;

  // Launch the batch
  size_t failIdx;
  unsigned long long flags = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(numOps, ops, &failIdx, flags, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Validation
  for (size_t i = 0; i < elements_3d; ++i) {
    INFO("Array FAILURE at Index: " << i << "\nval : " << srcPtr3[i]);
    REQUIRE(srcPtr3[i] == val1);
  }

  // Cleanup
  HIP_CHECK(hipFree(dstPtr1));
  HIP_CHECK(hipFree(dstPtr2));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the Asynchronus 3D batch memory copy.
 * 1. Test case verifies below batch mem copy operations.
 * 2. Op1: Host -> Array
 * 3. Op2: Array -> Device ptr
 * 4. Op3: Device ptr -> Array
 * 5. Op4: Array -> Array
 * 6. Op5: Array -> Host
 * 7. Prepare hipMemcpy3DBatchOp Array with appropriate data for ptr-ptr copy.
 * 8. Create Stream.
 * 9. Launch the hipMemcpy3DBatchAsync with appropriate fields.
 * 10. Vaidate the data.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpy3DBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEMPLATE_TEST_CASE("Unit_hipMemcpy3DBatchAsync_ArrayMemCpyBatchOps", "", char,
                   int, float) {
  CHECK_IMAGE_SUPPORT
  constexpr auto kfloatval1 = -1.5f;
  constexpr auto kfloatval2 = 2.25f;
  const TestType val1 = std::is_floating_point_v<TestType> ? kfloatval1
                        : std::is_integral_v<TestType>     ? 10
                                                           : 'a';
  const TestType val2 = std::is_floating_point_v<TestType> ? kfloatval2
                        : std::is_integral_v<TestType>     ? 7
                                                           : 'b';
  constexpr int numOps = 5;
  constexpr int numW = 16;
  constexpr int numH = 16;
  constexpr int depth = 10;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipExtent extent = make_hipExtent(numW, numH, depth);
  size_t elements_3d = extent.width * extent.height * extent.depth;

  // Host Pointers
  std::vector<TestType> srcPtr1(elements_3d, val1);
  std::vector<TestType> srcPtr2(elements_3d, val2);

  // Device Pointer
  void *dstPtr;
  HIP_CHECK(hipMalloc(&dstPtr, elements_3d * sizeof(TestType)));

  // Dev Arrays
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<TestType>();
  hipArray_t array1, array2, array3;
  HIP_CHECK(hipMalloc3DArray(&array1, &channelDesc, extent, 0));
  HIP_CHECK(hipMalloc3DArray(&array2, &channelDesc, extent, 0));
  HIP_CHECK(hipMalloc3DArray(&array3, &channelDesc, extent, 0));

  // Fill dev Array with val2
  std::vector<TestType> tmpHost(elements_3d, val2);
  hipMemcpy3DParms fillParms{};
  fillParms.srcPtr =
      make_hipPitchedPtr(tmpHost.data(), extent.width * sizeof(TestType),
                         extent.width, extent.height);
  fillParms.dstArray = array1;
  fillParms.extent = extent;
  fillParms.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fillParms));

  // Prepare batch ops array
  hipMemcpy3DBatchOp ops[numOps];

  // Op 1: host ptr -> device array
  ops[0].src.type = hipMemcpyOperandTypePointer;
  ops[0].src.op.ptr.ptr = srcPtr1.data();
  ops[0].src.op.ptr.rowLength = extent.width;
  ops[0].src.op.ptr.layerHeight = extent.height;
  ops[0].src.op.ptr.locHint.type = hipMemLocationTypeHost;
  ops[0].src.op.ptr.locHint.id = 0;
  ops[0].dst.type = hipMemcpyOperandTypeArray;
  ops[0].dst.op.array.array = array1;
  ops[0].dst.op.array.offset = {0, 0, 0};
  ops[0].extent = extent;
  ops[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[0].flags = hipMemcpyFlagDefault;

  // Op 2: device array -> dev ptr
  ops[1].src.type = hipMemcpyOperandTypeArray;
  ops[1].src.op.array.array = array1;
  ops[1].src.op.array.offset = {0, 0, 0};
  ops[1].dst.type = hipMemcpyOperandTypePointer;
  ops[1].dst.op.ptr.ptr = dstPtr;
  ops[1].dst.op.ptr.rowLength = extent.width;
  ops[1].dst.op.ptr.layerHeight = extent.height;
  ops[1].dst.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[1].dst.op.ptr.locHint.id = 0;
  ops[1].extent = extent;
  ops[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[1].flags = hipMemcpyFlagDefault;

  // Op 3: dev ptr -> device array
  ops[2].src.type = hipMemcpyOperandTypePointer;
  ops[2].src.op.ptr.ptr = dstPtr;
  ops[2].src.op.ptr.rowLength = extent.width;
  ops[2].src.op.ptr.layerHeight = extent.height;
  ops[2].src.op.ptr.locHint.type = hipMemLocationTypeDevice;
  ops[2].src.op.ptr.locHint.id = 0;
  ops[2].dst.type = hipMemcpyOperandTypeArray;
  ops[2].dst.op.array.array = array2;
  ops[2].dst.op.array.offset = {0, 0, 0};
  ops[2].extent = extent;
  ops[2].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[2].flags = hipMemcpyFlagDefault;

  // Op 4: hip array -> hip array
  ops[3].src.type = hipMemcpyOperandTypeArray;
  ops[3].src.op.array.array = array2;
  ops[3].src.op.array.offset = {0, 0, 0};
  ops[3].dst.type = hipMemcpyOperandTypeArray;
  ops[3].dst.op.array.array = array3;
  ops[3].dst.op.array.offset = {0, 0, 0};
  ops[3].extent = extent;
  ops[3].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[3].flags = hipMemcpyFlagDefault;

  // Op 5: device array -> host ptr
  ops[4].src.type = hipMemcpyOperandTypeArray;
  ops[4].src.op.array.array = array3;
  ops[4].src.op.array.offset = {0, 0, 0};
  ops[4].dst.type = hipMemcpyOperandTypePointer;
  ops[4].dst.op.ptr.ptr = srcPtr2.data();
  ops[4].dst.op.ptr.rowLength = extent.width;
  ops[4].dst.op.ptr.layerHeight = extent.height;
  ops[4].dst.op.ptr.locHint.type = hipMemLocationTypeHost;
  ops[4].dst.op.ptr.locHint.id = 0;
  ops[4].extent = extent;
  ops[4].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  ops[4].flags = hipMemcpyFlagDefault;

  // Launch the batch
  size_t failIdx;
  unsigned long long flags = 0;
  HIP_CHECK(hipMemcpy3DBatchAsync(numOps, ops, &failIdx, flags, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Check Random Array data
  checkArrayContent<TestType>(array2, extent.width, extent.height, extent.depth,
                              val1);
  // Check Final data.
  for (size_t i = 0; i < elements_3d; ++i) {
    INFO("Pointer Copy Failure at Index: " << i << "\nval : " << srcPtr2[i]);
    REQUIRE(srcPtr2[i] == val1);
  }

  // Cleanup
  HIP_CHECK(hipFree(dstPtr));
  HIP_CHECK(hipFreeArray(array1));
  HIP_CHECK(hipFreeArray(array2));
  HIP_CHECK(hipFreeArray(array3));
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
    HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(0, ops, &failIdx, flags, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Non Zero flag") {
    HIP_CHECK_ERROR(hipMemcpy3DBatchAsync(numOps, ops, &failIdx, 2, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Ops array as nullptr") {
    HIP_CHECK_ERROR(
        hipMemcpy3DBatchAsync(numOps, nullptr, &failIdx, flags, stream),
        hipErrorInvalidValue);
  }
  // Cleanup
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
