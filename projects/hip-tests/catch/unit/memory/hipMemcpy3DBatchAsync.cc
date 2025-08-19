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

// Helper to check array content
void checkArrayContent(hipArray_t array, size_t width, size_t height, size_t depth, char expected) {
  char* hostBuf = (char*)malloc(width * height * depth);
  hipMemcpy3DParms copyParms{};
  copyParms.srcArray = array;
  copyParms.dstPtr = make_hipPitchedPtr(hostBuf, width, width, height);
  copyParms.extent = make_hipExtent(width, height, depth);
  copyParms.kind = hipMemcpyDeviceToHost;
  // copyParms.srcPos = make_hipPos(0, 0, 0);
  // copyParms.dstPos = make_hipPos(0, 0, 0);
  HIP_CHECK(hipMemcpy3D(&copyParms));
  for (size_t i = 0; i < width * height * depth; ++i) {
    if (hostBuf[i] != expected) {
      printf("Array FAILURE : i %zu, val : %x\n", i, hostBuf[i]);
      REQUIRE(false);
    }
  }
  free(hostBuf);
}

TEST_CASE("Unit_hipMemcpy3DBatchAsync_BasicFunctional") {
  const int numOps = 2;
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // hipExtent extent_Array = make_hipExtent(16, 16, 4);
  hipExtent extent = make_hipExtent(16, 16, 4);
  // hipExtent extent = make_hipExtent(16 * sizeof(char), 16, 4);
  size_t elements_2d = extent.width * extent.height * extent.depth * sizeof(char);
  size_t volume = extent.width * extent.height * extent.depth * sizeof(char);

  // Allocate buffers for pointer-ptr copy
  void *srcPtr0, *dstPtr0;
  HIP_CHECK(hipMalloc(&srcPtr0, elements_2d));
  HIP_CHECK(hipMalloc(&dstPtr0, elements_2d));
  HIP_CHECK(hipMemset(srcPtr0, 'a', elements_2d));  // Fill with pattern

  // Allocate and fill hip array for array-to-ptr copy
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<char>();
  hipArray_t srcArray1, dstArray1;
  HIP_CHECK(hipMalloc3DArray(&srcArray1, &channelDesc, extent, 0));
  HIP_CHECK(hipMalloc3DArray(&dstArray1, &channelDesc, extent, 0));

  // Fill srcArray1 with 0xBB
  char* tmpHost = (char*)malloc(volume);
  memset(tmpHost, 'b', volume);
  hipMemcpy3DParms fillParms{};
  fillParms.srcPtr = make_hipPitchedPtr(tmpHost, extent.width, extent.width, extent.height);
  fillParms.dstArray = srcArray1;
  fillParms.extent = extent;
  // fillParms.srcPos = make_hipPos(0, 0, 0);
  // fillParms.dstPos = make_hipPos(0, 0, 0);
  fillParms.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&fillParms));
  free(tmpHost);

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
  // std::cout<<"Index: "<<failIdx<<std::endl;
  //  Validate pointer-ptr copy (op 0)
  char* hostBuf = (char*)malloc(volume);
  HIP_CHECK(hipMemcpy(hostBuf, dstPtr0, volume, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < volume; ++i) {
    if (hostBuf[i] != 'a') {
      printf("PTR FAILURE : i %zu, val : %x\n", i, hostBuf[i]);
      REQUIRE(false);
    }
  }
  free(hostBuf);

  // Validate array-array copy (op 1)
  checkArrayContent(dstArray1, extent.width, extent.height, extent.depth, 'b');

  // Cleanup
  HIP_CHECK(hipFree(srcPtr0));
  HIP_CHECK(hipFree(dstPtr0));
  HIP_CHECK(hipFreeArray(srcArray1));
  HIP_CHECK(hipFreeArray(dstArray1));
  HIP_CHECK(hipStreamDestroy(stream));
  printf("hipMemcpy3DBatchAsync (with array and pointer) test PASSED.\n");
}

TEST_CASE("Unit_hipMemcpy3DBatchAsync_NegativeTests") {
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
}
