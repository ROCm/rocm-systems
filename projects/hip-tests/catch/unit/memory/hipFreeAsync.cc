/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipFreeAsync hipFreeAsync
 * @{
 * @ingroup StreamOTest
 * `hipFreeAsync(void* dev_ptr, hipStream_t stream)`
 * - Frees memory with stream ordered semantics
 */


/**
 * Test Description
 * ------------------------
 *  - Test to verify hipFreeAsync behavior with invalid arguments:
 *    -# Nullptr dev_ptr
 *    -# Invalid stream handle
 *    -# Double hipFreeAsync
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipFreeAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipFreeAsync_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int* p = nullptr;
  size_t alloc_size = 1024;
  StreamGuard stream(Streams::created);

  SECTION("dev_ptr is nullptr") {
    HIP_CHECK(hipFreeAsync(nullptr, stream.stream()));
  }

  SECTION("Double free") {
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&p), alloc_size, stream.stream()));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(p), stream.stream()));
    HIP_CHECK(hipStreamSynchronize(stream.stream()));
    HIP_CHECK_ERROR(hipFreeAsync(reinterpret_cast<void*>(p), stream.stream()),
                    hipErrorInvalidValue);
  }
}

/**
 * End doxygen group StreamOTest.
 * @}
 */

/**
 * Test Description
 * ------------------------
 *  - Functional test cases to trigger capturehipFreeAsync internal api
 * Test source
 * ------------------------
 *  - unit/memory/hipFreeAsync
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipFreeAsync_capturehipFreeAsync") {
  HIP_CHECK(hipSetDevice(0));
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  hipStream_t stream;
  hipMemPool_t memPool;
  int rows, cols;
  rows = GENERATE(3, 4, 1024);
  cols = GENERATE(3, 4, 1024);
  HIP_CHECK(hipDeviceGetDefaultMemPool(&memPool, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  int* devMem;

  // Start Capturing
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&devMem), sizeof(int) * rows * cols,
                                   memPool, stream));
  HIP_CHECK(hipFreeAsync(devMem, stream));
  // End Capture
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // Create and Launch Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
