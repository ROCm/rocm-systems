/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
__device__ int devSymbol[10];
__constant__ int constSymbol[10];
TEST_CASE("Unit_hipMemcpyFromSymbolAsync_spt_Negative") {
  SECTION("Invalid Src Ptr") {
    HIP_CHECK_ERROR(hipMemcpyFromSymbolAsync_spt(nullptr, HIP_SYMBOL(devSymbol), sizeof(int), 0,
                                                 hipMemcpyDeviceToHost, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Dst Ptr") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbolAsync_spt(&result, nullptr, sizeof(int), 0,
                                                 hipMemcpyDeviceToHost, nullptr),
                    hipErrorInvalidSymbol);
  }
  SECTION("Invalid Size") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbolAsync_spt(&result, HIP_SYMBOL(devSymbol), sizeof(int) * 100,
                                                 0, hipMemcpyDeviceToHost, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Offset") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbolAsync_spt(&result, HIP_SYMBOL(devSymbol), sizeof(int), 300,
                                                 hipMemcpyDeviceToHost, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Direction") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbolAsync_spt(&result, HIP_SYMBOL(devSymbol), sizeof(int), 0,
                                                 hipMemcpyHostToDevice, nullptr),
                    hipErrorInvalidMemcpyDirection);
  }
}
TEST_CASE("Unit_hipMemcpyFromSymbolAsync_spt_PositiveTests") {
  enum StreamTestType { NullStream = 0, StreamPerThread, CreatedStream, NoStream };
  /* Test type NoStream - Use Sync variants, else use async variants */
  auto streamType = GENERATE(StreamTestType::NoStream, StreamTestType::NullStream,
                             StreamTestType::StreamPerThread, StreamTestType::CreatedStream);
  hipStream_t stream{nullptr};
  if (streamType == StreamTestType::StreamPerThread) {
    stream = hipStreamPerThread;
  } else if (streamType == StreamTestType::CreatedStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }
  INFO("Stream :: " << streamType);
  SECTION("Singular Value") {
    int set{42};
    int result{0};
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), &set, sizeof(int), 0,
                                     hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyFromSymbolAsync_spt(&result, HIP_SYMBOL(devSymbol), sizeof(int), 0,
                                           hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(result == set);
  }
  SECTION("Array Values") {
    constexpr size_t size{10};
    int set[size] = {4, 2, 4, 2, 4, 2, 4, 2, 4, 2};
    int result[size] = {0};

    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), set, sizeof(int) * size, 0,
                                     hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyFromSymbolAsync_spt(&result, HIP_SYMBOL(devSymbol), sizeof(int) * size, 0,
                                           hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    for (size_t i = 0; i < size; i++) {
      REQUIRE(result[i] == set[i]);
    }
  }
  SECTION("Offset'ed Values") {
    constexpr size_t size{10};
    constexpr size_t offset = 5 * sizeof(int);
    int set[size] = {9, 9, 9, 9, 9, 2, 4, 2, 4, 2};
    int result[size] = {0};

    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), set, offset, 0, hipMemcpyHostToDevice,
                                     stream));
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), set + 5, offset, offset,
                                     hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyFromSymbolAsync_spt(result, HIP_SYMBOL(devSymbol), offset, 0,
                                           hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyFromSymbolAsync_spt(result + 5, HIP_SYMBOL(devSymbol), offset, offset,
                                           hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t i = 0; i < size; i++) {
      REQUIRE(result[i] == set[i]);
    }
  }
}
/**
 * Test Description
 * ------------------------
 *  - Basic functional testcase to trigger capturehipMemcpyToSymbolAsync
 *  and capturehipMemcpyFromSymbolAsync internal apis to improve
 *  code coverage.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyFromSymbol.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemcpyFromSymbolAsync_spt_capturehipMemcpyToFromSymbolAsync") {
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int A_h = 0, B_h = 42;
  // Start Capturing
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  SECTION("__constant__ symbol") {
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(constSymbol), &B_h, sizeof(int), 0,
                                     hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyFromSymbolAsync_spt(&A_h, HIP_SYMBOL(constSymbol), sizeof(int), 0,
                                           hipMemcpyDeviceToHost, stream));
  }
  SECTION("__device__ symbol") {
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), &B_h, sizeof(int), 0,
                                     hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyFromSymbolAsync_spt(&A_h, HIP_SYMBOL(devSymbol), sizeof(int), 0,
                                           hipMemcpyDeviceToHost, stream));
  }
  // End Capture
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  // Create and Launch Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(A_h == B_h);
  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
