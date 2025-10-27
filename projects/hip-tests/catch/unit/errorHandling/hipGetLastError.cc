/*
Copyright (c) 2022 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>
#include <threaded_zig_zag_test.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_process.hh>
#include <chrono>  //NOLINT
#include <thread>  //NOLINT

static constexpr size_t WIDTH = 1024;
static constexpr size_t HEIGHT = 1024;
static constexpr size_t N = 1024 * 1024;
static constexpr size_t Nbytes = N * sizeof(int);

/**
 * @addtogroup hipGetLastError hipGetLastError
 * @{
 * @ingroup ErrorHandlingTest
 * `hipError_t hipGetLastError ( void )` -
 * Returns the last error from a runtime call.
 */

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMalloc api invalid arg call.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

TEST_CASE("Unit_hipGetLastError_Positive_Basic") {
  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipMalloc(nullptr, 1), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with ThreadedZigZagTest api call.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

TEST_CASE("Unit_hipGetLastError_Positive_Threaded") {
  class HipGetLastErrorThreadedTest : public ThreadedZigZagTest<HipGetLastErrorThreadedTest> {
   public:
    void TestPart2() { REQUIRE_THREAD(hipMalloc(nullptr, 1) == hipErrorInvalidValue); }
    void TestPart3() { HIP_CHECK(hipGetLastError()); }
    void TestPart4() { REQUIRE_THREAD(hipGetLastError() == hipErrorInvalidValue); }
  };

  HipGetLastErrorThreadedTest test;
  test.run();
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpyPeerAsync api invalid arg call
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipMemcpyPeerAsync") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int can_access_peer = 0;
  const auto src_device = 0;
  const auto dst_device = 1;

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    int *A_d, *B_d;
    HIP_CHECK(hipMalloc(&A_d, Nbytes));
    REQUIRE(A_d != nullptr);

    HIP_CHECK(hipSetDevice(dst_device));
    HIP_CHECK(hipMalloc(&B_d, Nbytes));
    REQUIRE(B_d != nullptr);

    HIP_CHECK(hipSetDevice(src_device));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipGetLastError());
    HIP_CHECK_ERROR(hipMemcpyPeerAsync(B_d, dst_device, A_d, src_device, Nbytes * 2, stream),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipSetDevice(dst_device));
    HIP_CHECK(hipFree(B_d));
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " and devices "
                                                          << dst_device);
  }
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpyDtoHAsync api invalid arg call
 *    Verify hipGetLastError status with hipMemcpyDtoDAsync api invalid arg call
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipMemcpyDtoHAsync") {
  int *A_d, *B_d, *A_h;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h, nullptr, nullptr, N, false);

  SECTION("Verify with hipMemcpyDtoHAsync api invalid arg call") {
    HIP_CHECK(hipGetLastError());
    HIP_CHECK_ERROR(hipMemcpyDtoHAsync(A_h, (hipDeviceptr_t)A_d, Nbytes * 2, stream),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK(hipGetLastError());
  }
  SECTION("Verify with hipMemcpyDtoDAsync api invalid arg call") {
    HIP_CHECK(hipGetLastError());
    HIP_CHECK_ERROR(
        hipMemcpyDtoDAsync((hipDeviceptr_t)A_d, (hipDeviceptr_t)B_d, Nbytes * 2, stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK(hipGetLastError());
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, nullptr, nullptr, false);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpyParam2DAsync api invalid arg
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGetLastError_with_hipMemcpyParam2DAsync") {
  CHECK_IMAGE_SUPPORT

  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{WIDTH * sizeof(float)};
  constexpr auto memsetval{100};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocating and Initializing the data
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, HEIGHT));
  HipTest::initArrays<float>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, width * HEIGHT, false);
  HipTest::setDefaultData<float>(WIDTH * HEIGHT, A_h, B_h, C_h);
  HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, WIDTH, HEIGHT));

  // Device to Host
  hip_Memcpy2D desc = {};
  desc.srcMemoryType = hipMemoryTypeDevice;
  desc.dstMemoryType = hipMemoryTypeHost;
  desc.srcHost = A_d;
  desc.srcDevice = hipDeviceptr_t(A_d);
  desc.srcPitch = pitch_A;
  desc.dstHost = A_h;
  desc.dstDevice = hipDeviceptr_t(A_h);
  desc.dstPitch = width;
  desc.WidthInBytes = WIDTH;
  desc.Height = HEIGHT;

  HIP_CHECK(hipGetLastError());
  desc.WidthInBytes = pitch_A + 1;
  HIP_CHECK_ERROR(hipMemcpyParam2DAsync(&desc, stream), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipStreamDestroy(stream));
  HipTest::freeArrays<float>(nullptr, nullptr, nullptr, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipDrvMemcpy3DAsync api invalid arg
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipDrvMemcpy3DAsync") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipDrvMemcpy3DAsync(nullptr, hipStreamPerThread), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpy3DAsync api invalid arg call
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipMemcpy3DAsync") {
  CHECK_IMAGE_SUPPORT

  constexpr int width{10}, height{10}, depth{10};
  auto size = width * height * depth * sizeof(int);
  hipArray_t devArray;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int* hData = reinterpret_cast<int*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);

  // Initialize host buffer
  HipTest::setDefaultData<int>(width * height * depth, hData, nullptr, nullptr);

  hipChannelFormatDesc channelDesc =
      hipCreateChannelDesc(sizeof(int) * 8, 0, 0, 0, hipChannelFormatKindSigned);
  HIP_CHECK(
      hipMalloc3DArray(&devArray, &channelDesc, make_hipExtent(width, height, 2), hipArrayDefault));

  hipMemcpy3DParms myparams;
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, height, depth);
  myparams.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int), width, height);
  myparams.dstArray = devArray;
  myparams.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipMemcpy3DAsync(&myparams, stream), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  // DeAllocating the memory
  HIP_CHECK(hipFreeArray(devArray));
  HIP_CHECK(hipStreamDestroy(stream));
  free(hData);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpy2DToArrayAsync api invalid arg
 *  Verify hipGetLastError status with hipMemcpy2DFromArrayAsync api invalid arg
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipMemcpy2D_To_From_ArrayAsync") {
  int* hData = reinterpret_cast<int*>(malloc(WIDTH));
  REQUIRE(hData != nullptr);
  memset(hData, 0, WIDTH);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Verify with hipMemcpyDtoHAsync api invalid arg call") {
    HIP_CHECK(hipGetLastError());
    HIP_CHECK_ERROR(hipMemcpy2DToArrayAsync(nullptr, 0, 0, hData, WIDTH, WIDTH, HEIGHT,
                                            hipMemcpyHostToDevice, stream),
                    hipErrorInvalidHandle);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidHandle);
    HIP_CHECK(hipGetLastError());
  }
  SECTION("Verify with hipMemcpyDtoHAsync api invalid arg call") {
    HIP_CHECK(hipGetLastError());
    HIP_CHECK_ERROR(hipMemcpy2DFromArrayAsync(hData, WIDTH, nullptr, 0, 0, WIDTH, HEIGHT,
                                              hipMemcpyDeviceToHost, stream),
                    hipErrorInvalidHandle);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidHandle);
    HIP_CHECK(hipGetLastError());
  }

  HIP_CHECK(hipStreamDestroy(stream));
  free(hData);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipStreamAttachMemAsync api invalid arg
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipStreamAttachMemAsync") {
  void* d_memory{nullptr};
  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipMemPrefetchAsync(reinterpret_cast<void*>(d_memory), 0, hipMemAttachHost, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipWaitExternalSemaphoresAsync api
 *invalid arg call Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipWaitExternalSemaphoresAsync") {
  hipExternalSemaphoreWaitParams wait_params = {};
  wait_params.params.fence.value = 1;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(nullptr, &wait_params, 1, nullptr),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipSignalExternalSemaphoresAsync api
 *invalid arg call Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipSignalExternalSemaphoresAsync") {
  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = 1;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipSignalExternalSemaphoresAsync(nullptr, &signal_params, 1, nullptr),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemPrefetchAsync api invalid arg
 *call Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipMemPrefetchAsync") {
  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipMemPrefetchAsync(nullptr, 1024, 0), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpy2DAsync api invalid arg call
 *    Verify hipGetLastError status with hipMemset2DAsync api invalid arg call
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipMemcpy2DAsync") {
  CHECK_IMAGE_SUPPORT

  int *A_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{WIDTH * sizeof(int)};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocating memory
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, WIDTH));
  REQUIRE(A_d != nullptr);

  // Initialize the data
  HipTest::setDefaultData<int>(WIDTH * HEIGHT, A_h, nullptr, nullptr);

  SECTION("Verify with hipMemcpy2DAsync api invalid arg call") {
    HIP_CHECK(hipGetLastError());
    HIP_CHECK_ERROR(hipMemcpy2DAsync(A_h, WIDTH * 2, A_d, pitch_A, WIDTH * sizeof(int), WIDTH,
                                     hipMemcpyDeviceToHost, stream),
                    hipErrorInvalidPitchValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidPitchValue);
    HIP_CHECK(hipGetLastError());
  }
  SECTION("Verify with hipMemset2DAsync api invalid arg call") {
    HIP_CHECK(hipGetLastError());
    HIP_CHECK_ERROR(hipMemset2DAsync(A_d, pitch_A, 22, WIDTH * sizeof(int), WIDTH * 9, stream),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK(hipGetLastError());
  }

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemsetAsync api invalid arg call.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_hipMemsetAsync") {
  int* A_d;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  REQUIRE(A_d != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipMemsetAsync(A_d, 0, Nbytes * 2, stream), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpyAsync api invalid arg call.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_with_MemCpyAsync") {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HipTest::vectorADD<<<1, 1, 0, stream>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // testing to check error manually
  HIP_CHECK_ERROR(hipMemcpyAsync(C_h, C_d, Nbytes + N, hipMemcpyDeviceToHost, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HIP_CHECK(hipStreamDestroy(stream));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipMemcpyAsync api invalid arg call
 *    Check in other thread this error should not report by hipGetLastError()
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

// Inside thread, both hipGetLastError() api call should not return error
static void thread_wait_func(int sleep_time) {
  HIP_CHECK(hipGetLastError());
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time * 1000));
  HIP_CHECK(hipGetLastError());
}

TEST_CASE("Unit_hipGetLastError_with_MemCpyAsync_thread") {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HipTest::vectorADD<<<1, 1, 0, stream>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGetLastError());

  std::thread t(thread_wait_func, 2);

  // testing to check error manually
  HIP_CHECK_ERROR(hipMemcpyAsync(C_h, C_d, Nbytes + N, hipMemcpyDeviceToHost, 0),
                  hipErrorInvalidValue);

  t.join();

  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HIP_CHECK(hipStreamDestroy(stream));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipGraphAddMemcpyNode1D api
 *    Create graph with one node as error consciously so it produces an error,
 *    which will be used to verify the behavior of hipGetLastError api.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */

TEST_CASE("Unit_hipGetLastError_with_hipGraphAddMemcpyNode1D") {
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, memcpy_E, kVecAdd;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  hipKernelNodeParams kNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, graph, nullptr, 0, &kNodeParams));

  //  hipGraphAddMemcpyNode1D is called conciously with double the size
  //  so that it produces an error, which will be used to verify the
  //  behavior of hipGetLastError api.
  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipGraphAddMemcpyNode1D(&memcpy_E, graph, nullptr, 0, C_h, C_d, Nbytes * 2,
                                          hipMemcpyDeviceToHost),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kVecAdd, &memcpy_C, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipStreamEndCapture api invalid arg
 *call Create a graph1 with stream with ketnelNode as vector_ADD and
 *    hipStreamEndCapture on graph1 with hipGraphInstantiate to create graphExec
 *    Again hipStreamEndCapture on graph2 which will return hipErrorIllegalState
 *    now verify the behavior of hipGetLastError api with this call.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */

TEST_CASE("Unit_hipGetLastError_with_hipStreamBegin_EndCapture") {
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  HipTest::vectorADD<int><<<1, 1, 0, stream>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph1));

  //  EndCapture is called conciously so that it produces an error,
  //  which will be used to verify the behavior of hipGetLastError api.
  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph2), hipErrorIllegalState);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorIllegalState);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with hipGraphCreate api invalid arg call.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_error_check_with_hipGraphCreate") {
  hipGraph_t graph;
  hipError_t ret;

  HIP_CHECK(hipGetLastError());
  ret = hipGraphCreate(&graph, 1);
  REQUIRE(ret == hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status where a success call after hip runtime error
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

#if HT_NVIDIA
TEST_CASE("Unit_hipGetLastError_success_before_hipGetLastError") {
  hipGraph_t graph;
  hipStream_t stream;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status should update with new api invalid arg call.
 *    Api hipGraphCreate -> return error hipErrorInvalidValue
 *    Api hipDeviceGetGraphMemAttribute -> return error hipErrorInvalidDevice
 *    Now hipGetLastError() api shoud return hipErrorInvalidDevice
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGetLastError_success_before_hipGetLastError_check_again") {
  int value = 0;
  hipGraph_t graph;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);

  HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidDevice);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with divide_by_zero kernel call
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

static void __global__ devideKernl(int* i, int x, int y) { *i = x / (x - y); }

TEST_CASE("Unit_hipGetLastError_with_Kernel_divide_by_zero") {
  int* i_d;
  int i = 9;
  HIP_CHECK(hipMalloc(&i_d, sizeof(int)));
  REQUIRE(i_d != nullptr);
  HIP_CHECK(hipMemcpy(i_d, &i, sizeof(int), hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGetLastError());
  devideKernl<<<1, 1, 0, stream>>>(i_d, 3, 3);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipFree(i_d));
  HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_NVIDIA
/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with Inalid Configuration in kernel call
 *    Kernel call invalid configuration- blocks=1 & threadsPerBlock=0
 *    Kernel call invalid configuration- blocks=1 & threadsPerBlock=1
 *    Kernel call invalid configuration- blocks=0 & threadsPerBlock=0
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

static void __global__ incrementKernl(int* i) { *i += 1; }

TEST_CASE("Unit_hipGetLastError_with_Kernel_Invalid_Configuration") {
  int* i_d;
  HIP_CHECK(hipMalloc(&i_d, sizeof(int)));
  REQUIRE(i_d != nullptr);

  hipError_t ret;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Kernel call invalid configuration- blocks=1 & threadsPerBlock=0") {
    HIP_CHECK(hipGetLastError());
    incrementKernl<<<1, 0, 0, stream>>>(i_d);
    ret = hipGetLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
  }
  SECTION("Kernel call invalid configuration- blocks=0 & threadsPerBlock=1") {
    HIP_CHECK(hipGetLastError());
    incrementKernl<<<0, 1, 0, stream>>>(i_d);
    ret = hipGetLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
  }
  SECTION("Kernel call invalid configuration- blocks=0 & threadsPerBlock=0") {
    HIP_CHECK(hipGetLastError());
    incrementKernl<<<0, 0, 0, stream>>>(i_d);
    ret = hipGetLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
  }

  SECTION("GGLKernel invalid configuration- blocks=1 & threadsPerBlock=0") {
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(incrementKernl, dim3(1), dim3(0), 0, stream, i_d);
    ret = hipGetLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
  }
  SECTION("GGLKernel invalid configuration- blocks=0 & threadsPerBlock=1") {
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(incrementKernl, dim3(0), dim3(1), 0, stream, i_d);
    ret = hipGetLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
  }
  SECTION("GGLKernel invalid configuration- blocks=0 & threadsPerBlock=0") {
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(incrementKernl, dim3(0), dim3(0), 0, stream, i_d);
    ret = hipGetLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
  }

  HIP_CHECK(hipFree(i_d));
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif
/**
 * Test Description
 * ------------------------
 * - Verify hipGetLastError status with an Error - Success calls
 *   Each time status should return the corresponding Error when it called.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGetLastError_With_Chk_Updated_Status") {
    hipGraph_t graph;
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    int* C_d;
    HIP_CHECK(hipMalloc(&C_d, 1024));
    HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
    HIP_CHECK(hipFree(C_d));
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status along with hipPeekAtLastError.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGetLastError_Chk_Along_hipPeekAtLastError") {
    hipGraph_t graph;
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
    int* C_d;
    HIP_CHECK(hipMalloc(&C_d, 1024));
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status
 *    with different Error and Success combinations.
 *    Each time status should return the corresponding Error when it called.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGetLastError_Error_Combinations") {
  int value = 0;
  hipGraph_t graph;
  SECTION("A case with Error-Error") {
      HIP_CHECK(hipGetLastError());
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
  }
  SECTION("A case with Error-Success-Error-Success") {
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      int* A_d;
      HIP_CHECK(hipMalloc(&A_d, 1024));
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
  }
  SECTION("A case with Success-Error-Error-Success") {
      int *A_d;
      HIP_CHECK(hipMalloc(&A_d, 1024));
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
  }
  SECTION("A Case with Success-Error-Success-Error") {
      int *A_d;
      HIP_CHECK(hipMalloc(&A_d, 1024));
      HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value),
                      hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidDevice);
      HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
  }
}

static void thread_func() {
  HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
  HIP_CHECK_ERROR(hipMalloc(nullptr, 1), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with a runtime api invalid arg call.
 *    Check in other thread this error should not report by hipGetLastError()
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGetLastError_With_Thread") {
  hipGraph_t graph;
    int *A_d;
    HIP_CHECK(hipMalloc(&A_d, 1024));
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    std::thread t(thread_func);
    t.join();
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
}
/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status in the multiple processes.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGetLastError_MultiProcess") {
  hipGraph_t graph;
    int *A_d;
    HIP_CHECK(hipMalloc(&A_d, 1024));
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    hip::SpawnProc proc("hipGetLastErrorEnv_Exe", true);
    HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue);
    REQUIRE(proc.run() == 1);
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
}
static void __global__ emptyKernl() { }
/**
 * Test Description
 * ------------------------
 *  - Verify hipGetLastError status with Invalid Configuration in kernel call.
 *    Kernel call invalid configuration- blocks=0 & threadsPerBlock=0
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipGetLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */

// Below test failed on NVIDIA due to error mismatch produced by the Invalid Kernel config.
// For more details please check the ticket SWDEV-501851 comments.
#if HT_AMD
TEST_CASE("Unit_hipGetLastError_Kernel_Invalid_Config") {
  hipError_t ret;
    hipLaunchKernelGGL(emptyKernl, dim3(0), dim3(0), 0, 0);
    int* A_d;
    HIP_CHECK(hipMalloc(&A_d, 1024));
    ret = hipGetLastError();
    REQUIRE(ret == hipErrorInvalidConfiguration);
    HIP_CHECK_ERROR(hipGetLastError(), hipSuccess);
    HIP_CHECK(hipFree(A_d));
}
#endif
/**
 * End doxygen group ErrorTest.
 * @}
 */
