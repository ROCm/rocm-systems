/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Functional Test for API - hipGraphUpload
 - Make graph from hipStreamBeginCapture with a stream.
   Upload the graph into different stream and execute the graph and verify.
 */

static void hipGraphUploadFunctional_with_hipStreamBeginCapture(hipStream_t iStream) {
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 1024;
  size_t Nbytes = N * sizeof(float);

  int *A_d, *C_d;
  int *A_h, *C_h;
  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  HIP_CHECK(hipGraphUpload(graphExec, iStream));
  HIP_CHECK(hipGraphLaunch(graphExec, iStream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamSynchronize(iStream));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("A and C not matching at " << i);
    }
  }
  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
}

/**
 * Functional Test for API - hipGraphUpload
 - Make graph by creating graph and add node and functionality to it.
   Upload the graph into different stream and execute the graph and verify.
 */

static void hipGraphUploadFunctional_with_stream(hipStream_t stream) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kNodeAdd;
  hipKernelNodeParams kNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeAdd, graph, nullptr, 0, &kNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kNodeAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kNodeAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNodeAdd, &memcpy_C, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  HIP_CHECK(hipGraphUpload(graphExec, stream));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphUpload_Functional") {
  SECTION("Pass a stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    hipGraphUploadFunctional_with_hipStreamBeginCapture(stream);
    hipGraphUploadFunctional_with_stream(stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  SECTION("Pass stream as default stream") {
    hipGraphUploadFunctional_with_hipStreamBeginCapture(0);
    hipGraphUploadFunctional_with_stream(0);
  }
  SECTION("Pass stream as hipStreamPerThread") {
    hipGraphUploadFunctional_with_hipStreamBeginCapture(hipStreamPerThread);
    hipGraphUploadFunctional_with_stream(hipStreamPerThread);
  }
}

TEST_CASE("Unit_hipGraphUpload_Functional_multidevice_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    SECTION("Pass a common stream for all device") {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));
      int currDevice = -1;
      HIP_CHECK(hipGetDevice(&currDevice));
      for (int i = 0; i < numDevices; i++) {
        if (i != currDevice) {
          int can_access_peer = 0;
          HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, currDevice, i));
          if (!can_access_peer) {
            INFO("Peer access cannot be enabled between devices " << currDevice << " " << i);
            continue;
          }
        }
        HIP_CHECK(hipSetDevice(i));
        hipGraphUploadFunctional_with_hipStreamBeginCapture(stream);
        hipGraphUploadFunctional_with_stream(stream);
      }

      HIP_CHECK(hipStreamDestroy(stream));
    }
    SECTION("Pass a separate stream for each device") {
      for (int i = 0; i < numDevices; i++) {
        HIP_CHECK(hipSetDevice(i));

        hipStream_t dStream;
        HIP_CHECK(hipStreamCreate(&dStream));
        hipGraphUploadFunctional_with_hipStreamBeginCapture(dStream);
        hipGraphUploadFunctional_with_stream(dStream);
        HIP_CHECK(hipStreamDestroy(dStream));
      }
    }
    SECTION("Pass stream as default stream for each device") {
      for (int i = 0; i < numDevices; i++) {
        HIP_CHECK(hipSetDevice(i));
        hipGraphUploadFunctional_with_hipStreamBeginCapture(0);
        hipGraphUploadFunctional_with_stream(0);
      }
    }
    SECTION("Pass stream as hipStreamPerThread for each device") {
      for (int i = 0; i < numDevices; i++) {
        HIP_CHECK(hipSetDevice(i));
        hipGraphUploadFunctional_with_hipStreamBeginCapture(hipStreamPerThread);
        hipGraphUploadFunctional_with_stream(hipStreamPerThread);
      }
    }
  } else {
    SUCCEED("Skipped the testcase as there is no device to test.");
  }
}

/**
 * Functional Test for API - hipGraphUpload
 - Make graph by using hipStreamBeginCapture with a low priority stream.
   Upload the graph into high priority stream and execute the graph and verify.
 */

TEST_CASE("Unit_hipGraphUpload_Functional_With_Priority_Stream") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream1, stream2;
  int minPriority = 0, maxPriority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream1, hipStreamDefault, minPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream2, hipStreamDefault, maxPriority));

  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HipTest::vectorADD<int><<<1, 1, 0, stream1>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphUpload(graphExec, stream2));

  HIP_CHECK(hipGraphLaunch(graphExec, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}

/**
* Negative Test for API - hipGraphUpload Argument Check
1) Pass graphExec node as nullptr.
2) Pass graphExec node as uninitialize object
3) Pass stream as uninitialize object
4) Graphexec is destroyed before upload
*/

TEST_CASE("Unit_hipGraphUpload_Negative_Parameters") {
  hipGraphExec_t graphExec{};
  hipError_t ret;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Pass graphExec node as nullptr") {
    HIP_CHECK_ERROR(hipGraphUpload(nullptr, stream), hipErrorInvalidValue);
  }
  SECTION("Pass graphExec node as uninitialize object") {
    HIP_CHECK_ERROR(hipGraphUpload(graphExec, stream), hipErrorInvalidValue);
  }
  SECTION("Pass stream as uninitialize object") {
    hipStream_t stream1{};
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    ret = hipGraphUpload(graphExec, stream1);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("graphExec is destroyed") {
    hipGraphExec_t graph_exec;
    hipGraph_t graph;

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphUpload(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK_ERROR(hipGraphUpload(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
