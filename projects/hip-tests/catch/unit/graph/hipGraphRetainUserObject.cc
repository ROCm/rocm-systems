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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "user_object_common.hh"

/**
 * Functional Test for API - hipGraphRetainUserObject
 */

/* 1) Create GraphUserObject and retain it by calling hipGraphRetainUserObject
      and release it by calling hipGraphReleaseUserObject. */
static void hipGraphRetainUserObject_Functional_1(void* object, void destroyObj(void*)) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 1, hipGraphUserObjectMove));

  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphRetainUserObject_Functional_1") {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyStructObj);
  }
}

/* 2) Create UserObject and GraphUserObject and retain using custom reference
      count and release it by calling hipGraphReleaseUserObject with count. */
TEST_CASE("Unit_hipGraphRetainUserObject_Functional_2") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode;
  hipKernelNodeParams kNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  HIP_CHECK(
      hipGraphAddKernelNode(&kNode, graph, dependencies.data(), dependencies.size(), &kNodeParams));

  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(), dependencies.size(),
                                    C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  int refCount = 2;
  int refCountRetain = 3;

  float* object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyFloatObj, refCount,
                                hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject, refCountRetain));
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, refCountRetain, hipGraphUserObjectMove));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  HIP_CHECK(hipUserObjectRelease(hObject, refCount + refCountRetain));
  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject, refCountRetain));

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Negative Test for API - hipGraphRetainUserObject
 1) Pass graph as nullptr
 2) Pass User Object as nullptr
 3) Pass initialRefcount as 0
 4) Pass initialRefcount as INT_MAX
 5) Pass flag as 0
 6) Pass flag as INT_MAX
 */
TEST_CASE("Unit_hipGraphRetainUserObject_Negative") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float* object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyFloatObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("Pass graph as nullptr") {
    HIP_CHECK_ERROR(hipGraphRetainUserObject(nullptr, hObject, 1, hipGraphUserObjectMove),
                    hipErrorInvalidValue);
  }
  SECTION("Pass User Object as nullptr") {
    HIP_CHECK_ERROR(hipGraphRetainUserObject(graph, nullptr, 1, hipGraphUserObjectMove),
                    hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as 0") {
    HIP_CHECK_ERROR(hipGraphRetainUserObject(graph, hObject, 0, hipGraphUserObjectMove),
                    hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    HIP_CHECK(hipGraphRetainUserObject(graph, hObject, INT_MAX, hipGraphUserObjectMove));
  }
  SECTION("Pass flag as 0") { HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 1, 0)); }
  SECTION("Pass flag as INT_MAX") {
    HIP_CHECK_ERROR(hipGraphRetainUserObject(graph, hObject, 1, INT_MAX), hipErrorInvalidValue);
  }

  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphRetainUserObject_Negative_Basic") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float* object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyFloatObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  // Retain graph object with reference count 2
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 2, hipGraphUserObjectMove));

  // Release graph object with reference count more than 2
  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject, 4));

  // Again Retain graph object with reference count 8
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 8, hipGraphUserObjectMove));

  // Release graph object with reference count 1
  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject, 1));

  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphRetainUserObject_Negative_Null_Object") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float* object = nullptr;  // this is used for Null_Object test
  hipUserObject_t hObject;

  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyFloatObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  // Retain graph object with reference count 2
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 2, hipGraphUserObjectMove));

  // Release graph object with reference count more than 2
  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject, 4));

  // Again Retain graph object with reference count 8
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 8, 0));

  // Release graph object with reference count 1
  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject, 1));

  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}