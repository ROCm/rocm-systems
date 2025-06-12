/*
Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_defgroups.hh>
#include <condition_variable>
#include "user_object_common.hh"
#include <chrono>

bool setVar = false;
void* globalPtr = nullptr;
std::mutex m;
std::condition_variable cv;
/**
 * @addtogroup hipUserObjectCreate hipUserObjectCreate
 * @{
 * @ingroup GraphTest
 * `hipError_t hipUserObjectCreate(hipUserObject_t* object_out,
 *                                 void* ptr, hipHostFn_t destroy,
 *                                 unsigned int initialRefcount,
 *                                 unsigned int flags);`
 * - Create an instance of userObject to manage lifetime of a resource.
 */
template<typename T>
__global__ void KernelFn(T *Ad, int clockrate, int WaitSecs) {
  uint64_t num_cycles = (uint64_t)clockrate;
  num_cycles = num_cycles * 1000 * WaitSecs;
  uint64_t start = clock64(), cycles = 0;
  while (cycles < num_cycles) {
    cycles = clock64() - start;
  }
  if ((std::is_same<float, T>::value) == true) {
    *Ad = 9999.0f;
  } else if ((std::is_same<int, T>::value) == true) {
    *Ad = 9999;
  } else {
    *Ad = 0;
  }
}
void threadFunc_dltMemory() {
  std::unique_lock lk(m);
  cv.wait(lk, []{ return setVar; });
  REQUIRE(globalPtr != nullptr);
  HIP_CHECK(hipHostFree(globalPtr));
  setVar = false;
}
void destroyPinnedObj(void* ptr) {
  globalPtr = ptr;
  setVar = true;
  cv.notify_one();
}
template <typename T>
void hipUserObjectCreate_int_float_Objects(T* hostArr,
                        T* devArr, void destroyObj(void*)) {
  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate,
          hipDeviceAttributeMemoryClockRate, 0));
  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream,
                                  hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(devArr, hostArr, sizeof(int),
                           hipMemcpyHostToDevice, stream));
  KernelFn<<< 1, 1, 0, stream>>>(devArr, clockrate, 5);
  HIP_CHECK(hipMemcpyAsync(hostArr, devArr, sizeof(int),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);
  hipUserObject_t Uobj;
  int refCount = 1;
  HIP_CHECK(hipUserObjectCreate(&Uobj, hostArr, destroyObj, refCount,
            hipUserObjectNoDestructorSync));
  SECTION("Flag as 0") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount, 0));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    SECTION("graph_instance is destroyed before async launch completes") {
      HIP_CHECK(hipGraphLaunch(graph_instance, stream));
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      HIP_CHECK(hipStreamSynchronize(stream));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 9999.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 9999);
      } else {
        REQUIRE(false);
      }
      HIP_CHECK(hipUserObjectRelease(Uobj, 1));
    }
    SECTION("graph_instance is destroyed after async launch completes") {
      HIP_CHECK(hipGraphLaunch(graph_instance, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 9999.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 9999);
      } else {
        REQUIRE(false);
      }
      HIP_CHECK(hipUserObjectRelease(Uobj, 1));
    }
    SECTION("graph_instance is destroyed without a launch") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 1111.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 1111);
      } else {
        REQUIRE(false);
      }
      HIP_CHECK(hipUserObjectRelease(Uobj, 1));
    }
    SECTION("streamSynchronize is not called") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 1111.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 1111);
      } else {
        REQUIRE(false);
      }
      HIP_CHECK(hipUserObjectRelease(Uobj, 1));
    }
  }
  SECTION("Flag as hipGraphUserObjectMove") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount,
                                       hipGraphUserObjectMove));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    SECTION("graph_instance is destroyed before async launch completes") {
      HIP_CHECK(hipGraphLaunch(graph_instance, stream));
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      HIP_CHECK(hipStreamSynchronize(stream));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 9999.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 9999);
      } else {
        REQUIRE(false);
      }
    }
    SECTION("graph_instance is destroyed after async launch completes") {
      HIP_CHECK(hipGraphLaunch(graph_instance, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 9999.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 9999);
      } else {
        REQUIRE(false);
      }
    }
    SECTION("graph_instance is destroyed without a launch") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 1111.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 1111);
      } else {
        REQUIRE(false);
      }
    }
    SECTION("streamSynchronize is not called") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      if ((std::is_same<float, T>::value) == true) {
        REQUIRE(*hostArr == 1111.0);
      } else if ((std::is_same<int, T>::value) == true) {
        REQUIRE(*hostArr == 1111);
      } else {
        REQUIRE(false);
      }
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Verify the release of reference and destructor execution
 *   will be deferred until the graph is synchronized.
 * - References are only released at hipGraphDestroy()
 *   and hipGraphExecDestroy() calls
 * - This test is to verify the above cases with a graph created
 *   from stream capture and with Dynamic memory.
 * Test source
 * ------------------------
 * - catch\unit\graph\hipGraphAsyncUserObj.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipGraphUserObj_Int_float_Objects") {
  SECTION("Called with Int Obj") {
    std::thread t1(threadFunc_dltMemory);
    int *hostArr = nullptr;
    HIP_CHECK(hipHostMalloc(&hostArr, sizeof(int)));
    REQUIRE(hostArr != nullptr);
    *hostArr = 1111;
    int *devArr = nullptr;
    HIP_CHECK(hipMalloc(&devArr, sizeof(int)));
    REQUIRE(devArr != nullptr);
    hipUserObjectCreate_int_float_Objects(hostArr, devArr, destroyPinnedObj);
    HIP_CHECK(hipFree(devArr));
    t1.join();
  }
  SECTION("Called with float Obj") {
    std::thread t1(threadFunc_dltMemory);
    float *hostArr = nullptr;
    HIP_CHECK(hipHostMalloc(&hostArr, sizeof(float)));
    REQUIRE(hostArr != nullptr);
    *hostArr = 1111.0f;
    float *devArr = nullptr;
    HIP_CHECK(hipMalloc(&devArr, sizeof(float)));
    REQUIRE(devArr != nullptr);
    hipUserObjectCreate_int_float_Objects(hostArr, devArr, destroyPinnedObj);
    HIP_CHECK(hipFree(devArr));
    t1.join();
  }
}
void destroyHostRegObj(void* ptr) {
  int* ptr2 = reinterpret_cast<int*>(ptr);
  delete ptr2;
}
/**
 * Test Description
 * ------------------------
 * - Verify the release of reference and destructor execution
 *   will be deferred until the graph is synchronized.
 * - References are only released at hipGraphDestroy()
 *   and hipGraphExecDestroy() calls
 * - This test is to verify the above cases with a graph created
 *   from stream capture and with Registered memory.
 * Test source
 * ------------------------
 * - catch\unit\graph\hipGraphAsyncUserObj.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipGraphUserObj_HostRegister") {
  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate,
          hipDeviceAttributeMemoryClockRate, 0));
  int *A_h = new  int();
  int *A_d = nullptr;
  HIP_CHECK(hipHostRegister(A_h, sizeof(int), 0));
  REQUIRE(A_h != nullptr);
  *A_h = 1;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
  REQUIRE(A_d != nullptr);
  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream,
                                  hipStreamCaptureModeGlobal));
  KernelFn<<< 1, 1, 0, stream>>>(A_d, clockrate, 5);
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);
  hipUserObject_t Uobj;
  int refCount = 1;
  HIP_CHECK(hipUserObjectCreate(&Uobj, A_h, destroyHostRegObj, refCount,
            hipUserObjectNoDestructorSync));
  SECTION("Flag as 0") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount, 0));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphLaunch(graph_instance, stream));
    HIP_CHECK(hipGraphExecDestroy(graph_instance));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(*A_h == 9999);
    HIP_CHECK(hipUserObjectRelease(Uobj, 1));
  }
  SECTION("Flag as hipGraphUserObjectMove") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount,
                                       hipGraphUserObjectMove));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphLaunch(graph_instance, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(*A_h == 9999);
    HIP_CHECK(hipGraphExecDestroy(graph_instance));
  }
  SECTION("Flag as hipGraphUserObjectMove with reference count more than 1") {
    hipUserObject_t Uobj_1;
    int refCount = 2;
    HIP_CHECK(hipUserObjectCreate(&Uobj_1, A_h, destroyHostRegObj, refCount,
            hipUserObjectNoDestructorSync));
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj_1, refCount,
                                       hipGraphUserObjectMove));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphLaunch(graph_instance, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(*A_h == 9999);
    HIP_CHECK(hipGraphExecDestroy(graph_instance));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostUnregister(A_h));
}
template <typename T>
__global__ void StructClassKernelFn(T *Obj,
                                 int clockrate, int WaitSecs) {
  uint64_t num_cycles = (uint64_t)clockrate;
  num_cycles = num_cycles * 1000 * WaitSecs;
  uint64_t start = clock64(), cycles = 0;
  while (cycles < num_cycles) {
    cycles = clock64() - start;
  }
  Obj->count = 9999;
}
template <typename T>
void hipUserObjectCreate_Struct_Class_Objects(T* Obj_h, T* Obj_d) {
  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate,
            hipDeviceAttributeMemoryClockRate, 0));
  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream,
                                  hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(Obj_d, Obj_h, sizeof(BoxStruct),
                           hipMemcpyHostToDevice, stream));
  StructClassKernelFn<<< 1, 1, 0, stream>>>(Obj_d, clockrate, 5);
  HIP_CHECK(hipMemcpyAsync(Obj_h, Obj_d, sizeof(BoxStruct),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);
  hipUserObject_t Uobj;
  int refCount = 1;
  HIP_CHECK(hipUserObjectCreate(&Uobj, Obj_h, destroyPinnedObj, refCount,
            hipUserObjectNoDestructorSync));
  SECTION("Flag as 0") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount, 0));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphLaunch(graph_instance, stream));
    HIP_CHECK(hipGraphExecDestroy(graph_instance));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(Obj_h->count == 9999);
    HIP_CHECK(hipUserObjectRelease(Uobj, 1));
  }
  SECTION("Flag as hipGraphUserObjectMove") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount,
                                       hipGraphUserObjectMove));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphLaunch(graph_instance, stream));
    HIP_CHECK(hipGraphExecDestroy(graph_instance));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(Obj_h->count == 9999);
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(Obj_d));
}
/**
 * Test Description
 * ------------------------
 * - Verify the release of reference and destructor execution
 *   will be deferred until the graph is synchronized.
 * - References are only released at hipGraphDestroy()
 *   and hipGraphExecDestroy() calls
 * - This test is to verify the above cases with a graph created
 *   from stream capture and with dynamic host memory as class
 *   and structure objects.
 * Test source
 * ------------------------
 * - catch\unit\graph\hipGraphAsyncUserObj.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipGraphUserObj_Struct_Class_Ojects") {
  SECTION("Called with Struct Object") {
    std::thread t1(threadFunc_dltMemory);
    BoxStruct* structObj_h;
    HIP_CHECK(hipHostMalloc(&structObj_h, sizeof(BoxStruct)));
    REQUIRE(structObj_h != nullptr);
    structObj_h->count = 1111;
    BoxStruct* structObj_d;
    HIP_CHECK(hipMalloc(&structObj_d, sizeof(BoxStruct)));
    REQUIRE(structObj_d != nullptr);
    hipUserObjectCreate_Struct_Class_Objects<BoxStruct>(structObj_h,
                                     structObj_d);
    t1.join();
  }
  SECTION("Called with Class Object") {
    std::thread t1(threadFunc_dltMemory);
    BoxClass* classObj_h;
    HIP_CHECK(hipHostMalloc(&classObj_h, sizeof(BoxClass)));
    REQUIRE(classObj_h != nullptr);
    classObj_h->count = 1111;
    BoxClass* classObj_d;
    HIP_CHECK(hipMalloc(&classObj_d, sizeof(BoxClass)));
    REQUIRE(classObj_d != nullptr);
    hipUserObjectCreate_Struct_Class_Objects<BoxClass>(classObj_h,
                                     classObj_d);
    t1.join();
  }
}
/**
 * Test Description
 * ------------------------
 * - Verify the release of reference and destructor execution
 *   will be deferred until the graph is synchronized.
 * - References are only released at hipGraphDestroy()
 *   and hipGraphExecDestroy() calls
 * - This test is to verify the above cases with a cloned graph
 *   created from stream capture and with Dynamic memory.
 * Test source
 * ------------------------
 * - catch\unit\graph\hipGraphAsyncUserObj.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipGraphUserObj_ClonedGraph") {
  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate,
          hipDeviceAttributeMemoryClockRate, 0));
  std::thread t1(threadFunc_dltMemory);
  int *hostArr = nullptr;
  HIP_CHECK(hipHostMalloc(&hostArr, sizeof(int)));
  REQUIRE(hostArr != nullptr);
  *hostArr = 1111;
  int *devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, sizeof(int)));
  REQUIRE(devArr != nullptr);
  hipGraph_t graph = nullptr, clonedgraph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream,
                                  hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(devArr, hostArr, sizeof(int),
                           hipMemcpyHostToDevice, stream));
  KernelFn<<< 1, 1, 0, stream>>>(devArr, clockrate, 5);
  HIP_CHECK(hipMemcpyAsync(hostArr, devArr, sizeof(int),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);
  hipUserObject_t Uobj;
  int refCount = 1;
  HIP_CHECK(hipUserObjectCreate(&Uobj, hostArr, destroyPinnedObj, refCount,
            hipUserObjectNoDestructorSync));
  SECTION("Flag as 0") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount, 0));
    hipGraphExec_t originalGraphInstance, clonedGraphInstance;
    // Instantiate and launch the original graph
    HIP_CHECK(hipGraphInstantiate(&originalGraphInstance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(originalGraphInstance, stream));
    HIP_CHECK(hipGraphExecDestroy(originalGraphInstance));
    REQUIRE(*hostArr == 1111);
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    REQUIRE(clonedgraph != nullptr);
    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphInstance, clonedgraph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphDestroy(clonedgraph));
    HIP_CHECK(hipGraphLaunch(clonedGraphInstance, stream));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphInstance));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(*hostArr == 9999);
    HIP_CHECK(hipUserObjectRelease(Uobj, 1));
  }
  SECTION("Flag as hipGraphUserObjectMove") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount,
                                       hipGraphUserObjectMove));
    hipGraphExec_t originalGraphInstance, clonedGraphInstance;
    // Instantiate and launch the original graph
    HIP_CHECK(hipGraphInstantiate(&originalGraphInstance, graph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(originalGraphInstance, stream));
    HIP_CHECK(hipGraphExecDestroy(originalGraphInstance));
    REQUIRE(*hostArr == 1111);
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    REQUIRE(clonedgraph != nullptr);
    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphInstance, clonedgraph,
                                  nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphDestroy(clonedgraph));
    HIP_CHECK(hipGraphLaunch(clonedGraphInstance, stream));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphInstance));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(*hostArr == 9999);
  }
  t1.join();
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devArr));
}
__global__ void ManualGraphKernelFn(int *Ad, int clockrate, int WaitSecs) {
  uint64_t num_cycles = (uint64_t)clockrate;
  num_cycles = num_cycles * 1000 * WaitSecs;
  uint64_t start = clock64(), cycles = 0;
  while (cycles < num_cycles) {
    cycles = clock64() - start;
  }
  *Ad = 9999;
}
/**
 * Test Description
 * ------------------------
 * - Verify the release of reference and destructor execution
 *   will be deferred until the graph is synchronized.
 * - References are only released at hipGraphDestroy()
 *   and hipGraphExecDestroy() calls
 * - This test is to verify the above cases with a Manual graph
 *   created from stream capture and with Dynamic memory.
 * Test source
 * ------------------------
 * - catch\unit\graph\hipGraphAsyncUserObj.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipGraphUserObj_ManualGraph") {
  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate,
            hipDeviceAttributeMemoryClockRate, 0));
  std::thread t1(threadFunc_dltMemory);
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *hostArr = nullptr;
  HIP_CHECK(hipHostMalloc(&hostArr, sizeof(int)));
  REQUIRE(hostArr != nullptr);
  *hostArr = 1111;
  int *devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, sizeof(int)));
  REQUIRE(devArr != nullptr);
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, devArr,
                                    hostArr, sizeof(int),
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  kNodeParams.func = reinterpret_cast<void*>(ManualGraphKernelFn);
  int delay = 5;
  void* kernelArgs[] = {reinterpret_cast<void*>(&devArr), &clockrate, &delay};
  kNodeParams.gridDim = dim3(1);
  kNodeParams.blockDim = dim3(1);
  kNodeParams.kernelParams = kernelArgs;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, dependencies.data(),
                                  dependencies.size(), &kNodeParams));
  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(),
                                    dependencies.size(),
                                    hostArr, devArr, sizeof(int),
                                    hipMemcpyDeviceToHost));
  REQUIRE(graph != nullptr);
  hipUserObject_t Uobj;
  int refCount = 1;
  HIP_CHECK(hipUserObjectCreate(&Uobj, hostArr, destroyPinnedObj, refCount,
            hipUserObjectNoDestructorSync));
  SECTION("Flag as 0") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount, 0));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphLaunch(graph_instance, stream));
    HIP_CHECK(hipGraphExecDestroy(graph_instance));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(*hostArr == 9999);
    HIP_CHECK(hipUserObjectRelease(Uobj, 1));
  }
  SECTION("Flag as hipGraphUserObjectMove") {
    HIP_CHECK(hipGraphRetainUserObject(graph, Uobj, refCount,
                                       hipGraphUserObjectMove));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph,
                                nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphLaunch(graph_instance, stream));
  HIP_CHECK(hipGraphExecDestroy(graph_instance));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(*hostArr == 9999);
  }
  t1.join();
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devArr));
}
/**
 * Test Description
 * ------------------------
 * - Verify the release of reference and destructor execution
 *   will be deferred until the graph is synchronized.
 * - References are only released at hipGraphDestroy()
 *   and hipGraphExecDestroy() calls
 * - This test is to verify the above cases with a graph
 *   created from stream capture and with Stack memory.
 * Test source
 * ------------------------
 * - catch\unit\graph\hipGraphAsyncUserObj.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.3
 */
__global__ void stackKernelFn(int clockrate, int WaitSecs) {
  uint64_t num_cycles = (uint64_t)clockrate;
  num_cycles = num_cycles * 1000 * WaitSecs;
  uint64_t start = clock64(), cycles = 0;
  while (cycles < num_cycles) {
    cycles = clock64() - start;
  }
}

TEST_CASE("Unit_hipGraphUserObj_StackMemory") {
  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate,
          hipDeviceAttributeMemoryClockRate, 0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int N = 1024;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  // dummy nodes to add to the graph
  void* devmem = nullptr;
  // Enqueue a slow kernel that takes time on the device
  stackKernelFn<<< 1, 1, 0, stream>>>(clockrate, 3);
  HIP_CHECK(hipMallocAsync(&devmem, sizeof(float) * N, stream));
  HIP_CHECK(hipFreeAsync(devmem, stream));
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  hipUserObject_t uo;
  int32_t if_uo_valid = 1;
  hipHostFn_t uo_dtor = [](void* cookie) -> void {
    int32_t* pvalue = static_cast<int32_t*>(cookie);
    std::cerr << "uo_dtor called on " << cookie << std::endl;
    *pvalue = 0;
  };
  uint32_t initial_ref_count = 1;
  HIP_CHECK(hipUserObjectCreate(&uo, &if_uo_valid, uo_dtor, initial_ref_count,
            hipUserObjectNoDestructorSync));
  SECTION("Flag as hipGraphUserObjectMove") {
  HIP_CHECK(hipGraphRetainUserObject(graph, uo, initial_ref_count,
                                     hipGraphUserObjectMove));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    SECTION("no release without a graph/graphExecDestroy()") {
      HIP_CHECK(hipGraphLaunch(graph_instance, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      REQUIRE(if_uo_valid == 1);
    }
    SECTION("graph_instance is destroyed without a launch") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      REQUIRE(if_uo_valid == 0);
    }
    SECTION("streamSynchronize is not called") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      REQUIRE(if_uo_valid == 0);
    }
  }
  SECTION("Flag as 0") {
    HIP_CHECK(hipGraphRetainUserObject(graph, uo, initial_ref_count, 0));
    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    SECTION("no release without a graph/graphExecDestroy()") {
      HIP_CHECK(hipGraphLaunch(graph_instance, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      REQUIRE(if_uo_valid == 1);
      HIP_CHECK(hipUserObjectRelease(uo, 1));
    }
    SECTION("graph_instance is destroyed without a launch") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      HIP_CHECK(hipUserObjectRelease(uo, 1));
      REQUIRE(if_uo_valid == 0);
    }
    SECTION("streamSynchronize is not called") {
      HIP_CHECK(hipGraphExecDestroy(graph_instance));
      HIP_CHECK(hipUserObjectRelease(uo, 1));
      REQUIRE(if_uo_valid == 0);
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
* End doxygen group GraphTest.
* @}
*/
