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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
1)  Add a kernel node which will do vector_add(d_a, d_b) and copy the result to d_c.
    Add one more kernel node which will do vector_sub(d_a, d_b) and copy the result to d_d.
    Add one more node which will do vector_add(d_c, d_d) and copy the result to d_r.
 -> Cloned the graph
    Instantiate and Launch original Graph.
    verify the result ( d_r = 2 * d_a ) for original graph [((a+b)+(a-b))=2a]
    Add one more kernel node to Cloned graph which will do vector_sub(d_r, a_d) and copy the result
to e_d. Instantiate and Launch Cloned Graph. verify the result ( e_d = a_d ) for Cloned graph
[(((a+b)+(a-b))-a)=a]
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define N (1024 * 128)

__device__ int globalIn[N];
__device__ int globalOut[N];

class ComplexGrph {
 public:
  size_t Nbytes;
  unsigned blocksPerCU;
  unsigned threadsPerBlock;
  unsigned blocks;
  hipGraph_t graph, clonedGraph;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_R;
  hipGraphNode_t kVecAdd, kVecSub, kVecRes;
  hipGraphNode_t kVecSub_r, memcpyD2H_R_C, kVecRes_cloned;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d, *D_d, *E_d, *X_d, *Y_d, *Z_d, *R_d;
  int *A_h, *B_h, *C_h, *D_h, *E_h, *X_h, *Y_h, *Z_h, *R_h;
  size_t NElem;

  ComplexGrph() {
    Nbytes = N * sizeof(int);
    blocksPerCU = 6;  // to hide latency
    threadsPerBlock = 256;
    blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
    NElem = N;

    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
    HipTest::initArrays<int>(&D_d, &E_d, &R_d, &D_h, &E_h, &R_h, N, false);
    HipTest::initArrays<int>(&X_d, &Y_d, &Z_d, &X_h, &Y_h, &Z_h, N, false);

    constructGraph();
    constructClonedGraph();
  }

  ~ComplexGrph() {
    HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
    HipTest::freeArrays<int>(D_d, E_d, R_d, D_h, E_h, R_h, false);
    HipTest::freeArrays<int>(X_d, Y_d, Z_d, X_h, Y_h, Z_h, false);
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphDestroy(clonedGraph));
    HIP_CHECK(hipStreamDestroy(stream));
  }

  void constructGraph() {
    hipGraphExec_t graphExec;

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                      hipMemcpyHostToDevice));

    void* kernelArgs[] = {&A_d, &B_d, &C_d, &NElem};
    kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kNodeParams.gridDim = dim3(blocks);
    kNodeParams.blockDim = dim3(threadsPerBlock);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, graph, nullptr, 0, &kNodeParams));

    memset(&kNodeParams, 0x00, sizeof(kNodeParams));
    void* kernelArgs1[] = {&A_d, &B_d, &D_d, &NElem};
    kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kNodeParams.gridDim = dim3(blocks);
    kNodeParams.blockDim = dim3(threadsPerBlock);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kVecSub, graph, nullptr, 0, &kNodeParams));

    memset(&kNodeParams, 0x00, sizeof(kNodeParams));
    void* kernelArgs2[] = {&C_d, &D_d, &R_d, &NElem};
    kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kNodeParams.gridDim = dim3(blocks);
    kNodeParams.blockDim = dim3(threadsPerBlock);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kVecRes, graph, nullptr, 0, &kNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_R, graph, nullptr, 0, R_h, R_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    // Dependencies list for the graph in execution
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kVecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kVecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kVecSub, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kVecSub, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kVecAdd, &kVecRes, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kVecSub, &kVecRes, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kVecRes, &memcpyD2H_R, 1));

    // Instantiate and launch the Original graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify Original graph execution result as [((a+b)+(a-b))=2a]
    for (size_t i = 0; i < NElem; i++) {
      if (R_h[i] != (2 * A_h[i])) {
        INFO("Validation failed for cloned graph at index " << i << " R_h[i] " << R_h[i]
                                                            << " A_h[i] " << A_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(graphExec));
  }

  void constructClonedGraph() {
    hipGraphExec_t clonedGraphExec;

    HIP_CHECK(hipGraphClone(&clonedGraph, graph));

    memset(&kNodeParams, 0x00, sizeof(kNodeParams));
    void* kernelArgs3[] = {&R_d, &A_d, &E_d, &NElem};
    kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kNodeParams.gridDim = dim3(blocks);
    kNodeParams.blockDim = dim3(threadsPerBlock);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs3);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kVecSub_r, clonedGraph, nullptr, 0, &kNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_R_C, clonedGraph, nullptr, 0, E_h, E_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, kVecRes, clonedGraph));

    HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &kVecSub_r, 1));
    HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecSub_r, &memcpyD2H_R_C, 1));

    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify cloned graph execution result as [(((a+b)+(a-b))-a)=a]
    for (size_t i = 0; i < NElem; i++) {
      if (E_h[i] != A_h[i]) {
        INFO("Validation failed for cloned graph at index " << i << " A_h[i] " << A_h[i]
                                                            << " E_h[i] " << E_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  }
};

/* Scenarios 2 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphKernelNodeSetParams and Instantiate and launch
 the clonedGraph and verify the update for hipGraphKernelNodeSetParams was
 done properly by verifying the result. */
static void hipGraphClone_Test_hipGraphKernelNodeSetParams() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned;
  hipKernelNodeParams kNodeParams{};

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  memset(&kNodeParams, 0x00, sizeof(kNodeParams));
  void* kernelArgs[] = {&cg.R_d, &cg.A_d, &cg.E_d, &cg.NElem};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(cg.blocks);
  kNodeParams.blockDim = dim3(cg.threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.kVecSub_r, clonedGraph));

  HIP_CHECK(hipGraphKernelNodeSetParams(kVecRes_cloned, &kNodeParams));

  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  // Verify cloned graph execution result as [(((a+b)+(a-b))+a)=3a]
  for (size_t i = 0; i < cg.NElem; i++) {
    if (cg.E_h[i] != (3 * cg.A_h[i])) {
      INFO("Validation failed for cloned graph 2 at index " << i << " A_h[i] " << cg.A_h[i]
                                                            << " E_h[i] " << cg.E_h[i]);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphKernelNodeSetParams") {
  hipGraphClone_Test_hipGraphKernelNodeSetParams();
}

/* Scenarios 3 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphExecKernelNodeSetParams and Instantiate and launch
 the clonedGraph and verify the update for hipGraphExecKernelNodeSetParams was
 done properly by verifying the result. */

static void hipGraphClone_Test_hipGraphExecKernelNodeSetParams() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned;
  hipKernelNodeParams kNodeParams{};

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  memset(&kNodeParams, 0x00, sizeof(kNodeParams));
  void* kernelArgs[] = {&cg.R_d, &cg.E_d, &cg.NElem};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kNodeParams.gridDim = dim3(cg.blocks);
  kNodeParams.blockDim = dim3(cg.threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.kVecSub_r, clonedGraph));
  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecKernelNodeSetParams(clonedGraphExec, kVecRes_cloned, &kNodeParams));
  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  // Verify cloned graph execution result as [(2a)*(2a)=4*a*a]
  for (size_t i = 0; i < cg.NElem; i++) {
    if (cg.E_h[i] != (4 * cg.A_h[i] * cg.A_h[i])) {
      INFO("Validation failed for cloned graph 3 at index " << i << " A_h[i] " << cg.A_h[i]
                                                            << " E_h[i] " << cg.E_h[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphExecKernelNodeSetParams") {
  hipGraphClone_Test_hipGraphExecKernelNodeSetParams();
}

/* Scenarios 4 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphAddMemcpy and hipGraphAddMemsetNode and Instantiate
 and launchthe clonedGraph and verify the update was
 done properly by verifying the result. */

static void hipGraphClone_Test_hipGraphAddMemcpy_and_memset() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  constexpr size_t memSetVal = 7;
  hipGraphNode_t kMemCpyH2D_X, kMemSet, memcpyD2D, memcpyD2H_RC;

  HIP_CHECK(hipGraphAddMemcpyNode1D(&kMemCpyH2D_X, clonedGraph, nullptr, 0, cg.X_d, cg.X_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(cg.X_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = cg.Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&kMemSet, clonedGraph, nullptr, 0, &memsetParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D, clonedGraph, nullptr, 0, cg.Y_d, cg.X_d, cg.Nbytes,
                                    hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_RC, clonedGraph, nullptr, 0, cg.Y_h, cg.Y_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &kMemCpyH2D_X, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kMemCpyH2D_X, &kMemSet, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kMemSet, &memcpyD2D, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyD2D, &memcpyD2H_RC, 1));

  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  memset(cg.Z_h, memSetVal, cg.Nbytes);
  // Verify cloned graph result as memset value = memSetVal
  for (size_t i = 0; i < cg.NElem; i++) {
    if (cg.Y_h[i] != cg.Z_h[i]) {
      INFO("Validation failed for cloned graph at index " << i << " Y_h[i] " << cg.Y_h[i]
                                                          << " Z_h[i] " << cg.Z_h[i]);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphAddMemcpy_and_memset") {
  hipGraphClone_Test_hipGraphAddMemcpy_and_memset();
}

/* Scenarios 5 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphMemcpyNodeSetParams and Instantiate and launch
 the clonedGraph and verify the update for hipGraphMemcpyNodeSetParams was
 done properly by verifying the result. */

static void hipGraphClone_Test_hipGraphMemcpyNodeSetParams() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  uint32_t width{128}, height{128}, depth{128};
  uint32_t size = width * height * depth * sizeof(int);
  hipGraphNode_t memcpyNodeH2D, memcpyNodeD2H, memcpyNodeD2D;
  hipMemcpy3DParms myparms, myparms1, myparms_updated;
  hipArray_t devArray, devArray_2;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;

  int *hData, *hDataTemp, *hOutputData;
  HipTest::initArrays<int>(nullptr, nullptr, nullptr, &hData, &hDataTemp, &hOutputData, size,
                           false);

  for (uint32_t i = 0; i < depth; i++) {
    for (uint32_t j = 0; j < height; j++) {
      for (uint32_t k = 0; k < width; k++) {
        hData[i * width * height + j * width + k] = i * width * height + j * width + k;
      }
    }
  }
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int) * 8, 0, 0, 0, formatKind);
  HIP_CHECK(hipMalloc3DArray(&devArray, &channelDesc, make_hipExtent(width, height, depth),
                             hipArrayDefault));
  HIP_CHECK(hipMalloc3DArray(&devArray_2, &channelDesc, make_hipExtent(width, height, depth),
                             hipArrayDefault));

  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));

  // Host to Device
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int), width, height);
  myparms.dstArray = devArray;
  myparms.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNodeH2D, clonedGraph, nullptr, 0, &myparms));

  // Device to host
  memset(&myparms1, 0x0, sizeof(hipMemcpy3DParms));
  myparms1.srcPos = make_hipPos(0, 0, 0);
  myparms1.dstPos = make_hipPos(0, 0, 0);
  myparms1.dstPtr = make_hipPitchedPtr(hDataTemp, width * sizeof(int), width, height);
  myparms1.srcArray = devArray;
  myparms1.extent = make_hipExtent(width, height, depth);
  myparms1.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNodeD2H, clonedGraph, nullptr, 0, &myparms1));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &memcpyNodeH2D, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyNodeH2D, &memcpyNodeD2H, 1));

  // Device to Device
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.srcArray = devArray;
  myparms.dstArray = devArray_2;
  myparms.kind = hipMemcpyDeviceToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNodeD2D, clonedGraph, nullptr, 0, &myparms));

  HIP_CHECK(hipGraphRemoveDependencies(clonedGraph, &memcpyNodeH2D, &memcpyNodeD2H, 1));

  // Device to host with updated host ptr hDataTemp -> hOutputData
  memset(&myparms_updated, 0x0, sizeof(hipMemcpy3DParms));
  myparms_updated.srcPos = make_hipPos(0, 0, 0);
  myparms_updated.dstPos = make_hipPos(0, 0, 0);
  myparms_updated.dstPtr = make_hipPitchedPtr(hOutputData, width * sizeof(int), width, height);
  myparms_updated.srcArray = devArray;
  myparms_updated.extent = make_hipExtent(width, height, depth);
  myparms_updated.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyNodeH2D, &memcpyNodeD2D, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyNodeD2D, &memcpyNodeD2H, 1));

  HIP_CHECK(hipGraphMemcpyNodeSetParams(memcpyNodeD2H, &myparms_updated));

  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  // Check result
  HipTest::checkArray(hData, hOutputData, width, height, depth);

  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipFreeArray(devArray));
  HIP_CHECK(hipFreeArray(devArray_2));
  HipTest::freeArrays<int>(nullptr, nullptr, nullptr, hData, hDataTemp, hOutputData, false);
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphMemcpyNodeSetParams") {
  CHECK_IMAGE_SUPPORT

  hipGraphClone_Test_hipGraphMemcpyNodeSetParams();
}

/* Scenarios 6 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphExecMemcpyNodeSetParams and Instantiate and launch
 the clonedGraph and verify the update for hipGraphExecMemcpyNodeSetParams was
 done properly by verifying the result. */

static void hipGraphClone_Test_hipGraphExecMemcpyNodeSetParams() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  constexpr int XSIZE = 1024;
  int harray1D[XSIZE]{};
  int harray1Dres[XSIZE]{};
  constexpr int width{XSIZE};
  hipArray_t devArray1, devArray2;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparams;
  hipGraphNode_t memcpyNode1, memcpyNode2, memcpyNode3;

  // Initialize 1D object
  for (int i = 0; i < XSIZE; i++) {
    harray1D[i] = i + 1;
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int) * 8, 0, 0, 0, formatKind);
  // Allocate 1D device array by passing depth(0), height(0)
  HIP_CHECK(
      hipMalloc3DArray(&devArray1, &channelDesc, make_hipExtent(width, 0, 0), hipArrayDefault));
  HIP_CHECK(
      hipMalloc3DArray(&devArray2, &channelDesc, make_hipExtent(width, 0, 0), hipArrayDefault));

  // Host to Device
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.srcPtr = make_hipPitchedPtr(harray1D, width * sizeof(int), width, 1);
  myparams.dstArray = devArray1;
  myparams.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode1, clonedGraph, nullptr, 0, &myparams));

  // Device to Device
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.srcArray = devArray1;
  myparams.dstArray = devArray2;
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.kind = hipMemcpyDeviceToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode2, clonedGraph, nullptr, 0, &myparams));

  // Device to host
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.dstPtr = make_hipPitchedPtr(harray1Dres, width * sizeof(int), width, 1);
  myparams.srcArray = devArray2;
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode3, clonedGraph, nullptr, 0, &myparams));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &memcpyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyNode1, &memcpyNode2, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyNode2, &memcpyNode3, 1));

  // Instantiate the graph
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));

  int harray1Dupdate[XSIZE]{};
  hipArray_t devArray3;
  HIP_CHECK(
      hipMalloc3DArray(&devArray3, &channelDesc, make_hipExtent(width, 0, 0), hipArrayDefault));

  // D2H updated with different pointer harray1Dres -> harray1Dupdate
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.dstPtr = make_hipPitchedPtr(harray1Dupdate, width * sizeof(int), width, 1);
  myparams.srcArray = devArray2;
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphExecMemcpyNodeSetParams(clonedGraphExec, memcpyNode3, &myparams));

  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  // Validate result
  for (int i = 0; i < XSIZE; i++) {
    if (harray1D[i] != harray1Dupdate[i]) {
      INFO("harray1D: " << harray1D[i] << " harray1Dupdate: " << harray1Dupdate[i]
                        << " mismatch at : " << i);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipFreeArray(devArray1));
  HIP_CHECK(hipFreeArray(devArray2));
  HIP_CHECK(hipFreeArray(devArray3));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphExecMemcpyNodeSetParams") {
  CHECK_IMAGE_SUPPORT

  hipGraphClone_Test_hipGraphExecMemcpyNodeSetParams();
}

/* Scenarios 7, 8 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphMemcpyNodeSetParams1D and
 hipGraphExecMemcpyNodeSetParams1D Instantiate and launch
 the clonedGraph and verify the update for hipGraphMemcpyNodeSetParams1D and
 hipGraphExecMemcpyNodeSetParams1D was done properly by verifying the result */

static void hipGraphClone_Test_hipGraphMemcpyNodeSetParams1D_and_exec() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph, clonedGraph_2;
  hipGraphExec_t clonedGraphExec, clonedGraphExec_2;
  hipGraphNode_t kVecRes_cloned, memcpyD2H_C_2;
  hipGraphNode_t memcpyH2D_E, memcpyH2D_B, memcpyD2H_C, kernel_vecAdd;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_E, clonedGraph, nullptr, 0, cg.E_d, cg.E_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, clonedGraph, nullptr, 0, cg.B_d, cg.B_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, clonedGraph, nullptr, 0, cg.C_h, cg.C_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&cg.E_d, &cg.B_d, &cg.C_d, &cg.NElem};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(cg.blocks);
  kernelNodeParams.blockDim = dim3(cg.threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, clonedGraph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &memcpyH2D_E, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyH2D_E, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kernel_vecAdd, &memcpyD2H_C, 1));

  HIP_CHECK(hipGraphClone(&clonedGraph_2, clonedGraph));

  SECTION("Verify hipGraphMemcpyNodeSetParams1D and result C_d->Y_h") {
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyD2H_C, cg.Y_h, cg.C_d, cg.Nbytes,
                                            hipMemcpyDeviceToHost));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Verify cloned graph result
    HipTest::checkVectorADD(cg.E_h, cg.B_h, cg.Y_h, N);
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  }
  SECTION("Verify hipGraphExecMemcpyNodeSetParams1D and result C_d->Z_h") {
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyD2H_C_2, memcpyD2H_C, clonedGraph_2));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_2, clonedGraph_2, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(clonedGraphExec_2, memcpyD2H_C_2, cg.Z_h, cg.C_d,
                                                cg.Nbytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec_2, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Verify cloned graph result after exec set call
    HipTest::checkVectorADD(cg.E_h, cg.B_h, cg.Z_h, N);
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_2));
  }
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(clonedGraph_2));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphMemcpyNodeSetParams1D_and_exec") {
  hipGraphClone_Test_hipGraphMemcpyNodeSetParams1D_and_exec();
}

/* Scenarios 9, 10 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphMemcpyNodeSetParamsFromSymbol and
 hipGraphExecMemcpyNodeSetParamsFromSymbol Instantiate and launch
 the clonedGraph and verify the update for hipGraphMemcpyNodeSetParamsFromSymbol
 and hipGraphExecMemcpyNodeSetParamsFromSymbol was done properly by verifying the result */

static void hipGraphClone_hipGraphMemcpyNodeSetParamsFromSymbol_exec() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph, clonedGraph_2;
  hipGraphExec_t clonedGraphExec, clonedGraphExec_2;
  hipGraphNode_t kVecRes_cloned, memcpyFromSymbol_C, memcpyD2H_Z_C;
  hipGraphNode_t memcpyToSymbol, memcpyFromSymbol, memcpyH2D_X, memcpyD2H_Z;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_X, clonedGraph, nullptr, 0, cg.X_d, cg.X_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbol, clonedGraph, nullptr, 0,
                                          HIP_SYMBOL(globalIn), cg.X_d, cg.Nbytes, 0,
                                          hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbol, clonedGraph, nullptr, 0, cg.Y_d,
                                            HIP_SYMBOL(globalIn), cg.Nbytes, 0,
                                            hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_Z, clonedGraph, nullptr, 0, cg.Z_h, cg.Z_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &memcpyH2D_X, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyH2D_X, &memcpyToSymbol, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyToSymbol, &memcpyFromSymbol, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyFromSymbol, &memcpyD2H_Z, 1));

  HIP_CHECK(hipGraphClone(&clonedGraph_2, clonedGraph));

  SECTION("Verify hipGraphMemcpyNodeSetParamsFromSymbol and result Y_d->Z_d") {
    // Update the node from Y_d -> Z_d
    HIP_CHECK(hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbol, cg.Z_d, HIP_SYMBOL(globalIn),
                                                    cg.Nbytes, 0, hipMemcpyDeviceToDevice));

    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Validating the result
    for (int i = 0; i < N; i++) {
      if (cg.X_h[i] != cg.Z_h[i]) {
        WARN("Validation failed X_h[i] " << cg.X_h[i] << " Z_h[i] " << cg.Z_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  }
  SECTION("Verify hipGraphExecMemcpyNodeSetParamsFromSymbol and Y_d->E_d") {
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyFromSymbol_C, memcpyFromSymbol, clonedGraph_2));
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyD2H_Z_C, memcpyD2H_Z, clonedGraph_2));

    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_2, clonedGraph_2, nullptr, nullptr, 0));

    // Update the node from Y_d -> E_d
    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol(clonedGraphExec_2, memcpyFromSymbol_C,
                                                        cg.E_d, HIP_SYMBOL(globalIn), cg.Nbytes, 0,
                                                        hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(clonedGraphExec_2, memcpyD2H_Z_C, cg.Z_h, cg.E_d,
                                                cg.Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphLaunch(clonedGraphExec_2, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Validating the result
    for (int i = 0; i < N; i++) {
      if (cg.X_h[i] != cg.Z_h[i]) {
        WARN("Validation failed X_h[i] " << cg.X_h[i] << " Z_h[i] " << cg.Z_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_2));
  }
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(clonedGraph_2));
}

TEST_CASE("Unit_hipGraphClone_hipGraphMemcpyNodeSetParamsFromSymbol_exec") {
  hipGraphClone_hipGraphMemcpyNodeSetParamsFromSymbol_exec();
}

/* Scenarios 11, 12 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphMemcpyNodeSetParamsToSymbol and
 hipGraphExecMemcpyNodeSetParamsToSymbol Instantiate and launch
 the clonedGraph and verify the update for hipGraphMemcpyNodeSetParamsToSymbol
 and hipGraphExecMemcpyNodeSetParamsToSymbol was done properly by verifying the result */

static void hipGraphClone_hipGraphMemcpyNodeSetParamsToSymbol_exec() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph, clonedGraph_2;
  hipGraphExec_t clonedGraphExec, clonedGraphExec_2;
  hipGraphNode_t kVecRes_cloned, memcpyToSymbol_C, memcpyH2D_Y_C;
  hipGraphNode_t memcpyToSymbol, memcpyFromSymbol, memcpyH2D_Y, memcpyD2H_Z;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_Y, clonedGraph, nullptr, 0, cg.Y_d, cg.Y_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbol, clonedGraph, nullptr, 0,
                                          HIP_SYMBOL(globalOut), cg.X_d, cg.Nbytes, 0,
                                          hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbol, clonedGraph, nullptr, 0, cg.Z_d,
                                            HIP_SYMBOL(globalOut), cg.Nbytes, 0,
                                            hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_Z, clonedGraph, nullptr, 0, cg.Z_h, cg.Z_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &memcpyH2D_Y, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyH2D_Y, &memcpyToSymbol, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyToSymbol, &memcpyFromSymbol, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyFromSymbol, &memcpyD2H_Z, 1));

  HIP_CHECK(hipGraphClone(&clonedGraph_2, clonedGraph));

  SECTION("Verify hipGraphMemcpyNodeSetParamsToSymbol and result X_d->Y_d") {
    // Update the node with source pointer from X_d to Y_d
    HIP_CHECK(hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbol, HIP_SYMBOL(globalOut), cg.Y_d,
                                                  cg.Nbytes, 0, hipMemcpyDeviceToDevice));

    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Validating the result
    for (int i = 0; i < N; i++) {
      if (cg.Z_h[i] != cg.Y_h[i]) {
        WARN("Validation failed Z_h[i] " << cg.Z_h[i] << " Y_h[i] " << cg.Y_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  }
  SECTION("Verify hipGraphExecMemcpyNodeSetParamsToSymbol and X_d->D_d") {
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyToSymbol_C, memcpyToSymbol, clonedGraph_2));
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyH2D_Y_C, memcpyH2D_Y, clonedGraph_2));

    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_2, clonedGraph_2, nullptr, nullptr, 0));

    // Update the node from X_d -> D_d
    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsToSymbol(clonedGraphExec_2, memcpyToSymbol_C,
                                                      HIP_SYMBOL(globalOut), cg.D_d, cg.Nbytes, 0,
                                                      hipMemcpyDeviceToDevice));

    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(clonedGraphExec_2, memcpyH2D_Y_C, cg.D_d, cg.Y_h,
                                                cg.Nbytes, hipMemcpyHostToDevice));

    HIP_CHECK(hipGraphLaunch(clonedGraphExec_2, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Validating the result
    for (int i = 0; i < N; i++) {
      if (cg.Z_h[i] != cg.Y_h[i]) {
        WARN("Validation failed Z_h[i] " << cg.Z_h[i] << " Y_h[i] " << cg.Y_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_2));
  }
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(clonedGraph_2));
}

TEST_CASE("Unit_hipGraphClone_hipGraphMemcpyNodeSetParamsToSymbol_exec") {
  hipGraphClone_hipGraphMemcpyNodeSetParamsToSymbol_exec();
}

/* Scenarios 13, 14 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphMemsetNodeSetParams and
 hipGraphExecMemsetNodeSetParams Instantiate and launch
 the clonedGraph and verify the update for hipGraphMemsetNodeSetParams
 and hipGraphExecMemsetNodeSetParams was done properly by verifying the result */

static void hipGraphClone_Test_hipGraphMemsetNodeSetParams_exec() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph, clonedGraph_2;
  hipGraphExec_t clonedGraphExec, clonedGraphExec_2;
  hipGraphNode_t kVecRes_cloned, kMemSet_cloned;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  constexpr size_t memSetVal = 7;
  constexpr size_t memSetVal_1 = 17;
  constexpr size_t memSetVal_2 = 77;
  hipGraphNode_t kMemCpyH2D_X, kMemSet, memcpyD2D, memcpyD2H_RC;

  HIP_CHECK(hipGraphAddMemcpyNode1D(&kMemCpyH2D_X, clonedGraph, nullptr, 0, cg.X_d, cg.X_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(cg.X_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = cg.Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&kMemSet, clonedGraph, nullptr, 0, &memsetParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D, clonedGraph, nullptr, 0, cg.Y_d, cg.X_d, cg.Nbytes,
                                    hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_RC, clonedGraph, nullptr, 0, cg.Y_h, cg.Y_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &kMemCpyH2D_X, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kMemCpyH2D_X, &kMemSet, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kMemSet, &memcpyD2D, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyD2D, &memcpyD2H_RC, 1));

  HIP_CHECK(hipGraphClone(&clonedGraph_2, clonedGraph));

  SECTION("Verify hipGraphMemsetNodeSetParams and memSetVal->memSetVal_1") {
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(cg.X_d);
    memsetParams.value = memSetVal_1;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = cg.Nbytes;
    memsetParams.height = 1;

    HIP_CHECK(hipGraphMemsetNodeSetParams(kMemSet, &memsetParams));

    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    memset(cg.Z_h, memSetVal_1, cg.Nbytes);
    // Verify cloned graph result as memset value = memSetVal_1
    for (size_t i = 0; i < cg.NElem; i++) {
      if (cg.Y_h[i] != cg.Z_h[i]) {
        INFO("Validation failed for cloned graph at index " << i << " Y_h[i] " << cg.Y_h[i]
                                                            << " Z_h[i] " << cg.Z_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  }
  SECTION("Verify hipGraphExecMemsetNodeSetParams & memSetVal->memSetVal_2") {
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(cg.X_d);
    memsetParams.value = memSetVal_2;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = cg.Nbytes;
    memsetParams.height = 1;

    HIP_CHECK(hipGraphNodeFindInClone(&kMemSet_cloned, kMemSet, clonedGraph_2));

    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_2, clonedGraph_2, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphExecMemsetNodeSetParams(clonedGraphExec_2, kMemSet_cloned, &memsetParams));

    HIP_CHECK(hipGraphLaunch(clonedGraphExec_2, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    memset(cg.Z_h, memSetVal_2, cg.Nbytes);
    // Verify cloned graph result as memset value = memSetVal_2
    for (size_t i = 0; i < cg.NElem; i++) {
      if (cg.Y_h[i] != cg.Z_h[i]) {
        INFO("Validation failed for cloned graph at index " << i << " Y_h[i] " << cg.Y_h[i]
                                                            << " Z_h[i] " << cg.Z_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_2));
  }
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(clonedGraph_2));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphMemsetNodeSetParams_exec") {
  hipGraphClone_Test_hipGraphMemsetNodeSetParams_exec();
}

#if HT_NVIDIA
/* Scenarios 15 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphRemoveDependencies and Instantiate and launch
 the clonedGraph and verify the update for hipGraphRemoveDependencies
 was done properly by verifying the result */

static void hipGraphClone_Test_hipGraphRemoveDependencies() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  constexpr size_t memSetVal = 9;
  hipGraphNode_t kMemCpyH2D_X, kMemSet, memcpyD2D, memcpyD2H_RC;

  HIP_CHECK(hipGraphAddMemcpyNode1D(&kMemCpyH2D_X, clonedGraph, nullptr, 0, cg.X_d, cg.X_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(cg.X_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = cg.Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&kMemSet, clonedGraph, nullptr, 0, &memsetParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D, clonedGraph, nullptr, 0, cg.Y_d, cg.X_d, cg.Nbytes,
                                    hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_RC, clonedGraph, nullptr, 0, cg.Y_h, cg.Y_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &kMemCpyH2D_X, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kMemCpyH2D_X, &kMemSet, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kMemSet, &memcpyD2D, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyD2D, &memcpyD2H_RC, 1));

  HIP_CHECK(hipGraphRemoveDependencies(clonedGraph, &kMemCpyH2D_X, &kMemSet, 1));
  HIP_CHECK(hipGraphRemoveDependencies(clonedGraph, &kMemSet, &memcpyD2D, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kMemCpyH2D_X, &memcpyD2D, 1));
  HIP_CHECK(hipGraphDestroyNode(kMemSet));

  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  // Verify cloned graph result as X_h == Y_h
  for (size_t i = 0; i < cg.NElem; i++) {
    if (cg.Y_h[i] != cg.X_h[i]) {
      INFO("Validation failed for cloned graph at index " << i << " Y_h[i] " << cg.Y_h[i]
                                                          << " X_h[i] " << cg.X_h[i]);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphRemoveDependencies") {
  hipGraphClone_Test_hipGraphRemoveDependencies();
}
#endif

/* Scenarios 16 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphExecChildGraphNodeSetParams and Instantiate and launch
 the clonedGraph and verify the update for hipGraphExecChildGraphNodeSetParams
 was done properly by verifying the result */

static void hipGraphClone_Test_hipGraphExecChildGraphNodeSetParams() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph, childgraph1, childgraph2;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned, kVecAdd, kVecSub, childGraphNode;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));

  hipGraphNode_t memcpyD2H_A, memcpyH2D_A, memcpyH2D_B, memcpyH2D_C;

  // Adding memcpy node to childgraph1
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, nullptr, 0, cg.B_d, cg.B_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, childgraph1, nullptr, 0, cg.C_d, cg.C_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&cg.B_d, &cg.C_d, &cg.A_d, &cg.NElem};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(cg.blocks);
  kernelNodeParams.blockDim = dim3(cg.threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, childgraph1, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddDependencies(childgraph1, &memcpyH2D_B, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph1, &memcpyH2D_C, &kVecAdd, 1));

  // Adding memcpy node to clonedGraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, clonedGraph, nullptr, 0, cg.A_d, cg.A_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  // Adding child node to clonedGraph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, clonedGraph, nullptr, 0, childgraph1));

  // Adding memcpy node to clonedGraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, clonedGraph, nullptr, 0, cg.A_h, cg.A_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyH2D_A, &childGraphNode, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &childGraphNode, &memcpyD2H_A, 1));

  // Creating another child graph with vectorADD->vectorSUB and
  // passing the new child graph to hipGraphExecChildGraphNodeSetParams API
  hipGraphNode_t memcpyH2D_B_2, memcpyH2D_C_2;

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B_2, childgraph2, nullptr, 0, cg.B_d, cg.B_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C_2, childgraph2, nullptr, 0, cg.C_d, cg.C_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  void* kernelArgs2[] = {&cg.B_d, &cg.C_d, &cg.A_d, &cg.NElem};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kernelNodeParams.gridDim = dim3(cg.blocks);
  kernelNodeParams.blockDim = dim3(cg.threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kVecSub, childgraph2, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddDependencies(childgraph2, &memcpyH2D_B_2, &kVecSub, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph2, &memcpyH2D_C_2, &kVecSub, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(clonedGraphExec, childGraphNode, childgraph2));

  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  // Verify child graph execution result
  HipTest::checkVectorSUB(cg.B_h, cg.C_h, cg.A_h, N);

  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(childgraph2));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphExecChildGraphNodeSetParams") {
  hipGraphClone_Test_hipGraphExecChildGraphNodeSetParams();
}

/* Scenarios 17, 18 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphEventRecordNodeSetEvent and
 hipGraphExecEventRecordNodeSetEvent Instantiate and launch
 the clonedGraph and verify the update for hipGraphEventRecordNodeSetEvent
 and hipGraphExecEventRecordNodeSetEvent was done properly by verifying the result */

static void hipGraphClone_Test_hipGraphEventRecordNodeSetEvent_and_Exec() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph, clonedGraph_3, clonedGraph_4, childgraph;
  hipGraphExec_t clonedGraphExec, clonedGraphExec_3;
  hipGraphNode_t kVecRes_cloned, kVecAdd, childGraphNode;
  hipGraphNode_t memcpyD2H_A, memcpyH2D_A, memcpyH2D_B, memcpyH2D_C;
  hipGraphNode_t event_rec_node_start, event_rec_node_end;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  hipEvent_t event_start, event_end;
  HIP_CHECK(hipEventCreateWithFlags(&event_start, hipEventDefault));
  HIP_CHECK(hipEventCreateWithFlags(&event_end, hipEventDefault));

  HIP_CHECK(hipGraphCreate(&childgraph, 0));

  // Adding memcpy node to childgraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph, nullptr, 0, cg.B_d, cg.B_h, cg.Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, childgraph, nullptr, 0, cg.C_d, cg.C_h, cg.Nbytes,
                                    hipMemcpyHostToDevice));

  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&cg.B_d, &cg.C_d, &cg.A_d, &cg.NElem};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(cg.blocks);
  kernelNodeParams.blockDim = dim3(cg.threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, childgraph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_B, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_C, &kVecAdd, 1));

  HIP_CHECK(
      hipGraphAddEventRecordNode(&event_rec_node_start, clonedGraph, nullptr, 0, event_start));

  HIP_CHECK(hipGraphAddEventRecordNode(&event_rec_node_end, clonedGraph, nullptr, 0, event_end));

  // Adding memcpy node to clonedGraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, clonedGraph, nullptr, 0, cg.A_d, cg.A_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  // Adding child node to clonedGraph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, clonedGraph, nullptr, 0, childgraph));

  // Adding memcpy node to clonedGraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, clonedGraph, nullptr, 0, cg.A_h, cg.A_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &event_rec_node_start, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &event_rec_node_start, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyH2D_A, &childGraphNode, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &childGraphNode, &memcpyD2H_A, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyD2H_A, &event_rec_node_end, 1));

  HIP_CHECK(hipGraphClone(&clonedGraph_3, clonedGraph));
  HIP_CHECK(hipGraphClone(&clonedGraph_4, clonedGraph));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));
  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));

  // Verify graph execution result
  HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

  float t1 = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&t1, event_start, event_end));
  REQUIRE(t1 > 0.0f);

  SECTION("Verify hipGraphEventRecordNodeSetEvent & event_end->event_end2") {
    hipEvent_t event_end2;

    HIP_CHECK(hipEventCreateWithFlags(&event_end2, hipEventBlockingSync));

    HIP_CHECK(hipGraphEventRecordNodeSetEvent(event_rec_node_end, event_end2));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    float t2 = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&t2, event_start, event_end2));
    REQUIRE(t2 > 0.0f);

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_end2));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  }
  SECTION("Verify hipGraphEventRecordNodeSetEvent & event_end->event_end3") {
    hipEvent_t event_end3;
    hipGraphNode_t event_rec_node_end_C;

    HIP_CHECK(hipEventCreateWithFlags(&event_end3, hipEventBlockingSync));

    HIP_CHECK(hipGraphNodeFindInClone(&event_rec_node_end_C, event_rec_node_end, clonedGraph_3));

    HIP_CHECK(hipGraphEventRecordNodeSetEvent(event_rec_node_end_C, event_end3));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_3, clonedGraph_3, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec_3, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    float t3 = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&t3, event_start, event_end3));
    REQUIRE(t3 > 0.0f);

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_end3));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_3));
  }
  SECTION("hipGraphExecEventRecordNodeSetEvent & event_end->event_end4") {
    hipGraphExec_t clonedGraphExec_4;
    hipEvent_t event_end4;
    hipGraphNode_t event_rec_node_end_C4;

    HIP_CHECK(hipEventCreateWithFlags(&event_end4, hipEventBlockingSync));

    HIP_CHECK(hipGraphNodeFindInClone(&event_rec_node_end_C4, event_rec_node_end, clonedGraph_4));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_4, clonedGraph_4, nullptr, nullptr, 0));

    HIP_CHECK(
        hipGraphExecEventRecordNodeSetEvent(clonedGraphExec_4, event_rec_node_end_C4, event_end4));

    HIP_CHECK(hipGraphLaunch(clonedGraphExec_4, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    float t4 = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&t4, event_start, event_end4));
    REQUIRE(t4 > 0.0f);

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_end4));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_4));
  }
  SECTION("hipGraphExecEventRecordNodeSetEvent & event_end->event_end5") {
    hipEvent_t event_end5;

    HIP_CHECK(hipEventCreateWithFlags(&event_end5, hipEventBlockingSync));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(clonedGraphExec, event_rec_node_end, event_end5));

    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    float t5 = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&t5, event_start, event_end5));
    REQUIRE(t5 > 0.0f);

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_end5));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  }

  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(clonedGraph_3));
  HIP_CHECK(hipGraphDestroy(clonedGraph_4));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIP_CHECK(hipEventDestroy(event_start));
  HIP_CHECK(hipEventDestroy(event_end));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphEventRecordNodeSetEvent_and_Exec") {
  hipGraphClone_Test_hipGraphEventRecordNodeSetEvent_and_Exec();
}

/* Scenarios 19, 20 - Once Graph and ClonedGraph created, modify Kernel node of
 clonedGraph by using hipGraphEventWaitNodeSetEvent and
 hipGraphExecEventWaitNodeSetEvent Instantiate and launch
 the clonedGraph and verify the update for hipGraphEventWaitNodeSetEvent
 and hipGraphExecEventWaitNodeSetEvent was done properly by verifying the result */

static void hipGraphClone_Test_hipGraphEventWaitNodeSetEvent_and_Exec() {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph
  hipGraph_t clonedGraph, childgraph;
  hipGraphExec_t clonedGraphExec;
  hipGraphNode_t kVecRes_cloned, kVecAdd, childGraphNode;
  hipGraphNode_t memcpyD2H_A, memcpyH2D_A, memcpyH2D_B, memcpyH2D_C;
  hipGraphNode_t event_rec_node, event_wait_node;

  HIP_CHECK(hipGraphClone(&clonedGraph, cg.clonedGraph));

  hipEvent_t event_1;
  HIP_CHECK(hipEventCreateWithFlags(&event_1, hipEventDefault));

  HIP_CHECK(hipGraphCreate(&childgraph, 0));

  // Adding memcpy node to childgraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph, nullptr, 0, cg.B_d, cg.B_h, cg.Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, childgraph, nullptr, 0, cg.C_d, cg.C_h, cg.Nbytes,
                                    hipMemcpyHostToDevice));

  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&cg.B_d, &cg.C_d, &cg.A_d, &cg.NElem};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(cg.blocks);
  kernelNodeParams.blockDim = dim3(cg.threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, childgraph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_B, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_C, &kVecAdd, 1));

  HIP_CHECK(hipGraphAddEventRecordNode(&event_rec_node, clonedGraph, nullptr, 0, event_1));

  // Adding memcpy node to clonedGraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, clonedGraph, nullptr, 0, cg.A_d, cg.A_h,
                                    cg.Nbytes, hipMemcpyHostToDevice));

  // Adding child node to clonedGraph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, clonedGraph, nullptr, 0, childgraph));

  // Adding memcpy node to clonedGraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, clonedGraph, nullptr, 0, cg.A_h, cg.A_d,
                                    cg.Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, clonedGraph, nullptr, 0, event_1));

  HIP_CHECK(hipGraphNodeFindInClone(&kVecRes_cloned, cg.memcpyD2H_R_C, clonedGraph));

  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &kVecRes_cloned, &event_rec_node, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &event_rec_node, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyH2D_A, &childGraphNode, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &childGraphNode, &memcpyD2H_A, 1));
  HIP_CHECK(hipGraphAddDependencies(clonedGraph, &memcpyD2H_A, &event_wait_node, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
  HIP_CHECK(hipStreamSynchronize(cg.stream));

  // Verify graph execution result
  HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

  SECTION("Verify hipGraphEventWaitNodeSetEvent & event_1->event_2") {
    hipEvent_t event_2;
    HIP_CHECK(hipEventCreateWithFlags(&event_2, hipEventBlockingSync));

    HIP_CHECK(hipGraphEventRecordNodeSetEvent(event_rec_node, event_2));
    HIP_CHECK(hipGraphEventWaitNodeSetEvent(event_wait_node, event_2));

    // Destroy clonedGraphExec before instantating a new one
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_2));
  }
  SECTION("Verify hipGraphEventWaitNodeSetEvent Cloned & event_1->event_3") {
    hipGraph_t clonedGraph_3;
    hipGraphExec_t clonedGraphExec_3;
    hipGraphNode_t event_rec_node_C, event_wait_node_C;
    hipEvent_t event_3;
    HIP_CHECK(hipEventCreateWithFlags(&event_3, hipEventBlockingSync));

    HIP_CHECK(hipGraphClone(&clonedGraph_3, clonedGraph));

    HIP_CHECK(hipGraphNodeFindInClone(&event_rec_node_C, event_rec_node, clonedGraph_3));
    HIP_CHECK(hipGraphNodeFindInClone(&event_wait_node_C, event_wait_node, clonedGraph_3));

    HIP_CHECK(hipGraphEventRecordNodeSetEvent(event_rec_node_C, event_3));
    HIP_CHECK(hipGraphEventWaitNodeSetEvent(event_wait_node_C, event_3));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_3, clonedGraph_3, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(clonedGraphExec_3, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_3));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_3));
    HIP_CHECK(hipGraphDestroy(clonedGraph_3));
  }
  SECTION("Verify hipGraphExecEventWaitNodeSetEvent & event_1->event_4") {
    hipEvent_t event_4;
    HIP_CHECK(hipEventCreateWithFlags(&event_4, hipEventBlockingSync));

    HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(clonedGraphExec, event_rec_node, event_4));
    HIP_CHECK(hipGraphExecEventWaitNodeSetEvent(clonedGraphExec, event_wait_node, event_4));

    HIP_CHECK(hipGraphLaunch(clonedGraphExec, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_4));
  }
  SECTION("Verify hipGraphExecEventWaitNodeSetEvent Cloned event_1->event_5") {
    hipGraph_t clonedGraph_5;
    hipGraphExec_t clonedGraphExec_5;
    hipGraphNode_t event_rec_node_C_5, event_wait_node_C_5;
    hipEvent_t event_5;
    HIP_CHECK(hipEventCreateWithFlags(&event_5, hipEventBlockingSync));

    HIP_CHECK(hipGraphClone(&clonedGraph_5, clonedGraph));

    HIP_CHECK(hipGraphNodeFindInClone(&event_rec_node_C_5, event_rec_node, clonedGraph_5));
    HIP_CHECK(hipGraphNodeFindInClone(&event_wait_node_C_5, event_wait_node, clonedGraph_5));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&clonedGraphExec_5, clonedGraph_5, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(clonedGraphExec_5, event_rec_node_C_5, event_5));
    HIP_CHECK(hipGraphExecEventWaitNodeSetEvent(clonedGraphExec_5, event_wait_node_C_5, event_5));

    HIP_CHECK(hipGraphLaunch(clonedGraphExec_5, cg.stream));
    HIP_CHECK(hipStreamSynchronize(cg.stream));

    // Verify graph execution result
    HipTest::checkVectorADD(cg.B_h, cg.C_h, cg.A_h, N);

    HIP_CHECK(hipEventDestroy(event_5));
    HIP_CHECK(hipGraphExecDestroy(clonedGraphExec_5));
    HIP_CHECK(hipGraphDestroy(clonedGraph_5));
  }

  HIP_CHECK(hipGraphExecDestroy(clonedGraphExec));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIP_CHECK(hipEventDestroy(event_1));
}

TEST_CASE("Unit_hipGraphClone_Test_hipGraphEventWaitNodeSetEvent_and_Exec") {
  hipGraphClone_Test_hipGraphEventWaitNodeSetEvent_and_Exec();
}

/* Scenarios - 21
 Using graph and cloned graph repetitively. Create a graph with Memcpy and Kernel nodes.
 Create a cloned graph. In the cloned graph modify the address in Memcpy and Kernel nodes.
 Execute both original graph and cloned graph in loop: with multiple device.
 Loop: Update input data -> Launch Graph -> Validate output data -> Goto Loop */

TEST_CASE("Unit_hipGraphClone_address_change_in_loop") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, graph_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C, kVecAdd;
  hipGraphNode_t memcpyH2D_AC, memcpyH2D_BC, memcpyD2H_CC, kVecAddC;
  hipKernelNodeParams kNodeParams{}, kNodeParams1{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d, *D_d, *E_d, *F_d;
  int *A_h, *B_h, *C_h, *D_h, *E_h, *F_h;
  hipGraphExec_t graphExec, graphExecC;
  size_t NElem{N};

  int devcount = 0;
  HIP_CHECK(hipGetDeviceCount(&devcount));

  for (int i = 0; i < 100; i++) {
    HIP_CHECK(hipSetDevice(i % devcount));

    HIP_CHECK(hipStreamCreate(&stream));
    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
    HipTest::initArrays(&D_d, &E_d, &F_d, &D_h, &E_h, &F_h, N, false);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                      hipMemcpyHostToDevice));

    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
    kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
    kNodeParams.gridDim = dim3(blocks);
    kNodeParams.blockDim = dim3(threadsPerBlock);
    kNodeParams.sharedMemBytes = 0;
    kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, graph, nullptr, 0, &kNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));

    // Dependencies list for the graph in execution
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kVecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kVecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kVecAdd, &memcpyD2H_C, 1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify graph execution result
    HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

    HIP_CHECK(hipGraphClone(&graph_C, graph));
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyH2D_AC, memcpyH2D_A, graph_C));
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyH2D_BC, memcpyH2D_B, graph_C));
    HIP_CHECK(hipGraphNodeFindInClone(&memcpyD2H_CC, memcpyD2H_C, graph_C));
    HIP_CHECK(hipGraphNodeFindInClone(&kVecAddC, kVecAdd, graph_C));

    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyH2D_AC, D_d, D_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyH2D_BC, E_d, E_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyD2H_CC, F_h, F_d, Nbytes, hipMemcpyDeviceToHost));

    void* kernelArgs1[] = {&D_d, &E_d, &F_d, reinterpret_cast<void*>(&NElem)};
    kNodeParams1.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
    kNodeParams1.gridDim = dim3(blocks);
    kNodeParams1.blockDim = dim3(threadsPerBlock);
    kNodeParams1.sharedMemBytes = 0;
    kNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
    kNodeParams1.extra = nullptr;
    HIP_CHECK(hipGraphKernelNodeSetParams(kVecAddC, &kNodeParams1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExecC, graph_C, NULL, NULL, 0));
    HIP_CHECK(hipGraphLaunch(graphExecC, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify graph execution result
    HipTest::checkVectorSUB<int>(D_h, E_h, F_h, N);

    HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
    HipTest::freeArrays(D_d, E_d, F_d, D_h, E_h, F_h, false);
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphExecDestroy(graphExecC));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphDestroy(graph_C));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

static void hipGraphClone_address_change_in_thread(hipGraph_t* graph, hipGraphNode_t* memcpyH2D_A,
                                                   hipGraphNode_t* memcpyH2D_B,
                                                   hipGraphNode_t* memcpyD2H_C,
                                                   hipGraphNode_t* kVecAdd, int dev) {
  HIP_CHECK(hipSetDevice(dev));

  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph_C;
  hipGraphExec_t graphExecC;
  hipGraphNode_t memcpyH2D_AC, memcpyH2D_BC, memcpyD2H_CC, kVecAddC;
  hipKernelNodeParams kNodeParams1{};
  hipStream_t stream;
  int *D_d, *E_d, *F_d, *D_h, *E_h, *F_h;
  size_t NElem{N};

  HipTest::initArrays(&D_d, &E_d, &F_d, &D_h, &E_h, &F_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphClone(&graph_C, *graph));
  HIP_CHECK(hipGraphNodeFindInClone(&memcpyH2D_AC, *memcpyH2D_A, graph_C));
  HIP_CHECK(hipGraphNodeFindInClone(&memcpyH2D_BC, *memcpyH2D_B, graph_C));
  HIP_CHECK(hipGraphNodeFindInClone(&memcpyD2H_CC, *memcpyD2H_C, graph_C));
  HIP_CHECK(hipGraphNodeFindInClone(&kVecAddC, *kVecAdd, graph_C));

  HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyH2D_AC, D_d, D_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyH2D_BC, E_d, E_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyD2H_CC, F_h, F_d, Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs1[] = {&D_d, &E_d, &F_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams1.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kNodeParams1.gridDim = dim3(blocks);
  kNodeParams1.blockDim = dim3(threadsPerBlock);
  kNodeParams1.sharedMemBytes = 0;
  kNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphKernelNodeSetParams(kVecAddC, &kNodeParams1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExecC, graph_C, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExecC, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB<int>(D_h, E_h, F_h, N);

  HipTest::freeArrays(D_d, E_d, F_d, D_h, E_h, F_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExecC));
  HIP_CHECK(hipGraphDestroy(graph_C));
  HIP_CHECK(hipStreamDestroy(stream));
}

/* Scenarios - 22
 Create a graph with Memcpy and Kernel nodes. Create numOfGPUs cloned graphs
 and create same number of thread, on each thread we will run the cloned graph
 with mentioned modification. Set the context to device N, Update the Src, Dst
 memory addresses in each Node and create executable graphs.
 Launch the graphs in their respective GPUs. Validate the outputs. */

TEST_CASE("Unit_hipGraphClone_address_change_in_thread") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C, kVecAdd;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&stream));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, graph, nullptr, 0, &kNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  // Dependencies list for the graph in execution
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kVecAdd, &memcpyD2H_C, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  int devcount = 0;
  HIP_CHECK(hipGetDeviceCount(&devcount));

  std::vector<std::thread> threads;

  for (int dev = 0; dev < devcount; dev++) {
    std::thread t(hipGraphClone_address_change_in_thread, &graph, &memcpyH2D_A, &memcpyH2D_B,
                  &memcpyD2H_C, &kVecAdd, dev);
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

static void hipGraphClone_Test_All_API(int dev) {
  HIP_CHECK(hipSetDevice(dev));

  hipGraphClone_Test_hipGraphKernelNodeSetParams();
  hipGraphClone_Test_hipGraphExecKernelNodeSetParams();
  hipGraphClone_Test_hipGraphAddMemcpy_and_memset();
  hipGraphClone_Test_hipGraphMemcpyNodeSetParams();
  hipGraphClone_Test_hipGraphExecMemcpyNodeSetParams();
  hipGraphClone_Test_hipGraphMemcpyNodeSetParams1D_and_exec();
  hipGraphClone_hipGraphMemcpyNodeSetParamsFromSymbol_exec();
  hipGraphClone_hipGraphMemcpyNodeSetParamsToSymbol_exec();
  hipGraphClone_Test_hipGraphMemsetNodeSetParams_exec();
#if HT_NVIDIA
  hipGraphClone_Test_hipGraphRemoveDependencies();
#endif
  hipGraphClone_Test_hipGraphExecChildGraphNodeSetParams();
  hipGraphClone_Test_hipGraphEventRecordNodeSetEvent_and_Exec();
  hipGraphClone_Test_hipGraphEventWaitNodeSetEvent_and_Exec();
}

/* Scenarios - 23
 Create a graph with Memcpy and Kernel nodes. and its cloned graph.
 Run all the above writen test cases for multiple GPU scenarios */

TEST_CASE("Unit_hipGraphClone_multi_GPU_test") {
  // FIXME: This test tests 3D as well, decouple it
  CHECK_IMAGE_SUPPORT

  int devcount = 0;
  HIP_CHECK(hipGetDeviceCount(&devcount));
  // If only single GPU is detected then return
  if (devcount < 2) {
    SUCCEED("Skipping the test-cases as number of Devices found less than 2");
    return;
  }

  for (int dev = 0; dev < devcount; dev++) {
    hipGraphClone_Test_All_API(dev);
  }
}

static void destroyIntObj(void* ptr) {
  int* ptr2 = reinterpret_cast<int*>(ptr);
  delete ptr2;
}

static void destroyFloatObj(void* ptr) {
  float* ptr2 = reinterpret_cast<float*>(ptr);
  delete ptr2;
}

/* Scenarios - 24
 Create a graph with Memcpy and Kernel nodes and make clonedGraph from this.
 Create UserObject and GraphUserObject and retain using custom reference count.
 Launch the graphs. Validate the outputs. Release the reference by calling
 hipGraphReleaseUserObject with count. */

TEST_CASE("Unit_hipGraphClone_hipUserObject_hipGraphUserObject") {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph

  int* object_i = new int();
  REQUIRE(object_i != nullptr);
  float* object_f = new float();
  REQUIRE(object_f != nullptr);

  hipUserObject_t hObject_i, hObject_f;

  HIP_CHECK(
      hipUserObjectCreate(&hObject_i, object_i, destroyIntObj, 2, hipUserObjectNoDestructorSync));
  REQUIRE(hObject_i != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject_i, 3));
  HIP_CHECK(hipGraphRetainUserObject(cg.graph, hObject_i, 2, hipGraphUserObjectMove));

  HIP_CHECK(
      hipUserObjectCreate(&hObject_f, object_f, destroyFloatObj, 3, hipUserObjectNoDestructorSync));
  REQUIRE(hObject_f != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject_f, 4));
  HIP_CHECK(hipGraphRetainUserObject(cg.clonedGraph, hObject_f, 4, hipGraphUserObjectMove));

  HIP_CHECK(hipUserObjectRelease(hObject_i, 5));
  HIP_CHECK(hipGraphReleaseUserObject(cg.graph, hObject_i, 2));

  HIP_CHECK(hipUserObjectRelease(hObject_f, 7));
  HIP_CHECK(hipGraphReleaseUserObject(cg.clonedGraph, hObject_f, 4));
}

/* Scenarios - 25
 Create a graph with Memcpy and Kernel nodes and make clonedGraph from this.
 Create UserObject and GraphUserObject and retain using custom reference count.
 Launch the graphs. Validate the outputs. Release the reference by calling
 hipGraphReleaseUserObject with count.
 (Negative - Check this should give error and reference was created for
 Oroginal graph and releasing it for other graph)*/

TEST_CASE("Unit_hipGraphClone_hipUserObject_hipGraphUserObject_Negative") {
  ComplexGrph cg;  // This will create skeleton of Graph and ClonedGraph

  int* object_i = new int();
  REQUIRE(object_i != nullptr);
  float* object_f = new float();
  REQUIRE(object_f != nullptr);

  hipUserObject_t hObject_i, hObject_f;

  HIP_CHECK(
      hipUserObjectCreate(&hObject_i, object_i, destroyIntObj, 2, hipUserObjectNoDestructorSync));
  REQUIRE(hObject_i != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject_i, 3));
  HIP_CHECK(hipGraphRetainUserObject(cg.graph, hObject_i, 2, hipGraphUserObjectMove));

  HIP_CHECK(
      hipUserObjectCreate(&hObject_f, object_f, destroyFloatObj, 3, hipUserObjectNoDestructorSync));
  REQUIRE(hObject_f != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject_f, 4));
  HIP_CHECK(hipGraphRetainUserObject(cg.clonedGraph, hObject_f, 4, hipGraphUserObjectMove));

  HIP_CHECK(hipUserObjectRelease(hObject_i, 5));
  HIP_CHECK(hipGraphReleaseUserObject(cg.clonedGraph, hObject_i, 2));

  HIP_CHECK(hipUserObjectRelease(hObject_f, 7));
  HIP_CHECK(hipGraphReleaseUserObject(cg.graph, hObject_f, 4));
}

/* Scenarios - 26
 Create a graph with Memcpy and Kernel nodes and make childGraph from this.
 Create UserObject and GraphUserObject and retain using custom reference count.
 Launch the graphs. Validate the outputs. Release the reference by calling
 hipGraphReleaseUserObject with count.
  Scenarios - 27
 Create a graph with Memcpy and Kernel nodes and make childGraph from this.
 Create UserObject and GraphUserObject and retain using custom reference count.
 Launch the graphs. Validate the outputs. Release the reference by calling
 hipGraphReleaseUserObject with count.
 (Negative - Check this should give error and reference was created for
 Oroginal graph and releasing it for other graph) */

TEST_CASE("Unit_hipGraphChild_hipUserObject_hipGraphUserObject") {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipStream_t stream;
  hipGraph_t graph, childgraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t kVecAdd, kVecSub, childGraphNode;
  hipGraphNode_t memcpyD2H_X, memcpyH2D_B, memcpyH2D_B_C, memcpyH2D_C;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  int *X_d, *X_h;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&stream));
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HipTest::initArrays<int>(&X_d, nullptr, nullptr, &X_h, nullptr, nullptr, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&childgraph, 0));

  // Adding memcpy node to childgraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B_C, childgraph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, childgraph, nullptr, 0, C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));

  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&B_d, &C_d, &A_d, &NElem};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kVecAdd, childgraph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_B_C, &kVecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_C, &kVecAdd, 1));

  int* object_i = new int();
  REQUIRE(object_i != nullptr);
  float* object_f = new float();
  REQUIRE(object_f != nullptr);

  hipUserObject_t hObject_i, hObject_f;

  HIP_CHECK(
      hipUserObjectCreate(&hObject_i, object_i, destroyIntObj, 2, hipUserObjectNoDestructorSync));
  REQUIRE(hObject_i != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject_i, 3));
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject_i, 2, hipGraphUserObjectMove));

  HIP_CHECK(
      hipUserObjectCreate(&hObject_f, object_f, destroyFloatObj, 3, hipUserObjectNoDestructorSync));
  REQUIRE(hObject_f != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject_f, 4));
  HIP_CHECK(hipGraphRetainUserObject(childgraph, hObject_f, 4, hipGraphUserObjectMove));

  // Adding child node to Graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childgraph));

  memset(&kernelNodeParams, 0x00, sizeof(hipKernelNodeParams));
  void* kernelArgs1[] = {&A_d, &B_d, &X_d, &NElem};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kVecSub, graph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_X, graph, nullptr, 0, X_h, X_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode, &kVecSub, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kVecSub, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kVecSub, &memcpyD2H_X, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  SECTION("reference created for graph and releasing it for same graph") {
    HIP_CHECK(hipUserObjectRelease(hObject_i, 5));
    HIP_CHECK(hipGraphReleaseUserObject(graph, hObject_i, 2));

    HIP_CHECK(hipUserObjectRelease(hObject_f, 7));
    HIP_CHECK(hipGraphReleaseUserObject(childgraph, hObject_f, 4));
  }

  // Verify graph execution result as C_h == X_h
  for (int i = 0; i < N; i++) {
    if (C_h[i] != X_h[i]) {
      INFO("Validation failed for graph at index " << i << " C_h[i] " << C_h[i] << " X_h[i] "
                                                   << X_h[i]);
      REQUIRE(false);
    }
  }

  SECTION("reference created for graph_i and releasing it for graph_f") {
    HIP_CHECK(hipUserObjectRelease(hObject_i, 5));
    HIP_CHECK(hipGraphReleaseUserObject(childgraph, hObject_i, 2));

    HIP_CHECK(hipUserObjectRelease(hObject_f, 7));
    HIP_CHECK(hipGraphReleaseUserObject(graph, hObject_f, 4));
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HipTest::freeArrays<int>(X_d, nullptr, nullptr, X_h, nullptr, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
