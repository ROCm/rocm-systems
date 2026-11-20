/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Test the Grid_Launch syntax.

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <utils.hh>

#include <array>
#include <cstdint>
#include <limits>

#if HT_AMD
#include <hip/hip_ext.h>
#endif


static unsigned threadsPerBlock = 256;
static unsigned blocksPerCU = 6;

// __device__ maps to __attribute__((hc))
__device__ int foo(int i) { return i + 1; }


template <typename T> __global__ void vectorADD2(T* A_d, T* B_d, T* C_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    double foo = __hiloint2double(A_d[i], B_d[i]);
    C_d[i] = __double2loint(foo) + __double2hiint(foo);
  }
}

int test_gl2(size_t N) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  // Full vadd in one large chunk, to get things started:
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(vectorADD2, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  // verify
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  return 0;
}

#if __HIP__
int test_triple_chevron(size_t N) {
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  // Full vadd in one large chunk, to get things started:
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
  vectorADD2<<<dim3(blocks), dim3(threadsPerBlock)>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  // verify
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  return 0;
}
#endif

/**
* @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 *    - Test case to verify the Grid_Launch syntax.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipGridLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

HIP_TEST_CASE(Unit_hipGridLaunch) {
  size_t N = isQuickLevel() ? 100 * 1024 : 4 * 1024 * 1024;
  SECTION("Test test_gl2") { test_gl2(N); }

#if __HIP__
  SECTION("Test triple_chevron") { test_triple_chevron(N); }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Test case to verify maxGrid limits from device properties.
 *    - Fetches maxGridSize from hipDeviceProp_t and verifies kernel launch
 *    - at maximum grid dimensions succeeds.
 *    - Also verifies that exceeding maxGridSize returns appropriate error codes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipGridLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

__global__ void emptyKernel() {}

HIP_TEST_CASE(Unit_hipGridLaunch_MaxGridDim_DeviceProperties) {
  hipDeviceProp_t deviceProp;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));

  unsigned int maxGridX = deviceProp.maxGridSize[0];
  unsigned int maxGridY = deviceProp.maxGridSize[1];
  unsigned int maxGridZ = deviceProp.maxGridSize[2];

  INFO("Device: " << deviceProp.name);
  INFO("maxGridSize[0]: " << maxGridX);
  INFO("maxGridSize[1]: " << maxGridY);
  INFO("maxGridSize[2]: " << maxGridZ);

  SECTION("Launch kernel with gridDim.x == maxGridDimX") {
    hipLaunchKernelGGL(emptyKernel, dim3(maxGridX, 1, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.y == maxGridDimY") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, maxGridY, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.z == maxGridDimZ") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, 1, maxGridZ), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }
}

HIP_TEST_CASE(Unit_hipGridLaunch_MaxGridDim_GetDeviceAttribute) {
  const unsigned int maxGridX = GetDeviceAttribute(hipDeviceAttributeMaxGridDimX, 0);
  const unsigned int maxGridY = GetDeviceAttribute(hipDeviceAttributeMaxGridDimY, 0);
  const unsigned int maxGridZ = GetDeviceAttribute(hipDeviceAttributeMaxGridDimZ, 0);

  INFO("maxGridDimX: " << maxGridX);
  INFO("maxGridDimY: " << maxGridY);
  INFO("maxGridDimZ: " << maxGridZ);

  SECTION("Launch kernel with gridDim.x == maxGridDimX using attribute") {
    hipLaunchKernelGGL(emptyKernel, dim3(maxGridX, 1, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.y == maxGridDimY using attribute") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, maxGridY, 1), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Launch kernel with gridDim.z == maxGridDimZ using attribute") {
    hipLaunchKernelGGL(emptyKernel, dim3(1, 1, maxGridZ), dim3(1, 1, 1), 0, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }
}

HIP_TEST_CASE(Unit_hipGridLaunch_ExceedMaxGridDim_Negative) {
  const int axis = GENERATE(0, 1, 2);
  constexpr hipDeviceAttribute_t attributes[] = {
      hipDeviceAttributeMaxGridDimX, hipDeviceAttributeMaxGridDimY, hipDeviceAttributeMaxGridDimZ};
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  const unsigned int maxGrid = GetDeviceAttribute(attributes[axis], device);
  REQUIRE(maxGrid > 0);
  REQUIRE(maxGrid < std::numeric_limits<unsigned int>::max());
  INFO("Axis: " << axis << ", maximum blocks: " << maxGrid);

  std::array<unsigned int, 3> grid = {1, 1, 1};
  grid[axis] = maxGrid + 1;
  SECTION("GGL rejects an oversized block grid") {
    hipLaunchKernelGGL(emptyKernel, dim3(grid[0], grid[1], grid[2]), dim3(1), 0, 0);
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidConfiguration);
  }

#if HT_AMD
  hipFunction_t function = nullptr;
  HIP_CHECK(hipGetFuncBySymbol(&function, reinterpret_cast<const void*>(emptyKernel)));

  // Keep new immediate launches bounded if a regression accepts invalid dimensions.
  // The existing GGL coverage above already exercises the immediate X limit.
  if (axis != 0 && maxGrid <= 65535) {
    SECTION("Module launch rejects an oversized block grid") {
      HIP_CHECK_ERROR(hipModuleLaunchKernel(function, grid[0], grid[1], grid[2], 1, 1, 1, 0,
                                            nullptr, nullptr, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("Extended module launch rejects too many work items with local size one") {
      HIP_CHECK_ERROR(hipExtModuleLaunchKernel(function, grid[0], grid[1], grid[2], 1, 1, 1, 0,
                                               nullptr, nullptr, nullptr),
                      hipErrorInvalidConfiguration);
    }
  }

  SECTION("Captured module launches enforce block limits without overflow or truncation") {
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    const uint64_t maxWorkItems = static_cast<uint64_t>(maxGrid) * 2;
    if (properties.maxThreadsDim[axis] < 2 || properties.maxThreadsPerBlock < 2 ||
        maxWorkItems + 1 > std::numeric_limits<unsigned int>::max()) {
      return;
    }
    const bool moduleLaunch = GENERATE(false, true);
    const bool exceed = GENERATE(false, true);
    grid[axis] = maxGrid + (exceed ? 1 : 0);
    std::array<unsigned int, 3> global = {1, 1, 1};
    std::array<unsigned int, 3> local = {1, 1, 1};
    global[axis] = static_cast<unsigned int>(maxWorkItems) + (exceed ? 1 : 0);
    local[axis] = 2;
    INFO((moduleLaunch ? "Module grid blocks: " : "Extended module work items: ")
         << (moduleLaunch ? grid[axis] : global[axis]) << ", local size: " << local[axis]);

    struct CaptureResources {
      hipStream_t stream = nullptr;
      hipGraph_t graph = nullptr;
      hipGraphExec_t executable = nullptr;
      bool capturing = false;

      ~CaptureResources() {
        if (capturing) {
          static_cast<void>(hipStreamEndCapture(stream, &graph));
        }
        if (executable != nullptr) {
          static_cast<void>(hipGraphExecDestroy(executable));
        }
        if (graph != nullptr) {
          static_cast<void>(hipGraphDestroy(graph));
        }
        if (stream != nullptr) {
          static_cast<void>(hipStreamDestroy(stream));
        }
      }
    } resources;
    HIP_CHECK(hipStreamCreateWithFlags(&resources.stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamBeginCapture(resources.stream, hipStreamCaptureModeThreadLocal));
    resources.capturing = true;
    const hipError_t launchStatus =
        moduleLaunch
            ? hipModuleLaunchKernel(function, grid[0], grid[1], grid[2], local[0], local[1],
                                    local[2], 0, resources.stream, nullptr, nullptr)
            : hipExtModuleLaunchKernel(function, global[0], global[1], global[2], local[0],
                                       local[1], local[2], 0, resources.stream, nullptr, nullptr);
    const hipError_t endStatus = hipStreamEndCapture(resources.stream, &resources.graph);
    resources.capturing = false;
    const hipError_t expectedFailureStatus =
        moduleLaunch ? hipErrorInvalidValue : hipErrorInvalidConfiguration;
    const hipError_t expectedLaunchStatus = exceed ? expectedFailureStatus : hipSuccess;
    HIP_CHECK_ERROR(launchStatus, expectedLaunchStatus);
    if (exceed) {
      HIP_CHECK_ERRORS(endStatus, hipSuccess, hipErrorStreamCaptureInvalidated);
    } else {
      HIP_CHECK(endStatus);
    }
    if (endStatus == hipSuccess) {
      size_t nodeCount = 0;
      HIP_CHECK(hipGraphGetNodes(resources.graph, nullptr, &nodeCount));
      REQUIRE(nodeCount == (exceed ? 0 : 1));
      if (!exceed) {
        hipGraphNode_t node = nullptr;
        HIP_CHECK(hipGraphGetNodes(resources.graph, &node, &nodeCount));
        hipKernelNodeParams parameters{};
        HIP_CHECK(hipGraphKernelNodeGetParams(node, &parameters));
        REQUIRE(parameters.gridDim.x == grid[0]);
        REQUIRE(parameters.gridDim.y == grid[1]);
        REQUIRE(parameters.gridDim.z == grid[2]);
        REQUIRE(parameters.blockDim.x == local[0]);
        REQUIRE(parameters.blockDim.y == local[1]);
        REQUIRE(parameters.blockDim.z == local[2]);
        // Validate the graph without dispatching billions of workgroups.
        HIP_CHECK(hipGraphInstantiate(&resources.executable, resources.graph, nullptr, nullptr, 0));
      }
    }
  }
#endif
}

/**
 * End doxygen group KernelTest.
 * @}
 */
