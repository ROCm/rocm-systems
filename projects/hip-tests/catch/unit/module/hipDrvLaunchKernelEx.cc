/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_cooperative_groups.h>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>

static constexpr int N = 10;
constexpr auto CODE_OBJ_SINGLEARCH = "coopKernel.code";
static constexpr auto kernel_name = "cooperativeKernelEx";

/**
 * @addtogroup hipDrvLaunchKernelEx hipDrvLaunchKernelEx
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config,
 hipFunction_t f, void** params, void** extra)`
 * Launches a HIP kernel using the driver API with the specified configuration.
 */

/**
 * Test Description
 * ------------------------
 * This test verfies the different negative scenarios of hipDrvLaunchKernelEx
 * Api Test source
 * ------------------------
 * - catch/unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipDrvLaunchKernelEx_NegTsts) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }
  int totalThreads = 64;
  int blockSize = 16;
  int numBlocks = (totalThreads + blockSize - 1) / blockSize;

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, totalThreads * sizeof(int)));
  HIP_CHECK(hipMemset(d_output, 0, totalThreads * sizeof(int)));

  // Set up the HIP_LAUNCH_CONFIG structure
  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = numBlocks;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = blockSize;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;  // default stream

  // Set up a cooperative launch attribute
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;

  config.attrs = &attr;
  config.numAttrs = 1;

  // Kernel parameters: address of d_output and totalThreads.
  void* kernelParams[] = {&d_output, &totalThreads};

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, CODE_OBJ_SINGLEARCH));

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, kernel_name));

  SECTION("Kernel config as nullptr") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(nullptr, function, kernelParams, NULL),
                    hipErrorInvalidValue);
  }
  SECTION("Kernel function as nullptr") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(&config, nullptr, kernelParams, NULL),
                    hipErrorInvalidResourceHandle);
  }
  SECTION("Kernel parameter as nullptr") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(&config, function, nullptr, NULL), hipErrorInvalidValue);
  }
  HIP_LAUNCH_CONFIG invalidConfig = {};
  invalidConfig.gridDimX = 0;
  invalidConfig.gridDimY = 1;
  invalidConfig.gridDimZ = 1;
  invalidConfig.blockDimX = 0;
  invalidConfig.blockDimY = 1;
  invalidConfig.blockDimZ = 1;
  invalidConfig.sharedMemBytes = 0;
  invalidConfig.hStream = 0;  // default stream

  // Set up a cooperative launch attribute
  hipDrvLaunchAttribute invalidAttr;
  invalidAttr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(invalidAttr.pad, 0, sizeof(invalidAttr.pad));
  invalidAttr.value.cooperative = 1;

  invalidConfig.attrs = &invalidAttr;
  invalidConfig.numAttrs = 1;

  hipError_t err = hipErrorInvalidConfiguration;
#if HT_NVIDIA
  err = hipErrorInvalidValue;
#endif
  SECTION("Invalid Kernel config") {
    HIP_CHECK_ERROR(hipDrvLaunchKernelEx(&invalidConfig, function, kernelParams, NULL), err);
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(d_output));
}

bool runTestDrvLaunch(const char* testName, std::string kernelFunc, int totalThreads, int blockSize,
                      int flagValue) {
  int numBlocks = (totalThreads + blockSize - 1) / blockSize;

  int* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, totalThreads * sizeof(int)));
  HIP_CHECK(hipMemset(d_output, 0, totalThreads * sizeof(int)));

  // Set up the HIP_LAUNCH_CONFIG structure
  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = numBlocks;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = blockSize;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;  // default stream

  // Set up a cooperative launch attribute
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;

  config.attrs = &attr;
  config.numAttrs = 1;

  // Kernel parameters: address of d_output and totalThreads.
  void* kernelParams[] = {&d_output, &totalThreads};

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, CODE_OBJ_SINGLEARCH));

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, kernelFunc.c_str()));

  // Launch the kernel using the driver API function.
  hipError_t err = hipDrvLaunchKernelEx(&config, function, kernelParams, NULL);
  if (err != hipSuccess) {
    printf("%s test failed to launch kernel: error code %d\n", testName, err);
    HIP_CHECK(hipFree(d_output));
    return false;
  }

  HIP_CHECK(hipDeviceSynchronize());

  int* h_output = (int*)malloc(totalThreads * sizeof(int));
  HIP_CHECK(hipMemcpy(h_output, d_output, totalThreads * sizeof(int), hipMemcpyDeviceToHost));

  // Verify results.
  bool success = true;
  if (h_output[0] != flagValue) {
    printf("%s test failed: Expected flag %d at index 0, got %d\n", testName, flagValue,
           h_output[0]);
    success = false;
  }
  for (int i = 1; i < totalThreads; i++) {
    int expectedValue = (flagValue == 1111) ? i : (i * 3);
    if (h_output[i] != expectedValue) {
      printf("%s test failed at index %d: Expected %d, got %d\n", testName, i, expectedValue,
             h_output[i]);
      success = false;
      break;
    }
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(d_output));
  free(h_output);
  return success;
}
/**
 * Test Description
 * ------------------------
 * This test verfies basic positive test of hipDrvLaunchKernelEx
 * Api Test source
 * ------------------------
 * - catch/unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipDrvLaunchKernelEx_Functional) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }
  REQUIRE(runTestDrvLaunch("hipDrvLaunchKernelEx", kernel_name, 64, 16, 2222) == true);
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios(Uses kernels from the module)
 *  - 1) Launch Normal kernel which has no arguments using hipDrvLaunchKernelEx
 *  - 2) Launch Normal kernel which has arguments by using hipDrvLaunchKernelEx
 *  - 3) Launch Cooperative kernel which has no arguments by using
 *  -    hipDrvLaunchKernelEx
 * Test source
 * ------------------------
 *  - unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipDrvLaunchKernelEx_With_Different_Kernels) {
  CTX_CREATE();
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "coopKernel.code"));

  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  SECTION("Normal kernel with no arguments") {
    hipFunction_t simpleKernel;
    HIP_CHECK(hipModuleGetFunction(&simpleKernel, module, "emptyKernel"));

    HIP_CHECK(hipDrvLaunchKernelEx(&config, simpleKernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, sizeof(int)));

    void* kernel_args[1] = {&devMem};

    hipFunction_t argKernel;
    HIP_CHECK(hipModuleGetFunction(&argKernel, module, "argKernel"));

    HIP_CHECK(hipDrvLaunchKernelEx(&config, argKernel, kernel_args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    int result = 0;
    HIP_CHECK(hipMemcpy(&result, devMem, sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 100);

    HIP_CHECK(hipFree(devMem));
  }

  SECTION("Cooperative kernel with no arguments") {
    hipFunction_t coopKernel;
    HIP_CHECK(hipModuleGetFunction(&coopKernel, module, "coopEmptykernel"));

    HIP_CHECK(hipDrvLaunchKernelEx(&config, coopKernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipModuleUnload(module));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *  - This testcase launches a kernel(Uses kernels from the module) which
 *  - uses the cooperative groups andwith N number of blocks and one thread
 *  - in each block by using the hipDrvLaunchKernelEx and validates
 *  - the kernel output
 * Test source
 * ------------------------
 *  - unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipDrvLaunchKernelEx_With_CooperativeKernelWithArgs) {
  CTX_CREATE();
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "coopKernel.code"));

  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = N;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 0;
  }

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(devMem1, hostMem, N * sizeof(int), hipMemcpyDefault));
  int* devMem2 = nullptr;
  HIP_CHECK(hipMalloc(&devMem2, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(devMem2, hostMem, N * sizeof(int), hipMemcpyDefault));

  int size = N;
  void* kernel_args[3] = {&devMem1, &devMem2, &size};

  hipFunction_t argKernel;
  HIP_CHECK(hipModuleGetFunction(&argKernel, module, "coopFillArrayKernel"));

  HIP_CHECK(hipDrvLaunchKernelEx(&config, argKernel, kernel_args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hostMem, devMem2, N * sizeof(int), hipMemcpyDefault));
  for (int i = 0; i < N; i++) {
    REQUIRE(hostMem[i] == 550);
  }

  HIP_CHECK(hipFree(devMem1));
  HIP_CHECK(hipFree(devMem2));
  HIP_CHECK(hipModuleUnload(module));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios(Uses kernels from the module)
 *  - 1) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the x direction by using hipLaunchKernelEx and hipDrvLaunchKernelEx
 *  - 2) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the y direction by using hipLaunchKernelEx and hipDrvLaunchKernelEx
 *  - 3) Launch a kernel with one block and maximum allowed threads in a block
 *  -    in the z direction by using hipLaunchKernelEx and hipDrvLaunchKernelEx
 * Test source
 * ------------------------
 *  - unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipDrvLaunchKernelEx_With_MaxBlockDims) {
  CTX_CREATE();
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "coopKernel.code"));

  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "emptyKernel"));

  HIP_LAUNCH_CONFIG config = {};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = 0;
  hipDrvLaunchAttribute attr;
  attr.id = hipDrvLaunchAttributeCooperative;
  // Zero out the padding bytes
  memset(attr.pad, 0, sizeof(attr.pad));
  attr.value.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    config.blockDimX = x;

    HIP_CHECK(hipDrvLaunchKernelEx(&config, kernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    config.blockDimY = y;

    HIP_CHECK(hipDrvLaunchKernelEx(&config, kernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    config.blockDimY = z;

    HIP_CHECK(hipDrvLaunchKernelEx(&config, kernel, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipModuleUnload(module));
  CTX_DESTROY();
}
#if !defined(_WIN32)
/**
 * Test Description
 * ------------------------
 *  - Verifies that hipDrvLaunchKernelEx with hipLaunchAttributeClusterDimension
 *  - is correctly captured during stream capture:
 *  - 1) Captured graph node count >= 1.
 *  - 2) hipGraphKernelNodeGetAttribute returns the captured cluster dims.
 *  - 3) hipGraphKernelNodeSetAttribute updates cluster dims on the graph node.
 *  - 4) Graph replay produces the expected kernel output.
 *  - Only runs on devices with cluster launch support (gfx1250/gfx1251).
 * Test source
 * ------------------------
 *  - unit/module/hipDrvLaunchKernelEx.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 *  - AMD GPU with cluster launch support (gfx1250/gfx1251)
 */
HIP_TEST_CASE(Unit_hipDrvLaunchKernelEx_StreamCapture_ClusterDim) {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  if (!prop.clusterLaunch) {
    HIP_SKIP_TEST("Test requires a device with cluster launch support");
  }

  static const char* fill_src = R"(
    extern "C" __global__ void fill(int* out, int val, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) out[i] = val;
    }
  )";

  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, fill_src, "fill.cu", 0, nullptr, nullptr));
  char arch_flag[64];
  snprintf(arch_flag, sizeof(arch_flag), "--offload-arch=%s", prop.gcnArchName);
  const char* opts[] = {arch_flag};
  REQUIRE(hiprtcCompileProgram(prog, 1, opts) == HIPRTC_SUCCESS);

  size_t code_size;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  hipModule_t mod;
  hipFunction_t fn;
  HIP_CHECK(hipModuleLoadData(&mod, code.data()));
  HIP_CHECK(hipModuleGetFunction(&fn, mod, "fill"));

  constexpr int N_ELEM  = 1024;
  constexpr int BLOCK   = 64;
  constexpr int GRID    = N_ELEM / BLOCK;  // 16 blocks
  constexpr int CLUSTER = 4;              // 4 blocks per cluster

  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, N_ELEM * sizeof(int)));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Warmup run before capture to ensure JIT compilation is done
  {
    HIP_LAUNCH_CONFIG cfg{};
    cfg.gridDimX = GRID; cfg.gridDimY = 1; cfg.gridDimZ = 1;
    cfg.blockDimX = BLOCK; cfg.blockDimY = 1; cfg.blockDimZ = 1;
    cfg.hStream = stream;
    hipLaunchAttribute attr{};
    attr.id = hipLaunchAttributeClusterDimension;
    attr.value.clusterDim = {CLUSTER, 1, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int val = 0, n = N_ELEM;
    void* kargs[] = {&d_out, &val, &n};
    HIP_CHECK(hipDrvLaunchKernelEx(&cfg, fn, kargs, nullptr));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipMemset(d_out, 0, N_ELEM * sizeof(int)));

  // Capture a cluster-dim kernel launch into a graph
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  {
    HIP_LAUNCH_CONFIG cfg{};
    cfg.gridDimX = GRID; cfg.gridDimY = 1; cfg.gridDimZ = 1;
    cfg.blockDimX = BLOCK; cfg.blockDimY = 1; cfg.blockDimZ = 1;
    cfg.hStream = stream;
    hipLaunchAttribute attr{};
    attr.id = hipLaunchAttributeClusterDimension;
    attr.value.clusterDim = {CLUSTER, 1, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int val = 42, n = N_ELEM;
    void* kargs[] = {&d_out, &val, &n};
    HIP_CHECK(hipDrvLaunchKernelEx(&cfg, fn, kargs, nullptr));
  }

  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // 1) Verify at least one kernel node was captured
  size_t num_nodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &num_nodes));
  REQUIRE(num_nodes >= 1);

  std::vector<hipGraphNode_t> nodes(num_nodes);
  HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &num_nodes));

  // Find the kernel node
  hipGraphNode_t kernel_node = nullptr;
  for (auto node : nodes) {
    hipGraphNodeType type;
    HIP_CHECK(hipGraphNodeGetType(node, &type));
    if (type == hipGraphNodeTypeKernel) {
      kernel_node = node;
      break;
    }
  }
  REQUIRE(kernel_node != nullptr);

  // 2) Verify hipGraphKernelNodeGetAttribute returns the captured cluster dims
  hipKernelNodeAttrValue got_attr{};
  HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_node, hipLaunchAttributeClusterDimension,
                                           &got_attr));
  REQUIRE(got_attr.clusterDim.x == CLUSTER);
  REQUIRE(got_attr.clusterDim.y == 1);
  REQUIRE(got_attr.clusterDim.z == 1);

  // 3) Update cluster dims via hipGraphKernelNodeSetAttribute and read back
  hipKernelNodeAttrValue set_attr{};
  set_attr.clusterDim = {2, 1, 1};
  HIP_CHECK(hipGraphKernelNodeSetAttribute(kernel_node, hipLaunchAttributeClusterDimension,
                                           &set_attr));

  hipKernelNodeAttrValue verify_attr{};
  HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_node, hipLaunchAttributeClusterDimension,
                                           &verify_attr));
  REQUIRE(verify_attr.clusterDim.x == 2);
  REQUIRE(verify_attr.clusterDim.y == 1);
  REQUIRE(verify_attr.clusterDim.z == 1);

  // Restore original cluster dim for replay
  set_attr.clusterDim = {CLUSTER, 1, 1};
  HIP_CHECK(hipGraphKernelNodeSetAttribute(kernel_node, hipLaunchAttributeClusterDimension,
                                           &set_attr));

  // 4) Instantiate and replay; verify correct kernel output
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> h_out(N_ELEM);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, N_ELEM * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < N_ELEM; i++) {
    REQUIRE(h_out[i] == 42);
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipModuleUnload(mod));
}
#endif  // !defined(_WIN32)

/**
 * End doxygen group ModuleTest.
 * @}
 */
