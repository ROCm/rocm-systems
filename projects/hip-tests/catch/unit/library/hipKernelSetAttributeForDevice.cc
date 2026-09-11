/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#if !HT_NVIDIA || CUDA_VERSION >= CUDA_12080

namespace {
void LoadLibraryKernel(hipLibrary_t* library, hipKernel_t* kernel) {
  std::string code_object = "library_code_load.code";
  HIP_CHECK(hipLibraryLoadFromFile(library, code_object.c_str(), nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(kernel, *library, "add_kernel"));
}
}  // namespace

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_Values) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  int observed = -1;
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  REQUIRE(observed >= 0);
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                  kernel, 0));
  REQUIRE(observed == -1);

  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0));
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  REQUIRE(observed == 0);

  for (const int value : {-1, 0, 50, 100}) {
    HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributePreferredSharedMemoryCarveout,
                                             value, 0));
    HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                    kernel, 0));
    REQUIRE(observed == value);
  }

  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterWidth, 2, 0));
  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterHeight, 1, 0));
  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterDepth, 1, 0));
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, kernel, 0));
  REQUIRE(observed == 2);
  HIP_CHECK(
      hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_CLUSTER_DIM_MUST_BE_SET, kernel, 0));
  REQUIRE(observed == 0);

  for (const int value : {-1, 0, 1, 2}) {
    HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeNonPortableClusterSizeAllowed,
                                             value, 0));
    HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                                    kernel, 0));
    REQUIRE(observed == value);
  }

  for (const int value : {0, 1, 2}) {
    HIP_CHECK(hipKernelSetAttributeForDevice(
        kernel, hipFuncAttributeClusterSchedulingPolicyPreference, value, 0));
    HIP_CHECK(hipKernelGetAttribute(
        &observed, HIP_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, kernel, 0));
    REQUIRE(observed == value);
  }
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_LaunchLimit) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  hipFunction_t function = nullptr;
  HIP_CHECK(hipKernelGetFunction(&function, kernel));

  float* output = nullptr;
  float* input_a = nullptr;
  float* input_b = nullptr;
  HIP_CHECK(hipMalloc(&output, sizeof(float)));
  HIP_CHECK(hipMalloc(&input_a, sizeof(float)));
  HIP_CHECK(hipMalloc(&input_b, sizeof(float)));
  void* args[] = {&output, &input_a, &input_b};

  constexpr unsigned int limit = 1024;
  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, limit, 0));
  HIP_CHECK(hipLaunchKernel(function, 1, 1, args, limit, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK_ERROR(hipLaunchKernel(function, 1, 1, args, limit + 1, nullptr), hipErrorInvalidValue);

  HIP_CHECK(hipFree(output));
  HIP_CHECK(hipFree(input_a));
  HIP_CHECK(hipFree(input_b));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_ClusterLaunch) {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  if (!properties.clusterLaunch) {
    HIP_SKIP_TEST("Cluster launch is not supported on this device.");
  }

  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  hipFunction_t function = nullptr;
  HIP_CHECK(hipKernelGetFunction(&function, kernel));

  float* output = nullptr;
  float* input_a = nullptr;
  float* input_b = nullptr;
  HIP_CHECK(hipMalloc(&output, sizeof(float)));
  HIP_CHECK(hipMalloc(&input_a, sizeof(float)));
  HIP_CHECK(hipMalloc(&input_b, sizeof(float)));
  void* args[] = {&output, &input_a, &input_b};

  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterWidth, 2, 0));

  hipKernelNodeParams graph_params{};
  graph_params.func = function;
  graph_params.gridDim = {2, 1, 1};
  graph_params.blockDim = {1, 1, 1};
  graph_params.kernelParams = args;
  hipGraph_t graph = nullptr;
  hipGraphNode_t graph_node = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK_ERROR(hipGraphAddKernelNode(&graph_node, graph, nullptr, 0, &graph_params),
                  hipErrorInvalidClusterSize);
  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterHeight, 1, 0));
  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterDepth, 1, 0));

  HIP_CHECK(hipLaunchKernel(function, 2, 1, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  graph = nullptr;
  graph_node = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddKernelNode(&graph_node, graph, nullptr, 0, &graph_params));
  hipKernelNodeAttrValue graph_cluster_attribute{};
  graph_cluster_attribute.clusterDim = {2, 1, 1};
  HIP_CHECK(hipGraphKernelNodeSetAttribute(graph_node, hipLaunchAttributeClusterDimension,
                                           &graph_cluster_attribute));
  graph_cluster_attribute.clusterDim = {4, 1, 1};
  HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(graph_node, hipLaunchAttributeClusterDimension,
                                                 &graph_cluster_attribute),
                  hipErrorInvalidClusterSize);
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  hipLaunchAttribute cluster_attribute{};
  cluster_attribute.id = hipLaunchAttributeClusterDimension;
  cluster_attribute.val.clusterDim = {2, 1, 1};
  hipLaunchConfig_t launch_config{};
  launch_config.gridDim = {2, 1, 1};
  launch_config.blockDim = {1, 1, 1};
  launch_config.attrs = &cluster_attribute;
  launch_config.numAttrs = 1;
  HIP_CHECK(hipLaunchKernelExC(&launch_config, function, args));
  HIP_CHECK(hipDeviceSynchronize());

  cluster_attribute.val.clusterDim = {4, 1, 1};
  launch_config.gridDim = {4, 1, 1};
  HIP_CHECK_ERROR(hipLaunchKernelExC(&launch_config, function, args), hipErrorInvalidClusterSize);

  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterHeight, 0, 0));
  HIP_CHECK_ERROR(hipLaunchKernel(function, 2, 1, args, 0, nullptr), hipErrorInvalidClusterSize);

  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterWidth, 0, 0));
  HIP_CHECK(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterDepth, 0, 0));
  HIP_CHECK(hipLaunchKernel(function, 1, 1, args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  cluster_attribute.val.clusterDim = {2, 1, 1};
  launch_config.gridDim = {2, 1, 1};
  HIP_CHECK(hipLaunchKernelExC(&launch_config, function, args));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(output));
  HIP_CHECK(hipFree(input_a));
  HIP_CHECK(hipFree(input_b));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_CrossDevice) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;

  SECTION("LoadFromFile") {
    LoadLibraryKernel(&library, &kernel);
  }

  SECTION("LoadData then overwrite image") {
    std::ifstream input("library_code_load.code", std::ios::binary | std::ios::ate);
    REQUIRE(input.good());
    const std::streamsize image_size = input.tellg();
    REQUIRE(image_size > 0);
    std::vector<char> image(static_cast<size_t>(image_size));
    input.seekg(0, std::ios::beg);
    REQUIRE(input.read(image.data(), image_size).good());

    HIP_CHECK(hipLibraryLoadData(&library, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));
    std::fill(image.begin(), image.end(), 0);
  }

  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0));
  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 1024, 1));

  int current_device = -1;
  HIP_CHECK(hipGetDevice(&current_device));
  REQUIRE(current_device == 0);

  int device_0_value = -1;
  int device_1_value = -1;
  HIP_CHECK(hipKernelGetAttribute(&device_0_value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  HIP_CHECK(hipKernelGetAttribute(&device_1_value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 1));
  REQUIRE(device_0_value == 0);
  REQUIRE(device_1_value == 1024);
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_FunctionOverride) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 1024, 0));

  hipFunction_t function = nullptr;
  HIP_CHECK(hipKernelGetFunction(&function, kernel));
  HIP_CHECK(hipFuncSetAttribute(function, hipFuncAttributeMaxDynamicSharedMemorySize, 0));

  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 2048, 0));

  int kernel_value = -1;
  int function_value = -1;
  HIP_CHECK(hipKernelGetAttribute(&kernel_value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  HIP_CHECK(hipFuncGetAttribute(&function_value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                function));
  REQUIRE(kernel_value == 2048);
  REQUIRE(function_value == 0);

  float* output = nullptr;
  float* input_a = nullptr;
  float* input_b = nullptr;
  HIP_CHECK(hipMalloc(&output, sizeof(float)));
  HIP_CHECK(hipMalloc(&input_a, sizeof(float)));
  HIP_CHECK(hipMalloc(&input_b, sizeof(float)));
  void* args[] = {&output, &input_a, &input_b};

  HIP_CHECK_ERROR(hipLaunchKernel(function, 1, 1, args, 1, nullptr), hipErrorInvalidValue);

  HIP_CHECK(hipFuncSetAttribute(function, hipFuncAttributeMaxDynamicSharedMemorySize, 2048));
  HIP_CHECK(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0));
  HIP_CHECK(hipLaunchKernel(function, 1, 1, args, 1024, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(output));
  HIP_CHECK(hipFree(input_a));
  HIP_CHECK(hipFree(input_b));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Negative_Parameters) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  HIP_CHECK_ERROR(
      hipKernelSetAttributeForDevice(nullptr, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0),
      hipErrorInvalidResourceHandle);

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  HIP_CHECK_ERROR(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, -1),
      hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize,
                                                 0, device_count),
                  hipErrorInvalidDevice);

  for (const auto attribute : {static_cast<hipFuncAttribute>(-1),
                               hipFuncAttributeClusterDimMustBeSet, hipFuncAttributeMax}) {
    HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(kernel, attribute, 0, 0), hipErrorInvalidValue);
  }

  HIP_CHECK_ERROR(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, -1, 0),
      hipErrorInvalidValue);
  HIP_CHECK_ERROR(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributePreferredSharedMemoryCarveout, -2, 0),
      hipErrorInvalidValue);
  HIP_CHECK_ERROR(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributePreferredSharedMemoryCarveout, 101, 0),
      hipErrorInvalidValue);
  HIP_CHECK_ERROR(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeRequiredClusterWidth, -1, 0),
      hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeClusterSchedulingPolicyPreference, -1, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeClusterSchedulingPolicyPreference, 3, 0),
                  hipErrorInvalidValue);
  HIP_CHECK(hipLibraryUnload(library));
}

#endif
