/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <algorithm>
#include <fstream>

#if HT_NVIDIA
#include <cuda_runtime_api.h>
#endif

namespace {
constexpr char kLibraryCodeObject[] = "library_code_load.code";

#if HT_NVIDIA && CUDA_VERSION >= CUDA_12080
__global__ void RegularKernelForAttributeTest() {}
#endif

// hipLibraryLoadData builds lazily, so the caller keeps this image alive through its first query.
std::vector<char> LoadLibrary(hipLibrary_t* library, bool load_from_data) {
  if (!load_from_data) {
    HIP_CHECK(hipLibraryLoadFromFile(library, kLibraryCodeObject, nullptr, nullptr, 0, nullptr,
                                     nullptr, 0));
    return {};
  }

  std::ifstream input(kLibraryCodeObject, std::ios::binary | std::ios::ate);
  REQUIRE(input.good());
  const std::streamsize image_size = input.tellg();
  REQUIRE(image_size > 0);
  std::vector<char> image(static_cast<size_t>(image_size));
  input.seekg(0, std::ios::beg);
  REQUIRE(input.read(image.data(), image_size).good());
  HIP_CHECK(hipLibraryLoadData(library, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  return image;
}
}  // namespace

// Renamed from Unit_hip_library_load_co (which collided with the more
// comprehensive case of the same name in loadlib_co.cc). This case is
// specifically a positive smoke test for hipKernelSetAttribute(
// MAX_DYNAMIC_SHARED_SIZE_BYTES) followed by a kernel launch via
// hipKernelGetFunction.
HIP_TEST_CASE(Unit_hipKernelSetAttribute_Positive_LaunchAfterSet) {
  constexpr size_t size = 32;
  constexpr size_t size_overwrite = 8;
  std::vector<float> input1, input2;
  input1.resize(size);
  input2.resize(size);
  for (size_t i = 0; i < size; i++) {
    input1[i] = (i + 1) * 2;
    input2[i] = i;
  }

  float *d_in1, *d_in2, *d_out;
  HIP_CHECK(hipMalloc(&d_in1, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_in2, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));

  HIP_CHECK(hipMemset(d_out, 0, sizeof(float) * size));
  HIP_CHECK(hipMemcpy(d_in1, input1.data(), sizeof(float) * size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_in2, input2.data(), sizeof(float) * size, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "library_code_load.code";

  hipLibrary_t library;
  hipFunction_t function;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));
  HIP_CHECK(hipKernelGetFunction(&function, kernel));
  HIP_CHECK(hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES ,sizeof(float) * size_overwrite, kernel, 0));

  void* args[] = {&d_out, &d_in1, &d_in2};

  HIP_CHECK(hipLaunchKernel(function, 1, size, args, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipLibraryUnload(library));

  std::vector<float> out(size, 0);
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < size; i++) {
    float tmp = input1[i] + input2[i];
    INFO("Index: " << i << " cpu res: " << tmp << " gpu res: " << out[i]);
    REQUIRE(out[i] == tmp);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_in1));
  HIP_CHECK(hipFree(d_in2));
  HIP_CHECK(hipFree(d_out));
}

HIP_TEST_CASE(Unit_hipKernelSetAttribute_Positive_FunctionOverride) {
  HIP_CHECK(hipSetDevice(0));
  std::string code_object = "library_code_load.code";
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  hipFunction_t function = nullptr;

  HIP_CHECK(hipLibraryLoadFromFile(&library, code_object.c_str(), nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));
  HIP_CHECK(hipKernelGetFunction(&function, kernel));

  HIP_CHECK(hipFuncSetAttribute(function, hipFuncAttributeMaxDynamicSharedMemorySize, 0));
  HIP_CHECK(
      hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 1, kernel, 0));

  int function_value = -1;
  int kernel_value = -1;
  HIP_CHECK(hipFuncGetAttribute(&function_value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                function));
  HIP_CHECK(hipKernelGetAttribute(&kernel_value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  REQUIRE(function_value == 0);
  REQUIRE(kernel_value == 1);

  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttribute_Negative_Parameters) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "library_code_load.code";

  hipLibrary_t library;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  int device_id = 0;

  SECTION("kernel == nullptr") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0, nullptr,
                              device_id),
        hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(static_cast<hipFunction_attribute>(-1), 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute MAX_THREADS_PER_BLOCK") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute NUM_REGS") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_NUM_REGS, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute SHARED_SIZE_BYTES") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("read-only attribute BINARY_VERSION") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_BINARY_VERSION, 0, kernel, device_id),
        hipErrorInvalidValue);
  }

  SECTION("MAX_DYNAMIC_SHARED_SIZE_BYTES negative value") {
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, -1, kernel,
                              device_id),
        hipErrorInvalidValue);
  }

  SECTION("invalid device id") {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    HIP_CHECK_ERROR(
        hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0, kernel,
                              device_count),
        hipErrorInvalidDevice);
  }

  SECTION("stale kernel") {
    HIP_CHECK(hipLibraryUnload(library));
    library = nullptr;
    HIP_CHECK_ERROR(hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0,
                                          kernel, device_id),
                    hipErrorInvalidHandle);
  }

  if (library != nullptr) {
    HIP_CHECK(hipLibraryUnload(library));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  (void)hipGetLastError();
}

HIP_TEST_CASE(Unit_hipKernelGetFunction_Negative_Parameters) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  std::string lib_co = "library_code_load.code";

  hipLibrary_t library;
  hipKernel_t kernel;

  HIP_CHECK(
      hipLibraryLoadFromFile(&library, lib_co.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  SECTION("pFunc == nullptr") {
    HIP_CHECK_ERROR(hipKernelGetFunction(nullptr, kernel), hipErrorInvalidValue);
  }

  SECTION("kernel == nullptr") {
    hipFunction_t function;
    HIP_CHECK_ERROR(hipKernelGetFunction(&function, nullptr), hipErrorInvalidValue);
  }

  SECTION("stale kernel") {
    HIP_CHECK(hipLibraryUnload(library));
    library = nullptr;
    hipFunction_t function = nullptr;
    HIP_CHECK_ERROR(hipKernelGetFunction(&function, kernel), hipErrorInvalidHandle);
  }

  if (library != nullptr) {
    HIP_CHECK(hipLibraryUnload(library));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  (void)hipGetLastError();
}

#if HT_NVIDIA && CUDA_VERSION >= CUDA_12080
// CUkernel and cudaKernel_t are interchangeable, but CUfunction is a distinct handle type.
HIP_TEST_CASE(Unit_hipKernelApis_CudaKernelHandleSemantics) {
  CTX_CREATE();
  HIP_CHECK(hipSetDevice(0));

  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleLoad(&module, kLibraryCodeObject));
  HIP_CHECK(hipModuleGetFunction(&function, module, "add_kernel"));
  const auto kernel = reinterpret_cast<hipKernel_t>(function);
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  int attribute_value = 0;
  HIP_CHECK_ERROR(
      hipKernelGetAttribute(&attribute_value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel, device),
      hipErrorInvalidHandle);
  HIP_CHECK_ERROR(
      hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0, kernel, device),
      hipErrorInvalidHandle);
  hipFunction_t resolved_function = nullptr;
  HIP_CHECK_ERROR(hipKernelGetFunction(&resolved_function, kernel), hipErrorInvalidHandle);
  HIP_CHECK_ERROR(hipKernelSetCacheConfig(kernel, hipFuncCachePreferNone, device),
                  hipErrorInvalidResourceHandle);
  HIP_CHECK_ERROR(
      hipKernelSetAttributeForDevice(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, device),
      hipErrorInvalidDeviceFunction);

  cudaKernel_t cuda_kernel = nullptr;
  REQUIRE(cudaGetKernel(&cuda_kernel, RegularKernelForAttributeTest) == cudaSuccess);
  const auto regular_kernel = reinterpret_cast<hipKernel_t>(cuda_kernel);
  HIP_CHECK(hipKernelGetAttribute(&attribute_value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION,
                                  regular_kernel, device));
  HIP_CHECK(hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0,
                                  regular_kernel, device));
  HIP_CHECK(hipKernelGetFunction(&resolved_function, regular_kernel));
  HIP_CHECK(hipKernelSetCacheConfig(regular_kernel, hipFuncCachePreferNone, device));
  HIP_CHECK(hipKernelSetAttributeForDevice(regular_kernel,
                                           hipFuncAttributeMaxDynamicSharedMemorySize, 0, device));

  HIP_CHECK(hipModuleUnload(module));
  (void)hipGetLastError();
  CTX_DESTROY();
}
#endif

// ROCM-34390: hipKernelGetFunction used to cast the hipKernel_t straight through, which names the
// amd::Kernel of the device the library was first built on, so launching the handle on any other
// device failed validation with hipErrorInvalidDevice and a library could not be loaded once and
// reused across GPUs. Unlike Unit_hipKernelSetAttributeForDevice_Positive_CrossDevice, this
// launches on each device instead of only reading attributes back.
HIP_TEST_CASE(Unit_hipKernelGetFunction_Positive_CrossDevice) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  constexpr size_t size = 32;
  std::vector<float> input1(size), input2(size);
  for (size_t i = 0; i < size; i++) {
    input1[i] = (i + 1) * 2;
    input2[i] = i;
  }

  HIP_CHECK(hipSetDevice(0));
  const bool load_from_data = GENERATE(false, true);
  INFO("load method: " << (load_from_data ? "data" : "file"));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  auto image = LoadLibrary(&library, load_from_data);
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));
  std::fill(image.begin(), image.end(), 0);

  for (int device = 0; device < device_count; device++) {
    INFO("device: " << device);
    HIP_CHECK(hipSetDevice(device));

    hipFunction_t function = nullptr;
    HIP_CHECK(hipKernelGetFunction(&function, kernel));
    REQUIRE(function != nullptr);

    LinearAllocGuard<float> d_out(LinearAllocs::hipMalloc, sizeof(float) * size);
    LinearAllocGuard<float> d_in1(LinearAllocs::hipMalloc, sizeof(float) * size);
    LinearAllocGuard<float> d_in2(LinearAllocs::hipMalloc, sizeof(float) * size);
    HIP_CHECK(hipMemset(d_out.ptr(), 0, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in1.ptr(), input1.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_in2.ptr(), input2.data(), sizeof(float) * size, hipMemcpyHostToDevice));

    float* out_ptr = d_out.ptr();
    float* in1_ptr = d_in1.ptr();
    float* in2_ptr = d_in2.ptr();
    void* args[] = {&out_ptr, &in1_ptr, &in2_ptr};
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, size, 1, 1, 0, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> out(size, 0);
    HIP_CHECK(hipMemcpy(out.data(), d_out.ptr(), sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("index: " << i);
      REQUIRE(out[i] == input1[i] + input2[i]);
    }
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}

// ROCM-34390: secondary library code objects originally existed only to answer attribute queries,
// so their managed symbols were never initialized. Returning one from hipKernelGetFunction made a
// managed-global kernel launchable but left its symbol null, faulting the GPU at address zero.
HIP_TEST_CASE(Unit_hipKernelGetFunction_Positive_CrossDeviceManagedGlobal) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  for (int device = 0; device < device_count; device++) {
    CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(device)
  }

  HIP_CHECK(hipSetDevice(0));
  const bool load_from_data = GENERATE(false, true);
  INFO("load method: " << (load_from_data ? "data" : "file"));
  hipLibrary_t library = nullptr;
  auto image = LoadLibrary(&library, load_from_data);

  hipKernel_t writer_kernel = nullptr;
  hipKernel_t reader_kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&writer_kernel, library, "write_m_var"));
  HIP_CHECK(hipLibraryGetKernel(&reader_kernel, library, "read_m_var"));
  std::fill(image.begin(), image.end(), 0);

  constexpr size_t size = 32;
  void* managed_ptr = nullptr;
  size_t managed_bytes = 0;
  HIP_CHECK(hipLibraryGetManaged(&managed_ptr, &managed_bytes, library, "m_var"));
  REQUIRE(managed_ptr != nullptr);
  REQUIRE(managed_bytes == sizeof(float) * size);

  for (int device = 0; device < device_count; device++) {
    INFO("device: " << device);
    HIP_CHECK(hipSetDevice(device));

    hipFunction_t writer = nullptr;
    hipFunction_t reader = nullptr;
    HIP_CHECK(hipKernelGetFunction(&writer, writer_kernel));
    HIP_CHECK(hipKernelGetFunction(&reader, reader_kernel));

    LinearAllocGuard<float> output(LinearAllocs::hipMalloc, sizeof(float) * size);
    HIP_CHECK(hipMemset(output.ptr(), 0, output.size_bytes()));

    std::vector<float> seeded(size);
    for (size_t i = 0; i < size; i++) {
      seeded[i] = static_cast<float>(device * size + i + 100);
    }
    HIP_CHECK(hipMemcpy(managed_ptr, seeded.data(), managed_bytes, hipMemcpyHostToDevice));

    float* output_ptr = output.ptr();
    void* args[] = {&output_ptr};
    HIP_CHECK(hipModuleLaunchKernel(reader, 1, 1, 1, size, 1, 1, 0, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> observed(size, 0);
    HIP_CHECK(hipMemcpy(observed.data(), output.ptr(), output.size_bytes(), hipMemcpyDeviceToHost));
    std::vector<float> managed_observed(size, 0);
    HIP_CHECK(
        hipMemcpy(managed_observed.data(), managed_ptr, managed_bytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("index: " << i);
      REQUIRE(observed[i] == seeded[i] + 1);
      REQUIRE(managed_observed[i] == seeded[i]);
    }

    HIP_CHECK(hipModuleLaunchKernel(writer, 1, 1, 1, size, 1, 1, 0, nullptr, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(
        hipMemcpy(managed_observed.data(), managed_ptr, managed_bytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("index: " << i);
      REQUIRE(managed_observed[i] == static_cast<float>(i + 1));
    }
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}

// ROCM-34390: hipKernelSetAttribute bounds-checked dev against the current hip::Device's own
// amd::Device list, which always holds exactly one entry, so every dev other than 0 was rejected
// with hipErrorInvalidDevice regardless of how many GPUs were present. It also indexed that list
// with dev, reaching the current device rather than the requested one.
HIP_TEST_CASE(Unit_hipKernelSetAttribute_Positive_CrossDevice) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&library, "library_code_load.code", nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  constexpr int device_0_value = 1024;
  constexpr int device_1_value = 2048;
  HIP_CHECK(hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, device_0_value,
                                  kernel, 0));
  HIP_CHECK(hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, device_1_value,
                                  kernel, 1));

  // The current device must not influence which device the attribute lands on.
  HIP_CHECK(hipSetDevice(1));
  int observed = -1;
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  REQUIRE(observed == device_0_value);
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 1));
  REQUIRE(observed == device_1_value);

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}
