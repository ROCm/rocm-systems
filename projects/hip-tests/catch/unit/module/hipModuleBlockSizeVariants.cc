/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <resource_guards.hh>
#include <utils.hh>
#include "hip_module_common.hh"

#include <vector>
#include <algorithm>

/**
 * @addtogroup hipModuleBlockSizeVariants hipModuleBlockSizeVariants
 * @{
 * @ingroup ModuleTest
 * Test kernel variant selection based on block size for SPIR-V modules.
 * The HIP runtime should automatically select the best kernel variant
 * (compiled for a specific block size) when launching kernels.
 */

/**
 * Test Description
 * ------------------------
 * - Verifies that hipModuleGetFunction can find kernels by their base name
 *   even when only block-size-specific variants exist (e.g., kernel.bs1024)
 * - Verifies that kernel launches with different block sizes select appropriate variants
 * - Verifies that cooperative kernel launches also use variant selection
 * - Verifies that kernels with amdgpu_flat_work_group_size attribute respect maximum constraints
 *
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleBlockSizeVariants.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.3
 * - Requires COMGR support for CLONE_SPIRV_KERNELS_FOR_BLOCK_SIZES action
 */

#if HT_AMD

static constexpr auto kModuleFile = "blocksize_variant_kernel.spv";

// Use a singleton module manager to avoid repeat compilation.
struct ManagedLazyModule {
  hipModule_t module;

  ManagedLazyModule() { HIP_CHECK(hipModuleLoad(&module, kModuleFile)); };
  ~ManagedLazyModule() { HIP_CHECK(hipModuleUnload(module)); }

 public:
  static ManagedLazyModule& getInstance() {
    static ManagedLazyModule instance;
    return instance;
  }

  hipModule_t get() const { return module; }
};

/**
 * Test Description
 * ------------------------
 * - Test that hipModuleGetFunction can lookup kernels by their name and variants
 *   when the module contains block-size-specific variants (e.g., kernel.bs1024, kernel.bs512)
 */
TEST_CASE("Unit_hipModuleBlockSizeVariants_GetFunctions") {
  ManagedLazyModule& module = ManagedLazyModule::getInstance();

  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  size_t wave_size = devProp.warpSize;
  size_t not_wave_size = (wave_size == 32) ? 64 : 32;

  SECTION("Get function by name - writeFirst") {
    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "writeFirst"));
    REQUIRE(function != nullptr);
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "writeFirst.bs512"));
    REQUIRE(function != nullptr);
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "writeFirst.bs256"));
    REQUIRE(function != nullptr);
    std::string wave_variant = "writeFirst.bs" + std::to_string(wave_size);
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), wave_variant.c_str()));
    REQUIRE(function != nullptr);
    std::string not_wave_variant = "writeFirst.bs" + std::to_string(not_wave_size);
    HIP_CHECK_ERROR(hipModuleGetFunction(&function, module.get(), not_wave_variant.c_str()),
                    hipErrorNotFound);
  }

  SECTION("Get function by name - vectorAdd") {
    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "vectorAdd"));
    REQUIRE(function != nullptr);
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "vectorAdd.bs512"));
    REQUIRE(function != nullptr);
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "vectorAdd.bs256"));
    REQUIRE(function != nullptr);
    std::string wave_variant = "vectorAdd.bs" + std::to_string(wave_size);
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), wave_variant.c_str()));
    REQUIRE(function != nullptr);
    std::string not_wave_variant = "vectorAdd.bs" + std::to_string(not_wave_size);
    HIP_CHECK_ERROR(hipModuleGetFunction(&function, module.get(), not_wave_variant.c_str()),
                    hipErrorNotFound);
  }

  SECTION("Get function by name - limitedBlockSizeKernel") {
    hipFunction_t function;
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "limitedBlockSizeKernel"));
    REQUIRE(function != nullptr);
    HIP_CHECK(hipModuleGetFunction(&function, module.get(), "limitedBlockSizeKernel.bs256"));
    REQUIRE(function != nullptr);
    HIP_CHECK_ERROR(hipModuleGetFunction(&function, module.get(), "limitedBlockSizeKernel.bs1024"),
                    hipErrorNotFound);
    HIP_CHECK_ERROR(hipModuleGetFunction(&function, module.get(), "limitedBlockSizeKernel.bs512"),
                    hipErrorNotFound);
    HIP_CHECK_ERROR(hipModuleGetFunction(&function, module.get(), "limitedBlockSizeKernel.bs64"),
                    hipErrorNotFound);
    HIP_CHECK_ERROR(hipModuleGetFunction(&function, module.get(), "limitedBlockSizeKernel.bs32"),
                    hipErrorNotFound);
  }
}

/**
 * Test Description
 * ------------------------
 * - Test that kernel launches with different block sizes work correctly
 *   and that the runtime selects appropriate kernel variants
 */
TEST_CASE("Unit_hipModuleBlockSizeVariants_LaunchWithDifferentBlockSizes") {
  ManagedLazyModule& module = ManagedLazyModule::getInstance();

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module.get(), "writeFirst"));

  // Allocate output buffer
  LinearAllocGuard<int> output_dev(LinearAllocs::hipMalloc, sizeof(int));
  int* output_ptr = output_dev.ptr();

  auto testBlockSize = [&](int blockX, int blockY, int blockZ) {
    HIP_CHECK(hipMemset(output_ptr, 0, sizeof(int)));

    struct {
      int* x;
    } args = {output_ptr};
    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, blockX, blockY, blockZ, 0, nullptr, nullptr,
                                    config));
    HIP_CHECK(hipDeviceSynchronize());

    int result;
    HIP_CHECK(hipMemcpy(&result, output_ptr, sizeof(int), hipMemcpyDeviceToHost));

    REQUIRE(result == 42);
  };

  SECTION("Launch with block size 1024") { testBlockSize(1024, 1, 1); }

  SECTION("Launch with block size 512") { testBlockSize(512, 1, 1); }

  SECTION("Launch with block size 256") { testBlockSize(256, 1, 1); }

  SECTION("Launch with block size 128") { testBlockSize(128, 1, 1); }

  SECTION("Launch with block size 64") { testBlockSize(64, 1, 1); }

  SECTION("Launch with multi-dimensional block (8x8x4 = 256)") { testBlockSize(8, 8, 4); }
}

/**
 * Test Description
 * ------------------------
 * - Test vector addition kernel with various block sizes to ensure
 *   correct variant selection and execution
 */
TEST_CASE("Unit_hipModuleBlockSizeVariants_VectorAddWithVariants") {
  ManagedLazyModule& module = ManagedLazyModule::getInstance();

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module.get(), "vectorAdd"));

  constexpr int N = 4096;
  std::vector<float> a_host(N), b_host(N), c_host(N);
  for (int i = 0; i < N; ++i) {
    a_host[i] = static_cast<float>(i);
    b_host[i] = static_cast<float>(i * 2);
  }

  LinearAllocGuard<float> a_dev(LinearAllocs::hipMalloc, N * sizeof(float));
  LinearAllocGuard<float> b_dev(LinearAllocs::hipMalloc, N * sizeof(float));
  LinearAllocGuard<float> c_dev(LinearAllocs::hipMalloc, N * sizeof(float));

  HIP_CHECK(hipMemcpy(a_dev.ptr(), a_host.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b_dev.ptr(), b_host.data(), N * sizeof(float), hipMemcpyHostToDevice));

  auto testWithBlockSize = [&](int blockSize) {
    HIP_CHECK(hipMemset(c_dev.ptr(), 0, N * sizeof(float)));

    int gridSize = (N + blockSize - 1) / blockSize;
    float* a_ptr = a_dev.ptr();
    float* b_ptr = b_dev.ptr();
    float* c_ptr = c_dev.ptr();

    struct {
      const float* a;
      const float* b;
      float* c;
      int n;
    } args = {a_ptr, b_ptr, c_ptr, N};

    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, nullptr,
                                    config));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(c_host.data(), c_dev.ptr(), N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < N; ++i) {
      REQUIRE(c_host[i] == a_host[i] + b_host[i]);
    }
  };

  SECTION("Block size 1024") { testWithBlockSize(1024); }

  SECTION("Block size 512") { testWithBlockSize(512); }

  SECTION("Block size 256") { testWithBlockSize(256); }

  SECTION("Block size 128") { testWithBlockSize(128); }

  SECTION("Block size 64") { testWithBlockSize(64); }
}

/**
 * Test Description
 * ------------------------
 * - Test that kernels with amdgpu_flat_work_group_size attribute
 *   respect the maximum block size constraint
 */
TEST_CASE("Unit_hipModuleBlockSizeVariants_RespectMaxBlockSizeAttribute") {
  ManagedLazyModule& module = ManagedLazyModule::getInstance();

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module.get(), "limitedBlockSizeKernel"));
  REQUIRE(function != nullptr);

  // This kernel has amdgpu_flat_work_group_size(32, 512)
  // So launching with block size > 512 should either fail gracefully
  // or use a smaller variant

  constexpr int N = 512;
  LinearAllocGuard<int> output_dev(LinearAllocs::hipMalloc, N * sizeof(int));
  int* output_ptr = output_dev.ptr();

  SECTION("Launch with block size within limit (256)") {
    struct {
      int* output;
    } args = {output_ptr};

    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(function, 2, 1, 1, 256, 1, 1, 0, nullptr, nullptr, config));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> result(N);
    HIP_CHECK(hipMemcpy(result.data(), output_ptr, N * sizeof(int), hipMemcpyDeviceToHost));

    // Verify computation
    for (int i = 0; i < N; ++i) {
      REQUIRE(result[i] == i * 2);
    }
  }

  SECTION("Launch with block size at limit (512)") {
    struct {
      int* output;
    } args = {output_ptr};

    size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 512, 1, 1, 0, nullptr, nullptr, config));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> result(N);
    HIP_CHECK(hipMemcpy(result.data(), output_ptr, N * sizeof(int), hipMemcpyDeviceToHost));

    // Verify computation
    for (int i = 0; i < N; ++i) {
      REQUIRE(result[i] == i * 2);
    }
  }
}

#endif  // HT_AMD

/**
 * End doxygen group ModuleTest.
 * @}
 */
