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

#include "vulkan_test.hh"
#include "wait_semaphore_common.hh"

TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Binary_Semaphore") {
  WaitExternalSemaphoreCommon(hipWaitExternalSemaphoresAsync);
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA
TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Timeline_Semaphore") {
  WaitExternalTimelineSemaphoreCommon(hipWaitExternalSemaphoresAsync);
}
#endif

TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Multiple_Semaphores") {
  WaitExternalMultipleSemaphoresCommon(hipWaitExternalSemaphoresAsync);
}

TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreWaitParams wait_params = {};
  wait_params.params.fence.value = 1;

  SECTION("extSemArray == nullptr") {
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(nullptr, &wait_params, 1, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("paramsArray == nullptr") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(&hip_ext_semaphore, nullptr, 1, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Wait params flag != 0") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    wait_params.flags = 1;
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(&hip_ext_semaphore, &wait_params, 1, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Invalid stream") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(&hip_ext_semaphore, &wait_params, 1, stream),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }
}