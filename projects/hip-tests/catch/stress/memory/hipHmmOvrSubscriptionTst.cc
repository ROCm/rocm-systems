/*
Copyright (c) 2021-Present Advanced Micro Devices, Inc. All rights reserved.

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

/* Test Case Description: This test case tests the working of OverSubscription
   feature which is part of HMM.*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_process.hh>

__global__ void floatx2(float* ptr, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    ptr[i] *= 2;
  }
}

TEST_CASE("Stress_HMM_OverSubscriptionTst") {
  int hmm = 0;
  HIP_CHECK(hipDeviceGetAttribute(&hmm, hipDeviceAttributeManagedMemory, 0));

  bool shouldRun = []() -> bool {
#if HT_AMD  // For AMD this gcn arch needs to have xnack+
    int device = 0;
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    std::string arch(props.gcnArchName);
    return arch.find("xnack+") != std::string::npos;
#else  // For CUDA this depends on SM and attribute check should be fine
    return true;
#endif
  }();

  if (hmm == 1 && shouldRun) {
    hip::SpawnProc proc("hold_memory", true);
    proc.run_async();
    size_t freeMem, totalMem;
    HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));

    constexpr float oversub_factor = 1.2f;
    auto system_ram = HipTest::getMemoryAmount();  // In MB

    // Take in account of system memory
    size_t max_memory = std::min(freeMem / (1024 * 1024), system_ram);

    size_t max_mem_used = (max_memory * oversub_factor) / 1024;  // GB

    auto OneGBTest = []() {
      constexpr size_t oneGB = 1024 * 1024 * 1024;

      hipStream_t stream;
      HIP_CHECK_THREAD(hipStreamCreate(&stream));

      float* data;
      constexpr size_t alloc_elem = oneGB / sizeof(float);
      HIP_CHECK_THREAD(hipMallocManaged(&data, oneGB, hipMemAttachGlobal));

      constexpr float init_val = 1.1f;

      std::for_each(data, data + alloc_elem, [](float& a) { a = init_val; });

      // basic sanity - first and last val are same
      REQUIRE_THREAD(data[0] == init_val);
      REQUIRE_THREAD(data[alloc_elem - 1] == init_val);

      // Page migrated to GPU
      floatx2<<<(alloc_elem / 256) + 1, 256, 0, stream>>>(data, alloc_elem);

      HIP_CHECK_THREAD(hipStreamSynchronize(stream));

      // Back to host
      REQUIRE_THREAD(
          std::all_of(data, data + alloc_elem, [](float a) { return a == (2.0f * init_val); }));

      HIP_CHECK_THREAD(hipFree(data));
      HIP_CHECK_THREAD(hipStreamDestroy(stream));
    };

    std::vector<std::thread> thread_pool;
    thread_pool.reserve(max_mem_used);

    for (size_t i = 0; i < max_mem_used; i++) {
      thread_pool.emplace_back(std::thread(OneGBTest));
    }

    std::for_each(thread_pool.begin(), thread_pool.end(),
                  [](std::thread& thread) { thread.join(); });

    HIP_CHECK_THREAD_FINALIZE();
    REQUIRE(proc.wait() == 0);
  } else {
    HipTest::HIP_SKIP_TEST("Tests only supposed to run on xnack+ devices");
  }
}
