/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_cooperative_groups.h>
#include <hip/cooperative_groups/memcpy_async.h>

// trivial vector add using shared memory
// a[i] * x + b[i]
__global__ void vector_add_mem(float* out, float* a, float* b, float x, size_t size) {
  extern __shared__ float b_smem[];
  auto tg = cooperative_groups::this_thread_block();

  // Async copy memory
  cooperative_groups::memcpy_async(tg, b_smem, b, size * sizeof(float));

  size_t i = threadIdx.x;

  // While copy is being done, we a[i] * x
  float tmp = a[i] * x;

  tg.sync();  // make sure copy is finished

  // do `+ b[i]` from shared mem
  tmp += b_smem[i];

  // Write back to shared mem, we can simplify this (do this in one go) but this is a test
  b_smem[i] = tmp;

  // Copy back data to global memory
  cooperative_groups::memcpy_async(tg, out, b_smem, size * sizeof(float));

  // Wait till its done
  tg.sync();
}

__global__ void vector_add_mem_layout(float* out, float* a, float* b, float x, size_t size) {
  extern __shared__ float b_smem[];
  auto tg = cooperative_groups::this_thread_block();

  // Async copy memory
  cooperative_groups::memcpy_async(tg, b_smem, size, b, size);

  size_t i = threadIdx.x;

  // While copy is being done, we a[i] * x
  float tmp = a[i] * x;

  tg.sync();  // make sure copy is finished

  // do `+ b[i]` from shared mem
  tmp += b_smem[i];

  // Write back to shared mem, we can simplify this (do this in one go) but this is a test
  b_smem[i] = tmp;

  // Copy back data to global memory
  cooperative_groups::memcpy_async(tg, out, size, b_smem, size);

  // Wait till its done
  tg.sync();
}

TEST_CASE("Unit_device_memcpy_async") {
  std::vector<std::string> supported_arch{"gfx1250"};
  hipDeviceProp_t prop;

  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string arch_name{prop.gcnArchName};
  std::cout << "cluster lauch: " << prop.clusterLaunch << std::endl;

  const bool should_run = std::any_of(
      supported_arch.begin(), supported_arch.end(),
      [&arch_name](const std::string_view& in) { return in.find(arch_name) != std::string::npos; });

  if (should_run) {
    SECTION("bytes copy") {
      for (const size_t size : {32, 64, 128, 129 /* weird non aligned size */}) {
        const size_t alloc_size = size * sizeof(float);
        const float x = 2.0f;
        float *d_out, *d_a, *d_b;

        HIP_CHECK(hipMalloc(&d_out, alloc_size));
        HIP_CHECK(hipMalloc(&d_a, alloc_size));
        HIP_CHECK(hipMalloc(&d_b, alloc_size));

        std::vector<float> a(size, 0.0f), b(size, 0.0f), cpu_out(size, 0.0f), gpu_out(size, 0.0f);
        for (size_t i = 0; i < size; i++) {
          a[i] = i + 1;
          b[i] = (i + 1) * 2;
          cpu_out[i] = (a[i] * x) + b[i];
        }

        HIP_CHECK(hipMemcpy(d_a, a.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, b.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_out, 0, alloc_size));

        vector_add_mem<<<1, size, alloc_size, nullptr>>>(d_out, d_a, d_b, x, size);

        HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, alloc_size, hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));

        for (size_t i = 0; i < size; i++) {
          INFO("size: " << size << " index: " << i << " calc: " << a[i] << " * " << x << " + "
                        << b[i] << " = " << cpu_out[i] << ", " << gpu_out[i]);
          REQUIRE(cpu_out[i] == gpu_out[i]);
        }
      }
    }

    SECTION("layout copy") {
      for (const size_t size : {31, 65, 128, 256}) {
        const size_t alloc_size = size * sizeof(float);
        const float x = 2.0f;
        float *d_out, *d_a, *d_b;

        HIP_CHECK(hipMalloc(&d_out, alloc_size));
        HIP_CHECK(hipMalloc(&d_a, alloc_size));
        HIP_CHECK(hipMalloc(&d_b, alloc_size));

        std::vector<float> a(size, 0.0f), b(size, 0.0f), cpu_out(size, 0.0f), gpu_out(size, 0.0f);
        for (size_t i = 0; i < size; i++) {
          a[i] = i + 1;
          b[i] = (i + 1) * 2;
          cpu_out[i] = (a[i] * x) + b[i];
        }

        HIP_CHECK(hipMemcpy(d_a, a.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, b.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_out, 0, alloc_size));

        vector_add_mem_layout<<<1, size, alloc_size, nullptr>>>(d_out, d_a, d_b, x, size);

        HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, alloc_size, hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));

        for (size_t i = 0; i < size; i++) {
          INFO("size: " << size << " index: " << i << " calc: " << a[i] << " * " << x << " + "
                        << b[i] << " = " << cpu_out[i] << ", " << gpu_out[i]);
          REQUIRE(cpu_out[i] == gpu_out[i]);
        }
      }
    }
  }
}
