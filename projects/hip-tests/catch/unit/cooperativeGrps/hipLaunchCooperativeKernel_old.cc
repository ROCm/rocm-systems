/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

namespace cg = cooperative_groups;

static constexpr size_t kBufferLen = 1024 * 1024;

__global__ void test_gws(int* buf, size_t buf_size, unsigned long long* tmp_buf,
                         unsigned long long* result) {
  extern __shared__ unsigned long long tmp[];

  cg::thread_block tb = cg::this_thread_block();
  cg::grid_group gg = cg::this_grid();

  const auto tid = gg.thread_rank();
  const auto stride = gg.size();
  const auto local_tid = tb.thread_rank();
  const auto wid = blockIdx.x;
  const auto workgroup_size = tb.size();
  const auto grid_size = gridDim.x;

  unsigned long long sum = 0;
  for (size_t i = tid; i < buf_size; i += stride) {
    sum += buf[i];
  }
  tmp[local_tid] = sum;
  tb.sync();

  if (local_tid == 0) {
    sum = 0;
    for (size_t i = 0; i < workgroup_size; i++) {
      sum += tmp[i];
    }
    tmp_buf[wid] = sum;
  }
  gg.sync();

  if (tid < grid_size) {
    atomicAdd(result, tmp_buf[tid]);
  }
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Basic") {
  // Use default device for validating the test
  int device;
  HIP_CHECK(hipGetDevice(&device));

  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  size_t buffer_size = kBufferLen * sizeof(int);

  int* A_h = nullptr;
  int* A_d = nullptr;
  unsigned long long* B_d = nullptr;
  unsigned long long* C_d = nullptr;
  hipStream_t stream;

  A_h = reinterpret_cast<int*>(malloc(buffer_size));
  for (uint32_t i = 0; i < kBufferLen; ++i) {
    A_h[i] = static_cast<int>(i);
  }

  HIP_CHECK(hipMalloc(&A_d, buffer_size));
  HIP_CHECK(hipMemcpy(A_d, A_h, buffer_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipHostMalloc(&C_d, sizeof(unsigned long long)));

  HIPCHECK(hipStreamCreate(&stream));

  uint32_t workgroup = GENERATE(32, 64, 128, 256);

  dim3 dimBlock = dim3(workgroup);
  dim3 dimGrid = dim3(1);

  int numBlocks = 0;
  // Calculate the device occupancy to know how many blocks can be run concurrently
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, test_gws,
                                                         dimBlock.x * dimBlock.y * dimBlock.z,
                                                         dimBlock.x * sizeof(unsigned long long)));

  dimGrid.x = device_properties.multiProcessorCount * std::min(numBlocks, 32);
  HIP_CHECK(hipMalloc(&B_d, dimGrid.x * sizeof(unsigned long long)));

  void* params[4];
  params[0] = (void*)&A_d;
  params[1] = (void*)&kBufferLen;
  params[2] = (void*)&B_d;
  params[3] = (void*)&C_d;

  INFO("Testing with grid size = " << dimGrid.x << " and block size = " << dimBlock.x << "\n");
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_gws), dimGrid, dimBlock, params,
                                       dimBlock.x * sizeof(unsigned long long), stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(((unsigned long long)*C_d) ==
          (((unsigned long long)(kBufferLen) * (kBufferLen - 1)) / 2));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(C_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}
