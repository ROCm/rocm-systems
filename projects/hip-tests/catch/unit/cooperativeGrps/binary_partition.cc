/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <vector>

static __global__ void binary_part_odd_even_val(int* data, int* res) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::coalesced_threads();

  int val = data[block.thread_rank()];
  auto part = cooperative_groups::binary_partition(tile, (val & 1));
  res[block.thread_rank()] = part.any(val % 2);
}

static __global__ void binary_part_tiled_odd_even_val(int* data, int* odd_res, int* even_res) {
  auto block = cooperative_groups::this_thread_block();
  auto tile16 = cooperative_groups::tiled_partition<16>(block);

  int val = data[block.thread_rank()];
  auto part = cooperative_groups::binary_partition(tile16, (val & 1));
  if (val & 1) {
    *odd_res = part.all((val % 2) != 0);
  } else {
    *even_res = part.all((val % 2) == 0);
  }
}

TEST_CASE("Unit_cg_binary_part") {
  const size_t warp_size = getWarpSize();

  int *data, *odd_res, *even_res;

  HIP_CHECK(hipMalloc(&data, sizeof(int) * warp_size));
  HIP_CHECK(hipMalloc(&odd_res, sizeof(int) * warp_size));
  HIP_CHECK(hipMalloc(&even_res, sizeof(int) * warp_size));

  HIP_CHECK(hipMemset(data, 0, sizeof(int) * warp_size));

  SECTION("odd_even part") {
    std::vector<int> input;
    std::vector<int> output(warp_size, -1);

    auto& res = odd_res;
    input.reserve(warp_size);
    for (size_t i = 0; i < warp_size; i++) {
      if (i < 16)
        input.push_back(10);
      else
        input.push_back(11);
    }

    HIP_CHECK(hipMemcpy(data, input.data(), sizeof(int) * input.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(res, output.data(), sizeof(int) * output.size(), hipMemcpyHostToDevice));

    binary_part_odd_even_val<<<1, warp_size>>>(data, res);
    HIP_CHECK(hipMemcpy(output.data(), res, sizeof(int) * output.size(), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < warp_size; i++) {
      if (i < 16) {
        INFO("Output <16, index: " << i << " output: " << output[i]);
        REQUIRE(output[i] == 0);
      } else {
        INFO("Output >=16, index: " << i << " output: " << output[i]);
        REQUIRE(output[i] == 1);
      }
    }
  }

  SECTION("tiled odd/even part") {
    std::vector<int> input;
    input.reserve(warp_size);
    for (size_t i = 0; i < warp_size; i++) {
      input.push_back(i + 1);
    }

    HIP_CHECK(hipMemcpy(data, input.data(), sizeof(int) * input.size(), hipMemcpyHostToDevice));
    binary_part_tiled_odd_even_val<<<1, warp_size>>>(data, odd_res, even_res);
    int odd_output, even_output;
    HIP_CHECK(hipMemcpy(&odd_output, odd_res, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&even_output, even_res, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(odd_output == 1);
    REQUIRE(even_output == 1);
  }

  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(odd_res));
  HIP_CHECK(hipFree(even_res));
}