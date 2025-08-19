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

// Wavefront sized launch
__global__ void coop_any_coal(unsigned int* data, unsigned int val, unsigned int* res) {
  auto cg = cooperative_groups::coalesced_threads();
  unsigned int tmp = data[threadIdx.x];
  res[threadIdx.x] = cg.any(tmp == val);
}

__global__ void coop_any_coal_odd_even(unsigned int* data, unsigned int even_val,
                                       unsigned int odd_val, unsigned int* res) {
  if ((threadIdx.x % 2) == 0) {
    auto cg = cooperative_groups::coalesced_threads();
    unsigned int tmp = data[threadIdx.x];
    res[threadIdx.x] = cg.any(tmp == even_val);
  } else {
    auto cg = cooperative_groups::coalesced_threads();
    unsigned int tmp = data[threadIdx.x];
    res[threadIdx.x] = cg.any(tmp == odd_val);
  }
}

TEST_CASE("Unit_coopgroups_any") {
  const size_t warp_size = getWarpSize();

  unsigned int *data, *res;

  HIP_CHECK(hipMalloc(&data, sizeof(int) * warp_size));
  HIP_CHECK(hipMalloc(&res, sizeof(int) * warp_size));

  HIP_CHECK(hipMemset(data, 0, sizeof(int) * warp_size));

  SECTION("all set") {
    constexpr unsigned int any_val = 10;
    std::vector<unsigned int> input(warp_size, any_val);
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("all set - odd/even coal") {
    constexpr unsigned int odd_val = 10, even_val = 20;
    std::vector<unsigned int> input;
    input.reserve(warp_size);
    for (size_t i = 0; i < warp_size; i++) {
      if ((i % 2) == 0) {
        input.push_back(even_val);
      } else {
        input.push_back(odd_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal_odd_even<<<1, warp_size>>>(data, even_val, odd_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("None set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input(warp_size, any_val);
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal<<<1, warp_size>>>(data, 10, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("None set - odd/even coal") {
    constexpr unsigned int odd_val = 10, even_val = 20;
    std::vector<unsigned int> input;
    input.reserve(warp_size);
    for (size_t i = 0; i < warp_size; i++) {
      if ((i % 2) == 0) {
        input.push_back(even_val);
      } else {
        input.push_back(odd_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal_odd_even<<<1, warp_size>>>(data, odd_val, even_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("Alternate set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i++) {
      if ((i % 2) == 0) {
        input.push_back(any_val + 1);
      } else {
        input.push_back(any_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("first half set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i++) {
      if (i < (warp_size / 2)) {
        input.push_back(any_val);
      } else {
        input.push_back(any_val + 1);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("last half set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i++) {
      if (i < (warp_size / 2)) {
        input.push_back(any_val + 1);
      } else {
        input.push_back(any_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("First set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input(warp_size, any_val + 1);
    input[0] = any_val;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("First set - odd/even coal") {
    constexpr unsigned int odd_val = 10, even_val = 20;
    std::vector<unsigned int> input(warp_size, 0);
    input[0] = even_val;
    input[1] = odd_val;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal_odd_even<<<1, warp_size>>>(data, even_val, odd_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("Last set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input(warp_size, any_val + 1);
    input[warp_size - 1] = any_val;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("Last set - odd/even coal") {
    constexpr unsigned int odd_val = 10, even_val = 20;
    std::vector<unsigned int> input(warp_size, 0);
    input[warp_size - 2] = even_val;
    input[warp_size - 1] = odd_val;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_any_coal_odd_even<<<1, warp_size>>>(data, even_val, odd_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(res));
}

__global__ void coop_all_coal(unsigned int* data, unsigned int val, unsigned int* res) {
  auto cg = cooperative_groups::coalesced_threads();
  unsigned int tmp = data[threadIdx.x];
  res[threadIdx.x] = cg.all(tmp == val);
}

__global__ void coop_all_coal_odd_even(unsigned int* data, unsigned int even_val,
                                       unsigned int odd_val, unsigned int* res) {
  if ((threadIdx.x % 2) == 0) {
    auto cg = cooperative_groups::coalesced_threads();
    unsigned int tmp = data[threadIdx.x];
    res[threadIdx.x] = cg.all(tmp == even_val);
  } else {
    auto cg = cooperative_groups::coalesced_threads();
    unsigned int tmp = data[threadIdx.x];
    res[threadIdx.x] = cg.all(tmp == odd_val);
  }
}

TEST_CASE("Unit_coopgroups_coal_all") {
  const size_t warp_size = getWarpSize();

  unsigned int *data, *res;

  HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * warp_size));
  HIP_CHECK(hipMalloc(&res, sizeof(unsigned int) * warp_size));

  HIP_CHECK(hipMemset(data, 0, sizeof(unsigned int) * warp_size));

  SECTION("All set - all in sync") {
    constexpr unsigned int any_val = 10;
    std::vector<unsigned int> input(warp_size, any_val);
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("All set - odd/even in sync") {
    constexpr unsigned int even_val = 10, odd_val = 20;
    std::vector<unsigned int> input;
    input.reserve(warp_size);
    for (size_t i = 0; i < warp_size; i++) {
      if ((i % 2) == 0) {
        input.push_back(even_val);
      } else {
        input.push_back(odd_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal_odd_even<<<1, warp_size>>>(data, even_val, odd_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 1);
    }
  }

  SECTION("None set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input(warp_size, any_val);
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal<<<1, warp_size>>>(data, 10, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("None set - odd/even sync") {
    constexpr unsigned int even_val = 10, odd_val = 20;
    std::vector<unsigned int> input;
    input.reserve(warp_size);
    for (size_t i = 0; i < warp_size; i++) {
      if ((i % 2) == 0) {
        input.push_back(odd_val);
      } else {
        input.push_back(even_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal_odd_even<<<1, warp_size>>>(data, even_val, odd_val, res);
    std::vector<unsigned int> output(warp_size, 1);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("Alternate set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i++) {
      if ((i % 2) == 0) {
        input.push_back(any_val + 1);
      } else {
        input.push_back(any_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("first half set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i++) {
      if (i < (warp_size / 2)) {
        input.push_back(any_val);
      } else {
        input.push_back(any_val + 1);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("last half set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i++) {
      if (i < (warp_size / 2)) {
        input.push_back(any_val + 1);
      } else {
        input.push_back(any_val);
      }
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("First set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input(warp_size, any_val + 1);
    input[0] = any_val;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  SECTION("Last set") {
    constexpr unsigned int any_val = 20;
    std::vector<unsigned int> input(warp_size, any_val + 1);
    input[warp_size - 1] = any_val;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_all_coal<<<1, warp_size>>>(data, any_val, res);
    std::vector<unsigned int> output(warp_size, 0);
    HIP_CHECK(
        hipMemcpy(output.data(), res, sizeof(unsigned int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == 0);
    }
  }

  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(res));
}

__global__ void coop_match_any_coal(unsigned int* data, unsigned long long* res) {
  auto cg = cooperative_groups::coalesced_threads();
  unsigned int tmp = data[threadIdx.x];
  res[threadIdx.x] = cg.match_any(tmp);
}

__global__ void coop_match_any_coal_odd_even(unsigned int* data, unsigned long long* res) {
  unsigned int tmp = data[threadIdx.x];
  if ((threadIdx.x % 2) == 0) {
    auto cg = cooperative_groups::coalesced_threads();
    res[threadIdx.x] = cg.match_any(tmp);
  } else {
    auto cg = cooperative_groups::coalesced_threads();
    res[threadIdx.x] = cg.match_any(tmp);
  }
}

TEST_CASE("Unit_coopgroups_match_any_coal") {
  const size_t warp_size = getWarpSize();

  unsigned int* data;
  unsigned long long* res;

  HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * warp_size));
  HIP_CHECK(hipMalloc(&res, sizeof(unsigned long long) * warp_size));

  HIP_CHECK(hipMemset(data, 0, sizeof(unsigned int) * warp_size));
  HIP_CHECK(hipMemset(res, 0, sizeof(unsigned long long) * warp_size));

  SECTION("all set") {
    constexpr unsigned int any_val = 10;
    std::vector<unsigned int> input(warp_size, any_val);
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_match_any_coal<<<1, warp_size>>>(data, res);
    std::vector<unsigned long long> output(warp_size, 0);
    HIP_CHECK(hipMemcpy(output.data(), res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    auto all_set_mask = warp_size == 32 ? 0xFFFF'FFFFull : 0xFFFF'FFFF'FFFF'FFFFull;
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == all_set_mask);
    }
  }

  SECTION("all set - odd/even") {
    constexpr unsigned int even_val = 10, odd_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i += 2) {
      input.push_back(even_val);
      input.push_back(odd_val);
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_match_any_coal_odd_even<<<1, warp_size>>>(data, res);
    std::vector<unsigned long long> output(warp_size, 0);
    HIP_CHECK(hipMemcpy(output.data(), res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    auto all_set_mask = warp_size == 32 ? 0xFFFFull : 0xFFFF'FFFFull;
    for (size_t i = 0; i < output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << output[i]);
      REQUIRE(output[i] == all_set_mask);
    }
  }

  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(res));
}

__global__ void coop_match_all_coal(unsigned int* data, unsigned long long* res, int* pred_res) {
  auto cg = cooperative_groups::coalesced_threads();
  unsigned int tmp = data[threadIdx.x];
  res[threadIdx.x] = cg.match_all(tmp, pred_res[threadIdx.x]);
}

__global__ void coop_match_all_coal_odd_even(unsigned int* data, unsigned long long* res,
                                             int* pred_res) {
  unsigned int tmp = data[threadIdx.x];
  if ((threadIdx.x % 2) == 0) {
    auto cg = cooperative_groups::coalesced_threads();
    res[threadIdx.x] = cg.match_all(tmp, pred_res[threadIdx.x]);
  } else {
    auto cg = cooperative_groups::coalesced_threads();
    res[threadIdx.x] = cg.match_all(tmp, pred_res[threadIdx.x]);
  }
}

TEST_CASE("Unit_coopgroups_match_all_coal") {
  const size_t warp_size = getWarpSize();

  unsigned int* data;
  unsigned long long* res;
  int* pred_res;

  HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * warp_size));
  HIP_CHECK(hipMalloc(&res, sizeof(unsigned long long) * warp_size));
  HIP_CHECK(hipMalloc(&pred_res, sizeof(int) * warp_size));

  HIP_CHECK(hipMemset(data, 0, sizeof(unsigned int) * warp_size));
  HIP_CHECK(hipMemset(res, 0, sizeof(unsigned long long) * warp_size));

  SECTION("all set") {
    constexpr unsigned int any_val = 10;
    std::vector<unsigned int> input(warp_size, any_val);
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_match_all_coal<<<1, warp_size>>>(data, res, pred_res);
    std::vector<unsigned long long> mask_output(warp_size, 0);
    std::vector<int> pred_output(warp_size, 0);
    HIP_CHECK(hipMemcpy(mask_output.data(), res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pred_output.data(), pred_res, sizeof(int) * warp_size, hipMemcpyDeviceToHost));
    auto all_set_mask = warp_size == 32 ? 0xFFFF'FFFFull : 0xFFFF'FFFF'FFFF'FFFFull;
    for (size_t i = 0; i < mask_output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << mask_output[i]
                                  << " pred: " << pred_output[i]);
      REQUIRE(mask_output[i] == all_set_mask);
      REQUIRE(pred_output[i] == 1);
    }
  }

  SECTION("one mismatch") {
    constexpr unsigned int any_val = 10;
    std::vector<unsigned int> input(warp_size, any_val);
    input[warp_size - 1]--;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_match_all_coal<<<1, warp_size>>>(data, res, pred_res);
    std::vector<unsigned long long> mask_output(warp_size, 0);
    std::vector<int> pred_output(warp_size, 0);
    HIP_CHECK(hipMemcpy(mask_output.data(), res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pred_output.data(), pred_res, sizeof(int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < mask_output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << mask_output[i]
                                  << " pred: " << pred_output[i]);
      REQUIRE(mask_output[i] == 0);
      REQUIRE(pred_output[i] == 0);
    }
  }

  SECTION("all set - odd/even") {
    constexpr unsigned int even_val = 10, odd_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i += 2) {
      input.push_back(even_val);
      input.push_back(odd_val);
    }
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_match_all_coal_odd_even<<<1, warp_size>>>(data, res, pred_res);
    std::vector<unsigned long long> mask_output(warp_size, 0);
    std::vector<int> pred_output(warp_size, 0);
    HIP_CHECK(hipMemcpy(mask_output.data(), res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pred_output.data(), pred_res, sizeof(int) * warp_size, hipMemcpyDeviceToHost));
    auto all_set_mask = warp_size == 32 ? 0xFFFFull : 0xFFFF'FFFFull;
    for (size_t i = 0; i < mask_output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << mask_output[i]
                                  << " pred: " << pred_output[i]);
      REQUIRE(mask_output[i] == all_set_mask);
      REQUIRE(pred_output[i] == 1);
    }
  }

  SECTION("one mismatch - odd/even") {
    constexpr unsigned int even_val = 10, odd_val = 20;
    std::vector<unsigned int> input;
    for (size_t i = 0; i < warp_size; i += 2) {
      input.push_back(even_val);
      input.push_back(odd_val);
    }
    input[warp_size - 1]--;
    input[warp_size - 2]--;
    HIP_CHECK(
        hipMemcpy(data, input.data(), sizeof(unsigned int) * warp_size, hipMemcpyHostToDevice));
    coop_match_all_coal_odd_even<<<1, warp_size>>>(data, res, pred_res);
    std::vector<unsigned long long> mask_output(warp_size, 0);
    std::vector<int> pred_output(warp_size, 0);
    HIP_CHECK(hipMemcpy(mask_output.data(), res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pred_output.data(), pred_res, sizeof(int) * warp_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < mask_output.size(); i++) {
      INFO("Checking for index: " << i << " output: " << mask_output[i]
                                  << " pred: " << pred_output[i]);
      REQUIRE(mask_output[i] == 0);
      REQUIRE(pred_output[i] == 0);
    }
  }

  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(res));
  HIP_CHECK(hipFree(pred_res));
}
