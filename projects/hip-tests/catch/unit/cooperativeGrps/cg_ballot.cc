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

__global__ void coop_ballot_coal_alternate(int* data, unsigned long long* d_even_mask,
                                           unsigned long long* d_odd_mask) {
  int tid = threadIdx.x;
  auto val = data[tid];

  if ((tid % 2) == 0) {
    auto cg = cooperative_groups::coalesced_threads();
    auto cg_ballot_res = cg.ballot(val == 10);
    if (tid == 0) {
      *d_even_mask = cg_ballot_res;
    }
  } else {
    auto cg = cooperative_groups::coalesced_threads();
    auto cg_ballot_res = cg.ballot(val == 10);
    if (tid == 1) {
      *d_odd_mask = cg_ballot_res;
    }
  }
}

TEST_CASE("Unit_coopgroups_ballot") {
  const size_t warp_size = getWarpSize();
  std::vector<int> input;
  input.reserve(warp_size);
  // Push 10 - 10 - 20 - 20 in vector
  for (size_t i = 0; i < warp_size; i += 4) {
    input.push_back(10);
    input.push_back(10);
    input.push_back(20);
    input.push_back(20);
  }

  int* data;
  HIP_CHECK(hipMalloc(&data, sizeof(int) * input.size()));
  HIP_CHECK(hipMemcpy(data, input.data(), sizeof(int) * input.size(), hipMemcpyHostToDevice));

  unsigned long long *d_even_mask, *d_odd_mask, even_mask = 0, odd_mask = 0l;
  HIP_CHECK(hipMalloc(&d_even_mask, sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&d_odd_mask, sizeof(unsigned long long)));

  coop_ballot_coal_alternate<<<1, warp_size>>>(data, d_even_mask, d_odd_mask);
  HIP_CHECK(hipMemcpy(&even_mask, d_even_mask, sizeof(unsigned long long), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&odd_mask, d_odd_mask, sizeof(unsigned long long), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(data));
  HIP_CHECK(hipFree(d_even_mask));
  HIP_CHECK(hipFree(d_odd_mask));

  if (warp_size == 32) {
    REQUIRE(even_mask == 0x5555);
    REQUIRE(odd_mask == 0x5555);
  } else {
    REQUIRE(even_mask == 0x55555555);
    REQUIRE(odd_mask == 0x55555555);
  }
}

static __global__ void coop_ballot_binary_part(int* data, unsigned long long* res) {
  int tid = threadIdx.x;
  auto val = data[tid];
  auto part =
      cooperative_groups::binary_partition(cooperative_groups::coalesced_threads(), (tid % 5) == 0);
  // Only tid % 5 lane will vote yes
  res[tid] = part.ballot((val % 5) == 0);
}

static __global__ void coop_ballot_binary_part_3_5_7(int* data, unsigned long long* res) {
  int tid = threadIdx.x;
  auto val = data[tid];
  auto part = cooperative_groups::binary_partition(cooperative_groups::coalesced_threads(),
                                                   (tid % 3) == 0 || (tid % 5) == 0);
  res[tid] = part.ballot((val % 7) == 0);
}

TEST_CASE("Unit_binary_part_ballot") {
  const size_t warp_size = getWarpSize();
  std::vector<int> input;
  input.reserve(warp_size);
  for (size_t i = 0; i < warp_size; i++) {
    input.push_back(i);
  }

  int* data;
  HIP_CHECK(hipMalloc(&data, sizeof(int) * input.size()));
  HIP_CHECK(hipMemcpy(data, input.data(), sizeof(int) * input.size(), hipMemcpyHostToDevice));

  unsigned long long* d_res;
  HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned long long) * warp_size));

  SECTION("Basic partition of 5th lane") {
    std::vector<unsigned long long> res(warp_size, 1);
    coop_ballot_binary_part<<<1, warp_size>>>(data, d_res);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    unsigned long long expected_result = warp_size == 32 ? 0x7F : 0x1FFF;
    for (size_t i = 0; i < res.size(); i++) {
      INFO("Index: " << i << " result: " << res[i]);
      REQUIRE((i % 5 == 0 ? res[i] == expected_result : res[i] == 0));
    }
  }

  SECTION("Lanes index multiple of 2 and 3 voting on value % 5") {
    std::vector<unsigned long long> res(warp_size, 1);

    std::vector<unsigned long long> expected;
    expected.reserve(warp_size);
    /* This will be fun to explain
     * So lane which are multiple of 3 or 5 will call ballot on values which are multiple of 7
     * lane:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
     * active:x     x   x x     x  x     x        x        x     x  x        x  x     x        x
     * res1:  1     0   0 0     0  0     0        0        0     0  1        0  0     0        0
     * res2:    0 0   0     1 0       0     0  1     0  0     0        0  0        0     1  0     0
     * Active lane here is how the warp will be partitioned into i.e. tid % 3 || tid % 5, so active
     * in one partition and ~active in other. res1 is what active part will report as result and
     * res2 will be reported by ~active partition.
     */
    const unsigned long long res1 = warp_size == 32 ? 1025 : 537986049;
    const unsigned long long res2 = warp_size == 32 ? 16520 : 603996296;
    for (size_t i = 0; i < warp_size; i++) {
      if ((i % 3) == 0 || (i % 5) == 0) {
        expected.push_back(res1);
      } else {
        expected.push_back(res2);
      }
    }
    coop_ballot_binary_part_3_5_7<<<1, warp_size>>>(data, d_res);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned long long) * warp_size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < res.size(); i++) {
      INFO("index: " << i << "expected: " << expected[i] << " got: " << res[i]);
      CHECK(expected[i] == res[i]);
    }
  }

  HIP_CHECK(hipFree(d_res));
  HIP_CHECK(hipFree(data));
}