/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "warp_common.hh"

#include <cpu_grid.h>
#include <resource_guards.hh>
#include <utils.hh>

inline uint64_t get_predicate_mask(unsigned int test_case, unsigned int warp_size) {
  uint64_t predicate_mask = 0;
  switch (test_case) {
    case 0:  // no thread
      predicate_mask = 0;
      break;
    case 1:  // 1st thread
      predicate_mask = 1;
      break;
    case 2:  // last thread
      predicate_mask = static_cast<uint64_t>(1) << (warp_size - 1);
      break;
    case 3:  // all threads
      predicate_mask = 0xFFFFFFFFFFFFFFFF;
      break;
    default:  // random
      static std::mt19937_64 mt(test_case);
      std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
      predicate_mask = dist(mt);
  }
  return predicate_mask;
}

inline uint64_t get_active_predicate(uint64_t predicate, size_t partition_size) {
  uint64_t active_predicate = predicate;
  for (int i = partition_size; i < 64; i++) {
    active_predicate &= ~(static_cast<uint64_t>(1) << i);
  }
  return active_predicate;
}

template <typename Derived, typename T> class WarpVoteTest {
 public:
  WarpVoteTest() : warp_size_{get_warp_size()} {}

  void run(bool random = false) {
    const auto blocks = GenerateBlockDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    const auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    grid_ = CPUGrid(blocks, threads);

    const auto alloc_size = grid_.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);
    HIP_CHECK(hipMemset(arr_dev.ptr(), 0, alloc_size));

    warps_in_block_ = (grid_.threads_in_block_count_ + warp_size_ - 1) / warp_size_;
    const auto warps_in_grid = warps_in_block_ * grid_.block_count_;
    LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                                warps_in_grid * sizeof(uint64_t));
    active_masks_.resize(warps_in_grid);

    if (random) {
      std::generate(active_masks_.begin(), active_masks_.end(), [] {
        return GenerateRandomInteger<unsigned long long>(0ul, std::numeric_limits<uint64_t>().max());
      });
    } else {
      unsigned long long int i = 0;
      std::generate(active_masks_.begin(), active_masks_.end(),
                    [this, &i]() { return get_active_mask(i++, warp_size_); });
    }

    HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks_.data(),
                        warps_in_grid * sizeof(uint64_t), hipMemcpyHostToDevice));
    cast_to_derived().launch_kernel(arr_dev.ptr(), active_masks_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    cast_to_derived().validate(arr.ptr());
  }

 private:
  int get_warp_size() const {
    int current_dev = -1;
    HIP_CHECK(hipGetDevice(&current_dev));
    int warp_size = 0u;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
    return warp_size;
  }

  Derived& cast_to_derived() { return reinterpret_cast<Derived&>(*this); }

 protected:
  const int warp_size_;
  CPUGrid grid_;
  unsigned int warps_in_block_;
  std::vector<uint64_t> active_masks_;
};