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

static bool operator==(__half x, __half y) {
  // __heq doesn't have a __host__ version
  return static_cast<__half_raw>(x).x == static_cast<__half_raw>(y).x;
}
static bool operator!=(__half x, __half y) { return static_cast<__half_raw>(x).x != static_cast<__half_raw>(y).x; }

static bool operator==(__half2 x, __half2 y) { return __hbeq2(x, y); }
static bool operator!=(__half2 x, __half2 y) { return !(__hbeq2(x, y)); }

template <typename Derived, typename T> class WarpShflTest {
 public:
  WarpShflTest() : warp_size_{get_warp_size()} {}

  void run(bool random = false) {
    const auto blocks = GenerateBlockDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    const auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    grid_ = CPUGrid(blocks, threads);

    const auto alloc_size = grid_.thread_count_ * sizeof(T);
    LinearAllocGuard<T> input_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> input(LinearAllocs::hipHostMalloc, alloc_size);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);
    HIP_CHECK(hipMemset(arr_dev.ptr(), 0, alloc_size));

    warps_in_block_ = (grid_.threads_in_block_count_ + warp_size_ - 1) / warp_size_;
    const auto warps_in_grid = warps_in_block_ * grid_.block_count_;
    LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                                warps_in_grid * sizeof(uint64_t));
    active_masks_.resize(warps_in_grid);

    generate_input(input.ptr(), random);

    HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks_.data(),
                        warps_in_grid * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(input_dev.ptr(), input.ptr(), alloc_size, hipMemcpyHostToDevice));
    cast_to_derived().launch_kernel(arr_dev.ptr(), input_dev.ptr(), active_masks_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    cast_to_derived().validate(arr.ptr(), input.ptr());
  }

 private:
  int get_warp_size() const {
    int current_dev = -1;
    HIP_CHECK(hipGetDevice(&current_dev));
    int warp_size = 0u;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
    return warp_size;
  }

  void generate_input(T* input, bool random) {
    if (random) {
      std::generate(active_masks_.begin(), active_masks_.end(), [] {
        return GenerateRandomInteger<unsigned long long>(0ul, std::numeric_limits<uint64_t>().max());
      });

      if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
        std::generate_n(input, grid_.thread_count_, [] {
          return static_cast<T>(
              GenerateRandomReal(std::numeric_limits<T>().min(), std::numeric_limits<T>().max()));
        });
      } else if constexpr (std::is_same_v<__half, T>) {
        std::generate_n(input, grid_.thread_count_, [] {
          return __float2half(GenerateRandomReal(std::numeric_limits<float>().min(),
                                                 std::numeric_limits<float>().max()));
        });
      } else if constexpr (std::is_same_v<__half2, T>) {
        std::generate_n(input, grid_.thread_count_, [] {
          return __float2half2_rn(GenerateRandomReal(std::numeric_limits<float>().min(),
                                                     std::numeric_limits<float>().max()));
        });
      } else {
        std::generate_n(input, grid_.thread_count_, [] {
          return static_cast<T>(GenerateRandomInteger(std::numeric_limits<T>().min(),
                                                      std::numeric_limits<T>().max()));
        });
      }
    } else {
      unsigned long long int i = 0;
      std::generate(active_masks_.begin(), active_masks_.end(),
                    [this, &i]() { return get_active_mask(i++, warp_size_); });

      i = 0;
      std::generate_n(input, grid_.thread_count_, [&i]() {
        if (static_cast<T>(i) > std::numeric_limits<T>().max())
          i = 0;
        else
          i++;
        return static_cast<T>(i);
      });
    }
  }

  Derived& cast_to_derived() { return reinterpret_cast<Derived&>(*this); }

 protected:
  const int warp_size_;
  CPUGrid grid_;
  unsigned int warps_in_block_;
  std::vector<uint64_t> active_masks_;
};
