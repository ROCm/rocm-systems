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

#include "warp_shfl_common.hh"

#include <bitset>

/**
 * @addtogroup shfl_xor shfl_xor
 * @{
 * @ingroup DeviceLanguageTest
 * `T __shfl_xor(T var, int lane_mask, int width = warpSize)` -
 * Contains unit test for warp shfl_xor function
 */

namespace cg = cooperative_groups;

template <typename T> __global__ void shfl_xor(T* const out, const T* const in,
                                               const uint64_t* const active_masks,
                                               const int lane_mask, const int width) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  T var = in[grid.thread_rank()];
  out[grid.thread_rank()] = __shfl_xor(var, lane_mask, width);
}

template <typename T> class WarpShflXOR : public WarpShflTest<WarpShflXOR<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, T* const input_dev, const uint64_t* const active_masks) {
    width_ = generate_width(this->warp_size_);
    INFO("Width: " << width_);
    lane_mask_ = GENERATE_COPY(range(0, this->warp_size_));
    INFO("Lane mask: " << lane_mask_);
    shfl_xor<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, input_dev, active_masks,
                                                                lane_mask_, width_);
  }

  void validate(const T* const arr, const T* const input) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this, &input](unsigned int i) -> std::optional<T> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const int warp_target = rank_in_warp ^ this->lane_mask_;
      const int target_offset = warp_target - rank_in_warp;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
                            rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const auto target_partition = warp_target / width_;
      const auto partition_rank = rank_in_warp / width_;
      if (!active_mask.test(rank_in_warp) ||
          (target_partition <= partition_rank && !active_mask.test(rank_in_warp + target_offset)) ||
          (target_partition <= partition_rank &&
           rank_in_block + target_offset >= this->grid_.threads_in_block_count_)) {
        return std::nullopt;
      }

      return target_partition > partition_rank ? input[i] : input[i + target_offset];
    });
  };

 private:
  int lane_mask_;
  int width_;
};

/**
 * Test Description
 * ------------------------
 *  - Validates the warp shuffle xor behavior for all valid width sizes {2, 4, 8, 16, 32,
 * 64(if supported)} for mask values of [0, width). The threads are deactivated based on the
 * passed active mask. The test is run for all overloads of shfl_xor.
 * Test source
 * ------------------------
 *  - unit/warp/warp_shfl_xor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Device supports warp shuffle
 */
TEMPLATE_TEST_CASE("Unit_Warp_Shfl_XOR_Positive_Basic", "", int, unsigned int, long, unsigned long,
                   long long, unsigned long long, float, double, __half, __half2) {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpShuffle) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Shuffle!");
    return;
  }

  SECTION("Shfl Xor with specified active mask and input values") {
    WarpShflXOR<TestType>().run(false);
  }

  SECTION("Shfl Xor with random active mask and input values") {
    WarpShflXOR<TestType>().run(true);
  }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
