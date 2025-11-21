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
 * @addtogroup shfl_down shfl_down
 * @{
 * @ingroup DeviceLanguageTest
 * `T __shfl_down(T var, unsigned int lane_delta, int width = warpSize)` -
 * Contains unit test for warp shfl_down function
 */

namespace cg = cooperative_groups;

template <typename T> __global__ void shfl_down(T* const out, const T* const in,
                                                const uint64_t* const active_masks,
                                                const unsigned int* const deltas, const int width) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  const auto block = cg::this_thread_block();
  T var = in[grid.thread_rank()];
  out[grid.thread_rank()] = __shfl_down(var, deltas[block.thread_rank() % width], width);
}

template <typename T> class WarpShflDown : public WarpShflTest<WarpShflDown<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, T* const input_dev, const uint64_t* const active_masks) {
    width_ = generate_width(this->warp_size_);
    INFO("Width: " << width_);
    const auto alloc_size = width_ * sizeof(unsigned int);
    LinearAllocGuard<unsigned int> deltas_dev(LinearAllocs::hipMalloc, alloc_size);
    deltas_.resize(width_);
    std::generate(deltas_.begin(), deltas_.end(),
                  [this] { return GenerateRandomInteger(0u, static_cast<unsigned int>(width_)); });
    HIP_CHECK(hipMemcpy(deltas_dev.ptr(), deltas_.data(), alloc_size, hipMemcpyHostToDevice));
    shfl_down<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, input_dev, active_masks,
                                                                 deltas_dev.ptr(), width_);
  }

  void validate(const T* const arr, const T* const input) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this, &input](unsigned int i) -> std::optional<T> {
      const int rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto rank_in_partition = rank_in_block % width_;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
                            rank_in_block / this->warp_size_;
      const unsigned int delta = deltas_[rank_in_partition] % width_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const int target = rank_in_block % width_ + delta;
      if (!active_mask.test(rank_in_warp) ||
          (target < width_ && !active_mask.test(rank_in_warp + delta)) ||
          (target < width_ && rank_in_block + delta >= this->grid_.threads_in_block_count_)) {
        return std::nullopt;
      }

      return (target >= width_ ? input[i] : input[i + delta]);
    });
  };

 private:
  std::vector<unsigned int> deltas_;
  int width_;
};

/**
 * Test Description
 * ------------------------
 *  - Validates the warp shuffle down behavior for all valid width sizes {2, 4, 8, 16, 32,
 * 64(if supported)} for generated delta values. The threads are deactivated based on the
 * passed active mask. The test is run for all overloads of shfl_down.
 * Test source
 * ------------------------
 *  - unit/warp/warp_shfl_down.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Device supports warp shuffle
 */
TEMPLATE_TEST_CASE("Unit_Warp_Shfl_Down_Positive_Basic", "", int, unsigned int, long, unsigned long,
                   long long, unsigned long long, float, double, __half, __half2) {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpShuffle) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Shuffle!");
    return;
  }

  SECTION("Shfl Down with specified active mask and input values") {
    WarpShflDown<TestType>().run(false);
  }

  SECTION("Shfl Down with random active mask and input values") {
    WarpShflDown<TestType>().run(true);
  }
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
