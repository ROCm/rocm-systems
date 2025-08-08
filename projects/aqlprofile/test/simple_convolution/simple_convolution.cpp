// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "simple_convolution/simple_convolution.h"

#include <assert.h>
#include <string.h>
#include <random>
#include <iostream>

#include "util/helper_funcs.h"
#include "util/test_assert.h"

#define IMAGE_SIZE 128
#define MASK_SIZE 3

std::vector<uint32_t> simple_convolution::get_input_data(size_t width_, size_t height_) {
  std::random_device dev;
  std::mt19937 rng(dev());
  // Low range to avoid floating point issues
  std::uniform_int_distribution<uint32_t> rsampler(0, INT8_MAX);

  std::vector<uint32_t> data;
  data.reserve(width_ * height_);

  for (size_t idy = 0; idy < height_; idy++)
    for (size_t idx = 0; idx < width_; idx++) data.push_back(rsampler(rng));

  return data;
}

simple_convolution::simple_convolution()
    : width_(IMAGE_SIZE), height_(IMAGE_SIZE), mask_width_(MASK_SIZE), mask_height_(MASK_SIZE) {
  assert(IsPowerOf2(width_));
  assert(IsPowerOf2(height_));
  // Ensure symmetric mask (odd number)
  assert(mask_width_ % 2 == 1);
  assert(mask_height_ % 2 == 1);

  const uint32_t input_size_bytes = width_ * height_ * sizeof(uint32_t);
  const uint32_t mask_size_bytes =
      static_cast<uint64_t>(mask_width_) * mask_height_ * sizeof(float);

  SetInDescr(KERNARG_BUF_ID, KERNARG_DES_ID, sizeof(kernel_args_t));
  SetInDescr(INPUT_BUF_ID, SYS_DES_ID, input_size_bytes);
  SetInDescr(MASK_BUF_ID, SYS_DES_ID, mask_size_bytes);
  SetOutDescr(LOCAL_BUF_ID, LOCAL_DES_ID, input_size_bytes);
  SetHostDescr(REFOUT_BUF_ID, REFOUT_DES_ID, input_size_bytes);

  input_data_ = get_input_data(width_, height_);
  TEST_ASSERT(input_data_.size() == static_cast<uint64_t>(width_) * height_);
}

void simple_convolution::Init() {
  std::clog << "simple_convolution::init :" << std::endl;

  mem_descr_t kernarg_des = GetDescr(KERNARG_BUF_ID);
  mem_descr_t input_des = GetDescr(INPUT_BUF_ID);
  mem_descr_t mask_des = GetDescr(MASK_BUF_ID);
  mem_descr_t output_des = GetDescr(LOCAL_BUF_ID);
#if 0
  printf("kernarg_des %p 0x%x\n", kernarg_des.ptr, kernarg_des.size);
  printf("input_des %p 0x%x\n", input_des.ptr, input_des.size);
  printf("mask_des %p 0x%x\n", mask_des.ptr, mask_des.size);
  printf("output_des %p 0x%x\n", output_des.ptr, output_des.size);
#endif
  uint32_t* input = reinterpret_cast<uint32_t*>(input_des.ptr);
  uint32_t* output_local = reinterpret_cast<uint32_t*>(output_des.ptr);
  float* mask = reinterpret_cast<float*>(mask_des.ptr);
  kernel_args_t* kernel_args = reinterpret_cast<kernel_args_t*>(kernarg_des.ptr);

  // Fill a blurr filter or some other filter of your choice
  const float val = 1.0f / (mask_width_ * 2.0f - 1.0f);
  for (uint32_t i = 0; i < (mask_width_ * mask_height_); i++) {
    mask[i] = 0;
  }
  for (uint32_t i = 0; i < mask_width_; i++) {
    uint32_t y = mask_height_ / 2;
    mask[y * mask_width_ + i] = val;
  }
  for (uint32_t i = 0; i < mask_height_; i++) {
    uint32_t x = mask_width_ / 2;
    mask[i * mask_width_ + x] = val;
  }

  // Print the INPUT array.
  std::clog << std::dec;
  PrintArray<uint32_t>("> Input[0]", input, width_, 1);
  PrintArray<float>("> Mask", mask, mask_width_, mask_height_);

  // Fill the kernel args
  kernel_args->arg1 = output_local;
  kernel_args->arg2 = input;
  kernel_args->arg3 = mask;
  kernel_args->arg4 = width_;
  kernel_args->arg41 = height_;
  kernel_args->arg5 = mask_width_;
  kernel_args->arg51 = mask_height_;

  // Calculate the reference output
  ReferenceImplementation(reinterpret_cast<uint32_t*>(GetRefOut()), input, mask, width_, height_,
                          mask_width_, mask_height_);
}

void simple_convolution::PrintOutput(const void* ptr) const {
  PrintArray<uint32_t>("> Output[0]", reinterpret_cast<const uint32_t*>(ptr), width_, 1);
}

bool simple_convolution::ReferenceImplementation(uint32_t* output, const uint32_t* input,
                                                 const float* mask, const uint32_t width,
                                                 const uint32_t height, const uint32_t mask_width,
                                                 const uint32_t mask_height) {
  const uint32_t vstep = (mask_width - 1) / 2;
  const uint32_t hstep = (mask_height - 1) / 2;

  // for each pixel in the input
  for (uint32_t x = 0; x < width; x++) {
    for (uint32_t y = 0; y < height; y++) {
      // find the left, right, top and bottom indices such that
      // the indices do not go beyond image boundaires
      const uint32_t left = (x < vstep) ? 0 : (x - vstep);
      const uint32_t right = ((x + vstep) >= width) ? width - 1 : (x + vstep);
      const uint32_t top = (y < hstep) ? 0 : (y - hstep);
      const uint32_t bottom = ((y + hstep) >= height) ? height - 1 : (y + hstep);

      // initializing wighted sum value
      float sum_fx = 0;
      for (uint32_t i = left; i <= right; ++i) {
        for (uint32_t j = top; j <= bottom; ++j) {
          // performing wighted sum within the mask boundaries
          uint32_t mask_idx = (j - (y - hstep)) * mask_width + (i - (x - vstep));
          uint32_t index = j * width + i;

          // to round to the nearest integer
          sum_fx += ((float)input[index] * mask[mask_idx]);
        }
      }
      output[y * width + x] = uint32_t(sum_fx + 0.5f);
    }
  }

  return true;
}
