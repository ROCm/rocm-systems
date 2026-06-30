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


#ifndef TEST_SIMPLE_CONVOLUTION_SIMPLE_CONVOLUTION_H_
#define TEST_SIMPLE_CONVOLUTION_SIMPLE_CONVOLUTION_H_

#include <map>
#include <string>
#include <vector>

#include "ctrl/test_kernel.h"

// Class implements simple_convolution kernel parameters
class simple_convolution : public TestKernel {
 public:
  // Kernel buffers IDs
  enum { INPUT_BUF_ID, LOCAL_BUF_ID, MASK_BUF_ID, KERNARG_BUF_ID, REFOUT_BUF_ID };

  // Constructor
  simple_convolution();

  // Initialize method
  void Init();

  // Return compute grid size
  uint32_t GetGridSize() const { return width_ * height_; }

  // Print output
  void PrintOutput(const void* ptr) const;

  // Return name
  std::string Name() const { return std::string("simple_convolution"); }

 private:
  // Local kernel arguments declaration
  struct kernel_args_t {
    void* arg1;
    void* arg2;
    void* arg3;
    uint32_t arg4;
    uint32_t arg41;
    uint32_t arg5;
    uint32_t arg51;
  };

  // Reference CPU implementation of Simple Convolution
  // @param output Output matrix after performing convolution
  // @param input  Input  matrix on which convolution is to be performed
  // @param mask   mask matrix using which convolution was to be performed
  // @param input_dimensions dimensions of the input matrix
  // @param mask_dimensions  dimensions of the mask matrix
  // @return bool true on success and false on failure
  bool ReferenceImplementation(uint32_t* output, const uint32_t* input, const float* mask,
                               const uint32_t width, const uint32_t height,
                               const uint32_t maskWidth, const uint32_t maskHeight);

  // Width of the Input array
  uint32_t width_;

  // Height of the Input array
  uint32_t height_;

  // Mask dimensions
  uint32_t mask_width_;

  // Mask dimensions
  uint32_t mask_height_;

  // Input data
  std::vector<uint32_t> input_data_;
  static std::vector<uint32_t> get_input_data(size_t width, size_t height);
};

#endif  // TEST_SIMPLE_CONVOLUTION_SIMPLE_CONVOLUTION_H_
