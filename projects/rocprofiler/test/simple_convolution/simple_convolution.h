/******************************************************************************

Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

#ifndef TEST_SIMPLE_CONVOLUTION_SIMPLE_CONVOLUTION_H_
#define TEST_SIMPLE_CONVOLUTION_SIMPLE_CONVOLUTION_H_

#include <map>
#include <vector>

#include "ctrl/test_kernel.h"

// Class implements SimpleConvolution kernel parameters
class SimpleConvolution : public TestKernel {
 public:
  // Kernel buffers IDs
  enum { INPUT_BUF_ID, LOCAL_BUF_ID, MASK_BUF_ID, KERNARG_BUF_ID, REFOUT_BUF_ID };

  // Constructor
  SimpleConvolution();

  // Initialize method
  void Init();

  // Return compute grid size
  uint32_t GetGridSize() const { return width_ * height_; }

  // Print output
  void PrintOutput(const void* ptr) const;

  // Return name
  std::string Name() const { return std::string("SimpleConvolution"); }

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

  // Randomize input data
  unsigned randomize_seed_;

  // Input data
  static const uint32_t input_data_[];
};

#endif  // TEST_SIMPLE_CONVOLUTION_SIMPLE_CONVOLUTION_H_
