/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
*******************************************************************************/

#ifndef TEST_DUMMY_KERNEL_DUMMY_KERNEL_H_
#define TEST_DUMMY_KERNEL_DUMMY_KERNEL_H_

#include <map>
#include <vector>

#include "ctrl/test_kernel.h"

// Class implements DummyKernel kernel parameters
class DummyKernel : public TestKernel {
 public:
  // Kernel buffers IDs
  enum { KERNARG_BUF_ID, LOCAL_BUF_ID };

  // Constructor
  DummyKernel() : width_(64), height_(64) {
    SetInDescr(KERNARG_BUF_ID, KERNARG_DES_ID, 0);
    SetOutDescr(LOCAL_BUF_ID, LOCAL_DES_ID, 0);
  }

  // Initialize method
  void Init() {}

  // Return compute grid size
  uint32_t GetGridSize() const { return width_ * height_; }

  // Print output
  void PrintOutput(const void* ptr) const {}

  // Return name
  std::string Name() const { return std::string("DummyKernel"); }

 private:
  // Reference CPU implementation
  bool ReferenceImplementation(uint32_t* output, const uint32_t* input, const float* mask,
                               const uint32_t width, const uint32_t height,
                               const uint32_t maskWidth, const uint32_t maskHeight) {
    return true;
  }

  // Width of the Input array
  const uint32_t width_;

  // Height of the Input array
  const uint32_t height_;
};

#endif  // TEST_DUMMY_KERNEL_DUMMY_KERNEL_H_
