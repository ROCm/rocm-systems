/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef _OCL_PINNED_MEMORY_H_
#define _OCL_PINNED_MEMORY_H_

#include <cstdint>

#include "OCLTestImp.h"

class OCLPinnedMemory : public OCLTestImp {
 public:
  OCLPinnedMemory();
  ~OCLPinnedMemory();

  void open(unsigned int test, char* units, double& conversion, unsigned int deviceId) override;
  void run() override;
  unsigned int close() override;

 private:
  void runNoPrepinnedMemory();
  void runPrepinnedMemory();

  static constexpr const float ratio_ = 0.4f;
  using row_data_t = uint64_t;

  row_data_t* host_memory_;
  size_t row_data_size_ = sizeof(row_data_t);
  size_t row_size_;
  size_t pin_size_;
};

#endif  // _OCL_PINNED_MEMORY_H_
