/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef _BUFFER_REGISTER_SYMMETRIC_TESTER_HPP_
#define _BUFFER_REGISTER_SYMMETRIC_TESTER_HPP_

#include "tester.hpp"

class BufferRegisterSymmetricTester : public Tester {
 public:
  explicit BufferRegisterSymmetricTester(TesterArguments args);
  ~BufferRegisterSymmetricTester() override;

 protected:
  void resetBuffers(uint64_t size) override;
  void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                    uint64_t size) override;
  void verifyResults(uint64_t size) override;

 private:
  void registerBuffer();
  void unregisterBuffer();

  bool skip_{false};
  bool pass_{true};
  void *original_{nullptr};
  unsigned char *alias_{nullptr};
  unsigned char *source_{nullptr};
  size_t allocation_size_{0};
  size_t worker_count_{0};
#if HIP_VERSION >= 70200000
  hipMemGenericAllocationHandle_t handle_{};
#endif
};

#endif  // _BUFFER_REGISTER_SYMMETRIC_TESTER_HPP_
