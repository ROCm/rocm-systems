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

#ifndef _DEVICE_BITCODE_TESTER_HPP_
#define _DEVICE_BITCODE_TESTER_HPP_

#include "tester.hpp"
#include <hip/hip_runtime.h>
#include <string>

class DeviceBitcodeTester : public Tester {
 public:
  explicit DeviceBitcodeTester(TesterArguments args);
  ~DeviceBitcodeTester() override;

  void execute() override;

 protected:
  void resetBuffers(size_t size) override;
  void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                    size_t size) override;
  void verifyResults(size_t size) override;

 private:
  std::string resolve_hsaco_path();
  void launch(const char* kernel, void** args,
              dim3 grid = 1, dim3 block = 64);

  template <typename T>
  void run_rma_test(const char* label, const char* kernel,
                    int count, T scale, T offset);

  template <typename T>
  void run_scalar_put_test(const char* label, const char* kernel,
                           T scale, T offset);

  int my_pe;
  int n_pes;
  hipModule_t module = nullptr;
  bool all_pass = true;
};

#endif
