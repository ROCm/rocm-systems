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

#ifndef _RANDOM_ACCESS_TESTER_HPP_
#define _RANDOM_ACCESS_TESTER_HPP_

#include "tester.hpp"

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void RandomAccessTest(int loop, int skip, long long int *start_time,
                                 long long int *end_time, int *s_buf,
                                 int *r_buf, int size, OpType type,
                                 int coal_coef, int num_bins, int num_waves,
                                 uint32_t *threads_bins, uint32_t *off_bins,
                                 uint32_t *PE_bins);

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class RandomAccessTester : public Tester {
 public:
  explicit RandomAccessTester(TesterArguments args);
  virtual ~RandomAccessTester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void verifyResults(size_t size) override;

  int *r_buf;
  int *s_buf;
  int *h_buf;
  int *h_dev_buf;
  uint32_t *_threads_bins;
  uint32_t *_off_bins;
  uint32_t *_PE_bins;
  int _num_waves;
  int _num_bins;
  static constexpr int space = 16 * 2;
};

#endif
