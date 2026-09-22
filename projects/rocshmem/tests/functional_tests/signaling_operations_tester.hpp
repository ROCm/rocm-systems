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

#ifndef _SIGNALING_OPERATIONS_
#define _SIGNALING_OPERATIONS_

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class SignalingOperationsTester : public Tester {
 public:
  explicit SignalingOperationsTester(TesterArguments args);
  explicit SignalingOperationsTester(TesterArguments args, int signal_op);
  virtual ~SignalingOperationsTester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void verifyResults(size_t size) override;

  int sig_op;
  char *s_buf = nullptr;
  char *r_buf = nullptr;
  uint64_t *sig_addr;
  uint64_t *fetched_value;
};

#endif
