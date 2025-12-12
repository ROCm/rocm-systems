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

#ifndef _PUTMEM_SIGNAL_ON_STREAM_TESTER_HPP_
#define _PUTMEM_SIGNAL_ON_STREAM_TESTER_HPP_

#include "tester.hpp"
#include <vector>
#include <hip/hip_runtime.h>

using namespace rocshmem;

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class PutmemSignalOnStreamTester : public Tester {
 public:
  explicit PutmemSignalOnStreamTester(TesterArguments args);
  virtual ~PutmemSignalOnStreamTester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void preLaunchKernel() override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void postLaunchKernel() override;

  virtual void verifyResults(size_t size) override;

 private:
  char *source_buf;
  char *dest_buf;
  uint64_t *sig_addr;
  int my_pe;
  int n_pes;
  size_t buf_size;
  int num_streams = 1;
  int pe_target;  // Target PE to put to
  int sig_op = ROCSHMEM_SIGNAL_SET;  // Signal operation
  std::vector<hipStream_t> streams;
  std::vector<hipEvent_t> start_events_timed;
  std::vector<hipEvent_t> stop_events_timed;
};

#include "putmem_signal_on_stream_tester.cpp"

#endif

