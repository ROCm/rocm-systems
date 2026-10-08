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

#ifndef _HOST_RMA_TESTER_HPP_
#define _HOST_RMA_TESTER_HPP_

#include "tester.hpp"

using namespace rocshmem;

/******************************************************************************
 * HOST TESTER CLASS
 *
 * Tests host-side blocking RMA (putmem, getmem) and AMOs (amo_fetch_add,
 * amo_fetch_cas) on the non-MPI IPC path (AIROCSHMEM-419).
 *
 * All operations are issued from the host via rocshmem_ctx_* APIs.
 *****************************************************************************/
class HostRmaTester : public Tester {
 public:
  explicit HostRmaTester(TesterArguments args);
  virtual ~HostRmaTester();

 protected:
  virtual void resetBuffers(size_t size) override;
  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;
  virtual void verifyResults(size_t size) override;

 private:
  int my_pe;
  int n_pes;
  int peer;

  /* Symmetric buffers for RMA tests */
  char  *source_buf{nullptr};
  char  *dest_buf{nullptr};

  /* Symmetric long counter/flag for long AMO tests */
  long *amo_buf{nullptr};

  /* Symmetric int counter/flag for int AMO tests (exercises 32-bit kernel path) */
  int *amo_int_buf{nullptr};

  /* Symmetric array of longs for wait_until_all/any/some/vector tests */
  static constexpr size_t WAIT_NELEMS = 4;
  long *wait_buf{nullptr};

  /* Local storage for results from _any / _some variants (non-symmetric) */
  size_t p2p_result_idx{SIZE_MAX};
  size_t p2p_result_ncompleted{0};
};

#endif  // _HOST_RMA_TESTER_HPP_
