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

#ifndef _MULTI_CTX_PUT_TESTER_HPP_
#define _MULTI_CTX_PUT_TESTER_HPP_

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *
 * Exercises multiple user contexts (each backed by a separate team), where
 * every workgroup creates its own context and performs put operations through
 * it.  With ROCSHMEM_GDA_NIC_POLICY=PER_CONTEXT and multiple NICs, each
 * context routes traffic through a different NIC.
 *****************************************************************************/
class MultiCtxPutTester : public Tester {
 public:
  explicit MultiCtxPutTester(TesterArguments args);
  virtual ~MultiCtxPutTester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void preLaunchKernel() override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void postLaunchKernel() override;

  virtual void verifyResults(size_t size) override;

  int num_ctxs_ {0};
  std::vector<rocshmem::rocshmem_team_t> teams_;
  rocshmem::rocshmem_team_t *teams_on_device_ {nullptr};

  char *source = nullptr;
  char *dest = nullptr;
};

#endif
