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

#ifndef _TYPED_RMA_TESTER_HPP_
#define _TYPED_RMA_TESTER_HPP_

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *
 * Exercises the typed RMA device APIs (rocshmem_ctx_T_put/get/put_nbi/get_nbi
 * and rocshmem_ctx_T_p/g) for a specific element type T.  Used to provide
 * __half and __hip_bfloat16 coverage for GetTestType, GetNBITestType,
 * PutTestType, PutNBITestType, PTestType, and GTestType.
 *****************************************************************************/
template <typename T>
class TypedRMATester : public Tester {
 public:
  explicit TypedRMATester(TesterArguments args);
  virtual ~TypedRMATester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void verifyResults(size_t size) override;

  T *source = nullptr;
  T *dest   = nullptr;
  int *grid_psync = nullptr;
};

#include "typed_rma_tester.cpp"

#endif
