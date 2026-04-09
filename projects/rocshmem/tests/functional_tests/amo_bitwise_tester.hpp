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

#ifndef _AMO_BITWISE_TESTER_HPP_
#define _AMO_BITWISE_TESTER_HPP_

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
template <typename T>
class AMOBitwiseTester : public Tester {
 public:
  explicit AMOBitwiseTester(TesterArguments args);
  virtual ~AMOBitwiseTester();

 protected:
  virtual void resetBuffers(size_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;

  virtual void verifyResults(size_t size) override;

  void verifyDestValues();
  void verifyReturnValues();

  int destIndex(int l, int elem_idx) const;
  int numElems() const;
  std::pair<T*, int> retChunk(int l, int elem_idx) const;

  T* dest{nullptr};        // symmetric target buffer [loop][elem]
  T* ret_val{nullptr};     // device returns [loop][thread]

  size_t n_in{0};          // num_wgs * wg_size
  size_t n_out{0};         // elements per loop: PerBlock->num_wgs, PerGrid->1
  size_t n_loops{0};       // loop + skip
};

#endif
