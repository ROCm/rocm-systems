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

#ifndef _SDMA_ABCE_PUT_NBI_TESTER_HPP_
#define _SDMA_ABCE_PUT_NBI_TESTER_HPP_

#include <hip/hip_ext_copy_engine.h>

#include "tester.hpp"

// ABCE copy-engine analog of SdmaPutNbiTester: one-way pipelined put using
// hipExtCopyEngine directly, drained by polling the engine completion. Mirrors
// David's copy-engine-latency. Self-contained: acquires its own copy-engine
// handle (does not use rocSHMEM's anvil SdmaImpl handles).
class SdmaAbcePutNbiTester : public Tester {
 public:
  explicit SdmaAbcePutNbiTester(TesterArguments args);
  virtual ~SdmaAbcePutNbiTester();

 protected:
  virtual void resetBuffers(size_t size) override;
  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            size_t size) override;
  virtual void verifyResults(size_t size) override;

  // Acquire the copy-engine handle for this rank -> peer once (after init).
  void ensureHandle();

  char *s_buf;
  char *r_buf;
  uint64_t *drain_signal_;                 // per-wg completion word for batch drain
  hipExtCopyEngineDeviceHandle handle_{};  // this rank's engine to the peer
  bool have_handle_{false};
};

#endif
