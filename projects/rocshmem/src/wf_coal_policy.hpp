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

#ifndef LIBRARY_SRC_WF_COAL_POLICY_HPP_
#define LIBRARY_SRC_WF_COAL_POLICY_HPP_

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "util.hpp"

namespace rocshmem {

class WfCoalOn {
 public:
  /**
   * Coalesce contiguous messages from a single wavefront.
   *
   * With regards to calling threads, the command must already be the
   * same for active threads otherwise they must have diverged at the
   * function call level.
   */
  __device__ bool coalesce(int pe, const void *source, const void *dest,
                           size_t *size);
};

// clang-format off
NOWARN(-Wunused-parameter,
class WfCoalOff {
 public:
  __device__ bool coalesce(int pe, const void *source, const void *dest,
                           size_t *size) {
    return true;
  }
};
)
// clang-format on

/**
 * Compile time configuration options will enable or disable this feature.
 */
#ifdef USE_WF_COAL
typedef WfCoalOn WavefrontCoalescer;
#else
typedef WfCoalOff WavefrontCoalescer;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_WF_COAL_POLICY_HPP_
