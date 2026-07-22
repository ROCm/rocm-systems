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

#ifndef _VERIFY_RESULTS_KERNELS_HPP_
#define _VERIFY_RESULTS_KERNELS_HPP_

namespace rocshmem {

[[maybe_unused]] static __global__ void verify_results_kernel_char(
    char *dest, size_t size, size_t stride,
    size_t concurrency, int loop, int skip, int batch,
    bool *verification_error) {

  int start_slot = (batch - (skip % batch)) % batch;
  int verify_iters = min(batch, loop + skip);

  size_t idx = get_flat_id();
  size_t check_per_buf = size * verify_iters;
  size_t total = check_per_buf * concurrency;

  if (idx >= total) {
    return;
  }

  size_t b = idx / check_per_buf;
  size_t local = idx % check_per_buf;
  size_t iter = local / size;
  size_t i = local % size;
  size_t slot = (start_slot + iter) % batch;

  if (dest[b * stride + slot * size + i] != 'a') {
    *verification_error = true;
  }
}

}

#endif /* _VERIFY_RESULTS_KERNELS_HPP_ */
