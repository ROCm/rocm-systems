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

#ifndef LIBRARY_SRC_ATOMIC_RETURN_HPP_
#define LIBRARY_SRC_ATOMIC_RETURN_HPP_

#include <hip/hip_runtime.h>

#include "memory/symmetric_heap.hpp"
#include "util.hpp"

namespace rocshmem {

const int max_nb_atomic = 4096;

struct atomic_ret_t {
  uint64_t* atomic_base_ptr;
  uint32_t atomic_lkey;
  uint64_t atomic_counter;
};

void allocate_atomic_region(atomic_ret_t** atomic_ret, int num_wg);

void init_g_ret(SymmetricHeap* heap_handle, MPI_Comm thread_comm, int num_wg,
                char** g_ret);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_ATOMIC_RETURN_HPP_
