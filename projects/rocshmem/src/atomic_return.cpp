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

#include "atomic_return.hpp"

#include <cassert>

#include "constants.hpp"

namespace rocshmem {

void allocate_atomic_region(atomic_ret_t** atomic_ret, int num_wg) {
  atomic_ret_t* tmp_ret{nullptr};
  /*
   * Allocate device-side control struct for the atomic return region.
   */
  CHECK_HIP(
      hipMalloc(reinterpret_cast<void**>(&tmp_ret), sizeof(atomic_ret_t)));

  /*
   * Allocate fine-grained device-side memory for the atomic return
   * region.
   */
  size_t size_bytes{max_nb_atomic * num_wg * sizeof(uint64_t)};
#ifdef HIP_SUPPORTS_MALLOC_UNCACHED
  CHECK_HIP(
      hipExtMallocWithFlags(reinterpret_cast<void**>(&tmp_ret->atomic_base_ptr),
                            size_bytes, hipDeviceMallocUncached));
#else
  CHECK_HIP(
      hipExtMallocWithFlags(reinterpret_cast<void**>(&tmp_ret->atomic_base_ptr),
                            size_bytes, hipDeviceMallocFinegrained));
#endif

  /*
   * Zero-initialize the entire atomic return region.
   */
  CHECK_HIP(hipMemset(tmp_ret->atomic_base_ptr, 0, size_bytes));

  *atomic_ret = tmp_ret;
}

void init_g_ret(SymmetricHeap* heap_handle, MPI_Comm thread_comm, int num_wg,
                char** g_ret) {
  /*
   * Create space on the symmetric heap
   */
  void* ptr{nullptr};
  size_t size_bytes{sizeof(int64_t) * MAX_WG_SIZE * num_wg};
  heap_handle->malloc(&ptr, size_bytes);
  assert(ptr);

  /*
   * Assign g_ret the output of the malloc
   */
  *g_ret = reinterpret_cast<char*>(ptr);

  /*
   * Make sure that all processing elements have done this before
   * continuing.
   */
  mpilib_ftable_.Barrier(thread_comm);
}

}  // namespace rocshmem
