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

#ifndef LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_
#define LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_

namespace rocshmem {

template <typename TYPE>
class FreeList {
public:
  struct PopBackResult {
    TYPE value;
    bool success;
  };

  /**
   * @brief Constructors, assignment operators, and destructor
   * for the stub FreeList are deleted.
   *
   * GIN-GDA only needs a stub FreeList definition.
   */
  __host__ FreeList()                           = delete;
  __host__ FreeList(const FreeList&)            = delete;
  __host__ FreeList(FreeList&&)                 = delete;
  __host__ FreeList& operator=(const FreeList&) = delete;
  __host__ FreeList& operator=(FreeList&&)      = delete;
  __host__ ~FreeList()                          = delete;

  /**
   * @brief  Inserts new element at the end of the FreeList.
   *
   * The element goes into the container right after its last
   * element. The content of val is copied (or moved) to the inserted
   * element.
   *
   * @note Host-side API is not thread safe.
   *
   * @param val The value to insert in the FreeList.
   * @return @c true if the operation succeed, and @c false otherwise.
   */
  __device__ bool push_back(const TYPE& val) { }

  /// @copydoc bool FreeList<TYPE>::push_back(const TYPE&)
  __device__ bool push_back(TYPE&& val) { }

  /**
   * @brief Removes the first element in FreeList, reducing its size by one.
   *
   * @return An object with two fields `value` and `success`. `success` is a
   * boolean indicating if the operation succeeded, and if the operation
   * succeeded, the `value` field contains the popped value.
   */
  __device__ PopBackResult pop_front() {
    return {{}, false};
  }
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_
