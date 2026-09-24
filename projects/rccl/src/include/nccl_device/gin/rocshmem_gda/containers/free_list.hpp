// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_
#define LIBRARY_SRC_CONTAINERS_FREE_LIST_HPP_

namespace rocshmem {

/*
 * @brief Stub class for rocSHMEM header compatability.
 *
 * QueuePairIONIC fetching AMOs need a real (non-stub) implementation;
 * RCCL GIN GDA currently uses only non-fetching AMOs.
 */
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
  __device__ bool push_back(const TYPE& val) { return false; }

  /// @copydoc bool FreeList<TYPE>::push_back(const TYPE&)
  __device__ bool push_back(TYPE&& val) { return false; }

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
