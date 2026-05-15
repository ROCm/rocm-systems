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

#ifndef LIBRARY_SRC_MEMORY_POW2_BINS_HPP_
#define LIBRARY_SRC_MEMORY_POW2_BINS_HPP_

#include <cassert>
#include <map>

#include "constants.hpp"
#include "bin.hpp"
#include "binner.hpp"
#include "shmem_allocator_strategy.hpp"

/**
 * @file pow2_bins.hpp
 *
 * @brief Contains an allocator strategy for the heap.
 *
 * This strategy returns memory chunks with power-of-two sizes.
 */

namespace rocshmem {

template <typename AR_T, typename HM_T>
class Pow2Bins : public ShmemAllocatorStrategy {
  /**
   * @brief Helper type for bins of address records
   */
  using BIN_T = Bin<AR_T>;

  /**
   * @brief Helper type for size to bin maps
   */
  using BINS_T = std::map<size_t, BIN_T>;

  /**
   * @brief Helper type for BINS_T iterator
   */
  using BINS_IT_T = typename BINS_T::iterator;

  /**
   * @brief Helper type for raw pointers to address record maps
   */
  using PROFFERED_T = std::map<char*, AR_T>;

 public:
  /**
   * @brief Required for default construction of other objects
   *
   * @note Not intended for direct usage.
   */
  Pow2Bins() = default;

  /**
   * @brief Primary constructor type
   *
   * @param[in] Raw pointer to heap memory type
   */
  explicit Pow2Bins(HM_T* heap_mem)
      : binner_{&bins_, heap_mem->get_ptr(), heap_mem->get_size()} {
    binner_.assign_heap_to_bins();
  }

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   */
  void alloc(char** ptr, size_t request_size) override {
    assert(ptr);
    *ptr = nullptr;

    if (!request_size) {
      return;
    }

    auto do_request_using = [&](auto it) {
      if (it != bins_.end()) {
        AR_T record{retrieve_record_from_bin(it)};
        /*
         * Record retrieval may have generated INVALID record.
         * If INVALID, do not mark record as proffered.
         * Also notice, input pointer "ptr" may have nullptr value.
         */
        if (record.get_address()) {
          emplace_in_proffered(record);
        }
        *ptr = record.get_address();
      }
    };

    /*
     * Try to find exact power-of-two matches.
     */
    do_request_using(bins_.find(request_size));
    if (*ptr) {
      return;
    }

    /*
     * Round up to the nearest power-of-two size.
     */
    do_request_using(bins_.lower_bound(request_size));
  }

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   *
   * @note Not implemented
   */
  __device__ void alloc([[maybe_unused]] char** ptr,
                        [[maybe_unused]] size_t request_size) override {}

  /**
   * @brief Frees memory from the heap
   *
   * Released memory is tracked by bookkeeping structures within this class.
   *
   * @param[in] Raw pointer to heap memory
   *
   * @note We do not attempt to combine records together into larger
   * even if it is possible to do so. This may eventually exhaust the
   * number of records in the larger bins.
   */
  void free(char* ptr) override {
    auto record{retrieve_from_proffered(ptr)};
    emplace_record_in_bin(record);
  }

  /**
   * @brief Frees memory from the heap
   *
   * Released memory is tracked by bookkeeping structures within this class.
   *
   * @param[in] Raw pointer to heap memory
   *
   * @note Not implemented
   */
  __device__ void free([[maybe_unused]] char* ptr) override {}

  /**
   * @brief Used memory from the heap
   *
   * Sum of all proffered_ memory sizes
   *
   * @return memory size
   */
  size_t get_used() override {
    return amount_proffered();
  }

  /**
   * @brief Sum of all proffered_ memory sizes
   *
   * @return memory size
   */
  size_t amount_proffered() {
    size_t size{0};
    auto sum_sizes = [&size](const auto& pair) {
      auto [address, record] = pair;
      size += record.get_size();
    };
    std::for_each(proffered_.begin(), proffered_.end(), sum_sizes);
    return size;
  }

  /**
   * @brief Accessor for bins_ internal data structure
   *
   * @return BINS_T raw pointer
   *
   * @note Used by unit-test test fixture
   */
  BINS_T* get_bins() { return &bins_; }

 private:
  /**
   * @brief Store address record in appropriate bin
   *
   * @param[in] An address record
   */
  void emplace_record_in_bin(AR_T record) {
    auto record_size{record.get_size()};

    auto bins_it = bins_.find(record_size);
    assert(bins_it != bins_.end());

    auto& [IGNORE_BIN_SIZE, bin]{*bins_it};
    bin.put(record);
  }

  /**
   * @brief Retrieve address record from bin within a BINS_IT_T
   *
   * @param[in] a VALID FORWARD iterator containing a bin
   *
   * @return An address record
   *
   * @note May return an empty record indicating a retrieval failure.
   * The failure must be resolved somewhere else within the allocation
   * framework.
   */
  AR_T retrieve_record_from_bin(BINS_IT_T it) {
    try_to_ensure_bin_has_records(it);

    auto& [IGNORE, bin]{*it};

    if (bin.empty()) {
      return AR_T{};
    }

    return bin.get();
  }

  /**
   * @brief Attempt to accommodate a subsequent record request
   *
   * The bin under iterator may not (or may) contain records when
   * this method is invoked. This method tries to make at least one
   * record available.
   *
   * @param[in] a VALID FORWARD iterator containing a bin
   *
   * @note BINS_T container must be sorted in ascending bin size for
   * delegate method "split_larger_bin"
   */
  void try_to_ensure_bin_has_records(BINS_IT_T it) {
    auto& [IGNORE, bin]{*it};

    if (!bin.empty()) {
      return;
    }

    split_larger_bin(it);
  }

  /**
   * @brief Split records in next largest bin
   *
   * This method tries to create records within this bin by decomposing
   * records in other bins (with larger bin sizes).
   *
   * This method may be called recursively if records are unavailable
   * at each step until records are made available to the initial
   * bin.
   *
   * @param[in] a VALID FORWARD iterator containing a bin
   *
   * @note BINS_T container must be sorted in ascending bin size
   */
  void split_larger_bin(BINS_IT_T it) {
    assert(it != bins_.end());

    auto next{std::next(it)};
    if (next == bins_.end()) {
      return;
    }

    auto& [IGNORE_NEXT, next_bin]{*next};
    if (next_bin.empty()) {
      split_larger_bin(next);
    }

    if (next_bin.empty()) {
      return;
    }

    auto record{next_bin.get()};
    auto [smaller, larger]{record.split()};

    auto& [IGNORE, bin]{*it};
    bin.put(larger);
    bin.put(smaller);
  }

  /**
   * @brief Store record in proffered_ data structure
   *
   * @param[in] An address record
   */
  void emplace_in_proffered(AR_T record) {
    assert(record.get_address());
    proffered_[record.get_address()] = record;
  }

  /**
   * @brief Retrieve record from proffered_ data structure
   *
   * @param[in] raw memory pointer
   *
   * The memory pointer must match a pointer previously
   * handed to the user.
   *
   * @return An address record
   */
  AR_T retrieve_from_proffered(char* ptr) {
    assert(ptr);

    auto prof_it{proffered_.find(ptr)};
    assert(prof_it != proffered_.end());

    auto [IGNORE_RECORD_ADDR, record]{*prof_it};
    proffered_.erase(prof_it);

    return record;
  }

  /**
   * @brief Holds a SORTED set of bin objects
   *
   * The bin objects are sorted by ascending bin sizes. Each bin holds
   * address records with matching sizes. The bin sizes increase by
   * powers-of-two. There is a minimum bin size set by memory alignment
   * constraints.
   */
  BINS_T bins_{};

  /**
   * @brief Initializes the bins_ object
   *
   * The binner functions as a bins_ builder. The binner access data in
   * the heap memory to establish the number of bins and each bin's size.
   *
   * Additionally, binner_ will assign the heap memory to bins_.
   */
  Binner<AR_T, BINS_T> binner_{};

  /**
   * @brief Tracks all address records handed over to user.
   *
   * This member is required to track the sizes of memory allocations
   * since the user is not required to provide a size_t field back
   * through the "free" interface.
   */
  PROFFERED_T proffered_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_POW2_BINS_HPP_
