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

#ifndef LIBRARY_SRC_MEMORY_BINNER_HPP_
#define LIBRARY_SRC_MEMORY_BINNER_HPP_

#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

#include "constants.hpp"
#include "bin.hpp"

/**
 * @file binner.hpp
 *
 * @brief Contains a builder for bins_ object in Pow2Bins.
 */

namespace rocshmem {

/**
 * @brief Helper to choose between clz variants
 *
 * @param[in] An input value
 *
 * @return __builtin_clz value
 */
template <typename T>
inline int clz_fn([[maybe_unused]] T v) {
  assert(false);
  return 0;
}

template <>
inline int clz_fn(unsigned v) {
  return __builtin_clz(v);
}

template <>
inline int clz_fn(unsigned long v) {
  return __builtin_clzl(v);
}

template <>
inline int clz_fn(unsigned long long v) {
  return __builtin_clzll(v);
}

/**
 * @brief Finds the bit position of the first one
 *
 * @param[in] An input value
 *
 * @return A bit position or UINT_MAX
 */
template <typename T>
inline unsigned find_first_set_one(T v) {
  if (v == 0) {
    return UINT_MAX;
  }
  auto num_bytes{sizeof(T)};
  auto num_bits{num_bytes * CHAR_BIT};
  auto num_leading_zeroes{clz_fn(v)};
  auto last_zero_bit_position{num_bits - num_leading_zeroes};
  auto first_one_bit_position{last_zero_bit_position - 1};
  return first_one_bit_position;
}

template <typename AR_T, typename BINS_T>
class Binner {
 public:
  /**
   * @brief Required for default construction of other objects
   *
   * @note Not intended for direct usage.
   */
  Binner() = default;

  /**
   * @brief Primary constructor type
   *
   * Constructs bins by inserting a set of bin into it.
   *
   * @param[in] Raw memory pointer to bins object
   * @param[in] Raw memory pointer to symmetric heap memory.
   * @param[in] Size of symmetric heap allocation.
   */
  Binner(BINS_T* bins, char* heap_ptr, size_t heap_size)
      : bins_{bins}, heap_{heap_ptr, heap_size} {
    auto bit_position_heap{find_first_set_one(heap_size)};
    assert(bit_position_heap != UINT_MAX);

    auto bit_position_align{find_first_set_one(ALIGNMENT)};
    assert(bit_position_align != UINT_MAX);

    assert(bit_position_heap >= bit_position_align);

    auto difference_in_position{bit_position_heap - bit_position_align};
    auto vector_size{difference_in_position + 1};

    std::vector<size_t> v;
    v.resize(vector_size);

    auto pow2 = [&bit_position_align]() {
      return std::pow(2, bit_position_align++);
    };

    std::generate(v.begin(), v.end(), pow2);

    for (auto x : v) {
      bins_->insert({x, Bin<AR_T>{}});
    }
  }

  /**
   * @brief Assign symmetric heap memory to bins object
   */
  void assign_heap_to_bins() {
    assert(bins_);
    assert(heap_.get_address());
    assert(heap_.get_size());

    ignore_unaligned_heap_memory();
    assign_heap_memory_to_bins();
  }

  /**
   * @brief Accessor for bins_ pointer.
   *
   * @return Raw pointer to bins_
   *
   * @note Used by unit-test test fixture
   */
  BINS_T* get_bins() { return bins_; }

  /**
   * @brief Dump the bins_ object to the standard out
   */
  void dump_bins() {
    auto dump_bin = [](const auto& pair) {
      auto [address, bin] = pair;
      std::cout << "address " << address << " bin.size " << bin.size()
                << std::endl;
    };
    std::for_each(bins_->begin(), bins_->end(), dump_bin);
  }

 private:
  /**
   * @brief Shift assignment point past unaligned memory.
   */
  void ignore_unaligned_heap_memory() {
    static_assert(sizeof(char) == 1);

    auto heap_ptr{heap_.get_address()};
    size_t off_by_bytes{reinterpret_cast<size_t>(heap_ptr) % ALIGNMENT};

    if (off_by_bytes) {
      size_t required_offset{ALIGNMENT - off_by_bytes};
      heap_ptr += required_offset;
      auto heap_size{heap_.get_size() - required_offset};
      heap_ = AR_T(heap_ptr, heap_size);
    }
  }

  /**
   * @brief Delegate method for top-level heap assignment.
   */
  void assign_heap_memory_to_bins() {
    AR_T chunk = heap_;
    while (chunk.get_address()) {
      chunk = assign_heap_chunk(chunk);
    }
  }

  /**
   * @brief Chunk heap memory up into bin sized allocations
   *
   * The method starts with the largest chunks possible and proceeds
   * to smaller chunks. It continues chunking until there are no more
   * chunks which will fit into bins.
   *
   * @param[in] An address record
   *
   * @return An address record minus a chunk
   */
  AR_T assign_heap_chunk(AR_T address_record) {
    char* memory_chunk{address_record.get_address()};
    size_t memory_chunk_size{address_record.get_size()};

    auto rit{bins_->rbegin()};
    auto rend{bins_->rend()};
    while (rit != rend) {
      auto& [bin_memory_chunk_size, bin] = *rit;
      if (memory_chunk_size >= bin_memory_chunk_size) {
        break;
      }
      rit++;
    }

    if (rit == rend) {
      return {nullptr, 0};
    }

    auto& [bin_memory_chunk_size, bin] = *rit;
    bin.put({memory_chunk, bin_memory_chunk_size});

    return {memory_chunk + bin_memory_chunk_size,
            memory_chunk_size - bin_memory_chunk_size};
  }

  /**
   * @brief Raw pointer to external bins_ object
   */
  BINS_T* bins_{nullptr};

  /**
   * @brief An address record holding heap object's information
   */
  AR_T heap_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_BINNER_HPP_
