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

#ifndef LIBRARY_SRC_MEMORY_ADDRESS_RECORD_HPP_
#define LIBRARY_SRC_MEMORY_ADDRESS_RECORD_HPP_

#include <cassert>
#include <utility>

/**
 * @file address_record.hpp
 *
 * @brief Contains an address record class
 *
 * The address record is intended to be used by allocators which need to
 * keep track of raw pointers and the corresponding allocation size.
 */

namespace rocshmem {

class AddressRecord {
  /**
   * @brief Helper type for address records
   */
  using AR_T = AddressRecord;

 public:
  /**
   * @brief Default constructor
   *
   * @note This constructor generates an empty record. In some contexts,
   * an empty record signifies an INVALID record and is treated
   * accordingly.
   */
  AddressRecord() = default;

  /**
   * @brief Primary constructor type
   *
   * @param[in] raw pointer holding an address
   * @param[in] size
   */
  AddressRecord(char* address, size_t size) : address_(address), size_(size) {}

  /**
   * @brief Accessor for address_
   *
   * @return raw pointer
   */
  char* get_address() { return address_; }

  /**
   * @brief Accessor for size_
   *
   * @return size
   */
  size_t get_size() { return size_; }

  /**
   * @brief Splits record in half
   *
   * @return pair containing initialized address records
   */
  std::pair<AR_T, AR_T> split() {
    assert(address_);
    assert(size_);
    auto half_size{size_ >> 1};
    assert(half_size);

    AR_T e1{address_, half_size};
    AR_T e2{address_ + half_size, half_size};

    return {e1, e2};
  }

  /**
   * @brief Combines record with another record
   *
   * @return an address record containing both input records
   */
  AR_T combine(AR_T other) {
    assert(address_);
    assert(other.get_address());
    bool this_smaller{address_ < other.get_address()};

    auto smaller_addr{this_smaller ? address_ : other.get_address()};
    [[maybe_unused]] auto larger_addr{this_smaller ? other.get_address()
                                                   : address_};

    assert(size_ == other.get_size());
    auto combined_size{size_ + size_};

    assert(smaller_addr + size_ == larger_addr);

    AR_T record{smaller_addr, combined_size};

    return record;
  }

 private:
  /**
   * @brief raw memory pointer
   */
  char* address_{nullptr};

  /**
   * @brief size of address record
   */
  size_t size_{0};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_ADDRESS_RECORD_HPP_
