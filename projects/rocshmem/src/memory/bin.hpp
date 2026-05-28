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

#ifndef LIBRARY_SRC_MEMORY_BIN_HPP_
#define LIBRARY_SRC_MEMORY_BIN_HPP_

#include <cassert>
#include <stack>

/**
 * @file bin.hpp
 *
 * @brief Simple container class
 *
 * The class wraps a stack and provides simple mutators. A different internal
 * container class might be preferable to a stack.
 */

namespace rocshmem {

template <typename T>
class Bin {
 public:
  /**
   * @brief Is stack empty?
   *
   * @return A boolean denoting stack emptiness
   */
  bool empty() { return stack_.empty(); }

  /**
   * @brief How many elements in stack?
   *
   * @return The number of elements in the stack
   */
  size_t size() { return stack_.size(); }

  /**
   * @brief Emplace an element in stack
   *
   * @param[in] An element
   */
  void put(T element) { stack_.emplace(element); }

  /**
   * @brief Retrieve an element from stack
   *
   * @return An element
   */
  T get() {
    assert(stack_.size());
    auto top = stack_.top();
    stack_.pop();
    return top;
  }

 private:
  /**
   * @brief Implementation container
   */
  std::stack<T> stack_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_BIN_HPP_
