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

#ifndef ROCSHMEM_FREE_LIST_GTEST_HPP
#define ROCSHMEM_FREE_LIST_GTEST_HPP

#include <numeric>
#include <vector>

#include "../src/containers/free_list_impl.hpp"
#include "gtest/gtest.h"
#include "../src/memory/hip_allocator.hpp"
#include "wf_size.hpp"

namespace rocshmem {

template <typename ValueType>
class FreeListTestFixture : public ::testing::Test {
 public:
  FreeListTestFixture() : h_input(num_elements) {
    std::iota(h_input.begin(), h_input.end(), T{1});
    free_list = list_proxy.get();
  }

 protected:
  void SetUp() override {
    free_list->push_back_range(h_input.begin(), h_input.end());
    wf_size = get_wf_size();
  }

  using T = ValueType;
  using Allocator = HIPAllocator;
  Allocator hip_allocator_ {};
  const std::size_t num_elements{32};
  std::vector<T> h_input{};
  int wf_size;

  FreeListProxy<Allocator, T> list_proxy{};
  FreeList<T, Allocator>* free_list{};
};

using TestTypes = ::testing::Types<std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(FreeListTestFixture, TestTypes);

}  // namespace rocshmem

#endif  // ROCSHMEM_FREE_LIST_GTEST_HPP
