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

#include "free_list_gtest.hpp"

#include "../src/util.hpp"

using namespace rocshmem;

/*****************************************************************************
 ******************************* Fixture Tests *******************************
 *****************************************************************************/

namespace rocshmem {

template <typename List, typename Value>
__global__ void pop_all(List* list, Value* values, const std::size_t count) {
  const auto stride = blockDim.x * gridDim.x;
  const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  // One push per block. block size is always WF_SIZE
  for (std::size_t i = thread_index; i < count * WF_SIZE; i += stride) {
    if (is_thread_zero_in_wave()) {
      auto last = list->pop_front();
      if (values != nullptr) {
        values[i / WF_SIZE] = last.value;
      }
    }
  }
}

template <typename List, typename Value>
__global__ void push_all(List* list, const Value* values,
                         const std::size_t count) {
  const auto stride = blockDim.x * gridDim.x;
  const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  // One push per block. block size is always WF_SIZE
  for (std::size_t i = thread_index; i < count * WF_SIZE; i += stride) {
    if (is_thread_zero_in_wave()) {
      list->push_back(values[i / WF_SIZE]);
    }
  }
}

template <typename List>
__global__ void pop_empty(List* list, bool* empty) {
  auto pop_result = list->pop_front();
  *empty = !pop_result.success;
}
}  // namespace rocshmem

TYPED_TEST(FreeListTestFixture, pop_empty_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;

  auto& h_input = this->h_input;
  auto& free_list = this->free_list;
  auto& hip_allocator_ = this->hip_allocator_;

  bool *is_empty {nullptr};
  hip_allocator_.allocate(reinterpret_cast<void**>(&is_empty),
                          sizeof(bool));

  CHECK_HIP(hipMemset(is_empty, 0, sizeof(bool)));
  FreeListProxy<Allocator, T> empty_list_proxy{};
  FreeList<T, Allocator>* empty_free_list{empty_list_proxy.get()};

  rocshmem::pop_empty<<<1, 1>>>(empty_free_list, is_empty);
  CHECK_HIP(hipDeviceSynchronize());
  EXPECT_TRUE(is_empty[0]);

  hip_allocator_.deallocate(is_empty);
}

TYPED_TEST(FreeListTestFixture, push_host_pop_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;

  auto& h_input = this->h_input;
  auto& free_list = this->free_list;
  auto& hip_allocator_ = this->hip_allocator_;

  T *results {nullptr};
  bool *is_empty {nullptr};
  size_t size_bytes = sizeof(T) * h_input.size() + sizeof(bool);
  hip_allocator_.allocate(reinterpret_cast<void**>(&results),
                          size_bytes);

  CHECK_HIP(hipMemset(results, 0, size_bytes));
  is_empty = reinterpret_cast<bool*>(results + h_input.size());
  const auto block_size = this->wf_size;
  rocshmem::pop_all<<<1, block_size>>>(free_list, results, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  for (std::size_t i = 0; i < h_input.size(); i++) {
    EXPECT_EQ(results[i], h_input[i]);
  }

  rocshmem::pop_empty<<<1, 1>>>(free_list, is_empty);
  CHECK_HIP(hipDeviceSynchronize());

  EXPECT_TRUE(is_empty[0]);
  hip_allocator_.deallocate(results);
}

TYPED_TEST(FreeListTestFixture, push_host_concurrent_pop_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;

  auto& h_input = this->h_input;
  auto& free_list = this->free_list;
  auto& hip_allocator_ = this->hip_allocator_;

  T *results {nullptr};
  bool *is_empty {nullptr};
  size_t size_bytes = sizeof(T) * h_input.size() + sizeof(bool);
  hip_allocator_.allocate(reinterpret_cast<void**>(&results),
                          size_bytes);

  CHECK_HIP(hipMemset(results, 0, size_bytes));
  is_empty = reinterpret_cast<bool*>(results + h_input.size());
  const auto num_blocks = h_input.size();
  const auto block_size = this->wf_size;
  rocshmem::pop_all<<<num_blocks, block_size>>>(
            free_list, results, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<T> h_results(h_input.size());
  CHECK_HIP(hipMemcpy(h_results.data(), results, sizeof(T) * h_input.size(),
                      hipMemcpyDeviceToHost));

  // sort to guarantee that the ordering is correct
  std::sort(h_input.begin(), h_input.end());
  std::sort(h_results.begin(), h_results.end());


  for (std::size_t i = 0; i < h_results.size(); i++) {
    EXPECT_EQ(h_results[i], h_input[i]);
  }

  rocshmem::pop_empty<<<1, 1>>>(free_list, is_empty);
  CHECK_HIP(hipDeviceSynchronize());

  EXPECT_TRUE(is_empty[0]);
  hip_allocator_.deallocate(results);
}

TYPED_TEST(FreeListTestFixture, push_host_pop_push_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;
  using FreeListType = FreeList<T, Allocator>;

  auto& h_input = this->h_input;
  auto& free_list = this->free_list;
  auto& hip_allocator_ = this->hip_allocator_;

  T *results {nullptr};
  T *d_input {nullptr};
  bool *is_empty {nullptr};
  size_t size_bytes = 2 * sizeof(T) * h_input.size() + sizeof(bool);
  hip_allocator_.allocate(reinterpret_cast<void**>(&results),
                          size_bytes);

  CHECK_HIP(hipMemset(results, 0, size_bytes));
  d_input = reinterpret_cast<T*>(results + h_input.size());
  is_empty = reinterpret_cast<bool*>(d_input + h_input.size());
  const auto block_size = this->wf_size;

  CHECK_HIP(hipMemcpy(d_input, h_input.data(), sizeof(T) * h_input.size(),
                      hipMemcpyHostToDevice));

  rocshmem::pop_all<FreeListType, T><<<1, block_size>>>(
            free_list, nullptr, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  rocshmem::push_all<<<1, block_size>>>(free_list, d_input, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  rocshmem::pop_all<<<1, block_size>>>(free_list, results, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  for (std::size_t i = 0; i < h_input.size(); i++) {
    EXPECT_EQ(results[i], h_input[i]);
  }

  hip_allocator_.deallocate(results);
}

TYPED_TEST(FreeListTestFixture, push_host_pop_concurrent_push_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;
  using FreeListType = FreeList<T, Allocator>;

  auto& h_input = this->h_input;
  auto& free_list = this->free_list;
  auto& hip_allocator_ = this->hip_allocator_;

  T *results {nullptr};
  T *d_input {nullptr};
  size_t size_bytes = 2 * sizeof(T) * h_input.size();
  hip_allocator_.allocate(reinterpret_cast<void**>(&results),
                          size_bytes);

  CHECK_HIP(hipMemset(results, 0, size_bytes));
  d_input = reinterpret_cast<T*>(results + h_input.size());
  const auto block_size = this->wf_size;

  CHECK_HIP(hipMemcpy(d_input, h_input.data(), sizeof(T) * h_input.size(),
                      hipMemcpyHostToDevice));

  rocshmem::pop_all<FreeListType, T><<<1, block_size>>>(
            free_list, nullptr,h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  // Concurrently push all values
  const auto num_blocks = h_input.size();
  rocshmem::push_all<<<num_blocks, block_size>>>(
      free_list, d_input, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  rocshmem::pop_all<<<1, block_size>>>(free_list, results, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<T> h_results(h_input.size());
  CHECK_HIP(hipMemcpy(h_results.data(), results, sizeof(T) * h_input.size(),
                      hipMemcpyDeviceToHost));

  // sort to guarantee that the ordering is correct
  std::sort(h_input.begin(), h_input.end());
  std::sort(h_results.begin(), h_results.end());

  for (std::size_t i = 0; i < h_results.size(); i++) {
    EXPECT_EQ(h_results[i], h_input[i]);
  }

  hip_allocator_.deallocate(results);
}

TYPED_TEST(FreeListTestFixture, push_host_concurrent_pop_push_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;
  using FreeListType = FreeList<T, Allocator>;

  auto& h_input = this->h_input;
  auto& free_list = this->free_list;
  auto& hip_allocator_ = this->hip_allocator_;

  T *results {nullptr};
  T *d_input {nullptr};
  size_t size_bytes = 2 * sizeof(T) * h_input.size();
  hip_allocator_.allocate(reinterpret_cast<void**>(&results),
                          size_bytes);

  CHECK_HIP(hipMemset(results, 0, size_bytes));
  d_input = reinterpret_cast<T*>(results + h_input.size());

  CHECK_HIP(hipMemcpy(d_input, h_input.data(), sizeof(T) * h_input.size(),
                      hipMemcpyHostToDevice));

  const auto block_size = this->wf_size;
  rocshmem::pop_all<FreeListType, T><<<1, block_size>>>(
            free_list, nullptr, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  // Concurrently push all values
  const auto num_blocks = h_input.size();
  rocshmem::push_all<<<num_blocks, block_size>>>(
      free_list, d_input, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  // Concurrently pop all values
  rocshmem::pop_all<<<num_blocks, block_size>>>(
            free_list, results, h_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<T> h_results(h_input.size());
  CHECK_HIP(hipMemcpy(h_results.data(), results, sizeof(T) * h_input.size(),
                      hipMemcpyDeviceToHost));

  // sort to guarantee that the ordering is correct
  std::sort(h_input.begin(), h_input.end());
  std::sort(h_results.begin(), h_results.end());

  for (std::size_t i = 0; i < h_results.size(); i++) {
    EXPECT_EQ(h_results[i], h_input[i]);
  }

  hip_allocator_.deallocate(results);
}
