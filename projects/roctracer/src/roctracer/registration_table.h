/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef UTIL_CALLBACK_TABLE_H_
#define UTIL_CALLBACK_TABLE_H_

#include "ext/prof_protocol.h"

#include <array>
#include <atomic>
#include <cassert>
#include <optional>
#include <shared_mutex>
#include <utility>

namespace roctracer::util {

namespace detail {
struct False {
  constexpr bool operator()() { return false; }
};
}  // namespace detail

// Generic callbacks table
template <typename T, uint32_t N, typename IsStopped = detail::False> class RegistrationTable {
 public:
  template <typename... Args> void Register(uint32_t operation_id, Args... args) {
    assert(operation_id < N && "operation_id is out of range");
    auto& entry = table_[operation_id];
    std::unique_lock lock(entry.mutex);
    if (!entry.enabled.exchange(true, std::memory_order_relaxed))
      registered_count_.fetch_add(1, std::memory_order_relaxed);
    entry.data = T{std::forward<Args>(args)...};
  }

  void Unregister(uint32_t operation_id) {
    assert(operation_id < N && "id is out of range");
    auto& entry = table_[operation_id];
    std::unique_lock lock(entry.mutex);
    if (entry.enabled.exchange(false, std::memory_order_relaxed))
      registered_count_.fetch_sub(1, std::memory_order_relaxed);
  }

  std::optional<T> Get(uint32_t operation_id) const {
    assert(operation_id < N && "id is out of range");
    auto& entry = table_[operation_id];
    if (!entry.enabled.load(std::memory_order_relaxed) || IsStopped{}()) return std::nullopt;
    std::shared_lock lock(entry.mutex);
    return entry.enabled.load(std::memory_order_relaxed) ? std::make_optional(entry.data)
                                                         : std::nullopt;
  }

  bool IsEmpty() const { return registered_count_.load(std::memory_order_relaxed) == 0; }

 private:
  std::atomic<size_t> registered_count_{0};
  struct {
    std::atomic<bool> enabled{false};
    mutable std::shared_mutex mutex;
    T data;
  } table_[N]{};
};


}  // namespace roctracer::util

#endif  // UTIL_CALLBACK_TABLE_H_
