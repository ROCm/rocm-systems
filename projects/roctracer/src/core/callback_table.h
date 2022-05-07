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

#ifndef CALLBACK_TABLE_H_
#define CALLBACK_TABLE_H_

#include <ext/prof_protocol.h>

#include <array>
#include <atomic>
#include <cassert>
#include <mutex>
#include <utility>

namespace roctracer {

// Generic callbacks table
template <activity_domain_t Domain, uint32_t N> class CallbackTable {
 public:
  CallbackTable()
      // Zero initialize the callbacks array as the function pointer is used to determine if the
      // callback is enabled.
      : callbacks_() {}

  void Set(uint32_t callback_id, activity_rtapi_callback_t callback_function, void* user_arg) {
    assert(callback_id < N && "callback_id is out of range");
    std::lock_guard lock(mutex_);
    auto& callback = callbacks_[callback_id];
    callback.first.store(callback_function, std::memory_order_relaxed);
    callback.second = user_arg;
  }

  auto Get(uint32_t callback_id) const {
    assert(callback_id < N && "id is out of range");
    std::lock_guard lock(mutex_);
    auto& callback = callbacks_[callback_id];
    return std::make_pair(callback.first.load(std::memory_order_relaxed), callback.second);
  }

  template <typename... Args> void Invoke(uint32_t callback_id, Args... args) {
    if (callbacks_[callback_id].first.load(std::memory_order_relaxed) == nullptr) return;
    if (auto [callback_function, user_arg] = Get(callback_id); callback_function != nullptr)
      callback_function(Domain, callback_id, std::forward<Args>(args)..., user_arg);
  }

 private:
  std::array<std::pair<std::atomic<activity_rtapi_callback_t>, void*>, N> callbacks_;
  mutable std::mutex mutex_;
};

}  // namespace roctracer

#endif  // CALLBACK_TABLE_H_
