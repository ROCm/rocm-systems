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

#include <cassert>
#include <mutex>
#include <utility>

namespace roctracer {

// Generic callbacks table
template <uint32_t N> class CallbackTable {
 public:
  CallbackTable()
      // Zero initialize the callbacks array as the function pointer is used to determine if the
      // callback is enabled.
      : callbacks_() {}

  void Set(uint32_t id, activity_rtapi_callback_t callback, void* arg) {
    assert(id < N && "id is out of range");
    std::lock_guard lock(mutex_);
    callbacks_[id] = {callback, arg};
  }

  void Get(uint32_t id, activity_rtapi_callback_t* callback, void** arg) const {
    assert(id < N && "id is out of range");
    std::lock_guard lock(mutex_);
    std::tie(*callback, *arg) = callbacks_[id];
  }

 private:
  std::array<std::pair<activity_rtapi_callback_t /* callback */, void* /* arg */>, N> callbacks_;
  mutable std::mutex mutex_;
};

}  // namespace roctracer

#endif  // CALLBACK_TABLE_H_
