/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#ifndef UTILS_HANDLE_H_
#define UTILS_HANDLE_H_

#include <type_traits>
#include <utility>
#include <cstddef>

namespace rocprofiler {

template <typename T, typename D> class handle_t {
  T value_;
  bool will_delete_;

 public:
  handle_t() noexcept : will_delete_(false) {}

  template <typename U, std::enable_if_t<std::is_same<T, U>::value, bool> = true>
  handle_t(U&& v, bool const will_delete = true) noexcept
      : value_(std::move(v)), will_delete_(will_delete) {}

  // No copy construction or copy assignment of handles
  handle_t(handle_t const&) = delete;
  handle_t& operator=(handle_t const&) = delete;

  handle_t& operator=(handle_t&& h) noexcept {
    reset();
    value_ = std::move(h.value_);
    will_delete_ = h.will_delete_;
    h.release();
    return *this;
  }

  handle_t(handle_t&& h) noexcept : value_(std::move(h.value_)) { h.release(); }

  ~handle_t() noexcept { reset(); }

  T const& get() const noexcept { return value_; }

  typename std::add_lvalue_reference<typename std::remove_pointer<T>::type>::type operator*()
      const noexcept {
    return *value_;
  }

  T operator->() const noexcept { return value_; }

  T const& release() noexcept {
    will_delete_ = false;
    return get();
  }

  void reset() noexcept {
    if (will_delete_) {
      will_delete_ = false;
      D{}(value_);
    }
  }

  void reset(T&& v) noexcept {
    reset();
    value_ = std::move(v);
    will_delete_ = true;
  }
};

}  // namespace rocprofiler

#endif  // UTILS_HANDLE_H_
