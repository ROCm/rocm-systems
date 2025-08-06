////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIESd OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_UTIL_LAZY_PTR_H_
#define HSA_RUNTIME_CORE_UTIL_LAZY_PTR_H_

#include <memory>
#include <utility>
#include <functional>

#include "core/util/locks.h"
#include "core/util/utils.h"

namespace rocr {

/*
 * Wrapper for a std::unique_ptr that initializes its object at first use.
 */
template <typename T> class lazy_ptr {
 public:
  lazy_ptr() {}

  explicit lazy_ptr(std::function<T*()> Constructor) { reset(Constructor); }

  lazy_ptr(lazy_ptr&& rhs) {
    obj = std::move(rhs.obj);
    func = std::move(rhs.func);
  }

  lazy_ptr& operator=(lazy_ptr&& rhs) {
    obj = std::move(rhs.obj);
    func = std::move(rhs.func);
  }

  lazy_ptr(lazy_ptr&) = delete;
  lazy_ptr& operator=(lazy_ptr&) = delete;

  void reset(std::function<T*()> Constructor = nullptr) {
    obj.reset();
    func = std::move(Constructor);
  }

  void reset(T* ptr) {
    obj.reset(ptr);
    func = nullptr;
  }

  bool operator==(T* rhs) const { return obj.get() == rhs; }
  bool operator!=(T* rhs) const { return obj.get() != rhs; }

  const std::unique_ptr<T>& operator->() const {
    make(true);
    assert(obj != nullptr && "Null dereference through lazy_ptr.");
    return obj;
  }

  std::unique_ptr<T>& operator*() {
    make(true);
    return obj;
  }

  const std::unique_ptr<T>& operator*() const {
    make(true);
    return obj;
  }

  /*
   * Ensures that the object is created or is being created.
   * This is useful when early construction of the object is required.
   */
  void touch() const { make(false); }

  // Tells if the lazy object has been constructed or not.
  // Construction may fail silently (return nullptr).
  bool created() const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return func == nullptr;
  }

  // Tells if the lazy object exists or not.
  bool empty() const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return obj == nullptr;
  }

 private:
  mutable std::unique_ptr<T> obj;
  mutable std::function<T*(void)> func;
  mutable KernelMutex lock;

  // Separated from make to improve inlining.
  void make_body(bool block) const {
    if (block) {
      lock.Acquire();
    } else if (!lock.Try()) {
      return;
    }
    MAKE_SCOPE_GUARD([&]() { lock.Release(); });
    if (func == nullptr) return;
    T* ptr = func();
    obj.reset(ptr);
    std::atomic_thread_fence(std::memory_order_release);
    func = nullptr;
  }

  __forceinline void make(bool block) const {
    if (!created()) {
      make_body(block);
    }
  }

};

}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_LAZY_PTR_H_
