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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTME_CORE_INC_SHARED_H_
#define HSA_RUNTME_CORE_INC_SHARED_H_

#include <assert.h>
#include <cstring>
#include <functional>
#include <memory>

#include "core/util/utils.h"

namespace rocr {
namespace core {
/// @brief Base class encapsulating the allocator and deallocator for
/// shared shared object.  As used this will allocate GPU visible host
/// memory mapped to all GPUs.
class BaseShared {
 public:
  static void SetAllocateAndFree(
      const std::function<void*(size_t, size_t, uint32_t, int)>& alloc,
      const std::function<void(void*)>& fr) {
    allocate_() = alloc;
    free_() = fr;
  }

 protected:
  static __forceinline std::function<void*(size_t, size_t, uint32_t, int)>&
   allocate_() {
    static std::function<void*(size_t, size_t, uint32_t, int)> alloc =
                                                                      nullptr;
    return alloc;
  }
  static __forceinline std::function<void(void*)>&
   free_() {
    static std::function<void(void*)> fr = nullptr;
    return fr;
  }

};

/// @brief Default Allocator for Shared.  Ensures allocations are whole pages.
template <typename T> class PageAllocator : private BaseShared {
 public:
  __forceinline static T* alloc(int flags = 0) {
    T* ret = reinterpret_cast<T*>(
                        allocate_()(AlignUp(sizeof(T), 4096), 4096, flags, 0));
    if (ret == nullptr) throw std::bad_alloc();

    MAKE_NAMED_SCOPE_GUARD(throwGuard, [&]() { free_()(ret); });

    new (ret) T;

    throwGuard.Dismiss();
    return ret;
  }

  __forceinline static T* alloc(int agent_node_id, int flags) {
    T* ret = reinterpret_cast<T*>(
            allocate_()(AlignUp(sizeof(T), 4096), 4096, flags, agent_node_id));
    if (ret == nullptr) throw std::bad_alloc();

    MAKE_NAMED_SCOPE_GUARD(throwGuard, [&]() { free_()(ret); });

    new (ret) T;

    throwGuard.Dismiss();
    return ret;
  }

  __forceinline static void free(T* ptr) {
    if (ptr != nullptr) {
      ptr->~T();
      free_()(ptr);
    }
  }
};

/// @brief Container for object located in GPU visible host memory.
/// If a custom allocator is not given then data will be placed in dedicated pages.
template <typename T, typename Allocator = PageAllocator<T>>
class Shared final : private BaseShared {
 public:
  explicit Shared(Allocator* pool = nullptr, int flags = 0) : pool_(pool) {
    assert(allocate_() != nullptr && free_() != nullptr &&
           "Shared object allocator is not set");

    if (pool_)
      shared_object_ = pool_->alloc();
    else
      shared_object_ = PageAllocator<T>::alloc(flags);
  }

  explicit Shared(int agent_node_id, Allocator* pool = nullptr, int flags = 0) : pool_(pool) {
    assert(allocate_() != nullptr && free_() != nullptr &&
           "Shared object allocator is not set");

    if (pool_)
      shared_object_ = pool_->alloc();
    else
      shared_object_ = PageAllocator<T>::alloc(agent_node_id, flags);
  }

  ~Shared() {
    assert(allocate_() != nullptr && free_() != nullptr &&
                                        "Shared object allocator is not set");

    if (pool_)
      pool_->free(shared_object_);
    else
      PageAllocator<T>::free(shared_object_);
  }

  Shared(Shared&& rhs) {
    this->~Shared();
    shared_object_ = rhs.shared_object_;
    rhs.shared_object_ = nullptr;
    pool_ = rhs.pool_;
    rhs.pool_ = nullptr;
  }
  Shared& operator=(Shared&& rhs) {
    this->~Shared();
    shared_object_ = rhs.shared_object_;
    rhs.shared_object_ = nullptr;
    pool_ = rhs.pool_;
    rhs.pool_ = nullptr;
    return *this;
  }

  T* shared_object() const { return shared_object_; }

 private:
  T* shared_object_;
  Allocator* pool_;
};

template <typename T> class Shared<T, PageAllocator<T>> final : private BaseShared {
 public:
  Shared(int flags = 0) {
    assert(allocate_() != nullptr && free_() != nullptr &&
                                        "Shared object allocator is not set");

    shared_object_ = PageAllocator<T>::alloc(flags);
  }

  Shared(int agent_node_id, int flags) {
    assert(allocate_() != nullptr && free_() != nullptr && "Shared object allocator is not set");

    shared_object_ = PageAllocator<T>::alloc(agent_node_id, flags);
  }

  ~Shared() {
    assert(allocate_() != nullptr && free_() != nullptr &&
           "Shared object allocator is not set");

    PageAllocator<T>::free(shared_object_);
  }

  Shared(Shared&& rhs) {
    this->~Shared();
    shared_object_ = rhs.shared_object_;
    rhs.shared_object_ = nullptr;
  }
  Shared& operator=(Shared&& rhs) {
    this->~Shared();
    shared_object_ = rhs.shared_object_;
    rhs.shared_object_ = nullptr;
    return *this;
  }

  T* shared_object() const { return shared_object_; }

 private:
  T* shared_object_;
};

/// @brief Container for array located in GPU visible host memory.
/// Alignment defaults to __alignof(T) but may be increased.
template <typename T, size_t Align> class SharedArray final : private BaseShared {
 public:
  SharedArray() : shared_object_(nullptr) {}

  explicit SharedArray(size_t length) : shared_object_(nullptr), len(length) {
    assert(allocate_() != nullptr && free_() != nullptr &&
                                        "Shared object allocator is not set");
    static_assert((__alignof(T) <= Align) || (Align == 0), "Align is less than alignof(T)");

    shared_object_ =
        reinterpret_cast<T*>(allocate_()(sizeof(T) * length, Max(__alignof(T), Align), 0, 0));
    if (shared_object_ == nullptr) throw std::bad_alloc();

    size_t i = 0;

    MAKE_NAMED_SCOPE_GUARD(loopGuard, [&]() {
      for (size_t t = 0; t < i - 1; t++) shared_object_[t].~T();
      free_()(shared_object_);
    });

    for (; i < length; i++) new (&shared_object_[i]) T;

    loopGuard.Dismiss();
  }

  ~SharedArray() {
    assert(allocate_() != nullptr && free_() != nullptr &&
                                        "Shared object allocator is not set");

    if (shared_object_ != nullptr) {
      for (size_t i = 0; i < len; i++) shared_object_[i].~T();
      free_()(shared_object_);
    }
  }

  SharedArray(SharedArray&& rhs) {
    this->~SharedArray();
    shared_object_ = rhs.shared_object_;
    rhs.shared_object_ = nullptr;
    len = rhs.len;
  }
  SharedArray& operator=(SharedArray&& rhs) {
    this->~SharedArray();
    shared_object_ = rhs.shared_object_;
    rhs.shared_object_ = nullptr;
    len = rhs.len;
    return *this;
  }

  T& operator[](size_t index) {
    assert(index < len && "Index out of bounds.");
    return shared_object_[index];
  }
  const T& operator[](size_t index) const {
    assert(index < len && "Index out of bounds.");
    return shared_object_[index];
  }

 private:
  T* shared_object_;
  size_t len;
};

}  // namespace core
}  // namespace rocr
#endif  // header guard
