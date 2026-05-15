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

#ifndef LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
#define LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_

/**
 * @file hip_allocator.hpp
 *
 * @brief Contains HIP wrapper class for memory allocator
 */

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "memory_allocator.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <limits>

// `hipDeviceMallocUncached` was introduced at ROCm 5.5
#if (HIP_VERSION_MAJOR > 5) || \
    (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 5)
#define HIP_SUPPORTS_MALLOC_UNCACHED
#elif defined USE_HEAP_DEVICE_UNCACHED
#error "USE_HEAP_DEVICE_UNCACHED unsupported in this HIP version"
#endif
namespace rocshmem {

class HIPAllocator : public MemoryAllocator {
 public:
  HIPAllocator() : MemoryAllocator(hipMalloc, hipFree) {}
};

class HIPAllocatorFinegrained : public MemoryAllocator {
 public:
  HIPAllocatorFinegrained()
      : MemoryAllocator(hipExtMallocWithFlags, hipFree,
                        hipDeviceMallocFinegrained) {}
};

#if defined HIP_SUPPORTS_MALLOC_UNCACHED
class HIPAllocatorUncached : public MemoryAllocator {
 public:
  HIPAllocatorUncached()
      : MemoryAllocator(hipExtMallocWithFlags, hipFree,
                        hipDeviceMallocUncached) {}
};

// The default fine-grained coherence allocator is the uncached allocator
using HIPDefaultFinegrainedAllocator = HIPAllocatorUncached;
#else
// The default fine-grained coherence allocator is the fine-grained allocator
using HIPDefaultFinegrainedAllocator = HIPAllocatorFinegrained;
#endif

class HIPAllocatorManaged : public MemoryAllocator {
 public:
  HIPAllocatorManaged()
      : MemoryAllocator(hipMallocManaged, hipFree, hipMemAttachHost) {
    _managed = true;
  }
};

class HIPHostAllocator : public MemoryAllocator {
 public:
  HIPHostAllocator()
      : MemoryAllocator(hipHostMalloc, hipFree, hipHostMallocCoherent) {}
};

class HostAllocator : public MemoryAllocator {
 public:
  HostAllocator() : MemoryAllocator(std::malloc, std::free) {}
};

class PosixAligned64Allocator : public MemoryAllocator {
 public:
  PosixAligned64Allocator() : MemoryAllocator(posix_memalign, std::free, 64) {}
};

template <class T>
class StdAllocatorHIP {
 public:
  typedef T value_type;

  StdAllocatorHIP() = default;

  template <class U>
  constexpr StdAllocatorHIP(const StdAllocatorHIP<U>&) noexcept {}

  [[nodiscard]] T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    T* p{nullptr};
    allocator_.allocate(reinterpret_cast<void**>(&p), n * sizeof(T));
    if (p) {
      return p;
    }

    throw std::bad_alloc();
  }

  void deallocate(T* p, [[maybe_unused]] size_t n) noexcept {
    allocator_.deallocate(p);
  }

 private:
  HIPDefaultFinegrainedAllocator allocator_{};
};

template <class T, class U>
bool operator==(const StdAllocatorHIP<T>&, const StdAllocatorHIP<U>&) {
  return true;
}

template <class T, class U>
bool operator!=(const StdAllocatorHIP<T>&, const StdAllocatorHIP<U>&) {
  return false;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
