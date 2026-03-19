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

#ifndef LIBRARY_SRC_MEMORY_DEFAULT_ALLOCATOR_HPP_
#define LIBRARY_SRC_MEMORY_DEFAULT_ALLOCATOR_HPP_

#include <hip/hip_runtime_api.h>

#include "envvar.hpp"
#include "hip_allocator.hpp"

    // the using statements remain in the code until we commit
    // the change to make the default allocator a runtime decision.
#if defined USE_HEAP_DEVICE_COARSEGRAIN
using HIPDefaultFinegrainedAllocator = rocshmem::HIPAllocatorCoarsegrained;
#endif
#if defined USE_HEAP_DEVICE_FINEGRAIN
using HIPDefaultFinegrainedAllocator = rocshmem::HIPAllocatorFinegrained;
#endif

#if defined USE_HEAP_DEVICE_UNCACHED
#if defined HAVE_DEVICE_MALLOC_UNCACHED
using HIPDefaultFinegrainedAllocator = rocshmem::HIPAllocatorUncached;
#else
#error "USE_HEAP_DEVICE_UNCACHED unsupported in this HIP version"
#endif
#endif

#if defined USE_HEAP_DEVICE_VMM_POSIX
#if HIP_VERSION >= 70000000
using HIPDefaultFinegrainedAllocator = rocshmem::HIPAllocatorVMMPosixFd;
#else
#error "USE_HEAP_DEVICE_VMM_POSIX requires ROCm 7.0 or newer (HIP_VERSION >= 70000000)"
#endif
#endif

namespace rocshmem {
  extern HIPAllocator *default_allocator_;

  static void set_default_allocator()
  {
    int hip_dev_id{};
    hipError_t err = hipGetDevice(&hip_dev_id);
    if (err != hipSuccess) {
      printf("Could not get current device. Aborting\n");
      abort();
    }

    char arch_name[256];
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, hip_dev_id);
    if (err != hipSuccess) {
      printf("Could not get device properties. Aborting\n");
      abort();
    }
    std::snprintf(arch_name, sizeof(arch_name), "%s",prop.gcnArchName);

#if defined USE_HEAP_DEVICE_COARSEGRAIN
    default_allocator_ = new HIPAllocatorCoarsegrained();
#endif
#if defined USE_HEAP_DEVICE_FINEGRAIN
    default_allocator_ = new HIPAllocatorFinegrained();
#endif
#if defined USE_HEAP_DEVICE_UNCACHED
#if defined HAVE_DEVICE_MALLOC_UNCACHED
    // Temporary hack that will be fixed when we add the ability
    // to use an environment variable to set the Heap Allocatory Type.
    // With that commit we will introduce also a generic 'default'
    // setting that can be adjusted for differnt architectures.
    // This is to avoid failures with rocSHMEM on gfx1201 if using
    // Uncached allocator with ROCm 7.2.0
    if (strncmp(arch_name, "gfx1201", strlen("gfx1201")) == 0) {
      default_allocator_ = new HIPAllocatorFinegrained();
    } else {
      default_allocator_ = new HIPAllocatorUncached();
    }
#endif
#endif
#if defined USE_HEAP_DEVICE_VMM_POSIX
#if HIP_VERSION >= 70000000
    default_allocator_ = new HIPAllocatorVMMPosixFd();
#endif
#endif
  }

  static HIPAllocator* get_default_allocator()
  {
    if (default_allocator_ == nullptr) {
      set_default_allocator();
    }

    return default_allocator_;
  }

  static void delete_default_allocator()
  {
    if (default_allocator_ != nullptr) {
      delete default_allocator_;
    }
  }

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_DEFAULT_ALLOCATOR_HPP_
