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

    std::string heap_mem_type = envvar::heap_mem_type;
    if (heap_mem_type.empty()) {
      // Note: not using get_arch_name(hip_dev_id) from ../util.cpp because 
      // the required data structure are not being initialized in the unit tests.
      char arch_name[256];
      hipDeviceProp_t prop;
      err = hipGetDeviceProperties(&prop, hip_dev_id);
    if (err != hipSuccess) {
      printf("Could not get device properties. Aborting\n");
      abort();
    }
      std::snprintf(arch_name, sizeof(arch_name), "%s",prop.gcnArchName);

      if (strncmp(arch_name, "gfx1201", strlen("gfx1201")) == 0) {
	default_allocator_ = new HIPAllocatorFinegrained();
      } else {
#if defined HAVE_DEVICE_MALLOC_UNCACHED
	default_allocator_ = new HIPAllocatorUncached();
#else
	default_allocator_ = new HIPAllocatorFinegrained();
#endif
      }
    } else {
      if (heap_mem_type.compare("coarsegrained") == 0) {
	default_allocator_ = new HIPAllocatorCoarsegrained();
      } else if (heap_mem_type.compare("finegrained") == 0) {
	default_allocator_ = new HIPAllocatorFinegrained();
      } else if (heap_mem_type.compare("uncached") == 0) {
#if defined HAVE_DEVICE_MALLOC_UNCACHED
	default_allocator_ = new HIPAllocatorUncached();
#else
	printf("Uncached Heap memory type requested, but ROCm version does not support Uncached memory. Aborting.\n");
	abort();
#endif
      }
    }
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
