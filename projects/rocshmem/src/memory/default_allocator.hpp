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

#include <algorithm>
#include <cstring>
#include <string>

#include <hip/hip_runtime_api.h>

#include "envvar.hpp"
#include "log.hpp"
#include "hip_allocator.hpp"

namespace rocshmem {
  extern HIPAllocator *default_allocator_;

  static void set_default_allocator()
  {
    int hip_dev_id{};
    hipError_t err = hipGetDevice(&hip_dev_id);
    if (err != hipSuccess) {
      LOG_ERROR_ABORT("Could not get current device. Aborting");
    }

    char arch_name[256];
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, hip_dev_id);
    if (err != hipSuccess) {
      LOG_ERROR_ABORT("Could not get device properties. Aborting");
    }
    std::snprintf(arch_name, sizeof(arch_name), "%s", prop.gcnArchName);

    // Arch-based default: finegrained for RDNA3/RDNA4 (gfx1100, gfx1201),
    // uncached for everything else.
    bool arch_wants_finegrained =
        (strncmp(arch_name, "gfx1100", strlen("gfx1100")) == 0 ||
         strncmp(arch_name, "gfx1201", strlen("gfx1201")) == 0);

    std::string requested = envvar::heap_allocator_type.get_value();
    // Normalise to lower-case for case-insensitive matching.
    std::transform(requested.begin(), requested.end(), requested.begin(), ::tolower);

    // Resolve the effective allocator type.
    // Priority: envvar > arch default.
    enum class AllocChoice { finegrained, coarsegrained, uncached, vmm_posix, vmm_fabric };
    AllocChoice choice;

    if (requested.empty()) {
      choice = arch_wants_finegrained ? AllocChoice::finegrained : AllocChoice::uncached;
    } else if (requested == "finegrained") {
      choice = AllocChoice::finegrained;
    } else if (requested == "coarsegrained") {
      choice = AllocChoice::coarsegrained;
    } else if (requested == "uncached") {
      choice = AllocChoice::uncached;
    } else if (requested == "vmm_posix") {
      choice = AllocChoice::vmm_posix;
    } else if (requested == "vmm_fabric") {
      choice = AllocChoice::vmm_fabric;
    } else {
      LOG_ERROR_ABORT("Unknown ROCSHMEM_HEAP_ALLOCATOR_TYPE value: '%s'. "
                      "Accepted values: uncached, finegrained, coarsegrained, vmm_posix, vmm_fabric.",
                      requested.c_str());
      return;
    }

    // Fall back to finegrained if uncached is not available in this build.
    if (choice == AllocChoice::uncached) {
#if !defined HAVE_DEVICE_MALLOC_UNCACHED
      LOG_WARN("ROCSHMEM_HEAP_ALLOCATOR_TYPE=uncached is not supported in this build "
               "(requires ROCm 6.0+). Falling back to finegrained.");
      choice = AllocChoice::finegrained;
#endif
    }

    switch (choice) {
      case AllocChoice::finegrained:
        default_allocator_ = new HIPAllocatorFinegrained();
        break;
      case AllocChoice::coarsegrained:
        default_allocator_ = new HIPAllocatorCoarsegrained();
        break;
      case AllocChoice::uncached:
#if defined HAVE_DEVICE_MALLOC_UNCACHED
        default_allocator_ = new HIPAllocatorUncached();
#endif
        break;
      case AllocChoice::vmm_posix:
#if HIP_VERSION >= 70200000
        default_allocator_ = new HIPAllocatorVMMPosixFd();
#else
        LOG_ERROR_ABORT("ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_posix requires ROCm 7.2 or newer.");
#endif
        break;
      case AllocChoice::vmm_fabric:
#if HIP_VERSION >= 70200000 && defined HAVE_AMDSMI_GPU_FABRIC_INFO
        default_allocator_ = new HIPAllocatorVMMFabric();
#else
        LOG_ERROR_ABORT("ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_fabric requires ROCm 7.14+ "
                        "with AMD SMI fabric handle support.");
#endif
        break;
    }
  }

  [[maybe_unused]] static HIPAllocator* get_default_allocator()
  {
    if (default_allocator_ == nullptr) {
      set_default_allocator();
    }

    return default_allocator_;
  }

  [[maybe_unused]] static void delete_default_allocator()
  {
    if (default_allocator_ != nullptr) {
      delete default_allocator_;
    }
  }

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_DEFAULT_ALLOCATOR_HPP_
