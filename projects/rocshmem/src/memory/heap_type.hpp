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

#ifndef LIBRARY_SRC_MEMORY_HEAP_TYPE_HPP_
#define LIBRARY_SRC_MEMORY_HEAP_TYPE_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "hip_allocator.hpp"

/**
 * @file heap_type.hpp
 *
 * @brief Contains the type of heap memory used to allocate the symmetric heap
 *
 * The heap type in this file is used by other classes to select the allocation
 * policy for the symmetric heap. The heap type choices depend on whether
 * the heap is cacheable in hardware and if the heap is a managed memory type.
 */

namespace rocshmem {

// Compilation error 'HEAP_T redefined' indicates that user had more than one
// USE_HEAP_* ON when configuring. Use ccmake to select only one.
#if defined USE_HEAP_DEVICE_COARSEGRAIN
using HEAP_T = HeapMemory<HIPAllocator>;
#endif
#if defined USE_HEAP_DEVICE_FINEGRAIN
using HEAP_T = HeapMemory<HIPAllocatorFinegrained>;
#endif
#if defined USE_HEAP_DEVICE_UNCACHED
using HEAP_T = HeapMemory<HIPAllocatorUncached>;
#endif
#if defined USE_HEAP_MANAGED
using HEAP_T = HeapMemory<HIPAllocatorManaged>;
#endif
#if defined USE_HEAP_HOST_HIP
using HEAP_T = HeapMemory<HIPHostAllocator>;
#endif
#if defined USE_HEAP_HOST
using HEAP_T = HeapMemory<HostAllocator>;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_HEAP_TYPE_HPP_
