// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Windows stand-ins for the KFD sources excluded from this build. The AMD kernel-mode
// driver is reached through D3DKMT on Windows rather than /dev/kfd ioctls, so none of
// the pool, copy queue, or dispatch log machinery exists here. These entry points are
// only reachable once an agent has been enumerated with a KFD handle, which cannot
// happen in this build.

#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
#include "lib/rocprofiler-sdk/kfd/resource.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"

#include <cstddef>
#include <cstdint>

namespace rocprofiler
{
namespace kfd
{
// stands in for resource.cpp
void* kfd_memory_pool_t::allocate(size_t, kfd_memory_kind_t, size_t) { return nullptr; }

void
kfd_memory_pool_t::deallocate(void*)
{}

bool
kfd_memory_pool_t::is_device_pointer(const void*) const
{
    return false;
}

bool
kfd_copy_queue_t::copy(void*, const void*, size_t)
{
    return false;
}

// stands in for signal_less.cpp
bool signal_less_id_is_leaked(uint64_t) { return false; }

// stands in for kfd_profiler.cpp
void
arm_dispatch_log_sessions()
{}
}  // namespace kfd
}  // namespace rocprofiler
