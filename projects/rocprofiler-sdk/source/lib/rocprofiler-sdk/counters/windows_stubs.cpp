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

// Windows stand-ins for the counter sources excluded from this build. Device counting
// arms the hardware through the KFD ioctl interface, which Windows reaches via D3DKMT
// instead, and the firmware restriction table is read from a YAML file shipped with the
// Linux packages. Reporting no device lock keeps callers on the path that does not
// assume a lock was taken.

#include "lib/rocprofiler-sdk/counters/device_counting.hpp"
#include "lib/rocprofiler-sdk/counters/firmware_restrictions.hpp"
#include "lib/rocprofiler-sdk/counters/ioctl.hpp"

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace counters
{
// stands in for device_counting.cpp
agent_callback_data::~agent_callback_data() = default;

rocprofiler_status_t
start_agent_ctx(const context::context*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
stop_agent_ctx(const context::context*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

// stands in for ioctl.cpp
bool
counter_collection_has_device_lock()
{
    return false;
}

rocprofiler_status_t
counter_collection_device_lock(const rocprofiler_agent_t*, bool)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

bool
ptl_control_supported()
{
    return false;
}

bool
use_device_lock_at_start()
{
    return false;
}

rocprofiler_status_t
counter_collection_ptl_disable(const rocprofiler_agent_t*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

// stands in for firmware_restrictions.cpp
bool
check_installed_firmware_restrictions()
{
    return true;
}
}  // namespace counters
}  // namespace rocprofiler
