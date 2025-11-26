// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

/// @file gmock_interfaces.hpp
/// @brief Virtual interfaces used ONLY for GMock-based unit testing.
/// Production code uses zero-overhead static policies from policies.hpp.

#include "core/state.hpp"
#include "core/trace_cache/sample_type.hpp"

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

#include <cstdint>

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{
#if ROCPROFSYS_USE_ROCM > 0

/// @brief Interface for AMD SMI driver (for GMock only)
class IDriver
{
public:
    virtual ~IDriver() = default;

    virtual amdsmi_status_t get_gpu_activity(amdsmi_processor_handle handle,
                                             amdsmi_engine_usage_t*  usage) = 0;

    virtual amdsmi_status_t get_temp_metric(amdsmi_processor_handle     handle,
                                            amdsmi_temperature_type_t   type,
                                            amdsmi_temperature_metric_t metric,
                                            int64_t*                    temp) = 0;

    virtual amdsmi_status_t get_power_info(amdsmi_processor_handle handle,
                                           amdsmi_power_info_t*    power) = 0;

    virtual amdsmi_status_t get_gpu_memory_usage(amdsmi_processor_handle handle,
                                                 amdsmi_memory_type_t    type,
                                                 uint64_t*               usage) = 0;

    virtual amdsmi_status_t get_gpu_metrics_info(amdsmi_processor_handle handle,
                                                 amdsmi_gpu_metrics_t*   metrics) = 0;

    virtual amdsmi_status_t status_to_string(amdsmi_status_t status,
                                             const char**    msg) = 0;

    virtual amdsmi_processor_handle get_handle(uint32_t device_id) = 0;
};

/// @brief Interface for clock (for GMock only)
class IClock
{
public:
    virtual ~IClock()       = default;
    virtual size_t now_ns() = 0;
};

/// @brief Interface for sample storage (for GMock only)
class IStorage
{
public:
    virtual ~IStorage() = default;
    virtual void store(const trace_cache::amd_smi_sample& sample) = 0;
};

/// @brief Interface for state management (for GMock only)
class IState
{
public:
    virtual ~IState()                  = default;
    virtual State get_state() const    = 0;
    virtual void  set_state(State s)   = 0;
    virtual bool  is_child_process() const = 0;
};

/// @brief Interface for GPU capabilities (for GMock only)
class IGpuCapabilities
{
public:
    virtual ~IGpuCapabilities() = default;
    virtual bool vcn_is_device_level_only(uint32_t dev_id) = 0;
    virtual bool jpeg_is_device_level_only(uint32_t dev_id) = 0;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys
