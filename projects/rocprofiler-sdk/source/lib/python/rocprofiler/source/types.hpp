// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace python
{
/**
 * @brief Information about a hardware counter
 */
struct CounterInfo
{
    uint64_t    id          = 0;
    std::string name        = {};
    std::string description = {};
    std::string block       = {};
    std::string expression  = {};
    bool        is_constant = false;
    bool        is_derived  = false;
};

/**
 * @brief Information about a GPU agent (device)
 */
struct AgentInfo
{
    uint64_t    id           = 0;
    std::string name         = {};
    std::string product_name = {};
    int         device_index = 0;
    int         gfx_version  = 0;
};

/**
 * @brief Information about a kernel dispatch
 */
struct DispatchInfo
{
    uint64_t    dispatch_id     = 0;
    uint64_t    kernel_id       = 0;
    uint64_t    correlation_id  = 0;
    uint64_t    queue_id        = 0;
    uint64_t    agent_id        = 0;
    uint64_t    start_timestamp = 0;
    uint64_t    end_timestamp   = 0;
    std::string kernel_name     = {};
};

/**
 * @brief A single hardware counter measurement
 */
struct CounterRecord
{
    uint64_t                                    dispatch_id  = 0;
    uint64_t                                    counter_id   = 0;
    std::string                                 counter_name = {};
    std::string                                 kernel_name  = {};
    double                                      value        = 0.0;
    uint64_t                                    agent_id     = 0;
    std::vector<std::pair<std::string, size_t>> dimensions   = {};
};

/**
 * @brief Helper macro for rocprofiler status checking
 */
#define ROCPROFILER_PYTHON_CALL(result, msg)                                                       \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            throw std::runtime_error(std::string("[rocprofiler] ") + msg +                         \
                                     " failed: " + status_msg);                                    \
        }                                                                                          \
    } while(0)

}  // namespace python
}  // namespace rocprofiler
