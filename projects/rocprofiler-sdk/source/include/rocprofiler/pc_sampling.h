// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler/agent.h>
#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup PC_SAMPLING_SERVICE PC Sampling
 * @brief Enabling PC (Program Counter) Sampling for GPU Activity
 * @{
 */

/**
 * @brief Create PC Sampling Service.
 *
 * @param [in] context_id
 * @param [in] agent
 * @param [in] method
 * @param [in] unit
 * @param [in] interval
 * @param [in] buffer_id
 * @return ::rocprofiler_status_t
 *
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                          rocprofiler_agent_t              agent,
                                          rocprofiler_pc_sampling_method_t method,
                                          rocprofiler_pc_sampling_unit_t   unit,
                                          uint64_t                         interval,
                                          rocprofiler_buffer_id_t          buffer_id);

struct rocprofiler_pc_sampling_configuration_s
{
    rocprofiler_pc_sampling_method_t method;
    rocprofiler_pc_sampling_unit_t   unit;
    size_t                           min_interval;
    size_t                           max_interval;
    uint64_t                         flags;
};

/** @} */

ROCPROFILER_EXTERN_C_FINI
