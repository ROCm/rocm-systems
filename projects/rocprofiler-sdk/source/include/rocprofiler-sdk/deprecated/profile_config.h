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

#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief (deprecated) Replaced by ::rocprofiler_create_counter_config.
 *
 */
ROCPROFILER_SDK_DEPRECATED("profile_config renamed to counter_config")
static inline rocprofiler_status_t
rocprofiler_create_profile_config(rocprofiler_agent_id_t           agent_id,
                                  rocprofiler_counter_id_t*        counters_list,
                                  size_t                           counters_count,
                                  rocprofiler_profile_config_id_t* config_id)
{
    return rocprofiler_create_counter_config(agent_id, counters_list, counters_count, config_id);
}

/**
 * @brief (deprecated) Replaced by ::rocprofiler_destroy_counter_config.
 *
 */
ROCPROFILER_SDK_DEPRECATED("profile_config renamed to counter_config")
static inline rocprofiler_status_t
rocprofiler_destroy_profile_config(rocprofiler_profile_config_id_t config_id)
{
    return rocprofiler_destroy_counter_config(config_id);
}

ROCPROFILER_EXTERN_C_FINI
