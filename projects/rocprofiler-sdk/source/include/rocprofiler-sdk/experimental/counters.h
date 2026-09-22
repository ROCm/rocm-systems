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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief (experimental) Apply a custom counter definition (YAML). This function must be called
 * before counter iteration functions like ::rocprofiler_iterate_agent_supported_counters if
 * custom counter definitions are to be used. This function will return an error if the definition
 * has already been loaded.
 * @param [in] yaml YAML string containing counter definitions
 * @param [in] size Size of the YAML string
 * @param [in] flags Flags to apply to the counter definition
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter definition applied
 * @retval ROCPROFILER_STATUS_ERROR if counter definition already loaded
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_load_counter_definition(const char*                yaml,
                                    size_t                     size,
                                    rocprofiler_counter_flag_t flags) ROCPROFILER_API;

ROCPROFILER_EXTERN_C_FINI
