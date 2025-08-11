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

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup SPM_SERVICE SPM Service
 * @brief Streaming Performance Monitoring
 *
 * @{
 */

/**
 * @brief (experimental) ROCProfiler SPM Record.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_record_t
{
    /**
     * Counters, including identifiers to get counter information and Counters
     * values
     */
    rocprofiler_counter_record_t* counters;
    uint64_t                      counters_count;
} rocprofiler_spm_record_t;

/**
 * @brief Configure SPM Service.
 *
 * @param [in] context_id
 * @param [in] buffer_id
 * @param [in] counter_config
 * @param [in] interval
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_spm_service(rocprofiler_context_id_t        context_id,
                                  rocprofiler_buffer_id_t         buffer_id,
                                  rocprofiler_counter_config_id_t counter_config,
                                  uint64_t                        interval) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
