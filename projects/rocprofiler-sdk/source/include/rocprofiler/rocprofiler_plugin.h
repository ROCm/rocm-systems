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

/** @section rocprofiler_plugin_api ROCProfiler Plugin API
 *
 * The ROCProfiler Plugin API is used by the ROCProfiler Tool to output all
 * profiling information. Different implementations of the ROCProfiler Plugin
 * API can be developed that output the data in different formats. The
 * ROCProfiler Tool can be configured to load a specific library that supports
 * the user desired format.
 *
 * The API is not thread safe. It is the responsibility of the ROCProfiler Tool
 * to ensure the operations are synchronized and not called concurrently. There
 * is no requirement for the ROCProfiler Tool to report trace data in any
 * specific order. If the format supported by plugin requires specific
 * ordering, it is the responsibility of the plugin implementation to perform
 * any necessary sorting.
 */

/**
 * @file
 * ROCProfiler Tool Plugin API interface.
 */

#pragma once

#include "rocprofiler/rocprofiler.h"

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup ROCPROFILER_PLUGINS ROCProfiler Plugin API Specification
 * @{
 */

/**
 * @defgroup INITIALIZATION_GROUP Initialization and Finalization
 * @brief The ROCProfiler Plugin API must be initialized before using any of the
 * operations to report trace data, and finalized after the last trace data has
 * been reported.
 * @ingroup ROCPROFILER_PLUGINS
 *
 * @{
 */

/**
 * @brief Initialize plugin. Must be called before any other operation.
 *
 * @param[in] rocprofiler_major_version The major version of the ROCProfiler API
 * being used by the ROCProfiler Tool. An error is reported if this does not
 * match the major version of the ROCProfiler API used to build the plugin
 * library. This ensures compatibility of the trace data format.
 * @param[in] rocprofiler_minor_version The minor version of the ROCProfiler API
 * being used by the ROCProfiler Tool. An error is reported if the
 * @p rocprofiler_major_version matches and this is greater than the minor
 * version of the ROCProfiler API used to build the plugin library. This ensures
 * compatibility of the trace data format.
 * @param[in] data Pointer to the data passed to the ROCProfiler Plugin by the tool
 * @return Returns 0 on success and -1 on error.
 */
ROCPROFILER_EXPORT int
rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                              uint32_t rocprofiler_minor_version,
                              void*    data);

/**
 * @brief Finalize plugin.
 * This must be called after ::rocprofiler_plugin_initialize and after all
 * profiling data has been reported by
 * rocprofiler_plugin_write_kernel_records
 */
ROCPROFILER_EXPORT void
rocprofiler_plugin_finalize();

/** @} */

/**
 * @defgroup profiling_record_write_functions Profiling data reporting
 * @brief Operations to output profiling data.
 * @ingroup ROCPROFILER_PLUGINS
 *
 * @{
 */

// TODO(aelwazir): Recheck wording of the description

/**
 * Report Buffer Records.
 *
 * @param[in] context_id context ID
 * @param[in] buffer_id Buffer ID
 * @param[in] headers Array of ::rocprofiler_record_header_t
 * @param[in] num_headers Number of ::rocprofiler_record_header_t entries in array
 * @return Returns 0 on success and -1 on error.
 */
ROCPROFILER_EXPORT int
rocprofiler_plugin_write_buffer_records(rocprofiler_context_id_t      context_id,
                                        rocprofiler_buffer_id_t       buffer_id,
                                        rocprofiler_record_header_t** headers,
                                        size_t                        num_headers);

/**
 * @brief Report Synchronous Record.
 *
 * @param[in] record Synchronous Tracer record.
 * @return Returns 0 on success and -1 on error.
 */

ROCPROFILER_EXPORT int
rocprofiler_plugin_write_record(rocprofiler_record_header_t record);

/** @} */
/** @} */

ROCPROFILER_EXTERN_C_FINI
