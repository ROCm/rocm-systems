/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/** \section rocprofiler_plugin_api ROCProfiler Plugin API
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
 * \file
 * ROCProfiler Tool Plugin API interface.
 */

#ifndef ROCPROFILER_PLUGIN_H_
#define ROCPROFILER_PLUGIN_H_

#include <stdint.h>

#include "rocprofiler.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** \defgroup rocprofiler_plugins ROCProfiler Plugin API Specification
 * @{
 */

/** \defgroup initialization_group Initialization and Finalization
 * \ingroup rocprofiler_plugins
 *
 * The ROCProfiler Plugin API must be initialized before using any of the
 * operations to report trace data, and finalized after the last trace data has
 * been reported.
 *
 * @{
 */

/**
 * Initialize plugin.
 * Must be called before any other operation.
 *
 * @param[in] rocprofiler_major_version The major version of the ROCProfiler API
 * being used by the ROCProfiler Tool. An error is reported if this does not
 * match the major version of the ROCProfiler API used to build the plugin
 * library. This ensures compatibility of the trace data format.
 * @param[in] rocprofiler_minor_version The minor version of the ROCProfiler API
 * being used by the ROCProfiler Tool. An error is reported if the
 * \p rocprofiler_major_version matches and this is greater than the minor
 * version of the ROCProfiler API used to build the plugin library. This ensures
 * compatibility of the trace data format.
 * @param[in] data Pointer to the data passed to the ROCProfiler Plugin by the tool
 * @return Returns 0 on success and -1 on error.
 */
ROCPROFILER_EXPORT int rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                                                     uint32_t rocprofiler_minor_version,
                                                     void* data);

/**
 * Finalize plugin.
 * This must be called after ::rocprofiler_plugin_initialize and after all
 * profiling data has been reported by
 * ::rocprofiler_plugin_write_kernel_records
 */
ROCPROFILER_EXPORT void rocprofiler_plugin_finalize();

/** @} */

/** \defgroup profiling_record_write_functions Profiling data reporting
 * \ingroup rocprofiler_plugins
 * Operations to output profiling data.
 * @{
 */

// TODO(aelwazir): Recheck wording of the description

/**
 * Report Buffer Records.
 *
 * @param[in] begin Pointer to the first record.
 * @param[in] end Pointer to one past the last record.
 * @param[in] session_id Session ID
 * @param[in] buffer_id Buffer ID
 * @return Returns 0 on success and -1 on error.
 */
ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id);

/**
 * Report Synchronous Record.
 *
 * @param[in] record Synchronous Tracer record.
 * @param[in] data : api_data
 * @param[in] tracer_data :Tracer record extra data such as function name and kernel name
 * @return Returns 0 on success and -1 on error.
 */

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record);

/** @} */

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* ROCPROFILER_PLUGIN_H_ */
