////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// ROC Profiler API
//
// The goal of the implementation is to provide a HW specific low-level
// performance analysis interface for profiling of GPU compute applications.
// The profiling includes HW performance counters (PMC) with complex
// performance metrics and thread traces (SQTT). The profiling is supported
// by the SQTT, PMC and Callback APIs.
//
// The library can be used by a tool library loaded by HSA runtime or by
// higher level HW independent performance analysis API like PAPI.
//
// The library is written on C and will be based on AQLprofile AMD specific
// HSA extension. The library implementation requires HSA API intercepting and
// a profiling queue supporting a submit callback interface.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef INC_ROCPROFILER_H_
#define INC_ROCPROFILER_H_

#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ven_amd_aqlprofile.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

////////////////////////////////////////////////////////////////////////////////
// Profiling info
//
// Profiling info objects have profiling feature info, type, parameters and data
// Also profiling data samplaes can be iterated using a callback

// Profiling feature type
typedef enum {
    ROCPROFILER_TYPE_METRIC = 0,
    ROCPROFILER_TYPE_TRACE = 1
} rocprofiler_type_t;

// Profiling feture parameter
typedef hsa_ven_amd_aqlprofile_parameter_t rocprofiler_parameter_t;

// Profiling data kind
typedef enum {
    ROCPROFILER_UNINIT = 0,
    ROCPROFILER_INT32 = 1,
    ROCPROFILER_INT64 = 2,
    ROCPROFILER_FLOAT = 3,
    ROCPROFILER_DOUBLE = 4,
    ROCPROFILER_BYTES = 5
} rocprofiler_metric_kind_t;

// Profiling data type
typedef struct {
  rocprofiler_metric_kind_t kind;  // result kind
  union {
    uint32_t result_int32;     // 32bit integer result
    uint64_t result_int64;     // 64bit integer result
    float result_float;    // float single-precision result
    double result_double;  // float double-precision result
    struct {
      void* ptr;
      uint32_t size;
      uint32_t instance_count;
      bool copy;
    } result_bytes;         // data by ptr and byte size
  };
} rocprofiler_data_t;

// Profiling feature info 
typedef struct {
    rocprofiler_type_t type;                          // feature type
    const char* name;                                 // [in] feature name
    const rocprofiler_parameter_t* parameters;        // feature parameters array
    uint32_t parameter_count;                         // feature parameters count
    rocprofiler_data_t data;                          // [out] profiling data
} rocprofiler_info_t;

////////////////////////////////////////////////////////////////////////////////
// Profiling context
//
// Profiling context object accumuate all profiling information

// Profiling context object
typedef void rocprofiler_t;

// Profiling group object
typedef struct {
  unsigned index; // group index
  rocprofiler_info_t** info; // profiling info array
  uint32_t info_count; // profiling info count
  rocprofiler_t* context; // context object
} rocprofiler_group_t;

// Profiling mode
typedef enum {
    ROCPROFILER_MODE_STANDALONE = 1,
    ROCPROFILER_MODE_CREATEQUEUE = 2,
} rocprofiler_mode_t;

// Profiling preperties
typedef struct {
    hsa_queue_t* queue;                      // queue for STANDALONE mode
                                             // the queue is created and returned in CREATEQUEUE mode
    uint32_t queue_depth;                    // created queue depth
} rocprofiler_properties_t;

// Create new profiling context
hsa_status_t rocprofiler_open(
    unsigned gpu_index, // GPU index
    rocprofiler_info_t* info, // [in] profiling info array
    uint32_t info_count, // profiling info count
    rocprofiler_t** context, // [out] context object
    uint32_t mode, // profiling mode mask
    rocprofiler_properties_t* properties); // profiling properties

// Delete profiling info
hsa_status_t rocprofiler_close(
    rocprofiler_t* context); // [in] profiling context

////////////////////////////////////////////////////////////////////////////////
// Runtime API observer
//
// Runtime API observer is called on enter and exit for the API

// Profiling callback data
typedef struct {
    uint64_t kernel_object;
    uint64_t queue_index;
    uint32_t gpu_index;
} rocprofiler_callback_data_t;

// Profiling callback type
typedef hsa_status_t (*rocprofiler_callback_t)(
    const rocprofiler_callback_data_t* callback_data, // [in] callback data union, data depends on
                                                      // the callback API id
    void* user_data,                                  // [in/out] user data passed to the callback 
    rocprofiler_group_t** group);                     // [out] profiling group

// Provided standard profiling callback
static inline hsa_status_t rocprofiler_set_dispatch_callback(
    const rocprofiler_callback_data_t* callback_data,
    void* user_data,
    rocprofiler_group_t** group) {
  *group = reinterpret_cast<rocprofiler_group_t*>(user_data);
  return HSA_STATUS_SUCCESS;
}

// Set/remove kernel dispatch observer
hsa_status_t rocprofiler_set_dispatch_observer(
    rocprofiler_callback_t callback, // observer callback
    void* data);                     // [in/out] passed callback data

hsa_status_t rocprofiler_remove_dispatch_observer();

////////////////////////////////////////////////////////////////////////////////
// Start/stop profiling
//
// Start/stop the context profiling invocation, have to be as many as
// contect.invocations' to collect all profiling data

// Start profiling
hsa_status_t rocprofiler_start(
    rocprofiler_t* context, // [in/out] profiling context
    uint32_t group_index = 0); // group index

// Stop profiling
hsa_status_t rocprofiler_stop(
    rocprofiler_t* context, // [in/out] profiling context
    uint32_t group_index = 0); // group index

// Read profiling data
hsa_status_t rocprofiler_get_data(
    rocprofiler_t* context, // [in/out] profiling context
    uint32_t group_index = 0); // group index

// Get profiling groups
hsa_status_t rocprofiler_get_groups(
  rocprofiler_t* context, // [in] profiling context
  rocprofiler_group_t** groups, // [out] profiling groups
  uint32_t* group_count); // [out] profiling groups count

// Start profiling
hsa_status_t rocprofiler_start_group(
  rocprofiler_group_t* group); // [in/out] profiling group

// Stop profiling
hsa_status_t rocprofiler_stop_group(
  rocprofiler_group_t* group); // [in/out] profiling group

// Get profiling data
hsa_status_t rocprofiler_get_group_data(
  rocprofiler_group_t* group); // [in/out] profiling group

// Get metrics data
hsa_status_t rocprofiler_get_metrics_data(
  const rocprofiler_t* context); // [in/out] profiling context

// Definition of output data iterator callback
typedef hsa_ven_amd_aqlprofile_data_callback_t rocprofiler_trace_data_callback_t;

// Method for iterating the events output data
hsa_status_t rocprofiler_iterate_trace_data(
    rocprofiler_t* context,                     // [in] profiling context
    rocprofiler_trace_data_callback_t callback,  // [in] callback to iterate the output data
    void* data);                                      // [in/out] callback data 

////////////////////////////////////////////////////////////////////////////////
// Returning the error string method

hsa_status_t rocprofiler_error_string (
    const char** str); // [out] the API error string pointer returning

////////////////////////////////////////////////////////////////////////////////
// HSA-runtime tool on-load method
bool OnLoad(
    HsaApiTable* table,
    uint64_t runtime_version,
    uint64_t failed_tool_count,
    const char* const * failed_tool_names);

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCPROFILER_H_
