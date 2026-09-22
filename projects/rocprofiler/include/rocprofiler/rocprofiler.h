/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//
// ROC Profiler API
//
// The goal of the implementation is to provide a HW specific low-level
// performance analysis interface for profiling of GPU compute applications.
// The profiling includes HW performance counters (PMC) with complex
// performance metrics and traces.
//
// The library can be used by a tool library loaded by HSA runtime or by
// higher level HW independent performance analysis API like PAPI.
//
// The library is written on C and will be based on AQLprofile AMD specific
// HSA extension. The library implementation requires HSA API intercepting and
// a profiling queue supporting a submit callback interface.
//
//

#ifndef INC_ROCPROFILER_H_
#define INC_ROCPROFILER_H_

/* Placeholder for calling convention and import/export macros */
#if !defined(ROCPROFILER_CALL)
#define ROCPROFILER_CALL
#endif /* !defined (ROCPROFILER_CALL) */

#if !defined(ROCPROFILER_EXPORT_DECORATOR)
#if defined(__GNUC__)
#define ROCPROFILER_EXPORT_DECORATOR __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define ROCPROFILER_EXPORT_DECORATOR __declspec(dllexport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (ROCPROFILER_EXPORT_DECORATOR) */

#if !defined(ROCPROFILER_IMPORT_DECORATOR)
#if defined(__GNUC__)
#define ROCPROFILER_IMPORT_DECORATOR
#elif defined(_MSC_VER)
#define ROCPROFILER_IMPORT_DECORATOR __declspec(dllimport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (ROCPROFILER_IMPORT_DECORATOR) */

#define ROCPROFILER_EXPORT ROCPROFILER_EXPORT_DECORATOR ROCPROFILER_CALL
#define ROCPROFILER_IMPORT ROCPROFILER_IMPORT_DECORATOR ROCPROFILER_CALL

#if !defined(ROCPROFILER)
#if defined(ROCPROFILER_EXPORTS)
#define ROCPROFILER_API ROCPROFILER_EXPORT
#else /* !defined (ROCPROFILER_EXPORTS) */
#define ROCPROFILER_API ROCPROFILER_IMPORT
#endif /* !defined (ROCPROFILER_EXPORTS) */
#endif /* !defined (ROCPROFILER) */


#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <stdint.h>


#define ROCPROFILER_VERSION_MAJOR 8
#define ROCPROFILER_VERSION_MINOR 0

////////////////////////////////////////////////////////////////////////////////
// Returning library version
uint32_t rocprofiler_version_major();
uint32_t rocprofiler_version_minor();

////////////////////////////////////////////////////////////////////////////////
// Global properties structure

typedef struct {
  uint32_t intercept_mode;
  uint32_t code_obj_tracking;
  uint32_t memcopy_tracking;
  uint32_t trace_size;
  uint32_t trace_local;
  uint64_t timeout;
  uint32_t timestamp_on;
  uint32_t hsa_intercepting;
  uint32_t k_concurrent;
  uint32_t opt_mode;
  uint32_t obj_dumping;
} rocprofiler_settings_t;

////////////////////////////////////////////////////////////////////////////////
// Returning the error string method

hsa_status_t rocprofiler_error_string(
    const char** str);  // [out] the API error string pointer returning

////////////////////////////////////////////////////////////////////////////////
// Profiling features and data
//
// Profiling features objects have profiling feature info, type, parameters and data
// Also profiling data samplaes can be iterated using a callback

// Profiling feature kind
typedef enum {
  ROCPROFILER_FEATURE_KIND_METRIC = 0,
  ROCPROFILER_FEATURE_KIND_TRACE = 1,
  ROCPROFILER_FEATURE_KIND_SPM_MOD = 2,
  ROCPROFILER_FEATURE_KIND_PCSMP_MOD = 4
} rocprofiler_feature_kind_t;

// Profiling feture parameter
typedef hsa_ven_amd_aqlprofile_parameter_t rocprofiler_parameter_t;

// Profiling data kind
typedef enum {
  ROCPROFILER_DATA_KIND_UNINIT = 0,
  ROCPROFILER_DATA_KIND_INT32 = 1,
  ROCPROFILER_DATA_KIND_INT64 = 2,
  ROCPROFILER_DATA_KIND_FLOAT = 3,
  ROCPROFILER_DATA_KIND_DOUBLE = 4,
  ROCPROFILER_DATA_KIND_BYTES = 5
} rocprofiler_data_kind_t;

// Profiling data type
typedef struct {
  rocprofiler_data_kind_t kind;  // result kind
  union {
    uint32_t result_int32;  // 32bit integer result
    uint64_t result_int64;  // 64bit integer result
    float result_float;     // float single-precision result
    double result_double;   // float double-precision result
    struct {
      void* ptr;
      uint32_t size;
      uint32_t instance_count;
      bool copy;
    } result_bytes;  // data by ptr and byte size
  };
} rocprofiler_data_t;

// Profiling feature type
typedef struct {
  rocprofiler_feature_kind_t kind;  // feature kind
  union {
    const char* name;  // feature name
    struct {
      const char* block;  // counter block name
      uint32_t event;     // counter event id
    } counter;
  };
  const rocprofiler_parameter_t* parameters;  // feature parameters array
  uint32_t parameter_count;                   // feature parameters count
  rocprofiler_data_t data;                    // profiling data
} rocprofiler_feature_t;

// Profiling features set type
typedef void rocprofiler_feature_set_t;

////////////////////////////////////////////////////////////////////////////////
// Profiling context
//
// Profiling context object accumuate all profiling information

// Profiling context object
typedef void rocprofiler_t;

// Profiling group object
typedef struct {
  unsigned index;                    // group index
  rocprofiler_feature_t** features;  // profiling info array
  uint32_t feature_count;            // profiling info count
  rocprofiler_t* context;            // context object
} rocprofiler_group_t;

// Profiling mode mask
typedef enum {
  ROCPROFILER_MODE_STANDALONE = 1,   // standalone mode when ROC profiler supports a queue
  ROCPROFILER_MODE_CREATEQUEUE = 2,  // ROC profiler creates queue in standalone mode
  ROCPROFILER_MODE_SINGLEGROUP = 4   // only one group is allowed, failed otherwise
} rocprofiler_mode_t;

// Profiling handler, calling on profiling completion
typedef bool (*rocprofiler_handler_t)(rocprofiler_group_t group, void* arg);

// Profiling preperties
typedef struct {
  hsa_queue_t* queue;             // queue for STANDALONE mode
                                  // the queue is created and returned in CREATEQUEUE mode
  uint32_t queue_depth;           // created queue depth
  rocprofiler_handler_t handler;  // handler on completion
  void* handler_arg;              // the handler arg
} rocprofiler_properties_t;

// Create new profiling context
hsa_status_t rocprofiler_open(hsa_agent_t agent,                // GPU handle
                              rocprofiler_feature_t* features,  // [in] profiling features array
                              uint32_t feature_count,           // profiling info count
                              rocprofiler_t** context,          // [out] context object
                              uint32_t mode,                    // profiling mode mask
                              rocprofiler_properties_t* properties);  // profiling properties

// Add feature to a features set
hsa_status_t rocprofiler_add_feature(
    const rocprofiler_feature_t* feature,      // [in]
    rocprofiler_feature_set_t* features_set);  // [in/out] profiling features set

// Create new profiling context
hsa_status_t rocprofiler_features_set_open(
    hsa_agent_t agent,                        // GPU handle
    rocprofiler_feature_set_t* features_set,  // [in] profiling features set
    rocprofiler_t** context,                  // [out] context object
    uint32_t mode,                            // profiling mode mask
    rocprofiler_properties_t* properties);    // profiling properties

// Delete profiling info
hsa_status_t rocprofiler_close(rocprofiler_t* context);  // [in] profiling context

// Context reset before reusing
hsa_status_t rocprofiler_reset(rocprofiler_t* context,  // [in] profiling context
                               uint32_t group_index);   // group index

// Return context agent
hsa_status_t rocprofiler_get_agent(rocprofiler_t* context,  // [in] profiling context
                                   hsa_agent_t* agent);     // [out] GPU handle

// Supported time value ID
typedef enum {
  ROCPROFILER_TIME_ID_CLOCK_REALTIME = 0,          // Linux realtime clock time
  ROCPROFILER_TIME_ID_CLOCK_REALTIME_COARSE = 1,   // Linux realtime-coarse clock time
  ROCPROFILER_TIME_ID_CLOCK_MONOTONIC = 2,         // Linux monotonic clock time
  ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_COARSE = 3,  // Linux monotonic-coarse clock time
  ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_RAW = 4,     // Linux monotonic-raw clock time
} rocprofiler_time_id_t;

// Return time value for a given time ID and profiling timestamp
hsa_status_t rocprofiler_get_time(
    rocprofiler_time_id_t time_id,  // identifier of the particular time to convert the timesatmp
    uint64_t timestamp,             // profiling timestamp
    uint64_t* value_ns,             // [out] returned time 'ns' value, ignored if NULL
    uint64_t* error_ns);            // [out] returned time error 'ns' value, ignored if NULL

////////////////////////////////////////////////////////////////////////////////
// Queue callbacks
//
// Queue callbacks for initiating profiling per kernel dispatch and to wait
// the profiling data on the queue destroy.

// Dispatch record
typedef struct {
  uint64_t dispatch;  // dispatch timestamp, ns
  uint64_t begin;     // kernel begin timestamp, ns
  uint64_t end;       // kernel end timestamp, ns
  uint64_t complete;  // completion signal timestamp, ns
} rocprofiler_dispatch_record_t;

// Profiling callback data
typedef struct {
  hsa_agent_t agent;         // GPU agent handle
  uint32_t agent_index;      // GPU index (GPU Driver Node ID as reported in the sysfs topology)
  const hsa_queue_t* queue;  // HSA queue
  uint64_t queue_index;      // Index in the queue
  uint32_t queue_id;         // Queue id
  hsa_signal_t completion_signal;               // Completion signal
  const hsa_kernel_dispatch_packet_t* packet;   // HSA dispatch packet
  const char* kernel_name;                      // Kernel name
  uint64_t kernel_object;                       // Kernel object address
  const amd_kernel_code_t* kernel_code;         // Kernel code pointer
  uint32_t thread_id;                           // Thread id
  const rocprofiler_dispatch_record_t* record;  // Dispatch record
} rocprofiler_callback_data_t;

// Profiling callback type
typedef hsa_status_t (*rocprofiler_callback_t)(
    const rocprofiler_callback_data_t* callback_data,  // [in] callback data
    void* user_data,                                   // [in/out] user data passed to the callback
    rocprofiler_group_t* group);                       // [out] returned profiling group

// Queue callbacks
typedef struct {
  rocprofiler_callback_t dispatch;                          // dispatch callback
  hsa_status_t (*create)(hsa_queue_t* queue, void* data);   // create callback
  hsa_status_t (*destroy)(hsa_queue_t* queue, void* data);  // destroy callback
} rocprofiler_queue_callbacks_t;

// Set queue callbacks
hsa_status_t rocprofiler_set_queue_callbacks(rocprofiler_queue_callbacks_t callbacks,  // callbacks
                                             void* data);  // [in/out] passed callbacks data

// Remove queue callbacks
hsa_status_t rocprofiler_remove_queue_callbacks();

// Start/stop queue callbacks
hsa_status_t rocprofiler_start_queue_callbacks();
hsa_status_t rocprofiler_stop_queue_callbacks();

////////////////////////////////////////////////////////////////////////////////
// Start/stop profiling
//
// Start/stop the context profiling invocation, have to be as many as
// contect.invocations' to collect all profiling data

// Start profiling
hsa_status_t rocprofiler_start(rocprofiler_t* context,  // [in/out] profiling context
                               uint32_t group_index);   // group index

// Stop profiling
hsa_status_t rocprofiler_stop(rocprofiler_t* context,  // [in/out] profiling context
                              uint32_t group_index);   // group index

// Read profiling
hsa_status_t rocprofiler_read(rocprofiler_t* context,  // [in/out] profiling context
                              uint32_t group_index);   // group index

// Read profiling data
hsa_status_t rocprofiler_get_data(rocprofiler_t* context,  // [in/out] profiling context
                                  uint32_t group_index);   // group index

// Get profiling groups count
hsa_status_t rocprofiler_group_count(const rocprofiler_t* context,  // [in] profiling context
                                     uint32_t* group_count);        // [out] profiling groups count

// Get profiling group for a given index
hsa_status_t rocprofiler_get_group(rocprofiler_t* context,       // [in] profiling context
                                   uint32_t group_index,         // profiling group index
                                   rocprofiler_group_t* group);  // [out] profiling group

// Start profiling
hsa_status_t rocprofiler_group_start(rocprofiler_group_t* group);  // [in/out] profiling group

// Stop profiling
hsa_status_t rocprofiler_group_stop(rocprofiler_group_t* group);  // [in/out] profiling group

// Read profiling
hsa_status_t rocprofiler_group_read(rocprofiler_group_t* group);  // [in/out] profiling group

// Get profiling data
hsa_status_t rocprofiler_group_get_data(rocprofiler_group_t* group);  // [in/out] profiling group

// Get metrics data
hsa_status_t rocprofiler_get_metrics(const rocprofiler_t* context);  // [in/out] profiling context

// Definition of output data iterator callback
typedef hsa_ven_amd_aqlprofile_data_callback_t rocprofiler_trace_data_callback_t;

// Method for iterating the events output data
hsa_status_t rocprofiler_iterate_trace_data(
    rocprofiler_t* context,                      // [in] profiling context
    rocprofiler_trace_data_callback_t callback,  // callback to iterate the output data
    void* data);                                 // [in/out] callback data

////////////////////////////////////////////////////////////////////////////////
// Profiling features and data
//
// Profiling features objects have profiling feature info, type, parameters and data
// Also profiling data samplaes can be iterated using a callback

// Profiling info kind
typedef enum {
  ROCPROFILER_INFO_KIND_METRIC = 0,                // metric info
  ROCPROFILER_INFO_KIND_METRIC_COUNT = 1,          // metric features count, int32
  ROCPROFILER_INFO_KIND_TRACE = 2,                 // trace info
  ROCPROFILER_INFO_KIND_TRACE_COUNT = 3,           // trace features count, int32
  ROCPROFILER_INFO_KIND_TRACE_PARAMETER = 4,       // trace parameter info
  ROCPROFILER_INFO_KIND_TRACE_PARAMETER_COUNT = 5  // trace parameter count, int32
} rocprofiler_info_kind_t;

// Profiling info query
typedef union {
  rocprofiler_info_kind_t info_kind;  // queried profiling info kind
  struct {
    const char* trace_name;  // queried info trace name
  } trace_parameter;
} rocprofiler_info_query_t;

// Profiling info data
typedef struct {
  uint32_t
      agent_index;  // GPU HSA agent index (GPU Driver Node ID as reported in the sysfs topology)
  rocprofiler_info_kind_t kind;  // info data kind
  union {
    struct {
      const char* name;         // metric name
      uint32_t instances;       // instances number
      const char* expr;         // metric expression, NULL for basic counters
      const char* description;  // metric description
      const char* block_name;   // block name
      uint32_t block_counters;  // number of block counters
    } metric;
    struct {
      const char* name;          // trace name
      const char* description;   // trace description
      uint32_t parameter_count;  // supported by the trace number parameters
    } trace;
    struct {
      uint32_t code;               // parameter code
      const char* trace_name;      // trace name
      const char* parameter_name;  // parameter name
      const char* description;     // trace parameter description
    } trace_parameter;
  };
} rocprofiler_info_data_t;

// Return the info for a given info kind
hsa_status_t rocprofiler_get_info(const hsa_agent_t* agent,      // [in] GFXIP handle
                                  rocprofiler_info_kind_t kind,  // kind of iterated info
                                  void* data);                   // [in/out] returned data

// Iterate over the info for a given info kind, and invoke an application-defined callback on every
// iteration
hsa_status_t rocprofiler_iterate_info(const hsa_agent_t* agent,      // [in] GFXIP handle
                                      rocprofiler_info_kind_t kind,  // kind of iterated info
                                      hsa_status_t (*callback)(const rocprofiler_info_data_t info,
                                                               void* data),  // callback
                                      void* data);  // [in/out] data passed to callback

// Iterate over the info for a given info query, and invoke an application-defined callback on every
// iteration
hsa_status_t rocprofiler_query_info(const hsa_agent_t* agent,        // [in] GFXIP handle
                                    rocprofiler_info_query_t query,  // iterated info query
                                    hsa_status_t (*callback)(const rocprofiler_info_data_t info,
                                                             void* data),  // callback
                                    void* data);  // [in/out] data passed to callback

// Create a profiled queue. All dispatches on this queue will be profiled
hsa_status_t rocprofiler_queue_create_profiled(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue);

////////////////////////////////////////////////////////////////////////////////
// Profiling pool
//
// Support for profiling contexts pool
// The API provide capability to create a contexts pool for a given agent and a set of features,
// to fetch/relase a context entry, to register a callback for the contexts completion.

// Profiling pool handle
typedef void rocprofiler_pool_t;

// Profiling pool entry
typedef struct {
  rocprofiler_t* context;  // context object
  void* payload;           // payload data object
} rocprofiler_pool_entry_t;

// Profiling handler, calling on profiling completion
typedef bool (*rocprofiler_pool_handler_t)(const rocprofiler_pool_entry_t* entry, void* arg);

// Profiling preperties
typedef struct {
  uint32_t num_entries;                // pool size entries
  uint32_t payload_bytes;              // payload size bytes
  rocprofiler_pool_handler_t handler;  // handler on context completion
  void* handler_arg;                   // the handler arg
} rocprofiler_pool_properties_t;

// Open profiling pool
hsa_status_t rocprofiler_pool_open(
    hsa_agent_t agent,                // GPU handle
    rocprofiler_feature_t* features,  // [in] profiling features array
    uint32_t feature_count,           // profiling info count
    rocprofiler_pool_t** pool,        // [out] context object
    uint32_t mode,                    // profiling mode mask
    rocprofiler_pool_properties_t*);  // pool properties

// Close profiling pool
hsa_status_t rocprofiler_pool_close(rocprofiler_pool_t* pool);  // profiling pool handle

// Fetch profiling pool entry
hsa_status_t rocprofiler_pool_fetch(
    rocprofiler_pool_t* pool,          // profiling pool handle
    rocprofiler_pool_entry_t* entry);  // [out] empty profiling pool entry

// Release profiling pool entry
hsa_status_t rocprofiler_pool_release(
    rocprofiler_pool_entry_t* entry);  // released profiling pool entry

// Iterate fetched profiling pool entries
hsa_status_t rocprofiler_pool_iterate(rocprofiler_pool_t* pool,  // profiling pool handle
                                      hsa_status_t (*callback)(rocprofiler_pool_entry_t* entry,
                                                               void* data),  // callback
                                      void* data);  // [in/out] data passed to callback

// Flush completed entries in profiling pool
hsa_status_t rocprofiler_pool_flush(rocprofiler_pool_t* pool);  // profiling pool handle

////////////////////////////////////////////////////////////////////////////////
// HSA intercepting API

// HSA callbacks ID enumeration
typedef enum {
  ROCPROFILER_HSA_CB_ID_ALLOCATE = 0,  // Memory allocate callback
  ROCPROFILER_HSA_CB_ID_DEVICE = 1,    // Device assign callback
  ROCPROFILER_HSA_CB_ID_MEMCOPY = 2,   // Memcopy callback
  ROCPROFILER_HSA_CB_ID_SUBMIT = 3,    // Packet submit callback
  ROCPROFILER_HSA_CB_ID_KSYMBOL = 4,   // Loading/unloading of kernel symbol
  ROCPROFILER_HSA_CB_ID_CODEOBJ = 5    // Loading/unloading of kernel symbol
} rocprofiler_hsa_cb_id_t;

// HSA callback data type
typedef struct {
  union {
    struct {
      const void* ptr;            // allocated area ptr
      size_t size;                // allocated area size, zero size means 'free' callback
      hsa_amd_segment_t segment;  // allocated area's memory segment type
      hsa_amd_memory_pool_global_flag_t global_flag;  // allocated area's memory global flag
      int is_code;                                    // equal to 1 if code is allocated
    } allocate;
    struct {
      hsa_device_type_t type;  // type of assigned device
      uint32_t id;             // id of assigned device
      hsa_agent_t agent;       // device HSA agent handle
      const void* ptr;         // ptr the device is assigned to
    } device;
    struct {
      const void* dst;  // memcopy dst ptr
      const void* src;  // memcopy src ptr
      size_t size;      // memcopy size bytes
    } memcopy;
    struct {
      const void* packet;       // submitted to GPU packet
      const char* kernel_name;  // kernel name, not NULL if dispatch
      hsa_queue_t* queue;       // HSA queue the kernel was submitted to
      uint32_t device_type;     // type of device the packed is submitted to
      uint32_t device_id;       // id of device the packed is submitted to
    } submit;
    struct {
      uint64_t object;       // kernel symbol object
      const char* name;      // kernel symbol name
      uint32_t name_length;  // kernel symbol name length
      int unload;            // symbol executable destroy
    } ksymbol;
    struct {
      uint32_t storage_type;  // code object storage type
      int storage_file;       // origin file descriptor
      uint64_t memory_base;   // origin memory base
      uint64_t memory_size;   // origin memory size
      uint64_t load_base;     // codeobj load base
      uint64_t load_size;     // codeobj load size
      uint64_t load_delta;    // codeobj load size
      uint32_t uri_length;    // URI string length
      char* uri;              // URI string
      int unload;             // unload flag
    } codeobj;
  };
} rocprofiler_hsa_callback_data_t;

// HSA callback function type
typedef hsa_status_t (*rocprofiler_hsa_callback_fun_t)(
    rocprofiler_hsa_cb_id_t id,                   // callback id
    const rocprofiler_hsa_callback_data_t* data,  // [in] callback data
    void* arg);                                   // [in/out] user passed data

// HSA callbacks structure
typedef struct {
  rocprofiler_hsa_callback_fun_t allocate;  // memory allocate callback
  rocprofiler_hsa_callback_fun_t device;    // agent assign callback
  rocprofiler_hsa_callback_fun_t memcopy;   // memory copy callback
  rocprofiler_hsa_callback_fun_t submit;    // packet submit callback
  rocprofiler_hsa_callback_fun_t ksymbol;   // kernel symbol callback
  rocprofiler_hsa_callback_fun_t codeobj;   // codeobject load/unload callback
} rocprofiler_hsa_callbacks_t;

// Set callbacks. If the callback is NULL then it is disabled.
// If callback returns a value that is not HSA_STATUS_SUCCESS the  callback
// will be unregistered.
hsa_status_t rocprofiler_set_hsa_callbacks(
    const rocprofiler_hsa_callbacks_t callbacks,  // HSA callback function
    void* arg);                                   // callback user data

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCPROFILER_H_
