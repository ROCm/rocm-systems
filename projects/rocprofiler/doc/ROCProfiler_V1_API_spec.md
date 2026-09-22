# ROC Profiler Library Specification
ROC Profiler API version 7

## 1. High level overview
```
The goal of the implementation is to provide a HW specific low-level performance analysis
interface for profiling of GPU compute applications. The profiling includes HW performance
counters with complex performance metrics and HW traces. The implementation distinguishes
two profiling features, metrics and traces. HW performance counters are treated as the basic
metrics and the formulas can be defined for derived complex metrics.
The library can be loaded by HSA runtime as a tool plugin and it can be loaded by higher
level HW independent performance analysis API like PAPI.
The library has C API and is based on AQLprofile AMD specific HSA extension.

  1. The library provides methods to query the list of supported HW features.
  2. The library provides profiling APIs to start, stop, read metrics results and tracing
  data.
  3. The library provides a intercepting API for collecting per-kernel profiling data for
  the kernels
  dispatched to HSA AQL queues.
  4. The library provides mechanism to load profiling tool library plugin by env variable
  ROCP_TOOL_LIB.
  5. The library is responsible for allocation of the buffers for profiling and notifying
  about output data buffer overflow for traces.
  6. The library is implemented based on AMD specific AQLprofile HSA extension.
  7. The library implementation is abstracted from the specific GFXIP.
  8. The library implementation is extensible:
    - Easy adding of counters and metrics
    - Counters enumeration 
    - Counters and metrics can be dynamically configured using XML configuration files with
    counters and metrics tables:
      o Counters table entry, basic metric: counter name, block name, event id
      o Complex metrics table entry: metric name, an expression for calculation the metric
      from the counters

Metrics XML file example:
<gfx8>
	<metric name=L1_CYCLES_COUNTER block=L1 event=0 >
	<metric name=L1_MISS_COUNTER block=L1 event=33 >
	. . .
</gfx8>

<gfx9>
	. . .
</gfx9>

<global>
	<metric name=L1_MISS_RATIO expr=L1_CYCLES_COUNT/ L1_MISS_COUNTER ></metric>
</global>
```
## 2. Environment
```
* HSA_TOOLS_LIB - required to be set to the name of rocprofiler library to be loaded by
HSA runtime
* ROCP_METRICS - path to the metrics XML file
* ROCP_TOOL_LIB - path to profiling tool library loaded by ROC Profiler
* ROCP_HSA_INTERCEPT - if set then HSA dispatches intercepting is enabled
```
## 3. General API
### 3.1. Description
```
The library supports method for getting the error number and error string of the last
failed library API call.
To check the conformance of used library APi header and the library binary the version
macros and API methods can be used.

Returning the error and error string methods:
- rocprofiler_error_string - method for returning the error string

Library version:
- ROCPROFILER_VERSION_MAJOR - API major version macro
- ROCPROFILER_VERSION_MINOR - API minor version macro
- rocprofiler_version_major - library major version
- rocprofiler_version_minor - library minor version
```
### 3.2. Returning the error and error string methods
```
hsa_status_t rocprofiler_error_string(const char** str);
```
### 3.3. Library version
```
The library provides back compatibility if the library major version is less or equal
then the API major version macro.

API version macros defined in the library API header 'rocprofiler.h':

ROCPROFILER_VERSION_MAJOR
ROCPROFILER_VERSION_MINOR

Methods to check library major and minor venison:

uint32_t rocprofiler_major_version();
uint32_t rocprofiler_minor_version();
```
## 4. Backend API
### 4.1. Description
```
The library provides the methods to open/close profiling context, to start, stop and read
HW performance counters and traces, to intercept kernel dispatches to collect per-kernel
profiling data. Also the library provides methods to calculate complex performance metrics
and to query the list of available metrics. The library distinguishes two profiling features,
metrics and traces, where HW performance counters are treated as the basic metrics. To check
if there was an error the library methods return HSA standard status code.
For a given context the profiling can be started/stopped and counters sampled in standalone
mode or profiling can be initiated by intercepting the kernel dispatches with registering
a dispatch callback.
For counters sampling, which is the usage model of higher level APIs like PAPI,
the start/stop/read APIs should be used.
For collecting per-kernel data for the submitted to HSA queues kernels the dispatch callback
API should be used.
The library provides back compatibility if the library major version is less or equal.

Returned API status:
- hsa_status_t - HSA status codes are used from hsa.h header

Loading and Configuring, loadable plugin on-load/unload methods:
- rocprofiler_settings_t – global properties
- OnLoadTool 
- OnLoadToolProp
- OnUnloadTool

Info API:
- rocprofiler_info_kind_t - profiling info kind
- rocprofiler_info_query_t - profiling info query
- rocprofiler_info_data_t - profiling info data
- rocprofiler_get_info - return the info for a given info kind
- rocprofiler_iterate_inf_ - iterate over the info for a given info kind 
- rocprofiler_query_info - iterate over the info for a given info query

Context API:
- rocprofiler_t - profiling context handle
- rocprofiler_feature_kind_t - profiling feature kind
- rocprofiler_feature_parameter_t - profiling feature parameter
- rocprofiler_data_kind_t - profiling data kind
- rocprofiler_data_t - profiling data
- rocprofiler_feature_t - profiling feature
- rocprofiler_mode_t - profiling modes
- rocprofiler_properties_t - profiler properties
- rocprofiler_open - open new profiling context
- rocprofiler_close - close profiling context and release all allocated resources
- rocprofiler_group_count - return profiling groups count
- rocprofiler_get_group - return profiling group for a given index
- rocprofiler_get_metrics - method for calculating the metrics data
- rocprofiler_iterate_trace_data - method for iterating output trace data instances
- rocprofiler_time_id_t - supported time value ID enumeration
- rocprofiler_get_time – return time for a given time ID and profiling timestamp value

Sampling API:
- rocprofiler_start - start profiling
- rocprofiler_stop - stop profiling
- rocprofiler_read - read profiling data to the profiling features objects
- rocprofiler_get_data - wait for profiling data
  Group versions of start/stop/read/get_data methods:
  o rocprofiler_group_start
  o rocprofiler_group_stop
  o rocprofiler_group_read
  o rocprofiler_group_get_data

Intercepting API:
- rocprofiler_callback_t - profiling callback type
- rocprofiler_callback_data_t - profiling callback data type
- rocprofiler_dispatch_record_t – dispatch record
- rocprofiler_queue_callbacks_t – queue callbacks, dispatch/destroy
- rocprofiler_set_queue_callbacks - set queue kernel dispatch and queue destroy callbacks
- rocprofiler_remove_queue_callbacks - remove queue callbacks

Context pool API:
- rocprofiler_pool_t – context pool handle
- rocprofiler_pool_entry_t – context pool entry
- rocprofiler_pool_properties_t – context pool properties
- rocprofiler_pool_handler_t – context pool completion handler
- rocprofiler_pool_open - context pool open
- rocprofiler_pool_close - context pool close
- rocprofiler_pool_fetch – fetch and empty context entry to pool
- rocprofiler_pool_release – release a context entry
- rocprofiler_pool_iterate – iterated fetched context entries
- rocprofiler_pool_flush – flush completed context entries
```
### 4.2. Loading and Configuring
```
Loading and Configuring
The profiling properties can be set by profiler plugin on loading by ROC runtime.
The profiler library plugin can be set by ROCP_TOOL_LIB env var.

Global properties:

typedef struct {
  	uint32_t intercept_mode;
  	uint64_t timeout;
  	uint32_t timestamp_on;
} rocprofiler_settings_t;

On load/unload methods defined in profiling tool library loaded by ROCP_TOOL_LIB env var:
extern "C" void OnLoadTool();
extern "C" void OnLoadToolProp(rocprofiler_settings_t* settings);
extern "C" void OnUnloadTool();

```
### 4.3. Info API
```
The profiling metrics are defined by name and the traces are defined by name and parameters.
All supported features can be iterated using 'iterate_info/query_info' methods. The counter
names are defined in counters table configuration file, each counter has a unique name and
defined by block name and event id. The traces and trace parameters names are same as in
the hardware documentation and the parameters codes are rocprofiler_feature_parameter_t values,
see below in the "Context API" section.

Profiling info kind:

typedef enum {
	ROCPROFILER_INFO_KIND_METRIC = 0,		// metric info
	ROCPROFILER_INFO_KIND_METRIC_COUNT = 1,		// metrics count
	ROCPROFILER_INFO_KIND_TRACE = 2,		// trace info
	ROCPROFILER_INFO_KIND_TRACE_COUNT = 3,		// traces count
} rocprofiler_info_kind_t;

Profiling info data:

typedef struct {
	rocprofiler_info_kind_t kind;			// info data kind
	union {
		struct {
			const char* name;		// metric name
			uint32_t instances;		// instances number
			const char* expr;		// metric expression, NULL for basic counters
			const char* description;	// metric description
			const char* block_name;		// block name
			uint32_t block_counters;	// number of block counters
		} metric;
		struct {
			const char* name;		// trace name
			const char* description;	// trace description
			uint32_t parameter_count;	// supported by the trace number
			                                // parameters
		} trace;
	};
} rocprofiler_info_data_t;

Return info for a given info kind:

has_status_t rocprofiler_get_info(
	const hsa_agent_t* agent,			// [in] GPU handle, NULL for all
	                                                // GPU agents
	rocprofiler info_kind_t kind,			// kind of iterated info
	void *data);					// data passed to callback

Iterate over the info for a given info kind, and invoke an application-defined callback on
every iteration:

has_status_t rocprofiler_iterate_info(
	const hsa_agent_t* agent,			// [in] GPU handle, NULL for all
	                                                // GPU agents
	rocprofiler info_kind_t kind,			// kind of iterated info
	hsa_status_t (*callback)(const rocprofiler_info_data_t info, void *data), // callback
	void *data);	

Iterate over the info for a given info query, and invoke an application-defined callback on
every iteration. The query
fields set to NULL define the query wildcard:

has_status_t rocprofiler_query_info(
	const hsa_agent_t* agent,			// [in] GPU handle, NULL for all
	                                                // GPU agents
	rocprofiler info_kind_t kind,			// kind of iterated info
	rocprofiler_info_data_t query,			// info query
	hsa_status_t (*callback)(const rocprofiler_info_data_t info, void *data), // callback
	void *data);					// data passed to callback
```
### 4.4. Context API
```
Profiling context is accumulating all profiling information including profiling features
which carry profiling data, required buffers for profiling command packets and output data.
The context can be created and deleted by the library open/close methods. By deleting
the context all accumulated by the library resources associated with this context will be
released. If it is required more than one run to collect all requested counters data then
data for all profiling groups should be collected and then the metrics can be calculated by
loading the saved groups' data to the profiling context. Saving and loading of the groups
data is responsibility of the tool. The groups are automatically identified on the profiling
context open and there is API to access them, see the "Profiling groups" section below.

Profiling context handle:

typename rocprofiler_t;

Profiling feature kind:

typedef enum {
	ROCPROFILER_FEATURE_KIND_METRIC = 0,	// metric
	ROCPROFILER_FEATURE_KIND_TRACE = 1	// trace
} rocprofiler_feature_kind_t;

Profiling feature parameter:

typedef hsa_ven_amd_aqlprofile_parameter_t rocprofiler_feature_parameter_t;

Profiling data kind:

typedef enum {
	ROCPROFILER_DATA_KIND_UNINIT = 0,	// data uninitialized
	ROCPROFILER_DATA_KIND_INT32 = 1,	// 32bit integer
	ROCPROFILER_DATA_KIND_INT64 = 2,	// 64bit integer
	ROCPROFILER_DATA_KIND_FLOAT = 3,	// float single-precision result
	ROCPROFILER_DATA_KIND_DOUBLE = 4,	// float double-precision result
	ROCPROFILER_DATA_KIND_BYTES = 5		// trace output as a bytes array
} rocprofiler_data_kind_t;


Profiling data:

typedef struct {
	rocprofiler_data_kind_t kind;		// result kind
	union {
		uint32_t result_int32;		// 32bit integer result
		uint64_t result_int64;		// 64bit integer result
		float result_float;		// float single-precision result
		double result_double;		// float double-precision result
		typedef struct {
 			void* ptr;		// pointer
			uint32_t size;		// byte size
			uint32_t instances;	// number of trace instances
		} result_bytes;			// data by ptr and byte size
	};
} rocprofiler_data_t;

Profiling feature:

typedef struct {
	rocprofiler_feature_kind_t type;			// feature type
	const char* name;					// feature name
	const rocprofiler_feature_parameter_t* parameters;	// feature parameters
	uint32_t parameter_count;				// feature parameter count
	rocprofiler_data_t* data;				// profiling data
} rocprofiler_feature_t;

Profiling mode masks:
There are several modes which can be specified for the profiling context.
STANDALONE mode can be used for the counters sampling in another then application context
to support statistical system wide profiling. In this mode the profiling context supports
its own queue which can be created on the context open if the CREATEQUEUE mode also specified.
See also "Profiler properties" section below for the standalone mode queue properties.
The profiler supports several profiling groups for collecting profiling data in several
runs and 'SINGLEGROUP' mode allows only one group and the context open will fail if more
groups are needed.

typedef enum {
	ROCPROFILER_MODE_STANDALONE = 1,	// standalone mode when ROC profiler
	                                        // supports own AQL queue
	ROCPROFILER_MODE_CREATEQUEUE = 2,	// profiler creates queue in STANDALONE mode
	ROCPROFILER_MODE_SINGLEGROUP = 4	// profiler allows one group only and fails
	                                        // if more groups are needed
} rocprofiler_mode_t;

Context data readiness callback:

typedef void (*rocprofiler_context_callback_t)(
	rocprofiler_group_t* group,		// profiling group
	void* arg);				// callback arg

Profiler properties:
There are several properties which can be specified for the context. A callback can be
registered which will be called when the context data is ready. In standalone profiling mode
'ROCPROFILER_MODE_STANDALONE' the context supports its own queue and the queue can be set by
the property 'queue' or a queue will be created with the specified depth 'queue_depth' if mode
'ROCPROFILER_MODE_CREATEQUEUE' also specified.

typedef struct {
	rocprofiler_context_callback_t callback; // callback on the context data readiness
	void* callback_arg;			 // callback arg
	has_queue_t* queue;			 // HSA queue for standalone mode
	uint32_t queue_depth;			 // created queue depth,for create-queue mode
} rocprofiler_properties_t;

Open/close profiling context:

hsa_status_t rocprofiler_open(
	hsa_agent_t agent,			// GPU handle
	rocprofiler_feature_t* features,	// [in/out] profiling feature array
	uint32_t feature_count,			// profiling feature count
	rocprofiler_t** context,		// [out] profiling context handle
	uint32_t mode,				// profiling mode mask
	rocprofiler_properties_t* properties);	// profiler properties

hsa_status_t rocprofiler_close(
	rocprofiler_t* context);		// [in] profiling context

Profiling groups:
The profiler on the context open automatically identifies a required number of the application
runs to collect all data needed for all specified metrics and creates a metric group per each
run. Data for all profiling groups should be collected and then the metrics can be calculated
by loading the saved groups' data to the profiling context. Saving and loading of he groups
data is responsibility of the tool.

typedef struct {
	uint32_t index;				// profiling group index
	rocprofiler_feature_t** features;	// profiling features array
	uint32_t feature_count;			// profiling feature count
	rocprofiler_t* context;			// profiling context handle
} rocprofiler_group_t;

Return profiling groups count:

hsa_status_t rocprofiler_group_count(
	rocprofiler_t* context);		// [in/out] profiling context
	uint32* count);				// [out] profiling groups count

Return the profiling group for a given index:

hsa_status_t rocprofiler_get_group(
	rocprofiler_t* context,			// [in/out] profiling context,
	                                        // will be returned as
						// a part of the group structure
	uint32_t index,				// [in] group index
	rocprofiler_group_t* group);		// [out] profiling group

Calculate metrics data. The data will be stored to the registered profiling features data fields:
After all profiling context data is ready the registered metrics can be calculated. The context
data readiness can be checked by 'get_data' API or using the context callback.

hsa_status_t rocprofiler_get_metrics(
	rocprofiler_t* context);		// [in/out] profiling context

Method for iterating trace data instances:
Trace data can have several instance, for example, one instance per Shader Engine.

hsa_status_t rocprofiler_iterate_trace_data(
	const rocprofiler_t* contex,			// [in] context object
	hsa_ven_amd_aqlprofile_data_callback_t callback, // [in] callback to iterate
	                                                // the output data
	void* callback_data);				// [in/out] passed to callback data

Converting of profiling timestamp to time value for suported time ID.
Supported time value ID enumeration:
typedef enum {
  ROCPROFILER_TIME_ID_CLOCK_REALTIME = 0,   // Linux realtime clock time
  ROCPROFILER_TIME_ID_CLOCK_MONOTONIC = 1,  // Linux monotonic clock time
} rocprofiler_time_id_t;

Method for converting of profiling timestamp to time value for a given time ID:
hsa_status_t rocprofiler_get_time(
  rocprofiler_time_id_t time_id,            // identifier of the particular
                                            // time to convert the timestamp
  uint64_t timestamp,                       // profiling timestamp
  uint64_t* value_ns);                      // [out] returned time ‘ns’ value
```
### 4.5. Sampling API
```
The API supports the counters sampling usage model with start/read/stop methods and also lets
to wait for the profiling data in the intercepting usage model with get_data method.

Start/stop/read methods:

hsa_status_t rocprofiler_start(
	rocprofiler_t* context,			// [in/out] profiling context
	uint32_t group_index = 0);		// group index

hsa_status_t rocprofiler_stop(
	rocprofiler_t* context,			// [in/out] profiling context
	uint32_t group_index = 0);		// group index
	
hsa_status_t rocprofiler_read(
	rocprofiler_t* context,			// [in/out] profiling context
	uint32_t group_index = 0);		// group index

Wait for profiling data:

hsa_status_t rocprofiler_get_data(
	rocprofiler_t* context,			// [in/out] profiling context
	uint32_t group_index = 0);		// group index

Group versions of the above start/stop/read/get_data methods:

hsa_status_t rocprofiler_group_start(
	rocprofiler_group_t* group);		// [in/out] profiling group

hsa_status_t rocprofiler_group_stop(
	rocprofiler_group_t* group);		// [in/out] profiling group

	
hsa_status_t rocprofiler_group_read(
	rocprofiler_group_t* group);		// [in/out] profiling group


hsa_status_t rocprofiler_group_get_data(
	rocprofiler_group_t* group);		// [in/out] profiling group
```
### 4.6.  Intercepting API 
```
The library provides a callback API for enabling profiling for the kernels dispatched to
HSA AQL queues. The API enables per-kernel profiling data collection.
Currently implemented the option with serializing the kernels execution.

ROC profiler callback type:

hsa_status_t (*rocprofiler_callback_t)(
	const rocprofiler_callback_data_t* callback_data, // callback data passed by HSA runtime
	void* user_data,				  // [in/out] user data passed
	                                                  // to the callback
	rocprofiler_group** group);			  // [out] returned profiling group

Profiling callback data:

typedef struct {
	uint64_t dispatch;                                   // dispatch timestamp
	uint64_t begin;                                      // begin timestamp
	uint64_t end;                                        // end timestamp
	uint64_t complete;                                   // completion signal timestamp
} rocprofiler_dispatch_record_t;

typedef struct {
	hsa_agent_t agent;                                   // GPU agent handle
	uint32_t agent_index;                                // GPU index
	const hsa_queue_t* queue;                            // HSA queue
	uint64_t queue_index;                                // Index in the queue
	const hsa_kernel_dispatch_packet_t* packet;          // HSA dispatch packet
	const char* kernel_name;                             // Kernel name
	const rocprofiler_dispatch_record_t* record;         // Dispatch record
} rocprofiler_callback_data_t;

Queue callbacks:

typedef struct {
    rocprofiler_callback_t dispatch;                             // kernel dispatch callback
    hsa_status_t (*destroy)(hsa_queue_t* queue, void* data);     // queue destroy callback
} rocprofiler_queue_callbacks_t;

Adding/removing kernel dispatch and queue destroy callbacks

hsa_status_t rocprofiler_set_intercepting(
    rocprofiler_intercepting_t callbacks,                        // intercepting callbacks
    void* data);                                                 // [in/out] passed callbacks data

hsa_status_t rocprofiler_remove_intercepting();
```
### 4.7.  Profiling Context Pools
```
The API provide capability to create a context pool for a given agent and a set of features, to fetch/release a context entry, to register a callback for  pool’s contexts completion.
Profiling pool handle:
typename rocprofiler_pool_t;
Profiling pool entry:
typedef struct {
  	rocprofiler_t* context;           // context object
  	void* payload;                    // payload data object
} rocprofiler_pool_entry_t;

Profiling handler, calling on profiling completion:
typedef bool (*rocprofiler_pool_handler_t)(const rocprofiler_pool_entry_t* entry, void* arg);

Profiling properties:
typedef struct {
   uint32_t num_entries;                    // pool size entries
   uint32_t payload_bytes;                  // payload size bytes
   rocprofiler_pool_handler_t handler;      // handler on context completion
   void* handler_arg;                       // the handler arg
} rocprofiler_pool_properties_t;

Open profiling pool:
hsa_status_t rocprofiler_pool_open(
   hsa_agent_t agent,                       // GPU handle
   rocprofiler_feature_t* features,         // [in] profiling features array
   uint32_t feature_count,                  // profiling info count
   rocprofiler_pool_t** pool,               // [out] context object
   uint32_t mode,                           // profiling mode mask
   rocprofiler_pool_properties_t*);         // pool properties

Close profiling pool:
hsa_status_t rocprofiler_pool_close(
   rocprofiler_pool_t* pool);               // profiling pool handle

Fetch profiling pool entry:
hsa_status_t rocprofiler_pool_fetch(
   rocprofiler_pool_t* pool,          // profiling pool handle
   rocprofiler_pool_entry_t* entry);  // [out] empty profiling pool entry

Release profiling pool entry:
hsa_status_t rocprofiler_pool_release(
   rocprofiler_pool_entry_t* entry);  // released profiling pool entry

Iterate fetched profiling pool entries:
hsa_status_t rocprofiler_pool_iterate(
   rocprofiler_pool_t* pool,           // profiling pool handle
   hsa_status_t (*callback)(rocprofiler_pool_entry_t* entry, void* data), 	            
                                       // callback
   void *data); 		               // [in/out] data passed to callback

Flush completed entries in profiling pool:
hsa_status_t rocprofiler_pool_flush(
  	rocprofiler_pool_t* pool);       // profiling pool handle
```
## 5. Application code examples
### 5.1. Querying available metrics
```
Info data callback:

    hsa_status_t info_data_callback(const rocprofiler_info_data_t info, void *data) {
        switch (info.kind) {
            case ROCPROFILER_INFO_KIND_METRIC: {
                if (info.metric.expr != NULL) {
                    fprintf(stdout, "Derived counter:  gpu-agent%d : %s : %s\n",
                        info.agent_index, info.metric.name, info.metric.description);
                    fprintf(stdout, "      %s = %s\n", info.metric.name, info.metric.expr);
                } else {
                    fprintf(stdout, "Basic counter:  gpu-agent%d : %s",
                        info.agent_index, info.metric.name);
                    if (info.metric.instances > 1) {
                        fprintf(stdout, "[0-%u]", info.metric.instances - 1);
                    }
                    fprintf(stdout, " : %s\n", info.metric.description);
                    fprintf(stdout, "      block %s has %u counters\n",
                        info.metric.block_name, info.metric.block_counters);
                }
                fflush(stdout);
                break;
            }
            default:
                printf("wrong info kind %u\n", kind);
                return HSA_STATUS_ERROR;
        }
        return HSA_STATUS_SUCCESS;
    }

Printing all available metrics:

    hsa_status_t status = rocprofiler_iterate_info(
        agent,
        ROCPROFILER_INFO_KIND_METRIC,
        info_data_callback,
        NULL);
    <check status>
```
### 5.2. Profiling code example
```
Profiling of L1 miss ratio, average memory bandwidth.
In the example below rocprofiler_group_get_data group APIs are used for the purpose of a usage
example but in SINGLEGROUP mode when only one group is allowed the context handle itself can be
saved and then direct context method rocprofiler_get_data with default group index equal to 0
can be used.

hsa_status_t dispatch_callback(
    const rocprofiler_callback_data_t* callback_data,
    void* user_data,
    rocprofiler_group_t* group)
{
    hsa_status_t status = HSA_STATUS_SUCCESS;
    // Profiling context
    rocprofiler_t* context;
    // Profiling info objects
    rocprofiler_feature_t features* = new rocprofiler_feature_t[2];
    // Tracing parameters
    rocprofiler_feature_parameter_t* parameters = new rocprofiler_feature_parameter_t[2];

    // Setting profiling features
    features[0].type = ROCPROFILER_METRIC;
    features[0].name = "L1_MISS_RATIO";
    features[1].type = ROCPROFILER_METRIC;
    features[1].name = "DRAM_BANDWIDTH";

    // Creating profiling context
    status = rocprofiler_open(callback_data->dispatch.agent, features, 2, &context,
        ROCPROFILER_MODE_SINGLEGROUP, NULL);
    <check status>
    
    // Get the profiling group
    // For general case with many groups there is rocprofiler_group_count() API
    const uint32_t group_index = 0
    status = rocprofiler_get_group(context, group_index, group);
    <check_status>

    // In SINGLEGROUP mode the context handle itself can be saved, because there is just one group
    <saving the callback data/profiling group/profiling features>

    return status;
}

Profiling tool constructor is adding the dispatch callback:

void profiling_libary_constructor() {
    // Defining callback data, no data in this simple example
    void* callback_data = NULL;

    // Adding observers
    hsa_sttaus_t status = rocprofiler_add_dispatch_callback(dispatch_callback, callback_data);
    <check status>

    // Dispatching profiled kernel
    <dispatching profiled kernels>
}

void profiling_libary_destructor() {
    <for entry : <saved callbacks data>> {
        // In SINGLEGROUP mode the rocprofiler_get_group() method with default zero group
        // index can be used, if context handle would be saved
        status = rocprofiler_group_get_data(entry->group);
        <check status>
        status = rocprofiler_get_metrics(entry->group->context);
        <check status>
        status = rocprofiler_close(entry->group->context);
        <check status>

        <tool_dump_data_method(entry->dispatch_data, entry->features, entry->features_count)>;
    }
}
```
### 5.3. Option to use completion callback
```
Creating profiling context with completion callback:
    . . .
    rocprofiler_properties_t properties = {};
    properties.callback = completion_callback;
    properties.callback_arg = NULL;     // no args defined
    status = rocprofiler_open(agent, features, 3, &context,
        ROCPROFILER_MODE_SINGLEGROUP, properties);
    <check status>
    . . .

Definition of completion callback:

void completion_callback(profiler_group_t group, void* arg) {
    <tool_dump_data_method(group)>
    hsa_status_t status = rocprofiler_close(group.context);
    <check status>
}
```
### 5.4. Option to Use Context Pool
```
Code example of context pool usage.
Creating profiling contexts pool:
   . . .
   rocprofiler_pool_properties_t properties{};
   properties.num_entries = 100;
   properties.payload_bytes = sizeof(context_entry_t);
   properties.handler = context_handler; 
   properties.handler_arg = handler_arg;
   status = rocprofiler_pool_open(agent, features, 3, &context,
		ROCPROFILER_MODE_SINGLEGROUP, properties);
   <check status>
    . . .

Fetching a context entry:
   rocprofiler_pool_entry_t pool_entry{};
   status = rocprofiler_pool_fetch(pool, &pool_entry);
   <check status>
   // Profiling context entry
   rocprofiler_t* context = pool_entry.context;
   context_entry_t* entry = reinterpret_cast <context_entry_t*>               
                            (pool_entry.payload);
```
### 5.5. Standalone Sampling Usage Code Example
```
The profiling metrics are being read from separate standalone queue other than the application kernels are submitted to.
To enable the sampling mode, the profiling mode in all user queues should be enabled.  It can be done by loading ROC-profiler
library to HSA runtime using the environment variable HSA_TOOLS_LIB for all shell sessions.
   // Sampling rate
   uint32_t sampling_rate = <some rate>;
   // Sampling count
   uint32_t sampling_count = <some count>;
   // HSA status
   hsa_status_t status = HSA_STATUS_ERROR;
   // HSA agent
   hsa_agent_t agent;
   // Profiling context
   rocprofiler_t* context = NULL;
   // Profiling properties
   rocprofiler_properties_t properties;

   // Getting HSA agent
   <query for HSA agent by ‘hsa_iterate_agents()’>
 
   // Profiling feature objects
   const unsigned feature_count = 2;
   rocprofiler_feature_t feature[feature_count];
  	
   // Counters and metrics
   feature[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
   feature[0].name = "GPUBusy";
   feature[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
   feature[1].name = "SQ_WAVES";

   // Creating profiling context with standalone queue
   properties = {};
   properties.queue_depth = 128;
   status = rocprofiler_open(agent, feature, feature_count, &context,
            ROCPROFILER_MODE_STANDALONE| ROCPROFILER_MODE_CREATEQUEUE|
            ROCPROFILER_MODE_SINGLEGROUP, &properties);
   <check status>

   // Start counters and sample them in the loop with the sampling rate
   status = rocprofiler_start(context, 0);
   <check status>

   for (unsigned ind = 0; ind < sampling_count; ++ind) {
      sleep(sampling_rate);
      status = rocprofiler_read(context, 0);
      <check status>
      status = rocprofiler_get_data(context, 0);
      <check status>
      status = rocprofiler_get_metrics(context);
      <check status>
      print_results(feature, feature_count);
   }

   // Stop counters
   status = rocprofiler_stop(context, group_n);
   <check status>

   // Finishing cleanup
   // Deleting profiling context will delete all allocated resources
   status = rocprofiler_close(context); 
   <check status>
```
### 5.6. Printing Out Profiling Results
```
Below  is a code example for printing out the profiling results from profiling features array:
void print_results(rocprofiler_feature_t* feature, uint32_t feature_count) {
   for (rocprofiler_feature_t* p = feature; p < feature + feature_count; ++p) 
   {
      std::cout << (p - feature) << ": " << p->name;
      switch (p->data.kind) {
         case ROCPROFILER_DATA_KIND_INT64:
            std::cout << " result_int64 (" << p->data.result_int64 << ")" 
                      << std::endl;
            break;

         case ROCPROFILER_DATA_KIND_BYTES: {
            std::cout << " result_bytes ptr(" << p->data.result_bytes.ptr << 
                 ") " << " size(" << p->data.result_bytes.size << ")"
                 << " instance_count(" << p->data.result_bytes.instance_count 
                 << ")";
            break;
         }
         default:
            std::cout << "bad result kind (" << p->data.kind << ")" 
                      << std::endl;
            <abort>
      }
   }
}
```
