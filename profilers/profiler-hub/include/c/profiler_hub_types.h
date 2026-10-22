/**
 * @file profiler_hub_types.h
 * @brief Types shared by the profiler-hub C ABI (see profiler_hub.h).
 */

#ifndef PROFILER_HUB_TYPES_INC
#define PROFILER_HUB_TYPES_INC

// NOLINTBEGIN
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Result code returned by every profiler-hub C API function. */
    typedef enum
    {
        PH_RESULT_SUCCESS,                   /**< Call completed successfully. */
        PH_RESULT_CONTEXT_ALLOCATION_FAILED, /**< Context could not be created (e.g. trace
                                                 could not be opened/parsed). */
        PH_RESULT_INVALID_CONTEXT,           /**< Context argument was null or invalid. */
        PH_RESULT_INVALID_ARGUMENT,          /**< A non-context argument was null or
                                                 invalid. */
        PH_RESULT_INTERNAL_ERROR,            /**< An internal error occurred while
                                                 servicing the call. */
        PH_RESULT_FUTURE_ALLOCATION_FAILED,  /**< An async task could not be submitted
                                                 (out of memory). */
        PH_RESULT_CANCELLED,                 /**< The operation was cancelled through its
                                                 future before it finished. */
        PH_RESULT_UNSUPPORTED_FILE,          /**< The file is not a format profiler-hub
                                                     can read. */
        PH_RESULT_INVALID_FILE,              /**< The file is missing, unreadable, or not
                                                     a usable trace. */
        PH_RESULT_UNSUPPORTED_SCHEMA_VERSION, /**< The trace schema version is recognized
                                                  but not supported. */
    } ph_result_t;

    struct ph_ctx;
    struct ph_future;
    struct ph_event_query;

    /** @brief Opaque handle to a trace context. Created by ph_ctx_create(),
     *         released by ph_ctx_free(). */
    typedef struct ph_ctx*         ph_ctx_t;
    typedef struct ph_trace*       ph_trace_t;
    typedef struct ph_event_query* ph_event_query_t;
    /** @brief Opaque handle to an asynchronous operation. Created by
     *         ph_future_create(), released by ph_future_free(). */
    typedef struct ph_future* ph_future_t;

    /** @brief Semantic version triple (major.minor.patch). */
    typedef struct
    {
        uint32_t major;
        uint32_t minor;
        uint32_t patch;
    } ph_version_t;

    /** @brief Trace schema family found in a recognized file. */
    typedef enum
    {
        PH_TRACE_SCHEMA_UNKNOWN,
        PH_TRACE_SCHEMA_ROCPD,
    } ph_trace_schema_t;

    /**
     * @brief Trace schema and schema version returned by ph_probe_file().
     *        The probe outcome is the function's ph_result_t. Fields stay zero
     *        when probing did not recognize them.
     */
    typedef struct
    {
        ph_trace_schema_t trace_schema;
        ph_version_t      schema_version;
    } ph_file_probe_t;

    /** @brief Distinguishes the identity columns a track was derived from. */
    typedef enum
    {
        PH_TRACK_CATEGORY_THREAD,        /**< (nid,pid,tid)-based, untagged events. */
        PH_TRACK_CATEGORY_THREAD_SAMPLE, /**< Same (nid,pid,tid) family, events
                                               tagged with a named track. */
        PH_TRACK_CATEGORY_PMC_AGENT,     /**< PMC/counter samples split by agent_id. */
        PH_TRACK_CATEGORY_KERNEL_DISPATCH_AGENT_QUEUE, /**< By (nid,agent_id,queue_id). */
        PH_TRACK_CATEGORY_MEMORY_ALLOCATE_AGENT_QUEUE, /**< By (nid,agent_id,queue_id). */
        PH_TRACK_CATEGORY_MEMORY_COPY_AGENT_QUEUE,     /**< By (nid,agent_id,queue_id). */
        PH_TRACK_CATEGORY_STREAM, /**< Merged kernel-dispatch/memory-allocate/
                                       memory-copy, by (nid,pid,stream_id). */
    } ph_track_category_t;

    /** @brief A min/max pair that may be absent. */
    typedef struct
    {
        double   min; /**< Smallest value. Only meaningful if is_valid is non-zero. */
        double   max; /**< Largest value. Only meaningful if is_valid is non-zero. */
        uint32_t is_valid; /**< Non-zero if min and max hold real values. */
    } ph_value_range_t;

    /** @brief An unsigned value that may be absent.
     *         @p value is 0 and only meaningful when @p is_valid is non-zero. */
    typedef struct
    {
        uint64_t value;
        uint32_t is_valid;
    } ph_optional_uint64_t;

    /** @brief A start/end window that may be absent.
     *         @p start and @p end are only meaningful when @p is_valid is
     *         non-zero. */
    typedef struct
    {
        uint64_t start;
        uint64_t end;
        uint32_t is_valid;
    } ph_time_filter_t;

    /**
     * @brief A single track in a trace.
     * @note track_name points into memory owned by the ph_ctx_t that
     *       produced it; do not free it and do not use it after the context
     *       is freed.
     */
    typedef struct
    {
        uint32_t    id;
        const char* track_name;
        uint32_t    nid;         /**< Node id this track belongs to. */
        uint32_t    pid;         /**< OS process id. Not rocpd_info_process.id. */
        uint32_t    tid;         /**< OS thread id, or 0 if this track has no thread.
                                       Not rocpd_info_thread.id. */
        uint32_t    event_count; /**< Total events recorded on this track. */
        uint32_t    agent_id;    /**< Owning device (agent) id, or 0 if not
                                       device-specific. */
        ph_track_category_t category; /**< What identity columns this track was
                                            derived from. */
        uint32_t queue_id;            /**< Owning queue id. Only meaningful for
                                       *_AGENT_QUEUE categories, else 0. */
        uint32_t stream_id;           /**< Owning stream id. Only meaningful for
                                            PH_TRACK_CATEGORY_STREAM, else 0. */
        uint64_t start_ts;            /**< Nanosecond timestamp of the track's
                                            earliest event. */
        uint64_t end_ts;              /**< Nanosecond timestamp of the track's
                                            latest event. */
        ph_value_range_t value_range; /**< Sample value range. Only valid for
                                            PH_TRACK_CATEGORY_PMC_AGENT tracks with
                                            at least one sample value. */
        uint32_t nesting_depth; /**< Largest ph_event_t::nesting_depth of the track's
                                      events; 0 for PMC tracks. */
        uint64_t process_row_id; /**< rocpd_info_process.id. Not pid. 0 if this
                                       track has no process. */
        uint64_t thread_row_id;  /**< rocpd_info_thread.id. Not tid. 0 if this
                                       track has no thread. */
    } ph_track_t;

    /**
     * @brief A list of tracks.
     * @note tracks points into memory owned by the ph_ctx_t that produced
     *       this list; valid only until that context is freed or the list is
     *       re-queried.
     */
    typedef struct
    {
        uint32_t    list_size;
        ph_track_t* tracks;
    } ph_track_list_t;

    /**
     * @brief Node identity information.
     * @note name points into memory owned by the producing ph_ctx_t.
     */
    typedef struct
    {
        uint32_t    id;
        const char* machine_id;
        const char* system_name;
        const char* hostname;
        const char* release;
        const char* version;
        const char* hardware_name;
        const char* domain_name;
    } ph_node_info_t;

    /**
     * @brief A compute agent (e.g. CPU/GPU) belonging to a node.
     * @note All const char* fields point into memory owned by the producing
     *       ph_ctx_t.
     */
    typedef struct
    {
        uint32_t             id; /**< type_index. Not rocpd_info_agent.id. */
        const char*          agent_type;
        const char*          name;
        const char*          model_name;
        const char*          vendor_name;
        const char*          product_name;
        const char*          user_name;
        const char*          extdata;
        ph_optional_uint64_t absolute_index;
        ph_optional_uint64_t logical_index;
        ph_optional_uint64_t uuid;
        uint64_t             row_id; /**< rocpd_info_agent.id. */
    } ph_agent_t;

    /** @brief A list of agents; same lifetime rule as ph_track_list_t. */
    typedef struct
    {
        uint32_t    list_size;
        ph_agent_t* agents;
    } ph_agent_list_t;

    /**
     * @brief A process belonging to a node.
     * @note command, environment, and extdata point into memory owned by the
     *       producing ph_ctx_t. An unrecorded string is empty.
     */
    typedef struct
    {
        uint32_t             id; /**< OS process id. Not rocpd_info_process.id. */
        const char*          command;
        const char*          environment;
        const char*          extdata;
        ph_optional_uint64_t ppid;
        ph_optional_uint64_t init;
        ph_optional_uint64_t fini;
        ph_optional_uint64_t start;
        ph_optional_uint64_t end;
        uint64_t             row_id; /**< rocpd_info_process.id. */
    } ph_process_t;

    /** @brief A list of processes; same lifetime rule as ph_track_list_t. */
    typedef struct
    {
        uint32_t      list_size;
        ph_process_t* processes;
    } ph_process_list_t;

    /**
     * @brief A thread belonging to a process.
     * @note name and extdata point into memory owned by the producing ph_ctx_t.
     */
    typedef struct
    {
        uint64_t             thread_id; /**< OS thread id. Not rocpd_info_thread.id. */
        const char*          name;
        const char*          extdata;
        ph_optional_uint64_t parent_process_id;
        ph_optional_uint64_t start;
        ph_optional_uint64_t end;
        uint64_t             row_id; /**< rocpd_info_thread.id. */
    } ph_thread_t;

    /** @brief A list of threads; same lifetime rule as ph_track_list_t. */
    typedef struct
    {
        uint32_t     list_size;
        ph_thread_t* threads;
    } ph_thread_list_t;

    /**
     * @brief A stream belonging to a process.
     * @note name and extdata point into memory owned by the producing ph_ctx_t.
     */
    typedef struct
    {
        uint64_t    stream_id;
        const char* name;
        const char* extdata;
        uint32_t    nid; /**< rocpd_info_node.id. */
        uint32_t    pid; /**< rocpd_info_process.id. Not the OS pid. */
    } ph_stream_t;

    /** @brief A list of streams; same lifetime rule as ph_track_list_t. */
    typedef struct
    {
        uint32_t     list_size;
        ph_stream_t* streams;
    } ph_stream_list_t;

    /**
     * @brief A queue belonging to a process.
     * @note name and extdata point into memory owned by the producing ph_ctx_t.
     */
    typedef struct
    {
        uint64_t    queue_id;
        const char* name;
        const char* extdata;
        uint32_t    nid; /**< rocpd_info_node.id. */
        uint32_t    pid; /**< rocpd_info_process.id. Not the OS pid. */
    } ph_queue_t;

    /** @brief A list of queues; same lifetime rule as ph_track_list_t. */
    typedef struct
    {
        uint32_t    list_size;
        ph_queue_t* queues;
    } ph_queue_list_t;

    /**
     * @brief A PMC counter description.
     * @note All const char* fields point into memory owned by the producing
     *       ph_ctx_t.
     */
    typedef struct
    {
        const char*          name;
        const char*          target_arch;
        const char*          symbol;
        const char*          description;
        const char*          long_description;
        const char*          component;
        const char*          units;
        const char*          value_type;
        const char*          block;
        const char*          expression;
        const char*          extdata;
        ph_optional_uint64_t event_code;
        ph_optional_uint64_t instance_id;
        ph_optional_uint64_t is_constant;
        ph_optional_uint64_t is_derived;
        uint64_t             id; /**< rocpd_info_pmc.id. Not a list index. */
    } ph_pmc_info_t;

    /** @brief A list of PMC descriptions; same lifetime rule as ph_track_list_t. */
    typedef struct
    {
        uint32_t       list_size;
        ph_pmc_info_t* pmc_infos;
    } ph_pmc_info_list_t;

    /**
     * @brief One distinct relationship from a stream to the hardware queue that
     *        ran its kernel dispatches, memory allocations, or memory copies.
     * @note stream_track_id and hardware_track_id refer to ph_track_t::id values
     *       in this node's track list. is_valid is 0 when that side was not
     *       recorded. For a memory copy, agent_id is the destination agent.
     *       pid is the process-table id stored on the event row. It matches
     *       the id used to find the stream track, and it is not ph_track_t::pid
     *       or ph_process_t::id, which are the OS process id.
     */
    typedef struct
    {
        ph_track_category_t  category;
        uint32_t             nid;
        uint32_t             pid;
        uint64_t             stream_id;
        ph_optional_uint64_t agent_id;
        ph_optional_uint64_t queue_id;
        ph_optional_uint64_t stream_track_id;
        ph_optional_uint64_t hardware_track_id;
    } ph_stream_hardware_link_t;

    /** @brief A list of stream-to-hardware links; same lifetime rule as
     *         ph_track_list_t. */
    typedef struct
    {
        uint32_t                   list_size;
        ph_stream_hardware_link_t* links;
    } ph_stream_hardware_link_list_t;

    /** @brief Node info bundled with its agents, processes, threads, streams,
     *         queues, PMC descriptions, tracks, and stream-to-hardware links. */
    typedef struct
    {
        ph_node_info_t                 info;
        ph_agent_list_t                agents;
        ph_process_list_t              process_list;
        ph_track_list_t                track_list;
        ph_thread_list_t               thread_list;
        ph_stream_list_t               stream_list;
        ph_queue_list_t                queue_list;
        ph_pmc_info_list_t             pmc_info_list;
        ph_stream_hardware_link_list_t stream_hardware_links;
    } ph_node_t;

    /** @brief Kind of a duration event. */
    typedef enum
    {
        PH_EVENT_TYPE_REGION,          /**< Host code region. */
        PH_EVENT_TYPE_KERNEL_DISPATCH, /**< Kernel dispatch on a device queue. */
        PH_EVENT_TYPE_MEMORY_COPY,     /**< Memory copy. */
        PH_EVENT_TYPE_MEMORY_ALLOCATE, /**< Memory allocation. */
    } ph_event_type_t;

    /** @brief A single duration event (region/kernel dispatch/memory
     *         copy/memory allocate).
     * @note name points into memory owned by the producing ph_ctx_t.
     * @note id is only unique together with type: every event type has its
     *       own id space.
     * @note nesting_depth is the lowest level of this track that is free when
     *       the event starts (1 = nothing on the track overlaps it). A later
     *       event reuses a lower level once that level no longer overlaps it.
     *       The value is relative to the whole track, also for a time-windowed
     *       request.
     * @note track_id is ph_track_t::id of the event's track. Track ids start
     *       at 0.
     */
    typedef struct
    {
        uint64_t        id;
        uint64_t        start;
        uint64_t        end;
        const char*     name;
        ph_event_type_t type;
        uint32_t        nesting_depth;
        uint32_t        track_id;
    } ph_event_t;

    /** @brief A list of duration events; same lifetime rule as
     *         ph_track_list_t. */
    typedef struct
    {
        uint32_t    list_size;
        ph_event_t* events;
    } ph_event_list_t;

    /**
     * @brief Text match against the trace string table and kernel symbol names.
     *        Profiler-hub resolves @p search_strings itself. The caller does
     *        not pass string ids.
     * @note Matching is case-insensitive. include_substring non-zero selects a
     *       substring match; zero selects the whole string. partial_matching
     *       non-zero keeps a string when any term matches; zero keeps it only
     *       when every term matches. include_category non-zero applies the
     *       same matches to category columns as well as names.
     * @note search_strings is read only during ph_event_query_create().
     */
    typedef struct
    {
        const char* const* search_strings;
        uint32_t           search_strings_count;
        uint32_t           include_substring;
        uint32_t           include_category;
        uint32_t           partial_matching;
    } ph_string_filter_t;

    /**
     * @brief Constraints for one table or search query.
     * @note Set @p size to sizeof(ph_event_filter_t). A zero count leaves that
     *       list unconstrained. time, node_id, agent_id, process_id, and
     *       thread_id constrain the query only when is_valid is non-zero.
     * @note process_id is rocpd_info_process.id, not the OS pid. thread_id is
     *       rocpd_info_thread.id, not the OS tid. tracks are ph_track_t::id.
     *       event_types are ph_event_type_t values.
     * @note Pointers are read only during ph_event_query_create().
     */
    typedef struct
    {
        uint32_t             size;
        ph_time_filter_t     time;
        const uint32_t*      tracks;
        uint32_t             track_count;
        const uint32_t*      event_types;
        uint32_t             event_type_count;
        ph_optional_uint64_t node_id;
        ph_optional_uint64_t agent_id;
        ph_optional_uint64_t process_id;
        ph_optional_uint64_t thread_id;
        ph_string_filter_t   string_filter;
    } ph_event_filter_t;

    /** @brief Code-object address range for one call-stack frame.
     *         Strings are owned by the producing ph_ctx_t. is_valid is 0 when
     *         the frame has no address range. */
    typedef struct
    {
        uint64_t    address_base;
        uint64_t    address_low;
        uint64_t    address_high;
        const char* extdata;
        uint32_t    is_valid;
    } ph_address_range_t;

    /** @brief Program-counter location for one call-stack frame.
     *         Strings are owned by the producing ph_ctx_t. is_valid is 0 when
     *         the frame has no program counter. line_number is valid only when
     *         its is_valid flag is non-zero. event_id is the region row id of
     *         the frame, the same value as ph_event_t::id, or 0 if unknown. */
    typedef struct
    {
        uint64_t             event_id;
        const char*          function;
        const char*          filename;
        ph_optional_uint64_t line_number;
        const char*          extdata;
        uint32_t             is_valid;
    } ph_program_counter_t;

    /** @brief One frame of a call stack.
     *         Index 0 is the top of the stack, the most recent call.
     *         extdata is owned by the producing ph_ctx_t. */
    typedef struct
    {
        ph_program_counter_t program_counter;
        ph_address_range_t   address_range;
        const char*          extdata;
    } ph_stack_frame_t;

    /** @brief Call stack for one event; same lifetime rule as ph_event_list_t. */
    typedef struct
    {
        uint32_t          list_size;
        ph_stack_frame_t* frames;
    } ph_call_stack_t;

    /** @brief Common event metadata from event_data_t.
     *         Strings are owned by the producing ph_ctx_t. is_valid is 0 when
     *         the event row had no event metadata. */
    typedef struct
    {
        uint32_t    is_valid;
        uint64_t    stack_id;
        uint64_t    parent_stack_id;
        uint64_t    correlation_id;
        const char* event_category;
        const char* extdata;
    } ph_event_data_t;

    /** @brief node_info_t fields. is_valid is 0 when the detail has no node. */
    typedef struct
    {
        uint32_t    is_valid;
        uint64_t    node_id;
        uint64_t    hash;
        const char* machine_id;
        const char* system_name;
        const char* hostname;
        const char* release;
        const char* version;
        const char* hardware_name;
        const char* domain_name;
    } ph_detail_node_t;

    /** @brief process_info_t fields. id is rocpd_info_process.id. pid is the
     *         OS process id. */
    typedef struct
    {
        uint32_t             is_valid;
        uint64_t             id;
        uint64_t             pid;
        ph_optional_uint64_t ppid;
        ph_optional_uint64_t init;
        ph_optional_uint64_t fini;
        ph_optional_uint64_t start;
        ph_optional_uint64_t end;
        const char*          command;
        const char*          environment;
        const char*          extdata;
    } ph_detail_process_t;

    /** @brief thread_info_t fields. id is rocpd_info_thread.id. thread_id is
     *         the OS thread id. */
    typedef struct
    {
        uint32_t             is_valid;
        uint64_t             id;
        uint64_t             thread_id;
        ph_optional_uint64_t parent_process_id;
        ph_optional_uint64_t start;
        ph_optional_uint64_t end;
        const char*          name;
        const char*          extdata;
    } ph_detail_thread_t;

    /** @brief agent_info_t fields. id is rocpd_info_agent.id. */
    typedef struct
    {
        uint32_t             is_valid;
        uint64_t             id;
        uint64_t             type_index;
        ph_optional_uint64_t absolute_index;
        ph_optional_uint64_t logical_index;
        ph_optional_uint64_t uuid;
        const char*          agent_type;
        const char*          name;
        const char*          model_name;
        const char*          vendor_name;
        const char*          product_name;
        const char*          user_name;
        const char*          extdata;
    } ph_detail_agent_t;

    /** @brief stream_info_t fields. pid is rocpd_info_process.id, not the OS pid. */
    typedef struct
    {
        uint32_t    is_valid;
        uint64_t    stream_id;
        uint64_t    nid;
        uint64_t    pid;
        const char* name;
        const char* extdata;
    } ph_detail_stream_t;

    /** @brief queue_info_t fields. pid is rocpd_info_process.id, not the OS pid. */
    typedef struct
    {
        uint32_t    is_valid;
        uint64_t    queue_id;
        uint64_t    nid;
        uint64_t    pid;
        const char* name;
        const char* extdata;
    } ph_detail_queue_t;

    /** @brief code_object_info_t fields. */
    typedef struct
    {
        uint32_t             is_valid;
        uint64_t             id;
        ph_optional_uint64_t load_base;
        ph_optional_uint64_t load_size;
        ph_optional_uint64_t load_delta;
        const char*          uri;
        const char*          storage_type;
        const char*          extdata;
    } ph_detail_code_object_t;

    /** @brief kernel_symbol_info_t fields. */
    typedef struct
    {
        uint32_t             is_valid;
        uint64_t             id;
        ph_optional_uint64_t kernel_object;
        ph_optional_uint64_t kernarg_segment_size;
        ph_optional_uint64_t kernarg_segment_alignment;
        ph_optional_uint64_t group_segment_size;
        ph_optional_uint64_t private_segment_size;
        ph_optional_uint64_t sgpr_count;
        ph_optional_uint64_t arch_vgpr_count;
        ph_optional_uint64_t accum_vgpr_count;
        const char*          name;
        const char*          display_name;
        const char*          extdata;
    } ph_detail_kernel_symbol_t;

    /** @brief One arg_data_t. */
    typedef struct
    {
        uint64_t    position;
        const char* type;
        const char* name;
        const char* value;
        const char* extdata;
    } ph_arg_data_t;

    /** @brief region_data_t. is_valid is 0 when the event is not a region or
     *         the row was not found. args is null when list_size is 0. */
    typedef struct
    {
        uint32_t        is_valid;
        uint64_t        start_timestamp;
        uint64_t        end_timestamp;
        const char*     name;
        const char*     extdata;
        ph_event_data_t event;
        uint32_t        arg_count;
        ph_arg_data_t*  args;
    } ph_region_details_t;

    /** @brief kernel_dispatch_data_t. is_valid is 0 when the event is not a
     *         kernel dispatch or the row was not found. */
    typedef struct
    {
        uint32_t                  is_valid;
        uint64_t                  dispatch_id;
        uint64_t                  start_timestamp;
        uint64_t                  end_timestamp;
        uint64_t                  private_segment_size;
        uint64_t                  group_segment_size;
        uint64_t                  workgroup_size_x;
        uint64_t                  workgroup_size_y;
        uint64_t                  workgroup_size_z;
        uint64_t                  grid_size_x;
        uint64_t                  grid_size_y;
        uint64_t                  grid_size_z;
        const char*               name;
        const char*               extdata;
        ph_event_data_t           event;
        ph_detail_node_t          node_info;
        ph_detail_process_t       process_info;
        ph_detail_thread_t        thread_info;
        ph_detail_agent_t         agent_info;
        ph_detail_kernel_symbol_t kernel_symbol_info;
        ph_detail_code_object_t   code_object_info;
        ph_detail_stream_t        stream_info;
        ph_detail_queue_t         queue_info;
    } ph_kernel_dispatch_details_t;

    /** @brief memory_copy_data_t. is_valid is 0 when the event is not a memory
     *         copy or the row was not found. */
    typedef struct
    {
        uint32_t             is_valid;
        uint64_t             start_timestamp;
        uint64_t             end_timestamp;
        ph_optional_uint64_t dst_address;
        ph_optional_uint64_t src_address;
        uint64_t             size;
        const char*          name;
        const char*          region_name;
        const char*          extdata;
        ph_event_data_t      event;
        ph_detail_agent_t    dst_agent_id;
        ph_detail_agent_t    src_agent_id;
        ph_detail_node_t     node_info;
        ph_detail_process_t  process_info;
        ph_detail_thread_t   thread_info;
        ph_detail_stream_t   stream_info;
        ph_detail_queue_t    queue_info;
    } ph_memory_copy_details_t;

    /** @brief memory_alloc_data_t. is_valid is 0 when the event is not a memory
     *         allocation or the row was not found. */
    typedef struct
    {
        uint32_t             is_valid;
        uint64_t             start_timestamp;
        uint64_t             end_timestamp;
        ph_optional_uint64_t address;
        uint64_t             size;
        const char*          type;
        const char*          level;
        const char*          extdata;
        ph_event_data_t      event;
        ph_detail_node_t     node_info;
        ph_detail_process_t  process_info;
        ph_detail_thread_t   thread_info;
        ph_detail_agent_t    agent_info;
        ph_detail_stream_t   stream_info;
        ph_detail_queue_t    queue_info;
    } ph_memory_alloc_details_t;

    /** @brief A single PMC/counter sample (instantaneous value). */
    typedef struct
    {
        uint64_t timestamp;
        double   value;
    } ph_sample_t;

    /** @brief A list of PMC/counter samples; same lifetime rule as
     *         ph_track_list_t. */
    typedef struct
    {
        uint32_t     list_size;
        ph_sample_t* samples;
    } ph_sample_list_t;

    /**
     * @brief One histogram bucket.
     *        For a duration track, event_count is the number of overlapping
     *        events and value is their overlap in nanoseconds. For a PMC
     *        track, event_count is the sample count and value is the average
     *        sample value. A PMC gap has event_count 0 and the previous value.
     */
    typedef struct
    {
        uint32_t bucket_index;
        uint64_t event_count;
        double   value;
    } ph_histogram_bucket_t;

    /** @brief Histogram buckets for one track. buckets may be null when
     *         list_size is 0. */
    typedef struct
    {
        uint32_t               track_id;
        uint32_t               list_size;
        ph_histogram_bucket_t* buckets;
    } ph_track_histogram_t;

    /**
     * @brief Histograms for every track, aligned to one time grid.
     *        Bucket 0 begins at origin_ts and each bucket is bucket_size
     *        nanoseconds wide. Missing duration buckets read as zero.
     */
    typedef struct
    {
        uint64_t              origin_ts;
        uint64_t              bucket_size;
        uint32_t              bucket_count;
        uint32_t              track_count;
        ph_track_histogram_t* tracks;
    } ph_histogram_t;

    typedef ph_version_t ph_library_version_t;
    typedef ph_version_t ph_schema_version_t;

    /** @brief How an asynchronous operation ended. */
    typedef enum
    {
        PH_FUTURE_FINISHED,  /**< The operation ran to its end; see the ph_result_t. */
        PH_FUTURE_CANCELLED, /**< The operation was cancelled before it finished. */
        PH_FUTURE_ERROR,     /**< The operation failed; see the ph_result_t. */
    } ph_future_status_t;

    /** @brief What the library is doing, reported with a progress value.
     *         Valid only for the duration of the progress callback. May be
     *         null. */
    typedef const char* ph_progress_description_t;

    /** @brief Progress callback of a future. @p value is in [0.0, 1.0]. */
    typedef void (*ph_progress_fn)(ph_future_t               future,
                                   double                    value,
                                   ph_progress_description_t description);

    /** @brief Completion callback of a future, called exactly once. */
    typedef void (*ph_finished_fn)(ph_future_t        future,
                                   ph_future_status_t status,
                                   ph_result_t        result);

    // NOLINTEND

#ifdef __cplusplus
}
#endif

#endif  // #ifndef PROFILER_HUB_TYPES_INC
