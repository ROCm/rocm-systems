// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Internal type definitions shared between datamodel and database layers

#include "rocprofvis_c_interface_types.h"
#include "rocprofvis_reader_enums.h"

#include <algorithm>
#include <cassert>
#include <list>
#include <map>
#include <string>
#include <vector>

/*******************************Assert Macros******************************/

#define ROCPROFVIS_ASSERT(cond) assert(cond)
#define ROCPROFVIS_ASSERT_MSG(cond, msg) assert((cond) && (msg))
#define ROCPROFVIS_ASSERT_RETURN(cond, retval) \
    if (!(cond)) { assert(cond); return retval; }
#define ROCPROFVIS_ASSERT_MSG_RETURN(cond, msg, retval) \
    if (!(cond)) { assert((cond) && (msg)); return retval; }
#define ROCPROFVIS_ASSERT_MSG_BREAK(cond, msg) \
    assert((cond) && (msg)); if (!(cond)) break
#define ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(msg, retval) \
    assert(false && (msg)); return retval
#define ROCPROFVIS_UNIMPLEMENTED assert(false && "Unimplemented")

/*******************************Error Strings******************************/

namespace RocProfVis {
namespace DataModel {
    inline const char* ERROR_INDEX_OUT_OF_RANGE = "Index out of range";
    inline const char* ERROR_TRACE_CANNOT_BE_NULL = "Trace reference cannot be NULL";
    inline const char* ERROR_TRACK_CANNOT_BE_NULL = "Track reference cannot be NULL";
    inline const char* ERROR_SLICE_CANNOT_BE_NULL = "Slice reference cannot be NULL";
    inline const char* ERROR_DATABASE_CANNOT_BE_NULL = "Database reference cannot be NULL";
    inline const char* ERROR_TRACK_PARAMETERS_NOT_ASSIGNED = "Track parameters not assigned";
    inline const char* ERROR_VIRTUAL_METHOD_CALL = "Virtual method call";
    inline const char* ERROR_FUTURE_CANNOT_BE_NULL = "Future reference cannot be NULL";
    inline const char* ERROR_FUTURE_CANNOT_BE_USED = "Future cannot be used";
    inline const char* ERROR_METADATA_IS_NOT_LOADED = "Metadata is not loaded";
    inline const char* ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL = "Trace properties cannot be NULL";
    inline const char* ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH = "Query parameters mismatch";
    inline const char* ERROR_MEMORY_ALLOCATION_FAILURE = "Memory allocation failure";
    inline const char* ERROR_VIRTUAL_METHOD_PROPERTY = "Virtual method property";
    inline const char* ERROR_INVALID_PROPERTY_GETTER = "Invalid property getter";
    inline const char* ERROR_UNSUPPORTED_PROPERTY = "Unsupported property";
    inline const char* ERROR_FLOW_TRACE_CANNOT_BE_NULL = "Flow trace cannot be NULL";
    inline const char* ERROR_STACK_TRACE_CANNOT_BE_NULL = "Stack trace cannot be NULL";
    inline const char* ERROR_TABLE_CANNOT_BE_NULL = "Table cannot be NULL";
    inline const char* ERROR_TABLE_ROW_CANNOT_BE_NULL = "Table row cannot be NULL";
    inline const char* ERROR_EXT_DATA_CANNOT_BE_NULL = "Extended data cannot be NULL";
    inline const char* ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL = "SQL query parameters cannot be NULL";
    inline const char* ERROR_REFERENCE_POINTER_CANNOT_BE_NULL = "Reference pointer cannot be NULL";
}  // namespace DataModel
}  // namespace RocProfVis

// For convenience, import into global namespace
using RocProfVis::DataModel::ERROR_INDEX_OUT_OF_RANGE;
using RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_TRACK_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_SLICE_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_TRACK_PARAMETERS_NOT_ASSIGNED;
using RocProfVis::DataModel::ERROR_VIRTUAL_METHOD_CALL;
using RocProfVis::DataModel::ERROR_FUTURE_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_FUTURE_CANNOT_BE_USED;
using RocProfVis::DataModel::ERROR_METADATA_IS_NOT_LOADED;
using RocProfVis::DataModel::ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH;
using RocProfVis::DataModel::ERROR_MEMORY_ALLOCATION_FAILURE;
using RocProfVis::DataModel::ERROR_VIRTUAL_METHOD_PROPERTY;
using RocProfVis::DataModel::ERROR_INVALID_PROPERTY_GETTER;
using RocProfVis::DataModel::ERROR_UNSUPPORTED_PROPERTY;
using RocProfVis::DataModel::ERROR_FLOW_TRACE_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_STACK_TRACE_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_TABLE_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_TABLE_ROW_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_EXT_DATA_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL;
using RocProfVis::DataModel::ERROR_REFERENCE_POINTER_CANNOT_BE_NULL;

/*******************************Profiling Macros******************************/

// No-op profiling macros (define TEST to enable profiling)
#define PROFILE
#define PROFILE_PROP_ACCESS(property, index)

/*******************************Types******************************/

typedef uint32_t    rocprofvis_dm_node_id_t;
typedef uint64_t    rocprofvis_dm_process_id_t;
typedef uint64_t    rocprofvis_dm_stream_id_t;
typedef std::string rocprofvis_dm_string_t;
typedef uint32_t    rocprofvis_dm_op_t;
typedef int64_t     rocprofvis_dm_duration_t;
typedef uint64_t    rocprofvis_dm_id_t;
typedef double      rocprofvis_dm_value_t;
typedef uint64_t    rocprofvis_db_timeout_ms_t;  // asynchronous call wait timeout (milliseconds)
typedef void*       rocprofvis_db_connection_t;

/*******************************Structures******************************/

// rocprofvis_db_record_data_t is used to pass record data from database to data model
typedef union {
    struct event_record_t
    {
        rocprofvis_dm_event_id_t  id;        // 60-bit event id and 4-bit operation type
        rocprofvis_dm_timestamp_t timestamp; // 64-bit timestamp
        rocprofvis_dm_duration_t  duration;  // signed 64-bit duration
        rocprofvis_dm_id_t        category;  // 32-bit category index
        rocprofvis_dm_id_t        symbol;    // 32-bit symbol index
        rocprofvis_dm_event_level_t level;
    } event;
    struct pmc_record_t
    {
        rocprofvis_dm_timestamp_t timestamp; // 64-bit timestamp
        rocprofvis_dm_value_t     value;     // double precision performance counter value
    } pmc;
} rocprofvis_db_record_data_t;

// Track identification parameters
#define NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS 3
#define TRACK_ID_NODE         0
#define TRACK_ID_PID          1
#define TRACK_ID_STREAM       1
#define TRACK_ID_AGENT        1
#define TRACK_ID_PID_OR_AGENT 1
#define TRACK_ID_TID          2
#define TRACK_ID_QUEUE        2
#define TRACK_ID_TID_OR_QUEUE 2
#define TRACK_ID_COUNTER      2
#define TRACK_ID_CATEGORY     3

typedef struct
{
    uint64_t id;
    uint64_t start_time;
    uint64_t end_time;
    uint32_t level;
} rocprofvis_event_timing_params_t;

typedef enum rocprofvis_db_query_type_t
{
    kRPVQuerySliceByQueue,
    kRPVQuerySliceByStream,
    kRPVQueryTable,
    kRPVQueryLevel,
    kRPVNumQueryTypes,
    kRPVQuerySliceByTrackSliceQuery,
} rocprofvis_db_query_type_t;

typedef struct rocprofvis_dm_process_identifiers_t
{
    rocprofvis_dm_track_category_t category;
    rocprofvis_dm_process_id_t     id[NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS];
    rocprofvis_dm_string_t         tag[NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS];
    rocprofvis_dm_string_t         name[NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS];
    bool                           is_numeric[NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS];
} rocprofvis_dm_process_identifiers_t;

typedef struct
{
    rocprofvis_dm_track_id_t            track_id;
    rocprofvis_dm_process_identifiers_t process;
    std::vector<rocprofvis_dm_string_t> query[kRPVNumQueryTypes];
    rocprofvis_dm_extdata_t             extdata;
    uint64_t                            record_count;
    rocprofvis_dm_timestamp_t           min_ts;
    rocprofvis_dm_timestamp_t           max_ts;
    std::list<rocprofvis_event_timing_params_t> m_active_events;
    rocprofvis_dm_value_t               min_value;
    rocprofvis_dm_value_t               max_value;
    std::map<uint32_t, uint32_t>        histogram;
} rocprofvis_dm_track_params_t;

typedef struct
{
    rocprofvis_dm_timestamp_t start_time;
    rocprofvis_dm_timestamp_t end_time;
    rocprofvis_dm_timestamp_t events_count[kRocProfVisDmNumOperation];
    uint64_t                  histogram_bucket_size;
    uint64_t                  histogram_bucket_count;
    bool                      metadata_loaded;
} rocprofvis_dm_trace_params_t;

typedef struct
{
    rocprofvis_dm_event_id_t    id;
    rocprofvis_dm_timestamp_t   time;
    rocprofvis_dm_track_id_t    track_id;
    rocprofvis_dm_id_t          symbol_id;
    rocprofvis_dm_id_t          category_id;
    rocprofvis_dm_event_level_t level;
    rocprofvis_dm_timestamp_t   end_time;
} rocprofvis_db_flow_data_t;

typedef struct
{
    rocprofvis_dm_charptr_t symbol;
    rocprofvis_dm_charptr_t args;
    rocprofvis_dm_charptr_t line;
    uint32_t                depth;
} rocprofvis_db_stack_data_t;

typedef struct
{
    rocprofvis_dm_charptr_t               category;
    rocprofvis_dm_charptr_t               name;
    rocprofvis_dm_charptr_t               data;
    rocprofvis_db_data_type_t             type;
    rocprofvis_event_data_category_enum_t category_enum;
} rocprofvis_db_ext_data_t;

/***********************Trace to Database binding info******************************/

typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_track_func_t)(const rocprofvis_dm_trace_t    object,
                                                                 rocprofvis_dm_track_params_t* params);
typedef rocprofvis_dm_slice_t (*rocprofvis_dm_add_slice_func_t)(const rocprofvis_dm_trace_t     object,
                                                                const rocprofvis_dm_track_id_t  track_id,
                                                                const rocprofvis_dm_timestamp_t start,
                                                                const rocprofvis_dm_timestamp_t end);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_record_func_t)(const rocprofvis_dm_slice_t object,
                                                                  rocprofvis_db_record_data_t& data);
typedef rocprofvis_dm_index_t (*rocprofvis_dm_add_string_func_t)(const rocprofvis_dm_trace_t object,
                                                                 const char*                 stringValue);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_flow_func_t)(const rocprofvis_dm_slice_t object,
                                                                rocprofvis_db_flow_data_t&  data);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_stack_frame_func_t)(const rocprofvis_dm_stacktrace_t object,
                                                                       rocprofvis_db_stack_data_t&      data);
typedef rocprofvis_dm_flowtrace_t (*rocprofvis_dm_add_flowtrace_func_t)(const rocprofvis_dm_trace_t object,
                                                                        rocprofvis_dm_event_id_t    event_id);
typedef rocprofvis_dm_stacktrace_t (*rocprofvis_dm_add_stacktrace_func_t)(const rocprofvis_dm_trace_t object,
                                                                          rocprofvis_dm_event_id_t    event_id);
typedef rocprofvis_dm_extdata_t (*rocprofvis_dm_add_extdata_func_t)(const rocprofvis_dm_trace_t object,
                                                                    rocprofvis_dm_event_id_t    event_id);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_extdata_record_func_t)(const rocprofvis_dm_extdata_t object,
                                                                          rocprofvis_db_ext_data_t&     data);
typedef rocprofvis_dm_table_t (*rocprofvis_dm_add_table_func_t)(const rocprofvis_dm_trace_t object,
                                                                rocprofvis_dm_charptr_t     query,
                                                                rocprofvis_dm_charptr_t     description);
typedef rocprofvis_dm_table_row_t (*rocprofvis_dm_add_table_row_func_t)(const rocprofvis_dm_table_t object);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_table_column_func_t)(const rocprofvis_dm_table_t object,
                                                                        rocprofvis_dm_charptr_t     column_name);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_table_row_cell_func_t)(const rocprofvis_dm_table_t object,
                                                                          rocprofvis_dm_charptr_t     cell_value);
typedef rocprofvis_dm_result_t (*rocprofvis_db_find_cached_table_value_func_t)(
    const rocprofvis_dm_database_t object,
    rocprofvis_dm_charptr_t        table,
    const rocprofvis_dm_id_t       id,
    rocprofvis_dm_charptr_t        column,
    rocprofvis_dm_charptr_t*       value);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_add_event_level_func_t)(const rocprofvis_dm_trace_t object,
                                                                       rocprofvis_dm_event_id_t    event_id,
                                                                       uint8_t                     level);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_check_slice_exists_t)(
    const rocprofvis_dm_trace_t     object,
    const rocprofvis_dm_timestamp_t start,
    const rocprofvis_dm_timestamp_t end,
    const rocprofvis_db_num_of_tracks_t   num,
    const rocprofvis_db_track_selection_t tracks);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_check_event_property_exists_t)(
    const rocprofvis_dm_trace_t        object,
    rocprofvis_dm_event_property_type_t type,
    const rocprofvis_dm_event_id_t     event_id);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_check_table_exists_t)(const rocprofvis_dm_trace_t object,
                                                                     const rocprofvis_dm_table_id_t table_id);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_complete_slice_func_t)(const rocprofvis_dm_slice_t object);
typedef rocprofvis_dm_result_t (*rocprofvis_dm_remove_slice_func_t)(const rocprofvis_dm_trace_t    trace,
                                                                    const rocprofvis_dm_track_id_t track_id,
                                                                    const rocprofvis_dm_slice_t    object);

typedef struct
{
    rocprofvis_dm_trace_t                       trace_object;
    rocprofvis_dm_trace_params_t*               trace_properties;
    rocprofvis_dm_add_track_func_t              FuncAddTrack;
    rocprofvis_dm_add_slice_func_t              FuncAddSlice;
    rocprofvis_dm_add_record_func_t             FuncAddRecord;
    rocprofvis_dm_add_string_func_t             FuncAddString;
    rocprofvis_dm_add_flowtrace_func_t          FuncAddFlowTrace;
    rocprofvis_dm_add_flow_func_t               FuncAddFlow;
    rocprofvis_dm_add_stacktrace_func_t         FuncAddStackTrace;
    rocprofvis_dm_add_stack_frame_func_t        FuncAddStackFrame;
    rocprofvis_dm_add_extdata_func_t            FuncAddExtData;
    rocprofvis_dm_add_extdata_record_func_t     FuncAddExtDataRecord;
    rocprofvis_dm_add_table_func_t              FuncAddTable;
    rocprofvis_dm_add_table_row_func_t          FuncAddTableRow;
    rocprofvis_dm_add_table_column_func_t       FuncAddTableColumn;
    rocprofvis_dm_add_table_row_cell_func_t     FuncAddTableRowCell;
    rocprofvis_db_find_cached_table_value_func_t FuncFindCachedTableValue;
    rocprofvis_dm_add_event_level_func_t        FuncAddEventLevel;
    rocprofvis_dm_check_slice_exists_t          FuncCheckSliceExists;
    rocprofvis_dm_check_event_property_exists_t FuncCheckEventPropertyExists;
    rocprofvis_dm_check_table_exists_t          FuncCheckTableExists;
    rocprofvis_dm_complete_slice_func_t         FuncCompleteSlice;
    rocprofvis_dm_remove_slice_func_t           FuncRemoveSlice;
} rocprofvis_dm_db_bind_struct;

inline uint64_t hash_combine(uint64_t a, uint64_t b)
{
    a ^= b + 0x9e3779b97f4a7c15 + (a << 12) + (a >> 4);
    return a;
}
