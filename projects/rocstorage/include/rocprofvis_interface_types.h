// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


/*
* This interface header file cannot have any pre-compiler conditions to be successfully built by CFFI   
* Use CInterfaceTypes.h for C/C++ code
*/

#define INVALID_INDEX 0xFFFFFFFF
#define TABLE_QUERY_PACK_OP_TYPE(rocprofvis_dm_event_operation_t_op_type) (rocprofvis_dm_event_operation_t_op_type << 28)
#define TABLE_QUERY_UNPACK_OP_TYPE(rocprofvis_dm_track_id_t_track_id) (rocprofvis_dm_track_id_t_track_id >> 28)
#define TABLE_QUERY_UNPACK_TRACK_ID(rocprofvis_dm_track_id_t_track_id) (rocprofvis_dm_track_id_t_track_id & 0x0FFFFFFF)

/*******************************Types******************************/

typedef 	void* 		                  rocprofvis_dm_handle_t;                       // Any data model object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_trace_t;                        // Trace object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_database_t;                     // Database object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_track_t;                        // Track object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_slice_t;                        // Time slice object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_flowtrace_t;                    // Flow trace object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_stacktrace_t;                   // Stack trace object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_extdata_t;                      // External data object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_table_t;                        // Table object handle
typedef     rocprofvis_dm_handle_t        rocprofvis_dm_table_row_t;                    // Table row object handle
typedef     uint32_t                      rocprofvis_dm_index_t;                        // Any data model array index, assuming array sizes will not exceed 32-bit value
typedef     uint64_t                      rocprofvis_dm_timestamp_t;                    // Timestamp
typedef     uint32_t                      rocprofvis_dm_property_t;                     // any property enumeration
typedef     uint64_t                      rocprofvis_dm_property_index_t;               // index of an indexed property
typedef     const char*                   rocprofvis_dm_json_blob_t;                    // json blob string
typedef     uint32_t                      rocprofvis_dm_track_id_t;                     // track id
typedef     uint64_t                      rocprofvis_dm_table_id_t;                     // table id


typedef     const char*                   rocprofvis_db_filename_t;                     // input file name
typedef     const char*                   rocprofvis_db_status_message_t;               // status message for progress report callback
typedef     uint16_t                      rocprofvis_db_progress_percent_t;             // percentage value of database request progress
typedef     uint32_t*                     rocprofvis_db_track_selection_t;              // array of track IDs for time slice request
typedef     uint16_t                      rocprofvis_db_num_of_tracks_t;                // number of tracks in time slice request
typedef     void*                         rocprofvis_db_future_t;                       // future object for asynchronous communication
typedef     uint64_t                      rocprofvis_db_timeout_sec_t;                  // asynchronous call wait timeout (seconds)
typedef     const char*                   rocprofvis_dm_charptr_t;                      // pointer to string
typedef     uint64_t                      rocprofvis_dm_size_t;                         // size of array
typedef     uint8_t                       rocprofvis_dm_event_level_t;                  // event level, for stacking events
typedef     uint16_t                      rocprofvis_dm_num_string_table_filters_t;     // number of string table filters for table query
typedef     const char**                  rocprofvis_dm_string_table_filters_t;         // string table filters for table query


/*******************************Enumerations******************************/

// Operation status
typedef enum rocprofvis_dm_result_t
{
    // Operation was successful
    kRocProfVisDmResultSuccess = 0,
    // Operation failed with a non-specific error
    kRocProfVisDmResultUnknownError = 1,
    // Operation failed due to a timeout
    kRocProfVisDmResultTimeout = 2,
    // Operation failed as the data was not yet loaded
    kRocProfVisDmResultNotLoaded = 3,
    // Operation failed due to resources allocation failure
    kRocProfVisDmResultAllocFailure = 4,
    // Operation failed due to invalid parameter passed
    kRocProfVisDmResultInvalidParameter = 5,
    // Operation failed due to database access problem
    kRocProfVisDmResultDbAccessFailed = 6,
    // Operation failed due to invalid property 
    kRocProfVisDmResultInvalidProperty = 7,
    // Operation failed due to unsupported property 
    kRocProfVisDmResultNotSupported = 8,
    // Operation failed due to busy status of resource 
    kRocProfVisDmResultResourceBusy = 9,
    // Operation failed due to busy status of resource
    kRocProfVisDmResultDbAbort = 10,
} rocprofvis_dm_result_t;

// Track category
typedef enum rocprofvis_dm_track_category_t {
    // Object is not track
    kRocProfVisDmNotATrack = 0,
    // Object is PMC track (performance counters track)
    kRocProfVisDmPmcTrack = 1,
    // Object is region track (HIP calls)
    kRocProfVisDmRegionTrack = 2,
    // Object is kernel track (kernel execution)
    kRocProfVisDmKernelDispatchTrack = 3,
    // Object is SQTT track
    kRocProfVisDmSQTTTrack = 4,
    // Object is NIC track
    kRocProfVisDmNICTrack = 5,
    // Object is memory allocation track
    kRocProfVisDmMemoryAllocationTrack = 6,
    // Object is memory copy track
    kRocProfVisDmMemoryCopyTrack = 7,
    // Object is stream track
    kRocProfVisDmStreamTrack = 8,
    // Object is region sample track
    kRocProfVisDmRegionMainTrack = 9,
    // Object is region sample track
    kRocProfVisDmRegionSampleTrack = 10,
} rocprofvis_dm_track_category_t;

//Event operation
typedef enum rocprofvis_dm_event_operation_t {
    // Event of unknown operation type
    kRocProfVisDmOperationNoOp = 0,
    // Launch event
    kRocProfVisDmOperationLaunch = 1,
    // Dispatch event
    kRocProfVisDmOperationDispatch = 2,
    // Memory allocation event
    kRocProfVisDmOperationMemoryAllocate = 3,
    // Memory copy event
    kRocProfVisDmOperationMemoryCopy = 4,
    // Memory copy event
    kRocProfVisDmOperationLaunchSample = 5,
    // Number of operations
    kRocProfVisDmNumOperation = kRocProfVisDmOperationLaunchSample + 1
} rocprofvis_dm_event_operation_t;

// Database type
typedef enum rocprofvis_db_type_t {
    // input file auto-detection
    kAutodetect = 0, 
    // old schema Rocpd database
	kRocpdSqlite = 1,
    // new schema Rocprof database
	kRocprofSqlite = 2
} rocprofvis_db_type_t;

// Database query status, reported by database query progress callback
typedef enum rocprofvis_db_status_t {
    kRPVDbSuccess = 0,
    kRPVDbError = 1,
    kRPVDbBusy = 2
} rocprofvis_db_status_t;

typedef enum rocprofvis_time_bucket_size {
    kRPVDbMicrosecond = 1000,
    kRPVDbMillisecond = 1000000,
    kRPVDbSecond = 1000000000,
} rocprofvis_time_bucket_size;

typedef enum rocprofvis_db_data_type_t
{
    kRPVDataTypeDefault = 0,
    kRPVDataTypeInt     = 1,
    kRPVDataTypeDouble  = 2,
    kRPVDataTypeString  = 3,
    kRPVDataTypeBlob    = 4,
    kRPVDataTypeNull    = 5
} rocprofvis_db_data_type_t;

// Trace properties
typedef enum  rocprofvis_dm_trace_property_t {
    // Trace start time
	kRPVDMStartTimeUInt64,
    // Trace end time
	kRPVDMEndTimeUInt64,
    // Number of track
	kRPVDMNumberOfTracksUInt64,
    // Number of tables
    kRPVDMNumberOfTablesUInt64,
    // Trace memory footprint
    kRPVDMTraceMemoryFootprintUInt64,
    // Track handle by specified index
    kRPVDMTrackHandleIndexed,
    // Flow trace handle by specified event id (60-bit event id and 4-bit operation type)
	kRPVDMFlowTraceHandleByEventID,
    // Stack trace handle by specified event id (60-bit event id and 4-bit operation type)
	kRPVDMStackTraceHandleByEventID,
    // Extended handle by specified event id (60-bit event id and 4-bit operation type)
    kRPVDMExtInfoHandleByEventID,
    // Handle of a SQL query result table, by specified index  
    kRPVDMTableHandleByID,
    // Database handle
	kRPVDMDatabaseHandle,
    // Number of buckets
    kRPVDMHistogramNumBuckets,
    // Size of histogram bucket
    kRPVDMHistogramBucketSize
} rocprofvis_dm_trace_property_t;

// Track properties
typedef enum rocprofvis_dm_track_property_t {
    // Track total number of records
    kRPVDMTrackNumRecordsUInt64,
    // Track minimum timestamp
    kRPVDMTrackMinimumTimestampUInt64,
    // Track maximum timestamp
    kRPVDMTrackMaximumTimestampUInt64,
    // Track category enumeration value
	kRPVDMTrackCategoryEnumUInt64,
    // Track category string value
    kRPVDMTrackCategoryEnumCharPtr,
    // Track ID 
    kRPVDMTrackIdUInt64,
    // Track Node ID
    kRPVDMTrackNodeIdUInt64,
    // Track main process string (PID, GPUID, etc)
    kRPVDMTrackMainProcessNameCharPtr,
    // Track sub process string (TID, QueueID, PMC name)
    kRPVDMTrackSubProcessNameCharPtr,
    // Number of slices in the track
    kRPVDMNumberOfSlicesUInt64,
    // Track memory footprint
    kRPVDMTrackMemoryFootprintUInt64,
    // Slice handle by specified index
	kRPVDMSliceHandleIndexed,	
    // Slice handle by specified time
    kRPVDMSliceHandleTimed,
    // Number of extended data records
    kRPVDMNumberOfTrackExtDataRecordsUInt64,
    // Extended data category by specified index
    kRPVDMTrackExtDataCategoryCharPtrIndexed,
    // Extended data name by specified index
	kRPVDMTrackExtDataNameCharPtrIndexed,
    // Extended data value by specified index
    kRPVDMTrackExtDataValueCharPtrIndexed,
    // Extended data in Json format
	kRPVDMTrackInfoJsonCharPtr, 
    // Database handle
	kRPVDMTrackDatabaseHandle,
    // Trace handle
    kRPVDMTrackTraceHandle,
    // Track minimum level or value
    kRPVDMTrackMinimumValueDouble,
    // Track maximum level or value
    kRPVDMTrackMaximumValueDouble,
    // Histogram bucket value
    kRPVDMTrackHistogramEventDensityUInt64Indexed,
} rocprofvis_dm_track_property_t;

// Slice properties
typedef enum rocprofvis_dm_slice_property_t {
    // Convert timestamp to an index
	kRPVDMRecordIndexByTimestampUInt64,
    // Slice memory footprint
    kRPVDMSliceMemoryFootprintUInt64,
    // Number of records
	kRPVDMNumberOfRecordsUInt64,
    // Record timestamp
	kRPVDMTimestampUInt64Indexed,
    // PMC record value
	kRPVDMPmcValueDoubleIndexed,
    // Event ID
	kRPVDMEventIdUInt64Indexed,
    // Event operation enumeration (none, launch, dispatch, allocate, copy)
    kRPVDMEventIdOperationEnumIndexed,
    // Event operation string (none, launch, dispatch, allocate, copy) 
    kRPVDMEventIdOperationCharPtrIndexed,
    // Event duration. Can be negative. Negative has to be invalidated by controller 
	kRPVDMEventDurationInt64Indexed,
    // Event type string
	kRPVDMEventTypeStringCharPtrIndexed,
    // Event symbol string
	kRPVDMEventSymbolStringCharPtrIndexed,
    // Event level in graph
    kRPVDMEventLevelUInt64Indexed
} rocprofvis_dm_slice_property_t;

//Flow trace properties
typedef enum rocprofvis_dm_flowtrace_property_t {
    // Number of flow endpoints
	kRPVDMNumberOfEndpointsUInt64,
    // Flow endpoint track id
    kRPVDMEndpointTrackIDUInt64Indexed,
    // Flow endpoint event id
	kRPVDMEndpointIDUInt64Indexed,
    // Flow endpoint timestamp
	kRPVDMEndpointTimestampUInt64Indexed,
    // Flow endpoint end timestamp
    kRPVDMEndpointEndTimestampUInt64Indexed,
    // Flow endpoint category
    kRPVDMEndpointCategoryCharPtrIndexed,
    // Flow endpoint symbol
    kRPVDMEndpointSymbolCharPtrIndexed,
    // Flow endpoint level in track
    kRPVDMEndpointLevelUInt64Indexed
} rocprofvis_dm_flowtrace_property_t;

//Stack trace properties
typedef enum rocprofvis_dm_stacktrace_property_t {
    // Number of frames
	kRPVDMNumberOfFramesUInt64,
    // Stack frame depth by specified frame index
	kRPVDMFrameDepthUInt64Indexed,
    // Stack frame symbol by specified frame index
    kRPVDMFrameSymbolCharPtrIndexed,
    // Stack frame arguments by specified frame index
    kRPVDMFrameArgsCharPtrIndexed,
    // Stack frame code line by specified frame index
    kRPVDMFrameCodeLineCharPtrIndexed,
} rocprofvis_dm_stacktrace_property_t;

// Extended data object properties
typedef enum rocprofvis_dm_extdata_property_t {
    // Number of extended data records
    kRPVDMNumberOfExtDataRecordsUInt64,
    // Extended data category by specified index
    kRPVDMExtDataCategoryCharPtrIndexed,
    // Extended data name by specified index
    kRPVDMExtDataNameCharPtrIndexed,
    // Extended data value by specified index
    kRPVDMExtDataValueCharPtrIndexed,
    // Extended data type
    kRPVDMExtDataTypeUint64Indexed,
    // Extended data category enumeration
    kRPVDMExtDataEnumUint64Indexed
} rocprofvis_dm_extdata_property_t;

// Table object properties
typedef enum rocprofvis_dm_table_property_t {
    // Table id
    kRPVDMNumberOfTableIdUInt64,
    // Number of columns
    kRPVDMNumberOfTableColumnsUInt64,
    // Number of rows
    kRPVDMNumberOfTableRowsUInt64,
    // Table description string
    kRPVDMExtTableDescriptionCharPtr,
    // Table query string
    kRPVDMExtTableQueryCharPtr,
    // Column name by specified index
    kRPVDMExtTableColumnNameCharPtrIndexed,
    // Row handle by specified index
    kRPVDMExtTableRowHandleIndexed
} rocprofvis_dm_table_property_t;

// Table row object properties
typedef enum rocprofvis_dm_table_row_property_t {
    // Number of cells
    kRPVDMNumberOfTableRowCellsUInt64,
    // Row cell value by specified index
    kRPVDMExtTableRowCellValueCharPtrIndexed
} rocprofvis_dm_table_row_property_t;

// Type of data can be requested per event
typedef enum rocprofvis_dm_event_property_type_t {
    // Flow trace
    kRPVDMEventFlowTrace,
    // Stack trace
    kRPVDMEventStackTrace,
    // Extended data
    kRPVDMEventExtData,
    // Number of event property types
    kRPVDMNumEventPropertyTypes,
} rocprofvis_dm_event_property_type_t;

// Type for sort order
typedef enum rocprofvis_dm_sort_order_t {
    // Ascending sort order
    kRPVDMSortOrderAsc,
    // Descending sort order
    kRPVDMSortOrderDesc,
} rocprofvis_dm_sort_order_t;

// Event id structure
typedef union { 
    struct {
        // Event ID from database
        uint64_t    event_id:60;
        // Event operation type
        uint64_t    event_op:4;
    } bitfield;
    uint64_t        value;
} rocprofvis_dm_event_id_t;

/*******************************Callbacks******************************/

// Database request progress callback
typedef void ( *rocprofvis_db_progress_callback_t)(
		        rocprofvis_db_filename_t,
                rocprofvis_db_progress_percent_t, 
                rocprofvis_db_status_t, 
                rocprofvis_db_status_message_t,
                void*
);
