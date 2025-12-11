// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT


/*
* This interface header file cannot have any pre-compiler conditions to be successfully built by CFFI   
* Use rocprofvis_c_interface.h for C/C++ code
*/


/************************************Database interface methods*************************************/
/****************************************************************************************************
 * @brief Opens database of provided path and type
 *      kAutodetect = 0, 
 *	    kRocpdSqlite = 1,
 *	    kRocprofSqlite = 2
 * 
 * @param filename path of the database file
 * @param type  type enumeration, kAutodetect for automatic detection 
 * @return handler to database object
 * 
 * @note Currently only old rocpd schema fully supported. Working on rocprof schema
 * 
 ***************************************************************************************************/
rocprofvis_dm_database_t rocprofvis_db_open_database(
                                    rocprofvis_db_filename_t, 
                                    rocprofvis_db_type_t);

/****************************************************************************************************
 * @brief Calculates size of memory used by database object
 * 
 * @param database database handle
 * @return size of used memory
 * 
 ***************************************************************************************************/
rocprofvis_dm_size_t rocprofvis_db_get_memory_footprint(
                                    rocprofvis_dm_database_t);

/****************************************************************************************************
 * @brief Allocates future object, to be used for asynchronous operations
 * 
 * @param callback callback method to report current database request progress and status. 
 *                  May be useful for command line tools and scripts
 * @return future object handle
 * 
 ***************************************************************************************************/
rocprofvis_db_future_t rocprofvis_db_future_alloc(
                                    rocprofvis_db_progress_callback_t callback, void* user_data=nullptr);

/****************************************************************************************************
 * @brief Waits until asynchronous operation is completed or timeout expires
 * 
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @param timeout timeout in seconds for the asyncronous call to expire
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_db_future_wait(
                                    rocprofvis_db_future_t, 
                                    rocprofvis_db_timeout_sec_t);

/****************************************************************************************************
 * @brief Free future object
 * 
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * 
 ***************************************************************************************************/
void rocprofvis_db_future_free(rocprofvis_db_future_t);

/****************************************************************************************************
 * @brief Cancel future job
 *
 * @param object future handle allocated by rocprofvis_db_future_alloc
 *
 ***************************************************************************************************/
void rocprofvis_db_future_cancel(rocprofvis_db_future_t);

/****************************************************************************************************
 * @brief Asynchronous call to read data model metadata 
 *              (static objects residing in trace class memory until trace is deleted)
 * 
 * @param database database handle
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_db_read_metadata_async(
                                    rocprofvis_dm_database_t, 
                                    rocprofvis_db_future_t);

/****************************************************************************************************
 * @brief Asynchronous call to read time slice of records for provided time frame and tracks selection
 *
 * @param database database handle
 * @param start begining of the time slice
 * @param start end of the time slice
 * @param num number of tracks in rocprofvis_db_track_selection_t array (uint32*)
 * @param track tracks selection array (uint32*)
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @return status of operation
 *
 * @note Object will stay in trace memory until deleted.
 *             Use rocprofvis_dm_delete_time_slice or rocprofvis_dm_delete_all_time_slices for deletion *
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_db_read_trace_slice_async( 
                                    rocprofvis_dm_database_t,
                                    rocprofvis_dm_timestamp_t,
                                    rocprofvis_dm_timestamp_t,
                                    rocprofvis_db_num_of_tracks_t,
                                    rocprofvis_db_track_selection_t,
                                    rocprofvis_db_future_t);    

rocprofvis_dm_result_t rocprofvis_db_build_table_query(
    rocprofvis_dm_database_t database, 
    rocprofvis_dm_timestamp_t start,
    rocprofvis_dm_timestamp_t end, 
    rocprofvis_db_num_of_tracks_t num,
    rocprofvis_db_track_selection_t tracks, 
    rocprofvis_dm_charptr_t where,
    rocprofvis_dm_charptr_t filter, 
    rocprofvis_dm_charptr_t group, 
    rocprofvis_dm_charptr_t group_cols, 
    rocprofvis_dm_charptr_t sort_column, 
    rocprofvis_dm_sort_order_t sort_order, 
    rocprofvis_dm_num_string_table_filters_t num_string_table_filters, 
    rocprofvis_dm_string_table_filters_t string_table_filters,
    uint64_t max_count, 
    uint64_t offset, 
    bool count_only, 
    bool summary,
    char** out_query);

/****************************************************************************************************
 * @brief Asynchronous call to write the results of a table query to .CSV
 *
 * @param database database handle
 * @param query sql table query (see rocprofvis_db_build_table_query)
 * @param file_path output path to write .CSV
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @return status of operation
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_db_export_table_csv_async(
    rocprofvis_dm_database_t database,
    rocprofvis_dm_charptr_t query,
    rocprofvis_dm_charptr_t file_path,
    rocprofvis_db_future_t object);

rocprofvis_dm_result_t rocprofvis_db_trim_save_async(rocprofvis_dm_database_t database, rocprofvis_dm_timestamp_t start,
                                            rocprofvis_dm_timestamp_t end,
                                            rocprofvis_dm_charptr_t new_db_path, 
                                            rocprofvis_db_future_t object);

/****************************************************************************************************
 * @brief Asynchronous call to read event property of specific type
 *
 * @param database database handle
 * @param type type of propery:
 *                             kRPVDMEventFlowTrace,
 *                             kRPVDMEventStackTrace,
 *                             kRPVDMEventExtData,
 * @param event_id 60-bit event id and 4-bit operation type
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @return status of operation
 *
 * @note Object will stay in trace memory until deleted.
 *          Use rocprofvis_dm_delete_event_property_for or
 *          rocprofvis_dm_delete_all_event_properties_for for deletion
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_db_read_event_property_async(
                                    rocprofvis_dm_database_t,
                                    rocprofvis_dm_event_property_type_t,
                                    rocprofvis_dm_event_id_t,
                                    rocprofvis_db_future_t);

/****************************************************************************************************
 * @brief Asynchronous call to read a table result of specified SQL query
 *
 * @param database database handle
 * @param query SQL query string
 * @param description description of a table
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @param id new id is assigned to the table and returned using this reference pointer
 * @return status of operation
 *
 * @note Object will stay in trace memory until deleted.
 *          Use rocprofvis_dm_delete_table_at or rocprofvis_dm_delete_all_tables for deletion
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_db_execute_query_async(
                                    rocprofvis_dm_database_t,                                                                
                                    rocprofvis_dm_charptr_t,
                                    rocprofvis_dm_charptr_t,
                                    rocprofvis_db_future_t,
                                    rocprofvis_dm_table_id_t*);

/************************************Data model trace interface*************************************/

/****************************************************************************************************
 * @brief Create trace object
 *
 * @return trace object handle
 *
 * @note trace object needs to bound to a database and database interface methods should be
 *       called to fill trace with data, starting with rocprofvis_db_read_metadata_async
 ***************************************************************************************************/
rocprofvis_dm_trace_t   rocprofvis_dm_create_trace(void); 

/****************************************************************************************************
 * @brief Deleting trace and all its resources including bound database
 *
 * @param trace trace object handle
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_trace( 
                                    rocprofvis_dm_trace_t);   

/****************************************************************************************************
 * @brief Binds trace to database
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param database database object handle created with rocprofvis_db_open_database()
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_bind_trace_to_database( 
                                    rocprofvis_dm_trace_t,
                                    rocprofvis_dm_database_t);                                      

/****************************************************************************************************
 * @brief Delete time slice with specified start and end timestamps
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param start time slice start timestamp
 * @param end time slice end timestamp
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_time_slice( 
                                    rocprofvis_dm_trace_t,
                                    rocprofvis_dm_timestamp_t,
                                    rocprofvis_dm_timestamp_t);     
/****************************************************************************************************
 * @brief Delete time slice with specified handle
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param track_id track id
 * @param slice hadle
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t
rocprofvis_dm_delete_time_slice_handle(   rocprofvis_dm_trace_t    trace,
                                          rocprofvis_dm_track_id_t track,
                                          rocprofvis_dm_slice_t    handle);

/****************************************************************************************************
 * @brief Delete all time slices
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_all_time_slices( 
                                    rocprofvis_dm_trace_t);                                      

/****************************************************************************************************
 * @brief Delete event property object of specified type
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param type type of property
 *                             kRPVDMEventFlowTrace,
 *                             kRPVDMEventStackTrace,
 *                             kRPVDMEventExtData,
 * @param event_id 60-bit event id and 4-bit operation type
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_event_property_for( 
                                    rocprofvis_dm_trace_t,
                                    rocprofvis_dm_event_property_type_t,
                                    rocprofvis_dm_event_id_t);     

/****************************************************************************************************
 * @brief Delete all event property objects of specified type
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param type type of property
 *                             kRPVDMEventFlowTrace,
 *                             kRPVDMEventStackTrace,
 *                             kRPVDMEventExtData,
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_all_event_properties_for( 
                                    rocprofvis_dm_trace_t,
                                    rocprofvis_dm_event_property_type_t);        

/****************************************************************************************************
 * @brief Delete a table by specified table index.
 *        Table is created when executed SQL query using rocprofvis_db_execute_query_async method
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param index index of a table. Number of existing tables can queried by reading Trace object
 * property kRPVDMNumberOfTablesUInt64
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_table_at( 
                                    rocprofvis_dm_trace_t,
                                    rocprofvis_dm_table_id_t); 

/****************************************************************************************************
 * @brief Delete all tables.
 *        Tables are created when executed SQL queries using rocprofvis_db_execute_query_async method
 *
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 *
 * @return status of operation
 *
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_all_tables( 
                                    rocprofvis_dm_trace_t);  

/***********************************Universal property getters**************************************
*
* There are 10 property getter methods. Two methods to get property of each type:
*       uint64, int64, double, char* and rocprofvis_dm_handle_t
* First set of methods returns values as pointer references and result of operation as return value
* Second set returns property values directly. No result of operation is returned.
* In second case if operation fails, 0 or nullptr will be returned.
****************************************************************************************************/


/****************************************************************************************************
 * @brief Return property value as uint64.
 *
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 *
 * @return uint64_t value
 *
 ***************************************************************************************************/
uint64_t  rocprofvis_dm_get_property_as_uint64(
                                    rocprofvis_dm_handle_t, 	
                                    rocprofvis_dm_property_t,
                                    rocprofvis_dm_property_index_t);                                      

/****************************************************************************************************
 * @brief Return property value as int64.
 *
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 *
 * @return int64_t value
 *
 ***************************************************************************************************/
int64_t   rocprofvis_dm_get_property_as_int64(
                                    rocprofvis_dm_handle_t, 	
                                    rocprofvis_dm_property_t,
                                    rocprofvis_dm_property_index_t); 

/****************************************************************************************************
 * @brief Return property value as double.
 *
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 *
 * @return double value
 *
 ***************************************************************************************************/
double  rocprofvis_dm_get_property_as_double(
                                    rocprofvis_dm_handle_t, 	
                                    rocprofvis_dm_property_t,
                                    rocprofvis_dm_property_index_t); 

/****************************************************************************************************
 * @brief Return property value as char*.
 *
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 *
 * @return char* value
 *
 ***************************************************************************************************/
char*   rocprofvis_dm_get_property_as_charptr(
                                    rocprofvis_dm_handle_t, 	
                                    rocprofvis_dm_property_t,
                                    rocprofvis_dm_property_index_t); 

/****************************************************************************************************
 * @brief Return property value as rocprofvis_dm_handle_t.
 *
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 *
 * @return rocprofvis_dm_handle_t value
 *
 ***************************************************************************************************/
rocprofvis_dm_handle_t  rocprofvis_dm_get_property_as_handle(
                                    rocprofvis_dm_handle_t, 	
                                    rocprofvis_dm_property_t,
                                    rocprofvis_dm_property_index_t); 

                           
