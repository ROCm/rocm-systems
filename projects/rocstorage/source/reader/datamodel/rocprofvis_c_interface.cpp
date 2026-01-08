// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_c_interface.h"
#include "internal_types.h"
#include "rocprofvis_db_rocpd.h"
#include "rocprofvis_db_rocprof.h"
#include "rocprofvis_dm_trace.h"

#include <rocstorage/trace.hpp>
#include <rocstorage/database.hpp>
#include <rocstorage/reader.hpp>

#ifdef TEST
#define ROCPROFVIS_DM_PROPSYMBOL(handle, property) ((RocProfVis::DataModel::DmBase*)handle)->GetPropertySymbol(property)
#else
#define ROCPROFVIS_DM_PROPSYMBOL(handle, property) 
#endif


/******************************************DATABASE INTERFACE***************************************/
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
                                        rocprofvis_db_filename_t filename,
                                        rocprofvis_db_type_t db_type){
    PROFILE;
    if (db_type == rocprofvis_db_type_t::kAutodetect) {
        db_type = RocProfVis::DataModel::Database::Autodetect(filename);
    }
    if (db_type == rocprofvis_db_type_t::kRocpdSqlite)
    {
        try {
            RocProfVis::DataModel::Database* db = new RocProfVis::DataModel::RocpdDatabase(filename);
            if (kRocProfVisDmResultSuccess == db->Open()) {
                return db;
            } else {
                ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Failed to open database!",
                                                    nullptr);
            }
        }
        catch(std::exception ex)
        {
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(
                RocProfVis::DataModel::ERROR_MEMORY_ALLOCATION_FAILURE, nullptr);
        }
    } else if (db_type == rocprofvis_db_type_t::kRocprofSqlite)
    {
        try {
            RocProfVis::DataModel::Database* db = new RocProfVis::DataModel::RocprofDatabase(filename);
            if (kRocProfVisDmResultSuccess == db->Open()) {
                return db;
            }
            else {
                ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Failed to open database!",
                                                    nullptr);
            }
        }
        catch (std::exception ex)
        {
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(
                RocProfVis::DataModel::ERROR_MEMORY_ALLOCATION_FAILURE, nullptr);
        }
    }
    else
    {
        spdlog::debug("Database type not supported!");
        return nullptr;
    }
}

/****************************************************************************************************
 * @brief Calculates size of memory used by database object
 * 
 * @param database database handle
 * @return size of used memory
 * 
 ***************************************************************************************************/

rocprofvis_dm_size_t rocprofvis_db_get_memory_footprint(rocprofvis_dm_database_t database){
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 0);
    rocstorage::database wrapper(database);
    return wrapper.memory_footprint();
}

/****************************************************************************************************
 * @brief Allocates future object, to be used for asynchronous operations
 * 
 * @param callback callback method to report current database request progress and status. 
 *                  May be useful for command line tools and scripts
 * @return future object handle
 * 
 ***************************************************************************************************/
rocprofvis_db_future_t rocprofvis_db_future_alloc(
                                        rocprofvis_db_progress_callback_t callback, void* user_data){
    PROFILE;
    try{
        return new RocProfVis::DataModel::Future(callback, user_data);
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(
            RocProfVis::DataModel::ERROR_MEMORY_ALLOCATION_FAILURE, nullptr);
    }
}

/****************************************************************************************************
 * @brief Waits until asynchronous operation is completed or timeout expires
 * 
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @param timeout timeout in seconds for the asyncronous call to expire
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_db_future_wait(
                                        rocprofvis_db_future_t object, 
                                        rocprofvis_db_timeout_sec_t timeout){
    PROFILE;
    RocProfVis::DataModel::Future * future = (RocProfVis::DataModel::Future*) object;
    ROCPROFVIS_ASSERT_MSG_RETURN(future,
                                 RocProfVis::DataModel::ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
#ifdef DEBUG
    return future->WaitForCompletion(timeout == UINT64_MAX ? timeout : timeout * 100000);
#else
    return future->WaitForCompletion(timeout == UINT64_MAX ? timeout : timeout * 1000);
#endif                                
}

/****************************************************************************************************
 * @brief Free future object
 * 
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * 
 ***************************************************************************************************/
void rocprofvis_db_future_free(
                                        rocprofvis_db_future_t object){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(object,
                                 RocProfVis::DataModel::ERROR_FUTURE_CANNOT_BE_NULL, );
    delete (RocProfVis::DataModel::Future*) object;
}

/****************************************************************************************************
 * @brief Cancel future job
 *
 * @param object future handle allocated by rocprofvis_db_future_alloc
 *
 ***************************************************************************************************/
void
rocprofvis_db_future_cancel(rocprofvis_db_future_t object)
{
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(object,
                                 RocProfVis::DataModel::ERROR_FUTURE_CANNOT_BE_NULL, );
    ((RocProfVis::DataModel::Future*) object)->SetInterrupted();
}

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
                                        rocprofvis_dm_database_t database,
                                        rocprofvis_db_future_t object){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    return ((RocProfVis::DataModel::Database*) database)->ReadTraceMetadataAsync(object);
}

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
 *             Use rocprofvis_dm_delete_time_slice or rocprofvis_dm_delete_all_time_slices for deletion
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_db_read_trace_slice_async(
                                        rocprofvis_dm_database_t database,
                                        rocprofvis_dm_timestamp_t start,
                                        rocprofvis_dm_timestamp_t end,
                                        rocprofvis_db_num_of_tracks_t num,
                                        rocprofvis_db_track_selection_t tracks,
                                        rocprofvis_db_future_t object){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    return ((RocProfVis::DataModel::Database*) database)->ReadTraceSliceAsync(start,end,num,tracks,object);
}

rocprofvis_dm_result_t rocprofvis_db_build_table_query(
    rocprofvis_dm_database_t database,
    rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end,
    rocprofvis_db_num_of_tracks_t num, rocprofvis_db_track_selection_t tracks,
    rocprofvis_dm_charptr_t where, rocprofvis_dm_charptr_t filter,
    rocprofvis_dm_charptr_t group, rocprofvis_dm_charptr_t group_cols,
    rocprofvis_dm_charptr_t sort_column, rocprofvis_dm_sort_order_t sort_order,
    rocprofvis_dm_num_string_table_filters_t num_string_table_filters, rocprofvis_dm_string_table_filters_t string_table_filters,
    uint64_t max_count, uint64_t offset,
    bool count_only, bool summary,
    char** out_query)
{
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(out_query, "Error! Query cannot be null.",
                                 kRocProfVisDmResultInvalidParameter);
    std::string query;
    rocprofvis_dm_result_t result = ((RocProfVis::DataModel::Database*) database)->BuildTableQuery(start, end, num, tracks, where, filter, group, group_cols, sort_column, sort_order,
                                                        num_string_table_filters, string_table_filters, max_count, offset, count_only,
                                                        summary, query);
    if (result == kRocProfVisDmResultSuccess)
    {
        char* ptr = (char*) calloc(query.length() + 1, 1);
        ROCPROFVIS_ASSERT_MSG_RETURN(ptr, "Error! Couldn't allocate query string.",
                                     kRocProfVisDmResultAllocFailure);
        strncpy(ptr, query.c_str(), query.length());
        *out_query = ptr;
    }
    return result;
}

rocprofvis_dm_result_t rocprofvis_db_export_table_csv_async(
    rocprofvis_dm_database_t database,
    rocprofvis_dm_charptr_t query,
    rocprofvis_dm_charptr_t file_path,
    rocprofvis_db_future_t object)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(object,
                                 RocProfVis::DataModel::ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(file_path, "Error! Output path cannot be null.",
                                kRocProfVisDmResultInvalidParameter);
    return ((RocProfVis::DataModel::Database*) database)->ExportTableCSVAsync(query, file_path, object);
}

rocprofvis_dm_result_t rocprofvis_db_trim_save_async(rocprofvis_dm_database_t database, rocprofvis_dm_timestamp_t start,
                                            rocprofvis_dm_timestamp_t end,
                                            rocprofvis_dm_charptr_t new_db_path,
                                            rocprofvis_db_future_t object)
{
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(object,
                                 RocProfVis::DataModel::ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(new_db_path, "Error! Database path cannot be null.",
                                 kRocProfVisDmResultInvalidParameter);
    return ((RocProfVis::DataModel::Database*) database)->SaveTrimmedDataAsync(start, end, new_db_path, object);
}

/****************************************************************************************************
 * @brief Asynchronous call to read event property of specific type
 *                                                     
 * @param database database handle
 * @param type type of propery:
 *                             kEventFlowTrace,
 *                             kEventStackTrace,
 *                             kEventExtData,
 * @param event_id 60-bit event id and 4-bit operation type
 * @param object future handle allocated by rocprofvis_db_future_alloc
 * @return status of operation
 * 
 * @note Object will stay in trace memory until deleted. 
 *          Use rocprofvis_dm_delete_event_property_for or 
 *          rocprofvis_dm_delete_all_event_properties_for for deletion
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_db_read_event_property_async(
                                        rocprofvis_dm_database_t database,
                                        rocprofvis_dm_event_property_type_t type,
                                        rocprofvis_dm_event_id_t event_id,
                                        rocprofvis_db_future_t object){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    return ((RocProfVis::DataModel::Database*) database)->ReadEventPropertyAsync(type, event_id, object);
}

/****************************************************************************************************
 * @brief Asynchronous call to read a table result of specified SQL query
 *                                                     
 * @param database database object handle
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
                                        rocprofvis_dm_database_t database,
                                        rocprofvis_dm_charptr_t query,
                                        rocprofvis_dm_charptr_t description,
                                        rocprofvis_db_future_t object,
                                        rocprofvis_dm_table_id_t* id)
{
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    return ((RocProfVis::DataModel::Database*) database)->ExecuteQueryAsync(query, description, object, id);
}

/*******************************************TRACE INTERFACE*****************************************/

/****************************************************************************************************
 * @brief Create trace object
 * 
 * @return trace object handle
 * 
 * @note trace object needs to bound to a database and database interface methods should be 
 *       called to fill trace with data, starting with rocprofvis_db_read_metadata_async  
 ***************************************************************************************************/
rocprofvis_dm_trace_t  rocprofvis_dm_create_trace(){
    PROFILE;
    try{
        return new RocProfVis::DataModel::Trace();
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(
            RocProfVis::DataModel::ERROR_MEMORY_ALLOCATION_FAILURE, nullptr);
    }
}

/****************************************************************************************************
 * @brief Deleting trace and all its resources including bound database
 *                                                     
 * @param trace trace object handle
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_dm_delete_trace(
                                        rocprofvis_dm_trace_t trace)
{
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    RocProfVis::DataModel::Trace* _trace = (RocProfVis::DataModel::Trace*)trace;
    RocProfVis::DataModel::Database* db = (RocProfVis::DataModel::Database*)((RocProfVis::DataModel::Trace*)trace)->Database();
    if (db) delete db;
    delete _trace;
    return kRocProfVisDmResultSuccess;
}                                      

/****************************************************************************************************
 * @brief Binds trace to database
 *                                                     
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param database database object handle created with rocprofvis_db_open_database()
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_dm_bind_trace_to_database(   rocprofvis_dm_trace_t trace,
                                        rocprofvis_dm_database_t database){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(database,
                                 RocProfVis::DataModel::ERROR_DATABASE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocprofvis_dm_db_bind_struct* bind_data=nullptr;
    if  (kRocProfVisDmResultSuccess == ((RocProfVis::DataModel::Trace*)trace)->BindDatabase(database, bind_data) &&
         kRocProfVisDmResultSuccess == ((RocProfVis::DataModel::Database*)database)->BindTrace(bind_data))
         return kRocProfVisDmResultSuccess;
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Cannot bind trace to database",
                                        kRocProfVisDmResultUnknownError);
} 

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
rocprofvis_dm_result_t rocprofvis_dm_delete_time_slice(
                                        rocprofvis_dm_trace_t trace,
                                        rocprofvis_dm_timestamp_t start,
                                        rocprofvis_dm_timestamp_t end){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocstorage::trace wrapper(trace);
    return wrapper.delete_slice(start, end)
        ? kRocProfVisDmResultSuccess
        : kRocProfVisDmResultUnknownError;
}  

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
                                          rocprofvis_dm_slice_t    handle)
{
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocstorage::trace wrapper(trace);
    return wrapper.delete_slice(track, handle)
        ? kRocProfVisDmResultSuccess
        : kRocProfVisDmResultUnknownError;
}

/****************************************************************************************************
 * @brief Delete all time slices 
 *                                                     
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t rocprofvis_dm_delete_all_time_slices(
                                        rocprofvis_dm_trace_t trace){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocstorage::trace wrapper(trace);
    return wrapper.delete_all_slices()
        ? kRocProfVisDmResultSuccess
        : kRocProfVisDmResultUnknownError;
}

/****************************************************************************************************
 * @brief Delete event property object of specified type
 *                                                     
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param type type of property
 *                             kEventFlowTrace,
 *                             kEventStackTrace,
 *                             kEventExtData,
 * @param event_id 60-bit event id and 4-bit operation type
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_event_property_for(
                                        rocprofvis_dm_trace_t trace,
                                        rocprofvis_dm_event_property_type_t type,
                                        rocprofvis_dm_event_id_t event_id){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocstorage::trace wrapper(trace);
    return wrapper.delete_event_property(
               static_cast<rocstorage::event_property_type>(type),
               rocstorage::event_id::from_raw(event_id.value))
        ? kRocProfVisDmResultSuccess
        : kRocProfVisDmResultUnknownError;
}     

/****************************************************************************************************
 * @brief Delete all event property objects of specified type
 *                                                     
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param type type of property
 *                             kEventFlowTrace,
 *                             kEventStackTrace,
 *                             kEventExtData,
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_all_event_properties_for(
                                        rocprofvis_dm_trace_t trace,
                                        rocprofvis_dm_event_property_type_t type){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocstorage::trace wrapper(trace);
    return wrapper.delete_all_event_properties(
               static_cast<rocstorage::event_property_type>(type))
        ? kRocProfVisDmResultSuccess
        : kRocProfVisDmResultUnknownError;
}    

/****************************************************************************************************
 * @brief Delete a table by specified table id. 
 *        Table is created when executed SQL query using rocprofvis_db_execute_query_async method
 *                                                     
 * @param trace trace object handle created with rocprofvis_dm_create_trace()
 * @param index index of a table. Number of existing tables can queried by reading Trace object
 * property kNumberOfTablesUInt64   
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_delete_table_at(
                                        rocprofvis_dm_trace_t trace,
                                        rocprofvis_dm_table_id_t id){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocstorage::trace wrapper(trace);
    return wrapper.delete_table(id)
        ? kRocProfVisDmResultSuccess
        : kRocProfVisDmResultUnknownError;
}

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
                                        rocprofvis_dm_trace_t trace){
    PROFILE;
    ROCPROFVIS_ASSERT_MSG_RETURN(trace, RocProfVis::DataModel::ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocstorage::trace wrapper(trace);
    return wrapper.delete_all_tables()
        ? kRocProfVisDmResultSuccess
        : kRocProfVisDmResultUnknownError;
}

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
 * @param value pointer to a uint64_t value
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_uint64(
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index,
                                        uint64_t* value){
    PROFILE_PROP_ACCESS(ROCPROFVIS_DM_PROPSYMBOL(handle, property), index);
    RocProfVis::DataModel::DmBase* object = (RocProfVis::DataModel::DmBase*) handle;
    if (object->Mutex() != nullptr)
    {
        std::shared_lock<std::shared_mutex> lock =
            std::shared_lock<std::shared_mutex>(*object->Mutex());
        return object->GetPropertyAsUint64(property, index, value);
    }
    return object->GetPropertyAsUint64(property, index, value);
}                                      

/****************************************************************************************************
 * @brief Return property value as int64. 
 *                                                     
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 * @param value pointer to a int64_t value
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_int64(
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index,
                                        int64_t* value){
    PROFILE_PROP_ACCESS(ROCPROFVIS_DM_PROPSYMBOL(handle, property), index);
    RocProfVis::DataModel::DmBase*      object = (RocProfVis::DataModel::DmBase*) handle;
    if(object->Mutex() != nullptr)
    {
        std::shared_lock<std::shared_mutex> lock =
            std::shared_lock<std::shared_mutex>(*object->Mutex());
        return object->GetPropertyAsInt64(property, index, value);
    }
    return object->GetPropertyAsInt64(property, index, value);
}                                      

/****************************************************************************************************
 * @brief Return property value as double. 
 *                                                     
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 * @param value pointer to a double type value
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_double(
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index,
                                        double* value){
    PROFILE_PROP_ACCESS(ROCPROFVIS_DM_PROPSYMBOL(handle, property), index);
    RocProfVis::DataModel::DmBase*      object = (RocProfVis::DataModel::DmBase*) handle;
    if(object->Mutex() != nullptr)
    {
        std::shared_lock<std::shared_mutex> lock =
            std::shared_lock<std::shared_mutex>(*object->Mutex());
        return object->GetPropertyAsDouble(property, index, value);
    }
    return object->GetPropertyAsDouble(property, index, value);
}                                       

/****************************************************************************************************
 * @brief Return property value as array of characters. 
 *                                                     
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 * @param value pointer to an to array of characters
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_charptr(
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index,
                                        char** value){
    PROFILE_PROP_ACCESS(ROCPROFVIS_DM_PROPSYMBOL(handle, property), index);
    RocProfVis::DataModel::DmBase*      object = (RocProfVis::DataModel::DmBase*) handle;
    if(object->Mutex() != nullptr)
    {
        std::shared_lock<std::shared_mutex> lock =
            std::shared_lock<std::shared_mutex>(*object->Mutex());
        return object->GetPropertyAsCharPtr(property, index, value);
    }
    return object->GetPropertyAsCharPtr(property, index, value);
}                                       

/****************************************************************************************************
 * @brief Return property value as rocprofvis_dm_handle_t (void*). 
 *                                                     
 * @param handle any object handle
 * @param property enumeration of properties for specified handle type
 * @param index index of any indexed property
 * @param value pointer to rocprofvis_dm_handle_t 
 * 
 * @return status of operation
 * 
 ***************************************************************************************************/
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_handle(
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index,
                                        rocprofvis_dm_handle_t* value){
    PROFILE_PROP_ACCESS(ROCPROFVIS_DM_PROPSYMBOL(handle, property), index);
    RocProfVis::DataModel::DmBase*      object = (RocProfVis::DataModel::DmBase*) handle;
    if(object->Mutex() != nullptr)
    {
        std::shared_lock<std::shared_mutex> lock =
            std::shared_lock<std::shared_mutex>(*object->Mutex());
        return object->GetPropertyAsHandle(property, index, value);
    }
    return object->GetPropertyAsHandle(property, index, value);
}  


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
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index){
    uint64_t value=0;
    if (kRocProfVisDmResultSuccess == rocprofvis_dm_get_property_as_uint64(handle,property,index,&value)) return value;
    return 0;
}                                     

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
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index){
    int64_t value=0;
    if (kRocProfVisDmResultSuccess == rocprofvis_dm_get_property_as_int64(handle,property,index,&value)) return value;
    return 0;
}

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
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index){
    double value=0;
    if (kRocProfVisDmResultSuccess == rocprofvis_dm_get_property_as_double(handle,property,index,&value)) return value;
    return 0.0;
} 

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
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index){
    char* value=nullptr;
    if (kRocProfVisDmResultSuccess == rocprofvis_dm_get_property_as_charptr(handle,property,index,&value)) return value;
    return nullptr;
} 

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
                                        rocprofvis_dm_handle_t handle, 	
                                        rocprofvis_dm_property_t property,
                                        rocprofvis_dm_property_index_t index){
    rocprofvis_dm_handle_t value=nullptr;
    if (kRocProfVisDmResultSuccess == rocprofvis_dm_get_property_as_handle(handle,property,index,&value)) return value;
    return nullptr;
} 
