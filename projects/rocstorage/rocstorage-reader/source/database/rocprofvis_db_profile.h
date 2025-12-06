// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_db_sqlite.h"

namespace RocProfVis
{
namespace DataModel
{

#define SINGLE_THREAD_RECORDS_COUNT_LIMIT 50000

typedef enum rocprofvis_db_async_tracks_flags_t
{
    kRocProfVisDmIncludePmcTracks = 1,
    kRocProfVisDmIncludeStreamTracks = 2,
    kRocProfVisDmTrySplitTrack = 4
} rocprofvis_db_async_tracks_flags_t;

typedef struct rocprofvis_db_event_level_t
{
    uint64_t id;
    uint8_t  level_for_queue;
    uint8_t  level_for_stream;
} rocprofvis_db_event_level_t;

typedef struct rocprofvis_db_sqlite_trim_parameters
{
    // Table names as we can't issue recursively
    std::map<std::string, std::string> tables;
} rocprofvis_db_sqlite_trim_parameters;

typedef std::map<uint64_t, std::map<std::string, rocprofvis_event_data_category_enum_t>>
    rocprofvis_event_data_category_map_t;

// class for methods and members common for all RocPd-based schemas
class ProfileDatabase : public SqliteDatabase
{
    
    public:
        // Database constructor
        // @param path - full path to database file
        ProfileDatabase( rocprofvis_db_filename_t path) : 
                        SqliteDatabase(path), 
                        m_symbols_offset(0){};
        // ProfileDatabase destructor, must be defined as virtual to free resources of derived classes 
        virtual ~ProfileDatabase() {}
        // worker method to read time slice
        // @param start - start timestamp of time slice 
        // @param end - end timestamp of time slice 
        // @param num - number of tracks
        // @param tracks - uint32_t array with track IDs  
        // @param object - future object providing asynchronous execution mechanism   
        // @return status of operation        
        rocprofvis_dm_result_t  ReadTraceSlice(
                        rocprofvis_dm_timestamp_t start,
                        rocprofvis_dm_timestamp_t end,
                        rocprofvis_db_num_of_tracks_t num,
                        rocprofvis_db_track_selection_t tracks,
                        Future* object) override;
        // worker method to execute database query
        // @param query - database query 
        // @param description - database description
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        rocprofvis_dm_result_t  ExecuteQuery(
                        rocprofvis_dm_charptr_t query,
                        rocprofvis_dm_charptr_t description,
                        Future* future) override; 
        // method to detect rocpd-based database type (rocpd vs rocprof)
        // @param filename - full path to database file
        static rocprofvis_db_type_t Detect(rocprofvis_db_filename_t filename);
        // method to execute table database query with appropriate .CSV writer callback based on existence of GROUP BY clause
        // @param query - database query 
        // @param file_path output path to write .CSV
        // @param future - future object providing asynchronous execution mechanism 
        // @return status of operation
        rocprofvis_dm_result_t ExportTableCSV(
                                    rocprofvis_dm_charptr_t query,
                                    rocprofvis_dm_charptr_t file_path,
                                    Future* future) override;

    private:

    // sqlite3_exec callback to add any record (Event or PMC) to time slice container. Used in all-selected-tracks time slice query
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names  
    // @return SQLITE_OK if successful
        static int CallbackAddAnyRecord(void* data, int argc, sqlite3_stmt* stmt, char** azColName);

        void BuildSliceQueryMap(slice_query_t& slice_query_map, rocprofvis_dm_track_params_t* props);

    protected:

    // method to build a query to read time slice of records for single track 
    // @param index - track index 
    // @param tyte - query type
    // @param query - reference to output query string  
    // @return status of operation
        rocprofvis_dm_result_t BuildTrackQuery(
                            rocprofvis_dm_index_t index,
                            rocprofvis_dm_index_t   type,
                            rocprofvis_dm_string_t& query,
                            uint32_t split_count,
                            uint32_t split_index) override;
    // method to build a query to read time slice of records for all tracks in one shot 
    // @param start - start timestamp of time slice 
    // @param end - end timestamp of time slice 
    // @param num - number of tracks
    // @param tracks - uint32_t array with track IDs 
    // @param query - reference to query string 
    // @param slices - reference map array for storing slice handlers for multi-track request   
    // @return status of operation  
        rocprofvis_dm_result_t BuildSliceQuery(
                            rocprofvis_dm_timestamp_t start,
                            rocprofvis_dm_timestamp_t end,
                            rocprofvis_db_num_of_tracks_t num,
                            rocprofvis_db_track_selection_t tracks,
                            rocprofvis_dm_string_t& query,
                            slice_array_t& slices) override;

        rocprofvis_dm_result_t BuildCounterSliceLeftNeighbourQuery(
            rocprofvis_dm_timestamp_t start, 
            rocprofvis_dm_timestamp_t end, 
            rocprofvis_dm_index_t track_index, 
            rocprofvis_dm_string_t& query);
        rocprofvis_dm_result_t BuildCounterSliceRightNeighbourQuery(
            rocprofvis_dm_timestamp_t start, 
            rocprofvis_dm_timestamp_t end, 
            rocprofvis_dm_index_t track_index, 
            rocprofvis_dm_string_t& query);

        rocprofvis_dm_result_t BuildTableQuery(
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
                            rocprofvis_dm_string_t& query) override;

        rocprofvis_dm_result_t ExecuteQueryForAllTracksAsync(
                            uint32_t flags, 
                            rocprofvis_dm_index_t query_type,
                            rocprofvis_dm_charptr_t prefix, 
                            rocprofvis_dm_charptr_t suffix,
                            RpvSqliteExecuteQueryCallback callback, 
                            std::function<void(rocprofvis_dm_track_params_t*)> func_clear);

    protected:
    // sqlite3_exec callback to add flowtrace record to FlowTrace container.
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names  
    // @return SQLITE_OK if successful
        static int CallbackAddFlowTrace(void *data, int argc, sqlite3_stmt* stmt, char **azColName);
    // sqlite3_exec callback to add extended info record ExtData container.
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names 
    // @return SQLITE_OK if successful
        static int CallbackAddExtInfo(void* data, int argc, sqlite3_stmt* stmt, char** azColName);
    // sqlite3_exec callback to add essential info into ExtData container.
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names 
    // @return SQLITE_OK if successful
        static int CallbackAddEssentialInfo(void* data, int argc, sqlite3_stmt* stmt, char** azColName);

    // sqlite3_exec callback to calculate graph level for an event and store it into trace object map array
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names  
    // @return SQLITE_OK if successful
       static int CalculateEventLevels(void* data, int argc, sqlite3_stmt* stmt, char** azColName);
    // sqlite3_exec callback to collect minimum/maximum timestamps and minimu/maximum value/level
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
        static int CallbackGetTrackProperties(void* data, int argc, sqlite3_stmt* stmt,
                                              char** azColName);
       // sqlite3_exec callback to collect number of records in te track
       // @param data - pointer to callback caller argument
       // @param argc - number of columns in the query
       // @param argv - pointer to row values
       // @param azColName - pointer to column names
       // @return SQLITE_OK if successful
        static int CallbackGetTrackRecordsCount(void* data, int argc, sqlite3_stmt* stmt,
                                                char** azColName);
        static int CallbackTrimTableQuery(void* data, int argc, sqlite3_stmt* stmt,
                                          char** azColName);
        static int CallbackMakeHistogramPerTrack(void* data, int argc, sqlite3_stmt* stmt,
            char** azColName);
        static rocprofvis_event_data_category_enum_t GetColumnDataCategory(
            const rocprofvis_event_data_category_map_t category_map,
            rocprofvis_dm_event_operation_t op, std::string column);

    virtual const rocprofvis_event_data_category_map_t* GetCategoryEnumMap() = 0;

    protected:
    // offset of kernel symbols in string table
        uint32_t m_symbols_offset;
        std::vector<rocprofvis_db_event_level_t> m_event_levels[kRocProfVisDmNumOperation];
        std::unordered_map<uint64_t, size_t> m_event_levels_id_to_index[kRocProfVisDmNumOperation];
        std::mutex   m_level_lock;
};

}  // namespace DataModel
}  // namespace RocProfVis