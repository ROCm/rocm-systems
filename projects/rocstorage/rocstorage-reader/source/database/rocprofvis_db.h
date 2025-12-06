// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_db_future.h"
#include <vector>
#include <map>
#include <unordered_map>

namespace RocProfVis
{
namespace DataModel
{

//types for creating multi-dimensional map array to cache non-essential track information
typedef std::map<std::string, std::string> table_dict_t;
typedef std::map<uint64_t, table_dict_t> table_map_t;
typedef std::map<std::string, table_map_t> ref_map_t;

typedef std::vector<std::unique_ptr<rocprofvis_dm_track_params_t>>::iterator rocprofvis_dm_track_params_it;

// type of map array for generating time slice query for multiple tracks
typedef std::map<std::string, std::string> slice_query_t;
// type of map array for storing slice handlers for multi-track request
typedef std::map<uint32_t, rocprofvis_dm_slice_t> slice_array_t;
// type of map array for storing string id filters for op table queries 
typedef std::unordered_map<rocprofvis_dm_event_operation_t, std::string> table_string_id_filter_map_t;

class Database;



// Helper class to manage cached information tables (node, agent, queue, process, thread information)
class DatabaseCache 
{
    public:
        // Add table cell parameters to a map 
        void AddTableCell(const char* table_name, uint64_t instance_id, const char* column_name, const char* cell_value) {
                                                                                                references[table_name][instance_id][column_name] = (cell_value == nullptr?"":cell_value); 
        };
        // Get table value by table name, instance id and column name
        const char* GetTableCell(const char* table_name, uint64_t instance_id, const char* column_name) {
                                                                                                return references[table_name][instance_id][column_name].c_str();
        };
        // Populate track extended data objects with table content
        rocprofvis_dm_result_t PopulateTrackExtendedDataTemplate(Database * db, const char* table_name, uint64_t instance_id ); 
        // Get amount of memory used by the cached values map
        rocprofvis_dm_size_t    GetMemoryFootprint(void); 

    private:
        
        // map of cached table values
        ref_map_t  references;
};

class Database
{
    public:
        // Database constructor
        // @param path - full path to database file
        Database(   
                    rocprofvis_db_filename_t path):
                    m_path(path),
                    m_binding_info(nullptr) {
        };
        // Database destructor, must be defined as virtual to free resources of derived classes 
        virtual ~Database(){};

        // Method to open database, must be overriden by derived classes
        // @return status of operation
        virtual rocprofvis_dm_result_t  Open() = 0;
        // Method to close database, must be overriden by derived classes
        // @return status of operation
        virtual rocprofvis_dm_result_t  Close() = 0;
        // Get amount of memory used by database resource
        // @return memory size
        virtual rocprofvis_dm_size_t    GetMemoryFootprint(void); 

        // Autodetect database type by launching specific database queries
        // @param filename - database filename
        // @return database type 
        static rocprofvis_db_type_t     Autodetect(   
                                                                rocprofvis_db_filename_t filename);
        // Bind database to trace
        // @param binding_info - pointer to binding info structure 
        // @return status of operation
        rocprofvis_dm_result_t          BindTrace(
                                                                rocprofvis_dm_db_bind_struct * binding_info);
        // Asynchronously read trace metadata from database
        // @param object - future object providing asynchronous execution mechanism
        // @return status of operation
        rocprofvis_dm_result_t          ReadTraceMetadataAsync( 
                                                                rocprofvis_db_future_t object);
        // Asynchronously read a time slice (records from specified number of tracks for specified time frame) from database  
        // @param start - start timestamp of time slice 
        // @param end - end timestamp of time slice 
        // @param num - number of tracks
        // @param tracks - uint32_t array with track IDs  
        // @param object - future object providing asynchronous execution mechanism         
        // @return status of operation                                            
        rocprofvis_dm_result_t          ReadTraceSliceAsync( 
                                                                rocprofvis_dm_timestamp_t start,
                                                                rocprofvis_dm_timestamp_t end,
                                                                rocprofvis_db_num_of_tracks_t num,
                                                                rocprofvis_db_track_selection_t tracks,
                                                                rocprofvis_db_future_t object);
        // Asynchronously read different types of event properties (flowtrace, stacktrace, extdata) for event ID
        // @param type - event property type (flowtrace, stacktrace, extdata) 
        // @param event_id - 60-bit event id and 4-bit operation type  
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        rocprofvis_dm_result_t          ReadEventPropertyAsync(
                                                                rocprofvis_dm_event_property_type_t type,
                                                                rocprofvis_dm_event_id_t event_id,
                                                                rocprofvis_db_future_t object);
        // Asynchronously run any table query and store results into Table object 
        // @param query - database query 
        // @param description - database description
        // @param object - future object providing asynchronous execution mechanism 
        // @param id new id is assigned to the table and returned using this reference pointer
        // @return status of operation
        rocprofvis_dm_result_t          ExecuteQueryAsync(
                                                                rocprofvis_dm_charptr_t query,
                                                                rocprofvis_dm_charptr_t description,
                                                                rocprofvis_db_future_t object,
                                                                rocprofvis_dm_table_id_t* id);

       virtual rocprofvis_dm_result_t BuildTableQuery(
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
                                                                rocprofvis_dm_string_t& query) = 0;

        // Searches for strings containing the passed in list of filter strings and builds a WHERE IN clause for the table query.
        // @param num_string_table_filters - number of filter strings
        // @param string_table_filters - array of filter strings
        // @param filter - output string containing WHERE clause
        // @return status of operation
       virtual rocprofvis_dm_result_t BuildTableStringIdFilter(
                                                                rocprofvis_dm_num_string_table_filters_t num_string_table_filters, 
                                                                rocprofvis_dm_string_table_filters_t string_table_filters,
                                                                table_string_id_filter_map_t& filter) = 0;

       virtual rocprofvis_dm_result_t BuildTableSummaryClause(
                                                                bool sample_query,
                                                                rocprofvis_dm_string_t& select,
                                                                rocprofvis_dm_string_t& group_by) = 0;

        // Asynchronously writes the results of a table query to .CSV
        // @param query - database query 
        // @param file_path - .CSV output path
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
       rocprofvis_dm_result_t ExportTableCSVAsync(rocprofvis_dm_string_t query,
                                                  rocprofvis_dm_string_t file_path,
                                                  rocprofvis_db_future_t object);

       virtual rocprofvis_dm_result_t SaveTrimmedData(rocprofvis_dm_timestamp_t start,
                                                      rocprofvis_dm_timestamp_t end,
                                                      rocprofvis_dm_charptr_t new_db_path,
                                                      Future* future) = 0;

       rocprofvis_dm_result_t SaveTrimmedDataAsync(rocprofvis_dm_timestamp_t start,
                                                   rocprofvis_dm_timestamp_t end,
                                                   rocprofvis_dm_string_t new_db_path, 
                                                   rocprofvis_db_future_t object);

       static rocprofvis_dm_result_t SaveTrimmedDataStatic(Database* db, 
                                                    rocprofvis_dm_timestamp_t start,
                                                    rocprofvis_dm_timestamp_t end, 
                                                    rocprofvis_dm_string_t new_db_path,
                                                    Future* object);
       virtual void InterruptQuery(void* connection) {};

    private:
    /************************static methods to be used as a parameter to std::thread**********************/

        //static method to read metadata. Required to launch a unique thread for asynchronous metadata read 
        // @param db - pointer to database object 
        // @param object - future object providing asynchronous execution mechanism   
        static rocprofvis_dm_result_t   ReadTraceMetadataStatic(
                                                                Database* db, 
                                                                Future* object);
        //static method to read time slice. Required to launch a unique thread for asynchronous time slice read
        // @param db - pointer to database object 
        // @param start - start timestamp of time slice 
        // @param end - end timestamp of time slice 
        // @param num - number of tracks
        // @param tracks - uint32_t array with track IDs  
        // @param object - future object providing asynchronous execution mechanism   
        // @return status of operation
        static rocprofvis_dm_result_t   ReadTraceSliceStatic(
                                                                Database* db,
                                                                rocprofvis_dm_timestamp_t start,
                                                                rocprofvis_dm_timestamp_t end,
                                                                rocprofvis_db_num_of_tracks_t num,
                                                                rocprofvis_db_track_selection_t tracks,
                                                                Future* object);
        //static method to read Event properties. Required to launch a unique thread for asynchronous event properties read
        // @param db - pointer to database object
        // @param type - event property type (flowtrace, stacktrace, extdata) 
        // @param event_id - 60-bit event id and 4-bit operation type  
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        static rocprofvis_dm_result_t   ReadEventPropertyStatic(
                                                                Database* db, 
                                                                rocprofvis_dm_event_property_type_t type,
                                                                rocprofvis_dm_event_id_t event_id,
                                                                Future* object);
        //static method to launch any query. Required to launch a unique thread for asynchronous database query
        // @param db - pointer to database object
        // @param query - database query 
        // @param description - database description
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        static rocprofvis_dm_result_t   ExecuteQueryStatic(
                                                                Database* db,
                                                                rocprofvis_dm_charptr_t query,
                                                                rocprofvis_dm_charptr_t description,
                                                                Future* object);
        // static method to find a value in cached tables by specifying reserved table name, instance id and column name
        // @param object - database handler
        // @param table_name - a name of cached table assigned at the time of caching
        // @param instance_id - cached table instance id
        // @param column_name - cached table column name 
        // @param value - reference pointer to database cell value
        // @return status of operation 
        static rocprofvis_dm_result_t   FindCachedTableValue(  const rocprofvis_dm_database_t object, 
                                                               rocprofvis_dm_charptr_t table_name, 
                                                               const rocprofvis_dm_id_t instance_id, 
                                                               rocprofvis_dm_charptr_t column_name,
                                                               rocprofvis_dm_charptr_t* value); 

        // static method to export the results of a table query to .CSV
        // @param db - pointer to database object
        // @param query - database query
        // @param file_path - .CSV output path
        // @param future - future object providing asynchronous execution mechanism 
        // @return status of operation 
        static rocprofvis_dm_result_t   ExportTableCSVStatic(  Database* db,
                                                               rocprofvis_dm_string_t query,
                                                               rocprofvis_dm_string_t file_path,
                                                               Future* future);

    /************************pure virtual worker methods to be implemented in derived classes**********************/

        // worker method to read trace metadata 
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        virtual rocprofvis_dm_result_t  ReadTraceMetadata(
                                                                Future* object) = 0;
        // worker method to read time slice
        // @param start - start timestamp of time slice 
        // @param end - end timestamp of time slice 
        // @param num - number of tracks
        // @param tracks - uint32_t array with track IDs  
        // @param object - future object providing asynchronous execution mechanism   
        // @return status of operation
        virtual rocprofvis_dm_result_t  ReadTraceSlice(
                                                                rocprofvis_dm_timestamp_t start,
                                                                rocprofvis_dm_timestamp_t end,
                                                                rocprofvis_db_num_of_tracks_t num,
                                                                rocprofvis_db_track_selection_t tracks,
                                                                Future* object) = 0;
        // worker method to read flow trace info, called from ReadEventPropertyStatic
        // @param event_id - 60-bit event id and 4-bit operation type  
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        virtual rocprofvis_dm_result_t  ReadFlowTraceInfo(
                                                                rocprofvis_dm_event_id_t event_id,
                                                                Future* object) = 0;
        // worker method to read stack trace info, called from ReadEventPropertyStatic
        // @param event_id - 60-bit event id and 4-bit operation type  
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        virtual rocprofvis_dm_result_t  ReadStackTraceInfo(
                                                                rocprofvis_dm_event_id_t event_id,
                                                                Future* object) = 0;
        // worker method to read extended info, called from ReadEventPropertyStatic
        // @param event_id - 60-bit event id and 4-bit operation type  
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        virtual rocprofvis_dm_result_t  ReadExtEventInfo(
                                                                rocprofvis_dm_event_id_t event_id,
                                                                Future* object) = 0;
        // worker method to execute database query
        // @param query - database query 
        // @param description - database description
        // @param object - future object providing asynchronous execution mechanism 
        // @return status of operation
        virtual rocprofvis_dm_result_t  ExecuteQuery(
                                                                rocprofvis_dm_charptr_t query,
                                                                rocprofvis_dm_charptr_t description,
                                                                Future* object) = 0;
        // method to build a query to read time slice of records for single track 
        // @param index - track index 
        // @param type - query type
        // @param query - reference to output query string  
        // @return status of operation
        virtual rocprofvis_dm_result_t  BuildTrackQuery(           
                                                                rocprofvis_dm_index_t index, 
                                                                rocprofvis_dm_index_t type,
                                                                rocprofvis_dm_string_t & query,
                                                                uint32_t split_count,
                                                                uint32_t split_index) = 0;

        // method to build a query to read time slice of records for all tracks in one shot 
        // @param start - start timestamp of time slice 
        // @param end - end timestamp of time slice 
        // @param num - number of tracks
        // @param tracks - uint32_t array with track IDs 
        // @param query - reference to query string 
        // @param slices - reference map array for storing slice handlers for multi-track request   
        // @return status of operation                                                      
        virtual rocprofvis_dm_result_t  BuildSliceQuery(      
                                                                rocprofvis_dm_timestamp_t start, 
                                                                rocprofvis_dm_timestamp_t end, 
                                                                rocprofvis_db_num_of_tracks_t num, 
                                                                rocprofvis_db_track_selection_t tracks, 
                                                                rocprofvis_dm_string_t& query, 
                                                                slice_array_t& slices) = 0;

        // method to export the results of a table query to .CSV
        // @param query - database query
        // @param file_path - .CSV output path
        // @return status of operation 
        virtual rocprofvis_dm_result_t ExportTableCSV(          rocprofvis_dm_charptr_t query,
                                                                rocprofvis_dm_charptr_t file_path,
                                                                Future* future);
    private:
        // pointer to a binding information structure physically located in Trace object and passed to Database object during binding
        // binding structure contains methods to transfer data between database and trace objects 
        rocprofvis_dm_db_bind_struct *m_binding_info;
        // database file path
        std::string m_path;
        // vector array of track parameters. Used as a reference for data model Track objects and for Database component to generate proper database queries 
        std::vector<std::unique_ptr<rocprofvis_dm_track_params_t>> m_track_properties;
        // map array of cached tables, mostly with non-essential Track information
        DatabaseCache m_cached_tables;

    protected:
        // returns pointer to binding structure
        rocprofvis_dm_db_bind_struct *  BindObject() {return m_binding_info;}
        // returns pointer to database file path
        rocprofvis_db_filename_t        Path() {return m_path.c_str();}
        // return current number of tracks
        rocprofvis_dm_size_t            NumTracks() { return m_track_properties.size(); }
        // returns pointer to track properties structure. Takes index of track as a parameter 
        rocprofvis_dm_track_params_t*   TrackPropertiesAt(rocprofvis_dm_index_t index) { return m_track_properties[index].get(); }
        // returns pointer to last registered Track properties structure
        rocprofvis_dm_track_params_t*   TrackPropertiesLast() { return m_track_properties.back().get(); }
        // returns track properties begin iterator
        rocprofvis_dm_track_params_it   TrackPropertiesBegin() { return m_track_properties.begin(); }
        // returns track properties end iterator
        rocprofvis_dm_track_params_it   TrackPropertiesEnd() { return m_track_properties.end(); }
        // returns pointer to trace properties, which contains shared trace information
        rocprofvis_dm_trace_params_t*   TraceProperties() { return m_binding_info->trace_properties; }
        // returns pointer to cached tables map array
        DatabaseCache*                  CachedTables() {return &m_cached_tables;}
        // register new track
        // @param props - track properties structure
        // @return status of operation
        rocprofvis_dm_result_t          AddTrackProperties(
                                                                rocprofvis_dm_track_params_t& props);
        // finds and return iterator to track properties array
        // @process track process identifiers structure
        rocprofvis_dm_track_params_it   FindTrack( rocprofvis_dm_process_identifiers_t& process);
        // adds a new query to the track queries collection 
        // multiple queries for single track are required to support data from multiple database tables on single track,
        // like Kernel Dispatch, Memory Copy and Memory Allocation
        // @param it - track properties array iterator
        // @param newprops - new track properties structure
        // @param newquery - new track records query. One track can have multiple queries.
        void                            UpdateQueryForTrack(rocprofvis_dm_track_params_it it, 
                                                            rocprofvis_dm_track_params_t& newprops,
                                                            rocprofvis_dm_charptr_t*      newqueries);
        // calls Future object callback method, if provided. The callback method is optionally provided by caller in order to display or save current database progress.
        // @param step - approximate percentage of single database operation
        // @param action - database operation description
        // @param status - database operation status 
        // @param future - future object providing callback mechanism
        void                            ShowProgress(
                                                                double step, 
                                                                rocprofvis_dm_charptr_t action, 
                                                                rocprofvis_db_status_t status, 
                                                                Future* future);
        // remap string IDs in new event record structure
        // @param record - event data record
        // @return status of operation
        virtual rocprofvis_dm_result_t  RemapStringIds(
                                                                rocprofvis_db_record_data_t & record) {return kRocProfVisDmResultSuccess;};
        virtual rocprofvis_dm_result_t RemapStringIds(
                                                                rocprofvis_db_flow_data_t& record) {return kRocProfVisDmResultSuccess;};
        virtual rocprofvis_dm_result_t  StringIndexToId(        
                                                                rocprofvis_dm_index_t index, rocprofvis_dm_id_t& id) {return kRocProfVisDmResultSuccess;};

        // return suffix to process name for provided track category ('PID', 'Agent')
        // @param category - track category
        // @return track process name suffix  ('PID', 'Agent')      
        static const char*              ProcessNameSuffixFor(   
                                                                rocprofvis_dm_track_category_t category);
        // return suffix to sub-process name for provided track category ('TID', 'Queue')
        // @param category - track category
        // @return track sub-process name suffix  ('TID', 'Queue')  
        static const char*              SubProcessNameSuffixFor(
                                                                rocprofvis_dm_track_category_t category);
        // check if string is number
        // @param s - string to check
        // @return True if number
        static bool IsNumber(const std::string& s);
        // finds and returns track id by 3 input parameters  (Node, Agent/PID, QueueId/PmcId/Metric name) 
        // @param node_id - node id
        // @param process_id - process id 
        // @param sub_process_name - metric name
        // @param operation - operation of event that requesting track id
        // @return status of operation
        virtual rocprofvis_dm_result_t          FindTrackId(
                                                                uint64_t node,
                                                                uint32_t process,
                                                                const char* subprocess,
                                                                rocprofvis_dm_op_t operation,
                                                                rocprofvis_dm_track_id_t& track_id)=0;

        virtual rocprofvis_dm_string_t GetEventOperationQuery(
                                                                const rocprofvis_dm_event_operation_t operation) = 0;

    public:
        // declare DatabaseCache as friend class, for having access to protected members
        friend class DatabaseCache;
};

}  // namespace DataModel
}  // namespace RocProfVis