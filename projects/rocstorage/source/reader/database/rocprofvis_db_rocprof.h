// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_db_profile.h"
#include "rocprofvis_db_query_factory.h"

namespace RocProfVis
{
namespace DataModel
{

class RocprofDatabase : public ProfileDatabase
{
    // map array for fast track ID search
    typedef std::map<uint64_t, uint32_t> sub_process_map_t;
    typedef std::map<uint64_t, sub_process_map_t> process_map_t;
    typedef std::map<uint64_t, process_map_t> op_map_t;
    typedef std::map<uint32_t, op_map_t> track_find_map_t;

public:
    RocprofDatabase(rocprofvis_db_filename_t path) :
        ProfileDatabase(path),
        m_query_factory(this)
    {
    };
    // class destructor, not really required, unless declared as virtual
    ~RocprofDatabase()override{};
    // worker method to read trace metadata
    // @param object - future object providing asynchronous execution mechanism 
    // @return status of operation
    rocprofvis_dm_result_t  ReadTraceMetadata(
                                        Future* object) override;

    // worker method to read flow trace info
    // @param event_id - 60-bit event id and 4-bit operation type  
    // @param object - future object providing asynchronous execution mechanism 
    // @return status of operation
    rocprofvis_dm_result_t  ReadFlowTraceInfo(
                                        rocprofvis_dm_event_id_t event_id,
                                        Future* object) override;
    // worker method to read stack trace info
    // @param event_id - 60-bit event id and 4-bit operation type  
    // @param object - future object providing asynchronous execution mechanism 
    // @return status of operation
    rocprofvis_dm_result_t  ReadStackTraceInfo(
                                        rocprofvis_dm_event_id_t event_id,
                                        Future* object) override;
    // worker method to read extended info
    // @param event_id - 60-bit event id and 4-bit operation type  
    // @param object - future object providing asynchronous execution mechanism 
    // @return status of operation
    rocprofvis_dm_result_t  ReadExtEventInfo(
                                        rocprofvis_dm_event_id_t event_id, 
                                        Future* object) override;

    rocprofvis_dm_result_t SaveTrimmedData(rocprofvis_dm_timestamp_t start, 
                                           rocprofvis_dm_timestamp_t end,
                                           rocprofvis_dm_charptr_t new_db_path,
                                           Future* future) override;

    rocprofvis_dm_result_t BuildTableStringIdFilter( 
                                        rocprofvis_dm_num_string_table_filters_t num_string_table_filters, 
                                        rocprofvis_dm_string_table_filters_t string_table_filters,
                                        table_string_id_filter_map_t& filters) override;

    rocprofvis_dm_string_t GetEventOperationQuery(
                                        const rocprofvis_dm_event_operation_t operation) override;

    
    rocprofvis_dm_result_t StringIndexToId(
                                        rocprofvis_dm_index_t index, rocprofvis_dm_id_t& id) override;

    rocprofvis_dm_result_t BuildTableSummaryClause(
                                        bool sample_query,
                                        rocprofvis_dm_string_t& select,
                                        rocprofvis_dm_string_t& group_by) override;

private:
    // sqlite3_exec callback to process track information query and add track object to Trace container
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
    static int CallBackAddTrack(void* data, int argc, sqlite3_stmt* stmt, char** azColName);
    // sqlite3_exec callback to process string list query and add string object to Trace container
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
    static int CallBackAddString(void *data, int argc, sqlite3_stmt* stmt, char **azColName);
    // sqlite3_exec callback to cache specified tables data
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
    static int CallbackCacheTable(void *data, int argc, sqlite3_stmt* stmt, char **azColName);
    // sqlite3_exec callback to process stack trace information query and add stack trace
    // object to StackTrace container
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
    static int CallbackAddStackTrace(void* data, int argc, sqlite3_stmt* stmt,
                                     char** azColName);
    // sqlite3_exec callback to detect nodes and table names in the database
    // object to StackTrace container
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
    static int CallbackNodeEnumeration(void* data, int argc, sqlite3_stmt* stmt,
                                       char** azColName);
    // sqlite3_exec callback to parse metadata table of new schema rocprof database
    // object to StackTrace container
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
    static int CallbackParseMetadata(void* data, int argc, sqlite3_stmt* stmt,
        char** azColName);
    
    // method to remap string IDs. Main reason for remapping is having strings and kernel symbol names in one array 
    // @param record - event record structure
    // @return status of operation
    rocprofvis_dm_result_t RemapStringIds(rocprofvis_db_record_data_t & record) override;
    rocprofvis_dm_result_t RemapStringIds(rocprofvis_db_flow_data_t& record) override;
    // finds and returns track id by 3 input parameters  (Node, Agent/PID, QueueId/PmcId/Metric name) 
    // @param node_id - node id
    // @param process_id - process id 
    // @param sub_process_name - metric name
    // @return status of operation
    // @param operation - operation of event that requesting track id
    // @return status of operation
    rocprofvis_dm_result_t          FindTrackId(
                                                        uint64_t node,
                                                        uint32_t process,
                                                        const char* subprocess,
                                                        rocprofvis_dm_op_t operation,
                                                        rocprofvis_dm_track_id_t& track_id) override;

    protected:
    const rocprofvis_event_data_category_map_t* GetCategoryEnumMap() override {
        return &s_rocprof_categorized_data;
    };
    const rocprofvis_null_data_exceptions_int* GetNullDataExceptionsInt() override
    {
        return &s_null_data_exceptions_int;
    }
    const rocprofvis_null_data_exceptions_string* GetNullDataExceptionsString() override
    {
        return &s_null_data_exceptions_string;
    }
    const rocprofvis_null_data_exceptions_skip* GetNullDataExceptionsSkip() override
    {
        return &s_null_data_exceptions_skip;
    }
    const rocprofvis_dm_track_category_t GetRegionTrackCategory() override
    {
        return kRocProfVisDmRegionMainTrack;
    }


    private:
        rocprofvis_dm_result_t CreateIndexes();
        
    private:
        track_find_map_t find_track_map;
        QueryFactory m_query_factory;
        std::string db_version;

        inline static const rocprofvis_event_data_category_map_t
            s_rocprof_categorized_data = {
                {
                    kRocProfVisDmOperationNoOp,
                    {
                        { "id", kRocProfVisEventEssentialDataId },
                        { "category", kRocProfVisEventEssentialDataCategory },
                        { "name", kRocProfVisEventEssentialDataName },
                        { "start", kRocProfVisEventEssentialDataStart },
                        { "end", kRocProfVisEventEssentialDataEnd },
                        { "duration", kRocProfVisEventEssentialDataDuration },
                        { "nid", kRocProfVisEventEssentialDataNode },
                        { "pid", kRocProfVisEventEssentialDataProcess },
                        { "tid", kRocProfVisEventEssentialDataThread },
                        { "queue_name", kRocProfVisEventEssentialDataQueue },
                        { "stream_name", kRocProfVisEventEssentialDataStream },
                        { "stack_id", kRocProfVisEventEssentialDataInternal },
                        { "parent_stack_id", kRocProfVisEventEssentialDataInternal },
                        { "corr_id", kRocProfVisEventEssentialDataInternal },
                        { "stream_id", kRocProfVisEventEssentialDataInternal },
                        { "queue_id", kRocProfVisEventEssentialDataInternal },
                    },
                },
                {
                    kRocProfVisDmOperationDispatch,
                    {
                        { "agent_type", kRocProfVisEventEssentialDataAgentType },
                        { "agent_type_index", kRocProfVisEventEssentialDataAgentIndex },
                    },
                },
                {
                    kRocProfVisDmOperationMemoryAllocate,
                    {
                        { "agent_type", kRocProfVisEventEssentialDataAgentType },
                        { "agent_type_index", kRocProfVisEventEssentialDataAgentIndex },
                        { "type", kRocProfVisEventEssentialDataName },
                    },
                },
                {
                    kRocProfVisDmOperationMemoryCopy,
                    {
                        { "dst_agent_type", kRocProfVisEventEssentialDataAgentType },
                        { "dst_agent_type_index", kRocProfVisEventEssentialDataAgentIndex },
                    } 
                }
            };

        inline static const rocprofvis_null_data_exceptions_skip
            s_null_data_exceptions_skip = { 
                { (void*)&CallBackAddTrack,
                  { 
                        Builder::AGENT_ID_SERVICE_NAME, 
                        Builder::QUEUE_ID_SERVICE_NAME,
                  }
                },
                { (void*)&CallbackGetTrackProperties,
                  { 
                        "MIN(startTs)",
                        "MAX(endTs)"
                  }
                } 
        };

        inline static const rocprofvis_null_data_exceptions_int 
            s_null_data_exceptions_int = {
            { 

            }
        };
        inline static const rocprofvis_null_data_exceptions_string
            s_null_data_exceptions_string = { 
            { 
                (void*)&CallbackCacheTable, 
                { 
                    { "name", "N/A" }, 
                    { "start", "0" }, 
                    { "end", "0" } 
                },
            },
            {
                (void*)&CallbackRunQuery,
                { 
                    { "name", "N/A" },  
                    { "start", "0" }, 
                    { "end", "0" } 
                },
            }

       };
};

}  // namespace DataModel
}  // namespace RocProfVis