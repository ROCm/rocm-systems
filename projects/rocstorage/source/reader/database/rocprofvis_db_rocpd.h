// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_db_profile.h"

#include <unordered_map>

namespace RocProfVis
{
namespace DataModel
{

// class for reading old schema Rocpd database
class RocpdDatabase : public ProfileDatabase
{
    // type of map array for string indexes remapping
    typedef std::unordered_map<uint64_t, uint32_t> string_index_map_t;
    typedef std::unordered_map<rocprofvis_dm_index_t, rocprofvis_dm_id_t> string_id_map_t;

    // map array for fast non-PMC track ID search
    typedef std::map<uint32_t, uint32_t> sub_process_map_t;
    typedef std::map<uint32_t, sub_process_map_t> track_find_map_t;

    // map array for fast PMC track ID search
    typedef std::map<std::string, uint32_t> sub_process_map_pmc_t;
    typedef std::map<uint32_t, sub_process_map_pmc_t> track_find_pmc_map_t;
    
public:
    // class constructor
    // @param path - database file path
    RocpdDatabase(rocprofvis_db_filename_t path) :
        ProfileDatabase(path) {}
    // class destructor, not really required, unless declared as virtual
    ~RocpdDatabase()override{};
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
    
    // get class memory usage
    // @return size of memory used by the class
    rocprofvis_dm_size_t GetMemoryFootprint(void) override;

    rocprofvis_dm_result_t SaveTrimmedData(rocprofvis_dm_timestamp_t start,
                                           rocprofvis_dm_timestamp_t end,
                                           rocprofvis_dm_charptr_t new_db_path,
                                           Future* future) override;

    rocprofvis_dm_result_t BuildTableStringIdFilter(
                                    rocprofvis_dm_num_string_table_filters_t num_string_table_filters, 
                                    rocprofvis_dm_string_table_filters_t string_table_filters,
                                    table_string_id_filter_map_t& filter) override;

    rocprofvis_dm_string_t GetEventOperationQuery(
                                        const rocprofvis_dm_event_operation_t operation) override;

    rocprofvis_dm_result_t StringIndexToId(
                                        rocprofvis_dm_index_t index, rocprofvis_dm_id_t& id) override;

    rocprofvis_dm_result_t BuildTableSummaryClause(bool sample_query,
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
    // sqlite3_exec callback to process stack trace information query and add stack trace object to StackTrace container
    // @param data - pointer to callback caller argument
    // @param argc - number of columns in the query
    // @param argv - pointer to row values
    // @param azColName - pointer to column names
    // @return SQLITE_OK if successful
    static int CallbackAddStackTrace(void *data, int argc, sqlite3_stmt* stmt, char **azColName);

    // map array for string indexes remapping. Main reason for remapping is older rocpd schema keeps duplicated symbols, one per GPU 
    string_index_map_t m_string_index_map; // id to index
    string_id_map_t m_string_id_map; // index to id

    // method to remap string IDs. Main reason for remapping is older rocpd schema keeps duplicated symbols, one per GPU 
    // @param record - event record structure
    // @return status of operation
    rocprofvis_dm_result_t RemapStringIds(rocprofvis_db_record_data_t & record) override;
    rocprofvis_dm_result_t RemapStringIds(rocprofvis_db_flow_data_t& record) override;

    // finds and returns track id by 3 input parameters  (Node, Agent/PID, QueueId/PmcId/Metric name) 
    // @param node_id - node id
    // @param process_id - process id 
    // @param sub_process_name - metric name
    // @param operation - operation of event that requesting track id
    // @return status of operation
    rocprofvis_dm_result_t          FindTrackId(
                                                        uint64_t node,
                                                        uint32_t process,
                                                        const char* subprocess,
                                                        rocprofvis_dm_op_t operation,
                                                        rocprofvis_dm_track_id_t& track_id) override;

    // method to remap single string ID. Main reason for remapping is older rocpd schema keeps duplicated symbols, one per GPU 
    // @param id - string id to be remapped
    // @return True if remapped
    const bool RemapStringId(uint64_t & id);

    const rocprofvis_event_data_category_map_t* GetCategoryEnumMap() override
    {
        return &s_rocpd_categorized_data;
    };
    const rocprofvis_null_data_exceptions_int* GetNullDataExceptionsInt() override
    {
        return &s_null_data_exception_int;
    }
    const rocprofvis_null_data_exceptions_string* GetNullDataExceptionsString() override
    {
        return &s_null_data_exception_string;
    }
    const rocprofvis_null_data_exceptions_skip* GetNullDataExceptionsSkip() override
    {
        return &s_null_data_exceptions_skip;
    }
    const rocprofvis_dm_track_category_t GetRegionTrackCategory() override 
    {
        return kRocProfVisDmRegionTrack;
    }
    private:
        rocprofvis_dm_result_t CreateIndexes();

    private:
        track_find_map_t find_track_map;
        track_find_pmc_map_t find_track_pmc_map;

        inline static const rocprofvis_event_data_category_map_t
            s_rocpd_categorized_data = {
                {
                    kRocProfVisDmOperationNoOp,
                    {
                        { "id", kRocProfVisEventEssentialDataId },
                        { "start", kRocProfVisEventEssentialDataStart },
                        { "end", kRocProfVisEventEssentialDataEnd },
                        { "duration", kRocProfVisEventEssentialDataDuration },
                    },
                },
                {
                    kRocProfVisDmOperationLaunch,
                    {
                        { "apiName", kRocProfVisEventEssentialDataCategory },
                        { "args", kRocProfVisEventEssentialDataName },
                        { "pid", kRocProfVisEventEssentialDataProcess },
                        { "tid", kRocProfVisEventEssentialDataThread },
                    },
                },
                {
                    kRocProfVisDmOperationDispatch,
                    {
                        { "opType", kRocProfVisEventEssentialDataCategory },
                        { "description", kRocProfVisEventEssentialDataName },
                        { "agent_type", kRocProfVisEventEssentialDataAgentType },
                        { "gpuId", kRocProfVisEventEssentialDataAgentIndex },
                        { "queueId", kRocProfVisEventEssentialDataQueue },
                        { "stream", kRocProfVisEventEssentialDataStream },
                    }
                }
            };

         inline static const rocprofvis_null_data_exceptions_skip
            s_null_data_exceptions_skip = {

            };

        inline static const rocprofvis_null_data_exceptions_int
            s_null_data_exception_int = {

            };
        inline static const rocprofvis_null_data_exceptions_string
            s_null_data_exception_string = {

            };

};

}  // namespace DataModel
}  // namespace RocProfVis