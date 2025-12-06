// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_db.h"

namespace RocProfVis
{
namespace DataModel
{

class SqliteDatabase;

typedef struct rocprofvis_db_sqlite_track_service_data_t
{
    rocprofvis_dm_event_operation_t op;
    rocprofvis_dm_track_category_t  category;
    uint64_t                        id;
    uint64_t                        nid;
    uint32_t                        stream_id;
    uint32_t                        process;
    uint32_t                        thread;
    std::string                     monitor_type;
} rocprofvis_db_sqlite_track_service_data_t;

typedef struct rocprofvis_db_sqlite_track_query_format
{
    static constexpr const int NUM_PARAMS = 4; 
    std::string                 parameters[NUM_PARAMS];
    std::vector<std::string>    from;
} rocprofvis_db_sqlite_track_query_format;

typedef struct rocprofvis_db_sqlite_region_track_query_format
{
    static constexpr const int NUM_PARAMS = 4;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
    std::vector<std::string>   where;
} rocprofvis_db_sqlite_region_track_query_format;

typedef struct rocprofvis_db_sqlite_level_query_format
{
    static constexpr const int NUM_PARAMS = 8;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
} rocprofvis_db_sqlite_level_query_format;

typedef struct rocprofvis_db_sqlite_slice_query_format
{
    static constexpr const int NUM_PARAMS = 10;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
} rocprofvis_db_sqlite_slice_query_format;

typedef struct rocprofvis_db_sqlite_table_query_format
{
    static constexpr const int NUM_PARAMS = 35;
    SqliteDatabase*            owner;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
} rocprofvis_db_sqlite_table_query_format;

typedef struct rocprofvis_db_sqlite_rocpd_table_query_format
{
    static constexpr const int NUM_PARAMS = 9;
    SqliteDatabase*            owner;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
} rocprofvis_db_sqlite_rocpd_table_query_format;

typedef struct rocprofvis_db_sqlite_sample_table_query_format
{
    static constexpr const int NUM_PARAMS = 7;
    SqliteDatabase*            owner;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
} rocprofvis_db_sqlite_sample_table_query_format;

typedef struct rocprofvis_db_sqlite_dataflow_query_format
{
    static constexpr const int NUM_PARAMS = 11;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
    std::vector<std::string>   where;
} rocprofvis_db_sqlite_dataflow_query_format;

typedef struct rocprofvis_db_sqlite_essential_data_query_format
{
    static constexpr const int NUM_PARAMS = 7;
    std::string                parameters[NUM_PARAMS];
    std::vector<std::string>   from;
    std::vector<std::string>   where;
} rocprofvis_db_sqlite_essential_data_query_format;

typedef enum rocprofvis_db_sqlite_table_query_column_mask
{
    kRPVTableQueryColumnMaskBlank = 0,
    kRPVTableQueryColumnMaskService,
    kRPVTableQueryColumnMaskTimestamp,
    kRPVTableQueryColumnMaskCount,
} rocprofvis_db_sqlite_table_query_column_mask;

class Builder
{
    public:
        static constexpr const char* AGENT_ID_SERVICE_NAME = "agentId";
        static constexpr const char* QUEUE_ID_SERVICE_NAME = "queueId";
        static constexpr const char* STREAM_ID_SERVICE_NAME = "streamId";
        static constexpr const char* NODE_ID_SERVICE_NAME = "nodeId";
        static constexpr const char* NODE_ID_PUBLIC_NAME = "nodeId";
        static constexpr const char* OPERATION_SERVICE_NAME = "op";
        static constexpr const char* CATEGORY_PUBLIC_NAME = "category";
        static constexpr const char* PROCESS_ID_PUBLIC_NAME = "PID";
        static constexpr const char* THREAD_ID_PUBLIC_NAME = "TID";
        static constexpr const char* PROCESS_ID_SERVICE_NAME = "processId";
        static constexpr const char* THREAD_ID_SERVICE_NAME  = "threadId";
        static constexpr const char* COUNTER_ID_SERVICE_NAME  = "counterId";
        static constexpr const char* COUNTER_VALUE_SERVICE_NAME = "counterValue";
        static constexpr const char* TRACK_ID_PUBLIC_NAME = "__trackId";
        static constexpr const char* STREAM_TRACK_ID_PUBLIC_NAME = "__streamTrackId";
        static constexpr const char* SPACESAVER_SERVICE_NAME     = "const";
        static constexpr const char* COUNTER_NAME_SERVICE_NAME   = "monitorType";
        static constexpr const char* BLANK_COLUMN_STR            = "''";
        static constexpr const char* START_SERVICE_NAME     = "startTs";
        static constexpr const char* END_SERVICE_NAME       = "endTs";

        static constexpr const int  LEVEL_CALCULATION_VERSION  = 3;

    public:
        static std::string Select(rocprofvis_db_sqlite_track_query_format params);
        static std::string Select(rocprofvis_db_sqlite_region_track_query_format params);
        static std::string Select(rocprofvis_db_sqlite_level_query_format params);
        static std::string Select(rocprofvis_db_sqlite_slice_query_format params);
        static std::string Select(rocprofvis_db_sqlite_table_query_format params);
        static std::string Select(rocprofvis_db_sqlite_sample_table_query_format params);
        static std::string Select(rocprofvis_db_sqlite_rocpd_table_query_format params);
        static std::string Select(rocprofvis_db_sqlite_dataflow_query_format params);
        static std::string Select(rocprofvis_db_sqlite_essential_data_query_format params);
        static std::string SelectAll(std::string query);
        static std::string QParam(std::string name, std::string public_name);
        static std::string Blank();
        static std::string QParamBlank(std::string public_name);
        static std::string QParam(std::string name);
        static std::string QParamOperation(const rocprofvis_dm_event_operation_t op);
        static std::string QParamCategory(const rocprofvis_dm_track_category_t category);
        static std::string From(std::string table);
        static std::string From(std::string table, std::string nick_name);
        static std::string InnerJoin(std::string table, std::string nick_name, std::string on);
        static std::string LeftJoin(std::string table, std::string nick_name, std::string on);
        static std::string RightJoin(std::string table, std::string nick_name, std::string on);
        static std::string SpaceSaver(int val);
        static std::string THeader(std::string header);
        static std::string TVar(std::string tag, std::string var);
        static std::string TVar(std::string tag, std::string var1, std::string var2);
        static std::string Concat(std::vector<std::string> strings);
        static std::string Where(std::string name, std::string condition, std::string value);
        static std::string Union();
        static std::string LevelTable(std::string operation);
        static std::vector<std::string> OldLevelTables(std::string operation);
    private:
        static void        BuildColumnMasks(SqliteDatabase* owner, int num_params, std::string* params); 
        static std::string BuildQuery(std::string select, int num_params,
                                      std::string* params, std::vector<std::string> from,
                                      std::string finalize_with);
        static std::string BuildQuery(std::string select, int num_params,
                                      std::string* params, std::vector<std::string> from,
                                      std::vector<std::string> where,
                                      std::string              finalize_with);
       
};

}  // namespace DataModel
}  // namespace RocProfVis