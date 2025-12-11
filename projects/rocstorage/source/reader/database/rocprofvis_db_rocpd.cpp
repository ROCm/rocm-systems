// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_db_rocpd.h"
#include "rocprofvis_dm_trace.h"
#include <sstream>
#include <string.h>

namespace RocProfVis
{
namespace DataModel
{

rocprofvis_dm_result_t RocpdDatabase::FindTrackId(
                                                    uint64_t node,
                                                    uint32_t process, 
                                                    const char* subprocess,
                                                    rocprofvis_dm_op_t operation,    
                                                    rocprofvis_dm_track_id_t& track_id) {
    if (operation > 0)
    {
        auto it1 = find_track_map.find(process);
        if (it1!=find_track_map.end()){
            auto it2 = it1->second.find(std::atol(subprocess));
            if (it2!=it1->second.end()){
                track_id = it2->second;
                return kRocProfVisDmResultSuccess;
            }
        }
        return kRocProfVisDmResultNotLoaded;
    } else
    {
        auto it1 = find_track_pmc_map.find(process);
        if (it1!=find_track_pmc_map.end()){
            auto it2 = it1->second.find(subprocess);
            if (it2!=it1->second.end()){
                track_id = it2->second;
                return kRocProfVisDmResultSuccess;
            }
        }
        return kRocProfVisDmResultNotLoaded;
    }
}

rocprofvis_dm_result_t RocpdDatabase::RemapStringIds(rocprofvis_db_record_data_t & record)
{
    if (!RemapStringId(record.event.category)) return kRocProfVisDmResultNotLoaded;
    if (!RemapStringId(record.event.symbol)) return kRocProfVisDmResultNotLoaded;
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RocpdDatabase::RemapStringIds(rocprofvis_db_flow_data_t & record)
{
    if (!RemapStringId(record.category_id)) return kRocProfVisDmResultNotLoaded;
    if (!RemapStringId(record.symbol_id)) return kRocProfVisDmResultNotLoaded;
    return kRocProfVisDmResultSuccess;
}

const bool RocpdDatabase::RemapStringId(uint64_t & id)
{
    string_index_map_t::const_iterator pos = m_string_index_map.find(id);
    if (pos == m_string_index_map.end()) return false;
    id = pos->second;
    return true;
}

rocprofvis_dm_size_t RocpdDatabase::GetMemoryFootprint()
{
    rocprofvis_dm_size_t size = sizeof(RocpdDatabase)+Database::GetMemoryFootprint();
    size+=m_string_index_map.size()* (sizeof(uint64_t) + sizeof(uint32_t));
    size+=m_string_index_map.size() * 3 * sizeof(void*);
    return size;
}


int RocpdDatabase::CallBackAddTrack(void *data, int argc, sqlite3_stmt* stmt, char **azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(argc== NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS + 1, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void* func = (void*)&CallBackAddTrack;
    rocprofvis_dm_track_params_t track_params = {0};
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    RocpdDatabase* db = (RocpdDatabase*)callback_params->db;
    if(callback_params->future->Interrupted()) return SQLITE_ABORT;
    track_params.track_id = (rocprofvis_dm_track_id_t)db->NumTracks();
    track_params.process.category = (rocprofvis_dm_track_category_t)db->Sqlite3ColumnInt(func, stmt, azColName, TRACK_ID_CATEGORY);
    for (int i = 0; i < NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS; i++) {
        track_params.process.tag[i] = azColName[i];
        char* arg = (char*) db->Sqlite3ColumnText(func, stmt, azColName, i);
        track_params.process.is_numeric[i] = Database::IsNumber(arg);
        if(track_params.process.is_numeric[i])
        {
            track_params.process.id[i] = db->Sqlite3ColumnInt(func, stmt, azColName, i);
        }
        else
        {
            track_params.process.name[i] = arg;
        }   
    }
    rocprofvis_dm_track_params_it it = db->FindTrack(track_params.process);
    db->UpdateQueryForTrack(it, track_params, callback_params->query);    
    if(it == db->TrackPropertiesEnd())
    {

        track_params.process.name[TRACK_ID_PID_OR_AGENT] = ProcessNameSuffixFor(track_params.process.category);
        track_params.process.name[TRACK_ID_PID_OR_AGENT] += (char*)db->Sqlite3ColumnText(func, stmt, azColName, TRACK_ID_PID_OR_AGENT);

        if (track_params.process.category != kRocProfVisDmPmcTrack){
            track_params.process.name[TRACK_ID_TID_OR_QUEUE] = SubProcessNameSuffixFor(track_params.process.category);
            track_params.process.name[TRACK_ID_TID_OR_QUEUE] += (char*)db->Sqlite3ColumnText(func, stmt, azColName, TRACK_ID_TID_OR_QUEUE);
            db->find_track_map[track_params.process.id[TRACK_ID_PID_OR_AGENT]][track_params.process.id[TRACK_ID_TID_OR_QUEUE]] = track_params.track_id;
        } else
        {
            track_params.process.name[TRACK_ID_TID_OR_QUEUE] = (char*)db->Sqlite3ColumnText(func, stmt, azColName, TRACK_ID_TID_OR_QUEUE);
            db->find_track_pmc_map[track_params.process.id[TRACK_ID_PID_OR_AGENT]][track_params.process.name[TRACK_ID_TID_OR_QUEUE]] = track_params.track_id;
        }

        if (kRocProfVisDmResultSuccess != db->AddTrackProperties(track_params)) return 1;
        if (db->BindObject()->FuncAddTrack(db->BindObject()->trace_object, db->TrackPropertiesLast()) != kRocProfVisDmResultSuccess) return 1; 

        if (track_params.process.category == kRocProfVisDmRegionTrack) {
            db->CachedTables()->AddTableCell("Process", track_params.process.id[TRACK_ID_PID], azColName[TRACK_ID_PID], (char*)db->Sqlite3ColumnText(func, stmt, azColName, TRACK_ID_PID));
            db->CachedTables()->AddTableCell("Thread", track_params.process.id[TRACK_ID_TID], azColName[TRACK_ID_TID], (char*)db->Sqlite3ColumnText(func, stmt, azColName, TRACK_ID_TID));
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Process", track_params.process.id[TRACK_ID_PID]) != kRocProfVisDmResultSuccess) return 1;
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Thread", track_params.process.id[TRACK_ID_TID]) != kRocProfVisDmResultSuccess) return 1;
        } else 
        if(track_params.process.category == kRocProfVisDmKernelDispatchTrack || 
            track_params.process.category == kRocProfVisDmMemoryAllocationTrack || 
            track_params.process.category == kRocProfVisDmMemoryCopyTrack || 
            track_params.process.category == kRocProfVisDmPmcTrack)
        {
            db->CachedTables()->AddTableCell("Agent", track_params.process.id[TRACK_ID_AGENT], azColName[TRACK_ID_AGENT], (char*)db->Sqlite3ColumnText(func, stmt, azColName, TRACK_ID_AGENT));
            db->CachedTables()->AddTableCell("Queue", track_params.process.id[TRACK_ID_QUEUE], azColName[TRACK_ID_QUEUE], (char*)db->Sqlite3ColumnText(func, stmt, azColName, TRACK_ID_QUEUE));
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Agent", track_params.process.id[TRACK_ID_AGENT]) != kRocProfVisDmResultSuccess) return 1;
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Queue", track_params.process.id[TRACK_ID_QUEUE]) != kRocProfVisDmResultSuccess) return 1;
        }
    }
    callback_params->future->CountThisRow();
    return 0;
}

int RocpdDatabase::CallBackAddString(void *data, int argc, sqlite3_stmt* stmt, char **azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(argc==2, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void* func = (void*)&CallBackAddString;
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    RocpdDatabase* db = (RocpdDatabase*)callback_params->db;
    if(callback_params->future->Interrupted()) return SQLITE_ABORT;
    std::stringstream ids((char*)db->Sqlite3ColumnText(func, stmt, azColName, 1));
    uint32_t string_id = db->BindObject()->FuncAddString(db->BindObject()->trace_object, (char*)db->Sqlite3ColumnText(func, stmt, azColName, 0));
    if (string_id == INVALID_INDEX) return 1;
    
    for (uint64_t id; ids >> id;) {   
        db->m_string_index_map[id] = string_id;
        db->m_string_id_map[string_id] = id;
        if (ids.peek() == ',') ids.ignore();
    }
    callback_params->future->CountThisRow();
    return 0;
}


int RocpdDatabase::CallbackAddStackTrace(void *data, int argc, sqlite3_stmt* stmt, char **azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(argc==4, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    void* func = (void*)&CallbackAddStackTrace;
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    RocpdDatabase* db = (RocpdDatabase*)callback_params->db;
    rocprofvis_db_stack_data_t record;
    if(callback_params->future->Interrupted()) return SQLITE_ABORT;
    record.symbol = (char*)db->Sqlite3ColumnText(func, stmt, azColName, 0);
    record.args = (char*)db->Sqlite3ColumnText(func, stmt, azColName, 1);
    record.line = (char*)db->Sqlite3ColumnText(func, stmt, azColName, 2);
    record.depth = db->Sqlite3ColumnInt(func, stmt, azColName, 3);
    if (db->BindObject()->FuncAddStackFrame(callback_params->handle,record) != kRocProfVisDmResultSuccess) return 1;
    callback_params->future->CountThisRow();
    return 0;
}       

rocprofvis_dm_result_t
RocpdDatabase::CreateIndexes()
{
    std::vector<std::string> vec;
    vec.push_back("CREATE INDEX IF NOT EXISTS pid_tid_idx ON rocpd_api(pid, tid, start);");
    vec.push_back("CREATE INDEX IF NOT EXISTS gid_qid_idx ON rocpd_op(gpuId, queueId, start);");
    vec.push_back("CREATE INDEX IF NOT EXISTS monitorTypeIdx on rocpd_monitor(gpuId,monitorType,start);");

    return ExecuteTransaction(vec);
}

rocprofvis_dm_result_t  RocpdDatabase::ReadTraceMetadata(Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    while (true)
    {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties, ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);

        ShowProgress(10, "Indexing tables (once per database)", kRPVDbBusy, future);
        CreateIndexes();

        ShowProgress(5, "Adding HIP API tracks", kRPVDbBusy, future );
        if(kRocProfVisDmResultSuccess != ExecuteSQLQuery(
            future,
            { 
                        // Track query by pid/tid
                     Builder::Select(rocprofvis_db_sqlite_track_query_format(
                     { { Builder::SpaceSaver(0),
                         Builder::QParam("pid", Builder::PROCESS_ID_SERVICE_NAME),
                         Builder::QParam("tid", Builder::THREAD_ID_SERVICE_NAME),
                         Builder::QParamCategory(kRocProfVisDmRegionTrack) },
                       { Builder::From("rocpd_api") } })), 
                        // Track query by stream
                        "",
                        // Level query
                     Builder::Select(rocprofvis_db_sqlite_level_query_format(
                     { { Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                         Builder::QParam("start", Builder::START_SERVICE_NAME), 
                         Builder::QParam("end", Builder::END_SERVICE_NAME),
                         Builder::QParam("id"), 
                         Builder::SpaceSaver(0),
                         Builder::QParam("pid", Builder::PROCESS_ID_SERVICE_NAME),
                         Builder::QParam("tid", Builder::THREAD_ID_SERVICE_NAME),
                         Builder::SpaceSaver(0) },
                       { Builder::From("rocpd_api") } })),
                        // Slice query by queue
                     Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                     { { Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                         Builder::QParam("start", Builder::START_SERVICE_NAME), 
                         Builder::QParam("end", Builder::END_SERVICE_NAME),
                         Builder::QParam("args_id"), 
                         Builder::QParam("apiName_id"),
                         Builder::QParam("id"),
                         Builder::SpaceSaver(0),
                         Builder::QParam("pid", Builder::PROCESS_ID_SERVICE_NAME),
                         Builder::QParam("tid", Builder::THREAD_ID_SERVICE_NAME),
                         Builder::QParam("L.level") },
                       { Builder::From("rocpd_api"),
                         Builder::LeftJoin(Builder::LevelTable("api"), "L", "id = L.eid") } })),
                        // Slice query by stream
                        "",
                        // Table query
                     GetEventOperationQuery(kRocProfVisDmOperationLaunch)
            },
                        &CallBackAddTrack)) break;

        ShowProgress(5, "Adding kernel dispatch tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(
            future, 
            {
                        // Track query by agent/queue
                     Builder::Select(rocprofvis_db_sqlite_track_query_format(
                     { { Builder::SpaceSaver(0),
                         Builder::QParam("gpuId", Builder::AGENT_ID_SERVICE_NAME),
                         Builder::QParam("queueId", Builder::QUEUE_ID_SERVICE_NAME),
                         Builder::QParamCategory(kRocProfVisDmKernelDispatchTrack) },
                       { Builder::From("rocpd_op") } })),
                        // Track query by stream
                        "",
                        // Level query
                     Builder::Select(rocprofvis_db_sqlite_level_query_format(
                       { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                         Builder::QParam("start", Builder::START_SERVICE_NAME), 
                         Builder::QParam("end", Builder::END_SERVICE_NAME),
                         Builder::QParam("id"), 
                         Builder::SpaceSaver(0),
                         Builder::QParam("gpuId", Builder::AGENT_ID_SERVICE_NAME),
                         Builder::QParam("queueId", Builder::QUEUE_ID_SERVICE_NAME),
                         Builder::SpaceSaver(0) },
                       { Builder::From("rocpd_op") } })),
                        // Slice query by queue
                     Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                       { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                         Builder::QParam("start", Builder::START_SERVICE_NAME), 
                         Builder::QParam("end", Builder::END_SERVICE_NAME),
                         Builder::QParam("opType_id"), 
                         Builder::QParam("description_id"),
                         Builder::QParam("id"), 
                         Builder::SpaceSaver(0),
                         Builder::QParam("gpuId", Builder::AGENT_ID_SERVICE_NAME),
                           Builder::QParam("queueId", Builder::QUEUE_ID_SERVICE_NAME),
                         Builder::QParam("L.level") },
                       { Builder::From("rocpd_op"),
                         Builder::LeftJoin(Builder::LevelTable("op"), "L", "id = L.eid") } })),
                        // Slice query by stream
                        "",
                        // Table query
                     GetEventOperationQuery(kRocProfVisDmOperationDispatch)
            },
                        &CallBackAddTrack)) break;

        ShowProgress(5, "Adding performance counters tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(
            future, 
            { 
                        // Track query by agent/monitorType
                     Builder::Select(rocprofvis_db_sqlite_track_query_format(
                     { { Builder::SpaceSaver(0),
                         Builder::QParam("deviceId", Builder::AGENT_ID_SERVICE_NAME),
                         Builder::QParam("monitorType", Builder::COUNTER_NAME_SERVICE_NAME),
                         Builder::QParamCategory(kRocProfVisDmPmcTrack) },
                       { Builder::From("rocpd_monitor where deviceId > 0") } })),
                        // Track query by stream
                        "",
                        // Level query
                     Builder::Select(rocprofvis_db_sqlite_level_query_format(
                     { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                         Builder::QParam("start", Builder::START_SERVICE_NAME), 
                         Builder::QParam("start", Builder::END_SERVICE_NAME),
                         Builder::SpaceSaver(0),
                         Builder::SpaceSaver(0),
                         Builder::QParam("deviceId", Builder::AGENT_ID_SERVICE_NAME),
                         Builder::QParam("monitorType", Builder::COUNTER_NAME_SERVICE_NAME),
                         Builder::SpaceSaver(0) },
                       { Builder::From("rocpd_monitor") } })),
                        // Slice query by monitorType
                     Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                     { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                         Builder::QParam("start", Builder::START_SERVICE_NAME), 
                         Builder::QParam("value"), 
                         Builder::QParam("start", Builder::END_SERVICE_NAME), 
                         Builder::SpaceSaver(0),
                         Builder::SpaceSaver(0), 
                         Builder::SpaceSaver(0),
                         Builder::QParam("deviceId", Builder::AGENT_ID_SERVICE_NAME),
                         Builder::QParam("monitorType", Builder::COUNTER_NAME_SERVICE_NAME),
                         Builder::QParam("CAST(value AS REAL)", "level") },
                       { Builder::From("rocpd_monitor") } })),
                        // Slice query by stream
                        "",
                     GetEventOperationQuery(kRocProfVisDmOperationNoOp)
            },
                        &CallBackAddTrack)) break;

        ShowProgress(20, "Loading strings", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, "SELECT string, GROUP_CONCAT(id) AS ids FROM rocpd_string GROUP BY string;", &CallBackAddString)) break;

        ShowProgress(5, "Counting events", kRPVDbBusy, future);
        TraceProperties()->events_count[kRocProfVisDmOperationLaunch]   = 0;
        TraceProperties()->events_count[kRocProfVisDmOperationDispatch] = 0;
        if(kRocProfVisDmResultSuccess !=
           ExecuteQueryForAllTracksAsync(kRocProfVisDmIncludePmcTracks | kRocProfVisDmIncludeStreamTracks, kRPVQueryLevel,
               "SELECT COUNT(*),  op, ",
               ";", &CallbackGetTrackRecordsCount,
               [](rocprofvis_dm_track_params_t* params) {}))
        {
            break;
        }

        std::vector<std::string> to_drop = Builder::OldLevelTables("api");
        for(auto table : to_drop)
        {
            DropSQLTable(table.c_str());
        }
        to_drop = Builder::OldLevelTables("op");
        for(auto table : to_drop)
        {
            DropSQLTable(table.c_str());
        }

        if(SQLITE_OK != DetectTable(GetServiceConnection(), Builder::LevelTable("api").c_str(), false) ||
           SQLITE_OK != DetectTable(GetServiceConnection(), Builder::LevelTable("op").c_str(), false))
        {
            m_event_levels[kRocProfVisDmOperationLaunch].reserve(
                TraceProperties()->events_count[kRocProfVisDmOperationLaunch]);
            m_event_levels[kRocProfVisDmOperationDispatch].reserve(
                TraceProperties()->events_count[kRocProfVisDmOperationDispatch]);

            ShowProgress(10, "Calculating event levels (once per database)", kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteQueryForAllTracksAsync(
                   0,
                   kRPVQueryLevel,
                   "SELECT *, ", (std::string(" ORDER BY ")+Builder::START_SERVICE_NAME).c_str(), &CalculateEventLevels,
                   [](rocprofvis_dm_track_params_t* params) {
                       params->m_active_events.clear();
                   }))
            {
                break;
            }

            ShowProgress(10, "Save event levels", kRPVDbBusy, future);
            SQLInsertParams params[] = { { "eid", "INTEGER PRIMARY KEY" },
                                         { "level", "INTEGER" } };
            CreateSQLTable(
                Builder::LevelTable("api").c_str(), params, 2,
                m_event_levels[kRocProfVisDmOperationLaunch].size(),
                [&](sqlite3_stmt* stmt, int index) {
                    sqlite3_bind_int64(
                        stmt, 1, m_event_levels[kRocProfVisDmOperationLaunch][index].id);
                    sqlite3_bind_int(
                        stmt, 2,
                        m_event_levels[kRocProfVisDmOperationLaunch][index].level_for_queue);
                });
            m_event_levels[kRocProfVisDmOperationLaunch].clear();
            m_event_levels_id_to_index[kRocProfVisDmOperationLaunch].clear();
            CreateSQLTable(
                Builder::LevelTable("op").c_str(), params, 2,
                m_event_levels[kRocProfVisDmOperationDispatch].size(),
                [&](sqlite3_stmt* stmt, int index) {
                    sqlite3_bind_int64(
                        stmt, 1,
                        m_event_levels[kRocProfVisDmOperationDispatch][index].id);
                    sqlite3_bind_int(
                        stmt, 2,
                        m_event_levels[kRocProfVisDmOperationDispatch][index].level_for_queue);
                });
            m_event_levels[kRocProfVisDmOperationDispatch].clear();
            m_event_levels_id_to_index[kRocProfVisDmOperationDispatch].clear();
        }
      
        ShowProgress(5, "Collecting track properties",
                     kRPVDbBusy, future);
        TraceProperties()->start_time                                   = UINT64_MAX;
        TraceProperties()->end_time                                     = 0;
        if(kRocProfVisDmResultSuccess !=
           ExecuteQueryForAllTracksAsync(
               kRocProfVisDmIncludePmcTracks | kRocProfVisDmIncludeStreamTracks,
               kRPVQuerySliceByTrackSliceQuery,
               "SELECT MIN(startTs), MAX(endTs), MIN(level), MAX(level), ",
               "WHERE startTs != 0 AND endTs != 0", &CallbackGetTrackProperties,
               [](rocprofvis_dm_track_params_t* params) {}))
        {
            break;
        }

        ShowProgress(5, "Collecting track histogram",
            kRPVDbBusy, future);

        constexpr uint64_t desired_bins = 500;
        uint64_t           trace_length =
            TraceProperties()->end_time - TraceProperties()->start_time;

        uint64_t bucket_size = (trace_length + desired_bins) / desired_bins;
        if(bucket_size == 0) bucket_size = 1;

        uint64_t bucket_count = (trace_length + bucket_size) / bucket_size;
        if(bucket_count == 0) bucket_count = 1;

        TraceProperties()->histogram_bucket_size = bucket_size;
        TraceProperties()->histogram_bucket_count = (trace_length+bucket_size)/bucket_size;

        std::string histogram_query = "SELECT (startTs - " +
            std::to_string(TraceProperties()->start_time) +") / " + 
            std::to_string(bucket_size) + " AS bucket, COUNT(*) AS count, ";

        if(kRocProfVisDmResultSuccess !=
            ExecuteQueryForAllTracksAsync(
                kRocProfVisDmTrySplitTrack, kRPVQuerySliceByTrackSliceQuery,
                histogram_query.c_str(),
                "GROUP BY bucket", &CallbackMakeHistogramPerTrack,
                [](rocprofvis_dm_track_params_t* params) {}))
        {
            break;
        }

        TraceProperties()->metadata_loaded=true;
        ShowProgress(100-future->Progress(), "Trace metadata successfully loaded", kRPVDbSuccess, future );
        return future->SetPromise(kRocProfVisDmResultSuccess);

    }
    ShowProgress(0, "Trace metadata not loaded!", kRPVDbError, future );
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t RocpdDatabase::SaveTrimmedData(rocprofvis_dm_timestamp_t start,
                                 rocprofvis_dm_timestamp_t end,
                                 rocprofvis_dm_charptr_t new_db_path, Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(new_db_path, "New DB path cannot be NULL.",
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocprofvis_dm_result_t result = kRocProfVisDmResultInvalidParameter;

    std::string                          query;
    rocprofvis_db_sqlite_trim_parameters trim_tables;
    rocprofvis_db_sqlite_trim_parameters trim_views;

    Future* internal_future = new Future(nullptr);

    {
        RocpdDatabase rpDb(new_db_path);
        result = rpDb.Open();

        if(result == kRocProfVisDmResultSuccess)
        {
            ShowProgress(1, "Iterate over tables", kRPVDbBusy, future);
            query = "SELECT name, sql FROM sqlite_master WHERE type='table';";
            rocprofvis_db_sqlite_callback_parameters params = {
                this,
                internal_future,
                &trim_tables,
                &CallbackTrimTableQuery,
                { query.c_str(), "", "", "" },
                static_cast<rocprofvis_dm_track_id_t>(-1)
            };

            result = ExecuteSQLQuery(query.c_str(), &params);
        }

        if(result == kRocProfVisDmResultSuccess)
        {
            for(auto const& table : trim_tables.tables)
            {
                if(table.first != "sqlite_master" && table.first != "sqlite_sequence")
                {
                    if(result == kRocProfVisDmResultSuccess)
                    {
                        std::string msg = "Copy table " + table.first;
                        ShowProgress(1, msg.c_str(), kRPVDbBusy, future);

                        query = "DROP TABLE ";
                        query += table.first;
                        query += ";";
                        rpDb.ExecuteSQLQuery(internal_future, query.c_str());

                        query  = table.second;
                        result = rpDb.ExecuteSQLQuery(internal_future, query.c_str());
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        if(result == kRocProfVisDmResultSuccess)
        {
            ShowProgress(1, "Iterate over views", kRPVDbBusy, future);
            query = "SELECT name, sql FROM sqlite_master WHERE type='view';";
            rocprofvis_db_sqlite_callback_parameters params = {
                this,
                internal_future,
                &trim_views,
                &CallbackTrimTableQuery,
                { query.c_str(), "", "", "" },
                static_cast<rocprofvis_dm_track_id_t>(-1)
            };

            result = ExecuteSQLQuery(query.c_str(), &params);
        }

        if(result == kRocProfVisDmResultSuccess)
        {
            for(auto const& view : trim_views.tables)
            {
                if(result == kRocProfVisDmResultSuccess)
                {
                    std::string msg = "Copy views " + view.first;
                    ShowProgress(1, msg.c_str(), kRPVDbBusy, future);

                    query = "DROP VIEW ";
                    query += view.first;
                    query += ";";
                    rpDb.ExecuteSQLQuery(internal_future, query.c_str());

                    query = view.second;

                    result = rpDb.ExecuteSQLQuery(internal_future, query.c_str());
                }
                else
                {
                    break;
                }
            }
        }

        if(result == kRocProfVisDmResultSuccess)
        {
            ShowProgress(0, "Attaching old DB to new", kRPVDbBusy, future);
            query = "ATTACH DATABASE '";
            query += Path();
            query += "' as 'oldDb';";
            result = rpDb.ExecuteSQLQuery(internal_future, query.c_str());
        }

        if(result == kRocProfVisDmResultSuccess)
        {
            for(auto const& table : trim_tables.tables)
            {
                if(table.first != "sqlite_master" && table.first != "sqlite_sequence")
                {
                    if(result == kRocProfVisDmResultSuccess)
                    {
                        std::string msg = "Copy table " + table.first;
                        ShowProgress(1, msg.c_str(), kRPVDbBusy, future);

                        if(result == kRocProfVisDmResultSuccess)
                        {
                            if(!strcmp(table.first.c_str(), "rocpd_api") ||
                               !strcmp(table.first.c_str(), "rocpd_monitor") ||
                               !strcmp(table.first.c_str(), "rocpd_op"))
                            {
                                query = "INSERT INTO ";
                                query += table.first;
                                query += " SELECT * FROM oldDb.";
                                query += table.first;
                                query += " WHERE start < ";
                                query += std::to_string(end);
                                query += " AND end > ";
                                query += std::to_string(start);
                                query += ";";
                            }
                            else
                            {
                                query = "INSERT INTO ";
                                query += table.first;
                                query += " SELECT * FROM oldDb.";
                                query += table.first;
                                query += ";";
                            }

                            result = rpDb.ExecuteSQLQuery(internal_future, query.c_str());
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        if(result == kRocProfVisDmResultSuccess)
        {
            ShowProgress(0, "Detaching old DB", kRPVDbBusy, future);
            query  = "DETACH oldDb;";
            result = rpDb.ExecuteSQLQuery(internal_future, query.c_str());
        }
    }

    if(result == kRocProfVisDmResultSuccess)
    {
        ShowProgress(100, "Trace trimmed successfully!", kRPVDbSuccess, future);
    }
    else
    {
        ShowProgress(0, "Failed to trim track!", kRPVDbError, future);
    }

    delete internal_future;

    return future->SetPromise(result);
}

rocprofvis_dm_result_t RocpdDatabase::BuildTableStringIdFilter(rocprofvis_dm_num_string_table_filters_t num_string_table_filters,
    rocprofvis_dm_string_table_filters_t     string_table_filters,
    table_string_id_filter_map_t&            filter)
{
    rocprofvis_dm_result_t result = kRocProfVisDmResultSuccess;
    if(num_string_table_filters > 0)
    {
        result = kRocProfVisDmResultUnknownError;
        ROCPROFVIS_ASSERT_RETURN(BindObject()->trace_object, kRocProfVisDmResultInvalidParameter);
        std::vector<rocprofvis_dm_index_t> string_indices;
        result = ((Trace*)BindObject()->trace_object)->GetStringIndicesWithSubstring(num_string_table_filters, string_table_filters, string_indices);
        ROCPROFVIS_ASSERT_RETURN(result == kRocProfVisDmResultSuccess, result);
        std::string string;
        for(const rocprofvis_dm_index_t& index : string_indices)
        {
            string += string.empty() ? "\"" + std::string(((Trace*)BindObject()->trace_object)->GetStringAt(index)) + "\"" 
                : ", \"" + std::string(((Trace*)BindObject()->trace_object)->GetStringAt(index)) + "\"";
        }
        if(!string.empty())
        {
            filter[kRocProfVisDmOperationLaunch] = " category IN (" + string + ") OR name IN (" + string;
            filter[kRocProfVisDmOperationDispatch] = " category IN (" + string + ") OR name IN (" + string;
        }
    }   
    return result;
}

rocprofvis_dm_string_t RocpdDatabase::GetEventOperationQuery(const rocprofvis_dm_event_operation_t operation)
{
    switch(operation)
    {
        case kRocProfVisDmOperationLaunch:
        {
            return Builder::Select(rocprofvis_db_sqlite_rocpd_table_query_format(
            { this,
            { Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                Builder::QParam("id"), 
                Builder::QParam("apiName", "category"),
                Builder::QParam("args", "name"), 
                Builder::QParam("start", Builder::START_SERVICE_NAME),
                Builder::QParam("end", Builder::END_SERVICE_NAME),
                Builder::QParam("(end-start)", "duration"),
                Builder::QParam("pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("tid", Builder::THREAD_ID_SERVICE_NAME) },
            { Builder::From("api") } }));
        }
        case kRocProfVisDmOperationDispatch:
        {
            return Builder::Select(rocprofvis_db_sqlite_rocpd_table_query_format(
            { this,
            { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("id"), 
                Builder::QParam("opType", "category"),
                Builder::QParam("description", "name"), 
                Builder::QParam("start", Builder::START_SERVICE_NAME),
                Builder::QParam("end", Builder::END_SERVICE_NAME),
                Builder::QParam("(end-start)", "duration"),
                Builder::QParam("gpuId", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("queueId", Builder::QUEUE_ID_SERVICE_NAME) },
            { Builder::From("op") } }));
        }
        case kRocProfVisDmOperationNoOp:
        {
            return Builder::Select(rocprofvis_db_sqlite_sample_table_query_format(
            { this,
            { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("id"), 
                Builder::QParam("monitorType", Builder::COUNTER_NAME_SERVICE_NAME),
                Builder::QParam("CAST(value AS REAL)", "value"), 
                Builder::QParam("start", Builder::START_SERVICE_NAME),
                Builder::QParam("start", Builder::END_SERVICE_NAME),
                Builder::QParam("deviceId",  Builder::AGENT_ID_SERVICE_NAME)},
            { Builder::From("rocpd_monitor") } }));
        }
        default:
        {
            return "";
        }
    }
}

rocprofvis_dm_result_t RocpdDatabase::StringIndexToId(rocprofvis_dm_index_t index, rocprofvis_dm_id_t& id)
{
    rocprofvis_dm_result_t result = kRocProfVisDmResultInvalidParameter;
    if(m_string_id_map.count(index) > 0)
    {
        id = m_string_id_map.at(index);
        result = kRocProfVisDmResultSuccess;
    }
    return result;
}

rocprofvis_dm_result_t
RocpdDatabase::BuildTableSummaryClause(bool sample_query,
                                 rocprofvis_dm_string_t& select,
                                 rocprofvis_dm_string_t& group_by)
{
    if(sample_query)
    {
        select = "monitorType, AVG(value) AS avg_value, MIN(value) AS min_value, MAX(value) AS max_value";
        group_by = "monitorType";
    }
    else
    {
        select = "name, COUNT(*) AS num_invocations, AVG(duration) AS avg_duration, MIN(duration) AS min_duration, MAX(duration) AS max_duration, SUM(duration) AS total_duration";
        group_by = "name";
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  RocpdDatabase::ReadFlowTraceInfo(
        rocprofvis_dm_event_id_t event_id,
        Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    while (true)
    {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties, ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);
        rocprofvis_dm_flowtrace_t flowtrace = BindObject()->FuncAddFlowTrace(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(flowtrace, ERROR_FLOW_TRACE_CANNOT_BE_NULL);
        std::stringstream query;
        if (event_id.bitfield.event_op == kRocProfVisDmOperationLaunch)
        {
            query << "select 2, rocpd_api_ops.api_id, rocpd_api_ops.op_id, 0, gpuId, queueId, rocpd_op.start, rocpd_op.opType_id, rocpd_op.description_id, " << Builder::LevelTable("op") << ".level, rocpd_op.end AS end "
                "from rocpd_api_ops "
                "INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id "
                "INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id "
                "INNER JOIN " << Builder::LevelTable("op") << " on rocpd_api_ops.op_id = " << Builder::LevelTable("op") << ".eid " 
                "where rocpd_api_ops.api_id = ";
            query << event_id.bitfield.event_id;  
            ShowProgress(0, query.str().c_str(),kRPVDbBusy, future);
            if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.str().c_str(), flowtrace, &CallbackAddFlowTrace)) break;
        } else
        if (event_id.bitfield.event_op == kRocProfVisDmOperationDispatch)
        {
            query << "select 1, rocpd_api_ops.op_id, rocpd_api_ops.api_id, 0, pid, tid, rocpd_api.apiName_id, rocpd_api.args_id, " << Builder::LevelTable("api") << ".level, rocpd_api.end "
                "from rocpd_api_ops "
                "INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id "
                "INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id "
                "INNER JOIN " << Builder::LevelTable("api") << " on rocpd_api_ops.api_id = " << Builder::LevelTable("api") << ".eid " 
                "where rocpd_api_ops.op_id = ";
            query << event_id.bitfield.event_id << ";";  
            ShowProgress(0, query.str().c_str(),kRPVDbBusy, future);
            if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.str().c_str(), flowtrace, &CallbackAddFlowTrace)) break;
        } else
        {
            ShowProgress(0, "Flow trace is not available for specified operation type!", kRPVDbError, future );
            return future->SetPromise(kRocProfVisDmResultInvalidParameter);
        }
        ShowProgress(100, "Flow trace successfully loaded!",kRPVDbSuccess, future);
        return future->SetPromise(kRocProfVisDmResultSuccess);
    }
    ShowProgress(0, "Flow trace not loaded!", kRPVDbError, future );
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t  RocpdDatabase::ReadStackTraceInfo(
        rocprofvis_dm_event_id_t event_id,
        Future* future)
{

    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    while (true)
    {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties, ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);
        rocprofvis_dm_stacktrace_t stacktrace = BindObject()->FuncAddStackTrace(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(stacktrace, ERROR_STACK_TRACE_CANNOT_BE_NULL);
#ifdef SUPPORT_OLD_SCHEMA_STACK_TRACE 
        std::stringstream query;
        if (event_id.bitfield.event_op == kRocProfVisDmOperationLaunch || event_id.bitfield.event_op == kRocProfVisDmOperationMemoryAllocate)
        {
            query << "select s2.string as hip_api, s3.string as args, s1.string as frame, sf.depth "
                     "from rocpd_stackframe sf join rocpd_string s1 on sf.name_id=s1.id join rocpd_api ap on "
                     "sf.api_ptr_id=ap.id join rocpd_string s2 on ap.apiName_id=s2.id join rocpd_string s3 on ap.args_id=s3.id where ap.id  == ";
            query << event_id.bitfield.event_id << ";";  
            ShowProgress(0, query.str().c_str(),kRPVDbBusy, future);
            if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.str().c_str(), stacktrace, &CallbackAddStackTrace)) break;
        } else
        {
            ShowProgress(0, "Stack trace is not available for specified operation type!", kRPVDbError, future );
            return future->SetPromise(kRocProfVisDmResultInvalidParameter);
        }
        ShowProgress(100, "Stack trace successfully loaded!", kRPVDbSuccess, future);
#else
        ShowProgress(100, "Stack trace is not supported for old schema database!", kRPVDbSuccess, future);
#endif

        return future->SetPromise(kRocProfVisDmResultSuccess);
    }
    ShowProgress(0, "Stack trace not loaded!", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort
                                                    : kRocProfVisDmResultDbAccessFailed);
}


rocprofvis_dm_result_t  RocpdDatabase::ReadExtEventInfo(
            rocprofvis_dm_event_id_t event_id, 
            Future* future){
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    while (true)
    {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties, ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);
        rocprofvis_dm_extdata_t extdata = BindObject()->FuncAddExtData(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(extdata, ERROR_EXT_DATA_CANNOT_BE_NULL);
        std::stringstream query;
        if (event_id.bitfield.event_op == kRocProfVisDmOperationLaunch)
        {
            query << "select *, end-start as duration from api where id == ";
            query << event_id.bitfield.event_id << ";";  
            ShowProgress(0, query.str().c_str(),kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.str().c_str(), "Properties", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;
            query.str("");
            query << "select *, end-start as duration from copy where id == ";
            query << event_id.bitfield.event_id << ";";  
            ShowProgress(50, query.str().c_str(),kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.str().c_str(), "Copy", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;

            std::string level_query = Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                    Builder::SpaceSaver(0),
                    Builder::QParam("pid", Builder::PROCESS_ID_SERVICE_NAME),
                    Builder::QParam("tid", Builder::THREAD_ID_SERVICE_NAME),
                    Builder::SpaceSaver(0), 
                    Builder::QParam("L.level"),
                    Builder::SpaceSaver(0) },
                  { Builder::From("rocpd_api"),
                    Builder::LeftJoin(Builder::LevelTable("api"), "L", "id = L.eid") },
                  { Builder::Where(
                      "id", "==", std::to_string(event_id.bitfield.event_id)) } }));
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(future, level_query.c_str(), extdata, &CallbackAddEssentialInfo))
                break;
        } else
        if (event_id.bitfield.event_op == kRocProfVisDmOperationDispatch)
        {
            query << "select *, end-start as duration, 'GPU' as agent_type from op where id == ";
            query << event_id.bitfield.event_id << ";";  
            ShowProgress(0, query.str().c_str(),kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.str().c_str(), "Properties", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;
            query.str("");
            query << "select * from copyop where id == ";
            query << event_id.bitfield.event_id << ";";  
            ShowProgress(25, query.str().c_str(),kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.str().c_str(), "Copy", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;
            query.str("");
            query << "select * from kernel where id == ";
            query << event_id.bitfield.event_id << ";";  
            ShowProgress(50, query.str().c_str(),kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.str().c_str(), "Dispatch", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;

            std::string level_query =
                Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                    { { Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                        Builder::SpaceSaver(0),
                        Builder::QParam("gpuId", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("queueId", Builder::QUEUE_ID_SERVICE_NAME),
                        Builder::SpaceSaver(0), Builder::QParam("L.level"),
                        Builder::SpaceSaver(0) },
                      { Builder::From("rocpd_op"),
                        Builder::LeftJoin(Builder::LevelTable("op"), "L", "id = L.eid") },
                      { Builder::Where(
                          "id", "==", std::to_string(event_id.bitfield.event_id)) } }));
        } else    
        {
            ShowProgress(0, "Extended data not available for specified operation type!", kRPVDbError, future );
            return future->SetPromise(kRocProfVisDmResultInvalidParameter);
        }
        ShowProgress(50, "Extended data successfully loaded!",kRPVDbSuccess, future);
        return future->SetPromise(kRocProfVisDmResultSuccess);
    }
    ShowProgress(0, "Extended data  not loaded!", kRPVDbError, future );
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultTimeout : kRocProfVisDmResultDbAccessFailed);    
}

}  // namespace DataModel
}  // namespace RocProfVis