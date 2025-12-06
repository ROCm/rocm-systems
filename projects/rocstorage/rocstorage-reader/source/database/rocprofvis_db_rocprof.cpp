// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_db_rocprof.h"
#include "rocprofvis_dm_trace.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <string.h>

namespace RocProfVis
{
namespace DataModel
{

rocprofvis_dm_result_t RocprofDatabase::FindTrackId(
                                                    uint64_t node,
                                                    uint32_t process,
                                                    const char* subprocess,
                                                    rocprofvis_dm_op_t operation,
                                                    rocprofvis_dm_track_id_t& track_id) {
    
    auto it_op = find_track_map.find(strcmp(subprocess, "-1") == 0 ? kRPVTrackSearchIdStreams : 
        GetTrackSearchId(TranslateOperationToTrackCategory((rocprofvis_dm_event_operation_t) operation)));
    if(it_op != find_track_map.end())
    {
        auto it_node = it_op->second.find(node);
        if(it_node != it_op->second.end())
        {
            auto it_process = it_node->second.find(process);
            if(it_process != it_node->second.end())
            {
                auto it_subprocess = it_process->second.find(std::atoll(subprocess));
                if(it_subprocess != it_process->second.end())
                {
                    track_id = it_subprocess->second;
                    return kRocProfVisDmResultSuccess;
                }
            }
        }
    }
    return kRocProfVisDmResultNotLoaded;
   
}

rocprofvis_dm_result_t RocprofDatabase::RemapStringIds(rocprofvis_db_record_data_t & record)
{
    if(record.event.id.bitfield.event_op==kRocProfVisDmOperationDispatch)
        record.event.symbol+=m_symbols_offset;
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RocprofDatabase::RemapStringIds(rocprofvis_db_flow_data_t& record)
{
    if(record.id.bitfield.event_op == kRocProfVisDmOperationDispatch)
        record.symbol_id += m_symbols_offset;
    return kRocProfVisDmResultSuccess;
}

int RocprofDatabase::CallbackParseMetadata(void* data, int argc, sqlite3_stmt* stmt,
    char** azColName) {
    ROCPROFVIS_ASSERT_MSG_RETURN(argc==3, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void* func = (void*)&CallbackParseMetadata;
    rocprofvis_dm_track_params_t track_params = {0};
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    RocprofDatabase* db = (RocprofDatabase*)callback_params->db;
    std::string tag = db->Sqlite3ColumnText(func, stmt, azColName, 1);
    if (tag == "schema_version")
    {
        db->db_version = db->Sqlite3ColumnText(func, stmt, azColName, 2);
    }
    callback_params->future->CountThisRow();
    return 0;
}

int RocprofDatabase::CallBackAddTrack(void *data, int argc, sqlite3_stmt* stmt, char **azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(argc==NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS+1, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void* func = (void*)&CallBackAddTrack;
    rocprofvis_dm_track_params_t track_params = {0};
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    RocprofDatabase* db = (RocprofDatabase*)callback_params->db;
    if(callback_params->future->Interrupted()) return SQLITE_ABORT;
    track_params.track_id = (rocprofvis_dm_track_id_t)db->NumTracks();
    track_params.process.category = (rocprofvis_dm_track_category_t)db->Sqlite3ColumnInt(func, stmt, azColName,TRACK_ID_CATEGORY);
    for (int i = 0; i < NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS; i++) {
        track_params.process.tag[i] = azColName[i];
        char* arg = (char*) db->Sqlite3ColumnText(func, stmt, azColName, i);
        track_params.process.is_numeric[i] = Database::IsNumber(arg);
        if (track_params.process.is_numeric[i]) {
            track_params.process.id[i] = db->Sqlite3ColumnInt64(func, stmt, azColName, i);
        } else {
            track_params.process.name[i] = arg;
        }        
    }
    
    rocprofvis_dm_track_params_it it = db->FindTrack(track_params.process);
    db->UpdateQueryForTrack(it, track_params, callback_params->query);
    if(it == db->TrackPropertiesEnd())
    {
        db->find_track_map[GetTrackSearchId(track_params.process.category)]
                          [track_params.process.id[TRACK_ID_NODE]]
                          [track_params.process.id[TRACK_ID_PID_OR_AGENT]]
                          [track_params.process.id[TRACK_ID_TID_OR_QUEUE]] =
            track_params.track_id;
        if(track_params.process.category == kRocProfVisDmRegionMainTrack ||
           track_params.process.category == kRocProfVisDmRegionSampleTrack)
        {

            track_params.process.name[TRACK_ID_PID] = db->CachedTables()->GetTableCell("Process", track_params.process.id[TRACK_ID_PID], "command");
            track_params.process.name[TRACK_ID_PID] += "(";
            track_params.process.name[TRACK_ID_PID] += std::to_string(track_params.process.id[TRACK_ID_PID]);
            track_params.process.name[TRACK_ID_PID] += ")";

            track_params.process.name[TRACK_ID_TID] = db->CachedTables()->GetTableCell("Thread", track_params.process.id[TRACK_ID_TID], "name");
            track_params.process.name[TRACK_ID_TID] += "(";
            track_params.process.name[TRACK_ID_TID] += std::to_string(track_params.process.id[TRACK_ID_TID]);
            track_params.process.name[TRACK_ID_TID] += ")";
        }
        else if(track_params.process.category == kRocProfVisDmKernelDispatchTrack ||
                track_params.process.category == kRocProfVisDmMemoryAllocationTrack ||
                track_params.process.category == kRocProfVisDmMemoryCopyTrack ||
                track_params.process.category == kRocProfVisDmPmcTrack)
          {
                track_params.process.name[TRACK_ID_AGENT] = db->CachedTables()->GetTableCell("Agent", track_params.process.id[TRACK_ID_AGENT], "product_name");
                track_params.process.name[TRACK_ID_AGENT] += "(";
                track_params.process.name[TRACK_ID_AGENT] += db->CachedTables()->GetTableCell("Agent", track_params.process.id[TRACK_ID_AGENT], "type_index");
                track_params.process.name[TRACK_ID_AGENT] += ")";
                if(track_params.process.category == kRocProfVisDmKernelDispatchTrack ||
                   track_params.process.category == kRocProfVisDmMemoryAllocationTrack ||
                   track_params.process.category == kRocProfVisDmMemoryCopyTrack)
                {
                    track_params.process.name[TRACK_ID_QUEUE] = db->CachedTables()->GetTableCell("Queue", track_params.process.id[TRACK_ID_QUEUE], "name");
                }
                else {
                    track_params.process.name[TRACK_ID_QUEUE] = db->CachedTables()->GetTableCell("PMC", track_params.process.id[TRACK_ID_QUEUE], "name");
                }
                
          }
          else if(track_params.process.category == kRocProfVisDmStreamTrack)
        {
                track_params.process.name[TRACK_ID_STREAM] = db->CachedTables()->GetTableCell("Stream", track_params.process.id[TRACK_ID_STREAM], "name");
        }
        if (kRocProfVisDmResultSuccess != db->AddTrackProperties(track_params)) return 1;
        if (db->BindObject()->FuncAddTrack(db->BindObject()->trace_object, db->TrackPropertiesLast()) != kRocProfVisDmResultSuccess) return 1;  
        if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Node", track_params.process.id[TRACK_ID_NODE]) != kRocProfVisDmResultSuccess) return 1;
        if(track_params.process.category == kRocProfVisDmRegionMainTrack ||
           track_params.process.category == kRocProfVisDmRegionSampleTrack)
        {
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Process", track_params.process.id[TRACK_ID_PID]) != kRocProfVisDmResultSuccess) return 1;
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Thread", track_params.process.id[TRACK_ID_TID]) != kRocProfVisDmResultSuccess) return 1;
        } 
        else if(track_params.process.category == kRocProfVisDmStreamTrack)
        {
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Stream", track_params.process.id[TRACK_ID_STREAM]) != kRocProfVisDmResultSuccess) return 1; 
        }
        else if(track_params.process.category == kRocProfVisDmKernelDispatchTrack ||
                track_params.process.category == kRocProfVisDmMemoryAllocationTrack ||
                track_params.process.category == kRocProfVisDmMemoryCopyTrack ||
                track_params.process.category == kRocProfVisDmPmcTrack)
        {
            if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Agent", track_params.process.id[TRACK_ID_AGENT]) != kRocProfVisDmResultSuccess) return 1; 
            if(track_params.process.category == kRocProfVisDmKernelDispatchTrack ||
               track_params.process.category == kRocProfVisDmMemoryAllocationTrack ||
               track_params.process.category == kRocProfVisDmMemoryCopyTrack)
            {
                if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "Queue", track_params.process.id[TRACK_ID_QUEUE]) != kRocProfVisDmResultSuccess) return 1;
            }
            else
            {
                if(track_params.process.id[TRACK_ID_QUEUE] > 0)
                    if (db->CachedTables()->PopulateTrackExtendedDataTemplate(db, "PMC", track_params.process.id[TRACK_ID_QUEUE]) != kRocProfVisDmResultSuccess) return 1;
            }
        }
    }
    callback_params->future->CountThisRow();
    return 0;
}

int RocprofDatabase::CallBackAddString(void *data, int argc, sqlite3_stmt* stmt, char **azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(argc==1, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void*  func = (void*)&CallBackAddString;
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    RocprofDatabase* db = (RocprofDatabase*)callback_params->db;
    if(callback_params->future->Interrupted()) return SQLITE_ABORT;
    uint32_t stringId = db->BindObject()->FuncAddString(db->BindObject()->trace_object, (char*) db->Sqlite3ColumnText(func, stmt, azColName,0));
    callback_params->future->CountThisRow();
    return 0;
}


int RocprofDatabase::CallbackCacheTable(void *data, int argc, sqlite3_stmt* stmt, char **azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void* func = (void*)&CallbackCacheTable;
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    RocprofDatabase* db = (RocprofDatabase*)callback_params->db;
    DatabaseCache * ref_tables = (DatabaseCache *)callback_params->handle;
    uint64_t id = db->Sqlite3ColumnInt64(func, stmt, azColName,0);
    for (int i=0; i < argc; i++)
    {
        ref_tables->AddTableCell(callback_params->query[kRPVCacheTableName], id, azColName[i],
                                 (char*) db->Sqlite3ColumnText(func, stmt, azColName, i));
    }
    callback_params->future->CountThisRow();
    return 0;
}

int
RocprofDatabase::CallbackNodeEnumeration(void* data, int argc, sqlite3_stmt* stmt,
                                         char** azColName)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void*  func = (void*)&CallbackNodeEnumeration;
    rocprofvis_db_sqlite_callback_parameters* callback_params =
        (rocprofvis_db_sqlite_callback_parameters*) data;
    RocprofDatabase* db = (RocprofDatabase*) callback_params->db;
    guid_list_t*     guid_list = (guid_list_t*) callback_params->handle;
    std::string      table_name_befor_guid = callback_params->query[kRPVCacheTableName];
    std::string      table_name            = (char*) db->Sqlite3ColumnText(func, stmt, azColName, 0);
    size_t           pos                   = table_name.find(table_name_befor_guid);
    if(pos == 0)
    {
        guid_list->push_back(table_name.substr(table_name_befor_guid.length()));
    }
    callback_params->future->CountThisRow();
    return 0;
}

int
RocprofDatabase::CallbackAddStackTrace(void* data, int argc, sqlite3_stmt* stmt,
                                       char** azColName)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(argc == 2, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    void*  func = (void*)&CallbackAddStackTrace;
    rocprofvis_db_sqlite_callback_parameters* callback_params =
        (rocprofvis_db_sqlite_callback_parameters*) data;
    RocprofDatabase*           db = (RocprofDatabase*) callback_params->db;
    rocprofvis_db_stack_data_t record;
    if(callback_params->future->Interrupted()) return 1;
    record.symbol = (char*) db->Sqlite3ColumnText(func, stmt, azColName, 0);
    record.line   = (char*) db->Sqlite3ColumnText(func, stmt, azColName, 1);
    record.args   = "";
    record.depth  = callback_params->future->GetProcessedRowsCount();
    if(db->BindObject()->FuncAddStackFrame(callback_params->handle, record) !=
       kRocProfVisDmResultSuccess)
        return 1;
    callback_params->future->CountThisRow();
    return 0;
}   

rocprofvis_dm_result_t
RocprofDatabase::CreateIndexes()
{ 
    std::vector<std::string> vec;
    for(int i = 0; i < GuidList().size(); i++)
    {
        if(m_query_factory.IsVersionGreaterOrEqual("4"))
        {
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_region_track_idx_") + GuidList()[i] +
                     " ON rocpd_track_" + GuidList()[i] + "(nid,pid,tid);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_kernel_dispatch_track_idx_") + GuidList()[i] +
                     " ON rocpd_track_" + GuidList()[i] + "(nid,agent_id,queue_id);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_stream_idx_") + GuidList()[i] +
                     " ON rocpd_track_" + GuidList()[i] + "(nid,stream_id);");
        } else
        {
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_region_idx_") + GuidList()[i] +
                     " ON rocpd_region_" + GuidList()[i] + "(nid,pid,tid,start);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_kernel_dispatch_idx_") + GuidList()[i] +
                     " ON rocpd_kernel_dispatch_" + GuidList()[i] + "(nid,agent_id,queue_id,start);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_kernel_dispatch_stream_idx_") + GuidList()[i] +
                     " ON rocpd_kernel_dispatch_" + GuidList()[i] + "(nid,stream_id,start);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_memory_allocate_idx_") + GuidList()[i] +
                     " ON rocpd_memory_allocate_" + GuidList()[i] + "(nid,agent_id,queue_id,start);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_memory_allocate_stream_idx_") + GuidList()[i] +
                     " ON rocpd_memory_allocate_" + GuidList()[i] + "(nid,stream_id,start);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_memory_copy_idx_") + GuidList()[i] +
                     " ON rocpd_memory_copy_" + GuidList()[i] + "(nid,dst_agent_id,queue_id,start);");
            vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_memory_copy_stream_idx_") + GuidList()[i] +
                     " ON rocpd_memory_copy_" + GuidList()[i] + "(nid,stream_id,start);");

        }
        vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_region_event_idx_") + GuidList()[i] +
                    " ON rocpd_region_" + GuidList()[i] + "(event_id);");
        vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_sample_event_idx_") + GuidList()[i] +
                    " ON rocpd_sample_" + GuidList()[i] + "(event_id);");
        vec.push_back(std::string("CREATE INDEX IF NOT EXISTS rocpd_region_stack_idx_") + GuidList()[i] +
                    " ON rocpd_event_" + GuidList()[i] + "(stack_id);");
    }

    return ExecuteTransaction(vec);
}



rocprofvis_dm_result_t  RocprofDatabase::ReadTraceMetadata(Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    while (true)
    {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties, ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        std::string value;
        uint64_t process_id;

        ShowProgress(2, "Detect Nodes", kRPVDbBusy, future);
        ExecuteSQLQuery(future,
                        "SELECT name from sqlite_master where type = 'table' and name like 'rocpd_info_node_%';",
                        "rocpd_info_node_", (rocprofvis_dm_handle_t) &GuidList(),
                        &CallbackNodeEnumeration);
        if(GuidList().size() > 1)
        {
            spdlog::debug("Multi-node databases are not yet supported!");
            break;
        }

        ShowProgress(1, "Get version", kRPVDbBusy, future);
        std::string version_str;
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,"SELECT * FROM rocpd_metadata;", &CallbackParseMetadata)) break;
        m_query_factory.SetVersion(db_version.c_str());

        ShowProgress(5, "Indexing tables (once per database)", kRPVDbBusy, future);
        CreateIndexes();

        ShowProgress(2, "Load Nodes information", kRPVDbBusy, future);
        ExecuteSQLQuery(future,"SELECT * from rocpd_info_node;", "Node", (rocprofvis_dm_handle_t)CachedTables(), &CallbackCacheTable);
        
        ShowProgress(2, "Load Agents information", kRPVDbBusy, future);
        ExecuteSQLQuery(future,"SELECT * from rocpd_info_agent;", "Agent", (rocprofvis_dm_handle_t)CachedTables(), &CallbackCacheTable);

        ShowProgress(2, "Load Queues information", kRPVDbBusy, future);
        ExecuteSQLQuery(future,"SELECT * from rocpd_info_queue;", "Queue", (rocprofvis_dm_handle_t)CachedTables(), &CallbackCacheTable);

        ShowProgress(2, "Load Streams information", kRPVDbBusy, future);
        ExecuteSQLQuery(future,"SELECT * from rocpd_info_stream;", "Stream", (rocprofvis_dm_handle_t)CachedTables(), &CallbackCacheTable);

        ShowProgress(2, "Load Process information", kRPVDbBusy, future);
        ExecuteSQLQuery(future,"SELECT * from rocpd_info_process;", "Process", (rocprofvis_dm_handle_t)CachedTables(), &CallbackCacheTable);

        ShowProgress(2, "Load Thread information", kRPVDbBusy, future);
        ExecuteSQLQuery(future,"SELECT * from rocpd_info_thread;", "Thread", (rocprofvis_dm_handle_t)CachedTables(), &CallbackCacheTable);

        ShowProgress(2, "Load PMC information", kRPVDbBusy, future);
        ExecuteSQLQuery(future,"SELECT id, guid, nid, pid, agent_id, target_arch, COALESCE(event_code,0) as event_code, COALESCE(instance_id,0) as instance_id, name, symbol, description, long_description, component, units, value_type, block, expression, is_constant, is_derived, extdata from rocpd_info_pmc;", "PMC", (rocprofvis_dm_handle_t)CachedTables(), &CallbackCacheTable);

        CachedTables()->AddTableCell("PMC", -1, "name", "MALLOC");


        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,"select distinct id from rocpd_info_process;", &CallbackGetValue, process_id)) break;
                
        ShowProgress(5, "Adding HIP API tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,
            { 
                 m_query_factory.GetRocprofRegionTrackQuery(false),
                 "",
                 m_query_factory.GetRocprofRegionLevelQuery(false),
                 m_query_factory.GetRocprofRegionSliceQuery(false),
                 "",
                 m_query_factory.GetRocprofRegionTableQuery(false),
            },
            &CallBackAddTrack)) break;

        ShowProgress(5, "Adding HIP API Sample tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,
            { 
                m_query_factory.GetRocprofRegionTrackQuery(true),
                "",
                m_query_factory.GetRocprofRegionLevelQuery(true),
                m_query_factory.GetRocprofRegionSliceQuery(true),
                "",
                m_query_factory.GetRocprofRegionTableQuery(true),
            },
            &CallBackAddTrack)) break;

        ShowProgress(5, "Adding kernel dispatch tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,
            {
                m_query_factory.GetRocprofKernelDispatchTrackQuery(),
                m_query_factory.GetRocprofKernelDispatchTrackQueryForStream(),
                m_query_factory.GetRocprofKernelDispatchLevelQuery(),
                m_query_factory.GetRocprofKernelDispatchSliceQuery(),
                m_query_factory.GetRocprofKernelDispatchSliceQueryForStream(),
                m_query_factory.GetRocprofKernelDispatchTableQuery(),
            },
            &CallBackAddTrack)) break;

        ShowProgress(5, "Adding memory allocation tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, 
            { 
                m_query_factory.GetRocprofMemoryAllocTrackQuery(),
                m_query_factory.GetRocprofMemoryAllocTrackQueryForStream(),
                m_query_factory.GetRocprofMemoryAllocLevelQuery(),
                m_query_factory.GetRocprofMemoryAllocSliceQuery(),
                m_query_factory.GetRocprofMemoryAllocSliceQueryForStream(),
                m_query_factory.GetRocprofMemoryAllocTableQuery(),
            },
            &CallBackAddTrack)) break;
        /*
// This will not work if full track is not requested
// Comment out for now. Will need to fetch all data, then cut samples outside of time frame.
        ShowProgress(5, "Adding Memory allocation graph tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, 
                    "select DISTINCT nid as nodeId, agent_id as agentId, -1 as const, 1 as category from rocpd_memory_allocate;", 
                    "select 0 as op, start, sum(CASE WHEN type = 'FREE' THEN (select -size from rocpd_memory_allocate MA1 where MA.address == MA1.address limit 1) ELSE size END) over (ORDER BY start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as current, start as end, 0, 0, nid as nodeId, coalesce(agent_id, 0) as agentId, -1 as queue  from rocpd_memory_allocate MA ", 
                    &CallBackAddTrack)) break;
*/
   
        ShowProgress(5, "Adding memory copy tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, 
            {
                m_query_factory.GetRocprofMemoryCopyTrackQuery(),
                m_query_factory.GetRocprofMemoryCopyTrackQueryForStream(),
                m_query_factory.GetRocprofMemoryCopyLevelQuery(),
                m_query_factory.GetRocprofMemoryCopySliceQuery(),
                m_query_factory.GetRocprofMemoryCopySliceQueryForStream(),
                m_query_factory.GetRocprofMemoryCopyTableQuery(),
            },
            &CallBackAddTrack)) break;

        // PMC schema is not fully defined yet
        ShowProgress(5, "Adding performance counters tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,   
            {
                m_query_factory.GetRocprofPerformanceCountersTrackQuery(),
                "",
                m_query_factory.GetRocprofPerformanceCountersLevelQuery(),
                m_query_factory.GetRocprofPerformanceCountersSliceQuery(),
                "",
                m_query_factory.GetRocprofPerformanceCountersTableQuery(),
            },
            &CallBackAddTrack)) break;                    
        // PMC schema is not fully defined yet
        ShowProgress(5, "Adding performance smi counters tracks", kRPVDbBusy, future );
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,   
            {
                m_query_factory.GetRocprofSMIPerformanceCountersTrackQuery(),
                "",
                m_query_factory.GetRocprofSMIPerformanceCountersLevelQuery(),
                m_query_factory.GetRocprofSMIPerformanceCountersSliceQuery(),
                "",
                m_query_factory.GetRocprofSMIPerformanceCountersTableQuery(),

            },
            &CallBackAddTrack)) break;
                                                                    

        ShowProgress(20, "Loading strings", kRPVDbBusy, future );
        BindObject()->FuncAddString(BindObject()->trace_object, ""); // 0 index string
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, "SELECT string FROM rocpd_string ORDER BY id;", &CallBackAddString)) break;
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future,"SELECT COUNT(*) FROM rocpd_string;", &CallbackGetValue, m_symbols_offset)) break;
        if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, "SELECT display_name FROM rocpd_info_kernel_symbol;", &CallBackAddString)) break;

        ShowProgress(5, "Counting events",
                     kRPVDbBusy, future);
        TraceProperties()->events_count[kRocProfVisDmOperationLaunch]   = 0;
        TraceProperties()->events_count[kRocProfVisDmOperationDispatch] = 0;
        TraceProperties()->events_count[kRocProfVisDmOperationMemoryAllocate] = 0;
        TraceProperties()->events_count[kRocProfVisDmOperationMemoryCopy]     = 0;
        TraceProperties()->events_count[kRocProfVisDmOperationLaunchSample]   = 0;

        if(kRocProfVisDmResultSuccess !=
           ExecuteQueryForAllTracksAsync(
               kRocProfVisDmIncludePmcTracks | kRocProfVisDmIncludeStreamTracks,
               kRPVQueryLevel,
               "SELECT COUNT(*), op, ",
               "", &CallbackGetTrackRecordsCount,
               [](rocprofvis_dm_track_params_t* params) {}))
        {
            break;
        }

        std::vector<std::string> to_drop = Builder::OldLevelTables("launch");
        for (auto table : to_drop)
        {
            DropSQLTable(table.c_str());
        }
        to_drop = Builder::OldLevelTables("launch_sample");
        for(auto table : to_drop)
        {
            DropSQLTable(table.c_str());
        }
        to_drop = Builder::OldLevelTables("dispatch");
        for(auto table : to_drop)
        {
            DropSQLTable(table.c_str());
        }
        to_drop = Builder::OldLevelTables("mem_alloc");
        for(auto table : to_drop)
        {
            DropSQLTable(table.c_str());
        }
        to_drop = Builder::OldLevelTables("mem_copy");
        for(auto table : to_drop)
        {
            DropSQLTable(table.c_str());
        }


        if(SQLITE_OK != DetectTable(GetServiceConnection(), Builder::LevelTable("launch").c_str(), false) ||
           SQLITE_OK != DetectTable(GetServiceConnection(), Builder::LevelTable("launch_sample").c_str(), false) ||
           SQLITE_OK != DetectTable(GetServiceConnection(), Builder::LevelTable("dispatch").c_str(), false) ||
           SQLITE_OK != DetectTable(GetServiceConnection(), Builder::LevelTable("mem_alloc").c_str(), false) ||
           SQLITE_OK != DetectTable(GetServiceConnection(), Builder::LevelTable("mem_copy").c_str(), false))
        {
            m_event_levels[kRocProfVisDmOperationLaunch].reserve(
                TraceProperties()->events_count[kRocProfVisDmOperationLaunch]);
            m_event_levels[kRocProfVisDmOperationLaunchSample].reserve(
                TraceProperties()->events_count[kRocProfVisDmOperationLaunchSample]);
            m_event_levels[kRocProfVisDmOperationDispatch].reserve(
                TraceProperties()->events_count[kRocProfVisDmOperationDispatch]);
            m_event_levels[kRocProfVisDmOperationMemoryAllocate].reserve(
                TraceProperties()->events_count[kRocProfVisDmOperationMemoryAllocate]);
            m_event_levels[kRocProfVisDmOperationMemoryCopy].reserve(
                TraceProperties()->events_count[kRocProfVisDmOperationMemoryCopy]);

             ShowProgress(10, "Calculating event levels (once per database)", kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteQueryForAllTracksAsync(
                   kRocProfVisDmIncludeStreamTracks, 
                   kRPVQueryLevel, "SELECT *, ", (std::string(" ORDER BY ")+Builder::START_SERVICE_NAME).c_str(), &CalculateEventLevels,
                   [](rocprofvis_dm_track_params_t* params) {
                       params->m_active_events.clear();
                   }))
            {
                break;
            }

            for (int j = 0; j < kRocProfVisDmNumOperation; j++)
            {
                if(m_event_levels[j].size() > 0)
                {
                    std::sort(m_event_levels[j].begin(),
                                m_event_levels[j].end(),
                                [](const rocprofvis_db_event_level_t& a,
                                    const rocprofvis_db_event_level_t& b) {
                                    return a.id < b.id;
                                });
                }
            }

            SQLInsertParams params[] = { { "eid", "INTEGER PRIMARY KEY" }, { "level", "INTEGER" } , { "level_for_stream", "INTEGER" }};
            CreateSQLTable(
                Builder::LevelTable("launch").c_str(), params, 3,
                m_event_levels[kRocProfVisDmOperationLaunch].size(),
                [&](sqlite3_stmt* stmt, int index) {
                    sqlite3_bind_int64(
                        stmt, 1, m_event_levels[kRocProfVisDmOperationLaunch][index].id);
                    sqlite3_bind_int(
                        stmt, 2,
                        m_event_levels[kRocProfVisDmOperationLaunch][index].level_for_queue);
                    sqlite3_bind_int(
                        stmt, 3, 0);
                });
            m_event_levels[kRocProfVisDmOperationLaunch].clear();
            m_event_levels_id_to_index[kRocProfVisDmOperationLaunch].clear();
            CreateSQLTable(
                Builder::LevelTable("launch_sample").c_str(), params, 3,
                m_event_levels[kRocProfVisDmOperationLaunchSample].size(),
                [&](sqlite3_stmt* stmt, int index) {
                    sqlite3_bind_int64(
                        stmt, 1, m_event_levels[kRocProfVisDmOperationLaunchSample][index].id);
                    sqlite3_bind_int(stmt, 2,
                                     m_event_levels[kRocProfVisDmOperationLaunchSample][index]
                                         .level_for_queue);
                    sqlite3_bind_int(stmt, 3, 0);
                });
            m_event_levels[kRocProfVisDmOperationLaunchSample].clear();
            m_event_levels_id_to_index[kRocProfVisDmOperationLaunchSample].clear();
            CreateSQLTable(
                Builder::LevelTable("dispatch").c_str(), params, 3,
                m_event_levels[kRocProfVisDmOperationDispatch].size(),
                [&](sqlite3_stmt* stmt, int index) {
                    sqlite3_bind_int64(
                        stmt, 1,
                        m_event_levels[kRocProfVisDmOperationDispatch][index].id);
                    sqlite3_bind_int(
                        stmt, 2,
                        m_event_levels[kRocProfVisDmOperationDispatch][index].level_for_queue);
                    sqlite3_bind_int(
                        stmt, 3,
                        m_event_levels[kRocProfVisDmOperationDispatch][index].level_for_stream);
                });
            m_event_levels[kRocProfVisDmOperationDispatch].clear();
            m_event_levels_id_to_index[kRocProfVisDmOperationDispatch].clear();
            CreateSQLTable(
                Builder::LevelTable("mem_alloc").c_str(), params, 3,
                m_event_levels[kRocProfVisDmOperationMemoryAllocate].size(),
                [&](sqlite3_stmt* stmt, int index) {
                    sqlite3_bind_int64(
                        stmt, 1,
                        m_event_levels[kRocProfVisDmOperationMemoryAllocate][index].id);
                    sqlite3_bind_int(
                        stmt, 2,
                        m_event_levels[kRocProfVisDmOperationMemoryAllocate][index].level_for_queue);
                    sqlite3_bind_int(
                        stmt, 3,
                        m_event_levels[kRocProfVisDmOperationMemoryAllocate][index].level_for_stream);
                });
            m_event_levels[kRocProfVisDmOperationMemoryAllocate].clear();
            m_event_levels_id_to_index[kRocProfVisDmOperationMemoryAllocate].clear();
            CreateSQLTable(
                Builder::LevelTable("mem_copy").c_str(), params, 3,
                m_event_levels[kRocProfVisDmOperationMemoryCopy].size(),
                [&](sqlite3_stmt* stmt, int index) {
                    sqlite3_bind_int64(
                        stmt, 1,
                        m_event_levels[kRocProfVisDmOperationMemoryCopy][index].id);
                    sqlite3_bind_int(
                        stmt, 2,
                        m_event_levels[kRocProfVisDmOperationMemoryCopy][index].level_for_queue);
                    sqlite3_bind_int(
                        stmt, 3,
                        m_event_levels[kRocProfVisDmOperationMemoryCopy][index].level_for_stream);
                });
            m_event_levels[kRocProfVisDmOperationMemoryCopy].clear();
            m_event_levels_id_to_index[kRocProfVisDmOperationMemoryCopy].clear();
        }

        ShowProgress(5, "Collecting track properties",
                     kRPVDbBusy, future);

        TraceProperties()->start_time = UINT64_MAX;
        TraceProperties()->end_time = 0;

        if(kRocProfVisDmResultSuccess !=
           ExecuteQueryForAllTracksAsync(
               kRocProfVisDmIncludePmcTracks | kRocProfVisDmIncludeStreamTracks , kRPVQuerySliceByTrackSliceQuery,
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

        uint64_t bucket_size = (trace_length + desired_bins ) / desired_bins;
        if(bucket_size == 0) bucket_size = 1;  

        uint64_t bucket_count = (trace_length + bucket_size ) / bucket_size;
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



rocprofvis_dm_result_t  RocprofDatabase::ReadFlowTraceInfo(
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
        std::string query;
        if(event_id.bitfield.event_op == kRocProfVisDmOperationLaunch ||
           event_id.bitfield.event_op == kRocProfVisDmOperationLaunchSample)
        {
            query = m_query_factory.GetRocprofDataFlowQueryForRegionEvent(event_id.bitfield.event_id);
            ShowProgress(0, query.c_str(),kRPVDbBusy, future);
            if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), flowtrace, &CallbackAddFlowTrace)) break;
        } else
        if (event_id.bitfield.event_op == kRocProfVisDmOperationDispatch)
        {
            query = m_query_factory.GetRocprofDataFlowQueryForKernelDispatchEvent(event_id.bitfield.event_id);
            ShowProgress(0, query.c_str(),kRPVDbBusy, future);
            if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), flowtrace, &CallbackAddFlowTrace)) break;
        } else
        if (event_id.bitfield.event_op == kRocProfVisDmOperationMemoryCopy)
        {
            query = m_query_factory.GetRocprofDataFlowQueryForMemoryCopyEvent(event_id.bitfield.event_id);
            ShowProgress(0, query.c_str(), kRPVDbBusy, future);
            if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), flowtrace, &CallbackAddFlowTrace)) break;
        }
        else
        if (event_id.bitfield.event_op == kRocProfVisDmOperationMemoryAllocate)
        {
            query = m_query_factory.GetRocprofDataFlowQueryForMemoryAllocEvent(event_id.bitfield.event_id);
            ShowProgress(0, query.c_str(), kRPVDbBusy, future);
            if (kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), flowtrace, &CallbackAddFlowTrace)) break;
        }
        else
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

rocprofvis_dm_result_t RocprofDatabase::SaveTrimmedData(rocprofvis_dm_timestamp_t start,
    rocprofvis_dm_timestamp_t end,
                                 rocprofvis_dm_charptr_t new_db_path, Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(new_db_path, "New DB path cannot be NULL.",
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    rocprofvis_dm_result_t result = kRocProfVisDmResultInvalidParameter;

    std::string query;
    rocprofvis_db_sqlite_trim_parameters trim_tables;
    rocprofvis_db_sqlite_trim_parameters trim_views;

    Future* internal_future = new Future(nullptr);

    {
        RocprofDatabase rpDb(new_db_path);
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

                        query = table.second;
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
                            if(strstr(table.first.c_str(), "rocpd_kernel_dispatch") ||
                               strstr(table.first.c_str(), "rocpd_memory_allocate") ||
                               strstr(table.first.c_str(), "rocpd_memory_copy") ||
                               strstr(table.first.c_str(), "rocpd_region"))
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
                            else if(strstr(table.first.c_str(), "rocpd_sample"))
                            {
                                query = "INSERT INTO ";
                                query += table.first;
                                query += " SELECT * FROM oldDb.";
                                query += table.first;
                                query += " WHERE timestamp < ";
                                query += std::to_string(end);
                                query += " AND timestamp > ";
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

    if (result == kRocProfVisDmResultSuccess)
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

rocprofvis_dm_result_t RocprofDatabase::BuildTableStringIdFilter( rocprofvis_dm_num_string_table_filters_t num_string_table_filters, 
    rocprofvis_dm_string_table_filters_t string_table_filters, table_string_id_filter_map_t& filter)
{
    rocprofvis_dm_result_t result = kRocProfVisDmResultSuccess;
    if(num_string_table_filters > 0)
    {
        result = kRocProfVisDmResultUnknownError;
        ROCPROFVIS_ASSERT_RETURN(BindObject()->trace_object, kRocProfVisDmResultInvalidParameter);
        std::vector<rocprofvis_dm_index_t> string_indices;
        result = ((Trace*)BindObject()->trace_object)->GetStringIndicesWithSubstring(num_string_table_filters, string_table_filters, string_indices);
        ROCPROFVIS_ASSERT_RETURN(result == kRocProfVisDmResultSuccess, result);
        std::string string_ids;
        std::string kernel_ids;
        for(const rocprofvis_dm_index_t& index : string_indices)
        {
            rocprofvis_dm_id_t string_id;
            result = StringIndexToId(index, string_id);
            ROCPROFVIS_ASSERT_RETURN(result == kRocProfVisDmResultSuccess, result);
            if(index > m_symbols_offset)
            {
                kernel_ids += kernel_ids.empty() ? std::to_string(string_id) : ", " + std::to_string(string_id);
            }
            else
            {
                string_ids += string_ids.empty() ? std::to_string(string_id) : ", " + std::to_string(string_id);
            }
        }
        if(!string_ids.empty())
        {
            filter[kRocProfVisDmOperationLaunch] = " SAMPLE.id IS NULL AND R.name_id IN (" + string_ids;
            filter[kRocProfVisDmOperationDispatch] = " (K.region_name_id IN (" + string_ids;
            filter[kRocProfVisDmOperationLaunchSample] = "R.name_id IN (" + string_ids;
            if(!kernel_ids.empty())
            {
                filter[kRocProfVisDmOperationDispatch] += ") OR K.kernel_id IN (" + kernel_ids;
            }
            filter[kRocProfVisDmOperationDispatch] += ") ";
        }
        else if(!kernel_ids.empty())
        {
            filter[kRocProfVisDmOperationDispatch] = " K.kernel_id IN (" + kernel_ids;
        }
    }
    return result;
}

rocprofvis_dm_string_t RocprofDatabase::GetEventOperationQuery(const rocprofvis_dm_event_operation_t operation)
{
    switch(operation)
    {
        case kRocProfVisDmOperationLaunch:
        {
            return m_query_factory.GetRocprofRegionTableQuery(false);
        }
        case kRocProfVisDmOperationDispatch:
        {
            return m_query_factory.GetRocprofKernelDispatchTableQuery();
        }
        case kRocProfVisDmOperationMemoryAllocate:
        {
            return m_query_factory.GetRocprofMemoryAllocTableQuery();
        }
        case kRocProfVisDmOperationMemoryCopy:
        {
            return m_query_factory.GetRocprofMemoryCopyTableQuery();
        }
        case kRocProfVisDmOperationLaunchSample:
        {
            return m_query_factory.GetRocprofRegionTableQuery(true);
        }
        default:
        {
            return "";
        }
    }
}

rocprofvis_dm_result_t RocprofDatabase::StringIndexToId(rocprofvis_dm_index_t index, rocprofvis_dm_id_t& id)
{
    id = (index >= m_symbols_offset) ? index - m_symbols_offset : index;
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t
RocprofDatabase::BuildTableSummaryClause(bool sample_query,
                                   rocprofvis_dm_string_t& select,
                                   rocprofvis_dm_string_t& group_by)
{
    if(sample_query)
    {
        select = Builder::NODE_ID_SERVICE_NAME;
        select += ", ";
        select += Builder::COUNTER_ID_SERVICE_NAME;
        select += ", AVG(";
        select += Builder::COUNTER_VALUE_SERVICE_NAME;
        select += ") AS avg_value, MIN(";
        select += Builder::COUNTER_VALUE_SERVICE_NAME;
        select += ") AS min_value, MAX(";
        select += Builder::COUNTER_VALUE_SERVICE_NAME;
        select += ") AS max_value";
        group_by = Builder::NODE_ID_SERVICE_NAME;
        group_by += ", ";
        group_by += Builder::COUNTER_ID_SERVICE_NAME;
    }
    else
    {
        select = "name, COUNT(*) AS num_invocations, AVG(duration) AS avg_duration, MIN(duration) AS min_duration, MAX(duration) AS max_duration, SUM(duration) AS total_duration";
        group_by = "name";
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  RocprofDatabase::ReadStackTraceInfo(
        rocprofvis_dm_event_id_t event_id,
        Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    while(true)
    {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties,
                                    ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties->metadata_loaded,
                                    ERROR_METADATA_IS_NOT_LOADED);
        rocprofvis_dm_stacktrace_t stacktrace =
            BindObject()->FuncAddStackTrace(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(stacktrace, ERROR_EXT_DATA_CANNOT_BE_NULL);
        std::stringstream query;
        if(event_id.bitfield.event_op == kRocProfVisDmOperationLaunch ||
           event_id.bitfield.event_op == kRocProfVisDmOperationLaunchSample)
        {
            query << "select call_stack, line_info from regions where id == ";
            query << event_id.bitfield.event_id << ";";
            ShowProgress(0, query.str().c_str(), kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.str().c_str(),
                                                             stacktrace,
                                                             &CallbackAddStackTrace))
            {
                break;
            }

        }
        else
        {
            ShowProgress(0, "Stack trace is not available for specified operation type!",
                         kRPVDbError, future);
            return future->SetPromise(kRocProfVisDmResultInvalidParameter);
        }

        ShowProgress(100, "Extended data successfully loaded!", kRPVDbSuccess, future);
        return future->SetPromise(kRocProfVisDmResultSuccess);
    }
    ShowProgress(0, "Extended data  not loaded!", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultTimeout
                                                    : kRocProfVisDmResultDbAccessFailed);
}


rocprofvis_dm_result_t  RocprofDatabase::ReadExtEventInfo(
            rocprofvis_dm_event_id_t event_id, 
            Future* future){

    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    while (true)
    {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties, ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);
        rocprofvis_dm_extdata_t extdata = BindObject()->FuncAddExtData(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(extdata, ERROR_EXT_DATA_CANNOT_BE_NULL);
        std::string query;
        if(event_id.bitfield.event_op == kRocProfVisDmOperationLaunch ||
           event_id.bitfield.event_op == kRocProfVisDmOperationLaunchSample)
        {
            query = "select * from regions where id == ";
            query += std::to_string(event_id.bitfield.event_id);
            ShowProgress(0, query.c_str(),kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.c_str(), "Properties", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;
            query = m_query_factory.GetRocprofEssentialInfoQueryForRegionEvent(event_id.bitfield.event_id, event_id.bitfield.event_op == kRocProfVisDmOperationLaunchSample);
            if(kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), extdata, &CallbackAddEssentialInfo)) break;

        } else
        if (event_id.bitfield.event_op == kRocProfVisDmOperationDispatch)
        {
            query = "select * from kernels where id == ";
            query += std::to_string(event_id.bitfield.event_id); 
            ShowProgress(0, query.c_str(),kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.c_str(), "Properties", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;
            query = m_query_factory.GetRocprofEssentialInfoQueryForKernelDispatchEvent(event_id.bitfield.event_id);
            if(kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), extdata, &CallbackAddEssentialInfo)) break;
        } else   
        if (event_id.bitfield.event_op == kRocProfVisDmOperationMemoryAllocate)
        {
            query = "select * from memory_allocations where id == ";
            query += std::to_string(event_id.bitfield.event_id);
            ShowProgress(0, query.c_str(), kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.c_str(), "Properties", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;
            query = m_query_factory.GetRocprofEssentialInfoQueryForMemoryAllocEvent(event_id.bitfield.event_id);
            if(kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), extdata, &CallbackAddEssentialInfo)) break;
        } else
        if (event_id.bitfield.event_op == kRocProfVisDmOperationMemoryCopy)
        {
            query = "select * from memory_copies where id == ";
            query += std::to_string(event_id.bitfield.event_id);
            ShowProgress(0, query.c_str(), kRPVDbBusy, future);
            if(kRocProfVisDmResultSuccess !=
               ExecuteSQLQuery(
                   future, query.c_str(), "Properties", extdata,
                   (rocprofvis_dm_event_operation_t) event_id.bitfield.event_op,
                   &CallbackAddExtInfo))
                break;
            query = m_query_factory.GetRocprofEssentialInfoQueryForMemoryCopyEvent(event_id.bitfield.event_id);
            if(kRocProfVisDmResultSuccess != ExecuteSQLQuery(future, query.c_str(), extdata, &CallbackAddEssentialInfo)) break;
        } else
        {
            ShowProgress(0, "Extended data not available for specified operation type!", kRPVDbError, future );
            return future->SetPromise(kRocProfVisDmResultInvalidParameter);
        }
        ShowProgress(100, "Extended data successfully loaded!",kRPVDbSuccess, future);
        return future->SetPromise(kRocProfVisDmResultSuccess);
    }
    ShowProgress(0, "Extended data  not loaded!", kRPVDbError, future );
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort : kRocProfVisDmResultDbAccessFailed);    
}

}  // namespace DataModel
}  // namespace RocProfVis
