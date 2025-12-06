// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_db_sqlite.h"
#include "internal_types.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <fstream>

namespace RocProfVis
{
namespace DataModel
{

int SqliteDatabase::CallbackGetValue(void* data, int argc, sqlite3_stmt* stmt, char** azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(argc==1, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void*  func = (void*)&CallbackGetValue;
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    SqliteDatabase* db = (SqliteDatabase*) callback_params->db;
    std::string * string_ptr = (rocprofvis_dm_string_t*)callback_params->handle;
    ROCPROFVIS_ASSERT_MSG_RETURN(string_ptr, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    *string_ptr = db->Sqlite3ColumnText(func, stmt, azColName, 0);
    return 0;
} 

bool
SqliteDatabase::isServiceColumn(char* name)
{
    static const std::vector<std::string> service_columns = {
        Builder::SPACESAVER_SERVICE_NAME,
        Builder::AGENT_ID_SERVICE_NAME,
        Builder::QUEUE_ID_SERVICE_NAME,
        Builder::STREAM_ID_SERVICE_NAME,
        Builder::OPERATION_SERVICE_NAME,
        Builder::PROCESS_ID_SERVICE_NAME,
        Builder::THREAD_ID_SERVICE_NAME
    };
    for (std::string service_column : service_columns)
    {
        if(service_column == name) return true;
    }
    return false;
}


uint64_t
SqliteDatabase::GetNullExceptionInt(void* func, char* column) {
    spdlog::debug("Column {} value is NULL!", column);
    const rocprofvis_null_data_exceptions_int* exceptions_map =
        GetNullDataExceptionsInt();
    if(exceptions_map != nullptr)
    {
        auto fcit = exceptions_map->find(func);
        if(fcit != exceptions_map->end())
        {
            auto it = fcit->second.find(column);
            if(it != fcit->second.end())
            {
                return it->second;
                spdlog::debug("Column {} value is NULL, replace with {}", column, it->second);
            }
        }
    }
    spdlog::debug("Column {} value is NULL, replace with 0", column);
    return 0;
}

char*
SqliteDatabase::GetNullExceptionString(void* func, char* column) {
    const rocprofvis_null_data_exceptions_string* exceptions_map =
        GetNullDataExceptionsString();
    if(exceptions_map != nullptr)
    {
        auto fcit = exceptions_map->find(func);
        if(fcit != exceptions_map->end())
        {
            auto it = fcit->second.find(column);
            if(it != fcit->second.end())
            {
                spdlog::debug("Column {} value is NULL, replace with {}", column, it->second.c_str());
                return (char*) it->second.c_str();
            }
        }
    }
    spdlog::debug("Column {} value is NULL, replace with empty string", column);
    return "";
}

bool
SqliteDatabase::NullExceptionSkip(void* func, char* column)
{
    const rocprofvis_null_data_exceptions_skip* exceptions_map =
        GetNullDataExceptionsSkip();
    if(exceptions_map != nullptr)
    {
        auto fcit = exceptions_map->find(func);
        if(fcit != exceptions_map->end())
        {
            auto it = fcit->second.find(column);
            if(it != fcit->second.end())
            {
                spdlog::debug("Column {} value is NULL, skip column", column);
                return true;
            }
        }
    }
    return false;
}

char*
SqliteDatabase::Sqlite3ColumnText(void* func, sqlite3_stmt* stmt, char** azColName, int index) {

    if(sqlite3_column_type(stmt, index) == SQLITE_NULL)
    {
        return GetNullExceptionString(func, azColName[index]);
    }
    else
    {
        return (char*) sqlite3_column_text(stmt, index);
    }
}

int
SqliteDatabase::Sqlite3ColumnInt(void* func, sqlite3_stmt* stmt, char** azColName, int index) {
    if(sqlite3_column_type(stmt, index) == SQLITE_NULL)
    {
        return GetNullExceptionInt(func, azColName[index]);
    }
    else
    {
        return sqlite3_column_int(stmt, index);
    }
}
int64_t
SqliteDatabase::Sqlite3ColumnInt64(void* func, sqlite3_stmt* stmt, char** azColName, int index) {
    if(sqlite3_column_type(stmt, index) == SQLITE_NULL)
    {
        return GetNullExceptionInt(func, azColName[index]);
    }
    else
    {
        return sqlite3_column_int64(stmt, index);
    }
}
double
SqliteDatabase::Sqlite3ColumnDouble(void* func, sqlite3_stmt* stmt, char** azColName, int index) {
    if(sqlite3_column_type(stmt, index) == SQLITE_NULL)
    {
        return GetNullExceptionInt(func, azColName[index]);
    }
    else
    {
        return sqlite3_column_double(stmt, index);
    }
}

void
SqliteDatabase::FindTrackIDs(
    SqliteDatabase* db, rocprofvis_db_sqlite_track_service_data_t& service_data,
    int& trackId, int & streamTrackId)
{ 
    trackId = -1;
    streamTrackId = -1;
    rocprofvis_dm_process_identifiers_t process;
    for(int i = 0; i < NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS; i++)
    {
        process.is_numeric[i] = true;
    }
    process.category = service_data.category;
    if(service_data.category == kRocProfVisDmKernelDispatchTrack ||
            service_data.category == kRocProfVisDmMemoryAllocationTrack ||
            service_data.category == kRocProfVisDmMemoryCopyTrack ||
            service_data.category == kRocProfVisDmRegionTrack ||
            service_data.category == kRocProfVisDmRegionMainTrack ||
            service_data.category == kRocProfVisDmRegionSampleTrack)
    {
        process.id[TRACK_ID_NODE]         = service_data.nid;
        process.id[TRACK_ID_PID_OR_AGENT] = service_data.process;
        process.id[TRACK_ID_TID_OR_QUEUE] = service_data.thread;
        rocprofvis_dm_track_params_it it = db->FindTrack(process);
        if(it != db->TrackPropertiesEnd())
        {
            trackId = it->get()->track_id;
        }
        process.category            = kRocProfVisDmStreamTrack;
        process.id[TRACK_ID_STREAM] = service_data.stream_id;
        process.id[TRACK_ID_QUEUE]  = -1;
        it                          = db->FindTrack(process);
        if(it != db->TrackPropertiesEnd())
        {
            streamTrackId = it->get()->track_id;
        }
    }
    else if(service_data.category == kRocProfVisDmPmcTrack)
    {
        process.id[TRACK_ID_NODE]    = service_data.nid;
        process.id[TRACK_ID_AGENT]   = service_data.process;
        process.id[TRACK_ID_COUNTER] = service_data.thread;
        if(service_data.monitor_type.length() > 0)
        {
            process.is_numeric[TRACK_ID_COUNTER] = false;
            process.name[TRACK_ID_COUNTER]       = service_data.monitor_type;
        }

        rocprofvis_dm_track_params_it it = db->FindTrack(process);
        if(it != db->TrackPropertiesEnd())
        {
            trackId = it->get()->track_id;
        }
    }
}

rocprofvis_dm_track_category_t
SqliteDatabase::TranslateOperationToTrackCategory(rocprofvis_dm_event_operation_t op) {
    switch (op)
    {
        case kRocProfVisDmOperationLaunch: return GetRegionTrackCategory();
        case kRocProfVisDmOperationLaunchSample: return kRocProfVisDmRegionSampleTrack;
        case kRocProfVisDmOperationDispatch: return kRocProfVisDmKernelDispatchTrack;
        case kRocProfVisDmOperationMemoryAllocate: return kRocProfVisDmMemoryAllocationTrack;
        case kRocProfVisDmOperationMemoryCopy: return kRocProfVisDmMemoryCopyTrack;
        case kRocProfVisDmOperationNoOp: return kRocProfVisDmPmcTrack;

    }
    return kRocProfVisDmNotATrack;
}

const rocprofvis_dm_track_search_id_t
SqliteDatabase::GetTrackSearchId(rocprofvis_dm_track_category_t category)
{
    switch (category)
    {
        case kRocProfVisDmPmcTrack:
            return kRPVTrackSearchIdCounters;
        case kRocProfVisDmRegionTrack:
        case kRocProfVisDmRegionMainTrack:
            return kRPVTrackSearchIdThreads;
        case kRocProfVisDmRegionSampleTrack:
            return kRPVTrackSearchIdThreadSamples;
        case kRocProfVisDmKernelDispatchTrack: 
            return kRPVTrackSearchIdDispatches;
        case kRocProfVisDmMemoryAllocationTrack: 
            return kRPVTrackSearchIdMemAllocs;
        case kRocProfVisDmMemoryCopyTrack: 
            return kRPVTrackSearchIdMemCopies;
        case kRocProfVisDmStreamTrack: 
            return kRPVTrackSearchIdStreams;
            
    }
    return kRPVTrackSearchIdUnknown;
}

void
SqliteDatabase::CollectTrackServiceData(
    SqliteDatabase* db,
    sqlite3_stmt* stmt, int column_index, char** azColName,
                        rocprofvis_db_sqlite_track_service_data_t& service_data)
{
    void* func = (void*)&CollectTrackServiceData;
    std::string column_name = azColName[column_index];
    if(column_name == Builder::OPERATION_SERVICE_NAME)
    {

        service_data.op = (rocprofvis_dm_event_operation_t)db->Sqlite3ColumnInt(func, stmt, azColName, column_index);
        service_data.category = db->TranslateOperationToTrackCategory(service_data.op);
    }
    else if(column_name == Builder::NODE_ID_SERVICE_NAME)
    {
        service_data.nid = db->Sqlite3ColumnInt64(func, stmt, azColName, column_index);
    }
    else if(column_name == Builder::AGENT_ID_SERVICE_NAME)
    {
        service_data.process = db->Sqlite3ColumnInt(func, stmt, azColName, column_index);
    }
    else if(column_name == Builder::QUEUE_ID_SERVICE_NAME)
    {
        service_data.thread = db->Sqlite3ColumnInt(func, stmt, azColName, column_index);
    }
    else if(column_name == Builder::STREAM_ID_SERVICE_NAME)
    {
        service_data.stream_id = db->Sqlite3ColumnInt(func, stmt, azColName, column_index);
    }
    else if(column_name == Builder::PROCESS_ID_SERVICE_NAME)
    {
        service_data.process = db->Sqlite3ColumnInt(func, stmt, azColName, column_index);
    }
    else if(column_name == Builder::THREAD_ID_SERVICE_NAME)
    {
        service_data.thread = db->Sqlite3ColumnInt(func, stmt, azColName, column_index);
    }
    else if(column_name == Builder::COUNTER_ID_SERVICE_NAME)
    {
        service_data.thread = db->Sqlite3ColumnInt(func, stmt, azColName, column_index);
    }
    else if(column_name == Builder::COUNTER_NAME_SERVICE_NAME)
    {
        service_data.monitor_type =
            db->Sqlite3ColumnText(func, stmt, azColName, column_index);
    }
}

int SqliteDatabase::CallbackRunQuery(void *data, int argc, sqlite3_stmt* stmt, char **azColName){
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    rocprofvis_db_sqlite_callback_parameters* callback_params = (rocprofvis_db_sqlite_callback_parameters*)data;
    SqliteDatabase* db = (SqliteDatabase*)callback_params->db;
    void* func = (void*)&CallbackRunQuery;
    if (callback_params->future->Interrupted()) return 1;
    rocprofvis_db_sqlite_track_service_data_t service_data{};
    bool  is_query_for_table_view = false;
    uint32_t op_pos = 0;
    int arg0 = 0;
    std::string  column_text;
    std::string  column = azColName[0];
    rocprofvis_dm_table_row_t row =
        db->BindObject()->FuncAddTableRow(callback_params->handle);
    ROCPROFVIS_ASSERT_MSG_RETURN(row, ERROR_TABLE_ROW_CANNOT_BE_NULL, 1);
    if(column == "NumRecords")
    {
        if(kRocProfVisDmResultSuccess !=
           db->BindObject()->FuncAddTableColumn(callback_params->handle, azColName[0]))
            return 1;
        column_text = db->Sqlite3ColumnText(func, stmt, azColName, 0);
        if(kRocProfVisDmResultSuccess !=
           db->BindObject()->FuncAddTableRowCell(row, column_text.c_str()))
            return 1;
        arg0 = 1;
    }
    for(int i = arg0; i < argc; i++)
    {
        column = azColName[i];
        
        if(column == Builder::OPERATION_SERVICE_NAME)
        {
            op_pos                  = i;
            is_query_for_table_view = true;
            break;
        }
    }
    uint64_t blanks_mask = is_query_for_table_view ? db->GetBlanksMaskForQuery(callback_params->query[0]) : 0;
    if(0 == callback_params->future->GetProcessedRowsCount())
    {
        for (int i=arg0; i < argc; i++)
        {
            if((blanks_mask & (uint64_t) 1 << (i-op_pos)) != 0) 
                continue;
            std::string column = azColName[i];
            CollectTrackServiceData(db, stmt, i, azColName, service_data);
            if(db->isServiceColumn(azColName[i]))
                continue;
            if (kRocProfVisDmResultSuccess != db->BindObject()->FuncAddTableColumn(callback_params->handle,azColName[i])) return 1;
        }
        if(is_query_for_table_view)
        {
            if(service_data.category != kRocProfVisDmNotATrack)
            {
                if(kRocProfVisDmResultSuccess !=
                   db->BindObject()->FuncAddTableColumn(callback_params->handle,
                                                        Builder::TRACK_ID_PUBLIC_NAME))
                    return 1;
                if(service_data.category == kRocProfVisDmKernelDispatchTrack ||
                   service_data.category == kRocProfVisDmMemoryAllocationTrack ||
                   service_data.category == kRocProfVisDmMemoryCopyTrack || 
                    service_data.category == kRocProfVisDmRegionTrack || 
                    service_data.category == kRocProfVisDmRegionMainTrack ||
                   service_data.category == kRocProfVisDmRegionSampleTrack)
                {
                    if(kRocProfVisDmResultSuccess !=
                       db->BindObject()->FuncAddTableColumn(
                           callback_params->handle, Builder::STREAM_TRACK_ID_PUBLIC_NAME))
                        return 1;
                }
            }
        }
    }


    uint64_t op = 0;

    service_data.category = kRocProfVisDmNotATrack;
    service_data.op       = kRocProfVisDmOperationNoOp;
    for (int i=arg0; i < argc; i++)
    {
        if((blanks_mask & (uint64_t) 1 << (i-op_pos)) != 0) 
            continue;
        
        std::string column = azColName[i];        
        if(is_query_for_table_view)
        {
            CollectTrackServiceData(db, stmt, i, azColName, service_data);
            if(db->isServiceColumn(azColName[i])) 
                continue;
            if(column == "id")
            {
                uint64_t id = db->Sqlite3ColumnInt64(func, stmt, azColName, i);
                id |= (uint64_t)service_data.op << 60;
                column_text = std::to_string(id);
            }
            else
            {
                column_text = db->Sqlite3ColumnText(func, stmt, azColName, i);
            }
        }
        else
        {
            column_text = db->Sqlite3ColumnText(func, stmt, azColName, i);
        }

        if (kRocProfVisDmResultSuccess != db->BindObject()->FuncAddTableRowCell(row, column_text.c_str())) return 1;
    }

    if(is_query_for_table_view)
    {
        int trackId       = -1;
        int streamTrackId = -1;

        FindTrackIDs(db, service_data, trackId, streamTrackId);

        if(kRocProfVisDmResultSuccess !=
           db->BindObject()->FuncAddTableRowCell(row, std::to_string(trackId).c_str()))
            return 1;
        if(service_data.category != kRocProfVisDmPmcTrack)
            {
                if(kRocProfVisDmResultSuccess !=
                   db->BindObject()->FuncAddTableRowCell(
                       row, std::to_string(streamTrackId).c_str()))
                    return 1;
            }
    }
    callback_params->future->CountThisRow();
    return 0;
}

int SqliteDatabase::CallbackTableQueryToCSV(void* data, int argc, sqlite3_stmt* stmt, char** azColName)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void *func = (void*)&CallbackTableQueryToCSV;
    rocprofvis_db_sqlite_callback_parameters* callback_params =
        (rocprofvis_db_sqlite_callback_parameters*) data;
    std::ofstream* file = (std::ofstream*)callback_params->handle;
    ROCPROFVIS_ASSERT_RETURN(file, 1);
    SqliteDatabase* db = (SqliteDatabase*) callback_params->db;
    if(callback_params->future->Interrupted()) return SQLITE_ABORT;
    const std::array<uint64_t, kRPVTableQueryColumnMaskCount> column_masks = db->GetColumnMasksForQuery(callback_params->query[0]);
    const uint64_t skip_mask = column_masks[kRPVTableQueryColumnMaskBlank] | column_masks[kRPVTableQueryColumnMaskService];
    const uint64_t& timestamp_mask = column_masks[kRPVTableQueryColumnMaskTimestamp];

    bool delim = false;
    if(callback_params->future->GetProcessedRowsCount() == 0)
    {
        for(int i = 0; i < argc; i++)
        {
            if((skip_mask & (uint64_t) 1 << i) != 0) 
            {
                continue;
            }
            if(delim)
            {
                *file << ',';
                delim = false;
            }
            *file << azColName[i];
            delim = true;
        }
        *file << "\n";
    }
    else
    {
        *file << "\n";
    }

    int startCol = 0;
    switch(argc)
    {
        case rocprofvis_db_sqlite_table_query_format::NUM_PARAMS:
        case rocprofvis_db_sqlite_rocpd_table_query_format::NUM_PARAMS:
        {
            *file << ((uint64_t)db->Sqlite3ColumnInt(func, stmt, azColName, 0) << 60 | (uint64_t)db->Sqlite3ColumnInt64(func, stmt, azColName, 1));
            delim = true;
            startCol = 2;
            break;
        }
        case rocprofvis_db_sqlite_sample_table_query_format::NUM_PARAMS:
        {
            startCol = 1;
            delim = false;
            break;
        }
        default:
        {
            ROCPROFVIS_ASSERT_MSG_RETURN(false, ERROR_DATABASE_QUERY_PARAMETERS_MISMATCH, 1);
            break;
        }
    }

    for(int i = startCol ; i < argc; i++)
    {
        if((skip_mask & (uint64_t) 1 << i) != 0) 
        {
            continue;
        }
        if(delim)
        {
            *file << ',';
            delim = false;
        }
        switch(sqlite3_column_type(stmt, i))
        {
            case SQLITE_NULL:
            {
                *file << db->GetNullExceptionString(func, azColName[i]);
                break;
            }
            case SQLITE_TEXT:
            {
                *file << '"';
                *file << sqlite3_column_text(stmt, i);
                *file << '"';
                break;
            }
            case SQLITE_INTEGER:
            {
                if((timestamp_mask & (uint64_t) 1 << i) != 0)
                {
                    *file << sqlite3_column_int64(stmt, i) - db->TraceProperties()->start_time;
                }
                else
                {
                    *file << sqlite3_column_text(stmt, i);
                }
                break;
            }
            default:
            {
                *file << sqlite3_column_text(stmt, i);
                break;
            }
        }
        delim = true;
    }

    callback_params->future->CountThisRow();
    return 0;
}

int
SqliteDatabase::CallbackQueryToCSV(void* data, int argc, sqlite3_stmt* stmt,
                                           char** azColName)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(data, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL, 1);
    void *func = (void*)&CallbackTableQueryToCSV;
    rocprofvis_db_sqlite_callback_parameters* callback_params =
        (rocprofvis_db_sqlite_callback_parameters*) data;
    std::ofstream* file = (std::ofstream*)callback_params->handle;
    ROCPROFVIS_ASSERT_RETURN(file, 1);
    SqliteDatabase* db = (SqliteDatabase*) callback_params->db;
    if(callback_params->future->Interrupted()) return SQLITE_ABORT;

    bool delim = false;
    if(callback_params->future->GetProcessedRowsCount() == 0)
    {
        for(int i = 0; i < argc; i++)
        {
            if(delim)
            {
                *file << ',';
                delim = false;
            }
            *file << azColName[i];
            delim = true;
        }
        *file << "\n";
    }
    else
    {
        *file << "\n";
    }

    delim = false;
    for(int i = 0 ; i < argc; i++)
    {
        if(delim)
        {
            *file << ',';
            delim = false;
        }
        switch(sqlite3_column_type(stmt, i))
        {
            case SQLITE_NULL:
            {
                *file << db->GetNullExceptionString(func, azColName[i]);
                break;
            }
            case SQLITE_TEXT:
            {
                *file << '"';
                *file << sqlite3_column_text(stmt, i);
                *file << '"';
                break;
            }
            default:
            {
                *file << sqlite3_column_text(stmt, i);
                break;
            }
        }
        delim = true;
    }

    callback_params->future->CountThisRow();
    return 0;
}

rocprofvis_dm_result_t SqliteDatabase::ExecuteSQLQueryStatic(
                                    SqliteDatabase* db, 
                                    Future* future,
                                    const char* query,
                                    RpvSqliteExecuteQueryCallback callback)
{
    return future->SetPromise(
        db->ExecuteSQLQuery(future, query, callback)
    );
}

int
SqliteDatabase::DetectTable(sqlite3* conn, const char* table, bool is_view)
{
    int rc = SQLITE_ERROR;
    if(conn)
    {
        sqlite3_mutex_enter(sqlite3_db_mutex(conn));
        std::stringstream query;
        char*             zErrMsg = 0;
        query << "SELECT COUNT(name) FROM sqlite_master WHERE type="
              << (is_view ? "'view'" : "'table'") << "AND name='" << table << "';";
        rc = sqlite3_exec(
            conn, query.str().c_str(),
            [](void* data, int argc, char** argv, char** azColName) {
                uint32_t num = std::stol(argv[0]);
                return num > 0 ? 0 : 1;
            },
            nullptr, &zErrMsg);
        if(rc != SQLITE_OK)
        {
            spdlog::debug("Detect table error ");
            spdlog::debug(std::to_string(rc).c_str());
            spdlog::debug(":");
            spdlog::debug(zErrMsg);
            sqlite3_free(zErrMsg);
        }
        sqlite3_mutex_leave(sqlite3_db_mutex(conn));
    }
    else
    {
        spdlog::debug("Error : Connection cannot be nullptr!");
    }
    return rc;
}


rocprofvis_dm_result_t SqliteDatabase::Open()
{
    sqlite3* conn = GetConnection();
    if(nullptr == conn)
    {
        spdlog::debug("Can't open database:");
        return kRocProfVisDmResultDbAccessFailed;
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t SqliteDatabase::OpenConnection(sqlite3** connection)
{
    *connection = nullptr;
    if(sqlite3_open(Path(), connection) != SQLITE_OK)
    {
        spdlog::debug("Cannot open database connection - {}",
                      sqlite3_errmsg(*connection));
        sqlite3_close(*connection);
        *connection = nullptr;
        return kRocProfVisDmResultUnknownError;
    }
    else
    {
        sqlite3_exec(*connection, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
        sqlite3_exec(*connection, "PRAGMA synchronous = NORMAL;", nullptr, nullptr, nullptr);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t SqliteDatabase::Close()
{
    rocprofvis_dm_result_t result = kRocProfVisDmResultSuccess;
    if (m_connections_inuse.size() != 1)
    {
        spdlog::debug("Error : At the time of closing only one active connection should remain!");
        result = kRocProfVisDmResultUnknownError;
    }
    ReleaseConnection(*m_connections_inuse.begin());

    for (auto it = m_available_connections.begin(); it != m_available_connections.end(); ++it)
    {
        if(sqlite3_close(*it) != SQLITE_OK)
        {
            spdlog::debug("Can't close database connection:");
            spdlog::debug(sqlite3_errmsg(*it));
            result = kRocProfVisDmResultUnknownError;
        }
    }  
    return result;
}

void
SqliteDatabase::InterruptQuery(void* connection) {
    if (connection != nullptr)
    {
        sqlite3_interrupt((sqlite3*) connection);
    }
}

sqlite3* SqliteDatabase::GetConnection() 
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if(!m_available_connections.empty())
    {

        auto it = std::prev(m_available_connections.end());
        m_connections_inuse.insert(*it);
        sqlite3* conn = *it;
        m_available_connections.erase(it);
        return conn;
    }
    else
    {
        
        sqlite3* conn;
        size_t   thread_count = std::thread::hardware_concurrency();
        if(m_connections_inuse.size() > thread_count ||
           kRocProfVisDmResultSuccess != OpenConnection(&conn))
        {
           m_inuse_cv.wait(lock, [&] { return !m_available_connections.empty(); });
           
           auto it = std::prev(m_available_connections.end());
           m_connections_inuse.insert(*it);
           sqlite3* conn = *it;
           m_available_connections.erase(it);
           return conn;
        }
        else
        {
            m_connections_inuse.insert(conn);
            return conn;
        }
    }
}

sqlite3*
SqliteDatabase::GetServiceConnection()
{
    if (!m_connections_inuse.empty())
    {
        return *m_connections_inuse.begin();
    }
    return nullptr;
}

void SqliteDatabase::ReleaseConnection(sqlite3* conn)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto it = m_connections_inuse.find(conn);
    if (it != m_connections_inuse.end())
    {
        m_available_connections.insert(*it);
        m_connections_inuse.erase(it);
        m_inuse_cv.notify_one();
    }
}


rocprofvis_dm_result_t SqliteDatabase::ExecuteSQLQuery(Future* future, const char* query){
    rocprofvis_db_sqlite_callback_parameters params = {
        this,
        future,
        nullptr,
        nullptr,
        { query },
        static_cast<rocprofvis_dm_track_id_t>(-1)
    };
    return SqliteDatabase::ExecuteSQLQuery(query, &params);
}

rocprofvis_dm_result_t  SqliteDatabase::ExecuteSQLQuery(Future* future, const char* query, 
                                                        RpvSqliteExecuteQueryCallback callback){
    rocprofvis_db_sqlite_callback_parameters params = {
        this,
        future,
        nullptr,
        callback,
        { query },
        static_cast<rocprofvis_dm_track_id_t>(-1)
    };
    return SqliteDatabase::ExecuteSQLQuery(query, &params);
}


rocprofvis_dm_result_t SqliteDatabase::ExecuteSQLQuery(Future* future, const char* query, 
                                                RpvSqliteExecuteQueryCallback callback,
                                                rocprofvis_dm_string_t* value){
    rocprofvis_db_sqlite_callback_parameters params = {
        this,
        future,
        (rocprofvis_dm_handle_t) value,
        callback,
        { query },
        static_cast<rocprofvis_dm_track_id_t>(-1)
    };
    return SqliteDatabase::ExecuteSQLQuery(query, &params);
}

rocprofvis_dm_result_t SqliteDatabase::ExecuteSQLQuery(Future* future, const char* query, 
                                                RpvSqliteExecuteQueryCallback callback,
                                                uint64_t & value){
    std::string str_value;
    rocprofvis_dm_result_t result = ExecuteSQLQuery(future, query, callback, &str_value);
    if (result == kRocProfVisDmResultSuccess)
    {
        value = std::stoll(str_value);
    }
    return result;
}

rocprofvis_dm_result_t SqliteDatabase::ExecuteSQLQuery(Future* future, const char* query, 
                                                RpvSqliteExecuteQueryCallback callback,
                                                uint32_t & value){
    std::string str_value;
    rocprofvis_dm_result_t result = ExecuteSQLQuery(future, query, callback, &str_value);
    if (result == kRocProfVisDmResultSuccess)
    {
        value = std::stol(str_value);
    }
    return result;
}

rocprofvis_dm_result_t SqliteDatabase::ExecuteSQLQuery(Future* future, 
                                                        const char* query,
                                                        rocprofvis_dm_handle_t handle, 
                                                        RpvSqliteExecuteQueryCallback callback){
    rocprofvis_db_sqlite_callback_parameters params = {
        this,
        future,
        handle,
        callback,
        { query },
        static_cast<rocprofvis_dm_track_id_t>(-1)
    };
    return SqliteDatabase::ExecuteSQLQuery(query, &params);
}


rocprofvis_dm_result_t SqliteDatabase::ExecuteSQLQuery(Future* future, 
                                                        const char* query,
                                                        const char* cache_table_name,
                                                        rocprofvis_dm_handle_t handle,
                                                        rocprofvis_dm_event_operation_t op,
                                                        RpvSqliteExecuteQueryCallback callback){
    rocprofvis_db_sqlite_callback_parameters params = {
        this,
        future,
        handle,
        callback,
        { query, cache_table_name },
        static_cast<rocprofvis_dm_track_id_t>(-1),
        op
    };
    return SqliteDatabase::ExecuteSQLQuery(query, &params);
}


rocprofvis_dm_result_t
SqliteDatabase::ExecuteSQLQuery(Future* future, const char* query,
                                const char*                     cache_table_name,
                                rocprofvis_dm_handle_t          handle,
                                RpvSqliteExecuteQueryCallback   callback)
{
    rocprofvis_db_sqlite_callback_parameters params = {
        this,
        future,
        handle,
        callback,
        { query, cache_table_name },
        static_cast<rocprofvis_dm_track_id_t>(-1),
    };
    return SqliteDatabase::ExecuteSQLQuery(query, &params);
}


rocprofvis_dm_result_t  SqliteDatabase::ExecuteSQLQuery(Future* future, 
                                                        std::vector<std::string> query,
                                                        RpvSqliteExecuteQueryCallback callback)
{
    rocprofvis_db_sqlite_callback_parameters params = {
        this,
        future,
        nullptr,
        callback,
        { query[kRPVSourceQueryTrackByQueue].c_str(),
          query[kRPVSourceQueryTrackByStream].c_str(),
          query[kRPVSourceQueryLevel].c_str(), 
          query[kRPVSourceQuerySliceByQueue].c_str(),
          query[kRPVSourceQuerySliceByStream].c_str(),
          query[kRPVSourceQueryTable].c_str() },
        static_cast<rocprofvis_dm_track_id_t>(-1)
    };
    rocprofvis_dm_result_t result = SqliteDatabase::ExecuteSQLQuery(
        query[kRPVSourceQueryTrackByQueue].c_str(), &params);
    if(result == kRocProfVisDmResultSuccess &&
       query[kRPVSourceQueryTrackByStream].length() > 0)
    {
        return SqliteDatabase::ExecuteSQLQuery(
            query[kRPVSourceQueryTrackByStream].c_str(),
                                               &params);
    }
    return result;
}

int SqliteDatabase::Sqlite3Exec(sqlite3* db, const char* query,
                            int (*callback)(void*, int, sqlite3_stmt*, char**),
                              void* user_data)
{
    int rc=0;
    sqlite3_stmt* stmt = nullptr;
    rocprofvis_db_sqlite_callback_parameters* callback_params =
        (rocprofvis_db_sqlite_callback_parameters*) user_data;
    if (callback_params->future != nullptr)
    {
        callback_params->future->LinkDatabase(this, db);
    }
    sqlite3_mutex_enter(sqlite3_db_mutex(db));
    rc = sqlite3_prepare_v2(db, query, -1, &stmt, nullptr);
    if(rc == SQLITE_OK)
    {
        int                cols = sqlite3_column_count(stmt);
        std::vector<char*> col_names;
        for(int i = 0; i < cols; ++i)
        {
            col_names.push_back(const_cast<char*>(sqlite3_column_name(stmt, i)));
        }

        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            bool skip_this_row = false;

            for(int i = 0; i < cols; ++i)
            {
                if(sqlite3_column_type(stmt, i) == SQLITE_NULL)
                {
                    skip_this_row = NullExceptionSkip((void*)callback, col_names[i]);
                    if(skip_this_row)
                    {
                        break;
                    }
                }
            }

            if(skip_this_row == false)
            {
                rc = callback(user_data, cols, stmt, col_names.data());
                if(rc != 0)
                {
                    break;
                }
            }
        }

        sqlite3_finalize(stmt);
    }
    sqlite3_mutex_leave(sqlite3_db_mutex(db));
    if(callback_params->future != nullptr)
    {
        callback_params->future->LinkDatabase(nullptr, nullptr);
    }
    return rc;
}

rocprofvis_dm_result_t  SqliteDatabase::ExecuteSQLQuery(const char* query, rocprofvis_db_sqlite_callback_parameters * params)
{
    PROFILE;
    rocprofvis_dm_result_t result  = kRocProfVisDmResultSuccess;
    char *zErrMsg = 0;
    sqlite3* conn = GetConnection();
    int   rc = Sqlite3Exec(conn, query, params->callback, params);       
    if(rc != SQLITE_OK)
    {
        if (rc == SQLITE_ABORT)
        {
            result = kRocProfVisDmResultDbAbort;
        } else
        {
            spdlog::debug("Query: "); spdlog::debug(query);
            spdlog::debug("SQL error "); spdlog::debug(std::to_string(rc).c_str()); spdlog::debug(":"); 
            spdlog::debug(sqlite3_errmsg(conn));
            result = kRocProfVisDmResultDbAccessFailed;
        }
    } 
    ReleaseConnection(conn);
    return result;
}

rocprofvis_dm_result_t
SqliteDatabase::DropSQLTable(const char* table_name)
{
    sqlite3* conn = GetConnection();
    sqlite3_mutex_enter(sqlite3_db_mutex(conn));

    std::string query = "DROP TABLE IF EXISTS ";
    query += table_name;
    query += ";";
    sqlite3_exec(conn, query.c_str(), nullptr, nullptr, nullptr);

    sqlite3_mutex_leave(sqlite3_db_mutex(conn));
    ReleaseConnection(conn);
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t
SqliteDatabase::DropSQLIndex(const char* index_name)
{
    sqlite3* conn = GetConnection();
    sqlite3_mutex_enter(sqlite3_db_mutex(conn));

    std::string query = "DROP INDEX IF EXISTS ";
    query += index_name;
    query += ";";
    sqlite3_exec(conn, query.c_str(), nullptr, nullptr, nullptr);

    sqlite3_mutex_leave(sqlite3_db_mutex(conn));
    ReleaseConnection(conn);
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t
SqliteDatabase::ExecuteTransaction(std::vector<std::string> queries)
{
    sqlite3* conn = GetConnection();
    rocprofvis_dm_result_t result = kRocProfVisDmResultSuccess;
    while (true)
    {
        if(sqlite3_exec(conn, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr) !=
           SQLITE_OK)
        {
            result = kRocProfVisDmResultDbAccessFailed;
            spdlog::error("Failed to start transaction after error {}",
                          sqlite3_errmsg(conn));
            break;
        }

        for (const auto &query : queries)
        {
            if(sqlite3_exec(conn, query.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK)
            {
                result = kRocProfVisDmResultDbAccessFailed;
                spdlog::error("Failed to execute query '{}' with error {}", query,
                              sqlite3_errmsg(conn));
                break;
            }
        }
        // Break out of the outer loop if there was a failure
        if(result == kRocProfVisDmResultDbAccessFailed) 
        {
            // Try to rollback the transaction
            if(sqlite3_exec(conn, "ROLLBACK;", nullptr, nullptr, nullptr) != SQLITE_OK)
            {
                spdlog::error("Failed to rollback transaction after error {}",
                              sqlite3_errmsg(conn));
            }
            break;
        }

        if (sqlite3_exec(conn, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK)
        {
            result = kRocProfVisDmResultDbAccessFailed;
            spdlog::error("Failed to commit transaction {}", sqlite3_errmsg(conn));
            break;
        }
        break;
    }
    ReleaseConnection(conn);
    return result;
}

rocprofvis_dm_result_t
SqliteDatabase::CreateSQLTable(
                                const char* table_name, 
                                SQLInsertParams* parameters, 
                                uint8_t num_cols, 
                                size_t num_row,
                                std::function<void(sqlite3_stmt* stmt, int index)> insert_func)
{
    sqlite3* conn = GetConnection();
    sqlite3_mutex_enter(sqlite3_db_mutex(conn));
    while(true)
    {
        std::string query = "DROP TABLE IF EXISTS ";
        query += table_name;
        query += ";";
        if(sqlite3_exec(conn, query.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK)
        {
            break;
        }
        query = "CREATE TABLE IF NOT EXISTS ";
        query += table_name;
        query += "(";
        for(int i = 0; i < num_cols; i++)
        {
            if(i > 0)
            {
                query += ", ";
            }
            query += parameters[i].column;
            query += " ";
            query += parameters[i].type;
        }
        query += ") WITHOUT ROWID;";
        if(sqlite3_exec(conn, query.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK)
        {
            break;
        }

        if(sqlite3_exec(conn, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr) !=
           SQLITE_OK)
        {
            break;
        }

        query = "INSERT INTO ";
        query += table_name;
        query += "(";

        for(int i = 0; i < num_cols; i++)
        {
            if(i > 0)
            {
                query += ", ";
            }
            query += parameters[i].column;
        }
        query += ") VALUES (";

        for(int i = 0; i < num_cols; i++)
        {
            if(i > 0)
            {
                query += ", ";
            }
            query += "?";
        }
        query += ");";

        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(conn, query.c_str(), -1, &stmt, nullptr);

        for(int i = 0; i < num_row; i++)
        {
            insert_func(stmt, i);
            int rc = sqlite3_step(stmt);
            if(rc != SQLITE_DONE)
            {
                spdlog::debug("Insert failed");
            }
            sqlite3_reset(stmt);
        }

        sqlite3_finalize(stmt);

        if(sqlite3_exec(conn, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK)
        {
            break;
        }
        break;
    }
    sqlite3_mutex_leave(sqlite3_db_mutex(conn));
    ReleaseConnection(conn);
    return kRocProfVisDmResultSuccess;
}

uint64_t SqliteDatabase::GetBlanksMaskForQuery(std::string query)
{
    uint64_t mask = 0;
    for(auto it = m_column_masks.begin(); it != m_column_masks.end(); ++it)
    {
        if(query.find(it->first) != std::string::npos)
        {
            if(mask == 0)
                mask = it->second[kRPVTableQueryColumnMaskBlank];
            else
                mask &= it->second[kRPVTableQueryColumnMaskBlank];
        }
    }
    return mask;
}

std::array<uint64_t, kRPVTableQueryColumnMaskCount>
SqliteDatabase::GetColumnMasksForQuery(std::string_view query)
{    
    std::array<uint64_t, kRPVTableQueryColumnMaskCount> mask {0, 0 ,0};
    for(auto it = m_column_masks.begin(); it != m_column_masks.end(); ++it)
    {
        if(query.find(it->first) != std::string::npos)
        {
            if(mask[kRPVTableQueryColumnMaskBlank] == 0)
            {
                mask[kRPVTableQueryColumnMaskBlank] = it->second[kRPVTableQueryColumnMaskBlank];
            }
            else
            {           
                mask[kRPVTableQueryColumnMaskBlank] &= it->second[kRPVTableQueryColumnMaskBlank];
            }
            mask[kRPVTableQueryColumnMaskService] = it->second[kRPVTableQueryColumnMaskService];
            mask[kRPVTableQueryColumnMaskTimestamp] = it->second[kRPVTableQueryColumnMaskTimestamp];
        }
    }
    return mask;
}

void
SqliteDatabase::SetBlankMask(std::string op, uint64_t mask)
{
    m_column_masks[op][kRPVTableQueryColumnMaskBlank] |= mask;
}

void
SqliteDatabase::SetServiceMask(std::string op, uint64_t mask)
{
    m_column_masks[op][kRPVTableQueryColumnMaskService] |= mask;
}

void
SqliteDatabase::SetTimestampMask(std::string op, uint64_t mask)
{
    m_column_masks[op][kRPVTableQueryColumnMaskTimestamp] |= mask;
}

}  // namespace DataModel
}  // namespace RocProfVis
