// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_db_remote.h"
#include "../datamodel/rocprofvis_dm_trace.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstring>
#include <fstream>
#include <sstream>

namespace RocProfVis {
namespace DataModel {

using json = nlohmann::json;

namespace {

size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb,
                         std::string* output) {
    size_t total_size = size * nmemb;
    output->append(static_cast<char*>(contents), total_size);
    return total_size;
}

void InitCurlOnce() {
    static bool initialized = false;
    if (!initialized) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        initialized = true;
    }
}

}  // namespace

RemoteDatabase::RemoteDatabase(const std::string& url)
    : Database(url.c_str()), m_server_url(url) {
    InitCurlOnce();
}

RemoteDatabase::~RemoteDatabase() {
    Close();
}

rocprofvis_dm_result_t RemoteDatabase::Open() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_connected) {
        return kRocProfVisDmResultSuccess;
    }

    std::string response;
    if (!HttpGet("/status", response)) {
        spdlog::error("RemoteDatabase: Failed to connect to {}", m_server_url);
        return kRocProfVisDmResultUnknownError;
    }

    try {
        auto status = json::parse(response);
        if (status.value("status", "") != "ok") {
            spdlog::error("RemoteDatabase: Server returned error status");
            return kRocProfVisDmResultUnknownError;
        }

        std::string db_path = status.value("database", "unknown");
        spdlog::info("RemoteDatabase: Connected to {} (database: {})",
                     m_server_url, db_path);

        m_connected = true;
        return kRocProfVisDmResultSuccess;

    } catch (const json::exception& e) {
        spdlog::error("RemoteDatabase: Failed to parse status response: {}",
                      e.what());
        return kRocProfVisDmResultUnknownError;
    }
}

rocprofvis_dm_result_t RemoteDatabase::Close() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_connected = false;
    return kRocProfVisDmResultSuccess;
}

void RemoteDatabase::InterruptQuery(void* connection) {
    m_interrupted = true;
}

bool RemoteDatabase::HttpGet(const std::string& endpoint, std::string& response) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        spdlog::error("RemoteDatabase: Failed to initialize CURL");
        return false;
    }

    std::string url = m_server_url + endpoint;
    response.clear();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        spdlog::error("RemoteDatabase: HTTP GET failed: {}",
                      curl_easy_strerror(res));
        return false;
    }

    if (http_code >= 400) {
        spdlog::error("RemoteDatabase: HTTP {} error from {}", http_code, url);
        return false;
    }

    return true;
}

bool RemoteDatabase::HttpPost(const std::string& endpoint, const std::string& body,
                              std::string& response) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        spdlog::error("RemoteDatabase: Failed to initialize CURL");
        return false;
    }

    std::string url = m_server_url + endpoint;
    response.clear();

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        spdlog::error("RemoteDatabase: HTTP POST failed: {}",
                      curl_easy_strerror(res));
        return false;
    }

    if (http_code >= 400) {
        spdlog::error("RemoteDatabase: HTTP {} error from {}", http_code, url);
        return false;
    }

    return true;
}

rocprofvis_dm_result_t RemoteDatabase::ExecuteRemoteQuery(const std::string& sql,
                                                           std::string& result) {
    if (!m_connected) {
        spdlog::error("RemoteDatabase: Not connected");
        return kRocProfVisDmResultUnknownError;
    }

    json request;
    request["sql"] = sql;
    request["cache"] = true;

    std::string response;
    if (!HttpPost("/query", request.dump(), response)) {
        return kRocProfVisDmResultUnknownError;
    }

    result = response;
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RemoteDatabase::ReadTraceMetadata(Future* future) {
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                  kRocProfVisDmResultInvalidParameter);

    while (true) {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject(), ERROR_TRACE_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject()->trace_properties,
                                     ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);

        ShowProgress(5, "Connecting to remote server", kRPVDbBusy, future);

        std::string response;
        if (!HttpGet("/metadata", response)) {
            ShowProgress(0, "Failed to fetch metadata", kRPVDbError, future);
            break;
        }

        try {
            auto metadata = json::parse(response);
            if (!metadata.value("success", false)) {
                spdlog::error("RemoteDatabase: Metadata query failed");
                break;
            }

            ShowProgress(20, "Loading time range", kRPVDbBusy, future);

            std::string traces_response;
            if (HttpGet("/traces", traces_response)) {
                auto traces = json::parse(traces_response);
                if (traces.value("success", false) && traces.contains("data")) {
                    auto trace_data = traces["data"];
                    if (trace_data.contains("time_range")) {
                        auto time_range = trace_data["time_range"];
                        TraceProperties()->start_time =
                            time_range.value("start_time", uint64_t(0));
                        TraceProperties()->end_time =
                            time_range.value("end_time", uint64_t(0));
                    }
                }
            }

            ShowProgress(50, "Loading track information", kRPVDbBusy, future);

            std::string tracks_response;
            if (HttpGet("/tracks", tracks_response)) {
                auto tracks = json::parse(tracks_response);
                if (tracks.value("success", false) && tracks.contains("data")) {
                    auto track_data = tracks["data"];
                    if (track_data.is_array()) {
                        for (const auto& track : track_data) {
                            rocprofvis_dm_track_params_t params = {};
                            params.track_id = track.value("id", 0);
                            params.process.category = static_cast<rocprofvis_dm_track_category_t>(
                                track.value("category", 0));
                            params.process.id[TRACK_ID_PID] = track.value("pid", uint64_t(0));
                            params.process.id[TRACK_ID_TID] = track.value("tid", uint64_t(0));

                            BindObject()->FuncAddTrack(BindObject()->trace_object, &params);
                        }
                    }
                }
            }

            TraceProperties()->metadata_loaded = true;
            ShowProgress(100, "Metadata loaded successfully", kRPVDbSuccess, future);
            return future->SetPromise(kRocProfVisDmResultSuccess);

        } catch (const json::exception& e) {
            spdlog::error("RemoteDatabase: Failed to parse metadata: {}", e.what());
            break;
        }
    }

    ShowProgress(0, "Failed to load metadata", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort
                                                     : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t RemoteDatabase::ReadTraceSlice(
    rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end,
    rocprofvis_db_num_of_tracks_t num, rocprofvis_db_track_selection_t tracks,
    Future* future) {

    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                  kRocProfVisDmResultInvalidParameter);

    while (true) {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject(), ERROR_TRACE_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties(), ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties()->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);

        ShowProgress(5, "Fetching slice data from remote", kRPVDbBusy, future);

        std::stringstream sql;
        sql << "SELECT * FROM rocpd_op WHERE start >= " << start
            << " AND end <= " << end;

        if (num > 0 && tracks) {
            sql << " AND track_id IN (";
            for (uint32_t i = 0; i < num; ++i) {
                if (i > 0) sql << ",";
                sql << tracks[i];
            }
            sql << ")";
        }

        sql << " ORDER BY start";

        std::string result;
        rocprofvis_dm_result_t status = ExecuteRemoteQuery(sql.str(), result);
        if (status != kRocProfVisDmResultSuccess) {
            break;
        }

        ShowProgress(50, "Processing slice data", kRPVDbBusy, future);

        try {
            auto response = json::parse(result);
            if (!response.value("success", false)) {
                std::string error = response.value("error", "Unknown error");
                spdlog::error("RemoteDatabase: Query failed: {}", error);
                break;
            }

            auto data = response["data"];
            if (!data.is_array()) {
                ShowProgress(100, "Slice loaded (no data)", kRPVDbSuccess, future);
                return future->SetPromise(kRocProfVisDmResultSuccess);
            }

            for (const auto& row : data) {
                if (!row.is_object()) continue;

                rocprofvis_db_record_data_t record = {};
                record.event.id.bitfield.event_id = row.value("id", uint64_t(0));
                record.event.id.bitfield.event_op = static_cast<uint8_t>(
                    row.value("op_type", 0));
                record.event.timestamp = row.value("start", uint64_t(0));
                record.event.duration = row.value("end", uint64_t(0)) - record.event.timestamp;
                record.event.category = row.value("category_id", uint64_t(0));
                record.event.symbol = row.value("symbol_id", uint64_t(0));
                record.event.level = row.value("level", uint8_t(0));

                rocprofvis_dm_track_id_t track_id = row.value("track_id", 0);

                rocprofvis_dm_slice_t slice = BindObject()->FuncAddSlice(
                    BindObject()->trace_object, track_id, start, end);
                if (slice) {
                    BindObject()->FuncAddRecord(slice, record);
                    BindObject()->FuncCompleteSlice(slice);
                }

                future->CountThisRow();
            }

            ShowProgress(100, "Slice loaded successfully", kRPVDbSuccess, future);
            return future->SetPromise(kRocProfVisDmResultSuccess);

        } catch (const json::exception& e) {
            spdlog::error("RemoteDatabase: Failed to parse slice data: {}", e.what());
            break;
        }
    }

    ShowProgress(0, "Failed to load slice", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort
                                                     : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t RemoteDatabase::ReadFlowTraceInfo(
    rocprofvis_dm_event_id_t event_id, Future* future) {

    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                  kRocProfVisDmResultInvalidParameter);

    while (true) {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject(), ERROR_TRACE_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties(), ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties()->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);

        rocprofvis_dm_flowtrace_t flowtrace =
            BindObject()->FuncAddFlowTrace(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(flowtrace, ERROR_FLOW_TRACE_CANNOT_BE_NULL);

        std::stringstream sql;
        sql << "SELECT * FROM rocpd_api_ops WHERE ";
        if (event_id.bitfield.event_op == kRocProfVisDmOperationLaunch) {
            sql << "api_id = " << event_id.bitfield.event_id;
        } else if (event_id.bitfield.event_op == kRocProfVisDmOperationDispatch) {
            sql << "op_id = " << event_id.bitfield.event_id;
        } else {
            ShowProgress(0, "Flow trace not available for this operation type",
                         kRPVDbError, future);
            return future->SetPromise(kRocProfVisDmResultInvalidParameter);
        }

        std::string result;
        rocprofvis_dm_result_t status = ExecuteRemoteQuery(sql.str(), result);
        if (status != kRocProfVisDmResultSuccess) {
            break;
        }

        try {
            auto response = json::parse(result);
            if (!response.value("success", false)) {
                break;
            }

            auto data = response["data"];
            if (data.is_array()) {
                for (const auto& row : data) {
                    rocprofvis_db_flow_data_t flow_data = {};
                    flow_data.id.bitfield.event_id = row.value("id", uint64_t(0));
                    flow_data.time = row.value("start", uint64_t(0));
                    flow_data.track_id = row.value("track_id", 0);
                    flow_data.symbol_id = row.value("symbol_id", uint64_t(0));
                    flow_data.category_id = row.value("category_id", uint64_t(0));

                    BindObject()->FuncAddFlow(flowtrace, flow_data);
                    future->CountThisRow();
                }
            }

            ShowProgress(100, "Flow trace loaded successfully", kRPVDbSuccess, future);
            return future->SetPromise(kRocProfVisDmResultSuccess);

        } catch (const json::exception& e) {
            spdlog::error("RemoteDatabase: Failed to parse flow trace: {}", e.what());
            break;
        }
    }

    ShowProgress(0, "Failed to load flow trace", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort
                                                     : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t RemoteDatabase::ReadStackTraceInfo(
    rocprofvis_dm_event_id_t event_id, Future* future) {

    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                  kRocProfVisDmResultInvalidParameter);

    while (true) {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject(), ERROR_TRACE_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties(), ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties()->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);

        rocprofvis_dm_stacktrace_t stacktrace =
            BindObject()->FuncAddStackTrace(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(stacktrace, ERROR_STACK_TRACE_CANNOT_BE_NULL);

        if (event_id.bitfield.event_op != kRocProfVisDmOperationLaunch &&
            event_id.bitfield.event_op != kRocProfVisDmOperationMemoryAllocate) {
            ShowProgress(0, "Stack trace not available for this operation type",
                         kRPVDbError, future);
            return future->SetPromise(kRocProfVisDmResultInvalidParameter);
        }

        std::stringstream sql;
        sql << "SELECT sf.name, ap.apiName, ap.args, sf.depth "
            << "FROM rocpd_stackframe sf "
            << "JOIN rocpd_api ap ON sf.api_ptr_id = ap.id "
            << "WHERE ap.id = " << event_id.bitfield.event_id;

        std::string result;
        rocprofvis_dm_result_t status = ExecuteRemoteQuery(sql.str(), result);
        if (status != kRocProfVisDmResultSuccess) {
            break;
        }

        try {
            auto response = json::parse(result);
            if (!response.value("success", false)) {
                break;
            }

            auto data = response["data"];
            if (data.is_array()) {
                for (const auto& row : data) {
                    rocprofvis_db_stack_data_t stack_data = {};
                    std::string symbol = row.value("name", "");
                    std::string args = row.value("args", "");
                    std::string line = row.value("apiName", "");
                    stack_data.symbol = symbol.c_str();
                    stack_data.args = args.c_str();
                    stack_data.line = line.c_str();
                    stack_data.depth = row.value("depth", uint32_t(0));

                    BindObject()->FuncAddStackFrame(stacktrace, stack_data);
                    future->CountThisRow();
                }
            }

            ShowProgress(100, "Stack trace loaded successfully", kRPVDbSuccess, future);
            return future->SetPromise(kRocProfVisDmResultSuccess);

        } catch (const json::exception& e) {
            spdlog::error("RemoteDatabase: Failed to parse stack trace: {}", e.what());
            break;
        }
    }

    ShowProgress(0, "Failed to load stack trace", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort
                                                     : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t RemoteDatabase::ReadExtEventInfo(
    rocprofvis_dm_event_id_t event_id, Future* future) {

    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                  kRocProfVisDmResultInvalidParameter);

    while (true) {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject(), ERROR_TRACE_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties(), ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties()->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);

        rocprofvis_dm_extdata_t extdata =
            BindObject()->FuncAddExtData(BindObject()->trace_object, event_id);
        ROCPROFVIS_ASSERT_MSG_BREAK(extdata, ERROR_EXT_DATA_CANNOT_BE_NULL);

        std::stringstream sql;
        sql << "SELECT * FROM rocpd_ext_data WHERE event_id = "
            << event_id.bitfield.event_id;

        std::string result;
        rocprofvis_dm_result_t status = ExecuteRemoteQuery(sql.str(), result);
        if (status != kRocProfVisDmResultSuccess) {
            break;
        }

        try {
            auto response = json::parse(result);
            if (!response.value("success", false)) {
                break;
            }

            auto data = response["data"];
            if (data.is_array()) {
                for (const auto& row : data) {
                    rocprofvis_db_ext_data_t ext_record = {};
                    std::string category = row.value("category", "");
                    std::string name = row.value("name", "");
                    std::string value = row.value("data", "");
                    ext_record.category = category.c_str();
                    ext_record.name = name.c_str();
                    ext_record.data = value.c_str();
                    ext_record.type = kRPVDataTypeString;

                    BindObject()->FuncAddExtDataRecord(extdata, ext_record);
                    future->CountThisRow();
                }
            }

            ShowProgress(100, "Extended data loaded successfully", kRPVDbSuccess, future);
            return future->SetPromise(kRocProfVisDmResultSuccess);

        } catch (const json::exception& e) {
            spdlog::error("RemoteDatabase: Failed to parse extended data: {}", e.what());
            break;
        }
    }

    ShowProgress(0, "Failed to load extended data", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort
                                                     : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t RemoteDatabase::ExecuteQuery(
    rocprofvis_dm_charptr_t query, rocprofvis_dm_charptr_t description,
    Future* future) {

    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                  kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(query, ERROR_SQL_QUERY_PARAMETERS_CANNOT_BE_NULL,
                                  kRocProfVisDmResultInvalidParameter);

    while (true) {
        ROCPROFVIS_ASSERT_MSG_BREAK(BindObject(), ERROR_TRACE_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties(), ERROR_TRACE_PROPERTIES_CANNOT_BE_NULL);
        ROCPROFVIS_ASSERT_MSG_BREAK(TraceProperties()->metadata_loaded, ERROR_METADATA_IS_NOT_LOADED);

        rocprofvis_dm_table_t table =
            BindObject()->FuncAddTable(BindObject()->trace_object, query, description);
        ROCPROFVIS_ASSERT_MSG_BREAK(table, ERROR_TABLE_CANNOT_BE_NULL);

        ShowProgress(10, description ? description : "Executing query", kRPVDbBusy, future);

        std::string result;
        rocprofvis_dm_result_t status = ExecuteRemoteQuery(query, result);
        if (status != kRocProfVisDmResultSuccess) {
            break;
        }

        try {
            auto response = json::parse(result);
            if (!response.value("success", false)) {
                std::string error = response.value("error", "Unknown error");
                spdlog::error("RemoteDatabase: Query failed: {}", error);
                break;
            }

            auto data = response["data"];
            if (!data.is_array() || data.empty()) {
                ShowProgress(100, "Query completed (no results)", kRPVDbSuccess, future);
                return future->SetPromise(kRocProfVisDmResultSuccess);
            }

            bool columns_added = false;
            for (const auto& row_json : data) {
                if (!row_json.is_object()) continue;

                if (!columns_added) {
                    for (const auto& [key, value] : row_json.items()) {
                        BindObject()->FuncAddTableColumn(table, key.c_str());
                    }
                    columns_added = true;
                }

                rocprofvis_dm_table_row_t row = BindObject()->FuncAddTableRow(table);
                if (!row) break;

                for (const auto& [key, value] : row_json.items()) {
                    std::string cell_value;
                    if (value.is_string()) {
                        cell_value = value.get<std::string>();
                    } else if (value.is_null()) {
                        cell_value = "";
                    } else {
                        cell_value = value.dump();
                    }

                    BindObject()->FuncAddTableRowCell(row, cell_value.c_str());
                }

                future->CountThisRow();
            }

            ShowProgress(100, "Query completed successfully", kRPVDbSuccess, future);
            return future->SetPromise(kRocProfVisDmResultSuccess);

        } catch (const json::exception& e) {
            spdlog::error("RemoteDatabase: Failed to parse query response: {}", e.what());
            break;
        }
    }

    ShowProgress(0, "Query failed", kRPVDbError, future);
    return future->SetPromise(future->Interrupted() ? kRocProfVisDmResultDbAbort
                                                     : kRocProfVisDmResultDbAccessFailed);
}

rocprofvis_dm_result_t RemoteDatabase::BuildTrackQuery(
    rocprofvis_dm_index_t index, rocprofvis_dm_index_t type,
    rocprofvis_dm_string_t& query, uint32_t split_count, uint32_t split_index) {

    std::stringstream ss;
    ss << "SELECT * FROM rocpd_track WHERE id = " << index;
    query = ss.str();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RemoteDatabase::BuildSliceQuery(
    rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end,
    rocprofvis_db_num_of_tracks_t num, rocprofvis_db_track_selection_t tracks,
    rocprofvis_dm_string_t& query, slice_array_t& slices) {

    std::stringstream ss;
    ss << "SELECT * FROM rocpd_op WHERE start >= " << start
       << " AND end <= " << end;

    if (num > 0 && tracks) {
        ss << " AND track_id IN (";
        for (uint32_t i = 0; i < num; ++i) {
            if (i > 0) ss << ",";
            ss << tracks[i];
        }
        ss << ")";
    }

    ss << " ORDER BY start";
    query = ss.str();

    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RemoteDatabase::BuildTableQuery(
    rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end,
    rocprofvis_db_num_of_tracks_t num, rocprofvis_db_track_selection_t tracks,
    rocprofvis_dm_charptr_t where, rocprofvis_dm_charptr_t filter,
    rocprofvis_dm_charptr_t group, rocprofvis_dm_charptr_t group_cols,
    rocprofvis_dm_charptr_t sort_column, rocprofvis_dm_sort_order_t sort_order,
    rocprofvis_dm_num_string_table_filters_t num_string_table_filters,
    rocprofvis_dm_string_table_filters_t string_table_filters,
    uint64_t max_count, uint64_t offset, bool count_only, bool summary,
    rocprofvis_dm_string_t& query) {

    std::stringstream ss;

    if (count_only) {
        ss << "SELECT COUNT(*) as count FROM rocpd_op";
    } else {
        ss << "SELECT * FROM rocpd_op";
    }

    ss << " WHERE start >= " << start << " AND end <= " << end;

    if (where && strlen(where) > 0) {
        ss << " AND (" << where << ")";
    }

    if (filter && strlen(filter) > 0) {
        ss << " AND (" << filter << ")";
    }

    if (group && strlen(group) > 0) {
        ss << " GROUP BY " << group;
    }

    if (sort_column && strlen(sort_column) > 0) {
        ss << " ORDER BY " << sort_column;
        ss << (sort_order == kRPVDMSortOrderAsc ? " ASC" : " DESC");
    }

    if (max_count > 0) {
        ss << " LIMIT " << max_count;
    }

    if (offset > 0) {
        ss << " OFFSET " << offset;
    }

    query = ss.str();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RemoteDatabase::BuildTableStringIdFilter(
    rocprofvis_dm_num_string_table_filters_t num_string_table_filters,
    rocprofvis_dm_string_table_filters_t string_table_filters,
    table_string_id_filter_map_t& filter) {

    // Not implemented for remote - requires local string table access
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RemoteDatabase::BuildTableSummaryClause(
    bool sample_query, rocprofvis_dm_string_t& select,
    rocprofvis_dm_string_t& group_by) {

    if (sample_query) {
        select = "name, COUNT(*) as count, AVG(end - start) as avg_duration";
        group_by = "name";
    } else {
        select = "*";
        group_by = "";
    }

    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t RemoteDatabase::SaveTrimmedData(
    rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end,
    rocprofvis_dm_charptr_t new_db_path, Future* future) {

    spdlog::error("RemoteDatabase: SaveTrimmedData not supported for remote databases");
    return kRocProfVisDmResultNotSupported;
}

rocprofvis_dm_result_t RemoteDatabase::ExportTableCSV(
    rocprofvis_dm_charptr_t query, rocprofvis_dm_charptr_t file_path,
    Future* future) {

    std::string result;
    rocprofvis_dm_result_t status = ExecuteRemoteQuery(query, result);
    if (status != kRocProfVisDmResultSuccess) {
        return status;
    }

    try {
        auto response = json::parse(result);
        if (!response.value("success", false)) {
            return kRocProfVisDmResultUnknownError;
        }

        auto data = response["data"];
        if (!data.is_array() || data.empty()) {
            return kRocProfVisDmResultSuccess;
        }

        std::ofstream file(file_path);
        if (!file.is_open()) {
            spdlog::error("RemoteDatabase: Failed to open file for writing: {}",
                          file_path);
            return kRocProfVisDmResultUnknownError;
        }

        bool first = true;
        for (const auto& [key, value] : data[0].items()) {
            if (!first) file << ",";
            file << key;
            first = false;
        }
        file << "\n";

        for (const auto& row : data) {
            first = true;
            for (const auto& [key, value] : row.items()) {
                if (!first) file << ",";
                if (value.is_string()) {
                    file << "\"" << value.get<std::string>() << "\"";
                } else {
                    file << value.dump();
                }
                first = false;
            }
            file << "\n";
        }

        return kRocProfVisDmResultSuccess;

    } catch (const std::exception& e) {
        spdlog::error("RemoteDatabase: Failed to export CSV: {}", e.what());
        return kRocProfVisDmResultUnknownError;
    }
}

rocprofvis_dm_result_t RemoteDatabase::FindTrackId(
    uint64_t node, uint32_t process, const char* subprocess,
    rocprofvis_dm_op_t operation, rocprofvis_dm_track_id_t& track_id) {

    // For remote database, we would need to query the server for track info
    // This is a simplified implementation that returns not found
    track_id = 0;
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_string_t RemoteDatabase::GetEventOperationQuery(
    rocprofvis_dm_event_operation_t operation) {

    // Return appropriate table name based on operation type
    switch (operation) {
        case kRocProfVisDmOperationLaunch:
            return "rocpd_api";
        case kRocProfVisDmOperationDispatch:
            return "rocpd_op";
        default:
            return "";
    }
}

}  // namespace DataModel
}  // namespace RocProfVis