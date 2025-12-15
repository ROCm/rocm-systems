// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_db.h"

#include <mutex>
#include <string>

namespace RocProfVis {
namespace DataModel {

// Database backend that connects to rocstorage-server or rocstorage-ssh-proxy
// via HTTP. Allows reading trace data from remote machines without direct
// filesystem access to the database file.
class RemoteDatabase : public Database {
public:
    explicit RemoteDatabase(const std::string& url);
    ~RemoteDatabase() override;

    rocprofvis_dm_result_t Open() override;
    rocprofvis_dm_result_t Close() override;
    void InterruptQuery(void* connection) override;

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

    rocprofvis_dm_result_t BuildTableStringIdFilter(
        rocprofvis_dm_num_string_table_filters_t num_string_table_filters,
        rocprofvis_dm_string_table_filters_t string_table_filters,
        table_string_id_filter_map_t& filter) override;

    rocprofvis_dm_result_t BuildTableSummaryClause(
        bool sample_query,
        rocprofvis_dm_string_t& select,
        rocprofvis_dm_string_t& group_by) override;

    rocprofvis_dm_result_t SaveTrimmedData(
        rocprofvis_dm_timestamp_t start,
        rocprofvis_dm_timestamp_t end,
        rocprofvis_dm_charptr_t new_db_path,
        Future* future) override;

    const std::string& GetServerUrl() const { return m_server_url; }
    bool IsConnected() const { return m_connected; }

protected:
    rocprofvis_dm_result_t ReadTraceMetadata(Future* object) override;

    rocprofvis_dm_result_t ReadTraceSlice(
        rocprofvis_dm_timestamp_t start,
        rocprofvis_dm_timestamp_t end,
        rocprofvis_db_num_of_tracks_t num,
        rocprofvis_db_track_selection_t tracks,
        Future* object) override;

    rocprofvis_dm_result_t ReadFlowTraceInfo(
        rocprofvis_dm_event_id_t event_id,
        Future* object) override;

    rocprofvis_dm_result_t ReadStackTraceInfo(
        rocprofvis_dm_event_id_t event_id,
        Future* object) override;

    rocprofvis_dm_result_t ReadExtEventInfo(
        rocprofvis_dm_event_id_t event_id,
        Future* object) override;

    rocprofvis_dm_result_t ExecuteQuery(
        rocprofvis_dm_charptr_t query,
        rocprofvis_dm_charptr_t description,
        Future* object) override;

    rocprofvis_dm_result_t BuildTrackQuery(
        rocprofvis_dm_index_t index,
        rocprofvis_dm_index_t type,
        rocprofvis_dm_string_t& query,
        uint32_t split_count,
        uint32_t split_index) override;

    rocprofvis_dm_result_t BuildSliceQuery(
        rocprofvis_dm_timestamp_t start,
        rocprofvis_dm_timestamp_t end,
        rocprofvis_db_num_of_tracks_t num,
        rocprofvis_db_track_selection_t tracks,
        rocprofvis_dm_string_t& query,
        slice_array_t& slices) override;

    rocprofvis_dm_result_t ExportTableCSV(
        rocprofvis_dm_charptr_t query,
        rocprofvis_dm_charptr_t file_path,
        Future* object) override;

    rocprofvis_dm_result_t FindTrackId(
        uint64_t node,
        uint32_t process,
        const char* subprocess,
        rocprofvis_dm_op_t operation,
        rocprofvis_dm_track_id_t& track_id) override;

    rocprofvis_dm_string_t GetEventOperationQuery(
        rocprofvis_dm_event_operation_t operation) override;

private:
    bool HttpGet(const std::string& endpoint, std::string& response);
    bool HttpPost(const std::string& endpoint, const std::string& body,
                  std::string& response);

    rocprofvis_dm_result_t ExecuteRemoteQuery(const std::string& sql,
                                               std::string& result);

    std::string m_server_url;
    bool m_connected = false;
    bool m_interrupted = false;
    std::mutex m_mutex;
};

}  // namespace DataModel
}  // namespace RocProfVis
