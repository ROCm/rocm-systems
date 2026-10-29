// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "trace_trim.hpp"

#include "ph_future.hpp"
#include "trim_schema.hpp"

#include "common/debug.hpp"
#include "data_storage/backends/sqlite_api_policy.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace profiler_hub
{
namespace
{

using data_storage::sqlite_api_policy;

struct master_row_t
{
    std::string type;
    std::string name;
    std::string table_name;
    std::string sql;
};

struct planned_copy_t
{
    std::string name;
    row_kind_t  kind{ row_kind_t::copy };
    std::string sql;
};

class sqlite_connection_t
{
public:
    explicit sqlite_connection_t(const std::string& path, bool readonly)
    {
        const int rc = readonly ? sqlite_api_policy::open_readonly(path.c_str(), &m_db)
                                : sqlite_api_policy::open(path.c_str(), &m_db);
        if(rc != sqlite_api_policy::result_ok || m_db == nullptr)
        {
            throw std::runtime_error("SQLite database could not be opened: " + path);
        }
    }

    ~sqlite_connection_t() { sqlite_api_policy::close(m_db); }

    sqlite_connection_t(const sqlite_connection_t&)            = delete;
    sqlite_connection_t& operator=(const sqlite_connection_t&) = delete;

    [[nodiscard]] sqlite_api_policy::database_t get() const { return m_db; }

    void execute(const std::string& sql) const
    {
        if(sqlite_api_policy::exec(m_db, sql.c_str()) != sqlite_api_policy::result_ok)
        {
            throw std::runtime_error(sqlite_api_policy::errmsg(m_db));
        }
    }

private:
    sqlite_api_policy::database_t m_db{ nullptr };
};

std::string
sql_text(sqlite_api_policy::statement_t statement, int column)
{
    return sqlite_api_policy::column_text(statement, column);
}

std::vector<master_row_t>
read_master(const sqlite_connection_t& source)
{
    sqlite_api_policy::statement_t statement = nullptr;
    const char* sql = "SELECT type, name, tbl_name, sql FROM sqlite_master "
                      "WHERE sql IS NOT NULL ORDER BY rowid";
    if(sqlite_api_policy::prepare(source.get(), sql, &statement) != sqlite_api_policy::result_ok)
    {
        throw std::runtime_error(sqlite_api_policy::errmsg(source.get()));
    }

    std::vector<master_row_t> rows;
    int                       rc = sqlite_api_policy::result_done;
    while((rc = sqlite_api_policy::step(statement)) == sqlite_api_policy::result_row)
    {
        master_row_t row;
        row.type       = sql_text(statement, 0);
        row.name       = sql_text(statement, 1);
        row.table_name = sql_text(statement, 2);
        row.sql        = sql_text(statement, 3);
        rows.push_back(std::move(row));
    }
    sqlite_api_policy::finalize(statement);
    if(rc != sqlite_api_policy::result_done)
    {
        throw std::runtime_error(sqlite_api_policy::errmsg(source.get()));
    }
    return rows;
}

bool
is_internal_name(std::string_view name)
{
    return name.rfind("sqlite_", 0) == 0 || name.rfind("roc_optiq_", 0) == 0;
}

bool
mentions_omitted(std::string_view sql, const std::unordered_set<std::string>& omitted)
{
    for(const std::string& name : omitted)
    {
        if(sql.find(name) != std::string_view::npos) return true;
    }
    return false;
}

std::string
attach_sql(const std::string& source_path)
{
    std::string escaped;
    escaped.reserve(source_path.size());
    for(const char ch : source_path)
    {
        if(ch == '\'') escaped += "''";
        else escaped += ch;
    }
    return "ATTACH DATABASE '" + escaped + "' AS oldDb";
}

void
report(ph_future* future, double value, const char* description)
{
    if(future != nullptr) future->report_progress(value, description);
}

bool
cancelled(ph_future* future)
{
    return future != nullptr && future->stop_token().stop_requested();
}

}  // namespace

ph_result_t
trim_trace_file(const std::string&           source_path,
                const std::string&           output_path,
                reader_types::trace_schema_t schema,
                version_t                    version,
                uint64_t                     start_ts,
                uint64_t                     end_ts,
                ph_future*                   future)
{
    if(source_path.empty() || output_path.empty()) return PH_RESULT_INVALID_ARGUMENT;
    std::error_code error;
    if(std::filesystem::exists(output_path, error) &&
       std::filesystem::equivalent(source_path, output_path, error))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    const std::unique_ptr<trim_schema_t> trim_schema = make_trim_schema(schema, version);
    if(trim_schema == nullptr) return PH_RESULT_UNSUPPORTED_FILE;

    std::filesystem::remove(output_path, error);
    if(error) return PH_RESULT_INVALID_FILE;

    const std::vector<master_row_t> master = read_master(sqlite_connection_t{ source_path, true });
    std::unordered_set<std::string> kept_tables;
    std::unordered_set<std::string> omitted;
    std::vector<master_row_t>       tables;
    std::vector<master_row_t>       views;
    std::vector<master_row_t>       indexes;
    for(const master_row_t& row : master)
    {
        if(is_internal_name(row.name) || is_internal_name(row.table_name))
        {
            omitted.insert(row.name);
            omitted.insert(row.table_name);
            continue;
        }
        if(row.type == "table")
        {
            kept_tables.insert(row.name);
            tables.push_back(row);
        }
        else if(row.type == "view") views.push_back(row);
        else if(row.type == "index" || row.type == "trigger") indexes.push_back(row);
    }

    std::vector<planned_copy_t> copies;
    copies.reserve(tables.size());
    for(const master_row_t& table : tables)
    {
        trim_statement_t statement =
            trim_schema->statement_for(table.name, kept_tables, start_ts, end_ts);
        copies.push_back(
            planned_copy_t{ table.name, statement.kind, std::move(statement.sql) });
    }
    std::sort(copies.begin(), copies.end(), [](const planned_copy_t& left,
                                                const planned_copy_t& right) {
        if(left.kind != right.kind)
            return static_cast<int>(left.kind) < static_cast<int>(right.kind);
        return left.name < right.name;
    });

    try
    {
        sqlite_connection_t destination{ output_path, false };
        destination.execute("PRAGMA foreign_keys = OFF");
        report(future, 0.05, "Creating trimmed database");
        for(const master_row_t& table : tables)
        {
            destination.execute(table.sql);
        }
        for(const master_row_t& view : views)
        {
            if(!mentions_omitted(view.sql, omitted)) destination.execute(view.sql);
        }

        destination.execute("BEGIN");
        destination.execute(attach_sql(std::filesystem::path{ source_path }.generic_string()));
        const double span = copies.empty() ? 1.0 : 0.9 / static_cast<double>(copies.size());
        for(size_t index = 0; index < copies.size(); ++index)
        {
            if(cancelled(future))
            {
                destination.execute("ROLLBACK");
                throw std::runtime_error("cancelled");
            }
            report(future,
                   0.05 + span * static_cast<double>(index),
                   copies[index].name.c_str());
            destination.execute(copies[index].sql);
        }
        destination.execute("COMMIT");
        destination.execute("DETACH oldDb");

        for(const master_row_t& index : indexes)
        {
            if(!mentions_omitted(index.sql, omitted)) destination.execute(index.sql);
        }
    } catch(const std::exception& err)
    {
        std::filesystem::remove(output_path, error);
        if(cancelled(future)) return PH_RESULT_CANCELLED;
        LOG_ERROR("Trace trim failed: {}", err.what());
        return PH_RESULT_INTERNAL_ERROR;
    }
    report(future, 1.0, "Trace trimmed");
    return PH_RESULT_SUCCESS;
}

}  // namespace profiler_hub
