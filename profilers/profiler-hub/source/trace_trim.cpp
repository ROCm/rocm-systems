// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "trace_trim.hpp"

#include "ph_future.hpp"
#include "sql_cancel.hpp"
#include "trim_schema.hpp"

#include "common/debug.hpp"
#include "data_storage/backends/sqlite_api_policy.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace profiler_hub
{

std::string
table_guid(std::string_view table)
{
    // Longer names first, so rocpd_pmc_event is not read as rocpd_event.
    constexpr std::string_view k_prefixes[] = {
        "rocpd_pmc_event",
        "rocpd_kernel_dispatch",
        "rocpd_memory_allocate",
        "rocpd_memory_copy",
        "rocpd_hip_event",
        "rocpd_region",
        "rocpd_sample",
        "rocpd_event",
        "rocpd_arg",
    };
    for(const std::string_view prefix : k_prefixes)
    {
        if(table.size() <= prefix.size() + 1 || table[prefix.size()] != '_') continue;
        if(table.substr(0, prefix.size()) != prefix) continue;
        return std::string{ table.substr(prefix.size() + 1) };
    }
    return {};
}

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

class sqlite_connection_t;
void
interrupt_trim_connection(void* connection) noexcept;

struct planned_copy_t
{
    std::string name;
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
        const profiler_hub::running_sql activity{ interrupt_trim_connection,
                                                  const_cast<sqlite_connection_t*>(this) };
        const int                       rc = sqlite_api_policy::exec(m_db, sql.c_str());
        if(rc == sqlite_api_policy::result_interrupt)
        {
            throw profiler_hub::sql_interrupted{};
        }
        if(rc != sqlite_api_policy::result_ok)
        {
            throw std::runtime_error(sqlite_api_policy::errmsg(m_db));
        }
    }

private:
    sqlite_api_policy::database_t m_db{ nullptr };
};

void
interrupt_trim_connection(void* connection) noexcept
{
    sqlite_api_policy::interrupt(static_cast<sqlite_connection_t*>(connection)->get());
}

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

std::string
quote_ident(std::string_view name)
{
    std::string quoted = "\"";
    for(const char ch : name)
    {
        if(ch == '"') quoted += "\"\"";
        else quoted += ch;
    }
    quoted += '"';
    return quoted;
}

std::string
union_view_sql(std::string_view name, const std::vector<std::string>& tables)
{
    std::string sql = "CREATE VIEW " + quote_ident(name) + " AS ";
    for(size_t index = 0; index < tables.size(); ++index)
    {
        if(index != 0) sql += " UNION ALL ";
        sql += "SELECT * FROM " + quote_ident(tables[index]);
    }
    return sql;
}

bool
starts_with_table_prefix(std::string_view table, std::string_view view)
{
    if(table.size() <= view.size() + 1 || table[view.size()] != '_') return false;
    return table.substr(0, view.size()) == view;
}

struct source_window_t
{
    std::string uuid;
    uint64_t    start_ts{};
    uint64_t    end_ts{};
};

struct file_group_t
{
    std::string                     path;
    std::vector<source_window_t>    windows;
    std::unordered_set<std::string> kept_tables;
    std::unordered_set<std::string> omitted;
    std::vector<master_row_t>       tables;
    std::vector<master_row_t>       views;
    std::vector<master_row_t>       indexes;
};

std::pair<uint64_t, uint64_t>
window_for(const file_group_t& group, std::string_view table)
{
    const std::string guid = table_guid(table);
    if(!guid.empty())
    {
        for(const source_window_t& window : group.windows)
        {
            if(window.uuid == guid) return { window.start_ts, window.end_ts };
        }
    }
    return { group.windows.front().start_ts, group.windows.front().end_ts };
}

void
classify(file_group_t& group, const std::vector<master_row_t>& master)
{
    for(const master_row_t& row : master)
    {
        if(is_internal_name(row.name) || is_internal_name(row.table_name))
        {
            group.omitted.insert(row.name);
            group.omitted.insert(row.table_name);
            continue;
        }
        if(row.type == "table")
        {
            group.kept_tables.insert(row.name);
            group.tables.push_back(row);
        }
        else if(row.type == "view") group.views.push_back(row);
        else if(row.type == "index" || row.type == "trigger") group.indexes.push_back(row);
    }
}

}  // namespace

ph_result_t
trim_trace_files(const std::vector<trim_source_t>& inputs,
                 const std::string&                output_path,
                 reader_types::trace_schema_t      schema,
                 version_t                         version,
                 ph_future*                        future)
{
    if(inputs.empty() || output_path.empty()) return PH_RESULT_INVALID_ARGUMENT;

    std::error_code error;
    const bool      output_exists = std::filesystem::exists(output_path, error);
    if(error) return PH_RESULT_INVALID_ARGUMENT;
    for(const trim_source_t& input : inputs)
    {
        if(input.path.empty()) return PH_RESULT_INVALID_ARGUMENT;
        if(output_exists && std::filesystem::equivalent(input.path, output_path, error))
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }
    }

    const std::unique_ptr<trim_schema_t> trim_schema = make_trim_schema(schema, version);
    if(trim_schema == nullptr) return PH_RESULT_UNSUPPORTED_FILE;

    std::filesystem::remove(output_path, error);
    if(error) return PH_RESULT_INVALID_FILE;

    try
    {
        std::vector<file_group_t> groups;
        for(const trim_source_t& input : inputs)
        {
            file_group_t* group = nullptr;
            for(file_group_t& existing : groups)
            {
                if(existing.path == input.path)
                {
                    group = &existing;
                    break;
                }
            }
            if(group == nullptr)
            {
                groups.push_back(file_group_t{});
                group       = &groups.back();
                group->path = input.path;
                classify(*group, read_master(sqlite_connection_t{ group->path, true }));
            }
            group->windows.push_back(
                source_window_t{ input.uuid, input.start_ts, input.end_ts });
        }

        // Later phases read rows an earlier phase wrote, including rows from
        // another file that landed in the same table. Finish one phase for
        // every file before starting the next.
        constexpr int phase_count = static_cast<int>(row_kind_t::event_child) + 1;
        std::vector<std::array<std::vector<planned_copy_t>, phase_count>> plans(groups.size());
        size_t total = 0;
        for(size_t index = 0; index < groups.size(); ++index)
        {
            for(const master_row_t& table : groups[index].tables)
            {
                const auto [start_ts, end_ts] = window_for(groups[index], table.name);
                trim_statement_t statement    = trim_schema->statement_for(
                    table.name, groups[index].kept_tables, start_ts, end_ts);
                const int phase = static_cast<int>(statement.kind);
                plans[index][phase].push_back(
                    planned_copy_t{ table.name, std::move(statement.sql) });
                ++total;
            }
            for(auto& phase : plans[index])
            {
                std::sort(phase.begin(), phase.end(), [](const planned_copy_t& left,
                                                          const planned_copy_t& right) {
                    return left.name < right.name;
                });
            }
        }

        std::vector<master_row_t>       views_in_order;
        std::unordered_set<std::string> seen_views;
        std::unordered_set<std::string> all_omitted;
        for(const file_group_t& group : groups)
        {
            all_omitted.insert(group.omitted.begin(), group.omitted.end());
            for(const master_row_t& view : group.views)
            {
                if(seen_views.insert(view.name).second) views_in_order.push_back(view);
            }
        }

        sqlite_connection_t destination{ output_path, false };
        destination.execute("PRAGMA foreign_keys = OFF");
        report(future, 0.05, "Creating trimmed database");

        std::unordered_set<std::string> created_tables;
        for(const file_group_t& group : groups)
        {
            for(const master_row_t& table : group.tables)
            {
                if(!created_tables.insert(table.name).second) continue;
                destination.execute(table.sql);
            }
        }

        size_t       done = 0;
        const double span = total == 0 ? 0.0 : 0.9 / static_cast<double>(total);
        for(int phase = 0; phase < phase_count; ++phase)
        {
            for(size_t index = 0; index < groups.size(); ++index)
            {
                const std::vector<planned_copy_t>& copies = plans[index][phase];
                if(copies.empty()) continue;

                destination.execute("BEGIN");
                destination.execute(
                    attach_sql(std::filesystem::path{ groups[index].path }.generic_string()));
                for(const planned_copy_t& copy : copies)
                {
                    if(cancelled(future))
                    {
                        destination.execute("ROLLBACK");
                        throw std::runtime_error("cancelled");
                    }
                    report(future, 0.05 + span * static_cast<double>(done), copy.name.c_str());
                    destination.execute(copy.sql);
                    ++done;
                }
                destination.execute("COMMIT");
                destination.execute("DETACH oldDb");
            }
        }

        std::unordered_map<std::string, std::vector<std::string>> parts_of;
        for(const std::string& table : created_tables)
        {
            const master_row_t* best = nullptr;
            for(const master_row_t& view : views_in_order)
            {
                if(!starts_with_table_prefix(table, view.name)) continue;
                if(best == nullptr || view.name.size() > best->name.size()) best = &view;
            }
            if(best != nullptr) parts_of[best->name].push_back(table);
        }

        std::unordered_set<std::string> made_views;
        for(const master_row_t& view : views_in_order)
        {
            const auto found = parts_of.find(view.name);
            if(found == parts_of.end() || created_tables.count(view.name) != 0) continue;

            bool mentioned = false;
            for(const std::string& part : found->second)
            {
                if(view.sql.find(part) != std::string::npos)
                {
                    mentioned = true;
                    break;
                }
            }
            if(!mentioned) continue;

            std::vector<std::string> parts = found->second;
            std::sort(parts.begin(), parts.end());
            destination.execute(union_view_sql(view.name, parts));
            made_views.insert(view.name);
        }
        for(const master_row_t& view : views_in_order)
        {
            if(made_views.count(view.name) != 0 || created_tables.count(view.name) != 0)
            {
                continue;
            }
            if(mentions_omitted(view.sql, all_omitted)) continue;
            destination.execute(view.sql);
            made_views.insert(view.name);
        }

        std::unordered_set<std::string> created_indexes;
        for(const file_group_t& group : groups)
        {
            for(const master_row_t& index : group.indexes)
            {
                if(mentions_omitted(index.sql, all_omitted)) continue;
                if(!created_indexes.insert(index.name).second) continue;
                destination.execute(index.sql);
            }
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

ph_result_t
trim_trace_file(const std::string&           source_path,
                const std::string&           output_path,
                reader_types::trace_schema_t schema,
                version_t                    version,
                uint64_t                     start_ts,
                uint64_t                     end_ts,
                ph_future*                   future)
{
    const trim_source_t source{ source_path, {}, start_ts, end_ts };
    return trim_trace_files({ source }, output_path, schema, version, future);
}

}  // namespace profiler_hub
