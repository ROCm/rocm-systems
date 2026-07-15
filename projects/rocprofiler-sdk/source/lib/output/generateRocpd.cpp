// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "generateRocpd.hpp"
#include "lib/common/uuid_v7.hpp"
#include "metadata.hpp"
#include "output_stream.hpp"
#include "pc_sample_transform.hpp"
#include "statistics.hpp"
#include "stream_info.hpp"
#include "timestamps.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/hasher.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/md5sum.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/simple_timer.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/sql/deferred_transaction.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/sql.h>
#include <rocprofiler-sdk-rocpd/types.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <sqlite3.h>
#include <unistd.h>

#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::argument_info& data)
{
    ar(cereal::make_nvp("pos", data.arg_number));
    ar(cereal::make_nvp("type", data.arg_type));
    ar(cereal::make_nvp("name", data.arg_name));
    ar(cereal::make_nvp("value", data.arg_value));
}

ROCPROFILER_SDK_CEREAL_NAMESPACE_END

namespace std
{
template <>
struct hash<::rocprofiler::tool::track_data>
{
    size_t operator()(const ::rocprofiler::tool::track_data& val) const { return val.hash(); }
};
}  // namespace std

namespace rocprofiler
{
namespace tool
{
namespace
{
namespace fs          = ::rocprofiler::common::filesystem;
using function_args_t = std::vector<argument_info>;

// Wrapper to distinguish raw BLOB bytes from TEXT strings in sql_insert_value.
struct blob_data
{
    std::string bytes;
};

struct sql_insert_value
{
    std::string_view name = {};
    std::variant<std::monostate, int64_t, uint64_t, double, std::string, std::nullptr_t, blob_data>
        value = {};
};

struct pending_insert_batch
{
    std::string                                table        = {};
    std::vector<std::string>                   fields       = {};
    std::vector<std::vector<sql_insert_value>> rows         = {};
    size_t                                     max_rows     = 1;
    uint64_t                                   last_touched = 0;
    bool                                       has_blob     = false;
};

struct rocpd_db
{
    using statement_cache_t   = std::unordered_map<std::string, sqlite3_stmt*>;
    using schema_map_t        = std::unordered_map<rocpd_sql_schema_kind_t, std::string>;
    using track_map_t         = std::unordered_map<track_data, uint64_t>;
    using batch_stats_t       = statistics<uint64_t, double>;
    using batch_stats_map_t   = std::unordered_map<std::string, batch_stats_t>;
    using pending_batch_map_t = std::unordered_map<std::string, pending_insert_batch>;
    using fk_parent_map_t     = std::unordered_map<std::string, std::vector<std::string>>;
    using flushing_set_t      = std::unordered_set<std::string>;

    static constexpr size_t max_pending_batches = 6;

    rocpd_db() = default;
    ~rocpd_db();
    rocpd_db(const rocpd_db&) = delete;
    rocpd_db& operator=(const rocpd_db&) = delete;
    rocpd_db(rocpd_db&&)                 = delete;
    rocpd_db& operator=(rocpd_db&&) = delete;

    sqlite3*            conn                = nullptr;
    std::string         uuid                = {};
    std::string         guid                = {};
    schema_map_t        schemas             = {};
    track_map_t         tracks              = {};
    size_t              event_id_counter    = 0;
    size_t              sample_id_counter   = 0;
    statement_cache_t   statements          = {};
    pending_batch_map_t pending_batches     = {};
    fk_parent_map_t     fk_parent_tables    = {};
    flushing_set_t      flushing_tables     = {};
    uint64_t            pending_touch_count = 0;
    batch_stats_map_t   batch_stats         = {};

    // Recycled storage for fixed-size BLOB payloads (e.g. PC sampling extdata).  Batching
    // needs each blob to have stable storage until its row is flushed, so rather than a fresh
    // heap allocation per row we reuse buffers returned by already-flushed batches.
    std::vector<std::string> blob_buffer_pool     = {};
    static constexpr size_t  max_blob_buffer_pool = 2048;

    size_t get_event_id() { return ++event_id_counter; }
    size_t get_sample_id() { return ++sample_id_counter; }

    // Acquire an empty pooled buffer (no copy); the caller fills it in place, e.g. when
    // assembling a PC-sample blob field-by-field.
    std::string acquire_blob_buffer()
    {
        auto buf = std::string{};
        if(!blob_buffer_pool.empty())
        {
            buf = std::move(blob_buffer_pool.back());
            blob_buffer_pool.pop_back();
        }
        buf.clear();
        return buf;
    }

    // Return a blob buffer for reuse once its batch has been fully stepped and cleared.
    void recycle_blob_buffer(std::string&& buf)
    {
        if(blob_buffer_pool.size() >= max_blob_buffer_pool) return;
        buf.clear();
        blob_buffer_pool.emplace_back(std::move(buf));
    }
};

void
finish_pending_insert_batch(rocpd_db& db);
void
flush_pending_insert_batch(rocpd_db& db, pending_insert_batch& pending);
std::string
normalize_batch_table_name(const rocpd_db& db, std::string_view table);
void
log_batch_flush_stats(const rocpd_db& db);

template <typename Tp>
auto
replace_all(std::string val, Tp from, std::string_view to)
{
    size_t pos = 0;
    while((pos = val.find(from, pos)) != std::string::npos)
    {
        if constexpr(std::is_same<common::mpl::unqualified_type_t<Tp>, char>::value)
        {
            val.replace(pos, 1, to);
            pos += to.length();
        }
        else
        {
            val.replace(pos, std::string_view{from}.length(), to);
            pos += to.length();
        }
    }
    return val;
}

template <typename FuncT, typename... Args>
std::string
get_json_string(FuncT&& _func, Args&&... _args)
{
    using json_archive         = cereal::MinimalJSONOutputArchive;
    constexpr auto json_prec   = 16;
    constexpr auto json_indent = json_archive::Options::IndentChar::space;

    auto json_ss = std::stringstream{};

    {
        auto json_opts = json_archive::Options{json_prec, json_indent, 0};
        auto ar        = json_archive{json_ss, json_opts};
        std::forward<FuncT>(_func)(ar, std::forward<Args>(_args)...);
    }

    return json_ss.str();
}

template <typename Tp>
size_t
get_hash_id(Tp&& _val)
{
    using value_type = common::mpl::unqualified_type_t<Tp>;

    if constexpr(!std::is_pointer<Tp>::value)
        return std::hash<value_type>{}(std::forward<Tp>(_val));
    else if constexpr(std::is_same<Tp, const char*>::value)
        return get_hash_id(std::string_view{_val});
    else
        return get_hash_id(*_val);
}

auto
replace_uuid(const rocpd_db& db, std::string_view inp)
{
    return replace_all(std::string{inp}, std::string_view{"{{uuid}}"}, db.uuid);
}

template <typename... Args>
void
add_string_entry(metadata& _metadata, Args&&... _args)
{
    (_metadata.add_string_entry(get_hash_id(std::string_view{std::forward<Args>(_args)}),
                                std::string_view{std::forward<Args>(_args)}),
     ...);
}

void
read_file(rocpd_sql_engine_t                        engine,
          rocpd_sql_schema_kind_t                   kind,
          rocpd_sql_options_t                       options,
          rocpd_version_triplet_t                   schema_version,
          const rocpd_sql_schema_jinja_variables_t* variables,
          const char*                               schema_path,
          const char*                               schema_content,
          void*                                     user_data)
{
    common::consume_args(engine, kind, options, schema_version, variables, schema_path);

    auto* _db     = static_cast<rocpd_db*>(user_data);
    auto& _schema = _db->schemas[kind];
    _schema       = replace_uuid(*_db, schema_content);
}

std::string
read_schema_file(rocpd_db& db, rocpd_sql_schema_kind_t schema_kind)
{
    auto _variables = common::init_public_api_struct(rocpd_sql_schema_jinja_variables_t{});
    auto _options   = ROCPD_SQL_OPTIONS_NONE;
    auto _version   = rocpd_version_triplet_t{3, 0, 4};  // default schema version

    _variables.uuid = db.uuid.c_str();
    _variables.guid = db.guid.c_str();

    ROCPD_CHECK(rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                      schema_kind,
                                      _options,
                                      _version,
                                      &_variables,
                                      read_file,
                                      nullptr,
                                      0,
                                      &db));

    return db.schemas.at(schema_kind);
}

int
iterate_args_callback(rocprofiler_buffer_tracing_kind_t /*kind*/,
                      rocprofiler_tracing_operation_t /*operation*/,
                      uint32_t arg_number,
                      const void* const /*arg_value_addr*/,
                      int32_t /*arg_indirection_count*/,
                      const char* arg_type,
                      const char* arg_name,
                      const char* arg_value_str,
                      void*       data)
{
    ROCP_FATAL_IF(data == nullptr) << "nullptr to data for iterate_args_callback";

    auto* _data = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
    {
        _data->emplace_back(
            argument_info{arg_number, common::cxx_demangle(arg_type), arg_name, arg_value_str});
    }
    return 0;
}

struct allow_empty_string
{};

template <typename Tp>
struct is_optional : std::false_type
{
    using value_type = Tp;
};

template <typename Tp>
struct is_optional<std::optional<Tp>> : std::true_type
{
    using value_type = Tp;
};

template <typename Tp, typename TraitT = int>
sql_insert_value
insert_value(std::string_view _name, const Tp& _value, TraitT = {})
{
    using value_type = common::mpl::unqualified_type_t<Tp>;

    static_assert(!is_optional<value_type>::value, "overload resolution failed");

    if constexpr(common::mpl::is_string_type<value_type>::value)
    {
        if constexpr(std::is_pointer<value_type>::value)
        {
            if(!_value)
            {
                return insert_value(_name, std::string_view{}, TraitT{});
            }
        }

        if constexpr(!std::is_same<TraitT, allow_empty_string>::value)
        {
            if(std::string_view{_value}.empty())
            {
                ROCP_CI_LOG(WARNING)
                    << fmt::format("sql text value for {} is empty. Using NULL instead", _name);
                return sql_insert_value{_name, nullptr};
            }
        }
        return sql_insert_value{_name, std::string{_value}};
    }
    else if constexpr(std::is_enum_v<value_type>)
    {
        return sql_insert_value{_name, static_cast<int64_t>(_value)};
    }
    else if constexpr(std::is_integral_v<value_type>)
    {
        if constexpr(std::is_unsigned_v<value_type>)
        {
            auto _uval = static_cast<uint64_t>(_value);
            if(_uval > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                return sql_insert_value{_name, fmt::format("{}", _uval)};
            return sql_insert_value{_name, _uval};
        }
        else
            return sql_insert_value{_name, static_cast<int64_t>(_value)};
    }
    else if constexpr(std::is_floating_point_v<value_type>)
    {
        return sql_insert_value{_name, static_cast<double>(_value)};
    }
    else
    {
        return sql_insert_value{_name, fmt::format("{}", _value)};
    }
}

//
// this overload works around around a (false) maybe-initialized warning in GCC 12.x
//
template <typename Tp, typename TraitT = int>
ROCPROFILER_NOINLINE sql_insert_value
insert_value(std::string_view _name, const std::optional<Tp>& _value, TraitT = {})
{
    if(!_value.has_value()) return sql_insert_value{};

    return insert_value(_name, _value.value(), TraitT{});
}

template <typename Tp, typename TraitT = int>
ROCPROFILER_NOINLINE sql_insert_value
insert_nullable_value(std::string_view _name, const std::optional<Tp>& _value, TraitT = {})
{
    if(!_value.has_value()) return sql_insert_value{_name, nullptr};
    return insert_value(_name, _value.value(), TraitT{});
}

// Store raw bytes as a SQLite BLOB column (not TEXT).
ROCPROFILER_NOINLINE sql_insert_value
insert_blob_value(std::string_view _name, std::string bytes)
{
    return sql_insert_value{_name, blob_data{std::move(bytes)}};
}

rocpd_db::~rocpd_db()
{
    finish_pending_insert_batch(*this);
    log_batch_flush_stats(*this);

    for(auto& [key, stmt] : statements)
    {
        if(stmt) SQLITE3_CHECK(sqlite3_finalize(stmt));
    }
    statements.clear();

    if(conn)
    {
        SQLITE3_CHECK(sqlite3_close_v2(conn));
        conn = nullptr;
    }
}

size_t
get_max_batch_rows(sqlite3* conn, size_t col_count)
{
    constexpr auto default_limit = 999;
    auto           var_limit     = sqlite3_limit(conn, SQLITE_LIMIT_VARIABLE_NUMBER, -1);
    auto           safe_cols     = std::max<size_t>(1, col_count);
    auto           max_rows      = static_cast<size_t>((var_limit > 0 ? var_limit : default_limit) /
                                        static_cast<int>(safe_cols));
    return std::clamp<size_t>(max_rows, 1, 1024);
}

int
bind_sql_value(sqlite3_stmt* stmt, int idx, const sql_insert_value& value)
{
    return std::visit(
        [&](auto&& val) {
            using value_type = common::mpl::unqualified_type_t<decltype(val)>;
            if constexpr(std::is_same_v<value_type, std::monostate> ||
                         std::is_same_v<value_type, std::nullptr_t>)
            {
                return sqlite3_bind_null(stmt, idx);
            }
            else if constexpr(std::is_same_v<value_type, int64_t>)
            {
                return sqlite3_bind_int64(stmt, idx, val);
            }
            else if constexpr(std::is_same_v<value_type, uint64_t>)
            {
                if(val <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                    return sqlite3_bind_int64(stmt, idx, static_cast<int64_t>(val));
                // Values > INT64_MAX cannot be stored as SQLite INTEGER without wrapping;
                // bind as TEXT to preserve the exact value.  This is a local temporary (unlike
                // the batched std::string/blob values below), so it must remain SQLITE_TRANSIENT.
                auto text = fmt::format("{}", val);
                return sqlite3_bind_text(
                    stmt, idx, text.c_str(), static_cast<int>(text.size()), SQLITE_TRANSIENT);
            }
            else if constexpr(std::is_same_v<value_type, double>)
            {
                return sqlite3_bind_double(stmt, idx, val);
            }
            else if constexpr(common::mpl::is_string_type<value_type>::value)
            {
                // SQLITE_STATIC: the std::string lives in pending.rows and is neither modified
                // nor freed until after sqlite3_step (rows are cleared only once the whole batch
                // has been stepped), so SQLite may reference it in place instead of copying.
                return sqlite3_bind_text(stmt, idx, val.c_str(), -1, SQLITE_STATIC);
            }
            else if constexpr(std::is_same_v<value_type, blob_data>)
            {
                // SQLITE_STATIC: blob bytes also live in pending.rows until after the step.
                return sqlite3_bind_blob64(
                    stmt, idx, val.bytes.data(), val.bytes.size(), SQLITE_STATIC);
            }
            else
            {
                static_assert(common::mpl::assert_false<value_type>::value,
                              "Unsupported data type");
            }
        },
        value.value);
}

void
reset_and_clear_statement(sqlite3_stmt* stmt)
{
    if(!stmt) return;
    SQLITE3_CHECK(sqlite3_reset(stmt));
    SQLITE3_CHECK(sqlite3_clear_bindings(stmt));
}

std::string
normalize_batch_table_name(const rocpd_db& db, std::string_view table)
{
    if(db.uuid.empty()) return std::string{table};

    auto suffix = db.uuid;
    if(table.size() > suffix.size() &&
       table.substr(table.size() - suffix.size(), suffix.size()) == suffix)
    {
        return std::string{table.substr(0, table.size() - suffix.size())};
    }

    return std::string{table};
}

std::string
quote_sql_identifier(std::string_view value)
{
    // SQLite accepts backtick-quoted identifiers; escape embedded backticks by doubling.
    return fmt::format("`{}`", replace_all(std::string{value}, '`', "``"));
}

void
log_batch_flush_stats(const rocpd_db& db)
{
    // Emit batch stats only when diagnostics/trace logging is enabled.
    if(!VLOG_IS_ON(ROCP_LOG_LEVEL_TRACE)) return;
    if(db.batch_stats.empty()) return;

    auto tables = std::vector<std::string>{};
    tables.reserve(db.batch_stats.size());
    for(const auto& [table, stats] : db.batch_stats)
    {
        if(stats.get_count() <= 0) continue;
        tables.emplace_back(table);
    }

    if(tables.empty()) return;

    std::sort(tables.begin(), tables.end());

    ROCP_TRACE << "ROCPD batch flush statistics by table:";
    for(const auto& table : tables)
    {
        const auto& stats = db.batch_stats.at(table);
        ROCP_TRACE << fmt::format(
            "  {}: flushes={}, min={}, max={}, mean={:.2f}, stddev={:.2f}, variance={:.2f}, "
            "rows_total={}",
            table,
            stats.get_count(),
            stats.get_min(),
            stats.get_max(),
            stats.get_mean(),
            stats.get_stddev(),
            stats.get_variance(),
            stats.get_sum());
    }
}

sqlite3_stmt*&
get_or_prepare_batch_statement(rocpd_db&                   db,
                               const pending_insert_batch& pending,
                               size_t                      rows_per_exec)
{
    const auto batch_key =
        fmt::format("{}:batch:{}:{}", pending.table, rows_per_exec, fmt::join(pending.fields, ","));

    auto& stmt = db.statements[batch_key];
    if(stmt) return stmt;

    const auto placeholders     = std::vector(pending.fields.size(), std::string{"?"});
    const auto row_placeholder  = fmt::format("({})", fmt::join(placeholders, ", "));
    const auto row_placeholders = std::vector(rows_per_exec, row_placeholder);
    const auto values_clause    = fmt::format("{}", fmt::join(row_placeholders, ", "));
    auto       quoted_fields    = std::vector<std::string>{};
    quoted_fields.reserve(pending.fields.size());
    for(const auto& field : pending.fields)
        quoted_fields.emplace_back(quote_sql_identifier(field));

    const auto table_name  = quote_sql_identifier(pending.table);
    const auto field_names = fmt::format("{}", fmt::join(quoted_fields, ", "));

    const auto sql =
        fmt::format("INSERT INTO {} ({}) VALUES {};", table_name, field_names, values_clause);

    auto prepare_rc = sqlite3_prepare_v2(db.conn, sql.c_str(), -1, &stmt, nullptr);
    ROCP_FATAL_IF(prepare_rc != SQLITE_OK)
        << "sqlite3_prepare_v2 failed with error code " << prepare_rc
        << ", sqlite3_errmsg: " << sqlite3_errmsg(db.conn) << ", table: " << pending.table
        << ", rows_per_exec: " << rows_per_exec << ", field_count: " << pending.fields.size()
        << ", sql: " << sql;
    return stmt;
}

size_t
get_bucketed_rows_per_exec(size_t remaining_rows, size_t max_rows)
{
    auto capped_rows = std::min(remaining_rows, max_rows);
    if(capped_rows <= 1) return capped_rows;

    // Keep statement cache bounded by using powers-of-two for remainder chunks.
    // Full-size batches still use max_rows directly.
    if(capped_rows == max_rows) return max_rows;

    size_t bucket = 1;
    while((bucket << 1) <= capped_rows)
        bucket <<= 1;

    return bucket;
}

const std::vector<std::string>&
get_fk_parent_tables(rocpd_db& db, const std::string& table)
{
    if(auto itr = db.fk_parent_tables.find(table); itr != db.fk_parent_tables.end())
        return itr->second;

    auto parents = std::vector<std::string>{};

    auto escaped_table = replace_all(table, '\'', "''");
    auto pragma_query  = fmt::format("PRAGMA foreign_key_list('{}');", escaped_table);

    sqlite3_stmt* stmt = nullptr;
    SQLITE3_CHECK(sqlite3_prepare_v2(db.conn, pragma_query.c_str(), -1, &stmt, nullptr));

    while(true)
    {
        auto step_rc = sqlite3_step(stmt);
        if(step_rc == SQLITE_DONE) break;
        ROCP_FATAL_IF(step_rc != SQLITE_ROW) << "sqlite3_step failed with error code " << step_rc
                                             << ", sqlite3_errmsg: " << sqlite3_errmsg(db.conn)
                                             << ", pragma_query: " << pragma_query;

        const auto* parent_table = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        if(parent_table == nullptr || parent_table[0] == '\0') continue;
        if(std::find(parents.begin(), parents.end(), parent_table) == parents.end())
            parents.emplace_back(parent_table);
    }

    auto finalize_rc = sqlite3_finalize(stmt);
    ROCP_FATAL_IF(finalize_rc != SQLITE_OK)
        << "sqlite3_finalize failed with error code " << finalize_rc
        << ", sqlite3_errmsg: " << sqlite3_errmsg(db.conn) << ", pragma_query: " << pragma_query;

    auto [itr, inserted] = db.fk_parent_tables.emplace(table, std::move(parents));
    common::consume_args(inserted);
    return itr->second;
}

rocpd_db::pending_batch_map_t::iterator
select_pending_batch_victim(rocpd_db& db)
{
    auto best_itr             = db.pending_batches.end();
    auto best_parent_pressure = std::numeric_limits<size_t>::max();
    auto best_touch           = std::numeric_limits<uint64_t>::max();

    for(auto itr = db.pending_batches.begin(); itr != db.pending_batches.end(); ++itr)
    {
        if(itr->second.rows.empty()) continue;

        auto parent_pressure = size_t{0};
        for(const auto& parent_table : get_fk_parent_tables(db, itr->first))
        {
            if(auto pitr = db.pending_batches.find(parent_table);
               pitr != db.pending_batches.end() && !pitr->second.rows.empty())
            {
                ++parent_pressure;
            }
        }

        if(best_itr == db.pending_batches.end() || parent_pressure < best_parent_pressure ||
           (parent_pressure == best_parent_pressure && itr->second.last_touched < best_touch))
        {
            best_itr             = itr;
            best_parent_pressure = parent_pressure;
            best_touch           = itr->second.last_touched;
        }
    }

    return best_itr;
}

void
flush_pending_insert_batch(rocpd_db& db, pending_insert_batch& pending)
{
    if(pending.rows.empty()) return;

    ROCP_FATAL_IF(db.conn == nullptr) << "Pending insert batch missing sqlite connection";
    ROCP_FATAL_IF(db.flushing_tables.count(pending.table) > 0)
        << "FK cycle detected: flush of '" << pending.table << "' is already in progress";

    db.flushing_tables.insert(pending.table);

    // Flush direct FK parents first so references are always materialized before child rows.
    // Parent flushes recurse; flushing_tables guards against cycles in the dependency graph.
    for(const auto& parent_table : get_fk_parent_tables(db, pending.table))
    {
        if(db.flushing_tables.count(parent_table) > 0) continue;
        if(auto pitr = db.pending_batches.find(parent_table);
           pitr != db.pending_batches.end() && !pitr->second.rows.empty())
        {
            flush_pending_insert_batch(db, pitr->second);
        }
    }

    const auto pending_table = normalize_batch_table_name(db, pending.table);

    const auto batch_size = pending.rows.size();
    const auto field_size = pending.fields.size();
    ROCP_FATAL_IF(field_size == 0)
        << "Pending insert batch has zero fields for table: " << pending.table;

    // Execute in multi-row chunks up to max_rows; this keeps partial flushes batched too.
    for(size_t begin_idx = 0; begin_idx < batch_size;)
    {
        const size_t remaining     = batch_size - begin_idx;
        const size_t rows_per_exec = get_bucketed_rows_per_exec(remaining, pending.max_rows);
        auto&        stmt          = get_or_prepare_batch_statement(db, pending, rows_per_exec);
        auto         end_idx       = begin_idx + rows_per_exec;
        reset_and_clear_statement(stmt);
        int bind_idx = 1;
        for(size_t idx = begin_idx; idx < end_idx; ++idx)
        {
            const auto& row = pending.rows[idx];
            ROCP_FATAL_IF(row.size() != field_size) << "Pending insert batch row has " << row.size()
                                                    << " values, expected " << field_size;
            for(const auto& value : row)
            {
                auto bind_rc = bind_sql_value(stmt, bind_idx++, value);
                ROCP_FATAL_IF(bind_rc != SQLITE_OK)
                    << "sqlite3_bind failed with error code " << bind_rc;
            }
        }

        auto step_rc = sqlite3_step(stmt);
        ROCP_FATAL_IF(step_rc != SQLITE_DONE)
            << "sqlite3_step failed with error code " << step_rc
            << ", sqlite3_errmsg: " << sqlite3_errmsg(db.conn) << ", table: " << pending.table
            << ", batch_size: " << batch_size
            << ", fields: " << fmt::format("{}", fmt::join(pending.fields, ", "));

        db.batch_stats[pending_table] += static_cast<uint64_t>(end_idx - begin_idx);
        begin_idx = end_idx;
    }

    // Return blob allocations to the pool for reuse.  Safe here: every row above has already
    // been stepped, moving the buffers only transfers ownership (the bytes stay valid), and the
    // statement is reset and re-bound before any further step.  Gated by has_blob so non-blob
    // batches keep their original clear() fast path.
    if(pending.has_blob)
    {
        for(auto& row : pending.rows)
            for(auto& value : row)
                if(auto* blob = std::get_if<blob_data>(&value.value))
                    db.recycle_blob_buffer(std::move(blob->bytes));
    }

    pending.rows.clear();
    db.flushing_tables.erase(pending.table);
}

void
finish_pending_insert_batch(rocpd_db& db)
{
    if(db.pending_batches.empty()) return;

    for(auto& [_, pending] : db.pending_batches)
        flush_pending_insert_batch(db, pending);

    db.pending_batches.clear();
}

void
get_insert_statement(rocpd_db& db, std::string_view _table, std::vector<sql_insert_value> _data)
{
    ROCP_FATAL_IF(db.conn == nullptr) << "SQLite connection not set for prepared statements";

    auto table = replace_uuid(db, _table);

    auto fields = std::vector<std::string>{};
    auto values = std::vector<sql_insert_value>{};
    fields.reserve(_data.size());
    values.reserve(_data.size());
    for(auto&& itr : _data)
    {
        if(itr.name.empty()) continue;
        fields.emplace_back(itr.name);
        values.emplace_back(std::move(itr));
    }

    if(fields.empty()) return;

    // Hot path: existing active batch for this table.
    if(auto itr = db.pending_batches.find(table); itr != db.pending_batches.end())
    {
        auto& pending = itr->second;

        if(pending.fields != fields)
        {
            flush_pending_insert_batch(db, pending);
            pending.fields   = std::move(fields);
            pending.max_rows = get_max_batch_rows(db.conn, pending.fields.size());
            pending.has_blob = std::any_of(values.begin(), values.end(), [](const auto& v) {
                return std::holds_alternative<blob_data>(v.value);
            });
            pending.rows.clear();
            pending.rows.reserve(pending.max_rows);
        }

#if !defined(NDEBUG)
        ROCP_FATAL_IF(values.size() != pending.fields.size())
            << "Pending insert batch row has " << values.size() << " values, expected "
            << pending.fields.size() << " for table: " << table;
#endif

        pending.last_touched = ++db.pending_touch_count;
        pending.rows.emplace_back(std::move(values));
        if(pending.rows.size() >= pending.max_rows) flush_pending_insert_batch(db, pending);
        return;
    }

    const auto live_count = std::count_if(db.pending_batches.begin(),
                                          db.pending_batches.end(),
                                          [](const auto& kv) { return !kv.second.rows.empty(); });
    if(static_cast<size_t>(live_count) >= rocpd_db::max_pending_batches)
    {
        auto victim = select_pending_batch_victim(db);

        if(victim != db.pending_batches.end())
        {
            flush_pending_insert_batch(db, victim->second);
            db.pending_batches.erase(victim);
        }
    }

    auto pending         = pending_insert_batch{};
    pending.table        = table;
    pending.fields       = std::move(fields);
    pending.max_rows     = get_max_batch_rows(db.conn, pending.fields.size());
    pending.last_touched = ++db.pending_touch_count;
    pending.has_blob     = std::any_of(values.begin(), values.end(), [](const auto& v) {
        return std::holds_alternative<blob_data>(v.value);
    });
    pending.rows.reserve(pending.max_rows);
    pending.rows.emplace_back(std::move(values));

    auto [batch_itr, inserted] = db.pending_batches.emplace(table, std::move(pending));
    common::consume_args(inserted);

    if(batch_itr->second.rows.size() >= batch_itr->second.max_rows)
        flush_pending_insert_batch(db, batch_itr->second);
}

auto
create_event(rocpd_db& _db, std::initializer_list<sql_insert_value> _data)
{
    auto evt_id = _db.get_event_id();
    auto data   = std::vector<sql_insert_value>{insert_value("id", evt_id)};
    data.insert(data.end(), _data.begin(), _data.end());
    get_insert_statement(_db, "rocpd_event{{uuid}}", std::move(data));
    return evt_id;
}

uint64_t
get_track_id(rocpd_db&        _db,
             uint64_t         node_id,
             pid_t            pid,
             pid_t            tid,
             uint64_t         name_id,
             std::string_view extdata)
{
    auto _track = track_data{node_id, pid, tid, name_id};
    auto itr    = _db.tracks.find(_track);
    if(itr == _db.tracks.end())
    {
        auto idx = _db.tracks.size() + 1;
        itr      = _db.tracks.emplace(_track, idx).first;
        get_insert_statement(_db,
                             "rocpd_track{{uuid}}",
                             {
                                 insert_value("id", idx),
                                 insert_value("nid", node_id),
                                 insert_value("pid", pid),
                                 insert_value("tid", tid),
                                 insert_value("name_id", name_id),
                                 insert_value("extdata", extdata),
                             });
        return idx;
    }

    return itr->second;
}

}  // namespace

size_t
track_data::hash() const
{
    return get_hash_id(fmt::format("{:#018x}{:#018x}{:#018x}{:#018x}", node_id, pid, tid, name_id));
}

bool
operator==(const track_data& lhs, const track_data& rhs)
{
    return std::tie(lhs.node_id, lhs.pid, lhs.tid, lhs.name_id) ==
           std::tie(rhs.node_id, rhs.pid, rhs.tid, rhs.name_id);
}

namespace
{
constexpr auto MEMORY_PREFIX  = std::string_view{"MEMORY_ALLOCATION_"};
constexpr auto SCRATCH_PREFIX = std::string_view{"SCRATCH_MEMORY_"};
constexpr auto VMEM_PREFIX    = std::string_view{"VMEM_"};
constexpr auto ASYNC_PREFIX   = std::string_view{"ASYNC_"};

std::pair<std::string, std::string>
memtype_to_db(std::string_view memory_type)
{
    std::string _type;
    std::string _level;
    if(memory_type.find(MEMORY_PREFIX) == 0)
    {
        _type = memory_type.substr(MEMORY_PREFIX.length());
        if(_type.find(VMEM_PREFIX) == 0)
        {
            _type  = _type.substr(VMEM_PREFIX.length());
            _level = "VIRTUAL";
        }
        else
        {
            _level = "REAL";
        }
    }
    else if(memory_type.find(SCRATCH_PREFIX) == 0)
    {
        _type  = memory_type.substr(SCRATCH_PREFIX.length());
        _level = "SCRATCH";
        if(memory_type.find(ASYNC_PREFIX) == 0)
        {
            _type = memory_type.substr(ASYNC_PREFIX.length());  // RECLAIM
        }
    }

    if(_type == "ALLOCATE")
    {
        _type = "ALLOC";
    }

    return std::make_pair(_type, _level);
}

template <typename Tp, typename Up = Tp>
auto
extract_flags_field(const Tp& _data, int) -> decltype(std::declval<Up>().flags, std::string{})
{
    return std::to_string(static_cast<int>(_data.flags));
}

template <typename Tp, typename Up = Tp>
std::string
extract_flags_field(const Tp&, long)
{
    return "";
}

std::optional<std::string>
get_code_object_storage_type(rocprofiler_code_object_storage_type_t storage_type)
{
    switch(storage_type)
    {
        case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: return "FILE";
        case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: return "MEMORY";
        default: return std::nullopt;
    }
}

template <typename Tp>
std::string
extract_flags_field(const Tp& _data)
{
    return extract_flags_field(_data, 0);
}

#define GENERATE_FIELD_ACCESSOR(FUNC_NAME, FIELD_NAME, DATA_TYPE, ...)                             \
    template <typename Tp, typename Up = Tp>                                                       \
    auto FUNC_NAME(const Tp& _data, int)->decltype(std::declval<Up>().FIELD_NAME, DATA_TYPE{})     \
    {                                                                                              \
        return _data.FIELD_NAME;                                                                   \
    }                                                                                              \
                                                                                                   \
    template <typename Tp, typename Up = Tp>                                                       \
    DATA_TYPE FUNC_NAME(const Tp&, long)                                                           \
    {                                                                                              \
        return DATA_TYPE{__VA_ARGS__};                                                             \
    }                                                                                              \
                                                                                                   \
    template <typename Tp>                                                                         \
    DATA_TYPE FUNC_NAME(const Tp& _data)                                                           \
    {                                                                                              \
        return FUNC_NAME(_data, 0);                                                                \
    }

GENERATE_FIELD_ACCESSOR(extract_stream_field, stream_id, rocprofiler_stream_id_t, 0)
GENERATE_FIELD_ACCESSOR(extract_queue_field, queue_id, rocprofiler_queue_id_t, 0)
GENERATE_FIELD_ACCESSOR(extract_allocation_size_field, allocation_size, uint64_t, 0)
GENERATE_FIELD_ACCESSOR(extract_address_field, address, rocprofiler_address_t, 0)
}  // namespace

namespace
{
struct kfd_pmc_event_data_t
{
    rocprofiler_buffer_tracing_kind_t kind      = ROCPROFILER_BUFFER_TRACING_NONE;
    std::string_view                  name      = {};
    uint32_t                          tid       = 0;
    rocprofiler_timestamp_t           start     = 0;
    rocprofiler_timestamp_t           end       = 0;
    uint64_t                          value     = 0;
    std::string                       json_data = {};
};

namespace impl
{
template <typename Tp>
struct rocpd_kfd_wrapper;
}

#define GENERATE_KFD_TRAIT_MAPPING(KFD_TYPE)                                                       \
    template <>                                                                                    \
    struct impl::rocpd_kfd_wrapper<rocprofiler_buffer_tracing_##KFD_TYPE>                          \
    {                                                                                              \
        using type = rocpd_##KFD_TYPE;                                                             \
    };

GENERATE_KFD_TRAIT_MAPPING(kfd_event_page_migrate_record_t)
GENERATE_KFD_TRAIT_MAPPING(kfd_event_page_fault_record_t)
GENERATE_KFD_TRAIT_MAPPING(kfd_event_queue_record_t)
GENERATE_KFD_TRAIT_MAPPING(kfd_event_unmap_from_gpu_record_t)
GENERATE_KFD_TRAIT_MAPPING(kfd_event_dropped_events_record_t)
GENERATE_KFD_TRAIT_MAPPING(kfd_page_migrate_record_t)
GENERATE_KFD_TRAIT_MAPPING(kfd_page_fault_record_t)
GENERATE_KFD_TRAIT_MAPPING(kfd_queue_record_t)

#undef GENERATE_KFD_TRAIT_MAPPING

template <typename Tp>
using rocpd_kfd_wrapper_t = typename impl::rocpd_kfd_wrapper<Tp>::type;

template <typename, typename = void>
struct has_member_timestamp : std::false_type
{};

template <typename T>
struct has_member_timestamp<T, std::void_t<decltype(std::declval<T>().timestamp)>> : std::true_type
{};

template <typename T>
constexpr bool has_member_timestamp_v = has_member_timestamp<T>::value;

template <typename RecordT>
auto
get_start(const RecordT& record)
{
    if constexpr(has_member_timestamp_v<RecordT>)
    {
        return record.timestamp;
    }
    else
    {
        return record.start_timestamp;
    }
}

template <typename RecordT>
auto
get_end(const RecordT& record)
{
    if constexpr(has_member_timestamp_v<RecordT>)
    {
        return record.timestamp;
    }
    else
    {
        return record.end_timestamp;
    }
}

template <typename RecordT>
kfd_pmc_event_data_t
construct_kfd_pmc_event(const metadata& tool_metadata, const RecordT& record)
{
    auto data  = kfd_pmc_event_data_t{};
    data.kind  = record.kind;
    data.name  = tool_metadata.buffer_names.at(data.kind, record.operation);
    data.tid   = record.pid;
    data.start = get_start(record);
    data.end   = get_end(record);

    auto wrapper   = rocpd_kfd_wrapper_t<RecordT>(record, tool_metadata);
    data.value     = wrapper.value();
    data.json_data = get_json_string([&wrapper](auto& ar) {
        ar(cereal::make_nvp("kfd", ::rocprofiler::tool::rocpd_kfd_event_data_t{wrapper}));
    });

    return data;
}

// ---------------------------------------------------------------------------
// Self-describing PC-sample blob machinery.
//
// A blob layout is an ordered list of reusable field descriptors.  The SAME list
// drives (1) the packed little-endian bytes written per sample and (2) the
// self-describing schema (rocpd_info_blob_schema / rocpd_info_blob_field), so the
// two can never drift.  Each sampling method has exactly one layout, composed from
// shared groups (hw_id, wave/workgroup, and -- for stochastic -- snapshot); adding
// or removing a field is a one-line edit to a group -- no packed struct, no
// per-layout macro, no static_assert to keep in sync.
// ---------------------------------------------------------------------------
struct blob_field_desc
{
    std::string_view name;
    size_t           offset;
    size_t           size;
    std::string_view data_type;
    bool             is_signed;
    std::string_view description;
};

struct blob_schema_desc
{
    std::string_view             name;
    std::string_view             source_table;
    std::string_view             description;
    std::string_view             byte_order;
    int64_t                      alignment;
    int64_t                      struct_size;
    int64_t                      version;
    std::vector<blob_field_desc> fields;
};

// One reusable blob field: how to describe it (schema) and how to read its value from
// a sample record (fill).  `get` is a captureless function pointer, so a layout carries
// no per-field heap state and the hot path is a plain indirect call.
template <typename SampleT>
struct pc_sample_blob_field
{
    std::string_view name;
    std::string_view data_type;
    uint32_t         size;  // bytes emitted to the packed blob (little-endian)
    bool             is_signed;
    std::string_view description;
    uint64_t (*get)(const SampleT&);
};

template <typename SampleT>
using pc_sample_blob_layout = std::vector<pc_sample_blob_field<SampleT>>;

// Recover the record type Tp from a generator<Tp> so callers need not name it.
template <typename>
struct generator_value;
template <typename Tp>
struct generator_value<generator<Tp>>
{
    using type = Tp;
};

// hw_id group -- the baseline hw_id field set, shared by both sampling methods.
template <typename SampleT>
void
append_pc_sample_hw_id_fields(pc_sample_blob_layout<SampleT>& out)
{
    out.push_back({"hw_id_chiplet",
                   "uint8_t",
                   1,
                   false,
                   "HW ID chiplet index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.chiplet; }});
    out.push_back({"hw_id_wave_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID wave slot index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.wave_id; }});
    out.push_back({"hw_id_simd_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID SIMD index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.simd_id; }});
    out.push_back({"hw_id_pipe_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID pipe index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.pipe_id; }});
    out.push_back({"hw_id_cu_or_wgp_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID CU (GFX9) or WGP (GFX10+) index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.cu_or_wgp_id; }});
    out.push_back({"hw_id_shader_array_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID shader array index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.shader_array_id; }});
    out.push_back({"hw_id_shader_engine_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID shader engine index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.shader_engine_id; }});
    out.push_back({"hw_id_workgroup_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID workgroup index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.workgroup_id; }});
    out.push_back({"hw_id_vm_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID virtual memory ID",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.vm_id; }});
    out.push_back(
        {"hw_id_queue_id", "uint8_t", 1, false, "HW ID queue ID", [](const SampleT& r) -> uint64_t {
             return r.hw_id.queue_id;
         }});
    out.push_back({"hw_id_microengine_id",
                   "uint8_t",
                   1,
                   false,
                   "HW ID microengine (ACE) index",
                   [](const SampleT& r) -> uint64_t { return r.hw_id.microengine_id; }});
}

// Wave/workgroup group -- the baseline wave/workgroup position fields, shared by both
// sampling methods.
template <typename SampleT>
void
append_pc_sample_wave_workgroup_fields(pc_sample_blob_layout<SampleT>& out)
{
    out.push_back({"wave_in_group",
                   "uint32_t",
                   4,
                   false,
                   "Wave position within workgroup",
                   [](const SampleT& r) -> uint64_t { return r.wave_in_group; }});
    out.push_back({"workgroup_id_x",
                   "uint32_t",
                   4,
                   false,
                   "Workgroup coordinate X",
                   [](const SampleT& r) -> uint64_t { return r.workgroup_id.x; }});
    out.push_back({"workgroup_id_y",
                   "uint32_t",
                   4,
                   false,
                   "Workgroup coordinate Y",
                   [](const SampleT& r) -> uint64_t { return r.workgroup_id.y; }});
    out.push_back({"workgroup_id_z",
                   "uint32_t",
                   4,
                   false,
                   "Workgroup coordinate Z",
                   [](const SampleT& r) -> uint64_t { return r.workgroup_id.z; }});
}

// Snapshot group (arbiter/dual-issue state) -- stochastic only.
template <typename SampleT>
void
append_pc_sample_snapshot_fields(pc_sample_blob_layout<SampleT>& out)
{
    out.push_back({"dual_issue_valu",
                   "uint8_t",
                   1,
                   false,
                   "Dual-issue VALU (stochastic only)",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.dual_issue_valu; }});
    out.push_back({"arb_state_issue_valu",
                   "uint8_t",
                   1,
                   false,
                   "Arbiter issued VALU instruction",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_valu; }});
    out.push_back({"arb_state_issue_matrix",
                   "uint8_t",
                   1,
                   false,
                   "Arbiter issued matrix instruction",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_matrix; }});
    out.push_back({"arb_state_issue_lds",
                   "uint8_t",
                   1,
                   false,
                   "Arbiter issued LDS instruction",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_lds; }});
    out.push_back({"arb_state_issue_scalar",
                   "uint8_t",
                   1,
                   false,
                   "Arbiter issued scalar instruction",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_scalar; }});
    out.push_back(
        {"arb_state_issue_vmem_tex",
         "uint8_t",
         1,
         false,
         "Arbiter issued VMEM/TEX instruction",
         [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_vmem_tex; }});
    out.push_back({"arb_state_issue_flat",
                   "uint8_t",
                   1,
                   false,
                   "Arbiter issued FLAT instruction",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_flat; }});
    out.push_back({"arb_state_issue_exp",
                   "uint8_t",
                   1,
                   false,
                   "Arbiter issued export instruction",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_exp; }});
    out.push_back({"arb_state_issue_misc",
                   "uint8_t",
                   1,
                   false,
                   "Arbiter issued misc instruction",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_issue_misc; }});
    out.push_back({"arb_state_stall_valu",
                   "uint8_t",
                   1,
                   false,
                   "VALU stall",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_stall_valu; }});
    out.push_back({"arb_state_stall_matrix",
                   "uint8_t",
                   1,
                   false,
                   "Matrix stall",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_stall_matrix; }});
    out.push_back(
        {"arb_state_stall_lds", "uint8_t", 1, false, "LDS stall", [](const SampleT& r) -> uint64_t {
             return r.snapshot.arb_state_stall_lds;
         }});
    out.push_back({"arb_state_stall_scalar",
                   "uint8_t",
                   1,
                   false,
                   "Scalar stall",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_stall_scalar; }});
    out.push_back(
        {"arb_state_stall_vmem_tex",
         "uint8_t",
         1,
         false,
         "VMEM/TEX stall",
         [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_stall_vmem_tex; }});
    out.push_back({"arb_state_stall_flat",
                   "uint8_t",
                   1,
                   false,
                   "Flat stall",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_stall_flat; }});
    out.push_back({"arb_state_stall_exp",
                   "uint8_t",
                   1,
                   false,
                   "Export stall",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_stall_exp; }});
    out.push_back({"arb_state_stall_misc",
                   "uint8_t",
                   1,
                   false,
                   "Misc stall",
                   [](const SampleT& r) -> uint64_t { return r.snapshot.arb_state_stall_misc; }});
}

// Compose the complete blob layout for one sampling method.  Host-trap uses the hw_id +
// wave/workgroup groups; stochastic additionally appends the snapshot (arbiter) group.
// IsStochastic is a compile-time parameter so the snapshot extractors (which read .snapshot)
// are only instantiated for the stochastic record.
template <bool IsStochastic, typename SampleT>
pc_sample_blob_layout<SampleT>
make_pc_sample_layout()
{
    auto out = pc_sample_blob_layout<SampleT>{};
    append_pc_sample_hw_id_fields(out);
    append_pc_sample_wave_workgroup_fields(out);
    if constexpr(IsStochastic) append_pc_sample_snapshot_fields(out);
    return out;
}

template <typename SampleT>
size_t
pc_sample_blob_size(const pc_sample_blob_layout<SampleT>& layout)
{
    auto total = size_t{0};
    for(const auto& f : layout)
        total += f.size;
    return total;
}

// Derive the self-describing schema (offsets auto-accumulated) from the same layout that
// produces the bytes -- a single source of truth for both.
template <typename SampleT>
blob_schema_desc
make_pc_sample_schema(std::string_view                      name,
                      std::string_view                      description,
                      const pc_sample_blob_layout<SampleT>& layout)
{
    auto fields = std::vector<blob_field_desc>{};
    fields.reserve(layout.size());
    auto offset = size_t{0};
    for(const auto& f : layout)
    {
        fields.push_back({f.name, offset, f.size, f.data_type, f.is_signed, f.description});
        offset += f.size;
    }
    return blob_schema_desc{name,
                            "rocpd_gpu_pc_sample",
                            description,
                            "little",
                            int64_t{1},
                            static_cast<int64_t>(offset),
                            int64_t{1},
                            std::move(fields)};
}

// Assemble the packed little-endian blob for one sample from its layout, reusing a pooled
// buffer to avoid a per-sample heap allocation.
template <typename SampleT>
std::string
build_pc_sample_blob(rocpd_db&                             db,
                     const pc_sample_blob_layout<SampleT>& layout,
                     size_t                                total_size,
                     const SampleT&                        rec)
{
    auto buf = db.acquire_blob_buffer();
    buf.resize(total_size);
    auto offset = size_t{0};
    for(const auto& f : layout)
    {
        const auto value = f.get(rec);
        for(uint32_t b = 0; b < f.size; ++b)
            buf[offset + b] = static_cast<char>((value >> (8U * b)) & 0xFFU);
        offset += f.size;
    }
    return buf;
}

// Register a self-describing blob schema and return its rowid.
//
// Design note: get_insert_statement batches inserts, so the schema row is not visible to
// sqlite3_last_insert_rowid() until its batch is flushed.  Flush the schema batch before
// reading the rowid, and read it before enqueueing any field rows.
inline uint64_t
register_blob_schema(rocpd_db&               db,
                     uint64_t                node_id,
                     uint64_t                this_pid,
                     const blob_schema_desc& schema)
{
    auto schema_table = replace_uuid(db, "rocpd_info_blob_schema{{uuid}}");

    // NOTE: the caller (insert_pc_sampling_data) already holds a deferred_transaction.
    // SQLite has no nested transactions, so we must NOT open another one here (doing so
    // fails with "cannot start a transaction within a transaction").
    get_insert_statement(
        db,
        "rocpd_info_blob_schema{{uuid}}",
        {
            insert_value("nid", node_id),
            insert_value("pid", this_pid),
            insert_value("name", std::string{schema.name}),
            insert_value("source_table", std::string{schema.source_table}),
            insert_value("description", std::string{schema.description}, allow_empty_string{}),
            insert_value("byte_order", std::string{schema.byte_order}),
            insert_value("alignment", schema.alignment),
            insert_value("struct_size", schema.struct_size),
            insert_value("version", schema.version),
        });

    if(auto itr = db.pending_batches.find(schema_table); itr != db.pending_batches.end())
        flush_pending_insert_batch(db, itr->second);

    const auto schema_id = static_cast<uint64_t>(sqlite3_last_insert_rowid(db.conn));

    for(const auto& field : schema.fields)
    {
        get_insert_statement(
            db,
            "rocpd_info_blob_field{{uuid}}",
            {
                insert_value("schema_id", schema_id),
                insert_value("name", std::string{field.name}),
                insert_value("offset", static_cast<int64_t>(field.offset)),
                insert_value("size", static_cast<int64_t>(field.size)),
                insert_value("data_type", std::string{field.data_type}),
                insert_value("is_signed", field.is_signed ? int64_t{1} : int64_t{0}),
                insert_value("description", std::string{field.description}, allow_empty_string{}),
            });
    }

    return schema_id;
}
}  // namespace

void
write_rocpd(
    const output_config&                                                    cfg,
    const metadata&                                                         tool_metadata,
    const std::vector<agent_info>&                                          agent_data,
    const generator<tool_buffer_tracing_hip_api_ext_record_t>&              hip_api_gen,
    const generator<rocprofiler_buffer_tracing_hsa_api_record_t>&           hsa_api_gen,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>&      kernel_dispatch_gen,
    const generator<tool_buffer_tracing_memory_copy_ext_record_t>&          memory_copy_gen,
    const generator<rocprofiler_buffer_tracing_marker_api_record_t>&        marker_api_gen,
    const generator<tool_buffer_tracing_memory_allocation_ext_record_t>&    memory_alloc_gen,
    const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>&    scratch_memory_gen,
    const generator<tool_buffer_tracing_kfd_record_t>&                      kfd_gen,
    const generator<rocprofiler_buffer_tracing_rccl_api_record_t>&          rccl_api_gen,
    const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& rocdecode_api_gen,
    const generator<tool_counter_record_t>&                                 counter_collection_gen,
    const generator<tool_spm_counter_record_t>&                             spm_collection_gen,
    const generator<rocprofiler_buffer_tracing_ompt_record_t>&              ompt_gen,
    const generator<rocprofiler_buffer_tracing_hip_graph_record_t>&         graph_launch_gen,
    const generator<rocprofiler_buffer_tracing_rocshmem_api_ext_record_t>&  rocshmem_api_gen,
    const generator<rocprofiler_buffer_tracing_hipfile_api_ext_record_t>&   hipfile_api_gen,
    const generator<rocprofiler_tool_pc_sampling_host_trap_record_t>&  pc_sampling_host_trap_gen,
    const generator<rocprofiler_tool_pc_sampling_stochastic_record_t>& pc_sampling_stochastic_gen)
{
    static auto get_simple_timer = [](std::string_view label) {
        return common::simple_timer{fmt::format("SQLite3 generation :: {:24}", label)};
    };

    auto _sqlgenperf = get_simple_timer("total");

    auto node_hash =
        get_hash_id(tool_metadata.node_data.machine_id) % std::numeric_limits<int64_t>::max();
    auto node_id = node_hash % std::numeric_limits<uint32_t>::max();

    const uint64_t this_pid         = tool_metadata.process_id;
    const uint64_t this_pid_init_ns = tool_metadata.process_start_ns;
    const uint64_t this_ppid        = tool_metadata.parent_process_id;
    const uint64_t this_nid         = node_id;

    ROCP_WARNING << fmt::format(
        "writing SQL database for process {} on node {}", this_pid, this_nid);

    auto      db   = rocpd_db{};
    sqlite3*& conn = db.conn;

    {
        const auto& mach_id = tool_metadata.node_data.machine_id;
        auto        ticks   = common::get_process_start_ticks_since_boot(this_pid);
        auto        seed    = common::compute_system_seed(mach_id, this_pid, this_ppid, ticks);
        auto        uuid_v7 = common::generate_uuid_v7(this_pid_init_ns, seed);

        db.guid = fmt::format("{}", uuid_v7);
        db.uuid = fmt::format("_{}", replace_all(uuid_v7, '-', "_"));

        // reading schemata
        auto table_schema = read_schema_file(db, ROCPD_SQL_SCHEMA_ROCPD_TABLES);
        auto output_file  = get_output_filename(cfg, "results", "db");
        if(fs::exists(output_file)) fs::remove(output_file);

        SQLITE3_CHECK(sqlite3_open_v2(
            output_file.c_str(), &conn, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr));
        SQLITE3_CHECK(sqlite3_busy_handler(conn, &sql::busy_handler, nullptr));

        ROCP_ERROR << fmt::format("Opened result file: {} (UUID={})", output_file, uuid_v7);

        execute_raw_sql_statements(conn, table_schema);

        for(auto itr : {ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
                        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
                        ROCPD_SQL_SCHEMA_ROCPD_METADATA,
                        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS})
        {
            auto views_schema = read_schema_file(db, itr);
            execute_raw_sql_statements(conn, views_schema);
        }
    }

    auto _metadata = metadata{};
    add_string_entry(_metadata, "");

    for(const auto& itr : agent_data)
        add_string_entry(_metadata, itr.name, itr.vendor_name, itr.product_name, itr.model_name);

    for(const auto& itr : tool_metadata.buffer_names)
    {
        add_string_entry(_metadata, itr.name);
        for(const auto& iitr : itr.operations)
            add_string_entry(_metadata, iitr);
    }

    for(const auto& itr : tool_metadata.callback_names)
    {
        add_string_entry(_metadata, itr.name);
        for(const auto& iitr : itr.operations)
            add_string_entry(_metadata, iitr);
    }

    for(const auto& itr : tool_metadata.marker_messages.get())
    {
        add_string_entry(_metadata, itr.second);
    }

    for(const auto& itr : tool_metadata.get_kernel_symbols())
        add_string_entry(_metadata,
                         itr.kernel_name,
                         itr.formatted_kernel_name,
                         itr.demangled_kernel_name,
                         itr.truncated_kernel_name);

    // Create a local copy of the rename map to use for this output run instead of using
    // get_kernel_name(), which may access a modified version of the map
    auto _rename_strings = std::unordered_map<uint64_t, std::string_view>{};
    for(const auto& itr : tool_metadata.kernel_rename_map.get())
    {
        if(!itr.first.empty())
        {
            add_string_entry(_metadata, itr.first);
            _rename_strings.emplace(itr.second, itr.first);
        }
    }

    for(const auto& itr : tool_metadata.get_code_objects())
        if(itr.uri != nullptr) add_string_entry(_metadata, itr.uri);

    for(const auto& itr : tool_metadata.get_counter_info())
    {
        add_string_entry(_metadata, itr.name);
        for(const auto& ditr : itr.dimensions)
            add_string_entry(_metadata, ditr.name);
    }

    for(const auto& itr : tool_metadata.get_counter_dimension_info())
        add_string_entry(_metadata, itr.name);

    add_string_entry(_metadata, "SPM");

    auto thread_ids = std::set<rocprofiler_thread_id_t>{};
    auto stream_set = std::unordered_set<rocprofiler_stream_id_t>{};
    auto queue_set  = std::unordered_set<rocprofiler_queue_id_t>{};

    auto get_stream_id = [&db, &stream_set, &node_id, &this_pid](rocprofiler_stream_id_t val) {
        if(stream_set.count(val) == 0)
        {
            ROCP_FATAL_IF(val.handle == 0 && !stream_set.empty()) << "Missing default stream";
            ROCP_FATAL_IF(val.handle != 0 && stream_set.empty()) << "Missing default stream";

            auto _name = std::string{};
            if(val.handle == 0)
                _name = std::string{"Default Stream"};
            else
                _name = fmt::format("Stream {}", stream_set.size() - 1);
            stream_set.emplace(val);

            get_insert_statement(db,
                                 "rocpd_info_stream{{uuid}}",
                                 {
                                     insert_value("id", val.handle),
                                     insert_value("nid", node_id),
                                     insert_value("pid", this_pid),
                                     insert_value("name", _name),
                                 });
        }

        return val.handle;
    };

    auto get_queue_id = [&db, &queue_set, &node_id, &this_pid](rocprofiler_queue_id_t val) {
        if(queue_set.count(val) == 0)
        {
            ROCP_FATAL_IF(val.handle == 0 && !queue_set.empty()) << "Missing default queue";
            ROCP_FATAL_IF(val.handle != 0 && queue_set.empty()) << "Missing default queue";

            auto _name = std::string{};
            if(val.handle == 0)
                _name = std::string{"Default Queue"};
            else
                _name = fmt::format("Queue {}", queue_set.size() - 1);
            queue_set.emplace(val);

            get_insert_statement(db,
                                 "rocpd_info_queue{{uuid}}",
                                 {
                                     insert_value("id", val.handle),
                                     insert_value("nid", node_id),
                                     insert_value("pid", this_pid),
                                     insert_value("name", _name),
                                 });
        }

        return val.handle;
    };

    auto get_thread_id =
        [&db, &tool_metadata, &thread_ids, &node_id, &this_pid](rocprofiler_thread_id_t val) {
            if(thread_ids.count(val) == 0)
            {
                thread_ids.emplace(val);

                get_insert_statement(db,
                                     "rocpd_info_thread{{uuid}}",
                                     {
                                         insert_value("id", val),
                                         insert_value("nid", node_id),
                                         insert_value("ppid", tool_metadata.parent_process_id),
                                         insert_value("pid", this_pid),
                                         insert_value("tid", val),
                                     });
            }

            return val;
        };

    // use this to lookup indexes of strings
    auto string_entries = _metadata.get_string_entries();

    {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_string");

        auto _deferred = sql::deferred_transaction{db.conn};
        for(const auto& itr : string_entries)
        {
            get_insert_statement(db,
                                 "rocpd_string{{uuid}}",
                                 {
                                     insert_value("id", itr.second),
                                     insert_value("string", itr.first, allow_empty_string{}),
                                 });
        }
    }

    auto insert_node_data = [&db, &tool_metadata, node_id, node_hash]() {
        auto        _sqlgenperf_rocpd = get_simple_timer("rocpd_info_node");
        const auto& _info             = tool_metadata.node_data;

        get_insert_statement(
            db,
            "rocpd_info_node{{uuid}}",
            {
                insert_value("id", node_id),
                insert_value("hash", node_hash),
                insert_value("machine_id", _info.machine_id),
                insert_value("system_name", _info.system_name, allow_empty_string{}),
                insert_value("hostname", _info.hostname, allow_empty_string{}),
                insert_value("release", _info.release, allow_empty_string{}),
                insert_value("version", _info.version, allow_empty_string{}),
                insert_value("hardware_name", _info.hardware_name, allow_empty_string{}),
                insert_value("domain_name", _info.domain_name, allow_empty_string{}),
            });
    };

    auto insert_process_data = [&db, &tool_metadata, &cfg, node_id, this_pid]() {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_info_process");
        auto json_cfg          = get_json_string([&cfg](auto& ar) { cfg.save(ar); });
        auto json_env          = get_json_string([](auto& ar) {
            size_t i = 0;
            while(true)
            {
                const char* itr = environ[i++];
                if(!itr) break;
                if(auto pos = std::string_view{itr}.find('='); pos != std::string_view::npos)
                {
                    auto evar = std::string{itr}.substr(0, pos);
                    auto eval = std::string{itr}.substr(pos + 1);
                    ROCP_TRACE << "ENV: " << evar << " = " << eval;
                    if(eval.find(';') != std::string::npos)
                    {
                        ROCP_INFO << fmt::format(
                            "Env variable {} was sanitized due to semi-colon in the value", evar);
                    }
                    else if(!evar.empty())
                    {
                        ar(cereal::make_nvp(evar.c_str(), eval));
                    }
                }
            }
        });

        auto command = fmt::format(
            "{}",
            fmt::join(tool_metadata.command_line.begin(), tool_metadata.command_line.end(), " "));

        get_insert_statement(db,
                             "rocpd_info_process{{uuid}}",
                             {
                                 insert_value("id", this_pid),
                                 insert_value("nid", node_id),
                                 insert_value("ppid", tool_metadata.parent_process_id),
                                 insert_value("pid", this_pid),
                                 insert_value("init", tool_metadata.process_start_ns),
                                 insert_value("fini", tool_metadata.process_end_ns),
                                 insert_value("start", tool_metadata.process_start_ns),
                                 insert_value("end", tool_metadata.process_end_ns),
                                 insert_value("command", command),
                                 insert_value("environment", json_env),
                                 insert_value("extdata", json_cfg),
                             });
    };

    auto insert_agent_data = [&db, &tool_metadata, node_id, this_pid]() {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_info_agent");
        auto _deferred         = sql::deferred_transaction{db.conn};
        for(auto itr : tool_metadata.agents)
        {
            auto json_info = get_json_string([&itr](auto& ar) { cereal::save(ar, itr); });
            auto type      = std::string_view{"UNK"};
            if(itr.type == ROCPROFILER_AGENT_TYPE_CPU)
                type = "CPU";
            else if(itr.type == ROCPROFILER_AGENT_TYPE_GPU)
                type = "GPU";

            get_insert_statement(
                db,
                "rocpd_info_agent{{uuid}}",
                {
                    insert_value("id", itr.node_id),
                    insert_value("nid", node_id),
                    insert_value("pid", this_pid),
                    insert_value("type", type),
                    insert_value("absolute_index", itr.node_id),
                    insert_value("logical_index", itr.logical_node_id),
                    insert_value("type_index", itr.logical_node_type_id),
                    insert_value("uuid", itr.device_id),
                    insert_value("name", itr.name),
                    insert_value("model_name", itr.model_name, allow_empty_string{}),
                    insert_value("vendor_name", itr.vendor_name, allow_empty_string{}),
                    insert_value("product_name", itr.product_name, allow_empty_string{}),
                    insert_value("user_name", itr.product_name, allow_empty_string{}),
                    insert_value("extdata", json_info),
                });
        }
    };

    auto insert_kernel_code_object_data = [&db, &tool_metadata, node_id, this_pid]() {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd kernel info");
        auto _deferred         = sql::deferred_transaction{db.conn};
        for(const auto& itr : tool_metadata.get_code_objects())
        {
            if(itr.size == 0) continue;

            const auto* _agent = tool_metadata.get_agent(itr.rocp_agent);
            auto        json_data =
                get_json_string([](auto& ar, const auto oitr) { cereal::save(ar, oitr); }, itr);

            get_insert_statement(
                db,
                "rocpd_info_code_object{{uuid}}",
                {
                    insert_value("id", itr.code_object_id),
                    insert_value("nid", node_id),
                    insert_value("pid", this_pid),
                    insert_value("agent_id", CHECK_NOTNULL(_agent)->node_id),
                    insert_value("uri", itr.uri),
                    insert_value("load_base", itr.load_base),
                    insert_value("load_size", itr.load_size),
                    insert_value("load_delta", itr.load_delta),
                    insert_nullable_value("storage_type",
                                          get_code_object_storage_type(itr.storage_type)),
                    insert_value("extdata", json_data),
                });
        }

        for(const auto& itr : tool_metadata.get_kernel_symbols())
        {
            if(itr.kernel_id == 0 && itr.code_object_id == 0) continue;

            auto json_data =
                get_json_string([](auto& ar, const auto& oitr) { cereal::save(ar, oitr); }, itr);

            get_insert_statement(
                db,
                "rocpd_info_kernel_symbol{{uuid}}",
                {
                    insert_value("id", itr.kernel_id),
                    insert_value("nid", node_id),
                    insert_value("pid", this_pid),
                    insert_value("code_object_id", itr.code_object_id),
                    insert_value("kernel_name", itr.kernel_name, allow_empty_string{}),
                    insert_value("display_name", itr.formatted_kernel_name, allow_empty_string{}),
                    insert_value("kernel_object", itr.kernel_object),
                    insert_value("kernarg_segment_size", itr.kernarg_segment_size),
                    insert_value("kernarg_segment_alignment", itr.kernarg_segment_alignment),
                    insert_value("group_segment_size", itr.group_segment_size),
                    insert_value("private_segment_size", itr.private_segment_size),
                    insert_value("sgpr_count", itr.sgpr_count),
                    insert_value("arch_vgpr_count", itr.arch_vgpr_count),
                    insert_value("accum_vgpr_count", itr.accum_vgpr_count),
                    insert_value("extdata", json_data),
                });
        }
    };

    auto insert_pmc_data = [&db, &tool_metadata, node_id, this_pid]() {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_info_pmc");
        auto recorded          = std::unordered_set<rocprofiler_counter_id_t>{};
        auto _deferred         = sql::deferred_transaction{db.conn};
        for(const auto& itr : tool_metadata.agent_counter_info)
        {
            for(const auto& aitr : itr.second)
            {
                const auto* agent = tool_metadata.get_agent(itr.first);
                if(agent == nullptr || !recorded.emplace(aitr.id).second) continue;

                auto json_data = get_json_string([agent](auto& ar) {
                    if(agent) cereal::save(ar, *agent);
                });

                const auto* _name        = aitr.name;
                const auto* _description = aitr.description;
                const auto* _block       = aitr.block;
                const auto* _expression  = aitr.expression;

                get_insert_statement(
                    db,
                    "rocpd_info_pmc{{uuid}}",
                    {
                        insert_value("id", aitr.id.handle),
                        insert_value("nid", node_id),
                        insert_value("pid", this_pid),
                        insert_value("target_arch", std::string_view{"GPU"}),
                        insert_value("agent_id", agent->node_id),
                        insert_value("name", _name, allow_empty_string{}),
                        insert_value("symbol", _name, allow_empty_string{}),
                        insert_value("description", _description, allow_empty_string{}),
                        insert_value("component", std::string_view{"rocm"}),
                        insert_value("value_type", std::string_view{"ABS"}),
                        insert_value("block", _block, allow_empty_string{}),
                        insert_value("expression", _expression, allow_empty_string{}),
                        insert_value("is_constant", aitr.is_constant),
                        insert_value("is_derived", aitr.is_derived),
                        insert_value("spm_support", aitr.spm_support),
                        insert_value("extdata", json_data),
                    });
            }
        }
    };

    auto insert_kernel_dispatch_data = [&, node_id, this_pid](auto& dispatch_evt_ids,
                                                              auto& dispatch_agent_ids,
                                                              auto& dispatch_thread_ids) {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_kernel_dispatch");
        auto _deferred         = sql::deferred_transaction{db.conn};

        auto process_dispatch = [&](uint64_t    dispatch_id,
                                    uint64_t    kernel_id,
                                    const auto& corr_id,
                                    const auto& info,
                                    const auto& kind,
                                    uint32_t    thread_id,
                                    uint64_t    queue_id,
                                    uint64_t    stream_id,
                                    uint64_t    start_timestamp,
                                    uint64_t    end_timestamp,
                                    const auto& grid,
                                    const auto& workgroup,
                                    uint64_t    graph_exec_id,
                                    uint64_t    graph_node_id,
                                    bool        enable_duplicate_check) {
            // Skip if we've already processed this dispatch_id.
            if(dispatch_evt_ids.count(dispatch_id) > 0) return;

            auto kern_name = (kernel_id > 0)
                                 ? tool_metadata.get_kernel_symbol(kernel_id)->formatted_kernel_name
                                 : "unknown_kernel";

            auto evt_id = create_event(db,
                                       {
                                           insert_value("category_id", string_entries.at(kind)),
                                           insert_value("stack_id", corr_id.internal),
                                           insert_value("parent_stack_id", corr_id.internal),
                                           insert_value("correlation_id", corr_id.external.value),
                                       });

            // Check for duplicates if requested
            if(enable_duplicate_check && dispatch_evt_ids.count(dispatch_id) > 0)
            {
                ROCP_CI_LOG(WARNING)
                    << fmt::format("duplicate kernel dispatch id {} :: event_id={}, kernel_id={}, "
                                   "corr_id={}, name='{}'",
                                   dispatch_id,
                                   evt_id,
                                   kernel_id,
                                   corr_id.internal,
                                   kern_name);
            }

            dispatch_evt_ids.emplace(dispatch_id, evt_id);
            // Unconditionally collect kernel rename data if it is available. rocpd needs to be able
            // to use kernel rename option after data has already been collected, so the kernel
            // rename data needs to be stored in generated db.
            auto _rename_it = _rename_strings.find(corr_id.external.value);
            auto region_name =
                (_rename_it != _rename_strings.end()) ? _rename_it->second : std::string_view{};

            auto agent_node_id = tool_metadata.get_agent(info.agent_id)->node_id;

            get_thread_id(thread_id);
            dispatch_agent_ids.emplace(dispatch_id, agent_node_id);
            dispatch_thread_ids.emplace(dispatch_id, static_cast<uint64_t>(thread_id));

            // Insert into kernel dispatch table
            get_insert_statement(
                db,
                "rocpd_kernel_dispatch{{uuid}}",
                {
                    insert_value("id", dispatch_id),
                    insert_value("nid", node_id),
                    insert_value("pid", this_pid),
                    insert_value("tid", thread_id),
                    insert_value("agent_id", agent_node_id),
                    insert_value("kernel_id", kernel_id),
                    insert_value("dispatch_id", dispatch_id),
                    insert_value("queue_id", queue_id),
                    insert_value("stream_id", stream_id),
                    insert_value("start", start_timestamp),
                    insert_value("end", end_timestamp),
                    insert_value("private_segment_size", info.private_segment_size),
                    insert_value("group_segment_size", info.group_segment_size),
                    insert_value("workgroup_size_x", workgroup.x),
                    insert_value("workgroup_size_y", workgroup.y),
                    insert_value("workgroup_size_z", workgroup.z),
                    insert_value("grid_size_x", grid.x),
                    insert_value("grid_size_y", grid.y),
                    insert_value("grid_size_z", grid.z),
                    insert_value("graph_exec_id", graph_exec_id),
                    insert_value("graph_node_id", graph_node_id),
                    insert_value("region_name_id", string_entries.at(region_name)),
                    insert_value("event_id", evt_id),
                });
        };

        if(kernel_dispatch_gen.empty())
        {
            for(auto pctr : counter_collection_gen)
            {
                for(const auto& record : counter_collection_gen.get(pctr))
                {
                    const auto& dispatch_data = record.dispatch_data;
                    const auto& info          = dispatch_data.dispatch_info;

                    // Register thread ID
                    get_thread_id(record.thread_id);

                    // Use buffer category for kernel dispatches
                    auto kind =
                        tool_metadata.buffer_names.at(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);

                    // Process this dispatch (counter-collection path: no graph attribution).
                    process_dispatch(info.dispatch_id,                 // dispatch_id
                                     info.kernel_id,                   // kernel_id
                                     dispatch_data.correlation_id,     // corr_id
                                     info,                             // info
                                     kind,                             // kind
                                     record.thread_id,                 // thread_id
                                     get_queue_id(info.queue_id),      // queue_id
                                     get_stream_id(record.stream_id),  // stream_id
                                     dispatch_data.start_timestamp,    // start_timestamp
                                     dispatch_data.end_timestamp,      // end_timestamp
                                     info.grid_size,                   // grid
                                     info.workgroup_size,              // workgroup
                                     0,                                // graph_exec_id
                                     0,                                // graph_node_id
                                     false                             // enable_duplicate_check
                    );
                }
            }

            for(auto pctr : spm_collection_gen)
            {
                for(const auto& record : spm_collection_gen.get(pctr))
                {
                    const auto& dispatch_data = record.dispatch_data;
                    const auto& info          = dispatch_data.dispatch_info;

                    // Register thread ID
                    get_thread_id(record.thread_id);

                    // Use buffer category for kernel dispatches
                    auto kind =
                        tool_metadata.buffer_names.at(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);

                    // Process this dispatch (SPM dispatch timestamps are not available)
                    process_dispatch(info.dispatch_id,              // dispatch_id
                                     info.kernel_id,                // kernel_id
                                     dispatch_data.correlation_id,  // corr_id
                                     info,                          // info
                                     kind,                          // kind
                                     record.thread_id,              // thread_id
                                     get_queue_id(info.queue_id),   // queue_id
                                     get_stream_id(record.stream_id),
                                     0,                    // start_timestamp
                                     0,                    // end_timestamp
                                     info.grid_size,       // grid
                                     info.workgroup_size,  // workgroup
                                     0,                    // graph_exec_id
                                     0,                    // graph_node_id
                                     false                 // enable_duplicate_check
                    );
                }
            }
        }
        else
        {
            for(auto pitr : kernel_dispatch_gen)
            {
                for(auto itr : kernel_dispatch_gen.get(pitr))
                {
                    // Register thread ID
                    get_thread_id(itr.thread_id);

                    // Process this dispatch
                    process_dispatch(itr.dispatch_info.dispatch_id,             // dispatch_id
                                     itr.dispatch_info.kernel_id,               // kernel_id
                                     itr.correlation_id,                        // corr_id
                                     itr.dispatch_info,                         // info
                                     tool_metadata.buffer_names.at(itr.kind),   // kind
                                     itr.thread_id,                             // thread_id
                                     get_queue_id(itr.dispatch_info.queue_id),  // queue_id
                                     get_stream_id(itr.stream_id),              // stream_id
                                     itr.start_timestamp,                       // start_timestamp
                                     itr.end_timestamp,                         // end_timestamp
                                     itr.dispatch_info.grid_size,               // grid
                                     itr.dispatch_info.workgroup_size,          // workgroup
                                     itr.graph_exec_id.handle,                  // graph_exec_id
                                     itr.graph_node_id.handle,                  // graph_node_id
                                     true  // enable_duplicate_check
                    );
                }
            }
        }
    };

    auto insert_pmc_event_data = [&db,
                                  &tool_metadata,
                                  &counter_collection_gen,
                                  &spm_collection_gen,
                                  &string_entries,
                                  node_id,
                                  this_pid](auto& dispatch_evt_ids) {
        auto   _sqlgenperf_rocpd = get_simple_timer("rocpd_pmc_event");
        auto   _deferred         = sql::deferred_transaction{db.conn};
        size_t idx               = tool_metadata.pmc_event_offset;
        for(auto ditr : counter_collection_gen)
        {
            for(const auto& record : counter_collection_gen.get(ditr))
            {
                const auto& info        = record.dispatch_data.dispatch_info;
                auto        dispatch_id = info.dispatch_id;

                auto dispatch_itr = dispatch_evt_ids.find(dispatch_id);
                if(dispatch_itr == dispatch_evt_ids.end()) continue;

                auto evt_id = dispatch_itr->second;
                for(const auto& count : record.read())
                {
                    get_insert_statement(db,
                                         "rocpd_pmc_event{{uuid}}",
                                         {
                                             insert_value("id", idx++),
                                             insert_value("event_id", evt_id),
                                             insert_value("pmc_id", count.id.handle),
                                             insert_value("value", count.value),
                                         });
                }
            }
        }

        auto spm_name_id = string_entries.at("SPM");

        for(auto ditr : spm_collection_gen)
        {
            for(const auto& record : spm_collection_gen.get(ditr))
            {
                const auto& info        = record.dispatch_data.dispatch_info;
                auto        dispatch_id = info.dispatch_id;
                auto        evt_id      = dispatch_evt_ids.at(dispatch_id);

                auto track_id =
                    get_track_id(db, node_id, this_pid, record.thread_id, spm_name_id, "{}");

                auto counter_values = record.read();
                auto values_by_timestamp =
                    std::map<uint64_t, std::vector<const tool::tool_spm_counter_value_t*>>{};
                for(const auto& count : counter_values)
                    values_by_timestamp[count.timestamp].emplace_back(&count);

                for(const auto& [timestamp, values] : values_by_timestamp)
                {
                    auto sample_id = db.get_sample_id();
                    get_insert_statement(db,
                                         "rocpd_sample{{uuid}}",
                                         {
                                             insert_value("id", sample_id),
                                             insert_value("track_id", track_id),
                                             insert_value("timestamp", timestamp),
                                             insert_value("event_id", evt_id),
                                         });

                    for(const auto* count_ptr : values)
                    {
                        const auto& count = *count_ptr;
                        get_insert_statement(
                            db,
                            "rocpd_pmc_event{{uuid}}",
                            {
                                insert_value("id", idx++),
                                insert_value("event_id", evt_id),
                                insert_value("sample_id", sample_id),
                                insert_value("pmc_id", count.id.handle),
                                insert_value("value", count.value),
                                insert_value(
                                    "xcc",
                                    counters::rec_to_dim_pos(count.instance_id,
                                                             counters::ROCPROFILER_DIMENSION_XCC)),
                                insert_value("shader_engine",
                                             counters::rec_to_dim_pos(
                                                 count.instance_id,
                                                 counters::ROCPROFILER_DIMENSION_SHADER_ENGINE)),
                                insert_value("instance",
                                             counters::rec_to_dim_pos(
                                                 count.instance_id,
                                                 counters::ROCPROFILER_DIMENSION_INSTANCE)),
                            });
                    }
                }
            }
        }
    };

    auto insert_memory_copy_data =
        [&db, &tool_metadata, &string_entries, node_id, this_pid, &get_thread_id, &get_stream_id](
            const auto& _gen) {
            auto   _sqlgenperf_rocpd = get_simple_timer("rocpd_memory_copy");
            auto   _deferred         = sql::deferred_transaction{db.conn};
            size_t copy_idx          = 1;

            for(auto pitr : _gen)
            {
                for(auto itr : _gen.get(pitr))
                {
                    // insert thread info if it doesn't already exist
                    get_thread_id(itr.thread_id);

                    auto kind = tool_metadata.buffer_names.at(itr.kind);
                    auto name = tool_metadata.buffer_names.at(itr.kind, itr.operation);

                    auto evt_id = create_event(
                        db,
                        {
                            insert_value("category_id", string_entries.at(kind)),
                            insert_value("stack_id", itr.correlation_id.internal),
                            insert_value("parent_stack_id", itr.correlation_id.internal),
                            insert_value("correlation_id", itr.correlation_id.external.value),
                        });

                    get_insert_statement(
                        db,
                        "rocpd_memory_copy{{uuid}}",
                        {
                            insert_value("id", copy_idx++),
                            insert_value("nid", node_id),
                            insert_value("pid", this_pid),
                            insert_value("tid", itr.thread_id),
                            insert_value("start", itr.start_timestamp),
                            insert_value("end", itr.end_timestamp),
                            insert_value("name_id", string_entries.at(name)),
                            insert_value("dst_agent_id",
                                         tool_metadata.get_agent(itr.dst_agent_id)->node_id),
                            insert_value("src_agent_id",
                                         tool_metadata.get_agent(itr.src_agent_id)->node_id),
                            insert_value("dst_address", itr.dst_address.value),
                            insert_value("src_address", itr.src_address.value),
                            insert_value("size", itr.bytes),
                            insert_value("stream_id", get_stream_id(itr.stream_id)),
                            insert_value("graph_exec_id", itr.graph_exec_id.handle),
                            insert_value("graph_node_id", itr.graph_node_id.handle),
                            insert_value("event_id", evt_id),
                        });
                }
            }
        };

    auto insert_graph_launch_data = [&db,
                                     &tool_metadata,
                                     &string_entries,
                                     node_id,
                                     this_pid,
                                     &get_thread_id,
                                     &get_queue_id](const auto& _gen) {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_graph_launch");
        auto _deferred         = sql::deferred_transaction{db.conn};

        for(auto pitr : _gen)
        {
            for(auto itr : _gen.get(pitr))
            {
                get_thread_id(itr.thread_id);

                auto kind = tool_metadata.buffer_names.at(itr.kind);

                auto evt_id = create_event(
                    db,
                    {
                        insert_value("category_id", string_entries.at(kind)),
                        insert_value("stack_id", itr.correlation_id.internal),
                        insert_value("parent_stack_id", itr.correlation_id.internal),
                        insert_value("correlation_id", itr.correlation_id.external.value),
                    });

                auto agent_node_id =
                    (itr.agent_id.handle != 0)
                        ? std::optional<uint64_t>{tool_metadata.get_agent(itr.agent_id)->node_id}
                        : std::nullopt;

                get_insert_statement(
                    db,
                    "rocpd_graph_launch{{uuid}}",
                    {
                        insert_value("nid", node_id),
                        insert_value("pid", this_pid),
                        insert_value("tid", itr.thread_id),
                        insert_value("agent_id", agent_node_id),
                        insert_value("queue_id", get_queue_id(itr.queue_id)),
                        insert_value("start", itr.start_timestamp),
                        insert_value("end", itr.end_timestamp),
                        insert_value("graph_exec_id", itr.graph_exec_id.handle),
                        insert_value("kernel_dispatch_count", itr.kernel_dispatch_count),
                        insert_value("event_id", evt_id),
                    });
            }
        }
    };

    auto insert_memory_alloc_data = [&db,
                                     &tool_metadata,
                                     &string_entries,
                                     node_id,
                                     this_pid,
                                     &get_thread_id,
                                     &get_stream_id,
                                     &get_queue_id](const auto& _gen) {
        auto address_to_agent_and_size =
            std::unordered_map<rocprofiler_address_t, rocprofiler::agent::index_and_size>{};
        auto _deferred = sql::deferred_transaction{db.conn};

        for(auto pitr : _gen)
        {
            for(auto itr : _gen.get(pitr))
            {
                // insert thread info if it doesn't already exist
                get_thread_id(itr.thread_id);

                auto _kind           = tool_metadata.buffer_names.at(itr.kind);
                auto _cpptype        = tool_metadata.buffer_names.at(itr.kind, itr.operation);
                auto [_type, _level] = memtype_to_db(_cpptype);

                ROCP_FATAL_IF(_type != "ALLOC" && _type != "FREE" && _type != "RECLAIM" &&
                              _type != "REALLOC")
                    << "erroneous db type: " << _type;

                ROCP_FATAL_IF(_level != "REAL" && _level != "VIRTUAL" && _level != "SCRATCH")
                    << "erroneous db level: " << _level;

                auto _stream_id       = get_stream_id(extract_stream_field(itr));
                auto _queue_id        = get_queue_id(extract_queue_field(itr));
                auto _address         = extract_address_field(itr);
                auto _allocation_size = extract_allocation_size_field(itr);

                auto _node_id = std::optional<uint64_t>{};
                if(_type == "ALLOC")
                {
                    _node_id = tool_metadata.get_agent(itr.agent_id)->node_id;
                    address_to_agent_and_size.emplace(
                        rocprofiler_address_t{.handle = _address.handle},
                        rocprofiler::agent::index_and_size{_node_id.value(), _allocation_size});
                }
                else if(_type == "FREE")
                {
                    if(address_to_agent_and_size.count(_address) == 0)
                    {
                        if(_address.handle == 0)
                        {
                            // Freeing null pointers is expected behavior and is occurs in HSA
                            // functions like hipStreamDestroy
                            ROCP_INFO << "null pointer freed due to HSA operation";
                        }
                        else
                        {
                            // Following should not occur
                            ROCP_INFO << "Unpaired free operation occurred";
                        }
                    }
                    else
                    {
                        auto [agent_abs_index, size] = address_to_agent_and_size[_address];
                        _node_id                     = agent_abs_index;
                        _allocation_size             = 0;
                    }
                }
                else
                {
                    ROCP_CI_LOG(WARNING) << "unhandled memory allocation type " << _type;
                }

                auto evt_id = create_event(
                    db,
                    {
                        insert_value("category_id", string_entries.at(_kind)),
                        insert_value("stack_id", itr.correlation_id.internal),
                        insert_value("parent_stack_id", itr.correlation_id.ancestor),
                        insert_value("correlation_id", itr.correlation_id.external.value),
                    });

                auto flags = extract_flags_field(itr);

                get_insert_statement(
                    db,
                    "rocpd_memory_allocate{{uuid}}",
                    {
                        insert_value("nid", node_id),
                        insert_value("pid", this_pid),
                        insert_value("tid", itr.thread_id),
                        insert_value("start", itr.start_timestamp),
                        insert_value("end", itr.end_timestamp),
                        insert_value("agent_id", _node_id),
                        insert_value("type", _type),
                        insert_value("level", _level),
                        insert_value("queue_id", _queue_id),
                        insert_value("stream_id", _stream_id),
                        insert_value("event_id", evt_id),
                        insert_value("address", _address.value),
                        insert_value("size", _allocation_size),
                        insert_value("extdata",
                                     flags == "" ? "{}" : get_json_string([&flags](auto& ar) {
                                         ar(cereal::make_nvp("flags", flags));
                                     })),
                    });
            }
        }
    };

    // new string entries argument types and names can be added to _metadata
    auto insert_api_data = [&db,
                            &tool_metadata,
                            &string_entries,
                            node_id,
                            this_pid,
                            &get_thread_id](const auto& _gen) {
        auto _deferred = sql::deferred_transaction{db.conn};

        for(auto pitr : _gen)
        {
            for(auto itr : _gen.get(pitr))
            {
                auto category = tool_metadata.buffer_names.at(itr.kind);
                auto name     = tool_metadata.buffer_names.at(itr.kind, itr.operation);

                auto msg = std::string{"{}"};
                if(itr.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API)
                {
                    if(static_cast<rocprofiler_tracing_operation_t>(itr.operation) !=
                       ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxGetThreadId)
                    {
                        // check generatePerfetto.cpp and generateOTF2.cpp, and the marker name
                        // in the view
                        auto message =
                            tool_metadata.get_marker_message(itr.correlation_id.internal);
                        if(!message.empty())
                        {
                            msg = get_json_string(
                                [](auto& ar, std::string_view _msg) {
                                    ar(cereal::make_nvp("message", std::string{_msg}));
                                },
                                message);
                        }
                    }
                    else
                    {
                        msg = get_json_string(
                            [](auto& ar, std::string_view _msg) {
                                ar(cereal::make_nvp("message", std::string{_msg}));
                            },
                            name);
                    }
                }

                auto args = function_args_t{};
                {
                    auto _record = rocprofiler_record_header_t{
                        .hash = rocprofiler_record_header_compute_hash(
                            ROCPROFILER_BUFFER_CATEGORY_TRACING, itr.kind),
                        .payload = &itr};

                    rocprofiler_iterate_buffer_tracing_record_args(
                        _record, iterate_args_callback, &args);
                }

                // insert thread info if it doesn't already exist
                get_thread_id(itr.thread_id);

                auto evt_id = create_event(
                    db,
                    {
                        insert_value("category_id", string_entries.at(category)),
                        insert_value("stack_id", itr.correlation_id.internal),
                        insert_value("parent_stack_id", itr.correlation_id.ancestor),
                        insert_value("correlation_id", itr.correlation_id.external.value),
                        insert_value("extdata", msg),
                    });

                // insert arguments into rocpd_arg table
                for(const auto& arg_info : args)
                {
                    auto demangled_type = common::cxx_demangle(arg_info.arg_type);

                    get_insert_statement(
                        db,
                        "rocpd_arg{{uuid}}",
                        {
                            insert_value("event_id", evt_id),
                            insert_value("position", arg_info.arg_number),
                            insert_value("type", demangled_type),
                            insert_value("name", arg_info.arg_name),
                            insert_value("value", arg_info.arg_value, allow_empty_string{}),
                        });
                }

                if(itr.start_timestamp != itr.end_timestamp)
                {
                    get_insert_statement(db,
                                         "rocpd_region{{uuid}}",
                                         {
                                             insert_value("id", itr.correlation_id.internal),
                                             insert_value("nid", node_id),
                                             insert_value("pid", this_pid),
                                             insert_value("tid", itr.thread_id),
                                             insert_value("start", itr.start_timestamp),
                                             insert_value("end", itr.end_timestamp),
                                             insert_value("name_id", string_entries.at(name)),
                                             insert_value("event_id", evt_id),
                                         });
                }
                else
                {
                    // OMPT instant samples are named by operation (not the "OMPT"
                    // category) so each event keeps its identity on its own track.
                    const auto& track_name =
                        (itr.kind == ROCPROFILER_BUFFER_TRACING_OMPT) ? name : category;
                    auto track_id = get_track_id(
                        db, node_id, this_pid, itr.thread_id, string_entries.at(track_name), "{}");

                    get_insert_statement(db,
                                         "rocpd_sample{{uuid}}",
                                         {
                                             insert_value("id", db.get_sample_id()),
                                             insert_value("track_id", track_id),
                                             insert_value("timestamp", itr.start_timestamp),
                                             insert_value("event_id", evt_id),
                                         });
                }
            }
        }
    };

    auto insert_kfd_data = [&db, &tool_metadata, node_id, this_pid](auto& pmc_ids) {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_info_pmc: kfd");

        struct kfd_pmc_info_t
        {
            rocprofiler_buffer_tracing_kind_t kind        = ROCPROFILER_BUFFER_TRACING_NONE;
            std::string_view                  description = {};
        };

        constexpr auto kfd_info = std::array<kfd_pmc_info_t, 8>{
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
                           "KFD page migration events"},
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT,
                           "KFD page fault events"},
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE,
                           "KFD queue eviction/restore events"},
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                           "KFD unmap from GPU events"},
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
                           "KFD dropped_events events"},
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE,
                           "KFD page migration paired records"},
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT,
                           "KFD page fault paired records"},
            kfd_pmc_info_t{ROCPROFILER_BUFFER_TRACING_KFD_QUEUE,
                           "KFD queue eviction/restore paired records"}};

        auto existing_pmc_ids = std::unordered_set<uint64_t>{};
        for(const auto& itr : tool_metadata.agent_counter_info)
            for(const auto& aitr : itr.second)
                existing_pmc_ids.emplace(aitr.id.handle);

        auto next_kfd_pmc_id = uint64_t{0x7000000000000000ULL};
        for(const auto& info : kfd_info)
        {
            while(existing_pmc_ids.count(next_kfd_pmc_id) > 0 || pmc_ids.count(info.kind) > 0)
                ++next_kfd_pmc_id;

            auto name   = tool_metadata.buffer_names.at(info.kind);
            auto pmc_id = next_kfd_pmc_id++;

            get_insert_statement(db,
                                 "rocpd_info_pmc{{uuid}}",
                                 {
                                     insert_value("id", pmc_id),
                                     insert_value("nid", node_id),
                                     insert_value("pid", this_pid),
                                     insert_value("name", name),
                                     insert_value("symbol", name),
                                     insert_value("description", info.description),
                                     insert_value("component", std::string_view{"rocm"}),
                                     insert_value("value_type", std::string_view{"ABS"}),
                                     insert_value("block", std::string_view{"KFD"}),
                                     insert_value("is_constant", false),
                                     insert_value("is_derived", false),
                                 });

            pmc_ids.emplace(info.kind, pmc_id);
        }
    };

    auto insert_kfd_event_data =
        [&db, &tool_metadata, &string_entries, node_id, this_pid, &get_thread_id](
            const auto& _gen, const auto& _pmc_ids) {
            auto _sqlgenperf_rocpd = get_simple_timer("rocpd_pmc_event: kfd");
            for(auto pitr : _gen)
            {
                auto _deferred = sql::deferred_transaction{db.conn};
                for(const auto& itr : _gen.get(pitr))
                {
                    auto data = std::visit(
                        [&tool_metadata](const auto& record) {
                            using record_type = common::mpl::unqualified_type_t<decltype(record)>;
                            if constexpr(!std::is_same<record_type, std::monostate>::value)
                                return construct_kfd_pmc_event(tool_metadata, record);
                            return kfd_pmc_event_data_t{};
                        },
                        itr.record);

                    if(data.kind == ROCPROFILER_BUFFER_TRACING_NONE) continue;

                    get_thread_id(data.tid);

                    auto category = tool_metadata.buffer_names.at(data.kind);
                    auto evt_id =
                        create_event(db,
                                     {
                                         insert_value("category_id", string_entries.at(category)),
                                         insert_value("stack_id", 0),
                                         insert_value("parent_stack_id", 0),
                                         insert_value("correlation_id", 0),
                                         insert_value("extdata", data.json_data),
                                     });

                    get_insert_statement(db,
                                         "rocpd_region{{uuid}}",
                                         {
                                             insert_value("nid", node_id),
                                             insert_value("pid", this_pid),
                                             insert_value("tid", data.tid),
                                             insert_value("start", data.start),
                                             insert_value("end", data.end),
                                             insert_value("name_id", string_entries.at(data.name)),
                                             insert_value("event_id", evt_id),
                                         });

                    get_insert_statement(db,
                                         "rocpd_pmc_event{{uuid}}",
                                         {
                                             insert_value("event_id", evt_id),
                                             insert_value("pmc_id", _pmc_ids.at(data.kind)),
                                             insert_value("value", data.value),
                                         });
                }
            }
        };

    auto dispatch_to_evt_id    = std::unordered_map<uint64_t, uint64_t>{};
    auto dispatch_to_agent_id  = std::unordered_map<uint64_t, uint64_t>{};
    auto dispatch_to_thread_id = std::unordered_map<uint64_t, uint64_t>{};
    // PC-sample extdata is described by reusable field descriptors defined at file
    // scope (pc_sample_blob_field / append_pc_sample_*_fields / make_pc_sample_layout).
    // The single layout for this method drives both the packed blob bytes and the
    // self-describing schema, so adding or removing a field is a one-line change to a
    // shared group.

    // Insert PC sampling rows, one per sample.
    //
    // Design note:
    // 1) dispatch_to_evt_id / dispatch_to_agent_id / dispatch_to_thread_id are
    //    sparse lookup tables keyed by dispatch_id.
    // 2) PC samples may exist without a matching kernel_dispatch record
    //    (e.g. counter-collection-only traces), so lookup misses are expected.
    // 3) On lookup miss we intentionally store NULL parent/agent/thread context
    //    instead of dropping the sample, preserving data with partial context.
    // 4) Each sample still gets a dedicated rocpd_event + rocpd_blob_event pair,
    //    and parent_id is linked only when dispatch context is available.
    auto disasm_seen             = std::set<std::pair<uint64_t, uint64_t>>{};
    auto insert_pc_sampling_data = [&db,
                                    &cfg,
                                    &tool_metadata,
                                    &disasm_seen,
                                    node_id,
                                    this_pid,
                                    &dispatch_to_evt_id,
                                    &dispatch_to_agent_id,
                                    &dispatch_to_thread_id](const auto& pc_sampling_gen) {
        if(pc_sampling_gen.empty()) return;

        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_gpu_pc_sample");
        auto _deferred         = sql::deferred_transaction{db.conn};

        // Method selects which reusable groups compose the layout: host-trap uses hw_id +
        // wave/workgroup; stochastic additionally appends the snapshot group.
        constexpr bool is_stochastic =
            std::is_same_v<common::mpl::unqualified_type_t<decltype(pc_sampling_gen)>,
                           generator<rocprofiler_tool_pc_sampling_stochastic_record_t>>;
        using tool_record_t = typename generator_value<
            common::mpl::unqualified_type_t<decltype(pc_sampling_gen)>>::type;
        using sample_t = common::mpl::unqualified_type_t<decltype(
            std::declval<tool_record_t>().pc_sample_record)>;

        // One builder composes the complete layout for this method; the same layout drives
        // both the packed bytes and the self-describing schema, so they cannot drift.
        const auto layout     = make_pc_sample_layout<is_stochastic, sample_t>();
        const auto total_size = pc_sample_blob_size(layout);

        // Schema identity is per method so consumers never confuse the two field sets.
        auto schema_name = std::string_view{"pc_sample_extdata_hosttrap"};
        auto schema_desc = std::string_view{"PC sampling arch-specific fields (host-trap, packed)"};
        if constexpr(is_stochastic)
        {
            schema_name = "pc_sample_extdata_stochastic";
            schema_desc = "PC sampling arch-specific fields (packed)";
        }
        const auto ext_schema_id = register_blob_schema(
            db, node_id, this_pid, make_pc_sample_schema(schema_name, schema_desc, layout));

        // Resolve an optional dispatch-indexed value; std::nullopt when the dispatch was not
        // captured by the kernel_dispatch buffer (e.g. counter-collection-only traces).
        auto lookup_dispatch = [](const auto& _map,
                                  uint64_t    _dispatch_id) -> std::optional<uint64_t> {
            if(auto _itr = _map.find(_dispatch_id); _itr != _map.end()) return _itr->second;
            return std::nullopt;
        };

        for(auto pitr : pc_sampling_gen)
        {
            for(const auto& itr : pc_sampling_gen.get(pitr))
            {
                const auto& record = itr.pc_sample_record;

                // Parent dispatch context by dispatch_id; NULL when the dispatch was not captured
                // by the kernel_dispatch buffer.  tid is the real CPU thread that launched the
                // parent dispatch, not the SDK correlation ID.
                auto parent_event_id = lookup_dispatch(dispatch_to_evt_id, record.dispatch_id);
                auto agent_id        = lookup_dispatch(dispatch_to_agent_id, record.dispatch_id);
                auto tid             = lookup_dispatch(dispatch_to_thread_id, record.dispatch_id);

                auto wave_issued  = std::optional<int64_t>{};
                auto wave_count   = std::optional<int64_t>{};
                auto inst_type    = std::optional<int64_t>{};
                auto stall_reason = std::optional<int64_t>{};

                if constexpr(is_stochastic)
                {
                    wave_issued  = static_cast<int64_t>(record.wave_issued);
                    wave_count   = static_cast<int64_t>(record.wave_count);
                    inst_type    = static_cast<int64_t>(record.inst_type);
                    stall_reason = static_cast<int64_t>(record.snapshot.reason_not_issued);
                }

                // Assemble the packed little-endian blob from the layout (reusing a pooled
                // buffer): host-trap fills hw_id + wave/workgroup, stochastic also fills snapshot.
                auto blob_bytes = build_pc_sample_blob(db, layout, total_size, record);

                const auto sample_event_id = create_event(
                    db,
                    {
                        insert_value("correlation_id", record.correlation_id.external.value),
                        insert_nullable_value("parent_id", parent_event_id),
                    });

                get_insert_statement(
                    db,
                    "rocpd_blob_event{{uuid}}",
                    {
                        insert_value("nid", node_id),
                        insert_value("pid", this_pid),
                        insert_value("event_id", static_cast<int64_t>(sample_event_id)),
                        insert_value("schema_id", static_cast<int64_t>(ext_schema_id)),
                        insert_blob_value("blob", std::move(blob_bytes)),
                    });

                get_insert_statement(
                    db,
                    "rocpd_gpu_pc_sample{{uuid}}",
                    {
                        insert_value("timestamp", record.timestamp),
                        insert_value("nid", node_id),
                        insert_value("pid", this_pid),
                        insert_nullable_value("tid", tid),
                        insert_nullable_value("agent_id", agent_id),
                        insert_value("event_id", static_cast<int64_t>(sample_event_id)),
                        insert_value("dispatch_id", record.dispatch_id),
                        insert_value("correlation_id", record.correlation_id.internal),
                        insert_value("exec_mask", record.exec_mask),
                        insert_value("code_object_id", record.pc.code_object_id),
                        insert_value("code_object_offset", record.pc.code_object_offset),
                        insert_nullable_value("wave_issued", wave_issued),
                        insert_nullable_value("inst_type", inst_type),
                        insert_nullable_value("stall_reason", stall_reason),
                        insert_nullable_value("wave_count", wave_count),
                    });

                // Persist disassembly only for PCs the transformation already resolved
                // (inst_index >= 0).  Mark a PC "seen" only when we write its row so a later
                // sample at the same (code_object_id, offset) that does resolve is still persisted.
                if(cfg.complete_isa_decode && itr.inst_index >= 0)
                {
                    auto _disasm_key =
                        std::make_pair(static_cast<uint64_t>(record.pc.code_object_id),
                                       static_cast<uint64_t>(record.pc.code_object_offset));
                    if(disasm_seen.insert(_disasm_key).second)
                    {
                        get_insert_statement(
                            db,
                            "rocpd_disassembly_data{{uuid}}",
                            {
                                insert_value("nid", node_id),
                                insert_value("pid", this_pid),
                                insert_value("code_object_id", record.pc.code_object_id),
                                insert_value("code_object_offset", record.pc.code_object_offset),
                                insert_value("instruction",
                                             tool_metadata.get_instruction(itr.inst_index),
                                             allow_empty_string{}),
                                insert_value("comment",
                                             tool_metadata.get_comment(itr.inst_index),
                                             allow_empty_string{}),
                            });
                    }
                }
            }
        }
    };

    insert_node_data();
    insert_process_data();

    get_stream_id(rocprofiler_stream_id_t{.handle = 0});
    get_queue_id(rocprofiler_queue_id_t{.handle = 0});

    insert_agent_data();
    insert_pmc_data();
    insert_kernel_code_object_data();

    {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_region");
        insert_api_data(hip_api_gen);  // arg string entries can be added to _metadata
        insert_api_data(hsa_api_gen);
        insert_api_data(marker_api_gen);
        insert_api_data(rccl_api_gen);
        insert_api_data(ompt_gen);
        insert_api_data(rocdecode_api_gen);
        insert_api_data(rocshmem_api_gen);
        insert_api_data(hipfile_api_gen);
    }

    insert_kernel_dispatch_data(dispatch_to_evt_id, dispatch_to_agent_id, dispatch_to_thread_id);
    insert_pmc_event_data(dispatch_to_evt_id);
    insert_memory_copy_data(memory_copy_gen);
    insert_graph_launch_data(graph_launch_gen);

    {
        // Register a schema only for the sampling method(s) actually present, then hand each
        // loop its own schema id.  Host-trap uses the compact hw_id + wave/workgroup layout;
        // stochastic adds the snapshot group.  A method that produced no samples
        // registers neither a schema row nor any field metadata.
        const bool has_host_trap  = !pc_sampling_host_trap_gen.empty();
        const bool has_stochastic = !pc_sampling_stochastic_gen.empty();

        if(has_host_trap || has_stochastic)
        {
            if(cfg.complete_isa_decode)
                ROCP_WARNING << "PC sampling instruction disassembly is enabled "
                                "(--complete-isa-decode); the output database size "
                                "may increase significantly.";

            // Each method builds its layout, registers the matching self-describing
            // schema, and writes one blob per sample.
            if(has_host_trap) insert_pc_sampling_data(pc_sampling_host_trap_gen);

            if(has_stochastic) insert_pc_sampling_data(pc_sampling_stochastic_gen);
        }
    }

    {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_memory_allocate");
        insert_memory_alloc_data(memory_alloc_gen);
        insert_memory_alloc_data(scratch_memory_gen);
    }

    {
        auto kfd_pmc_ids = std::unordered_map<rocprofiler_buffer_tracing_kind_t, uint64_t>{};
        insert_kfd_data(kfd_pmc_ids);
        insert_kfd_event_data(kfd_gen, kfd_pmc_ids);
    }

    {
        auto _deferred = sql::deferred_transaction{db.conn};

        {
            auto _sqlgenperf_rocpd = get_simple_timer("rocpd_flush_pending");
            finish_pending_insert_batch(db);
        }

        {
            auto _sqlgenperf_rocpd = get_simple_timer("SQL indexing");
            auto indexes_schema    = read_schema_file(db, ROCPD_SQL_SCHEMA_ROCPD_INDEXES);
            execute_raw_sql_statements(conn, indexes_schema);
        }
    }
}
}  // namespace tool
}  // namespace rocprofiler
