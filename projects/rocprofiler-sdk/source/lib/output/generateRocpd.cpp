// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "format_path.hpp"
#include "metadata.hpp"
#include "output_stream.hpp"
#include "stream_info.hpp"
#include "timestamps.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/hasher.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/md5sum.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/simple_timer.hpp"
#include "lib/common/utility.hpp"
#include "lib/common/uuid_v7.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/sql/deferred_transaction.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/perfetto.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/sql.h>
#include <rocprofiler-sdk-rocpd/types.h>

#include <fmt/format.h>
#include <sqlite3.h>
#include <unistd.h>

#include <dlfcn.h>
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <future>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

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

auto main_tid = common::get_tid();

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

std::string
sanitize_sql_string(std::string input)
{
    // Special characters that need to be escaped/sanitized in SQL strings
    constexpr auto ESCAPE_CHARS = std::string_view{";\n\r\t\b\f\v"};
    using strpair_t             = std::pair<std::string_view, std::string_view>;

    for(char c : ESCAPE_CHARS)
        input = replace_all(input, c, "");

    for(auto&& itr : {strpair_t{"'", "''"}})
        input = replace_all(input, itr.first, itr.second);

    return input;
}

std::string
sanitize_sql_string(const char* input)
{
    return (input) ? sanitize_sql_string(std::string{input}) : std::string{};
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

    return sanitize_sql_string(json_ss.str());
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

auto&
get_guid()
{
    static auto _v = std::string{};
    return _v;
}

auto&
get_uuid()
{
    static thread_local auto _v = std::string{};
    return _v;
}

auto
replace_uuid(std::string_view inp)
{
    const auto& _repl       = get_uuid();
    const auto  replacement = (_repl.empty()) ? std::string{} : fmt::format("_{}", _repl);
    return replace_all(std::string{inp}, std::string_view{"{{uuid}}"}, replacement);
}

auto
replace_placeholders(std::string_view inp)
{
    return replace_uuid(inp);
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
    common::consume_args(
        engine, kind, options, schema_version, variables, schema_path, schema_content, user_data);

    *static_cast<std::string*>(user_data) = replace_placeholders(schema_content);
}

rocpd_version_triplet_t
get_schema_version()
{
    return rocpd_version_triplet_t{4, 1, 0};
}

std::string
read_schema_file(rocpd_sql_schema_kind_t schema_kind)
{
    auto _variables     = common::init_public_api_struct(rocpd_sql_schema_jinja_variables_t{});
    auto _options       = ROCPD_SQL_OPTIONS_NONE;
    auto _version       = get_schema_version();  // default schema version
    auto _schema_result = std::string{};

    _variables.uuid = get_uuid().c_str();
    _variables.guid = get_guid().c_str();

    ROCPD_CHECK(rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                      schema_kind,
                                      _options,
                                      _version,
                                      &_variables,
                                      read_file,
                                      nullptr,
                                      0,
                                      &_schema_result));

    return _schema_result;
}

size_t
get_event_id()
{
    static size_t event_id_index = 0;
    return ++event_id_index;
}

auto&
get_tracks()
{
    static auto _tracks = std::unordered_map<track_data, uint64_t>{};
    return _tracks;
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
        _data->emplace_back(
            argument_info{arg_number, common::cxx_demangle(arg_type), arg_name, arg_value_str});
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
        if constexpr(!std::is_same<TraitT, allow_empty_string>::value)
        {
            if(std::string_view{_value}.empty())
            {
                ROCP_CI_LOG(WARNING)
                    << fmt::format("sql text value for {} is empty. Using NULL instead", _name);
                return sql_insert_value{_name,
                                        std::string_view{"?"},
                                        sql_insert_value::make_binder(std::string{"NULL"})};
            }
        }
        // Sanitize string values before embedding into SQL to escape quotes and remove
        // problematic control/separator characters.
        auto _sanitized = sanitize_sql_string(std::string{_value});
        return sql_insert_value{
            _name, std::string_view{"?"}, sql_insert_value::make_binder(_sanitized)};
    }
    else
    {
        return sql_insert_value{
            _name, std::string_view{"?"}, sql_insert_value::make_binder(_value)};
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

template <template <typename...> class ContainerT, typename... TypesT>
auto
get_insert_statement_impl(std::string_view _table, ContainerT<sql_insert_value, TypesT...>&& _data)
{
    auto fields  = std::vector<std::string_view>{};
    auto values  = std::vector<std::string_view>{};
    auto binders = std::vector<sql_insert_value::binder_type>{};

    fields.reserve(_data.size() + 1);
    values.reserve(_data.size() + 1);
    binders.reserve(_data.size() + 1);
    for(auto&& itr : _data)
    {
        if(itr.name.empty() && itr.value.empty()) continue;

        fields.emplace_back(itr.name);
        values.emplace_back(itr.value);
        binders.emplace_back(itr.binder);
    }

    return sql::bind_statement{fmt::format(R"(INSERT INTO {} ({}) VALUES ({});)",
                                           replace_uuid(_table),
                                           fmt::join(fields.begin(), fields.end(), ", "),
                                           fmt::join(values.begin(), values.end(), ", ")),
                               binders};
}

template <template <typename...> class ContainerT, typename... TypesT>
auto
get_insert_statement(std::string_view _table, ContainerT<sql_insert_value, TypesT...>&& _data)
{
    return get_insert_statement_impl(_table,
                                     std::forward<ContainerT<sql_insert_value, TypesT...>>(_data));
}

auto
get_insert_statement(std::string_view _table, std::initializer_list<sql_insert_value>&& _data)
{
    return get_insert_statement_impl(_table,
                                     std::forward<std::initializer_list<sql_insert_value>>(_data));
}

auto
create_event_impl(sqlite3* conn, std::initializer_list<sql_insert_value>&& _data, int line)
{
    auto evt_id    = get_event_id();
    auto _data_vec = std::vector<sql_insert_value>{};

    _data_vec.reserve(_data.size() + 1);
    _data_vec.emplace_back(insert_value("id", evt_id));
    for(auto&& itr : _data)
        _data_vec.emplace_back(itr);

    auto event_stmt = get_insert_statement("rocpd_event{{uuid}}", std::move(_data_vec));

    sql::execute_raw_sql_statements_impl(conn, event_stmt, line);

    return evt_id;
}

uint64_t
get_track_id_impl(int line, sqlite3* conn, track_data&& track)
{
    auto itr = get_tracks().find(track);
    if(itr == get_tracks().end())
    {
        auto idx  = get_tracks().size() + 1;
        itr       = get_tracks().emplace(track, idx).first;
        auto stmt = get_insert_statement("rocpd_track{{uuid}}", track.get_insert_values());

        sql::execute_raw_sql_statements_impl(conn, stmt, line);
        return idx;
    }

    return itr->second;
}

constexpr auto phase_none  = ROCPROFILER_CALLBACK_PHASE_NONE;
constexpr auto phase_enter = ROCPROFILER_CALLBACK_PHASE_ENTER;
constexpr auto phase_exit  = ROCPROFILER_CALLBACK_PHASE_EXIT;
constexpr auto phase_last  = ROCPROFILER_CALLBACK_PHASE_LAST;

int64_t
get_timestamp_id_impl(sqlite3*                     conn,
                      rocprofiler_timestamp_t      timestamp,
                      rocprofiler_callback_phase_t phase,
                      uint64_t                     track_id,
                      int                          line)
{
    using phase_value_type = std::underlying_type_t<rocprofiler_callback_phase_t>;

    if(phase < phase_none || phase >= phase_last)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "timestamp {} in track {} has an unsupported phase {} (expected {} <= x < {}). "
            "Treating the phase as NONE (i.e. 0)",
            timestamp,
            track_id,
            static_cast<phase_value_type>(phase),
            static_cast<phase_value_type>(phase_none),
            static_cast<phase_value_type>(phase_last));

        phase = phase_none;
    }

    auto stmt = get_insert_statement("rocpd_timestamp{{uuid}}",
                                     {insert_value("value", timestamp),
                                      insert_value("phase", static_cast<phase_value_type>(phase)),
                                      insert_value("track_id", track_id)});
    sql::execute_raw_sql_statements_impl(conn, stmt, line);

    return sqlite3_last_insert_rowid(conn);
}

// so that execute_raw_sql_statements returns the correct line
#define create_event(...)     create_event_impl(__VA_ARGS__, __LINE__)
#define get_track_id(...)     get_track_id_impl(__LINE__, __VA_ARGS__)
#define get_timestamp_id(...) get_timestamp_id_impl(__VA_ARGS__, __LINE__)
}  // namespace

size_t
track_data::hash() const
{
    return common::fnv1a_hasher::combine(
        nid, ppid, pid, tid, agent_id, queue_id, stream_id, name_id, extdata);
}

std::vector<sql_insert_value>
track_data::get_insert_values() const
{
    return std::vector<sql_insert_value>{
        insert_value("nid", nid),
        insert_value("ppid", ppid),
        insert_value("pid", pid),
        insert_value("tid", tid),
        insert_value("agent_id", agent_id),
        insert_value("queue_id", queue_id),
        insert_value("stream_id", stream_id),
        insert_value("name_id", name_id),
        insert_value("extdata", extdata),
    };
}

bool
operator==(const track_data& lhs, const track_data& rhs)
{
    return (lhs.tie() == rhs.tie());
}

namespace
{
#define GENERATE_FIELD_ACCESSOR(FUNC_NAME, FIELD_NAME, DATA_TYPE)                                  \
    template <typename Tp, typename Up = Tp>                                                       \
    auto FUNC_NAME(const Tp& _data, int)                                                           \
        ->decltype(std::declval<Up>().FIELD_NAME, std::optional<DATA_TYPE>{})                      \
    {                                                                                              \
        return _data.FIELD_NAME;                                                                   \
    }                                                                                              \
                                                                                                   \
    template <typename Tp, typename Up = Tp>                                                       \
    std::optional<DATA_TYPE> FUNC_NAME(const Tp&, long)                                            \
    {                                                                                              \
        return std::nullopt;                                                                       \
    }                                                                                              \
                                                                                                   \
    template <typename Tp>                                                                         \
    std::optional<DATA_TYPE> FUNC_NAME(const Tp& _data)                                            \
    {                                                                                              \
        return FUNC_NAME(_data, 0);                                                                \
    }

GENERATE_FIELD_ACCESSOR(extract_stream_field, stream_id, rocprofiler_stream_id_t)
GENERATE_FIELD_ACCESSOR(extract_queue_field, queue_id, rocprofiler_queue_id_t)
GENERATE_FIELD_ACCESSOR(extract_allocation_size_field, allocation_size, uint64_t)
GENERATE_FIELD_ACCESSOR(extract_address_field, address, rocprofiler_address_t)
GENERATE_FIELD_ACCESSOR(extract_flags_field, flags, rocprofiler_scratch_alloc_flag_t)

constexpr auto null_stream_id = rocprofiler_stream_id_t{.handle = 0};
constexpr auto null_queue_id  = rocprofiler_queue_id_t{.handle = 0};
constexpr auto null_agent_id  = rocprofiler_agent_id_t{.handle = 0};
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
    const generator<rocprofiler_buffer_tracing_rccl_api_record_t>&          rccl_api_gen,
    const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& rocdecode_api_gen,
    const generator<tool_counter_record_t>&                                 counter_collection_gen)
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

    sqlite3* conn = nullptr;

    {
        const auto& mach_id = tool_metadata.node_data.machine_id;
        auto        ticks   = common::get_process_start_ticks_since_boot(this_pid);
        auto        seed    = common::compute_system_seed(mach_id, this_pid, this_ppid, ticks);
        auto        uuid_v7 = common::generate_uuid_v7(this_pid_init_ns, seed);

        get_guid() = fmt::format("{}", uuid_v7);
        get_uuid() = fmt::format("{}", replace_all(uuid_v7, '-', "_"));

        // reading schemata
        auto table_schema  = read_schema_file(ROCPD_SQL_SCHEMA_ROCPD_TABLES);
        auto output_folder = get_output_folder(cfg, "results", "rpdb");

        if(auto output_index = fs::path{output_folder} / "index.yml"; fs::exists(output_index))
        {
            ROCP_CI_LOG(WARNING) << fmt::format(
                "rocpd index file already exists at {}. User should remove it before proceeding.",
                output_index.string());
        }

        auto output_file =
            fs::path{fs::path{output_folder} /
                     format_path(fmt::format(
                         "{}.{}.{}.{}.db", "{hostname}", "{ppid}", "{pid}", get_uuid()))}
                .string();

        if(fs::exists(output_file)) fs::remove(output_file);

        SQLITE3_CHECK(sqlite3_open_v2(
            output_file.c_str(), &conn, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr));
        SQLITE3_CHECK(sqlite3_busy_handler(conn, &sql::busy_handler, nullptr));

        {
            auto sv = get_schema_version();
            if(sv.major == 4 && sv.minor >= 1)
            {
                // v4.1+: exclusive locking (must precede first transaction) and FK enforcement off
                execute_raw_sql_statements(conn, "PRAGMA locking_mode = EXCLUSIVE;");
                execute_raw_sql_statements(conn, "PRAGMA foreign_keys = OFF;");
            }
        }

        ROCP_ERROR << fmt::format("Opened result file: {} (UUID={})", output_file, uuid_v7);

        execute_raw_sql_statements(conn, table_schema);

        for(auto itr : {
                ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
                ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
                ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS,
                ROCPD_SQL_SCHEMA_ROCPD_METADATA,
            })
        {
            auto views_schema = read_schema_file(itr);
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

    for(const auto& itr : tool_metadata.kernel_rename_map.get())
    {
        add_string_entry(_metadata, itr.first);
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

    auto thread_ids = std::set<rocprofiler_thread_id_t>{};
    auto stream_set = std::unordered_set<rocprofiler_stream_id_t>{};
    auto queue_set  = std::unordered_set<rocprofiler_queue_id_t>{};

    static auto get_stream_id =
        [&conn, &stream_set, &node_id, &this_pid](rocprofiler_stream_id_t val) {
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

                auto stmt = get_insert_statement("rocpd_info_stream{{uuid}}",
                                                 {
                                                     insert_value("id", val.handle),
                                                     insert_value("nid", node_id),
                                                     insert_value("pid", this_pid),
                                                     insert_value("name", _name),
                                                 });

                execute_raw_sql_statements(conn, stmt);
            }

            return val.handle;
        };

    static auto get_queue_id =
        [&conn, &queue_set, &node_id, &this_pid](rocprofiler_queue_id_t val) {
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

                auto stmt = get_insert_statement("rocpd_info_queue{{uuid}}",
                                                 {
                                                     insert_value("id", val.handle),
                                                     insert_value("nid", node_id),
                                                     insert_value("pid", this_pid),
                                                     insert_value("name", _name),
                                                 });

                execute_raw_sql_statements(conn, stmt);
            }

            return val.handle;
        };

    static auto get_thread_id =
        [&conn, &tool_metadata, &thread_ids, &node_id, &this_pid](rocprofiler_thread_id_t val) {
            if(thread_ids.count(val) == 0)
            {
                thread_ids.emplace(val);

                auto stmt =
                    get_insert_statement("rocpd_info_thread{{uuid}}",
                                         {
                                             insert_value("id", val),
                                             insert_value("nid", node_id),
                                             insert_value("ppid", tool_metadata.parent_process_id),
                                             insert_value("pid", this_pid),
                                             insert_value("tid", val),
                                         });

                execute_raw_sql_statements(conn, stmt);
            }

            return val;
        };

    // use this to lookup indexes of strings
    auto string_entries = _metadata.get_string_entries();
    auto category_ids   = std::unordered_map<rocprofiler_buffer_tracing_kind_t, uint64_t>{};

    {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_string");

        auto _deferred = sql::deferred_transaction{conn};
        for(const auto& itr : string_entries)
        {
            auto stmt =
                get_insert_statement("rocpd_string{{uuid}}",
                                     {
                                         insert_value("id", itr.second),
                                         insert_value("string", itr.first, allow_empty_string{}),
                                     });
            execute_raw_sql_statements(conn, stmt);
        }
    }

    auto insert_category_data = [&conn, &tool_metadata](auto& _category_ids) {
        auto _sqlgenperf_rocpd    = get_simple_timer("rocpd_info_category");
        auto _existing_categories = std::unordered_map<std::string_view, uint64_t>{};
        for(const auto& itr : tool_metadata.buffer_names)
        {
            if(auto _category = std::string_view{sdk::get_perfetto_category(itr.value)};
               _existing_categories.count(_category) == 0)
            {
                auto stmt = get_insert_statement("rocpd_info_category{{uuid}}",
                                                 {
                                                     insert_value("name", _category),
                                                 });

                _category_ids[itr.value] = execute_raw_sql_statements(conn, stmt);
            }
            else
            {
                _category_ids[itr.value] =
                    _existing_categories[_category];  // reuse existing category ID
            }
        };
    };

    auto insert_node_data = [&conn, &tool_metadata, node_id, node_hash]() {
        auto        _sqlgenperf_rocpd = get_simple_timer("rocpd_info_node");
        const auto& _info             = tool_metadata.node_data;

        auto stmt = get_insert_statement(
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

        execute_raw_sql_statements(conn, stmt);
    };

    auto insert_process_data = [&conn, &tool_metadata, &cfg, node_id, this_pid]() {
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
        auto _command = sanitize_sql_string(command);

        auto stmt = get_insert_statement("rocpd_info_process{{uuid}}",
                                         {
                                             insert_value("id", this_pid),
                                             insert_value("nid", node_id),
                                             insert_value("ppid", tool_metadata.parent_process_id),
                                             insert_value("pid", this_pid),
                                             insert_value("init", tool_metadata.process_start_ns),
                                             insert_value("fini", tool_metadata.process_end_ns),
                                             insert_value("start", tool_metadata.process_start_ns),
                                             insert_value("end", tool_metadata.process_end_ns),
                                             insert_value("command", _command),
                                             insert_value("environment", json_env),
                                             insert_value("extdata", json_cfg),
                                         });

        execute_raw_sql_statements(conn, stmt);
    };

    auto insert_agent_data = [&conn, &tool_metadata, node_id, this_pid]() {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_info_agent");
        auto _deferred         = sql::deferred_transaction{conn};
        for(auto itr : tool_metadata.agents)
        {
            auto json_info = get_json_string([&itr](auto& ar) { cereal::save(ar, itr); });
            auto type      = std::string_view{"UNK"};
            if(itr.type == ROCPROFILER_AGENT_TYPE_CPU)
                type = "CPU";
            else if(itr.type == ROCPROFILER_AGENT_TYPE_GPU)
                type = "GPU";

            auto _name = (itr.product_name && !std::string_view{itr.product_name}.empty())
                             ? std::make_optional(std::string_view{itr.product_name})
                             : std::nullopt;

            auto stmt = get_insert_statement(
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
                    insert_value("name", _name),
                    insert_value("generic_name", itr.name),
                    insert_value("model_name", itr.model_name, allow_empty_string{}),
                    insert_value("vendor_name", itr.vendor_name, allow_empty_string{}),
                    insert_value("product_name", itr.product_name, allow_empty_string{}),
                    insert_value("extdata", json_info),
                });

            execute_raw_sql_statements(conn, stmt);
        }
    };

    auto insert_kernel_code_object_data = [&conn, &tool_metadata, node_id, this_pid]() {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd kernel info");
        auto _deferred         = sql::deferred_transaction{conn};
        for(const auto& itr : tool_metadata.get_code_objects())
        {
            if(itr.size == 0) continue;

            const auto* _agent = tool_metadata.get_agent(itr.rocp_agent);
            auto        json_data =
                get_json_string([](auto& ar, const auto oitr) { cereal::save(ar, oitr); }, itr);

            auto stmt =
                get_insert_statement("rocpd_info_code_object{{uuid}}",
                                     {
                                         insert_value("id", itr.code_object_id),
                                         insert_value("nid", node_id),
                                         insert_value("pid", this_pid),
                                         insert_value("agent_id", CHECK_NOTNULL(_agent)->node_id),
                                         insert_value("uri", itr.uri),
                                         insert_value("load_base", itr.load_base),
                                         insert_value("load_size", itr.load_size),
                                         insert_value("load_delta", itr.load_delta),
                                         insert_value("extdata", json_data),
                                     });

            execute_raw_sql_statements(conn, stmt);
        }

        for(const auto& itr : tool_metadata.get_kernel_symbols())
        {
            if(itr.kernel_id == 0 && itr.code_object_id == 0) continue;

            auto json_data =
                get_json_string([](auto& ar, const auto oitr) { cereal::save(ar, oitr); }, itr);

            auto stmt = get_insert_statement(
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

            execute_raw_sql_statements(conn, stmt);
        }
    };

    auto insert_pmc_data = [&conn, &tool_metadata, node_id, this_pid]() {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_info_pmc");
        auto recorded          = std::unordered_set<rocprofiler_counter_id_t>{};
        auto _deferred         = sql::deferred_transaction{conn};
        for(const auto& itr : tool_metadata.agent_counter_info)
        {
            for(const auto& aitr : itr.second)
            {
                ROCP_CI_LOG_IF(WARNING, itr.first != aitr.agent_id)
                    << fmt::format("Agent ID mismatch for counter {}: {} vs {}",
                                   aitr.name,
                                   itr.first.handle,
                                   aitr.agent_id.handle);

                const auto* agent = tool_metadata.get_agent(itr.first);
                if(agent == nullptr || !recorded.emplace(aitr.id).second) continue;

                auto json_data = get_json_string(
                    [](auto& ar, const auto& counter_data) { cereal::save(ar, counter_data); },
                    aitr);

                auto _name        = sanitize_sql_string(aitr.name);
                auto _description = sanitize_sql_string(aitr.description);
                auto _block       = sanitize_sql_string(aitr.block);
                auto _expression  = sanitize_sql_string(aitr.expression);

                for(const auto& ditr : aitr.dimension_instances)
                {
                    // auto _counter_id  = ditr.counter_id;
                    auto _instance_id = ditr.instance_id;
                    auto _qualifiers  = std::vector<std::string>{};
                    _qualifiers.reserve(ditr.size() + 1);

                    for(const auto& iitr : ditr)
                    {
                        constexpr auto _prefix         = std::string_view{"dimension_"};
                        auto           _qualifier_name = sanitize_sql_string(iitr.dimension_name);
                        for(auto& qitr : _qualifier_name)
                            qitr = std::tolower(qitr);  // convert to lowercase

                        // remove "dimension_" prefix
                        if(auto _pos = _qualifier_name.find(_prefix);
                           _pos == 0 && _pos + _prefix.length() < _qualifier_name.length())
                            _qualifier_name = _qualifier_name.substr(_pos + _prefix.length());

                        _qualifiers.emplace_back(fmt::format("{}={}", _qualifier_name, iitr.index));
                    }

                    auto _qualifier =
                        fmt::format("{}", fmt::join(_qualifiers.begin(), _qualifiers.end(), ","));
                    auto _instance_name = fmt::format("{}:{}", _name, _qualifier);
                    auto _dim_stmt      = get_insert_statement(
                        "rocpd_info_pmc{{uuid}}",
                        {
                            insert_value("id", _instance_id),
                            insert_value("nid", node_id),
                            insert_value("pid", this_pid),
                            insert_value("target_arch", std::string_view{"GPU"}),
                            insert_value("agent_id", agent->node_id),
                            insert_value("name", _name, allow_empty_string{}),
                            insert_value("symbol", _instance_name, allow_empty_string{}),
                            insert_value("qualifier", _qualifier, allow_empty_string{}),
                            insert_value("description", _description, allow_empty_string{}),
                            insert_value("component", std::string_view{"rocm"}),
                            insert_value("value_type", std::string_view{"ABS"}),
                            insert_value("block", _block, allow_empty_string{}),
                            insert_value("expression", _expression, allow_empty_string{}),
                            insert_value("is_constant", aitr.is_constant),
                            insert_value("is_derived", aitr.is_derived),
                        });

                    execute_raw_sql_statements(conn, _dim_stmt);
                }
            }
        }
    };

    auto insert_kernel_dispatch_data = [&conn,
                                        &tool_metadata,
                                        &category_ids,
                                        &string_entries,
                                        &kernel_dispatch_gen,
                                        &counter_collection_gen,
                                        node_id,
                                        this_ppid,
                                        this_pid](auto& dispatch_evt_ids) {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_kernel_dispatch");

        auto process_dispatch = [&](uint64_t    dispatch_id,
                                    uint64_t    kernel_id,
                                    const auto& corr_id,
                                    const auto& info,
                                    auto        kind,
                                    uint32_t    thread_id,
                                    uint64_t    queue_id,
                                    uint64_t    stream_id,
                                    uint64_t    start_timestamp,
                                    uint64_t    end_timestamp,
                                    const auto& grid,
                                    const auto& workgroup,
                                    bool        enable_duplicate_check) {
            // Skip if we've already processed this dispatch_id
            if(dispatch_evt_ids.size() > dispatch_id && dispatch_evt_ids[dispatch_id] != 0) return;

            auto kern_name = (kernel_id > 0)
                                 ? tool_metadata.get_kernel_symbol(kernel_id)->formatted_kernel_name
                                 : "unknown_kernel";

            auto evt_id = create_event(conn,
                                       {
                                           insert_value("category_id", category_ids.at(kind)),
                                           insert_value("stack_id", corr_id.internal),
                                           insert_value("parent_stack_id", corr_id.internal),
                                           insert_value("correlation_id", corr_id.external.value),
                                       });

            // Ensure dispatch_evt_ids is large enough
            if(dispatch_evt_ids.size() < dispatch_id + 1)
                common::container::resize(dispatch_evt_ids, dispatch_id + 1, 0UL);

            // Check for duplicates if requested
            if(enable_duplicate_check && dispatch_evt_ids.at(dispatch_id) != 0)
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

            dispatch_evt_ids.at(dispatch_id) = evt_id;
            // Unconditionally collect kernel rename data if it is available. rocpd needs to be able
            // to use kernel rename option after data has already been collected, so the kernel
            // rename data needs to be stored in generated db.
            auto region_name =
                (corr_id.external.value > 0 && (enable_duplicate_check || kernel_id > 0))
                    ? tool_metadata.get_kernel_name(kernel_id, true, corr_id.external.value)
                    : std::string_view{};

            auto track_id =
                get_track_id(conn,
                             track_data{
                                 .nid       = node_id,
                                 .ppid      = this_ppid,
                                 .pid       = this_pid,
                                 .tid       = thread_id,
                                 .agent_id  = tool_metadata.get_agent(info.agent_id)->node_id,
                                 .queue_id  = queue_id,
                                 .stream_id = stream_id,
                                 .name_id   = std::nullopt,
                                 .extdata   = std::nullopt,
                             });

            // Insert into kernel dispatch table
            auto                sv_kd = get_schema_version();
            sql::bind_statement stmt;
            if(sv_kd.major == 4 && sv_kd.minor >= 1)
            {
                stmt = get_insert_statement(
                    "rocpd_kernel_dispatch{{uuid}}",
                    {
                        insert_value("id", dispatch_id),
                        insert_value("track_id", track_id),
                        insert_value("kernel_id", kernel_id),
                        insert_value("dispatch_id", dispatch_id),
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
                        insert_value("region_name_id", string_entries.at(region_name)),
                        insert_value("event_id", evt_id),
                    });
            }
            else
            {
                auto start_id = get_timestamp_id(conn, start_timestamp, phase_enter, track_id);
                auto end_id   = get_timestamp_id(conn, end_timestamp, phase_exit, track_id);
                stmt          = get_insert_statement(
                    "rocpd_kernel_dispatch{{uuid}}",
                    {
                        insert_value("id", dispatch_id),
                        insert_value("track_id", track_id),
                        insert_value("kernel_id", kernel_id),
                        insert_value("dispatch_id", dispatch_id),
                        insert_value("start_id", start_id),
                        insert_value("end_id", end_id),
                        insert_value("private_segment_size", info.private_segment_size),
                        insert_value("group_segment_size", info.group_segment_size),
                        insert_value("workgroup_size_x", workgroup.x),
                        insert_value("workgroup_size_y", workgroup.y),
                        insert_value("workgroup_size_z", workgroup.z),
                        insert_value("grid_size_x", grid.x),
                        insert_value("grid_size_y", grid.y),
                        insert_value("grid_size_z", grid.z),
                        insert_value("region_name_id", string_entries.at(region_name)),
                        insert_value("event_id", evt_id),
                    });
            }

            execute_raw_sql_statements(conn, stmt);
        };

        if(kernel_dispatch_gen.empty())
        {
            for(auto pctr : counter_collection_gen)
            {
                auto _deferred = sql::deferred_transaction{conn};
                for(const auto& record : counter_collection_gen.get(pctr))
                {
                    const auto& dispatch_data = record.dispatch_data;
                    const auto& info          = dispatch_data.dispatch_info;

                    // Register thread ID
                    get_thread_id(record.thread_id);

                    // Use buffer category for kernel dispatches
                    auto kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;

                    // Process this dispatch
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
                                     false                             // enable_duplicate_check
                    );
                }
            }
        }
        else
        {
            for(auto pitr : kernel_dispatch_gen)
            {
                auto _deferred = sql::deferred_transaction{conn};
                for(auto itr : kernel_dispatch_gen.get(pitr))
                {
                    // Register thread ID
                    get_thread_id(itr.thread_id);

                    // Process this dispatch
                    process_dispatch(itr.dispatch_info.dispatch_id,             // dispatch_id
                                     itr.dispatch_info.kernel_id,               // kernel_id
                                     itr.correlation_id,                        // corr_id
                                     itr.dispatch_info,                         // info
                                     itr.kind,                                  // kind
                                     itr.thread_id,                             // thread_id
                                     get_queue_id(itr.dispatch_info.queue_id),  // queue_id
                                     get_stream_id(itr.stream_id),              // stream_id
                                     itr.start_timestamp,                       // start_timestamp
                                     itr.end_timestamp,                         // end_timestamp
                                     itr.dispatch_info.grid_size,               // grid
                                     itr.dispatch_info.workgroup_size,          // workgroup
                                     true  // enable_duplicate_check
                    );
                }
            }
        }
    };

    auto insert_pmc_event_data = [&conn, &counter_collection_gen](auto& dispatch_evt_ids) {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_pmc_event");
        for(auto ditr : counter_collection_gen)
        {
            auto _deferred = sql::deferred_transaction{conn};
            for(const auto& itr : counter_collection_gen.get(ditr))
            {
                const auto& info        = itr.dispatch_data.dispatch_info;
                auto        dispatch_id = info.dispatch_id;
                auto        has_evt_id  = (dispatch_id < dispatch_evt_ids.size());

                if(!has_evt_id)
                {
                    ROCP_CI_LOG(WARNING) << fmt::format(
                        "dispatch counter collection is missing event id for dispatch_id={}",
                        dispatch_id);
                    continue;
                }

                auto evt_id = dispatch_evt_ids.at(dispatch_id);
                for(const auto& ritr : itr.read())
                {
                    auto stmt = get_insert_statement("rocpd_pmc_event{{uuid}}",
                                                     {
                                                         insert_value("event_id", evt_id),
                                                         insert_value("pmc_id", ritr.instance_id),
                                                         insert_value("value", ritr.value),
                                                     });

                    execute_raw_sql_statements(conn, stmt);
                }
            }
        }
    };

    auto insert_memory_copy_data =
        [&conn, &tool_metadata, &category_ids, &string_entries, node_id, this_ppid, this_pid](
            const auto& _gen) {
            auto   _sqlgenperf_rocpd = get_simple_timer("rocpd_memory_copy");
            size_t copy_idx          = 1;

            for(auto pitr : _gen)
            {
                auto _deferred = sql::deferred_transaction{conn};
                for(auto itr : _gen.get(pitr))
                {
                    // insert thread info if it doesn't already exist
                    get_thread_id(itr.thread_id);

                    auto name = tool_metadata.buffer_names.at(itr.kind, itr.operation);

                    auto evt_id = create_event(
                        conn,
                        {
                            insert_value("category_id", category_ids.at(itr.kind)),
                            insert_value("stack_id", itr.correlation_id.internal),
                            insert_value("parent_stack_id", itr.correlation_id.internal),
                            insert_value("correlation_id", itr.correlation_id.external.value),
                        });

                    auto track_id = get_track_id(
                        conn,
                        track_data{
                            .nid       = node_id,
                            .ppid      = this_ppid,
                            .pid       = this_pid,
                            .tid       = itr.thread_id,
                            .agent_id  = tool_metadata.get_agent(itr.dst_agent_id)->node_id,
                            .queue_id  = std::nullopt,
                            .stream_id = get_stream_id(itr.stream_id),
                            .name_id   = std::nullopt,
                            .extdata   = std::nullopt,
                        });

                    auto                sv_mc = get_schema_version();
                    sql::bind_statement mc_stmt;
                    if(sv_mc.major == 4 && sv_mc.minor >= 1)
                    {
                        mc_stmt = get_insert_statement(
                            "rocpd_memory_copy{{uuid}}",
                            {
                                insert_value("id", copy_idx++),
                                insert_value("track_id", track_id),
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
                                insert_value("event_id", evt_id),
                            });
                    }
                    else
                    {
                        auto start_id =
                            get_timestamp_id(conn, itr.start_timestamp, phase_enter, track_id);
                        auto end_id =
                            get_timestamp_id(conn, itr.end_timestamp, phase_exit, track_id);
                        mc_stmt = get_insert_statement(
                            "rocpd_memory_copy{{uuid}}",
                            {
                                insert_value("id", copy_idx++),
                                insert_value("track_id", track_id),
                                insert_value("start_id", start_id),
                                insert_value("end_id", end_id),
                                insert_value("name_id", string_entries.at(name)),
                                insert_value("dst_agent_id",
                                             tool_metadata.get_agent(itr.dst_agent_id)->node_id),
                                insert_value("src_agent_id",
                                             tool_metadata.get_agent(itr.src_agent_id)->node_id),
                                insert_value("dst_address", itr.dst_address.value),
                                insert_value("src_address", itr.src_address.value),
                                insert_value("size", itr.bytes),
                                insert_value("event_id", evt_id),
                            });
                    }

                    execute_raw_sql_statements(conn, mc_stmt);
                }
            }
        };

    auto insert_memory_alloc_data = [&conn,
                                     &tool_metadata,
                                     &category_ids,
                                     &string_entries,
                                     node_id,
                                     this_ppid,
                                     this_pid](const auto& _gen) {
        auto address_to_agent_and_size =
            std::unordered_map<rocprofiler_address_t, rocprofiler::agent::index_and_size>{};

        for(auto pitr : _gen)
        {
            auto _deferred = sql::deferred_transaction{conn};
            for(auto itr : _gen.get(pitr))
            {
                // insert thread info if it doesn't already exist
                get_thread_id(itr.thread_id);

                auto _kind = tool_metadata.buffer_names.at(itr.kind);
                auto _name = tool_metadata.buffer_names.at(itr.kind, itr.operation);

                auto _type  = std::string{};
                auto _level = std::string{};

                auto _emit_warning = [_kind, _name]() {
                    ROCP_CI_LOG(WARNING)
                        << fmt::format("rocpd does not know how to classify memory allocation "
                                       "of kind={} and operation={}",
                                       _kind,
                                       _name);
                };

                if(itr.kind == ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
                {
                    auto _operation =
                        static_cast<rocprofiler_memory_allocation_operation_t>(itr.operation);
                    if(_operation == ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE)
                    {
                        _type  = "ALLOC";
                        _level = "REAL";
                    }
                    else if(_operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE)
                    {
                        _type  = "ALLOC";
                        _level = "VIRTUAL";
                    }
                    else if(_operation == ROCPROFILER_MEMORY_ALLOCATION_FREE)
                    {
                        _type  = "FREE";
                        _level = "REAL";
                    }
                    else if(_operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE)
                    {
                        _type  = "FREE";
                        _level = "VIRTUAL";
                    }
                    else
                    {
                        _emit_warning();
                        continue;
                    }
                }
                else if(itr.kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
                {
                    auto _operation =
                        static_cast<rocprofiler_scratch_memory_operation_t>(itr.operation);
                    if(_operation == ROCPROFILER_SCRATCH_MEMORY_ALLOC)
                    {
                        _type  = "ALLOC";
                        _level = "SCRATCH";
                    }
                    else if(_operation == ROCPROFILER_SCRATCH_MEMORY_FREE)
                    {
                        _type  = "FREE";
                        _level = "SCRATCH";
                    }
                    else if(_operation == ROCPROFILER_SCRATCH_MEMORY_ASYNC_RECLAIM)
                    {
                        _type  = "RECLAIM";
                        _level = "SCRATCH";
                    }
                    else
                    {
                        _emit_warning();
                        continue;
                    }
                }
                else
                {
                    _emit_warning();
                    continue;
                }

                auto _stream_id = get_stream_id(extract_stream_field(itr).value_or(null_stream_id));
                auto _queue_id  = get_queue_id(extract_queue_field(itr).value_or(null_queue_id));
                auto _address   = extract_address_field(itr);
                auto _allocation_size = extract_allocation_size_field(itr);
                auto _flags           = extract_flags_field(itr);
                auto _node_id         = std::optional<uint64_t>{};
                auto _address_value =
                    (_address) ? std::make_optional(_address->value) : std::optional<uint64_t>{};
                auto _extdata = (_flags) ? std::make_optional(get_json_string([&_flags](auto& ar) {
                    ar(cereal::make_nvp("flags", *_flags));
                }))
                                         : std::optional<std::string>{};

                if(itr.agent_id != null_agent_id)
                {
                    if(const auto* _agent = tool_metadata.get_agent(itr.agent_id); _agent)
                        _node_id = _agent->node_id;
                    else
                    {
                        ROCP_CI_LOG(ERROR)
                            << fmt::format("nullptr to rocprofiler_agent_id_t{}.handle={}{}",
                                           '{',
                                           itr.agent_id.handle,
                                           '}');
                    }
                }

                // memory allocation counter track
                struct free_memory_information
                {
                    rocprofiler_timestamp_t start_timestamp = 0;
                    rocprofiler_timestamp_t end_timestamp   = 0;
                    rocprofiler_address_t   address         = {.handle = 0};
                };

                if(_type == "ALLOC" && _allocation_size && _node_id)  // alloc operation
                {
                    address_to_agent_and_size.emplace(
                        _address.value_or(rocprofiler_address_t{.handle = 0}),
                        rocprofiler::agent::index_and_size{_node_id.value(),
                                                           _allocation_size.value()});
                }
                else if(_type == "FREE" && _address)  // free operation
                {
                    if(address_to_agent_and_size.count(_address.value()) == 0)
                    {
                        if(_address->handle == 0)
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
                        auto [agent_abs_index, size] = address_to_agent_and_size[_address.value()];
                        _node_id                     = agent_abs_index;
                        _allocation_size             = 0;
                    }
                }
                else
                {
                    ROCP_CI_LOG(WARNING) << "unhandled memory allocation type " << _type;
                }

                auto evt_id = create_event(
                    conn,
                    {
                        insert_value("category_id", category_ids.at(itr.kind)),
                        insert_value("stack_id", itr.correlation_id.internal),
                        insert_value("parent_stack_id", itr.correlation_id.ancestor),
                        insert_value("correlation_id", itr.correlation_id.external.value),
                    });

                auto track_id = get_track_id(conn,
                                             track_data{
                                                 .nid       = node_id,
                                                 .ppid      = this_ppid,
                                                 .pid       = this_pid,
                                                 .tid       = itr.thread_id,
                                                 .agent_id  = _node_id,
                                                 .queue_id  = _queue_id,
                                                 .stream_id = _stream_id,
                                                 .name_id   = std::nullopt,
                                                 .extdata   = std::nullopt,
                                             });

                auto                sv_ma = get_schema_version();
                sql::bind_statement ma_stmt;
                if(sv_ma.major == 4 && sv_ma.minor >= 1)
                {
                    ma_stmt =
                        get_insert_statement("rocpd_memory_allocate{{uuid}}",
                                             {
                                                 insert_value("track_id", track_id),
                                                 insert_value("start", itr.start_timestamp),
                                                 insert_value("end", itr.end_timestamp),
                                                 insert_value("name_id", string_entries.at(_name)),
                                                 insert_value("type", _type),
                                                 insert_value("level", _level),
                                                 insert_value("event_id", evt_id),
                                                 insert_value("address", _address_value),
                                                 insert_value("size", _allocation_size),
                                                 insert_value("extdata", _extdata),
                                             });
                }
                else
                {
                    auto start_id =
                        get_timestamp_id(conn, itr.start_timestamp, phase_enter, track_id);
                    auto end_id = get_timestamp_id(conn, itr.end_timestamp, phase_exit, track_id);
                    ma_stmt =
                        get_insert_statement("rocpd_memory_allocate{{uuid}}",
                                             {
                                                 insert_value("track_id", track_id),
                                                 insert_value("start_id", start_id),
                                                 insert_value("end_id", end_id),
                                                 insert_value("name_id", string_entries.at(_name)),
                                                 insert_value("type", _type),
                                                 insert_value("level", _level),
                                                 insert_value("event_id", evt_id),
                                                 insert_value("address", _address_value),
                                                 insert_value("size", _allocation_size),
                                                 insert_value("extdata", _extdata),
                                             });
                }

                execute_raw_sql_statements(conn, ma_stmt);
            }
        }
    };

    // new string entries argument types and names can be added to _metadata
    auto insert_api_data =
        [&conn, &tool_metadata, &category_ids, &string_entries, node_id, this_ppid, this_pid](
            const auto& _gen) {
            for(auto pitr : _gen)
            {
                auto _deferred = sql::deferred_transaction{conn};

                for(auto itr : _gen.get(pitr))
                {
                    auto name = tool_metadata.buffer_names.at(itr.kind, itr.operation);

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
                                    name);
                                name = message;
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
                        conn,
                        {
                            insert_value("category_id", category_ids.at(itr.kind)),
                            insert_value("stack_id", itr.correlation_id.internal),
                            insert_value("parent_stack_id", itr.correlation_id.ancestor),
                            insert_value("correlation_id", itr.correlation_id.external.value),
                            insert_value("extdata", msg),
                        });

                    // insert arguments into rocpd_arg table
                    for(const auto& arg_info : args)
                    {
                        auto demangled_type = common::cxx_demangle(arg_info.arg_type);

                        auto args_stmt =
                            get_insert_statement("rocpd_arg{{uuid}}",
                                                 {
                                                     insert_value("event_id", evt_id),
                                                     insert_value("position", arg_info.arg_number),
                                                     insert_value("type", demangled_type),
                                                     insert_value("name", arg_info.arg_name),
                                                     insert_value("value", arg_info.arg_value),
                                                 });

                        execute_raw_sql_statements(conn, args_stmt);
                    }

                    auto track_id = get_track_id(conn,
                                                 track_data{
                                                     .nid       = node_id,
                                                     .ppid      = this_ppid,
                                                     .pid       = this_pid,
                                                     .tid       = itr.thread_id,
                                                     .agent_id  = std::nullopt,
                                                     .queue_id  = std::nullopt,
                                                     .stream_id = std::nullopt,
                                                     .name_id   = std::nullopt,
                                                     .extdata   = std::nullopt,
                                                 });

                    auto                sv_rg = get_schema_version();
                    sql::bind_statement region_stmt;
                    if(sv_rg.major == 4 && sv_rg.minor >= 1)
                    {
                        region_stmt = get_insert_statement(
                            "rocpd_region{{uuid}}",
                            {
                                insert_value("id", itr.correlation_id.internal),
                                insert_value("track_id", track_id),
                                insert_value("name_id", string_entries.at(name)),
                                insert_value("start", itr.start_timestamp),
                                insert_value("end", itr.end_timestamp),
                                insert_value("event_id", evt_id),
                            });
                    }
                    else
                    {
                        auto start_id =
                            get_timestamp_id(conn, itr.start_timestamp, phase_enter, track_id);
                        auto end_id =
                            get_timestamp_id(conn, itr.end_timestamp, phase_exit, track_id);
                        region_stmt = get_insert_statement(
                            "rocpd_region{{uuid}}",
                            {
                                insert_value("id", itr.correlation_id.internal),
                                insert_value("track_id", track_id),
                                insert_value("name_id", string_entries.at(name)),
                                insert_value("start_id", start_id),
                                insert_value("end_id", end_id),
                                insert_value("event_id", evt_id),
                            });
                    }

                    execute_raw_sql_statements(conn, region_stmt);
                }
            }
        };

    auto dispatch_to_evt_id = common::container::stable_vector<uint64_t, 512>{};

    insert_node_data();
    insert_process_data();

    get_stream_id(rocprofiler_stream_id_t{.handle = 0});
    get_queue_id(rocprofiler_queue_id_t{.handle = 0});

    insert_category_data(category_ids);
    insert_agent_data();
    insert_pmc_data();
    insert_kernel_code_object_data();

    {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_region");
        insert_api_data(hip_api_gen);  // arg string entries can be added to _metadata
        insert_api_data(hsa_api_gen);
        insert_api_data(marker_api_gen);
        insert_api_data(rccl_api_gen);
        insert_api_data(rocdecode_api_gen);
    }

    insert_kernel_dispatch_data(dispatch_to_evt_id);
    insert_pmc_event_data(dispatch_to_evt_id);
    insert_memory_copy_data(memory_copy_gen);

    {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_memory_allocate");
        insert_memory_alloc_data(memory_alloc_gen);
        insert_memory_alloc_data(scratch_memory_gen);
    }

    {
        auto _sqlgenperf_rocpd = get_simple_timer("SQL indexing");
        auto indexes_schema    = read_schema_file(ROCPD_SQL_SCHEMA_ROCPD_INDEXES);
        execute_raw_sql_statements(conn, indexes_schema);
    }

    SQLITE3_CHECK(sqlite3_close_v2(conn));
}
}  // namespace tool
}  // namespace rocprofiler
