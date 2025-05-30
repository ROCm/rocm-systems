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
#include "lib/common/uuid_v7.hpp"
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
#include "lib/output/sql/common.hpp"
#include "lib/output/sql/deferred_transaction.hpp"

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
#include <regex>
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
    const auto& _repl = get_uuid();
    return std::regex_replace(std::string{inp},
                              std::regex{"\\{\\{uuid\\}\\}"},
                              (_repl.empty()) ? std::string{} : fmt::format("_{}", _repl));
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
          const rocpd_sql_schema_jinja_variables_t* variables,
          const char*                               schema_path,
          const char*                               schema_content,
          void*                                     user_data)
{
    common::consume_args(engine, kind, options, variables, schema_path, schema_content, user_data);

    *static_cast<std::string*>(user_data) = replace_placeholders(schema_content);
}

std::string
read_schema_file(rocpd_sql_schema_kind_t schema_kind)
{
    auto _variables     = common::init_public_api_struct(rocpd_sql_schema_jinja_variables_t{});
    auto _options       = ROCPD_SQL_OPTIONS_NONE;
    auto _schema_result = std::string{};

    _variables.uuid = get_uuid().c_str();
    _variables.guid = get_guid().c_str();

    ROCPD_CHECK(rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                      schema_kind,
                                      _options,
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

struct sql_insert_value
{
    std::string_view name  = {};
    std::string      value = {};
};

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
                return sql_insert_value{_name, std::string{"NULL"}};
            }
        }
        return sql_insert_value{_name, fmt::format("'{}'", _value)};
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

template <template <typename...> class ContainerT, typename... TypesT>
auto
get_insert_statement_impl(std::string_view _table, ContainerT<sql_insert_value, TypesT...>&& _data)
{
    auto fields = std::vector<std::string_view>{};
    auto values = std::vector<std::string_view>{};

    fields.reserve(_data.size() + 1);
    values.reserve(_data.size() + 1);
    for(auto&& itr : _data)
    {
        if(itr.name.empty() && itr.value.empty()) continue;

        fields.emplace_back(itr.name);
        values.emplace_back(itr.value);
    }

    return fmt::format(R"(INSERT INTO {} ({}) VALUES ({});)",
                       replace_uuid(_table),
                       fmt::join(fields.begin(), fields.end(), ", "),
                       fmt::join(values.begin(), values.end(), ", "));
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
get_track_id_impl(sqlite3*         conn,
                  uint64_t         node_id,
                  pid_t            pid,
                  pid_t            tid,
                  uint64_t         name_id,
                  std::string_view extdata,
                  int              line)
{
    auto _track = track_data{node_id, pid, tid, name_id};
    auto itr    = get_tracks().find(_track);
    if(itr == get_tracks().end())
    {
        auto idx  = get_tracks().size() + 1;
        itr       = get_tracks().emplace(_track, idx).first;
        auto stmt = get_insert_statement("rocpd_track{{uuid}}",
                                         {
                                             insert_value("id", idx),
                                             insert_value("nid", node_id),
                                             insert_value("pid", pid),
                                             insert_value("tid", tid),
                                             insert_value("name_id", name_id),
                                             insert_value("extdata", extdata),
                                         });

        sql::execute_raw_sql_statements_impl(conn, stmt, line);
        return idx;
    }

    return itr->second;
}

// so that execute_raw_sql_statements returns the correct line
#define create_event(...) create_event_impl(__VA_ARGS__, __LINE__)
#define get_track_id(...) get_track_id_impl(__VA_ARGS__, __LINE__)
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

void
write_rocpd(
    const output_config&                                                    cfg,
    const metadata&                                                         tool_metadata,
    const std::vector<agent_info>&                                          agent_data,
    const generator<rocprofiler_buffer_tracing_hip_api_ext_record_t>&       hip_api_gen,
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
        auto table_schema = read_schema_file(ROCPD_SQL_SCHEMA_ROCPD_TABLES);
        auto output_file  = get_output_filename(cfg, "results", "db");
        if(fs::exists(output_file)) fs::remove(output_file);

        SQLITE3_CHECK(sqlite3_open_v2(
            output_file.c_str(), &conn, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr));
        SQLITE3_CHECK(sqlite3_busy_handler(conn, &sql::busy_handler, nullptr));

        ROCP_ERROR << fmt::format("Opened result file: {} (UUID={})", output_file, uuid_v7);

        execute_raw_sql_statements(conn, table_schema);

        for(auto itr : {ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
                        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
                        ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
                        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS})
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
                                             insert_value("command", command),
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
                    insert_value("name", itr.name),
                    insert_value("model_name", itr.model_name, allow_empty_string{}),
                    insert_value("vendor_name", itr.vendor_name, allow_empty_string{}),
                    insert_value("product_name", itr.product_name, allow_empty_string{}),
                    insert_value("user_name", itr.product_name, allow_empty_string{}),
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
                const auto* agent = tool_metadata.get_agent(itr.first);
                if(agent == nullptr || !recorded.emplace(aitr.id).second) continue;

                auto json_data = get_json_string([agent](auto& ar) {
                    if(agent) cereal::save(ar, *agent);
                });

                auto _name        = sanitize_sql_string(aitr.name);
                auto _description = sanitize_sql_string(aitr.description);
                auto _block       = sanitize_sql_string(aitr.block);
                auto _expression  = sanitize_sql_string(aitr.expression);

                auto stmt = get_insert_statement(
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
                        insert_value("extdata", json_data),
                    });

                execute_raw_sql_statements(conn, stmt);
            }
        }
    };

    auto insert_kernel_dispatch_data = [&conn, &tool_metadata, &string_entries, node_id, this_pid](
                                           const auto& _gen, auto& dispatch_evt_ids) {
        auto _sqlgenperf_rocpd = get_simple_timer("rocpd_kernel_dispatch");

        for(auto pitr : _gen)
        {
            auto _deferred = sql::deferred_transaction{conn};
            for(auto itr : _gen.get(pitr))
            {
                // insert thread info if it doesn't already exist
                get_thread_id(itr.thread_id);

                auto kind        = tool_metadata.buffer_names.at(itr.kind);
                auto info        = itr.dispatch_info;
                auto kernel_id   = info.kernel_id;
                auto dispatch_id = info.dispatch_id;
                auto corr_id     = itr.correlation_id;
                auto grid        = info.grid_size;
                auto workgroup   = info.workgroup_size;
                auto kern_name = tool_metadata.get_kernel_symbol(kernel_id)->formatted_kernel_name;
                auto stream_id = get_stream_id(itr.stream_id);
                auto queue_id  = get_queue_id(info.queue_id);
                auto region_name =
                    (corr_id.external.value > 0)
                        ? tool_metadata.get_kernel_name(kernel_id, corr_id.external.value)
                        : std::string_view{};

                auto evt_id =
                    create_event(conn,
                                 {
                                     insert_value("category_id", string_entries.at(kind)),
                                     insert_value("stack_id", corr_id.internal),
                                     insert_value("parent_stack_id", corr_id.internal),
                                     insert_value("correlation_id", corr_id.external.value),
                                 });

                if(dispatch_evt_ids.size() < dispatch_id + 1)
                    common::container::resize(dispatch_evt_ids, dispatch_id + 1, 0UL);

                ROCP_CI_LOG_IF(WARNING, dispatch_evt_ids.at(dispatch_id) != 0)
                    << fmt::format("duplicate kernel dispatch id {} :: event_id={}, kernel_id={}, "
                                   "corr_id={}, name='{}'",
                                   dispatch_id,
                                   evt_id,
                                   kernel_id,
                                   corr_id.internal,
                                   kern_name);

                dispatch_evt_ids.at(dispatch_id) = evt_id;

                auto stmt = get_insert_statement(
                    "rocpd_kernel_dispatch{{uuid}}",
                    {
                        insert_value("id", dispatch_id),
                        insert_value("nid", node_id),
                        insert_value("pid", this_pid),
                        insert_value("tid", itr.thread_id),
                        insert_value("agent_id", tool_metadata.get_agent(info.agent_id)->node_id),
                        insert_value("kernel_id", kernel_id),
                        insert_value("dispatch_id", dispatch_id),
                        insert_value("queue_id", queue_id),
                        insert_value("stream_id", stream_id),
                        insert_value("start", itr.start_timestamp),
                        insert_value("end", itr.end_timestamp),
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

                execute_raw_sql_statements(conn, stmt);
            }
        }
    };

    auto insert_pmc_event_data = [&conn,
                                  &tool_metadata,
                                  &string_entries,
                                  &counter_collection_gen,
                                  &kernel_dispatch_gen](auto& dispatch_evt_ids) {
        auto   _sqlgenperf_rocpd = get_simple_timer("rocpd_pmc_event");
        size_t idx               = tool_metadata.pmc_event_offset;
        for(auto ditr : counter_collection_gen)
        {
            auto _deferred = sql::deferred_transaction{conn};
            for(const auto& record : counter_collection_gen.get(ditr))
            {
                const auto& info        = record.dispatch_data.dispatch_info;
                auto        dispatch_id = info.dispatch_id;

                if(kernel_dispatch_gen.empty())
                {
                    auto kind =
                        tool_metadata.buffer_names.at(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);
                    auto kernel_id = info.kernel_id;
                    auto corr_id   = record.dispatch_data.correlation_id;
                    auto kern_name =
                        tool_metadata.get_kernel_symbol(kernel_id)->formatted_kernel_name;

                    auto evt_id =
                        create_event(conn,
                                     {
                                         insert_value("category_id", string_entries.at(kind)),
                                         insert_value("stack_id", corr_id.internal),
                                         insert_value("parent_stack_id", corr_id.internal),
                                         insert_value("correlation_id", corr_id.external.value),
                                     });

                    if(dispatch_evt_ids.size() < dispatch_id + 1)
                        common::container::resize(dispatch_evt_ids, dispatch_id + 1, 0UL);

                    ROCP_CI_LOG_IF(WARNING, dispatch_evt_ids.at(dispatch_id) != 0) << fmt::format(
                        "duplicate kernel dispatch id {} :: event_id={}, kernel_id={}, "
                        "corr_id={}, name='{}'",
                        dispatch_id,
                        evt_id,
                        kernel_id,
                        corr_id.internal,
                        kern_name);

                    dispatch_evt_ids.at(dispatch_id) = evt_id;
                }

                auto evt_id = dispatch_evt_ids.at(dispatch_id);
                for(const auto& count : record.read())
                {
                    auto stmt = get_insert_statement("rocpd_pmc_event{{uuid}}",
                                                     {
                                                         insert_value("id", idx++),
                                                         insert_value("event_id", evt_id),
                                                         insert_value("pmc_id", count.id.handle),
                                                         insert_value("value", count.value),
                                                     });

                    execute_raw_sql_statements(conn, stmt);
                }
            }
        }
    };

    auto insert_memory_copy_data =
        [&conn, &tool_metadata, &string_entries, node_id, this_pid](const auto& _gen) {
            auto   _sqlgenperf_rocpd = get_simple_timer("rocpd_memory_copy");
            size_t copy_idx          = 1;

            for(auto pitr : _gen)
            {
                auto _deferred = sql::deferred_transaction{conn};
                for(auto itr : _gen.get(pitr))
                {
                    // insert thread info if it doesn't already exist
                    get_thread_id(itr.thread_id);

                    auto kind = tool_metadata.buffer_names.at(itr.kind);
                    auto name = tool_metadata.buffer_names.at(itr.kind, itr.operation);

                    auto evt_id = create_event(
                        conn,
                        {
                            insert_value("category_id", string_entries.at(kind)),
                            insert_value("stack_id", itr.correlation_id.internal),
                            insert_value("parent_stack_id", itr.correlation_id.internal),
                            insert_value("correlation_id", itr.correlation_id.external.value),
                        });

                    auto stmt = get_insert_statement(
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
                            insert_value("event_id", evt_id),
                        });

                    execute_raw_sql_statements(conn, stmt);
                }
            }
        };

    auto insert_memory_alloc_data =
        [&conn, &tool_metadata, &string_entries, node_id, this_pid](const auto& _gen) {
            for(auto pitr : _gen)
            {
                auto _deferred = sql::deferred_transaction{conn};
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

                    auto _node_id = std::optional<uint64_t>{};
                    if(_type == "ALLOC")
                    {
                        _node_id = tool_metadata.get_agent(itr.agent_id)->node_id;
                    }

                    auto _stream_id       = get_stream_id(extract_stream_field(itr));
                    auto _queue_id        = get_queue_id(extract_queue_field(itr));
                    auto _address         = extract_address_field(itr);
                    auto _allocation_size = extract_allocation_size_field(itr);

                    auto evt_id = create_event(
                        conn,
                        {
                            insert_value("category_id", string_entries.at(_kind)),
                            insert_value("stack_id", itr.correlation_id.internal),
                            insert_value("parent_stack_id", itr.correlation_id.ancestor),
                            insert_value("correlation_id", itr.correlation_id.external.value),
                        });

                    auto flags = extract_flags_field(itr);

                    auto stmt = get_insert_statement(
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

                    execute_raw_sql_statements(conn, stmt);
                }
            }
        };

    // new string entries argument types and names can be added to _metadata
    auto insert_api_data = [&conn, &tool_metadata, &string_entries, node_id, this_pid](
                               const auto& _gen) {
        for(auto pitr : _gen)
        {
            auto _deferred = sql::deferred_transaction{conn};

            for(auto itr : _gen.get(pitr))
            {
                auto category = tool_metadata.buffer_names.at(itr.kind);
                auto name     = tool_metadata.buffer_names.at(itr.kind, itr.operation);

                auto msg = std::string{"{}"};
                if(itr.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API)
                {
                    if(static_cast<rocprofiler_tracing_operation_t>(itr.operation) !=
                       ROCPROFILER_MARKER_CORE_API_ID_roctxGetThreadId)
                    {
                        // check generatePerfetto.cpp and generateOTF2.cpp, and the marker name in
                        // the view
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
                    conn,
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

                if(itr.start_timestamp != itr.end_timestamp)
                {
                    auto region_stmt =
                        get_insert_statement("rocpd_region{{uuid}}",
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

                    execute_raw_sql_statements(conn, region_stmt);
                }
                else
                {
                    auto track_id = get_track_id(
                        conn, node_id, this_pid, itr.thread_id, string_entries.at(category), "{}");
                    auto sample_stmt =
                        get_insert_statement("rocpd_sample{{uuid}}",
                                             {
                                                 insert_value("id", itr.correlation_id.internal),
                                                 insert_value("track_id", track_id),
                                                 insert_value("timestamp", itr.start_timestamp),
                                                 insert_value("event_id", evt_id),
                                             });

                    execute_raw_sql_statements(conn, sample_stmt);
                }
            }
        }
    };

    auto dispatch_to_evt_id = common::container::stable_vector<uint64_t, 512>{};

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
        insert_api_data(rocdecode_api_gen);
    }

    insert_kernel_dispatch_data(kernel_dispatch_gen, dispatch_to_evt_id);
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
