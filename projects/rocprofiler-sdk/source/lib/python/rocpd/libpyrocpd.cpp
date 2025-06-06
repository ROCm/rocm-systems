// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "libpyrocpd.hpp"
#include "lib/output/format_path.hpp"
#include "lib/python/rocpd/source/common.hpp"
#include "lib/python/rocpd/source/csv.hpp"
#include "lib/python/rocpd/source/functions.hpp"
#include "lib/python/rocpd/source/interop.hpp"
#include "lib/python/rocpd/source/otf2.hpp"
#include "lib/python/rocpd/source/perfetto.hpp"
#include "lib/python/rocpd/source/serialization/sql.hpp"
#include "lib/python/rocpd/source/sql_generator.hpp"
#include "lib/python/rocpd/source/types.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/simple_timer.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/agent_info.hpp"
#include "lib/output/kernel_symbol_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/timestamps.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/details/mpl.hpp>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/perfetto.hpp>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/sql.h>

#include <fmt/format.h>
#include <gotcha/gotcha.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sqlite3.h>

#include <atomic>
#include <future>
#include <mutex>
#include <utility>

namespace py = ::pybind11;

namespace rocpd
{
template <typename Tp>
auto
read_impl(sqlite3* conn, std::string_view conditions)
{
    auto query = std::string_view{};

    if constexpr(std::is_same<Tp, types::node>::value)
        query = "rocpd_info_node";
    else if constexpr(std::is_same<Tp, types::process>::value)
        query = "processes";
    else if constexpr(std::is_same<Tp, types::thread>::value)
        query = "threads";
    else if constexpr(std::is_same<Tp, types::region>::value)
        query = "regions";
    else if constexpr(std::is_same<Tp, types::kernel_dispatch>::value)
        query = "kernels";
    else if constexpr(std::is_same<Tp, types::agent>::value)
        query = "rocpd_info_agent";
    else
        static_assert(rocprofiler::sdk::mpl::assert_false<Tp>::value, "Unsupported read type");

    auto data = std::vector<Tp>{};
    if(conn)
    {
        auto ar = cereal::SQLite3InputArchive{
            conn, fmt::format("SELECT * FROM {} {}", query, conditions)};
        cereal::load(ar, data);
    }
    return data;
}

template <typename Tp>
auto
read_impl(py::object       obj,  // NOLINT(performance-unnecessary-value-param)
          std::string_view conditions)
{
    return read_impl<Tp>(rocpd::interop::get_connection(std::move(obj)), conditions);
}

template <typename Tp, typename ArgT = py::object>
auto
read(ArgT&& arg, std::string_view conditions = {})
{
    return read_impl<Tp>(std::forward<ArgT>(arg), conditions);
}

bool
is_sqlite3_connection(const py::object& obj)
{
    py::module_ sqlite3         = py::module_::import("sqlite3");
    py::object  connection_type = sqlite3.attr("Connection");
    return py::isinstance(obj, connection_type);
}

std::string
get_type_name(const py::object& obj)
{
    return obj.get_type().attr("__name__").cast<std::string>();
}

struct RocpdImportData
{
    RocpdImportData()  = default;
    ~RocpdImportData() = default;

    RocpdImportData(const RocpdImportData&)     = default;
    RocpdImportData(RocpdImportData&&) noexcept = default;
    RocpdImportData& operator=(const RocpdImportData&) = default;
    RocpdImportData& operator=(RocpdImportData&&) noexcept = default;

    RocpdImportData(const py::object& _obj, const std::vector<std::string>& _dbs)
    : connection{_obj}
    , databases{_dbs}
    {
        if(py::isinstance<RocpdImportData>(_obj))
        {
            connection = _obj.cast<RocpdImportData>().connection;
            databases  = _obj.cast<RocpdImportData>().databases;
        }
        else
        {
            if(!is_sqlite3_connection(_obj))
            {
                auto _errmsg = fmt::format("libpyrocpd.RocpdImportData cannot be constructed "
                                           "from provided Python object of type {} (databases: {})",
                                           get_type_name(_obj),
                                           fmt::join(_dbs.begin(), _dbs.end(), ", "));
                ROCP_CI_LOG(WARNING) << _errmsg;
                throw py::type_error{_errmsg};
            }
        }
    }

    size_t size() const { return (connection) ? databases.size() : 0; }
    bool   empty() const { return databases.empty() || !connection; }

    py::object               connection = {};
    std::vector<std::string> databases  = {};
};

struct jinja_variables
{
    py::str uuid = py::none{};
    py::str guid = py::none{};
};
}  // namespace rocpd

PYBIND11_MODULE(libpyrocpd, pyrocpd)
{
    // namespace sdk  = ::rocprofiler::sdk;
    namespace tool   = ::rocprofiler::tool;
    namespace common = ::rocprofiler::common;

    py::doc("ROCm Profiling Data (RocPD) Python bindings");

    // initialize logging with control via ROCPD_LOG_LEVEL env variable
    common::init_logging("ROCPD");

    rocpd::interop::activate_gotcha_bindings();

    if(auto _thrd_safety = sqlite3_threadsafe(); _thrd_safety == 2)
    {
        ROCP_INFO_IF(sqlite3_config(SQLITE_CONFIG_MULTITHREAD) == SQLITE_MISUSE)
            << "sqlite3 cannot be configured to support multithreading";
    }

    pyrocpd.def(
        "format_path",
        [](std::string inp, const std::string& tag) {
            return tool::format_path(std::move(inp), tag);
        },
        "Resolve output keys in filepath");

    py::enum_<rocpd_sql_engine_t>(pyrocpd, "sql_engine", "Load schema engines")
        .value("sqlite3", ROCPD_SQL_ENGINE_SQLITE3);

    py::enum_<rocpd_sql_schema_kind_t>(pyrocpd, "sql_schema", "Load schema kinds")
        .value("rocpd_tables", ROCPD_SQL_SCHEMA_ROCPD_TABLES)
        .value("rocpd_indexes", ROCPD_SQL_SCHEMA_ROCPD_INDEXES)
        .value("rocpd_views", ROCPD_SQL_SCHEMA_ROCPD_VIEWS)
        .value("data_views", ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS)
        .value("summary_views", ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS)
        .value("marker_views", ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS);

    py::enum_<rocpd_sql_options_t>(pyrocpd, "sql_option", "Load schema options")
        .value("none", ROCPD_SQL_OPTIONS_NONE)
        .value("sqlite3_pragma_foreign_keys", ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS);

    py::enum_<tool::agent_indexing>(pyrocpd, "agent_indexing", "enum.Enum")
        .value("node", tool::agent_indexing::node)
        .value("logical_node", tool::agent_indexing::logical_node)
        .value("logical_node_type", tool::agent_indexing::logical_node_type);

    // demo for creating python bindings to a class
    py::class_<rocpd::types::agent>(pyrocpd, "agent")
        .def_readonly("node_id", &rocpd::types::agent::node_id)
        .def_readonly("logical_node_id", &rocpd::types::agent::logical_node_id)
        .def_readonly("gpu_index", &rocpd::types::agent::gpu_index)
        .def_readonly("name", &rocpd::types::agent::name)
        .def_readonly("user_name", &rocpd::types::agent::user_name)
        .def_readonly("product_name", &rocpd::types::agent::product_name);

    py::class_<rocpd::types::node>(pyrocpd, "node")
        .def(py::init<>())
        .def_readonly("nid", &rocpd::types::node::id)
        .def_readonly("hash", &rocpd::types::node::hash)
        .def_readonly("machine_id", &rocpd::types::node::machine_id)
        .def_readonly("hostname", &rocpd::types::node::hostname)
        .def_readonly("system_name", &rocpd::types::node::system_name)
        .def_readonly("system_release", &rocpd::types::node::release)
        .def_readonly("system_version", &rocpd::types::node::version);

    py::class_<rocpd::types::process>(pyrocpd, "process")
        .def(py::init<>())
        .def_readonly("nid", &rocpd::types::process::nid)
        .def_readonly("machine_id", &rocpd::types::process::machine_id)
        .def_readonly("hostname", &rocpd::types::process::hostname)
        .def_readonly("system_name", &rocpd::types::process::system_name)
        .def_readonly("system_release", &rocpd::types::process::system_release)
        .def_readonly("system_version", &rocpd::types::process::system_version)
        .def_readonly("ppid", &rocpd::types::process::ppid)
        .def_readonly("pid", &rocpd::types::process::pid)
        .def_readonly("init", &rocpd::types::process::init)
        .def_readonly("start", &rocpd::types::process::start)
        .def_readonly("end", &rocpd::types::process::end)
        .def_readonly("fini", &rocpd::types::process::fini)
        .def_readonly("command", &rocpd::types::process::command);

    py::class_<rocpd::types::thread>(pyrocpd, "thread")
        .def(py::init<>())
        .def_readonly("nid", &rocpd::types::thread::nid)
        .def_readonly("machine_id", &rocpd::types::thread::machine_id)
        .def_readonly("hostname", &rocpd::types::thread::hostname)
        .def_readonly("system_name", &rocpd::types::thread::system_name)
        .def_readonly("system_release", &rocpd::types::thread::system_release)
        .def_readonly("system_version", &rocpd::types::thread::system_version)
        .def_readonly("ppid", &rocpd::types::thread::ppid)
        .def_readonly("pid", &rocpd::types::thread::pid)
        .def_readonly("tid", &rocpd::types::thread::tid)
        .def_readonly("start", &rocpd::types::thread::start)
        .def_readonly("end", &rocpd::types::thread::end)
        .def_readonly("name", &rocpd::types::thread::name);

    py::class_<tool::output_config>(pyrocpd, "output_config", "Output configuration")
        .def(py::init<>())
        .def_readwrite("output_path", &tool::output_config::output_path)
        .def_readwrite("output_file", &tool::output_config::output_file)
        .def_readwrite("tmp_directory", &tool::output_config::tmp_directory)
        .def_readwrite("csv", &tool::output_config::csv_output)
        .def_readwrite("pftrace", &tool::output_config::pftrace_output)
        .def_readwrite("otf2", &tool::output_config::otf2_output)
        .def_readwrite("kernel_rename", &tool::output_config::kernel_rename)
        .def_readwrite("agent_index_value", &tool::output_config::agent_index_value)
        .def_readwrite("group_by_queue", &tool::output_config::group_by_queue)
        .def_readwrite("perfetto_shmem_size_hint", &tool::output_config::perfetto_shmem_size_hint)
        .def_readwrite("perfetto_buffer_size", &tool::output_config::perfetto_buffer_size)
        .def_readwrite("perfetto_backend", &tool::output_config::perfetto_backend)
        .def_readwrite("perfetto_buffer_fill_policy",
                       &tool::output_config::perfetto_buffer_fill_policy);

    py::class_<tool::metadata>(pyrocpd, "metadata")
        .def("set_process_id", &tool::metadata::set_process_id)
        .def("add_marker_message", &tool::metadata::add_marker_message)
        // .def("add_code_object", &tool::metadata::add_code_object)
        // .def("add_kernel_symbol", &tool::metadata::add_kernel_symbol)
        // .def("add_host_function", &tool::metadata::add_host_function)
        .def("add_string_entry", &tool::metadata::add_string_entry)
        .def("add_external_correlation_id", &tool::metadata::add_external_correlation_id)
        .def("add_agent",
             [](tool::metadata* md, const rocpd::types::agent& _agent) {
                 if(!md) return;
                 md->agents.emplace_back(_agent.base());
                 md->agents_map.emplace(_agent.id, _agent.base());
             })
        .def_readwrite("process_id", &tool::metadata::process_id)
        .def_readwrite("parent_process_id", &tool::metadata::parent_process_id)
        .def_readwrite("process_start_ns", &tool::metadata::process_start_ns)
        .def_readwrite("process_end_ns", &tool::metadata::process_end_ns)
        .def_readwrite("agents", &tool::metadata::agents_map)
        .def_readwrite("node_data", &tool::metadata::node_data)
        .def_readwrite("att_filenames", &tool::metadata::att_filenames)
        .def_readwrite("buffer_names", &tool::metadata::buffer_names)
        .def_readwrite("callback_names", &tool::metadata::callback_names)
        .def_readwrite("command_line", &tool::metadata::command_line);

    py::class_<rocpd::jinja_variables>(
        pyrocpd, "schema_jinja_variables", "Variables for jinja substitution")
        .def(py::init<>())
        .def_readwrite("uuid", &rocpd::jinja_variables::uuid)
        .def_readwrite("guid", &rocpd::jinja_variables::guid);

    py::class_<rocpd::RocpdImportData>(pyrocpd, "RocpdImportData", "RocPD database(s) instances")
        .def(py::init<>())
        .def(py::init<rocpd::RocpdImportData>())
        .def(py::init<py::object, std::vector<std::string>>())
        .def_readonly("connection", &rocpd::RocpdImportData::connection)
        .def_readonly("databases", &rocpd::RocpdImportData::databases);

    pyrocpd.def("load_schema",
                [](rocpd_sql_engine_t            engine,
                   rocpd_sql_schema_kind_t       kind,
                   rocpd_sql_options_t           options,
                   const rocpd::jinja_variables& variables) {
                    auto _callback = [](rocpd_sql_engine_t                        _engine,
                                        rocpd_sql_schema_kind_t                   _kind,
                                        rocpd_sql_options_t                       _options,
                                        const rocpd_sql_schema_jinja_variables_t* _variables,
                                        const char*                               _schema_path,
                                        const char*                               _schema_content,
                                        void* _user_data) -> void {
                        rocprofiler::common::consume_args(
                            _engine, _kind, _options, _variables, _schema_path);
                        auto* _data = static_cast<std::string*>(_user_data);
                        if(_data && _schema_content) *_data = std::string{_schema_content};
                    };

                    auto _uuid = std::optional<std::string>{};
                    if(!variables.uuid.is(py::none{}))
                        _uuid = py::cast<std::string>(variables.uuid);

                    auto _guid = std::optional<std::string>{};
                    if(!variables.guid.is(py::none{}))
                        _guid = py::cast<std::string>(variables.guid);

                    auto _rocpd_variables =
                        common::init_public_api_struct(rocpd_sql_schema_jinja_variables_t{});
                    if(_uuid) _rocpd_variables.uuid = _uuid->c_str();
                    if(_guid) _rocpd_variables.guid = _guid->c_str();

                    auto _hints = std::vector<const char*>{};
                    // for(const auto& itr : schema_path_hints)
                    //     _hints.emplace_back(itr.c_str());

                    auto _contents = std::string{};
                    ROCPD_CHECK(rocpd_sql_load_schema(engine,
                                                      kind,
                                                      options,
                                                      &_rocpd_variables,
                                                      _callback,
                                                      _hints.data(),
                                                      _hints.size(),
                                                      &_contents));
                    return _contents;
                });

    // NOLINTBEGIN(performance-unnecessary-value-param)

    // function which maps the python sqlite3.Connection object to the sqlite3*
    // pointer
    pyrocpd.def(
        "connect",
        [](std::string dbpath, py::args args, py::kwargs kwargs) {
            // import the sqlite3 module
            auto sqlite3_mod = py::module_::import("sqlite3");
            auto ret         = sqlite3_mod.attr("connect")(dbpath, *args, **kwargs);

            auto* db = rocpd::interop::map_connection(ret);
            // this is a no-op right now
            if(db) rocpd::functions::define_for_database(db);

            return ret;
        },
        "Open a database connection");

    pyrocpd.def(
        "write_perfetto",
        [](rocpd::RocpdImportData& data, const tool::output_config& output_cfg) -> bool {
            auto _create_agent_index =
                [&output_cfg](const rocpd::types::agent& _agent) -> tool::agent_index {
                auto ret_index = tool::create_agent_index(
                    output_cfg.agent_index_value,
                    _agent.node_id,                                      // absolute index
                    static_cast<uint32_t>(_agent.logical_node_id),       // relative index
                    static_cast<uint32_t>(_agent.logical_node_type_id),  // type-relative index
                    std::string_view(_agent.type));
                return ret_index;
            };
            // ORDER BY expression for kernel dispatches
            constexpr auto kernels_order_by =
                "agent_abs_index ASC, stream_id ASC, queue_id ASC, start ASC, end DESC";

            constexpr auto region_order_by = "start ASC, end DESC";
            constexpr auto sample_order_by = "timestamp ASC";

            auto perfetto_session = rocpd::output::PerfettoSession{output_cfg};
            auto sqlgen_perf      = common::simple_timer{
                fmt::format("Perfetto generation from {} SQL database(s)", data.size())};
            for(auto obj : {data.connection})
            {
                auto* conn  = rocpd::interop::get_connection(std::move(obj));
                auto  nodes = rocpd::read<rocpd::types::node>(conn);

                for(const auto& nitr : nodes)
                {
                    auto agents = rocpd::read<rocpd::types::agent>(
                        conn, fmt::format("WHERE guid = '{}' AND nid = {}", nitr.guid, nitr.id));
                    auto processes = rocpd::read<rocpd::types::process>(
                        conn, fmt::format("WHERE guid = '{}' AND nid = {}", nitr.guid, nitr.id));

                    for(const auto& pitr : processes)
                    {
                        ROCP_FATAL_IF(pitr.nid != nitr.id || pitr.guid != nitr.guid)
                            << fmt::format("Found process with a mismatched nid/guid. process: "
                                           "{}/{} vs. node: {}/{}",
                                           pitr.nid,
                                           pitr.guid,
                                           nitr.id,
                                           nitr.guid);
                        auto select_guid_nid_pid = [&nitr, &pitr](std::string_view tbl) {
                            return fmt::format("SELECT * FROM {} WHERE guid = '{}' AND nid "
                                               "= {} AND pid = {}",
                                               tbl,
                                               pitr.guid,
                                               nitr.id,
                                               pitr.pid);
                        };

                        auto _sqlgen_perft = common::simple_timer{fmt::format(
                            "Perfetto generation from SQL for process {} (total)", pitr.pid)};

                        auto kernels = rocpd::sql_generator<rocpd::types::kernel_dispatch>{
                            conn, select_guid_nid_pid("kernels"), kernels_order_by};

                        auto memory_allocations =
                            rocpd::sql_generator<rocpd::types::memory_allocation>{
                                conn, select_guid_nid_pid("memory_allocations")};

                        auto memory_copies = rocpd::sql_generator<rocpd::types::memory_copies>{
                            conn, select_guid_nid_pid("memory_copies")};

                        auto counters = rocpd::sql_generator<rocpd::types::counter>{
                            conn, select_guid_nid_pid("counters_collection")};

                        auto regions = rocpd::sql_generator<rocpd::types::region>{
                            conn, select_guid_nid_pid("regions"), region_order_by};

                        auto samples = rocpd::sql_generator<rocpd::types::sample>{
                            conn, select_guid_nid_pid("samples"), sample_order_by};

                        auto threads = rocpd::sql_generator<rocpd::types::thread>{
                            conn, select_guid_nid_pid("threads")};

                        // absolute_index |-> (agent, agent_index)
                        auto agents_map =
                            std::unordered_map<uint64_t,
                                               std::pair<rocpd::types::agent, tool::agent_index>>{};

                        for(const auto& itr : agents)
                        {
                            auto new_index = _create_agent_index(itr);
                            agents_map.emplace(itr.absolute_index, std::make_pair(itr, new_index));
                        }

                        ROCP_TRACE << "Starting Perfetto generation from SQL for process "
                                   << pitr.pid;
                        auto _sqlgen_perfw = common::simple_timer{fmt::format(
                            "Perfetto generation from SQL for process {} (write)", pitr.pid)};
                        rocpd::output::write_perfetto(perfetto_session,
                                                      pitr,
                                                      agents_map,
                                                      threads,
                                                      regions,
                                                      samples,
                                                      kernels,
                                                      memory_copies,
                                                      memory_allocations,
                                                      counters);
                    }
                }
            }
            return true;
        },
        "Write pftrace output file from rocpd SQLite3 database");

    pyrocpd.def(
        "write_csv",
        [](rocpd::RocpdImportData& data, const rocprofiler::tool::output_config& output_cfg) {
            auto sqlgen_csv = common::simple_timer{
                fmt::format("CSV generation from {} SQL database(s)", data.size())};

            if(data.empty()) return;

            auto csv_manager = rocpd::output::CsvManager{output_cfg};

            for(auto obj : {data.connection})
            {
                auto* conn  = rocpd::interop::get_connection(std::move(obj));
                auto  nodes = rocpd::read<rocpd::types::node>(conn);

                for(const auto& nitr : nodes)
                {
                    auto agents = rocpd::read<rocpd::types::agent>(
                        conn, fmt::format("WHERE guid = '{}' AND nid = {}", nitr.guid, nitr.id));
                    auto processes = rocpd::read<rocpd::types::process>(
                        conn, fmt::format("WHERE guid = '{}' AND nid = {}", nitr.guid, nitr.id));

                    for(const auto& pitr : processes)
                    {
                        ROCP_FATAL_IF(pitr.nid != nitr.id || pitr.guid != nitr.guid)
                            << fmt::format("Found process with a mismatched nid/guid. process: "
                                           "{}/{} vs. node: {}/{}",
                                           pitr.nid,
                                           pitr.guid,
                                           nitr.id,
                                           nitr.guid);
                        auto _sqlgen_csv = common::simple_timer{fmt::format(
                            "CSV generation from SQL for process {} (total)", pitr.pid)};

                        auto select_guid_nid_pid = [&nitr, &pitr](std::string_view tbl,
                                                                  std::string_view
                                                                      where_extra_condition = {}) {
                            return fmt::format(
                                "SELECT * FROM {} WHERE guid = '{}' AND nid = {} AND pid = {} {}",
                                tbl,
                                pitr.guid,
                                nitr.id,
                                pitr.pid,
                                where_extra_condition);
                        };

                        rocpd::output::write_agent_info_csv(csv_manager, agents);

                        constexpr auto region_order_by = "start ASC, end DESC";

                        auto kernels = rocpd::sql_generator<rocpd::types::kernel_dispatch>{
                            conn, select_guid_nid_pid("kernels"), region_order_by};
                        auto memory_copies = rocpd::sql_generator<rocpd::types::memory_copies>{
                            conn, select_guid_nid_pid("memory_copies"), region_order_by};
                        auto memory_allocations =
                            rocpd::sql_generator<rocpd::types::memory_allocation>{
                                conn, select_guid_nid_pid("memory_allocations"), region_order_by};
                        auto hip_api_calls = rocpd::sql_generator<rocpd::types::region>{
                            conn,
                            select_guid_nid_pid("regions", "AND category LIKE 'HIP_%'"),
                            region_order_by};
                        auto hsa_api_calls = rocpd::sql_generator<rocpd::types::region>{
                            conn,
                            select_guid_nid_pid("regions", "AND category LIKE 'HSA_%'"),
                            region_order_by};
                        auto marker_api_calls = rocpd::sql_generator<rocpd::types::region>{
                            conn,
                            select_guid_nid_pid("regions_and_samples",
                                                "AND category LIKE 'MARKER_%'"),
                            region_order_by};
                        auto counters_calls = rocpd::sql_generator<rocpd::types::counter>{
                            conn, select_guid_nid_pid("counters_collection"), region_order_by};
                        auto scratch_memory_calls =
                            rocpd::sql_generator<rocpd::types::scratch_memory>{
                                conn, select_guid_nid_pid("scratch_memory"), region_order_by};
                        auto rccl_calls = rocpd::sql_generator<rocpd::types::region>{
                            conn,
                            select_guid_nid_pid("regions", "AND category LIKE 'RCCL_%'"),
                            region_order_by};
                        auto rocdecode_calls = rocpd::sql_generator<rocpd::types::region>{
                            conn,
                            select_guid_nid_pid("regions", "AND category LIKE 'ROCDECODE_%'"),
                            region_order_by};
                        auto rocjpeg_calls = rocpd::sql_generator<rocpd::types::region>{
                            conn,
                            select_guid_nid_pid("regions", "AND category LIKE 'ROCJPEG_%'"),
                            region_order_by};

                        rocpd::output::write_csvs(csv_manager,
                                                  kernels,
                                                  memory_copies,
                                                  memory_allocations,
                                                  hip_api_calls,
                                                  hsa_api_calls,
                                                  marker_api_calls,
                                                  counters_calls,
                                                  scratch_memory_calls,
                                                  rccl_calls,
                                                  rocdecode_calls,
                                                  rocjpeg_calls);
                    }
                }
            }
        },
        "Write trace data to CSV files");

    pyrocpd.def(
        "write_otf2",
        [](rocpd::RocpdImportData& data, const tool::output_config& output_cfg) {
            auto _create_agent_index =
                [&output_cfg](const rocpd::types::agent& _agent) -> tool::agent_index {
                auto ret_index = tool::create_agent_index(
                    output_cfg.agent_index_value,
                    _agent.node_id,                                      // absolute index
                    static_cast<uint32_t>(_agent.logical_node_id),       // relative index
                    static_cast<uint32_t>(_agent.logical_node_type_id),  // type-relative index
                    std::string_view(_agent.type));
                return ret_index;
            };

            constexpr auto kernels_order_by =
                "agent_abs_index ASC, stream_id ASC, queue_id ASC, start ASC, end DESC";

            // to initialise the OTF@ session properly we need to know:
            // (1) the process with the earliest start time
            // (2) find the process with the longest duration
            uint64_t min_start_time = std::numeric_limits<uint64_t>::max();
            uint64_t max_fini_time  = 0;
            for(auto obj : {data.connection})
            {
                auto* conn = rocpd::interop::get_connection(std::move(obj));

                // min start
                sqlite3_stmt* _stmt_min_start;
                sqlite3_prepare_v2(
                    conn, "SELECT MIN(start) FROM processes;", -1, &_stmt_min_start, nullptr);
                uint64_t _min_start_time = std::numeric_limits<uint64_t>::max();
                if(sqlite3_step(_stmt_min_start) == SQLITE_ROW)
                {
                    _min_start_time =
                        static_cast<uint64_t>(sqlite3_column_int64(_stmt_min_start, 0));
                }

                sqlite3_finalize(_stmt_min_start);
                if(min_start_time > _min_start_time)
                {
                    min_start_time = _min_start_time;
                }
                //// max fini
                sqlite3_stmt* _stmt_max_fini;
                sqlite3_prepare_v2(
                    conn, "SELECT MAX(fini) FROM processes;", -1, &_stmt_max_fini, nullptr);
                uint64_t _max_fini_time = 0;
                if(sqlite3_step(_stmt_max_fini) == SQLITE_ROW)
                {
                    _max_fini_time = static_cast<uint64_t>(sqlite3_column_int64(_stmt_max_fini, 0));
                }

                sqlite3_finalize(_stmt_max_fini);
                if(max_fini_time < _max_fini_time)
                {
                    max_fini_time = _max_fini_time;
                }
            }

            auto otf2_session =
                rocpd::output::OTF2Session(output_cfg, min_start_time, max_fini_time);

            auto sqlgen_otf2 = common::simple_timer{
                fmt::format("OTF2 generation from {} SQL database(s)", data.size())};

            uint16_t _process_counter = 0;
            for(auto obj : {data.connection})
            {
                auto* conn  = rocpd::interop::get_connection(std::move(obj));
                auto  nodes = rocpd::read<rocpd::types::node>(conn);
                for(const auto& nitr : nodes)
                {
                    auto agents = rocpd::read<rocpd::types::agent>(
                        conn, fmt::format("WHERE guid = '{}' AND nid = {}", nitr.guid, nitr.id));
                    auto processes = rocpd::read<rocpd::types::process>(
                        conn, fmt::format("WHERE guid = '{}' AND nid = {}", nitr.guid, nitr.id));

                    // absolute_index |-> (agent, agent_index)
                    auto agents_map = std::unordered_map<uint64_t, rocpd::output::extended_agent>{};

                    for(const auto& itr : agents)
                    {
                        const rocprofiler::tool::agent_index new_index = _create_agent_index(itr);
                        const std::string labeled_name = fmt::format("{}", itr.name);
                        agents_map.emplace(
                            itr.absolute_index,
                            rocpd::output::extended_agent{itr, new_index, labeled_name});
                    }

                    for(const auto& pitr : processes)
                    {
                        ROCP_FATAL_IF(pitr.nid != nitr.id || pitr.guid != nitr.guid)
                            << fmt::format("Found process with a mismatched nid/guid. process: "
                                           "{}/{} vs. node: {}/{}",
                                           pitr.nid,
                                           pitr.guid,
                                           nitr.id,
                                           nitr.guid);

                        auto select_guid_nid_pid =
                            [&nitr, &pitr](std::string_view tbl,
                                           std::string_view where_extra_condition = "") {
                                return fmt::format("SELECT * FROM {} WHERE guid = '{}' AND "
                                                   "nid = {} AND pid = {} {}",
                                                   tbl,
                                                   pitr.guid,
                                                   nitr.id,
                                                   pitr.pid,
                                                   where_extra_condition);
                            };

                        constexpr auto region_order_by = "start ASC, end DESC";

                        auto _sqlgen_otf2 = common::simple_timer{fmt::format(
                            "OTF2 generation from SQL for process {} (total)", pitr.pid)};

                        auto kernels = rocpd::sql_generator<rocpd::types::kernel_dispatch>{
                            conn, select_guid_nid_pid("kernels"), kernels_order_by};

                        auto memory_allocations =
                            rocpd::sql_generator<rocpd::types::memory_allocation>{
                                conn, select_guid_nid_pid("memory_allocations"), region_order_by};

                        auto memory_copies = rocpd::sql_generator<rocpd::types::memory_copies>{
                            conn, select_guid_nid_pid("memory_copies"), region_order_by};

                        auto regions = rocpd::sql_generator<rocpd::types::region>{
                            conn, select_guid_nid_pid("regions"), region_order_by};

                        auto threads = rocpd::sql_generator<rocpd::types::thread>{
                            conn, select_guid_nid_pid("threads")};

                        ROCP_TRACE << "Starting OTF2 generation from SQL for process " << pitr.pid;
                        auto _sqlgen_perfw = common::simple_timer{fmt::format(
                            "OTF2 generation from SQL for process {} (write)", pitr.pid)};
                        rocpd::output::write_otf2(otf2_session,
                                                  pitr,
                                                  _process_counter,
                                                  agents_map,
                                                  threads,
                                                  regions,
                                                  kernels,
                                                  memory_copies,
                                                  memory_allocations);
                        _process_counter++;
                    }
                }
            }
        },
        "Write OTF2 output file from rocpd SQLite3 database");

    // NOLINTEND(performance-unnecessary-value-param)

    // reads in all the agent info from database
    pyrocpd.def(
        "read_agents",
        [](const rocpd::RocpdImportData& data, const std::string& conditions) {
            return rocpd::read<rocpd::types::agent>(data.connection, conditions);
        },
        "Reads in the rocprofiler-sdk agents from the database");

    pyrocpd.def(
        "read_nodes",
        [](const rocpd::RocpdImportData& data, const std::string& conditions) {
            return rocpd::read<rocpd::types::node>(data.connection, conditions);
        },
        "Reads in the node information from the database");

    pyrocpd.def(
        "read_processes",
        [](const rocpd::RocpdImportData& data, const std::string& conditions) {
            return rocpd::read<rocpd::types::process>(data.connection, conditions);
        },
        "Reads in the process information from the database");

    pyrocpd.def(
        "read_threads",
        [](const rocpd::RocpdImportData& data, const std::string& conditions) {
            return rocpd::read<rocpd::types::thread>(data.connection, conditions);
        },
        "Reads in the thread information from the database");
}
