// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// NOLINTBEGIN(readability-function-cognitive-complexity, readability-function-size)

#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/node_info.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/rocpd_processor.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/collectors/gpu_perf_counter/sample.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/collectors/nic/sample.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/thread_info.hpp"

#include <profiler-hub/reader.hpp>
#include <profiler-hub/reader_types.hpp>
#include <profiler-hub/storage.hpp>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/version.h>

#include <gtest/gtest.h>
#include <timemory/settings/settings.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using rocprofsys::agent;
using rocprofsys::agent_manager;
using rocprofsys::agent_type;
using rocprofsys::function_args_t;
using rocprofsys::get_args_string;
using rocprofsys::output_file_registry;
using rocprofsys::trace_cache::ainic_pmc_sample;
using rocprofsys::trace_cache::backtrace_region_sample;
using rocprofsys::trace_cache::gpu_perf_counter_sample;
using rocprofsys::trace_cache::in_time_sample;
using rocprofsys::trace_cache::kernel_dispatch_sample;
using rocprofsys::trace_cache::kfd_sample;
using rocprofsys::trace_cache::memory_copy_sample;
using rocprofsys::trace_cache::metadata_registry;
using rocprofsys::trace_cache::pmc_event_with_sample;
using rocprofsys::trace_cache::region_sample;
using rocprofsys::trace_cache::rocpd_processor_t;
using rocprofsys::trace_cache::scratch_memory_sample;
using rocprofsys::trace_cache::info::gpu_perf_counter_name_entry;
using rocprofsys::trace_cache::info::pmc;
using rocprofsys::trace_cache::info::track;
using cpu_pmc_sample = rocprofsys::pmc::collectors::cpu::sample;
using gpu_pmc_sample = rocprofsys::pmc::collectors::gpu::sample;
using gpu_perf_counter_value =
    rocprofsys::pmc::collectors::gpu_perf_counter::counter_value;
#if(ROCPROFILER_VERSION >= 600)
using rocprofsys::trace_cache::memory_allocate_sample;
#endif

namespace rocprofsys::trace_cache::detail
{
void
set_force_rocpd_metadata_registration_for_tests(bool enabled);
}

struct scoped_force_rocpd_metadata_registration
{
    scoped_force_rocpd_metadata_registration()
    {
        rocprofsys::trace_cache::detail::set_force_rocpd_metadata_registration_for_tests(
            true);
    }
    ~scoped_force_rocpd_metadata_registration()
    {
        rocprofsys::trace_cache::detail::set_force_rocpd_metadata_registration_for_tests(
            false);
    }
};

// rocprof-sys-unit-tests does not link library/thread_info.cpp (see
// source/tests/CMakeLists.txt). post_process_metadata() calls thread_info::get when
// metadata has thread rows; return empty so thread start/end come from metadata_registry
// only.
namespace rocprofsys
{
const std::optional<thread_info>&
thread_info::get(std::int64_t, ThreadIdType)
{
    static const std::optional<thread_info> k_none{};
    return k_none;
}

std::uint64_t
thread_info::get_start() const
{
    return 0;
}

std::uint64_t
thread_info::get_stop() const
{
    return 0;
}
}  // namespace rocprofsys

namespace
{
// Unwrap std::optional for tests. Prefer this over ASSERT_TRUE(opt.has_value())
// followed by *opt — clang-tidy's bugprone-unchecked-optional-access does not
// treat GTest ASSERT macros as a proven guard.
template <typename T>
[[nodiscard]] T
require_optional(std::optional<T> value, const char* message)
{
    if(!value.has_value())
    {
        throw std::runtime_error(message);
    }
    return *std::move(value);
}

struct nic_pmc_spec
{
    const char* name;
    const char* units;
};

pmc
make_nic_metadata_pmc(const nic_pmc_spec& spec)
{
    pmc row{};
    row.type             = agent_type::nic;
    row.agent_type_index = 0;
    row.target_arch      = "NIC";
    row.event_code       = 0;
    row.instance_id      = 0;
    row.name             = spec.name;
    row.symbol           = spec.name;
    row.description      = spec.name;
    row.units            = spec.units;
    row.value_type       = "ABS";
    row.extdata          = "{}";
    return row;
}

constexpr size_t k_nic_pmc_rx_ucast_bytes_idx = 4;
constexpr size_t k_nic_pmc_tx_ucast_bytes_idx = 5;

constexpr std::array k_nic_pmcs{
    nic_pmc_spec{ .name = "nic_rx_ucast_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_tx_ucast_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_rx_cnp_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_tx_cnp_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_rx_ucast_bytes", .units = "bytes" },
    nic_pmc_spec{ .name = "nic_tx_ucast_bytes", .units = "bytes" },
    nic_pmc_spec{ .name = "nic_tx_rdma_ack_timeout", .units = "timeouts" },
    nic_pmc_spec{ .name = "nic_resp_tx_pkt_seq_err", .units = "errors" },
    nic_pmc_spec{ .name = "nic_req_rx_pkt_seq_err", .units = "errors" },
    nic_pmc_spec{ .name = "nic_req_rx_impl_nak_seq_err", .units = "errors" },
};

template <typename PmcRow>
void
expect_nic_pmc_row(const PmcRow& pmc_row)
{
    EXPECT_EQ(pmc_row->target_arch, "NIC") << pmc_row->name;
    ASSERT_NE(pmc_row->agent_info, nullptr) << pmc_row->name;
    EXPECT_EQ(pmc_row->agent_info->agent_type, "NIC") << pmc_row->name;
}

template <typename PmcList>
void
expect_nic_pmc_rows(const PmcList& pmc_infos)
{
    for(const auto& pmc_row : pmc_infos)
    {
        expect_nic_pmc_row(pmc_row);
    }
}

[[nodiscard]] const nic_pmc_spec*
find_nic_pmc_spec(std::string_view name)
{
    for(const auto& entry : k_nic_pmcs)
    {
        if(entry.name == name)
        {
            return &entry;
        }
    }
    return nullptr;
}

template <typename PmcRow>
void
expect_nic_pmc_matches_catalog_entry(const PmcRow& pmc_row)
{
    expect_nic_pmc_row(pmc_row);
    const nic_pmc_spec* spec = find_nic_pmc_spec(pmc_row->name);
    ASSERT_NE(spec, nullptr) << pmc_row->name;
    EXPECT_EQ(pmc_row->symbol, spec->name);
    EXPECT_EQ(pmc_row->units, spec->units);
}

void
expect_all_nic_pmc_names_present(const std::unordered_set<std::string>& seen)
{
    for(const auto& spec : k_nic_pmcs)
    {
        EXPECT_TRUE(seen.count(spec.name)) << "missing PMC " << spec.name;
    }
}

template <typename PmcList>
void
expect_nic_pmc_catalog(const PmcList& pmc_infos)
{
    ASSERT_EQ(pmc_infos.size(), k_nic_pmcs.size());

    std::unordered_set<std::string> seen;
    for(const auto& pmc_row : pmc_infos)
    {
        expect_nic_pmc_matches_catalog_entry(pmc_row);
        seen.insert(pmc_row->name);
    }
    expect_all_nic_pmc_names_present(seen);
}

struct named_pmc_expect
{
    const char* name;
    const char* expected_arch;
    const char* expected_agent_type;
    const char* expected_symbol      = nullptr;
    const char* expected_units       = nullptr;
    const char* expected_description = nullptr;
};

template <typename PmcRow>
void
expect_pmc_row_required_arch_fields(const PmcRow&           pmc_row,
                                    const named_pmc_expect& expected)
{
    EXPECT_EQ(pmc_row->target_arch, expected.expected_arch) << pmc_row->name;
    ASSERT_NE(pmc_row->agent_info, nullptr) << pmc_row->name;
    EXPECT_EQ(pmc_row->agent_info->agent_type, expected.expected_agent_type)
        << pmc_row->name;
}

template <typename PmcRow>
void
expect_pmc_row_optional_fields(const PmcRow& pmc_row, const named_pmc_expect& expected)
{
    if(expected.expected_symbol != nullptr)
    {
        EXPECT_EQ(pmc_row->symbol, expected.expected_symbol) << pmc_row->name;
    }
    if(expected.expected_units != nullptr)
    {
        EXPECT_EQ(pmc_row->units, expected.expected_units) << pmc_row->name;
    }
    if(expected.expected_description != nullptr)
    {
        EXPECT_EQ(pmc_row->description, expected.expected_description) << pmc_row->name;
    }
}

template <typename PmcRow>
void
expect_pmc_row_named_fields(const PmcRow& pmc_row, const named_pmc_expect& expected)
{
    expect_pmc_row_required_arch_fields(pmc_row, expected);
    expect_pmc_row_optional_fields(pmc_row, expected);
}

template <typename PmcList>
void
expect_named_pmc_arch(const PmcList& pmc_infos, const named_pmc_expect& expected)
{
    bool found = false;
    for(const auto& pmc_row : pmc_infos)
    {
        if(pmc_row->name != expected.name)
        {
            continue;
        }
        found = true;
        expect_pmc_row_named_fields(pmc_row, expected);
    }
    EXPECT_TRUE(found) << "missing PMC " << expected.name;
}

struct agent_fields_expect
{
    const char* agent_type;
    const char* name;
    const char* model_name;
    const char* vendor_name;
    const char* product_name;
};

struct agent_pmc_spec
{
    agent_type  type;
    size_t      agent_type_index = 0;
    const char* name;
    const char* target_arch;
    const char* description = nullptr;
};
}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// rocpd_processor.cpp — integration: processor → flush → reader read-back
//
// Each TEST_F below is split into two parts:
//   Prepare — build metadata, call rocpd_processor_t::handle(), finalize, open reader
//   Validate — assert via profiler_hub::reader_t (no direct SQLite access)
// ═══════════════════════════════════════════════════════════════════════════

class rocpd_write_read_test_interface : public ::testing::Test
{
protected:
    static constexpr size_t k_node_id       = 1;
    static constexpr size_t k_pid           = 200;
    static constexpr size_t k_ppid          = 100;
    static constexpr size_t k_thread_id     = 300;
    static constexpr size_t k_queue_id      = 10;
    static constexpr size_t k_stream_id     = 20;
    static constexpr size_t k_process_start = 1000;
    static constexpr size_t k_process_end   = 9000;

    void SetUp() override
    {
        m_temp_dir = std::filesystem::temp_directory_path() /
                     ("rocpd_test_" + std::to_string(::getpid()) + "_" +
                      std::to_string(m_test_counter++));
        std::filesystem::create_directories(m_temp_dir);
    }

    void TearDown() override
    {
        m_reader.reset();
        std::filesystem::remove_all(m_temp_dir);
    }

    static void add_process_scoped_track(
        const std::shared_ptr<metadata_registry>& metadata, const std::string& track_name)
    {
        metadata->add_track(track{ .track_name = track_name,
                                   .thread_id  = std::nullopt,
                                   .extdata    = std::string{} });
    }

    static void seed_gpu_smi_pmc_row(const std::shared_ptr<metadata_registry>& metadata,
                                     const char*                               pmc_name,
                                     const char* description = nullptr)
    {
        metadata->add_pmc_info(make_agent_pmc_row({ .type        = agent_type::gpu,
                                                    .name        = pmc_name,
                                                    .target_arch = "GPU",
                                                    .description = description }));
        add_process_scoped_track(metadata, pmc_name);
    }

    [[nodiscard]] size_t count_timeline_events(
        profiler_hub::reader_types::event_type_t type) const
    {
        profiler_hub::reader_types::event_filter_t filter;
        filter.types = { type };
        return m_reader->get_events(filter).size();
    }

    // reader_t does not yet surface PMC samples on the timeline (get_pmc_event_details is
    // unimplemented). After handle() pathways that emit PMC data, verify track metadata.
    void expect_reader_has_tracks(const std::vector<std::string>& track_names) const
    {
        const auto                      tracks = m_reader->get_all_tracks();
        std::unordered_set<std::string> names;
        for(const auto& track : tracks)
        {
            names.insert(track->name);
        }
        for(const auto& expected : track_names)
        {
            EXPECT_TRUE(names.count(expected))
                << "track not found in reader: " << expected;
        }
    }

    // reader_t queue_id/stream_id are SQLite row ids, not ROCm queue/stream handles.
    void expect_readback_queue_named(const char* expected_name) const
    {
        const auto& queues = m_reader->get_all_queues();
        const auto  queue_it =
            std::find_if(queues.begin(), queues.end(),
                         [&](const auto& queue) { return queue->name == expected_name; });
        ASSERT_NE(queue_it, queues.end()) << "queue not in reader: " << expected_name;
        ASSERT_NE((*queue_it)->process_info, nullptr);
        EXPECT_EQ((*queue_it)->process_info->pid, k_pid);
    }

    void expect_readback_stream_named(const char* expected_name) const
    {
        const auto& streams = m_reader->get_all_streams();
        const auto  stream_it =
            std::find_if(streams.begin(), streams.end(), [&](const auto& stream) {
                return stream->name == expected_name;
            });
        ASSERT_NE(stream_it, streams.end()) << "stream not in reader: " << expected_name;
        ASSERT_NE((*stream_it)->process_info, nullptr);
        EXPECT_EQ((*stream_it)->process_info->pid, k_pid);
    }

    [[nodiscard]] static std::filesystem::path find_rocpd_database_in_directory(
        const std::filesystem::path& directory)
    {
        if(!std::filesystem::exists(directory))
        {
            return {};
        }
        for(const auto& entry : std::filesystem::directory_iterator(directory))
        {
            const auto ext = entry.path().extension().string();
            if(ext == ".db" || ext == ".rocpd")
            {
                return entry.path();
            }
        }
        return {};
    }

    void expect_rocpd_database_path_valid() const
    {
        ASSERT_FALSE(m_db_path.empty())
            << "run_processor_and_open_reader must run before DB assertions";
        const std::filesystem::path path{ m_db_path };
        ASSERT_TRUE(std::filesystem::exists(path)) << m_db_path;
        ASSERT_TRUE(std::filesystem::is_regular_file(path)) << m_db_path;
        EXPECT_GT(std::filesystem::file_size(path), 0U) << m_db_path;
    }

    void expect_rocpd_reader_open() const
    {
        ASSERT_NE(m_reader, nullptr);
        EXPECT_FALSE(m_uuid.empty());
    }

    void expect_rocpd_database_on_disk() const
    {
        expect_rocpd_database_path_valid();
        expect_rocpd_reader_open();
    }

    void expect_readback_agent_fields(const agent_fields_expect& expected) const
    {
        const auto agents = m_reader->get_all_agents();
        const auto agent_it =
            std::find_if(agents.begin(), agents.end(), [&](const auto& agent_ptr) {
                return agent_ptr->agent_type == expected.agent_type;
            });
        ASSERT_NE(agent_it, agents.end())
            << "agent type not found in read-back: " << expected.agent_type;
        EXPECT_EQ((*agent_it)->name, expected.name);
        EXPECT_EQ((*agent_it)->model_name, expected.model_name);
        EXPECT_EQ((*agent_it)->vendor_name, expected.vendor_name);
        EXPECT_EQ((*agent_it)->product_name, expected.product_name);
    }

    [[nodiscard]] std::optional<profiler_hub::reader_types::timeline_event_t>
    find_first_event(profiler_hub::reader_types::event_type_t type) const
    {
        for(const auto& event : m_reader->get_events())
        {
            if(event.unique_identifier.type == type)
            {
                return event;
            }
        }
        return std::nullopt;
    }

    void expect_memory_alloc(
        const std::function<void(const profiler_hub::reader_types::memory_alloc_data_t&)>&
            check) const
    {
        const auto event = require_optional(
            find_first_event(profiler_hub::reader_types::event_type_t::memory_allocate),
            "memory_allocate event not found in read-back");
        check(require_optional(m_reader->get_memory_alloc_details(event),
                               "memory_allocate detail not readable"));
    }

    void expect_region(
        const std::function<void(const profiler_hub::reader_types::region_data_t&,
                                 const profiler_hub::reader_types::timeline_event_t&)>&
            check) const
    {
        const auto event = require_optional(
            find_first_event(profiler_hub::reader_types::event_type_t::region),
            "region event not found in read-back");
        check(require_optional(m_reader->get_region_details(event),
                               "region detail not readable"),
              event);
    }

    void expect_kernel_dispatch(
        const std::function<
            void(const profiler_hub::reader_types::kernel_dispatch_data_t&)>& check) const
    {
        const auto event = require_optional(
            find_first_event(profiler_hub::reader_types::event_type_t::kernel_dispatch),
            "kernel_dispatch event not found in read-back");
        check(require_optional(m_reader->get_kernel_dispatch_details(event),
                               "kernel dispatch detail not readable"));
    }

    void expect_memory_copy(
        const std::function<void(const profiler_hub::reader_types::memory_copy_data_t&)>&
            check) const
    {
        const auto event = require_optional(
            find_first_event(profiler_hub::reader_types::event_type_t::memory_copy),
            "memory_copy event not found in read-back");
        check(require_optional(m_reader->get_memory_copy_details(event),
                               "memory_copy detail not readable"));
    }

    // Drive the production path: agent_manager →
    // rocpd_processor_t::prepare_for_processing() (post_process_metadata +
    // make_agent_uid) → finalize_processing() → reader.
    [[nodiscard]] std::shared_ptr<metadata_registry> make_seeded_metadata(
        const std::function<void(const std::shared_ptr<metadata_registry>&)>&
            metadata_setup) const
    {
        auto metadata = std::make_shared<metadata_registry>();
        rocprofsys::trace_cache::info::process proc{};
        proc.pid     = static_cast<pid_t>(k_pid);
        proc.ppid    = static_cast<pid_t>(k_ppid);
        proc.command = "test_binary";
        proc.start   = k_process_start;
        proc.end     = k_process_end;
        metadata->set_process(proc);
        seed_test_thread(metadata);
        // rocpd_processor_t::handle(memory_copy_sample) sets trace env queue_id to 0.
        metadata->add_queue(0);
        if(metadata_setup)
        {
            metadata_setup(metadata);
        }
        return metadata;
    }

    void open_reader_for_written_db(const std::filesystem::path& out_dir)
    {
        const auto db_path = find_rocpd_database_in_directory(out_dir);
        ASSERT_FALSE(db_path.empty())
            << "rocpd_processor_t did not write a .db/.rocpd in " << out_dir.string();

        m_db_path = db_path.string();
        m_uuid    = extract_rocpd_uuid(db_path);
        ASSERT_FALSE(m_uuid.empty())
            << "failed to read profiler-hub uuid from " << m_db_path;

        auto read_storage = std::make_unique<profiler_hub::storage_t>(m_db_path, m_uuid);
        m_reader = std::make_unique<profiler_hub::reader_t>(std::move(read_storage));
        expect_rocpd_database_on_disk();
    }

    void run_processor_and_open_reader(
        std::vector<agent> agents,
        const std::function<void(const std::shared_ptr<metadata_registry>&)>&
                                                       metadata_setup = {},
        const std::function<void(rocpd_processor_t&)>& on_processor   = {})
    {
        const scoped_force_rocpd_metadata_registration force_rocpd;

        const auto prev_out = tim::settings::output_path();
        const auto out_dir  = m_temp_dir / "processor_out";
        std::filesystem::create_directories(out_dir);
        tim::settings::output_path() = out_dir.string();
        rocprofsys::reset_database_path_memo();

        struct restore_output_path
        {
            std::string prev_out;
            ~restore_output_path() { tim::settings::output_path() = prev_out; }
        } const restore_out{ prev_out };

        auto metadata = make_seeded_metadata(metadata_setup);
        auto mgr      = std::make_shared<agent_manager>();
        for(auto& agent_entry : agents)
        {
            mgr->insert_agent(agent_entry);
        }

        const int            pid  = static_cast<int>(k_pid);
        const int            ppid = static_cast<int>(k_ppid);
        output_file_registry registry;
        {
            rocpd_processor_t processor{ metadata, mgr, pid, ppid, registry };
            processor.prepare_for_processing();
            if(on_processor)
            {
                on_processor(processor);
            }
            processor.finalize_processing();
        }

        open_reader_for_written_db(out_dir);
    }

    static std::string extract_rocpd_uuid(const std::filesystem::path& db_path)
    {
        std::ifstream     db_file{ db_path, std::ios::binary };
        const std::string data{ std::istreambuf_iterator<char>{ db_file },
                                std::istreambuf_iterator<char>{} };
        const auto        key = std::string{ "rocpd_info_agent_" };
        const auto        pos = data.find(key);
        if(pos == std::string::npos)
        {
            return {};
        }
        const auto start = pos + key.size();
        auto       end   = start;
        while(end < data.size() && std::isxdigit(static_cast<unsigned char>(data[end])))
        {
            ++end;
        }
        return data.substr(start, end - start);
    }

    static void seed_test_thread(const std::shared_ptr<metadata_registry>& metadata)
    {
        rocprofsys::trace_cache::info::thread thread_info{};
        thread_info.parent_process_id = static_cast<std::int32_t>(k_ppid);
        thread_info.process_id        = static_cast<std::int32_t>(k_pid);
        thread_info.thread_id         = k_thread_id;
        thread_info.start             = k_process_start;
        thread_info.end               = k_process_end;
        metadata->add_thread_info(thread_info);
    }

    static void seed_nic_pmc_catalog(const std::shared_ptr<metadata_registry>& metadata)
    {
        for(const auto& spec : k_nic_pmcs)
        {
            metadata->add_pmc_info(make_nic_metadata_pmc(spec));
        }
    }

    static constexpr std::uint64_t k_managed_gpu_handle = 1;
    static constexpr std::uint64_t k_managed_cpu_handle = 2;

    static agent managed_gpu_agent()
    {
        auto result   = gpu_agent();
        result.handle = k_managed_gpu_handle;
        return result;
    }

    static agent managed_cpu_agent()
    {
        auto result   = cpu_agent();
        result.handle = k_managed_cpu_handle;
        return result;
    }

    static pmc make_agent_pmc_row(const agent_pmc_spec& spec)
    {
        pmc row{};
        row.type             = spec.type;
        row.agent_type_index = spec.agent_type_index;
        row.target_arch      = spec.target_arch;
        row.name             = spec.name;
        row.symbol           = spec.name;
        row.description      = spec.description != nullptr ? spec.description : spec.name;
        row.units            = "";
        row.value_type       = "ABS";
        row.extdata          = "{}";
        return row;
    }

    static void seed_gpu_queue_stream(const std::shared_ptr<metadata_registry>& metadata)
    {
        metadata->add_queue(k_queue_id);
        metadata->add_stream(k_stream_id);
    }

    static void seed_code_object(const std::shared_ptr<metadata_registry>& metadata,
                                 std::uint64_t code_object_id, std::uint64_t agent_handle)
    {
        static const std::string k_uri = "file:///test_code_object.co";
        rocprofiler_callback_tracing_code_object_load_data_t code_object{};
        code_object.code_object_id = code_object_id;
        code_object.uri            = k_uri.c_str();
#if(ROCPROFILER_VERSION >= 600)
        code_object.agent_id.handle = agent_handle;
#else
        code_object.rocp_agent.handle = agent_handle;
#endif
        code_object.storage_type = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
        metadata->add_code_object(code_object);
    }

    static void seed_kernel_symbol(const std::shared_ptr<metadata_registry>& metadata,
                                   std::uint64_t kernel_id, const char* kernel_name,
                                   std::uint64_t agent_handle = k_managed_gpu_handle)
    {
        static std::unordered_map<std::uint64_t, std::string> s_stored_names;
        s_stored_names[kernel_id] = kernel_name;
        seed_code_object(metadata, 1, agent_handle);
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
            kernel_symbol{};
        kernel_symbol.kernel_id      = kernel_id;
        kernel_symbol.code_object_id = 1;
        kernel_symbol.kernel_name    = s_stored_names[kernel_id].c_str();
        metadata->add_kernel_symbol(kernel_symbol);
    }

    static agent gpu_agent()
    {
        agent result{};
        result.type              = agent_type::gpu;
        result.device_type_index = 0;
        result.name              = "gfx90a";
        result.model_name        = "MI210";
        result.vendor_name       = "AMD";
        result.product_name      = "Instinct MI210";
        return result;
    }

    static agent cpu_agent()
    {
        agent result{};
        result.type              = agent_type::cpu;
        result.device_type_index = 0;
        result.name              = "CPU0";
        result.model_name        = "EPYC";
        result.vendor_name       = "AMD";
        result.product_name      = "EPYC 7763";
        return result;
    }

    static agent nic_agent()
    {
        agent result{};
        result.type              = agent_type::nic;
        result.device_type_index = 0;
        result.name              = "NIC0";
        result.model_name        = "CX7";
        result.vendor_name       = "AI NIC";
        result.product_name      = "AI NIC";
        return result;
    }

    std::filesystem::path                   m_temp_dir;
    std::string                             m_db_path;
    std::string                             m_uuid;
    std::unique_ptr<profiler_hub::reader_t> m_reader;

    static int m_test_counter;
};

int rocpd_write_read_test_interface::m_test_counter = 0;

TEST_F(rocpd_write_read_test_interface, agents_round_trip_all_types)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({ gpu_agent(), cpu_agent(), nic_agent() });
    // Validate: profiler_hub::reader_t read-back matches inserted values.
    ASSERT_EQ(m_reader->get_all_agents().size(), 3U);
    expect_readback_agent_fields({ .agent_type   = "GPU",
                                   .name         = "gfx90a",
                                   .model_name   = "MI210",
                                   .vendor_name  = "AMD",
                                   .product_name = "Instinct MI210" });
    expect_readback_agent_fields({ .agent_type   = "CPU",
                                   .name         = "CPU0",
                                   .model_name   = "EPYC",
                                   .vendor_name  = "AMD",
                                   .product_name = "EPYC 7763" });
    expect_readback_agent_fields({ .agent_type   = "NIC",
                                   .name         = "NIC0",
                                   .model_name   = "CX7",
                                   .vendor_name  = "AI NIC",
                                   .product_name = "AI NIC" });
}

TEST_F(rocpd_write_read_test_interface, handle_ainic_pmc_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr double k_rx_ucast_bytes   = 1048576.0;
    static constexpr double k_tx_ucast_bytes   = 524288.0;
    static constexpr size_t k_sample_timestamp = 12000;

    run_processor_and_open_reader(
        { nic_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(
                make_nic_metadata_pmc(k_nic_pmcs[k_nic_pmc_rx_ucast_bytes_idx]));
            metadata->add_pmc_info(
                make_nic_metadata_pmc(k_nic_pmcs[k_nic_pmc_tx_ucast_bytes_idx]));
            add_process_scoped_track(metadata, "ainic_rx_rdma_ucast_bytes");
            add_process_scoped_track(metadata, "ainic_tx_rdma_ucast_bytes");
        },
        [](rocpd_processor_t& processor) {
            rocprofsys::pmc::collectors::nic::enabled_metrics enabled{};
            enabled.bits.rx_rdma_ucast_bytes = 1;
            enabled.bits.tx_rdma_ucast_bytes = 1;
            rocprofsys::pmc::collectors::nic::metrics metrics{};
            metrics.rx_rdma_ucast_bytes = static_cast<std::uint64_t>(k_rx_ucast_bytes);
            metrics.tx_rdma_ucast_bytes = static_cast<std::uint64_t>(k_tx_ucast_bytes);
            const ainic_pmc_sample sample{ enabled, 0, "NIC0", k_sample_timestamp,
                                           metrics };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    ASSERT_EQ(m_reader->get_all_agents().size(), 1U);
    expect_readback_agent_fields({ .agent_type   = "NIC",
                                   .name         = "NIC0",
                                   .model_name   = "CX7",
                                   .vendor_name  = "AI NIC",
                                   .product_name = "AI NIC" });

    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 2U);
    expect_named_pmc_arch(
        pmc_infos, { .name                = "nic_rx_ucast_bytes",
                     .expected_arch       = "NIC",
                     .expected_agent_type = "NIC",
                     .expected_symbol     = "nic_rx_ucast_bytes",
                     .expected_units = k_nic_pmcs[k_nic_pmc_rx_ucast_bytes_idx].units });
    expect_named_pmc_arch(
        pmc_infos, { .name                = "nic_tx_ucast_bytes",
                     .expected_arch       = "NIC",
                     .expected_agent_type = "NIC",
                     .expected_symbol     = "nic_tx_ucast_bytes",
                     .expected_units = k_nic_pmcs[k_nic_pmc_tx_ucast_bytes_idx].units });

    expect_reader_has_tracks(
        { "ainic_rx_rdma_ucast_bytes", "ainic_tx_rdma_ucast_bytes" });
}

// Mirrors tests/rocpd-validation-rules/ainic/ainic-rdma-rules.json: every
// cache_policy.hpp NIC PMC name is registered with target_arch NIC.
TEST_F(rocpd_write_read_test_interface, nic_rdma_pmc_catalog_target_arch)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({ nic_agent() },
                                  [](const std::shared_ptr<metadata_registry>& metadata) {
                                      seed_nic_pmc_catalog(metadata);
                                  });
    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_nic_pmc_catalog(m_reader->get_all_pmc_info());
}

TEST_F(rocpd_write_read_test_interface, nic_pmc_info_invalid_target_arch_rejected)
{
    const scoped_force_rocpd_metadata_registration force_rocpd;

    const auto prev_out = tim::settings::output_path();
    const auto out_dir  = m_temp_dir / "processor_invalid_arch";
    std::filesystem::create_directories(out_dir);
    tim::settings::output_path() = out_dir.string();
    rocprofsys::reset_database_path_memo();
    struct restore_out_path
    {
        std::string prev;
        ~restore_out_path() { tim::settings::output_path() = prev; }
    } const restore_out{ prev_out };

    auto metadata = std::make_shared<metadata_registry>();
    rocprofsys::trace_cache::info::process proc{};
    proc.pid     = static_cast<pid_t>(k_pid);
    proc.ppid    = static_cast<pid_t>(k_ppid);
    proc.command = "test_binary";
    proc.start   = k_process_start;
    proc.end     = k_process_end;
    metadata->set_process(proc);

    auto bad_pmc        = make_nic_metadata_pmc(k_nic_pmcs[0]);
    bad_pmc.target_arch = "AINIC";
    metadata->add_pmc_info(bad_pmc);

    auto  mgr = std::make_shared<agent_manager>();
    agent nic = nic_agent();
    mgr->insert_agent(nic);

    output_file_registry registry;
    rocpd_processor_t    processor{ metadata, mgr, static_cast<int>(k_pid),
                                 static_cast<int>(k_ppid), registry };

    // Validate: prepare_for_processing rejects invalid target_arch (no rocpd DB).
    EXPECT_THROW(processor.prepare_for_processing(), std::invalid_argument);
    EXPECT_TRUE(find_rocpd_database_in_directory(out_dir).empty())
        << "invalid PMC metadata must not create a rocpd database";
}

TEST_F(rocpd_write_read_test_interface, gpu_and_nic_pmc_keep_distinct_target_arch)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent(), nic_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(make_agent_pmc_row(
                { .type = agent_type::gpu, .name = "gfx_busy", .target_arch = "GPU" }));
            metadata->add_pmc_info(
                make_nic_metadata_pmc(k_nic_pmcs[k_nic_pmc_rx_ucast_bytes_idx]));
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 2U);
    expect_named_pmc_arch(pmc_infos, { .name                = "gfx_busy",
                                       .expected_arch       = "GPU",
                                       .expected_agent_type = "GPU",
                                       .expected_symbol     = "gfx_busy" });
    expect_named_pmc_arch(
        pmc_infos, { .name                = "nic_rx_ucast_bytes",
                     .expected_arch       = "NIC",
                     .expected_agent_type = "NIC",
                     .expected_symbol     = "nic_rx_ucast_bytes",
                     .expected_units = k_nic_pmcs[k_nic_pmc_rx_ucast_bytes_idx].units });
}

// ---------------------------------------------------------------------------
// Kernel dispatch: write, read back, validate timestamps + grid sizes
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, kernel_dispatch_values_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts         = 5000;
    static constexpr std::uint64_t k_end_ts           = 6000;
    static constexpr std::uint64_t k_dispatch_id      = 42;
    static constexpr std::uint64_t k_workgroup_size_x = 256;
    static constexpr std::uint64_t k_grid_size_x      = 1024;

    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "my_test_kernel");
        },
        [](rocpd_processor_t& processor) {
            const kernel_dispatch_sample kds{ k_start_ts,
                                              k_end_ts,
                                              k_thread_id,
                                              k_managed_gpu_handle,
                                              1,
                                              k_dispatch_id,
                                              k_queue_id,
                                              1,
                                              0,
                                              0,
                                              0,
                                              k_workgroup_size_x,
                                              1,
                                              1,
                                              k_grid_size_x,
                                              1,
                                              1,
                                              k_stream_id };
            processor.handle(kds);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_kernel_dispatch([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.dispatch_id, k_dispatch_id);
        EXPECT_EQ(detail.workgroup_size_x, k_workgroup_size_x);
        EXPECT_EQ(detail.workgroup_size_y, 1U);
        EXPECT_EQ(detail.workgroup_size_z, 1U);
        EXPECT_EQ(detail.grid_size_x, k_grid_size_x);
        EXPECT_EQ(detail.grid_size_y, 1U);
        EXPECT_EQ(detail.grid_size_z, 1U);
        EXPECT_EQ(detail.name, "my_test_kernel");
    });

    expect_readback_queue_named("Queue 10");
    expect_readback_stream_named("Stream 20");
    EXPECT_EQ(
        count_timeline_events(profiler_hub::reader_types::event_type_t::kernel_dispatch),
        1U);
}

// ---------------------------------------------------------------------------
// Region with args: write, read back, validate name + argument values
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, region_with_args_values_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts = 5000;
    static constexpr std::uint64_t k_end_ts   = 5200;
    const auto                     region_args =
        get_args_string(function_args_t{ { .arg_number = 0U,
                                           .arg_type   = "void*",
                                           .arg_name   = "dst",
                                           .arg_value  = "0x7f0000000000" },
                                         { .arg_number = 1U,
                                           .arg_type   = "size_t",
                                           .arg_name   = "sizeBytes",
                                           .arg_value  = "4096" } });
    run_processor_and_open_reader({}, {}, [&](rocpd_processor_t& processor) {
        const region_sample reg{ k_thread_id, "hipMemcpy", 1, 0, k_start_ts, k_end_ts, "",
                                 region_args, "HIP_API" };
        processor.handle(reg);
    });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([this](const auto& detail, const auto& tl_event) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.name, "hipMemcpy");
        ASSERT_NE(detail.event, nullptr);
        EXPECT_EQ(detail.event->event_category, "HIP_API");

        const auto args = m_reader->get_arguments(tl_event);
        ASSERT_EQ(args.size(), 2U);
        EXPECT_EQ(args[0]->position, 0U);
        EXPECT_EQ(args[0]->type, "void*");
        EXPECT_EQ(args[0]->name, "dst");
        EXPECT_EQ(args[0]->value, "0x7f0000000000");
        EXPECT_EQ(args[1]->position, 1U);
        EXPECT_EQ(args[1]->type, "size_t");
        EXPECT_EQ(args[1]->name, "sizeBytes");
        EXPECT_EQ(args[1]->value, "4096");
    });
}

// ---------------------------------------------------------------------------
// Memory copy: write, read back, validate size + agent ids
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, memory_copy_values_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts    = 6500;
    static constexpr std::uint64_t k_end_ts      = 7000;
    static constexpr std::uint64_t k_copy_size   = 4096;
    static constexpr std::uint64_t k_dst_address = 0x100000;
    static constexpr std::uint64_t k_src_address = 0x7F0000000000;

    run_processor_and_open_reader(
        { managed_gpu_agent(), managed_cpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_stream(k_stream_id);
        },
        [](rocpd_processor_t& processor) {
            const memory_copy_sample mcs{
                k_start_ts,
                k_end_ts,
                k_thread_id,
                k_managed_gpu_handle,
                k_managed_cpu_handle,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY),
                static_cast<std::int32_t>(ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE),
                k_copy_size,
                1,
                0,
                k_dst_address,
                k_src_address,
                k_stream_id
            };
            processor.handle(mcs);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_copy([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.size, k_copy_size);
        EXPECT_EQ(detail.name, "MEMORY_COPY_HOST_TO_DEVICE");
        EXPECT_EQ(require_optional(detail.dst_address, "dst_address missing"),
                  k_dst_address);
        EXPECT_EQ(require_optional(detail.src_address, "src_address missing"),
                  k_src_address);
        ASSERT_NE(detail.dst_agent_id, nullptr);
        EXPECT_EQ(detail.dst_agent_id->agent_type, "GPU");
        ASSERT_NE(detail.src_agent_id, nullptr);
        EXPECT_EQ(detail.src_agent_id->agent_type, "CPU");
    });
    expect_readback_stream_named("Stream 20");
}

// ---------------------------------------------------------------------------
// Multiple regions + kernel dispatch: verify event fields (not just counts)
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, event_counts_match_inserted_data)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_region_base_start = 1000;
    static constexpr std::uint64_t k_region_base_end   = 1050;
    static constexpr std::uint64_t k_region_stride     = 100;
    static constexpr std::uint64_t k_kd_start_ts       = 5000;
    static constexpr std::uint64_t k_kd_end_ts         = 6000;
    static constexpr std::uint64_t k_kd_queue_handle   = 10;
    static constexpr std::uint64_t k_workgroup_size_x  = 64;
    static constexpr std::uint64_t k_grid_size_x       = 256;

    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "count_test_kernel");
        },
        [](rocpd_processor_t& processor) {
            for(int idx = 0; idx < 2; ++idx)
            {
                const auto          name = "region_" + std::to_string(idx);
                const region_sample reg{
                    k_thread_id,
                    name,
                    static_cast<std::uint64_t>(idx),
                    0,
                    k_region_base_start +
                        static_cast<std::uint64_t>(idx) * k_region_stride,
                    k_region_base_end + static_cast<std::uint64_t>(idx) * k_region_stride,
                    "",
                    "",
                    "HIP_API"
                };
                processor.handle(reg);
            }

            const kernel_dispatch_sample kds{ k_kd_start_ts,
                                              k_kd_end_ts,
                                              k_thread_id,
                                              k_managed_gpu_handle,
                                              1,
                                              1,
                                              k_queue_id,
                                              k_kd_queue_handle,
                                              0,
                                              0,
                                              0,
                                              k_workgroup_size_x,
                                              1,
                                              1,
                                              k_grid_size_x,
                                              1,
                                              1,
                                              k_stream_id };
            processor.handle(kds);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    auto counts = m_reader->get_event_counts();

    auto region_it = counts.find(profiler_hub::reader_types::event_type_t::region);
    ASSERT_NE(region_it, counts.end());
    EXPECT_EQ(region_it->second, 2U);

    auto kd_it = counts.find(profiler_hub::reader_types::event_type_t::kernel_dispatch);
    ASSERT_NE(kd_it, counts.end());
    EXPECT_EQ(kd_it->second, 1U);

    std::unordered_map<std::string, std::pair<size_t, size_t>> region_times;
    for(const auto& tl_event : m_reader->get_events())
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::region)
        {
            continue;
        }
        const auto detail = require_optional(m_reader->get_region_details(tl_event),
                                             "region detail not readable");
        region_times[detail.name] = { detail.start_timestamp, detail.end_timestamp };
    }
    ASSERT_EQ(region_times.size(), 2U);
    EXPECT_EQ(region_times["region_0"].first, k_region_base_start);
    EXPECT_EQ(region_times["region_0"].second, k_region_base_end);
    EXPECT_EQ(region_times["region_1"].first, k_region_base_start + k_region_stride);
    EXPECT_EQ(region_times["region_1"].second, k_region_base_end + k_region_stride);

    for(const auto& tl_event : m_reader->get_events())
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::kernel_dispatch)
        {
            continue;
        }
        const auto detail =
            require_optional(m_reader->get_kernel_dispatch_details(tl_event),
                             "kernel dispatch detail not readable");
        EXPECT_EQ(detail.name, "count_test_kernel");
        EXPECT_EQ(detail.start_timestamp, k_kd_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_kd_end_ts);
        EXPECT_EQ(detail.workgroup_size_x, k_workgroup_size_x);
        EXPECT_EQ(detail.grid_size_x, k_grid_size_x);
    }
}

// ---------------------------------------------------------------------------
// Metadata round-trip: node, process, thread, queue, stream
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, metadata_round_trip)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({},
                                  [](const std::shared_ptr<metadata_registry>& metadata) {
                                      seed_test_thread(metadata);
                                      seed_gpu_queue_stream(metadata);
                                  });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto& host_node = rocprofsys::node_info::get_instance();
    const auto  nodes     = m_reader->get_all_nodes();
    ASSERT_GE(nodes.size(), 1U);
    EXPECT_EQ(nodes[0]->system_name, host_node.system_name);
    EXPECT_EQ(nodes[0]->hostname, host_node.node_name);

    auto processes = m_reader->get_all_processes();
    ASSERT_EQ(processes.size(), 1U);
    EXPECT_EQ(processes[0]->pid, k_pid);
    EXPECT_EQ(processes[0]->command, "test_binary");

    auto threads = m_reader->get_all_threads();
    ASSERT_EQ(threads.size(), 1U);
    EXPECT_EQ(threads[0]->thread_id, k_thread_id);
    EXPECT_EQ(threads[0]->name, "Thread 300");

    ASSERT_EQ(m_reader->get_all_queues().size(), 2U);
    expect_readback_queue_named("Queue 0");
    expect_readback_queue_named("Queue 10");

    ASSERT_EQ(m_reader->get_all_streams().size(), 1U);
    expect_readback_stream_named("Stream 20");

    EXPECT_EQ(processes[0]->ppid, k_ppid);
    EXPECT_EQ(processes[0]->start, k_process_start);
    EXPECT_EQ(processes[0]->end, k_process_end);
    EXPECT_EQ(threads[0]->start, k_process_start);
    EXPECT_EQ(threads[0]->end, k_process_end);
}

// ---------------------------------------------------------------------------
// Output file existence and non-empty after flush
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, flush_creates_nonempty_file)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts = 1000;
    static constexpr std::uint64_t k_end_ts   = 2000;

    run_processor_and_open_reader({}, {}, [](rocpd_processor_t& processor) {
        const region_sample reg{ k_thread_id, "flush_test", 0,  0,        k_start_ts,
                                 k_end_ts,    "",           "", "HIP_API" };
        processor.handle(reg);
    });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    ASSERT_EQ(count_timeline_events(profiler_hub::reader_types::event_type_t::region),
              1U);
    const auto events = m_reader->get_events();
    ASSERT_EQ(events.size(), 1U);
    EXPECT_EQ(events[0].display_name, "flush_test");
    EXPECT_EQ(events[0].start_timestamp, k_start_ts);
    EXPECT_EQ(events[0].end_timestamp, k_end_ts);
}

// ═══════════════════════════════════════════════════════════════════════════
// rocpd_processor_t::handle() — one TEST_F per overload in rocpd_processor.cpp.
// Each calls run_processor_and_open_reader (prepare → handle → finalize), which
// requires a nonempty .db/.rocpd on disk and a readable profiler_hub::reader_t.
//
//  kernel_dispatch_sample   → kernel_dispatch_values_persisted,
//                             handle_kernel_dispatch_full_grid
//  scratch_memory_sample    → handle_scratch_memory_pathway
//  memory_copy_sample       → memory_copy_values_persisted,
//                             handle_memory_copy_addresses_persisted
//  memory_allocate_sample   → handle_memory_allocate_pathway (ROCPROFILER >= 600)
//  region_sample            → region_with_args_values_persisted,
//                             handle_region_with_call_stack_pathway,
//                             flush_creates_nonempty_file
//  backtrace_region_sample  → handle_backtrace_region_pathway
//  in_time_sample           → handle_in_time_sample_pathway
//  pmc_event_with_sample    → handle_pmc_event_with_sample_pathway
//  gpu_pmc_sample           → handle_gpu_pmc_sample_pathway
//  gpu_perf_counter_sample  → handle_gpu_perf_counter_sample_pathway,
//                             handle_gpu_perf_counter_sample_empty_entries_noop
//  ainic_pmc_sample         → handle_ainic_pmc_sample_pathway
//  cpu_pmc_sample           → handle_cpu_pmc_sample_pathway
//  kfd_sample               → handle_kfd_sample_pathway
//
//  multiple_event_types_in_single_db — several handle() paths in one DB flush.
//  agents_round_trip / metadata_round_trip / nic_rdma_pmc_catalog — metadata-only
//  finalize (no handle) still writes a rocpd database.
// ═══════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// handle(scratch_memory_sample): memory_alloc with flags extdata
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_scratch_memory_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts     = 3000;
    static constexpr std::uint64_t k_end_ts       = 3100;
    static constexpr std::uint64_t k_alloc_size   = 131072;
    static constexpr std::uint64_t k_flags        = 100;
    static constexpr std::uint64_t k_queue_handle = 50;

    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
        },
        [](rocpd_processor_t& processor) {
            const scratch_memory_sample sms{
                k_start_ts,
                k_end_ts,
                k_thread_id,
                k_managed_gpu_handle,
                k_queue_id,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY),
                static_cast<std::int32_t>(ROCPROFILER_SCRATCH_MEMORY_ALLOC),
                0,
                k_alloc_size,
                k_flags,
                k_queue_handle,
                k_stream_id
            };
            processor.handle(sms);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_alloc([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.type, "ALLOC");
        EXPECT_EQ(detail.level, "SCRATCH");
        EXPECT_EQ(detail.size, k_alloc_size);
    });
    expect_readback_agent_fields({ .agent_type   = "GPU",
                                   .name         = "gfx90a",
                                   .model_name   = "MI210",
                                   .vendor_name  = "AMD",
                                   .product_name = "Instinct MI210" });
}

// ---------------------------------------------------------------------------
// handle(memory_allocate_sample): memory_alloc with agent + address
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_memory_allocate_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
#if(ROCPROFILER_VERSION < 600)
    GTEST_SKIP() << "memory_allocate_sample requires ROCPROFILER_VERSION >= 600";
#else
    static constexpr std::uint64_t k_start_ts   = 7000;
    static constexpr std::uint64_t k_end_ts     = 7200;
    static constexpr std::uint64_t k_alloc_size = 8192;
    static constexpr std::uint64_t k_corr_id    = 200;
    static constexpr std::uint64_t k_ancestor   = 100;
    static constexpr std::uint64_t k_address    = 0x7F0000100000;

    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_stream(k_stream_id);
        },
        [](rocpd_processor_t& processor) {
            const memory_allocate_sample mas{
                k_start_ts,
                k_end_ts,
                k_thread_id,
                k_managed_gpu_handle,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION),
                static_cast<std::int32_t>(ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE),
                k_alloc_size,
                k_corr_id,
                k_ancestor,
                k_address,
                k_stream_id
            };
            processor.handle(mas);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_alloc([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.type, "ALLOC");
        EXPECT_EQ(detail.level, "REAL");
        EXPECT_EQ(detail.size, k_alloc_size);
        EXPECT_EQ(require_optional(detail.address, "address missing"), k_address);
    });
    expect_readback_agent_fields({ .agent_type   = "GPU",
                                   .name         = "gfx90a",
                                   .model_name   = "MI210",
                                   .vendor_name  = "AMD",
                                   .product_name = "Instinct MI210" });
#endif
}

// ---------------------------------------------------------------------------
// handle(backtrace_region_sample): region with track_name and extdata
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_backtrace_region_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts = 8000;
    static constexpr std::uint64_t k_end_ts   = 8500;

    run_processor_and_open_reader(
        {},
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_track(track{ .track_name = "Sampling [CPU 0]",
                                       .thread_id  = k_thread_id,
                                       .extdata    = std::string{} });
        },
        [](rocpd_processor_t& processor) {
            const backtrace_region_sample bts{
                0,          k_thread_id, "Sampling [CPU 0]", "backtrace_sample_func",
                k_start_ts, k_end_ts,    "sampling",         R"({"backtrace": true})",
                "",         ""
            };
            processor.handle(bts);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([](const auto& detail, const auto&) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.name, "backtrace_sample_func");
        ASSERT_NE(detail.event, nullptr);
        EXPECT_EQ(detail.event->event_category, "sampling");
    });
    expect_reader_has_tracks({ "Sampling [CPU 0]" });
}

// ---------------------------------------------------------------------------
// handle(in_time_sample): pmc_event with track + sample timestamp
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_in_time_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(make_agent_pmc_row({ .type        = agent_type::gpu,
                                                        .name        = "my_track",
                                                        .target_arch = "GPU",
                                                        .description = "IN_TIME" }));
            add_process_scoped_track(metadata, "my_track");
        },
        [](rocpd_processor_t& processor) {
            constexpr std::uint64_t k_timestamp = 9500;
            constexpr std::uint64_t k_event_id  = 5;
            const in_time_sample    its{
                0, "my_track", k_timestamp, R"({"metadata": "test"})", k_event_id, 3,
                0, "",         ""
            };
            processor.handle(its);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    expect_named_pmc_arch(pmc_infos, { .name                 = "my_track",
                                       .expected_arch        = "GPU",
                                       .expected_agent_type  = "GPU",
                                       .expected_symbol      = "my_track",
                                       .expected_units       = "",
                                       .expected_description = "IN_TIME" });

    expect_reader_has_tracks({ "my_track" });
}

// ---------------------------------------------------------------------------
// handle(pmc_event_with_sample): pmc_event with agent + value + tid
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_pmc_event_with_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(
                make_agent_pmc_row({ .type        = agent_type::gpu,
                                     .name        = "SQ_WAVES",
                                     .target_arch = "GPU",
                                     .description = "Shader wavefronts" }));
            metadata->add_track(track{ .track_name = "SQ_WAVES [GPU 0]",
                                       .thread_id  = k_thread_id,
                                       .extdata    = std::string{} });
        },
        [](rocpd_processor_t& processor) {
            constexpr std::uint64_t     k_timestamp = 10000;
            constexpr std::uint64_t     k_event_id  = 10;
            constexpr std::uint64_t     k_stack_id  = 5;
            constexpr std::uint64_t     k_parent_id = 42;
            constexpr double            k_pmc_value = 1024.0;
            const pmc_event_with_sample pmc{ 0,
                                             "SQ_WAVES [GPU 0]",
                                             k_timestamp,
                                             "{}",
                                             k_event_id,
                                             k_stack_id,
                                             k_parent_id,
                                             "",
                                             "",
                                             0,
                                             static_cast<std::uint8_t>(agent_type::gpu),
                                             "SQ_WAVES",
                                             k_pmc_value,
                                             static_cast<std::int64_t>(k_thread_id) };
            processor.handle(pmc);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    expect_named_pmc_arch(pmc_infos, { .name                 = "SQ_WAVES",
                                       .expected_arch        = "GPU",
                                       .expected_agent_type  = "GPU",
                                       .expected_symbol      = "SQ_WAVES",
                                       .expected_units       = "",
                                       .expected_description = "Shader wavefronts" });

    expect_reader_has_tracks({ "SQ_WAVES [GPU 0]" });
}

// ---------------------------------------------------------------------------
// handle(gpu_pmc_sample): multiple scalar PMC inserts for GPU metrics
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_gpu_pmc_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_smi_pmc_row(metadata, "device_busy_gfx");
            seed_gpu_smi_pmc_row(metadata, "device_busy_umc");
            seed_gpu_smi_pmc_row(metadata, "device_temp");
        },
        [](rocpd_processor_t& processor) {
            constexpr double        k_gfx_activity        = 85.0;
            constexpr double        k_umc_activity        = 42.0;
            constexpr double        k_hotspot_temperature = 72.0;
            constexpr std::uint64_t k_timestamp           = 11000;
            rocprofsys::pmc::collectors::gpu::enabled_metrics enabled{};
            enabled.bits.gfx_activity        = 1;
            enabled.bits.umc_activity        = 1;
            enabled.bits.hotspot_temperature = 1;
            rocprofsys::pmc::collectors::gpu::metrics metrics{};
            metrics.gfx_activity        = k_gfx_activity;
            metrics.umc_activity        = k_umc_activity;
            metrics.hotspot_temperature = k_hotspot_temperature;
            const gpu_pmc_sample sample{ enabled, 0, k_timestamp, metrics };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 3U);
    expect_named_pmc_arch(pmc_infos, { .name                = "device_busy_gfx",
                                       .expected_arch       = "GPU",
                                       .expected_agent_type = "GPU",
                                       .expected_symbol     = "device_busy_gfx" });
    expect_named_pmc_arch(pmc_infos, { .name                = "device_busy_umc",
                                       .expected_arch       = "GPU",
                                       .expected_agent_type = "GPU",
                                       .expected_symbol     = "device_busy_umc" });
    expect_named_pmc_arch(pmc_infos, { .name                = "device_temp",
                                       .expected_arch       = "GPU",
                                       .expected_agent_type = "GPU",
                                       .expected_symbol     = "device_temp" });

    expect_reader_has_tracks({ "device_busy_gfx", "device_busy_umc", "device_temp" });
}

// ---------------------------------------------------------------------------
// handle(gpu_perf_counter_sample): SDK hardware counter batch → PMC events
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_gpu_perf_counter_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_counter_id = 42;
    static const std::string       k_pmc_name{ "SQ_WAVES[WGP=0,SA=0]" };
    static const std::string       k_track_name{ "GPU [0] SQ_WAVES (S)" };
    static constexpr std::uint64_t k_timestamp = 11500;
    static constexpr double        k_value     = 2048.0;

    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(
                make_agent_pmc_row({ .type        = agent_type::gpu,
                                     .name        = k_pmc_name.c_str(),
                                     .target_arch = "GPU",
                                     .description = "Shader wavefronts" }));
            // gpu_perf_counter handle() emits process-scoped tracks (no thread_id).
            metadata->add_track(track{ .track_name = k_track_name,
                                       .thread_id  = std::nullopt,
                                       .extdata    = std::string{} });
            gpu_perf_counter_name_entry name_entry{};
            name_entry.counter_id    = k_counter_id;
            name_entry.pmc_info_name = k_pmc_name;
            name_entry.track_name    = k_track_name;
            metadata->set_gpu_perf_counter_counter_names(0, { name_entry });
        },
        [](rocpd_processor_t& processor) {
            const gpu_perf_counter_sample sample{ 0, k_timestamp,
                                                  std::vector<gpu_perf_counter_value>{
                                                      { .counter_id = k_counter_id,
                                                        .value      = k_value } } };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    expect_named_pmc_arch(pmc_infos, { .name                 = k_pmc_name.c_str(),
                                       .expected_arch        = "GPU",
                                       .expected_agent_type  = "GPU",
                                       .expected_symbol      = k_pmc_name.c_str(),
                                       .expected_units       = "",
                                       .expected_description = "Shader wavefronts" });

    expect_reader_has_tracks({ k_track_name });
}

TEST_F(rocpd_write_read_test_interface, handle_gpu_perf_counter_sample_empty_entries_noop)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() }, {}, [](rocpd_processor_t& processor) {
            constexpr std::uint64_t       k_timestamp = 12000;
            const gpu_perf_counter_sample sample{ 0, k_timestamp, {} };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    auto pmc_infos = m_reader->get_all_pmc_info();
    EXPECT_TRUE(pmc_infos.empty());
}

// ---------------------------------------------------------------------------
// handle(cpu_pmc_sample): process-level + per-core PMC inserts
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_cpu_pmc_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    const auto cpu = managed_cpu_agent();
    run_processor_and_open_reader(
        { cpu },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(make_agent_pmc_row({ .type = agent_type::cpu,
                                                        .name = "process_physical_memory",
                                                        .target_arch = "CPU" }));
            metadata->add_pmc_info(make_agent_pmc_row({ .type        = agent_type::cpu,
                                                        .name        = "cpu_frequency",
                                                        .target_arch = "CPU" }));
            add_process_scoped_track(metadata, "process_physical_memory");
            add_process_scoped_track(metadata, "cpu_frequency [0] Core [0]");
            add_process_scoped_track(metadata, "cpu_frequency [0] Core [1]");
        },
        [](rocpd_processor_t& processor) {
            constexpr double        k_page_rss_mib   = 256.5;
            constexpr double        k_bytes_per_mib  = 1024.0 * 1024.0;
            constexpr float         k_core0_freq_mhz = 3200.0f;
            constexpr float         k_core1_freq_mhz = 3100.0f;
            constexpr std::uint64_t k_timestamp      = 13000;
            rocprofsys::pmc::collectors::cpu::enabled_metrics enabled{};
            enabled.bits.page_rss  = 1;
            enabled.bits.frequency = 1;
            rocprofsys::pmc::collectors::cpu::process_metrics proc{};
            proc.page_rss = static_cast<std::int64_t>(k_page_rss_mib * k_bytes_per_mib);
            const std::vector<rocprofsys::pmc::collectors::cpu::per_cpu_metrics> cores{
                { .cpu_id = 0, .frequency = k_core0_freq_mhz, .load = 0.0 },
                { .cpu_id = 1, .frequency = k_core1_freq_mhz, .load = 0.0 },
            };
            auto freqs = rocprofsys::pmc::collectors::cpu::serialize_frequencies(cores);
            const cpu_pmc_sample sample{ enabled,          0, k_timestamp, proc,
                                         std::move(freqs), {} };
            processor.handle(sample);
        });

    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 2U);
    expect_named_pmc_arch(pmc_infos, { .name                = "process_physical_memory",
                                       .expected_arch       = "CPU",
                                       .expected_agent_type = "CPU",
                                       .expected_symbol = "process_physical_memory" });
    expect_named_pmc_arch(pmc_infos, { .name                = "cpu_frequency",
                                       .expected_arch       = "CPU",
                                       .expected_agent_type = "CPU",
                                       .expected_symbol     = "cpu_frequency" });

    expect_readback_agent_fields({ .agent_type   = "CPU",
                                   .name         = "CPU0",
                                   .model_name   = "EPYC",
                                   .vendor_name  = "AMD",
                                   .product_name = "EPYC 7763" });

    expect_reader_has_tracks({ "process_physical_memory", "cpu_frequency [0] Core [0]",
                               "cpu_frequency [0] Core [1]" });
}

// ---------------------------------------------------------------------------
// handle(kfd_sample): region + pmc_event with args (KFD event pathway)
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_kfd_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts = 14000;
    static constexpr std::uint64_t k_end_ts   = 14500;
    const auto                     gpu        = managed_gpu_agent();
    const auto                     kfd_args =
        get_args_string(function_args_t{ { .arg_number = 0U,
                                           .arg_type   = "std::uint64_t",
                                           .arg_name   = "address",
                                           .arg_value  = "0x7f4a00001000" },
                                         { .arg_number = 1U,
                                           .arg_type   = "string",
                                           .arg_name   = "agent",
                                           .arg_value  = "5" } });
    run_processor_and_open_reader(
        { gpu },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(
                make_agent_pmc_row({ .type        = agent_type::gpu,
                                     .name        = "kfd_page_fault",
                                     .target_arch = "GPU",
                                     .description = "KFD page fault counter" }));
            metadata->add_track(track{ .track_name = "KFD Events [GPU 0]",
                                       .thread_id  = k_thread_id,
                                       .extdata    = std::string{} });
        },
        [&](rocpd_processor_t& processor) {
            const kfd_sample sample{ k_thread_id,
                                     "KFD_PAGE_FAULT",
                                     k_start_ts,
                                     k_end_ts,
                                     kfd_args,
                                     "kfd",
                                     "KFD Events [GPU 0]",
                                     R"({"source": "kfd"})",
                                     0,
                                     static_cast<std::uint8_t>(agent_type::gpu),
                                     "kfd_page_fault",
                                     1.0,
                                     static_cast<std::int64_t>(k_thread_id) };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([this](const auto& detail, const auto& tl_event) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.name, "KFD_PAGE_FAULT");
        const auto args = m_reader->get_arguments(tl_event);
        ASSERT_EQ(args.size(), 2U);
        EXPECT_EQ(args[0]->name, "address");
        EXPECT_EQ(args[0]->value, "0x7f4a00001000");
        EXPECT_EQ(args[1]->name, "agent");
        EXPECT_EQ(args[1]->value, "5");
    });

    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    expect_named_pmc_arch(pmc_infos,
                          { .name                 = "kfd_page_fault",
                            .expected_arch        = "GPU",
                            .expected_agent_type  = "GPU",
                            .expected_symbol      = "kfd_page_fault",
                            .expected_units       = "",
                            .expected_description = "KFD page fault counter" });

    expect_reader_has_tracks({ "KFD Events [GPU 0]" });
}

// ---------------------------------------------------------------------------
// DB file creation: every handle pathway creates a valid DB
// ---------------------------------------------------------------------------

struct multi_event_timestamps
{
    std::uint64_t thread_id          = 300;
    std::uint64_t managed_gpu_handle = 1;
    std::uint64_t managed_cpu_handle = 2;
    std::uint64_t queue_id           = 10;
    std::uint64_t stream_id          = 20;
    std::uint64_t hip_start_ts       = 1000;
    std::uint64_t hip_end_ts         = 1200;
    std::uint64_t kd_start_ts        = 1500;
    std::uint64_t kd_end_ts          = 2000;
    std::uint64_t kd_corr_id         = 2;
    std::uint32_t wg_size_x          = 128;
    std::uint32_t grid_size_x        = 512;
    std::uint64_t mc_start_ts        = 2500;
    std::uint64_t mc_end_ts          = 3000;
    std::uint64_t mc_size            = 2048;
    std::uint64_t sms_start_ts       = 3500;
    std::uint64_t sms_end_ts         = 3600;
    std::uint64_t sms_size           = 32768;
    std::uint64_t bt_start_ts        = 4000;
    std::uint64_t bt_end_ts          = 4100;
};

void
insert_multiple_event_type_samples(rocpd_processor_t&            processor,
                                   const multi_event_timestamps& ts)
{
    const region_sample hip_region{ ts.thread_id,    "hipLaunchKernel", 1,  0,
                                    ts.hip_start_ts, ts.hip_end_ts,     "", "",
                                    "HIP_API" };
    processor.handle(hip_region);

    const kernel_dispatch_sample kds{ ts.kd_start_ts,
                                      ts.kd_end_ts,
                                      ts.thread_id,
                                      ts.managed_gpu_handle,
                                      1,
                                      1,
                                      ts.queue_id,
                                      ts.kd_corr_id,
                                      1,
                                      0,
                                      0,
                                      ts.wg_size_x,
                                      1,
                                      1,
                                      ts.grid_size_x,
                                      1,
                                      1,
                                      ts.stream_id };
    processor.handle(kds);

    const memory_copy_sample mcs{
        ts.mc_start_ts,
        ts.mc_end_ts,
        ts.thread_id,
        ts.managed_gpu_handle,
        ts.managed_cpu_handle,
        static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY),
        static_cast<std::int32_t>(ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE),
        ts.mc_size,
        3,
        0,
        0,
        0,
        ts.stream_id
    };
    processor.handle(mcs);

    const scratch_memory_sample sms{
        ts.sms_start_ts,
        ts.sms_end_ts,
        ts.thread_id,
        ts.managed_gpu_handle,
        ts.queue_id,
        static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY),
        static_cast<std::int32_t>(ROCPROFILER_SCRATCH_MEMORY_ALLOC),
        0,
        ts.sms_size,
        4,
        0,
        ts.stream_id
    };
    processor.handle(sms);

    const backtrace_region_sample bts{
        0,          ts.thread_id, "Sampling", "bt_func", ts.bt_start_ts, ts.bt_end_ts,
        "sampling", "",           "",         ""
    };
    processor.handle(bts);
}

void
expect_multiple_event_type_counts(profiler_hub::reader_t& reader)
{
    auto counts = reader.get_event_counts();

    auto region_it = counts.find(profiler_hub::reader_types::event_type_t::region);
    ASSERT_NE(region_it, counts.end());
    EXPECT_EQ(region_it->second, 2U);

    auto kd_it = counts.find(profiler_hub::reader_types::event_type_t::kernel_dispatch);
    ASSERT_NE(kd_it, counts.end());
    EXPECT_EQ(kd_it->second, 1U);

    auto mc_it = counts.find(profiler_hub::reader_types::event_type_t::memory_copy);
    ASSERT_NE(mc_it, counts.end());
    EXPECT_EQ(mc_it->second, 1U);

    auto ma_it = counts.find(profiler_hub::reader_types::event_type_t::memory_allocate);
    ASSERT_NE(ma_it, counts.end());
    EXPECT_EQ(ma_it->second, 1U);
}

struct multi_event_saw_flags
{
    bool hip_region    = false;
    bool bt_region     = false;
    bool kernel        = false;
    bool memory_copy   = false;
    bool scratch_alloc = false;
};

void
expect_multi_db_region_details(
    profiler_hub::reader_t&                             reader,
    const profiler_hub::reader_types::timeline_event_t& tl_event,
    const multi_event_timestamps& ts, multi_event_saw_flags& saw)
{
    const auto detail = require_optional(reader.get_region_details(tl_event),
                                         "region detail not readable");
    if(detail.name == "hipLaunchKernel")
    {
        saw.hip_region = true;
        EXPECT_EQ(detail.start_timestamp, ts.hip_start_ts);
        EXPECT_EQ(detail.end_timestamp, ts.hip_end_ts);
        ASSERT_NE(detail.event, nullptr);
        EXPECT_EQ(detail.event->event_category, "HIP_API");
    }
    else if(detail.name == "bt_func")
    {
        saw.bt_region = true;
        EXPECT_EQ(detail.start_timestamp, ts.bt_start_ts);
        EXPECT_EQ(detail.end_timestamp, ts.bt_end_ts);
    }
}

void
expect_multi_db_kernel_details(
    profiler_hub::reader_t&                             reader,
    const profiler_hub::reader_types::timeline_event_t& tl_event,
    const multi_event_timestamps& ts, multi_event_saw_flags& saw)
{
    saw.kernel        = true;
    const auto detail = require_optional(reader.get_kernel_dispatch_details(tl_event),
                                         "kernel dispatch detail not readable");
    EXPECT_EQ(detail.name, "test_kernel");
    EXPECT_EQ(detail.start_timestamp, ts.kd_start_ts);
    EXPECT_EQ(detail.end_timestamp, ts.kd_end_ts);
    EXPECT_EQ(detail.workgroup_size_x, ts.wg_size_x);
    EXPECT_EQ(detail.grid_size_x, ts.grid_size_x);
}

void
expect_multi_db_memory_copy_details(
    profiler_hub::reader_t&                             reader,
    const profiler_hub::reader_types::timeline_event_t& tl_event,
    const multi_event_timestamps& ts, multi_event_saw_flags& saw)
{
    saw.memory_copy   = true;
    const auto detail = require_optional(reader.get_memory_copy_details(tl_event),
                                         "memory_copy detail not readable");
    EXPECT_EQ(detail.size, ts.mc_size);
    EXPECT_EQ(detail.start_timestamp, ts.mc_start_ts);
    EXPECT_EQ(detail.end_timestamp, ts.mc_end_ts);
    EXPECT_EQ(detail.name, "MEMORY_COPY_HOST_TO_DEVICE");
}

void
expect_multi_db_scratch_details(
    profiler_hub::reader_t&                             reader,
    const profiler_hub::reader_types::timeline_event_t& tl_event,
    const multi_event_timestamps& ts, multi_event_saw_flags& saw)
{
    saw.scratch_alloc = true;
    const auto detail = require_optional(reader.get_memory_alloc_details(tl_event),
                                         "memory_allocate detail not readable");
    EXPECT_EQ(detail.size, ts.sms_size);
    EXPECT_EQ(detail.type, "ALLOC");
    EXPECT_EQ(detail.level, "SCRATCH");
    EXPECT_EQ(detail.start_timestamp, ts.sms_start_ts);
    EXPECT_EQ(detail.end_timestamp, ts.sms_end_ts);
}

void
expect_multiple_event_type_details(profiler_hub::reader_t&       reader,
                                   const multi_event_timestamps& ts)
{
    multi_event_saw_flags saw{};

    for(const auto& tl_event : reader.get_events())
    {
        switch(tl_event.unique_identifier.type)
        {
            case profiler_hub::reader_types::event_type_t::region:
                expect_multi_db_region_details(reader, tl_event, ts, saw);
                break;
            case profiler_hub::reader_types::event_type_t::kernel_dispatch:
                expect_multi_db_kernel_details(reader, tl_event, ts, saw);
                break;
            case profiler_hub::reader_types::event_type_t::memory_copy:
                expect_multi_db_memory_copy_details(reader, tl_event, ts, saw);
                break;
            case profiler_hub::reader_types::event_type_t::memory_allocate:
                expect_multi_db_scratch_details(reader, tl_event, ts, saw);
                break;
            default: break;
        }
    }

    EXPECT_TRUE(saw.hip_region);
    EXPECT_TRUE(saw.bt_region);
    EXPECT_TRUE(saw.kernel);
    EXPECT_TRUE(saw.memory_copy);
    EXPECT_TRUE(saw.scratch_alloc);
}

TEST_F(rocpd_write_read_test_interface, multiple_event_types_in_single_db)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    const multi_event_timestamps ts{};
    const auto                   gpu = managed_gpu_agent();
    const auto                   cpu = managed_cpu_agent();
    run_processor_and_open_reader(
        { gpu, cpu },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "test_kernel");
            metadata->add_track(track{ .track_name = "Sampling",
                                       .thread_id  = k_thread_id,
                                       .extdata    = std::string{} });
        },
        [&ts](rocpd_processor_t& processor) {
            insert_multiple_event_type_samples(processor, ts);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_multiple_event_type_counts(*m_reader);
    expect_multiple_event_type_details(*m_reader, ts);
    expect_readback_agent_fields({ .agent_type   = "GPU",
                                   .name         = "gfx90a",
                                   .model_name   = "MI210",
                                   .vendor_name  = "AMD",
                                   .product_name = "Instinct MI210" });
    expect_readback_agent_fields({ .agent_type   = "CPU",
                                   .name         = "CPU0",
                                   .model_name   = "EPYC",
                                   .vendor_name  = "AMD",
                                   .product_name = "EPYC 7763" });
}

// ---------------------------------------------------------------------------
// Region with call_stack: handle(region_sample) sets ev.call_stack
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_region_with_call_stack_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts = 15000;
    static constexpr std::uint64_t k_end_ts   = 15500;

    run_processor_and_open_reader({}, {}, [](rocpd_processor_t& processor) {
        const region_sample reg{
            k_thread_id, "hsa_signal_wait",           1,  0,        k_start_ts,
            k_end_ts,    R"([{"function": "main"}])", "", "HSA_API"
        };
        processor.handle(reg);
    });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([this](const auto& detail, const auto& tl_event) {
        EXPECT_EQ(detail.name, "hsa_signal_wait");
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        ASSERT_NE(detail.event, nullptr);
        EXPECT_EQ(detail.event->event_category, "HSA_API");
        EXPECT_FALSE(m_reader->get_call_stack(tl_event).empty());
    });
}

// ---------------------------------------------------------------------------
// Kernel dispatch with full grid: validate all dimension fields
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_kernel_dispatch_full_grid)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts         = 20000;
    static constexpr std::uint64_t k_end_ts           = 25000;
    static constexpr std::uint64_t k_dispatch_id      = 77;
    static constexpr std::uint64_t k_queue_handle     = 10;
    static constexpr std::uint64_t k_corr_id_internal = 5;
    static constexpr std::uint64_t k_corr_id_ancestor = 512;
    static constexpr std::uint64_t k_corr_id_external = 16384;
    static constexpr std::uint64_t k_workgroup_size_x = 64;
    static constexpr std::uint64_t k_workgroup_size_y = 4;
    static constexpr std::uint64_t k_workgroup_size_z = 2;
    static constexpr std::uint64_t k_grid_size_x      = 256;
    static constexpr std::uint64_t k_grid_size_y      = 16;
    static constexpr std::uint64_t k_grid_size_z      = 8;

    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "matmul_kernel");
        },
        [](rocpd_processor_t& processor) {
            const kernel_dispatch_sample kds{ k_start_ts,
                                              k_end_ts,
                                              k_thread_id,
                                              k_managed_gpu_handle,
                                              1,
                                              k_dispatch_id,
                                              k_queue_id,
                                              k_queue_handle,
                                              k_corr_id_internal,
                                              k_corr_id_ancestor,
                                              k_corr_id_external,
                                              k_workgroup_size_x,
                                              k_workgroup_size_y,
                                              k_workgroup_size_z,
                                              k_grid_size_x,
                                              k_grid_size_y,
                                              k_grid_size_z,
                                              k_stream_id };
            processor.handle(kds);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_kernel_dispatch([](const auto& detail) {
        EXPECT_EQ(detail.dispatch_id, k_dispatch_id);
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.workgroup_size_x, k_workgroup_size_x);
        EXPECT_EQ(detail.workgroup_size_y, k_workgroup_size_y);
        EXPECT_EQ(detail.workgroup_size_z, k_workgroup_size_z);
        EXPECT_EQ(detail.grid_size_x, k_grid_size_x);
        EXPECT_EQ(detail.grid_size_y, k_grid_size_y);
        EXPECT_EQ(detail.grid_size_z, k_grid_size_z);
        EXPECT_EQ(detail.name, "matmul_kernel");
    });
}

// ---------------------------------------------------------------------------
// Memory copy with addresses: validate src/dst address fields
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test_interface, handle_memory_copy_addresses_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    static constexpr std::uint64_t k_start_ts    = 30000;
    static constexpr std::uint64_t k_end_ts      = 31000;
    static constexpr std::uint64_t k_copy_size   = 1024;
    static constexpr std::uint64_t k_dst_address = 0xDEAD0000;
    static constexpr std::uint64_t k_src_address = 0xBEEF0000;

    run_processor_and_open_reader(
        { managed_gpu_agent(), managed_cpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_stream(k_stream_id);
        },
        [](rocpd_processor_t& processor) {
            const memory_copy_sample mcs{
                k_start_ts,
                k_end_ts,
                k_thread_id,
                k_managed_gpu_handle,
                k_managed_cpu_handle,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY),
                static_cast<std::int32_t>(ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST),
                k_copy_size,
                1,
                0,
                k_dst_address,
                k_src_address,
                k_stream_id
            };
            processor.handle(mcs);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_copy([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, k_start_ts);
        EXPECT_EQ(detail.end_timestamp, k_end_ts);
        EXPECT_EQ(detail.size, k_copy_size);
        EXPECT_EQ(detail.name, "MEMORY_COPY_DEVICE_TO_HOST");
        EXPECT_EQ(require_optional(detail.dst_address, "dst_address missing"),
                  k_dst_address);
        EXPECT_EQ(require_optional(detail.src_address, "src_address missing"),
                  k_src_address);
    });
}

// NOLINTEND(readability-function-cognitive-complexity, readability-function-size)
