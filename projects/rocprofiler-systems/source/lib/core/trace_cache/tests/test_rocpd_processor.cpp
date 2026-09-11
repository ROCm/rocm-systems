// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/node_info.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/rocpd_processor.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/nic/sample.hpp"
#include "library/thread_info.hpp"
#include <cstdint>

#include <profiler-hub/reader.hpp>
#include <profiler-hub/reader_types.hpp>
#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer_types.hpp>
#include <rocprofiler-sdk/callback_tracing.h>

#include <gtest/gtest.h>
#include <timemory/settings.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <sys/types.h>
#include <unordered_set>
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
    static const std::optional<thread_info> none{};
    return none;
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

template <typename PmcList>
void
expect_nic_pmc_catalog(const PmcList& pmc_infos)
{
    ASSERT_EQ(pmc_infos.size(), k_nic_pmcs.size());

    std::unordered_set<std::string> seen;
    for(const auto& pmc_row : pmc_infos)
    {
        expect_nic_pmc_row(pmc_row);
        seen.insert(pmc_row->name);
        const nic_pmc_spec* spec = nullptr;
        for(const auto& entry : k_nic_pmcs)
        {
            if(entry.name == pmc_row->name)
            {
                spec = &entry;
                break;
            }
        }
        ASSERT_NE(spec, nullptr) << pmc_row->name;
        EXPECT_EQ(pmc_row->symbol, spec->name);
        EXPECT_EQ(pmc_row->units, spec->units);
    }

    for(const auto& spec : k_nic_pmcs)
    {
        EXPECT_TRUE(seen.count(spec.name)) << "missing PMC " << spec.name;
    }
}

template <typename PmcList>
void
expect_named_pmc_arch(const PmcList& pmc_infos, const char* name,
                      const char* expected_arch, const char* expected_agent_type,
                      const char* expected_symbol      = nullptr,
                      const char* expected_units       = nullptr,
                      const char* expected_description = nullptr)
{
    bool found = false;
    for(const auto& pmc_row : pmc_infos)
    {
        if(pmc_row->name != name)
        {
            continue;
        }
        found = true;
        EXPECT_EQ(pmc_row->target_arch, expected_arch) << pmc_row->name;
        ASSERT_NE(pmc_row->agent_info, nullptr) << pmc_row->name;
        EXPECT_EQ(pmc_row->agent_info->agent_type, expected_agent_type) << pmc_row->name;
        if(expected_symbol != nullptr)
        {
            EXPECT_EQ(pmc_row->symbol, expected_symbol) << pmc_row->name;
        }
        if(expected_units != nullptr)
        {
            EXPECT_EQ(pmc_row->units, expected_units) << pmc_row->name;
        }
        if(expected_description != nullptr)
        {
            EXPECT_EQ(pmc_row->description, expected_description) << pmc_row->name;
        }
    }
    EXPECT_TRUE(found) << "missing PMC " << name;
}
}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// rocpd_processor.cpp — integration: processor → flush → reader read-back
//
// Each TEST_F below is split into two parts:
//   Prepare — build metadata, call rocpd_processor_t::handle(), finalize, open reader
//   Validate — assert via profiler_hub::reader_t (no direct SQLite access)
// ═══════════════════════════════════════════════════════════════════════════

class rocpd_write_read_test : public ::testing::Test
{
protected:
    static constexpr size_t NODE_ID   = 1;
    static constexpr size_t PID       = 200;
    static constexpr size_t PPID      = 100;
    static constexpr size_t THREAD_ID = 300;
    static constexpr size_t QUEUE_ID  = 10;
    static constexpr size_t STREAM_ID = 20;

    void SetUp() override
    {
        m_temp_dir = std::filesystem::temp_directory_path() /
                     ("rocpd_test_" + std::to_string(::getpid()) + "_" +
                      std::to_string(test_counter_++));
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
        metadata->add_track(track{ track_name, std::nullopt, std::string{} });
    }

    static void seed_gpu_smi_pmc_row(const std::shared_ptr<metadata_registry>& metadata,
                                     const char*                               pmc_name,
                                     const char* description = nullptr)
    {
        metadata->add_pmc_info(
            make_agent_pmc_row(agent_type::gpu, 0, pmc_name, "GPU", description));
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
        const auto  it =
            std::find_if(queues.begin(), queues.end(),
                         [&](const auto& queue) { return queue->name == expected_name; });
        ASSERT_NE(it, queues.end()) << "queue not in reader: " << expected_name;
        ASSERT_NE((*it)->process_info, nullptr);
        EXPECT_EQ((*it)->process_info->pid, PID);
    }

    void expect_readback_stream_named(const char* expected_name) const
    {
        const auto& streams = m_reader->get_all_streams();
        const auto  it =
            std::find_if(streams.begin(), streams.end(), [&](const auto& stream) {
                return stream->name == expected_name;
            });
        ASSERT_NE(it, streams.end()) << "stream not in reader: " << expected_name;
        ASSERT_NE((*it)->process_info, nullptr);
        EXPECT_EQ((*it)->process_info->pid, PID);
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

    void expect_rocpd_database_on_disk() const
    {
        ASSERT_FALSE(m_db_path.empty())
            << "run_processor_and_open_reader must run before DB assertions";
        const std::filesystem::path path{ m_db_path };
        ASSERT_TRUE(std::filesystem::exists(path)) << m_db_path;
        ASSERT_TRUE(std::filesystem::is_regular_file(path)) << m_db_path;
        EXPECT_GT(std::filesystem::file_size(path), 0U) << m_db_path;
        ASSERT_NE(m_reader, nullptr);
        EXPECT_FALSE(m_uuid.empty());
    }

    void expect_readback_agent_fields(const char* agent_type, const char* name,
                                      const char* model_name, const char* vendor_name,
                                      const char* product_name) const
    {
        const auto agents = m_reader->get_all_agents();
        const auto it =
            std::find_if(agents.begin(), agents.end(), [&](const auto& agent_ptr) {
                return agent_ptr->agent_type == agent_type;
            });
        ASSERT_NE(it, agents.end())
            << "agent type not found in read-back: " << agent_type;
        EXPECT_EQ((*it)->name, name);
        EXPECT_EQ((*it)->model_name, model_name);
        EXPECT_EQ((*it)->vendor_name, vendor_name);
        EXPECT_EQ((*it)->product_name, product_name);
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
        const auto event =
            find_first_event(profiler_hub::reader_types::event_type_t::memory_allocate);
        ASSERT_TRUE(event.has_value()) << "memory_allocate event not found in read-back";
        const auto detail = m_reader->get_memory_alloc_details(*event);
        ASSERT_TRUE(detail.has_value());
        check(*detail);
    }

    void expect_region(
        const std::function<void(const profiler_hub::reader_types::region_data_t&,
                                 const profiler_hub::reader_types::timeline_event_t&)>&
            check) const
    {
        const auto event =
            find_first_event(profiler_hub::reader_types::event_type_t::region);
        ASSERT_TRUE(event.has_value()) << "region event not found in read-back";
        const auto detail = m_reader->get_region_details(*event);
        ASSERT_TRUE(detail.has_value());
        check(*detail, *event);
    }

    void expect_kernel_dispatch(
        const std::function<
            void(const profiler_hub::reader_types::kernel_dispatch_data_t&)>& check) const
    {
        const auto event =
            find_first_event(profiler_hub::reader_types::event_type_t::kernel_dispatch);
        ASSERT_TRUE(event.has_value()) << "kernel_dispatch event not found in read-back";
        const auto detail = m_reader->get_kernel_dispatch_details(*event);
        ASSERT_TRUE(detail.has_value()) << "kernel dispatch detail should be readable";
        check(*detail);
    }

    void expect_memory_copy(
        const std::function<void(const profiler_hub::reader_types::memory_copy_data_t&)>&
            check) const
    {
        const auto event =
            find_first_event(profiler_hub::reader_types::event_type_t::memory_copy);
        ASSERT_TRUE(event.has_value()) << "memory_copy event not found in read-back";
        const auto detail = m_reader->get_memory_copy_details(*event);
        ASSERT_TRUE(detail.has_value());
        check(*detail);
    }

    // Drive the production path: agent_manager →
    // rocpd_processor_t::prepare_for_processing() (post_process_metadata +
    // make_agent_uid) → finalize_processing() → reader.
    void run_processor_and_open_reader(
        std::vector<agent> agents,
        const std::function<void(const std::shared_ptr<metadata_registry>&)>&
                                                       metadata_setup = {},
        const std::function<void(rocpd_processor_t&)>& on_processor   = {})
    {
        scoped_force_rocpd_metadata_registration force_rocpd;

        const auto prev_out = tim::settings::output_path();
        const auto out_dir  = m_temp_dir / "processor_out";
        std::filesystem::create_directories(out_dir);
        tim::settings::output_path() = out_dir.string();
        rocprofsys::reset_database_path_memo();

        struct restore_output_path
        {
            std::string prev_out;
            ~restore_output_path() { tim::settings::output_path() = prev_out; }
        } restore_out{ prev_out };

        auto metadata = std::make_shared<metadata_registry>();
        rocprofsys::trace_cache::info::process proc{};
        proc.pid     = static_cast<pid_t>(PID);
        proc.ppid    = static_cast<pid_t>(PPID);
        proc.command = "test_binary";
        proc.start   = 1000;
        proc.end     = 9000;
        metadata->set_process(proc);
        seed_test_thread(metadata);
        // rocpd_processor_t::handle(memory_copy_sample) sets trace env queue_id to 0.
        metadata->add_queue(0);
        if(metadata_setup)
        {
            metadata_setup(metadata);
        }

        auto mgr = std::make_shared<agent_manager>();
        for(auto& a : agents)
        {
            mgr->insert_agent(a);
        }

        const int            pid  = static_cast<int>(PID);
        const int            ppid = static_cast<int>(PPID);
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

    static std::string extract_rocpd_uuid(const std::filesystem::path& db_path)
    {
        std::ifstream     in{ db_path, std::ios::binary };
        const std::string data{ std::istreambuf_iterator<char>{ in },
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
        thread_info.parent_process_id = static_cast<std::int32_t>(PPID);
        thread_info.process_id        = static_cast<std::int32_t>(PID);
        thread_info.thread_id         = THREAD_ID;
        thread_info.start             = 1000;
        thread_info.end               = 9000;
        metadata->add_thread_info(thread_info);
    }

    static void seed_nic_pmc_catalog(const std::shared_ptr<metadata_registry>& metadata)
    {
        for(const auto& spec : k_nic_pmcs)
        {
            metadata->add_pmc_info(make_nic_metadata_pmc(spec));
        }
    }

    static constexpr std::uint64_t MANAGED_GPU_HANDLE = 1;
    static constexpr std::uint64_t MANAGED_CPU_HANDLE = 2;

    static agent managed_gpu_agent()
    {
        auto result   = gpu_agent();
        result.handle = MANAGED_GPU_HANDLE;
        return result;
    }

    static agent managed_cpu_agent()
    {
        auto result   = cpu_agent();
        result.handle = MANAGED_CPU_HANDLE;
        return result;
    }

    static pmc make_agent_pmc_row(agent_type type, size_t agent_type_index,
                                  const char* name, const char* target_arch,
                                  const char* description = nullptr)
    {
        pmc row{};
        row.type             = type;
        row.agent_type_index = agent_type_index;
        row.target_arch      = target_arch;
        row.name             = name;
        row.symbol           = name;
        row.description      = description != nullptr ? description : name;
        row.units            = "";
        row.value_type       = "ABS";
        row.extdata          = "{}";
        return row;
    }

    static void seed_gpu_queue_stream(const std::shared_ptr<metadata_registry>& metadata)
    {
        metadata->add_queue(QUEUE_ID);
        metadata->add_stream(STREAM_ID);
    }

    static void seed_code_object(const std::shared_ptr<metadata_registry>& metadata,
                                 std::uint64_t code_object_id, std::uint64_t agent_handle)
    {
        static std::string uri = "file:///test_code_object.co";
        rocprofiler_callback_tracing_code_object_load_data_t co{};
        co.code_object_id = code_object_id;
        co.uri            = uri.c_str();
#if(ROCPROFILER_VERSION >= 600)
        co.agent_id.handle = agent_handle;
#else
        co.rocp_agent.handle = agent_handle;
#endif
        co.storage_type = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
        metadata->add_code_object(co);
    }

    static void seed_kernel_symbol(const std::shared_ptr<metadata_registry>& metadata,
                                   std::uint64_t kernel_id, const char* kernel_name,
                                   std::uint64_t agent_handle = MANAGED_GPU_HANDLE)
    {
        static std::unordered_map<std::uint64_t, std::string> stored_names;
        stored_names[kernel_id] = kernel_name;
        seed_code_object(metadata, 1, agent_handle);
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t ks{};
        ks.kernel_id      = kernel_id;
        ks.code_object_id = 1;
        ks.kernel_name    = stored_names[kernel_id].c_str();
        metadata->add_kernel_symbol(ks);
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

    static int test_counter_;
};

int rocpd_write_read_test::test_counter_ = 0;

TEST_F(rocpd_write_read_test, agents_round_trip_all_types)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({ gpu_agent(), cpu_agent(), nic_agent() });
    // Validate: profiler_hub::reader_t read-back matches inserted values.
    ASSERT_EQ(m_reader->get_all_agents().size(), 3U);
    expect_readback_agent_fields("GPU", "gfx90a", "MI210", "AMD", "Instinct MI210");
    expect_readback_agent_fields("CPU", "CPU0", "EPYC", "AMD", "EPYC 7763");
    expect_readback_agent_fields("NIC", "NIC0", "CX7", "AI NIC", "AI NIC");
}

TEST_F(rocpd_write_read_test, handle_ainic_pmc_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    constexpr double k_rx_ucast_bytes   = 1048576.0;
    constexpr double k_tx_ucast_bytes   = 524288.0;
    constexpr size_t k_sample_timestamp = 12000;

    run_processor_and_open_reader(
        { nic_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(make_nic_metadata_pmc(k_nic_pmcs[4]));
            metadata->add_pmc_info(make_nic_metadata_pmc(k_nic_pmcs[5]));
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
            ainic_pmc_sample sample{ enabled, 0, "NIC0", k_sample_timestamp, metrics };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    ASSERT_EQ(m_reader->get_all_agents().size(), 1U);
    expect_readback_agent_fields("NIC", "NIC0", "CX7", "AI NIC", "AI NIC");

    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 2U);
    expect_named_pmc_arch(pmc_infos, "nic_rx_ucast_bytes", "NIC", "NIC",
                          "nic_rx_ucast_bytes", k_nic_pmcs[4].units);
    expect_named_pmc_arch(pmc_infos, "nic_tx_ucast_bytes", "NIC", "NIC",
                          "nic_tx_ucast_bytes", k_nic_pmcs[5].units);

    expect_reader_has_tracks(
        { "ainic_rx_rdma_ucast_bytes", "ainic_tx_rdma_ucast_bytes" });
}

// Mirrors tests/rocpd-validation-rules/ainic/ainic-rdma-rules.json: every
// cache_policy.hpp NIC PMC name is registered with target_arch NIC.
TEST_F(rocpd_write_read_test, nic_rdma_pmc_catalog_target_arch)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({ nic_agent() },
                                  [](const std::shared_ptr<metadata_registry>& metadata) {
                                      seed_nic_pmc_catalog(metadata);
                                  });
    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_nic_pmc_catalog(m_reader->get_all_pmc_info());
}

TEST_F(rocpd_write_read_test, nic_pmc_info_invalid_target_arch_rejected)
{
    scoped_force_rocpd_metadata_registration force_rocpd;

    const auto prev_out = tim::settings::output_path();
    const auto out_dir  = m_temp_dir / "processor_invalid_arch";
    std::filesystem::create_directories(out_dir);
    tim::settings::output_path() = out_dir.string();
    rocprofsys::reset_database_path_memo();
    struct restore_out_path
    {
        std::string prev;
        ~restore_out_path() { tim::settings::output_path() = prev; }
    } restore_out{ prev_out };

    auto metadata = std::make_shared<metadata_registry>();
    rocprofsys::trace_cache::info::process proc{};
    proc.pid     = static_cast<pid_t>(PID);
    proc.ppid    = static_cast<pid_t>(PPID);
    proc.command = "test_binary";
    proc.start   = 1000;
    proc.end     = 9000;
    metadata->set_process(proc);

    auto bad_pmc        = make_nic_metadata_pmc(k_nic_pmcs[0]);
    bad_pmc.target_arch = "AINIC";
    metadata->add_pmc_info(bad_pmc);

    auto  mgr = std::make_shared<agent_manager>();
    agent nic = nic_agent();
    mgr->insert_agent(nic);

    output_file_registry registry;
    rocpd_processor_t    processor{ metadata, mgr, static_cast<int>(PID),
                                 static_cast<int>(PPID), registry };

    // Validate: prepare_for_processing rejects invalid target_arch (no rocpd DB).
    EXPECT_THROW(processor.prepare_for_processing(), std::invalid_argument);
    EXPECT_TRUE(find_rocpd_database_in_directory(out_dir).empty())
        << "invalid PMC metadata must not create a rocpd database";
}

TEST_F(rocpd_write_read_test, gpu_and_nic_pmc_keep_distinct_target_arch)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent(), nic_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(
                make_agent_pmc_row(agent_type::gpu, 0, "gfx_busy", "GPU"));
            metadata->add_pmc_info(make_nic_metadata_pmc(k_nic_pmcs[4]));
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 2U);
    expect_named_pmc_arch(pmc_infos, "gfx_busy", "GPU", "GPU", "gfx_busy");
    expect_named_pmc_arch(pmc_infos, "nic_rx_ucast_bytes", "NIC", "NIC",
                          "nic_rx_ucast_bytes", k_nic_pmcs[4].units);
}

// ---------------------------------------------------------------------------
// Kernel dispatch: write, read back, validate timestamps + grid sizes
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, kernel_dispatch_values_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "my_test_kernel");
        },
        [](rocpd_processor_t& processor) {
            kernel_dispatch_sample kds{ 5000, 6000,     THREAD_ID, MANAGED_GPU_HANDLE,
                                        1,    42,       QUEUE_ID,  1,
                                        0,    0,        0,         256,
                                        1,    1,        1024,      1,
                                        1,    STREAM_ID };
            processor.handle(kds);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_kernel_dispatch([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, 5000U);
        EXPECT_EQ(detail.end_timestamp, 6000U);
        EXPECT_EQ(detail.dispatch_id, 42U);
        EXPECT_EQ(detail.workgroup_size_x, 256U);
        EXPECT_EQ(detail.workgroup_size_y, 1U);
        EXPECT_EQ(detail.workgroup_size_z, 1U);
        EXPECT_EQ(detail.grid_size_x, 1024U);
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

TEST_F(rocpd_write_read_test, region_with_args_values_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    const auto region_args =
        get_args_string(function_args_t{ { 0U, "void*", "dst", "0x7f0000000000" },
                                         { 1U, "size_t", "sizeBytes", "4096" } });
    run_processor_and_open_reader({}, {}, [&](rocpd_processor_t& processor) {
        region_sample reg{ THREAD_ID, "hipMemcpy", 1,           0,        5000,
                           5200,      "",          region_args, "HIP_API" };
        processor.handle(reg);
    });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([this](const auto& detail, const auto& tl_event) {
        EXPECT_EQ(detail.start_timestamp, 5000U);
        EXPECT_EQ(detail.end_timestamp, 5200U);
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

TEST_F(rocpd_write_read_test, memory_copy_values_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent(), managed_cpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_stream(STREAM_ID);
        },
        [](rocpd_processor_t& processor) {
            memory_copy_sample mcs{
                6500,
                7000,
                THREAD_ID,
                MANAGED_GPU_HANDLE,
                MANAGED_CPU_HANDLE,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY),
                static_cast<std::int32_t>(ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE),
                4096,
                1,
                0,
                0x100000,
                0x7F0000000000,
                STREAM_ID
            };
            processor.handle(mcs);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_copy([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, 6500U);
        EXPECT_EQ(detail.end_timestamp, 7000U);
        EXPECT_EQ(detail.size, 4096U);
        EXPECT_EQ(detail.name, "MEMORY_COPY_HOST_TO_DEVICE");
        ASSERT_TRUE(detail.dst_address.has_value());
        EXPECT_EQ(detail.dst_address.value(), 0x100000U);
        ASSERT_TRUE(detail.src_address.has_value());
        EXPECT_EQ(detail.src_address.value(), 0x7F0000000000U);
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

TEST_F(rocpd_write_read_test, event_counts_match_inserted_data)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "count_test_kernel");
        },
        [](rocpd_processor_t& processor) {
            for(int idx = 0; idx < 2; ++idx)
            {
                const auto    name = "region_" + std::to_string(idx);
                region_sample reg{ THREAD_ID,
                                   name,
                                   static_cast<std::uint64_t>(idx),
                                   0,
                                   static_cast<std::uint64_t>(1000 + idx * 100),
                                   static_cast<std::uint64_t>(1050 + idx * 100),
                                   "",
                                   "",
                                   "HIP_API" };
                processor.handle(reg);
            }

            kernel_dispatch_sample kds{ 5000, 6000,     THREAD_ID, MANAGED_GPU_HANDLE,
                                        1,    1,        QUEUE_ID,  10,
                                        0,    0,        0,         64,
                                        1,    1,        256,       1,
                                        1,    STREAM_ID };
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
        auto detail = m_reader->get_region_details(tl_event);
        ASSERT_TRUE(detail.has_value());
        region_times[detail->name] = { detail->start_timestamp, detail->end_timestamp };
    }
    ASSERT_EQ(region_times.size(), 2U);
    EXPECT_EQ(region_times["region_0"].first, 1000U);
    EXPECT_EQ(region_times["region_0"].second, 1050U);
    EXPECT_EQ(region_times["region_1"].first, 1100U);
    EXPECT_EQ(region_times["region_1"].second, 1150U);

    for(const auto& tl_event : m_reader->get_events())
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::kernel_dispatch)
        {
            continue;
        }
        auto detail = m_reader->get_kernel_dispatch_details(tl_event);
        ASSERT_TRUE(detail.has_value());
        EXPECT_EQ(detail->name, "count_test_kernel");
        EXPECT_EQ(detail->start_timestamp, 5000U);
        EXPECT_EQ(detail->end_timestamp, 6000U);
        EXPECT_EQ(detail->workgroup_size_x, 64U);
        EXPECT_EQ(detail->grid_size_x, 256U);
    }
}

// ---------------------------------------------------------------------------
// Metadata round-trip: node, process, thread, queue, stream
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, metadata_round_trip)
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
    EXPECT_EQ(processes[0]->pid, PID);
    EXPECT_EQ(processes[0]->command, "test_binary");

    auto threads = m_reader->get_all_threads();
    ASSERT_EQ(threads.size(), 1U);
    EXPECT_EQ(threads[0]->thread_id, THREAD_ID);
    EXPECT_EQ(threads[0]->name, "Thread 300");

    ASSERT_EQ(m_reader->get_all_queues().size(), 2U);
    expect_readback_queue_named("Queue 0");
    expect_readback_queue_named("Queue 10");

    ASSERT_EQ(m_reader->get_all_streams().size(), 1U);
    expect_readback_stream_named("Stream 20");

    EXPECT_EQ(processes[0]->ppid, PPID);
    EXPECT_EQ(processes[0]->start, 1000U);
    EXPECT_EQ(processes[0]->end, 9000U);
    EXPECT_EQ(threads[0]->start, 1000U);
    EXPECT_EQ(threads[0]->end, 9000U);
}

// ---------------------------------------------------------------------------
// Output file existence and non-empty after flush
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, flush_creates_nonempty_file)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({}, {}, [](rocpd_processor_t& processor) {
        region_sample reg{ THREAD_ID, "flush_test", 0, 0, 1000, 2000, "", "", "HIP_API" };
        processor.handle(reg);
    });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    ASSERT_EQ(count_timeline_events(profiler_hub::reader_types::event_type_t::region),
              1U);
    const auto events = m_reader->get_events();
    ASSERT_EQ(events.size(), 1U);
    EXPECT_EQ(events[0].display_name, "flush_test");
    EXPECT_EQ(events[0].start_timestamp, 1000U);
    EXPECT_EQ(events[0].end_timestamp, 2000U);
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

TEST_F(rocpd_write_read_test, handle_scratch_memory_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
        },
        [](rocpd_processor_t& processor) {
            scratch_memory_sample sms{
                3000,
                3100,
                THREAD_ID,
                MANAGED_GPU_HANDLE,
                QUEUE_ID,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY),
                static_cast<std::int32_t>(ROCPROFILER_SCRATCH_MEMORY_ALLOC),
                0,
                131072,
                100,
                50,
                STREAM_ID
            };
            processor.handle(sms);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_alloc([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, 3000U);
        EXPECT_EQ(detail.end_timestamp, 3100U);
        EXPECT_EQ(detail.type, "ALLOC");
        EXPECT_EQ(detail.level, "SCRATCH");
        EXPECT_EQ(detail.size, 131072U);
    });
    expect_readback_agent_fields("GPU", "gfx90a", "MI210", "AMD", "Instinct MI210");
}

// ---------------------------------------------------------------------------
// handle(memory_allocate_sample): memory_alloc with agent + address
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_memory_allocate_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
#if(ROCPROFILER_VERSION < 600)
    GTEST_SKIP() << "memory_allocate_sample requires ROCPROFILER_VERSION >= 600";
#else
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_stream(STREAM_ID);
        },
        [](rocpd_processor_t& processor) {
            memory_allocate_sample mas{
                7000,
                7200,
                THREAD_ID,
                MANAGED_GPU_HANDLE,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION),
                static_cast<std::int32_t>(ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE),
                8192,
                200,
                100,
                0x7F0000100000,
                STREAM_ID
            };
            processor.handle(mas);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_alloc([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, 7000U);
        EXPECT_EQ(detail.end_timestamp, 7200U);
        EXPECT_EQ(detail.type, "ALLOC");
        EXPECT_EQ(detail.level, "REAL");
        EXPECT_EQ(detail.size, 8192U);
        ASSERT_TRUE(detail.address.has_value());
        EXPECT_EQ(detail.address.value(), 0x7F0000100000U);
    });
    expect_readback_agent_fields("GPU", "gfx90a", "MI210", "AMD", "Instinct MI210");
#endif
}

// ---------------------------------------------------------------------------
// handle(backtrace_region_sample): region with track_name and extdata
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_backtrace_region_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        {},
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_track(track{ "Sampling [CPU 0]", THREAD_ID, std::string{} });
        },
        [](rocpd_processor_t& processor) {
            backtrace_region_sample bts{
                0,    THREAD_ID, "Sampling [CPU 0]", "backtrace_sample_func",
                8000, 8500,      "sampling",         R"({"backtrace": true})",
                "",   ""
            };
            processor.handle(bts);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([](const auto& detail, const auto&) {
        EXPECT_EQ(detail.start_timestamp, 8000U);
        EXPECT_EQ(detail.end_timestamp, 8500U);
        EXPECT_EQ(detail.name, "backtrace_sample_func");
        ASSERT_NE(detail.event, nullptr);
        EXPECT_EQ(detail.event->event_category, "sampling");
    });
    expect_reader_has_tracks({ "Sampling [CPU 0]" });
}

// ---------------------------------------------------------------------------
// handle(in_time_sample): pmc_event with track + sample timestamp
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_in_time_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(
                make_agent_pmc_row(agent_type::gpu, 0, "my_track", "GPU", "IN_TIME"));
            add_process_scoped_track(metadata, "my_track");
        },
        [](rocpd_processor_t& processor) {
            in_time_sample its{ 0, "my_track", 9500, R"({"metadata": "test"})", 5, 3,
                                0, "",         "" };
            processor.handle(its);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    expect_named_pmc_arch(pmc_infos, "my_track", "GPU", "GPU", "my_track", "", "IN_TIME");

    expect_reader_has_tracks({ "my_track" });
}

// ---------------------------------------------------------------------------
// handle(pmc_event_with_sample): pmc_event with agent + value + tid
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_pmc_event_with_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(make_agent_pmc_row(agent_type::gpu, 0, "SQ_WAVES",
                                                      "GPU", "Shader wavefronts"));
            metadata->add_track(track{ "SQ_WAVES [GPU 0]", THREAD_ID, std::string{} });
        },
        [](rocpd_processor_t& processor) {
            pmc_event_with_sample pmc{ 0,
                                       "SQ_WAVES [GPU 0]",
                                       10000,
                                       "{}",
                                       10,
                                       5,
                                       42,
                                       "",
                                       "",
                                       0,
                                       static_cast<std::uint8_t>(agent_type::gpu),
                                       "SQ_WAVES",
                                       1024.0,
                                       static_cast<std::int64_t>(THREAD_ID) };
            processor.handle(pmc);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    expect_named_pmc_arch(pmc_infos, "SQ_WAVES", "GPU", "GPU", "SQ_WAVES", "",
                          "Shader wavefronts");

    expect_reader_has_tracks({ "SQ_WAVES [GPU 0]" });
}

// ---------------------------------------------------------------------------
// handle(gpu_pmc_sample): multiple scalar PMC inserts for GPU metrics
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_gpu_pmc_sample_pathway)
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
            rocprofsys::pmc::collectors::gpu::enabled_metrics enabled{};
            enabled.bits.gfx_activity        = 1;
            enabled.bits.umc_activity        = 1;
            enabled.bits.hotspot_temperature = 1;
            rocprofsys::pmc::collectors::gpu::metrics metrics{};
            metrics.gfx_activity        = 85.0;
            metrics.umc_activity        = 42.0;
            metrics.hotspot_temperature = 72.0;
            gpu_pmc_sample sample{ enabled, 0, 11000, metrics };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 3U);
    expect_named_pmc_arch(pmc_infos, "device_busy_gfx", "GPU", "GPU", "device_busy_gfx");
    expect_named_pmc_arch(pmc_infos, "device_busy_umc", "GPU", "GPU", "device_busy_umc");
    expect_named_pmc_arch(pmc_infos, "device_temp", "GPU", "GPU", "device_temp");

    expect_reader_has_tracks({ "device_busy_gfx", "device_busy_umc", "device_temp" });
}

// ---------------------------------------------------------------------------
// handle(gpu_perf_counter_sample): SDK hardware counter batch → PMC events
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_gpu_perf_counter_sample_pathway)
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
            metadata->add_pmc_info(make_agent_pmc_row(
                agent_type::gpu, 0, k_pmc_name.c_str(), "GPU", "Shader wavefronts"));
            // gpu_perf_counter handle() emits process-scoped tracks (no thread_id).
            metadata->add_track(track{ k_track_name, std::nullopt, std::string{} });
            gpu_perf_counter_name_entry name_entry{};
            name_entry.counter_id    = k_counter_id;
            name_entry.pmc_info_name = k_pmc_name;
            name_entry.track_name    = k_track_name;
            metadata->set_gpu_perf_counter_counter_names(0, { name_entry });
        },
        [](rocpd_processor_t& processor) {
            gpu_perf_counter_sample sample{ 0, k_timestamp,
                                            std::vector<gpu_perf_counter_value>{
                                                { k_counter_id, k_value } } };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    expect_named_pmc_arch(pmc_infos, k_pmc_name.c_str(), "GPU", "GPU", k_pmc_name.c_str(),
                          "", "Shader wavefronts");

    expect_reader_has_tracks({ k_track_name });
}

TEST_F(rocpd_write_read_test, handle_gpu_perf_counter_sample_empty_entries_noop)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({ managed_gpu_agent() }, {},
                                  [](rocpd_processor_t& processor) {
                                      gpu_perf_counter_sample sample{ 0, 12000, {} };
                                      processor.handle(sample);
                                  });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    auto pmc_infos = m_reader->get_all_pmc_info();
    EXPECT_TRUE(pmc_infos.empty());
}

// ---------------------------------------------------------------------------
// handle(cpu_pmc_sample): process-level + per-core PMC inserts
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_cpu_pmc_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    const auto cpu = managed_cpu_agent();
    run_processor_and_open_reader(
        { cpu },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(
                make_agent_pmc_row(agent_type::cpu, 0, "process_physical_memory", "CPU"));
            metadata->add_pmc_info(
                make_agent_pmc_row(agent_type::cpu, 0, "cpu_frequency", "CPU"));
            add_process_scoped_track(metadata, "process_physical_memory");
            add_process_scoped_track(metadata, "cpu_frequency [0] Core [0]");
            add_process_scoped_track(metadata, "cpu_frequency [0] Core [1]");
        },
        [](rocpd_processor_t& processor) {
            rocprofsys::pmc::collectors::cpu::enabled_metrics enabled{};
            enabled.bits.page_rss  = 1;
            enabled.bits.frequency = 1;
            rocprofsys::pmc::collectors::cpu::process_metrics proc{};
            proc.page_rss = static_cast<std::int64_t>(256.5 * 1024.0 * 1024.0);
            std::vector<rocprofsys::pmc::collectors::cpu::per_cpu_metrics> cores{
                { 0, 3200.0f, 0.0 },
                { 1, 3100.0f, 0.0 },
            };
            auto freqs = rocprofsys::pmc::collectors::cpu::serialize_frequencies(cores);
            cpu_pmc_sample sample{ enabled, 0, 13000, proc, std::move(freqs), {} };
            processor.handle(sample);
        });

    const auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 2U);
    expect_named_pmc_arch(pmc_infos, "process_physical_memory", "CPU", "CPU",
                          "process_physical_memory");
    expect_named_pmc_arch(pmc_infos, "cpu_frequency", "CPU", "CPU", "cpu_frequency");

    expect_readback_agent_fields("CPU", "CPU0", "EPYC", "AMD", "EPYC 7763");

    expect_reader_has_tracks({ "process_physical_memory", "cpu_frequency [0] Core [0]",
                               "cpu_frequency [0] Core [1]" });
}

// ---------------------------------------------------------------------------
// handle(kfd_sample): region + pmc_event with args (KFD event pathway)
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_kfd_sample_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    const auto gpu      = managed_gpu_agent();
    const auto kfd_args = get_args_string(
        function_args_t{ { 0U, "std::uint64_t", "address", "0x7f4a00001000" },
                         { 1U, "string", "agent", "5" } });
    run_processor_and_open_reader(
        { gpu },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_pmc_info(make_agent_pmc_row(
                agent_type::gpu, 0, "kfd_page_fault", "GPU", "KFD page fault counter"));
            metadata->add_track(track{ "KFD Events [GPU 0]", THREAD_ID, std::string{} });
        },
        [&](rocpd_processor_t& processor) {
            kfd_sample sample{ THREAD_ID,
                               "KFD_PAGE_FAULT",
                               14000,
                               14500,
                               kfd_args,
                               "kfd",
                               "KFD Events [GPU 0]",
                               R"({"source": "kfd"})",
                               0,
                               static_cast<std::uint8_t>(agent_type::gpu),
                               "kfd_page_fault",
                               1.0,
                               static_cast<std::int64_t>(THREAD_ID) };
            processor.handle(sample);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([this](const auto& detail, const auto& tl_event) {
        EXPECT_EQ(detail.start_timestamp, 14000U);
        EXPECT_EQ(detail.end_timestamp, 14500U);
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
    expect_named_pmc_arch(pmc_infos, "kfd_page_fault", "GPU", "GPU", "kfd_page_fault", "",
                          "KFD page fault counter");

    expect_reader_has_tracks({ "KFD Events [GPU 0]" });
}

// ---------------------------------------------------------------------------
// DB file creation: every handle pathway creates a valid DB
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, multiple_event_types_in_single_db)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    const auto gpu = managed_gpu_agent();
    const auto cpu = managed_cpu_agent();
    run_processor_and_open_reader(
        { gpu, cpu },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "test_kernel");
            metadata->add_track(track{ "Sampling", THREAD_ID, std::string{} });
        },
        [](rocpd_processor_t& processor) {
            region_sample hip_region{ THREAD_ID, "hipLaunchKernel", 1, 0, 1000, 1200, "",
                                      "",        "HIP_API" };
            processor.handle(hip_region);

            kernel_dispatch_sample kds{ 1500, 2000,     THREAD_ID, MANAGED_GPU_HANDLE,
                                        1,    1,        QUEUE_ID,  2,
                                        1,    0,        0,         128,
                                        1,    1,        512,       1,
                                        1,    STREAM_ID };
            processor.handle(kds);

            memory_copy_sample mcs{
                2500,
                3000,
                THREAD_ID,
                MANAGED_GPU_HANDLE,
                MANAGED_CPU_HANDLE,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY),
                static_cast<std::int32_t>(ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE),
                2048,
                3,
                0,
                0,
                0,
                STREAM_ID
            };
            processor.handle(mcs);

            scratch_memory_sample sms{
                3500,
                3600,
                THREAD_ID,
                MANAGED_GPU_HANDLE,
                QUEUE_ID,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY),
                static_cast<std::int32_t>(ROCPROFILER_SCRATCH_MEMORY_ALLOC),
                0,
                32768,
                4,
                0,
                STREAM_ID
            };
            processor.handle(sms);

            backtrace_region_sample bts{ 0,    THREAD_ID,  "Sampling", "bt_func", 4000,
                                         4100, "sampling", "",         "",        "" };
            processor.handle(bts);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    auto counts = m_reader->get_event_counts();

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

    bool saw_hip_region    = false;
    bool saw_bt_region     = false;
    bool saw_kernel        = false;
    bool saw_memory_copy   = false;
    bool saw_scratch_alloc = false;

    for(const auto& tl_event : m_reader->get_events())
    {
        switch(tl_event.unique_identifier.type)
        {
            case profiler_hub::reader_types::event_type_t::region:
            {
                auto detail = m_reader->get_region_details(tl_event);
                ASSERT_TRUE(detail.has_value());
                if(detail->name == "hipLaunchKernel")
                {
                    saw_hip_region = true;
                    EXPECT_EQ(detail->start_timestamp, 1000U);
                    EXPECT_EQ(detail->end_timestamp, 1200U);
                    ASSERT_NE(detail->event, nullptr);
                    EXPECT_EQ(detail->event->event_category, "HIP_API");
                }
                else if(detail->name == "bt_func")
                {
                    saw_bt_region = true;
                    EXPECT_EQ(detail->start_timestamp, 4000U);
                    EXPECT_EQ(detail->end_timestamp, 4100U);
                }
                break;
            }
            case profiler_hub::reader_types::event_type_t::kernel_dispatch:
            {
                saw_kernel  = true;
                auto detail = m_reader->get_kernel_dispatch_details(tl_event);
                ASSERT_TRUE(detail.has_value());
                EXPECT_EQ(detail->name, "test_kernel");
                EXPECT_EQ(detail->start_timestamp, 1500U);
                EXPECT_EQ(detail->end_timestamp, 2000U);
                EXPECT_EQ(detail->workgroup_size_x, 128U);
                EXPECT_EQ(detail->grid_size_x, 512U);
                break;
            }
            case profiler_hub::reader_types::event_type_t::memory_copy:
            {
                saw_memory_copy = true;
                auto detail     = m_reader->get_memory_copy_details(tl_event);
                ASSERT_TRUE(detail.has_value());
                EXPECT_EQ(detail->size, 2048U);
                EXPECT_EQ(detail->start_timestamp, 2500U);
                EXPECT_EQ(detail->end_timestamp, 3000U);
                EXPECT_EQ(detail->name, "MEMORY_COPY_HOST_TO_DEVICE");
                break;
            }
            case profiler_hub::reader_types::event_type_t::memory_allocate:
            {
                saw_scratch_alloc = true;
                auto detail       = m_reader->get_memory_alloc_details(tl_event);
                ASSERT_TRUE(detail.has_value());
                EXPECT_EQ(detail->size, 32768U);
                EXPECT_EQ(detail->type, "ALLOC");
                EXPECT_EQ(detail->level, "SCRATCH");
                EXPECT_EQ(detail->start_timestamp, 3500U);
                EXPECT_EQ(detail->end_timestamp, 3600U);
                break;
            }
            default: break;
        }
    }

    EXPECT_TRUE(saw_hip_region);
    EXPECT_TRUE(saw_bt_region);
    EXPECT_TRUE(saw_kernel);
    EXPECT_TRUE(saw_memory_copy);
    EXPECT_TRUE(saw_scratch_alloc);
    expect_readback_agent_fields("GPU", "gfx90a", "MI210", "AMD", "Instinct MI210");
    expect_readback_agent_fields("CPU", "CPU0", "EPYC", "AMD", "EPYC 7763");
}

// ---------------------------------------------------------------------------
// Region with call_stack: handle(region_sample) sets ev.call_stack
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_region_with_call_stack_pathway)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader({}, {}, [](rocpd_processor_t& processor) {
        region_sample reg{ THREAD_ID, "hsa_signal_wait",           1,  0,        15000,
                           15500,     R"([{"function": "main"}])", "", "HSA_API" };
        processor.handle(reg);
    });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_region([this](const auto& detail, const auto& tl_event) {
        EXPECT_EQ(detail.name, "hsa_signal_wait");
        EXPECT_EQ(detail.start_timestamp, 15000U);
        EXPECT_EQ(detail.end_timestamp, 15500U);
        ASSERT_NE(detail.event, nullptr);
        EXPECT_EQ(detail.event->event_category, "HSA_API");
        EXPECT_FALSE(m_reader->get_call_stack(tl_event).empty());
    });
}

// ---------------------------------------------------------------------------
// Kernel dispatch with full grid: validate all dimension fields
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_kernel_dispatch_full_grid)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            seed_gpu_queue_stream(metadata);
            seed_kernel_symbol(metadata, 1, "matmul_kernel");
        },
        [](rocpd_processor_t& processor) {
            kernel_dispatch_sample kds{ 20000, 25000,    THREAD_ID, MANAGED_GPU_HANDLE,
                                        1,     77,       QUEUE_ID,  10,
                                        5,     512,      16384,     64,
                                        4,     2,        256,       16,
                                        8,     STREAM_ID };
            processor.handle(kds);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_kernel_dispatch([](const auto& detail) {
        EXPECT_EQ(detail.dispatch_id, 77U);
        EXPECT_EQ(detail.start_timestamp, 20000U);
        EXPECT_EQ(detail.end_timestamp, 25000U);
        EXPECT_EQ(detail.workgroup_size_x, 64U);
        EXPECT_EQ(detail.workgroup_size_y, 4U);
        EXPECT_EQ(detail.workgroup_size_z, 2U);
        EXPECT_EQ(detail.grid_size_x, 256U);
        EXPECT_EQ(detail.grid_size_y, 16U);
        EXPECT_EQ(detail.grid_size_z, 8U);
        EXPECT_EQ(detail.name, "matmul_kernel");
    });
}

// ---------------------------------------------------------------------------
// Memory copy with addresses: validate src/dst address fields
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_memory_copy_addresses_persisted)
{
    // Prepare: seed metadata/agents/samples and run rocpd_processor_t (opens reader).
    run_processor_and_open_reader(
        { managed_gpu_agent(), managed_cpu_agent() },
        [](const std::shared_ptr<metadata_registry>& metadata) {
            metadata->add_stream(STREAM_ID);
        },
        [](rocpd_processor_t& processor) {
            memory_copy_sample mcs{
                30000,
                31000,
                THREAD_ID,
                MANAGED_GPU_HANDLE,
                MANAGED_CPU_HANDLE,
                static_cast<std::int32_t>(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY),
                static_cast<std::int32_t>(ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST),
                1024,
                1,
                0,
                0xDEAD0000,
                0xBEEF0000,
                STREAM_ID
            };
            processor.handle(mcs);
        });

    // Validate: profiler_hub::reader_t read-back matches inserted values.
    expect_memory_copy([](const auto& detail) {
        EXPECT_EQ(detail.start_timestamp, 30000U);
        EXPECT_EQ(detail.end_timestamp, 31000U);
        EXPECT_EQ(detail.size, 1024U);
        EXPECT_EQ(detail.name, "MEMORY_COPY_DEVICE_TO_HOST");
        ASSERT_TRUE(detail.dst_address.has_value());
        EXPECT_EQ(detail.dst_address.value(), 0xDEAD0000U);
        ASSERT_TRUE(detail.src_address.has_value());
        EXPECT_EQ(detail.src_address.value(), 0xBEEF0000U);
    });
}
