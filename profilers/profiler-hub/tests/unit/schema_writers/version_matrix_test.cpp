// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Same shape as the kdlib example in https://github.com/adjordje-amd/libray-tests:
// one database per schema version, the same unversioned structs, a minimum row and
// a full row. A field this schema cannot store is rejected. The example drops that
// field instead.

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace profiler_hub
{
void
PrintTo(const version_t& version, std::ostream* os)
{
    *os << version.to_string();
}
}  // namespace profiler_hub

namespace
{

using namespace profiler_hub;

constexpr size_t k_node        = 1;
constexpr size_t k_pid         = 100;
constexpr size_t k_tid         = 200;
constexpr size_t k_queue       = 10;
constexpr size_t k_stream      = 20;
constexpr size_t k_code_object = 30;
constexpr size_t k_kernel      = 40;
constexpr size_t k_start       = 1000;
constexpr size_t k_end         = 2000;

writer_types::agent_unique_id_t
gpu_agent()
{
    writer_types::agent_unique_id_t id;
    id.agent_type = "GPU";
    id.type_index = 0;
    return id;
}

writer_types::agent_unique_id_t
nic_agent()
{
    writer_types::agent_unique_id_t id;
    id.agent_type = "NIC";
    id.type_index = 0;
    return id;
}

writer_types::event_data_t
bare_event()
{
    return {};
}

void
register_minimum_catalog(writer_t& writer)
{
    writer.register_node_info({ k_node, 42, "machine" });

    writer_types::process_info_t process;
    process.pid     = k_pid;
    process.node_id = k_node;
    writer.register_process_info(process);

    writer_types::thread_info_t thread;
    thread.thread_id  = k_tid;
    thread.node_id    = k_node;
    thread.process_id = k_pid;
    writer.register_thread_info(thread);

    writer_types::agent_info_t agent;
    agent.unique_id  = gpu_agent();
    agent.node_id    = k_node;
    agent.process_id = k_pid;
    writer.register_agent_info(agent);

    writer_types::queue_info_t queue;
    queue.queue_id   = k_queue;
    queue.node_id    = k_node;
    queue.process_id = k_pid;
    writer.register_queue_info(queue);

    writer_types::stream_info_t stream;
    stream.stream_id  = k_stream;
    stream.node_id    = k_node;
    stream.process_id = k_pid;
    writer.register_stream_info(stream);

    writer_types::code_object_info_t code_object;
    code_object.id         = k_code_object;
    code_object.node_id    = k_node;
    code_object.process_id = k_pid;
    writer.register_code_object_info(code_object);

    writer_types::kernel_symbol_info_t kernel;
    kernel.id          = k_kernel;
    kernel.node_id     = k_node;
    kernel.process_id  = k_pid;
    kernel.code_obj_id = k_code_object;
    writer.register_kernel_symbol_info(kernel);

    writer_types::track_info_t track;
    track.name    = "pmc-track";
    track.node_id = k_node;
    writer.register_track_info(track);

    writer_types::pmc_info_t pmc;
    pmc.unique_id.name = "ctr";
    pmc.symbol         = "ctr-symbol";
    pmc.node_id        = k_node;
    pmc.process_id     = k_pid;
    writer.register_pmc_info(pmc);
}

writer_types::trace_environment_t
base_trace()
{
    writer_types::trace_environment_t trace;
    trace.node_id    = k_node;
    trace.process_id = k_pid;
    return trace;
}

writer_types::trace_environment_t
kernel_trace()
{
    auto trace      = base_trace();
    trace.thread_id = k_tid;
    trace.agent_id  = gpu_agent();
    trace.queue_id  = k_queue;
    trace.stream_id = k_stream;
    return trace;
}

class schema_version_matrix_test : public ::testing::TestWithParam<version_t>
{
protected:
    void SetUp() override
    {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        auto        name = std::string{ info->name() };
        for(char& character : name)
        {
            if(character == '/') character = '_';
        }
        m_db_path = (std::filesystem::temp_directory_path() /
                     ("schema_matrix_" + version().to_string() + "_" + name + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
        m_uuid = "matrix" + std::to_string(version().patch);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] version_t version() const { return GetParam(); }

    [[nodiscard]] bool stores_from(version_t introduced) const
    {
        return !(version() < introduced);
    }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer() const
    {
        return std::make_unique<writer_t>(
            std::make_unique<storage_t>(m_db_path, m_uuid, version()));
    }

    template <typename Fn>
    void expect_field(bool stored, Fn&& write) const
    {
        if(stored)
        {
            EXPECT_NO_THROW(std::forward<Fn>(write)());
            return;
        }

        try
        {
            std::forward<Fn>(write)();
            FAIL() << "schema " << version().to_string()
                   << " stored a field it does not own";
        } catch(const std::runtime_error& error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("not stored"), std::string::npos) << message;
            EXPECT_NE(message.find(version().to_string()), std::string::npos) << message;
        }
    }

    std::string m_db_path;
    std::string m_uuid;
};

TEST_P(schema_version_matrix_test, minimum_row_is_stored_by_every_schema)
{
    auto writer = make_writer();
    register_minimum_catalog(*writer);

    writer_types::region_data_t region;
    region.name            = "region";
    region.start_timestamp = k_start;
    region.end_timestamp   = k_end;
    region.event           = bare_event();
    auto region_trace      = kernel_trace();
    EXPECT_NO_THROW(writer->insert_region_data(region, region_trace));

    writer_types::kernel_dispatch_data_t kernel;
    kernel.kernel_symbol_id = k_kernel;
    kernel.dispatch_id      = 1;
    kernel.start_timestamp  = k_start;
    kernel.end_timestamp    = k_end;
    kernel.event            = bare_event();
    EXPECT_NO_THROW(writer->insert_kernel_dispatch_data(kernel, kernel_trace()));

    writer_types::memory_copy_data_t copy;
    copy.name            = "copy";
    copy.start_timestamp = k_start;
    copy.end_timestamp   = k_end;
    copy.size            = 64;
    copy.event           = bare_event();
    EXPECT_NO_THROW(writer->insert_memory_copy_data(copy, base_trace()));

    writer_types::memory_alloc_data_t alloc;
    alloc.start_timestamp = k_start;
    alloc.end_timestamp   = k_end;
    alloc.size            = 128;
    alloc.event           = bare_event();
    EXPECT_NO_THROW(writer->insert_memory_alloc_data(alloc, base_trace()));

    writer_types::pmc_event_data_t pmc_event;
    pmc_event.value = 1.5;
    writer_types::pmc_info_unique_id_t pmc_id;
    pmc_id.name = "ctr";
    EXPECT_NO_THROW(writer->insert_pmc_event_data(pmc_event, pmc_id));

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto events = reader->get_events();

    auto count = [&](reader_types::event_type_t type) {
        size_t found = 0;
        for(const auto& event : events)
        {
            if(event.unique_identifier.type == type && event.start_timestamp == k_start)
            {
                ++found;
            }
        }
        return found;
    };
    EXPECT_EQ(count(reader_types::event_type_t::region), 1);
    EXPECT_EQ(count(reader_types::event_type_t::kernel_dispatch), 1);
    EXPECT_EQ(count(reader_types::event_type_t::memory_copy), 1);
    EXPECT_EQ(count(reader_types::event_type_t::memory_allocate), 1);
    EXPECT_EQ(reader->get_all_agents().size(), 1);
    EXPECT_EQ(reader->get_all_pmc_info().size(), 1);
}

TEST_P(schema_version_matrix_test, superset_field_is_stored_only_by_its_schema)
{
    auto writer = make_writer();
    register_minimum_catalog(*writer);

    // 3.0.1: NIC on agent.type and pmc_info.target_arch.
    writer_types::agent_info_t nic;
    nic.unique_id  = nic_agent();
    nic.node_id    = k_node;
    nic.process_id = k_pid;
    expect_field(stores_from(version_t{ 3, 0, 1 }),
                 [&] { writer->register_agent_info(nic); });

    writer_types::pmc_info_t nic_pmc;
    nic_pmc.unique_id.name = "ctr-nic";
    nic_pmc.symbol         = "ctr-nic-symbol";
    nic_pmc.target_arch    = "NIC";
    nic_pmc.node_id        = k_node;
    nic_pmc.process_id     = k_pid;
    expect_field(stores_from(version_t{ 3, 0, 1 }),
                 [&] { writer->register_pmc_info(nic_pmc); });

    // 3.0.2: graph ids, and the graph-launch table.
    writer_types::kernel_dispatch_data_t kernel;
    kernel.kernel_symbol_id = k_kernel;
    kernel.dispatch_id      = 2;
    kernel.start_timestamp  = k_start;
    kernel.end_timestamp    = k_end;
    kernel.graph_exec_id    = 3;
    kernel.graph_node_id    = 4;
    expect_field(stores_from(version_t{ 3, 0, 2 }),
                 [&] { writer->insert_kernel_dispatch_data(kernel, kernel_trace()); });

    writer_types::memory_copy_data_t copy;
    copy.name            = "copy-graph";
    copy.start_timestamp = k_start;
    copy.end_timestamp   = k_end;
    copy.size            = 32;
    copy.graph_exec_id   = 3;
    copy.graph_node_id   = 4;
    expect_field(stores_from(version_t{ 3, 0, 2 }),
                 [&] { writer->insert_memory_copy_data(copy, base_trace()); });

    writer_types::graph_launch_data_t graph;
    graph.start_timestamp       = k_start;
    graph.end_timestamp         = k_end;
    graph.graph_exec_id         = 3;
    graph.kernel_dispatch_count = 1;
    expect_field(stores_from(version_t{ 3, 0, 2 }),
                 [&] { writer->insert_graph_launch_data(graph, base_trace()); });

    // 3.0.3: spm_support, and sample_id on the PMC event.
    writer_types::pmc_info_t spm;
    spm.unique_id.name = "ctr-spm";
    spm.symbol         = "ctr-spm-symbol";
    spm.spm_support    = 1;
    spm.node_id        = k_node;
    spm.process_id     = k_pid;
    expect_field(stores_from(version_t{ 3, 0, 3 }),
                 [&] { writer->register_pmc_info(spm); });

    writer_types::pmc_event_data_t sampled;
    sampled.value                = 2.0;
    sampled.sample_id            = 1;
    sampled.sample.timestamp     = k_start;
    sampled.sample.track.name    = "pmc-track";
    sampled.sample.track.node_id = k_node;
    writer_types::pmc_info_unique_id_t pmc_id;
    pmc_id.name = "ctr";
    expect_field(stores_from(version_t{ 3, 0, 3 }),
                 [&] { writer->insert_pmc_event_data(sampled, pmc_id); });

    // 3.0.3 and 3.0.4 store the PMC event location. 3.0.5 moves it to pmc_info.
    const bool event_location =
        stores_from(version_t{ 3, 0, 3 }) && version() < version_t{ 3, 0, 5 };
    writer_types::pmc_event_data_t located;
    located.value = 3.0;
    located.xcc   = 1;
    expect_field(event_location, [&] { writer->insert_pmc_event_data(located, pmc_id); });

    // 3.0.4: HIP event.
    writer_types::hip_event_data_t hip;
    hip.name             = "hipEventRecord";
    hip.start_timestamp  = k_start;
    hip.end_timestamp    = k_end;
    hip.hip_event_handle = 9;
    expect_field(stores_from(version_t{ 3, 0, 4 }),
                 [&] { writer->insert_hip_event_data(hip, base_trace()); });

    // 3.0.5: location columns on pmc_info.
    writer_types::pmc_info_t location;
    location.unique_id.name          = "ctr-location";
    location.unique_id.xcc           = 1;
    location.unique_id.shader_engine = 2;
    location.unique_id.shader_array  = 3;
    location.unique_id.wgp           = 4;
    location.unique_id.instance      = 5;
    location.symbol                  = "ctr-location-symbol";
    location.node_id                 = k_node;
    location.process_id              = k_pid;
    expect_field(stores_from(version_t{ 3, 0, 5 }),
                 [&] { writer->register_pmc_info(location); });
}

std::string
schema_name(const ::testing::TestParamInfo<version_t>& info)
{
    return "schema_" + std::to_string(info.param.major) + "_" +
           std::to_string(info.param.minor) + "_" + std::to_string(info.param.patch);
}

INSTANTIATE_TEST_SUITE_P(schemas_3_0_0_through_3_0_5,
                         schema_version_matrix_test,
                         ::testing::Values(version_t{ 3, 0, 0 },
                                           version_t{ 3, 0, 1 },
                                           version_t{ 3, 0, 2 },
                                           version_t{ 3, 0, 3 },
                                           version_t{ 3, 0, 4 },
                                           version_t{ 3, 0, 5 }),
                         schema_name);

}  // namespace
