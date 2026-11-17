// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Schema 3.0.5 writer. One folder per schema writer; this is the first.
// The database holds two rows of every table the 3.0.5 writer stores: the
// required columns only, then every column added from schema 3.0.0 through 3.0.5.

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{

using namespace profiler_hub;

constexpr version_t k_schema{ 3, 0, 5 };

constexpr size_t k_min_node         = 1;
constexpr size_t k_full_node        = 2;
constexpr size_t k_min_pid          = 100;
constexpr size_t k_full_pid         = 101;
constexpr size_t k_min_tid          = 200;
constexpr size_t k_full_tid         = 201;
constexpr size_t k_min_queue        = 10;
constexpr size_t k_full_queue       = 11;
constexpr size_t k_min_stream       = 20;
constexpr size_t k_full_stream      = 21;
constexpr size_t k_min_code_object  = 30;
constexpr size_t k_full_code_object = 31;
constexpr size_t k_min_kernel       = 40;
constexpr size_t k_full_kernel      = 41;

constexpr uint64_t k_min_start  = 1000;
constexpr uint64_t k_min_end    = 2000;
constexpr uint64_t k_full_start = 5000;
constexpr uint64_t k_full_end   = 6000;

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
    return writer_types::event_data_t{};
}

writer_types::event_data_t
full_event()
{
    writer_types::event_data_t event;
    event.stack_id        = 4;
    event.parent_stack_id = 3;
    event.correlation_id  = 9;
    event.event_category  = "HIP_API";
    event.extdata         = "{\"event\":1}";

    shared_types::program_counter_info_t pc;
    pc.function    = "launch";
    pc.filename    = "launch.cpp";
    pc.line_number = 12;

    shared_types::stack_frame_t frame;
    frame.program_counter = pc;
    event.call_stack.push_back(frame);

    shared_types::line_info_entry_t line;
    line.program_counter = pc;
    event.line_info_list.push_back(line);
    return event;
}

writer_types::trace_environment_t
min_trace()
{
    writer_types::trace_environment_t trace;
    trace.node_id    = k_min_node;
    trace.process_id = k_min_pid;
    return trace;
}

writer_types::trace_environment_t
kernel_min_trace()
{
    auto trace      = min_trace();
    trace.thread_id = k_min_tid;
    trace.agent_id  = gpu_agent();
    trace.queue_id  = k_min_queue;
    trace.stream_id = k_min_stream;
    return trace;
}

class schema_3_0_5_writer_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "schema_3_0_5_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer() const
    {
        return std::make_unique<writer_t>(
            std::make_unique<storage_t>(m_db_path, m_uuid, k_schema));
    }

    [[nodiscard]] std::unique_ptr<reader_t> open_reader() const
    {
        return std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    // Required columns for the info tables every data row refers to.
    static void register_minimum_catalog(writer_t& writer)
    {
        writer.register_node_info({ k_min_node, 42, "machine-min" });

        writer_types::process_info_t process;
        process.pid     = k_min_pid;
        process.node_id = k_min_node;
        writer.register_process_info(process);

        writer_types::thread_info_t thread;
        thread.thread_id  = k_min_tid;
        thread.node_id    = k_min_node;
        thread.process_id = k_min_pid;
        writer.register_thread_info(thread);

        writer_types::agent_info_t agent;
        agent.unique_id  = gpu_agent();
        agent.node_id    = k_min_node;
        agent.process_id = k_min_pid;
        writer.register_agent_info(agent);

        writer_types::queue_info_t queue;
        queue.queue_id   = k_min_queue;
        queue.node_id    = k_min_node;
        queue.process_id = k_min_pid;
        writer.register_queue_info(queue);

        writer_types::stream_info_t stream;
        stream.stream_id  = k_min_stream;
        stream.node_id    = k_min_node;
        stream.process_id = k_min_pid;
        writer.register_stream_info(stream);

        writer_types::code_object_info_t code_object;
        code_object.id         = k_min_code_object;
        code_object.node_id    = k_min_node;
        code_object.process_id = k_min_pid;
        writer.register_code_object_info(code_object);

        writer_types::kernel_symbol_info_t kernel;
        kernel.id          = k_min_kernel;
        kernel.node_id     = k_min_node;
        kernel.process_id  = k_min_pid;
        kernel.code_obj_id = k_min_code_object;
        writer.register_kernel_symbol_info(kernel);

        writer_types::track_info_t track;
        track.node_id = k_min_node;
        writer.register_track_info(track);

        writer_types::pmc_info_t pmc;
        pmc.unique_id.name = "ctr-min";
        pmc.symbol         = "ctr-min-symbol";
        pmc.node_id        = k_min_node;
        pmc.process_id     = k_min_pid;
        writer.register_pmc_info(pmc);

        writer.register_string("registered-symbol");
    }

    std::string m_db_path;
    std::string m_uuid = "schema305";
};

size_t
count_events(const reader_types::timeline_event_list_t& events,
             reader_types::event_type_t                 type,
             reader_types::timestamp_ns_t               start)
{
    size_t count = 0;
    for(const auto& event : events)
    {
        if(event.unique_identifier.type == type && event.start_timestamp == start)
        {
            ++count;
        }
    }
    return count;
}

TEST_F(schema_3_0_5_writer_test, stores_each_table_from_minimum_row_to_full_row)
{
    auto writer = make_writer();
    register_minimum_catalog(*writer);

    writer_types::region_data_t min_region;
    min_region.name            = "region-min";
    min_region.start_timestamp = k_min_start;
    min_region.end_timestamp   = k_min_end;
    min_region.event           = bare_event();
    auto min_region_trace      = kernel_min_trace();
    writer->insert_region_data(min_region, min_region_trace);

    writer_types::pmc_event_data_t min_pmc_event;
    min_pmc_event.value = 1.5;
    writer_types::pmc_info_unique_id_t min_pmc_id;
    min_pmc_id.name = "ctr-min";
    writer->insert_pmc_event_data(min_pmc_event, min_pmc_id);

    writer_types::kernel_dispatch_data_t min_kernel;
    min_kernel.kernel_symbol_id = k_min_kernel;
    min_kernel.dispatch_id      = 1;
    min_kernel.start_timestamp  = k_min_start;
    min_kernel.end_timestamp    = k_min_end;
    min_kernel.event            = bare_event();
    writer->insert_kernel_dispatch_data(min_kernel, kernel_min_trace());

    writer_types::memory_copy_data_t min_copy;
    min_copy.name            = "copy-min";
    min_copy.start_timestamp = k_min_start;
    min_copy.end_timestamp   = k_min_end;
    min_copy.size            = 64;
    min_copy.event           = bare_event();
    writer->insert_memory_copy_data(min_copy, min_trace());

    writer_types::memory_alloc_data_t min_alloc;
    min_alloc.start_timestamp = k_min_start;
    min_alloc.end_timestamp   = k_min_end;
    min_alloc.size            = 128;
    min_alloc.event           = bare_event();
    writer->insert_memory_alloc_data(min_alloc, min_trace());

    writer_types::graph_launch_data_t min_graph;
    min_graph.start_timestamp       = k_min_start;
    min_graph.end_timestamp         = k_min_end;
    min_graph.graph_exec_id         = 1;
    min_graph.kernel_dispatch_count = 0;
    writer->insert_graph_launch_data(min_graph, min_trace());

    writer_types::hip_event_data_t min_hip;
    min_hip.name             = "hip-min";
    min_hip.start_timestamp  = k_min_start;
    min_hip.end_timestamp    = k_min_end;
    min_hip.hip_event_handle = 7;
    writer->insert_hip_event_data(min_hip, min_trace());

    // Full rows. Columns added after 3.0.0: NIC (3.0.1), graph ids and
    // rocpd_graph_launch (3.0.2), spm_support and pmc sample_id (3.0.3),
    // rocpd_hip_event (3.0.4), PMC location on pmc_info (3.0.5).
    writer_types::node_info_t full_node_info;
    full_node_info.node_id       = k_full_node;
    full_node_info.hash          = 43;
    full_node_info.machine_id    = "machine-full";
    full_node_info.system_name   = "Linux";
    full_node_info.hostname      = "host-full";
    full_node_info.release       = "6.8";
    full_node_info.version       = "#1";
    full_node_info.hardware_name = "x86_64";
    full_node_info.domain_name   = "example.com";
    writer->register_node_info(full_node_info);

    writer_types::process_info_t full_process;
    full_process.ppid        = 1;
    full_process.pid         = k_full_pid;
    full_process.init        = 10;
    full_process.fini        = 20;
    full_process.start       = 30;
    full_process.end         = 40;
    full_process.command     = "/usr/bin/app";
    full_process.environment = "{\"PATH\":\"/usr/bin\"}";
    full_process.extdata     = "{\"process\":1}";
    full_process.node_id     = k_full_node;
    writer->register_process_info(full_process);

    writer_types::thread_info_t full_thread;
    full_thread.parent_process_id = k_full_pid;
    full_thread.thread_id         = k_full_tid;
    full_thread.name              = "worker";
    full_thread.start             = 30;
    full_thread.end               = 40;
    full_thread.extdata           = "{\"thread\":1}";
    full_thread.node_id           = k_full_node;
    full_thread.process_id        = k_full_pid;
    writer->register_thread_info(full_thread);

    writer_types::agent_info_t full_agent;
    full_agent.unique_id      = nic_agent();
    full_agent.absolute_index = 2;
    full_agent.logical_index  = 1;
    full_agent.uuid           = 99;
    full_agent.name           = "nic0";
    full_agent.model_name     = "pollara";
    full_agent.vendor_name    = "AMD";
    full_agent.product_name   = "NIC";
    full_agent.user_name      = "user";
    full_agent.extdata        = "{\"agent\":1}";
    full_agent.node_id        = k_full_node;
    full_agent.process_id     = k_full_pid;
    writer->register_agent_info(full_agent);

    writer_types::queue_info_t full_queue;
    full_queue.queue_id   = k_full_queue;
    full_queue.name       = "queue-full";
    full_queue.extdata    = "{\"queue\":1}";
    full_queue.node_id    = k_full_node;
    full_queue.process_id = k_full_pid;
    writer->register_queue_info(full_queue);

    writer_types::stream_info_t full_stream;
    full_stream.stream_id  = k_full_stream;
    full_stream.name       = "stream-full";
    full_stream.extdata    = "{\"stream\":1}";
    full_stream.node_id    = k_full_node;
    full_stream.process_id = k_full_pid;
    writer->register_stream_info(full_stream);

    writer_types::code_object_info_t full_code;
    full_code.id           = k_full_code_object;
    full_code.uri          = "file://kernel.co";
    full_code.load_base    = 0x1000;
    full_code.load_size    = 0x200;
    full_code.load_delta   = 0x10;
    full_code.storage_type = "FILE";
    full_code.extdata      = "{\"code\":1}";
    full_code.node_id      = k_full_node;
    full_code.process_id   = k_full_pid;
    full_code.agent_id     = nic_agent();
    writer->register_code_object_info(full_code);

    writer_types::kernel_symbol_info_t full_symbol;
    full_symbol.id                        = k_full_kernel;
    full_symbol.name                      = "vector_add";
    full_symbol.display_name              = "vectorAdd";
    full_symbol.kernel_object             = 77;
    full_symbol.kernarg_segment_size      = 32;
    full_symbol.kernarg_segment_alignment = 8;
    full_symbol.group_segment_size        = 256;
    full_symbol.private_segment_size      = 64;
    full_symbol.sgpr_count                = 16;
    full_symbol.arch_vgpr_count           = 8;
    full_symbol.accum_vgpr_count          = 4;
    full_symbol.extdata                   = "{\"kernel\":1}";
    full_symbol.node_id                   = k_full_node;
    full_symbol.process_id                = k_full_pid;
    full_symbol.code_obj_id               = k_full_code_object;
    writer->register_kernel_symbol_info(full_symbol);

    writer_types::track_info_t full_track;
    full_track.name       = "kernel-track";
    full_track.extdata    = "{\"track\":1}";
    full_track.node_id    = k_full_node;
    full_track.process_id = k_full_pid;
    full_track.thread_id  = k_full_tid;
    writer->register_track_info(full_track);

    writer_types::pmc_info_t full_pmc;
    full_pmc.unique_id.name          = "ctr-full";
    full_pmc.unique_id.agent_id      = nic_agent();
    full_pmc.unique_id.xcc           = 1;
    full_pmc.unique_id.shader_engine = 2;
    full_pmc.unique_id.shader_array  = 3;
    full_pmc.unique_id.wgp           = 4;
    full_pmc.unique_id.instance      = 5;
    full_pmc.target_arch             = "NIC";
    full_pmc.event_code              = 12;
    full_pmc.instance_id             = 6;
    full_pmc.symbol                  = "ctr-full-symbol";
    full_pmc.description             = "full counter";
    full_pmc.long_description        = "full counter long";
    full_pmc.component               = "SQ";
    full_pmc.units                   = "cycles";
    full_pmc.value_type              = "ABS";
    full_pmc.block                   = "SQ";
    full_pmc.expression              = "A+B";
    full_pmc.is_constant             = 1;
    full_pmc.is_derived              = 0;
    full_pmc.spm_support             = 1;
    full_pmc.extdata                 = "{\"pmc\":1}";
    full_pmc.node_id                 = k_full_node;
    full_pmc.process_id              = k_full_pid;
    writer->register_pmc_info(full_pmc);

    writer_types::trace_environment_t full_trace;
    full_trace.node_id    = k_full_node;
    full_trace.process_id = k_full_pid;
    full_trace.thread_id  = k_full_tid;
    full_trace.agent_id   = nic_agent();
    full_trace.queue_id   = k_full_queue;
    full_trace.stream_id  = k_full_stream;
    full_trace.track_name = "kernel-track";

    const auto event = full_event();

    writer_types::region_data_t full_region;
    full_region.event           = event;
    full_region.start_timestamp = k_full_start;
    full_region.end_timestamp   = k_full_end;
    full_region.name            = "region-full";
    full_region.extdata         = "{\"region\":1}";
    writer_types::arg_data_t arg;
    arg.position = 0;
    arg.type     = "int";
    arg.name     = "n";
    arg.value    = "4";
    arg.extdata  = "{\"arg\":1}";
    full_region.args.push_back(arg);
    writer->insert_region_data(full_region, full_trace);

    writer_types::pmc_event_data_t full_pmc_event;
    full_pmc_event.event            = event;
    full_pmc_event.value            = 9.5;
    full_pmc_event.sample_id        = 1;
    full_pmc_event.extdata          = "{\"pmc_event\":1}";
    full_pmc_event.sample.timestamp = k_full_start;
    full_pmc_event.sample.track     = full_track;
    full_pmc_event.sample.extdata   = "{\"sample\":1}";
    writer->insert_pmc_event_data(full_pmc_event, full_pmc.unique_id);

    writer_types::kernel_dispatch_data_t full_kernel_row;
    full_kernel_row.event                = event;
    full_kernel_row.dispatch_id          = 8;
    full_kernel_row.start_timestamp      = k_full_start;
    full_kernel_row.end_timestamp        = k_full_end;
    full_kernel_row.kernel_symbol_id     = k_full_kernel;
    full_kernel_row.code_object_id       = k_full_code_object;
    full_kernel_row.private_segment_size = 64;
    full_kernel_row.group_segment_size   = 256;
    full_kernel_row.workgroup_size_x     = 16;
    full_kernel_row.workgroup_size_y     = 2;
    full_kernel_row.workgroup_size_z     = 1;
    full_kernel_row.grid_size_x          = 32;
    full_kernel_row.grid_size_y          = 2;
    full_kernel_row.grid_size_z          = 1;
    full_kernel_row.graph_exec_id        = 3;
    full_kernel_row.graph_node_id        = 4;
    full_kernel_row.name                 = "kernel-full";
    full_kernel_row.extdata              = "{\"dispatch\":1}";
    writer->insert_kernel_dispatch_data(full_kernel_row, full_trace);

    writer_types::memory_copy_data_t full_copy;
    full_copy.event           = event;
    full_copy.start_timestamp = k_full_start;
    full_copy.end_timestamp   = k_full_end;
    full_copy.dst_agent_id    = nic_agent();
    full_copy.dst_address     = 0x2000;
    full_copy.src_agent_id    = nic_agent();
    full_copy.src_address     = 0x1000;
    full_copy.size            = 4096;
    full_copy.graph_exec_id   = 3;
    full_copy.graph_node_id   = 4;
    full_copy.name            = "copy-full";
    full_copy.region_name     = "copy-region";
    full_copy.extdata         = "{\"copy\":1}";
    writer->insert_memory_copy_data(full_copy, full_trace);

    writer_types::memory_alloc_data_t full_alloc;
    full_alloc.event           = event;
    full_alloc.type            = "ALLOC";
    full_alloc.level           = "REAL";
    full_alloc.start_timestamp = k_full_start;
    full_alloc.end_timestamp   = k_full_end;
    full_alloc.address         = 0x3000;
    full_alloc.size            = 8192;
    full_alloc.extdata         = "{\"alloc\":1}";
    writer->insert_memory_alloc_data(full_alloc, full_trace);

    writer_types::graph_launch_data_t full_graph;
    full_graph.start_timestamp       = k_full_start;
    full_graph.end_timestamp         = k_full_end;
    full_graph.graph_exec_id         = 3;
    full_graph.kernel_dispatch_count = 2;
    full_graph.event                 = event;
    full_graph.extdata               = "{\"graph\":1}";
    writer->insert_graph_launch_data(full_graph, full_trace);

    writer_types::hip_event_data_t full_hip;
    full_hip.start_timestamp  = k_full_start;
    full_hip.end_timestamp    = k_full_end;
    full_hip.name             = "hipEventRecord";
    full_hip.hip_event_handle = 9;
    full_hip.source_queue_id  = 15;
    full_hip.event            = event;
    full_hip.extdata          = "{\"hip\":1}";
    writer->insert_hip_event_data(full_hip, full_trace);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    ASSERT_TRUE(std::filesystem::exists(m_db_path));
    auto reader = open_reader();

    const auto nodes = reader->get_all_nodes();
    ASSERT_EQ(nodes.size(), 2);
    const reader_types::node_info_t* min_node  = nullptr;
    const reader_types::node_info_t* full_node = nullptr;
    for(const auto& node : nodes)
    {
        if(node->node_id == k_min_node) min_node = node.get();
        if(node->node_id == k_full_node) full_node = node.get();
    }
    ASSERT_NE(min_node, nullptr);
    EXPECT_EQ(min_node->hash, 42);
    EXPECT_EQ(min_node->machine_id, "machine-min");
    EXPECT_TRUE(min_node->system_name.empty());
    ASSERT_NE(full_node, nullptr);
    EXPECT_EQ(full_node->machine_id, "machine-full");
    EXPECT_EQ(full_node->system_name, "Linux");
    EXPECT_EQ(full_node->hostname, "host-full");
    EXPECT_EQ(full_node->release, "6.8");
    EXPECT_EQ(full_node->version, "#1");
    EXPECT_EQ(full_node->hardware_name, "x86_64");
    EXPECT_EQ(full_node->domain_name, "example.com");

    const auto processes = reader->get_all_processes();
    ASSERT_EQ(processes.size(), 2);
    const reader_types::process_info_t* full_process_row = nullptr;
    for(const auto& process : processes)
    {
        if(process->pid == k_full_pid) full_process_row = process.get();
    }
    ASSERT_NE(full_process_row, nullptr);
    EXPECT_EQ(full_process_row->command, "/usr/bin/app");
    ASSERT_NE(full_process_row->node_info, nullptr);
    EXPECT_EQ(full_process_row->node_info->node_id, k_full_node);

    const auto agents = reader->get_all_agents();
    ASSERT_EQ(agents.size(), 2);
    bool saw_gpu = false;
    bool saw_nic = false;
    for(const auto& agent : agents)
    {
        if(agent->agent_type == "GPU") saw_gpu = true;
        if(agent->agent_type == "NIC")
        {
            saw_nic = true;
            EXPECT_EQ(agent->name, "nic0");
            EXPECT_EQ(agent->vendor_name, "AMD");
        }
    }
    EXPECT_TRUE(saw_gpu);
    EXPECT_TRUE(saw_nic);

    const auto pmc_rows = reader->get_all_pmc_info();
    ASSERT_EQ(pmc_rows.size(), 2);
    const reader_types::pmc_info_t* full_pmc_row = nullptr;
    for(const auto& pmc : pmc_rows)
    {
        if(pmc->name == "ctr-full") full_pmc_row = pmc.get();
    }
    ASSERT_NE(full_pmc_row, nullptr);
    EXPECT_EQ(full_pmc_row->symbol, "ctr-full-symbol");
    EXPECT_EQ(full_pmc_row->target_arch, "NIC");
    EXPECT_EQ(full_pmc_row->value_type, "ABS");
    EXPECT_EQ(full_pmc_row->description, "full counter");
    ASSERT_NE(full_pmc_row->agent_info, nullptr);
    EXPECT_EQ(full_pmc_row->agent_info->agent_type, "NIC");

    const auto events = reader->get_events();
    EXPECT_EQ(count_events(events, reader_types::event_type_t::region, k_min_start), 1);
    EXPECT_EQ(count_events(events, reader_types::event_type_t::region, k_full_start), 1);
    EXPECT_EQ(
        count_events(events, reader_types::event_type_t::kernel_dispatch, k_min_start),
        1);
    EXPECT_EQ(
        count_events(events, reader_types::event_type_t::kernel_dispatch, k_full_start),
        1);
    EXPECT_EQ(count_events(events, reader_types::event_type_t::memory_copy, k_min_start),
              1);
    EXPECT_EQ(count_events(events, reader_types::event_type_t::memory_copy, k_full_start),
              1);
    EXPECT_EQ(
        count_events(events, reader_types::event_type_t::memory_allocate, k_min_start),
        1);
    EXPECT_EQ(
        count_events(events, reader_types::event_type_t::memory_allocate, k_full_start),
        1);

    const reader_types::timeline_event_t* full_region_event = nullptr;
    for(const auto& event_row : events)
    {
        if(event_row.display_name == "region-full") full_region_event = &event_row;
    }
    ASSERT_NE(full_region_event, nullptr);
    EXPECT_EQ(full_region_event->category, "HIP_API");
    const auto call_stack = reader->get_call_stack(*full_region_event);
    ASSERT_FALSE(call_stack.empty());
}

TEST_F(schema_3_0_5_writer_test, missing_minimum_value_is_rejected)
{
    auto writer = make_writer();

    EXPECT_THROW(writer->register_string(""), std::runtime_error);

    writer_types::process_info_t missing_node;
    missing_node.pid     = k_min_pid;
    missing_node.node_id = 999;
    EXPECT_THROW(writer->register_process_info(missing_node), std::runtime_error);

    register_minimum_catalog(*writer);

    writer_types::agent_info_t missing_process;
    missing_process.unique_id  = nic_agent();
    missing_process.node_id    = k_min_node;
    missing_process.process_id = 999;
    EXPECT_THROW(writer->register_agent_info(missing_process), std::runtime_error);

    writer_types::thread_info_t missing_thread_node;
    missing_thread_node.thread_id  = 777;
    missing_thread_node.node_id    = 999;
    missing_thread_node.process_id = k_min_pid;
    EXPECT_THROW(writer->register_thread_info(missing_thread_node), std::runtime_error);

    writer_types::queue_info_t missing_queue_process;
    missing_queue_process.queue_id   = 777;
    missing_queue_process.node_id    = k_min_node;
    missing_queue_process.process_id = 999;
    EXPECT_THROW(writer->register_queue_info(missing_queue_process), std::runtime_error);

    writer_types::stream_info_t missing_stream_node;
    missing_stream_node.stream_id  = 777;
    missing_stream_node.node_id    = 999;
    missing_stream_node.process_id = k_min_pid;
    EXPECT_THROW(writer->register_stream_info(missing_stream_node), std::runtime_error);

    writer_types::code_object_info_t missing_code_process;
    missing_code_process.id         = 777;
    missing_code_process.node_id    = k_min_node;
    missing_code_process.process_id = 999;
    EXPECT_THROW(writer->register_code_object_info(missing_code_process),
                 std::runtime_error);

    writer_types::kernel_symbol_info_t missing_code_object;
    missing_code_object.id          = 777;
    missing_code_object.node_id     = k_min_node;
    missing_code_object.process_id  = k_min_pid;
    missing_code_object.code_obj_id = 999;
    EXPECT_THROW(writer->register_kernel_symbol_info(missing_code_object),
                 std::runtime_error);

    writer_types::track_info_t missing_track_node;
    missing_track_node.node_id = 999;
    EXPECT_THROW(writer->register_track_info(missing_track_node), std::runtime_error);

    writer_types::pmc_info_t missing_pmc_process;
    missing_pmc_process.unique_id.name = "missing";
    missing_pmc_process.symbol         = "missing-symbol";
    missing_pmc_process.node_id        = k_min_node;
    missing_pmc_process.process_id     = 999;
    EXPECT_THROW(writer->register_pmc_info(missing_pmc_process), std::runtime_error);

    writer_types::region_data_t region;
    region.name            = "region";
    region.start_timestamp = k_min_start;
    region.end_timestamp   = k_min_end;

    writer_types::trace_environment_t no_node;
    no_node.process_id = k_min_pid;
    no_node.thread_id  = k_min_tid;
    EXPECT_THROW(writer->insert_region_data(region, no_node), std::runtime_error);

    writer_types::trace_environment_t no_process;
    no_process.node_id   = k_min_node;
    no_process.thread_id = k_min_tid;
    EXPECT_THROW(writer->insert_region_data(region, no_process), std::runtime_error);

    writer_types::trace_environment_t no_thread = min_trace();
    EXPECT_THROW(writer->insert_region_data(region, no_thread), std::runtime_error);

    writer_types::region_data_t nameless = region;
    nameless.name                        = "";
    EXPECT_THROW(writer->insert_region_data(nameless, kernel_min_trace()),
                 std::runtime_error);

    writer_types::region_data_t args_without_event = region;
    writer_types::arg_data_t    arg;
    arg.type = "int";
    arg.name = "n";
    args_without_event.args.push_back(arg);
    EXPECT_THROW(writer->insert_region_data(args_without_event, kernel_min_trace()),
                 std::runtime_error);

    writer_types::kernel_dispatch_data_t kernel;
    kernel.kernel_symbol_id = k_min_kernel;
    kernel.start_timestamp  = k_min_start;
    kernel.end_timestamp    = k_min_end;

    auto no_agent = kernel_min_trace();
    no_agent.agent_id.reset();
    EXPECT_THROW(writer->insert_kernel_dispatch_data(kernel, no_agent),
                 std::runtime_error);

    auto no_queue = kernel_min_trace();
    no_queue.queue_id.reset();
    EXPECT_THROW(writer->insert_kernel_dispatch_data(kernel, no_queue),
                 std::runtime_error);

    auto no_stream = kernel_min_trace();
    no_stream.stream_id.reset();
    EXPECT_THROW(writer->insert_kernel_dispatch_data(kernel, no_stream),
                 std::runtime_error);

    auto no_kernel_thread = kernel_min_trace();
    no_kernel_thread.thread_id.reset();
    EXPECT_THROW(writer->insert_kernel_dispatch_data(kernel, no_kernel_thread),
                 std::runtime_error);

    kernel.kernel_symbol_id = 999;
    EXPECT_THROW(writer->insert_kernel_dispatch_data(kernel, kernel_min_trace()),
                 std::runtime_error);

    writer_types::memory_copy_data_t copy;
    copy.name            = "copy";
    copy.start_timestamp = k_min_start;
    copy.end_timestamp   = k_min_end;
    copy.size            = 8;
    EXPECT_THROW(writer->insert_memory_copy_data(copy, no_node), std::runtime_error);

    copy.name = "";
    EXPECT_THROW(writer->insert_memory_copy_data(copy, min_trace()), std::runtime_error);

    writer_types::memory_alloc_data_t alloc;
    alloc.start_timestamp = k_min_start;
    alloc.end_timestamp   = k_min_end;
    alloc.size            = 8;
    EXPECT_THROW(writer->insert_memory_alloc_data(alloc, no_process), std::runtime_error);

    writer_types::graph_launch_data_t graph;
    graph.start_timestamp = k_min_start;
    graph.end_timestamp   = k_min_end;
    EXPECT_THROW(writer->insert_graph_launch_data(graph, no_node), std::runtime_error);

    writer_types::hip_event_data_t hip;
    hip.name             = "hipEventRecord";
    hip.start_timestamp  = k_min_start;
    hip.end_timestamp    = k_min_end;
    hip.hip_event_handle = 1;
    EXPECT_THROW(writer->insert_hip_event_data(hip, no_process), std::runtime_error);

    hip.name = "";
    EXPECT_THROW(writer->insert_hip_event_data(hip, min_trace()), std::runtime_error);

    writer_types::pmc_event_data_t pmc_event;
    pmc_event.value = 1.0;
    writer_types::pmc_info_unique_id_t missing_pmc;
    missing_pmc.name = "not-registered";
    EXPECT_THROW(writer->insert_pmc_event_data(pmc_event, missing_pmc),
                 std::runtime_error);

    writer_types::pmc_info_unique_id_t registered_pmc;
    registered_pmc.name = "ctr-min";
    pmc_event.sample_id = 1;
    EXPECT_THROW(writer->insert_pmc_event_data(pmc_event, registered_pmc),
                 std::runtime_error);

    // A rejected row must leave the writer able to store a later minimum row.
    writer_types::region_data_t recovered;
    recovered.name            = "region-recovered";
    recovered.start_timestamp = k_min_start;
    recovered.end_timestamp   = k_min_end;
    recovered.event           = bare_event();
    EXPECT_NO_THROW(writer->insert_region_data(recovered, kernel_min_trace()));
}

}  // namespace
