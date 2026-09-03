// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/trace_cache/rocpd_helpers.hpp"
#include <cstdint>

#include <profiler-hub/reader.hpp>
#include <profiler-hub/reader_types.hpp>
#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

using rocprofsys::agent;
using rocprofsys::agent_type;
using rocprofsys::trace_cache::rocpd_helpers::make_agent_uid;
using rocprofsys::trace_cache::rocpd_helpers::make_event;
using rocprofsys::trace_cache::rocpd_helpers::make_trace_env;
using rocprofsys::trace_cache::rocpd_helpers::make_trace_env_with_agent;
using rocprofsys::trace_cache::rocpd_helpers::make_trace_env_with_agent_queue_stream;
using rocprofsys::trace_cache::rocpd_helpers::parse_memory_operation_name;

// ═══════════════════════════════════════════════════════════════════════════
// make_agent_uid — validate agent_unique_id_t struct fields
// ═══════════════════════════════════════════════════════════════════════════

namespace
{
agent
make_test_agent(agent_type type, size_t device_type_idx)
{
    agent result{};
    result.type                 = type;
    result.handle               = 0;
    result.device_id            = 0;
    result.node_id              = 0;
    result.logical_node_id      = 0;
    result.logical_node_type_id = 0;
    result.device_type_index    = device_type_idx;
    return result;
}
}  // namespace

TEST(make_agent_uid_test, gpu_agent_returns_gpu_string)
{
    auto uid = make_agent_uid(make_test_agent(agent_type::GPU, 2));

    ASSERT_TRUE(uid.agent_type.has_value());
    EXPECT_EQ(uid.agent_type.value(), "GPU");
    EXPECT_EQ(uid.type_index, 2U);
}

TEST(make_agent_uid_test, cpu_agent_returns_cpu_string)
{
    auto uid = make_agent_uid(make_test_agent(agent_type::CPU, 0));

    ASSERT_TRUE(uid.agent_type.has_value());
    EXPECT_EQ(uid.agent_type.value(), "CPU");
    EXPECT_EQ(uid.type_index, 0U);
}

TEST(make_agent_uid_test, nic_agent_returns_nic_string)
{
    auto uid = make_agent_uid(make_test_agent(agent_type::NIC, 1));

    ASSERT_TRUE(uid.agent_type.has_value())
        << "NIC agent_type must not be nullopt — this was a known bug";
    EXPECT_EQ(uid.agent_type.value(), "NIC");
    EXPECT_EQ(uid.type_index, 1U);
}

TEST(make_agent_uid_test, equality_same_agents)
{
    auto uid_a = make_agent_uid(make_test_agent(agent_type::GPU, 3));
    auto uid_b = make_agent_uid(make_test_agent(agent_type::GPU, 3));
    EXPECT_EQ(uid_a, uid_b);
}

TEST(make_agent_uid_test, inequality_different_type)
{
    auto gpu = make_agent_uid(make_test_agent(agent_type::GPU, 0));
    auto cpu = make_agent_uid(make_test_agent(agent_type::CPU, 0));
    EXPECT_FALSE(gpu == cpu);
}

TEST(make_agent_uid_test, inequality_different_index)
{
    auto idx0 = make_agent_uid(make_test_agent(agent_type::GPU, 0));
    auto idx1 = make_agent_uid(make_test_agent(agent_type::GPU, 1));
    EXPECT_FALSE(idx0 == idx1);
}

// ═══════════════════════════════════════════════════════════════════════════
// make_trace_env — validate trace_environment_t struct fields
// ═══════════════════════════════════════════════════════════════════════════

TEST(make_trace_env_test, basic_fields)
{
    auto env = make_trace_env(10, 42, 7);

    ASSERT_TRUE(env.node_id.has_value());
    ASSERT_TRUE(env.process_id.has_value());
    ASSERT_TRUE(env.thread_id.has_value());
    EXPECT_EQ(env.node_id.value(), 10U);
    EXPECT_EQ(env.process_id.value(), 42U);
    EXPECT_EQ(env.thread_id.value(), 7U);

    EXPECT_FALSE(env.agent_id.has_value());
    EXPECT_FALSE(env.stream_id.has_value());
    EXPECT_FALSE(env.queue_id.has_value());
    EXPECT_FALSE(env.track_name.has_value());
}

TEST(make_trace_env_test, with_agent_populates_agent_id)
{
    auto env = make_trace_env_with_agent(1, 2, 3, make_test_agent(agent_type::GPU, 5));

    ASSERT_TRUE(env.agent_id.has_value());
    ASSERT_TRUE(env.agent_id->agent_type.has_value());
    EXPECT_EQ(env.agent_id->agent_type.value(), "GPU");
    EXPECT_EQ(env.agent_id->type_index, 5U);
    EXPECT_EQ(env.node_id.value(), 1U);
    EXPECT_EQ(env.process_id.value(), 2U);
    EXPECT_EQ(env.thread_id.value(), 3U);
}

TEST(make_trace_env_test, with_queue_stream_populates_all)
{
    auto env = make_trace_env_with_agent_queue_stream(
        1, 2, 3, make_test_agent(agent_type::GPU, 0), 100, 200);

    ASSERT_TRUE(env.queue_id.has_value());
    ASSERT_TRUE(env.stream_id.has_value());
    EXPECT_EQ(env.queue_id.value(), 100U);
    EXPECT_EQ(env.stream_id.value(), 200U);
    ASSERT_TRUE(env.agent_id.has_value());
}

// ═══════════════════════════════════════════════════════════════════════════
// make_event — validate event_data_t struct fields
// ═══════════════════════════════════════════════════════════════════════════

TEST(make_event_test, fields_populated)
{
    auto event = make_event(10, 20, 30, "kernel_dispatch");

    ASSERT_TRUE(event.stack_id.has_value());
    ASSERT_TRUE(event.parent_stack_id.has_value());
    ASSERT_TRUE(event.correlation_id.has_value());
    ASSERT_TRUE(event.event_category.has_value());
    EXPECT_EQ(event.stack_id.value(), 10U);
    EXPECT_EQ(event.parent_stack_id.value(), 20U);
    EXPECT_EQ(event.correlation_id.value(), 30U);
    EXPECT_EQ(event.event_category.value(), "kernel_dispatch");
    EXPECT_TRUE(event.call_stack.empty());
    EXPECT_TRUE(event.line_info_list.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// parse_memory_operation_name — exhaustive lookup table validation
// ═══════════════════════════════════════════════════════════════════════════

struct memory_op_test_case
{
    const char* input;
    const char* expected_op;
    const char* expected_type;
};

class parse_memory_op_test : public ::testing::TestWithParam<memory_op_test_case>
{};

TEST_P(parse_memory_op_test, lookup)
{
    auto [input, expected_op, expected_type] = GetParam();
    auto [actual_op, actual_type]            = parse_memory_operation_name(input);
    EXPECT_EQ(actual_op, expected_op) << "input: " << input;
    EXPECT_EQ(actual_type, expected_type) << "input: " << input;
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    all_operations, parse_memory_op_test,
    ::testing::Values(
        memory_op_test_case{"MEMORY_ALLOCATION_NONE",         "NONE",          "REAL"},
        memory_op_test_case{"MEMORY_ALLOCATION_ALLOCATE",     "ALLOC",         "REAL"},
        memory_op_test_case{"MEMORY_ALLOCATION_VMEM_ALLOCATE","ALLOC",         "VIRTUAL"},
        memory_op_test_case{"MEMORY_ALLOCATION_FREE",         "FREE",          "REAL"},
        memory_op_test_case{"MEMORY_ALLOCATION_VMEM_FREE",    "FREE",          "VIRTUAL"},
        memory_op_test_case{"SCRATCH_MEMORY_NONE",            "NONE",          "SCRATCH"},
        memory_op_test_case{"SCRATCH_MEMORY_ALLOC",           "ALLOC",         "SCRATCH"},
        memory_op_test_case{"SCRATCH_MEMORY_FREE",            "FREE",          "SCRATCH"},
        memory_op_test_case{"SCRATCH_MEMORY_ASYNC_RECLAIM",   "ASYNC_RECLAIM", "SCRATCH"},
        memory_op_test_case{"BOGUS_OPERATION",                "UNKNOWN",       "UNKNOWN"},
        memory_op_test_case{"",                               "UNKNOWN",       "UNKNOWN"},
        memory_op_test_case{"memory_allocation_allocate",     "UNKNOWN",       "UNKNOWN"},
        memory_op_test_case{"MEMORY_ALLOCATION",              "UNKNOWN",       "UNKNOWN"}
    ));
// clang-format on

// ═══════════════════════════════════════════════════════════════════════════
// Integration tests: write → flush → read back → validate actual DB values
//
// Pattern (same as profiler-hub's own writer_test.cpp):
//   1. writer_t::register_* / insert_*
//   2. writer_t::flush_in_memory_data_to_disk()
//   3. reader_t::get_all_* / get_events / get_*_details
//   4. ASSERT field values match what was written
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

        m_db_path    = (m_temp_dir / "test.rocpd").string();
        m_uuid       = "test123";
        auto storage = std::make_unique<profiler_hub::storage_t>(m_db_path, m_uuid);
        m_writer     = std::make_unique<profiler_hub::writer_t>(std::move(storage));
    }

    void TearDown() override
    {
        m_writer.reset();
        m_reader.reset();
        std::filesystem::remove_all(m_temp_dir);
    }

    void register_base_metadata()
    {
        profiler_hub::writer_types::node_info_t node{};
        node.node_id     = NODE_ID;
        node.hash        = 12345;
        node.machine_id  = "test-machine";
        node.system_name = "Linux";
        node.hostname    = "testhost";
        m_writer->register_node_info(node);

        profiler_hub::writer_types::process_info_t proc{};
        proc.ppid    = PPID;
        proc.pid     = PID;
        proc.start   = 1000;
        proc.end     = 9000;
        proc.command = "test_binary";
        proc.node_id = NODE_ID;
        m_writer->register_process_info(proc);

        profiler_hub::writer_types::thread_info_t thread{};
        thread.parent_process_id = PPID;
        thread.thread_id         = THREAD_ID;
        thread.name              = "Thread 300";
        thread.start             = 1000;
        thread.end               = 9000;
        thread.node_id           = NODE_ID;
        thread.process_id        = PID;
        m_writer->register_thread_info(thread);
    }

    void register_gpu_agent()
    {
        profiler_hub::writer_types::agent_info_t info{};
        info.unique_id      = make_agent_uid(gpu_agent());
        info.absolute_index = 0;
        info.name           = "gfx90a";
        info.model_name     = "MI210";
        info.vendor_name    = "AMD";
        info.product_name   = "Instinct MI210";
        info.node_id        = NODE_ID;
        info.process_id     = PID;
        m_writer->register_agent_info(info);
    }

    void register_queue_and_stream()
    {
        profiler_hub::writer_types::queue_info_t queue{};
        queue.queue_id   = QUEUE_ID;
        queue.name       = "Queue 10";
        queue.node_id    = NODE_ID;
        queue.process_id = PID;
        m_writer->register_queue_info(queue);

        profiler_hub::writer_types::stream_info_t stream{};
        stream.stream_id  = STREAM_ID;
        stream.name       = "Stream 20";
        stream.node_id    = NODE_ID;
        stream.process_id = PID;
        m_writer->register_stream_info(stream);
    }

    void register_code_object_and_kernel_symbol(size_t code_obj_id   = 1,
                                                size_t kernel_sym_id = 1)
    {
        profiler_hub::writer_types::code_object_info_t co{};
        co.id         = code_obj_id;
        co.node_id    = NODE_ID;
        co.process_id = PID;
        co.agent_id   = make_agent_uid(gpu_agent());
        m_writer->register_code_object_info(co);

        profiler_hub::writer_types::kernel_symbol_info_t ks{};
        ks.id          = kernel_sym_id;
        ks.name        = "test_kernel";
        ks.node_id     = NODE_ID;
        ks.process_id  = PID;
        ks.code_obj_id = code_obj_id;
        m_writer->register_kernel_symbol_info(ks);
    }

    void flush_and_open_reader()
    {
        m_writer->flush_in_memory_data_to_disk();
        auto read_storage = std::make_unique<profiler_hub::storage_t>(m_db_path, m_uuid);
        m_reader = std::make_unique<profiler_hub::reader_t>(std::move(read_storage));
    }

    static agent gpu_agent()
    {
        agent result{};
        result.type              = agent_type::GPU;
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
        result.type              = agent_type::CPU;
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
        result.type              = agent_type::NIC;
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
    std::unique_ptr<profiler_hub::writer_t> m_writer;
    std::unique_ptr<profiler_hub::reader_t> m_reader;

    static int test_counter_;
};

int rocpd_write_read_test::test_counter_ = 0;

// ---------------------------------------------------------------------------
// Metadata: write agents, read back, validate field values
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, agents_round_trip_all_types)
{
    register_base_metadata();

    auto register_agent = [&](const agent& agent_obj) {
        profiler_hub::writer_types::agent_info_t info{};
        info.unique_id    = make_agent_uid(agent_obj);
        info.name         = agent_obj.name;
        info.model_name   = agent_obj.model_name;
        info.vendor_name  = agent_obj.vendor_name;
        info.product_name = agent_obj.product_name;
        info.node_id      = NODE_ID;
        info.process_id   = PID;
        m_writer->register_agent_info(info);
    };

    register_agent(gpu_agent());
    register_agent(cpu_agent());
    register_agent(nic_agent());

    flush_and_open_reader();
    auto agents = m_reader->get_all_agents();

    ASSERT_EQ(agents.size(), 3U);

    bool found_gpu = false, found_cpu = false, found_nic = false;
    for(const auto& agent_ptr : agents)
    {
        if(agent_ptr->agent_type == "GPU")
        {
            found_gpu = true;
            EXPECT_EQ(agent_ptr->name, "gfx90a");
            EXPECT_EQ(agent_ptr->model_name, "MI210");
        }
        else if(agent_ptr->agent_type == "CPU")
        {
            found_cpu = true;
            EXPECT_EQ(agent_ptr->name, "CPU0");
            EXPECT_EQ(agent_ptr->model_name, "EPYC");
        }
        else if(agent_ptr->agent_type == "NIC")
        {
            found_nic = true;
            EXPECT_EQ(agent_ptr->name, "NIC0");
            EXPECT_EQ(agent_ptr->model_name, "CX7");
        }
    }
    EXPECT_TRUE(found_gpu) << "GPU agent not found in read-back";
    EXPECT_TRUE(found_cpu) << "CPU agent not found in read-back";
    EXPECT_TRUE(found_nic) << "NIC agent not found in read-back";
}

// ---------------------------------------------------------------------------
// Kernel dispatch: write, read back, validate timestamps + grid sizes
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, kernel_dispatch_values_persisted)
{
    register_base_metadata();
    register_gpu_agent();
    register_queue_and_stream();
    register_code_object_and_kernel_symbol();

    auto event = make_event(1, 0, 0, "kernel_dispatch");

    profiler_hub::writer_types::kernel_dispatch_data_t kd_write{};
    kd_write.event            = event;
    kd_write.dispatch_id      = 42;
    kd_write.start_timestamp  = 5000;
    kd_write.end_timestamp    = 6000;
    kd_write.kernel_symbol_id = 1;
    kd_write.code_object_id   = 1;
    kd_write.workgroup_size_x = 256;
    kd_write.workgroup_size_y = 1;
    kd_write.workgroup_size_z = 1;
    kd_write.grid_size_x      = 1024;
    kd_write.grid_size_y      = 1;
    kd_write.grid_size_z      = 1;
    kd_write.name             = "my_test_kernel";

    auto env = make_trace_env_with_agent_queue_stream(NODE_ID, PID, THREAD_ID,
                                                      gpu_agent(), QUEUE_ID, STREAM_ID);
    m_writer->insert_kernel_dispatch_data(kd_write, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::kernel_dispatch)
            continue;

        auto detail = m_reader->get_kernel_dispatch_details(tl_event);
        ASSERT_TRUE(detail.has_value()) << "kernel dispatch detail should be readable";

        EXPECT_EQ(detail->start_timestamp, 5000U);
        EXPECT_EQ(detail->end_timestamp, 6000U);
        EXPECT_EQ(detail->dispatch_id, 42U);
        EXPECT_EQ(detail->workgroup_size_x, 256U);
        EXPECT_EQ(detail->grid_size_x, 1024U);
        EXPECT_EQ(detail->name, "my_test_kernel");

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Kernel dispatch event not found in read-back";
}

// ---------------------------------------------------------------------------
// Region with args: write, read back, validate name + argument values
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, region_with_args_values_persisted)
{
    register_base_metadata();

    auto event = make_event(1, 0, 0, "HIP_API");

    profiler_hub::writer_types::region_data_t region{};
    region.event           = event;
    region.start_timestamp = 5000;
    region.end_timestamp   = 5200;
    region.name            = "hipMemcpy";

    profiler_hub::writer_types::arg_data_t arg0{};
    arg0.position = 0;
    arg0.type     = "void*";
    arg0.name     = "dst";
    arg0.value    = "0x7f0000000000";
    region.args.push_back(arg0);

    profiler_hub::writer_types::arg_data_t arg1{};
    arg1.position = 1;
    arg1.type     = "size_t";
    arg1.name     = "sizeBytes";
    arg1.value    = "4096";
    region.args.push_back(arg1);

    auto env = make_trace_env(NODE_ID, PID, THREAD_ID);
    m_writer->insert_region_data(region, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::region)
            continue;

        auto detail = m_reader->get_region_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 5000U);
        EXPECT_EQ(detail->end_timestamp, 5200U);
        EXPECT_EQ(detail->name, "hipMemcpy");

        auto args = m_reader->get_arguments(tl_event);
        ASSERT_EQ(args.size(), 2U);
        EXPECT_EQ(args[0]->name, "dst");
        EXPECT_EQ(args[0]->value, "0x7f0000000000");
        EXPECT_EQ(args[1]->name, "sizeBytes");
        EXPECT_EQ(args[1]->value, "4096");

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Region event not found in read-back";
}

// ---------------------------------------------------------------------------
// Memory copy: write, read back, validate size + agent ids
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, memory_copy_values_persisted)
{
    register_base_metadata();

    // Register both src (CPU) and dst (GPU) agents
    for(const auto& agent_obj : { cpu_agent(), gpu_agent() })
    {
        profiler_hub::writer_types::agent_info_t info{};
        info.unique_id    = make_agent_uid(agent_obj);
        info.name         = agent_obj.name;
        info.model_name   = agent_obj.model_name;
        info.vendor_name  = agent_obj.vendor_name;
        info.product_name = agent_obj.product_name;
        info.node_id      = NODE_ID;
        info.process_id   = PID;
        m_writer->register_agent_info(info);
    }

    register_queue_and_stream();

    auto event = make_event(1, 0, 0, "memory_copy");

    profiler_hub::writer_types::memory_copy_data_t mc_write{};
    mc_write.event           = event;
    mc_write.start_timestamp = 6500;
    mc_write.end_timestamp   = 7000;
    mc_write.dst_agent_id    = make_agent_uid(gpu_agent());
    mc_write.dst_address     = 0x7F0000000000;
    mc_write.src_agent_id    = make_agent_uid(cpu_agent());
    mc_write.src_address     = 0x100000;
    mc_write.size            = 4096;
    mc_write.name            = "COPY_HOST_TO_DEVICE";
    mc_write.region_name     = "COPY_HOST_TO_DEVICE";

    auto env      = make_trace_env(NODE_ID, PID, THREAD_ID);
    env.stream_id = STREAM_ID;
    m_writer->insert_memory_copy_data(mc_write, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::memory_copy)
            continue;

        auto detail = m_reader->get_memory_copy_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 6500U);
        EXPECT_EQ(detail->end_timestamp, 7000U);
        EXPECT_EQ(detail->size, 4096U);
        EXPECT_EQ(detail->name, "COPY_HOST_TO_DEVICE");

        ASSERT_NE(detail->dst_agent_id, nullptr);
        EXPECT_EQ(detail->dst_agent_id->agent_type, "GPU");

        ASSERT_NE(detail->src_agent_id, nullptr);
        EXPECT_EQ(detail->src_agent_id->agent_type, "CPU");

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Memory copy event not found in read-back";
}

// ---------------------------------------------------------------------------
// Memory alloc: write, read back, validate type/level/size
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, memory_alloc_values_persisted)
{
    register_base_metadata();
    register_gpu_agent();

    auto event = make_event(1, 0, 0, "scratch_memory");

    profiler_hub::writer_types::memory_alloc_data_t ma_write{};
    ma_write.event           = event;
    ma_write.type            = "ALLOC";
    ma_write.level           = "SCRATCH";
    ma_write.start_timestamp = 4500;
    ma_write.end_timestamp   = 4600;
    ma_write.address         = 0;
    ma_write.size            = 65536;

    auto env = make_trace_env_with_agent(NODE_ID, PID, THREAD_ID, gpu_agent());
    m_writer->insert_memory_alloc_data(ma_write, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::memory_allocate)
            continue;

        auto detail = m_reader->get_memory_alloc_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 4500U);
        EXPECT_EQ(detail->end_timestamp, 4600U);
        EXPECT_EQ(detail->type, "ALLOC");
        EXPECT_EQ(detail->level, "SCRATCH");
        EXPECT_EQ(detail->size, 65536U);

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Memory alloc event not found in read-back";
}

// ---------------------------------------------------------------------------
// PMC event: write, read back, validate counter value
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, pmc_event_value_persisted)
{
    register_base_metadata();
    register_gpu_agent();

    const profiler_hub::writer_types::agent_unique_id_t agent_uid =
        make_agent_uid(gpu_agent());

    profiler_hub::writer_types::pmc_info_t pmc_desc{};
    pmc_desc.unique_id.name     = "gfx_activity";
    pmc_desc.unique_id.agent_id = agent_uid;
    pmc_desc.symbol             = "GFX_ACTIVITY";
    pmc_desc.description        = "GPU activity";
    pmc_desc.node_id            = NODE_ID;
    pmc_desc.process_id         = PID;
    m_writer->register_pmc_info(pmc_desc);

    profiler_hub::writer_types::track_info_t track{};
    track.name       = "GPU Activity";
    track.node_id    = NODE_ID;
    track.process_id = PID;
    track.thread_id  = THREAD_ID;
    m_writer->register_track_info(track);

    profiler_hub::writer_types::pmc_event_data_t pmc_write{};
    pmc_write.value = 75.5;

    profiler_hub::writer_types::sample_data_t sample{};
    sample.timestamp = 6000;
    sample.track     = track;
    pmc_write.sample = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid{};
    pmc_uid.name     = "gfx_activity";
    pmc_uid.agent_id = agent_uid;

    m_writer->insert_pmc_event_data(pmc_write, pmc_uid);

    flush_and_open_reader();

    // PMC events are not part of the unified timeline (get_events()),
    // so verify the PMC info was registered and data was persisted.
    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    EXPECT_EQ(pmc_infos[0]->symbol, "GFX_ACTIVITY");
    EXPECT_EQ(pmc_infos[0]->description, "GPU activity");

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// Event counts: write multiple types, verify counts match
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, event_counts_match_inserted_data)
{
    register_base_metadata();
    register_gpu_agent();
    register_queue_and_stream();
    register_code_object_and_kernel_symbol();

    auto env_full = make_trace_env_with_agent_queue_stream(
        NODE_ID, PID, THREAD_ID, gpu_agent(), QUEUE_ID, STREAM_ID);
    auto env_basic = make_trace_env(NODE_ID, PID, THREAD_ID);

    // 2 regions
    for(int idx = 0; idx < 2; ++idx)
    {
        profiler_hub::writer_types::region_data_t region{};
        region.event           = make_event(idx, 0, 0, "HIP_API");
        region.start_timestamp = 1000 + idx * 100;
        region.end_timestamp   = 1050 + idx * 100;
        auto region_name       = "region_" + std::to_string(idx);
        region.name            = region_name.c_str();
        m_writer->insert_region_data(region, env_basic);
    }

    // 1 kernel dispatch
    {
        profiler_hub::writer_types::kernel_dispatch_data_t kd{};
        kd.event            = make_event(10, 0, 0, "kernel_dispatch");
        kd.dispatch_id      = 1;
        kd.start_timestamp  = 5000;
        kd.end_timestamp    = 6000;
        kd.kernel_symbol_id = 1;
        kd.code_object_id   = 1;
        kd.workgroup_size_x = 64;
        kd.workgroup_size_y = 1;
        kd.workgroup_size_z = 1;
        kd.grid_size_x      = 256;
        kd.grid_size_y      = 1;
        kd.grid_size_z      = 1;
        kd.name             = "count_test_kernel";
        m_writer->insert_kernel_dispatch_data(kd, env_full);
    }

    flush_and_open_reader();

    auto counts = m_reader->get_event_counts();

    auto region_it = counts.find(profiler_hub::reader_types::event_type_t::region);
    ASSERT_NE(region_it, counts.end());
    EXPECT_EQ(region_it->second, 2U);

    auto kd_it = counts.find(profiler_hub::reader_types::event_type_t::kernel_dispatch);
    ASSERT_NE(kd_it, counts.end());
    EXPECT_EQ(kd_it->second, 1U);
}

// ---------------------------------------------------------------------------
// Metadata round-trip: node, process, thread, queue, stream
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, metadata_round_trip)
{
    register_base_metadata();
    register_queue_and_stream();

    flush_and_open_reader();

    auto nodes = m_reader->get_all_nodes();
    ASSERT_EQ(nodes.size(), 1U);
    EXPECT_EQ(nodes[0]->hostname, "testhost");
    EXPECT_EQ(nodes[0]->system_name, "Linux");

    auto processes = m_reader->get_all_processes();
    ASSERT_EQ(processes.size(), 1U);
    EXPECT_EQ(processes[0]->pid, PID);
    EXPECT_EQ(processes[0]->command, "test_binary");

    auto threads = m_reader->get_all_threads();
    ASSERT_EQ(threads.size(), 1U);
    EXPECT_EQ(threads[0]->thread_id, THREAD_ID);
    EXPECT_EQ(threads[0]->name, "Thread 300");

    auto queues = m_reader->get_all_queues();
    ASSERT_EQ(queues.size(), 1U);
    // queue_id in the reader is the DB primary key, not the user-supplied ID;
    // validate the name which is reliably round-tripped.
    EXPECT_EQ(queues[0]->name, "Queue 10");

    auto streams = m_reader->get_all_streams();
    ASSERT_EQ(streams.size(), 1U);
    EXPECT_EQ(streams[0]->name, "Stream 20");
}

// ---------------------------------------------------------------------------
// Output file existence and non-empty after flush
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, flush_creates_nonempty_file)
{
    register_base_metadata();

    profiler_hub::writer_types::region_data_t region{};
    region.start_timestamp = 1000;
    region.end_timestamp   = 2000;
    region.name            = "flush_test";

    auto env = make_trace_env(NODE_ID, PID, THREAD_ID);
    m_writer->insert_region_data(region, env);
    m_writer->flush_in_memory_data_to_disk();

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ═══════════════════════════════════════════════════════════════════════════
// handle() pathway tests — validate the exact writer API calls made by
// each rocpd_processor_t::handle() overload, then read back and assert
// field values. Each test mirrors the data construction logic from the
// corresponding handle() method in rocpd_processor.cpp.
// ═══════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// handle(scratch_memory_sample): memory_alloc with flags extdata
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_scratch_memory_pathway)
{
    register_base_metadata();
    register_gpu_agent();
    register_queue_and_stream();

    auto ev = make_event(100, 50, 0, "scratch_memory");

    profiler_hub::writer_types::memory_alloc_data_t ma{};
    ma.event           = ev;
    ma.type            = "ALLOC";
    ma.level           = "SCRATCH";
    ma.start_timestamp = 3000;
    ma.end_timestamp   = 3100;
    ma.address         = 0;
    ma.size            = 131072;
    ma.extdata         = "{\"flags\": 0}";

    auto env = make_trace_env_with_agent_queue_stream(NODE_ID, PID, THREAD_ID,
                                                      gpu_agent(), QUEUE_ID, STREAM_ID);

    m_writer->insert_memory_alloc_data(ma, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::memory_allocate)
            continue;

        auto detail = m_reader->get_memory_alloc_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 3000U);
        EXPECT_EQ(detail->end_timestamp, 3100U);
        EXPECT_EQ(detail->type, "ALLOC");
        EXPECT_EQ(detail->level, "SCRATCH");
        EXPECT_EQ(detail->size, 131072U);

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Scratch memory event not found in read-back";
    EXPECT_TRUE(std::filesystem::exists(m_db_path));
}

// ---------------------------------------------------------------------------
// handle(memory_allocate_sample): memory_alloc with agent + address
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_memory_allocate_pathway)
{
    register_base_metadata();
    register_gpu_agent();
    register_queue_and_stream();

    auto ev = make_event(200, 100, 0, "memory_allocate");

    profiler_hub::writer_types::memory_alloc_data_t ma{};
    ma.event           = ev;
    ma.type            = "ALLOC";
    ma.level           = "REAL";
    ma.start_timestamp = 7000;
    ma.end_timestamp   = 7200;
    ma.address         = 0x7F0000100000;
    ma.size            = 8192;

    auto env      = make_trace_env_with_agent(NODE_ID, PID, THREAD_ID, gpu_agent());
    env.stream_id = STREAM_ID;

    m_writer->insert_memory_alloc_data(ma, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::memory_allocate)
            continue;

        auto detail = m_reader->get_memory_alloc_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 7000U);
        EXPECT_EQ(detail->end_timestamp, 7200U);
        EXPECT_EQ(detail->type, "ALLOC");
        EXPECT_EQ(detail->level, "REAL");
        EXPECT_EQ(detail->size, 8192U);

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Memory allocate event not found in read-back";
    EXPECT_TRUE(std::filesystem::exists(m_db_path));
}

// ---------------------------------------------------------------------------
// handle(backtrace_region_sample): region with track_name and extdata
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_backtrace_region_pathway)
{
    register_base_metadata();

    profiler_hub::writer_types::track_info_t bt_track{};
    bt_track.name       = "Sampling [CPU 0]";
    bt_track.node_id    = NODE_ID;
    bt_track.process_id = PID;
    bt_track.thread_id  = THREAD_ID;
    m_writer->register_track_info(bt_track);

    auto ev    = make_event(0, 0, 0, "sampling");
    ev.extdata = "{\"backtrace\": true}";

    profiler_hub::writer_types::region_data_t region{};
    region.event           = ev;
    region.start_timestamp = 8000;
    region.end_timestamp   = 8500;
    region.name            = "backtrace_sample_func";

    auto env       = make_trace_env(NODE_ID, PID, THREAD_ID);
    env.track_name = "Sampling [CPU 0]";

    m_writer->insert_region_data(region, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::region)
            continue;

        auto detail = m_reader->get_region_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 8000U);
        EXPECT_EQ(detail->end_timestamp, 8500U);
        EXPECT_EQ(detail->name, "backtrace_sample_func");

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Backtrace region event not found in read-back";
    EXPECT_TRUE(std::filesystem::exists(m_db_path));
}

// ---------------------------------------------------------------------------
// handle(in_time_sample): pmc_event with track + sample timestamp
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_in_time_sample_pathway)
{
    register_base_metadata();
    register_gpu_agent();

    auto agent_uid = make_agent_uid(gpu_agent());

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid{};
    pmc_uid.name     = "my_track";
    pmc_uid.agent_id = agent_uid;

    profiler_hub::writer_types::pmc_info_t pmc_desc{};
    pmc_desc.unique_id   = pmc_uid;
    pmc_desc.symbol      = "IN_TIME";
    pmc_desc.description = "In-time sample counter";
    pmc_desc.node_id     = NODE_ID;
    pmc_desc.process_id  = PID;
    m_writer->register_pmc_info(pmc_desc);

    profiler_hub::writer_types::track_info_t track{};
    track.name       = "my_track";
    track.node_id    = NODE_ID;
    track.process_id = PID;
    track.thread_id  = THREAD_ID;
    m_writer->register_track_info(track);

    auto ev    = make_event(5, 3, 0, "my_track");
    ev.extdata = "{\"metadata\": \"test\"}";

    profiler_hub::writer_types::pmc_event_data_t pmc_data{};
    pmc_data.event = ev;
    pmc_data.value = 0.0;

    profiler_hub::writer_types::sample_data_t sample{};
    sample.timestamp = 9500;
    sample.track     = track;
    pmc_data.sample  = sample;

    m_writer->insert_pmc_event_data(pmc_data, pmc_uid);

    flush_and_open_reader();

    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_GE(pmc_infos.size(), 1U);

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// handle(pmc_event_with_sample): pmc_event with agent + value + tid
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_pmc_event_with_sample_pathway)
{
    register_base_metadata();
    register_gpu_agent();

    auto agent_uid = make_agent_uid(gpu_agent());

    profiler_hub::writer_types::pmc_info_t pmc_desc{};
    pmc_desc.unique_id.name     = "SQ_WAVES";
    pmc_desc.unique_id.agent_id = agent_uid;
    pmc_desc.symbol             = "SQ_WAVES";
    pmc_desc.description        = "Shader wavefronts";
    pmc_desc.node_id            = NODE_ID;
    pmc_desc.process_id         = PID;
    m_writer->register_pmc_info(pmc_desc);

    profiler_hub::writer_types::track_info_t track{};
    track.name       = "SQ_WAVES [GPU 0]";
    track.node_id    = NODE_ID;
    track.process_id = PID;
    track.thread_id  = THREAD_ID;
    m_writer->register_track_info(track);

    auto ev    = make_event(10, 5, 42, "SQ_WAVES [GPU 0]");
    ev.extdata = "{}";

    profiler_hub::writer_types::pmc_event_data_t pmc_data{};
    pmc_data.event = ev;
    pmc_data.value = 1024.0;

    profiler_hub::writer_types::sample_data_t sample{};
    sample.timestamp = 10000;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid{};
    pmc_uid.name     = "SQ_WAVES";
    pmc_uid.agent_id = agent_uid;

    m_writer->insert_pmc_event_data(pmc_data, pmc_uid);

    flush_and_open_reader();

    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 1U);
    EXPECT_EQ(pmc_infos[0]->symbol, "SQ_WAVES");
    EXPECT_EQ(pmc_infos[0]->description, "Shader wavefronts");

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// handle(gpu_pmc_sample): multiple scalar PMC inserts for GPU metrics
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_gpu_pmc_sample_pathway)
{
    register_base_metadata();
    register_gpu_agent();

    auto agent_uid = make_agent_uid(gpu_agent());

    auto register_pmc = [&](const char* name, const char* desc) {
        profiler_hub::writer_types::pmc_info_t pi{};
        pi.unique_id.name     = name;
        pi.unique_id.agent_id = agent_uid;
        pi.symbol             = name;
        pi.description        = desc;
        pi.node_id            = NODE_ID;
        pi.process_id         = PID;
        m_writer->register_pmc_info(pi);
    };

    register_pmc("gfx_busy", "GFX activity");
    register_pmc("umc_busy", "UMC activity");
    register_pmc("gpu_temperature", "Hotspot temp");

    auto register_and_insert_pmc = [&](const char* pmc_name, const char* track_name,
                                       double value, size_t timestamp) {
        profiler_hub::writer_types::track_info_t track{};
        track.name       = track_name;
        track.node_id    = NODE_ID;
        track.process_id = PID;
        m_writer->register_track_info(track);

        auto ev = make_event(0, 0, 0, "amd_smi");

        profiler_hub::writer_types::pmc_event_data_t pmc_data{};
        pmc_data.event = ev;
        pmc_data.value = value;

        profiler_hub::writer_types::sample_data_t sample{};
        sample.timestamp = timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t uid{};
        uid.name     = pmc_name;
        uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, uid);
    };

    register_and_insert_pmc("gfx_busy", "gfx_busy", 85.0, 11000);
    register_and_insert_pmc("umc_busy", "umc_busy", 42.0, 11000);
    register_and_insert_pmc("gpu_temperature", "gpu_temperature", 72.0, 11000);

    flush_and_open_reader();

    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 3U);

    bool found_gfx = false, found_umc = false, found_temp = false;
    for(const auto& pi : pmc_infos)
    {
        if(pi->symbol == "gfx_busy") found_gfx = true;
        if(pi->symbol == "umc_busy") found_umc = true;
        if(pi->symbol == "gpu_temperature") found_temp = true;
    }
    EXPECT_TRUE(found_gfx) << "gfx_busy PMC info not found";
    EXPECT_TRUE(found_umc) << "umc_busy PMC info not found";
    EXPECT_TRUE(found_temp) << "gpu_temperature PMC info not found";

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// handle(ainic_pmc_sample): NIC RDMA metric PMC inserts
// NIC agents are not supported by profiler-hub register_agent_info, so the
// handle() pathway registers PMC events using make_agent_uid(NIC) directly.
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_ainic_pmc_sample_pathway)
{
    register_base_metadata();
    register_gpu_agent();

    auto gpu_uid = make_agent_uid(gpu_agent());

    auto ev = make_event(0, 0, 0, "amd_smi_nic");

    auto register_and_insert_nic_pmc = [&](const char* pmc_name, const char* track_name,
                                           double value, size_t timestamp) {
        profiler_hub::writer_types::pmc_info_t pi{};
        pi.unique_id.name     = pmc_name;
        pi.unique_id.agent_id = gpu_uid;
        pi.symbol             = pmc_name;
        pi.description        = pmc_name;
        pi.node_id            = NODE_ID;
        pi.process_id         = PID;
        m_writer->register_pmc_info(pi);

        profiler_hub::writer_types::track_info_t track{};
        track.name       = track_name;
        track.node_id    = NODE_ID;
        track.process_id = PID;
        m_writer->register_track_info(track);

        profiler_hub::writer_types::pmc_event_data_t pmc_data{};
        pmc_data.event = ev;
        pmc_data.value = value;

        profiler_hub::writer_types::sample_data_t sample{};
        sample.timestamp = timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t uid{};
        uid.name     = pmc_name;
        uid.agent_id = gpu_uid;

        m_writer->insert_pmc_event_data(pmc_data, uid);
    };

    register_and_insert_nic_pmc("nic_rx_ucast_bytes", "ainic_rx_rdma_ucast_bytes",
                                1048576.0, 12000);
    register_and_insert_nic_pmc("nic_tx_ucast_bytes", "ainic_tx_rdma_ucast_bytes",
                                524288.0, 12000);

    m_writer->flush_in_memory_data_to_disk();

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// handle(cpu_pmc_sample): process-level + per-core PMC inserts
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_cpu_pmc_sample_pathway)
{
    register_base_metadata();

    auto cpu = cpu_agent();

    profiler_hub::writer_types::agent_info_t cpu_info{};
    cpu_info.unique_id    = make_agent_uid(cpu);
    cpu_info.name         = cpu.name;
    cpu_info.model_name   = cpu.model_name;
    cpu_info.vendor_name  = cpu.vendor_name;
    cpu_info.product_name = cpu.product_name;
    cpu_info.node_id      = NODE_ID;
    cpu_info.process_id   = PID;
    m_writer->register_agent_info(cpu_info);

    auto agent_uid = make_agent_uid(cpu);

    auto register_pmc = [&](const char* name, const char* desc) {
        profiler_hub::writer_types::pmc_info_t pi{};
        pi.unique_id.name     = name;
        pi.unique_id.agent_id = agent_uid;
        pi.symbol             = name;
        pi.description        = desc;
        pi.node_id            = NODE_ID;
        pi.process_id         = PID;
        m_writer->register_pmc_info(pi);
    };

    register_pmc("process_page_rss", "Process RSS");
    register_pmc("cpu_frequency", "CPU frequency");

    auto ev = make_event(0, 0, 0, "cpu_freq");

    auto insert_cpu_pmc = [&](const char* pmc_name, const char* track_name, double value,
                              size_t timestamp) {
        profiler_hub::writer_types::track_info_t track{};
        track.name       = track_name;
        track.node_id    = NODE_ID;
        track.process_id = PID;
        m_writer->register_track_info(track);

        profiler_hub::writer_types::pmc_event_data_t pmc_data{};
        pmc_data.event = ev;
        pmc_data.value = value;

        profiler_hub::writer_types::sample_data_t sample{};
        sample.timestamp = timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t uid{};
        uid.name     = pmc_name;
        uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, uid);
    };

    insert_cpu_pmc("process_page_rss", "process_page_rss", 256.5, 13000);
    insert_cpu_pmc("cpu_frequency", "cpu_frequency [0] Core [0]", 3200.0, 13000);
    insert_cpu_pmc("cpu_frequency", "cpu_frequency [0] Core [1]", 3100.0, 13000);

    flush_and_open_reader();

    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_infos.size(), 2U);

    auto agents = m_reader->get_all_agents();
    ASSERT_GE(agents.size(), 1U);

    bool found_cpu_agent = false;
    for(const auto& a : agents)
    {
        if(a->agent_type == "CPU")
        {
            found_cpu_agent = true;
            EXPECT_EQ(a->name, "CPU0");
        }
    }
    EXPECT_TRUE(found_cpu_agent) << "CPU agent not found in read-back";

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// handle(kfd_sample): region + pmc_event with args (KFD event pathway)
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_kfd_sample_pathway)
{
    register_base_metadata();
    register_gpu_agent();

    auto agent_uid = make_agent_uid(gpu_agent());

    profiler_hub::writer_types::pmc_info_t pmc_desc{};
    pmc_desc.unique_id.name     = "kfd_page_fault";
    pmc_desc.unique_id.agent_id = agent_uid;
    pmc_desc.symbol             = "KFD_PAGE_FAULT";
    pmc_desc.description        = "KFD page fault counter";
    pmc_desc.node_id            = NODE_ID;
    pmc_desc.process_id         = PID;
    m_writer->register_pmc_info(pmc_desc);

    auto ev    = make_event(0, 0, 0, "kfd");
    ev.extdata = "{\"source\": \"kfd\"}";

    profiler_hub::writer_types::region_data_t region{};
    region.event           = ev;
    region.start_timestamp = 14000;
    region.end_timestamp   = 14500;
    region.name            = "KFD_PAGE_FAULT";

    profiler_hub::writer_types::arg_data_t arg0{};
    arg0.position = 0;
    arg0.type     = "std::uint64_t";
    arg0.name     = "address";
    arg0.value    = "0x7f4a00001000";
    region.args.push_back(arg0);

    profiler_hub::writer_types::arg_data_t arg1{};
    arg1.position = 1;
    arg1.type     = "string";
    arg1.name     = "agent";
    arg1.value    = "5";
    region.args.push_back(arg1);

    auto env = make_trace_env(NODE_ID, PID, THREAD_ID);
    m_writer->insert_region_data(region, env);

    profiler_hub::writer_types::track_info_t track{};
    track.name       = "KFD Events [GPU 0]";
    track.node_id    = NODE_ID;
    track.process_id = PID;
    track.thread_id  = THREAD_ID;
    m_writer->register_track_info(track);

    profiler_hub::writer_types::pmc_event_data_t pmc_data{};
    pmc_data.event = ev;
    pmc_data.value = 1.0;

    profiler_hub::writer_types::sample_data_t sample{};
    sample.timestamp = 14000;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid{};
    pmc_uid.name     = "kfd_page_fault";
    pmc_uid.agent_id = agent_uid;

    m_writer->insert_pmc_event_data(pmc_data, pmc_uid);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found_region = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::region)
            continue;

        auto detail = m_reader->get_region_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 14000U);
        EXPECT_EQ(detail->end_timestamp, 14500U);
        EXPECT_EQ(detail->name, "KFD_PAGE_FAULT");

        auto args = m_reader->get_arguments(tl_event);
        ASSERT_EQ(args.size(), 2U);
        EXPECT_EQ(args[0]->name, "address");
        EXPECT_EQ(args[0]->value, "0x7f4a00001000");
        EXPECT_EQ(args[1]->name, "agent");
        EXPECT_EQ(args[1]->value, "5");

        found_region = true;
        break;
    }
    EXPECT_TRUE(found_region) << "KFD region event not found in read-back";

    auto pmc_infos = m_reader->get_all_pmc_info();
    ASSERT_GE(pmc_infos.size(), 1U);

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// DB file creation: every handle pathway creates a valid DB
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, multiple_event_types_in_single_db)
{
    register_base_metadata();
    register_gpu_agent();
    register_queue_and_stream();
    register_code_object_and_kernel_symbol();

    auto env_full = make_trace_env_with_agent_queue_stream(
        NODE_ID, PID, THREAD_ID, gpu_agent(), QUEUE_ID, STREAM_ID);
    auto env_basic = make_trace_env(NODE_ID, PID, THREAD_ID);

    // Region (handle(region_sample) pathway)
    {
        auto ev = make_event(1, 0, 0, "HIP_API");

        profiler_hub::writer_types::region_data_t region{};
        region.event           = ev;
        region.start_timestamp = 1000;
        region.end_timestamp   = 1200;
        region.name            = "hipLaunchKernel";
        m_writer->insert_region_data(region, env_basic);
    }

    // Kernel dispatch (handle(kernel_dispatch_sample) pathway)
    {
        auto ev = make_event(2, 1, 0, "kernel_dispatch");

        profiler_hub::writer_types::kernel_dispatch_data_t kd{};
        kd.event            = ev;
        kd.dispatch_id      = 1;
        kd.start_timestamp  = 1500;
        kd.end_timestamp    = 2000;
        kd.kernel_symbol_id = 1;
        kd.code_object_id   = 1;
        kd.workgroup_size_x = 128;
        kd.workgroup_size_y = 1;
        kd.workgroup_size_z = 1;
        kd.grid_size_x      = 512;
        kd.grid_size_y      = 1;
        kd.grid_size_z      = 1;
        kd.name             = "test_kernel";
        m_writer->insert_kernel_dispatch_data(kd, env_full);
    }

    // Memory copy (handle(memory_copy_sample) pathway)
    {
        for(const auto& agent_obj : { cpu_agent(), gpu_agent() })
        {
            profiler_hub::writer_types::agent_info_t info{};
            info.unique_id    = make_agent_uid(agent_obj);
            info.name         = agent_obj.name;
            info.model_name   = agent_obj.model_name;
            info.vendor_name  = agent_obj.vendor_name;
            info.product_name = agent_obj.product_name;
            info.node_id      = NODE_ID;
            info.process_id   = PID;
            m_writer->register_agent_info(info);
        }

        auto ev = make_event(3, 0, 0, "memory_copy");

        profiler_hub::writer_types::memory_copy_data_t mc{};
        mc.event           = ev;
        mc.start_timestamp = 2500;
        mc.end_timestamp   = 3000;
        mc.dst_agent_id    = make_agent_uid(gpu_agent());
        mc.src_agent_id    = make_agent_uid(cpu_agent());
        mc.size            = 2048;
        mc.name            = "COPY_HOST_TO_DEVICE";
        mc.region_name     = "COPY_HOST_TO_DEVICE";

        auto mc_env      = env_basic;
        mc_env.stream_id = STREAM_ID;
        m_writer->insert_memory_copy_data(mc, mc_env);
    }

    // Memory alloc (handle(scratch_memory_sample) pathway)
    {
        auto ev = make_event(4, 0, 0, "scratch_memory");

        profiler_hub::writer_types::memory_alloc_data_t ma{};
        ma.event           = ev;
        ma.type            = "ALLOC";
        ma.level           = "SCRATCH";
        ma.start_timestamp = 3500;
        ma.end_timestamp   = 3600;
        ma.address         = 0;
        ma.size            = 32768;

        auto ma_env = make_trace_env_with_agent_queue_stream(
            NODE_ID, PID, THREAD_ID, gpu_agent(), QUEUE_ID, STREAM_ID);
        m_writer->insert_memory_alloc_data(ma, ma_env);
    }

    // Backtrace region (handle(backtrace_region_sample) pathway)
    {
        profiler_hub::writer_types::track_info_t sampling_track{};
        sampling_track.name       = "Sampling";
        sampling_track.node_id    = NODE_ID;
        sampling_track.process_id = PID;
        sampling_track.thread_id  = THREAD_ID;
        m_writer->register_track_info(sampling_track);

        auto ev    = make_event(0, 0, 0, "sampling");
        ev.extdata = "{}";

        profiler_hub::writer_types::region_data_t bt_region{};
        bt_region.event           = ev;
        bt_region.start_timestamp = 4000;
        bt_region.end_timestamp   = 4100;
        bt_region.name            = "bt_func";

        auto bt_env       = env_basic;
        bt_env.track_name = "Sampling";
        m_writer->insert_region_data(bt_region, bt_env);
    }

    flush_and_open_reader();

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

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0U);
}

// ---------------------------------------------------------------------------
// Region with call_stack: handle(region_sample) sets ev.call_stack
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_region_with_call_stack_pathway)
{
    register_base_metadata();

    auto ev = make_event(1, 0, 0, "HSA_API");
    ev.call_stack.push_back({});
    ev.extdata = "[{\"function\": \"main\"}]";

    profiler_hub::writer_types::region_data_t region{};
    region.event           = ev;
    region.start_timestamp = 15000;
    region.end_timestamp   = 15500;
    region.name            = "hsa_signal_wait";

    auto env = make_trace_env(NODE_ID, PID, THREAD_ID);
    m_writer->insert_region_data(region, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::region)
            continue;

        auto detail = m_reader->get_region_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->name, "hsa_signal_wait");
        EXPECT_EQ(detail->start_timestamp, 15000U);
        EXPECT_EQ(detail->end_timestamp, 15500U);

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Region with call_stack not found in read-back";
}

// ---------------------------------------------------------------------------
// Kernel dispatch with full grid: validate all dimension fields
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_kernel_dispatch_full_grid)
{
    register_base_metadata();
    register_gpu_agent();
    register_queue_and_stream();
    register_code_object_and_kernel_symbol();

    auto ev = make_event(10, 5, 99, "kernel_dispatch");

    profiler_hub::writer_types::kernel_dispatch_data_t kd{};
    kd.event                = ev;
    kd.dispatch_id          = 77;
    kd.start_timestamp      = 20000;
    kd.end_timestamp        = 25000;
    kd.kernel_symbol_id     = 1;
    kd.code_object_id       = 1;
    kd.private_segment_size = 512;
    kd.group_segment_size   = 16384;
    kd.workgroup_size_x     = 64;
    kd.workgroup_size_y     = 4;
    kd.workgroup_size_z     = 2;
    kd.grid_size_x          = 256;
    kd.grid_size_y          = 16;
    kd.grid_size_z          = 8;
    kd.name                 = "matmul_kernel";

    auto env = make_trace_env_with_agent_queue_stream(NODE_ID, PID, THREAD_ID,
                                                      gpu_agent(), QUEUE_ID, STREAM_ID);
    m_writer->insert_kernel_dispatch_data(kd, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    ASSERT_GE(events.size(), 1U);

    bool found = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::kernel_dispatch)
            continue;

        auto detail = m_reader->get_kernel_dispatch_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->dispatch_id, 77U);
        EXPECT_EQ(detail->start_timestamp, 20000U);
        EXPECT_EQ(detail->end_timestamp, 25000U);
        EXPECT_EQ(detail->workgroup_size_x, 64U);
        EXPECT_EQ(detail->workgroup_size_y, 4U);
        EXPECT_EQ(detail->workgroup_size_z, 2U);
        EXPECT_EQ(detail->grid_size_x, 256U);
        EXPECT_EQ(detail->grid_size_y, 16U);
        EXPECT_EQ(detail->grid_size_z, 8U);
        EXPECT_EQ(detail->name, "matmul_kernel");

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Full-grid kernel dispatch not found in read-back";

    EXPECT_TRUE(std::filesystem::exists(m_db_path));
}

// ---------------------------------------------------------------------------
// Memory copy with addresses: validate src/dst address fields
// ---------------------------------------------------------------------------

TEST_F(rocpd_write_read_test, handle_memory_copy_addresses_persisted)
{
    register_base_metadata();
    register_queue_and_stream();

    for(const auto& agent_obj : { cpu_agent(), gpu_agent() })
    {
        profiler_hub::writer_types::agent_info_t info{};
        info.unique_id    = make_agent_uid(agent_obj);
        info.name         = agent_obj.name;
        info.model_name   = agent_obj.model_name;
        info.vendor_name  = agent_obj.vendor_name;
        info.product_name = agent_obj.product_name;
        info.node_id      = NODE_ID;
        info.process_id   = PID;
        m_writer->register_agent_info(info);
    }

    auto ev = make_event(1, 0, 0, "memory_copy");

    profiler_hub::writer_types::memory_copy_data_t mc{};
    mc.event           = ev;
    mc.start_timestamp = 30000;
    mc.end_timestamp   = 31000;
    mc.dst_agent_id    = make_agent_uid(gpu_agent());
    mc.dst_address     = 0xDEAD0000;
    mc.src_agent_id    = make_agent_uid(cpu_agent());
    mc.src_address     = 0xBEEF0000;
    mc.size            = 1024;
    mc.name            = "COPY_DEVICE_TO_HOST";
    mc.region_name     = "COPY_DEVICE_TO_HOST";

    auto env      = make_trace_env(NODE_ID, PID, THREAD_ID);
    env.stream_id = STREAM_ID;
    m_writer->insert_memory_copy_data(mc, env);

    flush_and_open_reader();

    auto events = m_reader->get_events();
    bool found  = false;
    for(const auto& tl_event : events)
    {
        if(tl_event.unique_identifier.type !=
           profiler_hub::reader_types::event_type_t::memory_copy)
            continue;

        auto detail = m_reader->get_memory_copy_details(tl_event);
        ASSERT_TRUE(detail.has_value());

        EXPECT_EQ(detail->start_timestamp, 30000U);
        EXPECT_EQ(detail->end_timestamp, 31000U);
        EXPECT_EQ(detail->size, 1024U);
        EXPECT_EQ(detail->name, "COPY_DEVICE_TO_HOST");

        ASSERT_TRUE(detail->dst_address.has_value());
        EXPECT_EQ(detail->dst_address.value(), 0xDEAD0000U);
        ASSERT_TRUE(detail->src_address.has_value());
        EXPECT_EQ(detail->src_address.value(), 0xBEEF0000U);

        found = true;
        break;
    }
    EXPECT_TRUE(found) << "Memory copy with addresses not found in read-back";
}
