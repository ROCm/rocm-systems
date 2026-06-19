// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"
#include <profiler-hub/version.hpp>

#include <benchmark/benchmark.h>

#include "utility.hpp"

#include <cstdio>
#include <memory>
#include <string>

namespace
{

using namespace profiler_hub::writer_types;

// ============================================================================
// Benchmark Configuration
// ============================================================================

// Standard data insert counts for benchmarking
// Use ->Arg() to select: 10k (quick), 100k (standard), 1000k (stress)
constexpr size_t COUNT_10K   = 10000;
constexpr size_t COUNT_100K  = 100000;
constexpr size_t COUNT_1000K = 1000000;

// ============================================================================
// Test Data Generators
// ============================================================================

node_info_t
make_node_info(size_t node_id) noexcept
{
    return node_info_t{ .node_id       = node_id,
                        .hash          = 0xDEADBEEF + node_id,
                        .machine_id    = "bench-machine",
                        .system_name   = "Linux",
                        .hostname      = "benchmark-host",
                        .release       = "6.0.0",
                        .version       = "v1",
                        .hardware_name = "x86_64",
                        .domain_name   = "local" };
}

process_info_t
make_process_info(size_t node_id, size_t pid) noexcept
{
    return process_info_t{ .ppid        = 0,
                           .pid         = pid,
                           .init        = 0,
                           .fini        = 0,
                           .start       = 0,
                           .end         = 0,
                           .command     = "/bin/benchmark",
                           .environment = "{}",
                           .extdata     = "{}",
                           .node_id     = node_id };
}

thread_info_t
make_thread_info(size_t node_id, size_t pid, size_t tid) noexcept
{
    return thread_info_t{ .parent_process_id = pid,
                          .thread_id         = tid,
                          .name              = "main",
                          .start             = 0,
                          .end               = 0,
                          .extdata           = "{}",
                          .node_id           = node_id,
                          .process_id        = pid };
}

agent_info_t
make_gpu_agent(size_t node_id, size_t pid, size_t type_index = 0) noexcept
{
    return agent_info_t{ .unique_id = { .agent_type = "GPU", .type_index = type_index },
                         .absolute_index = type_index,
                         .logical_index  = type_index,
                         .uuid           = 0xABCD + type_index,
                         .name           = "gfx90a",
                         .model_name     = "MI200",
                         .vendor_name    = "AMD",
                         .product_name   = "MI210",
                         .user_name      = "",
                         .extdata        = "{}",
                         .node_id        = node_id,
                         .process_id     = pid };
}

agent_info_t
make_cpu_agent(size_t node_id, size_t pid, size_t type_index = 0) noexcept
{
    return agent_info_t{ .unique_id = { .agent_type = "CPU", .type_index = type_index },
                         .absolute_index = type_index,
                         .logical_index  = type_index,
                         .uuid           = 0,
                         .name           = "cpu0",
                         .model_name     = "EPYC",
                         .vendor_name    = "AMD",
                         .product_name   = "EPYC7763",
                         .user_name      = "",
                         .extdata        = "{}",
                         .node_id        = node_id,
                         .process_id     = pid };
}

queue_info_t
make_queue_info(size_t node_id, size_t pid, size_t queue_id) noexcept
{
    return queue_info_t{ .queue_id   = queue_id,
                         .name       = "hsa_queue_0",
                         .extdata    = "{}",
                         .node_id    = node_id,
                         .process_id = pid };
}

stream_info_t
make_stream_info(size_t node_id, size_t pid, size_t stream_id) noexcept
{
    return stream_info_t{ .stream_id  = stream_id,
                          .name       = "hip_stream_0",
                          .extdata    = "{}",
                          .node_id    = node_id,
                          .process_id = pid };
}

code_object_info_t
make_code_object(size_t            id,
                 size_t            node_id,
                 size_t            pid,
                 agent_unique_id_t agent_id) noexcept
{
    return code_object_info_t{ .id           = id,
                               .uri          = "file:///kernels.hsaco",
                               .load_base    = 0x10000,
                               .load_size    = 0x1000,
                               .load_delta   = 0,
                               .storage_type = "FILE",
                               .extdata      = "{}",
                               .node_id      = node_id,
                               .process_id   = pid,
                               .agent_id     = agent_id };
}

kernel_symbol_info_t
make_kernel_symbol(size_t id, size_t node_id, size_t pid, size_t code_obj_id) noexcept
{
    return kernel_symbol_info_t{ .id            = id,
                                 .name          = "vectorAdd",
                                 .display_name  = "vectorAdd(float*,float*,float*,int)",
                                 .kernel_object = 0x1234,
                                 .kernarg_segment_size      = 256,
                                 .kernarg_segment_alignment = 8,
                                 .group_segment_size        = 65536,
                                 .private_segment_size      = 0,
                                 .sgpr_count                = 32,
                                 .arch_vgpr_count           = 64,
                                 .accum_vgpr_count          = 0,
                                 .extdata                   = "{}",
                                 .node_id                   = node_id,
                                 .process_id                = pid,
                                 .code_obj_id               = code_obj_id };
}

pmc_info_t
make_pmc_info(size_t                           node_id,
              size_t                           pid,
              const char*                      name,
              std::optional<agent_unique_id_t> agent_id = std::nullopt) noexcept
{
    return pmc_info_t{ .unique_id        = { .name = name, .agent_id = agent_id },
                       .target_arch      = "GPU",
                       .event_code       = 0x100,
                       .instance_id      = 0,
                       .symbol           = name,
                       .description      = name,
                       .long_description = name,
                       .component        = "SQ",
                       .units            = "value",
                       .value_type       = "ABS",
                       .block            = "SQ",
                       .expression       = "",
                       .is_constant      = 0,
                       .is_derived       = 0,
                       .extdata          = "{}",
                       .node_id          = node_id,
                       .process_id       = pid };
}

track_info_t
make_track_info(const char*           name,
                size_t                node_id,
                std::optional<size_t> pid = std::nullopt,
                std::optional<size_t> tid = std::nullopt) noexcept
{
    return track_info_t{ .name       = name,
                         .extdata    = "{}",
                         .node_id    = node_id,
                         .process_id = pid,
                         .thread_id  = tid };
}

// ============================================================================
// Benchmark Fixture
// ============================================================================

class writer_fixture : public benchmark::Fixture
{
public:
    void SetUp(const benchmark::State& state) override
    {
        m_database_path =
            "benchmark_writer_" + std::to_string(state.thread_index()) + ".db";
        m_uuid    = std::to_string(state.thread_index());
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, m_uuid);
        m_writer  = std::make_shared<profiler_hub::writer_t>(std::move(m_storage));
        setup_schema();
    }

    void TearDown(const benchmark::State&) override
    {
        m_writer.reset();
        m_storage.reset();
        std::remove(m_database_path.c_str());
    }

    [[nodiscard]] std::string get_db_size_label() const
    {
        const size_t size = utility::get_file_size(m_database_path);
        return " DB: " + utility::format_file_size(size);
    }

protected:
    void setup_schema()
    {
        constexpr size_t node_id = 1;
        constexpr size_t pid     = 1000;
        constexpr size_t tid     = 1001;

        // Register infrastructure (not measured - this is setup cost)
        m_writer->register_node_info(make_node_info(node_id));
        m_writer->register_process_info(make_process_info(node_id, pid));
        m_writer->register_thread_info(make_thread_info(node_id, pid, tid));

        m_writer->register_agent_info(make_gpu_agent(node_id, pid, 0));
        m_writer->register_agent_info(make_cpu_agent(node_id, pid, 0));

        m_writer->register_queue_info(make_queue_info(node_id, pid, 1));
        m_writer->register_stream_info(make_stream_info(node_id, pid, 1));

        const agent_unique_id_t gpu_agent{ "GPU", 0 };
        m_writer->register_code_object_info(make_code_object(1, node_id, pid, gpu_agent));
        m_writer->register_kernel_symbol_info(make_kernel_symbol(1, node_id, pid, 1));

        // Register PMC descriptions
        m_writer->register_pmc_info(make_pmc_info(node_id, pid, "SQ_WAVES", gpu_agent));
        m_writer->register_pmc_info(make_pmc_info(node_id, pid, "SQ_INSTS", gpu_agent));
        m_writer->register_pmc_info(make_pmc_info(node_id, pid, "TA_BUSY", gpu_agent));

        // Register tracks
        m_writer->register_track_info(make_track_info("gpu_kernel", node_id, pid, tid));
        m_writer->register_track_info(make_track_info("gpu_memcpy", node_id, pid, tid));
        m_writer->register_track_info(make_track_info("cpu_sample", node_id, pid, tid));
        m_writer->register_track_info(
            make_track_info("amd_smi", node_id, std::nullopt, std::nullopt));

        // Store commonly used values
        m_trace_env = trace_environment_t{ .node_id    = node_id,
                                           .process_id = pid,
                                           .thread_id  = tid,
                                           .agent_id   = gpu_agent,
                                           .stream_id  = 1,
                                           .queue_id   = 1,
                                           .track_name = "gpu_kernel" };

        m_gpu_agent_id = gpu_agent;
    }

    std::string                              m_database_path;
    std::string                              m_uuid;
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::writer_t>  m_writer;
    trace_environment_t                      m_trace_env;
    agent_unique_id_t                        m_gpu_agent_id;
};

// ============================================================================
// Registration Benchmarks (Setup Cost)
// These benchmarks measure the cost of registering metadata
// ============================================================================

class registration_fixture : public benchmark::Fixture
{
public:
    void SetUp(const benchmark::State& state) override
    {
        m_database_path =
            "benchmark_registration_" + std::to_string(state.thread_index()) + ".db";
        m_uuid    = std::to_string(state.thread_index());
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, m_uuid);
        m_writer  = std::make_shared<profiler_hub::writer_t>(std::move(m_storage));
    }

    void TearDown(const benchmark::State&) override
    {
        m_writer.reset();
        m_storage.reset();
        std::remove(m_database_path.c_str());
    }

    [[nodiscard]] std::string get_db_size_label() const
    {
        const size_t size = utility::get_file_size(m_database_path);
        return " DB: " + utility::format_file_size(size);
    }

protected:
    std::string                              m_database_path;
    std::string                              m_uuid;
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::writer_t>  m_writer;
};

// Benchmark: Register multiple threads
BENCHMARK_DEFINE_F(registration_fixture, register_threads)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    constexpr size_t node_id = 1;
    constexpr size_t pid     = 1000;

    // Register required dependencies first
    m_writer->register_node_info(make_node_info(node_id));
    m_writer->register_process_info(make_process_info(node_id, pid));

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            m_writer->register_thread_info(make_thread_info(node_id, pid, 2000 + i));
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
}

BENCHMARK_REGISTER_F(registration_fixture, register_threads)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K });

// Benchmark: Register multiple kernel symbols
BENCHMARK_DEFINE_F(registration_fixture, register_kernel_symbols)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    constexpr size_t node_id = 1;
    constexpr size_t pid     = 1000;

    // Register required dependencies
    m_writer->register_node_info(make_node_info(node_id));
    m_writer->register_process_info(make_process_info(node_id, pid));

    const agent_unique_id_t gpu_agent{ "GPU", 0 };
    m_writer->register_agent_info(make_gpu_agent(node_id, pid, 0));
    m_writer->register_code_object_info(make_code_object(1, node_id, pid, gpu_agent));

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            m_writer->register_kernel_symbol_info(
                make_kernel_symbol(i + 1, node_id, pid, 1));
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
}

BENCHMARK_REGISTER_F(registration_fixture, register_kernel_symbols)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K });

// Benchmark: Register PMC descriptions
BENCHMARK_DEFINE_F(registration_fixture, register_pmc_info)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    constexpr size_t node_id = 1;
    constexpr size_t pid     = 1000;

    // Register required dependencies
    m_writer->register_node_info(make_node_info(node_id));
    m_writer->register_process_info(make_process_info(node_id, pid));

    const agent_unique_id_t gpu_agent{ "GPU", 0 };
    m_writer->register_agent_info(make_gpu_agent(node_id, pid, 0));

    // Generate unique PMC names
    std::vector<std::string> pmc_names;
    pmc_names.reserve(count);
    for(size_t i = 0; i < count; ++i)
    {
        pmc_names.push_back("PMC_COUNTER_" + std::to_string(i));
    }

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            m_writer->register_pmc_info(
                make_pmc_info(node_id, pid, pmc_names[i].c_str(), gpu_agent));
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
}

BENCHMARK_REGISTER_F(registration_fixture, register_pmc_info)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K });

// Benchmark: Full schema registration (realistic setup)
BENCHMARK_DEFINE_F(registration_fixture, register_full_schema)(benchmark::State& state)
{
    const auto thread_count = static_cast<size_t>(state.range(0));
    const auto kernel_count = static_cast<size_t>(state.range(1));

    for(auto _ : state)
    {
        constexpr size_t node_id = 1;
        constexpr size_t pid     = 1000;

        // Infrastructure
        m_writer->register_node_info(make_node_info(node_id));
        m_writer->register_process_info(make_process_info(node_id, pid));

        // Threads
        for(size_t i = 0; i < thread_count; ++i)
        {
            m_writer->register_thread_info(make_thread_info(node_id, pid, 2000 + i));
        }

        // GPU agent
        const agent_unique_id_t gpu_agent{ "GPU", 0 };
        m_writer->register_agent_info(make_gpu_agent(node_id, pid, 0));
        m_writer->register_queue_info(make_queue_info(node_id, pid, 1));
        m_writer->register_stream_info(make_stream_info(node_id, pid, 1));

        // Code objects and kernels
        m_writer->register_code_object_info(make_code_object(1, node_id, pid, gpu_agent));
        for(size_t i = 0; i < kernel_count; ++i)
        {
            m_writer->register_kernel_symbol_info(
                make_kernel_symbol(i + 1, node_id, pid, 1));
        }

        // Tracks
        for(size_t i = 0; i < thread_count; ++i)
        {
            m_writer->register_track_info(
                make_track_info("track", node_id, pid, 2000 + i));
        }

        m_writer->flush_in_memory_data_to_disk();
    }

    const size_t total_registrations =
        1 + 1 + thread_count + 1 + 1 + 1 + 1 + kernel_count + thread_count;
    state.SetItemsProcessed(state.iterations() * total_registrations);
    state.counters["threads"] = static_cast<double>(thread_count);
    state.counters["kernels"] = static_cast<double>(kernel_count);
}

BENCHMARK_REGISTER_F(registration_fixture, register_full_schema)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    // Small setup (12 threads, 100 kernels)
    ->Args({ 12, 100 })
    // Medium setup (64 threads, 1000 kernels)
    ->Args({ 64, 1000 })
    // Large setup (256 threads, 10000 kernels)
    ->Args({ 256, 10000 });

// ============================================================================
// Data Insertion Benchmarks (Hot Path - Primary Focus)
// ============================================================================

BENCHMARK_DEFINE_F(writer_fixture, kernel_dispatch)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            const kernel_dispatch_data_t data{ .event                = std::nullopt,
                                               .dispatch_id          = i,
                                               .start_timestamp      = i * 1000,
                                               .end_timestamp        = i * 1000 + 500,
                                               .kernel_symbol_id     = 1,
                                               .code_object_id       = 1,
                                               .private_segment_size = 0,
                                               .group_segment_size   = 65536,
                                               .workgroup_size_x     = 256,
                                               .workgroup_size_y     = 1,
                                               .workgroup_size_z     = 1,
                                               .grid_size_x          = 1024,
                                               .grid_size_y          = 1,
                                               .grid_size_z          = 1,
                                               .name                 = "vectorAdd",
                                               .extdata              = "{}" };
            m_writer->insert_kernel_dispatch_data(data, m_trace_env);
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetLabel(get_db_size_label());
}

BENCHMARK_REGISTER_F(writer_fixture, kernel_dispatch)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K })
    ->Args({ COUNT_1000K });

BENCHMARK_DEFINE_F(writer_fixture, memory_copy)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            const memory_copy_data_t data{ .event           = std::nullopt,
                                           .start_timestamp = i * 1000,
                                           .end_timestamp   = i * 1000 + 200,
                                           .dst_agent_id    = m_gpu_agent_id,
                                           .dst_address     = 0x7FFF00000000 + i * 4096,
                                           .src_agent_id    = std::nullopt,
                                           .src_address     = 0x100000 + i * 4096,
                                           .size            = 4096,
                                           .name            = "hipMemcpy",
                                           .region_name     = std::nullopt,
                                           .extdata         = "{}" };
            m_writer->insert_memory_copy_data(data, m_trace_env);
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetLabel(get_db_size_label());
}

BENCHMARK_REGISTER_F(writer_fixture, memory_copy)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K })
    ->Args({ COUNT_1000K });

BENCHMARK_DEFINE_F(writer_fixture, memory_alloc)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            const memory_alloc_data_t data{ .event           = std::nullopt,
                                            .type            = "ALLOC",
                                            .level           = "REAL",
                                            .start_timestamp = i * 1000,
                                            .end_timestamp   = i * 1000 + 100,
                                            .address         = 0x7FFF00000000 + i * 4096,
                                            .size            = 4096,
                                            .extdata         = "{}" };
            m_writer->insert_memory_alloc_data(data, m_trace_env);
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetLabel(get_db_size_label());
}

BENCHMARK_REGISTER_F(writer_fixture, memory_alloc)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K })
    ->Args({ COUNT_1000K });

BENCHMARK_DEFINE_F(writer_fixture, region)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            const region_data_t data{ .event           = std::nullopt,
                                      .start_timestamp = i * 500,
                                      .end_timestamp   = i * 500 + 100,
                                      .name            = "user_region",
                                      .extdata         = "{}",
                                      .args            = {} };
            m_writer->insert_region_data(data, m_trace_env);
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetLabel(get_db_size_label());
}

BENCHMARK_REGISTER_F(writer_fixture, region)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K })
    ->Args({ COUNT_1000K });

BENCHMARK_DEFINE_F(writer_fixture, region_with_args)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    // Pre-create argument data
    const std::vector<arg_data_t> args = {
        { .position = 0, .type = "int", .name = "level", .value = "0", .extdata = "{}" },
        { .position = 1,
          .type     = "char*",
          .name     = "name",
          .value    = "region",
          .extdata  = "{}" },
        { .position = 2,
          .type     = "size_t",
          .name     = "id",
          .value    = "12345",
          .extdata  = "{}" },
        { .position = 3,
          .type     = "void*",
          .name     = "ptr",
          .value    = "0x7fff000",
          .extdata  = "{}" },
        { .position = 4,
          .type     = "double",
          .name     = "value",
          .value    = "3.14159",
          .extdata  = "{}" }
    };

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            // Variable number of arguments (0-5) based on iteration
            const size_t            num_args = i % 6;
            std::vector<arg_data_t> iter_args(args.begin(), args.begin() + num_args);

            // Create event data when args are present (required for correlation)
            std::optional<event_data_t> event_opt = std::nullopt;
            if(!iter_args.empty())
            {
                event_opt = event_data_t{ .stack_id        = i,
                                          .parent_stack_id = 0,
                                          .correlation_id  = i,
                                          .call_stack      = {},
                                          .line_info_list  = {},
                                          .event_category  = "HIP_API",
                                          .extdata         = "{}" };
            }

            const region_data_t data{ .event           = std::move(event_opt),
                                      .start_timestamp = i * 500,
                                      .end_timestamp   = i * 500 + 100,
                                      .name            = "api_call",
                                      .extdata         = "{}",
                                      .args            = std::move(iter_args) };
            m_writer->insert_region_data(data, m_trace_env);
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetLabel(get_db_size_label());
}

BENCHMARK_REGISTER_F(writer_fixture, region_with_args)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K })
    ->Args({ COUNT_1000K });

BENCHMARK_DEFINE_F(writer_fixture, pmc_event)(benchmark::State& state)
{
    const auto count = static_cast<size_t>(state.range(0));

    const pmc_info_unique_id_t pmc_id{ .name = "SQ_WAVES", .agent_id = m_gpu_agent_id };
    const track_info_t         track = make_track_info("gpu_kernel", 1, 1000, 1001);

    for(auto _ : state)
    {
        for(size_t i = 0; i < count; ++i)
        {
            const pmc_event_data_t data{
                .event   = std::nullopt,
                .value   = static_cast<double>(i * 100),
                .extdata = "{}",
                .sample  = { .timestamp = i * 1000, .track = track, .extdata = "{}" }
            };
            m_writer->insert_pmc_event_data(data, pmc_id);
        }
        m_writer->flush_in_memory_data_to_disk();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetLabel(get_db_size_label());
}

BENCHMARK_REGISTER_F(writer_fixture, pmc_event)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K })
    ->Args({ COUNT_1000K });

// ============================================================================
// End-to-End Benchmark (Realistic Mixed Workload)
// ============================================================================

BENCHMARK_DEFINE_F(writer_fixture, end_to_end_mixed)(benchmark::State& state)
{
    const auto total_events = static_cast<size_t>(state.range(0));

    // Realistic distribution: 70% regions, 20% kernel dispatches, 10% memory ops
    const size_t region_count   = total_events * 70 / 100;
    const size_t kernel_count   = total_events * 20 / 100;
    const size_t memcopy_count  = total_events * 5 / 100;
    const size_t memalloc_count = total_events * 5 / 100;

    for(auto _ : state)
    {
        // Regions (most common)
        for(size_t i = 0; i < region_count; ++i)
        {
            const region_data_t data{ .event           = std::nullopt,
                                      .start_timestamp = i * 500,
                                      .end_timestamp   = i * 500 + 100,
                                      .name            = "hip_api",
                                      .extdata         = "{}",
                                      .args            = {} };
            m_writer->insert_region_data(data, m_trace_env);
        }

        // Kernel dispatches
        for(size_t i = 0; i < kernel_count; ++i)
        {
            const kernel_dispatch_data_t data{ .event                = std::nullopt,
                                               .dispatch_id          = i,
                                               .start_timestamp      = i * 10000,
                                               .end_timestamp        = i * 10000 + 5000,
                                               .kernel_symbol_id     = 1,
                                               .code_object_id       = 1,
                                               .private_segment_size = 0,
                                               .group_segment_size   = 65536,
                                               .workgroup_size_x     = 256,
                                               .workgroup_size_y     = 1,
                                               .workgroup_size_z     = 1,
                                               .grid_size_x          = 1024,
                                               .grid_size_y          = 1,
                                               .grid_size_z          = 1,
                                               .name                 = "vectorAdd",
                                               .extdata              = "{}" };
            m_writer->insert_kernel_dispatch_data(data, m_trace_env);
        }

        // Memory copies
        for(size_t i = 0; i < memcopy_count; ++i)
        {
            const memory_copy_data_t data{ .event           = std::nullopt,
                                           .start_timestamp = i * 2000,
                                           .end_timestamp   = i * 2000 + 500,
                                           .dst_agent_id    = m_gpu_agent_id,
                                           .dst_address     = 0x7FFF00000000 + i * 4096,
                                           .src_agent_id    = std::nullopt,
                                           .src_address     = 0x100000 + i * 4096,
                                           .size            = 4096,
                                           .name            = "hipMemcpy",
                                           .region_name     = std::nullopt,
                                           .extdata         = "{}" };
            m_writer->insert_memory_copy_data(data, m_trace_env);
        }

        // Memory allocations
        for(size_t i = 0; i < memalloc_count; ++i)
        {
            const memory_alloc_data_t data{ .event           = std::nullopt,
                                            .type            = "ALLOC",
                                            .level           = "REAL",
                                            .start_timestamp = i * 1000,
                                            .end_timestamp   = i * 1000 + 100,
                                            .address         = 0x7FFF00000000 + i * 4096,
                                            .size            = 4096,
                                            .extdata         = "{}" };
            m_writer->insert_memory_alloc_data(data, m_trace_env);
        }

        m_writer->flush_in_memory_data_to_disk();
    }

    state.SetItemsProcessed(state.iterations() * total_events);
    state.SetLabel(get_db_size_label());
    state.counters["regions"]   = static_cast<double>(region_count);
    state.counters["kernels"]   = static_cast<double>(kernel_count);
    state.counters["memcopies"] = static_cast<double>(memcopy_count);
    state.counters["memallocs"] = static_cast<double>(memalloc_count);
}

BENCHMARK_REGISTER_F(writer_fixture, end_to_end_mixed)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->Args({ COUNT_10K })
    ->Args({ COUNT_100K })
    ->Args({ COUNT_1000K });

}  // namespace
