// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <benchmark/benchmark.h>

#include "utility.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace profiler_hub::writer_types;

// ============================================================================
// Benchmark Configuration
// ============================================================================

// Configurable event counts for parallel scaling tests
constexpr size_t COUNT_10K   = 10000;
constexpr size_t COUNT_100K  = 100000;
constexpr size_t COUNT_1000K = 1000000;

// ============================================================================
// Per-Thread Context
// Each thread gets its own isolated database to avoid contention
// ============================================================================

struct thread_context
{
    std::string                              database_path;
    std::unique_ptr<profiler_hub::storage_t> storage;
    std::shared_ptr<profiler_hub::writer_t>  writer;
    size_t                                   context_id = 0;
    trace_environment_t                      trace_env;
    agent_unique_id_t                        gpu_agent_id;

    void setup(size_t thread_id)
    {
        context_id    = thread_id;
        database_path = "benchmark_writer_parallel_" + std::to_string(thread_id) + ".db";

        const std::string uuid = std::to_string(thread_id);
        storage = std::make_unique<profiler_hub::storage_t>(database_path, uuid);
        writer  = std::make_shared<profiler_hub::writer_t>(std::move(storage));

        constexpr size_t node_id = 1;
        const size_t     pid     = 1000 + thread_id;
        const size_t     tid     = 2000 + thread_id;

        // Register infrastructure
        writer->register_node_info(node_info_t{ .node_id       = node_id,
                                                .hash          = 0xDEADBEEF + thread_id,
                                                .machine_id    = "bench-machine",
                                                .system_name   = "Linux",
                                                .hostname      = "benchmark-host",
                                                .release       = "6.0.0",
                                                .version       = "v1",
                                                .hardware_name = "x86_64",
                                                .domain_name   = "local" });

        writer->register_process_info(process_info_t{ .ppid        = 0,
                                                      .pid         = pid,
                                                      .init        = 0,
                                                      .fini        = 0,
                                                      .start       = 0,
                                                      .end         = 0,
                                                      .command     = "/bin/benchmark",
                                                      .environment = "{}",
                                                      .extdata     = "{}",
                                                      .node_id     = node_id });

        writer->register_thread_info(thread_info_t{ .parent_process_id = pid,
                                                    .thread_id         = tid,
                                                    .name              = "worker",
                                                    .start             = 0,
                                                    .end               = 0,
                                                    .extdata           = "{}",
                                                    .node_id           = node_id,
                                                    .process_id        = pid });

        gpu_agent_id = agent_unique_id_t{ .agent_type = "GPU", .type_index = 0 };

        writer->register_agent_info(agent_info_t{ .unique_id      = gpu_agent_id,
                                                  .absolute_index = 0,
                                                  .logical_index  = 0,
                                                  .uuid           = 0xABCD,
                                                  .name           = "gfx90a",
                                                  .model_name     = "MI200",
                                                  .vendor_name    = "AMD",
                                                  .product_name   = "MI210",
                                                  .user_name      = "",
                                                  .extdata        = "{}",
                                                  .node_id        = node_id,
                                                  .process_id     = pid });

        writer->register_queue_info(queue_info_t{ .queue_id   = 1,
                                                  .name       = "hsa_queue_0",
                                                  .extdata    = "{}",
                                                  .node_id    = node_id,
                                                  .process_id = pid });

        writer->register_stream_info(stream_info_t{ .stream_id  = 1,
                                                    .name       = "hip_stream_0",
                                                    .extdata    = "{}",
                                                    .node_id    = node_id,
                                                    .process_id = pid });

        writer->register_code_object_info(
            code_object_info_t{ .id           = 1,
                                .uri          = "file:///kernels.hsaco",
                                .load_base    = 0x10000,
                                .load_size    = 0x1000,
                                .load_delta   = 0,
                                .storage_type = "FILE",
                                .extdata      = "{}",
                                .node_id      = node_id,
                                .process_id   = pid,
                                .agent_id     = gpu_agent_id });

        writer->register_kernel_symbol_info(
            kernel_symbol_info_t{ .id            = 1,
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
                                  .code_obj_id               = 1 });

        writer->register_track_info(track_info_t{ .name       = "gpu_kernel",
                                                  .extdata    = "{}",
                                                  .node_id    = node_id,
                                                  .process_id = pid,
                                                  .thread_id  = tid });

        // Store trace environment for data insertion
        trace_env = trace_environment_t{ .node_id    = node_id,
                                         .process_id = pid,
                                         .thread_id  = tid,
                                         .agent_id   = gpu_agent_id,
                                         .stream_id  = 1,
                                         .queue_id   = 1,
                                         .track_name = "gpu_kernel" };
    }

    void teardown()
    {
        writer.reset();
        storage.reset();
        std::remove(database_path.c_str());
    }

    [[nodiscard]] size_t get_db_size() const
    {
        return utility::get_file_size(database_path);
    }
};

// ============================================================================
// Realistic Workload Generator
// Simulates typical profiling workload with regions and kernel dispatches
// ============================================================================

void
run_realistic_workload(thread_context& ctx, size_t total_events)
{
    // Distribution: 99.67% regions, 0.33% kernel dispatches
    const size_t region_count = total_events * 9967 / 10000;
    const size_t kernel_count = total_events * 33 / 10000;

    // Insert regions (majority of workload) - no args for simplicity
    for(size_t i = 0; i < region_count; ++i)
    {
        const region_data_t data{ .event           = std::nullopt,
                                  .start_timestamp = i * 1000,
                                  .end_timestamp   = i * 1000 + 500,
                                  .name            = "user_region",
                                  .extdata         = "{}",
                                  .args            = {} };

        ctx.writer->insert_region_data(data, ctx.trace_env);
    }

    // Insert kernel dispatches
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
        ctx.writer->insert_kernel_dispatch_data(data, ctx.trace_env);
    }

    ctx.writer->flush_in_memory_data_to_disk();
}

// ============================================================================
// Parallel Writer Benchmark Fixture
// ============================================================================

class parallel_writer_fixture : public benchmark::Fixture
{
public:
    void SetUp(const benchmark::State&) override {}
    void TearDown(const benchmark::State&) override {}
};

// ============================================================================
// Scaling Benchmark
// Measures parallel scalability with isolated databases per thread
// All times should be similar regardless of thread count (no contention)
// ============================================================================

BENCHMARK_DEFINE_F(parallel_writer_fixture, scaling_test)(benchmark::State& state)
{
    const auto num_threads  = static_cast<size_t>(state.range(0));
    const auto total_events = static_cast<size_t>(state.range(1));

    for(auto _ : state)
    {
        std::vector<thread_context> contexts(num_threads);
        std::vector<std::thread>    threads;

        // Setup phase (not measured - done sequentially)
        for(size_t i = 0; i < num_threads; ++i)
        {
            contexts[i].setup(i);
        }

        // Measure parallel workload execution
        auto start_time = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < num_threads; ++i)
        {
            threads.emplace_back([&contexts, i, total_events]() {
                run_realistic_workload(contexts[i], total_events);
            });
        }

        for(auto& t : threads)
        {
            t.join();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
                .count();

        // Collect metrics
        size_t total_db_size = 0;
        for(size_t i = 0; i < num_threads; ++i)
        {
            total_db_size += contexts[i].get_db_size();
        }

        // Cleanup
        for(size_t i = 0; i < num_threads; ++i)
        {
            contexts[i].teardown();
        }

        state.SetIterationTime(static_cast<double>(duration) / 1000.0);
        state.counters["total_db_size_mb"] =
            static_cast<double>(total_db_size) / (1024.0 * 1024.0);
        state.counters["total_events"] = static_cast<double>(total_events);
        state.counters["events_per_sec"] =
            static_cast<double>(total_events) / (static_cast<double>(duration) / 1000.0);
    }

    state.SetItemsProcessed(state.iterations() * total_events);
}

BENCHMARK_REGISTER_F(parallel_writer_fixture, scaling_test)
    ->Unit(benchmark::kSecond)
    ->UseManualTime()
    ->Iterations(1)
    // in_memory: 10k events - quick validation
    ->Args({ 1, COUNT_10K })
    ->Args({ 2, COUNT_10K })
    ->Args({ 4, COUNT_10K })
    ->Args({ 8, COUNT_10K })
    // in_memory: 100k events - standard benchmark
    ->Args({ 1, COUNT_100K })
    ->Args({ 2, COUNT_100K })
    ->Args({ 4, COUNT_100K })
    ->Args({ 8, COUNT_100K })
    // in_memory: 1000k events - stress test
    ->Args({ 1, COUNT_1000K })
    ->Args({ 2, COUNT_1000K })
    ->Args({ 4, COUNT_1000K })
    ->Args({ 8, COUNT_1000K });

}  // namespace
