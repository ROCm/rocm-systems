// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Standalone CLI benchmark for kernel_dispatch writes. Used to compare
// baseline-vs-WAL throughput. Usage: bench_write <output.db> [N=100000]

#include "rocpdsna/storage.hpp"
#include "rocpdsna/storage_types.hpp"
#include "rocpdsna/writer.hpp"
#include "rocpdsna/writer_types.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace rocpdsna::writer_types;

namespace
{

constexpr size_t k_default_count = 100000;
constexpr size_t k_node_id       = 1;
constexpr size_t k_pid           = 1000;
constexpr size_t k_tid           = 1001;

void
register_setup(rocpdsna::writer_t& writer, agent_unique_id_t& gpu_agent_out)
{
    writer.register_node_info(node_info_t{ .node_id       = k_node_id,
                                           .hash          = 0xDEADBEEFULL,
                                           .machine_id    = "bench-machine",
                                           .system_name   = "Linux",
                                           .hostname      = "bench-host",
                                           .release       = "6.0.0",
                                           .version       = "v1",
                                           .hardware_name = "x86_64",
                                           .domain_name   = "local" });

    writer.register_process_info(process_info_t{ .ppid        = 0,
                                                 .pid         = k_pid,
                                                 .init        = 0,
                                                 .fini        = 0,
                                                 .start       = 0,
                                                 .end         = 0,
                                                 .command     = "/bin/bench_write",
                                                 .environment = "{}",
                                                 .extdata     = "{}",
                                                 .node_id     = k_node_id });

    writer.register_thread_info(thread_info_t{ .parent_process_id = k_pid,
                                               .thread_id         = k_tid,
                                               .name              = "main",
                                               .start             = 0,
                                               .end               = 0,
                                               .extdata           = "{}",
                                               .node_id           = k_node_id,
                                               .process_id        = k_pid });

    const agent_unique_id_t gpu_agent{ "GPU", 0 };
    writer.register_agent_info(agent_info_t{ .unique_id      = gpu_agent,
                                             .absolute_index = 0,
                                             .logical_index  = 0,
                                             .uuid           = 0xABCDULL,
                                             .name           = "gfx90a",
                                             .model_name     = "MI200",
                                             .vendor_name    = "AMD",
                                             .product_name   = "MI210",
                                             .user_name      = "",
                                             .extdata        = "{}",
                                             .node_id        = k_node_id,
                                             .process_id     = k_pid });

    writer.register_queue_info(queue_info_t{ .queue_id   = 1,
                                             .name       = "hsa_queue_0",
                                             .extdata    = "{}",
                                             .node_id    = k_node_id,
                                             .process_id = k_pid });

    writer.register_stream_info(stream_info_t{ .stream_id  = 1,
                                               .name       = "hip_stream_0",
                                               .extdata    = "{}",
                                               .node_id    = k_node_id,
                                               .process_id = k_pid });

    writer.register_code_object_info(code_object_info_t{ .id  = 1,
                                                         .uri = "file:///kernels.hsaco",
                                                         .load_base    = 0x10000,
                                                         .load_size    = 0x1000,
                                                         .load_delta   = 0,
                                                         .storage_type = "FILE",
                                                         .extdata      = "{}",
                                                         .node_id      = k_node_id,
                                                         .process_id   = k_pid,
                                                         .agent_id     = gpu_agent });

    writer.register_kernel_symbol_info(
        kernel_symbol_info_t{ .id                        = 1,
                              .name                      = "vectorAdd",
                              .display_name              = "vectorAdd",
                              .kernel_object             = 0x1234,
                              .kernarg_segment_size      = 256,
                              .kernarg_segment_alignment = 8,
                              .group_segment_size        = 65536,
                              .private_segment_size      = 0,
                              .sgpr_count                = 32,
                              .arch_vgpr_count           = 64,
                              .accum_vgpr_count          = 0,
                              .extdata                   = "{}",
                              .node_id                   = k_node_id,
                              .process_id                = k_pid,
                              .code_obj_id               = 1 });

    writer.register_track_info(track_info_t{ .name       = "gpu_kernel",
                                             .extdata    = "{}",
                                             .node_id    = k_node_id,
                                             .process_id = k_pid,
                                             .thread_id  = k_tid });

    gpu_agent_out = gpu_agent;
}

}  // namespace

int
main(int argc, char** argv)
{
    if(argc < 2 || argc > 3)
    {
        std::fprintf(stderr, "Usage: %s <output.db> [N=%zu]\n", argv[0], k_default_count);
        return 1;
    }

    const std::string db_path = argv[1];
    const size_t      count   = (argc == 3)
                                    ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10))
                                    : k_default_count;

    std::remove(db_path.c_str());

    auto storage = std::make_unique<rocpdsna::storage_t>(db_path, std::string{ "bench" });
    auto writer  = std::make_shared<rocpdsna::writer_t>(std::move(storage));

    agent_unique_id_t gpu_agent;
    register_setup(*writer, gpu_agent);

    const trace_environment_t trace_env{ .node_id    = k_node_id,
                                         .process_id = k_pid,
                                         .thread_id  = k_tid,
                                         .agent_id   = gpu_agent,
                                         .stream_id  = 1,
                                         .queue_id   = 1,
                                         .track_name = "gpu_kernel" };

    const auto t0 = std::chrono::steady_clock::now();

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
        writer->insert_kernel_dispatch_data(data, trace_env);
    }
    writer->flush_in_memory_data_to_disk();

    const auto t1 = std::chrono::steady_clock::now();

    const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double rows_per_sec = (wall_ms > 0.0) ? (count * 1000.0 / wall_ms) : 0.0;

    std::cout << "rows=" << count << " wall_ms=" << wall_ms
              << " rows_per_sec=" << rows_per_sec << '\n';

    return 0;
}
