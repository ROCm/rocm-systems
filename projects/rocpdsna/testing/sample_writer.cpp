/*
/home/rocm/AnujWork/rocpdsna/rocm-systems/projects/rocpdsna/testing && rm -f sample_writer.db && g++ -std=c++20 sample_writer.cpp -I/home/rocm/AnujWork/rocpdsna/rocm-systems/projects/backup/200326_1_working/rocpdsna/build/include -L/home/rocm/AnujWork/rocpdsna/rocm-systems/projects/backup/200326_1_working/rocpdsna/build/lib -lrocpdsna -lsqlite3 -lpthread -Wl,-rpath,/home/rocm/AnujWork/rocpdsna/rocm-systems/projects/backup/200326_1_working/rocpdsna/build/lib -o sample_writer && ./sample_writer

*/
#include <rocpdsna/storage.hpp>
#include <rocpdsna/reader.hpp>
#include <rocpdsna/writer.hpp>
#include <rocpdsna/writer_types.hpp>

#include <iostream>
#include <memory>
#include <string>

int main()
{
    using namespace rocpdsna;
    using namespace rocpdsna::writer_types;

    // 1) Create storage + writer
    const std::string db_path = "sample_writer.db";
    const std::string uuid    = "demouuid001";
    auto storage              = std::make_unique<storage_t>(db_path, uuid);
    writer_t writer(std::move(storage));

    // 2) Register base dependencies: node -> process -> thread
    writer.register_node_info(node_info_t{
        .node_id       = 1,
        .hash          = 123456789,
        .machine_id    = "machine-1",
        .system_name   = "Linux",
        .hostname      = "host-1",
        .release       = "6.x",
        .version       = "#1 SMP",
        .hardware_name = "x86_64",
        .domain_name   = "local"
    });

    writer.register_process_info(process_info_t{
        .ppid        = 1,
        .pid         = 4242,
        .init        = 1000000,
        .fini        = 2000000,
        .start       = 1000000,
        .end         = 2000000,
        .command     = "/usr/bin/demo",
        .environment = "{}",
        .extdata     = "{}",
        .node_id     = 1
    });

    writer.register_thread_info(thread_info_t{
        .parent_process_id = 4242,
        .thread_id         = 100,
        .name              = "main-thread",
        .start             = 1000000,
        .end               = 2000000,
        .extdata           = "{}",
        .node_id           = 1,
        .process_id        = 4242
    });

    // 3) Agent + code object + kernel symbol (required references for kernel dispatch)
    writer.register_agent_info(agent_info_t{
        .unique_id      = { .agent_type = "GPU", .type_index = 0 },
        .absolute_index = 0,
        .logical_index  = 0,
        .uuid           = 12345,
        .name           = "gfx1100",
        .model_name     = "AMD Radeon",
        .vendor_name    = "AMD",
        .product_name   = "Radeon",
        .user_name      = "gpu0",
        .extdata        = "{}",
        .node_id        = 1,
        .process_id     = 4242
    });

    writer.register_stream_info(stream_info_t{
        .stream_id   = 1,
        .name        = "stream-1",
        .extdata     = "{}",
        .node_id     = 1,
        .process_id  = 4242
    });

    writer.register_queue_info(queue_info_t{
        .queue_id    = 1,
        .name        = "queue-1",
        .extdata     = "{}",
        .node_id     = 1,
        .process_id  = 4242
    });

    writer.register_code_object_info(code_object_info_t{
        .id           = 1,
        .uri          = "file:///tmp/demo.co",
        .load_base    = 0x1000,
        .load_size    = 0x2000,
        .load_delta   = 0,
        .storage_type = "FILE",
        .extdata      = "{}",
        .node_id      = 1,
        .process_id   = 4242,
        .agent_id     = agent_unique_id_t{ .agent_type = "GPU", .type_index = 0 }
    });

    writer.register_kernel_symbol_info(kernel_symbol_info_t{
        .id                        = 1,
        .name                      = "demo_kernel",
        .display_name              = "Demo Kernel",
        .kernel_object             = 0x1000,
        .kernarg_segment_size      = 64,
        .kernarg_segment_alignment = 8,
        .group_segment_size        = 256,
        .private_segment_size      = 0,
        .sgpr_count                = 32,
        .arch_vgpr_count           = 64,
        .accum_vgpr_count          = 0,
        .extdata                   = "{}",
        .node_id                   = 1,
        .process_id                = 4242,
        .code_obj_id               = 1
    });

    // 4) Insert one kernel dispatch record
    trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 4242,
        .thread_id  = 100,
        .agent_id   = agent_unique_id_t{ .agent_type = "GPU", .type_index = 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = std::nullopt
    };

    kernel_dispatch_data_t kd{
        .event                = event_data_t{ .stack_id        = 1,
                                             .parent_stack_id = 0,
                                             .correlation_id  = 1,
                                             .call_stack      = {},
                                             .line_info_list  = {},
                                             .event_category  = "KERNEL_DISPATCH",
                                             .extdata         = "{}" },
        .dispatch_id          = 1,
        .start_timestamp      = 1000000,
        .end_timestamp        = 2000000,
        .kernel_symbol_id     = 1,
        .code_object_id       = 1,
        .private_segment_size = 0,
        .group_segment_size   = 256,
        .workgroup_size_x     = 64,
        .workgroup_size_y     = 1,
        .workgroup_size_z     = 1,
        .grid_size_x          = 1024,
        .grid_size_y          = 1,
        .grid_size_z          = 1,
        .name                 = "demo_kernel_dispatch",
        .extdata              = "{}"
    };

    writer.insert_kernel_dispatch_data(kd, trace_env);
    writer.flush_in_memory_data_to_disk();

    // 5) Read back kernel dispatch data
    auto read_storage = std::make_unique<storage_t>(db_path, uuid);
    reader_t reader(std::move(read_storage));

    reader_types::event_filter_t filter{};
    filter.types = { reader_types::event_type_t::kernel_dispatch };

    const auto events = reader.get_events(filter);
    std::cout << "kernel_dispatch events found: " << events.size() << "\n";

    for(const auto& event : events)
    {
        const auto details = reader.get_kernel_dispatch_details(event);
        if(!details.has_value())
        {
            continue;
        }

        std::cout << "dispatch_id=" << details->dispatch_id
                  << " start=" << details->start_timestamp
                  << " end=" << details->end_timestamp
                  << " stack_id= " << details->event->stack_id
                  << " kernel_name=" << details->name << "\n";
    }

    return 0;
}