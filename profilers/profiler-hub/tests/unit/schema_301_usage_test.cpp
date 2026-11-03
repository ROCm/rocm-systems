// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/reader.hpp"
#include "profiler-hub/reader_types.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include "backends/sqlite_backend.hpp"
#include "schema/pmc_event_input.hpp"
#include "schema/validation/schema_validator.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{

using namespace profiler_hub;
namespace writer_types = profiler_hub::writer_types;

constexpr size_t kNodeId = 1;
constexpr size_t kPid    = 4242;
constexpr size_t kTid    = 7;

// Same shape rocprofiler-systems builds in rocpd_helpers::make_event.
writer_types::event_data_t
make_event(size_t      stack_id,
           size_t      parent_stack_id,
           size_t      correlation_id,
           const char* category)
{
    writer_types::event_data_t event;
    event.stack_id        = stack_id;
    event.parent_stack_id = parent_stack_id;
    event.correlation_id  = correlation_id;
    event.event_category  = category;
    event.extdata         = R"({"origin":"rocprofiler-systems"})";

    shared_types::program_counter_info_t pc;
    pc.function    = "vector_add";
    pc.filename    = "vector_add.cpp";
    pc.line_number = 42;

    shared_types::address_range_info_t range;
    range.address_base = 0x1000;
    range.address_low  = 0x1000;
    range.address_high = 0x1800;

    shared_types::stack_frame_t frame;
    frame.program_counter = pc;
    frame.address_range   = range;
    event.call_stack.push_back(frame);

    shared_types::source_code_info_t source;
    source.filename                   = "vector_add.cpp";
    source.starting_line_number       = 40;
    source.source_code_lines          = { "void vector_add() {" };
    source.assembly_instruction_lines = { "s_load_dword s0, s[4:5], 0x0" };

    shared_types::line_info_entry_t line;
    line.source_code     = source;
    line.program_counter = pc;
    line.address_range   = range;
    event.line_info_list.push_back(line);
    return event;
}

writer_types::agent_unique_id_t
agent_id(const char* type, size_t index)
{
    writer_types::agent_unique_id_t id;
    id.agent_type = type;
    id.type_index = index;
    return id;
}

// Fills every writer input the 3.0.1 schema accepts. The calls follow the
// rocprofiler-systems handlers in rocpd_processor.cpp: info rows first, then
// region, kernel dispatch, memory copy, memory alloc, and a PMC sample.
void
write_schema_301(writer_t& writer)
{
    writer_types::node_info_t node;
    node.node_id       = kNodeId;
    node.hash          = 0x301;
    node.machine_id    = "machine-301";
    node.system_name   = "Linux";
    node.hostname      = "gfx-node";
    node.release       = "6.8.0";
    node.version       = "#1 SMP";
    node.hardware_name = "x86_64";
    node.domain_name   = "amd.com";
    writer.register_node_info(node);

    writer_types::process_info_t process;
    process.ppid        = 1;
    process.pid         = kPid;
    process.init        = 10;
    process.fini        = 90;
    process.start       = 100;
    process.end         = 900;
    process.command     = "rocprof-systems-sample";
    process.environment = R"({"ROCR_VISIBLE_DEVICES":"0"})";
    process.extdata     = R"({"mpi_rank":0})";
    process.node_id     = kNodeId;
    writer.register_process_info(process);

    writer_types::thread_info_t thread;
    thread.parent_process_id = kPid;
    thread.thread_id         = kTid;
    thread.name              = "main";
    thread.start             = 100;
    thread.end               = 900;
    thread.extdata           = R"({"role":"application"})";
    thread.node_id           = kNodeId;
    thread.process_id        = kPid;
    writer.register_thread_info(thread);

    const auto cpu = agent_id("CPU", 0);
    const auto gpu = agent_id("GPU", 0);
    const auto nic = agent_id("NIC", 0);

    auto add_agent = [&](const writer_types::agent_unique_id_t& id,
                         size_t                                 absolute,
                         const char*                            name,
                         const char*                            model,
                         const char*                            product) {
        writer_types::agent_info_t agent;
        agent.unique_id      = id;
        agent.absolute_index = absolute;
        agent.logical_index  = 0;
        agent.uuid           = 0xA000 + absolute;
        agent.name           = name;
        agent.model_name     = model;
        agent.vendor_name    = "AMD";
        agent.product_name   = product;
        agent.user_name      = name;
        agent.extdata        = "{}";
        agent.node_id        = kNodeId;
        agent.process_id     = kPid;
        writer.register_agent_info(agent);
    };
    add_agent(cpu, 0, "cpu0", "Zen4", "EPYC");
    add_agent(gpu, 1, "gpu0", "gfx942", "MI300X");
    add_agent(nic, 2, "nic0", "ionic", "Pollara");

    auto add_pmc = [&](const char*                            name,
                       const writer_types::agent_unique_id_t& id,
                       const char*                            arch,
                       const char*                            value_type,
                       size_t                                 event_code) {
        writer_types::pmc_info_t pmc;
        pmc.unique_id.name     = name;
        pmc.unique_id.agent_id = id;
        pmc.target_arch        = arch;
        pmc.event_code         = event_code;
        pmc.instance_id        = 0;
        pmc.symbol             = name;
        pmc.description        = "counter";
        pmc.long_description   = "hardware counter";
        pmc.component          = "SQ";
        pmc.units              = "count";
        pmc.value_type         = value_type;
        pmc.block              = "SQ";
        pmc.expression         = name;
        pmc.is_constant        = 0;
        pmc.is_derived         = 0;
        pmc.extdata            = "{}";
        pmc.node_id            = kNodeId;
        pmc.process_id         = kPid;
        writer.register_pmc_info(pmc);
    };
    add_pmc("SQ_WAVES", gpu, "GPU", "ABS", 0x11);
    add_pmc("NIC_BYTES", nic, "NIC", "ACCUM", 0x22);

    writer_types::queue_info_t queue;
    queue.queue_id   = 30;
    queue.name       = "hip-queue";
    queue.extdata    = "{}";
    queue.node_id    = kNodeId;
    queue.process_id = kPid;
    writer.register_queue_info(queue);

    writer_types::stream_info_t stream;
    stream.stream_id  = 40;
    stream.name       = "hip-stream";
    stream.extdata    = "{}";
    stream.node_id    = kNodeId;
    stream.process_id = kPid;
    writer.register_stream_info(stream);

    writer_types::code_object_info_t code_object;
    code_object.id           = 10;
    code_object.uri          = "file:///opt/app/vector_add.hsaco";
    code_object.load_base    = 0x1000;
    code_object.load_size    = 0x800;
    code_object.load_delta   = 0;
    code_object.storage_type = "FILE";
    code_object.extdata      = "{}";
    code_object.node_id      = kNodeId;
    code_object.process_id   = kPid;
    code_object.agent_id     = gpu;
    writer.register_code_object_info(code_object);

    writer_types::kernel_symbol_info_t symbol;
    symbol.id                        = 20;
    symbol.name                      = "vector_add";
    symbol.display_name              = "vector_add(float*, float*, int)";
    symbol.kernel_object             = 0x2000;
    symbol.kernarg_segment_size      = 24;
    symbol.kernarg_segment_alignment = 8;
    symbol.group_segment_size        = 256;
    symbol.private_segment_size      = 0;
    symbol.sgpr_count                = 16;
    symbol.arch_vgpr_count           = 8;
    symbol.accum_vgpr_count          = 4;
    symbol.extdata                   = "{}";
    symbol.node_id                   = kNodeId;
    symbol.process_id                = kPid;
    symbol.code_obj_id               = code_object.id;
    writer.register_kernel_symbol_info(symbol);

    writer_types::track_info_t track;
    track.name       = "main";
    track.extdata    = "{}";
    track.node_id    = kNodeId;
    track.process_id = kPid;
    track.thread_id  = kTid;
    writer.register_track_info(track);

    writer.register_string("usage-note");

    writer_types::trace_environment_t env;
    env.node_id    = kNodeId;
    env.process_id = kPid;
    env.thread_id  = kTid;
    env.agent_id   = gpu;
    env.queue_id   = queue.queue_id;
    env.stream_id  = stream.stream_id;
    env.track_name = "main";

    // region_sample: a host API span plus one argument.
    writer_types::region_data_t region;
    region.event           = make_event(1, 0, 77, "HIP_API");
    region.start_timestamp = 1000;
    region.end_timestamp   = 1500;
    region.name            = "hipLaunchKernel";
    region.extdata         = R"({"api":"hip"})";
    writer_types::arg_data_t arg;
    arg.position = 0;
    arg.type     = "const void*";
    arg.name     = "function_address";
    arg.value    = "0x2000";
    arg.extdata  = "{}";
    region.args.push_back(arg);
    writer.insert_region_data(region, env);

    // kernel_dispatch_sample: grid, workgroup, and scratch sizes.
    writer_types::kernel_dispatch_data_t kernel;
    kernel.event                = make_event(2, 1, 77, "KERNEL_DISPATCH");
    kernel.dispatch_id          = 1;
    kernel.start_timestamp      = 1600;
    kernel.end_timestamp        = 2400;
    kernel.kernel_symbol_id     = symbol.id;
    kernel.code_object_id       = code_object.id;
    kernel.private_segment_size = 0;
    kernel.group_segment_size   = 256;
    kernel.workgroup_size_x     = 64;
    kernel.workgroup_size_y     = 1;
    kernel.workgroup_size_z     = 1;
    kernel.grid_size_x          = 256;
    kernel.grid_size_y          = 1;
    kernel.grid_size_z          = 1;
    kernel.name                 = "vector_add";
    kernel.extdata              = R"({"stream":40})";
    writer.insert_kernel_dispatch_data(kernel, env);

    // memory_copy_sample: host to device.
    writer_types::memory_copy_data_t copy;
    copy.event           = make_event(3, 1, 78, "MEMORY_COPY");
    copy.start_timestamp = 2500;
    copy.end_timestamp   = 2600;
    copy.dst_agent_id    = gpu;
    copy.dst_address     = 0x8000;
    copy.src_agent_id    = cpu;
    copy.src_address     = 0x4000;
    copy.size            = 4096;
    copy.name            = "HOST_TO_DEVICE";
    copy.region_name     = "hipMemcpy";
    copy.extdata         = "{}";
    writer.insert_memory_copy_data(copy, env);

    // memory_allocate_sample. 3.0.1 allows ALLOC/FREE/REALLOC and
    // REAL/VIRTUAL/SCRATCH. rocprofiler-systems maps its own names onto these.
    auto add_alloc = [&](const char* type,
                         const char* level,
                         size_t      address,
                         size_t      size,
                         size_t      start) {
        writer_types::memory_alloc_data_t alloc;
        alloc.event           = make_event(4, 1, 79, "MEMORY_ALLOCATION");
        alloc.type            = type;
        alloc.level           = level;
        alloc.start_timestamp = start;
        alloc.end_timestamp   = start + 20;
        alloc.address         = address;
        alloc.size            = size;
        alloc.extdata         = "{}";
        writer.insert_memory_alloc_data(alloc, env);
    };
    add_alloc("ALLOC", "REAL", 0x8000, 4096, 2700);
    add_alloc("FREE", "VIRTUAL", 0x9000, 8192, 2800);
    add_alloc("REALLOC", "SCRATCH", 0xA000, 1024, 2900);

    // pmc_event_with_sample. The sample track must already be registered.
    writer_types::pmc_event_data_t pmc_event;
    pmc_event.event            = make_event(5, 2, 80, "PMC");
    pmc_event.value            = 12.5;
    pmc_event.extdata          = R"({"unit":"waves"})";
    pmc_event.sample.timestamp = 2400;
    pmc_event.sample.track     = track;
    pmc_event.sample.extdata   = "{}";
    const writer_types::pmc_info_unique_id_t pmc_id{ "SQ_WAVES", gpu };
    writer.insert_pmc_event_data(pmc_event, pmc_id);
}

class schema_301_usage_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path        = (std::filesystem::temp_directory_path() /
                     (std::string{ "schema_301_" } + info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> open_writer(version_t version) const
    {
        std::filesystem::remove(m_db_path);
        return std::make_unique<writer_t>(
            std::make_unique<storage_t>(m_db_path, m_uuid, version));
    }

    std::string m_db_path;
    std::string m_uuid = "usage301examp";
};

const char*
event_type_name(reader_types::event_type_t type)
{
    switch(type)
    {
        case reader_types::event_type_t::region: return "region";
        case reader_types::event_type_t::kernel_dispatch: return "kernel_dispatch";
        case reader_types::event_type_t::memory_copy: return "memory_copy";
        case reader_types::event_type_t::memory_allocate: return "memory_allocate";
        case reader_types::event_type_t::sample: return "sample";
        case reader_types::event_type_t::pmc_event: return "pmc_event";
    }
    return "unknown";
}

template <typename T>
void
print_optional(const char* label, const std::optional<T>& value)
{
    std::cout << "    " << label << ": ";
    if(value.has_value()) std::cout << *value << '\n';
    else std::cout << "<unset>\n";
}

void
print_database(const std::string& db_path, const std::string& uuid)
{
    std::cout << "database: " << db_path << '\n';

    auto reader = std::make_unique<reader_t>(std::make_unique<storage_t>(db_path, uuid));

    std::cout << "nodes: " << reader->get_all_nodes().size() << '\n';
    for(const auto& node : reader->get_all_nodes())
    {
        std::cout << "  node_id=" << node->node_id << " hash=" << node->hash
                  << " machine_id=" << node->machine_id
                  << " system_name=" << node->system_name
                  << " hostname=" << node->hostname << " release=" << node->release
                  << " version=" << node->version
                  << " hardware_name=" << node->hardware_name
                  << " domain_name=" << node->domain_name << '\n';
    }

    std::cout << "processes: " << reader->get_all_processes().size() << '\n';
    for(const auto& process : reader->get_all_processes())
    {
        std::cout << "  pid=" << process->pid << " command=" << process->command
                  << " environment=" << process->environment
                  << " extdata=" << process->extdata << '\n';
        print_optional("ppid", process->ppid);
        print_optional("init", process->init);
        print_optional("fini", process->fini);
        print_optional("start", process->start);
        print_optional("end", process->end);
    }

    std::cout << "threads: " << reader->get_all_threads().size() << '\n';
    for(const auto& thread : reader->get_all_threads())
    {
        std::cout << "  thread_id=" << thread->thread_id << " name=" << thread->name
                  << " extdata=" << thread->extdata << '\n';
        print_optional("parent_process_id", thread->parent_process_id);
        print_optional("start", thread->start);
        print_optional("end", thread->end);
    }

    std::cout << "agents: " << reader->get_all_agents().size() << '\n';
    for(const auto& agent : reader->get_all_agents())
    {
        std::cout << "  type=" << agent->agent_type << " type_index=" << agent->type_index
                  << " name=" << agent->name << " model=" << agent->model_name
                  << " vendor=" << agent->vendor_name
                  << " product=" << agent->product_name
                  << " user=" << agent->user_name << '\n';
        print_optional("absolute_index", agent->absolute_index);
        print_optional("logical_index", agent->logical_index);
        print_optional("uuid", agent->uuid);
    }

    std::cout << "pmc info: " << reader->get_all_pmc_info().size() << '\n';
    for(const auto& pmc : reader->get_all_pmc_info())
    {
        std::cout << "  name=" << pmc->name << " symbol=" << pmc->symbol
                  << " target_arch=" << pmc->target_arch
                  << " value_type=" << pmc->value_type
                  << " description=" << pmc->description
                  << " component=" << pmc->component << " units=" << pmc->units
                  << " block=" << pmc->block << " expression=" << pmc->expression
                  << '\n';
        print_optional("event_code", pmc->event_code);
        print_optional("instance_id", pmc->instance_id);
        print_optional("is_constant", pmc->is_constant);
        print_optional("is_derived", pmc->is_derived);
    }

    std::cout << "queues: " << reader->get_all_queues().size() << '\n';
    for(const auto& queue : reader->get_all_queues())
    {
        std::cout << "  queue_id=" << queue->queue_id << " name=" << queue->name
                  << '\n';
    }

    std::cout << "streams: " << reader->get_all_streams().size() << '\n';
    for(const auto& stream : reader->get_all_streams())
    {
        std::cout << "  stream_id=" << stream->stream_id << " name=" << stream->name
                  << '\n';
    }

    std::cout << "code objects: " << reader->get_all_code_objects().size() << '\n';
    for(const auto& code_object : reader->get_all_code_objects())
    {
        std::cout << "  id=" << code_object->id << " uri=" << code_object->uri
                  << " storage_type=" << code_object->storage_type << '\n';
        print_optional("load_base", code_object->load_base);
        print_optional("load_size", code_object->load_size);
        print_optional("load_delta", code_object->load_delta);
    }

    std::cout << "kernel symbols: " << reader->get_all_kernel_symbols().size() << '\n';
    for(const auto& symbol : reader->get_all_kernel_symbols())
    {
        std::cout << "  id=" << symbol->id << " name=" << symbol->name
                  << " display_name=" << symbol->display_name << '\n';
        print_optional("kernel_object", symbol->kernel_object);
        print_optional("kernarg_segment_size", symbol->kernarg_segment_size);
        print_optional("kernarg_segment_alignment", symbol->kernarg_segment_alignment);
        print_optional("group_segment_size", symbol->group_segment_size);
        print_optional("private_segment_size", symbol->private_segment_size);
        print_optional("sgpr_count", symbol->sgpr_count);
        print_optional("arch_vgpr_count", symbol->arch_vgpr_count);
        print_optional("accum_vgpr_count", symbol->accum_vgpr_count);
    }

    std::cout << "tracks: " << reader->get_all_tracks().size() << '\n';
    for(const auto& track : reader->get_all_tracks())
    {
        std::cout << "  name=" << track->name << " extdata=" << track->extdata << '\n';
    }

    std::cout << "event counts:\n";
    for(const auto& entry : reader->get_event_counts())
    {
        std::cout << "  " << event_type_name(entry.first) << "=" << entry.second << '\n';
    }

    const auto events = reader->get_events();
    std::cout << "timeline events: " << events.size() << '\n';
    for(const auto& event : events)
    {
        std::cout << "  \"" << event.display_name << "\" category=" << event.category
                  << " type=" << event_type_name(event.unique_identifier.type) << " ["
                  << event.start_timestamp << " -> " << event.end_timestamp << "]\n";

        if(auto region = reader->get_region_details(event))
        {
            std::cout << "    region name=" << region->name
                      << " extdata=" << region->extdata << '\n';
            const auto args = reader->get_arguments(event);
            std::cout << "    arguments: " << args.size() << '\n';
            for(const auto& arg : args)
            {
                std::cout << "      pos=" << arg->position << " type=" << arg->type
                          << " name=" << arg->name << " value=" << arg->value << '\n';
            }
        }
        if(auto kernel = reader->get_kernel_dispatch_details(event))
        {
            std::cout << "    kernel name=" << kernel->name
                      << " dispatch_id=" << kernel->dispatch_id
                      << " grid=" << kernel->grid_size_x << "x" << kernel->grid_size_y
                      << "x" << kernel->grid_size_z
                      << " workgroup=" << kernel->workgroup_size_x << "x"
                      << kernel->workgroup_size_y << "x" << kernel->workgroup_size_z
                      << '\n';
            print_optional("private_segment_size", kernel->private_segment_size);
            print_optional("group_segment_size", kernel->group_segment_size);
        }
        if(auto copy = reader->get_memory_copy_details(event))
        {
            std::cout << "    copy name=" << copy->name
                      << " region_name=" << copy->region_name << " size=" << copy->size
                      << '\n';
            print_optional("src_address", copy->src_address);
            print_optional("dst_address", copy->dst_address);
            std::cout << "    src_agent="
                      << (copy->src_agent_id ? copy->src_agent_id->agent_type : "<none>")
                      << " dst_agent="
                      << (copy->dst_agent_id ? copy->dst_agent_id->agent_type : "<none>")
                      << '\n';
        }
        if(auto alloc = reader->get_memory_alloc_details(event))
        {
            std::cout << "    alloc type=" << alloc->type << " level=" << alloc->level
                      << " size=" << alloc->size << '\n';
            print_optional("address", alloc->address);
        }
        if(auto pmc = reader->get_pmc_event_details(event))
        {
            std::cout << "    pmc value=" << pmc->value << " extdata=" << pmc->extdata
                      << '\n';
        }

        const auto stack = reader->get_call_stack(event);
        std::cout << "    call stack frames: " << stack.size() << '\n';
        for(const auto& frame : stack)
        {
            if(frame.program_counter)
            {
                std::cout << "      function=" << frame.program_counter->function
                          << " file=" << frame.program_counter->filename << '\n';
                print_optional("line", frame.program_counter->line_number);
            }
        }
    }

    try
    {
        struct row_t
        {
            double value = 0;
        };
        auto backend = data_storage::sqlite_backend::create(
            db_path, uuid, data_storage::sqlite_backend::storage_mode_t::on_disk);
        const std::string query = "SELECT value FROM rocpd_pmc_event_" + uuid;
        auto              read =
            backend->create_read_statement_executor<row_t>(query, &row_t::value);
        const auto rows = read().to_vector();
        std::cout << "rocpd_pmc_event rows: " << rows.size() << '\n';
        for(const auto& row : rows)
        {
            std::cout << "  value=" << row.value << '\n';
        }
    } catch(const std::exception& error)
    {
        std::cout << "rocpd_pmc_event read threw: " << error.what() << '\n';
    }
    std::cout << std::flush;
}

TEST_F(schema_301_usage_test, writes_every_3_0_1_input)
{
    std::cout << "\n=== schema 3.0.1 write ===\n";
    try
    {
        auto writer = open_writer(version_t{ 3, 0, 1 });
        write_schema_301(*writer);
        writer->flush_in_memory_data_to_disk();
        writer.reset();
        print_database(m_db_path, m_uuid);
    } catch(const std::exception& error)
    {
        std::cout << "3.0.1 write threw: " << error.what() << '\n' << std::flush;
    }
}

TEST_F(schema_301_usage_test, writer_for_3_0_0_opens)
{
    std::cout << "\n=== schema 3.0.0 writer ===\n";
    try
    {
        auto writer = open_writer(version_t{ 3, 0, 0 });
        std::cout << "writer opened\n" << std::flush;
        (void) writer;
    } catch(const std::exception& error)
    {
        std::cout << "writer threw: " << error.what() << '\n' << std::flush;
    }
}

TEST_F(schema_301_usage_test, writer_for_3_0_2_and_later_errors)
{
    const version_t versions[] = {
        version_t{ 3, 0, 2 }, version_t{ 3, 0, 3 }, version_t{ 3, 0, 4 },
        version_t{ 3, 0, 5 }, version_t{},
    };

    for(const auto version : versions)
    {
        std::cout << "\n=== schema " << version.to_string();
        if(version.is_latest()) std::cout << " (latest)";
        std::cout << " writer ===\n";
        try
        {
            auto writer = open_writer(version);
            std::cout << "writer opened\n" << std::flush;
            (void) writer;
        } catch(const std::exception& error)
        {
            std::cout << error.what() << '\n' << std::flush;
        }
    }
}

// 3.0.5 still has rocpd_pmc_event, but xcc, shader_engine, and instance moved
// to rocpd_info_pmc. validate() reports each of those fields. The writer is a
// separate check and is not opened here.
TEST_F(schema_301_usage_test, pmc_event_3_0_5_prints_unsupported_fields)
{
    schema::pmc_event_input_t input;
    input.pmc_id        = 1;
    input.value         = 12.5;
    input.sample_id     = 100;
    input.xcc           = 1;
    input.shader_engine = 2;
    input.instance      = 3;

    std::cout << "\n=== schema 3.0.5 PMC event, unsupported fields ===\n";
    const auto result = schema::schema_validator::validate(input, version_t{ 3, 0, 5 });
    std::cout << result.format() << std::flush;
}

}  // namespace
