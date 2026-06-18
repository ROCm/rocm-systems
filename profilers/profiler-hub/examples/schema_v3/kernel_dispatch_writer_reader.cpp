// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Kernel-dispatch round-trip launcher for the pytest-driven integration suite.
//
// Unlike the self-validating examples, this binary performs NO comparison. It:
//   * writer() - registers the full dependency set and writes one kernel
//                dispatch to a local rocpd database, then flushes it.
//   * reader() - re-opens that database, reads the row back, and prints every
//                reader-recoverable field as a "key=value" line on stdout.
//
//

#include <profiler-hub/reader.hpp>
#include <profiler-hub/reader_types.hpp>
#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>

#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

// Scratch directory for the database. Injected by CMake; falls back to the cwd
// so the file also builds standalone.
#ifndef PHUB_INTEGRATION_TMP_DIR
#    define PHUB_INTEGRATION_TMP_DIR "."
#endif

namespace
{
namespace wt = profiler_hub::writer_types;
namespace rt = profiler_hub::reader_types;

const std::string g_uuid = "integration";
const std::string g_db_path =
    std::string{ PHUB_INTEGRATION_TMP_DIR } + "/phub_kernel_dispatch_writer_reader.db";

// The agent identity is shared by the agent, code object and trace environment.
const wt::agent_unique_id_t g_agent_uid{ "GPU", 0 };

// Each record is built by a small maker function with named-field assignments,
// then stored in a global below. Required dependency: node (require_node).
wt::node_info_t
make_node()
{
    wt::node_info_t node{};
    node.node_id       = 1;
    node.hash          = 123456789;
    node.machine_id    = "integration-machine";
    node.system_name   = "Linux";
    node.hostname      = "integration-host";
    node.release       = "6.0.0";
    node.version       = "#1 SMP";
    node.hardware_name = "x86_64";
    node.domain_name   = "integration";
    return node;
}

// Required dependency: process (require_process).
wt::process_info_t
make_process()
{
    wt::process_info_t proc{};
    proc.ppid    = 1;
    proc.pid     = 1000;
    proc.node_id = 1;
    return proc;
}

// Required dependency: thread (require_thread).
wt::thread_info_t
make_thread()
{
    wt::thread_info_t thread{};
    thread.parent_process_id = 1000;
    thread.thread_id         = 100;
    thread.name              = "integration-thread";
    thread.start             = 1000000;
    thread.end               = 2000000;
    thread.node_id           = 1;
    thread.process_id        = 1000;
    return thread;
}

// Required dependency: agent (require_agent).
wt::agent_info_t
make_agent()
{
    wt::agent_info_t agent{};
    agent.unique_id      = g_agent_uid;
    agent.absolute_index = 0;
    agent.logical_index  = 0;
    agent.uuid           = 12345;
    agent.name           = "gfx1100";
    agent.model_name     = "AMD Radeon";
    agent.vendor_name    = "AMD";
    agent.product_name   = "Radeon";
    agent.user_name      = "gpu0";
    agent.node_id        = 1;
    agent.process_id     = 1000;
    return agent;
}

// Required dependency: queue (require_queue).
wt::queue_info_t
make_queue()
{
    wt::queue_info_t queue{};
    queue.queue_id   = 1;
    queue.name       = "integration-queue";
    queue.node_id    = 1;
    queue.process_id = 1000;
    return queue;
}

// Required dependency: stream (require_stream).
wt::stream_info_t
make_stream()
{
    wt::stream_info_t stream{};
    stream.stream_id  = 1;
    stream.name       = "integration-stream";
    stream.node_id    = 1;
    stream.process_id = 1000;
    return stream;
}

// Referenced by the kernel symbol (require_kernel_symbol -> code object).
wt::code_object_info_t
make_code_object()
{
    wt::code_object_info_t code_object{};
    code_object.id           = 1;
    code_object.uri          = "file:///integration/kernel.co";
    code_object.load_base    = 4096;
    code_object.load_size    = 8192;
    code_object.load_delta   = 0;
    code_object.storage_type = "FILE";
    code_object.node_id      = 1;
    code_object.process_id   = 1000;
    code_object.agent_id     = g_agent_uid;
    return code_object;
}

// Required dependency: kernel symbol (require_kernel_symbol).
wt::kernel_symbol_info_t
make_kernel_symbol()
{
    wt::kernel_symbol_info_t ksym{};
    ksym.id                        = 1;
    ksym.name                      = "integration_kernel";
    ksym.display_name              = "Integration Kernel";
    ksym.kernel_object             = 4096;
    ksym.kernarg_segment_size      = 64;
    ksym.kernarg_segment_alignment = 8;
    ksym.group_segment_size        = 256;
    ksym.private_segment_size      = 0;
    ksym.sgpr_count                = 32;
    ksym.arch_vgpr_count           = 64;
    ksym.accum_vgpr_count          = 0;
    ksym.node_id                   = 1;
    ksym.process_id                = 1000;
    ksym.code_obj_id               = 1;
    return ksym;
}

// Correlated event (needed for the row to be readable back).
wt::event_data_t
make_event()
{
    wt::event_data_t event{};
    event.stack_id        = 1;
    event.parent_stack_id = 0;
    event.correlation_id  = 1;
    event.event_category  = "kernel_dispatch";
    event.extdata         = "{test data event}";
    return event;
}

// The record under test.
wt::kernel_dispatch_data_t
make_kernel()
{
    wt::kernel_dispatch_data_t kernel{};
    kernel.event                = make_event();
    kernel.dispatch_id          = 7;
    kernel.start_timestamp      = 1000000;
    kernel.end_timestamp        = 2000000;
    kernel.kernel_symbol_id     = 1;
    kernel.code_object_id       = 1;
    kernel.private_segment_size = 0;
    kernel.group_segment_size   = 256;
    kernel.workgroup_size_x     = 128;
    kernel.workgroup_size_y     = 1;
    kernel.workgroup_size_z     = 1;
    kernel.grid_size_x          = 4096;
    kernel.grid_size_y          = 1;
    kernel.grid_size_z          = 1;
    kernel.name                 = "integration_kernel";
    kernel.extdata              = "{test data kernel}";
    return kernel;
}

// Trace environment for insert_kernel_dispatch_data.
wt::trace_environment_t
make_env()
{
    wt::trace_environment_t env{};
    env.node_id    = 1;
    env.process_id = 1000;
    env.thread_id  = 100;
    env.agent_id   = g_agent_uid;
    env.stream_id  = 1;
    env.queue_id   = 1;
    return env;
}

// Global records: Shared by writer() and reader().
const wt::node_info_t            g_node    = make_node();
const wt::process_info_t         g_proc    = make_process();
const wt::thread_info_t          g_thread  = make_thread();
const wt::agent_info_t           g_agent   = make_agent();
const wt::queue_info_t           g_queue   = make_queue();
const wt::stream_info_t          g_stream  = make_stream();
const wt::code_object_info_t     g_codeobj = make_code_object();
const wt::kernel_symbol_info_t   g_ksym    = make_kernel_symbol();
const wt::kernel_dispatch_data_t g_kernel  = make_kernel();
const wt::trace_environment_t    g_env     = make_env();

void
remove_db()
{
    std::remove(g_db_path.c_str());
}

// ----------------------------------------------------------------------------
// Writer method: register the full dependency set, insert, and flush to disk.
// ----------------------------------------------------------------------------
void
writer()
{
    auto storage = std::make_unique<profiler_hub::storage_t>(g_db_path, g_uuid);
    profiler_hub::writer_t writer(std::move(storage));

    writer.register_node_info(g_node);
    writer.register_process_info(g_proc);
    writer.register_thread_info(g_thread);
    writer.register_agent_info(g_agent);
    writer.register_queue_info(g_queue);
    writer.register_stream_info(g_stream);
    writer.register_code_object_info(g_codeobj);
    writer.register_kernel_symbol_info(g_ksym);
    writer.insert_kernel_dispatch_data(g_kernel, g_env);
    writer.flush_in_memory_data_to_disk();
}

// ----------------------------------------------------------------------------
// Reader: re-open the database, read the row back, and print every
// reader-recoverable field as a "key=value" line on stdout.
// ----------------------------------------------------------------------------
template <typename T>
void
print(const char* key, const T& value)
{
    std::cout << key << "=" << value << "\n";
}

void
reader()
{
    auto storage = std::make_unique<profiler_hub::storage_t>(g_db_path, g_uuid);
    profiler_hub::reader_t reader(std::move(storage));

    std::optional<rt::kernel_dispatch_data_t> detail;
    for(const auto& event : reader.get_events())
    {
        if(event.unique_identifier.type != rt::event_type_t::kernel_dispatch) continue;
        detail = reader.get_kernel_dispatch_details(event);
        if(detail.has_value()) break;
    }

    const auto& d = *detail;

    // kernel_dispatch_data fields
    print("dispatch_id", d.dispatch_id);
    print("start_timestamp", d.start_timestamp);
    print("end_timestamp", d.end_timestamp);
    // reader-side segment sizes are std::optional<size_t>.
    print("private_segment_size",
          d.private_segment_size.has_value()
              ? std::to_string(d.private_segment_size.value())
              : "null");
    print("group_segment_size",
          d.group_segment_size.has_value() ? std::to_string(d.group_segment_size.value())
                                           : "null");
    print("workgroup_size_x", d.workgroup_size_x);
    print("workgroup_size_y", d.workgroup_size_y);
    print("workgroup_size_z", d.workgroup_size_z);
    print("grid_size_x", d.grid_size_x);
    print("grid_size_y", d.grid_size_y);
    print("grid_size_z", d.grid_size_z);
    print("name", d.name);
    print("extdata", d.extdata);

    // event fields
    print("event.stack_id", d.event->stack_id);
    print("event.parent_stack_id", d.event->parent_stack_id);
    print("event.correlation_id", d.event->correlation_id);
    print("event.event_category", d.event->event_category);
    print("event.extdata", d.event->extdata);

    // node_info fields
    print("node_info.node_id", d.node_info->node_id);
    print("node_info.hash", d.node_info->hash);
    print("node_info.machine_id", d.node_info->machine_id);
    print("node_info.system_name", d.node_info->system_name);
    print("node_info.hostname", d.node_info->hostname);
    print("node_info.release", d.node_info->release);
    print("node_info.version", d.node_info->version);
    print("node_info.hardware_name", d.node_info->hardware_name);
    print("node_info.domain_name", d.node_info->domain_name);

    // process_info fields
    print("process_info.pid", d.process_info->pid);
    print("process_info.ppid",
          d.process_info->ppid.has_value() ? std::to_string(d.process_info->ppid.value())
                                           : "null");
    print("process_info.node_info.node_id", d.process_info->node_info->node_id);

    // thread_info fields
    print("thread_info.thread_id", d.thread_info->thread_id);
    print("thread_info.name", d.thread_info->name);

    // code_object_info fields
    print("code_object_info.id", d.code_object_info->id);
    print("code_object_info.uri", d.code_object_info->uri);

    // kernel_symbol_info fields
    print("kernel_symbol_info.id", d.kernel_symbol_info->id);
    print("kernel_symbol_info.name", d.kernel_symbol_info->name);
    print("kernel_symbol_info.display_name", d.kernel_symbol_info->display_name);

    return;
}

}  // namespace

int
main()
{
    remove_db();

    try
    {
        writer();
        reader();
    } catch(const std::exception& e)
    {
        std::cerr << "[error] exception: " << e.what() << "\n";
        remove_db();
        return 1;
    }

    remove_db();

    return 0;
}
