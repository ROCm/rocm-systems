// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Memory-copy round-trip launcher for the pytest-driven integration suite.
//
// This binary performs NO comparison. It writes one memory copy to a local rocpd
// database, reads it back, and prints every reader-recoverable field as
// "key=value" lines on stdout.
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

#ifndef PHUB_INTEGRATION_TMP_DIR
#    define PHUB_INTEGRATION_TMP_DIR "."
#endif

namespace
{
namespace wt = profiler_hub::writer_types;
namespace rt = profiler_hub::reader_types;

const std::string g_uuid = "integration";
const std::string g_db_path =
    std::string{ PHUB_INTEGRATION_TMP_DIR } + "/phub_memory_copy_writer_reader.db";

const wt::agent_unique_id_t g_src_agent_uid{ "GPU", 0 };
const wt::agent_unique_id_t g_dst_agent_uid{ "CPU", 0 };

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

wt::process_info_t
make_process()
{
    wt::process_info_t proc{};
    proc.ppid    = 1;
    proc.pid     = 1000;
    proc.node_id = 1;
    return proc;
}

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

wt::agent_info_t
make_agent(const wt::agent_unique_id_t& unique_id, const char* name)
{
    wt::agent_info_t agent{};
    agent.unique_id      = unique_id;
    agent.absolute_index = unique_id.type_index;
    agent.logical_index  = unique_id.type_index;
    agent.uuid           = unique_id.type_index + 100;
    agent.name           = name;
    agent.model_name     = "integration-model";
    agent.vendor_name    = "AMD";
    agent.product_name   = "integration-product";
    agent.user_name      = name;
    agent.node_id        = 1;
    agent.process_id     = 1000;
    return agent;
}

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

wt::event_data_t
make_event()
{
    wt::event_data_t event{};
    event.stack_id        = 2;
    event.parent_stack_id = 0;
    event.correlation_id  = 2;
    event.event_category  = "memory_copy";
    event.extdata         = "{test data event}";
    return event;
}

wt::memory_copy_data_t
make_memory_copy()
{
    wt::memory_copy_data_t copy{};
    copy.event           = make_event();
    copy.start_timestamp = 1200000;
    copy.end_timestamp   = 1300000;
    copy.dst_agent_id    = g_dst_agent_uid;
    copy.dst_address     = 4096;
    copy.src_agent_id    = g_src_agent_uid;
    copy.src_address     = 8192;
    copy.size            = 4096;
    copy.name            = "hipMemcpyDtoH";
    copy.region_name     = "integration_memcpy";
    copy.extdata         = "{test data memory copy}";
    return copy;
}

wt::trace_environment_t
make_env()
{
    wt::trace_environment_t env{};
    env.node_id    = 1;
    env.process_id = 1000;
    env.thread_id  = 100;
    env.queue_id   = 1;
    env.stream_id  = 1;
    return env;
}

const wt::node_info_t    g_node      = make_node();
const wt::process_info_t g_proc      = make_process();
const wt::thread_info_t  g_thread    = make_thread();
const wt::agent_info_t   g_src_agent = make_agent(g_src_agent_uid, "integration-src-gpu");
const wt::agent_info_t   g_dst_agent = make_agent(g_dst_agent_uid, "integration-dst-cpu");
const wt::queue_info_t   g_queue     = make_queue();
const wt::stream_info_t  g_stream    = make_stream();
const wt::memory_copy_data_t  g_copy = make_memory_copy();
const wt::trace_environment_t g_env  = make_env();

void
remove_db()
{
    std::remove(g_db_path.c_str());
}

void
writer()
{
    auto storage = std::make_unique<profiler_hub::storage_t>(g_db_path, g_uuid);
    profiler_hub::writer_t writer(std::move(storage));

    writer.register_node_info(g_node);
    writer.register_process_info(g_proc);
    writer.register_thread_info(g_thread);
    writer.register_agent_info(g_src_agent);
    writer.register_agent_info(g_dst_agent);
    writer.register_queue_info(g_queue);
    writer.register_stream_info(g_stream);
    writer.insert_memory_copy_data(g_copy, g_env);
    writer.flush_in_memory_data_to_disk();
}

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

    std::optional<rt::memory_copy_data_t> detail;
    for(const auto& event : reader.get_events())
    {
        if(event.unique_identifier.type != rt::event_type_t::memory_copy) continue;
        detail = reader.get_memory_copy_details(event);
        if(detail.has_value()) break;
    }

    const auto& d = *detail;

    print("start_timestamp", d.start_timestamp);
    print("end_timestamp", d.end_timestamp);
    print("dst_address",
          d.dst_address.has_value() ? std::to_string(d.dst_address.value()) : "null");
    print("src_address",
          d.src_address.has_value() ? std::to_string(d.src_address.value()) : "null");
    print("size", d.size);
    print("name", d.name);
    print("region_name", d.region_name);
    print("extdata", d.extdata);

    print("event.stack_id", d.event->stack_id);
    print("event.parent_stack_id", d.event->parent_stack_id);
    print("event.correlation_id", d.event->correlation_id);
    print("event.event_category", d.event->event_category);
    print("event.extdata", d.event->extdata);

    print("node_info.node_id", d.node_info->node_id);
    print("node_info.hash", d.node_info->hash);
    print("node_info.machine_id", d.node_info->machine_id);
    print("node_info.system_name", d.node_info->system_name);
    print("node_info.hostname", d.node_info->hostname);
    print("node_info.release", d.node_info->release);
    print("node_info.version", d.node_info->version);
    print("node_info.hardware_name", d.node_info->hardware_name);
    print("node_info.domain_name", d.node_info->domain_name);

    print("process_info.pid", d.process_info->pid);
    print("process_info.ppid",
          d.process_info->ppid.has_value() ? std::to_string(d.process_info->ppid.value())
                                           : "null");
    print("process_info.node_info.node_id", d.process_info->node_info->node_id);

    print("thread_info.thread_id", d.thread_info->thread_id);
    print("thread_info.name", d.thread_info->name);

    print("src_agent_info.agent_type", d.src_agent_id->agent_type);
    print("src_agent_info.type_index", d.src_agent_id->type_index);
    print("src_agent_info.name", d.src_agent_id->name);
    print("dst_agent_info.agent_type", d.dst_agent_id->agent_type);
    print("dst_agent_info.type_index", d.dst_agent_id->type_index);
    print("dst_agent_info.name", d.dst_agent_id->name);
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
