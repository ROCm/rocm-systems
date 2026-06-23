// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Memory-alloc launcher for the pytest-driven integration suite.
//
// This binary performs NO comparison. It writes one memory allocation to the
// Python-provided SQLite DB path, prints that path, and leaves DB validation and
// deletion to Python.
//

#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>

#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace
{
namespace wt = profiler_hub::writer_types;

const std::string g_uuid = "integration";

const wt::agent_unique_id_t g_agent_uid{ "GPU", 0 };

// ----------------------------------------------------------------------------
// Fixture records. These are the values written to the database. The expected
// values asserted in Python (test_memory_alloc.py) must mirror them.
// ----------------------------------------------------------------------------
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
    thread.start             = 900000;
    thread.end               = 1200000;
    thread.node_id           = 1;
    thread.process_id        = 1000;
    return thread;
}

wt::agent_info_t
make_agent()
{
    wt::agent_info_t agent{};
    agent.unique_id      = g_agent_uid;
    agent.absolute_index = 0;
    agent.logical_index  = 0;
    agent.uuid           = 42;
    agent.name           = "integration-gpu";
    agent.model_name     = "gfx-integration";
    agent.vendor_name    = "AMD";
    agent.product_name   = "Integration GPU";
    agent.user_name      = "gpu0";
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
    event.stack_id        = 1;
    event.parent_stack_id = 0;
    event.correlation_id  = 1;
    event.event_category  = "SCRATCH_MEMORY";
    event.extdata         = "{test data event}";
    return event;
}

wt::memory_alloc_data_t
make_alloc()
{
    wt::memory_alloc_data_t alloc{};
    alloc.event           = make_event();
    alloc.type            = "ALLOC";
    alloc.level           = "SCRATCH";
    alloc.start_timestamp = 1000000;
    alloc.end_timestamp   = 1100000;
    alloc.address         = 0;  // scratch sentinel (present, value 0)
    alloc.size            = 8192;
    alloc.extdata         = "{test data alloc}";
    return alloc;
}

wt::trace_environment_t
make_env()
{
    wt::trace_environment_t env{};
    env.node_id    = 1;
    env.process_id = 1000;
    env.thread_id  = 100;
    env.agent_id   = g_agent_uid;
    env.queue_id   = 1;
    env.stream_id  = 1;
    return env;
}

const wt::node_info_t         g_node   = make_node();
const wt::process_info_t      g_proc   = make_process();
const wt::thread_info_t       g_thread = make_thread();
const wt::agent_info_t        g_agent  = make_agent();
const wt::queue_info_t        g_queue  = make_queue();
const wt::stream_info_t       g_stream = make_stream();
const wt::memory_alloc_data_t g_alloc  = make_alloc();
const wt::trace_environment_t g_env    = make_env();

void
remove_db(const std::string& db_path)
{
    std::remove(db_path.c_str());
}

// ----------------------------------------------------------------------------
// Writer: register the minimal dependencies, insert, and flush to disk.
// ----------------------------------------------------------------------------
void
writer(const std::string& db_path)
{
    auto storage = std::make_unique<profiler_hub::storage_t>(db_path, g_uuid);
    profiler_hub::writer_t writer(std::move(storage));

    writer.register_node_info(g_node);
    writer.register_process_info(g_proc);
    writer.register_thread_info(g_thread);
    writer.register_agent_info(g_agent);
    writer.register_queue_info(g_queue);
    writer.register_stream_info(g_stream);

    writer.insert_memory_alloc_data(g_alloc, g_env);
    writer.flush_in_memory_data_to_disk();
}

template <typename T>
void
print(const char* key, const T& value)
{
    std::cout << key << "=" << value << "\n";
}

}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <db_path>\n";
        return 2;
    }

    const std::string db_path = argv[1];
    remove_db(db_path);

    try
    {
        writer(db_path);
        print("db_path", db_path);
    } catch(const std::exception& e)
    {
        std::cerr << "[error] exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
