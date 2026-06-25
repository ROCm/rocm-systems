// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Region launcher for the pytest-driven integration suite.
//
// This binary performs NO comparison. It writes one region to the Python-provided
// SQLite DB path, prints that path, and leaves DB validation and deletion to
// Python.
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

wt::track_info_t
make_track()
{
    wt::track_info_t track{};
    track.name       = "integration-region-sample-track";
    track.extdata    = "{test data track}";
    track.node_id    = 1;
    track.process_id = 1000;
    track.thread_id  = 100;
    return track;
}

wt::event_data_t
make_event()
{
    wt::event_data_t event{};
    event.stack_id        = 3;
    event.parent_stack_id = 0;
    event.correlation_id  = 3;
    event.event_category  = "region";
    event.extdata         = "{test data event}";
    return event;
}

wt::arg_data_t
make_arg()
{
    wt::arg_data_t arg{};
    arg.position = 0;
    arg.type     = "int";
    arg.name     = "iterations";
    arg.value    = "64";
    arg.extdata  = "{test data arg}";
    return arg;
}

wt::region_data_t
make_region()
{
    wt::region_data_t region{};
    region.event           = make_event();
    region.start_timestamp = 1400000;
    region.end_timestamp   = 1500000;
    region.name            = "integration_region";
    region.extdata         = "{test data region}";
    region.args.push_back(make_arg());
    return region;
}

wt::trace_environment_t
make_env()
{
    wt::trace_environment_t env{};
    env.node_id    = 1;
    env.process_id = 1000;
    env.thread_id  = 100;
    env.track_name = "integration-region-sample-track";
    return env;
}

const wt::node_info_t         g_node   = make_node();
const wt::process_info_t      g_proc   = make_process();
const wt::thread_info_t       g_thread = make_thread();
const wt::track_info_t        g_track  = make_track();
const wt::region_data_t       g_region = make_region();
const wt::trace_environment_t g_env    = make_env();

void
remove_db(const std::string& db_path)
{
    std::remove(db_path.c_str());
}

void
writer(const std::string& db_path)
{
    auto storage = std::make_unique<profiler_hub::storage_t>(db_path, g_uuid);
    profiler_hub::writer_t writer(std::move(storage));

    writer.register_node_info(g_node);
    writer.register_process_info(g_proc);
    writer.register_thread_info(g_thread);
    writer.register_track_info(g_track);
    writer.insert_region_data(g_region, g_env);
    writer.flush_in_memory_data_to_disk();
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
        std::cout << "db_path=" << db_path << "\n";
    } catch(const std::exception& e)
    {
        std::cerr << "[error] exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
