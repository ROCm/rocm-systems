// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Region round-trip launcher for the pytest-driven integration suite.
//
// This binary performs NO comparison. It writes one region to a local rocpd
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
    std::string{ PHUB_INTEGRATION_TMP_DIR } + "/phub_region_writer_reader.db";

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
    writer.register_track_info(g_track);
    writer.insert_region_data(g_region, g_env);
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

    std::optional<rt::region_data_t> detail;
    rt::timeline_event_t             timeline_event;
    rt::arg_data_list_t              args;
    for(const auto& event : reader.get_events())
    {
        if(event.unique_identifier.type != rt::event_type_t::region) continue;
        detail         = reader.get_region_details(event);
        timeline_event = event;
        args           = reader.get_arguments(event);
        if(detail.has_value()) break;
    }

    const auto& d = *detail;

    print("start_timestamp", d.start_timestamp);
    print("end_timestamp", d.end_timestamp);
    print("name", d.name);
    print("extdata", d.extdata);

    print("event.stack_id", d.event->stack_id);
    print("event.parent_stack_id", d.event->parent_stack_id);
    print("event.correlation_id", d.event->correlation_id);
    print("event.event_category", d.event->event_category);
    print("event.extdata", d.event->extdata);

    if(timeline_event.track)
    {
        print("sample.track.name", timeline_event.track->name);
        print("sample.track.extdata", timeline_event.track->extdata);
        if(timeline_event.track->thread_info)
        {
            print("sample.track.thread_id", timeline_event.track->thread_info->thread_id);
        }
    }

    print("arg_count", args.size());
    if(!args.empty())
    {
        print("arg.0.position", args.front()->position);
        print("arg.0.type", args.front()->type);
        print("arg.0.name", args.front()->name);
        print("arg.0.value", args.front()->value);
        print("arg.0.extdata", args.front()->extdata);
    }
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
