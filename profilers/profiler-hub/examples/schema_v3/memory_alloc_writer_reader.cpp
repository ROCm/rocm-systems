// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Memory-alloc round-trip launcher for the pytest-driven integration suite.
//
// Unlike the self-validating examples, this binary performs NO comparison. It:
//   * writer() - registers the minimal dependency set and writes one memory
//                allocation to a local rocpd database, then flushes it.
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
    std::string{ PHUB_INTEGRATION_TMP_DIR } + "/phub_memory_alloc_launcher.db";

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
    return env;
}

const wt::node_info_t         g_node  = make_node();
const wt::process_info_t      g_proc  = make_process();
const wt::event_data_t        g_event = make_event();
const wt::memory_alloc_data_t g_alloc = make_alloc();
const wt::trace_environment_t g_env   = make_env();

void
remove_db()
{
    std::remove(g_db_path.c_str());
}

// ----------------------------------------------------------------------------
// Writer: register the minimal dependencies, insert, and flush to disk.
// ----------------------------------------------------------------------------
void
writer()
{
    auto storage = std::make_unique<profiler_hub::storage_t>(g_db_path, g_uuid);
    profiler_hub::writer_t writer(std::move(storage));

    writer.register_node_info(g_node);
    writer.register_process_info(g_proc);

    writer.insert_memory_alloc_data(g_alloc, g_env);
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

    std::optional<rt::memory_alloc_data_t> detail;
    for(const auto& event : reader.get_events())
    {
        if(event.unique_identifier.type != rt::event_type_t::memory_allocate) continue;
        detail = reader.get_memory_alloc_details(event);
        if(detail.has_value()) break;
    }

    const auto& d = *detail;

    // memory_alloc_data fields
    print("type", d.type);
    print("level", d.level);
    print("start_timestamp", d.start_timestamp);
    print("end_timestamp", d.end_timestamp);
    print("address", d.address.has_value() ? std::to_string(d.address.value()) : "null");
    print("size", d.size);
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

    return;
}

}  // namespace

int
main()
{
    remove_db();

    bool found = false;
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
