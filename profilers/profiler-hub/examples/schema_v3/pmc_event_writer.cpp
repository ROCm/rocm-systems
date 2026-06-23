// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// PMC-event launcher for the pytest-driven integration suite.
//
// This binary performs NO comparison. It writes one PMC event to the
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

const wt::agent_unique_id_t    g_agent_uid{ "GPU", 0 };
const wt::pmc_info_unique_id_t g_pmc_uid{ "SQ_WAVES", g_agent_uid };

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
    agent.extdata        = "{test data agent}";
    agent.node_id        = 1;
    agent.process_id     = 1000;
    return agent;
}

wt::track_info_t
make_track()
{
    wt::track_info_t track{};
    track.name       = "integration-pmc-sample-track";
    track.extdata    = "{test data track}";
    track.node_id    = 1;
    track.process_id = 1000;
    track.thread_id  = 100;
    return track;
}

wt::pmc_info_t
make_pmc_info()
{
    wt::pmc_info_t pmc{};
    pmc.unique_id        = g_pmc_uid;
    pmc.target_arch      = "GPU";
    pmc.event_code       = 42;
    pmc.instance_id      = 7;
    pmc.symbol           = "SQ_WAVES";
    pmc.description      = "Wavefronts launched";
    pmc.long_description = "Number of wavefronts launched by shader queues";
    pmc.component        = "SQ";
    pmc.units            = "waves";
    pmc.value_type       = "ABS";
    pmc.block            = "SQ";
    pmc.expression       = "SQ_WAVES";
    pmc.is_constant      = 0;
    pmc.is_derived       = 0;
    pmc.extdata          = "{test data pmc info}";
    pmc.node_id          = 1;
    pmc.process_id       = 1000;
    return pmc;
}

wt::event_data_t
make_event()
{
    wt::event_data_t event{};
    event.stack_id        = 5;
    event.parent_stack_id = 0;
    event.correlation_id  = 5;
    event.event_category  = "pmc_event";
    event.extdata         = "{test data event}";
    return event;
}

wt::sample_data_t
make_sample()
{
    wt::sample_data_t sample{};
    sample.timestamp = 1600000;
    sample.track     = make_track();
    sample.extdata   = "{test data sample}";
    return sample;
}

wt::pmc_event_data_t
make_pmc_event()
{
    wt::pmc_event_data_t event{};
    event.event   = make_event();
    event.value   = 1234.5;
    event.extdata = "{test data pmc event}";
    event.sample  = make_sample();
    return event;
}

const wt::node_info_t      g_node      = make_node();
const wt::process_info_t   g_proc      = make_process();
const wt::thread_info_t    g_thread    = make_thread();
const wt::agent_info_t     g_agent     = make_agent();
const wt::track_info_t     g_track     = make_track();
const wt::pmc_info_t       g_pmc_info  = make_pmc_info();
const wt::pmc_event_data_t g_pmc_event = make_pmc_event();

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
    writer.register_agent_info(g_agent);
    writer.register_track_info(g_track);
    writer.register_pmc_info(g_pmc_info);
    writer.insert_pmc_event_data(g_pmc_event, g_pmc_uid);
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
