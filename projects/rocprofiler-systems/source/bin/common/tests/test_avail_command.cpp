// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/avail_command.hpp"

#include <gtest/gtest.h>

#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

using rocprofsys::cli::avail_options;
using rocprofsys::cli::parse_avail_options;
using rocprofsys::cli::print_avail_help;
using rocprofsys::cli::print_avail_snapshot;
using rocprofsys::cli::run_avail;

namespace
{
struct argv_builder
{
    std::vector<std::string> storage;
    std::vector<char*>       ptrs;

    explicit argv_builder(std::initializer_list<const char*> args)
    {
        storage.reserve(args.size());
        for(auto arg : args)
            storage.emplace_back(arg);
        ptrs.reserve(storage.size());
        for(auto& entry : storage)
            ptrs.push_back(entry.data());
    }

    [[nodiscard]] int    argc() noexcept { return static_cast<int>(storage.size()); }
    [[nodiscard]] char** argv() noexcept { return ptrs.data(); }
};

avail_options
parse(std::initializer_list<const char*> args)
{
    auto builder = argv_builder{ args };
    return parse_avail_options(builder.argc(), builder.argv());
}

rocprofsys::avail::device_record
make_device(std::size_t index, std::string name, std::string bdf)
{
    auto record            = rocprofsys::avail::device_record{};
    record.agent_handle    = index + 1;
    record.index           = index;
    record.name            = std::move(name);
    record.pci_bdf         = std::move(bdf);
    record.runtime_visible = true;
    return record;
}
}  // namespace

TEST(avail_command_test, no_options_requests_the_summary)
{
    const auto options = parse({ "rocsys" });
    EXPECT_TRUE(options.valid);
    EXPECT_TRUE(options.summary());
    EXPECT_FALSE(options.devices);
    EXPECT_FALSE(options.gpu_counters);
    EXPECT_FALSE(options.cpu_counters);
    EXPECT_FALSE(options.traces);
}

TEST(avail_command_test, summary_asks_for_devices_but_not_counters)
{
    const auto request = parse({ "rocsys" }).to_request();
    EXPECT_TRUE(request.devices);
    EXPECT_FALSE(request.gpu_counters);
    EXPECT_FALSE(request.traces);
}

TEST(avail_command_test, selectors_are_parsed_and_combinable)
{
    const auto devices_only = parse({ "rocsys", "--devices" });
    EXPECT_TRUE(devices_only.devices);
    EXPECT_FALSE(devices_only.gpu_counters);
    EXPECT_FALSE(devices_only.summary());

    const auto counters_only = parse({ "rocsys", "--gpu-counters" });
    EXPECT_FALSE(counters_only.devices);
    EXPECT_TRUE(counters_only.gpu_counters);

    const auto cpu_counters = parse({ "rocsys", "--cpu-counters" });
    EXPECT_TRUE(cpu_counters.cpu_counters);
    EXPECT_FALSE(cpu_counters.summary());

    const auto traces_only = parse({ "rocsys", "--traces" });
    EXPECT_TRUE(traces_only.traces);
    EXPECT_FALSE(traces_only.devices);
    EXPECT_FALSE(traces_only.summary());
    EXPECT_TRUE(traces_only.to_request().traces);
    EXPECT_FALSE(traces_only.to_request().devices);

    const auto list_ops = parse({ "rocsys", "--list-operations", "HIP_RUNTIME_API" });
    EXPECT_EQ(list_ops.list_operations, "hip_runtime_api");
    EXPECT_FALSE(list_ops.summary());
    EXPECT_TRUE(list_ops.to_request().traces);
    EXPECT_FALSE(list_ops.to_request().devices);

    const auto list_ops_eq = parse({ "rocsys", "--list-operations=marker_api" });
    EXPECT_EQ(list_ops_eq.list_operations, "marker_api");

    const auto both = parse({ "rocsys", "--devices", "--gpu-counters", "--traces" });
    EXPECT_TRUE(both.devices);
    EXPECT_TRUE(both.gpu_counters);
    EXPECT_TRUE(both.traces);
    EXPECT_TRUE(both.valid);

    const auto no_pager = parse({ "rocsys", "--gpu-counters", "--no-pager" });
    EXPECT_TRUE(no_pager.no_pager);
    EXPECT_TRUE(no_pager.gpu_counters);
    EXPECT_TRUE(no_pager.valid);

    const auto output_file =
        parse({ "rocsys", "--gpu-counters", "--output", "counters.txt" });
    EXPECT_EQ(output_file.output_path, "counters.txt");
    EXPECT_TRUE(output_file.gpu_counters);

    const auto output_eq = parse({ "rocsys", "--output=out.json" });
    EXPECT_EQ(output_eq.output_path, "out.json");
    EXPECT_TRUE(output_eq.summary());
}

TEST(avail_command_test, frozen_selectors_parse_before_their_backends_exist)
{
    for(const char* flag : { "--cpu-devices", "--nic-devices", "--cpu-metrics",
                             "--gpu-metrics", "--nic-metrics", "--storage-metrics" })
    {
        const auto options = parse({ "rocsys", flag });
        EXPECT_TRUE(options.valid) << flag;
        EXPECT_FALSE(options.summary()) << flag;
    }

    // None of the pending selectors trigger a catalog query yet.
    const auto request = parse({ "rocsys", "--nic-metrics" }).to_request();
    EXPECT_FALSE(request.devices);
    EXPECT_FALSE(request.gpu_counters);
    EXPECT_FALSE(request.traces);
}

TEST(avail_command_test, pending_listings_report_progress_instead_of_failing)
{
    auto builder = argv_builder{ { "rocsys", "avail", "--nic-devices", "--gpu-metrics",
                                   "--no-pager" } };
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run_avail(builder.argc(), builder.argv(), out, err), 0);

    const auto text = out.str();
    EXPECT_NE(text.find("NIC devices"), std::string::npos);
    EXPECT_NE(text.find("GPU metrics"), std::string::npos);
    EXPECT_NE(text.find("not implemented yet"), std::string::npos);
}

TEST(avail_command_test, help_topics_are_accepted_and_validated)
{
    for(const char* topic : { "--help=devices", "--help=traces", "--help=counters",
                              "--help=metrics", "--help=output" })
    {
        auto               builder = argv_builder{ { "rocsys", "avail", topic } };
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(run_avail(builder.argc(), builder.argv(), out, err), 0) << topic;
        EXPECT_NE(out.str().find("not available yet"), std::string::npos) << topic;
    }

    auto               builder = argv_builder{ { "rocsys", "avail", "--help=bogus" } };
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run_avail(builder.argc(), builder.argv(), out, err), 1);
    EXPECT_NE(err.str().find("unknown help topic"), std::string::npos);
}

TEST(avail_command_test, the_avail_verb_is_tolerated_in_argv)
{
    const auto options = parse({ "rocsys", "avail", "--devices" });
    EXPECT_TRUE(options.valid);
    EXPECT_TRUE(options.devices);
}

TEST(avail_command_test, help_flags_are_recognized)
{
    for(const char* flag : { "-h", "-?", "--help" })
    {
        const auto options = parse({ "rocsys", flag });
        EXPECT_TRUE(options.help) << flag;
        EXPECT_TRUE(options.valid) << flag;
    }
}

TEST(avail_command_test, an_unknown_option_is_rejected_with_a_hint)
{
    const auto options = parse({ "rocsys", "--settings" });
    EXPECT_FALSE(options.valid);
    EXPECT_NE(options.error_message.find("--settings"), std::string::npos);
    EXPECT_NE(options.error_message.find("--help"), std::string::npos);
}

TEST(avail_command_test, output_without_a_path_is_rejected)
{
    const auto options = parse({ "rocsys", "--gpu-counters", "--output" });
    EXPECT_FALSE(options.valid);
    EXPECT_NE(options.error_message.find("--output"), std::string::npos);
}

// The help text is the phase-1 interface contract: the sections and the flag
// names must not change as later phases fill in the backends.
TEST(avail_command_test, help_text_documents_the_frozen_surface)
{
    std::ostringstream out;
    print_avail_help(out, "rocsys");
    const auto text = out.str();

    for(const char* expected : { "Usage: rocsys avail",
                                 "Query",
                                 "Hardware",
                                 "Traces",
                                 "Counters",
                                 "Metrics",
                                 "Output",
                                 "Examples",
                                 "See also",
                                 "--devices",
                                 "--cpu-devices",
                                 "--nic-devices",
                                 "--traces",
                                 "--list-operations",
                                 "--gpu-counters",
                                 "--cpu-counters",
                                 "--cpu-metrics",
                                 "--gpu-metrics",
                                 "--nic-metrics",
                                 "--storage-metrics",
                                 "--output FILE",
                                 "--no-pager",
                                 "-h, --help",
                                 "--help=TOPIC",
                                 "rocsys profile --help" })
    {
        EXPECT_NE(text.find(expected), std::string::npos) << expected;
    }

    // Avail is a capability query: no environment variables and no collection
    // knobs belong on the first screen.
    EXPECT_EQ(text.find("ROCPROFSYS_"), std::string::npos);
}

TEST(avail_command_test, run_avail_help_writes_to_a_stringstream_without_paging)
{
    auto               builder = argv_builder{ { "rocsys", "avail", "--help" } };
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run_avail(builder.argc(), builder.argv(), out, err), 0);
    EXPECT_NE(out.str().find("Usage: rocsys avail"), std::string::npos);
    EXPECT_TRUE(err.str().empty());
}

TEST(avail_command_test, summary_reports_the_device_count)
{
    auto snapshot            = rocprofsys::avail::catalog_snapshot{};
    snapshot.devices_queried = true;
    snapshot.devices         = { make_device(0, "gfx942", "0000:0c:00.0"),
                                 make_device(1, "gfx942", "0000:22:00.0") };

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(print_avail_snapshot(snapshot, parse({ "rocsys" }), out, err));

    EXPECT_NE(out.str().find("What this machine can profile."), std::string::npos);
    EXPECT_NE(out.str().find("GPU devices : 2"), std::string::npos);
    EXPECT_NE(out.str().find("rocsys avail --devices"), std::string::npos);
    EXPECT_NE(out.str().find("rocsys avail --gpu-counters"), std::string::npos);
    EXPECT_NE(out.str().find("rocsys avail --traces"), std::string::npos);
    EXPECT_NE(out.str().find("rocsys avail --help"), std::string::npos);
    EXPECT_EQ(out.str().find("SDK"), std::string::npos);
    EXPECT_TRUE(err.str().empty());
}

TEST(avail_command_test, summary_says_so_when_no_gpu_is_present)
{
    auto snapshot            = rocprofsys::avail::catalog_snapshot{};
    snapshot.devices_queried = true;

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(print_avail_snapshot(snapshot, parse({ "rocsys" }), out, err));

    EXPECT_NE(out.str().find("GPU devices : 0 (none detected)"), std::string::npos);
    EXPECT_TRUE(err.str().empty());
}

TEST(avail_command_test, device_listing_shows_identity_and_smi_enrichment)
{
    auto device            = make_device(0, "gfx942", "0000:0c:00.0");
    device.product_name    = "Instinct MI300X";
    device.smi_market_name = "AMD Instinct MI300X";

    auto snapshot            = rocprofsys::avail::catalog_snapshot{};
    snapshot.devices_queried = true;
    snapshot.devices         = { device };

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(
        print_avail_snapshot(snapshot, parse({ "rocsys", "--devices" }), out, err));

    const auto text = out.str();
    EXPECT_NE(text.find("Available GPUs that can be profiled."), std::string::npos);
    EXPECT_NE(text.find("[0] gfx942"), std::string::npos);
    EXPECT_NE(text.find("0000:0c:00.0"), std::string::npos);
    EXPECT_NE(text.find("AMD Instinct MI300X"), std::string::npos);
}

TEST(avail_command_test, counter_listing_groups_by_device_and_shows_dimensions)
{
    auto counter        = rocprofsys::avail::counter_record{};
    counter.id          = 1;
    counter.name        = "SQ_WAVES";
    counter.description = "waves sent to SQs";
    counter.block       = "SQ";
    counter.dimensions  = { { "SHADER_ENGINE", 4 }, { "SIMD", 1 } };

    auto group         = rocprofsys::avail::device_counters{};
    group.agent_handle = 1;
    group.index        = 0;
    group.device_name  = "gfx942";
    group.counters     = { counter };

    auto snapshot             = rocprofsys::avail::catalog_snapshot{};
    snapshot.devices_queried  = true;
    snapshot.counters_queried = true;
    snapshot.counter_groups   = { group };

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(
        print_avail_snapshot(snapshot, parse({ "rocsys", "--gpu-counters" }), out, err));

    const auto text = out.str();
    EXPECT_NE(text.find("Hardware counters that can be collected on each GPU."),
              std::string::npos);
    EXPECT_NE(text.find("This listing can be long; use '--output FILE'"),
              std::string::npos);
    EXPECT_NE(text.find("[0] gfx942 (1 counters)"), std::string::npos);
    EXPECT_NE(text.find("SQ_WAVES"), std::string::npos);
    EXPECT_NE(text.find("[SQ]"), std::string::npos);
    EXPECT_NE(text.find("{SHADER_ENGINE[4], SIMD[1]}"), std::string::npos);
}

TEST(avail_command_test, cpu_counter_listing_points_at_output_for_long_dumps)
{
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(print_avail_snapshot(rocprofsys::avail::catalog_snapshot{},
                                     parse({ "rocsys", "--cpu-counters" }), out, err));

    const auto text = out.str();
    EXPECT_NE(text.find("Hardware counters that can be collected on each CPU."),
              std::string::npos);
    EXPECT_NE(text.find("This listing can be long; use '--output FILE'"),
              std::string::npos);
    EXPECT_NE(text.find("not implemented yet"), std::string::npos);
    EXPECT_TRUE(err.str().empty());
}

TEST(avail_command_test, trace_listing_groups_aliases_and_hides_members)
{
    auto domain       = rocprofsys::avail::trace_domain_record{};
    domain.name       = "hip_runtime_api";
    domain.callback   = true;
    domain.is_default = true;

    auto alias          = rocprofsys::avail::trace_domain_record{};
    alias.name          = "hip_api";
    alias.alias         = true;
    alias.alias_members = { "hip_compiler_api", "hip_runtime_api" };

    auto extra = rocprofsys::avail::trace_domain_record{};
    extra.name = "hip_stream";

    auto mpi = rocprofsys::avail::trace_domain_record{};
    mpi.name = "mpi";

    auto vaapi = rocprofsys::avail::trace_domain_record{};
    vaapi.name = "vaapi";

    auto osrt = rocprofsys::avail::trace_domain_record{};
    osrt.name = "osrt";

    auto snapshot           = rocprofsys::avail::catalog_snapshot{};
    snapshot.traces_queried = true;
    snapshot.traces         = { alias, extra, domain, mpi, vaapi, osrt };

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(
        print_avail_snapshot(snapshot, parse({ "rocsys", "--traces" }), out, err));

    const auto text = out.str();
    EXPECT_NE(text.find("Available APIs and runtimes that can be traced."),
              std::string::npos);
    EXPECT_NE(text.find("Defaults: hip_runtime_api, marker_api, kernel_dispatch, "
                        "memory_copy, scratch_memory"),
              std::string::npos);
    EXPECT_EQ(text.find("SDK traces"), std::string::npos);
    EXPECT_NE(text.find("GPU / ROCm runtime"), std::string::npos);
    EXPECT_NE(text.find("Host runtimes"), std::string::npos);
    EXPECT_NE(text.find("Other"), std::string::npos);
    EXPECT_NE(text.find("hip_api"), std::string::npos);
    EXPECT_NE(text.find("alias of hip_compiler_api, hip_runtime_api"), std::string::npos);
    // Alias members are catalogued for --list-operations, not listed as rows.
    EXPECT_EQ(text.find("\n  hip_runtime_api\n"), std::string::npos);
    EXPECT_NE(text.find("\n  mpi\n"), std::string::npos);
    EXPECT_NE(text.find("\n  vaapi\n"), std::string::npos);
    EXPECT_NE(text.find("\n  osrt\n"), std::string::npos);
    EXPECT_EQ(text.find("hip_stream"), std::string::npos);
    EXPECT_EQ(text.find("callback"), std::string::npos);
    EXPECT_NE(text.find("--list-operations NAME"), std::string::npos);
    EXPECT_NE(text.find("--list-operations marker_api"), std::string::npos);
}

TEST(avail_command_test, list_operations_prints_apis_for_a_domain)
{
    auto domain       = rocprofsys::avail::trace_domain_record{};
    domain.name       = "hip_runtime_api";
    domain.operations = { "hipFree", "hipMalloc" };

    auto snapshot           = rocprofsys::avail::catalog_snapshot{};
    snapshot.traces_queried = true;
    snapshot.traces         = { domain };

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(print_avail_snapshot(
        snapshot, parse({ "rocsys", "--list-operations", "hip_runtime_api" }), out, err));

    const auto text = out.str();
    EXPECT_NE(text.find("Operations for hip_runtime_api (2)"), std::string::npos);
    EXPECT_NE(text.find("hipMalloc"), std::string::npos);
    EXPECT_TRUE(err.str().empty());
}

TEST(avail_command_test, list_operations_rejects_an_unknown_domain)
{
    auto snapshot           = rocprofsys::avail::catalog_snapshot{};
    snapshot.traces_queried = true;

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_FALSE(print_avail_snapshot(
        snapshot, parse({ "rocsys", "--list-operations", "not_a_domain" }), out, err));

    EXPECT_TRUE(out.str().empty());
    EXPECT_NE(err.str().find("Domain 'not_a_domain' not found"), std::string::npos);
    EXPECT_NE(err.str().find("rocsys avail --traces"), std::string::npos);
}

TEST(avail_command_test, list_operations_without_a_name_is_rejected)
{
    const auto options = parse({ "rocsys", "--list-operations" });
    EXPECT_FALSE(options.valid);
    EXPECT_NE(options.error_message.find("--list-operations"), std::string::npos);
}

TEST(avail_command_test, diagnostics_go_to_stderr_and_leave_stdout_clean)
{
    auto snapshot            = rocprofsys::avail::catalog_snapshot{};
    snapshot.devices_queried = true;
    snapshot.devices         = { make_device(0, "gfx942", "0000:0c:00.0") };
    snapshot.diagnostics     = { { rocprofsys::avail::source_id::amd_smi,
                                   "amd-smi init failed" } };

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_TRUE(
        print_avail_snapshot(snapshot, parse({ "rocsys", "--devices" }), out, err));

    EXPECT_TRUE(snapshot.degraded());
    EXPECT_NE(out.str().find("gfx942"), std::string::npos);
    EXPECT_EQ(out.str().find("amd-smi init failed"), std::string::npos);
    EXPECT_NE(err.str().find("[amd-smi] amd-smi init failed"), std::string::npos);
}
