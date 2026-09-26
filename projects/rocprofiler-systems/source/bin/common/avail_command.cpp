// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/avail_command.hpp"

#include "avail/records.hpp"
#include "avail/traces.hpp"

#include "common/string_utility.hpp"

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace rocprofsys::cli
{
namespace
{
constexpr std::string_view k_indent      = "  ";
constexpr std::string_view k_sub_indent  = "      ";
constexpr std::size_t      k_name_column = 24;
constexpr std::string_view k_long_listing_hint =
    "This listing can be long; use '--output FILE' to write it to a file.\n";

[[nodiscard]] std::string_view
arg_at(int argc, char** argv, int index) noexcept
{
    if(index < 0 || index >= argc || argv == nullptr || argv[index] == nullptr) return {};
    return argv[index];
}

[[nodiscard]] std::string
take_option_value(int argc, char** argv, int& index, std::string_view arg,
                  std::string_view flag)
{
    const auto prefix = std::string{ flag } + "=";
    if(arg.starts_with(prefix) && arg.size() > prefix.size())
        return std::string{ arg.substr(prefix.size()) };

    const auto next = arg_at(argc, argv, index + 1);
    if(!next.empty() && !next.starts_with('-'))
    {
        ++index;
        return std::string{ next };
    }
    return {};
}

void
pad_to(std::ostream& out, std::size_t printed, std::size_t column)
{
    out << std::string(printed < column ? column - printed : 1, ' ');
}

/// Compact per-instance dimension summary, e.g. "{SHADER_ENGINE[0], SIMD[2]}".
[[nodiscard]] std::string
format_dimensions(const std::vector<avail::counter_dimension>& dimensions)
{
    std::ostringstream oss;
    bool               first = true;
    for(const auto& dim : dimensions)
    {
        oss << (first ? "{" : ", ") << dim.name << '[' << dim.position << ']';
        first = false;
    }
    if(first) return {};
    oss << '}';
    return oss.str();
}

void
print_device(std::ostream& out, const avail::device_record& device)
{
    out << k_indent << '[' << device.index << "] " << device.name << '\n';

    if(!device.product_name.empty())
        out << k_sub_indent << "product   : " << device.product_name << '\n';
    if(!device.vendor_name.empty())
        out << k_sub_indent << "vendor    : " << device.vendor_name << '\n';
    if(!device.pci_bdf.empty())
        out << k_sub_indent << "pci bdf   : " << device.pci_bdf << '\n';

    out << k_sub_indent << "visible   : " << (device.runtime_visible ? "yes" : "no")
        << '\n';

    if(device.smi_market_name)
        out << k_sub_indent << "amd-smi   : " << *device.smi_market_name << '\n';
    if(device.smi_vendor_name)
        out << k_sub_indent << "smi vendor: " << *device.smi_vendor_name << '\n';
}

void
print_counter(std::ostream& out, const avail::counter_record& counter)
{
    out << k_sub_indent << counter.name;
    pad_to(out, counter.name.size(), k_name_column);

    if(!counter.block.empty()) out << '[' << counter.block << "] ";

    if(!counter.description.empty())
        out << counter.description;
    else if(counter.is_derived && !counter.expression.empty())
        out << "derived: " << counter.expression;

    const auto dims = format_dimensions(counter.dimensions);
    if(!dims.empty()) out << ' ' << dims;

    out << '\n';
}

void
print_summary(std::ostream& out, const avail::catalog_snapshot& snapshot)
{
    out << "Query available profiling capabilities on this system.\n";
    out << '\n';
    out << k_indent << "GPU devices : " << snapshot.devices.size();
    if(snapshot.devices.empty()) out << " (none detected)";
    out << '\n';
    out << '\n';
    out << k_indent << "rocsys avail --devices         GPU details\n";
    out << k_indent << "rocsys avail --gpu-counters    GPU hardware counters\n";
    out << k_indent << "rocsys avail --traces          APIs that can be traced\n";
    out << k_indent << "rocsys avail --help            all queries\n";
}

void
print_devices(std::ostream& out, const avail::catalog_snapshot& snapshot)
{
    out << "Available GPUs that can be profiled.\n";
    out << "GPU devices (" << snapshot.devices.size() << ")\n";
    if(snapshot.devices.empty())
    {
        out << k_indent << "none detected\n";
        return;
    }
    for(const auto& device : snapshot.devices)
        print_device(out, device);
}

void
print_counters(std::ostream& out, const avail::catalog_snapshot& snapshot)
{
    out << "Hardware counters that can be collected on each GPU.\n";
    out << k_long_listing_hint;
    out << "GPU counters\n";
    if(snapshot.counter_groups.empty())
    {
        out << k_indent << "none available\n";
        return;
    }

    for(const auto& group : snapshot.counter_groups)
    {
        out << k_indent << '[' << group.index << "] " << group.device_name << " ("
            << group.counters.size() << " counters)\n";
        for(const auto& counter : group.counters)
            print_counter(out, counter);
    }
}

void
print_cpu_counters(std::ostream& out)
{
    out << "Hardware counters that can be collected on each CPU.\n";
    out << k_long_listing_hint;
    out << "CPU counters\n";
    out << k_indent << "not implemented yet\n";
}

/// Listings whose backends land in a later phase. The flag names are part of
/// the phase-1 interface, so they parse and report progress instead of being
/// rejected as unknown options.
void
print_pending_listing(std::ostream& out, std::string_view lead_in,
                      std::string_view heading)
{
    out << lead_in << '\n';
    out << heading << '\n';
    out << k_indent << "not implemented yet\n";
}

void
print_trace_row(std::ostream& out, const avail::trace_domain_record& trace)
{
    out << k_indent << trace.name;
    if(trace.alias)
    {
        pad_to(out, trace.name.size(), k_name_column);
        out << "alias of ";
        for(std::size_t i = 0; i < trace.alias_members.size(); ++i)
        {
            if(i > 0) out << ", ";
            out << trace.alias_members[i];
        }
    }
    out << '\n';
}

void
print_trace_section(std::ostream& out, std::string_view heading,
                    const std::vector<const avail::trace_domain_record*>& rows)
{
    if(rows.empty()) return;
    out << heading << '\n';
    for(const auto* trace : rows)
        print_trace_row(out, *trace);
}

void
print_traces(std::ostream& out, const avail::catalog_snapshot& snapshot)
{
    out << "Available APIs and runtimes that can be traced.\n\n";
    out << "Defaults: hip_runtime_api, marker_api, kernel_dispatch, "
           "memory_copy, scratch_memory.\n\n";

    std::vector<const avail::trace_domain_record*> gpu;
    std::vector<const avail::trace_domain_record*> host;
    std::vector<const avail::trace_domain_record*> other;
    for(const auto& trace : snapshot.traces)
    {
        if(avail::is_hidden_alias_member(trace, snapshot.traces)) continue;
        if(!avail::is_collectable_trace(trace.name)) continue;
        switch(avail::section_for_trace(trace.name))
        {
            case avail::trace_section::gpu_rocm: gpu.push_back(&trace); break;
            case avail::trace_section::host: host.push_back(&trace); break;
            case avail::trace_section::other: other.push_back(&trace); break;
        }
    }

    if(gpu.empty() && host.empty() && other.empty())
    {
        out << k_indent << "none available\n";
    }
    else
    {
        print_trace_section(out, "GPU / ROCm runtime", gpu);
        if(!gpu.empty() && (!host.empty() || !other.empty())) out << '\n';
        print_trace_section(out, "Host runtimes", host);
        if(!host.empty() && !other.empty()) out << '\n';
        print_trace_section(out, "Other", other);
    }

    out << '\n'
        << "Use 'rocsys avail --list-operations NAME' to see operations "
           "for a listed GPU / ROCm runtime trace, e.g. rocsys avail "
           "--list-operations marker_api.\n";
}

void
print_diagnostics(std::ostream& err, const avail::catalog_snapshot& snapshot)
{
    if(snapshot.diagnostics.empty()) return;
    for(const auto& diag : snapshot.diagnostics)
        err << "warning: [" << avail::to_string(diag.source) << "] " << diag.message
            << '\n';
}

[[nodiscard]] const avail::trace_domain_record*
find_trace(const avail::catalog_snapshot& snapshot, std::string_view name)
{
    for(const auto& trace : snapshot.traces)
    {
        if(trace.name == name) return &trace;
    }
    return nullptr;
}

/// Returns false when the domain is unknown so the process can exit 1.
[[nodiscard]] bool
print_operations(std::ostream& out, std::ostream& err,
                 const avail::catalog_snapshot& snapshot, std::string_view domain)
{
    const auto* record = find_trace(snapshot, domain);
    if(record == nullptr)
    {
        err << "Error: Domain '" << domain << "' not found.\n"
            << "Use 'rocsys avail --traces' to see available domains.\n";
        return false;
    }

    if(record->operations.empty())
    {
        err << "Domain '" << domain << "' has no operations.\n";
        return true;
    }

    out << "Operations for " << domain << " (" << record->operations.size() << "):\n";
    for(const auto& operation : record->operations)
        out << k_indent << operation << '\n';
    return true;
}

/// Returns true if a pager process ran. A non-zero wait status after that
/// (user quit, SIGPIPE) must not reprint the listing.
bool
write_to_pager(std::string_view command, std::string_view text)
{
    auto* previous = std::signal(SIGPIPE, SIG_IGN);
    FILE* pipe     = popen(std::string{ command }.c_str(), "w");
    if(pipe == nullptr)
    {
        if(previous != SIG_ERR) std::signal(SIGPIPE, previous);
        return false;
    }

    if(!text.empty()) std::fwrite(text.data(), 1, text.size(), pipe);

    const int status = pclose(pipe);
    if(previous != SIG_ERR) std::signal(SIGPIPE, previous);

    if(WIFEXITED(status) && WEXITSTATUS(status) == 127) return false;
    return true;
}

[[nodiscard]] bool
stdout_is_interactive(std::ostream& out)
{
    return out.rdbuf() == std::cout.rdbuf() && isatty(STDOUT_FILENO) == 1;
}

[[nodiscard]] std::string
resolve_pager_command()
{
    if(const char* pager = std::getenv("PAGER"))
    {
        if(pager[0] == '\0') return {};
        if(std::strcmp(pager, "cat") == 0) return {};
        return pager;
    }

    // -F: quit if the listing fits on one screen (no prompt)
    // -R: pass through ANSI color
    // -X: do not clear the terminal on exit
    return "less -FRX";
}

void
page_or_print(std::ostream& out, std::string_view text, bool no_pager)
{
    if(text.empty()) return;

    if(no_pager || !stdout_is_interactive(out))
    {
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        return;
    }

    auto command = resolve_pager_command();
    if(command.empty())
    {
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        return;
    }

    if(command.find("less") != std::string::npos && std::getenv("LESS") == nullptr)
        setenv("LESS", "FRX", 0);

    if(write_to_pager(command, text)) return;
    if(command.find("less") != std::string::npos && write_to_pager("more", text)) return;
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}
}  // namespace

avail_options
parse_avail_options(int argc, char** argv)
{
    auto options = avail_options{};

    // The dispatcher may or may not have stripped the verb; tolerate both so
    // the adapter can also be driven directly in tests.
    int index = 1;
    if(arg_at(argc, argv, index) == "avail") ++index;

    for(; index < argc; ++index)
    {
        const auto arg = arg_at(argc, argv, index);
        if(arg == "--devices")
            options.devices = true;
        else if(arg == "--cpu-devices")
            options.cpu_devices = true;
        else if(arg == "--nic-devices")
            options.nic_devices = true;
        else if(arg == "--gpu-counters")
            options.gpu_counters = true;
        else if(arg == "--cpu-counters")
            options.cpu_counters = true;
        else if(arg == "--cpu-metrics")
            options.cpu_metrics = true;
        else if(arg == "--gpu-metrics")
            options.gpu_metrics = true;
        else if(arg == "--nic-metrics")
            options.nic_metrics = true;
        else if(arg == "--storage-metrics")
            options.storage_metrics = true;
        else if(arg == "--traces")
            options.traces = true;
        else if(arg == "--list-operations" || arg.starts_with("--list-operations="))
        {
            auto domain = take_option_value(argc, argv, index, arg, "--list-operations");
            if(domain.empty())
            {
                options.valid = false;
                options.error_message =
                    "'--list-operations' requires a domain name.\n"
                    "hint: run 'rocsys avail --traces' to see available domains.";
                return options;
            }

            options.list_operations =
                rocprofsys::utility::string::to_lower(std::move(domain));
        }
        else if(arg == "--output" || arg.starts_with("--output="))
        {
            auto path = take_option_value(argc, argv, index, arg, "--output");
            if(path.empty())
            {
                options.valid         = false;
                options.error_message = "'--output' requires a file path.";
                return options;
            }
            options.output_path = std::move(path);
        }
        else if(arg == "--no-pager")
            options.no_pager = true;
        else if(arg.starts_with("--help="))
        {
            options.help = true;
            options.help_topic =
                rocprofsys::utility::string::to_lower(std::string{ arg.substr(7) });
            if(options.help_topic.empty())
            {
                options.valid         = false;
                options.error_message = "'--help' requires a topic name.";
                return options;
            }
        }
        else if(arg == "-h" || arg == "-?" || arg == "--help")
            options.help = true;
        else
        {
            options.valid = false;
            options.error_message =
                "unrecognized option '" + std::string{ arg } +
                "'\nhint: run 'rocsys avail --help' for supported options.";
            return options;
        }
    }

    return options;
}

void
print_avail_help(std::ostream& out, std::string_view program)
{
    out << "Usage: " << program << " avail [OPTIONS]\n"
        << '\n'
        << "List what this machine can profile. Avail does not start a session.\n"
        << '\n'
        << "Query\n"
        << "  (default)              Capability summary\n"
        << '\n'
        << "  Hardware\n"
        << "    --devices            GPUs on this machine\n"
        << "    --cpu-devices        CPUs / NUMA topology\n"
        << "    --nic-devices        Network / AI NIC interfaces\n"
        << '\n'
        << "  Traces\n"
        << "    --traces             GPU and runtime APIs that can be traced\n"
        << "    --list-operations NAME\n"
           "                         APIs or events inside one trace or alias "
           "from --traces\n"
        << '\n'
        << "  Counters\n"
        << "    --gpu-counters       GPU hardware counters (often long; see --output)\n"
        << "    --cpu-counters       CPU / uncore counters (often long; see --output)\n"
        << '\n'
        << "  Metrics\n"
        << "    --cpu-metrics        Host sampled metrics\n"
        << "    --gpu-metrics        GPU sampled metrics\n"
        << "    --nic-metrics        Network / AI NIC metrics\n"
        << "    --storage-metrics    GPU-direct storage metrics\n"
        << '\n'
        << "Output\n"
        << "  --output FILE          Write the listing to FILE instead of stdout\n"
        << "  --no-pager             Do not page long listings\n"
        << "  -h, --help             This message\n"
        << "  --help=TOPIC           Longer help for one query (devices, traces, "
           "counters,\n"
           "                         metrics, output)\n"
        << '\n'
        << "Examples\n"
        << "  " << program << " avail\n"
        << "  " << program << " avail --devices --gpu-counters\n"
        << "  " << program << " avail --traces\n"
        << "  " << program << " avail --list-operations marker_api\n"
        << "  " << program << " avail --gpu-counters --output counters.txt\n"
        << '\n'
        << "See also\n"
        << "  " << program
        << " profile --help     how to enable traces, counters, and metrics\n"
        << "  " << program << " --help             other subcommands\n";
}

bool
print_avail_snapshot(const avail::catalog_snapshot& snapshot,
                     const avail_options& options, std::ostream& out, std::ostream& err)
{
    auto ok = true;

    if(options.summary())
    {
        print_summary(out, snapshot);
    }
    else
    {
        // Sections are emitted in the order --help lists them, separated by a
        // blank line so combined selectors stay readable.
        auto sections = std::vector<std::function<void(std::ostream&)>>{};

        if(options.devices)
            sections.emplace_back([&](std::ostream& os) { print_devices(os, snapshot); });
        if(options.cpu_devices)
            sections.emplace_back([](std::ostream& os) {
                print_pending_listing(os, "CPUs that can be profiled.", "CPU devices");
            });
        if(options.nic_devices)
            sections.emplace_back([](std::ostream& os) {
                print_pending_listing(os, "Network interfaces that can be profiled.",
                                      "NIC devices");
            });
        if(options.traces)
            sections.emplace_back([&](std::ostream& os) { print_traces(os, snapshot); });
        if(options.gpu_counters)
            sections.emplace_back(
                [&](std::ostream& os) { print_counters(os, snapshot); });
        if(options.cpu_counters)
            sections.emplace_back([](std::ostream& os) { print_cpu_counters(os); });
        if(options.cpu_metrics)
            sections.emplace_back([](std::ostream& os) {
                print_pending_listing(os, "Host metrics that can be sampled.",
                                      "CPU metrics");
            });
        if(options.gpu_metrics)
            sections.emplace_back([](std::ostream& os) {
                print_pending_listing(os, "GPU metrics that can be sampled.",
                                      "GPU metrics");
            });
        if(options.nic_metrics)
            sections.emplace_back([](std::ostream& os) {
                print_pending_listing(os, "Network metrics that can be sampled.",
                                      "NIC metrics");
            });
        if(options.storage_metrics)
            sections.emplace_back([](std::ostream& os) {
                print_pending_listing(os, "Storage metrics that can be sampled.",
                                      "Storage metrics");
            });
        if(!options.list_operations.empty())
            sections.emplace_back([&](std::ostream& os) {
                ok = print_operations(os, err, snapshot, options.list_operations);
            });

        for(std::size_t i = 0; i < sections.size(); ++i)
        {
            if(i > 0) out << '\n';
            sections[i](out);
        }
    }

    print_diagnostics(err, snapshot);
    return ok;
}

int
run_avail(int argc, char** argv, std::ostream& out, std::ostream& err)
{
    const auto options = parse_avail_options(argc, argv);

    if(!options.valid)
    {
        err << "rocsys avail: " << options.error_message << '\n';
        return 1;
    }

    if(options.help)
    {
        if(!options.help_topic.empty())
        {
            static constexpr std::string_view k_topics[] = { "devices", "traces",
                                                             "counters", "metrics",
                                                             "output" };
            if(std::find(std::begin(k_topics), std::end(k_topics), options.help_topic) ==
               std::end(k_topics))
            {
                err << "rocsys avail: unknown help topic '" << options.help_topic
                    << "'\nhint: topics are devices, traces, counters, metrics, "
                       "output.\n";
                return 1;
            }
            out << "Help for '" << options.help_topic
                << "' is not available yet.\nRun 'rocsys avail --help' for the "
                   "available queries.\n";
            return 0;
        }

        // Keep --help on one screen; do not send it through the pager.
        print_avail_help(out, "rocsys");
        return 0;
    }

    // Buffer the listing so a TTY can go through less/more. Pipes, tests,
    // and --no-pager still write @p out unchanged.
    std::ostringstream listing;
    const auto ok = print_avail_snapshot(avail::query_catalog(options.to_request()),
                                         options, listing, err);

    if(!options.output_path.empty())
    {
        std::ofstream file{ options.output_path };
        if(!file)
        {
            err << "rocsys avail: cannot write '" << options.output_path << "'\n";
            return 1;
        }
        file << listing.str();
        return ok ? 0 : 1;
    }

    page_or_print(out, listing.str(), options.no_pager);
    return ok ? 0 : 1;
}
}  // namespace rocprofsys::cli
