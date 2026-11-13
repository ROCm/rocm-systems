// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// rocprofv3-collector: rocprofv3's out-of-process path.
//
//   rocprofv3-collector -- <application> [arguments...]
//
// rocprofv3 sets its ROCPROF_* environment as for the in-process path, then runs this
// program instead of the application. It runs the application under the collector, with
// no profiler code in it, then writes rocprofv3's output files from the collector's
// records with the same output library as the in-process tool, one set per traced process.
// It exits with the application's exit status.

#include <rocprofiler-sdk-collector/collector.h>
#include <rocprofiler-sdk-collector/records.h>

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/output/buffered_output.hpp"
#include "lib/output/generateCSV.hpp"
#include "lib/output/generateJSON.hpp"
#include "lib/output/generateOTF2.hpp"
#include "lib/output/generatePerfetto.hpp"
#include "lib/output/generateRocpd.hpp"
#include "lib/output/generateStats.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/output_key.hpp"

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace
{
namespace fs   = std::filesystem;
namespace tool = ::rocprofiler::tool;

constexpr auto log_prefix = "[rocprofv3-collector] ";

// ------------------------------------------------------------------------------------ //
//  Running the application
// ------------------------------------------------------------------------------------ //

std::atomic<int> target_pid = 0;

// Forward a signal to the application; it decides how to end, and we report its status.
void
forward_signal(int signo)
{
    if(auto pid = target_pid.load(); pid > 0) ::kill(pid, signo);
}

void
install_signal_forwarding()
{
    struct sigaction action = {};
    action.sa_handler       = forward_signal;
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    for(auto signo : {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2})
        ::sigaction(signo, &action, nullptr);
}

bool
check(rocprofiler_collector_status_t status, const char* what)
{
    if(status == ROCPROFILER_COLLECTOR_STATUS_SUCCESS) return true;
    std::fprintf(stderr,
                 "%s%s failed: %s\n",
                 log_prefix,
                 what,
                 rocprofiler_collector_get_status_string(status));
    return false;
}

// The capture directory: under ROCPROF_TMPDIR (the output directory by default), removed
// at the end unless ROCPROF_COLLECTOR_KEEP_TRACE is set.
std::string
make_capture_dir(const tool::output_config& cfg)
{
    auto base = fs::path{tool::format_path(cfg.tmp_directory)};
    auto ec   = std::error_code{};
    fs::create_directories(base, ec);
    auto templ = (base / ".rocprofv3-collector-XXXXXX").string();
    auto buf   = std::vector<char>{templ.begin(), templ.end()};
    buf.emplace_back('\0');
    if(::mkdtemp(buf.data()) == nullptr)
    {
        std::fprintf(stderr,
                     "%scannot create a capture directory in %s: %s\n",
                     log_prefix,
                     base.c_str(),
                     std::strerror(errno));
        return {};
    }
    return buf.data();
}

// ------------------------------------------------------------------------------------ //
//  Records
// ------------------------------------------------------------------------------------ //

struct trace_domains
{
    bool hip_runtime  = false;
    bool hip_compiler = false;

    bool wants(rocprofiler_buffer_tracing_kind_t kind) const
    {
        return (kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT && hip_runtime) ||
               (kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT && hip_compiler);
    }
};

struct read_state
{
    trace_domains           domains    = {};
    std::set<uint32_t>      pids       = {};  // first pass: the traced processes
    uint32_t                pid        = 0;   // second pass: the process being written
    uint64_t                written    = 0;
    rocprofiler_collector_t collector  = {};
    std::string             schema_dir = {};
};

void
collect_pids(const rocprofiler_buffer_tracing_hip_api_ext_record_t* record,
             const rocprofiler_collector_call_info_t*               info,
             void*                                                  user_data)
{
    auto* state = static_cast<read_state*>(user_data);
    if(state->domains.wants(record->kind)) state->pids.emplace(info->pid);
}

void
buffer_records(const rocprofiler_buffer_tracing_hip_api_ext_record_t* record,
               const rocprofiler_collector_call_info_t*               info,
               void*                                                  user_data)
{
    auto* state = static_cast<read_state*>(user_data);
    if(info->pid != state->pid || !state->domains.wants(record->kind)) return;
    // The stream a call used is known in process from stream tracing; here it is not.
    tool::write_ring_buffer(
        tool::tool_buffer_tracing_hip_api_ext_record_t{*record, rocprofiler_stream_id_t{}},
        ::domain_type::HIP);
    ++state->written;
}

bool
read_records(read_state* state, rocprofiler_collector_hip_api_callback_t callback)
{
    auto config             = rocprofiler_collector_read_config_t{};
    config.size             = sizeof(config);
    config.schema_dir       = state->schema_dir.empty() ? nullptr : state->schema_dir.c_str();
    config.hip_api_callback = callback;
    config.user_data        = state;
    auto stats              = rocprofiler_collector_read_statistics_t{};
    stats.size              = sizeof(stats);
    auto status             = rocprofiler_collector_read(state->collector, &config, &stats);
    if(status == ROCPROFILER_COLLECTOR_STATUS_ERROR_PROBE_SCHEMA)
        std::fprintf(stderr,
                     "%sthe HIP runtime's probe schema (share/rocm-trace/rocm_hip.json, installed "
                     "with a HIP runtime built with HIP_ENABLE_ROCM_TRACE=ON) was not found; set "
                     "ROCPROFILER_COLLECTOR_SCHEMA_DIR to its directory\n",
                     log_prefix);
    if(!check(status, "reading the trace")) return false;

    if(callback == collect_pids)
    {
        if(stats.discarded_events > 0)
            std::fprintf(stderr,
                         "%swarning: the trace buffers discarded %lu events; the calls they "
                         "belonged to are missing or have no end\n",
                         log_prefix,
                         static_cast<unsigned long>(stats.discarded_events));
        if(stats.calls_args_incomplete > 0)
            std::fprintf(stderr,
                         "%swarning: %lu calls have arguments the trace did not record\n",
                         log_prefix,
                         static_cast<unsigned long>(stats.calls_args_incomplete));
        if(stats.clock_stable == 0)
            std::fprintf(stderr,
                         "%swarning: the system was suspended while tracing; timestamps after "
                         "the resume are early by the suspended time\n",
                         log_prefix);
    }
    return true;
}

// ------------------------------------------------------------------------------------ //
//  Output: the HIP API domain, as the in-process tool writes it
// ------------------------------------------------------------------------------------ //

void
write_outputs(const tool::output_config& cfg, tool::metadata& metadata)
{
    using namespace ::rocprofiler::tool;

    auto hip_output = hip_buffered_output_t{true};
    hip_output.read();
    if(hip_output.get_generator().empty()) return;

    // The other domains have no records on this path: empty outputs, for the writers'
    // signatures.
    auto hsa_output                    = hsa_buffered_output_t{false};
    auto kernel_dispatch_output        = kernel_dispatch_buffered_output_ext_t{false};
    auto memory_copy_output            = memory_copy_buffered_output_ext_t{false};
    auto marker_output                 = marker_buffered_output_t{false};
    auto counters_output               = counter_collection_buffered_output_t{false};
    auto spm_counters_output           = spm_counter_collection_buffered_output_t{false};
    auto scratch_memory_output         = scratch_memory_buffered_output_t{false};
    auto rccl_output                   = rccl_buffered_output_t{false};
    auto ompt_output                   = ompt_buffered_output_t{false};
    auto memory_allocation_output      = memory_allocation_buffered_output_t{false};
    auto kfd_output                    = kfd_buffered_output_t{false};
    auto rocdecode_output              = rocdecode_buffered_output_t{false};
    auto rocjpeg_output                = rocjpeg_buffered_output_t{false};
    auto rocshmem_output               = rocshmem_buffered_output_t{false};
    auto hipfile_output                = hipfile_buffered_output_t{false};
    auto hip_event_output              = hip_event_buffered_output_ext_t{false};
    auto hip_graph_output              = hip_graph_buffered_output_t{false};
    auto pc_sampling_host_trap_output  = pc_sampling_host_trap_buffered_output_t{false};
    auto pc_sampling_stochastic_output = pc_sampling_stochastic_buffered_output_t{false};

    auto agents = metadata.agents;
    std::sort(agents.begin(), agents.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.node_id < rhs.node_id;
    });

    auto contributions = domain_stats_vec_t{};
    if(cfg.stats || cfg.summary_output)
    {
        hip_output.stats = generate_stats(cfg, metadata, hip_output.get_generator());
        if(hip_output.stats) contributions.emplace_back(hip_output.buffer_type_v, hip_output.stats);
    }

    if(cfg.csv_output)
    {
        generate_csv(cfg, metadata, hip_output.get_generator(), hip_output.stats);
        generate_csv(cfg, metadata, agents);
        if(cfg.stats) generate_csv(cfg, metadata, contributions);
    }

    if(cfg.json_output)
    {
        auto json_ar = open_json(cfg);
        json_ar.start_process();
        write_json(json_ar, cfg, metadata, metadata.process_id);
        write_json(json_ar,
                   cfg,
                   metadata,
                   contributions,
                   hip_output.get_generator(),
                   hsa_output.get_generator(),
                   kernel_dispatch_output.get_generator(),
                   memory_copy_output.get_generator(),
                   counters_output.get_generator(),
                   marker_output.get_generator(),
                   scratch_memory_output.get_generator(),
                   kfd_output.get_generator(),
                   rccl_output.get_generator(),
                   memory_allocation_output.get_generator(),
                   rocdecode_output.get_generator(),
                   rocjpeg_output.get_generator(),
                   pc_sampling_host_trap_output.get_generator(),
                   pc_sampling_stochastic_output.get_generator(),
                   spm_counters_output.get_generator(),
                   hip_graph_output.get_generator(),
                   rocshmem_output.get_generator(),
                   hipfile_output.get_generator(),
                   hip_event_output.get_generator());
        json_ar.finish_process();
        close_json(json_ar);
    }

    if(cfg.pftrace_output)
    {
        write_perfetto(cfg,
                       metadata,
                       agents,
                       hip_output.get_generator(),
                       hsa_output.get_generator(),
                       kernel_dispatch_output.get_generator(),
                       memory_copy_output.get_generator(),
                       counters_output.get_generator(),
                       marker_output.get_generator(),
                       scratch_memory_output.get_generator(),
                       rccl_output.get_generator(),
                       memory_allocation_output.get_generator(),
                       rocdecode_output.get_generator(),
                       rocjpeg_output.get_generator());
    }

    if(cfg.rocpd_output)
    {
        write_rocpd(cfg,
                    metadata,
                    agents,
                    hip_output.get_generator(),
                    hsa_output.get_generator(),
                    kernel_dispatch_output.get_generator(),
                    memory_copy_output.get_generator(),
                    marker_output.get_generator(),
                    memory_allocation_output.get_generator(),
                    scratch_memory_output.get_generator(),
                    kfd_output.get_generator(),
                    rccl_output.get_generator(),
                    rocdecode_output.get_generator(),
                    counters_output.get_generator(),
                    spm_counters_output.get_generator(),
                    ompt_output.get_generator(),
                    hip_graph_output.get_generator(),
                    rocshmem_output.get_generator(),
                    hipfile_output.get_generator(),
                    hip_event_output.get_generator());
    }

    if(cfg.otf2_output)
    {
        auto hip_data               = hip_output.load_all();
        auto hsa_data               = hsa_output.load_all();
        auto kernel_dispatch_data   = kernel_dispatch_output.load_all();
        auto memory_copy_data       = memory_copy_output.load_all();
        auto marker_data            = marker_output.load_all();
        auto scratch_memory_data    = scratch_memory_output.load_all();
        auto rccl_data              = rccl_output.load_all();
        auto memory_allocation_data = memory_allocation_output.load_all();
        auto rocdecode_data         = rocdecode_output.load_all();
        auto rocjpeg_data           = rocjpeg_output.load_all();
        write_otf2(cfg,
                   metadata,
                   metadata.process_id,
                   agents,
                   &hip_data,
                   &hsa_data,
                   &kernel_dispatch_data,
                   &memory_copy_data,
                   &marker_data,
                   &scratch_memory_data,
                   &rccl_data,
                   &memory_allocation_data,
                   &rocdecode_data,
                   &rocjpeg_data);
    }

    if(cfg.summary_output) generate_stats(cfg, metadata, contributions);

    hip_output.reset();
}
}  // namespace

int
main(int argc, char** argv)
{
    auto app = std::vector<const char*>{};
    for(int i = 1; i < argc; ++i)
    {
        if(app.empty() && std::strcmp(argv[i], "--") == 0) continue;
        app.emplace_back(argv[i]);
    }
    if(app.empty())
    {
        std::fprintf(stderr, "usage: %s -- <application> [arguments...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    app.emplace_back(nullptr);

    auto cfg   = tool::output_config::load_from_env();
    auto state = read_state{};
    state.domains.hip_runtime =
        ::rocprofiler::common::get_env("ROCPROF_HIP_RUNTIME_API_TRACE", false);
    state.domains.hip_compiler =
        ::rocprofiler::common::get_env("ROCPROF_HIP_COMPILER_API_TRACE", false);
    state.schema_dir = ::rocprofiler::common::get_env<std::string>(
        "ROCPROFILER_COLLECTOR_SCHEMA_DIR", std::string{});

    auto capture = make_capture_dir(cfg);
    if(capture.empty()) return EXIT_FAILURE;
    // The collector wants a new or empty directory.
    auto output_dir = capture + "/capture";
    auto keep_trace = ::rocprofiler::common::get_env("ROCPROF_COLLECTOR_KEEP_TRACE", false);
    auto finish     = [&](int status) {
        if(state.collector.handle != 0)
            check(rocprofiler_collector_destroy(state.collector), "stopping the collector");
        if(keep_trace)
        {
            std::fprintf(stderr, "%strace kept in %s\n", log_prefix, output_dir.c_str());
        }
        else
        {
            auto ec = std::error_code{};
            fs::remove_all(capture, ec);
        }
        return status;
    };

    auto collector_config         = rocprofiler_collector_config_t{};
    collector_config.size         = sizeof(collector_config);
    collector_config.output_dir   = output_dir.c_str();
    const char* const providers[] = {"rocm_hip", nullptr};
    collector_config.providers    = providers;

    if(!check(rocprofiler_collector_create(&collector_config, &state.collector),
              "starting the collector"))
        return finish(EXIT_FAILURE);
    if(!check(rocprofiler_collector_start(state.collector), "starting tracing"))
        return finish(EXIT_FAILURE);

    install_signal_forwarding();
    auto pid = 0;
    if(!check(rocprofiler_collector_launch(state.collector, app.data(), &pid), "launching"))
        return finish(127);
    target_pid.store(pid);
    auto exit_status = 0;
    check(rocprofiler_collector_wait(state.collector, pid, &exit_status),
          "waiting for the application");
    target_pid.store(0);
    check(rocprofiler_collector_stop(state.collector), "stopping tracing");

    // One set of output files per traced process, as the in-process tool writes.
    if(read_records(&state, collect_pids))
    {
        if(state.pids.empty())
            std::fprintf(stderr,
                         "%swarning: no HIP API calls were recorded. The HIP runtime needs its "
                         "trace probes (built with HIP_ENABLE_ROCM_TRACE=ON), and the "
                         "application must use HIP\n",
                         log_prefix);

        auto metadata = tool::metadata{tool::metadata::inprocess{}};
        for(auto traced : state.pids)
        {
            // The command line of the launched application; another process of the tree is
            // gone by now, and is described by its pid.
            auto command_line = std::vector<std::string>{};
            if(static_cast<int>(traced) == pid)
                for(const auto* itr : app)
                    if(itr != nullptr) command_line.emplace_back(itr);
            auto parent = (static_cast<int>(traced) == pid) ? getpid() : 0;

            metadata.set_process_id(static_cast<pid_t>(traced), parent, command_line);
            tool::set_output_process({static_cast<pid_t>(traced), parent, command_line});
            state.pid     = traced;
            state.written = 0;
            if(read_records(&state, buffer_records)) write_outputs(cfg, metadata);
        }
        tool::reset_output_process();
    }

    return finish(exit_status);
}
