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

// The header marks pointer arguments nonnull for callers' compile-time checks. The
// implementation still checks them (a C caller can pass NULL regardless) and returns
// ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT, so it compiles without the attribute.
#define ROCPROFILER_COLLECTOR_NONNULL(...)

#include <rocprofiler-sdk-collector/collector.h>

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk-collector/daemon.hpp"
#include "lib/rocprofiler-sdk-collector/launch.hpp"
#include "lib/rocprofiler-sdk-collector/session.hpp"

#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace
{
namespace fs = std::filesystem;

// The fields a caller must provide: everything up to the reserved padding.
constexpr auto min_config_size = offsetof(rocprofiler_collector_config_t, reserved_padding);

const auto default_providers = std::vector<std::string>{"rocm_hip", "rocm_hsa"};

// LTTng's per-process defaults (4 sub-buffers of a page per CPU) discard events of busy
// threads; these are its per-user defaults, which recorded the rocBLAS tests without loss.
constexpr uint64_t default_subbuffer_size = 512 * 1024;
constexpr uint64_t default_num_subbuffers = 4;

// Lossless: how long a thread waits for buffer space before an event is discarded. Long
// enough for the consumer daemon to empty a sub-buffer under load; short enough that a
// dead consumer slows the target instead of hanging it.
constexpr uint64_t default_blocking_timeout_us = 1000000;

struct collector_state
{
    uint64_t                   id            = 0;
    std::string                output_dir    = {};
    std::string                trace_path    = {};
    session_daemon             daemon        = {};
    trace_session              session       = {};
    std::optional<std::string> previous_home = {};
    bool                       stopped       = false;
    std::set<int>              launched      = {};
    std::vector<std::string>   target_env    = {};  // added to launched targets
};

std::mutex&
get_mutex()
{
    static auto* mtx = new std::mutex{};  // never destroyed: usable during exit
    return *mtx;
}

std::unique_ptr<collector_state>&
get_active()
{
    static auto* active = new std::unique_ptr<collector_state>{};
    return *active;
}

uint64_t&
get_next_id()
{
    static auto id = uint64_t{0};
    return id;
}

collector_state*
lookup(rocprofiler_collector_t collector)
{
    auto& active = get_active();
    return (active && active->id == collector.handle && collector.handle != 0) ? active.get()
                                                                               : nullptr;
}

void
set_lttng_home(collector_state* state, const std::string& home)
{
    if(const auto* prev = std::getenv("LTTNG_HOME")) state->previous_home = std::string{prev};
    ::setenv("LTTNG_HOME", home.c_str(), 1);
}

void
restore_lttng_home(collector_state* state)
{
    if(state->previous_home)
        ::setenv("LTTNG_HOME", state->previous_home->c_str(), 1);
    else
        ::unsetenv("LTTNG_HOME");
}

rocprofiler_collector_status_t
prepare_output_dir(const std::string& output_dir, std::string* absolute)
{
    auto ec   = std::error_code{};
    auto path = fs::absolute(fs::path{output_dir}, ec);
    if(!ec) fs::create_directories(path, ec);
    if(ec || !fs::is_directory(path))
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] cannot create output directory " << output_dir
                   << (ec ? ": " + ec.message() : std::string{});
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_OUTPUT_DIRECTORY;
    }
    if(!fs::is_empty(path, ec) || ec)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] output directory is not empty: " << path;
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_OUTPUT_DIRECTORY;
    }
    *absolute = path.lexically_normal().string();
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

void
teardown(collector_state* state)
{
    destroy_trace_session(&state->session);
    stop_session_daemon(&state->daemon);
    restore_lttng_home(state);
}
}  // namespace
}  // namespace collector
}  // namespace rocprofiler

namespace collector = ::rocprofiler::collector;

ROCPROFILER_COLLECTOR_EXTERN_C_INIT

rocprofiler_collector_status_t
rocprofiler_collector_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
{
    if(major) *major = ROCPROFILER_COLLECTOR_VERSION_MAJOR;
    if(minor) *minor = ROCPROFILER_COLLECTOR_VERSION_MINOR;
    if(patch) *patch = ROCPROFILER_COLLECTOR_VERSION_PATCH;
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
rocprofiler_collector_create(const rocprofiler_collector_config_t* config,
                             rocprofiler_collector_t*              collector)
{
    if(!config || !collector) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    if(config->size < collector::min_config_size)
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_INCOMPATIBLE_ABI;
    if(!config->output_dir || config->output_dir[0] == '\0')
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    if(config->providers && !config->providers[0])
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    if(config->buffer_policy != ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS &&
       config->buffer_policy != ROCPROFILER_COLLECTOR_BUFFER_POLICY_DISCARD)
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;

    auto lock = std::unique_lock{collector::get_mutex()};
    if(collector::get_active()) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE;

    auto state = std::make_unique<collector::collector_state>();

    if(auto status = collector::prepare_output_dir(config->output_dir, &state->output_dir);
       status != ROCPROFILER_COLLECTOR_STATUS_SUCCESS)
        return status;
    state->trace_path = (std::filesystem::path{state->output_dir} / "trace").string();

    auto prefix = std::string{};
    if(config->lttng_prefix)
        prefix = config->lttng_prefix;
    else if(const auto* env = std::getenv("ROCPROFILER_COLLECTOR_LTTNG_PREFIX"))
        prefix = env;
    auto tools = collector::lttng_tools{};
    if(auto status = collector::find_lttng_tools(prefix, &tools);
       status != ROCPROFILER_COLLECTOR_STATUS_SUCCESS)
        return status;

    auto session_config = collector::trace_session_config{};
    session_config.name = "rocprofiler-collector-" + std::to_string(::getpid()) + "-" +
                          std::to_string(++collector::get_next_id());
    session_config.trace_path = state->trace_path;
    session_config.subbuffer_size =
        (config->subbuffer_size > 0) ? config->subbuffer_size : collector::default_subbuffer_size;
    session_config.num_subbuffers =
        (config->num_subbuffers > 0) ? config->num_subbuffers : collector::default_num_subbuffers;
    if(config->buffer_policy == ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS)
    {
        auto timeout = (config->blocking_timeout_us > 0) ? config->blocking_timeout_us
                                                         : collector::default_blocking_timeout_us;
        // LTTng takes a signed timeout; -1 is no limit.
        session_config.blocking_timeout_us =
            (timeout > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                ? -1
                : static_cast<int64_t>(timeout);
        // liblttng-ust only waits in processes that allow it.
        state->target_env.emplace_back("LTTNG_UST_ALLOW_BLOCKING=1");
    }
    if(config->providers)
    {
        for(const auto* const* itr = config->providers; *itr; ++itr)
            session_config.providers.emplace_back(*itr);
    }
    else
    {
        session_config.providers = collector::default_providers;
    }
    auto home = std::string{};
    if(auto status = collector::make_lttng_home(&home);
       status != ROCPROFILER_COLLECTOR_STATUS_SUCCESS)
        return status;
    collector::set_lttng_home(state.get(), home);

    auto log_path = (std::filesystem::path{state->output_dir} / "lttng-sessiond.log").string();
    auto status   = collector::start_session_daemon(tools, home, log_path, &state->daemon);
    if(status == ROCPROFILER_COLLECTOR_STATUS_SUCCESS)
        status = collector::create_trace_session(session_config, &state->session);
    if(status != ROCPROFILER_COLLECTOR_STATUS_SUCCESS)
    {
        collector::teardown(state.get());
        return status;
    }

    state->id               = collector::get_next_id();
    collector->handle       = state->id;
    collector::get_active() = std::move(state);
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
rocprofiler_collector_start(rocprofiler_collector_t collector)
{
    auto  lock  = std::unique_lock{collector::get_mutex()};
    auto* state = collector::lookup(collector);
    if(!state) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    if(state->session.started || state->stopped)
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE;
    return collector::start_trace_session(&state->session);
}

rocprofiler_collector_status_t
rocprofiler_collector_launch(rocprofiler_collector_t collector, const char* const* argv, int* pid)
{
    if(!argv || !argv[0] || !pid) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;

    auto  lock  = std::unique_lock{collector::get_mutex()};
    auto* state = collector::lookup(collector);
    if(!state) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    if(!state->session.started) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE;

    // LTTNG_HOME is already set in this process's environment, so the child inherits it.
    auto status = collector::launch_process(argv, state->target_env, pid);
    if(status == ROCPROFILER_COLLECTOR_STATUS_SUCCESS) state->launched.emplace(*pid);
    return status;
}

rocprofiler_collector_status_t
rocprofiler_collector_wait(rocprofiler_collector_t collector, int pid, int* exit_status)
{
    if(!exit_status) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    {
        auto  lock  = std::unique_lock{collector::get_mutex()};
        auto* state = collector::lookup(collector);
        if(!state || state->launched.count(pid) == 0)
            return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // Not under the lock: the target may run for a long time.
    auto status = collector::wait_process(pid, exit_status);

    auto  lock  = std::unique_lock{collector::get_mutex()};
    auto* state = collector::lookup(collector);
    if(state && status == ROCPROFILER_COLLECTOR_STATUS_SUCCESS) state->launched.erase(pid);
    return status;
}

rocprofiler_collector_status_t
rocprofiler_collector_stop(rocprofiler_collector_t collector)
{
    auto  lock  = std::unique_lock{collector::get_mutex()};
    auto* state = collector::lookup(collector);
    if(!state) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    if(!state->session.started) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE;
    auto status = collector::stop_trace_session(&state->session);
    if(status == ROCPROFILER_COLLECTOR_STATUS_SUCCESS) state->stopped = true;
    return status;
}

rocprofiler_collector_status_t
rocprofiler_collector_get_trace_path(rocprofiler_collector_t collector, const char** path)
{
    if(!path) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    auto  lock  = std::unique_lock{collector::get_mutex()};
    auto* state = collector::lookup(collector);
    if(!state) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
    *path = state->trace_path.c_str();
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
rocprofiler_collector_destroy(rocprofiler_collector_t collector)
{
    auto  lock  = std::unique_lock{collector::get_mutex()};
    auto* state = collector::lookup(collector);
    if(!state) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;

    auto status = collector::destroy_trace_session(&state->session);
    collector::stop_session_daemon(&state->daemon);
    collector::restore_lttng_home(state);
    collector::get_active().reset();
    return status;
}

const char*
rocprofiler_collector_get_status_name(rocprofiler_collector_status_t status)
{
#define ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(NAME)                                               \
    case NAME: return #NAME;

    switch(status)
    {
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_SUCCESS)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_INCOMPATIBLE_ABI)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_LTTNG_NOT_FOUND)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_OUTPUT_DIRECTORY)
        ROCPROFILER_COLLECTOR_STATUS_NAME_CASE(ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH)
        case ROCPROFILER_COLLECTOR_STATUS_LAST: break;
    }
#undef ROCPROFILER_COLLECTOR_STATUS_NAME_CASE
    return nullptr;
}

const char*
rocprofiler_collector_get_status_string(rocprofiler_collector_status_t status)
{
    switch(status)
    {
        case ROCPROFILER_COLLECTOR_STATUS_SUCCESS: return "Success";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR: return "Error";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_INCOMPATIBLE_ABI:
            return "Configuration struct size not supported";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE:
            return "Call not valid in the collector's state";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_LTTNG_NOT_FOUND: return "lttng-sessiond not found";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON:
            return "Session daemon did not start";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION: return "Trace session control failed";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_OUTPUT_DIRECTORY:
            return "Output directory cannot be created, or is not empty";
        case ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH: return "Target could not be started";
        case ROCPROFILER_COLLECTOR_STATUS_LAST: break;
    }
    return nullptr;
}

ROCPROFILER_COLLECTOR_EXTERN_C_FINI
