// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "debugger_control.hpp"
#include "config.hpp"
#include "debugger_control_transport.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace rocprofiler::tool::debugger_control
{
namespace
{
struct state
{
    rocprofiler_context_id_t                        context{};
    rocprofiler_thread_trace_shader_data_callback_t output = nullptr;
    std::atomic<uint64_t>                           capture{0};
    std::atomic<bool>                               contexts_stopped{false};
    std::unique_ptr<worker>                         transport;
    pid_t                                           owner = getpid();
};

// Stable through callbacks and SDK finalization. A raw-fork child must not destroy
// the inherited std::thread or operate the parent's profiler context.
state* control = nullptr;

void
check(rocprofiler_status_t status)
{
    if(status != ROCPROFILER_STATUS_SUCCESS)
        throw std::runtime_error(std::string{"ATT context control failed: "} +
                                 rocprofiler_get_status_string(status));
}
}  // namespace

bool
requested()
{
    auto* endpoint = std::getenv("ROCPROFV3_GDB_SOCKET");
    return endpoint && *endpoint;
}

rocprofiler_context_id_t
configure(rocprofiler_thread_trace_shader_data_callback_t output)
{
    if(!control) control = new state{};
    control->output = output;
    check(rocprofiler_create_context(&control->context));
    return control->context;
}

bool
owns(rocprofiler_context_id_t context)
{
    return control && context.handle == control->context.handle;
}

void
shader_data_callback(rocprofiler_thread_trace_shader_data_t data, rocprofiler_user_data_t)
{
    // Single-buffer stop completes callbacks before the next capture changes this ID.
    // Never fall back to the ROCTx/consecutive-kernel global trace counter.
    control->output(data, rocprofiler_user_data_t{.value = control->capture.load()});
}

void
start_worker()
{
    if(!control || control->owner != getpid()) return;
    auto* session  = std::getenv("ROCPROFV3_GDB_SESSION");
    auto* endpoint = std::getenv("ROCPROFV3_GDB_SOCKET");
    if(!session || !endpoint ||
       std::string{session}.find_first_not_of("0123456789abcdef") != std::string::npos ||
       std::strlen(session) != 32)
        throw std::runtime_error("invalid ATT debugger session configuration");
    control->transport.reset();
    control->capture.store(0);
    control->contexts_stopped.store(false);
    auto instance  = std::to_string(getpid()) + "_" + std::to_string(monotonic_ns());
    auto directory = common::filesystem::absolute(format_path(get_config().output_path)).string() +
                     "/att_gdb_" + session;
    get_config().output_path = directory + "/process_" + instance;
    json identity{{"instance", instance},
                  {"backend", "sdk"},
                  {"directory", directory},
                  {"output", get_config().output_path}};
    control->transport =
        std::make_unique<worker>(endpoint, session, identity, [](bool start, uint64_t capture) {
            if(start)
            {
                if(control->contexts_stopped.load())
                    throw std::runtime_error("ATT participant is finalizing");
                control->capture.store(capture);
                check(rocprofiler_start_context(control->context));
            }
            else if(!control->contexts_stopped.load())
                check(rocprofiler_stop_context(control->context));
        });
    control->transport->start();
}

void
finalize()
{
    if(control && control->owner == getpid())
    {
        // The SDK stops client contexts before tool_detach/tool_fini. Raw ATT callbacks
        // have completed; the worker only needs to acknowledge exit and disconnect.
        control->contexts_stopped.store(true);
        control->transport.reset();
    }
}
}  // namespace rocprofiler::tool::debugger_control
