// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// CPU test adapter for the shared transport. The launcher uses rocprofv3's SDK backend.
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <stdexcept>
#include "debugger_control_transport.hpp"

namespace
{
using namespace rocprofiler::tool::debugger_control;
worker* control = nullptr;
pid_t   owner   = 0;

__attribute__((constructor)) void
initialize()
{
    auto* endpoint = std::getenv("ROCPROFV3_GDB_SOCKET");
    auto* session  = std::getenv("ROCPROFV3_GDB_SESSION");
    if(!endpoint || !session) return;
    owner = getpid();
    json identity{{"instance", std::to_string(owner) + "_" + std::to_string(monotonic_ns())},
                  {"backend", "test"}};
    control = new worker(endpoint, session, identity, [](bool start, uint64_t) {
        using initialized_fn = int (*)(int*);
        auto initialized =
            reinterpret_cast<initialized_fn>(dlsym(RTLD_DEFAULT, "rocprofiler_is_initialized"));
        int ready = 0;
        if(start && (!initialized || initialized(&ready) || !ready))
            throw std::runtime_error("SDK not initialized");
        using control_fn = int (*)(uint64_t);
        auto fn          = reinterpret_cast<control_fn>(
            dlsym(RTLD_DEFAULT, start ? "roctxProfilerResume" : "roctxProfilerPause"));
        if(!fn || fn(0)) throw std::runtime_error("test control failed");
    });
    control->start();
}

__attribute__((destructor)) void
finalize()
{
    if(getpid() == owner) delete control;
}
}  // namespace
