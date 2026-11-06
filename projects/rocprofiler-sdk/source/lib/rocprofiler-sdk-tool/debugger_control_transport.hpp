// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace rocprofiler::tool::debugger_control
{
using json = nlohmann::json;

// Control transport only. The participant owns the meaning of start and stop.
class worker
{
public:
    using control_fn = std::function<void(bool, uint64_t)>;
    worker(std::string endpoint, std::string session, json identity, control_fn control);
    ~worker();
    worker(const worker&) = delete;
    worker& operator=(const worker&) = delete;
    void    start();
    void    finish();
    void    close_after_fork();

private:
    void serve();
    bool send(json message);
    void stop(const char* reason);

    std::string endpoint_;
    std::string session_;
    json        identity_;
    control_fn  control_;
    std::thread thread_;
    int         owner_    = 0;
    int         socket_   = -1;
    int         timer_    = -1;
    int         wake_     = -1;
    uint64_t    capture_  = 0;
    uint64_t    deadline_ = 0;
    bool        active_   = false;
    json        last_stop_;
};

uint64_t
monotonic_ns();
}  // namespace rocprofiler::tool::debugger_control
