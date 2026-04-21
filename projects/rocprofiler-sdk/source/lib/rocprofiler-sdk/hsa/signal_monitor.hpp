// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <hsa/hsa.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace rocprofiler::hsa
{
enum class SignalMonitorBackend : uint32_t
{
    auto_select = 0,
    poll        = 1,
    ioctl       = 2
};

using signal_subscription_id_t = uint64_t;
using signal_monitor_callback_t = std::function<bool(hsa_signal_value_t)>;
using signal_load_fn_t = std::function<hsa_signal_value_t(hsa_signal_t)>;
using signal_event_id_fn_t = std::function<bool(hsa_signal_t, uint32_t&)>;
using wait_events_fn_t = std::function<bool(const std::vector<uint32_t>&, uint32_t)>;

struct SignalMonitorOps
{
    signal_load_fn_t     load = {};
    signal_event_id_fn_t get_event_id = {};
    wait_events_fn_t     wait_events = {};
};

struct SignalMonitorConfig
{
    uint32_t poll_interval_us      = 5;
    uint32_t active_spin_window_us = 200;
    uint32_t callback_threads      = 2;
    bool     allow_poll_fallback   = true;
};

namespace detail
{
std::shared_ptr<class SignalMonitor>
create_polling_signal_monitor(const SignalMonitorConfig& cfg, SignalMonitorOps ops);

std::shared_ptr<class SignalMonitor>
create_ioctl_signal_monitor(const SignalMonitorConfig& cfg, SignalMonitorOps ops);
}  // namespace detail

class SignalMonitor
{
public:
    virtual ~SignalMonitor() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual signal_subscription_id_t subscribe(hsa_signal_t signal,
                                               hsa_signal_condition_t condition,
                                               hsa_signal_value_t compare_value,
                                               signal_monitor_callback_t cb) = 0;
    virtual bool unsubscribe(signal_subscription_id_t id) = 0;
};

bool evaluate_signal_condition(hsa_signal_condition_t condition,
                               hsa_signal_value_t current,
                               hsa_signal_value_t compare_value);

SignalMonitorBackend parse_signal_monitor_backend_env();
std::shared_ptr<SignalMonitor> create_signal_monitor(SignalMonitorBackend backend,
                                                     const SignalMonitorConfig& cfg,
                                                     SignalMonitorOps ops = {});
}  // namespace rocprofiler::hsa
