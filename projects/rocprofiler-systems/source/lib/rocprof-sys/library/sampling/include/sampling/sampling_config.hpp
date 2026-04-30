// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Plain configuration data for the sampling subsystem.
// No std::function members — those live in sampling_callbacks.

#include <cstdint>
#include <string>

namespace rocprofsys::sampling
{

struct sampling_config
{
    int realtime_signal = 0;
    int cputime_signal  = 0;
    int overflow_signal = 0;

    double realtime_freq  = 0.0;
    double cputime_freq   = 0.0;
    double realtime_delay = 0.0;
    double cputime_delay  = 0.0;
    double overflow_freq  = 0.0;
    double duration       = 0.0;

    std::string overflow_event = "PERF_COUNT_SW_CPU_CLOCK";

    bool use_causal           = false;
    bool use_perfetto         = false;
    bool perfetto_annotations = false;
    bool trace_legacy         = false;
    bool use_process_sampling = false;
    bool use_amd_smi          = false;
};

}  // namespace rocprofsys::sampling
