// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cpu_snap_decoder.hpp"
#include "logger/debug.hpp"

#include <cpu_snap/sampling/backend_kind.hpp>
#include <cpu_snap/sampling/clock_kind.hpp>
#include <cpu_snap/sampling/logger.hpp>
#include <cpu_snap/sampling/sampler.hpp>

#include <fmt/format.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace rocprofsys::cpu_snap
{
/// rocprof-sys setting values the cpu-snap sampler is built from.
struct sampler_settings
{
    bool   cputime = false;
    double freq_hz = 0.0;
};

/**
 * @brief cpu-snap configuration for @p settings, without sink and logger.
 *
 * Samples wall time with the signal trigger: per-thread POSIX timers and a real-time
 * signal, no perf_event_open. CPU-time sampling is not supported yet. Timestamps use
 * CLOCK_BOOTTIME, the rocprof-sys timeline clock.
 *
 * @throws std::logic_error When CPU time is requested: configure_settings() rejects
 *         that for the cpu-snap engine.
 */
[[nodiscard]] inline ::cpu_snap::sampling::sampler::config
make_sampler_config(const sampler_settings& settings)
{
    if(settings.cputime)
    {
        throw std::logic_error("make_sampler_config: CPU-time sampling is requested; "
                               "configure_settings() should have rejected this");
    }

    auto cfg                  = ::cpu_snap::sampling::sampler::config{};
    cfg.backend               = ::cpu_snap::sampling::backend_kind::signal;
    cfg.signal.sample_freq_hz = static_cast<std::uint32_t>(std::lround(settings.freq_hz));
    cfg.clock                 = ::cpu_snap::sampling::clock_kind::boottime;
    return cfg;
}

/// Forwards a cpu-snap log record to the rocprof-sys logger.
inline void
forward_log_record(const ::cpu_snap::sampling::log_record& record)
{
    using ::cpu_snap::sampling::log_level;
    using trace_cache::k_cpu_snap_display_name;

    switch(record.level)
    {
        case log_level::debug:
            LOG_DEBUG("{}: {}", k_cpu_snap_display_name, record.message);
            return;
        case log_level::info:
            LOG_INFO("{}: {}", k_cpu_snap_display_name, record.message);
            return;
        case log_level::warn:
            LOG_WARNING("{}: {}", k_cpu_snap_display_name, record.message);
            return;
        case log_level::error:
            LOG_ERROR("{}: {}", k_cpu_snap_display_name, record.message);
            return;
        default:
            throw std::logic_error(
                fmt::format("cpu_snap::sampling::log_level {} not handled",
                            static_cast<int>(record.level)));
    }
}
}  // namespace rocprofsys::cpu_snap
