// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/cpu_snap/sampling.hpp"
#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/thread_list/internal_thread_list.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cpu_snap_decoder.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/cpu_snap/config.hpp"
#include "logger/debug.hpp"

#include <cpu_snap/sampling/sampler.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::cpu_snap
{
namespace
{
// Created by setup() and released by post_process(), both on the main thread.
// pause() and resume() run on other threads only in between: the control session that
// calls them subscribes after setup() and shuts down before post_process().
std::unique_ptr<::cpu_snap::sampling::sampler> g_sampler;

// cpu-snap's sink, called on the sampler's worker thread. cpu-snap catches and counts
// what it throws (chunk_framer::emit_to_sink); a chunk lost that way leaves a gap in
// the chunk numbering, which the decoder reports.
void
store_chunk(std::span<const std::byte> chunk)
{
    auto bytes = std::vector<std::uint8_t>(chunk.size());
    std::memcpy(bytes.data(), chunk.data(), chunk.size());
    trace_cache::get_buffer_storage().store(
        trace_cache::cpu_snap_sample{ std::move(bytes) });
}
}  // namespace

void
setup()
{
    auto const settings =
        sampler_settings{ .cputime = config::get_setting_value<bool>(
                                         std::string{ env_vars::SAMPLING_CPUTIME })
                                         .value_or(false),
                          .freq_hz = config::get_sampling_freq() };
    auto cfg           = make_sampler_config(settings);
    cfg.streaming.sink = &store_chunk;
    cfg.logger         = &forward_log_record;

    // Never sample rocprofsys's own threads. The scope blocks creators of excluded
    // threads, so a child is never listed before it is in the internal thread list.
    cfg.threads.is_excluded = [](pid_t tid) {
        return thread_list::get_internal_thread_list().is_excluded(tid);
    };
    cfg.threads.rescan_guard = [] {
        return std::make_shared<thread_list::internal_thread_list::rescan_scope>(
            thread_list::get_internal_thread_list().begin_rescan());
    };

    g_sampler = std::make_unique<::cpu_snap::sampling::sampler>(std::move(cfg));
    g_sampler->start();
    LOG_INFO("{}: sampling wall time (signal trigger) at {} Hz",
             trace_cache::k_cpu_snap_display_name, settings.freq_hz);
}

void
pause() noexcept
{
    if(g_sampler)
    {
        g_sampler->pause();
    }
}

void
resume() noexcept
{
    if(g_sampler)
    {
        g_sampler->resume();
    }
}

void
stop()
{
    if(g_sampler)
    {
        g_sampler->stop();
    }
}

void
post_process()
{
    if(!g_sampler)
    {
        return;
    }

    LOG_INFO("{}: {} samples on up to {} threads; {} lost by the kernel, {} dropped by "
             "the pipeline, {} internal errors, {} failed thread rescans",
             trace_cache::k_cpu_snap_display_name, g_sampler->total_samples(),
             g_sampler->tracked_threads(), g_sampler->lost_samples(),
             g_sampler->streaming_dropped_samples(),
             g_sampler->streaming_internal_errors(), g_sampler->rescan_failures());

    if(g_sampler->worker_failed())
    {
        LOG_WARNING("{}: the sampler's worker thread failed; later samples are missing",
                    trace_cache::k_cpu_snap_display_name);
    }

    g_sampler.reset();
}
}  // namespace rocprofsys::cpu_snap
