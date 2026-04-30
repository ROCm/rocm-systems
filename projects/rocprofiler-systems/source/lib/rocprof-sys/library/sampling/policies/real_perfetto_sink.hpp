// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"
#include "logger/debug.hpp"
#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/data/track_name.hpp"
#include "sampling/policies/string_interner.hpp"
#include "sampling/thread_info_data.hpp"

#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::sampling
{

// Production PerfettoSinkPolicy — template-parameterized on ThreadInfoResolverT.
// Config booleans (use_perfetto, annotations) are read from the sampling_config
// that the sampling_service passes at construction.
template <class ThreadInfoResolverT>
class perfetto_sink_impl
{
public:
    perfetto_sink_impl(ThreadInfoResolverT& resolver, bool use_perfetto, bool annotations)
    : resolver_(resolver)
    , use_perfetto_(use_perfetto)
    , annotations_(annotations)
    {}

    void emit_timer(int64_t                          tid, void const* /*info*/,
                    std::vector<timer_sample> const& samples)
    {
        if(!use_perfetto_ || samples.empty()) return;
        emit_samples(tid, samples, category::timer_sampling{},
                     [](auto seq_id, auto sys_id) {
                         return std::string("Thread ") + std::to_string(seq_id) +
                                " (S) " + std::to_string(sys_id);
                     });
    }

    void emit_overflow(int64_t                             tid, void const* /*info*/,
                       std::vector<overflow_sample> const& samples)
    {
        if(!use_perfetto_ || samples.empty()) return;
        emit_samples(
            tid, samples, category::overflow_sampling{}, [](auto seq_id, auto sys_id) {
                return make_thread_track_name(overflow_track_tag{}, seq_id, sys_id);
            });
    }

private:
    template <class CategoryTag, class Sample, class TrackNameFn>
    void emit_samples(int64_t tid, std::vector<Sample> const& samples,
                      CategoryTag category_tag, TrackNameFn make_track_name)
    {
        LOG_DEBUG("[{}] Post-processing metrics for perfetto...", tid);
        LOG_DEBUG("[{}] Post-processing backtraces for perfetto...", tid);

        auto info = resolver_.resolve(tid);
        if(!info) return;

        auto track = tracing::get_perfetto_track(category_tag, make_track_name,
                                                 info->sequent_value, info->system_value);

        const uint64_t beg_ns = std::max(samples.front().beg_ns, info->start_ns);
        const uint64_t end_ns = std::min(samples.back().end_ns, info->stop_ns);

        tracing::push_perfetto_track(category_tag, "samples [rocprof-sys]", track, beg_ns,
                                     [&](::perfetto::EventContext ctx) {
                                         if(annotations_)
                                             tracing::add_perfetto_annotation(
                                                 ctx, "begin_ns", beg_ns);
                                     });

        for(auto const& sample : samples)
        {
            if(!info->is_valid_lifetime(sample.beg_ns, sample.end_ns)) continue;

            for(auto const& frame : sample.stack)
            {
                const char* name = m_interner.intern(
                    frame.name.empty() ? fmt::format("0x{:x}", frame.address)
                                       : frame.name);
                tracing::push_perfetto_track(
                    category_tag, name, track, sample.beg_ns,
                    [&](::perfetto::EventContext ctx) {
                        if(annotations_)
                        {
                            tracing::add_perfetto_annotation(ctx, "begin_ns",
                                                             sample.beg_ns);
                            tracing::add_perfetto_annotation(ctx, "end_ns",
                                                             sample.end_ns);
                            tracing::add_perfetto_annotation(
                                ctx, "pc", fmt::format("0x{:x}", frame.address));
                        }
                    });
                tracing::pop_perfetto_track(category_tag, name, track, sample.end_ns);
            }
        }

        tracing::pop_perfetto_track(category_tag, "samples [rocprof-sys]", track, end_ns,
                                    [&](::perfetto::EventContext ctx) {
                                        if(annotations_)
                                            tracing::add_perfetto_annotation(
                                                ctx, "end_ns", end_ns);
                                    });
    }

    ThreadInfoResolverT& resolver_;
    bool                 use_perfetto_;
    bool                 annotations_;
    string_interner      m_interner;
};

}  // namespace rocprofsys::sampling
