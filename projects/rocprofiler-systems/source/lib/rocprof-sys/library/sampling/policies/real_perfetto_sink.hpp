// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/config.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"
#include "logger/debug.hpp"
#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/data/track_name.hpp"
#include "sampling/policies/string_interner.hpp"

#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::sampling
{

// Production PerfettoSinkPolicy: emits per-thread Perfetto track events for
// timer / overflow samples. Track names follow DEC-2 (perfetto-timer omits
// the "Timer" qualifier; overflow uses the qualified form). The owned
// string_interner provides pointer-stable string lifetimes for deferred
// Perfetto track-event writes.
class real_perfetto_sink
{
public:
    void emit_timer(int64_t                          tid, void const* /*info*/,
                    std::vector<timer_sample> const& samples)
    {
        if(!config::get_use_perfetto() || samples.empty()) return;
        emit_samples(tid, samples, category::timer_sampling{},
                     // Perfetto timer track historically omits the "Timer"
                     // qualifier (legacy NFR-P-3 form); trace_cache uses the
                     // qualified form. Do not change without revisiting
                     // DEC-2 / TF-13 parity reasoning.
                     [](auto seq_id, auto sys_id) {
                         return std::string("Thread ") + std::to_string(seq_id) +
                                " (S) " + std::to_string(sys_id);
                     });
    }

    void emit_overflow(int64_t                             tid, void const* /*info*/,
                       std::vector<overflow_sample> const& samples)
    {
        if(!config::get_use_perfetto() || samples.empty()) return;
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
        // L41 — matches legacy: "[{}] Post-processing metrics for perfetto..."
        LOG_DEBUG("[{}] Post-processing metrics for perfetto...", tid);
        // L42 — matches legacy: "[{}] Post-processing backtraces for perfetto..."
        LOG_DEBUG("[{}] Post-processing backtraces for perfetto...", tid);

        const auto& thread_inf = thread_info::get(tid, SequentTID);
        if(!thread_inf) return;

        auto track = tracing::get_perfetto_track(category_tag, make_track_name,
                                                 thread_inf->index_data->sequent_value,
                                                 thread_inf->index_data->system_value);

        const uint64_t beg_ns = std::max(samples.front().beg_ns, thread_inf->get_start());
        const uint64_t end_ns = std::min(samples.back().end_ns, thread_inf->get_stop());

        tracing::push_perfetto_track(category_tag, "samples [rocprof-sys]", track, beg_ns,
                                     [&](::perfetto::EventContext ctx) {
                                         if(config::get_perfetto_annotations())
                                             tracing::add_perfetto_annotation(
                                                 ctx, "begin_ns", beg_ns);
                                     });

        for(auto const& sample : samples)
        {
            if(!thread_inf->is_valid_lifetime({ sample.beg_ns, sample.end_ns })) continue;

            for(auto const& frame : sample.stack)
            {
                const char* name = m_interner.intern(
                    frame.name.empty() ? fmt::format("0x{:x}", frame.address)
                                       : frame.name);
                tracing::push_perfetto_track(
                    category_tag, name, track, sample.beg_ns,
                    [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
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
                                        if(config::get_perfetto_annotations())
                                            tracing::add_perfetto_annotation(
                                                ctx, "end_ns", end_ns);
                                    });
    }

    string_interner m_interner;
};

}  // namespace rocprofsys::sampling
