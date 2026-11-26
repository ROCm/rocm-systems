// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::trace_cache
{
/// How logs and messages name cpu-snap to users, who select it with
/// ROCPROFSYS_USE_EXPERIMENTAL_CALLSTACK_SAMPLER.
inline constexpr std::string_view k_cpu_snap_display_name =
    "experimental call-stack sampler";

/**
 * @brief Calculates the start of the slice that ends at the current sample.
 *
 * The slice starts at the latest resume in (@p previous_ns, @p current_ns], if
 * there is such a resume, otherwise at @p previous_ns.
 *
 * @param resumes      Times sampling resumed in this process, ascending.
 * @param previous_ns  The thread's previous sample.
 * @param current_ns   The current sample.
 */
[[nodiscard]] inline std::uint64_t
slice_start(std::span<const std::uint64_t> resumes, std::uint64_t previous_ns,
            std::uint64_t current_ns)
{
    auto const first_resume_after_current = std::ranges::upper_bound(resumes, current_ns);
    if(first_resume_after_current == resumes.begin())
    {
        return previous_ns;  // no resume at or before current_ns
    }
    auto const latest_resume = *std::prev(first_resume_after_current);
    return latest_resume > previous_ns ? latest_resume : previous_ns;
}

/**
 * @brief Module prefixes whose frames every processor's cpu_snap_decoder leaves out.
 *
 * Empty when ROCPROFSYS_SAMPLING_KEEP_INTERNAL is ON, otherwise rocprof-sys's own
 * libraries. All outputs use this list, so they show the same call stacks.
 */
[[nodiscard]] std::vector<std::string>
get_cpu_snap_skipped_module_prefixes();

/**
 * @brief Turns one process's cpu-snap chunks into backtrace_region_sample records.
 *
 * Feed the chunks of a single process in store order, starting with the first one.
 * Every sample becomes one slice per frame, outermost frame first, from slice_start()
 * to this sample; a pause stays empty; a thread's first sample draws no slice; it only
 * marks where the thread's first slice starts. Frames are named by cpu-snap's symbolizer
 * (in-process, libdwfl), resolved against the modules loaded when the sample was taken;
 * an address it cannot name keeps "0x<address>". The call_stack JSON carries name, pc and
 * binary path, as the old sampler writes it. Frames whose module file name starts with
 * one of the given prefixes are left out; the frames around them stay.
 */
class cpu_snap_decoder
{
public:
    /// Receives each slice. The sample's string views are valid only during the call.
    using on_slice_fn = std::function<void(const backtrace_region_sample&)>;

    /**
     * @param on_slice                Receives each slice.
     * @param skipped_module_prefixes Frames whose module file name starts with one of
     *                                these are left out; empty keeps every frame.
     * @param sampling_resumes        Timestamps of sampling resumed in this process,
     *                                in accending order.
     */
    cpu_snap_decoder(on_slice_fn                on_slice,
                     std::vector<std::string>   skipped_module_prefixes,
                     std::vector<std::uint64_t> sampling_resumes);
    ~cpu_snap_decoder();

    cpu_snap_decoder(const cpu_snap_decoder&)            = delete;
    cpu_snap_decoder& operator=(const cpu_snap_decoder&) = delete;
    cpu_snap_decoder(cpu_snap_decoder&&)                 = delete;
    cpu_snap_decoder& operator=(cpu_snap_decoder&&)      = delete;

    /// Decodes @p chunk and emits the slices it completes.
    void feed(const cpu_snap_sample& chunk);

    /// Logs a warning for each kind of data loss seen so far.
    void report_losses() const;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};
}  // namespace rocprofsys::trace_cache
