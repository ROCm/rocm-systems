#pragma once

/**
 * @file raw_sample.hpp
 * @brief Record produced by cpu_snap::sampling::sampler and
 *        consumed by cpu_snap::processing::symbols::sample_postprocessor.
 *
 * Split out of sampler.hpp so postprocessor TUs do not have to pull
 * the full sampler declaration (~471 lines, plus its transitive
 * kernel_drain / streaming_pipeline headers).
 *
 * Header-only. C++20.
 */

#include <cstdint>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

namespace cpu_snap::core
{

/**
 * @brief One captured stack sample.
 *
 * Written by the recorder's worker thread, read by callers after
 * stop().
 *
 * @note `frames` is innermost-first (matches the kernel callchain order) and
 *       holds exactly the captured frames.
 */
struct raw_sample
{
    /** Linux tid of the sampled thread. */
    pid_t tid = -1;  // NOLINT(misc-include-cleaner): pid_t via <sys/types.h>/<unistd.h>
    /** Sample timestamp in nanoseconds (perf_event TIME). */
    std::uint64_t timestamp_ns = 0;
    /** Instruction pointer at sample time. */
    std::uint64_t ip = 0;
    /** Callchain frames, innermost-first. */
    std::vector<std::uint64_t> frames;
};

}  // namespace cpu_snap::core
