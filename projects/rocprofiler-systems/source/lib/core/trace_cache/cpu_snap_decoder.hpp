// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"

#include <functional>
#include <memory>
#include <string_view>

namespace rocprofsys::trace_cache
{
/// How logs and messages name cpu-snap to users, who select it with
/// ROCPROFSYS_USE_EXPERIMENTAL_CALLSTACK_SAMPLER.
inline constexpr std::string_view k_cpu_snap_display_name =
    "experimental call-stack sampler";

/**
 * @brief Turns one process's cpu-snap chunks into backtrace_region_sample records.
 *
 * Feed the chunks of a single process in store order, starting with the first one.
 * Every sample becomes one slice per frame, outermost frame first, spanning from the
 * thread's previous sample to this one; a thread's first sample only opens its first
 * slice. Frames are named by cpu-snap's symbolizer (in-process, libdwfl), resolved
 * against the modules loaded when the sample was taken; an address it cannot name
 * keeps "0x<address>". The call_stack JSON carries name, pc and binary path, as the
 * old sampler writes it.
 */
class cpu_snap_decoder
{
public:
    /// Receives each slice. The sample's string views are valid only during the call.
    using emit_fn = std::function<void(const backtrace_region_sample&)>;

    explicit cpu_snap_decoder(emit_fn emit);
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
