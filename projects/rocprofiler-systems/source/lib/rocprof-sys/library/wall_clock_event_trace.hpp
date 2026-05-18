// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace wall_clock_event_trace
{
void
session_reset();

/// OS thread id for wall-clock host stacks (matches \c tim::threading::get_sys_tid()).
/// Kept for ABI compatibility with object files that still reference this symbol.
std::int64_t
stack_thread_id();

void
push_region(std::int64_t thread_id, const std::string& name);
void
pop_region(std::int64_t thread_id, std::string_view name);

/// ENTER/EXIT timestamps supplied by caller (e.g. \c rocprofiler_timestamp_t values from
/// ROCprofiler callbacks or buffer records). Both steady and wall fields use the same
/// nanosecond value so replay ordering and inclusive duration match the profiler domain.
/// When \p rocprofiler_correlation_internal is non-zero, it must match
/// \c rocprofiler_correlation_id_t::internal for this scope so buffered GPU activity can
/// attach under the correct parent via the ancestor correlation id.
void
push_region_ts(std::int64_t thread_id, const std::string& name, std::uint64_t steady_ns,
               std::uint64_t wall_ns, std::uint64_t rocprofiler_correlation_internal = 0);
void
pop_region_ts(std::int64_t thread_id, std::string_view name, std::uint64_t steady_ns,
              std::uint64_t wall_ns, std::uint64_t rocprofiler_correlation_internal = 0);

/// Synthetic ENTER/EXIT for buffered GPU intervals without touching the host stack.
/// Parent is resolved from \p rocprofiler_ancestor_internal when that id is registered
/// from a callback scope; otherwise the current stack top is used (flush-time parent).
void
emit_buffered_wall_clock_interval(std::int64_t thread_id, const std::string& name,
                                  std::uint64_t beg_ns, std::uint64_t end_ns,
                                  std::uint64_t rocprofiler_ancestor_internal = 0);

void
push_pthread_create(std::int64_t parent_thread_id, const std::string& name);
void
pop_pthread_create(std::int64_t parent_thread_id, std::string_view name);

void
push_start_thread(std::int64_t parent_thread_id, std::int64_t child_thread_id,
                  const std::string& name);
void
pop_start_thread(std::int64_t child_thread_id, std::string_view name);
}  // namespace wall_clock_event_trace
}  // namespace rocprofsys
