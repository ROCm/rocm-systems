// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// TraceSinkPolicy — named requirement
// ============================================================================
// Required member functions:
//   void store_timer(int64_t tid, std::vector<timer_sample> const& samples);
//       Push post-parse resolved timer samples to the trace cache.
//       Called from emit_resolved_to_trace_cache() after symbol resolution.
//
//   void store_overflow(int64_t tid, std::vector<overflow_sample> const& samples);
//       Push post-parse resolved overflow samples to the trace cache.
//       Called from emit_resolved_to_trace_cache() after symbol resolution.
//
// Production: rocprofsys::sampling::real_trace_cache_sink
//             - converts timer_sample/overflow_sample to
//               trace_cache::backtrace_region_sample and stores via
//               trace_cache::get_buffer_storage().store()
// Test double: rocprofsys::sampling::test::recording_trace_sink
//             - records tid + sample count into m_timer_records / m_overflow_records
//             - exposes accessors for assertions

namespace rocprofsys::sampling
{
class real_trace_cache_sink;
}
namespace rocprofsys::sampling::test
{
struct recording_trace_sink;
}
