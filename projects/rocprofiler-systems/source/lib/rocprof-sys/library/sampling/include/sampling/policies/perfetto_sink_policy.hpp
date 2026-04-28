// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// PerfettoSinkPolicy — named requirement
// ============================================================================
// Required member functions:
//   void emit_timer(int64_t tid, void const* /*info*/,
//                   std::vector<timer_sample> const& samples);
//       Emit Perfetto track events for timer samples. Track name:
//       "Thread {seq} (S) {sys}" (DEC-2). info is unused (reserved for future).
//   void emit_overflow(int64_t tid, void const* /*info*/,
//                      std::vector<overflow_sample> const& samples);
//       Emit Perfetto track events for overflow samples. Track name:
//       "Thread {seq} Overflow (S) {sys}" (DEC-2).
//
//   Annotation keys (NFR-P-3): "file", "pc", "line_address", "lineinfo-{n}",
//   "begin_ns", "end_ns", "inlined".
//
// Production: rocprofsys::sampling::real_perfetto_sink
//             - calls tracing::push_perfetto_track(timer_sampling{}) and
//               tracing::push_perfetto_track(overflow_sampling{})
//             - uses an instance member std::set<std::string> string-pool (intern()) for
//               pointer stability across Perfetto deferred track-event writes
// Test double: rocprofsys::sampling::test::recording_perfetto_sink
//             - records emit_timer / emit_overflow calls for assertions

namespace rocprofsys::sampling
{
class real_perfetto_sink;
}
namespace rocprofsys::sampling::test
{
struct noop_perfetto_sink;
}
