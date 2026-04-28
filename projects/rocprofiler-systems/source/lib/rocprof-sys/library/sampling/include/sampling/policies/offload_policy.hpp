// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// EmitterPolicy — named requirement (previously OffloadPolicy; renamed 2026-04-28
//                                    for Variant 2 / Task #31)
// ============================================================================
// Required member functions:
//   template <size_t N, class FatalErrorPolicy>
//   void write(int64_t tid, sample_ring_buffer<N>& buf, FatalErrorPolicy& fatal);
//       Drain the ring buffer for tid into the adapter's internal store. Called
//       from shutdown(tid) drain path. Thread-safe internally.
//   std::vector<backtrace_record> read(int64_t tid);
//       Return all stored records for tid. Called from emit_resolved_to_trace_cache().
//   std::vector<int64_t> tids() const;
//       Return all tids that have records.
//   void reset() noexcept;
//       Clear all stored records.
//   void erase(int64_t tid) noexcept;
//       Remove records for a single tid (called after per-tid emit).
//
// Production: rocprofsys::sampling::trace_cache_offload_adapter
//             - in-memory per-tid store (no /tmp I/O)
// Test double: rocprofsys::sampling::test::in_memory_emitter
//             - per-tid std::vector<backtrace_record> in memory; no /tmp I/O
//             - identical interface; supports inject() seam for tests

namespace rocprofsys::sampling
{
class trace_cache_offload_adapter;
}
namespace rocprofsys::sampling::test
{
struct in_memory_emitter;
}  // namespace rocprofsys::sampling::test
