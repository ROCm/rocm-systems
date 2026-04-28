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
//       Return all stored records for tid. Called single-threaded from
//       sampling_service::post_process() (Task #27 deletes post_process; until
//       then read() bridges the existing fan-out pipeline).
//   std::vector<int64_t> tids() const;
//       Return all tids that have records. Called from post_process() fan-out.
//   void reset() noexcept;
//       Clear all stored records. Called once at the end of post_process().
//
// Production: rocprofsys::sampling::trace_cache_offload_adapter
//             - in-memory per-tid store (no /tmp I/O)
//             - Task #30 extends write() to parse+resolve+emit backtrace_region_sample
//               directly to trace_cache::buffer_storage
// Test double: rocprofsys::sampling::test::in_memory_emitter
//             - per-tid std::vector<backtrace_record> in memory; no /tmp I/O
//             - identical interface; supports inject() seam for post_process tests

namespace rocprofsys::sampling
{
class trace_cache_offload_adapter;
}
namespace rocprofsys::sampling::test
{
struct in_memory_emitter;
struct in_memory_offload;  // legacy; retained until Task #27 cleanup
}  // namespace rocprofsys::sampling::test
