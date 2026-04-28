// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// OffloadPolicy — named requirement
// ============================================================================
// Required member functions:
//   void write(int64_t tid, SampleRingBuffer<2048> const& buf) noexcept;
//       Write the sealed buffer for tid. Called from the allocator pool drain
//       path. Concurrent writes from multiple threads are serialized internally
//       (production uses an atomic spinlock — OffloadSpinlock, DEC-5).
//   std::vector<BacktraceRecord> read(int64_t tid);
//       Read all offloaded records for tid. Called single-threaded from
//       SamplingService::post_process(). May throw std::runtime_error on I/O
//       failure — caller routes through FatalErrorPolicy.
//   void reset() noexcept;
//       Close any held file handle. Called once at the end of post_process().
//
// Production: rocprofsys::sampling::TmpFileOffloadStore
//             - opens config::get_tmp_file("sampling")
//             - emits legacy 26-byte ring header + bundle image per DEC-1 / Path A
//             - tracks per-tid file offsets in unordered_map<int64_t, set<pos_type>>
// Test double: rocprofsys::sampling::test::InMemoryOffload
//             - per-tid std::vector<BacktraceRecord> in memory; no /tmp I/O

namespace rocprofsys::sampling
{
class tmpfile_offload_store;
}
namespace rocprofsys::sampling::test
{
struct in_memory_offload;
}
