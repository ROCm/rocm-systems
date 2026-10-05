/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Replay Limits
 * @{
 * @ingroup HRRTest
 * GPU-free unit tests for the bounds replay puts on what a malformed archive
 * can make it wait for or allocate: a gap or duplicate in the
 * recorded sequence numbers, a file larger than the cap, and a HIP query that
 * never completes. Each case builds a synthetic archive on disk with the
 * TmpArchive fixture and fails (hangs, then times out) if its bound is removed.
 */

#include "hrr_test_common.hh"
#include "hrr_reader.h"
#include "hrr_replay_limits.h"
#include "hip_playback.h"
#include "hrr/hrr_api_args.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

hrr_event_header make_record(uint64_t seq, uint64_t thread_id = 42) {
  hrr_event_header h{};
  std::memset(&h, 0, sizeof(h));
  h.event_type     = static_cast<uint16_t>(HRR_API_HIPDEVICESYNCHRONIZE);
  h.sequence_id    = seq;
  h.timestamp_ns   = 1000 + seq;
  h.thread_id      = thread_id;
  h.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));
  return h;
}

// RAII synthetic archive directory with a caller-chosen sequence list.
struct TmpArchive {
  fs::path root;

  // Event i is recorded by thread 42 + (i % distinct_threads).
  TmpArchive(const std::string& name, const std::vector<uint64_t>& seqs,
             uint64_t distinct_threads = 1) {
    root = fs::temp_directory_path() / ("hrr_lim_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "blobs");
    fs::create_directories(root / "code_objects");
    std::ofstream f(root / "events.bin", std::ios::binary);
    hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    uint64_t i = 0;
    for (uint64_t s : seqs) {
      hrr_event_header h = make_record(s, 42 + (i++ % distinct_threads));
      f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }
  }
  ~TmpArchive() { fs::remove_all(root); }
  std::string path() const { return root.string(); }
};

// Restores the process-wide file cap when a test ends, pass or fail.
struct FileCapGuard {
  uint64_t saved = hrr::max_file_bytes();
  ~FileCapGuard() { hrr::set_max_file_bytes(saved); }
};

struct ReplayResult {
  hrr::TurnResult last = hrr::TurnResult::Turn;
  size_t          stalled_at = 0;
  uint64_t        observed   = 0;
  uint64_t        wanted     = 0;
};

// Drive the same hand-off dispatch_event uses: each event waits for its turn,
// then hands the turn to sequence_id + 1.
ReplayResult replay(const hrr::Archive& a, uint64_t max_waits) {
  std::atomic<uint64_t> next_seq{a.events.front().header().sequence_id};
  std::atomic<bool> fatal{false};
  ReplayResult r;
  for (size_t i = 0; i < a.events.size(); ++i) {
    const uint64_t seq = a.events[i].header().sequence_id;
    r.last = hrr::wait_for_turn(next_seq, seq, fatal, max_waits, &r.observed);
    if (r.last != hrr::TurnResult::Turn) {
      r.stalled_at = i;
      r.wanted = seq;
      return r;
    }
    next_seq.store(seq + 1, std::memory_order_release);
  }
  return r;
}

}  // namespace

/**
 * Test Description
 * ----------------
 *   - A well-formed sequence replays to the end without stalling.
 */
HRR_TEST_CASE(Unit_HRR_Limits_ContiguousSequenceReplays) {
  TmpArchive arc("contiguous", {0, 1, 2, 3});
  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  ReplayResult r = replay(a, /*max_waits=*/10);
  CHECK(r.last == hrr::TurnResult::Turn);
}

/**
 * Test Description
 * ----------------
 *   - A gap in the recorded sequence (0, 1, 3) leaves event 3 waiting for a
 *     turn nobody hands over. The wait ends as Stalled after the configured
 *     number of waits and reports the numbers: it wanted 3, the replay was
 *     at 2.
 */
HRR_TEST_CASE(Unit_HRR_Limits_SequenceGapEndsReplay) {
  TmpArchive arc("gap", {0, 1, 3});
  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  ReplayResult r = replay(a, /*max_waits=*/20);
  CHECK(r.last == hrr::TurnResult::Stalled);
  CHECK(r.stalled_at == 2);
  CHECK(r.wanted == 3);
  CHECK(r.observed == 2);
}

/**
 * Test Description
 * ----------------
 *   - A duplicate sequence number (0, 1, 1): the second event 1 finds the
 *     replay already past it (at 2) and stalls rather than spinning forever.
 */
HRR_TEST_CASE(Unit_HRR_Limits_DuplicateSequenceEndsReplay) {
  TmpArchive arc("dup", {0, 1, 1});
  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  ReplayResult r = replay(a, /*max_waits=*/20);
  CHECK(r.last == hrr::TurnResult::Stalled);
  CHECK(r.stalled_at == 2);
  CHECK(r.wanted == 1);
  CHECK(r.observed == 2);
}

/**
 * Test Description
 * ----------------
 *   - A wait that another thread cuts short by setting the fatal flag returns
 *     Fatal, not Stalled, so a failing thread still stops the others at once.
 */
HRR_TEST_CASE(Unit_HRR_Limits_FatalFlagEndsWait) {
  std::atomic<uint64_t> next_seq{0};
  std::atomic<bool> fatal{true};
  uint64_t observed = 99;
  CHECK(hrr::wait_for_turn(next_seq, 5, fatal, 1000000, &observed) ==
        hrr::TurnResult::Fatal);
  CHECK(observed == 0);
}

/**
 * Test Description
 * ----------------
 *   - read_file_capped refuses a file above the cap and the message names the
 *     path and the size; read_blob and read_code_object go through it. Raising
 *     the cap to the file's size lets the same file load.
 */
HRR_TEST_CASE(Unit_HRR_Limits_ReadFileRefusesOversize) {
  FileCapGuard guard;
  CHECK(guard.saved == hrr::kDefaultMaxFileBytes);  // max_file_bytes() default
  TmpArchive arc("bigfile", {0});
  const fs::path blob = arc.root / "blobs" / "big.bin";
  {
    std::ofstream f(blob, std::ios::binary);
    std::string bytes(200, 'x');
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  hrr::set_max_file_bytes(100);
  std::vector<uint8_t> data;
  std::string err;
  CHECK_FALSE(hrr::read_file_capped(blob.string(), data, &err));
  CHECK(err.find(blob.string()) != std::string::npos);
  CHECK(err.find("200") != std::string::npos);
  CHECK(data.empty());

  hrr::Archive a;
  a.blobs[hrr::hash_hex(1, 2)] = blob.string();
  a.code_objects[hrr::hash_hex(3, 4)] = blob.string();
  CHECK_FALSE(hrr::read_blob(a, 1, 2, data));
  CHECK_FALSE(hrr::read_code_object(a, 3, 4, data));

  hrr::set_max_file_bytes(200);
  CHECK(hrr::read_file_capped(blob.string(), data, &err));
  CHECK(data.size() == 200);
  CHECK(hrr::read_blob(a, 1, 2, data));
}

/**
 * Test Description
 * ----------------
 *   - A missing file is reported as such, not as a size refusal.
 */
HRR_TEST_CASE(Unit_HRR_Limits_ReadFileMissing) {
  std::vector<uint8_t> data;
  std::string err;
  CHECK_FALSE(hrr::read_file_capped("/nonexistent/hrr/blob", data, &err));
  CHECK(err.find("/nonexistent/hrr/blob") != std::string::npos);
  CHECK(err.find("file cap") == std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - retry_bounded, the loop behind hipEventQuery/hipStreamQuery replay, makes
 *     exactly max_attempts calls when the query never completes, stops at the
 *     first success, and makes no call when the bound is zero.
 */
HRR_TEST_CASE(Unit_HRR_Limits_QueryRetryIsBounded) {
  uint64_t calls = 0;
  CHECK_FALSE(hrr::retry_bounded(25, [&] { ++calls; return false; }));
  CHECK(calls == 25);

  calls = 0;
  CHECK(hrr::retry_bounded(25, [&] { return ++calls == 3; }));
  CHECK(calls == 3);

  calls = 0;
  CHECK_FALSE(hrr::retry_bounded(0, [&] { ++calls; return true; }));
  CHECK(calls == 0);
}

/**
 * Test Description
 * ----------------
 *   - A duplicate sequence number is reported at once: the turn has passed, so
 *     waiting longer cannot help, however large the wait bound is.
 */
HRR_TEST_CASE(Unit_HRR_Limits_DuplicateIsReportedImmediately) {
  std::atomic<uint64_t> next_seq{7};
  std::atomic<bool> fatal{false};
  uint64_t observed = 0;
  const auto t0 = std::chrono::steady_clock::now();
  CHECK(hrr::wait_for_turn(next_seq, 3, fatal, /*max_waits=*/UINT64_MAX, &observed) ==
        hrr::TurnResult::Stalled);
  CHECK(observed == 7);
  CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
}

/**
 * Test Description
 * ----------------
 *   - While a thread holds a turn and is inside its HIP call (busy > 0), a
 *     waiting thread does not count waits, so a call that runs longer than the
 *     bound is not mistaken for a missing event. When the call returns and the
 *     turn is handed over, the wait ends normally; a gap with nothing in
 *     flight still stalls.
 */
HRR_TEST_CASE(Unit_HRR_Limits_NoStallWhileACallIsInFlight) {
  std::atomic<uint64_t> next_seq{4};
  std::atomic<bool> fatal{false};
  std::atomic<int> busy{1};
  hrr::TurnResult result = hrr::TurnResult::Stalled;

  // The bound is tiny, and the call below outlasts it by a wide margin.
  std::thread waiter([&] { result = hrr::wait_for_turn(next_seq, 5, fatal, /*max_waits=*/20, nullptr, &busy); });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  next_seq.store(5);  // the call returns and hands the turn on
  busy.store(0);
  waiter.join();
  CHECK(result == hrr::TurnResult::Turn);

  // A gap with nothing in flight stalls as before.
  std::atomic<uint64_t> stuck{4};
  uint64_t observed = 0;
  CHECK(hrr::wait_for_turn(stuck, 6, fatal, /*max_waits=*/20, &observed, &busy) ==
        hrr::TurnResult::Stalled);
  CHECK(observed == 4);
}

/**
 * Test Description
 * ----------------
 *   - parse_u64 accepts decimal and 0x hex, and rejects what strtoull would
 *     quietly turn into a different number: an empty string, a sign, trailing
 *     text, and a value out of range ("-1" would otherwise switch a cap off).
 */
HRR_TEST_CASE(Unit_HRR_Limits_ParseU64) {
  uint64_t v = 99;
  CHECK(hrr::parse_u64("0", &v));
  CHECK(v == 0);
  CHECK(hrr::parse_u64("5000000", &v));
  CHECK(v == 5000000);
  CHECK(hrr::parse_u64("0x10", &v));
  CHECK(v == 16);
  CHECK(hrr::parse_u64("18446744073709551615", &v));
  CHECK(v == UINT64_MAX);
  for (const char* bad : {"", "-1", "+1", " 5", "5x", "x", "18446744073709551616", "1e3"}) {
    v = 99;
    CHECK_FALSE(hrr::parse_u64(bad, &v));
    CHECK(v == 99);
  }
  CHECK_FALSE(hrr::parse_u64(nullptr, &v));
}

// ---------------------------------------------------------------------------
// Caps on threads, host memory, events and wall time
// ---------------------------------------------------------------------------

// Restores the process-wide event cap when a test ends, pass or fail.
struct EventCapGuard {
  uint64_t saved = hrr::max_events();
  ~EventCapGuard() { hrr::set_max_events(saved); }
};

/**
 * Test Description
 * ----------------
 *   - check_thread_cap allows a recorded thread count up to the cap and
 *     refuses one above it, with the recorded count and the cap in the message.
 */
HRR_TEST_CASE(Unit_HRR_Limits_ThreadCap) {
  std::string why;
  CHECK(hrr::check_thread_cap(256, 256, &why));
  CHECK(why.empty());
  CHECK_FALSE(hrr::check_thread_cap(257, 256, &why));
  CHECK(why.find("257") != std::string::npos);
  CHECK(why.find("256") != std::string::npos);
  CHECK_FALSE(hrr::check_thread_cap(1, 0, nullptr));
}

/**
 * Test Description
 * ----------------
 *   - hrr-playback --multi-thread on an archive recording more threads than
 *     --max-threads exits non-zero with the recorded count in the message,
 *     before it creates a replay thread (and before it touches the GPU, so the
 *     case needs no device).
 */
HRR_TEST_CASE(Unit_HRR_Limits_PlaybackRefusesTooManyThreads) {
  TmpArchive arc("threads", {0, 1, 2, 3, 4, 5}, /*distinct_threads=*/6);
  auto [ret, out] = hrr_playback_merged(arc.root, "--multi-thread --max-threads 4");
  CHECK(ret != 0);
  CHECK(out.find("6 threads") != std::string::npos);
  CHECK(out.find("--max-threads") != std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - load_archive stops at the event cap: an archive with three events fails
 *     to load under a cap of two and loads under a cap of three.
 */
HRR_TEST_CASE(Unit_HRR_Limits_EventCap) {
  EventCapGuard guard;
  CHECK(guard.saved == hrr::kDefaultMaxEvents);  // max_events() default
  TmpArchive arc("events", {0, 1, 2});
  hrr::Archive a;
  hrr::set_max_events(2);
  CHECK_FALSE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 2);

  hrr::Archive b;
  hrr::set_max_events(3);
  CHECK(hrr::load_archive(arc.path(), b));
  CHECK(b.events.size() == 3);
}

/**
 * Test Description
 * ----------------
 *   - hrr-playback --max-events refuses an over-cap archive without a GPU.
 */
HRR_TEST_CASE(Unit_HRR_Limits_PlaybackRefusesTooManyEvents) {
  TmpArchive arc("events_cli", {0, 1, 2});
  auto [ret, out] = hrr_playback_merged(arc.root, "--max-events 2");
  CHECK(ret != 0);
  CHECK(out.find("--max-events") != std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - ByteBudget charges up to its cap and refuses beyond it, leaving the
 *     total unchanged on a refusal; release gives bytes back; a request near
 *     UINT64_MAX does not wrap past the cap.
 */
HRR_TEST_CASE(Unit_HRR_Limits_ByteBudget) {
  CHECK(hrr::ByteBudget().cap() == hrr::kDefaultMaxHostBytes);  // the default cap
  hrr::ByteBudget b(100);
  CHECK(b.cap() == 100);
  CHECK(b.used() == 0);
  CHECK(b.try_charge(60));
  CHECK(b.try_charge(40));
  CHECK(b.used() == 100);
  CHECK_FALSE(b.try_charge(1));
  CHECK(b.used() == 100);
  b.release(30);
  CHECK(b.try_charge(30));
  CHECK_FALSE(b.try_charge(UINT64_MAX));
  CHECK_FALSE(b.try_charge(UINT64_MAX - 50));
  CHECK(b.used() == 100);
  b.release(1000);  // over-release clamps at zero
  CHECK(b.used() == 0);
}

/**
 * Test Description
 * ----------------
 *   - Many threads charging the same budget never take it past the cap.
 */
HRR_TEST_CASE(Unit_HRR_Limits_ByteBudgetConcurrent) {
  hrr::ByteBudget b(1000);
  std::atomic<uint64_t> granted{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t)
    ts.emplace_back([&] {
      for (int i = 0; i < 1000; ++i)
        if (b.try_charge(7)) granted.fetch_add(7);
    });
  for (auto& t : ts) t.join();
  CHECK(granted.load() == b.used());
  CHECK(b.used() <= 1000);
  CHECK(b.used() >= 1000 - 7);
}

/**
 * Test Description
 * ----------------
 *   - ScopedCharge holds bytes while alive and returns them when it ends; a
 *     refused charge holds nothing.
 */
HRR_TEST_CASE(Unit_HRR_Limits_ScopedCharge) {
  hrr::ByteBudget b(100);
  {
    hrr::ScopedCharge a(b, 80);
    CHECK(a.ok());
    CHECK(b.used() == 80);
    hrr::ScopedCharge c(b, 80);
    CHECK_FALSE(c.ok());
    CHECK(b.used() == 80);
  }
  CHECK(b.used() == 0);
}

/**
 * Test Description
 * ----------------
 *   - The wall-clock watchdog calls its handler once after the limit, and not
 *     at all when it is destroyed before the limit or when the limit is zero.
 */
HRR_TEST_CASE(Unit_HRR_Limits_WallClockWatchdog) {
  using namespace std::chrono_literals;
  std::atomic<int> fired{0};
  {
    hrr::WallClockWatchdog w(50ms, [&] { fired.fetch_add(1); });
    for (int i = 0; i < 200 && fired.load() == 0; ++i)
      std::this_thread::sleep_for(10ms);
    CHECK(fired.load() == 1);
  }
  CHECK(fired.load() == 1);

  std::atomic<int> early{0};
  { hrr::WallClockWatchdog w(10s, [&] { early.fetch_add(1); }); }
  CHECK(early.load() == 0);

  std::atomic<int> off{0};
  {
    hrr::WallClockWatchdog w(0ms, [&] { off.fetch_add(1); });
    std::this_thread::sleep_for(50ms);
  }
  CHECK(off.load() == 0);
}

/**
 * Test Description
 * ----------------
 *   - Host allocations are charged before they are made and released when the
 *     recorded address is freed or recorded again (a second pass replays the
 *     same events), so the budget does not drift upward; a refused charge ends
 *     the whole replay (fatal_error), not just the call, and takes no charge.
 */
HRR_TEST_CASE(Unit_HRR_Limits_PlaybackContextHostBudget) {
  PlaybackContext ctx;
  ctx.host_budget.set_cap(100);

  REQUIRE(ctx.charge_host(60, "first"));
  ctx.record_alloc(0x10, nullptr, 60, AllocKind::HostMalloc);
  CHECK(ctx.host_budget.used() == 60);

  // The same recorded address replayed again: the old entry's charge goes back.
  REQUIRE(ctx.charge_host(40, "second"));
  CHECK(ctx.host_budget.used() == 100);
  ctx.record_alloc(0x10, nullptr, 40, AllocKind::HostMalloc);
  CHECK(ctx.host_budget.used() == 40);

  ctx.remove_alloc(0x10);
  CHECK(ctx.host_budget.used() == 0);

  // A device allocation is never charged to the host budget.
  ctx.record_alloc(0x20, nullptr, 500, AllocKind::Device);
  ctx.remove_alloc(0x20);
  CHECK(ctx.host_budget.used() == 0);

  // A refusal ends the replay and holds nothing.
  CHECK_FALSE(ctx.fatal_error.load());
  CHECK_FALSE(ctx.charge_host(101, "too big"));
  CHECK(ctx.fatal_error.load());
  CHECK(ctx.host_budget.used() == 0);

  PlaybackContext ctx2;
  ctx2.host_budget.set_cap(100);
  hrr::ScopedCharge held(ctx2.host_budget, 200);
  CHECK_FALSE(held.ok());
  ctx2.report_host_refusal(200, "read-back");
  CHECK(ctx2.fatal_error.load());
  CHECK(ctx2.host_budget.used() == 0);
}

/** @} */
