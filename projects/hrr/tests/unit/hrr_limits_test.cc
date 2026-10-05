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
#include "hrr/hrr_api_args.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

hrr_event_header make_record(uint64_t seq) {
  hrr_event_header h{};
  std::memset(&h, 0, sizeof(h));
  h.event_type     = static_cast<uint16_t>(HRR_API_HIPDEVICESYNCHRONIZE);
  h.sequence_id    = seq;
  h.timestamp_ns   = 1000 + seq;
  h.thread_id      = 42;
  h.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));
  return h;
}

// RAII synthetic archive directory with a caller-chosen sequence list.
struct TmpArchive {
  fs::path root;

  TmpArchive(const std::string& name, const std::vector<uint64_t>& seqs) {
    root = fs::temp_directory_path() / ("hrr_lim_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "blobs");
    fs::create_directories(root / "code_objects");
    std::ofstream f(root / "events.bin", std::ios::binary);
    hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    for (uint64_t s : seqs) {
      hrr_event_header h = make_record(s);
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

/** @} */
