/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Replay Limits
 * @{
 * @ingroup HRRTest
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

#ifndef _WIN32
#include <sys/stat.h>
#endif

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

std::vector<uint8_t> padded_records(int n, uint32_t len) {
  std::vector<uint8_t> out;
  for (int i = 0; i < n; ++i) {
    hrr_event_header h = make_record(static_cast<uint64_t>(i));
    h.payload_length = len;
    const auto* hp = reinterpret_cast<const uint8_t*>(&h);
    out.insert(out.end(), hp, hp + sizeof(h));
    out.resize(out.size() + (len - sizeof(h)), 0);
  }
  return out;
}

struct TmpArchive {
  fs::path root;

  explicit TmpArchive(const std::string& name) {
    root = fs::temp_directory_path() / ("hrr_lim_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "blobs");
    fs::create_directories(root / "code_objects");
  }

  TmpArchive(const std::string& name, const std::vector<uint64_t>& seqs,
             uint64_t distinct_threads = 1)
      : TmpArchive(name) {
    std::vector<uint8_t> body;
    uint64_t i = 0;
    for (uint64_t s : seqs) {
      hrr_event_header h = make_record(s, 42 + (i++ % distinct_threads));
      const auto* hp = reinterpret_cast<const uint8_t*>(&h);
      body.insert(body.end(), hp, hp + sizeof(h));
    }
    write_events(body);
  }
  ~TmpArchive() { fs::remove_all(root); }
  std::string path() const { return root.string(); }

  void write_events(const std::vector<uint8_t>& body) {
    std::ofstream f(root / "events.bin", std::ios::binary);
    hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    f.write(reinterpret_cast<const char*>(body.data()),
            static_cast<std::streamsize>(body.size()));
  }
};

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

HRR_TEST_CASE(Unit_HRR_Limits_ContiguousSequenceReplays) {
  TmpArchive arc("contiguous", {0, 1, 2, 3});
  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  ReplayResult r = replay(a, /*max_waits=*/10);
  CHECK(r.last == hrr::TurnResult::Turn);
}

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

HRR_TEST_CASE(Unit_HRR_Limits_FatalFlagEndsWait) {
  std::atomic<uint64_t> next_seq{0};
  std::atomic<bool> fatal{true};
  uint64_t observed = 99;
  CHECK(hrr::wait_for_turn(next_seq, 5, fatal, 1000000, &observed) ==
        hrr::TurnResult::Fatal);
  CHECK(observed == 0);
}

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

HRR_TEST_CASE(Unit_HRR_Limits_ReadFileMissing) {
  std::vector<uint8_t> data;
  std::string err;
  CHECK_FALSE(hrr::read_file_capped("/nonexistent/hrr/blob", data, &err));
  CHECK(err.find("/nonexistent/hrr/blob") != std::string::npos);
  CHECK(err.find("file cap") == std::string::npos);
}

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

HRR_TEST_CASE(Unit_HRR_Limits_NoStallWhileACallIsInFlight) {
  std::atomic<uint64_t> next_seq{4};
  std::atomic<bool> fatal{false};
  std::atomic<int> busy{1};
  hrr::TurnResult result = hrr::TurnResult::Stalled;

  std::thread waiter([&] { result = hrr::wait_for_turn(next_seq, 5, fatal, /*max_waits=*/20, nullptr, &busy); });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  next_seq.store(5);  // the call returns and hands the turn on
  busy.store(0);
  waiter.join();
  CHECK(result == hrr::TurnResult::Turn);

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
  for (const char* bad : {"", "-1", "+1", " 5", "\t-1", "\n-1", "\t5", "\v1", " -1", "5x", "x", "18446744073709551616", "1e3"}) {
    v = 99;
    CHECK_FALSE(hrr::parse_u64(bad, &v));
    CHECK(v == 99);
  }
  CHECK_FALSE(hrr::parse_u64(nullptr, &v));
}


struct EventCapGuard {
  uint64_t saved = hrr::max_events();
  ~EventCapGuard() { hrr::set_max_events(saved); }
};

HRR_TEST_CASE(Unit_HRR_Limits_ThreadCap) {
  std::string why;
  CHECK(hrr::check_thread_cap(256, 256, &why));
  CHECK(why.empty());
  CHECK_FALSE(hrr::check_thread_cap(257, 256, &why));
  CHECK(why.find("257") != std::string::npos);
  CHECK(why.find("256") != std::string::npos);
  CHECK_FALSE(hrr::check_thread_cap(1, 0, nullptr));
}

HRR_TEST_CASE(Unit_HRR_Limits_PlaybackRefusesTooManyThreads) {
  TmpArchive arc("threads", {0, 1, 2, 3, 4, 5}, /*distinct_threads=*/6);
  auto [ret, out] = hrr_playback_merged(arc.root, "--multi-thread --max-threads 4");
  CHECK(ret != 0);
  CHECK(out.find("6 threads") != std::string::npos);
  CHECK(out.find("--max-threads") != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_Limits_PlaybackRefusesWallSecondsAboveMaximum) {
  TmpArchive arc("wallmax", {0, 1});
  auto [ret, out] = hrr_playback_merged(arc.root, "--max-wall-seconds 31536001");
  CHECK(ret != 0);
  CHECK(out.find("31536000") != std::string::npos);
  CHECK(out.find("--max-wall-seconds") != std::string::npos);
}

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

HRR_TEST_CASE(Unit_HRR_Limits_PlaybackContextHostBudget) {
  PlaybackContext ctx;
  ctx.host_budget.set_cap(100);

  REQUIRE(ctx.charge_host(60, "first"));
  ctx.record_alloc(0x10, nullptr, 60, AllocKind::HostMalloc);
  CHECK(ctx.host_budget.used() == 60);

  // record_alloc over a live host entry does not give its charge back: the
  // buffer is still allocated, so the charge stays with it. The replaying
  // handlers free the replaced entry first (remove_alloc is the budget side of
  // that), which is what returns the charge.
  REQUIRE(ctx.charge_host(40, "second"));
  CHECK(ctx.host_budget.used() == 100);
  ctx.record_alloc(0x10, nullptr, 40, AllocKind::HostMalloc);
  CHECK(ctx.host_budget.used() == 100);
  ctx.remove_alloc(0x10);
  CHECK(ctx.host_budget.used() == 60);
  ctx.host_budget.release(60);  // the replaced 60-byte buffer's own release
  CHECK(ctx.host_budget.used() == 0);

  REQUIRE(ctx.charge_host(60, "third"));
  ctx.record_alloc(0x10, nullptr, 60, AllocKind::HostMalloc);
  ctx.remove_alloc(0x10);
  CHECK(ctx.host_budget.used() == 0);
  REQUIRE(ctx.charge_host(40, "fourth"));
  ctx.record_alloc(0x10, nullptr, 40, AllocKind::HostMalloc);
  CHECK(ctx.host_budget.used() == 40);

  ctx.remove_alloc(0x10);
  CHECK(ctx.host_budget.used() == 0);

  ctx.record_alloc(0x20, nullptr, 500, AllocKind::Device);
  ctx.remove_alloc(0x20);
  CHECK(ctx.host_budget.used() == 0);

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

HRR_TEST_CASE(Unit_HRR_Limits_FindDuplicateSeq) {
  uint64_t dup = 0;
  CHECK_FALSE(hrr::find_duplicate_seq({}, &dup));
  CHECK_FALSE(hrr::find_duplicate_seq({0, 1, 2, 5}, &dup));
  CHECK(hrr::find_duplicate_seq({3, 1, 4, 1, 5}, &dup));
  CHECK(dup == 1);
}

HRR_TEST_CASE(Unit_HRR_Limits_FindSeqGap) {
  uint64_t missing = 0;
  CHECK_FALSE(hrr::find_seq_gap({}, &missing));
  CHECK_FALSE(hrr::find_seq_gap({7}, &missing));
  CHECK_FALSE(hrr::find_seq_gap({0, 1, 2, 3}, &missing));
  CHECK_FALSE(hrr::find_seq_gap({12, 10, 11}, &missing));
  CHECK_FALSE(hrr::find_seq_gap({1, 1, 2}, &missing));
  CHECK(hrr::find_seq_gap({0, 1, 2, 5}, &missing));
  CHECK(missing == 3);
  CHECK(hrr::find_seq_gap({13, 11, 10}, &missing));
  CHECK(missing == 12);
}


HRR_TEST_CASE(Unit_HRR_Limits_LoadByteBudget) {
  constexpr uint32_t kLen = 100 * 1024;
  constexpr int kRecords = 40;  // 4 MB of records
  TmpArchive arc("loadbytes");
  arc.write_events(padded_records(kRecords, kLen));

  hrr::ByteBudget small(1u << 20);  // 1 MiB
  hrr::Archive refused;
  CHECK_FALSE(hrr::load_archive(arc.path(), refused, &small));
  CHECK(refused.events.size() < 11);  // 1 MiB holds ten such records
  CHECK(small.used() == 0);

  hrr::ByteBudget large(64u << 20);
  hrr::Archive loaded;
  CHECK(hrr::load_archive(arc.path(), loaded, &large));
  CHECK(loaded.events.size() == kRecords);
  CHECK(large.used() >= uint64_t(kRecords) * kLen);
}

HRR_TEST_CASE(Unit_HRR_Limits_RecordChargedBeforeAllocation) {
  const fs::path path = fs::temp_directory_path() / "hrr_lim_bigclaim.bin";
  {
    std::ofstream f(path, std::ios::binary);
    hrr_event_header h = make_record(0);
    h.payload_length = 60u << 20;  // the record claims 60 MiB; the file has none
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
  }
  hrr::ByteBudget budget(1u << 20);
  std::vector<uint8_t> out;
  FILE* f = fopen(path.string().c_str(), "rb");
  REQUIRE(f != nullptr);
  CHECK(hrr::read_raw_record(f, out, &budget) == hrr::RecordStatus::OverBudget);
  CHECK(out.size() == sizeof(hrr_event_header));
  CHECK(out.capacity() < (1u << 20));
  CHECK(budget.used() == 0);
  fclose(f);

  fs::remove(path);
}

HRR_TEST_CASE(Unit_HRR_Limits_LoadChargesParsedKernelExpansion) {
  constexpr uint16_t kArgs = 20000;
  const auto body = make_launch_with_snapshot(0, {}, 0, kArgs);
  TmpArchive arc("kernelexp");
  arc.write_events(body);
  const fs::path root = arc.root;
  const uint64_t raw = body.size();
  const uint64_t expanded = uint64_t(kArgs) * sizeof(hrr::KernelArg);
  REQUIRE(expanded > 4 * raw);  // otherwise the raw-bytes cap below proves nothing

  hrr::ByteBudget raw_only(raw + 64 * sizeof(hrr::Event) + 4096);
  hrr::Archive refused;
  CHECK_FALSE(hrr::load_archive(root.string(), refused, &raw_only));
  CHECK(raw_only.used() == 0);

  hrr::ByteBudget roomy_budget(raw + expanded + (1u << 20));
  hrr::Archive loaded;
  CHECK(hrr::load_archive(root.string(), loaded, &roomy_budget));
  REQUIRE(loaded.events.size() == 1);
  REQUIRE(loaded.events[0].kernel_launch != nullptr);
  CHECK(loaded.events[0].kernel_launch->args.size() == kArgs);
  CHECK(roomy_budget.used() >= raw + expanded);
}

HRR_TEST_CASE(Unit_HRR_Limits_LoadChargesPerAllocationOverhead) {
  constexpr uint16_t kArgs = 20000;
  const auto body = make_launch_with_snapshot(0, {}, 0, kArgs, /*pointer_args=*/true);
  TmpArchive arc("argalloc");
  arc.write_events(body);
  const uint64_t base = body.size() + 64 * sizeof(hrr::Event) + 4096;
  const uint64_t by_bytes = uint64_t(kArgs) * (sizeof(hrr::KernelArg) + 1 + 2);
  const uint64_t by_allocations = uint64_t(kArgs) * (sizeof(hrr::KernelArg) + 32 + 32);

  hrr::ByteBudget bytes_only(base + by_bytes + kArgs * 16);
  hrr::Archive refused;
  CHECK_FALSE(hrr::load_archive(arc.path(), refused, &bytes_only));
  CHECK(bytes_only.used() == 0);

  hrr::ByteBudget roomy(base + by_allocations + (1u << 20));
  hrr::Archive loaded;
  REQUIRE(hrr::load_archive(arc.path(), loaded, &roomy));
  REQUIRE(loaded.events.size() == 1);
  CHECK(loaded.events[0].kernel_launch->args.size() == kArgs);
  CHECK(roomy.used() >= body.size() + by_allocations);
}

HRR_TEST_CASE(Unit_HRR_Limits_PlaybackInfoRefusesOverByteBudget) {
  TmpArchive arc("info_bytes");
  arc.write_events(padded_records(40, 100 * 1024));
  auto [ret, out] = hrr_playback_merged(arc.root, "--info --max-host-bytes 1048576");
  CHECK(ret != 0);
  CHECK(out.find("--max-host-bytes") != std::string::npos);

  auto [ok_ret, ok_out] = hrr_playback_merged(arc.root, "--info --max-host-bytes 1073741824");
  CHECK(ok_ret == 0);
}

HRR_TEST_CASE(Unit_HRR_Limits_LoadChargesEventVectorAndSort) {
  constexpr uint64_t kSlot = sizeof(hrr::Event);
  {
    constexpr int kN = 5000;
    std::vector<uint64_t> seqs(kN);
    for (int i = 0; i < kN; ++i) seqs[i] = static_cast<uint64_t>(i);
    TmpArchive arc("vecgrow", seqs);
    hrr::ByteBudget one_slot_each(uint64_t(kN) * (sizeof(hrr_event_header) + kSlot) +
                                  256 * kSlot);
    hrr::Archive refused;
    CHECK_FALSE(hrr::load_archive(arc.path(), refused, &one_slot_each));
    CHECK(one_slot_each.used() == 0);

    hrr::ByteBudget roomy(uint64_t(kN) * sizeof(hrr_event_header) + 3 * 8192 * kSlot);
    hrr::Archive loaded;
    REQUIRE(hrr::load_archive(arc.path(), loaded, &roomy));
    CHECK(loaded.events.size() == kN);
    CHECK(roomy.used() >= uint64_t(loaded.events.capacity()) * kSlot);
  }
  {
    constexpr int kN = 4097;
    std::vector<uint64_t> seqs(kN);
    for (int i = 0; i < kN; ++i) seqs[i] = static_cast<uint64_t>(i);
    TmpArchive arc("vecpeak", seqs);
    hrr::ByteBudget one_buffer(uint64_t(kN) * sizeof(hrr_event_header) + 11000 * kSlot);
    hrr::Archive refused;
    CHECK_FALSE(hrr::load_archive(arc.path(), refused, &one_buffer));
    CHECK(one_buffer.used() == 0);

    hrr::ByteBudget both(uint64_t(kN) * sizeof(hrr_event_header) + 13000 * kSlot);
    hrr::Archive loaded;
    REQUIRE(hrr::load_archive(arc.path(), loaded, &both));
    CHECK(loaded.events.size() == kN);
    CHECK(both.used() >= uint64_t(loaded.events.capacity()) * kSlot);
  }
  {
    constexpr int kN = 8192;
    std::vector<uint64_t> seqs(kN);
    for (int i = 0; i < kN; ++i) seqs[i] = static_cast<uint64_t>(kN - i);  // out of order
    TmpArchive arc("vecsort", seqs);
    // Holds the growth to 8192 slots (4096 old + 8192 new) and the records, not
    // the sort scratch of 4097 slots on top of 8192.
    hrr::ByteBudget no_scratch(uint64_t(kN) * sizeof(hrr_event_header) + 12288 * kSlot);
    hrr::Archive refused;
    CHECK_FALSE(hrr::load_archive(arc.path(), refused, &no_scratch));
    CHECK(no_scratch.used() == 0);

    hrr::ByteBudget with_scratch(uint64_t(kN) * sizeof(hrr_event_header) + 13000 * kSlot);
    hrr::Archive loaded;
    CHECK(hrr::load_archive(arc.path(), loaded, &with_scratch));
  }
}

HRR_TEST_CASE(Unit_HRR_Limits_LoadChargesBlobAndCodeObjectMaps) {
  for (const char* kind : {"blobs", "code_objects"}) {
    const char* ext = std::string(kind) == "blobs" ? ".blob" : ".hsaco";
    TmpArchive arc(std::string("maps_") + kind, {0});
    for (int i = 0; i < 2000; ++i)
      std::ofstream(arc.root / kind / (std::string("f") + std::to_string(i) + ext)).put(0);

    hrr::ByteBudget tight(128 * 1024);
    hrr::Archive refused;
    INFO(kind);
    CHECK_FALSE(hrr::load_archive(arc.path(), refused, &tight));
    CHECK(tight.used() == 0);

    hrr::ByteBudget roomy(64u << 20);
    hrr::Archive loaded;
    REQUIRE(hrr::load_archive(arc.path(), loaded, &roomy));
    CHECK(loaded.blob_count + loaded.code_object_count == 2000);
  }
}

#ifndef _WIN32
/**
 * Test Description
 * ----------------
 *   - The wall-clock limit covers the archive load: hrr-playback --info on a
 *     pid directory whose events.bin is a FIFO nobody writes to blocks in the
 *     open, and --max-wall-seconds 1 ends it with exit status 124 instead of
 *     leaving it hanging.
 */
HRR_TEST_CASE(Unit_HRR_Limits_WallClockEndsBlockedLoad) {
  TmpArchive arc("fifo_load");
  REQUIRE(mkfifo((arc.root / "events.bin").c_str(), 0600) == 0);
  auto [ret, out] = hrr_playback_watchdog(arc.root, /*timeout_seconds=*/30,
                                          "--info --max-wall-seconds 1");
  INFO(out);
  CHECK(ret == 124);
  CHECK(out.find("--max-wall-seconds") != std::string::npos);
}
#endif  // !_WIN32

/** @} */
