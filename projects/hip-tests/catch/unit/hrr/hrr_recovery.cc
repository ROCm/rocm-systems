/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Recovery
 * @{
 * @ingroup HRRTest
 * GPU-free unit tests for crash-truncation recovery in the archive reader
 * (hrr::load_archive). Each test builds a synthetic events.bin on disk and
 * asserts that the reader recovers all complete records from torn / headerless
 * tails and flags archive completeness (Archive::complete / Archive::truncated)
 * correctly.
 *
 * Ported from the former standalone playback/hrr_recovery_test.cpp so the checks
 * run as part of HrrTest "[hrr]" in CI instead of an opt-in manual build.
 */

#include <hip_test_common.hh>
#include "hrr_reader.h"
#include "hrr_api_args.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// A minimal, header-only record. event_type HRR_API_HIPDEVICESYNCHRONIZE is
// parsed by the reader without reading any payload fields, so a 32-byte record
// (header only) is valid and self-contained.
hrr_event_header make_min_record(uint64_t seq) {
  hrr_event_header h{};
  std::memset(&h, 0, sizeof(h));
  h.event_type     = static_cast<uint16_t>(HRR_API_HIPDEVICESYNCHRONIZE);
  h.sequence_id    = seq;
  h.timestamp_ns   = 1000 + seq;
  h.thread_id      = 42;
  h.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));
  return h;
}

// RAII synthetic archive directory. Writes events.bin directly (the reader does
// not require blobs/ or code_objects/ for these header-only records, but we
// create them so the layout matches a real archive).
struct TmpArchive {
  fs::path root;
  std::ofstream f;

  explicit TmpArchive(const std::string& name) {
    root = fs::temp_directory_path() / ("hrr_rec_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "blobs");
    fs::create_directories(root / "code_objects");
    f.open(root / "events.bin", std::ios::binary);
    hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
  }

  ~TmpArchive() { if (f.is_open()) f.close(); fs::remove_all(root); }

  std::string path() const { return root.string(); }

  void write_records(int n) {
    for (int i = 0; i < n; ++i) {
      hrr_event_header h = make_min_record(static_cast<uint64_t>(i));
      f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }
  }

  void write_trailer(int total) {
    hrr_eof_record eof{};
    std::memset(&eof, 0, sizeof(eof));
    eof.hdr.event_type     = HRR_EOF_MARKER;
    eof.hdr.sequence_id    = static_cast<uint64_t>(total);
    eof.hdr.payload_length = static_cast<uint16_t>(sizeof(eof));
    eof.total_events       = static_cast<uint64_t>(total);
    eof.eof_magic          = HRR_EOF_MAGIC;
    f.write(reinterpret_cast<const char*>(&eof), sizeof(eof));
  }

  // Write a raw byte run (for torn tails).
  void write_bytes(const void* p, size_t n) {
    f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
  }

  void finish() { f.flush(); f.close(); }
};

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/**
 * Test Description
 * ----------------
 *   - A complete archive (N records + clean hrr_eof_record trailer) loads with
 *     complete == true, truncated == false, and exactly N events.
 */
HIP_TEST_CASE(Unit_HRR_Recovery_CompleteArchive) {
  TmpArchive arc("complete");
  arc.write_records(5);
  arc.write_trailer(5);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 5);
  CHECK(a.complete);
  CHECK_FALSE(a.truncated);
}

/**
 * Test Description
 * ----------------
 *   - N complete records followed by a half-written next header (crash mid-write
 *     of a record header). The reader recovers the N complete records, marks the
 *     archive not-complete and truncated.
 */
HIP_TEST_CASE(Unit_HRR_Recovery_TornHeader) {
  TmpArchive arc("torn_header");
  arc.write_records(5);
  hrr_event_header partial = make_min_record(5);
  arc.write_bytes(&partial, sizeof(partial) / 2);  // only half the header
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 5);
  CHECK_FALSE(a.complete);
  CHECK(a.truncated);
}

/**
 * Test Description
 * ----------------
 *   - N complete records followed by a full header that claims a payload that is
 *     not present (crash after header, before payload). The reader recovers the
 *     N complete records, marks the archive not-complete and truncated.
 */
HIP_TEST_CASE(Unit_HRR_Recovery_TornPayload) {
  TmpArchive arc("torn_payload");
  arc.write_records(3);
  hrr_event_header h = make_min_record(3);
  h.payload_length = 200;            // claims 168 bytes of payload that follow
  arc.write_bytes(&h, sizeof(h));    // ...but write none
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 3);
  CHECK_FALSE(a.complete);
  CHECK(a.truncated);
}

/**
 * Test Description
 * ----------------
 *   - N complete records, no trailer, clean EOF exactly at a record boundary
 *     (e.g. a crash between records, after a checkpoint but before atexit). The
 *     reader recovers all N records, marks not-complete but NOT truncated (the
 *     tail is a clean record boundary, not a torn record).
 */
HIP_TEST_CASE(Unit_HRR_Recovery_NoTrailer) {
  TmpArchive arc("no_trailer");
  arc.write_records(4);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 4);
  CHECK_FALSE(a.complete);
  CHECK_FALSE(a.truncated);
}
