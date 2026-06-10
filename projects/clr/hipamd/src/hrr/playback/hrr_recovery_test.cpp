/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// hrr_recovery_test — GPU-free unit test for crash-truncation recovery in the
// archive reader (load_archive). Builds synthetic events.bin files on disk and
// asserts the reader recovers complete records from torn/headerless tails and
// flags archive completeness correctly.
//
// Build (not part of the default bake; opt in explicitly):
//   ninja -C <build> hrr-recovery-test
//   ./hrr-recovery-test
//
// Or standalone (no HIP needed — the reader does not depend on HIP):
//   g++ -std=c++17 -I.. -I. -DHRR_API_ARGS_IMPLEMENTATION \
//       hrr_recovery_test.cpp hrr_reader.cpp -o hrr-recovery-test

#include "hrr_reader.h"
#include "hrr_api_args.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);      \
      g_failures++;                                                          \
    } else {                                                                 \
      printf("ok: %s\n", (msg));                                             \
    }                                                                        \
  } while (0)

// A minimal, header-only record. event_type HRR_API_HIPDEVICESYNCHRONIZE is
// parsed by the reader without reading any payload fields, so a 32-byte record
// (header only) is valid and self-contained.
static void make_min_record(hrr_event_header& h, uint64_t seq) {
  memset(&h, 0, sizeof(h));
  h.event_type     = HRR_API_HIPDEVICESYNCHRONIZE;
  h.sequence_id    = seq;
  h.timestamp_ns   = 1000 + seq;
  h.thread_id      = 42;
  h.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));
}

static void write_file_header(FILE* f) {
  hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
  fwrite(&fh, sizeof(fh), 1, f);
}

static std::string make_dir(const char* name) {
  std::string dir = (fs::temp_directory_path() / name).string();
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

// Whole archive with N records and a clean trailer.
static std::string build_complete(int n) {
  std::string dir = make_dir("hrr_rec_complete");
  FILE* f = fopen((dir + "/events.bin").c_str(), "wb");
  write_file_header(f);
  for (int i = 0; i < n; i++) { hrr_event_header h; make_min_record(h, i); fwrite(&h, sizeof(h), 1, f); }
  hrr_eof_record eof;
  memset(&eof, 0, sizeof(eof));
  eof.hdr.event_type     = HRR_EOF_MARKER;
  eof.hdr.sequence_id    = n;
  eof.hdr.payload_length = static_cast<uint16_t>(sizeof(eof));
  eof.total_events       = static_cast<uint64_t>(n);
  eof.eof_magic          = HRR_EOF_MAGIC;
  fwrite(&eof, sizeof(eof), 1, f);
  fclose(f);
  return dir;
}

// N complete records followed by a partial (torn) header — simulates a crash
// mid-write of the next record.
static std::string build_torn_header(int n) {
  std::string dir = make_dir("hrr_rec_torn_header");
  FILE* f = fopen((dir + "/events.bin").c_str(), "wb");
  write_file_header(f);
  for (int i = 0; i < n; i++) { hrr_event_header h; make_min_record(h, i); fwrite(&h, sizeof(h), 1, f); }
  hrr_event_header partial; make_min_record(partial, n);
  fwrite(&partial, 1, sizeof(partial) / 2, f);  // only half the header
  fclose(f);
  return dir;
}

// N complete records followed by a full header that claims a payload that is
// not present — simulates a crash after the header but before the payload.
static std::string build_torn_payload(int n) {
  std::string dir = make_dir("hrr_rec_torn_payload");
  FILE* f = fopen((dir + "/events.bin").c_str(), "wb");
  write_file_header(f);
  for (int i = 0; i < n; i++) { hrr_event_header h; make_min_record(h, i); fwrite(&h, sizeof(h), 1, f); }
  hrr_event_header h; make_min_record(h, n);
  h.payload_length = 200;          // claims 168 bytes of payload that follow
  fwrite(&h, sizeof(h), 1, f);     // ...but write none
  fclose(f);
  return dir;
}

// N complete records, no trailer, clean EOF at a record boundary — crash that
// happened exactly between records (or before atexit, after a checkpoint).
static std::string build_no_trailer(int n) {
  std::string dir = make_dir("hrr_rec_no_trailer");
  FILE* f = fopen((dir + "/events.bin").c_str(), "wb");
  write_file_header(f);
  for (int i = 0; i < n; i++) { hrr_event_header h; make_min_record(h, i); fwrite(&h, sizeof(h), 1, f); }
  fclose(f);
  return dir;
}

int main() {
  {
    hrr::Archive a;
    bool ok = hrr::load_archive(build_complete(5), a);
    CHECK(ok, "complete archive loads");
    CHECK(a.events.size() == 5, "complete: 5 events");
    CHECK(a.complete, "complete: complete flag set");
    CHECK(!a.truncated, "complete: not truncated");
  }
  {
    hrr::Archive a;
    bool ok = hrr::load_archive(build_torn_header(5), a);
    CHECK(ok, "torn-header archive still loads");
    CHECK(a.events.size() == 5, "torn-header: 5 complete events recovered");
    CHECK(!a.complete, "torn-header: not marked complete");
    CHECK(a.truncated, "torn-header: truncated flag set");
  }
  {
    hrr::Archive a;
    bool ok = hrr::load_archive(build_torn_payload(3), a);
    CHECK(ok, "torn-payload archive still loads");
    CHECK(a.events.size() == 3, "torn-payload: 3 complete events recovered");
    CHECK(!a.complete, "torn-payload: not marked complete");
    CHECK(a.truncated, "torn-payload: truncated flag set");
  }
  {
    hrr::Archive a;
    bool ok = hrr::load_archive(build_no_trailer(4), a);
    CHECK(ok, "no-trailer archive loads");
    CHECK(a.events.size() == 4, "no-trailer: 4 events recovered");
    CHECK(!a.complete, "no-trailer: not marked complete");
    CHECK(!a.truncated, "no-trailer: not truncated (clean record boundary)");
  }

  if (g_failures) { fprintf(stderr, "\n%d check(s) FAILED\n", g_failures); return 1; }
  printf("\nAll recovery checks passed.\n");
  return 0;
}
