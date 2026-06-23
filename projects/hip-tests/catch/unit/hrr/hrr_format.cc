/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Format
 * @{
 * @ingroup HRRTest
 * Unit tests for the HRR binary archive format:
 *   - hrr_file_header magic and version round-trips correctly
 *   - hrr_event_header fields (sequence_id, event_type, payload_length) survive
 *     a write-then-read cycle
 *   - hrr::load_archive() rejects truncated / bad-magic files
 *   - hrr::hash_hex() produces correctly formatted 32-char hex strings
 */

#include <hip_test_common.hh>
#include "hrr_reader.h"
#include "hrr_api_args.h"
#include "hip_playback.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Minimal archive on disk: events.bin with file header + N events, plus the
// required empty blobs/ and code_objects/ subdirectories.
struct TmpArchive {
  fs::path root;

  explicit TmpArchive(const std::string& name) {
    root = fs::temp_directory_path() / ("hrr_test_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "blobs");
    fs::create_directories(root / "code_objects");
  }

  ~TmpArchive() { fs::remove_all(root); }

  std::string path() const { return root.string(); }

  // Write events.bin with a valid file header followed by the given raw bytes.
  void write_events(const std::vector<uint8_t>& body) {
    std::ofstream f(root / "events.bin", std::ios::binary);
    hrr_file_header fh{};
    fh.magic   = HRR_MAGIC;
    fh.version = HRR_VERSION;
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    f.write(reinterpret_cast<const char*>(body.data()), body.size());
  }

  // Write a raw events.bin (no automatic header — for negative tests).
  void write_raw(const std::vector<uint8_t>& bytes) {
    std::ofstream f(root / "events.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
};

// Serialise an hrr_args_hipSetDevice struct (the simplest non-void API:
// hdr(32) + ret(4) + deviceId(4) = 40 bytes total).
static std::vector<uint8_t> make_set_device_event(uint64_t seq, int32_t device_id,
                                                   int32_t ret_val = 0) {
  hrr_args_hipSetDevice ev{};
  ev.hdr.event_type     = static_cast<uint16_t>(HRR_API_HIPSETDEVICE);
  ev.hdr.sequence_id    = seq;
  ev.hdr.timestamp_ns   = 0;
  ev.hdr.thread_id      = 1;
  ev.hdr.payload_length = static_cast<uint16_t>(sizeof(ev));
  ev.ret      = ret_val;
  ev.deviceId = device_id;

  std::vector<uint8_t> bytes(sizeof(ev));
  std::memcpy(bytes.data(), &ev, sizeof(ev));
  return bytes;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/**
 * Test Description
 * ----------------
 *   - Write a 3-event archive (hipSetDevice calls with distinct sequence IDs
 *     and device IDs), load with hrr::load_archive(), verify that:
 *       * version == HRR_VERSION
 *       * event_count == 3
 *       * each event's sequence_id, event_type, and deviceId field match
 */
HIP_TEST_CASE(Unit_HRR_Format_Roundtrip) {
  TmpArchive arc("roundtrip");

  // Build body: 3 hipSetDevice events with seq IDs 0, 1, 2 and device IDs 7, 3, 0
  std::vector<uint8_t> body;
  for (int i = 0; i < 3; ++i) {
    auto ev = make_set_device_event(/*seq=*/i, /*device=*/7 - i * 2);
    body.insert(body.end(), ev.begin(), ev.end());
  }
  arc.write_events(body);

  hrr::Archive archive;
  REQUIRE(hrr::load_archive(arc.path(), archive));

  REQUIRE(archive.version == HRR_VERSION);
  REQUIRE(archive.events.size() == 3);

  const int expected_devices[] = {7, 5, 3};
  for (int i = 0; i < 3; ++i) {
    const hrr::Event& ev = archive.events[i];
    const auto* a = reinterpret_cast<const hrr_args_hipSetDevice*>(ev.raw_payload.data());

    CHECK(a->hdr.event_type  == static_cast<uint16_t>(HRR_API_HIPSETDEVICE));
    CHECK(a->hdr.sequence_id == static_cast<uint64_t>(i));
    CHECK(a->hdr.thread_id   == 1u);
    CHECK(a->deviceId        == expected_devices[i]);
    CHECK(a->ret             == 0);
  }
}

/**
 * Test Description
 * ----------------
 *   - A file with a bad magic number must be rejected by load_archive().
 */
HIP_TEST_CASE(Unit_HRR_Format_BadMagic) {
  TmpArchive arc("bad_magic");

  hrr_file_header fh{};
  fh.magic   = 0xDEADBEEFu;  // wrong
  fh.version = HRR_VERSION;
  std::vector<uint8_t> raw(sizeof(fh));
  std::memcpy(raw.data(), &fh, sizeof(fh));
  arc.write_raw(raw);

  hrr::Archive archive;
  REQUIRE_FALSE(hrr::load_archive(arc.path(), archive));
}

/**
 * Test Description
 * ----------------
 *   - A truncated file (header present but first event cut short) must not
 *     crash and must return an empty event list.
 */
HIP_TEST_CASE(Unit_HRR_Format_TruncatedEvent) {
  TmpArchive arc("truncated");

  // Write one full event then only 10 bytes of a second.
  auto ev = make_set_device_event(0, 0);
  std::vector<uint8_t> body(ev.begin(), ev.end());
  body.insert(body.end(), 10, 0xAB);  // truncated second event
  arc.write_events(body);

  hrr::Archive archive;
  // load_archive may return true (with partial data) or false — either is
  // acceptable.  What must NOT happen is a crash or reading the truncated
  // second event as a valid event.
  hrr::load_archive(arc.path(), archive);
  // The first complete event must be present.
  REQUIRE(archive.events.size() >= 1);
  const auto* a0 =
      reinterpret_cast<const hrr_args_hipSetDevice*>(archive.events[0].raw_payload.data());
  CHECK(a0->hdr.sequence_id == 0u);
  // The truncated second event must NOT appear as a parsed event.
  CHECK(archive.events.size() == 1);
}

/**
 * Test Description
 * ----------------
 *   - hrr::hash_hex() must produce a 32-character lowercase hex string.
 *   - Known values: hash_hex(0,0) == "0000...0" (32 zeros),
 *     hash_hex(0x1, 0x2) == 32-char string with expected nibbles.
 */
HIP_TEST_CASE(Unit_HRR_Format_HashHex) {
  std::string z = hrr::hash_hex(0, 0);
  REQUIRE(z.size() == 32);
  REQUIRE(z == std::string(32, '0'));

  std::string h = hrr::hash_hex(0x000000000000000Full, 0xf000000000000000ull);
  REQUIRE(h.size() == 32);
  // lo printed first (little-endian convention in hash_hex): lo=0x0..0f, hi=0xf0..0
  REQUIRE(h.substr(0, 16) == "000000000000000f");
  REQUIRE(h.substr(16, 16) == "f000000000000000");
}

/**
 * Test Description
 * ----------------
 *   - event_type_name() returns a non-null, non-empty string for a known
 *     event type and "UNKNOWN" for an out-of-range type.
 */
HIP_TEST_CASE(Unit_HRR_Format_EventTypeName) {
  const char* name = hrr::event_type_name(HRR_API_HIPSETDEVICE);
  REQUIRE(name != nullptr);
  REQUIRE(std::string(name).find("hipSetDevice") != std::string::npos);

  const char* unk = hrr::event_type_name(0xFFFFu);
  REQUIRE(std::string(unk) == "UNKNOWN");
}

// ---------------------------------------------------------------------------
// CPU-only translate_ptr unit tests (no GPU required)
//
// These tests construct a PlaybackContext entirely on the CPU, populate
// alloc_map with fake entries, and verify translate_ptr behaviour.
// No HIP API calls are made.
// ---------------------------------------------------------------------------

/**
 * Test Description
 * ----------------
 *   - record_alloc maps a base address exactly.
 *   - translate_ptr(base) returns the live pointer.
 */
HIP_TEST_CASE(Unit_HRR_TranslatePtr_ExactMatch) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  CHECK(ctx.translate_ptr(0xBEEF0000ULL) == fake_live);
}

/**
 * Test Description
 * ----------------
 *   - An address that falls within a recorded allocation (base + offset) is
 *     returned as (live_ptr + offset).
 */
HIP_TEST_CASE(Unit_HRR_TranslatePtr_SubAlloc_ReturnsOffset) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  void* result = ctx.translate_ptr(0xBEEF0100ULL);  // +256 bytes into alloc
  CHECK(result == static_cast<char*>(fake_live) + 256);
}

/**
 * Test Description
 * ----------------
 *   - An address exactly one byte past the end of a recorded allocation
 *     returns nullptr (out of range).
 */
HIP_TEST_CASE(Unit_HRR_TranslatePtr_OutOfRange_ReturnsNull) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  CHECK(ctx.translate_ptr(0xBEEF0400ULL) == nullptr);  // base + 1024 = one past end
}

/**
 * Test Description
 * ----------------
 *   - translate_ptr(0) always returns nullptr regardless of alloc_map state.
 */
HIP_TEST_CASE(Unit_HRR_TranslatePtr_Zero_ReturnsNull) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  CHECK(ctx.translate_ptr(0) == nullptr);
}

// ---------------------------------------------------------------------------
// Archive reader SIZE_OK guard tests (CPU-only, no GPU required)
//
// These tests verify that load_archive() does not crash or produce a decoded
// event when an event's payload_length is too small for its declared type.
// The SIZE_OK macro in hrr_reader.cpp guards every typed cast; these tests
// confirm it fires correctly for known event types.
// ---------------------------------------------------------------------------

/**
 * Test Description
 * ----------------
 *   - A hipMalloc event with payload_length == sizeof(hrr_event_header) (header
 *     only, no fields) must not be decoded as a malloc event.  load_archive()
 *     should return true (archive is structurally valid) but the event must not
 *     have malloc_ev.ptr_handle set (SIZE_OK guard skips the typed cast).
 *
 * Policy decision: the reader RETAINS undersized known events with safe-default
 * typed fields (ptr_handle == 0).  The dispatch layer in hrr_playback.cpp
 * enforces a complementary size guard: it skips dispatch for any event whose
 * raw_payload is smaller than sizeof(hrr_event_header), preventing the handler
 * from doing an OOB reinterpret_cast.  Both layers together make the malformed
 * event safe: it is loaded without crash, not decoded, and not dispatched.
 */
HIP_TEST_CASE(Unit_HRR_Format_SmallPayloadForType) {
  TmpArchive arc("small_payload");

  // Build a hipMalloc event whose payload_length is exactly sizeof(hrr_event_header)
  // — the header is present but the malloc fields (ptr, size, ret) are absent.
  hrr_event_header hdr{};
  hdr.event_type     = static_cast<uint16_t>(HRR_API_HIPMALLOC);
  hdr.sequence_id    = 0;
  hdr.timestamp_ns   = 0;
  hdr.thread_id      = 1;
  hdr.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));  // header only
  memset(hdr.reserved, 0, sizeof(hdr.reserved));

  std::vector<uint8_t> body(sizeof(hdr));
  std::memcpy(body.data(), &hdr, sizeof(hdr));
  arc.write_events(body);

  hrr::Archive archive;
  // load_archive must not crash and should return true (file is structurally valid).
  bool ok = hrr::load_archive(arc.path(), archive);
  REQUIRE(ok);

  // Policy: the event is RETAINED in the archive (raw_payload intact for
  // tracing/debugging), but the SIZE_OK guard prevents typed decode.
  REQUIRE(archive.events.size() == 1);
  // Typed fields stay at safe defaults — no OOB cast occurred in the reader.
  CHECK(archive.events[0].malloc_ev.ptr_handle == 0u);
  CHECK(archive.events[0].malloc_ev.size == 0u);
  // raw_payload must be exactly header-sized (confirms reader did not over-read).
  CHECK(archive.events[0].raw_payload.size() == sizeof(hrr_event_header));
}

/**
 * Test Description
 * ----------------
 *   - An event with an unknown event_type and a valid payload_length must not
 *     crash load_archive() and must be silently accepted into the event list
 *     (raw_payload retained for future replay use).
 */
HIP_TEST_CASE(Unit_HRR_Format_UnknownEventType) {
  TmpArchive arc("unknown_type");

  hrr_event_header hdr{};
  hdr.event_type     = 0xFFFFu;  // unknown
  hdr.sequence_id    = 0;
  hdr.timestamp_ns   = 0;
  hdr.thread_id      = 1;
  hdr.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));
  memset(hdr.reserved, 0, sizeof(hdr.reserved));

  std::vector<uint8_t> body(sizeof(hdr));
  std::memcpy(body.data(), &hdr, sizeof(hdr));
  arc.write_events(body);

  hrr::Archive archive;
  REQUIRE(hrr::load_archive(arc.path(), archive));
  // Unknown types fall through to the default: case — raw_payload is kept,
  // no typed fields are populated, and the archive is not rejected.
  REQUIRE(archive.events.size() == 1);
  CHECK(archive.events[0].header().event_type == 0xFFFFu);
  CHECK(archive.events[0].malloc_ev.ptr_handle == 0u);
}
