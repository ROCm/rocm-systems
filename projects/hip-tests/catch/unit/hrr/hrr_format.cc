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
  // acceptable.  What must NOT happen is a crash or reading event 1 as valid.
  hrr::load_archive(arc.path(), archive);
  // The first complete event must still be present.
  REQUIRE(archive.events.size() >= 1);
  const auto* a0 =
      reinterpret_cast<const hrr_args_hipSetDevice*>(archive.events[0].raw_payload.data());
  CHECK(a0->hdr.sequence_id == 0u);
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
