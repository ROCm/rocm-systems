/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Format
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"
#include "hip_playback.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;


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

  void write_events(const std::vector<uint8_t>& body) {
    std::ofstream f(root / "events.bin", std::ios::binary);
    hrr_file_header fh{};
    fh.magic   = HRR_MAGIC;
    fh.version = HRR_VERSION;
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    f.write(reinterpret_cast<const char*>(body.data()), body.size());
  }

  void write_raw(const std::vector<uint8_t>& bytes) {
    std::ofstream f(root / "events.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
};

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


HRR_TEST_CASE(Unit_HRR_Format_Roundtrip) {
  TmpArchive arc("roundtrip");

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

HRR_TEST_CASE(Unit_HRR_Format_BadMagic) {
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

HRR_TEST_CASE(Unit_HRR_Format_TruncatedEvent) {
  TmpArchive arc("truncated");

  auto ev = make_set_device_event(0, 0);
  std::vector<uint8_t> body(ev.begin(), ev.end());
  body.insert(body.end(), 10, 0xAB);  // truncated second event
  arc.write_events(body);

  hrr::Archive archive;
  hrr::load_archive(arc.path(), archive);
  REQUIRE(archive.events.size() >= 1);
  const auto* a0 =
      reinterpret_cast<const hrr_args_hipSetDevice*>(archive.events[0].raw_payload.data());
  CHECK(a0->hdr.sequence_id == 0u);
  CHECK(archive.events.size() == 1);
}

HRR_TEST_CASE(Unit_HRR_Format_HashHex) {
  std::string z = hrr::hash_hex(0, 0);
  REQUIRE(z.size() == 32);
  REQUIRE(z == std::string(32, '0'));

  std::string h = hrr::hash_hex(0x000000000000000Full, 0xf000000000000000ull);
  REQUIRE(h.size() == 32);
  // hash_hex prints lo before hi.
  REQUIRE(h.substr(0, 16) == "000000000000000f");
  REQUIRE(h.substr(16, 16) == "f000000000000000");
}

HRR_TEST_CASE(Unit_HRR_Format_EventTypeName) {
  const char* name = hrr::event_type_name(HRR_API_HIPSETDEVICE);
  REQUIRE(name != nullptr);
  REQUIRE(std::string(name).find("hipSetDevice") != std::string::npos);

  const char* unk = hrr::event_type_name(0xFFFFu);
  REQUIRE(std::string(unk) == "UNKNOWN");
}


HRR_TEST_CASE(Unit_HRR_TranslatePtr_ExactMatch) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  CHECK(ctx.translate_ptr(0xBEEF0000ULL) == fake_live);
}

HRR_TEST_CASE(Unit_HRR_TranslatePtr_SubAlloc_ReturnsOffset) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  void* result = ctx.translate_ptr(0xBEEF0100ULL);  // +256 bytes into alloc
  // Use uintptr_t arithmetic and compare as void*. Catch2 formats char* operands with strlen(),
  // which crashes on these fabricated addresses.
  void* expected =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(fake_live) + 256);
  CHECK(result == expected);
}

HRR_TEST_CASE(Unit_HRR_TranslatePtr_OutOfRange_ReturnsNull) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  CHECK(ctx.translate_ptr(0xBEEF0400ULL) == nullptr);  // base + 1024 = one past end
}

HRR_TEST_CASE(Unit_HRR_TranslatePtr_Zero_ReturnsNull) {
  PlaybackContext ctx;
  void* fake_live = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD0000u));
  ctx.record_alloc(0xBEEF0000ULL, fake_live, 1024);
  CHECK(ctx.translate_ptr(0) == nullptr);
}

HRR_TEST_CASE(Unit_HRR_TranslatePtr_TightestEnclosing) {
  PlaybackContext ctx;
  void* liveOuter = reinterpret_cast<void*>(static_cast<uintptr_t>(0xA0000000u));
  void* liveInner = reinterpret_cast<void*>(static_cast<uintptr_t>(0xB0000000u));

  ctx.record_alloc(0x10000ULL, liveOuter, 0x2000);
  ctx.record_alloc(0x10800ULL, liveInner, 0x1000);

  void* result = ctx.translate_ptr(0x11000ULL);
  void* expectedInner =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(liveInner) + 0x800);
  CHECK(result == expectedInner);

  void* outer_only = ctx.translate_ptr(0x10100ULL);
  void* expectedOuter =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(liveOuter) + 0x100);
  CHECK(outer_only == expectedOuter);
}


namespace {
bool kind_uses_hipFree(AllocKind k) {
  switch (k) {
    case AllocKind::Device:         return true;
    case AllocKind::HostMalloc:     return false;  // hipHostFree
    case AllocKind::HostRegister:   return false;  // host_reg_bufs path
    case AllocKind::DevicePtrAlias: return false;  // not separately freed
  }
  return false;
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_AllocKind_FreeDispatch) {
  PlaybackContext ctx;
  void* dev   = reinterpret_cast<void*>(static_cast<uintptr_t>(0x10000000u));
  void* hmal  = reinterpret_cast<void*>(static_cast<uintptr_t>(0x20000000u));
  void* hreg  = reinterpret_cast<void*>(static_cast<uintptr_t>(0x30000000u));
  void* alias = reinterpret_cast<void*>(static_cast<uintptr_t>(0x40000000u));

  ctx.record_alloc(0x1000ULL, dev,   256, AllocKind::Device);
  ctx.record_alloc(0x2000ULL, hmal,  256, AllocKind::HostMalloc);
  ctx.record_alloc(0x3000ULL, hreg,  256, AllocKind::HostRegister);
  ctx.record_alloc(0x4000ULL, alias, 256, AllocKind::DevicePtrAlias);

  REQUIRE(ctx.alloc_map.size() == 4);
  CHECK(ctx.alloc_map.at(0x1000ULL).kind == AllocKind::Device);
  CHECK(ctx.alloc_map.at(0x2000ULL).kind == AllocKind::HostMalloc);
  CHECK(ctx.alloc_map.at(0x3000ULL).kind == AllocKind::HostRegister);
  CHECK(ctx.alloc_map.at(0x4000ULL).kind == AllocKind::DevicePtrAlias);

  for (const auto& [rec, entry] : ctx.alloc_map) {
    if (entry.kind == AllocKind::Device) {
      CHECK(kind_uses_hipFree(entry.kind));
    } else {
      CHECK_FALSE(kind_uses_hipFree(entry.kind));
    }
  }
}


/* The reader keeps undersized known events with zeroed typed fields.
 * The dispatch layer skips events smaller than a header, so none reaches a handler. */
HRR_TEST_CASE(Unit_HRR_Format_SmallPayloadForType) {
  TmpArchive arc("small_payload");

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
  bool ok = hrr::load_archive(arc.path(), archive);
  REQUIRE(ok);

  REQUIRE(archive.events.size() == 1);
  CHECK(archive.events[0].malloc_ev.ptr_handle == 0u);
  CHECK(archive.events[0].malloc_ev.size == 0u);
  CHECK(archive.events[0].raw_payload.size() == sizeof(hrr_event_header));
}

HRR_TEST_CASE(Unit_HRR_Format_UnknownEventType) {
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
  REQUIRE(archive.events.size() == 1);
  CHECK(archive.events[0].header().event_type == 0xFFFFu);
  CHECK(archive.events[0].malloc_ev.ptr_handle == 0u);
}

HRR_TEST_CASE(Unit_HRR_Format_HostSnapshotRecord) {
  auto record = [](uint64_t ptr, uint64_t off, uint64_t len, uint8_t direction) {
    const uint64_t f[5] = {ptr, off, len, 0x1122334455667788ULL, 0x99aabbccddeeff00ULL};
    std::vector<uint8_t> r(kHostSnapRecordSize);
    std::memcpy(r.data(), f, sizeof(f));
    r[40] = direction;
    return r;
  };
  const auto good = record(0x2000, 16, 64, 0);
  TmpArchive arc("host_snapshot");
  auto body = make_launch_with_snapshot(0, good);
  const auto torn = make_launch_with_snapshot(1, good, 1);
  body.insert(body.end(), torn.begin(), torn.end());
  arc.write_events(body);

  hrr::Archive archive;
  REQUIRE(hrr::load_archive(arc.path(), archive));
  REQUIRE(archive.events.size() == 2);
  REQUIRE(archive.events[0].kernel_launch != nullptr);
  REQUIRE(archive.events[0].kernel_launch->snapshots.size() == 1);
  const hrr::BufferSnapshot& s = archive.events[0].kernel_launch->snapshots[0];
  CHECK(s.ptr_handle == 0x2000);
  CHECK(s.offset == 16);
  CHECK(s.length == 64);
  CHECK(s.hash_lo == 0x1122334455667788ULL);
  CHECK(s.hash_hi == 0x99aabbccddeeff00ULL);
  CHECK(s.direction == 0);
  CHECK(archive.events[1].kernel_launch == nullptr);
  CHECK(hrr_host_snapshots_fit(good.data(), good.data() + good.size(), 1));
  CHECK_FALSE(hrr_host_snapshots_fit(good.data(), good.data() + good.size() - 1, 1));

  PlaybackContext ctx;
  std::vector<uint8_t> pinned(128), device(128), blob(64);
  ctx.record_alloc(0x2000, pinned.data(), pinned.size(), AllocKind::HostMalloc);
  ctx.record_alloc(0x3000, device.data(), device.size(), AllocKind::Device);
  const char* why = nullptr;
  CHECK(hrr_host_snapshot_target(ctx, good.data(), blob.data(), blob.size(), &why) ==
        pinned.data() + 16);
  CHECK(why == nullptr);
  CHECK(hrr_host_snapshot_target(ctx, good.data(), nullptr, 0, &why) == nullptr);
  for (const auto& bad : {record(0x3000, 0, 64, 0), record(0x2008, 0, 64, 0),
                          record(0x2000, 80, 64, 0), record(0x2000, 16, 32, 0),
                          record(0x2000, 16, 64, 1)}) {
    why = nullptr;
    CHECK(hrr_host_snapshot_target(ctx, bad.data(), blob.data(), blob.size(), &why) == nullptr);
    CHECK(why != nullptr);
  }
}
