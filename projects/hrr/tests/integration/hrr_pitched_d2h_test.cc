/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR pitched device-to-host validation
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"

#include "hrr_reader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr size_t kDevPitch = 256;
constexpr size_t kDevRows = 8;
constexpr size_t kDevDepth = 3;
constexpr size_t kDevSlice = kDevPitch * kDevRows;
constexpr size_t kDevBytes = kDevSlice * kDevDepth;

constexpr size_t kHostPitch = 160;
constexpr size_t kHostRows = 6;
constexpr size_t kHostDepth = 3;
constexpr size_t kHostSlice = kHostPitch * kHostRows;
constexpr size_t kHostBytes = kHostSlice * kHostDepth;

constexpr size_t kWidth = 48, kHeight = 3, kDepth = 2;
constexpr size_t kSrcX = 16, kSrcY = 2, kSrcZ = 1;
constexpr size_t kDstX = 40, kDstY = 1, kDstZ = 1;

constexpr size_t kFirst3D = kDstZ * kHostSlice + kDstY * kHostPitch + kDstX;

constexpr size_t kRows3D = kWidth * kHeight * kDepth;
constexpr size_t kRows2D = kWidth * kHeight;

// One byte per copy. It fills the copy's host buffer and salts the device
// window the copy reads, so every expected blob is distinct and a test can edit
// one without touching the others. The fill alone would not do it: the blobs
// hold only the copied rows.
constexpr uint8_t kFill[] = {0x31, 0x32, 0x33, 0x34, 0x35, 0x36,
                             0x37, 0x38, 0x39, 0x3A, 0x3B};
constexpr int kPitchedCopies = static_cast<int>(sizeof(kFill));

// Device contents: a byte hash of the offset, with no period that lines up
// with a row or a slice, so reading the wrong offset or pitch reads other bytes.
uint8_t dev_byte(size_t offset) { return static_cast<uint8_t>((offset * 2654435761u) >> 24); }

std::vector<uint8_t> expected_host(uint8_t fill, size_t depth, size_t dst_z) {
  std::vector<uint8_t> host(kHostBytes, fill);
  for (size_t z = 0; z < depth; ++z)
    for (size_t y = 0; y < kHeight; ++y)
      for (size_t x = 0; x < kWidth; ++x)
        host[(dst_z + z) * kHostSlice + (kDstY + y) * kHostPitch + kDstX + x] =
            dev_byte((kSrcZ + z) * kDevSlice + (kSrcY + y) * kDevPitch + kSrcX + x) ^ fill;
  return host;
}

// Empty when the two buffers match, otherwise the first offset where they
// differ. Comparing the vectors directly makes Catch2 print every byte as a
// char, which is unreadable and not valid UTF-8.
std::string byte_diff(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
  if (got.size() != want.size())
    return "size " + std::to_string(got.size()) + ", want " + std::to_string(want.size());
  const auto at = std::mismatch(got.begin(), got.end(), want.begin()).first;
  if (at == got.end()) return {};
  const size_t i = static_cast<size_t>(at - got.begin());
  char buf[64];
  std::snprintf(buf, sizeof(buf), "offset %zu: 0x%02x, want 0x%02x", i, got[i], want[i]);
  return buf;
}

std::vector<uint8_t> packed_rows(const std::vector<uint8_t>& host, size_t first, size_t depth) {
  std::vector<uint8_t> rows;
  for (size_t z = 0; z < depth; ++z)
    for (size_t y = 0; y < kHeight; ++y) {
      const auto row = host.begin() + static_cast<std::ptrdiff_t>(first + z * kHostSlice +
                                                                  y * kHostPitch);
      rows.insert(rows.end(), row, row + kWidth);
    }
  return rows;
}

template <typename Args>
fs::path d2h_blob(const hrr::Archive& arc, hrr_api_id_t api) {
  INFO("API: " << hrr::event_type_name(static_cast<uint16_t>(api)));
  const hrr::Event* event = nullptr;
  for (const auto& e : arc.events) {
    if (e.header().event_type != static_cast<uint16_t>(api)) continue;
    REQUIRE(event == nullptr);
    event = &e;
  }
  REQUIRE(event != nullptr);
  REQUIRE(event->raw_payload.size() >= sizeof(Args));
  const auto* a = reinterpret_cast<const Args*>(event->raw_payload.data());
  REQUIRE((a->d2h_hash_lo != 0 || a->d2h_hash_hi != 0));
  const auto it = arc.blobs.find(hrr::hash_hex(a->d2h_hash_lo, a->d2h_hash_hi));
  REQUIRE(it != arc.blobs.end());
  return it->second;
}

// Flips one byte of a blob file for the lifetime of the object. Playback reads
// blobs by name without re-hashing them, so replay compares against the edit.
struct ScopedBlobEdit {
  fs::path path;
  size_t offset;
  char original = 0;

  ScopedBlobEdit(fs::path p, size_t off) : path(std::move(p)), offset(off) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(f.good());
    f.seekg(static_cast<std::streamoff>(offset));
    f.get(original);
    f.seekp(static_cast<std::streamoff>(offset));
    f.put(static_cast<char>(original ^ 0xFF));
    REQUIRE(f.good());
  }
  ~ScopedBlobEdit() {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(offset));
    f.put(original);
  }
};

// Cuts the last byte off a blob file for the lifetime of the object, leaving it
// short of the rect the way an archive captured before rect-shaped D2H blobs is.
struct ScopedBlobCut {
  fs::path path;
  char last = 0;

  explicit ScopedBlobCut(fs::path p) : path(std::move(p)) {
    {
      std::ifstream in(path, std::ios::binary);
      in.seekg(-1, std::ios::end);
      in.get(last);
      REQUIRE(in.good());
    }
    fs::resize_file(path, fs::file_size(path) - 1);
  }
  ~ScopedBlobCut() {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out.put(last);
  }
};

// Sets the recorded host pitch of the one hipMemcpy3D event in events.bin to
// SIZE_MAX for the lifetime of the object, so the host rect replay derives from
// it overflows size_t the way a corrupt or hostile archive's can.
struct ScopedOverflowingPitch {
  fs::path events;
  size_t at = 0;
  size_t original = 0;

  ScopedOverflowingPitch(const fs::path& cap, const hrr::Archive& arc) {
    const hrr::Event* event = nullptr;
    for (const auto& e : arc.events)
      if (e.header().event_type == static_cast<uint16_t>(HRR_API_HIPMEMCPY3D)) event = &e;
    REQUIRE(event != nullptr);
    const std::vector<fs::path> archives = hrr_process_archives(cap);
    REQUIRE(archives.size() == 1);
    events = archives.front() / "events.bin";
    const std::vector<uint8_t> bytes = read_file(events);
    const auto& payload = event->raw_payload;
    const auto found = std::search(bytes.begin(), bytes.end(), payload.begin(), payload.end());
    REQUIRE(found != bytes.end());
    at = static_cast<size_t>(found - bytes.begin()) + offsetof(hrr_args_hipMemcpy3D, parms_bytes) +
         offsetof(hipMemcpy3DParms, dstPtr) + offsetof(hipPitchedPtr, pitch);
    std::memcpy(&original, bytes.data() + at, sizeof(original));
    REQUIRE(original == kHostPitch);
    write(SIZE_MAX);
  }
  ~ScopedOverflowingPitch() { write(original); }

  void write(size_t pitch) const {
    std::fstream f(events, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(at));
    f.write(reinterpret_cast<const char*>(&pitch), sizeof(pitch));
  }
};

// A v6 reader ignores the file flags and would replay a packed blob with the
// recorded pitch, reading past its end. Only the version keeps it out.
static_assert(HRR_VERSION >= 7, "packed host rects need HRR_VERSION 7 or later");

struct ScopedLegacyArchive {
  fs::path events;
  uint16_t original = 0;

  explicit ScopedLegacyArchive(const fs::path& cap) {
    const std::vector<fs::path> archives = hrr_process_archives(cap);
    REQUIRE(archives.size() == 1);
    events = archives.front() / "events.bin";
    const std::vector<uint8_t> bytes = read_file(events);
    REQUIRE(bytes.size() >= sizeof(hrr_file_header));
    uint16_t version = 0;
    std::memcpy(&version, bytes.data() + offsetof(hrr_file_header, version), sizeof(version));
    REQUIRE(version == HRR_VERSION);
    std::memcpy(&original, bytes.data() + offsetof(hrr_file_header, reserved), sizeof(original));
    REQUIRE((original & HRR_FILE_FLAG_PACKED_HOST_RECTS) != 0);
    write(0);
  }
  ~ScopedLegacyArchive() { write(original); }

  void write(uint16_t flags) const {
    std::fstream f(events, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(offsetof(hrr_file_header, reserved)));
    f.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
  }
};

std::pair<int, std::string> exact_replay(const fs::path& cap) {
  return hrr_playback_env(cap, {{"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0"}});
}

void require_replay(const fs::path& cap, int want_ret, int want_pass, int want_fail) {
  const auto [ret, out] = exact_replay(cap);
  INFO("Playback stdout:\n" << out);
  int pass = 0, fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
  CHECK(pass == want_pass);
  CHECK(fail == want_fail);
  REQUIRE(ret == want_ret);
}

void check_blob_edits(const fs::path& cap, const fs::path& blob, size_t rows) {
  INFO("Expected blob: " << blob.string());
  REQUIRE(fs::file_size(blob) == rows);
  for (const size_t offset : {kWidth, rows - 1}) {
    INFO("Edited offset: " << offset);
    ScopedBlobEdit edit(blob, offset);
    require_replay(cap, 1, kPitchedCopies - 1, 1);
  }
}

}  // namespace

TEST_CASE("Unit_HRR_PitchedD2H_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kDevBytes));
  std::vector<uint8_t> image(kDevBytes);
  auto load = [&](int k) {
    for (size_t i = 0; i < kDevBytes; ++i) image[i] = dev_byte(i) ^ kFill[k];
    HRR_HIP_CHECK(hipMemcpy(dev, image.data(), kDevBytes, hipMemcpyHostToDevice));
  };
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  HIP_MEMCPY3D drv3d{};
  drv3d.srcMemoryType = hipMemoryTypeDevice;
  drv3d.srcDevice = reinterpret_cast<hipDeviceptr_t>(dev);
  drv3d.srcXInBytes = kSrcX;
  drv3d.srcY = kSrcY;
  drv3d.srcZ = kSrcZ;
  drv3d.srcPitch = kDevPitch;
  drv3d.srcHeight = kDevRows;
  drv3d.dstMemoryType = hipMemoryTypeHost;
  drv3d.dstXInBytes = kDstX;
  drv3d.dstY = kDstY;
  drv3d.dstZ = kDstZ;
  drv3d.dstPitch = kHostPitch;
  drv3d.dstHeight = kHostRows;
  drv3d.WidthInBytes = kWidth;
  drv3d.Height = kHeight;
  drv3d.Depth = kDepth;

  load(0);
  std::vector<uint8_t> h0(kHostBytes, kFill[0]);
  drv3d.dstHost = h0.data();
  HRR_HIP_CHECK(hipDrvMemcpy3D(&drv3d));
  REQUIRE(byte_diff(h0, expected_host(kFill[0], kDepth, kDstZ)) == "");

  load(1);
  std::vector<uint8_t> h1(kHostBytes, kFill[1]);
  drv3d.dstHost = h1.data();
  HRR_HIP_CHECK(hipDrvMemcpy3DAsync(&drv3d, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  REQUIRE(byte_diff(h1, expected_host(kFill[1], kDepth, kDstZ)) == "");

  hip_Memcpy2D drv2d{};
  drv2d.srcMemoryType = hipMemoryTypeDevice;
  drv2d.srcDevice = reinterpret_cast<hipDeviceptr_t>(dev);
  drv2d.srcXInBytes = kSrcX;
  drv2d.srcY = kSrcZ * kDevRows + kSrcY;
  drv2d.srcPitch = kDevPitch;
  drv2d.dstMemoryType = hipMemoryTypeHost;
  drv2d.dstXInBytes = kDstX;
  drv2d.dstY = kDstY;
  drv2d.dstPitch = kHostPitch;
  drv2d.WidthInBytes = kWidth;
  drv2d.Height = kHeight;

  load(2);
  std::vector<uint8_t> h2(kHostBytes, kFill[2]);
  drv2d.dstHost = h2.data();
  HRR_HIP_CHECK(hipDrvMemcpy2DUnaligned(&drv2d));
  REQUIRE(byte_diff(h2, expected_host(kFill[2], 1, 0)) == "");

  hipMemcpy3DParms p3d{};
  p3d.srcPtr = make_hipPitchedPtr(dev, kDevPitch, kDevPitch, kDevRows);
  p3d.srcPos = make_hipPos(kSrcX, kSrcY, kSrcZ);
  p3d.dstPos = make_hipPos(kDstX, kDstY, kDstZ);
  p3d.extent = make_hipExtent(kWidth, kHeight, kDepth);
  p3d.kind = hipMemcpyDeviceToHost;

  load(3);
  std::vector<uint8_t> h3(kHostBytes, kFill[3]);
  p3d.dstPtr = make_hipPitchedPtr(h3.data(), kHostPitch, kHostPitch, kHostRows);
  HRR_HIP_CHECK(hipMemcpy3D(&p3d));
  REQUIRE(byte_diff(h3, expected_host(kFill[3], kDepth, kDstZ)) == "");

  load(4);
  std::vector<uint8_t> h4(kHostBytes, kFill[4]);
  p3d.dstPtr = make_hipPitchedPtr(h4.data(), kHostPitch, kHostPitch, kHostRows);
  HRR_HIP_CHECK(hipMemcpy3DAsync(&p3d, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  REQUIRE(byte_diff(h4, expected_host(kFill[4], kDepth, kDstZ)) == "");

  const void* src2d =
      static_cast<const uint8_t*>(dev) + kSrcZ * kDevSlice + kSrcY * kDevPitch + kSrcX;
  const size_t dst2d = kDstY * kHostPitch + kDstX;

  load(5);
  std::vector<uint8_t> h5(kHostBytes, kFill[5]);
  HRR_HIP_CHECK(hipMemcpy2D(h5.data() + dst2d, kHostPitch, src2d, kDevPitch, kWidth, kHeight,
                            hipMemcpyDeviceToHost));
  REQUIRE(byte_diff(h5, expected_host(kFill[5], 1, 0)) == "");

  load(6);
  std::vector<uint8_t> h6(kHostBytes, kFill[6]);
  HRR_HIP_CHECK(hipMemcpy2DAsync(h6.data() + dst2d, kHostPitch, src2d, kDevPitch, kWidth,
                                 kHeight, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  REQUIRE(byte_diff(h6, expected_host(kFill[6], 1, 0)) == "");

  // The async copies below run on the default stream, which capture has to
  // synchronise as well before it reads the host buffer.
  load(7);
  std::vector<uint8_t> h7(kHostBytes, kFill[7]);
  p3d.dstPtr = make_hipPitchedPtr(h7.data(), kHostPitch, kHostPitch, kHostRows);
  HRR_HIP_CHECK(hipMemcpy3D_spt(&p3d));
  REQUIRE(byte_diff(h7, expected_host(kFill[7], kDepth, kDstZ)) == "");

  load(8);
  std::vector<uint8_t> h8(kHostBytes, kFill[8]);
  p3d.dstPtr = make_hipPitchedPtr(h8.data(), kHostPitch, kHostPitch, kHostRows);
  HRR_HIP_CHECK(hipMemcpy3DAsync_spt(&p3d, nullptr));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(byte_diff(h8, expected_host(kFill[8], kDepth, kDstZ)) == "");

  load(9);
  std::vector<uint8_t> h9(kHostBytes, kFill[9]);
  drv2d.dstHost = h9.data();
  HRR_HIP_CHECK(hipMemcpyParam2D(&drv2d));
  REQUIRE(byte_diff(h9, expected_host(kFill[9], 1, 0)) == "");

  load(10);
  std::vector<uint8_t> h10(kHostBytes, kFill[10]);
  drv2d.dstHost = h10.data();
  HRR_HIP_CHECK(hipMemcpyParam2DAsync(&drv2d, nullptr));
  HRR_HIP_CHECK(hipStreamSynchronize(nullptr));
  REQUIRE(byte_diff(h10, expected_host(kFill[10], 1, 0)) == "");

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(dev));
}

HRR_TEST_CASE(Unit_HRR_PitchedD2HRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("pitched D2H HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_d2h"};
  hrr_capture_direct("Unit_HRR_PitchedD2H_Direct", cap.path);
  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    const std::vector<uint8_t> drv3d =
        read_file(d2h_blob<hrr_args_hipDrvMemcpy3D>(arc, HRR_API_HIPDRVMEMCPY3D));
    d2h_blob<hrr_args_hipDrvMemcpy3DAsync>(arc, HRR_API_HIPDRVMEMCPY3DASYNC);
    d2h_blob<hrr_args_hipDrvMemcpy2DUnaligned>(arc, HRR_API_HIPDRVMEMCPY2DUNALIGNED);
    d2h_blob<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D);
    d2h_blob<hrr_args_hipMemcpy3DAsync>(arc, HRR_API_HIPMEMCPY3DASYNC);
    const std::vector<uint8_t> m2d =
        read_file(d2h_blob<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D));
    d2h_blob<hrr_args_hipMemcpy2DAsync>(arc, HRR_API_HIPMEMCPY2DASYNC);
    d2h_blob<hrr_args_hipMemcpy3D_spt>(arc, HRR_API_HIPMEMCPY3D_SPT);
    d2h_blob<hrr_args_hipMemcpy3DAsync_spt>(arc, HRR_API_HIPMEMCPY3DASYNC_SPT);
    d2h_blob<hrr_args_hipMemcpyParam2D>(arc, HRR_API_HIPMEMCPYPARAM2D);
    d2h_blob<hrr_args_hipMemcpyParam2DAsync>(arc, HRR_API_HIPMEMCPYPARAM2DASYNC);
    CHECK(byte_diff(drv3d, packed_rows(expected_host(kFill[0], kDepth, kDstZ), kFirst3D,
                                       kDepth)) == "");
    CHECK(byte_diff(m2d, packed_rows(expected_host(kFill[5], 1, 0), kDstY * kHostPitch + kDstX,
                                     1)) == "");
  }
  require_replay(cap.path, 0, kPitchedCopies, 0);
}

HRR_TEST_CASE(Unit_HRR_PitchedD2HBlobEdits) {
#ifdef _WIN32
  HRR_SKIP("pitched D2H HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_d2h_edits"};
  hrr_capture_direct("Unit_HRR_PitchedD2H_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  SECTION("hipDrvMemcpy3D") {
    check_blob_edits(cap.path, d2h_blob<hrr_args_hipDrvMemcpy3D>(arc, HRR_API_HIPDRVMEMCPY3D),
                     kRows3D);
  }
  SECTION("hipMemcpy3D") {
    check_blob_edits(cap.path, d2h_blob<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D),
                     kRows3D);
  }
  SECTION("hipMemcpy2D") {
    check_blob_edits(cap.path, d2h_blob<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D),
                     kRows2D);
  }
}

HRR_TEST_CASE(Unit_HRR_PitchedD2HShortBlobs) {
#ifdef _WIN32
  HRR_SKIP("pitched D2H HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_d2h_short"};
  hrr_capture_direct("Unit_HRR_PitchedD2H_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  const std::vector<fs::path> blobs = {
      d2h_blob<hrr_args_hipDrvMemcpy3D>(arc, HRR_API_HIPDRVMEMCPY3D),
      d2h_blob<hrr_args_hipDrvMemcpy3DAsync>(arc, HRR_API_HIPDRVMEMCPY3DASYNC),
      d2h_blob<hrr_args_hipDrvMemcpy2DUnaligned>(arc, HRR_API_HIPDRVMEMCPY2DUNALIGNED),
      d2h_blob<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D),
      d2h_blob<hrr_args_hipMemcpy3DAsync>(arc, HRR_API_HIPMEMCPY3DASYNC),
      d2h_blob<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D),
      d2h_blob<hrr_args_hipMemcpy2DAsync>(arc, HRR_API_HIPMEMCPY2DASYNC),
      d2h_blob<hrr_args_hipMemcpy3D_spt>(arc, HRR_API_HIPMEMCPY3D_SPT),
      d2h_blob<hrr_args_hipMemcpy3DAsync_spt>(arc, HRR_API_HIPMEMCPY3DASYNC_SPT),
      d2h_blob<hrr_args_hipMemcpyParam2D>(arc, HRR_API_HIPMEMCPYPARAM2D),
      d2h_blob<hrr_args_hipMemcpyParam2DAsync>(arc, HRR_API_HIPMEMCPYPARAM2DASYNC)};
  REQUIRE(blobs.size() == static_cast<size_t>(kPitchedCopies));
  const auto all_skipped = [&cap] {
    const auto [ret, out] = exact_replay(cap.path);
    INFO("Playback stdout:\n" << out);
    CHECK(out.find(", 0 fail, " + std::to_string(kPitchedCopies) + " skipped") !=
          std::string::npos);
    CHECK(ret == 1);
  };
  {
    ScopedBlobCut cut(blobs[3]);
    const auto [ret, out] = exact_replay(cap.path);
    INFO("Playback stdout:\n" << out);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(pass == kPitchedCopies - 1);
    CHECK(out.find(", 0 fail, 1 skipped") != std::string::npos);
    CHECK(ret == 0);
  }
  {
    std::vector<std::unique_ptr<ScopedBlobCut>> cuts;
    for (const auto& b : blobs) cuts.push_back(std::make_unique<ScopedBlobCut>(b));
    all_skipped();
  }
  {
    ScopedLegacyArchive legacy(cap.path);
    all_skipped();
  }
  {
    ScopedOverflowingPitch overflow(cap.path, arc);
    require_replay(cap.path, 0, kPitchedCopies, 0);
    ScopedLegacyArchive legacy(cap.path);
    all_skipped();
  }
}

// A hipMemcpy2D D2H whose host rows are a page apart with a PROT_NONE page between: capture must not read it.
TEST_CASE("Unit_HRR_PitchedD2HGap_Direct", "[.][hrr-direct]") {
#ifndef _WIN32
  HRR_HIP_CHECK(hipSetDevice(0));
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  REQUIRE(page > kWidth);
  void* map = mmap(nullptr, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  REQUIRE(map != MAP_FAILED);
  uint8_t* base = static_cast<uint8_t*>(map);
  std::memset(base, kFill[0], 3 * page);
  REQUIRE(mprotect(base + page, page, PROT_NONE) == 0);

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kDevBytes));
  std::vector<uint8_t> image(kDevBytes);
  for (size_t i = 0; i < kDevBytes; ++i) image[i] = dev_byte(i);
  HRR_HIP_CHECK(hipMemcpy(dev, image.data(), kDevBytes, hipMemcpyHostToDevice));

  uint8_t* dst = base + page - kWidth;
  const size_t dpitch = 2 * page;
  const auto* src = static_cast<const uint8_t*>(dev) + kSrcY * kDevPitch + kSrcX;
  HRR_HIP_CHECK(hipMemcpy2D(dst, dpitch, src, kDevPitch, kWidth, 2, hipMemcpyDeviceToHost));
  for (size_t y = 0; y < 2; ++y)
    for (size_t x = 0; x < kWidth; ++x)
      REQUIRE(dst[y * dpitch + x] == dev_byte((kSrcY + y) * kDevPitch + kSrcX + x));

  HRR_HIP_CHECK(hipFree(dev));
  REQUIRE(munmap(map, 3 * page) == 0);
#endif
}

HRR_TEST_CASE(Unit_HRR_PitchedD2HGapRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("pitched D2H HRR roundtrip is disabled on Windows");
#else
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_d2h_gap"};
  hrr_capture_direct("Unit_HRR_PitchedD2HGap_Direct", cap.path);
  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    const std::vector<uint8_t> blob =
        read_file(d2h_blob<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D));
    std::vector<uint8_t> want(2 * kWidth);
    for (size_t y = 0; y < 2; ++y)
      for (size_t x = 0; x < kWidth; ++x)
        want[y * kWidth + x] = dev_byte((kSrcY + y) * kDevPitch + kSrcX + x);
    CHECK(byte_diff(blob, want) == "");
  }
  require_replay(cap.path, 0, 1, 0);
#endif
}


namespace {

constexpr uint8_t kDenseFill[] = {0x71, 0x72};
constexpr int kDenseCopies = 2;

constexpr size_t kDenseRowFirst = kFirst3D;
constexpr size_t kDenseRowExtent = kDenseRowFirst + kWidth;

constexpr size_t kPackedSlice = kWidth * kHeight;
constexpr size_t kPackedFirst = kPackedSlice;
constexpr size_t kPackedExtent = kPackedFirst + kDepth * kPackedSlice;

std::vector<uint8_t> expected_dense_row(uint8_t fill) {
  std::vector<uint8_t> host(kHostBytes, fill);
  for (size_t x = 0; x < kWidth; ++x)
    host[kDenseRowFirst + x] =
        dev_byte(kSrcZ * kDevSlice + kSrcY * kDevPitch + kSrcX + x) ^ fill;
  return host;
}

std::vector<uint8_t> expected_packed(uint8_t fill) {
  std::vector<uint8_t> host(kPackedExtent, fill);
  for (size_t z = 0; z < kDepth; ++z)
    for (size_t y = 0; y < kHeight; ++y)
      for (size_t x = 0; x < kWidth; ++x)
        host[kPackedFirst + z * kPackedSlice + y * kWidth + x] =
            dev_byte((kSrcZ + z) * kDevSlice + (kSrcY + y) * kDevPitch + kSrcX + x) ^ fill;
  return host;
}

std::vector<uint8_t> dense_blob(const std::vector<uint8_t>& host, size_t first, size_t extent) {
  return {host.begin() + static_cast<std::ptrdiff_t>(first),
          host.begin() + static_cast<std::ptrdiff_t>(extent)};
}

}  // namespace

TEST_CASE("Unit_HRR_DenseOffsetD2H_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kDevBytes));
  std::vector<uint8_t> image(kDevBytes);
  auto load = [&](uint8_t fill) {
    for (size_t i = 0; i < kDevBytes; ++i) image[i] = dev_byte(i) ^ fill;
    HRR_HIP_CHECK(hipMemcpy(dev, image.data(), kDevBytes, hipMemcpyHostToDevice));
  };

  HIP_MEMCPY3D drv3d{};
  drv3d.srcMemoryType = hipMemoryTypeDevice;
  drv3d.srcDevice = reinterpret_cast<hipDeviceptr_t>(dev);
  drv3d.srcXInBytes = kSrcX;
  drv3d.srcY = kSrcY;
  drv3d.srcZ = kSrcZ;
  drv3d.srcPitch = kDevPitch;
  drv3d.srcHeight = kDevRows;
  drv3d.dstMemoryType = hipMemoryTypeHost;
  drv3d.dstXInBytes = kDstX;
  drv3d.dstY = kDstY;
  drv3d.dstZ = kDstZ;
  drv3d.dstPitch = kHostPitch;
  drv3d.dstHeight = kHostRows;
  drv3d.WidthInBytes = kWidth;
  drv3d.Height = 1;
  drv3d.Depth = 1;

  load(kDenseFill[0]);
  std::vector<uint8_t> h0(kHostBytes, kDenseFill[0]);
  drv3d.dstHost = h0.data();
  HRR_HIP_CHECK(hipDrvMemcpy3D(&drv3d));
  REQUIRE(byte_diff(h0, expected_dense_row(kDenseFill[0])) == "");

  load(kDenseFill[1]);
  std::vector<uint8_t> h1(kPackedExtent, kDenseFill[1]);
  hipMemcpy3DParms p3d{};
  p3d.srcPtr = make_hipPitchedPtr(dev, kDevPitch, kDevPitch, kDevRows);
  p3d.srcPos = make_hipPos(kSrcX, kSrcY, kSrcZ);
  p3d.dstPtr = make_hipPitchedPtr(h1.data(), kWidth, kWidth, kHeight);
  p3d.dstPos = make_hipPos(0, 0, 1);
  p3d.extent = make_hipExtent(kWidth, kHeight, kDepth);
  p3d.kind = hipMemcpyDeviceToHost;
  HRR_HIP_CHECK(hipMemcpy3D(&p3d));
  REQUIRE(byte_diff(h1, expected_packed(kDenseFill[1])) == "");

  HRR_HIP_CHECK(hipFree(dev));
}

HRR_TEST_CASE(Unit_HRR_DenseOffsetD2HRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("dense offset D2H HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_dense_offset_d2h"};
  hrr_capture_direct("Unit_HRR_DenseOffsetD2H_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  const fs::path row = d2h_blob<hrr_args_hipDrvMemcpy3D>(arc, HRR_API_HIPDRVMEMCPY3D);
  const fs::path packed = d2h_blob<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D);
  CHECK(byte_diff(read_file(row), dense_blob(expected_dense_row(kDenseFill[0]),
                                             kDenseRowFirst, kDenseRowExtent)) == "");
  CHECK(byte_diff(read_file(packed), dense_blob(expected_packed(kDenseFill[1]), kPackedFirst,
                                                kPackedExtent)) == "");
  require_replay(cap.path, 0, kDenseCopies, 0);

  const std::pair<fs::path, std::pair<size_t, size_t>> edits[] = {
      {row, {kDenseRowFirst, kDenseRowExtent}}, {packed, {kPackedFirst, kPackedExtent}}};
  for (const auto& [blob, span] : edits) {
    INFO("Expected blob: " << blob.string());
    REQUIRE(fs::file_size(blob) == span.second - span.first);
    for (const size_t offset : {size_t{0}, span.second - span.first - 1}) {
      INFO("Edited offset: " << offset);
      ScopedBlobEdit edit(blob, offset);
      require_replay(cap.path, 1, kDenseCopies - 1, 1);
    }
  }
}


namespace {

constexpr uint8_t kDevFill[] = {0x51, 0x52, 0x53, 0x54};
constexpr int kH2DCopies = 4;

uint8_t host_byte(size_t offset) { return static_cast<uint8_t>(dev_byte(offset) ^ 0xA5); }

std::vector<uint8_t> expected_dev(uint8_t fill) {
  std::vector<uint8_t> dev(kDevBytes, fill);
  for (size_t z = 0; z < kDepth; ++z)
    for (size_t y = 0; y < kHeight; ++y)
      for (size_t x = 0; x < kWidth; ++x)
        dev[(kSrcZ + z) * kDevSlice + (kSrcY + y) * kDevPitch + kSrcX + x] =
            host_byte((kDstZ + z) * kHostSlice + (kDstY + y) * kHostPitch + kDstX + x);
  return dev;
}

template <typename Args>
std::vector<const Args*> event_args(const hrr::Archive& arc, hrr_api_id_t api) {
  std::vector<const Args*> out;
  for (const auto& e : arc.events) {
    if (e.header().event_type != static_cast<uint16_t>(api)) continue;
    REQUIRE(e.raw_payload.size() >= sizeof(Args));
    out.push_back(reinterpret_cast<const Args*>(e.raw_payload.data()));
  }
  return out;
}

fs::path blob_path(const hrr::Archive& arc, uint64_t lo, uint64_t hi) {
  REQUIRE((lo != 0 || hi != 0));
  const auto it = arc.blobs.find(hrr::hash_hex(lo, hi));
  REQUIRE(it != arc.blobs.end());
  return it->second;
}

template <typename Args>
std::pair<uint64_t, uint64_t> h2d_hash(const hrr::Archive& arc, hrr_api_id_t api) {
  const auto args = event_args<Args>(arc, api);
  REQUIRE(args.size() == 1);
  return {args[0]->blob_hash_lo, args[0]->blob_hash_hi};
}

fs::path h2d_blob(const hrr::Archive& arc) {
  const auto sync = h2d_hash<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D);
  REQUIRE(h2d_hash<hrr_args_hipMemcpy3DAsync>(arc, HRR_API_HIPMEMCPY3DASYNC) == sync);
  REQUIRE(h2d_hash<hrr_args_hipMemcpy3D_spt>(arc, HRR_API_HIPMEMCPY3D_SPT) == sync);
  REQUIRE(h2d_hash<hrr_args_hipMemcpy3DAsync_spt>(arc, HRR_API_HIPMEMCPY3DASYNC_SPT) == sync);
  return blob_path(arc, sync.first, sync.second);
}

}  // namespace

TEST_CASE("Unit_HRR_PitchedH2D_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  std::vector<uint8_t> host(kHostBytes);
  for (size_t i = 0; i < kHostBytes; ++i) host[i] = host_byte(i);
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  hipMemcpy3DParms p{};
  p.srcPtr = make_hipPitchedPtr(host.data(), kHostPitch, kHostPitch, kHostRows);
  p.srcPos = make_hipPos(kDstX, kDstY, kDstZ);
  p.dstPos = make_hipPos(kSrcX, kSrcY, kSrcZ);
  p.extent = make_hipExtent(kWidth, kHeight, kDepth);
  p.kind = hipMemcpyHostToDevice;

  void* dev[kH2DCopies] = {};
  for (int i = 0; i < kH2DCopies; ++i) {
    HRR_HIP_CHECK(hipMalloc(&dev[i], kDevBytes));
    HRR_HIP_CHECK(hipMemset(dev[i], kDevFill[i], kDevBytes));
    // hipMemset can return before the fill lands, and the non-blocking stream
    // does not wait for the null stream: without this the fill can overwrite
    // the async copy, at capture and again at replay.
    HRR_HIP_CHECK(hipDeviceSynchronize());
    p.dstPtr = make_hipPitchedPtr(dev[i], kDevPitch, kDevPitch, kDevRows);
    switch (i) {
      case 0:
        HRR_HIP_CHECK(hipMemcpy3D(&p));
        break;
      case 1:
        HRR_HIP_CHECK(hipMemcpy3DAsync(&p, s));
        HRR_HIP_CHECK(hipStreamSynchronize(s));
        break;
      case 2:
        HRR_HIP_CHECK(hipMemcpy3D_spt(&p));
        break;
      default:
        HRR_HIP_CHECK(hipMemcpy3DAsync_spt(&p, hipStreamPerThread));
        HRR_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        break;
    }
    std::vector<uint8_t> out(kDevBytes);
    HRR_HIP_CHECK(hipMemcpy(out.data(), dev[i], kDevBytes, hipMemcpyDeviceToHost));
    REQUIRE(byte_diff(out, expected_dev(kDevFill[i])) == "");
  }

  for (void* d : dev) {
    HRR_HIP_CHECK(hipFree(d));
  }
  HRR_HIP_CHECK(hipStreamDestroy(s));
}

HRR_TEST_CASE(Unit_HRR_PitchedH2DRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("pitched H2D HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_h2d"};
  hrr_capture_direct("Unit_HRR_PitchedH2D_Direct", cap.path);
  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    const std::vector<uint8_t> src = read_file(h2d_blob(arc));
    std::vector<uint8_t> host(kHostBytes);
    for (size_t i = 0; i < kHostBytes; ++i) host[i] = host_byte(i);
    CHECK(byte_diff(src, packed_rows(host, kFirst3D, kDepth)) == "");
  }
  require_replay(cap.path, 0, kH2DCopies, 0);
}

HRR_TEST_CASE(Unit_HRR_PitchedH2DShortBlob) {
#ifdef _WIN32
  HRR_SKIP("pitched H2D HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_h2d_short"};
  hrr_capture_direct("Unit_HRR_PitchedH2D_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));

  REQUIRE(fs::file_size(h2d_blob(arc)) == kRows3D);
  ScopedLegacyArchive legacy(cap.path);

  int readbacks = 0;
  for (const auto* a : event_args<hrr_args_hipMemcpy>(arc, HRR_API_HIPMEMCPY)) {
    if (a->kind != hipMemcpyDeviceToHost) continue;
    const fs::path blob = blob_path(arc, a->blob_hash_lo, a->blob_hash_hi);
    const std::vector<uint8_t> want = read_file(blob);
    REQUIRE(want.size() == kDevBytes);
    write_file(blob, std::vector<uint8_t>(kDevBytes, want[0]));
    ++readbacks;
  }
  REQUIRE(readbacks == kH2DCopies);
  require_replay(cap.path, 0, kH2DCopies, 0);
}


namespace {

constexpr uint8_t kDev2DFill[] = {0x61, 0x62};
constexpr int kH2D2DCopies = 2;

std::vector<uint8_t> expected_dev_2d(uint8_t fill) {
  std::vector<uint8_t> dev(kDevBytes, fill);
  for (size_t y = 0; y < kHeight; ++y)
    for (size_t x = 0; x < kWidth; ++x)
      dev[(kSrcY + y) * kDevPitch + kSrcX + x] = host_byte((kDstY + y) * kHostPitch + kDstX + x);
  return dev;
}

fs::path h2d_2d_blob(const hrr::Archive& arc) {
  const auto sync = h2d_hash<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D);
  REQUIRE(h2d_hash<hrr_args_hipMemcpy2DAsync>(arc, HRR_API_HIPMEMCPY2DASYNC) == sync);
  return blob_path(arc, sync.first, sync.second);
}

}  // namespace

TEST_CASE("Unit_HRR_PitchedH2D2D_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  std::vector<uint8_t> host(kHostBytes);
  for (size_t i = 0; i < kHostBytes; ++i) host[i] = host_byte(i);
  const uint8_t* src = host.data() + kDstY * kHostPitch + kDstX;
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  void* dev[kH2D2DCopies] = {};
  for (int i = 0; i < kH2D2DCopies; ++i) {
    HRR_HIP_CHECK(hipMalloc(&dev[i], kDevBytes));
    HRR_HIP_CHECK(hipMemset(dev[i], kDev2DFill[i], kDevBytes));
    HRR_HIP_CHECK(hipDeviceSynchronize());
    uint8_t* dst = static_cast<uint8_t*>(dev[i]) + kSrcY * kDevPitch + kSrcX;
    if (i == 0) {
      HRR_HIP_CHECK(hipMemcpy2D(dst, kDevPitch, src, kHostPitch, kWidth, kHeight,
                                hipMemcpyHostToDevice));
    } else {
      HRR_HIP_CHECK(hipMemcpy2DAsync(dst, kDevPitch, src, kHostPitch, kWidth, kHeight,
                                     hipMemcpyHostToDevice, s));
      HRR_HIP_CHECK(hipStreamSynchronize(s));
    }
    std::vector<uint8_t> out(kDevBytes);
    HRR_HIP_CHECK(hipMemcpy(out.data(), dev[i], kDevBytes, hipMemcpyDeviceToHost));
    REQUIRE(byte_diff(out, expected_dev_2d(kDev2DFill[i])) == "");
  }

  for (void* d : dev) {
    HRR_HIP_CHECK(hipFree(d));
  }
  HRR_HIP_CHECK(hipStreamDestroy(s));
}

HRR_TEST_CASE(Unit_HRR_PitchedH2D2DRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("pitched H2D HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_h2d_2d"};
  hrr_capture_direct("Unit_HRR_PitchedH2D2D_Direct", cap.path);
  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    const std::vector<uint8_t> src = read_file(h2d_2d_blob(arc));
    std::vector<uint8_t> host(kHostBytes);
    for (size_t i = 0; i < kHostBytes; ++i) host[i] = host_byte(i);
    CHECK(byte_diff(src, packed_rows(host, kDstY * kHostPitch + kDstX, 1)) == "");
  }
  require_replay(cap.path, 0, kH2D2DCopies, 0);
}

HRR_TEST_CASE(Unit_HRR_PitchedH2D2DShortBlob) {
#ifdef _WIN32
  HRR_SKIP("pitched H2D HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_h2d_2d_short"};
  hrr_capture_direct("Unit_HRR_PitchedH2D2D_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));

  const fs::path src = h2d_2d_blob(arc);
  std::vector<uint8_t> blob = read_file(src);
  REQUIRE(blob.size() == kRows2D);
  blob.pop_back();
  write_file(src, blob);

  int readbacks = 0;
  for (const auto* a : event_args<hrr_args_hipMemcpy>(arc, HRR_API_HIPMEMCPY)) {
    if (a->kind != hipMemcpyDeviceToHost) continue;
    const fs::path path = blob_path(arc, a->blob_hash_lo, a->blob_hash_hi);
    const std::vector<uint8_t> want = read_file(path);
    REQUIRE(want.size() == kDevBytes);
    write_file(path, std::vector<uint8_t>(kDevBytes, want[0]));
    ++readbacks;
  }
  REQUIRE(readbacks == kH2D2DCopies);
  require_replay(cap.path, 0, kH2D2DCopies, 0);
}

// A 48 MiB host span with only 192 bytes copied: blobs must cost the bytes copied, not the span.

namespace {

constexpr size_t kSparsePitch = size_t{16} << 20;
constexpr size_t kSparseRows = 4;
constexpr size_t kSparseBytes = kSparseRows * kWidth;

}  // namespace

TEST_CASE("Unit_HRR_SparsePitch_Direct", "[.][hrr-direct]") {
#ifndef _WIN32
  HRR_HIP_CHECK(hipSetDevice(0));
  const size_t span = (kSparseRows - 1) * kSparsePitch + kWidth;
  void* map = mmap(nullptr, span, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  REQUIRE(map != MAP_FAILED);
  uint8_t* host = static_cast<uint8_t*>(map);
  for (size_t y = 0; y < kSparseRows; ++y)
    for (size_t x = 0; x < kWidth; ++x) host[y * kSparsePitch + x] = host_byte(y * kWidth + x);

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kSparseBytes));
  HRR_HIP_CHECK(hipMemcpy2D(dev, kWidth, host, kSparsePitch, kWidth, kSparseRows,
                            hipMemcpyHostToDevice));
  for (size_t y = 0; y < kSparseRows; ++y) std::memset(host + y * kSparsePitch, 0, kWidth);
  HRR_HIP_CHECK(hipMemcpy2D(host, kSparsePitch, dev, kWidth, kWidth, kSparseRows,
                            hipMemcpyDeviceToHost));
  for (size_t y = 0; y < kSparseRows; ++y)
    for (size_t x = 0; x < kWidth; ++x)
      REQUIRE(host[y * kSparsePitch + x] == host_byte(y * kWidth + x));

  HRR_HIP_CHECK(hipFree(dev));
  REQUIRE(munmap(map, span) == 0);
#endif
}

HRR_TEST_CASE(Unit_HRR_SparsePitchRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("sparse pitch HRR roundtrip is disabled on Windows");
#else
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_sparse_pitch"};
  hrr_capture_direct("Unit_HRR_SparsePitch_Direct", cap.path);
  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    const auto args = event_args<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D);
    REQUIRE(args.size() == 2);
    std::vector<uint8_t> want(kSparseBytes);
    for (size_t i = 0; i < kSparseBytes; ++i) want[i] = host_byte(i);
    for (const auto* a : args) {
      const bool h2d = a->kind == hipMemcpyHostToDevice;
      INFO("Direction: " << (h2d ? "H2D" : "D2H"));
      const fs::path blob = h2d ? blob_path(arc, a->blob_hash_lo, a->blob_hash_hi)
                                : blob_path(arc, a->d2h_hash_lo, a->d2h_hash_hi);
      REQUIRE(fs::file_size(blob) == kSparseBytes);
      CHECK(byte_diff(read_file(blob), want) == "");
    }
  }
  require_replay(cap.path, 0, 1, 0);
#endif
}

// The runtime refuses a null device side before touching the host rect, so capture must not read that rect either.
TEST_CASE("Unit_HRR_RejectedMemcpy3D_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  std::vector<uint8_t> host(kHostBytes, 0x5A);
  hipMemcpy3DParms h2d{};
  h2d.srcPtr = make_hipPitchedPtr(host.data(), kHostPitch, kHostPitch, kHostRows);
  h2d.srcPos = make_hipPos(kDstX, kDstY, kDstZ);
  h2d.dstPtr = make_hipPitchedPtr(nullptr, kDevPitch, kDevPitch, kDevRows);
  h2d.dstPos = make_hipPos(kSrcX, kSrcY, kSrcZ);
  h2d.extent = make_hipExtent(kWidth, kHeight, kDepth);
  h2d.kind = hipMemcpyHostToDevice;

  hipMemcpy3DParms d2h{};
  d2h.srcPtr = make_hipPitchedPtr(nullptr, kDevPitch, kDevPitch, kDevRows);
  d2h.srcPos = make_hipPos(kSrcX, kSrcY, kSrcZ);
  d2h.dstPtr = make_hipPitchedPtr(host.data(), kHostPitch, kHostPitch, kHostRows);
  d2h.dstPos = make_hipPos(kDstX, kDstY, kDstZ);
  d2h.extent = make_hipExtent(kWidth, kHeight, kDepth);
  d2h.kind = hipMemcpyDeviceToHost;

  for (const hipMemcpy3DParms* p : {&h2d, &d2h}) {
    REQUIRE(hipMemcpy3D(p) == hipErrorInvalidValue);
    REQUIRE(hipMemcpy3DAsync(p, nullptr) == hipErrorInvalidValue);
    REQUIRE(hipMemcpy3D_spt(p) == hipErrorInvalidValue);
    REQUIRE(hipMemcpy3DAsync_spt(p, nullptr) == hipErrorInvalidValue);
  }
  (void)hipGetLastError();
}

namespace {

template <typename Args>
size_t check_rejected(const hrr::Archive& arc, hrr_api_id_t api) {
  const auto args = event_args<Args>(arc, api);
  for (const auto* a : args) {
    CHECK(a->ret != 0);
    CHECK(a->blob_hash_lo == 0);
    CHECK(a->blob_hash_hi == 0);
    CHECK(a->d2h_hash_lo == 0);
    CHECK(a->d2h_hash_hi == 0);
  }
  return args.size();
}

}  // namespace

HRR_TEST_CASE(Unit_HRR_RejectedMemcpy3DNoBlob) {
#ifdef _WIN32
  HRR_SKIP("rejected hipMemcpy3D HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_rejected_memcpy3d"};
  hrr_capture_direct("Unit_HRR_RejectedMemcpy3D_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  const size_t recorded =
      check_rejected<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D) +
      check_rejected<hrr_args_hipMemcpy3DAsync>(arc, HRR_API_HIPMEMCPY3DASYNC) +
      check_rejected<hrr_args_hipMemcpy3D_spt>(arc, HRR_API_HIPMEMCPY3D_SPT) +
      check_rejected<hrr_args_hipMemcpy3DAsync_spt>(arc, HRR_API_HIPMEMCPY3DASYNC_SPT);
  CHECK((recorded == 0 || recorded == 8));
  const auto [ret, out] = exact_replay(cap.path);
  INFO("Playback stdout:\n" << out);
  REQUIRE(ret == 0);
}

/**
 * @}
 */
