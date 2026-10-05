/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR pitched device-to-host validation
 * @{
 * @ingroup HRRTest
 * Device-to-host copies whose host side is a pitched rect: hipDrvMemcpy3D,
 * hipDrvMemcpy3DAsync, hipDrvMemcpy2DUnaligned, hipMemcpy3D, hipMemcpy3DAsync,
 * hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy3D_spt, hipMemcpy3DAsync_spt,
 * hipMemcpyParam2D and hipMemcpyParam2DAsync.
 *
 * Every copy reads a window at a non-zero offset of a pitched device buffer
 * into a window at a different offset of a host buffer with a different pitch,
 * so the copied rows are neither dense nor at the start of either buffer.
 * Replay has to compare exactly those rows. The first width*height*depth bytes
 * of either buffer are mostly bytes the copy never touched, which fails a
 * correct replay, and they miss the later rows, which passes a wrong one.
 * Two dense copies whose copied run starts past the host base pointer cover the
 * case where the rows are contiguous but still not at the start of the blob.
 *
 * The same rects serve pitched host-to-device copies through all four
 * hipMemcpy3D spellings and through hipMemcpy2D and hipMemcpy2DAsync, whose
 * source blob replay reads with the recorded pitch and position, and rejected
 * ones, which must leave no blob at all.
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

// Device buffer: kDevDepth slices of kDevRows rows, rows kDevPitch bytes apart.
constexpr size_t kDevPitch = 256;
constexpr size_t kDevRows = 8;
constexpr size_t kDevDepth = 3;
constexpr size_t kDevSlice = kDevPitch * kDevRows;
constexpr size_t kDevBytes = kDevSlice * kDevDepth;

// Host buffers: a different pitch, so no copied row lines up with the device.
constexpr size_t kHostPitch = 160;
constexpr size_t kHostRows = 6;
constexpr size_t kHostDepth = 3;
constexpr size_t kHostSlice = kHostPitch * kHostRows;
constexpr size_t kHostBytes = kHostSlice * kHostDepth;

// The copied window and where it starts on each side. The 2D copies take the
// first slice of it, into host slice 0.
constexpr size_t kWidth = 48, kHeight = 3, kDepth = 2;
constexpr size_t kSrcX = 16, kSrcY = 2, kSrcZ = 1;
constexpr size_t kDstX = 40, kDstY = 1, kDstZ = 1;

// Offsets into the expected blobs, which span the host destination from the
// pointer the copy was given through its last copied byte. hipMemcpy2D has no
// offset arguments, so its pointer is already the first copied byte.
constexpr size_t kFirst3D = kDstZ * kHostSlice + kDstY * kHostPitch + kDstX;
constexpr size_t kLastRow3D = kFirst3D + (kDepth - 1) * kHostSlice + (kHeight - 1) * kHostPitch;
constexpr size_t kExtent3D = kLastRow3D + kWidth;
constexpr size_t kLastRow2D = (kHeight - 1) * kHostPitch;
constexpr size_t kExtent2D = kLastRow2D + kWidth;

// One byte per copy. It fills the copy's host buffer and salts the device
// window the copy reads, so every expected blob is distinct and a test can edit
// one without touching the others. The fill alone would not do it: capture
// leaves every byte around the copied rows zero in the blob.
constexpr uint8_t kFill[] = {0x31, 0x32, 0x33, 0x34, 0x35, 0x36,
                             0x37, 0x38, 0x39, 0x3A, 0x3B};
constexpr int kPitchedCopies = static_cast<int>(sizeof(kFill));

// Device contents: a byte hash of the offset, with no period that lines up
// with a row or a slice, so reading the wrong offset or pitch reads other bytes.
uint8_t dev_byte(size_t offset) { return static_cast<uint8_t>((offset * 2654435761u) >> 24); }

// A host buffer after copying `depth` slices of the window, salted with
// `fill`, into host slice `dst_z` onwards: `fill` everywhere except the copied
// rows.
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

std::vector<uint8_t> read_file(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

// True when `blob`, laid out like the host buffer from `first`, holds the
// copied rows of `host` and zero in every byte around them.
bool rows_only(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& host, size_t base,
               size_t first, size_t depth) {
  std::vector<uint8_t> want(blob.size(), 0);
  for (size_t z = 0; z < depth; ++z)
    for (size_t y = 0; y < kHeight; ++y) {
      const size_t off = first + z * kHostSlice + y * kHostPitch;
      if (off + kWidth > want.size()) return false;
      std::memcpy(want.data() + off, host.data() + base + off, kWidth);
    }
  return blob == want;
}

// The expected-output blob of the one `api` event in the archive.
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

// Byte-exact D2H with the divergence guard off, so the exit code is the D2H
// verdict alone: 0 when every check passes, 1 when one fails.
std::pair<int, std::string> exact_replay(const fs::path& cap) {
  return hrr_playback_env(cap, {{"HIP_HRR_D2H_EXACT", "1"},
                                {"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0"}});
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

// Edits one expected blob twice and replays after each edit: first a byte
// between the first two copied rows, which the copy never wrote and replay must
// not judge, then a byte of the last copied row, which replay must report.
void check_blob_edits(const fs::path& cap, const fs::path& blob, size_t first, size_t last_row,
                      size_t extent) {
  INFO("Expected blob: " << blob.string());
  REQUIRE(fs::file_size(blob) == extent);
  {
    ScopedBlobEdit padding(blob, first + kWidth);
    require_replay(cap, 0, kPitchedCopies, 0);
  }
  {
    ScopedBlobEdit row(blob, last_row + kWidth / 2);
    require_replay(cap, 1, kPitchedCopies - 1, 1);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Workload: the eleven pitched copies, each into its own host buffer, each
// checked here against the layout the replay assertions assume.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_PitchedD2H_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kDevBytes));
  // Salts the device buffer with copy k's fill byte; see kFill.
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

  // hipMemcpy2D takes no offsets: the window is wherever the pointers point.
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

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedD2H_Direct and check that each of the eleven
 *     pitched copies recorded an expected-output blob, and that the
 *     hipDrvMemcpy3D and hipMemcpy2D blobs hold the copied rows and zero in
 *     every byte around them, not the host buffer's fill.
 *   - Replay with HIP_HRR_D2H_EXACT=1: all eleven checks must pass. Comparing
 *     the first width*height*depth bytes of each side instead compares device
 *     bytes outside the window with host fill bytes, and fails.
 */
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
    // The host bytes around the copied rows never reach the archive.
    REQUIRE(drv3d.size() == kExtent3D);
    CHECK(rows_only(drv3d, expected_host(kFill[0], kDepth, kDstZ), 0, kFirst3D, kDepth));
    REQUIRE(m2d.size() == kExtent2D);
    CHECK(rows_only(m2d, expected_host(kFill[5], 1, 0), kDstY * kHostPitch + kDstX, 0, 1));
  }
  require_replay(cap.path, 0, kPitchedCopies, 0);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedD2H_Direct, then edit the expected blob of
 *     hipDrvMemcpy3D, hipMemcpy3D and hipMemcpy2D in turn, replaying with
 *     HIP_HRR_D2H_EXACT=1 after each edit.
 *   - A flipped byte between the first two copied rows must not be reported:
 *     the copy never wrote it, and comparing it fails a correct replay.
 *   - A flipped byte in the last copied row must fail exactly that check.
 *     Comparing only the first width*height*depth bytes never reaches it.
 */
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
                     kFirst3D, kLastRow3D, kExtent3D);
  }
  SECTION("hipMemcpy3D") {
    check_blob_edits(cap.path, d2h_blob<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D),
                     kFirst3D, kLastRow3D, kExtent3D);
  }
  SECTION("hipMemcpy2D") {
    check_blob_edits(cap.path, d2h_blob<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D), 0,
                     kLastRow2D, kExtent2D);
  }
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedD2H_Direct, then cut the last byte off expected
 *     blobs, so they are short of the rect the way blobs captured before
 *     rect-shaped D2H blobs are, and replay with HIP_HRR_D2H_EXACT=1.
 *   - With one blob short, that check is counted as skipped and the other ten
 *     pass.
 *   - With all eleven short, every check is skipped and the replay fails rather
 *     than passing as an archive with no validation blobs.
 *   - With the recorded host pitch of hipMemcpy3D set to SIZE_MAX, its host
 *     rect overflows size_t and that check is skipped while the other ten
 *     pass; with the other ten blobs short as well, every check is skipped and
 *     the replay fails.
 */
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
    const auto [ret, out] = exact_replay(cap.path);
    INFO("Playback stdout:\n" << out);
    CHECK(out.find(", 0 fail, " + std::to_string(kPitchedCopies) + " skipped") !=
          std::string::npos);
    CHECK(ret == 1);
  }
  {
    ScopedOverflowingPitch overflow(cap.path, arc);
    {
      const auto [ret, out] = exact_replay(cap.path);
      INFO("Playback stdout:\n" << out);
      int pass = 0, fail = 0;
      REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
      CHECK(pass == kPitchedCopies - 1);
      CHECK(out.find(", 0 fail, 1 skipped") != std::string::npos);
      CHECK(ret == 0);
    }
    std::vector<std::unique_ptr<ScopedBlobCut>> cuts;
    for (size_t i = 0; i < blobs.size(); ++i)
      if (i != 3) cuts.push_back(std::make_unique<ScopedBlobCut>(blobs[i]));
    const auto [ret, out] = exact_replay(cap.path);
    INFO("Playback stdout:\n" << out);
    CHECK(out.find(", 0 fail, " + std::to_string(kPitchedCopies) + " skipped") !=
          std::string::npos);
    CHECK(ret == 1);
  }
}

// ---------------------------------------------------------------------------
// A hipMemcpy2D device-to-host whose two host rows are a page apart, with the
// page between them PROT_NONE. The copy never touches that page, so capture
// must not read it either: a blob taken from the first byte to the last faults.
// ---------------------------------------------------------------------------
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

  // Row 0 ends page 0 and row 1 ends page 2, so the pitch spans the gap.
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

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedD2HGap_Direct, whose host destination has a
 *     PROT_NONE page between its two rows. Capture must finish, and the
 *     expected blob must hold the two rows and zero across the gap.
 *   - Replay with HIP_HRR_D2H_EXACT=1: the one check must pass.
 */
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
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::vector<uint8_t> want(2 * page + kWidth, 0);
    for (size_t y = 0; y < 2; ++y)
      for (size_t x = 0; x < kWidth; ++x)
        want[y * 2 * page + x] = dev_byte((kSrcY + y) * kDevPitch + kSrcX + x);
    CHECK(byte_diff(blob, want) == "");
  }
  require_replay(cap.path, 0, 1, 0);
#endif
}

// ---------------------------------------------------------------------------
// Dense host rects that do not start at the base pointer: a hipDrvMemcpy3D of
// one row at the kDst* offsets, and a hipMemcpy3D whose host rows and slices are
// packed (pitch == width, ysize == height) into slices 1 and 2 of its buffer.
// The copied bytes are one contiguous run, but it starts `first` bytes in, so
// the blob still spans the host buffer from its base with zero before the run.
// ---------------------------------------------------------------------------

namespace {

constexpr uint8_t kDenseFill[] = {0x71, 0x72};
constexpr int kDenseCopies = 2;

// hipDrvMemcpy3D: one row at the 3D destination offsets.
constexpr size_t kDenseRowFirst = kFirst3D;
constexpr size_t kDenseRowExtent = kDenseRowFirst + kWidth;

// hipMemcpy3D: kDepth packed slices from host slice 1.
constexpr size_t kPackedSlice = kWidth * kHeight;
constexpr size_t kPackedFirst = kPackedSlice;
constexpr size_t kPackedExtent = kPackedFirst + kDepth * kPackedSlice;

// The hipDrvMemcpy3D host buffer after its copy: `fill` except the one row.
std::vector<uint8_t> expected_dense_row(uint8_t fill) {
  std::vector<uint8_t> host(kHostBytes, fill);
  for (size_t x = 0; x < kWidth; ++x)
    host[kDenseRowFirst + x] =
        dev_byte(kSrcZ * kDevSlice + kSrcY * kDevPitch + kSrcX + x) ^ fill;
  return host;
}

// The hipMemcpy3D host buffer after its copy: `fill` in slice 0, then the
// window's rows packed end to end.
std::vector<uint8_t> expected_packed(uint8_t fill) {
  std::vector<uint8_t> host(kPackedExtent, fill);
  for (size_t z = 0; z < kDepth; ++z)
    for (size_t y = 0; y < kHeight; ++y)
      for (size_t x = 0; x < kWidth; ++x)
        host[kPackedFirst + z * kPackedSlice + y * kWidth + x] =
            dev_byte((kSrcZ + z) * kDevSlice + (kSrcY + y) * kDevPitch + kSrcX + x) ^ fill;
  return host;
}

// `host` with every byte before `first` zeroed: the blob capture records for a
// dense copy whose run starts at `first`, cut at `extent`.
std::vector<uint8_t> dense_blob(std::vector<uint8_t> host, size_t first, size_t extent) {
  host.resize(extent);
  std::fill(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(first), 0);
  return host;
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

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_DenseOffsetD2H_Direct. Each expected blob must span the
 *     host buffer from the pointer the copy was given through its last copied
 *     byte, with zero before the first copied byte rather than the buffer's
 *     fill, and the copied bytes after it.
 *   - Replay with HIP_HRR_D2H_EXACT=1: both checks must pass. Comparing the
 *     first width*height*depth bytes of each side compares the device bytes at
 *     the base of the allocation with the zeros, and fails.
 *   - Editing a blob byte before the first copied byte must not be reported;
 *     editing its last byte must fail exactly that check.
 */
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

  // One capture serves every edit, so the edits run in turn, not as sections.
  const std::pair<fs::path, std::pair<size_t, size_t>> edits[] = {
      {row, {kDenseRowFirst, kDenseRowExtent}}, {packed, {kPackedFirst, kPackedExtent}}};
  for (const auto& [blob, span] : edits) {
    INFO("Expected blob: " << blob.string());
    {
      ScopedBlobEdit before(blob, span.first - 1);
      require_replay(cap.path, 0, kDenseCopies, 0);
    }
    ScopedBlobEdit last(blob, span.second - 1);
    require_replay(cap.path, 1, kDenseCopies - 1, 1);
  }
}

// ---------------------------------------------------------------------------
// Host-to-device: hipMemcpy3D, hipMemcpy3DAsync and their _spt forms read the
// pitched host rect the copies above write (the kDst* offsets) into the device rect they read
// (the kSrc* offsets). A flat width*height*depth source blob read with the
// recorded pitch and position runs past its end.
// ---------------------------------------------------------------------------

namespace {

// One fill per device buffer, so the four readbacks record distinct blobs.
constexpr uint8_t kDevFill[] = {0x51, 0x52, 0x53, 0x54};
constexpr int kH2DCopies = 4;

uint8_t host_byte(size_t offset) { return static_cast<uint8_t>(dev_byte(offset) ^ 0xA5); }

// A device buffer filled with `fill` after one pitched H2D copy of the window.
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

// The source blob hash of the one H2D event of type `api`.
template <typename Args>
std::pair<uint64_t, uint64_t> h2d_hash(const hrr::Archive& arc, hrr_api_id_t api) {
  const auto args = event_args<Args>(arc, api);
  REQUIRE(args.size() == 1);
  return {args[0]->blob_hash_lo, args[0]->blob_hash_hi};
}

// The H2D source blob shared by the four 3D copies, which read the same rect.
fs::path h2d_blob(const hrr::Archive& arc) {
  const auto sync = h2d_hash<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D);
  REQUIRE(h2d_hash<hrr_args_hipMemcpy3DAsync>(arc, HRR_API_HIPMEMCPY3DASYNC) == sync);
  REQUIRE(h2d_hash<hrr_args_hipMemcpy3D_spt>(arc, HRR_API_HIPMEMCPY3D_SPT) == sync);
  REQUIRE(h2d_hash<hrr_args_hipMemcpy3DAsync_spt>(arc, HRR_API_HIPMEMCPY3DASYNC_SPT) == sync);
  return blob_path(arc, sync.first, sync.second);
}

void write_file(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  REQUIRE(f.good());
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

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedH2D_Direct: the H2D source blob must span the
 *     host rect from srcPtr.ptr through the last copied byte, holding the
 *     copied rows and zero in every byte around them.
 *   - Replay with HIP_HRR_D2H_EXACT=1: all four device readbacks must match.
 */
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
    REQUIRE(src.size() == kExtent3D);
    // Only the copied rows of the host source reach the archive.
    std::vector<uint8_t> host(kHostBytes);
    for (size_t i = 0; i < kHostBytes; ++i) host[i] = host_byte(i);
    CHECK(rows_only(src, host, 0, kFirst3D, kDepth));
  }
  require_replay(cap.path, 0, kH2DCopies, 0);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedH2D_Direct, then cut its H2D source blob to the
 *     first width*height*depth bytes, as an archive captured before footprint
 *     H2D blobs holds, and rewrite each readback's expected blob to the buffer's
 *     fill alone, which is what the device holds if the copy is skipped.
 *   - Replay must skip all four copies and pass all four readbacks. Reading the short
 *     blob with the recorded pitch and position runs past its end and writes
 *     other bytes into the device rect.
 */
HRR_TEST_CASE(Unit_HRR_PitchedH2DShortBlob) {
#ifdef _WIN32
  HRR_SKIP("pitched H2D HRR roundtrip is disabled on Windows");
#endif
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_pitched_h2d_short"};
  hrr_capture_direct("Unit_HRR_PitchedH2D_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));

  const fs::path src = h2d_blob(arc);
  std::vector<uint8_t> flat = read_file(src);
  REQUIRE(flat.size() == kExtent3D);
  flat.resize(kWidth * kHeight * kDepth);
  write_file(src, flat);

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

// ---------------------------------------------------------------------------
// Host-to-device: hipMemcpy2D and hipMemcpy2DAsync read the first slice of the
// same host window, from a pointer at its first byte, into the first slice of
// the device window. The source blob spans spitch * (height - 1) + width bytes.
// ---------------------------------------------------------------------------

namespace {

constexpr uint8_t kDev2DFill[] = {0x61, 0x62};
constexpr int kH2D2DCopies = 2;

// A device buffer filled with `fill` after one hipMemcpy2D of the window.
std::vector<uint8_t> expected_dev_2d(uint8_t fill) {
  std::vector<uint8_t> dev(kDevBytes, fill);
  for (size_t y = 0; y < kHeight; ++y)
    for (size_t x = 0; x < kWidth; ++x)
      dev[(kSrcY + y) * kDevPitch + kSrcX + x] = host_byte((kDstY + y) * kHostPitch + kDstX + x);
  return dev;
}

// The H2D source blob shared by the two 2D copies, which read the same rect.
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
    // See Unit_HRR_PitchedH2D_Direct: the fill must land before the async copy.
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

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedH2D2D_Direct: the H2D source blob must span the
 *     host rect from src through the last copied byte, holding the copied rows
 *     and zero in every byte around them.
 *   - Replay with HIP_HRR_D2H_EXACT=1: both device readbacks must match.
 */
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
    REQUIRE(src.size() == kExtent2D);
    std::vector<uint8_t> host(kHostBytes);
    for (size_t i = 0; i < kHostBytes; ++i) host[i] = host_byte(i);
    CHECK(rows_only(src, host, kDstY * kHostPitch + kDstX, 0, 1));
  }
  require_replay(cap.path, 0, kH2D2DCopies, 0);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedH2D2D_Direct, cut the last byte off its H2D
 *     source blob, and rewrite each readback's expected blob to the buffer's
 *     fill alone, which is what the device holds if the copy is skipped.
 *   - Replay must skip both copies and pass both readbacks. Handing the short
 *     blob to hipMemcpy2D reads past its end and writes the window.
 */
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
  REQUIRE(blob.size() == kExtent2D);
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

// ---------------------------------------------------------------------------
// Rejected copies: the runtime refuses a null device side before it touches
// the host rect, so capture must not read that rect either.
// ---------------------------------------------------------------------------
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

// Every event of type `api` must be a failed call with no blob of either kind.
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

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_RejectedMemcpy3D_Direct: each of the four hipMemcpy3D
 *     spellings is refused once host-to-device and once device-to-host.
 *   - A refused call is either not recorded at all or recorded with its error
 *     and both blob hashes zero: its host rect was never validated, and reading
 *     it can run past the caller's buffer.
 *   - Replay must pass with nothing to substitute or validate.
 */
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
