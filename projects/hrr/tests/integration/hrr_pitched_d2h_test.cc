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
 * hipMemcpy2D and hipMemcpy2DAsync.
 *
 * Every copy reads a window at a non-zero offset of a pitched device buffer
 * into a window at a different offset of a host buffer with a different pitch,
 * so the copied rows are neither dense nor at the start of either buffer.
 * Replay has to compare exactly those rows. The first width*height*depth bytes
 * of either buffer are mostly bytes the copy never touched, which fails a
 * correct replay, and they miss the later rows, which passes a wrong one.
 */

#include "hrr_test_common.hh"

#include "hrr_reader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

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

// One fill byte per host buffer, so every expected blob is distinct and a test
// can edit one without touching the others.
constexpr uint8_t kFill[] = {0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37};
constexpr int kPitchedCopies = static_cast<int>(sizeof(kFill));

// Device contents: a byte hash of the offset, with no period that lines up
// with a row or a slice, so reading the wrong offset or pitch reads other bytes.
uint8_t dev_byte(size_t offset) { return static_cast<uint8_t>((offset * 2654435761u) >> 24); }

// A host buffer after copying `depth` slices of the window into host slice
// `dst_z` onwards: `fill` everywhere except the copied rows.
std::vector<uint8_t> expected_host(uint8_t fill, size_t depth, size_t dst_z) {
  std::vector<uint8_t> host(kHostBytes, fill);
  for (size_t z = 0; z < depth; ++z)
    for (size_t y = 0; y < kHeight; ++y)
      for (size_t x = 0; x < kWidth; ++x)
        host[(dst_z + z) * kHostSlice + (kDstY + y) * kHostPitch + kDstX + x] =
            dev_byte((kSrcZ + z) * kDevSlice + (kSrcY + y) * kDevPitch + kSrcX + x);
  return host;
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
// Workload: the seven pitched copies, each into its own host buffer, each
// checked here against the layout the replay assertions assume.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_PitchedD2H_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  std::vector<uint8_t> image(kDevBytes);
  for (size_t i = 0; i < kDevBytes; ++i) image[i] = dev_byte(i);
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kDevBytes));
  HRR_HIP_CHECK(hipMemcpy(dev, image.data(), kDevBytes, hipMemcpyHostToDevice));
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

  std::vector<uint8_t> h0(kHostBytes, kFill[0]);
  drv3d.dstHost = h0.data();
  HRR_HIP_CHECK(hipDrvMemcpy3D(&drv3d));
  REQUIRE(h0 == expected_host(kFill[0], kDepth, kDstZ));

  std::vector<uint8_t> h1(kHostBytes, kFill[1]);
  drv3d.dstHost = h1.data();
  HRR_HIP_CHECK(hipDrvMemcpy3DAsync(&drv3d, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  REQUIRE(h1 == expected_host(kFill[1], kDepth, kDstZ));

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

  std::vector<uint8_t> h2(kHostBytes, kFill[2]);
  drv2d.dstHost = h2.data();
  HRR_HIP_CHECK(hipDrvMemcpy2DUnaligned(&drv2d));
  REQUIRE(h2 == expected_host(kFill[2], 1, 0));

  hipMemcpy3DParms p3d{};
  p3d.srcPtr = make_hipPitchedPtr(dev, kDevPitch, kDevPitch, kDevRows);
  p3d.srcPos = make_hipPos(kSrcX, kSrcY, kSrcZ);
  p3d.dstPos = make_hipPos(kDstX, kDstY, kDstZ);
  p3d.extent = make_hipExtent(kWidth, kHeight, kDepth);
  p3d.kind = hipMemcpyDeviceToHost;

  std::vector<uint8_t> h3(kHostBytes, kFill[3]);
  p3d.dstPtr = make_hipPitchedPtr(h3.data(), kHostPitch, kHostPitch, kHostRows);
  HRR_HIP_CHECK(hipMemcpy3D(&p3d));
  REQUIRE(h3 == expected_host(kFill[3], kDepth, kDstZ));

  std::vector<uint8_t> h4(kHostBytes, kFill[4]);
  p3d.dstPtr = make_hipPitchedPtr(h4.data(), kHostPitch, kHostPitch, kHostRows);
  HRR_HIP_CHECK(hipMemcpy3DAsync(&p3d, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  REQUIRE(h4 == expected_host(kFill[4], kDepth, kDstZ));

  // hipMemcpy2D takes no offsets: the window is wherever the pointers point.
  const void* src2d =
      static_cast<const uint8_t*>(dev) + kSrcZ * kDevSlice + kSrcY * kDevPitch + kSrcX;
  const size_t dst2d = kDstY * kHostPitch + kDstX;

  std::vector<uint8_t> h5(kHostBytes, kFill[5]);
  HRR_HIP_CHECK(hipMemcpy2D(h5.data() + dst2d, kHostPitch, src2d, kDevPitch, kWidth, kHeight,
                            hipMemcpyDeviceToHost));
  REQUIRE(h5 == expected_host(kFill[5], 1, 0));

  std::vector<uint8_t> h6(kHostBytes, kFill[6]);
  HRR_HIP_CHECK(hipMemcpy2DAsync(h6.data() + dst2d, kHostPitch, src2d, kDevPitch, kWidth,
                                 kHeight, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  REQUIRE(h6 == expected_host(kFill[6], 1, 0));

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(dev));
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_PitchedD2H_Direct and check that each of the seven
 *     pitched copies recorded an expected-output blob.
 *   - Replay with HIP_HRR_D2H_EXACT=1: all seven checks must pass. Comparing
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
    d2h_blob<hrr_args_hipDrvMemcpy3D>(arc, HRR_API_HIPDRVMEMCPY3D);
    d2h_blob<hrr_args_hipDrvMemcpy3DAsync>(arc, HRR_API_HIPDRVMEMCPY3DASYNC);
    d2h_blob<hrr_args_hipDrvMemcpy2DUnaligned>(arc, HRR_API_HIPDRVMEMCPY2DUNALIGNED);
    d2h_blob<hrr_args_hipMemcpy3D>(arc, HRR_API_HIPMEMCPY3D);
    d2h_blob<hrr_args_hipMemcpy3DAsync>(arc, HRR_API_HIPMEMCPY3DASYNC);
    d2h_blob<hrr_args_hipMemcpy2D>(arc, HRR_API_HIPMEMCPY2D);
    d2h_blob<hrr_args_hipMemcpy2DAsync>(arc, HRR_API_HIPMEMCPY2DASYNC);
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
 * @}
 */
