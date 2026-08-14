/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR external region annotations
 * @{
 * @ingroup HRRTest
 * Tests for the region annotation channel (clr/hipamd/src/hrr/hrr_regions.h).
 *
 * HRR only observes memory that crosses a HIP API, so the per-object blocks a
 * framework allocator carves out of one large hipMalloc are invisible to it and
 * an intra-segment overrun reproduces at replay without being detectable. A
 * producer outside the runtime writes those bounds into a `.hrrr` sidecar under
 * the archive's `regions` directory; these tests are that producer, written
 * synthetically so the behaviour can be asserted without PyTorch.
 *
 * Two groups:
 *
 *   CPU-only  the sidecar framing itself — a well-formed stream reads back
 *             record for record, a torn tail costs only the torn batch, and a
 *             stream carrying the wrong magic is refused.
 *   GPU       a workload that carves blocks out of a segment by pointer
 *             arithmetic exactly as a caching allocator does, replayed against a
 *             synthetic sidecar. The contrast is the point: the same archive
 *             replays clean with a stale pointer merely reported, and stops hard
 *             once --guard-blocks makes the overrun fault.
 */

#include <hip_test_common.hh>

#include "hrr_regions.h"
#include "hrr_test_common.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Workload — a caching allocator in miniature
// ---------------------------------------------------------------------------

namespace {
constexpr size_t kSegBytes   = 1u << 20;  // one "segment", from a single hipMalloc
constexpr size_t kBlockBytes = 4096;      // one "block" carved out of it
constexpr size_t kBlock1Off  = 64 * 1024; // second block's offset in the segment
constexpr size_t kGapOff     = 32 * 1024; // inside the segment, inside no block

constexpr int kFloatsPerBlock = static_cast<int>(kBlockBytes / sizeof(float));
// Far enough past the block's end to clear the guard page under --guard-blocks,
// while staying inside the segment so an unguarded replay is silent.
constexpr int kOverrunFloats  = kFloatsPerBlock + 16384;

// A second, small allocation. The workload also launches a kernel with a
// pointer well past its end, which stands in for an allocation HRR never
// observed: at replay it resolves in no map, exactly like a buffer that came
// from a direct HSA call. The kernel is launched with n == 0 so nothing is
// dereferenced and the capture itself is harmless.
constexpr size_t kProbeBytes = 64 * 1024;
constexpr size_t kProbeFarOff = 1u << 20;  // past the end of the probe alloc
}  // namespace

__global__ void hrr_regions_fill(float* p, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

// ===========================================================================
// The captured workload. One hipMalloc, three launches at hand-computed offsets
// inside it — which is all a framework allocator's block layout is, as far as
// the HIP dispatch table can tell.
// ===========================================================================
TEST_CASE("Unit_HRR_Regions_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  char* seg = nullptr;
  HIP_CHECK(hipMalloc(&seg, kSegBytes));
  HIP_CHECK(hipMemset(seg, 0, kSegBytes));

  float* block0 = reinterpret_cast<float*>(seg);
  float* block1 = reinterpret_cast<float*>(seg + kBlock1Off);
  float* gap    = reinterpret_cast<float*>(seg + kGapOff);

  const int threads = 256;
  auto grid = [&](int n) { return dim3((n + threads - 1) / threads); };

  // In bounds. The ordinary case, and the control for everything below.
  hipLaunchKernelGGL(hrr_regions_fill, grid(kFloatsPerBlock), dim3(threads), 0,
                     nullptr, block0, kFloatsPerBlock, 1.0f);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(hrr_regions_fill, grid(kFloatsPerBlock), dim3(threads), 0,
                     nullptr, block1, kFloatsPerBlock, 2.0f);
  HIP_CHECK(hipGetLastError());

  // A pointer into the segment that belongs to no block: what a stale pointer
  // to a freed tensor looks like. In bounds as far as HIP is concerned.
  hipLaunchKernelGGL(hrr_regions_fill, grid(16), dim3(threads), 0, nullptr,
                     gap, 16, 3.0f);
  HIP_CHECK(hipGetLastError());

  // Past the end of block0, into whatever the allocator put next. Silent here
  // and silent at replay, because the segment is one contiguous allocation.
  hipLaunchKernelGGL(hrr_regions_fill, grid(kOverrunFloats), dim3(threads), 0,
                     nullptr, block0, kOverrunFloats, 4.0f);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // A pointer no allocation covers. At replay this resolves in no map, which is
  // what a buffer allocated below the HIP API looks like from the archive's
  // side. n == 0, so no thread dereferences it here or there.
  char* probe = nullptr;
  HIP_CHECK(hipMalloc(&probe, kProbeBytes));
  hipLaunchKernelGGL(hrr_regions_fill, dim3(1), dim3(threads), 0, nullptr,
                     reinterpret_cast<float*>(probe + kProbeFarOff), 0, 5.0f);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> host(kFloatsPerBlock);
  HIP_CHECK(hipMemcpy(host.data(), block0, kBlockBytes, hipMemcpyDeviceToHost));
  REQUIRE(host[0] == 4.0f);

  HIP_CHECK(hipFree(probe));
  HIP_CHECK(hipFree(seg));
}

// ---------------------------------------------------------------------------
// Synthetic producer
// ---------------------------------------------------------------------------

namespace {

// One batch, built and written exactly as producers/README.md specifies: whole
// batch in memory, one write, so a crash can only ever tear the tail.
void write_region_batch(const fs::path& file,
                        const std::vector<hrr_region_rec>& recs,
                        bool new_stream = true) {
  fs::create_directories(file.parent_path());
  FILE* f = fopen(file.string().c_str(), new_stream ? "wb" : "ab");
  REQUIRE(f != nullptr);
  if (new_stream) {
    const hrr_file_header fh = hrr_make_region_file_header();
    REQUIRE(fwrite(&fh, sizeof(fh), 1, f) == 1);
  }
  std::vector<uint8_t> buf(hrr_region_batch_bytes(
      static_cast<uint32_t>(recs.size())));
  const hrr_region_batch b =
      hrr_make_region_batch(static_cast<uint32_t>(recs.size()), 0);
  memcpy(buf.data(), &b, sizeof(b));
  if (!recs.empty())
    memcpy(buf.data() + sizeof(b), recs.data(),
           recs.size() * sizeof(hrr_region_rec));
  REQUIRE(fwrite(buf.data(), 1, buf.size(), f) == buf.size());
  fclose(f);
}

// mono_ns == 0: live since before the stream. That is how a producer declares
// the layout that already existed when it started watching, and it is all these
// tests need — the workload's blocks exist for its whole life.
hrr_region_rec region_rec(uint8_t op, uint8_t kind, uint64_t base, uint64_t size) {
  hrr_region_rec r{};
  r.op = op;
  r.kind = kind;
  r.base = base;
  r.size = size;
  return r;
}

// The recorded base of one of the workload's allocations, read back out of the
// archive. Passing it from the child would need an IPC channel; the archive
// already records every hipMalloc it made.
uint64_t find_alloc_base(const fs::path& archive_path, uint64_t want_size) {
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive_path.string(), arc));
  for (const auto& ev : arc.events) {
    if (ev.header().event_type == HRR_API_HIPMALLOC &&
        ev.malloc_ev.size == want_size)
      return ev.malloc_ev.ptr_handle;
  }
  return 0;
}

// True if any recorded allocation covers `addr`. The bypass test needs an
// address the replay genuinely cannot translate, and the driver is free to
// place the workload's two allocations wherever it likes.
bool covered_by_recorded_alloc(const fs::path& archive_path, uint64_t addr) {
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive_path.string(), arc));
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPMALLOC) continue;
    const uint64_t base = ev.malloc_ev.ptr_handle;
    if (addr >= base && addr < base + ev.malloc_ev.size) return true;
  }
  return false;
}

}  // namespace

// ===========================================================================
// CPU-only: the sidecar framing
// ===========================================================================

TEST_CASE("Unit_HRR_Regions_StreamFraming", "[hrr]") {
  ScopedDir dir(fs::temp_directory_path() / "hrr_region_framing");
  const fs::path archive = dir.path / "pid-1";
  const fs::path stream = archive / "regions" / "synthetic.hrrr";

  write_region_batch(stream, {region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT,
                                         0x700000000000ull, kSegBytes)});
  write_region_batch(stream,
                     {region_rec(HRR_REGION_ADD, HRR_REGION_BLOCK,
                                 0x700000000000ull, kBlockBytes),
                      region_rec(HRR_REGION_ADD, HRR_REGION_BLOCK,
                                 0x700000010000ull, kBlockBytes)},
                     /*new_stream=*/false);

  SECTION("discovery finds the sidecar") {
    const auto found = hrr::find_region_streams(archive.string());
    REQUIRE(found.size() == 1);
    CHECK(found[0] == stream.string());
    // An archive with no producer is the normal case and must not be an error.
    CHECK(hrr::find_region_streams((dir.path / "pid-2").string()).empty());
  }

  SECTION("records read back in order") {
    uint16_t version = 0;
    FILE* f = hrr::open_record_stream(stream.string(), HRR_REGION_MAGIC,
                                      HRR_REGION_VERSION, &version);
    REQUIRE(f != nullptr);
    CHECK(version == HRR_REGION_VERSION);

    std::vector<hrr_region_rec> all;
    std::vector<uint8_t> raw;
    while (hrr::read_raw_record(f, raw) == hrr::RecordStatus::Ok) {
      const auto* b = reinterpret_cast<const hrr_region_batch*>(raw.data());
      REQUIRE(b->hdr.event_type == HRR_REGION_EVENT);
      REQUIRE(raw.size() == hrr_region_batch_bytes(b->n));
      const auto* recs = reinterpret_cast<const hrr_region_rec*>(
          raw.data() + sizeof(hrr_region_batch));
      all.insert(all.end(), recs, recs + b->n);
    }
    fclose(f);

    REQUIRE(all.size() == 3);
    CHECK(all[0].kind == HRR_REGION_SEGMENT);
    CHECK(all[0].size == kSegBytes);
    CHECK(all[1].kind == HRR_REGION_BLOCK);
    CHECK(all[2].base == 0x700000010000ull);
  }

  SECTION("a torn tail costs only the torn batch") {
    // A producer killed mid-write. Everything it finished writing must survive,
    // because the process being recorded is usually one that crashes.
    const auto full = fs::file_size(stream);
    fs::resize_file(stream, full - 20);

    FILE* f = hrr::open_record_stream(stream.string(), HRR_REGION_MAGIC,
                                      HRR_REGION_VERSION, nullptr);
    REQUIRE(f != nullptr);
    size_t complete = 0;
    hrr::RecordStatus st;
    std::vector<uint8_t> raw;
    while ((st = hrr::read_raw_record(f, raw)) == hrr::RecordStatus::Ok)
      ++complete;
    fclose(f);
    CHECK(st == hrr::RecordStatus::Torn);
    CHECK(complete == 1);
  }

  SECTION("an events.bin is not a region stream") {
    // The magic is what keeps the two apart; without the check, a mis-set path
    // would parse event payloads as region records.
    CHECK(hrr::open_record_stream(stream.string(), HRR_MAGIC, HRR_VERSION,
                                  nullptr) == nullptr);
  }
}

// ===========================================================================
// GPU: replay against a synthetic sidecar
// ===========================================================================

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {

// Replace whatever sidecar is in the archive with these records.
void install_sidecar(const fs::path& archive_path,
                     const std::vector<hrr_region_rec>& recs) {
  const fs::path regions = archive_path / "regions";
  fs::remove_all(regions);
  write_region_batch(regions / "synthetic.hrrr", recs);
}

}  // namespace

// ---------------------------------------------------------------------------
// The layout HRR could not see, supplied after the fact, changes what replay
// can say about the same archive.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_Regions_Roundtrip", "[hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_regions_roundtrip.hrr");
  hrr_capture_direct("Unit_HRR_Regions_Direct", cap.path, /*min_events=*/5);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const uint64_t seg = find_alloc_base(archive, kSegBytes);
  INFO("Recorded segment base: 0x" << std::hex << seg);
  REQUIRE(seg != 0);

  // Without a sidecar the archive replays clean and says nothing about layout:
  // the overrun and the stale pointer are both in bounds of one allocation.
  {
    auto [rc, out] = hrr_playback_merged(archive);
    INFO("Baseline replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find("Regions ") == std::string::npos);
  }

  install_sidecar(archive, {
      region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT, seg, kSegBytes),
      region_rec(HRR_REGION_ADD, HRR_REGION_BLOCK, seg, kBlockBytes),
      region_rec(HRR_REGION_ADD, HRR_REGION_BLOCK, seg + kBlock1Off, kBlockBytes),
  });

  SECTION("the stale pointer is reported, and nothing moves") {
    auto [rc, out] = hrr_playback_merged(archive);
    INFO("Annotated replay:\n" << out);
    // Reporting only: the finding describes the recorded program, so it must
    // not by itself fail the replay.
    CHECK(rc == 0);
    CHECK(out.find("region OOB") != std::string::npos);
    CHECK(out.find("intra-segment out-of-bounds/stale") != std::string::npos);
    // The gap launch is the one pointer of the four that is in no block.
    CHECK(out.find(": 0 intra-segment") == std::string::npos);
  }

  SECTION("--regions-strict turns the finding into a failure") {
    auto [rc, out] = hrr_playback_merged(archive, "--regions-strict");
    INFO("Strict replay:\n" << out);
    CHECK(rc != 0);
  }

  SECTION("--no-regions ignores the sidecar entirely") {
    auto [rc, out] = hrr_playback_merged(archive, "--no-regions");
    INFO("Suppressed replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find("region OOB") == std::string::npos);
  }

  SECTION("--guard-blocks makes the overrun fault") {
    auto [rc, out] = hrr_playback_merged(archive, "--guard-blocks");
    INFO("Guarded replay:\n" << out);
    if (rc == 0) {
      // The only acceptable clean run is one where nothing was guarded — no VMM
      // support, or every reservation failed. If blocks were relocated and the
      // overrun still did not fault, the guard is not doing its job.
      INFO("Guard reported a clean run; it must not have relocated anything");
      CHECK(out.find(": 0 block relocation(s)") != std::string::npos);
    }
  }
}

// ---------------------------------------------------------------------------
// The bypass case: a kernel argument that resolves in no map at all, because
// the allocation behind it never crossed a HIP API. Without an annotation it
// reaches the GPU as an address from the recorded process; with one, replay
// backs the segment and the pointer translates.
//
// A producer declares its segments uniformly, without knowing which ones HIP
// saw, so this also checks the other half: declaring a segment the archive
// already has must not allocate anything, or the captured layout the replay is
// reproducing would be shadowed by a second buffer.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_Regions_SegmentMaterialization", "[hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_regions_segments.hrr");
  hrr_capture_direct("Unit_HRR_Regions_Direct", cap.path, /*min_events=*/5);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const uint64_t seg   = find_alloc_base(archive, kSegBytes);
  const uint64_t probe = find_alloc_base(archive, kProbeBytes);
  REQUIRE(seg != 0);
  REQUIRE(probe != 0);

  // The address the workload passed that nothing allocated.
  const uint64_t bypassed = probe + kProbeFarOff;
  if (covered_by_recorded_alloc(archive, bypassed)) {
    WARN("The driver placed an allocation over the probe address; "
         "skipping the bypass assertions for this run");
    return;
  }

  SECTION("without an annotation the pointer is untranslatable") {
    auto [rc, out] = hrr_playback_merged(archive, "--warn-untranslated-args");
    INFO("Unannotated replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find("Untranslated   : 0 ") == std::string::npos);
    CHECK(out.find("is in no known allocation") != std::string::npos);
  }

  SECTION("a SEGMENT annotation makes it resolve") {
    install_sidecar(archive, {
        // One segment the archive already has, and one it does not.
        region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT, seg, kSegBytes),
        region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT, bypassed - 4096,
                   64 * 1024),
    });

    auto [rc, out] = hrr_playback_merged(archive, "--warn-untranslated-args");
    INFO("Annotated replay:\n" << out);
    CHECK(rc == 0);
    // Exactly one segment backed: the bypassed one. Declaring the captured
    // segment must not have allocated a second buffer for it.
    CHECK(out.find("Region segs    : 1 materialised") != std::string::npos);
    CHECK(out.find("Untranslated   : 0 ") != std::string::npos);
  }
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
