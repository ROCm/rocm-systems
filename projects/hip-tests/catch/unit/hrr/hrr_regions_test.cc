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
#include <hip_test_process.hh>

#include "hrr_regions.h"
#include "hrr_reader.h"

#if defined(HRR_HAVE_HSA)
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Self-contained capture/replay harness.
//
// The shared hrr_test_common.h harness (ScopedDir, hrr_capture_direct,
// hrr_playback_merged, ...) is introduced by the API-coverage work; this test
// lands ahead of it, so it carries the small subset it needs. When the shared
// header lands, these fall away in favour of the include.
// ---------------------------------------------------------------------------
namespace {

#ifdef _WIN32
constexpr char kPathSep = ';';
#else
constexpr char kPathSep = ':';
#endif

// Make the ROCm runtime binaries findable from the spawned subprocess.
inline void set_proc_search_path(hip::SpawnProc& proc) {
  const char* cur_path = getenv("PATH");
  proc.setEnv("PATH",
              std::string(ROCM_BIN_PATH) + kPathSep + (cur_path ? cur_path : ""));
}

// RAII guard: removes a directory tree on scope exit (even on REQUIRE failure).
struct ScopedDir {
  fs::path path;
  explicit ScopedDir(fs::path p) : path(std::move(p)) { fs::remove_all(path); }
  ~ScopedDir() { fs::remove_all(path); }
};

// The single per-process archive under `root`, or `root` itself if the capture
// wrote a flat archive.
inline fs::path hrr_single_process_archive(const fs::path& root) {
  if (fs::exists(root / "events.bin")) return root;
  std::vector<fs::path> archives;
  for (const auto& ent : fs::directory_iterator(root)) {
    if (!ent.is_directory()) continue;
    const std::string name = ent.path().filename().string();
    if (name.rfind("pid-", 0) == 0 && fs::exists(ent.path() / "events.bin"))
      archives.push_back(ent.path());
  }
  INFO("Process archive count: " << archives.size());
  REQUIRE(archives.size() == 1);
  return archives.front();
}

// SpawnProc uses execvp on Linux (no shell), so quotes around the archive path
// would be taken literally. On Windows CreateProcess needs them for spaces.
inline std::string hrr_quote_path(const fs::path& p) {
#ifdef _WIN32
  return "\"" + p.string() + "\"";
#else
  return p.string();
#endif
}

}  // namespace

#if defined(HRR_TEST_EXE)
namespace {
// Spawn a hidden _Direct workload with HIP_HRR_CAPTURE_OUTPUT set, REQUIRE a
// clean capture, and assert the archive has >= min_events.
inline void hrr_capture_direct(const std::string& direct_case,
                               const fs::path& cap_path,
                               size_t min_events = 5) {
  { hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"" + direct_case + "\"");
    INFO("Capture exit: " << ret); REQUIRE(ret == 0); }
  fs::path archive_path = hrr_single_process_archive(cap_path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  hrr::Archive arc;
  bool arc_ok = hrr::load_archive(cap_path.string(), arc);
  INFO("Archive event count: " << arc.events.size());
  REQUIRE(arc_ok);
  REQUIRE(arc.events.size() >= min_events);
}
}  // namespace
#endif  // HRR_TEST_EXE

#if defined(HRR_PLAYBACK_EXE)
namespace {
// Run hrr-playback with stderr merged into the captured output. Region and
// guard diagnostics are written to stderr, so a test asserting on them must
// merge the two streams.
inline std::pair<int, std::string> hrr_playback_merged(
    const fs::path& cap_path,
    const std::string& extra_args = "",
    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true,
                      /*capture_stderr=*/true);
  set_proc_search_path(proc);
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  return {ret, proc.getOutput()};
}
}  // namespace
#endif  // HRR_PLAYBACK_EXE

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

// A second probe, for the same bypassed-allocation case reached through a
// pointer embedded in a by-value struct instead of passed whole. Sized apart
// from the first so find_alloc_base can tell the two allocations apart.
constexpr size_t kProbe2Bytes = 32 * 1024;
}  // namespace

__global__ void hrr_regions_fill(float* p, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

// Two device pointers inside one by-value argument, which is how a framework
// hands a kernel its operands (ATen's TensorListMetadata, an OffsetCalculator,
// any functor holding tensor bases). `known` resolves to a captured
// allocation and is what makes the capture side tag the argument as carrying
// embedded pointers at all; `bypassed` resolves to nothing, standing in for
// memory allocated below the HIP API.
struct HrrRegionsPtrPair {
  float* known;
  float* bypassed;
  int    n;
};

__global__ void hrr_regions_fill_pair(HrrRegionsPtrPair a, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.n) {
    a.known[i]    = v;
    a.bypassed[i] = v;
  }
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

  // The same unresolvable pointer, this time as a field of a by-value struct.
  // Pairing it with a pointer that does resolve is what gets the argument
  // recorded as one carrying embedded pointers. n == 0 again.
  char* probe2 = nullptr;
  HIP_CHECK(hipMalloc(&probe2, kProbe2Bytes));
  HrrRegionsPtrPair pair{block0,
                         reinterpret_cast<float*>(probe2 + kProbeFarOff), 0};
  hipLaunchKernelGGL(hrr_regions_fill_pair, dim3(1), dim3(threads), 0, nullptr,
                     pair, 6.0f);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> host(kFloatsPerBlock);
  HIP_CHECK(hipMemcpy(host.data(), block0, kBlockBytes, hipMemcpyDeviceToHost));
  REQUIRE(host[0] == 4.0f);

  HIP_CHECK(hipFree(probe2));
  HIP_CHECK(hipFree(probe));
  HIP_CHECK(hipFree(seg));
}

// ===========================================================================
// A memcpy whose destination HIP never saw.
//
// Every other bypassed-allocation case in this file fakes the address: the
// workload passes `probe + kProbeFarOff`, which nothing allocated, and no
// thread dereferences it. That trick cannot be reused for a memcpy, because a
// memcpy has to *succeed at capture* for the event to be recorded at all. The
// memory therefore has to genuinely exist while genuinely being invisible to
// HRR, which means a real allocation made below the HIP API.
//
// HSA is that allocator. A coarse-grained device pool gives a buffer the GPU
// can reach, allocated through a path the HIP dispatch table never sees, so
// nothing in the archive can explain the address — exactly the property the
// assertion needs.
// ===========================================================================

#if defined(HRR_HAVE_HSA)
namespace {

constexpr size_t kHsaSegBytes = 256 * 1024;

// The coarse-grained global pool of the first GPU agent, plus every agent in
// the system. Access has to be granted to all of them, not just the owning GPU:
// the copy HIP issues is serviced by a DMA engine, and on an unshared coarse
// grained allocation that engine faults exactly like a stray kernel would.
struct HsaTarget {
  hsa_agent_t gpu{};
  hsa_amd_memory_pool_t pool{};
  std::vector<hsa_agent_t> all_agents;
  bool ok = false;
};

hsa_status_t hrr_find_pool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment{};
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                   &segment) != HSA_STATUS_SUCCESS ||
      segment != HSA_AMD_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;

  uint32_t flags = 0;
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                   &flags) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;

  bool alloc_allowed = false;
  if (hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
          &alloc_allowed) != HSA_STATUS_SUCCESS ||
      !alloc_allowed)
    return HSA_STATUS_SUCCESS;

  // Coarse-grained: device-local, not host-coherent. What hipMalloc would give
  // us, had it been asked.
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hrr_visit_agent(hsa_agent_t agent, void* data) {
  auto* target = static_cast<HsaTarget*>(data);
  target->all_agents.push_back(agent);

  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) !=
          HSA_STATUS_SUCCESS ||
      type != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;
  if (target->ok) return HSA_STATUS_SUCCESS;  // keep collecting agents

  hsa_amd_memory_pool_t pool{};
  hsa_amd_agent_iterate_memory_pools(agent, hrr_find_pool, &pool);
  if (pool.handle == 0) return HSA_STATUS_SUCCESS;

  target->gpu  = agent;
  target->pool = pool;
  target->ok   = true;
  return HSA_STATUS_SUCCESS;
}

// hsa_init is reference counted and HIP has already initialised HSA in this
// process, so this attaches to the running runtime rather than starting one.
HsaTarget hrr_hsa_target() {
  HsaTarget target;
  if (hsa_init() != HSA_STATUS_SUCCESS) return target;
  hsa_iterate_agents(hrr_visit_agent, &target);
  return target;
}

}  // namespace

// The marker the parent parses out of the child's stdout. An HSA address
// appears in no hipMalloc record, so find_alloc_base cannot recover it and the
// child has to say where it put the buffer.
#define HRR_HSA_SEG_MARKER "HRR_HSA_SEG"

TEST_CASE("Unit_HRR_Regions_MemcpyDirect", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  HsaTarget target = hrr_hsa_target();
  if (!target.ok) {
    printf("%s none 0\n", HRR_HSA_SEG_MARKER);
    fflush(stdout);
    return;
  }

  void* seg = nullptr;
  if (hsa_amd_memory_pool_allocate(target.pool, kHsaSegBytes, 0, &seg) !=
          HSA_STATUS_SUCCESS ||
      seg == nullptr) {
    printf("%s none 0\n", HRR_HSA_SEG_MARKER);
    fflush(stdout);
    return;
  }
  // Without this the copy faults at capture instead of being recorded, and the
  // test would then be measuring a broken workload rather than the replay.
  REQUIRE(hsa_amd_agents_allow_access(
              static_cast<uint32_t>(target.all_agents.size()),
              target.all_agents.data(), nullptr, seg) == HSA_STATUS_SUCCESS);

  printf("%s 0x%llx %zu\n", HRR_HSA_SEG_MARKER,
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(seg)),
         kHsaSegBytes);
  fflush(stdout);

  std::vector<uint8_t> host(kHsaSegBytes);
  for (size_t i = 0; i < host.size(); ++i)
    host[i] = static_cast<uint8_t>(i * 7 + 1);

  // The first and only touch of `seg`. Deliberately not passed to any kernel:
  // a kernel argument would let the existing whole-pointer materialisation
  // resolve it, and the test would then pass with or without the memcpy fix.
  HIP_CHECK(hipMemcpy(seg, host.data(), kHsaSegBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());

  // A companion hipMalloc plus a D2H copy back, so the roundtrip has something
  // the replay can compare byte for byte.
  char* known = nullptr;
  HIP_CHECK(hipMalloc(&known, kHsaSegBytes));
  HIP_CHECK(hipMemcpy(known, seg, kHsaSegBytes, hipMemcpyDeviceToDevice));
  std::vector<uint8_t> back(kHsaSegBytes, 0);
  HIP_CHECK(hipMemcpy(back.data(), known, kHsaSegBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(back[0] == host[0]);
  REQUIRE(back[kHsaSegBytes - 1] == host[kHsaSegBytes - 1]);

  HIP_CHECK(hipFree(known));
  REQUIRE(hsa_amd_memory_pool_free(seg) == HSA_STATUS_SUCCESS);
}
#endif  // HRR_HAVE_HSA

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

TEST_CASE("Unit_HRR_Regions_StreamFraming", "[.][hrr]") {
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
    CHECK(all[0].op == HRR_REGION_ADD);
    CHECK(all[0].kind == HRR_REGION_SEGMENT);
    CHECK(all[0].base == 0x700000000000ull);
    CHECK(all[0].size == kSegBytes);
    CHECK(all[1].op == HRR_REGION_ADD);
    CHECK(all[1].kind == HRR_REGION_BLOCK);
    CHECK(all[1].base == 0x700000000000ull);
    CHECK(all[1].size == kBlockBytes);
    CHECK(all[2].op == HRR_REGION_ADD);
    CHECK(all[2].kind == HRR_REGION_BLOCK);
    CHECK(all[2].base == 0x700000010000ull);
    CHECK(all[2].size == kBlockBytes);
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

  SECTION("an implausible payload_length is torn, not allocated") {
    const fs::path bad = dir.path / "pid-huge" / "regions" / "huge.hrrr";
    fs::create_directories(bad.parent_path());
    FILE* wf = fopen(bad.string().c_str(), "wb");
    REQUIRE(wf != nullptr);
    const hrr_file_header fh = hrr_make_region_file_header();
    REQUIRE(fwrite(&fh, sizeof(fh), 1, wf) == 1);
    hrr_event_header hdr{};
    hdr.event_type = HRR_REGION_EVENT;
    hdr.payload_length = 0xFFFFFFF0u;
    REQUIRE(fwrite(&hdr, sizeof(hdr), 1, wf) == 1);
    fclose(wf);

    FILE* f = hrr::open_record_stream(bad.string(), HRR_REGION_MAGIC,
                                      HRR_REGION_VERSION, nullptr);
    REQUIRE(f != nullptr);
    std::vector<uint8_t> raw;
    CHECK(hrr::read_raw_record(f, raw) == hrr::RecordStatus::Torn);
    fclose(f);
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
TEST_CASE("Unit_HRR_Regions_Roundtrip", "[.][hrr]") {
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
    // Summary line is "... checked, N intra-segment out-of-bounds/stale".
    CHECK(out.find(", 0 intra-segment") == std::string::npos);
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
TEST_CASE("Unit_HRR_Regions_SegmentMaterialization", "[.][hrr]") {
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

// ---------------------------------------------------------------------------
// The same bypassed segment, reached only through a pointer embedded in a
// by-value struct. A whole-pointer argument and an embedded one are the same
// fact about memory, so both have to materialise the segment behind them —
// otherwise a framework that hands its kernels a struct of tensor bases, which
// is the common case, gets none of the benefit of annotating its allocator.
//
// The assertion works because nothing else reaches this segment: the sidecar
// declares only the range around the second probe, which the workload passes
// exclusively inside the struct. A replay that ignores embedded pointers
// materialises nothing at all.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_Regions_EmbeddedPointerMaterialization", "[.][hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_regions_embedded.hrr");
  hrr_capture_direct("Unit_HRR_Regions_Direct", cap.path, /*min_events=*/5);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const uint64_t probe2 = find_alloc_base(archive, kProbe2Bytes);
  REQUIRE(probe2 != 0);

  const uint64_t bypassed = probe2 + kProbeFarOff;
  if (covered_by_recorded_alloc(archive, bypassed)) {
    WARN("The driver placed an allocation over the second probe address; "
         "skipping the embedded-pointer assertions for this run");
    return;
  }

  install_sidecar(archive, {
      region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT, bypassed - 4096,
                 64 * 1024),
  });

  auto [rc, out] = hrr_playback_merged(archive, "--warn-untranslated-args");
  INFO("Embedded-pointer replay:\n" << out);
  CHECK(rc == 0);
  CHECK(out.find("Region segs    : 1 materialised") != std::string::npos);
}

// ---------------------------------------------------------------------------
// A memcpy is the first touch of a bypassed segment.
//
// Region materialisation used to be wired into kernel-argument translation
// only, so a sidecar-declared allocation whose first touch was a captured copy
// failed as unmapped in the memcpy handler before materialisation could run.
// That contradicted the promise made above `advance_to` in `dispatch_event`,
// which says the region timeline is advanced there so that memcpys and every
// other handler see the same view: the advance was shared, the materialisation
// was not.
//
// The destination here is HSA memory, so unlike the other bypass tests the
// address is not a fabricated one — it is a real buffer the dispatch table
// genuinely never saw, and the sidecar is the only thing that can explain it.
// ---------------------------------------------------------------------------
#if defined(HRR_HAVE_HSA)
namespace {

// Capture a hidden workload, keeping the child's stdout. The HSA base cannot be
// recovered from the archive the way find_alloc_base recovers a hipMalloc, so
// the child reports it and the parent reads it here.
inline std::pair<uint64_t, uint64_t> hrr_capture_direct_hsa(
    const std::string& direct_case, const fs::path& cap_path) {
  std::string out;
  { hip::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"" + direct_case + "\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }

  const size_t at = out.find(HRR_HSA_SEG_MARKER);
  if (at == std::string::npos) return {0, 0};
  unsigned long long base = 0, size = 0;
  if (sscanf(out.c_str() + at, HRR_HSA_SEG_MARKER " 0x%llx %llu", &base,
             &size) != 2)
    return {0, 0};
  return {base, size};
}

}  // namespace

TEST_CASE("Unit_HRR_Regions_MemcpyFirstTouchMaterialization", "[.][hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_regions_memcpy.hrr");
  auto [hsa_base, hsa_size] =
      hrr_capture_direct_hsa("Unit_HRR_Regions_MemcpyDirect", cap.path);

  if (hsa_base == 0 || hsa_size == 0) {
    WARN("No HSA agent with a coarse-grained pool on this platform; "
         "skipping the memcpy-first-touch assertions for this run");
    return;
  }

  const fs::path archive = hrr_single_process_archive(cap.path);

  // The whole test rests on this address being unexplainable from the archive.
  if (covered_by_recorded_alloc(archive, hsa_base)) {
    WARN("A recorded allocation covers the HSA segment; "
         "skipping the memcpy-first-touch assertions for this run");
    return;
  }

  SECTION("without an annotation the copy destination is unmapped") {
    auto [rc, out] = hrr_playback_merged(archive);
    INFO("Unannotated replay:\n" << out);
    CHECK(out.find("not mapped") != std::string::npos);
    CHECK(out.find("Region segs") == std::string::npos);
  }

  SECTION("a SEGMENT annotation is materialised on the memcpy path") {
    install_sidecar(archive, {
        region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT, hsa_base, hsa_size),
    });

    auto [rc, out] = hrr_playback_merged(archive);
    INFO("Annotated replay:\n" << out);
    CHECK(rc == 0);
    // Before the memcpy path learned to materialise, this line was absent
    // entirely: translate_ptr failed and the handler gave up before the
    // sidecar was ever consulted.
    CHECK(out.find("Region segs    : 1 materialised") != std::string::npos);
    CHECK(out.find("not mapped") == std::string::npos);
  }
}
#endif  // HRR_HAVE_HSA

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
