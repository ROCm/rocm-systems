/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR external region annotations
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include "hrr/hrr_regions.h"
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



namespace {
constexpr size_t kSegBytes   = 1u << 20;  // one "segment", from a single hipMalloc
constexpr size_t kBlockBytes = 4096;      // one "block" carved out of it
constexpr size_t kBlock1Off  = 64 * 1024; // second block's offset in the segment
constexpr size_t kGapOff     = 32 * 1024; // inside the segment, inside no block

constexpr int kFloatsPerBlock = static_cast<int>(kBlockBytes / sizeof(float));
// Far enough past the block's end to clear the guard page under --guard-blocks,
// while staying inside the segment so an unguarded replay is silent.
constexpr int kOverrunFloats  = kFloatsPerBlock + 16384;

// A second, small allocation. A kernel launched with n == 0 gets a pointer far past its end,
// standing in for memory HRR never observed (such as a direct HSA call). Nothing is dereferenced.
constexpr size_t kProbeBytes = 64 * 1024;
constexpr size_t kProbeFarOff = 1u << 20;  // past the end of the probe alloc

constexpr size_t kProbe2Bytes = 32 * 1024;
}  // namespace

__global__ void hrr_regions_fill(float* p, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

// Two device pointers inside one by-value argument, as frameworks pass kernel operands.
// `known` resolves to a captured allocation, so capture tags the argument as carrying embedded pointers.
// `bypassed` resolves to nothing, standing in for memory allocated below the HIP API.
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

TEST_CASE("Unit_HRR_Regions_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  char* seg = nullptr;
  HRR_HIP_CHECK(hipMalloc(&seg, kSegBytes));
  HRR_HIP_CHECK(hipMemset(seg, 0, kSegBytes));

  float* block0 = reinterpret_cast<float*>(seg);
  float* block1 = reinterpret_cast<float*>(seg + kBlock1Off);
  float* gap    = reinterpret_cast<float*>(seg + kGapOff);

  const int threads = 256;
  auto grid = [&](int n) { return dim3((n + threads - 1) / threads); };

  hipLaunchKernelGGL(hrr_regions_fill, grid(kFloatsPerBlock), dim3(threads), 0,
                     nullptr, block0, kFloatsPerBlock, 1.0f);
  HRR_HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(hrr_regions_fill, grid(kFloatsPerBlock), dim3(threads), 0,
                     nullptr, block1, kFloatsPerBlock, 2.0f);
  HRR_HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(hrr_regions_fill, grid(16), dim3(threads), 0, nullptr,
                     gap, 16, 3.0f);
  HRR_HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(hrr_regions_fill, grid(kOverrunFloats), dim3(threads), 0,
                     nullptr, block0, kOverrunFloats, 4.0f);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // A pointer no allocation covers, like a buffer allocated below the HIP API.
  // n == 0, so no thread dereferences it here or at replay.
  char* probe = nullptr;
  HRR_HIP_CHECK(hipMalloc(&probe, kProbeBytes));
  hipLaunchKernelGGL(hrr_regions_fill, dim3(1), dim3(threads), 0, nullptr,
                     reinterpret_cast<float*>(probe + kProbeFarOff), 0, 5.0f);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  char* probe2 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&probe2, kProbe2Bytes));
  HrrRegionsPtrPair pair{block0,
                         reinterpret_cast<float*>(probe2 + kProbeFarOff), 0};
  hipLaunchKernelGGL(hrr_regions_fill_pair, dim3(1), dim3(threads), 0, nullptr,
                     pair, 6.0f);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> host(kFloatsPerBlock);
  HRR_HIP_CHECK(hipMemcpy(host.data(), block0, kBlockBytes, hipMemcpyDeviceToHost));
  REQUIRE(host[0] == 4.0f);

  HRR_HIP_CHECK(hipFree(probe2));
  HRR_HIP_CHECK(hipFree(probe));
  HRR_HIP_CHECK(hipFree(seg));
}

// A memcpy needs a destination that exists yet is invisible to HRR, so it must be a real allocation made below the HIP API.
// HSA provides that: a coarse-grained device pool buffer the HIP dispatch table never sees.

#if defined(HRR_HAVE_HSA)
namespace {

constexpr size_t kHsaSegBytes = 256 * 1024;

// The coarse-grained global pool of the first GPU agent, plus the CPU and GPU agents. All need access,
// because a DMA engine services the copy and faults on an unshared coarse-grained allocation.
// hsa_amd_agents_allow_access rejects any other agent with HSA_STATUS_ERROR_INVALID_AGENT.
// gfx1151 enumerates an NPU, and including it fails the call before the copy runs.
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

  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hrr_visit_agent(hsa_agent_t agent, void* data) {
  auto* target = static_cast<HsaTarget*>(data);

  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) !=
          HSA_STATUS_SUCCESS ||
      (type != HSA_DEVICE_TYPE_CPU && type != HSA_DEVICE_TYPE_GPU))
    return HSA_STATUS_SUCCESS;
  target->all_agents.push_back(agent);

  if (type != HSA_DEVICE_TYPE_GPU)
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

// HIP already initialised HSA in this process, so hsa_init attaches to the running runtime.
HsaTarget hrr_hsa_target() {
  HsaTarget target;
  if (hsa_init() != HSA_STATUS_SUCCESS) return target;
  hsa_iterate_agents(hrr_visit_agent, &target);
  return target;
}

}  // namespace

// The parent parses this marker from the child's stdout. An HSA address is in no hipMalloc record,
// so find_alloc_base cannot recover it and the child must report it.
#define HRR_HSA_SEG_MARKER "HRR_HSA_SEG"

TEST_CASE("Unit_HRR_Regions_MemcpyDirect", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

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

  // The only touch of `seg`, deliberately not passed to a kernel: a kernel argument would let
  // whole-pointer materialisation resolve it, and the test would pass without the memcpy fix.
  HRR_HIP_CHECK(hipMemcpy(seg, host.data(), kHsaSegBytes, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  char* known = nullptr;
  HRR_HIP_CHECK(hipMalloc(&known, kHsaSegBytes));
  HRR_HIP_CHECK(hipMemcpy(known, seg, kHsaSegBytes, hipMemcpyDeviceToDevice));
  std::vector<uint8_t> back(kHsaSegBytes, 0);
  HRR_HIP_CHECK(hipMemcpy(back.data(), known, kHsaSegBytes, hipMemcpyDeviceToHost));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(back[0] == host[0]);
  REQUIRE(back[kHsaSegBytes - 1] == host[kHsaSegBytes - 1]);

  HRR_HIP_CHECK(hipFree(known));
  REQUIRE(hsa_amd_memory_pool_free(seg) == HSA_STATUS_SUCCESS);
}
#endif  // HRR_HAVE_HSA


namespace {

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

hrr_region_rec region_rec(uint8_t op, uint8_t kind, uint64_t base, uint64_t size) {
  hrr_region_rec r{};
  r.op = op;
  r.kind = kind;
  r.base = base;
  r.size = size;
  return r;
}

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
    // A producer killed mid-write: everything it finished writing must survive, since the recorded process often crashes.
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
    CHECK(hrr::open_record_stream(stream.string(), HRR_MAGIC, HRR_VERSION,
                                  nullptr) == nullptr);
  }
}


#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {

void install_sidecar(const fs::path& archive_path,
                     const std::vector<hrr_region_rec>& recs) {
  const fs::path regions = archive_path / "regions";
  fs::remove_all(regions);
  write_region_batch(regions / "synthetic.hrrr", recs);
}

}  // namespace

TEST_CASE("Unit_HRR_Regions_Roundtrip", "[.][hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_regions_roundtrip.hrr");
  hrr_capture_direct("Unit_HRR_Regions_Direct", cap.path, /*min_events=*/5);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const uint64_t seg = find_alloc_base(archive, kSegBytes);
  INFO("Recorded segment base: 0x" << std::hex << seg);
  REQUIRE(seg != 0);

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
    // Reporting only: the finding describes the recorded program and must not by itself fail the replay.
    CHECK(rc == 0);
    CHECK(out.find("region OOB") != std::string::npos);
    CHECK(out.find("intra-segment out-of-bounds/stale") != std::string::npos);
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
    // The overrun is meant to fault; keep the driver from writing a GPU core dump for it.
    // The variable is set on the spawned hrr-playback only.
    auto [rc, out] = hrr_playback_merged(
        archive, "--guard-blocks", {{"HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1"}});
    INFO("Guarded replay:\n" << out);
    if (rc == 0) {
      // A clean run is acceptable only when nothing was guarded: no VMM support, or every reservation failed.
      INFO("Guard reported a clean run; it must not have relocated anything");
      CHECK(out.find(": 0 block relocation(s)") != std::string::npos);
    }
  }
}

// A kernel argument resolves in no map because its allocation never crossed a HIP API. Annotation makes replay back that segment.
// Declaring a segment the archive already has must not allocate a second buffer.
TEST_CASE("Unit_HRR_Regions_SegmentMaterialization", "[.][hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_regions_segments.hrr");
  hrr_capture_direct("Unit_HRR_Regions_Direct", cap.path, /*min_events=*/5);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const uint64_t seg   = find_alloc_base(archive, kSegBytes);
  const uint64_t probe = find_alloc_base(archive, kProbeBytes);
  REQUIRE(seg != 0);
  REQUIRE(probe != 0);

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
        region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT, seg, kSegBytes),
        region_rec(HRR_REGION_ADD, HRR_REGION_SEGMENT, bypassed - 4096,
                   64 * 1024),
    });

    auto [rc, out] = hrr_playback_merged(archive, "--warn-untranslated-args");
    INFO("Annotated replay:\n" << out);
    CHECK(rc == 0);
    // Declaring the captured segment must not allocate a second buffer for it.
    CHECK(out.find("Region segs    : 1 materialised") != std::string::npos);
    CHECK(out.find("Untranslated   : 0 ") != std::string::npos);
  }
}

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

#if defined(HRR_HAVE_HSA)
namespace {

// The HSA base cannot be recovered from the archive as find_alloc_base recovers a hipMalloc,
// so the child reports it on stdout and the parent reads it.
inline std::pair<uint64_t, uint64_t> hrr_capture_direct_hsa(
    const std::string& direct_case, const fs::path& cap_path) {
  std::string out;
  { hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true);
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
    CHECK(out.find("Region segs    : 1 materialised") != std::string::npos);
    CHECK(out.find("not mapped") == std::string::npos);
  }
}
#endif  // HRR_HAVE_HSA

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
