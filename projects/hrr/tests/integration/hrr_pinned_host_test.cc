/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR pinned host snapshots
 * @{
 * @ingroup HRRTest
 * Tests for kernels that read pinned host memory directly.
 *
 * The host fills a hipHostMalloc buffer with ordinary CPU stores and hands its
 * address to a kernel. No HIP call carries those bytes, so before capture
 * snapshotted the buffer, replay allocated a fresh one and the kernel read
 * zeros. Capture now records the pinned allocations a launch's arguments point
 * into, before the launch, and replay writes them back before it launches.
 *
 * The workload reads the buffer through a whole-pointer argument and through a
 * by-value struct that holds only the pointer, and copies every result back to
 * the host, so the replay's D2H checks fail if any read saw the wrong bytes.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
// 384 KiB: one whole 256 KiB snapshot chunk and one partial one.
constexpr int      kPinnedInts  = 96 * 1024;
constexpr size_t   kPinnedBytes = kPinnedInts * sizeof(int);
constexpr uint64_t kChunk       = 256 * 1024;
constexpr int      kThreads     = 256;
constexpr int      kBlocks      = (kPinnedInts + kThreads - 1) / kThreads;
// Launches the workload makes before any HRR_PINNED_EXTRA_READS.
constexpr size_t   kBaseLaunches = 7;
// Reads among them, each followed by a D2H copy.
constexpr int      kBaseReads    = 5;

// The scalar keeps the struct a by-value argument: a struct holding only a
// pointer is passed as a plain pointer argument, which is not the path under
// test. The pointer lands at byte offset 8.
struct PinnedView {
  int        scale;
  const int* p;
};

int pattern(int which, int i) { return i * (7 + 4 * which) + 3 * which + 1; }

int env_int(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : 0;
}
}  // namespace

__global__ void hrr_pinned_read(const int* in, int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 3 + 1;
}

__global__ void hrr_pinned_read_view(PinnedView v, int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = v.p[i] * v.scale - 2;
}

__global__ void hrr_pinned_write(int* buf, int n, int seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = seed ^ i;
}

// ===========================================================================
// The captured workload.
//
// Launch order, which the parent tests index into:
//   0  read      host filled pattern 1          (new content)
//   1  read_view host filled pattern 2          (new content, struct argument)
//   2  read      nothing changed                (unchanged content)
//   3  write     device overwrites the buffer
//   4  read      host put pattern 2 back        (unchanged since launch 3's snapshot,
//                                                but not what replay's buffer holds)
//   5  write     device overwrites the buffer
//   6  read      straight after launch 5        (kernel A writes, kernel B reads)
//   7+ read      HRR_PINNED_EXTRA_READS more, unchanged unless
//                HRR_PINNED_EXTRA_CHANGE is set, which bumps one int each time
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));

  std::vector<int> got(kPinnedInts);
  auto read_back = [&](auto expect) {
    HRR_HIP_CHECK(hipDeviceSynchronize());
    HRR_HIP_CHECK(hipMemcpy(got.data(), out, kPinnedBytes, hipMemcpyDeviceToHost));
    for (int i = 0; i < kPinnedInts; ++i) {
      if (got[i] != expect(i)) {
        INFO("index " << i << ": got " << got[i] << ", want " << expect(i));
        REQUIRE(got[i] == expect(i));
      }
    }
  };
  auto read = [&](auto expect) {
    hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, nullptr,
                       h, out, kPinnedInts);
    HRR_HIP_CHECK(hipGetLastError());
    read_back(expect);
  };

  // 0
  for (int i = 0; i < kPinnedInts; ++i) h[i] = pattern(1, i);
  read([](int i) { return pattern(1, i) * 3 + 1; });

  // 1
  for (int i = 0; i < kPinnedInts; ++i) h[i] = pattern(2, i);
  const std::vector<int> saved(h, h + kPinnedInts);
  hipLaunchKernelGGL(hrr_pinned_read_view, dim3(kBlocks), dim3(kThreads), 0,
                     nullptr, PinnedView{5, h}, out, kPinnedInts);
  HRR_HIP_CHECK(hipGetLastError());
  read_back([](int i) { return pattern(2, i) * 5 - 2; });

  // 2
  read([](int i) { return pattern(2, i) * 3 + 1; });

  // 3, 4
  hipLaunchKernelGGL(hrr_pinned_write, dim3(kBlocks), dim3(kThreads), 0, nullptr,
                     h, kPinnedInts, 0x5a5a);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(h[1] == (0x5a5a ^ 1));
  std::memcpy(h, saved.data(), kPinnedBytes);
  read([](int i) { return pattern(2, i) * 3 + 1; });

  // 5, 6: no host synchronisation between the two launches.
  hipLaunchKernelGGL(hrr_pinned_write, dim3(kBlocks), dim3(kThreads), 0, nullptr,
                     h, kPinnedInts, 0x1234);
  HRR_HIP_CHECK(hipGetLastError());
  read([](int i) { return (0x1234 ^ i) * 3 + 1; });

  // 7+
  const int extra = env_int("HRR_PINNED_EXTRA_READS");
  const bool change = env_int("HRR_PINNED_EXTRA_CHANGE") != 0;
  for (int k = 0; k < extra; ++k) {
    if (change) h[0] += 1;
    const int h0 = h[0];
    read([h0](int i) { return (i == 0 ? h0 : (0x1234 ^ i)) * 3 + 1; });
  }

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

namespace {
constexpr const char* kDirect = "Unit_HRR_PinnedHost_Direct";

void capture_pinned(const fs::path& cap,
                    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  hrr::test::SpawnProc proc(HRR_TEST_EXE);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  set_proc_search_path(proc);
  const int ret = proc.run(std::string("\"") + kDirect + "\"");
  INFO("Capture exit: " << ret);
  REQUIRE(ret == 0);
}

// The archive's kernel launches, in order.
std::vector<const hrr::KernelLaunchEvent*> launches_of(const hrr::Archive& arc) {
  std::vector<const hrr::KernelLaunchEvent*> v;
  for (const auto& ev : arc.events)
    if (ev.kernel_launch) v.push_back(ev.kernel_launch);
  return v;
}

bool same_hashes(const hrr::KernelLaunchEvent* a, const hrr::KernelLaunchEvent* b) {
  if (a->snapshots.size() != b->snapshots.size()) return false;
  for (size_t i = 0; i < a->snapshots.size(); ++i)
    if (a->snapshots[i].hash_lo != b->snapshots[i].hash_lo ||
        a->snapshots[i].hash_hi != b->snapshots[i].hash_hi)
      return false;
  return true;
}

size_t snapshot_records(const std::vector<const hrr::KernelLaunchEvent*>& v) {
  size_t n = 0;
  for (const auto* kl : v) n += kl->snapshots.size();
  return n;
}

// "[HRR]   Host snapshots : N chunk(s) ... restored, M record(s) rejected".
// Absent when replay restored and rejected nothing.
void host_snapshot_summary(const std::string& out, unsigned long long& restored,
                           unsigned long long& rejected) {
  restored = rejected = 0;
  const size_t at = out.find("Host snapshots :");
  if (at == std::string::npos) return;
  REQUIRE(std::sscanf(out.c_str() + at,
                      "Host snapshots : %llu chunk(s) of pinned host memory "
                      "restored, %llu record(s) rejected",
                      &restored, &rejected) == 2);
}

std::vector<uint8_t> read_bytes(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

// Byte range of each kernel launch event in events.bin, in file order.
std::vector<std::pair<size_t, size_t>> launch_spans(const std::vector<uint8_t>& f) {
  std::vector<std::pair<size_t, size_t>> spans;
  size_t p = sizeof(hrr_file_header);
  while (p + sizeof(hrr_event_header) <= f.size()) {
    hrr_event_header h;
    std::memcpy(&h, f.data() + p, sizeof(h));
    if (h.payload_length < sizeof(h) || h.payload_length > f.size() - p) break;
    switch (h.event_type) {
      case HRR_API_HIPMODULELAUNCHKERNEL:
      case HRR_API_HIPEXTMODULELAUNCHKERNEL:
      case HRR_API_HIPLAUNCHKERNEL:
      case HRR_API_HIPLAUNCHBYPTR:
        spans.emplace_back(p, p + h.payload_length);
        break;
      default:
        break;
    }
    p += h.payload_length;
  }
  return spans;
}

// Overwrite one field of snapshot record `rec` of the `launch`-th launch.
// field: 0 ptr, 1 offset, 2 length.
void patch_snapshot(std::vector<uint8_t>& f,
                    const std::vector<std::pair<size_t, size_t>>& spans,
                    const std::vector<const hrr::KernelLaunchEvent*>& kls,
                    size_t launch, size_t rec, int field, uint64_t value) {
  INFO("launch " << launch << " record " << rec);
  REQUIRE(launch < kls.size());
  REQUIRE(launch < spans.size());
  REQUIRE(rec < kls[launch]->snapshots.size());
  const hrr::BufferSnapshot& s = kls[launch]->snapshots[rec];
  uint64_t needle[5] = {s.ptr_handle, s.offset, s.length, s.hash_lo, s.hash_hi};
  const auto* nb = reinterpret_cast<const uint8_t*>(needle);
  auto first = f.begin() + spans[launch].first;
  auto last  = f.begin() + spans[launch].second;
  auto at = std::search(first, last, nb, nb + sizeof(needle));
  REQUIRE(at != last);
  std::memcpy(&*at + field * 8, &value, sizeof(value));
}
}  // namespace

// ---------------------------------------------------------------------------
// Every read replays with the bytes the host left in the buffer.
//
// Before capture recorded pinned buffers, each read replayed on a buffer
// nobody filled and every D2H check after it failed.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_Restored) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_host.hrr");
  capture_pinned(cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches);

  // Every launch records both chunks of the one pinned allocation.
  for (size_t k = 0; k < kls.size(); ++k) {
    INFO("launch " << k);
    REQUIRE(kls[k]->snapshots.size() == 2);
    CHECK(kls[k]->snapshots[0].ptr_handle == kls[0]->snapshots[0].ptr_handle);
    CHECK(kls[k]->snapshots[1].ptr_handle == kls[0]->snapshots[0].ptr_handle);
    CHECK(kls[k]->snapshots[0].offset == 0);
    CHECK(kls[k]->snapshots[0].length == kChunk);
    CHECK(kls[k]->snapshots[1].offset == kChunk);
    CHECK(kls[k]->snapshots[1].length == kPinnedBytes - kChunk);
    CHECK(kls[k]->snapshots[0].direction == 0);
  }
  CHECK_FALSE(same_hashes(kls[0], kls[1]));  // the host wrote new content
  CHECK(same_hashes(kls[1], kls[2]));        // nothing changed
  CHECK(same_hashes(kls[1], kls[4]));        // the host put pattern 2 back
  CHECK_FALSE(same_hashes(kls[4], kls[6]));  // kernel A rewrote the buffer

  // The struct argument is marked as holding a pointer at offset 8, so replay
  // translates it rather than passing the capture-time host address.
  REQUIRE(!kls[1]->args.empty());
  CHECK(kls[1]->args[0].value_kind == 3);
  CHECK(kls[1]->args[0].ptr_offsets == std::vector<uint16_t>{8});

  auto [rc, out] = hrr_playback_merged(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);

  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
#ifndef _WIN32
  // Windows replay does not promise bit-exact device output; see hrr_run_playback.
  CHECK(d2h_pass >= kBaseReads);
  CHECK(d2h_fail == 0);
#endif

  // Replay copies a chunk only when its buffer holds something else, two
  // chunks per launch here. Launches 0 and 1 find the previous content and
  // launch 4 finds what kernel 3 wrote. Launches 2, 3 and 5 find what the
  // host left, and launch 6 finds what kernel 5 wrote on replay too.
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 6);
  CHECK(rejected == 0);
}

// ---------------------------------------------------------------------------
// A buffer that does not change is stored once.
//
// Four more unchanged reads add four launches with two records each, and no
// blob. The same four reads with one int bumped each time add blobs, which
// shows the count is sensitive to what the snapshots hold.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_UnchangedStoredOnce) {
  auto capture = [](const char* name,
                    std::vector<std::pair<std::string, std::string>> env,
                    size_t& blobs, size_t& launches, size_t& records) {
    ScopedDir cap(fs::temp_directory_path() / name);
    capture_pinned(cap.path, env);
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));
    const auto kls = launches_of(arc);
    blobs = arc.blob_count;
    launches = kls.size();
    records = snapshot_records(kls);
  };

  size_t base_blobs = 0, base_launches = 0, base_records = 0;
  capture("hrr_pinned_once_base.hrr", {}, base_blobs, base_launches, base_records);
  REQUIRE(base_launches == kBaseLaunches);

  size_t same_blobs = 0, same_launches = 0, same_records = 0;
  capture("hrr_pinned_once_same.hrr", {{"HRR_PINNED_EXTRA_READS", "4"}},
          same_blobs, same_launches, same_records);
  CHECK(same_launches == kBaseLaunches + 4);
  CHECK(same_records == base_records + 4 * 2);
  CHECK(same_blobs == base_blobs);

  size_t diff_blobs = 0, diff_launches = 0, diff_records = 0;
  capture("hrr_pinned_once_diff.hrr",
          {{"HRR_PINNED_EXTRA_READS", "4"}, {"HRR_PINNED_EXTRA_CHANGE", "1"}},
          diff_blobs, diff_launches, diff_records);
  CHECK(diff_launches == kBaseLaunches + 4);
  CHECK(diff_blobs >= base_blobs + 4);
}

// ---------------------------------------------------------------------------
// HIP_HRR_HOST_SNAPSHOTS=0 records no pinned contents, says so in the
// manifest, and replay then reads an unfilled buffer.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_OptOut) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_optout.hrr");
  capture_pinned(cap.path, {{"HIP_HRR_HOST_SNAPSHOTS", "0"}});
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest.find("\"host_snapshots\": false") != std::string::npos);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches);
  CHECK(snapshot_records(kls) == 0);

  // The struct pointer is still translated: opting out drops the contents, not
  // the pointer bookkeeping.
  CHECK(kls[1]->args[0].value_kind == 3);

  auto [rc, out] = hrr_playback_merged(archive);
  INFO("Replay:\n" << out);
  CHECK(rc < 128);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_fail >= 1);
#endif
}

// ---------------------------------------------------------------------------
// Archive contents are untrusted. A record whose range does not fit its
// allocation, whose length does not match its blob, or whose pointer names no
// allocation is refused by name and nothing is written for it.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_MalformedRecordRejected) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_malformed.hrr");
  capture_pinned(cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches);

  std::vector<uint8_t> events = read_bytes(archive / "events.bin");
  const auto spans = launch_spans(events);
  REQUIRE(spans.size() == kls.size());

  // Launch 0: a length past the end, and an offset that wraps when added.
  patch_snapshot(events, spans, kls, 0, 0, /*length*/ 2, 1ull << 40);
  patch_snapshot(events, spans, kls, 0, 1, /*offset*/ 1, ~0ull - 4095);
  // Launch 1: 8 bytes that would run past the allocation end.
  patch_snapshot(events, spans, kls, 1, 1, /*offset*/ 1, kPinnedBytes - 8);
  // Launch 2: in bounds, but shorter than its blob.
  patch_snapshot(events, spans, kls, 2, 0, /*length*/ 2, kChunk - 8);
  // Launch 4: a pointer no allocation covers.
  patch_snapshot(events, spans, kls, 4, 0, /*ptr*/ 0, 0x10);
  {
    std::ofstream o(archive / "events.bin", std::ios::binary | std::ios::trunc);
    o.write(reinterpret_cast<const char*>(events.data()), events.size());
  }

  auto [rc, out] = hrr_playback_merged(archive);
  INFO("Replay:\n" << out);
  // Rejected restores leave reads on the wrong bytes, so the D2H checks may
  // fail; the replay itself must finish.
  CHECK(rc < 128);

  auto count = [&](const std::string& s) {
    size_t n = 0;
    for (size_t at = out.find(s); at != std::string::npos; at = out.find(s, at + 1)) ++n;
    return n;
  };
  CHECK(count("is out of bounds of its allocation") == 3);
  CHECK(count("does not match the size of its blob") == 1);
  CHECK(count("names no live allocation") == 1);

  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(rejected == 5);
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
