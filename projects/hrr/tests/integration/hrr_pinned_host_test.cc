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
 * The host fills a pinned buffer with ordinary CPU stores and hands its address
 * to a kernel. No HIP call carries those bytes, so before capture snapshotted
 * the buffer, replay allocated a fresh one and the kernel read zeros. Capture
 * now records the pinned allocations a launch's arguments point into, before
 * the launch, and replay writes them back before it launches.
 *
 * Every workload copies each result back to the host, so the replay's D2H
 * checks fail if any read saw the wrong bytes.
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
// Launches the main workload makes before any HRR_PINNED_EXTRA_READS.
constexpr size_t   kBaseLaunches = 5;
// Reads among them, each followed by a D2H copy.
constexpr int      kBaseReads    = 4;

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

void fill(int* h, int which, size_t n = kPinnedInts) {
  for (size_t i = 0; i < n; ++i) h[i] = pattern(which, static_cast<int>(i));
}

// Wait for the device, copy `n` ints of out back and compare them.
template <typename F>
void check_out(const int* out, F expect, int n = kPinnedInts) {
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> got(n);
  HRR_HIP_CHECK(hipMemcpy(got.data(), out, n * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (got[i] != expect(i)) {
      INFO("index " << i << ": got " << got[i] << ", want " << expect(i));
      REQUIRE(got[i] == expect(i));
    }
  }
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

namespace {
// One hrr_pinned_read launch over the first n ints of in, then the D2H check.
void read_pinned(const int* in, int* out, int which, int n = kPinnedInts,
                 hipStream_t s = nullptr) {
  hipLaunchKernelGGL(hrr_pinned_read, dim3((n + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, s, in, out, n);
  HRR_HIP_CHECK(hipGetLastError());
  check_out(out, [which](int i) { return pattern(which, i) * 3 + 1; }, n);
}
}  // namespace

// ===========================================================================
// The main captured workload.
//
// Launch order, which the parent tests index into:
//   0  read      host filled pattern 1          (new content)
//   1  read_view host filled pattern 2          (new content, struct argument)
//   2  read      nothing changed                (unchanged content)
//   3  write     device overwrites the buffer
//   4  read      host put pattern 2 back        (unchanged since launch 3's snapshot,
//                                                but not what replay's buffer holds)
//   5+ read      HRR_PINNED_EXTRA_READS more, unchanged unless
//                HRR_PINNED_EXTRA_CHANGE is set, which bumps one int each time
// HRR_PINNED_SKIP_VIEW leaves out launch 1, so the others shift down by one.
// Every launch follows a host synchronisation, so the stream is idle.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));

  auto read = [&](auto expect) {
    hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, nullptr,
                       h, out, kPinnedInts);
    HRR_HIP_CHECK(hipGetLastError());
    check_out(out, expect);
  };

  // 0
  fill(h, 1);
  read([](int i) { return pattern(1, i) * 3 + 1; });

  // 1
  fill(h, 2);
  const std::vector<int> saved(h, h + kPinnedInts);
  if (env_int("HRR_PINNED_SKIP_VIEW") == 0) {
    hipLaunchKernelGGL(hrr_pinned_read_view, dim3(kBlocks), dim3(kThreads), 0,
                       nullptr, PinnedView{5, h}, out, kPinnedInts);
    HRR_HIP_CHECK(hipGetLastError());
    check_out(out, [](int i) { return pattern(2, i) * 5 - 2; });
  }

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

  // 5+
  const int extra = env_int("HRR_PINNED_EXTRA_READS");
  const bool change = env_int("HRR_PINNED_EXTRA_CHANGE") != 0;
  for (int k = 0; k < extra; ++k) {
    if (change) h[0] += 1;
    const int h0 = h[0];
    read([h0](int i) { return (i == 0 ? h0 : pattern(2, i)) * 3 + 1; });
  }

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// A launch queued behind work that waits for the host.
//
//   0  read      host filled pattern 1, stream idle
//   1  read      queued behind hipStreamWaitValue32 on a flag the host sets
//                only after the launch returns; the host changed one int of
//                chunk 0 first, and chunk 1 is unchanged
// Capture that waited for the stream before its snapshot would never return
// from launch 1.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_NoWait_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int can_wait = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, 0));
  REQUIRE(can_wait != 0);

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  uint32_t* flag = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(uint32_t),
                              hipHostMallocMapped));
  *flag = 0;
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // 0
  fill(h, 1);
  read_pinned(h, out, 1, kPinnedInts, s);

  // 1
  h[0] = -7;
  HRR_HIP_CHECK(hipStreamWaitValue32(s, flag, 1, hipStreamWaitValueEq, 0xFFFFFFFF));
  hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, s, h, out,
                     kPinnedInts);
  HRR_HIP_CHECK(hipGetLastError());
  // Still blocked: the launch returned without the stream draining.
  CHECK(hipStreamQuery(s) == hipErrorNotReady);
  __atomic_store_n(flag, 1u, __ATOMIC_SEQ_CST);
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  check_out(out, [](int i) { return (i == 0 ? -7 : pattern(1, i)) * 3 + 1; });

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(flag));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// Two pinned buffers of HRR_PINNED_KB KiB each (default 1536, six chunks),
// read in turn: launch 0 reads buffer A, launch 1 reads buffer B. The kernels
// read only the first 4096 ints; the snapshot covers each whole allocation.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_Sizes_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  const int    kb    = env_int("HRR_PINNED_KB") > 0 ? env_int("HRR_PINNED_KB") : 1536;
  const size_t bytes = static_cast<size_t>(kb) * 1024;
  const size_t ints  = bytes / sizeof(int);
  constexpr int kRead = 4096;

  int* a = nullptr;
  int* b = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&a), bytes, hipHostMallocDefault));
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&b), bytes, hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kRead * sizeof(int)));
  fill(a, 1, ints);
  fill(b, 2, ints);

  read_pinned(a, out, 1, kRead);
  read_pinned(b, out, 2, kRead);

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(b));
  HRR_HIP_CHECK(hipHostFree(a));
}

// ===========================================================================
// Every way to get pinned memory, each freed the way an application may.
//
//   0  hipHostAlloc,    freed with hipHostFree
//   1  hipMallocHost,   freed with hipFreeHost
//   2  hipMemAllocHost, freed with hipFree
//   3  hipHostMalloc,   freed with hipHostFree
//   4  hipHostRegister, read through the device pointer hipHostGetDevicePointer
//                       returns (which may equal the host pointer)
//   5  the same registered range, read through the host pointer;
//                       then hipHostUnregister
//   6  hipHostMalloc again, which may land where launch 3's buffer was
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_Lifetime_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  void* p = nullptr;

  // 0
  HRR_HIP_CHECK(hipHostAlloc(&p, kPinnedBytes, hipHostMallocDefault));
  fill(static_cast<int*>(p), 10);
  read_pinned(static_cast<int*>(p), out, 10);
  HRR_HIP_CHECK(hipHostFree(p));

  // 1
  HRR_HIP_CHECK(hipMallocHost(&p, kPinnedBytes));
  fill(static_cast<int*>(p), 11);
  read_pinned(static_cast<int*>(p), out, 11);
  HRR_HIP_CHECK(hipFreeHost(p));

  // 2
  HRR_HIP_CHECK(hipMemAllocHost(&p, kPinnedBytes));
  fill(static_cast<int*>(p), 12);
  read_pinned(static_cast<int*>(p), out, 12);
  HRR_HIP_CHECK(hipFree(p));

  // 3
  HRR_HIP_CHECK(hipHostMalloc(&p, kPinnedBytes, hipHostMallocDefault));
  fill(static_cast<int*>(p), 13);
  read_pinned(static_cast<int*>(p), out, 13);
  HRR_HIP_CHECK(hipHostFree(p));

  // 4, 5
  void* reg = std::aligned_alloc(4096, kPinnedBytes);
  REQUIRE(reg != nullptr);
  HRR_HIP_CHECK(hipHostRegister(reg, kPinnedBytes, hipHostRegisterMapped));
  void* dev = nullptr;
  HRR_HIP_CHECK(hipHostGetDevicePointer(&dev, reg, 0));
  fill(static_cast<int*>(reg), 14);
  read_pinned(static_cast<const int*>(dev), out, 14);
  read_pinned(static_cast<const int*>(reg), out, 14);
  HRR_HIP_CHECK(hipHostUnregister(reg));
  std::free(reg);

  // 6
  HRR_HIP_CHECK(hipHostMalloc(&p, kPinnedBytes, hipHostMallocDefault));
  fill(static_cast<int*>(p), 15);
  read_pinned(static_cast<int*>(p), out, 15);
  HRR_HIP_CHECK(hipHostFree(p));

  HRR_HIP_CHECK(hipFree(out));
}

// ===========================================================================
// Launch entry points other than hipLaunchKernelGGL.
//
//   0  hipLaunchKernel_spt on the per-thread stream
//   1  hipModuleLaunchKernel with the arguments packed in `extra`, on a handle
//      from hipGetFuncBySymbol
//   -  hipModuleLaunchKernel on a handle that is not a kernel, with a zero grid:
//      the runtime refuses the grid before it looks at the handle, and capture
//      must not read the handle first. Nothing is recorded for it.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_LaunchApis_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  int n = kPinnedInts;

  // 0
  fill(h, 1);
  void* args[] = {&h, &out, &n};
  HRR_HIP_CHECK(hipLaunchKernel_spt(reinterpret_cast<const void*>(hrr_pinned_read),
                                    dim3(kBlocks), dim3(kThreads), args, 0, nullptr));
  check_out(out, [](int i) { return pattern(1, i) * 3 + 1; });

  // 1
  fill(h, 2);
  hipFunction_t f = nullptr;
  HRR_HIP_CHECK(hipGetFuncBySymbol(&f, reinterpret_cast<const void*>(hrr_pinned_read)));
  struct {
    const int* in;
    int*       out;
    int        n;
  } packed{h, out, kPinnedInts};
  size_t packed_size = sizeof(packed);
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &packed,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &packed_size, HIP_LAUNCH_PARAM_END};
  HRR_HIP_CHECK(hipModuleLaunchKernel(f, kBlocks, 1, 1, kThreads, 1, 1, 0, nullptr,
                                      nullptr, extra));
  check_out(out, [](int i) { return pattern(2, i) * 3 + 1; });

  // Not a kernel handle.
  const hipError_t bogus = hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(0x10),
                                                 0, 1, 1, kThreads, 1, 1, 0, nullptr,
                                                 args, nullptr);
  INFO("bogus handle launch: " << hipGetErrorName(bogus));
  CHECK((bogus == hipErrorInvalidValue || bogus == hipErrorInvalidConfiguration));
  (void)hipGetLastError();

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// The pre-chevron launch ABI: hipConfigureCall, hipSetupArgument and
// hipLaunchByPtr. Captured only: replay of hipSetupArgument reads a
// capture-time host address (see Unit_HRR_ApiMatrix_LegacyLaunch_Direct).
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_ByPtr_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  int n = kPinnedInts;
  fill(h, 1);

  HRR_HIP_CHECK(hipConfigureCall(dim3(kBlocks), dim3(kThreads), 0, nullptr));
  HRR_HIP_CHECK(hipSetupArgument(&h, sizeof(h), 0));
  HRR_HIP_CHECK(hipSetupArgument(&out, sizeof(out), sizeof(h)));
  HRR_HIP_CHECK(hipSetupArgument(&n, sizeof(n), sizeof(h) + sizeof(out)));
  HRR_HIP_CHECK(hipLaunchByPtr(reinterpret_cast<const void*>(hrr_pinned_read)));
  check_out(out, [](int i) { return pattern(1, i) * 3 + 1; });

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// A read launched into a stream under graph capture.
//
//   0  read      ordinary launch on the stream, pattern 1
//   1  read      the same, launched while the stream is captured; the graph
//                then runs it
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_Graph_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // 0
  fill(h, 1);
  read_pinned(h, out, 1, kPinnedInts, s);

  // 1
  fill(h, 2);
  hipGraph_t g = nullptr;
  hipGraphExec_t ge = nullptr;
  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, s, h, out,
                     kPinnedInts);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipStreamEndCapture(s, &g));
  HRR_HIP_CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
  HRR_HIP_CHECK(hipGraphLaunch(ge, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  check_out(out, [](int i) { return pattern(2, i) * 3 + 1; });

  HRR_HIP_CHECK(hipGraphExecDestroy(ge));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

namespace {
constexpr const char* kDirect = "Unit_HRR_PinnedHost_Direct";
// A capture or replay that takes longer than this has hung.
constexpr int kCaptureTimeoutSec = 120;

void capture_case(const char* direct_case, const fs::path& cap,
                  const std::vector<std::pair<std::string, std::string>>& env = {}) {
  hrr::test::SpawnProc proc(HRR_TEST_EXE);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(std::string("\"") + direct_case + "\"",
                                      kCaptureTimeoutSec);
  INFO("Capture of " << direct_case << " exit: " << ret
       << (ret == hrr::test::SpawnProc::kKilledOnTimeout ? " (hung, killed)" : ""));
  REQUIRE(ret == 0);
}

// Replay with stderr merged and byte-exact D2H checks. The default checks
// accept any buffer that matches as floats within 1e-3. Most ints these
// kernels write are below 2^23, float denormals, so wrong output would pass.
// The deadline turns a hang into a failure.
std::pair<int, std::string> replay(
    const fs::path& archive, const std::string& extra_args = "",
    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  hrr::test::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true,
                            /*capture_stderr=*/true);
  set_proc_search_path(proc);
  proc.setEnv("HIP_HRR_D2H_EXACT", "1");
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  const int ret = proc.runWithTimeout(
      hrr_quote_path(archive) + (extra_args.empty() ? "" : " " + extra_args),
      kCaptureTimeoutSec);
  return {ret, proc.getOutput()};
}

void capture_pinned(const fs::path& cap,
                    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  capture_case(kDirect, cap, env);
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

// A numeric manifest field, or -1 when it is absent.
long long manifest_count(const std::string& manifest, const std::string& key) {
  const std::string k = "\"" + key + "\": ";
  const size_t at = manifest.find(k);
  if (at == std::string::npos) return -1;
  return std::strtoll(manifest.c_str() + at + k.size(), nullptr, 10);
}

size_t count_of(const std::string& out, const std::string& s) {
  size_t n = 0;
  for (size_t at = out.find(s); at != std::string::npos; at = out.find(s, at + 1)) ++n;
  return n;
}

std::vector<uint8_t> read_bytes(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

void write_bytes(const fs::path& p, const std::vector<uint8_t>& b) {
  std::ofstream o(p, std::ios::binary | std::ios::trunc);
  o.write(reinterpret_cast<const char*>(b.data()), b.size());
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

// Byte offset in f of the num_snapshots field of the launch at span.
size_t num_snapshots_at(const std::vector<uint8_t>& f, std::pair<size_t, size_t> span) {
  size_t p = span.first + sizeof(hrr_event_header) + 8;  // stream
  uint16_t name_len;
  std::memcpy(&name_len, f.data() + p, 2);
  p += 2 + name_len + 16 + 28 + 2;  // name, code object hash, dims, num_args
  REQUIRE(p + 2 <= span.second);
  return p;
}

// Byte offset in f of snapshot record `rec` of the `launch`-th launch.
size_t record_at(const std::vector<uint8_t>& f,
                 const std::vector<std::pair<size_t, size_t>>& spans,
                 const std::vector<const hrr::KernelLaunchEvent*>& kls,
                 size_t launch, size_t rec) {
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
  return static_cast<size_t>(at - f.begin());
}

// Overwrite one field of snapshot record `rec` of the `launch`-th launch.
// field: 0 ptr, 1 offset, 2 length; 5 is the one-byte direction.
void patch_snapshot(std::vector<uint8_t>& f,
                    const std::vector<std::pair<size_t, size_t>>& spans,
                    const std::vector<const hrr::KernelLaunchEvent*>& kls,
                    size_t launch, size_t rec, int field, uint64_t value) {
  const size_t at = record_at(f, spans, kls, launch, rec);
  if (field == 5)
    f[at + 40] = static_cast<uint8_t>(value);
  else
    std::memcpy(f.data() + at + field * 8, &value, sizeof(value));
}

const fs::path& require_file(const fs::path& p) {
  INFO(p);
  REQUIRE(fs::exists(p));
  return p;
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

  const std::string manifest = read_text_file(require_file(archive / "manifest.json"));
  INFO("manifest:\n" << manifest);
  CHECK(manifest.find("\"host_snapshots\": true") != std::string::npos);
  CHECK(manifest_count(manifest, "host_snapshots_unordered") == 0);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches);

  // Every launch records both chunks of the one pinned allocation. Every launch
  // found the stream idle, so every record is one replay restores.
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
    CHECK(kls[k]->snapshots[1].direction == 0);
  }
  CHECK_FALSE(same_hashes(kls[0], kls[1]));  // the host wrote new content
  CHECK(same_hashes(kls[1], kls[2]));        // nothing changed
  CHECK(same_hashes(kls[2], kls[3]));        // the write kernel had not run yet
  CHECK(same_hashes(kls[1], kls[4]));        // the host put pattern 2 back

  // The struct argument is marked as holding a pointer at offset 8, so replay
  // translates it rather than passing the capture-time host address.
  REQUIRE(!kls[1]->args.empty());
  CHECK(kls[1]->args[0].value_kind == 3);
  CHECK(kls[1]->args[0].ptr_offsets == std::vector<uint16_t>{8});

  auto [rc, out] = replay(archive);
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
  // launch 4 finds what kernel 3 wrote. Launches 2 and 3 find what the host
  // left.
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 6);
  CHECK(rejected == 0);
}

// ---------------------------------------------------------------------------
// A buffer that does not change is stored once.
//
// The manifest counts the chunks capture copied out. Four more unchanged
// reads add four launches with two records each and no copy. The same four
// reads with one int of chunk 0 bumped each time copy that chunk each time,
// which shows the count follows what the snapshots hold.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_UnchangedStoredOnce) {
  auto capture = [](const char* name,
                    std::vector<std::pair<std::string, std::string>> env,
                    long long& chunks, size_t& launches, size_t& records) {
    ScopedDir cap(fs::temp_directory_path() / name);
    capture_pinned(cap.path, env);
    const fs::path archive = hrr_single_process_archive(cap.path);
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(archive.string(), arc));
    const auto kls = launches_of(arc);
    chunks = manifest_count(read_text_file(archive / "manifest.json"),
                            "host_snapshot_chunks");
    launches = kls.size();
    records = snapshot_records(kls);
  };

  // Launches 0 and 1 copy both chunks; launches 2 to 4 find the bytes of the
  // last copy, launch 4 because the host put them back.
  long long same_chunks = 0;
  size_t same_launches = 0, same_records = 0;
  capture("hrr_pinned_once_same.hrr", {{"HRR_PINNED_EXTRA_READS", "4"}},
          same_chunks, same_launches, same_records);
  CHECK(same_launches == kBaseLaunches + 4);
  CHECK(same_records == (kBaseLaunches + 4) * 2);
  CHECK(same_chunks == 4);

  long long diff_chunks = 0;
  size_t diff_launches = 0, diff_records = 0;
  capture("hrr_pinned_once_diff.hrr",
          {{"HRR_PINNED_EXTRA_READS", "4"}, {"HRR_PINNED_EXTRA_CHANGE", "1"}},
          diff_chunks, diff_launches, diff_records);
  CHECK(diff_launches == kBaseLaunches + 4);
  CHECK(diff_records == (kBaseLaunches + 4) * 2);
  CHECK(diff_chunks == 4 + 4);
}

// ---------------------------------------------------------------------------
// HIP_HRR_HOST_SNAPSHOTS=0 records no pinned contents, says so in the
// manifest, and replay then reads an unfilled buffer. The struct launch is
// left out: its pointer is not translated either, and replay would hand the
// GPU the capture process's address.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_OptOut) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_optout.hrr");
  capture_pinned(cap.path, {{"HIP_HRR_HOST_SNAPSHOTS", "0"}, {"HRR_PINNED_SKIP_VIEW", "1"}});
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest.find("\"host_snapshots\": false") != std::string::npos);
  CHECK(manifest_count(manifest, "host_snapshot_chunks") == -1);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches - 1);
  CHECK(snapshot_records(kls) == 0);

  auto [rc, out] = replay(archive);
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
// Capture marks a pinned address inside a by-value argument as a pointer only
// when the launch recorded that allocation, so a scalar that happens to hold
// such an address is not rewritten. With HIP_HRR_HOST_SNAPSHOT_MAX_MB=0 nothing
// is recorded: the manifest says why, the struct argument stays a plain value,
// and HIP_HRR_REPLAY_AUDIT_HOST_ARGS names the word replay passes on as it is.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_UnrecordedStructAudited) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_audit.hrr");
  capture_pinned(cap.path, {{"HIP_HRR_HOST_SNAPSHOT_MAX_MB", "0"}});
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest.find("\"host_snapshots\": true") != std::string::npos);
  CHECK(manifest.find("\"pinned host snapshot\"") != std::string::npos);
  CHECK(manifest.find("HIP_HRR_HOST_SNAPSHOT_MAX_MB") != std::string::npos);
  CHECK(manifest_count(manifest, "host_snapshot_chunks") == 0);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches);
  CHECK(snapshot_records(kls) == 0);
  REQUIRE(!kls[1]->args.empty());
  CHECK(kls[1]->args[0].value_kind == 0);

  // The kernel may fault on the untranslated address; the audit line is
  // printed before the launch.
  auto [rc, out] = replay(archive, "",
                                       {{"HIP_HRR_REPLAY_AUDIT_HOST_ARGS", "1"}});
  INFO("Replay rc " << rc << ":\n" << out);
  CHECK(out.find("arg[0]+8 holds") != std::string::npos);
  CHECK(out.find("but capture did not mark it as a pointer") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Archive contents are untrusted. A record whose range does not fit its
// allocation, whose length does not match its blob, whose pointer names no
// allocation or a device one, or whose direction is unknown is refused by
// name and nothing is written for it. So are the records of a launch that
// claims more of them than it holds, or whose attribute tail does not fit.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_MalformedRecordRejected) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_malformed.hrr");
  capture_pinned(cap.path, {{"HRR_PINNED_EXTRA_READS", "2"}});
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches + 2);

  std::vector<uint8_t> events = read_bytes(archive / "events.bin");
  const auto spans = launch_spans(events);
  REQUIRE(spans.size() == kls.size());

  // The recorded address of the device output buffer: launch 0's arg 1.
  REQUIRE(kls[0]->args.size() >= 2);
  REQUIRE(kls[0]->args[1].data.size() == 8);
  uint64_t device_ptr = 0;
  std::memcpy(&device_ptr, kls[0]->args[1].data.data(), 8);

  // Launch 0: a length past the end, and an offset that wraps when added.
  patch_snapshot(events, spans, kls, 0, 0, /*length*/ 2, 1ull << 40);
  patch_snapshot(events, spans, kls, 0, 1, /*offset*/ 1, ~0ull - 4095);
  // Launch 1: 8 bytes that would run past the allocation end.
  patch_snapshot(events, spans, kls, 1, 1, /*offset*/ 1, kPinnedBytes - 8);
  // Launch 2: in bounds, but shorter than its blob.
  patch_snapshot(events, spans, kls, 2, 0, /*length*/ 2, kChunk - 8);
  // Launch 3: a direction no capture writes, and a device allocation.
  patch_snapshot(events, spans, kls, 3, 0, /*direction*/ 5, 2);
  patch_snapshot(events, spans, kls, 3, 1, /*ptr*/ 0, device_ptr);
  // Launch 4: a pointer no allocation covers.
  patch_snapshot(events, spans, kls, 4, 0, /*ptr*/ 0, 0x10);
  // Launch 5: more records than the event holds.
  {
    const uint16_t lie = 1000;
    std::memcpy(events.data() + num_snapshots_at(events, spans[5]), &lie, 2);
  }
  // Launch 6: an attribute tail longer than the event. A plain launch writes
  // a zero count, so the tail is the 8 bytes before the records.
  {
    const size_t tail = spans[6].second - kls[6]->snapshots.size() * 41 - 8;
    const uint32_t n_attrs = 1000, stride = 24;
    std::memcpy(events.data() + tail, &n_attrs, 4);
    std::memcpy(events.data() + tail + 4, &stride, 4);
  }
  write_bytes(archive / "events.bin", events);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  // Rejected restores leave reads on the wrong bytes, so the D2H checks may
  // fail; the replay itself must finish.
  CHECK(rc < 128);

  CHECK(count_of(out, "is out of bounds of its allocation") == 3);
  CHECK(count_of(out, "does not match the size of its blob") == 1);
  CHECK(count_of(out, "names no live allocation") == 1);
  CHECK(count_of(out, "has an unknown direction") == 1);
  CHECK(count_of(out, "does not name host memory") == 1);
  CHECK(count_of(out, "run past the end of the event") == 1);
  CHECK(count_of(out, "launch attribute tail is malformed") == 1);

  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(rejected == 7);
}

// ---------------------------------------------------------------------------
// Replay leaves a direction-1 record's bytes alone, and still treats the
// allocation it names as recorded.
//
// Launches 0 and 1 are rewritten to direction 1, so replay restores neither
// and both reads see the wrong bytes. Launch 1's struct pointer must still be
// translated: the record names its allocation.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_UnchangedRecordLeftAlone) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_dir1.hrr");
  capture_pinned(cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == kBaseLaunches);

  std::vector<uint8_t> events = read_bytes(archive / "events.bin");
  const auto spans = launch_spans(events);
  REQUIRE(spans.size() == kls.size());
  for (size_t launch : {0, 1})
    for (size_t rec : {0, 1}) patch_snapshot(events, spans, kls, launch, rec, 5, 1);
  write_bytes(archive / "events.bin", events);

  auto [rc, out] = replay(archive, "--verbose");
  INFO("Replay:\n" << out);
  CHECK(rc < 128);
  CHECK(out.find("embedded ptr @+8 unresolved") == std::string::npos);

  // Launch 2 finds the buffer replay never filled and launch 4 finds what
  // kernel 3 wrote: two chunks each.
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 4);
  CHECK(rejected == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_fail >= 2);
#endif
}

// ---------------------------------------------------------------------------
// A launch queued behind work that waits on the host does not hang capture.
//
// The snapshot is taken at once: the chunk the host changed is recorded for
// replay to restore, the unchanged one as direction 1, and the manifest counts
// the launch as unordered. Replay of hipStreamWaitValue32 does not wait, so the
// replay runs through and every read sees the captured bytes.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_NoWaitOnBusyStream) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_nowait.hrr");
  capture_case("Unit_HRR_PinnedHost_NoWait_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest_count(manifest, "host_snapshots_unordered") == 1);
  CHECK(manifest_count(manifest, "host_snapshot_chunks") == 3);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 2);
  REQUIRE(kls[0]->snapshots.size() == 2);
  CHECK(kls[0]->snapshots[0].direction == 0);
  CHECK(kls[0]->snapshots[1].direction == 0);
  REQUIRE(kls[1]->snapshots.size() == 2);
  CHECK(kls[1]->snapshots[0].direction == 0);  // the host changed it
  CHECK(kls[1]->snapshots[1].direction == 1);  // unchanged, stream busy
  CHECK(kls[1]->snapshots[1].hash_lo == kls[0]->snapshots[1].hash_lo);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 3);
  CHECK(rejected == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 2);
  CHECK(d2h_fail == 0);
#endif
}

// ---------------------------------------------------------------------------
// The two size limits.
//
// HIP_HRR_HOST_SNAPSHOT_MAX_MB skips an allocation larger than it, and
// HIP_HRR_HOST_SNAPSHOT_TOTAL_MB stops recording new allocations once their
// shadow copies would pass it. Either way the manifest says what was skipped.
// Two 1.5 MiB buffers fit the defaults, only the first fits a 2 MiB total, and
// neither fits a 1 MiB maximum.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_SizeLimits) {
  constexpr size_t kChunks = 1536 * 1024 / kChunk;
  auto capture = [](const char* name,
                    std::vector<std::pair<std::string, std::string>> env,
                    std::string& manifest, std::vector<size_t>& records) {
    ScopedDir cap(fs::temp_directory_path() / name);
    capture_case("Unit_HRR_PinnedHost_Sizes_Direct", cap.path, env);
    const fs::path archive = hrr_single_process_archive(cap.path);
    manifest = read_text_file(archive / "manifest.json");
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(archive.string(), arc));
    records.clear();
    for (const auto* kl : launches_of(arc)) records.push_back(kl->snapshots.size());
  };

  std::string manifest;
  std::vector<size_t> records;
  {
    capture("hrr_pinned_sizes_default.hrr", {}, manifest, records);
    INFO("defaults; manifest:\n" << manifest);
    CHECK(records == std::vector<size_t>{kChunks, kChunks});
    CHECK(manifest.find("\"pinned host snapshot\"") == std::string::npos);
  }
  {
    capture("hrr_pinned_sizes_total.hrr", {{"HIP_HRR_HOST_SNAPSHOT_TOTAL_MB", "2"}},
            manifest, records);
    INFO("TOTAL_MB=2; manifest:\n" << manifest);
    CHECK(records == std::vector<size_t>{kChunks, 0});
    CHECK(manifest.find("HIP_HRR_HOST_SNAPSHOT_TOTAL_MB") != std::string::npos);
  }
  {
    capture("hrr_pinned_sizes_max.hrr", {{"HIP_HRR_HOST_SNAPSHOT_MAX_MB", "1"}},
            manifest, records);
    INFO("MAX_MB=1; manifest:\n" << manifest);
    CHECK(records == std::vector<size_t>{0, 0});
    CHECK(manifest.find("HIP_HRR_HOST_SNAPSHOT_MAX_MB") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// Every allocation and free entry point, and a registered range read through
// its device pointer. Each read replays with the bytes the host wrote, single-
// and multi-threaded, so each allocator is tracked, each free forgets its
// buffer, and a new buffer at a reused address starts afresh.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_Lifetime) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_lifetime.hrr");
  capture_case("Unit_HRR_PinnedHost_Lifetime_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 7);
  for (size_t k = 0; k < kls.size(); ++k) {
    INFO("launch " << k);
    REQUIRE(kls[k]->snapshots.size() == 2);
    CHECK(kls[k]->snapshots[0].direction == 0);
  }
  // The device pointer and the host pointer name the same recorded bytes.
  CHECK(kls[4]->snapshots[0].ptr_handle == kls[5]->snapshots[0].ptr_handle);
  CHECK(same_hashes(kls[4], kls[5]));

  for (const char* mode : {"", "--multi-thread"}) {
    auto [rc, out] = replay(archive, mode);
    INFO("Replay " << mode << ":\n" << out);
    CHECK(rc == 0);
    unsigned long long restored = 0, rejected = 0;
    host_snapshot_summary(out, restored, rejected);
    CHECK(restored >= 12);
    CHECK(rejected == 0);
#ifndef _WIN32
    int d2h_pass = 0, d2h_fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
    CHECK(d2h_pass >= 7);
    CHECK(d2h_fail == 0);
#endif
  }
}

// ---------------------------------------------------------------------------
// Launch entry points: hipLaunchKernel_spt, hipModuleLaunchKernel with the
// arguments in `extra`, and a handle that is not a kernel, which gets the
// runtime's error rather than a crash in capture.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_LaunchApis) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_apis.hrr");
  capture_case("Unit_HRR_PinnedHost_LaunchApis_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  // The reader decodes the hipModuleLaunchKernel event but not the
  // hipLaunchKernel_spt one, which replay still runs; the replay's restore
  // count below covers both.
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 1);
  CHECK(kls[0]->snapshots.size() == 2);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  // Two chunks for each launch: the first finds an unfilled buffer, the
  // second finds pattern 1.
  CHECK(restored == 4);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 2);
  CHECK(d2h_fail == 0);
#endif
}

// ---------------------------------------------------------------------------
// hipLaunchByPtr resolves its kernel from the stub before the launch, and the
// launch carries the snapshot. Not replayed; see the workload.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_LaunchByPtr) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_byptr.hrr");
  capture_case("Unit_HRR_PinnedHost_ByPtr_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 1);
  REQUIRE(kls[0]->snapshots.size() == 2);
  CHECK(kls[0]->snapshots[0].offset == 0);
  CHECK(kls[0]->snapshots[1].offset == kChunk);
}

// ---------------------------------------------------------------------------
// A launch into a stream under graph capture is not snapshotted: the graph
// runs it later, when the host may have changed the bytes again. The manifest
// says so. Replay applies no record to a kernel it replays into a graph
// capture, and says so; a record is spliced onto the captured launch to show
// it.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_GraphCapture) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_graph.hrr");
  capture_case("Unit_HRR_PinnedHost_Graph_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest.find("under graph capture") != std::string::npos);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 2);
  REQUIRE(kls[0]->snapshots.size() == 2);
  CHECK(kls[1]->snapshots.empty());

  // Copy launch 0's first record onto launch 1.
  std::vector<uint8_t> events = read_bytes(archive / "events.bin");
  const auto spans = launch_spans(events);
  REQUIRE(spans.size() == 2);
  const size_t rec = record_at(events, spans, kls, 0, 0);
  const std::vector<uint8_t> record(events.begin() + rec, events.begin() + rec + 41);
  const size_t n_at = num_snapshots_at(events, spans[1]);
  uint16_t n = 0;
  std::memcpy(&n, events.data() + n_at, 2);
  REQUIRE(n == 0);
  n = 1;
  std::memcpy(events.data() + n_at, &n, 2);
  hrr_event_header hdr;
  std::memcpy(&hdr, events.data() + spans[1].first, sizeof(hdr));
  hdr.payload_length += 41;
  std::memcpy(events.data() + spans[1].first, &hdr, sizeof(hdr));
  events.insert(events.begin() + spans[1].second, record.begin(), record.end());
  write_bytes(archive / "events.bin", events);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc < 128);
  CHECK(out.find("pinned host snapshots are not applied to kernels replayed into a "
                 "graph capture") != std::string::npos);
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
