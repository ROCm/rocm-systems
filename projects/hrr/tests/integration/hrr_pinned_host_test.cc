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

#include <hip/hip_ext.h>  // hipExtModuleLaunchKernel
#include <hip/hiprtc.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#endif

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

// A workload that cannot run on this device writes why to the file
// HRR_PINNED_SKIP_FILE names, so its parent test can skip with that reason,
// then returns.
void skip_direct(const std::string& why) {
  WARN(why);
  if (const char* path = std::getenv("HRR_PINNED_SKIP_FILE")) {
    std::ofstream(path) << why;
  }
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

// hrr_pinned_read over a grid of any size, for the cooperative launches,
// whose whole grid has to be resident at once.
__global__ void hrr_pinned_read_strided(const int* in, int* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = in[i] * 3 + 1;
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
//   -  the same with a handle hipKernelGetFunction returned for a pointer that
//      is not a kernel
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

  // The same through hipKernelGetFunction, which hands back any pointer it is
  // given as a function handle and reports success.
  hipFunction_t from_kernel = nullptr;
  const hipError_t got = hipKernelGetFunction(&from_kernel, reinterpret_cast<hipKernel_t>(0x20));
  INFO("hipKernelGetFunction: " << hipGetErrorName(got));
  REQUIRE(got == hipSuccess);
  const hipError_t r = hipModuleLaunchKernel(from_kernel, 0, 1, 1, kThreads, 1, 1, 0,
                                             nullptr, args, nullptr);
  INFO("launch of its handle: " << hipGetErrorName(r));
  CHECK((r == hipErrorInvalidValue || r == hipErrorInvalidConfiguration));
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

// ===========================================================================
// A launch that fails after its snapshot, then a launch on a busy stream.
//
//   0  read      host filled pattern 1, stream idle
//   -  read      host filled pattern 2; the block is too large, so the launch
//                fails and nothing is recorded for it
//   1  read      queued behind hipStreamWaitValue32, as in NoWait; the host
//                has not touched the buffer since the failed launch
// The failed launch's snapshot already saw pattern 2. If capture kept it,
// launch 1 would find nothing changed and record both chunks as unchanged,
// and replay would run launch 1 on pattern 1.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_FailedLaunch_Direct", "[.][hrr-direct]") {
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

  // The failed launch.
  fill(h, 2);
  int n = kPinnedInts;
  void* args[] = {&h, &out, &n};
  const hipError_t bad = hipLaunchKernel(reinterpret_cast<const void*>(hrr_pinned_read),
                                         dim3(1), dim3(4096), args, 0, s);
  INFO("oversized block: " << hipGetErrorName(bad));
  REQUIRE(bad != hipSuccess);
  (void)hipGetLastError();

  // 1
  HRR_HIP_CHECK(hipStreamWaitValue32(s, flag, 1, hipStreamWaitValueEq, 0xFFFFFFFF));
  hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, s, h, out,
                     kPinnedInts);
  HRR_HIP_CHECK(hipGetLastError());
  CHECK(hipStreamQuery(s) == hipErrorNotReady);
  __atomic_store_n(flag, 1u, __ATOMIC_SEQ_CST);
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  check_out(out, [](int i) { return pattern(2, i) * 3 + 1; });

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(flag));
  HRR_HIP_CHECK(hipHostFree(h));
}

namespace {
#define HRR_PINNED_RTC_CHECK(expr)    \
  do {                                \
    const hiprtcResult r_ = (expr);   \
    REQUIRE(r_ == HIPRTC_SUCCESS);    \
  } while (0)

// The module launches need a kernel loaded through the module path.
const char* kPinnedRtcSource = R"(
extern "C" __global__ void hrr_rtc_pinned_read(const int* in, int* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = in[i] * 3 + 1;
}
)";

std::vector<char> compile_pinned_rtc() {
  hiprtcProgram prog = nullptr;
  HRR_PINNED_RTC_CHECK(hiprtcCreateProgram(&prog, kPinnedRtcSource, "hrr_pinned_rtc.hip", 0,
                                           nullptr, nullptr));
  HRR_PINNED_RTC_CHECK(hiprtcCompileProgram(prog, 0, nullptr));
  size_t size = 0;
  HRR_PINNED_RTC_CHECK(hiprtcGetCodeSize(prog, &size));
  std::vector<char> code(size);
  HRR_PINNED_RTC_CHECK(hiprtcGetCode(prog, code.data()));
  HRR_PINNED_RTC_CHECK(hiprtcDestroyProgram(&prog));
  return code;
}

// Grid of the entry-point launches. The kernels stride over the buffer, so
// any grid reads all of it.
constexpr int kEntryBlocks = 64;

// Grid of the cooperative launches: kEntryBlocks, or fewer when a small
// device cannot hold that many kThreads blocks at once. It covers the
// strided kernel and, if fn is set, the module kernel. 0 if not even one
// block fits per CU.
int coop_entry_blocks(hipFunction_t fn) {
  int cus = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
  int per_cu = 0;
  HRR_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_cu, reinterpret_cast<const void*>(hrr_pinned_read_strided), kThreads, 0));
  if (fn != nullptr) {
    int fn_per_cu = 0;
    HRR_HIP_CHECK(
        hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&fn_per_cu, fn, kThreads, 0));
    per_cu = std::min(per_cu, fn_per_cu);
  }
  return std::min(kEntryBlocks, per_cu * cus);
}
}  // namespace

// ===========================================================================
// The launch entry points LaunchApis does not reach, each on a fresh pattern,
// then a free that fails.
//
//   0  hipModuleLaunchKernel            on a hipModuleGetFunction handle,
//                                       its first launch
//   1  hipModuleLaunchCooperativeKernel
//   2  hipExtModuleLaunchKernel         (grid in work items)
//   3  hipDrvLaunchKernelEx
//   4  hipLaunchCooperativeKernel       by host stub
//   5  hipLaunchCooperativeKernel_spt
//   6  hipLaunchKernelExC
//   7  hipLaunchKernelGGL on a registered range after hipHostFree refused it;
//      capture must still know the range
// Every launch reads the whole buffer, which holds new bytes each time. The
// cooperative launches take the largest grid up to kEntryBlocks the device
// holds at once; a device that holds none skips the case.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_EntryPoints_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int coop = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0));
  REQUIRE(coop != 0);

  const std::vector<char> code = compile_pinned_rtc();
  hipModule_t mod = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod, code.data()));
  hipFunction_t fn = nullptr;
  HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod, "hrr_rtc_pinned_read"));
  const void* stub = reinterpret_cast<const void*>(hrr_pinned_read_strided);
  const int coop_blocks = coop_entry_blocks(fn);
  std::printf("cooperative grid: %d blocks\n", coop_blocks);
  if (coop_blocks == 0) {
    HRR_HIP_CHECK(hipModuleUnload(mod));
    skip_direct("a cooperative launch cannot hold one block of " + std::to_string(kThreads) +
                " threads per CU on this device");
    return;
  }

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  int n = kPinnedInts;
  void* args[] = {&h, &out, &n};
  auto expect = [](int which) {
    return [which](int i) { return pattern(which, i) * 3 + 1; };
  };

  // 0
  fill(h, 20);
  HRR_HIP_CHECK(hipModuleLaunchKernel(fn, kEntryBlocks, 1, 1, kThreads, 1, 1, 0, nullptr,
                                      args, nullptr));
  check_out(out, expect(20));

  // 1
  fill(h, 21);
  HRR_HIP_CHECK(hipModuleLaunchCooperativeKernel(fn, coop_blocks, 1, 1, kThreads, 1, 1, 0,
                                                 nullptr, args));
  check_out(out, expect(21));

  // 2
  fill(h, 22);
  HRR_HIP_CHECK(hipExtModuleLaunchKernel(fn, kEntryBlocks * kThreads, 1, 1, kThreads, 1, 1,
                                         0, nullptr, args, nullptr, nullptr, nullptr, 0));
  check_out(out, expect(22));

  // 3
  fill(h, 23);
  {
    HIP_LAUNCH_CONFIG cfg{};
    cfg.gridDimX = kEntryBlocks;
    cfg.gridDimY = 1;
    cfg.gridDimZ = 1;
    cfg.blockDimX = kThreads;
    cfg.blockDimY = 1;
    cfg.blockDimZ = 1;
    cfg.hStream = nullptr;
    HRR_HIP_CHECK(hipDrvLaunchKernelEx(&cfg, fn, args, nullptr));
  }
  check_out(out, expect(23));

  // 4
  fill(h, 24);
  HRR_HIP_CHECK(hipLaunchCooperativeKernel(stub, dim3(coop_blocks), dim3(kThreads), args, 0,
                                           nullptr));
  check_out(out, expect(24));

  // 5
  fill(h, 25);
  HRR_HIP_CHECK(hipLaunchCooperativeKernel_spt(stub, dim3(coop_blocks), dim3(kThreads), args,
                                               0, nullptr));
  check_out(out, expect(25));

  // 6
  fill(h, 26);
  {
    hipLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kEntryBlocks);
    cfg.blockDim = dim3(kThreads);
    cfg.stream = nullptr;
    HRR_HIP_CHECK(hipLaunchKernelExC(&cfg, stub, args));
  }
  check_out(out, expect(26));

  // 7
  void* reg = std::aligned_alloc(4096, kPinnedBytes);
  REQUIRE(reg != nullptr);
  HRR_HIP_CHECK(hipHostRegister(reg, kPinnedBytes, hipHostRegisterDefault));
  const hipError_t refused = hipHostFree(reg);
  INFO("hipHostFree of a registered range: " << hipGetErrorName(refused));
  REQUIRE(refused != hipSuccess);
  (void)hipGetLastError();
  fill(static_cast<int*>(reg), 27);
  read_pinned(static_cast<const int*>(reg), out, 27);
  HRR_HIP_CHECK(hipHostUnregister(reg));
  std::free(reg);

  HRR_HIP_CHECK(hipModuleUnload(mod));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// hipDeviceReset in the middle of a capture.
//
//   0  read      pinned buffer A, eight times the usual size, pattern 30
//   -  hipDeviceReset, which releases A
//   1  read      a new pinned buffer B, pattern 31
//   2  a launch whose pointer argument still holds an address inside A that
//      neither B nor anything else the runtime knows covers, with nothing
//      to read. Capture must have forgotten A: otherwise it reads the
//      unmapped bytes, or records whatever is mapped there now.
// Captured only: events after a reset replay against a reset device (see
// Unit_HRR_ApiMatrix_Reset_Direct).
//
// On POSIX the workload maps A's addresses with no access as soon as the
// reset frees them, so B and the runtime cannot reuse them, and a capture
// that still reads A crashes. Where the reset leaves A mapped, it looks for
// an address in A the runtime does not know and B does not cover, and skips
// the case if there is none.
// ===========================================================================
constexpr size_t kResetABytes = 8 * kPinnedBytes;
// Addresses in A the workload tries as the stale pointer.
constexpr size_t kResetProbes = 16;

TEST_CASE("Unit_HRR_PinnedHost_Reset_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* a = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&a), kResetABytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));

  // 0
  fill(a, 30);
  read_pinned(a, out, 30);

  HRR_HIP_CHECK(hipDeviceReset());
  bool held = false;
#ifndef _WIN32
  // MAP_FIXED_NOREPLACE fails if any of A is still mapped. A kernel that
  // does not know the flag treats the address as a hint instead.
  void* hold = mmap(a, kResetABytes, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
  held = hold == a;
  if (hold != MAP_FAILED && !held) munmap(hold, kResetABytes);
#endif
  HRR_HIP_CHECK(hipSetDevice(0));

  // 1
  int* b = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&b), kPinnedBytes,
                              hipHostMallocDefault));
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  fill(b, 31);
  read_pinned(b, out, 31);

  // 2
  const auto lo = reinterpret_cast<uintptr_t>(b);
  int* probe = nullptr;
  std::ostringstream probes;
  probes << "A " << a << ", B " << b << ", A held after reset: " << held;
  for (size_t k = 0; k < kResetProbes; ++k) {
    const size_t off = k * (kResetABytes / kResetProbes);
    int* p = a + off / sizeof(int);
    const auto at = reinterpret_cast<uintptr_t>(p);
    hipPointerAttribute_t attr{};
    const bool known = hipPointerGetAttributes(&attr, p) == hipSuccess &&
                       attr.type != hipMemoryTypeUnregistered;
    (void)hipGetLastError();
    const bool in_b = at >= lo && at < lo + kPinnedBytes;
    probes << "\n  A+" << off << ": known " << known << ", in B " << in_b;
    if (probe == nullptr && !known && !in_b) probe = p;
  }
  std::printf("%s\n", probes.str().c_str());
  if (probe == nullptr) {
    skip_direct("after hipDeviceReset every probed address in the old buffer is known "
                "to the runtime or inside the new one:\n" + probes.str());
  } else {
    hipLaunchKernelGGL(hrr_pinned_read, dim3(1), dim3(kThreads), 0, nullptr, probe, out, 0);
    HRR_HIP_CHECK(hipGetLastError());
    HRR_HIP_CHECK(hipDeviceSynchronize());
  }

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(b));
#ifndef _WIN32
  if (held) munmap(a, kResetABytes);
#endif
}

namespace {
// The device pointer makes the argument value_kind 3; the pinned one is at
// byte offset 8.
struct PinnedPair {
  int*       out;
  const int* in;
  int        scale;
};

// word, at byte offset 9, is not on an 8-byte boundary.
struct __attribute__((packed)) PinnedPackedWord {
  int*     out;
  uint8_t  tag;
  uint64_t word;
};
}  // namespace

__global__ void hrr_pinned_read_pair(PinnedPair v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v.out[i] = v.in[i] * v.scale - 2;
}

// Writes v.word after the n results, low half first.
__global__ void hrr_pinned_read_packed(PinnedPackedWord v, const int* in, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v.out[i] = in[i] * 3 + 1;
  if (i == 0) {
    const uint64_t w = v.word;
    v.out[n]     = static_cast<int>(static_cast<uint32_t>(w));
    v.out[n + 1] = static_cast<int>(static_cast<uint32_t>(w >> 32));
  }
}

// ===========================================================================
// A pinned pointer inside a by-value struct next to a device pointer.
//
//   0  write     the device fills the pinned buffer, so replay's buffer gets
//                the same bytes without a snapshot
//   1  read_pair the struct carries the device output and the pinned input
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_StructPair_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));

  // 0
  hipLaunchKernelGGL(hrr_pinned_write, dim3(kBlocks), dim3(kThreads), 0, nullptr,
                     h, kPinnedInts, 0x3c3c);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(h[1] == (0x3c3c ^ 1));

  // 1
  hipLaunchKernelGGL(hrr_pinned_read_pair, dim3(kBlocks), dim3(kThreads), 0, nullptr,
                     PinnedPair{out, h, 5}, kPinnedInts);
  HRR_HIP_CHECK(hipGetLastError());
  check_out(out, [](int i) { return (0x3c3c ^ i) * 5 - 2; });

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// A scalar that equals an address inside a recorded pinned allocation.
//
//   0  read_packed  reads the pinned buffer through its pointer argument, so
//                   the launch records it; the struct's unaligned word holds
//                   an address 400 bytes into the same buffer, and the kernel
//                   copies that word out after its results
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_PackedScalar_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, (kPinnedInts + 2) * sizeof(int)));
  fill(h, 3);

  const uint64_t word = reinterpret_cast<uint64_t>(h + 100);
  hipLaunchKernelGGL(hrr_pinned_read_packed, dim3(kBlocks), dim3(kThreads), 0, nullptr,
                     PinnedPackedWord{out, 7, word}, static_cast<const int*>(h), kPinnedInts);
  HRR_HIP_CHECK(hipGetLastError());
  check_out(out,
            [word](int i) {
              if (i < kPinnedInts) return pattern(3, i) * 3 + 1;
              const uint64_t half = i == kPinnedInts ? word : word >> 32;
              return static_cast<int>(static_cast<uint32_t>(half));
            },
            kPinnedInts + 2);

  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// hipModuleEnumerateFunctions into an array longer than the module's kernels.
//
//   0  read      through the one handle the runtime wrote, its first launch
//   -  the entries past it still hold what the application put there, which
//      is not a kernel; each is launched with a zero grid, so the runtime
//      refuses it, and capture must not read it first
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_Enumerate_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  int n = kPinnedInts;
  void* args[] = {&h, &out, &n};

  const std::vector<char> code = compile_pinned_rtc();
  hipModule_t mod = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod, code.data()));
  unsigned int count = 0;
  HRR_HIP_CHECK(hipModuleGetFunctionCount(&count, mod));
  REQUIRE(count == 1);
  // The SDK this suite builds against may neither declare nor export it; the
  // runtime it runs on does.
  using EnumerateFn = hipError_t (*)(hipFunction_t*, unsigned int, hipModule_t);
  void* sym = nullptr;
  HRR_HIP_CHECK(hipGetProcAddress("hipModuleEnumerateFunctions", &sym, HIP_VERSION, 0, nullptr));
  REQUIRE(sym != nullptr);
  const auto enumerate = reinterpret_cast<EnumerateFn>(sym);
  hipFunction_t fns[4];
  for (uintptr_t k = 0; k < 4; ++k) fns[k] = reinterpret_cast<hipFunction_t>(0x10 * (k + 1));
  HRR_HIP_CHECK(enumerate(fns, 4, mod));
  REQUIRE(fns[0] != reinterpret_cast<hipFunction_t>(0x10));

  // 0
  fill(h, 40);
  HRR_HIP_CHECK(hipModuleLaunchKernel(fns[0], kEntryBlocks, 1, 1, kThreads, 1, 1, 0, nullptr,
                                      args, nullptr));
  check_out(out, [](int i) { return pattern(40, i) * 3 + 1; });

  for (int k = 1; k < 4; ++k) {
    INFO("entry " << k << " left as " << reinterpret_cast<void*>(fns[k]));
    REQUIRE(fns[k] == reinterpret_cast<hipFunction_t>(0x10 * (k + 1)));
    const hipError_t r = hipModuleLaunchKernel(fns[k], 0, 1, 1, kThreads, 1, 1, 0, nullptr,
                                               args, nullptr);
    INFO("launch: " << hipGetErrorName(r));
    CHECK((r == hipErrorInvalidValue || r == hipErrorInvalidConfiguration));
    (void)hipGetLastError();
  }

  HRR_HIP_CHECK(hipModuleUnload(mod));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// A handle capture does not trust until it has launched.
//
//   0  hipModuleLaunchKernel on a handle from hipLibraryGetKernel and
//      hipKernelGetFunction, neither of which capture trusts; pattern 50
//   1  the same handle again, pattern 51
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_LibraryKernel_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  int n = kPinnedInts;
  void* args[] = {&h, &out, &n};

  const std::vector<char> code = compile_pinned_rtc();
  hipLibrary_t lib = nullptr;
  HRR_HIP_CHECK(hipLibraryLoadData(&lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  hipKernel_t kernel = nullptr;
  HRR_HIP_CHECK(hipLibraryGetKernel(&kernel, lib, "hrr_rtc_pinned_read"));
  hipFunction_t fn = nullptr;
  HRR_HIP_CHECK(hipKernelGetFunction(&fn, kernel));

  for (int which : {50, 51}) {
    fill(h, which);
    HRR_HIP_CHECK(hipModuleLaunchKernel(fn, kEntryBlocks, 1, 1, kThreads, 1, 1, 0, nullptr,
                                        args, nullptr));
    check_out(out, [which](int i) { return pattern(which, i) * 3 + 1; });
  }

  HRR_HIP_CHECK(hipLibraryUnload(lib));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// A read whose own stream is idle, queued behind a write into its buffer on a
// stream it waits for.
//
//   0  read      host filled pattern 1, everything idle
//   1  write     the buffer, seed 0x3c3c, held on the writer stream by
//                hipStreamWaitValue32 on a flag the host sets only after
//                launch 2 returns
//   2  read      on the launch stream, which has nothing queued of its own;
//                the launch waits on the GPU for the writer stream
// HRR_PINNED_ORDER picks the two streams:
//   0  launch on a hipStreamCreate stream, write on the null stream
//   1  launch on hipStreamPerThread, which is blocking; write on the null
//      stream
//   2  launch on hipStreamLegacy, write on a hipStreamCreate stream
// The writer is a kernel because capture synchronizes the stream of a D2H
// hipMemcpyAsync to record its bytes, which would never return here.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_OtherStream_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int can_wait = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, 0));
  REQUIRE(can_wait != 0);
  const int order = env_int("HRR_PINNED_ORDER");
  REQUIRE((order >= 0 && order <= 2));

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  uint32_t* flag = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(uint32_t),
                              hipHostMallocMapped));
  *flag = 0;
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  hipStream_t blocking = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&blocking));
  const hipStream_t launch = order == 0   ? blocking
                             : order == 1 ? hipStreamPerThread
                                          : hipStreamLegacy;
  const hipStream_t writer = order == 2 ? blocking : nullptr;
  constexpr int kSeed = 0x3c3c;

  // 0
  fill(h, 1);
  read_pinned(h, out, 1, kPinnedInts, launch);

  // 1. Errors are checked only after the flag is set: a check that failed
  // first would leave the writer stream blocked, and the process would hang in
  // teardown instead of failing.
  const hipError_t wait_err =
      hipStreamWaitValue32(writer, flag, 1, hipStreamWaitValueEq, 0xFFFFFFFF);
  hipLaunchKernelGGL(hrr_pinned_write, dim3(kBlocks), dim3(kThreads), 0, writer, h,
                     kPinnedInts, kSeed);
  const hipError_t write_err = hipGetLastError();

  // 2
  hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, launch, h, out,
                     kPinnedInts);
  const hipError_t read_err = hipGetLastError();
  const hipError_t query = hipStreamQuery(launch);
  const int h0 = h[0];
  __atomic_store_n(flag, 1u, __ATOMIC_SEQ_CST);
  HRR_HIP_CHECK(wait_err);
  HRR_HIP_CHECK(write_err);
  HRR_HIP_CHECK(read_err);
  // Still blocked: neither kernel has run.
  CHECK(query == hipErrorNotReady);
  CHECK(h0 == pattern(1, 0));
  check_out(out, [](int i) { return (kSeed ^ i) * 3 + 1; });

  HRR_HIP_CHECK(hipStreamDestroy(blocking));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(flag));
  HRR_HIP_CHECK(hipHostFree(h));
}

// Spins for `ticks` of the wall clock, then writes seed ^ i over buf and
// raises *done once the writes are visible to the host. One block.
__global__ void hrr_pinned_slow_write(int* buf, int n, int seed, unsigned long long ticks,
                                      unsigned int* done) {
  if (threadIdx.x == 0) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] = seed ^ i;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) *reinterpret_cast<volatile unsigned int*>(done) = 1;
}

// ===========================================================================
// A read on a blocking stream after a null-stream kernel the host waited for
// without a HIP call.
//
//   0  read      on a hipStreamCreate stream, host filled pattern 1
//   1  write     on the null stream: spins for kSpinMs, then writes seed
//                0x4d4d over the buffer and raises a done flag in pinned
//                memory
//   2  read      on the hipStreamCreate stream, after the host saw the flag
//                with plain loads and set h[0] = -7
// No HIP call orders launch 2 after launch 1 on the host, so replay reaches
// launch 2 while launch 1 still spins, and only the launch's own wait on the
// null stream orders the two on the GPU.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_PolledNull_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int kSpinMs = 500;
  constexpr int kSeed   = 0x4d4d;
  int rate_khz = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
  REQUIRE(rate_khz > 0);
  const unsigned long long ticks = static_cast<unsigned long long>(rate_khz) * kSpinMs;

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  unsigned int* done = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&done), sizeof(unsigned int),
                              hipHostMallocMapped));
  *done = 0;
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // 0
  fill(h, 1);
  read_pinned(h, out, 1, kPinnedInts, s);

  // 1
  hipLaunchKernelGGL(hrr_pinned_slow_write, dim3(1), dim3(kThreads), 0, nullptr, h,
                     kPinnedInts, kSeed, ticks, done);
  HRR_HIP_CHECK(hipGetLastError());
  while (__atomic_load_n(done, __ATOMIC_ACQUIRE) == 0) {
  }
  REQUIRE(h[1] == (kSeed ^ 1));

  // 2
  h[0] = -7;
  hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, s, h, out,
                     kPinnedInts);
  HRR_HIP_CHECK(hipGetLastError());
  check_out(out, [](int i) { return (i == 0 ? -7 : kSeed ^ i) * 3 + 1; });

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(done));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// A read whose launch stream waits for a null-stream hipStreamBatchMemOp wait
// that a later call releases, all on one host thread.
//
//   -  hipStreamBatchMemOp on the null stream: wait for a device flag to be 1
//   0  read      on a hipStreamCreate stream, host filled pattern 1; the launch
//                waits on the GPU for the null stream
//   -  hipStreamWriteValue32 on a non-blocking stream sets the flag to 1
// Replay of hipStreamBatchMemOp waits for real. A restore that waited on the
// host for the null stream before the write was replayed would never return.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_BatchWait_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int can_wait = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, 0));
  REQUIRE(can_wait != 0);

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  uint32_t* flag = nullptr;
  HRR_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&flag), sizeof(uint32_t)));
  HRR_HIP_CHECK(hipMemset(flag, 0, sizeof(uint32_t)));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  hipStream_t release = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&release, hipStreamNonBlocking));

  fill(h, 1);
  hipStreamBatchMemOpParams op{};
  op.operation = hipStreamMemOpWaitValue32;
  op.waitValue.operation = hipStreamMemOpWaitValue32;
  op.waitValue.address = reinterpret_cast<hipDeviceptr_t>(flag);
  op.waitValue.value = 1;
  op.waitValue.flags = hipStreamWaitValueEq;
  const hipError_t wait_err = hipStreamBatchMemOp(nullptr, 1, &op, 0);

  // 0
  hipLaunchKernelGGL(hrr_pinned_read, dim3(kBlocks), dim3(kThreads), 0, s, h, out,
                     kPinnedInts);
  const hipError_t read_err = hipGetLastError();

  const hipError_t release_err = hipStreamWriteValue32(release, flag, 1, 0);
  HRR_HIP_CHECK(wait_err);
  HRR_HIP_CHECK(read_err);
  HRR_HIP_CHECK(release_err);
  check_out(out, [](int i) { return pattern(1, i) * 3 + 1; });

  HRR_HIP_CHECK(hipStreamDestroy(release));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(flag));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(h));
}

// ===========================================================================
// A launch capture looks at while a blocking stream waits for the host, then
// a kernel on another blocking stream that the host waits for before it
// releases the first one.
//
//   -  hipStreamWaitValue32 on blocking stream A, on a flag the host sets last
//   -  the launch HRR_PINNED_VARIANT picks, reading pinned memory filled with
//      pattern 1:
//        0  hipLaunchKernel_spt into hipStreamLegacy
//        1  hipLaunchCooperativeKernel_spt into hipStreamLegacy
//        2  hipLaunchKernel into the null stream with an oversized block,
//           which fails its checks before it reaches a stream
//   -  a write into device memory on blocking stream B, then
//      hipStreamSynchronize(B)
//   -  the host sets the flag
//   -  variant 2 only: a read on the null stream
// The per-thread launches use the per-thread stream, which waits for the null
// stream only, and the failed launch waits for nothing. None of them orders B
// after A. Capture that queued a null-stream wait for A while it looked at the
// launch would hold B behind A, and the synchronize would never return.
// ===========================================================================
TEST_CASE("Unit_HRR_PinnedHost_NoNullBarrier_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int can_wait = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, 0));
  REQUIRE(can_wait != 0);
  const int variant = env_int("HRR_PINNED_VARIANT");
  REQUIRE((variant >= 0 && variant <= 2));
  int coop_blocks = 0;
  if (variant == 1) {
    int coop = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0));
    REQUIRE(coop != 0);
    coop_blocks = coop_entry_blocks(nullptr);
    if (coop_blocks == 0) {
      skip_direct("a cooperative launch cannot hold one block of " +
                  std::to_string(kThreads) + " threads per CU on this device");
      return;
    }
  }

  int* h = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), kPinnedBytes,
                              hipHostMallocDefault));
  uint32_t* flag = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(uint32_t),
                              hipHostMallocMapped));
  *flag = 0;
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kPinnedBytes));
  int* scratch = nullptr;
  HRR_HIP_CHECK(hipMalloc(&scratch, kPinnedBytes));
  hipStream_t a = nullptr;
  hipStream_t b = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&a));
  HRR_HIP_CHECK(hipStreamCreate(&b));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  fill(h, 1);
  int n = kPinnedInts;
  void* args[] = {&h, &out, &n};
  const hipError_t wait_err =
      hipStreamWaitValue32(a, flag, 1, hipStreamWaitValueEq, 0xFFFFFFFF);
  hipError_t launch_err = hipSuccess;
  if (variant == 0) {
    launch_err = hipLaunchKernel_spt(reinterpret_cast<const void*>(hrr_pinned_read),
                                     dim3(kBlocks), dim3(kThreads), args, 0, hipStreamLegacy);
  } else if (variant == 1) {
    launch_err = hipLaunchCooperativeKernel_spt(
        reinterpret_cast<const void*>(hrr_pinned_read_strided), dim3(coop_blocks),
        dim3(kThreads), args, 0, hipStreamLegacy);
  } else {
    launch_err = hipLaunchKernel(reinterpret_cast<const void*>(hrr_pinned_read), dim3(1),
                                 dim3(4096), args, 0, nullptr);
    (void)hipGetLastError();
  }
  hipLaunchKernelGGL(hrr_pinned_write, dim3(kBlocks), dim3(kThreads), 0, b, scratch,
                     kPinnedInts, 0x2e2e);
  const hipError_t write_err = hipGetLastError();
  const hipError_t sync_err = hipStreamSynchronize(b);
  __atomic_store_n(flag, 1u, __ATOMIC_SEQ_CST);

  HRR_HIP_CHECK(wait_err);
  HRR_HIP_CHECK(write_err);
  HRR_HIP_CHECK(sync_err);
  INFO("launch: " << hipGetErrorName(launch_err));
  if (variant == 2) {
    REQUIRE(launch_err != hipSuccess);
    read_pinned(h, out, 1);
  } else {
    REQUIRE(launch_err == hipSuccess);
    check_out(out, [](int i) { return pattern(1, i) * 3 + 1; });
  }

  HRR_HIP_CHECK(hipStreamDestroy(b));
  HRR_HIP_CHECK(hipStreamDestroy(a));
  HRR_HIP_CHECK(hipFree(scratch));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipHostFree(flag));
  HRR_HIP_CHECK(hipHostFree(h));
}

namespace {
constexpr const char* kDirect = "Unit_HRR_PinnedHost_Direct";
// A capture or replay that takes longer than this has hung.
constexpr int kCaptureTimeoutSec = 120;

// Returns why the workload skipped itself, or an empty string if it ran.
std::string capture_case(const char* direct_case, const fs::path& cap,
                         const std::vector<std::pair<std::string, std::string>>& env = {}) {
  const fs::path skip_file = cap.string() + ".skip";
  fs::remove(skip_file);
  hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  proc.setEnv("HRR_PINNED_SKIP_FILE", skip_file.string());
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(std::string("\"") + direct_case + "\"",
                                      kCaptureTimeoutSec);
  INFO("Capture of " << direct_case << " exit: " << ret
       << (ret == hrr::test::SpawnProc::kKilledOnTimeout ? " (hung, killed)" : "")
       << "\nWorkload output:\n" << proc.getOutput());
  REQUIRE(ret == 0);
  if (std::strstr(direct_case, "EntryPoints") || std::strstr(direct_case, "Reset") ||
      std::strstr(direct_case, "NoNullBarrier")) {
    WARN("cross-diag: " << direct_case << " skip file " << fs::exists(skip_file)
         << "\nWorkload output:\n" << proc.getOutput());
  }
  if (!fs::exists(skip_file)) return {};
  std::string why = read_text_file(skip_file);
  fs::remove(skip_file);
  return why.empty() ? std::string("the workload skipped itself") : why;
}

// Replay with stderr merged and byte-exact D2H checks. The default checks
// accept any buffer that matches as floats within 1e-3. Most ints these
// kernels write are below 2^23, float denormals, so wrong output would pass.
// The deadline turns a hang into a failure.
std::pair<int, std::string> replay(
    const fs::path& archive, const std::string& extra_args = "",
    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  hrr::test::SpawnProc proc(hrr_playback_exe(), /*capture_stdout=*/true,
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

// The first "[HRR]   Host snapshots : ..." line that matches fmt to its end,
// which must hold n %llu conversions. False when no line does.
bool host_snapshot_line(const std::string& out, const char* fmt, int n,
                        unsigned long long* a, unsigned long long* b = nullptr) {
  const std::string f = std::string(fmt) + "%n";
  for (size_t at = out.find("Host snapshots :"); at != std::string::npos;
       at = out.find("Host snapshots :", at + 1)) {
    std::string line = out.substr(at, out.find('\n', at) - at);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    int end = -1;
    const int got = n == 2 ? std::sscanf(line.c_str(), f.c_str(), a, b, &end)
                           : std::sscanf(line.c_str(), f.c_str(), a, &end);
    if (got == n && end == static_cast<int>(line.size())) return true;
  }
  return false;
}

// "[HRR]   Host snapshots : N chunk(s) ... restored, M record(s) rejected".
// Absent when replay restored and rejected nothing.
void host_snapshot_summary(const std::string& out, unsigned long long& restored,
                           unsigned long long& rejected) {
  restored = rejected = 0;
  const bool found = host_snapshot_line(out,
                                        "Host snapshots : %llu chunk(s) of pinned host memory "
                                        "restored, %llu record(s) rejected",
                                        2, &restored, &rejected);
  if (!found) restored = rejected = 0;
}

// "[HRR]   Host snapshots : N record(s) not applied, ... graph capture".
// Absent when there were none.
unsigned long long host_snapshots_in_graph(const std::string& out) {
  unsigned long long n = 0;
  return host_snapshot_line(out,
                            "Host snapshots : %llu record(s) not applied, their launches "
                            "replayed into a graph capture",
                            1, &n)
             ? n
             : 0;
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

// Whether one line of out holds both a and b.
bool line_with(const std::string& out, const std::string& a, const std::string& b) {
  for (size_t at = out.find(a); at != std::string::npos; at = out.find(a, at + 1)) {
    const size_t bol = out.rfind('\n', at);
    const size_t from = bol == std::string::npos ? 0 : bol + 1;
    const size_t eol = out.find('\n', at);
    if (out.substr(from, eol == std::string::npos ? std::string::npos : eol - from)
            .find(b) != std::string::npos)
      return true;
  }
  return false;
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

  // The replay first: the record checks below stop at the first launch without
  // records, and the reads are what this test is about.
  {
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
// HIP_HRR_HOST_SNAPSHOTS=0 takes no snapshot, says so in the manifest, and
// replay then reads an unfilled buffer. The struct launch is left out: its
// struct holds no device pointer, so its pinned one is not translated either,
// and replay would hand the GPU the capture process's address.
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
// A launch with no record for a pinned allocation leaves its pointers to
// replay's rescan, as before snapshots existed. Capture marks only the struct's
// device pointer; the rescan finds the pinned one at offset 8 and rewrites it.
// Snapshots off and an allocation over the cap both leave the launch without
// records.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_UnrecordedStructTranslated) {
  const std::vector<std::pair<std::string, std::string>> envs[] = {
      {{"HIP_HRR_HOST_SNAPSHOTS", "0"}}, {{"HIP_HRR_HOST_SNAPSHOT_MAX_MB", "0"}}};
  for (const auto& env : envs) {
    INFO(env[0].first << "=" << env[0].second);
    ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_pair.hrr");
    capture_case("Unit_HRR_PinnedHost_StructPair_Direct", cap.path, env);
    const fs::path archive = hrr_single_process_archive(cap.path);

    hrr::Archive arc;
    REQUIRE(hrr::load_archive(archive.string(), arc));
    const auto kls = launches_of(arc);
    REQUIRE(kls.size() == 2);
    CHECK(snapshot_records(kls) == 0);
    REQUIRE(!kls[1]->args.empty());
    CHECK(kls[1]->args[0].value_kind == 3);
    CHECK(kls[1]->args[0].ptr_offsets == std::vector<uint16_t>{0});

    auto [rc, out] = replay(archive, "--verbose");
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    CHECK(line_with(out, "arg[0]: embedded ptr @+8 0x", "[rescan]"));
#ifndef _WIN32
    int d2h_pass = 0, d2h_fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
    CHECK(d2h_pass >= 1);
    CHECK(d2h_fail == 0);
#endif
  }
}

// ---------------------------------------------------------------------------
// A word that capture did not mark, inside a by-value argument, into a pinned
// allocation the launch recorded, is a scalar: capture compared the
// 8-byte-aligned words of 8-byte-aligned arguments against that allocation, and
// this one is unaligned. Replay passes it on unchanged, and the kernel copies it
// out for the D2H check.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_RecordedScalarKept) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_scalar.hrr");
  capture_case("Unit_HRR_PinnedHost_PackedScalar_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 1);
  CHECK(kls[0]->snapshots.size() == 2);
  REQUIRE(!kls[0]->args.empty());
  CHECK(kls[0]->args[0].value_kind == 3);
  CHECK(kls[0]->args[0].ptr_offsets == std::vector<uint16_t>{0});

  auto [rc, out] = replay(archive, "--verbose");
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  CHECK(line_with(out, "arg[0]: embedded ptr @+0 0x", "[captured]"));
  CHECK(out.find("embedded ptr @+9") == std::string::npos);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 1);
  CHECK(d2h_fail == 0);
#endif
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

  // Seven records fail a check of their own. Launch 5's claimed thousand and
  // launch 6's two cannot be located, and count too.
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(rejected == 7 + 1000 + kls[6]->snapshots.size());
  CHECK(host_snapshots_in_graph(out) == 0);
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
  CHECK(line_with(out, "arg[0]: embedded ptr @+8 0x", "[captured]"));

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
// A launch whose own stream is idle but which waits for another stream is
// unordered too.
//
// A launch into a blocking stream, the per-thread stream among them, waits
// for the null stream; a launch into hipStreamLegacy waits for every blocking
// stream. Launch 2 has nothing queued on its own stream, while the write it
// waits for is held behind a flag. Capture must count it unordered and record
// its unchanged chunks as direction 1: recorded for restore, they would hold
// the bytes from before the write, and replay would put those back over what
// the replayed write left. The write itself is unordered as well. Replay
// restores launch 0's two chunks and nothing else, and the read sees the
// written bytes.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_WaitsForOtherStream) {
  for (const char* order : {"0", "1", "2"}) {
    INFO("HRR_PINNED_ORDER=" << order);
    ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_other_stream.hrr");
    capture_case("Unit_HRR_PinnedHost_OtherStream_Direct", cap.path,
                 {{"HRR_PINNED_ORDER", order}});
    const fs::path archive = hrr_single_process_archive(cap.path);

    const std::string manifest = read_text_file(archive / "manifest.json");
    INFO("manifest:\n" << manifest);
    CHECK(manifest_count(manifest, "host_snapshots_unordered") == 2);
    CHECK(manifest_count(manifest, "host_snapshot_chunks") == 2);

    hrr::Archive arc;
    REQUIRE(hrr::load_archive(archive.string(), arc));
    const auto kls = launches_of(arc);
    REQUIRE(kls.size() == 3);
    for (size_t k : {1, 2}) {
      INFO("launch " << k);
      REQUIRE(kls[k]->snapshots.size() == 2);
      CHECK(kls[k]->snapshots[0].direction == 1);
      CHECK(kls[k]->snapshots[1].direction == 1);
      CHECK(same_hashes(kls[0], kls[k]));
    }

    auto [rc, out] = replay(archive);
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    unsigned long long restored = 0, rejected = 0;
    host_snapshot_summary(out, restored, rejected);
    CHECK(restored == 2);
    CHECK(rejected == 0);
#ifndef _WIN32
    int d2h_pass = 0, d2h_fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
    CHECK(d2h_pass >= 2);
    CHECK(d2h_fail == 0);
#endif
  }
}

// ---------------------------------------------------------------------------
// The restore of a launch into a blocking stream waits for the null stream.
//
// The host waited for the null-stream kernel by polling a flag, which records
// nothing, so replay reaches launch 2 while that kernel still spins. Launch 2
// waits for it on the GPU, and the kernel then writes over the whole buffer: a
// restore of h[0] = -7 made before the kernel finished is lost. Capture saw
// the kernel's bytes, so both chunks are recorded for restore whether or not
// the null stream had retired the kernel by then.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_NullStreamDrainedFirst) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_polled_null.hrr");
  capture_case("Unit_HRR_PinnedHost_PolledNull_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 3);
  REQUIRE(kls[2]->snapshots.size() == 2);
  CHECK(kls[2]->snapshots[0].direction == 0);
  CHECK(kls[2]->snapshots[1].direction == 0);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(rejected == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 2);
  CHECK(d2h_fail == 0);
#endif
}

// ---------------------------------------------------------------------------
// The restore waits on the launch stream, not on the host.
//
// The launch stream is held on the GPU by a null-stream hipStreamBatchMemOp
// wait that only a later replayed hipStreamWriteValue32 releases. A restore
// that waited on the host for the null stream to drain would hang the replay
// before it reached the release; the deadline turns that into a failure.
// Capture saw the stream busy and the bytes new, so both chunks are recorded
// for restore, and the read must see them.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_RestoreWaitsOnStream) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_batch_wait.hrr");
  capture_case("Unit_HRR_PinnedHost_BatchWait_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest_count(manifest, "host_snapshots_unordered") == 1);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 1);
  REQUIRE(kls[0]->snapshots.size() == 2);
  CHECK(kls[0]->snapshots[0].direction == 0);
  CHECK(kls[0]->snapshots[1].direction == 0);

  auto [rc, out] = replay(archive);
  INFO("Replay exit: " << rc
       << (rc == hrr::test::SpawnProc::kKilledOnTimeout ? " (hung, killed)" : "")
       << "\nReplay:\n" << out);
  REQUIRE(rc == 0);
  // The wait was replayed, so the launch stream really was held.
  CHECK(out.find("hipStreamBatchMemOp: op address") == std::string::npos);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 2);
  CHECK(rejected == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 1);
  CHECK(d2h_fail == 0);
#endif
}

namespace {
// Capture and replay of Unit_HRR_PinnedHost_NoNullBarrier_Direct with one
// variant. A capture that holds stream B behind stream A never returns from
// the workload's hipStreamSynchronize(B), and is killed at the deadline.
void no_null_barrier(const char* variant) {
  INFO("HRR_PINNED_VARIANT=" << variant);
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_no_null_barrier.hrr");
  const std::string skipped = capture_case("Unit_HRR_PinnedHost_NoNullBarrier_Direct",
                                           cap.path, {{"HRR_PINNED_VARIANT", variant}});
  if (!skipped.empty()) HRR_SKIP(skipped);
  const fs::path archive = hrr_single_process_archive(cap.path);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 2);
  CHECK(rejected == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 1);
  CHECK(d2h_fail == 0);
#endif
}
}  // namespace

// ---------------------------------------------------------------------------
// A per-thread launch into hipStreamLegacy goes to the per-thread stream, and
// capture asks about that stream. Asking the null stream instead queued a
// null-stream wait for every blocking stream, which held an unrelated blocking
// stream behind one waiting for the host.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_SptLegacyNoBarrier) { no_null_barrier("0"); }

// The same through hipLaunchCooperativeKernel_spt.
HRR_TEST_CASE(Unit_HRR_PinnedHost_SptCoopLegacyNoBarrier) { no_null_barrier("1"); }

// ---------------------------------------------------------------------------
// A null-stream launch that fails its checks leaves nothing queued: capture
// finds out whether the blocking streams are busy without the null-stream
// query, whose wait for them would stay behind on the null stream.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_FailedNullLaunchNoBarrier) { no_null_barrier("2"); }

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
    CHECK(restored == 12);
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
// hipModuleEnumerateFunctions makes known only the handles the runtime wrote.
// The one real kernel is snapshotted on its first launch; the entries past
// it get the runtime's error rather than a crash in capture.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_EnumerateFunctions) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_enum.hrr");
  capture_case("Unit_HRR_PinnedHost_Enumerate_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 1);
  CHECK(kls[0]->snapshots.size() == 2);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 1);
  CHECK(d2h_fail == 0);
#endif
}

// ---------------------------------------------------------------------------
// A handle from hipKernelGetFunction is not read before its first launch, so
// that launch gets no snapshot. The launch succeeds, which proves the handle
// is a kernel, and the second launch is snapshotted.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_TrustedAfterLaunch) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_libkernel.hrr");
  capture_case("Unit_HRR_PinnedHost_LibraryKernel_Direct", cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 2);
  CHECK(kls[0]->snapshots.empty());
  REQUIRE(kls[1]->snapshots.size() == 2);
  CHECK(kls[1]->snapshots[0].direction == 0);
  CHECK(kls[1]->snapshots[1].direction == 0);
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
// it. A second spliced record, shorter than its blob, is still checked and
// rejected: not being applied does not exempt a record from the checks.
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

  // Copy launch 0's first record onto launch 1, then the same record with a
  // length 8 bytes short of its blob.
  std::vector<uint8_t> events = read_bytes(archive / "events.bin");
  const auto spans = launch_spans(events);
  REQUIRE(spans.size() == 2);
  const size_t rec = record_at(events, spans, kls, 0, 0);
  std::vector<uint8_t> records(events.begin() + rec, events.begin() + rec + 41);
  records.insert(records.end(), events.begin() + rec, events.begin() + rec + 41);
  const uint64_t short_len = kChunk - 8;
  std::memcpy(records.data() + 41 + 16, &short_len, 8);
  const size_t n_at = num_snapshots_at(events, spans[1]);
  uint16_t n = 0;
  std::memcpy(&n, events.data() + n_at, 2);
  REQUIRE(n == 0);
  n = 2;
  std::memcpy(events.data() + n_at, &n, 2);
  hrr_event_header hdr;
  std::memcpy(&hdr, events.data() + spans[1].first, sizeof(hdr));
  hdr.payload_length += records.size();
  std::memcpy(events.data() + spans[1].first, &hdr, sizeof(hdr));
  events.insert(events.begin() + spans[1].second, records.begin(), records.end());
  write_bytes(archive / "events.bin", events);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc < 128);
  CHECK(out.find("pinned host snapshots are not applied to kernels replayed into a "
                 "graph capture") != std::string::npos);
  // The valid record is counted on a line of its own, not as rejected.
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(rejected == 1);
  CHECK(count_of(out, "does not match the size of its blob") == 1);
  CHECK(host_snapshots_in_graph(out) == 1);
}

// ---------------------------------------------------------------------------
// A launch that fails after capture snapshotted it leaves nothing behind. The
// next launch, on a busy stream, finds the bytes changed since launch 0 and
// records both chunks to restore. Replay runs it on pattern 2.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_FailedLaunch) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_failed.hrr");
  capture_case("Unit_HRR_PinnedHost_FailedLaunch_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 4);
  CHECK(rejected == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= 2);
  CHECK(d2h_fail == 0);
#endif

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest_count(manifest, "host_snapshots_unordered") == 1);
  CHECK(manifest_count(manifest, "host_snapshot_chunks") == 4);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 2);
  REQUIRE(kls[1]->snapshots.size() == 2);
  CHECK(kls[1]->snapshots[0].direction == 0);
  CHECK(kls[1]->snapshots[1].direction == 0);
}

// ---------------------------------------------------------------------------
// The cooperative, Ext, DrvLaunchKernelEx and LaunchKernelExC launches, a
// hipModuleGetFunction handle on its first launch, and a registered range
// after a failed hipHostFree. Each launch reads bytes the previous one did
// not see, so replay restores both chunks of every launch.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_EntryPoints) {
  constexpr int kLaunches = 8;
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_entry.hrr");
  const std::string skipped = capture_case("Unit_HRR_PinnedHost_EntryPoints_Direct", cap.path);
  if (!skipped.empty()) HRR_SKIP(skipped);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest:\n" << manifest);
  CHECK(manifest_count(manifest, "host_snapshot_chunks") == 2 * kLaunches);

  // The reader decodes the module, Ext and hipLaunchKernel events only.
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  for (const auto* kl : launches_of(arc)) {
    INFO("launch of " << kl->kernel_name);
    CHECK(kl->snapshots.size() == 2);
  }

  auto [rc, out] = replay(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  unsigned long long restored = 0, rejected = 0;
  host_snapshot_summary(out, restored, rejected);
  CHECK(restored == 2 * kLaunches);
  CHECK(rejected == 0);
#ifndef _WIN32
  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  CHECK(d2h_pass >= kLaunches);
  CHECK(d2h_fail == 0);
#endif
}

// ---------------------------------------------------------------------------
// After hipDeviceReset capture records the new buffer and forgets the old
// one, so a stale pointer to it is not read.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_PinnedHost_DeviceReset) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_pinned_reset.hrr");
  const std::string skipped = capture_case("Unit_HRR_PinnedHost_Reset_Direct", cap.path);
  if (!skipped.empty()) HRR_SKIP(skipped);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));
  const auto kls = launches_of(arc);
  REQUIRE(kls.size() == 3);
  CHECK(kls[0]->snapshots.size() == kResetABytes / kChunk);
  REQUIRE(kls[1]->snapshots.size() == 2);
  CHECK(kls[1]->snapshots[0].direction == 0);
  CHECK(kls[1]->snapshots[1].direction == 0);
  CHECK(kls[2]->snapshots.empty());
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
