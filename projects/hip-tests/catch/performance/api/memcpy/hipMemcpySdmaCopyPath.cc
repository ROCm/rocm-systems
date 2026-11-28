/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTestMemory
 */

namespace {

// SDMA_PKT_POLL_REGMEM::DW5_UNION.interval, i.e. how long the SDMA engine waits between
// reads of the polled address. ROCr latches HSA_SDMA_POLL_INTERVAL once at runtime init,
// so it has to be set before the process starts; printing it here keeps each result line
// self-describing instead of relying on the operator's notes.
const char* PollInterval() {
  const char* env = std::getenv("HSA_SDMA_POLL_INTERVAL");
  return (env == nullptr || *env == '\0') ? "4(default)" : env;
}

// Who actually waits out a dependency signal. On the WDDM native-SDMA path libhsakmt leaves the
// POLL_REGMEM packet in the ring for the engine only when HSA_ENABLE_SDMA_GPU_POLL=1; by default
// SdmaThread waits on the host and strips the packet, so the engine never polls at all. Printed
// because it decides whether a long-poll run measured the engine or a host thread.
const char* PollOwner() {
  const char* env = std::getenv("HSA_ENABLE_SDMA_GPU_POLL");
  const bool gpu = (env != nullptr) && (env[0] == '1') && (env[1] == '\0');
  return gpu ? "sdma-engine" : "host-thread(default)";
}

constexpr size_t kMaxBytes = 8 * 1024 * 1024;
constexpr int kMinIters = 200;
constexpr int kMaxIters = 200000;

// How long the compute kernel occupies the GPU, and so how long the SDMA engine has to keep
// polling that kernel's completion signal. 2 s by default so the test stays cheap in CI; the
// KMD scenario this was written for is a two-minute poll, i.e. KERNEL_SLEEP=120000.
constexpr int kDefaultKernelSleepMs = 2000;

int KernelSleepMs() {
  const char* env = std::getenv("KERNEL_SLEEP");
  if (env == nullptr || *env == '\0') return kDefaultKernelSleepMs;
  const int ms = std::atoi(env);
  return (ms > 0) ? ms : kDefaultKernelSleepMs;
}

// Byte the buffers hold before the kernel runs. Distinct from anything FillValue() produces, so
// a copy that did not wait is visible as the stale byte rather than as plausible data.
constexpr unsigned char kStaleByte = 0xA5;

__host__ __device__ inline unsigned char FillValue(size_t i) {
  return static_cast<unsigned char>((i * 7u + 13u) & 0xFFu);
}

// Sleeps first and writes only afterwards. That ordering is what turns the data check into an
// ordering check: every byte of the pattern lands in the last instants of the kernel, so a copy
// that started early reads kStaleByte. Launched single-threaded, so the delay and the writes
// are strictly ordered; a wider launch would need every thread to delay, or the threads that
// skipped it would write before the sleep and defeat the check.
// wall_clock64() rather than the clock_function() macro: the tick budget below is derived from
// hipDeviceAttributeWallClockRate, and clock_function() becomes the shader clock under
// TEST_CLOCK_CYCLE, which runs at a different and power-state-dependent rate. Pairing the two
// would make the sleep miss its target by whatever the DPM state happens to be. There is no
// 32-bit wall_clock(); the 64-bit counter is the only constant-rate clock HIP exposes.
__global__ void SleepThenFill(unsigned char* out, size_t bytes, unsigned int ms,
                              unsigned int ticks_per_ms) {
  // One deadline for the whole sleep, not a loop of per-millisecond waits: each wait overshoots
  // its target by however long the last s_sleep ran, and ms of those accumulate into a sleep
  // that is a large multiple of the one requested.
  const long long total_ticks = static_cast<long long>(ms) * static_cast<long long>(ticks_per_ms);
  const long long start = wall_clock64();
  while (wall_clock64() - start < total_ticks) {
    __builtin_amdgcn_s_sleep(10);
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    out[i] = FillValue(i);
  }
}

unsigned int WallClockTicksPerMs() {
  int device = 0;
  int clock_khz = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate, device));
  // A device that reports 0 would make the delay loop spin forever, so fall back to 1 MHz.
  return (clock_khz > 0) ? static_cast<unsigned int>(clock_khz) : 1000u;
}

// Roughly fixed total bytes per configuration so every size costs about the same wall time.
size_t TargetBytesPerConfig() {
  return isQuickLevel() ? (512ull * 1024 * 1024) : (2ull * 1024 * 1024 * 1024);
}

void Measure(const char* dir, hipMemcpyKind kind, void* dst, const void* src, size_t bytes,
             hipStream_t stream) {
  int iters = static_cast<int>(TargetBytesPerConfig() / bytes);
  if (iters < kMinIters) iters = kMinIters;
  if (iters > kMaxIters) iters = kMaxIters;

  // Warm up outside the measurement: queue creation and lazy blit setup would swamp the A/B.
  const int warmup = (iters / 20 < 100) ? 100 : iters / 20;
  for (int i = 0; i < warmup; ++i) HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  const double total = static_cast<double>(iters) * static_cast<double>(bytes);
  const double gib_s = (secs > 0.0) ? total / secs / (1024.0 * 1024.0 * 1024.0) : 0.0;
  const double us = (iters > 0) ? secs * 1e6 / iters : 0.0;
  std::printf("[SDMA-PERF] poll=%-10s dir=%-8s size=%7zuKiB iters=%6d time=%6.3fs "
              "bw=%7.2f GiB/s per_copy=%9.2fus\n",
              PollInterval(), dir, bytes / 1024, iters, secs, gib_s, us);
  std::fflush(stdout);
}


// Shared body. cross_stream selects how the copy is made to depend on the kernel: inherited
// from sharing a stream, or stated explicitly with an event. Two independent streams is the
// one arrangement neither test uses - there the copy carries wait_event=0x0, no POLL_REGMEM
// is emitted at all, and it moves the stale byte.
void RunLongPoll(bool cross_stream) {
  // One byte. The transfer is not what is under test - the dependency signal is - so keep the
  // copy itself far below the poll it is waiting on, otherwise the time spent moving data shows
  // up in the elapsed figure and blunts the duration check.
  constexpr size_t kBytes = 1 * 1;
  const int sleep_ms = KernelSleepMs();
  const unsigned int ticks_per_ms = WallClockTicksPerMs();

  unsigned char* h_buf = nullptr;
  void* d_src = nullptr;
  void* d_dst = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_buf), kBytes));
  HIP_CHECK(hipMalloc(&d_src, kBytes));
  HIP_CHECK(hipMalloc(&d_dst, kBytes));

  // Both buffers start stale, so neither a skipped kernel nor a premature copy can leave the
  // expected pattern in the destination by accident.
  HIP_CHECK(hipMemset(d_src, kStaleByte, kBytes));
  HIP_CHECK(hipMemset(d_dst, kStaleByte, kBytes));
  HIP_CHECK(hipDeviceSynchronize());

  const char* arrangement = cross_stream ? "cross-stream(event)" : "same-stream";

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipStream_t copy_stream = stream;
  hipStream_t second = nullptr;
  hipEvent_t kernel_done = nullptr;
  if (cross_stream) {
    HIP_CHECK(hipStreamCreate(&second));
    HIP_CHECK(hipEventCreateWithFlags(&kernel_done, hipEventDisableTiming));
    copy_stream = second;
  }

  const auto start = std::chrono::steady_clock::now();
  SleepThenFill<<<1, 1, 0, stream>>>(static_cast<unsigned char*>(d_src), kBytes,
                                        static_cast<unsigned int>(sleep_ms), ticks_per_ms);
  HIP_CHECK(hipGetLastError());
  if (cross_stream) {
    // Record on the kernel's stream and have the copy's stream wait on it: the ordering the
    // copy would otherwise get for free from sharing a stream.
    HIP_CHECK(hipEventRecord(kernel_done, stream));
    HIP_CHECK(hipStreamWaitEvent(copy_stream, kernel_done, 0));
  }
  // Either way the copy carries the kernel's dependency, which is what makes the SDMA engine
  // poll for the whole sleep.
  HIP_CHECK(hipMemcpyAsync(d_dst, d_src, kBytes, hipMemcpyDeviceToDeviceNoCU, copy_stream));
  // Not HIP_CHECK: past the GPU watchdog this fails as a bare hipErrorIllegalAddress, which
  // reads like a bug in the copy rather than the kernel being killed for running too long.
  const hipError_t sync_status = hipStreamSynchronize(copy_stream);
  if (sync_status != hipSuccess) {
    std::printf("[SDMA-POLL] hipStreamSynchronize failed after a %d ms kernel: %s.\n"
                "            On stock Windows the GPU watchdog (TdrDelay, 2 s by default)"
                " terminates it; measured here, %d ms survives and %d ms does not. For the"
                " long-poll case raise or disable TDR under\n"
                "            HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers"
                " (TdrDelay / TdrLevel) and reboot.\n",
                sleep_ms, hipGetErrorString(sync_status), 2500, 4000);
    std::fflush(stdout);
  }
  INFO("hipStreamSynchronize after a " << sleep_ms << " ms kernel: "
                                       << hipGetErrorString(sync_status));
  REQUIRE(sync_status == hipSuccess);
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

  std::printf("[SDMA-POLL] poll=%-10s polled_by=%-21s order=%-19s"
              " kernel_sleep=%6dms"
              " elapsed=%9.1fms size=%zuB\n",
              PollInterval(), PollOwner(), arrangement, sleep_ms, elapsed_ms, kBytes);
  std::fflush(stdout);

  // The copy landed the kernel's output, not the stale byte, so the poll held the copy back for
  // the whole kernel and the transfer itself was intact.
  std::memset(h_buf, 0, kBytes);
  HIP_CHECK(hipMemcpy(h_buf, d_dst, kBytes, hipMemcpyDeviceToHost));
  size_t mismatches = 0;
  size_t first_bad = kBytes;
  size_t stale = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    if (h_buf[i] != FillValue(i)) {
      if (mismatches == 0) first_bad = i;
      ++mismatches;
      if (h_buf[i] == kStaleByte) ++stale;
    }
  }
  INFO("mismatches=" << mismatches << " first_bad=" << first_bad << " still_stale=" << stale
                     << " (all-stale means the copy ran before the kernel finished)");
  REQUIRE(mismatches == 0);

  // The kernel really did run for about as long as asked, so the poll it forced was long too
  // rather than the dependency being resolved early. Only a lower bound: the host timer also
  // covers launch and the copy, and a slow wall clock makes the kernel overshoot, never undershoot.
  REQUIRE(elapsed_ms >= 0.9 * static_cast<double>(sleep_ms));

  if (cross_stream) {
    HIP_CHECK(hipEventDestroy(kernel_done));
    HIP_CHECK(hipStreamDestroy(second));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipHostFree(h_buf));
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Times hipMemcpyAsync in every direction that can reach an SDMA engine and prints one
 * [SDMA-PERF] line each.
 *
 * NOTE: the queue-path A/B this test was originally written for is gone. ROCr picks the native
 * SDMA user queue from device capability, and HSA_ENABLE_SDMA_USER_QUEUE no longer exists in
 * clr, so there is no path to select or report.
 *
 * The live knob is HSA_SDMA_POLL_INTERVAL, which sets DW5_UNION.interval of the SDMA
 * POLL_REGMEM packet (how long the engine waits between reads of a dependency signal).
 * ROCr latches it once at runtime init, so it must be set before the process starts:
 *
 *     set HSA_SDMA_POLL_INTERVAL=0
 *     MemcpyPerformance.exe Performance_hipMemcpy_SdmaCopyPath -c "512 KiB"
 *
 * Only copies that carry a dependency signal emit a POLL_REGMEM packet, so the interval
 * affects the first copy after an engine switch rather than every copy in a burst.
 *
 * H2D, D2H and D2D do not share a path inside ROCr: H2D and D2H use different SDMA engines,
 * while same-device D2D goes to the blit kernel unless hipMemcpyDeviceToDeviceNoCU forces it
 * onto SDMA, so D2D is listed under both kinds. A direction whose numbers do not move between
 * the two runs is not on the SDMA user-queue path at all.
 *
 * The sizes straddle ROCr's force_sdma_size (1 MiB), below which D2D goes to SDMA and above
 * which it goes to the blit kernel, so a step there is a dispatch change rather than a
 * regression. Use a section filter for one size per run:
 *
 *     MemcpyPerformance.exe Performance_hipMemcpy_SdmaCopyPath -c "512 KiB"
 *
 * These copies are latency-bound from ~128 KiB to ~1 MiB, so compare per-copy microseconds,
 * not GiB/s. ctest hides stdout for passing tests; run the exe directly or use ctest -V.
 * ------------------------
 *    - catch\performance\api\memcpy\hipMemcpySdmaCopyPath.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Performance_hipMemcpy_SdmaCopyPath) {
  size_t bytes = 0;
  SECTION("16 KiB") { bytes = 16 * 1024; }
  SECTION("32 KiB") { bytes = 32 * 1024; }
  SECTION("64 KiB") { bytes = 64 * 1024; }
  SECTION("128 KiB") { bytes = 128 * 1024; }
  SECTION("256 KiB") { bytes = 256 * 1024; }
  SECTION("512 KiB") { bytes = 512 * 1024; }
  SECTION("1 MiB") { bytes = 1024 * 1024; }
  SECTION("2 MiB") { bytes = 2048 * 1024; }
  SECTION("4 MiB") { bytes = 4096 * 1024; }
  SECTION("8 MiB") { bytes = 8ull * 1024 * 1024; }

  unsigned char* h_src = nullptr;
  unsigned char* h_dst = nullptr;
  void* d_src = nullptr;
  void* d_dst = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_src), kMaxBytes));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_dst), kMaxBytes));
  HIP_CHECK(hipMalloc(&d_src, kMaxBytes));
  HIP_CHECK(hipMalloc(&d_dst, kMaxBytes));
  std::memset(h_src, 0xa5, kMaxBytes);
  std::memset(h_dst, 0, kMaxBytes);
  HIP_CHECK(hipMemcpy(d_src, h_src, kMaxBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_dst, 0, kMaxBytes));
  HIP_CHECK(hipDeviceSynchronize());

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::printf("SDMA copy-path performance\n");
  std::fflush(stdout);

  Measure("H2D", hipMemcpyHostToDevice, d_src, h_src, bytes, stream);
  Measure("D2H", hipMemcpyDeviceToHost, h_dst, d_src, bytes, stream);
  Measure("D2D", hipMemcpyDeviceToDevice, d_dst, d_src, bytes, stream);
  Measure("D2D-NoCU", hipMemcpyDeviceToDeviceNoCU, d_dst, d_src, bytes, stream);

  // Not redundant with the timing: a wrong ring read pointer lets the producer overwrite
  // packets the engine has not consumed, which corrupts data rather than stalling.
  std::memset(h_dst, 0, bytes);
  HIP_CHECK(hipMemcpy(h_dst, d_dst, bytes, hipMemcpyDeviceToHost));
  REQUIRE(std::memcmp(h_dst, h_src, bytes) == 0);
  std::memset(h_dst, 0, bytes);
  HIP_CHECK(hipMemcpy(h_dst, d_src, bytes, hipMemcpyDeviceToHost));
  REQUIRE(std::memcmp(h_dst, h_src, bytes) == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipHostFree(h_src));
  HIP_CHECK(hipHostFree(h_dst));
}

/**
 * Test Description
 * ------------------------
 *    - Holds an SDMA user queue polling a single dependency signal for a long, controlled
 * interval, which is the case the KMD asked for: the SDMA engine must survive polling one
 * address for minutes rather than the microseconds a back-to-back copy burst produces.
 *
 * A compute kernel occupies the GPU for KERNEL_SLEEP milliseconds, 2000 by default, which is
 * the two-second case shown below. The copy queued behind it on the same stream
 * carries that kernel's completion signal as a dependency, so ROCr emits a POLL_REGMEM packet
 * and the SDMA engine re-reads that address, every DW5_UNION.interval, until the kernel ends:
 *
 *     set KERNEL_SLEEP=2000
 *     MemcpyPerformance.exe Performance_hipMemcpy_SdmaLongPoll_in_stream
 *
 * What exactly gets polled, traced with AMD_LOG_LEVEL=5: the dispatch packet for the kernel
 * carries completion_signal=0x0. clr gives it no signal of its own and instead queues a
 * Barrier-AND behind it with barrier=1, and that packet's completion_signal is the handle the
 * copy then carries as its wait_event - the same value appears on both lines. ROCr turns each
 * dependency into a POLL_REGMEM on dep_signals[i]->ValueLocation() waiting for 0
 * (amd_blit_sdma.cpp BuildPollCommand). So the polled address is the barrier's signal, released
 * when the kernel retires, rather than the dispatch packet's own completion_signal. Anyone
 * looking for the poll on a particular address should expect the barrier's, not the kernel's.
 *
 * IMPORTANT: by default the SDMA engine does not poll that signal at all. On the WDDM native
 * path libhsakmt only leaves the POLL_REGMEM packet in the ring for the engine when
 * HSA_ENABLE_SDMA_GPU_POLL=1; otherwise SdmaThread waits the dependency out on the host and
 * strips the packet, which its own queue-creation line reports:
 *
 *     HSAKMT_DEBUG_LEVEL=7 -> "SDMA user queue: ... dependency polls: host thread"
 *
 * So a run with polled_by=host-thread(default) measures a host wait, not the engine. For the
 * KMD's question set HSA_ENABLE_SDMA_GPU_POLL=1 and confirm the line says "SDMA engine".
 *
 * HSA_SDMA_POLL_INTERVAL is not needed to run this - the default of 4 polls perfectly well.
 * Setting it only changes how hard the engine polls while it waits, so pairing a long sleep
 * with HSA_SDMA_POLL_INTERVAL=0 is the heaviest case available: the shortest gap between
 * re-reads, sustained for the whole kernel. Vary it to compare, not to make the test work.
 *
 * The GPU watchdog caps how long that kernel may run. With the Windows defaults - TdrDelay and
 * TdrLevel unset under HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers, so a 2 s delay -
 * KERNEL_SLEEP=2500 completes here and KERNEL_SLEEP=4000 is killed, surfacing as
 * hipErrorIllegalAddress from hipStreamSynchronize rather than as a timeout. Anything past a
 * couple of seconds needs TDR raised or disabled and a reboot first - including the two-minute
 * poll the KMD asked about, KERNEL_SLEEP=120000, which is why that is not the default here.
 * The default of 2000 sits inside the stock window so the test is runnable unconfigured; it is
 * the only part of this test that is not about SDMA.
 *
 * hipMemcpyDeviceToDeviceNoCU keeps the copy on SDMA; plain D2D above force_sdma_size would be
 * served by the blit kernel and would never emit the poll.
 *
 * The data check is an ordering check. The kernel writes its pattern only after sleeping, so a
 * copy that did not honour the dependency signal - the failure mode of a mis-timed or dropped
 * poll - reaches the destination holding kStaleByte and fails the comparison. Timing alone
 * cannot distinguish that from a correct run, which is why it is not the only assertion.
 * ------------------------
 *    - catch\performance\api\memcpy\hipMemcpySdmaCopyPath.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Performance_hipMemcpy_SdmaLongPoll_in_stream) { RunLongPoll(false); }

/**
 * Test Description
 * ------------------------
 *    - Same as Performance_hipMemcpy_SdmaLongPoll_in_stream, except the kernel and the copy sit
 * on two different streams and the dependency is stated with hipEventRecord / hipStreamWaitEvent
 * instead of inherited from a shared stream. The copy still carries a non-zero wait_event, so
 * the SDMA engine still has a signal to poll - a different handle though: the event's rather
 * than the trailing Barrier-AND's.
 * ------------------------
 *    - catch\performance\api\memcpy\hipMemcpySdmaCopyPath.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Performance_hipMemcpy_SdmaLongPoll_cross_streams) { RunLongPoll(true); }

/**
 * End doxygen group PerformanceTestMemory.
 * @}
 */
