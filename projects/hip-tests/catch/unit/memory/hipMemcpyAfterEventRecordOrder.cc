/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemcpy hipMemcpy
 * @{
 * @ingroup MemoryTest
 * Regression test for stream ordering of a device-to-host copy that follows a
 * hipEventRecord on the same stream (rocm-systems#12571).
 *
 * On the PAL (Windows) backend a copy may be dispatched on the SDMA engine. It
 * is ordered behind prior compute by two independent mechanisms: a per-buffer
 * memory-dependency check (which inspects kernel pointer arguments), and an
 * internal "wait for idle compute" fence on the last compute submission. The
 * first mechanism does not see a buffer whose pointer only reaches the kernel
 * inside a by-value struct argument (the way framework kernels typically pass
 * pointers). A hipEventRecord between the kernel and the copy used to defeat the
 * second mechanism too -- recording the marker and the following flush cleared
 * the tracked compute submission, so the fence became a no-op and the copy
 * raced the still-running kernel, returning stale data.
 *
 * The race reproduces for both the null and an explicit stream and for several
 * copy entry points (hipMemcpyWithStream, hipMemcpyAsync), so the regression is
 * checked across that matrix. The direct-pointer and no-event-record cases are
 * controls ordered by the surviving mechanism and must stay clean, so the test
 * cannot pass for a trivial reason.
 */

#include <hip_test_common.hh>

#include <cstdint>
#include <vector>

namespace {

constexpr int kWords = 10240;      // 80 KiB buffer
constexpr int kHalf = kWords / 2;  // the writer writes the first half
constexpr int kScratch = 256;
constexpr int kBehind = 15;  // queued-ahead kernels, so compute lags
constexpr int64_t kOld = 0x5EED5EED5EED5EEDLL;
constexpr int64_t kNew = 0x0777077707770777LL;
constexpr int64_t kSentinel = 0x0BADF00D0BADF00DLL;

enum class CopyApi { kWithStream, kAsync };

struct WriteArgs {
  int64_t* x;
  unsigned int n;
  int64_t w;
  int64_t cycles;
};
struct ReadArgs {
  const int64_t* x;
  int64_t* y;
  unsigned int n;
};

// The output pointer reaches the kernel only inside the by-value struct, so the
// runtime's per-buffer dependency check does not register it.
__global__ void writeStruct(WriteArgs a) {
  const int64_t t0 = wall_clock64();
  while (wall_clock64() - t0 < a.cycles) {
  }
  for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) a.x[i] = a.w;
}

// Same body, but `alias` (== a.x) is also passed directly, so the dependency
// check does register the buffer. Used only by the control case.
__global__ void writeStructAlias(WriteArgs a, int64_t* /*alias*/) {
  const int64_t t0 = wall_clock64();
  while (wall_clock64() - t0 < a.cycles) {
  }
  for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) a.x[i] = a.w;
}

__global__ void readPlusOne(ReadArgs a) {
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.n;
       i += blockDim.x * gridDim.x) {
    a.y[i] = a.x[i] + 1;
  }
}

void LaunchWriter(bool alias, int64_t* x, unsigned int n, int64_t cycles, hipStream_t stream) {
  WriteArgs a{x, n, kNew, cycles};
  if (alias) {
    hipLaunchKernelGGL(writeStructAlias, dim3(1), dim3(256), 0, stream, a, x);
  } else {
    hipLaunchKernelGGL(writeStruct, dim3(1), dim3(256), 0, stream, a);
  }
  HIP_CHECK(hipGetLastError());
}

// Device-time of one writer at `cycles`, used to size the spin to ~kernel_ms.
float WriterMs(int64_t* x, int64_t cycles) {
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(e0, 0));
  LaunchWriter(false, x, kHalf, cycles, 0);
  HIP_CHECK(hipEventRecord(e1, 0));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipEventDestroy(e1));
  return ms;
}

struct Config {
  const char* name;
  CopyApi api;
  bool useNullStream;
  bool aliasPointer;  // control: also pass the buffer as a direct pointer arg
  bool recordEvent;   // control (false): omit the intervening hipEventRecord
};

// Runs the kernel -> [event record] -> D2H copy sequence `iters` times for one
// configuration and returns how many iterations came back wrong, i.e. the
// copied first half was not entirely the writer's value (stale or partly raced).
int CountWrong(const Config& cfg, int iters, int64_t cycles, int64_t behindCycles) {
  hipStream_t stream = 0;
  if (!cfg.useNullStream) HIP_CHECK(hipStreamCreate(&stream));
  int64_t* x = nullptr;
  int64_t* y = nullptr;
  int64_t* scratch = nullptr;
  HIP_CHECK(hipMalloc(&x, kWords * sizeof(int64_t)));
  HIP_CHECK(hipMalloc(&y, kWords * sizeof(int64_t)));
  HIP_CHECK(hipMalloc(&scratch, kScratch * sizeof(int64_t)));
  std::vector<int64_t> old(kWords, kOld);
  std::vector<int64_t> host(kWords);  // pageable

  int wrong = 0;
  for (int it = 0; it < iters; ++it) {
    HIP_CHECK(hipMemcpy(x, old.data(), kWords * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (int k = 0; k < kBehind; ++k) {
      LaunchWriter(false, scratch, kScratch, behindCycles, stream);  // queue-ahead lag
    }
    LaunchWriter(cfg.aliasPointer, x, kHalf, cycles, stream);  // the writer
    hipLaunchKernelGGL(readPlusOne, dim3(10), dim3(256), 0, stream,
                       ReadArgs{x, y, static_cast<unsigned int>(kWords)});
    HIP_CHECK(hipGetLastError());

    hipEvent_t event = nullptr;
    if (cfg.recordEvent) {
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
      HIP_CHECK(hipEventRecord(event, stream));
    }

    for (int i = 0; i < kWords; ++i) host[i] = kSentinel;
    if (cfg.api == CopyApi::kWithStream) {
      // The blocking D2H-to-pageable call a framework makes for tensor.cpu().
      HIP_CHECK(hipMemcpyWithStream(host.data(), x, kWords * sizeof(int64_t), hipMemcpyDeviceToHost,
                                    stream));
    } else {
      HIP_CHECK(
          hipMemcpyAsync(host.data(), x, kWords * sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    if (event != nullptr) HIP_CHECK(hipEventDestroy(event));

    int newCount = 0;
    for (int i = 0; i < kHalf; ++i) {
      if (host[i] == kNew) ++newCount;
    }
    if (newCount != kHalf) {
      ++wrong;  // copy did not fully observe the writer's stores
    }
  }

  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(y));
  HIP_CHECK(hipFree(scratch));
  if (!cfg.useNullStream) HIP_CHECK(hipStreamDestroy(stream));
  return wrong;
}

// Sizes the writer spin to ~target_ms and the queued-ahead kernels to ~20 ms.
void Calibrate(int64_t* cycles, int64_t* behindCycles) {
  constexpr double kTargetMs = 10.0;
  constexpr float kMaxKernelMs = 500.0f;  // well below the TDR
  int64_t* x = nullptr;
  HIP_CHECK(hipMalloc(&x, kWords * sizeof(int64_t)));
  int64_t cyc = 100000;
  float ms = 0.f;
  for (int step = 0;; ++step) {
    ms = WriterMs(x, cyc);
    if (ms >= 0.8 * kTargetMs && ms <= 1.25 * kTargetMs) break;
    double factor = kTargetMs / (ms > 1e-3f ? ms : 1e-3f);
    factor = factor < 0.1 ? 0.1 : (factor > 50 ? 50 : factor);
    const int64_t next = static_cast<int64_t>(cyc * factor);
    REQUIRE(step < 9);
    REQUIRE(ms * next / cyc <= kMaxKernelMs);
    cyc = next;
  }
  HIP_CHECK(hipFree(x));
  *cycles = cyc;
  *behindCycles = static_cast<int64_t>(cyc * 20.0 / kTargetMs / kBehind);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - A D2H copy enqueued after a hipEventRecord on the same stream must be
 *    ordered behind the preceding kernel and observe its stores, even when the
 *    kernel's output buffer is passed only inside a by-value struct argument.
 *  - Checked for the null and an explicit stream and for hipMemcpyWithStream and
 *    hipMemcpyAsync, since the race reproduces across all of them.
 *  - The direct-pointer and no-event-record configurations are ordered by a
 *    surviving mechanism and must stay clean, so the test is not passing for a
 *    trivial reason.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAfterEventRecordOrder.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipMemcpy_OrderedAfterEventRecord) {
  const int iters = isQuickLevel() ? 15 : 40;
  int64_t cycles = 0;
  int64_t behindCycles = 0;
  Calibrate(&cycles, &behindCycles);

  const Config configs[] = {
      // Regression: pointer only in a struct + event record, across the matrix.
      {"withstream, null stream", CopyApi::kWithStream, true, false, true},
      {"withstream, explicit stream", CopyApi::kWithStream, false, false, true},
      {"async, null stream", CopyApi::kAsync, true, false, true},
      {"async, explicit stream", CopyApi::kAsync, false, false, true},
      // Controls: ordered by the surviving mechanism, must stay clean.
      {"control: direct pointer + record", CopyApi::kWithStream, true, true, true},
      {"control: struct pointer + no record", CopyApi::kWithStream, true, false, false},
  };

  for (const Config& cfg : configs) {
    const int wrong = CountWrong(cfg, iters, cycles, behindCycles);
    INFO(cfg.name << ": wrong " << wrong << " / " << iters);
    CHECK(wrong == 0);
  }
}
