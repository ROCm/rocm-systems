// AIRUNTIME-28 benchmark support: the non-blit kernels.
//
// The previous tree had cacheFlushKernel, victimKernel, consumeKernel and
// repeatRead defined in five files between them - and all four were the same
// kernel. A strided read sweep repeated `passes` times is a flush at one pass over
// a buffer larger than cache, a victim at many passes over a buffer smaller than
// cache, and a consumer at one pass over the copy destination. Only the launch
// geometry and the buffer differ, so only those are parameters here.
#ifndef AIRUNTIME28_KERNELS_H_
#define AIRUNTIME28_KERNELS_H_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "check.h"
#include "config.h"

namespace bench {

// Strided read sweep, `passes` times over n elements.
//
// The empty asm barrier between passes stops the compiler hoisting the loads out
// of the pass loop - without it the optimiser turns a 64-pass victim into a
// 1-pass victim and the measurement quietly stops measuring anything. The
// never-true sink store keeps the accumulator live for the same reason.
__global__ void sweepReadKernel(const u64x2* buf, u64 n, int passes, u64* sink) {
  const u64 tid = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
  const u64 stride = static_cast<u64>(gridDim.x) * blockDim.x;
  u64 acc = 0;
  for (int p = 0; p < passes; ++p) {
    for (u64 i = tid; i < n; i += stride) {
      u64x2 v = buf[i];
      acc += v.x ^ v.y;
    }
    __asm__ __volatile__("" ::: "memory");
  }
  if (acc == 0xdeadbeefULL) sink[0] = acc;
}

// Dependent-load chain: each thread walks its own segment of a randomised cycle
// of 64-byte-strided lines. Every hop's address comes from the previous load, so
// the runtime is steps x memory-latency and nothing else. That converts "is this
// buffer still in cache" from a bandwidth question, where prefetch and queueing
// blur the answer, into a directly readable time difference.
__global__ void chaseKernel(const u64x2* buf, const u64* starts, u64 steps, u64* sink) {
  u64 idx = starts[threadIdx.x];
  u64 acc = 0;
  for (u64 i = 0; i < steps; ++i) {
    idx = buf[idx].x;
    acc ^= idx;
  }
  if (acc == 0xdeadbeefULL) sink[0] = acc;
}

// ---------------------------------------------------------------------------
// Scratch shared by every experiment: a sink and a flush buffer.
// ---------------------------------------------------------------------------
class Scratch {
 public:
  // flushBytes defaults to the shared constant so no experiment can accidentally
  // flush against a different, smaller footprint than the others.
  explicit Scratch(u64 flushBytes = kFlushBytes) : flushBytes_(flushBytes) {
    HIP_CHECK(hipMalloc(&sink_, 64 * sizeof(u64)));
    HIP_CHECK(hipMalloc(&flush_, flushBytes_));
    HIP_CHECK(hipMemset(flush_, 0x5A, flushBytes_));
    flushElems_ = flushBytes_ / sizeof(u64x2);
  }
  ~Scratch() {
    (void)hipFree(sink_);
    (void)hipFree(flush_);
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;

  u64* sink() const { return sink_; }
  u64 flushBytes() const { return flushBytes_; }

  // Evicts cache by streaming a footprint several times GL2's measured capacity.
  // Enqueue only; the caller decides whether it is inside or outside a timed
  // region (it must be outside).
  void enqueueFlush(hipStream_t stream) const {
    hipLaunchKernelGGL(sweepReadKernel, dim3(2048), dim3(256), 0, stream, flush_, flushElems_, 1,
                       sink_);
  }

 private:
  u64 flushBytes_;
  u64 flushElems_ = 0;
  u64* sink_ = nullptr;
  u64x2* flush_ = nullptr;
};

// ---------------------------------------------------------------------------
// Chase-cycle construction
// ---------------------------------------------------------------------------
// One randomised cycle over the 64-byte-strided lines of a footprint, plus evenly
// spaced entry points into it. Randomisation is what stops the hardware
// prefetcher hiding the latency being measured; one shared cycle rather than one
// per thread keeps the footprint exactly the size requested.
constexpr int kChaseThreads = 64;  // one block, one CU

class Chase {
 public:
  // Writes the cycle into devBuf, which must be at least `bytes` long.
  Chase(void* devBuf, u64 bytes) {
    const u64 nElems = bytes / sizeof(u64x2);
    const u64 lineStride = 64 / sizeof(u64x2);  // 4 elements per 64-byte line
    const u64 nLines = nElems / lineStride;
    BENCH_ASSERT(nLines >= static_cast<u64>(kChaseThreads),
                 "chase footprint %llu B too small for %d entry points",
                 static_cast<unsigned long long>(bytes), kChaseThreads);

    std::vector<u64> perm(nLines);
    for (u64 i = 0; i < nLines; ++i) perm[i] = i * lineStride;
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<u64x2> host(nElems);
    std::memset(host.data(), 0, bytes);
    for (u64 i = 0; i < nLines; ++i) host[perm[i]].x = perm[(i + 1) % nLines];
    HIP_CHECK(hipMemcpy(devBuf, host.data(), bytes, hipMemcpyHostToDevice));

    std::vector<u64> starts(kChaseThreads);
    for (int c = 0; c < kChaseThreads; ++c)
      starts[c] = perm[(static_cast<u64>(c) * nLines) / kChaseThreads];
    HIP_CHECK(hipMalloc(&starts_, sizeof(u64) * kChaseThreads));
    HIP_CHECK(
        hipMemcpy(starts_, starts.data(), sizeof(u64) * kChaseThreads, hipMemcpyHostToDevice));
    stepsPerLap_ = nLines / kChaseThreads;
  }
  ~Chase() { (void)hipFree(starts_); }
  Chase(const Chase&) = delete;
  Chase& operator=(const Chase&) = delete;

  u64 stepsPerLap() const { return stepsPerLap_; }

  // 64 lanes, so bandwidth and queueing contribute; use for "was this buffer
  // evicted" questions where the signal is large.
  void enqueueLaps(hipStream_t s, const void* buf, u64 laps, u64* sink) const {
    hipLaunchKernelGGL(chaseKernel, dim3(1), dim3(kChaseThreads), 0, s,
                       reinterpret_cast<const u64x2*>(buf), starts_, stepsPerLap_ * laps, sink);
  }

  // One lane: a single outstanding request at a time, so the measured time is
  // pure dependent-load latency with no queuing or gather effects mixed in.
  void enqueueSingleLane(hipStream_t s, const void* buf, u64 steps, u64* sink) const {
    hipLaunchKernelGGL(chaseKernel, dim3(1), dim3(1), 0, s, reinterpret_cast<const u64x2*>(buf),
                       starts_, steps, sink);
  }

 private:
  u64* starts_ = nullptr;
  u64 stepsPerLap_ = 0;
};

}  // namespace bench

#endif  // AIRUNTIME28_KERNELS_H_
