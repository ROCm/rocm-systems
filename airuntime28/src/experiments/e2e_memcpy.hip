// End to end through the real runtime, not a transcription of it.
//
// Every other experiment here reimplements the blit kernel so that variants can
// be compared under identical conditions. That leaves one thing unverified: that
// the change, as it exists in CLR, actually reaches hipMemcpy and does what the
// standalone measurement says it does. This program calls hipMemcpyAsync and
// nothing else, so it exercises the shipped kernel, the shipped dispatch, and the
// shipped flag.
//
// It must be run twice, against a CLR build carrying the patch:
//
//   DEBUG_CLR_BLIT_NONTEMPORAL=0 ./build/e2e_memcpy
//   DEBUG_CLR_BLIT_NONTEMPORAL=1 ./build/e2e_memcpy
//
// The flag is read once when the runtime initialises and cannot be changed inside
// a process, so this is a between-process comparison - much noisier than the
// paired within-run comparisons elsewhere, and it carries a null control (two
// runs at the same setting) for exactly that reason. Treat it as a check that the
// plumbing works and the sign is right, not as a precise effect size.
//
// The victim working set is 32 MiB here, the size at which the standalone sweep
// found the largest effect. The original used 2 MiB, which is where the effect
// vanishes.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.h"
#include "../common/config.h"
#include "../common/harness.h"
#include "../common/kernels.h"
#include "../common/stats.h"

using namespace bench;

int main(int argc, char** argv) {
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", 30));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", 6));
  const char* flag = std::getenv("DEBUG_CLR_BLIT_NONTEMPORAL");

  Machine m = initMachine("end to end via hipMemcpyAsync");
  std::printf("  DEBUG_CLR_BLIT_NONTEMPORAL = %s\n", flag ? flag : "(unset)");
  std::printf("  iters/warmup  : %d / %d\n", iters, warmup);

  const u64 maxSize = 1 * kGiB;
  u8* src = nullptr;
  u8* dst = nullptr;
  HIP_CHECK(hipMalloc(&src, maxSize));
  HIP_CHECK(hipMalloc(&dst, maxSize));
  HIP_CHECK(hipMemset(src, 0xA5, maxSize));
  HIP_CHECK(hipMemset(dst, 0x00, maxSize));

  hipStream_t stream, sCopy;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamCreate(&sCopy));
  EventTimer timer;
  Scratch scratch;

  // -------------------------------------------------------------------------
  // Correctness. Sizes include non-multiples of 16 to exercise the scalar
  // remainder tail, and odd offsets to exercise the narrow unaligned path. The
  // guard byte past the end catches a variant that writes beyond the copy, which
  // a width change is a plausible way to introduce.
  // -------------------------------------------------------------------------
  std::printf("\n=== correctness (hipMemcpyAsync, device to device) ===\n");
  const u64 sizes[] = {1,        7,             64,        255,      4 * kKiB + 3,
                       64 * kKiB, 1 * kMiB,     3 * kMiB + 17, 64 * kMiB, 257 * kMiB};
  int failures = 0;
  int checked = 0;
  for (u64 sz : sizes) {
    for (u64 off : {u64(0), u64(1), u64(8)}) {
      if (sz + off + 1 > maxSize) continue;
      ++checked;
      std::vector<u8> h(sz), back(sz);
      for (u64 i = 0; i < sz; ++i) h[i] = static_cast<u8>((i * 37 + off * 11 + 5) & 0xFF);
      HIP_CHECK(hipMemcpy(src + off, h.data(), sz, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dst, 0xCD, std::min(maxSize, sz + off + 64)));
      HIP_CHECK(hipMemcpyAsync(dst + off, src + off, sz, hipMemcpyDeviceToDevice, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipMemcpy(back.data(), dst + off, sz, hipMemcpyDeviceToHost));

      u64 bad = 0;
      for (u64 i = 0; i < sz; ++i)
        if (h[i] != back[i]) ++bad;
      u8 guard = 0;
      HIP_CHECK(hipMemcpy(&guard, dst + off + sz, 1, hipMemcpyDeviceToHost));
      const bool overrun = (guard != 0xCD);
      if (bad || overrun) {
        ++failures;
        std::printf("  FAIL size=%llu off=%llu : %llu bad byte(s)%s\n",
                    static_cast<unsigned long long>(sz), static_cast<unsigned long long>(off),
                    static_cast<unsigned long long>(bad), overrun ? " AND wrote past the end" : "");
      }
    }
  }
  std::printf("  %s: %d case(s) checked, %d failure(s)\n", failures ? "FAILED" : "all pass", checked,
              failures);
  BENCH_ASSERT(failures == 0, "hipMemcpy is not byte-exact with the flag at %s",
               flag ? flag : "(unset)");
  emitRow("e2e", "correctness", "failures", static_cast<double>(failures), "count",
          flag ? flag : "unset");

  // -------------------------------------------------------------------------
  // Bandwidth through the real path.
  // -------------------------------------------------------------------------
  std::printf("\n=== hipMemcpyAsync bandwidth ===\n");
  std::printf("  %12s %12s %14s\n", "size", "median_ms", "GB/s (r+w)");
  const u64 bwSizes[] = {1 * kMiB, 16 * kMiB, 64 * kMiB, 128 * kMiB, 256 * kMiB, 1 * kGiB};
  for (u64 sz : bwSizes) {
    std::vector<double> v;
    for (int i = 0; i < warmup + iters; ++i) {
      const double ms = timer.timeMs(
          stream, [&] { scratch.enqueueFlush(stream); },
          [&] {
            HIP_CHECK(hipMemcpyAsync(dst, src, sz, hipMemcpyDeviceToDevice, stream));
          });
      if (i >= warmup) v.push_back(ms);
    }
    const double ms = median(v);
    char label[24];
    std::snprintf(label, sizeof(label), "%lluMiB", static_cast<unsigned long long>(sz / kMiB));
    std::printf("  %12s %12.4f %14.1f\n", label, ms,
                2.0 * static_cast<double>(sz) / (ms * 1e-3) / 1e9);
    char extra[48];
    std::snprintf(extra, sizeof(extra), "flag=%s", flag ? flag : "unset");
    emitRow("e2e", label, "memcpy_ms", ms, "ms", extra);
  }

  // -------------------------------------------------------------------------
  // The scenario the change exists for: a victim whose working set lives in GL2,
  // with large copies running alongside it.
  // -------------------------------------------------------------------------
  std::printf("\n=== victim with concurrent hipMemcpyAsync ===\n");
  const u64 wsBytes = 32 * kMiB;
  const u64 wsElems = wsBytes / sizeof(u64x2);
  u64x2* ws = nullptr;
  HIP_CHECK(hipMalloc(&ws, wsBytes));
  HIP_CHECK(hipMemset(ws, 0x5A, wsBytes));
  const int passes = 230;  // ~5 ms alone at 32 MiB, matching the standalone sweep
  const u64 copySize = 128 * kMiB;
  const int nCopies = 82;

  std::vector<double> alone, concurrent;
  for (int i = 0; i < warmup + iters; ++i) {
    alone.push_back(timer.timeMs(stream, [&] {
      hipLaunchKernelGGL(sweepReadKernel, dim3(128), dim3(256), 0, stream, ws, wsElems, passes,
                         scratch.sink());
    }));
    concurrent.push_back(timer.timeMs(
        stream, [] {},
        [&] {
          hipLaunchKernelGGL(sweepReadKernel, dim3(128), dim3(256), 0, stream, ws, wsElems, passes,
                             scratch.sink());
          for (int k = 0; k < nCopies; ++k)
            HIP_CHECK(hipMemcpyAsync(dst, src, copySize, hipMemcpyDeviceToDevice, sCopy));
        },
        [&] { HIP_CHECK(hipStreamSynchronize(sCopy)); }));
  }
  alone.erase(alone.begin(), alone.begin() + warmup);
  concurrent.erase(concurrent.begin(), concurrent.begin() + warmup);
  const double mAlone = median(alone);
  const double mConc = median(concurrent);
  std::printf("  victim working set : %llu MiB\n", static_cast<unsigned long long>(wsBytes / kMiB));
  std::printf("  victim alone       : %.3f ms\n", mAlone);
  std::printf("  victim + copies    : %.3f ms  (%.2fx slower)\n", mConc, mConc / mAlone);
  emitRow("e2e", "victim", "alone_ms", mAlone, "ms", flag ? flag : "unset");
  emitRow("e2e", "victim", "concurrent_ms", mConc, "ms", flag ? flag : "unset");

  // The comparison line the wrapper script pairs across the two flag settings.
  std::printf("\nRESULT flag=%s failures=%d victim_alone_ms=%.4f victim_conc_ms=%.4f\n",
              flag ? flag : "unset", failures, mAlone, mConc);

  HIP_CHECK(hipFree(ws));
  HIP_CHECK(hipStreamDestroy(sCopy));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
  flushRows();
  return exitCode();
}
