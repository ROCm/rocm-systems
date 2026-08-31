// How big is the cache, and does anything stay in it across a dispatch?
//
// Two scans, both of which had to be run before any other result in this
// investigation could be trusted.
//
// CAPACITY. Three sources disagree about GL2: the architecture notes say 96 MB
// per AID, while hipDeviceProp_t::l2CacheSize, rocminfo and amd-smi all report
// 4 MB (they read the same KFD record, whose geometry fields are all zero - an
// unpopulated stub). Measure instead of trusting either: read a buffer of size S
// repeatedly inside ONE dispatch, holding total traffic constant. While S fits in
// cache, bandwidth is high; once it does not, every pass goes to HBM and
// bandwidth flattens. The knee is the capacity. Everything stays inside one
// dispatch deliberately, because cross-dispatch retention is the subject of the
// second scan and relying on it here would confound the first.
//
// RESIDENCY. Dependent-load latency over a footprint, measured two ways: after
// flushing the cache, and immediately after an identical walk in a previous
// dispatch. If those two are equal at footprints that comfortably fit, nothing
// survives the dispatch boundary. The same-dispatch column is the control that
// separates "the cache does not work" from "the cache is not retained".

#include <cstdio>
#include <string>
#include <vector>

#include "../common/check.h"
#include "../common/config.h"
#include "../common/harness.h"
#include "../common/kernels.h"
#include "../common/stats.h"

using namespace bench;

static std::string sizeLabel(u64 bytes) {
  char b[32];
  if (bytes >= kMiB)
    std::snprintf(b, sizeof(b), "%llu MiB", static_cast<unsigned long long>(bytes / kMiB));
  else
    std::snprintf(b, sizeof(b), "%llu KiB", static_cast<unsigned long long>(bytes / kKiB));
  return b;
}

// ---------------------------------------------------------------------------
static void capacityScan(const Machine& m, int iters, int warmup) {
  std::printf("\n=== CAPACITY: read bandwidth vs footprint, all within one dispatch ===\n");
  std::printf("  Total traffic held constant per point, so only hit rate varies.\n");
  std::printf("  Footprints below ~4 MiB do not fill the grid and are bounded by that, not\n");
  std::printf("  by cache; read the knee off the large end of the curve.\n");
  std::printf("  %12s %10s %12s %10s\n", "footprint", "passes", "GB/s", "vs peak");

  const u64 sizes[] = {256 * kKiB, 1 * kMiB,  4 * kMiB,   8 * kMiB,   12 * kMiB,  16 * kMiB,
                       24 * kMiB,  32 * kMiB, 48 * kMiB,  64 * kMiB,  80 * kMiB,  96 * kMiB,
                       128 * kMiB, 192 * kMiB, 256 * kMiB, 512 * kMiB, 1 * kGiB};
  const u64 trafficPerPoint = 8 * kGiB;

  u64* sink = nullptr;
  HIP_CHECK(hipMalloc(&sink, 64 * sizeof(u64)));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  EventTimer timer;

  double peak = 0.0;
  std::vector<std::pair<u64, double>> curve;
  for (u64 size : sizes) {
    u64x2* buf = nullptr;
    HIP_CHECK(hipMalloc(&buf, size));
    HIP_CHECK(hipMemset(buf, 0x11, size));
    const u64 elems = size / sizeof(u64x2);
    const int passes = static_cast<int>(std::max<u64>(1, trafficPerPoint / size));

    std::vector<double> samples;
    for (int i = 0; i < warmup + iters; ++i) {
      const double ms = timer.timeMs(stream, [&] {
        hipLaunchKernelGGL(sweepReadKernel, dim3(m.blitWg * 4), dim3(256), 0, stream, buf, elems,
                           passes, sink);
      });
      if (i >= warmup) samples.push_back(ms);
    }
    const double ms = median(samples);
    const double gbs = static_cast<double>(size) * passes / (ms * 1e-3) / 1e9;
    peak = std::max(peak, gbs);
    curve.emplace_back(size, gbs);
    std::printf("  %12s %10d %12.1f %9.0f%%\n", sizeLabel(size).c_str(), passes, gbs,
                100.0 * gbs / peak);
    emitRow("cache_capacity", sizeLabel(size).c_str(), "read_gbps", gbs, "GB/s", "");
    HIP_CHECK(hipFree(buf));
  }

  // Read the capacity off the curve as the last footprint still running closer to
  // the cached plateau than to the HBM floor.
  //
  // The floor comes from the largest footprint measured, which cannot be
  // resident. The plateau is the median of the band above the near-cache knee and
  // below the driver's wildest claim, which is where the curve is flat if GL2 is
  // doing anything. A midpoint crossing is used rather than "some multiple of the
  // floor" because the latter moved the answer between 128 and 192 MiB on
  // consecutive runs while the underlying curve barely changed.
  const double floorGbs = curve.back().second;
  std::vector<double> plateau;
  for (auto& p : curve)
    if (p.first >= 24 * kMiB && p.first <= 80 * kMiB) plateau.push_back(p.second);
  const double plateauGbs = median(plateau);
  const double midpoint = 0.5 * (plateauGbs + floorGbs);
  u64 capacity = 0;
  for (auto& p : curve)
    if (p.second >= midpoint) capacity = std::max(capacity, p.first);

  std::printf("\n  HBM floor      %.0f GB/s (largest footprint, cannot be resident)\n", floorGbs);
  std::printf("  cached plateau %.0f GB/s (median of the 24-80 MiB band)\n", plateauGbs);
  std::printf("  capacity       %s (largest footprint above the midpoint, %.0f GB/s)\n",
              sizeLabel(capacity).c_str(), midpoint);
  std::printf("  driver reports %d MiB; config.h uses %llu MiB\n",
              m.prop.l2CacheSize / (1024 * 1024), static_cast<unsigned long long>(kGL2Bytes / kMiB));
  emitRow("cache_capacity", "capacity", "footprint_mib", static_cast<double>(capacity / kMiB), "MiB",
          "");
  emitRow("cache_capacity", "plateau", "read_gbps", plateauGbs, "GB/s", "");
  emitRow("cache_capacity", "hbm_floor", "read_gbps", floorGbs, "GB/s", "");

  BENCH_ASSERT(plateauGbs > 1.5 * floorGbs,
               "no cached plateau: %.0f GB/s at 24-80 MiB against a %.0f GB/s floor, so this "
               "part does not cache reads at GL2 scale and every footprint here is misjudged",
               plateauGbs, floorGbs);
  BENCH_ASSERT(capacity >= 48 * kMiB,
               "measured capacity %s is far below the %llu MiB config.h assumes, so every "
               "footprint in this suite is sized wrongly",
               sizeLabel(capacity).c_str(), static_cast<unsigned long long>(kGL2Bytes / kMiB));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(sink));
}

// ---------------------------------------------------------------------------
static void residencyScan(const Machine& m, int iters, int warmup, u64 flushBytes) {
  std::printf("\n=== RESIDENCY: dependent-load latency vs footprint (1 lane) ===\n");
  std::printf("  cold : %llu MiB streamed through cache first (%.1fx measured GL2)\n",
              static_cast<unsigned long long>(flushBytes / kMiB),
              static_cast<double>(flushBytes) / static_cast<double>(kGL2Bytes));
  std::printf("  warm : identical walk in the immediately preceding dispatch\n");
  std::printf("  same : both walks inside one dispatch (control: does the cache work at all)\n");
  std::printf("  cold == warm means cache state did not survive the dispatch boundary.\n\n");
  std::printf("  %12s %13s %13s %13s %10s\n", "footprint", "cold_ns/hop", "warm_ns/hop",
              "same_ns/hop", "cold/warm");

  const u64 sizes[] = {32 * kKiB, 128 * kKiB, 512 * kKiB, 1 * kMiB,   2 * kMiB,  4 * kMiB,
                       8 * kMiB,  16 * kMiB,  32 * kMiB,  64 * kMiB,  96 * kMiB, 256 * kMiB};

  Scratch scratch(flushBytes);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  EventTimer timer;

  u8* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, 256 * kMiB));

  double worstRatio = 1.0;
  u64 worstAt = 0;
  for (u64 size : sizes) {
    Chase chase(buf, size);
    const u64 steps = std::min<u64>(size / 64, 20000);

    auto scan = [&](bool flushFirst) {
      std::vector<double> v;
      for (int i = 0; i < warmup + iters; ++i) {
        const double ms = timer.timeMs(
            stream,
            [&] {
              // Both arms run a dispatch before the timed one: the cold arm's
              // evicts, the warm arm's populates. Without the warm arm's dummy
              // dispatch the two would differ by a launch as well as by cache
              // state.
              if (flushFirst) {
                scratch.enqueueFlush(stream);
              } else {
                chase.enqueueSingleLane(stream, buf, steps, scratch.sink());
              }
            },
            [&] { chase.enqueueSingleLane(stream, buf, steps, scratch.sink()); });
        if (i >= warmup) v.push_back(ms * 1e6 / static_cast<double>(steps));
      }
      return median(v);
    };

    const double cold = scan(true);
    const double warm = scan(false);

    // Same-dispatch control: four laps in one dispatch. If per-hop time here is
    // below the cold figure, the cache is working and simply is not retained.
    std::vector<double> same;
    for (int i = 0; i < warmup + iters; ++i) {
      const double ms = timer.timeMs(
          stream, [&] { scratch.enqueueFlush(stream); },
          [&] { chase.enqueueSingleLane(stream, buf, steps * 4, scratch.sink()); });
      if (i >= warmup) same.push_back(ms * 1e6 / static_cast<double>(steps * 4));
    }

    const double ratio = cold / warm;
    std::printf("  %12s %13.1f %13.1f %13.1f %9.2fx\n", sizeLabel(size).c_str(), cold, warm,
                median(same), ratio);
    emitRow("residency", sizeLabel(size).c_str(), "cold_ns_per_hop", cold, "ns", "");
    emitRow("residency", sizeLabel(size).c_str(), "warm_ns_per_hop", warm, "ns", "");
    emitRow("residency", sizeLabel(size).c_str(), "same_dispatch_ns_per_hop", median(same), "ns",
            "");
    if (size <= kGL2Bytes && ratio > worstRatio) {
      worstRatio = ratio;
      worstAt = size;
    }
  }

  std::printf("\n  largest cold/warm ratio at a footprint within GL2: %.2fx at %s\n", worstRatio,
              worstAt ? sizeLabel(worstAt).c_str() : "-");
  if (worstRatio < 1.10) {
    std::printf("  => nothing measurably survives the dispatch boundary at any GL2-sized\n");
    std::printf("     footprint. Cache-residency optimisations across dispatches cannot pay.\n");
  } else {
    std::printf("  => some retention detected. This CONTRADICTS the published finding and\n");
    std::printf("     must be investigated before the report is reused.\n");
  }
  emitRow("residency", "-", "max_cold_warm_ratio", worstRatio, "x", "");

  HIP_CHECK(hipFree(buf));
  HIP_CHECK(hipStreamDestroy(stream));
  (void)m;
}

int main(int argc, char** argv) {
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", 15));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", 4));
  const u64 flushBytes = static_cast<u64>(argValue(argc, argv, "--flush-mib",
                                                   static_cast<long long>(kFlushBytes / kMiB))) *
                         kMiB;

  Machine m = initMachine("cache capacity and residency");
  std::printf("  iters/warmup  : %d / %d\n", iters, warmup);

  if (!argFlag(argc, argv, "--residency-only")) capacityScan(m, iters, warmup);
  if (!argFlag(argc, argv, "--capacity-only")) residencyScan(m, iters, warmup, flushBytes);
  flushRows();
  return exitCode();
}
