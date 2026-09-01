// AIRUNTIME-28 benchmark support: the measurement discipline, in one place.
//
// Two rules produce every trustworthy number in this investigation, and both were
// previously reimplemented per experiment:
//
//   1. Arms are measured once per iteration in a per-iteration shuffled order, so
//      drift during a run cannot attach itself to arm identity. Measuring all
//      iterations of one arm and then the next silently credits later arms with
//      whatever the machine did in the meantime.
//
//   2. At least one arm occupies two slots. Both run identical code, so the gap
//      between them is the resolution limit of the rig, measured rather than
//      assumed. Give every arm two slots when a result is close to that limit -
//      an asymmetric layout hands the duplicated arm an advantage wherever
//      per-dispatch cost matters.
#ifndef AIRUNTIME28_HARNESS_H_
#define AIRUNTIME28_HARNESS_H_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "config.h"
#include "stats.h"

namespace bench {

// GPU-side timing. Events are recorded into the stream rather than around the host
// call, so the interval is GPU execution and excludes launch overhead - which on a
// small copy is several times the copy itself.
class EventTimer {
 public:
  EventTimer() {
    HIP_CHECK(hipEventCreate(&a_));
    HIP_CHECK(hipEventCreate(&b_));
  }
  ~EventTimer() {
    (void)hipEventDestroy(a_);
    (void)hipEventDestroy(b_);
  }
  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  // setup() is enqueued before the start event, timed() between the events. The
  // split matters: timing setup together with the thing it prepares lets a win on
  // one hide a loss on the other. That mistake was made once here already - the
  // first version of the adversarial suite timed the copy together with the thing
  // the copy was supposed to damage, so a win on one masked a loss on the other.
  //
  // finish() runs after the stop event and before the elapsed time is read, for
  // cases that enqueue on more than one stream and must drain all of them.
  template <typename Setup, typename Timed, typename Finish>
  double timeMs(hipStream_t stream, Setup setup, Timed timed, Finish finish) {
    setup();
    HIP_CHECK(hipEventRecord(a_, stream));
    timed();
    HIP_CHECK(hipEventRecord(b_, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    finish();
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a_, b_));
    return static_cast<double>(ms);
  }

  template <typename Setup, typename Timed>
  double timeMs(hipStream_t stream, Setup setup, Timed timed) {
    return timeMs(stream, setup, timed, [] {});
  }

  template <typename Timed>
  double timeMs(hipStream_t stream, Timed timed) {
    return timeMs(stream, [] {}, timed, [] {});
  }

 private:
  hipEvent_t a_{}, b_{};
};

// A set of slots to compare. Slots sharing an arm id run identical code; their
// mutual difference is a noise floor.
struct SlotPlan {
  std::vector<std::string> names;  // display name per slot
  std::vector<int> arm;            // arm id per slot; duplicates share one

  int size() const { return static_cast<int>(names.size()); }

  // Two slots per arm, which is the layout to prefer. Names get " a" / " b".
  static SlotPlan duplicated(const std::vector<std::string>& armNames) {
    SlotPlan p;
    for (size_t i = 0; i < armNames.size(); ++i) {
      p.names.push_back(armNames[i] + " a");
      p.arm.push_back(static_cast<int>(i));
    }
    for (size_t i = 0; i < armNames.size(); ++i) {
      p.names.push_back(armNames[i] + " b");
      p.arm.push_back(static_cast<int>(i));
    }
    return p;
  }
};

// Per-slot samples plus the comparisons derived from them.
class SlotResults {
 public:
  SlotResults(SlotPlan plan, std::vector<std::vector<double>> samples)
      : plan_(std::move(plan)), samples_(std::move(samples)) {}

  const std::vector<double>& samples(int slot) const { return samples_[slot]; }
  double medianMs(int slot) const { return median(samples_[slot]); }
  const SlotPlan& plan() const { return plan_; }

  // First slot belonging to an arm.
  int firstSlotOf(int armId) const {
    for (int s = 0; s < plan_.size(); ++s)
      if (plan_.arm[s] == armId) return s;
    return -1;
  }

  // Effect of one arm against another, paired.
  Delta effect(int armId, int baseArmId = 0) const {
    return pairedDelta(samples_[firstSlotOf(armId)], samples_[firstSlotOf(baseArmId)]);
  }

  // Every same-arm-twice gap available. These are the noise floors.
  std::vector<Delta> noiseFloors() const {
    std::vector<Delta> floors;
    const int nArms = 1 + *std::max_element(plan_.arm.begin(), plan_.arm.end());
    for (int a = 0; a < nArms; ++a) {
      std::vector<int> slots;
      for (int s = 0; s < plan_.size(); ++s)
        if (plan_.arm[s] == a) slots.push_back(s);
      for (size_t i = 1; i < slots.size(); ++i)
        floors.push_back(pairedDelta(samples_[slots[i]], samples_[slots[0]]));
    }
    return floors;
  }

  bool isSignificant(const Delta& d) const { return separable(d, noiseFloors()); }

  // Widest noise floor, for reporting the resolution limit alongside a result.
  double resolutionLimit() const {
    double w = 0.0;
    for (const Delta& f : noiseFloors()) w = std::max(w, f.width());
    return w;
  }

 private:
  SlotPlan plan_;
  std::vector<std::vector<double>> samples_;
};

// Runs every slot once per iteration in shuffled order. measure(slot) enqueues the
// work and returns its elapsed time in ms; use EventTimer inside it.
template <typename MeasureFn>
SlotResults runSlots(const SlotPlan& plan, int iters, int warmup, MeasureFn measure,
                     unsigned seed = 0xC0FFEEu) {
  std::vector<std::vector<double>> samples(plan.size());
  std::vector<int> order(plan.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(seed);

  for (int i = 0; i < warmup + iters; ++i) {
    std::shuffle(order.begin(), order.end(), rng);
    for (int slot : order) {
      const double ms = measure(slot);
      if (i >= warmup) samples[slot].push_back(ms);
    }
  }
  return SlotResults(plan, std::move(samples));
}

// One row per arm: median, effect against arm 0, significance. Arm 0 is the
// baseline and prints as the reference.
inline void printArmTable(const SlotResults& r, const std::vector<std::string>& armNames,
                          const char* unit = "ms") {
  std::printf("  %-20s %11s   %s\n", "arm", (std::string("median_") + unit).c_str(),
              "vs baseline [95% CI]");
  std::printf("  %s\n", std::string(66, '-').c_str());
  for (size_t a = 0; a < armNames.size(); ++a) {
    const int slot = r.firstSlotOf(static_cast<int>(a));
    const Delta d = r.effect(static_cast<int>(a));
    char cell[80];
    if (a == 0) {
      std::snprintf(cell, sizeof(cell), "%s", "-  (baseline)");
    } else {
      formatDelta(cell, sizeof(cell), d, r.isSignificant(d));
    }
    std::printf("  %-20s %11.4f   %-28s\n", armNames[a].c_str(), r.medianMs(slot), cell);
  }
  // Not written as +/- : this is the width of the interval on a same-arm-twice
  // comparison, and an effect must exceed it outright, not merely land outside a
  // band half that size.
  std::printf("  resolution limit: %.2f pp, the widest gap between two slots running the same\n",
              r.resolutionLimit());
  std::printf("  arm. '(ns)' = not separable from that, so not evidence of an effect.\n");
}

// Machine-readable rows so report tables can be regenerated instead of
// hand-copied. Two published tables have already gone stale by hand-copying.
//
// Buffered rather than printed inline, because interleaving them into the
// human-readable tables makes both unreadable. Call flushRows() before returning
// from main.
inline std::vector<std::string>& rowBuffer() {
  static std::vector<std::string> rows;
  return rows;
}

inline void emitRow(const char* experiment, const char* arm, const char* metric, double value,
                    const char* unit, const char* extra = "") {
  char line[512];
  std::snprintf(line, sizeof(line), "ROW\t%s\t%s\t%s\t%.6f\t%s\t%s", experiment, arm, metric, value,
                unit, extra);
  rowBuffer().push_back(line);
}

inline void flushRows() {
  for (const std::string& row : rowBuffer()) std::printf("%s\n", row.c_str());
  rowBuffer().clear();
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Startup: machine facts and argument parsing
// ---------------------------------------------------------------------------
struct Machine {
  hipDeviceProp_t prop{};
  u32 blitWg = 256;  // limit_blit_wg_, which the runtime sets to the CU count
};

// Prints the facts a reader needs to interpret the numbers, including both cache
// sizes: the driver's figure appears next to the measured one precisely because
// they disagree and the disagreement caused every footprint in the first round of
// this investigation to be sized wrongly.
inline Machine initMachine(const char* experimentName) {
  Machine m;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGetDeviceProperties(&m.prop, 0));
  m.blitWg = static_cast<u32>(m.prop.multiProcessorCount);

  std::printf("=== %s ===\n", experimentName);
  std::printf("  device        : %s (%s), %d CUs\n", m.prop.name, m.prop.gcnArchName,
              m.prop.multiProcessorCount);
  std::printf("  clock         : %d MHz\n", m.prop.clockRate / 1000);
  std::printf("  GL2 capacity  : %llu MiB measured / %d MiB reported by driver\n",
              static_cast<unsigned long long>(kGL2Bytes / kMiB), m.prop.l2CacheSize / (1024 * 1024));
  std::printf("  flush         : %llu MiB (%.1fx measured GL2)\n",
              static_cast<unsigned long long>(kFlushBytes / kMiB),
              static_cast<double>(kFlushBytes) / static_cast<double>(kGL2Bytes));
  std::printf("  dispatch      : %llu threads/WG x %u WGs, grid-stride (production)\n",
              static_cast<unsigned long long>(kLocalWorkSize), m.blitWg);
  std::printf("  bootstrap     : %d resamples, seed %u\n", kBootstrapResamples, kBootstrapSeed);
  std::fflush(stdout);
  return m;
}

// Prints the addresses a run happened to get, and their separation.
//
// Worth recording because mid-size copy time on this part is bimodal across
// process invocations - the same binary copying the same 20 MiB measures either
// ~56 us or ~103 us depending on the run, while being repeatable to under a
// percent within a run. The separation between source and destination is the
// prime suspect (channel and bank hashing), so it is reported rather than
// guessed at. The consequence for method is that absolute times in that band
// describe a run, not the hardware; only within-run paired comparisons carry
// across runs, which is what this harness produces.
inline void reportBuffers(const void* src, const void* dst) {
  const unsigned long long s = reinterpret_cast<unsigned long long>(src);
  const unsigned long long d = reinterpret_cast<unsigned long long>(dst);
  const unsigned long long delta = (d > s) ? (d - s) : (s - d);
  std::printf("  buffers       : src 0x%llx  dst 0x%llx  |delta| %llu MiB%s\n", s, d,
              delta / (1024 * 1024), (delta & (delta - 1)) == 0 ? " (power of two)" : "");
  emitRow("buffers", "src_dst_delta", "bytes", static_cast<double>(delta), "B", "");
}

inline long long argValue(int argc, char** argv, const char* flag, long long fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == flag) return std::atoll(argv[i + 1]);
  return fallback;
}

inline bool argFlag(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == flag) return true;
  return false;
}

}  // namespace bench

#endif  // AIRUNTIME28_HARNESS_H_
