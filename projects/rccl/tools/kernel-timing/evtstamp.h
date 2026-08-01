// Absolute dispatch timestamps out of a hipEvent_t.
//
// HIP exposes only hipEventElapsedTime, and that number is measured between the
// *completion* of the two events, so for an ext-launch pair it includes the
// launch gap ahead of the kernel. The runtime does hold the real dispatch
// start/end for the kernel, in CLOCK_BOOTTIME nanoseconds, reachable by
// pointer-chasing out of the stop event.
//
// Nothing here hardcodes a runtime ABI. discover() runs two calibration
// launches of known, different durations and keeps only the pointer chain whose
// end-minus-start reproduces both. If the layout changes, discovery fails and
// the caller falls back to hipEventElapsedTime.

#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#include <set>
#include <unistd.h>
#include <vector>

namespace evtstamp {

inline uint64_t bootNs() {
  timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

struct Chain {
  // event -> *(event+hop[0]) -> *(...+hop[1]) -> ... then start at leaf, end at leaf+8.
  std::vector<int> hop;
  int leaf = -1;
  bool valid() const { return leaf >= 0; }
};

namespace detail {

inline int memFd() {
  static int fd = open("/proc/self/mem", O_RDONLY);
  return fd;
}

inline bool safeRead(uint64_t addr, void* out, size_t n) {
  if (addr < 0x10000 || (addr & 7)) return false;
  return pread(memFd(), out, n, (off_t)addr) == (ssize_t)n;
}

static const int kWords = 32;

struct Cand {
  std::vector<int> hop;
  int leaf;
  uint64_t start, end;
};

inline void walk(uint64_t base, int depth, uint64_t lo, uint64_t hi, std::vector<int>& hop,
                 std::set<uint64_t>& seen, std::vector<Cand>& out) {
  if (depth > 3 || !seen.insert(base).second) return;
  uint64_t w[kWords];
  if (!safeRead(base, w, sizeof(w))) return;
  for (int i = 0; i + 1 < kWords; ++i) {
    if (w[i] >= lo && w[i] <= hi && w[i + 1] >= lo && w[i + 1] <= hi && w[i + 1] >= w[i])
      out.push_back({hop, i * 8, w[i], w[i + 1]});
  }
  for (int i = 0; i < kWords; ++i) {
    if (w[i] >= lo && w[i] <= hi) continue;
    hop.push_back(i * 8);
    walk(w[i], depth + 1, lo, hi, hop, seen, out);
    hop.pop_back();
  }
}

inline std::vector<Cand> candidates(hipEvent_t ev, uint64_t lo, uint64_t hi) {
  std::vector<Cand> out;
  std::vector<int> hop;
  std::set<uint64_t> seen;
  walk((uint64_t)ev, 0, lo, hi, hop, seen, out);
  return out;
}

inline bool follow(hipEvent_t ev, const Chain& c, uint64_t* start, uint64_t* end) {
  uint64_t p = (uint64_t)ev;
  for (int h : c.hop) {
    uint64_t next;
    if (!safeRead(p + h, &next, 8)) return false;
    p = next;
  }
  uint64_t v[2];
  if (!safeRead(p + c.leaf, v, 16)) return false;
  *start = v[0];
  *end = v[1];
  return true;
}

}  // namespace detail

// Runs two calibration launches through `launch`, which must dispatch a kernel
// of the requested microsecond duration with the given start/stop events
// attached and synchronize. Returns an empty chain if discovery fails.
template <typename LaunchFn>
Chain discover(LaunchFn&& launch) {
  struct Cal {
    double us;
    std::vector<detail::Cand> cands;
  } cals[2] = {{20.0, {}}, {80.0, {}}};

  for (auto& cal : cals) {
    hipEvent_t x0, x1;
    if (hipEventCreate(&x0) != hipSuccess || hipEventCreate(&x1) != hipSuccess) return {};
    uint64_t before = bootNs();
    launch(cal.us, x0, x1);
    uint64_t after = bootNs();
    cal.cands = detail::candidates(x1, before - 2000000000ull, after + 2000000000ull);
    hipEventDestroy(x0);
    hipEventDestroy(x1);
  }

  for (auto& a : cals[0].cands) {
    for (auto& b : cals[1].cands) {
      if (a.hop != b.hop || a.leaf != b.leaf) continue;
      double da = (a.end - a.start) / 1000.0, db = (b.end - b.start) / 1000.0;
      auto near = [](double got, double want) { return got > want * 0.8 && got < want * 1.25; };
      if (near(da, cals[0].us) && near(db, cals[1].us)) {
        Chain c;
        c.hop = a.hop;
        c.leaf = a.leaf;
        return c;
      }
    }
  }
  return {};
}

// Absolute kernel dispatch start/end in CLOCK_BOOTTIME ns, read from the stop
// event of an ext-launch pair.
inline bool read(hipEvent_t stopEvent, const Chain& c, uint64_t* start, uint64_t* end) {
  return c.valid() && detail::follow(stopEvent, c, start, end) && *end >= *start;
}

inline void describe(const Chain& c) {
  if (!c.valid()) {
    printf("chain: NOT FOUND\n");
    return;
  }
  printf("chain: event");
  for (int h : c.hop) printf(" -> *(+%d)", h);
  printf(" -> start@+%d end@+%d\n", c.leaf, c.leaf + 8);
}

}  // namespace evtstamp
