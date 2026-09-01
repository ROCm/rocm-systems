// Does the allocation type decide whether cache survives a dispatch?
//
// cache_capacity establishes that nothing survives a dispatch boundary for
// ordinary device memory. This asks whether that is a property of the memory type
// rather than of the dispatch, which is the one explanation with a documented
// mechanism behind it: the command processor may invalidate GL2 either
// selectively (NC lines only) or wholesale, and which lines are NC depends on the
// MTYPE the driver assigns, which in turn depends on how the memory was
// allocated. If any allocation kind retained its lines, the invalidation would be
// selective and could in principle be avoided.
//
// So try all of them. Same dependent-load probe as cache_capacity, six allocation
// kinds, two footprints. A single "RETAINS" row would change the finding.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/check.h"
#include "../common/config.h"
#include "../common/harness.h"
#include "../common/kernels.h"
#include "../common/stats.h"

using namespace bench;

namespace {

enum AllocKind { kDev, kDevFine, kHost, kHostNC, kHostReg, kManaged, kNumKinds };

const char* kKindName[kNumKinds] = {"hipMalloc",
                                    "hipMalloc finegrained",
                                    "hipHostMalloc",
                                    "hipHostMalloc NonCoherent",
                                    "malloc + hipHostRegister",
                                    "hipMallocManaged"};

// Returns the device-visible pointer, or nullptr when the kind is unsupported
// here. hostSideOut receives the pointer that has to be freed by the host path.
void* allocate(AllocKind k, u64 bytes, void** hostSideOut) {
  void* p = nullptr;
  *hostSideOut = nullptr;
  switch (k) {
    case kDev:
      if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
      return p;
    case kDevFine:
      if (hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained) != hipSuccess) return nullptr;
      return p;
    case kHost:
      if (hipHostMalloc(&p, bytes, hipHostMallocDefault) != hipSuccess) return nullptr;
      *hostSideOut = p;
      return p;
    case kHostNC:
      if (hipHostMalloc(&p, bytes, hipHostMallocNonCoherent) != hipSuccess) return nullptr;
      *hostSideOut = p;
      return p;
    case kHostReg: {
      void* h = std::malloc(bytes);
      if (!h) return nullptr;
      if (hipHostRegister(h, bytes, hipHostRegisterDefault) != hipSuccess) {
        std::free(h);
        return nullptr;
      }
      void* d = nullptr;
      if (hipHostGetDevicePointer(&d, h, 0) != hipSuccess) return nullptr;
      *hostSideOut = h;
      return d;
    }
    case kManaged:
      if (hipMallocManaged(&p, bytes) != hipSuccess) return nullptr;
      *hostSideOut = p;
      return p;
    default:
      return nullptr;
  }
}

// Failures are ignored deliberately: a release failing at the end of a probe has
// nothing to say about the probe's result, and HIP_CHECK would abort the sweep
// before the remaining allocation kinds were tested.
void release(AllocKind k, void* dev, void* hostSide) {
  if (k == kHostReg) {
    (void)hipHostUnregister(hostSide);
    std::free(hostSide);
  } else if (k == kHost || k == kHostNC) {
    (void)hipHostFree(dev);
  } else {
    (void)hipFree(dev);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", 25));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", 5));

  Machine m = initMachine("cross-dispatch retention by allocation kind");
  std::printf("  probe         : dependent-load chase, 1 lane, randomised cycle\n");
  std::printf("  cold / warm   : after a flush / after the identical walk in the previous "
              "dispatch\n");
  std::printf("  4laps         : four laps inside ONE dispatch, per hop - the control showing\n");
  std::printf("                  that caching works at all for this allocation kind\n");
  std::printf("  iters/warmup  : %d / %d\n", iters, warmup);
  (void)m;

  Scratch scratch;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  EventTimer timer;

  const u64 footprints[] = {1 * kMiB, 32 * kMiB};
  int retainingKinds = 0;

  for (u64 bytes : footprints) {
    std::printf("\n== footprint %llu MiB ==\n", static_cast<unsigned long long>(bytes / kMiB));
    std::printf("  %-27s %10s %10s %10s %10s  %s\n", "allocation", "cold_ns", "warm_ns", "4laps_ns",
                "warm/cold", "verdict");
    std::printf("  %s\n", std::string(88, '-').c_str());

    for (int k = 0; k < kNumKinds; ++k) {
      void* hostSide = nullptr;
      void* p = allocate(static_cast<AllocKind>(k), bytes, &hostSide);
      if (!p) {
        std::printf("  %-27s (unsupported on this build)\n", kKindName[k]);
        continue;
      }

      // Chase writes its cycle through the device pointer, which works for every
      // kind here since all of them are device-addressable.
      Chase chase(p, bytes);
      const u64 steps = std::min<u64>(bytes / 64, 20000);

      auto probe = [&](int mode) {  // 0 cold, 1 warm, 2 four laps in one dispatch
        std::vector<double> v;
        const u64 walkSteps = (mode == 2) ? steps * 4 : steps;
        for (int i = 0; i < warmup + iters; ++i) {
          const double ms = timer.timeMs(
              stream,
              [&] {
                if (mode == 1)
                  chase.enqueueSingleLane(stream, p, steps, scratch.sink());
                else
                  scratch.enqueueFlush(stream);
              },
              [&] { chase.enqueueSingleLane(stream, p, walkSteps, scratch.sink()); });
          if (i >= warmup) v.push_back(ms * 1e6 / static_cast<double>(walkSteps));
        }
        return median(v);
      };

      const double cold = probe(0);
      const double warm = probe(1);
      const double four = probe(2);
      const double ratio = warm / cold;
      const bool retains = ratio < 0.90;
      if (retains) ++retainingKinds;

      std::printf("  %-27s %10.1f %10.1f %10.1f %10.3f  %s\n", kKindName[k], cold, warm, four,
                  ratio, retains ? "*** RETAINS ***" : "no retention");
      std::fflush(stdout);

      char extra[80];
      std::snprintf(extra, sizeof(extra), "footprint_mib=%llu",
                    static_cast<unsigned long long>(bytes / kMiB));
      emitRow("residency_by_alloc", kKindName[k], "cold_ns_per_hop", cold, "ns", extra);
      emitRow("residency_by_alloc", kKindName[k], "warm_ns_per_hop", warm, "ns", extra);
      emitRow("residency_by_alloc", kKindName[k], "four_laps_ns_per_hop", four, "ns", extra);
      emitRow("residency_by_alloc", kKindName[k], "warm_over_cold", ratio, "x", extra);

      release(static_cast<AllocKind>(k), p, hostSide);
    }
  }

  std::printf("\n  allocation kinds retaining across a dispatch: %d\n", retainingKinds);
  if (retainingKinds == 0) {
    std::printf("  No allocation kind retains, so the invalidation is not selective by memory\n");
    std::printf("  type. The MTYPE explanation is ruled out; the mechanism remains open.\n");
  } else {
    std::printf("  At least one kind retains. That is a route to keeping cache across a\n");
    std::printf("  dispatch and CONTRADICTS the published finding - investigate before reuse.\n");
  }
  emitRow("residency_by_alloc", "-", "retaining_kinds", static_cast<double>(retainingKinds),
          "count", "");

  HIP_CHECK(hipStreamDestroy(stream));
  flushRows();
  return exitCode();
}
