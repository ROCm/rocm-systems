// Follow pointers out of a hipEvent_t looking for the dispatch timestamps.
//
// Reads go through /proc/self/mem so that walking a value that turns out not to
// be a pointer fails as an error instead of a segfault. Two independent
// signatures identify a real timestamp pair: the values sit in the
// CLOCK_BOOTTIME domain, and stop-minus-start reproduces hipEventElapsedTime.

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <set>
#include <unistd.h>
#include <vector>

#define HC(x)                                                                            \
  do {                                                                                   \
    hipError_t _e = (x);                                                                 \
    if (_e != hipSuccess) printf("FAIL %s -> %s\n", #x, hipGetErrorString(_e));           \
  } while (0)

__global__ void spin(uint64_t ticks) {
  uint64_t t0 = __builtin_amdgcn_s_memrealtime();
  while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) {
  }
}

static int memFd = -1;
static bool safeRead(uint64_t addr, void* out, size_t n) {
  if (addr < 0x1000) return false;
  return pread(memFd, out, n, (off_t)addr) == (ssize_t)n;
}

static uint64_t bootNs() {
  timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static const int kWords = 32;  // 256 bytes per block
static uint64_t g_lo, g_hi;

struct Hit {
  std::vector<int> path;  // byte offsets followed
  uint64_t value;
};

static void walk(uint64_t base, int depth, std::vector<int>& path, std::set<uint64_t>& seen,
                 std::vector<Hit>& hits) {
  if (depth > 3 || !seen.insert(base).second) return;
  uint64_t buf[kWords];
  if (!safeRead(base, buf, sizeof(buf))) return;
  for (int i = 0; i < kWords; ++i) {
    uint64_t v = buf[i];
    path.push_back(i * 8);
    if (v >= g_lo && v <= g_hi) {
      hits.push_back({path, v});
    } else if (v > 0x10000 && (v & 7) == 0) {
      walk(v, depth + 1, path, seen, hits);
    }
    path.pop_back();
  }
}

static void printPath(const std::vector<int>& p) {
  for (size_t i = 0; i < p.size(); ++i) printf("%s+%d", i ? " -> " : "", p[i]);
}

int main() {
  memFd = open("/proc/self/mem", O_RDONLY);
  if (memFd < 0) {
    printf("cannot open /proc/self/mem\n");
    return 1;
  }

  int khz = 0;
  HC(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
  uint64_t ticks = (uint64_t)(50.0 * khz / 1000.0);
  size_t sz = sizeof(ticks);
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ticks, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
  hipFunction_t fn;
  HC(hipGetFuncBySymbol(&fn, (const void*)spin));
  hipStream_t s;
  HC(hipStreamCreate(&s));

  hipEvent_t x0, x1;
  HC(hipEventCreate(&x0));
  HC(hipEventCreate(&x1));

  uint64_t before = bootNs();
  HC(hipExtModuleLaunchKernel(fn, 8 * 256, 1, 1, 256, 1, 1, 0, s, nullptr, extra, x0, x1, 0));
  HC(hipStreamSynchronize(s));
  uint64_t after = bootNs();

  float ms = 0;
  HC(hipEventElapsedTime(&ms, x0, x1));
  double elapsedNs = ms * 1e6;
  printf("host window %.3f us, hipEventElapsedTime %.3f us\n\n", (after - before) / 1000.0,
         elapsedNs / 1000.0);

  g_lo = before - 2000000000ull;
  g_hi = after + 2000000000ull;

  std::vector<Hit> h0, h1;
  {
    std::vector<int> p;
    std::set<uint64_t> seen;
    walk((uint64_t)x0, 0, p, seen, h0);
  }
  {
    std::vector<int> p;
    std::set<uint64_t> seen;
    walk((uint64_t)x1, 0, p, seen, h1);
  }

  printf("start event: %zu boot-domain values found\n", h0.size());
  for (auto& h : h0) {
    printf("  ");
    printPath(h.path);
    printf(" = %lu  (launch%+.3f us)\n", (unsigned long)h.value,
           (double)((int64_t)h.value - (int64_t)before) / 1000.0);
  }
  printf("stop event: %zu boot-domain values found\n", h1.size());
  for (auto& h : h1) {
    printf("  ");
    printPath(h.path);
    printf(" = %lu  (launch%+.3f us)\n", (unsigned long)h.value,
           (double)((int64_t)h.value - (int64_t)before) / 1000.0);
  }

  printf("\npairs whose difference reproduces hipEventElapsedTime:\n");
  for (auto& a : h0)
    for (auto& b : h1) {
      if (a.path != b.path) continue;
      double d = (double)((int64_t)b.value - (int64_t)a.value);
      if (d > elapsedNs - 600 && d < elapsedNs + 600) {
        printf("  ");
        printPath(a.path);
        printf(" : %.3f us  <-- dispatch timestamps\n", d / 1000.0);
      }
    }
  return 0;
}
