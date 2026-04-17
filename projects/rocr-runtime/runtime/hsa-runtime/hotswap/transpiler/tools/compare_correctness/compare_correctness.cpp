// Numerical correctness comparison across three transpilation paths.
//
// For each (kernel, shape) pair the harness runs:
//
//   native  — a gfx942 .co compiled directly for the target hardware;
//             no transpiler in the chain, serves as a harness sanity check.
//   legacy  — a gfx1250 .co loaded via the ROCR hotswap hook with
//             HSA_HOTSWAP_IR_RAISER unset.  The legacy byte-level
//             transpiler rewrites the binary to gfx942 before dispatch.
//   salmon  — the same gfx1250 .co with HSA_HOTSWAP_IR_RAISER=1, so
//             Salmon raises to LLVM IR and re-lowers to gfx942.
//
// All three children launch the same kernel with the same inputs and
// write the device-side output to a tempfile.  The parent computes a CPU
// reference (the gold) and compares every child's output against it.
//
// Because the loader freezes its mode selection in a `static const char*`
// initializer, each child must be a fresh process.  Because each load
// potentially mutates ROCR-internal state, we also spawn one child per
// (mode, recipe, shape) to keep runs independent.
//
// Build and run
// -------------
//   make ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build
//   ./compare_correctness                           # full default sweep
//   ./compare_correctness --recipe=vecadd           # one recipe
//   ./compare_correctness --shape=1024              # restrict N (cross-product with blocks)
//   ./compare_correctness --block=64                # restrict block (cross-product with Ns)
//   ./compare_correctness --shape=1024 --block=128  # pin a single (N, block)
//   ./compare_correctness --child=... ...           # internal: child invocation
//
// The harness builds its own gfx942 and gfx1250 kernel binaries on first
// run.  `make` handles that via hipcc + clang-offload-bundler.

#include <hip/hip_runtime.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

// Where the harness finds the compiled kernel code objects relative to
// the running binary.  Populated by the build (Makefile).
const char *KERNEL_DIR_ENV = "COMPARE_CORRECTNESS_KERNEL_DIR";
const char *KERNEL_DIR_DEFAULT = "./kernels/build";

std::string kernelDir() {
  if (const char *e = std::getenv(KERNEL_DIR_ENV); e && *e) return e;
  return KERNEL_DIR_DEFAULT;
}

std::string coPathFor(const std::string &name, const std::string &isa) {
  return kernelDir() + "/" + name + "." + isa + ".co";
}

// ─────────────────────────────────────────────────────────────────────────────
// Small utilities
// ─────────────────────────────────────────────────────────────────────────────

[[noreturn, gnu::format(printf, 1, 2)]]
void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  std::exit(2);
}

#define HIP_ASSERT(expr)                                                       \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess)                                                      \
      die("HIP error %d (%s) at %s:%d: %s", static_cast<int>(_e),              \
          hipGetErrorString(_e), __FILE__, __LINE__, #expr);                   \
  } while (0)

std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) die("cannot open %s: %s", path.c_str(), std::strerror(errno));
  auto n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (!f.read(reinterpret_cast<char *>(buf.data()), buf.size()))
    die("short read on %s", path.c_str());
  return buf;
}

void writeFile(const std::string &path, const std::vector<uint8_t> &buf) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) die("cannot create %s: %s", path.c_str(), std::strerror(errno));
  f.write(reinterpret_cast<const char *>(buf.data()), buf.size());
  if (!f) die("short write on %s", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe type
// ─────────────────────────────────────────────────────────────────────────────

struct Recipe {
  std::string name;

  // Sweep dimensions.  The harness cross-products Ns × Blocks unless the
  // user restricts with --shape= / --block=.
  std::vector<int> defaultNs;
  std::vector<int> defaultBlocks;

  // Serialize deterministic inputs for the given N.  Inputs are
  // block-size-independent; only N shapes the buffer.
  std::function<std::vector<uint8_t>(int N)> makeInput;

  // Output element count for a given (N, block).  Used for diff width
  // and allocation.  For most recipes this depends on N only; for
  // block-level reductions it depends on both.
  std::function<int(int N, int blockSize)> outputElems;

  // Size (in bytes) of one output element (for memcpy/allocation).
  int outputElemBytes;

  // CPU reference: inputs + (N, blockSize) → expected output bytes of
  // size outputElems(N, blockSize) * outputElemBytes.
  std::function<std::vector<uint8_t>(const std::vector<uint8_t> &input,
                                     int N, int blockSize)>
      cpuReference;

  // Dispatch: given a loaded HIP module, run the kernel with the given
  // inputs at (N, blockSize), return output bytes.  Kernarg layout and
  // launch geometry live here.
  std::function<std::vector<uint8_t>(hipModule_t mod,
                                     const std::vector<uint8_t> &input,
                                     int N, int blockSize)>
      dispatch;

  // Elementwise comparator over the first `outElems` output elements.
  // Returns (numMismatches, maxAbsErr, firstMismatchIndex,
  // firstExpected, firstActual).
  std::function<std::tuple<int, double, int, double, double>(
      const std::vector<uint8_t> &gold,
      const std::vector<uint8_t> &actual, int outElems)>
      compare;

  // Optional guard on (N, block).  If the recipe cannot produce a
  // meaningful comparison for this combination (e.g. the CPU reference
  // would depend on warpSize, or N must satisfy some parity), return a
  // short reason string and the harness will skip the combo with a
  // visible log line.  Returning nullopt (or leaving `validate` unset)
  // means the combination is accepted.
  std::function<std::optional<std::string>(int N, int blockSize)> validate;
};

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: vecadd (sanity; no cross-lane ops)
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeVecaddRecipe() {
  Recipe r;
  r.name = "vecadd";
  // Shapes include the warp-boundary regime (±1 around 32/64) to catch
  // off-by-one guard bugs introduced by any of the translation paths.
  r.defaultNs     = {16, 32, 33, 64, 65, 128, 256, 257, 1024, 4096, 65537};
  r.defaultBlocks = {32, 64, 128, 256, 512};
  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(2 * N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(12345 + N);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) f[i] = dist(rng);
    for (int i = 0; i < N; ++i) f[N + i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *a = reinterpret_cast<const float *>(input.data());
    const float *b = a + N;
    std::vector<uint8_t> out(N * sizeof(float));
    float *c = reinterpret_cast<float *>(out.data());
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "vecadd"));

    float *dA, *dB, *dC;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dA, bytes));
    HIP_ASSERT(hipMalloc(&dB, bytes));
    HIP_ASSERT(hipMalloc(&dC, bytes));
    // Initialise output with an obvious sentinel so un-written outputs
    // are visible in the comparison.
    HIP_ASSERT(hipMemset(dC, 0xA5, bytes));

    HIP_ASSERT(hipMemcpy(dA, input.data(),          bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dB, input.data() + bytes,  bytes, hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const float *a;
      const float *b;
      float *c;
      int n;
    } args = {dA, dB, dC, N};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};

    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0, nullptr,
                                     nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());

    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dC, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dA));
    HIP_ASSERT(hipFree(dB));
    HIP_ASSERT(hipFree(dC));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    double maxAbs = 0.0;
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    const double tol = 1e-5;
    for (int i = 0; i < outElems; ++i) {
      double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(g[i]));
      if (d > maxAbs) maxAbs = d;
      if (d > tol) {
        if (mismatches++ == 0) { firstIdx = i; firstG = g[i]; firstA = a[i]; }
      }
    }
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: block_sum_shfl (wave-size-agnostic output: one sum per block)
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeBlockSumRecipe() {
  Recipe r;
  r.name = "block_sum_shfl";
  r.defaultNs = {64, 128, 256, 1024, 4096, 65536};
  // Block >= 64 keeps both gfx942 (warpSize=64) and gfx1250 (warpSize=32)
  // with at least one full warp active; block < warpSize makes phase-1
  // shuffle read inactive lanes and the gfx942 native reference
  // becomes ISA-dependent, breaking the apples-to-apples compare.
  r.defaultBlocks = {64, 128, 256, 512};
  r.validate = [](int, int blockSize) -> std::optional<std::string> {
    if (blockSize < 64)
      return std::string("block<64: warp reduction would read inactive lanes "
                         "on gfx942 (warpSize=64)");
    return std::nullopt;
  };
  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int blockSize) {
    return (N + blockSize - 1) / blockSize;
  };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(67890 + N);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) f[i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int blockSize) {
    const float *in = reinterpret_cast<const float *>(input.data());
    int nBlocks = (N + blockSize - 1) / blockSize;
    std::vector<uint8_t> out(nBlocks * sizeof(float));
    float *o = reinterpret_cast<float *>(out.data());
    for (int b = 0; b < nBlocks; ++b) {
      double s = 0.0;  // double for reproducibility
      int start = b * blockSize;
      int end = std::min(start + blockSize, N);
      for (int i = start; i < end; ++i) s += in[i];
      o[b] = static_cast<float>(s);
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "block_sum_shfl"));

    int nBlocks = (N + blockSize - 1) / blockSize;

    float *dIn, *dOut;
    HIP_ASSERT(hipMalloc(&dIn, N * sizeof(float)));
    HIP_ASSERT(hipMalloc(&dOut, nBlocks * sizeof(float)));
    HIP_ASSERT(hipMemset(dOut, 0xA5, nBlocks * sizeof(float)));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), N * sizeof(float),
                         hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const float *in;
      float *out;
      int n;
    } args = {dIn, dOut, N};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};

    HIP_ASSERT(hipModuleLaunchKernel(fn, nBlocks, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());

    std::vector<uint8_t> out(nBlocks * sizeof(float));
    HIP_ASSERT(hipMemcpy(out.data(), dOut, nBlocks * sizeof(float),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    double maxAbs = 0.0;
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    // Tolerance is loose because warp-reduction ordering differs from
    // the naïve CPU summation.
    const double tol = 5e-4;
    for (int i = 0; i < outElems; ++i) {
      double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(g[i]));
      if (d > maxAbs) maxAbs = d;
      if (d > tol) {
        if (mismatches++ == 0) { firstIdx = i; firstG = g[i]; firstA = a[i]; }
      }
    }
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: lane_swap (1:1 output; unwritten slots stay at sentinel)
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeLaneSwapRecipe() {
  Recipe r;
  r.name = "lane_swap";
  // Keep shapes even so the pairwise-swap reference is unambiguous at the
  // boundary.  For odd N the last element's XOR partner lies past the
  // array; the kernel's behaviour there depends on whether the partner
  // lane is active in its wave, which makes the answer ISA-dependent.
  r.defaultNs     = {16, 32, 64, 128, 256, 1024, 4096, 65536};
  r.defaultBlocks = {32, 64, 128, 256};
  r.validate = [](int N, int) -> std::optional<std::string> {
    if (N % 2 != 0)
      return std::string("N must be even: last lane's XOR=1 partner "
                         "lies past the array, answer becomes ISA-dependent");
    return std::nullopt;
  };
  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    for (int i = 0; i < N; ++i) f[i] = static_cast<float>(i) + 0.5f;
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *in = reinterpret_cast<const float *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    float *o = reinterpret_cast<float *>(out.data());
    for (int i = 0; i + 1 < N; i += 2) {
      o[i]     = in[i + 1];
      o[i + 1] = in[i];
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "lane_swap"));

    float *dIn, *dOut;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    // Sentinel: NaN pattern — unwritten lanes are crisply visible.
    uint32_t sentinel = 0x7FC00000u;
    std::vector<uint32_t> sentinelHost(N, sentinel);
    HIP_ASSERT(hipMemcpy(dOut, sentinelHost.data(), bytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const float *in;
      float *out;
      int n;
    } args = {dIn, dOut, N};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};

    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0, nullptr,
                                     nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());

    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    double maxAbs = 0.0;
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    for (int i = 0; i < outElems; ++i) {
      // NaN in actual: treat as mismatch (unwritten sentinel).
      bool nanActual = std::isnan(a[i]);
      double d = nanActual ? std::numeric_limits<double>::infinity()
                           : std::fabs(static_cast<double>(a[i]) -
                                       static_cast<double>(g[i]));
      if (std::isfinite(d) && d > maxAbs) maxAbs = d;
      if (nanActual || d > 1e-5) {
        if (mismatches++ == 0) { firstIdx = i; firstG = g[i]; firstA = a[i]; }
      }
    }
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers for the cvt_* recipes.  Both live here rather than in the kernels
// because they define the CPU reference, not any device-side behaviour.
// ─────────────────────────────────────────────────────────────────────────────

// f32 -> f16 with round-toward-zero, matching v_cvt_pkrtz_f16_f32
// semantics: NaN -> qNaN with sign preserved; +/-Inf -> +/-Inf; overflow
// (|v| >= 65520) saturates to the largest finite f16 in that direction;
// normals truncate their 13 low mantissa bits; subnormals truncate the
// bits that shift past the denormal exponent.
uint16_t f32_to_f16_rtz(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  uint32_t sign = (u >> 31) & 1;
  uint32_t exp  = (u >> 23) & 0xffu;
  uint32_t mant = u & 0x7fffffu;
  if (exp == 0xff) {
    if (mant) return static_cast<uint16_t>((sign << 15) | 0x7e00);
    return static_cast<uint16_t>((sign << 15) | 0x7c00);
  }
  if (exp == 0) {
    return static_cast<uint16_t>(sign << 15);
  }
  int32_t e = static_cast<int32_t>(exp) - 127 + 15;
  if (e >= 31) {
    return static_cast<uint16_t>((sign << 15) | 0x7bff);
  }
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(sign << 15);
    uint32_t m = mant | 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - e);
    return static_cast<uint16_t>((sign << 15) | (m >> shift));
  }
  return static_cast<uint16_t>((sign << 15) |
                               (static_cast<uint32_t>(e) << 10) |
                               (mant >> 13));
}

// Exact CPU comparator over N u32s.  Used by the integer-output recipes
// (cvt_pkrtz, cvt_pk_f16, bfm_b32, swap_b32) where a single bit of
// difference is a bug, not numerical noise.
std::tuple<int, double, int, double, double>
compareU32Exact(const std::vector<uint8_t> &gold,
                const std::vector<uint8_t> &actual, int n) {
  const uint32_t *g = reinterpret_cast<const uint32_t *>(gold.data());
  const uint32_t *a = reinterpret_cast<const uint32_t *>(actual.data());
  int mismatches = 0, firstIdx = -1;
  double firstG = 0.0, firstA = 0.0;
  for (int i = 0; i < n; ++i) {
    if (g[i] != a[i]) {
      if (mismatches++ == 0) {
        firstIdx = i;
        firstG = static_cast<double>(g[i]);
        firstA = static_cast<double>(a[i]);
      }
    }
  }
  // maxAbsErr is not meaningful for bit-exact compare; return 0 when
  // all match, 1 otherwise so the existing grid shows a signal.
  double maxAbs = (mismatches == 0) ? 0.0 : 1.0;
  return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: cvt_pkrtz — V_CVT_PKRTZ_F16_F32 handler
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeCvtPkrtzRecipe() {
  Recipe r;
  r.name = "cvt_pkrtz";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(2 * N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(0xBEEF + N);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < 2 * N; ++i) f[i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *a = reinterpret_cast<const float *>(input.data());
    const float *b = a + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *o = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i < N; ++i) {
      uint32_t lo = f32_to_f16_rtz(a[i]);
      uint32_t hi = f32_to_f16_rtz(b[i]);
      o[i] = lo | (hi << 16);
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "cvt_pkrtz"));
    float *dA, *dB;
    uint32_t *dC;
    size_t inBytes = N * sizeof(float);
    size_t outBytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dA, inBytes));
    HIP_ASSERT(hipMalloc(&dB, inBytes));
    HIP_ASSERT(hipMalloc(&dC, outBytes));
    HIP_ASSERT(hipMemset(dC, 0xA5, outBytes));
    HIP_ASSERT(hipMemcpy(dA, input.data(),           inBytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dB, input.data() + inBytes, inBytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const float *a; const float *b; uint32_t *c; int n; }
        args = {dA, dB, dC, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(outBytes);
    HIP_ASSERT(hipMemcpy(out.data(), dC, outBytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dA)); HIP_ASSERT(hipFree(dB)); HIP_ASSERT(hipFree(dC));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: cvt_pk_f16 — V_CVT_PK_F16_F32 handler
// Same interface as cvt_pkrtz but round-to-nearest-even (default f16
// cast on CPU).  gfx942 native lowers this without the packed opcode
// but produces the same f16 bit pattern per lane, so the comparison
// works across all three engines.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeCvtPkF16Recipe() {
  Recipe r;
  r.name = "cvt_pk_f16";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(2 * N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(0xCAFE + N);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < 2 * N; ++i) f[i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *a = reinterpret_cast<const float *>(input.data());
    const float *b = a + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *o = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i < N; ++i) {
      _Float16 lo = static_cast<_Float16>(a[i]);
      _Float16 hi = static_cast<_Float16>(b[i]);
      uint16_t blo, bhi;
      std::memcpy(&blo, &lo, sizeof(blo));
      std::memcpy(&bhi, &hi, sizeof(bhi));
      o[i] = static_cast<uint32_t>(blo) |
             (static_cast<uint32_t>(bhi) << 16);
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "cvt_pk_f16"));
    float *dA, *dB;
    uint32_t *dC;
    size_t inBytes = N * sizeof(float);
    size_t outBytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dA, inBytes));
    HIP_ASSERT(hipMalloc(&dB, inBytes));
    HIP_ASSERT(hipMalloc(&dC, outBytes));
    HIP_ASSERT(hipMemset(dC, 0xA5, outBytes));
    HIP_ASSERT(hipMemcpy(dA, input.data(),           inBytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dB, input.data() + inBytes, inBytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const float *a; const float *b; uint32_t *c; int n; }
        args = {dA, dB, dC, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(outBytes);
    HIP_ASSERT(hipMemcpy(out.data(), dC, outBytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dA)); HIP_ASSERT(hipFree(dB)); HIP_ASSERT(hipFree(dC));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: bfm_b32 — V_BFM_B32 handler
// Inputs: width[N], offset[N].  Only the low 5 bits of each are used,
// matching the hardware.  Output: ((1<<w)-1) << off per lane.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeBfmB32Recipe() {
  Recipe r;
  r.name = "bfm_b32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(2 * N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    std::mt19937 rng(0xF00D + N);
    // Sweep all five-bit widths and offsets — the whole interesting
    // space is 0..31 for each, so uniformly cover it.
    for (int i = 0; i < N; ++i) u[i]     = rng();
    for (int i = 0; i < N; ++i) u[N + i] = rng();
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(input.data());
    const uint32_t *o = w + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *c = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i < N; ++i) {
      uint32_t wi = w[i] & 31u;
      uint32_t oi = o[i] & 31u;
      uint32_t mask = (wi == 0) ? 0u : ((1u << wi) - 1u);
      c[i] = mask << oi;
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "bfm_b32"));
    uint32_t *dW, *dO, *dC;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dW, bytes));
    HIP_ASSERT(hipMalloc(&dO, bytes));
    HIP_ASSERT(hipMalloc(&dC, bytes));
    HIP_ASSERT(hipMemset(dC, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dW, input.data(),         bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dO, input.data() + bytes, bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint32_t *w; const uint32_t *o; uint32_t *c; int n; }
        args = {dW, dO, dC, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dC, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dW)); HIP_ASSERT(hipFree(dO)); HIP_ASSERT(hipFree(dC));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: swap_b32 — V_SWAP_B32 handler
// Pairwise exchange of adjacent elements.  N must be even; the
// validate hook rejects odd N rather than producing garbage for the
// trailing lane.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeSwapB32Recipe() {
  Recipe r;
  r.name = "swap_b32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.validate = [](int N, int) -> std::optional<std::string> {
    if (N % 2 != 0)
      return std::string("swap_b32 consumes input in pairs; N must be even");
    return std::nullopt;
  };
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    for (int i = 0; i < N; ++i) u[i] = 0xA55A0000u + static_cast<uint32_t>(i);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint32_t *in = reinterpret_cast<const uint32_t *>(input.data());
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *o = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i + 1 < N; i += 2) {
      o[i]     = in[i + 1];
      o[i + 1] = in[i];
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "swap_b32"));
    uint32_t *dIn, *dOut;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint32_t *in; uint32_t *out; int n; }
        args = {dIn, dOut, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    // The kernel's tid selects pair index (N/2 pairs).
    int pairs = N / 2;
    int grd = (pairs + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn)); HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: mov_b64 — V_MOV_B64 handler
// 64-bit copy; gfx942 uses a natural move, gfx1250 uses v_mov_b64 via
// inline asm so the raiser sees the opcode.  Expected output equals
// input byte-for-byte.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeMovB64Recipe() {
  Recipe r;
  r.name = "mov_b64";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint64_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(uint64_t));
    auto *u = reinterpret_cast<uint64_t *>(buf.data());
    std::mt19937_64 rng(0xDEAD5E7 + N);
    for (int i = 0; i < N; ++i) u[i] = rng();
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int, int) {
    return input;  // identity
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "mov_b64"));
    uint64_t *dIn, *dOut;
    size_t bytes = N * sizeof(uint64_t);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint64_t *a; uint64_t *c; int n; }
        args = {dIn, dOut, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn)); HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    const uint64_t *g = reinterpret_cast<const uint64_t *>(gold.data());
    const uint64_t *a = reinterpret_cast<const uint64_t *>(actual.data());
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    for (int i = 0; i < n; ++i) {
      if (g[i] != a[i]) {
        if (mismatches++ == 0) {
          firstIdx = i;
          firstG = static_cast<double>(g[i]);
          firstA = static_cast<double>(a[i]);
        }
      }
    }
    double maxAbs = (mismatches == 0) ? 0.0 : 1.0;
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: cvt_f32_bf16 — V_CVT_F32_BF16 handler
// bf16 -> f32 is an exact upcast (low 16 bits of the f32 are zero and
// high 16 bits are the bf16 bit pattern), so the CPU reference and
// both compile paths agree bit-for-bit.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeCvtF32Bf16Recipe() {
  Recipe r;
  r.name = "cvt_f32_bf16";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(uint16_t));
    auto *u = reinterpret_cast<uint16_t *>(buf.data());
    std::mt19937 rng(0xB16B16 + N);
    // Cover the whole representable bf16 space, including NaN/Inf.
    std::uniform_int_distribution<uint32_t> dist(0, 0xffffu);
    for (int i = 0; i < N; ++i) u[i] = static_cast<uint16_t>(dist(rng));
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint16_t *in = reinterpret_cast<const uint16_t *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    auto *o = reinterpret_cast<float *>(out.data());
    for (int i = 0; i < N; ++i) {
      uint32_t bits = static_cast<uint32_t>(in[i]) << 16;
      std::memcpy(&o[i], &bits, sizeof(float));
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "cvt_f32_bf16"));
    uint16_t *dIn;
    float *dOut;
    size_t inBytes = N * sizeof(uint16_t);
    size_t outBytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dIn, inBytes));
    HIP_ASSERT(hipMalloc(&dOut, outBytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, outBytes));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), inBytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint16_t *a; float *c; int n; }
        args = {dIn, dOut, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(outBytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, outBytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn)); HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    // Bit-exact on u32 reinterpretation of floats.  Necessary because
    // NaN inputs produce NaNs whose bit pattern must be preserved, and
    // naive float subtraction would treat any NaN as mismatch noise.
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: v_add_lshl_u32 — V_ADD_LSHL_U32 handler
// Inputs: a[N], b[N], c[N] (u32).  Output per lane: ((a+b) << (c & 31)).
// Only the low 5 bits of c matter, matching the hardware.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeVAddLshlU32Recipe() {
  Recipe r;
  r.name = "v_add_lshl_u32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(3 * N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    std::mt19937 rng(0xADD1u + N);
    for (int i = 0; i < 3 * N; ++i) u[i] = rng();
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint32_t *a = reinterpret_cast<const uint32_t *>(input.data());
    const uint32_t *b = a + N;
    const uint32_t *c = b + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *o = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i < N; ++i) {
      uint32_t sum = a[i] + b[i];
      o[i] = sum << (c[i] & 31u);
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "v_add_lshl_u32"));
    uint32_t *dA, *dB, *dC, *dO;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dA, bytes));
    HIP_ASSERT(hipMalloc(&dB, bytes));
    HIP_ASSERT(hipMalloc(&dC, bytes));
    HIP_ASSERT(hipMalloc(&dO, bytes));
    HIP_ASSERT(hipMemset(dO, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dA, input.data(),             bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dB, input.data() + bytes,     bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dC, input.data() + 2 * bytes, bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint32_t *a; const uint32_t *b; const uint32_t *c;
                             uint32_t *o; int n; }
        args = {dA, dB, dC, dO, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dO, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dA)); HIP_ASSERT(hipFree(dB));
    HIP_ASSERT(hipFree(dC)); HIP_ASSERT(hipFree(dO));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: v_bfe_i32 — V_BFE_I32 handler
// Inputs: src[N] (u32 bit pattern, interpreted signed by the extract),
// off[N], w[N].  Only the low 5 bits of off/w matter.  Output per lane
// is the sign-extended bit-field.  Zero-width is defined to produce 0.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeVBfeI32Recipe() {
  Recipe r;
  r.name = "v_bfe_i32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(3 * N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    std::mt19937 rng(0xBFE1u + N);
    for (int i = 0; i < N; ++i) u[i] = rng();
    // Uniformly sweep 0..31 for both offset and width so every
    // interesting low-5-bits case (including width=0 and
    // offset+width > 32) is covered per shape.
    for (int i = 0; i < N; ++i) u[N + i]     = rng() & 31u;
    for (int i = 0; i < N; ++i) u[2 * N + i] = rng() & 31u;
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint32_t *s = reinterpret_cast<const uint32_t *>(input.data());
    const uint32_t *o = s + N;
    const uint32_t *w = o + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *c = reinterpret_cast<uint32_t *>(out.data());
    // Hardware v_bfe_i32:
    //   D = sign_ext(bits [off+width-1:off] of src, where src is treated
    //       as a signed 32-bit value extended past bit 31 by its sign).
    // Equivalent formula: ashr by off (so high bits carry src's sign),
    // then mask to width bits and sign-extend from bit (width-1).
    // Logical right shift would match hardware only for off+width <= 32;
    // ashr handles the wraparound case where off+width > 32 by letting
    // the extracted field inherit src[31].
    for (int i = 0; i < N; ++i) {
      uint32_t off = o[i] & 31u;
      uint32_t width = w[i] & 31u;
      if (width == 0) { c[i] = 0u; continue; }
      int32_t sshifted = static_cast<int32_t>(s[i]) >> off;
      uint32_t mask = (1u << width) - 1u;
      uint32_t field = static_cast<uint32_t>(sshifted) & mask;
      uint32_t signBit = 1u << (width - 1);
      int32_t sx = static_cast<int32_t>((field ^ signBit) - signBit);
      c[i] = static_cast<uint32_t>(sx);
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "v_bfe_i32"));
    uint32_t *dS, *dO, *dW, *dOut;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dS, bytes));
    HIP_ASSERT(hipMalloc(&dO, bytes));
    HIP_ASSERT(hipMalloc(&dW, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dS, input.data(),             bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dO, input.data() + bytes,     bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dW, input.data() + 2 * bytes, bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint32_t *s; const uint32_t *o; const uint32_t *w;
                             uint32_t *out; int n; }
        args = {dS, dO, dW, dOut, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dS)); HIP_ASSERT(hipFree(dO));
    HIP_ASSERT(hipFree(dW)); HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: s_bfe_i32 — S_BFE_I32 handler (scalar signed BFE)
// One output per block (lane 0 writes).  The kernel reads a per-block
// src and ctrl via readfirstlane so the inline asm sees SGPR inputs.
// ctrl packs offset in bits [4:0] and width in [22:16].
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeSBfeI32Recipe() {
  Recipe r;
  r.name = "s_bfe_i32";
  // Use moderate Ns — N here counts blocks (one output per block), so
  // this stresses grid dimensions rather than per-block work.
  r.defaultNs     = {16, 64, 256, 1024};
  r.defaultBlocks = {64, 128};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(2 * N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    std::mt19937 rng(0x5BFE1u + N);
    for (int i = 0; i < N; ++i) u[i] = rng();
    // ctrl[i]: pack off in [4:0] and width in [22:16].  Sweep 0..31 for
    // off and 0..32 for width to exercise width>=32 (which hardware
    // treats as "full 32 bits, sign-extend from bit 31"), plus zero
    // width.
    for (int i = 0; i < N; ++i) {
      uint32_t off = rng() & 31u;
      uint32_t w = rng() % 33u; // 0..32 inclusive
      u[N + i] = off | (w << 16);
    }
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint32_t *src = reinterpret_cast<const uint32_t *>(input.data());
    const uint32_t *ctrl = src + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *c = reinterpret_cast<uint32_t *>(out.data());
    // Hardware s_bfe_i32 matches:
    //   if length == 0: D = 0
    //   elif shift + length < 32:
    //       D = sign_ext((src << (32 - shift - length)) >> (32 - length))
    //   else: D = (int32)src >> shift   (length saturates, full width)
    // Using the shift-trick rather than "mask & sign-extend" so we stay
    // bit-identical to native when shift + length >= 32.
    for (int i = 0; i < N; ++i) {
      uint32_t shift = ctrl[i] & 0x1Fu;
      uint32_t length = (ctrl[i] >> 16) & 0x7Fu;
      if (length == 0) { c[i] = 0u; continue; }
      uint32_t sum = shift + length;
      int32_t sx;
      if (sum < 32u) {
        uint32_t shlAmt = 32u - sum;
        uint32_t shlVal = src[i] << shlAmt;
        uint32_t shrAmt = 32u - length;
        sx = static_cast<int32_t>(shlVal) >> shrAmt;
      } else {
        sx = static_cast<int32_t>(src[i]) >> shift;
      }
      c[i] = static_cast<uint32_t>(sx);
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "s_bfe_i32"));
    uint32_t *dSrc, *dCtrl, *dOut;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dSrc, bytes));
    HIP_ASSERT(hipMalloc(&dCtrl, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dSrc,  input.data(),         bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dCtrl, input.data() + bytes, bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint32_t *s; const uint32_t *ctrl; uint32_t *o; int n; }
        args = {dSrc, dCtrl, dOut, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    // One block per output element; only lane 0 does the scalar op.
    HIP_ASSERT(hipModuleLaunchKernel(fn, N, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dSrc)); HIP_ASSERT(hipFree(dCtrl)); HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: s_bitset0_b32 — S_BITSET0_B32 handler (scalar RMW bit-clear)
// Per block, clear bit `pos[block] & 31` in a scalar loaded from
// `initial[block]` and write the post-RMW value to `out[block]`.
// Exercises the tied `sdst_in` input path in the raiser.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeSBitset0B32Recipe() {
  Recipe r;
  r.name = "s_bitset0_b32";
  r.defaultNs     = {16, 64, 256, 1024};
  r.defaultBlocks = {64, 128};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(2 * N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    std::mt19937 rng(0x5B170u + N);
    for (int i = 0; i < N; ++i) u[i] = rng();
    for (int i = 0; i < N; ++i) u[N + i] = rng();
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint32_t *ini = reinterpret_cast<const uint32_t *>(input.data());
    const uint32_t *pos = ini + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *o = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i < N; ++i) {
      o[i] = ini[i] & ~(1u << (pos[i] & 31u));
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "s_bitset0_b32"));
    uint32_t *dIni, *dPos, *dOut;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dIni, bytes));
    HIP_ASSERT(hipMalloc(&dPos, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dIni, input.data(),         bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dPos, input.data() + bytes, bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const uint32_t *ini; const uint32_t *pos; uint32_t *o; int n; }
        args = {dIni, dPos, dOut, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    HIP_ASSERT(hipModuleLaunchKernel(fn, N, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIni)); HIP_ASSERT(hipFree(dPos)); HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: s_bitset0_b64 — S_BITSET0_B64 handler (scalar 64-bit RMW bit clear)
// One output per block (lane 0 writes).  Bit index is a 32-bit SGPR, but
// only [5:0] are consumed by hardware; sdst and tied sdst_in are SReg_64.
// Exercises the 64-bit sibling of the B32 variant, including the extra
// SGPR pair write-back.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeSBitset0B64Recipe() {
  Recipe r;
  r.name = "s_bitset0_b64";
  r.defaultNs     = {16, 64, 256, 1024};
  r.defaultBlocks = {64, 128};
  r.outputElemBytes = sizeof(uint64_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    // Layout: N × uint64_t initial, then N × uint32_t bit-positions.
    std::vector<uint8_t> buf(N * sizeof(uint64_t) + N * sizeof(uint32_t));
    auto *u64 = reinterpret_cast<uint64_t *>(buf.data());
    std::mt19937_64 rng64(0x5B170ULL + N);
    for (int i = 0; i < N; ++i) u64[i] = rng64();
    auto *u32 =
        reinterpret_cast<uint32_t *>(buf.data() + N * sizeof(uint64_t));
    std::mt19937 rng32(0x5B171u + N);
    for (int i = 0; i < N; ++i) u32[i] = rng32();
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint64_t *ini = reinterpret_cast<const uint64_t *>(input.data());
    const uint32_t *pos = reinterpret_cast<const uint32_t *>(
        input.data() + N * sizeof(uint64_t));
    std::vector<uint8_t> out(N * sizeof(uint64_t));
    auto *o = reinterpret_cast<uint64_t *>(out.data());
    for (int i = 0; i < N; ++i) {
      o[i] = ini[i] & ~(uint64_t(1) << (pos[i] & 0x3Fu));
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "s_bitset0_b64"));
    uint64_t *dIni, *dOut;
    uint32_t *dPos;
    size_t initialBytes = N * sizeof(uint64_t);
    size_t posBytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dIni, initialBytes));
    HIP_ASSERT(hipMalloc(&dPos, posBytes));
    HIP_ASSERT(hipMalloc(&dOut, initialBytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, initialBytes));
    HIP_ASSERT(hipMemcpy(dIni, input.data(), initialBytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dPos, input.data() + initialBytes, posBytes,
                         hipMemcpyHostToDevice));
    struct alignas(8) Args {
      const uint64_t *ini;
      const uint32_t *pos;
      uint64_t *o;
      int n;
    } args = {dIni, dPos, dOut, N};
    size_t argSize = sizeof(args);
    void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    HIP_ASSERT(hipModuleLaunchKernel(fn, N, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, cfg));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(initialBytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, initialBytes,
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIni));
    HIP_ASSERT(hipFree(dPos));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual, int n) {
    // Elementwise 64-bit exact compare.  Mirrors compareU32Exact shape.
    const uint64_t *g = reinterpret_cast<const uint64_t *>(gold.data());
    const uint64_t *a = reinterpret_cast<const uint64_t *>(actual.data());
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    for (int i = 0; i < n; ++i) {
      if (g[i] != a[i]) {
        if (mismatches++ == 0) {
          firstIdx = i;
          firstG = static_cast<double>(g[i]);
          firstA = static_cast<double>(a[i]);
        }
      }
    }
    double maxAbs = (mismatches == 0) ? 0.0 : 1.0;
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe registry
// ─────────────────────────────────────────────────────────────────────────────

const std::vector<Recipe> &allRecipes() {
  static const std::vector<Recipe> v = {
      makeVecaddRecipe(),
      makeBlockSumRecipe(),
      makeLaneSwapRecipe(),
      makeCvtPkrtzRecipe(),
      makeCvtPkF16Recipe(),
      makeBfmB32Recipe(),
      makeSwapB32Recipe(),
      makeMovB64Recipe(),
      makeCvtF32Bf16Recipe(),
      makeVAddLshlU32Recipe(),
      makeVBfeI32Recipe(),
      makeSBfeI32Recipe(),
      makeSBitset0B32Recipe(),
      makeSBitset0B64Recipe(),
  };
  return v;
}

const Recipe *findRecipe(const std::string &name) {
  for (const auto &r : allRecipes())
    if (r.name == name) return &r;
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Child side: run one (mode, recipe, shape) and dump output to a file
// ─────────────────────────────────────────────────────────────────────────────

enum class Mode { Native, Legacy, Salmon };

const char *modeName(Mode m) {
  switch (m) {
    case Mode::Native: return "native";
    case Mode::Legacy: return "legacy";
    case Mode::Salmon: return "salmon";
  }
  return "?";
}

std::optional<Mode> parseMode(const std::string &s) {
  if (s == "native") return Mode::Native;
  if (s == "legacy") return Mode::Legacy;
  if (s == "salmon") return Mode::Salmon;
  return std::nullopt;
}

int runChild(Mode mode, const std::string &recipeName, int N, int blockSize,
             const std::string &inputFile, const std::string &outputFile) {
  // Pin the engine selection for this child process.
  switch (mode) {
    case Mode::Salmon:
      setenv("HSA_HOTSWAP_IR_RAISER", "1", /*ow=*/1);
      break;
    case Mode::Legacy:
      unsetenv("HSA_HOTSWAP_IR_RAISER");
      break;
    case Mode::Native:
      // Native mode does not go through the hotswap hook; the ISA
      // matches the agent.  Leave env vars unchanged (the parent set
      // or unset them appropriately before forking).
      break;
  }

  const Recipe *r = findRecipe(recipeName);
  if (!r) die("child: unknown recipe %s", recipeName.c_str());

  std::string isa = (mode == Mode::Native) ? "gfx942" : "gfx1250";
  std::string coPath = coPathFor(recipeName, isa);
  auto coBytes = readFile(coPath);

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, coBytes.data()));

  auto input = readFile(inputFile);
  auto output = r->dispatch(mod, input, N, blockSize);
  writeFile(outputFile, output);

  HIP_ASSERT(hipModuleUnload(mod));
  return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent side: spawn a child, wait, collect result
// ─────────────────────────────────────────────────────────────────────────────

struct ChildRun {
  bool launched = false;
  bool exitedCleanly = false;
  int exitCode = -1;
  int signal = 0;
  double durationMs = 0.0;
  std::string stderrTail;
};

std::string selfExe() {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n < 0) die("readlink /proc/self/exe failed: %s", std::strerror(errno));
  buf[n] = '\0';
  return std::string(buf);
}

std::string drainFd(int fd, size_t maxBytes) {
  std::string out;
  char buf[4096];
  while (out.size() < maxBytes) {
    ssize_t n = read(fd, buf, std::min(sizeof(buf), maxBytes - out.size()));
    if (n < 0) { if (errno == EINTR) continue; break; }
    if (n == 0) break;
    out.append(buf, static_cast<size_t>(n));
  }
  return out;
}

std::string tail(const std::string &s, size_t n) {
  return s.size() <= n ? s : s.substr(s.size() - n);
}

ChildRun spawnChild(const std::string &exe, Mode mode,
                    const std::string &recipe, int N, int blockSize,
                    const std::string &inputFile,
                    const std::string &outputFile) {
  ChildRun cr;
  int errPipe[2];
  if (pipe(errPipe) != 0) {
    cr.stderrTail = std::string("pipe: ") + std::strerror(errno);
    return cr;
  }
  auto t0 = std::chrono::steady_clock::now();
  pid_t pid = fork();
  if (pid < 0) {
    close(errPipe[0]); close(errPipe[1]);
    cr.stderrTail = std::string("fork: ") + std::strerror(errno);
    return cr;
  }
  if (pid == 0) {
    // Child: silence stdout (to keep parent's report clean), capture stderr.
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) { dup2(devnull, STDOUT_FILENO); close(devnull); }
    dup2(errPipe[1], STDERR_FILENO);
    close(errPipe[0]); close(errPipe[1]);

    std::string nArg  = "--N="      + std::to_string(N);
    std::string bArg  = "--B="      + std::to_string(blockSize);
    std::string rArg  = "--recipe=" + recipe;
    std::string mArg  = "--child="  + std::string(modeName(mode));
    std::string iArg  = "--input="  + inputFile;
    std::string oArg  = "--output=" + outputFile;
    char *argv[] = {
        const_cast<char *>(exe.c_str()),
        const_cast<char *>(mArg.c_str()),
        const_cast<char *>(rArg.c_str()),
        const_cast<char *>(nArg.c_str()),
        const_cast<char *>(bArg.c_str()),
        const_cast<char *>(iArg.c_str()),
        const_cast<char *>(oArg.c_str()),
        nullptr,
    };
    execv(exe.c_str(), argv);
    fprintf(stderr, "execv failed: %s\n", std::strerror(errno));
    _exit(127);
  }
  close(errPipe[1]);
  std::string err = drainFd(errPipe[0], /*maxBytes=*/32 * 1024);
  close(errPipe[0]);
  int wstatus = 0;
  for (;;) {
    pid_t w = waitpid(pid, &wstatus, 0);
    if (w == pid) break;
    if (w < 0 && errno == EINTR) continue;
    cr.stderrTail = std::string("waitpid: ") + std::strerror(errno);
    return cr;
  }
  auto t1 = std::chrono::steady_clock::now();
  cr.durationMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  cr.launched = true;
  if (WIFEXITED(wstatus)) {
    cr.exitedCleanly = true;
    cr.exitCode = WEXITSTATUS(wstatus);
  } else if (WIFSIGNALED(wstatus)) {
    cr.signal = WTERMSIG(wstatus);
  }
  cr.stderrTail = tail(err, 800);
  return cr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent side: run one (recipe, shape) against all three modes
// ─────────────────────────────────────────────────────────────────────────────

struct ModeResult {
  ChildRun child;
  // Present only if the child wrote a readable output file.
  std::optional<std::vector<uint8_t>> output;
  // Diff stats vs. CPU gold.
  int mismatches = -1;
  double maxAbsErr = 0.0;
  int firstIdx = -1;
  double firstGold = 0.0;
  double firstActual = 0.0;
};

struct RunResult {
  const Recipe *recipe;
  int N;
  int blockSize;
  int outElems;
  std::vector<uint8_t> cpuGold;
  ModeResult native, legacy, salmon;
};

std::string tempPath(const std::string &tag) {
  // Honor $TMPDIR so we can redirect the harness's per-run tempfiles off
  // a full /tmp. Each `(parent, child)` pair shares these paths by PID+tag,
  // so we pick the directory once here and apply it uniformly.
  const char *td = std::getenv("TMPDIR");
  std::string dir = (td && *td) ? td : "/tmp";
  return dir + "/cmp_correct_" + std::to_string(getpid()) + "_" + tag + ".bin";
}

RunResult runOne(const std::string &exe, const Recipe &r, int N, int blockSize) {
  RunResult rr;
  rr.recipe = &r;
  rr.N = N;
  rr.blockSize = blockSize;
  rr.outElems = r.outputElems(N, blockSize);

  auto input = r.makeInput(N);
  rr.cpuGold = r.cpuReference(input, N, blockSize);
  std::string inPath  = tempPath("in");
  writeFile(inPath, input);

  auto runMode = [&](Mode m, ModeResult &mr, const std::string &tag) {
    std::string outPath = tempPath(tag);
    unlink(outPath.c_str());
    mr.child = spawnChild(exe, m, r.name, N, blockSize, inPath, outPath);
    if (mr.child.exitedCleanly && mr.child.exitCode == 0) {
      struct stat st;
      if (stat(outPath.c_str(), &st) == 0 && st.st_size > 0) {
        mr.output = readFile(outPath);
      }
    }
    if (mr.output) {
      auto [m, mab, idx, g, a] = r.compare(rr.cpuGold, *mr.output, rr.outElems);
      mr.mismatches = m;
      mr.maxAbsErr = mab;
      mr.firstIdx = idx;
      mr.firstGold = g;
      mr.firstActual = a;
    }
    unlink(outPath.c_str());
  };

  runMode(Mode::Native, rr.native, "native");
  runMode(Mode::Legacy, rr.legacy, "legacy");
  runMode(Mode::Salmon, rr.salmon, "salmon");

  unlink(inPath.c_str());
  return rr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reporting
// ─────────────────────────────────────────────────────────────────────────────

// Short, fixed-width status cell for the main grid.
std::string statusStr(const ModeResult &mr, int totalElems) {
  if (!mr.child.launched) return "spawn-fail";
  if (mr.child.signal > 0) {
    char b[64];
    std::snprintf(b, sizeof(b), "SIG%d", mr.child.signal);
    return b;
  }
  if (!mr.child.exitedCleanly) return "no-exit";
  if (mr.child.exitCode != 0)  return "EXIT=" + std::to_string(mr.child.exitCode);
  if (!mr.output)              return "no-output";
  if (mr.mismatches == 0)      return "match";
  char b[64];
  std::snprintf(b, sizeof(b), "WRONG %d/%d", mr.mismatches, totalElems);
  return b;
}

// Classify a mode result for summary counts.
enum class ResultCat { Match, Mismatch, Crash };

ResultCat classify(const ModeResult &m) {
  if (!m.child.launched)                             return ResultCat::Crash;
  if (m.child.signal > 0)                            return ResultCat::Crash;
  if (!m.child.exitedCleanly)                        return ResultCat::Crash;
  if (m.child.exitCode != 0)                         return ResultCat::Crash;
  if (!m.output)                                     return ResultCat::Crash;
  if (m.mismatches == 0)                             return ResultCat::Match;
  return ResultCat::Mismatch;
}

// Collapse a stderr tail to a single line for compact display.
std::string oneLineTail(const std::string &s, size_t cap = 160) {
  std::string out;
  out.reserve(s.size());
  bool lastWasSpace = false;
  for (char c : s) {
    char cc = (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
    if (cc == ' ') {
      if (lastWasSpace) continue;
      lastWasSpace = true;
    } else {
      lastWasSpace = false;
    }
    out.push_back(cc);
  }
  size_t a = out.find_first_not_of(' ');
  size_t b = out.find_last_not_of(' ');
  if (a == std::string::npos) return "";
  out = out.substr(a, b - a + 1);
  if (out.size() > cap) out = out.substr(out.size() - cap) + " [tail]";
  return out;
}

// Rich, one-line diagnostic for the Failures section.
std::string failureDetail(const ModeResult &m, int outElems) {
  if (!m.child.launched) {
    return "spawn-fail";
  }
  if (m.output && m.mismatches > 0) {
    char b[256];
    std::snprintf(b, sizeof(b),
                  "WRONG %d/%d  @idx=%d  ref=%.6g  actual=%.6g  max|err|=%.6g",
                  m.mismatches, outElems, m.firstIdx,
                  m.firstGold, m.firstActual, m.maxAbsErr);
    return b;
  }
  std::string head;
  if (m.child.signal > 0)           head = "SIG" + std::to_string(m.child.signal);
  else if (!m.child.exitedCleanly)  head = "no-exit";
  else if (m.child.exitCode != 0)   head = "EXIT=" + std::to_string(m.child.exitCode);
  else if (!m.output)               head = "no-output";
  else                              return "";

  std::string tail = oneLineTail(m.child.stderrTail);
  if (tail.empty()) return head;
  return head + "  stderr: " + tail;
}

void printReport(const std::vector<RunResult> &all) {
  // ── Grid: one atomic row per (recipe, N, block). No interleaved lines. ──
  const int W_RECIPE = 16;
  const int W_N      = 7;
  const int W_B      = 6;
  const int W_STATUS = 18;  // fits "WRONG 65536/65536"

  printf("\n");
  printf("%-*s %-*s %-*s  %-*s  %-*s  %-*s\n",
         W_RECIPE, "recipe", W_N, "N", W_B, "block",
         W_STATUS, "native", W_STATUS, "legacy", W_STATUS, "salmon");
  int sepW = W_RECIPE + W_N + W_B + 3 * W_STATUS + 8;
  printf("%s\n", std::string(sepW, '-').c_str());

  // Counters for the summary matrix: cnt[mode][category].
  // mode: 0=native, 1=legacy, 2=salmon.
  int cnt[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

  // Failures grouped per mode, in input order.
  struct FailRow {
    std::string recipeName;
    int N;
    int blockSize;
    std::string detail;
  };
  std::vector<FailRow> fails[3];

  for (const auto &rr : all) {
    const ModeResult *mm[3] = {&rr.native, &rr.legacy, &rr.salmon};
    std::string cells[3];
    for (int i = 0; i < 3; ++i) {
      cells[i] = statusStr(*mm[i], rr.outElems);
      ResultCat c = classify(*mm[i]);
      ++cnt[i][static_cast<int>(c)];
      if (c != ResultCat::Match) {
        fails[i].push_back({rr.recipe->name, rr.N, rr.blockSize,
                            failureDetail(*mm[i], rr.outElems)});
      }
    }
    printf("%-*s %-*d %-*d  %-*s  %-*s  %-*s\n",
           W_RECIPE, rr.recipe->name.c_str(),
           W_N, rr.N, W_B, rr.blockSize,
           W_STATUS, cells[0].c_str(),
           W_STATUS, cells[1].c_str(),
           W_STATUS, cells[2].c_str());
  }

  // ── Failures section: full detail, grouped by mode. ──
  const char *modeLabel[3] = {"native", "legacy", "salmon"};
  const char *modeNote[3]  = {
      "  (native != reference — harness or CPU-reference bug; suspect other rows)",
      "", ""};
  bool anyFail = false;
  for (int i = 0; i < 3; ++i) if (!fails[i].empty()) { anyFail = true; break; }
  if (anyFail) {
    printf("\n=== Failures ===\n");
    for (int i = 0; i < 3; ++i) {
      if (fails[i].empty()) continue;
      printf("\n%s%s\n", modeLabel[i], modeNote[i]);
      for (const auto &f : fails[i]) {
        printf("  %-*s N=%-6d block=%-4d  %s\n",
               W_RECIPE, f.recipeName.c_str(), f.N, f.blockSize,
               f.detail.c_str());
      }
    }
  }

  // ── Summary matrix: categories × modes, side-by-side. ──
  size_t T = all.size();
  printf("\n=== Summary (%zu runs) ===\n", T);
  printf("  %-16s %8s %8s %8s\n", "", "native", "legacy", "salmon");
  auto row = [&](const char *label, ResultCat c) {
    printf("  %-16s %8d %8d %8d\n", label,
           cnt[0][static_cast<int>(c)],
           cnt[1][static_cast<int>(c)],
           cnt[2][static_cast<int>(c)]);
  };
  row("match",         ResultCat::Match);
  row("mismatch",      ResultCat::Mismatch);
  row("crash/no-exit", ResultCat::Crash);
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

struct Options {
  std::optional<Mode> childMode;
  std::string recipeFilter;
  std::vector<int> shapeFilter;   // restrict N (parent sweep)
  std::vector<int> blockFilter;   // restrict block size (parent sweep)
  std::string inputPath;          // child-only
  std::string outputPath;         // child-only
  int N = -1;                     // child-only
  int blockSize = -1;             // child-only
};

void printHelp(const char *argv0) {
  fprintf(stderr,
      "Usage: %s [options]\n"
      "\n"
      "  --recipe=<name>   run only the named recipe\n"
      "  --shape=<N>       restrict N values (repeatable). Cross-product\n"
      "                    with --block.  If omitted, the recipe's default\n"
      "                    N list is used.\n"
      "  --block=<B>       restrict block sizes (repeatable). Cross-product\n"
      "                    with --shape.  If omitted, the recipe's default\n"
      "                    block list is used.\n"
      "  -h | --help       show this help\n"
      "\n"
      "Internal child mode (used by the parent when spawning):\n"
      "  --child=<native|legacy|salmon> --recipe=<name> --N=<int> --B=<int>\n"
      "    --input=<path> --output=<path>\n", argv0);
}

bool parseArgs(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto eq = [&](const char *k, std::string &dst) {
      size_t kn = std::strlen(k);
      if (a.rfind(k, 0) != 0) return false;
      dst = a.substr(kn);
      return true;
    };
    std::string tmp;
    if (a == "-h" || a == "--help") { printHelp(argv[0]); std::exit(0); }
    else if (eq("--child=", tmp)) {
      auto m = parseMode(tmp);
      if (!m) { fprintf(stderr, "bad --child=%s\n", tmp.c_str()); return false; }
      opt.childMode = *m;
    } else if (eq("--recipe=", tmp)) {
      opt.recipeFilter = tmp;
    } else if (eq("--shape=", tmp)) {
      opt.shapeFilter.push_back(std::atoi(tmp.c_str()));
    } else if (eq("--block=", tmp)) {
      opt.blockFilter.push_back(std::atoi(tmp.c_str()));
    } else if (eq("--N=", tmp)) {
      opt.N = std::atoi(tmp.c_str());
    } else if (eq("--B=", tmp)) {
      opt.blockSize = std::atoi(tmp.c_str());
    } else if (eq("--input=", tmp)) {
      opt.inputPath = tmp;
    } else if (eq("--output=", tmp)) {
      opt.outputPath = tmp;
    } else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return false;
    }
  }
  return true;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt)) return 2;

  // Child invocation: single recipe/N/block/mode run.
  if (opt.childMode.has_value()) {
    if (opt.recipeFilter.empty() || opt.N < 0 || opt.blockSize < 0 ||
        opt.inputPath.empty() || opt.outputPath.empty()) {
      fprintf(stderr,
              "child mode requires --recipe, --N, --B, --input, --output\n");
      return 2;
    }
    return runChild(*opt.childMode, opt.recipeFilter, opt.N, opt.blockSize,
                    opt.inputPath, opt.outputPath);
  }

  // Parent: set the forwarded env that matters to the ROCR hotswap hook.
  // Children inherit these; they further toggle HSA_HOTSWAP_IR_RAISER.
  if (!std::getenv("HSA_HOTSWAP_ISA_OVERRIDE"))
    setenv("HSA_HOTSWAP_ISA_OVERRIDE", "gfx942", 1);
  if (!std::getenv("HSA_HOTSWAP_RULES"))
    setenv("HSA_HOTSWAP_RULES", "/dev/null", 1);

  std::string exe = selfExe();

  printf("=== compare_correctness ===\n");
  printf("  exe                  : %s\n", exe.c_str());
  printf("  kernel_dir           : %s\n", kernelDir().c_str());
  printf("  HSA_HOTSWAP_ISA_OVERRIDE = %s\n",
         std::getenv("HSA_HOTSWAP_ISA_OVERRIDE"));
  printf("  HSA_HOTSWAP_RULES        = %s\n",
         std::getenv("HSA_HOTSWAP_RULES"));
  printf("  LD_PRELOAD               = %s\n",
         std::getenv("LD_PRELOAD") ? std::getenv("LD_PRELOAD") : "(unset)");
  printf("  LD_LIBRARY_PATH          = %s\n",
         std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "(unset)");

  std::vector<RunResult> all;
  for (const auto &r : allRecipes()) {
    if (!opt.recipeFilter.empty() && r.name != opt.recipeFilter) continue;
    const std::vector<int> &ns =
        opt.shapeFilter.empty() ? r.defaultNs : opt.shapeFilter;
    const std::vector<int> &blocks =
        opt.blockFilter.empty() ? r.defaultBlocks : opt.blockFilter;
    for (int N : ns) {
      for (int B : blocks) {
        if (r.validate) {
          if (auto reason = r.validate(N, B)) {
            fprintf(stderr, "  [skip] recipe=%s N=%d B=%d  reason: %s\n",
                    r.name.c_str(), N, B, reason->c_str());
            continue;
          }
        }
        fprintf(stderr, "  [run] recipe=%s N=%d B=%d ...\n",
                r.name.c_str(), N, B);
        all.push_back(runOne(exe, r, N, B));
      }
    }
  }

  if (all.empty()) {
    fprintf(stderr,
            "no runs matched your filter (check --recipe / --shape / --block).\n");
    return 2;
  }

  printReport(all);
  return 0;  // An output mismatch is a finding, not a harness error.
}
