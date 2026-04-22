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
// write the device-side output to a tempfile.  The parent obtains a
// "gold" buffer that every child is judged against.  Where the gold
// comes from depends on the recipe's `goldSource`:
//
//   CpuReference    — the parent runs a hand-written CPU implementation
//                     and uses its output as the gold; native is judged
//                     against it just like legacy/salmon.  Used by HIP
//                     recipes.
//   NativeExecution — the native gfx942 child runs first and its output
//                     IS the gold; legacy/salmon are judged against
//                     that.  Used by Triton recipes (writing a CPU
//                     reference per kernel doesn't scale).
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
#include <poll.h>
#include <signal.h>
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
#include <map>
#include <unordered_map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
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

// Where each run's gold comes from.
//   CpuReference    — Recipe::cpuReference computes the expected output on the
//                     host.  Every mode (native / legacy / salmon) is judged
//                     against that external ground truth.  This is the HIP
//                     recipe default: it gives CPU-grounded verification of
//                     the native column too.
//   NativeExecution — The native gfx942 run *is* the gold.  legacy / salmon
//                     are judged against whatever native produced.  Used by
//                     Triton recipes, where authoring a CPU reference for
//                     every kernel is impractical; the trust boundary moves
//                     to "hipcc / Triton on gfx942 compute the right answer".
enum class GoldSource { CpuReference, NativeExecution };

struct Recipe {
  std::string name;

  // Where the gold comes from for this recipe.  Defaults to CpuReference so
  // every existing recipe is unchanged.
  GoldSource goldSource = GoldSource::CpuReference;

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
  // size outputElems(N, blockSize) * outputElemBytes.  Must be set when
  // goldSource == CpuReference; ignored (may be empty) when goldSource ==
  // NativeExecution.
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
  // firstExpected, firstActual).  N and blockSize are passed so recipes
  // whose output structure depends on the shape (e.g. Triton recipes
  // with multiple per-output dtype/comparator slices) can re-derive
  // their per-buffer layout without smuggling shape state across calls.
  // HIP recipes that ignore them are free to do so.
  std::function<std::tuple<int, double, int, double, double>(
      const std::vector<uint8_t> &gold,
      const std::vector<uint8_t> &actual,
      int N, int blockSize, int outElems)>
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
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
// Recipe: canary_readlane_last_lane — C1 per-source-wave broadcast rewrite guard
//
// Canary for the `v_readlane_b32 sDST, vSRC, K` pattern with K = W_s - 1
// — the "broadcast from the last lane of the source wave" idiom.  Pins
// the per-source-wave readlane emulation that
// `rewrite_cross_lane_divergent.cpp` performs on cross-widening lifts,
// and documents the matching wave32-source vs wave64-target semantic
// gap that a native-compiled gfx942 binary (which has no source-wave
// notion) cannot close.
//
// Gold semantics (CpuReference) encodes the wave32 source behaviour:
// per-wave broadcast of lane (W_s - 1)'s seed.  For the canonical
// workgroup of 64 lanes on gfx1250 wave32 (two source waves of 32
// each), that's:
//   * out[tid in  0..31] == in[31]   (source wave 0's lane 31)
//   * out[tid in 32..63] == in[63]   (source wave 1's lane 31)
//
// Observed verdicts at landing:
//   native : WRONG 32/64 (target-wave64 semantics, no source-wave
//            notion — inherent, not a hipcc bug).
//   legacy : WRONG 32/64 (text transpiler copies the mnemonic through
//            without any per-source-wave rewrite).
//   salmon : match on every shape under the runtime's
//            WaveNativeProjection default.  (The intercept library
//            does not expose an env-var MODREP override — see
//            `raiser.hpp`'s "No process-global env-var override"
//            comment for why — so the MODREP arm is only reachable
//            via `raise_cli --disable-wave-native` for offline
//            triage; `raise_cli` succeeds on this fixture under both
//            projections.)  The rewrite at
//            `rewrite_cross_lane_divergent.cpp` recognises the lifted
//            `@llvm.amdgcn.readlane(val, 31)` as a cross-wave-divergent
//            site and rewrites it to an on-target `ds_bpermute` that
//            reads lane `(lane_id & ~(W_s - 1)) | 31` — the "last lane
//            of THIS source-wave replica".  The resulting per-replica
//            broadcast matches the wave32-source gold bit-exactly.
//
// Regression contract (what a verdict drift means):
//
//   salmon `match` -> `WRONG 32/64`  ==  rewrite regression.  One of
//     (a) the classifier stopped flagging the site, (b) the rewrite
//     produced a wrong selector expression, or (c) a downstream
//     projection stopped threading `cwd_lane_id`.  The mismatch count
//     is fixed at half the output size, so the regression is loud.
//
//   salmon `match` -> `EXIT=2`  ==  classifier moved from "rewrite" to
//     "refuse".  Intentional only if a tighter §7.1 C1 validator lands
//     that refuses all wave-dependent lane constants; retire the
//     `match` expectation in that case, otherwise investigate.
//
//   native / legacy `WRONG` -> `match`  ==  CpuReference drift or
//     hipcc started doing source-wave-aware readlane emulation
//     (unlikely — hipcc has no source-wave notion).  Investigate
//     before relaxing the canary's failure-side expectation.
//
// See `hotswap/docs/gpt-oss-derisking.md` §7.1 / §9.2 item 5 and
// `kernels/canary_readlane_last_lane.hip` for the source shape and
// inline-asm contract.  The canary complements the existing
// `readlane_divergent_rewrite.ll` lit fixture: lit pins the IR shape
// the rewrite produces, the canary pins its end-to-end runtime
// correctness on real hardware.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeCanaryReadlaneLastLaneRecipe() {
  Recipe r;
  r.name = "canary_readlane_last_lane";
  // Fixed workgroup size of W_t = 64 (gfx942 wave64) gives the cleanest
  // demonstration of the miscompile: exactly one target wave covers the
  // output, and the two halves of that wave correspond one-to-one with
  // the two source waves the wave32 kernel ran.  Sweeping multiple N
  // values lets the regression signal distinguish "always WRONG 32/64"
  // (the miscompile class) from "WRONG only at small N" (harness bug)
  // or "WRONG only at large N" (some other coincidence).  Block size is
  // pinned to 64 for the same reason — any blockSize other than the
  // native target wave would muddy the per-wave-broadcast analysis.
  r.defaultNs     = {64, 128, 256, 1024};
  r.defaultBlocks = {64};
  r.validate = [](int N, int blockSize) -> std::optional<std::string> {
    if (blockSize != 64)
      return std::string("blockSize must be 64 (the target wave64 width): "
                         "this canary's per-wave-broadcast analysis assumes "
                         "exactly one target wave per workgroup");
    if (N % blockSize != 0)
      return std::string("N must be a multiple of blockSize (64) to keep "
                         "every workgroup full of lanes that actually "
                         "participate in the inline-asm v_readlane_b32");
    return std::nullopt;
  };
  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    // in[i] = i + 0.5, distinct for every lane so the per-source-wave
    // broadcast outputs are unambiguously distinguishable in the diff.
    for (int i = 0; i < N; ++i) f[i] = static_cast<float>(i) + 0.5f;
    return buf;
  };

  // CPU reference: per source-wave-of-W_s=32, broadcast lane (W_s - 1)'s
  // seed to every lane in that source wave.  This is what the wave32
  // source kernel means; the canary judges every mode against it.
  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int blockSize) {
    constexpr int kSrcWaveSize = 32;  // gfx1250 / wave32 source.
    const float *in = reinterpret_cast<const float *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    float *o = reinterpret_cast<float *>(out.data());
    // The launch grids ceil(N/blockSize) workgroups of `blockSize`
    // threads each.  Per-source-wave broadcast picks lane (W_s - 1)
    // of each source wave within a workgroup; cross workgroups the
    // result is simply a shifted copy of that pattern.
    for (int tid = 0; tid < N; ++tid) {
      int wg_id = tid / blockSize;
      int lane_in_wg = tid % blockSize;
      int src_wave_in_wg = lane_in_wg / kSrcWaveSize;
      int last_lane_of_src_wave_in_wg =
          src_wave_in_wg * kSrcWaveSize + (kSrcWaveSize - 1);
      int broadcast_tid = wg_id * blockSize + last_lane_of_src_wave_in_wg;
      // Bounds check: if the last lane of a source wave would fall
      // past N (ragged tail), the CPU reference follows the kernel's
      // early-return shape and leaves the out-of-range tail undefined
      // — but our validate() above pins N % blockSize == 0 so this
      // never fires on sanctioned shapes.
      o[tid] = (broadcast_tid < N) ? in[broadcast_tid] : 0.0f;
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "canary_readlane_last_lane"));

    float *dIn, *dOut;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    // Sentinel NaN so unwritten lanes surface crisply in the diff.
    uint32_t sentinel = 0x7FC00000u;
    std::vector<uint32_t> sentinelHost(N, sentinel);
    HIP_ASSERT(hipMemcpy(dOut, sentinelHost.data(), bytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const float *in;
      float *out;
    } args = {dIn, dOut};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    // Bit-exact compare — the values on both sides are IEEE-representable
    // seed-plus-0.5 integers, so any difference is a miscompile, not
    // rounding.
    int mismatches = 0;
    double maxAbs = 0.0;
    int firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    for (int i = 0; i < outElems; ++i) {
      if (g[i] != a[i]) {
        if (firstIdx < 0) {
          firstIdx = i;
          firstG = g[i];
          firstA = a[i];
        }
        double d = std::fabs(g[i] - a[i]);
        if (d > maxAbs) maxAbs = d;
        ++mismatches;
      }
    }
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: canary_ds_swizzle_swap1 — P6 ds_swizzle_b32 SWAP-1 regression guard
//
// Canary for the `ds_swizzle_b32 ... offset:swizzle(SWAP,1)` imm shape
// (BITMASK_PERM with and=0x1F, or=0, xor=1; offset=0x041F) — the exact
// instruction GPT-OSS's `sum_bitmatrix_rows` emits 8× per kernel
// invocation.  Complements the existing `Gfx1250Gpu.DsSwizzle` GPU
// test (which uses SWAP-2 / offset=0x081F to catch imm-extraction
// bugs that happen to round-trip SWAP-1) by pinning the SWAP-1 imm
// with the same end-to-end correctness contract.
//
// Gold semantics = SWAP-1 within each 32-lane group: for lane L,
//   out[L] = in[(L & ~31) | ((L & 31) ^ 1)]
// This formula holds across wave32 source and wave64 target because
// hardware `ds_swizzle_b32` preserves bit 5 of the lane id (the
// wave64-family-wide contract documented in
// `handle_ds.cpp::DS_SWIZZLE_B32`).
//
// Expected verdicts (all match) at landing:
//   native : match.  hipcc -> gfx942 -> same `ds_swizzle_b32
//            offset:swizzle(SWAP,1)` executed byte-for-byte; bit-5
//            preservation yields per-32-lane-half SWAP-1.
//   legacy : match.  Text transpiler copies the MCInst through.
//   salmon : match.  P6 lift routes through
//            `@llvm.amdgcn.ds.swizzle(src, 0x041F)` -> gfx942 backend
//            re-lowers to the identical instruction.
//
// Regression contract: a `match` -> `WRONG` drift on salmon means
// either (a) imm extraction in `decode.cpp::decodeDsSwizzleImm`
// regressed, (b) the MODREP classifier in
// `wave_projection.cpp::dsSwizzleSafeForModRep` started refusing
// SWAP-1, or (c) the intrinsic-to-backend path broke.  A drift to
// `EXIT=2` means the classifier moved from rewrite to refuse — only
// intentional if a future ISA breaks the bit-5 contract.  See
// `kernels/canary_ds_swizzle_swap1.hip` for the source shape and
// `hotswap/docs/gpt-oss-derisking.md` §7.2 / §9.2 item 4 for the
// P6 derisking write-up.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeCanaryDsSwizzleSwap1Recipe() {
  Recipe r;
  r.name = "canary_ds_swizzle_swap1";
  // Fixed blockSize=64 for the same reason as canary_readlane_last_lane:
  // exactly one target wave per workgroup keeps the per-32-lane-half
  // SWAP-1 pattern easy to reason about in both the CPU reference and
  // in post-mortem diff inspection.  N values above 64 cover multi-
  // workgroup launches so a blockIdx-dependent miscompile (e.g. a
  // future regression that drops blockIdx.x from the kernarg plumbing)
  // would surface as a shape-dependent WRONG count.
  r.defaultNs     = {64, 128, 256, 1024};
  r.defaultBlocks = {64};
  r.validate = [](int N, int blockSize) -> std::optional<std::string> {
    if (blockSize != 64)
      return std::string("blockSize must be 64 (the target wave64 width): "
                         "ds_swizzle SWAP-1 operates on a single 32-lane "
                         "group within a 64-lane wave, and blockSize=64 "
                         "keeps the CPU reference's group arithmetic "
                         "1:1 with one target wave per workgroup");
    if (N % blockSize != 0)
      return std::string("N must be a multiple of blockSize (64) to keep "
                         "every workgroup full; ragged tails would make "
                         "the last group's SWAP-1 read from an out-of-"
                         "range lane whose behaviour is ISA-dependent");
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

  // Gold: per-32-lane-group SWAP-1 (lane L reads from lane L XOR 1
  // within the group).  Block-scoped because the kernel is launched
  // with `grid = ceil(N / blockSize)`, so the swap is always within
  // a workgroup (one 32-lane group per wave, two waves per block at
  // blockSize=64, and each wave swaps within its own 32 lanes).
  r.cpuReference = [](const std::vector<uint8_t> &input, int N,
                      int blockSize) {
    const float *in = reinterpret_cast<const float *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    float *o = reinterpret_cast<float *>(out.data());
    constexpr int kGroupSize = 32;  // ds_swizzle bit-5-preservation boundary.
    for (int tid = 0; tid < N; ++tid) {
      int wg_id = tid / blockSize;
      int lane_in_wg = tid % blockSize;
      int group_base = lane_in_wg & ~(kGroupSize - 1);
      int lane_in_group = lane_in_wg & (kGroupSize - 1);
      int swap_partner = lane_in_group ^ 1;
      int src_tid = wg_id * blockSize + group_base + swap_partner;
      // validate() pins N % blockSize == 0 so src_tid is always in
      // range; the guard is defensive against a future validate()
      // loosening that forgets this invariant.
      o[tid] = (src_tid < N) ? in[src_tid] : 0.0f;
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "canary_ds_swizzle_swap1"));
    float *dIn, *dOut;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    uint32_t sentinel = 0x7FC00000u;
    std::vector<uint32_t> sentinelHost(N, sentinel);
    HIP_ASSERT(hipMemcpy(dOut, sentinelHost.data(), bytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const float *in;
      float *out;
    } args = {dIn, dOut};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    int mismatches = 0;
    double maxAbs = 0.0;
    int firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    for (int i = 0; i < outElems; ++i) {
      if (g[i] != a[i]) {
        if (firstIdx < 0) {
          firstIdx = i;
          firstG = g[i];
          firstA = a[i];
        }
        double d = std::fabs(g[i] - a[i]);
        if (d > maxAbs) maxAbs = d;
        ++mismatches;
      }
    }
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: canary_ds_swizzle_quad_perm — P6 QUAD_PERM envelope regression guard
//
// Canary for `ds_swizzle_b32 offset:swizzle(QUAD_PERM, 3, 2, 1, 0)`
// (imm = 0x801B), the in-quad reversal within every 4-lane group.
// Complements `canary_ds_swizzle_swap1` by exercising a DIFFERENT
// top-level envelope of the P6 handler — QUAD_PERM (top-byte 0x80)
// vs BITMASK_PERM (top-bit 0) — so imm-decoder / classifier
// regressions that special-case one envelope without the other
// would fail here but pass the SWAP-1 canary.
//
// Gold semantics = per-quad reversal: for every lane L,
//   out[L] = in[(L & ~3) | Q[L & 3]]
// where Q = {3, 2, 1, 0} — i.e. lane 0 reads lane 3, lane 1 reads
// lane 2, lane 2 reads lane 1, lane 3 reads lane 0, repeating
// every 4 lanes.  bit-5-preservation means this formula is
// identical on wave32 source and wave64 target, so all three
// modes (native / legacy / salmon) MUST match.
//
// See `kernels/canary_ds_swizzle_quad_perm.hip` for the imm-
// encoding rationale (and the gotcha that tripped the initial
// version of this recipe — the assembler prints quad selectors
// in position order, which is NOT the same as the encoding
// order).  See `handle_ds.cpp::DS_SWIZZLE_B32` for the four-
// envelope taxonomy and the wave-size-obliviousness argument.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeCanaryDsSwizzleQuadPermRecipe() {
  Recipe r;
  r.name = "canary_ds_swizzle_quad_perm";
  // Fixed blockSize=64 for the same reason as the sibling SWAP-1
  // canary: one target wave64 per workgroup makes the per-quad
  // reversal unambiguous in post-mortem diff inspection.  N values
  // above 64 cover multi-workgroup launches.
  r.defaultNs     = {64, 128, 256, 1024};
  r.defaultBlocks = {64};
  r.validate = [](int N, int blockSize) -> std::optional<std::string> {
    if (blockSize != 64)
      return std::string("blockSize must be 64 (the target wave64 width): "
                         "per-quad reversal is analysed within a single "
                         "target wave");
    if (N % 4 != 0)
      return std::string("N must be a multiple of 4 (the QUAD size): "
                         "ragged tails fall off the end of a quad group "
                         "and the reversal reads an out-of-bounds lane "
                         "whose value is ISA-dependent");
    if (N % blockSize != 0)
      return std::string("N must be a multiple of blockSize (64): "
                         "half-workgroup tails complicate the per-quad "
                         "analysis without adding coverage");
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

  r.cpuReference = [](const std::vector<uint8_t> &input, int N,
                      int blockSize) {
    const float *in = reinterpret_cast<const float *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    float *o = reinterpret_cast<float *>(out.data());
    // Quad selector Q = (3, 2, 1, 0): position k reads lane
    // `(quad_base | (3 - k))` — the reversal formula.
    constexpr int kQuad = 4;
    for (int tid = 0; tid < N; ++tid) {
      int wg_id = tid / blockSize;
      int lane_in_wg = tid % blockSize;
      int quad_base = lane_in_wg & ~(kQuad - 1);
      int lane_in_quad = lane_in_wg & (kQuad - 1);
      int src_lane_in_wg = quad_base | (3 - lane_in_quad);
      int src_tid = wg_id * blockSize + src_lane_in_wg;
      o[tid] = (src_tid < N) ? in[src_tid] : 0.0f;
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "canary_ds_swizzle_quad_perm"));
    float *dIn, *dOut;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    uint32_t sentinel = 0x7FC00000u;
    std::vector<uint32_t> sentinelHost(N, sentinel);
    HIP_ASSERT(hipMemcpy(dOut, sentinelHost.data(), bytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const float *in;
      float *out;
    } args = {dIn, dOut};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    int mismatches = 0;
    double maxAbs = 0.0;
    int firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    for (int i = 0; i < outElems; ++i) {
      if (g[i] != a[i]) {
        if (firstIdx < 0) {
          firstIdx = i;
          firstG = g[i];
          firstA = a[i];
        }
        double d = std::fabs(g[i] - a[i]);
        if (d > maxAbs) maxAbs = d;
        ++mismatches;
      }
    }
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: canary_cvt_scale_pk8_bf16_fp4 — gfx1250-only MXFP4 dequant primitive
//
// Canary for the Salmon cross-target (gfx1250 -> gfx942) lowering of the
// gfx1250-only VOP3 scaled packed-8 FP4 -> BF16 convert
// (`v_cvt_scale_pk8_bf16_fp4`, VOP3Instructions.td:1873; intrinsic
// `int_amdgcn_cvt_scale_pk8_bf16_fp4` declared inside the
// `isGFX125xOnly` block of IntrinsicsAMDGPU.td:686).
//
// Corpus scope
// ============
//
// Observed in 64 instances each of
// `scope_discovery/kernels/_matmul_ogs_06d912ce88af.hsaco` and
// `_matmul_ogs_0af655e6ea2b.hsaco` (the Triton GPT-OSS MoE matmul
// kernels `_matmul_ogs_NNT_bf16xbf16xmxfp4_*`).  Both corpus blobs
// use only `scale_sel=0` across all 128 instances combined.  The
// cross-target handler's declared support set tracks this: scale_sel
// != 0 refuses loudly.  See `hotswap/docs/matrix-translation.md §7.N`
// (the dequant-primitive subsection added alongside this canary) for
// the declared support set and its corpus provenance.
//
// Landing this canary + the cross-target handler is standalone
// cross-target coverage for the MXFP4 dequant primitive; end-to-end
// lift of the two matmul_ogs kernels still blocks on pending Template
// A (BF16 16x16x32 WMMA -> MFMA, `matrix-translation.md §T2`) and on
// TDM (`global_load_async_to_lds_*`).
//
// See `kernels/canary_cvt_scale_pk8_bf16_fp4.hip` for the source
// shape (HIP + clang builtin; Triton cannot force this primitive in
// isolation from the dense WMMA — see the file-level docstring) and
// the portable-gfx942 dequant that `native` exercises.
// ─────────────────────────────────────────────────────────────────────────────

// Portable FP4 E2M1 -> BF16 bit-pattern derivation.  Kept here (in
// the harness) rather than shared via
// `transpiler/mxfp4_dequant.hpp` because `compare_correctness`
// builds without linking LLVM and the `.hpp`'s companion `.cpp`
// depends on `llvm::report_fatal_error`.  The primary reference
// implementation is `mxfp4::mxfp4BitAlgebraBf16Bits`; the gtest
// `Mxfp4Dequant.*` suite sweeps all 4096 (nibble, scale_byte) inputs
// and asserts that reference matches the LUT-via-double reference
// bit-for-bit.  This function mirrors the bit-algebra reference
// step-for-step; a drift surfaces immediately in the canary's native
// vs. CPU-reference comparison.  If the boundary ever goes away
// (e.g. the harness starts linking LLVM), switch to including the
// header and delete the host mirror.
uint16_t host_fp4_nibble_to_bf16_bits(uint32_t nibble) {
  uint32_t sign_bit  = (nibble >> 3) & 0x1u;
  uint32_t exp_fp4   = (nibble >> 1) & 0x3u;
  uint32_t mant_fp4  =  nibble       & 0x1u;
  uint32_t bf16_exp_norm  = exp_fp4 + 126u;
  uint32_t bf16_mant_norm = mant_fp4 ? 0x40u : 0x00u;
  uint32_t bf16_exp_sub   = mant_fp4 ? 126u : 0u;
  uint32_t bf16_mant_sub  = 0u;
  uint32_t is_sub = (exp_fp4 == 0u) ? 1u : 0u;
  uint32_t bf16_exp  = is_sub ? bf16_exp_sub  : bf16_exp_norm;
  uint32_t bf16_mant = is_sub ? bf16_mant_sub : bf16_mant_norm;
  return static_cast<uint16_t>((sign_bit << 15) | (bf16_exp << 7) | bf16_mant);
}

// Apply an E8M0 scale byte to a BF16 bit pattern.  Host mirror of the
// device `apply_e8m0_scale_to_bf16` in the .hip file.  See that
// function's comment block for the corner-case table.
uint16_t host_apply_e8m0_scale_to_bf16(uint16_t fp4_bf16_bits, uint32_t scale_byte) {
  if (scale_byte == 0xFFu) return static_cast<uint16_t>(0x7FC0);
  uint32_t magnitude = fp4_bf16_bits & 0x7FFFu;
  if (magnitude == 0u) return fp4_bf16_bits;
  uint32_t sign_bits = fp4_bf16_bits & 0x8000u;
  uint32_t fp4_exp  = (fp4_bf16_bits >> 7) & 0xFFu;
  uint32_t fp4_mant =  fp4_bf16_bits       & 0x7Fu;
  int32_t new_exp_i = static_cast<int32_t>(fp4_exp)
                    + static_cast<int32_t>(scale_byte) - 127;
  if (new_exp_i >= 0xFF)
    return static_cast<uint16_t>(sign_bits | 0x7F80u);
  if (new_exp_i >= 1)
    return static_cast<uint16_t>(sign_bits
        | ((static_cast<uint32_t>(new_exp_i) & 0xFFu) << 7)
        | fp4_mant);
  uint32_t implicit_1_mant = 0x80u | fp4_mant;
  int32_t shift_amt = 1 - new_exp_i;
  uint32_t sub_mant = (shift_amt >= 8) ? 0u : (implicit_1_mant >> shift_amt);
  return static_cast<uint16_t>(sign_bits | sub_mant);
}

// Scale bytes swept by the canary's input generator.  Mix of:
//   * 0x00       — 2^-127, the E8M0 subnormal floor; exercises the
//                  underflow -> BF16 subnormal / zero branch.
//   * 0x01       — 2^-126, the smallest E8M0 normal scale.
//   * 0x40       — 2^-63.
//   * 0x7E,0x7F,0x80,0x81 — around identity (2^0 is 0x7F); the hot
//                           path for matmul MXFP4 where most scales
//                           are near 1.
//   * 0xC0       — 2^65.
//   * 0xFC,0xFD,0xFE — near the overflow boundary; exercises the
//                      saturate-to-Inf branch.
//   * 0xFF       — E8M0 NaN; exercises NaN propagation.
static constexpr uint8_t kCanaryScaleBytes[] = {
    0x7Fu, 0x80u, 0x7Eu, 0x81u,
    0x00u, 0x01u, 0x40u, 0xC0u,
    0xFCu, 0xFDu, 0xFEu, 0xFFu,
};
static constexpr int kCanaryScaleBytesCount =
    sizeof(kCanaryScaleBytes) / sizeof(kCanaryScaleBytes[0]);

// Deterministic packed-FP4 generator.  Derived from tid via a
// Weyl-sequence-style hash so that across the full N sweep, every
// 4-bit nibble value appears in every one of the 8 lane positions.
// The constants 0x9E3779B1 (golden-ratio-ish) and 0x7F4A7C15 are
// standard mixing primes from the PCG / splitmix family; pairing
// them against tid gives uniform coverage of the 32-bit packed-FP4
// space with no bias toward any nibble position.
//
// This generator also feeds the on-device kernel's input buffer, so
// the CPU reference and the device-side portable dequant operate on
// the same bytes byte-for-byte.
uint32_t host_packed_fp4_for_tid(int tid) {
  uint32_t t = static_cast<uint32_t>(tid);
  return (t * 0x9E3779B1u) ^ ((t >> 3) * 0x7F4A7C15u);
}

// Recipe
Recipe makeCanaryCvtScalePk8Bf16Fp4Recipe() {
  Recipe r;
  r.name = "canary_cvt_scale_pk8_bf16_fp4";
  // No cross-lane dependency in the primitive itself; blockSize only
  // affects launch shape.  Sweep a mix of single-wave and multi-wave
  // workgroups so a per-wave miscompile in the cross-target handler
  // surfaces as a per-lane pattern mismatch.  Shapes chosen at the
  // same scale as the rest of the canary family.
  r.defaultNs     = {128, 1024, 4096};
  r.defaultBlocks = {64, 128};
  r.outputElemBytes = sizeof(uint16_t);
  // Each thread emits 8 BF16 words.  Output element count is 8*N.
  r.outputElems = [](int N, int) { return 8 * N; };

  r.makeInput = [](int N) {
    // Input buffer layout: N pairs of {packed : u32, scale : u32}.
    // Matches `struct InputPair` in the .hip kernel.  Packed as raw
    // bytes so the harness and the kernel agree on offsets without
    // depending on C++ struct padding.
    std::vector<uint8_t> buf(N * 2 * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    for (int tid = 0; tid < N; ++tid) {
      u[2 * tid + 0] = host_packed_fp4_for_tid(tid);
      // scale low byte cycles through the representative table;
      // high bytes are zero (scale_sel=0 only consumes the low byte
      // today).  A future handler extension that honours scale_sel
      // != 0 would change this generator to populate the other bytes
      // and expand the output-side verification.
      uint8_t sb = kCanaryScaleBytes[tid % kCanaryScaleBytesCount];
      u[2 * tid + 1] = static_cast<uint32_t>(sb);
    }
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int /*block*/) {
    const uint32_t *u = reinterpret_cast<const uint32_t *>(input.data());
    std::vector<uint8_t> out(8 * N * sizeof(uint16_t));
    uint16_t *o = reinterpret_cast<uint16_t *>(out.data());
    for (int tid = 0; tid < N; ++tid) {
      uint32_t packed     = u[2 * tid + 0];
      uint32_t scale_word = u[2 * tid + 1];
      uint32_t scale_byte = scale_word & 0xFFu;   // scale_sel == 0
      for (int i = 0; i < 8; ++i) {
        uint32_t nibble = (packed >> (i * 4)) & 0xFu;
        uint16_t fp4_bf = host_fp4_nibble_to_bf16_bits(nibble);
        o[tid * 8 + i]  = host_apply_e8m0_scale_to_bf16(fp4_bf, scale_byte);
      }
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod,
                                    "canary_cvt_scale_pk8_bf16_fp4"));

    uint8_t *dIn;
    uint16_t *dOut;
    size_t inBytes  = input.size();
    size_t outBytes = 8 * N * sizeof(uint16_t);
    HIP_ASSERT(hipMalloc(&dIn,  inBytes));
    HIP_ASSERT(hipMalloc(&dOut, outBytes));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), inBytes, hipMemcpyHostToDevice));
    // Sentinel-fill the output so any lane the kernel fails to write
    // surfaces crisply as a bit-level mismatch against the CPU
    // reference rather than as a zero that might silently match.
    std::vector<uint16_t> sentinel(8 * N, 0xDEAD);
    HIP_ASSERT(hipMemcpy(dOut, sentinel.data(), outBytes,
                         hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const uint8_t *in;
      uint16_t *out;
    } args = {dIn, dOut};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};
    int grd = (N + blockSize - 1) / blockSize;
    HIP_ASSERT(hipModuleLaunchKernel(fn, grd, 1, 1, blockSize, 1, 1, 0,
                                     nullptr, nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());
    std::vector<uint8_t> out(outBytes);
    HIP_ASSERT(hipMemcpy(out.data(), dOut, outBytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dIn));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*block*/, int outElems) {
    const uint16_t *g = reinterpret_cast<const uint16_t *>(gold.data());
    const uint16_t *a = reinterpret_cast<const uint16_t *>(actual.data());
    int mismatches = 0;
    double maxAbs = 0.0;
    int firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    // Bit-exact BF16 compare.  Both sides compute the exact same
    // arithmetic on the exact same inputs; any bit-level difference
    // is a real divergence (handler bug, device dequant bug, or
    // input-pack mismatch).  We surface the differing BF16 bit
    // patterns as doubles so the existing Failures formatter prints
    // them usefully; the `(double)` cast is safe because both
    // operands are 16-bit unsigned integers.
    for (int i = 0; i < outElems; ++i) {
      if (g[i] != a[i]) {
        if (firstIdx < 0) {
          firstIdx = i;
          firstG   = static_cast<double>(g[i]);
          firstA   = static_cast<double>(a[i]);
        }
        double d = std::fabs(static_cast<double>(g[i])
                           - static_cast<double>(a[i]));
        if (d > maxAbs) maxAbs = d;
        ++mismatches;
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
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
// Recipe: v_bfi_b32 — V_BFI_B32 handler
// Bit-field insert: out = (mask & one_src) | (~mask & zero_src).
// Forced with inline asm because hipcc lowers the ternary pattern to
// and/andn2/or rather than emitting the fused opcode.  Pins the real
// libdevice-asin coverage gap (see kernels/v_bfi_b32.hip comment) to a
// one-instruction reproducer.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeVBfiB32Recipe() {
  Recipe r;
  r.name = "v_bfi_b32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    // Three u32 arrays packed back-to-back: [mask | one_src | zero_src].
    // Random coverage of the ternary input space; a fixed seed per N
    // makes the sweep reproducible.
    std::vector<uint8_t> buf(3 * N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    std::mt19937 rng(0xBF1u + N);
    for (int i = 0; i < 3 * N; ++i) u[i] = rng();
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const uint32_t *m = reinterpret_cast<const uint32_t *>(input.data());
    const uint32_t *o = m + N;
    const uint32_t *z = o + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *c = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i < N; ++i) c[i] = (m[i] & o[i]) | (~m[i] & z[i]);
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "v_bfi_b32"));
    uint32_t *dM, *dO, *dZ, *dC;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dM, bytes));
    HIP_ASSERT(hipMalloc(&dO, bytes));
    HIP_ASSERT(hipMalloc(&dZ, bytes));
    HIP_ASSERT(hipMalloc(&dC, bytes));
    HIP_ASSERT(hipMemset(dC, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dM, input.data() + 0 * bytes, bytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dO, input.data() + 1 * bytes, bytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dZ, input.data() + 2 * bytes, bytes,
                         hipMemcpyHostToDevice));
    struct alignas(8) Args {
      const uint32_t *m; const uint32_t *o; const uint32_t *z;
      uint32_t *c; int n;
    } args = {dM, dO, dZ, dC, N};
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
    HIP_ASSERT(hipFree(dM)); HIP_ASSERT(hipFree(dO));
    HIP_ASSERT(hipFree(dZ)); HIP_ASSERT(hipFree(dC));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: v_cmp_cndmask_sgpr — V_CMP_GE_F32_e64 -> SGPR -> V_CNDMASK_B32_e64
//
// Single-instruction runtime probe for the per-lane-predication
// idiom every libdevice math branch (asin / acos / atan / log /
// exp) and every Triton e64-compare-across-BB epilogue lowers to.
// Forced via inline asm so hipcc cannot substitute an e32 -> VCC
// form that would route through a different handler path
// (`loadVCC` — which was never broken).
//
// Pre-fix, the SGPR arm of `V_CNDMASK_B32` in
// handle_valu_vop3p.cpp folded the entire source SGPR into one
// wave-uniform `i1` via `icmp ne <sgpr>, 0`, making every lane of
// a wave pick the same side of the select. Reducer shape here
// (random floats in [-1, 1] -> +1.0f if |x|>=0.5 else -1.0f)
// produced ~50% wrong elements on a 128-thread block. Post-fix
// the handler routes through the WaveProjection's
// `extractLaneBitFromWaveMask`, mirroring the consumer symmetry of
// VCC's `readVCCAsWaveMask`, and the same-wave / modulo-
// replication paths are bit-exact against the CPU reference. The
// wave-native cross-widening direction has a separate residual
// (V_CMP -> SGPR narrow-write-and-replicate, documented in
// handle_valu_vcmp.cpp) that surfaces as ~25% residual errors; the
// obstruction-classifier follow-up is meant to refuse that shape.
// See `lit_tests/v_cmp_cndmask_sgpr/` for the lifted-IR shape
// fixture companion.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeVCmpCndmaskSgprRecipe() {
  Recipe r;
  r.name = "v_cmp_cndmask_sgpr";
  // Full block-size sweep. The shadow map in `RaiseContext::
  // lastSgprWaveMaskI1` (see hotswap/docs/sgpr-wave-mask-translation.md
  // section 3.1) makes the V_CMP -> V_CNDMASK_B32 per-lane dataflow
  // correct under cross-widening for every block size, because the
  // consumer reads the full-fidelity per-lane `i1` directly from the
  // producer without going through the narrow-ballot round-trip.
  // A historical narrower sweep (`defaultBlocks = {16, 32}`, capped at
  // single-wave-32-slice) was used as a probe before the shadow landed;
  // the full sweep is restored now because every configuration below
  // matches bit-exactly under the fix, and any regression on the V_CMP
  // writer's shadow-record or the V_CNDMASK consumer's shadow-lookup
  // fails this recipe loudly on the first WRONG row.
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {16, 32, 64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    // Floats uniform in [-1, 1] reinterpret-cast to u32 bytes; fixed
    // seed per N keeps the sweep reproducible. The range is picked so
    // that |x|>=0.5 splits roughly 50/50 across lanes — the pre-fix
    // wave-uniform-collapse bug was most visible (and this probe is
    // most sensitive) when the compare result is genuinely data-
    // dependent per lane.
    std::vector<uint8_t> buf(N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(0xC9Du + N);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) f[i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *x = reinterpret_cast<const float *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    auto *o = reinterpret_cast<float *>(out.data());
    // Match the kernel's rule exactly: `|x| >= 0.5 ? +1.0f : -1.0f`.
    // Both sides of the select are exactly representable in fp32 so
    // the bit patterns 0x3f800000 / 0xbf800000 are stable and can be
    // compared with compareU32Exact (no rounding slack needed).
    for (int i = 0; i < N; ++i)
      o[i] = (std::fabs(x[i]) >= 0.5f) ? 1.0f : -1.0f;
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "v_cmp_cndmask_sgpr"));
    float *dX, *dOut;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dX, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    // Distinctive poison pattern so an untouched lane reads back as
    // neither +1.0f nor -1.0f and surfaces as a bit-exact mismatch
    // against the CPU reference.
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dX, input.data(), bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args {
      const float *x; float *out; int n;
    } args = {dX, dOut, N};
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
    HIP_ASSERT(hipFree(dX));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
    // Bit-exact: the two selectable values (+1.0f / -1.0f) are
    // exactly representable and independent of wave-size reordering,
    // so any elementwise difference is a real finding.
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: v_cmpx_gt_i32 — V_CMPX_GT_I32 handler (compare-and-write-exec)
// Each thread: out[tid] = (a[tid] > b[tid]) ? 1 : 0 via a predicated
// store.  The predicate source is data-dependent, not lane-position-
// dependent, so the wave-size classifier passes the kernel and salmon
// has to actually lower the opcode.  Reduces the `lane_swap` "unsupported
// instruction: v_cmpx_gt_i32" failure to one opcode.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeVCmpxGtI32Recipe() {
  Recipe r;
  r.name = "v_cmpx_gt_i32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    // Two i32 arrays packed back-to-back: [a | b].  Inputs in a small
    // signed range force an interesting mix of > / <= outcomes without
    // biasing one way.
    std::vector<uint8_t> buf(2 * N * sizeof(int32_t));
    auto *s = reinterpret_cast<int32_t *>(buf.data());
    std::mt19937 rng(0xC1Fu + N);
    std::uniform_int_distribution<int32_t> dist(-1024, 1024);
    for (int i = 0; i < 2 * N; ++i) s[i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const int32_t *a = reinterpret_cast<const int32_t *>(input.data());
    const int32_t *b = a + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t));
    auto *o = reinterpret_cast<uint32_t *>(out.data());
    for (int i = 0; i < N; ++i) o[i] = (a[i] > b[i]) ? 1u : 0u;
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "v_cmpx_gt_i32"));
    int32_t *dA, *dB;
    uint32_t *dOut;
    size_t inBytes = N * sizeof(int32_t);
    size_t outBytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dA, inBytes));
    HIP_ASSERT(hipMalloc(&dB, inBytes));
    HIP_ASSERT(hipMalloc(&dOut, outBytes));
    // hipMemset to 0 so untouched lanes read back as the "not greater"
    // sentinel the CPU reference expects.
    HIP_ASSERT(hipMemset(dOut, 0x00, outBytes));
    HIP_ASSERT(hipMemcpy(dA, input.data(),           inBytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dB, input.data() + inBytes, inBytes,
                         hipMemcpyHostToDevice));
    struct alignas(8) Args {
      const int32_t *a; const int32_t *b; uint32_t *out; int n;
    } args = {dA, dB, dOut, N};
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
    HIP_ASSERT(hipFree(dA)); HIP_ASSERT(hipFree(dB));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: s_and_saveexec_b32 — scalar exec save/restore handler
// Kernel uses inline asm to force the fused `s_and_saveexec_b32`
// opcode on gfx1250 (current hipcc prefers the decomposed
// `s_mov_b32 sX,exec_lo` + `v_cmpx_*` + `s_or_b32 exec_lo,…,sX`
// form for simple branches — covered by the v_cmpx_gt_i32 probe).
//
// Inputs: a per-block u32 `mask_in` (packed as the first N-values-
// long slice) and a per-thread u32 `pred` (second N slice).  `mask_in`
// is placed in an SGPR via `readfirstlane`; the opcode narrows
// exec_lo to `exec_lo & mask`, the body stores `body(pred[tid])`
// from each surviving lane, and exec_lo is restored.  A masked-off
// lane keeps its init-zero sentinel.  The CPU reference mirrors the
// wave-relative-lane semantics exactly (see kernel file comment).
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeSAndSaveexecB32Recipe() {
  Recipe r;
  r.name = "s_and_saveexec_b32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(uint32_t);
  r.outputElems = [](int N, int) { return N; };

  auto body = [](uint32_t p) -> uint32_t {
    uint32_t acc = p;
    acc ^= 0xDEADBEEFu;
    acc = acc * 2654435761u;
    acc ^= acc >> 13;
    return acc;
  };

  r.makeInput = [](int N) {
    // Layout: [mask_in (N u32, per-block value repeated) | pred (N u32)].
    // We store one mask per block to keep makeInput block-shape-agnostic:
    // the kernel reads mask_in[blockIdx.x], so only the first ceil(N/B)
    // entries actually matter.  We fill all N to keep the buffer a
    // round 2N u32.  Predicates are arbitrary non-zero u32s.
    std::vector<uint8_t> buf(2 * N * sizeof(uint32_t));
    auto *u = reinterpret_cast<uint32_t *>(buf.data());
    std::mt19937 rng(0x5AEu + N);
    for (int i = 0; i < N; ++i) u[i]       = rng();  // mask_in
    for (int i = 0; i < N; ++i) u[N + i]   = rng();  // pred
    return buf;
  };

  r.cpuReference = [body](const std::vector<uint8_t> &input, int N,
                           int blockSize) {
    const uint32_t *m = reinterpret_cast<const uint32_t *>(input.data());
    const uint32_t *p = m + N;
    std::vector<uint8_t> out(N * sizeof(uint32_t), 0);
    auto *o = reinterpret_cast<uint32_t *>(out.data());
    for (int tid = 0; tid < N; ++tid) {
      int bi = tid / blockSize;
      int laneInBlock = tid - bi * blockSize;
      // On gfx1250 wave32 the exec mask is 32 bits per wave — each
      // thread's wave-relative lane (0..31) checks the corresponding
      // bit of the block-uniform mask.  Inside a 64- or 256-wide block
      // the pattern repeats for every wave.  Our reference uses
      // laneInBlock & 31, which matches the gfx942 fallback in the
      // kernel (wave64 lanes 0..63 all fold to the 32-bit mask).
      uint32_t mask = m[bi];
      if ((mask >> (laneInBlock & 31)) & 1u) {
        o[tid] = body(p[tid]);
      }
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "s_and_saveexec_b32"));
    uint32_t *dMask, *dPred, *dOut;
    size_t bytes = N * sizeof(uint32_t);
    HIP_ASSERT(hipMalloc(&dMask, bytes));
    HIP_ASSERT(hipMalloc(&dPred, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    // Zero-init — masked-off lanes must read back as the sentinel the
    // CPU reference expects (0 for "store skipped").
    HIP_ASSERT(hipMemset(dOut, 0x00, bytes));
    HIP_ASSERT(hipMemcpy(dMask, input.data(),         bytes,
                         hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dPred, input.data() + bytes, bytes,
                         hipMemcpyHostToDevice));
    struct alignas(8) Args {
      const uint32_t *mask; const uint32_t *pred; uint32_t *out; int n;
    } args = {dMask, dPred, dOut, N};
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
    HIP_ASSERT(hipFree(dMask)); HIP_ASSERT(hipFree(dPred));
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int n) {
    return compareU32Exact(gold, actual, n);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: c4_lane_dep_cmpx — runtime evidence for Class 4 (lane-
// position-dependent EXEC writes) per hotswap/docs/wave-size-
// translation.md §6.
//
// Companion to lit_tests/c4_lane_dep_cmpx (offline classifier pin).
// The harness runs the same kernel under native / legacy / salmon:
//
//   - native  : gfx942 direct. CPU reference matches it exactly.
//   - legacy  : gfx1250 CO via byte-level translator. Upper 32 target
//               lanes are dead under half-wave masking, so positions
//               32..63 of each wave stay at sentinel. DIVERGES from
//               native at those positions.
//   - salmon  : gfx1250 CO via IR raiser. Under the new classifier
//               gate (wave_size_obstruction.{hpp,cpp}) the raise
//               refuses with `cross-wave-Class-4-*`; the harness
//               observes salmon as "EXIT=<nonzero>" rather than a
//               silent wrong output.
//
// Shape. One kernel launch per (N, block). Kernel writes one int per
// thread; observable output is N ints. CPU reference computes the
// expected native gfx942 wave64 behaviour: out[tid] = 1 iff the
// wave-relative lane id (clamped by mbcnt_lo's 32-bit mask on
// wave64) is >= 16.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeC4LaneDepCmpxRecipe() {
  Recipe r;
  r.name = "c4_lane_dep_cmpx";
  // Pick block sizes > warpSize=64 so both target waves of a block
  // exercise the EXEC-write divergence; N values chosen to cover
  // single-block and multi-block sweeps.
  r.defaultNs     = {64, 128, 256, 1024};
  r.defaultBlocks = {64, 128};
  r.outputElemBytes = sizeof(int);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    // Kernel takes no input buffer (just the output); deterministic
    // input is the empty byte sequence sized to N for harness
    // bookkeeping.
    return std::vector<uint8_t>(N * sizeof(int), 0);
  };

  r.cpuReference = [](const std::vector<uint8_t> & /*input*/, int N,
                       int blockSize) {
    // Reference matches native gfx942 (wave64). On wave64,
    // v_mbcnt_lo(-1, 0) gives `min(lane_in_wave64, 32)` because
    // exec_lo only counts the lower 32 lanes. v_cmpx_ge_u32 v10, 16
    // then enables wave-relative lanes 16..63. Lane 0..15 stay at
    // sentinel (0 — we hipMemset to 0 before dispatch).
    constexpr int kWaveSize = 64;
    std::vector<uint8_t> out(N * sizeof(int));
    int *o = reinterpret_cast<int *>(out.data());
    for (int tid = 0; tid < N; ++tid) {
      int waveRelative = tid % kWaveSize;
      int mbcnt = (waveRelative < 32) ? waveRelative : 32;
      o[tid] = (mbcnt >= 16) ? 1 : 0;
      (void)blockSize; // compute is wave-relative, not block-relative
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> & /*input*/,
                   int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "c4_lane_dep_cmpx"));
    int *dOut;
    size_t bytes = N * sizeof(int);
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    // hipMemset to 0x00 so untouched lanes read back 0 — matches
    // the CPU reference's "sentinel = 0" convention.
    HIP_ASSERT(hipMemset(dOut, 0x00, bytes));
    struct alignas(8) Args { int *o; } args = {dOut};
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
    HIP_ASSERT(hipFree(dOut));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
    const int *g = reinterpret_cast<const int *>(gold.data());
    const int *a = reinterpret_cast<const int *>(actual.data());
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    for (int i = 0; i < outElems; ++i) {
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

  r.validate = [](int /*N*/, int blockSize) -> std::optional<std::string> {
    if (blockSize < 64)
      return std::string(
          "block<64: fewer than one full wave64 makes the probe's "
          "wave-relative lane comparison degenerate on native gfx942");
    return std::nullopt;
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: wmma_f32_16x16x4_gemm — numerical probe for the
// gfx1250 WMMA -> gfx942 MFMA cross-target lowering.
//
// Each block computes one independent 16x16x4 f32 GEMM tile:
//
//   D[i][j] = sum_{k=0..3} A[i][k] * B[k][j] + C[i][j]
//
// `N` is the tile count. Block size is pinned at 32 (one Wave32 on
// gfx1250; on gfx942 wave64 the kernel launches a half-masked wave
// and uses a scalar shared-memory fallback — see the kernel's file
// comment for the gather layout).
//
// The CPU reference is straight scalar matmul with double-precision
// accumulation; at K=4 with inputs in [-1, 1] the worst case per
// output element is ~4 f32 rounding errors on the kernel side, so the
// tolerance is set to a loose f32 ulp budget (1e-5 absolute).
//
// This is the direct numerical counterpart to the
// lit_tests/wmma_f32_16x16x4_f32 IR-shape fixture: together they
// cover "Salmon raises the right IR" (lit) and "the raised IR
// computes the right answer on real hardware" (this recipe).
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeWmmaF32_16x16x4_GemmRecipe() {
  Recipe r;
  r.name = "wmma_f32_16x16x4_gemm";
  r.goldSource = GoldSource::CpuReference;
  // One tile is the smallest meaningful N (isolates the single WMMA);
  // larger Ns stress the grid-level launch path and catch any
  // per-tile state leak between Salmon-translated waves.
  r.defaultNs     = {1, 4, 16, 64};
  r.defaultBlocks = {32};

  r.validate = [](int /*N*/, int blockSize) -> std::optional<std::string> {
    if (blockSize != 32)
      return std::string(
          "blockSize must be 32: WMMA is a wave-32 collective and the "
          "kernel maps exactly one 16x16x4 tile to one wavefront; "
          "other block sizes violate the per-tile layout assumption");
    return std::nullopt;
  };

  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int) { return N * 16 * 16; };

  // Input buffer layout (packed): [A | B | C]
  //   A: N * 16 * 4 floats
  //   B: N *  4 *16 floats
  //   C: N * 16 *16 floats
  // The same PRNG seed family as other recipes (a fixed offset added
  // to N) makes every per-shape input deterministic across runs.
  r.makeInput = [](int N) {
    size_t szA = size_t(N) * 16 * 4;
    size_t szB = size_t(N) * 4 * 16;
    size_t szC = size_t(N) * 16 * 16;
    std::vector<uint8_t> buf((szA + szB + szC) * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(24680u + static_cast<unsigned>(N));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0, n = szA + szB + szC; i < n; ++i) f[i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *a = reinterpret_cast<const float *>(input.data());
    const float *b = a + size_t(N) * 16 * 4;
    const float *c = b + size_t(N) * 4 * 16;
    size_t outBytes = size_t(N) * 16 * 16 * sizeof(float);
    std::vector<uint8_t> out(outBytes);
    float *d = reinterpret_cast<float *>(out.data());
    for (int t = 0; t < N; ++t) {
      const float *tA = a + size_t(t) * 16 * 4;
      const float *tB = b + size_t(t) * 4 * 16;
      const float *tC = c + size_t(t) * 16 * 16;
      float *tD = d + size_t(t) * 16 * 16;
      for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
          // Double accumulator for reproducibility; the kernel (WMMA
          // or the scalar gfx942 fallback) uses f32 accumulation, so
          // the tolerance in `compare` below absorbs the K=4
          // rounding-order delta.
          double acc = static_cast<double>(tC[i * 16 + j]);
          for (int k = 0; k < 4; ++k)
            acc += static_cast<double>(tA[i * 4 + k]) *
                   static_cast<double>(tB[k * 16 + j]);
          tD[i * 16 + j] = static_cast<float>(acc);
        }
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int /*blockSize*/) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "wmma_f32_16x16x4_gemm"));

    size_t bytesA = size_t(N) * 16 * 4 * sizeof(float);
    size_t bytesB = size_t(N) * 4 * 16 * sizeof(float);
    size_t bytesC = size_t(N) * 16 * 16 * sizeof(float);
    size_t bytesD = bytesC;

    float *dA, *dB, *dC, *dD;
    HIP_ASSERT(hipMalloc(&dA, bytesA));
    HIP_ASSERT(hipMalloc(&dB, bytesB));
    HIP_ASSERT(hipMalloc(&dC, bytesC));
    HIP_ASSERT(hipMalloc(&dD, bytesD));
    // Sentinel so unwritten D slots are visible in the comparison.
    HIP_ASSERT(hipMemset(dD, 0xA5, bytesD));

    HIP_ASSERT(hipMemcpy(dA, input.data(),
                         bytesA, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dB, input.data() + bytesA,
                         bytesB, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dC, input.data() + bytesA + bytesB,
                         bytesC, hipMemcpyHostToDevice));

    struct alignas(8) Args {
      const float *a;
      const float *b;
      const float *c;
      float *d;
    } args = {dA, dB, dC, dD};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};
    // One block per tile; block = one Wave32 (32 threads).
    HIP_ASSERT(hipModuleLaunchKernel(fn, N, 1, 1, 32, 1, 1, 0,
                                     nullptr, nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());

    std::vector<uint8_t> out(bytesD);
    HIP_ASSERT(hipMemcpy(out.data(), dD, bytesD, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dA));
    HIP_ASSERT(hipFree(dB));
    HIP_ASSERT(hipFree(dC));
    HIP_ASSERT(hipFree(dD));
    return out;
  };

  r.compare = [](const std::vector<uint8_t> &gold,
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    double maxAbs = 0.0;
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    // K=4 with inputs in [-1, 1]: worst-case kernel accumulation
    // error is a handful of f32 ulps against the double-accumulated
    // CPU gold. 1e-5 is tight enough to flag a genuinely-wrong
    // lowering while absorbing the legitimate rounding delta.
    const double tol = 1e-5;
    for (int i = 0; i < outElems; ++i) {
      double d = std::fabs(static_cast<double>(a[i]) -
                           static_cast<double>(g[i]));
      if (d > maxAbs) maxAbs = d;
      if (d > tol) {
        if (mismatches++ == 0) {
          firstIdx = i;
          firstG = g[i];
          firstA = a[i];
        }
      }
    }
    return std::make_tuple(mismatches, maxAbs, firstIdx, firstG, firstA);
  };
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe: mubuf_store_b32 — isolate the raw buffer_store_b32 lowering.
//
// Authored 2026-04-20 to bisect Gfx1250Gpu.Softmax (which writes nothing on
// the salmon path). Softmax is the only currently-failing kernel that stores
// through MUBUF raw buffer ops; every passing kernel stores through global
// addressing. This probe keeps MUBUF as the only differentiator: one
// global_load, one pointwise op, one buffer_store_b32 via
// __builtin_amdgcn_raw_buffer_store_b32. If this recipe passes on salmon,
// MUBUF storing is exonerated and the Softmax failure is in the cross-wave
// reduction / MODE-register / WaveNativeProjection path upstream of the
// store. If it fails, buildMubufSRD() or handle_mubuf.cpp is the fix site.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeMubufStoreB32Recipe() {
  Recipe r;
  r.name = "mubuf_store_b32";
  r.defaultNs     = {16, 64, 256, 1024, 4096};
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(0xB0FFACEu + N);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) f[i] = dist(rng);
    return buf;
  };

  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *in = reinterpret_cast<const float *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    auto *o = reinterpret_cast<float *>(out.data());
    for (int i = 0; i < N; ++i) o[i] = in[i] * 2.0f + 1.0f;
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "mubuf_store_b32"));
    float *dIn, *dOut;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    // Sentinel 0xA5A5A5A5 as float ≈ -1.5e-16 — if the kernel writes
    // nothing, the comparator sees a wildly wrong value and the probe
    // fails loudly rather than silently passing on a zero output.
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const float *in; float *out; int n; }
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    double maxAbs = 0.0;
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    const double tol = 1e-5;
    for (int i = 0; i < outElems; ++i) {
      double d = std::fabs(static_cast<double>(a[i]) -
                           static_cast<double>(g[i]));
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
// Recipe: s_set_vgpr_msb_probe — force gfx1250 VGPR-bank MSB emission.
//
// Authored 2026-04-20 to bisect Gfx1250Gpu.Matmul128x128_1tile (16 238
// numerical mismatches on salmon; Matmul64x64 passes). An instruction diff
// of the two HSACOs shows s_set_vgpr_msb present 3× in the 128 kernel and
// absent in the 64 kernel — it is the distinguishing feature. This probe
// pressures the allocator with 64 live f32 values carried across a
// non-collapsible pointwise chain, so the compiler names VGPRs ≥ 256 on
// gfx1250 and emits s_set_vgpr_msb between blocks. No WMMA, no cross-lane,
// no LDS: the MSB handling in handle_sopp.cpp + computeVGPRAdjust() is the
// only axis under test.
// ─────────────────────────────────────────────────────────────────────────────

Recipe makeSSetVgprMsbRecipe() {
  Recipe r;
  r.name = "s_set_vgpr_msb";
  r.defaultNs     = {128, 1024, 4096};
  // Small blocks keep per-wave VGPR pressure high; the kernel body carries
  // 64 live f32s regardless of block size, and the per-wave allocator
  // decides its VGPR count per-wave, not per-block. Mirror the Matmul128
  // launch geometry (blockSize=128) explicitly.
  r.defaultBlocks = {64, 128, 256};
  r.outputElemBytes = sizeof(float);
  r.outputElems = [](int N, int) { return N; };

  r.makeInput = [](int N) {
    std::vector<uint8_t> buf(N * sizeof(float));
    auto *f = reinterpret_cast<float *>(buf.data());
    std::mt19937 rng(0x5E7C0DEu + N);
    // Inputs in [-1, 1] keep the 64-step chain numerically bounded
    // (|v_i| stays < ~2 across the whole chain), so mismatches point at
    // compute error rather than overflow.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) f[i] = dist(rng);
    return buf;
  };

  // Mirror the kernel's dependency chain exactly. Any drift here would
  // make the recipe catch its own CPU/kernel divergence instead of a real
  // transpiler bug, so keep this in lock-step with kernels/s_set_vgpr_msb.hip.
  r.cpuReference = [](const std::vector<uint8_t> &input, int N, int) {
    const float *in = reinterpret_cast<const float *>(input.data());
    std::vector<uint8_t> out(N * sizeof(float));
    auto *o = reinterpret_cast<float *>(out.data());
    for (int tid = 0; tid < N; ++tid) {
      float v[64];
      v[0] = in[tid];
      for (int i = 1; i < 64; ++i) {
        float prev = v[i - 1];
        float step = static_cast<float>(i);
        v[i] = ((i & 1) ? -prev : prev) * 1.001f + step * 1e-3f;
      }
      float sum = 0.0f;
      for (int i = 0; i < 32; ++i) sum += v[i] * v[63 - i];
      o[tid] = sum;
    }
    return out;
  };

  r.dispatch = [](hipModule_t mod, const std::vector<uint8_t> &input,
                  int N, int blockSize) {
    hipFunction_t fn;
    HIP_ASSERT(hipModuleGetFunction(&fn, mod, "s_set_vgpr_msb_probe"));
    float *dIn, *dOut;
    size_t bytes = N * sizeof(float);
    HIP_ASSERT(hipMalloc(&dIn, bytes));
    HIP_ASSERT(hipMalloc(&dOut, bytes));
    HIP_ASSERT(hipMemset(dOut, 0xA5, bytes));
    HIP_ASSERT(hipMemcpy(dIn, input.data(), bytes, hipMemcpyHostToDevice));
    struct alignas(8) Args { const float *in; float *out; int n; }
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
                 const std::vector<uint8_t> &actual,
                 int /*N*/, int /*blockSize*/, int outElems) {
    const float *g = reinterpret_cast<const float *>(gold.data());
    const float *a = reinterpret_cast<const float *>(actual.data());
    double maxAbs = 0.0;
    int mismatches = 0, firstIdx = -1;
    double firstG = 0.0, firstA = 0.0;
    // 64-step chain with |step|<2 and 32-term inner product: absolute
    // error budget is ~128 f32 ulps at the output magnitude, which at
    // worst is ~1e-4. Keep the tolerance tight enough to flag a real
    // MSB-handling bug (those are order-of-magnitude errors) while
    // absorbing legitimate rounding.
    const double tol = 1e-4;
    for (int i = 0; i < outElems; ++i) {
      double d = std::fabs(static_cast<double>(a[i]) -
                           static_cast<double>(g[i]));
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
// Triton recipe plumbing
// ─────────────────────────────────────────────────────────────────────────────
//
// Triton recipes are the same three-mode comparison as HIP recipes, with two
// differences:
//
//   1. The gold is the native gfx942 run, not a CPU reference.  Writing a CPU
//      reference per Triton kernel would kill most of the leverage of being
//      able to throw arbitrary kernels at the harness.  The trust boundary
//      moves to "hipcc / Triton on gfx942 compute the right answer".
//
//   2. Kernarg packing is driven from the code object's .note.amdgpu_metadata
//      (extracted at AOT-compile time and recorded in a per-recipe .sidecar
//      file) rather than a hand-written `struct Args { ... }`.  That means a
//      new Triton recipe is just a new .py kernel plus its sidecar — no C++
//      to write per kernel.
//
// The sidecar is a minimal JSON document produced by kernels/triton/aot_compile.py.
// See kernels/triton/README.md for the schema.

// Per-ISA metadata extracted from the .co's .note.amdgpu_metadata section and
// cached in the sidecar.  kernargSegmentSize and args[] are ISA-stable in
// practice (Triton's ABI is ISA-agnostic) but groupSegmentFixedSize and
// maxFlatWorkgroupSize can legitimately differ between gfx942 (wave64) and
// gfx1250 (wave32), so we keep both around.
struct TritonArchMeta {
  int kernargSegmentSize     = 0;
  int groupSegmentFixedSize  = 0;
  // Per-thread scratch (register spill).  Phase 1 refuses to launch when
  // this is non-zero — we'd need to allocate a scratch buffer and fill the
  // matching `global_scratch_base` implicit kernarg slot for the kernel to
  // run correctly, and we don't.  Asserting up-front turns a silent fault
  // into a loud error at sidecar load time.
  int privateSegmentFixedSize = 0;
  int maxFlatWorkgroupSize   = 256;
  // Dynamic LDS (per-block shared-memory scratch) that Triton's AMD
  // backend reserves for reductions / softmax / similar cross-wave
  // accumulators.  Sourced from `compiled.metadata.shared` at AOT time
  // (see kernels/triton/aot_compile.py), NOT from the HSACO ELF:
  // `group_segment_fixed_size` covers *static* LDS only and is 0 for
  // every Triton kernel we emit (Triton always uses the dynamic path).
  // Passed as hipModuleLaunchKernel's `dynamicSharedMemBytes` argument;
  // passing 0 when the kernel needs N bytes makes the reduction load/
  // store to address 0 in LDS and silently return 0 (which is how
  // layer-norm and softmax silently produced broken output before this
  // field was plumbed through).
  int sharedMemBytes         = 0;
  // ALL kernarg slots in offset order, including the implicit args Triton
  // appends past the user signature (typically `global_scratch_base` and
  // similar).  The dispatch only writes the first signature.size() slots
  // and leaves the trailing ones as zero in the kernarg buffer; that's
  // safe iff privateSegmentFixedSize == 0 and the kernel doesn't deref
  // the other implicit pointers (hostcall_buffer, etc.).
  struct Arg {
    int offset = 0;
    int size  = 0;
    std::string valueKind;  // "global_buffer", "by_value", "hidden_*", …
  };
  std::vector<Arg> args;
};

struct TritonSigArg {
  std::string name;
  std::string type;  // Triton signature type: "*fp16", "*fp32", "i32", etc.
};

// TritonBufferDesc and TritonRecipe are defined further down so they can
// hold CachedExpr by value (CachedExpr depends on the expression parser,
// which is declared below the JSON parser).

// ─────────────────────────────────────────────────────────────────────────────
// Parse error context
// ─────────────────────────────────────────────────────────────────────────────
//
// Both the JSON parser and the expression parser are reused across many
// sidecar files and many expressions.  A bare die() from one of those
// helpers leaves the user staring at "json: missing required key foo" with
// no way to tell which file produced it.  ParseContextScope sets a
// file-scope global describing the currently-active parse target; helpers
// append it to their die() messages via contextSuffix().  Stacked scopes
// nest correctly because each scope captures and restores the previous
// value in its destructor.
//
// All parsing happens single-threaded at recipe-load / dispatch time, so
// there's no race to worry about.
std::string gParseContext;

class ParseContextScope {
 public:
  explicit ParseContextScope(std::string ctx) : prev_(gParseContext) {
    // Nest the new context inside the previous one so error messages show
    // the full chain (innermost first), e.g. "missing key foo [in arch_meta
    // gfx942 < sidecar /path/foo.sidecar.json]".  Replacing prev_ entirely
    // would lose the outer file context, which is exactly the bug this
    // class exists to prevent.
    if (prev_.empty())
      gParseContext = std::move(ctx);
    else
      gParseContext = std::move(ctx) + " < " + prev_;
  }
  ~ParseContextScope() { gParseContext = std::move(prev_); }
  ParseContextScope(const ParseContextScope &) = delete;
  ParseContextScope &operator=(const ParseContextScope &) = delete;
  ParseContextScope(ParseContextScope &&) = delete;
  ParseContextScope &operator=(ParseContextScope &&) = delete;
 private:
  std::string prev_;
};

std::string contextSuffix() {
  return gParseContext.empty() ? std::string()
                               : (" [in " + gParseContext + "]");
}

// Concatenate any mix of std::string / const char* into one string.  Used
// to build descriptive ParseContextScope strings without printf-style
// formatting noise at the call site.
namespace ctx_detail {
inline void append(std::string &) {}
template <typename T, typename... Rest>
inline void append(std::string &out, const T &first, const Rest &... rest) {
  out += first;
  append(out, rest...);
}
}  // namespace ctx_detail
template <typename... Args>
std::string ctx_str(const Args &... args) {
  std::string s;
  ctx_detail::append(s, args...);
  return s;
}

// Minimal JSON parser, specific to the sidecar schema.
// Supports: objects, arrays, strings (with \n, \t, \", \\, \/), numbers
// (integer and floating), booleans, null.  No unicode escapes (we don't need
// them for the sidecar), no trailing comma tolerance, no comments.  Errors
// fail loudly with the surrounding context; this matches the project-wide
// "fail loudly" rule.
class JsonValue {
 public:
  enum Kind { NullK, BoolK, IntK, DoubleK, StringK, ArrayK, ObjectK };
  Kind kind = NullK;
  bool boolVal = false;
  int64_t intVal = 0;
  double doubleVal = 0.0;
  std::string stringVal;
  std::vector<JsonValue> arrayVal;
  // Insertion-order preserved.
  std::vector<std::pair<std::string, JsonValue>> objectVal;

  bool isObject() const { return kind == ObjectK; }
  bool isArray() const  { return kind == ArrayK; }
  const std::vector<JsonValue> &asArray() const {
    if (kind != ArrayK)
      die("json: expected array, got kind=%d%s",
          (int)kind, contextSuffix().c_str());
    return arrayVal;
  }
  const std::string &asString() const {
    if (kind != StringK)
      die("json: expected string, got kind=%d%s",
          (int)kind, contextSuffix().c_str());
    return stringVal;
  }
  int64_t asInt() const {
    if (kind == IntK) return intVal;
    if (kind == DoubleK) return static_cast<int64_t>(doubleVal);
    die("json: expected integer, got kind=%d%s",
        (int)kind, contextSuffix().c_str());
  }
  double asDouble() const {
    if (kind == DoubleK) return doubleVal;
    if (kind == IntK)    return static_cast<double>(intVal);
    die("json: expected number, got kind=%d%s",
        (int)kind, contextSuffix().c_str());
  }
  const JsonValue *find(const std::string &key) const {
    if (kind != ObjectK)
      die("json: expected object (looking up key %s), got kind=%d%s",
          key.c_str(), (int)kind, contextSuffix().c_str());
    for (const auto &kv : objectVal)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  const JsonValue &get(const std::string &key) const {
    if (const auto *v = find(key)) return *v;
    die("json: missing required key %s%s",
        key.c_str(), contextSuffix().c_str());
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string &text) : s_(text), i_(0) {}
  JsonValue parse() {
    skipWs();
    JsonValue v = parseValue();
    skipWs();
    if (i_ < s_.size())
      die("json: trailing data at offset %zu%s",
          i_, contextSuffix().c_str());
    return v;
  }
 private:
  const std::string &s_;
  size_t i_;

  [[noreturn]] void fail(const char *what) const {
    // Include ~40 chars of context around the current offset.
    size_t lo = i_ > 40 ? i_ - 40 : 0;
    size_t hi = std::min(s_.size(), i_ + 20);
    die("json: %s at offset %zu (context: …%s<HERE>%s…)%s", what, i_,
        s_.substr(lo, i_ - lo).c_str(),
        s_.substr(i_, hi - i_).c_str(),
        contextSuffix().c_str());
  }
  void skipWs() {
    while (i_ < s_.size()) {
      char c = s_[i_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i_;
      else break;
    }
  }
  bool eat(char c) {
    if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; }
    return false;
  }
  void expect(char c) {
    if (!eat(c)) fail((std::string("expected '") + c + "'").c_str());
  }
  JsonValue parseValue() {
    skipWs();
    if (i_ >= s_.size()) fail("unexpected end of input");
    char c = s_[i_];
    if (c == '{') return parseObject();
    if (c == '[') return parseArray();
    if (c == '"') return parseString();
    if (c == 't' || c == 'f') return parseBool();
    if (c == 'n') return parseNull();
    return parseNumber();
  }
  JsonValue parseObject() {
    expect('{');
    JsonValue v; v.kind = JsonValue::ObjectK;
    skipWs();
    if (eat('}')) return v;
    for (;;) {
      skipWs();
      auto key = parseString().stringVal;
      skipWs(); expect(':');
      JsonValue val = parseValue();
      v.objectVal.emplace_back(std::move(key), std::move(val));
      skipWs();
      if (eat(',')) continue;
      expect('}');
      break;
    }
    return v;
  }
  JsonValue parseArray() {
    expect('[');
    JsonValue v; v.kind = JsonValue::ArrayK;
    skipWs();
    if (eat(']')) return v;
    for (;;) {
      v.arrayVal.push_back(parseValue());
      skipWs();
      if (eat(',')) continue;
      expect(']');
      break;
    }
    return v;
  }
  JsonValue parseString() {
    expect('"');
    JsonValue v; v.kind = JsonValue::StringK;
    std::string &out = v.stringVal;
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '"') return v;
      if (c == '\\') {
        if (i_ >= s_.size()) fail("unterminated escape");
        char e = s_[i_++];
        switch (e) {
          case '"':  out.push_back('"');  break;
          case '\\': out.push_back('\\'); break;
          case '/':  out.push_back('/');  break;
          case 'n':  out.push_back('\n'); break;
          case 't':  out.push_back('\t'); break;
          case 'r':  out.push_back('\r'); break;
          case 'b':  out.push_back('\b'); break;
          case 'f':  out.push_back('\f'); break;
          default:
            fail("unsupported escape (only \\\" \\\\ \\/ \\n \\t \\r \\b \\f)");
        }
      } else {
        out.push_back(c);
      }
    }
    fail("unterminated string");
  }
  JsonValue parseBool() {
    JsonValue v; v.kind = JsonValue::BoolK;
    if (s_.compare(i_, 4, "true") == 0)  { i_ += 4; v.boolVal = true;  return v; }
    if (s_.compare(i_, 5, "false") == 0) { i_ += 5; v.boolVal = false; return v; }
    fail("expected true|false");
  }
  JsonValue parseNull() {
    if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return JsonValue(); }
    fail("expected null");
  }
  JsonValue parseNumber() {
    size_t start = i_;
    if (s_[i_] == '-') ++i_;
    bool hasDot = false, hasExp = false;
    while (i_ < s_.size()) {
      char c = s_[i_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '+' || c == '-') {
        if (c == '.') hasDot = true;
        if (c == 'e' || c == 'E') hasExp = true;
        ++i_;
      } else {
        break;
      }
    }
    std::string tok = s_.substr(start, i_ - start);
    JsonValue v;
    if (hasDot || hasExp) {
      v.kind = JsonValue::DoubleK;
      v.doubleVal = std::strtod(tok.c_str(), nullptr);
    } else {
      v.kind = JsonValue::IntK;
      v.intVal = std::strtoll(tok.c_str(), nullptr, 10);
    }
    return v;
  }
};

JsonValue parseJsonFile(const std::string &path) {
  // Use the path itself as the parse context so missing-key / type-error
  // messages from JsonValue::get / asInt / etc. include the file name.
  // Caller may install a more specific context (e.g. "sidecar foo.json"
  // followed by per-section sub-contexts) before calling into JsonValue
  // helpers; the scope stack handles nesting.
  ParseContextScope ctx(path);
  auto bytes = readFile(path);
  std::string text(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  JsonParser p(text);
  return p.parse();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiny expression evaluator for grid/elems expressions
// ─────────────────────────────────────────────────────────────────────────────
//
// Supports: integer literals, identifiers (looked up in scope), the
// `ceil_div(a, b)` function, parens, and infix `+ - * /`.  This is enough for
// Triton launch grids and for buffer-size expressions like "M * K".
//
// Two-phase: parsing builds an AST (ExprNode); evaluation interprets the AST
// against a scope map.  Recipes parse each grid/elems expression once at
// sidecar load time and cache the AST in CachedExpr; per-launch evaluation
// is then an AST walk with no string allocation.  The string source text is
// retained on CachedExpr for diagnostic messages.
//
// The grammar is intentionally limited; anything outside the supported set
// fails loudly at parse time so the sidecar producer sees a clear error
// instead of a silently-wrong computation.

struct ExprNode {
  enum Kind { Lit, Ident, Add, Sub, Mul, Div, CeilDiv };
  Kind kind = Lit;
  int64_t literal = 0;
  std::string ident;                // Ident only
  std::shared_ptr<ExprNode> a, b;   // BinOp / CeilDiv operands
};

class ExprParser {
 public:
  explicit ExprParser(std::string text) : s_(std::move(text)), i_(0) {}
  std::shared_ptr<ExprNode> parse() {
    skipWs();
    auto v = parseAddSub();
    skipWs();
    if (i_ < s_.size())
      die("expr: trailing junk at offset %zu in %s%s",
          i_, s_.c_str(), contextSuffix().c_str());
    return v;
  }
 private:
  std::string s_;
  size_t i_;

  void skipWs() {
    while (i_ < s_.size() &&
           (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n')) ++i_;
  }
  std::shared_ptr<ExprNode> parseAddSub() {
    auto v = parseMulDiv();
    for (;;) {
      skipWs();
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) {
        char op = s_[i_++];
        auto r = parseMulDiv();
        auto n = std::make_shared<ExprNode>();
        n->kind = (op == '+') ? ExprNode::Add : ExprNode::Sub;
        n->a = std::move(v);
        n->b = std::move(r);
        v = std::move(n);
      } else break;
    }
    return v;
  }
  std::shared_ptr<ExprNode> parseMulDiv() {
    auto v = parsePrimary();
    for (;;) {
      skipWs();
      if (i_ < s_.size() && (s_[i_] == '*' || s_[i_] == '/')) {
        char op = s_[i_++];
        auto r = parsePrimary();
        auto n = std::make_shared<ExprNode>();
        n->kind = (op == '*') ? ExprNode::Mul : ExprNode::Div;
        n->a = std::move(v);
        n->b = std::move(r);
        v = std::move(n);
      } else break;
    }
    return v;
  }
  std::shared_ptr<ExprNode> parsePrimary() {
    skipWs();
    if (i_ >= s_.size())
      die("expr: unexpected end in %s%s",
          s_.c_str(), contextSuffix().c_str());
    char c = s_[i_];
    if (c == '(') {
      ++i_;
      auto v = parseAddSub();
      skipWs();
      if (i_ >= s_.size() || s_[i_] != ')')
        die("expr: missing ')' in %s%s",
            s_.c_str(), contextSuffix().c_str());
      ++i_;
      return v;
    }
    if ((c >= '0' && c <= '9') || c == '-') {
      size_t start = i_;
      if (c == '-') ++i_;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
      auto n = std::make_shared<ExprNode>();
      n->kind = ExprNode::Lit;
      n->literal =
          std::strtoll(s_.substr(start, i_ - start).c_str(), nullptr, 10);
      return n;
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
      size_t start = i_;
      while (i_ < s_.size() &&
             ((s_[i_] >= 'a' && s_[i_] <= 'z') ||
              (s_[i_] >= 'A' && s_[i_] <= 'Z') ||
              (s_[i_] >= '0' && s_[i_] <= '9') || s_[i_] == '_')) ++i_;
      std::string ident = s_.substr(start, i_ - start);
      skipWs();
      if (i_ < s_.size() && s_[i_] == '(') {
        // Function call.  Only ceil_div is supported.
        ++i_;
        auto a = parseAddSub();
        skipWs();
        if (i_ >= s_.size() || s_[i_] != ',')
          die("expr: expected ',' in call to %s%s",
              ident.c_str(), contextSuffix().c_str());
        ++i_;
        auto b = parseAddSub();
        skipWs();
        if (i_ >= s_.size() || s_[i_] != ')')
          die("expr: missing ')' in call to %s%s",
              ident.c_str(), contextSuffix().c_str());
        ++i_;
        if (ident != "ceil_div")
          die("expr: unknown function %s (only ceil_div supported)%s",
              ident.c_str(), contextSuffix().c_str());
        auto n = std::make_shared<ExprNode>();
        n->kind = ExprNode::CeilDiv;
        n->a = std::move(a);
        n->b = std::move(b);
        return n;
      }
      auto n = std::make_shared<ExprNode>();
      n->kind = ExprNode::Ident;
      n->ident = std::move(ident);
      return n;
    }
    die("expr: unexpected '%c' at offset %zu in %s%s",
        c, i_, s_.c_str(), contextSuffix().c_str());
  }
};

int64_t evalNode(const ExprNode &n,
                 const std::map<std::string, int> &scope) {
  switch (n.kind) {
    case ExprNode::Lit:   return n.literal;
    case ExprNode::Ident: {
      auto it = scope.find(n.ident);
      if (it == scope.end())
        die("expr: unknown identifier %s%s",
            n.ident.c_str(), contextSuffix().c_str());
      return it->second;
    }
    case ExprNode::Add: return evalNode(*n.a, scope) + evalNode(*n.b, scope);
    case ExprNode::Sub: return evalNode(*n.a, scope) - evalNode(*n.b, scope);
    case ExprNode::Mul: return evalNode(*n.a, scope) * evalNode(*n.b, scope);
    case ExprNode::Div: {
      int64_t r = evalNode(*n.b, scope);
      if (r == 0) die("expr: division by zero%s", contextSuffix().c_str());
      return evalNode(*n.a, scope) / r;
    }
    case ExprNode::CeilDiv: {
      int64_t a = evalNode(*n.a, scope);
      int64_t r = evalNode(*n.b, scope);
      if (r == 0)
        die("expr: ceil_div divisor is zero%s", contextSuffix().c_str());
      return (a + r - 1) / r;
    }
  }
  die("expr: invalid AST kind=%d%s",
      static_cast<int>(n.kind), contextSuffix().c_str());
}

// Source text + parsed AST.  Source is retained so error messages can show
// the original expression instead of a reconstructed pretty-print.
struct CachedExpr {
  std::string source;
  std::shared_ptr<ExprNode> ast;
};

CachedExpr makeCachedExpr(std::string text) {
  ExprParser p(text);
  auto ast = p.parse();
  return CachedExpr{std::move(text), std::move(ast)};
}

int64_t evalCached(const CachedExpr &e,
                   const std::map<std::string, int> &scope) {
  if (!e.ast)
    die("expr: empty AST for source=%s%s",
        e.source.c_str(), contextSuffix().c_str());
  return evalNode(*e.ast, scope);
}

// ─────────────────────────────────────────────────────────────────────────────
// Triton buffer / recipe types (depend on CachedExpr)
// ─────────────────────────────────────────────────────────────────────────────

// A buffer that the recipe materializes as an input or reads back as an
// output.  `elems` is parsed once at sidecar load time and evaluated per
// dispatch against the current shape and constexprs (e.g. "N" or "M * K").
// Multi-dim shapes are supported at the expression level; only the
// top-level shape sweep is single-dim in Phase 1.
//
// Inputs may carry an optional uniform-random range [rangeLo, rangeHi).
// The defaults ([-1, 1)) suit elementwise kernels whose outputs stay in a
// well-behaved range; reduction kernels (softmax, layernorm) typically
// want a tighter range like [-0.1, 0.1) so per-row sums don't overflow
// the fp16 representable range.  Integer dtypes ignore the range fields
// (they always sample full-range bits).
//
// Outputs may carry an optional per-output comparator that overrides the
// recipe-level comparator.  This is useful when a single recipe produces
// outputs with very different numerical properties (e.g. layer_norm
// returns Y, mean, rstd: Y wants relative error, rstd wants absolute).
// `hasComparator == false` means "fall back to TritonRecipe::comparator".
struct TritonComparator {
  std::string kind;  // "abs", "rel", or "rel-rms"
  double tol = 1e-5;
};

// Output pre-launch init mode.  Default is Sentinel (0xA5 fill) so any
// lane the kernel fails to write stays visibly nonsense in the diff.
// Zero is required for kernels whose outputs are written via
// `tl.atomic_add` / atomic_max / similar read-modify-write primitives:
// a 0xA5 pre-fill would poison the initial value and produce
// `0xA5A5A5A5 + sum` instead of `sum`.  Each recipe's outputs declare
// their mode explicitly; silently defaulting to Sentinel for a kernel
// that needs Zero is precisely the class of silent miscompile this
// field exists to prevent.
enum class TritonOutputInit { Sentinel, Zero };

struct TritonBufferDesc {
  std::string name;        // must match a signature arg
  std::string dtype;       // "fp16", "bf16", "fp32", "fp64", "i32", "i64"
  CachedExpr elems;        // expression in identifiers from shape/constexprs
  // Inputs: uniform-random range for float dtypes (ignored for ints).
  double rangeLo = -1.0;
  double rangeHi =  1.0;
  // Outputs: optional per-output comparator override.
  bool hasComparator = false;
  TritonComparator comparator;
  // Outputs: pre-launch init policy (see TritonOutputInit).  Ignored
  // for inputs.
  TritonOutputInit init = TritonOutputInit::Sentinel;
};

struct TritonRecipe {
  std::string name;           // recipe name; also used for the .co filename stem
  std::string kernelSymbol;   // symbol to look up via hipModuleGetFunction
  int numWarps = 4;
  std::vector<TritonSigArg> signature;
  // Triton-side constexprs.  Triton bakes these as compile-time constants
  // into the .co; their names must match `: tl.constexpr` parameters of
  // the kernel function.  They *also* land in the C++ scope so `elems` /
  // `grid` expressions can reference them by name.
  std::map<std::string, int> constexprs;
  // Harness-only scope entries.  These feed the integer scope used by
  // `elems` / `grid` / `scalar_args` expression eval, but they are NOT
  // passed to Triton — useful for "logical" recipe dimensions that
  // aren't kernel parameters at all (e.g. layer-norm's row count `M`,
  // which is encoded purely via the launch grid), or for runtime sig
  // args that the harness wants to hold constant across the sweep
  // without having Triton bake them as constexprs (e.g. softmax's
  // `n_rows`, where baking would also drop the kernarg slot).
  std::map<std::string, int> harnessConstants;
  // Sweep: each entry is one scalar shape value (Phase 1 limits Triton shapes
  // to a single scalar dimension whose name is `shapeDimName`, e.g. "N").
  // The harness sweeps one run per value.  Multi-dim shapes can be added in
  // Phase 2 without touching the per-recipe code: the TritonRecipe just grows
  // a richer default-shape list and a shape-label reporting column.
  std::string shapeDimName;
  std::vector<int> defaultShapeValues;
  // Launch grid: expressions parsed once at sidecar load time and evaluated
  // per dispatch against the shape scope ({dim, constexprs...}).
  // `ceil_div(a, b)` is supported.
  struct { CachedExpr x, y, z; } grid;
  std::vector<TritonBufferDesc> inputs;   // materialized from deterministic RNG
  std::vector<TritonBufferDesc> outputs;  // read back and compared
  // Default comparator applied to any output that doesn't carry its own.
  TritonComparator comparator;
  // Computed scalar args.  Each entry maps a scalar sig arg name to an
  // expression evaluated at dispatch time against (shapeDim + constexprs).
  // For integer sig types the expression is the usual integer expr (same
  // grammar as `elems` and `grid`).  For floating-point sig types the
  // expression is a literal numeric value (e.g. "1e-5") parsed once at
  // sidecar-load time into `scalarArgsFloat`.
  //
  // A scalar sig arg may be resolved by, in priority order:
  //   1. scalarArgsInt[name]   (integer expression)
  //   2. scalarArgsFloat[name] (float literal; only for fp* sig types)
  //   3. scope[name]           (legacy: arg name == constexpr/shape-dim key)
  //
  // The legacy fallback is what makes vecadd's `N` work without a
  // scalar_args entry.
  std::map<std::string, CachedExpr> scalarArgsInt;
  std::map<std::string, double>     scalarArgsFloat;
  std::map<std::string, TritonArchMeta> archMeta;  // keyed by "gfx942", "gfx1250"
};

// ─────────────────────────────────────────────────────────────────────────────
// Sidecar loader
// ─────────────────────────────────────────────────────────────────────────────

// Bytes per Triton dtype.  Keep narrow — the sidecar writer only emits what we
// actually handle, and any new dtype triggers a fatal error rather than a
// silent fallback.
int dtypeBytes(const std::string &dtype) {
  if (dtype == "fp16" || dtype == "bf16" ||
      dtype == "i16"  || dtype == "u16")                    return 2;
  if (dtype == "fp32" ||
      dtype == "i32"  || dtype == "u32")                    return 4;
  if (dtype == "fp64" ||
      dtype == "i64"  || dtype == "u64")                    return 8;
  if (dtype == "i8"   || dtype == "u8")                     return 1;
  die("triton: unsupported dtype %s (extend dtypeBytes when needed)",
      dtype.c_str());
}

TritonArchMeta parseArchMeta(const JsonValue &v) {
  // The caller already installed a sidecar-level ParseContextScope; add
  // arch-level granularity so a missing/wrong field reports both file and
  // arch (e.g. "missing required key kernarg_segment_size [in arch_meta
  // gfx942 [in sidecar foo.json]]").
  ParseContextScope ctx("arch_meta entry");
  TritonArchMeta m;
  m.kernargSegmentSize       = static_cast<int>(v.get("kernarg_segment_size").asInt());
  m.groupSegmentFixedSize    = static_cast<int>(v.get("group_segment_fixed_size").asInt());
  // Required as of the post-#scratch-guard sidecar schema.  An older
  // sidecar without this field will fail loudly at parse time, forcing a
  // `make clean kernels` rebuild — that's the desired behaviour: stale
  // sidecars don't get to silently skip the scratch check.
  m.privateSegmentFixedSize  = static_cast<int>(v.get("private_segment_fixed_size").asInt());
  m.maxFlatWorkgroupSize     = static_cast<int>(v.get("max_flat_workgroup_size").asInt());
  // Required as of the post-#dyn-lds sidecar schema.  Same stale-sidecar
  // contract as privateSegmentFixedSize above: a sidecar missing this
  // field fails loudly rather than silently defaulting to 0, because
  // "0" is exactly the wrong value for any Triton kernel with a
  // reduction (layer-norm, softmax, ...) — the reduction would silently
  // return 0 output instead of visibly crashing.  See
  // TritonArchMeta::sharedMemBytes for the full provenance.
  m.sharedMemBytes           = static_cast<int>(v.get("shared_mem_bytes").asInt());
  if (m.sharedMemBytes < 0)
    die("triton sidecar: shared_mem_bytes=%d is negative; the sidecar "
        "is malformed (expected a non-negative byte count from "
        "compiled.metadata.shared)", m.sharedMemBytes);
  for (const auto &a : v.get("args").asArray()) {
    TritonArchMeta::Arg arg;
    arg.offset    = static_cast<int>(a.get("offset").asInt());
    arg.size      = static_cast<int>(a.get("size").asInt());
    arg.valueKind = a.get("value_kind").asString();
    m.args.push_back(arg);
  }
  return m;
}

TritonRecipe parseTritonSidecar(const std::string &path) {
  // Outer scope: every die() raised below this point picks up the sidecar
  // file name through contextSuffix().
  ParseContextScope ctx("sidecar " + path);
  JsonValue root = parseJsonFile(path);
  TritonRecipe t;
  t.name         = root.get("name").asString();
  t.kernelSymbol = root.get("kernel_symbol").asString();
  t.numWarps     = static_cast<int>(root.get("num_warps").asInt());
  for (const auto &s : root.get("signature").asArray()) {
    TritonSigArg sa;
    sa.name = s.get("name").asString();
    sa.type = s.get("type").asString();
    t.signature.push_back(sa);
  }
  if (const auto *ce = root.find("constexprs"); ce && ce->isObject()) {
    for (const auto &kv : ce->objectVal)
      t.constexprs[kv.first] = static_cast<int>(kv.second.asInt());
  }
  if (const auto *hc = root.find("harness_constants"); hc && hc->isObject()) {
    for (const auto &kv : hc->objectVal) {
      // Harness constants share the integer scope with constexprs and
      // the shape dim.  Overlap would make the scope's value depend on
      // insertion order — refuse loudly so the recipe writer picks one
      // place to declare each name.
      if (t.constexprs.count(kv.first))
        die("triton sidecar: harness_constants[%s] also appears in "
            "constexprs; declare each name in exactly one place.",
            kv.first.c_str());
      t.harnessConstants[kv.first] = static_cast<int>(kv.second.asInt());
    }
  }
  const auto &shape = root.get("shape");
  t.shapeDimName = shape.get("dim").asString();
  for (const auto &sv : shape.get("values").asArray())
    t.defaultShapeValues.push_back(static_cast<int>(sv.asInt()));
  // Grid expressions: parse once now; per-launch eval is then an AST walk
  // with no string allocation.
  const auto &grid = root.get("grid");
  {
    ParseContextScope sub(ctx_str("grid expression for ", path));
    t.grid.x = makeCachedExpr(grid.get("x").asString());
    t.grid.y = makeCachedExpr(grid.get("y").asString());
    t.grid.z = makeCachedExpr(grid.get("z").asString());
  }
  // Buffer descriptions: parse the elems expression once per buffer, plus
  // the optional input range and optional per-output comparator override.
  auto loadBuffers = [&](const char *section, bool isInput,
                         std::vector<TritonBufferDesc> &dst) {
    for (const auto &b : root.get(section).asArray()) {
      TritonBufferDesc d;
      d.name  = b.get("name").asString();
      d.dtype = b.get("dtype").asString();
      ParseContextScope sub(
          ctx_str(section, " elems for ", d.name, " in ", path));
      d.elems = makeCachedExpr(b.get("elems").asString());
      if (isInput) {
        if (const auto *lo = b.find("range_lo")) d.rangeLo = lo->asDouble();
        if (const auto *hi = b.find("range_hi")) d.rangeHi = hi->asDouble();
        if (!(d.rangeLo < d.rangeHi))
          die("triton sidecar: input %s has invalid range [%g, %g) "
              "(rangeLo must be strictly less than rangeHi)",
              d.name.c_str(), d.rangeLo, d.rangeHi);
      } else {
        if (const auto *c = b.find("comparator")) {
          d.hasComparator    = true;
          d.comparator.kind  = c->get("kind").asString();
          d.comparator.tol   = c->get("tol").asDouble();
        }
        // Optional per-output pre-launch init mode.  Default:
        // Sentinel (0xA5).  Explicit "zero" is required for atomic-
        // add / atomic-max / atomic-min outputs where the kernel
        // reads the initial value.  Any value other than "sentinel"
        // / "zero" is a hard error — no silent interpretation.
        if (const auto *ini = b.find("init")) {
          const std::string &s = ini->asString();
          if (s == "sentinel") {
            d.init = TritonOutputInit::Sentinel;
          } else if (s == "zero") {
            d.init = TritonOutputInit::Zero;
          } else {
            die("triton sidecar: output %s has init=%s "
                "(must be 'sentinel' or 'zero')",
                d.name.c_str(), s.c_str());
          }
        }
      }
      dst.push_back(std::move(d));
    }
  };
  loadBuffers("inputs",  /*isInput=*/true,  t.inputs);
  loadBuffers("outputs", /*isInput=*/false, t.outputs);
  if (t.outputs.empty())
    die("triton sidecar: at least one output is required");
  // Recipe-level comparator: required, applied to any output that doesn't
  // carry its own override.
  const auto &cmp = root.get("comparator");
  t.comparator.kind = cmp.get("kind").asString();
  t.comparator.tol  = cmp.get("tol").asDouble();
  // Optional `scalar_args`: maps a scalar sig arg name to either an
  // integer expression (string) or a numeric literal (number).  The
  // type of the corresponding scalar sig arg decides how to interpret
  // the value.  Strings → integer expression; numbers → float literal.
  // Anything that doesn't match the sig type is a hard error here, not
  // a silent reinterpret.
  if (const auto *sa = root.find("scalar_args"); sa && sa->isObject()) {
    auto sigArgType = [&](const std::string &name) -> const std::string & {
      for (const auto &s : t.signature)
        if (s.name == name) return s.type;
      die("triton sidecar: scalar_args refers to unknown sig arg %s",
          name.c_str());
    };
    auto isFloatType = [](const std::string &type) {
      return type == "fp16" || type == "bf16" ||
             type == "fp32" || type == "fp64";
    };
    auto isIntType = [](const std::string &type) {
      return type == "i1"  || type == "i8"   || type == "u8"  ||
             type == "i16" || type == "u16"  ||
             type == "i32" || type == "u32"  ||
             type == "i64" || type == "u64";
    };
    for (const auto &kv : sa->objectVal) {
      const std::string &name = kv.first;
      const JsonValue   &val  = kv.second;
      const std::string &type = sigArgType(name);
      ParseContextScope sub(
          ctx_str("scalar_args entry ", name, " in ", path));
      if (val.kind == JsonValue::StringK) {
        if (!isIntType(type))
          die("triton sidecar: scalar_args[%s] is a string expression but "
              "sig type is %s; expression-form is integer-only.  Use a "
              "numeric literal for float sig types.",
              name.c_str(), type.c_str());
        t.scalarArgsInt.emplace(name, makeCachedExpr(val.stringVal));
      } else if (val.kind == JsonValue::IntK ||
                 val.kind == JsonValue::DoubleK) {
        if (!isFloatType(type))
          die("triton sidecar: scalar_args[%s] is a numeric literal but "
              "sig type is %s; numeric-literal form is float-only.  Use "
              "a string expression for integer sig types (e.g. \"%s\").",
              name.c_str(), type.c_str(), name.c_str());
        t.scalarArgsFloat.emplace(name, val.asDouble());
      } else {
        die("triton sidecar: scalar_args[%s] must be a string (integer "
            "expression) or a number (float literal); got JSON kind=%d.",
            name.c_str(), static_cast<int>(val.kind));
      }
    }
  }
  const auto &meta = root.get("metadata");
  if (const auto *m942  = meta.find("gfx942"))  t.archMeta["gfx942"]  = parseArchMeta(*m942);
  if (const auto *m1250 = meta.find("gfx1250")) t.archMeta["gfx1250"] = parseArchMeta(*m1250);
  if (t.archMeta.empty())
    die("triton sidecar: no arch metadata entries (expected gfx942 and/or gfx1250)");
  return t;
}

// Scan kernels/triton/ for *.sidecar.json next to .co files.  The loader runs
// once at harness startup; missing sidecar dir is non-fatal (the tool still
// works as it does today with HIP-only recipes).
std::vector<TritonRecipe> loadTritonRecipes() {
  std::vector<TritonRecipe> out;
  std::string dir = kernelDir();  // same as for HIP .co files
  DIR *d = opendir(dir.c_str());
  if (!d) return out;  // no kernels built yet — the report will be HIP-only
  std::vector<std::string> sidecars;
  while (dirent *e = readdir(d)) {
    std::string name = e->d_name;
    // Match "<stem>.sidecar.json".
    const std::string suffix = ".sidecar.json";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      sidecars.push_back(dir + "/" + name);
  }
  closedir(d);
  std::sort(sidecars.begin(), sidecars.end());
  for (const auto &p : sidecars)
    out.push_back(parseTritonSidecar(p));
  return out;
}

const std::vector<TritonRecipe> &allTritonRecipes() {
  static const std::vector<TritonRecipe> v = loadTritonRecipes();
  return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Triton dispatch (shared across all Triton recipes)
// ─────────────────────────────────────────────────────────────────────────────
//
// The child sets `gCurrentChildIsa` before invoking the Recipe's dispatch
// lambda, so the Triton dispatch can pick the right archMeta entry.  HIP
// dispatches ignore this global.  Single-threaded per child, so no races.
//
// Use ChildIsaScope (RAII) at the call site so the global is always
// cleared on scope exit.  Any future code that batches multiple dispatches
// per child gets correct behaviour without each callsite remembering to
// reset the global.
std::string gCurrentChildIsa;

class ChildIsaScope {
 public:
  explicit ChildIsaScope(const std::string &isa) {
    if (!gCurrentChildIsa.empty())
      die("ChildIsaScope: gCurrentChildIsa already set to %s when "
          "entering scope for %s; nested or stale scope",
          gCurrentChildIsa.c_str(), isa.c_str());
    gCurrentChildIsa = isa;
  }
  ~ChildIsaScope() { gCurrentChildIsa.clear(); }
  ChildIsaScope(const ChildIsaScope &) = delete;
  ChildIsaScope &operator=(const ChildIsaScope &) = delete;
  ChildIsaScope(ChildIsaScope &&) = delete;
  ChildIsaScope &operator=(ChildIsaScope &&) = delete;
};

// RAII wrapper around hipMalloc'd buffers so any die() path inside the
// dispatch unwinds before the child exits.  Reclaiming on process death
// would happen anyway, but freeing explicitly avoids surprises if the
// dispatch is ever reused outside a forked-child context.
class DeviceBufferGuard {
 public:
  DeviceBufferGuard() = default;
  ~DeviceBufferGuard() {
    // hipFree is [[nodiscard]] but a destructor has nowhere to surface the
    // error — at this point either the dispatch finished cleanly (and we'd
    // re-raise on a HIP error from the next call anyway) or we're already
    // unwinding to die().  Cast to void to acknowledge.
    for (void *p : ptrs_) if (p) (void)hipFree(p);
  }
  DeviceBufferGuard(const DeviceBufferGuard &) = delete;
  DeviceBufferGuard &operator=(const DeviceBufferGuard &) = delete;
  void *alloc(size_t bytes) {
    void *p = nullptr;
    HIP_ASSERT(hipMalloc(&p, bytes));
    // push_back can throw bad_alloc if the vector needs to grow and we're
    // out of host RAM.  Without this catch the freshly-hipMalloc'd device
    // buffer would leak (the vector never recorded it).  Reserve up-front
    // would help in the common case but doesn't cover repeated alloc()
    // calls past the reserved capacity, so we keep the defensive try.
    try {
      ptrs_.push_back(p);
    } catch (...) {
      (void)hipFree(p);
      throw;
    }
    return p;
  }
 private:
  std::vector<void *> ptrs_;
};

// Bytes per Triton scalar sig type.  Used to validate that a kernarg slot's
// size matches what the user wrote in the signature — catches the silent
// "i32 at an i64 slot" class of bugs.  Pointer args are checked separately.
int sigScalarBytes(const std::string &type) {
  if (type == "i1")                                                        return 1;
  if (type == "i8"   || type == "u8")                                      return 1;
  if (type == "i16"  || type == "u16" || type == "fp16" || type == "bf16") return 2;
  if (type == "i32"  || type == "u32" || type == "fp32")                   return 4;
  if (type == "i64"  || type == "u64" || type == "fp64")                   return 8;
  return -1;  // unknown; caller reports it
}

// Implicit kernarg slots that Triton appends past the user signature.  We
// don't write to these slots — they stay zero in our kernarg buffer — so
// only kinds the kernel either ignores or reads as "absent" can be safely
// left unfilled.
//
// "Safe" means one of:
//   • The kernel descriptor's enable_* bits don't load it into an SGPR
//     and the kernel's HSAIL/code never dereferences it.
//   • The slot only matters when a runtime feature is enabled (printf,
//     malloc, profile, scratch) and we already gate on the corresponding
//     size being zero (private_segment_fixed_size for scratch).
//
// "Unsafe" kinds (printf hostcall, malloc heap) actively dereference the
// pointer in any kernel that uses the corresponding intrinsic; leaving
// them at null would fault.  Refusing them up front turns a kernel-side
// fault into a clear harness-side error, and forces the engineer adding
// such a kernel to either implement filling or document that the recipe
// can't run in this harness.
const std::set<std::string> &implicitSafeKinds() {
  static const std::set<std::string> s = {
      // global_buffer here is Triton's `global_scratch_base` and
      // `profile_scratch` slots.  Both are NULL-safe iff scratch and
      // profiling are zero-sized — we already assert
      // private_segment_fixed_size == 0 for scratch, and Triton's
      // non-profiling builds emit a zero-sized profile buffer.
      "global_buffer",
      "hidden_global_offset_x",
      "hidden_global_offset_y",
      "hidden_global_offset_z",
      "hidden_dynamic_lds_size",
      "hidden_default_queue",
      "hidden_completion_action",
      "hidden_multigrid_sync_arg",
      "hidden_private_base",
      "hidden_shared_base",
      "hidden_queue_ptr",
      "hidden_grid_dims",
  };
  return s;
}
const std::set<std::string> &implicitUnsafeKinds() {
  static const std::set<std::string> s = {
      "hidden_hostcall_buffer",  // tl.device_print would dereference
      "hidden_heap_v1",          // tl.malloc would dereference
      // hidden_block_count_{x,y,z} are *not* unsafe: tritonDispatch fills
      // them from the launch grid before calling hipModuleLaunchKernel.
      // They used to live here back when the harness left implicit slots
      // at zero, which silently made `tl.num_programs` return 0.
  };
  return s;
}

// Build the integer scope used to evaluate grid / elems / scalar_args
// expressions.  The scope is the union of (constexprs, harness_constants,
// shape_dim).  Overlap between constexprs and harness_constants is already
// rejected at parse time; overlap with shape_dim is rejected here.
std::map<std::string, int>
tritonScope(const TritonRecipe &t, int shapeValue) {
  std::map<std::string, int> scope = t.constexprs;
  for (const auto &kv : t.harnessConstants) {
    scope[kv.first] = kv.second;  // overlap-with-constexprs already rejected
  }
  if (scope.count(t.shapeDimName))
    die("triton %s: shape_dim %s also appears in constexprs / "
        "harness_constants; the swept value would be silently ignored",
        t.name.c_str(), t.shapeDimName.c_str());
  scope[t.shapeDimName] = shapeValue;
  return scope;
}

// Compute total input-blob bytes and the per-input-buffer byte offsets.
// The harness's input blob concatenates all `inputs` buffers in order.
struct TritonInputLayout {
  std::vector<size_t> offsets;  // one per t.inputs entry
  std::vector<size_t> sizes;    // one per t.inputs entry
  size_t totalBytes = 0;
};
TritonInputLayout tritonInputLayout(const TritonRecipe &t, int shapeValue) {
  TritonInputLayout L;
  auto scope = tritonScope(t, shapeValue);
  L.offsets.reserve(t.inputs.size());
  L.sizes.reserve(t.inputs.size());
  size_t off = 0;
  for (const auto &b : t.inputs) {
    int64_t n = evalCached(b.elems, scope);
    if (n < 0) die("triton %s: negative input elems for %s (expr=%s)",
                   t.name.c_str(), b.name.c_str(), b.elems.source.c_str());
    size_t bytes = static_cast<size_t>(n) * dtypeBytes(b.dtype);
    L.offsets.push_back(off);
    L.sizes.push_back(bytes);
    off += bytes;
  }
  L.totalBytes = off;
  return L;
}

// Total output-blob bytes for a given shape.  The output blob concatenates
// all `outputs` buffers in order.  outputElemBytes (for the Recipe struct) is
// just the dtype width of the first output; we use that only as an
// allocation hint.  The comparator operates on the flat byte buffer.
size_t tritonOutputBytes(const TritonRecipe &t, int shapeValue) {
  auto scope = tritonScope(t, shapeValue);
  size_t total = 0;
  for (const auto &b : t.outputs) {
    int64_t n = evalCached(b.elems, scope);
    total += static_cast<size_t>(n) * dtypeBytes(b.dtype);
  }
  return total;
}

// Deterministic random input fill.  fp16 uses __half (stored as uint16_t);
// fp32 uses float.  Integer dtypes get a simple PRNG-derived pattern.  The
// seed is derived from the recipe name and shape so re-runs are reproducible.
std::vector<uint8_t>
tritonMakeInput(const TritonRecipe &t, int shapeValue) {
  auto L = tritonInputLayout(t, shapeValue);
  std::vector<uint8_t> buf(L.totalBytes, 0);
  for (size_t bi = 0; bi < t.inputs.size(); ++bi) {
    const auto &b = t.inputs[bi];
    size_t off = L.offsets[bi];
    size_t sz  = L.sizes[bi];
    // Per-buffer seed: stable across runs, distinct across (recipe, shape,
    // buffer) so different slots don't alias.
    uint64_t seed = std::hash<std::string>{}(t.name + "|" + b.name) ^
                    (static_cast<uint64_t>(shapeValue) * 0x9E3779B97F4A7C15ull);
    std::mt19937_64 rng(seed);
    if (b.dtype == "fp16") {
      auto *out = reinterpret_cast<uint16_t *>(buf.data() + off);
      size_t n = sz / 2;
      std::uniform_real_distribution<float> dist(
          static_cast<float>(b.rangeLo), static_cast<float>(b.rangeHi));
      for (size_t i = 0; i < n; ++i) {
        float f = dist(rng);
        // IEEE 754 f32 → f16 RTZ.  Approximate is fine — identical bytes
        // feed both native and transpiled paths.
        uint32_t u; std::memcpy(&u, &f, sizeof(u));
        uint32_t sign = (u >> 16) & 0x8000u;
        int exp = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x3ff;
        uint16_t h;
        if (exp <= 0)       h = static_cast<uint16_t>(sign);
        else if (exp >= 31) h = static_cast<uint16_t>(sign | 0x7bff);
        else                h = static_cast<uint16_t>(sign | (exp << 10) | mant);
        out[i] = h;
      }
    } else if (b.dtype == "bf16") {
      // bf16 = top 16 bits of fp32, truncated (no rounding).  Both paths
      // see the identical byte pattern, so the truncation choice is moot
      // for cross-ISA comparison; what matters is determinism.
      auto *out = reinterpret_cast<uint16_t *>(buf.data() + off);
      size_t n = sz / 2;
      std::uniform_real_distribution<float> dist(
          static_cast<float>(b.rangeLo), static_cast<float>(b.rangeHi));
      for (size_t i = 0; i < n; ++i) {
        float f = dist(rng);
        uint32_t u; std::memcpy(&u, &f, sizeof(u));
        out[i] = static_cast<uint16_t>(u >> 16);
      }
    } else if (b.dtype == "fp32") {
      auto *out = reinterpret_cast<float *>(buf.data() + off);
      size_t n = sz / 4;
      std::uniform_real_distribution<float> dist(
          static_cast<float>(b.rangeLo), static_cast<float>(b.rangeHi));
      for (size_t i = 0; i < n; ++i) out[i] = dist(rng);
    } else if (b.dtype == "fp64") {
      auto *out = reinterpret_cast<double *>(buf.data() + off);
      size_t n = sz / 8;
      std::uniform_real_distribution<double> dist(b.rangeLo, b.rangeHi);
      for (size_t i = 0; i < n; ++i) out[i] = dist(rng);
    } else if (b.dtype == "i32" || b.dtype == "u32") {
      auto *out = reinterpret_cast<int32_t *>(buf.data() + off);
      size_t n = sz / 4;
      // Full-range 32-bit — masking off the high bits (as an earlier
      // version did) hides any kernel that depends on the sign bit or on
      // the full 32-bit range.  We bit-copy the unsigned 32-bit RNG output
      // into int32 instead of casting because uint32→int32 conversion is
      // implementation-defined when the value exceeds INT32_MAX in C++17;
      // memcpy gives us the unambiguous 2's-complement reinterpretation.
      // i32 and u32 share the same fill path because the RNG samples the
      // full 32-bit bit range either way; the kernel interprets the bits
      // per its declared sig type.
      for (size_t i = 0; i < n; ++i) {
        uint32_t u = static_cast<uint32_t>(rng());
        std::memcpy(&out[i], &u, sizeof(uint32_t));
      }
    } else if (b.dtype == "i64") {
      auto *out = reinterpret_cast<int64_t *>(buf.data() + off);
      size_t n = sz / 8;
      for (size_t i = 0; i < n; ++i) {
        uint64_t u = rng();
        std::memcpy(&out[i], &u, sizeof(int64_t));
      }
    } else {
      die("triton %s: makeInput doesn't handle dtype=%s (extend when needed)",
          t.name.c_str(), b.dtype.c_str());
    }
  }
  return buf;
}

// Metadata-driven kernel launch.  This is the shared body for every Triton
// recipe's dispatch lambda.  It:
//   1. Looks up the archMeta for the current child's ISA.
//   2. Allocates device buffers for every `*<dtype>` sig arg.  Input buffers
//      are copied from the harness's input blob; output buffers are zeroed.
//   3. Builds the kernarg buffer at the exact offsets the code object
//      expects (from archMeta.args), so we never rely on a C struct layout
//      matching Triton's ABI.  Implicit args (past the last sig arg) stay
//      zero-initialized, which is what Triton expects when no scratch is
//      required.
//   4. Launches with the grid computed from the sidecar's grid exprs.
//   5. Reads output buffers back into the output blob in output-spec order.
std::vector<uint8_t>
tritonDispatch(const TritonRecipe &t, hipModule_t mod,
               const std::vector<uint8_t> &inputBlob, int shapeValue) {
  if (gCurrentChildIsa.empty())
    die("triton %s: dispatch called without gCurrentChildIsa set",
        t.name.c_str());
  auto metaIt = t.archMeta.find(gCurrentChildIsa);
  if (metaIt == t.archMeta.end())
    die("triton %s: no archMeta for isa=%s", t.name.c_str(),
        gCurrentChildIsa.c_str());
  const TritonArchMeta &M = metaIt->second;

  // Phase 1 refuses to launch kernels that need scratch.  Filling
  // global_scratch_base / hidden_private_base / similar implicit kernarg
  // slots requires a per-launch scratch buffer that we don't currently
  // allocate; running with zeros there would either fault (best case) or
  // silently corrupt the output (worst).  Surface this loudly and tell the
  // user how to triage.
  if (M.privateSegmentFixedSize != 0) {
    die("triton %s on %s: private_segment_fixed_size=%d (kernel needs "
        "scratch); the harness can't safely launch this in Phase 1.  "
        "Either reduce register pressure (smaller BLOCK_SIZE / num_warps) "
        "or extend tritonDispatch to allocate a scratch buffer and fill "
        "the matching implicit kernarg slot.",
        t.name.c_str(), gCurrentChildIsa.c_str(), M.privateSegmentFixedSize);
  }

  auto scope = tritonScope(t, shapeValue);

  hipFunction_t fn;
  HIP_ASSERT(hipModuleGetFunction(&fn, mod, t.kernelSymbol.c_str()));

  // Track per-pointer-arg device buffers so we can read outputs back; the
  // RAII guard frees them on every exit path (including die()).
  struct DeviceBuf {
    void *ptr = nullptr;
    size_t bytes = 0;
    int outputIndex = -1;  // -1 if this is an input; otherwise index into t.outputs
  };
  std::vector<DeviceBuf> devBufs(t.signature.size());
  DeviceBufferGuard guard;

  auto inputLayout = tritonInputLayout(t, shapeValue);

  // Sanity-check: the number of sig args must be ≤ number of real args in
  // the metadata (Triton appends implicit args after).  Mismatch is a
  // producer/consumer drift bug.
  if (t.signature.size() > M.args.size())
    die("triton %s: signature has %zu args but archMeta has only %zu",
        t.name.c_str(), t.signature.size(), M.args.size());

  // Kernarg buffer: size comes from metadata, not from a hand-packed struct.
  std::vector<uint8_t> kargBuf(M.kernargSegmentSize, 0);

  for (size_t i = 0; i < t.signature.size(); ++i) {
    const auto &sa = t.signature[i];
    const auto &slot = M.args[i];
    bool isPtr = (!sa.type.empty() && sa.type[0] == '*');
    // Two-sided check: pointer args must land in `global_buffer` slots
    // AND scalar args must land in `by_value` slots.  The negation-only
    // form (isPtr != (kind=="global_buffer")) would let a scalar slip into
    // a `hidden_*` slot undetected if Triton ever interleaved an implicit
    // arg inside the user-signature range.  Belt-and-suspenders against
    // future ABI drift.
    const std::string expectedKind = isPtr ? "global_buffer" : "by_value";
    if (slot.valueKind != expectedKind) {
      die("triton %s: arg %s sig_type=%s expects metadata kind=%s "
          "but slot %zu has kind=%s",
          t.name.c_str(), sa.name.c_str(), sa.type.c_str(),
          expectedKind.c_str(), i, slot.valueKind.c_str());
    }
    if (isPtr) {
      // Find whether this arg is an input or an output by name.
      int inIdx = -1, outIdx = -1;
      for (size_t k = 0; k < t.inputs.size();  ++k) if (t.inputs[k].name  == sa.name) inIdx  = (int)k;
      for (size_t k = 0; k < t.outputs.size(); ++k) if (t.outputs[k].name == sa.name) outIdx = (int)k;
      if (inIdx < 0 && outIdx < 0)
        die("triton %s: pointer arg %s is neither in `inputs` nor in `outputs`",
            t.name.c_str(), sa.name.c_str());
      if (inIdx >= 0 && outIdx >= 0)
        die("triton %s: pointer arg %s appears in both `inputs` and `outputs`",
            t.name.c_str(), sa.name.c_str());

      size_t bytes;
      if (inIdx >= 0) {
        bytes = inputLayout.sizes[inIdx];
      } else {
        int64_t n = evalCached(t.outputs[outIdx].elems, scope);
        bytes = static_cast<size_t>(n) * dtypeBytes(t.outputs[outIdx].dtype);
      }

      // Pointers are 8 bytes.
      if (slot.size != 8)
        die("triton %s: pointer arg %s has size=%d (expected 8)",
            t.name.c_str(), sa.name.c_str(), slot.size);

      void *dptr = guard.alloc(bytes);
      // Pre-launch fill.  Inputs will be overwritten with the
      // deterministic host data immediately below, so the initial
      // fill doesn't matter (we use 0xA5 for consistency and to
      // catch a missed memcpy loudly).  Outputs honor their per-
      // buffer `init` mode (Sentinel / Zero).  A kernel that writes
      // via `tl.atomic_add` declares `init: "zero"` in the recipe
      // and the kernel sees a clean initial value; one that writes
      // deterministically declares the Sentinel default and any
      // missed write surfaces as 0xA5 in the diff.
      uint8_t fillByte = 0xA5;
      if (outIdx >= 0 &&
          t.outputs[outIdx].init == TritonOutputInit::Zero) {
        fillByte = 0x00;
      }
      HIP_ASSERT(hipMemset(dptr, fillByte, bytes));
      if (inIdx >= 0) {
        HIP_ASSERT(hipMemcpy(dptr,
                             inputBlob.data() + inputLayout.offsets[inIdx],
                             bytes, hipMemcpyHostToDevice));
      }
      devBufs[i] = {dptr, bytes, outIdx};

      std::memcpy(kargBuf.data() + slot.offset, &dptr, 8);
    } else {
      // Scalar sig arg.  Resolve in priority order:
      //   1. scalarArgsInt[name]   (integer expression)
      //   2. scalarArgsFloat[name] (float literal; only for fp* sig types)
      //   3. scope[name]           (legacy: arg name == constexpr/shape-dim
      //                             key, e.g. vecadd's `N`)
      // A scalar arg that resolves through none of these is a recipe bug;
      // surface loudly with the full lookup chain so the user knows what
      // to fix.
      bool isFloat = (sa.type == "fp16" || sa.type == "bf16" ||
                      sa.type == "fp32" || sa.type == "fp64");

      // Validate the metadata slot size matches what the user typed.
      // Catches "sig says i32, metadata slot is 8 bytes" — that would
      // either drop the upper word silently or step on the next slot.
      int expectedBytes = sigScalarBytes(sa.type);
      if (expectedBytes < 0)
        die("triton %s: scalar arg %s has unsupported sig type %s "
            "(extend sigScalarBytes when needed)",
            t.name.c_str(), sa.name.c_str(), sa.type.c_str());
      if (slot.size != expectedBytes)
        die("triton %s: scalar arg %s sig_type=%s expects %d bytes "
            "but metadata slot is %d bytes",
            t.name.c_str(), sa.name.c_str(), sa.type.c_str(),
            expectedBytes, slot.size);

      if (isFloat) {
        auto fit = t.scalarArgsFloat.find(sa.name);
        if (fit == t.scalarArgsFloat.end())
          die("triton %s: float scalar arg %s sig_type=%s has no value; "
              "add scalar_args[%s] = <numeric literal> to the recipe.",
              t.name.c_str(), sa.name.c_str(), sa.type.c_str(),
              sa.name.c_str());
        double dval = fit->second;
        if (sa.type == "fp64") {
          double v = dval;
          std::memcpy(kargBuf.data() + slot.offset, &v, 8);
        } else if (sa.type == "fp32") {
          float v = static_cast<float>(dval);
          std::memcpy(kargBuf.data() + slot.offset, &v, 4);
        } else if (sa.type == "fp16") {
          // Encode IEEE 754 binary16 from a finite double.  We only
          // need to handle the values a user will reasonably write
          // here (small positive eps, etc.); subnormals and ±inf are
          // out of scope.  Round-to-nearest-even via float32 hop.
          float f32 = static_cast<float>(dval);
          uint32_t u32;
          std::memcpy(&u32, &f32, 4);
          uint32_t sign = (u32 >> 31) & 0x1;
          int32_t  exp  = static_cast<int32_t>((u32 >> 23) & 0xFF) - 127;
          uint32_t mant = u32 & 0x7FFFFF;
          uint16_t h;
          if (exp > 15) {
            // Overflow → ±inf, surface so the user notices.
            die("triton %s: fp16 scalar arg %s value %g overflows fp16",
                t.name.c_str(), sa.name.c_str(), dval);
          } else if (exp < -14) {
            die("triton %s: fp16 scalar arg %s value %g is fp16-subnormal "
                "or zero; encode it through a non-subnormal path or "
                "extend the dispatch encoder",
                t.name.c_str(), sa.name.c_str(), dval);
          } else {
            uint32_t hexp = static_cast<uint32_t>(exp + 15);
            uint32_t hmant = mant >> 13;
            // Round-to-nearest-even on the dropped 13 bits.
            uint32_t low = mant & 0x1FFF;
            if (low > 0x1000 || (low == 0x1000 && (hmant & 1)))
              hmant++;
            if (hmant == 0x400) { hmant = 0; hexp++; }
            h = static_cast<uint16_t>((sign << 15) | (hexp << 10) | hmant);
          }
          std::memcpy(kargBuf.data() + slot.offset, &h, 2);
        } else if (sa.type == "bf16") {
          // bf16 = top 16 bits of fp32 with round-to-nearest-even.
          float f32 = static_cast<float>(dval);
          uint32_t u32;
          std::memcpy(&u32, &f32, 4);
          uint32_t rounding_bias = 0x7FFF + ((u32 >> 16) & 1);
          uint16_t b = static_cast<uint16_t>((u32 + rounding_bias) >> 16);
          std::memcpy(kargBuf.data() + slot.offset, &b, 2);
        }
        continue;
      }

      // Integer scalar.
      int64_t val;
      auto eit = t.scalarArgsInt.find(sa.name);
      if (eit != t.scalarArgsInt.end()) {
        val = evalCached(eit->second, scope);
      } else {
        auto sit = scope.find(sa.name);
        if (sit == scope.end())
          die("triton %s: integer scalar arg %s not resolvable.  Lookup "
              "chain tried: scalar_args[%s] (computed expression), "
              "scope[%s] (shape dim or constexpr by name).  Add the "
              "arg to scalar_args or rename the shape dim / constexpr "
              "to match.",
              t.name.c_str(), sa.name.c_str(),
              sa.name.c_str(), sa.name.c_str());
        val = sit->second;
      }

      if (slot.size == 1) {
        int8_t v = static_cast<int8_t>(val);
        std::memcpy(kargBuf.data() + slot.offset, &v, 1);
      } else if (slot.size == 2) {
        int16_t v = static_cast<int16_t>(val);
        std::memcpy(kargBuf.data() + slot.offset, &v, 2);
      } else if (slot.size == 4) {
        int32_t v = static_cast<int32_t>(val);
        std::memcpy(kargBuf.data() + slot.offset, &v, 4);
      } else if (slot.size == 8) {
        int64_t v = val;
        std::memcpy(kargBuf.data() + slot.offset, &v, 8);
      } else {
        die("triton %s: scalar arg %s has unsupported slot size %d",
            t.name.c_str(), sa.name.c_str(), slot.size);
      }
    }
  }

  // We need the launch grid extents for two reasons: the actual launch
  // below, and filling the hidden_block_count / hidden_group_size /
  // hidden_remainder / hidden_grid_dims implicit kernarg slots that
  // Triton lowerings read for tl.num_programs / tl.program_id / etc.
  int64_t gx = evalCached(t.grid.x, scope);
  int64_t gy = evalCached(t.grid.y, scope);
  int64_t gz = evalCached(t.grid.z, scope);
  int wgSize = M.maxFlatWorkgroupSize > 0 ? M.maxFlatWorkgroupSize : 256;
  // The launch we issue below uses (wgSize, 1, 1) for the workgroup
  // shape; if that ever changes, mirror the new shape here too or
  // tl.program_id will read stale group sizes.
  uint32_t bx = static_cast<uint32_t>(wgSize), by = 1, bz = 1;

  // Trailing implicit kernarg slots (i >= signature.size()).  Categorise
  // each slot:
  //   * Some have well-defined values the harness must fill (block
  //     counts, group sizes, remainders, grid_dims).  Filling them
  //     correctly is what makes any non-trivial Triton kernel — anything
  //     that calls tl.num_programs / tl.program_id under a non-default
  //     lowering — actually compute the right answer.
  //   * Some are NULL-safe (offsets, dynamic LDS, scratch base, etc.)
  //     and stay at the zero-initialised value in kargBuf.
  //   * Some require runtime infra we haven't built (hostcall, heap)
  //     and we refuse loudly.
  // Anything else surfaces a hard error so an unrecognised slot doesn't
  // get silently zeroed.
  auto fill32 = [&](const TritonArchMeta::Arg &slot, uint32_t v,
                    const char *what) {
    if (slot.size != 4)
      die("triton %s: implicit slot %s expected size=4, got %d",
          t.name.c_str(), what, slot.size);
    std::memcpy(kargBuf.data() + slot.offset, &v, 4);
  };
  auto fill16 = [&](const TritonArchMeta::Arg &slot, uint16_t v,
                    const char *what) {
    if (slot.size != 2)
      die("triton %s: implicit slot %s expected size=2, got %d",
          t.name.c_str(), what, slot.size);
    std::memcpy(kargBuf.data() + slot.offset, &v, 2);
  };
  for (size_t i = t.signature.size(); i < M.args.size(); ++i) {
    const auto &slot = M.args[i];
    const std::string &k = slot.valueKind;
    // Block counts: 32-bit each.  tl.num_programs(N) lowers to a load
    // from these.
    if (k == "hidden_block_count_x") { fill32(slot, static_cast<uint32_t>(gx), k.c_str()); continue; }
    if (k == "hidden_block_count_y") { fill32(slot, static_cast<uint32_t>(gy), k.c_str()); continue; }
    if (k == "hidden_block_count_z") { fill32(slot, static_cast<uint32_t>(gz), k.c_str()); continue; }
    // Workgroup sizes: 16-bit each.  Some Triton lowerings of
    // tl.program_id rebuild the program id from these.
    if (k == "hidden_group_size_x")  { fill16(slot, static_cast<uint16_t>(bx), k.c_str()); continue; }
    if (k == "hidden_group_size_y")  { fill16(slot, static_cast<uint16_t>(by), k.c_str()); continue; }
    if (k == "hidden_group_size_z")  { fill16(slot, static_cast<uint16_t>(bz), k.c_str()); continue; }
    // Remainders: 16-bit each.  For a uniform grid (block count is
    // fixed across the launch) the remainder is 0.  We don't expose
    // partial-workgroup launches, so 0 is correct here.
    if (k == "hidden_remainder_x" || k == "hidden_remainder_y" ||
        k == "hidden_remainder_z") { fill16(slot, 0, k.c_str()); continue; }
    // Number of *meaningful* grid dims (1, 2, or 3).  Pick the highest
    // dim with extent > 1.  Triton lowerings sometimes use this to
    // skip work on collapsed dims.
    if (k == "hidden_grid_dims") {
      uint16_t nd = (gz > 1) ? 3 : (gy > 1 ? 2 : 1);
      fill16(slot, nd, k.c_str());
      continue;
    }
    if (implicitUnsafeKinds().count(k)) {
      die("triton %s on %s: trailing implicit kernarg slot %zu has "
          "value_kind=%s, which the kernel will dereference at runtime "
          "(printf/malloc).  The harness leaves implicit slots at zero; "
          "this recipe is unsupported in Phase 1.",
          t.name.c_str(), gCurrentChildIsa.c_str(), i, k.c_str());
    }
    if (!implicitSafeKinds().count(k)) {
      die("triton %s on %s: trailing implicit kernarg slot %zu has "
          "unknown value_kind=%s.  The harness only allows a known-safe "
          "whitelist; investigate whether this slot needs a real value, "
          "and either extend implicitSafeKinds() or refuse to launch.",
          t.name.c_str(), gCurrentChildIsa.c_str(), i, k.c_str());
    }
  }

  // Launch.  wgSize was already computed above so the implicit-slot
  // filler could pass it into hidden_group_size_*.
  size_t kSize = kargBuf.size();
  void *cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargBuf.data(),
                 HIP_LAUNCH_PARAM_BUFFER_SIZE,   &kSize,
                 HIP_LAUNCH_PARAM_END};
  // Static LDS (group_segment_fixed_size) is allocated by HIP from the
  // kernel descriptor automatically; it is 0 for every Triton kernel we
  // emit because Triton's AMD backend always routes shared memory
  // through the *dynamic* LDS path instead.  The 7th
  // hipModuleLaunchKernel arg is that dynamic LDS size, and Triton's
  // own launcher passes `metadata.shared` there (see `pack_metadata` in
  // triton/backends/amd/compiler.py).
  //
  // We propagate the same value through the sidecar as
  // `shared_mem_bytes`; see TritonArchMeta::sharedMemBytes for the full
  // rationale.  Before this plumbing existed, passing 0 here caused
  // layer-norm and softmax to silently produce zero output — the
  // reduction path loads/stores to offset 0 inside the (non-existent)
  // dynamic LDS allocation and the accumulator never sees any data.
  // Elementwise kernels (add, asin) naturally have shared_mem_bytes=0
  // and are unaffected.
  HIP_ASSERT(hipModuleLaunchKernel(fn,
                                   static_cast<unsigned>(gx),
                                   static_cast<unsigned>(gy),
                                   static_cast<unsigned>(gz),
                                   static_cast<unsigned>(wgSize), 1, 1,
                                   static_cast<unsigned>(M.sharedMemBytes),
                                   nullptr, nullptr, cfg));
  HIP_ASSERT(hipDeviceSynchronize());

  // Read outputs back into a flat blob, in output-spec order.
  size_t outBytes = tritonOutputBytes(t, shapeValue);
  std::vector<uint8_t> outBlob(outBytes, 0);
  size_t outOff = 0;
  for (size_t oi = 0; oi < t.outputs.size(); ++oi) {
    const auto &b = t.outputs[oi];
    int64_t n = evalCached(b.elems, scope);
    size_t bytes = static_cast<size_t>(n) * dtypeBytes(b.dtype);
    // Find the device buffer we allocated for this output.
    void *dptr = nullptr;
    for (const auto &db : devBufs)
      if (db.outputIndex == (int)oi) { dptr = db.ptr; break; }
    if (!dptr)
      die("triton %s: no device buffer for output %s",
          t.name.c_str(), b.name.c_str());
    HIP_ASSERT(hipMemcpy(outBlob.data() + outOff, dptr, bytes,
                         hipMemcpyDeviceToHost));
    outOff += bytes;
  }

  // guard's destructor frees every device buffer.
  return outBlob;
}

// Decode IEEE 754 binary16 → binary32, full-precision (no flush-to-zero).
// Subnormals are normalised to f32 normals; NaNs propagate (mantissa
// preserved); ±inf preserved.  Hoisted out of tritonCompare so the bf16
// path can sit next to it without duplicating the structure.
inline double half2double(uint16_t h) {
  uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
  uint32_t exp  = (h & 0x7c00u) >> 10;
  uint32_t mant = (h & 0x03ffu);
  uint32_t u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;  // ±0
    } else {
      // Subnormal half (value = mant * 2^-24).  Renormalise into f32 by
      // left-shifting until the implicit bit appears, then assemble with
      // the appropriate biased exponent.
      int e = -1;
      do { ++e; mant <<= 1; } while ((mant & 0x400) == 0);
      mant &= 0x3ff;
      u = sign | static_cast<uint32_t>((127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    // inf or NaN — preserve sign and mantissa (mant != 0 ⇒ NaN).
    u = sign | 0x7f800000u | (mant << 13);
  } else {
    u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f; std::memcpy(&f, &u, 4); return f;
}

// Decode bf16 → fp32: bf16 is the top 16 bits of fp32, so reinflating is
// just `(bf << 16)` and a memcpy.  No subnormal renormalisation is needed
// because bf16 and fp32 share an exponent range.
inline double bf162double(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f; std::memcpy(&f, &u, 4);
  return f;
}

// Comparator for Triton recipes: walk each declared output, decode its
// slice with the right dtype, and judge it with the right comparator
// (per-output override if set, else the recipe-level default).
//
// `shapeValue` is the resolved scalar shape (e.g. 1024 for vecadd N=1024).
// We re-derive the per-output byte slices from shape + constexprs so each
// output is judged on its own elements, even when outputs differ in dtype
// or in element count (no longer pinned to "same dtype, same elems").
//
// NaN handling: NaN-vs-NaN is treated as a match (both paths agree on "no
// defined value here").  NaN-vs-finite is always a mismatch.  Both ±inf
// with matching sign agree (kernels routinely saturate to ±inf on
// overflow; agreeing on the sign is the desired invariant).
//
// Subnormal handling: fp16 denormals are decoded properly (no flush-to-
// zero) so a kernel that produces denormals on one path and ±0 on the
// other is reported as a mismatch instead of a silent agreement.
//
// Comparator kinds:
//   abs      — |g - a| <= tol, per element
//   rel      — |g - a| / max(|g|, 1e-6) <= tol, per element
//   rel-rms  — sqrt(mean((g-a)^2)) / max(sqrt(mean(g^2)), 1e-6) <= tol,
//              one verdict per output buffer.  Robust to per-element
//              reordering (typical of cross-wave-size reductions).  When
//              the verdict is "bad", the mismatch counter is set to the
//              full element count of the buffer and firstIdx points at
//              the worst element so the report shows where to look.
std::tuple<int, double, int, double, double>
tritonCompare(const TritonRecipe &t,
              const std::vector<uint8_t> &gold,
              const std::vector<uint8_t> &actual, int shapeValue) {
  if (gold.size() != actual.size()) {
    int total = static_cast<int>(gold.size() / 1);  // bytes; coarse but loud
    return std::make_tuple(total, 1.0, 0,
                           static_cast<double>(gold.size()),
                           static_cast<double>(actual.size()));
  }
  auto scope = tritonScope(t, shapeValue);

  int    totalMismatches = 0;
  double maxAbs          = 0.0;
  int    firstIdx        = -1;     // global element index across all outputs
  double firstG          = 0.0;
  double firstA          = 0.0;
  size_t globalOffsetEl  = 0;      // running element index across outputs
  size_t globalOffsetByt = 0;      // running byte offset across outputs

  for (const auto &out : t.outputs) {
    int64_t n = evalCached(out.elems, scope);
    int es = dtypeBytes(out.dtype);
    size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(es);
    if (globalOffsetByt + bytes > gold.size())
      die("triton %s: output %s wants %zu bytes but only %zu remain in the "
          "blob (recipe and dispatch disagree on output sizing)",
          t.name.c_str(), out.name.c_str(), bytes,
          gold.size() - globalOffsetByt);
    const auto &cmp = out.hasComparator ? out.comparator : t.comparator;

    // Per-output running stats; merged into the global stats below.
    int    bufMismatches = 0;
    double bufMaxAbs     = 0.0;
    int    bufFirstIdx   = -1;
    double bufFirstG     = 0.0;
    double bufFirstA     = 0.0;
    // For rel-rms we accumulate sums of squares; per-element judging is
    // skipped and a single verdict is assembled at the end of the buffer.
    double sumDiff2 = 0.0;
    double sumGold2 = 0.0;
    int    rmsWorstIdx = -1;
    double rmsWorstD   = 0.0;
    double rmsWorstG   = 0.0;
    double rmsWorstA   = 0.0;

    auto judge = [&](double gv, double av, int i) {
      bool gNaN = std::isnan(gv);
      bool aNaN = std::isnan(av);
      if (gNaN && aNaN) return;  // both NaN → agree
      bool gInf = std::isinf(gv);
      bool aInf = std::isinf(av);
      if (gInf && aInf && std::signbit(gv) == std::signbit(av)) return;
      double d;
      if (gNaN || aNaN || (gInf != aInf) ||
          (gInf && aInf /* opposite signs */)) {
        d = std::numeric_limits<double>::infinity();
      } else {
        d = std::fabs(gv - av);
      }
      if (std::isfinite(d) && d > bufMaxAbs) bufMaxAbs = d;

      if (cmp.kind == "rel-rms") {
        // Track the worst single-element diff for reporting; the verdict
        // itself is computed once at end-of-buffer.
        double gv2 = std::isfinite(gv) ? gv * gv : 0.0;
        sumGold2 += gv2;
        if (std::isfinite(d)) {
          sumDiff2 += d * d;
          if (d > rmsWorstD) {
            rmsWorstD = d; rmsWorstIdx = i; rmsWorstG = gv; rmsWorstA = av;
          }
        } else {
          // Treat inf-vs-finite or NaN-vs-finite as catastrophic for the
          // RMS — bias the squared sum to definitely fail the threshold.
          sumDiff2 = std::numeric_limits<double>::infinity();
          if (rmsWorstIdx < 0) {
            rmsWorstIdx = i; rmsWorstG = gv; rmsWorstA = av;
            rmsWorstD = std::numeric_limits<double>::infinity();
          }
        }
        return;
      }

      bool bad;
      if (cmp.kind == "abs") {
        bad = !(d <= cmp.tol);
      } else if (cmp.kind == "rel") {
        double denom = std::max(std::fabs(gv), 1e-6);
        bad = !(d / denom <= cmp.tol);
      } else {
        die("triton %s output %s: unsupported comparator kind %s",
            t.name.c_str(), out.name.c_str(), cmp.kind.c_str());
      }
      if (bad) {
        if (bufMismatches++ == 0) {
          bufFirstIdx = i; bufFirstG = gv; bufFirstA = av;
        }
      }
    };

    const uint8_t *gp = gold.data()   + globalOffsetByt;
    const uint8_t *ap = actual.data() + globalOffsetByt;
    int ne = static_cast<int>(n);

    if (out.dtype == "fp16") {
      const auto *g = reinterpret_cast<const uint16_t *>(gp);
      const auto *a = reinterpret_cast<const uint16_t *>(ap);
      for (int i = 0; i < ne; ++i)
        judge(half2double(g[i]), half2double(a[i]), i);
    } else if (out.dtype == "bf16") {
      const auto *g = reinterpret_cast<const uint16_t *>(gp);
      const auto *a = reinterpret_cast<const uint16_t *>(ap);
      for (int i = 0; i < ne; ++i)
        judge(bf162double(g[i]), bf162double(a[i]), i);
    } else if (out.dtype == "fp32") {
      const auto *g = reinterpret_cast<const float *>(gp);
      const auto *a = reinterpret_cast<const float *>(ap);
      for (int i = 0; i < ne; ++i) judge(g[i], a[i], i);
    } else if (out.dtype == "fp64") {
      const auto *g = reinterpret_cast<const double *>(gp);
      const auto *a = reinterpret_cast<const double *>(ap);
      for (int i = 0; i < ne; ++i) judge(g[i], a[i], i);
    } else if (out.dtype == "i32" || out.dtype == "u32") {
      // Compare as uint32 bit-pattern regardless of i32/u32 sig — the
      // comparator is bit-exact and two's complement makes the signed
      // interpretation equivalent for equality tests.
      const auto *g = reinterpret_cast<const uint32_t *>(gp);
      const auto *a = reinterpret_cast<const uint32_t *>(ap);
      for (int i = 0; i < ne; ++i)
        if (g[i] != a[i] && bufMismatches++ == 0) {
          bufFirstIdx = i;
          bufFirstG = static_cast<double>(g[i]);
          bufFirstA = static_cast<double>(a[i]);
          if (bufMaxAbs < 1.0) bufMaxAbs = 1.0;
        }
    } else if (out.dtype == "i64") {
      const auto *g = reinterpret_cast<const int64_t *>(gp);
      const auto *a = reinterpret_cast<const int64_t *>(ap);
      for (int i = 0; i < ne; ++i)
        if (g[i] != a[i] && bufMismatches++ == 0) {
          bufFirstIdx = i;
          bufFirstG = static_cast<double>(g[i]);
          bufFirstA = static_cast<double>(a[i]);
          if (bufMaxAbs < 1.0) bufMaxAbs = 1.0;
        }
    } else {
      die("triton %s output %s: compare doesn't handle dtype=%s",
          t.name.c_str(), out.name.c_str(), out.dtype.c_str());
    }

    // Per-buffer rel-rms verdict.
    if (cmp.kind == "rel-rms") {
      // Numerator: RMS of the diff.  Denominator: RMS of the gold,
      // floored at 1e-6 so a near-zero gold doesn't divide by ~0 and
      // turn every tiny diff into "infinitely wrong".
      double rmsDiff = std::sqrt(sumDiff2 / std::max<double>(1, ne));
      double rmsGold = std::sqrt(sumGold2 / std::max<double>(1, ne));
      double denom = std::max(rmsGold, 1e-6);
      double ratio = rmsDiff / denom;
      bool bad = !(ratio <= cmp.tol);
      if (bad) {
        // Mark the whole buffer as failing; the worst element drives the
        // diagnostic columns so the user knows where to dig.
        bufMismatches = ne;
        bufMaxAbs     = std::max(bufMaxAbs, rmsWorstD);
        bufFirstIdx   = rmsWorstIdx >= 0 ? rmsWorstIdx : 0;
        bufFirstG     = rmsWorstG;
        bufFirstA     = rmsWorstA;
      }
    }

    // Merge per-buffer stats into global; first mismatch wins (no
    // overwrite from a later, larger-index buffer).
    if (bufMaxAbs > maxAbs) maxAbs = bufMaxAbs;
    if (totalMismatches == 0 && bufMismatches > 0) {
      firstIdx = static_cast<int>(globalOffsetEl) + bufFirstIdx;
      firstG = bufFirstG;
      firstA = bufFirstA;
    }
    totalMismatches += bufMismatches;

    globalOffsetEl  += static_cast<size_t>(ne);
    globalOffsetByt += bytes;
  }
  return std::make_tuple(totalMismatches, maxAbs, firstIdx, firstG, firstA);
}

// Convert a TritonRecipe into a Recipe the existing harness understands.
// The shape index is encoded in `N` (position into t.defaultShapeValues),
// with blockSize always 0.  When Phase 2 introduces multi-dim Triton shapes
// (matmul's (M, N, K)), the same indexing scheme generalises — N stays the
// sweep index and the report grows a shape-label column rather than trying
// to pretty-print the dict in the existing N/block columns.
Recipe tritonToRecipe(const TritonRecipe &t) {
  const TritonRecipe *tp = &t;  // stable pointer: allTritonRecipes() is a static
  Recipe r;
  r.name = t.name;
  r.goldSource = GoldSource::NativeExecution;
  r.defaultNs.reserve(t.defaultShapeValues.size());
  for (size_t i = 0; i < t.defaultShapeValues.size(); ++i)
    r.defaultNs.push_back(static_cast<int>(i));
  r.defaultBlocks = {0};
  r.outputElemBytes = t.outputs.empty() ? 1
                                         : dtypeBytes(t.outputs.front().dtype);
  r.outputElems = [tp](int N, int) -> int {
    int v = tp->defaultShapeValues.at(static_cast<size_t>(N));
    // Total element count across the output blob, in units of the first
    // output's dtype.  Used for display only (see report grid).
    return static_cast<int>(tritonOutputBytes(*tp, v) /
                            dtypeBytes(tp->outputs.front().dtype));
  };
  r.makeInput = [tp](int N) {
    int v = tp->defaultShapeValues.at(static_cast<size_t>(N));
    return tritonMakeInput(*tp, v);
  };
  r.dispatch = [tp](hipModule_t mod, const std::vector<uint8_t> &input,
                    int N, int) {
    int v = tp->defaultShapeValues.at(static_cast<size_t>(N));
    return tritonDispatch(*tp, mod, input, v);
  };
  r.compare = [tp](const std::vector<uint8_t> &gold,
                   const std::vector<uint8_t> &actual,
                   int N, int /*blockSize*/, int /*outElems*/) {
    int v = tp->defaultShapeValues.at(static_cast<size_t>(N));
    return tritonCompare(*tp, gold, actual, v);
  };
  // NativeExecution gold source => cpuReference is never invoked.  Leave it
  // empty; runOne's branch skips the call.
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recipe registry
// ─────────────────────────────────────────────────────────────────────────────

const std::vector<Recipe> &allRecipes() {
  static const std::vector<Recipe> v = []() {
    std::vector<Recipe> r = {
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
        makeVBfiB32Recipe(),
        makeVCmpCndmaskSgprRecipe(),
        makeVCmpxGtI32Recipe(),
        makeSAndSaveexecB32Recipe(),
        makeC4LaneDepCmpxRecipe(),
        makeWmmaF32_16x16x4_GemmRecipe(),
        makeMubufStoreB32Recipe(),
        makeSSetVgprMsbRecipe(),
        makeCanaryReadlaneLastLaneRecipe(),
        makeCanaryDsSwizzleSwap1Recipe(),
        makeCanaryDsSwizzleQuadPermRecipe(),
        makeCanaryCvtScalePk8Bf16Fp4Recipe(),
    };
    for (const auto &t : allTritonRecipes())
      r.push_back(tritonToRecipe(t));
    return r;
  }();
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
  // Make the isa visible to Triton dispatches (HIP dispatches ignore it).
  // The scope clears it on function exit so any future code that loops
  // over multiple recipes per child works without each iteration
  // remembering to reset the global.
  ChildIsaScope isaScope(isa);
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
  // Per-child timeout.  Without it, a single hanging child (most often a
  // salmon transpilation that never completes, or a transpiled kernel
  // that hangs the GPU) blocks the entire run.  We need to interleave
  // pipe draining with child-status polling so neither blocks the
  // other: a pure read() blocks until EOF (which never comes if the
  // child hangs), and a pure waitpid() hides any stderr the child
  // already produced.  poll() on the pipe with a small timeout, plus
  // non-blocking waitpid in the same loop, gives us both.
  long timeoutSec = 60;
  if (const char *t = std::getenv("COMPARE_CORRECTNESS_CHILD_TIMEOUT_S")) {
    char *end = nullptr;
    long v = std::strtol(t, &end, 10);
    if (end != t) timeoutSec = v;
  }
  std::string err;
  err.reserve(4096);
  const size_t maxStderr = 32 * 1024;
  bool stderrEof = false;
  bool timedOut = false;
  int wstatus = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(timeoutSec);
  for (;;) {
    // Drain any available stderr without blocking.
    if (!stderrEof) {
      pollfd pfd{errPipe[0], POLLIN, 0};
      int pr = poll(&pfd, 1, /*timeout_ms=*/100);
      if (pr > 0 && (pfd.revents & (POLLIN | POLLHUP))) {
        char buf[4096];
        ssize_t n = read(errPipe[0], buf, sizeof(buf));
        if (n > 0) {
          if (err.size() < maxStderr) {
            size_t take = std::min(static_cast<size_t>(n),
                                   maxStderr - err.size());
            err.append(buf, take);
          }
        } else if (n == 0) {
          stderrEof = true;
        } else if (errno != EINTR && errno != EAGAIN) {
          stderrEof = true;
        }
      }
    }
    // Non-blocking child-status check.
    pid_t w = waitpid(pid, &wstatus, WNOHANG);
    if (w == pid) break;
    if (w < 0 && errno != EINTR) {
      cr.stderrTail = std::string("waitpid: ") + std::strerror(errno);
      close(errPipe[0]);
      return cr;
    }
    if (timeoutSec > 0 && std::chrono::steady_clock::now() >= deadline) {
      // SIGKILL — a hung HIP child usually doesn't process SIGTERM
      // because it's stuck inside a driver ioctl.  Reap synchronously.
      kill(pid, SIGKILL);
      timedOut = true;
      for (;;) {
        pid_t w2 = waitpid(pid, &wstatus, 0);
        if (w2 == pid) break;
        if (w2 < 0 && errno == EINTR) continue;
        cr.stderrTail = std::string("waitpid after SIGKILL: ") +
                        std::strerror(errno);
        close(errPipe[0]);
        return cr;
      }
      break;
    }
  }
  // Final drain after the child exits — anything still buffered in the
  // pipe is now flushed; without this we'd lose the tail of the message
  // any child that printed-then-exited produced.
  if (!stderrEof) {
    char buf[4096];
    while (err.size() < maxStderr) {
      ssize_t n = read(errPipe[0], buf, sizeof(buf));
      if (n <= 0) break;
      size_t take = std::min(static_cast<size_t>(n), maxStderr - err.size());
      err.append(buf, take);
    }
  }
  close(errPipe[0]);
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
  if (timedOut) {
    // Append (don't prepend) so the timeout note survives any later
    // tail() trimming the reporter does — the captured err is often
    // dominated by hotswap chatter and would otherwise crowd this out.
    std::string note = "  [TIMEOUT after " + std::to_string(timeoutSec) +
                       "s; harness sent SIGKILL.  Override with "
                       "COMPARE_CORRECTNESS_CHILD_TIMEOUT_S=<seconds>.]";
    cr.stderrTail = cr.stderrTail.empty() ? note
                                          : (cr.stderrTail + "\n" + note);
  }
  return cr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent side: run one (recipe, shape) against all three modes
// ─────────────────────────────────────────────────────────────────────────────

struct ModeResult {
  ChildRun child;
  // Present only if the child wrote a readable output file.
  std::optional<std::vector<uint8_t>> output;
  // Diff stats vs. the recipe's gold.  For HIP recipes the gold is the
  // CPU reference; for Triton recipes it's the native gfx942 child's
  // output (see Recipe::goldSource).
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
  // Mirrors Recipe::goldSource at the time of runOne().  Captured here so
  // the reporter can choose the right rendering without chasing the Recipe.
  GoldSource goldSource = GoldSource::CpuReference;
  // True iff goldSource == NativeExecution and the native child failed to
  // produce usable output.  legacy / salmon are skipped in that case; their
  // ModeResult stays launched=false so the reporter knows not to judge them.
  bool goldMissing = false;
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
  rr.goldSource = r.goldSource;

  auto input = r.makeInput(N);
  std::string inPath  = tempPath("in");
  writeFile(inPath, input);

  // Compute the CPU reference up front if we have one.  Under
  // NativeExecution we defer gold generation until the native child runs.
  if (r.goldSource == GoldSource::CpuReference) {
    if (!r.cpuReference)
      die("recipe %s: goldSource=CpuReference but cpuReference is not set",
          r.name.c_str());
    rr.cpuGold = r.cpuReference(input, N, blockSize);
  }

  auto runMode = [&](Mode m, ModeResult &mr, const std::string &tag,
                     bool skipCompare) {
    std::string outPath = tempPath(tag);
    unlink(outPath.c_str());
    mr.child = spawnChild(exe, m, r.name, N, blockSize, inPath, outPath);
    if (mr.child.exitedCleanly && mr.child.exitCode == 0) {
      struct stat st;
      if (stat(outPath.c_str(), &st) == 0 && st.st_size > 0) {
        mr.output = readFile(outPath);
      }
    }
    if (mr.output && !skipCompare) {
      auto [mm, mab, idx, g, a] =
          r.compare(rr.cpuGold, *mr.output, N, blockSize, rr.outElems);
      mr.mismatches = mm;
      mr.maxAbsErr = mab;
      mr.firstIdx = idx;
      mr.firstGold = g;
      mr.firstActual = a;
    }
    // Optional preservation of per-mode output files for offline
    // triage.  Set `HSA_SALMON_DUMP_DIFF=<dir>` to have each mode's
    // output copied to `<dir>/<recipe>_N<N>_B<B>_<mode>.bin` before
    // the harness tempfile is unlinked.  Intended for cases where the
    // one-line `@idx=<I> ref=<X> actual=<Y> max|err|=<E>` summary
    // doesn't disambiguate the miscompile pattern (per-lane /
    // per-element / per-wave-half mismatches all compress to the same
    // summary), and per-index numpy diffing is the faster path to a
    // reproducer.  Cheap and side-effect-free when the env var is
    // unset; survives across multiple recipes so a single run can
    // populate `<dir>/` with a full gallery of diffs.
    if (mr.output) {
      const char *dd = std::getenv("HSA_SALMON_DUMP_DIFF");
      if (dd && dd[0]) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/%s_N%d_B%d_%s.bin", dd,
                      r.name.c_str(), N, blockSize, tag.c_str());
        writeFile(buf, *mr.output);
      }
    }
    unlink(outPath.c_str());
  };

  if (r.goldSource == GoldSource::NativeExecution) {
    // Run native first; its output IS the gold.  No compare for native
    // itself (by definition it matches itself).  If native didn't produce
    // output we can't judge legacy/salmon, so we flag the row as
    // gold-missing and skip their spawns.
    runMode(Mode::Native, rr.native, "native", /*skipCompare=*/true);
    if (rr.native.output) {
      rr.cpuGold = *rr.native.output;
      rr.native.mismatches = 0;  // native IS the gold
      runMode(Mode::Legacy, rr.legacy, "legacy", /*skipCompare=*/false);
      runMode(Mode::Salmon, rr.salmon, "salmon", /*skipCompare=*/false);
    } else {
      rr.goldMissing = true;
    }
  } else {
    runMode(Mode::Native, rr.native, "native", /*skipCompare=*/false);
    runMode(Mode::Legacy, rr.legacy, "legacy", /*skipCompare=*/false);
    runMode(Mode::Salmon, rr.salmon, "salmon", /*skipCompare=*/false);
  }

  unlink(inPath.c_str());
  return rr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reporting
// ─────────────────────────────────────────────────────────────────────────────

// Status cell for a non-native mode, or for native under CpuReference.  Used
// whenever we actually did (or tried to) compare the child's output against a
// pre-existing gold.
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

// Status cell for the native column under NativeExecution gold.  If native
// produced output we show `gold` (that output is what we're judging the
// other modes against).  If it didn't, we fall back to the usual failure
// string.
std::string statusStrNativeAsGold(const ModeResult &mr) {
  if (!mr.child.launched) return "spawn-fail";
  if (mr.child.signal > 0) {
    char b[64];
    std::snprintf(b, sizeof(b), "SIG%d", mr.child.signal);
    return b;
  }
  if (!mr.child.exitedCleanly) return "no-exit";
  if (mr.child.exitCode != 0)  return "EXIT=" + std::to_string(mr.child.exitCode);
  if (!mr.output)              return "no-output";
  return "gold";
}

// Status cell for legacy / salmon when native-as-gold failed: we never even
// spawned them in that case, so there's no child to report.
std::string statusStrNoGold() { return "no-gold"; }

// Classify a mode result for summary counts.
enum class ResultCat { Match, Mismatch, Crash, Gold, GoldMissing };

ResultCat classify(const ModeResult &m) {
  if (!m.child.launched)                             return ResultCat::Crash;
  if (m.child.signal > 0)                            return ResultCat::Crash;
  if (!m.child.exitedCleanly)                        return ResultCat::Crash;
  if (m.child.exitCode != 0)                         return ResultCat::Crash;
  if (!m.output)                                     return ResultCat::Crash;
  if (m.mismatches == 0)                             return ResultCat::Match;
  return ResultCat::Mismatch;
}

ResultCat classifyNativeAsGold(const ModeResult &m) {
  if (!m.child.launched)       return ResultCat::GoldMissing;
  if (m.child.signal > 0)      return ResultCat::GoldMissing;
  if (!m.child.exitedCleanly)  return ResultCat::GoldMissing;
  if (m.child.exitCode != 0)   return ResultCat::GoldMissing;
  if (!m.output)               return ResultCat::GoldMissing;
  return ResultCat::Gold;
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

// For Triton recipes the shape column shows the recipe's shape-dim name and
// value instead of the HIP-style "N=X B=Y" pair.  We compute it per
// RunResult and reuse it in both the grid and the Failures section.
std::string shapeLabelFor(const RunResult &rr) {
  if (rr.goldSource == GoldSource::NativeExecution) {
    // Triton path: N is the sweep index; look up the actual shape value
    // through the recipe's TritonRecipe twin.  We find it by name.
    for (const auto &t : allTritonRecipes()) {
      if (t.name == rr.recipe->name) {
        if (rr.N < 0 || static_cast<size_t>(rr.N) >= t.defaultShapeValues.size())
          return "?";
        char b[64];
        std::snprintf(b, sizeof(b), "%s=%d",
                      t.shapeDimName.c_str(),
                      t.defaultShapeValues[static_cast<size_t>(rr.N)]);
        return b;
      }
    }
    return "?";
  }
  char b[64];
  std::snprintf(b, sizeof(b), "N=%d B=%d", rr.N, rr.blockSize);
  return b;
}

void printReport(const std::vector<RunResult> &all) {
  // ── Grid: one atomic row per (recipe, shape). No interleaved lines. ──
  const int W_RECIPE = 16;
  const int W_SHAPE  = 14;   // fits "M=128 N=4096" and "N=65536"
  const int W_STATUS = 18;   // fits "WRONG 65536/65536"

  printf("\n");
  printf("%-*s %-*s  %-*s  %-*s  %-*s\n",
         W_RECIPE, "recipe", W_SHAPE, "shape",
         W_STATUS, "native", W_STATUS, "legacy", W_STATUS, "salmon");
  int sepW = W_RECIPE + W_SHAPE + 3 * W_STATUS + 8;
  printf("%s\n", std::string(sepW, '-').c_str());

  // Counters for the summary matrix: cnt[mode][category].  mode: 0=native,
  // 1=legacy, 2=salmon.  We track all categories even though some only
  // apply to certain gold sources.
  const int NUM_CATS = 5;
  int cnt[3][NUM_CATS] = {{0}, {0}, {0}};

  // Failures grouped per mode, in input order.
  struct FailRow {
    std::string recipeName;
    std::string shape;
    std::string detail;
  };
  std::vector<FailRow> fails[3];

  for (const auto &rr : all) {
    const ModeResult *mm[3] = {&rr.native, &rr.legacy, &rr.salmon};
    std::string cells[3];
    bool nativeAsGold = (rr.goldSource == GoldSource::NativeExecution);
    std::string shapeLabel = shapeLabelFor(rr);
    for (int i = 0; i < 3; ++i) {
      if (nativeAsGold && rr.goldMissing && i > 0) {
        // legacy / salmon were never spawned.
        cells[i] = statusStrNoGold();
        ++cnt[i][static_cast<int>(ResultCat::GoldMissing)];
        fails[i].push_back({rr.recipe->name, shapeLabel,
                            "no-gold (native failed to produce output)"});
        continue;
      }
      ResultCat c;
      if (nativeAsGold && i == 0) {
        cells[i] = statusStrNativeAsGold(*mm[i]);
        c = classifyNativeAsGold(*mm[i]);
      } else {
        cells[i] = statusStr(*mm[i], rr.outElems);
        c = classify(*mm[i]);
      }
      ++cnt[i][static_cast<int>(c)];
      bool ok = (c == ResultCat::Match) || (c == ResultCat::Gold);
      if (!ok) {
        std::string detail;
        if (c == ResultCat::GoldMissing) {
          // Two flavours of GoldMissing:
          //   - i == 0  (native column): the gold itself failed; surface
          //     the actual child stderr so the user can see WHY native
          //     died, not just that it did.
          //   - i  > 0  (legacy / salmon): we never spawned this child
          //     because there was no gold; reflect that.
          if (i == 0) detail = failureDetail(*mm[i], rr.outElems);
          else        detail = "gold-missing (native failed to produce output)";
        } else {
          detail = failureDetail(*mm[i], rr.outElems);
        }
        fails[i].push_back({rr.recipe->name, shapeLabel, detail});
      }
    }
    printf("%-*s %-*s  %-*s  %-*s  %-*s\n",
           W_RECIPE, rr.recipe->name.c_str(),
           W_SHAPE, shapeLabel.c_str(),
           W_STATUS, cells[0].c_str(),
           W_STATUS, cells[1].c_str(),
           W_STATUS, cells[2].c_str());
  }

  // ── Failures section: full detail, grouped by mode. ──
  const char *modeLabel[3] = {"native", "legacy", "salmon"};
  const char *modeNote[3]  = {
      ("  (native != reference — harness, CPU-reference, or hipcc bug; "
       "suspect other rows)"),
      "", ""};
  bool anyFail = false;
  for (int i = 0; i < 3; ++i) if (!fails[i].empty()) { anyFail = true; break; }
  if (anyFail) {
    printf("\n=== Failures ===\n");
    for (int i = 0; i < 3; ++i) {
      if (fails[i].empty()) continue;
      printf("\n%s%s\n", modeLabel[i], modeNote[i]);
      for (const auto &f : fails[i]) {
        printf("  %-*s %-*s  %s\n",
               W_RECIPE, f.recipeName.c_str(),
               W_SHAPE, f.shape.c_str(),
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
  // Only print the native-as-gold rows when there's something to show —
  // otherwise they're just noise for HIP-only runs.
  if (cnt[0][static_cast<int>(ResultCat::Gold)] +
      cnt[1][static_cast<int>(ResultCat::Gold)] +
      cnt[2][static_cast<int>(ResultCat::Gold)] +
      cnt[0][static_cast<int>(ResultCat::GoldMissing)] +
      cnt[1][static_cast<int>(ResultCat::GoldMissing)] +
      cnt[2][static_cast<int>(ResultCat::GoldMissing)] > 0) {
    row("gold",          ResultCat::Gold);
    row("gold-missing",  ResultCat::GoldMissing);
  }
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
      "  --shape=<N>       restrict shape VALUES to the listed Ns (repeatable;\n"
      "                    cross-product with --block).  For non-Triton recipes\n"
      "                    N is the raw shape integer consumed by the kernel.\n"
      "                    For Triton (native-as-gold) recipes N is matched\n"
      "                    against the recipe's declared `defaultShapeValues`;\n"
      "                    e.g. `--shape=1024` picks the N=1024 point on a\n"
      "                    recipe that declares a 1024-column row (recipe\n"
      "                    sweep index mapping is resolved internally).  If a\n"
      "                    recipe does not declare the requested value the\n"
      "                    recipe contributes no runs; the harness then\n"
      "                    surfaces `no runs matched` if all recipes drop to\n"
      "                    empty.  If omitted, every recipe uses its own\n"
      "                    default shape list.\n"
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
  // Precompute a fast name -> TritonRecipe* map so the shape-filter
  // translation below is O(1) per Triton recipe (and avoids another
  // linear scan of `allTritonRecipes()` nested inside the cross-product
  // loop below).
  std::unordered_map<std::string, const TritonRecipe *> tritonByName;
  for (const auto &t : allTritonRecipes()) tritonByName[t.name] = &t;

  for (const auto &r : allRecipes()) {
    if (!opt.recipeFilter.empty() && r.name != opt.recipeFilter) continue;
    // Resolve `--shape=<X>` to the recipe's internal N representation.
    //
    // Historically this harness plumbed two independent "shape"
    // notions through the same `int N` slot on Recipe::dispatch etc.:
    //
    //   * Non-Triton recipes (vecadd_f32, block_sum, … — all
    //     `GoldSource::CpuReference`): `N` IS the shape value
    //     passed straight through to the kernel (and equal to the
    //     integer stored in `defaultNs`).
    //   * Triton recipes (`GoldSource::NativeExecution`): `N` is an
    //     *index* into `TritonRecipe::defaultShapeValues`, and
    //     `tritonToRecipe` explicitly populates `defaultNs` with
    //     `[0, 1, ..., defaultShapeValues.size()-1]`.  Every
    //     Triton-side closure (`dispatch`, `makeInput`, `compare`,
    //     `outputElems`) turns the N back into the value via
    //     `defaultShapeValues.at(N)`.
    //
    // That dual meaning leaked through the CLI: typing
    // `compare_correctness --shape=128` was ambiguous — a literal
    // value on non-Triton recipes, but interpreted as an index on
    // Triton recipes (and blew up with
    // `vector::_M_range_check: __n (which is 128) >= this->size()
    // (which is 4)` when the Triton recipe's defaults had fewer
    // than 128 entries).  There is no realistic user who wants
    // "index 128 into whatever shape list"; the shape the user
    // cares about IS the value.
    //
    // Fix: `--shape=<X>` is uniformly the shape *value*.  For
    // Triton recipes we translate each filter value to the matching
    // position in `defaultShapeValues` here, so the downstream
    // closures keep seeing the index they already expect.  Values
    // that no current recipe supports silently drop out of the
    // per-recipe `effectiveNs` — the harness-wide "no runs matched
    // your filter" gate below still fires if every recipe drops
    // everything.  No silent fallback to a different value; if a
    // Triton recipe's `defaultShapeValues` is `{256, 512, 1024}`
    // and the user asks for `--shape=128`, that recipe contributes
    // zero runs rather than rounding up / down to a neighbour.
    //
    // Backwards-compat note: the original index-based semantics on
    // Triton recipes is not preserved under a separate flag.  Any
    // user who genuinely wanted "index N" was relying on an
    // implementation detail of `tritonToRecipe` that is expressly
    // internal (the defaultShapeValues ordering itself is not
    // promised to be stable across refactors); replacing an
    // implementation-detail interface with the principled value-
    // based one is the whole point of this change.
    std::vector<int> shapeFilterForRecipe;
    if (!opt.shapeFilter.empty() && r.goldSource == GoldSource::NativeExecution) {
      auto it = tritonByName.find(r.name);
      if (it != tritonByName.end()) {
        const TritonRecipe *tp = it->second;
        for (int wanted : opt.shapeFilter) {
          for (size_t i = 0; i < tp->defaultShapeValues.size(); ++i) {
            if (tp->defaultShapeValues[i] == wanted) {
              shapeFilterForRecipe.push_back(static_cast<int>(i));
              break;
            }
          }
        }
      }
    } else if (!opt.shapeFilter.empty()) {
      // Non-Triton recipe: filter values are literal Ns.  Intersect
      // with `defaultNs` so that passing a value the recipe does not
      // declare as a supported default contributes zero runs (same
      // silent-drop behaviour as the Triton branch above — any user
      // who wanted a value the recipe does not support would be
      // launching a shape the recipe's `makeInput` / `compare` may
      // not handle correctly, which is the expensive half of the
      // cross-product to silence at the CLI layer).
      for (int v : opt.shapeFilter)
        if (std::find(r.defaultNs.begin(), r.defaultNs.end(), v) !=
            r.defaultNs.end())
          shapeFilterForRecipe.push_back(v);
    }
    const std::vector<int> &ns =
        opt.shapeFilter.empty() ? r.defaultNs : shapeFilterForRecipe;
    const std::vector<int> &blocks =
        opt.blockFilter.empty() ? r.defaultBlocks : opt.blockFilter;
    // For Triton recipes (native-as-gold) print the resolved shape label
    // instead of the opaque (N=<sweep-index>, B=0) pair.
    auto shapeLog = [&](int N, int B) -> std::string {
      if (r.goldSource != GoldSource::NativeExecution) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "N=%d B=%d", N, B);
        return buf;
      }
      for (const auto &t : allTritonRecipes()) {
        if (t.name == r.name &&
            N >= 0 && static_cast<size_t>(N) < t.defaultShapeValues.size()) {
          char buf[64];
          std::snprintf(buf, sizeof(buf), "%s=%d",
                        t.shapeDimName.c_str(), t.defaultShapeValues[N]);
          return buf;
        }
      }
      return "?";
    };
    for (int N : ns) {
      for (int B : blocks) {
        if (r.validate) {
          if (auto reason = r.validate(N, B)) {
            fprintf(stderr, "  [skip] recipe=%s %s  reason: %s\n",
                    r.name.c_str(), shapeLog(N, B).c_str(),
                    reason->c_str());
            continue;
          }
        }
        fprintf(stderr, "  [run] recipe=%s %s ...\n",
                r.name.c_str(), shapeLog(N, B).c_str());
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
