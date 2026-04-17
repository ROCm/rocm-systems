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
// Recipe registry
// ─────────────────────────────────────────────────────────────────────────────

const std::vector<Recipe> &allRecipes() {
  static const std::vector<Recipe> v = {
      makeVecaddRecipe(),
      makeBlockSumRecipe(),
      makeLaneSwapRecipe(),
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
  std::string t = "/tmp/cmp_correct_" + std::to_string(getpid()) + "_" + tag + ".bin";
  return t;
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
