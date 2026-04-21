#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Build the divergent-exec prototype code object (see
// tests/divergent_exec.hip). Returns true if the unbundled .co is
// available after the call. Mirrors ensureMfmaCo() in test_common.hpp.
static bool ensureDivergentExecCo(const std::string &transpilerSrcDir) {
  std::string co =
      transpilerSrcDir + "/build/divergent_exec_gfx942_unbundled.co";
  if (fileExists(co))
    return true;

  std::string hip = transpilerSrcDir + "/tests/divergent_exec.hip";
  if (!fileExists(hip))
    return false;

  std::string buildDir = transpilerSrcDir + "/build";
  mkdir(buildDir.c_str(), 0755);

  std::string bundled = buildDir + "/divergent_exec_gfx942.co";

  // hipcc -> bundled fatbin
  {
    pid_t pid = fork();
    if (pid == 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
      execlp("hipcc", "hipcc", "--genco", "--offload-arch=gfx942", "-o",
             bundled.c_str(), hip.c_str(), nullptr);
      _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
      return false;
  }

  const char *bundlers[] = {"/opt/rocm/lib/llvm/bin/clang-offload-bundler",
                            "/opt/rocm/llvm/bin/clang-offload-bundler"};
  for (auto *b : bundlers) {
    if (!fileExists(b))
      continue;
    pid_t pid = fork();
    if (pid == 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
      execl(b, b, "--type=o", ("--input=" + bundled).c_str(),
            ("--output=" + co).c_str(),
            "--targets=hipv4-amdgcn-amd-amdhsa--gfx942", "--unbundle",
            nullptr);
      _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0 && fileExists(co))
      return true;
  }
  return false;
}

class DivergentExecGpu : public GpuTest {};

// Reproducer for issue #11: intra-block EXEC divergence via two v_cmpx-
// gated global stores. On native hardware the per-lane semantics are:
//
//   lane  0..15:  out[lane] = 0xAA
//   lane 16..31:  out[lane] = 0xBB
//   lane 32..63:  out[lane] = 0
//
// The current translator models EXEC as an alloca that is not connected
// to the backend's hardware EXEC management, so the raised IR emits
// both global stores unmasked. The last store (0xBB) wins on every
// lane -> this test fails today. Marked WILL_FAIL in xfail.cmake; after
// SPE lands, the XFAIL entry is removed.
TEST_F(DivergentExecGpu, TwoCmpxGatedStores) {
  if (!ensureDivergentExecCo(TRANSPILER_SRC_DIR))
    GTEST_SKIP() << "Could not build divergent-exec code object (hipcc or "
                    "clang-offload-bundler not available)";

  std::string coPath =
      std::string(TRANSPILER_SRC_DIR) +
      "/build/divergent_exec_gfx942_unbundled.co";
  if (!fileExists(coPath))
    GTEST_SKIP() << "Code object not found: " << coPath;

  auto coData = transpiler::readFile(coPath);
  ASSERT_FALSE(coData.empty()) << "Failed to read code object: " << coPath;

  const char *symbol = "divergent_exec_kernel";
  constexpr int N = 64;
  const int expected[N] = {
      0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
      0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
      0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB,
      0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB,
      0,    0,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    0,    0,    0,    0,
  };

  // --- Sanity check: native (original) hardware produces `expected`. ---
  {
    hipModule_t mod;
    HIP_ASSERT(hipModuleLoadData(&mod, coData.data()));
    hipFunction_t kernel;
    HIP_ASSERT(hipModuleGetFunction(&kernel, mod, symbol));

    int *d;
    HIP_ASSERT(hipMalloc(&d, N * sizeof(int)));
    HIP_ASSERT(hipMemset(d, 0xCC, N * sizeof(int)));

    struct {
      int *out;
    } args = {d};
    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};
    HIP_ASSERT(hipModuleLaunchKernel(kernel, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                     nullptr, config));
    HIP_ASSERT(hipDeviceSynchronize());

    std::vector<int> h(N);
    HIP_ASSERT(hipMemcpy(h.data(), d, N * sizeof(int), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(d));
    HIP_ASSERT(hipModuleUnload(mod));

    for (int i = 0; i < N; i++) {
      ASSERT_EQ(h[i], expected[i])
          << "Native kernel produced unexpected value at lane " << i
          << ": got 0x" << std::hex << h[i] << " expected 0x" << expected[i];
    }
  }

  // --- Pipeline: raise + reassemble, then run raised version. ---
  // Single-ISA lift: pass the ISA twice (source == target).  The
  // previous 3-string convenience overload was removed after it
  // was shown to silently capture 4-string cross-arch calls under
  // C++ overload resolution (pointer-to-bool standard conversion
  // outranking user-defined const char*→std::string); see
  // pipeline.hpp for the full derivation.
  auto pipeResult =
      transpiler::runPipeline(coData, "gfx942", "gfx942", symbol);
  ASSERT_TRUE(pipeResult.success) << "Pipeline failed for " << symbol;
  printf("  Raised %d/%d instructions\n", pipeResult.liftedCount,
         pipeResult.totalCount);

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, pipeResult.hsaco.data()));
  hipFunction_t kernel;
  HIP_ASSERT(hipModuleGetFunction(&kernel, mod, symbol));

  int *d;
  HIP_ASSERT(hipMalloc(&d, N * sizeof(int)));
  HIP_ASSERT(hipMemset(d, 0xCC, N * sizeof(int)));

  struct {
    int *out;
  } args = {d};
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};
  HIP_ASSERT(hipModuleLaunchKernel(kernel, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                   nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<int> h(N);
  HIP_ASSERT(hipMemcpy(h.data(), d, N * sizeof(int), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(d));
  HIP_ASSERT(hipModuleUnload(mod));

  int mismatches = 0;
  for (int i = 0; i < N; i++) {
    if (h[i] != expected[i]) {
      if (mismatches < 8)
        fprintf(stderr, "  MISMATCH lane %2d: got 0x%02x expected 0x%02x\n", i,
                h[i], expected[i]);
      mismatches++;
    }
  }
  EXPECT_EQ(mismatches, 0)
      << mismatches << "/" << N
      << " lanes mismatched between translated and native kernel; this is "
         "the intra-block EXEC divergence bug (issue #11)";
}
