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

// Build the divergent-vgpr prototype code object (see
// tests/divergent_vgpr.hip). Mirrors ensureDivergentExecCo() in the
// divergent_exec test — we cannot share the helper because each kernel
// has its own .co and hipcc --genco accepts a single source file.
static bool ensureDivergentVgprCo(const std::string &transpilerSrcDir) {
  std::string co =
      transpilerSrcDir + "/build/divergent_vgpr_gfx942_unbundled.co";
  if (fileExists(co))
    return true;

  std::string hip = transpilerSrcDir + "/tests/divergent_vgpr.hip";
  if (!fileExists(hip))
    return false;

  std::string buildDir = transpilerSrcDir + "/build";
  mkdir(buildDir.c_str(), 0755);

  std::string bundled = buildDir + "/divergent_vgpr_gfx942.co";

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

class DivergentVgprGpu : public GpuTest {};

// Verification test for the SPE model's per-lane VGPR semantics.
//
// Unlike divergent_exec_gpu_test (which tests divergent memory stores),
// this test has a SINGLE unmasked global store at the end, but two
// v_cmpx-gated regions that each modify the same VGPR via v_mov_b32.
// Each lane's final VGPR value depends on which regions it was active
// in:
//
//   lane  0..15  -> 0xAA  (region 1 wrote)
//   lane 16..31  -> 0xBB  (region 2 wrote)
//   lane 32..63  -> 0xCC  (neither wrote; default preserved)
//
// Under SPE, for each VGPR write the raiser emits an `if (lane_active)`
// diamond with a store into the per-thread VGPR alloca. mem2reg then
// promotes those stores into phi nodes that merge the lane's previous
// value with the new value on the predicate. That chain is what
// preserves each lane's distinct VGPR history across the two regions.
// If any step silently broadcasts or drops the predicate, lanes will
// smear values from neighbours and this test will fail.
TEST_F(DivergentVgprGpu, PerLaneVgprPersistsAcrossRegions) {
  if (!ensureDivergentVgprCo(TRANSPILER_SRC_DIR))
    GTEST_SKIP() << "Could not build divergent-vgpr code object (hipcc or "
                    "clang-offload-bundler not available)";

  std::string coPath =
      std::string(TRANSPILER_SRC_DIR) +
      "/build/divergent_vgpr_gfx942_unbundled.co";
  if (!fileExists(coPath))
    GTEST_SKIP() << "Code object not found: " << coPath;

  auto coData = transpiler::readFile(coPath);
  ASSERT_FALSE(coData.empty()) << "Failed to read code object: " << coPath;

  const char *symbol = "divergent_vgpr_kernel";
  constexpr int N = 64;
  int expected[N];
  for (int i = 0; i < N; i++) {
    if (i < 16)
      expected[i] = 0xAA;
    else if (i < 32)
      expected[i] = 0xBB;
    else
      expected[i] = 0xCC;
  }

  // --- Sanity check: native (original) hardware produces `expected`. ---
  //
  // The memset(0xCC,...) pre-fills out[] with 0xCCCCCCCC per dword, but
  // the kernel writes the in-register default (C literal 0xCC = 0x000000CC)
  // for inactive lanes. So lanes 32..63 observe 0x000000CC, matching
  // `expected[i] = 0xCC` above. (The two 0xCCs are chosen to both end up
  // equal to rule out "store never happened" as a false pass.)
  {
    hipModule_t mod;
    HIP_ASSERT(hipModuleLoadData(&mod, coData.data()));
    hipFunction_t kernel;
    HIP_ASSERT(hipModuleGetFunction(&kernel, mod, symbol));

    int *d;
    HIP_ASSERT(hipMalloc(&d, N * sizeof(int)));
    HIP_ASSERT(hipMemset(d, 0, N * sizeof(int)));

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
  auto pipeResult = transpiler::runPipeline(coData, "gfx942", symbol);
  ASSERT_TRUE(pipeResult.success) << "Pipeline failed for " << symbol;
  printf("  Raised %d/%d instructions\n", pipeResult.liftedCount,
         pipeResult.totalCount);

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, pipeResult.hsaco.data()));
  hipFunction_t kernel;
  HIP_ASSERT(hipModuleGetFunction(&kernel, mod, symbol));

  int *d;
  HIP_ASSERT(hipMalloc(&d, N * sizeof(int)));
  HIP_ASSERT(hipMemset(d, 0, N * sizeof(int)));

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
      << " lanes mismatched between translated and native kernel; this "
         "indicates the SPE per-lane VGPR divergence model is broken.";
}
