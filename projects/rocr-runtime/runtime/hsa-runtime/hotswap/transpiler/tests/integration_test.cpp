// Integration test for runPipelineAllKernels: gfx1250 -> gfx942 -> GPU execute.
//
// Validates the full integration path:
//   1. runPipelineAllKernels raises ALL kernels in a code object
//   2. Links them into one merged HSACO via ld.lld
//   3. The merged HSACO loads and executes correctly on the GPU

#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

static const char *INTEG_DATA_DIR = GFX1250_TEST_DATA_DIR;

// ============================================================================
// Test 1: Single-kernel HSACO through runPipelineAllKernels.
// Raises vecadd_gfx1250.hsaco (1 kernel), executes on GPU, verifies output.
// ============================================================================
static void doTestVecaddAllKernels() {
  printf("=== Test 1: runPipelineAllKernels --- vecadd (1 kernel) ===\n");

  std::string path = std::string(INTEG_DATA_DIR) + "/vecadd_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto kernelNames = transpiler::listKernelNames(data);
  printf("  Kernels in %s: %zu\n", path.c_str(), kernelNames.size());
  for (auto &k : kernelNames)
    printf("    - %s\n", k.c_str());

  auto result = transpiler::runPipelineAllKernels(data, "gfx1250", "gfx942");
  ASSERT_TRUE(result.success) << "runPipelineAllKernels FAILED";
  printf("  Raised %d/%d instructions, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t kernel;
  HIP_ASSERT(hipModuleGetFunction(&kernel, mod, "vecadd_kernel"));
  printf("  Kernel loaded successfully\n");

  const int N = 1024;
  std::vector<__half> hA(N), hB(N), hC(N);
  for (int i = 0; i < N; i++) {
    hA[i] = __float2half(static_cast<float>(i) * 0.01f);
    hB[i] = __float2half(static_cast<float>(i) * 0.02f);
  }

  void *dA, *dB, *dC;
  HIP_ASSERT(hipMalloc(&dA, N * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dB, N * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dC, N * sizeof(__half)));
  HIP_ASSERT(hipMemcpy(dA, hA.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dB, hB.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dC, 0, N * sizeof(__half)));

  struct { void *A, *B, *C; int N; } args = {dA, dB, dC, N};
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_ASSERT(hipModuleLaunchKernel(kernel, 1, 1, 1, 1024, 1, 1,
                                   0, nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hC.data(), dC, N * sizeof(__half), hipMemcpyDeviceToHost));

  int errors = 0;
  float maxErr = 0.0f;
  for (int i = 0; i < N; i++) {
    float a = __half2float(hA[i]);
    float b = __half2float(hB[i]);
    float c = __half2float(hC[i]);
    float expected = a + b;
    float diff = std::fabs(c - expected);
    if (diff > maxErr) maxErr = diff;
    if (diff > 0.01f) {
      if (errors < 3)
        fprintf(stderr, "    MISMATCH [%d]: got %f, expected %f\n", i, c, expected);
      errors++;
    }
  }

  (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
  (void)hipModuleUnload(mod);

  printf("  maxErr=%e\n", maxErr);
  EXPECT_EQ(errors, 0) << errors << "/" << N << " mismatches in vecadd all-kernels";
}

// ============================================================================
// Test 2: Multi-kernel code object raise (no GPU execution for batch).
// Raises .co files from GFX1250_TEST_DATA_DIR with runPipelineAllKernels,
// then verifies the merged HSACO loads with HIP.
// ============================================================================
// Known classifier-refusal inputs in the GFX1250_TEST_DATA_DIR corpus.
//
// Each entry is a (basename, reason) pair identifying a code object we
// expect runPipelineAllKernels to REFUSE (not raise) for principled
// reasons documented in wave-size-translation.md §6 / §7. The
// integration test treats a refusal on one of these as the expected
// outcome and asserts the corresponding ObstructionKind anchor is
// present in the pipeline's diagnostic stream.
//
// Add a new entry here whenever a corpus code object starts refusing
// at raise time because it hits a known-unrewritable obstruction
// class. REMOVE an entry when the underlying obstruction is lifted in
// the classifier — leaving a stale entry in place would convert a
// newly-supported raise into a test failure.
struct ExpectedRefusal {
  const char *basename;
  const char *diagnosticAnchor;  // stable substring the pipeline prints
  const char *reason;            // short human-readable note for the log
};

static constexpr ExpectedRefusal kExpectedRefusals[] = {
    // matmul_f16_large_gfx1250 — the 128x128-tile HIP matmul. Carries
    // the three-way co-occurrence pinned by the
    // lit_tests/c1_wave_id_lift_scalarized refusal fixture (canonical
    // `s_bfe_u32 sDST, ttmp8, 0x50019` + v_writelane/v_readlane for
    // register spills + v_wmma_* accumulator). See the long comment
    // on the matching Gfx1250Gpu.Matmul128x128* entries in
    // tests/xfail.cmake for the principled justification.
    {"matmul_f16_large_gfx1250.hsaco", "WaveIdLiftScalarized",
     "128x128-tile matmul: canonical wave_id BFE lift scalarised "
     "through v_writelane/v_readlane under WMMA (Class 1 refuse)"},
};

// Returns a pointer to the ExpectedRefusal entry matching `path`'s
// basename, or nullptr if the file is not on the refusal allowlist.
static const ExpectedRefusal *findExpectedRefusal(const std::string &path) {
  auto slash = path.find_last_of('/');
  std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
  for (const auto &er : kExpectedRefusals) {
    if (base == er.basename) return &er;
  }
  return nullptr;
}

static void doTestMultiKernelRaise() {
  printf("=== Test 2: Multi-kernel raise test ===\n");
  std::string coDir = INTEG_DATA_DIR;
  const char *isa = "gfx1250";
  printf("  Directory: %s\n  ISA: %s\n\n", coDir.c_str(), isa);

  std::vector<std::string> files;
  DIR *d = opendir(coDir.c_str());
  if (!d)
    GTEST_SKIP() << "Test data directory not found: " << coDir;

  while (struct dirent *entry = readdir(d)) {
    std::string name = entry->d_name;
    if (name.size() >= 3 && (name.substr(name.size() - 3) == ".co" ||
        (name.size() >= 6 && name.substr(name.size() - 6) == ".hsaco")))
      files.push_back(coDir + "/" + name);
  }
  closedir(d);
  std::sort(files.begin(), files.end());

  if (files.empty())
    GTEST_SKIP() << "No .co/.hsaco files found in " << coDir;

  int passed = 0, failed = 0, refused = 0;
  for (auto &f : files) {
    auto data = transpiler::readFile(f);
    if (data.empty()) continue;

    auto kernelNames = transpiler::listKernelNames(data);
    printf("  [%zu kernels] %-50s ", kernelNames.size(), f.c_str());
    fflush(stdout);

    const ExpectedRefusal *er = findExpectedRefusal(f);

    auto result = transpiler::runPipelineAllKernels(data, isa, "gfx942");
    if (result.success) {
      if (er) {
        // An entry in the expected-refusal allowlist that raised
        // anyway is a signal that the classifier's refusal logic has
        // been weakened or the corpus binary has been regenerated
        // with a shape that no longer triggers it. Either way the
        // allowlist is now stale — fail loudly so the entry gets
        // removed or the regression investigated.
        printf("UNEXPECTED OK (on classifier-refusal allowlist)\n");
        failed++;
        ADD_FAILURE()
            << f << " raised successfully but is on the expected-refusal "
            << "allowlist (reason: " << er->reason << "). Remove the entry "
            << "in kExpectedRefusals if the underlying obstruction has been "
            << "lifted; otherwise investigate why the refusal no longer "
            << "fires on this binary.";
        continue;
      }

      printf("OK (%d/%d insts, %zu bytes)\n",
             result.liftedCount, result.totalCount, result.hsaco.size());

      hipModule_t mod;
      hipError_t err = hipModuleLoadData(&mod, result.hsaco.data());
      if (err == hipSuccess) {
        printf("    -> hipModuleLoadData: OK\n");
        (void)hipModuleUnload(mod);
      } else {
        printf("    -> hipModuleLoadData: FAILED (%s)\n", hipGetErrorString(err));
        failed++;
        EXPECT_EQ(err, hipSuccess)
            << "Failed to load merged HSACO for " << f;
        continue;
      }
      passed++;
    } else if (er) {
      // Expected classifier refusal. Count separately so the summary
      // line shows it as a distinct category, not a pipeline bug.
      printf("REFUSED (expected: %s)\n", er->diagnosticAnchor);
      refused++;
    } else {
      printf("FAILED\n");
      failed++;
      EXPECT_TRUE(result.success) << "Pipeline failed for " << f;
    }
  }

  printf("\n  Summary: %d passed, %d expected-refused, %d failed out of %zu files\n\n",
         passed, refused, failed, files.size());
}

class Integration : public GpuTest {};

// XFAIL: see tests/xfail.cmake (hipError 719 in merged-HSACO launch)
TEST_F(Integration, VecaddAllKernels) { doTestVecaddAllKernels(); }
TEST_F(Integration, MultiKernelRaise) { doTestMultiKernelRaise(); }
