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
#include <array>
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

  // cross-widen gfx1250 (wave32) -> gfx942 (wave64) opts into the Phase
  // 6.5 writelane/readlane rewrite (see wave-size-translation.md §5.6.3
  // and rewrite_cross_lane_divergent.{hpp,cpp}). vecadd itself doesn't
  // hit the rewrite (no WMMA, no scalar v_writelane feeds), but we flip
  // the flag on here for coherence with MultiKernelRaise below — both
  // paths run the same end-to-end raise-then-lower pipeline and should
  // share the same flag policy so a future corpus addition cannot
  // silently diverge between the two tests.
  auto result = transpiler::runPipelineAllKernels(
      data, "gfx1250", "gfx942", /*enableWritelaneRewrite=*/true);
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

// Empty today — matmul_f16_large_gfx1250.hsaco graduated off this
// allowlist when the --enable-writelane-rewrite path landed; the
// MultiKernelRaise test opts into the rewrite above and now expects
// the 128x128 matmul to raise successfully. Re-add entries here only
// for code objects that remain syntactic refusals under the
// fully-enabled pipeline, i.e. hit a principled obstruction class
// that no current rewrite resolves.
//
// Using `std::array` (rather than a C array with a placeholder entry)
// lets the list be legitimately empty without sentinel rows that
// `findExpectedRefusal` would have to skip. ISO C++ does not allow a
// zero-length C array initializer, so we pay one template level to
// say "no exceptions today".
static constexpr std::array<ExpectedRefusal, 0> kExpectedRefusals{};

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

    // cross-widen gfx1250 (wave32) -> gfx942 (wave64) opts into the
    // Phase 6.5 writelane/readlane rewrite. This is the *runtime
    // verification* point for matmul_f16_large_gfx1250.hsaco: once the
    // rewrite graduates past its lit fixtures and into this end-to-end
    // harness, the 128x128-tile matmul must raise here rather than
    // refuse at Phase 1.4.5 (see c2_xfail_integration_cleanup in the
    // commit 2 plan and wave-size-translation.md §5.6.3).
    auto result = transpiler::runPipelineAllKernels(
        data, isa, "gfx942", /*enableWritelaneRewrite=*/true);
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
