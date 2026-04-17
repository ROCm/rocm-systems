// Integration test for runPipelineAllKernels: gfx1250 → gfx942 → GPU execute.
//
// Validates the full integration path:
//   1. runPipelineAllKernels raises ALL kernels in a code object
//   2. Links them into one merged HSACO via ld.lld
//   3. The merged HSACO loads and executes correctly on the GPU
//
// Uses vecadd_gfx1250.hsaco (single kernel) and, if available, AITER .co
// files (which may contain multiple kernels per file).

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

#define HIP_CHECK(call)                                                      \
  do {                                                                       \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,                  \
              hipGetErrorString(err), __FILE__, __LINE__);                    \
      return false;                                                          \
    }                                                                        \
  } while (0)

static const char *TEST_DATA_DIR = GFX1250_TEST_DATA_DIR;

// ============================================================================
// Test 1: Single-kernel HSACO through runPipelineAllKernels.
// Raises vecadd_gfx1250.hsaco (1 kernel), executes on GPU, verifies output.
// ============================================================================
static bool testVecaddAllKernels() {
  printf("=== Test 1: runPipelineAllKernels — vecadd (1 kernel) ===\n");

  std::string path = std::string(TEST_DATA_DIR) + "/vecadd_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  if (data.empty()) {
    fprintf(stderr, "Cannot read %s\n", path.c_str());
    return false;
  }

  auto kernelNames = transpiler::listKernelNames(data);
  printf("  Kernels in %s: %zu\n", path.c_str(), kernelNames.size());
  for (auto &k : kernelNames)
    printf("    - %s\n", k.c_str());

  auto result = transpiler::runPipelineAllKernels(data, "gfx1250", "gfx942");
  if (!result.success) {
    fprintf(stderr, "  runPipelineAllKernels FAILED\n");
    return false;
  }
  printf("  Raised %d/%d instructions, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  // Load the merged HSACO
  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, result.hsaco.data()));

  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, mod, "vecadd_kernel"));
  printf("  Kernel loaded successfully\n");

  // Prepare fp16 data: C[i] = A[i] + B[i]
  const int N = 1024;
  std::vector<__half> hA(N), hB(N), hC(N);
  for (int i = 0; i < N; i++) {
    hA[i] = __float2half(static_cast<float>(i) * 0.01f);
    hB[i] = __float2half(static_cast<float>(i) * 0.02f);
  }

  void *dA, *dB, *dC;
  HIP_CHECK(hipMalloc(&dA, N * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dB, N * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dC, N * sizeof(__half)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, N * sizeof(__half)));

  struct { void *A, *B, *C; int N; } args = {dA, dB, dC, N};
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1024, 1, 1,
                                   0, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hC.data(), dC, N * sizeof(__half), hipMemcpyDeviceToHost));

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

  if (errors == 0) {
    printf("  PASSED — all %d elements correct (maxErr=%e)\n\n", N, maxErr);
    return true;
  }
  printf("  FAILED — %d/%d mismatches (maxErr=%e)\n\n", errors, N, maxErr);
  return false;
}

// ============================================================================
// Test 2: Multi-kernel code object raise (no GPU execution).
// Takes an optional path to a .co directory, raises each .co with
// runPipelineAllKernels, and reports success/failure.
// ============================================================================
static int testMultiKernelRaise(const char *coDir, const char *isa) {
  printf("=== Test 2: Multi-kernel raise test ===\n");
  printf("  Directory: %s\n  ISA: %s\n\n", coDir, isa);

  // Reuse the batch collection logic: scan directory for .co/.hsaco files
  std::vector<std::string> files;
  DIR *d = opendir(coDir);
  if (!d) {
    fprintf(stderr, "  Cannot open directory: %s\n", coDir);
    return 1;
  }
  while (struct dirent *entry = readdir(d)) {
    std::string name = entry->d_name;
    if (name.size() >= 3 && (name.substr(name.size() - 3) == ".co" ||
        (name.size() >= 6 && name.substr(name.size() - 6) == ".hsaco")))
      files.push_back(std::string(coDir) + "/" + name);
  }
  closedir(d);
  std::sort(files.begin(), files.end());

  if (files.empty()) {
    printf("  No .co/.hsaco files found\n");
    return 0;
  }

  int passed = 0, failed = 0;
  for (auto &f : files) {
    auto data = transpiler::readFile(f);
    if (data.empty()) continue;

    auto kernelNames = transpiler::listKernelNames(data);
    printf("  [%zu kernels] %-50s ", kernelNames.size(), f.c_str());
    fflush(stdout);

    auto result = transpiler::runPipelineAllKernels(data, isa, "gfx942");
    if (result.success) {
      printf("OK (%d/%d insts, %zu bytes)\n",
             result.liftedCount, result.totalCount, result.hsaco.size());

      // Verify the merged HSACO can be loaded
      hipModule_t mod;
      hipError_t err = hipModuleLoadData(&mod, result.hsaco.data());
      if (err == hipSuccess) {
        printf("    -> hipModuleLoadData: OK\n");
        (void)hipModuleUnload(mod);
      } else {
        printf("    -> hipModuleLoadData: FAILED (%s)\n", hipGetErrorString(err));
        failed++;
        continue;
      }

      passed++;
    } else {
      printf("FAILED\n");
      failed++;
    }
  }

  printf("\n  Summary: %d passed, %d failed out of %zu files\n\n",
         passed, failed, files.size());
  return failed > 0 ? 1 : 0;
}

int main(int argc, char **argv) {
  printf("=== Integration Test: runPipelineAllKernels ===\n\n");

  bool vecaddOk = testVecaddAllKernels();

  int multiResult = 0;
  if (argc >= 2) {
    const char *isa = (argc >= 3) ? argv[2] : "gfx942";
    multiResult = testMultiKernelRaise(argv[1], isa);
  }

  printf("================================================================\n");
  printf("  vecadd (GPU verified):  %s\n", vecaddOk ? "PASSED" : "FAILED");
  if (argc >= 2)
    printf("  multi-kernel raise:     %s\n", multiResult == 0 ? "PASSED" : "FAILED");
  printf("================================================================\n");

  return (!vecaddOk || multiResult != 0) ? 1 : 0;
}
