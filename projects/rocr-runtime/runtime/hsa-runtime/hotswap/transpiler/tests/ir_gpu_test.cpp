#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

class IrGpu : public GpuTest {};

TEST_F(IrGpu, VecaddRoundtrip) {
  printf("=== LLVM IR Binary Translation — GPU Execution Test ===\n\n");

  std::string coPath = MVE_CO_PATH;
  if (!fileExists(coPath))
    GTEST_SKIP() << "Code object not found (build artifact): " << coPath;

  printf("[1] Loading original code object: %s\n", coPath.c_str());
  auto coData = transpiler::readFile(coPath);
  ASSERT_FALSE(coData.empty()) << "Failed to read code object: " << coPath;
  printf("    Code object size: %zu bytes\n", coData.size());

  printf("\n[2] Running LLVM IR raise + compile pipeline...\n");
  auto pipeResult =
      transpiler::runPipeline(coData, "gfx942", "_Z6vecaddPfS_S_i");
  ASSERT_TRUE(pipeResult.success) << "Pipeline failed";
  printf("    Raised %d/%d instructions to LLVM IR\n", pipeResult.liftedCount,
         pipeResult.totalCount);
  printf("    Generated HSACO: %zu bytes\n", pipeResult.hsaco.size());

  printf("\n[3] Loading generated HSACO into HIP...\n");
  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, pipeResult.hsaco.data()));

  hipFunction_t kernel;
  HIP_ASSERT(hipModuleGetFunction(&kernel, mod, "_Z6vecaddPfS_S_i"));
  printf("    Kernel loaded successfully\n");

  const int N = 1024;
  std::vector<float> hA(N), hB(N), hC(N, 0.0f);
  for (int i = 0; i < N; i++) {
    hA[i] = static_cast<float>(i);
    hB[i] = static_cast<float>(i * 2);
  }

  float *dA, *dB, *dC;
  HIP_ASSERT(hipMalloc(&dA, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&dB, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&dC, N * sizeof(float)));
  HIP_ASSERT(
      hipMemcpy(dA, hA.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(
      hipMemcpy(dB, hB.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dC, 0, N * sizeof(float)));

  printf("\n[4] Launching translated kernel (N=%d, 256 threads)...\n", N);
  struct {
    float *A, *B, *C;
    int N;
  } args = {dA, dB, dC, N};

  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_ASSERT(hipModuleLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1, 0,
                                   nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  printf("    Kernel completed\n");

  HIP_ASSERT(
      hipMemcpy(hC.data(), dC, N * sizeof(float), hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < N; i++) {
    float expected = hA[i] + hB[i];
    if (std::fabs(hC[i] - expected) > 1e-5f) {
      if (errors < 5)
        fprintf(stderr, "  MISMATCH at [%d]: got %f, expected %f\n", i, hC[i],
                expected);
      errors++;
    }
  }
  EXPECT_EQ(errors, 0) << errors << "/" << N << " element mismatches";

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipModuleUnload(mod);
}
