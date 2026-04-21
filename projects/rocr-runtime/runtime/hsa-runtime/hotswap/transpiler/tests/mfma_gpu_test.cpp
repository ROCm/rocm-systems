#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void testMfmaKernel(const char *kernelSymbol, int M, int N, int K) {
  printf("\n=== Testing %s: %dx%dx%d ===\n", kernelSymbol, M, N, K);

  if (!ensureMfmaCo(TRANSPILER_SRC_DIR))
    GTEST_SKIP() << "Could not build MFMA code object (hipcc or "
                    "clang-offload-bundler not available)";

  std::string coPath = MFMA_CO_PATH;
  if (!fileExists(coPath))
    GTEST_SKIP() << "MFMA code object not found: " << coPath;

  auto coData = transpiler::readFile(coPath);
  ASSERT_FALSE(coData.empty()) << "Failed to read code object: " << coPath;

  // Run original kernel to get reference output
  hipModule_t origMod;
  HIP_ASSERT(hipModuleLoadData(&origMod, coData.data()));
  hipFunction_t origKernel;
  HIP_ASSERT(hipModuleGetFunction(&origKernel, origMod, kernelSymbol));

  std::vector<__half> hA(M * K), hB(K * N);
  for (int i = 0; i < M * K; i++)
    hA[i] = __float2half(1.0f + (i % 7) * 0.1f);
  for (int i = 0; i < K * N; i++)
    hB[i] = __float2half(1.0f + (i % 11) * 0.1f);

  __half *dA, *dB;
  float *dC;
  HIP_ASSERT(hipMalloc(&dA, M * K * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dB, K * N * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dC, M * N * sizeof(float)));
  HIP_ASSERT(hipMemcpy(dA, hA.data(), M * K * sizeof(__half),
                        hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dB, hB.data(), K * N * sizeof(__half),
                        hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dC, 0, M * N * sizeof(float)));

  int tileSize = 16;
  int gridX = (N + tileSize - 1) / tileSize;
  int gridY = (M + tileSize - 1) / tileSize;

  struct {
    float *C;
    const __half *A;
    const __half *B;
    int M, N, K;
  } args = {dC, dA, dB, M, N, K};

  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_ASSERT(hipModuleLaunchKernel(origKernel, gridX, gridY, 1, 64, 1, 1, 0,
                                   nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<float> hRef(M * N);
  HIP_ASSERT(hipMemcpy(hRef.data(), dC, M * N * sizeof(float),
                        hipMemcpyDeviceToHost));
  HIP_ASSERT(hipModuleUnload(origMod));

  printf("  Original kernel reference: C[0]=%f C[1]=%f C[%d]=%f\n", hRef[0],
         hRef[1], M * N - 1, hRef[M * N - 1]);

  // Run translated kernel
  // Single-ISA lift: pass the ISA twice (source == target).  See
  // pipeline.hpp for why the 3-string convenience overload was
  // removed (silent capture of 4-string cross-arch calls under
  // standard pointer-to-bool conversion).
  auto pipeResult = transpiler::runPipeline(coData, "gfx942", "gfx942",
                                            kernelSymbol);
  ASSERT_TRUE(pipeResult.success) << "Pipeline failed for " << kernelSymbol;
  printf("  Raised %d/%d instructions\n", pipeResult.liftedCount,
         pipeResult.totalCount);

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, pipeResult.hsaco.data()));
  hipFunction_t kernel;
  HIP_ASSERT(hipModuleGetFunction(&kernel, mod, kernelSymbol));

  HIP_ASSERT(hipMemset(dC, 0, M * N * sizeof(float)));
  args.C = dC;
  HIP_ASSERT(hipModuleLaunchKernel(kernel, gridX, gridY, 1, 64, 1, 1, 0,
                                   nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<float> hC(M * N);
  HIP_ASSERT(
      hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

  printf("  Translated kernel result:  C[0]=%f C[1]=%f C[%d]=%f\n", hC[0],
         hC[1], M * N - 1, hC[M * N - 1]);

  int errors = 0;
  float maxRelErr = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float ref = hRef[i];
    float got = hC[i];
    float absErr = std::fabs(got - ref);
    float relErr = (ref != 0.0f) ? absErr / std::fabs(ref) : absErr;
    if (relErr > maxRelErr) maxRelErr = relErr;
    if (absErr > 0.01f) {
      if (errors < 5)
        fprintf(stderr, "  MISMATCH [%d]: translated=%f original=%f (abs=%e)\n",
                i, got, ref, absErr);
      errors++;
    }
  }

  printf("  Max relative error vs original: %e\n", maxRelErr);
  EXPECT_EQ(errors, 0)
      << errors << "/" << (M * N) << " mismatches for " << kernelSymbol
      << " " << M << "x" << N << "x" << K;

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipModuleUnload(mod);
}

class MfmaGpu : public GpuTest {};

TEST_F(MfmaGpu, Gemm16x16x16) { testMfmaKernel("mfma_gemm_16x16", 16, 16, 16); }
TEST_F(MfmaGpu, Gemm32x32x32) { testMfmaKernel("mfma_gemm_16x16", 32, 32, 32); }
TEST_F(MfmaGpu, Gemm64x64x64) { testMfmaKernel("mfma_gemm_16x16", 64, 64, 64); }
