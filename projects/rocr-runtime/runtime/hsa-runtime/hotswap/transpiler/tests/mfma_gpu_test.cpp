#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,                    \
              hipGetErrorString(err), __FILE__, __LINE__);                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int testMfmaKernel(const char *kernelSymbol, int M, int N, int K) {
  printf("\n=== Testing %s: %dx%dx%d ===\n", kernelSymbol, M, N, K);

  std::string coPath = MFMA_CO_PATH;
  auto coData = transpiler::readFile(coPath);
  if (coData.empty()) {
    fprintf(stderr, "ERROR: Failed to read code object: %s\n", coPath.c_str());
    return 1;
  }

  // Step 1: Run original kernel to get reference output
  hipModule_t origMod;
  HIP_CHECK(hipModuleLoadData(&origMod, coData.data()));
  hipFunction_t origKernel;
  HIP_CHECK(hipModuleGetFunction(&origKernel, origMod, kernelSymbol));

  std::vector<__half> hA(M * K), hB(K * N);
  for (int i = 0; i < M * K; i++)
    hA[i] = __float2half(1.0f + (i % 7) * 0.1f);
  for (int i = 0; i < K * N; i++)
    hB[i] = __float2half(1.0f + (i % 11) * 0.1f);

  __half *dA, *dB;
  float *dC;
  HIP_CHECK(hipMalloc(&dA, M * K * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dB, K * N * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dC, M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), M * K * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, M * N * sizeof(float)));

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

  HIP_CHECK(hipModuleLaunchKernel(origKernel, gridX, gridY, 1, 64, 1, 1,
                                   0, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> hRef(M * N);
  HIP_CHECK(hipMemcpy(hRef.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipModuleUnload(origMod));

  printf("  Original kernel reference: C[0]=%f C[1]=%f C[%d]=%f\n",
         hRef[0], hRef[1], M*N-1, hRef[M*N-1]);

  // Step 2: Run translated kernel
  auto pipeResult = transpiler::runPipeline(coData, "gfx942", kernelSymbol);
  if (!pipeResult.success) {
    fprintf(stderr, "ERROR: Pipeline failed for %s\n", kernelSymbol);
    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    return 1;
  }
  printf("  Raised %d/%d instructions\n", pipeResult.liftedCount,
         pipeResult.totalCount);

  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, pipeResult.hsaco.data()));
  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, mod, kernelSymbol));

  HIP_CHECK(hipMemset(dC, 0, M * N * sizeof(float)));
  args.C = dC;
  HIP_CHECK(hipModuleLaunchKernel(kernel, gridX, gridY, 1, 64, 1, 1,
                                   0, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> hC(M * N);
  HIP_CHECK(hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

  printf("  Translated kernel result:  C[0]=%f C[1]=%f C[%d]=%f\n",
         hC[0], hC[1], M*N-1, hC[M*N-1]);

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
  if (errors == 0) {
    printf("  PASSED\n");
  } else {
    printf("  FAILED: %d/%d mismatches\n", errors, M * N);
  }

  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dC));
  HIP_CHECK(hipModuleUnload(mod));

  return errors;
}

int main() {
  printf("=== MFMA Binary Translation GPU Test ===\n");

  int total_errors = 0;

  total_errors += testMfmaKernel("mfma_gemm_16x16", 16, 16, 16);
  total_errors += testMfmaKernel("mfma_gemm_16x16", 32, 32, 32);
  total_errors += testMfmaKernel("mfma_gemm_16x16", 64, 64, 64);

  printf("\n=== MFMA Test %s ===\n",
         total_errors == 0 ? "PASSED" : "FAILED");
  return total_errors == 0 ? 0 : 1;
}
