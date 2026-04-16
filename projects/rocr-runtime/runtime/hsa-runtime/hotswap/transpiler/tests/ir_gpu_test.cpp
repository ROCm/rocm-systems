#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

int main() {
  printf("=== LLVM IR Binary Translation — GPU Execution Test ===\n\n");

  // Step 1: Read the original code object
  std::string coPath = MVE_CO_PATH;
  printf("[1] Loading original code object: %s\n", coPath.c_str());
  auto coData = transpiler::readFile(coPath);
  if (coData.empty()) {
    fprintf(stderr, "ERROR: Failed to read code object\n");
    return 1;
  }
  printf("    Code object size: %zu bytes\n", coData.size());

  // Step 2: Run the raise + compile pipeline
  printf("\n[2] Running LLVM IR raise + compile pipeline...\n");
  auto pipeResult = transpiler::runPipeline(coData, "gfx942",
                                          "_Z6vecaddPfS_S_i");
  if (!pipeResult.success) {
    fprintf(stderr, "ERROR: Pipeline failed\n");
    return 1;
  }
  printf("    Raised %d/%d instructions to LLVM IR\n", pipeResult.liftedCount,
         pipeResult.totalCount);
  printf("    Generated HSACO: %zu bytes\n", pipeResult.hsaco.size());

  // Step 3: Load the generated HSACO via HIP
  printf("\n[3] Loading generated HSACO into HIP...\n");
  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, pipeResult.hsaco.data()));

  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, mod, "_Z6vecaddPfS_S_i"));
  printf("    Kernel loaded successfully\n");

  // Step 4: Prepare data
  const int N = 1024;
  std::vector<float> hA(N), hB(N), hC(N, 0.0f);
  for (int i = 0; i < N; i++) {
    hA[i] = static_cast<float>(i);
    hB[i] = static_cast<float>(i * 2);
  }

  float *dA, *dB, *dC;
  HIP_CHECK(hipMalloc(&dA, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dB, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dC, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, N * sizeof(float)));

  // Step 5: Launch the translated kernel
  printf("\n[4] Launching translated kernel (N=%d, 256 threads)...\n", N);
  struct {
    float *A, *B, *C;
    int N;
  } args = {dA, dB, dC, N};

  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel,
                                   (N + 255) / 256, 1, 1,
                                   256, 1, 1,
                                   0, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  printf("    Kernel completed\n");

  // Step 6: Verify results
  HIP_CHECK(hipMemcpy(hC.data(), dC, N * sizeof(float), hipMemcpyDeviceToHost));

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

  printf("\n[5] Verification: ");
  if (errors == 0) {
    printf("PASSED — all %d elements correct\n", N);
  } else {
    printf("FAILED — %d/%d mismatches\n", errors, N);
  }

  // Cleanup
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dC));
  HIP_CHECK(hipModuleUnload(mod));

  printf("\n=== LLVM IR Binary Translation Test %s ===\n",
         errors == 0 ? "PASSED" : "FAILED");
  return errors == 0 ? 0 : 1;
}
