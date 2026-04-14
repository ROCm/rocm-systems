#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,                    \
              hipGetErrorString(err), __FILE__, __LINE__);                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  printf("=============================================================\n");
  printf("  LLVM MIR Proto: Binary → LLVM MIR → Assembly → HSACO → GPU\n");
  printf("=============================================================\n\n");

  // --- GPU check ---
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("[GPU] %s  arch=%s\n\n", props.name, props.gcnArchName);

  // --- Step 1: Read pre-compiled code object ---
  printf("[1/5] Reading code object: %s\n", MVE_CO_PATH);
  auto coData = mir_proto::readFile(MVE_CO_PATH);
  if (coData.empty()) {
    fprintf(stderr, "FATAL: Cannot read code object\n");
    return 1;
  }
  printf("       Code object: %zu bytes\n", coData.size());

  // --- Step 2: Extract .text section ---
  printf("[2/5] Extracting .text section...\n");
  auto textSec = mir_proto::extractTextSection(coData);
  if (!textSec.valid) {
    fprintf(stderr, "FATAL: Cannot extract .text section\n");
    return 1;
  }
  printf("       .text: %zu bytes (%zu instructions approx)\n",
         textSec.bytes.size(), textSec.bytes.size() / 4);

  // --- Step 3: Run LLVM MIR pipeline ---
  printf("[3/5] Running LLVM MIR pipeline (lift → MIR → llc → HSACO)...\n");
  auto result = mir_proto::runPipeline(textSec.bytes, "gfx942", "vecadd");

  printf("\n       Lift summary:\n");
  printf("         Lifted:      %d instructions\n", result.liftedCount);
  printf("         Total:       %d instructions\n", result.totalCount);
  printf("         Coverage:    %.1f%%\n",
         result.totalCount > 0
             ? 100.0 * result.liftedCount / result.totalCount
             : 0.0);

  if (!result.success) {
    printf("\n[RESULT] Pipeline did not produce HSACO.\n");
    printf("[MIR] Dumping MIR for debugging:\n%s\n", result.mirText.c_str());
    return 1;
  }

  // --- Step 4: Load HSACO on GPU ---
  printf("[4/5] Loading HSACO on GPU (%zu bytes)...\n", result.hsaco.size());
  hipModule_t module;
  HIP_CHECK(hipModuleLoadData(&module, result.hsaco.data()));

  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "vecadd"));
  printf("       Kernel 'vecadd' loaded successfully\n");

  // --- Step 5: Execute vecadd on GPU ---
  printf("[5/5] Executing vecadd on GPU...\n");
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
  HIP_CHECK(
      hipMemcpy(dA, hA.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(dB, hB.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, N * sizeof(float)));

  struct {
    float *A;
    float *B;
    float *C;
    int N;
  } args = {dA, dB, dC, N};
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1, 0,
                                  nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(
      hipMemcpy(hC.data(), dC, N * sizeof(float), hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < N; i++) {
    float expected = hA[i] + hB[i];
    if (std::fabs(hC[i] - expected) > 1e-5f) {
      if (errors < 5)
        printf("  MISMATCH [%d]: %f vs %f\n", i, hC[i], expected);
      errors++;
    }
  }
  printf("\n=============================================================\n");
  if (errors == 0) {
    printf("  PASS: All %d elements correct!\n", N);
    printf(
        "  Pipeline: GFX942 binary → LLVM MIR → MachineInstr→MCInst → ASM → HSACO → GPU\n");
    printf("  Assembly generated from live MachineFunction (MIR in codegen path).\n");
    printf("  Implicit operands (VCC, SCC, EXEC) from MCInstrDesc/TableGen.\n");
  } else {
    printf("  FAIL: %d / %d elements incorrect\n", errors, N);
  }
  printf("=============================================================\n");

  hipFree(dA);
  hipFree(dB);
  hipFree(dC);
  hipModuleUnload(module);
  return errors == 0 ? 0 : 1;
}
