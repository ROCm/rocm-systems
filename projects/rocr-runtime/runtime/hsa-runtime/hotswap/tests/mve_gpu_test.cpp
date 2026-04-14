////////////////////////////////////////////////////////////////////////////////
//
// MVE GPU Round-Trip Test
//
// Demonstrates the MLIR pipeline is sound by:
//   1. Reading a pre-compiled vecadd kernel code object (gfx942 ELF)
//   2. Extracting .text bytes (raw instructions)
//   3. Running them through the MLIR pipeline (gfx942 -> gfx942)
//   4. Splicing re-emitted assembly into the device assembly template
//   5. Reassembling into a fresh code object via llvm-mc + ld.lld
//   6. Loading the new code object on the GPU via hipModuleLoadData
//   7. Launching the kernel and verifying correct output
//
////////////////////////////////////////////////////////////////////////////////

#include "code_object_builder.hpp"
#include "pipeline.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP ERROR at %s:%d: %s (code %d)\n", __FILE__,         \
              __LINE__, hipGetErrorString(err), err);                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const char *KERNEL_SYMBOL = "_Z6vecaddPfS_S_i";
static const char *LLVM_BIN_DIR = "/opt/rocm/llvm/bin";

int main() {
  printf("════════════════════════════════════════════════════════════════\n");
  printf("  MLIR Binary Translation – GPU Round-Trip Verification\n");
  printf("════════════════════════════════════════════════════════════════\n\n");

  // Print GPU info to prove we're on real hardware
  {
    hipDevice_t dev;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&props, dev));
    const char *displayName =
        (props.name[0] != '\0') ? props.name : props.gcnArchName;
    printf("  GPU: %s (%d CUs, %.0f MB VRAM)\n\n",
           displayName, props.multiProcessorCount,
           props.totalGlobalMem / (1024.0 * 1024.0));
  }

  printf("  Test: compile a HIP vecadd kernel → extract .text bytes →\n");
  printf("        lift to waveasm MLIR → re-emit assembly → reassemble\n");
  printf("        into a NEW code object → load on GPU → verify output\n\n");
  printf("────────────────────────────────────────────────────────────────\n\n");

  // ── Step 1: Read the pre-compiled code object ──
  printf("[1/7] Reading original code object\n");
  printf("      Path: %s\n", MVE_CO_PATH);
  auto coBytes = hotswap::readFile(MVE_CO_PATH);
  if (coBytes.empty()) {
    fprintf(stderr, "FATAL: Cannot read code object %s\n", MVE_CO_PATH);
    return 1;
  }
  printf("      Original code object: %zu bytes\n", coBytes.size());

  // ── Step 2: Extract .text section ──
  printf("\n[2/7] Extracting .text section (raw GPU instructions)\n");
  auto textSection = hotswap::extractTextSection(coBytes);
  if (!textSection.valid) {
    fprintf(stderr, "FATAL: Cannot extract .text section from code object\n");
    return 1;
  }
  printf("      .text section: %zu bytes at ELF offset 0x%lx\n",
         textSection.bytes.size(), textSection.offset);

  // ── Step 3: Run MLIR pipeline (gfx942 -> gfx942) ──
  printf("\n[3/7] Lifting to waveasm MLIR and re-emitting assembly\n");
  printf("      Pipeline: gfx942 (binary) → waveasm IR → gfx942 (assembly)\n");
  auto result =
      hotswap::runPipeline(textSection.bytes.data(), textSection.bytes.size(),
                           "gfx942", "gfx942", "vecadd");

  if (!result.success) {
    fprintf(stderr, "FATAL: Pipeline failed: %s\n",
            result.errorMessage.c_str());
    return 1;
  }

  printf("      ┌──────────────────────────────────────┐\n");
  printf("      │ Total instructions:   %4lu            │\n",
         result.stats.totalInstructions);
  printf("      │ Lifted (typed ops):   %4lu            │\n",
         result.stats.liftedInstructions);
  printf("      │ Raw fallbacks:        %4lu            │\n",
         result.stats.rawFallbacks);
  printf("      │ Failed disassembly:   %4lu            │\n",
         result.stats.failedDisassembly);
  printf("      └──────────────────────────────────────┘\n");

  if (result.stats.rawFallbacks > 0) {
    fprintf(stderr,
            "      WARNING: %lu instructions fell back to waveasm.raw\n",
            result.stats.rawFallbacks);
  }

  // Count non-trivial instructions (not s_nop)
  size_t nontrivialCount = 0;
  {
    std::string line;
    for (char c : result.assemblyText) {
      if (c == '\n') {
        auto first = line.find_first_not_of(" \t");
        if (first != std::string::npos) {
          auto trimmed = line.substr(first);
          if (!trimmed.empty() && trimmed.find("s_nop") != 0)
            nontrivialCount++;
        }
        line.clear();
      } else {
        line += c;
      }
    }
  }
  printf("      Non-NOP instructions: %zu\n", nontrivialCount);

  // ── Step 4: Read device assembly template and splice ──
  printf("\n[4/7] Splicing re-emitted instructions into assembly template\n");
  auto deviceAsmTemplate = hotswap::readFileAsString(MVE_DEVICE_ASM_PATH);
  if (deviceAsmTemplate.empty()) {
    fprintf(stderr, "FATAL: Cannot read device assembly template %s\n",
            MVE_DEVICE_ASM_PATH);
    return 1;
  }

  auto splicedAsm = hotswap::spliceInstructions(
      deviceAsmTemplate, result.assemblyText, KERNEL_SYMBOL);
  if (splicedAsm.empty()) {
    fprintf(stderr, "FATAL: Splicing failed\n");
    return 1;
  }
  printf("      Template + new instructions: %zu bytes of assembly\n",
         splicedAsm.size());

  // ── Step 5: Rebuild code object ──
  printf("\n[5/7] Assembling new code object (llvm-mc → ld.lld)\n");
  auto newCO = hotswap::rebuildCodeObject(splicedAsm, "gfx942", LLVM_BIN_DIR);
  if (newCO.empty()) {
    fprintf(stderr, "FATAL: Code object rebuild failed\n");
    fprintf(stderr, "\n--- Spliced assembly ---\n%s\n--- End ---\n",
            splicedAsm.c_str());
    return 1;
  }
  printf("      Original code object:  %5zu bytes\n", coBytes.size());
  printf("      Rebuilt code object:   %5zu bytes\n", newCO.size());
  if (newCO.size() != coBytes.size())
    printf("      (Different size confirms this is a FRESH binary, not a copy)\n");

  // ── Step 6: Load on GPU ──
  printf("\n[6/7] Loading rebuilt code object on GPU\n");
  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, newCO.data()));

  hipFunction_t func;
  HIP_CHECK(hipModuleGetFunction(&func, mod, KERNEL_SYMBOL));
  printf("      Kernel function: %s\n", KERNEL_SYMBOL);
  printf("      hipModuleLoadData:    SUCCESS\n");
  printf("      hipModuleGetFunction: SUCCESS\n");

  // ── Step 7: Execute and verify ──
  const int N = 1024;
  printf("\n[7/7] Executing on GPU and verifying all %d elements\n", N);
  printf("      Kernel: C[i] = A[i] + B[i]\n");
  printf("      Input:  A[i] = i,  B[i] = i*2\n");
  printf("      Expect: C[i] = i + i*2 = 3*i\n\n");

  std::vector<float> h_A(N), h_B(N), h_C(N);
  for (int i = 0; i < N; i++) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 2);
    h_C[i] = -1.0f;
  }

  float *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc(&d_A, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), N * sizeof(float), hipMemcpyHostToDevice));

  struct {
    float *A;
    float *B;
    float *C;
    int N;
  } args = {d_A, d_B, d_C, N};
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  unsigned blocks = (N + 63) / 64;
  HIP_CHECK(hipModuleLaunchKernel(func, blocks, 1, 1, 64, 1, 1, 0, nullptr,
                                  nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost));

  int passCount = 0;
  int failCount = 0;
  for (int i = 0; i < N; i++) {
    float expected = static_cast<float>(i) + static_cast<float>(i * 2);
    if (h_C[i] == expected) {
      passCount++;
    } else {
      failCount++;
      if (failCount <= 5)
        fprintf(stderr, "      MISMATCH [%d]: got %.1f, expected %.1f\n", i,
                h_C[i], expected);
    }
  }

  // Print a sample of verified results
  printf("      Sample results:\n");
  int sampleIndices[] = {0, 1, 2, N / 2, N - 2, N - 1};
  for (int idx : sampleIndices) {
    if (idx < N)
      printf("        C[%4d] = %8.1f  (expected %8.1f)  %s\n", idx,
             h_C[idx], static_cast<float>(idx) + static_cast<float>(idx * 2),
             h_C[idx] == static_cast<float>(idx) + static_cast<float>(idx * 2)
                 ? "OK"
                 : "FAIL");
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
  HIP_CHECK(hipModuleUnload(mod));

  printf("\n════════════════════════════════════════════════════════════════\n");
  if (failCount == 0) {
    printf("  RESULT: PASS  (%d/%d elements correct)\n", passCount, N);
    printf("\n  The MLIR pipeline produced a functionally correct kernel.\n");
    printf("  Binary was lifted, re-emitted, reassembled, and executed\n");
    printf("  on GPU hardware — output matches expected values exactly.\n");
  } else {
    printf("  RESULT: FAIL  (%d/%d elements wrong)\n", failCount, N);
  }
  printf("════════════════════════════════════════════════════════════════\n");

  return failCount > 0 ? 1 : 0;
}
