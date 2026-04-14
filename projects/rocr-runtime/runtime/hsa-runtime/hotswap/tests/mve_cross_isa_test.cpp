////////////////////////////////////////////////////////////////////////////////
//
// Cross-ISA GPU Round-Trip Test
//
// Demonstrates cross-architecture binary translation by:
//   1. Reading a gfx1250 (RDNA4) vecadd kernel code object
//   2. Extracting .text bytes (raw RDNA4 instructions)
//   3. Running them through the MLIR pipeline (gfx1250 -> gfx942)
//      - Mnemonic remapping (GFX12 -> GFX9 names)
//      - GFX12 scheduling hints dropped (s_clause, s_delay_alu)
//      - scale_offset addressing lowered to explicit 64-bit address math
//      - Wave width translation (wave32 -> wave64)
//   4. Extracting the translated "core" computation
//   5. Splicing it into the gfx942 device assembly template
//   6. Reassembling into a fresh gfx942 code object
//   7. Loading on a gfx942 GPU and verifying correct output
//
////////////////////////////////////////////////////////////////////////////////

#include "code_object_builder.hpp"
#include "pipeline.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
  printf("  Cross-ISA Binary Translation: gfx1250 (RDNA4) → gfx942 (CDNA3)\n");
  printf("════════════════════════════════════════════════════════════════\n\n");

  {
    hipDevice_t dev;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&props, dev));
    const char *displayName =
        (props.name[0] != '\0') ? props.name : props.gcnArchName;
    printf("  Target GPU: %s (%d CUs, %.0f MB VRAM)\n\n",
           displayName, props.multiProcessorCount,
           props.totalGlobalMem / (1024.0 * 1024.0));
  }

  printf("  Goal: take a binary compiled ONLY for gfx1250 (RDNA4) and\n");
  printf("        make it run on gfx942 (CDNA3) via MLIR translation.\n\n");
  printf("────────────────────────────────────────────────────────────────\n\n");

  // ── Step 1: Read the gfx1250 code object ──
  printf("[1/8] Reading gfx1250 (RDNA4) code object\n");
  printf("      Path: %s\n", MVE_CO_1250_PATH);
  auto coBytes = hotswap::readFile(MVE_CO_1250_PATH);
  if (coBytes.empty()) {
    fprintf(stderr, "FATAL: Cannot read gfx1250 code object\n");
    return 1;
  }
  printf("      Size: %zu bytes\n", coBytes.size());

  // ── Step 2: Extract .text section ──
  printf("\n[2/8] Extracting .text section (raw RDNA4 instructions)\n");
  auto textSection = hotswap::extractTextSection(coBytes);
  if (!textSection.valid) {
    fprintf(stderr, "FATAL: Cannot extract .text section\n");
    return 1;
  }
  printf("      .text: %zu bytes at ELF offset 0x%lx\n",
         textSection.bytes.size(), textSection.offset);

  // ── Step 3: Run MLIR pipeline (gfx1250 -> gfx942) ──
  printf("\n[3/8] Running MLIR pipeline: gfx1250 → gfx942\n");
  printf("      Lift (RDNA4 binary) → waveasm IR → retarget → emit (CDNA3 asm)\n");
  auto result =
      hotswap::runPipeline(textSection.bytes.data(), textSection.bytes.size(),
                           "gfx1250", "gfx942", "vecadd");

  if (!result.success) {
    fprintf(stderr, "FATAL: Pipeline failed: %s\n",
            result.errorMessage.c_str());
    return 1;
  }

  printf("      ┌──────────────────────────────────────────────┐\n");
  printf("      │ Total instructions disassembled:  %4lu        │\n",
         result.stats.totalInstructions);
  printf("      │ Lifted to typed waveasm ops:      %4lu        │\n",
         result.stats.liftedInstructions);
  printf("      │ Raw fallbacks:                    %4lu        │\n",
         result.stats.rawFallbacks);
  printf("      │ Failed disassembly:               %4lu        │\n",
         result.stats.failedDisassembly);
  printf("      └──────────────────────────────────────────────┘\n");

  // ── Step 4: Extract the core computation ──
  printf("\n[4/8] Extracting translated core computation\n");
  printf("      (Skipping RDNA4 ABI preamble — gfx942 template provides its own)\n");
  auto core = hotswap::extractCoreFromPipelineOutput(result.assemblyText);
  if (core.empty()) {
    fprintf(stderr, "FATAL: Could not extract core from pipeline output\n");
    fprintf(stderr, "\n--- Full pipeline output ---\n%s\n--- End ---\n",
            result.assemblyText.c_str());
    return 1;
  }

  // Count and display the translated core lines
  size_t coreLineCount = 0;
  {
    std::string line;
    for (char c : core) {
      if (c == '\n') {
        auto first = line.find_first_not_of(" \t");
        if (first != std::string::npos && !line.substr(first).empty())
          coreLineCount++;
        line.clear();
      } else {
        line += c;
      }
    }
  }
  printf("      Translated core: %zu instruction lines\n", coreLineCount);
  printf("\n      Translated assembly:\n");
  printf("      ─────────────────────\n");
  {
    std::string line;
    for (char c : core) {
      if (c == '\n') {
        printf("      %s\n", line.c_str());
        line.clear();
      } else {
        line += c;
      }
    }
    if (!line.empty())
      printf("      %s\n", line.c_str());
  }
  printf("      ─────────────────────\n");

  // ── Step 5: Read gfx942 template and splice ──
  printf("\n[5/8] Splicing into gfx942 device assembly template\n");
  printf("      Template: %s\n", MVE_GFX942_TEMPLATE_PATH);
  auto gfx942Template = hotswap::readFileAsString(MVE_GFX942_TEMPLATE_PATH);
  if (gfx942Template.empty()) {
    fprintf(stderr, "FATAL: Cannot read gfx942 template\n");
    return 1;
  }

  auto splicedAsm = hotswap::spliceCoreInstructions(
      gfx942Template, core, KERNEL_SYMBOL);
  if (splicedAsm.empty()) {
    fprintf(stderr, "FATAL: Core splicing failed\n");
    return 1;
  }
  printf("      Spliced assembly: %zu bytes\n", splicedAsm.size());

  // ── Step 6: Rebuild code object ──
  printf("\n[6/8] Assembling gfx942 code object (llvm-mc → ld.lld)\n");
  auto newCO = hotswap::rebuildCodeObject(splicedAsm, "gfx942", LLVM_BIN_DIR);
  if (newCO.empty()) {
    fprintf(stderr, "FATAL: Code object rebuild failed\n");
    fprintf(stderr, "\n--- Spliced assembly ---\n%s\n--- End ---\n",
            splicedAsm.c_str());
    return 1;
  }
  printf("      Source (gfx1250):    %5zu bytes\n", coBytes.size());
  printf("      Rebuilt (gfx942):    %5zu bytes\n", newCO.size());

  // ── Step 7: Load on GPU ──
  printf("\n[7/8] Loading translated code object on gfx942 GPU\n");
  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, newCO.data()));

  hipFunction_t func;
  HIP_CHECK(hipModuleGetFunction(&func, mod, KERNEL_SYMBOL));
  printf("      Kernel: %s\n", KERNEL_SYMBOL);
  printf("      hipModuleLoadData:    SUCCESS\n");
  printf("      hipModuleGetFunction: SUCCESS\n");

  // ── Step 8: Execute and verify ──
  const int N = 1024;
  printf("\n[8/8] Executing translated kernel — verifying %d elements\n", N);
  printf("      Kernel: C[i] = A[i] + B[i]\n");
  printf("      Input:  A[i] = i,  B[i] = i*2\n");
  printf("      Expect: C[i] = 3*i\n\n");

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
  HIP_CHECK(
      hipMemcpy(d_A, h_A.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_B, h_B.data(), N * sizeof(float), hipMemcpyHostToDevice));

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
  HIP_CHECK(
      hipMemcpy(h_C.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost));

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

  printf("      Sample results:\n");
  int sampleIndices[] = {0, 1, 2, N / 2, N - 2, N - 1};
  for (int idx : sampleIndices) {
    if (idx < N) {
      float expected = static_cast<float>(idx) + static_cast<float>(idx * 2);
      printf("        C[%4d] = %8.1f  (expected %8.1f)  %s\n", idx,
             h_C[idx], expected, h_C[idx] == expected ? "OK" : "FAIL");
    }
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
  HIP_CHECK(hipModuleUnload(mod));

  printf("\n════════════════════════════════════════════════════════════════\n");
  if (failCount == 0) {
    printf("  RESULT: PASS  (%d/%d elements correct)\n\n", passCount, N);
    printf("  A gfx1250 (RDNA4) binary was successfully translated to\n");
    printf("  gfx942 (CDNA3) via the MLIR pipeline and executed on GPU\n");
    printf("  hardware. All output values match expected results exactly.\n");
  } else {
    printf("  RESULT: FAIL  (%d/%d elements wrong)\n", failCount, N);
  }
  printf("════════════════════════════════════════════════════════════════\n");

  return failCount > 0 ? 1 : 0;
}
