//===-- AboveTheCountGPUTest.cpp - GPU test for above-the-count regs ------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GPU test: patches a real GEMM kernel using above-the-count registers
/// (no liveness analysis, no RegScavenger). Loads original and patched
/// ELFs via hipModule, dispatches both, compares results.
///
/// Build (from aegisbit/build):
///   cmake --build . --target above_the_count_gpu_test
///   ./test/integration/above_the_count_gpu_test
///
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/CFGBuilder.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/TrampolineBridge.h"

#include "fixtures/gemm_gfx950_elf.h"

using namespace aegisbit;
using namespace llvm;

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
              hipGetErrorString(err)); \
      return 1; \
    } \
  } while(0)

int main() {
  printf("=== Above-The-Count GPU Test ===\n\n");

  // ---- Step 1: Parse ELF ----
  ArrayRef<uint8_t> ELFBytes(gemm_gfx950_elf, gemm_gfx950_elf_len);

  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(ELFBytes);
  if (!HandlerOrErr) {
    fprintf(stderr, "Failed to parse ELF: %s\n",
            toString(HandlerOrErr.takeError()).c_str());
    return 1;
  }
  auto &Handler = *HandlerOrErr;

  const KernelInfo *KInfo = Handler.getKernel("sgemm_naive");
  if (!KInfo) {
    fprintf(stderr, "sgemm_naive not found\n");
    return 1;
  }

  printf("Kernel: sgemm_naive\n");
  printf("  Code: %lu bytes at offset 0x%lx\n", KInfo->CodeSize, KInfo->CodeOffset);
  printf("  Descriptor: VGPRs=%u, SGPRs=%u, Granularity=%u\n",
         KInfo->Descriptor.VGPRCount, KInfo->Descriptor.SGPRCount,
         KInfo->Descriptor.VGPRGranularity);

  ArrayRef<uint8_t> Text = Handler.getTextSection();
  uint64_t KStart = KInfo->CodeOffset;
  uint64_t KEnd = std::min(KStart + KInfo->CodeSize,
                            static_cast<uint64_t>(Text.size()));
  ArrayRef<uint8_t> KCode = Text.slice(KStart, KEnd - KStart);

  // ---- Step 2: Build CFG + find memory sites ----
  auto DisasmOrErr = Disassembler::create("amdgcn-amd-amdhsa", "gfx950",
                                           "+wavefrontsize64");
  if (!DisasmOrErr) {
    fprintf(stderr, "Disassembler: %s\n",
            toString(DisasmOrErr.takeError()).c_str());
    return 1;
  }
  auto &Disasm = **DisasmOrErr;

  CFGBuilder CfgB(Disasm);
  auto CFGOrErr = CfgB.build(KCode, KStart);
  if (!CFGOrErr) {
    fprintf(stderr, "CFG: %s\n", toString(CFGOrErr.takeError()).c_str());
    return 1;
  }

  auto Sites = TrampolineBridge::findMemorySites(*CFGOrErr, KStart, Disasm);
  printf("  Memory sites: %zu\n", Sites.size());

  for (size_t i = 0; i < Sites.size(); i++) {
    printf("    [%zu] %s at offset 0x%lx (%lu bytes)\n",
           i, Sites[i].IsLoad ? "LOAD " : "STORE",
           Sites[i].Offset, Sites[i].OrigInstSize);
  }

  if (Sites.empty()) {
    fprintf(stderr, "No global memory ops found\n");
    return 1;
  }

  // ---- Step 3: Pick above-the-count scratch registers ----
  ScratchRegisters Scratch =
      ScratchRegisters::fromDescriptor(KInfo->Descriptor);

  printf("\n  Scratch registers (above-the-count):\n");
  printf("    Return addr SGPR pair: s[%u:%u]\n",
         KInfo->Descriptor.SGPRCount, KInfo->Descriptor.SGPRCount + 1);
  printf("    Scratch VGPR: v%u\n", KInfo->Descriptor.VGPRCount);
  printf("    Extra VGPRs: +%u, Extra SGPRs: +%u\n",
         Scratch.ExtraVGPRs, Scratch.ExtraSGPRs);

  // ---- Step 4: Build trampoline ----
  auto BridgeOrErr = TrampolineBridge::create("gfx950", Disasm);
  if (!BridgeOrErr) {
    fprintf(stderr, "Bridge: %s\n",
            toString(BridgeOrErr.takeError()).c_str());
    return 1;
  }

  auto BROrErr = (*BridgeOrErr)->buildEmpty(
      KCode, KStart, Text.size(), Sites, Scratch);
  if (!BROrErr) {
    fprintf(stderr, "buildEmpty: %s\n",
            toString(BROrErr.takeError()).c_str());
    return 1;
  }
  auto &BR = *BROrErr;

  {
    uint64_t TotalBytes = 0;
    for (const auto &Isl : BR.Islands) TotalBytes += Isl.Bytes.size();
    printf("\n  Trampoline: %u sites patched, %zu island(s), %lu total bytes\n",
           BR.PatchedCount, BR.Islands.size(), TotalBytes);
  }

  if (BR.PatchedCount == 0) {
    fprintf(stderr, "No sites patched\n");
    return 1;
  }

  // ---- Step 5: Build patched .text + ELF ----
  std::vector<uint8_t> NewText(Text.begin(), Text.end());
  for (const auto &Slot : BR.Slots) {
    std::memcpy(NewText.data() + Slot.OriginalPC,
                Slot.PatchBytes.data(), Slot.PatchBytes.size());
  }
  for (const auto &Isl : BR.Islands) {
    if (Isl.Offset + Isl.Bytes.size() > NewText.size())
      NewText.resize(Isl.Offset + Isl.Bytes.size(), 0x00);
    std::memcpy(NewText.data() + Isl.Offset,
                Isl.Bytes.data(), Isl.Bytes.size());
  }

  auto Handler2OrErr = CodeObjectHandler::loadFromBytes(ELFBytes);
  if (!Handler2OrErr) {
    fprintf(stderr, "Handler2 failed: %s\n",
            toString(Handler2OrErr.takeError()).c_str());
    return 1;
  }
  Handler2OrErr->setTextSection(NewText);

  auto ApplyErr = Handler2OrErr->applyPatch(
      NewText, "sgemm_naive",
      Scratch.ExtraVGPRs,
      Scratch.ExtraSGPRs,
      /*AdditionalKernargSize=*/0);
  if (ApplyErr) {
    fprintf(stderr, "applyPatch: %s\n",
            toString(std::move(ApplyErr)).c_str());
    return 1;
  }

  auto OutputOrErr = Handler2OrErr->build();
  if (!OutputOrErr) {
    fprintf(stderr, "ELF build: %s\n",
            toString(OutputOrErr.takeError()).c_str());
    return 1;
  }
  auto &PatchedELF = *OutputOrErr;
  printf("  Patched ELF: %zu bytes\n\n", PatchedELF.size());

  // Dump patched ELF for inspection
  {
    FILE *f = fopen("/tmp/aegisbit_patched.elf", "wb");
    if (f) {
      fwrite(PatchedELF.data(), 1, PatchedELF.size(), f);
      fclose(f);
      printf("  Wrote patched ELF to /tmp/aegisbit_patched.elf\n");
    }
    // Also dump original
    f = fopen("/tmp/aegisbit_original.elf", "wb");
    if (f) {
      fwrite(ELFBytes.data(), ELFBytes.size(), 1, f);
      fclose(f);
      printf("  Wrote original ELF to /tmp/aegisbit_original.elf\n");
    }
  }

  // Dump patch bytes for debugging
  for (size_t i = 0; i < BR.Slots.size(); i++) {
    const auto &Slot = BR.Slots[i];
    printf("  Slot[%zu] at 0x%lx: patch=", i, Slot.OriginalPC);
    for (uint8_t b : Slot.PatchBytes) printf("%02x", b);
    printf(" tramp=");
    for (uint8_t b : Slot.TrampolineBytes) printf("%02x", b);
    printf("\n");
  }

  // ---- Step 6: Load both on GPU ----
  hipModule_t origMod, patchedMod;
  hipFunction_t origFunc, patchedFunc;

  HIP_CHECK(hipModuleLoadData(&origMod, ELFBytes.data()));
  HIP_CHECK(hipModuleGetFunction(&origFunc, origMod, "sgemm_naive"));
  printf("Original kernel loaded OK\n");

  HIP_CHECK(hipModuleLoadData(&patchedMod, PatchedELF.data()));
  HIP_CHECK(hipModuleGetFunction(&patchedFunc, patchedMod, "sgemm_naive"));
  printf("Patched kernel loaded OK\n");

  // ---- Step 7: Run both on 16x16x16 GEMM ----
  const int M = 16, N = 16, K = 16;
  std::vector<float> h_A(M * K), h_B(K * N);
  std::vector<float> h_C_orig(M * N), h_C_patched(M * N);

  for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(i % 7) * 0.1f;
  for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(i % 5) * 0.2f;

  float *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

  struct {
    const float *A;
    const float *B;
    float *C;
    int M, N, K;
  } args{d_A, d_B, d_C, M, N, K};

  size_t argSize = sizeof(args);
  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
    HIP_LAUNCH_PARAM_END
  };

  HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
  HIP_CHECK(hipModuleLaunchKernel(origFunc,
    1, 1, 1, 16, 16, 1, 0, 0, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_C_orig.data(), d_C, M * N * sizeof(float),
                       hipMemcpyDeviceToHost));
  printf("Original executed OK\n");

  HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
  HIP_CHECK(hipModuleLaunchKernel(patchedFunc,
    1, 1, 1, 16, 16, 1, 0, 0, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_C_patched.data(), d_C, M * N * sizeof(float),
                       hipMemcpyDeviceToHost));
  printf("Patched executed OK\n");

  // ---- Step 8: Compare results ----
  int errors = 0;
  for (int i = 0; i < M * N; i++) {
    if (std::abs(h_C_orig[i] - h_C_patched[i]) > 1e-5f) {
      if (errors < 10) {
        printf("  MISMATCH [%d]: orig=%.6f patched=%.6f\n",
               i, h_C_orig[i], h_C_patched[i]);
      }
      errors++;
    }
  }

  float sum = 0;
  for (int i = 0; i < M * N; i++) sum += std::abs(h_C_orig[i]);
  printf("\nOriginal output sum: %.2f (should be > 0)\n", sum);
  printf("\n=== RESULT: %d/%d mismatches ===\n", errors, M * N);

  if (errors == 0 && sum > 0) {
    printf("PASS: Above-the-count trampoline produces identical results.\n");
  } else if (sum == 0) {
    printf("FAIL: Original kernel produced all zeros.\n");
  } else {
    printf("FAIL: Trampoline corrupted execution.\n");
  }

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
  hipModuleUnload(origMod);
  hipModuleUnload(patchedMod);

  return (errors == 0 && sum > 0) ? 0 : 1;
}
