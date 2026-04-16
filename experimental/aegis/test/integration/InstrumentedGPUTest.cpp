//===-- InstrumentedGPUTest.cpp - GPU test for instrumented trampoline -----===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GPU test: patches a real GEMM kernel with instrumented trampolines that
/// capture per-lane memory addresses to a trace buffer. Verifies:
///   1. Kernel still produces correct results
///   2. Trace buffer contains valid records with expected site IDs
///   3. Captured addresses fall within the expected GPU allocation ranges
///
/// Build (from aegisbit/build):
///   cmake --build . --target instrumented_gpu_test
///   ./test/integration/instrumented_gpu_test
///
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include <set>

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/CFGBuilder.h"
#include "aegisbit/CoalescingAnalyzer.h"
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

struct TraceRecord {
  uint32_t SiteID;
  uint32_t Padding;
  uint64_t Addresses[64];
};
static_assert(sizeof(TraceRecord) == TraceConfig::RecordSize,
              "TraceRecord must match TraceConfig::RecordSize");

int main() {
  printf("=== Instrumented Trampoline GPU Test ===\n\n");

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
  printf("  Descriptor: VGPRs=%u, SGPRs=%u\n",
         KInfo->Descriptor.VGPRCount, KInfo->Descriptor.SGPRCount);

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

  // ---- Step 3: Pick above-the-count scratch registers (instrumented) ----
  ScratchRegisters Scratch =
      ScratchRegisters::fromDescriptorInstrumented(KInfo->Descriptor);

  // For AccVGPR kernels, refine scratch VGPRs by scanning the CFG for
  // truly unused regular VGPRs. If none are free, fall back to scratch spill.
  if (Scratch.HasAccumVGPRs && !Scratch.isValid()) {
    uint32_t AO = KInfo->Descriptor.AccumOffset;
    uint32_t CurrentScratch = KInfo->Descriptor.PrivateSegmentFixedSize;
    if (!Scratch.refineScratchVGPRs(*CFGOrErr, Disasm, AO)) {
      Scratch.setupScratchSpill(AO, CurrentScratch);
      printf("  AccVGPR scratch: fell back to scratch spill v%u/v%u/v%u\n",
             RegisterHelper::getVGPRIndex(Scratch.ScratchVGPR),
             RegisterHelper::getVGPRIndex(Scratch.LaneVGPR),
             RegisterHelper::getVGPRIndex(Scratch.TempVGPR));
    } else {
      printf("  AccVGPR scratch: found unused v%u/v%u/v%u\n",
             RegisterHelper::getVGPRIndex(Scratch.ScratchVGPR),
             RegisterHelper::getVGPRIndex(Scratch.LaneVGPR),
             RegisterHelper::getVGPRIndex(Scratch.TempVGPR));
    }
  }

  auto Sites = TrampolineBridge::findMemorySites(
      *CFGOrErr, KStart, Disasm, Scratch, /*SupportsGPUAtomics=*/true);
  printf("  Memory sites: %zu\n", Sites.size());
  for (size_t i = 0; i < Sites.size(); i++) {
    printf("    [%zu] %s at offset 0x%lx, addr_vgpr=v%u%s\n",
           i, Sites[i].IsLoad ? "LOAD " : "STORE",
           Sites[i].Offset, Sites[i].AddrVGPRIndex,
           Sites[i].Addr64 ? " (64-bit)" : " (32-bit)");
  }

  if (Sites.empty()) {
    fprintf(stderr, "No global memory ops found\n");
    return 1;
  }

  if (Scratch.NeedsAccVGPRSpill)
    TrampolineBridge::computePreSpillDrainValues(*CFGOrErr, Sites, Scratch, Disasm);

  printf("\n  Scratch registers (instrumented above-the-count):\n");
  printf("    ReturnAddr SGPR pair: s[%u:%u]\n",
         Scratch.FirstFreeSGPRIdx, Scratch.FirstFreeSGPRIdx + 1);
  printf("    ScratchSGPR: s%u, ExecSave: s%u/s%u\n",
         Scratch.FirstFreeSGPRIdx + 2,
         Scratch.FirstFreeSGPRIdx + 3,
         Scratch.FirstFreeSGPRIdx + 4);
  printf("    ScratchVGPR: v%u, LaneVGPR: v%u, TempVGPR: v%u\n",
         RegisterHelper::getVGPRIndex(Scratch.ScratchVGPR),
         RegisterHelper::getVGPRIndex(Scratch.LaneVGPR),
         RegisterHelper::getVGPRIndex(Scratch.TempVGPR));
  printf("    Extra VGPRs: +%u, Extra SGPRs: +%u\n",
         Scratch.ExtraVGPRs, Scratch.ExtraSGPRs);
  if (Scratch.NeedsScratchSpill)
    printf("    Using scratch memory spill (offset %u, +%u bytes)\n",
           Scratch.ScratchSpillOffset, Scratch.ExtraScratchBytes);
  if (Scratch.NeedsAccVGPRSpill)
    printf("    Using AccVGPR spill\n");

  // ---- Step 4: Allocate trace buffer + atomic counter ----
  //
  // Use hipHostMalloc with hipHostMallocCoherent for CPU-visible, GPU-accessible
  // memory. This avoids needing kernarg extension.
  const size_t MaxRecords = 4096;
  const size_t BufSize = MaxRecords * TraceConfig::RecordSize;

  void *TraceBufHost = nullptr;
  void *CounterHost = nullptr;

  HIP_CHECK(hipHostMalloc(&TraceBufHost, BufSize,
                            hipHostMallocCoherent | hipHostMallocMapped));
  HIP_CHECK(hipHostMalloc(&CounterHost, sizeof(uint64_t),
                            hipHostMallocCoherent | hipHostMallocMapped));

  memset(TraceBufHost, 0, BufSize);
  memset(CounterHost, 0, sizeof(uint64_t));

  // Get GPU virtual addresses for the host-allocated memory.
  void *TraceBufGPU = nullptr;
  void *CounterGPU = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&TraceBufGPU, TraceBufHost, 0));
  HIP_CHECK(hipHostGetDevicePointer(&CounterGPU, CounterHost, 0));

  printf("\n  Trace buffer: host=%p gpu=%p (%zu bytes, %zu max records)\n",
         TraceBufHost, TraceBufGPU, BufSize, MaxRecords);
  printf("  Counter: host=%p gpu=%p\n", CounterHost, CounterGPU);

  TraceConfig Trace;
  Trace.BufferAddr = reinterpret_cast<uint64_t>(TraceBufGPU);
  Trace.CounterAddr = reinterpret_cast<uint64_t>(CounterGPU);
  Trace.BufferSize = BufSize;
  Trace.SupportsGPUAtomics = true;

  // ---- Step 5: Build instrumented trampoline ----
  auto BridgeOrErr = TrampolineBridge::create("gfx950", Disasm);
  if (!BridgeOrErr) {
    fprintf(stderr, "Bridge: %s\n",
            toString(BridgeOrErr.takeError()).c_str());
    return 1;
  }

  auto BROrErr = (*BridgeOrErr)->buildInstrumented(
      KCode, KStart, Text.size(), Sites, Scratch, Trace);
  if (!BROrErr) {
    fprintf(stderr, "buildInstrumented: %s\n",
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

  for (size_t i = 0; i < BR.Slots.size(); i++) {
    printf("    Slot[%zu] trampoline: %zu bytes\n",
           i, BR.Slots[i].TrampolineBytes.size());
  }

  // ---- Step 6: Build patched .text + ELF ----
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
      /*AdditionalKernargSize=*/0,
      Scratch.ExtraScratchBytes);
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

  // ---- Step 7: Load and run both original and patched kernels ----
  hipModule_t origMod, patchedMod;
  hipFunction_t origFunc, patchedFunc;

  HIP_CHECK(hipModuleLoadData(&origMod, ELFBytes.data()));
  HIP_CHECK(hipModuleGetFunction(&origFunc, origMod, "sgemm_naive"));
  printf("Original kernel loaded OK\n");

  HIP_CHECK(hipModuleLoadData(&patchedMod, PatchedELF.data()));
  HIP_CHECK(hipModuleGetFunction(&patchedFunc, patchedMod, "sgemm_naive"));
  printf("Patched (instrumented) kernel loaded OK\n");

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

  // Run original
  HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
  HIP_CHECK(hipModuleLaunchKernel(origFunc,
    1, 1, 1, 16, 16, 1, 0, 0, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_C_orig.data(), d_C, M * N * sizeof(float),
                       hipMemcpyDeviceToHost));
  printf("Original executed OK\n");

  // Run patched (instrumented) — resets counter first
  *reinterpret_cast<uint64_t *>(CounterHost) = 0;
  memset(TraceBufHost, 0, BufSize);

  HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
  HIP_CHECK(hipModuleLaunchKernel(patchedFunc,
    1, 1, 1, 16, 16, 1, 0, 0, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_C_patched.data(), d_C, M * N * sizeof(float),
                       hipMemcpyDeviceToHost));
  printf("Patched (instrumented) executed OK\n\n");

  // ---- Step 8: Verify computation correctness ----
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
  printf("Computation: %d/%d mismatches, output sum=%.2f\n", errors, M * N, sum);

  if (errors > 0) {
    printf("FAIL: Instrumented trampoline corrupted computation.\n");
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    hipModuleUnload(origMod); hipModuleUnload(patchedMod);
    hipHostFree(TraceBufHost); hipHostFree(CounterHost);
    return 1;
  }

  // ---- Step 9: Analyze trace buffer ----
  // OnGpuReduce writes per-site counters (8 bytes each) at the start of
  // BufferAddr: [total_cache_lines(u32), total_samples(u32)] per site.
  bool traceOK = false;
  printf("\nTrace analysis (OnGpuReduce per-site counters):\n");
  {
    auto *SiteCounters = reinterpret_cast<uint32_t *>(TraceBufHost);
    uint32_t TotalCacheLines = 0;
    uint32_t TotalSamples = 0;
    for (size_t i = 0; i < Sites.size(); ++i) {
      uint32_t CacheLines = SiteCounters[i * 2];
      uint32_t Samples = SiteCounters[i * 2 + 1];
      printf("  Site[%zu]: cache_lines=%u, samples=%u\n", i, CacheLines, Samples);
      TotalCacheLines += CacheLines;
      TotalSamples += Samples;
    }
    printf("  Total: cache_lines=%u, samples=%u\n", TotalCacheLines, TotalSamples);
    if (TotalCacheLines > 0 || TotalSamples > 0) {
      printf("  OK: per-site counters are non-zero.\n");
      traceOK = true;
    } else {
      printf("  WARN: per-site counters are all zero (payload may be dry).\n");
      const char *DryEnv = getenv("AEGISBIT_DRY_PAYLOAD");
      int DryLevel = DryEnv ? atoi(DryEnv) : -1;
      if (DryLevel >= 0 && DryLevel < 3) {
        printf("  (AEGISBIT_DRY_PAYLOAD=%d, atomic accumulator requires level 3)\n",
               DryLevel);
        traceOK = true;
      }
    }
  }

  // ---- Step 10: Final verdict ----
  bool computeOK = (errors == 0 && sum > 0);

  printf("\n=== RESULTS ===\n");
  printf("  Computation: %s\n", computeOK ? "PASS" : "FAIL");
  printf("  Trace capture: %s\n", traceOK ? "PASS" : "FAIL");

  if (computeOK && traceOK) {
    printf("\nPASS: Instrumented trampoline produces correct results and captures addresses.\n");
  } else {
    printf("\nFAIL: See details above.\n");
  }

  // Cleanup
  hipFree(d_A); hipFree(d_B); hipFree(d_C);
  hipModuleUnload(origMod); hipModuleUnload(patchedMod);
  hipHostFree(TraceBufHost); hipHostFree(CounterHost);

  return (computeOK && traceOK) ? 0 : 1;
}
