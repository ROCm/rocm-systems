//===-- SwapPCDispatchGPUTest.cpp - GPU dispatch test for SwapPC mode ------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GPU integration test: patches the mega_gather kernel using both SharedBody
/// and SwapPC trampoline strategies, dispatches both on real GPU hardware, and
/// asserts both complete without hanging or crashing.
///
/// This is a regression test for the SwapPC codegen bug where the patched
/// mega_gather kernel (512 VGPRs, AccumOffset=256, ~110KB code) crashes at
/// dispatch time despite correct static ELF layout.
///
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/KernelPatcher.h"
#include "aegisbit/Types.h"

using namespace aegisbit;
using namespace llvm;

#ifndef MEGA_GATHER_FIXTURE_PATH
#error "MEGA_GATHER_FIXTURE_PATH must be defined at compile time"
#endif

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(err));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static std::vector<uint8_t> loadFixture(const char *Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F)
    return {};
  auto Size = F.tellg();
  F.seekg(0);
  std::vector<uint8_t> Bytes(Size);
  F.read(reinterpret_cast<char *>(Bytes.data()), Size);
  return Bytes;
}

struct PatchResult {
  std::vector<uint8_t> PatchedELF;
  void *TraceBufHost = nullptr;
  void *CounterHost = nullptr;

  ~PatchResult() {
    if (TraceBufHost) (void)hipHostFree(TraceBufHost);
    if (CounterHost) (void)hipHostFree(CounterHost);
  }
  PatchResult() = default;
  PatchResult(PatchResult &&O) noexcept
      : PatchedELF(std::move(O.PatchedELF)),
        TraceBufHost(O.TraceBufHost), CounterHost(O.CounterHost) {
    O.TraceBufHost = nullptr;
    O.CounterHost = nullptr;
  }
  PatchResult &operator=(PatchResult &&) = delete;
  PatchResult(const PatchResult &) = delete;
};

static Expected<PatchResult>
patchKernel(const std::vector<uint8_t> &ELFBytes, const char *GPUArch,
            bool ForceSwapPC) {
  auto Patcher = KernelPatcher::create(GPUArch);
  if (!Patcher)
    return Patcher.takeError();

  auto Handler = CodeObjectHandler::loadFromBytes(ELFBytes);
  if (!Handler)
    return Handler.takeError();

  auto Names = Handler->getKernelNames();
  if (Names.empty())
    return createStringError(inconvertibleErrorCode(), "No kernels in ELF");

  const KernelInfo *KI = Handler->getKernel(Names[0]);
  if (!KI)
    return createStringError(inconvertibleErrorCode(), "Kernel not found");

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 1;
  CodeObj.Bytes = ELFBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 1;
  Symbol.CodeObjectId = 1;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  PatchResult PR;
  if (hipHostMalloc(&PR.TraceBufHost, 32768,
                    hipHostMallocCoherent | hipHostMallocMapped) != hipSuccess)
    return createStringError(inconvertibleErrorCode(), "hipHostMalloc trace");
  if (hipHostMalloc(&PR.CounterHost, 64,
                    hipHostMallocCoherent | hipHostMallocMapped) != hipSuccess)
    return createStringError(inconvertibleErrorCode(), "hipHostMalloc counter");
  memset(PR.TraceBufHost, 0, 32768);
  memset(PR.CounterHost, 0, 64);

  void *TraceBufGPU = nullptr;
  void *CounterGPU = nullptr;
  if (hipHostGetDevicePointer(&TraceBufGPU, PR.TraceBufHost, 0) != hipSuccess)
    return createStringError(inconvertibleErrorCode(), "hipHostGetDevicePointer");
  if (hipHostGetDevicePointer(&CounterGPU, PR.CounterHost, 0) != hipSuccess)
    return createStringError(inconvertibleErrorCode(), "hipHostGetDevicePointer");

  TraceConfig Trace;
  Trace.BufferAddr = reinterpret_cast<uint64_t>(TraceBufGPU);
  Trace.CounterAddr = reinterpret_cast<uint64_t>(CounterGPU);
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  if (ForceSwapPC)
    setenv("AEGISBIT_FORCE_SWAPPC", "1", 1);
  else
    unsetenv("AEGISBIT_FORCE_SWAPPC");

  setenv("AEGISBIT_MAX_SITES", "1", 1);

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  if (!ResultOrErr)
    return ResultOrErr.takeError();

  const PatchedKernel *Result = *ResultOrErr;
  if (!Result || Result->PatchedELF.empty())
    return createStringError(inconvertibleErrorCode(), "Empty patched ELF");

  PR.PatchedELF = Result->PatchedELF;
  unsetenv("AEGISBIT_FORCE_SWAPPC");
  unsetenv("AEGISBIT_MAX_SITES");

  return std::move(PR);
}

static int dispatchKernel(hipFunction_t Func, int Blocks, int Threads,
                          float *dA, float *dB, float *dC, float *dD,
                          float *dOut, int Stride) {
  struct {
    const float *A;
    const float *B;
    const float *C;
    const float *D;
    float *Out;
    int Stride;
  } Args{dA, dB, dC, dD, dOut, Stride};

  size_t ArgSize = sizeof(Args);
  void *Config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &Args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &ArgSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(
      hipModuleLaunchKernel(Func, Blocks, 1, 1, Threads, 1, 1, 0, 0, nullptr,
                            Config));
  HIP_CHECK(hipDeviceSynchronize());
  return 0;
}

int main(int argc, char *argv[]) {
  setbuf(stdout, NULL);
  setbuf(stderr, NULL);
  printf("=== SwapPC GPU Dispatch Regression Test ===\n");

  // Parse optional --swappc-only flag to skip SharedBody test and only run
  // SwapPC. Useful because AEGISBIT_FORCE_SWAPPC is a static-init env var.
  bool SwapPCOnly = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--swappc-only") == 0)
      SwapPCOnly = true;
  }

  int DeviceCount = 0;
  hipError_t Err = hipGetDeviceCount(&DeviceCount);
  if (Err != hipSuccess || DeviceCount == 0) {
    printf("SKIP: No HIP devices available\n");
    return 0;
  }
  HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t Props{};
  HIP_CHECK(hipGetDeviceProperties(&Props, 0));
  printf("Device: %s\n", Props.name);

  // Load fixture ELF
  auto FixtureBytes = loadFixture(MEGA_GATHER_FIXTURE_PATH);
  if (FixtureBytes.empty()) {
    fprintf(stderr, "FAIL: Cannot load fixture: %s\n", MEGA_GATHER_FIXTURE_PATH);
    return 1;
  }
  printf("Fixture ELF: %zu bytes\n", FixtureBytes.size());

  // Allocate GPU buffers (same layout as mega_gather expects)
  constexpr int kBlocks = 256;
  constexpr int kThreads = 256;
  constexpr int kStride = 256;
  constexpr size_t kN = static_cast<size_t>(256) * 512 * 256;
  const size_t Bytes = kN * sizeof(float);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr,
        *dOut = nullptr;
  HIP_CHECK(hipMalloc(&dA, Bytes));
  HIP_CHECK(hipMalloc(&dB, Bytes));
  HIP_CHECK(hipMalloc(&dC, Bytes));
  HIP_CHECK(hipMalloc(&dD, Bytes));
  HIP_CHECK(hipMalloc(&dOut, Bytes));
  HIP_CHECK(hipMemset(dA, 0, Bytes));
  HIP_CHECK(hipMemset(dB, 0, Bytes));
  HIP_CHECK(hipMemset(dC, 0, Bytes));
  HIP_CHECK(hipMemset(dD, 0, Bytes));
  HIP_CHECK(hipMemset(dOut, 0, Bytes));
  printf("GPU buffers allocated: %zu MB each\n", Bytes / (1024 * 1024));

  const char *KernelName = "_ZN12_GLOBAL__N_111mega_gatherEPKfS1_S1_S1_Pfi";

  // ---- Test 1: Original kernel (no patching) ----
  printf("\n--- Test 1: Original (unpatched) kernel ---\n");
  {
    hipModule_t Mod;
    hipFunction_t Func;
    HIP_CHECK(hipModuleLoadData(&Mod, FixtureBytes.data()));
    HIP_CHECK(hipModuleGetFunction(&Func, Mod, KernelName));
    int Ret = dispatchKernel(Func, kBlocks, kThreads, dA, dB, dC, dD, dOut,
                             kStride);
    (void)hipModuleUnload(Mod);
    if (Ret != 0) {
      fprintf(stderr, "FAIL: Original kernel dispatch failed\n");
      goto cleanup;
    }
    printf("PASS: Original kernel executed OK\n");
  }

  // ---- Test 2: SharedBody patched kernel (1 site) ----
  if (!SwapPCOnly) {
  printf("\n--- Test 2: SharedBody patched kernel (1 site) ---\n");
  {
    auto PatchedOrErr = patchKernel(FixtureBytes, "gfx950", /*ForceSwapPC=*/false);
    if (!PatchedOrErr) {
      fprintf(stderr, "FAIL: SharedBody patching failed: %s\n",
              toString(PatchedOrErr.takeError()).c_str());
      goto cleanup;
    }
    printf("SharedBody patched ELF: %zu bytes\n", PatchedOrErr->PatchedELF.size());

    hipModule_t Mod;
    hipFunction_t Func;
    HIP_CHECK(hipModuleLoadData(&Mod, PatchedOrErr->PatchedELF.data()));
    HIP_CHECK(hipModuleGetFunction(&Func, Mod, KernelName));
    HIP_CHECK(hipMemset(dOut, 0, Bytes));
    int Ret = dispatchKernel(Func, kBlocks, kThreads, dA, dB, dC, dD, dOut,
                             kStride);
    (void)hipModuleUnload(Mod);
    if (Ret != 0) {
      fprintf(stderr, "FAIL: SharedBody kernel dispatch failed\n");
      goto cleanup;
    }
    printf("PASS: SharedBody kernel executed OK\n");
  }
  } // !SwapPCOnly

  // ---- Test 3: SwapPC patched kernel (1 site) — THE REGRESSION TEST ----
  printf("\n--- Test 3: SwapPC patched kernel (1 site) ---\n");
  {
    auto PatchedOrErr = patchKernel(FixtureBytes, "gfx950", /*ForceSwapPC=*/true);
    if (!PatchedOrErr) {
      fprintf(stderr, "FAIL: SwapPC patching failed: %s\n",
              toString(PatchedOrErr.takeError()).c_str());
      goto cleanup;
    }
    printf("SwapPC patched ELF: %zu bytes\n", PatchedOrErr->PatchedELF.size());

    hipModule_t Mod;
    hipFunction_t Func;
    HIP_CHECK(hipModuleLoadData(&Mod, PatchedOrErr->PatchedELF.data()));
    HIP_CHECK(hipModuleGetFunction(&Func, Mod, KernelName));
    HIP_CHECK(hipMemset(dOut, 0, Bytes));
    int Ret = dispatchKernel(Func, kBlocks, kThreads, dA, dB, dC, dD, dOut,
                             kStride);
    (void)hipModuleUnload(Mod);
    if (Ret != 0) {
      fprintf(stderr, "FAIL: SwapPC kernel dispatch CRASHED (this is the bug)\n");
      goto cleanup;
    }
    printf("PASS: SwapPC kernel executed OK\n");
  }

  printf("\n=== ALL TESTS PASSED ===\n");
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dOut);
  return 0;

cleanup:
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dOut);
  return 1;
}
