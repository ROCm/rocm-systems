//===-- HugeMemKernelGTest.cpp - Huge-memory kernel integration scaffold --===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GTest scaffold for the exact stress_huge_mem kernel used by the Python E2E
/// runner. For now this test is intentionally diagnostic-only: it prints kernel
/// and launch properties, runs the kernel, and reports HIP status without
/// asserting on the outcome.
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#ifdef AEGISBIT_HAS_GPU

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "aegisbit/CFGBuilder.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/JumpHeuristics.h"
#include "aegisbit/TrampolineBridge.h"

using namespace aegisbit;

namespace {

#define DO_PASS(arr, acc, base)                  \
  _Pragma("unroll")                              \
  for (int i = 0; i < 64; i++) {                 \
    acc += arr[tid + (base + i) * stride];       \
  }

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void mega_gather(const float* __restrict__ A,
                 const float* __restrict__ B,
                 const float* __restrict__ C,
                 const float* __restrict__ D,
                 float* __restrict__ out,
                 int stride) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
  float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
  float b4 = 0, b5 = 0, b6 = 0, b7 = 0;

  DO_PASS(A, a0,   0) DO_PASS(A, a1,  64) DO_PASS(A, a2, 128) DO_PASS(A, a3, 192)
  DO_PASS(A, a4, 256) DO_PASS(A, a5, 320) DO_PASS(A, a6, 384) DO_PASS(A, a7, 448)

  DO_PASS(B, b0,   0) DO_PASS(B, b1,  64) DO_PASS(B, b2, 128) DO_PASS(B, b3, 192)
  DO_PASS(B, b4, 256) DO_PASS(B, b5, 320) DO_PASS(B, b6, 384) DO_PASS(B, b7, 448)

  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  float c4 = 0, c5 = 0, c6 = 0, c7 = 0;
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  float d4 = 0, d5 = 0, d6 = 0, d7 = 0;

  DO_PASS(C, c0,   0) DO_PASS(C, c1,  64) DO_PASS(C, c2, 128) DO_PASS(C, c3, 192)
  DO_PASS(C, c4, 256) DO_PASS(C, c5, 320) DO_PASS(C, c6, 384) DO_PASS(C, c7, 448)

  DO_PASS(D, d0,   0) DO_PASS(D, d1,  64) DO_PASS(D, d2, 128) DO_PASS(D, d3, 192)
  DO_PASS(D, d4, 256) DO_PASS(D, d5, 320) DO_PASS(D, d6, 384) DO_PASS(D, d7, 448)

  float total = a0+a1+a2+a3+a4+a5+a6+a7 + b0+b1+b2+b3+b4+b5+b6+b7 +
                c0+c1+c2+c3+c4+c5+c6+c7 + d0+d1+d2+d3+d4+d5+d6+d7;

  #pragma unroll
  for (int i = 0; i < 32; i++) {
    out[tid + i * stride] = total + static_cast<float>(i);
  }
}

void printHipStatus(const char* label, hipError_t err) {
  std::cout << label << ": " << hipGetErrorName(err)
            << " (" << hipGetErrorString(err) << ")" << std::endl;
}

class HugeMemKernelGTest : public ::testing::Test {};

TEST_F(HugeMemKernelGTest, RunExactHugeMemKernelAndPrintDiagnostics) {
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    std::cout << "No HIP devices available; skipping diagnostic run." << std::endl;
    return;
  }

  err = hipSetDevice(0);
  if (err != hipSuccess) {
    printHipStatus("hipSetDevice failed", err);
    return;
  }

  hipDeviceProp_t props{};
  err = hipGetDeviceProperties(&props, 0);
  if (err == hipSuccess) {
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Total global mem: " << props.totalGlobalMem << " bytes" << std::endl;
  } else {
    printHipStatus("hipGetDeviceProperties failed", err);
  }

  constexpr int kThreadsPerBlock = 256;
  constexpr int kBlocks = 256;
  constexpr int kStride = 256;
  constexpr int kStoreSites = 32;
  constexpr int kUnrolledPassesPerArray = 8;
  constexpr int kIterationsPerPass = 64;
  constexpr int kArrays = 4;
  constexpr size_t kN = static_cast<size_t>(256) * 512 * 256;
  const size_t bytes = kN * sizeof(float);

  std::cout << "=== HugeMem kernel diagnostic ===" << std::endl;
  std::cout << "Kernel name: mega_gather" << std::endl;
  std::cout << "Blocks: " << kBlocks << " Threads/block: " << kThreadsPerBlock << std::endl;
  std::cout << "Stride: " << kStride << std::endl;
  std::cout << "Elements per buffer: " << kN << std::endl;
  std::cout << "Bytes per buffer: " << bytes << std::endl;
  std::cout << "Expected VMEM load sites: "
            << (kArrays * kUnrolledPassesPerArray * kIterationsPerPass)
            << std::endl;
  std::cout << "Expected store sites: " << kStoreSites << std::endl;
  std::cout << "Expected total memory sites: "
            << (kArrays * kUnrolledPassesPerArray * kIterationsPerPass + kStoreSites)
            << std::endl;
  const char* max_sites = std::getenv("AEGISBIT_MAX_SITES");
  std::cout << "AEGISBIT_MAX_SITES="
            << (max_sites ? max_sites : "<unset>") << std::endl;

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dOut = nullptr;

  err = hipMalloc(&dA, bytes);
  printHipStatus("hipMalloc(A)", err);
  if (err != hipSuccess) return;

  err = hipMalloc(&dB, bytes);
  printHipStatus("hipMalloc(B)", err);
  if (err != hipSuccess) return;

  err = hipMalloc(&dC, bytes);
  printHipStatus("hipMalloc(C)", err);
  if (err != hipSuccess) return;

  err = hipMalloc(&dD, bytes);
  printHipStatus("hipMalloc(D)", err);
  if (err != hipSuccess) return;

  err = hipMalloc(&dOut, bytes);
  printHipStatus("hipMalloc(Out)", err);
  if (err != hipSuccess) return;

  std::cout << "Allocations:" << std::endl;
  std::cout << "  A=" << dA << std::endl;
  std::cout << "  B=" << dB << std::endl;
  std::cout << "  C=" << dC << std::endl;
  std::cout << "  D=" << dD << std::endl;
  std::cout << "  Out=" << dOut << std::endl;

  err = hipMemset(dA, 0, bytes);
  printHipStatus("hipMemset(A)", err);
  err = hipMemset(dB, 0, bytes);
  printHipStatus("hipMemset(B)", err);
  err = hipMemset(dC, 0, bytes);
  printHipStatus("hipMemset(C)", err);
  err = hipMemset(dD, 0, bytes);
  printHipStatus("hipMemset(D)", err);
  err = hipMemset(dOut, 0, bytes);
  printHipStatus("hipMemset(Out)", err);

  mega_gather<<<kBlocks, kThreadsPerBlock>>>(dA, dB, dC, dD, dOut, kStride);
  printHipStatus("hipGetLastError after launch", hipGetLastError());

  err = hipDeviceSynchronize();
  printHipStatus("hipDeviceSynchronize", err);

  if (err == hipSuccess) {
    std::cout << "Kernel launched OK" << std::endl;
  } else {
    std::cout << "Kernel did not complete successfully; see HIP status above."
              << std::endl;
  }

  if (dA) (void)hipFree(dA);
  if (dB) (void)hipFree(dB);
  if (dC) (void)hipFree(dC);
  if (dD) (void)hipFree(dD);
  if (dOut) (void)hipFree(dOut);
}

}  // namespace

#else

TEST(HugeMemKernelGTest, GPUNotAvailable) {
  GTEST_SKIP() << "Built without GPU support";
}

#endif
