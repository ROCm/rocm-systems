// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file occupancy.cpp
/// @brief Optimize occupancy by tuning register usage.
///
/// Three kernels with the same work but different register pressure:
///   low_reg    — trivial, few VGPRs
///   high_reg   — many live values, high VGPRs (occupancy-limiting)
///   capped_reg — same body as high_reg + __launch_bounds__ to cap VGPRs
///
/// rocjitsu reports the *allocated* VGPRs per dispatch under RJ_LOG=1
/// (rounded up to the allocation granularity). This program independently
/// queries the compiler's register usage via hipFuncGetAttributes and computes
/// the register-limited occupancy from the CU limits declared in the config.
///
/// IMPORTANT: rocjitsu is a functional emulator; it does NOT compute occupancy.
/// The calculation below is the standard AMD register-limited occupancy formula
/// applied to the CU limits in configs/amdgpu_cdna4_kmd.json. LDS-limited
/// occupancy is not shown here (RJ_LOG does not surface LDS usage).

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>

// --- CU limits from configs/amdgpu_cdna4_kmd.json (emulated MI350X, gfx950) ---
namespace cu {
constexpr int max_waves_per_simd = 8;         // device.max_waves_per_simd
constexpr int simd_per_cu = 4;                // device.simd_per_cu
constexpr int wavefront_size = 64;            // device.wave_front_size
constexpr int wf_slots_per_cu = 32;           // cu.num_wf_slots (= simd_per_cu * max_waves_per_simd)
constexpr int vgpr_budget_per_simd = 512;     // cu.vgprs_per_wf (per-SIMD VGPR budget)
constexpr int vgpr_granularity = 8;           // VGPR allocation granularity
} // namespace cu

__global__ void low_reg(float *out, const float *in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] = in[i] * 2.0f;
}

__global__ void high_reg(float *out, const float *in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc[32];
#pragma unroll
  for (int k = 0; k < 32; k++)
    acc[k] = in[(i + k) % N] + k;
#pragma unroll
  for (int it = 0; it < 4; it++)
#pragma unroll
    for (int k = 0; k < 32; k++)
      acc[k] = acc[k] * acc[(k + 1) & 31] + acc[(k + 7) & 31];
  float s = 0;
#pragma unroll
  for (int k = 0; k < 32; k++)
    s += acc[k];
  if (i < N)
    out[i] = s;
}

// Same body as high_reg, but __launch_bounds__(maxThreadsPerBlock, minWavesPerSIMD)
// asks the compiler to fit 8 waves/SIMD -> it caps VGPRs to <= 512/8 = 64.
__global__ void __launch_bounds__(256, 8)
    capped_reg(float *out, const float *in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc[32];
#pragma unroll
  for (int k = 0; k < 32; k++)
    acc[k] = in[(i + k) % N] + k;
#pragma unroll
  for (int it = 0; it < 4; it++)
#pragma unroll
    for (int k = 0; k < 32; k++)
      acc[k] = acc[k] * acc[(k + 1) & 31] + acc[(k + 7) & 31];
  float s = 0;
#pragma unroll
  for (int k = 0; k < 32; k++)
    s += acc[k];
  if (i < N)
    out[i] = s;
}

static int round_up(int v, int g) { return ((v + g - 1) / g) * g; }

// Register-limited waves per SIMD, then occupancy vs the hardware max.
static int waves_per_simd_by_vgpr(int alloc_vgprs) {
  if (alloc_vgprs <= 0)
    return cu::max_waves_per_simd;
  return std::min(cu::max_waves_per_simd, cu::vgpr_budget_per_simd / alloc_vgprs);
}

static void analyze(const char *name, const void *fn) {
  hipFuncAttributes attr{};
  hipError_t e = hipFuncGetAttributes(&attr, fn);
  if (e != hipSuccess) {
    printf("  %-11s hipFuncGetAttributes failed: %s\n", name, hipGetErrorString(e));
    return;
  }
  int used = attr.numRegs;
  int alloc = round_up(used, cu::vgpr_granularity);
  int waves = waves_per_simd_by_vgpr(alloc);
  double occ = 100.0 * waves / cu::max_waves_per_simd;
  printf("  %-11s used_vgprs=%-3d alloc_vgprs=%-3d -> %d/%d waves/SIMD  (%5.1f%% occupancy)\n",
         name, used, alloc, waves, cu::max_waves_per_simd, occ);
}

int main() {
  const int N = 4096;
  const int blockSize = 256;
  const int gridSize = N / blockSize;

  float *d_in = nullptr, *d_out = nullptr;
  (void)hipMalloc(&d_in, N * sizeof(float));
  (void)hipMalloc(&d_out, N * sizeof(float));

  printf("Occupancy vs register usage - emulated MI350X CU (from amdgpu_cdna4_kmd.json):\n");
  printf("  %d SIMDs/CU, max %d waves/SIMD (= %d wave slots/CU), wave = %d threads,\n",
         cu::simd_per_cu, cu::max_waves_per_simd, cu::wf_slots_per_cu, cu::wavefront_size);
  printf("  VGPR budget/SIMD = %d, allocation granularity = %d\n",
         cu::vgpr_budget_per_simd, cu::vgpr_granularity);
  printf("  block = %d threads = %d waves/block\n\n", blockSize, blockSize / cu::wavefront_size);

  // Launch each kernel so RJ_LOG=1 prints its dispatch (alloc_vgprs == RJ_LOG vgprs).
  low_reg<<<gridSize, blockSize>>>(d_out, d_in, N);
  (void)hipDeviceSynchronize();
  high_reg<<<gridSize, blockSize>>>(d_out, d_in, N);
  (void)hipDeviceSynchronize();
  capped_reg<<<gridSize, blockSize>>>(d_out, d_in, N);
  (void)hipDeviceSynchronize();

  printf("Register-limited occupancy (computed on host; rocjitsu does NOT compute this):\n");
  analyze("low_reg", reinterpret_cast<const void *>(low_reg));
  analyze("high_reg", reinterpret_cast<const void *>(high_reg));
  analyze("capped_reg", reinterpret_cast<const void *>(capped_reg));

  printf("\nTakeaway: high_reg's register pressure drops occupancy below 100%%;\n");
  printf("__launch_bounds__ caps VGPRs and restores it. Cross-check alloc_vgprs\n");
  printf("against the vgprs= field in the RJ_LOG=1 dispatch output.\n");

  (void)hipFree(d_in);
  (void)hipFree(d_out);
  return 0;
}
