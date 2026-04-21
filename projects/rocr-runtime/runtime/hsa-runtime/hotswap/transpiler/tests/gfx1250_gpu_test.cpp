// End-to-end test: gfx1250 Triton kernels -> LLVM IR -> gfx942 -> execute on GPU
//
// Kernels tested:
//   vecadd_kernel  (fp16):  C[i] = A[i] + B[i]
//   matmul_kernel  (fp16->f32):  C = A @ B  (64x64x32 and 128x128x32 tile sizes)
//   softmax_kernel (fp32):  row-wise softmax

#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const char *GFX1250_DATA_DIR = GFX1250_TEST_DATA_DIR;

// ============================================================================
// Vecadd: C[i] = A[i] + B[i], fp16
// ============================================================================
static void doTestVecadd() {
  printf("--- vecadd_kernel (fp16, BLOCK_SIZE=1024) ---\n");
  std::string path = std::string(GFX1250_DATA_DIR) + "/vecadd_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942", "vecadd_kernel");
  ASSERT_TRUE(result.success) << "Pipeline failed for vecadd";
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  const int N = 4096;
  std::vector<__half> hA(N), hB(N), hC(N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < N; i++) {
    hA[i] = __float2half(dist(rng));
    hB[i] = __float2half(dist(rng));
  }

  __half *dA, *dB, *dC;
  HIP_ASSERT(hipMalloc(&dA, N * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dB, N * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dC, N * sizeof(__half)));
  HIP_ASSERT(hipMemcpy(dA, hA.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dB, hB.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dC, 0, N * sizeof(__half)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, "vecadd_kernel"));

  auto meta = transpiler::extractKernelMeta(data, "vecadd_kernel");

  std::vector<uint8_t> argBuf(meta.kernargSegmentSize, 0);
  memcpy(argBuf.data() + 0,  &dA, 8);
  memcpy(argBuf.data() + 8,  &dB, 8);
  memcpy(argBuf.data() + 16, &dC, 8);
  int32_t n = N;
  memcpy(argBuf.data() + 24, &n, 4);
  size_t argSz = argBuf.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuf.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz, HIP_LAUNCH_PARAM_END};
  int wgSize = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 1024;
  int gridX = (N + wgSize - 1) / wgSize;
  HIP_ASSERT(hipModuleLaunchKernel(func, gridX, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hC.data(), dC, N * sizeof(__half), hipMemcpyDeviceToHost));

  int errors = 0;
  float maxErr = 0;
  for (int i = 0; i < N; i++) {
    float expected = __half2float(hA[i]) + __half2float(hB[i]);
    float got = __half2float(hC[i]);
    float diff = std::fabs(got - expected);
    if (diff > maxErr) maxErr = diff;
    if (diff > 0.01f) {
      if (errors < 3)
        fprintf(stderr, "  [%d] got=%f expected=%f diff=%e\n", i, got, expected, diff);
      errors++;
    }
  }

  (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC); (void)hipModuleUnload(mod);
  printf("  Result: %d errors, maxErr=%e\n", errors, maxErr);
  EXPECT_EQ(errors, 0) << errors << " element mismatches in vecadd";
}

// ============================================================================
// rcp_sqrt: per-thread `out[i] = 1.0f / sqrtf(in[i])`, fp32.
//
// End-to-end runtime gate on the v_div_scale_f32-literal-numer decoder
// fix in `handle_valu.cpp::V_DIV_SCALE_F32` (commit 52ccf893aa +
// follow-up 46f6acdb88).  Pair with the lit fixture
// `lit_tests/v_div_scale_f32_literal_numer/` which pins the IR-level
// call shape; this GPU test pins the numerical result after re-
// lowering onto gfx942 and execution on-device.
//
// Pure per-thread elementwise — no reductions, no cross-lane ops, no
// wave-size-sensitive idioms — so cross-widening has no observable
// effect and the only thing this test actually exercises is whether
// the IEEE-conformant div_scale+Newton+div_fixup chain round-trips
// from gfx1250's literal-numer encoding through salmon's raise, back
// through the gfx942 backend's lowering of the lifted
// `@llvm.amdgcn.div.scale.f32(1.0, ...)` intrinsic, into a correct
// runtime result.  Pre-fix, both scale calls decoded to
// `(<vgpr>, <vgpr>, false)` and the backend's div_fixup collapsed
// every output to 1.0 bit-exact; this test's ULP-envelope assertion
// catches that regression as a max-relative-error well beyond the
// 2-ULP tolerance.
//
// Input sweep: 256 positive normalized fp32 values in [1e-4, 1e4].
// No subnormals, no zeros, no negatives — keeping the input domain
// inside IEEE-happy-path avoids NaN / inf / -0 edge cases that would
// put the ULP envelope at the mercy of host vs device rounding-mode
// drift rather than testing the div-scale decoder.
// ============================================================================
static void doTestRcpSqrt() {
  printf("--- rcp_sqrt_kernel (fp32, per-thread 1/sqrt(x)) ---\n");
  std::string path = std::string(GFX1250_DATA_DIR) + "/rcp_sqrt_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942",
                                         "rcp_sqrt_kernel");
  ASSERT_TRUE(result.success) << "Pipeline failed for rcp_sqrt";
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  const int N = 256;
  std::vector<float> hIn(N), hOut(N), hRef(N);
  std::mt19937 rng(42);
  // Positive normalized range [1e-4, 1e4]; sqrt domain trivially
  // satisfies IEEE's "well-conditioned" precondition for the divide
  // that follows.  Picked to cover small / medium / large magnitudes
  // without dipping into subnormal / zero / infinity territory.
  std::uniform_real_distribution<float> dist(1e-4f, 1e4f);
  for (int i = 0; i < N; i++) {
    hIn[i] = dist(rng);
    hRef[i] = 1.0f / std::sqrt(hIn[i]);
  }

  float *dIn, *dOut;
  HIP_ASSERT(hipMalloc(&dIn, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&dOut, N * sizeof(float)));
  HIP_ASSERT(hipMemcpy(dIn, hIn.data(), N * sizeof(float),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dOut, 0, N * sizeof(float)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, "rcp_sqrt_kernel"));

  auto meta = transpiler::extractKernelMeta(data, "rcp_sqrt_kernel");

  // Kernel signature: `rcp_sqrt_kernel(float *out, const float *in)`.
  // First 8 bytes = out pointer, next 8 = in pointer.  hipcc lays
  // hidden args (workgroup dims / implicit args) after the explicit
  // args; kernargSegmentSize reflects the full padded layout and is
  // what we size argBuf to.
  std::vector<uint8_t> argBuf(meta.kernargSegmentSize, 0);
  memcpy(argBuf.data() + 0, &dOut, 8);
  memcpy(argBuf.data() + 8, &dIn, 8);
  size_t argSz = argBuf.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuf.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz,
                    HIP_LAUNCH_PARAM_END};
  int wgSize = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 256;
  // Cap at N — the kernel uses `blockIdx.x * blockDim.x + threadIdx.x`
  // as its global index without a bounds check, so N must be an exact
  // multiple of wgSize (or we must ensure the launch covers exactly N
  // threads).  Clamping wgSize to N and using gridX = 1 guarantees
  // the latter for any N ≤ hipcc's default work-group limit.
  if (wgSize > N) wgSize = N;
  int gridX = (N + wgSize - 1) / wgSize;
  HIP_ASSERT(hipModuleLaunchKernel(func, gridX, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr,
                                   nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hOut.data(), dOut, N * sizeof(float),
                       hipMemcpyDeviceToHost));

  int errors = 0;
  float maxRelErr = 0.0f;
  // 2 ULP at fp32's 1.0 = ~2.4e-7; the AMDGPU div_scale+Newton+div_fixup
  // chain is IEEE-conformant on normalized positive inputs (correctly-
  // rounded up to the last bit), and `std::sqrt` on the host is the
  // same correctly-rounded primitive, so a strict < 2 ULP relative-
  // error envelope is the tightest band that also tolerates any
  // host/device rounding-mode configuration drift without false
  // positives.  The pre-fix bug produces max-rel-err ≈ 1 (output
  // collapses to 1.0 bit-exact regardless of input), so the test's
  // signal margin is enormous compared to this envelope.
  const float kTolRel = 2.0e-7f;
  for (int i = 0; i < N; i++) {
    float diff = std::fabs(hOut[i] - hRef[i]);
    float rel = diff / std::fabs(hRef[i]);
    if (rel > maxRelErr) maxRelErr = rel;
    if (rel > kTolRel) {
      if (errors < 3)
        fprintf(stderr, "  [%d] in=%g got=%g ref=%g rel_err=%e\n",
                i, hIn[i], hOut[i], hRef[i], rel);
      errors++;
    }
  }

  (void)hipFree(dIn);
  (void)hipFree(dOut);
  (void)hipModuleUnload(mod);
  printf("  Result: %d errors, max_rel_err=%e\n", errors, maxRelErr);
  EXPECT_EQ(errors, 0) << errors << " element mismatches in rcp_sqrt";
}

// ============================================================================
// WMMA probe: scaled-down chained/parallel WMMA matmul probes.
//
// Ladder probes for isolating the gfx1250 → gfx942 `v_wmma_f32_16x16x32_f16`
// lowering failure seen in `Gfx1250Gpu.Matmul128x128` (the large 128×128
// Triton matmul). Each probe pins one specific (tiles × chain) topology so
// the first failing step localizes the root cause in the (tile count, chain
// depth, VGPR-MSB usage) axis (see
// `test_data/gfx1250/wmma_chain_probe_kernel.hip` for the probe sources).
//
// All probes seed A = B = <half 1.0>×16 per lane. Per WMMA: 32 (K-dim).
// Expected per output element = 32 * chainDepth, independent of tile.
// CPU check is a single-scalar compare over every output element.
// ============================================================================
static void doTestWmmaProbe(const char *kernelName, int numTiles,
                              int chainDepth, int numABuffers,
                              int numBBuffers) {
  printf("--- %s (numTiles=%d chainDepth=%d) ---\n", kernelName, numTiles,
         chainDepth);
  std::string path = std::string(GFX1250_DATA_DIR) +
                     "/wmma_chain_probe_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942", kernelName);
  ASSERT_TRUE(result.success) << "Pipeline failed for " << kernelName;
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  auto meta = transpiler::extractKernelMeta(data, kernelName);

  // A/B buffers: seed with <half 1.0>×16 per 16-half vector, one vector per
  // buffer slot. The probe kernels index a_ptr[0..numABuffers-1] and
  // b_ptr[0..numBBuffers-1]; each caller passes both counts explicitly to
  // match exactly what the specific kernel reads (wmma_parallel2 reads
  // b_ptr[0..1], wmma_parallel4 reads b_ptr[0..3], etc.). Under-sizing
  // either buffer causes the kernel to read out-of-bounds memory, which
  // manifests as zeroed-out tiles and was previously misdiagnosed as a
  // transpiler numerical bug.
  ASSERT_GT(numABuffers, 0);
  ASSERT_GT(numBBuffers, 0);
  std::vector<__half> hA(numABuffers * 16);
  std::vector<__half> hB(numBBuffers * 16);
  for (size_t i = 0; i < hA.size(); i++) hA[i] = __float2half(1.0f);
  for (size_t i = 0; i < hB.size(); i++) hB[i] = __float2half(1.0f);

  __half *dA, *dB;
  float *dC;
  HIP_ASSERT(hipMalloc(&dA, hA.size() * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dB, hB.size() * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dC, numTiles * 8 * sizeof(float)));
  HIP_ASSERT(hipMemcpy(dA, hA.data(), hA.size() * sizeof(__half),
                        hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dB, hB.data(), hB.size() * sizeof(__half),
                        hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dC, 0, numTiles * 8 * sizeof(float)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, kernelName));

  // Kernarg layout: a_ptr, b_ptr, c_ptr  (hipcc packs pointers at offsets
  // 0, 8, 16 before the hidden pads).
  std::vector<uint8_t> argBuf(meta.kernargSegmentSize, 0);
  memcpy(argBuf.data() + 0,  &dA, 8);
  memcpy(argBuf.data() + 8,  &dB, 8);
  memcpy(argBuf.data() + 16, &dC, 8);
  size_t argSz = argBuf.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuf.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz,
                    HIP_LAUNCH_PARAM_END};

  // The probe kernels use __launch_bounds__(32); on gfx942 that still
  // projects to one wave64 per block, so a grid of (1,1,1) exercises the
  // full WMMA → MFMA lowering path.
  int wgSize = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 32;
  HIP_ASSERT(hipModuleLaunchKernel(func, 1, 1, 1, wgSize, 1, 1,
                                    meta.groupSegmentFixedSize, nullptr,
                                    nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<float> hC(numTiles * 8);
  HIP_ASSERT(hipMemcpy(hC.data(), dC, numTiles * 8 * sizeof(float),
                        hipMemcpyDeviceToHost));

  // Expected: every per-lane dword of the accumulator equals
  // 32 * chainDepth (K-dim × chain depth).  WMMA output is 8 dwords
  // per lane × 32 Wave32 lanes = a 16×16 f32 tile where EVERY element
  // is 32 * chainDepth.  The probe writes the lane-local v8f back to
  // consecutive memory slots per wave, so hC holds one full v8f (8
  // floats) per thread per tile.  numTiles iterates across tiles.
  float expected = 32.0f * (float)chainDepth;
  int errors = 0;
  float maxErr = 0.0f;
  printf("  per-tile summary (showing dw0 of each tile):\n   ");
  for (int t = 0; t < numTiles; t++) {
    printf(" [%d]%.3f", t, hC[t * 8]);
  }
  printf("\n");
  for (int t = 0; t < numTiles; t++) {
    int tileErrs = 0;
    for (int i = 0; i < 8; i++) {
      float got = hC[t * 8 + i];
      float diff = std::fabs(got - expected);
      if (diff > maxErr) maxErr = diff;
      if (diff > 1e-3f) {
        if (errors < 3)
          fprintf(stderr, "  tile %d dw %d: got=%f exp=%f\n", t, i, got,
                  expected);
        errors++;
        tileErrs++;
      }
    }
    if (tileErrs > 0)
      printf("  tile %d: %d/8 dw wrong  (dw0=%f)\n", t, tileErrs,
             hC[t * 8]);
  }

  (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
  (void)hipModuleUnload(mod);
  printf("  Result: %d errors over %d tiles × 8 dwords, maxErr=%e\n",
         errors, numTiles, maxErr);
  EXPECT_EQ(errors, 0) << errors << " per-lane dword mismatches in "
                        << kernelName;
}

// ============================================================================
// Matmul: C = A @ B, A/B fp16, C fp32
// ============================================================================
// Diagnostic data pattern for A / B inputs.
//
//   Random    — default, stresses the full lane-layout pipeline.
//   Uniform   — A = B = 1.0 everywhere ⇒ reference C[i,j] = K for every
//               (i, j).  This isolates the kernel's control-flow /
//               address-generation from the WMMA per-lane layout /
//               ds_bpermute redistribution: a uniform input produces
//               the SAME output on every lane so any lane-permutation
//               bug is masked.  When uniform passes but random fails,
//               the remaining bug is isolated to the WMMA operand
//               layout or accumulator redistribution.
enum class MatmulDataPattern {
  Random,
  Uniform,
  RowIdA,
  RowOnly124,
  EvenRows,
  KStripedRow124,  // row 124 tagged by K-strip; 0 elsewhere
};

static void doTestMatmul(const char *hsacoFile, int M, int N, int K,
                         const char *label,
                         MatmulDataPattern pattern = MatmulDataPattern::Random) {
  const char *patternStr = "random";
  if (pattern == MatmulDataPattern::Uniform) patternStr = "uniform";
  else if (pattern == MatmulDataPattern::RowIdA) patternStr = "rowIdA";
  else if (pattern == MatmulDataPattern::RowOnly124) patternStr = "rowOnly124";
  else if (pattern == MatmulDataPattern::EvenRows) patternStr = "evenRows";
  else if (pattern == MatmulDataPattern::KStripedRow124) patternStr = "kStripedRow124";
  printf("--- matmul_kernel (%s, M=%d N=%d K=%d, pattern=%s) ---\n",
         label, M, N, K, patternStr);
  std::string path = std::string(GFX1250_DATA_DIR) + "/" + hsacoFile;
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  // Matmul is the pre-eminent opt-in consumer of the Phase 6.5
  // writelane/readlane rewrite: the 128x128-tile variant is the exact
  // kernel the rewrite was designed to unblock (canonical `s_bfe_u32
  // ttmp8, 0x50019` wave_id extraction + v_writelane/v_readlane
  // register spills + v_wmma_* accumulator → implicit scalarisation
  // via v_readfirstlane, see wave-size-translation.md §5.6.3). The
  // 64x64-tile sibling does not trigger the rewrite (no divergent
  // writelane sites) and is unaffected — the flag is only observed at
  // sites where the oracle flags a lane-divergent scalar feed.
  // Matmul also opts in to `WaveNativeProjection` for wave32 → wave64
  // cross-widening: the WMMA → MFMA redistribute / MFMA / collect
  // pipeline requires hardware EXEC = -1 on all 64 target lanes so
  // the upper half (lanes 32..63, which hold source wave 3's portion
  // of a 128×128 output tile) actually participates in the MFMA
  // collective. Without this, the classic partial-EXEC failure mode
  // leaves rows 12..15 of each Wave64 MFMA sub-tile undefined, which
  // manifested as ~3% numerical errors on random-input runs before
  // the flag (uniform-diag masked the defect because its reference
  // is position-invariant). See `wmma_lowering.cpp`'s "Partial-wave
  // correctness and hardware EXEC" header and
  // `WaveNativeProjection::emitInitialExec` for the contract.
  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942",
                                        "matmul_kernel",
                                        /*enableWritelaneRewrite=*/true,
                                        /*enableWaveNative=*/true);
  ASSERT_TRUE(result.success) << "Pipeline failed for matmul " << label;
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  std::vector<__half> hA(M * K), hB(K * N);
  std::vector<float> hC(M * N, 0.0f);
  if (pattern == MatmulDataPattern::Uniform) {
    for (auto &h : hA) h = __float2half(1.0f);
    for (auto &h : hB) h = __float2half(1.0f);
  } else if (pattern == MatmulDataPattern::RowIdA) {
    // A[i,k] = (i+1) * 0.001, B[k,j] = 1.0.
    // Expected C[i,j] = K * (i+1) * 0.001 for every column j.
    for (int i = 0; i < M; i++)
      for (int k = 0; k < K; k++)
        hA[i * K + k] = __float2half((i + 1) * 0.001f);
    for (auto &h : hB) h = __float2half(1.0f);
  } else if (pattern == MatmulDataPattern::RowOnly124) {
    // A[i,k] = 1 only for i==124, else 0. B[k,j] = 1.
    // Expected C[124,j] = K for all j, other rows = 0.
    for (int i = 0; i < M; i++)
      for (int k = 0; k < K; k++)
        hA[i * K + k] = __float2half((i == 124) ? 1.0f : 0.0f);
    for (auto &h : hB) h = __float2half(1.0f);
  } else if (pattern == MatmulDataPattern::EvenRows) {
    // A[i,k] = 1 iff i is even, else 0. B = 1.
    // Expected C[even row, j] = K; C[odd row, j] = 0.
    //
    // Diagnostic: if odd-row outputs are non-zero, the kernel is
    // substituting EVEN source rows into ODD target rows (matches
    // the `A[0], A[2], A[4], A[6]` substitution pattern observed
    // in RowIdA for output rows 124..127). If ONLY rows 125, 127
    // (not 60-63, 92-95) are non-zero, the defect is wave-3-
    // specific; if ALL odd rows (12-15 → 0, 28-31 → 0, 44-47 → 0
    // etc., 60-63, 92-95, 124-127) show substitution, the defect
    // is a general pass-2 collect bug that is MASKED for rows 12-
    // 15 etc. by some other factor.
    for (int i = 0; i < M; i++)
      for (int k = 0; k < K; k++)
        hA[i * K + k] = __float2half((i & 1) == 0 ? 1.0f : 0.0f);
    for (auto &h : hB) h = __float2half(1.0f);
  } else if (pattern == MatmulDataPattern::KStripedRow124) {
    // A[124, k] distinct per K-strip so each K-iter contributes a
    // unique amount. A[other rows] = 0. B = 1.
    //   k in [  0,  32): A[124,k] = 0.1  → ref contribution 3.2
    //   k in [ 32,  64): A[124,k] = 0.2  → ref contribution 6.4
    //   k in [ 64,  96): A[124,k] = 0.4  → ref contribution 12.8
    //   k in [ 96, 128): A[124,k] = 0.8  → ref contribution 25.6
    // Expected C[124, j] = 48.0. Other rows = 0.
    //
    // The bug drops one 32-k-step chunk; the arithmetic difference
    // from 48.0 identifies WHICH K-iter is buggy:
    //   got = 48.0 - 3.2   → K-iter 0 (k=  0.. 31)
    //   got = 48.0 - 6.4   → K-iter 1 (k= 32.. 63)
    //   got = 48.0 - 12.8  → K-iter 2 (k= 64.. 95)
    //   got = 48.0 - 25.6  → K-iter 3 (k= 96..127)
    for (int i = 0; i < M; i++)
      for (int k = 0; k < K; k++) {
        float v = 0.0f;
        if (i == 124) {
          if (k < 32)       v = 0.1f;
          else if (k < 64)  v = 0.2f;
          else if (k < 96)  v = 0.4f;
          else              v = 0.8f;
        }
        hA[i * K + k] = __float2half(v);
      }
    for (auto &h : hB) h = __float2half(1.0f);
  } else {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < M * K; i++) hA[i] = __float2half(dist(rng));
    for (int i = 0; i < K * N; i++) hB[i] = __float2half(dist(rng));
  }

  __half *dA, *dB; float *dC;
  HIP_ASSERT(hipMalloc(&dA, M * K * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dB, K * N * sizeof(__half)));
  HIP_ASSERT(hipMalloc(&dC, M * N * sizeof(float)));
  HIP_ASSERT(hipMemcpy(dA, hA.data(), M * K * sizeof(__half), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dB, hB.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));
  std::vector<float> sentinel(M * N, 42.0f);
  HIP_ASSERT(hipMemcpy(dC, sentinel.data(), M * N * sizeof(float), hipMemcpyHostToDevice));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, "matmul_kernel"));

  struct alignas(8) Args {
    __half *a, *b; float *c;
    int m, n, k;
    int stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn;
    int _pad_60_63;
    void *unused12;
    void *unused13;
  };
  static_assert(sizeof(Args) == 80, "matmul_kernel kernarg layout mismatch");
  Args args = {dA, dB, dC, M, N, K, K, 1, N, 1, N, 1, 0, nullptr, nullptr};
  size_t argSz = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz, HIP_LAUNCH_PARAM_END};

  auto meta = transpiler::extractKernelMeta(data, "matmul_kernel");
  printf("  meta: kernargSz=%d maxFlatWgSz=%d groupSegFixedSz=%d\n",
         meta.kernargSegmentSize, meta.maxFlatWorkgroupSize,
         meta.groupSegmentFixedSize);
  int wgSize = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 256;
  int blockM = (std::string(hsacoFile).find("large") != std::string::npos) ? 128 : 64;
  int blockN = blockM;
  int numPidM = (M + blockM - 1) / blockM;
  int numPidN = (N + blockN - 1) / blockN;
  int gridX = numPidM * numPidN;
  int ldsRequired = (blockM >= 128) ? 65536 : 32768;
  auto transpMeta = transpiler::extractKernelMeta(result.hsaco, "matmul_kernel");
  int dynamicLds = std::max(0, ldsRequired - transpMeta.groupSegmentFixedSize);
  printf("  launch: grid=(%d,1,1) wg=(%d,1,1) sharedMem=%d "
         "(src static=%d / transp static=%d + dynamic=%d)\n",
         gridX, wgSize, transpMeta.groupSegmentFixedSize + dynamicLds,
         meta.groupSegmentFixedSize, transpMeta.groupSegmentFixedSize,
         dynamicLds);
  HIP_ASSERT(hipModuleLaunchKernel(func, gridX, 1, 1, wgSize, 1, 1,
                                   dynamicLds, nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference
  std::vector<float> ref(M * N, 0.0f);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int kk = 0; kk < K; kk++)
        sum += __half2float(hA[i * K + kk]) * __half2float(hB[kk * N + j]);
      ref[i * N + j] = sum;
    }

  printf("  First 16 got: ");
  for (int i = 0; i < std::min(16, M*N); i++) printf("%.4f ", hC[i]);
  printf("\n  First 16 ref: ");
  for (int i = 0; i < std::min(16, M*N); i++) printf("%.4f ", ref[i]);
  printf("\n");

  printf("  Row starts (got|ref): ");
  for (int r = 0; r < std::min(8, M); r++)
    printf("[%d]%.3f|%.3f ", r, hC[r*N], ref[r*N]);
  printf("\n");

  int zeroCount = 0;
  for (int i = 0; i < M * N; i++)
    if (hC[i] == 0.0f) zeroCount++;
  printf("  Zero elements: %d / %d\n", zeroCount, M*N);

  // Sentinel-write map: coarse 16x16-tile view of which output tiles
  // received a store vs. retained the 42.0 sentinel. Helps localise
  // wave-size-translation regressions where only a subset of source
  // waves end up writing their outputs.
  if (pattern == MatmulDataPattern::Uniform) {
    int tileCols = (N + 15) / 16;
    int tileRows = (M + 15) / 16;
    if (tileRows <= 16 && tileCols <= 16) {
      printf("  Tile write map (W=written, s=sentinel-42, ?=mixed):\n");
      for (int tr = 0; tr < tileRows; tr++) {
        printf("    [row %3d]: ", tr * 16);
        for (int tc = 0; tc < tileCols; tc++) {
          int written = 0, sentinel = 0;
          for (int dr = 0; dr < 16 && tr*16+dr < M; dr++)
            for (int dc = 0; dc < 16 && tc*16+dc < N; dc++) {
              float v = hC[(tr*16+dr)*N + (tc*16+dc)];
              if (v == 42.0f) sentinel++;
              else written++;
            }
          if (sentinel == 0) printf("W ");
          else if (written == 0) printf("s ");
          else printf("? ");
        }
        printf("\n");
      }
    }
  }

  int errors = 0;
  float maxErr = 0;
  std::vector<int> rowErrors(M, 0);
  std::vector<int> colErrors(N, 0);
  for (int i = 0; i < M * N; i++) {
    float diff = std::fabs(hC[i] - ref[i]);
    float denom = std::max(std::fabs(ref[i]), 1e-6f);
    float relErr = diff / denom;
    if (diff > maxErr) maxErr = diff;
    if (relErr > 0.05f && diff > 0.01f) {
      if (errors < 3)
        fprintf(stderr, "  [%d] got=%f ref=%f diff=%e relErr=%e\n", i, hC[i], ref[i], diff, relErr);
      errors++;
      rowErrors[i / N]++;
      colErrors[i % N]++;
    }
  }
  (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC); (void)hipModuleUnload(mod);
  if (errors) {
    printf("  Row histogram (rows with errors):\n   ");
    int printed = 0;
    for (int r = 0; r < M; r++) {
      if (rowErrors[r] > 0) {
        printf(" r%d:%d", r, rowErrors[r]);
        if (++printed % 8 == 0) printf("\n   ");
      }
    }
    printf("\n");
    printf("  Col histogram (cols with errors): ");
    printed = 0;
    for (int c = 0; c < N; c++) {
      if (colErrors[c] > 0) {
        printf(" c%d:%d", c, colErrors[c]);
        if (++printed > 16) { printf(" ..."); break; }
      }
    }
    printf("\n");
    for (int r = 0; r < M; r++) {
      if (rowErrors[r] > 0) {
        printf("  Row %d first 8 got|ref|diff:\n    ", r);
        for (int c = 0; c < std::min(8, N); c++) {
          float g = hC[r*N + c], rf = ref[r*N + c];
          printf("[%d: %.4f|%.4f|%.4f] ", c, g, rf, g - rf);
        }
        printf("\n");
      }
    }
  }
  printf("  Result: %d errors, maxErr=%e\n", errors, maxErr);
  EXPECT_EQ(errors, 0) << errors << " mismatches in matmul " << label;
}

// ============================================================================
// Softmax: row-wise softmax, fp32
// ============================================================================
static void doTestSoftmax() {
  printf("--- softmax_kernel (fp32, BLOCK_SIZE=1024) ---\n");
  std::string path = std::string(GFX1250_DATA_DIR) + "/softmax_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942", "softmax_kernel");
  ASSERT_TRUE(result.success) << "Pipeline failed for softmax";
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  const int nRows = 1, nCols = 512;
  std::vector<float> hIn(nRows * nCols), hOut(nRows * nCols, 0.0f);
  for (auto &v : hIn) v = 1.0f;

  size_t allocElems = 1024 * 1024;
  float *dIn, *dOut;
  HIP_ASSERT(hipMalloc(&dIn, allocElems * sizeof(float)));
  HIP_ASSERT(hipMalloc(&dOut, allocElems * sizeof(float)));
  HIP_ASSERT(hipMemcpy(dIn, hIn.data(), nRows * nCols * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dOut, 0, allocElems * sizeof(float)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, "softmax_kernel"));

  auto meta = transpiler::extractKernelMeta(data, "softmax_kernel");
  printf("  meta: kernargSegmentSize=%d, maxFlatWorkgroupSize=%d, "
         "groupSegmentFixedSize=%d\n",
         meta.kernargSegmentSize, meta.maxFlatWorkgroupSize,
         meta.groupSegmentFixedSize);

  std::vector<uint8_t> argBuf(meta.kernargSegmentSize, 0);
  memcpy(argBuf.data() + 0,  &dOut, 8);
  memcpy(argBuf.data() + 8,  &dIn, 8);
  int32_t inStride = nCols, outStride = nCols, nColsArg = nCols;
  memcpy(argBuf.data() + 16, &inStride, 4);
  memcpy(argBuf.data() + 20, &outStride, 4);
  memcpy(argBuf.data() + 24, &nColsArg, 4);
  size_t argSz = argBuf.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuf.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz, HIP_LAUNCH_PARAM_END};
  int wgSize = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 1024;
  printf("  launch: grid=(%d,1,1) wg=(%d,1,1) sharedMem=%d argSz=%zu\n",
         nRows, wgSize, meta.groupSegmentFixedSize, argSz);
  HIP_ASSERT(hipModuleLaunchKernel(func, nRows, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hOut.data(), dOut, nRows * nCols * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference
  std::vector<float> ref(nRows * nCols);
  for (int r = 0; r < nRows; r++) {
    float maxVal = *std::max_element(hIn.begin() + r * nCols, hIn.begin() + (r + 1) * nCols);
    float sum = 0;
    for (int c = 0; c < nCols; c++) {
      ref[r * nCols + c] = std::exp(hIn[r * nCols + c] - maxVal);
      sum += ref[r * nCols + c];
    }
    for (int c = 0; c < nCols; c++)
      ref[r * nCols + c] /= sum;
  }

  printf("  First 16 outputs:");
  for (int i = 0; i < 16 && i < nRows * nCols; i++) printf(" %.4f", hOut[i]);
  printf("\n  First 16 refs:   ");
  for (int i = 0; i < 16 && i < nRows * nCols; i++) printf(" %.4f", ref[i]);
  printf("\n");

  int nInf = 0, nNan = 0, nZero = 0, nNeg = 0;
  for (int i = 0; i < nRows * nCols; i++) {
    if (std::isinf(hOut[i])) nInf++;
    if (std::isnan(hOut[i])) nNan++;
    if (hOut[i] == 0.0f) nZero++;
    if (hOut[i] < 0.0f) nNeg++;
  }
  printf("  Pattern: inf=%d nan=%d zero=%d neg=%d of %d total\n",
         nInf, nNan, nZero, nNeg, nRows * nCols);

  int errors = 0;
  float maxErr = 0;
  for (int i = 0; i < nRows * nCols; i++) {
    float diff = std::fabs(hOut[i] - ref[i]);
    if (diff > maxErr) maxErr = diff;
    if (diff > 1e-4f) {
      if (errors < 3)
        fprintf(stderr, "  [%d] got=%f ref=%f diff=%e\n", i, hOut[i], ref[i], diff);
      errors++;
    }
  }

  (void)hipFree(dIn); (void)hipFree(dOut); (void)hipModuleUnload(mod);
  printf("  Result: %d errors, maxErr=%e\n", errors, maxErr);
  EXPECT_EQ(errors, 0) << errors << " mismatches in softmax";
}

// ============================================================================
// Cross-lane regression tests (P4 / P5 / P6 rewrites; see
// hotswap/docs/wave-size-translation.md §5.3)
// ============================================================================
//
// CI-resident hardware regression gates for the cross-lane
// emulation/lift handlers. Each test:
//
//   1. Reads a pre-built gfx1250 (wave32) `.hsaco` (committed
//      alongside its `.hip` source for reproducibility).
//   2. Lifts through the transpiler pipeline to gfx942 (wave64).
//   3. Runs the lifted kernel on gfx942 hardware.
//   4. Verifies per-lane output matches the source-intended pattern.
//
// Each test enforces the contract documented in the corresponding
// handler comment block — a back-reference is included in each
// per-test docstring so the contract is auditable from either
// direction.

// ----- P4: v_permlane16_swap_b32 → paired ds_bpermute emulation -----
//
// Spec: handle_valu_cross_lane.cpp::V_PERMLANE16_SWAP_B32 MODREP
// block. The handler reads vdst_in (tied to vdst output) and src0_in
// (tied to src0_out output), then emits two ds_bpermute calls with
// `partner = lane_id XOR 16` and `byte_addr = partner << 2` —
// per-32-lane-half independent swap on wave64 by the bit-5
// preservation hardware contract.
//
// Per-lane setup: vdst_in[L]=L, src0_in[L]=1000+L.
// Expected: new_vdst[L]=1000+(L^16), new_src0_out[L]=(L^16).
static void doTestPermlane16Swap() {
  printf("--- permlane16_swap_kernel (P4 end-to-end) ---\n");
  std::string path =
      std::string(GFX1250_DATA_DIR) + "/permlane16_swap_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942",
                                         "permlane16_swap_kernel");
  ASSERT_TRUE(result.success) << "Pipeline failed for permlane16_swap";
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  constexpr int N = 64; // wave64 = 1 wave on gfx942; full block of 64 lanes.
  auto meta = transpiler::extractKernelMeta(data, "permlane16_swap_kernel");

  // Defensive: the .hip declares __launch_bounds__(64). If someone
  // edits the .hip without rebuilding the .hsaco AND this test, drift
  // between the binary's max-flat and the test's hard-coded N would
  // silently happen; surface it loudly here.
  ASSERT_GE(meta.maxFlatWorkgroupSize, (uint32_t)N)
      << "Binary's max_flat_workgroup_size (" << meta.maxFlatWorkgroupSize
      << ") < test's N (" << N << "). The .hip's __launch_bounds__ and the "
         "test's N have drifted.";

  std::vector<int> hVdst(N, -1), hSrc0(N, -1);

  int *dVdst, *dSrc0;
  HIP_ASSERT(hipMalloc(&dVdst, N * sizeof(int)));
  HIP_ASSERT(hipMalloc(&dSrc0, N * sizeof(int)));
  HIP_ASSERT(hipMemset(dVdst, 0xff, N * sizeof(int)));
  HIP_ASSERT(hipMemset(dSrc0, 0xff, N * sizeof(int)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, "permlane16_swap_kernel"));

  std::vector<uint8_t> argBuf(meta.kernargSegmentSize, 0);
  memcpy(argBuf.data() + 0, &dVdst, 8);
  memcpy(argBuf.data() + 8, &dSrc0, 8);
  size_t argSz = argBuf.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuf.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz,
                    HIP_LAUNCH_PARAM_END};
  HIP_ASSERT(hipModuleLaunchKernel(func, /*grid*/ 1, 1, 1,
                                   /*block*/ N, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr,
                                   nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hVdst.data(), dVdst, N * sizeof(int),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(hSrc0.data(), dSrc0, N * sizeof(int),
                       hipMemcpyDeviceToHost));

  int errors = 0;
  for (int L = 0; L < N; L++) {
    int partner = L ^ 16;
    int expVdst = 1000 + partner;
    int expSrc0 = partner;
    if (hVdst[L] != expVdst || hSrc0[L] != expSrc0) {
      if (errors < 4)
        fprintf(stderr,
                "  lane %2d: vdst got=%d exp=%d, src0 got=%d exp=%d\n",
                L, hVdst[L], expVdst, hSrc0[L], expSrc0);
      errors++;
    }
  }

  (void)hipFree(dVdst);
  (void)hipFree(dSrc0);
  (void)hipModuleUnload(mod);
  printf("  Result: %d errors over %d lanes\n", errors, N);
  EXPECT_EQ(errors, 0)
      << errors << " lane mismatches in permlane16_swap (expected per-half "
                   "XOR-16 partner pattern)";
}

// ----- P5: DPP modifier → llvm.amdgcn.update.dpp lift -----
//
// Spec: raise_context.cpp::emitUpdateDpp + the OpResolver
// `wrapDppIfNeeded` hook in raise_context.hpp. The handler
// transparently wraps src0 reads through `update.dpp` whenever
// `di.hasDpp == true`, populated at decode time by
// `decodeDppModifiers` from the original (pre-canonicalisation)
// MCInstrDesc's named operands (dpp_ctrl, row_mask, bank_mask,
// bound_ctrl).
//
// This test's kernel uses `v_mov_b32_dpp ... quad_perm:[1,0,3,2]
// row_mask:0xf bank_mask:0xf` — within each 4-lane quad, swap
// adjacent pairs. Equivalently: lane L reads input[L XOR 1].
// Per-quad-independent by construction (DPP quad_perm operates on
// 4-lane groups), so wave64 modulo-replication is trivially correct.
//
// Per-lane setup: input[L] = L.  Expected: output[L] = L XOR 1.
static void doTestDppQuadPerm() {
  printf("--- dpp_quad_perm_kernel (P5 end-to-end) ---\n");
  std::string path =
      std::string(GFX1250_DATA_DIR) + "/dpp_quad_perm_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942",
                                         "dpp_quad_perm_kernel");
  ASSERT_TRUE(result.success) << "Pipeline failed for dpp_quad_perm";
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  constexpr int N = 64;
  auto meta = transpiler::extractKernelMeta(data, "dpp_quad_perm_kernel");
  ASSERT_GE(meta.maxFlatWorkgroupSize, (uint32_t)N)
      << "Binary's max_flat (" << meta.maxFlatWorkgroupSize << ") < N (" << N
      << "); .hip / .cpp drift";

  std::vector<int> hOut(N, -1);
  int *dOut;
  HIP_ASSERT(hipMalloc(&dOut, N * sizeof(int)));
  HIP_ASSERT(hipMemset(dOut, 0xff, N * sizeof(int)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, "dpp_quad_perm_kernel"));

  std::vector<uint8_t> argBuf(meta.kernargSegmentSize, 0);
  memcpy(argBuf.data() + 0, &dOut, 8);
  size_t argSz = argBuf.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuf.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz,
                    HIP_LAUNCH_PARAM_END};
  HIP_ASSERT(hipModuleLaunchKernel(func, 1, 1, 1, N, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr,
                                   nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hOut.data(), dOut, N * sizeof(int),
                       hipMemcpyDeviceToHost));

  int errors = 0;
  for (int L = 0; L < N; L++) {
    int expected = L ^ 1;
    if (hOut[L] != expected) {
      if (errors < 4)
        fprintf(stderr, "  lane %2d: got=%d exp=%d\n", L, hOut[L], expected);
      errors++;
    }
  }
  (void)hipFree(dOut);
  (void)hipModuleUnload(mod);
  printf("  Result: %d errors over %d lanes\n", errors, N);
  EXPECT_EQ(errors, 0) << errors
                       << " lane mismatches in dpp_quad_perm (expected "
                          "per-quad XOR-1 swap)";
}

// ----- P6: ds_swizzle_b32 → llvm.amdgcn.ds.swizzle lift -----
//
// Spec: handle_ds.cpp::DS_SWIZZLE_B32 MODREP block. The handler
// reads `di.dsSwizzleImm` (extracted at decode time by
// `decodeDsSwizzleImm` in decode.cpp) and emits `ds.swizzle(value,
// imm)`. The classifier accepts QUAD_PERM, BITMASK_PERM, valid
// FFT_MODE, and valid ROTATE_MODE encodings (with strict reserved-
// bit validation); per-32-lane-half independence on wave64 follows
// from the bit-5-preservation hardware contract documented in the
// MODREP block.
//
// This test's kernel uses `ds_swizzle_b32 offset:0x081f` =
// BITMASK_PERM with and=0x1F, or=0, xor=2 (a SWAP-2 pattern).
// Distinct from the GPT-OSS `sum_bitmatrix_rows` corpus pattern
// (offset:0x041F = SWAP-1) — exercises a different bit of the
// XOR mask, catching imm-extraction bugs that happen to round-trip
// SWAP-1.
//
// Per-lane setup: input[L] = L.  Expected: output[L] = L XOR 2 (per
// 32-lane half on wave64).
static void doTestDsSwizzle() {
  printf("--- ds_swizzle_kernel (P6 end-to-end) ---\n");
  std::string path =
      std::string(GFX1250_DATA_DIR) + "/ds_swizzle_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942",
                                         "ds_swizzle_kernel");
  ASSERT_TRUE(result.success) << "Pipeline failed for ds_swizzle";
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  constexpr int N = 64;
  auto meta = transpiler::extractKernelMeta(data, "ds_swizzle_kernel");
  ASSERT_GE(meta.maxFlatWorkgroupSize, (uint32_t)N)
      << "Binary's max_flat (" << meta.maxFlatWorkgroupSize << ") < N (" << N
      << "); .hip / .cpp drift";

  std::vector<int> hOut(N, -1);
  int *dOut;
  HIP_ASSERT(hipMalloc(&dOut, N * sizeof(int)));
  HIP_ASSERT(hipMemset(dOut, 0xff, N * sizeof(int)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_ASSERT(hipModuleGetFunction(&func, mod, "ds_swizzle_kernel"));

  std::vector<uint8_t> argBuf(meta.kernargSegmentSize, 0);
  memcpy(argBuf.data() + 0, &dOut, 8);
  size_t argSz = argBuf.size();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuf.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz,
                    HIP_LAUNCH_PARAM_END};
  HIP_ASSERT(hipModuleLaunchKernel(func, 1, 1, 1, N, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr,
                                   nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hOut.data(), dOut, N * sizeof(int),
                       hipMemcpyDeviceToHost));

  int errors = 0;
  for (int L = 0; L < N; L++) {
    int expected = L ^ 2;
    if (hOut[L] != expected) {
      if (errors < 4)
        fprintf(stderr, "  lane %2d: got=%d exp=%d\n", L, hOut[L], expected);
      errors++;
    }
  }
  (void)hipFree(dOut);
  (void)hipModuleUnload(mod);
  printf("  Result: %d errors over %d lanes\n", errors, N);
  EXPECT_EQ(errors, 0) << errors
                       << " lane mismatches in ds_swizzle (expected "
                          "per-half XOR-2 BITMASK_PERM pattern)";
}

// ============================================================================
// TEST registrations — grouped by category for readability.
// Cross-lane regression block runs first (smallest, highest signal); then
// elementwise; then matmul (XFAIL block in xfail.cmake).
// ============================================================================
class Gfx1250Gpu : public GpuTest {};

// Cross-lane regression block (P2/P4/P5/P6 rewrites from
// hotswap/docs/wave-size-translation.md §5.3 + Triton corpus).
TEST_F(Gfx1250Gpu, Softmax)        { doTestSoftmax(); }        // P2 (permlanex16) implicit
TEST_F(Gfx1250Gpu, Permlane16Swap) { doTestPermlane16Swap(); } // P4 explicit
TEST_F(Gfx1250Gpu, DppQuadPerm)    { doTestDppQuadPerm(); }    // P5 explicit
TEST_F(Gfx1250Gpu, DsSwizzle)      { doTestDsSwizzle(); }      // P6 explicit

// Elementwise.
TEST_F(Gfx1250Gpu, Vecadd)         { doTestVecadd(); }
// `1.0f / sqrtf(x)` — IEEE fdiv expansion end-to-end gate; pairs with
// `lit_tests/v_div_scale_f32_literal_numer/` at the IR layer.
TEST_F(Gfx1250Gpu, RcpSqrt)        { doTestRcpSqrt(); }

// WMMA probe ladder (see doTestWmmaProbe / wmma_chain_probe_kernel.hip).
// One probe per (tiles × chain) combination; first failing probe localizes
// the root cause of the `Matmul128x128` numerical mismatch on the
// gfx1250 → gfx942 WMMA → MFMA lowering path.
TEST_F(Gfx1250Gpu, WmmaProbe_Chain1) {
  doTestWmmaProbe("wmma_chain1_kernel", /*numTiles=*/1,
                   /*chainDepth=*/1, /*numABuffers=*/1, /*numBBuffers=*/1);
}
TEST_F(Gfx1250Gpu, WmmaProbe_Chain2) {
  doTestWmmaProbe("wmma_chain2_kernel", /*numTiles=*/1,
                   /*chainDepth=*/2, /*numABuffers=*/1, /*numBBuffers=*/1);
}
TEST_F(Gfx1250Gpu, WmmaProbe_Chain4) {
  doTestWmmaProbe("wmma_chain4_kernel", /*numTiles=*/1,
                   /*chainDepth=*/4, /*numABuffers=*/1, /*numBBuffers=*/1);
}
TEST_F(Gfx1250Gpu, WmmaProbe_Parallel2) {
  doTestWmmaProbe("wmma_parallel2_kernel", /*numTiles=*/2,
                   /*chainDepth=*/1, /*numABuffers=*/1, /*numBBuffers=*/2);
}
TEST_F(Gfx1250Gpu, WmmaProbe_Parallel4) {
  doTestWmmaProbe("wmma_parallel4_kernel", /*numTiles=*/4,
                   /*chainDepth=*/1, /*numABuffers=*/1, /*numBBuffers=*/4);
}
TEST_F(Gfx1250Gpu, WmmaProbe_Parallel16) {
  doTestWmmaProbe("wmma_parallel16_kernel", /*numTiles=*/16,
                   /*chainDepth=*/1, /*numABuffers=*/4, /*numBBuffers=*/4);
}
TEST_F(Gfx1250Gpu, WmmaProbe_Parallel16_Chain4) {
  doTestWmmaProbe("wmma_parallel16_chain4_kernel", /*numTiles=*/16,
                   /*chainDepth=*/4, /*numABuffers=*/4, /*numBBuffers=*/4);
}

// Matmul gradation ladder. All six variants are expected to PASS
// since commit da404faf84 ("V_CMP -> V_CNDMASK per-lane-i1 shadow
// restores cross-widening"); the four diagnostic patterns (RowIdA /
// RowOnly124 / EvenRows / KStripedRow124) remain as positive
// regression guards for the exact shape that tripped the bug (warp 3
// / K-iter 0 / upper-VGPR-bank substitution from the narrow-ballot
// wave-mask round trip).
TEST_F(Gfx1250Gpu, Matmul64x64) {
  doTestMatmul("matmul_f16_gfx1250.hsaco", 128, 128, 64, "64x64 tile");
}
TEST_F(Gfx1250Gpu, Matmul128x128_1tile) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 128, 128, 128,
               "128x128 tile 1-tile");
}
TEST_F(Gfx1250Gpu, Matmul128x128) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 256, 256, 128,
               "128x128 tile");
}

// Uniform-input diagnostic: A = B = 1.0 ⇒ ref C[i,j] = K = 128
// everywhere. Position-invariant reference: any lane-routing bug
// that moves data between lanes/tiles still produces 128.0 at every
// output position, so this probe is insensitive to cross-widening
// wave-mask defects and remains as a separate gate on address-
// generation / tile-write / accumulator-init correctness alongside
// the four data-pattern probes below.
TEST_F(Gfx1250Gpu, Matmul128x128_1tile_RowIdA) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 128, 128, 128,
               "128x128 tile 1-tile rowIdA",
               MatmulDataPattern::RowIdA);
}
TEST_F(Gfx1250Gpu, Matmul128x128_1tile_RowOnly124) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 128, 128, 128,
               "128x128 tile 1-tile rowOnly124",
               MatmulDataPattern::RowOnly124);
}
TEST_F(Gfx1250Gpu, Matmul128x128_1tile_EvenRows) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 128, 128, 128,
               "128x128 tile 1-tile evenRows",
               MatmulDataPattern::EvenRows);
}
TEST_F(Gfx1250Gpu, Matmul128x128_1tile_KStripedRow124) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 128, 128, 128,
               "128x128 tile 1-tile kStripedRow124",
               MatmulDataPattern::KStripedRow124);
}
TEST_F(Gfx1250Gpu, Matmul128x128_1tile_UniformDiag) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 128, 128, 128,
               "128x128 tile 1-tile uniform-diag",
               MatmulDataPattern::Uniform);
}
