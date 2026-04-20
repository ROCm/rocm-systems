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
// Matmul: C = A @ B, A/B fp16, C fp32
// ============================================================================
static void doTestMatmul(const char *hsacoFile, int M, int N, int K,
                         const char *label) {
  printf("--- matmul_kernel (%s, M=%d N=%d K=%d) ---\n", label, M, N, K);
  std::string path = std::string(GFX1250_DATA_DIR) + "/" + hsacoFile;
  auto data = transpiler::readFile(path);
  ASSERT_FALSE(data.empty()) << "Cannot read " << path;

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942", "matmul_kernel");
  ASSERT_TRUE(result.success) << "Pipeline failed for matmul " << label;
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  std::vector<__half> hA(M * K), hB(K * N);
  std::vector<float> hC(M * N, 0.0f);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (int i = 0; i < M * K; i++) hA[i] = __float2half(dist(rng));
  for (int i = 0; i < K * N; i++) hB[i] = __float2half(dist(rng));

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

  int errors = 0;
  float maxErr = 0;
  for (int i = 0; i < M * N; i++) {
    float diff = std::fabs(hC[i] - ref[i]);
    float denom = std::max(std::fabs(ref[i]), 1e-6f);
    float relErr = diff / denom;
    if (diff > maxErr) maxErr = diff;
    if (relErr > 0.05f && diff > 0.01f) {
      if (errors < 3)
        fprintf(stderr, "  [%d] got=%f ref=%f diff=%e relErr=%e\n", i, hC[i], ref[i], diff, relErr);
      errors++;
    }
  }

  (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC); (void)hipModuleUnload(mod);
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

// Matmul (XFAIL block, see tests/xfail.cmake).
TEST_F(Gfx1250Gpu, Matmul64x64) {
  doTestMatmul("matmul_f16_gfx1250.hsaco", 128, 128, 64, "64x64 tile");
}
// XFAIL: see tests/xfail.cmake (hipError 700 in large-tile matmul)
TEST_F(Gfx1250Gpu, Matmul128x128_1tile) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 128, 128, 128,
               "128x128 tile 1-tile");
}
// XFAIL: see tests/xfail.cmake (hipError 700 / hang in large-tile matmul)
TEST_F(Gfx1250Gpu, Matmul128x128) {
  doTestMatmul("matmul_f16_large_gfx1250.hsaco", 256, 256, 128,
               "128x128 tile");
}
