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
// Permlane16 swap: end-to-end regression for CROSS_LANE_SURVEY P4
// ============================================================================
//
// The pre-built `.hsaco` is sourced from
// `test_data/gfx1250/permlane16_swap_kernel.hip` (committed alongside
// the .hsaco for reproducibility). Per-lane setup:
//
//   vdst_in[L]  = L
//   src0_in[L]  = 1000 + L
//
// After v_permlane16_swap_b32 (per the VOP_PERMLANE_SWAP profile in
// VOP1Instructions.td):
//
//   new_vdst[L]      = src0_in[L XOR 16]  = 1000 + (L XOR 16)
//   new_src0_out[L]  = vdst_in[L XOR 16]  = (L XOR 16)
//
// The lifted gfx942 wave64 kernel runs each 32-lane half
// independently under modulo-replication, so the expected output
// for every L in [0, 64) is the per-half XOR-16 partner. The test
// verifies all 64 entries — direct empirical check that the
// emulation produces the source-intended swap.
//
// Without this test, the only check that the lifted ds_bpermute
// chain actually swaps correctly is the embedded probe results in
// the handler comment block. This test gives the P4 emulation a
// CI-resident regression gate; a future change that breaks the
// XOR-16 partner pattern OR the per-32-lane-half independence
// would fail here.
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

  auto meta = transpiler::extractKernelMeta(data, "permlane16_swap_kernel");

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

// ============================================================================
// TEST registrations
// ============================================================================
class Gfx1250Gpu : public GpuTest {};

TEST_F(Gfx1250Gpu, Softmax) { doTestSoftmax(); }
TEST_F(Gfx1250Gpu, Vecadd)  { doTestVecadd(); }
TEST_F(Gfx1250Gpu, Permlane16Swap) { doTestPermlane16Swap(); }

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
