// End-to-end test: gfx1250 Triton kernels → LLVM IR → gfx942 → execute on GPU
//
// Kernels tested:
//   vecadd_kernel  (fp16):  C[i] = A[i] + B[i]
//   matmul_kernel  (fp16→f32):  C = A @ B  (64×64×32 and 128×128×32 tile sizes)
//   softmax_kernel (fp32):  row-wise softmax
//
// Each kernel is:
//   1. Raised from gfx1250 binary to LLVM IR
//   2. Lowered to gfx942 via llc → llvm-mc → ld.lld
//   3. Loaded and launched on the GPU via HIP
//   4. Output verified against CPU reference

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

#define HIP_CHECK(call)                                                      \
  do {                                                                       \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,                  \
              hipGetErrorString(err), __FILE__, __LINE__);                    \
      return false;                                                          \
    }                                                                        \
  } while (0)

static const char *TEST_DATA_DIR = GFX1250_TEST_DATA_DIR;

// ============================================================================
// Vecadd: C[i] = A[i] + B[i], fp16
// Signature: (a_ptr: *fp16, b_ptr: *fp16, c_ptr: *fp16, N: i32)
// ============================================================================
static bool testVecadd() {
  printf("--- vecadd_kernel (fp16, BLOCK_SIZE=1024) ---\n");
  std::string path = std::string(TEST_DATA_DIR) + "/vecadd_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  if (data.empty()) { fprintf(stderr, "Cannot read %s\n", path.c_str()); return false; }

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942", "vecadd_kernel");
  if (!result.success) { fprintf(stderr, "Pipeline failed\n"); return false; }
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
  HIP_CHECK(hipMalloc(&dA, N * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dB, N * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dC, N * sizeof(__half)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, N * sizeof(__half)));

  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_CHECK(hipModuleGetFunction(&func, mod, "vecadd_kernel"));

  auto meta = transpiler::extractKernelMeta(data, "vecadd_kernel");

  // Build kernarg buffer matching HSACO metadata layout (48 bytes for Triton vecadd:
  // 3 ptrs + 1 i32 explicit + 2 internal Triton ptrs at offsets 32,40)
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
  HIP_CHECK(hipModuleLaunchKernel(func, gridX, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hC.data(), dC, N * sizeof(__half), hipMemcpyDeviceToHost));

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

  hipFree(dA); hipFree(dB); hipFree(dC); hipModuleUnload(mod);
  printf("  Result: %d errors, maxErr=%e\n", errors, maxErr);
  return errors == 0;
}

// ============================================================================
// Matmul: C = A @ B, A/B fp16, C fp32
// Signature: (a_ptr: *fp16, b_ptr: *fp16, c_ptr: *fp32,
//             M: i32, N: i32, K: i32,
//             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)
// ============================================================================
static bool testMatmul(const char *hsacoFile, int M, int N, int K, const char *label) {
  printf("--- matmul_kernel (%s, M=%d N=%d K=%d) ---\n", label, M, N, K);
  std::string path = std::string(TEST_DATA_DIR) + "/" + hsacoFile;
  auto data = transpiler::readFile(path);
  if (data.empty()) { fprintf(stderr, "Cannot read %s\n", path.c_str()); return false; }

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942", "matmul_kernel");
  if (!result.success) { fprintf(stderr, "Pipeline failed\n"); return false; }
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  std::vector<__half> hA(M * K), hB(K * N);
  std::vector<float> hC(M * N, 0.0f);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (int i = 0; i < M * K; i++) hA[i] = __float2half(dist(rng));
  for (int i = 0; i < K * N; i++) hB[i] = __float2half(dist(rng));

  __half *dA, *dB; float *dC;
  HIP_CHECK(hipMalloc(&dA, M * K * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dB, K * N * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dC, M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), M * K * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, M * N * sizeof(float)));

  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_CHECK(hipModuleGetFunction(&func, mod, "matmul_kernel"));

  // Row-major: A is MxK (stride_am=K, stride_ak=1), B is KxN (stride_bk=N, stride_bn=1)
  struct {
    __half *a, *b; float *c;
    int m, n, k;
    int stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn;
  } args = {dA, dB, dC, M, N, K, K, 1, N, 1, N, 1};
  size_t argSz = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSz, HIP_LAUNCH_PARAM_END};

  auto meta = transpiler::extractKernelMeta(data, "matmul_kernel");
  int wgSize = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 256;
  // Grid: num_pid_m * num_pid_n (each tile is BLOCK_M x BLOCK_N)
  // For the small kernel: BLOCK_M=BLOCK_N=64; large: BLOCK_M=BLOCK_N=128
  // We determine tile size from file name
  int blockM = (std::string(hsacoFile).find("large") != std::string::npos) ? 128 : 64;
  int blockN = blockM;
  int numPidM = (M + blockM - 1) / blockM;
  int numPidN = (N + blockN - 1) / blockN;
  int gridX = numPidM * numPidN;

  HIP_CHECK(hipModuleLaunchKernel(func, gridX, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference
  std::vector<float> ref(M * N, 0.0f);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int kk = 0; kk < K; kk++)
        sum += __half2float(hA[i * K + kk]) * __half2float(hB[kk * N + j]);
      ref[i * N + j] = sum;
    }

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

  hipFree(dA); hipFree(dB); hipFree(dC); hipModuleUnload(mod);
  printf("  Result: %d errors, maxErr=%e\n", errors, maxErr);
  return errors == 0;
}

// ============================================================================
// Softmax: row-wise softmax, fp32
// Signature: (output_ptr: *fp32, input_ptr: *fp32,
//             input_row_stride: i32, output_row_stride: i32, n_cols: i32)
// ============================================================================
static bool testSoftmax() {
  printf("--- softmax_kernel (fp32, BLOCK_SIZE=1024) ---\n");
  std::string path = std::string(TEST_DATA_DIR) + "/softmax_gfx1250.hsaco";
  auto data = transpiler::readFile(path);
  if (data.empty()) { fprintf(stderr, "Cannot read %s\n", path.c_str()); return false; }

  auto result = transpiler::runPipeline(data, "gfx1250", "gfx942", "softmax_kernel");
  if (!result.success) { fprintf(stderr, "Pipeline failed\n"); return false; }
  printf("  Pipeline: raised %d/%d insts, HSACO=%zu bytes\n",
         result.liftedCount, result.totalCount, result.hsaco.size());

  const int nRows = 1, nCols = 512;
  const int BLOCK_SIZE = 1024;
  std::vector<float> hIn(nRows * nCols), hOut(nRows * nCols, 0.0f);
  for (auto &v : hIn) v = 1.0f;  // constant input: softmax should give 1/nCols

  // Pad GPU allocations: the transpiled kernel's buffer descriptor uses a large
  // NUM_RECORDS (0xFFFFFF), so OOB column stores (column >= nCols but < BLOCK_SIZE)
  // write past the row boundary.  On gfx1250 these hit mapped pages silently;
  // on gfx942 flat stores fault.  Allocate enough for BLOCK_SIZE columns on the
  // last row to absorb the overflow.
  size_t allocElems = 1024 * 1024;
  float *dIn, *dOut;
  HIP_CHECK(hipMalloc(&dIn, allocElems * sizeof(float)));
  HIP_CHECK(hipMalloc(&dOut, allocElems * sizeof(float)));
  HIP_CHECK(hipMemcpy(dIn, hIn.data(), nRows * nCols * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dOut, 0, allocElems * sizeof(float)));

  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, result.hsaco.data()));
  hipFunction_t func;
  HIP_CHECK(hipModuleGetFunction(&func, mod, "softmax_kernel"));

  auto meta = transpiler::extractKernelMeta(data, "softmax_kernel");
  printf("  meta: kernargSegmentSize=%d, maxFlatWorkgroupSize=%d, "
         "groupSegmentFixedSize=%d\n",
         meta.kernargSegmentSize, meta.maxFlatWorkgroupSize,
         meta.groupSegmentFixedSize);
  printf("  dIn=%p  dOut=%p  allocElems=%zu\n", (void*)dIn, (void*)dOut, allocElems);

  // Build kernarg buffer matching HSACO metadata layout (48 bytes for Triton softmax:
  // 2 ptrs + 3 i32 explicit + 2 internal Triton ptrs at offsets 32,40)
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
  HIP_CHECK(hipModuleLaunchKernel(func, nRows, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hOut.data(), dOut, nRows * nCols * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference: softmax per row
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

  // Diagnostic: dump first 16 output values
  printf("  First 16 outputs:");
  for (int i = 0; i < 16 && i < nRows * nCols; i++)
    printf(" %.4f", hOut[i]);
  printf("\n");
  printf("  First 16 refs:   ");
  for (int i = 0; i < 16 && i < nRows * nCols; i++)
    printf(" %.4f", ref[i]);
  printf("\n");
  // Check for patterns
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

  hipFree(dIn); hipFree(dOut); hipModuleUnload(mod);
  printf("  Result: %d errors, maxErr=%e\n", errors, maxErr);
  return errors == 0;
}

// ============================================================================
int main() {
  printf("================================================================\n");
  printf("  gfx1250 → gfx942 End-to-End GPU Test\n");
  printf("================================================================\n\n");

  int pass = 0, fail = 0;

  if (testSoftmax()) { pass++; printf("  >> PASS\n\n"); }
  else               { fail++; printf("  >> FAIL\n\n"); }

  if (testVecadd()) { pass++; printf("  >> PASS\n\n"); }
  else              { fail++; printf("  >> FAIL\n\n"); }

  if (testMatmul("matmul_f16_gfx1250.hsaco", 128, 128, 64, "64x64 tile"))
    { pass++; printf("  >> PASS\n\n"); }
  else { fail++; printf("  >> FAIL\n\n"); }

  if (testMatmul("matmul_f16_large_gfx1250.hsaco", 256, 256, 128, "128x128 tile"))
    { pass++; printf("  >> PASS\n\n"); }
  else { fail++; printf("  >> FAIL\n\n"); }

  printf("================================================================\n");
  printf("  SUMMARY: %d PASS, %d FAIL out of 4 kernels\n", pass, fail);
  printf("================================================================\n");

  return fail;
}
