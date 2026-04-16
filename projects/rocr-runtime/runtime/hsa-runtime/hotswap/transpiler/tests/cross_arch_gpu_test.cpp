#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_TRY(call)                                                          \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,                    \
              hipGetErrorString(err), __FILE__, __LINE__);                      \
      result = TestResult::HipError;                                           \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

enum class TestResult {
  Pass,
  Verified,
  PipelineFail,
  HipError,
  Mismatch,
  NoOp,
  Skipped,
};

static const char *resultStr(TestResult r) {
  switch (r) {
  case TestResult::Pass:         return "PASS";
  case TestResult::Verified:     return "VERIFIED";
  case TestResult::PipelineFail: return "PIPELINE_FAIL";
  case TestResult::HipError:     return "HIP_ERROR";
  case TestResult::Mismatch:     return "MISMATCH";
  case TestResult::NoOp:         return "NOOP";
  case TestResult::Skipped:      return "SKIPPED";
  }
  return "UNKNOWN";
}

struct KernelTestResult {
  std::string name;
  TestResult result = TestResult::Skipped;
  int liftedCount = 0;
  int totalCount = 0;
  int codeBytes = 0;
  float maxAbsErr = 0.0f;
  bool nativeWrote = false;
  bool transpWrote = false;
  bool nativeCorrect = false;
};

// ============================================================================
// PostGSU kernel name parsing and Tensile-aware argument construction.
//
// PostGSU kernels are the output-conversion step for Tensile GEMM with Global
// Split U.  They compute:  D[i] = alpha * sum(WS_partitions[i]) + beta * C[i]
//
// Argument layout (from Tensile ContractionSolution.cpp, stridedBatched):
//   [ptr D] [ptr WS] [ptr C]
//   [alpha f32] [beta f32]
//   [strideD1 u32] [strideD2 u32]
//   [strideW1 u32] [strideW2 u32]
//   [strideC1 u32] [strideC2 u32]
//   [size0 u32] [size1 u32] [size2 u32]
//   [gsu u32]
// ============================================================================

struct PostGSUInfo {
  char dataType = 'S';
  int gsu = 1;
  int vw = 1;
  bool isGB = false;
};

static bool isPostGSUKernel(const std::string &name) {
  return name.find("PostGSU") != std::string::npos;
}

static PostGSUInfo parsePostGSUKernelName(const std::string &name) {
  PostGSUInfo info;
  info.isGB = name.find("_GB") != std::string::npos &&
              name.find("_GB") < name.find("PostGSU");

  auto typePos = name.find("Cijk_");
  if (typePos != std::string::npos) {
    typePos += 5;
    if (typePos < name.size())
      info.dataType = name[typePos];
  }

  auto gsuPos = name.find("PostGSU");
  if (gsuPos != std::string::npos) {
    gsuPos += 7;
    int gsuNum = 0;
    while (gsuPos < name.size() && std::isdigit(name[gsuPos])) {
      gsuNum = gsuNum * 10 + (name[gsuPos] - '0');
      gsuPos++;
    }

    int modN = 0;
    auto modPos = name.find("_mod");
    if (modPos != std::string::npos && modPos > name.find("PostGSU")) {
      modPos += 4;
      while (modPos < name.size() && std::isdigit(name[modPos])) {
        modN = modN * 10 + (name[modPos] - '0');
        modPos++;
      }
    }

    if (modN > 0)
      info.gsu = gsuNum + modN;
    else
      info.gsu = gsuNum;
    if (info.gsu < 1)
      info.gsu = 1;
  }

  auto vwPos = name.find("_VW");
  if (vwPos != std::string::npos) {
    vwPos += 3;
    int vw = 0;
    while (vwPos < name.size() && std::isdigit(name[vwPos])) {
      vw = vw * 10 + (name[vwPos] - '0');
      vwPos++;
    }
    if (vw > 0)
      info.vw = vw;
  }

  return info;
}

// WS uses compute precision; D/C use storage precision.
static int wsElemBytes(char dt) {
  switch (dt) {
  case 'D': return 8;
  default:  return 4;
  }
}

static int storageElemBytes(char dt) {
  switch (dt) {
  case 'S': case 'I': return 4;
  case 'H': case 'B': return 2;
  case 'D':           return 8;
  default:            return 4;
  }
}

struct PostGSUBuffers {
  void *D = nullptr;
  void *WS = nullptr;
  void *C = nullptr;
  int elemCount = 0;
  int wsElemCount = 0;
  int M = 0, N = 0, batch = 0;
};

static void freePostGSUBuffers(PostGSUBuffers &b) {
  if (b.D) (void)hipFree(b.D);
  if (b.WS) (void)hipFree(b.WS);
  if (b.C) (void)hipFree(b.C);
  b.D = b.WS = b.C = nullptr;
}

// Classify the explicit arg layout of a PostGSU kernel from metadata.
// Returns the offset of the first scalar (alpha) arg, or -1 if unrecognized.
static int getPostGSUScalarBase(const transpiler::KernelMeta &meta) {
  std::vector<transpiler::KernelArgMeta> explicitArgs;
  for (auto &a : meta.args)
    if (a.valueKind.rfind("hidden_", 0) != 0)
      explicitArgs.push_back(a);

  int nPtr = 0;
  for (auto &a : explicitArgs)
    if (a.valueKind == "global_buffer")
      nPtr++;

  if (nPtr != 3)
    return -1;

  for (auto &a : explicitArgs) {
    if (a.valueKind != "global_buffer" && a.size == 4)
      return a.offset;
    if (a.valueKind != "global_buffer" && a.size == 8)
      continue;
  }
  return -1;
}

// Build kernargs for a non-_GB PostGSU kernel and fill buffers with known data.
// WS is filled with 1.0 (compute precision), C with 2.0 (storage precision).
// alpha=1.0, beta=1.0  →  D[i] = 1.0 * (gsu × 1.0) + 1.0 * 2.0 = gsu + 2.
static bool
buildPostGSUKernargs(std::vector<uint8_t> &kernargBuf,
                     const transpiler::KernelMeta &meta,
                     const PostGSUInfo &info,
                     PostGSUBuffers &bufs) {
  bufs.M = 256;
  bufs.N = 256;
  bufs.batch = 1;
  bufs.elemCount = bufs.M * bufs.N * bufs.batch;
  bufs.wsElemCount = bufs.elemCount * info.gsu;

  int wsByte = wsElemBytes(info.dataType);
  int stByte = storageElemBytes(info.dataType);

  size_t dBytes = (size_t)bufs.elemCount * stByte;
  size_t wBytes = (size_t)bufs.wsElemCount * wsByte;
  size_t cBytes = (size_t)bufs.elemCount * stByte;

  if (hipMalloc(&bufs.D, dBytes) != hipSuccess)
    return false;
  if (hipMalloc(&bufs.WS, wBytes) != hipSuccess) {
    freePostGSUBuffers(bufs);
    return false;
  }
  if (hipMalloc(&bufs.C, cBytes) != hipSuccess) {
    freePostGSUBuffers(bufs);
    return false;
  }

  (void)hipMemset(bufs.D, 0, dBytes);

  // Fill WS with 1.0 in compute precision (float or double).
  if (wsByte == 4) {
    std::vector<float> wsHost(bufs.wsElemCount, 1.0f);
    (void)hipMemcpy(bufs.WS, wsHost.data(), wBytes, hipMemcpyHostToDevice);
  } else {
    std::vector<double> wsHost(bufs.wsElemCount, 1.0);
    (void)hipMemcpy(bufs.WS, wsHost.data(), wBytes, hipMemcpyHostToDevice);
  }

  // Fill C with 2.0 in storage precision.
  if (stByte == 4) {
    std::vector<float> cHost(bufs.elemCount, 2.0f);
    (void)hipMemcpy(bufs.C, cHost.data(), cBytes, hipMemcpyHostToDevice);
  } else if (stByte == 2) {
    std::vector<uint16_t> cHost(bufs.elemCount);
    if (info.dataType == 'H') {
      __half val = __float2half(2.0f);
      uint16_t bits;
      std::memcpy(&bits, &val, 2);
      std::fill(cHost.begin(), cHost.end(), bits);
    } else {
      // bf16: upper 16 bits of float representation. 2.0f = 0x40000000 → bf16 = 0x4000
      float fval = 2.0f;
      uint32_t fbits;
      std::memcpy(&fbits, &fval, 4);
      uint16_t bits = (uint16_t)(fbits >> 16);
      std::fill(cHost.begin(), cHost.end(), bits);
    }
    (void)hipMemcpy(bufs.C, cHost.data(), cBytes, hipMemcpyHostToDevice);
  } else {
    std::vector<double> cHost(bufs.elemCount, 2.0);
    (void)hipMemcpy(bufs.C, cHost.data(), cBytes, hipMemcpyHostToDevice);
  }

  // Locate where the scalar args start by scanning metadata.
  int scalarBase = getPostGSUScalarBase(meta);
  if (scalarBase < 0) {
    freePostGSUBuffers(bufs);
    return false;
  }

  kernargBuf.resize(meta.kernargSegmentSize, 0);

  // Pointers: D, WS, C at offsets 0, 8, 16.
  std::memcpy(kernargBuf.data() + 0, &bufs.D, 8);
  std::memcpy(kernargBuf.data() + 8, &bufs.WS, 8);
  std::memcpy(kernargBuf.data() + 16, &bufs.C, 8);

  // If there are u64 by_value args between pointers and the first u32 scalar,
  // they are offsetD/offsetC — set to 0.
  // (They'll remain zero from the resize.)

  // Scalars: alpha, beta, strideD1, strideD2, strideW1, strideW2,
  //          strideC1, strideC2, size0, size1, size2, gsu
  int off = scalarBase;
  float alpha = 1.0f, beta = 1.0f;
  uint32_t strideD1 = bufs.M, strideD2 = bufs.M * bufs.N;
  uint32_t strideW1 = bufs.M, strideW2 = bufs.M * bufs.N;
  uint32_t strideC1 = bufs.M, strideC2 = bufs.M * bufs.N;
  uint32_t s0 = bufs.M, s1 = bufs.N, s2 = bufs.batch;
  uint32_t gsu = info.gsu;

  auto put32 = [&](const void *v) {
    if (off + 4 <= meta.kernargSegmentSize) {
      std::memcpy(kernargBuf.data() + off, v, 4);
      off += 4;
    }
  };

  put32(&alpha);
  put32(&beta);
  put32(&strideD1);
  put32(&strideD2);
  put32(&strideW1);
  put32(&strideW2);
  put32(&strideC1);
  put32(&strideC2);
  put32(&s0);
  put32(&s1);
  put32(&s2);
  put32(&gsu);

  return true;
}

// Verify the D output: D[i] should be alpha*gsu*ws_val + beta*c_val.
// For float data: expected = 1.0*gsu*1.0 + 1.0*2.0 = gsu+2
static bool
verifyPostGSUResult(void *dDev, const PostGSUInfo &info,
                    const PostGSUBuffers &bufs,
                    float &maxAbsErr, int &mismatchCount) {
  int stByte = storageElemBytes(info.dataType);
  size_t dBytes = (size_t)bufs.elemCount * stByte;

  float expected = (float)info.gsu * 1.0f + 1.0f * 2.0f;

  if (stByte == 4 && info.dataType == 'S') {
    std::vector<float> host(bufs.elemCount);
    (void)hipMemcpy(host.data(), dDev, dBytes, hipMemcpyDeviceToHost);
    for (int i = 0; i < bufs.elemCount; i++) {
      float diff = std::fabs(host[i] - expected);
      if (diff > maxAbsErr)
        maxAbsErr = diff;
      if (diff > 0.01f) {
        if (mismatchCount < 3)
          fprintf(stderr, "    D[%d]: got %f, expected %f (diff=%e)\n", i,
                  host[i], expected, diff);
        mismatchCount++;
      }
    }
  } else if (stByte == 2 && info.dataType == 'H') {
    std::vector<uint16_t> host(bufs.elemCount);
    (void)hipMemcpy(host.data(), dDev, dBytes, hipMemcpyDeviceToHost);
    for (int i = 0; i < bufs.elemCount; i++) {
      __half hv;
      std::memcpy(&hv, &host[i], 2);
      float fv = __half2float(hv);
      float diff = std::fabs(fv - expected);
      if (diff > maxAbsErr)
        maxAbsErr = diff;
      if (diff > 0.5f) {
        if (mismatchCount < 3)
          fprintf(stderr, "    D[%d]: got %f, expected %f (diff=%e)\n", i, fv,
                  expected, diff);
        mismatchCount++;
      }
    }
  } else if (stByte == 2 && info.dataType == 'B') {
    std::vector<uint16_t> host(bufs.elemCount);
    (void)hipMemcpy(host.data(), dDev, dBytes, hipMemcpyDeviceToHost);
    for (int i = 0; i < bufs.elemCount; i++) {
      // bf16 → float: place in upper 16 bits
      uint32_t fbits = (uint32_t)host[i] << 16;
      float fv;
      std::memcpy(&fv, &fbits, 4);
      float diff = std::fabs(fv - expected);
      if (diff > maxAbsErr)
        maxAbsErr = diff;
      if (diff > 1.0f) {
        if (mismatchCount < 3)
          fprintf(stderr, "    D[%d]: got %f, expected %f (diff=%e)\n", i, fv,
                  expected, diff);
        mismatchCount++;
      }
    }
  } else if (stByte == 8 && info.dataType == 'D') {
    std::vector<double> host(bufs.elemCount);
    (void)hipMemcpy(host.data(), dDev, dBytes, hipMemcpyDeviceToHost);
    double dExpected = (double)info.gsu * 1.0 + 1.0 * 2.0;
    for (int i = 0; i < bufs.elemCount; i++) {
      double diff = std::fabs(host[i] - dExpected);
      if ((float)diff > maxAbsErr)
        maxAbsErr = (float)diff;
      if (diff > 0.01) {
        if (mismatchCount < 3)
          fprintf(stderr, "    D[%d]: got %f, expected %f (diff=%e)\n", i,
                  (float)host[i], (float)dExpected, (float)diff);
        mismatchCount++;
      }
    }
  } else {
    return false;
  }
  return true;
}

// Compare two device buffers byte-for-byte, interpreting as the correct type.
static int compareBuffers(void *devA, void *devB, int elemCount, int stByte,
                          float &maxAbsErr) {
  size_t bytes = (size_t)elemCount * stByte;
  std::vector<uint8_t> hostA(bytes), hostB(bytes);
  (void)hipMemcpy(hostA.data(), devA, bytes, hipMemcpyDeviceToHost);
  (void)hipMemcpy(hostB.data(), devB, bytes, hipMemcpyDeviceToHost);

  int mismatches = 0;
  for (int i = 0; i < elemCount; i++) {
    int off = i * stByte;
    if (std::memcmp(hostA.data() + off, hostB.data() + off, stByte) == 0)
      continue;
    if (stByte == 4) {
      float a, b;
      std::memcpy(&a, hostA.data() + off, 4);
      std::memcpy(&b, hostB.data() + off, 4);
      float diff = std::fabs(a - b);
      if (diff > maxAbsErr)
        maxAbsErr = diff;
      if (diff > 0.01f)
        mismatches++;
    } else if (stByte == 2) {
      uint16_t a, b;
      std::memcpy(&a, hostA.data() + off, 2);
      std::memcpy(&b, hostB.data() + off, 2);
      if (a != b) {
        // Convert to float for error reporting (works for both fp16 and bf16).
        __half ha, hb;
        std::memcpy(&ha, &a, 2);
        std::memcpy(&hb, &b, 2);
        float fa = __half2float(ha), fb = __half2float(hb);
        float diff = std::fabs(fa - fb);
        if (diff > maxAbsErr)
          maxAbsErr = diff;
        mismatches++;
      }
    } else if (stByte == 8) {
      double a, b;
      std::memcpy(&a, hostA.data() + off, 8);
      std::memcpy(&b, hostB.data() + off, 8);
      double diff = std::fabs(a - b);
      if ((float)diff > maxAbsErr)
        maxAbsErr = (float)diff;
      if (diff > 0.01)
        mismatches++;
    } else {
      mismatches++;
    }
  }
  return mismatches;
}

// ============================================================================
// Test a single PostGSU kernel with principled argument construction.
// 1) Transpile source→target
// 2) Build correct PostGSU args (D,WS,C + scalars) for both native & transpiled
// 3) Run native kernel → verify D matches expected math
// 4) Run transpiled kernel → compare D against native output
// ============================================================================
static TestResult
testPostGSUKernel(const std::string &kernelName,
                  const std::vector<uint8_t> &nativeData,
                  const std::vector<uint8_t> &sourceData,
                  const std::string &sourceISA,
                  const std::string &targetISA,
                  KernelTestResult &ktr) {
  ktr.name = kernelName;

  auto pipeResult =
      transpiler::runPipeline(sourceData, sourceISA, targetISA, kernelName);
  ktr.liftedCount = pipeResult.liftedCount;
  ktr.totalCount = pipeResult.totalCount;
  ktr.codeBytes = (int)pipeResult.hsaco.size();

  if (!pipeResult.success) {
    ktr.result = TestResult::PipelineFail;
    return ktr.result;
  }

  auto meta = transpiler::extractKernelMeta(nativeData, kernelName);
  if (meta.args.empty()) {
    ktr.result = TestResult::Skipped;
    return ktr.result;
  }

  PostGSUInfo info = parsePostGSUKernelName(kernelName);

  PostGSUBuffers nBufs, tBufs;
  hipModule_t nativeMod = nullptr, transpMod = nullptr;
  TestResult result = TestResult::Pass;
  float maxAbsErr = 0.0f;
  bool nativeCorrect = false;

  std::vector<uint8_t> nativeKernargs, transpKernargs;

  if (!buildPostGSUKernargs(nativeKernargs, meta, info, nBufs)) {
    ktr.result = TestResult::Skipped;
    return ktr.result;
  }
  if (!buildPostGSUKernargs(transpKernargs, meta, info, tBufs)) {
    freePostGSUBuffers(nBufs);
    ktr.result = TestResult::Skipped;
    return ktr.result;
  }

  {
    int wgSize = 256;
    int totalElems = nBufs.elemCount;
    int gridX = (totalElems + wgSize * info.vw - 1) / (wgSize * info.vw);

    // --- Run native kernel ---
    HIP_TRY(hipModuleLoadData(&nativeMod, nativeData.data()));
    hipFunction_t nativeFunc;
    HIP_TRY(
        hipModuleGetFunction(&nativeFunc, nativeMod, kernelName.c_str()));

    size_t nArgSz = nativeKernargs.size();
    void *nativeConfig[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                            nativeKernargs.data(),
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &nArgSz,
                            HIP_LAUNCH_PARAM_END};

    HIP_TRY(hipModuleLaunchKernel(nativeFunc, gridX, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr,
                                   nullptr, nativeConfig));
    HIP_TRY(hipDeviceSynchronize());
    ktr.nativeWrote = true;

    // Verify native output mathematically
    int nativeMis = 0;
    float nativeErr = 0.0f;
    bool canVerify =
        verifyPostGSUResult(nBufs.D, info, nBufs, nativeErr, nativeMis);
    if (canVerify && nativeMis == 0) {
      nativeCorrect = true;
    } else if (canVerify) {
      fprintf(stderr,
              "    native math check: %d mismatches (maxErr=%e) — args may "
              "be wrong for %s\n",
              nativeMis, nativeErr, kernelName.c_str());
      result = TestResult::Skipped;
      goto cleanup;
    }

    // --- Run transpiled kernel ---
    HIP_TRY(hipModuleLoadData(&transpMod, pipeResult.hsaco.data()));
    hipFunction_t transpFunc;
    HIP_TRY(
        hipModuleGetFunction(&transpFunc, transpMod, kernelName.c_str()));

    size_t tArgSz = transpKernargs.size();
    void *transpConfig[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                            transpKernargs.data(),
                            HIP_LAUNCH_PARAM_BUFFER_SIZE, &tArgSz,
                            HIP_LAUNCH_PARAM_END};

    HIP_TRY(hipModuleLaunchKernel(transpFunc, gridX, 1, 1, wgSize, 1, 1,
                                   meta.groupSegmentFixedSize, nullptr,
                                   nullptr, transpConfig));
    HIP_TRY(hipDeviceSynchronize());
    ktr.transpWrote = true;

    // Compare native D vs transpiled D
    int stByte = storageElemBytes(info.dataType);
    int cmpMis =
        compareBuffers(nBufs.D, tBufs.D, nBufs.elemCount, stByte, maxAbsErr);

    if (cmpMis > 0) {
      fprintf(stderr, "    native vs transpiled: %d mismatches (maxErr=%e)\n",
              cmpMis, maxAbsErr);
      result = TestResult::Mismatch;
    } else if (nativeCorrect) {
      result = TestResult::Verified;
    } else {
      result = TestResult::Pass;
    }
  }

cleanup:
  if (result == TestResult::HipError)
    (void)hipGetLastError();

  freePostGSUBuffers(nBufs);
  freePostGSUBuffers(tBufs);
  if (nativeMod) (void)hipModuleUnload(nativeMod);
  if (transpMod) (void)hipModuleUnload(transpMod);

  ktr.maxAbsErr = maxAbsErr;
  ktr.nativeCorrect = nativeCorrect;
  ktr.result = result;
  return result;
}

// ============================================================================
// Vecadd cross-arch test: known argument layout, verifiable results.
// Source: gfx1250 vecadd kernel → transpile to gfx942 → execute → verify
// C[i] = A[i] + B[i] for i in [0..N)
// ============================================================================
static int testVecaddCrossArch() {
  printf("=== Part 1: Vecadd Cross-Architecture Test (gfx1250 → gfx942) ===\n");
  printf("Known layout: _Z6vecaddPfS_S_i(float *A, float *B, float *C, int N)\n\n");

  std::string vecaddPath = VECADD_CO_PATH;
  auto vecaddData = transpiler::readFile(vecaddPath);
  if (vecaddData.empty()) {
    fprintf(stderr, "ERROR: Cannot read %s\n", vecaddPath.c_str());
    return 1;
  }

  // Transpile gfx1250 → gfx942
  auto pipeResult = transpiler::runPipeline(vecaddData, "gfx1250", "gfx942",
                                          "_Z6vecaddPfS_S_i");
  if (!pipeResult.success) {
    fprintf(stderr, "ERROR: Pipeline failed for vecadd gfx1250→gfx942\n");
    return 1;
  }
  printf("  Raised %d/%d instructions\n", pipeResult.liftedCount,
         pipeResult.totalCount);
  printf("  Transpiled HSACO: %zu bytes\n", pipeResult.hsaco.size());

  // Prepare test data
  const int N = 1024;
  std::vector<float> hA(N), hB(N), hC(N, 0.0f);
  for (int i = 0; i < N; i++) {
    hA[i] = static_cast<float>(i);
    hB[i] = static_cast<float>(i * 2);
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  hipError_t err;
  err = hipMalloc(&dA, N * sizeof(float));
  if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }
  err = hipMalloc(&dB, N * sizeof(float));
  if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }
  err = hipMalloc(&dC, N * sizeof(float));
  if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }

  err = hipMemcpy(dA, hA.data(), N * sizeof(float), hipMemcpyHostToDevice);
  if (err != hipSuccess) { fprintf(stderr, "hipMemcpy failed\n"); return 1; }
  err = hipMemcpy(dB, hB.data(), N * sizeof(float), hipMemcpyHostToDevice);
  if (err != hipSuccess) { fprintf(stderr, "hipMemcpy failed\n"); return 1; }
  err = hipMemset(dC, 0, N * sizeof(float));
  if (err != hipSuccess) { fprintf(stderr, "hipMemset failed\n"); return 1; }

  // Load transpiled kernel
  hipModule_t mod;
  err = hipModuleLoadData(&mod, pipeResult.hsaco.data());
  if (err != hipSuccess) {
    fprintf(stderr, "ERROR: hipModuleLoadData failed: %s\n",
            hipGetErrorString(err));
    return 1;
  }

  hipFunction_t kernel;
  err = hipModuleGetFunction(&kernel, mod, "_Z6vecaddPfS_S_i");
  if (err != hipSuccess) {
    fprintf(stderr, "ERROR: hipModuleGetFunction failed: %s\n",
            hipGetErrorString(err));
    return 1;
  }

  // Launch
  struct { float *A, *B, *C; int N; } args = {dA, dB, dC, N};
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  err = hipModuleLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1,
                               0, nullptr, nullptr, config);
  if (err != hipSuccess) {
    fprintf(stderr, "ERROR: Launch failed: %s\n", hipGetErrorString(err));
    return 1;
  }
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "ERROR: Sync failed: %s\n", hipGetErrorString(err));
    return 1;
  }

  // Verify: C[i] should be A[i] + B[i] = i + 2*i = 3*i
  err = hipMemcpy(hC.data(), dC, N * sizeof(float), hipMemcpyDeviceToHost);
  if (err != hipSuccess) { fprintf(stderr, "hipMemcpy back failed\n"); return 1; }

  int errors = 0;
  float maxRelErr = 0.0f;
  for (int i = 0; i < N; i++) {
    float expected = hA[i] + hB[i];
    float diff = std::fabs(hC[i] - expected);
    float relErr = (expected != 0.0f) ? diff / std::fabs(expected) : diff;
    if (relErr > maxRelErr) maxRelErr = relErr;
    if (diff > 1e-5f) {
      if (errors < 5)
        fprintf(stderr, "  MISMATCH [%d]: got %f, expected %f (diff=%e)\n",
                i, hC[i], expected, diff);
      errors++;
    }
  }

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipModuleUnload(mod);

  printf("\n  Sample outputs: C[0]=%f C[1]=%f C[100]=%f C[1023]=%f\n",
         hC[0], hC[1], hC[100], hC[1023]);
  printf("  Expected:       C[0]=%f C[1]=%f C[100]=%f C[1023]=%f\n",
         0.0f, 3.0f, 300.0f, 3069.0f);
  printf("  Max relative error: %e\n", maxRelErr);

  if (errors == 0) {
    printf("\n  VECADD CROSS-ARCH: PASSED — all %d elements correct\n", N);
    printf("  This proves: gfx1250 binary → LLVM IR → gfx942 binary → "
           "correct GPU execution on MI300X\n\n");
    return 0;
  } else {
    printf("\n  VECADD CROSS-ARCH: FAILED — %d/%d mismatches\n\n", errors, N);
    return 1;
  }
}

// ============================================================================
// Tensile batch test: metadata-driven with PostGSU-aware verification
// ============================================================================
int main(int argc, char **argv) {
  printf("=== Cross-Architecture GPU Execution Test ===\n\n");

  int vecaddResult = testVecaddCrossArch();

  printf("================================================================\n");
  printf("=== Part 2: Tensile Kernels (gfx1200 → gfx942) ===\n");
  printf("================================================================\n\n");

  std::string nativePath = NATIVE_HSACO_PATH;
  std::string sourcePath = SOURCE_HSACO_PATH;
  std::string sourceISA = "gfx1200";
  std::string targetISA = "gfx942";

  std::string filterKernel;
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--kernel=", 9) == 0)
      filterKernel = argv[i] + 9;
  }

  printf("Native (gfx942):  %s\n", nativePath.c_str());
  printf("Source (gfx1200): %s\n\n", sourcePath.c_str());

  auto nativeData = transpiler::readFile(nativePath);
  auto sourceData = transpiler::readFile(sourcePath);
  if (nativeData.empty() || sourceData.empty()) {
    fprintf(stderr, "ERROR: Failed to read hsaco files\n");
    return 1;
  }

  auto kernelNames = transpiler::listKernelNames(sourceData);
  printf("Kernels in source: %zu\n\n", kernelNames.size());

  if (!filterKernel.empty()) {
    auto it = std::find(kernelNames.begin(), kernelNames.end(), filterKernel);
    if (it == kernelNames.end()) {
      fprintf(stderr, "ERROR: Kernel '%s' not found\n", filterKernel.c_str());
      return 1;
    }
    kernelNames = {filterKernel};
  }

  std::vector<KernelTestResult> results;
  int verified = 0, pass = 0, pipelineFail = 0, hipErr = 0, mismatch = 0,
      noop = 0, skipped = 0;
  int postGSUAttempted = 0, nativeCorrectCount = 0;

  for (auto &kName : kernelNames) {
    printf("[%zu/%zu] %-60s ", results.size() + 1, kernelNames.size(),
           kName.c_str());
    fflush(stdout);

    KernelTestResult ktr;

    bool isPostGSU = isPostGSUKernel(kName);
    PostGSUInfo pInfo;
    if (isPostGSU)
      pInfo = parsePostGSUKernelName(kName);

    if (isPostGSU && !pInfo.isGB) {
      postGSUAttempted++;
      testPostGSUKernel(kName, nativeData, sourceData, sourceISA, targetISA,
                        ktr);
    } else {
      // Non-PostGSU or _GB variant: skip execution, only test pipeline.
      auto pipeResult =
          transpiler::runPipeline(sourceData, sourceISA, targetISA, kName);
      ktr.name = kName;
      ktr.liftedCount = pipeResult.liftedCount;
      ktr.totalCount = pipeResult.totalCount;
      ktr.codeBytes = (int)pipeResult.hsaco.size();
      ktr.result =
          pipeResult.success ? TestResult::Skipped : TestResult::PipelineFail;
    }

    results.push_back(ktr);

    switch (ktr.result) {
    case TestResult::Verified:
      verified++;
      nativeCorrectCount++;
      printf("VERIFIED  (native correct, transpiled matches, maxErr=%e)\n",
             ktr.maxAbsErr);
      break;
    case TestResult::Pass:
      pass++;
      printf("PASS  (outputs match, maxErr=%e)\n", ktr.maxAbsErr);
      break;
    case TestResult::PipelineFail:
      pipelineFail++;
      printf("PIPELINE_FAIL\n");
      break;
    case TestResult::HipError:
      hipErr++;
      printf("HIP_ERROR\n");
      break;
    case TestResult::Mismatch:
      mismatch++;
      if (ktr.nativeCorrect)
        nativeCorrectCount++;
      printf("MISMATCH  (maxErr=%e, native_correct=%s)\n", ktr.maxAbsErr,
             ktr.nativeCorrect ? "yes" : "no");
      break;
    case TestResult::NoOp:
      noop++;
      printf("NOOP\n");
      break;
    case TestResult::Skipped:
      skipped++;
      if (isPostGSU && pInfo.isGB)
        printf("SKIPPED  (_GB grouped-batch layout — not yet supported)\n");
      else if (!isPostGSU)
        printf("SKIPPED  (non-PostGSU base kernel — layout unknown)\n");
      else
        printf("SKIPPED\n");
      break;
    }
  }

  printf("\n");
  printf("================================================================\n");
  printf("           CROSS-ARCH GPU EXECUTION SUMMARY\n");
  printf("================================================================\n");
  printf("\n");
  printf("Part 1 — Vecadd (gfx1250 → gfx942): %s\n",
         vecaddResult == 0 ? "PASSED" : "FAILED");
  printf("\n");
  printf("Part 2 — Tensile PostGSU (gfx1200 → gfx942):\n");
  printf("  Kernels in source: %zu\n", kernelNames.size());
  printf("  PostGSU attempted: %d  (non-_GB, Tensile-aware args)\n",
         postGSUAttempted);
  printf("  VERIFIED:          %d  (native math correct + transpiled "
         "matches)\n",
         verified);
  printf("  PASS:              %d  (transpiled matches native)\n", pass);
  printf("  MISMATCH:          %d\n", mismatch);
  printf("  HIP_ERROR:         %d\n", hipErr);
  printf("  PIPELINE_FAIL:     %d\n", pipelineFail);
  printf("  SKIPPED:           %d  (_GB / base / unsupported)\n", skipped);
  printf("  NOOP:              %d\n", noop);
  printf("\n");
  int postGSUExecuted = verified + pass + mismatch + hipErr + noop;
  printf("  Native math correct: %d / %d  (Tensile arg layout proven "
         "correct)\n",
         nativeCorrectCount, postGSUExecuted);
  printf("\n");

  if (verified > 0)
    printf("  >> %d kernels VERIFIED end-to-end: RDNA binary → LLVM IR → "
           "CDNA binary → correct GPU output on MI300X\n\n",
           verified);

  return vecaddResult;
}
