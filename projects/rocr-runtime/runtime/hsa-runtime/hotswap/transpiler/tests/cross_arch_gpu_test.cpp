#include "test_common.hpp"

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

// HIP_TRY is used inside testPostGSUKernel which returns TestResult via goto.
// This pattern does not translate to ASSERT_* (which requires void return),
// so we keep it as-is for the internal kernel test function.
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

  if (wsByte == 4) {
    std::vector<float> wsHost(bufs.wsElemCount, 1.0f);
    (void)hipMemcpy(bufs.WS, wsHost.data(), wBytes, hipMemcpyHostToDevice);
  } else {
    std::vector<double> wsHost(bufs.wsElemCount, 1.0);
    (void)hipMemcpy(bufs.WS, wsHost.data(), wBytes, hipMemcpyHostToDevice);
  }

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

  int scalarBase = getPostGSUScalarBase(meta);
  if (scalarBase < 0) {
    freePostGSUBuffers(bufs);
    return false;
  }

  kernargBuf.resize(meta.kernargSegmentSize, 0);

  std::memcpy(kernargBuf.data() + 0, &bufs.D, 8);
  std::memcpy(kernargBuf.data() + 8, &bufs.WS, 8);
  std::memcpy(kernargBuf.data() + 16, &bufs.C, 8);

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
// Test a single PostGSU kernel (uses HIP_TRY/goto, returns TestResult).
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

    int nativeMis = 0;
    float nativeErr = 0.0f;
    bool canVerify =
        verifyPostGSUResult(nBufs.D, info, nBufs, nativeErr, nativeMis);
    if (canVerify && nativeMis == 0) {
      nativeCorrect = true;
    } else if (canVerify) {
      fprintf(stderr,
              "    native math check: %d mismatches (maxErr=%e) --- args may "
              "be wrong for %s\n",
              nativeMis, nativeErr, kernelName.c_str());
      result = TestResult::Skipped;
      goto cleanup;
    }

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
// Vecadd cross-arch test
// ============================================================================
static void doTestVecaddCrossArch() {
  printf("=== Part 1: Vecadd Cross-Architecture Test (gfx1250 -> gfx942) ===\n");

  std::string vecaddPath = VECADD_CO_PATH;
  if (!fileExists(vecaddPath))
    GTEST_SKIP() << "Code object not found (build artifact): " << vecaddPath;

  auto vecaddData = transpiler::readFile(vecaddPath);
  ASSERT_FALSE(vecaddData.empty()) << "Cannot read " << vecaddPath;

  auto pipeResult = transpiler::runPipeline(vecaddData, "gfx1250", "gfx942",
                                            "_Z6vecaddPfS_S_i");
  ASSERT_TRUE(pipeResult.success) << "Pipeline failed for vecadd gfx1250->gfx942";
  printf("  Raised %d/%d instructions\n", pipeResult.liftedCount,
         pipeResult.totalCount);
  printf("  Transpiled HSACO: %zu bytes\n", pipeResult.hsaco.size());

  const int N = 1024;
  std::vector<float> hA(N), hB(N), hC(N, 0.0f);
  for (int i = 0; i < N; i++) {
    hA[i] = static_cast<float>(i);
    hB[i] = static_cast<float>(i * 2);
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  HIP_ASSERT(hipMalloc(&dA, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&dB, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&dC, N * sizeof(float)));
  HIP_ASSERT(hipMemcpy(dA, hA.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(dB, hB.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dC, 0, N * sizeof(float)));

  hipModule_t mod;
  HIP_ASSERT(hipModuleLoadData(&mod, pipeResult.hsaco.data()));
  hipFunction_t kernel;
  HIP_ASSERT(hipModuleGetFunction(&kernel, mod, "_Z6vecaddPfS_S_i"));

  struct { float *A, *B, *C; int N; } args = {dA, dB, dC, N};
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_ASSERT(hipModuleLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1,
                                   0, nullptr, nullptr, config));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(hC.data(), dC, N * sizeof(float), hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < N; i++) {
    float expected = hA[i] + hB[i];
    float diff = std::fabs(hC[i] - expected);
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

  EXPECT_EQ(errors, 0) << errors << "/" << N << " mismatches in vecadd cross-arch";
}

// ============================================================================
// Tensile batch test: metadata-driven with PostGSU-aware verification
// ============================================================================
static void doTestTensilePostGSU() {
  printf("=== Part 2: Tensile Kernels (gfx1200 -> gfx942) ===\n\n");

  std::string nativePath = NATIVE_HSACO_PATH;
  std::string sourcePath = SOURCE_HSACO_PATH;
  std::string sourceISA = "gfx1200";
  std::string targetISA = "gfx942";

  if (!fileExists(nativePath))
    GTEST_SKIP() << "Native HSACO not found (system file): " << nativePath;
  if (!fileExists(sourcePath))
    GTEST_SKIP() << "Source HSACO not found (system file): " << sourcePath;

  printf("Native (gfx942):  %s\n", nativePath.c_str());
  printf("Source (gfx1200): %s\n\n", sourcePath.c_str());

  auto nativeData = transpiler::readFile(nativePath);
  auto sourceData = transpiler::readFile(sourcePath);
  ASSERT_FALSE(nativeData.empty()) << "Failed to read " << nativePath;
  ASSERT_FALSE(sourceData.empty()) << "Failed to read " << sourcePath;

  auto kernelNames = transpiler::listKernelNames(sourceData);
  printf("Kernels in source: %zu\n\n", kernelNames.size());

  int verified = 0, pass = 0, pipelineFail = 0, hipErr = 0, mismatch = 0,
      noop = 0, skipped = 0;
  int postGSUAttempted = 0, nativeCorrectCount = 0;

  for (size_t idx = 0; idx < kernelNames.size(); idx++) {
    auto &kName = kernelNames[idx];
    printf("[%zu/%zu] %-60s ", idx + 1, kernelNames.size(), kName.c_str());
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
      auto pipeResult =
          transpiler::runPipeline(sourceData, sourceISA, targetISA, kName);
      ktr.name = kName;
      ktr.liftedCount = pipeResult.liftedCount;
      ktr.totalCount = pipeResult.totalCount;
      ktr.codeBytes = (int)pipeResult.hsaco.size();
      ktr.result =
          pipeResult.success ? TestResult::Skipped : TestResult::PipelineFail;
    }

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
      EXPECT_NE(ktr.result, TestResult::PipelineFail)
          << "Pipeline failed for " << kName;
      break;
    case TestResult::HipError:
      hipErr++;
      printf("HIP_ERROR\n");
      EXPECT_NE(ktr.result, TestResult::HipError)
          << "HIP runtime error for " << kName;
      break;
    case TestResult::Mismatch:
      mismatch++;
      if (ktr.nativeCorrect)
        nativeCorrectCount++;
      printf("MISMATCH  (maxErr=%e, native_correct=%s)\n", ktr.maxAbsErr,
             ktr.nativeCorrect ? "yes" : "no");
      EXPECT_NE(ktr.result, TestResult::Mismatch)
          << "Output mismatch for " << kName;
      break;
    case TestResult::NoOp:
      noop++;
      printf("NOOP\n");
      break;
    case TestResult::Skipped:
      skipped++;
      if (isPostGSU && pInfo.isGB)
        printf("SKIPPED  (_GB grouped-batch layout --- not yet supported)\n");
      else if (!isPostGSU)
        printf("SKIPPED  (non-PostGSU base kernel --- layout unknown)\n");
      else
        printf("SKIPPED\n");
      break;
    }
  }

  printf("\nCross-arch summary:\n");
  printf("  PostGSU attempted: %d, VERIFIED: %d, PASS: %d\n",
         postGSUAttempted, verified, pass);
  printf("  MISMATCH: %d, HIP_ERROR: %d, PIPELINE_FAIL: %d, SKIPPED: %d\n",
         mismatch, hipErr, pipelineFail, skipped);

  EXPECT_EQ(hipErr, 0)
      << hipErr << " kernel(s) had HIP runtime errors";
  EXPECT_EQ(pipelineFail, 0)
      << pipelineFail << " kernel(s) failed the transpiler pipeline";
  EXPECT_EQ(mismatch, 0)
      << mismatch << " kernel(s) produced mismatched outputs";
  EXPECT_GT(postGSUAttempted, 0)
      << "No PostGSU kernels were attempted -- check input files";
  EXPECT_GT(verified + pass, 0)
      << "No kernels verified or passed -- all attempted kernels failed";
}

class CrossArchGpu : public GpuTest {};

TEST_F(CrossArchGpu, VecaddCrossArch) { doTestVecaddCrossArch(); }

TEST_F(CrossArchGpu, TensilePostGSU) {
  if (!g_config.testAll)
    GTEST_SKIP() << "Skipped: pass --test-all to run TensilePostGSU corpus";
  doTestTensilePostGSU();
}
