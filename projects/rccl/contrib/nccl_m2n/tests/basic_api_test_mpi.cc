/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * basic_api_test_mpi — gtest-parametrized multi-process MPI driver for
 * the basic_api matrix. Each TestCase is one gtest parameter (or one
 * TestCase × algorithm parameter when --algorithm all is used).
 *
 * Bootstrap: standard ncclGetUniqueId + MPI_Bcast + ncclCommInitRank.
 * Drives the shared core in basic_api_test_core.h.
 ************************************************************************/

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_m2n.h"
#include "basic_api_test_core.h"

namespace {

static MPI_Comm testMpiWorld() {
  return MPI_COMM_WORLD; // NOLINT(bugprone-casting-through-void)
}

static MPI_Datatype testMpiByte() {
  return MPI_BYTE; // NOLINT(bugprone-casting-through-void)
}

static MPI_Datatype testMpiInt() {
  return MPI_INT; // NOLINT(bugprone-casting-through-void)
}

static MPI_Op testMpiMin() {
  return MPI_MIN; // NOLINT(bugprone-casting-through-void)
}

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = (cmd);                                                     \
    if (e != MPI_SUCCESS) {                                            \
      fprintf(stderr, "MPI error %s:%d: %d\n", __FILE__, __LINE__, e); \
      MPI_Abort(testMpiWorld(), 1);                                    \
    }                                                                  \
  } while (0)

/* ======================================================================
 * Bootstrap-specific TestEnv hooks.
 * ====================================================================*/

static void mpiBarrier(TestEnv* env) {
  (void)env;
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

static int mpiAllreduceMinInt(TestEnv* env, int local) {
  (void)env;
  int g = local;
  MPICHECK(MPI_Allreduce(&local, &g, 1, testMpiInt(), testMpiMin(), testMpiWorld()));
  return g;
}

static bool mpiIsRank0Printer(TestEnv* env) {
  return env->rank == 0;
}

/* ======================================================================
 * gtest parameter registry state.
 * ====================================================================*/

struct MpiParam {
  TestCase tc;
  std::string algorithmEnv; /* RING or DIRECT */
  ApiKind api = ApiKind::Window;
  int streamPoolMode = -1; /* -1 = inherited, 0 = user stream, 1 = internal pool */
};

/* gtest pretty-printer hook for MpiParam — found via ADL by
 * INSTANTIATE_TEST_SUITE_P's value-printer. The lookup is template-driven,
 * so plain `-Wunused-function` cannot see the use; mark it accordingly. */
[[maybe_unused]] static void printTo(const MpiParam& param, std::ostream* os) {
  *os << param.algorithmEnv << ":" << (param.api == ApiKind::Default ? "default" : "window");
  if (param.streamPoolMode >= 0) {
    *os << ":" << (param.streamPoolMode == 0 ? "user_stream" : "internal_pool");
  }
  *os << ":" << param.tc.name;
}

static BasicApiCliArgs gCli;
static const char* gResolvedCopyAlgorithm = "PACK";
static std::vector<TestCase> gCases;
static int gWorldRank = 0;
static int gWorldSize = 0;
static int gNumDevices = 0;
static int gDevice = 0;
static ncclComm_t gComm = nullptr;
static cudaStream_t gStream = nullptr;
static ncclM2nHandle_t gM2nHandle = nullptr;
static void* gBuffer = nullptr;
static size_t gBufferBytes = 4096;
static void* gCopyBuffer = nullptr;
static size_t gCopyBufferBytes = 0;
static ncclComm_t gGroupCommA = nullptr;
static ncclComm_t gGroupCommB = nullptr;
static cudaStream_t gGroupStreamA = nullptr;
static cudaStream_t gGroupStreamB = nullptr;
static void* gGroupBufferA = nullptr;
static void* gGroupBufferB = nullptr;
static ncclComm_t gReducedBucketCommA = nullptr;
static ncclComm_t gReducedBucketCommB = nullptr;
static ncclComm_t gNaturalReducedBucketCommA = nullptr;
static ncclComm_t gNaturalReducedBucketCommB = nullptr;
static std::string gActiveAlgorithm;
static int gActiveStreamPoolMode = -1;
static bool gInitialUseInternalStreamsEnvSet = false;
static std::string gInitialUseInternalStreamsEnv;

static void CUDART_CB waitForStagingReuseRelease(void* data) {
  auto* release = static_cast<std::atomic<bool>*>(data);
  while (!release->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

constexpr size_t kReducedBucketBytes = 4096;

struct ReducedBucketTestContext {
  bool inA = false;
  bool inB = false;
  int rankA = -1;
  int rankB = -1;
  ncclComm_t commA = nullptr;
  ncclComm_t commB = nullptr;
  cudaStream_t streamA = nullptr;
  cudaStream_t streamB = nullptr;
  void* bufferA = nullptr;
  void* bufferB = nullptr;
  int meshDims[2] = {1, 1};
  size_t localShape[1] = {kReducedBucketBytes};
  int placements[2] = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_REPLICATE};
  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
  ncclDistTensor_t srcA = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  ncclDistTensor_t dstA = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  ncclDistTensor_t srcB = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  ncclDistTensor_t dstB = NCCL_M2N_DIST_TENSOR_INITIALIZER;
};

static ncclDistTensor_t makeReducedBucketTensor(void* data, ncclMesh_t* mesh, size_t* localShape,
                                                int* placements) {
  ncclDistTensor_t tensor = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  tensor.dataPtr = data;
  tensor.localShape = localShape;
  tensor.ndims = 1;
  tensor.dtype = ncclUint8;
  tensor.mesh = mesh;
  tensor.placements = placements;
  return tensor;
}

static void initializeReducedBucketTest(ReducedBucketTestContext* ctx, ncclComm_t* retainedCommA,
                                        ncclComm_t* retainedCommB) {
  ncclUniqueId ids[2];
  if (gWorldRank == 0) {
    TEST_NCCLCHECK(ncclGetUniqueId(&ids[0]));
    TEST_NCCLCHECK(ncclGetUniqueId(&ids[1]));
  }
  MPICHECK(MPI_Bcast(ids, sizeof(ids), testMpiByte(), 0, testMpiWorld()));

  ctx->inA = gWorldRank == 0 || gWorldRank == 2;
  ctx->inB = gWorldRank == 1 || gWorldRank == 2;
  ctx->rankA = gWorldRank == 0 ? 0 : 1;
  ctx->rankB = gWorldRank == 1 ? 0 : 1;
  if (ctx->inA) {
    TEST_NCCL_COMM_CHECK(*retainedCommA, ncclCommInitRank(retainedCommA, 2, ids[0], ctx->rankA));
    ctx->commA = *retainedCommA;
    TEST_CUDACHECK(cudaStreamCreateWithFlags(&ctx->streamA, cudaStreamNonBlocking));
    TEST_CUDACHECK(cudaMalloc(&ctx->bufferA, kReducedBucketBytes));
  }
  if (ctx->inB) {
    TEST_NCCL_COMM_CHECK(*retainedCommB, ncclCommInitRank(retainedCommB, 2, ids[1], ctx->rankB));
    ctx->commB = *retainedCommB;
    TEST_CUDACHECK(cudaStreamCreateWithFlags(&ctx->streamB, cudaStreamNonBlocking));
    TEST_CUDACHECK(cudaMalloc(&ctx->bufferB, kReducedBucketBytes));
  }

  ctx->srcMesh.ndims = 2;
  ctx->srcMesh.dims = ctx->meshDims;
  ctx->srcMesh.startRank = 0;
  ctx->dstMesh.ndims = 2;
  ctx->dstMesh.dims = ctx->meshDims;
  ctx->dstMesh.startRank = 1;
  ctx->srcA = makeReducedBucketTensor(ctx->inA && ctx->rankA == 0 ? ctx->bufferA : nullptr, &ctx->srcMesh,
                                      ctx->localShape, ctx->placements);
  ctx->dstA = makeReducedBucketTensor(ctx->inA && ctx->rankA == 1 ? ctx->bufferA : nullptr, &ctx->dstMesh,
                                      ctx->localShape, ctx->placements);
  ctx->srcB = makeReducedBucketTensor(ctx->inB && ctx->rankB == 0 ? ctx->bufferB : nullptr, &ctx->srcMesh,
                                      ctx->localShape, ctx->placements);
  ctx->dstB = makeReducedBucketTensor(ctx->inB && ctx->rankB == 1 ? ctx->bufferB : nullptr, &ctx->dstMesh,
                                      ctx->localShape, ctx->placements);
}

static void expectReducedBucketPattern(void* buffer, unsigned char expected) {
  std::array<unsigned char, kReducedBucketBytes> actual{};
  TEST_CUDACHECK(cudaMemcpy(actual.data(), buffer, actual.size(), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::all_of(actual.begin(), actual.end(),
                          [expected](unsigned char value) { return value == expected; }));
}

static void warmReducedBucketCommunicators(ReducedBucketTestContext* ctx) {
  constexpr int kWarmPatternA = 0x11;
  constexpr int kWarmPatternB = 0x22;
  if (ctx->inA) {
    TEST_CUDACHECK(cudaMemsetAsync(ctx->bufferA, ctx->rankA == 0 ? kWarmPatternA : 0, kReducedBucketBytes,
                                  ctx->streamA));
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, ctx->commA, &ctx->srcA, &ctx->dstA, ctx->streamA));
    TEST_NCCL_ASYNC_CHECK(ctx->commA, ctx->streamA);
    if (ctx->rankA == 1) expectReducedBucketPattern(ctx->bufferA, kWarmPatternA);
  }
  if (ctx->inB) {
    TEST_CUDACHECK(cudaMemsetAsync(ctx->bufferB, ctx->rankB == 0 ? kWarmPatternB : 0, kReducedBucketBytes,
                                  ctx->streamB));
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, ctx->commB, &ctx->srcB, &ctx->dstB, ctx->streamB));
    TEST_NCCL_ASYNC_CHECK(ctx->commB, ctx->streamB);
    if (ctx->rankB == 1) expectReducedBucketPattern(ctx->bufferB, kWarmPatternB);
  }
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

static void destroyReducedBucketUserResources(ReducedBucketTestContext* ctx) {
  if (ctx->bufferA != nullptr) TEST_CUDACHECK(cudaFree(ctx->bufferA));
  if (ctx->streamA != nullptr) TEST_CUDACHECK(cudaStreamDestroy(ctx->streamA));
  if (ctx->bufferB != nullptr) TEST_CUDACHECK(cudaFree(ctx->bufferB));
  if (ctx->streamB != nullptr) TEST_CUDACHECK(cudaStreamDestroy(ctx->streamB));
}

static bool runAllAlgorithms() {
  return basicApiRunAllAlgorithms(gCli);
}

static const char* requestedAlgorithmEnv() {
  return basicApiRequestedAlgorithmEnv(gCli, gWorldRank == 0);
}

static std::vector<MpiParam> selectedParams() {
  std::vector<MpiParam> params;
  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);

  auto appendParams = [&](const TestCase& tc, const char* algo, ApiKind api) {
    /* window_null asserts that ncclReshardWithWindow tolerates a NULL window.
     * The default API takes no window at all, so the case is meaningless there
     * and would only inflate default-API case counts. */
    if (tc.group == "window_null" && api != ApiKind::Window) {
      return;
    }
    if (tc.group == "stream_ordering" || tc.group == "stream_churn" || tc.group == "graph_capture") {
      params.push_back(MpiParam{tc, algo, api, 1});
      params.push_back(MpiParam{tc, algo, api, 0});
    } else {
      params.push_back(MpiParam{tc, algo, api, -1});
    }
  };

  std::vector<ApiKind> apis;
  if (basicApiRunAllApis(gCli)) {
    apis = {ApiKind::Window, ApiKind::Default};
  } else {
    apis = {basicApiRequestedApi(gCli)};
  }

  if (runAllAlgorithms()) {
    const char* algos[] = {"RING", "DIRECT"};
    for (const char* algo : algos) {
      for (ApiKind api : apis) {
        for (const TestCase& tc : cases) {
          appendParams(tc, algo, api);
        }
      }
    }
  } else {
    const char* algo = requestedAlgorithmEnv();
    for (ApiKind api : apis) {
      for (const TestCase& tc : cases) {
        appendParams(tc, algo, api);
      }
    }
  }
  return params;
}

static std::string gtestCaseName(const ::testing::TestParamInfo<MpiParam>& info) {
  std::string prefix = info.param.algorithmEnv;
  prefix += (info.param.api == ApiKind::Default) ? "_default" : "_window";
  if (info.param.streamPoolMode >= 0) {
    prefix += (info.param.streamPoolMode == 0) ? "_user_stream" : "_internal_pool";
  }
  return basicApiGtestCaseName(info.param.tc.name, info.index, prefix.c_str());
}

static void applyStreamPoolMode(int streamPoolMode) {
  if (streamPoolMode == 0) {
    testSetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS", "0");
  } else if (streamPoolMode > 0) {
    testSetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS", "1");
  } else if (gInitialUseInternalStreamsEnvSet) {
    testSetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS", gInitialUseInternalStreamsEnv.c_str());
  } else {
    testUnsetEnv("NCCL_RESHARD_USE_INTERNAL_STREAMS");
  }
}

static void activateRuntime(const MpiParam& param, bool bForceReset = false) {
  if (!bForceReset && gActiveAlgorithm == param.algorithmEnv && gActiveStreamPoolMode == param.streamPoolMode) return;

  if (testUsesNonBlockingComm()) {
    TEST_NCCL_ASYNC_CHECK(gComm, gStream);
  } else {
    TEST_CUDACHECK(cudaStreamSynchronize(gStream));
  }
  gResolvedCopyAlgorithm = basicApiConfigureReshardEnv(gCli, param.algorithmEnv.c_str());
  applyStreamPoolMode(param.streamPoolMode);
  TEST_NCCLCHECK(ncclM2nFinalize(gM2nHandle));
  gM2nHandle = nullptr;
  TEST_NCCLCHECK(ncclM2nInit(&gM2nHandle, NULL));
  gActiveAlgorithm = param.algorithmEnv;
  gActiveStreamPoolMode = param.streamPoolMode;
}

class BasicApiMpiTest : public ::testing::TestWithParam<MpiParam> {};

static CaseResult runStreamChurn(const TestCase& tc, TestEnv* env) {
  CaseShape shape;
  const char* skipReason = nullptr;
  if (!caseFeasibleAt(tc, env->worldSize, &shape, &skipReason)) {
    return runOneCase(tc, env);
  }

  constexpr int kStreamCount = 3;
  cudaStream_t freshStreams[kStreamCount] = {};
  const bool bUseFreshStreams = env->rank >= shape.srcTotal;
  if (bUseFreshStreams) {
    // Keep every handle live so CUDA cannot recycle one and hide cache churn.
    for (cudaStream_t& stream : freshStreams) {
      TEST_CUDACHECK(cudaStreamCreate(&stream));
    }
  }

  CaseResult result = makePass();
  for (int i = 0; i < kStreamCount; i++) {
    env->stream = bUseFreshStreams ? freshStreams[i] : gStream;
    result = runOneCase(tc, env);
    if (result.status != CASE_PASS) {
      break;
    }
  }

  if (bUseFreshStreams) {
    for (cudaStream_t stream : freshStreams) {
      TEST_CUDACHECK(cudaStreamDestroy(stream));
    }
  }
  env->stream = gStream;
  return result;
}

static CaseResult runStreamOrdering(const TestCase& tc, TestEnv* env) {
  cudaStream_t alternateStream = nullptr;
  TEST_CUDACHECK(cudaStreamCreateWithFlags(&alternateStream, cudaStreamNonBlocking));
  env->alternateStream = alternateStream;
  CaseResult result = runOneCase(tc, env);
  env->alternateStream = nullptr;
  TEST_CUDACHECK(cudaStreamDestroy(alternateStream));
  return result;
}

TEST(M2nGroupMpiTest, OverlappingCommunicatorsPreserveBucketOrder) {
  if (gCli.filter == nullptr || strcmp(gCli.filter, "group_mixed_context") != 0) {
    GTEST_SKIP() << "needs --filter group_mixed_context";
  }
  if (gWorldSize < 3) {
    GTEST_SKIP() << "needs at least 3 ranks";
  }

  ncclUniqueId ids[2];
  if (gWorldRank == 0) {
    TEST_NCCLCHECK(ncclGetUniqueId(&ids[0]));
    TEST_NCCLCHECK(ncclGetUniqueId(&ids[1]));
  }
  MPICHECK(MPI_Bcast(ids, sizeof(ids), testMpiByte(), 0, testMpiWorld()));

  const bool inA = gWorldRank <= 1;
  const bool inB = gWorldRank == 1 || gWorldRank == 2;
  const int rankA = gWorldRank;
  const int rankB = gWorldRank - 1;
  if (inA) {
    TEST_NCCL_COMM_CHECK(gGroupCommA, ncclCommInitRank(&gGroupCommA, 2, ids[0], rankA));
    TEST_CUDACHECK(cudaStreamCreate(&gGroupStreamA));
    TEST_NCCLCHECK(ncclMemAlloc(&gGroupBufferA, 4096));
  }
  if (inB) {
    TEST_NCCL_COMM_CHECK(gGroupCommB, ncclCommInitRank(&gGroupCommB, 2, ids[1], rankB));
    TEST_CUDACHECK(cudaStreamCreate(&gGroupStreamB));
    TEST_NCCLCHECK(ncclMemAlloc(&gGroupBufferB, 4096));
  }

  int meshDims[NCCL_RESHARD_MAX_MESH_DIMS] = {1, 1};
  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  srcMesh.ndims = NCCL_RESHARD_MAX_MESH_DIMS;
  srcMesh.dims = meshDims;
  srcMesh.startRank = 0;
  ncclMesh_t dstMesh = srcMesh;
  dstMesh.startRank = 1;
  auto makeTensor = [](void* data, ncclMesh_t* mesh) {
    static size_t localShape[1] = {32};
    static int placements[NCCL_RESHARD_MAX_MESH_DIMS] = {
      NCCL_RESHARD_REPLICATE,
      NCCL_RESHARD_REPLICATE,
    };
    ncclDistTensor_t tensor = NCCL_M2N_DIST_TENSOR_INITIALIZER;
    tensor.dataPtr = data;
    tensor.localShape = localShape;
    tensor.ndims = 1;
    tensor.dtype = ncclUint8;
    tensor.mesh = mesh;
    tensor.placements = placements;
    return tensor;
  };
  std::array<ncclDistTensor_t, 2> srcA{};
  std::array<ncclDistTensor_t, 2> dstA{};
  std::array<ncclDistTensor_t, 2> srcB{};
  std::array<ncclDistTensor_t, 2> dstB{};
  for (size_t i = 0; i < 2; i++) {
    const size_t offset = i * 64;
    if (inA) {
      srcA[i] = makeTensor(rankA == 0 ? static_cast<char*>(gGroupBufferA) + offset : nullptr, &srcMesh);
      dstA[i] = makeTensor(rankA == 1 ? static_cast<char*>(gGroupBufferA) + offset : nullptr, &dstMesh);
      if (rankA == 0) TEST_CUDACHECK(cudaMemsetAsync(srcA[i].dataPtr, 0x20 + static_cast<int>(i), 32, gGroupStreamA));
      if (rankA == 1) TEST_CUDACHECK(cudaMemsetAsync(dstA[i].dataPtr, 0, 32, gGroupStreamA));
    }
    if (inB) {
      srcB[i] = makeTensor(rankB == 0 ? static_cast<char*>(gGroupBufferB) + offset : nullptr, &srcMesh);
      dstB[i] = makeTensor(rankB == 1 ? static_cast<char*>(gGroupBufferB) + offset : nullptr, &dstMesh);
      if (rankB == 0) TEST_CUDACHECK(cudaMemsetAsync(srcB[i].dataPtr, 0x40 + static_cast<int>(i), 32, gGroupStreamB));
      if (rankB == 1) TEST_CUDACHECK(cudaMemsetAsync(dstB[i].dataPtr, 0, 32, gGroupStreamB));
    }
  }

  TEST_NCCLCHECK(ncclM2nGroupStart());
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[0], &dstA[0], gGroupStreamA));
  if (inB) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommB, &srcB[0], &dstB[0], gGroupStreamB));
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[1], &dstA[1], gGroupStreamA));
  if (inB) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommB, &srcB[1], &dstB[1], gGroupStreamB));
  TEST_NCCLCHECK(ncclM2nGroupEnd());
  if (inA) TEST_NCCL_ASYNC_CHECK(gGroupCommA, gGroupStreamA);
  if (inB) TEST_NCCL_ASYNC_CHECK(gGroupCommB, gGroupStreamB);

  std::array<unsigned char, 32> actual{};
  auto expectPattern = [&](const ncclDistTensor_t& tensor, int expected) {
    TEST_CUDACHECK(cudaMemcpy(actual.data(), tensor.dataPtr, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual[0]);
    EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [&](unsigned char value) { return value == actual[0]; }));
  };
  for (size_t i = 0; i < 2; i++) {
    if (inA && rankA == 1) expectPattern(dstA[i], 0x20 + static_cast<int>(i));
    if (inB && rankB == 1) expectPattern(dstB[i], 0x40 + static_cast<int>(i));
  }

  if (inA && rankA == 1) {
    TEST_CUDACHECK(cudaMemsetAsync(dstA[0].dataPtr, 0, 32, gGroupStreamA));
    TEST_CUDACHECK(cudaMemsetAsync(dstA[1].dataPtr, 0, 32, gGroupStreamA));
  }
  TEST_NCCLCHECK(ncclM2nGroupStart());
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[0], &dstA[0], gGroupStreamA));
  if (inB) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommB, nullptr, nullptr, gGroupStreamB));
  if (inA) TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[1], &dstA[1], gGroupStreamA));
  ncclResult_t bucketOrderResult = ncclM2nGroupEnd();
  EXPECT_EQ(inB ? ncclInvalidArgument : ncclSuccess, bucketOrderResult);
  if (inA) TEST_NCCL_ASYNC_CHECK(gGroupCommA, gGroupStreamA);
  if (inA && rankA == 1) {
    expectPattern(dstA[0], 0x20);
    expectPattern(dstA[1], 0x21);
  }
  MPICHECK(MPI_Barrier(testMpiWorld()));

  if (inA && rankA == 1) TEST_CUDACHECK(cudaMemsetAsync(dstA[0].dataPtr, 0, 32, gGroupStreamA));
  TEST_NCCLCHECK(ncclM2nGroupStart());
  if (inA) {
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, &srcA[0], &dstA[0], gGroupStreamA));
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, gGroupCommA, nullptr, nullptr, gGroupStreamA));
  }
  ncclResult_t entryOrderResult = ncclM2nGroupEnd();
  EXPECT_EQ(inA ? ncclInvalidArgument : ncclSuccess, entryOrderResult);
  if (inA) TEST_NCCL_ASYNC_CHECK(gGroupCommA, gGroupStreamA);
  if (inA && rankA == 1) expectPattern(dstA[0], 0x20);
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

TEST(PackMpiTest, ReducedBucketWaitsForDelayedRemoteConsumer) {
  if (gCli.filter == nullptr || strcmp(gCli.filter, "pack_reduced_bucket") != 0) {
    GTEST_SKIP() << "needs --filter pack_reduced_bucket";
  }
  if (gWorldSize < 3) {
    GTEST_SKIP() << "needs at least 3 ranks";
  }

  /* Warm both communicators before contention so the first-use host-RMA ring
   * cannot serialize B behind A and accidentally mask a missing GRANT lease. */
  ReducedBucketTestContext ctx;
  initializeReducedBucketTest(&ctx, &gReducedBucketCommA, &gReducedBucketCommB);
  warmReducedBucketCommunicators(&ctx);

  constexpr int kPatternA = 0x2a;
  constexpr int kPatternB = 0x4b;
  if (ctx.inA) {
    TEST_CUDACHECK(cudaMemsetAsync(ctx.bufferA, ctx.rankA == 0 ? kPatternA : 0, kReducedBucketBytes, ctx.streamA));
  }
  if (ctx.inB) {
    TEST_CUDACHECK(cudaMemsetAsync(ctx.bufferB, ctx.rankB == 0 ? kPatternB : 0, kReducedBucketBytes, ctx.streamB));
  }

  std::atomic<bool> releaseA{false};
  if (ctx.inA && ctx.rankA == 1) {
    TEST_CUDACHECK(cudaLaunchHostFunc(ctx.streamA, waitForStagingReuseRelease, &releaseA));
  }
  if (ctx.inA) {
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, ctx.commA, &ctx.srcA, &ctx.dstA, ctx.streamA));
  }
  MPICHECK(MPI_Barrier(testMpiWorld()));

  if (ctx.inB) {
    TEST_NCCLCHECK(ncclReshard(gM2nHandle, ctx.commB, &ctx.srcB, &ctx.dstB, ctx.streamB));
  }
  MPICHECK(MPI_Barrier(testMpiWorld()));
  if (ctx.inA && ctx.rankA == 1) {
    releaseA.store(true, std::memory_order_release);
  }

  if (ctx.inA) TEST_NCCL_ASYNC_CHECK(ctx.commA, ctx.streamA);
  if (ctx.inB) TEST_NCCL_ASYNC_CHECK(ctx.commB, ctx.streamB);
  if (ctx.inA && ctx.rankA == 1) expectReducedBucketPattern(ctx.bufferA, kPatternA);
  if (ctx.inB && ctx.rankB == 1) expectReducedBucketPattern(ctx.bufferB, kPatternB);

  destroyReducedBucketUserResources(&ctx);
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

TEST(PackMpiTest, ReducedBucketWorksWithoutExternalPhaseBarriers) {
  if (gCli.filter == nullptr || strcmp(gCli.filter, "pack_reduced_bucket") != 0) {
    GTEST_SKIP() << "needs --filter pack_reduced_bucket";
  }
  if (gWorldSize < 3) {
    GTEST_SKIP() << "needs at least 3 ranks";
  }

  /* Submit repeated warmed A/B waves without an MPI transfer-phase barrier or
   * intermediate stream synchronization. Per-iteration snapshots ensure a
   * later successful call cannot hide an earlier overwrite. */
  ReducedBucketTestContext ctx;
  initializeReducedBucketTest(&ctx, &gNaturalReducedBucketCommA, &gNaturalReducedBucketCommB);
  warmReducedBucketCommunicators(&ctx);

  constexpr int kIterations = 8;
  constexpr size_t kSnapshotBytes = kIterations * kReducedBucketBytes;
  void* snapshotsA = nullptr;
  void* snapshotsB = nullptr;
  if (ctx.inA && ctx.rankA == 1) TEST_CUDACHECK(cudaMalloc(&snapshotsA, kSnapshotBytes));
  if (ctx.inB && ctx.rankB == 1) TEST_CUDACHECK(cudaMalloc(&snapshotsB, kSnapshotBytes));

  for (int i = 0; i < kIterations; i++) {
    const int patternA = 0x30 + i;
    const int patternB = 0x50 + i;
    if (ctx.inA) {
      TEST_CUDACHECK(cudaMemsetAsync(ctx.bufferA, ctx.rankA == 0 ? patternA : 0, kReducedBucketBytes, ctx.streamA));
      TEST_NCCLCHECK(ncclReshard(gM2nHandle, ctx.commA, &ctx.srcA, &ctx.dstA, ctx.streamA));
      if (ctx.rankA == 1) {
        TEST_CUDACHECK(cudaMemcpyAsync(static_cast<char*>(snapshotsA) + static_cast<size_t>(i) * kReducedBucketBytes,
                                      ctx.bufferA, kReducedBucketBytes, cudaMemcpyDeviceToDevice, ctx.streamA));
      }
    }
    if (ctx.inB) {
      TEST_CUDACHECK(cudaMemsetAsync(ctx.bufferB, ctx.rankB == 0 ? patternB : 0, kReducedBucketBytes, ctx.streamB));
      TEST_NCCLCHECK(ncclReshard(gM2nHandle, ctx.commB, &ctx.srcB, &ctx.dstB, ctx.streamB));
      if (ctx.rankB == 1) {
        TEST_CUDACHECK(cudaMemcpyAsync(static_cast<char*>(snapshotsB) + static_cast<size_t>(i) * kReducedBucketBytes,
                                      ctx.bufferB, kReducedBucketBytes, cudaMemcpyDeviceToDevice, ctx.streamB));
      }
    }
  }

  if (ctx.inA) TEST_NCCL_ASYNC_CHECK(ctx.commA, ctx.streamA);
  if (ctx.inB) TEST_NCCL_ASYNC_CHECK(ctx.commB, ctx.streamB);

  std::vector<unsigned char> actual(kSnapshotBytes);
  if (snapshotsA != nullptr) {
    TEST_CUDACHECK(cudaMemcpy(actual.data(), snapshotsA, actual.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kIterations; i++) {
      SCOPED_TRACE(::testing::Message() << "communicator A iteration " << i);
      const unsigned char expected = static_cast<unsigned char>(0x30 + i);
      const auto first = actual.begin() + static_cast<size_t>(i) * kReducedBucketBytes;
      EXPECT_TRUE(std::all_of(first, first + kReducedBucketBytes,
                              [expected](unsigned char value) { return value == expected; }));
    }
  }
  if (snapshotsB != nullptr) {
    TEST_CUDACHECK(cudaMemcpy(actual.data(), snapshotsB, actual.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kIterations; i++) {
      SCOPED_TRACE(::testing::Message() << "communicator B iteration " << i);
      const unsigned char expected = static_cast<unsigned char>(0x50 + i);
      const auto first = actual.begin() + static_cast<size_t>(i) * kReducedBucketBytes;
      EXPECT_TRUE(std::all_of(first, first + kReducedBucketBytes,
                              [expected](unsigned char value) { return value == expected; }));
    }
  }

  if (snapshotsA != nullptr) TEST_CUDACHECK(cudaFree(snapshotsA));
  if (snapshotsB != nullptr) TEST_CUDACHECK(cudaFree(snapshotsB));
  destroyReducedBucketUserResources(&ctx);
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

TEST_P(BasicApiMpiTest, Reshard) {
  const MpiParam& param = GetParam();
  SCOPED_TRACE(param.tc.name);
  SCOPED_TRACE(param.algorithmEnv);

  activateRuntime(param, param.tc.group == "graph_capture");

  TestEnv env{};
  env.rank = gWorldRank;
  env.worldSize = gWorldSize;
  env.device = gDevice;
  env.comm = gComm;
  env.stream = gStream;
  env.m2nHandle = gM2nHandle;
  env.buffer = gBuffer;
  env.bufferBytes = gBufferBytes;
  env.copyBuffer = gCopyBuffer;
  env.copyBufferBytes = gCopyBufferBytes;
  env.apiKind = param.api;
  env.expectPack = strcmp(gResolvedCopyAlgorithm, "PACK") == 0;
  env.verbose = gCli.verbose;
  env.barrier = mpiBarrier;
  env.allreduceMinInt = mpiAllreduceMinInt;
  env.isRank0Printer = mpiIsRank0Printer;
  env.ctx = nullptr;

  CaseResult res;
  if (param.tc.group == "stream_churn") {
    res = runStreamChurn(param.tc, &env);
  } else if (param.tc.group == "stream_ordering") {
    res = runStreamOrdering(param.tc, &env);
  } else {
    res = runOneCase(param.tc, &env);
  }

  if (res.status == CASE_SKIP) {
    env.barrier(&env);
    GTEST_SKIP() << ((res.skipReason != nullptr) ? res.skipReason : "skipped");
    return;
  }

  if (res.status == CASE_FAIL) ADD_FAILURE() << ((res.failReason != nullptr) ? res.failReason : "case failed");
  env.barrier(&env);
}

// The mixed-communicator regression intentionally selects no matrix cases.
// Allow the standalone group test to run without GoogleTest treating that as
// an uninstantiated parameterized suite failure.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(BasicApiMpiTest);

INSTANTIATE_TEST_SUITE_P(Matrix, BasicApiMpiTest, ::testing::ValuesIn(selectedParams()), gtestCaseName);

static int initMpiRuntime() {
  TEST_CUDACHECK(cudaGetDeviceCount(&gNumDevices));
  MPI_Comm localComm;
  int localRank = 0;
  MPICHECK(MPI_Comm_split_type(testMpiWorld(), MPI_COMM_TYPE_SHARED, gWorldRank, MPI_INFO_NULL, &localComm));
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_free(&localComm));
  gDevice = localRank % (gNumDevices > 0 ? gNumDevices : 1);
  TEST_CUDACHECK(cudaSetDevice(gDevice));

  ncclUniqueId uid;
  if (gWorldRank == 0) TEST_NCCLCHECK(ncclGetUniqueId(&uid));
  MPICHECK(MPI_Bcast(&uid, sizeof(uid), testMpiByte(), 0, testMpiWorld()));

  TEST_NCCL_COMM_CHECK(gComm, ncclCommInitRank(&gComm, gWorldSize, uid, gWorldRank));

  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);
  gBufferBytes = computeMaxBufferBytes(cases, gWorldSize);

  const char* initialAlgorithm = requestedAlgorithmEnv();
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — captured once during MPI setup
  const char* initialUseInternalStreamsEnv = getenv("NCCL_RESHARD_USE_INTERNAL_STREAMS");
  if (initialUseInternalStreamsEnv != nullptr) {
    gInitialUseInternalStreamsEnvSet = true;
    gInitialUseInternalStreamsEnv = initialUseInternalStreamsEnv;
  }
  gResolvedCopyAlgorithm = basicApiConfigureReshardEnv(gCli, initialAlgorithm);
  TEST_NCCLCHECK(ncclM2nInit(&gM2nHandle, NULL));
  gActiveAlgorithm = initialAlgorithm;
  gActiveStreamPoolMode = -1;

  TEST_CUDACHECK(cudaStreamCreate(&gStream));
  TEST_NCCLCHECK(ncclMemAlloc(&gBuffer, gBufferBytes));

  if (basicApiRunAllApis(gCli) || basicApiRequestedApi(gCli) == ApiKind::Default) {
    gCopyBufferBytes = gBufferBytes;
    TEST_CUDACHECK(cudaMalloc(&gCopyBuffer, gCopyBufferBytes));
  }

  if (gWorldRank == 0) {
    std::vector<MpiParam> params = selectedParams();
    basicApiPrintRuntimeSummary("basic_api_test_mpi (gtest)", gWorldSize, gNumDevices, gCli, gResolvedCopyAlgorithm,
                                gBufferBytes, "num_tests", params.size(), true);
  }
  return 0;
}

static void shutdownMpiRuntime() {
  if (gStream != nullptr) {
    if (testUsesNonBlockingComm()) {
      TEST_NCCL_ASYNC_CHECK(gComm, gStream);
    } else {
      TEST_CUDACHECK(cudaStreamSynchronize(gStream));
    }
  }
  if (gGroupStreamA != nullptr) TEST_NCCL_ASYNC_CHECK(gGroupCommA, gGroupStreamA);
  if (gGroupStreamB != nullptr) TEST_NCCL_ASYNC_CHECK(gGroupCommB, gGroupStreamB);
  TEST_NCCLCHECK(ncclM2nFinalize(gM2nHandle));
  gM2nHandle = nullptr;
  if (gGroupBufferB != nullptr) TEST_NCCLCHECK(ncclMemFree(gGroupBufferB));
  if (gGroupStreamB != nullptr) TEST_CUDACHECK(cudaStreamDestroy(gGroupStreamB));
  if (gGroupCommB != nullptr) TEST_NCCLCHECK(testDestroyComm(gGroupCommB));
  if (gGroupBufferA != nullptr) TEST_NCCLCHECK(ncclMemFree(gGroupBufferA));
  if (gGroupStreamA != nullptr) TEST_CUDACHECK(cudaStreamDestroy(gGroupStreamA));
  if (gGroupCommA != nullptr) TEST_NCCLCHECK(testDestroyComm(gGroupCommA));
  if (gNaturalReducedBucketCommB != nullptr) TEST_NCCLCHECK(testDestroyComm(gNaturalReducedBucketCommB));
  if (gNaturalReducedBucketCommA != nullptr) TEST_NCCLCHECK(testDestroyComm(gNaturalReducedBucketCommA));
  if (gReducedBucketCommB != nullptr) TEST_NCCLCHECK(testDestroyComm(gReducedBucketCommB));
  if (gReducedBucketCommA != nullptr) TEST_NCCLCHECK(testDestroyComm(gReducedBucketCommA));
  if (gCopyBuffer != nullptr) {
    TEST_CUDACHECK(cudaFree(gCopyBuffer));
  }
  gCopyBuffer = nullptr;
  if (gBuffer != nullptr) TEST_NCCLCHECK(ncclMemFree(gBuffer));
  if (gStream != nullptr) TEST_CUDACHECK(cudaStreamDestroy(gStream));
  if (gComm != nullptr) TEST_NCCLCHECK(testDestroyComm(gComm));
}

} // namespace

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(testMpiWorld(), &gWorldRank));
  MPICHECK(MPI_Comm_size(testMpiWorld(), &gWorldSize));

  gCli = basicApiParseCli(argc, argv, "mpirun -np <N> %s [options] [--gtest_* flags]", false, true);
  gCases = buildAllTestCases();

  if (gCli.listOnly) {
    basicApiPrintCaseList(gCases, gCli, gWorldRank == 0);
    MPICHECK(MPI_Finalize());
    return 0;
  }

  ::testing::InitGoogleTest(&argc, argv);
  if (gWorldRank != 0) {
    FILE* devnull = freopen("/dev/null", "w", stdout);
    if (devnull == nullptr) fprintf(stderr, "rank %d: failed to redirect stdout\n", gWorldRank);
  }

  if (::testing::GTEST_FLAG(list_tests)) {
    int rc = RUN_ALL_TESTS();
    MPICHECK(MPI_Finalize());
    return rc;
  }

  int initRc = initMpiRuntime();
  if (initRc != 0) {
    MPICHECK(MPI_Finalize());
    return initRc;
  }

  int rc = RUN_ALL_TESTS();
  shutdownMpiRuntime();
  MPICHECK(MPI_Finalize());
  return rc;
}
