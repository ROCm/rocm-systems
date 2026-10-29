/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * basic_api_test_local — gtest-parametrized single-process driver for the
 * basic_api matrix. Each TestCase is one gtest parameter; the fixture still
 * runs one pthread per rank over ncclCommInitAll.
 *
 * Useful for dev workstations: no MPI install, no mpirun, no scheduler.
 * Cases requiring more ranks than available GPUs report as SKIP/no-op.
 ************************************************************************/

#include <gtest/gtest.h>

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_m2n.h"
#include "basic_api_test_core.h"
#include "m2n_log.h"
#include "reshard_internal.h"
#include "reshard_call_setup.h"

namespace {

/* ======================================================================
 * Bootstrap-specific shared context.
 * ====================================================================*/

struct LocalCtx {
  pthread_barrier_t barrier;
  std::vector<int> aggBuf; /* size == worldSize */
  pthread_mutex_t printMu; /* serializes optional verbose prints */
};

/* ======================================================================
 * TestEnv hooks.
 * ====================================================================*/

static void localBarrier(TestEnv* env) {
  LocalCtx* c = (LocalCtx*)env->ctx;
  pthread_barrier_wait(&c->barrier);
}

static int localAllreduceMinInt(TestEnv* env, int local) {
  LocalCtx* c = (LocalCtx*)env->ctx;
  c->aggBuf[env->rank] = local;
  pthread_barrier_wait(&c->barrier);
  int m = c->aggBuf[0];
  for (int i = 1; i < env->worldSize; ++i)
    if (c->aggBuf[i] < m) m = c->aggBuf[i];
  pthread_barrier_wait(&c->barrier); /* ensure all read aggBuf before next overwrite */
  return m;
}

static bool localIsRank0Printer(TestEnv* env) {
  return env->rank == 0;
}

/* ======================================================================
 * gtest parameter registry state.
 * ====================================================================*/

struct LocalParam {
  TestCase tc;
  ApiKind api = ApiKind::Window;
};

[[maybe_unused]] static void printTo(const LocalParam& param, std::ostream* os) {
  *os << (param.api == ApiKind::Default ? "default" : "window") << ":" << param.tc.name;
}

static BasicApiCliArgs gCli;
static const char* gResolvedCopyAlgorithm = "PACK";
static std::vector<TestCase> gCases;
static int gWorldSize = 0;
static int gNumDevices = 0;
static size_t gBufferBytes = 4096;
static std::vector<ncclComm_t> gComms;
static std::vector<ncclM2nHandle_t> gM2nHandles;

#ifdef NCCL_M2N_TESTING
/* The completion-fence failure paths decide whether an epoch is quarantined,
 * and nothing exercised them. Injection and quarantine are both process-global,
 * so every case here restores them through a guard rather than on the happy
 * path only. */
class DevCommFailureStateGuard {
public:
  DevCommFailureStateGuard()
    : quarantineWasRequired_(reshardResourcesNeedQuarantine()), logLevel_(reshardGetLogLevel()) {}

  /* Runs on every exit path, including an assertion failure part-way through a
   * case. cacheFinalize disarms any injection the case never reached, so there
   * is no bookkeeping here about how far it got. */
  ~DevCommFailureStateGuard() {
    cacheFinalize();
    if (quarantineWasRequired_) {
      reshardRequireResourceQuarantine();
    } else {
      reshardClearResourceQuarantine();
    }
    reshardSetLogLevel(logLevel_);
  }

  void failNextCompletionRecord(bool failStreamSynchronize) {
    reshardFailNextCompletionEventRecordForTest(failStreamSynchronize);
  }

  void failNextCacheEventSynchronize() { reshardFailNextCacheEventSynchronizeForTest(); }

  void enableWarningLogs() { reshardSetLogLevel(RESHARD_LOG_WARN); }

private:
  bool quarantineWasRequired_;
  ReshardLogLevel logLevel_;
};

/* A CUDA stream and event owned for the duration of one case. */
struct FenceProbeResources {
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;

  ~FenceProbeResources() {
    if (event != nullptr) {
      (void)cudaEventDestroy(event);
    }
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamDestroy(stream);
    }
  }

  cudaError_t create() {
    cudaError_t result = cudaSetDevice(0);
    if (result == cudaSuccess) {
      result = cudaStreamCreate(&stream);
    }
    if (result == cudaSuccess) {
      result = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    }
    return result;
  }
};

static bool fenceTestSelected() {
  return gCli.filter != nullptr && strcmp(gCli.filter, "devcomm_fence_failures") == 0;
}
#endif /* NCCL_M2N_TESTING */

static void* callInvalidReshard(void* resultPtr) {
  ncclResult_t* result = static_cast<ncclResult_t*>(resultPtr);
  *result = ncclReshard(nullptr, nullptr, nullptr, nullptr, nullptr);
  return nullptr;
}

TEST(ReshardMeshValidationTest, RequiresDisjointIntervalsForMultiRankCommunicators) {
  int multiRankDims[2] = {2, 1};
  ncclMesh_t src = NCCL_M2N_MESH_INITIALIZER;
  src.ndims = 2;
  src.dims = multiRankDims;
  src.startRank = 0;
  ncclMesh_t overlappingDst = src;
  overlappingDst.startRank = 1;
  ncclMesh_t disjointDst = src;
  disjointDst.startRank = 2;

  EXPECT_EQ(ncclInvalidArgument, validateReshardMeshBounds(&src, &overlappingDst, 4, 0));
  EXPECT_EQ(ncclSuccess, validateReshardMeshBounds(&src, &disjointDst, 4, 0));

  int selfDims[2] = {1, 1};
  ncclMesh_t self = NCCL_M2N_MESH_INITIALIZER;
  self.ndims = 2;
  self.dims = selfDims;
  EXPECT_EQ(ncclSuccess, validateReshardMeshBounds(&self, &self, 1, 0));
}

TEST(M2nGroupTest, DeeplyNestedGroupDefersSubmissionToOutermostEnd) {
  EXPECT_EQ(ncclSuccess, ncclM2nGroupStart());
  EXPECT_EQ(ncclSuccess, ncclReshard(nullptr, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(ncclSuccess, ncclM2nGroupStart());
  EXPECT_EQ(ncclSuccess, ncclReshard(nullptr, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(ncclSuccess, ncclM2nGroupStart());
  EXPECT_EQ(ncclSuccess, ncclReshard(nullptr, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(ncclSuccess, ncclM2nGroupEnd());
  EXPECT_EQ(ncclSuccess, ncclM2nGroupEnd());
  EXPECT_EQ(ncclInvalidArgument, ncclM2nGroupEnd());
  EXPECT_NE(std::string::npos, std::string(ncclM2nGetLastError()).find("entry 0 of 3 failed"));
}

TEST(M2nGroupTest, DeferredValidationReportsOriginalEntryIndex) {
  EXPECT_EQ(ncclSuccess, ncclM2nGroupStart());
  EXPECT_EQ(ncclSuccess, ncclReshard(nullptr, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(ncclInvalidArgument, ncclM2nGroupEnd());
  EXPECT_NE(std::string::npos, std::string(ncclM2nGetLastError()).find("entry 0 of 1 failed"));
  EXPECT_NE(std::string::npos, std::string(ncclM2nGetLastError()).find("comm must be non-null"));
}

TEST(M2nGroupTest, MixedContextsCanBeRecordedAndAborted) {
  int meshDims[1] = {1};
  size_t localShape[1] = {1};
  int placements[1] = {NCCL_RESHARD_REPLICATE};
  ncclMesh_t mesh = NCCL_M2N_MESH_INITIALIZER;
  mesh.ndims = 1;
  mesh.dims = meshDims;
  ncclDistTensor_t src = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  src.localShape = localShape;
  dst.localShape = localShape;
  src.ndims = 1;
  dst.ndims = 1;
  src.dtype = ncclUint8;
  dst.dtype = ncclUint8;
  src.placements = placements;
  dst.placements = placements;
  src.mesh = &mesh;
  dst.mesh = &mesh;

  EXPECT_EQ(ncclSuccess, ncclM2nGroupStart());
  EXPECT_EQ(ncclSuccess,
            ncclReshard(nullptr, reinterpret_cast<ncclComm_t>(1), &src, &dst, nullptr));
  EXPECT_EQ(ncclSuccess,
            ncclReshard(nullptr, reinterpret_cast<ncclComm_t>(2), &src, &dst, cudaStreamPerThread));
  EXPECT_EQ(ncclSuccess,
            ncclReshardWithWindow(reinterpret_cast<ncclM2nHandle_t>(3),
                                  reinterpret_cast<ncclComm_t>(1),
                                  reinterpret_cast<ncclWindow_t>(4), &src, &dst,
                                  cudaStreamLegacy));
  EXPECT_EQ(ncclSuccess, ncclM2nGroupAbort());
}

TEST(NcclReshardStackTest, RejectsInvalidCallOnTwoMiBThread) {
  pthread_attr_t attr;
  ASSERT_EQ(0, pthread_attr_init(&attr));
  ASSERT_EQ(0, pthread_attr_setstacksize(&attr, 2 * 1024 * 1024));

  ncclResult_t result = ncclSuccess;
  pthread_t thread;
  ASSERT_EQ(0, pthread_create(&thread, &attr, callInvalidReshard, &result));
  ASSERT_EQ(0, pthread_attr_destroy(&attr));

  ASSERT_EQ(0, pthread_join(thread, nullptr));
  EXPECT_EQ(ncclInvalidArgument, result);
}

static std::vector<LocalParam> selectedLocalParams() {
  std::vector<LocalParam> params;
  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);
  std::vector<ApiKind> apis;
  if (basicApiRunAllApis(gCli)) {
    apis = {ApiKind::Window, ApiKind::Default};
  } else {
    apis = {basicApiRequestedApi(gCli)};
  }
  for (ApiKind api : apis) {
    for (const TestCase& tc : cases) {
      if (tc.group == "graph_capture") {
        continue;
      }
      /* window_null asserts that ncclReshardWithWindow tolerates a NULL
       * window. The default API takes no window at all, so the case is
       * meaningless there and would only inflate default-API case counts. */
      if (tc.group == "window_null" && api != ApiKind::Window) {
        continue;
      }
      params.push_back(LocalParam{tc, api});
    }
  }
  return params;
}

static std::string gtestCaseName(const ::testing::TestParamInfo<LocalParam>& info) {
  std::string prefix = (info.param.api == ApiKind::Default) ? "default" : "window";
  return basicApiGtestCaseName(info.param.tc.name, info.index, prefix.c_str());
}

/* ======================================================================
 * Per-thread payload.
 * ====================================================================*/

struct ThreadArg {
  int rank;
  int worldSize;
  int device;
  ncclComm_t comm;
  ncclM2nHandle_t m2nHandle;
  LocalCtx* ctx;
  const TestCase* tc;
  ApiKind api;
  bool verbose;

  CaseStatus status;
  char skipReason[192];
  char failReason[192];
};

static void* threadMain(void* p) {
  ThreadArg* a = (ThreadArg*)p;

  TEST_CUDACHECK(cudaSetDevice(a->device));

  cudaStream_t stream;
  TEST_CUDACHECK(cudaStreamCreate(&stream));

  cudaStream_t alternateStream = nullptr;
  if (a->tc->bAsyncOrdering) {
    TEST_CUDACHECK(cudaStreamCreateWithFlags(&alternateStream, cudaStreamNonBlocking));
  }

  void* buffer = nullptr;
  TEST_NCCLCHECK(ncclMemAlloc(&buffer, gBufferBytes));

  void* copyBuffer = nullptr;
  if (a->api == ApiKind::Default) {
    TEST_CUDACHECK(cudaMalloc(&copyBuffer, gBufferBytes));
  }

  TestEnv env{};
  env.rank = a->rank;
  env.worldSize = a->worldSize;
  env.device = a->device;
  env.comm = a->comm;
  env.stream = stream;
  env.alternateStream = alternateStream;
  env.m2nHandle = a->m2nHandle;
  env.buffer = buffer;
  env.bufferBytes = gBufferBytes;
  env.copyBuffer = copyBuffer;
  env.copyBufferBytes = (copyBuffer != nullptr) ? gBufferBytes : 0;
  env.apiKind = a->api;
  env.expectPack = strcmp(gResolvedCopyAlgorithm, "PACK") == 0;
  env.verbose = a->verbose;
  env.barrier = localBarrier;
  env.allreduceMinInt = localAllreduceMinInt;
  env.isRank0Printer = localIsRank0Printer;
  env.ctx = a->ctx;

  CaseResult res = runOneCase(*a->tc, &env);
  a->status = res.status;
  if (res.skipReason != nullptr) snprintf(a->skipReason, sizeof(a->skipReason), "%s", res.skipReason);
  if (res.failReason != nullptr) snprintf(a->failReason, sizeof(a->failReason), "%s", res.failReason);

  TEST_CUDACHECK(cudaStreamSynchronize(stream));
  if (alternateStream != nullptr) {
    TEST_CUDACHECK(cudaStreamSynchronize(alternateStream));
  }

  /* Keep rank-local buffers alive until every thread has left runOneCase. */
  pthread_barrier_wait(&a->ctx->barrier);

  if (copyBuffer != nullptr) {
    TEST_CUDACHECK(cudaFree(copyBuffer));
  }
  TEST_NCCLCHECK(ncclMemFree(buffer));
  if (alternateStream != nullptr) {
    TEST_CUDACHECK(cudaStreamDestroy(alternateStream));
  }
  TEST_CUDACHECK(cudaStreamDestroy(stream));
  return nullptr;
}

struct LocalCaseResult {
  CaseStatus status;
  std::string reason;
};

static LocalCaseResult runLocalCase(const TestCase& tc, ApiKind api) {
  LocalCtx ctx{};
  ctx.aggBuf.assign(gWorldSize, 0);
  pthread_mutex_init(&ctx.printMu, nullptr);
  pthread_barrier_init(&ctx.barrier, nullptr, gWorldSize);

  std::vector<pthread_t> tids(gWorldSize);
  std::vector<ThreadArg> args(gWorldSize);
  for (int r = 0; r < gWorldSize; ++r) {
    args[r].rank = r;
    args[r].worldSize = gWorldSize;
    args[r].device = r;
    args[r].comm = gComms[r];
    args[r].m2nHandle = gM2nHandles[r];
    args[r].ctx = &ctx;
    args[r].tc = &tc;
    args[r].api = api;
    args[r].verbose = gCli.verbose;
    args[r].status = CASE_FAIL;
    args[r].skipReason[0] = '\0';
    args[r].failReason[0] = '\0';
    int rc = pthread_create(&tids[r], nullptr, threadMain, &args[r]);
    if (rc != 0) {
      fprintf(stderr, "pthread_create rank=%d failed: %d\n", r, rc);
      _Exit(1);
    }
  }
  for (int r = 0; r < gWorldSize; ++r) pthread_join(tids[r], nullptr);

  pthread_barrier_destroy(&ctx.barrier);
  pthread_mutex_destroy(&ctx.printMu);

  for (const ThreadArg& arg : args) {
    if (arg.status == CASE_FAIL) {
      return LocalCaseResult{
        CASE_FAIL,
        (arg.failReason[0] != '\0') ? arg.failReason : "rank reported failure",
      };
    }
  }
  for (const ThreadArg& arg : args) {
    if (arg.status == CASE_SKIP) {
      return LocalCaseResult{
        CASE_SKIP,
        (arg.skipReason[0] != '\0') ? arg.skipReason : "rank reported skip",
      };
    }
  }
  return LocalCaseResult{CASE_PASS, ""};
}

class BasicApiLocalTest : public ::testing::TestWithParam<LocalParam> {};

/* Exercises cross-rank agreement on the split decision: the two ranks supply
 * deliberately different local values and must receive the same reduced
 * result. The helper it drives is test-only instrumentation, so the whole
 * case compiles away in a build without it. */
#ifdef NCCL_M2N_TESTING
TEST(LocalRankProgressTest, CollectiveSectionsAdmitPeerRanks) {
  if (gCli.filter == nullptr || strcmp(gCli.filter, "local_collective_progress") != 0) {
    GTEST_SKIP() << "run with --filter local_collective_progress";
  }
  if (gWorldSize != 2) {
    GTEST_SKIP() << "requires exactly two local ranks";
  }

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  ncclResult_t results[2] = {};
  int reduced[2] = {};
  cudaError_t cudaResults[2] = {};
  std::thread threads[2];
  for (int rank = 0; rank < 2; rank++) {
    threads[rank] = std::thread([&, rank] {
      cudaResults[rank] = cudaSetDevice(rank);
      cudaStream_t stream = nullptr;
      if (cudaResults[rank] == cudaSuccess) {
        cudaResults[rank] = cudaStreamCreate(&stream);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (cudaResults[rank] == cudaSuccess) {
        M2nApiLock apiLock;
        results[rank] = reshardTestBroadcastMaxInt(gComms[rank], stream, rank, &reduced[rank]);
        cudaResults[rank] = cudaStreamDestroy(stream);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(ncclSuccess, results[0]);
  EXPECT_EQ(ncclSuccess, results[1]);
  EXPECT_EQ(1, reduced[0]);
  EXPECT_EQ(1, reduced[1]);
  EXPECT_EQ(cudaSuccess, cudaResults[0]);
  EXPECT_EQ(cudaSuccess, cudaResults[1]);
}

/* A failed record leaves work possibly already enqueued, so the stream is
 * synchronized instead. That still fences the resource, so the call reports the
 * failure without condemning the epoch. */
TEST(DevCommFenceFailureTest, RecordFailureFencedByStreamSyncDoesNotQuarantine) {
  if (!fenceTestSelected()) {
    GTEST_SKIP() << "run with --filter devcomm_fence_failures";
  }
  DevCommFailureStateGuard stateGuard;
  reshardClearResourceQuarantine();
  FenceProbeResources probe;
  ASSERT_EQ(cudaSuccess, probe.create());

  stateGuard.failNextCompletionRecord(/*failStreamSynchronize=*/false);
  EXPECT_EQ(ncclSystemError, reshardRecordCompletionEvent(probe.event, probe.stream, "DevComm"));
  EXPECT_FALSE(reshardResourcesNeedQuarantine());
}

/* Neither the event nor its stream can fence the work, so there is no way left
 * to prove the GPU is done and the epoch has to be quarantined. */
TEST(DevCommFenceFailureTest, RecordAndStreamSyncFailureQuarantinesEpoch) {
  if (!fenceTestSelected()) {
    GTEST_SKIP() << "run with --filter devcomm_fence_failures";
  }
  DevCommFailureStateGuard stateGuard;
  reshardClearResourceQuarantine();
  FenceProbeResources probe;
  ASSERT_EQ(cudaSuccess, probe.create());

  stateGuard.failNextCompletionRecord(/*failStreamSynchronize=*/true);
  EXPECT_EQ(ncclSystemError, reshardRecordCompletionEvent(probe.event, probe.stream, "DevComm"));
  EXPECT_TRUE(reshardResourcesNeedQuarantine());
}

/* The poisoned flag is what stops a DevComm whose last use could not be fenced
 * from being handed straight back out. bSerializeUses is set explicitly because
 * it normally tracks NCCL_RESHARD_USE_INTERNAL_STREAMS, and the refusal must
 * hold regardless of how the process was configured. */
TEST(DevCommFenceFailureTest, PoisonedUseStateIsRefusedOnNextAcquisition) {
  if (!fenceTestSelected()) {
    GTEST_SKIP() << "run with --filter devcomm_fence_failures";
  }
  DevCommFailureStateGuard stateGuard;
  reshardClearResourceQuarantine();
  FenceProbeResources probe;
  ASSERT_EQ(cudaSuccess, probe.create());

  std::shared_ptr<ReshardDevCommUseState> useState = std::make_shared<ReshardDevCommUseState>();
  useState->bSerializeUses = true;

  ReshardDevCommUse firstUse;
  ASSERT_EQ(ncclSuccess, reshardPrepareDevCommUse(probe.event, useState, probe.stream, &firstUse));
  stateGuard.failNextCompletionRecord(/*failStreamSynchronize=*/true);
  EXPECT_EQ(ncclSystemError, reshardRecordDevCommUse(&firstUse, probe.stream));
  EXPECT_TRUE(useState->bPoisoned);

  /* The reservation must have been released even though recording failed, or
   * this second prepare would deadlock rather than return. */
  ReshardDevCommUse secondUse;
  EXPECT_EQ(ncclSystemError, reshardPrepareDevCommUse(probe.event, useState, probe.stream, &secondUse));
}

/* Retirement is where the fence actually protects something: if the completion
 * event cannot be synchronized, the DevComm must be retained rather than
 * destroyed under a running kernel. Needs a genuinely cached DevComm, so this
 * case creates one collectively across both local ranks. */
TEST(DevCommFenceFailureTest, RetirementRetainsDevCommWhenCompletionCannotBeProven) {
  if (!fenceTestSelected()) {
    GTEST_SKIP() << "run with --filter devcomm_fence_failures";
  }
  if (gWorldSize != 2) {
    GTEST_SKIP() << "requires exactly two local ranks";
  }

  DevCommFailureStateGuard stateGuard;
  stateGuard.enableWarningLogs();
  reshardClearResourceQuarantine();

  /* ncclDevCommCreate is collective, so both ranks must reach it. */
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  ncclResult_t results[2] = {};
  cudaError_t cudaResults[2] = {};
  std::thread threads[2];
  for (int rank = 0; rank < 2; rank++) {
    threads[rank] = std::thread([&, rank] {
      cudaResults[rank] = cudaSetDevice(rank);
      cudaStream_t stream = nullptr;
      if (cudaResults[rank] == cudaSuccess) {
        cudaResults[rank] = cudaStreamCreate(&stream);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (cudaResults[rank] == cudaSuccess) {
        M2nApiLock apiLock;
        ncclDevComm devComm{};
        ReshardDevCommUse use;
        results[rank] = reshardGetOrCreateDevComm(gComms[rank], /*numCtas=*/1, /*ginSignalCount=*/1,
                                                  /*ginCounterCount=*/0, RESHARD_DEVCOMM_BARRIER_WORLD,
                                                  reshardGetGinContextCount(), stream, &devComm, &use);
        if (results[rank] == ncclSuccess) {
          results[rank] = reshardRecordDevCommUse(&use, stream);
        }
        cudaResults[rank] = cudaStreamDestroy(stream);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  ASSERT_EQ(cudaSuccess, cudaResults[0]);
  ASSERT_EQ(cudaSuccess, cudaResults[1]);
  ASSERT_EQ(ncclSuccess, results[0]);
  ASSERT_EQ(ncclSuccess, results[1]);

  /* Finalize is the deterministic retirement path; filling all 64 cache
   * identities would be testing eviction policy instead. */
  stateGuard.failNextCacheEventSynchronize();
  testing::internal::CaptureStdout();
  cacheFinalize();
  const std::string warnings = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(reshardResourcesNeedQuarantine());
  EXPECT_NE(std::string::npos, warnings.find("Retaining DevComm after its completion event"));
}
#endif  /* NCCL_M2N_TESTING */

TEST_P(BasicApiLocalTest, Reshard) {
  const LocalParam& param = GetParam();
  SCOPED_TRACE(param.tc.name);

  LocalCaseResult res = runLocalCase(param.tc, param.api);
  if (res.status == CASE_SKIP) {
    GTEST_SKIP() << res.reason;
    return;
  }

  EXPECT_EQ(CASE_PASS, res.status) << res.reason;
}

INSTANTIATE_TEST_SUITE_P(Matrix, BasicApiLocalTest, ::testing::ValuesIn(selectedLocalParams()), gtestCaseName);

static int initLocalRuntime() {
  TEST_CUDACHECK(cudaGetDeviceCount(&gNumDevices));
  if (gNumDevices <= 0) {
    fprintf(stderr, "No CUDA devices visible.\n");
    return 2;
  }

  gWorldSize = gCli.requestedRanks > 0 ? gCli.requestedRanks : gNumDevices;
  if (gWorldSize > gNumDevices) {
    fprintf(stderr,
            "Requested %d ranks but only %d CUDA device(s) visible -- "
            "clamping.\n",
            gWorldSize, gNumDevices);
    gWorldSize = gNumDevices;
  }

  gResolvedCopyAlgorithm = basicApiConfigureReshardEnv(gCli, basicApiRequestedAlgorithmEnv(gCli, true));

  std::vector<int> devlist(gWorldSize);
  gComms.assign(gWorldSize, nullptr);
  gM2nHandles.assign(gWorldSize, nullptr);
  for (int i = 0; i < gWorldSize; ++i) devlist[i] = i;

  TEST_NCCLCHECK(ncclCommInitAll(gComms.data(), gWorldSize, devlist.data()));
  testWaitComms(gComms);
  for (int i = 0; i < gWorldSize; ++i) {
    TEST_NCCLCHECK(ncclM2nInit(&gM2nHandles[i], NULL));
  }

  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);
  gBufferBytes = computeMaxBufferBytes(cases, gWorldSize);

  std::vector<LocalParam> localParams = selectedLocalParams();
  basicApiPrintRuntimeSummary("basic_api_test_local (gtest, no MPI)", gWorldSize, gNumDevices, gCli,
                              gResolvedCopyAlgorithm, gBufferBytes, "num_tests", localParams.size(), true);
  return 0;
}

static void shutdownLocalRuntime() {
  for (ncclM2nHandle_t m2nHandle : gM2nHandles) {
    if (m2nHandle != nullptr) {
      TEST_NCCLCHECK(ncclM2nFinalize(m2nHandle));
    }
  }
  gM2nHandles.clear();
  testWaitComms(gComms);
  for (ncclComm_t comm : gComms) {
    TEST_NCCLCHECK(ncclCommDestroy(comm));
  }
  gComms.clear();
}

} // namespace

int main(int argc, char** argv) {
  gCli = basicApiParseCli(argc, argv, "%s [options] [--gtest_* flags]", true, false);
  gCases = buildAllTestCases();

  if (gCli.listOnly) {
    basicApiPrintCaseList(gCases, gCli, true);
    return 0;
  }

  ::testing::InitGoogleTest(&argc, argv);
  if (::testing::GTEST_FLAG(list_tests)) return RUN_ALL_TESTS();

  int initRc = initLocalRuntime();
  if (initRc != 0) return initRc;

  int rc = RUN_ALL_TESTS();
  shutdownLocalRuntime();
  return rc;
}
