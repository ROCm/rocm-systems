/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for `src/ras/diagnostics.cc`.

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#define rasClientsHead DiagnosticsTestRasClientsHead
#define rasCollFree DiagnosticsTestRasCollFree
#define rasCollReqInit DiagnosticsTestRasCollReqInit
#define rasNetSendCollReq DiagnosticsTestRasNetSendCollReq
#define rasDiagnosticsGpuModelCollectLocal DiagnosticsTestGpuModelCollectLocal
#define rasDiagnosticsGpuModelSummarize DiagnosticsTestGpuModelSummarize
#define rasDiagnosticsCudaDriverVersionCollectLocal DiagnosticsTestCudaDriverVersionCollectLocal
#define rasDiagnosticsCudaDriverVersionSummarize DiagnosticsTestCudaDriverVersionSummarize
#define rasDiagnosticsEccCollectLocal DiagnosticsTestEccCollectLocal
#define rasDiagnosticsEccSummarize DiagnosticsTestEccSummarize
#define rasDiagnosticsNvLinkCollectLocal DiagnosticsTestNvLinkCollectLocal
#define rasDiagnosticsNvLinkSummarize DiagnosticsTestNvLinkSummarize
#define rasDiagnosticsNcclEnvCollectLocal DiagnosticsTestNcclEnvCollectLocal
#define rasDiagnosticsNcclEnvSummarize DiagnosticsTestNcclEnvSummarize
#define rasDiagnosticsRdmaTopoCollectLocal DiagnosticsTestRdmaTopoCollectLocal
#define rasDiagnosticsRdmaTopoSummarize DiagnosticsTestRdmaTopoSummarize
#define rasDiagnosticsIommuCollectLocal DiagnosticsTestIommuCollectLocal
#define rasDiagnosticsIommuSummarize DiagnosticsTestIommuSummarize
#define rasDiagnosticsAtsCollectLocal DiagnosticsTestAtsCollectLocal
#define rasDiagnosticsAtsSummarize DiagnosticsTestAtsSummarize
#define rasDiagnosticsXidCollectLocal DiagnosticsTestXidCollectLocal
#define rasDiagnosticsXidSummarize DiagnosticsTestXidSummarize
#define rasDiagnosticsNvidiaDriverVersionCollectLocal DiagnosticsTestNvidiaDriverVersionCollectLocal
#define rasDiagnosticsNvidiaDriverVersionSummarize DiagnosticsTestNvidiaDriverVersionSummarize
#define rasDiagnosticsPathsCollectLocal DiagnosticsTestPathsCollectLocal
#define rasDiagnosticsPathsSummarize DiagnosticsTestPathsSummarize

#include "comm.h"
#include "ras/diagnostics.h"
#include "ras/diagnostics_checks.h"
#include "ras/ras_internal.h"

namespace {

int g_allocationCalls;
int g_failAllocationCall;
int g_reallocationCalls;
int g_failReallocationCall;
int g_freeCalls;

template <typename T>
ncclResult_t DiagnosticsTestCalloc(T** ptr, size_t count) {
  ++g_allocationCalls;
  if (g_allocationCalls == g_failAllocationCall) return ncclSystemError;
  return ncclCallocDebug(ptr, count, __FILE__, __LINE__, __func__, false);
}

template <typename T>
ncclResult_t DiagnosticsTestRealloc(T** ptr, size_t oldCount, size_t newCount) {
  ++g_reallocationCalls;
  if (g_reallocationCalls == g_failReallocationCall) return ncclSystemError;
  return ncclReallocDebug(ptr, oldCount, newCount, __FILE__, __LINE__, __func__, false);
}

void DiagnosticsTestFree(void* ptr) {
  ++g_freeCalls;
  std::free(ptr);
}

}  // namespace

#define clockNano DiagnosticsTestClockNano
#undef ncclCalloc
#define ncclCalloc(...) DiagnosticsTestCalloc(__VA_ARGS__)
#undef ncclRealloc
#define ncclRealloc(...) DiagnosticsTestRealloc(__VA_ARGS__)
#define free DiagnosticsTestFree

uint64_t DiagnosticsTestClockNano();

#include RAS_DIAGNOSTICS_CC_PATH

#undef clockNano
#undef ncclCalloc
#undef ncclRealloc
#undef free

namespace {
int64_t g_clockNano = 0;
}  // namespace

uint64_t DiagnosticsTestClockNano() { return static_cast<uint64_t>(g_clockNano); }

// Test-local collaborator fakes.

// Keep synthetic clients paired with this TU's rasCollFree and fixture cleanup,
// independently of ras-test.cc's client ownership and collaborator fakes.
struct rasClient* rasClientsHead = nullptr;

namespace {
int g_collFreeCalls = 0;
struct rasCollective* g_lastCollFree = nullptr;

uint64_t g_collReqLastRootId = 0;

int g_netSendCollReqCalls = 0;
ncclResult_t g_netSendCollReqResult = ncclSuccess;
bool g_netSendCollReqAllDone = true;
struct rasCollective* g_netSendCollReqCollToAssign = nullptr;
struct rasCollRequest g_lastSentReq{};
struct rasConnection* g_lastSendFromConn = nullptr;

struct CheckHook {
  ncclResult_t collectLocalResult = ncclSuccess;
  struct rasDiagnosticsLocalData collectLocalData{};
  int collectLocalCalls = 0;
  struct rasDiagnosticsContext lastCollectLocalCtx{};

  ncclResult_t summarizeResult = ncclSuccess;
  int summarizeCalls = 0;
  // Copied, not a raw pointer: rasDiagnosticsSummarizePeerPayloads frees its
  // combined[id].records buffer immediately after this call returns.
  std::vector<char> lastSummarizeData;
  int lastSummarizeNData = -1;
  struct rasDiagnosticsContext lastSummarizeCtx{};
  struct rasDiagnosticsReporter lastSummarizeReporter{};
};
CheckHook g_checkHooks[RAS_DIAG_CHECK_COUNT];

void ResetCheckHooks() {
  for (auto& h : g_checkHooks) h = CheckHook{};
}
}  // namespace

void rasCollFree(struct rasCollective* coll) {
  ++g_collFreeCalls;
  g_lastCollFree = coll;
  if (coll == nullptr) return;
  free(coll->fwdConns);
  free(coll->peers);
  free(coll->data);
  free(coll);
}

void rasCollReqInit(struct rasCollRequest* req) { req->rootId = ++g_collReqLastRootId; }

ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, bool* pAllDone, struct rasCollective** pColl,
                               struct rasConnection* fromConn) {
  ++g_netSendCollReqCalls;
  g_lastSentReq = *req;
  g_lastSendFromConn = fromConn;
  if (g_netSendCollReqResult != ncclSuccess) return g_netSendCollReqResult;
  if (pAllDone) *pAllDone = g_netSendCollReqAllDone;
  if (pColl) *pColl = g_netSendCollReqCollToAssign;
  return ncclSuccess;
}

// Collectors publish owned records only on success; failure hooks must not supply records.
#define DEFINE_CHECK_FAKE(idx, collectFn, summarizeFn)                                                            \
  ncclResult_t collectFn(const struct rasDiagnosticsContext* ctx, struct rasDiagnosticsLocalData* data) {         \
    ++g_checkHooks[idx].collectLocalCalls;                                                                        \
    g_checkHooks[idx].lastCollectLocalCtx = *ctx;                                                                 \
    *data = {};                                                                                                 \
    if (g_checkHooks[idx].collectLocalResult != ncclSuccess) {                                                    \
      EXPECT_EQ(nullptr, g_checkHooks[idx].collectLocalData.records);                                             \
      return g_checkHooks[idx].collectLocalResult;                                                               \
    }                                                                                                           \
    *data = g_checkHooks[idx].collectLocalData;                                                                   \
    g_checkHooks[idx].collectLocalData.records = nullptr;                                                         \
    return g_checkHooks[idx].collectLocalResult;                                                                  \
  }                                                                                                                \
  ncclResult_t summarizeFn(const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, \
                           const char* data, int nData) {                                                          \
    ++g_checkHooks[idx].summarizeCalls;                                                                            \
    g_checkHooks[idx].lastSummarizeCtx = *ctx;                                                                     \
    if (reporter != nullptr) g_checkHooks[idx].lastSummarizeReporter = *reporter;                                  \
    if (data != nullptr && nData > 0) g_checkHooks[idx].lastSummarizeData.assign(data, data + nData);              \
    else g_checkHooks[idx].lastSummarizeData.clear();                                                              \
    g_checkHooks[idx].lastSummarizeNData = nData;                                                                  \
    return g_checkHooks[idx].summarizeResult;                                                                      \
  }

DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_GPU_MODEL, rasDiagnosticsGpuModelCollectLocal, rasDiagnosticsGpuModelSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_CUDA_DRIVER_VERSION, rasDiagnosticsCudaDriverVersionCollectLocal,
                  rasDiagnosticsCudaDriverVersionSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_ECC, rasDiagnosticsEccCollectLocal, rasDiagnosticsEccSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_NVLINK, rasDiagnosticsNvLinkCollectLocal, rasDiagnosticsNvLinkSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_NCCL_ENV, rasDiagnosticsNcclEnvCollectLocal, rasDiagnosticsNcclEnvSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_RDMA_TOPO, rasDiagnosticsRdmaTopoCollectLocal, rasDiagnosticsRdmaTopoSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_IOMMU_MODE, rasDiagnosticsIommuCollectLocal, rasDiagnosticsIommuSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_ATS, rasDiagnosticsAtsCollectLocal, rasDiagnosticsAtsSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_XID_SXID, rasDiagnosticsXidCollectLocal, rasDiagnosticsXidSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_NVIDIA_DRIVER_VERSION, rasDiagnosticsNvidiaDriverVersionCollectLocal,
                  rasDiagnosticsNvidiaDriverVersionSummarize)
DEFINE_CHECK_FAKE(RAS_DIAG_CHECK_PATHS, rasDiagnosticsPathsCollectLocal, rasDiagnosticsPathsSummarize)

#undef DEFINE_CHECK_FAKE

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

namespace {

struct rasClient* MakeClient() {
  auto* client = static_cast<struct rasClient*>(calloc(1, sizeof(struct rasClient)));
  client->status = RAS_CLIENT_CONNECTED;
  return client;
}

struct rasClient* MakeDiagClient(const struct rasDiagnosticsContext& ctx,
                                 const struct rasDiagnosticsReporter* reporter = nullptr,
                                 bool withCollective = false) {
  auto* client = MakeClient();
  const ncclResult_t result = rasDiagnosticsClientInit(client, &ctx, reporter);
  EXPECT_EQ(ncclSuccess, result);
  if (result != ncclSuccess) {
    free(client);
    return nullptr;
  }
  if (withCollective) {
    client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
    EXPECT_NE(nullptr, client->coll);
    if (client->coll == nullptr) {
      rasDiagnosticsClientCleanup(client);
      free(client);
      return nullptr;
    }
  }
  return client;
}

void LinkClient(struct rasClient* client) {
  client->next = rasClientsHead;
  if (rasClientsHead) rasClientsHead->prev = client;
  rasClientsHead = client;
}

void FreeClientList() {
  struct rasClient* client = rasClientsHead;
  while (client) {
    struct rasClient* next = client->next;
    free(client->diagnostics);
    if (client->coll) rasCollFree(client->coll);
    free(client);
    client = next;
  }
  rasClientsHead = nullptr;
}

int g_emitCalls;
std::vector<std::string> g_emittedLines;
ncclResult_t g_finishResult;
int g_finishCalls;
ncclResult_t g_lastFinishResult;
void* g_lastEmitTarget;
void* g_lastFinishTarget;

ncclResult_t RecordingEmit(void* target, const char* line) {
  ++g_emitCalls;
  g_lastEmitTarget = target;
  g_emittedLines.emplace_back(line);
  return ncclSuccess;
}

ncclResult_t RecordingFinish(void* target, ncclResult_t result) {
  ++g_finishCalls;
  g_lastFinishTarget = target;
  g_lastFinishResult = result;
  return g_finishResult;
}

void ResetRecordingReporter() {
  g_emitCalls = 0;
  g_emittedLines.clear();
  g_finishResult = ncclSuccess;
  g_finishCalls = 0;
  g_lastFinishResult = ncclSuccess;
  g_lastEmitTarget = nullptr;
  g_lastFinishTarget = nullptr;
}

rasDiagnosticsReporter MakeRecordingReporter() {
  rasDiagnosticsReporter reporter{};
  reporter.emit = RecordingEmit;
  reporter.finish = RecordingFinish;
  reporter.target = nullptr;
  return reporter;
}

struct FakeComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::vector<ncclPeerInfo> peerInfos;

  FakeComm(uint64_t commHash, int nRanks) {
    comm->commHash = commHash;
    comm->nRanks = nRanks;
    comm->peerInfoValid = true;
    peerInfos.assign(nRanks > 0 ? nRanks : 1, ncclPeerInfo{});
    peerInfos[0].hostHash = commHash + 1;
    peerInfos[0].pidHash = commHash + 2;
    comm->peerInfo = peerInfos.data();
  }
  ncclComm* get() { return comm.get(); }
};

// Builds one peer's self-delimiting payload: [rasDiagnosticsPeerPayloadHeader]
// followed by one [rasDiagnosticsCheckPayloadHeader][records] block per contribution.
// Unlike rasDiagnosticsAppendCheckPayload, this helper always writes a header for a
// check that produced zero records so malformed/zero-record receive paths can be tested.
struct CheckContribution {
  int checkId;
  int recordStride;
  std::vector<char> records;
};

std::vector<char> BuildPeerPayload(const std::vector<CheckContribution>& checks) {
  std::vector<char> buf(sizeof(struct rasDiagnosticsPeerPayloadHeader), 0);
  int nChecks = 0;
  for (auto& c : checks) {
    struct rasDiagnosticsCheckPayloadHeader h{};
    static_assert(sizeof(h.checkId) == sizeof(c.checkId));
    h.recordStride = c.recordStride;
    h.nRecords = c.recordStride > 0 ? static_cast<int>(c.records.size() / c.recordStride) : 0;
    h.payloadBytes = static_cast<int>(c.records.size());
    size_t off = buf.size();
    buf.resize(off + sizeof(h));
    memcpy(buf.data() + off, &h, sizeof(h));
    memcpy(buf.data() + off + offsetof(rasDiagnosticsCheckPayloadHeader, checkId), &c.checkId, sizeof(c.checkId));
    buf.insert(buf.end(), c.records.begin(), c.records.end());
    nChecks++;
  }
  auto* hdr = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(buf.data());
  hdr->nChecks = nChecks;
  hdr->payloadBytes = static_cast<int>(buf.size());
  return buf;
}

std::vector<char> BuildGatheredData(const std::vector<std::vector<CheckContribution>>& peers) {
  std::vector<char> all;
  for (auto& p : peers) {
    auto buf = BuildPeerPayload(p);
    all.insert(all.end(), buf.begin(), buf.end());
  }
  return all;
}

struct rasDiagnosticsPeerPayloadHeader* PeerPayloadHeader(char* payload) {
  return reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(payload);
}

struct rasDiagnosticsPeerPayloadHeader* PeerPayloadHeader(std::vector<char>& payload) {
  return PeerPayloadHeader(payload.data());
}

struct rasDiagnosticsCheckPayloadHeader* FirstCheckHeader(char* payload) {
  return reinterpret_cast<struct rasDiagnosticsCheckPayloadHeader*>(PeerPayloadHeader(payload) + 1);
}

struct rasDiagnosticsCheckPayloadHeader* FirstCheckHeader(std::vector<char>& payload) {
  return FirstCheckHeader(payload.data());
}

void ExpectSummaryRejected(const struct rasDiagnosticsContext& ctx, const std::vector<char>& payload) {
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, payload.data(), static_cast<int>(payload.size())));
}

void ResetWholeFileSeams() {
  g_allocationCalls = 0;
  g_failAllocationCall = 0;
  g_reallocationCalls = 0;
  g_failReallocationCall = 0;
  g_freeCalls = 0;
  g_collFreeCalls = 0;
  g_lastCollFree = nullptr;
  g_collReqLastRootId = 0;
  g_netSendCollReqCalls = 0;
  g_netSendCollReqResult = ncclSuccess;
  g_netSendCollReqAllDone = true;
  g_netSendCollReqCollToAssign = nullptr;
  g_lastSendFromConn = nullptr;
  memset(&g_lastSentReq, 0, sizeof(g_lastSentReq));
  ResetCheckHooks();
  ResetRecordingReporter();
  FreeClientList();
  g_clockNano = 0;
}

class RasDiagnosticsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetWholeFileSeams(); }
  void TearDown() override { ResetWholeFileSeams(); }
};

}  // namespace

// ===========================================================================
// rasDiagnosticsFormatLine
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, FormatLine_NullArgsReturnInternalError) {
  char buf[64];
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(nullptr, sizeof(buf), "line"));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(buf, 0, "line"));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(buf, sizeof(buf), nullptr));
}

TEST_F(RasDiagnosticsMicrotest, FormatLine_TruncationReturnsInternalError) {
  char buf[8];
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(buf, sizeof(buf), "a much too long diagnostics line"));
}

TEST_F(RasDiagnosticsMicrotest, FormatLine_SuccessIncludesHostAndPid) {
  char buf[256];
  char host[64] = {};
  char expected[256];
  ASSERT_EQ(0, gethostname(host, sizeof(host) - 1));
  ASSERT_GT(snprintf(expected, sizeof(expected), "%s:%d NCCL DIAG hello", host, static_cast<int>(getpid())), 0);
  ASSERT_EQ(ncclSuccess, rasDiagnosticsFormatLine(buf, sizeof(buf), "hello"));
  EXPECT_STREQ(expected, buf);
}

TEST_F(RasDiagnosticsMicrotest, FormatLine_ExactFitSucceedsAtTheBoundary) {
  char big[256];
  ASSERT_EQ(ncclSuccess, rasDiagnosticsFormatLine(big, sizeof(big), "hello"));
  size_t exactLen = strlen(big) + 1;  // Room for the line plus the terminating NUL.
  auto exact = std::make_unique<char[]>(exactLen);
  EXPECT_EQ(ncclSuccess, rasDiagnosticsFormatLine(exact.get(), exactLen, "hello"));
  EXPECT_STREQ(big, exact.get());
  // One byte short of that exact fit must fail (snprintf's return equals outSize).
  auto short_ = std::make_unique<char[]>(exactLen - 1);
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(short_.get(), exactLen - 1, "hello"));
}

// ===========================================================================
// rasDiagnosticsContextInit
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, ContextInit_NullContextReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsContextInit(nullptr, nullptr));
}

TEST_F(RasDiagnosticsMicrotest, ContextInit_NullCommZeroesUnscopedContext) {
  struct rasDiagnosticsContext ctx;
  memset(&ctx, 0xAB, sizeof(ctx));
  ASSERT_EQ(ncclSuccess, rasDiagnosticsContextInit(&ctx, nullptr));
  EXPECT_FALSE(ctx.hasCommFilter);
  EXPECT_EQ(0, ctx.commNRanks);
}

TEST_F(RasDiagnosticsMicrotest, ContextInit_InvalidPeerInfoReturnsInternalError) {
  FakeComm fc(0x1000, 4);
  fc.get()->peerInfoValid = false;
  struct rasDiagnosticsContext ctx{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsContextInit(&ctx, fc.get()));
}

TEST_F(RasDiagnosticsMicrotest, ContextInit_ValidCommCopiesFilterFields) {
  FakeComm fc(0x2000, 4);
  fc.get()->rank = 2;
  fc.peerInfos[2].hostHash = 0xDEAD;
  fc.peerInfos[2].pidHash = 0xBEEF;
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsContextInit(&ctx, fc.get()));
  EXPECT_TRUE(ctx.hasCommFilter);
  EXPECT_EQ(0x2000u, ctx.commFilter.commHash);
  EXPECT_EQ(0x2001u, ctx.commFilter.hostHash);
  EXPECT_EQ(0x2002u, ctx.commFilter.pidHash);
  EXPECT_EQ(4, ctx.commNRanks);
}

// ===========================================================================
// Static payload helpers
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, GetCheck_NullOutputReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsGetCheck(RAS_DIAG_CHECK_GPU_MODEL, nullptr));
}

TEST_F(RasDiagnosticsMicrotest, AppendData_RejectsInvalidSizesAndPointers) {
  char* data = nullptr;
  int nData = 0;
  const char byte = 1;

  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(nullptr, &nData, &byte, 1));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, nullptr, &byte, 1));
  nData = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, &byte, 1));
  nData = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, &byte, -1));
  nData = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, &byte, 1));
  nData = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, nullptr, 1));
}

TEST_F(RasDiagnosticsMicrotest, AppendData_HandlesNoopSuccessAndAllocationFailure) {
  char* data = nullptr;
  int nData = 0;
  const char extra[] = "abc";

  ASSERT_EQ(ncclSuccess, rasDiagnosticsAppendData(&data, &nData, nullptr, 0));
  EXPECT_EQ(nullptr, data);
  EXPECT_EQ(0, nData);

  ASSERT_EQ(ncclSuccess, rasDiagnosticsAppendData(&data, &nData, extra, 3));
  ASSERT_NE(nullptr, data);
  EXPECT_EQ(3, nData);
  EXPECT_EQ(0, memcmp(data, extra, 3));

  g_failReallocationCall = g_reallocationCalls + 1;
  EXPECT_EQ(ncclSystemError, rasDiagnosticsAppendData(&data, &nData, extra, 3));
  EXPECT_EQ(3, nData);
  EXPECT_EQ(0, memcmp(data, extra, 3));
  free(data);
}

TEST_F(RasDiagnosticsMicrotest, ValidateLocalData_RejectsInvalidMetadata) {
  struct rasDiagnosticsLocalData data{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, nullptr));

  data.recordsBytes = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data = {};
  data.recordStride = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data = {};
  data.nRecords = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));

  char record = 0;
  data = {};
  data.records = &record;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data = {};
  data.recordsBytes = 1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));

  data = {};
  data.nRecords = 1;
  data.recordStride = 1;
  data.recordsBytes = 1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data.records = &record;
  data.recordStride = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data.recordStride = 2;
  data.nRecords = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data.nRecords = 1;
  data.recordsBytes = 1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
}

TEST_F(RasDiagnosticsMicrotest, CollectLocalPeerPayload_RejectsInvalidOutputsAndContext) {
  struct rasDiagnosticsContext ctx{};
  char* data = reinterpret_cast<char*>(1);
  int nData = 7;

  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalPeerPayload(&ctx, nullptr, &nData));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalPeerPayload(&ctx, &data, nullptr));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalPeerPayload(nullptr, &data, &nData));
  EXPECT_EQ(nullptr, data);
  EXPECT_EQ(0, nData);
}

TEST_F(RasDiagnosticsMicrotest, AccountCheckRecords_RejectsStrideAndSizeOverflow) {
  struct rasDiagnosticsLocalData combined{};
  struct rasDiagnosticsCheckPayloadHeader header{};
  header.checkId = RAS_DIAG_CHECK_GPU_MODEL;
  header.recordStride = 4;
  header.nRecords = 1;
  header.payloadBytes = 4;

  combined.nRecords = 1;
  combined.recordStride = 8;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAccountCheckRecords(&combined, &header));

  combined = {};
  combined.recordStride = 4;
  combined.nRecords = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAccountCheckRecords(&combined, &header));

  combined = {};
  combined.recordStride = 4;
  combined.recordsBytes = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAccountCheckRecords(&combined, &header));
}

// ===========================================================================
// rasDiagnosticsClientInit / rasDiagnosticsClientCleanup
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, ClientInit_NullClientOrCtxReturnsInternalError) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(nullptr, &ctx, nullptr));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(client, nullptr, nullptr));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_ReporterWithNullEmitReturnsInternalError) {
  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsReporter reporter{};
  auto* client = MakeClient();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(client, &ctx, &reporter));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_ExistingDiagnosticsStateReturnsInternalError) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  client->diagnostics = reinterpret_cast<struct rasDiagnosticsClientState*>(0x1);
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(client, &ctx, nullptr));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_NullReporterUsesDefault) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  ASSERT_NE(nullptr, client->diagnostics);
  EXPECT_EQ(rasDiagnosticsDefaultEmit, client->diagnostics->reporter.emit);
  rasDiagnosticsClientCleanup(client);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_AllocationFailureLeavesStateNull) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  g_failAllocationCall = 1;
  EXPECT_EQ(ncclSystemError, rasDiagnosticsClientInit(client, &ctx, nullptr));
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_CustomReporterAndContextAreStored) {
  int target;
  struct rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = true;
  ctx.commFilter.commHash = 0x42;
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  auto* client = MakeClient();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  EXPECT_EQ(RecordingEmit, client->diagnostics->reporter.emit);
  EXPECT_EQ(RecordingFinish, client->diagnostics->reporter.finish);
  EXPECT_EQ(reporter.target, client->diagnostics->reporter.target);
  EXPECT_EQ(0x42u, client->diagnostics->ctx.commFilter.commHash);
  rasDiagnosticsClientCleanup(client);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientCleanup_NullClientIsNoOp) { rasDiagnosticsClientCleanup(nullptr); }

TEST_F(RasDiagnosticsMicrotest, ClientCleanup_FreesAndNullsDiagnostics) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  rasDiagnosticsClientCleanup(client);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

// ===========================================================================
// rasDiagnosticsInProgress / rasDiagnosticsCancelTarget
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, InProgress_EmptyClientListIsFalse) { EXPECT_FALSE(rasDiagnosticsInProgress()); }

TEST_F(RasDiagnosticsMicrotest, InProgress_ClientWithoutDiagnosticsIsFalse) {
  LinkClient(MakeClient());
  EXPECT_FALSE(rasDiagnosticsInProgress());
}

TEST_F(RasDiagnosticsMicrotest, InProgress_ClientWithDiagnosticsIsTrue) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx);
  ASSERT_NE(nullptr, client);
  LinkClient(client);
  EXPECT_TRUE(rasDiagnosticsInProgress());
}

TEST_F(RasDiagnosticsMicrotest, CancelTarget_NullTargetIsNoOp) {
  int target;
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter);
  ASSERT_NE(nullptr, client);
  LinkClient(client);
  rasDiagnosticsCancelTarget(nullptr);
  EXPECT_EQ(RecordingEmit, client->diagnostics->reporter.emit);
}

TEST_F(RasDiagnosticsMicrotest, CancelTarget_MatchingTargetSwapsToNoopReporter) {
  int target;
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter);
  ASSERT_NE(nullptr, client);
  LinkClient(client);
  auto* clientWithoutDiagnostics = MakeClient();
  LinkClient(clientWithoutDiagnostics);
  rasDiagnosticsCancelTarget(&target);
  EXPECT_EQ(nullptr, clientWithoutDiagnostics->diagnostics);
  EXPECT_EQ(rasDiagnosticsNoopEmit, client->diagnostics->reporter.emit);
  EXPECT_EQ(nullptr, client->diagnostics->reporter.finish);
  EXPECT_EQ(nullptr, client->diagnostics->reporter.target);
  // The noop reporter must not crash or touch state when invoked.
  EXPECT_EQ(ncclSuccess, client->diagnostics->reporter.emit(client->diagnostics->reporter.target, "ignored"));
  EXPECT_EQ(0, g_emitCalls);
}

TEST_F(RasDiagnosticsMicrotest, CancelTarget_NonMatchingTargetLeavesReporterAlone) {
  int target, other;
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter);
  ASSERT_NE(nullptr, client);
  LinkClient(client);
  rasDiagnosticsCancelTarget(&other);
  EXPECT_EQ(RecordingEmit, client->diagnostics->reporter.emit);
}

// ===========================================================================
// rasCollDiagInit / rasDiagnosticsCollectLocalPeerPayload (static, indirect)
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_UnscopedRequestPropagatesNoFilter) {
  struct rasCollRequest req{};
  req.diag.hasCommFilter = false;
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  ASSERT_EQ(ncclSuccess, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
  EXPECT_EQ(rasCollDataLength(RAS_COLL_DIAG), reqLen);
  for (const auto& hook : g_checkHooks) EXPECT_FALSE(hook.lastCollectLocalCtx.hasCommFilter);
  ASSERT_NE(nullptr, data);
  auto* peerHeader = PeerPayloadHeader(data);
  EXPECT_EQ(0, peerHeader->nChecks);  // No hooks produced records by default.
  EXPECT_EQ((int)sizeof(*peerHeader), peerHeader->payloadBytes);
  free(data);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_ScopedRequestPassesFilterToChecks) {
  struct rasCollRequest req{};
  req.diag.hasCommFilter = true;
  req.diag.commFilter.commHash = 0x99;
  req.diag.commFilter.hostHash = 0x88;
  req.diag.commFilter.pidHash = 0x77;
  g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.nRecords = 1;
  g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.recordStride = 4;
  g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.recordsBytes = 4;
  g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.records = static_cast<char*>(malloc(4));
  memcpy(g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.records, "test", 4);
  g_checkHooks[RAS_DIAG_CHECK_CUDA_DRIVER_VERSION].collectLocalData.nRecords = 2;
  g_checkHooks[RAS_DIAG_CHECK_CUDA_DRIVER_VERSION].collectLocalData.recordStride = 2;
  g_checkHooks[RAS_DIAG_CHECK_CUDA_DRIVER_VERSION].collectLocalData.recordsBytes = 4;
  g_checkHooks[RAS_DIAG_CHECK_CUDA_DRIVER_VERSION].collectLocalData.records = static_cast<char*>(malloc(4));
  memcpy(g_checkHooks[RAS_DIAG_CHECK_CUDA_DRIVER_VERSION].collectLocalData.records, "next", 4);
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  ASSERT_EQ(ncclSuccess, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
  EXPECT_EQ(nullptr, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.records);
  EXPECT_EQ(nullptr, g_checkHooks[RAS_DIAG_CHECK_CUDA_DRIVER_VERSION].collectLocalData.records);
  for (const auto& hook : g_checkHooks) {
    EXPECT_EQ(1, hook.collectLocalCalls);
    EXPECT_TRUE(hook.lastCollectLocalCtx.hasCommFilter);
    EXPECT_EQ(0x99u, hook.lastCollectLocalCtx.commFilter.commHash);
    EXPECT_EQ(0x88u, hook.lastCollectLocalCtx.commFilter.hostHash);
    EXPECT_EQ(0x77u, hook.lastCollectLocalCtx.commFilter.pidHash);
  }
  ASSERT_NE(nullptr, data);
  auto* peerHeader = PeerPayloadHeader(data);
  EXPECT_EQ(2, peerHeader->nChecks);
  const int expectedBytes = sizeof(*peerHeader) + 2 * sizeof(struct rasDiagnosticsCheckPayloadHeader) + 8;
  EXPECT_EQ(expectedBytes, peerHeader->payloadBytes);
  EXPECT_EQ(expectedBytes, nData);
  auto* checkHeader = FirstCheckHeader(data);
  EXPECT_EQ(RAS_DIAG_CHECK_GPU_MODEL, checkHeader->checkId);
  EXPECT_EQ(4, checkHeader->recordStride);
  EXPECT_EQ(1, checkHeader->nRecords);
  EXPECT_EQ(4, checkHeader->payloadBytes);
  EXPECT_EQ(0, memcmp(data + sizeof(*peerHeader) + sizeof(*checkHeader), "test", 4));
  auto* secondHeader = reinterpret_cast<struct rasDiagnosticsCheckPayloadHeader*>(
    data + sizeof(*peerHeader) + sizeof(*checkHeader) + checkHeader->payloadBytes);
  EXPECT_EQ(RAS_DIAG_CHECK_CUDA_DRIVER_VERSION, secondHeader->checkId);
  EXPECT_EQ(2, secondHeader->recordStride);
  EXPECT_EQ(2, secondHeader->nRecords);
  EXPECT_EQ(4, secondHeader->payloadBytes);
  EXPECT_EQ(0, memcmp(reinterpret_cast<char*>(secondHeader + 1), "next", 4));
  free(data);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_CheckWithZeroRecordsIsOmitted) {
  struct rasCollRequest req{};
  auto& contribution = g_checkHooks[RAS_DIAG_CHECK_ECC].collectLocalData;
  contribution.nRecords = 1;
  contribution.recordStride = 4;
  contribution.recordsBytes = 4;
  contribution.records = static_cast<char*>(malloc(4));
  ASSERT_NE(nullptr, contribution.records);
  memcpy(contribution.records, "test", 4);
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  ASSERT_EQ(ncclSuccess, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
  for (auto& h : g_checkHooks) EXPECT_EQ(1, h.collectLocalCalls);
  ASSERT_NE(nullptr, data);
  auto* peerHeader = PeerPayloadHeader(data);
  EXPECT_EQ(1, peerHeader->nChecks);
  EXPECT_EQ(sizeof(*peerHeader) + sizeof(rasDiagnosticsCheckPayloadHeader) + 4, static_cast<size_t>(nData));
  auto* checkHeader = FirstCheckHeader(data);
  EXPECT_EQ(RAS_DIAG_CHECK_ECC, checkHeader->checkId);
  EXPECT_EQ(1, checkHeader->nRecords);
  EXPECT_EQ(4, checkHeader->payloadBytes);
  EXPECT_EQ(0, memcmp(checkHeader + 1, "test", 4));
  free(data);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_CheckFailurePropagatesErrorAndFreesPayload) {
  struct rasCollRequest req{};
  g_checkHooks[RAS_DIAG_CHECK_ECC].collectLocalResult = ncclSystemError;
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  EXPECT_EQ(ncclSystemError, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
  EXPECT_EQ(nullptr, data);
  EXPECT_EQ(0, nData);
  EXPECT_EQ(1, g_freeCalls);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_AllocationFailuresLeaveOutputsEmpty) {
  struct rasCollRequest req{};
  g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.nRecords = 1;
  g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.recordStride = 4;
  g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.recordsBytes = 4;

  for (int failAt = 1; failAt <= 3; failAt++) {
    char* records = static_cast<char*>(malloc(4));
    g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.records = records;
    memcpy(records, "test", 4);
    const int collectCallsBefore = g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalCalls;
    g_reallocationCalls = 0;
    g_failReallocationCall = failAt;
    char* data = reinterpret_cast<char*>(1);
    int nData = 7;
    size_t reqLen = 0;
    struct rasCollRequest* pReq = &req;
    EXPECT_EQ(ncclSystemError, rasCollDiagInit(&pReq, &reqLen, &data, &nData)) << failAt;
    EXPECT_EQ(nullptr, data);
    EXPECT_EQ(0, nData);
    if (g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalCalls == collectCallsBefore) free(records);
    g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].collectLocalData.records = nullptr;
  }
}

// ===========================================================================
// rasCollDiagMerge
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_NullArgsReturnInternalError) {
  struct rasCollective coll{};
  struct rasMsg msg{};
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(nullptr, &msg));
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, nullptr));
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_NegativeSizesReturnInternalError) {
  struct rasCollective coll{};
  struct rasMsg msg{};

  coll.nData = -1;
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));

  coll.nData = 0;
  msg.collResp.nData = -1;
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));

  msg.collResp.nData = 0;
  msg.collResp.nPeers = -1;
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_ZeroIncomingDataIsNoOp) {
  struct rasCollective coll{};
  struct rasMsg msg{};
  msg.collResp.nData = 0;
  msg.collResp.nPeers = INT_MAX;
  EXPECT_EQ(ncclSuccess, rasCollDiagMerge(&coll, &msg));
  EXPECT_EQ(0, coll.nData);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_RejectsPeerCountAndPayloadSizeOverflow) {
  struct rasCollective coll{};
  struct rasMsg msg{};
  msg.collResp.nData = 1;
  msg.collResp.nPeers = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));

  msg.collResp.nPeers = 0;
  coll.nData = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_AppendsIncomingBytes) {
  struct rasCollective coll{};
  coll.data = static_cast<char*>(calloc(1, 4));
  memcpy(coll.data, "abcd", 4);
  coll.nData = 4;

  std::vector<char> extra = {'w', 'x', 'y', 'z'};
  constexpr int declaredData = 6;
  constexpr int nPeers = 1;
  int msgLen = static_cast<int>(rasMsgLength(RAS_MSG_COLLRESP));
  int dataOffset = msgLen + nPeers * static_cast<int>(sizeof(union ncclSocketAddress));
  ALIGN_SIZE(dataOffset, alignof(int64_t));
  msgLen = dataOffset + declaredData;
  std::vector<char> buf(std::max(sizeof(struct rasMsg), static_cast<size_t>(msgLen)), 0x5a);
  auto* msg = reinterpret_cast<struct rasMsg*>(buf.data());
  msg->collResp.nData = declaredData;
  msg->collResp.nPeers = nPeers;
  memcpy(buf.data() + dataOffset, extra.data(), extra.size());

  ASSERT_EQ(ncclSuccess, rasCollDiagMerge(&coll, msg));
  EXPECT_EQ(10, coll.nData);
  EXPECT_EQ(0, memcmp(coll.data, "abcdwxyz", 8));
  EXPECT_EQ(0x5a, static_cast<unsigned char>(coll.data[8]));
  EXPECT_EQ(0x5a, static_cast<unsigned char>(coll.data[9]));
  free(coll.data);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_AllocationFailurePreservesExistingData) {
  struct rasCollective coll{};
  coll.data = static_cast<char*>(malloc(4));
  memcpy(coll.data, "abcd", 4);
  coll.nData = 4;

  int dataOffset = static_cast<int>(rasMsgLength(RAS_MSG_COLLRESP));
  ALIGN_SIZE(dataOffset, alignof(int64_t));
  std::vector<char> buf(std::max(sizeof(struct rasMsg), static_cast<size_t>(dataOffset + 4)), 0);
  auto* msg = reinterpret_cast<struct rasMsg*>(buf.data());
  msg->collResp.nData = 4;
  g_failReallocationCall = 1;

  EXPECT_EQ(ncclSystemError, rasCollDiagMerge(&coll, msg));
  EXPECT_EQ(4, coll.nData);
  EXPECT_EQ(0, memcmp(coll.data, "abcd", 4));
  free(coll.data);
}

// ===========================================================================
// rasDiagnosticsResume (drives the static rasDiagnosticsSummarizePeerPayloads)
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, Resume_NullClientReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsResume(nullptr));
}

TEST_F(RasDiagnosticsMicrotest, Resume_MissingDiagnosticsStateReturnsInternalError) {
  auto* client = MakeClient();
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  ASSERT_NE(nullptr, client->coll);
  EXPECT_EQ(ncclInternalError, rasDiagnosticsResume(client));
  free(client->coll);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_MissingCollectiveReturnsInternalError) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx);
  ASSERT_NE(nullptr, client);
  ASSERT_NE(nullptr, client->diagnostics);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(ncclInternalError, rasDiagnosticsResume(client));
  rasDiagnosticsClientCleanup(client);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_EmptyGatheredDataStillEmitsHeaderAndFooter) {
  int target;
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);
  client->coll->nPeers = 3;
  client->coll->startTime = 1'000'000;
  g_clockNano = 3'500'000;

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  ASSERT_GE(g_emitCalls, 2);
  EXPECT_EQ("=== RAS Diagnostics ===", g_emittedLines.front());
  EXPECT_NE(std::string::npos, g_emittedLines.back().find("completed in 2.5 ms"));
  EXPECT_NE(std::string::npos, g_emittedLines.back().find("3 RAS peers"));
  EXPECT_EQ(&target, g_lastEmitTarget);
  EXPECT_EQ(1, g_finishCalls);
  EXPECT_EQ(&target, g_lastFinishTarget);
  EXPECT_EQ(ncclSuccess, g_lastFinishResult);
  EXPECT_EQ(1, g_collFreeCalls);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_InternalClientSkipsHeaderLine) {
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);
  client->internal = true;

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  ASSERT_EQ(1u, g_emittedLines.size());
  EXPECT_NE(std::string::npos, g_emittedLines.front().find("RAS diagnostics completed"));
  for (auto& line : g_emittedLines) EXPECT_EQ(std::string::npos, line.find("=== RAS Diagnostics ==="));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_CommScopedReportsCommNRanks) {
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = true;
  ctx.commNRanks = 7;
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);
  client->coll->nPeers = 999;  // Should be ignored in favor of ctx.commNRanks.
  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 0, 0, 0}}}});
  client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
  memcpy(client->coll->data, gathered.data(), gathered.size());
  client->coll->nData = static_cast<int>(gathered.size());

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  EXPECT_NE(std::string::npos, g_emittedLines.back().find("7 ranks"));
  EXPECT_TRUE(g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeCtx.hasCommFilter);
  EXPECT_EQ(7, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeCtx.commNRanks);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_SummarizeCalledForEveryCheckEvenWithoutRecords) {
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);

  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_ECC, 4, {1, 2, 3, 4}}}});
  client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
  memcpy(client->coll->data, gathered.data(), gathered.size());
  client->coll->nData = static_cast<int>(gathered.size());

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  for (int id = 0; id < RAS_DIAG_CHECK_COUNT; id++) {
    EXPECT_EQ(1, g_checkHooks[id].summarizeCalls);
    EXPECT_FALSE(g_checkHooks[id].lastSummarizeCtx.hasCommFilter);
    EXPECT_EQ(RecordingEmit, g_checkHooks[id].lastSummarizeReporter.emit);
    EXPECT_EQ(RecordingFinish, g_checkHooks[id].lastSummarizeReporter.finish);
  }
  EXPECT_EQ(4, g_checkHooks[RAS_DIAG_CHECK_ECC].lastSummarizeNData);
  ASSERT_EQ(4u, g_checkHooks[RAS_DIAG_CHECK_ECC].lastSummarizeData.size());
  EXPECT_EQ(0, memcmp(g_checkHooks[RAS_DIAG_CHECK_ECC].lastSummarizeData.data(), "\x01\x02\x03\x04", 4));
  // A check that no peer contributed to is still summarized, with an empty payload.
  EXPECT_EQ(0, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeNData);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_MultiplePeersAccumulateSameCheck) {
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);

  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 1, 1, 1}}},
                                     {{RAS_DIAG_CHECK_GPU_MODEL, 4, {2, 2, 2, 2}}}});
  client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
  memcpy(client->coll->data, gathered.data(), gathered.size());
  client->coll->nData = static_cast<int>(gathered.size());

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  EXPECT_EQ(8, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeNData);
  ASSERT_EQ(8u, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeData.size());
  const char expected[8] = {1, 1, 1, 1, 2, 2, 2, 2};
  EXPECT_EQ(0, memcmp(g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeData.data(), expected, 8));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_MultipleChecksInOnePeerReachCorrectHooks) {
  struct rasDiagnosticsContext ctx{};
  auto reporter = MakeRecordingReporter();
  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 2, 0, 0}},
                                      {RAS_DIAG_CHECK_ECC, 4, {3, 4, 0, 0}}}});

  ASSERT_EQ(ncclSuccess, rasDiagnosticsSummarizePeerPayloads(&ctx, &reporter, gathered.data(),
                                                             static_cast<int>(gathered.size())));
  EXPECT_EQ((std::vector<char>{1, 2, 0, 0}), g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeData);
  EXPECT_EQ((std::vector<char>{3, 4, 0, 0}), g_checkHooks[RAS_DIAG_CHECK_ECC].lastSummarizeData);
}

TEST_F(RasDiagnosticsMicrotest, Resume_TruncatedPeerHeaderReturnsErrorButStillFinishes) {
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);
  client->coll->data = static_cast<char*>(calloc(1, 2));
  client->coll->nData = 2;  // Smaller than sizeof(rasDiagnosticsPeerPayloadHeader).

  ASSERT_EQ(ncclInternalError, rasDiagnosticsResume(client));
  EXPECT_EQ(1, g_finishCalls);
  EXPECT_EQ(ncclInternalError, g_lastFinishResult);
  EXPECT_EQ(1, g_collFreeCalls);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, GetCheck_CountSentinelReturnsInternalError) {
  const auto* sentinel = reinterpret_cast<const struct rasDiagnosticsCheck*>(0x1);
  const struct rasDiagnosticsCheck* check = sentinel;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsGetCheck(RAS_DIAG_CHECK_COUNT, &check));
  EXPECT_EQ(sentinel, check);
}

TEST_F(RasDiagnosticsMicrotest, EveryRegisteredCheckDispatchesToItsOwnHooks) {
  struct rasDiagnosticsContext ctx{};
  auto reporter = MakeRecordingReporter();
  for (int id = 0; id < RAS_DIAG_CHECK_COUNT; ++id) {
    SCOPED_TRACE(id);
    const struct rasDiagnosticsCheck* check = nullptr;
    ASSERT_EQ(ncclSuccess, rasDiagnosticsGetCheck(static_cast<rasDiagnosticsCheckId>(id), &check));
    ASSERT_NE(nullptr, check);
    struct rasDiagnosticsLocalData data{};
    ASSERT_EQ(ncclSuccess, check->collectLocal(&ctx, &data));
    ASSERT_EQ(ncclSuccess, check->summarize(&ctx, &reporter, nullptr, 0));
    EXPECT_EQ(1, g_checkHooks[id].collectLocalCalls);
    EXPECT_EQ(1, g_checkHooks[id].summarizeCalls);
  }
}

TEST_F(RasDiagnosticsMicrotest, Resume_CountSentinelReturnsInternalError) {
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);

  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_COUNT, 4, {1, 2, 3, 4}}}});
  client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
  memcpy(client->coll->data, gathered.data(), gathered.size());
  client->coll->nData = static_cast<int>(gathered.size());

  EXPECT_EQ(ncclInternalError, rasDiagnosticsResume(client));
  EXPECT_EQ(1, g_finishCalls);
  EXPECT_EQ(ncclInternalError, g_lastFinishResult);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, DISABLED_Resume_UnknownWireCheckIdReturnsError) {
  SCOPED_TRACE("AICOMRCCL-2741: production loads an out-of-range enum before validating the wire ID");
  for (int id : {-1, 99}) {
    auto reporter = MakeRecordingReporter();
    struct rasDiagnosticsContext ctx{};
    auto* client = MakeDiagClient(ctx, &reporter, true);
    ASSERT_NE(nullptr, client);

    auto gathered = BuildGatheredData({{{id, 4, {1, 2, 3, 4}}}});
    client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
    memcpy(client->coll->data, gathered.data(), gathered.size());
    client->coll->nData = static_cast<int>(gathered.size());

    EXPECT_EQ(ncclInternalError, rasDiagnosticsResume(client)) << id;
    free(client);
  }
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsInvalidTopLevelArguments) {
  struct rasDiagnosticsContext ctx{};
  const char byte = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsSummarizePeerPayloads(nullptr, nullptr, &byte, 1));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, nullptr, 1));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, &byte, -1));
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsMalformedPeerHeaders) {
  struct rasDiagnosticsContext ctx{};
  auto payload = BuildPeerPayload({});
  auto* header = PeerPayloadHeader(payload);

  header->nChecks = -1;
  header->payloadBytes = static_cast<int>(payload.size());
  ExpectSummaryRejected(ctx, payload);

  header->nChecks = 0;
  header->payloadBytes = static_cast<int>(sizeof(*header)) - 1;
  ExpectSummaryRejected(ctx, payload);

  header->payloadBytes = static_cast<int>(payload.size()) + 1;
  ExpectSummaryRejected(ctx, payload);

  auto gathered = BuildGatheredData({{}, {}});
  auto* secondHeader = PeerPayloadHeader(gathered.data() + sizeof(struct rasDiagnosticsPeerPayloadHeader));
  secondHeader->payloadBytes = static_cast<int>(sizeof(*secondHeader)) + 1;
  ExpectSummaryRejected(ctx, gathered);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_AcceptsHealthyEmptyPeerBlock) {
  struct rasDiagnosticsContext ctx{};
  auto reporter = MakeRecordingReporter();
  auto gathered = BuildGatheredData({{}});
  auto* header = PeerPayloadHeader(gathered);
  ASSERT_EQ(0, header->nChecks);
  ASSERT_EQ(static_cast<int>(sizeof(*header)), header->payloadBytes);

  ASSERT_EQ(ncclSuccess, rasDiagnosticsSummarizePeerPayloads(&ctx, &reporter, gathered.data(),
                                                             static_cast<int>(gathered.size())));
  for (int id = 0; id < RAS_DIAG_CHECK_COUNT; id++) {
    EXPECT_EQ(1, g_checkHooks[id].summarizeCalls);
    EXPECT_EQ(0, g_checkHooks[id].lastSummarizeNData);
  }
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsTruncatedCheckHeaderAndTrailingBytes) {
  struct rasDiagnosticsContext ctx{};
  auto payload = BuildPeerPayload({});
  auto* peerHeader = PeerPayloadHeader(payload);
  peerHeader->nChecks = 1;
  peerHeader->payloadBytes = static_cast<int>(payload.size());
  ExpectSummaryRejected(ctx, payload);

  payload = BuildPeerPayload({});
  payload.push_back(0);
  peerHeader = PeerPayloadHeader(payload);
  peerHeader->nChecks = 0;
  peerHeader->payloadBytes = static_cast<int>(payload.size());
  ExpectSummaryRejected(ctx, payload);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsMalformedCheckMetadata) {
  struct rasDiagnosticsContext ctx{};
  auto expectRejected = [&](auto mutate) {
    auto payload = BuildPeerPayload({{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 2, 3, 4}}});
    mutate(*FirstCheckHeader(payload));
    ExpectSummaryRejected(ctx, payload);
  };

  expectRejected([](auto& header) { header.recordStride = 0; });
  expectRejected([](auto& header) { header.nRecords = -1; });
  expectRejected([](auto& header) { header.payloadBytes = -1; });
  expectRejected([](auto& header) { header.payloadBytes = 5; });
  expectRejected([](auto& header) {
    header.recordStride = 2;
    header.nRecords = INT_MAX;
  });
  expectRejected([](auto& header) { header.nRecords = 2; });

  auto payload = BuildPeerPayload({{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 2, 3, 4}}});
  auto* checkHeader = FirstCheckHeader(payload);
  checkHeader->nRecords = 2;
  checkHeader->payloadBytes = 8;
  ExpectSummaryRejected(ctx, payload);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_SummarizerFailurePropagates) {
  struct rasDiagnosticsContext ctx{};
  auto reporter = MakeRecordingReporter();
  auto gathered = BuildGatheredData({{}});
  g_checkHooks[RAS_DIAG_CHECK_ECC].summarizeResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsSummarizePeerPayloads(&ctx, &reporter, gathered.data(),
                                                                  static_cast<int>(gathered.size())));
  EXPECT_EQ(1, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].summarizeCalls);
  EXPECT_EQ(1, g_checkHooks[RAS_DIAG_CHECK_CUDA_DRIVER_VERSION].summarizeCalls);
  EXPECT_EQ(1, g_checkHooks[RAS_DIAG_CHECK_ECC].summarizeCalls);
  EXPECT_EQ(0, g_checkHooks[RAS_DIAG_CHECK_NVLINK].summarizeCalls);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_ZeroRecordBlockDoesNotSetStride) {
  struct rasDiagnosticsContext ctx{};
  auto reporter = MakeRecordingReporter();
  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 8, {}}},
                                     {{RAS_DIAG_CHECK_GPU_MODEL, 4, {5, 6, 7, 8}}}});

  ASSERT_EQ(ncclSuccess, rasDiagnosticsSummarizePeerPayloads(&ctx, &reporter, gathered.data(),
                                                             static_cast<int>(gathered.size())));
  EXPECT_EQ(1, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].summarizeCalls);
  EXPECT_EQ(4, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeNData);
  EXPECT_EQ((std::vector<char>{5, 6, 7, 8}), g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].lastSummarizeData);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_AllocationFailurePropagates) {
  struct rasDiagnosticsContext ctx{};
  auto reporter = MakeRecordingReporter();
  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 2, 3, 4}}}});
  g_failAllocationCall = 1;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsSummarizePeerPayloads(&ctx, &reporter, gathered.data(),
                                                                 static_cast<int>(gathered.size())));
  EXPECT_EQ(0, g_checkHooks[RAS_DIAG_CHECK_GPU_MODEL].summarizeCalls);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsStrideChangesAcrossPeers) {
  struct rasDiagnosticsContext ctx{};
  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 2, 3, 4}}},
                                     {{RAS_DIAG_CHECK_GPU_MODEL, 2, {5, 6}}}});
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, gathered.data(), static_cast<int>(gathered.size())));
}

TEST_F(RasDiagnosticsMicrotest, Resume_ReporterFinishFailureIsLoggedButResultStillReturned) {
  auto reporter = MakeRecordingReporter();
  g_finishResult = ncclSystemError;
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, &reporter, true);
  ASSERT_NE(nullptr, client);

  EXPECT_EQ(ncclSuccess, rasDiagnosticsResume(client));  // finish()'s own result doesn't override ret.
  EXPECT_EQ(1, g_finishCalls);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_NullReporterFinishDoesNotCrash) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, nullptr, true);  // Default reporter has finish==nullptr.
  ASSERT_NE(nullptr, client);

  EXPECT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  free(client);
}

// ===========================================================================
// rasDiagnosticsStart
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, Start_NullClientReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsStart(nullptr));
}

TEST_F(RasDiagnosticsMicrotest, Start_MissingDiagnosticsStateReturnsInternalError) {
  auto* client = MakeClient();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsStart(client));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_PostsCollectiveAndAdvancesStateWhenIncomplete) {
  struct rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = true;
  ctx.commFilter.commHash = 0x77;
  ctx.commFilter.hostHash = 0x66;
  ctx.commFilter.pidHash = 0x55;
  auto* client = MakeDiagClient(ctx);
  ASSERT_NE(nullptr, client);
  client->timeout = 12345;
  g_netSendCollReqAllDone = false;
  auto* fakeColl = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  g_netSendCollReqCollToAssign = fakeColl;

  EXPECT_EQ(ncclInProgress, rasDiagnosticsStart(client));
  EXPECT_EQ(1, g_netSendCollReqCalls);
  EXPECT_NE(0u, g_lastSentReq.rootId);
  EXPECT_EQ(RAS_COLL_DIAG, g_lastSentReq.type);
  EXPECT_EQ(12345, g_lastSentReq.timeout);
  EXPECT_TRUE(g_lastSentReq.diag.hasCommFilter);
  EXPECT_EQ(0x77u, g_lastSentReq.diag.commFilter.commHash);
  EXPECT_EQ(0x66u, g_lastSentReq.diag.commFilter.hostHash);
  EXPECT_EQ(0x55u, g_lastSentReq.diag.commFilter.pidHash);
  EXPECT_EQ(nullptr, g_lastSendFromConn);
  EXPECT_EQ(RAS_CLIENT_DIAG_FINI, client->status);
  EXPECT_EQ(fakeColl, client->coll);
  free(client->coll);
  rasDiagnosticsClientCleanup(client);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_ReturnsSuccessWhenAlreadyAllDone) {
  struct rasDiagnosticsContext ctx{};
  auto reporter = MakeRecordingReporter();
  auto* client = MakeDiagClient(ctx, &reporter);
  ASSERT_NE(nullptr, client);
  g_netSendCollReqAllDone = true;
  auto* completed = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  ASSERT_NE(nullptr, completed);
  g_netSendCollReqCollToAssign = completed;

  EXPECT_EQ(ncclSuccess, rasDiagnosticsStart(client));
  EXPECT_EQ(RAS_CLIENT_DIAG_FINI, client->status);
  EXPECT_FALSE(g_lastSentReq.diag.hasCommFilter);
  ASSERT_EQ(completed, client->coll);
  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  EXPECT_EQ(1, g_finishCalls);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_SendFailureLeavesNoCollectiveAndCleansUpDiagnostics) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx);
  ASSERT_NE(nullptr, client);
  g_netSendCollReqResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsStart(client));
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_SendFailureFreesStaleCollectiveAndCleansUpDiagnostics) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeDiagClient(ctx, nullptr, true);
  ASSERT_NE(nullptr, client);
  auto* staleCollective = client->coll;
  g_netSendCollReqResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsStart(client));
  EXPECT_EQ(1, g_collFreeCalls);
  EXPECT_EQ(staleCollective, g_lastCollFree);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

#undef rasDiagnosticsPathsSummarize
#undef rasDiagnosticsPathsCollectLocal
#undef rasDiagnosticsNvidiaDriverVersionSummarize
#undef rasDiagnosticsNvidiaDriverVersionCollectLocal
#undef rasDiagnosticsXidSummarize
#undef rasDiagnosticsXidCollectLocal
#undef rasDiagnosticsAtsSummarize
#undef rasDiagnosticsAtsCollectLocal
#undef rasDiagnosticsIommuSummarize
#undef rasDiagnosticsIommuCollectLocal
#undef rasDiagnosticsRdmaTopoSummarize
#undef rasDiagnosticsRdmaTopoCollectLocal
#undef rasDiagnosticsNcclEnvSummarize
#undef rasDiagnosticsNcclEnvCollectLocal
#undef rasDiagnosticsNvLinkSummarize
#undef rasDiagnosticsNvLinkCollectLocal
#undef rasDiagnosticsEccSummarize
#undef rasDiagnosticsEccCollectLocal
#undef rasDiagnosticsCudaDriverVersionSummarize
#undef rasDiagnosticsCudaDriverVersionCollectLocal
#undef rasDiagnosticsGpuModelSummarize
#undef rasDiagnosticsGpuModelCollectLocal
#undef rasNetSendCollReq
#undef rasCollReqInit
#undef rasCollFree
#undef rasClientsHead
