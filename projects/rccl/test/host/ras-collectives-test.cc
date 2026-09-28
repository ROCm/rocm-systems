/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for `src/ras/collectives.cc`. Dependencies on sibling RAS
// units are replaced by translation-unit-local fakes.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <gtest/gtest.h>

#include "../common/LogCapture.hpp"

#define ncclSocketToString CollectivesTestNcclSocketToString
#define rasNetListeningSocket CollectivesTestRasNetListeningSocket
#define ncclCommsMutex CollectivesTestNcclCommsMutex
#define ncclComms CollectivesTestNcclComms
#define nNcclComms CollectivesTestNNcclComms
#define ncclCommsSorted CollectivesTestNcclCommsSorted
#define rasLine CollectivesTestRasLine
#define rasNextLink CollectivesTestRasNextLink
#define rasPrevLink CollectivesTestRasPrevLink
#define rasConnsHead CollectivesTestRasConnsHead
#define rasConnsTail CollectivesTestRasConnsTail
#define rasPeers CollectivesTestRasPeers
#define nRasPeers CollectivesTestNRasPeers
#define rasTimeoutFactorNs CollectivesTestRasTimeoutFactorNs
#define rasMsgAlloc CollectivesTestRasMsgAlloc
#define rasMsgHandleBCDeadPeer CollectivesTestRasMsgHandleBCDeadPeer
#define rasMsgHandleBCProfilerMask CollectivesTestRasMsgHandleBCProfilerMask
#define rasCollDiagInit CollectivesTestRasCollDiagInit
#define rasCollDiagMerge CollectivesTestRasCollDiagMerge
#define rasClientResume CollectivesTestRasClientResume
#define rasConnEnqueueMsg CollectivesTestRasConnEnqueueMsg
#define rasCollFree CollectivesTestRasCollFree
#define rasCollReqInit CollectivesTestRasCollReqInit
#define rasCollsPurgeConn CollectivesTestRasCollsPurgeConn
#define rasNetSendCollReq CollectivesTestRasNetSendCollReq
#define rasMsgHandleCollReq CollectivesTestRasMsgHandleCollReq
#define rasCollRecordHistory CollectivesTestRasCollRecordHistory
#define rasMsgHandleCollResp CollectivesTestRasMsgHandleCollResp
#define rasCollsHandleTimeouts CollectivesTestRasCollsHandleTimeouts
#define rasCollectivesTerminate CollectivesTestRasCollectivesTerminate
#define rasCollectivesHead CollectivesTestRasCollectivesHead
#define rasCollectivesTail CollectivesTestRasCollectivesTail

#include "comm.h"
#include "ras/ras_internal.h"

#define clockNano CollectivesTestClockNano

uint64_t CollectivesTestClockNano();

const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, const int) {
  buf[0] = '\0';
  return buf;
}

namespace {

int g_allocationCalls = 0;
int g_failAllocationCall = 0;

template <typename T>
ncclResult_t CollectivesTestCalloc(T** ptr, size_t count) {
  ++g_allocationCalls;
  if (g_allocationCalls == g_failAllocationCall) {
    *ptr = nullptr;
    return ncclSystemError;
  }
  return ncclCallocDebug(ptr, count, __FILE__, __LINE__, __func__, false);
}

}  // namespace

#undef ncclCalloc
#define ncclCalloc(...) CollectivesTestCalloc(__VA_ARGS__)
#include COLLECTIVES_CC_PATH
#undef ncclCalloc

#undef clockNano

namespace {

int64_t g_clockNano = 1'000'000'000;

}  // namespace

uint64_t CollectivesTestClockNano() { return static_cast<uint64_t>(g_clockNano); }

// ---------------------------------------------------------------------------
// TU-local fakes for collectives.cc's dependencies on ras.cc, rasnet.cc,
// peers.cc, and client_support.cc. Plain (non-namespaced) symbols, matching
// client-support-test.cc's convention for the same reason: they must be
// visible to the #include'd source above.
// ---------------------------------------------------------------------------

struct ncclSocket rasNetListeningSocket;
std::mutex ncclCommsMutex;
struct ncclComm** ncclComms = nullptr;
int nNcclComms = 0;
bool ncclCommsSorted = false;
char rasLine[SOCKET_NAME_MAXLEN + 1];

struct rasLink rasNextLink = {1}, rasPrevLink = {-1};
struct rasConnection* rasConnsHead = nullptr;
struct rasConnection* rasConnsTail = nullptr;

struct rasPeerInfo* rasPeers = nullptr;
int nRasPeers = 0;

// Unscaled (1 unit == 1 "second"): keeps RAS_COLLECTIVE_EXTRA_TIMEOUT commensurate with the
// small toy startTime/timeout values the tests below use, rather than real nanosecond scale.
int64_t rasTimeoutFactorNs(int64_t baseSeconds) { return baseSeconds; }

ncclResult_t rasMsgAlloc(struct rasMsg** msg, size_t msgLen) {
  const size_t totalSize = offsetof(struct rasMsgMeta, msg) + msgLen;
  auto* meta = static_cast<struct rasMsgMeta*>(calloc(1, totalSize));
  if (meta == nullptr) return ncclSystemError;
  *msg = &meta->msg;
  return ncclSuccess;
}

void FreeAllocatedMsg(struct rasMsg* msg) {
  if (msg == nullptr) return;
  auto* meta = reinterpret_cast<struct rasMsgMeta*>(reinterpret_cast<char*>(msg) - offsetof(struct rasMsgMeta, msg));
  free(meta);
}

namespace {

struct EnqueuedMsg {
  struct rasConnection* conn;
  struct rasMsg* msg;
  size_t msgLen;
  bool front;
};
std::vector<EnqueuedMsg> g_enqueuedMsgs;

int g_bcDeadPeerCalls = 0;
int g_bcProfilerMaskCalls = 0;
bool g_bcDeadPeerDone = false;
bool g_bcProfilerMaskDone = false;

ncclResult_t g_collDiagInitResult = ncclSuccess;
int g_collDiagInitCalls = 0;
ncclResult_t g_collDiagMergeResult = ncclSuccess;
int g_collDiagMergeCalls = 0;
struct rasCollective* g_lastDiagMergeColl = nullptr;

int g_clientResumeCalls = 0;
struct rasCollective* g_lastClientResumeColl = nullptr;
ncclResult_t g_clientResumeResult = ncclSuccess;

}  // namespace

void rasMsgHandleBCDeadPeer(struct rasCollRequest** pReq, size_t* pReqLen, bool* pDone) {
  ++g_bcDeadPeerCalls;
  *pReqLen = rasCollDataLength((*pReq)->type);
  *pDone = g_bcDeadPeerDone;
}

void rasMsgHandleBCProfilerMask(struct rasCollRequest** pReq, size_t* pReqLen, bool* pDone) {
  ++g_bcProfilerMaskCalls;
  *pReqLen = rasCollDataLength((*pReq)->type);
  *pDone = g_bcProfilerMaskDone;
}

ncclResult_t rasCollDiagInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData) {
  ++g_collDiagInitCalls;
  *pReqLen = rasCollDataLength((*pReq)->type);
  *pData = nullptr;
  *pNData = 0;
  if (g_collDiagInitResult != ncclSuccess) return g_collDiagInitResult;
  *pNData = 4;
  *pData = static_cast<char*>(malloc(*pNData));
  memcpy(*pData, "diag", *pNData);
  return ncclSuccess;
}

ncclResult_t rasCollDiagMerge(struct rasCollective* coll, struct rasMsg*) {
  ++g_collDiagMergeCalls;
  g_lastDiagMergeColl = coll;
  return g_collDiagMergeResult;
}

ncclResult_t rasClientResume(struct rasCollective* coll) {
  ++g_clientResumeCalls;
  g_lastClientResumeColl = coll;
  return g_clientResumeResult;
}

void rasConnEnqueueMsg(struct rasConnection* conn, struct rasMsg* msg, size_t msgLen, bool front) {
  g_enqueuedMsgs.push_back({conn, msg, msgLen, front});
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

namespace {

union ncclSocketAddress MakeAddr(uint16_t port) {
  union ncclSocketAddress addr{};
  addr.sin.sin_family = AF_INET;
  addr.sin.sin_port = htons(port);
  addr.sin.sin_addr.s_addr = htonl(0x7f000001);
  return addr;
}

void LinkConn(struct rasConnection** head, struct rasConnection** tail, struct rasConnection* conn) {
  if (*head) {
    (*tail)->next = conn;
    conn->prev = *tail;
    *tail = conn;
  } else {
    *head = *tail = conn;
  }
}

// Every connection reachable via a rasLink's conns list must also live on rasConnsHead in
// production (getNewCollEntry sizes coll->fwdConns by walking rasConnsHead) -- so MakeConn
// links onto the master list itself, and FreeConnList()/ResetWholeFileSeams() is the only
// place that frees it back. Tests must NOT separately free() a MakeConn() connection.
struct rasConnection* MakeConn(uint16_t port) {
  auto* conn = static_cast<struct rasConnection*>(calloc(1, sizeof(struct rasConnection)));
  conn->addr = MakeAddr(port);
  LinkConn(&rasConnsHead, &rasConnsTail, conn);
  return conn;
}

struct rasSocket* MakeSocketForConn(struct rasConnection* conn, rasSocketStatus status, int64_t createTime) {
  auto* sock = static_cast<struct rasSocket*>(calloc(1, sizeof(struct rasSocket)));
  sock->conn = conn;
  sock->status = status;
  sock->createTime = createTime;
  conn->sock = sock;
  return sock;
}

// Adds a rasLinkConn entry (pointing at conn) to link's conns list.
void AddLinkConn(struct rasLink* link, struct rasConnection* conn) {
  auto* lc = static_cast<struct rasLinkConn*>(calloc(1, sizeof(struct rasLinkConn)));
  lc->conn = conn;
  lc->peerIdx = -1;
  lc->next = link->conns;
  link->conns = lc;
}

void FreeLinkConns(struct rasLink* link) {
  struct rasLinkConn* lc = link->conns;
  while (lc) {
    struct rasLinkConn* next = lc->next;
    free(lc);
    lc = next;
  }
  link->conns = nullptr;
}

void FreeConnList() {
  struct rasConnection* conn = rasConnsHead;
  while (conn) {
    struct rasConnection* next = conn->next;
    free(conn->sock);
    free(conn);
    conn = next;
  }
  rasConnsHead = rasConnsTail = nullptr;
}

// Builds a heap RAS_MSG_COLLREQ message (collReq portion only, matching rasCollDataLength).
struct rasMsg* MakeCollReqMsg(rasCollectiveType type, const union ncclSocketAddress& rootAddr, uint64_t rootId,
                              int64_t timeout) {
  struct rasMsg* msg = nullptr;
  size_t reqLen = rasCollDataLength(type);
  EXPECT_EQ(ncclSuccess, rasMsgAlloc(&msg, rasMsgLength(RAS_MSG_COLLREQ, type)));
  (void)reqLen;
  msg->type = RAS_MSG_COLLREQ;
  msg->collReq.rootAddr = rootAddr;
  msg->collReq.rootId = rootId;
  msg->collReq.timeout = timeout;
  msg->collReq.type = type;
  return msg;
}

// Builds a heap RAS_MSG_COLLRESP message with optional peers/data, matching rasConnSendCollResp's layout.
struct rasMsg* MakeCollRespMsg(const union ncclSocketAddress& rootAddr, uint64_t rootId, int nLegTimeouts,
                               const std::vector<union ncclSocketAddress>& peers, const std::vector<char>& data) {
  struct rasMsg* msg = nullptr;
  int msgLen = static_cast<int>(rasMsgLength(RAS_MSG_COLLRESP)) + static_cast<int>(peers.size() * sizeof(union ncclSocketAddress));
  int dataOffset = 0;
  if (!data.empty()) {
    ALIGN_SIZE(msgLen, alignof(int64_t));
    dataOffset = msgLen;
    msgLen += static_cast<int>(data.size());
  }
  EXPECT_EQ(ncclSuccess, rasMsgAlloc(&msg, msgLen));
  msg->type = RAS_MSG_COLLRESP;
  msg->collResp.rootAddr = rootAddr;
  msg->collResp.rootId = rootId;
  msg->collResp.nLegTimeouts = nLegTimeouts;
  msg->collResp.nPeers = static_cast<int>(peers.size());
  msg->collResp.nData = static_cast<int>(data.size());
  if (!peers.empty()) memcpy(msg->collResp.peers, peers.data(), peers.size() * sizeof(union ncclSocketAddress));
  if (!data.empty()) memcpy(reinterpret_cast<char*>(msg) + dataOffset, data.data(), data.size());
  return msg;
}

// Constructs a rasCollective directly (bypassing rasNetSendCollReq) and links it onto
// rasCollectivesHead/Tail, mirroring getNewCollEntry.
struct rasCollective* MakeCollective(rasCollectiveType type, const union ncclSocketAddress& rootAddr, uint64_t rootId,
                                     struct rasConnection* fromConn = nullptr, int64_t timeout = 0) {
  auto* coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  coll->rootAddr = rootAddr;
  coll->rootId = rootId;
  coll->type = type;
  coll->timeout = timeout;
  coll->fromConn = fromConn;
  coll->startTime = g_clockNano;
  coll->fwdConns = static_cast<struct rasConnection**>(calloc(8, sizeof(*coll->fwdConns)));

  if (rasCollectivesHead) {
    rasCollectivesTail->next = coll;
    coll->prev = rasCollectivesTail;
    rasCollectivesTail = coll;
  } else {
    rasCollectivesHead = rasCollectivesTail = coll;
  }
  return coll;
}

ncclResult_t DeliverResp(struct rasMsg* msg, struct rasConnection* respConn = nullptr) {
  struct rasSocket sock{};
  sock.conn = respConn;
  const ncclResult_t result = rasMsgHandleCollResp(msg, &sock);
  FreeAllocatedMsg(msg);
  return result;
}

void SetPeers(struct rasCollective* coll, const std::vector<union ncclSocketAddress>& peers) {
  free(coll->peers);
  coll->nPeers = static_cast<int>(peers.size());
  coll->peers = static_cast<union ncclSocketAddress*>(calloc(peers.size() ? peers.size() : 1, sizeof(*coll->peers)));
  if (!peers.empty()) memcpy(coll->peers, peers.data(), peers.size() * sizeof(*coll->peers));
}

void SetData(struct rasCollective* coll, const std::vector<char>& data) {
  free(coll->data);
  coll->nData = static_cast<int>(data.size());
  coll->data = static_cast<char*>(calloc(data.size() ? data.size() : 1, 1));
  if (!data.empty()) memcpy(coll->data, data.data(), data.size());
}

// -------- RAS_COLL_CONNS payload builder, matching struct rasCollConns's layout. --------

std::vector<char> BuildConnsData(int64_t travelTimeMin, int64_t travelTimeMax, int64_t travelTimeSum,
                                  int64_t travelTimeCount, int nConns,
                                  const std::vector<std::pair<union ncclSocketAddress, union ncclSocketAddress>>& negMins,
                                  int64_t negMinValue = -1) {
  size_t total = sizeof(struct rasCollConns) + negMins.size() * sizeof(struct rasCollConns::negativeMin);
  std::vector<char> buf(total, 0);
  auto* d = reinterpret_cast<struct rasCollConns*>(buf.data());
  d->travelTimeMin = travelTimeMin;
  d->travelTimeMax = travelTimeMax;
  d->travelTimeSum = travelTimeSum;
  d->travelTimeCount = travelTimeCount;
  d->nConns = nConns;
  d->nNegativeMins = static_cast<int>(negMins.size());
  for (size_t i = 0; i < negMins.size(); i++) {
    d->negativeMins[i].source = negMins[i].first;
    d->negativeMins[i].dest = negMins[i].second;
    d->negativeMins[i].travelTimeMin = negMinValue;
  }
  return buf;
}

// -------- RAS_COLL_COMMS payload builder, matching struct rasCollComms's nested layout. --------

struct RankSpec {
  int commRank;
  int peerIdx = 0;
};

struct CommSpec {
  uint64_t commHash;
  uint64_t hostHash;
  uint64_t pidHash;
  int commNRanks;
  std::vector<RankSpec> ranks;
  std::vector<int> missingRanks;  // commRank values with no addr data (recorded as index into a peers vector below)
  std::vector<union ncclSocketAddress> missingAddrs;
};

CommSpec MakeCommSpec(uint64_t commHash, int commNRanks, std::vector<RankSpec> ranks = {}) {
  CommSpec spec;
  spec.commHash = commHash;
  spec.hostHash = commHash + 1;
  spec.pidHash = commHash + 2;
  spec.commNRanks = commNRanks;
  spec.ranks = std::move(ranks);
  return spec;
}

std::vector<char> BuildRasCollComms(const std::vector<CommSpec>& comms) {
  int nComms = static_cast<int>(comms.size());
  int nRanks = 0, nMissingRanks = 0;
  for (auto& c : comms) {
    nRanks += static_cast<int>(c.ranks.size());
    nMissingRanks += static_cast<int>(c.missingRanks.size());
  }
  size_t total = sizeof(struct rasCollComms) + nComms * sizeof(struct rasCollComms::comm) +
                 nRanks * sizeof(struct rasCollComms::comm::rank) + nMissingRanks * sizeof(struct rasCollCommsMissingRank);
  std::vector<char> buf(total, 0);
  auto* d = reinterpret_cast<struct rasCollComms*>(buf.data());
  d->nComms = nComms;
  auto* comm = d->comms;
  for (auto& c : comms) {
    comm->commId.commHash = c.commHash;
    comm->commId.hostHash = c.hostHash;
    comm->commId.pidHash = c.pidHash;
    comm->commNRanks = c.commNRanks;
    comm->nRanks = static_cast<int>(c.ranks.size());
    comm->nMissingRanks = static_cast<int>(c.missingRanks.size());
    for (size_t i = 0; i < c.ranks.size(); i++) {
      comm->ranks[i].commRank = c.ranks[i].commRank;
      comm->ranks[i].peerIdx = c.ranks[i].peerIdx;
    }
    auto* missing = reinterpret_cast<struct rasCollCommsMissingRank*>(comm->ranks + comm->nRanks);
    for (size_t i = 0; i < c.missingRanks.size(); i++) {
      missing[i].commRank = c.missingRanks[i];
      if (i < c.missingAddrs.size()) missing[i].addr = c.missingAddrs[i];
    }
    comm = reinterpret_cast<struct rasCollComms::comm*>(reinterpret_cast<char*>(comm + 1) +
                                                        comm->nRanks * sizeof(*comm->ranks) +
                                                        comm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
  }
  return buf;
}

struct rasCollComms::comm* NextComm(struct rasCollComms::comm* comm) {
  return reinterpret_cast<struct rasCollComms::comm*>(reinterpret_cast<char*>(comm + 1) +
                                                      comm->nRanks * sizeof(*comm->ranks) +
                                                      comm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
}

void ResetWholeFileSeams() {
  g_enqueuedMsgs.clear();
  g_bcDeadPeerCalls = g_bcProfilerMaskCalls = 0;
  g_bcDeadPeerDone = g_bcProfilerMaskDone = false;
  g_collDiagInitResult = ncclSuccess;
  g_collDiagInitCalls = 0;
  g_collDiagMergeResult = ncclSuccess;
  g_collDiagMergeCalls = 0;
  g_lastDiagMergeColl = nullptr;
  g_clientResumeCalls = 0;
  g_lastClientResumeColl = nullptr;
  g_clientResumeResult = ncclSuccess;
  g_allocationCalls = 0;
  g_failAllocationCall = 0;

  rasCollectivesTerminate();
  rasCollectivesHead = rasCollectivesTail = nullptr;

  FreeConnList();
  FreeLinkConns(&rasNextLink);
  FreeLinkConns(&rasPrevLink);

  free(rasPeers);
  rasPeers = nullptr;
  nRasPeers = 0;

  free(ncclComms);
  ncclComms = nullptr;
  nNcclComms = 0;
  ncclCommsSorted = false;

  memset(&rasNetListeningSocket, 0, sizeof(rasNetListeningSocket));
  g_clockNano = 1'000'000'000;

  nRasCollHistory = 0;
  rasCollHistNextIdx = 0;
  rasCollLastId = 0;
}

class RasCollectivesMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetWholeFileSeams(); }
  void TearDown() override {
    for (auto& e : g_enqueuedMsgs) FreeAllocatedMsg(e.msg);
    ResetWholeFileSeams();
  }
};

}  // namespace

// ===========================================================================
// rasCollReqInit
// ===========================================================================

TEST_F(RasCollectivesMicrotest, CollReqInit_CopiesListeningAddressAndAssignsUniqueId) {
  rasNetListeningSocket.addr = MakeAddr(4321);
  struct rasCollRequest req{};
  rasCollReqInit(&req);
  EXPECT_EQ(htons(4321), req.rootAddr.sin.sin_port);
  EXPECT_EQ(1u, req.rootId);
}

TEST_F(RasCollectivesMicrotest, CollReqInit_IdsAreMonotonicallyIncreasing) {
  struct rasCollRequest req1{}, req2{};
  rasCollReqInit(&req1);
  rasCollReqInit(&req2);
  EXPECT_EQ(req1.rootId + 1, req2.rootId);
}

// ===========================================================================
// rasCollHistoryAdd / rasCollRecordHistory (direct: same-TU statics)
// ===========================================================================

TEST_F(RasCollectivesMicrotest, HistoryAdd_RecordsEntryAtCurrentIndex) {
  union ncclSocketAddress addr = MakeAddr(111);
  rasCollHistoryAdd(&addr, 42);
  EXPECT_EQ(1, nRasCollHistory);
  EXPECT_EQ(1, rasCollHistNextIdx);
  EXPECT_EQ(42u, rasCollHistory[0].rootId);
}

TEST_F(RasCollectivesMicrotest, HistoryAdd_WrapsAroundAfterCollHistorySize) {
  union ncclSocketAddress addr = MakeAddr(111);
  for (int i = 0; i < COLL_HISTORY_SIZE; i++) rasCollHistoryAdd(&addr, i);
  EXPECT_EQ(COLL_HISTORY_SIZE, nRasCollHistory);
  EXPECT_EQ(0, rasCollHistNextIdx);
  rasCollHistoryAdd(&addr, 999);
  EXPECT_EQ(COLL_HISTORY_SIZE, nRasCollHistory);  // Capped, not still growing.
  EXPECT_EQ(1, rasCollHistNextIdx);
  EXPECT_EQ(999u, rasCollHistory[0].rootId);  // Oldest slot overwritten.
}

TEST_F(RasCollectivesMicrotest, RecordHistory_NullCollIsNoOp) {
  rasCollRecordHistory(nullptr);
  EXPECT_EQ(0, nRasCollHistory);
}

TEST_F(RasCollectivesMicrotest, RecordHistory_DelegatesToHistoryAdd) {
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(55), 7);
  rasCollRecordHistory(coll);
  EXPECT_EQ(1, nRasCollHistory);
  EXPECT_EQ(7u, rasCollHistory[0].rootId);
}

// ===========================================================================
// rasCollFree / rasCollectivesTerminate
// ===========================================================================

TEST_F(RasCollectivesMicrotest, CollFree_NullIsNoOp) { rasCollFree(nullptr); }

TEST_F(RasCollectivesMicrotest, CollFree_UnlinksSoleEntry) {
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(1), 1);
  rasCollFree(coll);
  EXPECT_EQ(nullptr, rasCollectivesHead);
  EXPECT_EQ(nullptr, rasCollectivesTail);
}

TEST_F(RasCollectivesMicrotest, CollFree_UnlinksHeadKeepingRest) {
  auto* c1 = MakeCollective(RAS_COLL_CONNS, MakeAddr(1), 1);
  auto* c2 = MakeCollective(RAS_COLL_CONNS, MakeAddr(2), 2);
  rasCollFree(c1);
  EXPECT_EQ(c2, rasCollectivesHead);
  EXPECT_EQ(nullptr, c2->prev);
}

TEST_F(RasCollectivesMicrotest, CollFree_UnlinksTailKeepingRest) {
  auto* c1 = MakeCollective(RAS_COLL_CONNS, MakeAddr(1), 1);
  auto* c2 = MakeCollective(RAS_COLL_CONNS, MakeAddr(2), 2);
  rasCollFree(c2);
  EXPECT_EQ(c1, rasCollectivesTail);
  EXPECT_EQ(nullptr, c1->next);
}

TEST_F(RasCollectivesMicrotest, CollFree_UnlinksMiddleEntry) {
  auto* c1 = MakeCollective(RAS_COLL_CONNS, MakeAddr(1), 1);
  auto* c2 = MakeCollective(RAS_COLL_CONNS, MakeAddr(2), 2);
  auto* c3 = MakeCollective(RAS_COLL_CONNS, MakeAddr(3), 3);
  rasCollFree(c2);
  EXPECT_EQ(c3, c1->next);
  EXPECT_EQ(c1, c3->prev);
}

TEST_F(RasCollectivesMicrotest, CollectivesTerminate_FreesEveryEntry) {
  MakeCollective(RAS_COLL_CONNS, MakeAddr(1), 1);
  MakeCollective(RAS_COLL_CONNS, MakeAddr(2), 2);
  rasCollectivesTerminate();
  EXPECT_EQ(nullptr, rasCollectivesHead);
  EXPECT_EQ(nullptr, rasCollectivesTail);
}

// ===========================================================================
// rasNetSendCollReq
// ===========================================================================

TEST_F(RasCollectivesMicrotest, NetSendCollReq_DeadPeerBroadcastDoneSkipsForwarding) {
  g_bcDeadPeerDone = true;
  auto* conn = MakeConn(10);
  MakeSocketForConn(conn, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, conn);
  struct rasCollRequest req{};
  req.type = RAS_BC_DEADPEER;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req));
  EXPECT_EQ(1, g_bcDeadPeerCalls);
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  // Duplicate history is still recorded even when the handler reports done.
  EXPECT_EQ(1, nRasCollHistory);
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_ProfilerMaskBroadcastNotDoneForwards) {
  g_bcProfilerMaskDone = false;
  auto* conn = MakeConn(10);
  MakeSocketForConn(conn, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, conn);
  struct rasCollRequest req{};
  req.type = RAS_BC_PROFILER_MASK;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req));
  EXPECT_EQ(1, g_bcProfilerMaskCalls);
  EXPECT_EQ(1u, g_enqueuedMsgs.size());

  req.rootId = 6;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req));
  EXPECT_EQ(2, g_bcProfilerMaskCalls);
  ASSERT_EQ(2u, g_enqueuedMsgs.size());
  EXPECT_EQ(conn, g_enqueuedMsgs[1].conn);
  EXPECT_EQ(6u, g_enqueuedMsgs[1].msg->collReq.rootId);
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_CollectiveAllocationFailureReturnsError) {
  struct rasCollRequest req{};
  req.type = RAS_COLL_CONNS;
  struct rasCollective* coll = nullptr;
  g_failAllocationCall = 1;
  EXPECT_EQ(ncclSystemError, rasNetSendCollReq(&req, nullptr, &coll));
  EXPECT_EQ(nullptr, coll);
  EXPECT_EQ(nullptr, rasCollectivesHead);
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_ForwardArrayAllocationFailureCleansUpCollective) {
  MakeConn(10);
  struct rasCollRequest req{};
  req.type = RAS_COLL_CONNS;
  struct rasCollective* coll = nullptr;
  g_failAllocationCall = 2;
  EXPECT_EQ(ncclSystemError, rasNetSendCollReq(&req, nullptr, &coll));
  EXPECT_EQ(nullptr, coll);
  EXPECT_EQ(nullptr, rasCollectivesHead);
  EXPECT_EQ(nullptr, rasCollectivesTail);
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_TrackedTypeCreatesCollectiveWithSelfAsFirstPeer) {
  rasNetListeningSocket.addr = MakeAddr(4321);
  struct rasCollRequest req{};
  req.type = RAS_COLL_CONNS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  req.timeout = 123;
  struct rasCollective* coll = nullptr;
  bool allDone = false;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, &allDone, &coll));
  ASSERT_NE(nullptr, coll);
  EXPECT_EQ(coll, rasCollectivesHead);
  EXPECT_EQ(1, coll->nPeers);
  EXPECT_EQ(htons(4321), coll->peers[0].sin.sin_port);
  EXPECT_EQ(123, coll->timeout);
  EXPECT_TRUE(allDone);  // No connections to forward to.
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_CommsInitPopulatesEmptyPayloadWhenNoComms) {
  struct rasCollRequest req{};
  req.type = RAS_COLL_COMMS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  ASSERT_NE(nullptr, coll);
  ASSERT_NE(nullptr, coll->data);
  EXPECT_EQ(0, reinterpret_cast<struct rasCollComms*>(coll->data)->nComms);
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_DiagInitFailureContinuesWithEmptyResponse) {
  g_collDiagInitResult = ncclSystemError;
  struct rasCollRequest req{};
  req.type = RAS_COLL_DIAG;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  ASSERT_NE(nullptr, coll);
  EXPECT_EQ(1, g_collDiagInitCalls);
  EXPECT_EQ(0, coll->nData);
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_DiagInitSuccessKeepsLocalPayload) {
  struct rasCollRequest req{};
  req.type = RAS_COLL_DIAG;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  ASSERT_NE(nullptr, coll);
  EXPECT_EQ(1, g_collDiagInitCalls);
  ASSERT_EQ(4, coll->nData);
  EXPECT_EQ(0, memcmp(coll->data, "diag", 4));
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_ForwardsOnlyThroughReadyNonDelayedConnections) {
  auto* ready = MakeConn(10);
  MakeSocketForConn(ready, RAS_SOCK_READY, 0);
  auto* notReady = MakeConn(11);
  MakeSocketForConn(notReady, RAS_SOCK_CONNECTING, 0);
  auto* delayed = MakeConn(12);
  MakeSocketForConn(delayed, RAS_SOCK_READY, 0);
  delayed->experiencingDelays = true;
  auto* noSocket = MakeConn(13);
  AddLinkConn(&rasNextLink, ready);
  AddLinkConn(&rasNextLink, notReady);
  AddLinkConn(&rasNextLink, delayed);
  AddLinkConn(&rasNextLink, noSocket);
  AddLinkConn(&rasNextLink, nullptr);

  struct rasCollRequest req{};
  req.type = RAS_COLL_CONNS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  bool allDone = false;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, &allDone, &coll));
  EXPECT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(ready, g_enqueuedMsgs[0].conn);
  EXPECT_EQ(1, coll->nFwdSent);
  EXPECT_EQ(ready, coll->fwdConns[0]);
  EXPECT_FALSE(allDone);  // Sent one, received zero.
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_SkipsFromConnAndSkipsAlreadyLinkFlaggedConn) {
  auto* shared = MakeConn(10);
  MakeSocketForConn(shared, RAS_SOCK_READY, 0);
  // shared appears in both links (a small ring); should be sent through at most once.
  AddLinkConn(&rasNextLink, shared);
  AddLinkConn(&rasPrevLink, shared);

  struct rasCollRequest req{};
  req.type = RAS_COLL_CONNS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  EXPECT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(1, coll->nFwdSent);
}

TEST_F(RasCollectivesMicrotest, NetSendCollReq_FromConnIsNeverSentBackTo) {
  auto* fromConn = MakeConn(10);
  MakeSocketForConn(fromConn, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, fromConn);

  struct rasCollRequest req{};
  req.type = RAS_COLL_CONNS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll, fromConn));
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  EXPECT_EQ(fromConn, coll->fromConn);
}

// ===========================================================================
// rasMsgHandleCollReq
// ===========================================================================

TEST_F(RasCollectivesMicrotest, MsgHandleCollReq_NullConnReturnsInternalError) {
  auto* msg = MakeCollReqMsg(RAS_COLL_CONNS, MakeAddr(9), 5, 0);
  struct rasSocket sock{};
  sock.conn = nullptr;
  EXPECT_EQ(ncclInternalError, rasMsgHandleCollReq(msg, &sock));
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollReq_DuplicateFinishedSendsEmptyResponse) {
  union ncclSocketAddress root = MakeAddr(9);
  rasCollHistoryAdd(&root, 5);
  auto* conn = MakeConn(1);
  auto* msg = MakeCollReqMsg(RAS_COLL_CONNS, root, 5, 0);
  struct rasSocket sock{};
  sock.conn = conn;
  EXPECT_EQ(ncclSuccess, rasMsgHandleCollReq(msg, &sock));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(conn, g_enqueuedMsgs[0].conn);
  auto* resp = g_enqueuedMsgs[0].msg;
  EXPECT_EQ(RAS_MSG_COLLRESP, resp->type);
  EXPECT_EQ(0, resp->collResp.nPeers);
  EXPECT_EQ(0, resp->collResp.nData);
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollReq_SameRootDifferentIdIsRebroadcast) {
  union ncclSocketAddress root = MakeAddr(9);
  rasCollHistoryAdd(&root, 5);
  auto* fromConn = MakeConn(1);
  auto* forward = MakeConn(2);
  MakeSocketForConn(forward, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, forward);
  auto* msg = MakeCollReqMsg(RAS_COLL_CONNS, root, 6, 0);
  struct rasSocket sock{};
  sock.conn = fromConn;
  EXPECT_EQ(ncclSuccess, rasMsgHandleCollReq(msg, &sock));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(forward, g_enqueuedMsgs[0].conn);
  EXPECT_EQ(RAS_MSG_COLLREQ, g_enqueuedMsgs[0].msg->type);
  EXPECT_EQ(6u, g_enqueuedMsgs[0].msg->collReq.rootId);
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollReq_BroadcastDuplicateFinishedSendsNoResponse) {
  union ncclSocketAddress root = MakeAddr(9);
  rasCollHistoryAdd(&root, 5);
  auto* conn = MakeConn(1);
  auto* msg = MakeCollReqMsg(RAS_BC_DEADPEER, root, 5, 0);
  struct rasSocket sock{};
  sock.conn = conn;
  EXPECT_EQ(ncclSuccess, rasMsgHandleCollReq(msg, &sock));
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  EXPECT_EQ(0, g_bcDeadPeerCalls);
  EXPECT_EQ(1, nRasCollHistory);
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollReq_DuplicateOngoingSendsEmptyResponseAndSkipsRebroadcast) {
  union ncclSocketAddress root = MakeAddr(9);
  MakeCollective(RAS_COLL_CONNS, root, 5);
  auto* conn = MakeConn(1);
  auto* msg = MakeCollReqMsg(RAS_COLL_CONNS, root, 5, 0);
  struct rasSocket sock{};
  sock.conn = conn;
  EXPECT_EQ(ncclSuccess, rasMsgHandleCollReq(msg, &sock));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(0, g_enqueuedMsgs[0].msg->collResp.nPeers);
  // Only the pre-existing collective is on the list; rebroadcast was skipped (goto exit).
  ASSERT_NE(nullptr, rasCollectivesHead);
  EXPECT_EQ(nullptr, rasCollectivesHead->next);
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollReq_TypeMismatchLogsButStillRebroadcasts) {
  union ncclSocketAddress root = MakeAddr(9);
  MakeCollective(RAS_COLL_CONNS, root, 5);  // Ongoing collective has a different type than the incoming request.
  // A live forwarding connection keeps the new (COMMS) collective from completing
  // immediately, so both collectives are still observable on the list afterward.
  auto* fwd = MakeConn(20);
  MakeSocketForConn(fwd, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, fwd);
  auto* conn = MakeConn(1);
  auto* msg = MakeCollReqMsg(RAS_COLL_COMMS, root, 5, 0);
  struct rasSocket sock{};
  sock.conn = conn;
  EXPECT_EQ(ncclSuccess, rasMsgHandleCollReq(msg, &sock));
  // Falls through to rebroadcast, which creates a *second* rasCollective (the COMMS one).
  ASSERT_NE(nullptr, rasCollectivesHead);
  ASSERT_NE(nullptr, rasCollectivesHead->next);
  EXPECT_EQ(RAS_COLL_COMMS, rasCollectivesHead->next->type);
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollReq_LeafProcessSendsReadyResponseImmediately) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* conn = MakeConn(1);  // No rasNextLink/rasPrevLink conns => nFwdSent stays 0 => allDone.
  auto* msg = MakeCollReqMsg(RAS_COLL_CONNS, root, 5, 0);
  struct rasSocket sock{};
  sock.conn = conn;
  EXPECT_EQ(ncclSuccess, rasMsgHandleCollReq(msg, &sock));
  // rasCollReadyResp saw fromConn == conn (non-null) => sent a response and freed the collective.
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(conn, g_enqueuedMsgs[0].conn);
  EXPECT_EQ(nullptr, rasCollectivesHead);
  EXPECT_EQ(1, nRasCollHistory);
  FreeAllocatedMsg(msg);
}

// ===========================================================================
// rasCollReadyResp (indirect) / rasConnSendCollResp (indirect)
// ===========================================================================

TEST_F(RasCollectivesMicrotest, ReadyResp_RemotelyInitiatedSendsResponseRecordsHistoryAndFrees) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* fromConn = MakeConn(1);
  auto* coll = MakeCollective(RAS_COLL_CONNS, root, 5, fromConn);
  SetPeers(coll, {MakeAddr(1), MakeAddr(2)});
  SetData(coll, {'d', 'a', 't', 'a'});
  coll->nLegTimeouts = 3;
  // nFwdSent == nFwdRecv == 0 already, so triggering via rasCollsHandleTimeouts is simplest:
  // an expired collective with no outstanding legs completes immediately.
  coll->startTime = 0;
  coll->timeout = 5;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(100, &nextWakeup);
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(fromConn, g_enqueuedMsgs[0].conn);
  EXPECT_EQ(RAS_MSG_COLLRESP, g_enqueuedMsgs[0].msg->type);
  EXPECT_EQ(2, g_enqueuedMsgs[0].msg->collResp.nPeers);
  EXPECT_EQ(htons(1), g_enqueuedMsgs[0].msg->collResp.peers[0].sin.sin_port);
  EXPECT_EQ(htons(2), g_enqueuedMsgs[0].msg->collResp.peers[1].sin.sin_port);
  ASSERT_EQ(4, g_enqueuedMsgs[0].msg->collResp.nData);
  int dataOffset = rasMsgLength(RAS_MSG_COLLRESP) +
                   g_enqueuedMsgs[0].msg->collResp.nPeers * sizeof(*g_enqueuedMsgs[0].msg->collResp.peers);
  ALIGN_SIZE(dataOffset, alignof(int64_t));
  EXPECT_EQ(0, memcmp(reinterpret_cast<char*>(g_enqueuedMsgs[0].msg) + dataOffset, "data", 4));
  EXPECT_EQ(3, g_enqueuedMsgs[0].msg->collResp.nLegTimeouts);
  EXPECT_EQ(nullptr, rasCollectivesHead);  // Freed.
  EXPECT_EQ(1, nRasCollHistory);
}

TEST_F(RasCollectivesMicrotest, ReadyResp_LocallyInitiatedResumesClientWithoutFreeing) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* coll = MakeCollective(RAS_COLL_CONNS, root, 5, /*fromConn*/ nullptr);
  coll->startTime = 0;
  coll->timeout = 5;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(100, &nextWakeup);
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  EXPECT_EQ(1, g_clientResumeCalls);
  EXPECT_EQ(coll, g_lastClientResumeColl);
  EXPECT_EQ(coll, rasCollectivesHead);  // rasCollReadyResp itself never frees this path.
  rasCollFree(coll);                    // Simulate what the (untested-here) client_support.cc side eventually does.
}

// ===========================================================================
// rasMsgHandleCollResp
// ===========================================================================

TEST_F(RasCollectivesMicrotest, MsgHandleCollResp_UnknownCollectiveIsIgnored) {
  auto* msg = MakeCollRespMsg(MakeAddr(9), 5, 0, {}, {});
  EXPECT_EQ(ncclSuccess, DeliverResp(msg));
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollResp_NullConnReturnsInternalError) {
  union ncclSocketAddress root = MakeAddr(9);
  MakeCollective(RAS_COLL_CONNS, root, 5);
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, {});
  EXPECT_EQ(ncclInternalError, DeliverResp(msg));
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollResp_AccumulatesLegTimeoutsAndPeers) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* firstConn = MakeConn(1);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_CONNS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = firstConn;
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {MakeAddr(100)});
  auto* msg = MakeCollRespMsg(root, 5, /*nLegTimeouts*/ 2, {MakeAddr(200), MakeAddr(201)}, {});
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  EXPECT_EQ(2, coll->nLegTimeouts);
  EXPECT_EQ(firstConn, coll->fwdConns[0]);
  EXPECT_EQ(nullptr, coll->fwdConns[1]);  // Only the matching slot is cleared.
  EXPECT_EQ(1, coll->nFwdRecv);
  EXPECT_EQ(3, coll->nPeers);
  EXPECT_EQ(htons(200), coll->peers[1].sin.sin_port);
  EXPECT_EQ(htons(201), coll->peers[2].sin.sin_port);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollResp_AllReceivedTriggersReadyResp) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* fromConn = MakeConn(1);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_CONNS, root, 5, fromConn);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, {});
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  EXPECT_EQ(nullptr, rasCollectivesHead);  // rasCollReadyResp freed it.
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(fromConn, g_enqueuedMsgs[0].conn);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollResp_ConnsTypeDispatchesToConnsMerge) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_CONNS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetData(coll, BuildConnsData(100, 200, 300, 2, 2, {}));
  auto msgData = BuildConnsData(50, 500, 700, 3, 3, {});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgData);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* collData = reinterpret_cast<struct rasCollConns*>(coll->data);
  EXPECT_EQ(50, collData->travelTimeMin);
  EXPECT_EQ(500, collData->travelTimeMax);
  EXPECT_EQ(1000, collData->travelTimeSum);
  EXPECT_EQ(5, collData->travelTimeCount);
  EXPECT_EQ(5, collData->nConns);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollResp_ConnsMergeAppendsNegativeMins) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_CONNS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetData(coll, BuildConnsData(1, 2, 3, 1, 1, {{MakeAddr(10), MakeAddr(11)}}, -5));
  auto msgData = BuildConnsData(1, 2, 3, 1, 1, {{MakeAddr(1), MakeAddr(2)}}, -7);
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgData);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* collData = reinterpret_cast<struct rasCollConns*>(coll->data);
  ASSERT_EQ(2, collData->nNegativeMins);
  EXPECT_EQ(-5, collData->negativeMins[0].travelTimeMin);
  EXPECT_EQ(htons(10), collData->negativeMins[0].source.sin.sin_port);
  EXPECT_EQ(-7, collData->negativeMins[1].travelTimeMin);
  EXPECT_EQ(htons(1), collData->negativeMins[1].source.sin.sin_port);
}

TEST_F(RasCollectivesMicrotest, MsgHandleCollResp_DiagTypeDispatchesToDiagMerge) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_DIAG, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  std::vector<char> data(4, 'x');
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, data);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  EXPECT_EQ(1, g_collDiagMergeCalls);
  EXPECT_EQ(coll, g_lastDiagMergeColl);
}

// ===========================================================================
// rasCollsPurgeConn
// ===========================================================================

TEST_F(RasCollectivesMicrotest, PurgeConn_FreesCollectiveOriginatingFromThatConn) {
  auto* fromConn = MakeConn(1);
  MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn);
  rasCollsPurgeConn(fromConn);
  EXPECT_EQ(nullptr, rasCollectivesHead);
}

TEST_F(RasCollectivesMicrotest, PurgeConn_ClearsMatchingFwdConnAndIncrementsCounters) {
  auto* respConn = MakeConn(2);
  auto* other = MakeConn(3);
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  coll->fwdConns[coll->nFwdSent++] = other;
  rasCollsPurgeConn(respConn);
  EXPECT_EQ(nullptr, coll->fwdConns[0]);
  EXPECT_EQ(other, coll->fwdConns[1]);
  EXPECT_EQ(1, coll->nFwdRecv);
  EXPECT_EQ(1, coll->nLegTimeouts);
  EXPECT_NE(nullptr, rasCollectivesHead);  // Still one outstanding (other).
}

TEST_F(RasCollectivesMicrotest, PurgeConn_LastOutstandingTriggersReadyResp) {
  auto* fromConn = MakeConn(1);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  rasCollsPurgeConn(respConn);
  EXPECT_EQ(nullptr, rasCollectivesHead);  // Ready => freed.
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
}

TEST_F(RasCollectivesMicrotest, PurgeConn_UnrelatedCollectiveIsUntouched) {
  auto* fromConn = MakeConn(1);
  auto* unrelated = MakeConn(99);
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn);
  rasCollsPurgeConn(unrelated);
  EXPECT_EQ(coll, rasCollectivesHead);
}

// ===========================================================================
// rasCollsHandleTimeouts
// ===========================================================================

TEST_F(RasCollectivesMicrotest, HandleTimeouts_ZeroTimeoutIsNeverProcessed) {
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5);
  coll->timeout = 0;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(1'000'000, &nextWakeup);
  EXPECT_EQ(INT64_MAX, nextWakeup);
  EXPECT_EQ(coll, rasCollectivesHead);  // Untouched.
}

TEST_F(RasCollectivesMicrotest, HandleTimeouts_NotYetExpiredUpdatesNextWakeup) {
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5);
  coll->startTime = 100;
  coll->timeout = 50;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(120, &nextWakeup);
  EXPECT_EQ(150, nextWakeup);
}

TEST_F(RasCollectivesMicrotest, HandleTimeouts_StillHealthyConnectionIsNotDeclaredTimedOut) {
  auto* fromConn = MakeConn(1);
  auto* healthy = MakeConn(2);
  MakeSocketForConn(healthy, RAS_SOCK_READY, /*createTime*/ 0);  // Must be strictly before coll->startTime.
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn);
  coll->startTime = 100;
  coll->timeout = 10;
  coll->fwdConns[coll->nFwdSent++] = healthy;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(115, &nextWakeup);
  EXPECT_EQ(healthy, coll->fwdConns[0]);  // Not cleared: sock ready, created before startTime, no delays.
  EXPECT_EQ(0, coll->nFwdRecv);
}

TEST_F(RasCollectivesMicrotest, HandleTimeouts_DelayedConnectionIsDeclaredTimedOutAndReadyRespFires) {
  auto* fromConn = MakeConn(1);
  auto* delayed = MakeConn(2);
  MakeSocketForConn(delayed, RAS_SOCK_READY, 0);
  delayed->experiencingDelays = true;
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn);
  coll->startTime = 0;
  coll->timeout = 10;
  coll->fwdConns[coll->nFwdSent++] = delayed;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(50, &nextWakeup);
  EXPECT_EQ(nullptr, rasCollectivesHead);  // nFwdSent==nFwdRecv -> ready -> freed.
  EXPECT_EQ(1u, g_enqueuedMsgs.size());
}

TEST_F(RasCollectivesMicrotest, HandleTimeouts_SocketNotReadyIsDeclaredTimedOut) {
  auto* connecting = MakeConn(2);
  MakeSocketForConn(connecting, RAS_SOCK_CONNECTING, 0);
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5);
  coll->startTime = 0;
  coll->timeout = 10;
  coll->fwdConns[coll->nFwdSent++] = connecting;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(50, &nextWakeup);
  EXPECT_EQ(1, coll->nFwdRecv);
  EXPECT_EQ(1, coll->nLegTimeouts);
}

TEST_F(RasCollectivesMicrotest, HandleTimeouts_PartialTimeoutBelowExtraWaitsLonger) {
  auto* fromConn = MakeConn(1);
  auto* pending = MakeConn(2);
  auto* stillOut = MakeConn(3);
  MakeSocketForConn(pending, RAS_SOCK_CONNECTING, 0);
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn);
  coll->startTime = 5;
  coll->timeout = 10;
  coll->fwdConns[coll->nFwdSent++] = pending;   // Will be declared timed-out this round.
  coll->fwdConns[coll->nFwdSent++] = stillOut;  // No sock at all -> also times out immediately (nullptr sock).
  // Force nFwdRecv to stay below nFwdSent by adding one more untouched leg (createTime
  // strictly before coll->startTime, so it's recognized as healthy and left alone).
  auto* untouched = MakeConn(4);
  MakeSocketForConn(untouched, RAS_SOCK_READY, /*createTime*/ 0);
  coll->fwdConns[coll->nFwdSent++] = untouched;
  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_RAS);
  int64_t nextWakeup = INT64_MAX;
  const std::string logs = RcclUnitTesting::CaptureLog([&]() {
    rasCollsHandleTimeouts(19, &nextWakeup);  // now-start=14 > timeout=10, but < timeout+EXTRA(5)=15.
    EXPECT_EQ(coll, rasCollectivesHead);
    EXPECT_LE(nextWakeup, 20);
    EXPECT_TRUE(coll->timeoutWarned);
    const int nFwdRecv = coll->nFwdRecv;
    const int nLegTimeouts = coll->nLegTimeouts;
    nextWakeup = INT64_MAX;
    rasCollsHandleTimeouts(20, &nextWakeup);  // Exactly timeout+EXTRA: strict > must keep waiting.
    EXPECT_EQ(nFwdRecv, coll->nFwdRecv);
    EXPECT_EQ(nLegTimeouts, coll->nLegTimeouts);
  });
  EXPECT_EQ(coll, rasCollectivesHead);      // Not yet force-completed.
  EXPECT_LE(nextWakeup, 20);                // start(5)+timeout(10)+EXTRA(5).
  EXPECT_TRUE(coll->timeoutWarned);
  constexpr char warning[] = "timeout warning";
  ASSERT_TRUE(RcclUnitTesting::LogHas(logs, warning));
  EXPECT_EQ(logs.find(warning), logs.rfind(warning));
}

TEST_F(RasCollectivesMicrotest, HandleTimeouts_ExceedsExtraTimeoutForcesCompletion) {
  auto* fromConn = MakeConn(1);
  auto* untouched = MakeConn(2);
  // createTime strictly before coll->startTime -> the per-conn loop's "continue" keeps this
  // leg untouched (still legitimately outstanding), so completion can only come from the
  // "exceeded even the longer timeout" branch, not the ordinary all-declared-timeout path.
  MakeSocketForConn(untouched, RAS_SOCK_READY, /*createTime*/ 0);
  auto* coll = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn);
  coll->startTime = 100;
  coll->timeout = 10;
  coll->fwdConns[coll->nFwdSent++] = untouched;
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(115, &nextWakeup);  // Equal to timeout+EXTRA: strict > keeps waiting.
  EXPECT_EQ(coll, rasCollectivesHead);
  EXPECT_EQ(0, coll->nFwdRecv);
  rasCollsHandleTimeouts(116, &nextWakeup);  // now-start=16 > timeout(10)+EXTRA(5)=15.
  EXPECT_EQ(nullptr, rasCollectivesHead);    // Forced ready -> freed.
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(1, g_enqueuedMsgs[0].msg->collResp.nLegTimeouts);
}

TEST_F(RasCollectivesMicrotest, HandleTimeouts_SurvivesFreeingHeadWhileWalkingList) {
  auto* fromConn1 = MakeConn(1);
  auto* fromConn2 = MakeConn(2);
  auto* c1 = MakeCollective(RAS_COLL_CONNS, MakeAddr(9), 5, fromConn1);
  c1->startTime = 0;
  c1->timeout = 10;  // Will complete (no outstanding legs) and get freed mid-walk.
  auto* c2 = MakeCollective(RAS_COLL_CONNS, MakeAddr(10), 6, fromConn2);
  c2->startTime = 90;
  c2->timeout = 10;  // Not yet expired; should still be visited safely afterward.
  int64_t nextWakeup = INT64_MAX;
  rasCollsHandleTimeouts(100, &nextWakeup);
  EXPECT_EQ(c2, rasCollectivesHead);
  EXPECT_EQ(nullptr, c2->next);
}

// ===========================================================================
// RAS_COLL_CONNS: local statistics
// ===========================================================================

TEST_F(RasCollectivesMicrotest, ConnsInit_AggregatesTravelTimesAndNegativeMinimums) {
  auto* first = MakeConn(10);
  first->travelTimeMin = 8;
  first->travelTimeMax = 20;
  first->travelTimeSum = 42;
  first->travelTimeCount = 3;
  auto* second = MakeConn(20);
  second->travelTimeMin = -4;
  second->travelTimeMax = 30;
  second->travelTimeSum = 50;
  second->travelTimeCount = 2;
  auto* unused = MakeConn(30);
  unused->travelTimeMin = INT64_MAX;
  rasNetListeningSocket.addr = MakeAddr(99);

  struct rasCollRequest req{};
  req.type = RAS_COLL_CONNS;
  struct rasCollective* coll = nullptr;
  ASSERT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  ASSERT_NE(nullptr, coll);
  auto* data = reinterpret_cast<struct rasCollConns*>(coll->data);
  EXPECT_EQ(-4, data->travelTimeMin);
  EXPECT_EQ(30, data->travelTimeMax);
  EXPECT_EQ(92, data->travelTimeSum);
  EXPECT_EQ(5, data->travelTimeCount);
  EXPECT_EQ(2, data->nConns);
  ASSERT_EQ(1, data->nNegativeMins);
  EXPECT_EQ(htons(99), data->negativeMins[0].source.sin.sin_port);
  EXPECT_EQ(htons(20), data->negativeMins[0].dest.sin.sin_port);
  EXPECT_EQ(-4, data->negativeMins[0].travelTimeMin);
}

// ===========================================================================
// RAS_COLL_COMMS: init (via rasNetSendCollReq, real ncclComm objects)
// ===========================================================================

namespace {

struct FakeComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::vector<ncclPeerInfo> peerInfos;
  uint32_t abortFlagStorage = 0;

  FakeComm(uint64_t commHash, int rank, int nRanks) {
    comm->commHash = commHash;
    comm->rank = rank;
    comm->nRanks = nRanks;
    comm->peerInfoValid = true;
    comm->abortFlag = &abortFlagStorage;
    comm->proxyState = nullptr;
    comm->cudaDev = 3;
    comm->nvmlDev = 4;
    peerInfos.assign(nRanks, ncclPeerInfo{});
    for (int i = 0; i < nRanks; i++) {
      peerInfos[i].hostHash = commHash + 1;
      peerInfos[i].pidHash = commHash + 2;
    }
    comm->peerInfo = peerInfos.data();
  }
  ncclComm* get() { return comm.get(); }
};

void InstallNcclComms(std::vector<ncclComm*> comms) {
  nNcclComms = static_cast<int>(comms.size());
  ncclComms = static_cast<ncclComm**>(calloc(comms.size() ? comms.size() : 1, sizeof(*ncclComms)));
  for (size_t i = 0; i < comms.size(); i++) ncclComms[i] = comms[i];
  ncclCommsSorted = false;
}

}  // namespace

TEST_F(RasCollectivesMicrotest, CommsInit_SingleRankSingleCommCapturesFields) {
  FakeComm fc(0xAAAA, /*rank*/ 0, /*nRanks*/ 1);
  fc.get()->seqNumber[0] = 42;
  fc.get()->seqNumber[NCCL_NUM_FUNCTIONS - 1] = 84;
  fc.get()->initState = ncclInProgress;
  fc.get()->asyncResult = ncclUnhandledCudaError;
  fc.get()->finalizeCalled = true;
  fc.get()->destroyFlag = 1;
  fc.abortFlagStorage = 1;
  fc.get()->nvmlDev = 17;
  InstallNcclComms({fc.get()});
  auto* forward = MakeConn(10);
  MakeSocketForConn(forward, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, forward);

  struct rasCollRequest req{};
  req.type = RAS_COLL_COMMS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, data->nComms);
  EXPECT_EQ(0xAAAAu, data->comms[0].commId.commHash);
  EXPECT_EQ(1, data->comms[0].commNRanks);
  ASSERT_EQ(1, data->comms[0].nRanks);
  EXPECT_EQ(0, data->comms[0].ranks[0].commRank);
  EXPECT_EQ(0, data->comms[0].ranks[0].peerIdx);
  EXPECT_EQ(42u, data->comms[0].ranks[0].collOpCounts[0]);
  EXPECT_EQ(84u, data->comms[0].ranks[0].collOpCounts[NCCL_NUM_FUNCTIONS - 1]);
  EXPECT_EQ(ncclInProgress, data->comms[0].ranks[0].status.initState);
  EXPECT_EQ(ncclUnhandledCudaError, data->comms[0].ranks[0].status.asyncError);
  EXPECT_TRUE(data->comms[0].ranks[0].status.finalizeCalled);
  EXPECT_TRUE(data->comms[0].ranks[0].status.destroyFlag);
  EXPECT_TRUE(data->comms[0].ranks[0].status.abortFlag);
  EXPECT_EQ(3, data->comms[0].ranks[0].cudaDev);
  EXPECT_EQ(17, data->comms[0].ranks[0].nvmlDev);
  EXPECT_EQ(0, data->comms[0].nMissingRanks);
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  ASSERT_EQ(RAS_MSG_COLLREQ, g_enqueuedMsgs[0].msg->type);
  const auto& forwarded = g_enqueuedMsgs[0].msg->collReq;
  ASSERT_EQ(1, forwarded.comms.nSkipMissingRanksComms);
  EXPECT_EQ(0xAAAAu, forwarded.comms.skipMissingRanksComms[0].commHash);
  EXPECT_EQ(fc.peerInfos[0].hostHash, forwarded.comms.skipMissingRanksComms[0].hostHash);
  EXPECT_EQ(fc.peerInfos[0].pidHash, forwarded.comms.skipMissingRanksComms[0].pidHash);
}

TEST_F(RasCollectivesMicrotest, CommsInit_MultipleGpusSameProcessGroupIntoOneComm) {
  FakeComm fc0(0xBBBB, 0, 2);
  FakeComm fc1(0xBBBB, 1, 2);
  InstallNcclComms({fc0.get(), fc1.get()});

  struct rasCollRequest req{};
  req.type = RAS_COLL_COMMS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, data->nComms);
  EXPECT_EQ(2, data->comms[0].nRanks);
  EXPECT_EQ(0, data->comms[0].ranks[0].commRank);
  EXPECT_EQ(1, data->comms[0].ranks[1].commRank);
}

TEST_F(RasCollectivesMicrotest, CommsInit_MultipleCommunicatorsAdvanceVariableRecordsAndSortRequest) {
  FakeComm high(0x3000, 0, 2);
  FakeComm low(0x1000, 0, 1);
  InstallNcclComms({high.get(), low.get()});
  auto* forward = MakeConn(10);
  MakeSocketForConn(forward, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, forward);

  const size_t reqLen = rasCollDataLength(RAS_COLL_COMMS) + sizeof(struct rasCommId);
  std::vector<char> requestStorage(reqLen, 0);
  auto* req = reinterpret_cast<struct rasCollRequest*>(requestStorage.data());
  req->type = RAS_COLL_COMMS;
  req->comms.nSkipMissingRanksComms = 1;
  req->comms.skipMissingRanksComms[0].commHash = 0xF000;
  req->comms.skipMissingRanksComms[0].hostHash = 0xF001;
  req->comms.skipMissingRanksComms[0].pidHash = 0xF002;

  struct rasCollective* coll = nullptr;
  ASSERT_EQ(ncclSuccess, rasNetSendCollReq(req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(2, data->nComms);
  EXPECT_EQ(0x1000u, data->comms[0].commId.commHash);
  auto* second = NextComm(data->comms);
  EXPECT_EQ(0x3000u, second->commId.commHash);
  ASSERT_EQ(1, second->nMissingRanks);
  auto* missing = reinterpret_cast<struct rasCollCommsMissingRank*>(second->ranks + second->nRanks);
  EXPECT_EQ(1, missing[0].commRank);

  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  const auto& forwarded = g_enqueuedMsgs[0].msg->collReq;
  ASSERT_EQ(3, forwarded.comms.nSkipMissingRanksComms);
  EXPECT_EQ(0x1000u, forwarded.comms.skipMissingRanksComms[0].commHash);
  EXPECT_EQ(0x3000u, forwarded.comms.skipMissingRanksComms[1].commHash);
  EXPECT_EQ(0xF000u, forwarded.comms.skipMissingRanksComms[2].commHash);
}

TEST_F(RasCollectivesMicrotest, CommsInit_InvalidPeerInfoIsIgnored) {
  FakeComm fc(0xCCCC, 0, 1);
  fc.get()->peerInfoValid = false;
  InstallNcclComms({fc.get()});

  struct rasCollRequest req{};
  req.type = RAS_COLL_COMMS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  EXPECT_EQ(0, data->nComms);
}

TEST_F(RasCollectivesMicrotest, CommsInit_NullEntriesSortToTheEnd) {
  FakeComm fc(0xCCCD, 0, 1);
  InstallNcclComms({nullptr, fc.get()});

  struct rasCollRequest req{};
  req.type = RAS_COLL_COMMS;
  struct rasCollective* coll = nullptr;
  ASSERT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, data->nComms);
  EXPECT_EQ(0xCCCDu, data->comms[0].commId.commHash);
  EXPECT_EQ(nullptr, ncclComms[1]);
}

TEST_F(RasCollectivesMicrotest, CommsInit_UsesProxyErrorWhenCommIsOtherwiseHealthy) {
  FakeComm fc(0xCCCE, 0, 1);
  struct ncclProxyState proxyState{};
  proxyState.asyncResult = ncclSystemError;
  fc.get()->asyncResult = ncclSuccess;
  fc.get()->proxyState = &proxyState;
  InstallNcclComms({fc.get()});

  struct rasCollRequest req{};
  req.type = RAS_COLL_COMMS;
  struct rasCollective* coll = nullptr;
  ASSERT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, data->nComms);
  ASSERT_EQ(1, data->comms[0].nRanks);
  EXPECT_EQ(ncclSystemError, data->comms[0].ranks[0].status.asyncError);
}

TEST_F(RasCollectivesMicrotest, CommsInit_MissingRankFillsAddrFromRasPeersLookup) {
  // A 2-rank communicator where only rank 0 (this process) is locally known;
  // rank 1's identity must be looked up in rasPeers via the commHash-subtracted hash.
  FakeComm fc(0xDDDD, /*rank*/ 0, /*nRanks*/ 2);
  // ncclComm->peerInfo is indexed by commRank across the whole communicator; rank 1 needs its own entry.
  fc.peerInfos[1].hostHash = 0xDDDD + 1;
  fc.peerInfos[1].pidHash = 0xDDDD + 2;
  fc.peerInfos[1].cudaDev = 7;
  fc.peerInfos[1].nvmlDev = 8;
  InstallNcclComms({fc.get()});

  // Two entries so the peersReSorted qsort actually invokes peersHashesCompare (a
  // single-element array never calls the comparator).
  nRasPeers = 2;
  rasPeers = static_cast<struct rasPeerInfo*>(calloc(2, sizeof(*rasPeers)));
  rasPeers[0].addr = MakeAddr(600);
  rasPeers[0].hostHash = 0xAAAA;
  rasPeers[0].pidHash = 0xBBBB;
  rasPeers[1].addr = MakeAddr(500);
  // Production stores hash - commHash (communicator-independent); mirror that here.
  rasPeers[1].hostHash = (0xDDDD + 1) - 0xDDDD;
  rasPeers[1].pidHash = (0xDDDD + 2) - 0xDDDD;

  struct rasCollRequest req{};
  req.type = RAS_COLL_COMMS;
  req.rootAddr = MakeAddr(9);
  req.rootId = 5;
  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(&req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, data->nComms);
  ASSERT_EQ(1, data->comms[0].nMissingRanks);
  auto* missing = reinterpret_cast<struct rasCollCommsMissingRank*>(data->comms[0].ranks + data->comms[0].nRanks);
  EXPECT_EQ(1, missing[0].commRank);
  EXPECT_EQ(htons(500), missing[0].addr.sin.sin_port);
  EXPECT_EQ(7, missing[0].cudaDev);
  EXPECT_EQ(8, missing[0].nvmlDev);
}

TEST_F(RasCollectivesMicrotest, CommsInit_SkipMissingRanksCommFilterSuppressesMissingRankFill) {
  FakeComm fc(0xEEEE, 0, 2);
  InstallNcclComms({fc.get()});

  // Pre-populate the request's skip-list with this exact commId, mirroring a peer that
  // already told us it doesn't need missingRanks data for this communicator.
  size_t reqLen = rasCollDataLength(RAS_COLL_COMMS) + sizeof(struct rasCommId);
  std::vector<char> buf(reqLen, 0);
  auto* req = reinterpret_cast<struct rasCollRequest*>(buf.data());
  req->type = RAS_COLL_COMMS;
  req->rootAddr = MakeAddr(9);
  req->rootId = 5;
  req->comms.nSkipMissingRanksComms = 1;
  req->comms.skipMissingRanksComms[0].commHash = 0xEEEE;
  req->comms.skipMissingRanksComms[0].hostHash = fc.peerInfos[0].hostHash;
  req->comms.skipMissingRanksComms[0].pidHash = fc.peerInfos[0].pidHash;

  struct rasCollective* coll = nullptr;
  EXPECT_EQ(ncclSuccess, rasNetSendCollReq(req, nullptr, &coll));
  auto* data = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, data->nComms);
  EXPECT_EQ(0, data->comms[0].nMissingRanks);
}

// ===========================================================================
// RAS_COLL_COMMS: merge (via rasMsgHandleCollResp, hand-built blobs)
// ===========================================================================

TEST_F(RasCollectivesMicrotest, CommsMerge_EmptyMessageIsNoOp) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  auto collBlob = BuildRasCollComms({});
  SetData(coll, collBlob);
  auto msgBlob = BuildRasCollComms({});  // nComms == 0.
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  EXPECT_EQ(0, reinterpret_cast<struct rasCollComms*>(coll->data)->nComms);
}

TEST_F(RasCollectivesMicrotest, CommsMerge_DisjointCommsAreUnioned) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {MakeAddr(1)});
  auto collBlob = BuildRasCollComms({MakeCommSpec(0x1000, 2, {{0, 0}})});
  SetData(coll, collBlob);
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x2000, 2, {{0, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(2, merged->nComms);
  EXPECT_EQ(0x1000u, merged->comms[0].commId.commHash);
  EXPECT_EQ(0x2000u, NextComm(merged->comms)->commId.commHash);
}

TEST_F(RasCollectivesMicrotest, CommsMerge_SameCommMergesRanksAndShiftsMsgSidePeerIdx) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {MakeAddr(1), MakeAddr(2)});  // coll->nPeers == 2 at merge time.
  auto collBlob = BuildRasCollComms({MakeCommSpec(0x3000, 4, {{0, 0}})});
  SetData(coll, collBlob);
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x3000, 4, {{1, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {MakeAddr(200)}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, merged->nComms);
  ASSERT_EQ(2, merged->comms[0].nRanks);
  EXPECT_EQ(0, merged->comms[0].ranks[0].commRank);
  EXPECT_EQ(0, merged->comms[0].ranks[0].peerIdx);   // Coll-side rank: unshifted.
  EXPECT_EQ(1, merged->comms[0].ranks[1].commRank);
  EXPECT_EQ(2, merged->comms[0].ranks[1].peerIdx);   // Msg-side rank: shifted by coll->nPeers (2).
  ASSERT_EQ(3, coll->nPeers);
  EXPECT_EQ(htons(200), coll->peers[2].sin.sin_port);
}

TEST_F(RasCollectivesMicrotest, CommsMerge_AllocationFailurePreservesExistingPayload) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  auto collBlob = BuildRasCollComms({MakeCommSpec(0x3100, 1, {{0, 0}})});
  SetData(coll, collBlob);
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x3200, 1, {{0, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  char* originalData = coll->data;

  g_failAllocationCall = 1;
  EXPECT_EQ(ncclSystemError, rasCollCommsMerge(coll, msg));
  EXPECT_EQ(originalData, coll->data);
  EXPECT_EQ(static_cast<int>(collBlob.size()), coll->nData);
  EXPECT_EQ(0x3100u, reinterpret_cast<struct rasCollComms*>(coll->data)->comms[0].commId.commHash);
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, CommsMerge_MissingRankEntryClearedWhenFilledByPeer) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {MakeAddr(1)});

  CommSpec collSpec = MakeCommSpec(0x4000, /*commNRanks*/ 2, {{0, 0}});
  collSpec.missingRanks = {1};
  collSpec.missingAddrs = {MakeAddr(50)};
  auto collBlob = BuildRasCollComms({collSpec});
  SetData(coll, collBlob);

  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x4000, 2, {{1, 0}})});  // Peer supplies the missing rank.
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, merged->nComms);
  EXPECT_EQ(2, merged->comms[0].nRanks);
  EXPECT_EQ(0, merged->comms[0].nMissingRanks);  // The hole was filled, so no missingRanks remain.
}

TEST_F(RasCollectivesMicrotest, CommsMerge_HashCollisionSizeMismatchOrdersBySize) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {});
  auto collBlob = BuildRasCollComms({MakeCommSpec(0x5000, /*commNRanks*/ 2, {{0, 0}})});
  SetData(coll, collBlob);
  // Same commId but a different commNRanks -- a "hash collision"; production keeps both
  // rather than crashing, ordered by size.
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x5000, /*commNRanks*/ 4, {{0, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  EXPECT_EQ(2, merged->nComms);  // Kept as two separate entries rather than merged.
}

TEST_F(RasCollectivesMicrotest, CommsMerge_CommOnlyInMsgIsCopiedWithShiftedPeerIdx) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {MakeAddr(1), MakeAddr(2), MakeAddr(3)});  // nPeers == 3.
  auto collBlob = BuildRasCollComms({});  // Nothing local yet.
  SetData(coll, collBlob);
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x6000, 1, {{0, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, merged->nComms);
  EXPECT_EQ(3, merged->comms[0].ranks[0].peerIdx);  // Shifted by coll->nPeers.
}

TEST_F(RasCollectivesMicrotest, CommsMerge_PartiallyFilledMissingRanksAreCarriedForward) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {MakeAddr(1)});

  // commNRanks=3: rank 0 known locally, ranks 1 and 2 are both missing.
  CommSpec collSpec = MakeCommSpec(0x7000, /*commNRanks*/ 3, {{0, 0}});
  collSpec.missingRanks = {1, 2};
  collSpec.missingAddrs = {MakeAddr(51), MakeAddr(52)};
  auto collBlob = BuildRasCollComms({collSpec});
  SetData(coll, collBlob);

  // The peer only supplies rank 1, leaving rank 2 still missing.
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x7000, 3, {{1, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, merged->nComms);
  EXPECT_EQ(2, merged->comms[0].nRanks);
  ASSERT_EQ(1, merged->comms[0].nMissingRanks);  // Rank 2's entry survives, carried forward.
  auto* missing = reinterpret_cast<struct rasCollCommsMissingRank*>(merged->comms[0].ranks + merged->comms[0].nRanks);
  EXPECT_EQ(2, missing[0].commRank);
}

TEST_F(RasCollectivesMicrotest, CommsMerge_DuplicateRankIsLoggedAndSkipped) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {});
  // Both sides mistakenly report data for the same commRank (0) -- a "should never
  // happen" overlap that production logs and skips rather than crashing on.
  auto collBlob = BuildRasCollComms({MakeCommSpec(0x8000, 2, {{0, 3}})});
  SetData(coll, collBlob);
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x8000, 2, {{0, 7}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, merged->nComms);
  // Only one rank struct is actually written (the msg-side duplicate is skipped, not
  // appended) -- but newComm->nRanks is set to collComm->nRanks + msgComm->nRanks up front
  // and is never corrected back down when a collision shortens the real merged count. This
  // is current (not fixed here) production behavior for a path its own comments already
  // flag as "should never happen" / "possible hash collision".
  EXPECT_EQ(2, merged->comms[0].nRanks);
  EXPECT_EQ(3, merged->comms[0].ranks[0].peerIdx);
  EXPECT_EQ(0, merged->comms[0].ranks[1].peerIdx);
}

TEST_F(RasCollectivesMicrotest, CommsMerge_RankCountOverflowClampsToCommNRanks) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* respConn = MakeConn(2);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  coll->fwdConns[coll->nFwdSent++] = respConn;
  SetPeers(coll, {});
  // commNRanks=2 but the two sides together report 3 ranks -- more than the
  // communicator's declared size (a hash-collision-adjacent "should never happen").
  auto collBlob = BuildRasCollComms({MakeCommSpec(0x9000, /*commNRanks*/ 2, {{0, 0}, {1, 0}})});
  SetData(coll, collBlob);
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0x9000, 2, {{2, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
  EXPECT_EQ(ncclSuccess, DeliverResp(msg, respConn));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, merged->nComms);
  EXPECT_EQ(2, merged->comms[0].nRanks);  // Clamped to commNRanks; the extra rank was skipped.
}

TEST_F(RasCollectivesMicrotest, CommsMerge_MalformedBuffersStopAtTheirBounds) {
  union ncclSocketAddress root = MakeAddr(9);
  constexpr int headerSize = static_cast<int>(offsetof(struct rasCollComms, comms));

  auto runMerge = [&](std::vector<char> collBlob, int collDataSize, std::vector<char> msgBlob, int msgDataSize) {
    auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
    SetData(coll, collBlob);
    coll->nData = collDataSize;
    auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);
    msg->collResp.nData = msgDataSize;
    EXPECT_EQ(ncclSuccess, rasCollCommsMerge(coll, msg));
    ASSERT_NE(coll->data, nullptr);
    EXPECT_EQ(0, reinterpret_cast<struct rasCollComms*>(coll->data)->nComms);
    FreeAllocatedMsg(msg);
    rasCollFree(coll);
  };

  auto empty = BuildRasCollComms({});
  auto malformed = BuildRasCollComms({MakeCommSpec(0xA000, 1)});
  auto valid = BuildRasCollComms({MakeCommSpec(0xA100, 1)});

  runMerge(empty, static_cast<int>(empty.size()), malformed, headerSize);
  runMerge(malformed, headerSize, valid, static_cast<int>(valid.size()));
  runMerge(valid, static_cast<int>(valid.size()), malformed, headerSize);
}

TEST_F(RasCollectivesMicrotest, CommsMerge_MissingRankMismatchKeepsAvailableMetadata) {
  union ncclSocketAddress root = MakeAddr(9);
  auto* coll = MakeCollective(RAS_COLL_COMMS, root, 5);
  SetPeers(coll, {MakeAddr(1)});

  CommSpec collSpec = MakeCommSpec(0xA100, 4, {{0, 0}});
  collSpec.missingRanks = {2};
  collSpec.missingAddrs = {MakeAddr(52)};
  SetData(coll, BuildRasCollComms({collSpec}));
  auto msgBlob = BuildRasCollComms({MakeCommSpec(0xA100, 4, {{1, 0}})});
  auto* msg = MakeCollRespMsg(root, 5, 0, {}, msgBlob);

  ASSERT_EQ(ncclSuccess, rasCollCommsMerge(coll, msg));
  auto* merged = reinterpret_cast<struct rasCollComms*>(coll->data);
  ASSERT_EQ(1, merged->nComms);
  EXPECT_EQ(2, merged->comms[0].nRanks);
  EXPECT_EQ(1, merged->comms[0].nMissingRanks);
  FreeAllocatedMsg(msg);
}

TEST_F(RasCollectivesMicrotest, ComparisonHelpersCoverAllOrderingKeys) {
  FakeComm lowComm(1, 0, 1);
  FakeComm highComm(2, 0, 1);
  FakeComm higherRank(1, 1, 2);
  ncclComm* nullComm = nullptr;
  ncclComm* low = lowComm.get();
  ncclComm* high = highComm.get();
  ncclComm* rank = higherRank.get();
  EXPECT_LT(ncclCommsCompare(&low, &high), 0);
  EXPECT_GT(ncclCommsCompare(&high, &low), 0);
  EXPECT_LT(ncclCommsCompare(&low, &rank), 0);
  EXPECT_GT(ncclCommsCompare(&rank, &low), 0);
  EXPECT_EQ(0, ncclCommsCompare(&low, &low));
  EXPECT_LT(ncclCommsCompare(&low, &nullComm), 0);
  EXPECT_GT(ncclCommsCompare(&nullComm, &low), 0);
  EXPECT_EQ(0, ncclCommsCompare(&nullComm, &nullComm));

  struct rasPeerInfo lowPeer{};
  struct rasPeerInfo highPeer{};
  struct rasPeerInfo higherPid{};
  lowPeer.hostHash = 1;
  lowPeer.pidHash = 2;
  highPeer.hostHash = 2;
  highPeer.pidHash = 1;
  higherPid.hostHash = 1;
  higherPid.pidHash = 3;
  struct rasPeerInfo* lowPeerPtr = &lowPeer;
  struct rasPeerInfo* highPeerPtr = &highPeer;
  struct rasPeerInfo* higherPidPtr = &higherPid;
  EXPECT_LT(peersHashesCompare(&lowPeerPtr, &highPeerPtr), 0);
  EXPECT_GT(peersHashesCompare(&highPeerPtr, &lowPeerPtr), 0);
  EXPECT_LT(peersHashesCompare(&lowPeerPtr, &higherPidPtr), 0);
  EXPECT_GT(peersHashesCompare(&higherPidPtr, &lowPeerPtr), 0);
  EXPECT_EQ(0, peersHashesCompare(&lowPeerPtr, &lowPeerPtr));

  uint64_t key[2] = {1, 2};
  EXPECT_EQ(0, peersHashesSearch(key, &lowPeerPtr));
  key[1] = 1;
  EXPECT_LT(peersHashesSearch(key, &lowPeerPtr), 0);
  key[1] = 3;
  EXPECT_GT(peersHashesSearch(key, &lowPeerPtr), 0);
  key[0] = 0;
  EXPECT_LT(peersHashesSearch(key, &lowPeerPtr), 0);
  key[0] = 2;
  EXPECT_GT(peersHashesSearch(key, &lowPeerPtr), 0);

  struct rasCommId lowId{1, 2, 3};
  struct rasCommId highHash{2, 1, 1};
  struct rasCommId highHost{1, 3, 1};
  struct rasCommId highPid{1, 2, 4};
  EXPECT_LT(rasCommIdCompare(&lowId, &highHash), 0);
  EXPECT_GT(rasCommIdCompare(&highHash, &lowId), 0);
  EXPECT_LT(rasCommIdCompare(&lowId, &highHost), 0);
  EXPECT_GT(rasCommIdCompare(&highHost, &lowId), 0);
  EXPECT_LT(rasCommIdCompare(&lowId, &highPid), 0);
  EXPECT_GT(rasCommIdCompare(&highPid, &lowId), 0);
  EXPECT_EQ(0, rasCommIdCompare(&lowId, &lowId));

  struct rasCollCommsMissingRank missing{};
  missing.commRank = 4;
  int missingKey = 3;
  EXPECT_LT(rasCollCommsMissingRankSearch(&missingKey, &missing), 0);
  missingKey = 4;
  EXPECT_EQ(0, rasCollCommsMissingRankSearch(&missingKey, &missing));
  missingKey = 5;
  EXPECT_GT(rasCollCommsMissingRankSearch(&missingKey, &missing), 0);
}
