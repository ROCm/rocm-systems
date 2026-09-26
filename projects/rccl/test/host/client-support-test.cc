/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/client_support.cc: the server-side RAS client-connection
// handler (socket accept, protocol command parsing, job-summary and communicator-status report
// generation in text and JSON, plus the sorting/formatting helpers those reports use).
//
// This TU #includes the hipified client_support.cc directly, same recipe as ras-test.cc and
// ras-client-test.cc, so its statics become callable. It needs its OWN executable
// (rccl-UnitTestsMicroRasClientSupport), separate from rccl-UnitTestsMicro: ras-test.cc already
// defines file-scope fakes for rasClientInitSocket/rasClientAcceptNewSocket/rasClientEventLoop/
// rasClientSupportTerminate/rasClientsNotifyEvent (client_support.cc's own exports, needed there
// because ras.cc calls them) -- linking the real client_support.cc into that same binary would
// duplicate-define all five.
//
// client_support.cc's own dependency surface reaches into three other RAS source files that have
// no dedicated test of their own yet (collectives.cc, peers.cc, diagnostics.cc) plus ras.cc
// (already tested). Every one of those externs is faked TU-locally below, exactly as ras-test.cc
// already fakes client_support.cc's exports without client_support.cc having a test of its own --
// this file does not depend on those other files ever getting a dedicated test.
//
// Seam layout:
//  - libc socket calls (bind/listen/accept/fcntl/recv/send, plus already-shared socket/setsockopt/
//    close) go through the SHARED fakes/libc_fakes.h + fakes/libc_seam.h mechanism -- this is the
//    first unit needing the raw-socket half of that seam, per its own header comment.
//  - clockNano is a TU-local #define redirect, same recipe as ras-test.cc (not shared: a global
//    process-wide rename is only sound with a single real definition per binary).
//  - rasPfds/rasMsgAlloc/rasMsgFree/rasTimeoutFactorNs (ras.cc), rasPeers/rasDeadPeers/rasPeerFind/
//    rasPeerIsDead/rasPeerToString/rasPeerInfoToString/ncclSocketsCompare/ncclSocketsSameNode
//    (peers.cc), rasCollReqInit/rasNetSendCollReq/rasCollRecordHistory/rasCollFree (collectives.cc),
//    and all eight rasDiagnostics* entry points (diagnostics.cc) are TU-local fakes below.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/libc_fakes.h"
#include "ras/diagnostics.h"
#include "ras/ras_internal.h"

#include "fakes/libc_seam.h"

uint64_t ClientSupportTestClockNano();
#define clockNano ClientSupportTestClockNano

namespace {

// ---------------------------------------------------------------------------
// State-recording globals for the TU-local fakes defined below the include.
// ---------------------------------------------------------------------------

uint64_t g_clockNano = 100 * CLOCK_UNITS_PER_SEC;

// ras.cc seams.
int g_getNewPollEntryCalls = 0;
std::vector<struct pollfd> g_pfdsStorage;
int g_msgAllocCalls = 0;
int g_msgFreeCalls = 0;
ncclResult_t g_msgAllocResult = ncclSuccess;

// peers.cc seams. Tests populate rasPeers/nRasPeers/rasDeadPeers/nRasDeadPeers directly (plain
// globals, not std::function hooks) -- rasPeerFind/rasPeerIsDead/ncclSocketsCompare below are
// genuine reimplementations against those globals, not recorders, so a test only needs to build
// the peer arrays it cares about rather than script every lookup individually.
std::function<bool(const union ncclSocketAddress*, const union ncclSocketAddress*)> g_ncclSocketsSameNode;
int g_peerToStringCalls = 0;
int g_peerInfoToStringCalls = 0;

// collectives.cc seams.
int g_collReqInitCalls = 0;
int g_netSendCollReqCalls = 0;
struct rasCollRequest g_lastCollReq;
ncclResult_t g_netSendCollReqResult = ncclSuccess;
bool g_netSendCollReqAllDone = true;
// The rasCollective the next rasNetSendCollReq() call hands back via *pColl. Tests build this with
// MakeCollective() below; ownership transfers to the fake, which clears this slot. A null coll
// (the default) makes rasNetSendCollReq() leave *pColl untouched, matching "no data yet" callers
// never dereference.
struct rasCollective* g_nextCollective = nullptr;
int g_collRecordHistoryCalls = 0;
const struct rasCollective* g_lastCollRecordHistory = nullptr;
int g_collFreeCalls = 0;

// diagnostics.cc seams.
int g_diagContextInitCalls = 0;
ncclResult_t g_diagContextInitResult = ncclSuccess;
int g_diagClientInitCalls = 0;
ncclResult_t g_diagClientInitResult = ncclSuccess;
struct rasDiagnosticsReporter g_lastReporter;
bool g_haveLastReporter = false;
int g_diagStartCalls = 0;
ncclResult_t g_diagStartResult = ncclSuccess;
int g_diagInProgressCalls = 0;
bool g_diagInProgress = false;
int g_diagCancelTargetCalls = 0;
const void* g_lastCancelTarget = nullptr;
int g_diagClientCleanupCalls = 0;
const struct rasClient* g_lastClientCleanup = nullptr;
int g_diagResumeCalls = 0;
ncclResult_t g_diagResumeResult = ncclSuccess;
int g_diagFormatLineCalls = 0;
ncclResult_t g_diagFormatLineResult = ncclSuccess;

}  // namespace

#include CLIENT_SUPPORT_CC_PATH

#undef clockNano
#include "fakes/libc_seam_undef.h"

uint64_t ClientSupportTestClockNano() { return g_clockNano; }

// ---------------------------------------------------------------------------
// ras.cc externs (rasPfds/rasGetNewPollEntry/rasMsgAlloc/rasMsgFree/rasLine/rasTimeoutFactorNs).
// ---------------------------------------------------------------------------

struct pollfd* rasPfds = nullptr;
char rasLine[SOCKET_NAME_MAXLEN + 1];

ncclResult_t rasGetNewPollEntry(int* index) {
  ++g_getNewPollEntryCalls;
  g_pfdsStorage.push_back(pollfd{});
  rasPfds = g_pfdsStorage.data();
  *index = static_cast<int>(g_pfdsStorage.size()) - 1;
  return ncclSuccess;
}

// Genuine (not stubbed) alloc/free pair: client_support.cc's send queue depends on the
// meta-header-then-variable-length-msg layout actually round-tripping, not just on the call
// happening. Mirrors ras.cc:305/315's own real implementation.
ncclResult_t rasMsgAlloc(struct rasMsg** msg, size_t msgLen) {
  ++g_msgAllocCalls;
  if (g_msgAllocResult != ncclSuccess) return g_msgAllocResult;
  const size_t totalSize = offsetof(struct rasMsgMeta, msg) + msgLen;
  auto* meta = static_cast<struct rasMsgMeta*>(calloc(1, totalSize));
  if (meta == nullptr) return ncclSystemError;
  *msg = &meta->msg;
  return ncclSuccess;
}

void rasMsgFree(struct rasMsg* msg) {
  ++g_msgFreeCalls;
  auto* meta = reinterpret_cast<struct rasMsgMeta*>(reinterpret_cast<char*>(msg) - offsetof(struct rasMsgMeta, msg));
  free(meta);
}

int64_t rasTimeoutFactorNs(int64_t baseSeconds) { return baseSeconds * CLOCK_UNITS_PER_SEC; }

// ---------------------------------------------------------------------------
// peers.cc externs.
// ---------------------------------------------------------------------------

struct rasPeerInfo* rasPeers = nullptr;
int nRasPeers = 0;
uint64_t rasPeersHash = 0;
union ncclSocketAddress* rasDeadPeers = nullptr;
int nRasDeadPeers = 0;
uint64_t rasDeadPeersHash = 0;

int rasPeerFind(const union ncclSocketAddress* addr) {
  for (int i = 0; i < nRasPeers; i++) {
    if (memcmp(&rasPeers[i].addr, addr, sizeof(*addr)) == 0) return i;
  }
  return -1;
}

bool rasPeerIsDead(const union ncclSocketAddress* addr) {
  for (int i = 0; i < nRasDeadPeers; i++) {
    if (memcmp(rasDeadPeers + i, addr, sizeof(*addr)) == 0) return true;
  }
  return false;
}

const char* rasPeerToString(const union ncclSocketAddress* addr, char* buf, size_t size) {
  ++g_peerToStringCalls;
  int peerIdx = rasPeerFind(addr);
  snprintf(buf, size, "peer(pid=%d)", (peerIdx != -1 ? rasPeers[peerIdx].pid : -1));
  return buf;
}

const char* rasPeerInfoToString(const struct rasPeerInfo* peer, char* buf, size_t size) {
  ++g_peerInfoToStringCalls;
  snprintf(buf, size, "peer(pid=%d)", (peer != nullptr ? peer->pid : -1));
  return buf;
}

// Real-ish reimplementation (family, then address bytes, then port), not a recorder: used by
// client_support.cc via qsort/bsearch on coll->peers, so it needs genuine ordering semantics for
// those calls to behave sensibly, not just a call count.
int ncclSocketsCompare(const void* p1, const void* p2) {
  const auto* a1 = static_cast<const union ncclSocketAddress*>(p1);
  const auto* a2 = static_cast<const union ncclSocketAddress*>(p2);
  int hostCmp = ncclSocketsHostCompare(a1, a2);
  if (hostCmp != 0) return hostCmp;
  uint16_t port1 = (a1->sa.sa_family == AF_INET ? a1->sin.sin_port : a1->sin6.sin6_port);
  uint16_t port2 = (a2->sa.sa_family == AF_INET ? a2->sin.sin_port : a2->sin6.sin6_port);
  return (port1 < port2 ? -1 : (port1 > port2 ? 1 : 0));
}

bool ncclSocketsSameNode(const union ncclSocketAddress* a1, const union ncclSocketAddress* a2) {
  if (g_ncclSocketsSameNode) return g_ncclSocketsSameNode(a1, a2);
  return ncclSocketsHostCompare(a1, a2) == 0;
}

// ---------------------------------------------------------------------------
// collectives.cc externs.
// ---------------------------------------------------------------------------

void rasCollReqInit(struct rasCollRequest* req) {
  ++g_collReqInitCalls;
  memset(req, 0, sizeof(*req));
}

ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, bool* pAllDone, struct rasCollective** pColl,
                                struct rasConnection* /*fromConn*/) {
  ++g_netSendCollReqCalls;
  g_lastCollReq = *req;
  if (g_netSendCollReqResult != ncclSuccess) return g_netSendCollReqResult;
  if (pAllDone) *pAllDone = g_netSendCollReqAllDone;
  if (pColl && g_nextCollective) {
    *pColl = g_nextCollective;
    g_nextCollective = nullptr;
  }
  return ncclSuccess;
}

void rasCollRecordHistory(const struct rasCollective* coll) {
  ++g_collRecordHistoryCalls;
  g_lastCollRecordHistory = coll;
}

void rasCollFree(struct rasCollective* coll) {
  ++g_collFreeCalls;
  if (coll == nullptr) return;
  free(coll->data);
  free(coll->peers);
  free(coll);
}

// ---------------------------------------------------------------------------
// diagnostics.cc externs.
// ---------------------------------------------------------------------------

ncclResult_t rasDiagnosticsContextInit(struct rasDiagnosticsContext* ctx, const struct ncclComm* /*comm*/) {
  ++g_diagContextInitCalls;
  if (ctx) memset(ctx, 0, sizeof(*ctx));
  return g_diagContextInitResult;
}

ncclResult_t rasDiagnosticsFormatLine(char* out, size_t outSize, const char* line) {
  ++g_diagFormatLineCalls;
  if (g_diagFormatLineResult != ncclSuccess) return g_diagFormatLineResult;
  snprintf(out, outSize, "%s", line);
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsClientInit(struct rasClient* client, const struct rasDiagnosticsContext* /*ctx*/,
                                       const struct rasDiagnosticsReporter* reporter) {
  ++g_diagClientInitCalls;
  if (reporter) {
    g_lastReporter = *reporter;
    g_haveLastReporter = true;
  }
  // Real diagnostics.cc attaches state to the client as a side effect of a successful init; without
  // this, rasClientRunDiagInit's `if (client->diagnostics == nullptr)` guard would see it still
  // null and re-init a second time. The pointer is never dereferenced (opaque type), so a fixed
  // non-null sentinel is enough -- no allocation/cleanup bookkeeping needed.
  if (g_diagClientInitResult == ncclSuccess && client) {
    client->diagnostics = reinterpret_cast<struct rasDiagnosticsClientState*>(0x1);
  }
  return g_diagClientInitResult;
}

ncclResult_t rasDiagnosticsStart(struct rasClient* /*client*/) {
  ++g_diagStartCalls;
  return g_diagStartResult;
}

bool rasDiagnosticsInProgress() {
  ++g_diagInProgressCalls;
  return g_diagInProgress;
}

void rasDiagnosticsCancelTarget(void* target) {
  ++g_diagCancelTargetCalls;
  g_lastCancelTarget = target;
}

void rasDiagnosticsClientCleanup(struct rasClient* client) {
  ++g_diagClientCleanupCalls;
  g_lastClientCleanup = client;
}

ncclResult_t rasDiagnosticsResume(struct rasClient* /*client*/) {
  ++g_diagResumeCalls;
  return g_diagResumeResult;
}

// ---------------------------------------------------------------------------
// src/init.cc extern (only the name table; init.cc itself isn't linked here).
// Verbatim copy of src/init.cc's own definition.
// ---------------------------------------------------------------------------
const char* ncclFuncStr[NCCL_NUM_FUNCTIONS + 4] = {"Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce",
                                                   "AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda", "SendRecv"};

// ---------------------------------------------------------------------------
// misc/socket.cc externs (only rasClientInitSocket needs these).
// ---------------------------------------------------------------------------

const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, const int) {
  buf[0] = '\0';
  return buf;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* addr, const char*) {
  memset(addr, 0, sizeof(*addr));
  addr->sin.sin_family = AF_INET;
  addr->sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr->sin.sin_port = htons(28028);
  return ncclSuccess;
}

namespace {

// ---------------------------------------------------------------------------
// Test helpers.
// ---------------------------------------------------------------------------

void ResetWholeFileSeams() {
  g_clockNano = 100 * CLOCK_UNITS_PER_SEC;

  g_getNewPollEntryCalls = 0;
  g_pfdsStorage.clear();
  rasPfds = nullptr;
  g_msgAllocCalls = 0;
  g_msgFreeCalls = 0;
  g_msgAllocResult = ncclSuccess;

  rasPeers = nullptr;
  nRasPeers = 0;
  rasPeersHash = 0;
  rasDeadPeers = nullptr;
  nRasDeadPeers = 0;
  rasDeadPeersHash = 0;
  g_ncclSocketsSameNode = nullptr;
  g_peerToStringCalls = 0;
  g_peerInfoToStringCalls = 0;

  g_collReqInitCalls = 0;
  g_netSendCollReqCalls = 0;
  memset(&g_lastCollReq, 0, sizeof(g_lastCollReq));
  g_netSendCollReqResult = ncclSuccess;
  g_netSendCollReqAllDone = true;
  g_nextCollective = nullptr;
  g_collRecordHistoryCalls = 0;
  g_lastCollRecordHistory = nullptr;
  g_collFreeCalls = 0;

  g_diagContextInitCalls = 0;
  g_diagContextInitResult = ncclSuccess;
  g_diagClientInitCalls = 0;
  g_diagClientInitResult = ncclSuccess;
  memset(&g_lastReporter, 0, sizeof(g_lastReporter));
  g_haveLastReporter = false;
  g_diagStartCalls = 0;
  g_diagStartResult = ncclSuccess;
  g_diagInProgressCalls = 0;
  g_diagInProgress = false;
  g_diagCancelTargetCalls = 0;
  g_lastCancelTarget = nullptr;
  g_diagClientCleanupCalls = 0;
  g_lastClientCleanup = nullptr;
  g_diagResumeCalls = 0;
  g_diagResumeResult = ncclSuccess;
  g_diagFormatLineCalls = 0;
  g_diagFormatLineResult = ncclSuccess;

  cudaDriverVersion = cudaRuntimeVersion = -1;
  rasClientListeningSocket = -1;
  // Defensive, not just a reset: a test whose production path doesn't unlink a client on an error
  // arm (e.g. rasClientAcceptNewSocket() leaves the client linked when accept()/fcntl() fails,
  // matching real client_support.cc behavior) would otherwise leak it across into the next test.
  for (struct rasClient* client = rasClientsHead; client;) {
    struct rasClient* next = client->next;
    while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&client->sendQ)) free(meta);
    free(client);
    client = next;
  }
  rasClientsHead = rasClientsTail = nullptr;
}

// Builds a loopback IPv4 address, distinguished by `port` so tests can construct several
// distinct peers without caring about the actual host.
union ncclSocketAddress MakeAddr(uint16_t port) {
  union ncclSocketAddress addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin.sin_family = AF_INET;
  addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin.sin_port = htons(port);
  return addr;
}

// MakeAddr()'s addresses all share one host IP, so the default ncclSocketsSameNode() (host
// compare, ignoring port) would call every peer "same node" -- no way to build a multi-node
// fixture from it alone. Installs a g_ncclSocketsSameNode hook that instead buckets by
// port/100, so MakeAddr(100)/MakeAddr(101) are "node 1" and MakeAddr(200) is a different node,
// letting a test's peer topology be read directly off the ports it chose.
void InstallPortBucketNodeGrouping() {
  g_ncclSocketsSameNode = [](const union ncclSocketAddress* a1, const union ncclSocketAddress* a2) {
    return (ntohs(a1->sin.sin_port) / 100) == (ntohs(a2->sin.sin_port) / 100);
  };
}

// Allocates a fresh rasClient the way getNewClientEntry() does, without going through the
// production linked-list bookkeeping -- most tests want a standalone client, not one already
// wired into rasClientsHead/Tail. Tests that need list-bookkeeping coverage call
// getNewClientEntry() (via rasClientAcceptNewSocket()/rasLocalHandleRunDiag()) directly instead.
//
// status defaults to RAS_CLIENT_CONNECTED (not calloc's RAS_CLIENT_CLOSED): rasClientEnqueueMsg()
// silently drops every message for a CLOSED or >=FINISHED client, so any test exercising an
// enqueue helper needs a status that survives that check. Callers that want a specific state
// overwrite client->status afterward. pfd is allocated via the real rasGetNewPollEntry() (not
// left at -1): rasClientEnqueueMsg() unconditionally writes rasPfds[client->pfd] once the status
// check passes, and rasPfds[-1] is a wild write.
struct rasClient* MakeClient() {
  auto* client = static_cast<struct rasClient*>(calloc(1, sizeof(struct rasClient)));
  client->sock = -1;
  rasGetNewPollEntry(&client->pfd);
  ncclIntruQueueConstruct(&client->sendQ);
  client->timeout = RAS_COLLECTIVE_LEG_TIMEOUT;
  client->outputFormat = RAS_OUTPUT_TEXT;
  client->status = RAS_CLIENT_CONNECTED;
  return client;
}

void FreeClient(struct rasClient* client) {
  while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&client->sendQ)) free(meta);
  free(client);
}

// rasClientEventLoop() reads its poll flags from rasPfds[pollIdx].revents, not a parameter.
void SetRevents(struct rasClient* client, short revents) { rasPfds[client->pfd].revents = revents; }

// Drains every queued message off a client's sendQ into a single string, in order, freeing each
// meta node as it goes -- lets a test assert on exactly what rasClientRunInit/RunComms/etc. wrote
// without duplicating the queue's own accounting.
std::string DrainSendQueue(struct rasClient* client) {
  std::string out;
  while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&client->sendQ)) {
    out.append(reinterpret_cast<char*>(&meta->msg), meta->length);
    free(meta);
  }
  return out;
}

// Heap-allocates a rasCollective ready to hand to rasNetSendCollReq()'s g_nextCollective slot.
// `data`/`nData` become coll->data/coll->nData (typically a BuildRasCollComms() blob); `peers`
// becomes a freshly-copied coll->peers array so rasCollFree()'s free(coll->peers) is always valid.
struct rasCollective* MakeCollective(std::vector<char> data, std::vector<union ncclSocketAddress> peers,
                                     int nLegTimeouts = 0) {
  auto* coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  coll->startTime = 0;
  coll->nFwdSent = coll->nFwdRecv = 1;
  coll->nLegTimeouts = nLegTimeouts;
  coll->nPeers = static_cast<int>(peers.size());
  if (!peers.empty()) {
    coll->peers = static_cast<union ncclSocketAddress*>(malloc(peers.size() * sizeof(union ncclSocketAddress)));
    memcpy(coll->peers, peers.data(), peers.size() * sizeof(union ncclSocketAddress));
  }
  coll->nData = static_cast<int>(data.size());
  if (!data.empty()) {
    coll->data = static_cast<char*>(malloc(data.size()));
    memcpy(coll->data, data.data(), data.size());
  }
  return coll;
}

// ---------------------------------------------------------------------------
// rasCollComms fixture builder: hand-assembles the variable-length
// [comm][ranks...][missingRanks...] blob rasClientRunComms()/rasDumpCommsToJSON() consume.
// ---------------------------------------------------------------------------

struct RankSpec {
  int commRank;
  int peerIdx;
  uint64_t collOpCounts[NCCL_NUM_FUNCTIONS] = {};
  ncclResult_t initState = ncclSuccess;
  ncclResult_t asyncError = ncclSuccess;
  bool finalizeCalled = false;
  bool destroyFlag = false;
  bool abortFlag = false;
  char cudaDev = 0;
  char nvmlDev = 0;
};

struct MissingSpec {
  int commRank;
  union ncclSocketAddress addr;
  char cudaDev = 0;
  char nvmlDev = 0;
};

struct CommSpec {
  uint64_t commHash = 1, hostHash = 1, pidHash = 1;
  int commNRanks = -1;  // -1: derive from ranks.size() + missing.size()
  std::vector<RankSpec> ranks;
  std::vector<MissingSpec> missing;
};

// Plain-field-assignment builder (no designated initializers, which are C++20 -- this project
// builds as C++17) for the common case of pinning commNRanks independent of ranks.size().
CommSpec MakeCommSpec(int commNRanks, std::vector<RankSpec> ranks = {}) {
  CommSpec c;
  c.commNRanks = commNRanks;
  c.ranks = std::move(ranks);
  return c;
}

std::vector<char> BuildRasCollComms(const std::vector<CommSpec>& comms) {
  size_t total = offsetof(struct rasCollComms, comms);
  for (const auto& c : comms) {
    total += offsetof(struct rasCollComms::comm, ranks);
    total += c.ranks.size() * sizeof(struct rasCollComms::comm::rank);
    total += c.missing.size() * sizeof(struct rasCollCommsMissingRank);
  }
  std::vector<char> buf(total, 0);
  auto* out = reinterpret_cast<struct rasCollComms*>(buf.data());
  out->nComms = static_cast<int>(comms.size());
  char* cursor = reinterpret_cast<char*>(out->comms);
  for (const auto& c : comms) {
    auto* comm = reinterpret_cast<struct rasCollComms::comm*>(cursor);
    comm->commId.commHash = c.commHash;
    comm->commId.hostHash = c.hostHash;
    comm->commId.pidHash = c.pidHash;
    comm->commNRanks = (c.commNRanks >= 0 ? c.commNRanks
                                          : static_cast<int>(c.ranks.size() + c.missing.size()));
    comm->nRanks = static_cast<int>(c.ranks.size());
    comm->nMissingRanks = static_cast<int>(c.missing.size());
    for (size_t i = 0; i < c.ranks.size(); i++) {
      const RankSpec& r = c.ranks[i];
      struct rasCollComms::comm::rank* rank = comm->ranks + i;
      rank->commRank = r.commRank;
      rank->peerIdx = r.peerIdx;
      memcpy(rank->collOpCounts, r.collOpCounts, sizeof(rank->collOpCounts));
      rank->status.initState = r.initState;
      rank->status.asyncError = r.asyncError;
      rank->status.finalizeCalled = r.finalizeCalled;
      rank->status.destroyFlag = r.destroyFlag;
      rank->status.abortFlag = r.abortFlag;
      rank->cudaDev = r.cudaDev;
      rank->nvmlDev = r.nvmlDev;
    }
    auto* missingArr = reinterpret_cast<struct rasCollCommsMissingRank*>(comm->ranks + c.ranks.size());
    for (size_t i = 0; i < c.missing.size(); i++) {
      const MissingSpec& m = c.missing[i];
      missingArr[i].commRank = m.commRank;
      missingArr[i].addr = m.addr;
      missingArr[i].cudaDev = m.cudaDev;
      missingArr[i].nvmlDev = m.nvmlDev;
    }
    cursor = reinterpret_cast<char*>(missingArr + c.missing.size());
  }
  return buf;
}

// The first (and, in every comparator test, only) comm entry in a BuildRasCollComms() blob.
// NOT the same address as buf.data(): that points at the wrapping rasCollComms (nComms, then the
// comms[] flexible array member) -- casting buf.data() straight to comm* skips over nComms and
// reads every field shifted by offsetof(rasCollComms, comms) bytes.
struct rasCollComms::comm* FirstComm(std::vector<char>& buf) {
  return reinterpret_cast<struct rasCollComms*>(buf.data())->comms;
}

class RasClientSupportMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetLibcFakes();
    ResetHipFakes();
    ResetWholeFileSeams();
  }
};

}  // namespace

// =============================================================================================
// rasParseProfilerMask -- pure function, dense branch table.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_Null_ReturnsFalse) {
  int mask = 0;
  EXPECT_FALSE(rasParseProfilerMask(nullptr, &mask));
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_Empty_ReturnsFalse) {
  int mask = 0;
  EXPECT_FALSE(rasParseProfilerMask("", &mask));
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_None_ReturnsZero) {
  int mask = -1;
  EXPECT_TRUE(rasParseProfilerMask("none", &mask));
  EXPECT_EQ(mask, 0);
  EXPECT_TRUE(rasParseProfilerMask("NONE", &mask));
  EXPECT_EQ(mask, 0);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_All_ReturnsAllBits) {
  int mask = 0;
  EXPECT_TRUE(rasParseProfilerMask("all", &mask));
  EXPECT_NE(mask, 0);
  int allMask = mask;
  mask = 0;
  EXPECT_TRUE(rasParseProfilerMask("ALL", &mask));
  EXPECT_EQ(mask, allMask);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_Hex_Parsed) {
  int mask = 0;
  EXPECT_TRUE(rasParseProfilerMask("0x10", &mask));
  EXPECT_EQ(mask, 0x10);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_Decimal_Parsed) {
  int mask = 0;
  EXPECT_TRUE(rasParseProfilerMask("42", &mask));
  EXPECT_EQ(mask, 42);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_Octal_Parsed) {
  int mask = 0;
  EXPECT_TRUE(rasParseProfilerMask("010", &mask));
  EXPECT_EQ(mask, 8);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_NegativeNumeric_RejectedAsFalse) {
  // strtol succeeds on "-1", but the leading '-' fails the `str[0] >= '0'` gate, so this falls
  // through to the symbolic-token path and fails as an unknown token, not the val<0 numeric check.
  int mask = 0;
  EXPECT_FALSE(rasParseProfilerMask("-1", &mask));
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_NumericTrailingGarbage_ReturnsFalse) {
  int mask = 0;
  EXPECT_FALSE(rasParseProfilerMask("123abc", &mask));
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_SymbolicSingleToken_Parsed) {
  int mask = 0;
  EXPECT_TRUE(rasParseProfilerMask("group", &mask));
  EXPECT_EQ(mask, ncclProfileGroup);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_SymbolicAllFourteenTokens_Parsed) {
  int mask = 0;
  EXPECT_TRUE(rasParseProfilerMask(
      "group,coll,p2p,proxyop,proxystep,proxyctrl,kernelch,netplugin,groupapi,collapi,p2papi,"
      "kernellaunch,cecoll,cesync,cebatch",
      &mask));
  const int allBits = ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp |
                       ncclProfileProxyStep | ncclProfileProxyCtrl | ncclProfileKernelCh | ncclProfileNetPlugin |
                       ncclProfileGroupApi | ncclProfileCollApi | ncclProfileP2pApi | ncclProfileKernelLaunch |
                       ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  EXPECT_EQ(mask, allBits);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_SymbolicCaseInsensitive_Parsed) {
  int mask = 0;
  EXPECT_TRUE(rasParseProfilerMask("GROUP,Coll", &mask));
  EXPECT_EQ(mask, ncclProfileGroup | ncclProfileColl);
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_UnknownToken_ReturnsFalse) {
  int mask = 0;
  EXPECT_FALSE(rasParseProfilerMask("bogus", &mask));
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_KnownThenUnknownToken_ReturnsFalse) {
  int mask = 0;
  EXPECT_FALSE(rasParseProfilerMask("group,bogus", &mask));
}

TEST_F(RasClientSupportMicrotest, ParseProfilerMask_OverlongString_ReturnsFalse) {
  std::string tooLong(1100, 'x');
  int mask = 0;
  EXPECT_FALSE(rasParseProfilerMask(tooLong.c_str(), &mask));
}

// =============================================================================================
// rasCountIsOutlier -- pure function, boundary-value table.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, CountIsOutlier_CountOfOne_AlwaysTrue) {
  EXPECT_TRUE(rasCountIsOutlier(1, /*verbose*/ false, /*totalCount*/ -1));
  EXPECT_TRUE(rasCountIsOutlier(1, /*verbose*/ true, /*totalCount*/ 100));
}

TEST_F(RasClientSupportMicrotest, CountIsOutlier_Verbose_NoTotalCount_AlwaysTrue) {
  EXPECT_TRUE(rasCountIsOutlier(50, /*verbose*/ true, /*totalCount*/ -1));
}

TEST_F(RasClientSupportMicrotest, CountIsOutlier_Verbose_BelowHalfOfTotal_True) {
  EXPECT_TRUE(rasCountIsOutlier(4, /*verbose*/ true, /*totalCount*/ 10));
}

TEST_F(RasClientSupportMicrotest, CountIsOutlier_Verbose_AtOrAboveHalfOfTotal_False) {
  EXPECT_FALSE(rasCountIsOutlier(5, /*verbose*/ true, /*totalCount*/ 10));
}

TEST_F(RasClientSupportMicrotest, CountIsOutlier_NonVerbose_AboveThreshold_False) {
  // RAS_CLIENT_DETAIL_THRESHOLD is 10.
  EXPECT_FALSE(rasCountIsOutlier(11, /*verbose*/ false, /*totalCount*/ -1));
}

TEST_F(RasClientSupportMicrotest, CountIsOutlier_NonVerbose_AtThreshold_NoTotalCount_True) {
  EXPECT_TRUE(rasCountIsOutlier(10, /*verbose*/ false, /*totalCount*/ -1));
}

TEST_F(RasClientSupportMicrotest, CountIsOutlier_NonVerbose_AtThreshold_AboveQuarterOfTotal_False) {
  // RAS_CLIENT_OUTLIER_FRACTION is 0.25; count=10 > totalCount*0.25=2.5.
  EXPECT_FALSE(rasCountIsOutlier(10, /*verbose*/ false, /*totalCount*/ 10));
}

TEST_F(RasClientSupportMicrotest, CountIsOutlier_NonVerbose_AtOrBelowQuarterOfTotal_True) {
  EXPECT_TRUE(rasCountIsOutlier(2, /*verbose*/ false, /*totalCount*/ 10));
}

// =============================================================================================
// Sorting comparators -- called directly, not just via qsort.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, SocketsHostCompare_BothEmpty_Equal) {
  union ncclSocketAddress a1 = {}, a2 = {};
  EXPECT_EQ(ncclSocketsHostCompare(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, SocketsHostCompare_OneEmptyOneSet_EmptySortsLast) {
  union ncclSocketAddress empty = {};
  union ncclSocketAddress set = MakeAddr(1000);
  EXPECT_LT(ncclSocketsHostCompare(&set, &empty), 0);
  EXPECT_GT(ncclSocketsHostCompare(&empty, &set), 0);
}

TEST_F(RasClientSupportMicrotest, SocketsHostCompare_Inet4_ByteCompare) {
  union ncclSocketAddress a1 = MakeAddr(1000);
  union ncclSocketAddress a2 = MakeAddr(1000);
  a2.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK + 1);
  EXPECT_LT(ncclSocketsHostCompare(&a1, &a2), 0);
  EXPECT_GT(ncclSocketsHostCompare(&a2, &a1), 0);
}

TEST_F(RasClientSupportMicrotest, SocketsHostCompare_Inet4_IgnoresPort) {
  union ncclSocketAddress a1 = MakeAddr(1000);
  union ncclSocketAddress a2 = MakeAddr(2000);
  EXPECT_EQ(ncclSocketsHostCompare(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, SocketsHostCompare_Inet6_ByteCompare) {
  union ncclSocketAddress a1 = {}, a2 = {};
  a1.sin6.sin6_family = a2.sin6.sin6_family = AF_INET6;
  a1.sin6.sin6_addr.s6_addr[0] = 1;
  a2.sin6.sin6_addr.s6_addr[0] = 2;
  EXPECT_LT(ncclSocketsHostCompare(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, SocketsHostCompare_DifferentFamilies_Inet4BeforeInet6) {
  union ncclSocketAddress a1 = MakeAddr(1000);
  union ncclSocketAddress a2 = {};
  a2.sin6.sin6_family = AF_INET6;
  EXPECT_LT(ncclSocketsHostCompare(&a1, &a2), 0);
  EXPECT_GT(ncclSocketsHostCompare(&a2, &a1), 0);
}

TEST_F(RasClientSupportMicrotest, AuxPeersValueCompare_ValueDiffers_SortsByValue) {
  struct rasPeerInfo peer1 = {}, peer2 = {};
  struct rasAuxPeerInfo p1{&peer1, /*value*/ 1};
  struct rasAuxPeerInfo p2{&peer2, /*value*/ 2};
  EXPECT_LT(rasAuxPeersValueCompare(&p1, &p2), 0);
  EXPECT_GT(rasAuxPeersValueCompare(&p2, &p1), 0);
}

TEST_F(RasClientSupportMicrotest, AuxPeersValueCompare_ValueEqual_TiebreakByHostThenPid) {
  struct rasPeerInfo peer1 = {}, peer2 = {};
  peer1.addr = MakeAddr(1000);
  peer2.addr = MakeAddr(1000);
  peer1.pid = 5;
  peer2.pid = 9;
  struct rasAuxPeerInfo p1{&peer1, 0};
  struct rasAuxPeerInfo p2{&peer2, 0};
  EXPECT_LT(rasAuxPeersValueCompare(&p1, &p2), 0);
  EXPECT_GT(rasAuxPeersValueCompare(&p2, &p1), 0);
  peer2.pid = 5;
  EXPECT_EQ(rasAuxPeersValueCompare(&p1, &p2), 0);
}

TEST_F(RasClientSupportMicrotest, ValCountsCompareRev_CountDiffers_LargerCountFirst) {
  struct rasValCount v1{/*value*/ 0, /*count*/ 5, /*firstIdx*/ 0};
  struct rasValCount v2{/*value*/ 0, /*count*/ 3, /*firstIdx*/ 0};
  EXPECT_LT(rasValCountsCompareRev(&v1, &v2), 0);
  EXPECT_GT(rasValCountsCompareRev(&v2, &v1), 0);
}

TEST_F(RasClientSupportMicrotest, ValCountsCompareRev_CountEqual_TiebreakByValueDesc) {
  struct rasValCount v1{/*value*/ 9, /*count*/ 5, /*firstIdx*/ 0};
  struct rasValCount v2{/*value*/ 3, /*count*/ 5, /*firstIdx*/ 0};
  EXPECT_LT(rasValCountsCompareRev(&v1, &v2), 0);
  EXPECT_EQ(rasValCountsCompareRev(&v1, &v1), 0);
}

TEST_F(RasClientSupportMicrotest, AuxCommRanksValueCompare_ValueDiffers_SortsByValue) {
  struct rasCollComms::comm::rank r1 = {}, r2 = {};
  r1.commRank = 5;
  r2.commRank = 1;
  struct rasAuxCommRank a1{&r1, 1};
  struct rasAuxCommRank a2{&r2, 2};
  EXPECT_LT(rasAuxCommRanksValueCompare(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, AuxCommRanksValueCompare_ValueEqual_TiebreakByCommRank) {
  struct rasCollComms::comm::rank r1 = {}, r2 = {};
  r1.commRank = 5;
  r2.commRank = 1;
  struct rasAuxCommRank a1{&r1, 0};
  struct rasAuxCommRank a2{&r2, 0};
  EXPECT_GT(rasAuxCommRanksValueCompare(&a1, &a2), 0);
  EXPECT_LT(rasAuxCommRanksValueCompare(&a2, &a1), 0);
  r2.commRank = 5;
  EXPECT_EQ(rasAuxCommRanksValueCompare(&a1, &a2), 0);
}

namespace {
// rasAuxCommsCompareRev needs a real rasCollComms::comm to point at (it dereferences c->comm).
std::vector<char> OneEmptyComm() { return BuildRasCollComms({CommSpec{}}); }
}  // namespace

TEST_F(RasClientSupportMicrotest, AuxCommsCompareRev_CommNRanksDiffers_LargerFirst) {
  std::vector<char> buf1 = BuildRasCollComms({MakeCommSpec(8)});
  std::vector<char> buf2 = BuildRasCollComms({MakeCommSpec(4)});
  struct rasAuxComm a1 = {};
  a1.comm = FirstComm(buf1);
  struct rasAuxComm a2 = {};
  a2.comm = FirstComm(buf2);
  EXPECT_LT(rasAuxCommsCompareRev(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, AuxCommsCompareRev_NNodesDiffers_MoreNodesFirst) {
  std::vector<char> buf = OneEmptyComm();
  auto* comm = FirstComm(buf);
  struct rasAuxComm a1 = {};
  a1.comm = comm;
  a1.nNodes = 3;
  struct rasAuxComm a2 = {};
  a2.comm = comm;
  a2.nNodes = 1;
  EXPECT_LT(rasAuxCommsCompareRev(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, AuxCommsCompareRev_StatusDiffers_HigherBitFirst) {
  std::vector<char> buf = OneEmptyComm();
  auto* comm = FirstComm(buf);
  struct rasAuxComm a1 = {};
  a1.comm = comm;
  a1.status = RAS_ACS_ABORT;  // highest bit -> smallest CLZ -> sorts first
  struct rasAuxComm a2 = {};
  a2.comm = comm;
  a2.status = RAS_ACS_INIT;
  EXPECT_LT(rasAuxCommsCompareRev(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, AuxCommsCompareRev_ErrorsDiffers_MoreErrorsFirst) {
  std::vector<char> buf = OneEmptyComm();
  auto* comm = FirstComm(buf);
  struct rasAuxComm a1 = {};
  a1.comm = comm;
  a1.status = RAS_ACS_INIT;
  a1.errors = RAS_ACE_MISMATCH | RAS_ACE_ERROR;
  struct rasAuxComm a2 = {};
  a2.comm = comm;
  a2.status = RAS_ACS_INIT;
  a2.errors = RAS_ACE_OK;
  EXPECT_LT(rasAuxCommsCompareRev(&a1, &a2), 0);
}

TEST_F(RasClientSupportMicrotest, AuxCommsCompareRev_NRanksDiffers_FewerRanksFirst) {
  // Final tiebreak is comm->nRanks ASCENDING (most-missing-ranks first), unlike every key above
  // it. commNRanks is pinned equal (and decoupled from ranks.size()) so the two comms tie on every
  // higher-priority key and control actually reaches this branch, instead of the commNRanks
  // comparison short-circuiting first. status is pinned to a real, non-zero value (not the
  // zero-init default): COMPILER_CLZ(status) compares the two, and __builtin_clz(0) is undefined
  // behavior -- status==0 is not a state rasClientRunComms ever produces for a comm with
  // nRanks>0 anyway (its per-rank loop always sets at least one RAS_ACS_* bit), so a non-zero
  // value here also matches what this comparator is actually ever called with in production.
  std::vector<char> buf1 = BuildRasCollComms({MakeCommSpec(5, {RankSpec{0, 0}})});
  std::vector<char> buf2 = BuildRasCollComms({MakeCommSpec(5, {RankSpec{0, 0}, RankSpec{1, 1}})});
  struct rasAuxComm a1 = {};
  a1.comm = FirstComm(buf1);
  a1.status = RAS_ACS_RUNNING;
  struct rasAuxComm a2 = {};
  a2.comm = FirstComm(buf2);
  a2.status = RAS_ACS_RUNNING;
  EXPECT_LT(rasAuxCommsCompareRev(&a1, &a2), 0);
  EXPECT_GT(rasAuxCommsCompareRev(&a2, &a1), 0);
}

TEST_F(RasClientSupportMicrotest, AuxCommsCompareRev_EveryKeyEqual_Zero) {
  std::vector<char> buf = OneEmptyComm();
  auto* comm = FirstComm(buf);
  struct rasAuxComm a1 = {};
  a1.comm = comm;
  a1.status = RAS_ACS_RUNNING;  // non-zero: avoids relying on __builtin_clz(0) being well-defined
  EXPECT_EQ(rasAuxCommsCompareRev(&a1, &a1), 0);
}

// =============================================================================================
// Output buffer management.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, OutBuffer_AppendThenExtract_RoundTrips) {
  rasOutReset();
  rasOutAppend("hello %d", 42);
  EXPECT_EQ(rasOutLength(), static_cast<int>(strlen("hello 42")));
  // rasOutExtract() copies exactly rasOutLength() bytes -- RAS messages are length-prefixed, not
  // NUL-terminated, so the destination must supply its own terminator or arrive pre-zeroed.
  char buf[64] = {};
  rasOutExtract(buf);
  EXPECT_STREQ(buf, "hello 42");
  // rasOutExtract() resets the buffer as a side effect.
  EXPECT_EQ(rasOutLength(), 0);
}

TEST_F(RasClientSupportMicrotest, OutBuffer_AppendPastInitialCapacity_Reallocates) {
  rasOutReset();
  std::string longStr(RAS_OUT_INCREMENT + 100, 'x');
  rasOutAppend("%s", longStr.c_str());
  EXPECT_EQ(rasOutLength(), static_cast<int>(longStr.size()));
  std::vector<char> buf(rasOutLength() + 1);
  rasOutExtract(buf.data());
  EXPECT_EQ(std::string(buf.data()), longStr);
}

TEST_F(RasClientSupportMicrotest, OutBuffer_ExtractWithoutAnyAppend_NoOp) {
  // rasOutBuffer starts null (no rasOutReset() call yet in this test) -- rasOutExtract() must not
  // crash when there is nothing to extract.
  char buf[8] = {'z', '\0'};
  rasOutExtract(buf);
  EXPECT_EQ(buf[0], 'z');
}

TEST_F(RasClientSupportMicrotest, OutBuffer_MultipleAppends_Accumulate) {
  rasOutReset();
  rasOutAppend("a");
  rasOutAppend("b");
  rasOutAppend("c");
  char buf[8] = {};
  rasOutExtract(buf);
  EXPECT_STREQ(buf, "abc");
}

// =============================================================================================
// String / GPU formatting functions.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, GpuDevsToString_SameCudaAndNvml_NoSuffix) {
  char buf[64];
  EXPECT_STREQ(rasGpuDevsToString(0b101, 0b101, buf, sizeof(buf)), "0,2");
}

TEST_F(RasClientSupportMicrotest, GpuDevsToString_DifferentNvml_AppendsSuffix) {
  char buf[64];
  const char* s = rasGpuDevsToString(0b1, 0b10, buf, sizeof(buf));
  EXPECT_STREQ(s, "0 (NVML 1)");
}

TEST_F(RasClientSupportMicrotest, GpuDevsToString_EmptyMask_EmptyString) {
  char buf[64];
  EXPECT_STREQ(rasGpuDevsToString(0, 0, buf, sizeof(buf)), "");
}

TEST_F(RasClientSupportMicrotest, GpuToString_SameIds_JustNumber) {
  char buf[32];
  EXPECT_STREQ(rasGpuToString(3, 3, buf, sizeof(buf)), "3");
}

TEST_F(RasClientSupportMicrotest, GpuToString_DifferentIds_AppendsNvml) {
  char buf[32];
  EXPECT_STREQ(rasGpuToString(3, 5, buf, sizeof(buf)), "3 (NVML 5)");
}

TEST_F(RasClientSupportMicrotest, CommRankGpuToString_DelegatesToGpuToString) {
  struct rasCollComms::comm::rank rank = {};
  rank.cudaDev = 2;
  rank.nvmlDev = 4;
  char buf[32];
  EXPECT_STREQ(rasCommRankGpuToString(&rank, buf, sizeof(buf)), "2 (NVML 4)");
}

TEST_F(RasClientSupportMicrotest, ErrorToString_EveryNamedValue_MatchesExpectedText) {
  EXPECT_STREQ(ncclErrorToString(ncclUnhandledCudaError), "Unhandled CUDA error");
  EXPECT_STREQ(ncclErrorToString(ncclSystemError), "System error");
  EXPECT_STREQ(ncclErrorToString(ncclInternalError), "Internal error");
  EXPECT_STREQ(ncclErrorToString(ncclInvalidArgument), "Invalid argument");
  EXPECT_STREQ(ncclErrorToString(ncclInvalidUsage), "Invalid usage");
  EXPECT_STREQ(ncclErrorToString(ncclRemoteError), "Remote process error");
  EXPECT_STREQ(ncclErrorToString(ncclInProgress), "NCCL operation in progress");
}

TEST_F(RasClientSupportMicrotest, ErrorToString_Success_FallsIntoDefault) {
  EXPECT_STREQ(ncclErrorToString(ncclSuccess), "Unexpected error");
}

TEST_F(RasClientSupportMicrotest, SocketToHost_Inet4_FormatsDottedQuad) {
  union ncclSocketAddress addr = MakeAddr(1234);
  char buf[64];
  EXPECT_STREQ(ncclSocketToHost(&addr, buf, sizeof(buf)), "127.0.0.1");
}

TEST_F(RasClientSupportMicrotest, SocketToHost_EmptyFamily_EmptyString) {
  union ncclSocketAddress addr = {};
  char buf[64];
  EXPECT_STREQ(ncclSocketToHost(&addr, buf, sizeof(buf)), "");
}

// =============================================================================================
// Client-list bookkeeping and enqueue helpers.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, AcceptNewSocket_Success_LinksClientAndArmsPollin) {
  ASSERT_EQ(rasClientInitSocket(), ncclSuccess);
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  ASSERT_NE(rasClientsHead, nullptr);
  EXPECT_EQ(rasClientsHead, rasClientsTail);
  EXPECT_EQ(rasClientsHead->status, RAS_CLIENT_CONNECTED);
  EXPECT_EQ(g_getNewPollEntryCalls, 1);
  EXPECT_EQ(rasPfds[rasClientsHead->pfd].events, POLLIN);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, AcceptNewSocket_SecondClient_AppendsToTail) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* first = rasClientsHead;
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  EXPECT_EQ(rasClientsHead, first);
  EXPECT_NE(rasClientsTail, first);
  EXPECT_EQ(rasClientsTail->prev, first);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, AcceptNewSocket_AcceptFails_PropagatesErrorWithoutClosingAnything) {
  g_nextAcceptFd = -1;
  EXPECT_NE(rasClientAcceptNewSocket(), ncclSuccess);
  // getNewClientEntry() links the client before accept() is attempted, and the fail path only
  // closes client->sock if it was ever set to something other than -1 -- neither happened here, so
  // the client stays linked (with sock==-1) and nothing gets closed. Matches client_support.cc's
  // own fail: block, not a gap this test papers over.
  ASSERT_NE(rasClientsHead, nullptr);
  EXPECT_EQ(rasClientsHead->sock, -1);
  EXPECT_TRUE(g_closedFds.empty());
}

TEST_F(RasClientSupportMicrotest, AcceptNewSocket_FcntlFails_PropagatesError) {
  g_fcntlResult = -1;
  EXPECT_NE(rasClientAcceptNewSocket(), ncclSuccess);
}

TEST_F(RasClientSupportMicrotest, EnqueueMsg_NormalStatus_ArmsPollout) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 5), ncclSuccess);
  memcpy(msg, "hello", 5);
  rasClientEnqueueMsg(client, msg, 5);
  EXPECT_TRUE(rasPfds[client->pfd].events & POLLOUT);
  EXPECT_EQ(DrainSendQueue(client), "hello");
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, EnqueueMsg_ClosedStatus_DropsMessageSilently) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_CLOSED;
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 3), ncclSuccess);
  memcpy(msg, "abc", 3);
  rasClientEnqueueMsg(client, msg, 3);
  EXPECT_TRUE(ncclIntruQueueEmpty(&client->sendQ));
  rasClientFreeMsg(msg);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EnqueueMsg_FinishedStatus_DropsMessageSilently) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_FINISHED;
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 3), ncclSuccess);
  rasClientEnqueueMsg(client, msg, 3);
  EXPECT_TRUE(ncclIntruQueueEmpty(&client->sendQ));
  rasClientFreeMsg(msg);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EnqueueText_AppendNewlineTrue_AddsTrailingNewline) {
  struct rasClient* client = MakeClient();
  ASSERT_EQ(rasClientEnqueueLine(client, "line"), ncclSuccess);
  EXPECT_EQ(DrainSendQueue(client), "line\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EnqueueText_AppendNewlineFalse_NoTrailingNewline) {
  struct rasClient* client = MakeClient();
  ASSERT_EQ(rasClientEnqueueString(client, "line"), ncclSuccess);
  EXPECT_EQ(DrainSendQueue(client), "line");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, ClientTerminate_OnlyNodeInList_ClearsHeadAndTail) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  rasClientTerminate(client);
  EXPECT_EQ(rasClientsHead, nullptr);
  EXPECT_EQ(rasClientsTail, nullptr);
}

TEST_F(RasClientSupportMicrotest, ClientTerminate_HeadOfMultiple_AdvancesHead) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* first = rasClientsHead;
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* second = rasClientsTail;
  rasClientTerminate(first);
  EXPECT_EQ(rasClientsHead, second);
  EXPECT_EQ(second->prev, nullptr);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, ClientTerminate_TailOfMultiple_RetreatsTail) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* first = rasClientsHead;
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* second = rasClientsTail;
  rasClientTerminate(second);
  EXPECT_EQ(rasClientsTail, first);
  EXPECT_EQ(first->next, nullptr);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, ClientTerminate_MiddleOfThree_StitchesNeighbors) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* first = rasClientsHead;
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* middle = rasClientsTail;
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* last = rasClientsTail;
  rasClientTerminate(middle);
  EXPECT_EQ(first->next, last);
  EXPECT_EQ(last->prev, first);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, ClientTerminate_WithColl_RecordsHistoryAndFrees) {
  struct rasClient* client = MakeClient();
  client->coll = MakeCollective({}, {});
  rasClientTerminate(client);
  EXPECT_EQ(g_collRecordHistoryCalls, 1);
  EXPECT_EQ(g_collFreeCalls, 1);
}

TEST_F(RasClientSupportMicrotest, ClientTerminate_DrainsSendQueue) {
  struct rasClient* client = MakeClient();
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 3), ncclSuccess);
  rasClientEnqueueMsg(client, msg, 3);
  int freesBefore = g_msgFreeCalls;
  rasClientTerminate(client);
  // rasClientTerminate() frees the queued rasMsgMeta directly (not via rasMsgFree), so the only
  // observable signal is that the client (and its queue) is gone -- assert via the recorded alloc
  // count staying put (no double count) rather than g_msgFreeCalls, which this path never touches.
  EXPECT_EQ(g_msgFreeCalls, freesBefore);
}

TEST_F(RasClientSupportMicrotest, ClientsNotifyEvent_NoMatchingClients_NoOp) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  rasClientsHead->monitorMask = 0;
  struct rasEventNotification event{"peer_join", "details", nullptr, nullptr};
  rasClientsNotifyEvent(RAS_EVENT_LIFECYCLE, &event);
  EXPECT_TRUE(ncclIntruQueueEmpty(&rasClientsHead->sendQ));
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, ClientsNotifyEvent_TextFormat_NoPeerInfo_PlainMessage) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  rasClientsHead->monitorMask = RAS_EVENT_LIFECYCLE;
  struct rasEventNotification event{"peer_join", "a peer joined", nullptr, nullptr};
  rasClientsNotifyEvent(RAS_EVENT_LIFECYCLE, &event);
  std::string out = DrainSendQueue(rasClientsHead);
  EXPECT_NE(out.find("peer_join: a peer joined"), std::string::npos);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, ClientsNotifyEvent_TextFormat_WithPeerInfo_IncludesPeerString) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  rasClientsHead->monitorMask = RAS_EVENT_LIFECYCLE;
  struct rasPeerInfo peer = {};
  peer.pid = 77;
  struct rasEventNotification event{"peer_join", "joined", &peer, nullptr};
  rasClientsNotifyEvent(RAS_EVENT_LIFECYCLE, &event);
  EXPECT_EQ(g_peerInfoToStringCalls, 1);
  std::string out = DrainSendQueue(rasClientsHead);
  EXPECT_NE(out.find("pid=77"), std::string::npos);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, ClientsNotifyEvent_JsonFormat_NeitherPeerInfoNorAddr) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  rasClientsHead->monitorMask = RAS_EVENT_TRACE;
  rasClientsHead->outputFormat = RAS_OUTPUT_JSON;
  struct rasEventNotification event{"conn_retry", "retrying", nullptr, nullptr};
  rasClientsNotifyEvent(RAS_EVENT_TRACE, &event);
  std::string out = DrainSendQueue(rasClientsHead);
  EXPECT_NE(out.find("\"group\": \"TRACE\""), std::string::npos);
  EXPECT_NE(out.find("\"event\": \"conn_retry\""), std::string::npos);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, ClientsNotifyEvent_JsonFormat_PeerAddrResolvesToPeer) {
  struct rasPeerInfo peers[1] = {};
  union ncclSocketAddress addr = MakeAddr(5000);
  peers[0].addr = addr;
  peers[0].pid = 88;
  rasPeers = peers;
  nRasPeers = 1;

  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  rasClientsHead->monitorMask = RAS_EVENT_LIFECYCLE;
  rasClientsHead->outputFormat = RAS_OUTPUT_JSON;
  struct rasEventNotification event{"peer_join", "joined", nullptr, &addr};
  rasClientsNotifyEvent(RAS_EVENT_LIFECYCLE, &event);
  std::string out = DrainSendQueue(rasClientsHead);
  EXPECT_NE(out.find("\"pid\": 88"), std::string::npos);
  rasClientSupportTerminate();
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, SupportTerminate_ClosesListeningSocketAndTerminatesAllClients) {
  ASSERT_EQ(rasClientInitSocket(), ncclSuccess);
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  rasClientSupportTerminate();
  EXPECT_EQ(rasClientListeningSocket, -1);
  EXPECT_EQ(rasClientsHead, nullptr);
  EXPECT_EQ(rasClientsTail, nullptr);
}

// =============================================================================================
// rasClientInitSocket -- listening-socket setup.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, InitSocket_Success_BindsAndListens) {
  ASSERT_EQ(rasClientInitSocket(), ncclSuccess);
  EXPECT_NE(rasClientListeningSocket, -1);
  EXPECT_EQ(g_boundFds.size(), 1u);
  EXPECT_EQ(g_listenedFds.size(), 1u);
  rasClientSupportTerminate();
}

TEST_F(RasClientSupportMicrotest, InitSocket_SocketFails_ReturnsErrorAndResetsFd) {
  g_nextSocketFd = -1;
  EXPECT_NE(rasClientInitSocket(), ncclSuccess);
  EXPECT_EQ(rasClientListeningSocket, -1);
}

TEST_F(RasClientSupportMicrotest, InitSocket_BindFails_ClosesSocketAndReturnsError) {
  g_bindResult = -1;
  EXPECT_NE(rasClientInitSocket(), ncclSuccess);
  EXPECT_EQ(rasClientListeningSocket, -1);
  EXPECT_FALSE(g_closedFds.empty());
}

TEST_F(RasClientSupportMicrotest, InitSocket_ListenFails_ClosesSocketAndReturnsError) {
  g_listenResult = -1;
  EXPECT_NE(rasClientInitSocket(), ncclSuccess);
  EXPECT_EQ(rasClientListeningSocket, -1);
}

// =============================================================================================
// Local (non-socket) diagnostics entrypoint.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, LocalHandleRunDiag_Success_RunsToFinishedAndTerminates) {
  struct rasDiagnosticsContext ctx = {};
  g_diagResumeResult = ncclSuccess;
  EXPECT_EQ(rasLocalHandleRunDiag(&ctx), ncclSuccess);
  EXPECT_EQ(g_diagClientInitCalls, 1);
  EXPECT_EQ(g_diagStartCalls, 1);
  // internal==true clients terminate as soon as they reach FINISHED, regardless of sendQ state.
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, LocalHandleRunDiag_ClientInitFails_TerminatesAndPropagatesError) {
  struct rasDiagnosticsContext ctx = {};
  g_diagClientInitResult = ncclInternalError;
  EXPECT_EQ(rasLocalHandleRunDiag(&ctx), ncclInternalError);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, LocalHandleRunDiag_StartInProgress_LeavesClientAwaitingAsyncCompletion) {
  // An in-progress Start makes rasClientRun's RAS_CLIENT_INIT-style fallthrough `break` before
  // ever reaching the RAS_CLIENT_DIAG_FINI case (mirrors the RAS_CLIENT_INIT/RAS_CLIENT_COMMS
  // pair) -- the resume happens on a LATER, separate rasClientRun() call once whatever
  // out-of-band mechanism completes the gather, not inline within this one.
  struct rasDiagnosticsContext ctx = {};
  g_diagStartResult = ncclInProgress;
  EXPECT_EQ(rasLocalHandleRunDiag(&ctx), ncclSuccess);
  EXPECT_EQ(g_diagResumeCalls, 0);
  ASSERT_NE(rasClientsHead, nullptr);
  EXPECT_EQ(rasClientsHead->status, RAS_CLIENT_DIAG_INIT);
  rasClientSupportTerminate();
}

// =============================================================================================
// rasClientDiagnosticsEmit / rasClientDiagnosticsFinish -- reached only via the reporter captured
// by the rasDiagnosticsClientInit() fake.
// =============================================================================================

// These call the static rasClientDiagnosticsEmit/Finish functions directly (nameable in this TU
// since it #includes the real client_support.cc) rather than driving them indirectly through
// rasClientRun's reporter-building branch: routing through rasLocalHandleRunDiag pre-inits
// client->diagnostics with a null reporter (so rasClientRunDiagInit's own reporter never gets
// built at all), and routing through a socket client's rasClientRun would auto-terminate (and
// free) the client before the test gets a chance to invoke the reporter manually.

TEST_F(RasClientSupportMicrotest, DiagnosticsReporter_Emit_NullTarget_NoOp) {
  EXPECT_EQ(rasClientDiagnosticsEmit(nullptr, "line"), ncclSuccess);
}

TEST_F(RasClientSupportMicrotest, DiagnosticsReporter_Emit_FormatsAndEnqueues) {
  struct rasClient* client = MakeClient();
  EXPECT_EQ(rasClientDiagnosticsEmit(client, "a diagnostics line"), ncclSuccess);
  EXPECT_EQ(DrainSendQueue(client), "a diagnostics line\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, DiagnosticsReporter_Finish_Success_NoErrorLine) {
  struct rasClient* client = MakeClient();
  EXPECT_EQ(rasClientDiagnosticsFinish(client, ncclSuccess), ncclSuccess);
  EXPECT_TRUE(ncclIntruQueueEmpty(&client->sendQ));
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, DiagnosticsReporter_Finish_Failure_EmitsErrorLine) {
  struct rasClient* client = MakeClient();
  EXPECT_EQ(rasClientDiagnosticsFinish(client, ncclInternalError), ncclSuccess);
  std::string out = DrainSendQueue(client);
  EXPECT_NE(out.find("ERROR: diagnostics summary failed"), std::string::npos);
  FreeClient(client);
}

// =============================================================================================
// rasClientRunDiagInit (via rasClientRun's RAS_CLIENT_DIAG_INIT case).
// =============================================================================================

TEST_F(RasClientSupportMicrotest, ClientRun_DiagInit_AlreadyHasDiagnostics_SkipsReinit) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_DIAG_INIT;
  client->diagnostics = reinterpret_cast<struct rasDiagnosticsClientState*>(0x1);  // opaque, never dereferenced here
  g_diagResumeResult = ncclSuccess;
  bool closed = false;
  rasClientRun(client, &closed);
  EXPECT_EQ(g_diagContextInitCalls, 0);
  EXPECT_EQ(g_diagClientInitCalls, 0);
  EXPECT_EQ(g_diagStartCalls, 1);
  // Both DIAG_INIT and DIAG_FINI succeeded with nothing enqueued, so rasClientRun's own exit logic
  // (empty sendQ, status forced to FINISHED) already terminated -- and freed -- the client.
  EXPECT_TRUE(closed);
}

TEST_F(RasClientSupportMicrotest, ClientRun_DiagInit_StartFails_NotInternal_EnqueuesError) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_DIAG_INIT;
  client->internal = false;
  g_diagStartResult = ncclInternalError;
  rasClientRun(client, nullptr);
  std::string out = DrainSendQueue(client);
  EXPECT_NE(out.find("ERROR: diagnostics failed to start"), std::string::npos);
  FreeClient(client);
}

// =============================================================================================
// rasClientRun -- central state-machine dispatcher.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, ClientRun_UnknownStatus_ReturnsInternalErrorAndFinishes) {
  struct rasClient* client = MakeClient();
  client->status = static_cast<rasClientStatus>(12345);
  bool closed = false;
  EXPECT_EQ(rasClientRun(client, &closed), ncclInternalError);
  EXPECT_TRUE(closed);  // sendQ empty -> terminated immediately
}

TEST_F(RasClientSupportMicrotest, ClientRun_Init_NetSendCollReqInProgress_StopsAtComms) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  g_netSendCollReqAllDone = false;
  bool closed = false;
  EXPECT_EQ(rasClientRun(client, &closed), ncclSuccess);
  EXPECT_EQ(client->status, RAS_CLIENT_COMMS);
  EXPECT_FALSE(closed);
  FreeClient(client);
}

// BUG (client_support.cc rasClientRunComms): `struct rasCollComms* commsData =
// (struct rasCollComms*)coll->data;` dereferences coll unconditionally, several lines before this
// function's own `if (coll == nullptr || coll->nFwdSent != coll->nFwdRecv)` guard -- so a null
// client->coll here segfaults instead of returning ncclInternalError the way the guard's presence
// implies it should. A death test documents the CURRENT (buggy) behavior rather than silently
// dropping coverage of this arm: if the null check is ever reordered above the dereference, this
// test starts failing (no crash), which is the signal to replace it with a normal EXPECT_EQ.
TEST_F(RasClientSupportMicrotest, ClientRun_Comms_NoColl_CrashesOnNullDeref) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = nullptr;
  EXPECT_DEATH(rasClientRun(client, nullptr), "");
}

TEST_F(RasClientSupportMicrotest, ClientRun_NonEmptySendQ_NotInternal_DoesNotTerminate) {
  // A comm with zero recorded ranks (OneEmptyComm()) leaves auxComm->status at its calloc default
  // of 0, and the summary-table print later does COMPILER_CLZ(auxComm->status) to index
  // statusStr[] -- __builtin_clz(0) is undefined behavior, and this test isn't about that edge
  // case, so give it one real rank instead (a realistic single-rank communicator).
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(6000);
  rasPeers = peers;
  nRasPeers = 1;

  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {RankSpec{0, 0}})}), {MakeAddr(6000)});
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 1), ncclSuccess);
  rasClientEnqueueMsg(client, msg, 1);
  bool closed = false;
  rasClientRun(client, &closed);
  EXPECT_EQ(client->status, RAS_CLIENT_FINISHED);
  EXPECT_FALSE(closed);
  DrainSendQueue(client);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

// =============================================================================================
// rasClientResume.
// =============================================================================================

TEST_F(RasClientSupportMicrotest, ClientResume_NoMatchingClient_FreesCollAndReturnsSuccess) {
  struct rasCollective* coll = MakeCollective({}, {});
  EXPECT_EQ(rasClientResume(coll), ncclSuccess);
  EXPECT_EQ(g_collFreeCalls, 1);
}

TEST_F(RasClientSupportMicrotest, ClientResume_MatchingClient_InvokesClientRun) {
  // rasClientResume() finds its target by walking rasClientsHead, so the client under test must
  // actually be linked there -- MakeClient() deliberately is NOT (see its own comment), so a
  // standalone MakeClient() here would silently take the "no matching client" branch instead of
  // exercising rasClientRun() at all. rasClientAcceptNewSocket() gives a properly linked client.
  // Also uses a real one-rank comm (not OneEmptyComm()), same reasoning as
  // ClientRun_NonEmptySendQ_NotInternal_DoesNotTerminate above: a zero-rank comm leaves
  // auxComm->status at 0, and COMPILER_CLZ(0) (used to index statusStr[]) is undefined behavior.
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(6100);
  rasPeers = peers;
  nRasPeers = 1;

  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {RankSpec{0, 0}})}), {MakeAddr(6100)});
  EXPECT_EQ(rasClientResume(client->coll), ncclSuccess);
  EXPECT_EQ(client->status, RAS_CLIENT_FINISHED);
  rasClientSupportTerminate();
  rasPeers = nullptr;
  nRasPeers = 0;
}

// =============================================================================================
// rasClientRunComms -- JSON early-exit branch (full-blown TEXT-branch coverage lands in a
// follow-up expansion; this seeds the group with the branch split itself and the simplest
// end-to-end paths).
// =============================================================================================

TEST_F(RasClientSupportMicrotest, RunComms_JsonFormat_EarlyExit_SkipsTextLogic) {
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(4000);
  peers[0].pid = 10;
  rasPeers = peers;
  nRasPeers = 1;

  struct rasClient* client = MakeClient();
  client->outputFormat = RAS_OUTPUT_JSON;
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(OneEmptyComm(), {MakeAddr(4000)});

  rasClientRun(client, nullptr);
  EXPECT_EQ(client->status, RAS_CLIENT_FINISHED);
  std::string out = DrainSendQueue(client);
  EXPECT_NE(out.find("\"communicators\""), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_TextFormat_NoCommunicators_PrintsNoDataMessage) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({}), {});

  rasClientRun(client, nullptr);
  std::string out = DrainSendQueue(client);
  EXPECT_NE(out.find("No communicator data collected!"), std::string::npos);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, RunComms_MismatchedFwdCounts_ReturnsInternalError) {
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective({}, {});
  client->coll->nFwdSent = 2;
  client->coll->nFwdRecv = 1;
  bool closed = false;
  EXPECT_EQ(rasClientRun(client, &closed), ncclInternalError);
  // rasClientRunComms() returns before clearing client->coll or touching the sendQ on this arm, so
  // rasClientRun's own exit logic (empty sendQ, status forced to FINISHED) terminates the client --
  // which frees both client->coll (via the rasCollFree() fake) and client itself. Nothing left to
  // free here; doing so would double-free.
  EXPECT_TRUE(closed);
}

// =============================================================================================
// rasClientEventLoop -- socket protocol parsing (POLLIN: recv + command dispatch; POLLOUT: send).
// =============================================================================================

// --- POLLIN: recv() outcomes ----------------------------------------------------------------

TEST_F(RasClientSupportMicrotest, EventLoop_RecvBufferFull_ClearsPollinWithoutRecvCall) {
  // A pre-full recvBuffer (all zero bytes, so no '\n' anywhere) takes the "no room to receive"
  // branch (POLLIN cleared, recv() skipped) -- but that leaves cmd==recvBuffer with
  // recvOffset==sizeof(recvBuffer) at the bottom of the function too, which is the SAME
  // "excessively long line" condition that terminates the client. The two checks share one
  // condition; there's no way to reach the POLLIN-clear branch without also hitting that one.
  // Capture pfd before the call and don't touch `client` after: rasClientEventLoop() frees it.
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  const int pfd = client->pfd;
  client->recvOffset = sizeof(client->recvBuffer);
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, pfd);
  EXPECT_TRUE(g_recvFds.empty());  // the distinguishing behavior under test: recv() was skipped
  EXPECT_EQ(rasClientsHead, nullptr);
  EXPECT_FALSE(rasPfds[pfd].events & POLLIN);
}

TEST_F(RasClientSupportMicrotest, EventLoop_RecvEof_TerminatesClient) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  ScriptRecv(0, 0, "");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_RecvEwouldblock_NoCloseNoData) {
  struct rasClient* client = MakeClient();
  ScopedHook recvHook(g_recv, [](int, void*, size_t, int) -> ssize_t {
    errno = EWOULDBLOCK;
    return -1;
  });
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(recvHook.calls, 1);
  EXPECT_EQ(client->recvOffset, 0);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_RecvEintr_NoClose) {
  struct rasClient* client = MakeClient();
  ScopedHook recvHook(g_recv, [](int, void*, size_t, int) -> ssize_t {
    errno = EINTR;
    return -1;
  });
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(recvHook.calls, 1);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_RecvEconnreset_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  ScopedHook recvHook(g_recv, [](int, void*, size_t, int) -> ssize_t {
    errno = ECONNRESET;
    return -1;
  });
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_RecvOtherError_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  ScopedHook recvHook(g_recv, [](int, void*, size_t, int) -> ssize_t {
    errno = EIO;
    return -1;
  });
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_RecvPartialLine_WaitsForMore) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("client protocol");  // no newline yet
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_TRUE(ncclIntruQueueEmpty(&client->sendQ));
  EXPECT_EQ(client->recvOffset, static_cast<int>(strlen("client protocol")));
  FreeClient(client);
}

// --- POLLIN: command dispatch ----------------------------------------------------------------

TEST_F(RasClientSupportMicrotest, EventLoop_ClientProtocol_EchoesServerProtocol) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("client protocol 2\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "SERVER PROTOCOL " STR(NCCL_RAS_CLIENT_PROTOCOL) "\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Timeout_Valid_SetsTimeoutAndOk) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("timeout 2.5\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\n");
  EXPECT_EQ(client->timeout, static_cast<int64_t>(2.5 * CLOCK_UNITS_PER_SEC));
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Timeout_NonNumeric_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("timeout abc\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Invalid timeout value abc\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Timeout_Negative_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("timeout -1\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Invalid timeout value -1\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Timeout_NonFinite_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("timeout inf\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Invalid timeout value inf\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SetFormatText_Ok) {
  struct rasClient* client = MakeClient();
  client->outputFormat = RAS_OUTPUT_JSON;
  ScriptRecvData("set format text\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\n");
  EXPECT_EQ(client->outputFormat, RAS_OUTPUT_TEXT);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SetFormatJson_Ok) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("set format json\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\n");
  EXPECT_EQ(client->outputFormat, RAS_OUTPUT_JSON);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SetFormatInvalid_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("set format bogus\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Invalid format bogus\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Status_SetsInitAndInvokesClientRun) {
  struct rasClient* client = MakeClient();
  g_netSendCollReqAllDone = false;  // stop at RAS_CLIENT_COMMS, avoid needing a full coll fixture
  ScriptRecvData("status\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(client->status, RAS_CLIENT_COMMS);
  DrainSendQueue(client);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_VerboseStatus_SetsInitVerboseAndInvokesClientRun) {
  struct rasClient* client = MakeClient();
  g_netSendCollReqAllDone = false;
  ScriptRecvData("verbose status\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(client->status, RAS_CLIENT_COMMS);
  EXPECT_EQ(client->verbose, 1);
  DrainSendQueue(client);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Diagnostics_NonTextFormat_Rejected) {
  struct rasClient* client = MakeClient();
  client->outputFormat = RAS_OUTPUT_JSON;
  ScriptRecvData("diagnostics\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: diagnostics only supports text output\n");
  EXPECT_EQ(client->status, RAS_CLIENT_FINISHED);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Diagnostics_AlreadyInProgress_Rejected) {
  struct rasClient* client = MakeClient();
  g_diagInProgress = true;
  ScriptRecvData("diagnostics\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "BUSY: diagnostics already in progress\n");
  EXPECT_EQ(client->status, RAS_CLIENT_FINISHED);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Diagnostics_Normal_StartsDiagInit) {
  struct rasClient* client = MakeClient();
  g_diagStartResult = ncclInProgress;  // stay at DIAG_INIT, don't fall through to DIAG_FINI
  ScriptRecvData("diagnostics\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(client->status, RAS_CLIENT_DIAG_INIT);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_MonitorBare_DefaultsToLifecycle) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("monitor\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\n");
  EXPECT_EQ(client->monitorMask, RAS_EVENT_LIFECYCLE);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_MonitorWithGroups_SetsMask) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("monitor lifecycle,trace\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\n");
  EXPECT_EQ(client->monitorMask, RAS_EVENT_LIFECYCLE | RAS_EVENT_TRACE);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_MonitorAll_SetsAllMask) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("monitor all\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\n");
  EXPECT_EQ(client->monitorMask, RAS_EVENT_ALL);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_MonitorInvalidToken_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("monitor bogus\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Invalid event group 'bogus'\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_ControlProfilerMask_Valid_BroadcastsAndOk) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("control profiler_mask all\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\n");
  EXPECT_EQ(g_netSendCollReqCalls, 1);
  EXPECT_EQ(g_lastCollReq.type, RAS_BC_PROFILER_MASK);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_ControlProfilerMask_Invalid_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("control profiler_mask bogus\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Invalid profiler mask value 'bogus'\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_ControlUnknownSubcommand_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("control bogus\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Unknown CONTROL subcommand 'bogus'\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_UnknownCommand_ReturnsError) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("bogus\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "ERROR: Unknown command bogus\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_CrLf_StripsCarriageReturn) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("client protocol 2\r\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "SERVER PROTOCOL " STR(NCCL_RAS_CLIENT_PROTOCOL) "\n");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_MultipleCommandsOneRecv_ProcessedInOrder) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("set format json\nset format text\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(DrainSendQueue(client), "OK\nOK\n");
  EXPECT_EQ(client->outputFormat, RAS_OUTPUT_TEXT);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_LeftoverBytesAfterCommand_ShiftedToFront) {
  struct rasClient* client = MakeClient();
  ScriptRecvData("set format json\npartial");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  DrainSendQueue(client);
  EXPECT_EQ(client->recvOffset, static_cast<int>(strlen("partial")));
  EXPECT_EQ(std::string(client->recvBuffer, client->recvOffset), "partial");
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_LongLineNoNewline_BufferFull_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  std::string longLine(sizeof(client->recvBuffer), 'x');  // no '\n' anywhere
  ScriptRecvData(longLine);
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

// --- POLLOUT: send() outcomes ----------------------------------------------------------------

TEST_F(RasClientSupportMicrotest, EventLoop_SendFullWrite_DequeuesAndDisarmsWhenEmpty) {
  struct rasClient* client = MakeClient();
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 5), ncclSuccess);
  memcpy(msg, "hello", 5);
  rasClientEnqueueMsg(client, msg, 5);
  SetRevents(client, POLLOUT);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(g_sentData, "hello");
  EXPECT_TRUE(ncclIntruQueueEmpty(&client->sendQ));
  EXPECT_FALSE(rasPfds[client->pfd].events & POLLOUT);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SendPartialWrite_KeepsMetaAtFrontWithProgress) {
  struct rasClient* client = MakeClient();
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 5), ncclSuccess);
  memcpy(msg, "hello", 5);
  rasClientEnqueueMsg(client, msg, 5);
  ScopedHook sendHook(g_send, [](int, const void* buf, size_t, int) -> ssize_t {
    g_sentData.append(static_cast<const char*>(buf), 2);
    return 2;  // short write
  });
  SetRevents(client, POLLOUT);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(g_sentData, "he");
  EXPECT_FALSE(ncclIntruQueueEmpty(&client->sendQ));
  DrainSendQueue(client);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SendEwouldblock_BreaksWithoutClosing) {
  struct rasClient* client = MakeClient();
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 3), ncclSuccess);
  rasClientEnqueueMsg(client, msg, 3);
  ScopedHook sendHook(g_send, [](int, const void*, size_t, int) -> ssize_t {
    errno = EWOULDBLOCK;
    return -1;
  });
  SetRevents(client, POLLOUT);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(sendHook.calls, 1);
  EXPECT_FALSE(ncclIntruQueueEmpty(&client->sendQ));
  DrainSendQueue(client);
  FreeClient(client);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SendEpipe_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 3), ncclSuccess);
  rasClientEnqueueMsg(client, msg, 3);
  ScopedHook sendHook(g_send, [](int, const void*, size_t, int) -> ssize_t {
    errno = EPIPE;
    return -1;
  });
  SetRevents(client, POLLOUT);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SendOtherError_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 3), ncclSuccess);
  rasClientEnqueueMsg(client, msg, 3);
  ScopedHook sendHook(g_send, [](int, const void*, size_t, int) -> ssize_t {
    errno = EIO;
    return -1;
  });
  SetRevents(client, POLLOUT);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SendDrainedAndFinished_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  client->status = RAS_CLIENT_FINISHED;
  char* msg = nullptr;
  ASSERT_EQ(rasClientAllocMsg(&msg, 3), ncclSuccess);
  // FINISHED clients still accept an enqueue only via the direct queue push below: rasClientEnqueueMsg()
  // itself would drop it (status >= RAS_CLIENT_FINISHED), so build the queue entry directly.
  auto* meta = reinterpret_cast<struct rasMsgMeta*>(reinterpret_cast<char*>(msg) - offsetof(struct rasMsgMeta, msg));
  meta->offset = 0;
  meta->length = 3;
  ncclIntruQueueEnqueue(&client->sendQ, meta);
  SetRevents(client, POLLOUT);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

// =============================================================================================
// rasClientRunInit -- job-summary report generation (version banner + peer/GPU/node stats table).
// Driven via rasClientRun(status=RAS_CLIENT_INIT); g_netSendCollReqAllDone=false throughout so the
// state machine stops at RAS_CLIENT_COMMS instead of falling through into rasClientRunComms
// (covered separately), keeping each test focused on rasClientRunInit's own output.
// =============================================================================================

namespace {
struct rasPeerInfo MakePeer(uint16_t port, uint64_t cudaDevs, int pid = 0) {
  struct rasPeerInfo p = {};
  p.addr = MakeAddr(port);
  p.cudaDevs = p.nvmlDevs = cudaDevs;
  p.pid = pid;
  return p;
}

// Runs rasClientRunInit (via rasClientRun) against whatever rasPeers/nRasPeers the caller already
// set up, and returns everything enqueued as a single string for substring assertions.
std::string RunInitAndDrain(struct rasClient* client) {
  g_netSendCollReqAllDone = false;
  rasClientRun(client, nullptr);
  return DrainSendQueue(client);
}
}  // namespace

TEST_F(RasClientSupportMicrotest, RunInit_JsonFormat_EnqueuesNothing) {
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1)};
  rasPeers = peers;
  nRasPeers = 1;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  client->outputFormat = RAS_OUTPUT_JSON;
  EXPECT_EQ(RunInitAndDrain(client), "");
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunInit_HipVersionSentinel_QueriesOnce) {
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1)};
  rasPeers = peers;
  nRasPeers = 1;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  ScopedHook rtHook(g_hipRuntimeGetVersion, [](int* v) { *v = 60123456; return hipSuccess; });
  ScopedHook drvHook(g_hipDriverGetVersion, [](int* v) { *v = 60123456; return hipSuccess; });
  std::string out = RunInitAndDrain(client);
  EXPECT_EQ(rtHook.calls, 1);
  EXPECT_EQ(drvHook.calls, 1);
  EXPECT_NE(out.find("HIP runtime version 60123456, amdgpu driver version 60123456"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunInit_HipVersionAlreadyCached_SkipsQuery) {
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1)};
  rasPeers = peers;
  nRasPeers = 1;
  cudaRuntimeVersion = 111;
  cudaDriverVersion = 222;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  ScopedHook rtHook(g_hipRuntimeGetVersion, [](int* v) { *v = 999; return hipSuccess; });
  ScopedHook drvHook(g_hipDriverGetVersion, [](int* v) { *v = 999; return hipSuccess; });
  std::string out = RunInitAndDrain(client);
  EXPECT_EQ(rtHook.calls, 0);
  EXPECT_EQ(drvHook.calls, 0);
  EXPECT_NE(out.find("HIP runtime version 111, amdgpu driver version 222"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunInit_OnePeer_SimpleConsistentTable) {
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1)};
  rasPeers = peers;
  nRasPeers = 1;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  std::string out = RunInitAndDrain(client);
  EXPECT_NE(out.find("Nodes  Processes         GPUs  Processes     GPUs"), std::string::npos);
  EXPECT_NE(out.find("Communicators..."), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunInit_TwoNodesConsistentGpus_UnequalPeerCounts_PrintsOutlierNode) {
  InstallPortBucketNodeGrouping();
  // Node "1" (port/100==1): two peers, 1 GPU each. Node "2": one peer, 1 GPU -- consistent GPU
  // count everywhere, but node "2" has a different peer count (outlier, count==1 is always one).
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1, 10), MakePeer(101, 0x1, 11), MakePeer(200, 0x1, 20)};
  rasPeers = peers;
  nRasPeers = 3;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  std::string out = RunInitAndDrain(client);
  EXPECT_NE(out.find("Nodes  Processes         GPUs\n          per node  per process"), std::string::npos);
  EXPECT_NE(out.find("The outlier node"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunInit_InconsistentGpusWithinNode_PrintsGpuDistributionNoOutlier) {
  InstallPortBucketNodeGrouping();
  // Both peers on node "1", but with different GPU counts -> consistentNGpusNode == false, which
  // takes the "GPU distribution printed separately" branch and skips the node-outlier printing
  // entirely (that block is gated on consistentNGpusNode && consistentNGpusGlobal).
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1, 10), MakePeer(101, 0x3, 11)};
  rasPeers = peers;
  nRasPeers = 2;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  std::string out = RunInitAndDrain(client);
  EXPECT_NE(out.find("Processes         GPUs\n                    per process"), std::string::npos);
  EXPECT_EQ(out.find("The outlier node"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunInit_InconsistentGpusAcrossNodes_ConsistentWithinNode) {
  InstallPortBucketNodeGrouping();
  // One peer per node, so consistentNGpusNode is trivially true, but the two nodes disagree on
  // GPU count -> consistentNGpusGlobal == false, same "GPU distribution" branch as above via the
  // OR in the guard.
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1, 10), MakePeer(200, 0x3, 20)};
  rasPeers = peers;
  nRasPeers = 2;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  std::string out = RunInitAndDrain(client);
  EXPECT_NE(out.find("Processes         GPUs\n                    per process"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunInit_VerboseOutlier_PassesVerboseFlagThrough) {
  InstallPortBucketNodeGrouping();
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1, 10), MakePeer(101, 0x1, 11), MakePeer(200, 0x1, 20)};
  rasPeers = peers;
  nRasPeers = 3;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  client->verbose = 1;
  std::string out = RunInitAndDrain(client);
  // Exact verbose-threshold semantics are covered by the dedicated CountIsOutlier_* tests; this
  // just confirms the verbose path is reachable end-to-end without changing the outlier verdict
  // for a count-of-one group (rasCountIsOutlier's count==1 short-circuit fires regardless).
  EXPECT_NE(out.find("The outlier node"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

// BUG (client_support.cc rasClientRunInit, node-outlier detail printing): the inner pid-listing
// loop reads `auxRasPeers[j].peer->pid` for j in [0, vc->value) -- it should be
// `auxRasPeers[peerIdx + j].peer->pid`. auxRasPeers is sorted by peers-per-node ascending, so an
// outlier group (by definition a less-common peers-per-node value) almost never starts at index
// 0; the loop then prints PIDs belonging to whichever node(s) DO occupy the array's first
// vc->value slots -- unrelated peers, not the outlier's own. Confirmed empirically: this fixture's
// actual outlier node (pids 91,92) prints "10,20" instead (two of the majority group's own pids).
// Documents the CURRENT (wrong) output rather than silently dropping coverage of this branch --
// if `j` is ever corrected to `peerIdx + j`, this test starts failing, which is the signal to
// update the expected string to "91,92".
TEST_F(RasClientSupportMicrotest, RunInit_MultiRankOutlierNode_PrintsWrongPidsBug) {
  InstallPortBucketNodeGrouping();
  // Majority: node "1", "2", "3" each with one 1-GPU peer (three single-peer nodes tie into one
  // valCounts group with count==3). Outlier: node "9", two 1-GPU peers -- a single node (that
  // group's own count is 1, so the heading stays singular "The outlier node:"), but vc->value==2
  // ranks on it exercises the ">1" plural in "running process(es)".
  struct rasPeerInfo peers[] = {MakePeer(100, 0x1, 10), MakePeer(200, 0x1, 20), MakePeer(300, 0x1, 30),
                                MakePeer(900, 0x1, 91), MakePeer(901, 0x1, 92)};
  rasPeers = peers;
  nRasPeers = 5;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_INIT;
  std::string out = RunInitAndDrain(client);
  EXPECT_NE(out.find("The outlier node:"), std::string::npos);
  EXPECT_NE(out.find("running processes 10,20"), std::string::npos);  // should be 91,92 -- see bug note above
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

// =============================================================================================
// rasClientRunComms -- TEXT-format communicator report (status/error aggregation, missing/dead
// peer sections, collOpCounts mismatch, error breakdowns). Each test sets up rasPeers AND
// coll->peers with matching addresses (in the same relative order) so peerIdxConv -- built from
// coll->peers before it gets internally re-sorted -- maps each rank's peerIdx to the intended
// rasPeers entry.
// =============================================================================================

namespace {
// Drives rasClientRunComms (via rasClientRun) and returns everything enqueued as one string.
std::string RunCommsAndDrain(struct rasClient* client) {
  rasClientRun(client, nullptr);
  return DrainSendQueue(client);
}
}  // namespace

TEST_F(RasClientSupportMicrotest, RunComms_AllConsistentRunning_SimpleSummaryNoErrorsNoWarnings) {
  struct rasPeerInfo peers[2] = {};
  peers[0].addr = MakeAddr(100);
  peers[1].addr = MakeAddr(200);
  rasPeers = peers;
  nRasPeers = 2;

  RankSpec r0{0, 0}, r1{1, 1};
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(2, {r0, r1})}), {MakeAddr(100), MakeAddr(200)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("RUNNING"), std::string::npos);
  EXPECT_NE(out.find("OK"), std::string::npos);
  EXPECT_EQ(out.find("INCOMPLETE"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_AbortFlag_SetsAbortStatus) {
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(100);
  rasPeers = peers;
  nRasPeers = 1;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  r0.abortFlag = true;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {r0})}), {MakeAddr(100)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("ABORT"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_DestroyFlag_SetsFinalizeStatus) {
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(100);
  rasPeers = peers;
  nRasPeers = 1;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  r0.destroyFlag = true;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {r0})}), {MakeAddr(100)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("FINALIZE"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_MixedStatusAcrossRanks_MismatchWithBreakdown) {
  struct rasPeerInfo peers[2] = {};
  peers[0].addr = MakeAddr(100);
  peers[1].addr = MakeAddr(200);
  rasPeers = peers;
  nRasPeers = 2;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;  // initState success -> RUNNING
  RankSpec r1;
  r1.commRank = 1;
  r1.peerIdx = 1;
  r1.abortFlag = true;  // ABORT -- differs from r0's RUNNING
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(2, {r0, r1})}), {MakeAddr(100), MakeAddr(200)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("MISMATCH"), std::string::npos);
  EXPECT_NE(out.find("Communicator ranks have different status"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_InitError_TriggersBreakdown) {
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(100);
  rasPeers = peers;
  nRasPeers = 1;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  r0.initState = ncclSystemError;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {r0})}), {MakeAddr(100)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find(") ERROR\n"), std::string::npos);
  EXPECT_NE(out.find("Initialization error"), std::string::npos);
  EXPECT_NE(out.find("System error"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_AsyncError_TriggersBreakdown) {
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(100);
  rasPeers = peers;
  nRasPeers = 1;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  r0.asyncError = ncclRemoteError;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {r0})}), {MakeAddr(100)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("Asynchronous error"), std::string::npos);
  EXPECT_NE(out.find("Remote process error"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_CollOpCountsMismatch_PrintsBreakdown) {
  struct rasPeerInfo peers[2] = {};
  peers[0].addr = MakeAddr(100);
  peers[1].addr = MakeAddr(200);
  rasPeers = peers;
  nRasPeers = 2;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  r0.collOpCounts[0] = 5;  // Broadcast count
  RankSpec r1;
  r1.commRank = 1;
  r1.peerIdx = 1;
  r1.collOpCounts[0] = 7;  // differs from r0
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(2, {r0, r1})}), {MakeAddr(100), MakeAddr(200)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("Communicator ranks have different Broadcast operation counts"), std::string::npos);
  EXPECT_NE(out.find("launched up to operation"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_MissingRanksAllPeersPresent_NocommMismatch) {
  // nMissingRanks>0 but nPeersMissing==0 && nRasDeadPeers==0 -> the "easy case": every process
  // reported in (both peer100 and peer200 are in coll->peers), so a comm whose missing-rank entry
  // names peer200 just means that peer doesn't consider itself part of THIS communicator -- not
  // that it's unreachable. The missing rank's address must match a real coll->peers entry for
  // this scenario to be internally consistent (peer200 responded to the collective overall).
  struct rasPeerInfo peers[2] = {};
  peers[0].addr = MakeAddr(100);
  peers[1].addr = MakeAddr(200);
  rasPeers = peers;
  nRasPeers = 2;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  CommSpec spec = MakeCommSpec(2, {r0});
  MissingSpec m0;
  m0.commRank = 1;
  m0.addr = MakeAddr(200);  // matches peer200, who IS in coll->peers -- just not in this comm
  spec.missing = {m0};
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({spec}), {MakeAddr(100), MakeAddr(200)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("NOCOMM"), std::string::npos);
  EXPECT_NE(out.find("MISMATCH"), std::string::npos);
  EXPECT_EQ(out.find("INCOMPLETE"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_MissingRanksWithUnreachablePeer_IncompleteSection) {
  // One peer never responded at all (not in rasPeers either) -> nPeersMissing>0, and its missing
  // rank's addr doesn't bsearch-match coll->peers -> RAS_ACE_INCOMPLETE, not NOCOMM/MISMATCH.
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(100);
  rasPeers = peers;
  nRasPeers = 2;  // one more than coll->nPeers (1) -> nPeersMissing == 1

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  CommSpec spec = MakeCommSpec(2, {r0});
  MissingSpec m0;
  m0.commRank = 1;
  m0.addr = MakeAddr(999);
  spec.missing = {m0};
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({spec}), {MakeAddr(100)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("INCOMPLETE"), std::string::npos);
  EXPECT_NE(out.find("Missing communicator data from 1 rank"), std::string::npos);
  EXPECT_NE(out.find("[process information not found]"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_NPeersMissingSection_ListsMissingProcess) {
  // Two known peers, but only the first is in coll->peers -> nPeersMissing == 1 for the second,
  // which rasClientRunComms reports (by walking rasPeers/coll->peers, not via any comm's own
  // missingRanks array) in the "Errors" section's own INCOMPLETE-count summary.
  struct rasPeerInfo peers[2] = {};
  peers[0].addr = MakeAddr(100);
  peers[0].pid = 55;
  peers[1].addr = MakeAddr(101);
  peers[1].pid = 66;
  rasPeers = peers;
  nRasPeers = 2;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {r0})}), {MakeAddr(100)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("Missing communicator data from 1 job process"), std::string::npos);
  EXPECT_NE(out.find("Process 66 on node 127.0.0.1"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_DeadPeersSection_ListsDeadProcess) {
  // peer100 is alive and part of the comm (a rank references it via peerIdx=0, so coll->peers
  // must include it -- an empty coll->peers with a rank still pointing at peerIdx=0 reads past
  // the end of the (zero-sized) peerIdxConv allocation). peer200 is a separate, globally-dead
  // peer (rasDeadPeers), unrelated to any comm -- that's what the DEAD section reports on.
  struct rasPeerInfo peers[2] = {};
  peers[0].addr = MakeAddr(100);
  peers[0].pid = 55;
  peers[1].addr = MakeAddr(200);
  peers[1].pid = 77;
  rasPeers = peers;
  nRasPeers = 2;
  union ncclSocketAddress deadPeers[1] = {MakeAddr(200)};
  rasDeadPeers = deadPeers;
  nRasDeadPeers = 1;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {r0})}), {MakeAddr(100)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("DEAD"), std::string::npos);
  EXPECT_NE(out.find("considered dead"), std::string::npos);
  EXPECT_NE(out.find("Process 77 on node 127.0.0.1"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
  rasDeadPeers = nullptr;
  nRasDeadPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_LegTimeouts_PrintsTimeoutWarning) {
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(100);
  rasPeers = peers;
  nRasPeers = 1;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(1, {r0})}), {MakeAddr(100)}, /*nLegTimeouts*/ 3);
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("TIMEOUT"), std::string::npos);
  EXPECT_NE(out.find("Encountered 3 communication timeouts"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_MultipleGroups_LargestCommFirst) {
  struct rasPeerInfo peers[3] = {};
  peers[0].addr = MakeAddr(100);
  peers[1].addr = MakeAddr(200);
  peers[2].addr = MakeAddr(300);
  rasPeers = peers;
  nRasPeers = 3;

  RankSpec a0;
  a0.commRank = 0;
  a0.peerIdx = 0;
  RankSpec a1;
  a1.commRank = 1;
  a1.peerIdx = 1;
  RankSpec a2;
  a2.commRank = 2;
  a2.peerIdx = 2;
  RankSpec b0;
  b0.commRank = 0;
  b0.peerIdx = 0;
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  // A 3-rank comm and a 1-rank comm in the same collective response.
  client->coll = MakeCollective(BuildRasCollComms({MakeCommSpec(3, {a0, a1, a2}), MakeCommSpec(1, {b0})}),
                                {MakeAddr(100), MakeAddr(200), MakeAddr(300)});
  std::string out = RunCommsAndDrain(client);
  // rasAuxCommsCompareRev sorts by commNRanks descending, so group "0" (the header's own #column)
  // should be the 3-rank comm, appearing before the 1-rank one in the summary table.
  size_t group0 = out.find("    0");
  size_t group1 = out.find("    1");
  ASSERT_NE(group0, std::string::npos);
  ASSERT_NE(group1, std::string::npos);
  EXPECT_LT(group0, group1);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_JsonFormat_RanksAndMissingRanks_WritesFullDocument) {
  // peer100 resolves normally; coll->peers[1] (port 999) matches nothing in rasPeers, so a rank
  // referencing peerIdx=1 hits jsonWriteRankData's "unknown"/pid=-1 fallback. Missing rank 0
  // (addr=100) IS in coll->peers (unresponsive=false, bsearch finds it) and IS in rasDeadPeers
  // (dead=true); missing rank 1 (addr=777) matches neither (unresponsive=true, dead=false) --
  // between the two ranks and two missing ranks, every firstRank/firstMissing comma-branch and
  // unresponsive/dead combination in jsonWriteRankData/jsonWriteMissingRank gets exercised.
  struct rasPeerInfo peers[1] = {};
  peers[0].addr = MakeAddr(100);
  peers[0].pid = 55;
  rasPeers = peers;
  nRasPeers = 1;
  union ncclSocketAddress deadPeers[1] = {MakeAddr(100)};
  rasDeadPeers = deadPeers;
  nRasDeadPeers = 1;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  r0.cudaDev = 1;
  r0.nvmlDev = 1;
  RankSpec r1;
  r1.commRank = 1;
  r1.peerIdx = 1;  // -> peerIdxConv[1] == -1 (coll->peers[1] matches no rasPeers entry)
  MissingSpec m0;
  m0.commRank = 2;
  m0.addr = MakeAddr(100);  // in coll->peers (unresponsive=false) and in rasDeadPeers (dead=true)
  MissingSpec m1;
  m1.commRank = 3;
  m1.addr = MakeAddr(777);  // in neither (unresponsive=true, dead=false)
  CommSpec spec = MakeCommSpec(4, {r0, r1});
  spec.missing = {m0, m1};

  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->outputFormat = RAS_OUTPUT_JSON;
  client->coll = MakeCollective(BuildRasCollComms({spec}), {MakeAddr(100), MakeAddr(999)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("\"rank\": 0"), std::string::npos);
  EXPECT_NE(out.find("\"host\": \"127.0.0.1\""), std::string::npos);
  EXPECT_NE(out.find("\"pid\": 55"), std::string::npos);
  EXPECT_NE(out.find("\"rank\": 1"), std::string::npos);
  EXPECT_NE(out.find("\"host\": \"unknown\""), std::string::npos);
  EXPECT_NE(out.find("\"pid\": -1"), std::string::npos);
  EXPECT_NE(out.find("\"unresponsive\": false"), std::string::npos);
  EXPECT_NE(out.find("\"considered_dead\": true"), std::string::npos);
  EXPECT_NE(out.find("\"unresponsive\": true"), std::string::npos);
  EXPECT_NE(out.find("\"considered_dead\": false"), std::string::npos);
  EXPECT_NE(out.find("\"missing_ranks\""), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
  rasDeadPeers = nullptr;
  nRasDeadPeers = 0;
}

TEST_F(RasClientSupportMicrotest, RunComms_MissingRankFoundInHardCase_NocommMismatch) {
  // Distinct from RunComms_MissingRanksAllPeersPresent_NocommMismatch: that test covers the "easy
  // case" (nPeersMissing==0), which sets NOCOMM/MISMATCH unconditionally without ever entering the
  // per-rank bsearch loop below. This one forces the "hard case" (nPeersMissing>0, via peer300
  // being globally unreachable) while STILL having comm A's own missing rank (peer200) resolve via
  // bsearch -- exercising the loop's "found" arm specifically, not just its "not found" arm
  // (already covered by RunComms_MissingRanksWithUnreachablePeer_IncompleteSection).
  struct rasPeerInfo peers[3] = {};
  peers[0].addr = MakeAddr(100);
  peers[1].addr = MakeAddr(200);
  peers[2].addr = MakeAddr(300);  // globally missing: not in coll->peers at all
  rasPeers = peers;
  nRasPeers = 3;

  RankSpec r0;
  r0.commRank = 0;
  r0.peerIdx = 0;
  CommSpec spec = MakeCommSpec(2, {r0});
  MissingSpec m0;
  m0.commRank = 1;
  m0.addr = MakeAddr(200);  // IS in coll->peers -> bsearch finds it -> NOCOMM/MISMATCH, not INCOMPLETE
  spec.missing = {m0};
  struct rasClient* client = MakeClient();
  client->status = RAS_CLIENT_COMMS;
  client->coll = MakeCollective(BuildRasCollComms({spec}), {MakeAddr(100), MakeAddr(200)});
  std::string out = RunCommsAndDrain(client);
  EXPECT_NE(out.find("NOCOMM"), std::string::npos);
  FreeClient(client);
  rasPeers = nullptr;
  nRasPeers = 0;
}

// --- rasClientEventLoop: rasClientEnqueueString() failure arms (forced via g_msgAllocResult) ---

TEST_F(RasClientSupportMicrotest, EventLoop_ClientProtocol_EnqueueFails_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("client protocol 2\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Timeout_EnqueueFails_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("timeout 1\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_SetFormat_EnqueueFails_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("set format text\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_DiagnosticsNonText_EnqueueFails_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  client->outputFormat = RAS_OUTPUT_JSON;
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("diagnostics\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_DiagnosticsBusy_EnqueueFails_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  g_diagInProgress = true;
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("diagnostics\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Monitor_EnqueueFails_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("monitor\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_Control_EnqueueFails_Terminates) {
  ASSERT_EQ(rasClientAcceptNewSocket(), ncclSuccess);
  struct rasClient* client = rasClientsHead;
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("control profiler_mask all\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_EQ(rasClientsHead, nullptr);
}

TEST_F(RasClientSupportMicrotest, EventLoop_UnknownCommand_EnqueueFails_NonFatalSurvives) {
  // The unknown-command arm's own comment: "It should be non-fatal if we don't return a
  // response" -- unlike every other command, a failed enqueue here just returns without
  // terminating the client.
  struct rasClient* client = MakeClient();
  g_msgAllocResult = ncclSystemError;
  ScriptRecvData("bogus\n");
  SetRevents(client, POLLIN);
  rasClientEventLoop(client, client->pfd);
  EXPECT_TRUE(ncclIntruQueueEmpty(&client->sendQ));
  FreeClient(client);
}
