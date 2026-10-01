/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/ras.cc. The suite executes every source line
// and function; branch-heavy network state machines retain intentionally
// unexercised combinations that require integration coverage. External network,
// peer, collective, client, socket-pair, and event-loop operations are TU-local
// seams because this is their sole test consumer.

#include <poll.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/signature-drift.h"
#include "os_socket_pair.h"
#include "ras/ras_internal.h"
#include "socket.h"

int RasTestPoll(struct pollfd*, nfds_t, int);
int RasTestClose(int);
int RasTestAtexit(void (*)(void));
uint64_t RasTestClockNano();

// Redirect the process-wide APIs used by ras.cc before including that file,
// then restore their real names immediately afterward. Every header that uses
// the symbols below must stay above this guard-less macro block.
#define poll RasTestPoll
#define close RasTestClose
#define atexit RasTestAtexit
#define clockNano RasTestClockNano

namespace {

ncclResult_t DefaultSocketProgress(int, struct ncclSocket*, void*, int size, int* offset, int* closed) {
  *offset = size;
  if (closed) *closed = 0;
  return ncclSuccess;
}

std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, int*)> g_socketProgress =
    DefaultSocketProgress;
std::atomic<bool> g_initComplete{false};

}  // namespace

ASSERT_HOOK_MATCHES_PROD(g_socketProgress, ncclSocketProgress);

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed) {
  return g_socketProgress(op, sock, ptr, size, offset, closed);
}

const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, const int) {
  buf[0] = '\0';
  return buf;
}

#include RAS_CC_PATH

#undef atexit
#undef close
#undef clockNano
#undef poll

namespace {

ncclResult_t g_socketInitResult = ncclSuccess;
ncclResult_t g_socketListenResult = ncclSuccess;
ncclResult_t g_socketPairCreateResult = ncclSuccess;
ncclResult_t g_pairWriteResult = ncclSuccess;
ncclResult_t g_pairReadResult = ncclSuccess;
size_t g_pairChunk = SIZE_MAX;
std::vector<char> g_pairBytes;
size_t g_pairReadPos = 0;
int g_socketPairCloseCalls = 0;
int g_socketCloseCalls = 0;
int g_closeCalls = 0;
int g_atexitCalls = 0;
int g_pollCalls = 0;
std::function<int(struct pollfd*, nfds_t, int)> g_poll = [](struct pollfd*, nfds_t, int) { return 0; };
ASSERT_HOOK_MATCHES_PROD(g_poll, ::poll);
#undef ASSERT_HOOK_MATCHES_PROD
int g_localAddRanksCalls = 0;
rasRankInit* g_localAddRanksLast = nullptr;
int g_localAddRanksLastCount = 0;
ncclResult_t g_localAddRanksResult = ncclSuccess;
int g_cleanupCalls[4] = {};
ncclResult_t g_diagnosticsInitResult = ncclSuccess;
int g_diagnosticsInitCalls = 0;
const ncclComm* g_diagnosticsInitComm = nullptr;
ncclResult_t g_localRunDiagResult = ncclSuccess;
int g_localRunDiagCalls = 0;
int g_profilerMask = -1;
ncclSocketPairDescriptor g_lastPairWriteDescriptor = NCCL_SOCKET_PAIR_INVALID;
ncclSocketPairDescriptor g_lastPairReadDescriptor = NCCL_SOCKET_PAIR_INVALID;
rasDiagnosticsContext g_lastRunDiagContext{};
void ResetWholeFileSeams();

void SetPairNotification(const rasNotification& msg) {
  const char* bytes = reinterpret_cast<const char*>(&msg);
  g_pairBytes.assign(bytes, bytes + sizeof(msg));
  g_pairReadPos = 0;
}

void SetPairNotification(rasNotificationType type) {
  rasNotification msg{};
  msg.type = type;
  SetPairNotification(msg);
}

void InitPollFds(int count) {
  rasPfds = static_cast<pollfd*>(std::calloc(count, sizeof(*rasPfds)));
  ASSERT_NE(nullptr, rasPfds);
  nRasPfds = count;
  for (int i = 0; i < count; ++i) rasPfds[i].fd = NCCL_INVALID_SOCKET;
}

void ExpectCommInitCleanupRan() {
  EXPECT_EQ(1, g_socketPairCloseCalls);
  EXPECT_EQ(1, g_closeCalls);
  EXPECT_EQ(1, g_socketCloseCalls);
  EXPECT_EQ(0, g_atexitCalls);
  EXPECT_FALSE(rasInitialized);
}

class RasMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetWholeFileSeams();
    g_initComplete.store(false, std::memory_order_relaxed);
    g_socketProgress = DefaultSocketProgress;
    rasInitialized = false;
    rasInitRefCount = 0;
    rasNotificationPipe[0] = rasNotificationPipe[1] = NCCL_SOCKET_PAIR_INVALID;
    std::free(ncclComms);
    ncclComms = nullptr;
    nNcclComms = 0;
    ncclCommsSorted = false;
    std::free(rasPfds);
    rasPfds = nullptr;
    nRasPfds = 0;
    std::memset(&rasNetListeningSocket, 0, sizeof(rasNetListeningSocket));
  }

  void TearDown() override {
    if (rasThread.joinable()) rasThread.join();
    std::free(rasPfds);
    rasPfds = nullptr;
    nRasPfds = 0;
    std::free(ncclComms);
    ncclComms = nullptr;
    nNcclComms = 0;
    rasInitialized = false;
    rasInitRefCount = 0;
    g_socketProgress = DefaultSocketProgress;
    ResetWholeFileSeams();
  }
};

}  // namespace

TEST_F(RasMicrotest, CommInitAlreadyInitializedRegistersCommAndCopiesListeningAddress) {
  rasInitialized = true;
  rasNetListeningSocket.addr.sin.sin_family = AF_INET;
  rasNetListeningSocket.addr.sin.sin_port = htons(4321);
  auto comm = std::make_unique<ncclComm>();
  rasRankInit rank{};
  ASSERT_EQ(ncclSuccess, ncclRasCommInit(comm.get(), &rank));
  EXPECT_EQ(1, rasInitRefCount);
  EXPECT_EQ(RAS_INCREMENT * 8, nNcclComms);
  EXPECT_EQ(comm.get(), ncclComms[0]);
  EXPECT_FALSE(ncclCommsSorted);
  EXPECT_EQ(htons(4321), rank.addr.sin.sin_port);
}

TEST_F(RasMicrotest, CommInitAlreadyInitializedAcceptsNullRank) {
  rasInitialized = true;
  auto comm = std::make_unique<ncclComm>();
  ASSERT_EQ(ncclSuccess, ncclRasCommInit(comm.get(), nullptr));
  EXPECT_EQ(1, rasInitRefCount);
  EXPECT_EQ(comm.get(), ncclComms[0]);
}

TEST_F(RasMicrotest, CommInitReusesVacantCommSlot) {
  rasInitialized = true;
  nNcclComms = 2;
  ncclComms = static_cast<ncclComm**>(std::calloc(2, sizeof(*ncclComms)));
  auto incumbent = std::make_unique<ncclComm>();
  auto newcomer = std::make_unique<ncclComm>();
  ncclComms[0] = incumbent.get();
  rasRankInit rank{};
  ASSERT_EQ(ncclSuccess, ncclRasCommInit(newcomer.get(), &rank));
  EXPECT_EQ(newcomer.get(), ncclComms[1]);
  EXPECT_EQ(2, nNcclComms);
}

TEST_F(RasMicrotest, CommInitGrowthPreservesRegisteredComms) {
  rasInitialized = true;
  nNcclComms = 2;
  ncclComms = static_cast<ncclComm**>(std::calloc(2, sizeof(*ncclComms)));
  auto first = std::make_unique<ncclComm>();
  auto second = std::make_unique<ncclComm>();
  auto newcomer = std::make_unique<ncclComm>();
  ncclComms[0] = first.get();
  ncclComms[1] = second.get();
  rasRankInit rank{};
  ASSERT_EQ(ncclSuccess, ncclRasCommInit(newcomer.get(), &rank));
  EXPECT_EQ(first.get(), ncclComms[0]);
  EXPECT_EQ(second.get(), ncclComms[1]);
  EXPECT_EQ(newcomer.get(), ncclComms[2]);
  EXPECT_GT(nNcclComms, 2);
}

TEST_F(RasMicrotest, CommInitSocketInitFailureRunsCleanup) {
  g_socketInitResult = ncclSystemError;
  auto comm = std::make_unique<ncclComm>();
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET;
  EXPECT_EQ(ncclSystemError, ncclRasCommInit(comm.get(), &rank));
  ExpectCommInitCleanupRan();
}

TEST_F(RasMicrotest, CommInitListenFailureRunsCleanup) {
  g_socketListenResult = ncclSystemError;
  auto comm = std::make_unique<ncclComm>();
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET6;
  EXPECT_EQ(ncclSystemError, ncclRasCommInit(comm.get(), &rank));
  ExpectCommInitCleanupRan();
}

TEST_F(RasMicrotest, CommInitSocketPairFailureRunsCleanup) {
  g_socketPairCreateResult = ncclSystemError;
  auto comm = std::make_unique<ncclComm>();
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET;
  EXPECT_EQ(ncclSystemError, ncclRasCommInit(comm.get(), &rank));
  ExpectCommInitCleanupRan();
}

TEST_F(RasMicrotest, CommInitColdSuccessStartsThreadAndPublishesAddress) {
  SetPairNotification(RAS_TERMINATE);
  g_poll = [](pollfd* fds, nfds_t, int) {
    while (!g_initComplete.load(std::memory_order_acquire)) std::this_thread::yield();
    fds[0].revents = POLLIN;
    return 1;
  };

  auto comm = std::make_unique<ncclComm>();
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET6;
  ncclResult_t result = ncclRasCommInit(comm.get(), &rank);
  g_initComplete.store(true, std::memory_order_release);
  ASSERT_EQ(ncclSuccess, result);
  ASSERT_TRUE(rasThread.joinable());
  rasThread.join();
  EXPECT_EQ(1, g_atexitCalls);
  EXPECT_EQ(1, g_pollCalls);
  EXPECT_EQ(0, rasInitRefCount);
  EXPECT_FALSE(rasInitialized);
}

TEST_F(RasMicrotest, CommFiniUninitializedIsNoOp) {
  auto comm = std::make_unique<ncclComm>();
  EXPECT_EQ(ncclSuccess, ncclRasCommFini(comm.get()));
  EXPECT_EQ(0, rasInitRefCount);
}

TEST_F(RasMicrotest, CommFiniRemovesMatchingCommAndDropsReference) {
  rasInitialized = true;
  rasInitRefCount = 1;
  nNcclComms = 2;
  ncclComms = static_cast<ncclComm**>(std::calloc(2, sizeof(*ncclComms)));
  auto first = std::make_unique<ncclComm>();
  auto second = std::make_unique<ncclComm>();
  ncclComms[0] = first.get();
  ncclComms[1] = second.get();
  ncclCommsSorted = true;
  EXPECT_EQ(ncclSuccess, ncclRasCommFini(second.get()));
  EXPECT_EQ(first.get(), ncclComms[0]);
  EXPECT_EQ(nullptr, ncclComms[1]);
  EXPECT_FALSE(ncclCommsSorted);
  EXPECT_EQ(0, rasInitRefCount);
}

TEST_F(RasMicrotest, AddRanksBeforeInitializationIsIgnored) {
  rasRankInit ranks[2]{};
  EXPECT_EQ(ncclSuccess, ncclRasAddRanks(ranks, 2));
  EXPECT_TRUE(g_pairBytes.empty());
}

TEST_F(RasMicrotest, AddRanksWritesCompleteNotificationAcrossPartialWrites) {
  rasInitialized = true;
  rasNotificationPipe[1] = 52;
  g_pairChunk = 3;
  rasRankInit ranks[2]{};
  ASSERT_EQ(ncclSuccess, ncclRasAddRanks(ranks, 2));
  EXPECT_EQ(52, g_lastPairWriteDescriptor);
  ASSERT_EQ(sizeof(rasNotification), g_pairBytes.size());
  const auto* msg = reinterpret_cast<const rasNotification*>(g_pairBytes.data());
  EXPECT_EQ(RAS_ADD_RANKS, msg->type);
  EXPECT_EQ(ranks, msg->addRanks.ranks);
  EXPECT_EQ(2, msg->addRanks.nranks);
}

TEST_F(RasMicrotest, AddRanksPropagatesSocketPairWriteFailure) {
  rasInitialized = true;
  g_pairWriteResult = ncclSystemError;
  rasRankInit rank{};
  EXPECT_EQ(ncclSystemError, ncclRasAddRanks(&rank, 1));
}

TEST_F(RasMicrotest, LocalHandleDispatchesAddRanksAndIgnoresHandlerFailure) {
  rasNotification msg{};
  rasRankInit ranks[2]{};
  msg.type = RAS_ADD_RANKS;
  msg.addRanks.ranks = ranks;
  msg.addRanks.nranks = 2;
  SetPairNotification(msg);
  g_pairChunk = 2;
  g_localAddRanksResult = ncclSystemError;
  bool terminate = false;
  EXPECT_EQ(ncclSuccess, rasLocalHandle(&terminate));
  EXPECT_EQ(1, g_localAddRanksCalls);
  EXPECT_EQ(ranks, g_localAddRanksLast);
  EXPECT_EQ(2, g_localAddRanksLastCount);
  EXPECT_FALSE(terminate);
}

TEST_F(RasMicrotest, LocalHandleTerminateSetsFlag) {
  rasNotification msg{};
  msg.type = RAS_TERMINATE;
  SetPairNotification(msg);
  bool terminate = false;
  EXPECT_EQ(ncclSuccess, rasLocalHandle(&terminate));
  EXPECT_TRUE(terminate);
}

TEST_F(RasMicrotest, LocalHandleEofAndUnknownTypeReturnErrors) {
  bool terminate = false;
  EXPECT_EQ(ncclSystemError, rasLocalHandle(&terminate));

  rasNotification msg{};
  msg.type = static_cast<rasNotificationType>(99);
  SetPairNotification(msg);
  g_pairReadPos = 0;
  EXPECT_EQ(ncclInternalError, rasLocalHandle(&terminate));
}

TEST_F(RasMicrotest, LocalHandlePropagatesSocketPairReadFailure) {
  g_pairReadResult = ncclSystemError;
  bool terminate = false;
  EXPECT_EQ(ncclSystemError, rasLocalHandle(&terminate));
  EXPECT_FALSE(terminate);
}

TEST_F(RasMicrotest, LocalHandleRunsDiagnosticsAndIgnoresHandlerFailure) {
  rasNotificationPipe[0] = 51;
  rasNotification msg{};
  msg.type = RAS_RUN_DIAG;
  msg.runDiag.ctx.hasCommFilter = true;
  SetPairNotification(msg);
  g_localRunDiagResult = ncclSystemError;
  bool terminate = false;
  EXPECT_EQ(ncclSuccess, rasLocalHandle(&terminate));
  EXPECT_EQ(1, g_localRunDiagCalls);
  EXPECT_TRUE(g_lastRunDiagContext.hasCommFilter);
  EXPECT_EQ(rasNotificationPipe[0], g_lastPairReadDescriptor);
  EXPECT_FALSE(terminate);
}

TEST_F(RasMicrotest, ThreadCleanupResetsAllGlobalState) {
  rasInitialized = true;
  rasInitRefCount = 3;
  nNcclComms = 1;
  ncclComms = static_cast<ncclComm**>(std::calloc(1, sizeof(*ncclComms)));
  InitPollFds(1);
  rasThreadCleanup();
  for (int calls : g_cleanupCalls) EXPECT_EQ(1, calls);
  EXPECT_EQ(1, g_socketPairCloseCalls);
  EXPECT_EQ(1, g_socketCloseCalls);
  EXPECT_FALSE(rasInitialized);
  EXPECT_EQ(0, rasInitRefCount);
  EXPECT_EQ(nullptr, ncclComms);
  EXPECT_EQ(0, nNcclComms);
  EXPECT_EQ(nullptr, rasPfds);
  EXPECT_EQ(0, nRasPfds);
}

namespace {

ncclResult_t g_socketGetFdResult = ncclSuccess;
int g_socketGetFdValue = 41;

int g_keepAliveCalls = 0;
int g_peersUpdateCalls = 0;
int g_collReqCalls = 0;
int g_collRespCalls = 0;
ncclResult_t g_dispatchResult = ncclSuccess;
const rasMsg* g_lastDispatchMsg = nullptr;
rasSocket* g_lastDispatchSocket = nullptr;

rasConnection* g_connFindResult = nullptr;
rasConnection g_newConn{};
ncclResult_t g_newConnResult = ncclSuccess;
int g_socketTerminateCalls = 0;
rasSocket* g_lastTerminatedSocket = nullptr;
int g_socketCompareResult = 0;
const void* g_socketCompareLeft = nullptr;
const void* g_socketCompareRight = nullptr;
int g_clientsNotifyCalls = 0;
rasEventGroup g_lastEventGroup = RAS_EVENT_TRACE;
const union ncclSocketAddress* g_lastEventPeerAddr = nullptr;
int g_peerFindResult = -1;
int g_linkUpdateCalls = 0;
std::vector<rasLink*> g_updatedLinks;
std::vector<rasConnection*> g_updatedConnections;
std::vector<int> g_updatedPeerIndices;
int g_sendPeersUpdateCalls = 0;
rasConnection* g_sentPeersUpdateConn = nullptr;
const rasPeerInfo* g_sentPeers = nullptr;
int g_sentPeerCount = -1;
bool g_peerDead = false;
int g_connDisconnectCalls = 0;
int g_peerDeclareDeadCalls = 0;
union ncclSocketAddress g_disconnectedAddr{};
union ncclSocketAddress g_declaredDeadAddr{};
int g_netAcceptCalls = 0;
int g_clientAcceptCalls = 0;
int g_sockEventCalls = 0;
int g_clientEventCalls = 0;
int g_timeoutCalls[4] = {};
int g_lastSockPollIdx = -1;
int g_lastClientPollIdx = -1;
rasSocket* g_lastSockEventSocket = nullptr;
rasClient* g_lastClientEventClient = nullptr;
int64_t g_nextWakeupOverride = 0;
uint64_t g_clockNano = 100 * CLOCK_UNITS_PER_SEC;

void ResetWholeFileSeams() {
  g_socketInitResult = ncclSuccess;
  g_socketListenResult = ncclSuccess;
  g_socketPairCreateResult = ncclSuccess;
  g_socketGetFdResult = ncclSuccess;
  g_socketGetFdValue = 41;
  g_socketCloseCalls = g_socketPairCloseCalls = g_closeCalls = g_atexitCalls = g_pollCalls = 0;
  g_poll = [](struct pollfd*, nfds_t, int) { return 0; };
  g_pairBytes.clear();
  g_pairReadPos = 0;
  g_pairChunk = SIZE_MAX;
  g_pairWriteResult = g_pairReadResult = ncclSuccess;
  g_localAddRanksCalls = 0;
  g_localAddRanksLast = nullptr;
  g_localAddRanksLastCount = 0;
  g_localAddRanksResult = ncclSuccess;
  std::memset(g_cleanupCalls, 0, sizeof(g_cleanupCalls));
  g_diagnosticsInitResult = ncclSuccess;
  g_diagnosticsInitCalls = 0;
  g_diagnosticsInitComm = nullptr;
  g_localRunDiagResult = ncclSuccess;
  g_localRunDiagCalls = 0;
  g_profilerMask = -1;
  g_keepAliveCalls = g_peersUpdateCalls = g_collReqCalls = g_collRespCalls = 0;
  g_dispatchResult = ncclSuccess;
  g_lastDispatchMsg = nullptr;
  g_lastDispatchSocket = nullptr;
  g_connFindResult = nullptr;
  std::memset(&g_newConn, 0, sizeof(g_newConn));
  g_newConnResult = ncclSuccess;
  g_socketTerminateCalls = 0;
  g_lastTerminatedSocket = nullptr;
  g_socketCompareResult = 0;
  g_socketCompareLeft = g_socketCompareRight = nullptr;
  g_clientsNotifyCalls = 0;
  g_lastEventGroup = RAS_EVENT_TRACE;
  g_lastEventPeerAddr = nullptr;
  g_peerFindResult = -1;
  g_linkUpdateCalls = g_sendPeersUpdateCalls = 0;
  g_sentPeersUpdateConn = nullptr;
  g_sentPeers = nullptr;
  g_sentPeerCount = -1;
  g_updatedLinks.clear();
  g_updatedConnections.clear();
  g_updatedPeerIndices.clear();
  g_peerDead = false;
  g_connDisconnectCalls = g_peerDeclareDeadCalls = 0;
  std::memset(&g_disconnectedAddr, 0, sizeof(g_disconnectedAddr));
  std::memset(&g_declaredDeadAddr, 0, sizeof(g_declaredDeadAddr));
  g_netAcceptCalls = g_clientAcceptCalls = 0;
  g_sockEventCalls = g_clientEventCalls = 0;
  std::memset(g_timeoutCalls, 0, sizeof(g_timeoutCalls));
  g_lastSockPollIdx = g_lastClientPollIdx = -1;
  g_lastSockEventSocket = nullptr;
  g_lastClientEventClient = nullptr;
  g_nextWakeupOverride = 0;
  g_clockNano = 100 * CLOCK_UNITS_PER_SEC;
  g_lastPairWriteDescriptor = g_lastPairReadDescriptor = NCCL_SOCKET_PAIR_INVALID;
  std::memset(&g_lastRunDiagContext, 0, sizeof(g_lastRunDiagContext));
  rasSocketsHead = nullptr;
  rasClientsHead = nullptr;
  rasPeersHash = 0;
  rasDeadPeersHash = 0;
  rasPeers = nullptr;
  nRasPeers = 0;
  rasDeadPeers = nullptr;
  nRasDeadPeers = 0;
  rasClientListeningSocket = 43;
}

}  // namespace

int rasClientListeningSocket = 43;
rasSocket* rasSocketsHead = nullptr;
rasClient* rasClientsHead = nullptr;
rasLink rasNextLink{};
rasLink rasPrevLink{};
rasPeerInfo* rasPeers = nullptr;
int nRasPeers = 0;
uint64_t rasPeersHash = 0;
union ncclSocketAddress* rasDeadPeers = nullptr;
int nRasDeadPeers = 0;
uint64_t rasDeadPeersHash = 0;

int RasTestPoll(struct pollfd* fds, nfds_t n, int timeout) {
  ++g_pollCalls;
  return g_poll(fds, n, timeout);
}
int RasTestClose(int) {
  ++g_closeCalls;
  return 0;
}
int RasTestAtexit(void (*)(void)) {
  ++g_atexitCalls;
  return 0;
}
uint64_t RasTestClockNano() { return g_clockNano; }

uint64_t ncclSocketDefaultMagic() { return 0x1234; }
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t,
                            enum ncclSocketType, volatile uint32_t*, int, int) {
  if (g_socketInitResult == ncclSuccess && sock && addr) std::memcpy(&sock->addr, addr, sizeof(*addr));
  return g_socketInitResult;
}
ncclResult_t ncclSocketListen(struct ncclSocket*) { return g_socketListenResult; }
ncclResult_t ncclSocketGetFd(struct ncclSocket*, ncclSocketDescriptor* fd) {
  if (g_socketGetFdResult == ncclSuccess && fd) *fd = g_socketGetFdValue;
  return g_socketGetFdResult;
}
ncclResult_t ncclSocketClose(struct ncclSocket*, bool) {
  ++g_socketCloseCalls;
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairCreate(ncclSocketPairDescriptor pair[2]) {
  if (g_socketPairCreateResult == ncclSuccess) {
    pair[0] = 51;
    pair[1] = 52;
  }
  return g_socketPairCreateResult;
}
ncclResult_t ncclOsSocketPairClose(ncclSocketPairDescriptor pair[2]) {
  ++g_socketPairCloseCalls;
  pair[0] = pair[1] = NCCL_SOCKET_PAIR_INVALID;
  return ncclSuccess;
}
ncclResult_t ncclOsSocketPairWrite(ncclSocketPairDescriptor descriptor, const void* buf, size_t len, size_t* written) {
  g_lastPairWriteDescriptor = descriptor;
  if (g_pairWriteResult != ncclSuccess) return g_pairWriteResult;
  size_t n = std::min(len, g_pairChunk);
  const char* bytes = static_cast<const char*>(buf);
  g_pairBytes.insert(g_pairBytes.end(), bytes, bytes + n);
  *written = n;
  return ncclSuccess;
}
ncclResult_t ncclOsSocketPairRead(ncclSocketPairDescriptor descriptor, void* buf, size_t len, size_t* nread) {
  g_lastPairReadDescriptor = descriptor;
  if (g_pairReadResult != ncclSuccess) return g_pairReadResult;
  size_t available = g_pairBytes.size() - g_pairReadPos;
  size_t n = std::min(std::min(len, available), g_pairChunk);
  if (n) std::memcpy(buf, g_pairBytes.data() + g_pairReadPos, n);
  g_pairReadPos += n;
  *nread = n;
  return ncclSuccess;
}
void ncclSetThreadName(std::thread&, const char*, ...) {}

ncclResult_t rasClientInitSocket() { return ncclSuccess; }
ncclResult_t rasClientAcceptNewSocket() {
  ++g_clientAcceptCalls;
  return ncclSuccess;
}
void rasClientEventLoop(struct rasClient* client, int pollIdx) {
  ++g_clientEventCalls;
  g_lastClientPollIdx = pollIdx;
  g_lastClientEventClient = client;
}
void rasClientSupportTerminate() { ++g_cleanupCalls[0]; }
void rasNetTerminate() { ++g_cleanupCalls[1]; }
void rasCollectivesTerminate() { ++g_cleanupCalls[2]; }
void rasPeersTerminate() { ++g_cleanupCalls[3]; }
ncclResult_t rasDiagnosticsContextInit(struct rasDiagnosticsContext* ctx, const struct ncclComm* comm) {
  ++g_diagnosticsInitCalls;
  g_diagnosticsInitComm = comm;
  if (ctx) std::memset(ctx, 0, sizeof(*ctx));
  return g_diagnosticsInitResult;
}
ncclResult_t rasLocalHandleRunDiag(const struct rasDiagnosticsContext* ctx) {
  ++g_localRunDiagCalls;
  if (ctx) g_lastRunDiagContext = *ctx;
  return g_localRunDiagResult;
}
void ncclProfilerSetRasOverride(int mask) { g_profilerMask = mask; }
int64_t rasTimeoutFactorNs(int64_t baseSeconds) { return baseSeconds * CLOCK_UNITS_PER_SEC; }
void rasSocksHandleTimeouts(int64_t, int64_t* nextWakeup) {
  ++g_timeoutCalls[0];
  if (g_nextWakeupOverride) *nextWakeup = g_nextWakeupOverride;
}
void rasConnsHandleTimeouts(int64_t, int64_t*) { ++g_timeoutCalls[1]; }
void rasNetHandleTimeouts(int64_t, int64_t*) { ++g_timeoutCalls[2]; }
void rasCollsHandleTimeouts(int64_t, int64_t*) { ++g_timeoutCalls[3]; }
ncclResult_t rasNetAcceptNewSocket() {
  ++g_netAcceptCalls;
  return ncclSuccess;
}
void rasSockEventLoop(struct rasSocket* sock, int pollIdx) {
  ++g_sockEventCalls;
  g_lastSockPollIdx = pollIdx;
  g_lastSockEventSocket = sock;
}

ncclResult_t rasLocalHandleAddRanks(struct rasRankInit* ranks, int nranks) {
  ++g_localAddRanksCalls;
  g_localAddRanksLast = ranks;
  g_localAddRanksLastCount = nranks;
  return g_localAddRanksResult;
}
ncclResult_t rasMsgHandleKeepAlive(const struct rasMsg* msg, struct rasSocket* sock) {
  ++g_keepAliveCalls;
  g_lastDispatchMsg = msg;
  g_lastDispatchSocket = sock;
  return g_dispatchResult;
}
ncclResult_t rasMsgHandlePeersUpdate(struct rasMsg* msg, struct rasSocket* sock) {
  ++g_peersUpdateCalls;
  g_lastDispatchMsg = msg;
  g_lastDispatchSocket = sock;
  return g_dispatchResult;
}
ncclResult_t rasMsgHandleCollReq(struct rasMsg* msg, struct rasSocket* sock) {
  ++g_collReqCalls;
  g_lastDispatchMsg = msg;
  g_lastDispatchSocket = sock;
  return g_dispatchResult;
}
ncclResult_t rasMsgHandleCollResp(struct rasMsg* msg, struct rasSocket* sock) {
  ++g_collRespCalls;
  g_lastDispatchMsg = msg;
  g_lastDispatchSocket = sock;
  return g_dispatchResult;
}

rasConnection* rasConnFind(const union ncclSocketAddress*) { return g_connFindResult; }
ncclResult_t getNewConnEntry(struct rasConnection** conn) {
  if (g_newConnResult == ncclSuccess) *conn = &g_newConn;
  return g_newConnResult;
}
void rasSocketTerminate(struct rasSocket* sock, bool, uint64_t, bool) {
  ++g_socketTerminateCalls;
  g_lastTerminatedSocket = sock;
}
int ncclSocketsCompare(const void* left, const void* right) {
  g_socketCompareLeft = left;
  g_socketCompareRight = right;
  return g_socketCompareResult;
}
void rasClientsNotifyEvent(rasEventGroup group, const struct rasEventNotification* event) {
  ++g_clientsNotifyCalls;
  g_lastEventGroup = group;
  g_lastEventPeerAddr = event ? event->peerAddr : nullptr;
}
int rasPeerFind(const union ncclSocketAddress*) { return g_peerFindResult; }
ncclResult_t rasLinkConnUpdate(struct rasLink* link, struct rasConnection* conn, int peerIdx) {
  ++g_linkUpdateCalls;
  g_updatedLinks.push_back(link);
  g_updatedConnections.push_back(conn);
  g_updatedPeerIndices.push_back(peerIdx);
  return ncclSuccess;
}
ncclResult_t rasConnSendPeersUpdate(struct rasConnection* conn, const struct rasPeerInfo* peers, int nPeers) {
  ++g_sendPeersUpdateCalls;
  g_sentPeersUpdateConn = conn;
  g_sentPeers = peers;
  g_sentPeerCount = nPeers;
  return ncclSuccess;
}
bool rasPeerIsDead(const union ncclSocketAddress*) { return g_peerDead; }
void rasConnDisconnect(const union ncclSocketAddress* addr) {
  ++g_connDisconnectCalls;
  if (addr) g_disconnectedAddr = *addr;
}
ncclResult_t rasPeerDeclareDead(const union ncclSocketAddress* addr) {
  ++g_peerDeclareDeadCalls;
  if (addr) g_declaredDeadAddr = *addr;
  return ncclSuccess;
}

namespace {

struct OwnedMsg {
  rasMsg* ptr = nullptr;
  explicit OwnedMsg(size_t len) { EXPECT_EQ(ncclSuccess, rasMsgAlloc(&ptr, len)); }
  ~OwnedMsg() { rasMsgFree(ptr); }
  rasMsg* release() {
    rasMsg* out = ptr;
    ptr = nullptr;
    return out;
  }
};

void EnqueueMessage(rasConnection* conn, rasSocket* sock, rasSocketStatus status, rasMsgType type, int pfd = -1) {
  sock->status = status;
  if (pfd >= 0) sock->pfd = pfd;
  conn->sock = sock;
  OwnedMsg owned(rasMsgLength(type));
  ASSERT_NE(nullptr, owned.ptr);
  owned.ptr->type = type;
  rasConnEnqueueMsg(conn, owned.release(), rasMsgLength(type));
}

void FreeSendQueue(rasConnection* conn) {
  while (!ncclIntruQueueEmpty(&conn->sendQ)) rasMsgFree(&ncclIntruQueueDequeue(&conn->sendQ)->msg);
}

rasMsg MakeConnInitMsg() {
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINIT;
  msg.connInit.ncclVersion = NCCL_VERSION_CODE;
  msg.connInit.listeningAddr.sin.sin_family = AF_INET;
  msg.connInit.listeningAddr.sin.sin_port = htons(4242);
  return msg;
}

rasMsg PrepareConnInit(rasSocket* sock, uint64_t peersHash, uint64_t deadPeersHash) {
  InitPollFds(1);
  sock->pfd = 0;
  rasMsg msg = MakeConnInitMsg();
  msg.connInit.peersHash = peersHash;
  msg.connInit.deadPeersHash = deadPeersHash;
  return msg;
}

}  // namespace

TEST_F(RasMicrotest, MessageLengthsCoverEveryFixedAndCollectiveType) {
  EXPECT_EQ(56u, rasMsgLength(RAS_MSG_CONNINIT));
  EXPECT_EQ(12u, rasMsgLength(RAS_MSG_CONNINITACK));
  EXPECT_EQ(56u, rasMsgLength(RAS_MSG_KEEPALIVE));
  EXPECT_EQ(32u, rasMsgLength(RAS_MSG_PEERSUPDATE));
  EXPECT_EQ(64u, rasMsgLength(RAS_MSG_COLLRESP));
  // These are wire-format sizes on the Linux host ABI, not formulas copied
  // from rasCollDataLength/rasMsgLength. A layout change must be deliberate.
  EXPECT_EQ(84u, rasCollDataLength(RAS_BC_DEADPEER));
  EXPECT_EQ(60u, rasCollDataLength(RAS_BC_PROFILER_MASK));
  EXPECT_EQ(57u, rasCollDataLength(RAS_COLL_CONNS));
  EXPECT_EQ(64u, rasCollDataLength(RAS_COLL_COMMS));
  EXPECT_EQ(88u, rasCollDataLength(RAS_COLL_DIAG));
  EXPECT_EQ(92u, rasMsgLength(RAS_MSG_COLLREQ, RAS_BC_DEADPEER));
  EXPECT_EQ(68u, rasMsgLength(RAS_MSG_COLLREQ, RAS_BC_PROFILER_MASK));
  EXPECT_EQ(65u, rasMsgLength(RAS_MSG_COLLREQ, RAS_COLL_CONNS));
  EXPECT_EQ(72u, rasMsgLength(RAS_MSG_COLLREQ, RAS_COLL_COMMS));
  EXPECT_EQ(96u, rasMsgLength(RAS_MSG_COLLREQ, RAS_COLL_DIAG));
  EXPECT_EQ(0u, rasCollDataLength(static_cast<rasCollectiveType>(0)));
  EXPECT_EQ(0u, rasCollDataLength(static_cast<rasCollectiveType>(-1)));
  EXPECT_EQ(0u, rasMsgLength(static_cast<rasMsgType>(0)));
  EXPECT_EQ(0u, rasMsgLength(static_cast<rasMsgType>(-1)));
}

TEST_F(RasMicrotest, DiagnosticsParameterUsesDefaultAndOverride) {
  EXPECT_EQ(0, ncclParamRasDiagnostics());
  g_loadParam = [](const char* name, int64_t defaultValue) {
    return std::strcmp(name, "RUN_RAS_DIAGNOSTICS") == 0 ? int64_t{1} : defaultValue;
  };
  EXPECT_EQ(1, ncclParamRasDiagnostics());
}

TEST_F(RasMicrotest, RunDiagnosticsPassivePropagatesInitFailureAndNotifiesOnSuccess) {
  auto comm = std::make_unique<ncclComm>();
  g_diagnosticsInitResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclRunDiagnosticsPassive(comm.get()));
  EXPECT_EQ(1, g_diagnosticsInitCalls);
  EXPECT_EQ(comm.get(), g_diagnosticsInitComm);
  EXPECT_TRUE(g_pairBytes.empty());

  g_diagnosticsInitResult = ncclSuccess;
  rasInitialized = true;
  rasNotificationPipe[1] = 52;
  EXPECT_EQ(ncclSuccess, ncclRunDiagnosticsPassive(comm.get()));
  EXPECT_EQ(2, g_diagnosticsInitCalls);
  ASSERT_EQ(sizeof(rasNotification), g_pairBytes.size());
  EXPECT_EQ(RAS_RUN_DIAG, reinterpret_cast<const rasNotification*>(g_pairBytes.data())->type);
}

TEST_F(RasMicrotest, RunDiagnosticsPassiveReportsUnscopedInitFailure) {
  g_diagnosticsInitResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclRunDiagnosticsPassive(nullptr));
  EXPECT_EQ(nullptr, g_diagnosticsInitComm);
}

TEST_F(RasMicrotest, TerminateUninitializedIsNoOp) {
  rasTerminate();
  EXPECT_TRUE(g_pairBytes.empty());
}

TEST_F(RasMicrotest, TerminateNotifiesAndJoinsThread) {
  rasInitialized = true;
  rasNotificationPipe[1] = 52;
  rasThread = std::thread([] {});
  rasTerminate();
  EXPECT_FALSE(rasThread.joinable());
  ASSERT_EQ(sizeof(rasNotification), g_pairBytes.size());
  EXPECT_EQ(RAS_TERMINATE, reinterpret_cast<const rasNotification*>(g_pairBytes.data())->type);
}

TEST_F(RasMicrotest, TerminateWriteFailureLeavesThreadForCallerToJoin) {
  rasInitialized = true;
  rasNotificationPipe[1] = 52;
  g_pairWriteResult = ncclSystemError;
  rasThread = std::thread([] {});
  rasTerminate();
  EXPECT_TRUE(rasThread.joinable());
  rasThread.join();
}

TEST_F(RasMicrotest, MessageDispatchRoutesExternalTypesAndPropagatesErrors) {
  rasSocket sock{};
  rasMsg msg{};
  for (auto entry : {std::pair{RAS_MSG_KEEPALIVE, &g_keepAliveCalls},
                     std::pair{RAS_MSG_PEERSUPDATE, &g_peersUpdateCalls},
                     std::pair{RAS_MSG_COLLREQ, &g_collReqCalls},
                     std::pair{RAS_MSG_COLLRESP, &g_collRespCalls}}) {
    msg.type = entry.first;
    EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
    EXPECT_EQ(1, *entry.second);
    EXPECT_EQ(&msg, g_lastDispatchMsg);
    EXPECT_EQ(&sock, g_lastDispatchSocket);
  }
  g_dispatchResult = ncclSystemError;
  for (rasMsgType type : {RAS_MSG_KEEPALIVE, RAS_MSG_PEERSUPDATE, RAS_MSG_COLLREQ, RAS_MSG_COLLRESP}) {
    msg.type = type;
    EXPECT_EQ(ncclSystemError, rasMsgHandle(&msg, &sock));
    EXPECT_EQ(&msg, g_lastDispatchMsg);
    EXPECT_EQ(&sock, g_lastDispatchSocket);
  }
  msg.type = static_cast<rasMsgType>(99);
  EXPECT_EQ(ncclInternalError, rasMsgHandle(&msg, &sock));
}

TEST_F(RasMicrotest, ConnInitVersionMismatchSendsNackAndTerminatesSocket) {
  rasSocket sock{};
  rasMsg msg = MakeConnInitMsg();
  msg.connInit.ncclVersion = NCCL_VERSION_CODE - 1;
  int progressCalls = 0;
  g_socketProgress = [&](int, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    ++progressCalls;
    if (progressCalls == 2) {
      const auto* nack = static_cast<const rasMsg*>(ptr);
      EXPECT_EQ(RAS_MSG_CONNINITACK, nack->type);
      EXPECT_EQ(1, nack->connInitAck.nack);
    }
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclInvalidUsage, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(2, progressCalls);
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(&sock, g_lastTerminatedSocket);
}

TEST_F(RasMicrotest, ConnInitKnownDeadPeerNacksWithoutCreatingConnection) {
  g_peerDead = true;
  rasSocket sock{};
  rasMsg msg = MakeConnInitMsg();
  int progressCalls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int size, int* offset, int* closed) {
    ++progressCalls;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(2, progressCalls);
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(nullptr, sock.conn);
}

TEST_F(RasMicrotest, ConnInitPropagatesNewConnectionAllocationFailure) {
  g_newConnResult = ncclSystemError;
  rasSocket sock{};
  rasMsg msg = MakeConnInitMsg();
  EXPECT_EQ(ncclSystemError, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(nullptr, sock.conn);
  EXPECT_EQ(0, g_clientsNotifyCalls);
}

TEST_F(RasMicrotest, ConnInitCreatesReadyConnectionAndQueuesAck) {
  InitPollFds(1);
  rasSocket sock{};
  sock.pfd = 0;
  rasMsg msg = MakeConnInitMsg();
  msg.connInit.peersHash = rasPeersHash;
  msg.connInit.deadPeersHash = rasDeadPeersHash;
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(RAS_SOCK_READY, sock.status);
  EXPECT_EQ(&g_newConn, sock.conn);
  EXPECT_EQ(&sock, g_newConn.sock);
  EXPECT_EQ(AF_INET, g_newConn.addr.sa.sa_family);
  EXPECT_EQ(htons(4242), g_newConn.addr.sin.sin_port);
  EXPECT_EQ(AF_INET, sock.sock.addr.sa.sa_family);
  EXPECT_EQ(htons(4242), sock.sock.addr.sin.sin_port);
  EXPECT_EQ(1, g_clientsNotifyCalls);
  EXPECT_EQ(RAS_EVENT_TRACE, g_lastEventGroup);
  EXPECT_EQ(&msg.connInit.listeningAddr, g_lastEventPeerAddr);
  EXPECT_EQ(0, g_linkUpdateCalls);
  EXPECT_EQ(0, g_sendPeersUpdateCalls);
  rasMsgMeta* meta = ncclIntruQueueHead(&g_newConn.sendQ);
  ASSERT_NE(nullptr, meta);
  EXPECT_EQ(RAS_MSG_CONNINITACK, meta->msg.type);
  EXPECT_EQ(0, meta->msg.connInitAck.nack);
  FreeSendQueue(&g_newConn);
}

TEST_F(RasMicrotest, ConnInitHashMismatchSendsPeersUpdateAndUpdatesBothLinks) {
  g_peerFindResult = 3;
  rasPeerInfo peer{};
  rasPeers = &peer;
  nRasPeers = 1;
  rasPeersHash = 10;
  rasDeadPeersHash = 20;
  rasSocket sock{};
  rasMsg msg = PrepareConnInit(&sock, 11, 21);
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(2, g_linkUpdateCalls);
  ASSERT_EQ(2u, g_updatedLinks.size());
  EXPECT_EQ(&rasNextLink, g_updatedLinks[0]);
  EXPECT_EQ(&rasPrevLink, g_updatedLinks[1]);
  ASSERT_EQ(2u, g_updatedConnections.size());
  EXPECT_EQ(&g_newConn, g_updatedConnections[0]);
  EXPECT_EQ(&g_newConn, g_updatedConnections[1]);
  ASSERT_EQ(2u, g_updatedPeerIndices.size());
  EXPECT_EQ(3, g_updatedPeerIndices[0]);
  EXPECT_EQ(3, g_updatedPeerIndices[1]);
  EXPECT_EQ(1, g_sendPeersUpdateCalls);
  EXPECT_EQ(&g_newConn, g_sentPeersUpdateConn);
  EXPECT_EQ(&peer, g_sentPeers);
  EXPECT_EQ(1, g_sentPeerCount);
  EXPECT_EQ(11u, g_newConn.lastRecvPeersHash);
  EXPECT_EQ(21u, g_newConn.lastRecvDeadPeersHash);
  ASSERT_FALSE(ncclIntruQueueEmpty(&g_newConn.sendQ));
  FreeSendQueue(&g_newConn);
}

TEST_F(RasMicrotest, ConnInitDeadPeersHashMismatchAloneSendsPeersUpdate) {
  rasPeersHash = 10;
  rasDeadPeersHash = 20;
  rasSocket sock{};
  rasMsg msg = PrepareConnInit(&sock, rasPeersHash, rasDeadPeersHash + 1);
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(1, g_sendPeersUpdateCalls);
  FreeSendQueue(&g_newConn);
}

TEST_F(RasMicrotest, ConnInitPeersHashMismatchAloneSendsPeersUpdate) {
  rasPeersHash = 10;
  rasDeadPeersHash = 20;
  rasSocket sock{};
  rasMsg msg = PrepareConnInit(&sock, rasPeersHash + 1, rasDeadPeersHash);
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(1, g_sendPeersUpdateCalls);
  FreeSendQueue(&g_newConn);
}

TEST_F(RasMicrotest, ConnInitMatchingHashesUpdatesLinksForKnownPeer) {
  g_peerFindResult = 3;
  rasPeersHash = 10;
  rasDeadPeersHash = 20;
  rasSocket sock{};
  rasMsg msg = PrepareConnInit(&sock, rasPeersHash, rasDeadPeersHash);
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(2, g_linkUpdateCalls);
  EXPECT_EQ(0, g_sendPeersUpdateCalls);
  ASSERT_EQ(2u, g_updatedPeerIndices.size());
  EXPECT_EQ(3, g_updatedPeerIndices[0]);
  EXPECT_EQ(3, g_updatedPeerIndices[1]);
  FreeSendQueue(&g_newConn);
}

TEST_F(RasMicrotest, ConnInitExistingLowerAddressRejectsNewSocket) {
  rasConnection existing{};
  rasSocket existingSock{};
  existing.sock = &existingSock;
  g_connFindResult = &existing;
  g_socketCompareResult = -1;
  rasSocket incoming{};
  rasMsg msg = MakeConnInitMsg();
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &incoming));
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(&incoming, g_lastTerminatedSocket);
  EXPECT_EQ(nullptr, incoming.conn);
  EXPECT_EQ(&rasNetListeningSocket.addr, g_socketCompareLeft);
  EXPECT_EQ(&existing.addr, g_socketCompareRight);
}

TEST_F(RasMicrotest, ConnInitExistingConnectionWithoutSocketUsesIncomingSocket) {
  InitPollFds(1);
  rasConnection existing{};
  g_connFindResult = &existing;
  rasSocket incoming{};
  incoming.pfd = 0;
  rasMsg msg = MakeConnInitMsg();
  OwnedMsg queued(rasMsgLength(RAS_MSG_KEEPALIVE));
  ASSERT_NE(nullptr, queued.ptr);
  queued.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&existing, queued.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &incoming));
  EXPECT_EQ(&incoming, existing.sock);
  EXPECT_EQ(&existing, incoming.conn);
  rasMsgMeta* head = ncclIntruQueueHead(&existing.sendQ);
  ASSERT_NE(nullptr, head);
  EXPECT_EQ(RAS_MSG_CONNINITACK, head->msg.type);
  FreeSendQueue(&existing);
}

TEST_F(RasMicrotest, ConnInitExistingHigherAddressReplacesOldSocket) {
  InitPollFds(1);
  rasConnection existing{};
  rasSocket oldSock{};
  existing.sock = &oldSock;
  g_connFindResult = &existing;
  g_socketCompareResult = 1;
  rasSocket incoming{};
  incoming.pfd = 0;
  rasMsg msg = MakeConnInitMsg();
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &incoming));
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(&oldSock, g_lastTerminatedSocket);
  EXPECT_EQ(&incoming, existing.sock);
  EXPECT_EQ(&existing, incoming.conn);
  ASSERT_FALSE(ncclIntruQueueEmpty(&existing.sendQ));
  FreeSendQueue(&existing);
}

TEST_F(RasMicrotest, ConnInitEqualAddressKeepsIncomingSocket) {
  InitPollFds(1);
  rasConnection existing{};
  rasSocket oldSock{};
  existing.sock = &oldSock;
  g_connFindResult = &existing;
  g_socketCompareResult = 0;
  rasSocket incoming{};
  incoming.pfd = 0;
  rasMsg msg = MakeConnInitMsg();
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &incoming));
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(&oldSock, g_lastTerminatedSocket);
  EXPECT_EQ(&incoming, existing.sock);
  EXPECT_EQ(&existing, incoming.conn);
  FreeSendQueue(&existing);
}

TEST_F(RasMicrotest, ConnInitAckNackDeclaresPeerDead) {
  rasSocket sock{};
  sock.sock.addr.sin.sin_family = AF_INET;
  sock.sock.addr.sin.sin_port = htons(7331);
  sock.status = RAS_SOCK_HANDSHAKE;
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINITACK;
  msg.connInitAck.nack = 1;
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(1, g_connDisconnectCalls);
  EXPECT_EQ(1, g_peerDeclareDeadCalls);
  EXPECT_EQ(AF_INET, g_disconnectedAddr.sa.sa_family);
  EXPECT_EQ(htons(7331), g_disconnectedAddr.sin.sin_port);
  EXPECT_EQ(0, std::memcmp(&g_disconnectedAddr, &g_declaredDeadAddr, sizeof(g_disconnectedAddr)));
  EXPECT_EQ(RAS_SOCK_HANDSHAKE, sock.status);
}

TEST_F(RasMicrotest, ConnInitAckSuccessMarksSocketReady) {
  rasSocket sock{};
  sock.status = RAS_SOCK_HANDSHAKE;
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINITACK;
  msg.connInitAck.nack = 0;
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(RAS_SOCK_READY, sock.status);
}

TEST_F(RasMicrotest, DeadPeerBroadcastHandlesNewAndKnownPeers) {
  rasCollRequest req{};
  rasCollRequest* reqPtr = &req;
  size_t reqLen = 0;
  bool done = true;
  rasMsgHandleBCDeadPeer(&reqPtr, &reqLen, &done);
  EXPECT_EQ(84u, reqLen);
  EXPECT_FALSE(done);
  EXPECT_EQ(1, g_connDisconnectCalls);
  EXPECT_EQ(1, g_peerDeclareDeadCalls);

  g_peerDead = true;
  done = false;
  rasMsgHandleBCDeadPeer(&reqPtr, &reqLen, &done);
  EXPECT_TRUE(done);
  EXPECT_EQ(1, g_connDisconnectCalls);
  EXPECT_EQ(1, g_peerDeclareDeadCalls);
}

TEST_F(RasMicrotest, ProfilerMaskBroadcastSetsOverrideAndKeepsPropagating) {
  rasCollRequest req{};
  req.profilerMask.eventMask = 0x5a;
  rasCollRequest* reqPtr = &req;
  size_t reqLen = 0;
  bool done = true;
  rasMsgHandleBCProfilerMask(&reqPtr, &reqLen, &done);
  EXPECT_EQ(60u, reqLen);
  EXPECT_EQ(0x5a, g_profilerMask);
  EXPECT_FALSE(done);
}

TEST_F(RasMicrotest, NetSendNackStopsAfterPartialLengthAndPropagatesError) {
  rasSocket sock{};
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int, int* offset, int* closed) {
    ++calls;
    *offset = 1;
    *closed = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclSuccess, rasNetSendNack(&sock));
  EXPECT_EQ(1, calls);

  calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int, int*, int*) {
    ++calls;
    return ncclSystemError;
  };
  EXPECT_EQ(ncclSystemError, rasNetSendNack(&sock));
  EXPECT_EQ(1, calls);
}

TEST_F(RasMicrotest, ThreadMainSocketFdFailureRunsCleanup) {
  rasNotificationPipe[0] = 51;
  g_socketGetFdResult = ncclSystemError;
  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  for (int calls : g_cleanupCalls) EXPECT_EQ(1, calls);
  EXPECT_EQ(nullptr, rasPfds);
}

TEST_F(RasMicrotest, ThreadMainDispatchesEveryFdClassAndTerminatesCleanly) {
  rasNotificationPipe[0] = 51;
  rasClientListeningSocket = 43;
  InitPollFds(8);
  rasPfds[3].fd = 61;
  rasPfds[4].fd = 62;
  rasPfds[5].fd = 64;

  rasSocket rasSock{};
  rasSock.sock.socketDescriptor = 61;
  rasSocketsHead = &rasSock;
  rasClient skippedClient{};
  skippedClient.sock = 63;
  rasClient client{};
  client.sock = 62;
  skippedClient.next = &client;
  rasClientsHead = &skippedClient;

  SetPairNotification(RAS_TERMINATE);

  std::vector<int> timeouts;
  g_nextWakeupOverride = g_clockNano + CLOCK_UNITS_PER_SEC / 4;
  g_poll = [&](pollfd* fds, nfds_t n, int timeout) {
    timeouts.push_back(timeout);
    EXPECT_EQ(8u, n);
    for (nfds_t i = 0; i < n; ++i) fds[i].revents = 0;
    if (g_pollCalls == 1) {
      EXPECT_EQ(POLLIN, fds[0].events);
      EXPECT_EQ(POLLIN, fds[1].events);
      EXPECT_EQ(POLLIN, fds[2].events);
      fds[1].revents = POLLIN;
      fds[2].revents = POLLIN;
      fds[3].revents = POLLIN;
      fds[4].revents = POLLIN;
      fds[5].revents = POLLIN;
      return 5;
    }
    fds[0].revents = POLLIN;
    return 1;
  };

  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  EXPECT_EQ(2, g_pollCalls);
  ASSERT_EQ(2u, timeouts.size());
  EXPECT_EQ(1000, timeouts[0]);
  EXPECT_EQ(251, timeouts[1]);
  EXPECT_EQ(1, g_netAcceptCalls);
  EXPECT_EQ(1, g_clientAcceptCalls);
  EXPECT_EQ(1, g_sockEventCalls);
  EXPECT_EQ(1, g_clientEventCalls);
  EXPECT_EQ(3, g_lastSockPollIdx);
  EXPECT_EQ(4, g_lastClientPollIdx);
  EXPECT_EQ(&rasSock, g_lastSockEventSocket);
  EXPECT_EQ(&client, g_lastClientEventClient);
  for (int calls : g_timeoutCalls) EXPECT_EQ(1, calls);
  for (int calls : g_cleanupCalls) EXPECT_EQ(1, calls);
}

TEST_F(RasMicrotest, ThreadMainToleratesPollErrorAndInvalidFdBeforeTerminating) {
  rasNotificationPipe[0] = 51;
  InitPollFds(4);
  rasPfds[3].fd = 77;

  SetPairNotification(RAS_TERMINATE);
  g_nextWakeupOverride = 1;

  std::vector<int> timeouts;
  g_poll = [&](pollfd* fds, nfds_t n, int timeout) {
    timeouts.push_back(timeout);
    for (nfds_t i = 0; i < n; ++i) fds[i].revents = 0;
    if (g_pollCalls == 1) {
      errno = EBADF;
      return -1;
    }
    if (g_pollCalls == 2) {
      fds[3].revents = POLLNVAL;
      return 1;
    }
    EXPECT_EQ(POLL_FD_IGNORE, fds[3].fd);
    fds[0].revents = POLLIN;
    return 1;
  };

  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  EXPECT_EQ(3, g_pollCalls);
  ASSERT_EQ(3u, timeouts.size());
  EXPECT_EQ(1000, timeouts[0]);
  EXPECT_EQ(1, timeouts[1]);
  EXPECT_EQ(1, timeouts[2]);
  for (int calls : g_timeoutCalls) EXPECT_EQ(2, calls);
}

TEST_F(RasMicrotest, ThreadMainTreatsInterruptedPollAsAnEmptyIteration) {
  rasNotificationPipe[0] = 51;
  SetPairNotification(RAS_TERMINATE);
  g_poll = [](pollfd* fds, nfds_t n, int) {
    for (nfds_t i = 0; i < n; ++i) fds[i].revents = 0;
    if (g_pollCalls == 1) {
      errno = EINTR;
      return -1;
    }
    fds[0].revents = POLLIN;
    return 1;
  };

  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  EXPECT_EQ(2, g_pollCalls);
  for (int calls : g_timeoutCalls) EXPECT_EQ(1, calls);
}

TEST_F(RasMicrotest, MsgAllocReturnsZeroedPayloadAndFreeAcceptsNull) {
  rasMsg* msg = nullptr;
  ASSERT_EQ(ncclSuccess, rasMsgAlloc(&msg, rasMsgLength(RAS_MSG_KEEPALIVE)));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_NONE, msg->type);
  rasMsgFree(msg);
  rasMsgFree(nullptr);
}

TEST_F(RasMicrotest, ConnEnqueueBackInitializesMetadataAndArmsReadySocket) {
  InitPollFds(1);
  rasSocket sock{};
  rasConnection conn{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_READY, RAS_MSG_KEEPALIVE, 0);

  rasMsgMeta* meta = ncclIntruQueueHead(&conn.sendQ);
  ASSERT_NE(nullptr, meta);
  EXPECT_EQ(0, meta->offset);
  EXPECT_EQ((int)rasMsgLength(RAS_MSG_KEEPALIVE), meta->length);
  EXPECT_NE(0, rasPfds[0].events & POLLOUT);
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, ConnEnqueueFrontPrecedesExistingMessage) {
  rasConnection conn{};
  OwnedMsg first(rasMsgLength(RAS_MSG_KEEPALIVE));
  OwnedMsg second(rasMsgLength(RAS_MSG_CONNINIT));
  ASSERT_NE(nullptr, first.ptr);
  ASSERT_NE(nullptr, second.ptr);
  first.ptr->type = RAS_MSG_KEEPALIVE;
  second.ptr->type = RAS_MSG_CONNINIT;
  rasMsg* firstRaw = first.release();
  rasMsg* secondRaw = second.release();
  rasConnEnqueueMsg(&conn, firstRaw, rasMsgLength(RAS_MSG_KEEPALIVE), false);
  rasConnEnqueueMsg(&conn, secondRaw, rasMsgLength(RAS_MSG_CONNINIT), true);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  EXPECT_EQ(secondRaw, &ncclIntruQueueHead(&conn.sendQ)->msg);
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, ConnEnqueueHandshakeArmsOnlyConnInitMessage) {
  InitPollFds(1);
  rasSocket sock{};
  rasConnection conn{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_HANDSHAKE, RAS_MSG_CONNINIT, 0);
  EXPECT_NE(0, rasPfds[0].events & POLLOUT);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  FreeSendQueue(&conn);

  rasPfds[0].events = 0;
  EnqueueMessage(&conn, &sock, RAS_SOCK_HANDSHAKE, RAS_MSG_KEEPALIVE, 0);
  EXPECT_EQ(0, rasPfds[0].events & POLLOUT);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, ConnSendEmptyQueueReportsAllSent) {
  rasConnection conn{};
  rasSocket sock{};
  conn.sock = &sock;
  int closed = -1;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(0, closed);
  EXPECT_TRUE(allSent);
}

TEST_F(RasMicrotest, ConnSendHandshakeBlocksNonInitMessageWithoutCallingSocket) {
  rasConnection conn{};
  rasSocket sock{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_HANDSHAKE, RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int, int*, int*) {
    ++calls;
    return ncclSuccess;
  };
  int closed;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(0, calls);
  EXPECT_TRUE(allSent);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, ConnSendHandshakeAllowsConnInitMessage) {
  InitPollFds(1);
  rasConnection conn{};
  rasSocket sock{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_HANDSHAKE, RAS_MSG_CONNINIT, 0);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int size, int* offset, int* closed) {
    ++calls;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(2, calls);
  EXPECT_TRUE(allSent);
  EXPECT_TRUE(ncclIntruQueueEmpty(&conn.sendQ));
}

TEST_F(RasMicrotest, ConnSendCompleteMessageDequeuesIt) {
  rasConnection conn{};
  rasSocket sock{};
  // Closed avoids arming rasPfds; rasConnSendMsg itself accepts this state.
  EnqueueMessage(&conn, &sock, RAS_SOCK_CLOSED, RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_SEND, op);
    ++calls;
    EXPECT_EQ(RAS_MSG_KEEPALIVE,
              reinterpret_cast<const rasMsg*>(static_cast<char*>(ptr) + sizeof(int))->type);
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(2, calls);
  EXPECT_TRUE(allSent);
  EXPECT_TRUE(ncclIntruQueueEmpty(&conn.sendQ));
}

TEST_F(RasMicrotest, ConnSendPartialLengthKeepsMessageQueued) {
  rasConnection conn{};
  rasSocket sock{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_CLOSED, RAS_MSG_KEEPALIVE);
  g_socketProgress = [](int, ncclSocket*, void*, int, int* offset, int* closed) {
    *offset = 2;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = true;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_FALSE(allSent);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  EXPECT_EQ(2, ncclIntruQueueHead(&conn.sendQ)->offset);
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, ConnSendClosedSocketReturnsWithoutDequeuing) {
  rasConnection conn{};
  rasSocket sock{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_CLOSED, RAS_MSG_KEEPALIVE);
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int* closed) {
    *closed = 1;
    return ncclSuccess;
  };
  int closed = 0;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(1, closed);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, ConnSendPartialBodyKeepsMessageQueued) {
  rasConnection conn{};
  rasSocket sock{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_CLOSED, RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int size, int* offset, int* closed) {
    ++calls;
    *offset = calls == 1 ? size : size - 1;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = true;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(2, calls);
  EXPECT_FALSE(allSent);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, ConnSendSocketErrorPropagates) {
  rasConnection conn{};
  rasSocket sock{};
  EnqueueMessage(&conn, &sock, RAS_SOCK_CLOSED, RAS_MSG_KEEPALIVE);
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int*) { return ncclSystemError; };
  int closed;
  bool allSent;
  EXPECT_EQ(ncclSystemError, rasConnSendMsg(&conn, &closed, &allSent));
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  FreeSendQueue(&conn);
}

TEST_F(RasMicrotest, MsgRecvCompletesLengthThenBodyAndResetsSocketState) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    ++calls;
    if (calls == 1) {
      *static_cast<int*>(ptr) = msgLen;
    } else {
      auto* msg = reinterpret_cast<rasMsg*>(static_cast<char*>(ptr) + sizeof(int));
      msg->type = RAS_MSG_KEEPALIVE;
    }
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  ASSERT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, msg->type);
  EXPECT_EQ(2, calls);
  EXPECT_EQ(0, sock.recvOffset);
  EXPECT_EQ(0, sock.recvLength);
  EXPECT_EQ(nullptr, sock.recvMsg);
  std::free(msg);
}

TEST_F(RasMicrotest, MsgRecvAcceptsPeerSuppliedOneByteLength) {
  rasSocket sock{};
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    ++calls;
    if (calls == 1) {
      *static_cast<int*>(ptr) = 1;
    } else {
      *(static_cast<char*>(ptr) + sizeof(int)) = 0x5a;
    }
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  ASSERT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(0x5a, *reinterpret_cast<unsigned char*>(msg));
  EXPECT_EQ(2, calls);
  EXPECT_EQ(0, closed);
  EXPECT_EQ(0, sock.recvOffset);
  EXPECT_EQ(0, sock.recvLength);
  EXPECT_EQ(nullptr, sock.recvMsg);
  std::free(msg);
}

TEST_F(RasMicrotest, MsgRecvPartialLengthReturnsWithoutAllocatingBody) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int* offset, int* closed) {
    *offset = 2;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
  EXPECT_EQ(2, sock.recvOffset);
}

TEST_F(RasMicrotest, MsgRecvClosedDuringLengthReturnsImmediately) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int* closed) {
    *closed = 1;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(1, closed);
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
}

TEST_F(RasMicrotest, MsgRecvSocketErrorPropagates) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int*) { return ncclSystemError; };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSystemError, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
}

TEST_F(RasMicrotest, MsgRecvBodySocketErrorPropagatesAndPreservesState) {
  rasSocket sock{};
  sock.recvLength = rasMsgLength(RAS_MSG_KEEPALIVE);
  sock.recvOffset = sizeof(sock.recvLength);
  sock.recvMsg = static_cast<rasMsg*>(std::calloc(1, sock.recvLength));
  ASSERT_NE(nullptr, sock.recvMsg);
  g_socketProgress = [](int op, ncclSocket*, void*, int, int*, int*) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    return ncclSystemError;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  EXPECT_EQ(ncclSystemError, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(sizeof(sock.recvLength), sock.recvOffset);
  ASSERT_NE(nullptr, sock.recvMsg);
  std::free(sock.recvMsg);
  sock.recvMsg = nullptr;
}

TEST_F(RasMicrotest, MsgRecvClosedDuringBodyPreservesAllocatedMessage) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    ++calls;
    if (calls == 1) {
      *static_cast<int*>(ptr) = msgLen;
      *offset = size;
      *closed = 0;
    } else {
      *closed = 1;
    }
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(1, closed);
  EXPECT_EQ(nullptr, msg);
  ASSERT_NE(nullptr, sock.recvMsg);
  std::free(sock.recvMsg);
  sock.recvMsg = nullptr;
}

TEST_F(RasMicrotest, MsgRecvPartialBodyPreservesProgressForNextCall) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    ++calls;
    if (calls == 1) *static_cast<int*>(ptr) = msgLen;
    *offset = calls == 1 ? size : size - 1;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  ASSERT_NE(nullptr, sock.recvMsg);
  EXPECT_EQ(msgLen + (int)sizeof(int) - 1, sock.recvOffset);
  std::free(sock.recvMsg);
  sock.recvMsg = nullptr;
}

TEST_F(RasMicrotest, MsgRecvResumesPartialBodyWithoutRereadingLength) {
  rasSocket sock{};
  sock.recvLength = rasMsgLength(RAS_MSG_KEEPALIVE);
  sock.recvOffset = sizeof(sock.recvLength) + 3;
  sock.recvMsg = static_cast<rasMsg*>(std::calloc(1, sock.recvLength));
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    EXPECT_EQ(sock.recvLength + (int)sizeof(sock.recvLength), size);
    ++calls;
    auto* msg = reinterpret_cast<rasMsg*>(static_cast<char*>(ptr) + sizeof(sock.recvLength));
    msg->type = RAS_MSG_KEEPALIVE;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  ASSERT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, msg->type);
  EXPECT_EQ(1, calls);
  EXPECT_EQ(0, sock.recvOffset);
  EXPECT_EQ(0, sock.recvLength);
  EXPECT_EQ(nullptr, sock.recvMsg);
  std::free(msg);
}

TEST_F(RasMicrotest, GetNewPollEntryGrowsInChunksAndReusesVacancies) {
  int first = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&first));
  EXPECT_EQ(0, first);
  EXPECT_EQ(RAS_INCREMENT, nRasPfds);
  EXPECT_EQ(NCCL_INVALID_SOCKET, rasPfds[first].fd);

  rasPfds[0].fd = 7;
  int second = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&second));
  EXPECT_EQ(1, second);
  rasPfds[0].fd = NCCL_INVALID_SOCKET;
  rasPfds[0].events = POLLIN;
  rasPfds[0].revents = POLLOUT;
  int reused = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&reused));
  EXPECT_EQ(0, reused);
  EXPECT_EQ(0, rasPfds[0].events);
  EXPECT_EQ(0, rasPfds[0].revents);
}

TEST_F(RasMicrotest, GetNewPollEntryExpandsAgainWhenEverySlotIsOccupied) {
  int index;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&index));
  ASSERT_EQ(RAS_INCREMENT, nRasPfds);
  for (int i = 0; i < nRasPfds; ++i) rasPfds[i].fd = i + 10;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&index));
  EXPECT_EQ(RAS_INCREMENT, index);
  EXPECT_EQ(2 * RAS_INCREMENT, nRasPfds);
  EXPECT_EQ(NCCL_INVALID_SOCKET, rasPfds[index].fd);
  EXPECT_EQ(0, rasPfds[index].events);
  EXPECT_EQ(0, rasPfds[index].revents);
  rasPfds[index].fd = 99;
  int next = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&next));
  EXPECT_EQ(RAS_INCREMENT + 1, next);
  EXPECT_EQ(2 * RAS_INCREMENT, nRasPfds);
}
