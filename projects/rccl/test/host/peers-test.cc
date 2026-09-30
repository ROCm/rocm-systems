/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/peers.cc.

#include <arpa/inet.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"
#include "socket.h"

// Rename peers.cc symbols that ras-test.cc already supplies in the shared microtest binary.
#define ncclSocketToString PeersTestNcclSocketToString
#define ncclSocketToHost PeersTestNcclSocketToHost
#define rasGpuDevsToString PeersTestRasGpuDevsToString
#define rasClientsNotifyEvent PeersTestRasClientsNotifyEvent
#define rasNetListeningSocket PeersTestRasNetListeningSocket
#define rasLine PeersTestRasLine
#define rasNextLink PeersTestRasNextLink
#define rasPrevLink PeersTestRasPrevLink
#define rasConnCreate PeersTestRasConnCreate
#define rasConnFind PeersTestRasConnFind
#define rasConnDisconnect PeersTestRasConnDisconnect
#define rasConnEnqueueMsg PeersTestRasConnEnqueueMsg
#define rasLinkAddFallback PeersTestRasLinkAddFallback
#define rasMsgAlloc PeersTestRasMsgAlloc
#define rasMsgFree PeersTestRasMsgFree
#define rasPeers PeersTestRasPeers
#define nRasPeers PeersTestNRasPeers
#define rasPeersHash PeersTestRasPeersHash
#define rasDeadPeers PeersTestRasDeadPeers
#define nRasDeadPeers PeersTestNRasDeadPeers
#define rasDeadPeersHash PeersTestRasDeadPeersHash
#define rasLocalHandleAddRanks PeersTestRasLocalHandleAddRanks
#define rasPeerFind PeersTestRasPeerFind
#define rasConnSendPeersUpdate PeersTestRasConnSendPeersUpdate
#define rasMsgHandlePeersUpdate PeersTestRasMsgHandlePeersUpdate
#define rasLinkCalculatePeer PeersTestRasLinkCalculatePeer
#define rasPeerDeclareDead PeersTestRasPeerDeclareDead
#define rasPeerIsDead PeersTestRasPeerIsDead
#define rasPeerInfoToString PeersTestRasPeerInfoToString
#define rasPeerToString PeersTestRasPeerToString
#define ncclSocketsCompare PeersTestNcclSocketsCompare
#define ncclSocketsSameNode PeersTestNcclSocketsSameNode
#define rasPeersTerminate PeersTestRasPeersTerminate

#include "ras/ras_internal.h"

const char* ncclSocketToString(const ncclSocketAddress* addr, char* buf, int numericHostForm = 1);

namespace {

int g_allocationCalls = 0;
int g_failAllocationCall = 0;

template <typename T>
ncclResult_t PeersTestCalloc(T** ptr, size_t count) {
  ++g_allocationCalls;
  if (g_allocationCalls == g_failAllocationCall) return ncclSystemError;
  return ncclCallocDebug(ptr, count, __FILE__, __LINE__, __func__, false);
}

template <typename T>
ncclResult_t PeersTestRealloc(T** ptr, size_t oldCount, size_t newCount) {
  ++g_allocationCalls;
  if (g_allocationCalls == g_failAllocationCall) return ncclSystemError;
  return ncclReallocDebug(ptr, oldCount, newCount, __FILE__, __LINE__, __func__, false);
}

}  // namespace

#define clockNano PeersTestClockNano
#undef ncclCalloc
#define ncclCalloc(...) PeersTestCalloc(__VA_ARGS__)
#undef ncclRealloc
#define ncclRealloc(...) PeersTestRealloc(__VA_ARGS__)

uint64_t PeersTestClockNano();

#include PEERS_CC_PATH

#undef clockNano
#undef ncclCalloc
#undef ncclRealloc

struct ncclSocket rasNetListeningSocket;
char rasLine[SOCKET_NAME_MAXLEN + 1];
struct rasLink rasNextLink = {1};
struct rasLink rasPrevLink = {-1};

namespace {

struct EnqueuedMsg {
  rasConnection* conn;
  rasMsg* msg;
  size_t len;
  bool front;
};

struct EventRecord {
  std::string type;
  bool hasPeer;
  rasPeerInfo peer;
  bool hasAddr;
  ncclSocketAddress addr;
};

std::vector<rasConnection*> g_connections;
std::vector<EnqueuedMsg> g_enqueuedMsgs;
std::vector<ncclSocketAddress> g_createdAddrs;
std::vector<ncclSocketAddress> g_disconnectedAddrs;
std::vector<EventRecord> g_events;
ncclResult_t g_connCreateResult = ncclSuccess;
ncclResult_t g_msgAllocResult = ncclSuccess;
ncclResult_t g_addFallbackResult = ncclSuccess;
int g_addFallbackCalls = 0;
int64_t g_clockNano = 123456;

ncclSocketAddress MakeIpv4(const char* ip, uint16_t port) {
  ncclSocketAddress addr{};
  addr.sin.sin_family = AF_INET;
  EXPECT_EQ(1, inet_pton(AF_INET, ip, &addr.sin.sin_addr));
  addr.sin.sin_port = htons(port);
  return addr;
}

ncclSocketAddress MakeIpv6(const char* ip, uint16_t port) {
  ncclSocketAddress addr{};
  addr.sin6.sin6_family = AF_INET6;
  EXPECT_EQ(1, inet_pton(AF_INET6, ip, &addr.sin6.sin6_addr));
  addr.sin6.sin6_port = htons(port);
  return addr;
}

rasPeerInfo MakePeer(const ncclSocketAddress& addr, int pid, uint64_t cudaDevs = 1, uint64_t nvmlDevs = 1) {
  rasPeerInfo peer{};
  peer.addr = addr;
  peer.pid = pid;
  peer.cudaDevs = cudaDevs;
  peer.nvmlDevs = nvmlDevs;
  peer.hostHash = static_cast<uint64_t>(pid + 100);
  peer.pidHash = static_cast<uint64_t>(pid + 200);
  return peer;
}

rasRankInit MakeRank(const ncclSocketAddress& addr, int pid, int cudaDev, int nvmlDev) {
  rasRankInit rank{};
  rank.addr = addr;
  rank.pid = pid;
  rank.cudaDev = cudaDev;
  rank.nvmlDev = nvmlDev;
  rank.hostHash = static_cast<uint64_t>(pid + 100);
  rank.pidHash = static_cast<uint64_t>(pid + 200);
  return rank;
}

void SetPeers(std::initializer_list<rasPeerInfo> peers, int selfIdx) {
  free(rasPeers);
  nRasPeers = static_cast<int>(peers.size());
  rasPeers = static_cast<rasPeerInfo*>(calloc(peers.size() ? peers.size() : 1, sizeof(*rasPeers)));
  int index = 0;
  for (const rasPeerInfo& peer : peers) rasPeers[index++] = peer;
  ncclSocketAddress selfAddr{};
  if (selfIdx >= 0) selfAddr = rasPeers[selfIdx].addr;
  qsort(rasPeers, nRasPeers, sizeof(*rasPeers), rasAddrPeerInfoCompare);
  myPeerIdx = selfIdx >= 0 ? rasPeerFind(&selfAddr) : -1;
  if (selfIdx >= 0) rasNetListeningSocket.addr = selfAddr;
  rasPeersHash = getHash(reinterpret_cast<const char*>(rasPeers), nRasPeers * sizeof(*rasPeers));
}

void SetDeadPeers(std::initializer_list<ncclSocketAddress> peers) {
  free(rasDeadPeers);
  rasDeadPeersSize = std::max(RAS_INCREMENT, static_cast<int>(peers.size()));
  rasDeadPeers = static_cast<ncclSocketAddress*>(calloc(rasDeadPeersSize, sizeof(*rasDeadPeers)));
  nRasDeadPeers = static_cast<int>(peers.size());
  int index = 0;
  for (const ncclSocketAddress& peer : peers) rasDeadPeers[index++] = peer;
  qsort(rasDeadPeers, nRasDeadPeers, sizeof(*rasDeadPeers), ncclSocketsCompare);
  rasDeadPeersHash = getHash(reinterpret_cast<const char*>(rasDeadPeers), nRasDeadPeers * sizeof(*rasDeadPeers));
}

rasConnection* AddConnection(const ncclSocketAddress& addr, rasSocketStatus status = RAS_SOCK_READY) {
  auto* conn = static_cast<rasConnection*>(calloc(1, sizeof(rasConnection)));
  conn->addr = addr;
  auto* sock = static_cast<rasSocket*>(calloc(1, sizeof(rasSocket)));
  sock->status = status;
  sock->conn = conn;
  conn->sock = sock;
  g_connections.push_back(conn);
  return conn;
}

rasLinkConn* AddLinkConn(rasLink* link, rasConnection* conn, int peerIdx = -1) {
  auto* entry = static_cast<rasLinkConn*>(calloc(1, sizeof(rasLinkConn)));
  entry->conn = conn;
  entry->peerIdx = peerIdx;
  if (link->conns == nullptr) {
    link->conns = entry;
  } else {
    rasLinkConn* tail = link->conns;
    while (tail->next) tail = tail->next;
    tail->next = entry;
  }
  return entry;
}

void FreeLink(rasLink* link) {
  while (link->conns) {
    rasLinkConn* next = link->conns->next;
    free(link->conns);
    link->conns = next;
  }
  link->lastUpdatePeersTime = 0;
}

rasMsg* MakePeersUpdate(const std::vector<rasPeerInfo>& peers, const std::vector<ncclSocketAddress>& deadPeers,
                        uint64_t peersHash, uint64_t deadPeersHash) {
  int len = static_cast<int>(rasMsgLength(RAS_MSG_PEERSUPDATE) + peers.size() * sizeof(rasPeerInfo));
  int deadOffset = 0;
  if (!deadPeers.empty()) {
    ALIGN_SIZE(len, alignof(ncclSocketAddress));
    deadOffset = len;
    len += static_cast<int>(deadPeers.size() * sizeof(ncclSocketAddress));
  }
  rasMsg* msg = nullptr;
  EXPECT_EQ(ncclSuccess, rasMsgAlloc(&msg, len));
  msg->type = RAS_MSG_PEERSUPDATE;
  msg->peersUpdate.peersHash = peersHash;
  msg->peersUpdate.deadPeersHash = deadPeersHash;
  msg->peersUpdate.nPeers = static_cast<int>(peers.size());
  msg->peersUpdate.nDeadPeers = static_cast<int>(deadPeers.size());
  if (!peers.empty()) memcpy(msg->peersUpdate.peers, peers.data(), peers.size() * sizeof(rasPeerInfo));
  if (!deadPeers.empty()) {
    memcpy(reinterpret_cast<char*>(msg) + deadOffset, deadPeers.data(), deadPeers.size() * sizeof(ncclSocketAddress));
  }
  return msg;
}

const ncclSocketAddress* DeadPeersInMsg(const rasMsg* msg) {
  int offset = static_cast<int>(rasMsgLength(RAS_MSG_PEERSUPDATE)) +
               msg->peersUpdate.nPeers * static_cast<int>(sizeof(rasPeerInfo));
  ALIGN_SIZE(offset, alignof(ncclSocketAddress));
  return reinterpret_cast<const ncclSocketAddress*>(reinterpret_cast<const char*>(msg) + offset);
}

void ResetState() {
  for (const EnqueuedMsg& entry : g_enqueuedMsgs) rasMsgFree(entry.msg);
  g_enqueuedMsgs.clear();
  FreeLink(&rasNextLink);
  FreeLink(&rasPrevLink);
  for (rasConnection* conn : g_connections) {
    free(conn->sock);
    free(conn);
  }
  g_connections.clear();
  g_createdAddrs.clear();
  g_disconnectedAddrs.clear();
  g_events.clear();
  rasPeersTerminate();
  memset(&rasNetListeningSocket, 0, sizeof(rasNetListeningSocket));
  memset(rasLine, 0, sizeof(rasLine));
  rasNextLink.direction = 1;
  rasPrevLink.direction = -1;
  g_connCreateResult = ncclSuccess;
  g_msgAllocResult = ncclSuccess;
  g_addFallbackResult = ncclSuccess;
  g_addFallbackCalls = 0;
  g_allocationCalls = 0;
  g_failAllocationCall = 0;
  g_clockNano = 123456;
}

class RasPeersMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetState(); }
  void TearDown() override { ResetState(); }
};

}  // namespace

uint64_t PeersTestClockNano() { return static_cast<uint64_t>(g_clockNano); }

const char* ncclSocketToHost(const ncclSocketAddress* addr, char* buf, size_t size) {
  if (addr->sa.sa_family <= 0) {
    if (size > 0) buf[0] = '\0';
    return buf;
  }
  const void* source = addr->sa.sa_family == AF_INET ? static_cast<const void*>(&addr->sin.sin_addr)
                                                     : static_cast<const void*>(&addr->sin6.sin6_addr);
  return inet_ntop(addr->sa.sa_family, source, buf, size);
}

const char* ncclSocketToString(const ncclSocketAddress* addr, char* buf, const int) {
  if (buf == nullptr || addr == nullptr || (addr->sa.sa_family != AF_INET && addr->sa.sa_family != AF_INET6)) {
    if (buf) buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST]{};
  char service[NI_MAXSERV]{};
  if (getnameinfo(&addr->sa, sizeof(ncclSocketAddress), host, sizeof(host), service, sizeof(service),
                  NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
    buf[0] = '\0';
    return buf;
  }
  snprintf(buf, SOCKET_NAME_MAXLEN + 1, "%s<%s>", host, service);
  return buf;
}

const char* rasGpuDevsToString(uint64_t cudaDevs, uint64_t nvmlDevs, char* buf, size_t size) {
  bool first = true;
  buf[0] = '\0';
  for (int index = 0; index < static_cast<int>(sizeof(cudaDevs) * 8); index++) {
    if (cudaDevs & (1ULL << index)) {
      snprintf(buf + strlen(buf), size - strlen(buf), "%s%d", first ? "" : ",", index);
      first = false;
    }
  }
  if (cudaDevs != nvmlDevs) {
    snprintf(buf + strlen(buf), size - strlen(buf), " (NVML ");
    first = true;
    for (int index = 0; index < static_cast<int>(sizeof(nvmlDevs) * 8); index++) {
      if (nvmlDevs & (1ULL << index)) {
        snprintf(buf + strlen(buf), size - strlen(buf), "%s%d", first ? "" : ",", index);
        first = false;
      }
    }
    snprintf(buf + strlen(buf), size - strlen(buf), ")");
  }
  return buf;
}

void rasClientsNotifyEvent(rasEventGroup, const rasEventNotification* event) {
  EventRecord record{};
  record.type = event->eventType;
  record.hasPeer = event->peerInfo != nullptr;
  if (record.hasPeer) record.peer = *event->peerInfo;
  record.hasAddr = event->peerAddr != nullptr;
  if (record.hasAddr) record.addr = *event->peerAddr;
  g_events.push_back(record);
}

ncclResult_t rasMsgAlloc(rasMsg** msg, size_t msgLen) {
  if (g_msgAllocResult != ncclSuccess) return g_msgAllocResult;
  const size_t total = offsetof(rasMsgMeta, msg) + msgLen;
  auto* meta = static_cast<rasMsgMeta*>(calloc(1, total));
  if (meta == nullptr) return ncclSystemError;
  *msg = &meta->msg;
  return ncclSuccess;
}

void rasMsgFree(rasMsg* msg) {
  if (msg == nullptr) return;
  free(reinterpret_cast<char*>(msg) - offsetof(rasMsgMeta, msg));
}

void rasConnEnqueueMsg(rasConnection* conn, rasMsg* msg, size_t msgLen, bool front) {
  g_enqueuedMsgs.push_back({conn, msg, msgLen, front});
}

rasConnection* rasConnFind(const ncclSocketAddress* addr) {
  for (rasConnection* conn : g_connections) {
    if (ncclSocketsCompare(&conn->addr, addr) == 0) return conn;
  }
  return nullptr;
}

ncclResult_t rasConnCreate(const ncclSocketAddress* addr, rasConnection** conn) {
  g_createdAddrs.push_back(*addr);
  if (g_connCreateResult != ncclSuccess) return g_connCreateResult;
  *conn = AddConnection(*addr, RAS_SOCK_CONNECTING);
  return ncclSuccess;
}

void rasConnDisconnect(const ncclSocketAddress* addr) { g_disconnectedAddrs.push_back(*addr); }

ncclResult_t rasLinkAddFallback(rasLink*, const rasConnection*) {
  ++g_addFallbackCalls;
  return g_addFallbackResult;
}

TEST_F(RasPeersMicrotest, SocketComparisonOrdersFamiliesAddressesAndPorts) {
  const ncclSocketAddress empty{};
  const ncclSocketAddress a = MakeIpv4("10.0.0.1", 10);
  const ncclSocketAddress b = MakeIpv4("10.0.0.1", 11);
  const ncclSocketAddress c = MakeIpv4("10.0.0.2", 1);
  const ncclSocketAddress lowPort = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress highPort = MakeIpv4("10.0.0.1", 256);
  const ncclSocketAddress lowAddress = MakeIpv4("10.0.0.255", 1);
  const ncclSocketAddress highAddress = MakeIpv4("10.0.1.0", 1);
  const ncclSocketAddress v6 = MakeIpv6("::1", 1);

  EXPECT_LT(ncclSocketsCompare(&a, &b), 0);
  EXPECT_LT(ncclSocketsCompare(&b, &c), 0);
  EXPECT_LT(ncclSocketsCompare(&lowPort, &highPort), 0);
  EXPECT_LT(ncclSocketsCompare(&lowAddress, &highAddress), 0);
  EXPECT_LT(ncclSocketsCompare(&a, &v6), 0);
  EXPECT_LT(ncclSocketsCompare(&a, &empty), 0);
  EXPECT_GT(ncclSocketsCompare(&empty, &a), 0);
  EXPECT_EQ(0, ncclSocketsCompare(&empty, &empty));
}

TEST_F(RasPeersMicrotest, SocketComparisonOrdersIpv6AddressThenPort) {
  const ncclSocketAddress a = MakeIpv6("2001:db8::1", 10);
  const ncclSocketAddress b = MakeIpv6("2001:db8::1", 11);
  const ncclSocketAddress c = MakeIpv6("2001:db8::2", 1);
  EXPECT_LT(ncclSocketsCompare(&a, &b), 0);
  EXPECT_LT(ncclSocketsCompare(&b, &c), 0);
}

TEST_F(RasPeersMicrotest, SameNodeIgnoresPortAndRequiresMatchingFamilyAndAddress) {
  const ncclSocketAddress a = MakeIpv4("10.0.0.1", 10);
  const ncclSocketAddress same = MakeIpv4("10.0.0.1", 20);
  const ncclSocketAddress other = MakeIpv4("10.0.0.2", 10);
  const ncclSocketAddress v6 = MakeIpv6("::1", 10);
  const ncclSocketAddress empty{};
  EXPECT_TRUE(ncclSocketsSameNode(&a, &same));
  EXPECT_FALSE(ncclSocketsSameNode(&a, &other));
  EXPECT_FALSE(ncclSocketsSameNode(&a, &v6));
  EXPECT_TRUE(ncclSocketsSameNode(&empty, &empty));
}

TEST_F(RasPeersMicrotest, SameNodeSupportsIpv6Addresses) {
  const ncclSocketAddress a = MakeIpv6("2001:db8::1", 10);
  const ncclSocketAddress same = MakeIpv6("2001:db8::1", 20);
  const ncclSocketAddress other = MakeIpv6("2001:db8::2", 10);
  EXPECT_TRUE(ncclSocketsSameNode(&a, &same));
  EXPECT_FALSE(ncclSocketsSameNode(&a, &other));
}

TEST_F(RasPeersMicrotest, RankConversionSortsMergesDevicesAndSkipsEmptyEntries) {
  const ncclSocketAddress a = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress b = MakeIpv4("10.0.0.2", 1);
  rasRankInit ranks[] = {MakeRank(b, 22, 3, 4), rasRankInit{}, MakeRank(a, 11, 2, 5), MakeRank(a, 11, 0, 1)};
  rasPeerInfo* rankPeers = nullptr;
  int nRankPeers = -1;
  int newCount = -1;

  ASSERT_EQ(ncclSuccess, rasRanksConvertToPeers(ranks, 4, &rankPeers, &nRankPeers, &newCount));
  ASSERT_EQ(2, nRankPeers);
  EXPECT_EQ(2, newCount);
  EXPECT_EQ(0, ncclSocketsCompare(&rankPeers[0].addr, &a));
  EXPECT_EQ((1ULL << 0) | (1ULL << 2), rankPeers[0].cudaDevs);
  EXPECT_EQ((1ULL << 1) | (1ULL << 5), rankPeers[0].nvmlDevs);
  EXPECT_EQ(11, rankPeers[0].pid);
  EXPECT_EQ(111u, rankPeers[0].hostHash);
  EXPECT_EQ(211u, rankPeers[0].pidHash);
  EXPECT_EQ(0, ncclSocketsCompare(&rankPeers[1].addr, &b));
  EXPECT_EQ(22, rankPeers[1].pid);
  EXPECT_EQ(122u, rankPeers[1].hostHash);
  EXPECT_EQ(222u, rankPeers[1].pidHash);
  free(rankPeers);
}

TEST_F(RasPeersMicrotest, RankConversionPropagatesAllocationFailure) {
  rasRankInit rank = MakeRank(MakeIpv4("10.0.0.1", 1), 11, 0, 0);
  rasPeerInfo* rankPeers = reinterpret_cast<rasPeerInfo*>(0x1);
  int nRankPeers = -1;
  int newCount = -1;
  g_failAllocationCall = 1;

  EXPECT_EQ(ncclSystemError, rasRanksConvertToPeers(&rank, 1, &rankPeers, &nRankPeers, &newCount));
  EXPECT_EQ(nullptr, rankPeers);
  EXPECT_EQ(1, g_allocationCalls);
}

TEST_F(RasPeersMicrotest, RankConversionDropsPidMismatchAgainstExistingPeer) {
  const ncclSocketAddress addr = MakeIpv4("10.0.0.1", 1);
  SetPeers({MakePeer(addr, 10)}, 0);
  rasRankInit rank = MakeRank(addr, 11, 1, 1);
  rasPeerInfo* rankPeers = nullptr;
  int nRankPeers = -1;
  int newCount = -1;
  ASSERT_EQ(ncclSuccess, rasRanksConvertToPeers(&rank, 1, &rankPeers, &nRankPeers, &newCount));
  EXPECT_EQ(0, nRankPeers);
  EXPECT_EQ(1, newCount);
  free(rankPeers);
}

TEST_F(RasPeersMicrotest, RankConversionFindsMatchingPeerAfterLowerExistingPeer) {
  const ncclSocketAddress lower = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress matching = MakeIpv4("10.0.0.2", 1);
  SetPeers({MakePeer(lower, 10), MakePeer(matching, 20, 1, 1)}, 0);
  rasRankInit rank = MakeRank(matching, 20, 2, 3);
  rasPeerInfo* rankPeers = nullptr;
  int nRankPeers = -1;
  int newCount = -1;

  ASSERT_EQ(ncclSuccess, rasRanksConvertToPeers(&rank, 1, &rankPeers, &nRankPeers, &newCount));
  ASSERT_EQ(1, nRankPeers);
  EXPECT_EQ(2, newCount);
  ASSERT_EQ(ncclSuccess, rasPeersUpdate(rankPeers, &nRankPeers, newCount));
  ASSERT_EQ(2, nRasPeers);
  EXPECT_EQ((1ULL << 0) | (1ULL << 2), rasPeers[1].cudaDevs);
  EXPECT_EQ((1ULL << 0) | (1ULL << 3), rasPeers[1].nvmlDevs);
  free(rankPeers);
}

TEST_F(RasPeersMicrotest, RankConversionCountsIncomingPeerBeforeExistingRegistry) {
  const ncclSocketAddress incoming = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress existing = MakeIpv4("10.0.0.2", 1);
  SetPeers({MakePeer(existing, 20)}, 0);
  rasRankInit rank = MakeRank(incoming, 10, 0, 0);
  rasPeerInfo* rankPeers = nullptr;
  int nRankPeers = -1;
  int newCount = -1;

  ASSERT_EQ(ncclSuccess, rasRanksConvertToPeers(&rank, 1, &rankPeers, &nRankPeers, &newCount));
  ASSERT_EQ(1, nRankPeers);
  EXPECT_EQ(2, newCount);
  EXPECT_EQ(0, ncclSocketsCompare(&rankPeers[0].addr, &incoming));
  free(rankPeers);
}

TEST_F(RasPeersMicrotest, PeersUpdateBuildsInitialSortedRegistryAndFindsSelf) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.2", 1);
  rasNetListeningSocket.addr = self;
  rasPeerInfo updates[] = {MakePeer(MakeIpv4("10.0.0.3", 1), 3), MakePeer(self, 2),
                           MakePeer(MakeIpv4("10.0.0.1", 1), 1)};
  qsort(updates, 3, sizeof(updates[0]), rasAddrPeerInfoCompare);
  int count = 3;
  ASSERT_EQ(ncclSuccess, rasPeersUpdate(updates, &count));
  ASSERT_EQ(3, nRasPeers);
  EXPECT_EQ(1, myPeerIdx);
  EXPECT_EQ(3, count);
  ASSERT_EQ(3u, g_events.size());
  for (size_t index = 0; index < g_events.size(); index++) {
    EXPECT_EQ("PEER_NEW", g_events[index].type);
    ASSERT_TRUE(g_events[index].hasPeer);
    EXPECT_EQ(rasPeers[index].pid, g_events[index].peer.pid);
    EXPECT_EQ(rasPeers[index].hostHash, g_events[index].peer.hostHash);
    EXPECT_EQ(rasPeers[index].pidHash, g_events[index].peer.pidHash);
    EXPECT_EQ(0, ncclSocketsCompare(&rasPeers[index].addr, &g_events[index].peer.addr));
  }
  EXPECT_EQ(1, rasPeerFind(&self));
}

TEST_F(RasPeersMicrotest, PeersUpdatePropagatesRegistryAllocationFailure) {
  rasPeerInfo update = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  int count = 1;
  g_failAllocationCall = 1;

  EXPECT_EQ(ncclSystemError, rasPeersUpdate(&update, &count));
  EXPECT_EQ(nullptr, rasPeers);
  EXPECT_EQ(0, nRasPeers);
  EXPECT_EQ(0u, rasPeersHash);
  EXPECT_EQ(1, count);
  EXPECT_TRUE(g_events.empty());
  EXPECT_EQ(1, g_allocationCalls);
}

TEST_F(RasPeersMicrotest, PeersUpdateMergesOnlyNewDeviceBitsIntoDiff) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  SetPeers({MakePeer(self, 1, 1, 1)}, 0);
  rasPeerInfo update = MakePeer(self, 1, 5, 3);
  int count = 1;
  ASSERT_EQ(ncclSuccess, rasPeersUpdate(&update, &count));
  EXPECT_EQ(5u, rasPeers[0].cudaDevs);
  EXPECT_EQ(3u, rasPeers[0].nvmlDevs);
  EXPECT_EQ(4u, update.cudaDevs);
  EXPECT_EQ(2u, update.nvmlDevs);
  EXPECT_EQ(1, count);

  update = MakePeer(self, 1, 1, 1);
  count = 1;
  ASSERT_EQ(ncclSuccess, rasPeersUpdate(&update, &count));
  EXPECT_EQ(0, count);
}

TEST_F(RasPeersMicrotest, PeersUpdateInsertsAroundExistingSelfAndTracksNewIndex) {
  const ncclSocketAddress a = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress self = MakeIpv4("10.0.0.2", 1);
  const ncclSocketAddress c = MakeIpv4("10.0.0.3", 1);
  SetPeers({MakePeer(self, 2)}, 0);
  rasPeerInfo updates[] = {MakePeer(a, 1), MakePeer(c, 3)};
  int count = 2;
  ASSERT_EQ(ncclSuccess, rasPeersUpdate(updates, &count));
  ASSERT_EQ(3, nRasPeers);
  EXPECT_EQ(1, myPeerIdx);
  EXPECT_EQ(0, ncclSocketsCompare(&rasPeers[0].addr, &a));
  EXPECT_EQ(0, ncclSocketsCompare(&rasPeers[2].addr, &c));
  ASSERT_EQ(2u, g_events.size());
  EXPECT_EQ("PEER_NEW", g_events[0].type);
  ASSERT_TRUE(g_events[0].hasPeer);
  EXPECT_EQ(0, ncclSocketsCompare(&g_events[0].peer.addr, &a));
  EXPECT_EQ("PEER_NEW", g_events[1].type);
  ASSERT_TRUE(g_events[1].hasPeer);
  EXPECT_EQ(0, ncclSocketsCompare(&g_events[1].peer.addr, &c));
}

TEST_F(RasPeersMicrotest, PeersUpdateCopiesTrailingExistingPeers) {
  const ncclSocketAddress inserted = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress self = MakeIpv4("10.0.0.2", 1);
  const ncclSocketAddress trailing = MakeIpv4("10.0.0.3", 1);
  SetPeers({MakePeer(self, 2), MakePeer(trailing, 3)}, 0);
  rasPeerInfo update = MakePeer(inserted, 1);
  int count = 1;
  ASSERT_EQ(ncclSuccess, rasPeersUpdate(&update, &count));
  ASSERT_EQ(3, nRasPeers);
  EXPECT_EQ(1, myPeerIdx);
  EXPECT_EQ(0, ncclSocketsCompare(&rasPeers[2].addr, &trailing));
}

TEST_F(RasPeersMicrotest, PeersUpdateCompactsDiffAfterUnchangedEntry) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress remote = MakeIpv4("10.0.0.2", 1);
  SetPeers({MakePeer(self, 1, 1, 1)}, 0);
  rasPeerInfo updates[] = {MakePeer(self, 1, 1, 1), MakePeer(remote, 2, 2, 2)};
  int count = 2;
  ASSERT_EQ(ncclSuccess, rasPeersUpdate(updates, &count));
  ASSERT_EQ(1, count);
  EXPECT_EQ(0, ncclSocketsCompare(&updates[0].addr, &remote));
}

TEST_F(RasPeersMicrotest, LocalAddRanksMergesAndReinitializesLinks) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress remote = MakeIpv4("10.0.0.2", 1);
  rasNetListeningSocket.addr = self;
  auto* ranks = static_cast<rasRankInit*>(calloc(2, sizeof(rasRankInit)));
  ranks[0] = MakeRank(remote, 2, 0, 0);
  ranks[1] = MakeRank(self, 1, 0, 0);

  ASSERT_EQ(ncclSuccess, rasLocalHandleAddRanks(ranks, 2));
  EXPECT_EQ(2, nRasPeers);
  EXPECT_EQ(0, myPeerIdx);
  EXPECT_EQ(1u, g_createdAddrs.size());
  ASSERT_NE(nullptr, rasNextLink.conns);
  ASSERT_NE(nullptr, rasPrevLink.conns);
}

TEST_F(RasPeersMicrotest, LocalAddRanksFreesConvertedPeersWhenRegistryAllocationFails) {
  auto* ranks = static_cast<rasRankInit*>(calloc(1, sizeof(rasRankInit)));
  ranks[0] = MakeRank(MakeIpv4("10.0.0.1", 1), 1, 0, 0);
  g_failAllocationCall = 2;

  EXPECT_EQ(ncclSystemError, rasLocalHandleAddRanks(ranks, 1));
  EXPECT_EQ(2, g_allocationCalls);
  EXPECT_EQ(nullptr, rasPeers);
  EXPECT_EQ(0, nRasPeers);
  EXPECT_EQ(0u, rasPeersHash);
  EXPECT_EQ(nullptr, rasNextLink.conns);
  EXPECT_EQ(nullptr, rasPrevLink.conns);
  EXPECT_TRUE(g_events.empty());
}

TEST_F(RasPeersMicrotest, SendPeersUpdateSkipsWhenHashesAreKnownFromEitherDirection) {
  const rasPeerInfo peer = MakePeer(MakeIpv4("10.0.0.2", 2), 2);
  SetDeadPeers({MakeIpv4("10.0.0.3", 3)});
  rasPeersHash = 10;
  rasDeadPeersHash = 20;

  rasConnection sentPeersRecvDead{};
  sentPeersRecvDead.lastSentPeersHash = 10;
  sentPeersRecvDead.lastRecvDeadPeersHash = 20;
  ASSERT_EQ(ncclSuccess, rasConnSendPeersUpdate(&sentPeersRecvDead, &peer, 1));
  EXPECT_TRUE(g_enqueuedMsgs.empty());

  rasConnection recvPeersSentDead{};
  recvPeersSentDead.lastRecvPeersHash = 10;
  recvPeersSentDead.lastSentDeadPeersHash = 20;
  ASSERT_EQ(ncclSuccess, rasConnSendPeersUpdate(&recvPeersSentDead, &peer, 1));
  EXPECT_TRUE(g_enqueuedMsgs.empty());
}

TEST_F(RasPeersMicrotest, SendPeersUpdateSerializesPeersAndDeadPeers) {
  const rasPeerInfo peer = MakePeer(MakeIpv4("10.0.0.2", 2), 2);
  const ncclSocketAddress dead = MakeIpv4("10.0.0.3", 3);
  SetDeadPeers({dead});
  rasPeersHash = 10;
  rasDeadPeersHash = 20;
  rasConnection conn{};

  ASSERT_EQ(ncclSuccess, rasConnSendPeersUpdate(&conn, &peer, 1));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  const rasMsg* msg = g_enqueuedMsgs[0].msg;
  EXPECT_EQ(RAS_MSG_PEERSUPDATE, msg->type);
  EXPECT_EQ(10u, msg->peersUpdate.peersHash);
  EXPECT_EQ(20u, msg->peersUpdate.deadPeersHash);
  EXPECT_EQ(1, msg->peersUpdate.nPeers);
  EXPECT_EQ(1, msg->peersUpdate.nDeadPeers);
  EXPECT_EQ(10u, conn.lastSentPeersHash);
  EXPECT_EQ(20u, conn.lastSentDeadPeersHash);
  EXPECT_EQ(0, ncclSocketsCompare(&msg->peersUpdate.peers[0].addr, &peer.addr));
  EXPECT_EQ(0, ncclSocketsCompare(DeadPeersInMsg(msg), &dead));
  int expectedLen = static_cast<int>(rasMsgLength(RAS_MSG_PEERSUPDATE) + sizeof(rasPeerInfo));
  ALIGN_SIZE(expectedLen, alignof(ncclSocketAddress));
  expectedLen += sizeof(ncclSocketAddress);
  EXPECT_EQ(static_cast<size_t>(expectedLen), g_enqueuedMsgs[0].len);
}

TEST_F(RasPeersMicrotest, ConnectionPropagationRequiresReadyNonParticipant) {
  const ncclSocketAddress addr = MakeIpv4("10.0.0.2", 2);
  const rasPeerInfo peer = MakePeer(addr, 2);
  rasConnection* conn = AddConnection(addr, RAS_SOCK_CONNECTING);
  rasPeersHash = 10;
  ASSERT_EQ(ncclSuccess, rasConnPropagateUpdate(conn, &peer, 1, false, nullptr, 0));
  EXPECT_TRUE(g_enqueuedMsgs.empty());

  conn->sock->status = RAS_SOCK_READY;
  rasRankInit ranks[] = {MakeRank(MakeIpv4("10.0.0.1", 1), 1, 0, 0), MakeRank(addr, 2, 0, 0),
                         MakeRank(MakeIpv4("10.0.0.3", 3), 3, 0, 0)};
  ASSERT_EQ(ncclSuccess, rasConnPropagateUpdate(conn, &peer, 1, false, ranks, 3));
  EXPECT_TRUE(g_enqueuedMsgs.empty());

  ASSERT_EQ(ncclSuccess, rasConnPropagateUpdate(conn, &peer, 1, true, ranks, 3));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(conn, g_enqueuedMsgs[0].conn);
  ASSERT_EQ(1, g_enqueuedMsgs[0].msg->peersUpdate.nPeers);
  EXPECT_EQ(0, ncclSocketsCompare(&g_enqueuedMsgs[0].msg->peersUpdate.peers[0].addr, &peer.addr));
}

TEST_F(RasPeersMicrotest, ConnectionPropagationSendsWhenRanksOmitConnectionPeer) {
  const ncclSocketAddress connectionAddr = MakeIpv4("10.0.0.2", 2);
  const rasPeerInfo peer = MakePeer(MakeIpv4("10.0.0.4", 4), 4);
  rasConnection* conn = AddConnection(connectionAddr);
  rasRankInit ranks[] = {MakeRank(MakeIpv4("10.0.0.1", 1), 1, 0, 0),
                         MakeRank(MakeIpv4("10.0.0.3", 3), 3, 0, 0)};
  rasPeersHash = 10;

  ASSERT_EQ(ncclSuccess, rasConnPropagateUpdate(conn, &peer, 1, false, ranks, 2));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(conn, g_enqueuedMsgs[0].conn);
  EXPECT_EQ(1, g_enqueuedMsgs[0].msg->peersUpdate.nPeers);
}

TEST_F(RasPeersMicrotest, ConnectionPropagationSkipsConnectionWithoutSocket) {
  const rasPeerInfo peer = MakePeer(MakeIpv4("10.0.0.2", 2), 2);
  rasConnection* conn = AddConnection(MakeIpv4("10.0.0.3", 3));
  free(conn->sock);
  conn->sock = nullptr;
  rasPeersHash = 10;

  EXPECT_EQ(ncclSuccess, rasConnPropagateUpdate(conn, &peer, 1, false, nullptr, 0));
  EXPECT_TRUE(g_enqueuedMsgs.empty());
}

TEST_F(RasPeersMicrotest, LinkPropagationSkipsOriginatingConnection) {
  const rasPeerInfo peer = MakePeer(MakeIpv4("10.0.0.4", 4), 4);
  rasConnection* origin = AddConnection(MakeIpv4("10.0.0.2", 2));
  rasConnection* other = AddConnection(MakeIpv4("10.0.0.3", 3));
  AddLinkConn(&rasNextLink, nullptr);
  AddLinkConn(&rasNextLink, origin);
  AddLinkConn(&rasNextLink, other);
  rasPeersHash = 10;
  ASSERT_EQ(ncclSuccess, rasLinkPropagateUpdate(&rasNextLink, &peer, 1, false, nullptr, 0, origin));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(other, g_enqueuedMsgs[0].conn);
  ASSERT_EQ(1, g_enqueuedMsgs[0].msg->peersUpdate.nPeers);
  EXPECT_EQ(0, ncclSocketsCompare(&g_enqueuedMsgs[0].msg->peersUpdate.peers[0].addr, &peer.addr));
}

TEST_F(RasPeersMicrotest, LinkReinitPropagatesConnectionCreationFailure) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo remote = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  SetPeers({self, remote}, 0);
  g_connCreateResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasLinkReinitConns(&rasNextLink));
}

TEST_F(RasPeersMicrotest, LinkReinitPreviousPropagatesConnectionCreationFailure) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo remote = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  SetPeers({self, remote}, 0);
  g_connCreateResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasLinkReinitConns(&rasPrevLink));
  ASSERT_EQ(1u, g_createdAddrs.size());
  EXPECT_EQ(0, ncclSocketsCompare(&g_createdAddrs[0], &remote.addr));
}

TEST_F(RasPeersMicrotest, NetUpdatePeersPropagatesNextLinkConnectionCreationFailure) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo remote = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  SetPeers({self, remote}, 0);
  g_connCreateResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasNetUpdatePeers(&remote, 1, false));
  ASSERT_NE(nullptr, rasNextLink.conns);
  EXPECT_EQ(nullptr, rasNextLink.conns->conn);
  EXPECT_EQ(nullptr, rasPrevLink.conns);
}

TEST_F(RasPeersMicrotest, NetUpdatePeersPropagatesPreviousLinkConnectionCreationFailure) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo next = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  const rasPeerInfo previous = MakePeer(MakeIpv4("10.0.0.3", 1), 3);
  SetPeers({self, next, previous}, 0);
  rasConnection* nextConn = AddConnection(next.addr);
  g_connCreateResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasNetUpdatePeers(&previous, 1, false));
  ASSERT_NE(nullptr, rasNextLink.conns);
  EXPECT_EQ(nextConn, rasNextLink.conns->conn);
  ASSERT_NE(nullptr, rasPrevLink.conns);
  EXPECT_EQ(nullptr, rasPrevLink.conns->conn);
  ASSERT_EQ(1u, g_createdAddrs.size());
  EXPECT_EQ(0, ncclSocketsCompare(&g_createdAddrs[0], &previous.addr));
}

TEST_F(RasPeersMicrotest, LinkReinitPropagatesEntryAllocationFailure) {
  g_failAllocationCall = 1;
  EXPECT_EQ(ncclSystemError, rasLinkReinitConns(&rasNextLink));
  EXPECT_EQ(nullptr, rasNextLink.conns);
  EXPECT_EQ(1, g_allocationCalls);
}

TEST_F(RasPeersMicrotest, SendPeersUpdatePropagatesAllocationFailure) {
  const rasPeerInfo peer = MakePeer(MakeIpv4("10.0.0.2", 2), 2);
  rasConnection conn{};
  rasPeersHash = 10;
  g_msgAllocResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasConnSendPeersUpdate(&conn, &peer, 1));
  EXPECT_TRUE(g_enqueuedMsgs.empty());
}

TEST_F(RasPeersMicrotest, HandlePeersUpdatePropagatesReplyAllocationFailure) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress remote = MakeIpv4("10.0.0.2", 2);
  SetPeers({MakePeer(self, 1)}, 0);
  rasConnection* conn = AddConnection(remote);
  rasMsg* msg = MakePeersUpdate({MakePeer(remote, 2)}, {}, 1234, rasDeadPeersHash);
  g_msgAllocResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasMsgHandlePeersUpdate(msg, conn->sock));
  EXPECT_EQ(1, nRasPeers);
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  rasMsgFree(msg);
}

TEST_F(RasPeersMicrotest, HandlePeersUpdateRejectsSocketWithoutConnection) {
  rasMsg* msg = MakePeersUpdate({}, {}, 0, 0);
  rasSocket sock{};
  EXPECT_EQ(ncclInternalError, rasMsgHandlePeersUpdate(msg, &sock));
  rasMsgFree(msg);
}

TEST_F(RasPeersMicrotest, HandlePeersUpdateWithMatchingHashesOnlyRecordsReceipt) {
  rasPeersHash = 10;
  rasDeadPeersHash = 20;
  rasConnection* conn = AddConnection(MakeIpv4("10.0.0.2", 2));
  rasMsg* msg = MakePeersUpdate({}, {}, 10, 20);
  ASSERT_EQ(ncclSuccess, rasMsgHandlePeersUpdate(msg, conn->sock));
  EXPECT_EQ(10u, conn->lastRecvPeersHash);
  EXPECT_EQ(20u, conn->lastRecvDeadPeersHash);
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  rasMsgFree(msg);
}

TEST_F(RasPeersMicrotest, HandlePeersUpdateMergesNewPeerWithoutEchoWhenHashMatches) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress remote = MakeIpv4("10.0.0.2", 2);
  SetPeers({MakePeer(self, 1)}, 0);
  rasConnection* conn = AddConnection(remote);
  const rasPeerInfo newPeer = MakePeer(remote, 2);
  const rasPeerInfo merged[] = {rasPeers[0], newPeer};
  const uint64_t mergedHash = getHash(reinterpret_cast<const char*>(merged), sizeof(merged));
  rasMsg* msg = MakePeersUpdate({newPeer}, {}, mergedHash, rasDeadPeersHash);

  ASSERT_EQ(ncclSuccess, rasMsgHandlePeersUpdate(msg, conn->sock));
  EXPECT_EQ(2, nRasPeers);
  EXPECT_EQ(1, msg->peersUpdate.nPeers);
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  EXPECT_EQ(mergedHash, rasPeersHash);
  rasMsgFree(msg);
}

TEST_F(RasPeersMicrotest, HandlePeersUpdateMergesDeadPeersAndDisconnectsNewEntries) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress dead = MakeIpv4("10.0.0.3", 3);
  SetPeers({MakePeer(self, 1)}, 0);
  rasConnection* conn = AddConnection(MakeIpv4("10.0.0.2", 2));
  const uint64_t expectedHash = getHash(reinterpret_cast<const char*>(&dead), sizeof(dead));
  rasMsg* msg = MakePeersUpdate({}, {dead}, rasPeersHash, expectedHash);

  ASSERT_EQ(ncclSuccess, rasMsgHandlePeersUpdate(msg, conn->sock));
  EXPECT_EQ(1, nRasDeadPeers);
  ASSERT_EQ(1u, g_disconnectedAddrs.size());
  EXPECT_EQ(0, ncclSocketsCompare(&g_disconnectedAddrs[0], &dead));
  ASSERT_EQ(1u, g_events.size());
  EXPECT_EQ("PEER_DEAD", g_events[0].type);
  ASSERT_TRUE(g_events[0].hasAddr);
  EXPECT_EQ(0, ncclSocketsCompare(&g_events[0].addr, &dead));
  EXPECT_TRUE(g_enqueuedMsgs.empty());
  rasMsgFree(msg);
}

TEST_F(RasPeersMicrotest, HandlePeersUpdateEchoesDeadPeersWhenHashStillDiffers) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress dead = MakeIpv4("10.0.0.3", 3);
  SetPeers({MakePeer(self, 1)}, 0);
  rasConnection* conn = AddConnection(MakeIpv4("10.0.0.2", 2));
  rasMsg* msg = MakePeersUpdate({}, {dead}, rasPeersHash, 1234);

  ASSERT_EQ(ncclSuccess, rasMsgHandlePeersUpdate(msg, conn->sock));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(0, g_enqueuedMsgs[0].msg->peersUpdate.nPeers);
  EXPECT_EQ(1, g_enqueuedMsgs[0].msg->peersUpdate.nDeadPeers);
  EXPECT_EQ(rasDeadPeersHash, g_enqueuedMsgs[0].msg->peersUpdate.deadPeersHash);
  EXPECT_EQ(0, ncclSocketsCompare(DeadPeersInMsg(g_enqueuedMsgs[0].msg), &dead));
  EXPECT_EQ(rasDeadPeersHash, conn->lastSentDeadPeersHash);
  rasMsgFree(msg);
}

TEST_F(RasPeersMicrotest, HandlePeersUpdateEchoesLocalStateWhenHashesStillDiffer) {
  const ncclSocketAddress self = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress remote = MakeIpv4("10.0.0.2", 2);
  SetPeers({MakePeer(self, 1)}, 0);
  rasConnection* conn = AddConnection(remote);
  rasMsg* msg = MakePeersUpdate({MakePeer(remote, 2)}, {}, 1234, rasDeadPeersHash);

  ASSERT_EQ(ncclSuccess, rasMsgHandlePeersUpdate(msg, conn->sock));
  ASSERT_EQ(1u, g_enqueuedMsgs.size());
  EXPECT_EQ(rasPeersHash, g_enqueuedMsgs[0].msg->peersUpdate.peersHash);
  ASSERT_EQ(1, g_enqueuedMsgs[0].msg->peersUpdate.nPeers);
  EXPECT_EQ(0, ncclSocketsCompare(&g_enqueuedMsgs[0].msg->peersUpdate.peers[0].addr, &self));
  rasMsgFree(msg);
}

TEST_F(RasPeersMicrotest, LinkCalculatePeerMovesByDirectionAndSkipsDeadPeers) {
  const rasPeerInfo a = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo b = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  const rasPeerInfo c = MakePeer(MakeIpv4("10.0.0.3", 1), 3);
  SetPeers({a, b, c}, 0);
  EXPECT_EQ(1, rasLinkCalculatePeer(&rasNextLink, 0, false));
  EXPECT_EQ(2, rasLinkCalculatePeer(&rasPrevLink, 0, false));
  SetDeadPeers({b.addr});
  EXPECT_EQ(2, rasLinkCalculatePeer(&rasNextLink, 0, false));
  SetDeadPeers({b.addr, c.addr});
  EXPECT_EQ(-1, rasLinkCalculatePeer(&rasNextLink, 0, false));
}

TEST_F(RasPeersMicrotest, FallbackSkipsAnUnresponsiveRemoteNode) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo first = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  const rasPeerInfo sameNode = MakePeer(MakeIpv4("10.0.0.2", 2), 3);
  const rasPeerInfo nextNode = MakePeer(MakeIpv4("10.0.0.3", 1), 4);
  SetPeers({self, first, sameNode, nextNode}, 0);
  rasConnection* sameNodeConn = AddConnection(sameNode.addr, RAS_SOCK_CONNECTING);
  EXPECT_EQ(3, rasLinkCalculatePeer(&rasNextLink, 1, true));

  free(sameNodeConn->sock);
  sameNodeConn->sock = nullptr;
  EXPECT_EQ(3, rasLinkCalculatePeer(&rasNextLink, 1, true));

  sameNodeConn->sock = static_cast<rasSocket*>(calloc(1, sizeof(rasSocket)));
  sameNodeConn->sock->conn = sameNodeConn;
  sameNodeConn->sock->status = RAS_SOCK_READY;
  sameNodeConn->experiencingDelays = true;
  EXPECT_EQ(3, rasLinkCalculatePeer(&rasNextLink, 1, true));

  sameNodeConn->experiencingDelays = false;
  EXPECT_EQ(2, rasLinkCalculatePeer(&rasNextLink, 1, true));

  SetDeadPeers({sameNode.addr});
  EXPECT_EQ(3, rasLinkCalculatePeer(&rasNextLink, 1, true));
}

TEST_F(RasPeersMicrotest, FallbackCanSelectCandidateOnLocalNode) {
  const rasPeerInfo lower = MakePeer(MakeIpv4("10.0.0.0", 1), 1);
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 2);
  const rasPeerInfo sameNode = MakePeer(MakeIpv4("10.0.0.1", 2), 3);
  const rasPeerInfo remoteFallback = MakePeer(MakeIpv4("10.0.0.2", 1), 4);
  SetPeers({lower, self, sameNode, remoteFallback}, 1);

  EXPECT_EQ(2, rasLinkCalculatePeer(&rasPrevLink, 3, true));
}

TEST_F(RasPeersMicrotest, LinkReinitWithOnlySelfLeavesNoPrimaryPeer) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  SetPeers({self}, 0);

  ASSERT_EQ(ncclSuccess, rasLinkReinitConns(&rasNextLink));
  ASSERT_NE(nullptr, rasNextLink.conns);
  EXPECT_EQ(-1, rasNextLink.conns->peerIdx);
  EXPECT_EQ(nullptr, rasNextLink.conns->conn);
  EXPECT_TRUE(g_createdAddrs.empty());
  EXPECT_EQ(0, g_addFallbackCalls);
}

TEST_F(RasPeersMicrotest, LinkReinitCreatesOrDefersPrimaryConnection) {
  const rasPeerInfo a = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo b = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  SetPeers({a, b}, 0);
  ASSERT_EQ(ncclSuccess, rasLinkReinitConns(&rasNextLink));
  EXPECT_EQ(1, rasNextLink.conns->peerIdx);
  ASSERT_NE(nullptr, rasNextLink.conns->conn);
  EXPECT_EQ(0, ncclSocketsCompare(&rasNextLink.conns->conn->addr, &b.addr));
  EXPECT_EQ(1u, g_createdAddrs.size());
  EXPECT_EQ(0, g_addFallbackCalls);

  FreeLink(&rasPrevLink);
  SetPeers({a, b}, 1);
  ASSERT_EQ(ncclSuccess, rasLinkReinitConns(&rasPrevLink));
  EXPECT_EQ(0, rasPrevLink.conns->peerIdx);
  EXPECT_EQ(g_clockNano, rasPrevLink.lastUpdatePeersTime);
}

TEST_F(RasPeersMicrotest, LinkReinitReusesDelayedConnectionAndAddsFallback) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo remote = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  SetPeers({self, remote}, 0);
  rasConnection* conn = AddConnection(remote.addr);
  conn->experiencingDelays = true;
  conn->startRetryTime = 10;
  AddLinkConn(&rasNextLink, conn, 1);
  AddLinkConn(&rasNextLink, conn, 1);
  rasNextLink.lastUpdatePeersTime = 999;

  ASSERT_EQ(ncclSuccess, rasLinkReinitConns(&rasNextLink));
  EXPECT_EQ(conn, rasNextLink.conns->conn);
  EXPECT_EQ(1, g_addFallbackCalls);
  EXPECT_EQ(nullptr, rasNextLink.conns->next);
  EXPECT_EQ(0, rasNextLink.lastUpdatePeersTime);
}

TEST_F(RasPeersMicrotest, LinkReinitPropagatesFallbackFailure) {
  const rasPeerInfo self = MakePeer(MakeIpv4("10.0.0.1", 1), 1);
  const rasPeerInfo remote = MakePeer(MakeIpv4("10.0.0.2", 1), 2);
  SetPeers({self, remote}, 0);
  rasConnection* conn = AddConnection(remote.addr);
  conn->experiencingDelays = true;
  AddLinkConn(&rasNextLink, conn, 1);
  g_addFallbackResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasLinkReinitConns(&rasNextLink));
  EXPECT_EQ(conn, rasNextLink.conns->conn);
  EXPECT_EQ(1, g_addFallbackCalls);
}

TEST_F(RasPeersMicrotest, DeclareDeadIsSortedIdempotentAndNotifiesOnce) {
  const ncclSocketAddress high = MakeIpv4("10.0.0.3", 1);
  const ncclSocketAddress low = MakeIpv4("10.0.0.2", 1);
  ASSERT_EQ(ncclSuccess, rasPeerDeclareDead(&high));
  ASSERT_EQ(ncclSuccess, rasPeerDeclareDead(&low));
  ASSERT_EQ(ncclSuccess, rasPeerDeclareDead(&high));
  ASSERT_EQ(2, nRasDeadPeers);
  EXPECT_LT(ncclSocketsCompare(&rasDeadPeers[0], &rasDeadPeers[1]), 0);
  EXPECT_EQ(getHash(reinterpret_cast<const char*>(rasDeadPeers), nRasDeadPeers * sizeof(*rasDeadPeers)),
            rasDeadPeersHash);
  EXPECT_TRUE(rasPeerIsDead(&high));
  ASSERT_EQ(2u, g_events.size());
  ASSERT_TRUE(g_events[0].hasAddr);
  ASSERT_TRUE(g_events[1].hasAddr);
  EXPECT_EQ(0, ncclSocketsCompare(&g_events[0].addr, &high));
  EXPECT_EQ(0, ncclSocketsCompare(&g_events[1].addr, &low));
}

TEST_F(RasPeersMicrotest, DeclareDeadPropagatesRegistryReallocationFailure) {
  const ncclSocketAddress peer = MakeIpv4("10.0.0.1", 1);
  g_failAllocationCall = 1;

  EXPECT_EQ(ncclSystemError, rasPeerDeclareDead(&peer));
  EXPECT_EQ(nullptr, rasDeadPeers);
  EXPECT_EQ(0, nRasDeadPeers);
  EXPECT_EQ(0, rasDeadPeersSize);
  EXPECT_EQ(0u, rasDeadPeersHash);
  EXPECT_TRUE(g_events.empty());
  EXPECT_EQ(1, g_allocationCalls);
}

TEST_F(RasPeersMicrotest, DeadPeersUpdateMergesDuplicatesAndDisconnectsOnlyNewPeers) {
  const ncclSocketAddress a = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress b = MakeIpv4("10.0.0.2", 1);
  const ncclSocketAddress c = MakeIpv4("10.0.0.3", 1);
  SetDeadPeers({b});
  ncclSocketAddress updates[] = {a, b, c};
  int count = 3;
  ASSERT_EQ(ncclSuccess, rasDeadPeersUpdate(updates, &count));
  EXPECT_EQ(2, count);
  EXPECT_EQ(3, nRasDeadPeers);
  ASSERT_EQ(2u, g_disconnectedAddrs.size());
  EXPECT_EQ(0, ncclSocketsCompare(&g_disconnectedAddrs[0], &a));
  EXPECT_EQ(0, ncclSocketsCompare(&g_disconnectedAddrs[1], &c));
  ASSERT_EQ(2u, g_events.size());
  ASSERT_TRUE(g_events[0].hasAddr);
  ASSERT_TRUE(g_events[1].hasAddr);
  EXPECT_EQ(0, ncclSocketsCompare(&g_events[0].addr, &a));
  EXPECT_EQ(0, ncclSocketsCompare(&g_events[1].addr, &c));
  EXPECT_EQ(getHash(reinterpret_cast<const char*>(rasDeadPeers), nRasDeadPeers * sizeof(*rasDeadPeers)),
            rasDeadPeersHash);
}

TEST_F(RasPeersMicrotest, DeadPeersUpdateKeepsOldPeersAfterUpdatesEnd) {
  ncclSocketAddress incoming = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress oldFirst = MakeIpv4("10.0.0.2", 1);
  const ncclSocketAddress oldLast = MakeIpv4("10.0.0.3", 1);
  SetDeadPeers({oldFirst, oldLast});
  int count = 1;

  ASSERT_EQ(ncclSuccess, rasDeadPeersUpdate(&incoming, &count));
  EXPECT_EQ(1, count);
  ASSERT_EQ(3, nRasDeadPeers);
  const ncclSocketAddress expected[] = {incoming, oldFirst, oldLast};
  EXPECT_EQ(0, memcmp(rasDeadPeers, expected, sizeof(expected)));
  ASSERT_EQ(1u, g_disconnectedAddrs.size());
  EXPECT_EQ(0, ncclSocketsCompare(&g_disconnectedAddrs[0], &incoming));
}

TEST_F(RasPeersMicrotest, DeadPeersUpdateGrowsRegistryForMultipleNewEntries) {
  const ncclSocketAddress a = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress b = MakeIpv4("10.0.0.2", 1);
  const ncclSocketAddress c = MakeIpv4("10.0.0.3", 1);
  const ncclSocketAddress d = MakeIpv4("10.0.0.4", 1);
  const ncclSocketAddress e = MakeIpv4("10.0.0.5", 1);
  const ncclSocketAddress f = MakeIpv4("10.0.0.6", 1);
  SetDeadPeers({b, c, d, e});
  ncclSocketAddress updates[] = {a, f};
  const ncclSocketAddress expected[] = {a, b, c, d, e, f};
  int count = 2;

  ASSERT_EQ(ncclSuccess, rasDeadPeersUpdate(updates, &count));
  EXPECT_EQ(2, count);
  ASSERT_EQ(6, nRasDeadPeers);
  EXPECT_EQ(8, rasDeadPeersSize);
  EXPECT_EQ(1, g_allocationCalls);
  EXPECT_EQ(0, memcmp(rasDeadPeers, expected, sizeof(expected)));
  EXPECT_EQ(getHash(reinterpret_cast<const char*>(expected), sizeof(expected)), rasDeadPeersHash);
  ASSERT_EQ(2u, g_disconnectedAddrs.size());
  ASSERT_EQ(2u, g_events.size());
}

TEST_F(RasPeersMicrotest, DeadPeersUpdatePropagatesRegistryAllocationFailure) {
  ncclSocketAddress update = MakeIpv4("10.0.0.1", 1);
  int count = 1;
  g_failAllocationCall = 1;

  EXPECT_EQ(ncclSystemError, rasDeadPeersUpdate(&update, &count));
  EXPECT_EQ(nullptr, rasDeadPeers);
  EXPECT_EQ(0, nRasDeadPeers);
  EXPECT_EQ(0, rasDeadPeersSize);
  EXPECT_EQ(0u, rasDeadPeersHash);
  EXPECT_EQ(1, count);
  EXPECT_TRUE(g_events.empty());
  EXPECT_EQ(1, g_allocationCalls);
}

TEST_F(RasPeersMicrotest, PeerFormattingUsesInventoryOrFallsBackToAddress) {
  const ncclSocketAddress addr = MakeIpv4("10.0.0.1", 123);
  const rasPeerInfo peer = MakePeer(addr, 77, 3, 5);
  const rasPeerInfo singleGpu = MakePeer(MakeIpv4("10.0.0.2", 124), 78, 1, 1);
  SetPeers({peer}, 0);
  char buf[256];
  EXPECT_EQ(buf, rasPeerInfoToString(&peer, buf, sizeof(buf)));
  EXPECT_STREQ("Process 77 on node 10.0.0.1 managing GPUs 0,1 (NVML 0,2)", buf);
  EXPECT_EQ(buf, rasPeerInfoToString(&singleGpu, buf, sizeof(buf)));
  EXPECT_STREQ("Process 78 on node 10.0.0.2 managing GPU 0", buf);
  EXPECT_EQ(buf, rasPeerToString(&addr, buf, sizeof(buf)));
  EXPECT_STREQ("Process 77 on node 10.0.0.1 managing GPUs 0,1 (NVML 0,2)", buf);

  const ncclSocketAddress unknown = MakeIpv4("10.0.0.9", 9);
  EXPECT_EQ(buf, rasPeerToString(&unknown, buf, sizeof(buf)));
  EXPECT_NE(std::string::npos, std::string(buf).find("10.0.0.9"));
}

TEST_F(RasPeersMicrotest, RankComparatorAndPeerDumpCoverDiagnosticCases) {
  const ncclSocketAddress addr = MakeIpv4("10.0.0.1", 1);
  const ncclSocketAddress laterAddr = MakeIpv4("10.0.0.2", 1);
  rasRankInit emptyA{};
  rasRankInit emptyB{};
  EXPECT_EQ(0, rasRanksCompare(&emptyA, &emptyB));

  rasRankInit lower = MakeRank(addr, 10, 0, 0);
  rasRankInit higher = MakeRank(addr, 11, 1, 1);
  EXPECT_LT(rasRanksCompare(&lower, &higher), 0);
  EXPECT_GT(rasRanksCompare(&higher, &lower), 0);
  higher.cudaDev = lower.cudaDev;
  EXPECT_EQ(0, rasRanksCompare(&lower, &higher));

  rasRankInit laterRank = MakeRank(laterAddr, 12, 0, 0);
  EXPECT_LT(rasAddrRankInitCompare(&addr, &laterRank), 0);
  EXPECT_GT(rasAddrRankInitCompare(&laterAddr, &lower), 0);
  EXPECT_EQ(0, rasAddrRankInitCompare(&addr, &lower));

  rasPeerInfo laterPeer = MakePeer(laterAddr, 12);
  rasPeerInfo matchingPeer = MakePeer(addr, 10);
  EXPECT_LT(rasAddrPeerInfoCompare(&addr, &laterPeer), 0);
  EXPECT_GT(rasAddrPeerInfoCompare(&laterAddr, &matchingPeer), 0);
  EXPECT_EQ(0, rasAddrPeerInfoCompare(&addr, &matchingPeer));

  const rasPeerInfo peer = MakePeer(addr, 10, 1, 1);
  char buf[256];
  EXPECT_EQ(buf, rasPeerDump(&peer, buf, sizeof(buf)));
  EXPECT_NE(std::string::npos, std::string(buf).find("socket 10.0.0.1<1>"));
  EXPECT_NE(std::string::npos, std::string(buf).find("pid 10"));
  EXPECT_NE(std::string::npos, std::string(buf).find("GPU 0"));
}

TEST_F(RasPeersMicrotest, TerminateClearsPeerAndDeadPeerState) {
  SetPeers({MakePeer(MakeIpv4("10.0.0.1", 1), 1)}, 0);
  SetDeadPeers({MakeIpv4("10.0.0.2", 2)});
  rasPeersTerminate();
  EXPECT_EQ(nullptr, rasPeers);
  EXPECT_EQ(0, nRasPeers);
  EXPECT_EQ(0u, rasPeersHash);
  EXPECT_EQ(nullptr, rasDeadPeers);
  EXPECT_EQ(0, nRasDeadPeers);
  EXPECT_EQ(0, rasDeadPeersSize);
  EXPECT_EQ(0u, rasDeadPeersHash);
  EXPECT_EQ(-1, myPeerIdx);
}
