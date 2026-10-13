/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * TLS encryption of NCCL-owned TCP sockets (ncclSetEncryption, src/misc/crypt.cc).
 *
 * Covers the public API contract and, on TLS_BACKEND=OPENSSL3 builds, the socket
 * paths RCCL's bootstrap drives differently from upstream NCCL:
 *   - the bootstrap ring is connected by socketRingConnectStart/Finish on ONE thread
 *     (async connect + async accept, polled with ncclSocketReady) instead of
 *     upstream's helper thread, so a blocking TLS handshake step would deadlock;
 *   - the bidirectional ring AllGather runs send and recv on the same socket pair at
 *     once through ncclSocketMultiOp.
 * It also checks that payload bytes never reach the wire in plaintext, and that
 * peers with a mismatched mode or key are dropped without killing the listener.
 *
 * The encryption mode is process-global, so every case runs in an isolated child.
 */

#include "crypt.h"
#include "socket.h"

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <functional>
#include <netinet/in.h>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace RcclUnitTesting {
namespace {

constexpr const char* kPskA = "rccl-socket-crypt-test-key-A-0123456789";
constexpr const char* kPskB = "rccl-socket-crypt-test-key-B-0123456789";
constexpr uint64_t kMagic = NCCL_SOCKET_MAGIC;
constexpr enum ncclSocketType kType = ncclSocketTypeBootstrap;

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
constexpr bool kTlsBuild = true;
#else
constexpr bool kTlsBuild = false;
#endif

#define SKIP_UNLESS_TLS_BUILD()                                                   \
  do {                                                                            \
    if (!kTlsBuild) GTEST_SKIP() << "RCCL was built without TLS_BACKEND=OPENSSL3"; \
  } while (0)

ncclResult_t SetPsk(const char* psk) {
  ncclEncryptionConfig_t cfg = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  cfg.mode = NCCL_ENCRYPTION_MODE_PSK;
  cfg.psk = psk;
  return ncclSetEncryption(&cfg);
}

ncclResult_t ClearPsk() { return ncclSetEncryption(nullptr); }

union ncclSocketAddress LoopbackAddr() {
  union ncclSocketAddress addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin.sin_family = AF_INET;
  addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin.sin_port = 0;
  return addr;
}

std::vector<char> Pattern(size_t size, unsigned seed) {
  std::vector<char> v(size);
  for (size_t i = 0; i < size; i++) v[i] = static_cast<char>((i * 131 + seed * 17 + (i >> 8)) & 0xff);
  return v;
}

// Payload made of a repeated ASCII marker, so a plaintext leak is trivially searchable.
std::vector<char> MarkerPayload(const std::string& marker, size_t size) {
  std::vector<char> v(size);
  for (size_t i = 0; i < size; i++) v[i] = marker[i % marker.size()];
  return v;
}

bool Contains(const std::string& haystack, const void* needle, size_t len) {
  return std::search(haystack.begin(), haystack.end(), (const char*)needle, (const char*)needle + len) !=
         haystack.end();
}

// ncclSocket holds a pointer to abortFlag, so a Listener must stay where it was built.
struct Listener {
  struct ncclSocket sock;
  union ncclSocketAddress addr;
  volatile uint32_t abortFlag = 0;
  Listener() { memset(&sock, 0, sizeof(sock)); }
  Listener(const Listener&) = delete;
  ~Listener() { ncclSocketClose(&sock); }
  bool Start() {
    union ncclSocketAddress a = LoopbackAddr();
    return ncclSocketInit(&sock, &a, kMagic, kType, &abortFlag) == ncclSuccess && ncclSocketListen(&sock) == ncclSuccess &&
           ncclSocketGetAddr(&sock, &addr) == ncclSuccess;
  }
};

struct Pair {
  struct ncclSocket client;
  struct ncclSocket server;
  Pair() {
    memset(&client, 0, sizeof(client));
    memset(&server, 0, sizeof(server));
  }
  Pair(const Pair&) = delete;
  ~Pair() {
    ncclSocketClose(&client);
    ncclSocketClose(&server);
  }
};

// Blocking connect on this thread, blocking accept on a helper thread: both handshake
// halves must progress at once. `via` redirects the client (e.g. through a WireTap).
bool ConnectPair(Listener& l, Pair& p, const union ncclSocketAddress* via = nullptr) {
  std::atomic<int> acceptRes{ncclInternalError};
  std::thread acceptor([&]() {
    if (ncclSocketInit(&p.server) == ncclSuccess) acceptRes = ncclSocketAccept(&p.server, &l.sock);
  });
  ncclResult_t res = ncclSocketInit(&p.client, via ? via : &l.addr, kMagic, kType);
  if (res == ncclSuccess) res = ncclSocketConnect(&p.client);
  if (res != ncclSuccess) l.abortFlag = 1;
  acceptor.join();
  return res == ncclSuccess && acceptRes == ncclSuccess && p.client.state == ncclSocketStateReady &&
         p.server.state == ncclSocketStateReady;
}

bool Transfer(struct ncclSocket* from, struct ncclSocket* to, size_t size, unsigned seed) {
  std::vector<char> out = Pattern(size, seed);
  std::vector<char> in(size, 0);
  std::atomic<int> recvRes{ncclInternalError};
  std::thread receiver([&]() { recvRes = ncclSocketRecv(to, in.data(), (int)size); });
  ncclResult_t sendRes = ncclSocketSend(from, out.data(), (int)size);
  if (sendRes != ncclSuccess) ncclSocketShutdown(from, SHUT_RDWR);
  receiver.join();
  return sendRes == ncclSuccess && recvRes == ncclSuccess && in == out;
}

// Polls until the socket fails or the peer closes it, for up to `timeoutMs`.
bool PeerDropped(struct ncclSocket* sock, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  char byte;
  while (std::chrono::steady_clock::now() < deadline) {
    bool closed = false;
    if (ncclSocketTryRecv(sock, &byte, 1, &closed, /*blocking=*/false) != ncclSuccess || closed) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

// Plain TCP relay between a client and a target listener that records every byte it
// forwards, i.e. what an observer on the wire would see.
class WireTap {
public:
  ~WireTap() { Stop(); }
  bool Start(const union ncclSocketAddress& target) {
    target_ = target;
    listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(a);
    if (listenFd_ < 0 || bind(listenFd_, (struct sockaddr*)&a, sizeof(a)) != 0 || listen(listenFd_, 4) != 0 ||
        getsockname(listenFd_, (struct sockaddr*)&a, &len) != 0)
      return false;
    memset(&addr_, 0, sizeof(addr_));
    addr_.sin = a;
    thread_ = std::thread([this]() { Run(); });
    return true;
  }
  const union ncclSocketAddress* addr() const { return &addr_; }
  // Joins the relay; the logs are only read after this.
  void Stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    if (listenFd_ >= 0) close(listenFd_);
    listenFd_ = -1;
  }
  std::string clientToServer;
  std::string serverToClient;

private:
  void Run() {
    struct pollfd lp = {listenFd_, POLLIN, 0};
    int cfd = -1;
    while (!stop_ && cfd < 0)
      if (poll(&lp, 1, 50) > 0) cfd = accept(listenFd_, nullptr, nullptr);
    if (cfd < 0) return;
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd >= 0 && connect(sfd, &target_.sa, sizeof(struct sockaddr_in)) == 0) Pump(cfd, sfd);
    close(cfd);
    if (sfd >= 0) close(sfd);
  }
  void Pump(int cfd, int sfd) {
    bool cOpen = true, sOpen = true;
    std::vector<char> buf(1 << 16);
    while (!stop_ && (cOpen || sOpen)) {
      struct pollfd p[2] = {{cfd, (short)(cOpen ? POLLIN : 0), 0}, {sfd, (short)(sOpen ? POLLIN : 0), 0}};
      if (poll(p, 2, 50) <= 0) continue;
      if (cOpen && p[0].revents) cOpen = Forward(cfd, sfd, clientToServer, buf);
      if (sOpen && p[1].revents) sOpen = Forward(sfd, cfd, serverToClient, buf);
    }
  }
  static bool Forward(int from, int to, std::string& log, std::vector<char>& buf) {
    ssize_t r = recv(from, buf.data(), buf.size(), 0);
    if (r <= 0) {
      shutdown(to, SHUT_WR);
      return false;
    }
    log.append(buf.data(), r);
    for (ssize_t off = 0; off < r;) {
      ssize_t w = send(to, buf.data() + off, r - off, MSG_NOSIGNAL);
      if (w <= 0) return false;
      off += w;
    }
    return true;
  }
  union ncclSocketAddress target_;
  union ncclSocketAddress addr_;
  int listenFd_ = -1;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

void RunIsolated(const char* name, std::function<void()> body) {
  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.stopOnFirstFailure = false;
  options.verboseLogging = true;
  RUN_ISOLATED_TESTS_WITH_OPTIONS(
    options, ProcessIsolatedTestRunner::TestConfig(name, body).withTimeout(std::chrono::seconds(90)));
}

// Full-duplex exchange over one pair through ncclSocketMultiOp on a single thread, the
// way socketDoubleSendRecv drives the bidirectional bootstrap ring, relayed through a
// WireTap whose capture must not contain the hello or payload when encrypted.
void CheckDuplexWire(bool encrypted) {
  if (encrypted) ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
  const std::string c2sMarker = "client->server plaintext marker|";
  const std::string s2cMarker = "server->client plaintext marker|";
  const size_t size = 256 << 10;
  WireTap tap;
  {
    Listener l;
    ASSERT_TRUE(l.Start());
    ASSERT_TRUE(tap.Start(l.addr));
    Pair p;
    ASSERT_TRUE(ConnectPair(l, p, tap.addr()));
    EXPECT_EQ(encrypted, p.client.crypto != nullptr);
    EXPECT_EQ(encrypted, p.server.crypto != nullptr);

    std::vector<char> c2s = MarkerPayload(c2sMarker, size), s2c = MarkerPayload(s2cMarker, size);
    std::vector<char> atServer(size), atClient(size);
    struct ncclSocketOp ops[4] = {{NCCL_SOCKET_SEND, &p.client, c2s.data(), (int)size, 0},
                                  {NCCL_SOCKET_RECV, &p.server, atServer.data(), (int)size, 0},
                                  {NCCL_SOCKET_SEND, &p.server, s2c.data(), (int)size, 0},
                                  {NCCL_SOCKET_RECV, &p.client, atClient.data(), (int)size, 0}};
    ASSERT_EQ(ncclSuccess, ncclSocketMultiOp(ops, 4));
    EXPECT_TRUE(atServer == c2s);
    EXPECT_TRUE(atClient == s2c);
  }
  tap.Stop();
  EXPECT_GE(tap.clientToServer.size(), size);
  EXPECT_GE(tap.serverToClient.size(), size);
  bool magicSeen = Contains(tap.clientToServer, &kMagic, sizeof(kMagic));
  bool c2sLeak = Contains(tap.clientToServer, c2sMarker.data(), c2sMarker.size());
  bool s2cLeak = Contains(tap.serverToClient, s2cMarker.data(), s2cMarker.size());
  TEST_INFO("encrypted=%d wire c2s=%zu s2c=%zu bytes magicSeen=%d c2sLeak=%d s2cLeak=%d", encrypted,
            tap.clientToServer.size(), tap.serverToClient.size(), magicSeen, c2sLeak, s2cLeak);
  EXPECT_EQ(!encrypted, magicSeen) << "socket hello (magic) visibility on the wire";
  EXPECT_EQ(!encrypted, c2sLeak) << "client->server payload visibility on the wire";
  EXPECT_EQ(!encrypted, s2cLeak) << "server->client payload visibility on the wire";
}

} // namespace

TEST(SocketCrypt, SetEncryptionValidatesConfig) {
  RunIsolated("SetEncryptionValidatesConfig", []() {
    ncclEncryptionConfig_t cfg = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    cfg.mode = NCCL_ENCRYPTION_MODE_PSK;
    cfg.psk = kPskA;
    cfg.magic = 0;
    EXPECT_EQ(ncclInvalidArgument, ncclSetEncryption(&cfg)) << "config not built from the initializer";

    ncclEncryptionConfig_t noKey = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    noKey.mode = NCCL_ENCRYPTION_MODE_PSK;
    EXPECT_EQ(ncclInvalidArgument, ncclSetEncryption(&noKey)) << "PSK mode without a key";

    std::string key31(31, 'k');
    EXPECT_EQ(ncclInvalidArgument, SetPsk(key31.c_str())) << "31-byte key is below the 32-byte minimum";

    ncclEncryptionConfig_t badMode = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    badMode.mode = 7;
    EXPECT_EQ(ncclInvalidArgument, ncclSetEncryption(&badMode));

    std::string key32(32, 'k');
    EXPECT_EQ(ncclSuccess, SetPsk(key32.c_str()));
    EXPECT_EQ(ncclSuccess, ClearPsk());
    ncclEncryptionConfig_t none = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    EXPECT_EQ(ncclSuccess, ncclSetEncryption(&none));
  });
}

// A rejected call must leave the previously configured key in place, and clearing the
// key must return sockets to plaintext.
TEST(SocketCrypt, ConnectionModeFollowsApi) {
  RunIsolated("ConnectionModeFollowsApi", []() {
    bool encrypted = true;
    ASSERT_EQ(ncclSuccess, ncclGetCryptConnectionMode(&encrypted));
    EXPECT_FALSE(encrypted) << "plaintext is the default";

    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    std::string key31(31, 'k');
    EXPECT_EQ(ncclInvalidArgument, SetPsk(key31.c_str()));
    if (kTlsBuild) {
      ASSERT_EQ(ncclSuccess, ncclGetCryptConnectionMode(&encrypted));
      EXPECT_TRUE(encrypted);
    } else {
      // Without TLS support a configured key is an error at first use, never a silent
      // fallback to plaintext.
      EXPECT_EQ(ncclInvalidUsage, ncclGetCryptConnectionMode(&encrypted));
      EXPECT_FALSE(encrypted);
    }

    ASSERT_EQ(ncclSuccess, ClearPsk());
    ASSERT_EQ(ncclSuccess, ncclGetCryptConnectionMode(&encrypted));
    EXPECT_FALSE(encrypted);
  });
}

// Without TLS support, connecting with a key configured fails instead of sending plaintext.
TEST(SocketCrypt, KeyWithoutTlsSupportFailsConnect) {
  if (kTlsBuild) GTEST_SKIP() << "covers builds without TLS_BACKEND=OPENSSL3";
  RunIsolated("KeyWithoutTlsSupportFailsConnect", []() {
    Listener l;
    ASSERT_TRUE(l.Start());
    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    struct ncclSocket client;
    memset(&client, 0, sizeof(client));
    ASSERT_EQ(ncclSuccess, ncclSocketInit(&client, &l.addr, kMagic, kType));
    EXPECT_NE(ncclSuccess, ncclSocketConnect(&client));
    ncclSocketClose(&client);
  });
}

TEST(SocketCrypt, TlsRoundTripAcrossRecordSizes) {
  SKIP_UNLESS_TLS_BUILD();
  RunIsolated("TlsRoundTripAcrossRecordSizes", []() {
    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    Listener l;
    ASSERT_TRUE(l.Start());
    Pair p;
    ASSERT_TRUE(ConnectPair(l, p));
    EXPECT_NE(nullptr, p.client.crypto);
    EXPECT_NE(nullptr, p.server.crypto);
    // TLS records carry at most 16 KiB, so these cover a partial record, a record
    // boundary plus one byte, and a payload far larger than the socket buffers.
    unsigned seed = 1;
    for (size_t size : {(size_t)1, (size_t)4096, (size_t)16384, (size_t)16385, (size_t)(4 << 20)}) {
      EXPECT_TRUE(Transfer(&p.client, &p.server, size, seed++)) << "client->server " << size << " bytes";
      EXPECT_TRUE(Transfer(&p.server, &p.client, size, seed++)) << "server->client " << size << " bytes";
    }
  });
}

// A client that only writes (bootstrapSend, the unidirectional ring AllGather) closes
// its socket as soon as the last byte is queued. Anything the server sent it and it
// never read (TLS 1.3 session tickets) turns that close into a RST, which drops the
// bytes still in the client's send queue. The server must get the whole payload.
// The payload is sized past what the server's (shrunk) receive buffer holds but within
// the client's send buffer, so the send completes and the close finds bytes still
// queued; the server only starts reading after the close.
TEST(SocketCrypt, WriteOnlyClientCloseKeepsQueuedData) {
  SKIP_UNLESS_TLS_BUILD();
  RunIsolated("WriteOnlyClientCloseKeepsQueuedData", []() {
    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    Listener l;
    ASSERT_TRUE(l.Start());
    int rcvBuf = 32 << 10, sndBuf = 4 << 20;
    socklen_t optLen = sizeof(int);
    ASSERT_EQ(0, setsockopt(l.sock.socketDescriptor, SOL_SOCKET, SO_RCVBUF, &rcvBuf, sizeof(rcvBuf)));
    ASSERT_EQ(0, getsockopt(l.sock.socketDescriptor, SOL_SOCKET, SO_RCVBUF, &rcvBuf, &optLen));
    struct ncclSocket client, server;
    memset(&client, 0, sizeof(client));
    memset(&server, 0, sizeof(server));
    volatile uint32_t clientAbort = 0;
    ASSERT_EQ(ncclSuccess, ncclSocketInit(&client, &l.addr, kMagic, kType, &clientAbort));
    ASSERT_EQ(0, setsockopt(client.socketDescriptor, SOL_SOCKET, SO_SNDBUF, &sndBuf, sizeof(sndBuf)));
    ASSERT_EQ(0, getsockopt(client.socketDescriptor, SOL_SOCKET, SO_SNDBUF, &sndBuf, &optLen));
    const size_t size = rcvBuf + sndBuf / 4;
    TEST_INFO("receive buffer %d, send buffer %d, payload %zu bytes", rcvBuf, sndBuf, size);
    std::vector<char> out = Pattern(size, 3), in(size, 0);

    std::atomic<int> acceptRes{ncclInternalError}, recvRes{ncclInternalError};
    std::atomic<bool> clientClosed{false}, done{false};
    std::thread watchdog([&]() {
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
      while (!done && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (!done) {
        clientAbort = 1;
        l.abortFlag = 1;
      }
    });
    std::thread receiver([&]() {
      if (ncclSocketInit(&server) != ncclSuccess || (acceptRes = ncclSocketAccept(&server, &l.sock)) != ncclSuccess)
        return;
      while (!clientClosed && !l.abortFlag) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      recvRes = ncclSocketRecv(&server, in.data(), (int)size);
    });
    ncclResult_t res = ncclSocketConnect(&client);
    if (res == ncclSuccess) res = ncclSocketSend(&client, out.data(), (int)size);
    ncclSocketClose(&client);
    clientClosed = true;
    receiver.join();
    done = true;
    watchdog.join();
    ASSERT_EQ(ncclSuccess, res) << "client connect/send";
    ASSERT_EQ(ncclSuccess, acceptRes);
    EXPECT_EQ(ncclSuccess, recvRes) << "server lost the tail of the payload after the client closed";
    EXPECT_TRUE(in == out);
    ncclSocketClose(&server);
  });
}

TEST(SocketCrypt, DuplexMultiOpIsEncryptedOnTheWire) {
  SKIP_UNLESS_TLS_BUILD();
  RunIsolated("DuplexMultiOpIsEncryptedOnTheWire", []() { CheckDuplexWire(/*encrypted=*/true); });
}

// Control for the wire check: the same exchange in plaintext must be visible to the tap,
// otherwise the encrypted case above proves nothing.
TEST(SocketCrypt, PlaintextDuplexIsVisibleOnTheWire) {
  RunIsolated("PlaintextDuplexIsVisibleOnTheWire", []() { CheckDuplexWire(/*encrypted=*/false); });
}

// Mirrors socketRingConnectStart/Finish: every rank starts an async connect to its next
// rank and an async accept from its previous one, and a single thread polls all sockets
// with ncclSocketReady. Any TLS handshake step that blocks deadlocks here (caught by the
// isolated-test timeout). The bidirectional exchange then mirrors socketRingAllGather.
TEST(SocketCrypt, SingleThreadRingConnectCompletesTlsHandshakes) {
  SKIP_UNLESS_TLS_BUILD();
  RunIsolated("SingleThreadRingConnectCompletesTlsHandshakes", []() {
    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    constexpr int kRanks = 4;
    Listener listeners[kRanks];
    struct ncclSocket sendSock[kRanks], recvSock[kRanks];
    memset(sendSock, 0, sizeof(sendSock));
    memset(recvSock, 0, sizeof(recvSock));
    volatile uint32_t abortFlag = 0;
    for (int r = 0; r < kRanks; r++) ASSERT_TRUE(listeners[r].Start());
    for (int r = 0; r < kRanks; r++) {
      ASSERT_EQ(ncclSuccess, ncclSocketInit(&recvSock[r]));
      ASSERT_EQ(ncclSuccess, ncclSocketInit(&sendSock[r], &listeners[(r + 1) % kRanks].addr, kMagic, kType,
                                            &abortFlag, /*asyncFlag=*/1));
      ASSERT_EQ(ncclSuccess, ncclSocketConnect(&sendSock[r]));
      int oldAsync = listeners[r].sock.asyncFlag;
      listeners[r].sock.asyncFlag = 1;
      ASSERT_EQ(ncclSuccess, ncclSocketAccept(&recvSock[r], &listeners[r].sock));
      listeners[r].sock.asyncFlag = oldAsync;
    }

    int sendReady[kRanks] = {}, recvReady[kRanks] = {};
    int remaining = 2 * kRanks;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
      remaining = 0;
      for (int r = 0; r < kRanks; r++) {
        if (!sendReady[r]) ASSERT_EQ(ncclSuccess, ncclSocketReady(&sendSock[r], &sendReady[r]));
        if (!recvReady[r]) ASSERT_EQ(ncclSuccess, ncclSocketReady(&recvSock[r], &recvReady[r]));
        remaining += !sendReady[r] + !recvReady[r];
      }
      if (remaining) sched_yield();
    }
    ASSERT_EQ(0, remaining) << "ring sockets not ready before the deadline";
    for (int r = 0; r < kRanks; r++) {
      EXPECT_NE(nullptr, sendSock[r].crypto) << "rank " << r;
      EXPECT_NE(nullptr, recvSock[r].crypto) << "rank " << r;
    }

    const size_t size = 64 << 10;
    std::vector<std::vector<char>> token(kRanks), fromPrev(kRanks), fromNext(kRanks);
    std::vector<struct ncclSocketOp> ops;
    for (int r = 0; r < kRanks; r++) {
      token[r] = Pattern(size, 100 + r);
      fromPrev[r].assign(size, 0);
      fromNext[r].assign(size, 0);
    }
    for (int r = 0; r < kRanks; r++) {
      ops.push_back({NCCL_SOCKET_SEND, &sendSock[r], token[r].data(), (int)size, 0});
      ops.push_back({NCCL_SOCKET_RECV, &recvSock[r], fromPrev[r].data(), (int)size, 0});
      ops.push_back({NCCL_SOCKET_SEND, &recvSock[r], token[r].data(), (int)size, 0});
      ops.push_back({NCCL_SOCKET_RECV, &sendSock[r], fromNext[r].data(), (int)size, 0});
    }
    ASSERT_EQ(ncclSuccess, ncclSocketMultiOp(ops.data(), (int)ops.size()));
    for (int r = 0; r < kRanks; r++) {
      EXPECT_TRUE(fromPrev[r] == token[(r + kRanks - 1) % kRanks]) << "rank " << r << " from prev";
      EXPECT_TRUE(fromNext[r] == token[(r + 1) % kRanks]) << "rank " << r << " from next";
    }
    for (int r = 0; r < kRanks; r++) {
      ncclSocketClose(&sendSock[r]);
      ncclSocketClose(&recvSock[r]);
    }
  });
}

// A plaintext peer reaching a TLS listener is discarded and the accept loop keeps
// serving: the next (encrypted) peer is accepted.
TEST(SocketCrypt, TlsListenerDropsPlaintextPeer) {
  SKIP_UNLESS_TLS_BUILD();
  RunIsolated("TlsListenerDropsPlaintextPeer", []() {
    Listener l;
    ASSERT_TRUE(l.Start());
    struct ncclSocket plain;
    memset(&plain, 0, sizeof(plain));
    ASSERT_EQ(ncclSuccess, ncclSocketInit(&plain, &l.addr, kMagic, kType));
    // The plaintext hello fits in the kernel buffers, so this completes before any accept.
    ASSERT_EQ(ncclSuccess, ncclSocketConnect(&plain));

    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    Pair p;
    ASSERT_TRUE(ConnectPair(l, p)) << "the encrypted peer queued behind the plaintext one was not accepted";
    EXPECT_NE(nullptr, p.server.crypto);
    EXPECT_TRUE(Transfer(&p.client, &p.server, 1024, 7));
    EXPECT_TRUE(PeerDropped(&plain, 5000)) << "plaintext peer was not disconnected";
    ncclSocketClose(&plain);
  });
}

// An encrypted peer reaching a plaintext listener sees its handshake fail.
TEST(SocketCrypt, PlaintextListenerDropsTlsPeer) {
  SKIP_UNLESS_TLS_BUILD();
  RunIsolated("PlaintextListenerDropsTlsPeer", []() {
    Listener l;
    ASSERT_TRUE(l.Start());
    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    struct ncclSocket tls;
    memset(&tls, 0, sizeof(tls));
    volatile uint32_t clientAbort = 0;
    ASSERT_EQ(ncclSuccess, ncclSocketInit(&tls, &l.addr, kMagic, kType, &clientAbort, /*asyncFlag=*/1));
    ASSERT_EQ(ncclSuccess, ncclSocketConnect(&tls));
    ASSERT_EQ(ncclSuccess, ClearPsk());

    struct ncclSocket server;
    memset(&server, 0, sizeof(server));
    std::atomic<int> acceptRes{ncclSuccess};
    std::thread acceptor([&]() {
      if (ncclSocketInit(&server) == ncclSuccess) acceptRes = ncclSocketAccept(&server, &l.sock);
    });
    ncclResult_t res = ncclSuccess;
    int ready = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (res == ncclSuccess && !ready && std::chrono::steady_clock::now() < deadline) {
      res = ncclSocketReady(&tls, &ready);
      if (res == ncclSuccess && !ready) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    l.abortFlag = 1;
    acceptor.join();
    EXPECT_FALSE(ready) << "TLS client became ready against a plaintext listener";
    EXPECT_NE(ncclSuccess, res) << "TLS client handshake did not fail";
    EXPECT_NE(ncclSocketStateReady, server.state) << "plaintext listener accepted the TLS peer";
    ncclSocketClose(&server);
    ncclSocketClose(&tls);
  });
}

// A peer holding a different key cannot connect, and the listener keeps working for a
// peer with the right key. The wrong-key peer runs in a forked process because the key
// is process-global.
TEST(SocketCrypt, WrongKeyPeerIsRejected) {
  SKIP_UNLESS_TLS_BUILD();
  RunIsolated("WrongKeyPeerIsRejected", []() {
    ASSERT_EQ(ncclSuccess, SetPsk(kPskA));
    Listener l;
    ASSERT_TRUE(l.Start());
    pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      if (SetPsk(kPskB) != ncclSuccess) _exit(2);
      struct ncclSocket c;
      memset(&c, 0, sizeof(c));
      if (ncclSocketInit(&c, &l.addr, kMagic, kType) != ncclSuccess) _exit(3);
      ncclResult_t r = ncclSocketConnect(&c);
      _exit(r == ncclSuccess ? 1 : 0);
    }

    Pair p;
    std::atomic<int> acceptRes{ncclInternalError};
    std::thread acceptor([&]() {
      if (ncclSocketInit(&p.server) == ncclSuccess) acceptRes = ncclSocketAccept(&p.server, &l.sock);
    });
    int status = -1;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (waitpid(child, &status, WNOHANG) == 0 && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!WIFEXITED(status)) {
      kill(child, SIGKILL);
      waitpid(child, &status, 0);
    }
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
      << "wrong-key peer connected or hung (status " << status << ")";

    ncclResult_t res = ncclSocketInit(&p.client, &l.addr, kMagic, kType);
    if (res == ncclSuccess) res = ncclSocketConnect(&p.client);
    if (res != ncclSuccess) l.abortFlag = 1;
    acceptor.join();
    ASSERT_EQ(ncclSuccess, res) << "right-key peer could not connect after the rejected one";
    ASSERT_EQ(ncclSuccess, acceptRes);
    EXPECT_TRUE(Transfer(&p.client, &p.server, 1024, 9));
  });
}

} // namespace RcclUnitTesting
