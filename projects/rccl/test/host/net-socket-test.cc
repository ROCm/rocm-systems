/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_socket.cc and src/transport/net_socket/common.cc.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

// ras-test.cc already defines several ncclSocket* names in this binary; keep these doubles TU-local.
#define ncclSocketDefaultMagic NetSocketTestSocketDefaultMagic
#define ncclSocketInit NetSocketTestSocketInit
#define ncclSocketMove NetSocketTestSocketMove
#define ncclSocketListen NetSocketTestSocketListen
#define ncclSocketGetAddr NetSocketTestSocketGetAddr
#define ncclSocketConnect NetSocketTestSocketConnect
#define ncclSocketReady NetSocketTestSocketReady
#define ncclSocketAccept NetSocketTestSocketAccept
#define ncclSocketProgress NetSocketTestSocketProgress
#define ncclSocketClose NetSocketTestSocketClose
#define ncclSocketToString NetSocketTestSocketToString
#define ncclFindInterfaces NetSocketTestFindInterfaces
#define ncclOsRealpath NetSocketTestOsRealpath

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/nccl_fakes.h"

#include "comm.h"
#include "core.h"
#include "net.h"
#include "os.h"
#include "param.h"
#include "profiler/net_socket.h"
#include "socket.h"

#include "fakes/param_redirect.h"

namespace {

constexpr uint64_t kMagic = 0x5eedULL;
constexpr uint16_t kListenPort = 40000;

ncclSocketAddress DevAddr(int i) {
  ncclSocketAddress addr{};
  addr.sin.sin_family = AF_INET;
  addr.sin.sin_addr.s_addr = htonl(0x0a000001 + i);
  return addr;
}

// Byte streams keyed by socket descriptor: what the unit sent, and what a peer has queued for it.
struct FakeSockets {
  std::mutex mu;
  std::map<int, std::string> sent;
  std::map<int, std::string> inbound;
  std::vector<int> closed;
  std::deque<uint8_t> acceptIndices;
  int nextFd = 100;
  size_t chunk = SIZE_MAX;
};
FakeSockets g_sock;

ncclResult_t DefaultSockInit(ncclSocket* sock, const ncclSocketAddress* addr, uint64_t magic, ncclSocketType type,
                             volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  std::memset(sock, 0, sizeof(*sock));
  if (addr) sock->addr = *addr;
  sock->magic = magic;
  sock->type = type;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->customRetry = customRetry;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateInitialized;
  return ncclSuccess;
}

ncclResult_t DefaultSockListen(ncclSocket* sock) {
  sock->socketDescriptor = g_sock.nextFd++;
  sock->addr.sin.sin_port = htons(kListenPort);
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

ncclResult_t DefaultSockGetAddr(ncclSocket* sock, ncclSocketAddress* addr) {
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  *addr = sock->addr;
  return ncclSuccess;
}

ncclResult_t DefaultSockConnect(ncclSocket* sock) {
  sock->socketDescriptor = g_sock.nextFd++;
  sock->state = ncclSocketStateConnecting;
  return ncclSuccess;
}

ncclResult_t DefaultSockAccept(ncclSocket* sock, ncclSocket*, bool) {
  std::lock_guard<std::mutex> lock(g_sock.mu);
  sock->socketDescriptor = g_sock.nextFd++;
  sock->state = ncclSocketStateAccepting;
  if (!g_sock.acceptIndices.empty()) {
    g_sock.inbound[sock->socketDescriptor] = std::string(1, static_cast<char>(g_sock.acceptIndices.front()));
    g_sock.acceptIndices.pop_front();
  }
  return ncclSuccess;
}

ncclResult_t DefaultSockReady(ncclSocket* sock, int* running) {
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) return ncclRemoteError;
  if (sock->state == ncclSocketStateConnecting || sock->state == ncclSocketStateAccepting) {
    sock->state = ncclSocketStateReady;
  }
  *running = sock->state == ncclSocketStateReady;
  return ncclSuccess;
}

ncclResult_t DefaultSockProgress(int op, ncclSocket* sock, void* ptr, int size, int* offset, bool*) {
  std::lock_guard<std::mutex> lock(g_sock.mu);
  size_t n = std::min(static_cast<size_t>(size - *offset), g_sock.chunk);
  char* at = static_cast<char*>(ptr) + *offset;
  if (op == NCCL_SOCKET_SEND) {
    g_sock.sent[sock->socketDescriptor].append(at, n);
  } else {
    std::string& in = g_sock.inbound[sock->socketDescriptor];
    n = std::min(n, in.size());
    std::memcpy(at, in.data(), n);
    in.erase(0, n);
  }
  *offset += static_cast<int>(n);
  return ncclSuccess;
}

ncclResult_t DefaultSockClose(ncclSocket* sock, bool) {
  std::lock_guard<std::mutex> lock(g_sock.mu);
  if (sock->state > ncclSocketStateNone && sock->socketDescriptor != NCCL_INVALID_SOCKET) {
    g_sock.closed.push_back(sock->socketDescriptor);
  }
  sock->state = ncclSocketStateClosed;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  return ncclSuccess;
}

ncclResult_t DefaultFindInterfaces(char*, ncclSocketAddress*, int, int, int*) { return ncclInternalError; }

char* DefaultOsRealpath(const char*, char*) { return nullptr; }

std::function<ncclResult_t(ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType, volatile uint32_t*, int,
                           int)>
    g_sockInit = DefaultSockInit;
std::function<ncclResult_t(ncclSocket*)> g_sockListen = DefaultSockListen;
std::function<ncclResult_t(ncclSocket*, ncclSocketAddress*)> g_sockGetAddr = DefaultSockGetAddr;
std::function<ncclResult_t(ncclSocket*)> g_sockConnect = DefaultSockConnect;
std::function<ncclResult_t(ncclSocket*, ncclSocket*, bool)> g_sockAccept = DefaultSockAccept;
std::function<ncclResult_t(ncclSocket*, int*)> g_sockReady = DefaultSockReady;
std::function<ncclResult_t(int, ncclSocket*, void*, int, int*, bool*)> g_sockProgress = DefaultSockProgress;
std::function<ncclResult_t(ncclSocket*, bool)> g_sockClose = DefaultSockClose;
std::function<ncclResult_t(char*, ncclSocketAddress*, int, int, int*)> g_findInterfaces = DefaultFindInterfaces;
std::function<char*(const char*, char*)> g_osRealpath = DefaultOsRealpath;

void ResetSocketFakes() {
  g_sockInit = DefaultSockInit;
  g_sockListen = DefaultSockListen;
  g_sockGetAddr = DefaultSockGetAddr;
  g_sockConnect = DefaultSockConnect;
  g_sockAccept = DefaultSockAccept;
  g_sockReady = DefaultSockReady;
  g_sockProgress = DefaultSockProgress;
  g_sockClose = DefaultSockClose;
  g_findInterfaces = DefaultFindInterfaces;
  g_osRealpath = DefaultOsRealpath;
  g_sock.sent.clear();
  g_sock.inbound.clear();
  g_sock.closed.clear();
  g_sock.acceptIndices.clear();
  g_sock.nextFd = 100;
  g_sock.chunk = SIZE_MAX;
}

ncclResult_t ProfilerA(void**, int, void*, int64_t, void*) { return ncclSuccess; }
ncclResult_t ProfilerB(void**, int, void*, int64_t, void*) { return ncclSuccess; }
}  // namespace

uint64_t NetSocketTestSocketDefaultMagic() { return kMagic; }
ncclResult_t NetSocketTestSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                                     enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag,
                                     int customRetry) {
  return g_sockInit(sock, addr, magic, type, abortFlag, asyncFlag, customRetry);
}
void NetSocketTestSocketMove(struct ncclSocket* dst, struct ncclSocket* src) {
  std::memcpy(dst, src, sizeof(*dst));
  src->socketDescriptor = NCCL_INVALID_SOCKET;
  src->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  src->state = ncclSocketStateNone;
}
ncclResult_t NetSocketTestSocketListen(struct ncclSocket* sock) { return g_sockListen(sock); }
ncclResult_t NetSocketTestSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  return g_sockGetAddr(sock, addr);
}
ncclResult_t NetSocketTestSocketConnect(struct ncclSocket* sock) { return g_sockConnect(sock); }
ncclResult_t NetSocketTestSocketReady(struct ncclSocket* sock, int* running) { return g_sockReady(sock, running); }
ncclResult_t NetSocketTestSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  return g_sockAccept(sock, listenSock, retry);
}
ncclResult_t NetSocketTestSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset,
                                         bool* closed) {
  return g_sockProgress(op, sock, ptr, size, offset, closed);
}
ncclResult_t NetSocketTestSocketClose(struct ncclSocket* sock, bool wait) { return g_sockClose(sock, wait); }
const char* NetSocketTestSocketToString(const union ncclSocketAddress*, char* buf, size_t bufLen, const int) {
  std::snprintf(buf, bufLen, "peer");
  return buf;
}
ncclResult_t NetSocketTestFindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs, int ifNameMaxSize,
                                         int maxIfs, int* nIfs) {
  return g_findInterfaces(ifNames, ifAddrs, ifNameMaxSize, maxIfs, nIfs);
}
char* NetSocketTestOsRealpath(const char* path, char* resolvedPath) { return g_osRealpath(path, resolvedPath); }

#include "fakes/libc_seam.h"
#include NET_SOCKET_COMMON_CC_PATH
#include NET_SOCKET_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

using namespace std::chrono_literals;

std::string SizeFrame(int size) {
  std::string frame(sizeof(int), '\0');
  std::memcpy(&frame[0], &size, sizeof(int));
  return frame;
}

std::string Pattern(size_t n) {
  std::string bytes(n, '\0');
  for (size_t i = 0; i < n; i++) bytes[i] = static_cast<char>(i * 31 + 7);
  return bytes;
}

class NetSocketMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetSocketFakes();
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
  }

  void TearDown() override {
    for (ncclNetSocketComm* comm : comms_) EXPECT_EQ(ncclNetSocketClose(comm), ncclSuccess);
    comms_.clear();
    for (ncclNetSocketDev& dev : ncclNetSocketDevs) free(dev.pciPath);
    std::memset(ncclNetSocketDevs, 0, sizeof(ncclNetSocketDevs));
    ncclNetIfs = -1;
    netRefCount = 0;
    ncclProfilerFunction = nullptr;
    ResetSocketFakes();
    ResetLibcFakes();
    ResetNcclFakes();
    ResetHipFakes();
  }

  void SetParam(const char* env, int64_t value) { params_[env] = value; }

  void SetThreads(int64_t nThreads, int64_t nSocksPerThread) {
    SetParam("SOCKET_NTHREADS", nThreads);
    SetParam("NSOCKS_PERTHREAD", nSocksPerThread);
  }

  void SetDevices(int n) {
    ncclNetIfs = n;
    for (int i = 0; i < n; i++) {
      std::snprintf(ncclNetSocketDevs[i].devName, MAX_IF_NAME_SIZE, "eth%d", i);
      ncclNetSocketDevs[i].addr = DevAddr(i);
    }
  }

  void InstallInterfaces(std::vector<std::string> names) {
    g_findInterfaces = [names](char* ifNames, ncclSocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs, int* nIfs) {
      EXPECT_EQ(ifNameMaxSize, MAX_IF_NAME_SIZE);
      EXPECT_EQ(maxIfs, MAX_IFS);
      for (size_t i = 0; i < names.size(); i++) {
        std::strncpy(ifNames + i * ifNameMaxSize, names[i].c_str(), ifNameMaxSize);
        ifAddrs[i] = DevAddr(static_cast<int>(i));
      }
      *nIfs = static_cast<int>(names.size());
      return ncclSuccess;
    };
  }

  static void ReadySocket(ncclSocket* sock) {
    sock->socketDescriptor = g_sock.nextFd++;
    sock->state = ncclSocketStateReady;
  }

  ncclNetSocketComm* MakeComm(int nSocks, int nThreads) {
    auto* comm = new ncclNetSocketComm();
    comm->nSocks = nSocks;
    comm->nThreads = nThreads;
    ReadySocket(&comm->ctrlSock);
    for (int i = 0; i < nSocks; i++) ReadySocket(&comm->socks[i]);
    comm->inlineData = calloc(MAX_REQUESTS, SOCKET_CTRL_SIZE + ncclParamSocketInlineSize());
    comms_.push_back(comm);
    return comm;
  }

  void Track(void* comm) { comms_.push_back(static_cast<ncclNetSocketComm*>(comm)); }

  ncclResult_t CloseComm(ncclNetSocketComm* comm) {
    comms_.erase(std::find(comms_.begin(), comms_.end(), comm));
    return ncclNetSocketClose(comm);
  }

  static void* Isend(ncclNetSocketComm* comm, void* data, size_t size) {
    void* request = nullptr;
    EXPECT_EQ(ncclNetSocketIsend(comm, data, size, 0, nullptr, nullptr, &request), ncclSuccess);
    return request;
  }

  static void* Irecv(ncclNetSocketComm* comm, void* data, size_t size) {
    void* request = nullptr;
    int tag = 0;
    EXPECT_EQ(ncclNetSocketIrecv(comm, 1, &data, &size, &tag, nullptr, nullptr, &request), ncclSuccess);
    return request;
  }

  static ncclResult_t TestUntilDone(void* request, int* size) {
    auto deadline = std::chrono::steady_clock::now() + 10s;
    int done = 0;
    while (true) {
      ncclResult_t ret = ncclNetSocketTest(request, &done, size);
      if (ret != ncclSuccess || done) return ret;
      if (std::chrono::steady_clock::now() > deadline) {
        ADD_FAILURE() << "request did not complete";
        return ncclInternalError;
      }
      std::this_thread::yield();
    }
  }

  static std::string Sent(const ncclSocket& sock) {
    std::lock_guard<std::mutex> lock(g_sock.mu);
    return g_sock.sent[sock.socketDescriptor];
  }

  static void QueueInbound(int fd, const std::string& bytes) {
    std::lock_guard<std::mutex> lock(g_sock.mu);
    g_sock.inbound[fd] += bytes;
  }

  static void QueueInbound(const ncclSocket& sock, const std::string& bytes) {
    QueueInbound(sock.socketDescriptor, bytes);
  }

  static std::string Inbound(const ncclSocket& sock) {
    std::lock_guard<std::mutex> lock(g_sock.mu);
    return g_sock.inbound[sock.socketDescriptor];
  }

  std::map<std::string, int64_t> params_;
  std::vector<ncclNetSocketComm*> comms_;
};

class NetSocketDevicesMicrotest : public NetSocketMicrotest {
 protected:
  int speed_ = 12345;

  ncclResult_t GetSpeed(const char* devName) {
    char name[MAX_IF_NAME_SIZE];
    std::snprintf(name, sizeof(name), "%s", devName);
    return ncclNetSocketGetSpeed(name, &speed_);
  }
};

TEST_F(NetSocketDevicesMicrotest, InitDevices_InterfacesFound_RecordsNameAddressAndPciPath) {
  InstallInterfaces({"eth0", "ib1"});
  std::vector<std::string> asked;
  ScopedHook realpath(g_osRealpath, [&](const char* path, char*) -> char* {
    asked.push_back(path);
    return std::string(path) == "/sys/class/net/eth0/device" ? strdup("/sys/devices/pci0000:00/0000:00:01.0")
                                                             : nullptr;
  });

  ASSERT_EQ(ncclNetSocketInitDevices("test"), ncclSuccess);

  EXPECT_EQ(ncclNetIfs, 2);
  EXPECT_STREQ(ncclNetSocketDevs[0].devName, "eth0");
  EXPECT_STREQ(ncclNetSocketDevs[1].devName, "ib1");
  EXPECT_EQ(ncclNetSocketDevs[0].addr.sin.sin_addr.s_addr, DevAddr(0).sin.sin_addr.s_addr);
  EXPECT_EQ(ncclNetSocketDevs[1].addr.sin.sin_addr.s_addr, DevAddr(1).sin.sin_addr.s_addr);
  EXPECT_STREQ(ncclNetSocketDevs[0].pciPath, "/sys/devices/pci0000:00/0000:00:01.0");
  EXPECT_EQ(ncclNetSocketDevs[1].pciPath, nullptr);
  EXPECT_EQ(asked, (std::vector<std::string>{"/sys/class/net/eth0/device", "/sys/class/net/ib1/device"}));
}

TEST_F(NetSocketDevicesMicrotest, InitDevices_AlreadyInitialized_SkipsDiscovery) {
  ncclNetIfs = 1;
  ScopedHook find(g_findInterfaces, [](char*, ncclSocketAddress*, int, int, int*) { return ncclInternalError; });

  EXPECT_EQ(ncclNetSocketInitDevices("test"), ncclSuccess);

  EXPECT_EQ(find.calls, 0);
  EXPECT_EQ(ncclNetIfs, 1);
}

TEST_F(NetSocketDevicesMicrotest, InitDevices_DiscoveryFails_ReturnsErrorAndStaysUninitialized) {
  ScopedHook find(g_findInterfaces, [](char*, ncclSocketAddress*, int, int, int* nIfs) {
    *nIfs = 3;
    return ncclSystemError;
  });

  EXPECT_EQ(ncclNetSocketInitDevices("test"), ncclSystemError);

  EXPECT_EQ(ncclNetIfs, -1);
}

TEST_F(NetSocketDevicesMicrotest, InitDevices_NoInterfaces_ReturnsInternalErrorAndStaysUninitialized) {
  InstallInterfaces({});

  EXPECT_EQ(ncclNetSocketInitDevices("test"), ncclInternalError);

  EXPECT_EQ(ncclNetIfs, -1);
}

TEST_F(NetSocketDevicesMicrotest, GetSpeed_SysfsReportsSpeed_ReturnsItAndClosesFile) {
  std::string opened;
  ScopedHook openHook(g_open, [&](const char* path, int flags, mode_t) {
    opened = path;
    EXPECT_EQ(flags, O_RDONLY);
    return 7;
  });
  ScriptReadData("25000\n");

  EXPECT_EQ(GetSpeed("eth0"), ncclSuccess);

  EXPECT_EQ(speed_, 25000);
  EXPECT_EQ(opened, "/sys/class/net/eth0/speed");
  EXPECT_EQ(g_readFds, std::vector<int>{7});
  EXPECT_EQ(g_closedFds, std::vector<int>{7});
}

TEST_F(NetSocketDevicesMicrotest, GetSpeed_NoSpeedFile_Defaults10Gbps) {
  EXPECT_EQ(GetSpeed("eth0"), ncclSuccess);

  EXPECT_EQ(speed_, 10000);
  EXPECT_TRUE(g_closedFds.empty());
}

TEST_F(NetSocketDevicesMicrotest, GetSpeed_UnusableSpeed_Defaults10Gbps) {
  ScopedHook openHook(g_open, [](const char*, int, mode_t) { return 7; });
  std::vector<MicroReadStep> reads = {{0, 0, ""}, {-1, EIO, ""}, {2, 0, "0\n"}, {3, 0, "-1\n"}, {4, 0, "down"}};
  for (size_t i = 0; i < reads.size(); i++) {
    SCOPED_TRACE(i);
    g_readScript = {reads[i]};
    g_readScriptPos = 0;
    g_closedFds.clear();

    EXPECT_EQ(GetSpeed("eth0"), ncclSuccess);

    EXPECT_EQ(speed_, 10000);
    EXPECT_EQ(g_closedFds, std::vector<int>{7});
  }
}

TEST_F(NetSocketDevicesMicrotest, GetSpeed_LongSpeedString_ReadsAtMostEightCharacters) {
  ScopedHook openHook(g_open, [](const char*, int, mode_t) { return 7; });
  size_t asked = 0;
  ScopedHook readHook(g_read, [&](int, void* buf, size_t count) {
    asked = count;
    return DeliverReadStep({9, 0, "123456789"}, buf, count);
  });

  EXPECT_EQ(GetSpeed("eth0"), ncclSuccess);

  EXPECT_EQ(asked, 8u);
  EXPECT_EQ(speed_, 12345678);
}

TEST_F(NetSocketDevicesMicrotest, GetSpeed_OpenInterrupted_RetriesOpen) {
  int calls = 0;
  ScopedHook openHook(g_open, [&](const char*, int, mode_t) {
    if (calls++ == 0) {
      errno = EINTR;
      return -1;
    }
    return 7;
  });
  ScriptReadData("40000");

  EXPECT_EQ(GetSpeed("eth0"), ncclSuccess);

  EXPECT_EQ(openHook.calls, 2);
  EXPECT_EQ(speed_, 40000);
}

TEST_F(NetSocketDevicesMicrotest, GetSpeed_CloseFails_ReturnsSystemError) {
  ScopedHook openHook(g_open, [](const char*, int, mode_t) { return 7; });
  ScriptReadData("100");
  ScopedHook closeHook(g_close, [](int) {
    errno = EIO;
    return -1;
  });

  EXPECT_EQ(GetSpeed("eth0"), ncclSystemError);
}

TEST_F(NetSocketDevicesMicrotest, CreateListener_DeviceOutOfRange_ReturnsInternalErrorWithoutSocket) {
  SetDevices(2);
  ScopedHook init(g_sockInit, DefaultSockInit);
  for (int dev : {-1, 2}) {
    SCOPED_TRACE(dev);
    ncclSocket sock{};
    ncclSocketAddress addr{};
    uint64_t magic = 0;

    EXPECT_EQ(ncclNetSocketCreateListener("test", dev, &sock, &addr, &magic), ncclInternalError);
  }
  EXPECT_EQ(init.calls, 0);
}

TEST_F(NetSocketDevicesMicrotest, CreateListener_ValidDevice_ListensAndPublishesBoundAddressAndMagic) {
  SetDevices(2);
  ScopedHook listenHook(g_sockListen, DefaultSockListen);
  ncclSocket sock{};
  ncclSocketAddress connectAddr{};
  uint64_t magic = 0;

  ASSERT_EQ(ncclNetSocketCreateListener("test", 1, &sock, &connectAddr, &magic), ncclSuccess);

  EXPECT_EQ(magic, kMagic);
  EXPECT_EQ(sock.magic, kMagic);
  EXPECT_EQ(sock.type, ncclSocketTypeNetSocket);
  EXPECT_EQ(sock.abortFlag, nullptr);
  EXPECT_EQ(sock.asyncFlag, 1);
  EXPECT_EQ(sock.addr.sin.sin_addr.s_addr, DevAddr(1).sin.sin_addr.s_addr);
  EXPECT_EQ(listenHook.calls, 1);
  EXPECT_EQ(connectAddr.sin.sin_addr.s_addr, DevAddr(1).sin.sin_addr.s_addr);
  EXPECT_EQ(connectAddr.sin.sin_port, htons(kListenPort));
}

TEST_F(NetSocketDevicesMicrotest, CreateListener_SocketStepFails_PropagatesError) {
  SetDevices(1);
  ncclSocket sock{};
  ncclSocketAddress addr{};
  uint64_t magic = 0;
  {
    ScopedHook init(g_sockInit, [](ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType,
                                   volatile uint32_t*, int, int) { return ncclSystemError; });
    EXPECT_EQ(ncclNetSocketCreateListener("test", 0, &sock, &addr, &magic), ncclSystemError);
  }
  {
    ScopedHook listenHook(g_sockListen, [](ncclSocket*) { return ncclRemoteError; });
    EXPECT_EQ(ncclNetSocketCreateListener("test", 0, &sock, &addr, &magic), ncclRemoteError);
  }
  {
    ScopedHook getAddr(g_sockGetAddr, [](ncclSocket*, ncclSocketAddress*) { return ncclInvalidArgument; });
    EXPECT_EQ(ncclNetSocketCreateListener("test", 0, &sock, &addr, &magic), ncclInvalidArgument);
  }
}

using NetSocketInitMicrotest = NetSocketMicrotest;

TEST_F(NetSocketInitMicrotest, Init_FirstCall_DiscoversDevicesAndStoresProfilerCallback) {
  InstallInterfaces({"eth0", "eth1"});
  void* ctx = nullptr;

  ASSERT_EQ(ncclNetSocketInit(&ctx, 0, nullptr, nullptr, ProfilerA), ncclSuccess);

  int ndev = 0;
  EXPECT_EQ(ncclNetSocketDevices(&ndev), ncclSuccess);
  EXPECT_EQ(ndev, 2);
  EXPECT_EQ(ncclProfilerFunction, ProfilerA);
}

TEST_F(NetSocketInitMicrotest, Init_AlreadyInitialized_KeepsFirstProfilerCallback) {
  InstallInterfaces({"eth0"});
  void* ctx = nullptr;
  ASSERT_EQ(ncclNetSocketInit(&ctx, 0, nullptr, nullptr, ProfilerA), ncclSuccess);

  EXPECT_EQ(ncclNetSocketInit(&ctx, 1, nullptr, nullptr, ProfilerB), ncclSuccess);

  EXPECT_EQ(ncclProfilerFunction, ProfilerA);
}

TEST_F(NetSocketInitMicrotest, Init_DiscoveryFails_ReturnsErrorWithoutTakingReference) {
  InstallInterfaces({});
  void* ctx = nullptr;

  EXPECT_EQ(ncclNetSocketInit(&ctx, 0, nullptr, nullptr, ProfilerA), ncclInternalError);

  EXPECT_EQ(ncclNetSocketFinalize(ctx), ncclInternalError);
}

TEST_F(NetSocketInitMicrotest, Finalize_WithoutInit_ReturnsInternalError) {
  EXPECT_EQ(ncclNetSocketFinalize(nullptr), ncclInternalError);
}

TEST_F(NetSocketInitMicrotest, Finalize_AfterTwoInits_SucceedsTwiceThenFails) {
  InstallInterfaces({"eth0"});
  void* ctx = nullptr;
  ASSERT_EQ(ncclNetSocketInit(&ctx, 0, nullptr, nullptr, ProfilerA), ncclSuccess);
  ASSERT_EQ(ncclNetSocketInit(&ctx, 1, nullptr, nullptr, ProfilerA), ncclSuccess);

  EXPECT_EQ(ncclNetSocketFinalize(ctx), ncclSuccess);
  EXPECT_EQ(ncclNetSocketFinalize(ctx), ncclSuccess);
  EXPECT_EQ(ncclNetSocketFinalize(ctx), ncclInternalError);
}

TEST_F(NetSocketInitMicrotest, Init_AfterLastFinalize_TakesNewProfilerCallback) {
  InstallInterfaces({"eth0"});
  void* ctx = nullptr;
  ASSERT_EQ(ncclNetSocketInit(&ctx, 0, nullptr, nullptr, ProfilerA), ncclSuccess);
  ASSERT_EQ(ncclNetSocketFinalize(ctx), ncclSuccess);

  EXPECT_EQ(ncclNetSocketInit(&ctx, 0, nullptr, nullptr, ProfilerB), ncclSuccess);

  EXPECT_EQ(ncclProfilerFunction, ProfilerB);
}

TEST_F(NetSocketInitMicrotest, GetProperties_ValidDevice_ReportsIdentitySpeedAndHostOnlyLimits) {
  SetDevices(2);
  ncclNetSocketDevs[1].pciPath = strdup("/sys/devices/pci0000:00/0000:01:00.0");
  ScopedHook openHook(g_open, [](const char* path, int, mode_t) {
    if (std::string(path) == "/sys/class/net/eth1/speed") return 9;
    errno = ENOENT;
    return -1;
  });
  ScriptReadData("100000");
  ncclNetProperties_t props;
  std::memset(&props, 0xa5, sizeof(props));

  ASSERT_EQ(ncclNetSocketGetProperties(1, &props), ncclSuccess);

  EXPECT_STREQ(props.name, "eth1");
  EXPECT_STREQ(props.pciPath, "/sys/devices/pci0000:00/0000:01:00.0");
  EXPECT_EQ(props.guid, 1u);
  EXPECT_EQ(props.ptrSupport, NCCL_PTR_HOST);
  EXPECT_EQ(props.regIsGlobal, 0);
  EXPECT_EQ(props.forceFlush, 0);
  EXPECT_EQ(props.speed, 100000);
  EXPECT_EQ(props.latency, 0);
  EXPECT_EQ(props.port, 0);
  EXPECT_EQ(props.maxComms, 65536);
  EXPECT_EQ(props.maxRecvs, 1);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_HOST);
  EXPECT_EQ(props.netDeviceVersion, NCCL_NET_DEVICE_INVALID_VERSION);
  EXPECT_EQ(props.maxP2pBytes, NCCL_MAX_NET_SIZE_BYTES);
  EXPECT_EQ(props.maxCollBytes, static_cast<size_t>(MAX_COLLNET_SIZE));
  EXPECT_EQ(props.maxMultiRequestSize, 1);
  EXPECT_EQ(props.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(props.planeId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(props.vProps.ndevs, 1);
  EXPECT_EQ(props.vProps.devs[0], 1);
}

TEST_F(NetSocketInitMicrotest, RegMr_HostMemory_Succeeds) {
  void* mhandle = nullptr;
  EXPECT_EQ(ncclNetSocketRegMr(nullptr, nullptr, 0, NCCL_PTR_HOST, &mhandle), ncclSuccess);
}

TEST_F(NetSocketInitMicrotest, DeregMr_AnyHandle_Succeeds) {
  EXPECT_EQ(ncclNetSocketDeregMr(nullptr, nullptr), ncclSuccess);
}

TEST_F(NetSocketInitMicrotest, RegMr_DeviceMemory_ReturnsInternalError) {
  void* mhandle = nullptr;
  EXPECT_EQ(ncclNetSocketRegMr(nullptr, nullptr, 0, NCCL_PTR_CUDA, &mhandle), ncclInternalError);
}

TEST_F(NetSocketInitMicrotest, Iflush_AnyCall_ReturnsInternalError) {
  void* request = nullptr;
  EXPECT_EQ(ncclNetSocketIflush(nullptr, 1, nullptr, nullptr, nullptr, &request), ncclInternalError);
}

TEST_F(NetSocketInitMicrotest, Plugin_ExposesSocketEntryPoints) {
  EXPECT_STREQ(ncclNetSocket.name, "Socket");
  EXPECT_EQ(ncclNetSocket.init, ncclNetSocketInit);
  EXPECT_EQ(ncclNetSocket.devices, ncclNetSocketDevices);
  EXPECT_EQ(ncclNetSocket.getProperties, ncclNetSocketGetProperties);
  EXPECT_EQ(ncclNetSocket.listen, ncclNetSocketListen);
  EXPECT_EQ(ncclNetSocket.connect, ncclNetSocketConnect);
  EXPECT_EQ(ncclNetSocket.accept, ncclNetSocketAccept);
  EXPECT_EQ(ncclNetSocket.regMr, ncclNetSocketRegMr);
  EXPECT_EQ(ncclNetSocket.regMrDmaBuf, nullptr);
  EXPECT_EQ(ncclNetSocket.deregMr, ncclNetSocketDeregMr);
  EXPECT_EQ(ncclNetSocket.isend, ncclNetSocketIsend);
  EXPECT_EQ(ncclNetSocket.irecv, ncclNetSocketIrecv);
  EXPECT_EQ(ncclNetSocket.iflush, ncclNetSocketIflush);
  EXPECT_EQ(ncclNetSocket.test, ncclNetSocketTest);
  EXPECT_EQ(ncclNetSocket.closeSend, ncclNetSocketClose);
  EXPECT_EQ(ncclNetSocket.closeRecv, ncclNetSocketClose);
  EXPECT_EQ(ncclNetSocket.closeListen, ncclNetSocketCloseListen);
  EXPECT_EQ(ncclNetSocket.getDeviceMr, nullptr);
  EXPECT_EQ(ncclNetSocket.irecvConsumed, nullptr);
  EXPECT_EQ(ncclNetSocket.makeVDevice, nullptr);
  EXPECT_EQ(ncclNetSocket.finalize, ncclNetSocketFinalize);
  EXPECT_EQ(ncclNetSocket.setNetAttr, nullptr);
}

class NetSocketThreadsMicrotest : public NetSocketMicrotest {
 protected:
  static constexpr int kVendorFd = 11;
  std::string realpathAsked_;
  std::string opened_;
  int ns_ = -1;
  int nt_ = -1;

  void SetUp() override {
    NetSocketMicrotest::SetUp();
    SetDevices(1);
  }

  void InstallVendorFile() {
    g_osRealpath = [this](const char* path, char*) {
      realpathAsked_ = path;
      return strdup("/sys/devices/pci0000:00/0000:00:03.0/vendor");
    };
    g_open = [this](const char* path, int, mode_t) {
      opened_ = path;
      return kVendorFd;
    };
  }

  ncclResult_t GetCounts() { return ncclNetSocketGetNsockNthread(0, &ns_, &nt_); }
};

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_ExplicitCounts_UsesThemWithoutProbingVendor) {
  SetThreads(2, 3);
  ScopedHook realpath(g_osRealpath, DefaultOsRealpath);

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 2);
  EXPECT_EQ(ns_, 6);
  EXPECT_EQ(realpath.calls, 0);
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_ThreadsAboveMax_ClampsToMax) {
  SetThreads(MAX_THREADS + 4, 1);

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, MAX_THREADS);
  EXPECT_EQ(ns_, MAX_THREADS);
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_SocketsAboveMax_ReducesSocketsPerThread) {
  SetThreads(3, 30);

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 3);
  EXPECT_EQ(ns_, (MAX_SOCKETS / 3) * 3);
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_AutoOnAws_UsesTwoThreadsOfEightSockets) {
  SetThreads(-2, -2);
  InstallVendorFile();
  ScriptReadData("0x1d0f\n");

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 2);
  EXPECT_EQ(ns_, 16);
  EXPECT_EQ(realpathAsked_, "/sys/class/net/eth0/device/vendor");
  EXPECT_EQ(opened_, "/sys/devices/pci0000:00/0000:00:03.0/vendor");
  EXPECT_EQ(g_closedFds, std::vector<int>{kVendorFd});
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_AutoOnGcp_UsesFourThreadsOfOneSocket) {
  SetThreads(-2, -2);
  InstallVendorFile();
  ScriptReadData("0x1ae0\n");

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 4);
  EXPECT_EQ(ns_, 4);
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_AutoOnOtherVendor_UsesNoHelperThreads) {
  SetThreads(-2, -2);
  InstallVendorFile();
  ScriptReadData("0x8086\n");

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 0);
  EXPECT_EQ(ns_, 0);
  EXPECT_EQ(g_closedFds, std::vector<int>{kVendorFd});
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_AutoWithoutVendorFile_UsesNoHelperThreads) {
  SetThreads(-2, -2);

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 0);
  EXPECT_EQ(ns_, 0);
  EXPECT_TRUE(g_closedFds.empty());
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_AutoThreadsOnly_KeepsExplicitSocketsPerThread) {
  SetThreads(-2, 3);
  InstallVendorFile();
  ScriptReadData("0x1d0f\n");

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 2);
  EXPECT_EQ(ns_, 6);
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_AutoSocketsOnly_KeepsExplicitThreads) {
  SetThreads(3, -2);
  InstallVendorFile();
  ScriptReadData("0x1d0f\n");

  ASSERT_EQ(GetCounts(), ncclSuccess);

  EXPECT_EQ(nt_, 3);
  EXPECT_EQ(ns_, 24);
}

TEST_F(NetSocketThreadsMicrotest, GetNsockNthread_VendorReadFails_ReturnsSystemErrorAndClosesFile) {
  SetThreads(-2, -2);
  InstallVendorFile();
  ScriptRead(-1, EIO, "");

  EXPECT_EQ(GetCounts(), ncclSystemError);

  EXPECT_EQ(g_closedFds, std::vector<int>{kVendorFd});
}

class NetSocketListenMicrotest : public NetSocketMicrotest {
 protected:
  alignas(ncclNetSocketHandle) char handle_[NCCL_NET_HANDLE_MAXSIZE];

  void SetUp() override {
    NetSocketMicrotest::SetUp();
    std::memset(handle_, 0xa5, sizeof(handle_));
  }
};

TEST_F(NetSocketListenMicrotest, Listen_ValidDevice_PublishesHandleAndRecordsCounts) {
  SetDevices(2);
  SetThreads(2, 3);
  void* listenComm = nullptr;

  ASSERT_EQ(ncclNetSocketListen(nullptr, 1, handle_, &listenComm), ncclSuccess);

  auto* handle = reinterpret_cast<ncclNetSocketHandle*>(handle_);
  auto* lComm = static_cast<ncclNetSocketListenComm*>(listenComm);
  EXPECT_EQ(handle->connectAddr.sin.sin_addr.s_addr, DevAddr(1).sin.sin_addr.s_addr);
  EXPECT_EQ(handle->connectAddr.sin.sin_port, htons(kListenPort));
  EXPECT_EQ(handle->magic, kMagic);
  EXPECT_EQ(handle->nSocks, 6);
  EXPECT_EQ(handle->nThreads, 2);
  EXPECT_EQ(handle->stage.state, ncclNetSocketCommStateStart);
  EXPECT_EQ(handle->stage.iteration, 0);
  EXPECT_EQ(handle->stage.sock, nullptr);
  EXPECT_EQ(handle->stage.comm, nullptr);
  EXPECT_EQ(lComm->dev, 1);
  EXPECT_EQ(lComm->nSocks, 6);
  EXPECT_EQ(lComm->nThreads, 2);
  EXPECT_EQ(ncclNetSocketCloseListen(listenComm), ncclSuccess);
}

TEST_F(NetSocketListenMicrotest, Listen_InvalidDevice_ReturnsInternalErrorWithoutComm) {
  SetDevices(1);
  void* listenComm = this;

  EXPECT_EQ(ncclNetSocketListen(nullptr, 1, handle_, &listenComm), ncclInternalError);

  EXPECT_EQ(listenComm, this);
}

TEST_F(NetSocketListenMicrotest, Listen_ListenerFails_ClosesSocketAndPropagatesError) {
  SetDevices(1);
  ScopedHook listenHook(g_sockListen, [](ncclSocket* sock) {
    sock->socketDescriptor = 55;
    return ncclSystemError;
  });
  void* listenComm = this;

  EXPECT_EQ(ncclNetSocketListen(nullptr, 0, handle_, &listenComm), ncclSystemError);

  EXPECT_EQ(listenComm, this);
  EXPECT_EQ(g_sock.closed, std::vector<int>{55});
}

TEST_F(NetSocketListenMicrotest, Listen_ThreadProbeFails_ClosesListenerAndPropagatesError) {
  SetDevices(1);
  SetThreads(-2, -2);
  ScopedHook realpath(g_osRealpath, [](const char*, char*) { return strdup("/vendor"); });
  ScopedHook openHook(g_open, [](const char*, int, mode_t) { return 11; });
  g_readScript = {{-1, EIO, ""}};
  int listenFd = g_sock.nextFd;
  void* listenComm = this;

  EXPECT_EQ(ncclNetSocketListen(nullptr, 0, handle_, &listenComm), ncclSystemError);

  EXPECT_EQ(listenComm, this);
  EXPECT_EQ(g_sock.closed, std::vector<int>{listenFd});
}

TEST_F(NetSocketListenMicrotest, CloseListen_Null_Succeeds) {
  EXPECT_EQ(ncclNetSocketCloseListen(nullptr), ncclSuccess);
}

TEST_F(NetSocketListenMicrotest, CloseListen_ReadyListener_ClosesSocket) {
  auto* lComm = static_cast<ncclNetSocketListenComm*>(calloc(1, sizeof(ncclNetSocketListenComm)));
  ReadySocket(&lComm->sock);
  int fd = lComm->sock.socketDescriptor;

  EXPECT_EQ(ncclNetSocketCloseListen(lComm), ncclSuccess);

  EXPECT_EQ(g_sock.closed, std::vector<int>{fd});
}

TEST_F(NetSocketListenMicrotest, CloseListen_UnreadyListener_SkipsClose) {
  auto* lComm = static_cast<ncclNetSocketListenComm*>(calloc(1, sizeof(ncclNetSocketListenComm)));
  lComm->sock.state = ncclSocketStateInitialized;
  ScopedHook closeHook(g_sockClose, DefaultSockClose);

  EXPECT_EQ(ncclNetSocketCloseListen(lComm), ncclSuccess);

  EXPECT_EQ(closeHook.calls, 0);
}

TEST_F(NetSocketListenMicrotest, CloseListen_ReadyCheckFails_PropagatesError) {
  auto* lComm = static_cast<ncclNetSocketListenComm*>(calloc(1, sizeof(ncclNetSocketListenComm)));
  ReadySocket(&lComm->sock);
  ScopedHook ready(g_sockReady, [](ncclSocket*, int*) { return ncclRemoteError; });

  EXPECT_EQ(ncclNetSocketCloseListen(lComm), ncclRemoteError);

  free(lComm);
}

class NetSocketConnectMicrotest : public NetSocketMicrotest {
 protected:
  ncclNetSocketHandle handle_{};

  void SetUp() override {
    NetSocketMicrotest::SetUp();
    SetDevices(1);
  }

  void MakeHandle(int nSocks, int nThreads) {
    handle_ = ncclNetSocketHandle{};
    handle_.connectAddr = DevAddr(7);
    handle_.connectAddr.sin.sin_port = htons(5000);
    handle_.magic = 0xabcULL;
    handle_.nSocks = nSocks;
    handle_.nThreads = nThreads;
  }

  ncclResult_t Connect(void** sendComm, int dev = 0) {
    return ncclNetSocketConnect(nullptr, dev, &handle_, sendComm, nullptr);
  }

  void TearDown() override {
    DiscardStagedComm();
    NetSocketMicrotest::TearDown();
  }

  // A completed connect hands its comm to the test; a failed one leaves it staged.
  void DiscardStagedComm() {
    ncclNetSocketComm* comm = handle_.stage.comm;
    if (comm == nullptr || std::find(comms_.begin(), comms_.end(), comm) != comms_.end()) return;
    free(comm->inlineData);
    delete comm;
    handle_.stage.comm = nullptr;
  }
};

TEST_F(NetSocketConnectMicrotest, Connect_DeviceOutOfRange_ReturnsInternalError) {
  MakeHandle(1, 1);
  ScopedHook init(g_sockInit, DefaultSockInit);
  void* sendComm = nullptr;

  EXPECT_EQ(Connect(&sendComm, -1), ncclInternalError);
  EXPECT_EQ(Connect(&sendComm, 1), ncclInternalError);

  EXPECT_EQ(init.calls, 0);
}

TEST_F(NetSocketConnectMicrotest, Connect_AllSocketsReady_SendsEachIndexAndReturnsConfiguredComm) {
  SetDevices(2);
  MakeHandle(2, 1);
  ScopedHook device(g_hipGetDevice, [](int* dev) {
    *dev = 3;
    return hipSuccess;
  });
  std::vector<ncclSocket*> inits;
  ScopedHook init(g_sockInit, [&](ncclSocket* sock, const ncclSocketAddress* addr, uint64_t magic,
                                  ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int retry) {
    inits.push_back(sock);
    EXPECT_EQ(std::memcmp(addr, &handle_.connectAddr, sizeof(*addr)), 0);
    EXPECT_EQ(magic, handle_.magic);
    EXPECT_EQ(type, ncclSocketTypeNetSocket);
    EXPECT_EQ(abortFlag, nullptr);
    EXPECT_EQ(asyncFlag, 1);
    return DefaultSockInit(sock, addr, magic, type, abortFlag, asyncFlag, retry);
  });
  void* sendComm = nullptr;

  ASSERT_EQ(Connect(&sendComm, 1), ncclSuccess);

  ASSERT_NE(sendComm, nullptr);
  Track(sendComm);
  auto* comm = static_cast<ncclNetSocketComm*>(sendComm);
  EXPECT_EQ(inits, (std::vector<ncclSocket*>{&comm->socks[0], &comm->socks[1], &comm->ctrlSock}));
  EXPECT_EQ(Sent(comm->socks[0]), std::string(1, '\0'));
  EXPECT_EQ(Sent(comm->socks[1]), std::string(1, '\1'));
  EXPECT_EQ(Sent(comm->ctrlSock), std::string(1, '\2'));
  EXPECT_EQ(comm->nSocks, 2);
  EXPECT_EQ(comm->nThreads, 1);
  EXPECT_EQ(comm->dev, 1);
  EXPECT_EQ(comm->cudaDev, 3);
  EXPECT_NE(comm->inlineData, nullptr);
}

TEST_F(NetSocketConnectMicrotest, Connect_NoDataSockets_ConnectsControlSocketOnly) {
  MakeHandle(0, 0);
  ScopedHook init(g_sockInit, DefaultSockInit);
  void* sendComm = nullptr;

  ASSERT_EQ(Connect(&sendComm), ncclSuccess);

  ASSERT_NE(sendComm, nullptr);
  Track(sendComm);
  EXPECT_EQ(init.calls, 1);
  EXPECT_EQ(Sent(static_cast<ncclNetSocketComm*>(sendComm)->ctrlSock), std::string(1, '\0'));
}

TEST_F(NetSocketConnectMicrotest, Connect_FirstSocketNotReady_ResumesWithoutReconnecting) {
  MakeHandle(1, 1);
  ScopedHook init(g_sockInit, DefaultSockInit);
  int calls = 0;
  ScopedHook ready(g_sockReady, [&](ncclSocket* sock, int* running) {
    if (calls++ == 0) {
      *running = 0;
      return ncclSuccess;
    }
    return DefaultSockReady(sock, running);
  });
  void* sendComm = this;

  ASSERT_EQ(Connect(&sendComm), ncclSuccess);
  EXPECT_EQ(sendComm, nullptr);

  ASSERT_EQ(Connect(&sendComm), ncclSuccess);

  ASSERT_NE(sendComm, nullptr);
  Track(sendComm);
  EXPECT_EQ(init.calls, 2);
}

TEST_F(NetSocketConnectMicrotest, Connect_LaterSocketNotReady_ResumesOnThatSocket) {
  MakeHandle(2, 1);
  ScopedHook init(g_sockInit, DefaultSockInit);
  int calls = 0;
  ScopedHook ready(g_sockReady, [&](ncclSocket* sock, int* running) {
    if (calls++ == 1) {
      *running = 0;
      return ncclSuccess;
    }
    return DefaultSockReady(sock, running);
  });
  void* sendComm = this;

  ASSERT_EQ(Connect(&sendComm), ncclSuccess);
  EXPECT_EQ(sendComm, nullptr);

  ASSERT_EQ(Connect(&sendComm), ncclSuccess);

  ASSERT_NE(sendComm, nullptr);
  Track(sendComm);
  auto* comm = static_cast<ncclNetSocketComm*>(sendComm);
  EXPECT_EQ(init.calls, 3);
  EXPECT_EQ(Sent(comm->socks[0]), std::string(1, '\0'));
  EXPECT_EQ(Sent(comm->socks[1]), std::string(1, '\1'));
  EXPECT_EQ(Sent(comm->ctrlSock), std::string(1, '\2'));
}

TEST_F(NetSocketConnectMicrotest, Connect_IndexNotSent_ReturnsNoCommAndResumesSend) {
  MakeHandle(1, 1);
  ScopedHook connectHook(g_sockConnect, DefaultSockConnect);
  int calls = 0;
  ScopedHook progress(g_sockProgress, [&](int op, ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
    if (calls++ == 0) return ncclSuccess;
    return DefaultSockProgress(op, sock, ptr, size, offset, closed);
  });
  void* sendComm = this;

  ASSERT_EQ(Connect(&sendComm), ncclSuccess);
  EXPECT_EQ(sendComm, nullptr);

  ASSERT_EQ(Connect(&sendComm), ncclSuccess);

  ASSERT_NE(sendComm, nullptr);
  Track(sendComm);
  auto* comm = static_cast<ncclNetSocketComm*>(sendComm);
  EXPECT_EQ(connectHook.calls, 2);
  EXPECT_EQ(Sent(comm->socks[0]), std::string(1, '\0'));
  EXPECT_EQ(Sent(comm->ctrlSock), std::string(1, '\1'));
}

TEST_F(NetSocketConnectMicrotest, Connect_SocketStepFails_PropagatesError) {
  {
    MakeHandle(1, 1);
    ScopedHook init(g_sockInit, [](ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType,
                                   volatile uint32_t*, int, int) { return ncclSystemError; });
    void* sendComm = this;
    EXPECT_EQ(Connect(&sendComm), ncclSystemError);
    EXPECT_EQ(sendComm, nullptr);
    DiscardStagedComm();
  }
  {
    MakeHandle(1, 1);
    ScopedHook connectHook(g_sockConnect, [](ncclSocket*) { return ncclRemoteError; });
    void* sendComm = this;
    EXPECT_EQ(Connect(&sendComm), ncclRemoteError);
    EXPECT_EQ(sendComm, nullptr);
    DiscardStagedComm();
  }
  {
    MakeHandle(1, 1);
    ScopedHook ready(g_sockReady, [](ncclSocket*, int*) { return ncclInvalidUsage; });
    void* sendComm = this;
    EXPECT_EQ(Connect(&sendComm), ncclInvalidUsage);
    EXPECT_EQ(sendComm, nullptr);
    DiscardStagedComm();
  }
  {
    MakeHandle(1, 1);
    ScopedHook progress(g_sockProgress,
                        [](int, ncclSocket*, void*, int, int*, bool*) { return ncclInternalError; });
    void* sendComm = this;
    EXPECT_EQ(Connect(&sendComm), ncclInternalError);
    EXPECT_EQ(sendComm, nullptr);
    DiscardStagedComm();
  }
}

TEST_F(NetSocketConnectMicrotest, Connect_GetDeviceFails_ReturnsCudaErrorBeforeConnecting) {
  MakeHandle(1, 1);
  ScopedHook device(g_hipGetDevice, [](int*) { return hipErrorNoDevice; });
  ScopedHook init(g_sockInit, DefaultSockInit);
  void* sendComm = this;

  EXPECT_EQ(Connect(&sendComm), ncclUnhandledCudaError);

  EXPECT_EQ(sendComm, nullptr);
  EXPECT_EQ(init.calls, 0);
}

class NetSocketAcceptMicrotest : public NetSocketMicrotest {
 protected:
  ncclNetSocketListenComm* lComm_ = nullptr;

  void TearDown() override {
    TearDownListenComm();
    NetSocketMicrotest::TearDown();
  }

  void TearDownListenComm() {
    if (!lComm_) return;
    free(lComm_->stage.sock);
    if (lComm_->stage.comm) free(lComm_->stage.comm->inlineData);
    delete lComm_->stage.comm;
    free(lComm_);
    lComm_ = nullptr;
  }

  void MakeListenComm(int nSocks, int nThreads) {
    lComm_ = static_cast<ncclNetSocketListenComm*>(calloc(1, sizeof(ncclNetSocketListenComm)));
    ReadySocket(&lComm_->sock);
    lComm_->nSocks = nSocks;
    lComm_->nThreads = nThreads;
    lComm_->dev = 1;
  }

  ncclResult_t Accept(void** recvComm) { return ncclNetSocketAccept(lComm_, recvComm, nullptr); }

  // Accept already freed the rejected socket.
  void ForgetStagedSocket() { lComm_->stage.sock = nullptr; }
};

TEST_F(NetSocketAcceptMicrotest, Accept_PeerSendsIndices_PlacesEachConnectionAndReturnsConfiguredComm) {
  MakeListenComm(2, 1);
  ScopedHook device(g_hipGetDevice, [](int* dev) {
    *dev = 5;
    return hipSuccess;
  });
  std::vector<ncclSocket*> inits;
  ScopedHook init(g_sockInit, [&](ncclSocket* sock, const ncclSocketAddress* addr, uint64_t magic,
                                  ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int retry) {
    inits.push_back(sock);
    EXPECT_EQ(addr, nullptr);
    return DefaultSockInit(sock, addr, magic, type, abortFlag, asyncFlag, retry);
  });
  std::vector<ncclSocket*> accepted;
  ScopedHook acceptHook(g_sockAccept, [&](ncclSocket* sock, ncclSocket* listenSock, bool retry) {
    accepted.push_back(sock);
    EXPECT_EQ(listenSock, &lComm_->sock);
    EXPECT_EQ(sock->state, ncclSocketStateInitialized);
    return DefaultSockAccept(sock, listenSock, retry);
  });
  g_sock.acceptIndices = {2, 0, 1};
  int firstFd = g_sock.nextFd;
  void* recvComm = nullptr;

  ASSERT_EQ(Accept(&recvComm), ncclSuccess);

  ASSERT_NE(recvComm, nullptr);
  Track(recvComm);
  auto* comm = static_cast<ncclNetSocketComm*>(recvComm);
  EXPECT_EQ(acceptHook.calls, 3);
  EXPECT_EQ(inits, accepted);
  EXPECT_EQ(comm->ctrlSock.socketDescriptor, firstFd);
  EXPECT_EQ(comm->socks[0].socketDescriptor, firstFd + 1);
  EXPECT_EQ(comm->socks[1].socketDescriptor, firstFd + 2);
  EXPECT_EQ(comm->nSocks, 2);
  EXPECT_EQ(comm->nThreads, 1);
  EXPECT_EQ(comm->dev, 1);
  EXPECT_EQ(comm->cudaDev, 5);
  EXPECT_NE(comm->inlineData, nullptr);
}

TEST_F(NetSocketAcceptMicrotest, Accept_Completes_ResetsListenerStage) {
  MakeListenComm(1, 1);
  g_sock.acceptIndices = {0, 1};
  void* first = nullptr;
  ASSERT_EQ(Accept(&first), ncclSuccess);
  ASSERT_NE(first, nullptr);
  Track(first);

  EXPECT_EQ(lComm_->stage.state, ncclNetSocketCommStateStart);
  EXPECT_EQ(lComm_->stage.iteration, 0);
  EXPECT_EQ(lComm_->stage.sock, nullptr);
  EXPECT_EQ(lComm_->stage.comm, nullptr);
}

TEST_F(NetSocketAcceptMicrotest, Accept_FirstSocketNotReady_ResumesWithoutAcceptingAgain) {
  MakeListenComm(1, 1);
  g_sock.acceptIndices = {0, 1};
  ScopedHook acceptHook(g_sockAccept, DefaultSockAccept);
  int calls = 0;
  ScopedHook ready(g_sockReady, [&](ncclSocket* sock, int* running) {
    if (calls++ == 0) {
      *running = 0;
      return ncclSuccess;
    }
    return DefaultSockReady(sock, running);
  });
  void* recvComm = this;

  ASSERT_EQ(Accept(&recvComm), ncclSuccess);
  EXPECT_EQ(recvComm, nullptr);

  ASSERT_EQ(Accept(&recvComm), ncclSuccess);

  ASSERT_NE(recvComm, nullptr);
  Track(recvComm);
  EXPECT_EQ(acceptHook.calls, 2);
}

TEST_F(NetSocketAcceptMicrotest, Accept_LaterSocketNotReady_ResumesOnThatSocket) {
  MakeListenComm(1, 1);
  g_sock.acceptIndices = {1, 0};
  ScopedHook acceptHook(g_sockAccept, DefaultSockAccept);
  int calls = 0;
  ScopedHook ready(g_sockReady, [&](ncclSocket* sock, int* running) {
    if (calls++ == 1) {
      *running = 0;
      return ncclSuccess;
    }
    return DefaultSockReady(sock, running);
  });
  int firstFd = g_sock.nextFd;
  void* recvComm = this;

  ASSERT_EQ(Accept(&recvComm), ncclSuccess);
  EXPECT_EQ(recvComm, nullptr);

  ASSERT_EQ(Accept(&recvComm), ncclSuccess);

  ASSERT_NE(recvComm, nullptr);
  Track(recvComm);
  auto* comm = static_cast<ncclNetSocketComm*>(recvComm);
  EXPECT_EQ(acceptHook.calls, 2);
  EXPECT_EQ(comm->ctrlSock.socketDescriptor, firstFd);
  EXPECT_EQ(comm->socks[0].socketDescriptor, firstFd + 1);
}

TEST_F(NetSocketAcceptMicrotest, Accept_IndexNotReceived_ReturnsNoCommAndResumes) {
  MakeListenComm(1, 1);
  std::vector<int> acceptedFds;
  ScopedHook acceptHook(g_sockAccept, [&](ncclSocket* sock, ncclSocket* listenSock, bool retry) {
    ncclResult_t ret = DefaultSockAccept(sock, listenSock, retry);
    acceptedFds.push_back(sock->socketDescriptor);
    return ret;
  });
  void* recvComm = this;

  ASSERT_EQ(Accept(&recvComm), ncclSuccess);
  EXPECT_EQ(recvComm, nullptr);
  ASSERT_EQ(acceptedFds.size(), 1u);

  QueueInbound(acceptedFds[0], std::string(1, '\1'));
  g_sock.acceptIndices = {0};
  ASSERT_EQ(Accept(&recvComm), ncclSuccess);

  ASSERT_NE(recvComm, nullptr);
  Track(recvComm);
  auto* comm = static_cast<ncclNetSocketComm*>(recvComm);
  ASSERT_EQ(acceptedFds.size(), 2u);
  EXPECT_EQ(comm->ctrlSock.socketDescriptor, acceptedFds[0]);
  EXPECT_EQ(comm->socks[0].socketDescriptor, acceptedFds[1]);
}

TEST_F(NetSocketAcceptMicrotest, Accept_IndexOutOfRange_ClosesConnectionAndReturnsInternalError) {
  MakeListenComm(1, 1);
  g_sock.acceptIndices = {2};
  int fd = g_sock.nextFd;
  void* recvComm = this;

  EXPECT_EQ(Accept(&recvComm), ncclInternalError);

  EXPECT_EQ(recvComm, nullptr);
  EXPECT_EQ(g_sock.closed, std::vector<int>{fd});
  ForgetStagedSocket();
}

TEST_F(NetSocketAcceptMicrotest, Accept_DuplicateIndex_ClosesDuplicateKeepsFirstAndReturnsInternalError) {
  MakeListenComm(1, 1);
  g_sock.acceptIndices = {0, 0};
  int firstFd = g_sock.nextFd;
  void* recvComm = this;

  EXPECT_EQ(Accept(&recvComm), ncclInternalError);

  EXPECT_EQ(recvComm, nullptr);
  EXPECT_EQ(g_sock.closed, std::vector<int>{firstFd + 1});
  EXPECT_EQ(lComm_->stage.comm->socks[0].socketDescriptor, firstFd);
  ForgetStagedSocket();
}

TEST_F(NetSocketAcceptMicrotest, Accept_SocketStepFails_PropagatesError) {
  {
    MakeListenComm(1, 1);
    ScopedHook acceptHook(g_sockAccept, [](ncclSocket*, ncclSocket*, bool) { return ncclSystemError; });
    void* recvComm = this;
    EXPECT_EQ(Accept(&recvComm), ncclSystemError);
    EXPECT_EQ(recvComm, nullptr);
    TearDownListenComm();
  }
  {
    MakeListenComm(1, 1);
    ScopedHook ready(g_sockReady, [](ncclSocket*, int*) { return ncclRemoteError; });
    void* recvComm = this;
    EXPECT_EQ(Accept(&recvComm), ncclRemoteError);
    EXPECT_EQ(recvComm, nullptr);
    TearDownListenComm();
  }
  {
    MakeListenComm(1, 1);
    ScopedHook progress(g_sockProgress,
                        [](int, ncclSocket*, void*, int, int*, bool*) { return ncclInternalError; });
    void* recvComm = this;
    EXPECT_EQ(Accept(&recvComm), ncclInternalError);
    EXPECT_EQ(recvComm, nullptr);
  }
}

TEST_F(NetSocketAcceptMicrotest, Accept_GetDeviceFails_ReturnsCudaErrorBeforeAccepting) {
  MakeListenComm(1, 1);
  ScopedHook device(g_hipGetDevice, [](int*) { return hipErrorNoDevice; });
  ScopedHook acceptHook(g_sockAccept, DefaultSockAccept);
  void* recvComm = this;

  EXPECT_EQ(Accept(&recvComm), ncclUnhandledCudaError);

  EXPECT_EQ(recvComm, nullptr);
  EXPECT_EQ(acceptHook.calls, 0);
}

using NetSocketRequestMicrotest = NetSocketMicrotest;

TEST_F(NetSocketRequestMicrotest, Isend_FreeSlot_ReturnsSendRequestOnComm) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  char data[10];

  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, data, sizeof(data)));

  ASSERT_EQ(req, &comm->requests[0]);
  EXPECT_EQ(req->op, NCCL_SOCKET_SEND);
  EXPECT_EQ(req->data, data);
  EXPECT_EQ(req->size, 10);
  EXPECT_EQ(req->comm, comm);
  EXPECT_EQ(req->ctrlSock, &comm->ctrlSock);
  EXPECT_EQ(req->inlineData, comm->inlineData);
  EXPECT_EQ(req->nSubs, 0);
  EXPECT_EQ(req->used, 1);
}

TEST_F(NetSocketRequestMicrotest, Irecv_FreeSlot_ReturnsRecvRequestOnComm) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  char data[10];

  auto* req = static_cast<ncclNetSocketRequest*>(Irecv(comm, data, sizeof(data)));

  ASSERT_EQ(req, &comm->requests[0]);
  EXPECT_EQ(req->op, NCCL_SOCKET_RECV);
  EXPECT_EQ(req->data, data);
  EXPECT_EQ(req->size, 10);
  EXPECT_EQ(req->comm, comm);
  EXPECT_EQ(req->ctrlSock, &comm->ctrlSock);
  EXPECT_EQ(req->used, 1);
}

TEST_F(NetSocketRequestMicrotest, Irecv_MoreThanOneBuffer_ReturnsInternalErrorWithoutRequest) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  char a[4], b[4];
  void* data[2] = {a, b};
  size_t sizes[2] = {4, 4};
  int tags[2] = {0, 0};
  void* request = nullptr;

  EXPECT_EQ(ncclNetSocketIrecv(comm, 2, data, sizes, tags, nullptr, nullptr, &request), ncclInternalError);

  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(Isend(comm, a, 4), &comm->requests[0]);
}

TEST_F(NetSocketRequestMicrotest, Isend_EarlierSlotsBusy_TakesNextSlotWithItsInlineArea) {
  SetParam("SOCKET_INLINE", 64);
  ncclNetSocketComm* comm = MakeComm(0, 0);
  char data[4];

  for (int i = 0; i < 3; i++) {
    auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, data, sizeof(data)));
    ASSERT_EQ(req, &comm->requests[i]);
    EXPECT_EQ(req->inlineData, static_cast<char*>(comm->inlineData) + i * (SOCKET_CTRL_SIZE + 64));
  }
}

TEST_F(NetSocketRequestMicrotest, Isend_CompletedSlot_IsReused) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  char data[4] = {};
  void* first = Isend(comm, data, sizeof(data));
  void* second = Isend(comm, data, sizeof(data));
  int done = 0;
  ASSERT_EQ(ncclNetSocketTest(first, &done, nullptr), ncclSuccess);
  ASSERT_EQ(done, 1);

  EXPECT_EQ(Isend(comm, data, sizeof(data)), first);
  EXPECT_NE(first, second);
}

TEST_F(NetSocketRequestMicrotest, Isend_AllSlotsBusy_ReturnsInternalError) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  char data[4];
  for (int i = 0; i < MAX_REQUESTS; i++) Isend(comm, data, sizeof(data));
  void* request = nullptr;

  EXPECT_EQ(ncclNetSocketIsend(comm, data, sizeof(data), 0, nullptr, nullptr, &request), ncclInternalError);
}

class NetSocketTransferMicrotest : public NetSocketMicrotest {
 protected:
  int done_ = -1;
  int size_ = -1;
  int progressCalls_ = 0;

  ncclResult_t Poll(void* request) { return ncclNetSocketTest(request, &done_, &size_); }

  // Only the first progress call moves bytes.
  void BlockAfterFirstProgress() {
    progressCalls_ = 0;
    g_sockProgress = [this](int op, ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
      if (progressCalls_++ > 0) return ncclSuccess;
      return DefaultSockProgress(op, sock, ptr, size, offset, closed);
    };
  }
};

TEST_F(NetSocketTransferMicrotest, Test_NullRequest_ReturnsInternalErrorNotDone) {
  EXPECT_EQ(Poll(nullptr), ncclInternalError);
  EXPECT_EQ(done_, 0);
}

TEST_F(NetSocketTransferMicrotest, Test_SmallSend_SendsSizeAndPayloadAndCompletes) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string data = Pattern(100);
  void* req = Isend(comm, &data[0], data.size());

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 100);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(100) + data);
}

TEST_F(NetSocketTransferMicrotest, Test_LargeSend_SendsSizeFrameBeforePayload) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string data = Pattern(1000);
  void* req = Isend(comm, &data[0], data.size());
  BlockAfterFirstProgress();

  ASSERT_EQ(Poll(req), ncclSuccess);
  EXPECT_EQ(done_, 0);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(1000));

  g_sockProgress = DefaultSockProgress;
  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 1000);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(1000) + data);
}

TEST_F(NetSocketTransferMicrotest, Test_SendUpToInlineSize_CarriesPayloadInSizeFrame) {
  for (int64_t inlineSize : {int64_t{128}, int64_t{16}}) {
    SCOPED_TRACE(inlineSize);
    SetParam("SOCKET_INLINE", inlineSize);
    ncclNetSocketComm* comm = MakeComm(0, 0);
    std::string data = Pattern(inlineSize + 1);

    BlockAfterFirstProgress();
    ASSERT_EQ(Poll(Isend(comm, &data[0], inlineSize)), ncclSuccess);
    EXPECT_EQ(done_, 1);

    BlockAfterFirstProgress();
    ASSERT_EQ(Poll(Isend(comm, &data[0], inlineSize + 1)), ncclSuccess);
    EXPECT_EQ(done_, 0);
    g_sockProgress = DefaultSockProgress;
  }
}

TEST_F(NetSocketTransferMicrotest, Test_SendNotAccepted_ReturnsNotDoneAndSendsOnce) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string data = Pattern(50);
  void* req = Isend(comm, &data[0], data.size());
  int calls = 0;
  ScopedHook progress(g_sockProgress, [&](int op, ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
    if (calls++ == 0) return ncclSuccess;
    return DefaultSockProgress(op, sock, ptr, size, offset, closed);
  });

  ASSERT_EQ(Poll(req), ncclSuccess);
  EXPECT_EQ(done_, 0);
  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(50) + data);
}

TEST_F(NetSocketTransferMicrotest, Test_SmallSendAcceptedOneByteAtATime_CompletesInOneCall) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string data = Pattern(20);
  void* req = Isend(comm, &data[0], data.size());
  g_sock.chunk = 1;

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(20) + data);
}

TEST_F(NetSocketTransferMicrotest, Test_LargeSendPartialProgress_CompletesAcrossCalls) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string data = Pattern(1000);
  void* req = Isend(comm, &data[0], data.size());
  g_sock.chunk = 300;

  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(Poll(req), ncclSuccess);
    EXPECT_EQ(done_, 0);
  }
  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 1000);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(1000) + data);
}

TEST_F(NetSocketTransferMicrotest, Test_ZeroByteSend_CompletesWithSizeFrameOnly) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  void* req = Isend(comm, nullptr, 0);

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 0);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(0));
}

TEST_F(NetSocketTransferMicrotest, Test_MainThreadTransferWithNullSizeOut_Completes) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string data = Pattern(1000);
  void* req = Isend(comm, &data[0], data.size());

  ASSERT_EQ(ncclNetSocketTest(req, &done_, nullptr), ncclSuccess);

  EXPECT_EQ(done_, 1);
}

TEST_F(NetSocketTransferMicrotest, Test_SmallRecv_ReceivesPayloadAndReportsSenderSize) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string payload = Pattern(100);
  std::string buf(256, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  QueueInbound(comm->ctrlSock, SizeFrame(100) + payload);

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 100);
  EXPECT_EQ(buf.substr(0, 100), payload);
  EXPECT_EQ(Inbound(comm->ctrlSock), "");
}

TEST_F(NetSocketTransferMicrotest, Test_SmallRecvDeliveredOneByteAtATime_CompletesInOneCall) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string payload = Pattern(20);
  std::string buf(64, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  QueueInbound(comm->ctrlSock, SizeFrame(20) + payload);
  g_sock.chunk = 1;

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 20);
  EXPECT_EQ(buf.substr(0, 20), payload);
}

TEST_F(NetSocketTransferMicrotest, Test_LargeRecv_ReceivesPayloadOnControlSocket) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string payload = Pattern(1000);
  std::string buf(2000, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  QueueInbound(comm->ctrlSock, SizeFrame(1000) + payload);

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 1000);
  EXPECT_EQ(buf.substr(0, 1000), payload);
}

TEST_F(NetSocketTransferMicrotest, Test_RecvNothingArrived_ReturnsNotDoneUntilDataArrives) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string payload = Pattern(10);
  std::string buf(32, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());

  ASSERT_EQ(Poll(req), ncclSuccess);
  EXPECT_EQ(done_, 0);

  QueueInbound(comm->ctrlSock, SizeFrame(10) + payload);
  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(buf.substr(0, 10), payload);
}

TEST_F(NetSocketTransferMicrotest, Test_RecvPartialPayload_CompletesAcrossCalls) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string payload = Pattern(1000);
  std::string buf(2000, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  QueueInbound(comm->ctrlSock, SizeFrame(1000) + payload.substr(0, 300));

  ASSERT_EQ(Poll(req), ncclSuccess);
  EXPECT_EQ(done_, 0);

  QueueInbound(comm->ctrlSock, payload.substr(300));
  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(buf.substr(0, 1000), payload);
}

TEST_F(NetSocketTransferMicrotest, Test_RecvSenderLargerThanPosted_ReturnsInvalidUsage) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string buf(100, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  QueueInbound(comm->ctrlSock, SizeFrame(101));

  EXPECT_EQ(Poll(req), ncclInvalidUsage);

  EXPECT_EQ(done_, 0);
}

TEST_F(NetSocketTransferMicrotest, Test_RecvTruncatedAndPeerAddressUnavailable_PropagatesGetAddrError) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string buf(100, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  QueueInbound(comm->ctrlSock, SizeFrame(101));
  ScopedHook getAddr(g_sockGetAddr, [](ncclSocket*, ncclSocketAddress*) { return ncclRemoteError; });

  EXPECT_EQ(Poll(req), ncclRemoteError);
}

TEST_F(NetSocketTransferMicrotest, Test_RecvSenderEqualToPosted_Completes) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string payload = Pattern(100);
  std::string buf(100, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  QueueInbound(comm->ctrlSock, SizeFrame(100) + payload);

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(buf, payload);
}

TEST_F(NetSocketTransferMicrotest, Test_SizeExchangeFails_PropagatesError) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  char data[8] = {};
  ScopedHook progress(g_sockProgress, [](int, ncclSocket*, void*, int, int*, bool*) { return ncclRemoteError; });

  EXPECT_EQ(Poll(Isend(comm, data, sizeof(data))), ncclRemoteError);
  EXPECT_EQ(Poll(Irecv(comm, data, sizeof(data))), ncclRemoteError);
}

TEST_F(NetSocketTransferMicrotest, Test_PayloadFails_PropagatesError) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  std::string data = Pattern(1000);
  int calls = 0;
  ScopedHook progress(g_sockProgress, [&](int op, ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
    if (calls++ > 0) return ncclRemoteError;
    return DefaultSockProgress(op, sock, ptr, size, offset, closed);
  });

  EXPECT_EQ(Poll(Isend(comm, &data[0], data.size())), ncclRemoteError);
}

class NetSocketTaskMicrotest : public NetSocketTransferMicrotest {
 protected:
  // A queue that already exists keeps GetTask from starting a helper thread.
  static void PreprovisionQueues(ncclNetSocketComm* comm, int len) {
    for (int t = 0; t < comm->nThreads; t++) {
      ncclNetSocketTaskQueue& queue = comm->threadResources[t].threadTaskQueue;
      queue.tasks = static_cast<ncclNetSocketTask*>(calloc(len, sizeof(ncclNetSocketTask)));
      queue.len = len;
      queue.next = 0;
    }
  }

  static void FinishTasks(ncclNetSocketRequest* req, int count) {
    for (int i = 0; i < count; i++) req->tasks[i]->offset = req->tasks[i]->size;
  }
};

TEST_F(NetSocketTaskMicrotest, Test_LargeSend_SplitsPayloadIntoOneTaskPerDataSocket) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(1000);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 0);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(1000));
  ASSERT_EQ(req->nSubs, 2);
  EXPECT_EQ(req->tasks[0]->data, &data[0]);
  EXPECT_EQ(req->tasks[0]->size, 500);
  EXPECT_EQ(req->tasks[1]->data, &data[500]);
  EXPECT_EQ(req->tasks[1]->size, 500);
}

TEST_F(NetSocketTaskMicrotest, Test_MinTaskSizeAboveShare_UsesFewerLargerTasks) {
  SetParam("SOCKET_MIN_TASKSIZE", 400);
  ncclNetSocketComm* comm = MakeComm(4, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(1000);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));

  ASSERT_EQ(Poll(req), ncclSuccess);

  ASSERT_EQ(req->nSubs, 3);
  EXPECT_EQ(req->tasks[0]->size, 400);
  EXPECT_EQ(req->tasks[1]->data, &data[400]);
  EXPECT_EQ(req->tasks[1]->size, 400);
  EXPECT_EQ(req->tasks[2]->data, &data[800]);
  EXPECT_EQ(req->tasks[2]->size, 200);
}

TEST_F(NetSocketTaskMicrotest, Test_MinTaskSizeCoversMessage_UsesOneTask) {
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(1000);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));

  ASSERT_EQ(Poll(req), ncclSuccess);

  ASSERT_EQ(req->nSubs, 1);
  EXPECT_EQ(req->tasks[0]->size, 1000);
}

TEST_F(NetSocketTaskMicrotest, Test_InlineSend_UsesNoTasks) {
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(100);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(req->nSubs, 0);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(100) + data);
}

TEST_F(NetSocketTaskMicrotest, Test_LargeRecv_SplitsSenderSizeAcrossDataSockets) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string buf(4000, '\0');
  auto* req = static_cast<ncclNetSocketRequest*>(Irecv(comm, &buf[0], buf.size()));
  QueueInbound(comm->ctrlSock, SizeFrame(1000));

  ASSERT_EQ(Poll(req), ncclSuccess);

  ASSERT_EQ(req->nSubs, 2);
  EXPECT_EQ(req->tasks[0]->op, NCCL_SOCKET_RECV);
  EXPECT_EQ(req->tasks[0]->size, 500);
  EXPECT_EQ(req->tasks[1]->data, &buf[500]);
  EXPECT_EQ(req->tasks[1]->size, 500);
}

TEST_F(NetSocketTaskMicrotest, Test_SomeTasksPending_ReturnsNotDone) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(1000);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));
  ASSERT_EQ(Poll(req), ncclSuccess);
  FinishTasks(req, 1);

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 0);
}

TEST_F(NetSocketTaskMicrotest, Test_AllTasksFinished_CompletesAndReleasesTasks) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(1000);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));
  ASSERT_EQ(Poll(req), ncclSuccess);
  FinishTasks(req, 2);

  ASSERT_EQ(Poll(req), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(size_, 1000);
  EXPECT_EQ(req->tasks[0]->used, 0);
  EXPECT_EQ(req->tasks[1]->used, 0);
  EXPECT_EQ(Isend(comm, &data[0], data.size()), req);
}

TEST_F(NetSocketTaskMicrotest, Test_AllTasksFinishedWithNullSizeOut_Completes) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(1000);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));
  ASSERT_EQ(Poll(req), ncclSuccess);
  FinishTasks(req, 2);

  ASSERT_EQ(ncclNetSocketTest(req, &done_, nullptr), ncclSuccess);

  EXPECT_EQ(done_, 1);
}

TEST_F(NetSocketTaskMicrotest, Test_TaskFailed_ReturnsTaskError) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 1);
  PreprovisionQueues(comm, 8);
  std::string data = Pattern(1000);
  auto* req = static_cast<ncclNetSocketRequest*>(Isend(comm, &data[0], data.size()));
  ASSERT_EQ(Poll(req), ncclSuccess);
  req->tasks[1]->result = ncclRemoteError;

  EXPECT_EQ(Poll(req), ncclRemoteError);

  EXPECT_EQ(done_, 0);
}

TEST_F(NetSocketTaskMicrotest, GetTask_ConsecutiveTasks_RotateAcrossSocketsAndThreads) {
  ncclNetSocketComm* comm = MakeComm(4, 2);
  PreprovisionQueues(comm, 4);
  char data[16];
  ncclProfilerInfo pInfo{};

  for (int k = 0; k < 5; k++) {
    SCOPED_TRACE(k);
    ncclNetSocketTask* task = nullptr;
    ASSERT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_SEND, data + k, k + 1, &task), ncclSuccess);
    EXPECT_EQ(task, &comm->threadResources[k % 2].threadTaskQueue.tasks[k / 2]);
    EXPECT_EQ(task->sock, &comm->socks[k % 4]);
    EXPECT_EQ(task->op, NCCL_SOCKET_SEND);
    EXPECT_EQ(task->data, data + k);
    EXPECT_EQ(task->size, k + 1);
    EXPECT_EQ(task->offset, 0);
    EXPECT_EQ(task->result, ncclSuccess);
    EXPECT_EQ(task->used, 1);
  }
}

TEST_F(NetSocketTaskMicrotest, GetTask_QueueWraps_ReusesFreedFirstSlot) {
  ncclNetSocketComm* comm = MakeComm(1, 1);
  PreprovisionQueues(comm, 2);
  ncclNetSocketTaskQueue& queue = comm->threadResources[0].threadTaskQueue;
  char data[4];
  ncclProfilerInfo pInfo{};
  ncclNetSocketTask* task = nullptr;
  ASSERT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_RECV, data, 4, &task), ncclSuccess);
  ASSERT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_RECV, data, 4, &task), ncclSuccess);
  queue.tasks[0].offset = 4;
  queue.tasks[0].result = ncclRemoteError;
  queue.tasks[0].used = 0;

  ASSERT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_RECV, data, 4, &task), ncclSuccess);

  EXPECT_EQ(task, &queue.tasks[0]);
  EXPECT_EQ(task->offset, 0);
  EXPECT_EQ(task->result, ncclSuccess);
}

TEST_F(NetSocketTaskMicrotest, GetTask_NextSlotBusy_ReturnsInternalError) {
  ncclNetSocketComm* comm = MakeComm(1, 1);
  PreprovisionQueues(comm, 2);
  comm->threadResources[0].threadTaskQueue.tasks[0].used = 1;
  char data[4];
  ncclProfilerInfo pInfo{};
  ncclNetSocketTask* task = nullptr;

  EXPECT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_SEND, data, 4, &task), ncclInternalError);

  EXPECT_EQ(task, nullptr);
}

class NetSocketHelperThreadMicrotest : public NetSocketMicrotest {
 protected:
  ncclNetSocketComm* comm_ = nullptr;
  ncclNetSocketThreadResources* res_ = nullptr;
  ncclNetSocketTask* tasks_ = nullptr;
  ncclProfilerInfo pInfo_{};

  void SetUp() override {
    NetSocketMicrotest::SetUp();
    comm_ = MakeComm(2, 1);
    res_ = &comm_->threadResources[0];
    res_->comm = comm_;
    res_->pInfo = &pInfo_;
    res_->threadTaskQueue.tasks = static_cast<ncclNetSocketTask*>(calloc(4, sizeof(ncclNetSocketTask)));
    res_->threadTaskQueue.len = 4;
    tasks_ = res_->threadTaskQueue.tasks;
  }

  void SetTask(int i, int op, std::string& data, int sock) {
    tasks_[i].op = op;
    tasks_[i].data = &data[0];
    tasks_[i].size = static_cast<int>(data.size());
    tasks_[i].sock = &comm_->socks[sock];
    tasks_[i].offset = 0;
    tasks_[i].result = ncclSuccess;
    tasks_[i].used = 1;
  }
};

TEST_F(NetSocketHelperThreadMicrotest, SocketThread_PendingTasks_DrivesEachToCompletionBeforeStopping) {
  std::string a = Pattern(25), b = Pattern(25);
  SetTask(0, NCCL_SOCKET_SEND, a, 0);
  SetTask(1, NCCL_SOCKET_SEND, b, 1);
  g_sock.chunk = 10;
  res_->stop = 1;

  EXPECT_EQ(persistentSocketThread(res_), nullptr);

  EXPECT_EQ(tasks_[0].offset, 25);
  EXPECT_EQ(tasks_[1].offset, 25);
  EXPECT_EQ(tasks_[0].result, ncclSuccess);
  EXPECT_EQ(Sent(comm_->socks[0]), a);
  EXPECT_EQ(Sent(comm_->socks[1]), b);
}

TEST_F(NetSocketHelperThreadMicrotest, SocketThread_PendingTasksInEveryQueueSlot_AreAllDriven) {
  std::string data[4] = {Pattern(5), Pattern(6), Pattern(7), Pattern(8)};
  for (int i = 0; i < 4; i++) SetTask(i, NCCL_SOCKET_SEND, data[i], i % 2);
  res_->stop = 1;

  EXPECT_EQ(persistentSocketThread(res_), nullptr);

  for (int i = 0; i < 4; i++) EXPECT_EQ(tasks_[i].offset, tasks_[i].size) << i;
  EXPECT_EQ(Sent(comm_->socks[0]), data[0] + data[2]);
  EXPECT_EQ(Sent(comm_->socks[1]), data[1] + data[3]);
}

TEST_F(NetSocketHelperThreadMicrotest, SocketThread_FreeAndFinishedTasks_AreNotProgressed) {
  std::string a = Pattern(10), b = Pattern(10);
  SetTask(0, NCCL_SOCKET_SEND, a, 0);
  tasks_[0].used = 0;
  SetTask(1, NCCL_SOCKET_SEND, b, 1);
  tasks_[1].offset = tasks_[1].size;
  res_->stop = 1;
  ScopedHook progress(g_sockProgress, DefaultSockProgress);

  EXPECT_EQ(persistentSocketThread(res_), nullptr);

  EXPECT_EQ(progress.calls, 0);
}

TEST_F(NetSocketHelperThreadMicrotest, SocketThread_ProgressFails_RecordsErrorAndExitsBeforeLaterTasks) {
  std::string a = Pattern(10), b = Pattern(10);
  SetTask(0, NCCL_SOCKET_SEND, a, 0);
  SetTask(1, NCCL_SOCKET_SEND, b, 1);
  res_->stop = 1;
  ScopedHook progress(g_sockProgress, [](int, ncclSocket*, void*, int, int*, bool*) { return ncclRemoteError; });

  EXPECT_EQ(persistentSocketThread(res_), nullptr);

  EXPECT_EQ(tasks_[0].result, ncclRemoteError);
  EXPECT_EQ(progress.calls, 1);
  EXPECT_EQ(tasks_[1].offset, 0);
}

TEST_F(NetSocketHelperThreadMicrotest, SocketThread_Idle_WakesForNewTaskAndExitsOnStop) {
  std::thread helper(persistentSocketThread, res_);
  std::string data = Pattern(40);
  {
    std::lock_guard<std::mutex> lock(res_->threadMutex);
    SetTask(0, NCCL_SOCKET_SEND, data, 0);
    res_->threadTaskQueue.next = 1;
    res_->threadCond.notify_one();
  }
  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (Sent(comm_->socks[0]).size() < data.size() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  {
    std::lock_guard<std::mutex> lock(res_->threadMutex);
    res_->stop = 1;
    res_->threadCond.notify_one();
  }
  helper.join();

  EXPECT_EQ(Sent(comm_->socks[0]), data);
}

using NetSocketMultiSocketMicrotest = NetSocketMicrotest;

TEST_F(NetSocketMultiSocketMicrotest, Test_LargeSend_DeliversPayloadSplitAcrossDataSockets) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 2);
  std::string data = Pattern(1000);
  void* req = Isend(comm, &data[0], data.size());
  int size = -1;

  EXPECT_EQ(TestUntilDone(req, &size), ncclSuccess);

  EXPECT_EQ(size, 1000);
  EXPECT_EQ(Sent(comm->ctrlSock), SizeFrame(1000));
  EXPECT_EQ(Sent(comm->socks[0]), data.substr(0, 500));
  EXPECT_EQ(Sent(comm->socks[1]), data.substr(500));
  EXPECT_EQ(CloseComm(comm), ncclSuccess);
}

TEST_F(NetSocketMultiSocketMicrotest, Test_LargeRecv_GathersPayloadFromDataSockets) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 2);
  std::string payload = Pattern(1000);
  QueueInbound(comm->ctrlSock, SizeFrame(1000));
  QueueInbound(comm->socks[0], payload.substr(0, 500));
  QueueInbound(comm->socks[1], payload.substr(500));
  std::string buf(1000, '\0');
  void* req = Irecv(comm, &buf[0], buf.size());
  int size = -1;

  EXPECT_EQ(TestUntilDone(req, &size), ncclSuccess);

  EXPECT_EQ(size, 1000);
  EXPECT_EQ(buf, payload);
  EXPECT_EQ(CloseComm(comm), ncclSuccess);
}

TEST_F(NetSocketMultiSocketMicrotest, Test_ConsecutiveRequests_StartOnNextDataSocket) {
  ncclNetSocketComm* comm = MakeComm(2, 2);
  std::string a = Pattern(1000), b = Pattern(2000);
  int size = -1;

  EXPECT_EQ(TestUntilDone(Isend(comm, &a[0], a.size()), &size), ncclSuccess);
  EXPECT_EQ(TestUntilDone(Isend(comm, &b[0], b.size()), &size), ncclSuccess);

  EXPECT_EQ(Sent(comm->socks[0]), a);
  EXPECT_EQ(Sent(comm->socks[1]), b);
  EXPECT_EQ(CloseComm(comm), ncclSuccess);
}

TEST_F(NetSocketMultiSocketMicrotest, Test_DataSocketFails_ReturnsSocketError) {
  SetParam("SOCKET_MIN_TASKSIZE", 16);
  ncclNetSocketComm* comm = MakeComm(2, 1);
  ncclSocket* failing = &comm->socks[1];
  ScopedHook progress(g_sockProgress, [failing](int op, ncclSocket* sock, void* ptr, int size, int* offset,
                                                bool* closed) {
    if (sock == failing) return ncclRemoteError;
    return DefaultSockProgress(op, sock, ptr, size, offset, closed);
  });
  std::string data = Pattern(1000);
  void* req = Isend(comm, &data[0], data.size());

  EXPECT_EQ(TestUntilDone(req, nullptr), ncclRemoteError);

  EXPECT_EQ(CloseComm(comm), ncclSuccess);
}

TEST_F(NetSocketMultiSocketMicrotest, GetTask_FirstTaskForThread_StartsOnlyThatThreadsHelper) {
  ncclNetSocketComm* comm = MakeComm(4, 2);
  std::string data = Pattern(10);
  ncclProfilerInfo pInfo{};
  ncclNetSocketTask* task = nullptr;

  EXPECT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_SEND, &data[0], 10, &task), ncclSuccess);

  EXPECT_TRUE(comm->helperThread[0].joinable());
  EXPECT_FALSE(comm->helperThread[1].joinable());
  EXPECT_EQ(comm->threadResources[0].threadTaskQueue.len, MAX_REQUESTS * 2);
  EXPECT_EQ(comm->threadResources[0].comm, comm);
  EXPECT_EQ(task, &comm->threadResources[0].threadTaskQueue.tasks[0]);
  EXPECT_EQ(CloseComm(comm), ncclSuccess);
}

using NetSocketCloseMicrotest = NetSocketMicrotest;

TEST_F(NetSocketCloseMicrotest, Close_Null_Succeeds) { EXPECT_EQ(ncclNetSocketClose(nullptr), ncclSuccess); }

TEST_F(NetSocketCloseMicrotest, Close_ReadySockets_ClosesControlAndDataSockets) {
  ncclNetSocketComm* comm = MakeComm(2, 1);
  std::vector<int> fds = {comm->ctrlSock.socketDescriptor, comm->socks[0].socketDescriptor,
                          comm->socks[1].socketDescriptor};

  EXPECT_EQ(CloseComm(comm), ncclSuccess);

  EXPECT_EQ(g_sock.closed, fds);
}

TEST_F(NetSocketCloseMicrotest, Close_SocketSlotsBeyondNSocks_AreNotTouched) {
  ncclNetSocketComm* comm = MakeComm(1, 1);
  ReadySocket(&comm->socks[1]);
  std::vector<int> fds = {comm->ctrlSock.socketDescriptor, comm->socks[0].socketDescriptor};

  EXPECT_EQ(CloseComm(comm), ncclSuccess);

  EXPECT_EQ(g_sock.closed, fds);
}

TEST_F(NetSocketCloseMicrotest, Close_EveryThreadSlotInUse_Succeeds) {
  ncclNetSocketComm* comm = MakeComm(MAX_THREADS, MAX_THREADS);
  for (ncclNetSocketThreadResources& res : comm->threadResources) {
    res.threadTaskQueue.tasks = static_cast<ncclNetSocketTask*>(calloc(1, sizeof(ncclNetSocketTask)));
    res.threadTaskQueue.len = 1;
  }

  EXPECT_EQ(CloseComm(comm), ncclSuccess);
}

TEST_F(NetSocketCloseMicrotest, Close_SocketNotReady_SkipsClose) {
  ncclNetSocketComm* comm = MakeComm(2, 1);
  comm->socks[1].state = ncclSocketStateInitialized;
  std::vector<int> fds = {comm->ctrlSock.socketDescriptor, comm->socks[0].socketDescriptor};

  EXPECT_EQ(CloseComm(comm), ncclSuccess);

  EXPECT_EQ(g_sock.closed, fds);
}

TEST_F(NetSocketCloseMicrotest, Close_ReadyCheckFails_PropagatesError) {
  ncclNetSocketComm* comm = MakeComm(0, 0);
  ScopedHook ready(g_sockReady, [](ncclSocket*, int*) { return ncclRemoteError; });

  EXPECT_EQ(CloseComm(comm), ncclRemoteError);

  free(comm->inlineData);
  delete comm;
}

TEST_F(NetSocketCloseMicrotest, Close_RunningHelperThreads_StopsAndJoinsThem) {
  ncclNetSocketComm* comm = MakeComm(2, 2);
  std::string data = Pattern(10);
  ncclProfilerInfo pInfo{};
  ncclNetSocketTask* task = nullptr;
  EXPECT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_SEND, &data[0], 10, &task), ncclSuccess);
  EXPECT_EQ(ncclNetSocketGetTask(comm, &pInfo, NCCL_SOCKET_SEND, &data[0], 10, &task), ncclSuccess);
  EXPECT_TRUE(comm->helperThread[1].joinable());

  EXPECT_EQ(CloseComm(comm), ncclSuccess);
}

}  // namespace
