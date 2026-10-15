/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/proxy.cc. Sockets are socketpair-backed fakes; the IPC socket, shm,
// OS, HIP, profiler and transport layers are seamed in this TU.

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "alloc.h"
#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "socket.h"
#include "shmutils.h"
#include "profiler.h"
#include "transport.h"
#include "cpuset.h"
#include "compiler.h"
#include "os.h"

// --- Logging and params: this binary links no shared fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {
std::mutex g_logMutex;
std::vector<std::pair<ncclDebugLogLevel, std::string>> g_logged;
std::map<std::string, std::string> g_env;
std::map<std::string, int64_t> g_params;

bool Logged(ncclDebugLogLevel level, const char* needle) {
  std::lock_guard<std::mutex> lock(g_logMutex);
  for (const auto& m : g_logged)
    if (m.first == level && m.second.find(needle) != std::string::npos) return true;
  return false;
}
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lock(g_logMutex);
  g_logged.emplace_back(level, buf);
}

const char* ncclGetEnv(const char* name) {
  auto it = g_env.find(name);
  return it == g_env.end() ? nullptr : it->second.c_str();
}

// Never fills the cache, so each test sees its own g_params.
int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t, int64_t*, int8_t*) {
  auto it = g_params.find(env);
  return it == g_params.end() ? deftVal : it->second;
}

// --- Sockets: socketDescriptor is one end of a socketpair (or pipe) the test owns.
namespace {
struct SockFakes {
  ncclResult_t initResult, connectResult, acceptResult, readyResult, progressResult, tryRecvResult;
  int sendFailAt, recvFailAt, getFdFailAt;  // 0-based call index, -1 = never
  int sends, recvs, getFds, inits, connects, closes;
  int progressChunk;  // max bytes one ncclSocketProgress call moves
  std::deque<int> connectFds, acceptFds;
  const union ncclSocketAddress* lastInitAddr;
  uint64_t lastInitMagic;
  int lastInitType;
};
SockFakes g_sock;

bool WaitReadable(int fd) {
  struct pollfd p = {fd, POLLIN, 0};
  return poll(&p, 1, 2000) > 0;
}
}  // namespace

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t* abortFlag, int, int) {
  g_sock.inits++;
  g_sock.lastInitAddr = addr;
  g_sock.lastInitMagic = magic;
  g_sock.lastInitType = type;
  sock->socketDescriptor = -1;
  sock->abortFlag = abortFlag;
  return g_sock.initResult;
}
ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
  g_sock.connects++;
  if (g_sock.connectResult != ncclSuccess) return g_sock.connectResult;
  sock->socketDescriptor = g_sock.connectFds.empty() ? -1 : g_sock.connectFds.front();
  if (!g_sock.connectFds.empty()) g_sock.connectFds.pop_front();
  return ncclSuccess;
}
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  char c;
  if (read(listenSock->socketDescriptor, &c, 1) != 1 || retry) return ncclInternalError;
  if (g_sock.acceptResult != ncclSuccess || g_sock.acceptFds.empty()) return ncclSystemError;
  sock->socketDescriptor = g_sock.acceptFds.front();
  g_sock.acceptFds.pop_front();
  return ncclSuccess;
}
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running) {
  *running = sock->socketDescriptor != -1;
  return g_sock.readyResult;
}
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* fd) {
  if (g_sock.getFds++ == g_sock.getFdFailAt) return ncclInternalError;
  *fd = sock->socketDescriptor;
  return ncclSuccess;
}
ncclResult_t ncclSocketSetFd(ncclSocketDescriptor fd, struct ncclSocket* sock) {
  sock->socketDescriptor = fd;
  return ncclSuccess;
}
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool) {
  g_sock.closes++;
  if (sock->socketDescriptor >= 0) close(sock->socketDescriptor);
  sock->socketDescriptor = -1;
  return ncclSuccess;
}
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  if (g_sock.sends++ == g_sock.sendFailAt) return ncclRemoteError;
  for (int off = 0; off < size;) {
    ssize_t n = write(sock->socketDescriptor, (char*)ptr + off, size - off);
    if (n <= 0) return ncclSystemError;
    off += n;
  }
  return ncclSuccess;
}
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  if (g_sock.recvs++ == g_sock.recvFailAt) return ncclRemoteError;
  for (int off = 0; off < size;) {
    if (!WaitReadable(sock->socketDescriptor)) return ncclSystemError;
    ssize_t n = read(sock->socketDescriptor, (char*)ptr + off, size - off);
    if (n <= 0) return ncclSystemError;
    off += n;
  }
  return ncclSuccess;
}
ncclResult_t ncclSocketProgress(int, struct ncclSocket* sock, void* ptr, int size, int* offset, bool*) {
  if (g_sock.progressResult != ncclSuccess) return g_sock.progressResult;
  ssize_t n = recv(sock->socketDescriptor, (char*)ptr + *offset, std::min(size - *offset, g_sock.progressChunk),
                   MSG_DONTWAIT);
  if (n > 0) *offset += n;
  return ncclSuccess;
}
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, bool* closed, bool) {
  *closed = false;
  if (g_sock.tryRecvResult != ncclSuccess) return g_sock.tryRecvResult;
  ssize_t n = recv(sock->socketDescriptor, ptr, size, MSG_DONTWAIT);
  if (n == 0) *closed = true;
  if (n <= 0) return n == 0 ? ncclSuccess : ncclInProgress;
  return n == size ? ncclSuccess : ncclSocketRecv(sock, (char*)ptr + n, size - n);
}
const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, size_t len, const int) {
  std::snprintf(buf, len, "fake");
  return buf;
}

// --- IPC (UDS) sockets: recorded, never touch the filesystem.
namespace {
struct IpcSent {
  std::vector<char> data;
  ncclIpcFd fd;
  int rank;
  uint64_t hash;
};
struct IpcFakes {
  ncclResult_t initResult, sendResult, recvResult, closeResult, getFdResult, sendFdResult;
  int inits, closes, fdCloses, fdCloseResult;
  int lastInitRank;
  uint64_t lastInitHash;
  std::vector<IpcSent> sent;
  std::vector<char> recvBytes;  // copied into the receiver's buffer
  ncclIpcFd recvFd;
  std::vector<ncclIpcFd> closedFds;
  std::function<void()> onRecv;
};
IpcFakes g_ipc;
}  // namespace

ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket* handle, int rank, uint64_t hash, volatile uint32_t* abortFlag) {
  g_ipc.inits++;
  g_ipc.lastInitRank = rank;
  g_ipc.lastInitHash = hash;
  handle->fd = 1000 + rank;
  handle->abortFlag = abortFlag;
  return g_ipc.initResult;
}
ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket*) {
  g_ipc.closes++;
  return g_ipc.closeResult;
}
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, ncclIpcFd* fd) {
  *fd = handle->fd;
  return g_ipc.getFdResult;
}
ncclResult_t ncclIpcSocketSendFd(struct ncclIpcSocket*, ncclIpcFd fd, int rank, uint64_t hash) {
  g_ipc.sent.push_back({{}, fd, rank, hash});
  return g_ipc.sendFdResult;
}
ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket*, void* hdr, int hdrLen, ncclIpcFd sendFd, int rank, uint64_t hash) {
  g_ipc.sent.push_back({std::vector<char>((char*)hdr, (char*)hdr + hdrLen), sendFd, rank, hash});
  return g_ipc.sendResult;
}
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket*, void* hdr, int hdrLen, ncclIpcFd* recvFd) {
  if (g_ipc.onRecv) g_ipc.onRecv();
  std::memcpy(hdr, g_ipc.recvBytes.data(), std::min<size_t>(hdrLen, g_ipc.recvBytes.size()));
  if (recvFd) *recvFd = g_ipc.recvFd;
  return g_ipc.recvResult;
}
int ncclIpcFdClose(ncclIpcFd fd) {
  g_ipc.fdCloses++;
  g_ipc.closedFds.push_back(fd);
  return g_ipc.fdCloseResult;
}

// --- Shared memory: heap-backed.
namespace {
struct ShmFakes {
  ncclResult_t openResult, closeResult, unlinkResult;
  int opens, closes, unlinks, lastRefcount;
  std::string lastPath;
};
ShmFakes g_shm;
}  // namespace

ncclResult_t ncclShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void**, int refcount,
                         ncclShmHandle_t* handle) {
  g_shm.opens++;
  g_shm.lastRefcount = refcount;
  if (g_shm.openResult != ncclSuccess) return g_shm.openResult;
  if (shmPath[0] == '\0') std::snprintf(shmPath, shmPathSize, "/dev/shm/nccl-AbC123");
  g_shm.lastPath = shmPath;
  *shmPtr = *handle = calloc(1, shmSize);
  return ncclSuccess;
}
ncclResult_t ncclShmClose(ncclShmHandle_t handle) {
  g_shm.closes++;
  free(handle);
  return g_shm.closeResult;
}
ncclResult_t ncclShmUnlink(ncclShmHandle_t) {
  g_shm.unlinks++;
  return g_shm.unlinkResult;
}

// --- OS, HIP, cuMem, thread naming, profiler.
namespace {
int g_mutexCondShared, g_mutexCondUnshared, g_setAffinity, g_threadNames, g_deviceSyncs;
hipError_t g_hipSetDeviceResult, g_hipExportResult, g_hipIpcCloseResult;
std::vector<int> g_hipSetDevices;
std::vector<void*> g_hipIpcClosed;
int g_cuMemEnable, g_cuMemHostEnable;
bool g_proxyDiagEnabled;
int g_profOpStarts, g_profCtrlStarts, g_profCtrlStops;
std::vector<int> g_profCtrlStates;
}  // namespace

void ncclOsSetMutexCondShared(std::mutex&, std::condition_variable&, int* initialized) {
  g_mutexCondShared++;
  *initialized = 1;
}
void ncclOsUnsetMutexCondShared(std::mutex&, std::condition_variable&, int* initialized) {
  g_mutexCondUnshared++;
  *initialized = 0;
}
void ncclOsCpuZero(ncclAffinity& a) { CPU_ZERO(&a); }
int ncclOsCpuCount(const ncclAffinity& a) { return CPU_COUNT(&a); }
void ncclOsCpuSet(ncclAffinity& a, int cpu) { CPU_SET(cpu, &a); }
bool ncclOsCpuIsSet(const ncclAffinity& a, int cpu) { return CPU_ISSET(cpu, &a); }
ncclResult_t ncclOsGetAffinity(ncclAffinity* a) {
  CPU_ZERO(a);
  CPU_SET(0, a);
  return ncclSuccess;
}
ncclResult_t ncclOsSetAffinity(const ncclAffinity&) {
  g_setAffinity++;
  return ncclSuccess;
}
int ncclOsGetCpu() { return 0; }
uint64_t ncclOsGetTid() { return 0; }

hipError_t hipSetDevice(int dev) {
  g_hipSetDevices.push_back(dev);
  return g_hipSetDeviceResult;
}
hipError_t hipDeviceSynchronize(void) {
  g_deviceSyncs++;
  return hipErrorInvalidValue;
}
hipError_t hipGetLastError(void) { return hipSuccess; }
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }
hipError_t hipIpcCloseMemHandle(void* p) {
  g_hipIpcClosed.push_back(p);
  return g_hipIpcCloseResult;
}
hipError_t hipMemExportToShareableHandle(void* shareable, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                                         unsigned long long) {
  *(int*)shareable = 77;
  return g_hipExportResult;
}
int ncclCuMemEnable() { return g_cuMemEnable; }
int ncclCuMemHostEnable() { return g_cuMemHostEnable; }
void ncclSetThreadName(std::thread&, const char*, ...) { g_threadNames++; }

bool ncclProfilerProxyDiagEnabled(void) { return g_proxyDiagEnabled; }
ncclResult_t ncclProfilerStartProxyOpEvent(int, struct ncclProxyArgs*) {
  g_profOpStarts++;
  return ncclSuccess;
}
ncclResult_t ncclProfilerStartProxyCtrlEvent(void*, void** eHandle) {
  g_profCtrlStarts++;
  *eHandle = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclProfilerStopProxyCtrlEvent(void*) {
  g_profCtrlStops++;
  return ncclSuccess;
}
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void*, int, ncclProfilerEventState_t state) {
  g_profCtrlStates.push_back(state);
  return ncclSuccess;
}

// --- Transports: every hook records and returns a scripted result.
namespace {
struct TransportFakes {
  std::vector<std::pair<ncclProxyConnection*, int>> frees;  // (connection, send)
  ncclResult_t freeResult, setupResult, connectResult, sharedInitResult, registerResult, deregisterResult;
  int setupDone, connectDone, registerDone, deregisterDone;
  int setups, connectsCalled, sharedInits, registers, deregisters;
  int lastSharedInitChannels;
};
TransportFakes g_tp;

ncclResult_t FakeProxyFree(ncclProxyConnection* c, ncclProxyState*) {
  g_tp.frees.push_back({c, c->send});
  return g_tp.freeResult;
}
ncclResult_t FakeProxySetup(ncclProxyConnection*, ncclProxyState*, void* req, int reqSize, void* resp, int respSize,
                            int* done) {
  g_tp.setups++;
  if (respSize >= 4 && reqSize >= 4) std::memcpy(resp, req, 4);  // echo
  *done = g_tp.setupDone;
  return g_tp.setupResult;
}
ncclResult_t FakeProxyConnect(ncclProxyConnection*, ncclProxyState*, void*, int, void*, int, int* done) {
  g_tp.connectsCalled++;
  *done = g_tp.connectDone;
  return g_tp.connectResult;
}
ncclResult_t FakeProxySharedInit(ncclProxyConnection*, ncclProxyState*, int nChannels) {
  g_tp.sharedInits++;
  g_tp.lastSharedInitChannels = nChannels;
  return g_tp.sharedInitResult;
}
ncclResult_t FakeProxyRegister(ncclProxyConnection*, ncclProxyState*, void*, int, void*, int, int* done) {
  g_tp.registers++;
  *done = g_tp.registerDone;
  return g_tp.registerResult;
}
ncclResult_t FakeProxyDeregister(ncclProxyConnection*, ncclProxyState*, void*, int, int* done) {
  g_tp.deregisters++;
  *done = g_tp.deregisterDone;
  return g_tp.deregisterResult;
}
std::atomic<int> g_progressCalls;
ncclResult_t FakeProxyProgress(ncclProxyState*, ncclProxyArgs*) {
  g_progressCalls++;
  return ncclSuccess;
}

// Transport 0 has every hook; transport 1 has none (all null).
ncclTransport g_transports[NTRANSPORTS] = {{"full"}, {"empty"}, {"t2"}, {"t3"}};
}  // namespace
struct ncclTransport* ncclTransports[NTRANSPORTS] = {&g_transports[0], &g_transports[1], &g_transports[2],
                                                     &g_transports[3]};

// --- Allocation seams over the unit only: fail the Nth ncclCalloc / ncclRealloc / malloc it makes (0-based).
namespace {
int g_callocCalls, g_callocFailAt = -1, g_reallocCalls, g_reallocFailAt = -1, g_mallocCalls, g_mallocFailAt = -1;
}  // namespace
template <typename... Args>
static ncclResult_t ProxyTestCalloc(const char* file, int line, const char* fn, Args&&... args) {
  if (g_callocCalls++ == g_callocFailAt) return ncclSystemError;
  return ncclCallocDebug(std::forward<Args>(args)..., file, line, fn, true);
}
template <typename T>
static ncclResult_t ProxyTestRealloc(T** ptr, size_t oldNelem, size_t nelem, const char* file, int line,
                                     const char* fn) {
  if (g_reallocCalls++ == g_reallocFailAt) return ncclSystemError;
  return ncclReallocDebug(ptr, oldNelem, nelem, file, line, fn, true);
}
static void* ProxyTestMalloc(size_t n) { return g_mallocCalls++ == g_mallocFailAt ? nullptr : std::malloc(n); }
#undef ncclCalloc
#define ncclCalloc(...) ProxyTestCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)
#undef ncclRealloc
#define ncclRealloc(ptr, o, n) ProxyTestRealloc(ptr, o, n, __FILE__, __LINE__, __func__)
#define malloc(n) ProxyTestMalloc(n)

#include PROXY_CC_PATH

#undef malloc

namespace {

class ProxyFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    g_logged.clear();
    g_env.clear();
    g_params.clear();
    g_sock = SockFakes{};
    g_sock.sendFailAt = g_sock.recvFailAt = g_sock.getFdFailAt = -1;
    g_sock.progressChunk = 1 << 20;
    g_ipc = IpcFakes{};
    g_ipc.recvFd = NCCL_INVALID_IPC_FD;
    g_shm = ShmFakes{};
    g_mutexCondShared = g_mutexCondUnshared = g_setAffinity = g_threadNames = g_deviceSyncs = 0;
    g_hipSetDeviceResult = g_hipExportResult = g_hipIpcCloseResult = hipSuccess;
    g_hipSetDevices.clear();
    g_hipIpcClosed.clear();
    g_cuMemEnable = g_cuMemHostEnable = 0;
    g_proxyDiagEnabled = false;
    g_profOpStarts = g_profCtrlStarts = g_profCtrlStops = 0;
    g_profCtrlStates.clear();
    g_tp = TransportFakes{};
    g_tp.setupDone = g_tp.connectDone = g_tp.registerDone = g_tp.deregisterDone = 1;
    g_progressCalls = 0;
    ncclTransportComm full = {};
    full.proxyFree = FakeProxyFree;
    full.proxySetup = FakeProxySetup;
    full.proxyConnect = FakeProxyConnect;
    full.proxySharedInit = FakeProxySharedInit;
    full.proxyRegister = FakeProxyRegister;
    full.proxyDeregister = FakeProxyDeregister;
    full.proxyProgress = FakeProxyProgress;
    g_transports[0].send = g_transports[0].recv = full;
    g_transports[1].send = g_transports[1].recv = ncclTransportComm{};
    g_callocCalls = g_reallocCalls = g_mallocCalls = 0;
    g_callocFailAt = g_reallocFailAt = g_mallocFailAt = -1;
  }
  void TearDown() override {
    for (int fd : fds_) close(fd);
    free(comm_);
  }

  // Returns {ours, theirs}: the unit gets `ours`, the test reads/writes `theirs`.
  std::pair<int, int> SockPair() {
    int sv[2];
    EXPECT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    fds_.push_back(sv[1]);
    return {sv[0], sv[1]};
  }
  template <typename T>
  static void Put(int fd, const T& v) {
    ASSERT_EQ(write(fd, &v, sizeof(v)), (ssize_t)sizeof(v));
  }
  template <typename T>
  static T Get(int fd) {
    T v{};
    EXPECT_TRUE(WaitReadable(fd));
    EXPECT_EQ(read(fd, &v, sizeof(v)), (ssize_t)sizeof(v));
    return v;
  }

  std::vector<int> fds_;
  ncclComm* comm_ = (ncclComm*)calloc(1, sizeof(ncclComm));
};

// Connection pool

class ProxyConnPoolMicrotest : public ProxyFixture {
 protected:
  void TearDown() override {
    for (int b = 0; b < pool.banks; b++) free(pool.pools[b]);
    free(pool.pools);
    ProxyFixture::TearDown();
  }
  // As ncclProxyService initializes it: the first NewConnection allocates bank 0.
  ncclProxyConnectionPool pool = {nullptr, 0, NCCL_PROXY_CONN_POOL_SIZE};
};

TEST_F(ProxyConnPoolMicrotest, NewConnection_AllocatesBanksOnDemandAndEncodesBankInId) {
  int id = -1;
  ASSERT_EQ(ncclProxyNewConnection(&pool, &id), ncclSuccess);
  EXPECT_EQ(id, 0);
  EXPECT_EQ(pool.banks, 1);
  for (int i = 1; i < NCCL_PROXY_CONN_POOL_SIZE; i++) ASSERT_EQ(ncclProxyNewConnection(&pool, &id), ncclSuccess);
  EXPECT_EQ(id, NCCL_PROXY_CONN_POOL_SIZE - 1);
  EXPECT_EQ(pool.banks, 1);
  ncclProxyConnection* bank0 = pool.pools[0];
  ASSERT_EQ(ncclProxyNewConnection(&pool, &id), ncclSuccess);
  EXPECT_EQ(id, 1 << NCCL_PROXY_CONN_POOL_SIZE_POW2);
  EXPECT_EQ(pool.banks, 2);
  EXPECT_EQ(pool.offset, 1);
  EXPECT_EQ(pool.pools[0], bank0);  // the realloc keeps existing banks
  EXPECT_EQ(pool.pools[1][0].state, connUninitialized);
}

TEST_F(ProxyConnPoolMicrotest, NewConnection_AllocationFailuresPropagate) {
  int id = -1;
  g_callocFailAt = 0;
  EXPECT_EQ(ncclProxyNewConnection(&pool, &id), ncclSystemError);
  EXPECT_EQ(pool.banks, 0);
  g_reallocFailAt = g_reallocCalls;
  EXPECT_EQ(ncclProxyNewConnection(&pool, &id), ncclSystemError);
  EXPECT_EQ(pool.banks, 0);
}

TEST_F(ProxyConnPoolMicrotest, GetConnection_RejectsIdsOutsideTheIssuedRange) {
  ncclProxyConnection* conn = nullptr;
  EXPECT_EQ(ncclProxyGetConnection(&pool, 0, &conn), ncclInvalidArgument);  // no banks
  int id;
  for (int i = 0; i < NCCL_PROXY_CONN_POOL_SIZE + 2; i++) ASSERT_EQ(ncclProxyNewConnection(&pool, &id), ncclSuccess);
  EXPECT_EQ(ncclProxyGetConnection(&pool, -1, &conn), ncclInvalidArgument);
  EXPECT_EQ(ncclProxyGetConnection(&pool, 2 << NCCL_PROXY_CONN_POOL_SIZE_POW2, &conn), ncclInvalidArgument);
  EXPECT_EQ(ncclProxyGetConnection(&pool, id + 1, &conn), ncclInvalidArgument);  // past the last bank's mark
  EXPECT_EQ(conn, nullptr);
  ASSERT_EQ(ncclProxyGetConnection(&pool, id, &conn), ncclSuccess);
  EXPECT_EQ(conn, pool.pools[1] + 1);
  // Full earlier banks accept every offset, even past the last bank's high-water mark.
  ASSERT_EQ(ncclProxyGetConnection(&pool, NCCL_PROXY_CONN_POOL_SIZE - 1, &conn), ncclSuccess);
  EXPECT_EQ(conn, pool.pools[0] + NCCL_PROXY_CONN_POOL_SIZE - 1);
  free(pool.pools[1]);
  pool.pools[1] = nullptr;
  EXPECT_EQ(ncclProxyGetConnection(&pool, id, &conn), ncclInvalidArgument);
}

TEST_F(ProxyConnPoolMicrotest, ValidateConnection_ChecksBoundsAlignmentAndOwner) {
  int id;
  for (int i = 0; i < NCCL_PROXY_CONN_POOL_SIZE + 2; i++) ASSERT_EQ(ncclProxyNewConnection(&pool, &id), ncclSuccess);
  pool.pools[0][5].peerId = 7;
  pool.pools[1][1].peerId = 9;
  EXPECT_EQ(ncclProxyValidateConnection(&pool, &pool.pools[0][5], 7), ncclSuccess);
  EXPECT_EQ(ncclProxyValidateConnection(&pool, &pool.pools[0][NCCL_PROXY_CONN_POOL_SIZE - 1], 0), ncclSuccess);
  EXPECT_EQ(ncclProxyValidateConnection(&pool, &pool.pools[1][1], 9), ncclSuccess);
  EXPECT_EQ(ncclProxyValidateConnection(&pool, &pool.pools[0][5], 8), ncclInvalidArgument);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Invalid proxy connection peer: 7 != 8"));
  EXPECT_EQ(ncclProxyValidateConnection(&pool, (ncclProxyConnection*)((char*)&pool.pools[0][5] + 8), 7),
            ncclInvalidArgument);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "wrong alignment"));
  EXPECT_EQ(ncclProxyValidateConnection(&pool, &pool.pools[1][2], 0), ncclInvalidArgument);  // past offset
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "not found"));
  ncclProxyConnection outside = {};
  EXPECT_EQ(ncclProxyValidateConnection(&pool, &outside, 0), ncclInvalidArgument);
  free(pool.pools[0]);
  pool.pools[0] = nullptr;  // a NULL bank is skipped, not dereferenced
  EXPECT_EQ(ncclProxyValidateConnection(&pool, &pool.pools[1][1], 9), ncclSuccess);
}

TEST_F(ProxyConnPoolMicrotest, FreeConnections_FreesInitializedOnesThroughTheirTransportSide) {
  int id;
  for (int i = 0; i < NCCL_PROXY_CONN_POOL_SIZE + 3; i++) ASSERT_EQ(ncclProxyNewConnection(&pool, &id), ncclSuccess);
  ncclProxyConnection* c = pool.pools[0];
  c[1] = {1, 0};  // send, transport 0
  c[1].state = connInitialized;
  c[2].state = connConnected;  // recv, transport 0
  c[3] = {1, 1};  // send, transport 1: no proxyFree hook
  c[3].state = connConnected;
  c[4] = {0, 1};
  c[4].state = connSetupDone;
  pool.pools[1][2].state = connConnected;  // last bank, within offset
  ncclProxyState state = {};
  ASSERT_EQ(ncclProxyFreeConnections(&pool, &state), ncclSuccess);
  pool = {nullptr, 0, 0};
  ASSERT_EQ(g_tp.frees.size(), 3u);
  EXPECT_EQ(g_tp.frees[0].second, 1);
  EXPECT_EQ(g_tp.frees[1].second, 0);
  EXPECT_EQ(g_tp.frees[2].second, 0);
}

TEST_F(ProxyConnPoolMicrotest, FreeConnections_StopsAtTheFirstTransportError) {
  int id;
  for (int i = 0; i < 3; i++) ASSERT_EQ(ncclProxyNewConnection(&pool, &id), ncclSuccess);
  pool.pools[0][0].state = connConnected;
  pool.pools[0][1].state = connConnected;
  g_tp.freeResult = ncclRemoteError;
  ncclProxyState state = {};
  EXPECT_EQ(ncclProxyFreeConnections(&pool, &state), ncclRemoteError);
  EXPECT_EQ(g_tp.frees.size(), 1u);  // the pool is left allocated; TearDown frees it
}

// Expected-response queue (client side of async RPCs)

class ProxyResponseQueueMicrotest : public ProxyFixture {
 protected:
  void TearDown() override {
    expectedProxyResponseFree(&state);
    ProxyFixture::TearDown();
  }
  std::vector<void*> Ids() {
    std::vector<void*> ids;
    for (auto* e = state.expectedResponses; e; e = e->next) ids.push_back(e->opId);
    return ids;
  }
  static void* Heap(int v) {
    int* p = (int*)std::malloc(sizeof(int));
    *p = v;
    return p;
  }
  ncclProxyState state = {};
  char a, b, c;
};

TEST_F(ProxyResponseQueueMicrotest, Enqueue_AppendsInOrderWithAPendingInternalError) {
  ASSERT_EQ(expectedProxyResponseEnqueue(&state, &a, 4), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseEnqueue(&state, &b, 0), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseEnqueue(&state, &c, 8), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a, &b, &c}));
  auto* e = state.expectedResponses;
  EXPECT_EQ(e->respSize, 4);
  EXPECT_FALSE(e->done);
  EXPECT_EQ(e->res, ncclInternalError);
  EXPECT_NE(e->respBuff, nullptr);
}

TEST_F(ProxyResponseQueueMicrotest, Enqueue_AllocationFailures) {
  g_callocFailAt = 0;
  EXPECT_EQ(expectedProxyResponseEnqueue(&state, &a, 4), ncclSystemError);
  g_mallocFailAt = 0;
  EXPECT_EQ(expectedProxyResponseEnqueue(&state, &a, 4), ncclSystemError);
  EXPECT_EQ(state.expectedResponses, nullptr);
}

TEST_F(ProxyResponseQueueMicrotest, Store_CopiesAndFreesTheBufferAndMarksDone) {
  ASSERT_EQ(expectedProxyResponseEnqueue(&state, &a, 0), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseEnqueue(&state, &b, sizeof(int)), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseStore(&state, &b, Heap(42), sizeof(int), ncclRemoteError), ncclSuccess);
  auto* e = state.expectedResponses->next;
  EXPECT_TRUE(e->done);
  EXPECT_EQ(e->res, ncclRemoteError);
  EXPECT_EQ(*(int*)e->respBuff, 42);
  ASSERT_EQ(expectedProxyResponseStore(&state, &a, nullptr, 0, ncclSuccess), ncclSuccess);
  EXPECT_TRUE(state.expectedResponses->done);
}

TEST_F(ProxyResponseQueueMicrotest, Store_RejectsSizeMismatchRepeatsAndUnknownIds) {
  ASSERT_EQ(expectedProxyResponseEnqueue(&state, &a, sizeof(int)), ncclSuccess);
  void* buf = Heap(1);
  EXPECT_EQ(expectedProxyResponseStore(&state, &a, buf, 2, ncclSuccess), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "got 2, expected 4"));
  EXPECT_FALSE(state.expectedResponses->done);
  ASSERT_EQ(expectedProxyResponseStore(&state, &a, buf, sizeof(int), ncclSuccess), ncclSuccess);
  buf = Heap(2);
  EXPECT_EQ(expectedProxyResponseStore(&state, &a, buf, sizeof(int), ncclSuccess), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "already completed"));
  EXPECT_EQ(expectedProxyResponseStore(&state, &b, buf, sizeof(int), ncclSuccess), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "doesn't match any expected response"));
  free(buf);
}

TEST_F(ProxyResponseQueueMicrotest, Dequeue_ReturnsOnlyCompletedResponsesAndUnlinksThem) {
  for (void* id : {(void*)&a, (void*)&b, (void*)&c})
    ASSERT_EQ(expectedProxyResponseEnqueue(&state, id, sizeof(int)), ncclSuccess);
  int out = 0, found = 7;
  EXPECT_EQ(expectedProxyResponseDequeue(&state, &b, &out, &found), ncclSuccess);
  EXPECT_EQ(found, 0);  // pending
  ASSERT_EQ(expectedProxyResponseStore(&state, &b, Heap(5), sizeof(int), ncclRemoteError), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseStore(&state, &a, Heap(6), sizeof(int), ncclSuccess), ncclSuccess);
  EXPECT_EQ(expectedProxyResponseDequeue(&state, &b, &out, &found), ncclRemoteError);
  EXPECT_EQ(found, 1);
  EXPECT_EQ(out, 5);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a, &c}));
  EXPECT_EQ(expectedProxyResponseDequeue(&state, &a, &out, &found), ncclSuccess);
  EXPECT_EQ(out, 6);
  EXPECT_EQ(Ids(), (std::vector<void*>{&c}));
}

TEST_F(ProxyResponseQueueMicrotest, Remove_UnlinksHeadOrMiddleAndWarnsOnUnknown) {
  for (void* id : {(void*)&a, (void*)&b, (void*)&c})
    ASSERT_EQ(expectedProxyResponseEnqueue(&state, id, 1), ncclSuccess);
  EXPECT_EQ(expectedProxyResponseRemove(&state, &b), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a, &c}));
  EXPECT_EQ(expectedProxyResponseRemove(&state, &a), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&c}));
  EXPECT_EQ(expectedProxyResponseRemove(&state, &a), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Couldn't find opId"));
}

// Service-side async op queue

class ProxyAsyncOpQueueMicrotest : public ProxyFixture {
 protected:
  void TearDown() override {
    while (peer.asyncOps) asyncProxyOpDequeue(&peer, peer.asyncOps);
    ProxyFixture::TearDown();
  }
  ncclProxyAsyncOp* Op(void* opId, bool bufs) {
    auto* op = (ncclProxyAsyncOp*)calloc(1, sizeof(ncclProxyAsyncOp));
    op->opId = opId;
    if (bufs) {
      op->reqBuff = (char*)std::malloc(4);
      op->respBuff = (char*)std::malloc(4);
    }
    return op;
  }
  std::vector<void*> Ids() {
    std::vector<void*> ids;
    for (auto* e = peer.asyncOps; e; e = e->next) ids.push_back(e->opId);
    return ids;
  }
  ncclProxyLocalPeer peer = {};
  char a, b, c;
};

TEST_F(ProxyAsyncOpQueueMicrotest, EnqueueDequeue_KeepsOrderAndFreesBuffers) {
  ASSERT_EQ(asyncProxyOpEnqueue(&peer, Op(&a, true)), ncclSuccess);
  ASSERT_EQ(asyncProxyOpEnqueue(&peer, Op(&b, false)), ncclSuccess);
  ASSERT_EQ(asyncProxyOpEnqueue(&peer, Op(&c, true)), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a, &b, &c}));
  ncclProxyAsyncOp key = {};
  key.opId = &b;
  EXPECT_EQ(asyncProxyOpDequeue(&peer, &key), ncclSuccess);  // matched by opId, not by pointer
  EXPECT_EQ(Ids(), (std::vector<void*>{&a, &c}));
  EXPECT_EQ(asyncProxyOpDequeue(&peer, peer.asyncOps), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&c}));
  key.opId = &a;
  EXPECT_EQ(asyncProxyOpDequeue(&peer, &key), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "dequeue nonexistent async opId"));
}

TEST_F(ProxyAsyncOpQueueMicrotest, Dequeue_NullOpOnAnEmptyQueueWarns) {
  EXPECT_EQ(asyncProxyOpDequeue(&peer, nullptr), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "dequeue null operation"));
}

// Proxy state teardown

class ProxyDestroyMicrotest : public ProxyFixture {};

TEST_F(ProxyDestroyMicrotest, Destroy_FreesSharedArraysAndPendingResponses) {
  ncclSharedResources res = {};
  ncclComm& comm = *comm_;
  comm.sharedRes = &res;
  EXPECT_EQ(ncclProxyDestroy(&comm), ncclSuccess);  // nothing to free
  res.proxyState = new ncclProxyState{};
  res.proxyState->refCount = 1;
  EXPECT_EQ(ncclProxyDestroy(&comm), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "refCount is 1, expected 0"));
  res.proxyState->refCount = 0;
  res.proxyState->peerAddresses = (ncclSocketAddress*)calloc(1, sizeof(ncclSocketAddress));
  res.proxyState->peerAddressesUDS = (uint64_t*)calloc(1, 8);
  res.proxyState->peerSocks = (ncclSocket*)calloc(1, sizeof(ncclSocket));
  res.proxyState->proxyOps = (ncclProxyOps*)calloc(1, sizeof(ncclProxyOps));
  res.proxyState->sharedDevMems = (void**)calloc(1, sizeof(void*));
  ASSERT_EQ(expectedProxyResponseEnqueue(res.proxyState, &res, 16), ncclSuccess);
  EXPECT_EQ(ncclProxyDestroy(&comm), ncclSuccess);
}

// Proxy-op queue (enqueue side): ncclLocalOpAppend / ncclProxyPost / ncclProxyStart / ncclProxySaveOp

class TestRing : public RingAlgorithm {
 public:
  void getNextSendAddr(int, uint8_t**, size_t*, void**) override {}
  void getNextRecvAddr(int, uint8_t**, size_t*, void**) override {}
  int refs() const { return refCount; }
};

// Threads each local rank's MAX_OPS_PER_PEER slice of the pool into its own free list, as proxyProgressInit does.
void InitFreeLists(ncclProxyOpsPool* pool, int nLocal) {
  pool->nextOps = pool->nextOpsEnd = -1;
  for (int r = 0; r < nLocal; r++) {
    pool->freeOps[r] = r * MAX_OPS_PER_PEER;
    for (int i = 0; i < MAX_OPS_PER_PEER - 1; i++) pool->ops[r * MAX_OPS_PER_PEER + i].next = r * MAX_OPS_PER_PEER + i + 1;
    pool->ops[(r + 1) * MAX_OPS_PER_PEER - 1].next = -1;
  }
}

class ProxyOpsFixture : public ProxyFixture {
 protected:
  static constexpr int kRanks = 8;
  struct Saved {
    int peer, send, connIndex, opPeer, nsteps, prevRank, nextRank;
    bool operator==(const Saved& o) const {
      return peer == o.peer && send == o.send && connIndex == o.connIndex && opPeer == o.opPeer &&
             nsteps == o.nsteps;
    }
  };

  void SetUp() override {
    ProxyFixture::SetUp();
    pool_ = (ncclProxyOpsPool*)calloc(1, sizeof(ncclProxyOpsPool));
    InitFreeLists(pool_, 2);
    for (auto& o : ops_) o = {pool_, nullptr, 0, -1, -1, -1};
    ps_.proxyOps = ops_;
    ps_.peerArraySize = 2;
    comm_->proxyState = &ps_;
    comm_->topParentLocalRanks = tpLocal_;
    comm_->nRanks = kRanks;
    comm_->rank = 0;
    comm_->localRanks = 1;
    for (int i = 0; i < kRanks; i++) userRanks_[i] = i;
    ncclChannel& ch = comm_->channels[0];
    ch.peers = peerPtrs_;
    ch.ring.userRanks = userRanks_;
    ch.ring.prev = kRanks - 1;
    ch.ring.next = 1;
    for (int p = 0; p < kRanks; p++) {
      peerPtrs_[p] = &peers_[p];
      for (int c = 0; c < NCCL_MAX_CONNS; c++) {
        for (int send = 0; send < 2; send++) {
          ncclConnector& conn = send ? peers_[p].send[c] : peers_[p].recv[c];
          conn.transportComm = send ? &g_transports[0].send : &g_transports[0].recv;
          conn.proxyConn.proxyProgress = FakeProxyProgress;
          conn.proxyConn.connection = &conns_[p][send][c];
        }
      }
    }
  }
  void TearDown() override {
    free(pool_);
    ProxyFixture::TearDown();
  }

  ncclProxyConnector* Conn(int peer, int send, int c = 0) {
    return send ? &peers_[peer].send[c].proxyConn : &peers_[peer].recv[c].proxyConn;
  }
  // The ops queued (not yet posted) for local proxy `r`, identified by the connection each was saved on.
  std::vector<Saved> Queued(int r = 0) {
    std::vector<Saved> out;
    for (int i = ops_[r].nextOps; i != -1; i = pool_->ops[i].next) {
      const ncclProxyOp& op = pool_->ops[i];
      int idx = (ncclProxyConnection*)op.connection - &conns_[0][0][0];
      out.push_back({idx / (2 * NCCL_MAX_CONNS), idx / NCCL_MAX_CONNS % 2, idx % NCCL_MAX_CONNS, op.peer, op.nsteps,
                     op.prevRank, op.nextRank});
      if (i == ops_[r].nextOpsEnd) break;
    }
    return out;
  }
  static Saved S(int peer, int send, int connIndex, int nsteps = 0) {
    return {peer, send, connIndex, peer, nsteps, 0, 0};
  }
  ncclProxyOp Op(ncclPattern_t pattern, int root = 0) {
    ncclProxyOp op = {};
    op.pattern = pattern;
    op.root = root;
    return op;
  }
  ncclResult_t Save(ncclProxyOp op) { return ncclProxySaveOp(comm_, &op, nullptr); }
  bool Inquire(ncclProxyOp op) {
    bool need = true;
    EXPECT_EQ(ncclProxySaveOp(comm_, &op, &need), ncclSuccess);
    return need;
  }

  ncclProxyOpsPool* pool_ = nullptr;
  ncclProxyOps ops_[2];
  ncclProxyState ps_ = {};
  int tpLocal_[2] = {0, 1};
  int userRanks_[kRanks];
  ncclChannelPeer peers_[kRanks] = {};
  ncclChannelPeer* peerPtrs_[kRanks];
  ncclProxyConnection conns_[kRanks][2][NCCL_MAX_CONNS] = {};
};

class ProxyLocalOpAppendMicrotest : public ProxyOpsFixture {};

TEST_F(ProxyLocalOpAppendMicrotest, NoProxyOpsIsAnInternalError) {
  ps_.proxyOps = nullptr;
  ncclProxyOp op = {};
  EXPECT_EQ(ncclLocalOpAppend(comm_, Conn(1, 1), &op), ncclInternalError);
}

TEST_F(ProxyLocalOpAppendMicrotest, ClaimsOwnFreeListOnceThenChainsOpsInOrder) {
  comm_->localRank = 1;  // our ops live in the target pool's rank-1 slice
  Conn(2, 1)->tpLocalRank = 0;
  TestRing ring;
  ncclProxyOp op = {};
  op.opCount = 5;
  op.ringAlgo = &ring;
  op.next = 1234;
  ASSERT_EQ(ncclLocalOpAppend(comm_, Conn(2, 1), &op), ncclSuccess);
  const int first = MAX_OPS_PER_PEER;
  EXPECT_EQ(pool_->freeOps[1], -1);  // the whole list moved to the client side
  EXPECT_EQ(pool_->freeOps[0], 0);
  EXPECT_EQ(ops_[0].freeOp, first + 1);
  EXPECT_EQ(ops_[0].nextOps, first);
  EXPECT_EQ(ops_[0].nextOpsEnd, first);
  EXPECT_EQ(ops_[0].count, 1);
  EXPECT_EQ(pool_->ops[first].opCount, 5u);
  EXPECT_EQ(pool_->ops[first].next, -1);
  EXPECT_EQ(pool_->ops[first].connection, Conn(2, 1)->connection);
  EXPECT_EQ(ring.refs(), 1);
  op.ringAlgo = nullptr;
  ASSERT_EQ(ncclLocalOpAppend(comm_, Conn(2, 1), &op), ncclSuccess);
  EXPECT_EQ(ops_[0].freeOp, first + 2);
  EXPECT_EQ(ops_[0].nextOps, first);
  EXPECT_EQ(ops_[0].nextOpsEnd, first + 1);
  EXPECT_EQ(pool_->ops[first].next, first + 1);
  EXPECT_EQ(ops_[0].count, 2);
  EXPECT_EQ(ops_[1].nextOps, -1);  // other targets untouched
}

TEST_F(ProxyLocalOpAppendMicrotest, WaitsUntilTheProxyReturnsOps) {
  pool_->freeOps[0] = -1;
  std::thread proxy([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    COMPILER_ATOMIC_STORE(&pool_->freeOps[0], 7, std::memory_order_release);
  });
  ncclProxyOp op = {};
  ASSERT_EQ(ncclLocalOpAppend(comm_, Conn(1, 1), &op), ncclSuccess);
  proxy.join();
  EXPECT_EQ(ops_[0].nextOps, 7);
  EXPECT_EQ(ops_[0].freeOp, 8);
}

TEST_F(ProxyLocalOpAppendMicrotest, FullQueuePostsAllButTheTrailingOpCount) {
  ncclProxyOp op = {};
  for (int i = 0; i < MAX_OPS_PER_PEER; i++) {
    op.opCount = i < MAX_OPS_PER_PEER - 3 ? i / 2 : 1000000;  // the last 3 ops share one opCount
    ASSERT_EQ(ncclLocalOpAppend(comm_, Conn(1, 1), &op), ncclSuccess);
  }
  EXPECT_EQ(pool_->nextOps, 0);
  EXPECT_EQ(pool_->nextOpsEnd, MAX_OPS_PER_PEER - 4);
  EXPECT_EQ(pool_->ops[MAX_OPS_PER_PEER - 4].next, -1);
  EXPECT_EQ(ops_[0].count, 3);
  EXPECT_EQ(ops_[0].nextOps, MAX_OPS_PER_PEER - 3);
  EXPECT_EQ(ops_[0].nextOpsEnd, MAX_OPS_PER_PEER - 1);
}

TEST_F(ProxyLocalOpAppendMicrotest, FullQueueOfASingleOpCountCannotBePosted) {
  ncclProxyOp op = {};
  op.opCount = 9;
  for (int i = 0; i < MAX_OPS_PER_PEER - 1; i++) ASSERT_EQ(ncclLocalOpAppend(comm_, Conn(1, 1), &op), ncclSuccess);
  EXPECT_EQ(ncclLocalOpAppend(comm_, Conn(1, 1), &op), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Unable to post incomplete proxy op chain"));
  EXPECT_EQ(pool_->nextOps, -1);
}

class ProxyPostMicrotest : public ProxyOpsFixture {};

TEST_F(ProxyPostMicrotest, Post_StartsOrExtendsThePoolChain) {
  ASSERT_EQ(ncclProxyPost(pool_, 3, 4), ncclSuccess);
  EXPECT_EQ(pool_->nextOps, 3);
  EXPECT_EQ(pool_->nextOpsEnd, 4);
  ASSERT_EQ(ncclProxyPost(pool_, 10, 12), ncclSuccess);
  EXPECT_EQ(pool_->nextOps, 3);
  EXPECT_EQ(pool_->ops[4].next, 10);
  EXPECT_EQ(pool_->nextOpsEnd, 12);
}

TEST_F(ProxyPostMicrotest, Start_PostsEveryPendingPeerQueueAndResetsIt) {
  ps_.proxyOps = nullptr;
  EXPECT_EQ(ncclProxyStart(comm_), ncclSuccess);
  ps_.proxyOps = ops_;
  ncclProxyOp op = {};
  ASSERT_EQ(ncclLocalOpAppend(comm_, Conn(1, 1), &op), ncclSuccess);
  ASSERT_EQ(ncclLocalOpAppend(comm_, Conn(1, 1), &op), ncclSuccess);
  ops_[1].pool = nullptr;
  ops_[1].nextOps = 3;  // no pool: skipped
  ASSERT_EQ(ncclProxyStart(comm_), ncclSuccess);
  EXPECT_EQ(pool_->nextOps, 0);
  EXPECT_EQ(pool_->nextOpsEnd, 1);
  EXPECT_EQ(ops_[0].nextOps, -1);
  EXPECT_EQ(ops_[0].nextOpsEnd, -1);
  EXPECT_EQ(ops_[0].count, 0);
  EXPECT_EQ(ops_[1].nextOps, 3);
}

class ProxySaveOpMicrotest : public ProxyOpsFixture {};

TEST_F(ProxySaveOpMicrotest, SaveProxy_SkipsAbsentPeersAndConnectionsWithoutProgress) {
  ncclProxyOp op = {};
  bool need = false;
  EXPECT_EQ(SaveProxy(comm_, &comm_->channels[0], proxySend, -1, &op, 0, &need), ncclSuccess);
  Conn(3, 0)->proxyProgress = nullptr;
  EXPECT_EQ(SaveProxy(comm_, &comm_->channels[0], proxyRecv, 3, &op, 0, &need), ncclSuccess);
  EXPECT_FALSE(need);
  EXPECT_EQ(SaveProxy(comm_, &comm_->channels[0], proxySend, 3, &op, 0, &need), ncclSuccess);
  EXPECT_TRUE(need);
  EXPECT_TRUE(Queued().empty());
  peers_[4].send[1].transportComm = nullptr;
  comm_->channels[0].id = 0;
  EXPECT_EQ(SaveProxy(comm_, &comm_->channels[0], proxySend, 4, &op, 1, nullptr), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Rank 0 has no transport for send peer 4 on channel 0/1"));
  peers_[4].recv[0].transportComm = nullptr;
  EXPECT_EQ(SaveProxy(comm_, &comm_->channels[0], proxyRecv, 4, &op, 0, nullptr), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "no transport for recv peer 4"));
}

TEST_F(ProxySaveOpMicrotest, Ring_SavesRecvFromPrevThenSendToNext) {
  comm_->commHash = 0xabc;
  ncclProxyOp op = Op(ncclPatternRingTwice);
  op.connIndex = 2;
  op.nsteps = 6;
  ASSERT_EQ(ncclProxySaveOp(comm_, &op, nullptr), ncclSuccess);
  EXPECT_EQ(op.commHash, 0xabcu);
  auto q = Queued();
  ASSERT_EQ(q, (std::vector<Saved>{S(kRanks - 1, 0, 2, 6), S(1, 1, 2, 6)}));
  EXPECT_EQ(q[0].prevRank, kRanks - 1);
  EXPECT_EQ(q[0].nextRank, 1);
  EXPECT_TRUE(Inquire(Op(ncclPatternRing)));
}

TEST_F(ProxySaveOpMicrotest, Pipelines_SkipTheSideFacingTheRoot) {
  // Broadcast: the root never receives; the rank before the root never sends.
  ASSERT_EQ(Save(Op(ncclPatternPipelineFrom, /*root=*/0)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(1, 1, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternPipelineFrom, /*root=*/1)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(kRanks - 1, 0, 0)}));
  ops_[0].nextOps = -1;
  // Reduce: the root never sends; the rank after the root never receives.
  ASSERT_EQ(Save(Op(ncclPatternPipelineTo, /*root=*/0)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(kRanks - 1, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternPipelineTo, /*root=*/kRanks - 1)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(1, 1, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternPipelineTo, /*root=*/3)), ncclSuccess);
  EXPECT_EQ(Queued().size(), 2u);
}

TEST_F(ProxySaveOpMicrotest, AllGatherVRing_TakesStepsFromSlicesAndSkipsEmptySides) {
  ncclProxyOp op = Op(ncclPatternRing);
  op.coll = ncclFuncAllGatherV;
  op.specifics.bcast.recvSlices = 3;
  op.specifics.bcast.sendSlices = 0;
  ASSERT_EQ(Save(op), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(kRanks - 1, 0, 0, 3)}));
  ops_[0].nextOps = -1;
  op.specifics.bcast.recvSlices = 0;
  op.specifics.bcast.sendSlices = 4;
  ASSERT_EQ(Save(op), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(1, 1, 0, 4)}));
  op.specifics.bcast.sendSlices = 0;
  EXPECT_FALSE(Inquire(op));
  op.pattern = ncclPatternRingTwice;  // only plain Ring takes the slice counts
  EXPECT_TRUE(Inquire(op));
}

TEST_F(ProxySaveOpMicrotest, Trees_SaveUpAndDownPeersPerDirection) {
  ncclTree& t = comm_->channels[0].tree;
  t.up = 2;
  t.down[0] = 3;
  t.down[1] = -1;
  t.down[2] = 5;
  ASSERT_EQ(Save(Op(ncclPatternTreeUp)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 0, 0), S(5, 0, 0), S(2, 1, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternTreeDown)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 1, 0), S(5, 1, 0), S(2, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternTreeUpDown)), ncclSuccess);
  EXPECT_EQ(Queued().size(), 6u);
  t.up = -1;
  t.down[0] = t.down[2] = -1;
  EXPECT_FALSE(Inquire(Op(ncclPatternTreeUpDown)));
}

TEST_F(ProxySaveOpMicrotest, CollnetAndNvls_SaveSendOnConn1AndRecvOnConn0) {
  comm_->channels[0].collnetChain.up = 2;
  comm_->channels[0].collnetDirect.out = 3;
  comm_->channels[0].nvls.out = 4;
  ASSERT_EQ(Save(Op(ncclPatternCollnetChain)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(2, 1, 1), S(2, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternCollnetDirect)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 1, 1), S(3, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternNvls)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(4, 1, 1), S(4, 0, 0)}));
}

TEST_F(ProxySaveOpMicrotest, NvlsTree_SavesBothTreeDownPeersAndTheTreeUpPeer) {
  ncclNvls& n = comm_->channels[0].nvls;
  n.treeDown[0] = 7;  // index 0 is never a proxy peer
  n.treeDown[1] = 2;
  n.treeDown[2] = 3;
  n.treeUp = 4;
  ASSERT_EQ(Save(Op(ncclPatternNvlsTree)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(2, 0, 0), S(3, 0, 0), S(4, 1, 0), S(2, 1, 0), S(3, 1, 0), S(4, 0, 0)}));
}

TEST_F(ProxySaveOpMicrotest, SendRecv_SaveTheRootPeerUnlessItIsSelf) {
  ncclProxyOp op = Op(ncclPatternSend, 3);
  op.connIndex = 1;
  ASSERT_EQ(Save(op), ncclSuccess);
  op = Op(ncclPatternRecv, 5);
  ASSERT_EQ(Save(op), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 1, 1), S(5, 0, 0)}));
  EXPECT_FALSE(Inquire(Op(ncclPatternSend, 0)));
  EXPECT_FALSE(Inquire(Op((ncclPattern_t)99)));  // unknown patterns need no proxy
}

TEST_F(ProxySaveOpMicrotest, Errors_FromSaveProxyPropagate) {
  peers_[kRanks - 1].recv[0].transportComm = nullptr;
  EXPECT_EQ(Save(Op(ncclPatternRing)), ncclInternalError);
  peers_[kRanks - 1].recv[0].transportComm = &g_transports[0].recv;
  peers_[1].send[0].transportComm = nullptr;
  EXPECT_EQ(Save(Op(ncclPatternRing)), ncclInternalError);
  comm_->channels[0].tree = {0, 1, {-1, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeUp)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternTreeDown)), ncclSuccess);  // recv from up=1 is fine; no down peers
  comm_->channels[0].tree = {0, -1, {1, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeDown)), ncclInternalError);
  peers_[2].recv[0].transportComm = nullptr;
  comm_->channels[0].tree = {0, 3, {2, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeUp)), ncclInternalError);
  comm_->channels[0].tree = {0, 2, {-1, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeDown)), ncclInternalError);
  peers_[1].send[1].transportComm = nullptr;  // fails the send save; peer 2 then fails the recv save
  comm_->channels[0].collnetChain.up = comm_->channels[0].collnetDirect.out = comm_->channels[0].nvls.out = 1;
  EXPECT_EQ(Save(Op(ncclPatternCollnetChain)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternCollnetDirect)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternNvls)), ncclInternalError);
  comm_->channels[0].collnetChain.up = comm_->channels[0].collnetDirect.out = comm_->channels[0].nvls.out = 2;
  EXPECT_EQ(Save(Op(ncclPatternCollnetChain)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternCollnetDirect)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternNvls)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternSend, 1)), ncclInternalError);
}

TEST_F(ProxySaveOpMicrotest, NvlsTree_ErrorsFromEachSavePropagate) {
  ncclNvls& n = comm_->channels[0].nvls;
  n.treeDown[1] = n.treeDown[2] = n.treeUp = -1;
  for (int which = 0; which < 6; which++) {
    int* slot[] = {&n.treeDown[1], &n.treeDown[2], &n.treeUp};
    int dir[] = {0, 0, 1, 1, 1, 0};  // send side of each SaveProxy in order
    *slot[which % 3] = 1;
    (dir[which] ? peers_[1].send[0] : peers_[1].recv[0]).transportComm = nullptr;
    EXPECT_EQ(Save(Op(ncclPatternNvlsTree)), ncclInternalError) << which;
    peers_[1].send[0].transportComm = &g_transports[0].send;
    peers_[1].recv[0].transportComm = &g_transports[0].recv;
    *slot[which % 3] = -1;
  }
}

class ProxySavePatMicrotest : public ProxyOpsFixture {
 protected:
  void SetUp() override {
    ProxyOpsFixture::SetUp();
    comm_->nRanks = 4;
    comm_->nNodes = 4;
    comm_->node = 0;
    for (int i = 0; i < kRanks; i++) dense_[i] = kRanks - 1 - i;
  }
  std::vector<std::pair<int, int>> Peers() {  // (send, peer)
    std::vector<std::pair<int, int>> out;
    for (const Saved& s : Queued()) {
      EXPECT_GT(s.nsteps, 0);
      EXPECT_EQ(s.connIndex, 0);
      out.push_back({s.send, s.peer});
    }
    return out;
  }
  ncclProxyOp PatOp(ncclPattern_t p) {
    ncclProxyOp op = Op(p);
    op.nbytes = 4 << 20;
    op.chunkSize = 1 << 16;
    return op;
  }
  int dense_[kRanks];
};

TEST_F(ProxySavePatMicrotest, PatUp_SendsToHigherNodesAndReceivesFromLowerOnes) {
  ASSERT_EQ(Save(PatOp(ncclPatternPatUp)), ncclSuccess);
  // Dimension i pairs node+2^i (send) with node-2^i (recv).
  EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 1}, {0, 3}, {1, 2}, {0, 2}}));
}

TEST_F(ProxySavePatMicrotest, PatDown_MirrorsPatUpUnlessQpsAreShared) {
  ASSERT_EQ(Save(PatOp(ncclPatternPatDown)), ncclSuccess);
  auto own = Peers();
  EXPECT_EQ(own, (std::vector<std::pair<int, int>>{{1, 3}, {0, 1}, {1, 2}, {0, 2}}));
  ops_[0].nextOps = -1;
  comm_->patSharedQps = true;
  ASSERT_EQ(Save(PatOp(ncclPatternPatDown)), ncclSuccess);
  EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 1}, {0, 3}, {1, 2}, {0, 2}}));
}

TEST_F(ProxySavePatMicrotest, Pat_MapsDensePeersThroughHeadRankAndDenseToUserRank) {
  comm_->nRanks = 8;
  comm_->nNodes = 2;
  comm_->localRanks = 4;
  comm_->channels[0].nvls.headRank = 1;
  comm_->denseToUserRank = dense_;
  ASSERT_EQ(Save(PatOp(ncclPatternPatUp)), ncclSuccess);
  // dense peer = node 1 * 4 local ranks + head rank 1 = 5, mapped to user rank 8-1-5 = 2.
  EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 2}, {0, 2}}));
  ops_[0].nextOps = -1;
  comm_->isOneRPN = true;
  ASSERT_EQ(Save(PatOp(ncclPatternPatDown)), ncclSuccess);
  EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 2}, {0, 2}}));
}

TEST_F(ProxySavePatMicrotest, Pat_AllocationAndSaveErrorsPropagate) {
  for (ncclPattern_t p : {ncclPatternPatUp, ncclPatternPatDown}) {
    for (int fail = 0; fail < 2; fail++) {
      g_callocCalls = 0;
      g_callocFailAt = fail;
      EXPECT_EQ(Save(PatOp(p)), ncclSystemError);
    }
    g_callocFailAt = -1;
    peers_[1].send[0].transportComm = nullptr;
    peers_[3].send[0].transportComm = nullptr;
    EXPECT_EQ(Save(PatOp(p)), ncclInternalError);
    peers_[1].send[0].transportComm = peers_[3].send[0].transportComm = &g_transports[0].send;
    peers_[1].recv[0].transportComm = peers_[3].recv[0].transportComm = nullptr;
    EXPECT_EQ(Save(PatOp(p)), ncclInternalError);
    peers_[1].recv[0].transportComm = peers_[3].recv[0].transportComm = &g_transports[0].recv;
  }
}

}  // namespace
