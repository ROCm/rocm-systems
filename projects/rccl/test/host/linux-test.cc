/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/os/linux.cc on real sockets, files, shared memory and dlopen. The libc calls
// a test cannot make fail or vary for real go through lx:: hooks that default to the real call.

#include <gtest/gtest.h>

#include "hip/hip_runtime.h"
#include "os.h"
#include "checks.h"
#include "comm.h"
#include "crypt.h"
#include "param.h"
#include "socket.h"
#include "utils.h"

#include <arpa/inet.h>
#include <cctype>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <set>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "linux-test-common.h"

// --- Logging, env and params: this binary links no shared fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
std::string g_linuxLog;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) level = NCCL_LOG_INFO;
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  const char* tag = level == NCCL_LOG_WARN ? "WARN" : level == NCCL_LOG_ATTN ? "ATTN" : "INFO";
  g_linuxLog += std::string(tag) + " " + buf + "\n";
}

const char* ncclGetEnv(const char* name) { return std::getenv(name); }

namespace {
std::vector<std::pair<int64_t*, int64_t>> g_paramCaches;
}

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  const char* str = std::getenv(env);
  int64_t value = (str && *str) ? std::strtoll(str, nullptr, 0) : deftVal;
  g_paramCaches.push_back({cache, uninitialized});
  *cache = value;
  if (noCache) *noCache = 0;
  return value;
}

void LinuxResetParams() {
  for (auto& c : g_paramCaches) *c.first = c.second;
  g_paramCaches.clear();
}

// Defined by init.cc in production.
int g_setCpuStackSize = 0;
int ncclParamSetCpuStackSize() { return g_setCpuStackSize; }

// --- HIP: only the SHM device-mapping path calls into it.
namespace lx {
std::function<hipError_t(hipStreamCaptureMode*)> hipExchangeCaptureMode;
std::function<hipError_t(void*, size_t, unsigned int)> hipHostRegister;
std::function<hipError_t(void**, void*, unsigned int)> hipHostGetDevicePointer;
std::function<hipError_t(void*)> hipHostUnregister;
}  // namespace lx

hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) { return lx::hipExchangeCaptureMode(mode); }
hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags) { return lx::hipHostRegister(ptr, size, flags); }
hipError_t hipHostGetDevicePointer(void** dptr, void* hptr, unsigned int flags) {
  return lx::hipHostGetDevicePointer(dptr, hptr, flags);
}
hipError_t hipHostUnregister(void* ptr) { return lx::hipHostUnregister(ptr); }
const char* hipGetErrorString(hipError_t) { return "hip error"; }
hipError_t hipGetLastError() { return hipSuccess; }

// --- Libc hooks, scoped to linux.cc by the macros below.
namespace lx {
#define LX_HOOKS(X)                                                                                  \
  X(accept) X(poll) X(connect) X(getsockopt) X(setsockopt) X(getrlimit) X(setrlimit) X(getifaddrs)   \
  X(freeifaddrs) X(sched_getaffinity) X(dlopen) X(dlerror) X(realpath) X(fopen) X(opendir) X(calloc) \
  X(mkstemp) X(fallocate) X(mmap) X(munmap) X(unlink)
#define LX_DECLARE(fn) Hook<decltype(&::fn)>::type fn = ::fn;
LX_HOOKS(LX_DECLARE)
#define LX_RESET(fn) fn = ::fn;
void Reset() {
  LX_HOOKS(LX_RESET)
  hipExchangeCaptureMode = [](hipStreamCaptureMode*) { return hipSuccess; };
  hipHostRegister = [](void*, size_t, unsigned int) { return hipSuccess; };
  hipHostGetDevicePointer = [](void** d, void* h, unsigned int) {
    *d = h;
    return hipSuccess;
  };
  hipHostUnregister = [](void*) { return hipSuccess; };
}
}  // namespace lx

#define accept(...) lx::accept(__VA_ARGS__)
#define poll(...) lx::poll(__VA_ARGS__)
#define connect(...) lx::connect(__VA_ARGS__)
#define getsockopt(...) lx::getsockopt(__VA_ARGS__)
#define setsockopt(...) lx::setsockopt(__VA_ARGS__)
#define getrlimit(...) lx::getrlimit(__VA_ARGS__)
#define setrlimit(...) lx::setrlimit(__VA_ARGS__)
#define getifaddrs(...) lx::getifaddrs(__VA_ARGS__)
#define freeifaddrs(...) lx::freeifaddrs(__VA_ARGS__)
#define sched_getaffinity(...) lx::sched_getaffinity(__VA_ARGS__)
#define dlopen(...) lx::dlopen(__VA_ARGS__)
#define dlerror(...) lx::dlerror(__VA_ARGS__)
#define realpath(...) lx::realpath(__VA_ARGS__)
#define fopen(...) lx::fopen(__VA_ARGS__)
#define opendir(...) lx::opendir(__VA_ARGS__)
#define calloc(...) lx::calloc(__VA_ARGS__)
#define mkstemp(...) lx::mkstemp(__VA_ARGS__)
#define fallocate(...) lx::fallocate(__VA_ARGS__)
#define mmap(...) lx::mmap(__VA_ARGS__)
#define munmap(...) lx::munmap(__VA_ARGS__)
#define unlink(...) lx::unlink(__VA_ARGS__)
#include LINUX_CC_PATH
#undef accept
#undef poll
#undef connect
#undef getsockopt
#undef setsockopt
#undef getrlimit
#undef setrlimit
#undef getifaddrs
#undef freeifaddrs
#undef sched_getaffinity
#undef dlopen
#undef dlerror
#undef realpath
#undef fopen
#undef opendir
#undef calloc
#undef mkstemp
#undef fallocate
#undef mmap
#undef munmap
#undef unlink

namespace {

bool FdOpen(int fd) { return fcntl(fd, F_GETFD) != -1; }
bool Exists(const std::string& path) { return access(path.c_str(), F_OK) == 0; }

class LinuxMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    lx::Reset();
    g_linuxLog.clear();
    g_setCpuStackSize = 0;
  }
  void TearDown() override { lx::Reset(); }
  LinuxScopedEnv retrySleep_{"NCCL_SOCKET_RETRY_SLEEP_MSEC", "0"};
};

// Dynamic loading

using LinuxDlMicrotest = LinuxMicrotest;

TEST_F(LinuxDlMicrotest, Dlopen_LoadsLocallyAndResolvesASymbol) {
  int seenFlags = -1;
  lx::dlopen = [&](const char* f, int flags) {
    seenFlags = flags;
    return ::dlopen(f, flags);
  };
  ncclOsLibraryHandle h = ncclOsDlopen("libc.so.6");
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(seenFlags, RTLD_NOW | RTLD_LOCAL);
  auto pid = reinterpret_cast<pid_t (*)()>(ncclOsDlsym(h, "getpid"));
  ASSERT_NE(pid, nullptr);
  EXPECT_EQ(pid(), getpid());
  ncclOsDlclose(h);
  ncclOsDlclose(nullptr);
}

TEST_F(LinuxDlMicrotest, Dlopen_ModeSelectsNowOrLazy) {
  int seenFlags = -1;
  lx::dlopen = [&](const char* f, int flags) {
    seenFlags = flags;
    return ::dlopen(f, flags);
  };
  ncclOsLibraryHandle h = ncclOsDlopen("libc.so.6", NCCL_OS_DL_NOW);
  EXPECT_EQ(seenFlags, RTLD_NOW);
  ncclOsDlclose(h);
  h = ncclOsDlopen("libc.so.6", NCCL_OS_DL_LAZY);
  EXPECT_EQ(seenFlags, RTLD_LAZY);
  ncclOsDlclose(h);
}

TEST_F(LinuxDlMicrotest, Failures_KeepTheLoaderMessageForDlerror) {
  EXPECT_EQ(ncclOsDlopen("libnccl-no-such-lib.so"), nullptr);
  EXPECT_NE(std::string(ncclOsDlerror()).find("libnccl-no-such-lib.so"), std::string::npos);

  EXPECT_EQ(ncclOsDlopen("libnccl-no-such-lib-2.so", NCCL_OS_DL_NOW), nullptr);
  EXPECT_NE(std::string(ncclOsDlerror()).find("libnccl-no-such-lib-2.so"), std::string::npos);

  ncclOsLibraryHandle h = ncclOsDlopen("libc.so.6");
  EXPECT_NE(std::string(ncclOsDlerror()).find("libnccl-no-such-lib-2.so"), std::string::npos);  // success keeps it
  EXPECT_EQ(ncclOsDlsym(h, "ncclNoSuchSymbol"), nullptr);
  EXPECT_NE(std::string(ncclOsDlerror()).find("ncclNoSuchSymbol"), std::string::npos);
  ncclOsDlclose(h);

  lx::dlerror = []() -> char* { return nullptr; };
  EXPECT_EQ(ncclOsDlopen("libnccl-no-such-lib.so"), nullptr);
  EXPECT_STREQ(ncclOsDlerror(), "");
}

TEST_F(LinuxDlMicrotest, Dlerror_IsPerThread) {
  EXPECT_EQ(ncclOsDlopen("libnccl-no-such-lib.so"), nullptr);
  std::string other = "unset";
  std::thread([&] { other = ncclOsDlerror(); }).join();
  EXPECT_EQ(other, "");
  EXPECT_STRNE(ncclOsDlerror(), "");
}

TEST_F(LinuxDlMicrotest, NvmlOpen_LoadsLibnvidiaMl) {
  std::string seen;
  lx::dlopen = [&](const char* f, int) {
    seen = f;
    return reinterpret_cast<void*>(0x1);
  };
  ncclOsLibraryHandle h = nullptr;
  ASSERT_EQ(ncclOsNvmlOpen(&h), ncclSuccess);
  EXPECT_EQ(seen, "libnvidia-ml.so.1");
  EXPECT_EQ(h, reinterpret_cast<void*>(0x1));
  EXPECT_TRUE(LinuxLogged("INFO Loaded NVML from libnvidia-ml.so.1"));
}

TEST_F(LinuxDlMicrotest, NvmlOpen_MissingLibraryIsASystemError) {
  lx::dlopen = [](const char*, int) -> void* { return nullptr; };
  lx::dlerror = []() { return const_cast<char*>("not here"); };
  ncclOsLibraryHandle h = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(ncclOsNvmlOpen(&h), ncclSystemError);
  EXPECT_EQ(h, nullptr);
  EXPECT_TRUE(LinuxLogged("WARN Failed to open libnvidia-ml.so.1: not here"));
}

// Process and memory helpers

using LinuxProcessMicrotest = LinuxMicrotest;

TEST_F(LinuxProcessMicrotest, PidTidAndPageSize) {
  EXPECT_EQ(ncclOsGetPid(), (uint64_t)getpid());
  EXPECT_EQ(ncclOsGetTid(), (uint64_t)syscall(SYS_gettid));
  uint64_t otherTid = 0;
  std::thread([&] { otherTid = ncclOsGetTid(); }).join();
  EXPECT_NE(otherTid, ncclOsGetTid());
  EXPECT_EQ(ncclOsGetPageSize(), (size_t)sysconf(_SC_PAGESIZE));
  EXPECT_EQ(ncclOsGetCommMempoolMaxSize(), 0u);  // no cap
}

TEST_F(LinuxProcessMicrotest, LocaltimeFillsTheCallersBuffer) {
  time_t t = 86400 * 365;
  std::tm buf{}, expected{};
  EXPECT_EQ(ncclOsLocaltime(&t, &buf), &buf);
  localtime_r(&t, &expected);
  EXPECT_EQ(buf.tm_year, expected.tm_year);
  EXPECT_EQ(buf.tm_yday, expected.tm_yday);
  EXPECT_EQ(buf.tm_hour, expected.tm_hour);
}

TEST_F(LinuxProcessMicrotest, AlignedAllocHonoursTheAlignment) {
  void* p = ncclOsAlignedAlloc(4096, 8192);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 4096, 0u);
  std::memset(p, 0xab, 8192);
  ncclOsAlignedFree(p);
}

TEST_F(LinuxProcessMicrotest, SetEnvNeverOverwrites) {
  LinuxScopedEnv env("NCCL_LINUX_TEST_VAR", nullptr);
  ncclOsSetEnv("NCCL_LINUX_TEST_VAR", "first");
  EXPECT_STREQ(getenv("NCCL_LINUX_TEST_VAR"), "first");
  ncclOsSetEnv("NCCL_LINUX_TEST_VAR", "second");
  EXPECT_STREQ(getenv("NCCL_LINUX_TEST_VAR"), "first");
}

TEST_F(LinuxProcessMicrotest, StrSepAndRealpath) {
  char text[] = "a:b";
  char* cursor = text;
  EXPECT_STREQ(ncclOsStrSep(&cursor, ":"), "a");
  EXPECT_STREQ(ncclOsStrSep(&cursor, ":"), "b");
  EXPECT_EQ(cursor, nullptr);

  char resolved[PATH_MAX], expected[PATH_MAX];
  ASSERT_NE(::realpath("/tmp", expected), nullptr);
  EXPECT_EQ(ncclOsRealpath("/tmp/../tmp/.", resolved), resolved);
  EXPECT_STREQ(resolved, expected);
  EXPECT_EQ(ncclOsRealpath("/nccl/no/such/dir", resolved), nullptr);
}

// Stack size and file limits

class LinuxInitMicrotest : public LinuxMicrotest {
 protected:
  void SetUp() override {
    LinuxMicrotest::SetUp();
    ASSERT_EQ(pthread_getattr_default_np(&saved_), 0);
  }
  void TearDown() override {
    pthread_setattr_default_np(&saved_);
    pthread_attr_destroy(&saved_);
    LinuxMicrotest::TearDown();
  }
  static void SetDefaultStack(size_t bytes) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, bytes);
    pthread_setattr_default_np(&attr);
    pthread_attr_destroy(&attr);
  }
  static size_t DefaultStack() {
    pthread_attr_t attr;
    size_t size = 0;
    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr, &size);
    pthread_attr_destroy(&attr);
    return size;
  }
  pthread_attr_t saved_;
};

TEST_F(LinuxInitMicrotest, Initialize_WithoutTheParamLeavesTheStackAlone) {
  SetDefaultStack(1 << 20);
  ASSERT_EQ(ncclOsInitialize(), ncclSuccess);
  EXPECT_EQ(DefaultStack(), 1u << 20);
  EXPECT_TRUE(g_linuxLog.empty());
}

TEST_F(LinuxInitMicrotest, Initialize_RaisesASmallDefaultStackTo8MB) {
  g_setCpuStackSize = 1;
  SetDefaultStack(1 << 20);
  lx::getrlimit = [](int resource, rlimit* lim) {
    int rc = ::getrlimit((__rlimit_resource_t)resource, lim);
    if (resource == RLIMIT_STACK) lim->rlim_cur = 2048 * 1024;
    return rc;
  };
  ASSERT_EQ(ncclOsInitialize(), ncclSuccess);
  EXPECT_EQ(DefaultStack(), 8192u * 1024);
  EXPECT_TRUE(LinuxLogged("Stack size limit (2048KB) is unsafe; will use 8192KB"));
}

TEST_F(LinuxInitMicrotest, Initialize_ReportsAnUnlimitedStackLimit) {
  g_setCpuStackSize = 1;
  SetDefaultStack(2 << 20);  // glibc's fallback when RLIMIT_STACK is unlimited
  lx::getrlimit = [](int resource, rlimit* lim) {
    int rc = ::getrlimit((__rlimit_resource_t)resource, lim);
    if (resource == RLIMIT_STACK) lim->rlim_cur = RLIM_INFINITY;
    return rc;
  };
  ASSERT_EQ(ncclOsInitialize(), ncclSuccess);
  EXPECT_TRUE(LinuxLogged("Stack size limit (unlimited) is unsafe"));
}

TEST_F(LinuxInitMicrotest, Initialize_KeepsAStackThatIsAlready8MB) {
  g_setCpuStackSize = 1;
  SetDefaultStack(8192 * 1024);
  ASSERT_EQ(ncclOsInitialize(), ncclSuccess);
  EXPECT_FALSE(LinuxLogged("unsafe"));
  SetDefaultStack(16 << 20);
  ASSERT_EQ(ncclOsInitialize(), ncclSuccess);
  EXPECT_EQ(DefaultStack(), 16u << 20);
}

TEST_F(LinuxInitMicrotest, Initialize_GetrlimitFailureIsASystemError) {
  g_setCpuStackSize = 1;
  SetDefaultStack(1 << 20);
  lx::getrlimit = [](int, rlimit*) {
    errno = EPERM;
    return -1;
  };
  EXPECT_EQ(ncclOsInitialize(), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to getrlimit failed"));
}

TEST_F(LinuxInitMicrotest, SetFilesLimit_RaisesTheSoftLimitToTheHardLimit) {
  rlimit saved;
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &saved), 0);
  rlimit lowered = saved;
  lowered.rlim_cur = std::min<rlim_t>(saved.rlim_max, 256);
  ASSERT_EQ(::setrlimit(RLIMIT_NOFILE, &lowered), 0);
  rlimit seen{};
  lx::setrlimit = [&](int resource, const rlimit* lim) {
    seen = *lim;
    return ::setrlimit((__rlimit_resource_t)resource, lim);
  };
  EXPECT_EQ(ncclOsSetFilesLimit(), ncclSuccess);
  EXPECT_EQ(seen.rlim_cur, saved.rlim_max);
  EXPECT_EQ(seen.rlim_max, saved.rlim_max);
  ::setrlimit(RLIMIT_NOFILE, &saved);
}

TEST_F(LinuxInitMicrotest, SetFilesLimit_SyscallFailuresAreSystemErrors) {
  lx::setrlimit = [](int, const rlimit*) {
    errno = EPERM;
    return -1;
  };
  EXPECT_EQ(ncclOsSetFilesLimit(), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to setrlimit failed"));
  lx::getrlimit = [](int, rlimit*) {
    errno = EPERM;
    return -1;
  };
  EXPECT_EQ(ncclOsSetFilesLimit(), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to getrlimit failed"));
}

// Sockets

class LinuxSocketMicrotest : public LinuxMicrotest {
 protected:
  void TearDown() override {
    for (int fd : fds_) ::close(fd);
    LinuxMicrotest::TearDown();
  }
  int Track(int fd) {
    if (fd >= 0) fds_.push_back(fd);
    return fd;
  }
  // A listening loopback TCP socket; *addr gets its port.
  int Listen(sockaddr_in* addr, bool nonblocking = false) {
    int fd = Track(::socket(AF_INET, SOCK_STREAM, 0));
    std::memset(addr, 0, sizeof(*addr));
    addr->sin_family = AF_INET;
    addr->sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    EXPECT_EQ(::bind(fd, (sockaddr*)addr, sizeof(*addr)), 0);
    socklen_t len = sizeof(*addr);
    EXPECT_EQ(::getsockname(fd, (sockaddr*)addr, &len), 0);
    EXPECT_EQ(::listen(fd, 16), 0);
    if (nonblocking) fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
    return fd;
  }
  // A loopback port nothing listens on.
  sockaddr_in ClosedPort() {
    sockaddr_in addr;
    int fd = Listen(&addr);
    ::close(fd);
    fds_.pop_back();
    return addr;
  }
  static ncclSocket Sock(int fd, const sockaddr_in* addr = nullptr) {
    ncclSocket sock;
    std::memset(&sock, 0, sizeof(sock));
    sock.socketDescriptor = fd;
    sock.acceptSocketDescriptor = NCCL_INVALID_SOCKET;
    if (addr) std::memcpy(&sock.addr.sin, addr, sizeof(*addr));
    else sock.addr.sa.sa_family = AF_INET;
    sock.salen = sizeof(sockaddr_in);
    return sock;
  }
  void Pair(int sv[2]) {
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    Track(sv[0]);
    Track(sv[1]);
  }
  static int IntOpt(int fd, int level, int name) {
    int v = -1;
    socklen_t len = sizeof(v);
    ::getsockopt(fd, level, name, &v, &len);
    return v;
  }
  std::vector<int> fds_;
};

TEST_F(LinuxSocketMicrotest, DescriptorValidity) {
  EXPECT_FALSE(ncclOsSocketDescriptorIsValid(NCCL_INVALID_SOCKET));
  EXPECT_TRUE(ncclOsSocketDescriptorIsValid(0));
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  EXPECT_FALSE(ncclOsSocketIsValid(&sock));
  sock.socketDescriptor = 3;
  EXPECT_TRUE(ncclOsSocketIsValid(&sock));
}

TEST_F(LinuxSocketMicrotest, PollSocket_WaitsForTheOpsDirectionWithThePollTimeout) {
  LinuxScopedEnv env("NCCL_SOCKET_POLL_TIMEOUT_MSEC", "7");
  std::vector<std::pair<short, int>> calls;
  lx::poll = [&](pollfd* p, nfds_t n, int timeout) {
    EXPECT_EQ(n, 1u);
    EXPECT_EQ(p->fd, 5);
    calls.push_back({p->events, timeout});
    return 1;
  };
  ncclOsPollSocket(5, NCCL_SOCKET_RECV);
  ncclOsPollSocket(5, NCCL_SOCKET_SEND);
  ASSERT_EQ(calls.size(), 2u);
  EXPECT_EQ(calls[0], std::make_pair((short)POLLIN, 7));
  EXPECT_EQ(calls[1], std::make_pair((short)POLLOUT, 7));
}

TEST_F(LinuxSocketMicrotest, TryAccept_AcceptsAPendingConnection) {
  sockaddr_in addr;
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  sock.acceptSocketDescriptor = Listen(&addr);
  sock.state = ncclSocketStateAccepting;
  int client = Track(::socket(AF_INET, SOCK_STREAM, 0));
  ASSERT_EQ(::connect(client, (sockaddr*)&addr, sizeof(addr)), 0);
  ASSERT_EQ(ncclOsSocketTryAccept(&sock), ncclSuccess);
  Track(sock.socketDescriptor);
  EXPECT_TRUE(FdOpen(sock.socketDescriptor));
  EXPECT_EQ(sock.state, ncclSocketStateAccepted);
  sockaddr_in local;
  socklen_t len = sizeof(local);
  ::getsockname(client, (sockaddr*)&local, &len);
  EXPECT_EQ(sock.addr.sin.sin_port, local.sin_port);  // the peer's address
}

TEST_F(LinuxSocketMicrotest, TryAccept_NothingPendingIsNotAnError) {
  sockaddr_in addr;
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  sock.acceptSocketDescriptor = Listen(&addr, /*nonblocking=*/true);
  sock.state = ncclSocketStateAccepting;
  EXPECT_EQ(ncclOsSocketTryAccept(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateAccepting);
  EXPECT_EQ(sock.errorRetries, 0);
  EXPECT_TRUE(g_linuxLog.empty());
}

TEST_F(LinuxSocketMicrotest, TryAccept_OtherErrorsAreSystemErrors) {
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  sock.acceptSocketDescriptor = Track(::socket(AF_INET, SOCK_STREAM, 0));  // not listening: EINVAL
  sock.state = ncclSocketStateAccepting;
  EXPECT_EQ(ncclOsSocketTryAccept(&sock), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN ncclOsSocketTryAccept: Accept failed"));
}

// Network errors that accept() may report for an already-failed peer are retried up to NCCL_SOCKET_RETRY_CNT.
TEST_F(LinuxSocketMicrotest, TryAccept_RetriesPendingNetworkErrorsUpToTheRetryCount) {
  LinuxScopedEnv env("NCCL_SOCKET_RETRY_CNT", "3");
  for (int err : {ENETDOWN, EPROTO, ENOPROTOOPT, EHOSTDOWN, ENONET, EHOSTUNREACH, EOPNOTSUPP, ENETUNREACH, EINTR}) {
    SCOPED_TRACE(err);
    g_linuxLog.clear();
    lx::accept = [err](int, sockaddr*, socklen_t*) {
      errno = err;
      return -1;
    };
    ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
    sock.state = ncclSocketStateAccepting;
    EXPECT_EQ(ncclOsSocketTryAccept(&sock), ncclSuccess);
    EXPECT_EQ(ncclOsSocketTryAccept(&sock), ncclSuccess);
    EXPECT_EQ(sock.errorRetries, 2);
    EXPECT_TRUE(LinuxLogged("INFO Call to accept returned"));
    EXPECT_EQ(ncclOsSocketTryAccept(&sock), ncclSystemError);
    EXPECT_TRUE(LinuxLogged("exceeded error retry count after 3 attempts"));
    EXPECT_EQ(sock.state, ncclSocketStateAccepting);
  }
}

TEST_F(LinuxSocketMicrotest, TryAccept_WouldBlockIsNotRetried) {
  for (int err : {EAGAIN, EWOULDBLOCK}) {
    lx::accept = [err](int, sockaddr*, socklen_t*) {
      errno = err;
      return -1;
    };
    ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
    EXPECT_EQ(ncclOsSocketTryAccept(&sock), ncclSuccess);
    EXPECT_EQ(sock.errorRetries, 0);
  }
  EXPECT_TRUE(g_linuxLog.empty());
}

TEST_F(LinuxSocketMicrotest, SetFlags_RejectsAnInvalidSocket) {
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  EXPECT_EQ(ncclOsSocketSetFlags(&sock), ncclInvalidArgument);
  EXPECT_TRUE(LinuxLogged("WARN ncclOsSocketSetFlags: invalid socket"));
}

TEST_F(LinuxSocketMicrotest, SetFlags_BlockingSocketGetsNoDelayAndDefaultBuffers) {
  int fd = Track(::socket(AF_INET, SOCK_STREAM, 0));
  int sndDefault = IntOpt(fd, SOL_SOCKET, SO_SNDBUF), rcvDefault = IntOpt(fd, SOL_SOCKET, SO_RCVBUF);
  ncclSocket sock = Sock(fd);
  ASSERT_EQ(ncclOsSocketSetFlags(&sock), ncclSuccess);
  EXPECT_EQ(fcntl(fd, F_GETFL) & O_NONBLOCK, 0);
  EXPECT_NE(IntOpt(fd, IPPROTO_TCP, TCP_NODELAY), 0);
  EXPECT_EQ(IntOpt(fd, SOL_SOCKET, SO_SNDBUF), sndDefault);
  EXPECT_EQ(IntOpt(fd, SOL_SOCKET, SO_RCVBUF), rcvDefault);
}

TEST_F(LinuxSocketMicrotest, SetFlags_AsyncOrAbortableSocketsAreNonBlocking) {
  volatile uint32_t abortFlag = 0;
  for (int variant = 0; variant < 2; variant++) {
    int fd = Track(::socket(AF_INET, SOCK_STREAM, 0));
    ncclSocket sock = Sock(fd);
    if (variant == 0) sock.asyncFlag = 1;
    else sock.abortFlag = &abortFlag;
    ASSERT_EQ(ncclOsSocketSetFlags(&sock), ncclSuccess);
    EXPECT_NE(fcntl(fd, F_GETFL) & O_NONBLOCK, 0) << variant;
  }
}

TEST_F(LinuxSocketMicrotest, SetFlags_AppliesTheBufferSizeParams) {
  LinuxScopedEnv snd("NCCL_SOCKET_SNDBUF", "65536");
  LinuxScopedEnv rcv("NCCL_SOCKET_RCVBUF", "32768");
  int fd = Track(::socket(AF_INET, SOCK_STREAM, 0));
  ncclSocket sock = Sock(fd);
  ASSERT_EQ(ncclOsSocketSetFlags(&sock), ncclSuccess);
  EXPECT_EQ(IntOpt(fd, SOL_SOCKET, SO_SNDBUF), 2 * 65536);  // Linux doubles the request
  EXPECT_EQ(IntOpt(fd, SOL_SOCKET, SO_RCVBUF), 2 * 32768);
}

TEST_F(LinuxSocketMicrotest, SetFlags_SyscallFailuresAreSystemErrors) {
  ncclSocket sock = Sock(1 << 20);  // valid-looking, but not open: fcntl fails
  sock.asyncFlag = 1;
  EXPECT_EQ(ncclOsSocketSetFlags(&sock), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to fcntl failed"));

  int p[2];
  ASSERT_EQ(pipe(p), 0);
  Track(p[0]);
  Track(p[1]);
  sock = Sock(p[0]);  // not a socket: setsockopt fails
  EXPECT_EQ(ncclOsSocketSetFlags(&sock), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to setsockopt TCP NODELAY failed"));
}

TEST_F(LinuxSocketMicrotest, SetFlags_BufferSizeFailuresAreSystemErrors) {
  LinuxScopedEnv snd("NCCL_SOCKET_SNDBUF", "65536");
  LinuxScopedEnv rcv("NCCL_SOCKET_RCVBUF", "32768");
  int fd = Track(::socket(AF_INET, SOCK_STREAM, 0));
  ncclSocket sock = Sock(fd);
  for (int failing : {SO_SNDBUF, SO_RCVBUF}) {
    lx::setsockopt = [failing](int s, int level, int name, const void* v, socklen_t len) {
      if (level == SOL_SOCKET && name == failing) {
        errno = ENOBUFS;
        return -1;
      }
      return ::setsockopt(s, level, name, v, len);
    };
    EXPECT_EQ(ncclOsSocketSetFlags(&sock), ncclSystemError);
  }
  EXPECT_TRUE(LinuxLogged("WARN Call to setsockopt SO_SNDBUF failed"));
  EXPECT_TRUE(LinuxLogged("WARN Call to setsockopt SO_RCVBUF failed"));
}

TEST_F(LinuxSocketMicrotest, ResetAccept_ClosesThePeerAndMarksABadHandshake) {
  int sv[2];
  Pair(sv);
  ncclSocket sock = Sock(sv[0]);
  sock.state = ncclSocketStateAccepted;
  sock.finalizeCounter = 5;
  ncclOsSocketResetAccept(&sock);
  EXPECT_FALSE(FdOpen(sv[0]));
  EXPECT_EQ(sock.socketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(sock.state, ncclSocketStateBadHandshake);
  EXPECT_EQ(sock.finalizeCounter, 0);
}

TEST_F(LinuxSocketMicrotest, ResetFd_CreatesAConfiguredSocket) {
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  sock.asyncFlag = 1;
  ASSERT_EQ(ncclOsSocketResetFd(&sock), ncclSuccess);
  Track(sock.socketDescriptor);
  ASSERT_TRUE(ncclOsSocketIsValid(&sock));
  EXPECT_EQ(IntOpt(sock.socketDescriptor, SOL_SOCKET, SO_TYPE), SOCK_STREAM);
  EXPECT_NE(fcntl(sock.socketDescriptor, F_GETFL) & O_NONBLOCK, 0);
  EXPECT_NE(IntOpt(sock.socketDescriptor, IPPROTO_TCP, TCP_NODELAY), 0);
}

// An existing descriptor keeps its number but becomes a fresh socket.
TEST_F(LinuxSocketMicrotest, ResetFd_ReusesTheExistingDescriptorNumber) {
  sockaddr_in addr;
  int fd = Listen(&addr);
  ncclSocket sock = Sock(fd);
  ASSERT_EQ(ncclOsSocketResetFd(&sock), ncclSuccess);
  EXPECT_EQ(sock.socketDescriptor, fd);
  EXPECT_EQ(IntOpt(fd, SOL_SOCKET, SO_ACCEPTCONN), 0);
  int probe = ::socket(AF_INET, SOCK_STREAM, 0);  // the temporary descriptor was closed
  EXPECT_LE(probe, fd + 1);
  ::close(probe);
}

TEST_F(LinuxSocketMicrotest, ResetFd_SocketFailureLeavesTheDescriptorAlone) {
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  sock.addr.sa.sa_family = AF_UNSPEC;
  EXPECT_EQ(ncclOsSocketResetFd(&sock), ncclSystemError);
  EXPECT_EQ(sock.socketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_TRUE(LinuxLogged("WARN Call to socket failed"));
}

TEST_F(LinuxSocketMicrotest, ResetFd_Dup2FailureClosesTheNewSocket) {
  int probe = ::socket(AF_INET, SOCK_STREAM, 0);  // the number the new socket will get
  ::close(probe);
  ncclSocket sock = Sock(1 << 29);  // above any RLIMIT_NOFILE: dup2 fails
  EXPECT_EQ(ncclOsSocketResetFd(&sock), ncclSystemError);
  EXPECT_EQ(sock.socketDescriptor, 1 << 29);
  EXPECT_TRUE(LinuxLogged("WARN Call to dup2 failed"));
  EXPECT_FALSE(FdOpen(probe));
}

TEST_F(LinuxSocketMicrotest, ResetFd_PropagatesASetFlagsFailure) {
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  sock.addr.sa.sa_family = AF_UNIX;  // a UNIX stream socket rejects TCP_NODELAY
  EXPECT_EQ(ncclOsSocketResetFd(&sock), ncclSystemError);
  Track(sock.socketDescriptor);
  EXPECT_TRUE(LinuxLogged("WARN Call to setsockopt TCP NODELAY failed"));
}

TEST_F(LinuxSocketMicrotest, StartConnect_ConnectsToAListener) {
  sockaddr_in addr;
  Listen(&addr);
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
  ASSERT_EQ(ncclOsSocketStartConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateConnected);
}

TEST_F(LinuxSocketMicrotest, StartConnect_InProgressPollsUntilConnected) {
  sockaddr_in addr;
  Listen(&addr);
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
  lx::connect = [](int fd, const sockaddr* a, socklen_t l) {
    fcntl(fd, F_SETFL, O_NONBLOCK);
    int rc = ::connect(fd, a, l);
    if (rc == 0) errno = EINPROGRESS;  // loopback may finish at once
    return -1;
  };
  ASSERT_EQ(ncclOsSocketStartConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateConnectPolling);
  for (int i = 0; i < 1000 && sock.state == ncclSocketStateConnectPolling; i++)
    ASSERT_EQ(ncclOsSocketPollConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateConnected);
}

TEST_F(LinuxSocketMicrotest, StartConnect_RefusedIsRetriedWithAGrowingSleep) {
  LinuxScopedEnv retries("NCCL_SOCKET_RETRY_CNT", "2");
  sockaddr_in addr = ClosedPort();
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
  int fd = sock.socketDescriptor;
  ASSERT_EQ(ncclOsSocketStartConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateConnecting);
  EXPECT_EQ(sock.errorRetries, 1);
  EXPECT_EQ(sock.socketDescriptor, fd);  // reset in place
  EXPECT_TRUE(LinuxLogged("retrying (1/2) after sleep for 0 msec"));

  ASSERT_EQ(ncclOsSocketStartConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.errorRetries, 2);
  EXPECT_EQ(ncclOsSocketStartConnect(&sock), ncclRemoteError);
  EXPECT_EQ(sock.state, ncclSocketStateError);
  EXPECT_TRUE(LinuxLogged("exceeded error retry count after 3 attempts"));
}

TEST_F(LinuxSocketMicrotest, StartConnect_SleepGrowsWithTheRetryCount) {
  LinuxScopedEnv sleep("NCCL_SOCKET_RETRY_SLEEP_MSEC", "3");
  sockaddr_in addr = ClosedPort();
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
  sock.errorRetries = 2;
  ASSERT_EQ(ncclOsSocketStartConnect(&sock), ncclSuccess);
  EXPECT_TRUE(LinuxLogged("retrying (3/34) after sleep for 9 msec"));
}

TEST_F(LinuxSocketMicrotest, StartConnect_TransientErrnosAreRetried) {
  sockaddr_in addr = ClosedPort();
  for (int err : {EINTR, EWOULDBLOCK, EAGAIN, ETIMEDOUT, EHOSTUNREACH, ECONNREFUSED}) {
    SCOPED_TRACE(err);
    lx::connect = [err](int, const sockaddr*, socklen_t) {
      errno = err;
      return -1;
    };
    ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
    ASSERT_EQ(ncclOsSocketStartConnect(&sock), ncclSuccess);
    EXPECT_EQ(sock.state, ncclSocketStateConnecting);
    EXPECT_EQ(sock.errorRetries, 1);
  }
}

// A caller with its own retry policy gets the reset but no count and no sleep.
TEST_F(LinuxSocketMicrotest, StartConnect_CustomRetryOnlyResets) {
  LinuxScopedEnv retries("NCCL_SOCKET_RETRY_CNT", "0");
  sockaddr_in addr = ClosedPort();
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
  sock.customRetry = 1;
  ASSERT_EQ(ncclOsSocketStartConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateConnecting);
  EXPECT_EQ(sock.errorRetries, 0);
  EXPECT_FALSE(LinuxLogged("retrying"));
}

TEST_F(LinuxSocketMicrotest, StartConnect_OtherErrorsFail) {
  sockaddr_in addr;
  Listen(&addr);
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
  sock.salen = 1;  // EINVAL
  EXPECT_EQ(ncclOsSocketStartConnect(&sock), ncclSystemError);
  EXPECT_EQ(sock.state, ncclSocketStateError);
  EXPECT_TRUE(LinuxLogged("WARN ncclOsSocketStartConnect: connect to 127.0.0.1<"));
  EXPECT_TRUE(LinuxLogged("failed : Invalid argument"));
}

TEST_F(LinuxSocketMicrotest, StartConnect_PropagatesAResetFailure) {
  lx::connect = [](int, const sockaddr*, socklen_t) {
    errno = ECONNREFUSED;
    return -1;
  };
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)));
  sock.addr.sa.sa_family = AF_UNSPEC;  // the replacement socket cannot be created
  EXPECT_EQ(ncclOsSocketStartConnect(&sock), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to socket failed"));
}

TEST_F(LinuxSocketMicrotest, PollConnect_NotReadyYetIsSuccess) {
  ncclSocket sock = Sock(3);
  sock.state = ncclSocketStateConnectPolling;
  int pollRc = 0;
  lx::poll = [&](pollfd* p, nfds_t, int timeout) {
    EXPECT_EQ(p->events, POLLOUT);
    EXPECT_EQ(timeout, 1);
    errno = EINTR;
    return pollRc;
  };
  EXPECT_EQ(ncclOsSocketPollConnect(&sock), ncclSuccess);
  pollRc = -1;  // interrupted
  EXPECT_EQ(ncclOsSocketPollConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateConnectPolling);
}

TEST_F(LinuxSocketMicrotest, PollConnect_PollFailureIsASystemError) {
  ncclSocket sock = Sock(3);
  lx::poll = [](pollfd*, nfds_t, int) {
    errno = ENOMEM;
    return -1;
  };
  EXPECT_EQ(ncclOsSocketPollConnect(&sock), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN ncclOsSocketPollConnect to"));
}

TEST_F(LinuxSocketMicrotest, PollConnect_ChecksThePendingSocketError) {
  sockaddr_in addr = ClosedPort();
  ncclSocket sock = Sock(Track(::socket(AF_INET, SOCK_STREAM, 0)), &addr);
  lx::poll = [](pollfd*, nfds_t, int) { return 1; };
  lx::getsockopt = [](int, int level, int name, void* v, socklen_t*) {
    EXPECT_EQ(level, SOL_SOCKET);
    EXPECT_EQ(name, SO_ERROR);
    *(int*)v = ECONNREFUSED;
    return 0;
  };
  ASSERT_EQ(ncclOsSocketPollConnect(&sock), ncclSuccess);
  EXPECT_EQ(sock.state, ncclSocketStateConnecting);
  EXPECT_TRUE(LinuxLogged("ncclOsSocketPollConnect: connect to"));

  lx::getsockopt = [](int, int, int, void*, socklen_t*) {
    errno = EBADF;
    return -1;
  };
  EXPECT_EQ(ncclOsSocketPollConnect(&sock), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to getsockopt failed"));
}

TEST_F(LinuxSocketMicrotest, ProgressOpt_BlockingSendAndRecvMoveEverything) {
  int sv[2];
  Pair(sv);
  ncclSocket a = Sock(sv[0]), b = Sock(sv[1]);
  char out[64], in[64] = {};
  for (int i = 0; i < 64; i++) out[i] = (char)i;
  int offset = 0, closed = -1;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &a, out, sizeof(out), &offset, 1, &closed), ncclSuccess);
  EXPECT_EQ(offset, 64);
  EXPECT_EQ(closed, 0);
  offset = 16;  // resumes at the offset
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &b, in, sizeof(in), &offset, 1, &closed), ncclSuccess);
  EXPECT_EQ(offset, 64);
  EXPECT_EQ(std::memcmp(in + 16, out, 48), 0);
}

// A blocking recv keeps reading until the buffer is full.
TEST_F(LinuxSocketMicrotest, ProgressOpt_BlockingRecvWaitsForTheRest) {
  int sv[2];
  Pair(sv);
  ncclSocket b = Sock(sv[1]);
  std::thread writer([&] {
    ::send(sv[0], "abcd", 4, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ::send(sv[0], "efgh", 4, 0);
  });
  char in[8];
  int offset = 0, closed = 0;
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &b, in, sizeof(in), &offset, 1, &closed), ncclSuccess);
  writer.join();
  EXPECT_EQ(offset, 8);
  EXPECT_EQ(std::string(in, 8), "abcdefgh");
}

TEST_F(LinuxSocketMicrotest, ProgressOpt_NonBlockingReturnsWhatIsThere) {
  int sv[2];
  Pair(sv);
  ncclSocket b = Sock(sv[1]);
  char in[8];
  int offset = 0, closed = 0;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &b, in, sizeof(in), &offset, 0, &closed), ncclSuccess);
  EXPECT_EQ(offset, 0);
  ::send(sv[0], "abc", 3, 0);
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &b, in, sizeof(in), &offset, 0, &closed), ncclSuccess);
  EXPECT_EQ(offset, 3);
  EXPECT_EQ(closed, 0);
}

// Async and abortable sockets never block, whatever the caller asks.
TEST_F(LinuxSocketMicrotest, ProgressOpt_AsyncOrAbortableNeverBlocks) {
  int sv[2];
  Pair(sv);
  volatile uint32_t abortFlag = 0;
  char in[8];
  int offset = 0, closed = 0;
  ncclSocket b = Sock(sv[1]);
  b.asyncFlag = 1;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &b, in, sizeof(in), &offset, 1, &closed), ncclSuccess);
  EXPECT_EQ(offset, 0);
  b = Sock(sv[1]);
  b.abortFlag = &abortFlag;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &b, in, sizeof(in), &offset, 1, &closed), ncclSuccess);
  EXPECT_EQ(offset, 0);
}

// A synchronous socket keeps reading while data comes in; an async one makes a single attempt.
TEST_F(LinuxSocketMicrotest, ProgressOpt_AsyncMakesOneAttempt) {
  for (int async = 0; async < 2; async++) {
    int sv[2];
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv), 0);  // one record per recv()
    Track(sv[0]);
    Track(sv[1]);
    ::send(sv[1], "abcd", 4, 0);
    ::send(sv[1], "efgh", 4, 0);
    ncclSocket b = Sock(sv[0]);
    b.asyncFlag = async;
    char in[8];
    int offset = 0, closed = 0;
    ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &b, in, sizeof(in), &offset, 0, &closed), ncclSuccess);
    EXPECT_EQ(offset, async ? 4 : 8);
  }
}

TEST_F(LinuxSocketMicrotest, ProgressOpt_PeerCloseIsReportedNotFailed) {
  int sv[2];
  Pair(sv);
  ncclSocket a = Sock(sv[0]);
  ::shutdown(sv[1], SHUT_RDWR);
  char buf[4] = {};
  int offset = 0, closed = 0;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &a, buf, sizeof(buf), &offset, 1, &closed), ncclSuccess);
  EXPECT_EQ(closed, 1);
  closed = 0;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &a, buf, sizeof(buf), &offset, 1, &closed), ncclSuccess);
  EXPECT_EQ(closed, 1);  // EPIPE, and no SIGPIPE
  EXPECT_EQ(offset, 0);
}

// Closing a UNIX stream socket with unread data resets the peer.
TEST_F(LinuxSocketMicrotest, ProgressOpt_ConnectionResetOnRecvIsAClose) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  Track(sv[0]);
  ::send(sv[0], "x", 1, 0);
  ::close(sv[1]);
  ncclSocket a = Sock(sv[0]);
  char buf[4];
  int offset = 0, closed = 0;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &a, buf, sizeof(buf), &offset, 1, &closed), ncclSuccess);
  EXPECT_EQ(closed, 1);
  EXPECT_TRUE(g_linuxLog.empty());
}

TEST_F(LinuxSocketMicrotest, ProgressOpt_OtherErrorsAreRemoteErrors) {
  int p[2];
  ASSERT_EQ(pipe(p), 0);
  Track(p[0]);
  Track(p[1]);
  char buf[4] = {};
  int offset = 0, closed = 0;
  ncclSocket sock = Sock(p[0]);
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &sock, buf, sizeof(buf), &offset, 1, &closed), ncclRemoteError);
  EXPECT_TRUE(LinuxLogged("WARN ncclOsSocketProgressOpt: Call to recv from"));
  sock = Sock(p[1]);
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &sock, buf, sizeof(buf), &offset, 1, &closed), ncclRemoteError);
  EXPECT_TRUE(LinuxLogged("WARN ncclOsSocketProgressOpt: Call to send to"));
  EXPECT_EQ(closed, 0);
}

TEST_F(LinuxSocketMicrotest, ProgressOpt_AbortStopsAfterTheTransfer) {
  int sv[2];
  Pair(sv);
  volatile uint32_t abortFlag = 1;
  ncclSocket a = Sock(sv[0]);
  a.abortFlag = &abortFlag;
  int offset = 0, closed = 0;
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &a, (void*)"abcd", 4, &offset, 1, &closed), ncclInternalError);
  EXPECT_EQ(offset, 4);
  EXPECT_TRUE(LinuxLogged("INFO ncclOsSocketProgressOpt: abort called"));
}

// Interface enumeration over a scripted getifaddrs() list.
class LinuxIfMicrotest : public LinuxMicrotest {
 protected:
  void SetUp() override {
    LinuxMicrotest::SetUp();
    ifs_.Install();
  }
  void TearDown() override {
    EXPECT_EQ(ifs_.freed, ifs_.listed);
    LinuxMicrotest::TearDown();
  }
  template <typename... A>
  void Add(A... args) {
    ifs_.Add(args...);
  }
  std::vector<std::string> Find(const char* prefixList, int family = -1, int maxIfs = MAX_IFS) {
    int found = -1;
    EXPECT_EQ(ncclOsFindInterfaces(prefixList, names_, addrs_, family, MAX_IF_NAME_SIZE, maxIfs, &found), ncclSuccess);
    std::vector<std::string> out;
    for (int i = 0; i < found; i++) out.push_back(names_ + i * MAX_IF_NAME_SIZE);
    return out;
  }
  std::string Addr(int i) {
    char host[INET6_ADDRSTRLEN] = "";
    if (addrs_[i].sa.sa_family == AF_INET) inet_ntop(AF_INET, &addrs_[i].sin.sin_addr, host, sizeof(host));
    else inet_ntop(AF_INET6, &addrs_[i].sin6.sin6_addr, host, sizeof(host));
    return host;
  }
  // The local interface in the remote address's subnet, or "".
  std::string MatchSubnet(int family, const char* remote, uint32_t scope = 0) {
    sockaddr_storage ss;
    LinuxFakeIfaddrs::Fill(&ss, family, remote, scope);
    union ncclSocketAddress remoteAddr, local;
    std::memset(&local, 0, sizeof(local));
    std::memcpy(&remoteAddr, &ss, sizeof(remoteAddr));
    char ifName[MAX_IF_NAME_SIZE] = "";
    int found = -1;
    EXPECT_EQ(ncclFindInterfaceMatchSubnet(ifName, &local, &remoteAddr, sizeof(ifName), &found), ncclSuccess);
    if (!found) return "";
    addrs_[0] = local;
    return ifName;
  }
  LinuxFakeIfaddrs ifs_;
  char names_[MAX_IFS * MAX_IF_NAME_SIZE];
  union ncclSocketAddress addrs_[MAX_IFS];
};

TEST_F(LinuxIfMicrotest, FindInterfaces_KeepsRunningIpInterfacesOncePerName) {
  Add("noaddr", AF_UNSPEC);
  Add("eth0", AF_INET, "10.0.0.1");
  Add("eth0", AF_INET6, "2001:db8::1");  // same name: only the first address is kept
  Add("pkt0", AF_PACKET);
  Add("down0", AF_INET, "10.0.0.2", nullptr, IFF_UP);  // not RUNNING
  Add("lo", AF_INET6, "::1");                          // IPv6 loopback is skipped
  Add("eth1", AF_INET6, "2001:db8::2");
  Add("lo", AF_INET, "127.0.0.1");
  EXPECT_EQ(Find(nullptr), (std::vector<std::string>{"eth0", "eth1", "lo"}));
  EXPECT_EQ(Addr(0), "10.0.0.1");
  EXPECT_EQ(addrs_[0].sin.sin_family, AF_INET);
  EXPECT_EQ(Addr(1), "2001:db8::2");
  EXPECT_EQ(addrs_[1].sin6.sin6_family, AF_INET6);
}

TEST_F(LinuxIfMicrotest, FindInterfaces_PrefixExclusionAndExactLists) {
  Add("eth0", AF_INET, "10.0.0.1");
  Add("eth10", AF_INET, "10.0.0.2");
  Add("ib0", AF_INET, "10.0.1.1");
  Add("lo", AF_INET, "127.0.0.1");
  EXPECT_EQ(Find("eth"), (std::vector<std::string>{"eth0", "eth10"}));
  EXPECT_EQ(Find("ib,lo"), (std::vector<std::string>{"ib0", "lo"}));
  EXPECT_EQ(Find("^eth"), (std::vector<std::string>{"ib0", "lo"}));
  EXPECT_EQ(Find("=eth0"), (std::vector<std::string>{"eth0"}));
  EXPECT_EQ(Find("=eth"), (std::vector<std::string>{}));
  EXPECT_EQ(Find("^=eth0"), (std::vector<std::string>{"eth10", "ib0", "lo"}));
}

TEST_F(LinuxIfMicrotest, FindInterfaces_FamilyFilterAndMaxIfs) {
  Add("eth0", AF_INET, "10.0.0.1");
  Add("eth1", AF_INET6, "2001:db8::2");
  Add("eth2", AF_INET, "10.0.0.3");
  EXPECT_EQ(Find(nullptr, AF_INET6), (std::vector<std::string>{"eth1"}));
  EXPECT_EQ(Find(nullptr, AF_INET), (std::vector<std::string>{"eth0", "eth2"}));
  EXPECT_EQ(Find(nullptr, -1, 2), (std::vector<std::string>{"eth0", "eth1"}));
}

// The address union is cleared past the stored sockaddr_in.
TEST_F(LinuxIfMicrotest, FindInterfaces_ClearsTheStoredAddress) {
  Add("eth0", AF_INET, "10.0.0.1");
  std::memset(addrs_, 0xff, sizeof(addrs_));
  ASSERT_EQ(Find(nullptr).size(), 1u);
  const char* tail = (const char*)&addrs_[0] + sizeof(sockaddr_in);
  for (size_t i = 0; i < sizeof(addrs_[0]) - sizeof(sockaddr_in); i++) EXPECT_EQ(tail[i], 0);
}

TEST_F(LinuxIfMicrotest, FindInterfaces_GetifaddrsFailureIsASystemError) {
  lx::getifaddrs = [](ifaddrs**) {
    errno = ENOMEM;
    return -1;
  };
  int found = -1;
  EXPECT_EQ(ncclOsFindInterfaces(nullptr, names_, addrs_, -1, MAX_IF_NAME_SIZE, MAX_IFS, &found), ncclSystemError);
  EXPECT_EQ(found, 0);
  EXPECT_EQ(ncclFindInterfaceMatchSubnet(names_, addrs_, addrs_, MAX_IF_NAME_SIZE, &found), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN Call to getifaddrs failed"));
}

TEST_F(LinuxIfMicrotest, MatchSubnet_IPv4UsesTheNetmask) {
  Add("noaddr", AF_UNSPEC);
  Add("pkt0", AF_PACKET);
  Add("eth1", AF_INET6, "2001:db8::1", "ffff:ffff:ffff:ffff::");
  Add("eth0", AF_INET, "10.0.0.1", "255.255.0.0");
  Add("eth2", AF_INET, "10.0.9.9", "255.0.0.0");
  EXPECT_EQ(MatchSubnet(AF_INET, "10.0.200.3"), "eth0");  // first match wins
  EXPECT_EQ(Addr(0), "10.0.0.1");
  EXPECT_EQ(MatchSubnet(AF_INET, "10.7.0.1"), "eth2");
  EXPECT_EQ(MatchSubnet(AF_INET, "11.0.0.1"), "");
}

TEST_F(LinuxIfMicrotest, MatchSubnet_IPv6ComparesThePrefixAndTheScope) {
  Add("eth0", AF_INET, "10.0.0.1", "255.0.0.0");
  Add("eth1", AF_INET6, "2001:db8:0:1::1", "ffff:ffff:ffff:ffff::");
  Add("eth2", AF_INET6, "fe80::1", "ffff:ffff:ffff:ffff::", IFF_UP | IFF_RUNNING, 2);
  EXPECT_EQ(MatchSubnet(AF_INET6, "2001:db8:0:1::99"), "eth1");
  EXPECT_EQ(Addr(0), "2001:db8:0:1::1");
  EXPECT_EQ(MatchSubnet(AF_INET6, "2001:db8:0:2::1"), "");  // differs in the last prefix byte
  EXPECT_EQ(MatchSubnet(AF_INET6, "fe80::5", 2), "eth2");
  EXPECT_EQ(MatchSubnet(AF_INET6, "fe80::5", 3), "");  // same prefix, other link
}

TEST_F(LinuxIfMicrotest, RealGetifaddrsFindsLoopback) {
  lx::Reset();
  int found = 0;
  ASSERT_EQ(ncclOsFindInterfaces("=lo", names_, addrs_, AF_INET, MAX_IF_NAME_SIZE, MAX_IFS, &found), ncclSuccess);
  ASSERT_EQ(found, 1);
  EXPECT_STREQ(names_, "lo");
  EXPECT_EQ(Addr(0), "127.0.0.1");
}

// Socket close

using LinuxSocketCloseMicrotest = LinuxSocketMicrotest;

TEST_F(LinuxSocketCloseMicrotest, ClosesAnOpenSocket) {
  EXPECT_EQ(ncclSocketClose(nullptr), ncclSuccess);
  int sv[2];
  Pair(sv);
  ncclSocket sock = Sock(sv[0]);
  sock.state = ncclSocketStateReady;
  ASSERT_EQ(ncclSocketClose(&sock), ncclSuccess);
  EXPECT_FALSE(FdOpen(sv[0]));
  EXPECT_EQ(sock.state, ncclSocketStateClosed);
  EXPECT_EQ(sock.socketDescriptor, NCCL_INVALID_SOCKET);
  char c;
  EXPECT_EQ(::recv(sv[1], &c, 1, 0), 0);  // the peer saw the FIN
}

// shutdown() reaches the peer even if another descriptor still references the socket.
TEST_F(LinuxSocketCloseMicrotest, ShutsDownADuplicatedSocket) {
  int sv[2];
  Pair(sv);
  int dupFd = Track(dup(sv[0]));
  ncclSocket sock = Sock(sv[0]);
  sock.state = ncclSocketStateReady;
  ASSERT_EQ(ncclSocketClose(&sock), ncclSuccess);
  char c;
  EXPECT_EQ(::recv(sv[1], &c, 1, 0), 0);
  EXPECT_TRUE(FdOpen(dupFd));
}

TEST_F(LinuxSocketCloseMicrotest, LeavesTheDescriptorOfANoneOrClosedSocket) {
  for (ncclSocketState state : {ncclSocketStateNone, ncclSocketStateNum}) {
    int sv[2];
    Pair(sv);
    ncclSocket sock = Sock(sv[0]);
    sock.state = state;
    ASSERT_EQ(ncclSocketClose(&sock), ncclSuccess);
    EXPECT_TRUE(FdOpen(sv[0]));
    EXPECT_EQ(sock.state, ncclSocketStateClosed);
    EXPECT_EQ(sock.socketDescriptor, NCCL_INVALID_SOCKET);
  }
  ncclSocket sock = Sock(NCCL_INVALID_SOCKET);
  sock.state = ncclSocketStateReady;
  EXPECT_EQ(ncclSocketClose(&sock), ncclSuccess);
}

TEST_F(LinuxSocketCloseMicrotest, WaitDrainsUntilThePeerCloses) {
  int sv[2];
  Pair(sv);
  ::send(sv[1], "leftover", 8, 0);
  std::thread peer([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ::shutdown(sv[1], SHUT_WR);
  });
  ncclSocket sock = Sock(sv[0]);
  sock.state = ncclSocketStateReady;
  EXPECT_EQ(ncclSocketClose(&sock, /*wait=*/true), ncclSuccess);
  peer.join();
  EXPECT_FALSE(FdOpen(sv[0]));
}

#ifndef NCCL_TLS_BACKEND_OPENSSL3  // the fake TLS state below is only safe without a backend
TEST_F(LinuxSocketCloseMicrotest, WaitStopsOnAReadErrorAndFreesTlsState) {
  int sv[2];
  Pair(sv);
  ncclSocket sock = Sock(sv[0]);
  sock.state = ncclSocketStateReady;
  sock.crypto = (ncclSocketCrypto*)malloc(16);  // reads now fail: no TLS backend
  ASSERT_EQ(ncclSocketClose(&sock, /*wait=*/true), ncclSuccess);
  EXPECT_EQ(sock.crypto, nullptr);
  EXPECT_FALSE(FdOpen(sv[0]));
  EXPECT_TRUE(LinuxLogged("ncclCryptSocketRecv called without TLS support"));
}
#endif

// Shared mutex/condition variable and CPU sets

using LinuxSyncMicrotest = LinuxMicrotest;

TEST_F(LinuxSyncMicrotest, MutexCondShared_HonoursTheInitializedFlag) {
  std::mutex m;
  std::condition_variable cv;
  int initialized = 1;
  ncclOsSetMutexCondShared(m, cv, &initialized);  // already set: untouched
  EXPECT_EQ(initialized, 1);
  initialized = 0;
  ncclOsUnsetMutexCondShared(m, cv, &initialized);  // not set: untouched
  EXPECT_EQ(initialized, 0);
  ncclOsSetMutexCondShared(m, cv, &initialized);
  EXPECT_EQ(initialized, 1);
  { std::lock_guard<std::mutex> lock(m); }
  ncclOsUnsetMutexCondShared(m, cv, &initialized);
  EXPECT_EQ(initialized, 0);
  ncclOsSetMutexCondShared(m, cv, nullptr);
  ncclOsUnsetMutexCondShared(m, cv, nullptr);
}

// A process-private condition variable would not wake a waiter in another process.
TEST_F(LinuxSyncMicrotest, MutexCondShared_WakesAWaiterInAnotherProcess) {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    int ready = 0;
  };
  void* mem = ::mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mem, MAP_FAILED);
  Shared* s = new (mem) Shared;
  ncclOsSetMutexCondShared(s->m, s->cv, nullptr);
  pid_t child = fork();
  if (child == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
      std::lock_guard<std::mutex> lock(s->m);
      s->ready = 1;
    }
    s->cv.notify_all();
    _exit(0);
  }
  auto start = std::chrono::steady_clock::now();
  {
    std::unique_lock<std::mutex> lock(s->m);
    while (!s->ready && std::chrono::steady_clock::now() - start < std::chrono::seconds(10))
      s->cv.wait_for(lock, std::chrono::seconds(5));
  }
  auto waited = std::chrono::steady_clock::now() - start;
  int status = 0;
  waitpid(child, &status, 0);
  EXPECT_EQ(s->ready, 1);
  EXPECT_LT(waited, std::chrono::seconds(4));
  ncclOsUnsetMutexCondShared(s->m, s->cv, nullptr);
  ::munmap(mem, sizeof(Shared));
}

TEST_F(LinuxSyncMicrotest, CpuSetOperations) {
  ncclAffinity a, b;
  std::memset(&a, 0xff, sizeof(a));
  ncclOsCpuZero(a);
  EXPECT_EQ(ncclOsCpuCount(a), 0);
  ncclOsCpuSet(a, 1);
  ncclOsCpuSet(a, 3);
  ncclOsCpuSet(a, 200);
  EXPECT_EQ(ncclOsCpuCount(a), 3);
  EXPECT_TRUE(ncclOsCpuIsSet(a, 200));
  EXPECT_FALSE(ncclOsCpuIsSet(a, 2));
  ncclOsCpuZero(b);
  ncclOsCpuSet(b, 3);
  ncclOsCpuSet(b, 4);
  ncclAffinity both = ncclOsCpuAnd(a, b);
  EXPECT_EQ(ncclOsCpuCount(both), 1);
  EXPECT_TRUE(ncclOsCpuIsSet(both, 3));
}

TEST_F(LinuxSyncMicrotest, Affinity_GetAndSetTheCallingThread) {
  ncclAffinity mine, expected;
  ASSERT_EQ(ncclOsGetAffinity(&mine), ncclSuccess);
  ASSERT_EQ(sched_getaffinity(0, sizeof(expected), &expected), 0);
  EXPECT_TRUE(CPU_EQUAL(&mine, &expected));
  EXPECT_EQ(ncclOsSetAffinity(mine), ncclSuccess);
  int cpu = ncclOsGetCpu();
  EXPECT_GE(cpu, 0);
  EXPECT_TRUE(ncclOsCpuIsSet(mine, cpu));
}

TEST_F(LinuxSyncMicrotest, Affinity_FailuresAreSystemErrors) {
  ncclAffinity none;
  CPU_ZERO(&none);
  EXPECT_EQ(ncclOsSetAffinity(none), ncclSystemError);  // EINVAL
  EXPECT_TRUE(LinuxLogged("WARN sched_setaffinity failed with error: Invalid argument"));
  lx::sched_getaffinity = [](pid_t, size_t, cpu_set_t*) {
    errno = EFAULT;
    return -1;
  };
  EXPECT_EQ(ncclOsGetAffinity(&none), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN sched_getaffinity failed with error: Bad address"));
}

// sysfs readers, over a scratch tree that the /sys hooks redirect to.

class LinuxSysfsMicrotest : public LinuxMicrotest {
 protected:
  void SetUp() override {
    LinuxMicrotest::SetUp();
    char tmpl[] = "/tmp/nccl-linux-sysfs-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    root_ = tmpl;
    auto redirect = [this](const char* path) {
      std::string p = path;
      return p.rfind("/sys/", 0) == 0 ? root_ + p : p;
    };
    lx::realpath = [=](const char* p, char* out) {
      seenPaths_.push_back(p);
      return ::realpath(redirect(p).c_str(), out);
    };
    lx::fopen = [=](const char* p, const char* mode) { return ::fopen(redirect(p).c_str(), mode); };
    lx::opendir = [=](const char* p) { return ::opendir(redirect(p).c_str()); };
  }
  void TearDown() override {
    std::string cmd = "rm -rf " + root_;
    EXPECT_EQ(system(cmd.c_str()), 0);
    LinuxMicrotest::TearDown();
  }
  void Mkdirs(const std::string& rel) {
    std::string path = root_;
    size_t pos = 0;
    while (pos != std::string::npos) {
      pos = rel.find('/', pos + 1);
      path = root_ + rel.substr(0, pos);
      mkdir(path.c_str(), 0755);
    }
  }
  void Write(const std::string& rel, const std::string& content) {
    FILE* f = ::fopen((root_ + rel).c_str(), "w");
    ASSERT_NE(f, nullptr);
    fwrite(content.data(), 1, content.size(), f);
    fclose(f);
  }
  // A PCI device the way sysfs lays it out: /sys/class/pci_bus/<bus> links into /sys/devices.
  std::string AddPci(const std::string& busId) {
    std::string bus = busId.substr(0, 7), dev = "/sys/devices/pci" + bus;
    Mkdirs(dev + "/pci_bus/" + bus);
    Mkdirs(dev + "/" + busId);
    Mkdirs("/sys/class/pci_bus");
    symlink(("../../devices/pci" + bus + "/pci_bus/" + bus).c_str(), (root_ + "/sys/class/pci_bus/" + bus).c_str());
    return root_ + dev + "/" + busId;
  }
  std::string root_;
  std::vector<std::string> seenPaths_;
};

TEST_F(LinuxSysfsMicrotest, GetPciPath_LowercasesTheBusIdAndResolves) {
  std::string dev = AddPci("0000:1a:00.0");
  char* path = nullptr;
  ASSERT_EQ(ncclOsGetPciPath("0000:1A:00.0", &path), ncclSuccess);
  EXPECT_EQ(std::string(path), dev);
  free(path);
  ASSERT_EQ(seenPaths_.size(), 1u);
  EXPECT_EQ(seenPaths_[0], "/sys/class/pci_bus/0000:1a/../../0000:1a:00.0");
}

TEST_F(LinuxSysfsMicrotest, GetPciPath_MissingDeviceIsASystemError) {
  char* path = (char*)0x1;
  EXPECT_EQ(ncclOsGetPciPath("0000:1b:00.0", &path), ncclSystemError);
  EXPECT_EQ(path, nullptr);
  EXPECT_TRUE(LinuxLogged("WARN Could not find real path of /sys/class/pci_bus/0000:1b/../../0000:1b:00.0"));
}

TEST_F(LinuxSysfsMicrotest, TopoGetStrFromSys_DropsTheLastCharacter) {
  Mkdirs("/d");
  Write("/d/nl", "0x030000\n");
  Write("/d/raw", "abc");
  Write("/d/empty", "");
  char buf[16];
  std::string dir = root_ + "/d";
  ASSERT_EQ(ncclOsTopoGetStrFromSys(dir.c_str(), "nl", buf, sizeof(buf)), ncclSuccess);
  EXPECT_STREQ(buf, "0x030000");
  // Observed: a value without a trailing newline loses its last character.
  ASSERT_EQ(ncclOsTopoGetStrFromSys(dir.c_str(), "raw", buf, sizeof(buf)), ncclSuccess);
  EXPECT_STREQ(buf, "ab");
  ASSERT_EQ(ncclOsTopoGetStrFromSys(dir.c_str(), "nl", buf, 4), ncclSuccess);  // truncated to maxLen
  EXPECT_STREQ(buf, "0x0");
  EXPECT_FALSE(LinuxLogged("could not read"));
  ASSERT_EQ(ncclOsTopoGetStrFromSys(dir.c_str(), "empty", buf, sizeof(buf)), ncclSuccess);
  EXPECT_STREQ(buf, "");
  EXPECT_TRUE(LinuxLogged(("could not read " + dir + "/empty").c_str()));  // empty reads as unreadable
  ASSERT_EQ(ncclOsTopoGetStrFromSys(dir.c_str(), "missing", buf, sizeof(buf)), ncclSuccess);
  EXPECT_STREQ(buf, "");
  EXPECT_TRUE(LinuxLogged(("INFO Topology detection : could not read " + dir + "/missing, ignoring").c_str()));
}

TEST_F(LinuxSysfsMicrotest, DeviceClass_ReadsTheClassAttribute) {
  std::string dev = AddPci("0000:1a:00.0");
  FILE* f = ::fopen((dev + "/class").c_str(), "w");
  fputs("0x038000\n", f);
  fclose(f);
  char cls[32] = "junk";
  ASSERT_EQ(ncclOsGetPciDeviceClassByBusId("0000:1a:00.0", cls, sizeof(cls)), ncclSuccess);
  EXPECT_STREQ(cls, "0x038000");
}

TEST_F(LinuxSysfsMicrotest, DeviceClass_MissingDeviceIsAQuietSystemError) {
  char cls[32] = "junk";
  EXPECT_EQ(ncclOsGetPciDeviceClassByBusId("0000:1b:00.0", cls, sizeof(cls)), ncclSystemError);
  EXPECT_STREQ(cls, "");
  EXPECT_FALSE(LinuxLogged("WARN"));
  EXPECT_TRUE(LinuxLogged("INFO Could not find real path"));
}

TEST_F(LinuxSysfsMicrotest, ComputePartition_ReadsTheAttributeOrEmpty) {
  std::string dev = AddPci("0000:1a:00.0");
  char part[16] = "junk";
  ASSERT_EQ(ncclOsGetPciDeviceComputePartitionByBusId("0000:1a:00.0", part, sizeof(part)), ncclSuccess);
  EXPECT_STREQ(part, "");  // attribute absent
  FILE* f = ::fopen((dev + "/current_compute_partition").c_str(), "w");
  fputs("CPX\n", f);
  fclose(f);
  ASSERT_EQ(ncclOsGetPciDeviceComputePartitionByBusId("0000:1a:00.0", part, sizeof(part)), ncclSuccess);
  EXPECT_STREQ(part, "CPX");
}

TEST_F(LinuxSysfsMicrotest, ComputePartition_BadArgumentsAndMissingDevice) {
  char part[16] = "junk";
  EXPECT_EQ(ncclOsGetPciDeviceComputePartitionByBusId("0000:1a:00.0", nullptr, 16), ncclInvalidArgument);
  EXPECT_EQ(ncclOsGetPciDeviceComputePartitionByBusId("0000:1a:00.0", part, 0), ncclInvalidArgument);
  EXPECT_STREQ(part, "junk");
  EXPECT_EQ(ncclOsGetPciDeviceComputePartitionByBusId("0000:1b:00.0", part, sizeof(part)), ncclSystemError);
  EXPECT_STREQ(part, "");
  EXPECT_FALSE(LinuxLogged("WARN"));
}

TEST_F(LinuxSysfsMicrotest, BcmLinks_ListsPeersThatArePciDevices) {
  AddPci("0000:2b:00.0");
  AddPci("0000:2c:00.0");
  std::string dir = "/sys/kernel/pci_switch_link/virtual_switch_links/0000:1a:00.0";
  Mkdirs(dir + "/0000:2b:00.0");
  Mkdirs(dir + "/0000:2c:00.0");
  Mkdirs(dir + "/0000:3d:00.0");  // no such PCI device
  Mkdirs(dir + "/short");
  int n = -1;
  char* peers = nullptr;
  ASSERT_EQ(ncclOsGetBcmLinks("0000:1A:00.0", &n, &peers), ncclSuccess);
  ASSERT_EQ(n, 2);
  std::set<std::string> got{peers, peers + sizeof("0000:00:00.0")};
  EXPECT_EQ(got, (std::set<std::string>{"0000:2b:00.0", "0000:2c:00.0"}));
  free(peers);
}

TEST_F(LinuxSysfsMicrotest, BcmLinks_NoSwitchDirectoryMeansNoLinks) {
  int n = -1;
  char* peers = (char*)0x1;
  ASSERT_EQ(ncclOsGetBcmLinks("0000:1a:00.0", &n, &peers), ncclSuccess);
  EXPECT_EQ(n, 0);
  EXPECT_EQ(peers, nullptr);
}

TEST_F(LinuxSysfsMicrotest, NumaNodeAffinity_StripsLineEndings) {
  Mkdirs("/sys/devices/system/node/node3");
  Write("/sys/devices/system/node/node3/cpumap", "00000000,0000ffff\r\n");
  char str[64];
  int offset = -1;
  ASSERT_EQ(ncclOsGetNumaNodeAffinity(3, str, sizeof(str), &offset), ncclSuccess);
  EXPECT_STREQ(str, "00000000,0000ffff");
  EXPECT_EQ(offset, 0);
  ASSERT_EQ(ncclOsGetNumaNodeAffinity(3, str, 6, &offset), ncclSuccess);  // maxLen counts the NUL
  EXPECT_STREQ(str, "00000");
  EXPECT_FALSE(LinuxLogged("could not read"));
}

TEST_F(LinuxSysfsMicrotest, NumaNodeAffinity_MissingNodeIsEmpty) {
  char str[64] = "junk";
  int offset = -1;
  ASSERT_EQ(ncclOsGetNumaNodeAffinity(7, str, sizeof(str), &offset), ncclSuccess);
  EXPECT_STREQ(str, "");
  EXPECT_TRUE(LinuxLogged("could not read /sys/devices/system/node/node7/cpumap, using empty affinity"));
}

// Shared memory, on real files.

class LinuxShmMicrotest : public LinuxMicrotest {
 protected:
  void SetUp() override {
    LinuxMicrotest::SetUp();
    char tmpl[] = "/tmp/nccl-linux-shm-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    dir_ = tmpl;
  }
  void TearDown() override {
    for (auto* h : handles_) ncclOsShmClose(h);
    std::string cmd = "rm -rf " + dir_;
    EXPECT_EQ(system(cmd.c_str()), 0);
    LinuxMicrotest::TearDown();
  }
  ncclResult_t Open(char* path, size_t size, int refcount, void** ptr, ncclShmHandleInternal** h,
                    void** dev = nullptr) {
    ncclResult_t ret = ncclOsShmOpen(path, PATH_MAX, size, ptr, dev, refcount, h);
    if (*h) handles_.push_back(*h);
    return ret;
  }
  void Forget(ncclShmHandleInternal* h) { handles_.erase(std::find(handles_.begin(), handles_.end(), h)); }
  std::string dir_;
  std::vector<ncclShmHandleInternal*> handles_;
};

TEST_F(LinuxShmMicrotest, CreateInDevShmThenAttachUntilTheLastPeerUnlinks) {
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 4096, /*refcount=*/2, &ptr, &h), ncclSuccess);
  EXPECT_EQ(std::string(path).rfind("/dev/shm/nccl-", 0), 0u);
  EXPECT_TRUE(Exists(path));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(h->shmPtr, ptr);
  EXPECT_EQ(h->shmSize, 4096u);
  EXPECT_EQ(h->realShmSize, 4096u + sizeof(int));
  EXPECT_EQ(h->refcount, (int*)((char*)ptr + 4096));
  EXPECT_EQ(*h->refcount, 2);
  EXPECT_STREQ(h->shmPath, path);
  EXPECT_NE(h->shmPath, path);  // a copy
  EXPECT_EQ(((char*)ptr)[0], 0);
  std::strcpy((char*)ptr, "shared");

  void* peer1 = nullptr;
  ncclShmHandleInternal* h1 = nullptr;
  ASSERT_EQ(Open(path, 4096, 0, &peer1, &h1), ncclSuccess);
  EXPECT_STREQ((char*)peer1, "shared");
  EXPECT_EQ(h1->shmPath, nullptr);
  EXPECT_EQ(*h->refcount, 1);
  EXPECT_TRUE(Exists(path));

  void* peer2 = nullptr;
  ncclShmHandleInternal* h2 = nullptr;
  ASSERT_EQ(Open(path, 4096, 0, &peer2, &h2), ncclSuccess);
  EXPECT_EQ(*h->refcount, 0);
  EXPECT_FALSE(Exists(path));  // every peer attached

  EXPECT_EQ(ncclOsShmClose(h), ncclSuccess);  // refcount 0: no second unlink
  Forget(h);
  EXPECT_FALSE(LinuxLogged("WARN"));
}

TEST_F(LinuxShmMicrotest, CreateAtAGivenPath) {
  std::string p = dir_ + "/seg";
  char path[PATH_MAX];
  std::strcpy(path, p.c_str());
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 100, 1, &ptr, &h), ncclSuccess);
  EXPECT_STREQ(path, p.c_str());
  struct stat st;
  ASSERT_EQ(stat(path, &st), 0);
  EXPECT_EQ((size_t)st.st_size, 100 + sizeof(int));
  EXPECT_EQ(st.st_mode & 0777, (mode_t)(S_IRUSR | S_IWUSR));
  EXPECT_TRUE(LinuxLogged("Allocated 104 bytes of shared memory in"));
}

// open() does not truncate, so a creator reusing a path must clear what the file held.
TEST_F(LinuxShmMicrotest, CreateClearsAReusedFile) {
  std::string p = dir_ + "/seg";
  FILE* f = ::fopen(p.c_str(), "w");
  fputs("stale contents", f);
  fclose(f);
  char path[PATH_MAX];
  std::strcpy(path, p.c_str());
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h), ncclSuccess);
  EXPECT_EQ(std::string((char*)ptr, 14), std::string(14, '\0'));
}

// Closing a creator handle while peers have not attached yet removes the file (abort cleanup).
TEST_F(LinuxShmMicrotest, CloseUnlinksWhilePeersAreStillExpected) {
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h), ncclSuccess);
  Forget(h);
  EXPECT_EQ(ncclOsShmClose(h), ncclSuccess);
  EXPECT_FALSE(Exists(path));
  EXPECT_EQ(ncclOsShmClose(nullptr), ncclSuccess);
}

TEST_F(LinuxShmMicrotest, CloseReportsUnlinkAndMunmapFailures) {
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h), ncclSuccess);
  ::unlink(path);
  lx::munmap = [](void* p, size_t n) {
    ::munmap(p, n);
    errno = EINVAL;
    return -1;
  };
  Forget(h);
  EXPECT_EQ(ncclOsShmClose(h), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN unlink shared memory"));
  EXPECT_TRUE(LinuxLogged("WARN munmap of shared memory"));
}

TEST_F(LinuxShmMicrotest, UnlinkRemovesTheFileOnce) {
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h), ncclSuccess);
  ASSERT_EQ(ncclOsShmUnlink(h), ncclSuccess);
  EXPECT_FALSE(Exists(path));
  EXPECT_EQ(h->shmPath, nullptr);
  EXPECT_EQ(ncclOsShmUnlink(h), ncclSuccess);  // nothing left to unlink
  EXPECT_EQ(ncclOsShmUnlink(nullptr), ncclSuccess);
  EXPECT_FALSE(LinuxLogged("WARN"));
}

TEST_F(LinuxShmMicrotest, UnlinkSkipsAFullyAttachedSegmentAndReportsFailures) {
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h), ncclSuccess);
  *h->refcount = 0;
  EXPECT_EQ(ncclOsShmUnlink(h), ncclSuccess);
  EXPECT_TRUE(Exists(path));
  *h->refcount = 1;
  ::unlink(path);
  EXPECT_EQ(ncclOsShmUnlink(h), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("WARN unlink shared memory"));
}

TEST_F(LinuxShmMicrotest, OpenFailuresReturnNoHandle) {
  std::string missing = dir_ + "/no/such/dir/seg";
  char path[PATH_MAX];
  std::strcpy(path, missing.c_str());
  void* ptr = (void*)0x1;
  ncclShmHandleInternal* h = (ncclShmHandleInternal*)0x1;
  EXPECT_EQ(Open(path, 64, 1, &ptr, &h), ncclSystemError);
  EXPECT_EQ(h, nullptr);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_TRUE(LinuxLogged("Error while creating shared memory segment"));
  EXPECT_EQ(Open(path, 64, 0, &ptr, &h), ncclSystemError);
  EXPECT_TRUE(LinuxLogged("Error while attaching to shared memory segment"));
}

TEST_F(LinuxShmMicrotest, InterruptedMkstempAndFallocateAreRetried) {
  int mkstemps = 0, fallocates = 0;
  lx::mkstemp = [&](char* tmpl) {
    if (mkstemps++ == 0) {
      errno = EINTR;
      return -1;
    }
    return ::mkstemp(tmpl);
  };
  lx::fallocate = [&](int fd, int mode, off_t off, off_t len) {
    if (fallocates++ == 0) {
      errno = EINTR;
      return -1;
    }
    return ::fallocate(fd, mode, off, len);
  };
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h), ncclSuccess);
  EXPECT_EQ(mkstemps, 2);
  EXPECT_EQ(fallocates, 2);
  EXPECT_TRUE(LinuxLogged("mkstemp: Failed to create"));
  EXPECT_TRUE(LinuxLogged("fallocate: Failed to extend"));
}

TEST_F(LinuxShmMicrotest, MkstempFailure) {
  lx::mkstemp = [](char*) {
    errno = EACCES;
    return -1;
  };
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  EXPECT_EQ(Open(path, 64, 1, &ptr, &h), ncclSystemError);
  EXPECT_EQ(h, nullptr);
  EXPECT_TRUE(LinuxLogged("WARN Error: failed to create shared memory file /dev/shm/nccl-XXXXXX"));
}

// Observed: a segment that fails after the file was created leaves the file behind, because close
// only unlinks once a refcount has been mapped.
TEST_F(LinuxShmMicrotest, FallocateFailureLeavesTheCreatedFile) {
  lx::fallocate = [](int, int, off_t, off_t) {
    errno = ENOSPC;
    return -1;
  };
  std::string p = dir_ + "/seg";
  char path[PATH_MAX];
  std::strcpy(path, p.c_str());
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  EXPECT_EQ(Open(path, 64, 1, &ptr, &h), ncclSystemError);
  EXPECT_EQ(h, nullptr);
  EXPECT_TRUE(LinuxLogged("WARN Error: failed to extend"));
  EXPECT_TRUE(Exists(p));
}

TEST_F(LinuxShmMicrotest, MmapAndCallocFailures) {
  lx::mmap = [](void*, size_t, int, int, int, off_t) {
    errno = ENOMEM;
    return MAP_FAILED;
  };
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  EXPECT_EQ(Open(path, 64, 1, &ptr, &h), ncclSystemError);
  EXPECT_EQ(h, nullptr);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_TRUE(LinuxLogged("WARN Error: Could not map"));
  ::unlink(path);

  lx::calloc = [](size_t, size_t) -> void* { return nullptr; };
  path[0] = '\0';
  EXPECT_EQ(Open(path, 64, 1, &ptr, &h), ncclSystemError);
  EXPECT_EQ(h, nullptr);
  EXPECT_STREQ(path, "");  // failed before creating anything
}

TEST_F(LinuxShmMicrotest, LastAttachUnlinkFailureIsOnlyLogged) {
  char path[PATH_MAX] = "";
  void* ptr = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h), ncclSuccess);
  lx::unlink = [](const char*) {
    errno = EBUSY;
    return -1;
  };
  void* peer = nullptr;
  ncclShmHandleInternal* hp = nullptr;
  EXPECT_EQ(Open(path, 64, 0, &peer, &hp), ncclSuccess);
  EXPECT_TRUE(LinuxLogged("INFO unlink shared memory"));
  lx::unlink = ::unlink;
  ::unlink(path);
}

TEST_F(LinuxShmMicrotest, DevicePointerRegistersTheMappingWithHip) {
  std::vector<std::string> calls;
  unsigned int regFlags = 0;
  lx::hipExchangeCaptureMode = [&](hipStreamCaptureMode* m) {
    calls.push_back("exchange");
    EXPECT_EQ(*m, hipStreamCaptureModeRelaxed);
    return hipSuccess;
  };
  lx::hipHostRegister = [&](void*, size_t size, unsigned int flags) {
    calls.push_back("register");
    regFlags = flags;
    EXPECT_EQ(size, 64 + sizeof(int));
    return hipSuccess;
  };
  lx::hipHostGetDevicePointer = [&](void** d, void*, unsigned int) {
    calls.push_back("devptr");
    *d = (void*)0xd00d;
    return hipSuccess;
  };
  lx::hipHostUnregister = [&](void*) {
    calls.push_back("unregister");
    return hipSuccess;
  };
  char path[PATH_MAX] = "";
  void *ptr = nullptr, *dev = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h, &dev), ncclSuccess);
  EXPECT_EQ(dev, (void*)0xd00d);
  EXPECT_EQ(h->devShmPtr, (void*)0xd00d);
  EXPECT_EQ(regFlags & (hipHostRegisterPortable | hipHostRegisterMapped),
            (unsigned)(hipHostRegisterPortable | hipHostRegisterMapped));
  Forget(h);
  EXPECT_EQ(ncclOsShmClose(h), ncclSuccess);
  EXPECT_EQ(calls, (std::vector<std::string>{"exchange", "register", "devptr", "exchange", "unregister"}));
}

TEST_F(LinuxShmMicrotest, HipFailuresUnwindTheSegment) {
  for (int step = 0; step < 4; step++) {
    SCOPED_TRACE(step);
    int exchanges = 0;
    lx::Reset();
    if (step == 0) lx::hipExchangeCaptureMode = [](hipStreamCaptureMode*) { return hipErrorInvalidValue; };
    if (step == 1) lx::hipHostRegister = [](void*, size_t, unsigned int) { return hipErrorInvalidValue; };
    if (step == 2) lx::hipHostGetDevicePointer = [](void**, void*, unsigned int) { return hipErrorInvalidValue; };
    if (step == 3)
      lx::hipExchangeCaptureMode = [&](hipStreamCaptureMode*) {
        return exchanges++ ? hipErrorInvalidValue : hipSuccess;
      };
    char path[PATH_MAX] = "";
    void *ptr = nullptr, *dev = (void*)0x1;
    ncclShmHandleInternal* h = nullptr;
    EXPECT_EQ(Open(path, 64, 1, &ptr, &h, &dev), ncclUnhandledCudaError);
    EXPECT_EQ(h, nullptr);
    EXPECT_EQ(dev, nullptr);
    EXPECT_FALSE(Exists(path));
  }
}

TEST_F(LinuxShmMicrotest, CloseReportsAHipUnregisterFailure) {
  lx::hipHostUnregister = [](void*) { return hipErrorInvalidValue; };
  char path[PATH_MAX] = "";
  void *ptr = nullptr, *dev = nullptr;
  ncclShmHandleInternal* h = nullptr;
  ASSERT_EQ(Open(path, 64, 1, &ptr, &h, &dev), ncclSuccess);
  Forget(h);
  void* mapping = h->shmPtr;
  size_t size = h->realShmSize;
  EXPECT_EQ(ncclOsShmClose(h), ncclUnhandledCudaError);  // returns before munmap and free
  ::munmap(mapping, size);
}

TEST_F(LinuxShmMicrotest, HandleInitWithoutAMapping) {
  ncclShmHandleInternal h;
  char path[] = "/x";
  ncclOsShmHandleInit(9, path, 10, 14, nullptr, nullptr, /*create=*/false, &h);
  EXPECT_EQ(h.shmDesc, 9);
  EXPECT_EQ(h.refcount, nullptr);
  EXPECT_EQ(h.shmPath, nullptr);
  ncclOsShmHandleInit(9, path, 10, 14, nullptr, nullptr, /*create=*/true, &h);
  EXPECT_STREQ(h.shmPath, "/x");
  free(h.shmPath);
}

}  // namespace
