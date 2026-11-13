/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/libc_fakes.h"

// Puts the seam's micro_* prototypes in scope so the compiler checks them against the definitions at the bottom of
// this file. Without it the two lists are hand-maintained and both extern "C", so a drifted parameter type would link
// cleanly and corrupt arguments at run time. Include the undef half immediately: this file's defaults call real libc.
#include "fakes/libc_seam.h"
#include "fakes/libc_seam_undef.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>

#include "signature-drift.h"

// LogCapture.hpp's ncclDebugLevelMask/ncclDebugMask come from fakes/nccl_fakes.cc,
// which this binary already links. A libc-only unit reports via plain
// fprintf(stderr), so CaptureLog works without raising the level.

std::string g_writtenData;
std::string g_stdoutData;
std::vector<MicroFwriteCall> g_fwriteCalls;
std::vector<MicroPerrorCall> g_perrorCalls;
std::vector<int> g_closedFds;
std::vector<void*> g_freedPointers;
std::vector<void (*)(void)> g_atexitFns;
std::vector<FILE*> g_fprintfCalls;
std::vector<int> g_writtenFds;
std::vector<int> g_readFds;
std::vector<MicroReadStep> g_readScript;
size_t g_readScriptPos = 0;
std::vector<int> g_boundFds;
std::vector<int> g_listenedFds;
std::vector<int> g_acceptedFds;
std::vector<MicroFcntlCall> g_fcntlCalls;
std::vector<int> g_recvFds;
std::vector<int> g_recvFlags;
std::vector<MicroReadStep> g_recvScript;
size_t g_recvScriptPos = 0;
std::vector<int> g_sentFds;
std::vector<int> g_sendFlags;
std::string g_sentData;
int g_nextSocketFd = 42;
int g_socketFailErrno = EAFNOSUPPORT;
int g_bindResult = 0;
int g_bindErrno = EADDRINUSE;
int g_listenResult = 0;
int g_listenErrno = EADDRINUSE;
int g_nextAcceptFd = 44;
int g_acceptErrno = ECONNABORTED;
int g_fcntlResult = 0;
int g_fcntlErrno = EBADF;
int g_fcntlGetFlags = 0;
int g_lastSetsockoptLevel = -1;
int g_lastSetsockoptOptname = -1;
struct timeval g_lastSetsockoptTimeval = {-1, -1};
int g_getaddrinfoResult = 0;
int g_addrinfoCount = 1;
// Arbitrary non-privileged base; a unit that asserts on the port should set it.
int g_addrinfoBasePort = 28028;
int g_freeaddrinfoCalls = 0;
int g_connectResult = 0;
int g_connectErrno = ECONNREFUSED;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

static ssize_t DefaultWrite(int fd, const void* buf, size_t count) {
  g_writtenFds.push_back(fd);
  g_writtenData.append(static_cast<const char*>(buf), count);
  return static_cast<ssize_t>(count);
}

ssize_t DeliverReadStep(const MicroReadStep& step, void* buf, size_t count) {
  // A failing or EOF step delivers nothing, so a payload on one is a script the fake would silently ignore.
  assert((step.ret > 0 || step.data.empty()) &&
         "DeliverReadStep: a non-positive ret cannot deliver data; drop the payload or make ret positive");
  if (step.ret < 0) {
    errno = step.err;
    return step.ret;
  }
  if (step.ret == 0) return 0;
  // A positive ret is the byte count the step promises. Assert the script means what it says, and clamp to the promise
  // as well so a step of ret=3 with 8 bytes of data cannot hand over eight bytes in a Release build, where NDEBUG
  // drops the assert. Without both, a mismatched script silently pins the fake's behaviour instead of the unit's.
  assert(static_cast<size_t>(step.ret) == step.data.size() &&
         "DeliverReadStep: a positive ret must equal data.size(); use ScriptReadData to derive it");
  const size_t promised = std::min(static_cast<size_t>(step.ret), step.data.size());
  const size_t n = std::min(promised, count);
  std::memcpy(buf, step.data.data(), n);
  return static_cast<ssize_t>(n);
}

static ssize_t DefaultRead(int fd, void* buf, size_t count) {
  g_readFds.push_back(fd);
  // A zero-length read returns 0 without consuming a step, as read(2) does and as RecordingReader does. rasRead asks
  // for 0 once its buffer is full, so spending a step there would shift every later step onto the wrong call.
  if (count == 0) return 0;
  if (g_readScriptPos >= g_readScript.size()) return 0;  // EOF past the end of the script
  const MicroReadStep& step = g_readScript[g_readScriptPos++];
  return DeliverReadStep(step, buf, count);
}

static int DefaultClose(int fd) {
  g_closedFds.push_back(fd);
  return 0;
}

static int DefaultSocket(int, int, int) {
  if (g_nextSocketFd == -1) errno = g_socketFailErrno;
  return g_nextSocketFd;
}

static int DefaultConnect(int, const struct sockaddr*, socklen_t) {
  if (g_connectResult != 0) errno = g_connectErrno;
  return g_connectResult;
}

static int DefaultBind(int fd, const struct sockaddr*, socklen_t) {
  g_boundFds.push_back(fd);
  if (g_bindResult != 0) errno = g_bindErrno;
  return g_bindResult;
}

static int DefaultListen(int fd, int) {
  g_listenedFds.push_back(fd);
  if (g_listenResult != 0) errno = g_listenErrno;
  return g_listenResult;
}

static int DefaultAccept(int fd, struct sockaddr* addr, socklen_t* addrlen) {
  g_acceptedFds.push_back(fd);
  if (g_nextAcceptFd == -1) {
    errno = g_acceptErrno;
    return -1;
  }
  if (addr && addrlen && *addrlen >= static_cast<socklen_t>(sizeof(struct sockaddr_in))) {
    struct sockaddr_in accepted = {};
    accepted.sin_family = AF_INET;
    accepted.sin_port = htons(0);
    accepted.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    std::memcpy(addr, &accepted, sizeof(accepted));
    *addrlen = sizeof(accepted);
  }
  return g_nextAcceptFd;
}

static int DefaultFcntl(int fd, int cmd, int arg) {
  g_fcntlCalls.push_back(MicroFcntlCall{fd, cmd, arg});
  if (g_fcntlResult == -1) errno = g_fcntlErrno;
  return cmd == F_GETFL && g_fcntlResult != -1 ? g_fcntlGetFlags : g_fcntlResult;
}

static ssize_t DefaultRecv(int fd, void* buf, size_t count, int flags) {
  g_recvFds.push_back(fd);
  g_recvFlags.push_back(flags);
  if (count == 0) return 0;
  if (g_recvScriptPos >= g_recvScript.size()) return 0;
  return DeliverReadStep(g_recvScript[g_recvScriptPos++], buf, count);
}

static ssize_t DefaultSend(int fd, const void* buf, size_t count, int flags) {
  g_sentFds.push_back(fd);
  g_sendFlags.push_back(flags);
  g_sentData.append(static_cast<const char*>(buf), count);
  return static_cast<ssize_t>(count);
}

static int DefaultSetsockopt(int, int level, int optname, const void* optval, socklen_t optlen) {
  g_lastSetsockoptLevel = level;
  g_lastSetsockoptOptname = optname;
  if (optval && optlen >= static_cast<socklen_t>(sizeof(struct timeval))) {
    std::memcpy(&g_lastSetsockoptTimeval, optval, sizeof(struct timeval));
  }
  return 0;
}

// Builds g_addrinfoCount single-linked IPv4 entries. Paired with
// DefaultFreeaddrinfo; an override of either half must use the same allocator.
// Deliberately unlike getaddrinfo(3) in one respect: g_addrinfoCount == 0
// returns success with an empty list, where the real call returns EAI_NONAME.
// That is how a test reaches a caller's address-walk with no candidate at all
// (see ConnectFailLabel_SockNeverOpened_...); set g_getaddrinfoResult for the
// resolver-failed path instead.
static int DefaultGetaddrinfo(const char*, const char*, const struct addrinfo*, struct addrinfo** res) {
  if (g_getaddrinfoResult != 0) return g_getaddrinfoResult;
  struct addrinfo* head = nullptr;
  struct addrinfo** tail = &head;
  for (int i = 0; i < g_addrinfoCount; ++i) {
    auto* ai = static_cast<struct addrinfo*>(std::calloc(1, sizeof(struct addrinfo)));
    auto* sa = static_cast<struct sockaddr_in*>(std::calloc(1, sizeof(struct sockaddr_in)));
    // Stop at the short list rather than dereferencing null; the unit still gets a well-formed chain of what was built.
    if (ai == nullptr || sa == nullptr) {
      std::free(ai);
      std::free(sa);
      break;
    }
    sa->sin_family = AF_INET;
    sa->sin_port = htons(static_cast<uint16_t>(g_addrinfoBasePort + i));
    sa->sin_addr.s_addr = htonl(INADDR_LOOPBACK + i);
    ai->ai_family = AF_INET;
    ai->ai_socktype = SOCK_STREAM;
    ai->ai_protocol = IPPROTO_TCP;
    ai->ai_addr = reinterpret_cast<struct sockaddr*>(sa);
    ai->ai_addrlen = sizeof(struct sockaddr_in);
    *tail = ai;
    tail = &ai->ai_next;
  }
  *res = head;
  return 0;
}

static void DefaultFreeaddrinfo(struct addrinfo* ai) {
  ++g_freeaddrinfoCalls;
  while (ai) {
    struct addrinfo* next = ai->ai_next;
    std::free(ai->ai_addr);
    std::free(ai);
    ai = next;
  }
}

// IPv4 only, and says so rather than guessing: on an AF_INET6 address sin_addr overlays sin6_flowinfo, so an
// unconditional cast would report success with a bogus dotted quad to the next unit that hands this seam a v6 sockaddr.
static int DefaultGetnameinfo(const struct sockaddr* sa, socklen_t salen, char* host, socklen_t hostlen, char* serv,
                              socklen_t servlen, int) {
  if (sa == nullptr || sa->sa_family != AF_INET || salen < static_cast<socklen_t>(sizeof(struct sockaddr_in))) {
    return EAI_FAMILY;
  }
  const auto* in = reinterpret_cast<const struct sockaddr_in*>(sa);
  if (host && hostlen > 0) {
    if (!inet_ntop(AF_INET, &in->sin_addr, host, hostlen)) return EAI_OVERFLOW;
  }
  if (serv && servlen > 0) snprintf(serv, servlen, "%u", ntohs(in->sin_port));
  return 0;
}

static const char* DefaultGaiStrerror(int code) { return gai_strerror(code); }

// Records the shape of each call, not just the bytes: g_stdoutData reads identically whether the unit chose stdout or
// stderr, and identically for fwrite(buf,1,n) and fwrite(buf,n,1). Recording here rather than in a per-test hook means
// every unit sharing this seam gets the same oracle.
static size_t DefaultFwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
  g_fwriteCalls.push_back(MicroFwriteCall{size, nmemb, stream});
  g_stdoutData.append(static_cast<const char*>(ptr), size * nmemb);
  return nmemb;
}

static int DefaultFflush(FILE*) { return 0; }

// Records the prefix and the errno the unit branched on, then forwards to the real perror so stderr still carries the
// message. Tests assert on the record: the errno is the branch condition, where the sentence is only its wording.
static void DefaultPerror(const char* prefix) {
  const int err = errno;  // read before the std::string below can allocate and perturb it
  g_perrorCalls.push_back(MicroPerrorCall{prefix ? prefix : "", err});
  ::perror(prefix);
}

static void DefaultExit(int status) { throw MicroExit{status}; }

static int DefaultGethostname(char* name, size_t len) { return ::gethostname(name, len); }

static int DefaultAccess(const char* path, int mode) { return ::access(path, mode); }

static void* DefaultMalloc(size_t size) { return std::malloc(size); }

static void* DefaultCalloc(size_t nmemb, size_t size) { return std::calloc(nmemb, size); }

static void DefaultFree(void* ptr) {
  g_freedPointers.push_back(ptr);
  std::free(ptr);
}

// Fails rather than forwarding: the default read and close are fakes, and real files would make results
// host-dependent.
static int DefaultOpen(const char*, int, mode_t) {
  errno = ENOENT;
  return -1;
}

// The filesystem and ioctl defaults below fail too, so a unit never sees the host.
static FILE* DefaultFopen(const char*, const char*) {
  errno = ENOENT;
  return nullptr;
}

static DIR* DefaultOpendir(const char*) {
  errno = ENOENT;
  return nullptr;
}

// Only reached with a DIR* a test hook handed out, so forward to real libc.
static struct dirent* DefaultReaddir(DIR* dir) { return ::readdir(dir); }
static int DefaultClosedir(DIR* dir) { return ::closedir(dir); }

static ssize_t DefaultReadlink(const char*, char*, size_t) {
  errno = ENOENT;
  return -1;
}

static int DefaultIoctl(int, unsigned long, void*) {
  errno = ENOTTY;
  return -1;
}

static int DefaultAtexit(void (*fn)(void)) {
  g_atexitFns.push_back(fn);
  return 0;
}

static int DefaultClockGettime(clockid_t clk, struct timespec* ts) { return ::clock_gettime(clk, ts); }
static int DefaultPosixMemalign(void** ptr, size_t alignment, size_t size) {
  return ::posix_memalign(ptr, alignment, size);
}
static int DefaultPthreadCreate(pthread_t* thread, const pthread_attr_t* attr, void* (*fn)(void*), void* arg) {
  return ::pthread_create(thread, attr, fn, arg);
}
static int DefaultNanosleep(const struct timespec* req, struct timespec* rem) { return ::nanosleep(req, rem); }

// ---------------------------------------------------------------------------

std::function<ssize_t(int, const void*, size_t)> g_write = DefaultWrite;
std::function<ssize_t(int, void*, size_t)> g_read = DefaultRead;
std::function<int(int)> g_close = DefaultClose;
std::function<int(int, int, int)> g_socket = DefaultSocket;
std::function<int(int, const struct sockaddr*, socklen_t)> g_connect = DefaultConnect;
std::function<int(int, const struct sockaddr*, socklen_t)> g_bind = DefaultBind;
std::function<int(int, int)> g_listen = DefaultListen;
std::function<int(int, struct sockaddr*, socklen_t*)> g_accept = DefaultAccept;
std::function<int(int, int, int)> g_fcntl = DefaultFcntl;
std::function<ssize_t(int, void*, size_t, int)> g_recv = DefaultRecv;
std::function<ssize_t(int, const void*, size_t, int)> g_send = DefaultSend;
std::function<int(int, int, int, const void*, socklen_t)> g_setsockopt = DefaultSetsockopt;
std::function<int(const char*, const char*, const struct addrinfo*, struct addrinfo**)> g_getaddrinfo =
    DefaultGetaddrinfo;
std::function<void(struct addrinfo*)> g_freeaddrinfo = DefaultFreeaddrinfo;
std::function<int(const struct sockaddr*, socklen_t, char*, socklen_t, char*, socklen_t, int)> g_getnameinfo =
    DefaultGetnameinfo;
std::function<const char*(int)> g_gaiStrerror = DefaultGaiStrerror;
std::function<size_t(const void*, size_t, size_t, FILE*)> g_fwrite = DefaultFwrite;
std::function<int(FILE*)> g_fflush = DefaultFflush;
std::function<void(const char*)> g_perror = DefaultPerror;
std::function<void(int)> g_exit = DefaultExit;
std::function<int(char*, size_t)> g_gethostname = DefaultGethostname;
std::function<int(const char*, int)> g_access = DefaultAccess;
std::function<int(const char*, int, mode_t)> g_open = DefaultOpen;
std::function<void*(size_t)> g_malloc = DefaultMalloc;
std::function<void*(size_t, size_t)> g_calloc = DefaultCalloc;
std::function<void(void*)> g_free = DefaultFree;
std::function<FILE*(const char*, const char*)> g_fopen = DefaultFopen;
std::function<DIR*(const char*)> g_opendir = DefaultOpendir;
std::function<struct dirent*(DIR*)> g_readdir = DefaultReaddir;
std::function<int(DIR*)> g_closedir = DefaultClosedir;
std::function<ssize_t(const char*, char*, size_t)> g_readlink = DefaultReadlink;
std::function<int(int, unsigned long, void*)> g_ioctl = DefaultIoctl;
std::function<int(void (*)(void))> g_atexit = DefaultAtexit;
std::function<int(clockid_t, struct timespec*)> g_clockGettime = DefaultClockGettime;
std::function<int(void**, size_t, size_t)> g_posixMemalign = DefaultPosixMemalign;
std::function<int(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*)> g_pthreadCreate =
    DefaultPthreadCreate;
std::function<int(const struct timespec*, struct timespec*)> g_nanosleep = DefaultNanosleep;

static int RealFcntl(int fd, int cmd, int arg) { return ::fcntl(fd, cmd, arg); }
static void RealExit(int status) { ::exit(status); }

std::function<int(struct pollfd*, nfds_t, int)> g_poll = ::poll;
std::function<int(int, int, int, void*, socklen_t*)> g_getsockopt = ::getsockopt;
std::function<int(int, struct sockaddr*, socklen_t*)> g_getsockname = ::getsockname;
std::function<int(int, int)> g_shutdown = ::shutdown;
std::function<ssize_t(int, struct msghdr*, int)> g_recvmsg = ::recvmsg;
std::function<ssize_t(int, const struct msghdr*, int)> g_sendmsg = ::sendmsg;
std::function<int(int*)> g_pipe = ::pipe;
std::function<int(struct ifaddrs**)> g_getifaddrs = ::getifaddrs;
std::function<void(struct ifaddrs*)> g_freeifaddrs = ::freeifaddrs;
std::function<int(int, struct rlimit*)> g_getrlimit = ::getrlimit;
std::function<int(int, const struct rlimit*)> g_setrlimit = ::setrlimit;
std::function<int(pid_t, size_t, cpu_set_t*)> g_schedGetaffinity = ::sched_getaffinity;
std::function<char*(const char*, char*)> g_realpath = ::realpath;
std::function<int(char*)> g_mkstemp = ::mkstemp;
std::function<int(int, int, off_t, off_t)> g_fallocate = ::fallocate;
std::function<void*(void*, size_t, int, int, int, off_t)> g_mmap = ::mmap;
std::function<int(void*, size_t)> g_munmap = ::munmap;
std::function<int(const char*)> g_unlink = ::unlink;
std::function<sighandler_t(int, sighandler_t)> g_signal = ::signal;

ASSERT_HOOK_MATCHES_PROD(g_accept, ::accept);
ASSERT_HOOK_MATCHES_PROD(g_recv, ::recv);
ASSERT_HOOK_MATCHES_PROD(g_send, ::send);
ASSERT_HOOK_MATCHES_PROD(g_poll, ::poll);
ASSERT_HOOK_MATCHES_PROD(g_getsockopt, ::getsockopt);
ASSERT_HOOK_MATCHES_PROD(g_getsockname, ::getsockname);
ASSERT_HOOK_MATCHES_PROD(g_shutdown, ::shutdown);
ASSERT_HOOK_MATCHES_PROD(g_recvmsg, ::recvmsg);
ASSERT_HOOK_MATCHES_PROD(g_sendmsg, ::sendmsg);
ASSERT_HOOK_MATCHES_PROD(g_pipe, ::pipe);
ASSERT_HOOK_MATCHES_PROD(g_getifaddrs, ::getifaddrs);
ASSERT_HOOK_MATCHES_PROD(g_freeifaddrs, ::freeifaddrs);
ASSERT_HOOK_MATCHES_PROD(g_getrlimit, ::getrlimit);
ASSERT_HOOK_MATCHES_PROD(g_setrlimit, ::setrlimit);
ASSERT_HOOK_MATCHES_PROD(g_schedGetaffinity, ::sched_getaffinity);
ASSERT_HOOK_MATCHES_PROD(g_realpath, ::realpath);
ASSERT_HOOK_MATCHES_PROD(g_fopen, ::fopen);
ASSERT_HOOK_MATCHES_PROD(g_opendir, ::opendir);
ASSERT_HOOK_MATCHES_PROD(g_mkstemp, ::mkstemp);
ASSERT_HOOK_MATCHES_PROD(g_fallocate, ::fallocate);
ASSERT_HOOK_MATCHES_PROD(g_mmap, ::mmap);
ASSERT_HOOK_MATCHES_PROD(g_munmap, ::munmap);
ASSERT_HOOK_MATCHES_PROD(g_unlink, ::unlink);
ASSERT_HOOK_MATCHES_PROD(g_signal, ::signal);

#undef ASSERT_HOOK_MATCHES_PROD

void ScriptRead(ssize_t ret, int err, std::string data) {
  g_readScript.push_back(MicroReadStep{ret, err, std::move(data)});
}

void ScriptReadData(std::string data) {
  const ssize_t n = static_cast<ssize_t>(data.size());
  g_readScript.push_back(MicroReadStep{n, 0, std::move(data)});
}

void ScriptRecv(ssize_t ret, int err, std::string data) {
  g_recvScript.push_back(MicroReadStep{ret, err, std::move(data)});
}

void ScriptRecvData(std::string data) {
  const ssize_t n = static_cast<ssize_t>(data.size());
  g_recvScript.push_back(MicroReadStep{n, 0, std::move(data)});
}

void ResetLibcFakes() {
  g_write = DefaultWrite;
  g_read = DefaultRead;
  g_close = DefaultClose;
  g_socket = DefaultSocket;
  g_connect = DefaultConnect;
  g_bind = DefaultBind;
  g_listen = DefaultListen;
  g_accept = DefaultAccept;
  g_fcntl = DefaultFcntl;
  g_recv = DefaultRecv;
  g_send = DefaultSend;
  g_setsockopt = DefaultSetsockopt;
  g_getaddrinfo = DefaultGetaddrinfo;
  g_freeaddrinfo = DefaultFreeaddrinfo;
  g_getnameinfo = DefaultGetnameinfo;
  g_gaiStrerror = DefaultGaiStrerror;
  g_fwrite = DefaultFwrite;
  g_fflush = DefaultFflush;
  g_perror = DefaultPerror;
  g_exit = DefaultExit;
  g_gethostname = DefaultGethostname;
  g_access = DefaultAccess;
  g_open = DefaultOpen;
  g_malloc = DefaultMalloc;
  g_calloc = DefaultCalloc;
  g_free = DefaultFree;
  g_fopen = DefaultFopen;
  g_opendir = DefaultOpendir;
  g_readdir = DefaultReaddir;
  g_closedir = DefaultClosedir;
  g_readlink = DefaultReadlink;
  g_ioctl = DefaultIoctl;
  g_atexit = DefaultAtexit;
  g_clockGettime = DefaultClockGettime;
  g_posixMemalign = DefaultPosixMemalign;
  g_pthreadCreate = DefaultPthreadCreate;
  g_nanosleep = DefaultNanosleep;
  g_poll = ::poll;
  g_getsockopt = ::getsockopt;
  g_getsockname = ::getsockname;
  g_shutdown = ::shutdown;
  g_recvmsg = ::recvmsg;
  g_sendmsg = ::sendmsg;
  g_pipe = ::pipe;
  g_getifaddrs = ::getifaddrs;
  g_freeifaddrs = ::freeifaddrs;
  g_getrlimit = ::getrlimit;
  g_setrlimit = ::setrlimit;
  g_schedGetaffinity = ::sched_getaffinity;
  g_realpath = ::realpath;
  g_mkstemp = ::mkstemp;
  g_fallocate = ::fallocate;
  g_mmap = ::mmap;
  g_munmap = ::munmap;
  g_unlink = ::unlink;
  g_signal = ::signal;

  g_writtenData.clear();
  g_stdoutData.clear();
  g_fwriteCalls.clear();
  g_perrorCalls.clear();
  g_closedFds.clear();
  g_freedPointers.clear();
  g_atexitFns.clear();
  g_fprintfCalls.clear();
  g_writtenFds.clear();
  g_readFds.clear();
  g_readScript.clear();
  g_readScriptPos = 0;
  g_boundFds.clear();
  g_listenedFds.clear();
  g_acceptedFds.clear();
  g_fcntlCalls.clear();
  g_recvFds.clear();
  g_recvFlags.clear();
  g_recvScript.clear();
  g_recvScriptPos = 0;
  g_sentFds.clear();
  g_sendFlags.clear();
  g_sentData.clear();
  g_nextSocketFd = 42;
  g_socketFailErrno = EAFNOSUPPORT;
  g_bindResult = 0;
  g_bindErrno = EADDRINUSE;
  g_listenResult = 0;
  g_listenErrno = EADDRINUSE;
  g_nextAcceptFd = 44;
  g_acceptErrno = ECONNABORTED;
  g_fcntlResult = 0;
  g_fcntlErrno = EBADF;
  g_fcntlGetFlags = 0;
  g_lastSetsockoptLevel = -1;
  g_lastSetsockoptOptname = -1;
  g_lastSetsockoptTimeval = {-1, -1};
  g_getaddrinfoResult = 0;
  g_addrinfoCount = 1;
  g_addrinfoBasePort = 28028;
  g_freeaddrinfoCalls = 0;
  g_connectResult = 0;
  g_connectErrno = ECONNREFUSED;
  errno = 0;
}

void UseRealLibc() {
  ResetLibcFakes();
  g_write = ::write;
  g_read = ::read;
  g_close = ::close;
  g_socket = ::socket;
  g_connect = ::connect;
  g_bind = ::bind;
  g_listen = ::listen;
  g_accept = ::accept;
  g_fcntl = RealFcntl;
  g_recv = ::recv;
  g_send = ::send;
  g_setsockopt = ::setsockopt;
  g_getaddrinfo = ::getaddrinfo;
  g_freeaddrinfo = ::freeaddrinfo;
  g_getnameinfo = ::getnameinfo;
  g_fwrite = ::fwrite;
  g_fflush = ::fflush;
  g_perror = ::perror;
  g_exit = RealExit;
  g_free = ::free;
  g_open = [](const char* path, int flags, mode_t mode) { return ::open(path, flags, mode); };
  g_fopen = ::fopen;
  g_opendir = ::opendir;
  g_readlink = ::readlink;
}

// ---------------------------------------------------------------------------
// Trampolines. fakes/libc_seam.h macro-renames each libc call in the unit
// under test to the matching micro_* name; these dispatch through the slots
// above so the seam is swappable per test rather than baked in at compile time.
// ---------------------------------------------------------------------------
extern "C" {

ssize_t micro_write(int fd, const void* buf, size_t count) { return g_write(fd, buf, count); }
ssize_t micro_read(int fd, void* buf, size_t count) { return g_read(fd, buf, count); }
int micro_close(int fd) { return g_close(fd); }
int micro_socket(int domain, int type, int protocol) { return g_socket(domain, type, protocol); }
int micro_connect(int fd, const struct sockaddr* addr, socklen_t len) { return g_connect(fd, addr, len); }
int micro_bind(int fd, const struct sockaddr* addr, socklen_t len) { return g_bind(fd, addr, len); }
int micro_listen(int fd, int backlog) { return g_listen(fd, backlog); }
int micro_accept(int fd, struct sockaddr* addr, socklen_t* addrlen) { return g_accept(fd, addr, addrlen); }
int micro_fcntl(int fd, int cmd, ...) {
  int arg = 0;
  if (cmd == F_SETFL) {
    va_list args;
    va_start(args, cmd);
    arg = va_arg(args, int);
    va_end(args);
  }
  return g_fcntl(fd, cmd, arg);
}
ssize_t micro_recv(int fd, void* buf, size_t count, int flags) { return g_recv(fd, buf, count, flags); }
ssize_t micro_send(int fd, const void* buf, size_t count, int flags) { return g_send(fd, buf, count, flags); }
int micro_setsockopt(int fd, int level, int optname, const void* optval, socklen_t optlen) {
  return g_setsockopt(fd, level, optname, optval, optlen);
}
int micro_getaddrinfo(const char* node, const char* service, const struct addrinfo* hints, struct addrinfo** res) {
  return g_getaddrinfo(node, service, hints, res);
}
void micro_freeaddrinfo(struct addrinfo* ai) { g_freeaddrinfo(ai); }
int micro_getnameinfo(const struct sockaddr* sa, socklen_t salen, char* host, socklen_t hostlen, char* serv,
                      socklen_t servlen, int flags) {
  return g_getnameinfo(sa, salen, host, hostlen, serv, servlen, flags);
}
const char* micro_gai_strerror(int code) { return g_gaiStrerror(code); }
size_t micro_fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
  return g_fwrite(ptr, size, nmemb, stream);
}
int micro_fflush(FILE* stream) { return g_fflush(stream); }
void micro_perror(const char* prefix) { g_perror(prefix); }
int micro_gethostname(char* name, size_t len) { return g_gethostname(name, len); }
int micro_access(const char* path, int mode) { return g_access(path, mode); }
int micro_open(const char* path, int flags, ...) {
  mode_t mode = 0;
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, mode_t);
    va_end(args);
  }
  return g_open(path, flags, mode);
}
void* micro_malloc(size_t size) { return g_malloc(size); }
void* micro_calloc(size_t nmemb, size_t size) { return g_calloc(nmemb, size); }
void micro_free(void* ptr) { g_free(ptr); }
FILE* micro_fopen(const char* path, const char* mode) { return g_fopen(path, mode); }
DIR* micro_opendir(const char* path) { return g_opendir(path); }
struct dirent* micro_readdir(DIR* dir) { return g_readdir(dir); }
int micro_closedir(DIR* dir) { return g_closedir(dir); }
ssize_t micro_readlink(const char* path, char* buf, size_t len) { return g_readlink(path, buf, len); }
int micro_ioctl(int fd, unsigned long request, ...) {
  va_list args;
  va_start(args, request);
  void* arg = va_arg(args, void*);
  va_end(args);
  return g_ioctl(fd, request, arg);
}
int micro_atexit(void (*fn)(void)) { return g_atexit(fn); }
int micro_clock_gettime(clockid_t clk, struct timespec* ts) { return g_clockGettime(clk, ts); }
int micro_posix_memalign(void** ptr, size_t alignment, size_t size) { return g_posixMemalign(ptr, alignment, size); }
int micro_pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*fn)(void*), void* arg) {
  return g_pthreadCreate(thread, attr, fn, arg);
}
int micro_nanosleep(const struct timespec* req, struct timespec* rem) { return g_nanosleep(req, rem); }
int micro_poll(struct pollfd* fds, nfds_t nfds, int timeout) { return g_poll(fds, nfds, timeout); }
int micro_getsockopt(int fd, int level, int optname, void* optval, socklen_t* optlen) {
  return g_getsockopt(fd, level, optname, optval, optlen);
}
int micro_getsockname(int fd, struct sockaddr* addr, socklen_t* len) { return g_getsockname(fd, addr, len); }
int micro_shutdown(int fd, int how) { return g_shutdown(fd, how); }
ssize_t micro_recvmsg(int fd, struct msghdr* msg, int flags) { return g_recvmsg(fd, msg, flags); }
ssize_t micro_sendmsg(int fd, const struct msghdr* msg, int flags) { return g_sendmsg(fd, msg, flags); }
int micro_pipe(int* fds) { return g_pipe(fds); }
int micro_getifaddrs(struct ifaddrs** ifap) { return g_getifaddrs(ifap); }
void micro_freeifaddrs(struct ifaddrs* ifa) { g_freeifaddrs(ifa); }
int micro_getrlimit(int resource, struct rlimit* lim) { return g_getrlimit(resource, lim); }
int micro_setrlimit(int resource, const struct rlimit* lim) { return g_setrlimit(resource, lim); }
int micro_sched_getaffinity(pid_t pid, size_t size, cpu_set_t* set) { return g_schedGetaffinity(pid, size, set); }
char* micro_realpath(const char* path, char* resolved) { return g_realpath(path, resolved); }
int micro_mkstemp(char* tmpl) { return g_mkstemp(tmpl); }
int micro_fallocate(int fd, int mode, off_t offset, off_t len) { return g_fallocate(fd, mode, offset, len); }
void* micro_mmap(void* addr, size_t len, int prot, int flags, int fd, off_t offset) {
  return g_mmap(addr, len, prot, flags, fd, offset);
}
int micro_munmap(void* addr, size_t len) { return g_munmap(addr, len); }
int micro_unlink(const char* path) { return g_unlink(path); }
sighandler_t micro_signal(int sig, sighandler_t handler) { return g_signal(sig, handler); }

// Always forwards to the real vfprintf so stderr output still happens; records only the FILE* and never the
// formatted text, since no test needs the diagnostic's wording, only whether and where one was printed.
int micro_fprintf(FILE* stream, const char* fmt, ...) {
  g_fprintfCalls.push_back(stream);
  va_list args;
  va_start(args, fmt);
  const int ret = vfprintf(stream, fmt, args);
  va_end(args);
  return ret;
}

// noreturn: the default throws MicroExit. If a hook ever returns normally
// the unit would fall through a path production treats as unreachable, so make
// that a loud abort rather than silent corruption. The attribute is spelled
// here as well as on the seam's declaration; the two merge, but a reader of
// either one should not have to check the other.
__attribute__((noreturn)) void micro_exit(int status) {
  g_exit(status);
  std::abort();
}

}  // extern "C"
