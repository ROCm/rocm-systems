/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/libc_fakes.h"

// Puts the seam's 13 micro_* prototypes in scope so the compiler checks them against the definitions at the bottom of
// this file. Without it the two lists are hand-maintained and both extern "C", so a drifted parameter type would link
// cleanly and corrupt arguments at run time. Include the undef half immediately: this file's defaults call real libc.
#include "fakes/libc_seam.h"
#include "fakes/libc_seam_undef.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>

// LogCapture.hpp's ncclDebugLevel/ncclDebugMask come from fakes/nccl_fakes.cc,
// which this binary already links. A libc-only unit reports via plain
// fprintf(stderr), so CaptureLog works without raising the level.

std::string g_writtenData;
std::string g_stdoutData;
FILE* g_lastFwriteStream = nullptr;
std::vector<int> g_closedFds;
std::vector<int> g_writtenFds;
std::vector<int> g_readFds;
std::vector<MicroReadStep> g_readScript;
size_t g_readScriptPos = 0;
int g_nextSocketFd = 42;
int g_socketFailErrno = EAFNOSUPPORT;
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

static ssize_t DefaultRead(int fd, void* buf, size_t count) {
  g_readFds.push_back(fd);
  // A zero-length read returns 0 without consuming a step, as read(2) does and as RecordingReader does. rasRead asks
  // for 0 once its buffer is full, so spending a step there would shift every later step onto the wrong call.
  if (count == 0) return 0;
  if (g_readScriptPos >= g_readScript.size()) return 0;  // EOF past the end of the script
  const MicroReadStep& step = g_readScript[g_readScriptPos++];
  // A failing or EOF step delivers nothing, so a payload on one is a script the fake would silently ignore.
  assert((step.ret > 0 || step.data.empty()) &&
         "ScriptRead: a non-positive ret cannot deliver data; drop the payload or make ret positive");
  if (step.ret < 0) {
    errno = step.err;
    return step.ret;
  }
  if (step.ret == 0) return 0;
  // A positive ret is the byte count the step promises. Assert the script means what it says, and clamp to the promise
  // as well so ScriptRead(3, 0, "abcdefgh") cannot hand over eight bytes in a Release build, where NDEBUG drops the
  // assert. Without both, a mismatched script silently pins the fake's behaviour instead of the unit's.
  assert(static_cast<size_t>(step.ret) == step.data.size() &&
         "ScriptRead: a positive ret must equal data.size(); use ScriptReadData to derive it");
  const size_t promised = std::min(static_cast<size_t>(step.ret), step.data.size());
  const size_t n = std::min(promised, count);
  std::memcpy(buf, step.data.data(), n);
  return static_cast<ssize_t>(n);
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

static int DefaultSetsockopt(int, int, int optname, const void* optval, socklen_t optlen) {
  g_lastSetsockoptOptname = optname;
  if (optval && optlen >= static_cast<socklen_t>(sizeof(struct timeval))) {
    std::memcpy(&g_lastSetsockoptTimeval, optval, sizeof(struct timeval));
  }
  return 0;
}

// Builds g_addrinfoCount single-linked IPv4 entries. Paired with
// DefaultFreeaddrinfo; a test that overrides one must override both.
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

// Records the stream too: without it g_stdoutData reads identically whether the unit chose stdout or stderr, so a
// stream swap in the unit under test would leave every assertion on the bytes green.
static size_t DefaultFwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
  g_lastFwriteStream = stream;
  g_stdoutData.append(static_cast<const char*>(ptr), size * nmemb);
  return nmemb;
}

static int DefaultFflush(FILE*) { return 0; }

static void DefaultExit(int status) { throw MicroExit{status}; }

// ---------------------------------------------------------------------------

std::function<ssize_t(int, const void*, size_t)> g_write = DefaultWrite;
std::function<ssize_t(int, void*, size_t)> g_read = DefaultRead;
std::function<int(int)> g_close = DefaultClose;
std::function<int(int, int, int)> g_socket = DefaultSocket;
std::function<int(int, const struct sockaddr*, socklen_t)> g_connect = DefaultConnect;
std::function<int(int, int, int, const void*, socklen_t)> g_setsockopt = DefaultSetsockopt;
std::function<int(const char*, const char*, const struct addrinfo*, struct addrinfo**)> g_getaddrinfo =
    DefaultGetaddrinfo;
std::function<void(struct addrinfo*)> g_freeaddrinfo = DefaultFreeaddrinfo;
std::function<int(const struct sockaddr*, socklen_t, char*, socklen_t, char*, socklen_t, int)> g_getnameinfo =
    DefaultGetnameinfo;
std::function<const char*(int)> g_gaiStrerror = DefaultGaiStrerror;
std::function<size_t(const void*, size_t, size_t, FILE*)> g_fwrite = DefaultFwrite;
std::function<int(FILE*)> g_fflush = DefaultFflush;
std::function<void(int)> g_exit = DefaultExit;

void ScriptRead(ssize_t ret, int err, std::string data) {
  g_readScript.push_back(MicroReadStep{ret, err, std::move(data)});
}

void ScriptReadData(std::string data) {
  const ssize_t n = static_cast<ssize_t>(data.size());
  g_readScript.push_back(MicroReadStep{n, 0, std::move(data)});
}

void ResetLibcFakes() {
  g_write = DefaultWrite;
  g_read = DefaultRead;
  g_close = DefaultClose;
  g_socket = DefaultSocket;
  g_connect = DefaultConnect;
  g_setsockopt = DefaultSetsockopt;
  g_getaddrinfo = DefaultGetaddrinfo;
  g_freeaddrinfo = DefaultFreeaddrinfo;
  g_getnameinfo = DefaultGetnameinfo;
  g_gaiStrerror = DefaultGaiStrerror;
  g_fwrite = DefaultFwrite;
  g_fflush = DefaultFflush;
  g_exit = DefaultExit;

  g_writtenData.clear();
  g_stdoutData.clear();
  g_lastFwriteStream = nullptr;
  g_closedFds.clear();
  g_writtenFds.clear();
  g_readFds.clear();
  g_readScript.clear();
  g_readScriptPos = 0;
  g_nextSocketFd = 42;
  g_socketFailErrno = EAFNOSUPPORT;
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
