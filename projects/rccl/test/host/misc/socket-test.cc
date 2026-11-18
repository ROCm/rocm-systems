/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/socket.cc over real loopback TCP, with the real os/linux.cc and
// the plaintext crypt.cc handshake underneath. Its libc calls go through fakes/libc_seam.h, real
// unless a test hooks one.

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <ifaddrs.h>
#include <mutex>
#include <net/if.h>
#include <netdb.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "crypt.h"
#include "os.h"
#include "param.h"
#include "socket.h"
#include "utils.h"

#include "linux-test-common.h"

#include "fakes/param_redirect.h"
#include "fakes/libc_seam.h"
#include SOCKET_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

constexpr uint64_t kMagic = 0x5eed5eed12345678ull;

ncclSocketAddress V4(const char* ip, uint16_t port) {
  ncclSocketAddress a;
  std::memset(&a, 0, sizeof(a));
  a.sin.sin_family = AF_INET;
  a.sin.sin_port = htons(port);
  inet_pton(AF_INET, ip, &a.sin.sin_addr);
  return a;
}

ncclSocketAddress V6(const char* ip, uint16_t port) {
  ncclSocketAddress a;
  std::memset(&a, 0, sizeof(a));
  a.sin6.sin6_family = AF_INET6;
  a.sin6.sin6_port = htons(port);
  inet_pton(AF_INET6, ip, &a.sin6.sin6_addr);
  return a;
}

std::string Str(const ncclSocketAddress& a) {
  char buf[SOCKET_NAME_MAXLEN + 1];
  return ncclSocketToString(&a, buf, sizeof(buf));
}

class SocketMicrotest : public LinuxTestBase {
 protected:
  void SetUp() override {
    for (ncclSocket* s : {&listen_, &client_, &server_}) {
      std::memset(s, 0, sizeof(*s));
      s->socketDescriptor = s->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
    }
  }
  void TearDown() override {
    for (ncclSocket* s : {&client_, &server_, &listen_}) ncclSocketClose(s);
    for (int fd : fds_) ::close(fd);
    ncclSetEncryption(nullptr);
  }
  int Track(int fd) {
    fds_.push_back(fd);
    return fd;
  }
  void Listen(int asyncFlag = 0, volatile uint32_t* abortFlag = nullptr) {
    ncclSocketAddress any = V4("127.0.0.1", 0);
    ASSERT_EQ(ncclSocketInit(&listen_, &any, kMagic, ncclSocketTypeBootstrap, abortFlag, asyncFlag), ncclSuccess);
    ASSERT_EQ(ncclSocketListen(&listen_), ncclSuccess);
  }
  ncclSocketAddress ListenAddr() {
    ncclSocketAddress a;
    EXPECT_EQ(ncclSocketGetAddr(&listen_, &a), ncclSuccess);
    return a;
  }
  // client_ connects to listen_ with the given magic and type; blocking unless asyncFlag.
  ncclResult_t Connect(uint64_t magic = kMagic, ncclSocketType type = ncclSocketTypeBootstrap, int asyncFlag = 0) {
    ncclSocketAddress a = ListenAddr();
    EXPECT_EQ(ncclSocketInit(&client_, &a, magic, type, nullptr, asyncFlag), ncclSuccess);
    return ncclSocketConnect(&client_);
  }
  // client_ and server_ end Ready.
  void Pair() {
    Listen();
    ASSERT_EQ(Connect(), ncclSuccess);
    ASSERT_EQ(ncclSocketAccept(&server_, &listen_), ncclSuccess);
    ASSERT_EQ(client_.state, ncclSocketStateReady);
    ASSERT_EQ(server_.state, ncclSocketStateReady);
  }
  // A raw TCP client of listen_ that sends the given hello bytes, or none.
  int RawClient(const std::string& hello) {
    ncclSocketAddress a = ListenAddr();
    int fd = Track(::socket(AF_INET, SOCK_STREAM, 0));
    EXPECT_EQ(::connect(fd, &a.sa, sizeof(a.sin)), 0);
    if (!hello.empty()) EXPECT_EQ(::send(fd, hello.data(), hello.size(), 0), (ssize_t)hello.size());
    return fd;
  }
  static std::string Hello(uint64_t magic, ncclSocketType type) {
    std::string h(NCCL_SOCKET_PLAIN_HELLO_BYTES, '\0');
    std::memcpy(&h[0], &magic, sizeof(magic));
    std::memcpy(&h[sizeof(magic)], &type, sizeof(type));
    return h;
  }
  ncclSocket listen_, client_, server_;
  std::vector<int> fds_;
};

// Basics

TEST_F(SocketMicrotest, DefaultMagic_EnvOverridesOnceElseBuiltIn) {
  // Each case needs a fresh process, as the value is cached on first use.
  const std::string style = ::testing::GTEST_FLAG(death_test_style);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  auto magicIs = [](const char* env, uint64_t expected) {
    if (env) SetMicroEnv("NCCL_SOCKET_MAGIC", env);
    bool ok = ncclSocketDefaultMagic() == expected;
    SetMicroEnv("NCCL_SOCKET_MAGIC", "0x99");
    ok = ok && ncclSocketDefaultMagic() == expected;  // cached
    exit(ok ? 0 : 1);
  };
  EXPECT_EXIT(magicIs(nullptr, NCCL_SOCKET_MAGIC), ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT(magicIs("", NCCL_SOCKET_MAGIC), ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT(magicIs("0x1234", 0x1234), ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT(magicIs("12abc", NCCL_SOCKET_MAGIC), ::testing::ExitedWithCode(0), "");
  ::testing::GTEST_FLAG(death_test_style) = style;
}

TEST_F(SocketMicrotest, Move_TransfersOwnershipAndInvalidatesTheSource) {
  Pair();
  ncclSocket moved;
  int fd = client_.socketDescriptor;
#ifndef NCCL_TLS_BACKEND_OPENSSL3  // a TLS backend would rebind this fake state
  client_.crypto = (ncclSocketCrypto*)0x1;
#endif
  ncclSocketMove(&moved, &client_);
  EXPECT_EQ(moved.socketDescriptor, fd);
  EXPECT_EQ(moved.state, ncclSocketStateReady);
  EXPECT_EQ(moved.magic, kMagic);
#ifndef NCCL_TLS_BACKEND_OPENSSL3
  EXPECT_EQ(moved.crypto, (ncclSocketCrypto*)0x1);
#endif
  EXPECT_EQ(client_.socketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(client_.acceptSocketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(client_.state, ncclSocketStateNone);
  EXPECT_EQ(client_.crypto, nullptr);
  moved.crypto = nullptr;
  ncclSocketClose(&moved);
}

TEST_F(SocketMicrotest, ToPort_ReadsEitherFamily) {
  ncclSocketAddress a = V4("10.0.0.1", 5000);
  EXPECT_EQ(ncclSocketToPort(&a), 5000);
  a = V6("::1", 6001);
  EXPECT_EQ(ncclSocketToPort(&a), 6001);
}

TEST_F(SocketMicrotest, GetAndSetFd) {
  int fd = 0;
  EXPECT_EQ(ncclSocketGetFd(nullptr, &fd), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketSetFd(3, nullptr), ncclInvalidArgument);
  ASSERT_EQ(ncclSocketSetFd(7, &client_), ncclSuccess);
  EXPECT_EQ(ncclSocketGetFd(&client_, nullptr), ncclSuccess);
  ASSERT_EQ(ncclSocketGetFd(&client_, &fd), ncclSuccess);
  EXPECT_EQ(fd, 7);
  client_.socketDescriptor = NCCL_INVALID_SOCKET;
}

TEST_F(SocketMicrotest, Shutdown_SignalsThePeerAndMarksTerminating) {
  EXPECT_EQ(ncclSocketShutdown(nullptr, SHUT_RDWR), ncclSuccess);
  ASSERT_EQ(ncclSocketShutdown(&client_, SHUT_RDWR), ncclSuccess);  // no descriptor: state only
  EXPECT_EQ(client_.state, ncclSocketStateTerminating);
  Pair();
  ASSERT_EQ(ncclSocketShutdown(&client_, SHUT_WR), ncclSuccess);
  EXPECT_EQ(client_.state, ncclSocketStateTerminating);
  char c;
  EXPECT_EQ(::recv(server_.socketDescriptor, &c, 1, 0), 0);
}

TEST_F(SocketMicrotest, Shutdown_FailureIsASystemError) {
  int p[2];
  ASSERT_EQ(pipe(p), 0);
  Track(p[0]);
  Track(p[1]);
  ncclSocket sock = client_;
  sock.socketDescriptor = p[0];
  EXPECT_EQ(ncclSocketShutdown(&sock, SHUT_RDWR), ncclSystemError);
}

// Listen

TEST_F(SocketMicrotest, Listen_RejectsNullOrUnopenedSockets) {
  EXPECT_EQ(ncclSocketListen(nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketListen(&listen_), ncclInvalidArgument);
}

TEST_F(SocketMicrotest, Listen_AnyPortGetsAnEphemeralPort) {
  Listen();
  EXPECT_EQ(listen_.state, ncclSocketStateReady);
  EXPECT_EQ(listen_.acceptSocketDescriptor, listen_.socketDescriptor);
  EXPECT_NE(ncclSocketToPort(&listen_.addr), 0);
  EXPECT_EQ(IntOpt(listen_.socketDescriptor, SOL_SOCKET, SO_ACCEPTCONN), 1);
  EXPECT_EQ(IntOpt(listen_.socketDescriptor, SOL_SOCKET, SO_REUSEADDR), 0);
}

TEST_F(SocketMicrotest, Listen_AFixedPortIsReusable) {
  Listen();
  ncclSocketAddress fixed = ListenAddr();
  ncclSocketClose(&listen_);
  ASSERT_EQ(ncclSocketInit(&listen_, &fixed), ncclSuccess);
  ASSERT_EQ(ncclSocketListen(&listen_), ncclSuccess);
  EXPECT_EQ(ncclSocketToPort(&listen_.addr), ncclSocketToPort(&fixed));
  EXPECT_EQ(IntOpt(listen_.socketDescriptor, SOL_SOCKET, SO_REUSEADDR), 1);
  EXPECT_EQ(IntOpt(listen_.socketDescriptor, SOL_SOCKET, SO_REUSEPORT), 1);
}

TEST_F(SocketMicrotest, Listen_PortInUseIsASystemError) {
  Listen();  // without SO_REUSEPORT, so nobody can share the port
  ncclSocketAddress taken = ListenAddr();
  ncclSocket other;
  ASSERT_EQ(ncclSocketInit(&other, &taken), ncclSuccess);
  EXPECT_EQ(ncclSocketListen(&other), ncclSystemError);
  ncclSocketClose(&other);
}

TEST_F(SocketMicrotest, Listen_SyscallFailuresAreSystemErrors) {
  auto fail = FailWith(EACCES);
  // A fixed port takes the setsockopt path. Init sets socket options too, so each hook goes in after it.
  ncclSocket sock[4];
  auto init = [&](int i, uint16_t port) {
    ncclSocketAddress a = V4("127.0.0.1", port);
    EXPECT_EQ(ncclSocketInit(&sock[i], &a), ncclSuccess);
  };
  init(0, 1);
  {
    ScopedHook hook(g_setsockopt, fail);
    EXPECT_EQ(ncclSocketListen(&sock[0]), ncclSystemError);
    EXPECT_EQ(hook.calls, 1);
  }
  init(1, 1);
  {
    ScopedHook hook(g_setsockopt, [](int fd, int level, int name, const void* v, socklen_t len) {
      if (name != SO_REUSEPORT) return ::setsockopt(fd, level, name, v, len);
      errno = EACCES;
      return -1;
    });
    EXPECT_EQ(ncclSocketListen(&sock[1]), ncclSystemError);
    EXPECT_EQ(hook.calls, 2);  // SO_REUSEADDR went through
  }
  init(2, 0);
  {
    ScopedHook hook(g_getsockname, fail);
    EXPECT_EQ(ncclSocketListen(&sock[2]), ncclSystemError);
    EXPECT_EQ(hook.calls, 1);
  }
  init(3, 0);
  {
    ScopedHook hook(g_listen, fail);
    EXPECT_EQ(ncclSocketListen(&sock[3]), ncclSystemError);
    EXPECT_EQ(hook.calls, 1);
  }
  for (ncclSocket& s : sock) ncclSocketClose(&s);
}

// Addresses

TEST_F(SocketMicrotest, ToString_NumericHostAndPort) {
  EXPECT_EQ(Str(V4("127.0.0.1", 5000)), "127.0.0.1<5000>");
  EXPECT_EQ(Str(V6("::1", 80)), "::1<80>");
  char buf[6] = "junk";
  ncclSocketAddress a = V4("127.0.0.1", 5000);
  EXPECT_STREQ(ncclSocketToString(&a, buf, sizeof(buf)), "127.0");
}

TEST_F(SocketMicrotest, ToString_UnprintableAddressesAreEmpty) {
  char buf[64] = "junk";
  EXPECT_EQ(ncclSocketToString(nullptr, buf, sizeof(buf)), buf);
  EXPECT_STREQ(buf, "");
  ncclSocketAddress a = V4("127.0.0.1", 1);
  EXPECT_EQ(ncclSocketToString(&a, nullptr, 0), nullptr);
  a.sa.sa_family = AF_UNIX;
  std::strcpy(buf, "junk");
  EXPECT_STREQ(ncclSocketToString(&a, buf, sizeof(buf)), "");
  a = V4("127.0.0.1", 1);
  ScopedHook getnameinfoHook(g_getnameinfo, [](const sockaddr*, socklen_t, char*, socklen_t, char*, socklen_t, int) {
    return EAI_FAIL;
  });
  std::strcpy(buf, "junk");
  EXPECT_STREQ(ncclSocketToString(&a, buf, sizeof(buf)), "");
}

TEST_F(SocketMicrotest, ToString_NumericHostFormIsOptional) {
  std::vector<int> flags;
  ScopedHook getnameinfoHook(g_getnameinfo, [&](const sockaddr* sa, socklen_t len, char* host, socklen_t hl,
                                                char* serv, socklen_t sl, int f) {
    flags.push_back(f);
    return ::getnameinfo(sa, len, host, hl, serv, sl, f | NI_NUMERICHOST);
  });
  ncclSocketAddress a = V4("127.0.0.1", 1);
  char buf[64];
  ncclSocketToString(&a, buf, sizeof(buf));
  ncclSocketToString(&a, buf, sizeof(buf), /*numericHostForm=*/0);
  EXPECT_EQ(flags, (std::vector<int>{NI_NUMERICSERV | NI_NUMERICHOST, NI_NUMERICSERV}));
}

TEST_F(SocketMicrotest, EnvSocketFamily) {
  EXPECT_EQ(ncclEnvSocketFamily(), -1);
  EXPECT_TRUE(log_.Text().empty());
  const std::pair<const char*, int> cases[] = {{"AF_INET", AF_INET}, {"AF_INET6", AF_INET6}, {"AF_UNIX", -1}};
  for (auto& c : cases) {
    SetMicroEnv("NCCL_SOCKET_FAMILY", c.first);
    EXPECT_EQ(ncclEnvSocketFamily(), c.second) << c.first;
  }
}

TEST_F(SocketMicrotest, GetAddrFromString_IPv4WithPort) {
  ncclSocketAddress a;
  ASSERT_EQ(ncclSocketGetAddrFromString(&a, "10.1.2.3:5000"), ncclSuccess);
  EXPECT_EQ(Str(a), "10.1.2.3<5000>");
}

// Without a port, the missing -1 becomes port 65535. Pins AICOMRCCL-2977; a fix flips this pin.
TEST_F(SocketMicrotest, GetAddrFromString_KnownBug_MissingPortBecomes65535) {
  ncclSocketAddress a;
  ASSERT_EQ(ncclSocketGetAddrFromString(&a, "10.1.2.3"), ncclSuccess);
  EXPECT_EQ(ncclSocketToPort(&a), 65535);
}

TEST_F(SocketMicrotest, GetAddrFromString_ResolvedIPv6IsGlobalScope) {
  ScopedHook getaddrinfoHook(g_getaddrinfo, [](const char*, const char* svc, const addrinfo* hints, addrinfo** res) {
    EXPECT_EQ(hints->ai_family, AF_UNSPEC);
    EXPECT_EQ(hints->ai_socktype, SOCK_STREAM);
    int rc = ::getaddrinfo("fe80::1", svc, hints, res);
    ((sockaddr_in6*)(*res)->ai_addr)->sin6_scope_id = 5;
    ((sockaddr_in6*)(*res)->ai_addr)->sin6_flowinfo = 7;
    return rc;
  });
  ncclSocketAddress a;
  ASSERT_EQ(ncclSocketGetAddrFromString(&a, "somehost:4000"), ncclSuccess);
  EXPECT_EQ(a.sa.sa_family, AF_INET6);
  EXPECT_EQ(ncclSocketToPort(&a), 4000);
  EXPECT_EQ(a.sin6.sin6_scope_id, 0u);
  EXPECT_EQ(a.sin6.sin6_flowinfo, 0u);
}

TEST_F(SocketMicrotest, GetAddrFromString_IPv4Failures) {
  ncclSocketAddress a;
  ScopedHook getaddrinfoHook(g_getaddrinfo,
                             [](const char*, const char*, const addrinfo*, addrinfo**) { return EAI_NONAME; });
  EXPECT_EQ(ncclSocketGetAddrFromString(&a, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketGetAddrFromString(&a, "x"), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketGetAddrFromString(&a, ":5000"), ncclInvalidArgument);
  EXPECT_EQ(getaddrinfoHook.calls, 0);  // rejected before resolving
  EXPECT_EQ(ncclSocketGetAddrFromString(&a, "nohost:1"), ncclInvalidArgument);
  EXPECT_EQ(getaddrinfoHook.calls, 1);
  ScopedHook unixFamilyHook(g_getaddrinfo, [](const char*, const char* svc, const addrinfo* hints, addrinfo** res) {
    int rc = ::getaddrinfo("127.0.0.1", svc, hints, res);
    (*res)->ai_family = AF_UNIX;
    return rc;
  });
  EXPECT_EQ(ncclSocketGetAddrFromString(&a, "odd:1"), ncclInvalidArgument);
  EXPECT_EQ(unixFamilyHook.calls, 1);
}

TEST_F(SocketMicrotest, GetAddrFromString_BracketedIPv6) {
  ncclSocketAddress a;
  ASSERT_EQ(ncclSocketGetAddrFromString(&a, "[2001:db8::7]:8080"), ncclSuccess);
  EXPECT_EQ(Str(a), "2001:db8::7<8080>");
  EXPECT_EQ(a.sin6.sin6_scope_id, 0u);
  ASSERT_EQ(ncclSocketGetAddrFromString(&a, "[fe80::1%lo]:80"), ncclSuccess);
  EXPECT_EQ(ncclSocketToPort(&a), 80);
  EXPECT_EQ(a.sin6.sin6_scope_id, if_nametoindex("lo"));
  char host[INET6_ADDRSTRLEN];
  EXPECT_STREQ(inet_ntop(AF_INET6, &a.sin6.sin6_addr, host, sizeof(host)), "fe80::1");
  EXPECT_EQ(ncclSocketGetAddrFromString(&a, "[::1:80"), ncclInvalidArgument);
}

TEST_F(SocketMicrotest, GetAddr_OnlyFromAReadySocket) {
  ncclSocketAddress a;
  EXPECT_EQ(ncclSocketGetAddr(nullptr, &a), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketGetAddr(&listen_, &a), ncclInternalError);
  Listen();
  ASSERT_EQ(ncclSocketGetAddr(&listen_, &a), ncclSuccess);
  EXPECT_EQ(std::memcmp(&a, &listen_.addr, sizeof(a)), 0);
}

// Interface selection, over a scripted getifaddrs() list

class SocketIfMicrotest : public SocketMicrotest {
 protected:
  std::vector<std::string> Find() {
    char names[MAX_IFS * MAX_IF_NAME_SIZE];
    ncclSocketAddress addrs[MAX_IFS];
    int n = -1;
    EXPECT_EQ(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &n), ncclSuccess);
    std::vector<std::string> out;
    for (int i = 0; i < n; i++) out.push_back(names + i * MAX_IF_NAME_SIZE);
    return out;
  }
  LinuxFakeIfaddrs ifs_;
};

TEST_F(SocketIfMicrotest, Auto_PrefersIbThenAnythingButDockerLoopbackAndBridges) {
  ifs_.Add("lo", AF_INET, "127.0.0.1");
  ifs_.Add("docker0", AF_INET, "172.17.0.1");
  ifs_.Add("virbr0", AF_INET, "192.168.122.1");
  ifs_.Add("eth0", AF_INET, "10.0.0.1");
  ifs_.Add("ib0", AF_INET, "10.1.0.1");
  EXPECT_EQ(Find(), (std::vector<std::string>{"ib0"}));
}

TEST_F(SocketIfMicrotest, Auto_FallsBackToDockerThenLoopbackThenBridges) {
  ifs_.Add("lo", AF_INET, "127.0.0.1");
  ifs_.Add("virbr0", AF_INET, "192.168.122.1");
  ifs_.Add("docker0", AF_INET, "172.17.0.1");
  ifs_.Add("eth0", AF_INET, "10.0.0.1");
  EXPECT_EQ(Find(), (std::vector<std::string>{"eth0"}));
  {
    LinuxFakeIfaddrs noEth;
    noEth.Add("lo", AF_INET, "127.0.0.1");
    noEth.Add("virbr0", AF_INET, "192.168.122.1");
    noEth.Add("docker0", AF_INET, "172.17.0.1");
    EXPECT_EQ(Find(), (std::vector<std::string>{"docker0"}));
  }
  {
    LinuxFakeIfaddrs noDocker;
    noDocker.Add("virbr0", AF_INET, "192.168.122.1");
    noDocker.Add("lo", AF_INET, "127.0.0.1");
    EXPECT_EQ(Find(), (std::vector<std::string>{"lo"}));
  }
  {
    LinuxFakeIfaddrs bridgeOnly;
    bridgeOnly.Add("virbr0", AF_INET, "192.168.122.1");
    EXPECT_EQ(Find(), (std::vector<std::string>{"virbr0"}));
  }
  LinuxFakeIfaddrs none;
  EXPECT_EQ(Find(), (std::vector<std::string>{}));
}

TEST_F(SocketIfMicrotest, Auto_CommIdPicksTheInterfaceInItsSubnet) {
  ifs_.Add("eth0", AF_INET, "192.168.0.1", "255.255.255.0");
  ifs_.Add("eth1", AF_INET, "10.0.0.1", "255.0.0.0");
  SetMicroEnv("NCCL_COMM_ID", "10.9.9.9:5000");
  EXPECT_EQ(Find(), (std::vector<std::string>{"eth1"}));
  SetMicroEnv("NCCL_COMM_ID", "x");  // too short: ignored
  EXPECT_EQ(Find(), (std::vector<std::string>{"eth0", "eth1"}));
  SetMicroEnv("NCCL_COMM_ID", "11.0.0.1:5000");  // no subnet match: fall through
  EXPECT_EQ(Find(), (std::vector<std::string>{"eth0", "eth1"}));
}

TEST_F(SocketIfMicrotest, Auto_BadCommIdIsAnError) {
  SetMicroEnv("NCCL_COMM_ID", "[::1");
  ifs_.Add("eth0", AF_INET, "10.0.0.1");
  char names[MAX_IFS * MAX_IF_NAME_SIZE];
  ncclSocketAddress addrs[MAX_IFS];
  int n = -1;
  EXPECT_EQ(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &n), ncclInvalidArgument);
}

TEST_F(SocketIfMicrotest, IfnameAndFamilyComeFromTheEnvironment) {
  ifs_.Add("eth0", AF_INET, "10.0.0.1");
  ifs_.Add("eth0", AF_INET6, "2001:db8::1");
  ifs_.Add("ib0", AF_INET, "10.1.0.1");
  SetMicroEnv("NCCL_SOCKET_IFNAME", "eth");
  EXPECT_EQ(Find(), (std::vector<std::string>{"eth0"}));
  SetMicroEnv("NCCL_SOCKET_IFNAME", "e");  // one character: ignored
  EXPECT_EQ(Find(), (std::vector<std::string>{"ib0"}));
  SetMicroEnv("NCCL_SOCKET_IFNAME", "eth0");
  SetMicroEnv("NCCL_SOCKET_FAMILY", "AF_INET6");
  char names[MAX_IFS * MAX_IF_NAME_SIZE];
  ncclSocketAddress addrs[MAX_IFS];
  int n = -1;
  ASSERT_EQ(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &n), ncclSuccess);
  ASSERT_EQ(n, 1);
  EXPECT_EQ(addrs[0].sa.sa_family, AF_INET6);
}

TEST_F(SocketIfMicrotest, EnumerationFailuresPropagate) {
  ScopedHook getifaddrsHook(g_getifaddrs, FailWith(ENOMEM));
  char names[MAX_IFS * MAX_IF_NAME_SIZE];
  ncclSocketAddress addrs[MAX_IFS];
  int n = -1;
  EXPECT_EQ(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &n), ncclSystemError);
  SetMicroEnv("NCCL_SOCKET_IFNAME", "eth0");
  EXPECT_EQ(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &n), ncclSystemError);
}

// Fails the n-th getifaddrs() call, to reach each fallback's error check.
TEST_F(SocketIfMicrotest, EachFallbackPropagatesAnEnumerationFailure) {
  SetMicroEnv("NCCL_COMM_ID", "10.9.9.9:5000");
  for (int failAt = 1; failAt <= 6; failAt++) {
    SCOPED_TRACE(failAt);
    int calls = 0;
    ScopedHook getifaddrsHook(g_getifaddrs, [&](ifaddrs** out) {
      if (++calls == failAt) {
        errno = ENOMEM;
        return -1;
      }
      *out = nullptr;
      return 0;
    });
    ScopedHook freeifaddrsHook(g_freeifaddrs, [](ifaddrs*) {});
    char names[MAX_IFS * MAX_IF_NAME_SIZE];
    ncclSocketAddress addrs[MAX_IFS];
    int n = -1;
    EXPECT_EQ(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &n), ncclSystemError);
  }
}

// Init

TEST_F(SocketMicrotest, Init_WithoutAnAddressOnlyResetsState) {
  EXPECT_EQ(ncclSocketInit(nullptr), ncclSuccess);
  volatile uint32_t abortFlag = 0;
  std::memset(&client_, 0x5a, sizeof(client_));
  ASSERT_EQ(ncclSocketInit(&client_, nullptr, kMagic, ncclSocketTypeProxy, &abortFlag, 1, 1), ncclSuccess);
  EXPECT_EQ(client_.state, ncclSocketStateInitialized);
  EXPECT_EQ(client_.socketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(client_.acceptSocketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(client_.errorRetries, 0);
  EXPECT_EQ(client_.abortFlag, &abortFlag);
  EXPECT_EQ(client_.asyncFlag, 1);
  EXPECT_EQ(client_.customRetry, 1);
  EXPECT_EQ(client_.magic, kMagic);
  EXPECT_EQ(client_.type, ncclSocketTypeProxy);
  EXPECT_EQ(client_.finalizeCounter, 0);
  EXPECT_EQ(client_.crypto, nullptr);
  ncclSocketAddress zero;
  std::memset(&zero, 0, sizeof(zero));
  EXPECT_EQ(std::memcmp(&client_.addr, &zero, sizeof(zero)), 0);
}

TEST_F(SocketMicrotest, Init_OpensASocketOfTheAddressFamily) {
  ncclSocketAddress a = V4("127.0.0.1", 1);
  ASSERT_EQ(ncclSocketInit(&client_, &a), ncclSuccess);
  EXPECT_TRUE(FdOpen(client_.socketDescriptor));
  EXPECT_EQ(client_.salen, (int)sizeof(sockaddr_in));
  EXPECT_EQ(client_.magic, ncclSocketDefaultMagic());
  EXPECT_EQ(client_.type, ncclSocketTypeUnknown);
  EXPECT_EQ(IntOpt(client_.socketDescriptor, SOL_SOCKET, SO_REUSEADDR), 0);
  linger l{};
  socklen_t len = sizeof(l);
  ::getsockopt(client_.socketDescriptor, SOL_SOCKET, SO_LINGER, &l, &len);
  EXPECT_EQ(l.l_onoff, 0);
  ncclSocketAddress b = V6("::1", 1);
  ASSERT_EQ(ncclSocketInit(&server_, &b), ncclSuccess);
  EXPECT_EQ(server_.salen, (int)sizeof(sockaddr_in6));
  EXPECT_EQ(IntOpt(server_.socketDescriptor, SOL_SOCKET, SO_DOMAIN), AF_INET6);
}

TEST_F(SocketMicrotest, Init_AppliesTheReuseAddrAndLingerParams) {
  params_.Set("RCCL_SOCKET_REUSEADDR", 1);
  params_.Set("RCCL_SOCKET_LINGER", 0);
  ncclSocketAddress a = V4("127.0.0.1", 1);
  ASSERT_EQ(ncclSocketInit(&client_, &a), ncclSuccess);
  EXPECT_EQ(IntOpt(client_.socketDescriptor, SOL_SOCKET, SO_REUSEADDR), 1);
  linger l{};
  socklen_t len = sizeof(l);
  ::getsockopt(client_.socketDescriptor, SOL_SOCKET, SO_LINGER, &l, &len);
  EXPECT_EQ(l.l_onoff, 1);
  EXPECT_EQ(l.l_linger, 0);
}

TEST_F(SocketMicrotest, Init_RejectsANonIpAddress) {
  ncclSocketAddress a = V4("127.0.0.1", 1);
  a.sa.sa_family = AF_UNIX;
  EXPECT_EQ(ncclSocketInit(&client_, &a), ncclInternalError);
  EXPECT_EQ(client_.socketDescriptor, NCCL_INVALID_SOCKET);
}

// Failures after the descriptor exists close it again.
TEST_F(SocketMicrotest, Init_FailuresCloseTheSocket) {
  params_.Set("RCCL_SOCKET_REUSEADDR", 1);
  params_.Set("RCCL_SOCKET_LINGER", 3);
  ncclSocketAddress a = V4("127.0.0.1", 1);
  for (int failing : {SO_REUSEADDR, SO_LINGER}) {
    int created = -1;
    ScopedHook setsockoptHook(g_setsockopt, [&](int fd, int level, int name, const void* v, socklen_t len) {
      created = fd;
      if (name == failing) {
        errno = EINVAL;
        return -1;
      }
      return ::setsockopt(fd, level, name, v, len);
    });
    EXPECT_EQ(ncclSocketInit(&client_, &a), ncclSystemError);
    EXPECT_FALSE(FdOpen(created));
    EXPECT_EQ(client_.state, ncclSocketStateClosed);
    EXPECT_EQ(client_.socketDescriptor, NCCL_INVALID_SOCKET);
  }
}

TEST_F(SocketMicrotest, Init_SocketCreationFailureClosesNothing) {
  rlimit saved;
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &saved), 0);
  int lowest = open("/dev/null", O_RDONLY);  // not dup(0): stdin may be closed
  ASSERT_GE(lowest, 0);
  ::close(lowest);
  rlimit tight = saved;
  tight.rlim_cur = lowest;  // the next descriptor would exceed the limit
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &tight), 0);
  ncclSocketAddress a = V4("127.0.0.1", 1);
  ncclResult_t ret = ncclSocketInit(&client_, &a);
  setrlimit(RLIMIT_NOFILE, &saved);
  EXPECT_EQ(ret, ncclSystemError);
  EXPECT_EQ(client_.state, ncclSocketStateClosed);
}

// Connect, accept and the plaintext hello

TEST_F(SocketMicrotest, ConnectAndAccept_HandshakeThenExchangeData) {
  Pair();
  EXPECT_EQ(client_.finalizeCounter, (int)NCCL_SOCKET_PLAIN_HELLO_BYTES);
  EXPECT_EQ(server_.magic, kMagic);
  EXPECT_NE(server_.socketDescriptor, listen_.socketDescriptor);
  EXPECT_EQ(server_.acceptSocketDescriptor, listen_.socketDescriptor);
  int running = 0;
  ASSERT_EQ(ncclSocketReady(&client_, &running), ncclSuccess);
  EXPECT_EQ(running, 1);
  char out[] = "ping", in[5] = {};
  ASSERT_EQ(ncclSocketSend(&client_, out, sizeof(out)), ncclSuccess);
  ASSERT_EQ(ncclSocketRecv(&server_, in, sizeof(in)), ncclSuccess);
  EXPECT_STREQ(in, "ping");
}

TEST_F(SocketMicrotest, Connect_RejectsBadSocketsAndStates) {
  EXPECT_EQ(ncclSocketConnect(nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketConnect(&client_), ncclInvalidArgument);
  ncclSocketAddress a = V4("127.0.0.1", 1);
  ASSERT_EQ(ncclSocketInit(&client_, &a), ncclSuccess);
  client_.state = ncclSocketStateReady;
  EXPECT_EQ(ncclSocketConnect(&client_), ncclInternalError);
  client_.state = ncclSocketStateError;
  EXPECT_EQ(ncclSocketConnect(&client_), ncclRemoteError);
}

#ifndef NCCL_TLS_BACKEND_OPENSSL3
TEST_F(SocketMicrotest, Connect_EncryptionWithoutTlsFailsBeforeConnecting) {
  Listen();
  ncclEncryptionConfig_t config = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  config.mode = NCCL_ENCRYPTION_MODE_PSK;
  config.psk = "0123456789abcdef0123456789abcdef";
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  EXPECT_EQ(Connect(), ncclInvalidUsage);
  EXPECT_EQ(client_.state, ncclSocketStateInitialized);
}
#endif

TEST_F(SocketMicrotest, Connect_RefusedPastTheRetryCountIsARemoteError) {
  params_.Set("SOCKET_RETRY_CNT", 1);
  Listen();
  ncclSocketAddress a = ListenAddr();
  ncclSocketClose(&listen_);
  ASSERT_EQ(ncclSocketInit(&client_, &a), ncclSuccess);
  EXPECT_EQ(ncclSocketConnect(&client_), ncclRemoteError);
  EXPECT_EQ(client_.state, ncclSocketStateError);
  EXPECT_EQ(client_.errorRetries, 2);
}

TEST_F(SocketMicrotest, Connect_AbortReturnsAnInternalError) {
  Listen();
  ncclSocketAddress a = ListenAddr();
  volatile uint32_t abortFlag = 1;
  ASSERT_EQ(ncclSocketInit(&client_, &a, kMagic, ncclSocketTypeBootstrap, &abortFlag), ncclSuccess);
  EXPECT_EQ(ncclSocketConnect(&client_), ncclInternalError);
}

TEST_F(SocketMicrotest, Connect_AsyncReturnsEarlyAndReadyFinishes) {
  Listen();
  ASSERT_EQ(Connect(kMagic, ncclSocketTypeBootstrap, /*asyncFlag=*/1), ncclSuccess);
  EXPECT_TRUE(client_.state == ncclSocketStateConnecting || client_.state == ncclSocketStateConnectPolling ||
              client_.state == ncclSocketStateConnected || client_.state == ncclSocketStateReady);
  int running = 0;
  for (int i = 0; i < 1000 && !running; i++) ASSERT_EQ(ncclSocketReady(&client_, &running), ncclSuccess);
  EXPECT_EQ(running, 1);
  ASSERT_EQ(ncclSocketAccept(&server_, &listen_), ncclSuccess);
  EXPECT_EQ(server_.state, ncclSocketStateReady);
}

TEST_F(SocketMicrotest, Connect_AsyncStopsInWhicheverConnectStateItReached) {
  params_.Set("SOCKET_RETRY_CNT", 5);
  Listen();
  for (int err : {EINPROGRESS, ECONNREFUSED}) {
    ScopedHook connectHook(g_connect, FailWith(err));
    ScopedHook pollHook(g_poll, [](pollfd*, nfds_t, int) { return 0; });  // still connecting
    ncclSocketClose(&client_);
    ASSERT_EQ(Connect(kMagic, ncclSocketTypeBootstrap, /*asyncFlag=*/1), ncclSuccess);
    EXPECT_EQ(client_.state, err == EINPROGRESS ? ncclSocketStateConnectPolling : ncclSocketStateConnecting);
  }
}

TEST_F(SocketMicrotest, PollConnect_CompletesAPendingConnect) {
  EXPECT_EQ(ncclSocketPollConnect(nullptr), ncclInvalidArgument);
  Listen();
  ncclSocketAddress a = ListenAddr();
  ASSERT_EQ(ncclSocketInit(&client_, &a), ncclSuccess);
  ASSERT_EQ(::connect(client_.socketDescriptor, &a.sa, sizeof(a.sin)), 0);
  client_.state = ncclSocketStateConnectPolling;
  ASSERT_EQ(ncclSocketPollConnect(&client_), ncclSuccess);
  EXPECT_EQ(client_.state, ncclSocketStateConnected);
  ncclSocket bad = client_;
  bad.socketDescriptor = 1 << 20;  // poll reports POLLNVAL; getsockopt then fails
  EXPECT_EQ(ncclSocketPollConnect(&bad), ncclSystemError);
}

// An async connect whose hello cannot be sent yet stays Connected.
TEST_F(SocketMicrotest, Ready_HelloThatCannotBeSentYetStaysConnected) {
  Pair();
  client_.asyncFlag = 1;
  fcntl(client_.socketDescriptor, F_SETFL, O_NONBLOCK);
  std::vector<char> junk(1 << 16);
  while (::send(client_.socketDescriptor, junk.data(), junk.size(), MSG_DONTWAIT) > 0) {
  }
  client_.state = ncclSocketStateConnected;
  client_.finalizeCounter = 0;
  int running = 1;
  ASSERT_EQ(ncclSocketReady(&client_, &running), ncclSuccess);
  EXPECT_EQ(running, 0);
  EXPECT_EQ(client_.state, ncclSocketStateConnected);
}

TEST_F(SocketMicrotest, Ready_NullStoppedOrFailedSockets) {
  int running = 1;
  ASSERT_EQ(ncclSocketReady(nullptr, &running), ncclSuccess);
  EXPECT_EQ(running, 0);
  for (ncclSocketState state : {ncclSocketStateError, ncclSocketStateClosed}) {
    client_.state = state;
    EXPECT_EQ(ncclSocketReady(&client_, &running), ncclRemoteError);
  }
  client_.state = ncclSocketStateNone;
}

TEST_F(SocketMicrotest, Accept_RejectsBadArgumentsAndListenerStates) {
  EXPECT_EQ(ncclSocketAccept(nullptr, &listen_), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketAccept(&server_, nullptr), ncclInvalidArgument);
  listen_.state = ncclSocketStateInitialized;
  EXPECT_EQ(ncclSocketAccept(&server_, &listen_), ncclInternalError);
  listen_.state = ncclSocketStateError;
  EXPECT_EQ(ncclSocketAccept(&server_, &listen_), ncclSystemError);
  listen_.state = ncclSocketStateNone;
}

TEST_F(SocketMicrotest, Accept_AbortReturnsAnInternalError) {
  volatile uint32_t abortFlag = 1;
  Listen(0, &abortFlag);  // non-blocking: nothing pending, so the loop sees the flag
  EXPECT_EQ(ncclSocketAccept(&server_, &listen_), ncclInternalError);
}

TEST_F(SocketMicrotest, Accept_WrongMagicIsDroppedThenTheNextPeerAccepted) {
  Listen();
  int bad = RawClient(Hello(kMagic ^ 1, ncclSocketTypeBootstrap));
  ASSERT_EQ(Connect(), ncclSuccess);
  ASSERT_EQ(ncclSocketAccept(&server_, &listen_), ncclSuccess);
  EXPECT_EQ(server_.state, ncclSocketStateReady);
  char c;
  EXPECT_EQ(::recv(bad, &c, 1, 0), 0);  // the bad peer was closed
}

TEST_F(SocketMicrotest, Accept_WithoutRetryReturnsAfterABadPeer) {
  Listen();
  RawClient(Hello(kMagic, ncclSocketTypeProxy));
  ASSERT_EQ(ncclSocketAccept(&server_, &listen_, /*retry=*/false), ncclSuccess);
  EXPECT_EQ(server_.state, ncclSocketStateAccepting);
  EXPECT_EQ(server_.socketDescriptor, NCCL_INVALID_SOCKET);
}

// A blocking accept keeps reading a hello that arrives in pieces.
TEST_F(SocketMicrotest, Accept_BlockingWaitsForTheWholeHello) {
  Listen();
  std::string hello = Hello(kMagic, ncclSocketTypeBootstrap);
  int raw = RawClient(hello.substr(0, 3));
  bool sentRest = false;
  ScopedHook recvHook(g_recv, [&](int fd, void* buf, size_t n, int f) {
    ssize_t got = ::recv(fd, buf, n, f);
    int err = errno;
    if (got > 0 && !sentRest) {  // the rest only arrives after a partial read
      sentRest = true;
      ::send(raw, hello.data() + 3, hello.size() - 3, 0);
    }
    errno = err;
    return got;
  });
  EXPECT_EQ(ncclSocketAccept(&server_, &listen_), ncclSuccess);
  EXPECT_TRUE(sentRest);
  EXPECT_GE(recvHook.calls, 2);
  EXPECT_EQ(server_.state, ncclSocketStateReady);
}

TEST_F(SocketMicrotest, Accept_AsyncProgressesThroughReady) {
  Listen(/*asyncFlag=*/1);
  ASSERT_EQ(ncclSocketAccept(&server_, &listen_), ncclSuccess);  // nobody yet
  EXPECT_EQ(server_.state, ncclSocketStateAccepting);
  std::string hello = Hello(kMagic, ncclSocketTypeBootstrap);
  int raw = RawClient("");
  int running = 1;
  for (int i = 0; i < 1000 && server_.state == ncclSocketStateAccepting; i++)
    ASSERT_EQ(ncclSocketReady(&server_, &running), ncclSuccess);
  EXPECT_EQ(server_.state, ncclSocketStateAccepted);  // waiting for the hello
  EXPECT_EQ(running, 0);
  ::send(raw, hello.data(), hello.size(), 0);
  for (int i = 0; i < 1000 && !running; i++) ASSERT_EQ(ncclSocketReady(&server_, &running), ncclSuccess);
  EXPECT_EQ(running, 1);
  // Re-entering accept on an accepted socket keeps it.
  ASSERT_EQ(ncclSocketAccept(&server_, &listen_), ncclSuccess);
  EXPECT_EQ(server_.state, ncclSocketStateReady);
}

TEST_F(SocketMicrotest, Ready_BadHandshakeGoesBackToAccepting) {
  Listen(/*asyncFlag=*/1);
  ASSERT_EQ(ncclSocketAccept(&server_, &listen_), ncclSuccess);
  int bad = RawClient(Hello(kMagic ^ 1, ncclSocketTypeBootstrap));
  int running = 1;
  char c;
  for (int i = 0; i < 1000 && ::recv(bad, &c, 1, MSG_DONTWAIT) != 0; i++)  // until the bad peer is closed
    ASSERT_EQ(ncclSocketReady(&server_, &running), ncclSuccess);
  EXPECT_EQ(server_.state, ncclSocketStateAccepting);
  EXPECT_EQ(running, 0);
  // A caller re-entering after a reset goes straight back to accepting the next peer.
  RawClient(Hello(kMagic, ncclSocketTypeBootstrap));
  server_.state = ncclSocketStateBadHandshake;
  ASSERT_EQ(ncclSocketReady(&server_, &running), ncclSuccess);
  EXPECT_NE(server_.state, ncclSocketStateAccepting);
  for (int i = 0; i < 1000 && !running; i++) ASSERT_EQ(ncclSocketReady(&server_, &running), ncclSuccess);
  EXPECT_EQ(running, 1);
}

TEST_F(SocketMicrotest, Accept_SetFlagsFailurePropagates) {
  Listen();
  RawClient(Hello(kMagic, ncclSocketTypeBootstrap));
  ScopedHook setsockoptHook(g_setsockopt, FailWith(ENOMEM));
  EXPECT_EQ(ncclSocketAccept(&server_, &listen_), ncclSystemError);
}

// Data transfer

TEST_F(SocketMicrotest, Transfers_RejectNullSockets) {
  int offset = 0;
  char c;
  bool closed;
  EXPECT_EQ(ncclSocketProgress(NCCL_SOCKET_SEND, nullptr, &c, 1, &offset), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketWait(NCCL_SOCKET_SEND, nullptr, &c, 1, &offset), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketSend(nullptr, &c, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketRecv(nullptr, &c, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketTryRecv(nullptr, &c, 1, &closed, true), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketSendRecv(nullptr, &c, 1, &server_, &c, 1), ncclInternalError);
  EXPECT_EQ(ncclSocketSendRecv(&client_, &c, 1, nullptr, &c, 1), ncclInternalError);
}

TEST_F(SocketMicrotest, Transfers_NeedAReadySocket) {
  char c = 0;
  client_.state = ncclSocketStateTerminating;
  EXPECT_EQ(ncclSocketSend(&client_, &c, 1), ncclInternalError);
  client_.state = ncclSocketStateConnected;
  EXPECT_EQ(ncclSocketRecv(&client_, &c, 1), ncclInternalError);
  server_.state = ncclSocketStateReady;
  EXPECT_EQ(ncclSocketSendRecv(&server_, &c, 1, &client_, &c, 1), ncclInternalError);
  EXPECT_EQ(ncclSocketSendRecv(&client_, &c, 1, &server_, &c, 1), ncclInternalError);
  client_.state = server_.state = ncclSocketStateNone;
}

// A terminating socket can still drain what the peer sent.
TEST_F(SocketMicrotest, Recv_WorksWhileTerminating) {
  Pair();
  char out[] = "tail", in[5] = {};
  ASSERT_EQ(ncclSocketSend(&client_, out, sizeof(out)), ncclSuccess);
  ASSERT_EQ(ncclSocketShutdown(&server_, SHUT_WR), ncclSuccess);
  ASSERT_EQ(ncclSocketRecv(&server_, in, sizeof(in)), ncclSuccess);
  EXPECT_STREQ(in, "tail");
  ASSERT_EQ(ncclSocketSendRecv(&client_, out, sizeof(out), &server_, in, sizeof(in)), ncclSuccess);
}

TEST_F(SocketMicrotest, Recv_PeerCloseIsARemoteError) {
  Pair();
  ncclSocketClose(&client_);
  char in[4];
  EXPECT_EQ(ncclSocketRecv(&server_, in, sizeof(in)), ncclRemoteError);
}

TEST_F(SocketMicrotest, Wait_PollsUntilTheRestArrives) {
  params_.Set("SOCKET_POLL_TIMEOUT_MSEC", 1000);
  Pair();
  server_.asyncFlag = 1;  // single non-blocking attempts, so the wait goes through poll
  ScopedHook pollHook(g_poll, [&](pollfd* p, nfds_t n, int timeout) {
    EXPECT_EQ(p->events, POLLIN);
    EXPECT_EQ(timeout, 1000);
    ::send(client_.socketDescriptor, "abcd", 4, 0);  // the data only arrives once the wait polls
    return ::poll(p, n, timeout);
  });
  char in[4];
  int offset = 0;
  EXPECT_EQ(ncclSocketWait(NCCL_SOCKET_RECV, &server_, in, sizeof(in), &offset), ncclSuccess);
  EXPECT_EQ(pollHook.calls, 1);
  EXPECT_EQ(offset, 4);
  EXPECT_EQ(std::string(in, 4), "abcd");
}

TEST_F(SocketMicrotest, SendRecv_ExchangesBothWaysAtOnce) {
  Pair();
  std::vector<char> big(1 << 20, 'c'), fromServer(1 << 20);
  std::thread peer([&] {
    std::vector<char> in(1 << 20);
    std::vector<char> out(1 << 20, 's');
    EXPECT_EQ(ncclSocketSendRecv(&server_, out.data(), out.size(), &server_, in.data(), in.size()), ncclSuccess);
    EXPECT_EQ(in, big);
  });
  EXPECT_EQ(ncclSocketSendRecv(&client_, big.data(), big.size(), &client_, fromServer.data(), fromServer.size()),
            ncclSuccess);
  peer.join();
  EXPECT_EQ(fromServer, std::vector<char>(1 << 20, 's'));
}

TEST_F(SocketMicrotest, SendRecv_PropagatesErrors) {
  Pair();
  ncclSocketClose(&server_);
  char c = 'x', in;
  EXPECT_EQ(ncclSocketSendRecv(&client_, &c, 0, &client_, &in, 1), ncclRemoteError);
  int p[2];
  ASSERT_EQ(pipe(p), 0);
  Track(p[0]);
  Track(p[1]);
  ncclSocket notSock = client_;
  notSock.socketDescriptor = p[0];
  EXPECT_EQ(ncclSocketSendRecv(&notSock, &c, 1, &client_, &in, 0), ncclRemoteError);
}

TEST_F(SocketMicrotest, MultiOp_ValidatesEveryOp) {
  char buf[4];
  ncclSocketOp ops[2] = {{NCCL_SOCKET_SEND, &client_, buf, 4, 0}, {NCCL_SOCKET_RECV, &client_, buf, 4, 0}};
  EXPECT_EQ(ncclSocketMultiOp(nullptr, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketMultiOp(ops, 0), ncclInvalidArgument);
  ops[1].sock = nullptr;
  EXPECT_EQ(ncclSocketMultiOp(ops, 2), ncclInvalidArgument);
  ops[1].sock = &client_;
  ops[1].op = 7;
  EXPECT_EQ(ncclSocketMultiOp(ops, 2), ncclInvalidArgument);
  ops[1].op = NCCL_SOCKET_RECV;
  ops[1].size = -1;
  EXPECT_EQ(ncclSocketMultiOp(ops, 2), ncclInvalidArgument);
  ops[1].size = 4;
  ops[1].ptr = nullptr;
  EXPECT_EQ(ncclSocketMultiOp(ops, 2), ncclInvalidArgument);
}

TEST_F(SocketMicrotest, MultiOp_RunsSendsAndReceivesToCompletion) {
  Pair();
  std::vector<char> a(1 << 20, 'a'), b(1 << 20, 'b'), gotA(1 << 20), gotB(1 << 20);
  ncclSocketOp clientOps[3] = {{NCCL_SOCKET_SEND, &client_, a.data(), (int)a.size(), 9},
                               {NCCL_SOCKET_RECV, &client_, gotB.data(), (int)gotB.size(), 9},
                               {NCCL_SOCKET_SEND, &client_, nullptr, 0, 9}};
  ncclSocketOp serverOps[2] = {{NCCL_SOCKET_RECV, &server_, gotA.data(), (int)gotA.size(), 0},
                               {NCCL_SOCKET_SEND, &server_, b.data(), (int)b.size(), 0}};
  std::thread peer([&] { EXPECT_EQ(ncclSocketMultiOp(serverOps, 2), ncclSuccess); });
  EXPECT_EQ(ncclSocketMultiOp(clientOps, 3), ncclSuccess);
  peer.join();
  EXPECT_EQ(clientOps[0].offset, (int)a.size());
  EXPECT_EQ(clientOps[2].offset, 0);
  EXPECT_EQ(gotA, a);
  EXPECT_EQ(gotB, b);
  ncclSocketOp onlyEmpty = {NCCL_SOCKET_RECV, &client_, nullptr, 0, 0};
  EXPECT_EQ(ncclSocketMultiOp(&onlyEmpty, 1), ncclSuccess);
  ncclSocketClose(&server_);
  char c;
  ncclSocketOp closedPeer = {NCCL_SOCKET_RECV, &client_, &c, 1, 0};
  EXPECT_EQ(ncclSocketMultiOp(&closedPeer, 1), ncclRemoteError);
}

TEST_F(SocketMicrotest, TryRecv_NonBlockingReportsInProgress) {
  Pair();
  char in[4];
  bool closed = true;
  EXPECT_EQ(ncclSocketTryRecv(&server_, in, sizeof(in), &closed, /*blocking=*/false), ncclInProgress);
  EXPECT_FALSE(closed);
  ::send(client_.socketDescriptor, "ab", 2, 0);
  for (int i = 0; i < 1000; i++) {
    int avail = 0;
    ::ioctl(server_.socketDescriptor, FIONREAD, &avail);
    if (avail) break;
  }
  std::string rest = "cd";
  ScopedHook recvHook(g_recv, [&](int fd, void* buf, size_t n, int f) {
    ssize_t got = ::recv(fd, buf, n, f);
    int err = errno;
    if (got < 0 && !rest.empty()) {  // the rest trickles in, a byte per empty read
      ::send(client_.socketDescriptor, rest.data(), 1, 0);
      rest.erase(0, 1);
    }
    errno = err;
    return got;
  });
  EXPECT_EQ(ncclSocketTryRecv(&server_, in, sizeof(in), &closed, /*blocking=*/false), ncclSuccess);
  EXPECT_TRUE(rest.empty());
  EXPECT_EQ(std::string(in, 4), "abcd");  // once data started, it finishes the message
}

TEST_F(SocketMicrotest, TryRecv_BlockingWaitsForTheFirstBytes) {
  Pair();
  int empty = 0;
  ScopedHook recvHook(g_recv, [&](int fd, void* buf, size_t n, int f) {
    ssize_t got = ::recv(fd, buf, n, f);
    int err = errno;
    if (got < 0 && empty++ == 0) ::send(client_.socketDescriptor, "wxyz", 4, 0);  // only after a miss
    errno = err;
    return got;
  });
  char in[4];
  bool closed = true;
  EXPECT_EQ(ncclSocketTryRecv(&server_, in, sizeof(in), &closed, /*blocking=*/true), ncclSuccess);
  EXPECT_GE(empty, 1);
  EXPECT_FALSE(closed);
  EXPECT_EQ(std::string(in, 4), "wxyz");
}

TEST_F(SocketMicrotest, TryRecv_ReportsAClosedPeer) {
  Pair();
  bool closed = false;
  char in[4];
  ::send(client_.socketDescriptor, "ab", 2, 0);
  ncclSocketClose(&client_);
  ASSERT_EQ(ncclSocketTryRecv(&server_, in, sizeof(in), &closed, /*blocking=*/true), ncclSuccess);
  EXPECT_TRUE(closed);  // closed mid-message
  closed = false;
  ASSERT_EQ(ncclSocketTryRecv(&server_, in, sizeof(in), &closed, /*blocking=*/true), ncclSuccess);
  EXPECT_TRUE(closed);  // closed before any data
}

TEST_F(SocketMicrotest, TryRecv_PropagatesErrors) {
  int p[2];
  ASSERT_EQ(pipe(p), 0);
  Track(p[0]);
  Track(p[1]);
  ncclSocket notSock = client_;
  notSock.socketDescriptor = p[0];
  bool closed;
  char in[4];
  EXPECT_EQ(ncclSocketTryRecv(&notSock, in, sizeof(in), &closed, true), ncclRemoteError);
  Pair();
  volatile uint32_t abortFlag = 0;
  server_.abortFlag = &abortFlag;
  ::send(client_.socketDescriptor, "ab", 2, 0);
  ScopedHook recvHook(g_recv, [&](int fd, void* buf, size_t n, int f) {
    ssize_t got = ::recv(fd, buf, n, f);
    if (got > 0) abortFlag = 1;  // aborted once "ab" is in
    return got;
  });
  EXPECT_EQ(ncclSocketTryRecv(&server_, in, sizeof(in), &closed, true), ncclInternalError);
  EXPECT_EQ(std::string(in, 2), "ab");
}

#ifndef NCCL_TLS_BACKEND_OPENSSL3
// With TLS state attached, I/O goes through crypt.cc, which this build has no backend for.
TEST_F(SocketMicrotest, Progress_TlsStateRoutesThroughCrypt) {
  Pair();
  client_.crypto = (ncclSocketCrypto*)malloc(16);
  ScopedHook sendHook(g_send, [](int fd, const void* buf, size_t n, int f) { return ::send(fd, buf, n, f); });
  ScopedHook recvHook(g_recv, [](int fd, void* buf, size_t n, int f) { return ::recv(fd, buf, n, f); });
  char buf[4] = {};
  int offset = 0;
  EXPECT_EQ(ncclSocketProgress(NCCL_SOCKET_SEND, &client_, buf, sizeof(buf), &offset), ncclInternalError);
  EXPECT_EQ(ncclSocketProgress(NCCL_SOCKET_RECV, &client_, buf, sizeof(buf), &offset), ncclInternalError);
  EXPECT_EQ(sendHook.calls + recvHook.calls, 0);  // neither reached the plain socket
  offset = sizeof(buf);  // nothing left to move: no call
  log_.Clear();
  EXPECT_EQ(ncclSocketProgress(NCCL_SOCKET_SEND, &client_, buf, sizeof(buf), &offset), ncclSuccess);
  EXPECT_TRUE(log_.Text().empty());
}
#endif

}  // namespace
