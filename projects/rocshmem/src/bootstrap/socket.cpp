/******************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation.
 * Modifications Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <cstring>

#include "envvar.hpp"
#include "socket.hpp"
#include "utils.hpp"
#include "util.hpp"

namespace rocshmem {

#define ROCSHMEM_SOCKET_SEND 0
#define ROCSHMEM_SOCKET_RECV 1

/* Format a string representation of a (union SocketAddress *)
 * socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char* SocketToString(union SocketAddress* addr, char* buf,
			   const int numericHostForm /*= 1*/) {
  if (buf == NULL || addr == NULL) return NULL;
  struct sockaddr* saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  (void)getnameinfo(saddr, sizeof(union SocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

// Equivalent with ($ cat /proc/sys/net/ipv4/tcp_fin_timeout)
static int getTcpFinTimeout() {
  std::ifstream ifs("/proc/sys/net/ipv4/tcp_fin_timeout");
  if (!ifs.is_open()) {
    ERROR("open /proc/sys/net/ipv4/tcp_fin_timeout failed errno %d\n", errno);
    return -1;
  }
  int timeout;
  ifs >> timeout;
  return timeout;
}

static uint16_t socketToPort(union SocketAddress* addr) {
  struct sockaddr* saddr = &addr->sa;
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static int envSocketFamily(void) {
  // envvar::types::socket_family enum is defined directly from AF_* constants
  return static_cast<int>(envvar::bootstrap::socket_family.get_value());
}

static int findInterfaces(const char* prefixList, char* names, union SocketAddress* addrs,
			  int sock_family, int maxIfNameSize, int maxIfs) {
#ifdef DEBUG
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    DPRINTF("Found interface %s:%s\n", interface->ifa_name,
          SocketToString((union SocketAddress*)interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != AF_UNSPEC && family != sock_family) continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names + i * maxIfNameSize) == 0) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names + found * maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      std::memcpy(addrs + found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static bool matchSubnet(struct ifaddrs local_if, union SocketAddress* remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;                    // IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  // Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    ERROR("Net : Unsupported address family type\n");
    return false;
  }
}

int FindInterfaceMatchSubnet(char* ifNames, union SocketAddress* localAddrs, union SocketAddress* remoteAddr,
                             int ifNameMaxSize, int maxIfs) {
#ifdef DEBUG
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  char line_a[SOCKET_NAME_MAXLEN + 1];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    std::memcpy(localAddrs + found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames + found * ifNameMaxSize, interface->ifa_name, ifNameMaxSize);

    DPRINTF("NET : Found interface %s:%s in the same subnet as remote address %s\n",
          interface->ifa_name, SocketToString(localAddrs + found, line), SocketToString(remoteAddr, line_a));
    found++;
    if (found == maxIfs) break;
  }

  if (found == 0) {
    ERROR("Net : No interface found in the same subnet as remote address %s\n",
         SocketToString(remoteAddr, line_a));
  }
  freeifaddrs(interfaces);
  return found;
}

void SocketGetAddrFromString(union SocketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    ERROR("Net : string is null\n");
    return;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      ERROR("Net : No valid <IPv4_or_hostname>:<port> pair found\n");
      return;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      ERROR("Net : error encountered when getting address info : %s\n", gai_strerror(rv));
      return;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      std::memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;  // IPv4
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);  // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      std::memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;      // IPv6
      sin6.sin6_port = htons(ni.port);  // port
      sin6.sin6_flowinfo = 0;           // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;           // should be global scope, set to 0
    } else {
      ERROR("Net : unsupported IP family\n");
      return;
    }

    freeaddrinfo(p);  // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      ERROR("Net : No valid [IPv6]:port pair found\n");
      return;
    }
    bool global_scope = (j == -1 ? true : false);  // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);
    int port = atoi(port_str);

    // If not global scope, we need the intf name
    if (!global_scope)
        strncpy(if_name, ip_port_pair + j + 1, i - j - 1);

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                                      // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));                   // IP address
    sin6.sin6_port = htons(port);                                     // port
    sin6.sin6_flowinfo = 0;                                           // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
}

int FindInterfaces(char* ifNames, union SocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs,
                   const char* inputIfName) {
  static int shownIfName = 0;
  int nIfs = 0;

  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();

  // User specified interface
  const std::string& socketIfname = envvar::bootstrap::socket_ifname;
  if (inputIfName) {
    DPRINTF("using iterface %s", inputIfName);
    nIfs = findInterfaces(inputIfName, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  } else if (socketIfname != "") {
    // Specified by user : find or fail
    if (shownIfName++ == 0) DPRINTF ("ROCSHMEM_SOCKET_IFNAME set to %s", socketIfname.c_str());
    nIfs = findInterfaces(socketIfname.c_str(), ifNames, ifAddrs, sock_family,
                          ifNameMaxSize, maxIfs);
  } else {
    // Try to automatically pick the right one
    // Look for anything (but not docker or lo)
    if (nIfs == 0) nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family,
                                         ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0) nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family,
                                         ifNameMaxSize, maxIfs);
    if (nIfs == 0) nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family,
                                         ifNameMaxSize, maxIfs);
  }
  return nIfs;
}

Socket::Socket(const SocketAddress* addr, uint64_t magic, enum SocketType type, volatile uint32_t* abortFlag,
               int asyncFlag) {
  fd_ = -1;
  acceptFd_ = -1;
  connectRetries_ = 0;
  acceptRetries_ = 0;
  abortFlag_ = abortFlag;
  asyncFlag_ = asyncFlag;
  state_ = SocketStateInitialized;
  magic_ = magic;
  type_ = type;

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    std::memcpy(&addr_, addr, sizeof(union SocketAddress));
    family = addr_.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN + 1];
      ERROR("SocketInit: connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)\n",
	   SocketToString(&addr_, line), family, (int)AF_INET, (int)AF_INET6);
      return;
    }
    salen_ = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);

    /* Connect to a hostname / port */
    fd_ = ::socket(family, SOCK_STREAM, 0);
    if (fd_ == -1) {
      ERROR("socket creation failed %d\n", errno);
      return;
    }
  } else {
    memset(&addr_, 0, sizeof(union SocketAddress));
  }

  /* Set socket as non-blocking if async or if we need to be able to abort */
  if ((asyncFlag_ || abortFlag_) && fd_ >= 0) {
    int flags = fcntl(fd_, F_GETFL);
    if (flags == -1) {
      ERROR("fcntl(F_GETFL) failed errno %d\n", errno);
      return;
    }
    if (fcntl(fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
      ERROR("fcntl(F_SETFL) failed errno %d\n", errno);
      return;
    }
  }
}

Socket::~Socket() { close(); }

void Socket::bind() {
  if (fd_ == -1) {
    ERROR("file descriptor is -1\n");
    return;
  }

  if (socketToPort(&addr_)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) != 0) {
      ERROR("::setsockopt(SO_REUSEADDR | SO_REUSEPORT) failed errno %d\n", errno);
      return;
    }
#else
    if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
      ERROR("setsockopt(SO_REUSEADDR) failed errno %d\n", errno);
      return;
    }
#endif
  }

  int finTimeout = getTcpFinTimeout();
  int retrySecs = finTimeout + 1;
  int remainSecs = retrySecs;

  // addr port should be 0 (Any port)
  while (::bind(fd_, &addr_.sa, salen_) != 0) {
    // upon EADDRINUSE, retry up to for (finTimeout + 1) seconds
    if (errno != EADDRINUSE) {
      ERROR("bind failed errno %d\n", errno);
      return;
    }
    if (remainSecs > 0) {
      DPRINTF("No available ephemeral ports found, will retry after 1 second");
      sleep(1);
      remainSecs--;
    } else {
      ERROR("No available ephemeral ports found for %d seconds \n", retrySecs);
      return;
    }
  }

  /* Get the assigned Port */
  socklen_t size = salen_;
  if (::getsockname(fd_, &addr_.sa, &size) != 0) {
    ERROR("getsockname failed errno %d\n", errno);
    return;
  }
  state_ = SocketStateBound;
}

void Socket::bindAndListen() {
#ifdef DEBUG
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  bind();
  DPRINTF("Listening on socket %s\n", SocketToString(&addr_, line));

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  if (::listen(fd_, 16384) != 0) {
    ERROR("listen failed errno %d\n", errno);
    return;
  }
  state_ = SocketStateReady;
}

void Socket::connect(int64_t timeout) {
#ifdef DEBUG
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  Timer timer;
  const int one = 1;

  if (fd_ == -1) {
    ERROR("file descriptor is -1\n");
    return;
  }

  if (state_ != SocketStateInitialized) {
    ERROR("wrong socket state %d\n", state_);
    return;
  }
  DPRINTF("Connecting to socket %s \n", SocketToString(&addr_, line));

  if (setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)) != 0) {
    DPRINTF("setsockopt(TCP_NODELAY) failed, errno %d\n", errno);
    return;
  }

  state_ = SocketStateConnecting;
  do {
    progressState();
    if (timeout > 0 && timer.elapsed() > timeout) {
      ERROR("connect timeout\n");
      return;
    }
  } while (asyncFlag_ == 0 && (abortFlag_ == NULL || *abortFlag_ == 0) &&
           (state_ == SocketStateConnecting || state_ == SocketStateConnectPolling || state_ == SocketStateConnected));

  if (abortFlag_ && *abortFlag_ != 0) {
    ERROR("aborted\n");
    return;
  }
}

void Socket::accept(const Socket* listenSocket, int64_t timeout) {
  Timer timer;

  if (listenSocket == NULL) {
    ERROR("listenSocket is NULL\n");
    return;
  }
  if (listenSocket->getState() != SocketStateReady) {
    ERROR("listenSocket is in error state %u\n", listenSocket->getState());
    return;
  }

  if (acceptFd_ == -1) {
    fd_ = listenSocket->getFd();
    connectRetries_ = listenSocket->getConnectRetries();
    acceptRetries_ = listenSocket->getAcceptRetries();
    abortFlag_ = listenSocket->getAbortFlag();
    asyncFlag_ = listenSocket->getAsyncFlag();
    magic_ = listenSocket->getMagic();
    type_ = listenSocket->getType();
    addr_ = listenSocket->getAddr();
    salen_ = listenSocket->getSalen();

    acceptFd_ = listenSocket->getFd();
    state_ = SocketStateAccepting;
  }

  do {
    progressState();
    if (timeout > 0 && timer.elapsed() > timeout) {
      ERROR("accept timeout\n");
      return;
    }
  } while (asyncFlag_ == 0 && (abortFlag_ == NULL || *abortFlag_ == 0) &&
           (state_ == SocketStateAccepting || state_ == SocketStateAccepted));

  if (abortFlag_ && *abortFlag_ != 0) {
    ERROR("aborted\n");
    return;
  }
}

void Socket::send(void* ptr, int size) {
  int offset = 0;
  if (state_ != SocketStateReady) {
    ERROR("socket state (%d) is not ready\n", state_);
    return;
  }
  socketWait(ROCSHMEM_SOCKET_SEND, ptr, size, &offset);
}

void Socket::recv(void* ptr, int size) {
  int offset = 0;
  if (state_ != SocketStateReady) {
    ERROR("socket state (%d) is not read\n", state_);
    return;
  }
  socketWait(ROCSHMEM_SOCKET_RECV, ptr, size, &offset);
}

void Socket::recvUntilEnd(void* ptr, int size, int* closed) {
  int offset = 0;
  *closed = 0;
  if (state_ != SocketStateReady) {
    ERROR("socket state (%d) is not ready in recvUntilEnd\n", state_);
    return;
  }

  int bytes = 0;
  char* data = (char*)ptr;

  do {
    bytes = ::recv(fd_, data + (offset), size - (offset), 0);
    if (bytes == 0) {
      *closed = 1;
      return;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN && state_ != SocketStateClosed) {
        ERROR("recv until end failed errno %d\n", errno);
        return;
      } else {
        bytes = 0;
      }
    }
    (offset) += bytes;
    if (abortFlag_ && *abortFlag_ != 0) {
      ERROR("aborted\n");
      return;
    }
  } while (bytes > 0 && (offset) < size);
}

void Socket::close() {
  if (fd_ >= 0) ::close(fd_);
  state_ = SocketStateClosed;
  fd_ = -1;
}

void Socket::progressState() {
  if (state_ == SocketStateAccepting) {
    tryAccept();
  }
  if (state_ == SocketStateAccepted) {
    finalizeAccept();
  }
  if (state_ == SocketStateConnecting) {
    startConnect();
  }
  if (state_ == SocketStateConnectPolling) {
    pollConnect();
  }
  if (state_ == SocketStateConnected) {
    finalizeConnect();
  }
}

void Socket::tryAccept() {
  socklen_t socklen = sizeof(union SocketAddress);
  fd_ = ::accept(acceptFd_, &addr_.sa, &socklen);
  if (fd_ != -1) {
    state_ = SocketStateAccepted;
  } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
    ERROR("accept failed (fd %d) errno %d\n", acceptFd_, errno);
  } else {
    usleep(SLEEP_INT);
    if (++acceptRetries_ % 1000 == 0)
      DPRINTF("tryAccept: Call to try accept returned %s, retrying", strerror(errno));
  }
}

void Socket::finalizeAccept() {
  uint64_t magic;
  enum SocketType type;
  int received = 0;
  socketProgress(ROCSHMEM_SOCKET_RECV, &magic, sizeof(magic), &received);
  if (received == 0) return;
  socketWait(ROCSHMEM_SOCKET_RECV, &magic, sizeof(magic), &received);
  if (magic != magic_) {
    ERROR("finalizeAccept: wrong magic %lx != %lx\n", magic, magic_);
    ::close(fd_);
    fd_ = -1;
    // Ignore spurious connection and accept again
    state_ = SocketStateAccepting;
    return;
  } else {
    received = 0;
    socketWait(ROCSHMEM_SOCKET_RECV, &type, sizeof(type), &received);
    if (type != type_) {
      state_ = SocketStateError;
      ::close(fd_);
      fd_ = -1;
      ERROR("wrong socket type %d != %d \n", type, type_);
      return;
    } else {
      state_ = SocketStateReady;
    }
  }
}

void Socket::startConnect() {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = ::connect(fd_, &addr_.sa, salen_);
  if (ret == 0) {
    state_ = SocketStateConnected;
    return;
  } else if (errno == EINPROGRESS) {
    state_ = SocketStateConnectPolling;
    return;
  } else if (errno == ECONNREFUSED || errno == ETIMEDOUT) {
    usleep(SLEEP_INT);
    if (++connectRetries_ % 1000 == 0) DPRINTF("Call to connect returned %s, retrying", strerror(errno));
    return;
  } else {
    char line[SOCKET_NAME_MAXLEN + 1];
    state_ = SocketStateError;
    ERROR("connect to %s failed, errno %d\n", SocketToString(&addr_, line), errno);
    return;
  }
}

void Socket::pollConnect() {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = fd_;
  pfd.events = POLLOUT;
  ret = ::poll(&pfd, 1, timeout);
  if (ret == -1) {
    ERROR("poll failed errno %d\n", errno);
    return;
  }
  if (ret == 0) return;

  /* check socket status */
  if ((ret == 1 && (pfd.revents & POLLOUT)) == 0) {
    ERROR("poll failed\n");
    return;
  }
  if (getsockopt(fd_, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen) == -1) {
    ERROR("getsockopt failed, errno %d\n", errno);
    return;
  }

  if (ret == 0) {
    state_ = SocketStateConnected;
  } else if (ret == ECONNREFUSED || ret == ETIMEDOUT) {
    if (++connectRetries_ % 1000 == 0) {
      DPRINTF("Call to connect returned %s, retrying", strerror(errno));
    }
    usleep(SLEEP_INT);

    ::close(fd_);
    fd_ = ::socket(addr_.sa.sa_family, SOCK_STREAM, 0);
    state_ = SocketStateConnecting;
  } else if (ret != EINPROGRESS) {
    state_ = SocketStateError;
    ERROR("connect failed \n");
    return;
  }
}

void Socket::finalizeConnect() {
  int sent = 0;
  socketProgress(ROCSHMEM_SOCKET_SEND, &magic_, sizeof(magic_), &sent);
  if (sent == 0) return;
  socketWait(ROCSHMEM_SOCKET_SEND, &magic_, sizeof(magic_), &sent);
  sent = 0;
  socketWait(ROCSHMEM_SOCKET_SEND, &type_, sizeof(type_), &sent);
  state_ = SocketStateReady;
}

void Socket::socketProgressOpt(int op, void* ptr, int size, int* offset, int block, int* closed) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;

  do {
    if (op == ROCSHMEM_SOCKET_RECV) bytes = ::recv(fd_, data + (*offset), size - (*offset), block ? 0 : MSG_DONTWAIT);
    if (op == ROCSHMEM_SOCKET_SEND)
      bytes = ::send(fd_, data + (*offset), size - (*offset), block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    if (op == ROCSHMEM_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        ERROR("recv failed, errno %d\n", errno);
        return;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
    if (abortFlag_ && *abortFlag_ != 0) {
      ERROR("aborted\n");
      return;
    }
  } while (bytes > 0 && (*offset) < size);
}

void Socket::socketProgress(int op, void* ptr, int size, int* offset) {
  int closed;
  socketProgressOpt(op, ptr, size, offset, 0, &closed);
  if (closed) {
    char line[SOCKET_NAME_MAXLEN + 1];
    ERROR("connection closed by remote peer %s\n", SocketToString(&addr_, line, 0));
    return;
  }
}

void Socket::socketWait(int op, void* ptr, int size, int* offset) {
  while (*offset < size) socketProgress(op, ptr, size, offset);
}

}  // namespace rocshmem
