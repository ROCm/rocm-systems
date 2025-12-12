/******************************************************************************
 * Copyright (c) Microsoft Corporation.
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

#ifndef ROCSHMEM_SOCKET_H_
#define ROCSHMEM_SOCKET_H_

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stddef.h>
#include <sys/socket.h>

namespace rocshmem {

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT 1000  // connection retry sleep interval in usec
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define ROCSHMEM_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union SocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum SocketState {
  SocketStateNone = 0,
  SocketStateInitialized = 1,
  SocketStateAccepting = 2,
  SocketStateAccepted = 3,
  SocketStateConnecting = 4,
  SocketStateConnectPolling = 5,
  SocketStateConnected = 6,
  SocketStateBound = 7,
  SocketStateReady = 8,
  SocketStateClosed = 9,
  SocketStateError = 10,
  SocketStateNum = 11
};

enum SocketType {
  SocketTypeUnknown = 0,
  SocketTypeBootstrap = 1,
  SocketTypeProxy = 2,
  SocketTypeNetSocket = 3,
  SocketTypeNetIb = 4
};

const char* SocketToString(union SocketAddress* addr, char* buf, const int numericHostForm = 1);
void SocketGetAddrFromString(union SocketAddress* ua, const char* ip_port_pair);
int FindInterfaceMatchSubnet(char* ifNames, union SocketAddress* localAddrs, union SocketAddress* remoteAddr,
                             int ifNameMaxSize, int maxIfs);
int FindInterfaces(char* ifNames, union SocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs,
                   const char* inputIfName = nullptr);

class Socket {
 public:
  Socket(const SocketAddress* addr = nullptr, uint64_t magic = ROCSHMEM_SOCKET_MAGIC,
         enum SocketType type = SocketTypeUnknown, volatile uint32_t* abortFlag = nullptr, int asyncFlag = 0);
  ~Socket();

  void bind();
  void bindAndListen();
  void connect(int64_t timeout = -1);
  void accept(const Socket* listenSocket, int64_t timeout = -1);
  void send(void* ptr, int size);
  void recv(void* ptr, int size);
  void recvUntilEnd(void* ptr, int size, int* closed);
  void close();

  int getFd() const { return fd_; }
  int getAcceptFd() const { return acceptFd_; }
  int getConnectRetries() const { return connectRetries_; }
  int getAcceptRetries() const { return acceptRetries_; }
  volatile uint32_t* getAbortFlag() const { return abortFlag_; }
  int getAsyncFlag() const { return asyncFlag_; }
  enum SocketState getState() const { return state_; }
  uint64_t getMagic() const { return magic_; }
  enum SocketType getType() const { return type_; }
  SocketAddress getAddr() const { return addr_; }
  int getSalen() const { return salen_; }

 private:
  void tryAccept();
  void finalizeAccept();
  void startConnect();
  void pollConnect();
  void finalizeConnect();
  void progressState();

  void socketProgressOpt(int op, void* ptr, int size, int* offset, int block, int* closed);
  void socketProgress(int op, void* ptr, int size, int* offset);
  void socketWait(int op, void* ptr, int size, int* offset);

  int fd_;
  int acceptFd_;
  int connectRetries_;
  int acceptRetries_;
  volatile uint32_t* abortFlag_;
  int asyncFlag_;
  enum SocketState state_;
  uint64_t magic_;
  enum SocketType type_;

  union SocketAddress addr_;
  int salen_;
};

}  // namespace rocshmem

#endif  // ROCSHMEM_SOCKET_H_
