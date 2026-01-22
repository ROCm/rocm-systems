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

#include <sys/resource.h>

#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#include "bootstrap.hpp"
#include "utils.hpp"
#include "util.hpp"
#include "socket.hpp"

namespace rocshmem {

static void setFilesLimit() {
  rlimit filesLimit;
  if (getrlimit(RLIMIT_NOFILE, &filesLimit) != 0) {
    DPRINTF("getrlimit failed\n");
    return;
  }
  filesLimit.rlim_cur = filesLimit.rlim_max;
  if (setrlimit(RLIMIT_NOFILE, &filesLimit) != 0) {
    DPRINTF("setrlimit failed\n");
    return;
  }
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

struct ExtInfo {
  int rank;
  int nRanks;
  SocketAddress extAddressListenRoot;
  SocketAddress extAddressListen;
};

 void Bootstrap::groupBarrier(const std::vector<int>& ranks) {
  int dummy = 0;
  for (auto rank : ranks) {
    if (rank != this->getRank()) {
      this->send(static_cast<void*>(&dummy), sizeof(dummy), rank, 0);
    }
  }
  for (auto rank : ranks) {
    if (rank != this->getRank()) {
      this->recv(static_cast<void*>(&dummy), sizeof(dummy), rank, 0);
    }
  }
}

 void Bootstrap::groupAllGather(void* allData, int size, const std::vector<int>& ranks) {
   char* data = static_cast<char*>(allData);
   int rank = this->getRank();
   int nRanks = ranks.size();
   int rank_pos = -1;

   // Confirm that rank is in the vectors of ranks
   for (int i = 0; i < ranks.size(); i++) {
     if (rank == ranks[i]) {
       rank_pos = i;
       break;
     }
   }

   if (rank_pos == -1) {
     printf("Bootstrap::groupAllGather: called with process that is not in list of ranks. Aborting\n");
     abort();
   }

   DPRINTF("groupAllGather: rank %d nranks %d size %d\n", rank, nRanks, size);

   int sendto = (rank_pos + 1 + nRanks) % nRanks;
   int recvfrom = (rank_pos - 1 + nRanks) % nRanks;
   for (int i = 0; i < nRanks - 1; i++) {
     size_t rSlice = (rank_pos - i - 1 + nRanks) % nRanks;
     size_t sSlice = (rank_pos - i + nRanks) % nRanks;

     char *tmpsend = data + sSlice * size;
     char *tmprecv = data + rSlice * size;
     this->send(tmpsend, size, ranks[sendto], i);
     this->recv(tmprecv, size, ranks[recvfrom], i);
   }

   DPRINTF("groupAllGather: rank %d nranks %d size %d - DONE\n", rank, nRanks, size);
 }

 void Bootstrap::groupAlltoall(void* allData, int size, const std::vector<int>& ranks) {
   char* data = static_cast<char*>(allData);
   int num_pes = ranks.size();
   int rank = this->getRank();
   int rank_pos = -1;

   // Confirm that rank is in the vectors of ranks
   for (int i = 0; i < ranks.size(); i++) {
     if (rank == ranks[i]) {
       rank_pos = i;
       break;
     }
   }

   if (rank_pos == -1) {
     printf("Bootstrap::groupAlltoall: called with process that is not in list of ranks. Aborting\n");
     abort();
   }

   DPRINTF("groupAlltoall: rank %d nranks %d size %d\n", rank, num_pes, size);

   // Since this is an in-place algorithm, allocate temporary receive buffer
   char *recv_buf = new char[size * num_pes];
   std::memset(recv_buf, 0, num_pes * size);

   // Perform pairwise exchange - local copy is ommitted
   for (int step = 1; step < num_pes; step++) {
     int sendto   = (rank_pos + step) % num_pes;
     int recvfrom = (rank_pos + num_pes - step) % num_pes;

     char *tmpsend = (char*)data + (ptrdiff_t)sendto * size;
     char *tmprecv = (char*)recv_buf + (ptrdiff_t)recvfrom * size;

     this->send(tmpsend, size, ranks[sendto], step /* used as tag */);
     this->recv(tmprecv, size, ranks[recvfrom], step);
   }

   //Since this is an in_place all-to-all, copy data back into the user buffer
   for (int step = 0; step < num_pes; step++) {
     if (step == rank_pos) continue;
     std::memcpy(&data[step*size], &recv_buf[step*size], size);
   }

   DPRINTF("groupAlltoall: rank %d nranks %d size %d DONE \n", rank, num_pes, size);
   delete[] recv_buf;
 }

 void Bootstrap::send(const std::vector<char>& data, int peer, int tag) {
  size_t size = data.size();
  send((void*)&size, sizeof(size_t), peer, tag);
  send((void*)data.data(), data.size(), peer, tag + 1);
}

 void Bootstrap::recv(std::vector<char>& data, int peer, int tag) {
  size_t size;
  recv((void*)&size, sizeof(size_t), peer, tag);
  data.resize(size);
  recv((void*)data.data(), data.size(), peer, tag + 1);
}

struct UniqueIdInternal {
  uint64_t magic;
  union SocketAddress addr;
};
static_assert(sizeof(UniqueIdInternal) <= sizeof(rocshmem_uniqueid_t), "UniqueIdInternal is too large to fit into rocshmem_uniqueid_t");

class TcpBootstrap::Impl {
 public:
  static rocshmem_uniqueid_t createUniqueId();
  static rocshmem_uniqueid_t getUniqueId(const UniqueIdInternal& uniqueId);

  Impl(int rank, int nRanks);
  ~Impl();
  void initialize(const rocshmem_uniqueid_t& uniqueId, int64_t timeoutSec);
  void initialize(const std::string& ifIpPortTrio, int64_t timeoutSec);
  void establishConnections(int64_t timeoutSec);
  rocshmem_uniqueid_t getUniqueId() const;
  int getRank();
  int getNranks();
  int getNranksPerNode();
  std::vector<int> getLocalRanks();
  void allGather(void* allData, int size);
  void send(void* data, int size, int peer, int tag);
  void recv(void* data, int size, int peer, int tag);
  void barrier();
  void close();

 private:
  UniqueIdInternal uniqueId_;
  int rank_;
  int nRanks_;
  int nRanksPerNode_;
  bool netInitialized;
  std::unique_ptr<Socket> listenSockRoot_;
  std::unique_ptr<Socket> listenSock_;
  std::unique_ptr<Socket> ringRecvSocket_;
  std::unique_ptr<Socket> ringSendSocket_;
  std::vector<SocketAddress> peerCommAddresses_;
  std::vector<int> barrierArr_;
  std::unique_ptr<uint32_t> abortFlagStorage_;
  volatile uint32_t* abortFlag_;
  std::thread rootThread_;
  SocketAddress netIfAddr_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<Socket>, PairHash> peerSendSockets_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<Socket>, PairHash> peerRecvSockets_;
  std::vector<int> localRanks_;

  void netSend(Socket* sock, const void* data, int size);
  void netRecv(Socket* sock, void* data, int size);

  std::shared_ptr<Socket> getPeerSendSocket(int peer, int tag);
  std::shared_ptr<Socket> getPeerRecvSocket(int peer, int tag);

  static void assignPortToUniqueId(UniqueIdInternal& uniqueId);
  static void netInit(std::string ipPortPair, std::string interface, SocketAddress& netIfAddr);

  void bootstrapCreateRoot();
  void bootstrapRoot();
  void getRemoteAddresses(Socket* listenSock, std::vector<SocketAddress>& rankAddresses,
                          std::vector<SocketAddress>& rankAddressesRoot, int& rank);
  void sendHandleToPeer(int peer, const std::vector<SocketAddress>& rankAddresses,
                        const std::vector<SocketAddress>& rankAddressesRoot);
};

rocshmem_uniqueid_t TcpBootstrap::Impl::createUniqueId() {
  UniqueIdInternal uniqueId;
  SocketAddress netIfAddr;
  netInit("", "", netIfAddr);
  getRandomData(&uniqueId.magic, sizeof(uniqueId_.magic));
  std::memcpy(&uniqueId.addr, &netIfAddr, sizeof(SocketAddress));
  assignPortToUniqueId(uniqueId);
  return getUniqueId(uniqueId);
}

rocshmem_uniqueid_t TcpBootstrap::Impl::getUniqueId(const UniqueIdInternal& uniqueId) {
  rocshmem_uniqueid_t ret;
  std::memcpy(&ret, &uniqueId, sizeof(uniqueId));
  return ret;
}

TcpBootstrap::Impl::Impl(int rank, int nRanks)
    : rank_(rank),
      nRanks_(nRanks),
      nRanksPerNode_(0),
      netInitialized(false),
      peerCommAddresses_(nRanks, SocketAddress()),
      barrierArr_(nRanks, 0),
      abortFlagStorage_(new uint32_t(0)),
      abortFlag_(abortFlagStorage_.get()) {}

rocshmem_uniqueid_t TcpBootstrap::Impl::getUniqueId() const { return getUniqueId(uniqueId_); }

int TcpBootstrap::Impl::getRank() { return rank_; }

int TcpBootstrap::Impl::getNranks() { return nRanks_; }

std::vector<int>  TcpBootstrap::Impl::getLocalRanks() { return localRanks_; }

void TcpBootstrap::Impl::initialize(const rocshmem_uniqueid_t& uniqueId, int64_t timeoutSec) {
  if (!netInitialized) {
    netInit("", "", netIfAddr_);
    netInitialized = true;
  }

  std::memcpy(&uniqueId_, &uniqueId, sizeof(uniqueId_));
  if (rank_ == 0) {
    bootstrapCreateRoot();
  }

  char line[MAX_IF_NAME_SIZE + 1];
  SocketToString(&uniqueId_.addr, line);
  DPRINTF("rank %d nranks %d - connecting to %s\n", rank_, nRanks_, line);
  establishConnections(timeoutSec);
}

void TcpBootstrap::Impl::initialize(const std::string& ifIpPortTrio, int64_t timeoutSec) {
  // first check if it is a trio
  int nColons = 0;
  for (auto c : ifIpPortTrio) {
    if (c == ':') {
      nColons++;
    }
  }
  std::string ipPortPair = ifIpPortTrio;
  std::string interface = "";
  if (nColons == 2) {
    // we know the <interface>
    interface = ifIpPortTrio.substr(0, ipPortPair.find_first_of(':'));
    ipPortPair = ifIpPortTrio.substr(ipPortPair.find_first_of(':') + 1);
  }

  if (!netInitialized) {
    netInit(ipPortPair, interface, netIfAddr_);
    netInitialized = true;
  }

  uniqueId_.magic = 0xdeadbeef;
  std::memcpy(&uniqueId_.addr, &netIfAddr_, sizeof(SocketAddress));
  SocketGetAddrFromString(&uniqueId_.addr, ipPortPair.c_str());

  if (rank_ == 0) {
    bootstrapCreateRoot();
  }

  establishConnections(timeoutSec);
}

TcpBootstrap::Impl::~Impl() {
  if (abortFlag_) {
    *abortFlag_ = 1;
  }
  if (rootThread_.joinable()) {
    rootThread_.join();
  }
}

void TcpBootstrap::Impl::getRemoteAddresses(Socket* listenSock, std::vector<SocketAddress>& rankAddresses,
                                            std::vector<SocketAddress>& rankAddressesRoot, int& rank) {
  ExtInfo info;
  SocketAddress zero;
  std::memset(&zero, 0, sizeof(SocketAddress));

  {
    Socket sock(nullptr, ROCSHMEM_SOCKET_MAGIC, SocketTypeUnknown, abortFlag_);
    sock.accept(listenSock);
    netRecv(&sock, &info, sizeof(info));
  }

  if (this->nRanks_ != info.nRanks) {
    ERROR("Bootstrap Root : mismatch in rank count from procs %d : %d\n", this->nRanks_, info.nRanks);
    return;
  }

  if (std::memcmp(&zero, &rankAddressesRoot[info.rank], sizeof(SocketAddress)) != 0) {
    ERROR("Bootstrap Root : rank %d of %d has already checked in\n", info.rank, this->nRanks_);
    return;
  }

  // Save the connection handle for that rank
  rankAddressesRoot[info.rank] = info.extAddressListenRoot;
  rankAddresses[info.rank] = info.extAddressListen;
  rank = info.rank;
}

void TcpBootstrap::Impl::sendHandleToPeer(int peer, const std::vector<SocketAddress>& rankAddresses,
                                          const std::vector<SocketAddress>& rankAddressesRoot) {
  int next = (peer + 1) % nRanks_;
  Socket sock(&rankAddressesRoot[peer], uniqueId_.magic, SocketTypeBootstrap, abortFlag_);
  sock.connect();
  netSend(&sock, &rankAddresses[next], sizeof(SocketAddress));
}

void TcpBootstrap::Impl::assignPortToUniqueId(UniqueIdInternal& uniqueId) {
  std::unique_ptr<Socket> socket = std::make_unique<Socket>(&uniqueId.addr, uniqueId.magic, SocketTypeBootstrap);
  socket->bind();
  uniqueId.addr = socket->getAddr();
}

void TcpBootstrap::Impl::bootstrapCreateRoot() {
  listenSockRoot_ = std::make_unique<Socket>(&uniqueId_.addr, uniqueId_.magic, SocketTypeBootstrap, abortFlag_, 0);
  listenSockRoot_->bindAndListen();
  uniqueId_.addr = listenSockRoot_->getAddr();

  rootThread_ = std::thread([this]() {
      // try {
      bootstrapRoot();
      //} catch (const std::exception& e) {
      //if (abortFlag_ && *abortFlag_) r;
      //throw e;
      //}
  });
}

void TcpBootstrap::Impl::bootstrapRoot() {
  int numCollected = 0;
  std::vector<SocketAddress> rankAddresses(nRanks_, SocketAddress());
  // for initial rank <-> root information exchange
  std::vector<SocketAddress> rankAddressesRoot(nRanks_, SocketAddress());

  std::memset(rankAddresses.data(), 0, sizeof(SocketAddress) * nRanks_);
  std::memset(rankAddressesRoot.data(), 0, sizeof(SocketAddress) * nRanks_);
  setFilesLimit();

  DPRINTF("BEGIN bootstrapRoot\n");
  /* Receive addresses from all ranks */
  do {
    int rank;
    getRemoteAddresses(listenSockRoot_.get(), rankAddresses, rankAddressesRoot, rank);
    ++numCollected;
    DPRINTF("Received connect from rank %d total %d/%d\n", rank, numCollected, nRanks_);
  } while (numCollected < nRanks_ && (!abortFlag_ || *abortFlag_ == 0));

  if (abortFlag_ && *abortFlag_) {
    DPRINTF("ABORTED\n");
    return;
  }

  DPRINTF("COLLECTED ALL %d HANDLES\n", nRanks_);

  // Send the connect handle for the next rank in the AllGather ring
  for (int peer = 0; peer < nRanks_; ++peer) {
    sendHandleToPeer(peer, rankAddresses, rankAddressesRoot);
  }

  DPRINTF("DONE bootstrapRoot\n");
}

void TcpBootstrap::Impl::netInit(std::string ipPortPair, std::string interface,
                                 SocketAddress& netIfAddr) {
  char netIfName[MAX_IF_NAME_SIZE + 1];
  if (!ipPortPair.empty()) {
    if (interface != "") {
      // we know the <interface>
      int ret = FindInterfaces(netIfName, &netIfAddr, MAX_IF_NAME_SIZE, 1, interface.c_str());
      if (ret <= 0) {
        ERROR("NET/Socket : No interface named %s found\n", interface.c_str());
        return;
      }
    } else {
      // we do not know the <interface> try to match it next
      SocketAddress remoteAddr;
      SocketGetAddrFromString(&remoteAddr, ipPortPair.c_str());
      if (FindInterfaceMatchSubnet(netIfName, &netIfAddr, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
        ERROR("NET/Socket : No usable listening interface found\n");
        return;
      }
    }

  } else {
    int ret = FindInterfaces(netIfName, &netIfAddr, MAX_IF_NAME_SIZE, 1);
    if (ret <= 0) {
      ERROR("TcpBootstrap : no socket interface found\n");
      return;
    }
  }

  char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
  std::sprintf(line, " %s:", netIfName);
  SocketToString(&netIfAddr, line + strlen(line));
  DPRINTF("TcpBootstrap : Using%s", line);
}

#define TIMEOUT(__exp)                                                      \
  do {                                                                      \
    try {                                                                   \
      __exp;                                                                \
    } catch (const Error& e) {                                              \
      if (e.getErrorCode() == ErrorCode::Timeout) {                         \
        throw Error("TcpBootstrap connection timeout", ErrorCode::Timeout); \
      }                                                                     \
      throw;                                                                \
    }                                                                       \
  } while (0);

void TcpBootstrap::Impl::establishConnections(int64_t timeoutSec) {
  const int64_t connectionTimeoutUs = timeoutSec * 1000000;
  Timer timer;
  SocketAddress nextAddr;
  ExtInfo info;

  DPRINTF("establishConnections: rank %d nranks %d\n", rank_, nRanks_);

  auto getLeftTime = [&]() {
    if (connectionTimeoutUs < 0) {
      // no timeout: always return a large number
      return int64_t(1e9);
    }
    int64_t timeout = connectionTimeoutUs - timer.elapsed();
    if (timeout <= 0) {
      ERROR("TcpBootstrap connection timeout\n");
      return (long int)-1;
    }
    return timeout;
  };

  info.rank = rank_;
  info.nRanks = nRanks_;

  uint64_t magic = uniqueId_.magic;
  // Create socket for other ranks to contact me
  listenSock_ = std::make_unique<Socket>(&netIfAddr_, magic, SocketTypeBootstrap, abortFlag_);
  listenSock_->bindAndListen();
  info.extAddressListen = listenSock_->getAddr();

  {
    // Create socket for root to contact me
    Socket lsock(&netIfAddr_, magic, SocketTypeBootstrap, abortFlag_);
    lsock.bindAndListen();
    info.extAddressListenRoot = lsock.getAddr();

    // stagger connection times to avoid an overload of the root
    auto randomSleep = [](int rank) {
      timespec tv;
      tv.tv_sec = rank / 1000;
      tv.tv_nsec = 1000000 * (rank % 1000);
      DPRINTF("rank %d delaying connection to root by %ld sec %ld nsec\n", rank,
            tv.tv_sec, tv.tv_nsec);
      (void)nanosleep(&tv, NULL);
    };
    if (nRanks_ > 128) {
      randomSleep(rank_);
    }

    // send info on my listening socket to root
    {
      Socket sock(&uniqueId_.addr, magic, SocketTypeBootstrap, abortFlag_);
      //TIMEOUT(sock.connect(getLeftTime()));
      sock.connect(getLeftTime());
      netSend(&sock, &info, sizeof(info));
    }

    // get info on my "next" rank in the bootstrap ring from root
    {
      Socket sock(nullptr, ROCSHMEM_SOCKET_MAGIC, SocketTypeUnknown, abortFlag_);
      //TIMEOUT(sock.accept(&lsock, getLeftTime()));
      sock.accept(&lsock, getLeftTime());
      netRecv(&sock, &nextAddr, sizeof(SocketAddress));
    }
  }

  ringSendSocket_ = std::make_unique<Socket>(&nextAddr, magic, SocketTypeBootstrap, abortFlag_);
  //TIMEOUT(ringSendSocket_->connect(getLeftTime()));
  ringSendSocket_->connect(getLeftTime());
  // Accept the connect request from the previous rank in the AllGather ring
  ringRecvSocket_ = std::make_unique<Socket>(nullptr, ROCSHMEM_SOCKET_MAGIC, SocketTypeUnknown,
                                             abortFlag_);
  //TIMEOUT(ringRecvSocket_->accept(listenSock_.get(), getLeftTime()));
  ringRecvSocket_->accept(listenSock_.get(), getLeftTime());

  // AllGather all listen handlers
  peerCommAddresses_[rank_] = listenSock_->getAddr();
  allGather(peerCommAddresses_.data(), sizeof(SocketAddress));

  DPRINTF("rank %d nranks %d - DONE\n", rank_, nRanks_);
}

int TcpBootstrap::Impl::getNranksPerNode() {
  if (nRanksPerNode_ > 0) return nRanksPerNode_;
  int nRanksPerNode = 0;
  bool useIpv4 = peerCommAddresses_[rank_].sa.sa_family == AF_INET;
  for (int i = 0; i < nRanks_; i++) {
    if (useIpv4) {
      if (peerCommAddresses_[i].sin.sin_addr.s_addr ==
          peerCommAddresses_[rank_].sin.sin_addr.s_addr) {
        localRanks_.push_back(i);
        nRanksPerNode++;
      }
    } else {
      if (std::memcmp(&(peerCommAddresses_[i].sin6.sin6_addr),
                      &(peerCommAddresses_[rank_].sin6.sin6_addr),
                      sizeof(in6_addr)) == 0) {
        localRanks_.push_back(i);
        nRanksPerNode++;
      }
    }
  }
  nRanksPerNode_ = nRanksPerNode;
  return nRanksPerNode_;
}

void TcpBootstrap::Impl::allGather(void* allData, int size) {
  char* data = static_cast<char*>(allData);
  int rank = rank_;
  int nRanks = nRanks_;

  DPRINTF("allGather: rank %d nranks %d size %d\n", rank, nRanks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i = 0; i < nRanks - 1; i++) {
    size_t rSlice = (rank - i - 1 + nRanks) % nRanks;
    size_t sSlice = (rank - i + nRanks) % nRanks;

    // Send slice to the right
    netSend(ringSendSocket_.get(), data + sSlice * size, size);
    // Recv slice from the left
    netRecv(ringRecvSocket_.get(), data + rSlice * size, size);
  }

  DPRINTF("allGather: rank %d nranks %d size %d - DONE\n", rank, nRanks, size);
}

std::shared_ptr<Socket> TcpBootstrap::Impl::getPeerSendSocket(int peer, int tag) {
  auto it = peerSendSockets_.find(std::make_pair(peer, tag));
  if (it != peerSendSockets_.end()) {
    return it->second;
  }
  auto sock = std::make_shared<Socket>(&peerCommAddresses_[peer], uniqueId_.magic,
                                       SocketTypeBootstrap, abortFlag_);
  sock->connect();
  netSend(sock.get(), &rank_, sizeof(int));
  netSend(sock.get(), &tag, sizeof(int));
  peerSendSockets_[std::make_pair(peer, tag)] = sock;
  return sock;
}

std::shared_ptr<Socket> TcpBootstrap::Impl::getPeerRecvSocket(int peer, int tag) {
  auto it = peerRecvSockets_.find(std::make_pair(peer, tag));
  if (it != peerRecvSockets_.end()) {
    return it->second;
  }
  for (;;) {
    auto sock = std::make_shared<Socket>(nullptr, ROCSHMEM_SOCKET_MAGIC, SocketTypeUnknown,
                                         abortFlag_);
    sock->accept(listenSock_.get());
    int recvPeer, recvTag;
    netRecv(sock.get(), &recvPeer, sizeof(int));
    netRecv(sock.get(), &recvTag, sizeof(int));
    peerRecvSockets_[std::make_pair(recvPeer, recvTag)] = sock;
    if (recvPeer == peer && recvTag == tag) {
      return sock;
    }
  }
}

void TcpBootstrap::Impl::netSend(Socket* sock, const void* data, int size) {
  sock->send(&size, sizeof(int));
  sock->send(const_cast<void*>(data), size);
}

void TcpBootstrap::Impl::netRecv(Socket* sock, void* data, int size) {
  int recvSize;
  sock->recv(&recvSize, sizeof(int));
  if (recvSize > size) {
    ERROR("Message truncated : received %d bytes instead of %d\n", recvSize, size);
    return;
  }
  sock->recv(data, std::min(recvSize, size));
}

void TcpBootstrap::Impl::send(void* data, int size, int peer, int tag) {
  auto sock = getPeerSendSocket(peer, tag);
  netSend(sock.get(), data, size);
}

void TcpBootstrap::Impl::recv(void* data, int size, int peer, int tag) {
  auto sock = getPeerRecvSocket(peer, tag);
  netRecv(sock.get(), data, size);
}

void TcpBootstrap::Impl::barrier() { allGather(barrierArr_.data(), sizeof(int)); }

void TcpBootstrap::Impl::close() {
  listenSockRoot_.reset(nullptr);
  listenSock_.reset(nullptr);
  ringRecvSocket_.reset(nullptr);
  ringSendSocket_.reset(nullptr);
  peerSendSockets_.clear();
  peerRecvSockets_.clear();
}

 rocshmem_uniqueid_t TcpBootstrap::createUniqueId() { return Impl::createUniqueId(); }

 TcpBootstrap::TcpBootstrap(int rank, int nRanks) { pimpl_ = std::make_unique<Impl>(rank, nRanks); }

 rocshmem_uniqueid_t TcpBootstrap::getUniqueId() const { return pimpl_->getUniqueId(); }

 int TcpBootstrap::getRank() { return pimpl_->getRank(); }

 int TcpBootstrap::getNranks() { return pimpl_->getNranks(); }

 int TcpBootstrap::getNranksPerNode() { return pimpl_->getNranksPerNode(); }

 std::vector<int> TcpBootstrap::getLocalRanks() { return pimpl_->getLocalRanks(); }

 void TcpBootstrap::send(void* data, int size, int peer, int tag) {
  pimpl_->send(data, size, peer, tag);
}

 void TcpBootstrap::recv(void* data, int size, int peer, int tag) {
  pimpl_->recv(data, size, peer, tag);
}

 void TcpBootstrap::allGather(void* allData, int size) { pimpl_->allGather(allData, size); }

 void TcpBootstrap::initialize(rocshmem_uniqueid_t uniqueId, int64_t timeoutSec) {
  pimpl_->initialize(uniqueId, timeoutSec);
}

 void TcpBootstrap::initialize(const std::string& ipPortPair, int64_t timeoutSec) {
  pimpl_->initialize(ipPortPair, timeoutSec);
}

 void TcpBootstrap::barrier() { pimpl_->barrier(); }

 TcpBootstrap::~TcpBootstrap() { pimpl_->close(); }

}  // namespace rocshmem
