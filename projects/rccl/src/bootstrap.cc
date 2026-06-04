/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include <unistd.h>
#include <sched.h>
#include <sys/types.h>
#include "proxy.h"
#include "signals.h" // [RCCL]
#include "param.h"
#include "ras.h"
#include <mutex>
#include "os.h"
#include <thread>
#include <chrono>

#define BOOTSTRAP_N_CHECK_ABORT           10000
#define BOOTSTRAP_TAG_CONNECT             (0x1 << 31)
#define BOOTSTRAP_TAG_ALLGATHER           (0x1 << 30)
#define BOOTSTRAP_TAG_COMMSPLIT           (0x1 << 29)
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)
#define BOOTSTRAP_TAG_GROW_BOUNDARY       (0x1 << 27)
// Tag used to exchange reverse-ring listen handles during bidirectional NET
// bootstrap setup (fallback path when no piggybacked handle is available).
// Picked from the non-bit-shifted space so it cannot collide with the
// BOOTSTRAP_TAG_* one-hot tags above.
static constexpr uint32_t BOOTSTRAP_TAG_BIDIR_REV_HANDLE = 0x7E5E5EB1;

#define BOOTSTRAP_INIT_TIME_CREATE 0
#define BOOTSTRAP_INIT_TIME_SEND   1
#define BOOTSTRAP_INIT_TIME_RECV   2
#define BOOTSTRAP_INIT_TIME_RING   3
#define BOOTSTRAP_INIT_TIME_TOTAL  4
#define BOOTSTRAP_INIT_TIME_DELAY  5
#define BOOTSTRAP_INIT_TIME_N      6
#define BOOTSTRAP_INIT_ROOT_WAIT   0
#define BOOTSTRAP_INIT_ROOT_SEND   1
#define BOOTSTRAP_INIT_ROOT_RECV   2
#define BOOTSTRAP_INIT_ROOT_N      3
#define BOOTSTRAP_PROF_OPEN(time) \
  do {                            \
    time = clockNano();           \
  } while (0)
#define BOOTSTRAP_PROF_CLOSE(time) \
  do {                             \
    time = clockNano() - time;     \
  } while (0)

#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))
// returns the first rank associated to the root. must have root >=0
// if root >= n_roots, it does NOT assume periodicity
static int firstRankFromRoot(int root, int n_ranks, int nRoots,int offset) {
  if (root == -1) return 0;
  // only distribute the n_ranks - offset on the roots
  n_ranks -= offset;
  return offset + root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}
// returns the root of a rank, must have rank >=0
// if rank >= n_ranks, it does NOT assume periodicity
static int rootIdFromRank(int rank, int nRanks, int nRoots, int offset) {
  // ranks < offset have no root (id = -1), ranks above the offset will get assigned to their respective root
  if(nRoots == 0 || rank < offset) return -1;
  nRanks -= offset;
  rank -= offset;
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  int D = rmr * (rpr + 1);
  if (rank < D)
    return rank / (rpr + 1);
  else
    return (rank - D) / rpr + rmr;
}
// return the number of child for a root, root will be periodized
static int nRankFromRoot(int root, int nRanks, int nRoots, int offset) {
  if(root == -1) return 0;
  nRanks -= offset;
  int ir = BOOTSTRAP_PID(root, nRoots);
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  return rpr + ((ir < rmr) ? 1 : 0);
}
// return the local id of a given rank for a given root
// root will be periodize, rank will not
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  // any rank for root -1 has a local id that is the rank id
  if(root == -1) return rank;
  int ir = BOOTSTRAP_PID(root, nRoots);
  return rank - firstRankFromRoot(ir, nRanks, nRoots, offset);
}
// Check if the given rank is the first rank from the root
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots, offset));
}

struct bootstrapRootArgs {
  struct ncclSocket* listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
static std::mutex bootstrapNetMutex;

// Net OOB transport (IB/OFI via net plugin): tristate (-1 auto by threshold, 0 off, 1 on).
// Off by default; opt in explicitly via env var or set to -1 to use the threshold gate.
NCCL_PARAM(BootstrapNetEnable,"OOB_NET_ENABLE", 0);

// Bidirectional ring AllGather (N/2 steps).
//   BOOTSTRAP_BIDIR_ALLGATHER : on/off flag for the socket OOB path. Default on; setting
//                               it to 0 falls back to the unidirectional ring. Has no
//                               effect when net OOB is enabled (the net path uses
//                               BOOTSTRAP_BIDIR_NET instead).
//   BOOTSTRAP_BIDIR_NET       : tristate for the net OOB path (-1 auto by threshold,
//                               0 off, 1 on). Off by default; opt in explicitly. Also
//                               requires net OOB itself to be active (NCCL_OOB_NET_ENABLE).
//   BOOTSTRAP_BIDIR_THRESHOLD : rank threshold consulted only when the corresponding
//                               knob is set to -1 (auto). Default 128 ranks (16 nodes
//                               at 8 ppn), motivated by net-OOB plugin regMr/connect
//                               setup overhead being too heavy to amortise below that.
NCCL_PARAM(BootstrapBidirAllGather, "BOOTSTRAP_BIDIR_ALLGATHER",  1);
NCCL_PARAM(BootstrapBidirNet,       "BOOTSTRAP_BIDIR_NET",        0);
NCCL_PARAM(BootstrapBidirThreshold, "BOOTSTRAP_BIDIR_THRESHOLD", 128);

// Returns true when net OOB transport (IB/OFI via net plugin) should be used.
// Tristate: 1 = on, 0 = off (even above threshold), -1 = auto (threshold-gated).
static inline bool bootstrapNetEnabledEffective(int nranks) {
  int64_t v = ncclParamBootstrapNetEnable();
  if (v == 0) return false;
  if (v >= 1) return true;
  // -1 (auto): threshold-gated
  int64_t thr = ncclParamBootstrapBidirThreshold();
  return thr > 0 && nranks >= (int)thr;
}

// kind: 0 = socket OOB, 1 = net (IB) OOB. Setup, dispatch and cleanup all consult this.
// Socket path runs at any nranks>=3. Net path is tristate; when set to -1 the ring-count
// threshold gates it, since the net plugin's MR registration + reverse-pair connect
// overhead must be amortised over enough ranks. Defaults are documented above.
//
// Exposed (non-static) so test/BootstrapBidirTests.cpp can verify the env-var contract.
// Visibility follows the global -fvisibility flag (see src/CMakeLists.txt): production
// and Release/non-Debug test builds compile with -fvisibility=hidden, so the symbol
// stays off librccl.so's exported dynsym table. The BUILD_TESTS + Debug configuration
// switches to -fvisibility=default precisely so rccl-UnitTestsFixturesDebug can link
// against internal helpers like this one — do NOT pin an explicit visibility("hidden")
// here, as the attribute overrides the command-line flag and breaks that link step
// (ld.lld: undefined hidden symbol: bootstrapBidirEnabled).
bool bootstrapBidirEnabled(int nranks, int kind) {
  if (nranks < 3) return false;
  bool netOn = bootstrapNetEnabledEffective(nranks);
  if (kind == 0) return (ncclParamBootstrapBidirAllGather() != 0) && !netOn;
  if (kind == 1) {
    if (!netOn) return false;
    int64_t v = ncclParamBootstrapBidirNet();
    if (v == 0) return false;
    if (v >= 1) return true;
    // v == -1: auto, threshold-gated.
    int64_t thr = ncclParamBootstrapBidirThreshold();
    return thr > 0 && nranks >= (int)thr;
  }
  // Any unrecognised kind (negative, >=2, etc.) falls through to false.
  // This is the contract verified by BootstrapBidir.UnknownKind_ReturnsFalse:
  // callers adding a third transport must extend this gate explicitly.
  return false;
}

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (bootstrapNetInitDone == 0) {
      const char* env = ncclGetEnv("NCCL_COMM_ID");
      int nIfs = 0;
      if (env) {
        union ncclSocketAddress remoteAddr;
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        NCCLCHECK(ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                               &nIfs));
        if (nIfs <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        NCCLCHECK(ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1, &nIfs));
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInvalidUsage;
        }
      }
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line+strlen(line));
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line);
      bootstrapNetInitDone = 1;
    }
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// check abort function
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && COMPILER_ATOMIC_LOAD(flag, std::memory_order_acquire)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}
// send/recv functions
static ncclResult_t netReg(ncclNet_t* net, void* comm, void* data, size_t size, void** handle) {
  NCCLCHECK(net->regMr(comm, data, size, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag, void** sendReq,
                             int* done) {
  if (*done) return ncclSuccess;
  if (!*sendReq) {
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag, void** recvReq,
                             int* done) {
  if (*done) return ncclSuccess;
  if (!*recvReq) {
    size_t size64 = size;
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }
  if (*recvReq) {
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      *recvReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle, void* recvComm,
                                void* recvData, int recvSize, void* recvDataHandle, int tag, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int doneSend = 0, doneRecv = 0;
  void *sendReq = NULL, *recvReq = NULL;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!doneRecv) {
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);
  return ncclSuccess;
}

#define NCCL_NET_OP_SEND 0
#define NCCL_NET_OP_RECV 1

// Net analogue of struct ncclSocketOp / ncclSocketMultiOp: drives an arbitrary number of
// non-blocking netIsend/netIrecv operations to completion in a single thread by round-robin
// polling. Used by the bidirectional bootstrap AllGather to overlap send/recv on both ring
// directions across the four IB comms (forward sendComm/recvComm + reverse pair) without
// spawning std::thread. MRs must be pre-registered by the caller (handle field).
struct ncclNetOp {
  int op;        // NCCL_NET_OP_SEND or NCCL_NET_OP_RECV
  void* comm;    // send or recv comm to operate on
  void* data;    // data pointer (must lie inside the buffer registered as 'handle')
  int size;      // payload size
  void* handle;  // pre-registered MR handle for 'data'
  int tag;       // IB tag
  void* req;     // in-flight NCCL net request (set internally)
  int done;      // completion flag (set internally)
};

static ncclResult_t netMultiOp(ncclNet_t* net, struct ncclNetOp* ops, int numOps,
                               volatile uint32_t* abortFlag) {
  if (ops == NULL || numOps <= 0) {
    WARN("netMultiOp: invalid arguments ops=%p numOps=%d", ops, numOps);
    return ncclInvalidArgument;
  }
  for (int i = 0; i < numOps; i++) {
    if (ops[i].op != NCCL_NET_OP_SEND && ops[i].op != NCCL_NET_OP_RECV) {
      WARN("netMultiOp: invalid op %d at index %d", ops[i].op, i);
      return ncclInvalidArgument;
    }
    if (ops[i].comm == NULL) {
      WARN("netMultiOp: invalid comm at index %d", i);
      return ncclInvalidArgument;
    }
    if (ops[i].size < 0) {
      WARN("netMultiOp: invalid size %d at index %d", ops[i].size, i);
      return ncclInvalidArgument;
    }
    if (ops[i].size > 0 && ops[i].data == NULL) {
      WARN("netMultiOp: NULL data with size %d at index %d", ops[i].size, i);
      return ncclInvalidArgument;
    }
    // Note: handle is allowed to be NULL — the Socket NET plugin's regMr returns
    // success without populating mhandle (no MR concept on TCP). IB/OFI plugins
    // produce a real handle. The plugin's isend/irecv handle this consistently.
    ops[i].req = NULL;
    ops[i].done = 0;
  }
  int abortCounter = 0;
  int completed = 0;
  while (completed < numOps) {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    bool allIssued = true, madeProgress = false;
    for (int i = 0; i < numOps; i++) {
      if (ops[i].done) continue;
      void* prevReq = ops[i].req;
      if (ops[i].op == NCCL_NET_OP_SEND) {
        NCCLCHECK(netIsend(net, ops[i].comm, ops[i].data, ops[i].size,
                           ops[i].handle, ops[i].tag, &ops[i].req, &ops[i].done));
      } else {
        NCCLCHECK(netIrecv(net, ops[i].comm, ops[i].data, ops[i].size,
                           ops[i].handle, ops[i].tag, &ops[i].req, &ops[i].done));
      }
      if (ops[i].done) { completed++; madeProgress = true; }
      else if (ops[i].req != prevReq) madeProgress = true;
      if (!ops[i].done && ops[i].req == NULL) allIssued = false;
    }
    if (allIssued && !madeProgress) sched_yield();
  }
  return ncclSuccess;
}

// Additional socket based functions, first send the size, then send the message
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  if (size > 0)
    NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}
static ncclResult_t socketRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  int actualSize = std::min(recvSize, size);
  if (actualSize > 0)
    NCCLCHECK(ncclSocketRecv(sock, data, actualSize));
  return ncclSuccess;
}
static ncclResult_t socketSendRecv(struct ncclSocket* sendSock, void* sendData, int sendSize, struct ncclSocket* recvSock,
                                   void* recvData, int recvSize) {
  int senderRecvSize;
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

static ncclResult_t socketDoubleSendRecv(struct ncclSocketOp ops[4]) {
  // ops synchronously exchange size then asynchronously exchange data in send->recv->send->recv order
  int senderRecvSize1, senderRecvSize2;
  struct ncclSocketOp sizeOps[4] = {
    {NCCL_SOCKET_SEND, ops[0].sock, &ops[0].size, sizeof(int), 0},
    {NCCL_SOCKET_RECV, ops[1].sock, &senderRecvSize1, sizeof(int), 0},
    {NCCL_SOCKET_SEND, ops[2].sock, &ops[2].size, sizeof(int), 0},
    {NCCL_SOCKET_RECV, ops[3].sock, &senderRecvSize2, sizeof(int), 0}
  };
  NCCLCHECK(ncclSocketMultiOp(sizeOps, 4));
  if (senderRecvSize1 > ops[1].size || senderRecvSize2 > ops[3].size) {
    WARN("Message truncated : received %d,%d bytes instead of %d,%d", senderRecvSize1, senderRecvSize2, ops[1].size, ops[3].size);
    return ncclInternalError;
  }
  ops[1].size = std::min(ops[1].size, senderRecvSize1);
  ops[3].size = std::min(ops[3].size, senderRecvSize2);
  NCCLCHECK(ncclSocketMultiOp(ops, 4));
  return ncclSuccess;
}

// [RCCL] Was a union; now a struct so we can piggyback the reverse-ring listen handle
// for net bidir bootstrap inside the existing root rendezvous (which is O(1) per rank
// and incurs no extra RTT). The 'fwd' member is the forward-ring connection target as
// before (socket address for socket OOB, net handle for net OOB). The 'revHandle' member
// is the reverse-ring listen handle this rank is offering; root forwards each rank's
// revHandle to the next-1 rank in a single pass so that bootstrapBidirRingSetup can
// connect/accept the reverse pair *without* an extra forward-ring netSendRecv exchange
// (saving one full O(N)-latency step on the net bidir init path).
struct ringConnectInfo {
  union {
    union ncclSocketAddress addr;
    char handle[NCCL_NET_HANDLE_MAXSIZE];
  } fwd;
  char revHandle[NCCL_NET_HANDLE_MAXSIZE];
};

struct extInfo {
  int rank;                                  // rank of the process reaching out
  int nranks;                                // total number of ranks
  int iroot;                                 // current root index
  int nroots;                                // total number of roots
  int offset;                                // offset for rank distribution
  union ncclSocketAddress listenRootAddress; // address of my listenSocket for the root
  struct ringConnectInfo connectInfo;
};
#define NET_HANDLE(h, rank)    ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

// Bootstrap-side accept wrapper. ncclSocketAccept's default behavior
// (retryOnBadMagic=true) loops internally on bad-magic events, which can
// monopolize CPU and starve legitimate peer connects when a non-NCCL TCP
// peer hits the bootstrap listen socket. We pass retryOnBadMagic=false and
// drive the retry from here: close the rejected socket, yield 1ms, retry.
// abortFlag is honored between iterations.
static ncclResult_t bootstrapAccept(struct ncclSocket* sock, struct ncclSocket* listenSock,
                                    volatile uint32_t* abortFlag) {
  while (1) {
    if (abortFlag != NULL && __atomic_load_n(abortFlag, __ATOMIC_ACQUIRE) != 0) {
      return ncclInternalError;
    }
    NCCLCHECK(ncclSocketInit(sock));
    NCCLCHECK(ncclSocketAccept(sock, listenSock, /*retryOnBadMagic=*/false));
    if (sock->state != ncclSocketStateBadMagic) return ncclSuccess;
    INFO(NCCL_INIT|NCCL_NET, "bootstrap: rejected bad-magic peer connection on listen sock, retrying accept");
    (void)ncclSocketClose(sock);
    usleep(1000);
  }
}

static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, struct ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct ringConnectInfo)), res, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);
  return res;
}
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  int iroot = 0, nroots = 0, localId = 0;
  int nrecv = 0, n2send = 0, offset = 0;
  struct extInfo info;
  struct ringConnectInfo* rankInfo = NULL;
  union ncclSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  // get zeros for comparison
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  struct ringConnectInfo zeroInfo;
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(struct ringConnectInfo));
  setFilesLimit();

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
  /* Receive addresses from all ranks */
  do {
    struct ncclSocket sock;
    // No abortFlag: bootstrapRoot is a detached thread without a comm.
    NCCLCHECKGOTO(bootstrapAccept(&sock, listenSock, /*abortFlag=*/NULL), res, out);
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);

    if (c == 0) {
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);
      nranks = info.nranks;
      iroot = info.iroot;
      nroots = info.nroots;
      offset  = info.offset;
      // if the number of root > 1, we will receive one extra info from the first local_id of the next root
      n2send = nRankFromRoot(iroot, nranks, nroots, offset);
      // offset>0 automatically means that we need to switch to the multiroot logic
      nrecv = n2send + ((offset > 0 || nroots > 1) ? 1 : 0);
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot || offset != info.offset) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d, offset %d vs %d",
        nranks, info.nranks, nroots, info.nroots, iroot, info.iroot, offset, info.offset);
        goto out;
    }

    localId = localIdFromRoot(info.rank, iroot, nranks, nroots, offset);
    if (localId < 0 || localId >= nrecv) {
      WARN("Bootstrap Root : localId %d is out of range", localId);
      goto out;
    }
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(struct ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }
    // [RCCL] Always store this rank's info immediately. Inline-send below conditions on
    // having BOTH the recipient address AND the prev-of-recipient info already known —
    // any rank that doesn't satisfy that condition gets its (combined) info via the
    // final loop. This avoids partial-piggyback races where some ranks see a populated
    // revHandle and others see zero (which would cause asymmetric fallback behaviour
    // and deadlock in bootstrapBidirRingSetup).
    memcpy(&rankInfo[localId], &info.connectInfo, sizeof(struct ringConnectInfo));

    // Try to inline-send next-info to the just-arrived rank's previous (whose connection
    // address we may already have). For bidir IB we additionally need prev-of-prev's
    // revHandle to be available — otherwise defer to the final loop.
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
    if (prev >= 0 && prev < n2send && memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      int prevprev = (nroots > 1) ? (prev - 1) : BOOTSTRAP_PID(prev - 1, nrecv);
      bool revKnown = (prevprev >= 0 && prevprev < nrecv &&
                       memcmp(&zeroInfo, &rankInfo[prevprev], sizeof(struct ringConnectInfo)) != 0);
      if (revKnown) {
        struct ringConnectInfo outBound = rankInfo[localId];  // = info.connectInfo
        memcpy(outBound.revHandle, rankInfo[prevprev].revHandle, NCCL_NET_HANDLE_MAXSIZE);
        NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &outBound), res, out);
        // Mark as already-sent so the final loop skips it.
        memset(&rankAddressesRoot[prev], 0, sizeof(union ncclSocketAddress));
      }
      // else: defer to final loop
    }

    // Try to inline-send next-info back to the just-arrived rank itself.
    int next = BOOTSTRAP_PID(localId + 1, nrecv);
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(struct ringConnectInfo)) != 0) {
      int recvPrev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
      bool revKnown = (recvPrev >= 0 && recvPrev < nrecv &&
                       memcmp(&zeroInfo, &rankInfo[recvPrev], sizeof(struct ringConnectInfo)) != 0);
      if (revKnown) {
        struct ringConnectInfo outBound = rankInfo[next];
        memcpy(outBound.revHandle, rankInfo[recvPrev].revHandle, NCCL_NET_HANDLE_MAXSIZE);
        NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &outBound), res, out);
      } else {
        // We can't safely inline-send (prev's revHandle unknown); defer by saving the addr.
        memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
      }
    } else {
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }
    ++c;
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);
  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);

  // send the remaining info to the ranks who haven't received anything
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  // here we need to send info only to my own local process
  for (int r = 0; r < n2send; ++r) {
    // use nrecv to periodize: if 1 root, we will send the first one to the last one, if >1 roots we will send the additional one we have received
    int next = BOOTSTRAP_PID(r + 1, nrecv);
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(struct ringConnectInfo)) != 0) {
      // [RCCL] Combined: next.fwd + prev.revHandle, where prev = rank r's previous.
      struct ringConnectInfo outBound = rankInfo[next];
      int prev = (nroots > 1) ? (r - 1) : BOOTSTRAP_PID(r - 1, nrecv);
      if (prev >= 0 && prev < nrecv && memcmp(&zeroInfo, &rankInfo[prev], sizeof(struct ringConnectInfo)) != 0) {
        memcpy(outBound.revHandle, rankInfo[prev].revHandle, NCCL_NET_HANDLE_MAXSIZE);
      } else {
        memset(outBound.revHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
      }
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &outBound), res, out);
    }
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)", timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9, timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9, timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);
out:
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo)
    free(rankInfo);
  if (rankAddressesRoot)
    free(rankAddressesRoot);
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct bootstrapRootArgs* args = NULL;
  std::thread thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;
  thread = std::thread(bootstrapRoot, args);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  thread.detach();
exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // If comm is provided (grow operation), NCCL_COMM_ID should not be set
    if (comm) {
      WARN("ncclCommGetUniqueId should not be called when NCCL_COMM_ID is set");
      return ncclInvalidUsage;
    }
    // Normal init: use NCCL_COMM_ID from environment
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    handle->magic = NCCL_MAGIC;
  } else {
    if (comm) {
      // comm->childCount will be increment in ncclCommGrow for all existing ranks, use +1 here
      handle->magic = hashCombine(comm->magic, comm->childCount + 1);
    } else {
      NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    }
    handle->nRanks = comm ? comm->nRanks : 0;
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot) {
  if (!parent || !handle) {
    WARN("bcastGrowHandle: parent comm and handle must be provided");
    return ncclInvalidArgument;
  }

  // Single rank parent already has the handle, no need to broadcast
  if (parent->nRanks == 1) return ncclSuccess;
  if (isRoot) {
    NCCLCHECK(bootstrapSend(parent->bootstrap, 0, BOOTSTRAP_TAG_GROW_BOUNDARY, handle, sizeof(struct ncclBootstrapHandle)));
    NCCLCHECK(bootstrapSend(parent->bootstrap, parent->nRanks - 1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle, sizeof(struct ncclBootstrapHandle)));
  } else {
    NCCLCHECK(bootstrapRecv(parent->bootstrap, -1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle, sizeof(struct ncclBootstrapHandle)));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

struct bootstrapRing_t {
  union {
    struct {
      void *sendComm, *recvComm;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
      // Reverse ring (large-scale bidirectional net OOB AllGather, opt-in via
      // NCCL_BOOTSTRAP_BIDIR_NET). NULL when disabled. The socket OOB path is always
      // bidirectional but does not need a reverse pair (TCP is full-duplex).
      void *revSendComm, *revRecvComm;
      ncclNetDeviceHandle_t *revSendDevHandle, *revRecvDevHandle;
    } net;
    struct {
      struct ncclSocket recv; // recv from prev
      struct ncclSocket send; // send to next
    } socket;
  };
};
struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
      // Second listen handle for the reverse ring (only valid in bidirectional IB bootstrap).
      void* revComm;
      char  revHandle[NCCL_NET_HANDLE_MAXSIZE];
    } net;
    struct {
      struct ncclSocket fwd; // forward ring listen
    } socket;
  };
};

struct bootstrapState {
  struct bootstrapRing_t ring;
  struct bootstrapListen_t listen;
  ncclNet_t* net;
  uint64_t* peerProxyAddressesUDS;
  union ncclSocketAddress* peerProxyAddresses;
  union ncclSocketAddress* peerP2pAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};
#define STATE_RING(s, f) (s->ring.f)
#define STATE_LISTEN(s, f) (s->listen.f)

// helper functions
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket, union ncclSocketAddress* addr,
                                       ncclSocketType type) {
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(socket));
  NCCLCHECK(ncclSocketGetAddr(socket, addr));
  return ncclSuccess;
}
static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));
  *peerUDS = getPidHash() + randId;
  return ncclSuccess;
}
#define MAX_OOB_DEVS 16
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev) {
  static int devOOB = -1;
  if (devOOB < 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (devOOB < 0) {
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        bool searchNot = userIfEnv && userIfEnv[0] == '^';
        if (searchNot) userIfEnv++;
        bool searchExact = userIfEnv && userIfEnv[0] == '=';
        if (searchExact) userIfEnv++;
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // loop over the device and return the first one matching
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);
          // check against user specified HCAs/ports
          if (matchIfList(props.name, props.port, userIfs, nUserIfs, searchExact) ^ searchNot) {
            // All plain physical devices have been initialized at this point
            devOOB = devId;
            break;
          }
          devId++;
        }
        if (devOOB == -1) {
          if (!searchNot)
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          else
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          return ncclInvalidArgument;
        }
      } else {
        // default choice is device 0
        devOOB = 0;
      }
      // display info on the chosen device
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(devOOB, &props);
      bool hasProp = res == ncclSuccess;
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
  }
  *dev = devOOB;
  return ncclSuccess;
}

static ncclResult_t netRingConnect(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen, char peerHandle[NCCL_NET_HANDLE_MAXSIZE],
                                   void** sendComm, ncclNetDeviceHandle_t** sendDevHandle,
                                   void** recvComm, ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!*sendComm)
      NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
    if (!*recvComm)
      NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);
  return ncclSuccess;
}

static ncclResult_t netRingConnectProgress(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen, char peerHandle[NCCL_NET_HANDLE_MAXSIZE],
                                           void** sendComm, ncclNetDeviceHandle_t** sendDevHandle,
                                           void** recvComm, ncclNetDeviceHandle_t** recvDevHandle, int* done) {
  if (!*sendComm)
    NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
  if (!*recvComm)
    NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  *done = (*sendComm && *recvComm) ? 1 : 0;
  return ncclSuccess;
}

static ncclResult_t netRingConnectFinish(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen, char peerHandle[NCCL_NET_HANDLE_MAXSIZE],
                                         void** sendComm, ncclNetDeviceHandle_t** sendDevHandle,
                                         void** recvComm, ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {
  int abortCounter = 0, done = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    NCCLCHECK(netRingConnectProgress(ctx, net, listen, peerHandle, sendComm, sendDevHandle, recvComm, recvDevHandle, &done));
    if (!done) sched_yield();
  } while (!done);
  return ncclSuccess;
}

// Start the socket ring connect in non-blocking mode. This lets bootstrapInit overlap
// TCP connect/accept handshakes with local proxy/P2P socket setup, without adding
// threads. socketRingConnect() below keeps the old blocking semantics for callers that
// do not have useful work to overlap (notably bootstrapSplit).
static ncclResult_t socketRingConnectStart(ncclSocketAddress* addr, struct ncclSocket* sendSocket, struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic, volatile uint32_t* abortFlag) {
  ncclResult_t ret = ncclSuccess;
  NCCLCHECK(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag, /*asyncFlag*/1));
  NCCLCHECK(ncclSocketConnect(sendSocket));
  int oldAsync = listenSock->asyncFlag;
  listenSock->asyncFlag = 1;
  NCCLCHECKGOTO(ncclSocketInit(recvSocket), ret, exit);
  NCCLCHECKGOTO(ncclSocketAccept(recvSocket, listenSock), ret, exit);
exit:
  listenSock->asyncFlag = oldAsync;
  return ret;
}

static ncclResult_t socketRingConnectFinish(struct ncclSocket* sendSocket, struct ncclSocket* recvSocket, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int sendReady = 0, recvReady = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!sendReady) NCCLCHECK(ncclSocketReady(sendSocket, &sendReady));
    if (!recvReady) NCCLCHECK(ncclSocketReady(recvSocket, &recvReady));
    if (!sendReady || !recvReady) sched_yield();
  } while (!sendReady || !recvReady);
  return ncclSuccess;
}

static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket, struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic, volatile uint32_t* abortFlag) {
  NCCLCHECK(socketRingConnectStart(addr, sendSocket, listenSock, recvSocket, magic, abortFlag));
  NCCLCHECK(socketRingConnectFinish(sendSocket, recvSocket, abortFlag));
  return ncclSuccess;
}

// Set up the reverse direction of the bootstrap ring for the net OOB path:
// revSendComm → prev, revRecvComm ← next. Must be called *after* the forward ring is
// fully connected. Uses one forward-ring exchange to circulate the freshly-created
// reverse listen handle to the previous rank, then performs a mirrored connect/accept —
// symmetric across all ranks, no deadlock by construction.
//
// Socket OOB path no longer needs this: TCP is full-duplex and the new socketRingAllGather
// drives both ring directions over the existing forward socket pair via socketDoubleSendRecv
// (mirrors NCCL 2.28.7).
// Step 1 of bidirectional setup: ensure rev listen exists and resolve prev's revHandle
// (piggybacked or via forward-ring netSendRecv fallback). When it's a piggyback, also
// kick off non-blocking connect()/accept() on the reverse QP so they can make progress
// while the caller does its other local setup work. The actual completion polling is in
// bootstrapBidirRingSetupFinish.
//
// Returns *prevRevHandleOut populated for callers that need it to call Finish later.
// *startedOut == true means connect/accept were issued (Finish will only poll); false
// means the caller chose to call the legacy one-shot bootstrapBidirRingSetup instead
// (typically because piggyback wasn't available).
static ncclResult_t bootstrapBidirRingSetupStart(struct ncclComm* comm, struct bootstrapState* state,
                                                 const char* prevRevHandlePiggyback,
                                                 char prevRevHandleOut[NCCL_NET_HANDLE_MAXSIZE],
                                                 bool* startedOut) {
  *startedOut = false;
  memset(prevRevHandleOut, 0, NCCL_NET_HANDLE_MAXSIZE);
  bool wantNetBidir = bootstrapBidirEnabled(state->nranks, 1);
  if (!wantNetBidir) return ncclSuccess;

  // Listen on the reverse device. Eager listen in bootstrapInit normally already did
  // this; fall back to in-place listen if some other caller didn't.
  if (STATE_LISTEN(state, net.revComm) == NULL) {
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev),
                                 STATE_LISTEN(state, net.revHandle), &STATE_LISTEN(state, net.revComm)));
  }

  // Without a piggybacked prev revHandle we can't kick off connect() yet — the legacy
  // path needs the forward ring up to do netSendRecv. Caller will fall back.
  if (prevRevHandlePiggyback == NULL) return ncclSuccess;
  char zeros[NCCL_NET_HANDLE_MAXSIZE]; memset(zeros, 0, NCCL_NET_HANDLE_MAXSIZE);
  if (memcmp(prevRevHandlePiggyback, zeros, NCCL_NET_HANDLE_MAXSIZE) == 0) return ncclSuccess;
  memcpy(prevRevHandleOut, prevRevHandlePiggyback, NCCL_NET_HANDLE_MAXSIZE);

  // Issue the first non-blocking connect()/accept() so the transport-layer handshake
  // starts now, overlapping with whatever local setup the caller does next
  // (proxy/UDS/peer/RAS, and/or the forward-ring connect handshake).
  NCCLCHECK(state->net->connect(comm->netContext, STATE_LISTEN(state, net.dev), prevRevHandleOut,
                                &STATE_RING(state, net.revSendComm), &STATE_RING(state, net.revSendDevHandle)));
  NCCLCHECK(state->net->accept(STATE_LISTEN(state, net.revComm),
                               &STATE_RING(state, net.revRecvComm), &STATE_RING(state, net.revRecvDevHandle)));
  *startedOut = true;
  return ncclSuccess;
}

// Step 2: poll until the reverse ring connect/accept complete. Caller must have first
// invoked bootstrapBidirRingSetupStart with started==true and saved prevRevHandle.
static ncclResult_t bootstrapBidirRingSetupFinish(struct ncclComm* comm, struct bootstrapState* state,
                                                  const char prevRevHandle[NCCL_NET_HANDLE_MAXSIZE]) {
  bool wantNetBidir = bootstrapBidirEnabled(state->nranks, 1);
  if (!wantNetBidir) return ncclSuccess;
  int abortCounter = 0;
  while (!STATE_RING(state, net.revSendComm) || !STATE_RING(state, net.revRecvComm)) {
    NCCLCHECK(checkAbort(state->abortFlag, &abortCounter));
    bool madeProgress = false;
    if (!STATE_RING(state, net.revSendComm)) {
      void* prevPtr = STATE_RING(state, net.revSendComm);
      NCCLCHECK(state->net->connect(comm->netContext, STATE_LISTEN(state, net.dev), (void*)prevRevHandle,
                                    &STATE_RING(state, net.revSendComm), &STATE_RING(state, net.revSendDevHandle)));
      if (STATE_RING(state, net.revSendComm) != prevPtr) madeProgress = true;
    }
    if (!STATE_RING(state, net.revRecvComm)) {
      void* prevPtr = STATE_RING(state, net.revRecvComm);
      NCCLCHECK(state->net->accept(STATE_LISTEN(state, net.revComm),
                                   &STATE_RING(state, net.revRecvComm), &STATE_RING(state, net.revRecvDevHandle)));
      if (STATE_RING(state, net.revRecvComm) != prevPtr) madeProgress = true;
    }
    if (!madeProgress) sched_yield();
  }
  INFO(NCCL_BOOTSTRAP, "Bootstrap bidirectional ring (net) connected: nranks %d", state->nranks);
  return ncclSuccess;
}

static ncclResult_t bootstrapBidirRingSetup(struct ncclComm* comm, struct bootstrapState* state,
                                            const char* prevRevHandlePiggyback) {
  // Cheap exit when bidirectional net bootstrap is disabled, pointless (≤2 ranks) or below
  // the BOOTSTRAP_BIDIR_THRESHOLD in auto mode: the net plugin's regMr/connect setup is
  // heavy enough that small comms regress.
  bool wantNetBidir = bootstrapBidirEnabled(state->nranks, 1);
  if (!wantNetBidir) return ncclSuccess;

  char prevRevHandle[NCCL_NET_HANDLE_MAXSIZE];
  memset(prevRevHandle, 0, NCCL_NET_HANDLE_MAXSIZE);

  // 1. Reverse listen: piggyback path creates it eagerly inside bootstrapInit so its
  //    handle can ride along with the existing root rendezvous. If that didn't happen
  //    (caller passed NULL prevRevHandlePiggyback or revComm wasn't set up), fall back
  //    to the legacy in-place listen here.
  if (STATE_LISTEN(state, net.revComm) == NULL) {
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev),
                                 STATE_LISTEN(state, net.revHandle), &STATE_LISTEN(state, net.revComm)));
  }

  // 2. Determine prev's revHandle. Best case: piggybacked through the root rendezvous
  //    (zero-cost). Fallback: do the legacy forward-ring netSendRecv exchange (~20ms
  //    on IB at N=64, dominated by 4× regMr/deregMr).
  bool havePiggyback = false;
  if (prevRevHandlePiggyback != NULL) {
    char zeros[NCCL_NET_HANDLE_MAXSIZE];
    memset(zeros, 0, NCCL_NET_HANDLE_MAXSIZE);
    if (memcmp(prevRevHandlePiggyback, zeros, NCCL_NET_HANDLE_MAXSIZE) != 0) {
      memcpy(prevRevHandle, prevRevHandlePiggyback, NCCL_NET_HANDLE_MAXSIZE);
      havePiggyback = true;
    }
  }
  if (!havePiggyback) {
    char* myRevHandle = STATE_LISTEN(state, net.revHandle);
    void *sendH = NULL, *recvH = NULL;
    ncclResult_t regRes = netReg(state->net, STATE_RING(state, net.sendComm), myRevHandle,   NCCL_NET_HANDLE_MAXSIZE, &sendH);
    if (regRes == ncclSuccess) regRes = netReg(state->net, STATE_RING(state, net.recvComm), prevRevHandle, NCCL_NET_HANDLE_MAXSIZE, &recvH);
    if (regRes == ncclSuccess) {
      regRes = netSendRecv(state->net,
                           STATE_RING(state, net.sendComm), myRevHandle,   NCCL_NET_HANDLE_MAXSIZE, sendH,
                           STATE_RING(state, net.recvComm), prevRevHandle, NCCL_NET_HANDLE_MAXSIZE, recvH,
                           BOOTSTRAP_TAG_BIDIR_REV_HANDLE, state->abortFlag);
    }
    if (sendH) netDereg(state->net, STATE_RING(state, net.sendComm), &sendH);
    if (recvH) netDereg(state->net, STATE_RING(state, net.recvComm), &recvH);
    if (regRes != ncclSuccess) return regRes;
  }

  return bootstrapBidirRingSetupFinish(comm, state, prevRevHandle);
}

static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresss,
                                union ncclSocketAddress* peerProxy, uint64_t* peerUDS,
                                struct rasRankInit* rasRanks) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  struct bootstrapRingData {
    union ncclSocketAddress peerAddress;
    union ncclSocketAddress peerProxy;
    uint64_t peerUDS;
    struct rasRankInit rasRank;
  }* ringData = NULL;

  NCCLCHECK(ncclCalloc(&ringData, nRanks));
  // pack
  if (peerAddresss)
    memcpy(&(ringData[rank].peerAddress), peerAddresss + rank, sizeof(union ncclSocketAddress));
  if (peerProxy)
    memcpy(&(ringData[rank].peerProxy), peerProxy + rank, sizeof(union ncclSocketAddress));
  if (peerUDS)
    memcpy(&(ringData[rank].peerUDS), peerUDS + rank, sizeof(uint64_t));
  if (rasRanks)
    memcpy(&(ringData[rank].rasRank), rasRanks + rank, sizeof(*rasRanks));

  // allgather
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData)), res, exit);

  // unpack
  for (int irank = 0; irank < nRanks; ++irank) {
    if (peerAddresss)
      memcpy(peerAddresss + irank, &(ringData[irank].peerAddress), sizeof(union ncclSocketAddress));
    if (peerProxy)
      memcpy(peerProxy + irank, &(ringData[irank].peerProxy), sizeof(union ncclSocketAddress));
    if (peerUDS)
      memcpy(peerUDS + irank, &(ringData[irank].peerUDS), sizeof(uint64_t));
    if (rasRanks)
      memcpy(rasRanks + irank, &(ringData[irank].rasRank), sizeof(*rasRanks));
  }

exit:
  free(ringData);
  return res;
}

static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);

NCCL_PARAM(RasEnable, "RAS_ENABLE", 1);

ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  // char nextPeerHandle[NCCL_NET_HANDLE_MAXSIZE];
  struct bootstrapState* state;
  struct ncclSocket* proxySocket;
  struct ncclSocket sock, listenSockRoot;
  struct extInfo info = {0};
  struct ringConnectInfo nextPeer;
  bool performRasAddRanks = true;
  struct rasRankInit* rasRanks = nullptr;
  // Bidirectional net OOB split-setup state. Stored at function scope so the prev
  // revHandle survives across the local-setup overlap region between
  // bootstrapBidirRingSetupStart and bootstrapBidirRingSetupFinish.
  bool bidirSplitStarted = false;
  char bidirPrevRevHandle[NCCL_NET_HANDLE_MAXSIZE];
  memset(bidirPrevRevHandle, 0, NCCL_NET_HANDLE_MAXSIZE);

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};
  // Multi-root rendezvous can't propagate revHandle cross-root; bidir requires nHandles==1.
  bool wantNetBidir = (nHandles == 1) && bootstrapBidirEnabled(nranks, 1);

  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;

  // Set magic: for grow existing ranks, receive from coordinator; otherwise use handle magic.
  // This is consistent with the magic created in ncclCommGetUniqueId.
  if (handles != NULL) {
    comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic; // state and comm magic set to the first magic ID
  } else if (parent != NULL) {
    comm->magic = state->magic = hashCombine(parent->magic, parent->childCount);
  } else {
    WARN("bootstrapInit: handles and parent are NULL");
    return ncclSystemError;
  }

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  // [RCCL] Register custom signal handlers if requested
  RegisterSignalHandlers();
  // [/RCCL]

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // fill up the info
  info.nranks = nranks;
  info.nroots = nHandles;
  // get the ring connection info
  memset(&nextPeer, 0, sizeof(struct ringConnectInfo));
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);
  if (bootstrapNetEnabledEffective(nranks)) {
    // Create net interface for other ranks to contact me (all gather)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)));
    memcpy(info.connectInfo.fwd.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
    // [RCCL] Eagerly create the reverse listen if we expect to use net bidir bootstrap.
    // The cost (one IB listen ≈ 1ms) is paid here in parallel with the staggered sendToRoot
    // and the resulting handle is piggybacked into info.connectInfo.revHandle so that
    // bootstrapBidirRingSetup can skip its own netSendRecv handle exchange (~20ms saving
    // dominated by 4× regMr/deregMr on the net plugin).
    if (wantNetBidir) {
      NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev),
                                   STATE_LISTEN(state, net.revHandle), &STATE_LISTEN(state, net.revComm)));
      memcpy(info.connectInfo.revHandle, STATE_LISTEN(state, net.revHandle), NCCL_NET_HANDLE_MAXSIZE);
    }
  } else {
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket.fwd), &info.connectInfo.fwd.addr, ncclSocketTypeBootstrap));
  }
  // Create socket for root to contact me using the root's magic
  // For grow operations, offset is parent->nRanks - 1 (last existing rank joins the root)
  // For normal init, offset is 0
  int offset = 0;
  if(comm->isGrow) {
    if(parent != NULL) {
      offset = parent->nRanks - 1;
    } else {
      if(handles != NULL) {
        offset = BOOTSTRAP_HANDLE(handles, 0)->nRanks - 1;
      } else {
        WARN("bootstrapInit: handles and parent are NULL");
        return ncclSystemError;
      }
    }
  }
  int curr_root = rootIdFromRank(rank, nranks, nHandles, offset);
  if(curr_root >= 0) NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot, &info.listenRootAddress, ncclSocketTypeBootstrap));
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // stagger connection times to avoid an overload of the root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles, offset);
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // for socket the message rate in microsec
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles, offset) / msg_rate;
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    std::this_thread::sleep_for(std::chrono::microseconds(musec));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // send info on my listening socket to root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);
  // send contact info to my own root
  info.rank = rank;
  info.iroot = curr_root;
  info.offset = offset;
  if(curr_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));
  if(parent && comm->isGrow && rank != 0) {
    // Grow: Ranks 1 to N-1 use the parent bootstrap to send connection information to the previous rank
    NCCLCHECK(bootstrapSend(parent->bootstrap, rank - 1, 0, &info.connectInfo, sizeof(info.connectInfo)));
  }
  // if needed, send the connection info to the previous root
  // commGrow with more than = 1 rank in the parent comm is a special case of multiroot
  if (((comm->isGrow && parent && (parent->nRanks > 1)) || nHandles > 1) && isFirstFromRoot(rank, curr_root, nranks, nHandles, offset)) {
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles, offset);
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    // only send if the root is valid, existing rank N-1 will use the bootstrapSend just above
    if(prev_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // get info on my "next" rank in the bootstrap ring from root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  if (curr_root >= 0) {
    NCCLCHECK(bootstrapAccept(&sock, &listenSockRoot, comm->abortFlag));
    NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
    NCCLCHECK(ncclSocketClose(&sock));
    NCCLCHECK(ncclSocketClose(&listenSockRoot));
  }
  if (parent && comm->isGrow && rank != parent->nRanks - 1) {
    // Grow: Ranks 0 to N-2 use the parent bootstrap to recv connection information to the next rank. This is consistent with the bootstrapSend above.
    NCCLCHECK(bootstrapRecv(parent->bootstrap, rank + 1, 0, &nextPeer, sizeof(nextPeer)));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // Start the ring connect/accept as early as possible, then overlap its non-blocking
  // progress with local proxy/P2P socket setup below. This is intentionally single-
  // threaded: net connect/accept are plugin-level non-blocking calls, and socket uses
  // ncclSocket async mode plus ncclSocketReady() in the finish step.
  if (bootstrapNetEnabledEffective(nranks)) {
    int ringConnectDone = 0;
    NCCLCHECK(netRingConnectProgress(comm->netContext, state->net, &state->listen, nextPeer.fwd.handle,
                                     &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                     &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), &ringConnectDone));
  } else {
    NCCLCHECK(socketRingConnectStart(&nextPeer.fwd.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket.fwd), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // [RCCL] Bidirectional net OOB: kick off the reverse-ring connect()/accept() now so
  // they overlap with both forward-ring connect and the local proxy/UDS/peer/RAS setup
  // below. This requires a piggybacked prev revHandle (delivered via the root rendezvous
  // or PMIx fast-path) so we can avoid the legacy netSendRecv handle exchange. When
  // piggyback isn't available (shrunk/split comms) we fall back below to the one-shot
  // bootstrapBidirRingSetup that does listen + handle exchange + connect/accept after
  // the forward ring is up.
  if (wantNetBidir) {
    NCCLCHECK(bootstrapBidirRingSetupStart(comm, state, nextPeer.revHandle, bidirPrevRevHandle, &bidirSplitStarted));
  }

  // AllGather all listen handlers
  // in case of failure, those resources will be free'd when calling bootstrapDestroy, so we can return immediatly
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // Initialize RAS
  if (ncclParamRasEnable() == 1) {
    // The RAS thread will take care of freeing the memory allocated below.
    NCCLCHECK(ncclCalloc(&rasRanks, nranks));
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = ncclOsGetPid();
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks+rank) != ncclSuccess) {
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should still participate in the ringAllInfo below as the peers will be waiting for us.
      // Just make sure that the address is clearly invalid...
      memset(rasRanks+rank, '\0', sizeof(*rasRanks));
      performRasAddRanks = false;
    }
  }

  // Finish forward ring connect before using the ring in bootstrapBidirRingSetup and
  // ringAllInfo. By this point the connection handshake has overlapped with the local
  // setup above (proxy listen, UDS lookup, peer P2P listen and RAS payload creation).
  if (bootstrapNetEnabledEffective(nranks)) {
    NCCLCHECKGOTO(netRingConnectFinish(comm->netContext, state->net, &state->listen, nextPeer.fwd.handle,
                                       &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                       &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag),
                  result, fail);
  } else {
    NCCLCHECKGOTO(socketRingConnectFinish(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), state->abortFlag), result, fail);
  }

  // Optionally bring up the second (reverse) ring used by the bidirectional bootstrap
  // AllGather. The reverse listen + handle are piggybacked through the root rendezvous
  // (nextPeer.revHandle = prev's reverse listen), so the heavy synchronous netSendRecv
  // handle exchange inside bootstrapBidirRingSetup is skipped on the hot path.
  // When the split-Start above already kicked off connect/accept (piggyback path), we
  // only need to poll for completion here — it has been progressing in parallel with
  // proxy/UDS/peer/RAS setup and the forward-ring handshake. Otherwise fall back to
  // the legacy one-shot path which also does the listen + handle exchange.
  if (bidirSplitStarted) {
    NCCLCHECKGOTO(bootstrapBidirRingSetupFinish(comm, state, bidirPrevRevHandle), result, fail);
  } else if (wantNetBidir) {
    NCCLCHECKGOTO(bootstrapBidirRingSetup(comm, state, nextPeer.revHandle), result, fail);
  }

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, rasRanks), result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  // Create the service proxy and get the UDS
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result, fail);

  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)", timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  return result;
fail:
  free(proxySocket);
  goto exit;
}

ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* parentRanks) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int prev, next;
  struct ringConnectInfo info = {};
  struct ringConnectInfo nextPeer = {};
  struct ncclSocket* proxySocket = NULL;
  struct bootstrapState* state;

  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  comm->magic = state->magic = magic;

  prev = parentRanks[(rank - 1 + nranks) % nranks];
  next = parentRanks[(rank + 1) % nranks];

  // create a handle for the others to reach out to me
  if (bootstrapNetEnabledEffective(nranks)) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)), ret, fail);
    memcpy(info.fwd.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neighbor to contact me
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket.fwd), &info.fwd.addr, ncclSocketTypeBootstrap));
  }
  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap));

  if (ncclParamRasEnable() == 1) {
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  // Get addr from next rank using the parent's connections
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(struct ringConnectInfo)), ret, fail);
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer, sizeof(struct ringConnectInfo)), ret, fail);
  if (bootstrapNetEnabledEffective(nranks)) {
    NCCLCHECKGOTO(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.fwd.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag),
                  ret, fail);
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.fwd.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket.fwd), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // Mirror the bidirectional ring setup done in bootstrapInit so that split communicators
  // can also benefit from the N/2-step AllGather. bootstrapSplit doesn't go through the
  // root rendezvous, so no piggybacked revHandle is available — pass NULL to use the
  // legacy in-place handle exchange.
  NCCLCHECKGOTO(bootstrapBidirRingSetup(comm, state, NULL), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));
  if (parent->shareResources) {
    /* map local rank to top parent local rank. */
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    // Create the service proxy and get the UDS
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), ret, fail);
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, NULL), ret, fail);
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE", comm, parent, rank, nranks,
        color, key, prev, next);

exit:
  return ret;
fail:
  free(proxySocket);
  goto exit;
}

struct socketAckInfo {
  int rank;
  int tag;
};
static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  struct socketAckInfo ack = (struct socketAckInfo){.rank = state->rank, .tag = tag};
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap, state->abortFlag), ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}
// Bootstrap send/receive functions
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock, int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    // peer < 0 means wildcard (accept from any peer)
    if ((peer < 0 || elem->peer == peer) && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // Search unexpected connections first
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return ncclSuccess;

  // Then look for new connections
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(bootstrapAccept(sock, &STATE_LISTEN(state, peerSocket), state->abortFlag), ret, fail);
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
    // Match: tag must match, and peer must match (peer < 0 means wildcard)
    if (ack.tag == tag && (peer < 0 || ack.rank == peer)) return ncclSuccess;
    // No match: queue for later and try next connection
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);
  }
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait*/true), ret, fail);
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm, void* recvComm, int rank, int nranks, char* data, int size, volatile uint32_t* abortFlag) {
  ncclResult_t res;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;
  NCCLCHECKGOTO(netReg(net, sendComm, data, (size_t)nranks * size, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, (size_t)nranks * size, &recvDataHandle), res, exit);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data, size, sendDataHandle, recvComm, recv_data, size, recvDataHandle, tag, abortFlag), res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  // do not fail in case of error, try to deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}
// Classic unidirectional ring AllGather over the socket OOB path (N-1 rounds).
// Used when BOOTSTRAP_BIDIR_ALLGATHER=0 is set explicitly.
static ncclResult_t socketRingAllGatherUnidir(struct ncclSocket* sendSock, struct ncclSocket* recvSock, int rank, int nranks, char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGatherUnidir started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(socketSendRecv(sendSock, send_data, size, recvSock, recv_data, size), res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGatherUnidir first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}
// Bidirectional ring AllGather over sockets, mirrors NCCL 2.28.7.
// Single shared forward pair drives both ring directions via socketDoubleSendRecv
// (TCP is full-duplex). Algorithmic ⌊N/2⌋ steps vs N-1 for unidirectional
// (for even N the final step is single-directional, so total slices == N-1
// in both even and odd cases).
static ncclResult_t socketRingAllGather(struct ncclSocket* nextSock, struct ncclSocket* prevSock,
                                        int rank, int nranks, char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started: rank=%d nranks=%d", rank, nranks);
  int totalSteps = nranks / 2;
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int step = 0; step < totalSteps; step++) {
    bool isFinalUnidirectional = (step == totalSteps - 1) && (nranks % 2 == 0);
    int sendSliceRing0 = (rank - step + nranks) % nranks;
    int recvSliceRing0 = (rank - step - 1 + nranks) % nranks;
    int sendSliceRing1 = (rank + step) % nranks;
    int recvSliceRing1 = (rank + step + 1) % nranks;
    if (isFinalUnidirectional) {
      NCCLCHECKGOTO(socketSendRecv(nextSock, data + sendSliceRing0 * size, size, prevSock, data + recvSliceRing0 * size, size), res, exit);
    } else {
      struct ncclSocketOp ops[4] = {
        {NCCL_SOCKET_SEND, nextSock, data + sendSliceRing0 * size, size, 0},
        {NCCL_SOCKET_RECV, prevSock, data + recvSliceRing0 * size, size, 0},
        {NCCL_SOCKET_SEND, prevSock, data + sendSliceRing1 * size, size, 0},
        {NCCL_SOCKET_RECV, nextSock, data + recvSliceRing1 * size, size, 0}
      };
      NCCLCHECKGOTO(socketDoubleSendRecv(ops), res, exit);
    }
    if (step == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  // Reported numerator is per-rank "useful AllGather payload" (N-1 slices), not wire bytes,
  // so it stays comparable across the unidir/bidir variants (which exchange the same total
  // payload, just split across one vs two directions). For the first-step bandwidth we count
  // the two slices actually delivered in that step (vs one in the unidir variant), otherwise
  // the bidir "first MB/sec" would read as half of the true rate.
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (2.0 * size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}

// Net OOB variant of the bidirectional ring AllGather. Mirrors the structure
// of socketRingAllGather (the upstream NCCL 2.28.7 socket variant): totalSteps = N/2,
// each step exchanges two slices in opposite directions, last step is a single
// netSendRecv when N is even.
//
// IB-specific design notes:
// - Each comm owns its own QP (and PD on most plugins), so MRs are not shareable
//   across comms; we must register the same data buffer four times (forward+reverse
//   send and recv). MRs are registered upfront and torn down once the loop completes.
// - The four ops per step (ring0_send/ring0_recv on forward QP + ring1_send/ring1_recv
//   on reverse QP) are driven via netMultiOp, which round-robin polls netIsend/netIrecv
//   in a single thread. This replaces the previous std::thread-based forward/reverse
//   split: both directions now share one CPU but progress concurrently in the kernel.
// - Forward and reverse writes target disjoint slice sets (forward → [rank - N/2 ..
//   rank - 1], reverse → [rank + 1 .. rank + (N-1)/2]; rank itself is untouched), so
//   even though both ring0 and ring1 ops are in-flight at the same time, they cannot
//   race on memory.
// - On any failure we still attempt to dereg every successfully-registered handle
//   to avoid leaking pinned host memory.
static ncclResult_t netBiDirRingAllGather(ncclNet_t* net,
                                          void* sendComm, void* recvComm,
                                          void* revSendComm, void* revRecvComm,
                                          int rank, int nranks, char* data, int size,
                                          volatile uint32_t* abortFlag) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  void *sendH = NULL, *recvH = NULL, *revSendH = NULL, *revRecvH = NULL;
  int totalSteps = nranks / 2;

  TRACE(NCCL_BOOTSTRAP, "netBiDirRingAllGather started: rank=%d nranks=%d totalSteps=%d", rank, nranks, totalSteps);

  res = netReg(net, sendComm, data, (size_t)nranks * size, &sendH);
  if (res == ncclSuccess) res = netReg(net, recvComm, data, (size_t)nranks * size, &recvH);
  if (res == ncclSuccess) res = netReg(net, revSendComm, data, (size_t)nranks * size, &revSendH);
  if (res == ncclSuccess) res = netReg(net, revRecvComm, data, (size_t)nranks * size, &revRecvH);
  if (res != ncclSuccess) goto cleanup;

  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int step = 0; step < totalSteps; step++) {
    bool isFinalUnidirectional = (step == totalSteps - 1) && (nranks % 2 == 0);
    int sendSliceRing0 = (rank - step + nranks) % nranks;     // Ring0 send to next
    int recvSliceRing0 = (rank - step - 1 + nranks) % nranks; // Ring0 recv from prev
    int sendSliceRing1 = (rank + step) % nranks;              // Ring1 send to prev
    int recvSliceRing1 = (rank + step + 1) % nranks;          // Ring1 recv from next
    if (isFinalUnidirectional) {
      // Final step on even N: only ring0 over the forward QP pair.
      res = netSendRecv(net, sendComm, data + sendSliceRing0 * size, size, sendH,
                        recvComm, data + recvSliceRing0 * size, size, recvH,
                        /*tag*/step, abortFlag);
    } else {
      // Bidirectional step: 4 in-flight ops driven by single-threaded round-robin polling.
      // Tags are kept identical inside one MultiOp call (each comm has its own tag space
      // on IB; collisions between forward/reverse are impossible because they live on
      // different QPs).
      struct ncclNetOp ops[4] = {
        {NCCL_NET_OP_SEND, sendComm,    data + sendSliceRing0 * size, size, sendH,    /*tag*/step, NULL, 0},
        {NCCL_NET_OP_RECV, recvComm,    data + recvSliceRing0 * size, size, recvH,    /*tag*/step, NULL, 0},
        {NCCL_NET_OP_SEND, revSendComm, data + sendSliceRing1 * size, size, revSendH, /*tag*/step, NULL, 0},
        {NCCL_NET_OP_RECV, revRecvComm, data + recvSliceRing1 * size, size, revRecvH, /*tag*/step, NULL, 0}
      };
      res = netMultiOp(net, ops, 4, abortFlag);
    }
    if (res != ncclSuccess) break;
    if (step == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  // Numerator convention matches socketRingAllGather: per-rank useful AllGather payload
  // (N-1 slices), so the metric is comparable across unidir/bidir variants. First-step
  // numerator is 2*size because the first bidir step delivers two slices (one per direction).
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE,
        "netBiDirRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (2.0 * size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));

cleanup:
  // Best-effort cleanup: try every dereg even if some failed (avoids leaking pinned memory).
  if (sendH)    netDereg(net, sendComm,    &sendH);
  if (recvH)    netDereg(net, recvComm,    &recvH);
  if (revSendH) netDereg(net, revSendComm, &revSendH);
  if (revRecvH) netDereg(net, revRecvComm, &revRecvH);
  return res;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", rank, nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  if (bootstrapNetEnabledEffective(nranks)) {
    // Take the bidirectional path only when bootstrapBidirEnabled() agrees AND the
    // reverse comms were actually set up (defensive against future code paths that
    // may bypass bootstrapBidirRingSetup, e.g. shrunk/split comms).
    bool useBidir = bootstrapBidirEnabled(nranks, 1)
                    && STATE_RING(state, net.revSendComm) != NULL
                    && STATE_RING(state, net.revRecvComm) != NULL;
    if (useBidir) {
      NCCLCHECKGOTO(netBiDirRingAllGather(state->net,
                                          STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm),
                                          STATE_RING(state, net.revSendComm), STATE_RING(state, net.revRecvComm),
                                          rank, nranks, (char*)allData, size, state->abortFlag), res, exit);
    } else {
      NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm), rank, nranks, (char*)allData, size, state->abortFlag), res, exit);
    }
  } else {
    if (bootstrapBidirEnabled(nranks, 0)) {
      NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv),
                                        rank, nranks, (char*)allData, size), res, exit);
    } else {
      NCCLCHECKGOTO(socketRingAllGatherUnidir(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), rank, nranks, (char*)allData, size), res, exit);
    }
  }
exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9, ((size_t)nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", rank, nranks, size);
  return res;
}

static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  if (nranks == 1)
    return ncclSuccess;
  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1] = {0};
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  int prevRank = ranks[(rank - 1 + nranks) % nranks];
  int nextRank = ranks[(rank + 1) % nranks];
  // intraNode bootstrap is done defacto using the socket-based implementation
  struct ncclSocket recvSocket, sendSocket;
  NCCLCHECK(socketConnect(commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket));
  NCCLCHECK(socketAccept(commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket));

  // Intra-node AllGather is socket-only and so cannot reuse bootstrapBidirEnabled
  // (its `!netOn` clause is geared to inter-node OOB selection). Gate purely on the
  // BOOTSTRAP_BIDIR_ALLGATHER knob; socketRingAllGather handles nranks<3 itself
  // (nranks==2 → single bidirectional socketSendRecv; nranks==1 is short-circuited
  // earlier in this function).
  if (ncclParamBootstrapBidirAllGather() != 0) {
    NCCLCHECK(socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size));
  } else {
    NCCLCHECK(socketRingAllGatherUnidir(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size));
  }

  NCCLCHECK(ncclSocketClose(&sendSocket));
  NCCLCHECK(ncclSocketClose(&recvSocket));

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

// [IntraNode] in-place Broadcast
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  if (nranks == 1) return ncclSuccess;
  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i != root) NCCLCHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
    }
  } else {
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData, size));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size, time / 1e9, ((size_t)nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int nranks = state->nranks;
  // close unexpected and return an error if we are not aborting and still operations in the pipe
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (COMPILER_ATOMIC_LOAD(state->abortFlag, std::memory_order_acquire) == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }
  if (bootstrapNetEnabledEffective(nranks)) {
    NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
    NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
    NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
    // Reverse-direction net comms only exist when bidirectional bootstrap was enabled.
    if (STATE_RING(state, net.revSendComm)) NCCLCHECK(state->net->closeSend(STATE_RING(state, net.revSendComm)));
    if (STATE_RING(state, net.revRecvComm)) NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.revRecvComm)));
    if (STATE_LISTEN(state, net.revComm))   NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.revComm)));
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket.fwd)));
  }
  // close the p2p socket
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  // proxy things are free'd elsewhere
  free(state->peerP2pAddresses);
  free(state);
  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // when aborting we need to close the proxy here (maybe?)
  free(state->peerProxyAddresses);
  free(state->peerProxyAddressesUDS);
  NCCLCHECK(bootstrapClose(commState));
  return ncclSuccess;
}
