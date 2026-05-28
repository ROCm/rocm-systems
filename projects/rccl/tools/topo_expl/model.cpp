/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "graph.h"
#include "argcheck.h"
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "model.h"
#include "topo.h"

extern NodeModel *node_model;

ncclNet_t ncclNetDummy = {
  "IB",
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0
};

ncclNet_t* ncclNet = &ncclNetDummy;

int ncclNetVersion() {
  return 4;
}

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
int busIdToCudaDev(int64_t busId) {
  return node_model->busIdToCudaDev(busId);
}

static int useMemcpy = 0;

/* Determine if two peers can communicate with P2P */
ncclResult_t p2pCanConnect(int* ret,  struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  if (!info1->hasFineGrain || !info2->hasFineGrain)  {
    *ret = 0;
    return ncclSuccess;
  }

  // Rule out different nodes / isolated containers
  if (info1->hostHash != info2->hostHash || info1->shmDev != info2->shmDev) {
    *ret = 0;
    return ncclSuccess;
  }

  // Check topology / p2p level.
  int intermediateRank;
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, ret, NULL, &intermediateRank));
  if (*ret == 0) return ncclSuccess;
  if (intermediateRank != -1) {
    if (useMemcpy) *ret = 0;
    return ncclSuccess;
  }

  // Check if NET would work better
  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) {
    *ret = 0;
    return ncclSuccess;
  }

  *ret = 1;
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  if (myInfo->pidHash == peerInfo->pidHash) {
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] -> %d[%d] via P2P/common device", channelId, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
      return ncclInternalError;
    } else {
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] -> %d[%lx] via P2P/direct pointer",
          channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    }
  } else {
    INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] -> %d[%lx] via P2P/IPC",
        channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId, int connIndex) {
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, NULL, NULL, NULL },
  { p2pRecvSetup, NULL, NULL, NULL }
};

NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);

/* Determine if two peers can communicate with SHM */
ncclResult_t shmCanConnect(int* ret,  struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;
  if (ncclParamShmDisable() == 1) return ncclSuccess;

  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) return ncclSuccess;

  // Same host?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 hostHash %lx peer2 hostHash %lx", info1->hostHash, info2->hostHash);
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // Common /dev/shm (between containers) ?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 shmDev %lx peer2 shmDev %lx", info1->shmDev, info2->shmDev);
  if (info1->shmDev != info2->shmDev) return ncclSuccess;

  *ret = 1;
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t shmSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  INFO(NCCL_INIT|NCCL_SHM,"Ring %02d : %d[%lx] -> %d[%lx] via direct shared memory", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSuccess;
}

ncclResult_t shmRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  return ncclSuccess;
}

struct ncclTransport shmTransport = {
  "SHM",
  shmCanConnect,
  { shmSendSetup, NULL, NULL, NULL },
  { shmRecvSetup, NULL, NULL, NULL }
};

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);

struct setupReq {
  int rank;
  int localRank;
  int remoteRank;
  int shared;
  int netDev;
  enum ncclTopoGdrMode useGdr;
  int channelId;
  int connIndex;
};

/* Determine if two peers can communicate with NET */
ncclResult_t netCanConnect(int* ret,  struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

ncclResult_t netSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req;

  send->conn.shared = req.shared = graph ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;
  req.netDev = -1;

  int proxyRank = myInfo->rank;
  int64_t netId;
  if (connIndex == NCCL_CONN_IDX_P2P_NET) NCCLCHECK(ncclTopoGetIntraNetDev(comm->topo, myInfo->rank, graph, channelId, 1, &netId, &req.netDev));
  if (req.netDev < 0) NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));

  if (proxyRank == myInfo->rank) {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s%s", channelId, connIndex, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, comm->ncclNet->name, req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  } else {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d(%d)%s%s", channelId, connIndex, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, comm->ncclNet->name, req.netDev,
        proxyRank, req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  }
  *((int*)connectInfo) = proxyRank;
  return ncclSuccess;
}

NCCL_PARAM(NetGdrLevel, "NET_GDR_LEVEL", PATH_PHB);

ncclResult_t netRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
 struct setupReq req;

  recv->conn.shared = req.shared = graph ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;
  req.netDev = -1;

  // Use myInfo->rank as the receiver uses its own NIC
  int proxyRank = myInfo->rank;
  int64_t netId;
  if (connIndex == NCCL_CONN_IDX_P2P_NET) NCCLCHECK(ncclTopoGetIntraNetDev(comm->topo, myInfo->rank, graph, channelId, 0, &netId, &req.netDev));
  if (req.netDev < 0) NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 0, &req.useGdr));

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s%s", channelId, connIndex, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, comm->ncclNet->name, req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netCanConnect,
  { netSendSetup, NULL, NULL, NULL },
  { netRecvSetup, NULL, NULL, NULL }
};

/* Determine if two peers can communicate with NET */
ncclResult_t collNetCanConnect(int* ret,  struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

ncclResult_t collNetSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  int netDev, proxy;
  enum ncclTopoGdrMode useGdr = ncclTopoGdrModeDisable;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, NULL, &netDev, &proxy));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netDev, 1, &useGdr));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, "SHARP", netDev, useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

ncclResult_t collNetRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  int netDev, proxy;
  enum ncclTopoGdrMode useGdr = ncclTopoGdrModeDisable;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, NULL, &netDev,  &proxy));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netDev, 0, &useGdr));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, "SHARP", netDev, useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

struct ncclTransport collNetTransport = {
  "COL",
  collNetCanConnect,
  { collNetSendSetup, NULL, NULL, NULL },
  { collNetRecvSetup, NULL, NULL, NULL }
};

struct ncclTransport* ncclTransports[] = {
  &p2pTransport,
  &shmTransport,
  &netTransport,
  &collNetTransport,
};
