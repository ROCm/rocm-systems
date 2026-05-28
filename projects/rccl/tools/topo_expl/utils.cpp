/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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
#include <cstdarg>
#include "xml.h"
#include "coll_net.h"
#include "model.h"
#include "utils.h"
#include "rocm_smi/rocm_smi.h"

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+2] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv", "AllToAllPivot" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

extern NodeModel *node_model;

RCCL_PARAM(P2pNetDisable, "P2P_NET_DISABLE", 0);
RCCL_PARAM(PivotAlltoallEnable, "PIVOT_ALLTOALL_ENABLE", 1);
RCCL_PARAM(LL128ForceEnable, "LL128_FORCE_ENABLE", 0);

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 0);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

thread_local int ncclDebugNoWarn = 0;
ncclCollNet_t* ncclCollNet = NULL;

// Get current Compute Capability
int ncclCudaCompCap() {
  int ccMajor = 1, ccMinor = 0;
  return ccMajor*10+ccMinor;
}

ncclResult_t int64ToBusId(int64_t id, char* busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
  return ncclSuccess;
}

ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
  const int size = strlen(busId);
  char* hexStr;
  NCCLCHECK(ncclCalloc(&hexStr, size));
  int hexOffset = 0;
  for (int i=0; i<size; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  free(hexStr);
  return ncclSuccess;
}

void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align) {
  // `me->hunks` points to the top of the stack non-empty hunks. Hunks above
  // this (reachable via `->above`) are empty.
  struct Hunk* top = me->topFrame.hunk;
  size_t mallocSize = 0;

  // If we have lots of space left in hunk but that wasn't enough then we'll
  // allocate the object unhunked.
  if (me->topFrame.end - me->topFrame.bumper >= 8<<10)
    goto unhunked;

  // If we have another hunk (which must be empty) waiting above this one and
  // the object fits then use that.
  if (top && top->above) {
    struct Hunk* top1 = top->above;
    uintptr_t uobj = (reinterpret_cast<uintptr_t>(top1) + sizeof(struct Hunk) + align-1) & -uintptr_t(align);
    if (uobj + size <= reinterpret_cast<uintptr_t>(top1) + top1->size) {
      me->topFrame.hunk = top1;
      me->topFrame.bumper = uobj + size;
      me->topFrame.end = reinterpret_cast<uintptr_t>(top1) + top1->size;
      return reinterpret_cast<void*>(uobj);
    }
  }

  { // If the next hunk we're going to allocate wouldn't be big enough but the
    // Unhunk proxy fits in the current hunk then go allocate as unhunked.
    size_t nextSize = (top ? top->size : 0) + (64<<10);
    constexpr size_t maxAlign = 64;
    if (nextSize < sizeof(struct Hunk) + maxAlign + size) {
      uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk)-1) & -uintptr_t(alignof(Unhunk));
      if (uproxy + sizeof(struct Unhunk) <= me->topFrame.end)
        goto unhunked;
    }

    // At this point we must need another hunk, either to fit the object
    // itself or its Unhunk proxy.
    mallocSize = nextSize;
    INFO(NCCL_ALLOC, "%s:%d memory stack hunk malloc(%llu)", __FILE__, __LINE__, (unsigned long long)mallocSize);
    struct Hunk *top1 = (struct Hunk*)malloc(mallocSize);
    if (top1 == nullptr) goto malloc_exhausted;
    top1->size = nextSize;
    top1->above = nullptr;
    if (top) top->above = top1;
    top = top1;
    me->topFrame.hunk = top;
    me->topFrame.end = reinterpret_cast<uintptr_t>(top) + nextSize;
    me->topFrame.bumper = reinterpret_cast<uintptr_t>(top) + sizeof(struct Hunk);
  }

  { // Try to fit object in the new top hunk.
    uintptr_t uobj = (me->topFrame.bumper + align-1) & -uintptr_t(align);
    if (uobj + size <= me->topFrame.end) {
      me->topFrame.bumper = uobj + size;
      return reinterpret_cast<void*>(uobj);
    }
  }

unhunked:
  { // We need to allocate the object out-of-band and put an Unhunk proxy in-band
    // to keep track of it.
    uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk)-1) & -uintptr_t(alignof(Unhunk));
    Unhunk* proxy = reinterpret_cast<Unhunk*>(uproxy);
    me->topFrame.bumper = uproxy + sizeof(Unhunk);
    proxy->next = me->topFrame.unhunks;
    me->topFrame.unhunks = proxy;
    mallocSize = size;
    proxy->obj = malloc(mallocSize);
    INFO(NCCL_ALLOC, "%s:%d memory stack non-hunk malloc(%llu)", __FILE__, __LINE__, (unsigned long long)mallocSize);
    if (proxy->obj == nullptr) goto malloc_exhausted;
    return proxy->obj;
  }

malloc_exhausted:
  WARN("%s:%d Unrecoverable error detected: malloc(size=%llu) returned null.", __FILE__, __LINE__, (unsigned long long)mallocSize);
  abort();
}

void ncclMemoryStackDestruct(struct ncclMemoryStack* me) {
  // Free unhunks first because both the frames and unhunk proxies lie within the hunks.
  struct ncclMemoryStack::Frame* f = &me->topFrame;
  while (f != nullptr) {
    struct ncclMemoryStack::Unhunk* u = f->unhunks;
    while (u != nullptr) {
      free(u->obj);
      u = u->next;
    }
    f = f->below;
  }
  // Free hunks
  struct ncclMemoryStack::Hunk* h = me->stub.above;
  while (h != nullptr) {
    struct ncclMemoryStack::Hunk *h1 = h->above;
    free(h);
    h = h1;
  }
}

int ncclDebugLevel = -1;

void ncclDebugInit() {
  if (ncclDebugLevel != -1) return;
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    ncclDebugLevel = NCCL_LOG_TRACE;
  }
}

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (ncclDebugLevel == -1) ncclDebugInit();
  if (level == NCCL_LOG_TRACE && ncclDebugLevel != NCCL_LOG_TRACE) return;
  if (ncclDebugLevel < level || ((flags & (NCCL_INIT|NCCL_GRAPH|NCCL_TUNING)) == 0)) return;

  char buffer[2048];
  size_t len = 0;
  if (node_model) len = snprintf(buffer, sizeof(buffer),
    "[%d:%d] ", node_model->nodeId, node_model->currRank);
  va_list args;
  va_start(args, fmt);
  vsprintf(buffer+len, fmt, args);
  va_end(args);
  printf("%s\n", buffer);
#if 0
  if (level == NCCL_LOG_WARN) {
    fprintf(stderr,"[%d:%d] %s:%d TOPO EXPL ABORT\n",
            node_model->nodeId, node_model->currRank, filefunc, line);
    abort();
  }
#endif
}

ncclResult_t ncclTopoGetSystem(const char* xmlTopoFile, struct ncclTopoSystem** system) {
  struct ncclXml* xml;
  NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
  NCCLCHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 0));
  NCCLCHECK(ncclTopoGetSystemFromXml(xml, system, 0));
  free(xml);
  return ncclSuccess;
}

NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);

void initCollNet() {
  if (ncclParamCollNetEnable() == 1 && ncclCollNet == 0)
    ncclCollNet = (ncclCollNet_t*)0x12345678;
}

ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  if (channel->id != -1) return ncclSuccess;

  int nRanks = comm->nRanks;
  int nPeers = nRanks + 1 /* Collnet */ + comm->localRanks /* NVLS */;
  channel->id = channelId;
  channel->workFifoProduced = 0;

  struct ncclSharedResources* sharedRes = comm->sharedRes;

  //NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));

  if (channel->peers == NULL) {
    // The extra on nRanks+1 is for collnet root (i.e. network)
    // Allocate everything related to sharedRes with ncclCalloc as this can be
    // shared between communicators hence should not be tied to comm.
    if (sharedRes->peers[channelId] == NULL) {
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }
    channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
    for (int r = 0; r < nRanks; r++) {
      channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
    }
  }
#if 0
  if (channel->devPeers == NULL) {
    if (sharedRes->devPeers[channelId] == NULL) {
      NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, sharedRes->deviceStream.cudaStream));
    }
    /* channel->devPeers is not shared, so just free it when calling commFree() */
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, sharedRes->deviceStream.cudaStream));
    ncclCommPushCudaFree(comm, channel->devPeers);
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    for (int r = 0; r < nRanks; r++) {
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  }
#endif
  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  //NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, sharedRes->deviceStream.cudaStream));
  //ncclCommPushCudaFree(comm, channel->devRingUserRanks);

  /* guarantee addr has been copied into channel->devPeers */
  //NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  //NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  info->cudaDev = node_model->rankToCudaDev(comm->rank);
  info->hostHash = node_model->hostHash;
  info->pidHash = node_model->pidHash;

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  //struct stat statbuf;
  //SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = 0x19;

  info->busId = node_model->getGpuBusId(comm->rank);

  // detect if fine grained memory is available on this GPU
  info->hasFineGrain = true;
  info->gdrSupport = 1;

  info->comm = comm;
  info->cudaCompCap = 1;
  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero=0, ixRank=0;
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];
  }
  return ncclSuccess;
}

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType, bool* needsProxy) {
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                                  comm->channels[channelId].peers[peer]->recv + connIndex;
  // handle intra-node network connections
  int n1 = -1, n2 = -1;
  if (connIndex == NCCL_CONN_IDX_P2P_NET) {
    NCCLCHECK(ncclTopoGetIntraNetDev(comm->topo, comm->rank, graph, channelId, (type == 1) ? 1 : 0, nullptr, &n1));
    NCCLCHECK(ncclTopoGetIntraNetDev(comm->topo, peer, graph, channelId, (type == 1) ? 0 : 1, nullptr, &n2));
  }
  bool xgmi;
  NCCLCHECK(ncclTopoGetLinkType(comm->topo, myInfo->cudaDev, peerInfo->cudaDev, &xgmi));

  for (int t=0; t<NTRANSPORTS; t++) {
    if (graph == NULL && connIndex == NCCL_CONN_IDX_P2P_NET && (t == TRANSPORT_SHM || (!xgmi && t == TRANSPORT_P2P))) continue;
    if (graph && n1 >= 0 && n2 >= 0 && t != TRANSPORT_NET) continue;
    struct ncclTransport *transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      if (transportType) *transportType = t;
      if (needsProxy) *needsProxy = (transportComm->proxyProgress != NULL);
      return ncclSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSystemError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct ncclChannel* channel = &comm->channels[channelId];
  struct channelMasks mask;
  for (int i = 0; i < MAXCHANNELS/64; i++)
    mask.masks[i] = 0;
  mask.masks[channel->id/64] = (1UL << (channel->id%64));

  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected) continue;
    comm->connectRecv[peer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)].masks[channel->id/64] |= mask.masks[channel->id/64];
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected) continue;
    comm->connectSend[peer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)].masks[channel->id/64] |= mask.masks[channel->id/64];
  }
  return ncclSuccess;
}

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

NCCL_PARAM(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
NCCL_PARAM(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);
#include <sys/time.h>

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex, int* highestTransportType/*=NULL*/, bool* needsProxy/*=NULL*/) {
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  ncclResult_t ret = ncclSuccess;
  int highestType = TRANSPORT_P2P;  // track highest transport type
  bool needsProxyResult = false;
  struct ncclConnect** data; // Store intermediate send/recvData structs for connect
  struct ncclConnect** recvData; // Points to entries inside data for given recv connection within a channel
  struct ncclConnect** sendData; // Points to entries inside data for given send connection within a channel
  int done = 0;

  int maxPeers = ncclParamConnectRoundMaxPeers();
  NCCLCHECK(ncclCalloc(&data, maxPeers));
  NCCLCHECK(ncclCalloc(&recvData, maxPeers));
  NCCLCHECK(ncclCalloc(&sendData, maxPeers));
  //NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream), ret, fail);
  // First time initialization
  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    struct channelMasks recvMask = comm->connectRecv[recvPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)];
    struct channelMasks sendMask = comm->connectSend[sendPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)];

    // Data[i] contains all ncclConnect information for all send and receive connections with a given send and recv peer
    // This data is packed in the array based on the number of sendChannels and recvChannels connected with these peers
    // The first N entries contain recvData, connection information for recv connections
    // The next M entries contain sendData, connection information for send connections
    // It's not guaranteed that each entry of data has the same number of total or send/recv specific connections
    int p = i-(done+1);
    uint64_t recvMaskBits = 0, sendMaskBits = 0;
    for (int i = 0; i < MAXCHANNELS/64; i++) {
      recvMaskBits |= recvMask.masks[i];
      sendMaskBits |= sendMask.masks[i];
    }
    if (recvMaskBits || sendMaskBits) NCCLCHECK(ncclCalloc(data+p, 2*MAXCHANNELS));
    recvData[p] = data[p];
    int sendChannels = 0, recvChannels = 0;
    int type;
    bool proxy;
    TIME_START(0);
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask.masks[c/64] & (1UL<<(c%64))) {
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[p]+recvChannels++, c, recvPeer, connIndex, &type, &proxy), ret, fail);
        if (type > highestType) highestType = type;
      }
    }
    TIME_STOP(0);
    TIME_START(1);
    sendData[p] = recvData[p]+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask.masks[c/64] & (1UL<<(c%64))) {
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, sendData[p]+sendChannels++, c, sendPeer, connIndex, &type, &proxy), ret, fail);
        if (type > highestType) highestType = type;
        needsProxyResult |= proxy;
      }
    }
    TIME_STOP(1);

    TIME_START(2);
    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
        //NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[i], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        //NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[i], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        sendData[p] = data[p];
        recvData[p] = data[p]+sendChannels;
      }
    } else {
      //if (recvChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[i], sizeof(struct ncclConnect)*recvChannels), ret, fail);
      //if (sendChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[i], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      //if (sendChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[i], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      //if (recvChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[i], sizeof(struct ncclConnect)*recvChannels), ret, fail);
    }
    TIME_STOP(2);

    if (i-done == maxPeers || i == comm->nRanks-1) {
      // Loop until all channels with all ranks have been connected
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        for (int j=done+1; j<=i; j++) {
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          struct channelMasks recvMask = comm->connectRecv[recvPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)];
          struct channelMasks sendMask = comm->connectSend[sendPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)];

          int p = j-(done+1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;
          for (int c=0; c<MAXCHANNELS; c++) {
            TIME_START(3);
            if (sendMask.masks[c/64] & (1UL<<(c%64))) {
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                //NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset++, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[sendPeer]->send[connIndex] is a device memory access. */
                  //CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
            }
            TIME_STOP(3);

            // Start with recv channels
            TIME_START(4);
            if (recvMask.masks[c/64] & (1UL<<(c%64))) {
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                //NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset++, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[recvPeer]->recv[connIndex] is a device memory access. */
                  //CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
            }
            TIME_STOP(4);
          }
          if (sendMaskBits || recvMaskBits) {
            free(data[p]);
            data[p] = NULL;
          }
        }
      }
      done = i;
    }
  }


  /* We need to sync ranks here since some ranks might run too fast after connection setup
   * and start to destroy the connection after returning from this function; however, the
   * others might still be trying to connect and import the buffer. No sync can lead to invalid
   * shmem/cuda buffer. In addition, we also clear all connect masks and free each connectInfo array */
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    int flag = 0;

    /*if (recvPeer != sendPeer) {
      if (comm->connectSend[sendPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, &flag, sizeof(int)), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, &flag, sizeof(int)), ret, fail);

      if (comm->connectSend[sendPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, &flag, sizeof(int)), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL)
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, &flag, sizeof(int)), ret, fail);
    } else {
      if (comm->connectSend[sendPeer] != 0UL || comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, &flag, sizeof(int)), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, &flag, sizeof(int)), ret, fail);
      }
    }*/
    for (int j = 0; j<MAXCHANNELS/64; j++)
      comm->connectRecv[recvPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)].masks[j] = comm->connectSend[sendPeer+comm->nRanks*(connIndex == NCCL_CONN_IDX_P2P_NET ? NCCL_CONN_IDX_P2P_NET : 0)].masks[j] = 0UL;
  }

  free(data);
  free(sendData);
  free(recvData);

  if (highestTransportType != NULL) *highestTransportType = highestType;
  if (needsProxy != NULL) *needsProxy = needsProxyResult;
  TIME_PRINT("P2P Setup/Connect");
exit:
  //NCCLCHECK(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream));
  //NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream));
  return ret;
fail:
  goto exit;
}

extern struct ncclTransport collNetTransport;

// All ranks must participate in collNetSetup call
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
int ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type) {
  int fail = 1;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;
  int rankInCollNet = -1;
  int isMaster = (rank == masterRank) ? 1 : 0;
  struct {
    int collNetRank;
    ncclConnect connect;
  } sendrecvExchange;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;

  // send master receives connect info from peer recv master
  if (isMaster && type == collNetSend) {
    //NCCLCHECK(bootstrapRecv(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)));
    rankInCollNet = sendrecvExchange.collNetRank;
    TRACE(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }

  // select
  struct ncclChannelPeer* root = channel->peers[nranks];
  // connector index: 0 for recv, 1 for send
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv+type : root->send+type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect;
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId, type));
  }
  // prepare connect handles
  ncclResult_t res;
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == collNetRecv) {  // recv side: AllGather
    // all ranks must participate
    NCCLCHECK(ncclCalloc(&allConnects, nranks));
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    //NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), res, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        if (r == rank) rankInCollNet = c;
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    //if (isMaster) memcpy(masterConnects+rankInCollNet, &(sendrecvExchange.connect), sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, rankInCollNet, conn), res, cleanup);
    struct ncclDevChannelPeer* devRoot;
    //CUDACHECKGOTO(cudaMemcpy(&devRoot, channel->devPeers + nranks, sizeof(struct ncclDevChannelPeer*), cudaMemcpyDeviceToHost), res, cleanup);
    struct ncclConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv + type : devRoot->send + type;
    //CUDACHECKGOTO(cudaMemcpy(devConnInfo, &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice), res, cleanup);
  }
  // recv side sends connect info to send side
  if (isMaster && type == collNetRecv) {
    sendrecvExchange.collNetRank = rankInCollNet;
    //memcpy(&sendrecvExchange.connect, masterConnects+rankInCollNet, sizeof(struct ncclConnect));
    //NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)), res, cleanup);
    TRACE(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }
  fail = 0;
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return fail;
}

ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail) {
  // AllGather collNet setup results
  int allGatherFailures[NCCL_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;
  //NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, allGatherFailures, sizeof(int)));
  for (int i=0; i<comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm) {
  // Free collNet resources
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    struct ncclChannelPeer* peer = channel->peers[comm->nRanks];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* send = peer->send + b;
          if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
          send->transportResources = NULL; // avoid double free
        }
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* recv = peer->recv + b;
          if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
          recv->transportResources = NULL; // avoid double free
        }
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGatherInfo *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph, struct ncclTopoGraph& nvlsGraph, struct ncclComm* parent) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  cpu_set_t affinitySave;
  //struct ncclTopoGraph ringGraph;
  //struct ncclTopoGraph treeGraph;
  //struct ncclTopoGraph collNetGraph;
  //struct ncclTopoGraph nvlsGraph;
  struct ncclTopoGraph* graphs[] = { &treeGraph, &ringGraph, &collNetGraph, &collNetGraph, &nvlsGraph, &nvlsGraph, &nvlsGraph};

  int nChannelsOrig;
  struct ncclTopoRanks** allTopoRanks = NULL;
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;
  int *rings = NULL;
  int* nvbPeers = NULL;
  struct ncclProxyConnector proxyConn;
  int* pxnPeers = NULL;
  int *topParentLocalRanks = NULL;
  int tpProxyRank;

  // AllGather1 - begin
  //NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), ret, fail); // Extra rank to represent CollNet root
  //NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, commHash), ret, fail);
  //NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);

  for (int i = 0; i < nranks; i++) {
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }
  // AllGather1 - end

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
    for (int i = 0; i < nranks; i++) comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[rank].cudaCompCap);
    for (int i = 0; i < nranks; i++) comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[rank].cudaCompCap);
    for (int i = 0; i < nranks; i++) {
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }
    }
    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      ret = ncclInternalError;
      goto fail;
    }
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank==0 ? comm==comm0 : true);
    comm->intraComm0 = comm0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

  // Topo detection / System graph creation
  //NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);
  // save nRanks to ncclTopoSystem as indicator of multi-node
  comm->topo->nRanks = comm->nRanks;
  // init netGdrLevel
  comm->topo->netGdrLevel = -2;
  // init Pivot A2A related fields
  comm->topo->pivotA2AEnabled = false;
  comm->topo->pivotA2ANumBiRings = 0;
  // LL128
  comm->topo->ll128Enabled = false;
  // Topology hint for MSCCL internal scheduler about whether to enable MSCCL
  comm->topo->mscclEnabled = false;
  // Topology hint if tree has been defined by model or User
  comm->topo->treeDefined = false;
  // Compute paths between GPUs and NICs
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);
  // Recompute paths after trimming
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Init search
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);
  // Print final topology
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  //NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);
  //if (CPU_COUNT(&comm->cpuAffinity)) {
  //  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
  //  sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  //}

  // Determine local CollNet support
  if (collNetSupport(comm)) {
    const char *collNetEnable = ncclGetEnv("NCCL_COLLNET_ENABLE");
    if (collNetEnable != NULL) {
      INFO(NCCL_ALL, "NCCL_COLLNET_ENABLE set by environment to %s.", collNetEnable);
      if (strcmp(collNetEnable, "1") == 0) {
        comm->config.collnetEnable = 1;
      }
    }
  }

  // Determine local Nvls support
  //NCCLCHECK(ncclNvlsInit(comm));

  // [RCCL] Compute hostIdx (based on hostHash)
  {
    comm->topo->nHosts = 0;
    for (int r = 0; r < nranks; r++) {
      int isNewHost = 1;
      // Check if this is the first time this hostname has been used
      for (int i = 0; i < r && isNewHost; i++) {
        if (comm->peerInfo[i].hostHash == comm->peerInfo[r].hostHash)
          isNewHost = 0;
      }
      if (isNewHost)
      {
        // Check if this is the same hostname associated with this rank
        if (comm->peerInfo[r].hostHash == comm->peerInfo[rank].hostHash)
        {
          comm->topo->hostIdx = comm->topo->nHosts;
          printf("Rank %d is on host %d\n", rank, comm->topo->hostIdx);
        }
        comm->topo->nHosts++;
      }
    }
  }

  // Get rings and trees
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &ringGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &ringGraph), ret, fail);

  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.collNet = 0;
  treeGraph.minChannels = comm->topo->nodes[NET].count != 0 ? 1 : ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &treeGraph), ret, fail);

  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  if (comm->config.collnetEnable) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &collNetGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &collNetGraph), ret, fail);
  }

  nvlsGraph.id = 3;
  nvlsGraph.pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph.minChannels = 1;
  nvlsGraph.maxChannels = MAXCHANNELS;
  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &nvlsGraph), ret, fail);
  }

  bool allXgmi, hasPeerAccess;
  allXgmi = true;
  hasPeerAccess = true;
  // Check that all the GPUs have peer access to one another and are XGMI connected
  for (int i = 0; i < nranks && hasPeerAccess; i++) {
    int cudaDev1 = comm->peerInfo[i].cudaDev;
    for (int j = 0; j < nranks; j++) {
      if (i == j) continue;
      int cudaDev2 = comm->peerInfo[j].cudaDev;
      int p2p;
      if (hipDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != hipSuccess || !p2p)
      {
        hasPeerAccess = false;
        break;
      }

      bool isXGMI;
      // Limit to single intermediate GPU for enabling clique
      NCCLCHECK(ncclTopoGetLinkType(comm->topo, i, j, &isXGMI, 1));
      allXgmi &= isXGMI;
    }
  }
  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* dumpGraphs[4] = { &ringGraph, &treeGraph, &collNetGraph, &nvlsGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 4, dumpGraphs), ret, fail);
  }

  if ((comm->topo->type & RCCL_TOPO_4P2H_ROME) && (comm->topo->type & RCCL_TOPO_GDR_ALL)) {
    if (rcclParamP2pNetDisable() == 0) {
      if (!(comm->topo->type & RCCL_TOPO_FORCE_INTRA)) comm->p2pNet = 1;
      INFO(NCCL_INIT, "RCCL enabled same node P2P over network");
    }
    else
      INFO(NCCL_INIT, "RCCL force disabled same node P2P over network");
  }
  // AllGather3 - begin
  //NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);
  int idx;
  NCCLCHECK(ncclTopoIdToIndex(comm->topo, GPU, comm->busId, &idx));
  allGather3Data[rank].nc = 2;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks &&
      IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx906") && allXgmi)
    allGather3Data[rank].nc = 4;
  if (IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx908"))
    allGather3Data[rank].nc = std::max(4/ringGraph.nChannels, 2);
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks &&
       (comm->topo->type & RCCL_TOPO_CR8G))
    allGather3Data[rank].nc = 4;
  if (comm->topo->nodes[GPU].count == comm->topo->nRanks &&
      IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx90a"))
    allGather3Data[rank].nc = 4;
  if (IsArchMatch(comm->topo->nodes[GPU].nodes[idx].gpu.gcn, "gfx90a"))
    allGather3Data[rank].nc = std::max(allGather3Data[rank].nc, 4/ringGraph.nChannels);
  if (ringGraph.nChannels > MAXCHANNELS/2)
    allGather3Data[rank].nc = 1;
  allGather3Data[rank].pivotA2AEnabled = comm->topo->pivotA2AEnabled && rcclParamPivotAlltoallEnable();
  comm->topo->ll128Enabled =  comm->topo->ll128Enabled || rcclParamLL128ForceEnable();
  allGather3Data[rank].ll128Enabled = comm->topo->ll128Enabled;
  allGather3Data[rank].mscclEnabled = comm->topo->mscclEnabled;

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
  }

  comm->nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  TRACE(NCCL_INIT,"treeGraph.nChannels: %d , ringGraph.nChannels: %d", treeGraph.nChannels, ringGraph.nChannels);
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);
fail:
  return ret;
}

ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGatherInfo *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph, struct ncclTopoGraph& nvlsGraph) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  cpu_set_t affinitySave;

  struct ncclTopoGraph* graphs[] = { &treeGraph, &ringGraph, &collNetGraph, &collNetGraph, &nvlsGraph, &nvlsGraph, &nvlsGraph};

  int nChannelsOrig;
  struct ncclTopoRanks** allTopoRanks = NULL;
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;
  int *rings = NULL;
  int* nvbPeers = NULL;
  struct ncclProxyConnector proxyConn;
  int* pxnPeers = NULL;
  int *topParentLocalRanks = NULL;
  int tpProxyRank;

  int highestTransportType = TRANSPORT_P2P;
  bool needsProxy = false;
  //NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  // Determine nNodes, firstRanks, ...
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail);
  for (int r=0; r<nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    comm->rankToNode[r] = node;
  }
  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    ret = ncclInternalError;
    goto fail;
  }

  nChannelsOrig = comm->nChannels;
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);
  int nc;
  nc = allGather3Data[0].nc;
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    nc = std::min(allGather3Data[i].nc, nc);
    // Make sure we align all ranks so that the tuning is consistent across ranks
    comm->topo->pivotA2AEnabled = comm->topo->pivotA2AEnabled && allGather3Data[i].pivotA2AEnabled;
    comm->topo->ll128Enabled = comm->topo->ll128Enabled && allGather3Data[i].ll128Enabled;
    comm->topo->mscclEnabled = comm->topo->mscclEnabled && allGather3Data[i].mscclEnabled;
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
    }
    if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->config.collnetEnable = 0;
    if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = 0;
  }

  comm->nChannels = treeGraph.nChannels = ringGraph.nChannels =
    (comm->topo->nodes[GPU].count != comm->topo->nRanks && comm->topo->nodes[NET].count)
    ? std::min(treeGraph.nChannels, ringGraph.nChannels) : ringGraph.nChannels;
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->config.collnetEnable == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->config.collnetEnable = 0;
    }
    for (int n=0; n<comm->nNodes; n++) {
      if (comm->nodeRanks[n].localRanks > NCCL_MAX_DIRECT_ARITY+1) {
        WARN("CollNet currently only supports up to %d GPUs per node, disabling CollNet", NCCL_MAX_DIRECT_ARITY+1);
        comm->config.collnetEnable = 0;
        break;
      }
    }
  }

  NCCLCHECKGOTO(ncclCalloc(&rings, nranks*MAXCHANNELS), ret, fail);
  TRACE(NCCL_INIT, "rank %d nranks %d nchannels %d", rank, nranks, comm->nChannels);

  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, NULL, nc), ret, fail);
  if (comm->topo->treeDefined) NCCLCHECK(ncclTreeBasePostset(comm, &treeGraph));

  // AllGather3 - end

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[4096];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 2047-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %d : %d -> %d -> %d comm %p nRanks %02d busId %lx", c, comm->channels[c].ring.prev,
         comm->rank, comm->channels[c].ring.next, comm, comm->nRanks, comm->busId);
  }
  line[4095] = '\0';
  INFO(NCCL_INIT, "Trees%s comm %p nRanks %02d busId %lx", line, comm, comm->nRanks, comm->busId);

  //NCCLCHECKGOTO(computeBuffSizes(comm), ret, fail);

  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  /* until now, all info of comm should be known. We can initialize shared resources and
   * map localRanks to top parent local ranks. NOTE: this shareRes init must be put before
   * all proxy operations. */
  if (comm->sharedRes->owner == comm) {
    comm->sharedRes->tpNLocalRanks = comm->localRanks;
    comm->sharedRes->magic = comm->magic;
    comm->sharedRes->tpNChannels = comm->nChannels;
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]];
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank];
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  // Launch proxy service thread, after this, the proxy calls can be used.
  //NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, 0, &highestTransportType, &needsProxy), ret, fail);
  if (ringGraph.nIntraChannels && rcclParamP2pNetDisable() == 0) {
    comm->useIntraNet = 1;
    // Connect NET for intranode use
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      if (comm->nRanks == 1) continue;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, NCCL_CONN_IDX_P2P_NET), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, NCCL_CONN_IDX_P2P_NET, &highestTransportType, &needsProxy), ret, fail);
  }
  INFO(NCCL_INIT, "Connected all rings");

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &treeGraph, 0, &highestTransportType, &needsProxy), ret, fail);
  INFO(NCCL_INIT, "Connected all trees");

#if 0
  // Setup NVLS
  NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
  // And NVLS trees if needed
  if (comm->nvlsSupport && comm->localRanks > 1) {
    for (int c=0; c<comm->nvlsChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 1, &channel->nvls.treeUp, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->nvls.treeUp, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &nvlsGraph, 0), ret, fail);
    INFO(NCCL_INIT, "Connected NVLS tree");
  }
#endif
#if CUDART_VERSION >= 12010
  // Check if we can setup CollNet
  if (comm->config.collnetEnable > 0) collNetTrySetup(comm, parent, &collNetGraph);
#endif

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

#if 0
  do { // Setup p2p structures in comm->tasks
    struct ncclTasks* tasks = &comm->tasks;
    int node = comm->node;
    int nNodes = comm->nNodes;
    struct ncclNodeRanks *nodeRanks = comm->nodeRanks;
    int localRank = comm->localRank;
    // We want to fuse along node boundaries. Make sure nsteps is a multiple or divides 8.
    int steps = ALIGN_POWER(comm->maxLocalRanks, NCCL_MAX_WORK_ELEMENTS_P2P/2);
    tasks->p2pOrderSteps = comm->nNodes * steps;
    tasks->peers = ncclMemoryStackAlloc<ncclTasks::Peer>(&comm->memPermanent, tasks->p2pOrderSteps);
    tasks->p2pSendOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, tasks->p2pOrderSteps);
    tasks->p2pRecvOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, tasks->p2pOrderSteps);
    int i=0;
    // schedule delta 0, +1, -1, +2, -2, ...
    // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
    for (int d=0; d <= nNodes/4; d++) {
      int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
      int index = 0;
      int delta = deltas[index];
    sched_delta:
      int recvNode = (node+nNodes-delta)%nNodes;
      int sendNode = (node+delta)%nNodes;
      for (int step=0; step < steps; step++) {
        int recvIndex = (localRank-step+steps)%steps;
	int recvRank = recvIndex < nodeRanks[recvNode].localRanks ? nodeRanks[recvNode].localRankToRank[recvIndex] : -1;
        tasks->p2pRecvOrder[i] = recvRank;
        int sendIndex = (localRank+step)%steps;
        int sendRank = sendIndex < nodeRanks[sendNode].localRanks ? nodeRanks[sendNode].localRankToRank[sendIndex] : -1;
        tasks->p2pSendOrder[i] = sendRank;
        i++;
      }
      index++;
      if (index == 1 && deltas[1] == deltas[0]) index++;
      if (index == 2 && deltas[2] == deltas[0]) index++;
      if (index == 3 && deltas[3] == deltas[2]) index++;
      if (index == 3 && deltas[3] == deltas[1]) index++;
      if (index < 4) {
        delta = deltas[index];
        goto sched_delta;
      }
    }
    assert(i == tasks->p2pOrderSteps);
  } while (0);

  if (ncclParamNvbPreconnect()) {
    // Connect p2p when using NVB path
    int nvbNpeers;
    NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
    for (int r=0; r<nvbNpeers; r++) {
      int peer = nvbPeers[r];
      int channelId;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECKGOTO(ncclChannelCompute(comm, peer, c, ncclFuncSend, &channelId), ret, fail);
        if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
          comm->connectSend[peer] |= (1UL<<channelId);
        }
      }
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECKGOTO(ncclChannelCompute(comm, peer, c, ncclFuncRecv, &channelId), ret, fail);
        if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1UL<<channelId);
        }
      }
    }

    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
  }
#endif
  // Connect to local net proxy
  tpProxyRank = comm->topParentRanks[comm->rank];
  //NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, tpProxyRank, &proxyConn), ret, fail);
  //NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

  // Then to remote ones when using PXN
  if (ncclPxnDisable(comm) == 0) {
    int nranks;
    NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
    for (int r=0; r<nranks; r++) {
      tpProxyRank = comm->topParentRanks[pxnPeers[r]];
      //NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, tpProxyRank, &proxyConn), ret, fail);
      //NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
    }
  }

#if 0
  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    char* str = getenv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all cuda mem allocation is complete. That could cause a deadlock.
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  /* Local intra-node barrier */
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);
#endif
  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

exit:
  //if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  /* If split resource is shared, we are not able to unlink the proxy ops pool here since the child comm can
   * attach the proxy ops pool of parent at any time; otherwise, unlink it here to make sure the pool will be
   * properly cleaned up. */
  //if (comm->sharedRes->owner == comm && !comm->config.splitShare && ret == ncclSuccess) ncclProxyShmUnlink(comm);
  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  //free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);
  return ret;
fail:
  goto exit;
}

ncclResult_t rocm_smi_init() {
  return ncclSuccess;
}

ncclResult_t rocm_smi_getNumDevice(uint32_t* num_devs) {
  return ncclSuccess;
}

ncclResult_t rocm_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* busId, size_t len) {
  return ncclSuccess;
}

ncclResult_t rocm_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex) {
  return ncclSuccess;
}

ncclResult_t rocm_smi_getLinkInfo(int srcIndex, int dstIndex, RSMI_IO_LINK_TYPE* rsmi_type, int *hops, int *count) {
  return ncclSuccess;
}

int ncclNetVersion(struct ncclComm* comm) {
  return 4;
}

bool mscclEnabled() {
  return false;
}

bool mscclForceEnabled() {
  return false;
}

ncclResult_t mscclSchedulerInit(ncclComm_t comm, int* numChannelsRequired) {
  return ncclSuccess;
}

uint64_t getHostHash(void) {
  return 0xdeadbeef;
}

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int *ranks, int rank, int nranks, void* allData, int size) {
  return ncclSuccess;
}
