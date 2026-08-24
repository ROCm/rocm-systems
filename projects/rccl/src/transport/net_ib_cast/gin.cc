/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common_cast.h"
#include "connect_cast.h"

#include "gin/gin_host.h"
#include "gin_cast.h"
#include "alloc.h"

const int IBCAST_GIN_IB_ALLGATHER_TAG = 0xa0;
const int IBCAST_GIN_IB_ALLTOALL_TAG = 0xa1;

// Check GDR support for GIN. This is run at init, so we don't know yet whether the GPU will support DMA-BUF.
static ncclResult_t IbCastGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
  bool peerMemSupport = gdaki ? IbCastPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
                                IbCastGdrSupport() == ncclSuccess;
#else
  bool peerMemSupport = IbCastGdrSupport() == ncclSuccess;
#endif
  if (peerMemSupport) return ncclSuccess;

  if (IbCastDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  *gdrSupport = false;
  INFO(NCCL_NET, "Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclSuccess;
}

// Check the current GPU supports GDR for GIN. This is run during connect().
static ncclResult_t IbCastGinIbGdrGpuSupport(bool gdaki) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (IbCastDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  if (IbCastGdrSupport() == ncclSuccess) return ncclSuccess;
  WARN("Unable to use GIN: Peermem is not supported, and DMA-BUF is not available.");
#else
  bool peerMemSupport = gdaki ? IbCastPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
                                IbCastGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int dmaBufSupportOnDevice = 1;
  CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
  if (dmaBufSupportOnDevice == 1) return ncclSuccess;
  WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
#endif
  return ncclInvalidUsage;
}

NCCL_PARAM(CastGinType, "GIN_TYPE", -1);
NCCL_PARAM(CastGinIbTc, "GIN_IB_TC", -1);
extern int64_t ncclParamIbCastTc();

#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
static std::mutex IbCastGinGdakiLockMutex;
static int IbCastGinGdakiNDevs = -1;
int IbCastGinGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t IbCastGinIbGdakiInit() {
  std::lock_guard<std::mutex> lock(IbCastGinGdakiLockMutex);
  if (IbCastGinGdakiNDevs == -1) {
    int ndevs = 0;
    for (int i = 0; i < IbCastNDevs; i++) {
      if (IbCastDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        IbCastGinGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    IbCastGinGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}
#endif // RCCL_NET_IB_CAST_ENABLE_GDAKI

#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
extern ncclGin_t IbCastGinIbGdaki;
#endif

// Initialize the IB-CAST backend devices and net comm config. Under the v14
// GIN/RMA split this shared helper is used both by the device-initiated GDAKI GIN
// backend and by the IB-CAST RMA proxy backend (IbCastRmaIbProxy). Proxy GIN over
// IB-CAST is provided generically by ncclGinProxy layered on top of the RMA
// backend, so there is no bespoke IB-CAST GIN proxy vtable anymore.
ncclResult_t IbCastGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction, int type) {
  NCCLCHECK(IbCastInitDevices(logFunction, nullptr));
  if (IbCastNDevs <= 0) return ncclInternalError; // Caught in plugin init code, not propagated to user.

#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
  if (type == NCCL_GIN_TYPE_GDAKI) {
    NCCLCHECK(IbCastGinIbGdakiInit());
    if (IbCastGinGdakiNDevs == 0) return ncclInternalError;
  }
#endif // RCCL_NET_IB_CAST_ENABLE_GDAKI

  bool gdrSupport;
  NCCLCHECK(IbCastGinIbGdrSupport(&gdrSupport, type == NCCL_GIN_TYPE_GDAKI));
  if (!gdrSupport) return ncclInternalError;

  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbFinalize(void* ctx) {
  if (ctx) free(ctx);
  return IbCastFinalizeDevices();
}

// v14 GIN plugins expose GIN capability flags via getGinProperties. IB-CAST's
// IB proxy/GDAKI backends support both strong and VA signals.
ncclResult_t IbCastGinIbGetGinProperties(ncclGinProperties_v14_t* ginProps) {
  ginProps->supportsStrongSignals = true;
  ginProps->supportsVASignals = true;
  return ncclSuccess;
}

static ncclResult_t IbCastGinIbAllGather(struct CastIbGinCollComm* cComm, void* srcBuf, void* recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer;
  int rpeer;
  void* rbuf;
  int tag;
  int done;

  NCCLCHECKGOTO(netIbCast.regMr(cComm->recvComm, recvBuf, cComm->nranks * len, NCCL_PTR_HOST, &rMhandle), status, out);
  NCCLCHECKGOTO(netIbCast.regMr(cComm->sendComm, recvBuf, cComm->nranks * len, NCCL_PTR_HOST, &sMhandle), status, out);

  speer = cComm->rank;
  memcpy((void*)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void*)((uintptr_t)recvBuf + rpeer * len);
      tag = IBCAST_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(netIbCast.isend(cComm->sendComm, (void*)((uintptr_t)recvBuf + speer * len), len, tag, sMhandle,
                                      NULL, &srequest),
                      status, out);
      if (rrequest == NULL)
        NCCLCHECKGOTO(netIbCast.irecv(cComm->recvComm, 1, &rbuf, &len, &tag, &rMhandle, NULL, &rrequest), status, out);
    }
    while (srequest || rrequest) {
      if (rrequest) NCCLCHECKGOTO(netIbCast.test(rrequest, &done, NULL), status, out);
      if (done) rrequest = NULL;
      if (srequest) NCCLCHECKGOTO(netIbCast.test(srequest, &done, NULL), status, out);
      if (done) srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle) netIbCast.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle) netIbCast.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t IbCastGinIbAllToAll(struct CastIbGinCollComm* cComm, void* src_buf, void* recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;

  void* tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void**)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void*)((uintptr_t)recv_buf + i * len),
           (void*)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len), len);
  }

out:
  if (tmp_buf) free(tmp_buf);

  return status;
}

ncclResult_t IbCastGinIbP2PBarrier(struct CastIbGinCollComm* cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int* dummy;
  NCCLCHECK(ncclIbMalloc((void**)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(IbCastGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t IbCastGinIbConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct CastIbGinCollComm* cCommArray = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void**)&cCommArray, sizeof(*cCommArray)));

  struct CastIbGinCollComm* cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(IbCastConnectImpl(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL,
                                  ncclParamCastGinIbTc() != -1 ? ncclParamCastGinIbTc() : ncclParamIbCastTc()));
    }
    if (cComm->recvComm == NULL) NCCLCHECK(netIbCast.accept(lComm, &cComm->recvComm, NULL));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t (*)(int dev, void* props))IbCastGetProperties;
  cComm->allGather = IbCastGinIbAllGather;
  cComm->allToAll = IbCastGinIbAllToAll;
  cComm->getGidIndex = IbCastGetGidIndex;
  cComm->dev = lComm->dev;

  cComm->ib.context = IbCastDevs[cComm->dev].context;
  cComm->ib.pd = IbCastDevs[cComm->dev].pd;

  *collComm = cCommArray;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbCloseColl(void* collComm) {
  struct CastIbGinCollComm* cCommArray = (struct CastIbGinCollComm*)collComm;
  if (!cCommArray) return ncclSuccess;

  struct CastIbGinCollComm* cComm = cCommArray;
  if (cComm->recvComm) {
    NCCLCHECK(netIbCast.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(netIbCast.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cCommArray);
  return ncclSuccess;
}

#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
#include "gdaki/gin_host_gdaki.h"

ncclResult_t IbCastGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return IbCastGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI);
}

ncclResult_t IbCastGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(IbCastGinGdakiLockMutex);
  *ndev = IbCastGinGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(IbCastGinGdakiLockMutex);
  if (dev >= IbCastGinGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created", dev,
         IbCastGinGdakiNDevs);
    return ncclInvalidUsage;
  }
  NCCLCHECK(IbCastGetPhysProperties(IbCastGinGdakiDevIndexes[dev], props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbGdakiListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(IbCastGinGdakiLockMutex);
  return netIbCast.listen(ctx, IbCastGinGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

ncclResult_t IbCastGinIbGdakiConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                     void** collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(IbCastGinIbGdrGpuSupport(/*gdaki*/ true));

  NCCLCHECK(IbCastGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)*collComm;
  cComm->getProperties = (ncclResult_t (*)(int dev, void* props))IbCastGinIbGdakiGetProperties;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbGdakiCreateContext(void* collComm, ncclGinConfig_v14_t* config, void** ginCtx,
                                           ncclNetDeviceHandle_t** devHandle) {
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)collComm;

  if (ncclParamCastGinIbTc() != -1) config->trafficClass = ncclParamCastGinIbTc();
  else if (ncclParamIbCastTc() != -1) config->trafficClass = ncclParamIbCastTc();

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config->nSignals, config->nCounters, config->nContexts, config->queueDepth,
                                      config->trafficClass, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t IbCastGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags,
                                      void** mhandle, void** ginHandle) {
  return ncclGinGdakiRegMrSym((struct CastIbGinCollComm*)collComm, data, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t IbCastGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct CastIbGinCollComm*)collComm, mhandle);
}

ncclResult_t IbCastGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t IbCastGinIbGdakiProgress(void* collComm) {
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t IbCastGinIbGdakiQueryLastError(void* ginCtx, bool* hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t IbCastGinIbGdaki = {"GIN_IB_GDAKI",
                              IbCastGinIbGdakiInit,
                              IbCastGinIbGdakiDevices,
                              IbCastGinIbGetGinProperties,
                              IbCastGinIbGdakiGetProperties,
                              IbCastGinIbGdakiListen,
                              IbCastGinIbGdakiConnect,
                              IbCastGinIbGdakiCreateContext,
                              IbCastGinIbGdakiRegMrSym,
                              NULL, // regMrSymDmaBuf
                              IbCastGinIbGdakiDeregMrSym,
                              IbCastGinIbGdakiDestroyContext,
                              IbCastGinIbCloseColl,
                              IbCastCloseListen,
                              IbCastGinIbGdakiProgress,
                              IbCastGinIbGdakiQueryLastError,
                              IbCastGinIbFinalize};
#endif // RCCL_NET_IB_CAST_ENABLE_GDAKI

struct IbCastRmaProxyMrHandle {
  int nSegments;
  // segOff[0]==0, segOff[nSegments]==size; per-segment local MRs.
  // base_vas indexed [rank*nSegments + seg].
  // rkeys indexed [rank][seg][dev]:
  //   (rank * nSegments + seg) * NCCL_IB_MAX_DEVS_PER_NIC + remDevIdx
  size_t segOff[NCCL_RMA_MAX_SEGMENTS + 1];
  struct ncclIbMrHandle* mrHandle[NCCL_RMA_MAX_SEGMENTS];
  uintptr_t* base_vas;
  uint32_t* rkeys;
};

static inline size_t IbCastRmaRkeyIndex(int nSegments, int rank, int seg, int remDevIdx) {
  return ((size_t)rank * nSegments + (size_t)seg) * NCCL_IB_MAX_DEVS_PER_NIC + (size_t)remDevIdx;
}

static inline uint32_t IbCastRmaRemoteRkey(const struct IbCastRmaProxyMrHandle* h, int rank, int seg, int remDevIdx) {
  return h->rkeys[IbCastRmaRkeyIndex(h->nSegments, rank, seg, remDevIdx)];
}

static inline int IbCastRmaSegOf(const struct IbCastRmaProxyMrHandle* h, uint64_t off) {
  for (int s = 0; s < h->nSegments; s++)
    if (off < h->segOff[s + 1]) return s;
  return h->nSegments - 1;
}

static inline uint64_t IbCastRmaMrBytes(const struct IbCastRmaProxyMrHandle* h) {
  return h->segOff[h->nSegments];
}

static inline bool IbCastRmaRangeOk(const struct IbCastRmaProxyMrHandle* h, uint64_t off, size_t size) {
  uint64_t bytes = IbCastRmaMrBytes(h);
  return off <= bytes && (uint64_t)size <= bytes - off;
}

// Build a chained RDMA WR list, splitting at segment boundaries so every WR
// stays within a single physical segment's MR.
//
// Paired mode (flushSge == NULL): move `size` bytes between a local and a
// remote symmetric buffer, splitting on both sides; each WR carries one local
// lkey and one remote rkey.
//
// Flush mode (flushSge != NULL): the local handle is ignored; emit one tiny
// RDMA_READ per remote segment touched by [remoteOff, remoteOff+size), each
// landing a single byte in the caller's flush scratch.
static ncclResult_t IbCastRmaBuildSegmentedWrs(struct ibv_send_wr* wr, struct ibv_sge* sge, int maxWr, int* nWr,
                                               enum ibv_wr_opcode opcode, uint64_t wrId, const struct ncclIbQp* qp,
                                               struct IbCastRmaProxyMrHandle* localH, int localRank, uint64_t localOff,
                                               struct IbCastRmaProxyMrHandle* remoteH, int remoteRank, uint64_t remoteOff,
                                               size_t size, const struct ibv_sge* flushSge) {
  int n = 0;
  uint64_t lOff = localOff, rOff = remoteOff;
  size_t rem = size;
  while (rem > 0) {
    if (n >= maxWr) {
      WARN("NET/IB-CAST/RMA: transfer of %zu bytes spans more than %d segment slices", size, maxWr);
      return ncclInternalError;
    }
    int rs = IbCastRmaSegOf(remoteH, rOff);
    size_t chunk = rem;
    if (remoteH->segOff[rs + 1] - rOff < chunk) chunk = remoteH->segOff[rs + 1] - rOff;

    int ls = 0;
    if (flushSge == NULL) {
      ls = IbCastRmaSegOf(localH, lOff);
      if (localH->segOff[ls + 1] - lOff < chunk) chunk = localH->segOff[ls + 1] - lOff;
    }

    uintptr_t rAddr = remoteH->base_vas[(size_t)remoteRank * remoteH->nSegments + rs] + (rOff - remoteH->segOff[rs]);

    memset(&wr[n], 0, sizeof(wr[n]));
    memset(&sge[n], 0, sizeof(sge[n]));
    wr[n].opcode = opcode;
    wr[n].wr_id = wrId;
    wr[n].next = NULL;
    wr[n].wr.rdma.remote_addr = (uint64_t)rAddr;
    wr[n].wr.rdma.rkey = IbCastRmaRemoteRkey(remoteH, remoteRank, rs, qp->remDevIdx);
    wr[n].sg_list = &sge[n];
    wr[n].num_sge = 1;
    if (flushSge != NULL) {
      sge[n] = *flushSge;
      sge[n].length = 1;
    } else {
      uintptr_t lAddr = localH->base_vas[(size_t)localRank * localH->nSegments + ls] + (lOff - localH->segOff[ls]);
      struct ibv_mr* lmr = localH->mrHandle[ls]->mrs[qp->devIndex];
      if (lmr == NULL) {
        WARN("NET/IB-CAST/RMA: no local MR for segment %d device %d", ls, qp->devIndex);
        return ncclInternalError;
      }
      sge[n].addr = (uintptr_t)lAddr;
      sge[n].length = chunk;
      sge[n].lkey = lmr->lkey;
    }
    if (n > 0) wr[n - 1].next = &wr[n];

    lOff += chunk;
    rOff += chunk;
    rem -= chunk;
    n++;
  }
  *nWr = n;
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return IbCastGinIbInitType(ctx, commId, logFunction, ncclParamCastGinType());
}

ncclResult_t IbCastRmaIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(netIbCast.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                     void** collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(IbCastGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(IbCastGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  return ncclSuccess;
}

struct IbCastRmaIbProxyCtx {
  void** fullRecvComm;
  void** fullSendComm;
  int rank, nranks;
  int nContexts;
};

ncclResult_t IbCastRmaIbProxyCreateContext(void* collComm, ncclRmaConfig_t* config, void** ginCtx) {
  ncclResult_t ret = ncclSuccess;
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)collComm;
  // Make sure all QP we create use the provided traffic class.
  IbCastSetTrafficClass(cComm->ctx, config->trafficClass);

  if (config->rankStride <= 0 || (cComm->nranks % config->rankStride) != 0) {
    WARN("RMA_IB_PROXY create context: invalid rank stride %d, must be > 0 and nranks (%d) must be a multiple of it",
         config->rankStride, cComm->nranks);
    return ncclInternalError;
  }

  int nranks;
  struct IbCastRmaIbProxyCtx* ginProxyCtx = NULL;
  *ginCtx = NULL;
  NCCLCHECK(ncclCalloc(&ginProxyCtx, config->nContexts));
  ginProxyCtx[0].nContexts = config->nContexts;
  ginProxyCtx[0].nranks = nranks = cComm->nranks;

  void* lComm = NULL;
  char *handle = NULL, *handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE * cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE * cComm->rank;

  NCCLCHECKGOTO(netIbCast.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, end);

  // Mark communicator as RMA communicator: 1QP, Flush enabled, no CTS offload.
  // Must be set after listen() which memsets the handle, but before allGather().
  ((struct ncclIbHandle*)handle)->isRMA = true;

  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c = 0; c < config->nContexts; c++) {
    struct IbCastRmaIbProxyCtx* gc = ginProxyCtx + c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void*) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void*) * nranks), ret, end);
    gc->rank = cComm->rank;

    for (int i = 0; i < nranks; i += config->rankStride) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;
      do {
        if (gc->fullSendComm[connectPeer] == NULL)
          NCCLCHECKGOTO(IbCastConnectImpl(cComm->ctx, cComm->dev, handles + NCCL_NET_HANDLE_MAXSIZE * connectPeer,
                                          &gc->fullSendComm[connectPeer], NULL,
                                          ncclParamCastGinIbTc() != -1 ? ncclParamCastGinIbTc() : ncclParamIbCastTc()),
                        ret, end);
        if (gc->fullRecvComm[acceptPeer] == NULL)
          NCCLCHECKGOTO(netIbCast.accept(lComm, &gc->fullRecvComm[acceptPeer], NULL), ret, end);
      } while ((gc->fullSendComm[connectPeer] == NULL) || (gc->fullRecvComm[acceptPeer] == NULL));
      NCCLCHECKGOTO(IbCastGinIbP2PBarrier(cComm), ret, end);
    }
  }

end:
  free(handles);
  if (lComm) netIbCast.closeListen(lComm);
  if (ret != ncclSuccess) free(ginProxyCtx);
  else *ginCtx = ginProxyCtx;
  return ret;
}

ncclResult_t IbCastRmaIbProxyDestroyContext(void* ginCtx) {
  struct IbCastRmaIbProxyCtx* gc = (struct IbCastRmaIbProxyCtx*)ginCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;
  for (int c = 0; c < nContexts; c++) {
    if (gc[c].fullRecvComm) {
      for (int i = 0; i < nranks; i++) {
        NCCLCHECK(netIbCast.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i = 0; i < nranks; i++) {
        NCCLCHECK(netIbCast.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }
  }
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                            uint64_t mr_flags, void** mhandle) {
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)collComm;
  struct IbCastRmaProxyMrHandle* ginMrHandle = NULL;
  uintptr_t localVas[NCCL_RMA_MAX_SEGMENTS];
  uint32_t localRkeys[NCCL_RMA_MAX_SEGMENTS * NCCL_IB_MAX_DEVS_PER_NIC];
  ncclResult_t ret = ncclSuccess;
  int nSeg = 1;
  int registered = 0;

  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));
  // calloc zeroes nSegments; fail paths below only dereg `registered` complete
  // handles, so a half-built ncclIbMrHandle is never passed to deregMr.

  // Count physical segments; ROCm/HIP describes only the first per export,
  // so multi-segment ranges register one MR per segment below.
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  if (type == NCCL_PTR_CUDA && ncclCuMemEnable()) {
    CUdeviceptr base = 0;
    size_t baseSize = 0;
    NCCLCHECKGOTO(ncclCuMemGetAddressRange((CUdeviceptr)data, size, &base, &baseSize, &nSeg), ret, fail);
  }
#endif
  if (nSeg < 1) nSeg = 1;
  if (nSeg > NCCL_RMA_MAX_SEGMENTS) {
    WARN("NET/IB-CAST/RMA: buffer %p (size %zu) spans %d segments, exceeds NCCL_RMA_MAX_SEGMENTS=%d", data, size, nSeg,
         NCCL_RMA_MAX_SEGMENTS);
    ret = ncclInvalidUsage;
    goto fail;
  }
  ginMrHandle->nSegments = nSeg;
  ginMrHandle->segOff[0] = 0;

  if (nSeg == 1) {
    NCCLCHECKGOTO(IbCastRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags,
                                            (void**)&ginMrHandle->mrHandle[0]),
                  ret, fail);
    registered = 1;
    ginMrHandle->segOff[1] = size;
    localVas[0] = (uintptr_t)data;
  } else {
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
    uintptr_t segPtr = (uintptr_t)data;
    size_t remaining = size;
    size_t cum = 0;
    for (int s = 0; s < nSeg; s++) {
      CUdeviceptr segBase = 0;
      size_t segSize = 0;
      CUCHECKGOTO(cuMemGetAddressRange(&segBase, &segSize, (CUdeviceptr)segPtr), ret, fail);
      size_t inSeg = segSize - (segPtr - (uintptr_t)segBase);
      size_t thisLen = remaining < inSeg ? remaining : inSeg;
      int segFd = -1;
      CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&segFd, (CUdeviceptr)segPtr, thisLen,
                                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                  ret, fail);
      ret = IbCastRegMrDmaBufInternal(cComm->recvComm, (void*)segPtr, thisLen, type, 0ULL, segFd, mr_flags,
                                      (void**)&ginMrHandle->mrHandle[s]);
      (void)close(segFd);
      if (ret != ncclSuccess) goto fail;
      registered = s + 1;
      localVas[s] = segPtr;
      cum += thisLen;
      ginMrHandle->segOff[s + 1] = cum;
      segPtr += thisLen;
      remaining -= thisLen;
    }
    INFO(NCCL_NET | NCCL_REG, "NET/IB-CAST/RMA: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", data,
         size, nSeg);
#else
    WARN("NET/IB-CAST/RMA: multi-segment (%d) registration requires HIP >= 7.12.60540", nSeg);
    ret = ncclInvalidUsage;
    goto fail;
#endif
  }

  {
    struct ncclIbNetCommBase* recvBase = (struct ncclIbNetCommBase*)cComm->recvComm;
    int ndevs = recvBase->vProps.ndevs;
    memset(localRkeys, 0, sizeof(localRkeys));
    if (ndevs < 1 || ndevs > NCCL_IB_MAX_DEVS_PER_NIC) {
      WARN("NET/IB-CAST/RMA: invalid ndevs %d for buffer %p", ndevs, data);
      ret = ncclInternalError;
      goto fail;
    }
    for (int s = 0; s < nSeg; s++) {
      for (int d = 0; d < ndevs; d++) {
        struct ibv_mr* mr = ginMrHandle->mrHandle[s]->mrs[d];
        if (mr == NULL) {
          WARN("NET/IB-CAST/RMA: missing MR for segment %d device %d", s, d);
          ret = ncclInternalError;
          goto fail;
        }
        localRkeys[(size_t)s * NCCL_IB_MAX_DEVS_PER_NIC + d] = mr->rkey;
      }
    }
  }

  {
    int* allNSeg = NULL;
    NCCLCHECKGOTO(ncclCalloc(&allNSeg, cComm->nranks), ret, fail);
    ret = cComm->allGather(cComm, &nSeg, allNSeg, sizeof(int));
    for (int r = 0; ret == ncclSuccess && r < cComm->nranks; r++) {
      if (allNSeg[r] != nSeg) {
        WARN("NET/IB-CAST/RMA: buffer %p segment-count mismatch (rank %d has %d, local %d); "
             "symmetric registration required",
             data, r, allNSeg[r], nSeg);
        ret = ncclInternalError;
      }
    }
    free(allNSeg);
    if (ret != ncclSuccess) goto fail;
  }

  NCCLCHECKGOTO(ncclCalloc(&ginMrHandle->base_vas, (size_t)cComm->nranks * nSeg), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ginMrHandle->rkeys, (size_t)cComm->nranks * nSeg * NCCL_IB_MAX_DEVS_PER_NIC), ret, fail);

  NCCLCHECKGOTO(cComm->allGather(cComm, localVas, ginMrHandle->base_vas, sizeof(uintptr_t) * nSeg), ret, fail);
  NCCLCHECKGOTO(cComm->allGather(cComm, localRkeys, ginMrHandle->rkeys,
                                 sizeof(uint32_t) * nSeg * NCCL_IB_MAX_DEVS_PER_NIC),
                ret, fail);

  *mhandle = ginMrHandle;
  return ncclSuccess;

fail:
  if (ginMrHandle) {
    for (int s = 0; s < registered; s++) {
      if (ginMrHandle->mrHandle[s]) (void)netIbCast.deregMr(cComm->recvComm, ginMrHandle->mrHandle[s]);
    }
    free(ginMrHandle->base_vas);
    free(ginMrHandle->rkeys);
    free(ginMrHandle);
  }
  return ret;
}

ncclResult_t IbCastRmaIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags,
                                      void** mhandle) {
  return IbCastRmaIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle);
}

ncclResult_t IbCastRmaIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)collComm;
  struct IbCastRmaProxyMrHandle* ginMrHandle = (struct IbCastRmaProxyMrHandle*)mhandle;

  for (int s = 0; s < ginMrHandle->nSegments; s++) {
    if (ginMrHandle->mrHandle[s]) NCCLCHECK(netIbCast.deregMr(cComm->recvComm, ginMrHandle->mrHandle[s]));
  }
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

static ncclResult_t IbCastRmaIbProxyGetSendComm(struct IbCastRmaIbProxyCtx* ginProxyCtx, int rank,
                                                struct ncclIbSendComm** commPtr) {
  *commPtr = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  if (*commPtr == NULL) {
    WARN("NET/IB-CAST/RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
static ncclResult_t IbCastRmaIbProxyGetRecvComm(struct IbCastRmaIbProxyCtx* ginProxyCtx, int rank,
                                                struct ncclIbRecvComm** commPtr) {
  *commPtr = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
  if (*commPtr == NULL) {
    WARN("NET/IB-CAST/RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyIPut(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                  uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];

  struct IbCastRmaProxyMrHandle* srcMrHandle = (struct IbCastRmaProxyMrHandle*)srcMhandle;
  struct IbCastRmaProxyMrHandle* dstMrHandle = (struct IbCastRmaProxyMrHandle*)dstMhandle;

  if (!IbCastRmaRangeOk(srcMrHandle, srcOff, size) || !IbCastRmaRangeOk(dstMrHandle, dstOff, size)) {
    WARN("NET/IB-CAST/RMA: iput out of range (srcOff=%lu dstOff=%lu size=%zu)", srcOff, dstOff, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetSendComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2 * NCCL_RMA_MAX_SEGMENTS];
  struct ibv_sge sge[2 * NCCL_RMA_MAX_SEGMENTS];
  int nWr = 0;
  NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, 2 * NCCL_RMA_MAX_SEGMENTS, &nWr, IBV_WR_RDMA_WRITE, req - comm->base.reqs,
                                     qp, srcMrHandle, ginProxyCtx->rank, srcOff, dstMrHandle, rank, dstOff, size,
                                     /*flushSge=*/NULL));
  for (int i = 0; i < nWr; i++) {
    wr[i].send_flags = IBV_SEND_SIGNALED;
    IbCastAddEvent(req, qp->devIndex);
  }

  if (nWr > 0) {
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr[0], &bad_wr));
  }

  *request = req;
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyIGet(void* ginCtx, int context, uint64_t remoteOffset, void* remoteMhandle, size_t size,
                                  uint64_t localOffset, void* localMhandle, uint32_t rank, void** request) {
  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];

  struct IbCastRmaProxyMrHandle* remoteMrHandle = (struct IbCastRmaProxyMrHandle*)remoteMhandle;
  struct IbCastRmaProxyMrHandle* localMrHandle = (struct IbCastRmaProxyMrHandle*)localMhandle;

  if (!IbCastRmaRangeOk(remoteMrHandle, remoteOffset, size) || !IbCastRmaRangeOk(localMrHandle, localOffset, size)) {
    WARN("NET/IB-CAST/RMA: iget out of range (remoteOff=%lu localOff=%lu size=%zu)", remoteOffset, localOffset, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetSendComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2 * NCCL_RMA_MAX_SEGMENTS];
  struct ibv_sge sge[2 * NCCL_RMA_MAX_SEGMENTS];
  int nWr = 0;
  NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, 2 * NCCL_RMA_MAX_SEGMENTS, &nWr, IBV_WR_RDMA_READ, req - comm->base.reqs,
                                     qp, localMrHandle, ginProxyCtx->rank, localOffset, remoteMrHandle, rank,
                                     remoteOffset, size, /*flushSge=*/NULL));
  for (int i = 0; i < nWr; i++) {
    wr[i].send_flags = IBV_SEND_SIGNALED;
    IbCastAddEvent(req, qp->devIndex);
  }

  if (nWr > 0) {
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr[0], &bad_wr));
  }

  *request = req;
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyIPutSignal(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                        uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                        void* signalMhandle, uint64_t signalValue, uint32_t signalOp,
                                        bool isStrongSignal, void** request) {
  (void)isStrongSignal;
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("IbCastRmaIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];

  struct IbCastRmaProxyMrHandle* srcMrHandle = (struct IbCastRmaProxyMrHandle*)srcMhandle;
  struct IbCastRmaProxyMrHandle* dstMrHandle = (struct IbCastRmaProxyMrHandle*)dstMhandle;
  struct IbCastRmaProxyMrHandle* signalMrHandle = (struct IbCastRmaProxyMrHandle*)signalMhandle;

  if ((size > 0 && (!srcMrHandle || !dstMrHandle || !IbCastRmaRangeOk(srcMrHandle, srcOff, size) ||
                    !IbCastRmaRangeOk(dstMrHandle, dstOff, size))) ||
      !signalMrHandle || !IbCastRmaRangeOk(signalMrHandle, signalOff, sizeof(uint64_t))) {
    WARN("NET/IB-CAST/RMA: iputSignal out of range (srcOff=%lu dstOff=%lu size=%zu signalOff=%lu)", srcOff, dstOff, size,
         signalOff);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetSendComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2 * NCCL_RMA_MAX_SEGMENTS + 1];
  struct ibv_sge sge[2 * NCCL_RMA_MAX_SEGMENTS + 1];
  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));
  int nPut = 0;

  if (size > 0 && dstMrHandle) {
    NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, 2 * NCCL_RMA_MAX_SEGMENTS, &nPut, IBV_WR_RDMA_WRITE,
                                       req - comm->base.reqs, qp, srcMrHandle, ginProxyCtx->rank, srcOff, dstMrHandle,
                                       rank, dstOff, size, /*flushSge=*/NULL));
    for (int i = 0; i < nPut; i++) wr[i].send_flags = 0;
  }

  int sig = IbCastRmaSegOf(signalMrHandle, signalOff);
  void* signalPtr = (void*)(signalMrHandle->base_vas[(size_t)rank * signalMrHandle->nSegments + sig] +
                            (signalOff - signalMrHandle->segOff[sig]));
  uint32_t signalRkey = IbCastRmaRemoteRkey(signalMrHandle, rank, sig, qp->remDevIdx);

  struct ibv_send_wr* sigWr = &wr[nPut];
  struct ibv_sge* sigSge = &sge[nPut];
  memset(sigWr, 0, sizeof(*sigWr));
  memset(sigSge, 0, sizeof(*sigSge));
  sigWr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  sigWr->send_flags = IBV_SEND_SIGNALED;
  sigWr->wr_id = req - comm->base.reqs;
  sigWr->next = NULL;
  sigWr->wr.atomic.remote_addr = (uint64_t)signalPtr;
  sigWr->wr.atomic.compare_add = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  sigWr->wr.atomic.rkey = signalRkey;
  sigWr->sg_list = sigSge;
  sigWr->num_sge = 1;

  sigSge->addr = (uintptr_t)&comm->putSignalScratchpad;
  sigSge->length = sizeof(comm->putSignalScratchpad);
  sigSge->lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  if (nPut > 0) wr[nPut - 1].next = sigWr;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, nPut > 0 ? &wr[0] : sigWr, &bad_wr));
  IbCastAddEvent(req, qp->devIndex);
  *request = req;
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyTest(void* collComm, void* request, int* done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct IbCastRmaIbProxyCtx* ginProxyCtx = (struct IbCastRmaIbProxyCtx*)req->ginProxyCtx;
  int rank = req->iput.rank;
  *done = 0;

  if (req->events[0] == 0) {
    *done = 1;
    NCCLCHECK(IbCastFreeRequest(req));
    return ncclSuccess;
  }
  int wrDone = 0;
  struct ibv_wc wc[4];

  ncclIbNetCommBase* commBase;
  ncclIbNetCommDevBase* devBase;
  if (req->type == NCCL_NET_IB_REQ_FLUSH) {
    struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;
  } else {
    struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;
  }
  NCCLCHECK(wrap_ibv_poll_cq(devBase->cq, 4, wc, &wrDone));
  for (int i = 0; i < wrDone; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      union ncclSocketAddress addr;
      ncclSocketGetAddr(req->sock, &addr);
      char localGidString[INET6_ADDRSTRLEN] = "";
      char remoteGidString[INET6_ADDRSTRLEN] = "";
      const char *localGidStr = NULL, *remoteGidStr = NULL;
      if (req->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&commBase->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN + 1];
      char* hcaName = devBase->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
           ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err,
           IbCastReqTypeStr[req->type], localGidStr ? " localGid " : "", localGidString,
           remoteGidStr ? " remoteGids" : "", remoteGidString, hcaName);
      return ncclRemoteError;
    }

    struct ncclIbRequest* wcReq = commBase->reqs + wc[i].wr_id;

    wcReq->events[0]--;
    if (wcReq == req && wcReq->events[0] == 0) {
      *done = 1;
      NCCLCHECK(IbCastFreeRequest(wcReq));
    }
  }
  return ncclSuccess;
}

ncclResult_t IbCastRmaIbProxyIFlush(void* ginCtx, int context, void* mhandle, uint32_t rank, void** request) {
  struct IbCastRmaIbProxyCtx* ginProxyCtx = &((struct IbCastRmaIbProxyCtx*)ginCtx)[context];
  struct IbCastRmaProxyMrHandle* ginMrHandle = (struct IbCastRmaProxyMrHandle*)mhandle;
  struct ncclIbRecvComm* comm;
  NCCLCHECK(IbCastRmaIbProxyGetRecvComm(ginProxyCtx, rank, &comm));
  struct ncclIbQp* qp = &comm->devs[0].gpuFlush.qp;

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->ginProxyCtx = ginProxyCtx;

  struct ibv_send_wr wr[NCCL_RMA_MAX_SEGMENTS];
  struct ibv_sge sge[NCCL_RMA_MAX_SEGMENTS];
  int nWr = 0;
  NCCLCHECK(IbCastRmaBuildSegmentedWrs(wr, sge, NCCL_RMA_MAX_SEGMENTS, &nWr, IBV_WR_RDMA_READ, req - comm->base.reqs, qp,
                                     /*localH=*/NULL, /*localRank=*/0, /*localOff=*/0,
                                     /*remoteH=*/ginMrHandle, /*remoteRank=*/ginProxyCtx->rank, /*remoteOff=*/0,
                                     /*size=*/IbCastRmaMrBytes(ginMrHandle),
                                     /*flushSge=*/&comm->devs[qp->devIndex].gpuFlush.sge));
  for (int i = 0; i < nWr; i++) {
    wr[i].send_flags = IBV_SEND_SIGNALED;
    IbCastAddEvent(req, qp->devIndex);
  }

  TRACE(NCCL_NET, "NET/IB: %s: Posting %d-segment flush request (req=%p, comm=%p)", __func__, nWr, req, req->base);
  TIME_START(4);
  if (nWr > 0) {
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr[0], &bad_wr));
  }
  TIME_STOP(4);

  *request = req;
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
// RMA_IB_PROXY: host/proxy-initiated RMA backend for the IB-CAST transport.
// Under the v14 GIN/RMA split, the iput/iputSignal/iget/iflush/test data-path ops
// live in the ncclRma_t vtable (they were part of the old v13 ncclGin_t vtable).
// GIN over IB-CAST is provided by the generic ncclGinProxy layered on top of this
// backend (see gin_host_proxy.cc). rmaProgress/queryLastError stay NULL.
ncclRma_t IbCastRmaIbProxy = {"RMA_IB_PROXY",
                              IbCastRmaIbProxyInit,
                              IbCastDevices,
                              IbCastRmaIbProxyGetProperties,
                              IbCastListen,
                              IbCastRmaIbProxyConnect,
                              IbCastRmaIbProxyCreateContext,
                              IbCastRmaIbProxyRegMrSym,
                              IbCastRmaIbProxyRegMrSymDmaBuf,
                              IbCastRmaIbProxyDeregMrSym,
                              IbCastRmaIbProxyDestroyContext,
                              IbCastGinIbCloseColl,
                              IbCastCloseListen,
                              IbCastRmaIbProxyIPut,
                              IbCastRmaIbProxyIPutSignal,
                              IbCastRmaIbProxyIGet,
                              IbCastRmaIbProxyIFlush,
                              IbCastRmaIbProxyTest,
                              NULL,
                              NULL,
                              IbCastGinIbFinalize};
