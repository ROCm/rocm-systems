/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

#include "gin/gin_host.h"
#include "gin.h"

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

// Check GDR support for GIN. This is run at init, so we don't know yet whether the GPU will support DMA-BUF.
static ncclResult_t ncclGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  if (ncclIbDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  *gdrSupport = false;
  INFO(NCCL_NET, "Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclSuccess;
}

// Check the current GPU supports GDR for GIN. This is run during connect().
static ncclResult_t ncclGinIbGdrGpuSupport(bool gdaki) {
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

#if !defined(__HIP_PLATFORM_AMD__)
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int dmaBufSupportOnDevice = 1;
  CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
  if (dmaBufSupportOnDevice == 1) return ncclSuccess;

  WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
  return ncclInvalidUsage;
#else
  if (ncclIbDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  WARN("Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclInvalidUsage;
#endif
}

NCCL_PARAM(GinType, "GIN_TYPE", -1);

static std::mutex ncclGinIbGdakiLockMutex;
static int ncclGinIbGdakiNDevs = -1;
int ncclGinIbGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t ncclGinIbGdakiInit() {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (ncclGinIbGdakiNDevs == -1) {
    int ndevs = 0;
    for (int i = 0; i < ncclNIbDevs; i++) {
      if (ncclIbDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        ncclGinIbGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    ncclGinIbGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}

extern ncclGin_t ncclGinIb;
#if !defined(__HIP_PLATFORM_AMD__)
extern ncclGin_t ncclGinIbGdaki;
#endif // !defined(__HIP_PLATFORM_AMD__)
extern ncclGin_t ncclGinIbProxy;

// Initlialize GDAKI or PROXY backend. ginType can force a particular backend.
// If provided, overwrite ginIb with the backend (generic ginIb case).
ncclResult_t ncclGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction, int ginType, ncclGin_t* ginIb) {
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  if (ncclNIbDevs == 0) return ncclInternalError; // Caught in plugin init code, not propagated to user.

#if !defined(__HIP_PLATFORM_AMD__)
  if (ginType == NCCL_GIN_TYPE_GDAKI) goto try_gdaki;
#endif // !defined(__HIP_PLATFORM_AMD__)
  if (ginType == NCCL_GIN_TYPE_PROXY) goto try_proxy;
  if (ginType != -1) {
    INFO(NCCL_INIT|NCCL_NET, "NET_IB: no support for GIN type %ld", ncclParamGinType());
    return ncclInternalError;
  }

  bool gdrSupport;

#if !defined(__HIP_PLATFORM_AMD__)
  // First try GDAKI
try_gdaki:
  NCCLCHECK(ncclGinIbGdakiInit());
  if (ncclGinIbGdakiNDevs == 0 && ginType == -1) goto try_proxy;
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ true));
  if (!gdrSupport && ginType == -1) goto try_proxy;
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbGdaki, sizeof(ncclGinIb));
  goto end;
#endif // !defined(__HIP_PLATFORM_AMD__)

  // Then Proxy
try_proxy:
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ false));
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbProxy, sizeof(ncclGinIb));

end:
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  return ncclSuccess;
}
ncclResult_t ncclGinIbInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, ncclParamGinType(), &ncclGinIb);
}

// Forward declarations for Proxy functions referenced in ncclGinIb dispatcher below.
ncclResult_t ncclGinIbProxyGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle);
ncclResult_t ncclGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void **ginHandle);
ncclResult_t ncclGinIbProxyDeregMrSym(void* collComm, void* mhandle);
ncclResult_t ncclGinIbProxyIPut(void *collComm, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank, int connectionId,
                                void **request);
ncclResult_t ncclGinIbProxyIPutSignal(void *collComm, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                      uint64_t signalOff, void *signalMhandle, uint64_t signalValue,
                                      uint32_t signalOp, int connectionId, void **request);
ncclResult_t ncclGinIbProxyTest(void *collComm, void *request, int *done);
ncclResult_t ncclGinIbFinalize(void *ctx);
ncclResult_t ncclGinIbConnect(void *ctx, void *handles[], int nranks, int rank, int nConnections,
                              int queueDepth, void *listenComm, void **collComm);
ncclResult_t ncclGinIbCloseColl(void* collComm);

// [RCCL] ncclGinIb (the "GIN_IB" built-in) is defined *after* ncclGinIbProxy,
// near the end of this file. It must be defined there so its initializer binds
// to the v13-shaped (NCCL 2.30) ncclGin_t op definitions further down, not to
// the stale v12-era forward declarations above (which would mis-shape it).

ncclResult_t ncclGinIbFinalize(void *ctx) {
  if (ctx) free(ctx);
  return ncclIbFinalizeDevices();
}

static ncclResult_t ncclGinIbAllGather(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer;
  int rpeer;
  void *rbuf;
  int tag;
  int done;

  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->recvComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &rMhandle),
                status, out);
  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->sendComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &sMhandle),
                status, out);

  speer = cComm->rank;
  memcpy((void *)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void *)((uintptr_t)recvBuf + rpeer * len);
      tag = NCCL_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.isend(cComm->sendComm,
                                      (void *)((uintptr_t)recvBuf + speer * len),
                                      len, tag, sMhandle, NULL, &srequest),
                      status, out);
      if (rrequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.irecv(cComm->recvComm, 1, &rbuf, &len,
                                      &tag, &rMhandle, NULL, &rrequest),
                      status, out);
    }
    while (srequest || rrequest) {
      if (rrequest)
        NCCLCHECKGOTO(ncclNetIb.test(rrequest, &done, NULL),
                      status, out);
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECKGOTO(ncclNetIb.test(srequest, &done, NULL),
                      status, out);
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle)
    ncclNetIb.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle)
    ncclNetIb.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t ncclGinIbAllToAll(struct ncclGinIbCollComm *cComm, void *src_buf, void *recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;

  void *tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void **)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void *)((uintptr_t)recv_buf + i * len), (void *)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len), len);
  }

out:
  if (tmp_buf)
    free(tmp_buf);

  return status;
}

ncclResult_t ncclGinIbP2PBarrier(struct ncclGinIbCollComm *cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int *dummy;
  NCCLCHECK(ncclIbMalloc((void **)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(ncclGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t ncclGinIbConnect(void *ctx, void *handles[], int nranks, int rank,
                              void *listenComm, void **collComm) {
  struct ncclIbListenComm *lComm = (struct ncclIbListenComm *)listenComm;
  struct ncclGinIbCollComm *cCommArray = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void **)&cCommArray, sizeof(*cCommArray)));

  struct ncclGinIbCollComm *cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(ncclNetIb.connect(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL));
    }
    if (cComm->recvComm == NULL)
      NCCLCHECK(ncclNetIb.accept(lComm, &cComm->recvComm, NULL));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclIbGetProperties;
  cComm->allGather = ncclGinIbAllGather;
  cComm->allToAll = ncclGinIbAllToAll;
  cComm->getGidIndex = ncclIbGetGidIndex;
  cComm->dev = lComm->dev;

  cComm->ib.context = ncclIbDevs[cComm->dev].context;
  cComm->ib.pd = ncclIbDevs[cComm->dev].pd;

  *collComm = cCommArray;
  return ncclSuccess;
}

ncclResult_t ncclGinIbCloseColl(void* collComm) {
  struct ncclGinIbCollComm* cCommArray = (struct ncclGinIbCollComm*)collComm;
  if (!cCommArray) return ncclSuccess;

  struct ncclGinIbCollComm *cComm = cCommArray;
  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIb.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIb.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cCommArray);
  return ncclSuccess;
}

#if !defined(__HIP_PLATFORM_AMD__)
#include "gdaki/gin_host_gdaki.h"

ncclResult_t ncclGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI, NULL);
}

ncclResult_t ncclGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  *ndev = ncclGinIbGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (dev >= ncclGinIbGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created", dev, ncclGinIbGdakiNDevs);
    return ncclInvalidUsage;
  }
  NCCLCHECK(ncclIbGetPhysProperties(ncclGinIbGdakiDevIndexes[dev], props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  return ncclNetIb.listen(ctx, ncclGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ true));

  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)*collComm;
  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclGinIbGdakiGetProperties;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiCreateContext(void* collComm, ncclGinConfig_v13_t* config, void **ginCtx, ncclNetDeviceHandle_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config->nSignals, config->nCounters, config->nContexts, config->queueDepth, config->trafficClass, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinGdakiRegMrSym((struct ncclGinIbCollComm *)collComm, data, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct ncclGinIbCollComm *)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiProgress(void *collComm)
{
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t ncclGinIbGdakiQueryLastError(void *ginCtx, bool *hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t ncclGinIbGdaki = {
  "GIN_IB_GDAKI",
  ncclGinIbGdakiInit,
  ncclGinIbGdakiDevices,
  ncclGinIbGdakiGetProperties,
  ncclGinIbGdakiListen,
  ncclGinIbGdakiConnect,
  ncclGinIbGdakiCreateContext,
  ncclGinIbGdakiRegMrSym,
  NULL, // regMrSymDmaBuf
  ncclGinIbGdakiDeregMrSym,
  ncclGinIbGdakiDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiProgress,
  ncclGinIbGdakiQueryLastError,
  ncclGinIbFinalize
};
#endif // !defined(__HIP_PLATFORM_AMD__)


// Cap on physical segments per symmetric buffer.
#define NCCL_GIN_MAX_SEGMENTS 16

struct ncclIbGinProxyMrHandle {
  int nSegments;
  // segOff[0]==0, segOff[nSegments]==size; per-segment local MRs; remote view
  // base_vas/rkeys indexed [rank*nSegments + seg].
  size_t segOff[NCCL_GIN_MAX_SEGMENTS + 1];
  struct ncclIbMrHandle *mrHandle[NCCL_GIN_MAX_SEGMENTS];
  uintptr_t *base_vas;
  uint32_t  *rkeys;
};

// Return the segment index that contains byte offset off within the buffer.
static inline int ncclGinSegOf(const struct ncclIbGinProxyMrHandle* h, uint64_t off) {
  for (int s = 0; s < h->nSegments; s++)
    if (off < h->segOff[s + 1]) return s;
  return h->nSegments - 1;
}

// Total registered bytes (segOff[nSegments] is the cumulative end offset).
static inline uint64_t ncclGinMrBytes(const struct ncclIbGinProxyMrHandle* h) {
  return h->segOff[h->nSegments];
}

// True iff [off, off+size) fits in the window (overflow-safe).
static inline bool ncclGinRangeOk(const struct ncclIbGinProxyMrHandle* h, uint64_t off, size_t size) {
  uint64_t bytes = ncclGinMrBytes(h);
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
// landing a single byte in the caller's flush scratch. Passing the whole
// window thus fences GPUDirect writes across every physical segment (the
// caller reads its own buffer via base_vas[self]/rkeys[self] on a loopback QP).
static ncclResult_t ncclGinBuildSegmentedWrs(
    struct ibv_send_wr* wr, struct ibv_sge* sge, int maxWr, int* nWr, enum ibv_wr_opcode opcode, uint64_t wrId,
    struct ncclIbGinProxyMrHandle* localH,  int localRank,  uint64_t localOff,
    struct ncclIbGinProxyMrHandle* remoteH, int remoteRank, uint64_t remoteOff,
    size_t size, const struct ibv_sge* flushSge) {
  int n = 0;
  uint64_t lOff = localOff, rOff = remoteOff;
  size_t rem = size;
  while (rem > 0) {
    if (n >= maxWr) {
      WARN("NET/IB/GIN: transfer of %zu bytes spans more than %d segment slices", size, maxWr);
      return ncclInternalError;
    }
    int rs = ncclGinSegOf(remoteH, rOff);
    size_t chunk = rem;
    if (remoteH->segOff[rs + 1] - rOff < chunk) chunk = remoteH->segOff[rs + 1] - rOff;

    int ls = 0;
    if (flushSge == NULL) {
      ls = ncclGinSegOf(localH, lOff);
      if (localH->segOff[ls + 1] - lOff < chunk) chunk = localH->segOff[ls + 1] - lOff;
    }

    uintptr_t rAddr = remoteH->base_vas[(size_t)remoteRank * remoteH->nSegments + rs] + (rOff - remoteH->segOff[rs]);

    memset(&wr[n], 0, sizeof(wr[n]));
    memset(&sge[n], 0, sizeof(sge[n]));
    wr[n].opcode = opcode;
    wr[n].wr_id = wrId;
    wr[n].next = NULL;
    wr[n].wr.rdma.remote_addr = (uint64_t)rAddr;
    wr[n].wr.rdma.rkey = remoteH->rkeys[(size_t)remoteRank * remoteH->nSegments + rs];
    wr[n].sg_list = &sge[n];
    wr[n].num_sge = 1;
    if (flushSge != NULL) {
      // Touch one byte of this segment; the data lands in the fixed scratch.
      sge[n] = *flushSge;
      sge[n].length = 1;
    } else {
      uintptr_t lAddr = localH->base_vas[(size_t)localRank * localH->nSegments + ls] + (lOff - localH->segOff[ls]);
      sge[n].addr = (uintptr_t)lAddr;
      sge[n].length = chunk;
      sge[n].lkey = localH->mrHandle[ls]->mrs[0]->lkey;
    }
    if (n > 0) wr[n - 1].next = &wr[n];

    lOff += chunk; rOff += chunk; rem -= chunk; n++;
  }
  *nWr = n;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_PROXY, NULL);
}

ncclResult_t ncclGinIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetIb.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  return ncclSuccess;
}

struct ncclGinIbProxyCtx {
  void**        fullRecvComm;
  void**        fullSendComm;
  int rank, nranks;
  int nContexts;
};

ncclResult_t ncclGinIbProxyCreateContext(void* collComm, ncclGinConfig_v13_t* config, void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  // Make sure all QP we create use the provided traffic class.
  ncclIbSetTrafficClass(cComm->ctx, config->trafficClass);

  if (config->queueDepth != 0) {
    WARN("GIN_IB_PROXY does not support specifying qp depth");
    return ncclInvalidUsage;
  }

  int nranks;
  struct ncclGinIbProxyCtx* ginProxyCtx = NULL;
  *ginCtx = NULL;
  NCCLCHECK(ncclCalloc(&ginProxyCtx, config->nContexts));
  ginProxyCtx[0].nContexts = config->nContexts;
  ginProxyCtx[0].nranks = nranks = cComm->nranks;

  void *lComm = NULL;
  char* handle = NULL, *handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE*cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE*cComm->rank;

  NCCLCHECKGOTO(ncclNetIb.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, end);
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c=0; c<config->nContexts; c++) {
    struct ncclGinIbProxyCtx* gc = ginProxyCtx+c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void *) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void *) * nranks), ret, end);
    gc->rank = cComm->rank;

    for (int i = 0; i < nranks; i++) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;
      do {
        if (gc->fullSendComm[connectPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.connect(cComm->ctx, cComm->dev, handles+NCCL_NET_HANDLE_MAXSIZE*connectPeer, &gc->fullSendComm[connectPeer], NULL), ret, end);
        if (gc->fullRecvComm[acceptPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.accept(lComm, &gc->fullRecvComm[acceptPeer], NULL), ret, end);
      } while ((gc->fullSendComm[connectPeer] == NULL) ||
          (gc->fullRecvComm[acceptPeer] == NULL));
      NCCLCHECKGOTO(ncclGinIbP2PBarrier(cComm), ret, end);
    }
  }

end:
  free(handles);
  if (lComm) ncclNetIb.closeListen(lComm);
  if (ret != ncclSuccess) free(ginProxyCtx);
  else *ginCtx = ginProxyCtx;
  return ret;
}

ncclResult_t ncclGinIbProxyDestroyContext(void* ginCtx) {
  struct ncclGinIbProxyCtx* gc = (struct ncclGinIbProxyCtx*)ginCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;
  for (int c=0; c<nContexts; c++) {
    if (gc[c].fullRecvComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle = NULL;
  uintptr_t localVas[NCCL_GIN_MAX_SEGMENTS];
  uint32_t  localRkeys[NCCL_GIN_MAX_SEGMENTS];
  ncclResult_t ret = ncclSuccess;
  int nSeg = 1;
  int registered = 0;

  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));

  // Count physical segments; ROCm/HIP describes only the first per export
  // (AIRUNTIME-2351), so multi-segment ranges register one MR per segment below.
#if CUDA_VERSION >= 11070 || HIP_VERSION >= 71260540
  if (type == NCCL_PTR_CUDA && ncclCuMemEnable()) {
    CUdeviceptr base = 0;
    size_t baseSize = 0;
    NCCLCHECKGOTO(ncclCuMemGetAddressRange((CUdeviceptr)data, size, &base, &baseSize, &nSeg), ret, fail);
  }
#endif
  if (nSeg < 1) nSeg = 1;
  if (nSeg > NCCL_GIN_MAX_SEGMENTS) {
    WARN("NET/IB/GIN: buffer %p (size %zu) spans %d segments, exceeds NCCL_GIN_MAX_SEGMENTS=%d",
         data, size, nSeg, NCCL_GIN_MAX_SEGMENTS);
    ret = ncclInvalidUsage;
    goto fail;
  }
  ginMrHandle->nSegments = nSeg;
  ginMrHandle->segOff[0] = 0;

  if (nSeg == 1) {
    // Single segment: reuse the caller's fd (fd==-1 falls back to ibv_reg_mr).
    NCCLCHECKGOTO(ncclIbRegMrDmaBuf(cComm->recvComm, data, size, type, offset, fd,
                                    (void**)&ginMrHandle->mrHandle[0]), ret, fail);
    registered = 1;
    ginMrHandle->segOff[1] = size;
    localVas[0] = (uintptr_t)data;
    localRkeys[0] = ginMrHandle->mrHandle[0]->mrs[0]->rkey;
  } else {
#if CUDA_VERSION >= 11070 || HIP_VERSION >= 71260540
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
      // Export this segment alone: one physical allocation, so its fd is complete.
      CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&segFd, (CUdeviceptr)segPtr, thisLen,
                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0), ret, fail);
      ret = ncclIbRegMrDmaBuf(cComm->recvComm, (void*)segPtr, thisLen, type, 0ULL, segFd,
                              (void**)&ginMrHandle->mrHandle[s]);
      (void)close(segFd);
      if (ret != ncclSuccess) goto fail;
      registered = s + 1;
      localVas[s] = segPtr;
      localRkeys[s] = ginMrHandle->mrHandle[s]->mrs[0]->rkey;
      cum += thisLen;
      ginMrHandle->segOff[s + 1] = cum;
      segPtr += thisLen;
      remaining -= thisLen;
    }
    INFO(NCCL_NET|NCCL_REG, "NET/IB/GIN: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs",
         data, size, nSeg);
#else
    WARN("NET/IB/GIN: multi-segment (%d) registration requires HIP >= 7.12.60540", nSeg);
    ret = ncclInvalidUsage;
    goto fail;
#endif
  }

  // Cross-rank symmetry guard: every rank must register the same segment count,
  // else the fixed-stride all-gather/indexing below would corrupt memory.
  {
    int* allNSeg = NULL;
    NCCLCHECKGOTO(ncclCalloc(&allNSeg, cComm->nranks), ret, fail);
    ret = cComm->allGather(cComm, &nSeg, allNSeg, sizeof(int));
    for (int r = 0; ret == ncclSuccess && r < cComm->nranks; r++) {
      if (allNSeg[r] != nSeg) {
        WARN("NET/IB/GIN: buffer %p segment-count mismatch (rank %d has %d, local %d); "
             "symmetric registration required", data, r, allNSeg[r], nSeg);
        ret = ncclInternalError;
      }
    }
    free(allNSeg);
    if (ret != ncclSuccess) goto fail;
  }

  NCCLCHECKGOTO(ncclCalloc(&ginMrHandle->base_vas, (size_t)cComm->nranks * nSeg), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ginMrHandle->rkeys, (size_t)cComm->nranks * nSeg), ret, fail);

  // Gather per-segment base VAs/rkeys; symmetry verified above keeps nSeg/sizes aligned.
  NCCLCHECKGOTO(cComm->allGather(cComm, localVas, ginMrHandle->base_vas, sizeof(uintptr_t) * nSeg), ret, fail);
  NCCLCHECKGOTO(cComm->allGather(cComm, localRkeys, ginMrHandle->rkeys, sizeof(uint32_t) * nSeg), ret, fail);

  *mhandle = ginMrHandle;
  *ginHandle = ginMrHandle;
  return ncclSuccess;

fail:
  if (ginMrHandle) {
    for (int s = 0; s < registered; s++) {
      if (ginMrHandle->mrHandle[s]) (void)ncclNetIb.deregMr(cComm->recvComm, ginMrHandle->mrHandle[s]);
    }
    free(ginMrHandle->base_vas);
    free(ginMrHandle->rkeys);
    free(ginMrHandle);
  }
  return ret;
}

ncclResult_t ncclGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;

  for (int s = 0; s < ginMrHandle->nSegments; s++) {
    if (ginMrHandle->mrHandle[s]) NCCLCHECK(ncclNetIb.deregMr(cComm->recvComm, ginMrHandle->mrHandle[s]));
  }
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPut(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;

  // Reject out-of-range transfers before any WR is posted.
  if (!ncclGinRangeOk(srcMrHandle, srcOff, size) || !ncclGinRangeOk(dstMrHandle, dstOff, size)) {
    WARN("NET/IB/GIN: iput out of range (srcOff=%lu dstOff=%lu size=%zu)", srcOff, dstOff, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  // Split the transfer at segment boundaries on both the local (src) and remote
  // (dst) buffers; each slice maps to one local lkey and one remote rkey.
  struct ibv_send_wr wr[2 * NCCL_GIN_MAX_SEGMENTS];
  struct ibv_sge sge[2 * NCCL_GIN_MAX_SEGMENTS];
  int nWr = 0;
  NCCLCHECK(ncclGinBuildSegmentedWrs(wr, sge, 2 * NCCL_GIN_MAX_SEGMENTS, &nWr, IBV_WR_RDMA_WRITE,
                                     req - comm->base.reqs,
                                     srcMrHandle, ginProxyCtx->rank, srcOff,
                                     dstMrHandle, rank, dstOff, size, /*flushSge=*/NULL));
  for (int i = 0; i < nWr; i++) {
    wr[i].send_flags = IBV_SEND_SIGNALED;
    ncclIbAddEvent(req, qp->devIndex);
  }

  // size==0 yields nWr==0: nothing to post; the request completes in test()
  // (events[0]==0). Posting wr[0] here would submit an uninitialized WR.
  if (nWr > 0) {
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr[0], &bad_wr));
  }

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIGet(void *ginCtx, int context, uint64_t remoteOffset, void *remoteMhandle,
                                 size_t size, uint64_t localOffset, void *localMhandle, uint32_t rank,
                                 void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle *remoteMrHandle = (struct ncclIbGinProxyMrHandle *)remoteMhandle;
  struct ncclIbGinProxyMrHandle *localMrHandle = (struct ncclIbGinProxyMrHandle *)localMhandle;

  // Reject out-of-range transfers before any WR is posted.
  if (!ncclGinRangeOk(remoteMrHandle, remoteOffset, size) || !ncclGinRangeOk(localMrHandle, localOffset, size)) {
    WARN("NET/IB/GIN: iget out of range (remoteOff=%lu localOff=%lu size=%zu)", remoteOffset, localOffset, size);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  // RDMA READ: local buffer is the destination (lkey), remote buffer the source
  // (rkey). Split at segment boundaries on both sides.
  struct ibv_send_wr wr[2 * NCCL_GIN_MAX_SEGMENTS];
  struct ibv_sge sge[2 * NCCL_GIN_MAX_SEGMENTS];
  int nWr = 0;
  NCCLCHECK(ncclGinBuildSegmentedWrs(wr, sge, 2 * NCCL_GIN_MAX_SEGMENTS, &nWr, IBV_WR_RDMA_READ,
                                     req - comm->base.reqs,
                                     localMrHandle, ginProxyCtx->rank, localOffset,
                                     remoteMrHandle, rank, remoteOffset, size, /*flushSge=*/NULL));
  for (int i = 0; i < nWr; i++) {
    wr[i].send_flags = IBV_SEND_SIGNALED;
    ncclIbAddEvent(req, qp->devIndex);
  }

  // size==0 yields nWr==0: nothing to post; the request completes in test().
  if (nWr > 0) {
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr[0], &bad_wr));
  }

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPutSignal(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                      uint64_t signalOff, void *signalMhandle, uint64_t signalValue,
                                      uint32_t signalOp, void **request) {
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("ncclGinIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;
  struct ncclIbGinProxyMrHandle *signalMrHandle = (struct ncclIbGinProxyMrHandle *)signalMhandle;

  // Reject out-of-range payload/signal before any WR is posted (signal is an 8-byte atomic).
  if ((size > 0 && (!srcMrHandle || !dstMrHandle ||
                    !ncclGinRangeOk(srcMrHandle, srcOff, size) ||
                    !ncclGinRangeOk(dstMrHandle, dstOff, size))) ||
      !signalMrHandle || !ncclGinRangeOk(signalMrHandle, signalOff, sizeof(uint64_t))) {
    WARN("NET/IB/GIN: iputSignal out of range (srcOff=%lu dstOff=%lu size=%zu signalOff=%lu)",
         srcOff, dstOff, size, signalOff);
    return ncclInvalidArgument;
  }

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  // Up to 2*NCCL_GIN_MAX_SEGMENTS slices for the (segmented) PUT plus one signal WR.
  struct ibv_send_wr wr[2 * NCCL_GIN_MAX_SEGMENTS + 1];
  struct ibv_sge sge[2 * NCCL_GIN_MAX_SEGMENTS + 1];
  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));
  int nPut = 0;

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  if (size > 0 && dstMrHandle) {
    // PUT slices carry no CQE; only the trailing signal is signaled. Same-QP RC
    // ordering guarantees all writes land before the signal.
    NCCLCHECK(ncclGinBuildSegmentedWrs(wr, sge, 2 * NCCL_GIN_MAX_SEGMENTS, &nPut, IBV_WR_RDMA_WRITE,
                                       req - comm->base.reqs,
                                       srcMrHandle, ginProxyCtx->rank, srcOff,
                                       dstMrHandle, rank, dstOff, size, /*flushSge=*/NULL));
    for (int i = 0; i < nPut; i++) wr[i].send_flags = 0;
  }

  // SIGNAL (route to the segment that contains signalOff)
  int sig = ncclGinSegOf(signalMrHandle, signalOff);
  void *signalPtr = (void *)(signalMrHandle->base_vas[(size_t)rank * signalMrHandle->nSegments + sig]
                             + (signalOff - signalMrHandle->segOff[sig]));
  uint32_t signalRkey = signalMrHandle->rkeys[(size_t)rank * signalMrHandle->nSegments + sig];

  struct ibv_send_wr* sigWr = &wr[nPut];
  struct ibv_sge* sigSge = &sge[nPut];
  memset(sigWr, 0, sizeof(*sigWr));
  memset(sigSge, 0, sizeof(*sigSge));
  sigWr->opcode                  = IBV_WR_ATOMIC_FETCH_AND_ADD;
  sigWr->send_flags              = IBV_SEND_SIGNALED;
  sigWr->wr_id                   = req - comm->base.reqs;  // used for matching completions with request
  sigWr->next                    = NULL;
  sigWr->wr.atomic.remote_addr   = (uint64_t)signalPtr;
  sigWr->wr.atomic.compare_add   = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  sigWr->wr.atomic.rkey          = signalRkey;
  sigWr->sg_list = sigSge;
  sigWr->num_sge = 1;

  sigSge->addr = (uintptr_t)&comm->putSignalScratchpad;
  sigSge->length = sizeof(comm->putSignalScratchpad);
  sigSge->lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  if (nPut > 0) wr[nPut - 1].next = sigWr;

  // Send the put and the signal in one go
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, nPut > 0 ? &wr[0] : sigWr, &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyTest(void* collComm, void *request, int *done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct ncclGinIbProxyCtx* ginProxyCtx = (struct ncclGinIbProxyCtx*)req->ginProxyCtx;
  int rank = req->iput.rank;
  *done = 0;

  if (req->events[0] == 0) {
    *done = 1;
    NCCLCHECK(ncclIbFreeRequest(req));
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
      const char* localGidStr = NULL, *remoteGidStr = NULL;
      if (req->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&commBase->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN+1];
      char *hcaName = devBase->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
          ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err, ncclIbReqTypeStr[req->type],
          localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
      return ncclRemoteError;
    }

    struct ncclIbRequest* wcReq = commBase->reqs + wc[i].wr_id;

    wcReq->events[0]--;
    if (wcReq == req && wcReq->events[0] == 0) {
      *done = 1;
      NCCLCHECK(ncclIbFreeRequest(wcReq));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIFlush(void *ginCtx, int context, void* mhandle, uint32_t rank, void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;
  struct ncclIbQp *qp = &comm->devs[0].gpuFlush.qp;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->ginProxyCtx = ginProxyCtx;

  // Fence GPUDirect writes across EVERY physical segment, not just segment 0.
  // The builder (flush mode) emits one loopback RDMA_READ per segment of the
  // local recv buffer -- read via base_vas[self]/rkeys[self], the same own-MR
  // addr/rkey ncclIbIflush uses -- each landing one byte in the flush scratch.
  struct ibv_send_wr wr[NCCL_GIN_MAX_SEGMENTS];
  struct ibv_sge sge[NCCL_GIN_MAX_SEGMENTS];
  int nWr = 0;
  NCCLCHECK(ncclGinBuildSegmentedWrs(wr, sge, NCCL_GIN_MAX_SEGMENTS, &nWr, IBV_WR_RDMA_READ,
                                     req - comm->base.reqs,
                                     /*localH=*/NULL, /*localRank=*/0, /*localOff=*/0,
                                     /*remoteH=*/ginMrHandle, /*remoteRank=*/ginProxyCtx->rank, /*remoteOff=*/0,
                                     /*size=*/ncclGinMrBytes(ginMrHandle),
                                     /*flushSge=*/&comm->devs[qp->devIndex].gpuFlush.sge));
  for (int i = 0; i < nWr; i++) {
    wr[i].send_flags = IBV_SEND_SIGNALED;
    ncclIbAddEvent(req, qp->devIndex);
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
ncclGin_t ncclGinIbProxy = {
  "GIN_IB_PROXY",
  ncclGinIbProxyInit,
  ncclIbDevices,
  ncclGinIbProxyGetProperties,
  ncclIbListen,
  ncclGinIbProxyConnect,
  ncclGinIbProxyCreateContext,
  ncclGinIbProxyRegMrSym,
  ncclGinIbProxyRegMrSymDmaBuf,
  ncclGinIbProxyDeregMrSym,
  ncclGinIbProxyDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclGinIbProxyIPut,
  ncclGinIbProxyIPutSignal,
  ncclGinIbProxyIGet,
  ncclGinIbProxyIFlush,
  ncclGinIbProxyTest,
  NULL,
  NULL,
  ncclGinIbFinalize
};

// [RCCL] NCCL 2.29.7 introduced a top-level "ncclGinIb" dispatcher that picks
// between GDAKI and Proxy at runtime. AMD doesn't ship the GDAKI driver-mode
// kernels yet, so ncclGinIb points at the Proxy implementation -- a strict
// subset that always works on ROCm HCAs. Defined here (after the proxy ops)
// so it binds to the v13 ncclGin_t layout (NCCL 2.30). It mirrors the prior
// behaviour exactly: same ops as before, with the new v13-only slots
// (createContext/destroyContext/iget/iflush) left NULL as in the v12 subset.
ncclGin_t ncclGinIb = {
  "GIN_IB",
  ncclGinIbInit,                 // dispatcher init (honours NCCL_GIN_TYPE)
  ncclIbDevices,
  ncclGinIbProxyGetProperties,
  ncclIbListen,
  ncclGinIbConnect,
  NULL,                          // createContext (subset; NULL as in prior v12 table)
  ncclGinIbProxyRegMrSym,
  ncclGinIbProxyRegMrSymDmaBuf,
  ncclGinIbProxyDeregMrSym,
  NULL,                          // destroyContext (subset)
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclGinIbProxyIPut,
  ncclGinIbProxyIPutSignal,
  NULL,                          // iget (added in v13 ncclGin_t; GIN_IB has no get)
  NULL,                          // iflush (added in v13 ncclGin_t)
  ncclGinIbProxyTest,
  NULL,                          // ginProgress
  NULL,                          // queryLastError
  ncclGinIbFinalize
};
