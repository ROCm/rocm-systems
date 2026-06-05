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

const int IBCAST_GIN_IB_ALLGATHER_TAG = 0xa0;
const int IBCAST_GIN_IB_ALLTOALL_TAG = 0xa1;

// Check GDR support for GIN. This is run at init, so we don't know yet whether the GPU will support DMA-BUF.
static ncclResult_t IbCastGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
  bool peerMemSupport =
     gdaki ? IbCastPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
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

  WARN("Unable to use GIN: Peermem is not supported, and DMA-BUF is not available.");
#else
  bool peerMemSupport =
     gdaki ? IbCastPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
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

extern ncclGin_t IbCastGinIb;
#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
extern ncclGin_t IbCastGinIbGdaki;
#endif
extern ncclGin_t IbCastGinIbProxy;

// Initlialize GDAKI or PROXY backend. ginType can force a particular backend.
// If provided, overwrite ginIb with the backend (generic ginIb case).
ncclResult_t IbCastGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction, int ginType, ncclGin_t* ginIb) {
  NCCLCHECK(IbCastInitDevices(logFunction, nullptr));
  if (IbCastNDevs == 0) return ncclInternalError; // Caught in plugin init code, not propagated to user.

#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
  if (ginType == NCCL_GIN_TYPE_GDAKI) goto try_gdaki;
#endif
  if (ginType == NCCL_GIN_TYPE_PROXY) goto try_proxy;
  if (ginType != -1) {
    INFO(NCCL_INIT|NCCL_NET, "NET_IB: no support for GIN type %ld", ncclParamCastGinType());
    return ncclInternalError;
  }

  bool gdrSupport;

#ifdef RCCL_NET_IB_CAST_ENABLE_GDAKI
  // First try GDAKI
try_gdaki:
  NCCLCHECK(IbCastGinIbGdakiInit());
  if (IbCastGinGdakiNDevs == 0 && ginType == -1) goto try_proxy;
  NCCLCHECK(IbCastGinIbGdrSupport(&gdrSupport, /*gdaki*/ true));
  if (!gdrSupport && ginType == -1) goto try_proxy;
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &IbCastGinIbGdaki, sizeof(IbCastGinIb));
  goto end;
#endif // RCCL_NET_IB_CAST_ENABLE_GDAKI

  // Then Proxy
try_proxy:
  NCCLCHECK(IbCastGinIbGdrSupport(&gdrSupport, /*gdaki*/ false));
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &IbCastGinIbProxy, sizeof(IbCastGinIb));

end:
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  return ncclSuccess;
}
ncclResult_t IbCastGinIbInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return IbCastGinIbInitType(ctx, commId, logFunction, ncclParamCastGinType(), &IbCastGinIb);
}

// GIN Entry point, which will then morph into either the GDAKI or PROXY backend
ncclGin_t IbCastGinIb = {
  "GIN_IB",
  IbCastGinIbInit,
  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};

ncclResult_t IbCastGinIbFinalize(void *ctx) {
  if (ctx) free(ctx);
  return IbCastFinalizeDevices();
}

static ncclResult_t IbCastGinIbAllGather(struct CastIbGinCollComm *cComm, void *srcBuf, void *recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer;
  int rpeer;
  void *rbuf;
  int tag;
  int done;

  NCCLCHECKGOTO(netIbCast.regMr(cComm->recvComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &rMhandle),
                status, out);
  NCCLCHECKGOTO(netIbCast.regMr(cComm->sendComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &sMhandle),
                status, out);

  speer = cComm->rank;
  memcpy((void *)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void *)((uintptr_t)recvBuf + rpeer * len);
      tag = IBCAST_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(netIbCast.isend(cComm->sendComm,
                                      (void *)((uintptr_t)recvBuf + speer * len),
                                      len, tag, sMhandle, NULL, &srequest),
                      status, out);
      if (rrequest == NULL)
        NCCLCHECKGOTO(netIbCast.irecv(cComm->recvComm, 1, &rbuf, &len,
                                      &tag, &rMhandle, NULL, &rrequest),
                      status, out);
    }
    while (srequest || rrequest) {
      if (rrequest)
        NCCLCHECKGOTO(netIbCast.test(rrequest, &done, NULL),
                      status, out);
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECKGOTO(netIbCast.test(srequest, &done, NULL),
                      status, out);
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle)
    netIbCast.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle)
    netIbCast.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t IbCastGinIbAllToAll(struct CastIbGinCollComm *cComm, void *src_buf, void *recv_buf, size_t len) {
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

ncclResult_t IbCastGinIbP2PBarrier(struct CastIbGinCollComm *cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int *dummy;
  NCCLCHECK(ncclIbMalloc((void **)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(IbCastGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t IbCastGinIbConnect(void *ctx, void *handles[], int nranks, int rank,
                              void *listenComm, void **collComm) {
  struct ncclIbListenComm *lComm = (struct ncclIbListenComm *)listenComm;
  struct CastIbGinCollComm *cCommArray = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void **)&cCommArray, sizeof(*cCommArray)));

  struct CastIbGinCollComm *cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(netIbCast.connect(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL));
    }
    if (cComm->recvComm == NULL)
      NCCLCHECK(netIbCast.accept(lComm, &cComm->recvComm, NULL));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))IbCastGetProperties;
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

  struct CastIbGinCollComm *cComm = cCommArray;
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
  return IbCastGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI, NULL);
}

ncclResult_t IbCastGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(IbCastGinGdakiLockMutex);
  *ndev = IbCastGinGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(IbCastGinGdakiLockMutex);
  if (dev >= IbCastGinGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created", dev, IbCastGinGdakiNDevs);
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

ncclResult_t IbCastGinIbGdakiConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(IbCastGinIbGdrGpuSupport(/*gdaki*/ true));

  NCCLCHECK(
    IbCastGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct CastIbGinCollComm *cComm = (struct CastIbGinCollComm *)*collComm;
  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))IbCastGinIbGdakiGetProperties;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbGdakiCreateContext(void* collComm, ncclGinConfig_v13_t* config, void **ginCtx, ncclNetDeviceHandle_t** devHandle) {
  struct CastIbGinCollComm* cComm = (struct CastIbGinCollComm*)collComm;

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config->nSignals, config->nCounters, config->nContexts, config->queueDepth, config->trafficClass, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t IbCastGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinGdakiRegMrSym((struct CastIbGinCollComm *)collComm, data, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t IbCastGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct CastIbGinCollComm *)collComm, mhandle);
}

ncclResult_t IbCastGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t IbCastGinIbGdakiProgress(void *collComm)
{
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t IbCastGinIbGdakiQueryLastError(void *ginCtx, bool *hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t IbCastGinIbGdaki = {
  "GIN_IB_GDAKI",
  IbCastGinIbGdakiInit,
  IbCastGinIbGdakiDevices,
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
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  IbCastGinIbGdakiProgress,
  IbCastGinIbGdakiQueryLastError,
  IbCastGinIbFinalize
};
#endif // RCCL_NET_IB_CAST_ENABLE_GDAKI


struct IbCastGinProxyMrHandle {
  struct ncclIbMrHandle *mrHandle;
  uintptr_t *base_vas;
  uint32_t *rkeys;
};

ncclResult_t IbCastGinIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return IbCastGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_PROXY, NULL);
}

ncclResult_t IbCastGinIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(netIbCast.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(IbCastGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(
    IbCastGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  return ncclSuccess;
}

struct IbCastGinIbProxyCtx {
  void**        fullRecvComm;
  void**        fullSendComm;
  int rank, nranks;
  int nContexts;
};

ncclResult_t IbCastGinIbProxyCreateContext(void* collComm, ncclGinConfig_v13_t* config, void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  ncclResult_t ret = ncclSuccess;
  struct CastIbGinCollComm *cComm = (struct CastIbGinCollComm *)collComm;
  // Make sure all QP we create use the provided traffic class.
  IbCastSetTrafficClass(cComm->ctx, config->trafficClass);

  if (config->queueDepth != 0) {
    WARN("GIN_IB_PROXY does not support specifying qp depth");
    return ncclInvalidUsage;
  }

  int nranks;
  struct IbCastGinIbProxyCtx* ginProxyCtx = NULL;
  *ginCtx = NULL;
  NCCLCHECK(ncclCalloc(&ginProxyCtx, config->nContexts));
  ginProxyCtx[0].nContexts = config->nContexts;
  ginProxyCtx[0].nranks = nranks = cComm->nranks;

  void *lComm = NULL;
  char* handle = NULL, *handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE*cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE*cComm->rank;

  //Mark communicator as RMA communicator: 1QP, Flush enabled, no CTS offload.
  ((struct ncclIbHandle*)handle)->isRMA = true;

  NCCLCHECKGOTO(netIbCast.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, end);
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c=0; c<config->nContexts; c++) {
    struct IbCastGinIbProxyCtx* gc = ginProxyCtx+c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void *) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void *) * nranks), ret, end);
    gc->rank = cComm->rank;

    for (int i = 0; i < nranks; i++) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;
      do {
        if (gc->fullSendComm[connectPeer] == NULL)
          NCCLCHECKGOTO(netIbCast.connect(cComm->ctx, cComm->dev, handles+NCCL_NET_HANDLE_MAXSIZE*connectPeer, &gc->fullSendComm[connectPeer], NULL), ret, end);
        if (gc->fullRecvComm[acceptPeer] == NULL)
          NCCLCHECKGOTO(netIbCast.accept(lComm, &gc->fullRecvComm[acceptPeer], NULL), ret, end);
      } while ((gc->fullSendComm[connectPeer] == NULL) ||
          (gc->fullRecvComm[acceptPeer] == NULL));
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

ncclResult_t IbCastGinIbProxyDestroyContext(void* ginCtx) {
  struct IbCastGinIbProxyCtx* gc = (struct IbCastGinIbProxyCtx*)ginCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;
  for (int c=0; c<nContexts; c++) {
    if (gc[c].fullRecvComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(netIbCast.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(netIbCast.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }
  }
  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  struct CastIbGinCollComm *cComm = (struct CastIbGinCollComm *)collComm;
  struct IbCastGinProxyMrHandle *ginMrHandle;
  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));

  NCCLCHECKNOWARN(IbCastRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags, (void **)&ginMrHandle->mrHandle), NCCL_NET);

  NCCLCHECK(ncclCalloc(&ginMrHandle->base_vas, cComm->nranks));
  NCCLCHECK(ncclCalloc(&ginMrHandle->rkeys, cComm->nranks));

  NCCLCHECK(cComm->allGather(cComm, &data, ginMrHandle->base_vas, sizeof(uintptr_t)));
  NCCLCHECK(cComm->allGather(cComm, &ginMrHandle->mrHandle->mrs[0]->rkey, ginMrHandle->rkeys, sizeof(uint32_t)));

  *mhandle = ginMrHandle;
  *ginHandle = ginMrHandle;

  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return IbCastGinIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle, ginHandle);
}

ncclResult_t IbCastGinIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct CastIbGinCollComm *cComm = (struct CastIbGinCollComm *)collComm;
  struct IbCastGinProxyMrHandle *ginMrHandle = (struct IbCastGinProxyMrHandle *)mhandle;

  NCCLCHECK(netIbCast.deregMr(cComm->recvComm, ginMrHandle->mrHandle));
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyIPut(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                void **request) {
  struct IbCastGinIbProxyCtx* ginProxyCtx = &((struct IbCastGinIbProxyCtx*)ginCtx)[context];

  struct IbCastGinProxyMrHandle *srcMrHandle = (struct IbCastGinProxyMrHandle *)srcMhandle;
  struct IbCastGinProxyMrHandle *dstMrHandle = (struct IbCastGinProxyMrHandle *)dstMhandle;

  void *srcPtr = (void *)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
  void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_WRITE;
  wr.send_flags              = IBV_SEND_SIGNALED;
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)dstPtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)srcPtr;  // Local buffer address
  sge.length = size;  // Size of the transfer
  sge.lkey = lkey;  // Local key

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  IbCastAddEvent(req, qp->devIndex);

  *request = req;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyIGet(void *ginCtx, int context, uint64_t remoteOffset, void *remoteMhandle,
                                 size_t size, uint64_t localOffset, void *localMhandle, uint32_t rank,
                                 void **request) {
  struct IbCastGinIbProxyCtx* ginProxyCtx = &((struct IbCastGinIbProxyCtx*)ginCtx)[context];

  struct IbCastGinProxyMrHandle *remoteMrHandle = (struct IbCastGinProxyMrHandle *)remoteMhandle;
  struct IbCastGinProxyMrHandle *localMrHandle = (struct IbCastGinProxyMrHandle *)localMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  void *remotePtr = (void *)(remoteMrHandle->base_vas[rank] + remoteOffset);
  void *localPtr = (void *)(localMrHandle->base_vas[ginProxyCtx->rank] + localOffset);
  uint32_t rkey = remoteMrHandle->rkeys[rank];
  uint32_t lkey = localMrHandle->mrHandle->mrs[0]->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_READ;
  wr.send_flags              = IBV_SEND_SIGNALED; // TODO: Potentially optimize this?
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)remotePtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)localPtr;
  sge.length = size;
  sge.lkey = lkey;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  IbCastAddEvent(req, qp->devIndex);

  *request = req;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyIPutSignal(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                      uint64_t signalOff, void *signalMhandle, uint64_t signalValue,
                                      uint32_t signalOp, void **request) {
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("IbCastGinIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct IbCastGinIbProxyCtx* ginProxyCtx = &((struct IbCastGinIbProxyCtx*)ginCtx)[context];

  struct IbCastGinProxyMrHandle *srcMrHandle = (struct IbCastGinProxyMrHandle *)srcMhandle;
  struct IbCastGinProxyMrHandle *dstMrHandle = (struct IbCastGinProxyMrHandle *)dstMhandle;
  struct IbCastGinProxyMrHandle *signalMrHandle = (struct IbCastGinProxyMrHandle *)signalMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];
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

  struct ibv_send_wr wr[2];
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge[2];
  memset(&sge, 0, sizeof(sge));

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  if (size > 0 && dstMrHandle) {
    void *srcPtr = (void *)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
    void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
    uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
    uint32_t rkey = dstMrHandle->rkeys[rank];

    // PUT
    wr[0].opcode                  = IBV_WR_RDMA_WRITE;
    wr[0].send_flags              = 0; // We only need the CQE from the signal
    wr[0].wr_id                   = req - comm->base.reqs;
    wr[0].next                    = &wr[1];
    wr[0].wr.rdma.remote_addr     = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey            = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;  // Local buffer address
    sge[0].length = size;  // Size of the transfer
    sge[0].lkey = lkey;  // Local key
  }

  void *signalPtr = (void *)(signalMrHandle->base_vas[rank] + signalOff);
  uint32_t signalRkey = signalMrHandle->rkeys[rank];

  // SIGNAL
  wr[1].opcode                  = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags              = IBV_SEND_SIGNALED;
  wr[1].wr_id                   = req - comm->base.reqs;  // used for matching completions with request
  wr[1].next                    = NULL;
  wr[1].wr.atomic.remote_addr   = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add   = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  wr[1].wr.atomic.rkey          = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  // Send the put and the signal in one go
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, size > 0 ? &wr[0] : &wr[1], &bad_wr));
  IbCastAddEvent(req, qp->devIndex);
  *request = req;
  return ncclSuccess;
}

ncclResult_t IbCastGinIbProxyTest(void* collComm, void *request, int *done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct IbCastGinIbProxyCtx* ginProxyCtx = (struct IbCastGinIbProxyCtx*)req->ginProxyCtx;
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
      const char* localGidStr = NULL, *remoteGidStr = NULL;
      if (req->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&commBase->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN+1];
      char *hcaName = devBase->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
          ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err, IbCastReqTypeStr[req->type],
          localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
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

ncclResult_t IbCastGinIbProxyIFlush(void *ginCtx, int context, void* mhandle, uint32_t rank, void **request) {
  struct IbCastGinIbProxyCtx* ginProxyCtx = &((struct IbCastGinIbProxyCtx*)ginCtx)[context];
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
  struct IbCastGinProxyMrHandle *ginMrHandle = (struct IbCastGinProxyMrHandle *)mhandle;
  struct ncclIbQp *qp = &comm->devs[0].gpuFlush.qp;

  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->ginProxyCtx = ginProxyCtx;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
 
  // The flush QP is a loopback (self-connected on the local recv comm), so the
  // RDMA READ target must be locally-registered memory. Use localRank, not rank
  // (the remote peer): rkeys[rank] is the remote's rkey, invalid in the local PD.
  int localRank = ginProxyCtx->rank;
  void *flushPtr = (void *)(ginMrHandle->base_vas[localRank]);
  wr.wr.rdma.remote_addr = (uint64_t)flushPtr;
  wr.wr.rdma.rkey = ginMrHandle->rkeys[localRank];
  wr.sg_list = &comm->devs[qp->devIndex].gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
  TIME_START(4);
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  TIME_STOP(4);

  IbCastAddEvent(req, qp->devIndex);

  TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);

  *request = req;
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
ncclGin_t IbCastGinIbProxy = {
  "GIN_IB_PROXY",
  IbCastGinIbProxyInit,
  IbCastDevices,
  IbCastGinIbProxyGetProperties,
  IbCastListen,
  IbCastGinIbProxyConnect,
  IbCastGinIbProxyCreateContext,
  IbCastGinIbProxyRegMrSym,
  IbCastGinIbProxyRegMrSymDmaBuf,
  IbCastGinIbProxyDeregMrSym,
  IbCastGinIbProxyDestroyContext,
  IbCastGinIbCloseColl,
  IbCastCloseListen,
  IbCastGinIbProxyIPut,
  IbCastGinIbProxyIPutSignal,
  IbCastGinIbProxyIGet,
  IbCastGinIbProxyIFlush,
  IbCastGinIbProxyTest,
  NULL,
  NULL,
  IbCastGinIbFinalize
};
