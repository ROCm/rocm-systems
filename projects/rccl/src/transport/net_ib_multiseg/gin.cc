/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common_mseg.h"

#include "gin/gin_host.h"
#include "gin_mseg.h"
#include "utils.h"

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

// Check GDR support for GIN. This is run at init, so we don't know yet whether the GPU will support DMA-BUF.
static ncclResult_t IbMsegGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
  bool peerMemSupport = gdaki ? IbMsegPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
                                IbMsegGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  if (IbMsegDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  *gdrSupport = false;
  INFO(NCCL_NET, "Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclSuccess;
}

// Check the current GPU supports GDR for GIN. This is run during connect().
static ncclResult_t IbMsegGinIbGdrGpuSupport(bool gdaki) {
  bool peerMemSupport = gdaki ? IbMsegPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
                                IbMsegGdrSupport() == ncclSuccess;
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
  if (IbMsegDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  WARN("Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclInvalidUsage;
#endif
}

extern int64_t ncclParamGinType();
extern int64_t ncclParamGinIbTc();
USE_NCCL_PARAM(ncclParamIbTc, int32_t);

// [RCCL] GDAKI (driver-mode GIN over MLX5) is NVIDIA-only: its device kernels and
// gin_host_gdaki.cc are not built on ROCm (see src/CMakeLists.txt), so guard every
// GDAKI symbol out on HIP. The Proxy path below becomes the RMA_IB_PROXY backend,
// which is what ROCm uses (GIN proxy dispatches over the RMA backend).
#if !defined(__HIP_PLATFORM_AMD__)
static std::mutex IbMsegGinIbGdakiLockMutex;
static int IbMsegGinIbGdakiNDevs = -1;
int IbMsegGinIbGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t IbMsegGinIbGdakiInitOnce() {
  std::lock_guard<std::mutex> lock(IbMsegGinIbGdakiLockMutex);
  if (IbMsegGinIbGdakiNDevs == -1) {
    int ndevs = 0;
    int64_t ginType = ncclParamGinType();
    if (ginType != -1 && ginType != NCCL_GIN_TYPE_GDAKI) {
      IbMsegGinIbGdakiNDevs = 0;
      return ncclSuccess;
    }
    for (int i = 0; i < IbMsegNDevs; i++) {
      if (IbMsegDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        IbMsegGinIbGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    IbMsegGinIbGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}
#endif // !defined(__HIP_PLATFORM_AMD__)

// True iff device `dev`'s NICs are all MLX5. Returns false for any non-MLX5 or out-of-range device.
// net_ib-internal: the only caller is the IB RMA proxy's getRmaProperties below.
static bool IbMsegNetDevIsMlx5(int dev) {
  if (dev < 0 || dev >= IbMsegNMergedDevs) return false;
  struct IbMsegMergedDev* mDev = &IbMsegMergedDevs[dev];
  if (mDev->vProps.ndevs <= 0) return false;
  for (int i = 0; i < mDev->vProps.ndevs; i++) {
    int phys = mDev->vProps.devs[i];
    if (phys < 0 || phys >= IbMsegNDevs) return false;
    if (IbMsegDevs[phys].ibProvider != IB_PROVIDER_MLX5) return false;
  }
  return true;
}

// Initlialize GDAKI or PROXY backend. ginType can force a particular backend.
// If provided, overwrite ginIb with the backend (generic ginIb case).
ncclResult_t IbMsegGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction, int type) {
  NCCLCHECK(IbMsegInitDevices(logFunction, nullptr));
  if (IbMsegNDevs == 0) return ncclInternalError; // Caught in plugin init code, not propagated to user.

#if !defined(__HIP_PLATFORM_AMD__)
  if (type == NCCL_GIN_TYPE_GDAKI) {
    NCCLCHECK(IbMsegGinIbGdakiInitOnce());
    if (IbMsegGinIbGdakiNDevs == 0) return ncclInternalError;
  }
#endif // !defined(__HIP_PLATFORM_AMD__)

  bool gdrSupport;
  NCCLCHECK(IbMsegGinIbGdrSupport(&gdrSupport, type == NCCL_GIN_TYPE_GDAKI));
  if (!gdrSupport) return ncclInternalError;

  struct IbMsegNetCtx* netCtx = nullptr;
  NCCLCHECK(ncclCalloc(&netCtx, 1));
  netCtx->config.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCtx;
  return ncclSuccess;
}

ncclResult_t IbMsegGinIbFinalize(void* ctx) {
  free(ctx);
  return IbMsegFinalizeDevices();
}

static inline bool IbMsegGinIbAborted(volatile uint32_t* abortFlag) {
  return abortFlag && COMPILER_ATOMIC_LOAD(abortFlag, std::memory_order_acquire) != 0;
}

static ncclResult_t IbMsegGinIbAllGather(struct IbMsegGinIbCollComm* cComm, void* srcBuf, void* recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int sendDone, recvDone;
  int speer;
  int rpeer;
  int step = 0;
  bool aborted = false;
  void* rbuf;
  int tag;

  NCCLCHECKGOTO(ncclNetIbMseg.regMr(cComm->recvComm, recvBuf, cComm->nranks * len, NCCL_PTR_HOST, &rMhandle), status, out);
  NCCLCHECKGOTO(ncclNetIbMseg.regMr(cComm->sendComm, recvBuf, cComm->nranks * len, NCCL_PTR_HOST, &sMhandle), status, out);

  speer = cComm->rank;
  memcpy((void*)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (step = 0; step < cComm->nranks - 1; step++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    rbuf = (void*)((uintptr_t)recvBuf + rpeer * len);
    tag = NCCL_GIN_IB_ALLGATHER_TAG;
    sendDone = recvDone = 0;

    // post and complete both directions in a single while loop until both send and recv are done
    while (!sendDone || !recvDone) {
      // 1. post send & recv
      if (!srequest && !sendDone) {
        NCCLCHECKGOTO(ncclNetIbMseg.isend(cComm->sendComm, (void*)((uintptr_t)recvBuf + speer * len), len, tag, sMhandle,
                                      NULL, &srequest),
                      status, out);
      }
      if (!rrequest && !recvDone) {
        NCCLCHECKGOTO(ncclNetIbMseg.irecv(cComm->recvComm, 1, &rbuf, &len, &tag, &rMhandle, NULL, &rrequest), status, out);
      }

      // 2. poll completion
      if (rrequest) {
        recvDone = 0;
        NCCLCHECKGOTO(ncclNetIbMseg.test(rrequest, &recvDone, NULL), status, out);
        if (recvDone) {
          rrequest = NULL;
        }
      }
      if (srequest) {
        sendDone = 0;
        NCCLCHECKGOTO(ncclNetIbMseg.test(srequest, &sendDone, NULL), status, out);
        if (sendDone) {
          srequest = NULL;
        }
      }
      if (IbMsegGinIbAborted(cComm->abortFlag)) {
        aborted = true;
        status = ncclInternalError;
        goto out;
      }
    }
    speer = rpeer;
  }

out:
  if (aborted) {
    INFO(NCCL_NET, "NET/IB/GIN: allgather aborted on rank %d/%d at step %d/%d", cComm->rank, cComm->nranks, step,
         cComm->nranks - 1);
  } else if (status != ncclSuccess) {
    WARN("NET/IB/GIN: allgather failed (res=%d) on rank %d/%d at step %d/%d", status, cComm->rank, cComm->nranks, step,
         cComm->nranks - 1);
  }

  if (rMhandle) ncclNetIbMseg.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle) ncclNetIbMseg.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t IbMsegGinIbAllToAll(struct IbMsegGinIbCollComm* cComm, void* src_buf, void* recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;

  void* tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void**)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void*)((uintptr_t)recv_buf + i * len),
           (void*)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len), len);
  }

out:
  free(tmp_buf);

  return status;
}

ncclResult_t IbMsegGinIbP2PBarrier(struct IbMsegGinIbCollComm* cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int* dummy;
  NCCLCHECK(ncclIbMalloc((void**)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(IbMsegGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t IbMsegGinIbConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm,
                              volatile uint32_t* abortFlag) {
  struct IbMsegListenComm* lComm = (struct IbMsegListenComm*)listenComm;
  struct IbMsegGinIbCollComm* cCommArray = nullptr;
  ncclResult_t ret = ncclSuccess;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void**)&cCommArray, sizeof(*cCommArray)));

  struct IbMsegGinIbCollComm* cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;
  cComm->abortFlag = abortFlag;

  next = (cComm->rank + 1) % nranks;
  do {
    if (cComm->sendComm == NULL) {
      NCCLCHECKGOTO(IbMsegConnectImpl(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL, /*nQpsPerDev*/ 1,
                                      ncclParamGinIbTc() != NCCL_PARAM_VAL_AUTO ? ncclParamGinIbTc() : ncclParamIbTc()),
                    ret, fail);
    }
    if (cComm->recvComm == NULL) {
      NCCLCHECKGOTO(IbMsegAcceptImpl(lComm, &cComm->recvComm, NULL, /*nQpsPerDev*/ 1), ret, fail);
    }
    if (IbMsegGinIbAborted(abortFlag)) {
      INFO(NCCL_NET, "NET/IB/GIN: connect aborted on rank %d/%d", rank, nranks);
      ret = ncclInternalError;
      goto fail;
    }
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t (*)(int dev, void* props))IbMsegGetProperties;
  cComm->allGather = IbMsegGinIbAllGather;
  cComm->allToAll = IbMsegGinIbAllToAll;
  cComm->getGidIndex = IbMsegGetGidIndex;
  cComm->dev = lComm->dev;

  cComm->ib.context = IbMsegDevs[cComm->dev].context;
  cComm->ib.pd = IbMsegDevs[cComm->dev].pd;

  *collComm = cCommArray;
  return ncclSuccess;

fail:
  if (cComm->recvComm) ncclNetIbMseg.closeRecv(cComm->recvComm);
  if (cComm->sendComm) ncclNetIbMseg.closeSend(cComm->sendComm);
  free(cCommArray);
  return ret;
}

ncclResult_t IbMsegGinIbCloseColl(void* collComm) {
  struct IbMsegGinIbCollComm* cCommArray = (struct IbMsegGinIbCollComm*)collComm;
  if (!cCommArray) return ncclSuccess;

  struct IbMsegGinIbCollComm* cComm = cCommArray;
  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIbMseg.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIbMseg.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cCommArray);
  return ncclSuccess;
}

#if !defined(__HIP_PLATFORM_AMD__)
#include "gdaki/gin_host_gdaki.h"

ncclResult_t IbMsegGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return IbMsegGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI);
}

ncclResult_t IbMsegGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(IbMsegGinIbGdakiLockMutex);
  *ndev = IbMsegGinIbGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t IbMsegGinIbGdakiGetGinProperties(ncclGinProperties_t* ginProps) {
  ginProps->supportsStrongSignals = true;
  ginProps->supportsVASignals = true;
  ginProps->barrierOptions = NCCL_GIN_BARRIER_DEFAULT;
  return ncclSuccess;
}

ncclResult_t IbMsegGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(IbMsegGinIbGdakiLockMutex);
  if (dev >= IbMsegGinIbGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created", dev,
         IbMsegGinIbGdakiNDevs);
    return ncclInvalidUsage;
  }
  NCCLCHECK(IbMsegGetPhysProperties(IbMsegGinIbGdakiDevIndexes[dev], props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t IbMsegGinIbGdakiListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(IbMsegGinIbGdakiLockMutex);
  return ncclNetIbMseg.listen(ctx, IbMsegGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

static ncclResult_t IbMsegGinIbGdakiValidateDeviceSupport(struct IbMsegGinIbCollComm* cComm) {
  ncclResult_t status = ncclSuccess;
  int* rankSupported = nullptr;
  int localSupported = 1;
  int allSupported = 1;

  if (IbMsegGinIbGdrGpuSupport(true) != ncclSuccess) localSupported = 0;
  if (ncclGinGdakiCheckDeviceSupport(cComm) != ncclSuccess) localSupported = 0;

  NCCLCHECKGOTO(ncclIbMalloc((void**)&rankSupported, cComm->nranks * sizeof(*rankSupported)), status, out);
  NCCLCHECKGOTO(cComm->allGather(cComm, &localSupported, rankSupported, sizeof(localSupported)), status, out);

  for (int r = 0; r < cComm->nranks; r++) {
    allSupported &= rankSupported[r];
  }

  if (!allSupported) {
    cComm->gdakiUnsupported = true;
  }

out:
  if (rankSupported) free(rankSupported);
  return status;
}

ncclResult_t IbMsegGinIbGdakiConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm,
                                   volatile uint32_t* abortFlag) {
  NCCLCHECK(IbMsegGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm, abortFlag));

  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)*collComm;
  cComm->getProperties = (ncclResult_t (*)(int dev, void* props))IbMsegGinIbGdakiGetProperties;
  NCCLCHECK(IbMsegGinIbGdakiValidateDeviceSupport(cComm));
  return ncclSuccess;
}

ncclResult_t IbMsegGinIbGdakiCreateContext(void* collComm, ncclGinConfig_t* config, void** ginCtx,
                                         ncclNetDeviceHandle_t** devHandle) {
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;

  if (cComm->gdakiUnsupported) return ncclInvalidUsage;

  if (ncclParamGinIbTc() != NCCL_PARAM_VAL_AUTO) config->trafficClass = ncclParamGinIbTc();
  else if (ncclParamIbTc() != NCCL_PARAM_VAL_AUTO) config->trafficClass = ncclParamIbTc();

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t IbMsegGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags,
                                    void** mhandle, void** ginHandle) {
  return ncclGinGdakiRegMrSym((struct IbMsegGinIbCollComm*)collComm, data, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t IbMsegGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct IbMsegGinIbCollComm*)collComm, mhandle);
}

ncclResult_t IbMsegGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t IbMsegGinIbGdakiProgress(void* collComm) {
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t IbMsegGinIbGdakiQueryLastError(void* ginCtx, bool* hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t IbMsegGinIbGdaki = {"GIN_IB_GDAKI",
                            IbMsegGinIbGdakiInit,
                            IbMsegGinIbGdakiDevices,
                            IbMsegGinIbGdakiGetGinProperties,
                            IbMsegGinIbGdakiGetProperties,
                            IbMsegGinIbGdakiListen,
                            IbMsegGinIbGdakiConnect,
                            IbMsegGinIbGdakiCreateContext,
                            IbMsegGinIbGdakiRegMrSym,
                            NULL, // regMrSymDmaBuf
                            IbMsegGinIbGdakiDeregMrSym,
                            IbMsegGinIbGdakiDestroyContext,
                            IbMsegGinIbCloseColl,
                            IbMsegCloseListen,
                            IbMsegGinIbGdakiProgress,
                            IbMsegGinIbGdakiQueryLastError,
                            IbMsegGinIbFinalize};
#endif // !defined(__HIP_PLATFORM_AMD__)

struct IbMsegRmaIbProxyMrHandle {
  struct IbMsegMrHandle* mrHandle;
  uintptr_t* base_vas;
  uint32_t* rkeys;
};

ncclResult_t IbMsegRmaIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return IbMsegGinIbInitType(ctx, commId, logFunction, ncclParamGinType());
}

ncclResult_t IbMsegRmaIbProxyGetRmaProperties(void* collComm, ncclRmaProperties_t* rmaProps) {
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // Relaxed-ordering writes can land after a later signal.
  (void)cComm;
  rmaProps->flushesAllPutsOnAnySignal = false;
#else
  rmaProps->flushesAllPutsOnAnySignal = IbMsegNetDevIsMlx5(cComm->dev);
#endif
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetIbMseg.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm,
                                   volatile uint32_t* abortFlag) {
  // Check the current GPU supports GDR
  NCCLCHECK(IbMsegGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(IbMsegGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm, abortFlag));

  return ncclSuccess;
}

#define NCCL_RMA_MAX_IB_WRS_PER_OP 2  // put+Signal (2 wrs)
#define NCCL_RMA_IB_WR_BATCHSIZE 64
extern int64_t ncclParamRmaIbWrBatchsize();

struct IbMsegRmaIbProxyWrBatch {
  struct ibv_qp* qp;
  int nWrs;
  struct ibv_send_wr wrs[NCCL_RMA_IB_WR_BATCHSIZE];
  struct ibv_sge sges[NCCL_RMA_IB_WR_BATCHSIZE];
};

struct IbMsegRmaIbProxyCtx {
  void** fullRecvComm;
  void** fullSendComm;
  int rank, nranks;
  int nContexts;
  int wrBatchsize;
  struct IbMsegRmaIbProxyWrBatch batch;
};

// Ring the doorbell for the pending chain, if any.
static ncclResult_t IbMsegRmaIbProxyPostBatch(struct IbMsegRmaIbProxyCtx* ctx) {
  struct IbMsegRmaIbProxyWrBatch* batch = &ctx->batch;
  if (batch->nWrs == 0) return ncclSuccess;  // no wrs to post

  // make sure the last wr is signaled and NULL-terminated
  batch->wrs[batch->nWrs - 1].send_flags |= IBV_SEND_SIGNALED;
  batch->wrs[batch->nWrs - 1].next = NULL;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(batch->qp, &batch->wrs[0], &bad_wr));

  // reset the batch
  batch->nWrs = 0;
  batch->qp = NULL;
  return ncclSuccess;
}

// allocate a slot (memset to 0)
static ncclResult_t IbMsegRmaIbProxyReserveWrs(struct IbMsegRmaIbProxyCtx* ctx, struct ibv_qp* qp, int nWrs,
                                             struct ibv_send_wr** wrs, struct ibv_sge** sges) {
  struct IbMsegRmaIbProxyWrBatch* batch = &ctx->batch;
  if (batch->nWrs && (batch->qp != qp || batch->nWrs + nWrs > ctx->wrBatchsize)) {
    NCCLCHECK(IbMsegRmaIbProxyPostBatch(ctx));
  }
  batch->qp = qp;
  *wrs = batch->wrs + batch->nWrs;
  *sges = batch->sges + batch->nWrs;
  memset(*wrs, 0, nWrs * sizeof(struct ibv_send_wr));
  memset(*sges, 0, nWrs * sizeof(struct ibv_sge));
  return ncclSuccess;
}

// fill the slot (link the wrs)
static ncclResult_t IbMsegRmaIbProxyCommitWrs(struct IbMsegRmaIbProxyCtx* ctx, int nWrs, bool aggregate) {
  struct IbMsegRmaIbProxyWrBatch* batch = &ctx->batch;
  for (int i = 0; i < nWrs; i++) {
    batch->wrs[batch->nWrs + i].next = batch->wrs + batch->nWrs + i + 1;
  }
  batch->nWrs += nWrs;

  // in case we don't have enough space to fit the next put+signal
  if (!aggregate || batch->nWrs + NCCL_RMA_MAX_IB_WRS_PER_OP > ctx->wrBatchsize) {
    NCCLCHECK(IbMsegRmaIbProxyPostBatch(ctx));
  }
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyCreateContext(void* collComm, ncclRmaConfig_t* config, void** rmaCtx) {
  ncclResult_t ret = ncclSuccess;
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;
  // Make sure all QP we create use the provided traffic class.
  IbMsegSetTrafficClass(cComm->ctx, config->trafficClass);

  int nranks;
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = NULL;
  *rmaCtx = NULL;
  NCCLCHECK(ncclCalloc(&rmaProxyCtx, config->nContexts));
  rmaProxyCtx[0].nContexts = config->nContexts;
  rmaProxyCtx[0].nranks = nranks = cComm->nranks;

  int64_t wrBatchsize = ncclParamRmaIbWrBatchsize();
  if (wrBatchsize < NCCL_RMA_MAX_IB_WRS_PER_OP) {
    wrBatchsize = NCCL_RMA_MAX_IB_WRS_PER_OP;
    INFO(NCCL_INIT, "RMA: NCCL_RMA_IB_WR_BATCHSIZE is less than (%d), setting to %ld", NCCL_RMA_MAX_IB_WRS_PER_OP,
         wrBatchsize);
  }
  if (wrBatchsize > NCCL_RMA_IB_WR_BATCHSIZE) {
    wrBatchsize = NCCL_RMA_IB_WR_BATCHSIZE;
    INFO(NCCL_INIT, "RMA: NCCL_RMA_IB_WR_BATCHSIZE is greater than (%d), setting to %ld", NCCL_RMA_IB_WR_BATCHSIZE,
         wrBatchsize);
  }

  void* lComm = NULL;
  char *handle = NULL, *handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE * cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE * cComm->rank;

  NCCLCHECKGOTO(ncclNetIbMseg.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, end);
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c = 0; c < config->nContexts; c++) {
    struct IbMsegRmaIbProxyCtx* gc = rmaProxyCtx + c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void*) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void*) * nranks), ret, end);
    gc->rank = cComm->rank;
    gc->wrBatchsize = (int)wrBatchsize;

    // iterate through all ranks and barrier on each round to handle non-uniform peer counts across ranks.
    for (int i = 0; i < nranks; i++) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;

      // Always connect to self. The self-QP in self-recvComm is used in iFlush.
      bool wantConnect = (connectPeer == cComm->rank);
      bool wantAccept = (acceptPeer == cComm->rank);

      if (config->peerArrayCount > 0) {
        wantConnect |= (bool)bsearch(&connectPeer, config->peerArray, config->peerArrayCount, sizeof(int), compareInts);
        wantAccept |= (bool)bsearch(&acceptPeer, config->peerArray, config->peerArrayCount, sizeof(int), compareInts);
      }

      do {
        if (wantConnect && gc->fullSendComm[connectPeer] == NULL) {
          NCCLCHECKGOTO(IbMsegConnectImpl(cComm->ctx, cComm->dev, handles + NCCL_NET_HANDLE_MAXSIZE * connectPeer,
                                          &gc->fullSendComm[connectPeer], NULL, /*nQpsPerDev*/ 1,
                                          ncclParamGinIbTc() != NCCL_PARAM_VAL_AUTO ? ncclParamGinIbTc() :
                                                                                      ncclParamIbTc()),
                        ret, end);
        }
        if (wantAccept && gc->fullRecvComm[acceptPeer] == NULL) {
          NCCLCHECKGOTO(IbMsegAcceptImpl(lComm, &gc->fullRecvComm[acceptPeer], NULL, /*nQpsPerDev*/ 1), ret, end);
        }
        if (IbMsegGinIbAborted(cComm->abortFlag)) {
          INFO(NCCL_NET, "NET/IB/GIN: proxy context connect aborted on rank %d/%d (peers %d/%d)", cComm->rank, nranks,
               connectPeer, acceptPeer);
          ret = ncclInternalError;
          goto end;
        }
      } while ((wantConnect && gc->fullSendComm[connectPeer] == NULL) ||
               (wantAccept && gc->fullRecvComm[acceptPeer] == NULL));
      NCCLCHECKGOTO(IbMsegGinIbP2PBarrier(cComm), ret, end);
    }
  }

end:
  free(handles);
  if (lComm) ncclNetIbMseg.closeListen(lComm);
  if (ret != ncclSuccess) free(rmaProxyCtx);
  else *rmaCtx = rmaProxyCtx;
  return ret;
}

ncclResult_t IbMsegRmaIbProxyDestroyContext(void* rmaCtx) {
  struct IbMsegRmaIbProxyCtx* gc = (struct IbMsegRmaIbProxyCtx*)rmaCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;
  for (int c = 0; c < nContexts; c++) {
    NCCLCHECK(IbMsegRmaIbProxyPostBatch(&gc[c]));
    if (gc[c].fullRecvComm) {
      for (int i = 0; i < nranks; i++) {
        NCCLCHECK(ncclNetIbMseg.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i = 0; i < nranks; i++) {
        NCCLCHECK(ncclNetIbMseg.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }
  }
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                          uint64_t mr_flags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;
  struct IbMsegRmaIbProxyMrHandle* rmaMrHandle = NULL;
  NCCLCHECKGOTO(ncclCalloc(&rmaMrHandle, 1), ret, fail);

  NCCLCHECKGOTONOWARN(IbMsegRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags,
                                                (void**)&rmaMrHandle->mrHandle),
                      ret, fail, NCCL_NET);

  NCCLCHECKGOTO(ncclCalloc(&rmaMrHandle->base_vas, cComm->nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&rmaMrHandle->rkeys, cComm->nranks), ret, fail);

  NCCLCHECKGOTO(cComm->allGather(cComm, &data, rmaMrHandle->base_vas, sizeof(uintptr_t)), ret, fail);
  NCCLCHECKGOTO(cComm->allGather(cComm, &rmaMrHandle->mrHandle->mrs[0]->rkey, rmaMrHandle->rkeys, sizeof(uint32_t)),
                ret, fail);

  *mhandle = rmaMrHandle;

  return ncclSuccess;
fail:
  if (rmaMrHandle) {
    if (rmaMrHandle->mrHandle) ncclNetIbMseg.deregMr(cComm->recvComm, rmaMrHandle->mrHandle);
    free(rmaMrHandle->base_vas);
    free(rmaMrHandle->rkeys);
    free(rmaMrHandle);
  }
  return ret;
}

ncclResult_t IbMsegRmaIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags,
                                    void** mhandle) {
  return IbMsegRmaIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle);
}

ncclResult_t IbMsegRmaIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;
  struct IbMsegRmaIbProxyMrHandle* rmaMrHandle = (struct IbMsegRmaIbProxyMrHandle*)mhandle;

  NCCLCHECK(ncclNetIbMseg.deregMr(cComm->recvComm, rmaMrHandle->mrHandle));
  free(rmaMrHandle->base_vas);
  free(rmaMrHandle->rkeys);
  free(rmaMrHandle);
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

static ncclResult_t IbMsegRmaIbProxyGetSendComm(struct IbMsegRmaIbProxyCtx* rmaProxyCtx, int rank,
                                              struct IbMsegSendComm** commPtr) {
  *commPtr = (struct IbMsegSendComm*)rmaProxyCtx->fullSendComm[rank];
  if (*commPtr == NULL) {
    WARN("RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
static ncclResult_t IbMsegRmaIbProxyGetRecvComm(struct IbMsegRmaIbProxyCtx* rmaProxyCtx, int rank,
                                              struct IbMsegRecvComm** commPtr) {
  *commPtr = (struct IbMsegRecvComm*)rmaProxyCtx->fullRecvComm[rank];
  if (*commPtr == NULL) {
    WARN("RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request) {
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];

  struct IbMsegRmaIbProxyMrHandle* srcMrHandle = (struct IbMsegRmaIbProxyMrHandle*)srcMhandle;
  struct IbMsegRmaIbProxyMrHandle* dstMrHandle = (struct IbMsegRmaIbProxyMrHandle*)dstMhandle;

  void* srcPtr = (void*)(srcMrHandle->base_vas[rmaProxyCtx->rank] + srcOff);
  void* dstPtr = (void*)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct IbMsegSendComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct IbMsegQp* qp = &comm->base.qps[0];

  struct IbMsegRequest* req;
  NCCLCHECK(IbMsegGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr* wr;
  struct ibv_sge* sge;

  // reserve empty wr and sge
  NCCLCHECK(IbMsegRmaIbProxyReserveWrs(rmaProxyCtx, qp->qp, 1, &wr, &sge));

  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->wr_id = req - comm->base.reqs;
  wr->next = NULL;
  wr->wr.rdma.remote_addr = (uint64_t)dstPtr;
  wr->wr.rdma.rkey = rkey;
  wr->sg_list = sge;
  wr->num_sge = 1;

  sge->addr = (uintptr_t)srcPtr;  // Local buffer address
  sge->length = size;  // Size of the transfer
  sge->lkey = lkey;  // Local key

  NCCLCHECK(IbMsegRmaIbProxyCommitWrs(rmaProxyCtx, 1, optFlags & ncclRmaOptFlagsAggregateRequests));
  IbMsegAddEvent(req, qp->devIndex);

  *request = req;
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyIGet(void* rmaCtx, int context, uint64_t remoteOffset, void* remoteMhandle, size_t size,
                                uint64_t localOffset, void* localMhandle, uint32_t rank, uint32_t optFlags,
                                void** request) {
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];

  struct IbMsegRmaIbProxyMrHandle* remoteMrHandle = (struct IbMsegRmaIbProxyMrHandle*)remoteMhandle;
  struct IbMsegRmaIbProxyMrHandle* localMrHandle = (struct IbMsegRmaIbProxyMrHandle*)localMhandle;

  struct IbMsegSendComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct IbMsegQp* qp = &comm->base.qps[0];

  struct IbMsegRequest* req;
  NCCLCHECK(IbMsegGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  void* remotePtr = (void*)(remoteMrHandle->base_vas[rank] + remoteOffset);
  void* localPtr = (void*)(localMrHandle->base_vas[rmaProxyCtx->rank] + localOffset);
  uint32_t rkey = remoteMrHandle->rkeys[rank];
  uint32_t lkey = localMrHandle->mrHandle->mrs[0]->lkey;

  struct ibv_send_wr* wr;
  struct ibv_sge* sge;
  NCCLCHECK(IbMsegRmaIbProxyReserveWrs(rmaProxyCtx, qp->qp, 1, &wr, &sge));

  wr->opcode = IBV_WR_RDMA_READ;
  wr->wr_id = req - comm->base.reqs;
  wr->next = NULL;
  wr->wr.rdma.remote_addr = (uint64_t)remotePtr;
  wr->wr.rdma.rkey = rkey;
  wr->sg_list = sge;
  wr->num_sge = 1;

  sge->addr = (uintptr_t)localPtr;
  sge->length = size;
  sge->lkey = lkey;

  NCCLCHECK(IbMsegRmaIbProxyCommitWrs(rmaProxyCtx, 1, optFlags & ncclRmaOptFlagsAggregateRequests));

  *request = req;
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                      uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                      void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                      uint32_t optFlags, void** request) {
  (void)isStrongSignal;
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("IbMsegRmaIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];

  struct IbMsegRmaIbProxyMrHandle* srcMrHandle = (struct IbMsegRmaIbProxyMrHandle*)srcMhandle;
  struct IbMsegRmaIbProxyMrHandle* dstMrHandle = (struct IbMsegRmaIbProxyMrHandle*)dstMhandle;
  struct IbMsegRmaIbProxyMrHandle* signalMrHandle = (struct IbMsegRmaIbProxyMrHandle*)signalMhandle;

  struct IbMsegSendComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct IbMsegQp* qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct IbMsegRequest* req;
  NCCLCHECK(IbMsegGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  bool hasPut = size > 0 && dstMrHandle;
  int nWrs = hasPut ? 2 : 1;
  struct ibv_send_wr* wr;
  struct ibv_sge* sge;
  NCCLCHECK(IbMsegRmaIbProxyReserveWrs(rmaProxyCtx, qp->qp, nWrs, &wr, &sge));
  // The signal is the last work request, whether or not a put precedes it.
  struct ibv_send_wr* signalWr = wr + nWrs - 1;
  struct ibv_sge* signalSge = sge + nWrs - 1;

  if (hasPut) {
    void* srcPtr = (void*)(srcMrHandle->base_vas[rmaProxyCtx->rank] + srcOff);
    void* dstPtr = (void*)(dstMrHandle->base_vas[rank] + dstOff);
    uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
    uint32_t rkey = dstMrHandle->rkeys[rank];

    // PUT
    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].send_flags = 0; // We only need the CQE from the signal
    wr[0].wr_id = req - comm->base.reqs;
    wr[0].next = signalWr;
    wr[0].wr.rdma.remote_addr = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;  // Local buffer address
    sge[0].length = size;  // Size of the transfer
    sge[0].lkey = lkey;  // Local key
  }

  void* signalPtr = (void*)(signalMrHandle->base_vas[rank] + signalOff);
  uint32_t signalRkey = signalMrHandle->rkeys[rank];

  // SIGNAL
  signalWr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  signalWr->wr_id = req - comm->base.reqs;  // used for matching completions with request
  signalWr->next = NULL;
  signalWr->wr.atomic.remote_addr = (uint64_t)signalPtr;
  signalWr->wr.atomic.compare_add = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  signalWr->wr.atomic.rkey = signalRkey;
  signalWr->sg_list = signalSge;
  signalWr->num_sge = 1;

  signalSge->addr = (uintptr_t)&comm->putSignalScratchpad;
  signalSge->length = sizeof(comm->putSignalScratchpad);
  signalSge->lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  // Send the put and the signal in one go
  NCCLCHECK(IbMsegRmaIbProxyCommitWrs(rmaProxyCtx, nWrs, optFlags & ncclRmaOptFlagsAggregateRequests));
  *request = req;
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyTest(void* collComm, void* request, int* done) {
  struct IbMsegRequest* req = (struct IbMsegRequest*)request;
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = (struct IbMsegRmaIbProxyCtx*)req->rmaProxyCtx;
  int rank = req->iput.rank;
  bool isFlush = req->type == NCCL_NET_IB_REQ_FLUSH;
  *done = 0;

  // The caller is already waiting on this op, so a deferred doorbell must not
  // hold it back.
  NCCLCHECK(IbMsegRmaIbProxyPostBatch(rmaProxyCtx));

  IbMsegNetCommBase* commBase;
  IbMsegNetCommDevBase* devBase;

  uint64_t* completedSeq;
  if (isFlush) {
    struct IbMsegRecvComm* comm = (struct IbMsegRecvComm*)rmaProxyCtx->fullRecvComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;
    completedSeq = &comm->ginSeq.completed;
  } else {
    struct IbMsegSendComm* comm = (struct IbMsegSendComm*)rmaProxyCtx->fullSendComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;
    completedSeq = &comm->ginSeq.completed;
  }

  if (req->id <= *completedSeq) {
    *done = 1;
    NCCLCHECK(IbMsegFreeRequest(req));
    return ncclSuccess;
  }

  int wrDone = 0;
  struct ibv_wc wc[4];
  NCCLCHECK(wrap_ibv_poll_cq(devBase->cq, 4, wc, &wrDone));
  for (int i = 0; i < wrDone; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      union ncclSocketAddress addr;
      ncclSocketGetAddr(req->sock, &addr);
      char localGidString[INET6_ADDRSTRLEN] = "";
      char remoteGidString[INET6_ADDRSTRLEN] = "";
      const char *localGidStr = NULL, *remoteGidStr = NULL;
      if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&commBase->remDevs[0].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN + 1];
      char* hcaName = devBase->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
           ncclSocketToString(&addr, line, sizeof(line)), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err,
           IbMsegReqTypeStr[req->type], localGidStr ? " localGid " : "", localGidString,
           remoteGidStr ? " remoteGids" : "", remoteGidString, hcaName);
      printIbWcStatusHint(wc[i].status);
      return ncclRemoteError;
    }

    struct IbMsegRequest* wcReq = commBase->reqs + wc[i].wr_id;

    if (wcReq->id > *completedSeq) *completedSeq = wcReq->id;
  }

    // quick check if we get completion during this poll
  if (req->id <= *completedSeq) {
    *done = 1;
    NCCLCHECK(IbMsegFreeRequest(req));
  }
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request) {
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];
  struct IbMsegRmaIbProxyMrHandle* rmaMrHandle = (struct IbMsegRmaIbProxyMrHandle*)mhandle;
  struct IbMsegRecvComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetRecvComm(rmaProxyCtx, rank, &comm));
  if (comm->flushEnabled == 0) {
    *request = NULL;
    return ncclSuccess;
  }

  // drain batch before posting the flush
  NCCLCHECK(IbMsegRmaIbProxyPostBatch(rmaProxyCtx));
  NCCLCHECK(IbMsegCreateFlushQp(comm));
  struct IbMsegQp* qp = &comm->devs[0].gpuFlush.qp;

  struct IbMsegRequest* req;
  NCCLCHECK(IbMsegGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->rmaProxyCtx = rmaProxyCtx;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;

  // The flush QP is a loopback (self-connected on the local recv comm), so the
  // RDMA READ target must be locally-registered memory. Use localRank, not rank
  // (the remote peer): rkeys[rank] is the remote's rkey, invalid in the local PD.
  int localRank = rmaProxyCtx->rank;
  void* flushPtr = (void*)(rmaMrHandle->base_vas[localRank]);
  wr.wr.rdma.remote_addr = (uint64_t)flushPtr;
  wr.wr.rdma.rkey = rmaMrHandle->rkeys[localRank];
  wr.sg_list = &comm->devs[qp->devIndex].gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base,
        wr.wr_id);
  TIME_START(4);
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  TIME_STOP(4);

  TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);

  *request = req;
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
ncclRma_t IbMsegRmaIbProxy = {"RMA_IB_PROXY",
                            IbMsegRmaIbProxyInit,
                            IbMsegDevices,
                            IbMsegRmaIbProxyGetRmaProperties,
                            IbMsegRmaIbProxyGetProperties,
                            IbMsegListen,
                            IbMsegRmaIbProxyConnect,
                            IbMsegRmaIbProxyCreateContext,
                            IbMsegRmaIbProxyRegMrSym,
                            IbMsegRmaIbProxyRegMrSymDmaBuf,
                            IbMsegRmaIbProxyDeregMrSym,
                            IbMsegRmaIbProxyDestroyContext,
                            IbMsegGinIbCloseColl,
                            IbMsegCloseListen,
                            IbMsegRmaIbProxyIPut,
                            IbMsegRmaIbProxyIPutSignal,
                            IbMsegRmaIbProxyIGet,
                            IbMsegRmaIbProxyIFlush,
                            IbMsegRmaIbProxyTest,
                            NULL,
                            NULL,
                            IbMsegGinIbFinalize};
