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
#include "ib_multiseg_mseg.h" // RCCL: multi-segment RMA splits.
#include "seg_wr_mseg.h"
#include "seg_rma_round_mseg.h"
#include "seg_reg_mseg.h"
#include "seg_export_mseg.h"
#include "seg_sym_mseg.h"
#include "rma.h"      // RCCL: IbMsegRmaIbProxyInit installs the multi-segment ops into IbMsegRmaIbProxy.

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

  free(cComm->regRecords); // RCCL: rcclRmaIbProxyConnectFinish allocated the per-rank registration records.
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
  NCCLCHECK(ncclNetIbMseg.listen(ctx, IbMsegGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm));
  rcclIbMarkGinListen(opaqueHandle, *listenComm); // RCCL: GIN comms skip the multi-segment layout.
  return ncclSuccess;
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

// RCCL: multi-segment MR handles have their own layout, so every MR and data op takes the multi-segment entry.
static void rcclRmaIbProxyInstallMultiSeg(ncclRma_t* rma) {
  rma->regMrSym = rcclRmaIbProxyRegMrSym;
  rma->regMrSymDmaBuf = rcclRmaIbProxyRegMrSymDmaBuf;
  rma->deregMrSym = rcclRmaIbProxyDeregMrSym;
  rma->iput = rcclRmaIbProxyIPut;
  rma->iputSignal = rcclRmaIbProxyIPutSignal;
  rma->iget = rcclRmaIbProxyIGet;
  rma->iflush = rcclRmaIbProxyIFlush;
}

ncclResult_t IbMsegRmaIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  static std::once_flag rcclMultiSegOnce; // RCCL: the switch is process-wide; install once, before any op runs.
  std::call_once(rcclMultiSegOnce, [] {
    if (rcclRmaMultiSegEnabled()) rcclRmaIbProxyInstallMultiSeg(&IbMsegRmaIbProxy);
  });
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

static ncclResult_t IbMsegRmaIbProxyListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  NCCLCHECK(IbMsegListen(ctx, dev, opaqueHandle, listenComm));
  rcclIbMarkGinListen(opaqueHandle, *listenComm); // RCCL: GIN comms skip the multi-segment layout.
  return ncclSuccess;
}

ncclResult_t IbMsegRmaIbProxyConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm,
                                   volatile uint32_t* abortFlag) {
  // Check the current GPU supports GDR
  NCCLCHECK(IbMsegGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(IbMsegGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm, abortFlag));
  NCCLCHECK(rcclRmaIbProxyConnectFinish(collComm, nranks));

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
  struct rcclRmaRoundQueues pending; // RCCL: multi-segment ops wider than one round, per peer.
};

// Ring the doorbell for the pending chain, if any.
static ncclResult_t IbMsegRmaIbProxyPostBatch(struct IbMsegRmaIbProxyCtx* ctx) {
  struct IbMsegRmaIbProxyWrBatch* batch = &ctx->batch;
  if (batch->nWrs == 0) return ncclSuccess;  // no wrs to post

  // make sure the last wr is signaled and NULL-terminated
  batch->wrs[batch->nWrs - 1].send_flags |= IBV_SEND_SIGNALED;
  batch->wrs[batch->nWrs - 1].next = NULL;

  struct ibv_send_wr* bad_wr;
  // RCCL: reset even on failure, so a later post cannot resend WRs the HCA accepted.
  ncclResult_t ret = wrap_ibv_post_send(batch->qp, &batch->wrs[0], &bad_wr);

  // reset the batch
  batch->nWrs = 0;
  batch->qp = NULL;
  return ret;
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
  rcclIbMarkGinListen(handle, lComm); // RCCL: per-peer RMA comms skip the multi-segment layout.
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c = 0; c < config->nContexts; c++) {
    struct IbMsegRmaIbProxyCtx* gc = rmaProxyCtx + c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void*) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void*) * nranks), ret, end);
    gc->rank = cComm->rank;
    gc->nranks = nranks; // RCCL: the per-peer round queues are nranks long.
    gc->wrBatchsize = (int)wrBatchsize;
    if (rcclRmaMultiSegEnabled()) {
      NCCLCHECKGOTO(ncclCalloc(&gc->pending.send, nranks), ret, end);
    }

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
  if (ret != ncclSuccess) {
    for (int c = 0; c < config->nContexts; c++) rcclRmaRoundQueuesFree(&rmaProxyCtx[c].pending, nranks);
    free(rmaProxyCtx);
  } else {
    *rmaCtx = rmaProxyCtx;
  }
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
    rcclRmaRoundQueuesFree(&gc[c].pending, nranks);
  }
  return ncclSuccess;
}

// RCCL: a NIC without peermem (bng_re) cannot register a GPU VA, so a single-segment GPU buffer passed with fd -1
// exports its own DMA-BUF fd. The ROCr export returns the offset of a suballocated plain buffer within its fd; it
// rejects VMM memory, which the cuMem export of the exact range covers. *fd stays -1 when neither applies.
static void rcclRmaExportSingleSegDmaBuf(void* data, size_t size, int type, int* fd, uint64_t* offset) {
  *fd = -1;
  *offset = 0;
  if (type != NCCL_PTR_CUDA || !ncclParamDmaBufEnable() || IbMsegGdrSupport() == ncclSuccess ||
      IbMsegDmaBufSupport(0) != ncclSuccess) {
    return;
  }
  const uintptr_t pageSize = (uintptr_t)sysconf(_SC_PAGESIZE);
  const uintptr_t aligned = (uintptr_t)data & ~(pageSize - 1);
  const size_t alignedSize = ((uintptr_t)data + size - aligned + pageSize - 1) & ~(pageSize - 1);
  if (hsa_amd_portable_export_dmabuf((const void*)aligned, alignedSize, fd, offset) == HSA_STATUS_SUCCESS &&
      *fd >= 0) {
    return;
  }
  *fd = -1;
  *offset = 0;
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  if (!ncclCuMemEnable() || cuMemGetHandleForAddressRange((void*)fd, (CUdeviceptr)aligned, alignedSize,
                                                          CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0) != CUDA_SUCCESS) {
    *fd = -1;
  }
#endif
}

// RCCL: registers one MR, with an fd of its own when fd is -1 and the NIC has no peermem.
static ncclResult_t rcclRmaRegSingleMr(void* recvComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mr_flags, void** mhandle) {
  int ownFd = -1;
  uint64_t ownOffset = 0;
  if (fd == -1) rcclRmaExportSingleSegDmaBuf(data, size, type, &ownFd, &ownOffset);
  ncclResult_t ret;
  NOWARN(ret = IbMsegRegMrDmaBufInternal(recvComm, data, size, type, ownFd >= 0 ? ownOffset : offset,
                                         ownFd >= 0 ? ownFd : fd, mr_flags, mhandle),
         NCCL_NET);
  if (ownFd >= 0) (void)close(ownFd);
  return ret;
}

ncclResult_t IbMsegRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                          uint64_t mr_flags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;
  struct IbMsegRmaIbProxyMrHandle* rmaMrHandle = NULL;
  NCCLCHECKGOTO(ncclCalloc(&rmaMrHandle, 1), ret, fail);

  NCCLCHECKGOTONOWARN(rcclRmaRegSingleMr(cComm->recvComm, data, size, type, offset, fd, mr_flags,
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

// RCCL: per-segment DMA-BUF MRs for the RMA proxy. IbMsegRmaIbProxyInit installs these when multi-segment is on.

struct rcclRmaIbProxySegMrHandle {
  uint32_t tag; // rcclRmaSegMrTag. The single-MR handle does not carry this tag.
  int nSegments;
  // Sized from the walked count. segOff has nSegments + 1 entries (segOff[0]==0, segOff[nSegments]==size) and
  // mrHandle holds one IbMsegMrHandle per segment. sym holds every rank's offsets, segment starts and per-port rkeys.
  size_t* segOff;
  void** mrHandle;
  struct rcclIbSegSymTable sym;
  // The segment a flush reads: the first device segment, or -1 when every segment is host memory.
  int flushSeg;
  // nSegments == 1: the single-MR layout over mrHandle[0] and sym.vas, with device-0 rkeys, for the single-MR ops.
  struct IbMsegRmaIbProxyMrHandle single;
};

static_assert(offsetof(struct rcclRmaIbProxySegMrHandle, tag) == 0, "segment-count tag is the first field");
static_assert(offsetof(struct rcclRmaIbProxySegMrHandle, nSegments) == sizeof(uint32_t),
              "nSegments follows the tag");

// RCCL: tests read nSegments from the handle. A single-MR handle has no tag, so the count is 0.
int rcclRmaIbProxyMrSegmentCount(void* mhandle) {
  return rcclRmaMrSegmentCount(mhandle);
}

ncclResult_t rcclRmaIbProxySegStats(void* rmaCtx, int context, int rank, struct rcclIbSegStats* out) {
  struct IbMsegRmaIbProxyCtx* gc = (struct IbMsegRmaIbProxyCtx*)rmaCtx;
  if (gc == NULL || out == NULL || context < 0 || context >= gc[0].nContexts || rank < 0 || rank >= gc[0].nranks)
    return ncclInvalidArgument;
  struct IbMsegSendComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetSendComm(gc + context, rank, &comm));
  rcclIbSegStatsRead(&comm->segStats, out);
  return ncclSuccess;
}

static inline uint32_t rcclRmaRemoteRkey(const struct rcclRmaIbProxySegMrHandle* h, int rank, int seg, int remDevIdx) {
  return (uint32_t)rcclIbSegSymKey(&h->sym, h->nSegments, NCCL_IB_MAX_DEVS_PER_NIC, rank, seg, remDevIdx);
}

static inline const size_t* rcclRmaRankOff(const struct rcclRmaIbProxySegMrHandle* h, int rank) {
  return rcclRmaPeerSegOff(h->sym.rankOff, h->segOff, rank, h->nSegments);
}

static inline int rcclRmaSegOf(const struct rcclRmaIbProxySegMrHandle* h, int rank, uint64_t off) {
  return rcclRmaSegIndexOf(rcclRmaRankOff(h, rank), h->nSegments, off);
}

static inline bool rcclRmaRangeOk(const struct rcclRmaIbProxySegMrHandle* h, int rank, uint64_t off, size_t size) {
  const size_t* offs = rcclRmaRankOff(h, rank);
  uint64_t bytes = offs[h->nSegments];
  return off <= bytes && (uint64_t)size <= bytes - off;
}

// RCCL: keys and per-rank bases stay here; the slice loop is rcclIbSegAppendWrs.
static ncclResult_t rcclRmaBuildSegmentedWrs(struct ibv_send_wr* wr, struct ibv_sge* sge, int maxWr, int* nWr,
                                             enum ibv_wr_opcode opcode, const struct IbMsegQp* qp,
                                             struct rcclRmaIbProxySegMrHandle* localH, int localRank, uint64_t localOff,
                                             struct rcclRmaIbProxySegMrHandle* remoteH, int remoteRank,
                                             uint64_t remoteOff, size_t size, uint64_t* covered = NULL) {
  const size_t* rOffs = rcclRmaRankOff(remoteH, remoteRank);
  const int nRemote = remoteH->nSegments;
  const int nLocal = localH->nSegments;
  if (nRemote < 1 || nLocal < 1) {
    WARN("NET/IB/RMA: transfer of %zu bytes has segment count local %d remote %d", size, nLocal, nRemote);
    return ncclInternalError;
  }
  // RCCL: a window over the segments this transfer touches. Each touched segment costs at least one WR, so the
  // key arrays are bounded by the WR budget and not by the segment count.
  const uint64_t last = size == 0 ? 0 : size - 1;
  const int r0 = rcclRmaSegIndexOf(rOffs, nRemote, remoteOff);
  int nr = rcclRmaSegIndexOf(rOffs, nRemote, remoteOff + last) - r0 + 1;
  const size_t* lOffs = rcclRmaRankOff(localH, localRank);
  const int l0 = rcclRmaSegIndexOf(lOffs, nLocal, localOff);
  int nl = rcclRmaSegIndexOf(lOffs, nLocal, localOff + last) - l0 + 1;
  if (covered != NULL) rcclRmaRoundClampWindow(maxWr, &nr, &nl);
  if (nr > maxWr || nl > maxWr || maxWr > RCCL_RMA_MAX_DATA_WRS) {
    WARN("NET/IB/RMA: transfer of %zu bytes touches %d local and %d remote segments, more than %d WRs", size, nl, nr,
         maxWr);
    return ncclInternalError;
  }
  uint32_t rkeys[RCCL_RMA_MAX_DATA_WRS];
  for (int i = 0; i < nr; i++) rkeys[i] = rcclRmaRemoteRkey(remoteH, remoteRank, r0 + i, qp->remDevIdx);
  uint32_t lkeys[RCCL_RMA_MAX_DATA_WRS];
  for (int i = 0; i < nl; i++) {
    struct IbMsegMrHandle* lh = (struct IbMsegMrHandle*)localH->mrHandle[l0 + i];
    struct ibv_mr* lmr = lh == NULL ? NULL : lh->mrs[qp->devIndex];
    if (lmr == NULL) {
      WARN("NET/IB/RMA: no local MR for segment %d device %d", l0 + i, qp->devIndex);
      return ncclInternalError;
    }
    lkeys[i] = lmr->lkey;
  }
  // Registration walks the buffer in order, so segment s starts at base + segOff[s]; a window keeps that base.
  struct rcclIbSegTable remote = {(uint64_t)rcclIbSegSymVa(&remoteH->sym, nRemote, remoteRank, 0), (uint32_t)nr,
                                  (const uint64_t*)(rOffs + r0)};
  struct rcclIbSegTable local = {(uint64_t)rcclIbSegSymVa(&localH->sym, nLocal, localRank, 0), (uint32_t)nl,
                                 (const uint64_t*)(lOffs + l0)};
  struct rcclIbSegWrIn in = {};
  in.local = &local;
  in.lkeys = lkeys;
  in.remote = &remote;
  in.rkeys = rkeys;
  in.localOff = localOff;
  in.remoteOff = remoteOff;
  in.len = size;
  in.opcode = opcode;
  in.maxWr = maxWr;
  in.covered = covered;
  int n = rcclIbSegAppendWrs(&in, wr, sge);
  if (n < 0) {
    WARN("NET/IB/RMA: transfer of %zu bytes needs more than %d segment slices or runs past the registered segments",
         size, maxWr);
    return ncclInternalError;
  }
  *nWr = n;
  return ncclSuccess;
}

// RCCL: ibv_post_send can fail after accepting a prefix, whose unsignaled WRs then need the request. Count them.
static ncclResult_t rcclRmaPostWrs(struct IbMsegQp* qp, struct ibv_send_wr* wr, int nWr, int* posted) {
  *posted = 0;
  if (nWr <= 0) return ncclSuccess;
  struct ibv_send_wr* bad_wr = NULL;
  ncclResult_t ret = wrap_ibv_post_send(qp->qp, wr, &bad_wr);
  if (ret == ncclSuccess) {
    *posted = nWr;
    return ncclSuccess;
  }
  *posted = rcclRmaPostedWrCount(wr, nWr, bad_wr, offsetof(struct ibv_send_wr, next));
  if (*posted > 0) {
    WARN("NET/IB/RMA: ibv_post_send failed after %d/%d WRs; leaving request for CQ drain", *posted, nWr);
  }
  return ret;
}

// RCCL: queue a chain behind earlier batched WRs. A chain longer than the batch posts right after it, in id order.
static ncclResult_t rcclRmaIbProxyQueueWrs(struct IbMsegRmaIbProxyCtx* ctx, struct IbMsegQp* qp, struct ibv_send_wr* wr,
                                           struct ibv_sge* sge, int nWr, bool aggregate, int* posted) {
  *posted = 0;
  if (nWr == 0) return ncclSuccess;
  if (nWr <= ctx->wrBatchsize) {
    struct ibv_send_wr* slotWr;
    struct ibv_sge* slotSge;
    NCCLCHECK(IbMsegRmaIbProxyReserveWrs(ctx, qp->qp, nWr, &slotWr, &slotSge));
    for (int i = 0; i < nWr; i++) {
      slotWr[i] = wr[i];
      slotSge[i] = sge[i];
      slotWr[i].sg_list = &slotSge[i];
    }
    // Once queued, a failed doorbell may already have handed these WRs to the QP.
    *posted = nWr;
    return IbMsegRmaIbProxyCommitWrs(ctx, nWr, aggregate);
  }
  NCCLCHECK(IbMsegRmaIbProxyPostBatch(ctx));
  wr[nWr - 1].send_flags |= IBV_SEND_SIGNALED;
  wr[nWr - 1].next = NULL;
  return rcclRmaPostWrs(qp, wr, nWr, posted);
}

// RCCL: free the request and its id if nothing posted. Otherwise publish it and mark FAILED if the signaled tail was
// lost.
static ncclResult_t rcclRmaFinishPostedRequest(struct IbMsegRequest* req, uint64_t* postedSeq, ncclResult_t postRet,
                                               int posted, int nWr, void** request) {
  int keep = 0, failed = 0;
  uint64_t id = req->id;
  ncclResult_t status = rcclRmaCompletePostedRequest(postRet, posted, nWr, &keep, &failed);
  if (!keep) {
    rcclRmaReleaseSeqId(id, postedSeq);
    (void)IbMsegFreeRequest(req);
    return status;
  }
  *request = req;
  if (failed) req->type = NCCL_NET_IB_REQ_FAILED;
  return status;
}

// RCCL: take a request for a multi-segment chain and stamp its WRs. A chain without WRs reuses the previous id.
static ncclResult_t rcclRmaIbProxyStartRequest(struct IbMsegRmaIbProxyCtx* ctx, struct IbMsegNetCommBase* base,
                                               uint64_t* postedSeq, int type, uint32_t rank, struct ibv_send_wr* wr,
                                               int nWr, int signalLast, struct IbMsegRequest** reqPtr) {
  struct IbMsegRequest* req;
  NCCLCHECK(IbMsegGetRequest(base, &req));
  req->rmaProxyCtx = ctx;
  req->type = type;
  req->sock = &base->sock;
  if (type == NCCL_NET_IB_REQ_GIN_IGET) {
    req->iget.rank = rank;
  } else {
    req->iput.rank = rank;
  }
  req->id = ++*postedSeq;
  for (int i = 0; i < base->vProps.ndevs; i++) req->devBases[i] = IbMsegGetNetCommDevBase(base, i);
  rcclRmaStampWrs(wr, nWr, req - base->reqs, signalLast);
  if (nWr == 0) req->id = --*postedSeq;
  *reqPtr = req;
  return ncclSuccess;
}

// RCCL: queue a started request's chain and finish it. Only IPut adds a WQE event, as in the single-MR ops.
static ncclResult_t rcclRmaIbProxyQueueRequest(struct IbMsegRmaIbProxyCtx* ctx, struct IbMsegQp* qp,
                                               struct IbMsegRequest* req, uint64_t* postedSeq, struct ibv_send_wr* wr,
                                               struct ibv_sge* sge, int nWr, uint32_t optFlags, bool addEvent,
                                               void** request) {
  int posted = 0;
  ncclResult_t postRet =
    rcclRmaIbProxyQueueWrs(ctx, qp, wr, sge, nWr, optFlags & ncclRmaOptFlagsAggregateRequests, &posted);
  if (addEvent) IbMsegAddEvent(req, qp->devIndex);
  return rcclRmaFinishPostedRequest(req, postedSeq, postRet, posted, nWr, request);
}

// RCCL: this plugin's half of a multi-segment round. A round ends with its signaled tail's CQE; the request
// completes with its last round, in id order.
static ncclResult_t rcclRmaIbRoundBuild(const struct rcclRmaRoundOp* op, struct ibv_send_wr* wr, struct ibv_sge* sge,
                                        int maxWr, int* nWr, uint64_t* covered) {
  return rcclRmaBuildSegmentedWrs(wr, sge, maxWr, nWr, op->opcode, (const struct IbMsegQp*)op->qp,
                                  (struct rcclRmaIbProxySegMrHandle*)op->localH, op->localRank, op->localOff,
                                  (struct rcclRmaIbProxySegMrHandle*)op->remoteH, op->remoteRank, op->remoteOff,
                                  op->remaining, covered);
}

// RCCL: counts and traces one posted round of op that ends before localEnd and remoteEnd. Counting at post, not
// build, keeps the probe build of rcclRmaRoundFirst out of the counters.
static void rcclRmaIbNoteRound(const struct rcclRmaRoundOp* op, uint64_t localEnd, uint64_t remoteEnd, int nWr) {
  const int nData = nWr - (op->hasSignal && op->remaining == 0 ? 1 : 0);
  if (nData <= 0 || localEnd == 0 || remoteEnd == 0) return;
  const int localSeg = rcclRmaSegOf((const struct rcclRmaIbProxySegMrHandle*)op->localH, op->localRank, localEnd - 1);
  const int remoteSeg =
      rcclRmaSegOf((const struct rcclRmaIbProxySegMrHandle*)op->remoteH, op->remoteRank, remoteEnd - 1);
  rcclIbSegStatsAdd(&((struct IbMsegSendComm*)op->comm)->segStats, nData, localSeg > 0 || remoteSeg > 0);
  TRACE(NCCL_NET, "NET/IB/RMA: opcode %d round of %d WRs ends at local rank %d off %lu seg %d, remote rank %d off %lu seg %d",
        (int)op->opcode, nData, op->localRank, (unsigned long)localEnd, localSeg, op->remoteRank,
        (unsigned long)remoteEnd, remoteSeg);
}

// rcclRmaRoundTailDone clears roundInFlight, so a round still flagged has not completed.
static bool rcclRmaIbRoundOutstanding(const struct rcclRmaRoundOp*) { return true; }

static ncclResult_t rcclRmaIbRoundPost(struct rcclRmaRoundOp* op, struct ibv_send_wr* wr, int nWr, int* posted) {
  struct IbMsegRequest* req = (struct IbMsegRequest*)op->req;
  rcclRmaStampWrs(wr, nWr, req - ((struct IbMsegNetCommBase*)op->comm)->reqs, /*signalLast=*/1);
  NCCLCHECK(rcclRmaPostWrs((struct IbMsegQp*)op->qp, wr, nWr, posted));
  if (*posted == nWr) rcclRmaIbNoteRound(op, op->localOff, op->remoteOff, nWr);
  return ncclSuccess;
}

static void rcclRmaIbRoundRetire(struct rcclRmaRoundOp* op, bool ok, int nWr, int posted) {
  if (ok) return;
  struct IbMsegRequest* req = (struct IbMsegRequest*)op->req;
  WARN("NET/IB/RMA: round of request %lu failed after %d/%d WRs", req->id, posted, nWr);
  req->type = NCCL_NET_IB_REQ_FAILED;
}

static bool rcclRmaIbRoundFailed(const struct rcclRmaRoundOp* op) {
  return ((const struct IbMsegRequest*)op->req)->type == NCCL_NET_IB_REQ_FAILED;
}

static const struct rcclRmaRoundHooks rcclRmaIbRoundHooks = {rcclRmaIbRoundBuild, rcclRmaIbRoundOutstanding,
                                                             rcclRmaIbRoundPost, rcclRmaIbRoundRetire,
                                                             rcclRmaIbRoundFailed};

// RCCL: a flush is one read posted at once, so only sends queue.
static inline struct rcclRmaRoundQueue* rcclRmaIbProxyQueueOf(struct IbMsegRmaIbProxyCtx* ctx, uint32_t rank) {
  return rcclRmaRoundQueueOf(&ctx->pending, rank);
}

// RCCL: behind a queued op every op queues, so a single-MR op cannot complete ahead of it.
static inline bool rcclRmaIbProxyPeerBusy(struct IbMsegRmaIbProxyCtx* ctx, uint32_t rank) {
  return rcclRmaRoundQueueBusy(rcclRmaIbProxyQueueOf(ctx, rank));
}

// RCCL: one chain with nothing queued ahead: batched for data, posted at once for a flush.
static ncclResult_t rcclRmaIbProxyPostChain(struct IbMsegRmaIbProxyCtx* ctx, struct IbMsegNetCommBase* base,
                                            uint64_t* postedSeq, int type, uint32_t rank, struct IbMsegQp* qp,
                                            struct ibv_send_wr* wr, struct ibv_sge* sge, int nWr, uint32_t optFlags,
                                            bool addEvent, void** request) {
  const bool flush = type == NCCL_NET_IB_REQ_FLUSH;
  struct IbMsegRequest* req;
  NCCLCHECK(rcclRmaIbProxyStartRequest(ctx, base, postedSeq, type, rank, wr, nWr, /*signalLast=*/flush, &req));
  if (!flush) return rcclRmaIbProxyQueueRequest(ctx, qp, req, postedSeq, wr, sge, nWr, optFlags, addEvent, request);
  int posted = 0;
  ncclResult_t postRet = rcclRmaPostWrs(qp, wr, nWr, &posted);
  return rcclRmaFinishPostedRequest(req, postedSeq, postRet, posted, nWr, request);
}

// RCCL: queue an op behind the peer's earlier ones. As the new head it posts after the batched WRs ahead of it.
static ncclResult_t rcclRmaIbProxyEnqueue(struct IbMsegRmaIbProxyCtx* ctx, struct rcclRmaRoundQueue* q,
                                          struct IbMsegNetCommBase* base, uint64_t* postedSeq, int type,
                                          uint32_t rank, const struct rcclRmaRoundOp* op, struct ibv_send_wr* wr,
                                          int nWr, bool addEvent, void** request) {
  if (q == NULL) {
    WARN("NET/IB/RMA: transfer of %lu bytes needs more than one round and this context has no pending queue",
         op->remaining);
    return ncclInternalError;
  }
  // RCCL: ring the batch before publishing the request, so a doorbell failure leaves nothing queued.
  if (!rcclRmaRoundQueueBusy(q)) NCCLCHECK(IbMsegRmaIbProxyPostBatch(ctx));
  struct rcclRmaRoundOp* node;
  NCCLCHECK(ncclCalloc(&node, 1));
  rcclRmaRoundFillNode(node, op);
  struct IbMsegRequest* req;
  ncclResult_t ret = rcclRmaIbProxyStartRequest(ctx, base, postedSeq, type, rank, wr, nWr, /*signalLast=*/1, &req);
  if (ret != ncclSuccess) {
    free(node);
    return ret;
  }
  node->req = req;
  if (addEvent) IbMsegAddEvent(req, ((struct IbMsegQp*)op->qp)->devIndex);
  *request = req;
  if (rcclRmaRoundPush(q, node)) rcclRmaRoundPump(&rcclRmaIbRoundHooks, q);
  return ncclSuccess;
}

// RCCL: post a multi-segment op as one chain when it fits in one round with nothing queued ahead, otherwise round by
// round from the peer's queue. A chain without WRs reuses the previous id, so it completes with what posted before it.
static ncclResult_t rcclRmaIbProxySubmit(struct IbMsegRmaIbProxyCtx* ctx, struct IbMsegNetCommBase* base,
                                         uint64_t* postedSeq, int type, uint32_t rank, const struct rcclRmaRoundOp* op,
                                         uint32_t optFlags, bool addEvent, void** request) {
  struct rcclRmaRoundQueue* q = rcclRmaIbProxyQueueOf(ctx, rank);
  struct ibv_send_wr wr[RCCL_RMA_MAX_SIGNAL_WRS];
  struct ibv_sge sge[RCCL_RMA_MAX_SIGNAL_WRS];
  int nWr = 0;
  bool direct = true;
  NCCLCHECK(rcclRmaRoundFirst(&rcclRmaIbRoundHooks, q, op, wr, sge, &nWr, &direct));
  if (direct) {
    NCCLCHECK(rcclRmaIbProxyPostChain(ctx, base, postedSeq, type, rank, (struct IbMsegQp*)op->qp, wr, sge, nWr,
                                      optFlags, addEvent, request));
    struct rcclRmaRoundOp whole = *op;
    whole.remaining = 0;
    rcclRmaIbNoteRound(&whole, op->localOff + op->remaining, op->remoteOff + op->remaining, nWr);
    return ncclSuccess;
  }
  return rcclRmaIbProxyEnqueue(ctx, q, base, postedSeq, type, rank, op, wr, nWr, addEvent, request);
}

// RCCL: the RMA segment handle is an IbMsegMrHandle, so the plugin's deregMr releases it whole.
static ncclResult_t rcclRmaDeregOneSegment(void* recvComm, int, void* mr) {
  return ncclNetIbMseg.deregMr(recvComm, mr);
}

// RCCL: release the first nRegistered segment MRs through the shared deregister, then free the handle.
static ncclResult_t rcclRmaIbProxySegMrHandleFree(void* recvComm, struct rcclRmaIbProxySegMrHandle* h,
                                                  int nRegistered) {
  struct rcclIbSegRegOps ops = {NULL, rcclRmaDeregOneSegment};
  ncclResult_t ret = rcclIbSegDeregister(&ops, recvComm, h->mrHandle, h->mrHandle == NULL ? 0 : nRegistered);
  free(h->segOff);
  free(h->mrHandle);
  rcclIbSegSymTableFree(&h->sym);
  free(h->single.rkeys);
  free(h);
  return ret;
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
struct rcclRmaSegRegCtx {
  struct IbMsegGinIbCollComm* cComm;
  int type;
  uint64_t mr_flags;
  struct rcclIbSegWalkState walk;
  int flushSeg;  // first device segment, -1 until one is found
};

// RCCL: one segment of the RMA walk. A failure returns to rcclIbSegRegister, which releases the earlier segments.
static ncclResult_t rcclRmaRegOneSegment(void* vctx, int seg, void** mr) {
  struct rcclRmaSegRegCtx* ctx = (struct rcclRmaSegRegCtx*)vctx;
  struct rcclIbSegExport e;
  NCCLCHECK(rcclIbSegExportNext(&ctx->walk, &e));
  ncclResult_t ret = ncclSuccess;
  if (ctx->flushSeg < 0) {
    bool host = true;
    ret = rcclIbSegExportIsHost(&e, &host);
    if (ret != ncclSuccess) {
      rcclIbSegExportClose(&e);
      return ret;
    }
    if (!host) ctx->flushSeg = seg;
  }
  void* raw = NULL;
  ret = IbMsegRegMrDmaBufInternal(ctx->cComm->recvComm, (void*)e.ptr, e.len, ctx->type, e.fdOffset, e.fd,
                                  ctx->mr_flags, &raw);
  rcclIbSegExportClose(&e);
  if (ret != ncclSuccess) return ret;
  *mr = raw;
  rcclIbSegWalkAdvance(&ctx->walk, seg, e.len);
  return ncclSuccess;
}
#endif

static ncclResult_t rcclRmaRegAllGather(void* ctx, void* src, void* dst, size_t bytes) {
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)ctx;
  return cComm->allGather(cComm, src, dst, bytes);
}

ncclResult_t rcclRmaIbProxyConnectFinish(void** collComm, int nranks) {
  if (!rcclRmaMultiSegEnabled()) return ncclSuccess;
  // RCCL: registration must not fail before its first collective, so its page-aligned allGather buffer is made here.
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)*collComm;
  ncclResult_t ret = ncclIbMalloc(&cComm->regRecords, (size_t)nranks * sizeof(struct rcclIbSegSymRecord));
  if (ret != ncclSuccess) {
    IbMsegGinIbCloseColl(*collComm);
    *collComm = NULL;
    return ret;
  }
  return ncclSuccess;
}

// RCCL: one DMA-BUF MR per physical segment, then one consensus record per rank.
ncclResult_t rcclRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                          uint64_t mr_flags, void** mhandle) {
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;
  struct rcclRmaIbProxySegMrHandle* rmaMrHandle = NULL;
  struct rcclIbSegSymRecord* registrations = (struct rcclIbSegSymRecord*)cComm->regRecords;
  uintptr_t* localVas = NULL;
  uint64_t* localRkeys = NULL;
  ncclResult_t ret = ncclSuccess;
  int nSeg = 1;
  int registered = 0;

  *mhandle = NULL;
  ret = ncclCalloc(&rmaMrHandle, 1);
  if (ret != ncclSuccess) goto reconcile;
  // Fail paths deregister only the `registered` complete handles.

  // A HIP export describes only its first physical segment, so count them and register one MR each.
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  if (type == NCCL_PTR_CUDA && ncclCuMemEnable()) {
    CUdeviceptr base = 0;
    size_t baseSize = 0;
    NCCLCHECKGOTO(ncclCuMemGetAddressRange((CUdeviceptr)data, size, &base, &baseSize, &nSeg), ret, reconcile);
  }
#endif
  if (nSeg < 1) nSeg = 1;
  if (!rcclIbSegCountSupported(nSeg)) {
    WARN("NET/IB/RMA: buffer %p (size %zu) spans %d segments, more than the supported %d", data, size, nSeg,
         RCCL_IB_REG_SEG_DEFAULT);
    ret = ncclInvalidUsage;
    goto reconcile;
  }
  if (!rcclRmaSegTableFits(cComm->nranks, nSeg)) {
    WARN("NET/IB/RMA: buffer %p: %d ranks x %d segments is past the %d-entry segment table", data, cComm->nranks,
         nSeg, RCCL_RMA_SEG_TABLE_MAX_ENTRIES);
    ret = ncclInvalidUsage;
    goto reconcile;
  }
  if (nSeg > 1 && !rcclIbMrAdmit((struct IbMsegNetCommBase*)cComm->recvComm, nSeg)) {
    WARN("NET/IB/RMA: buffer %p: %d more MRs would pass the device max_mr", data, nSeg);
    ret = ncclInvalidUsage;
    goto reconcile;
  }
  // RCCL: every table below is sized from the walked count.
  NCCLCHECKGOTO(ncclCalloc(&rmaMrHandle->segOff, (size_t)nSeg + 1), ret, reconcile);
  NCCLCHECKGOTO(ncclCalloc(&rmaMrHandle->mrHandle, (size_t)nSeg), ret, reconcile);
  NCCLCHECKGOTO(ncclCalloc(&localVas, (size_t)nSeg), ret, reconcile);
  NCCLCHECKGOTO(ncclCalloc(&localRkeys, (size_t)nSeg * NCCL_IB_MAX_DEVS_PER_NIC), ret, reconcile);
  rmaMrHandle->tag = rcclRmaSegMrTag();
  rmaMrHandle->nSegments = nSeg;
  rmaMrHandle->flushSeg = 0;

  if (nSeg == 1) {
    // Single segment: reuse the caller's fd (fd==-1 falls back to ibv_reg_mr).
    NCCLCHECKGOTONOWARN(IbMsegRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags,
                                                  &rmaMrHandle->mrHandle[0]),
                        ret, reconcile, NCCL_NET);
    registered = 1;
    rmaMrHandle->segOff[1] = size;
    localVas[0] = (uintptr_t)data;
  } else if (!ncclParamDmaBufEnable()) {
    // Rejected through the status exchange below so every rank fails together.
    WARN("NET/IB/RMA: multi-segment (%d) registration of %p requires NCCL_DMABUF_ENABLE=1", nSeg, data);
    ret = ncclInvalidUsage;
    goto reconcile;
  } else {
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
    struct rcclRmaSegRegCtx regCtx = {};
    regCtx.cComm = cComm;
    regCtx.type = type;
    regCtx.mr_flags = mr_flags;
    regCtx.walk.vas = localVas;
    regCtx.walk.segOff = rmaMrHandle->segOff;
    regCtx.walk.ptr = (uintptr_t)data;
    regCtx.walk.remaining = size;
    regCtx.flushSeg = -1;
    struct rcclIbSegRegOps ops = {rcclRmaRegOneSegment, rcclRmaDeregOneSegment};
    ret = rcclIbSegRegister(&ops, &regCtx, rmaMrHandle->mrHandle, nSeg);
    if (ret != ncclSuccess) goto reconcile;
    registered = nSeg;
    rmaMrHandle->flushSeg = regCtx.flushSeg;
    if (regCtx.walk.remaining != 0) {
      WARN("NET/IB/RMA: enumerated %d/%d segments for buffer %p but %zu bytes remain unregistered", registered, nSeg,
           data, regCtx.walk.remaining);
      ret = ncclInvalidUsage;
      goto reconcile;
    }
    INFO(NCCL_NET | NCCL_REG, "NET/IB/RMA: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", data, size,
         nSeg);
#else
    WARN("NET/IB/RMA: multi-segment (%d) registration requires HIP >= 7.12.60540", nSeg);
    ret = ncclInvalidUsage;
    goto reconcile;
#endif
  }

  {
    struct IbMsegNetCommBase* recvBase = (struct IbMsegNetCommBase*)cComm->recvComm;
    int ndevs = recvBase->vProps.ndevs;
    if (ndevs < 1 || ndevs > NCCL_IB_MAX_DEVS_PER_NIC) {
      WARN("NET/IB/RMA: invalid ndevs %d for buffer %p", ndevs, data);
      ret = ncclInternalError;
      goto reconcile;
    }
    for (int s = 0; s < nSeg; s++) {
      for (int d = 0; d < ndevs; d++) {
        struct ibv_mr* mr = ((struct IbMsegMrHandle*)rmaMrHandle->mrHandle[s])->mrs[d];
        if (mr == NULL) {
          WARN("NET/IB/RMA: missing MR for segment %d device %d", s, d);
          ret = ncclInternalError;
          goto reconcile;
        }
        localRkeys[(size_t)s * NCCL_IB_MAX_DEVS_PER_NIC + d] = mr->rkey;
      }
    }
  }
  if (nSeg == 1) NCCLCHECKGOTO(ncclCalloc(&rmaMrHandle->single.rkeys, cComm->nranks), ret, reconcile);

reconcile:
  // No rank-local failure may return before the exchange's first collective; it carries the status and the count.
  NCCLCHECKGOTO(rcclIbSegSymExchange(rcclRmaRegAllGather, cComm, cComm->nranks, registrations, ret, nSeg,
                                     NCCL_IB_MAX_DEVS_PER_NIC, rmaMrHandle ? rmaMrHandle->segOff : NULL, localVas,
                                     localRkeys, data, "NET/IB/RMA", rmaMrHandle ? &rmaMrHandle->sym : NULL),
                ret, fail);
  if (nSeg == 1) {
    rmaMrHandle->single.mrHandle = (struct IbMsegMrHandle*)rmaMrHandle->mrHandle[0];
    rmaMrHandle->single.base_vas = rmaMrHandle->sym.vas;
    for (int r = 0; r < cComm->nranks; r++) rmaMrHandle->single.rkeys[r] = rcclRmaRemoteRkey(rmaMrHandle, r, 0, 0);
  }

  *mhandle = rmaMrHandle;
  free(localVas);
  free(localRkeys);
  return ncclSuccess;

fail:
  // rcclIbSegRegister has already released a failed multi-segment walk, so `registered` stays 0 for that path.
  if (rmaMrHandle) (void)rcclRmaIbProxySegMrHandleFree(cComm->recvComm, rmaMrHandle, registered);
  free(localVas);
  free(localRkeys);
  return ret;
}

// RCCL: release every segment MR, not only the first.
ncclResult_t rcclRmaIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct IbMsegGinIbCollComm* cComm = (struct IbMsegGinIbCollComm*)collComm;
  struct rcclRmaIbProxySegMrHandle* rmaMrHandle = (struct rcclRmaIbProxySegMrHandle*)mhandle;
  return rcclRmaIbProxySegMrHandleFree(cComm->recvComm, rmaMrHandle, rmaMrHandle->nSegments);
}

ncclResult_t IbMsegRmaIbProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request);
ncclResult_t IbMsegRmaIbProxyIGet(void* rmaCtx, int context, uint64_t remoteOffset, void* remoteMhandle, size_t size,
                                uint64_t localOffset, void* localMhandle, uint32_t rank, uint32_t optFlags,
                                void** request);
ncclResult_t IbMsegRmaIbProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                      uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                      void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                      uint32_t optFlags, void** request);
ncclResult_t IbMsegRmaIbProxyIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request);

// RCCL: a one-segment handle uses the single-MR op body through the embedded single MR.
static bool rcclRmaIbProxyUseSingle(void** h0, void** h1, void** h2) {
  void** handles[3] = {h0, h1, h2};
  for (void** h : handles) {
    if (h && *h && ((struct rcclRmaIbProxySegMrHandle*)*h)->nSegments != 1) return false;
  }
  for (void** h : handles) {
    if (h && *h) *h = &((struct rcclRmaIbProxySegMrHandle*)*h)->single;
  }
  return true;
}

ncclResult_t rcclRmaIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags,
                                    void** mhandle) {
  return rcclRmaIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle);
}

// RCCL: split the put at segment boundaries and keep the request if a prefix posted.
ncclResult_t rcclRmaIbProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request) {
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];
  if (!rcclRmaIbProxyPeerBusy(rmaProxyCtx, rank) && rcclRmaIbProxyUseSingle(&srcMhandle, &dstMhandle, NULL)) {
    return IbMsegRmaIbProxyIPut(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, optFlags, request);
  }

  struct rcclRmaIbProxySegMrHandle* srcMrHandle = (struct rcclRmaIbProxySegMrHandle*)srcMhandle;
  struct rcclRmaIbProxySegMrHandle* dstMrHandle = (struct rcclRmaIbProxySegMrHandle*)dstMhandle;

  // Reject out-of-range transfers before any WR is posted.
  if (!rcclRmaRangeOk(srcMrHandle, rmaProxyCtx->rank, srcOff, size) ||
      !rcclRmaRangeOk(dstMrHandle, rank, dstOff, size)) {
    WARN("NET/IB/RMA: iput out of range (srcOff=%lu dstOff=%lu size=%zu)", srcOff, dstOff, size);
    return ncclInvalidArgument;
  }

  struct IbMsegSendComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));

  // Each slice maps to one local lkey and one remote rkey.
  struct rcclRmaRoundOp op = rcclRmaRoundDataOp(&comm->base, &comm->base.qps[0], IBV_WR_RDMA_WRITE, srcMrHandle,
                                                rmaProxyCtx->rank, srcOff, dstMrHandle, rank, dstOff, size);
  return rcclRmaIbProxySubmit(rmaProxyCtx, &comm->base, &comm->ginSeq.posted, NCCL_NET_IB_REQ_GIN_IPUT, rank, &op,
                              optFlags, /*addEvent=*/true, request);
}

// RCCL: split the get at segment boundaries and keep the request if a prefix posted.
ncclResult_t rcclRmaIbProxyIGet(void* rmaCtx, int context, uint64_t remoteOffset, void* remoteMhandle, size_t size,
                                uint64_t localOffset, void* localMhandle, uint32_t rank, uint32_t optFlags,
                                void** request) {
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];
  if (!rcclRmaIbProxyPeerBusy(rmaProxyCtx, rank) &&
      rcclRmaIbProxyUseSingle(&remoteMhandle, &localMhandle, NULL)) {
    return IbMsegRmaIbProxyIGet(rmaCtx, context, remoteOffset, remoteMhandle, size, localOffset, localMhandle, rank,
                              optFlags, request);
  }

  struct rcclRmaIbProxySegMrHandle* remoteMrHandle = (struct rcclRmaIbProxySegMrHandle*)remoteMhandle;
  struct rcclRmaIbProxySegMrHandle* localMrHandle = (struct rcclRmaIbProxySegMrHandle*)localMhandle;

  // Reject out-of-range transfers before any WR is posted.
  if (!rcclRmaRangeOk(remoteMrHandle, rank, remoteOffset, size) ||
      !rcclRmaRangeOk(localMrHandle, rmaProxyCtx->rank, localOffset, size)) {
    WARN("NET/IB/RMA: iget out of range (remoteOff=%lu localOff=%lu size=%zu)", remoteOffset, localOffset, size);
    return ncclInvalidArgument;
  }

  struct IbMsegSendComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));

  // RDMA READ: the local buffer is the destination (lkey), the remote buffer the source (rkey).
  struct rcclRmaRoundOp op = rcclRmaRoundDataOp(&comm->base, &comm->base.qps[0], IBV_WR_RDMA_READ, localMrHandle,
                                                rmaProxyCtx->rank, localOffset, remoteMrHandle, rank, remoteOffset,
                                                size);
  return rcclRmaIbProxySubmit(rmaProxyCtx, &comm->base, &comm->ginSeq.posted, NCCL_NET_IB_REQ_GIN_IGET, rank, &op,
                              optFlags, /*addEvent=*/false, request);
}

// RCCL: the signal rides the last segment WR, so a split put still signals once.
ncclResult_t rcclRmaIbProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                      uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                      void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                      uint32_t optFlags, void** request) {
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];
  if (!rcclRmaIbProxyPeerBusy(rmaProxyCtx, rank) &&
      rcclRmaIbProxyUseSingle(&srcMhandle, &dstMhandle, &signalMhandle)) {
    return IbMsegRmaIbProxyIPutSignal(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff,
                                    signalMhandle, signalValue, signalOp, isStrongSignal, optFlags, request);
  }
  (void)isStrongSignal;
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("NET/IB/RMA: iputSignal unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct rcclRmaIbProxySegMrHandle* srcMrHandle = (struct rcclRmaIbProxySegMrHandle*)srcMhandle;
  struct rcclRmaIbProxySegMrHandle* dstMrHandle = (struct rcclRmaIbProxySegMrHandle*)dstMhandle;
  struct rcclRmaIbProxySegMrHandle* signalMrHandle = (struct rcclRmaIbProxySegMrHandle*)signalMhandle;

  // Reject out-of-range payload/signal before any WR is posted (signal is an 8-byte atomic).
  if ((size > 0 && (!srcMrHandle || !dstMrHandle || !rcclRmaRangeOk(srcMrHandle, rmaProxyCtx->rank, srcOff, size) ||
                    !rcclRmaRangeOk(dstMrHandle, rank, dstOff, size))) ||
      !signalMrHandle || !rcclRmaRangeOk(signalMrHandle, rank, signalOff, sizeof(uint64_t))) {
    WARN("NET/IB/RMA: iputSignal out of range (srcOff=%lu dstOff=%lu size=%zu signalOff=%lu)", srcOff, dstOff, size,
         signalOff);
    return ncclInvalidArgument;
  }
  int sig = rcclRmaSegOf(signalMrHandle, rank, signalOff);
  if (!rcclRmaSignalOffsetValid(signalOff, rcclRmaRankOff(signalMrHandle, rank)[sig + 1])) {
    WARN("NET/IB/RMA: iputSignal atomic must be 8-byte aligned and contained in one segment (signalOff=%lu)",
         signalOff);
    return ncclInvalidArgument;
  }

  struct IbMsegSendComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct IbMsegQp* qp = &comm->base.qps[0];
  struct rcclRmaRoundOp op = rcclRmaRoundDataOp(&comm->base, qp, IBV_WR_RDMA_WRITE, srcMrHandle, rmaProxyCtx->rank,
                                                srcOff, dstMrHandle, rank, dstOff, size);
  // SIGNAL (route to the segment that contains signalOff). It rides the last round; same-QP RC ordering lands every
  // unsignaled write before it, and size 0 sends only the signal.
  rcclRmaRoundSetSignal(&op,
                        rcclIbSegSymVa(&signalMrHandle->sym, signalMrHandle->nSegments, rank, sig) +
                            (signalOff - rcclRmaRankOff(signalMrHandle, rank)[sig]),
                        rcclRmaRemoteRkey(signalMrHandle, rank, sig, qp->remDevIdx),
                        signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue, (uintptr_t)&comm->putSignalScratchpad,
                        sizeof(comm->putSignalScratchpad), comm->devs[qp->devIndex].putSignalScratchpadMr->lkey);
  return rcclRmaIbProxySubmit(rmaProxyCtx, &comm->base, &comm->ginSeq.posted, NCCL_NET_IB_REQ_GIN_IPUT, rank, &op,
                              optFlags, /*addEvent=*/false, request);
}

// RCCL: a PCIe read cannot pass the posted writes ahead of it, so one read of any device segment flushes them all.
ncclResult_t rcclRmaIbProxyIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request) {
  struct IbMsegRmaIbProxyCtx* rmaProxyCtx = &((struct IbMsegRmaIbProxyCtx*)rmaCtx)[context];
  if (rcclRmaIbProxyUseSingle(&mhandle, NULL, NULL)) {
    return IbMsegRmaIbProxyIFlush(rmaCtx, context, mhandle, rank, request);
  }
  struct rcclRmaIbProxySegMrHandle* rmaMrHandle = (struct rcclRmaIbProxySegMrHandle*)mhandle;
  struct IbMsegRecvComm* comm;
  NCCLCHECK(IbMsegRmaIbProxyGetRecvComm(rmaProxyCtx, rank, &comm));
  // Host memory needs no GPUDirect flush.
  if (comm->flushEnabled == 0 || rmaMrHandle->flushSeg < 0) {
    *request = NULL;
    return ncclSuccess;
  }

  // drain batch before posting the flush
  NCCLCHECK(IbMsegRmaIbProxyPostBatch(rmaProxyCtx));
  NCCLCHECK(IbMsegCreateFlushQp(comm));
  struct IbMsegQp* qp = &comm->devs[0].gpuFlush.qp;

  // Loopback read of this rank's own flush segment, so the rkey is the local MR's.
  const int seg = rmaMrHandle->flushSeg;
  struct ibv_mr* mr = ((struct IbMsegMrHandle*)rmaMrHandle->mrHandle[seg])->mrs[qp->devIndex];
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr.rdma.remote_addr = (uint64_t)rcclIbSegSymVa(&rmaMrHandle->sym, rmaMrHandle->nSegments, rmaProxyCtx->rank, seg);
  wr.wr.rdma.rkey = mr->rkey;
  wr.sg_list = &comm->devs[qp->devIndex].gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  TRACE(NCCL_NET, "NET/IB: %s: Posting a flush of segment %d/%d (comm=%p)", __func__, seg, rmaMrHandle->nSegments,
        comm);
  return rcclRmaIbProxyPostChain(rmaProxyCtx, &comm->base, &comm->ginSeq.posted, NCCL_NET_IB_REQ_FLUSH, rank, qp, &wr,
                                 wr.sg_list, 1, /*optFlags=*/0, /*addEvent=*/false, request);
}

// RCCL: a signaled tail that never posted has no CQE. Later completions must not hide it.
ncclResult_t rcclRmaIbProxyTestFailed(struct IbMsegRequest* req, int* done) {
  *done = 1;
  NCCLCHECK(IbMsegFreeRequest(req));
  return ncclRemoteError;
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
  // RCCL: a request whose signaled tail never posted completes as an error.
  if (req->type == NCCL_NET_IB_REQ_FAILED) return rcclRmaIbProxyTestFailed(req, done);

  IbMsegNetCommBase* commBase;
  IbMsegNetCommDevBase* devBase;
  struct rcclRmaRoundQueue* pending = isFlush ? NULL : rcclRmaIbProxyQueueOf(rmaProxyCtx, rank);

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
    if (rcclRmaRoundTailDone(pending, wcReq)) continue;

    if (wcReq->id > *completedSeq) *completedSeq = wcReq->id;
  }
  rcclRmaRoundPump(&rcclRmaIbRoundHooks, pending);

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
                            IbMsegRmaIbProxyListen,
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
