/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// RCCL: classic NET/IB per-segment MR registration (see rccl_ib_multiseg.h).

#include "common.h"
#include "rccl_ib_multiseg.h"

ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset,
                                        int fd, uint64_t mrFlags, ibv_mr** mhandle);
ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle);

/* Multi-segment DMA-BUF support (AIRUNTIME-2351 classic-path follow-up).
 *
 * ROCm/HIP dma-buf export describes only the first physical segment of a
 * multi-segment cuMem/VMM range, so registering the whole VA range as one MR
 * fails with EINVAL. Register one MR per segment (per device) instead. The
 * caller (net.cc proxy register) has already exported one dma-buf fd per
 * segment; segAddrs[s]/segLens[s]/segOffsets[s]/segFds[s] describe segment s.
 */
ncclResult_t ncclIbRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  *mhandle = NULL;
  if (nSeg < 1 || nSeg > NCCL_IB_MAX_SEGMENTS) {
    WARN("NET/IB: multi-segment registration with %d segments exceeds NCCL_IB_MAX_SEGMENTS=%d", nSeg,
         NCCL_IB_MAX_SEGMENTS);
    return ncclInvalidUsage;
  }
  uint32_t peerCaps =
    base->isSend ? ((struct ncclIbSendComm*)comm)->peerCaps : ((struct ncclIbRecvComm*)comm)->peerCaps;
  if (ncclIbDeclineMultiSegRegistration(peerCaps, nSeg)) {
    INFO(NCCL_NET | NCCL_REG,
         "NET/IB: peer does not advertise multi-segment CTS side table; declining %d-segment registration", nSeg);
    return ncclInvalidUsage;
  }
  struct ncclIbMrHandle* mhandleWrapper = NULL;
  NCCLCHECK(ncclCalloc(&mhandleWrapper, 1));
  mhandleWrapper->nSegments = nSeg;
  for (int s = 0; s < nSeg; s++) {
    mhandleWrapper->segStart[s] = (uintptr_t)segAddrs[s];
    mhandleWrapper->segLen[s] = segLens[s];
    for (int i = 0; i < base->vProps.ndevs; i++) {
      struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
      NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, segAddrs[s], segLens[s], type, segOffsets[s], segFds[s], 0ULL,
                                               &mhandleWrapper->segMrs[s][i]),
                    ret, fail);
    }
  }
  // Alias segment 0 into mrs[] so single-segment consumers keep working.
  for (int i = 0; i < base->vProps.ndevs; i++) mhandleWrapper->mrs[i] = mhandleWrapper->segMrs[0][i];
  INFO(NCCL_NET | NCCL_REG, "NET/IB: registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", segAddrs[0],
       (size_t)(mhandleWrapper->segStart[nSeg - 1] + mhandleWrapper->segLen[nSeg - 1] - mhandleWrapper->segStart[0]),
       nSeg);
  *mhandle = (void*)mhandleWrapper;
  return ncclSuccess;
fail:
  for (int s = 0; s < nSeg; s++) {
    for (int i = 0; i < base->vProps.ndevs; i++) {
      if (mhandleWrapper->segMrs[s][i]) {
        struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
        (void)ncclIbDeregMrInternal(devComm, mhandleWrapper->segMrs[s][i]);
      }
    }
  }
  free(mhandleWrapper);
  return ret;
}

// mrs[] aliases segment 0, so it is freed via segMrs[0] here (do not double-free).
ncclResult_t rcclIbDeregMrMultiSeg(void* comm, void* mhandle) {
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  for (int s = 0; s < mhandleWrapper->nSegments; s++) {
    for (int i = 0; i < base->vProps.ndevs; i++) {
      struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
      NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->segMrs[s][i]));
    }
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
