/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// RCCL: NET/IB-CAST per-segment MR registration (see rccl_ib_multiseg.h).

#include "common_cast.h"
#include "qp_sharing.h"
#include "rccl_ib_multiseg.h"

ncclResult_t IbCastDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle);

// Distinct symbol from ncclIbRegMrDmaBufMultiSeg (both plugins link into librccl).
// Declines nSegments>1 without peer NCCL_IB_CAP_MULTISEG, or under CTS offload, BY_ORDER or QP sharing.
ncclResult_t IbCastRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  struct ncclIbMrHandle* mhandleWrapper = NULL;
  *mhandle = NULL;
  if (nSeg < 1 || nSeg > NCCL_IB_MAX_SEGMENTS) {
    WARN("NET/IB: multi-segment registration with %d segments exceeds NCCL_IB_MAX_SEGMENTS=%d", nSeg,
         NCCL_IB_MAX_SEGMENTS);
    return ncclInvalidUsage;
  }
  if (nSeg > 1) {
    uint32_t peerCaps;
    bool useCtsOffload;
    if (base->isSend) {
      peerCaps = ((struct ncclIbSendComm*)comm)->peerCaps;
      useCtsOffload = ((struct ncclIbSendComm*)comm)->useCtsOffload;
    } else {
      peerCaps = ((struct ncclIbRecvComm*)comm)->peerCaps;
      useCtsOffload = ((struct ncclIbRecvComm*)comm)->useCtsOffload;
    }
    // BY_ORDER adds byte_len per imm, and shared QPs need a commId and per-QP WR budget;
    // segmented sends publish size out of band and carry neither.
    if (useCtsOffload || base->recvMatchingScheme == BY_ORDER || IbCastQpSharingEnabled() ||
        ncclIbDeclineMultiSegRegistration(peerCaps, nSeg)) {
      INFO(NCCL_NET | NCCL_REG,
           "NET/IB: CAST declining %d-segment registration (ctsOffload=%d scheme=%d qpSharing=%d peerCaps=0x%x)", nSeg,
           (int)useCtsOffload, base->recvMatchingScheme, (int)IbCastQpSharingEnabled(), peerCaps);
      return ncclInvalidUsage;
    }
  }
  NCCLCHECK(ncclCalloc(&mhandleWrapper, 1));
  mhandleWrapper->nSegments = nSeg;
  for (int s = 0; s < nSeg; s++) {
    struct ncclIbMrHandle* segHandle = NULL;
    mhandleWrapper->segStart[s] = (uintptr_t)segAddrs[s];
    mhandleWrapper->segLen[s] = segLens[s];
    // reg.cc keeps the per-device helper file-local; this path registers the same cached MRs per device.
    NCCLCHECKGOTO(IbCastRegMrDmaBufInternal(comm, segAddrs[s], segLens[s], type, segOffsets[s], segFds[s], 0ULL,
                                            (void**)&segHandle),
                  ret, fail);
    for (int i = 0; i < base->vProps.ndevs; i++) mhandleWrapper->segMrs[s][i] = segHandle->mrs[i];
    free(segHandle);
  }
  // Alias segment 0 into mrs[] so single-segment consumers keep working.
  for (int i = 0; i < base->vProps.ndevs; i++) mhandleWrapper->mrs[i] = mhandleWrapper->segMrs[0][i];
  INFO(NCCL_NET | NCCL_REG, "NET/IB: CAST registered multi-segment buffer %p size %zu as %d DMA-BUF MRs", segAddrs[0],
       (size_t)(mhandleWrapper->segStart[nSeg - 1] + mhandleWrapper->segLen[nSeg - 1] - mhandleWrapper->segStart[0]),
       nSeg);
  *mhandle = (void*)mhandleWrapper;
  return ncclSuccess;
fail:
  for (int s = 0; s < nSeg; s++) {
    for (int i = 0; i < base->vProps.ndevs; i++) {
      if (mhandleWrapper->segMrs[s][i]) {
        struct ncclIbNetCommDevBase* devComm = IbCastGetNetCommDevBase(base, i);
        (void)IbCastDeregMrInternal(devComm, mhandleWrapper->segMrs[s][i]);
      }
    }
  }
  free(mhandleWrapper);
  return ret;
}

// mrs[] aliases segment 0, so it is freed via segMrs[0] here (do not double-free).
ncclResult_t rcclIbCastDeregMrMultiSeg(void* comm, void* mhandle) {
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  for (int s = 0; s < mhandleWrapper->nSegments; s++) {
    for (int i = 0; i < base->vProps.ndevs; i++) {
      struct ncclIbNetCommDevBase* devComm = IbCastGetNetCommDevBase(base, i);
      NCCLCHECK(IbCastDeregMrInternal(devComm, mhandleWrapper->segMrs[s][i]));
    }
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
