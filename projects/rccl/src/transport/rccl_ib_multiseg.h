/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RCCL_IB_MULTISEG_H_
#define RCCL_IB_MULTISEG_H_

#include "nccl.h"

// RCCL-only IB helper (same convention as rccl_wrap.cc): register one dma-buf
// MR per physical segment. Not part of the NCCL-synced plugin API in
// include/net.h. Implemented by the classic NET/IB plugin (net_ib/rccl_reg_multiseg.cc).
// CAST is a separate follow-up PR.
ncclResult_t ncclIbRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle);

// RCCL-only proxy helper (net.cc): export and register a multi-segment buffer.
struct ncclProxyState;
ncclResult_t rcclNetIbRegMrMultiSeg(struct ncclProxyState* proxyState, void* netComm, void* buffer, size_t totalSize,
                                    int numSegments, int handleFlags, void** handle);
// Frees a handle from ncclIbRegMrDmaBufMultiSeg (net_ib/reg.cc dereg hook).
ncclResult_t rcclIbDeregMrMultiSeg(void* comm, void* mhandle);

#endif
