/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RCCL_IB_MULTISEG_H_
#define RCCL_IB_MULTISEG_H_

#include "nccl.h"

// RCCL-only IB helpers (same convention as rccl_wrap.cc): register one dma-buf
// MR per physical segment. Not part of the NCCL-synced plugin API in
// include/net.h. Classic is implemented in net_ib/reg.cc; CAST in
// net_ib_cast/reg.cc. Distinct symbols because both plugins link into librccl.
ncclResult_t ncclIbRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle);
ncclResult_t IbCastRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle);

#endif
