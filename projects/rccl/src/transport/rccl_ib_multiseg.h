/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RCCL_IB_MULTISEG_H_
#define RCCL_IB_MULTISEG_H_

#include "nccl.h"

// RCCL: one dma-buf MR per physical segment. Not part of the NCCL plugin API.
// Classic: ncclIbRegMrDmaBuf calls ncclIbRegMrDmaBufMultiSeg when fd < 0.
// CAST: IbCastRegMrDmaBufMultiSeg is defined in net_ib_cast/reg.cc.
ncclResult_t ncclIbRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle);
ncclResult_t IbCastRegMrDmaBufMultiSeg(void* comm, int nSeg, void** segAddrs, size_t* segLens, uint64_t* segOffsets,
                                       int* segFds, int type, void** mhandle);

#endif
