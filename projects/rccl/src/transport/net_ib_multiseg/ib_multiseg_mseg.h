/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_
#define RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_

#include <stdint.h>

#include "nccl.h"

// RCCL: segmented-builder counters per send comm, for tests. laterSegPosts counts posts that addressed a segment
// past the first on either side, which only a working segment walk can do.
struct rcclIbSegStats {
  uint64_t posts;
  uint64_t wrs;
  uint64_t laterSegPosts;
};
ncclResult_t rcclIbSendCommSegStats(void* sendComm, struct rcclIbSegStats* out);

// RCCL: one connect-time decision. multiSegOn is the core's fd -1 gate, so the CTS follows registration; GIN has no
// CTS. Resiliency replays a whole slot, which a send resumed across tests cannot do yet.
static inline bool IbMsegUseMultiSeg(bool multiSegOn, bool forGin, bool resilient) {
  return multiSegOn && !forGin && !resilient;
}

#endif // RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_
