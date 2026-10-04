/*************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CAST_MULTIPLANE_H_
#define NET_IB_CAST_MULTIPLANE_H_

#include "nccl.h"
#include "ibvwrap.h"
#include "graph/xml.h"

#define MULTIPLANE_MAX_PIPS 8

struct ncclIbPipInfo {
  char ip[MAX_STR_LEN];
  char interface[MAX_STR_LEN];
  union ibv_gid gid;  // pre-parsed at load time; avoids repeated inet_pton per QP RTR
};

// Load and parse the multiplane VIP-to-PIP mapping file (idempotent).
ncclResult_t IbCastMultiplaneLoad(void);

// Check whether multiplane routing is configured (RCCL_MULTIPLANE_MAP_FILE set).
ncclResult_t IbCastMultiplaneEnabled(bool* enabled);

// Look up PIP GIDs for a given VIP GID.  On return, pipGids[0..(*nPips)-1]
// are filled and *nPips is set.  If the VIP is not in the map, *nPips == 0.
ncclResult_t IbCastMultiplaneGetPipGids(const union ibv_gid* vipGid, union ibv_gid* pipGids, int* nPips);

// Whether to replace the QP dgid with the local GID (loopback) on multi-plane
// QPs. False only when a verbs shim (libmrcshim) defines
// mrc_multiplane_override_dgid() returning false.
bool IbCastMultiplaneOverrideDgid(void);

#endif  // NET_IB_CAST_MULTIPLANE_H_
