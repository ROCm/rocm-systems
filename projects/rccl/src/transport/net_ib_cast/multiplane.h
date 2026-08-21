/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CAST_MULTIPLANE_H_
#define NET_IB_CAST_MULTIPLANE_H_

#include "nccl.h"
#include "ibvwrap.h"
#include "graph/xml.h"

#define MULTIPLANE_MAX_PIPS 16

struct ncclIbPipInfo {
  char ip[MAX_STR_LEN];
  char interface[MAX_STR_LEN];
};

// Load and parse the multiplane VIP-to-PIP mapping file (idempotent).
ncclResult_t ibCastMultiplaneLoad(void);

// Check whether multiplane routing is configured (RCCL_MULTIPLANE_MAP_FILE set).
ncclResult_t ibCastMultiplaneEnabled(bool* enabled);

// Look up PIP GIDs for a given VIP GID.  On return, pipGids[0..(*nPips)-1]
// are filled and *nPips is set.  If the VIP is not in the map, *nPips == 0.
ncclResult_t ibCastMultiplaneGetPipGids(const union ibv_gid* vipGid, union ibv_gid* pipGids, int* nPips);

#endif  // NET_IB_CAST_MULTIPLANE_H_
