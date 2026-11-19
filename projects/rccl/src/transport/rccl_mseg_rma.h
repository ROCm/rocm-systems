/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_RMA_H_
#define RCCL_TRANSPORT_MSEG_RMA_H_

#include <stddef.h>
#include <stdint.h>

#include "rccl_mseg.h"

// RCCL: multi-segment GIN/RMA limits shared by the RMA plugins and the device runtime.

// RCCL: every rank holds every rank's RMA segment table (48 bytes per entry), so RMA bounds ranks x segments.
#define RCCL_RMA_SEG_TABLE_MAX_ENTRIES (1 << 20)
static inline bool rcclRmaSegTableFits(int nranks, int nSeg) {
  if (nranks < 1 || !rcclIbSegCountSupported(nSeg)) return false;
  return nSeg == 1 || (int64_t)nranks * nSeg <= RCCL_RMA_SEG_TABLE_MAX_ENTRIES;
}

// With multi-segment on, the RMA and GIN proxies keep at most this many requests in flight per peer and context.
#define RCCL_RMA_MAX_INFLIGHT_REQUESTS 32

// Requests a proxy keeps in flight per peer: the cap when the backend takes multi-segment chains, else its own limit.
static inline uint32_t rcclRmaInflightRequestsCap(bool multiSegChains, uint32_t maxRequests) {
  return multiSegChains && maxRequests > RCCL_RMA_MAX_INFLIGHT_REQUESTS ? RCCL_RMA_MAX_INFLIGHT_REQUESTS : maxRequests;
}

#endif // RCCL_TRANSPORT_MSEG_RMA_H_
