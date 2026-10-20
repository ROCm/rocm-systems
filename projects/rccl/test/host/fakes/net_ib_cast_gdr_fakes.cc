/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/gdr.cc fakes. See
// net_ib_cast_gdr_fakes.h.

#include "net_ib_cast_gdr_fakes.h"

#include "common_cast.h"
#include "signature-drift.h"

static ncclResult_t DefaultIbCastGdrSupport() { return ncclSystemError; }
std::function<ncclResult_t()> g_ibCastGdrSupport = DefaultIbCastGdrSupport;

static ncclResult_t DefaultIbCastPeerMemSupport() { return ncclSystemError; }
std::function<ncclResult_t()> g_ibCastPeerMemSupport = DefaultIbCastPeerMemSupport;

static ncclResult_t DefaultIbCastDmaBufSupport(int) { return ncclSystemError; }
std::function<ncclResult_t(int)> g_ibCastDmaBufSupport = DefaultIbCastDmaBufSupport;

ASSERT_HOOK_MATCHES_PROD(g_ibCastGdrSupport, IbCastGdrSupport);
ASSERT_HOOK_MATCHES_PROD(g_ibCastPeerMemSupport, IbCastPeerMemSupport);
ASSERT_HOOK_MATCHES_PROD(g_ibCastDmaBufSupport, IbCastDmaBufSupport);

ncclResult_t IbCastGdrSupport() { return g_ibCastGdrSupport(); }
ncclResult_t IbCastPeerMemSupport() { return g_ibCastPeerMemSupport(); }
ncclResult_t IbCastDmaBufSupport(int dev) { return g_ibCastDmaBufSupport(dev); }

void ResetNetIbCastGdrFakes() {
  g_ibCastGdrSupport = DefaultIbCastGdrSupport;
  g_ibCastPeerMemSupport = DefaultIbCastPeerMemSupport;
  g_ibCastDmaBufSupport = DefaultIbCastDmaBufSupport;
}
