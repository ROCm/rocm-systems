/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/ionicdvwrap.cc fakes. See ionicdvwrap_fakes.h.

#include "ionicdvwrap_fakes.h"

#include "ionic/ionicdvwrap.h"
#include "signature-drift.h"

static ncclResult_t DefaultIonicdvSymbols() { return ncclSystemError; }
std::function<ncclResult_t()> g_ionicdvSymbols = DefaultIonicdvSymbols;

static ncclResult_t DefaultIonicdvQpSetGda(struct ibv_qp*, bool, bool) { return ncclSystemError; }
std::function<ncclResult_t(struct ibv_qp*, bool, bool)> g_ionicdvQpSetGda = DefaultIonicdvQpSetGda;

static ncclResult_t DefaultIonicdvPdSetUdmaMask(struct ibv_pd*, uint8_t) { return ncclSystemError; }
std::function<ncclResult_t(struct ibv_pd*, uint8_t)> g_ionicdvPdSetUdmaMask = DefaultIonicdvPdSetUdmaMask;

static ncclResult_t DefaultIonicdvQpSetPuecPlaneRoute(struct ibv_qp*, uint8_t, struct ionic_dv_puec_route*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_qp*, uint8_t, struct ionic_dv_puec_route*)> g_ionicdvQpSetPuecPlaneRoute =
    DefaultIonicdvQpSetPuecPlaneRoute;

ASSERT_HOOK_MATCHES_PROD(g_ionicdvSymbols, wrap_ionicdv_symbols);
ASSERT_HOOK_MATCHES_PROD(g_ionicdvQpSetGda, wrap_ionicdv_qp_set_gda);
ASSERT_HOOK_MATCHES_PROD(g_ionicdvPdSetUdmaMask, wrap_ionicdv_pd_set_udma_mask);
ASSERT_HOOK_MATCHES_PROD(g_ionicdvQpSetPuecPlaneRoute, wrap_ionicdv_qp_set_puec_plane_route);

ncclResult_t wrap_ionicdv_symbols(void) { return g_ionicdvSymbols(); }
ncclResult_t wrap_ionicdv_qp_set_gda(struct ibv_qp* ibqp, bool enable_send, bool enable_recv) {
  return g_ionicdvQpSetGda(ibqp, enable_send, enable_recv);
}
ncclResult_t wrap_ionicdv_pd_set_udma_mask(struct ibv_pd* ibpd, uint8_t udma_mask) {
  return g_ionicdvPdSetUdmaMask(ibpd, udma_mask);
}
ncclResult_t wrap_ionicdv_qp_set_puec_plane_route(struct ibv_qp* qp, uint8_t plane_idx,
                                                  struct ionic_dv_puec_route* route) {
  return g_ionicdvQpSetPuecPlaneRoute(qp, plane_idx, route);
}

void ResetIonicdvwrapFakes() {
  g_ionicdvSymbols = DefaultIonicdvSymbols;
  g_ionicdvQpSetGda = DefaultIonicdvQpSetGda;
  g_ionicdvPdSetUdmaMask = DefaultIonicdvPdSetUdmaMask;
  g_ionicdvQpSetPuecPlaneRoute = DefaultIonicdvQpSetPuecPlaneRoute;
}
