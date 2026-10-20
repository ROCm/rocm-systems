/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the ionic direct-verbs wrappers defined by
// src/misc/ionicdvwrap.cc. Every default fails with ncclSystemError, the real
// wrappers' answer when libionic is missing.

#ifndef RCCL_TEST_HOST_IONICDVWRAP_FAKES_H_
#define RCCL_TEST_HOST_IONICDVWRAP_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"

struct ibv_pd;
struct ibv_qp;
struct ionic_dv_puec_route;

extern std::function<ncclResult_t()> g_ionicdvSymbols;
extern std::function<ncclResult_t(struct ibv_qp* /*ibqp*/, bool /*enable_send*/, bool /*enable_recv*/)>
    g_ionicdvQpSetGda;
extern std::function<ncclResult_t(struct ibv_pd* /*ibpd*/, uint8_t /*udma_mask*/)> g_ionicdvPdSetUdmaMask;
extern std::function<ncclResult_t(struct ibv_qp* /*qp*/, uint8_t /*plane_idx*/,
                                  struct ionic_dv_puec_route* /*route*/)>
    g_ionicdvQpSetPuecPlaneRoute;

void ResetIonicdvwrapFakes();

#endif  // RCCL_TEST_HOST_IONICDVWRAP_FAKES_H_
