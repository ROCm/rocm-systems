/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the IB verbs wrappers defined by src/misc/ibvwrap.cc.
// Only the entry points the units under test reach are faked; add more here as
// further units come under test.
//
// Defaults are fail-loud for acquisitions and queries (they fail with *ret =
// NULL, the direct reg_mr entry points return NULL, matching the real wrappers
// on failure) and succeed for releases (dealloc_pd / dereg_mr / destroy_*), so an
// un-hooked call path can never report a resource it did not actually get.

#ifndef RCCL_TEST_HOST_IBVWRAP_FAKES_H_
#define RCCL_TEST_HOST_IBVWRAP_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "ibvwrap.h"

extern std::function<ncclResult_t(struct ibv_pd** /*ret*/, struct ibv_context* /*context*/)> g_ibvAllocPd;
extern std::function<ncclResult_t(struct ibv_pd* /*pd*/)> g_ibvDeallocPd;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/, int /*access*/)>
    g_ibvDirectRegMr;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/, uint64_t /*iova*/,
                                    int /*access*/)>
    g_ibvDirectRegMrIova2;
extern std::function<ncclResult_t(struct ibv_mr* /*mr*/)> g_ibvDeregMr;
extern std::function<ncclResult_t(struct ibv_mr** /*ret*/, struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/,
                                  int /*access*/)>
    g_ibvRegMr;
extern std::function<ncclResult_t(struct ibv_mr** /*ret*/, struct ibv_pd* /*pd*/, uint64_t /*offset*/,
                                  size_t /*length*/, uint64_t /*iova*/, int /*fd*/, int /*access*/)>
    g_ibvRegDmabufMr;
extern std::function<ncclResult_t(struct ibv_cq** /*ret*/, struct ibv_context* /*context*/, int /*cqe*/,
                                  void* /*cq_context*/, struct ibv_comp_channel* /*channel*/, int /*comp_vector*/)>
    g_ibvCreateCq;
extern std::function<ncclResult_t(struct ibv_cq* /*cq*/)> g_ibvDestroyCq;
extern std::function<ncclResult_t(struct ibv_qp** /*ret*/, struct ibv_pd* /*pd*/,
                                  struct ibv_qp_init_attr* /*qp_init_attr*/)>
    g_ibvCreateQp;
extern std::function<ncclResult_t(struct ibv_qp* /*qp*/, struct ibv_qp_attr* /*attr*/, int /*attr_mask*/)>
    g_ibvModifyQp;
extern std::function<ncclResult_t(struct ibv_qp* /*qp*/)> g_ibvDestroyQp;
extern std::function<ncclResult_t(struct ibv_context* /*context*/, uint8_t /*port_num*/, int /*index*/,
                                  union ibv_gid* /*gid*/)>
    g_ibvQueryGid;
extern std::function<ncclResult_t(struct ibv_qp* /*qp*/, struct ibv_ece* /*ece*/, int* /*supported*/)> g_ibvQueryEce;
extern std::function<ncclResult_t(struct ibv_qp* /*qp*/, struct ibv_ece* /*ece*/, int* /*supported*/)> g_ibvSetEce;
extern std::function<const char*(struct ibv_device* /*device*/)> g_ibvGetDeviceName;

extern std::function<ncclResult_t()> g_ibvSymbols;
extern std::function<ncclResult_t()> g_ibvForkInit;
extern std::function<ncclResult_t(struct ibv_device*** /*ret*/, int* /*num_devices*/)> g_ibvGetDeviceList;
extern std::function<ncclResult_t(struct ibv_device** /*list*/)> g_ibvFreeDeviceList;
extern std::function<ncclResult_t(struct ibv_context** /*ret*/, struct ibv_device* /*device*/)> g_ibvOpenDevice;
extern std::function<ncclResult_t(struct ibv_context* /*context*/)> g_ibvCloseDevice;
extern std::function<ncclResult_t(struct ibv_context* /*context*/, struct ibv_async_event* /*event*/)>
    g_ibvGetAsyncEvent;
extern std::function<ncclResult_t(struct ibv_async_event* /*event*/)> g_ibvAckAsyncEvent;
extern std::function<ncclResult_t(char** /*ret*/, enum ibv_event_type /*event*/)> g_ibvEventTypeStr;
extern std::function<ncclResult_t(struct ibv_context* /*context*/, struct ibv_device_attr* /*device_attr*/)>
    g_ibvQueryDevice;
extern std::function<ncclResult_t(struct ibv_context* /*context*/, uint8_t /*port_num*/,
                                  struct ibv_port_attr* /*port_attr*/)>
    g_ibvQueryPort;
extern std::function<ncclResult_t(struct ibv_context* /*context*/, uint8_t /*port_num*/, uint64_t* /*speed*/)>
    g_ibvQueryPortSpeed;
extern std::function<ncclResult_t(struct ibv_qp* /*qp*/, struct ibv_qp_attr* /*attr*/, int /*attr_mask*/,
                                  struct ibv_qp_init_attr* /*init_attr*/)>
    g_ibvQueryQp;
extern std::function<ncclResult_t(struct ibv_mr** /*ret*/, struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/,
                                  uint64_t /*iova*/, int /*access*/)>
    g_ibvRegMrIova2;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, uint64_t /*offset*/, size_t /*length*/, uint64_t /*iova*/,
                                    int /*fd*/, int /*access*/)>
    g_ibvDirectRegDmabufMr;

void ResetIbvwrapFakes();

#endif  // RCCL_TEST_HOST_IBVWRAP_FAKES_H_
