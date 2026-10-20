/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the IB verbs wrappers defined by src/misc/ibvwrap.cc.
// Only the protection-domain and memory-registration entry points are faked;
// add more here as further units come under test.
//
// Defaults are fail-loud for acquisitions (alloc_pd fails with *ret = NULL, the
// reg_mr entry points return NULL, matching the real wrappers on failure) and
// succeed for releases (dealloc_pd / dereg_mr), so an un-hooked call path can
// never report a resource it did not actually get.

#ifndef RCCL_TEST_HOST_IBVWRAP_FAKES_H_
#define RCCL_TEST_HOST_IBVWRAP_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "ibvwrap.h"
#include "nccl.h"

extern std::function<ncclResult_t()> g_ibvSymbols;
extern std::function<ncclResult_t()> g_ibvForkInit;
extern std::function<ncclResult_t(struct ibv_device***, int*)> g_ibvGetDeviceList;
extern std::function<ncclResult_t(struct ibv_device**)> g_ibvFreeDeviceList;
extern std::function<ncclResult_t(struct ibv_context**, struct ibv_device*)> g_ibvOpenDevice;
extern std::function<ncclResult_t(struct ibv_context*)> g_ibvCloseDevice;
extern std::function<ncclResult_t(struct ibv_context*, struct ibv_async_event*)> g_ibvGetAsyncEvent;
extern std::function<ncclResult_t(struct ibv_async_event*)> g_ibvAckAsyncEvent;
extern std::function<ncclResult_t(struct ibv_context*, struct ibv_device_attr*)> g_ibvQueryDevice;
extern std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*)> g_ibvQueryPort;
extern std::function<ncclResult_t(struct ibv_qp*, struct ibv_qp_attr*, int, struct ibv_qp_init_attr*)> g_ibvQueryQp;

extern std::function<ncclResult_t(struct ibv_pd** /*ret*/, struct ibv_context* /*context*/)> g_ibvAllocPd;
extern std::function<ncclResult_t(struct ibv_pd* /*pd*/)> g_ibvDeallocPd;
extern std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, void*, size_t, int)> g_ibvRegMr;
extern std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, void*, size_t, uint64_t, int)> g_ibvRegMrIova2;
extern std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int)> g_ibvRegDmaBufMr;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/, int /*access*/)>
    g_ibvDirectRegMr;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, void* /*addr*/, size_t /*length*/, uint64_t /*iova*/,
                                    int /*access*/)>
    g_ibvDirectRegMrIova2;
extern std::function<struct ibv_mr*(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int)> g_ibvDirectRegDmaBufMr;
extern std::function<ncclResult_t(struct ibv_mr* /*mr*/)> g_ibvDeregMr;
extern std::function<ncclResult_t(struct ibv_qp*, struct ibv_qp_attr*, int)> g_ibvModifyQp;
extern std::function<ncclResult_t(struct ibv_context*, uint8_t, uint64_t*)> g_ibvQueryPortSpeed;
extern std::function<ncclResult_t(char**, enum ibv_event_type)> g_ibvEventTypeStr;

void ResetIbvwrapFakes();

#endif  // RCCL_TEST_HOST_IBVWRAP_FAKES_H_
