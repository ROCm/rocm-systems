/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the IB verbs wrappers defined by src/misc/ibvwrap.cc:
// symbol/device discovery, ports, QPs, async events, protection domains,
// memory registration (including DMA-BUF), and status formatting.
//
// Defaults are fail-loud for acquisitions (alloc_pd fails with *ret = NULL, the
// reg_mr entry points return NULL, matching the real wrappers on failure) and
// succeed for releases (dealloc_pd / dereg_mr), so an un-hooked call path can
// never report a resource it did not actually get.

#ifndef RCCL_TEST_HOST_IBVWRAP_FAKES_H_
#define RCCL_TEST_HOST_IBVWRAP_FAKES_H_

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "ScopedHook.h"
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

class ScopedDmaBufProbe {
 private:
  struct ibv_pd* pd_;
  int probeErrno_;

 public:
  ScopedDmaBufProbe(struct ibv_pd* pd, int probeErrno)
      : pd_(pd),
        probeErrno_(probeErrno),
        alloc(g_ibvAllocPd, [this](struct ibv_pd** out, struct ibv_context* context) {
          allocContext = context;
          *out = pd_;
          return ncclSuccess;
        }),
        probe(g_ibvDirectRegDmaBufMr,
              [this](struct ibv_pd* pd, uint64_t, size_t, uint64_t, int fd, int) -> struct ibv_mr* {
                probePd = pd;
                probeFd = fd;
                errno = probeErrno_;
                return nullptr;
              }),
        dealloc(g_ibvDeallocPd, [this](struct ibv_pd* pd) {
          deallocPd = pd;
          return ncclSuccess;
        }) {}

  struct ibv_context* allocContext = nullptr;
  struct ibv_pd* probePd = nullptr;
  int probeFd = 0;
  struct ibv_pd* deallocPd = nullptr;

  ScopedHook<ncclResult_t(struct ibv_pd**, struct ibv_context*)> alloc;
  ScopedHook<struct ibv_mr*(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int)> probe;
  ScopedHook<ncclResult_t(struct ibv_pd*)> dealloc;
};

void ResetIbvwrapFakes();

#endif  // RCCL_TEST_HOST_IBVWRAP_FAKES_H_
