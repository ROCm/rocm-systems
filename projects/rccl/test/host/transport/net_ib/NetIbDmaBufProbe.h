/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Scripted DMA-BUF capability probe for the src/transport/net_ib/{gdr,init}.cc
 * tests: alloc_pd hands out a fixed PD, the fd=-1 ibv_reg_dmabuf_mr probe fails
 * with the chosen errno, and every call records what it was given.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_DMA_BUF_PROBE_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_DMA_BUF_PROBE_H_

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"

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
        probe(g_ibvDirectRegDmabufMr,
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

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_DMA_BUF_PROBE_H_
