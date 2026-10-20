/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only fixture builders shared by src/transport/rma_socket*.cc tests.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_RMA_SOCKET_TEST_H_
#define RCCL_TEST_HOST_TRANSPORT_RMA_SOCKET_TEST_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "transport/rma_socket.h"

namespace rma_socket_test {

constexpr int kRanks = 2;
constexpr int kQueueCapacity = 4;

struct Scene {
  explicit Scene(int nranks = kRanks, int nContexts = 1, int queueCapacity = kQueueCapacity)
      : senders(nranks),
        receivers(nranks),
        taskStorage(nranks),
        contexts(nContexts),
        requestStorage(nContexts) {
    comm.rank = 0;
    comm.nranks = nranks;
    comm.peerSender = senders.data();
    comm.peerReceiver = receivers.data();
    comm.nContexts = nContexts;
    comm.lastError = ncclSuccess;

    for (int peer = 0; peer < nranks; ++peer) {
      taskStorage[peer].resize(queueCapacity);
      senders[peer].tasks = taskStorage[peer].data();
      senders[peer].capacity = queueCapacity;
      senders[peer].lastCompletedRequestGlobalId = UINT64_MAX;
      receivers[peer].lastCompletedRequestGlobalId = UINT64_MAX;
    }

    const int requestCount = NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER * nranks;
    for (int c = 0; c < nContexts; ++c) {
      requestStorage[c].resize(requestCount);
      contexts[c].collComm = &comm;
      contexts[c].nContexts = nContexts;
      contexts[c].requests = requestStorage[c].data();
      contexts[c].requestCount = requestCount;
    }
  }

  ncclRmaSocketProxyMrHandle MakeMr(void* data, size_t size, int type, uint64_t id) {
    ncclRmaSocketProxyMrHandle handle{};
    handle.data = data;
    handle.size = size;
    handle.type = type;
    handle.mrId = id;
    return handle;
  }

  void LinkMrs(std::vector<ncclRmaSocketProxyMrHandle>& handles) {
    for (size_t i = 0; i < handles.size(); ++i) {
      handles[i].next = i + 1 < handles.size() ? &handles[i + 1] : nullptr;
    }
    comm.mrHead = handles.empty() ? nullptr : &handles[0];
  }

  ncclRmaSocketProxyCollComm comm{};
  std::vector<ncclRmaSocketProxyPeerSender> senders;
  std::vector<ncclRmaSocketProxyPeerReceiver> receivers;
  std::vector<std::vector<ncclRmaSocketProxySendTask>> taskStorage;
  std::vector<ncclRmaSocketProxyCtx> contexts;
  std::vector<std::vector<ncclRmaSocketProxyRequest>> requestStorage;
};

}  // namespace rma_socket_test

#endif  // RCCL_TEST_HOST_TRANSPORT_RMA_SOCKET_TEST_H_
