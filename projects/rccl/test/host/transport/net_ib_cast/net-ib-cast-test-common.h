/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared by net-ib-cast-{capability,common,gdr,gid-inspect,gin,init,multiplane,reg}-test.cc: param and comm
// helpers, fake verbs objects whose context ops call back into the test, and the reset of the transport's globals.
#pragma once

#include <gtest/gtest.h>

#include <arpa/inet.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "IbVerbsWorld.h"
#include "ScopedHook.h"
#include "common_cast.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/ionicdvwrap_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/net_ib_cast_connect_fakes.h"
#include "fakes/net_ib_cast_p2p_resiliency_recovery_fakes.h"
#include "fakes/net_ib_cast_scheduler_fakes.h"
#include "fakes/signature-drift.h"
#include "fakes/socket_fakes.h"

// Puts every net_ib_cast global back to its load-time value. Defined in
// net-ib-cast-init-test.cc, the TU that can reach init.cc's file-scope state.
void NetIbCastResetGlobals();

namespace {

// Params reach the units through param_redirect.h's g_loadParam; one not Set() keeps its default.
class ParamScript {
 public:
  void Set(const char* name, int64_t value) { values_[name] = value; }

 private:
  std::map<std::string, int64_t> values_;
  ScopedHook<int64_t(const char*, int64_t)> hook_{g_loadParam, [this](const char* name, int64_t deftVal) {
                                                     auto it = values_.find(name);
                                                     return it == values_.end() ? deftVal : it->second;
                                                   }};
};

// Zeroed heap storage for the large comm structs, aligned as their alignas() asks (calloc only gives 16).
template <typename T>
struct ZeroedPtr {
  ZeroedPtr() : p(static_cast<T*>(aligned_alloc(alignof(T), (sizeof(T) + alignof(T) - 1) / alignof(T) * alignof(T)))) {
    memset(p, 0, sizeof(T));
  }
  ~ZeroedPtr() { free(p); }
  ZeroedPtr(const ZeroedPtr&) = delete;
  ZeroedPtr& operator=(const ZeroedPtr&) = delete;
  T* operator->() const { return p; }
  T* get() const { return p; }
  T* p;
};

// wrap_ibv_post_send/poll_cq/post_recv are inline and call through ibv_context::ops, so a
// fake context routes them here.
std::function<int(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**)> g_fakePostSend =
    [](struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) { return EIO; };
std::function<int(struct ibv_cq*, int, struct ibv_wc*)> g_fakePollCq = [](struct ibv_cq*, int, struct ibv_wc*) {
  return -1;
};
std::function<int(struct ibv_qp*, struct ibv_recv_wr*, struct ibv_recv_wr**)> g_fakePostRecv =
    [](struct ibv_qp*, struct ibv_recv_wr*, struct ibv_recv_wr**) { return EIO; };

inline void InitFakeContext(struct ibv_context* ctx, struct ibv_device* dev) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->device = dev;
  ctx->ops.post_send = [](struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad) {
    return g_fakePostSend(qp, wr, bad);
  };
  ctx->ops.poll_cq = [](struct ibv_cq* cq, int n, struct ibv_wc* wc) { return g_fakePollCq(cq, n, wc); };
  ctx->ops.post_recv = [](struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad) {
    return g_fakePostRecv(qp, wr, bad);
  };
}

// IbVerbsWorld as IbCastInitDevices sees it. The async-event threads it starts end on their first event read; the
// destructor waits for them, since they read hooks and IbCastDevs that the test is about to reset.
class FakeIbWorld : public IbVerbsWorld {
 public:
  ~FakeIbWorld() { Quiesce(); }

  // Waits until every async-event thread started so far has read its event and left.
  void Quiesce() {
    const int started = setThreadName.calls.load();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (asyncExits_.load() < started && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    EXPECT_EQ(asyncExits_.load(), started) << "async-event threads still running";
  }

  // IbVerbsWorld's HCA, with context ops that reach the g_fake* hooks above.
  FakeHca& AddHca(const char* name) {
    FakeHca& hca = IbVerbsWorld::AddHca(name);
    InitFakeContext(&hca.context, &hca.device);
    return hca;
  }

  // IbCastInitDevices names each async-event thread it starts, so the hook also counts them.
  std::vector<std::string> threadNames;

 private:
  std::atomic<int> asyncExits_{0};

 public:
  HookOf<decltype(g_ncclSetThreadName)> setThreadName{
      g_ncclSetThreadName, [this](pthread_t, const char* name) { threadNames.push_back(name); }};
  HookOf<decltype(g_ionicdvSymbols)> ionicSymbols{g_ionicdvSymbols, [] { return ncclSuccess; }};
  HookOf<decltype(g_ibCastGidInfoQuery)> gidInfoQuery{
      g_ibCastGidInfoQuery, [](struct ibv_context*, uint8_t, struct ibv_port_attr* port, struct ncclIbGidInfo* info) {
        info->link_layer = port->link_layer;
        info->localGidIndex = 3;
        return ncclSuccess;
      }};
  HookOf<decltype(g_ibvGetAsyncEvent)> asyncEvent{g_ibvGetAsyncEvent, [this](struct ibv_context*, struct ibv_async_event*) {
                                                    asyncExits_++;
                                                    return ncclSystemError;
                                                  }};
  HookOf<decltype(g_ibCastQpSchedInitParms)> schedInitParms{g_ibCastQpSchedInitParms, [](struct ncclIbQpSchedParms* p) {
                                                              memset(p, 0, sizeof(*p));
                                                              return ncclSuccess;
                                                            }};
  HookOf<decltype(g_rcclUseIbCastQpSched)> useQpSched{g_rcclUseIbCastQpSched, [] { return false; }};
  HookOf<decltype(g_ibCastPortRecoveryThreadStart)> recoveryStart{g_ibCastPortRecoveryThreadStart,
                                                                  [] { return ncclSuccess; }};
  HookOf<decltype(g_ibCastPortRecoveryThreadStop)> recoveryStop{g_ibCastPortRecoveryThreadStop,
                                                                [] { return ncclSuccess; }};
};

}  // namespace
