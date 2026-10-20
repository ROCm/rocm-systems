/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared by the rccl-UnitTestsMicroNetIbCastInit test TUs: param and comm helpers, fake verbs
// objects whose context ops call back into the test, and the reset of the transport's globals.
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
// IbCastInitDevices starts one async-event thread per device it adds; this counts them.
extern std::atomic<int> g_netIbCastAsyncThreadsStarted;

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

// Zeroed heap storage for the large comm structs, as the transport's own calloc gives them.
template <typename T>
struct ZeroedPtr {
  ZeroedPtr() : p(static_cast<T*>(calloc(1, sizeof(T)))) {}
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

template <typename Slot>
using HookOf = ScopedHook<rccl_test_host::FnSigOf_t<Slot>>;

// One fake HCA: the verbs objects IbCastInitDevices opens and queries. Port n is ports[n - 1].
struct FakeHca {
  struct ibv_device device;
  struct ibv_context context;
  struct ibv_device_attr attr;
  std::vector<struct ibv_port_attr> ports;
  uint64_t portSpeed = 0;  // ibv_query_port_speed, in 100 Mb/s; 0 falls back to the port attributes
  bool openFails = false;
  bool queryDeviceFails = false;
  bool mlx5 = false;
  int closes = 0;
};

// A host with HCAs, one IP interface and every library present, as IbCastInitDevices sees it.
// The async-event threads it starts end on their first event read; the destructor waits for them,
// since they read hooks and IbCastDevs that the test is about to reset.
class FakeIbWorld {
 public:
  FakeIbWorld() : asyncThreadsBefore_(g_netIbCastAsyncThreadsStarted.load()) {}
  ~FakeIbWorld() { Quiesce(); }

  // Waits until every async-event thread started so far has read its event and left.
  void Quiesce() {
    const int started = g_netIbCastAsyncThreadsStarted.load() - asyncThreadsBefore_;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (asyncExits_.load() < started && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    EXPECT_EQ(asyncExits_.load(), started) << "async-event threads still running";
  }

  // An HCA with one active RoCE port at 100 Gb/s (EDR, 4x).
  FakeHca& AddHca(const char* name) {
    hcas.emplace_back();
    FakeHca& hca = hcas.back();
    memset(&hca.device, 0, sizeof(hca.device));
    snprintf(hca.device.name, sizeof(hca.device.name), "%s", name);
    snprintf(hca.device.dev_name, sizeof(hca.device.dev_name), "uverbs%zu", hcas.size() - 1);
    InitFakeContext(&hca.context, &hca.device);
    memset(&hca.attr, 0, sizeof(hca.attr));
    hca.attr.sys_image_guid = 0x1000 + hcas.size();
    hca.attr.max_qp = 1024;
    hca.attr.max_cqe = 4096;
    hca.ports.push_back(ActivePort(IBV_LINK_LAYER_ETHERNET));
    list.push_back(&hca.device);
    return hca;
  }
  static struct ibv_port_attr ActivePort(uint8_t linkLayer) {
    struct ibv_port_attr port;
    memset(&port, 0, sizeof(port));
    port.state = IBV_PORT_ACTIVE;
    port.link_layer = linkLayer;
    port.active_speed = 32;  // EDR
    port.active_width = 2;   // 4x
    port.active_mtu = IBV_MTU_4096;
    port.lid = 7;
    return port;
  }
  FakeHca* Find(struct ibv_context* ctx) {
    for (FakeHca& hca : hcas)
      if (&hca.context == ctx) return &hca;
    return nullptr;
  }

  std::deque<FakeHca> hcas;
  std::vector<struct ibv_device*> list;

 private:
  int asyncThreadsBefore_;
  std::atomic<int> asyncExits_{0};
  std::vector<struct ibv_device*> listed_;

 public:
  HookOf<decltype(g_ibvSymbols)> ibvSymbols{g_ibvSymbols, [] { return ncclSuccess; }};
  HookOf<decltype(g_mlx5dvSymbols)> mlx5Symbols{g_mlx5dvSymbols, [] { return ncclSuccess; }};
  HookOf<decltype(g_ionicdvSymbols)> ionicSymbols{g_ionicdvSymbols, [] { return ncclSuccess; }};
  HookOf<decltype(g_ibvForkInit)> forkInit{g_ibvForkInit, [] { return ncclSuccess; }};
  HookOf<decltype(g_ncclFindInterfaces)> findInterfaces{
      g_ncclFindInterfaces, [](char* names, union ncclSocketAddress* addrs, int nameSize, int, int* nIfs) {
        snprintf(names, nameSize, "eth0");
        memset(addrs, 0, sizeof(*addrs));
        addrs->sin.sin_family = AF_INET;
        addrs->sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        *nIfs = 1;
        return ncclSuccess;
      }};
  // Like ibv_get_device_list: a NULL-terminated array, non-NULL even when there is no HCA.
  HookOf<decltype(g_ibvGetDeviceList)> getDeviceList{g_ibvGetDeviceList, [this](struct ibv_device*** ret, int* n) {
                                                          listed_ = list;
                                                          listed_.push_back(nullptr);
                                                          *ret = listed_.data();
                                                          *n = (int)list.size();
                                                          return ncclSuccess;
                                                        }};
  HookOf<decltype(g_ibvFreeDeviceList)> freeDeviceList{g_ibvFreeDeviceList,
                                                        [](struct ibv_device**) { return ncclSuccess; }};
  HookOf<decltype(g_ibvOpenDevice)> openDevice{g_ibvOpenDevice, [this](struct ibv_context** ret, struct ibv_device* d) {
                                                 for (FakeHca& hca : hcas) {
                                                   if (&hca.device != d) continue;
                                                   *ret = hca.openFails ? nullptr : &hca.context;
                                                   return hca.openFails ? ncclSystemError : ncclSuccess;
                                                 }
                                                 return ncclSystemError;
                                               }};
  HookOf<decltype(g_ibvCloseDevice)> closeDevice{g_ibvCloseDevice, [this](struct ibv_context* ctx) {
                                                   if (FakeHca* hca = Find(ctx)) hca->closes++;
                                                   return ncclSuccess;
                                                 }};
  HookOf<decltype(g_mlx5dvIsSupported)> mlx5IsSupported{g_mlx5dvIsSupported, [this](struct ibv_device* d) {
                                                          for (FakeHca& hca : hcas)
                                                            if (&hca.device == d) return hca.mlx5;
                                                          return false;
                                                        }};
  HookOf<decltype(g_ibvQueryDevice)> queryDevice{g_ibvQueryDevice, [this](struct ibv_context* ctx,
                                                                          struct ibv_device_attr* attr) {
                                                   FakeHca* hca = Find(ctx);
                                                   if (hca == nullptr || hca->queryDeviceFails) return ncclSystemError;
                                                   *attr = hca->attr;
                                                   attr->phys_port_cnt = (uint8_t)hca->ports.size();
                                                   return ncclSuccess;
                                                 }};
  HookOf<decltype(g_ibvQueryPort)> queryPort{g_ibvQueryPort, [this](struct ibv_context* ctx, uint8_t portNum,
                                                                    struct ibv_port_attr* attr) {
                                               FakeHca* hca = Find(ctx);
                                               if (hca == nullptr || portNum < 1 || portNum > hca->ports.size())
                                                 return ncclSystemError;
                                               *attr = hca->ports[portNum - 1];
                                               return ncclSuccess;
                                             }};
  HookOf<decltype(g_ibvQueryPortSpeed)> queryPortSpeed{
      g_ibvQueryPortSpeed, [this](struct ibv_context* ctx, uint8_t, uint64_t* speed) {
        FakeHca* hca = Find(ctx);
        *speed = hca ? hca->portSpeed : 0;
        return ncclSuccess;
      }};
  HookOf<decltype(g_ibCastGidInfoQuery)> gidInfoQuery{
      g_ibCastGidInfoQuery, [](struct ibv_context*, uint8_t, struct ibv_port_attr* port, struct ncclIbGidInfo* info) {
        info->link_layer = port->link_layer;
        info->localGidIndex = 3;
        return ncclSuccess;
      }};
  // Relaxed ordering needs IBVERBS_1.8; this library lacks it unless a test says otherwise.
  HookOf<decltype(g_ibvRegMrIova2)> regMrIova2{
      g_ibvRegMrIova2, [](struct ibv_mr**, struct ibv_pd*, void*, size_t, uint64_t, int) { return ncclInternalError; }};
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
