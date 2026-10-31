/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The host an IB device-discovery pass sees: fake HCAs behind the ibv/mlx5dv wrappers, one IP interface, and every
// verbs library present. Shared by the net_ib and net_ib_cast init tests; each adds the hooks its own unit owns.
#pragma once

#include <arpa/inet.h>

#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/signature-drift.h"
#include "fakes/socket_fakes.h"

namespace {

template <typename Slot>
using HookOf = ScopedHook<rccl_test_host::FnSigOf_t<Slot>>;

// One fake HCA: the verbs objects device discovery opens and queries. Port n is ports[n - 1].
struct FakeHca {
  struct ibv_device device;
  struct ibv_context context;
  struct ibv_device_attr attr;
  std::vector<struct ibv_port_attr> ports;
  uint64_t portSpeed = 0;  // ibv_query_port_speed, in 100 Mb/s; net_ib_cast falls back to the port attributes on 0
  bool openFails = false;
  bool queryDeviceFails = false;
  bool mlx5 = false;
  int closes = 0;
};

class IbVerbsWorld {
 public:
  // An HCA with one active RoCE port at 100 Gb/s (EDR, 4x).
  FakeHca& AddHca(const char* name) {
    hcas.emplace_back();
    FakeHca& hca = hcas.back();
    memset(&hca.device, 0, sizeof(hca.device));
    snprintf(hca.device.name, sizeof(hca.device.name), "%s", name);
    snprintf(hca.device.dev_name, sizeof(hca.device.dev_name), "uverbs%zu", hcas.size() - 1);
    memset(&hca.context, 0, sizeof(hca.context));
    hca.context.device = &hca.device;
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
    for (FakeHca& hca : hcas) {
      if (&hca.context == ctx) return &hca;
    }
    return nullptr;
  }

  std::deque<FakeHca> hcas;
  std::vector<struct ibv_device*> list;
  // The maxIfs ncclFindInterfaces was last asked for.
  int findInterfacesMaxIfs = -1;

 private:
  std::vector<struct ibv_device*> listed_;

 public:
  HookOf<decltype(g_ibvSymbols)> ibvSymbols{g_ibvSymbols, [] { return ncclSuccess; }};
  HookOf<decltype(g_mlx5dvSymbols)> mlx5Symbols{g_mlx5dvSymbols, [] { return ncclSuccess; }};
  HookOf<decltype(g_ibvForkInit)> forkInit{g_ibvForkInit, [] { return ncclSuccess; }};
  HookOf<decltype(g_ncclFindInterfaces)> findInterfaces{
      g_ncclFindInterfaces, [this](char* names, union ncclSocketAddress* addrs, int nameSize, int maxIfs, int* nIfs) {
        findInterfacesMaxIfs = maxIfs;
        snprintf(names, nameSize, "eth0");
        memset(addrs, 0, sizeof(*addrs));
        addrs->sin.sin_family = AF_INET;
        addrs->sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        *nIfs = 1;
        return ncclSuccess;
      }};
  HookOf<decltype(g_ncclSocketToString)> socketToString{
      g_ncclSocketToString, [](const union ncclSocketAddress*, char* buf, size_t, int) {
        buf[0] = '\0';
        return static_cast<const char*>(buf);
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
                                                          for (FakeHca& hca : hcas) {
                                                            if (&hca.device == d) return hca.mlx5;
                                                          }
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
                                               if (hca == nullptr || portNum < 1 || portNum > hca->ports.size()) {
                                                 return ncclSystemError;
                                               }
                                               *attr = hca->ports[portNum - 1];
                                               return ncclSuccess;
                                             }};
  HookOf<decltype(g_ibvQueryPortSpeed)> queryPortSpeed{
      g_ibvQueryPortSpeed, [this](struct ibv_context* ctx, uint8_t, uint64_t* speed) {
        FakeHca* hca = Find(ctx);
        *speed = hca ? hca->portSpeed : 0;
        return ncclSuccess;
      }};
  // Relaxed ordering needs IBVERBS_1.8; this library lacks it unless a test says otherwise.
  HookOf<decltype(g_ibvRegMrIova2)> regMrIova2{
      g_ibvRegMrIova2, [](struct ibv_mr**, struct ibv_pd*, void*, size_t, uint64_t, int) { return ncclInternalError; }};
};

}  // namespace
