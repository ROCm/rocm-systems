/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/connect.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <arpa/inet.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <set>
#include <vector>

#include "IbVerbsScene.h"
#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/os_fakes.h"
#include "fakes/socket_fakes.h"
#include "fakes/transport_net_ib_fakes.h"
#include "fakes/transport_net_ib_resiliency_fakes.h"
#include "fakes/transport_net_ib_wqe_lat_fakes.h"
#include "transport/net_ib/NetIbGlobals.h"
#include "transport/net_ib/NetIbSupportSeams.h"

namespace {

std::function<int(const char*, int)> g_connectOpen = [](const char*, int) { return -1; };
std::function<ssize_t(int, void*, size_t)> g_connectRead = [](int, void*, size_t) { return ssize_t{-1}; };
std::function<int(int)> g_connectClose = [](int) { return 0; };
std::function<ncclResult_t(void**, size_t)> g_connectMemAlloc = [](void** ptr, size_t) {
  if (ptr) *ptr = nullptr;
  return ncclSystemError;
};

int TestConnectOpen(const char* path, int flags) { return g_connectOpen(path, flags); }
ssize_t TestConnectRead(int fd, void* data, size_t size) { return g_connectRead(fd, data, size); }
int TestConnectClose(int fd) { return g_connectClose(fd); }
ncclResult_t TestConnectMemAlloc(void** ptr, size_t size) { return g_connectMemAlloc(ptr, size); }

}  // namespace

// Keep the generic owner fakes linked for sibling transport tests while this
// TU compiles the real connect.cc under unit-qualified names.
#define ncclParamIbQpsPerConn NetIbConnectParamIbQpsPerConn
#define ncclIbGidInfoQuery NetIbConnectGidInfoQuery
#define ncclIbListen NetIbConnectListen
#define ncclIbNetListen NetIbConnectNetListen
#define ncclIbGdrFlushFlagLocalSupport NetIbConnectGdrFlushFlagLocalSupport
#define ncclIbSetGdrFlushFlag NetIbConnectSetGdrFlushFlag
#define ncclIbConnect NetIbConnectConnect
#define ncclIbAccept NetIbConnectAccept
#define ncclIbCloseSend NetIbConnectCloseSend
#define ncclIbCloseRecv NetIbConnectCloseRecv
#define ncclIbCloseListen NetIbConnectCloseListen
#define open TestConnectOpen
#define read TestConnectRead
#define close TestConnectClose
#define ncclIbGdrSupport NetIbTestGdrSupport
#define ncclIbDmaBufSupport NetIbTestDmaBufSupport
#define ncclMemAlloc TestConnectMemAlloc
#include NET_IB_CONNECT_CC_PATH
#undef ncclMemAlloc
#undef ncclIbDmaBufSupport
#undef ncclIbGdrSupport
#undef close
#undef read
#undef open
#undef ncclIbCloseListen
#undef ncclIbCloseRecv
#undef ncclIbCloseSend
#undef ncclIbAccept
#undef ncclIbConnect
#undef ncclIbSetGdrFlushFlag
#undef ncclIbGdrFlushFlagLocalSupport
#undef ncclIbNetListen
#undef ncclIbListen
#undef ncclIbGidInfoQuery
#undef ncclParamIbQpsPerConn

namespace {

union ibv_gid Ipv4Gid(const char* address) {
  union ibv_gid gid{};
  gid.raw[10] = 0xff;
  gid.raw[11] = 0xff;
  EXPECT_EQ(inet_pton(AF_INET, address, gid.raw + 12), 1);
  return gid;
}

union ibv_gid Ipv6Gid(const char* address) {
  union ibv_gid gid{};
  EXPECT_EQ(inet_pton(AF_INET6, address, gid.raw), 1);
  return gid;
}

// The connection metadata a peer with one RoCE device sends.
ncclIbConnectionMetadata PeerMetadata(uint64_t ctsAddr, uint32_t qpn, uint16_t lid, uint32_t rkey,
                                      const ibv_gid& gid) {
  ncclIbConnectionMetadata meta{};
  meta.ndevs = 1;
  meta.isP2p = 1;
  meta.addr = ctsAddr;
  meta.devs[0].ib_port = 1;
  meta.devs[0].mtu = IBV_MTU_4096;
  meta.devs[0].lid = lid;
  meta.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  meta.devs[0].gid = gid;
  meta.devs[0].rkey = rkey;
  meta.devs[0].currSpeed = 100000;
  meta.qpInfo[0].qpn = qpn;
  meta.qpInfo[0].devIndex = 0;
  return meta;
}

// The peer's side of a connect or accept handshake over the fake socket. Each stage's first transfer stops one byte
// short; after that the peer sends one device, then `remoteMeta`, then a ready flag.
class ScriptedHandshakePeer {
 public:
  ScriptedHandshakePeer(std::function<int()> stage, const ncclIbConnectionMetadata& remoteMeta)
      : stage_(std::move(stage)),
        remoteMeta_(remoteMeta),
        init_(g_ncclSocketInit, [](ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType,
                                   volatile uint32_t*, int, int) { return ncclSuccess; }),
        progress_(g_ncclSocketProgress, [this](int op, ncclSocket*, void* data, int size, int* offset, bool*) {
          return Progress(op, data, size, offset);
        }) {}

  int sendCalls = 0;
  int recvCalls = 0;
  int sentIsP2p = -1;
  std::set<int> partialStages;

 private:
  ncclResult_t Progress(int op, void* data, int size, int* offset) {
    if (partialStages.insert(stage_()).second) {
      *offset = size - 1;
      return ncclSuccess;
    }
    if (op == NCCL_SOCKET_SEND) {
      ++sendCalls;
      if (size == static_cast<int>(sizeof(ncclIbConnectionMetadata))) {
        sentIsP2p = static_cast<ncclIbConnectionMetadata*>(data)->isP2p;
      }
    } else if (recvCalls++ == 0) {
      ncclNetVDeviceProps_t remote{};
      remote.ndevs = 1;
      remote.devs[0] = 0;
      std::memcpy(data, &remote, sizeof(remote));
      ncclIbDevExtraProps extra{};
      std::memcpy(static_cast<char*>(data) + sizeof(remote), &extra, sizeof(extra));
    } else if (recvCalls == 2) {
      std::memcpy(data, &remoteMeta_, sizeof(remoteMeta_));
    } else {
      *static_cast<int*>(data) = 1;
    }
    *offset = size;
    return ncclSuccess;
  }

  std::function<int()> stage_;
  ncclIbConnectionMetadata remoteMeta_;
  ScopedHook<ncclResult_t(ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType, volatile uint32_t*, int,
                          int)>
      init_;
  ScopedHook<ncclResult_t(int, ncclSocket*, void*, int, int*, bool*)> progress_;
};

class NetIbConnectMicrotest : public ::testing::Test {
 protected:
  void ConfigureSingleDevice() {
    context_.device = &device_;
    std::snprintf(device_.name, sizeof(device_.name), "micro0");
    pd_.context = &context_;
    cq_.context = &context_;
    ncclNIbDevs = 1;
    ncclNMergedIbDevs = 1;
    ncclIbDevs[0].context = &context_;
    ncclIbDevs[0].portNum = 1;
    ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
    ncclIbDevs[0].portAttr.link_layer = IBV_LINK_LAYER_ETHERNET;
    ncclIbDevs[0].portAttr.lid = 7;
    ncclIbDevs[0].ar = 1;
    ncclIbDevs[0].ibProvider = IB_PROVIDER_NONE;
    ncclIbDevs[0].oooRqSize = 0;
    ncclIbDevs[0].pd = nullptr;
    ncclIbDevs[0].pdRefs = 0;
    ncclIbDevs[0].gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
    ncclIbDevs[0].gidInfo.localGidIndex = 0;
    ncclIbDevs[0].gidInfo.localGid = Ipv6Gid("2001:db8::1");
    COMPILER_ATOMIC_STORE(&ncclIbDevs[0].currSpeed, uint64_t{100000}, std::memory_order_relaxed);
    ncclIbMergedDevs[0].vProps = {};
    ncclIbMergedDevs[0].vProps.ndevs = 1;
    ncclIbMergedDevs[0].vProps.devs[0] = 0;
    std::snprintf(ncclIbMergedDevs[0].devName, sizeof(ncclIbMergedDevs[0].devName), "micro-merged");
  }

  void InstallSingleDeviceResourceFakes() {
    g_ncclIbResiliencyInit = [](ncclIbNetCommBase*, ncclIbResiliency** out) {
      *out = nullptr;
      return ncclSuccess;
    };
    g_loadParam = [](const char* name, int64_t fallback) {
      if (std::strcmp(name, "IB_ECE_ENABLE") == 0) return int64_t{0};
      if (std::strcmp(name, "IB_EVENT_BASED_LB") == 0) return int64_t{0};
      if (std::strcmp(name, "IB_EVENT_BASED_LB_REMOTE") == 0) return int64_t{0};
      if (std::strcmp(name, "IB_PKEY_VALUE") == 0) return int64_t{-1};
      if (std::strcmp(name, "IB_PKEY") == 0) return int64_t{0};
      if (std::strcmp(name, "IB_QPS_PER_CONNECTION") == 0) return int64_t{1};
      if (std::strcmp(name, "IB_SUBNET_AWARE_ROUTING") == 0) return int64_t{0};
      return fallback;
    };
    g_ibvAllocPd = [this](ibv_pd** out, ibv_context*) {
      *out = &pd_;
      return ncclSuccess;
    };
    g_ibvCreateCq = [this](ibv_cq** out, ibv_context*, int, void* cqContext, ibv_comp_channel*, int) {
      cq_.cq_context = cqContext;
      *out = &cq_;
      return ncclSuccess;
    };
    g_ibvCreateQp = [this](ibv_qp** out, ibv_pd*, ibv_qp_init_attr* attr) {
      qps_.emplace_back();
      auto& qp = qps_.back();
      qp.context = &context_;
      qp.qp_num = nextQp_++;
      qp.qp_type = attr->qp_type;
      qp.qp_context = attr->qp_context;
      *out = &qp;
      return ncclSuccess;
    };
    g_ibvModifyQp = [](ibv_qp*, ibv_qp_attr*, int) { return ncclSuccess; };
    g_ibvRegMr = [this](ibv_mr** out, ibv_pd* pd, void*, size_t, int) {
      mrs_.emplace_back();
      auto& mr = mrs_.back();
      mr.pd = pd;
      mr.lkey = nextKey_++;
      mr.rkey = nextKey_++;
      *out = &mr;
      return ncclSuccess;
    };
    g_ibvDeregMr = [](ibv_mr*) { return ncclSuccess; };
    g_ibvDestroyQp = [](ibv_qp*) { return ncclSuccess; };
    g_ibvDestroyCq = [](ibv_cq*) { return ncclSuccess; };
    g_ibvDeallocPd = [](ibv_pd*) { return ncclSuccess; };
    g_ncclSocketClose = [](ncclSocket*, bool) { return ncclSuccess; };
  }

  void TearDown() override {
    NetIbResetGlobals();
    ResetNetIbSupportSeams();
    ResetTransportNetIbWqeLatFakes();
    ResetTransportNetIbResiliencyFakes();
    ResetTransportNetIbFakes();
    ResetSocketFakes();
    ResetOsFakes();
    ResetHipFakes();
    ResetIbvwrapFakes();
    ResetMlx5dvwrapFakes();
    ResetEnvFakes();
    ResetNcclFakes();
    g_connectOpen = [](const char*, int) { return -1; };
    g_connectRead = [](int, void*, size_t) { return ssize_t{-1}; };
    g_connectClose = [](int) { return 0; };
    g_connectMemAlloc = [](void** ptr, size_t) {
      if (ptr) *ptr = nullptr;
      return ncclSystemError;
    };
    qps_.clear();
    mrs_.clear();
  }

  ibv_context context_{};
  ibv_device device_{};
  ibv_pd pd_{};
  ibv_cq cq_{};
  std::deque<ibv_qp> qps_;
  std::deque<ibv_mr> mrs_;
  uint32_t nextQp_ = 100;
  uint32_t nextKey_ = 1000;
};

TEST_F(NetIbConnectMicrotest, QpsPerDev_ExplicitCountOverridesEnvironment) {
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_QPS_PER_CONNECTION") == 0) return int64_t{8};
    return fallback;
  });

  EXPECT_EQ(rcclIbQpsPerDev(/*nQpsPerDev=*/2, /*isP2p=*/0), 2);
  EXPECT_EQ(rcclIbCqQpsPerDev(/*nQpsPerDev=*/2), 2);
}

TEST_F(NetIbConnectMicrotest, QpsPerDev_P2pOverrideAppliesOnlyToP2pConnections) {
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "RCCL_IB_QPS_PER_P2P") == 0) return int64_t{4};
    if (std::strcmp(name, "IB_QPS_PER_CONNECTION") == 0) return int64_t{2};
    return fallback;
  });

  EXPECT_EQ(rcclIbQpsPerDev(/*nQpsPerDev=*/0, /*isP2p=*/1), 4);
  EXPECT_EQ(rcclIbQpsPerDev(/*nQpsPerDev=*/0, /*isP2p=*/0), 2);
  EXPECT_EQ(rcclIbCqQpsPerDev(/*nQpsPerDev=*/0), 2);
}

TEST_F(NetIbConnectMicrotest, SelectNqps_AsymmetricEndpointsUsesLargerTotal) {
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_QPS_PER_CONNECTION") == 0 ? int64_t{2} : fallback;
  });

  EXPECT_EQ(rcclIbMaxNqps(/*qpPerDev=*/2, /*localNdevs=*/1, /*remoteNdevs=*/3, "test"), 6);
  EXPECT_EQ(rcclIbSelectNqps(/*nQpsPerDev=*/0, /*isP2p=*/0, /*localNdevs=*/1, /*remoteNdevs=*/3, "test"), 6);
}

TEST_F(NetIbConnectMicrotest, AddrFamily_UnsetUnknownAndExplicitValuesSelectExpectedFamily) {
  SetMicroEnvAbsent("NCCL_IB_ADDR_FAMILY");
  EXPECT_EQ(envIbAddrFamily(), AF_INET);
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "unknown");
  EXPECT_EQ(envIbAddrFamily(), AF_INET);
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "AF_INET6");
  EXPECT_EQ(envIbAddrFamily(), AF_INET6);
}

TEST_F(NetIbConnectMicrotest, AddrRange_ValidIpv4AndIpv6CidrPublishesAddressAndMask) {
  int mask = -1;
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "10.2.0.0/16");
  auto* ipv4 = static_cast<in_addr*>(envIbAddrRange(AF_INET, &mask));
  ASSERT_NE(ipv4, nullptr);
  EXPECT_EQ(mask, 16);
  EXPECT_EQ(ipv4->s_addr, inet_addr("10.2.0.0"));

  SetMicroEnv("NCCL_IB_ADDR_RANGE", "2001:db8::/64");
  auto* ipv6 = static_cast<in6_addr*>(envIbAddrRange(AF_INET6, &mask));
  ASSERT_NE(ipv6, nullptr);
  EXPECT_EQ(mask, 64);
  in6_addr expected{};
  ASSERT_EQ(inet_pton(AF_INET6, "2001:db8::", &expected), 1);
  EXPECT_EQ(std::memcmp(ipv6, &expected, sizeof(expected)), 0);
}

TEST_F(NetIbConnectMicrotest, AddrRange_MissingSeparatorInvalidAddressOrMaskReturnsNull) {
  int mask = -1;
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "10.0.0.0");
  EXPECT_EQ(envIbAddrRange(AF_INET, &mask), nullptr);
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "not-an-address/24");
  EXPECT_EQ(envIbAddrRange(AF_INET, &mask), nullptr);
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "10.0.0.0/33");
  EXPECT_EQ(envIbAddrRange(AF_INET, &mask), nullptr);
  EXPECT_EQ(mask, 0);
  SetMicroEnv("NCCL_IB_ADDR_RANGE", "2001:db8::/129");
  EXPECT_EQ(envIbAddrRange(AF_INET6, &mask), nullptr);
  EXPECT_EQ(mask, 0);
}

TEST_F(NetIbConnectMicrotest, GidFamilyRecognizesIpv4MappedAndNativeIpv6) {
  auto ipv4 = Ipv4Gid("10.2.3.4");
  auto ipv6 = Ipv6Gid("2001:db8::1");

  EXPECT_EQ(getGidAddrFamily(&ipv4), AF_INET);
  EXPECT_EQ(getGidAddrFamily(&ipv6), AF_INET6);
}

TEST_F(NetIbConnectMicrotest, MatchGidPrefixHonorsIpv4AndPartialIpv6Masks) {
  auto gid4 = Ipv4Gid("10.2.3.4");
  in_addr prefix4{};
  ASSERT_EQ(inet_pton(AF_INET, "10.2.0.0", &prefix4), 1);
  EXPECT_TRUE(matchGidAddrPrefix(AF_INET, &prefix4, 16, &gid4));
  ASSERT_EQ(inet_pton(AF_INET, "10.3.0.0", &prefix4), 1);
  EXPECT_FALSE(matchGidAddrPrefix(AF_INET, &prefix4, 16, &gid4));

  auto gid6 = Ipv6Gid("2001:db8:abcd:12::1");
  in6_addr prefix6{};
  ASSERT_EQ(inet_pton(AF_INET6, "2001:db8:abcd::", &prefix6), 1);
  EXPECT_TRUE(matchGidAddrPrefix(AF_INET6, &prefix6, 48, &gid6));
  ASSERT_EQ(inet_pton(AF_INET6, "2001:db9::", &prefix6), 1);
  EXPECT_FALSE(matchGidAddrPrefix(AF_INET6, &prefix6, 32, &gid6));
}

TEST_F(NetIbConnectMicrotest, ValidGid_RejectsZeroAndLinkLocalButAcceptsRoutable) {
  union ibv_gid zero{};
  auto linkLocal = Ipv6Gid("fe80::1");
  auto routable = Ipv6Gid("2001:db8::1");

  EXPECT_FALSE(configuredGid(&zero));
  EXPECT_FALSE(validGid(&zero));
  EXPECT_TRUE(linkLocalGid(&linkLocal));
  EXPECT_FALSE(validGid(&linkLocal));
  EXPECT_TRUE(validGid(&routable));
}

TEST_F(NetIbConnectMicrotest, GidSameSubnet_UsesIpv4PrefixAndIpv6SubnetPrefix) {
  auto left4 = Ipv4Gid("10.2.1.1");
  auto right4 = Ipv4Gid("10.2.9.9");
  EXPECT_TRUE(gidSameSubnet(&left4, &right4, 16));
  EXPECT_FALSE(gidSameSubnet(&left4, &right4, 24));

  auto left6 = Ipv6Gid("2001:db8:1::1");
  auto right6 = Ipv6Gid("2001:db8:1::2");
  auto other6 = Ipv6Gid("2001:db8:2::1");
  EXPECT_TRUE(gidSameSubnet(&left6, &right6, 64));
  EXPECT_FALSE(gidSameSubnet(&left6, &other6, 64));
  EXPECT_FALSE(gidSameSubnet(&left4, &left6, 24));
}

TEST_F(NetIbConnectMicrotest, TrafficClass_NullContextIsUndefinedAndValidContextRoundTrips) {
  EXPECT_EQ(ncclIbGetTrafficClass(nullptr), NCCL_NET_TRAFFIC_CLASS_UNDEF);
  ncclNetCommConfig_t config{};
  ncclIbSetTrafficClass(&config, 37);
  EXPECT_EQ(ncclIbGetTrafficClass(&config), 37);
}

TEST_F(NetIbConnectMicrotest, P2pPolicy_InvalidHandleIsRejectedAndValidHandlePersistsPolicy) {
  EXPECT_EQ(rcclNetP2pPolicy(nullptr, 1), ncclInvalidArgument);
  ncclIbHandle handle{};
  EXPECT_EQ(rcclNetP2pPolicy(&handle, 1), ncclInvalidArgument);
  handle.magic = NCCL_SOCKET_MAGIC;

  EXPECT_EQ(rcclNetP2pPolicy(&handle, 1), ncclSuccess);
  EXPECT_EQ(handle.isP2p, 1);
  EXPECT_EQ(rcclNetP2pPolicy(&handle, 0), ncclSuccess);
  EXPECT_EQ(handle.isP2p, 0);
}

TEST_F(NetIbConnectMicrotest, RoceVersion_KnownTypePublishesVersionAndClosesFile) {
  ScopedHook open(g_connectOpen, [](const char* path, int) {
    EXPECT_STREQ(path, "/sys/class/infiniband/micro/ports/2/gid_attrs/types/3");
    return 17;
  });
  ScopedHook read(g_connectRead, [](int fd, void* data, size_t) {
    EXPECT_EQ(fd, 17);
    std::memcpy(data, "RoCE v2\n", 8);
    return ssize_t{8};
  });
  ScopedHook close(g_connectClose, [](int fd) {
    EXPECT_EQ(fd, 17);
    return 0;
  });
  int version = -1;

  EXPECT_EQ(ncclIbRoceGetVersionNum("micro", 2, 3, &version), ncclSuccess);
  EXPECT_EQ(version, 2);
  EXPECT_EQ(open.calls, 1);
  EXPECT_EQ(read.calls, 1);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbConnectMicrotest, RoceVersion_OpenOrReadFailureReturnsExpectedStatus) {
  int version = -1;
  EXPECT_EQ(ncclIbRoceGetVersionNum("micro", 1, 0, &version), ncclSystemError);

  ScopedHook open(g_connectOpen, [](const char*, int) { return 5; });
  ScopedHook read(g_connectRead, [](int, void*, size_t) {
    errno = EINVAL;
    return ssize_t{-1};
  });
  EXPECT_EQ(ncclIbRoceGetVersionNum("micro", 1, 0, &version), ncclSuccess);
}

TEST_F(NetIbConnectMicrotest, RoceVersion_NonContainerReadFailureReturnsSystemError) {
  ScopedHook open(g_connectOpen, [](const char*, int) { return 6; });
  ScopedHook read(g_connectRead, [](int, void*, size_t) {
    errno = EIO;
    return ssize_t{-1};
  });
  ScopedHook close(g_connectClose, [](int fd) {
    EXPECT_EQ(fd, 6);
    return 0;
  });
  int version = -1;

  EXPECT_EQ(ncclIbRoceGetVersionNum("micro", 1, 0, &version), ncclSystemError);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbConnectMicrotest, GidInfoQuery_ConfiguredIndexPublishesLinkLayerIndexAndGid) {
  ibv_context context{};
  ibv_port_attr port{};
  port.link_layer = IBV_LINK_LAYER_ETHERNET;
  port.gid_tbl_len = 4;
  const auto expected = Ipv6Gid("2001:db8::7");
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_GID_INDEX") == 0 ? int64_t{2} : fallback;
  });
  ScopedHook query(g_ibvQueryGid, [&expected, &context](ibv_context* seenContext, uint8_t portNum, int index,
                                                       ibv_gid* gid) {
    EXPECT_EQ(portNum, 3);
    EXPECT_EQ(index, 2);
    EXPECT_EQ(seenContext, &context);
    *gid = expected;
    return ncclSuccess;
  });
  ncclIbGidInfo info{};

  ASSERT_EQ(NetIbConnectGidInfoQuery(&context, 3, &port, &info), ncclSuccess);
  EXPECT_EQ(info.link_layer, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(info.localGidIndex, 2);
  EXPECT_EQ(std::memcmp(&info.localGid, &expected, sizeof(expected)), 0);
  EXPECT_EQ(query.calls, 1);
}

TEST_F(NetIbConnectMicrotest, GidInfoQuery_NullInputIsRejected) {
  ibv_context context{};
  ibv_port_attr port{};
  ncclIbGidInfo info{};

  EXPECT_EQ(NetIbConnectGidInfoQuery(nullptr, 1, &port, &info), ncclInternalError);
  EXPECT_EQ(NetIbConnectGidInfoQuery(&context, 1, nullptr, &info), ncclInternalError);
  EXPECT_EQ(NetIbConnectGidInfoQuery(&context, 1, &port, nullptr), ncclInternalError);
}

TEST_F(NetIbConnectMicrotest, PkeyIndex_ConfiguredIndexSkipsTableQuery) {
  ibv_context context{};
  ibv_port_attr port{};
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_PKEY_VALUE") == 0) return int64_t{-1};
    if (std::strcmp(name, "IB_PKEY") == 0) return int64_t{5};
    return fallback;
  });
  int index = -1;

  EXPECT_EQ(ncclIbGetPkeyIndex(&context, 1, &port, &index), ncclSuccess);
  EXPECT_EQ(index, 5);
}

TEST_F(NetIbConnectMicrotest, PkeyIndex_MembershipBitIsIgnoredWhenMatchingValue) {
  ibv_context context{};
  ibv_port_attr port{};
  port.pkey_tbl_len = 3;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_PKEY_VALUE") == 0 ? int64_t{0x1234} : fallback;
  });
  ScopedHook query(g_ibvQueryPkey, [](ibv_context*, uint8_t, int index, uint16_t* pkey) {
    *pkey = htons(index == 1 ? 0x9234 : 0x1111);
    return ncclSuccess;
  });
  int index = -1;

  EXPECT_EQ(ncclIbGetPkeyIndex(&context, 1, &port, &index), ncclSuccess);
  EXPECT_EQ(index, 1);
  EXPECT_EQ(query.calls, 2);
}

TEST_F(NetIbConnectMicrotest, PkeyIndex_MissingValueReturnsInvalidUsage) {
  ibv_context context{};
  ibv_port_attr port{};
  port.pkey_tbl_len = 2;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_PKEY_VALUE") == 0 ? int64_t{0x2222} : fallback;
  });
  ScopedHook query(g_ibvQueryPkey, [](ibv_context*, uint8_t, int, uint16_t* pkey) {
    *pkey = htons(0x1111);
    return ncclSuccess;
  });
  int index = -1;

  EXPECT_EQ(ncclIbGetPkeyIndex(&context, 1, &port, &index), ncclInvalidUsage);
}

TEST_F(NetIbConnectMicrotest, InitAndDestroyCommDevBase_SharedPdIsAllocatedOnceAndReleasedByLastBase) {
  ibv_context context{};
  ibv_pd pd{};
  std::array<ibv_cq, 2> cqs{};
  ncclIbDevs[0].context = &context;
  ncclIbDevs[0].portNum = 2;
  ncclIbDevs[0].gidInfo.localGidIndex = 3;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_PKEY_VALUE") == 0) return int64_t{-1};
    if (std::strcmp(name, "IB_PKEY") == 0) return int64_t{5};
    return fallback;
  });
  ScopedHook alloc(g_ibvAllocPd, [&pd](ibv_pd** out, ibv_context*) {
    *out = &pd;
    return ncclSuccess;
  });
  int nextCq = 0;
  ScopedHook createCq(g_ibvCreateCq, [&cqs, &nextCq](ibv_cq** out, ibv_context*, int cqe, void* contextPtr,
                                                     ibv_comp_channel*, int) {
    EXPECT_EQ(cqe, 32);
    EXPECT_EQ(contextPtr, reinterpret_cast<void*>(0x1234));
    *out = &cqs[nextCq++];
    return ncclSuccess;
  });
  ScopedHook destroyCq(g_ibvDestroyCq, [](ibv_cq*) { return ncclSuccess; });
  ScopedHook dealloc(g_ibvDeallocPd, [&pd](ibv_pd* seen) {
    EXPECT_EQ(seen, &pd);
    return ncclSuccess;
  });
  ncclIbNetCommDevBase first{};
  ncclIbNetCommDevBase second{};

  ASSERT_EQ(ncclIbInitCommDevBase(0, &first, reinterpret_cast<void*>(0x1234), 32), ncclSuccess);
  ASSERT_EQ(ncclIbInitCommDevBase(0, &second, reinterpret_cast<void*>(0x1234), 32), ncclSuccess);
  EXPECT_EQ(alloc.calls, 1);
  EXPECT_EQ(ncclIbDevs[0].pdRefs, 2);
  for (const ncclIbNetCommDevBase* base : {&first, &second}) {
    EXPECT_EQ(base->ibDevN, 0);
    EXPECT_EQ(base->pd, &pd);
    EXPECT_EQ(base->gidInfo.localGidIndex, 3);
    EXPECT_EQ(base->pkeyIndex, 5);
  }
  EXPECT_EQ(first.cq, &cqs[0]);
  EXPECT_EQ(second.cq, &cqs[1]);

  EXPECT_EQ(ncclIbDestroyBase(&first), ncclSuccess);
  EXPECT_EQ(ncclIbDevs[0].pdRefs, 1);
  EXPECT_EQ(dealloc.calls, 0);
  EXPECT_EQ(ncclIbDestroyBase(&second), ncclSuccess);
  EXPECT_EQ(ncclIbDevs[0].pdRefs, 0);
  EXPECT_EQ(destroyCq.calls, 2);
  EXPECT_EQ(dealloc.calls, 1);
}

TEST_F(NetIbConnectMicrotest, QpInit_ForwardsConfiguredStatePkeyPortAndAccess) {
  ibv_qp verbsQp{};
  ncclIbQp qp{};
  qp.qp = &verbsQp;
  qp.initAttr.state = IBV_QPS_INIT;
  qp.initAttr.pkeyIndex = 4;
  qp.initAttr.portNum = 2;
  qp.initAttr.qpAccessFlags = IBV_ACCESS_REMOTE_WRITE;
  ScopedHook modify(g_ibvModifyQp, [](ibv_qp*, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(attr->qp_state, IBV_QPS_INIT);
    EXPECT_EQ(attr->pkey_index, 4);
    EXPECT_EQ(attr->port_num, 2);
    EXPECT_EQ(attr->qp_access_flags, IBV_ACCESS_REMOTE_WRITE);
    EXPECT_EQ(mask, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbQpInit(&qp), ncclSuccess);
  EXPECT_EQ(modify.calls, 1);
}

TEST_F(NetIbConnectMicrotest, QpCreate_StandardPathForwardsCapabilities) {
  ibv_context context{};
  ibv_pd pd{};
  pd.context = &context;
  ibv_cq cq{};
  ibv_qp verbsQp{};
  verbsQp.context = &context;
  ScopedHook create(g_ibvCreateQp, [&verbsQp, &pd, &cq](ibv_qp** out, ibv_pd* seenPd, ibv_qp_init_attr* attr) {
    EXPECT_EQ(seenPd, &pd);
    EXPECT_EQ(attr->send_cq, &cq);
    EXPECT_EQ(attr->recv_cq, &cq);
    EXPECT_EQ(attr->qp_type, IBV_QPT_RC);
    EXPECT_EQ(attr->cap.max_send_wr, 17u);
    EXPECT_EQ(attr->cap.max_recv_wr, 19u);
    *out = &verbsQp;
    return ncclSuccess;
  });
  ncclIbQpCreateAttr attrs{};
  attrs.pd = &pd;
  attrs.cq = &cq;
  attrs.type = IBV_QPT_RC;
  attrs.maxSendWorkRequest = 17;
  attrs.maxRecvWorkRequest = 19;
  ncclIbQp qp{};

  ASSERT_EQ(ncclIbQpCreate(&qp, &attrs), ncclSuccess);
  EXPECT_EQ(qp.qp, &verbsQp);
}

TEST_F(NetIbConnectMicrotest, QpCreate_OooPathUsesMlx5ExtendedCreation) {
  ibv_context context{};
  ibv_pd pd{};
  pd.context = &context;
  ibv_qp verbsQp{};
  ScopedHook create(g_mlx5dvCreateQp,
                    [&verbsQp](ibv_context*, ibv_qp_init_attr_ex* attr, mlx5dv_qp_init_attr* dvAttr) {
                      EXPECT_NE(dvAttr->create_flags & MLX5DV_QP_CREATE_OOO_DP, 0u);
                      EXPECT_NE(dvAttr->comp_mask & MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS, 0u);
                      EXPECT_NE(attr->comp_mask & IBV_QP_INIT_ATTR_PD, 0u);
                      return &verbsQp;
                    });
  ncclIbQpCreateAttr attrs{};
  attrs.pd = &pd;
  attrs.type = IBV_QPT_RC;
  attrs.oooRq = true;
  ncclIbQp qp{};

  EXPECT_EQ(ncclIbQpCreate(&qp, &attrs), ncclSuccess);
  EXPECT_EQ(qp.qp, &verbsQp);
  EXPECT_EQ(create.calls, 1);
}

TEST_F(NetIbConnectMicrotest, SenderQps_EceEnabledQueriesAndAppliesReducedEce) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  g_loadParam = [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_ECE_ENABLE") == 0) return int64_t{1};
    return fallback;
  };
  ncclIbSendComm comm{};
  comm.base.vProps.ndevs = 1;
  comm.base.nqps = 1;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[0].base.pd = &pd_;
  comm.devs[0].base.cq = &cq_;
  comm.devs[0].base.pkeyIndex = 0;
  comm.devs[0].base.gidInfo = ncclIbDevs[0].gidInfo;
  ncclIbConnectionMetadata local{};
  ScopedHook query(g_ibvQueryEce, [](ibv_qp*, ibv_ece* ece, int* supported) {
    ece->vendor_id = 0x1234;
    ece->options = 0x55;
    *supported = 1;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclIbSenderQpsCreate(&comm, &local), ncclSuccess);
  EXPECT_EQ(local.qpInfo[0].ece_supported, 1);
  EXPECT_EQ(local.qpInfo[0].ece.vendor_id, 0x1234u);
  EXPECT_EQ(query.calls, 1);

  ncclIbConnectionMetadata remote{};
  remote.ndevs = 1;
  remote.sl = 2;
  remote.tc = 3;
  remote.qpInfo[0].qpn = 200;
  remote.qpInfo[0].devIndex = 0;
  remote.qpInfo[0].ece_supported = 1;
  remote.qpInfo[0].ece.vendor_id = 0x4321;
  remote.qpInfo[0].ece.options = 0xaa;
  remote.devs[0].mtu = IBV_MTU_4096;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 1;
  remote.devs[0].gid = Ipv6Gid("2001:db8::2");
  ScopedHook set(g_ibvSetEce, [](ibv_qp*, ibv_ece* ece, int* supported) {
    EXPECT_EQ(ece->vendor_id, 0x4321u);
    EXPECT_EQ(ece->options, 0xaau);
    EXPECT_EQ(*supported, 1);
    return ncclSuccess;
  });
  std::vector<ibv_qp_attr> rtrs;
  ScopedHook modify(g_ibvModifyQp, [&rtrs](ibv_qp*, ibv_qp_attr* attr, int) {
    if (attr->qp_state == IBV_QPS_RTR) rtrs.push_back(*attr);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbSenderQpsToRts(&comm, &remote), ncclSuccess);
  EXPECT_EQ(set.calls, 1);
  EXPECT_EQ(comm.base.qps[0].ece.vendor_id, 0x4321u);
  ASSERT_EQ(rtrs.size(), 1u);
  EXPECT_EQ(rtrs[0].dest_qp_num, 200u);
  EXPECT_EQ(std::memcmp(&rtrs[0].ah_attr.grh.dgid, &remote.devs[0].gid, sizeof(remote.devs[0].gid)), 0);
  EXPECT_EQ(rtrs[0].ah_attr.sl, 2);
  EXPECT_EQ(rtrs[0].ah_attr.grh.traffic_class, 3);
}

TEST_F(NetIbConnectMicrotest, QpRtr_RcRoceUsesGlobalRouteAndRcOnlyAttributes) {
  ibv_qp verbsQp{};
  verbsQp.qp_type = IBV_QPT_RC;
  ncclIbQp qp{};
  qp.qp = &verbsQp;
  qp.rtrAttr.mtu = IBV_MTU_4096;
  qp.rtrAttr.linkLayer = IBV_LINK_LAYER_ETHERNET;
  qp.rtrAttr.tc = 31;
  qp.rtrAttr.sl = 2;
  qp.rtrAttr.remoteQpNum = 77;
  qp.rtrAttr.remoteLid = 9;
  const auto remoteGid = Ipv6Gid("2001:db8::2");
  qp.rtrAttr.remoteGid = remoteGid;
  qp.rtrAttr.localGidIndex = 3;
  qp.rtrAttr.localIbPort = 2;
  ScopedHook modify(g_ibvModifyQp, [&remoteGid](ibv_qp*, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(attr->qp_state, IBV_QPS_RTR);
    EXPECT_EQ(attr->dest_qp_num, 77u);
    EXPECT_EQ(attr->ah_attr.is_global, 1);
    EXPECT_EQ(std::memcmp(&attr->ah_attr.grh.dgid, &remoteGid, sizeof(remoteGid)), 0);
    EXPECT_EQ(attr->ah_attr.sl, 2);
    EXPECT_EQ(attr->ah_attr.grh.sgid_index, 3);
    EXPECT_EQ(attr->ah_attr.grh.traffic_class, 31);
    EXPECT_EQ(attr->ah_attr.port_num, 2);
    EXPECT_NE(mask & IBV_QP_MAX_DEST_RD_ATOMIC, 0);
    EXPECT_NE(mask & IBV_QP_MIN_RNR_TIMER, 0);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbQpRtr(&qp), ncclSuccess);
}

TEST_F(NetIbConnectMicrotest, QpRtr_UcSameSubnetIbUsesLidAndOmitsRcOnlyAttributes) {
  ibv_qp verbsQp{};
  verbsQp.qp_type = IBV_QPT_UC;
  ncclIbQp qp{};
  qp.qp = &verbsQp;
  qp.rtrAttr.mtu = IBV_MTU_2048;
  qp.rtrAttr.linkLayer = IBV_LINK_LAYER_INFINIBAND;
  qp.rtrAttr.remoteLid = 19;
  qp.rtrAttr.localGid.global.subnet_prefix = 0x1234;
  qp.rtrAttr.remoteGid.global.subnet_prefix = 0x1234;
  qp.rtrAttr.localIbPort = 1;
  ScopedHook modify(g_ibvModifyQp, [](ibv_qp*, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(attr->ah_attr.is_global, 0);
    EXPECT_EQ(attr->ah_attr.dlid, 19);
    EXPECT_EQ(mask & IBV_QP_MAX_DEST_RD_ATOMIC, 0);
    EXPECT_EQ(mask & IBV_QP_MIN_RNR_TIMER, 0);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbQpRtr(&qp), ncclSuccess);
}

TEST_F(NetIbConnectMicrotest, QpRts_RcUsesRetryAttributesWhileUcOmitsThem) {
  ibv_qp verbsQp{};
  ncclIbQp qp{};
  qp.qp = &verbsQp;
  qp.rtsAttr.timeout = 12;
  qp.rtsAttr.retryCnt = 5;
  std::vector<int> masks;
  ScopedHook modify(g_ibvModifyQp, [&masks](ibv_qp*, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(attr->qp_state, IBV_QPS_RTS);
    if (mask & IBV_QP_TIMEOUT) {
      EXPECT_EQ(attr->timeout, 12);
      EXPECT_EQ(attr->retry_cnt, 5);
      EXPECT_EQ(attr->rnr_retry, 7);
      EXPECT_EQ(attr->max_rd_atomic, 1);
    }
    masks.push_back(mask);
    return ncclSuccess;
  });

  verbsQp.qp_type = IBV_QPT_RC;
  ASSERT_EQ(ncclIbQpRts(&qp), ncclSuccess);
  verbsQp.qp_type = IBV_QPT_UC;
  ASSERT_EQ(ncclIbQpRts(&qp), ncclSuccess);
  ASSERT_EQ(masks.size(), 2u);
  EXPECT_NE(masks[0] & IBV_QP_TIMEOUT, 0);
  EXPECT_EQ(masks[1] & IBV_QP_TIMEOUT, 0);
}

TEST_F(NetIbConnectMicrotest, QpResetAndErrorTransitionToRequestedState) {
  ibv_qp verbsQp{};
  ncclIbQp qp{};
  qp.qp = &verbsQp;
  std::vector<ibv_qp_state> states;
  ScopedHook modify(g_ibvModifyQp, [&states](ibv_qp*, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(mask, IBV_QP_STATE);
    states.push_back(attr->qp_state);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbQpReset(&qp), ncclSuccess);
  EXPECT_EQ(ncclIbQpError(&qp), ncclSuccess);
  EXPECT_EQ(states, (std::vector<ibv_qp_state>{IBV_QPS_RESET, IBV_QPS_ERR}));
}

TEST_F(NetIbConnectMicrotest, FindDevBySubnet_InvalidPrefixOrNoValidRemoteKeepsDefault) {
  union ibv_gid remote[2] = {};
  int found = -1;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_SUBNET_PREFIX_LEN") == 0 ? int64_t{0} : fallback;
  });
  EXPECT_EQ(ncclIbFindDevBySubnet(remote, 2, 3, &found), ncclInvalidArgument);
  EXPECT_EQ(found, 3);

  ScopedHook validPrefix(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_SUBNET_PREFIX_LEN") == 0 ? int64_t{24} : fallback;
  });
  EXPECT_EQ(ncclIbFindDevBySubnet(remote, 2, 3, &found), ncclSuccess);
  EXPECT_EQ(found, 3);
}

TEST_F(NetIbConnectMicrotest, FindDevBySubnet_DefaultFullMatchPreservesMergedDevice) {
  ConfigureSingleDevice();
  auto remote = Ipv4Gid("10.2.9.9");
  auto local = Ipv4Gid("10.2.1.1");
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_SUBNET_PREFIX_LEN") == 0) return int64_t{16};
    if (std::strcmp(name, "IB_GID_INDEX") == 0) return int64_t{0};
    return fallback;
  });
  ScopedHook query(g_ibvQueryGid, [&local](ibv_context*, uint8_t, int, ibv_gid* gid) {
    *gid = local;
    return ncclSuccess;
  });
  int found = -1;

  EXPECT_EQ(ncclIbFindDevBySubnet(&remote, 1, 0, &found), ncclSuccess);
  EXPECT_EQ(found, 0);
  EXPECT_EQ(query.calls, 1);
}

TEST_F(NetIbConnectMicrotest, FindDevBySubnet_DefaultMismatchSelectsFullyReachableAlternative) {
  ConfigureSingleDevice();
  ibv_context secondContext{};
  ncclNIbDevs = 2;
  ncclNMergedIbDevs = 2;
  ncclIbDevs[1].context = &secondContext;
  ncclIbDevs[1].portNum = 2;
  ncclIbDevs[1].portAttr.link_layer = IBV_LINK_LAYER_ETHERNET;
  ncclIbMergedDevs[1].vProps = {};
  ncclIbMergedDevs[1].vProps.ndevs = 1;
  ncclIbMergedDevs[1].vProps.devs[0] = 1;
  auto remote = Ipv4Gid("10.2.9.9");
  auto wrong = Ipv4Gid("10.1.1.1");
  auto matching = Ipv4Gid("10.2.1.1");
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_SUBNET_PREFIX_LEN") == 0) return int64_t{16};
    if (std::strcmp(name, "IB_GID_INDEX") == 0) return int64_t{0};
    return fallback;
  });
  ScopedHook query(g_ibvQueryGid,
                   [this, &wrong, &matching](ibv_context* context, uint8_t, int, ibv_gid* gid) {
                     *gid = context == &context_ ? wrong : matching;
                     return ncclSuccess;
                   });
  int found = -1;

  EXPECT_EQ(ncclIbFindDevBySubnet(&remote, 1, 0, &found), ncclSuccess);
  EXPECT_EQ(found, 1);
  EXPECT_EQ(query.calls, 2);
}

TEST_F(NetIbConnectMicrotest, FindDevBySubnet_FusedDefaultWithOnePfOffSubnetIsReplaced) {
  ConfigureSingleDevice();
  // Merged device 0 fuses PFs 0 and 1; merged device 1 is PF 2 alone.
  std::array<ibv_context, 2> otherContexts{};
  ncclNIbDevs = 3;
  ncclNMergedIbDevs = 2;
  for (int pf = 1; pf < 3; ++pf) {
    ncclIbDevs[pf].context = &otherContexts[pf - 1];
    ncclIbDevs[pf].portNum = 1;
    ncclIbDevs[pf].portAttr.link_layer = IBV_LINK_LAYER_ETHERNET;
  }
  ncclIbMergedDevs[0].vProps.ndevs = 2;
  ncclIbMergedDevs[0].vProps.devs[1] = 1;
  ncclIbMergedDevs[1].vProps.ndevs = 1;
  ncclIbMergedDevs[1].vProps.devs[0] = 2;
  auto remote = Ipv4Gid("10.2.9.9");
  auto matching = Ipv4Gid("10.2.1.1");
  auto wrong = Ipv4Gid("10.1.1.1");
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_SUBNET_PREFIX_LEN") == 0) return int64_t{16};
    if (std::strcmp(name, "IB_GID_INDEX") == 0) return int64_t{0};
    return fallback;
  });
  ScopedHook query(g_ibvQueryGid, [&](ibv_context* context, uint8_t, int, ibv_gid* gid) {
    *gid = context == &otherContexts[0] ? wrong : matching;
    return ncclSuccess;
  });
  int found = -1;

  EXPECT_EQ(ncclIbFindDevBySubnet(&remote, 1, 0, &found), ncclSuccess);
  EXPECT_EQ(found, 1);
  EXPECT_EQ(query.calls, 3);
}

TEST_F(NetIbConnectMicrotest, SubnetMatchesAny_IgnoresInvalidGidsAndMatchesReachablePeer) {
  auto local = Ipv4Gid("10.2.1.1");
  std::array<ibv_gid, 2> remotes{};
  remotes[1] = Ipv4Gid("10.2.9.9");

  EXPECT_TRUE(subnetMatchesAny(&local, remotes.data(), remotes.size(), 16));
  EXPECT_FALSE(subnetMatchesAny(&local, remotes.data(), remotes.size(), 24));
}

TEST_F(NetIbConnectMicrotest, UpdateGidIndex_CandidateWithRequestedFamilyAndSubnetIsSelected) {
  ibv_context context{};
  auto current = Ipv6Gid("2001:db8::1");
  auto candidate = Ipv4Gid("10.2.3.4");
  ScopedHook query(g_ibvQueryGid, [&current, &candidate](ibv_context*, uint8_t, int index, ibv_gid* gid) {
    *gid = index == 0 ? current : candidate;
    return ncclSuccess;
  });
  in_addr prefix{};
  ASSERT_EQ(inet_pton(AF_INET, "10.2.0.0", &prefix), 1);
  int index = 0;

  EXPECT_EQ(ncclUpdateGidIndex(&context, 1, AF_INET, &prefix, 16, 2, 1, &index), ncclSuccess);
  EXPECT_EQ(index, 1);
}

TEST_F(NetIbConnectMicrotest, UpdateGidIndex_SameFamilyCandidateIsTakenOnlyForConfiguredRoceVersion) {
  ibv_device device{};
  ibv_context context{};
  context.device = &device;
  auto current = Ipv4Gid("10.2.1.1");
  auto candidate = Ipv4Gid("10.2.3.4");
  ScopedHook query(g_ibvQueryGid, [&current, &candidate](ibv_context*, uint8_t, int index, ibv_gid* gid) {
    *gid = index == 0 ? current : candidate;
    return ncclSuccess;
  });
  ScopedHook name(g_ibvGetDeviceName, [](ibv_device*) { return "micro"; });
  // gid_attrs/types/0 reads back RoCE v1 and types/1 RoCE v2.
  ScopedHook open(g_connectOpen, [](const char* path, int) {
    return std::strcmp(path, "/sys/class/infiniband/micro/ports/1/gid_attrs/types/0") == 0 ? 30 : 31;
  });
  ScopedHook read(g_connectRead, [](int fd, void* data, size_t) {
    std::memcpy(data, fd == 30 ? "RoCE v1\n" : "RoCE v2\n", 8);
    return ssize_t{8};
  });
  in_addr prefix{};
  ASSERT_EQ(inet_pton(AF_INET, "10.2.0.0", &prefix), 1);

  int index = 0;
  EXPECT_EQ(ncclUpdateGidIndex(&context, 1, AF_INET, &prefix, 16, /*roceVer=*/2, 1, &index), ncclSuccess);
  EXPECT_EQ(index, 1);
  index = 0;
  EXPECT_EQ(ncclUpdateGidIndex(&context, 1, AF_INET, &prefix, 16, /*roceVer=*/1, 1, &index), ncclSuccess);
  EXPECT_EQ(index, 0);
  EXPECT_EQ(read.calls, 4);
}

TEST_F(NetIbConnectMicrotest, Listen_SuccessPublishesHandleAndCloseReleasesComm) {
  ScopedHook init(g_ncclSocketInit,
                  [](ncclSocket*, const ncclSocketAddress*, uint64_t magic, ncclSocketType type,
                     volatile uint32_t*, int async, int) {
                    EXPECT_EQ(magic, NCCL_SOCKET_MAGIC);
                    EXPECT_EQ(type, ncclSocketTypeNetIb);
                    EXPECT_EQ(async, 1);
                    return ncclSuccess;
                  });
  ScopedHook listen(g_ncclSocketListen, [](ncclSocket*) { return ncclSuccess; });
  ScopedHook address(g_ncclSocketGetAddr, [](ncclSocket*, ncclSocketAddress* out) {
    std::memset(out, 0, sizeof(*out));
    out->sa.sa_family = AF_INET;
    return ncclSuccess;
  });
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  ncclIbHandle handle{};
  void* comm = nullptr;

  ASSERT_EQ(NetIbConnectListen(nullptr, 0, &handle, &comm), ncclSuccess);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(handle.magic, NCCL_SOCKET_MAGIC);
  EXPECT_EQ(handle.connectAddr.sa.sa_family, AF_INET);
  EXPECT_EQ(NetIbConnectCloseListen(comm), ncclSuccess);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbConnectMicrotest, Listen_SetupFailureClosesSocketAndDoesNotPublishComm) {
  ScopedHook init(g_ncclSocketInit,
                  [](ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType, volatile uint32_t*, int, int) {
                    return ncclSystemError;
                  });
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  ncclIbHandle handle{};
  void* comm = nullptr;

  EXPECT_EQ(NetIbConnectListen(nullptr, 0, &handle, &comm), ncclSystemError);
  EXPECT_EQ(comm, nullptr);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbConnectMicrotest, Connect_NotReadyThenCompletedHandshakePublishesSendComm) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  ncclIbHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  handle.isP2p = 1;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             PeerMetadata(/*ctsAddr=*/0x9000, /*qpn=*/200, /*lid=*/8, /*rkey=*/0x91,
                                          Ipv6Gid("2001:db8::2")));
  g_ncclSocketConnect = [](ncclSocket*) { return ncclSuccess; };
  int readyCalls = 0;
  g_ncclSocketReady = [&readyCalls](ncclSocket*, int* ready) {
    *ready = readyCalls++ > 0;
    return ncclSuccess;
  };
  void* sendComm = reinterpret_cast<void*>(0x1);

  ASSERT_EQ(ncclIbConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1,
                              /*envTrafficClass=*/17),
            ncclSuccess);
  EXPECT_EQ(sendComm, nullptr);
  EXPECT_EQ(handle.stage.state, ncclIbCommStateConnect);
  ASSERT_NE(handle.stage.comm, nullptr);

  void* const stageComm = handle.stage.comm;
  for (int step = 0; step < 10 && sendComm == nullptr; ++step) {
    ASSERT_EQ(ncclIbConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1,
                                /*envTrafficClass=*/17),
              ncclSuccess);
    if (sendComm == nullptr) EXPECT_EQ(handle.stage.comm, stageComm);
  }
  auto* comm = static_cast<ncclIbSendComm*>(sendComm);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->base.ready, 1);
  EXPECT_EQ(comm->base.nqps, 1);
  EXPECT_EQ(comm->base.nDataQps, 1);
  EXPECT_EQ(comm->base.nRemDevs, 1);
  EXPECT_EQ(comm->base.qps[0].remDevIdx, 0);
  EXPECT_EQ(handle.stage.state, ncclIbCommStateStart);
  EXPECT_GE(peer.sendCalls, 3);
  EXPECT_GE(peer.recvCalls, 2);
  EXPECT_EQ(peer.partialStages.size(), 5u);
  EXPECT_EQ(peer.sentIsP2p, 1);

  EXPECT_EQ(NetIbConnectCloseSend(comm), ncclSuccess);
}

TEST_F(NetIbConnectMicrotest, ConnectWrapper_UnexpectedStageIsRejected) {
  ncclIbHandle handle{};
  handle.stage.state = ncclIbCommStateAccept;
  void* comm = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(NetIbConnectConnect(nullptr, 0, &handle, &comm, nullptr), ncclInternalError);
  EXPECT_EQ(comm, nullptr);
}

TEST_F(NetIbConnectMicrotest, Accept_CompletedHandshakePublishesRecvCommAndClearsStage) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  const auto remoteGid = Ipv6Gid("2001:db8::2");
  std::vector<ibv_qp_attr> rtrs;
  ScopedHook modify(g_ibvModifyQp, [&rtrs](ibv_qp*, ibv_qp_attr* attr, int) {
    if (attr->qp_state == IBV_QPS_RTR) rtrs.push_back(*attr);
    return ncclSuccess;
  });
  ncclIbListenComm listenComm{};
  listenComm.dev = 0;
  auto remoteMeta = PeerMetadata(/*ctsAddr=*/0xa000, /*qpn=*/300, /*lid=*/9, /*rkey=*/0xa1, remoteGid);
  remoteMeta.sl = 2;
  remoteMeta.tc = 19;
  ScriptedHandshakePeer peer([&listenComm] { return static_cast<int>(listenComm.stage->state); }, remoteMeta);
  g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
  g_ncclSocketReady = [](ncclSocket*, int* ready) {
    *ready = 1;
    return ncclSuccess;
  };
  void* recvComm = nullptr;

  for (int step = 0; step < 10 && recvComm == nullptr; ++step) {
    ASSERT_EQ(ncclIbAcceptImpl(&listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1), ncclSuccess);
  }
  auto* comm = static_cast<ncclIbRecvComm*>(recvComm);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->base.ready, 1);
  EXPECT_EQ(comm->base.nqps, 1);
  EXPECT_EQ(comm->base.nRemDevs, 1);
  EXPECT_EQ(comm->flushEnabled, 0);
  EXPECT_EQ(comm->remCtsFifo.addr, 0xa000u);
  EXPECT_EQ(listenComm.stage, nullptr);
  EXPECT_GE(peer.sendCalls, 2);
  EXPECT_GE(peer.recvCalls, 3);
  EXPECT_EQ(peer.partialStages.size(), 5u);
  EXPECT_EQ(peer.sentIsP2p, 1);
  ASSERT_EQ(rtrs.size(), 1u);
  EXPECT_EQ(rtrs[0].dest_qp_num, 300u);
  EXPECT_EQ(std::memcmp(&rtrs[0].ah_attr.grh.dgid, &remoteGid, sizeof(remoteGid)), 0);
  EXPECT_EQ(rtrs[0].ah_attr.sl, 2);
  EXPECT_EQ(rtrs[0].ah_attr.grh.traffic_class, 19);

  EXPECT_EQ(NetIbConnectCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbConnectMicrotest, AcceptWrapper_UnexpectedStageIsRejected) {
  ncclIbListenComm listen{};
  ncclIbCommStage stage{};
  stage.state = ncclIbCommStateConnect;
  listen.stage = &stage;
  void* comm = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(NetIbConnectAccept(&listen, &comm, nullptr), ncclInternalError);
  EXPECT_EQ(comm, nullptr);
}

TEST_F(NetIbConnectMicrotest, CreateFlushQp_DisabledOrAlreadyCreatedIsNoOp) {
  ncclIbRecvComm comm{};
  comm.base.vProps.ndevs = 1;
  ScopedHook create(g_ibvCreateQp, [](ibv_qp**, ibv_pd*, ibv_qp_init_attr*) {
    ADD_FAILURE() << "guarded no-op must not create a QP";
    return ncclSystemError;
  });

  EXPECT_EQ(ncclIbCreateFlushQp(&comm), ncclSuccess);
  comm.flushEnabled = 1;
  comm.flushQpsCreated = true;
  EXPECT_EQ(ncclIbCreateFlushQp(&comm), ncclSuccess);
  EXPECT_EQ(create.calls, 0);
}

TEST_F(NetIbConnectMicrotest, CreateFlushQp_FirstUseCreatesLoopbackQpAndHostMr) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  ncclIbRecvComm comm{};
  comm.flushEnabled = 1;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[0].base.pd = &pd_;
  comm.devs[0].base.cq = &cq_;
  comm.devs[0].base.pkeyIndex = 4;
  comm.devs[0].base.gidInfo = ncclIbDevs[0].gidInfo;

  ASSERT_EQ(ncclIbCreateFlushQp(&comm), ncclSuccess);
  EXPECT_TRUE(comm.flushQpsCreated);
  ASSERT_NE(comm.devs[0].gpuFlush.qp.qp, nullptr);
  EXPECT_EQ(comm.devs[0].gpuFlush.qp.rtrAttr.remoteQpNum, comm.devs[0].gpuFlush.qp.qp->qp_num);
  EXPECT_NE(comm.devs[0].gpuFlush.hostMr, nullptr);
  EXPECT_EQ(comm.devs[0].gpuFlush.sge.lkey, comm.devs[0].gpuFlush.hostMr->lkey);
}

TEST_F(NetIbConnectMicrotest, PostReceiveWorkRequests_ResiliencyDepthControlsPostCount) {
  IbVerbsScene verbs("mlx5_0", 71);
  ncclIbRecvComm comm{};
  ncclIbResiliency resiliency{};
  comm.base.resiliency = &resiliency;
  ncclIbQp qp{};
  qp.qp = &verbs.qps[0];
  qp.devIndex = 0;
  g_ncclIbResiliencyDataRqSizeGet = [](ncclIbResiliency*, uint, uint32_t* size) {
    *size = 3;
    return ncclSuccess;
  };

  EXPECT_EQ(ncclIbPostReceiveWorkRequestsOnQp(&comm, &qp), ncclSuccess);
  EXPECT_EQ(verbs.postRecvQps, std::vector<ibv_qp*>(3, &verbs.qps[0]));
}

TEST_F(NetIbConnectMicrotest, ReceiverPrePostReceiveWorkRequests_VisitsEveryQp) {
  IbVerbsScene verbs("mlx5_0", 72);
  ncclIbRecvComm comm{};
  comm.base.nqps = 2;
  comm.base.qps[0].qp = &verbs.qps[0];
  comm.base.qps[1].qp = &verbs.qps[1];

  EXPECT_EQ(ncclIbReceiverPrePostReceiveWorkRequests(&comm), ncclSuccess);
  const auto& posts = verbs.postRecvQps;
  EXPECT_EQ(std::count(posts.begin(), posts.end(), &verbs.qps[0]), NET_IB_MAX_REQUESTS);
  EXPECT_EQ(std::count(posts.begin(), posts.end(), &verbs.qps[1]), NET_IB_MAX_REQUESTS);
  EXPECT_EQ(posts.size(), 2u * NET_IB_MAX_REQUESTS);
}

TEST_F(NetIbConnectMicrotest, FreeGpuFlushMem_HipAllocationIsReleasedAndStateCleared) {
  ncclIbGpuFlush flush{};
  flush.gpuFlushGpuMem = static_cast<int*>(std::malloc(sizeof(int)));
  ASSERT_NE(flush.gpuFlushGpuMem, nullptr);
  void* const allocation = flush.gpuFlushGpuMem;
  flush.gpuFlushMemIsHipAlloc = true;
  ScopedHook release(g_hipFree, [allocation](void* ptr) {
    EXPECT_EQ(ptr, allocation);
    std::free(ptr);
    return hipSuccess;
  });

  EXPECT_EQ(ncclIbFreeGpuFlushMem(&flush), ncclSuccess);
  EXPECT_EQ(flush.gpuFlushGpuMem, nullptr);
  EXPECT_FALSE(flush.gpuFlushMemIsHipAlloc);
  EXPECT_EQ(release.calls, 1);
}

TEST_F(NetIbConnectMicrotest, CloseFunctions_NullCommunicatorsAreNoOps) {
  EXPECT_EQ(NetIbConnectCloseSend(nullptr), ncclSuccess);
  EXPECT_EQ(NetIbConnectCloseRecv(nullptr), ncclSuccess);
  EXPECT_EQ(NetIbConnectCloseListen(nullptr), ncclSuccess);
}

}  // namespace
