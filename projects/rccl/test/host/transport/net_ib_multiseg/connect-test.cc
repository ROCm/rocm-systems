/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/connect.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "HipVmmLedger.h"
#include "IbVerbsScene.h"
#include "IbVerbsWorld.h"
#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/os_fakes.h"
#include "fakes/rccl_mseg_fakes.h"
#include "fakes/rocmwrap_fakes.h"
#include "fakes/socket_fakes.h"
#include "fakes/transport_net_ib_mseg_resiliency_fakes.h"
#include "fakes/transport_net_ib_mseg_wqe_lat_fakes.h"
#include "transport/net_ib_multiseg/NetIbMsegGlobals.h"
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

#define open TestConnectOpen
#define read TestConnectRead
#define close TestConnectClose
#define IbMsegGdrSupport NetIbTestGdrSupport
#define IbMsegDmaBufSupport NetIbTestDmaBufSupport
#define ncclMemAlloc TestConnectMemAlloc
#include NET_IB_MSEG_CONNECT_CC_PATH
#undef ncclMemAlloc
#undef IbMsegDmaBufSupport
#undef IbMsegGdrSupport
#undef close
#undef read
#undef open
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
IbMsegConnectionMetadata PeerMetadata(uint64_t ctsAddr, uint32_t qpn, uint16_t lid, uint32_t rkey,
                                      const ibv_gid& gid) {
  IbMsegConnectionMetadata meta{};
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
  ScriptedHandshakePeer(std::function<int()> stage, const IbMsegConnectionMetadata& remoteMeta)
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
  int sentUseMultiSeg = -1;
  std::set<int> partialStages;

 private:
  ncclResult_t Progress(int op, void* data, int size, int* offset) {
    if (partialStages.insert(stage_()).second) {
      *offset = size - 1;
      return ncclSuccess;
    }
    if (op == NCCL_SOCKET_SEND) {
      ++sendCalls;
      if (size == static_cast<int>(sizeof(IbMsegConnectionMetadata))) {
        sentIsP2p = static_cast<IbMsegConnectionMetadata*>(data)->isP2p;
        sentUseMultiSeg = static_cast<IbMsegConnectionMetadata*>(data)->useMultiSeg;
      }
    } else if (recvCalls++ == 0) {
      ncclNetVDeviceProps_t remote{};
      remote.ndevs = 1;
      remote.devs[0] = 0;
      std::memcpy(data, &remote, sizeof(remote));
      IbMsegDevExtraProps extra{};
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
  IbMsegConnectionMetadata remoteMeta_;
  ScopedHook<ncclResult_t(ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType, volatile uint32_t*, int,
                          int)>
      init_;
  ScopedHook<ncclResult_t(int, ncclSocket*, void*, int, int*, bool*)> progress_;
};

class NetIbMsegConnectMicrotest : public ::testing::Test {
 protected:
  void ConfigureSingleDevice() {
    context_.device = &device_;
    std::snprintf(device_.name, sizeof(device_.name), "micro0");
    pd_.context = &context_;
    cq_.context = &context_;
    IbMsegNDevs = 1;
    IbMsegNMergedDevs = 1;
    IbMsegDevs[0].context = &context_;
    IbMsegDevs[0].portNum = 1;
    IbMsegDevs[0].portAttr.active_mtu = IBV_MTU_4096;
    IbMsegDevs[0].portAttr.link_layer = IBV_LINK_LAYER_ETHERNET;
    IbMsegDevs[0].portAttr.lid = 7;
    IbMsegDevs[0].ar = 1;
    IbMsegDevs[0].ibProvider = IB_PROVIDER_NONE;
    IbMsegDevs[0].oooRqSize = 0;
    IbMsegDevs[0].pd = nullptr;
    IbMsegDevs[0].pdRefs = 0;
    IbMsegDevs[0].gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
    IbMsegDevs[0].gidInfo.localGidIndex = 0;
    IbMsegDevs[0].gidInfo.localGid = Ipv6Gid("2001:db8::1");
    COMPILER_ATOMIC_STORE(&IbMsegDevs[0].currSpeed, uint64_t{100000}, std::memory_order_relaxed);
    IbMsegMergedDevs[0].vProps = {};
    IbMsegMergedDevs[0].vProps.ndevs = 1;
    IbMsegMergedDevs[0].vProps.devs[0] = 0;
    std::snprintf(IbMsegMergedDevs[0].devName, sizeof(IbMsegMergedDevs[0].devName), "micro-merged");
  }

  void InstallSingleDeviceResourceFakes() {
    g_ncclIbResiliencyInit = [](IbMsegNetCommBase*, IbMsegResiliency** out) {
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
    g_ibvRegMr = [this](ibv_mr** out, ibv_pd* pd, void* addr, size_t length, int) {
      mrs_.emplace_back();
      auto& mr = mrs_.back();
      mr.pd = pd;
      mr.addr = addr;
      mr.length = length;
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
    NetIbMsegResetGlobals();
    ResetRcclMsegFakes();
    ResetNetIbSupportSeams();
    ResetTransportNetIbWqeLatFakes();
    ResetTransportNetIbResiliencyFakes();
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

TEST_F(NetIbMsegConnectMicrotest, QpsPerDev_ExplicitCountOverridesEnvironment) {
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_QPS_PER_CONNECTION") == 0) return int64_t{8};
    return fallback;
  });

  EXPECT_EQ(rcclIbQpsPerDev(/*nQpsPerDev=*/2, /*isP2p=*/0), 2);
  EXPECT_EQ(rcclIbCqQpsPerDev(/*nQpsPerDev=*/2), 2);
}

TEST_F(NetIbMsegConnectMicrotest, QpsPerDev_P2pOverrideAppliesOnlyToP2pConnections) {
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "RCCL_IB_QPS_PER_P2P") == 0) return int64_t{4};
    if (std::strcmp(name, "IB_QPS_PER_CONNECTION") == 0) return int64_t{2};
    return fallback;
  });

  EXPECT_EQ(rcclIbQpsPerDev(/*nQpsPerDev=*/0, /*isP2p=*/1), 4);
  EXPECT_EQ(rcclIbQpsPerDev(/*nQpsPerDev=*/0, /*isP2p=*/0), 2);
  EXPECT_EQ(rcclIbCqQpsPerDev(/*nQpsPerDev=*/0), 2);
}

TEST_F(NetIbMsegConnectMicrotest, SelectNqps_AsymmetricEndpointsUsesLargerTotal) {
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_QPS_PER_CONNECTION") == 0 ? int64_t{2} : fallback;
  });

  EXPECT_EQ(rcclIbMaxNqps(/*qpPerDev=*/2, /*localNdevs=*/1, /*remoteNdevs=*/3, "test"), 6);
  EXPECT_EQ(rcclIbSelectNqps(/*nQpsPerDev=*/0, /*isP2p=*/0, /*localNdevs=*/1, /*remoteNdevs=*/3, "test"), 6);
}

TEST_F(NetIbMsegConnectMicrotest, AddrFamily_UnsetUnknownAndExplicitValuesSelectExpectedFamily) {
  SetMicroEnvAbsent("NCCL_IB_ADDR_FAMILY");
  EXPECT_EQ(envIbAddrFamily(), AF_INET);
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "unknown");
  EXPECT_EQ(envIbAddrFamily(), AF_INET);
  SetMicroEnv("NCCL_IB_ADDR_FAMILY", "AF_INET6");
  EXPECT_EQ(envIbAddrFamily(), AF_INET6);
}

TEST_F(NetIbMsegConnectMicrotest, AddrRange_ValidIpv4AndIpv6CidrPublishesAddressAndMask) {
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

TEST_F(NetIbMsegConnectMicrotest, AddrRange_MissingSeparatorInvalidAddressOrMaskReturnsNull) {
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

TEST_F(NetIbMsegConnectMicrotest, GidFamilyRecognizesIpv4MappedAndNativeIpv6) {
  auto ipv4 = Ipv4Gid("10.2.3.4");
  auto ipv6 = Ipv6Gid("2001:db8::1");

  EXPECT_EQ(getGidAddrFamily(&ipv4), AF_INET);
  EXPECT_EQ(getGidAddrFamily(&ipv6), AF_INET6);
}

TEST_F(NetIbMsegConnectMicrotest, MatchGidPrefixHonorsIpv4AndPartialIpv6Masks) {
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

TEST_F(NetIbMsegConnectMicrotest, ValidGid_RejectsZeroAndLinkLocalButAcceptsRoutable) {
  union ibv_gid zero{};
  auto linkLocal = Ipv6Gid("fe80::1");
  auto routable = Ipv6Gid("2001:db8::1");

  EXPECT_FALSE(configuredGid(&zero));
  EXPECT_FALSE(validGid(&zero));
  EXPECT_TRUE(linkLocalGid(&linkLocal));
  EXPECT_FALSE(validGid(&linkLocal));
  EXPECT_TRUE(validGid(&routable));
}

TEST_F(NetIbMsegConnectMicrotest, GidSameSubnet_UsesIpv4PrefixAndIpv6SubnetPrefix) {
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

TEST_F(NetIbMsegConnectMicrotest, TrafficClass_NullContextIsUndefinedAndValidContextRoundTrips) {
  EXPECT_EQ(IbMsegGetTrafficClass(nullptr), NCCL_NET_TRAFFIC_CLASS_UNDEF);
  ncclNetCommConfig_t config{};
  IbMsegSetTrafficClass(&config, 37);
  EXPECT_EQ(IbMsegGetTrafficClass(&config), 37);
}

TEST_F(NetIbMsegConnectMicrotest, P2pPolicy_InvalidHandleIsRejectedAndValidHandlePersistsPolicy) {
  EXPECT_EQ(IbMsegNetP2pPolicy(nullptr, 1), ncclInvalidArgument);
  IbMsegHandle handle{};
  EXPECT_EQ(IbMsegNetP2pPolicy(&handle, 1), ncclInvalidArgument);
  handle.magic = NCCL_SOCKET_MAGIC;

  EXPECT_EQ(IbMsegNetP2pPolicy(&handle, 1), ncclSuccess);
  EXPECT_EQ(handle.isP2p, 1);
  EXPECT_EQ(IbMsegNetP2pPolicy(&handle, 0), ncclSuccess);
  EXPECT_EQ(handle.isP2p, 0);
}

TEST_F(NetIbMsegConnectMicrotest, RoceVersion_KnownTypePublishesVersionAndClosesFile) {
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

  EXPECT_EQ(IbMsegRoceGetVersionNum("micro", 2, 3, &version), ncclSuccess);
  EXPECT_EQ(version, 2);
  EXPECT_EQ(open.calls, 1);
  EXPECT_EQ(read.calls, 1);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, RoceVersion_OpenOrReadFailureReturnsExpectedStatus) {
  int version = -1;
  EXPECT_EQ(IbMsegRoceGetVersionNum("micro", 1, 0, &version), ncclSystemError);

  ScopedHook open(g_connectOpen, [](const char*, int) { return 5; });
  ScopedHook read(g_connectRead, [](int, void*, size_t) {
    errno = EINVAL;
    return ssize_t{-1};
  });
  EXPECT_EQ(IbMsegRoceGetVersionNum("micro", 1, 0, &version), ncclSuccess);
}

TEST_F(NetIbMsegConnectMicrotest, RoceVersion_NonContainerReadFailureReturnsSystemError) {
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

  EXPECT_EQ(IbMsegRoceGetVersionNum("micro", 1, 0, &version), ncclSystemError);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, GidInfoQuery_ConfiguredIndexPublishesLinkLayerIndexAndGid) {
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
  IbMsegGidInfo info{};

  ASSERT_EQ(IbMsegGidInfoQuery(&context, 3, &port, &info), ncclSuccess);
  EXPECT_EQ(info.link_layer, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(info.localGidIndex, 2);
  EXPECT_EQ(std::memcmp(&info.localGid, &expected, sizeof(expected)), 0);
  EXPECT_EQ(query.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, GidInfoQuery_NullInputIsRejected) {
  ibv_context context{};
  ibv_port_attr port{};
  IbMsegGidInfo info{};

  EXPECT_EQ(IbMsegGidInfoQuery(nullptr, 1, &port, &info), ncclInternalError);
  EXPECT_EQ(IbMsegGidInfoQuery(&context, 1, nullptr, &info), ncclInternalError);
  EXPECT_EQ(IbMsegGidInfoQuery(&context, 1, &port, nullptr), ncclInternalError);
}

TEST_F(NetIbMsegConnectMicrotest, PkeyIndex_ConfiguredIndexSkipsTableQuery) {
  ibv_context context{};
  ibv_port_attr port{};
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_PKEY_VALUE") == 0) return int64_t{-1};
    if (std::strcmp(name, "IB_PKEY") == 0) return int64_t{5};
    return fallback;
  });
  int index = -1;

  EXPECT_EQ(IbMsegGetPkeyIndex(&context, 1, &port, &index), ncclSuccess);
  EXPECT_EQ(index, 5);
}

TEST_F(NetIbMsegConnectMicrotest, PkeyIndex_MembershipBitIsIgnoredWhenMatchingValue) {
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

  EXPECT_EQ(IbMsegGetPkeyIndex(&context, 1, &port, &index), ncclSuccess);
  EXPECT_EQ(index, 1);
  EXPECT_EQ(query.calls, 2);
}

TEST_F(NetIbMsegConnectMicrotest, PkeyIndex_MissingValueReturnsInvalidUsage) {
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

  EXPECT_EQ(IbMsegGetPkeyIndex(&context, 1, &port, &index), ncclInvalidUsage);
}

TEST_F(NetIbMsegConnectMicrotest, InitAndDestroyCommDevBase_SharedPdIsAllocatedOnceAndReleasedByLastBase) {
  ibv_context context{};
  ibv_pd pd{};
  std::array<ibv_cq, 2> cqs{};
  IbMsegDevs[0].context = &context;
  IbMsegDevs[0].portNum = 2;
  IbMsegDevs[0].gidInfo.localGidIndex = 3;
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
  IbMsegNetCommDevBase first{};
  IbMsegNetCommDevBase second{};

  ASSERT_EQ(IbMsegInitCommDevBase(0, &first, reinterpret_cast<void*>(0x1234), 32), ncclSuccess);
  ASSERT_EQ(IbMsegInitCommDevBase(0, &second, reinterpret_cast<void*>(0x1234), 32), ncclSuccess);
  EXPECT_EQ(alloc.calls, 1);
  EXPECT_EQ(IbMsegDevs[0].pdRefs, 2);
  for (const IbMsegNetCommDevBase* base : {&first, &second}) {
    EXPECT_EQ(base->ibDevN, 0);
    EXPECT_EQ(base->pd, &pd);
    EXPECT_EQ(base->gidInfo.localGidIndex, 3);
    EXPECT_EQ(base->pkeyIndex, 5);
  }
  EXPECT_EQ(first.cq, &cqs[0]);
  EXPECT_EQ(second.cq, &cqs[1]);

  EXPECT_EQ(IbMsegDestroyBase(&first), ncclSuccess);
  EXPECT_EQ(IbMsegDevs[0].pdRefs, 1);
  EXPECT_EQ(dealloc.calls, 0);
  EXPECT_EQ(IbMsegDestroyBase(&second), ncclSuccess);
  EXPECT_EQ(IbMsegDevs[0].pdRefs, 0);
  EXPECT_EQ(destroyCq.calls, 2);
  EXPECT_EQ(dealloc.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, QpInit_ForwardsConfiguredStatePkeyPortAndAccess) {
  ibv_qp verbsQp{};
  IbMsegQp qp{};
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

  EXPECT_EQ(IbMsegQpInit(&qp), ncclSuccess);
  EXPECT_EQ(modify.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, QpCreate_StandardPathForwardsCapabilities) {
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
  IbMsegQpCreateAttr attrs{};
  attrs.pd = &pd;
  attrs.cq = &cq;
  attrs.type = IBV_QPT_RC;
  attrs.maxSendWorkRequest = 17;
  attrs.maxRecvWorkRequest = 19;
  IbMsegQp qp{};

  ASSERT_EQ(IbMsegQpCreate(&qp, &attrs), ncclSuccess);
  EXPECT_EQ(qp.qp, &verbsQp);
}

TEST_F(NetIbMsegConnectMicrotest, QpCreate_OooPathUsesMlx5ExtendedCreation) {
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
  IbMsegQpCreateAttr attrs{};
  attrs.pd = &pd;
  attrs.type = IBV_QPT_RC;
  attrs.oooRq = true;
  IbMsegQp qp{};

  EXPECT_EQ(IbMsegQpCreate(&qp, &attrs), ncclSuccess);
  EXPECT_EQ(qp.qp, &verbsQp);
  EXPECT_EQ(create.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, SenderQps_EceEnabledQueriesAndAppliesReducedEce) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  g_loadParam = [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_ECE_ENABLE") == 0) return int64_t{1};
    return fallback;
  };
  IbMsegSendComm comm{};
  comm.base.vProps.ndevs = 1;
  comm.base.nqps = 1;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[0].base.pd = &pd_;
  comm.devs[0].base.cq = &cq_;
  comm.devs[0].base.pkeyIndex = 0;
  comm.devs[0].base.gidInfo = IbMsegDevs[0].gidInfo;
  IbMsegConnectionMetadata local{};
  ScopedHook query(g_ibvQueryEce, [](ibv_qp*, ibv_ece* ece, int* supported) {
    ece->vendor_id = 0x1234;
    ece->options = 0x55;
    *supported = 1;
    return ncclSuccess;
  });

  ASSERT_EQ(IbMsegSenderQpsCreate(&comm, &local), ncclSuccess);
  EXPECT_EQ(local.qpInfo[0].ece_supported, 1);
  EXPECT_EQ(local.qpInfo[0].ece.vendor_id, 0x1234u);
  EXPECT_EQ(query.calls, 1);

  IbMsegConnectionMetadata remote{};
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

  EXPECT_EQ(IbMsegSenderQpsToRts(&comm, &remote), ncclSuccess);
  EXPECT_EQ(set.calls, 1);
  EXPECT_EQ(comm.base.qps[0].ece.vendor_id, 0x4321u);
  ASSERT_EQ(rtrs.size(), 1u);
  EXPECT_EQ(rtrs[0].dest_qp_num, 200u);
  EXPECT_EQ(std::memcmp(&rtrs[0].ah_attr.grh.dgid, &remote.devs[0].gid, sizeof(remote.devs[0].gid)), 0);
  EXPECT_EQ(rtrs[0].ah_attr.sl, 2);
  EXPECT_EQ(rtrs[0].ah_attr.grh.traffic_class, 3);
}

TEST_F(NetIbMsegConnectMicrotest, QpRtr_RcRoceUsesGlobalRouteAndRcOnlyAttributes) {
  ibv_qp verbsQp{};
  verbsQp.qp_type = IBV_QPT_RC;
  IbMsegQp qp{};
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

  EXPECT_EQ(IbMsegQpRtr(&qp), ncclSuccess);
}

TEST_F(NetIbMsegConnectMicrotest, QpRtr_UcSameSubnetIbUsesLidAndOmitsRcOnlyAttributes) {
  ibv_qp verbsQp{};
  verbsQp.qp_type = IBV_QPT_UC;
  IbMsegQp qp{};
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

  EXPECT_EQ(IbMsegQpRtr(&qp), ncclSuccess);
}

TEST_F(NetIbMsegConnectMicrotest, QpRts_RcUsesRetryAttributesWhileUcOmitsThem) {
  ibv_qp verbsQp{};
  IbMsegQp qp{};
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
  ASSERT_EQ(IbMsegQpRts(&qp), ncclSuccess);
  verbsQp.qp_type = IBV_QPT_UC;
  ASSERT_EQ(IbMsegQpRts(&qp), ncclSuccess);
  ASSERT_EQ(masks.size(), 2u);
  EXPECT_NE(masks[0] & IBV_QP_TIMEOUT, 0);
  EXPECT_EQ(masks[1] & IBV_QP_TIMEOUT, 0);
}

TEST_F(NetIbMsegConnectMicrotest, QpResetAndErrorTransitionToRequestedState) {
  ibv_qp verbsQp{};
  IbMsegQp qp{};
  qp.qp = &verbsQp;
  std::vector<ibv_qp_state> states;
  ScopedHook modify(g_ibvModifyQp, [&states](ibv_qp*, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(mask, IBV_QP_STATE);
    states.push_back(attr->qp_state);
    return ncclSuccess;
  });

  EXPECT_EQ(IbMsegQpReset(&qp), ncclSuccess);
  EXPECT_EQ(IbMsegQpError(&qp), ncclSuccess);
  EXPECT_EQ(states, (std::vector<ibv_qp_state>{IBV_QPS_RESET, IBV_QPS_ERR}));
}

TEST_F(NetIbMsegConnectMicrotest, FindDevBySubnet_InvalidPrefixOrNoValidRemoteKeepsDefault) {
  union ibv_gid remote[2] = {};
  int found = -1;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_SUBNET_PREFIX_LEN") == 0 ? int64_t{0} : fallback;
  });
  EXPECT_EQ(IbMsegFindDevBySubnet(remote, 2, 3, &found), ncclInvalidArgument);
  EXPECT_EQ(found, 3);

  ScopedHook validPrefix(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_SUBNET_PREFIX_LEN") == 0 ? int64_t{24} : fallback;
  });
  EXPECT_EQ(IbMsegFindDevBySubnet(remote, 2, 3, &found), ncclSuccess);
  EXPECT_EQ(found, 3);
}

TEST_F(NetIbMsegConnectMicrotest, FindDevBySubnet_DefaultFullMatchPreservesMergedDevice) {
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

  EXPECT_EQ(IbMsegFindDevBySubnet(&remote, 1, 0, &found), ncclSuccess);
  EXPECT_EQ(found, 0);
  EXPECT_EQ(query.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, FindDevBySubnet_DefaultMismatchSelectsFullyReachableAlternative) {
  ConfigureSingleDevice();
  ibv_context secondContext{};
  IbMsegNDevs = 2;
  IbMsegNMergedDevs = 2;
  IbMsegDevs[1].context = &secondContext;
  IbMsegDevs[1].portNum = 2;
  IbMsegDevs[1].portAttr.link_layer = IBV_LINK_LAYER_ETHERNET;
  IbMsegMergedDevs[1].vProps = {};
  IbMsegMergedDevs[1].vProps.ndevs = 1;
  IbMsegMergedDevs[1].vProps.devs[0] = 1;
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

  EXPECT_EQ(IbMsegFindDevBySubnet(&remote, 1, 0, &found), ncclSuccess);
  EXPECT_EQ(found, 1);
  EXPECT_EQ(query.calls, 2);
}

TEST_F(NetIbMsegConnectMicrotest, FindDevBySubnet_FusedDefaultWithOnePfOffSubnetIsReplaced) {
  ConfigureSingleDevice();
  // Merged device 0 fuses PFs 0 and 1; merged device 1 is PF 2 alone.
  std::array<ibv_context, 2> otherContexts{};
  IbMsegNDevs = 3;
  IbMsegNMergedDevs = 2;
  for (int pf = 1; pf < 3; ++pf) {
    IbMsegDevs[pf].context = &otherContexts[pf - 1];
    IbMsegDevs[pf].portNum = 1;
    IbMsegDevs[pf].portAttr.link_layer = IBV_LINK_LAYER_ETHERNET;
  }
  IbMsegMergedDevs[0].vProps.ndevs = 2;
  IbMsegMergedDevs[0].vProps.devs[1] = 1;
  IbMsegMergedDevs[1].vProps.ndevs = 1;
  IbMsegMergedDevs[1].vProps.devs[0] = 2;
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

  EXPECT_EQ(IbMsegFindDevBySubnet(&remote, 1, 0, &found), ncclSuccess);
  EXPECT_EQ(found, 1);
  EXPECT_EQ(query.calls, 3);
}

TEST_F(NetIbMsegConnectMicrotest, SubnetMatchesAny_IgnoresInvalidGidsAndMatchesReachablePeer) {
  auto local = Ipv4Gid("10.2.1.1");
  std::array<ibv_gid, 2> remotes{};
  remotes[1] = Ipv4Gid("10.2.9.9");

  EXPECT_TRUE(subnetMatchesAny(&local, remotes.data(), remotes.size(), 16));
  EXPECT_FALSE(subnetMatchesAny(&local, remotes.data(), remotes.size(), 24));
}

TEST_F(NetIbMsegConnectMicrotest, UpdateGidIndex_CandidateWithRequestedFamilyAndSubnetIsSelected) {
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

TEST_F(NetIbMsegConnectMicrotest, UpdateGidIndex_SameFamilyCandidateIsTakenOnlyForConfiguredRoceVersion) {
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

TEST_F(NetIbMsegConnectMicrotest, Listen_SuccessPublishesHandleAndCloseReleasesComm) {
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
  IbMsegHandle handle{};
  void* comm = nullptr;

  ASSERT_EQ(IbMsegListen(nullptr, 0, &handle, &comm), ncclSuccess);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(handle.magic, NCCL_SOCKET_MAGIC);
  EXPECT_EQ(handle.connectAddr.sa.sa_family, AF_INET);
  EXPECT_EQ(IbMsegCloseListen(comm), ncclSuccess);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, Listen_SetupFailureClosesSocketAndDoesNotPublishComm) {
  ScopedHook init(g_ncclSocketInit,
                  [](ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType, volatile uint32_t*, int, int) {
                    return ncclSystemError;
                  });
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  IbMsegHandle handle{};
  void* comm = nullptr;

  EXPECT_EQ(IbMsegListen(nullptr, 0, &handle, &comm), ncclSystemError);
  EXPECT_EQ(comm, nullptr);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, Connect_NotReadyThenCompletedHandshakePublishesSendComm) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  IbMsegHandle handle{};
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

  ASSERT_EQ(IbMsegConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1,
                              /*envTrafficClass=*/17),
            ncclSuccess);
  EXPECT_EQ(sendComm, nullptr);
  EXPECT_EQ(handle.stage.state, IbMsegCommStateConnect);
  ASSERT_NE(handle.stage.comm, nullptr);

  void* const stageComm = handle.stage.comm;
  for (int step = 0; step < 10 && sendComm == nullptr; ++step) {
    ASSERT_EQ(IbMsegConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1,
                                /*envTrafficClass=*/17),
              ncclSuccess);
    if (sendComm == nullptr) EXPECT_EQ(handle.stage.comm, stageComm);
  }
  auto* comm = static_cast<IbMsegSendComm*>(sendComm);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->base.ready, 1);
  EXPECT_EQ(comm->base.nqps, 1);
  EXPECT_EQ(comm->base.nDataQps, 1);
  EXPECT_EQ(comm->base.nRemDevs, 1);
  EXPECT_EQ(comm->base.qps[0].remDevIdx, 0);
  EXPECT_EQ(handle.stage.state, IbMsegCommStateStart);
  EXPECT_GE(peer.sendCalls, 3);
  EXPECT_GE(peer.recvCalls, 2);
  EXPECT_EQ(peer.partialStages.size(), 5u);
  EXPECT_EQ(peer.sentIsP2p, 1);

  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectMicrotest, ConnectWrapper_UnexpectedStageIsRejected) {
  IbMsegHandle handle{};
  handle.stage.state = IbMsegCommStateAccept;
  void* comm = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegConnect(nullptr, 0, &handle, &comm, nullptr), ncclInternalError);
  EXPECT_EQ(comm, nullptr);
}

TEST_F(NetIbMsegConnectMicrotest, Accept_CompletedHandshakePublishesRecvCommAndClearsStage) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  const auto remoteGid = Ipv6Gid("2001:db8::2");
  std::vector<ibv_qp_attr> rtrs;
  ScopedHook modify(g_ibvModifyQp, [&rtrs](ibv_qp*, ibv_qp_attr* attr, int) {
    if (attr->qp_state == IBV_QPS_RTR) rtrs.push_back(*attr);
    return ncclSuccess;
  });
  IbMsegListenComm listenComm{};
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
    ASSERT_EQ(IbMsegAcceptImpl(&listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1), ncclSuccess);
  }
  auto* comm = static_cast<IbMsegRecvComm*>(recvComm);
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

  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectMicrotest, AcceptWrapper_UnexpectedStageIsRejected) {
  IbMsegListenComm listen{};
  IbMsegCommStage stage{};
  stage.state = IbMsegCommStateConnect;
  listen.stage = &stage;
  void* comm = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegAccept(&listen, &comm, nullptr), ncclInternalError);
  EXPECT_EQ(comm, nullptr);
}

// The cached decision needs the switch, and a GIN or resilient comm never takes it.
TEST(NetIbMsegConnectUseMultiSeg, NeedsTheSwitchAndIsNotGin) {
  EXPECT_TRUE(IbMsegUseMultiSeg(/*multiSegOn*/ true, /*forGin*/ false, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ true, /*forGin*/ true, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ false, /*forGin*/ false, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ false, /*forGin*/ true, /*resilient*/ false));
  EXPECT_FALSE(IbMsegUseMultiSeg(/*multiSegOn*/ true, /*forGin*/ false, /*resilient*/ true));
}

// rcclMultiSegOn true: both ends of a classic comm use the CTS slots; GIN comms keep the legacy FIFO.
class NetIbMsegConnectMultiSegMicrotest : public NetIbMsegConnectMicrotest {
 protected:
  void SetUp() override {
    ConfigureSingleDevice();
    InstallSingleDeviceResourceFakes();
    g_multiSegOn = [this](bool pluginRegistersPerSegment) { return pluginRegistersPerSegment && multiSegOn_; };
  }

  static IbMsegConnectionMetadata Peer(uint64_t ctsAddr, uint32_t qpn, bool useMultiSeg) {
    IbMsegConnectionMetadata meta = PeerMetadata(ctsAddr, qpn, /*lid=*/8, /*rkey=*/0x91, Ipv6Gid("2001:db8::2"));
    meta.useMultiSeg = useMultiSeg ? 1 : 0;
    return meta;
  }

  IbMsegSendComm* Connect(IbMsegHandle* handle) {
    g_ncclSocketConnect = [](ncclSocket*) { return ncclSuccess; };
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 1;
      return ncclSuccess;
    };
    void* sendComm = nullptr;
    for (int step = 0; step < 10 && sendComm == nullptr; ++step) {
      EXPECT_EQ(IbMsegConnectImpl(nullptr, 0, handle, &sendComm, nullptr, /*nQpsPerDev=*/1, /*envTrafficClass=*/17),
                ncclSuccess);
    }
    return static_cast<IbMsegSendComm*>(sendComm);
  }

  IbMsegRecvComm* Accept(IbMsegListenComm* listenComm) {
    g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 1;
      return ncclSuccess;
    };
    void* recvComm = nullptr;
    for (int step = 0; step < 10 && recvComm == nullptr; ++step) {
      EXPECT_EQ(IbMsegAcceptImpl(listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1), ncclSuccess);
    }
    return static_cast<IbMsegRecvComm*>(recvComm);
  }

  const ibv_mr* MrAt(const void* addr) const {
    for (const auto& mr : mrs_) {
      if (mr.addr == addr) return &mr;
    }
    return nullptr;
  }

  bool multiSegOn_ = true;
};

TEST_F(NetIbMsegConnectMultiSegMicrotest, Connect_ClassicCommRegistersCtsSlots) {
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/true));

  IbMsegSendComm* comm = Connect(&handle);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(peer.sentUseMultiSeg, 1);
  ASSERT_NE(comm->ctsSlots, nullptr);
  EXPECT_EQ(comm->ctsFifo, nullptr);
  const ibv_mr* ctsMr = MrAt(comm->ctsSlots);
  ASSERT_NE(ctsMr, nullptr);
  EXPECT_EQ(ctsMr->length, NCCL_IB_CTS_SLOTS_BYTES);
  EXPECT_EQ(comm->useMultiSeg, 1);

  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectMultiSegMicrotest, Connect_GinCommKeepsTheLegacyFifo) {
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  handle.isGin = 1;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/false));

  IbMsegSendComm* comm = Connect(&handle);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->ctsSlots, nullptr);
  EXPECT_EQ(comm->useMultiSeg, 0);
  const ibv_mr* ctsMr = MrAt(comm->ctsFifo);
  ASSERT_NE(ctsMr, nullptr);
  EXPECT_EQ(ctsMr->length, NCCL_IB_CTS_FIFO_BYTES);

  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

// The receiver allocates the slots from its own decision; the sender reaches the same one.
TEST_F(NetIbMsegConnectMultiSegMicrotest, Accept_ClassicListenAllocatesCtsSlotsAndGinListenDoesNot) {
  for (bool isGin : {false, true}) {
    SCOPED_TRACE(isGin);
    IbMsegListenComm listenComm{};
    listenComm.isGin = isGin;
    ScriptedHandshakePeer peer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                               Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/!isGin));

    IbMsegRecvComm* comm = Accept(&listenComm);
    ASSERT_NE(comm, nullptr);
    EXPECT_EQ(comm->useMultiSeg, isGin ? 0 : 1);
    EXPECT_EQ(peer.sentUseMultiSeg, isGin ? 0 : 1);
    EXPECT_EQ(comm->remCtsFifo.addr, 0xa000u);
    if (isGin) {
      EXPECT_EQ(comm->ctsSlots, nullptr);
    } else {
      ASSERT_NE(comm->ctsSlots, nullptr);
      EXPECT_EQ(comm->ctsSlotStride, sizeof(IbMsegCtsSlot));
      ASSERT_NE(comm->devs[0].ctsFifoMr, nullptr);
      EXPECT_EQ(comm->devs[0].ctsFifoMr->addr, static_cast<void*>(comm->ctsSlots));
      EXPECT_EQ(comm->devs[0].ctsFifoMr->length, NCCL_IB_CTS_SLOTS_BYTES);
    }

    EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
  }
}

TEST_F(NetIbMsegConnectMultiSegMicrotest, ConnectAndAccept_SwitchOffKeepTheLegacyCts) {
  multiSegOn_ = false;
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer sendPeer([&handle] { return static_cast<int>(handle.stage.state); },
                                 Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/false));
  IbMsegSendComm* send = Connect(&handle);
  ASSERT_NE(send, nullptr);
  EXPECT_EQ(send->useMultiSeg, 0);
  EXPECT_EQ(send->ctsSlots, nullptr);
  EXPECT_NE(send->ctsFifo, nullptr);
  EXPECT_EQ(IbMsegCloseSend(send), ncclSuccess);

  IbMsegListenComm listenComm{};
  ScriptedHandshakePeer recvPeer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                                 Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/false));
  IbMsegRecvComm* recv = Accept(&listenComm);
  ASSERT_NE(recv, nullptr);
  EXPECT_EQ(recv->useMultiSeg, 0);
  EXPECT_EQ(recv->ctsSlots, nullptr);
  EXPECT_EQ(IbMsegCloseRecv(recv), ncclSuccess);
}

// Only a multi-segment comm pays for the send cursor and layout cache; it drops the receiver's legacy FIFO shadow.
TEST_F(NetIbMsegConnectMultiSegMicrotest, ConnectAndAccept_MultiSegStateFollowsUseMultiSeg) {
  for (bool on : {true, false}) {
    SCOPED_TRACE(on);
    multiSegOn_ = on;
    IbMsegHandle handle{};
    handle.magic = NCCL_SOCKET_MAGIC;
    ScriptedHandshakePeer sendPeer([&handle] { return static_cast<int>(handle.stage.state); },
                                   Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/on));
    IbMsegSendComm* send = Connect(&handle);
    ASSERT_NE(send, nullptr);
    EXPECT_EQ(send->useMultiSeg, on ? 1 : 0);
    EXPECT_EQ(send->sendCursor != nullptr, on);
    EXPECT_EQ(send->layoutCache != nullptr, on);
    EXPECT_EQ(IbMsegCloseSend(send), ncclSuccess);

    IbMsegListenComm listenComm{};
    ScriptedHandshakePeer recvPeer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                                   Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/on));
    IbMsegRecvComm* recv = Accept(&listenComm);
    ASSERT_NE(recv, nullptr);
    EXPECT_EQ(recv->useMultiSeg, on ? 1 : 0);
    if (on) {
      EXPECT_EQ(recv->remCtsFifo.elems, nullptr);
    } else {
      ASSERT_NE(recv->remCtsFifo.elems, nullptr);
      ASSERT_NE(recv->devs[0].ctsFifoMr, nullptr);
      EXPECT_EQ(recv->devs[0].ctsFifoMr->addr, static_cast<void*>(recv->remCtsFifo.elems));
      EXPECT_EQ(recv->devs[0].ctsFifoMr->length, NCCL_IB_CTS_FIFO_BYTES);
    }
    EXPECT_EQ(IbMsegCloseRecv(recv), ncclSuccess);
  }
}

// Resilient comms replay whole slots, so connect and accept keep the legacy CTS with the switch on.
class NetIbMsegConnectResilientMicrotest : public NetIbMsegConnectMultiSegMicrotest {
 protected:
  void SetUp() override {
    NetIbMsegConnectMultiSegMicrotest::SetUp();
    g_ncclIbResiliencyInit = [this](IbMsegNetCommBase*, IbMsegResiliency** out) {
      *out = &resiliency_;
      return ncclSuccess;
    };
    g_ncclSocketConnect = [](ncclSocket*) { return ncclSuccess; };
    g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
    // The decision is made before the socket is ready, so the first step shows it.
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 0;
      return ncclSuccess;
    };
  }

  IbMsegResiliency resiliency_{};
};

TEST_F(NetIbMsegConnectResilientMicrotest, Connect_ResilientCommKeepsTheLegacyCts) {
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/false));
  void* sendComm = nullptr;
  ASSERT_EQ(IbMsegConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1, /*envTrafficClass=*/17),
            ncclSuccess);
  EXPECT_EQ(sendComm, nullptr);
  auto* comm = static_cast<IbMsegSendComm*>(handle.stage.comm);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->base.resiliency, &resiliency_);
  EXPECT_EQ(comm->useMultiSeg, 0);
  EXPECT_EQ(comm->ctsSlots, nullptr);
  EXPECT_NE(comm->ctsFifo, nullptr);
  EXPECT_EQ(comm->sendCursor, nullptr);
  EXPECT_EQ(comm->layoutCache, nullptr);
  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectResilientMicrotest, Accept_ResilientCommKeepsTheLegacyCts) {
  IbMsegListenComm listenComm{};
  ScriptedHandshakePeer peer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                             Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/false));
  void* recvComm = nullptr;
  ASSERT_EQ(IbMsegAcceptImpl(&listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1), ncclSuccess);
  EXPECT_EQ(recvComm, nullptr);
  ASSERT_NE(listenComm.stage, nullptr);
  auto* comm = static_cast<IbMsegRecvComm*>(listenComm.stage->comm);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->base.resiliency, &resiliency_);
  EXPECT_EQ(comm->useMultiSeg, 0);
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
  free(listenComm.stage->buffer);
  free(listenComm.stage);
  listenComm.stage = nullptr;
}

TEST_F(NetIbMsegConnectMultiSegMicrotest, Connect_CuMemOffStillUsesCtsSlots) {
  g_cuMemEnable = [] { return 0; };
  IbMsegHandle handle{};
  handle.magic = NCCL_SOCKET_MAGIC;
  ScriptedHandshakePeer peer([&handle] { return static_cast<int>(handle.stage.state); },
                             Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/true));

  IbMsegSendComm* comm = Connect(&handle);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->useMultiSeg, 1);
  EXPECT_NE(comm->ctsSlots, nullptr);

  EXPECT_EQ(IbMsegCloseSend(comm), ncclSuccess);
}

// A peer whose multi-segment setting differs fails the connect or accept instead of misreading the CTS.
TEST_F(NetIbMsegConnectMultiSegMicrotest, ConnectAndAccept_PeerWithTheOtherSettingFails) {
  for (bool localOn : {true, false}) {
    SCOPED_TRACE(localOn);
    multiSegOn_ = localOn;
    g_ncclSocketConnect = [](ncclSocket*) { return ncclSuccess; };
    g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 1;
      return ncclSuccess;
    };

    IbMsegHandle handle{};
    handle.magic = NCCL_SOCKET_MAGIC;
    ScriptedHandshakePeer sendPeer([&handle] { return static_cast<int>(handle.stage.state); },
                                   Peer(/*ctsAddr=*/0x9000, /*qpn=*/200, /*useMultiSeg=*/!localOn));
    void* sendComm = nullptr;
    ncclResult_t sendRet = ncclSuccess;
    for (int step = 0; step < 10 && sendComm == nullptr && sendRet == ncclSuccess; ++step) {
      sendRet = IbMsegConnectImpl(nullptr, 0, &handle, &sendComm, nullptr, /*nQpsPerDev=*/1, /*envTrafficClass=*/17);
    }
    EXPECT_EQ(sendRet, ncclInvalidUsage);
    EXPECT_EQ(sendComm, nullptr);

    IbMsegListenComm listenComm{};
    ScriptedHandshakePeer recvPeer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                                   Peer(/*ctsAddr=*/0xa000, /*qpn=*/300, /*useMultiSeg=*/!localOn));
    void* recvComm = nullptr;
    ncclResult_t recvRet = ncclSuccess;
    for (int step = 0; step < 10 && recvComm == nullptr && recvRet == ncclSuccess; ++step) {
      recvRet = IbMsegAcceptImpl(&listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1);
    }
    EXPECT_EQ(recvRet, ncclInvalidUsage);
    EXPECT_EQ(recvComm, nullptr);
    EXPECT_EQ(recvPeer.sentUseMultiSeg, -1); // the accept fails before it replies
  }
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
// Accept's GDR flush scratchpad under cuMem: a VMM buffer registered through its own DMA-BUF fd, or the HSA buffer
// when any step of that fails. rcclParamIbGdrFlushGpuMemNoRelaxedOrdering, which gates both, defaults on.
class NetIbMsegConnectCuMemFlushMicrotest : public NetIbMsegConnectMicrotest {
 protected:
  struct DmaBufReg {
    ibv_pd* pd;
    uint64_t offset;
    size_t length;
    uint64_t iova;
    int fd;
    int access;
    ibv_mr* mr;
  };

  void SetUp() override {
    ConfigureSingleDevice();
    InstallSingleDeviceResourceFakes();
    vmm_.Install();
    exports_.Install();
    g_hipDeviceGetPCIBusIdResult = hipSuccess;  // ncclCuMemAlloc looks its side stream up by bus id
    g_cuMemEnable = [] { return 1; };
    g_netIbGdrSupport = [] { return ncclSuccess; };  // peermem, so accept enables the flush
    g_connectMemAlloc = [](void** ptr, size_t size) {
      return ncclCuMemAlloc(ptr, nullptr, hipMemHandleTypePosixFileDescriptor, size);
    };
    g_ibvRegDmabufMr = [this](ibv_mr** out, ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd,
                              int access) {
      ibv_mr* mr = dmabufRegResult_ == ncclSuccess ? &mrs_.emplace_back() : nullptr;
      dmabufRegs_.push_back({pd, offset, length, iova, fd, access, mr});
      if (mr == nullptr) return dmabufRegResult_;
      *out = mr;
      return ncclSuccess;
    };
    g_connectClose = [this](int fd) {
      closed_.push_back(fd);
      return ::close(fd);
    };
  }

  void TearDown() override {
    exports_.CloseOpen();
    NetIbMsegConnectMicrotest::TearDown();
  }

  // Runs accept to completion against a one-device peer.
  ncclResult_t Accept(IbMsegRecvComm** comm) {
    IbMsegListenComm listenComm{};
    ScriptedHandshakePeer peer([&listenComm] { return static_cast<int>(listenComm.stage->state); },
                               PeerMetadata(/*ctsAddr=*/0xa000, /*qpn=*/300, /*lid=*/9, /*rkey=*/0xa1,
                                            Ipv6Gid("2001:db8::2")));
    g_ncclSocketAccept = [](ncclSocket*, ncclSocket*, bool) { return ncclSuccess; };
    g_ncclSocketReady = [](ncclSocket*, int* ready) {
      *ready = 1;
      return ncclSuccess;
    };
    void* recvComm = nullptr;
    for (int step = 0; step < 10 && recvComm == nullptr; ++step) {
      ncclResult_t ret = IbMsegAcceptImpl(&listenComm, &recvComm, nullptr, /*nQpsPerDev=*/1);
      if (ret != ncclSuccess) return ret;
    }
    *comm = static_cast<IbMsegRecvComm*>(recvComm);
    return *comm != nullptr ? ncclSuccess : ncclInternalError;
  }

  // The HSA buffer, registered through peermem, with nothing of the cuMem attempt left live.
  void ExpectHsaFallback(const IbMsegRecvComm& comm) {
    const IbMsegGpuFlush& flush = comm.devs[0].gpuFlush;
    EXPECT_TRUE(flush.gpuFlushMemIsHipAlloc);
    EXPECT_EQ(vmm_.liveBuffers.count(flush.gpuFlushGpuMem), 1u);
    ASSERT_NE(flush.gpuMr, nullptr);
    EXPECT_EQ(flush.gpuMr->addr, flush.gpuFlushGpuMem);
    EXPECT_EQ(flush.gpuMr->length, sizeof(int));
    EXPECT_EQ(flush.dmabuf_fd, -1);
    EXPECT_TRUE(vmm_.reserved.empty());
    EXPECT_TRUE(vmm_.liveHandles.empty());
  }

  // Closes the comm and returns every MR it deregistered.
  std::vector<ibv_mr*> CloseRecvDeregs(IbMsegRecvComm* comm) {
    std::vector<ibv_mr*> deregs;
    ScopedHook dereg(g_ibvDeregMr, [&deregs](ibv_mr* mr) {
      deregs.push_back(mr);
      return ncclSuccess;
    });
    EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
    return deregs;
  }

  HipVmmLedger vmm_;
  CuMemExportRecorder exports_;
  ncclResult_t dmabufRegResult_ = ncclSuccess;
  std::vector<DmaBufReg> dmabufRegs_;
  std::vector<int> closed_;
};

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, Accept_GdrFlushGpuMemOff_RegistersNoScratchpad) {
  auto params = g_loadParam;
  ScopedHook scratchpadOff(g_loadParam, [params](const char* name, int64_t fallback) {
    if (std::strcmp(name, "RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING") == 0) return int64_t{0};
    return params(name, fallback);
  });
  auto vmmAlloc = g_connectMemAlloc;
  ScopedHook alloc(g_connectMemAlloc, vmmAlloc);
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);

  EXPECT_EQ(comm->flushEnabled, 1);
  const IbMsegGpuFlush& flush = comm->devs[0].gpuFlush;
  EXPECT_EQ(flush.gpuFlushGpuMem, nullptr);
  EXPECT_EQ(flush.gpuMr, nullptr);
  EXPECT_EQ(flush.dmabuf_fd, -1);
  EXPECT_EQ(alloc.calls, 0);
  EXPECT_TRUE(exports_.empty());
  EXPECT_TRUE(vmm_.liveBuffers.empty());
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, Accept_CuMemOff_RegistersTheHsaBuffer) {
  ScopedHook cuMemOff(g_cuMemEnable, [] { return 0; });
  auto vmmAlloc = g_connectMemAlloc;
  ScopedHook alloc(g_connectMemAlloc, vmmAlloc);
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);

  ExpectHsaFallback(*comm);
  EXPECT_EQ(alloc.calls, 0);
  EXPECT_TRUE(exports_.empty());
  EXPECT_TRUE(dmabufRegs_.empty());
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, Accept_CuMemStepsSucceed_RegistersTheVmmBufferThroughItsDmaBufFd) {
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);
  ASSERT_EQ(comm->flushEnabled, 1);
  const IbMsegGpuFlush& flush = comm->devs[0].gpuFlush;

  ASSERT_EQ(exports_.size(), 1u);
  EXPECT_EQ(exports_[0].ptr, flush.gpuFlushGpuMem);
  EXPECT_EQ(exports_[0].size, sizeof(int));
  EXPECT_EQ(exports_[0].type, static_cast<int>(hipMemRangeHandleTypeDmaBufFd));
  EXPECT_EQ(exports_[0].flags, 0u);
  EXPECT_EQ(vmm_.reserved.count(flush.gpuFlushGpuMem), 1u);
  EXPECT_TRUE(vmm_.liveBuffers.empty());
  ASSERT_EQ(dmabufRegs_.size(), 1u);
  EXPECT_EQ(dmabufRegs_[0].pd, comm->devs[0].base.pd);
  EXPECT_EQ(dmabufRegs_[0].offset, 0u);
  EXPECT_EQ(dmabufRegs_[0].length, sizeof(int));
  EXPECT_EQ(dmabufRegs_[0].iova, reinterpret_cast<uint64_t>(flush.gpuFlushGpuMem));
  EXPECT_EQ(dmabufRegs_[0].fd, exports_[0].fd);
  // Strictly ordered: no IBV_ACCESS_RELAXED_ORDERING.
  EXPECT_EQ(dmabufRegs_[0].access, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  EXPECT_EQ(flush.gpuMr, dmabufRegs_[0].mr);
  EXPECT_EQ(flush.dmabuf_fd, exports_[0].fd);
  EXPECT_FALSE(flush.gpuFlushMemIsHipAlloc);
  EXPECT_TRUE(closed_.empty());
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, Accept_VmmAllocFails_FallsBackToTheHsaBuffer) {
  ScopedHook alloc(g_connectMemAlloc, [](void** ptr, size_t) {
    *ptr = nullptr;
    return ncclSystemError;
  });
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);

  ExpectHsaFallback(*comm);
  EXPECT_EQ(alloc.calls, 1);
  EXPECT_TRUE(exports_.empty());
  EXPECT_TRUE(dmabufRegs_.empty());
  EXPECT_TRUE(closed_.empty());
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, Accept_DmaBufExportFails_FreesTheVmmBufferAndFallsBack) {
  exports_.result = hipErrorNotSupported;
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);

  ExpectHsaFallback(*comm);
  EXPECT_EQ(vmm_.addressFrees.size(), 1u);
  EXPECT_TRUE(dmabufRegs_.empty());
  EXPECT_TRUE(closed_.empty());
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, Accept_DmaBufRegistrationFails_ClosesTheFdFreesTheVmmBufferAndFallsBack) {
  dmabufRegResult_ = ncclSystemError;
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);

  ExpectHsaFallback(*comm);
  ASSERT_EQ(exports_.size(), 1u);
  ASSERT_EQ(dmabufRegs_.size(), 1u);
  EXPECT_EQ(closed_, std::vector<int>{exports_[0].fd});
  EXPECT_EQ(vmm_.addressFrees.size(), 1u);
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, Accept_DmaBufRegistrationFailsOnFdZero_ClosesFdZero) {
  dmabufRegResult_ = ncclSystemError;
  ScopedHook exportFdZero(g_hipMemGetHandleForAddressRange,
                          [](void* handle, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long) {
                            *static_cast<int*>(handle) = 0;
                            return hipSuccess;
                          });
  // Records fd 0 without closing the runner's stdin.
  ScopedHook keepStdin(g_connectClose, [this](int fd) {
    closed_.push_back(fd);
    return 0;
  });
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);

  ExpectHsaFallback(*comm);
  ASSERT_EQ(dmabufRegs_.size(), 1u);
  EXPECT_EQ(dmabufRegs_[0].fd, 0);
  EXPECT_EQ(closed_, std::vector<int>{0});
  EXPECT_EQ(IbMsegCloseRecv(comm), ncclSuccess);
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, CloseRecv_VmmFlushBuffer_DeregistersItsMrClosesItsFdAndFreesIt) {
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);
  ASSERT_EQ(dmabufRegs_.size(), 1u);
  ASSERT_EQ(exports_.size(), 1u);

  const std::vector<ibv_mr*> deregs = CloseRecvDeregs(comm);
  EXPECT_EQ(std::count(deregs.begin(), deregs.end(), dmabufRegs_[0].mr), 1);
  EXPECT_EQ(closed_, std::vector<int>{exports_[0].fd});
  EXPECT_TRUE(vmm_.Clean()) << vmm_.Summary();
}

TEST_F(NetIbMsegConnectCuMemFlushMicrotest, CloseRecv_HsaFlushBuffer_DeregistersItsMrAndFreesIt) {
  ScopedHook cuMemOff(g_cuMemEnable, [] { return 0; });
  IbMsegRecvComm* comm = nullptr;
  ASSERT_EQ(Accept(&comm), ncclSuccess);
  ibv_mr* const mr = comm->devs[0].gpuFlush.gpuMr;
  ASSERT_NE(mr, nullptr);

  const std::vector<ibv_mr*> deregs = CloseRecvDeregs(comm);
  EXPECT_EQ(std::count(deregs.begin(), deregs.end(), mr), 1);
  EXPECT_TRUE(closed_.empty());
  EXPECT_TRUE(vmm_.Clean()) << vmm_.Summary();
}
#endif

TEST_F(NetIbMsegConnectMicrotest, CreateFlushQp_DisabledOrAlreadyCreatedIsNoOp) {
  IbMsegRecvComm comm{};
  comm.base.vProps.ndevs = 1;
  ScopedHook create(g_ibvCreateQp, [](ibv_qp**, ibv_pd*, ibv_qp_init_attr*) {
    ADD_FAILURE() << "guarded no-op must not create a QP";
    return ncclSystemError;
  });

  EXPECT_EQ(IbMsegCreateFlushQp(&comm), ncclSuccess);
  comm.flushEnabled = 1;
  comm.flushQpsCreated = true;
  EXPECT_EQ(IbMsegCreateFlushQp(&comm), ncclSuccess);
  EXPECT_EQ(create.calls, 0);
}

TEST_F(NetIbMsegConnectMicrotest, CreateFlushQp_FirstUseCreatesLoopbackQpAndHostMr) {
  ConfigureSingleDevice();
  InstallSingleDeviceResourceFakes();
  IbMsegRecvComm comm{};
  comm.flushEnabled = 1;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base.ibDevN = 0;
  comm.devs[0].base.pd = &pd_;
  comm.devs[0].base.cq = &cq_;
  comm.devs[0].base.pkeyIndex = 4;
  comm.devs[0].base.gidInfo = IbMsegDevs[0].gidInfo;

  ASSERT_EQ(IbMsegCreateFlushQp(&comm), ncclSuccess);
  EXPECT_TRUE(comm.flushQpsCreated);
  ASSERT_NE(comm.devs[0].gpuFlush.qp.qp, nullptr);
  EXPECT_EQ(comm.devs[0].gpuFlush.qp.rtrAttr.remoteQpNum, comm.devs[0].gpuFlush.qp.qp->qp_num);
  EXPECT_NE(comm.devs[0].gpuFlush.hostMr, nullptr);
  EXPECT_EQ(comm.devs[0].gpuFlush.sge.lkey, comm.devs[0].gpuFlush.hostMr->lkey);
}

TEST_F(NetIbMsegConnectMicrotest, PostReceiveWorkRequests_ResiliencyDepthControlsPostCount) {
  IbVerbsScene verbs("mlx5_0", 71);
  IbMsegRecvComm comm{};
  IbMsegResiliency resiliency{};
  comm.base.resiliency = &resiliency;
  IbMsegQp qp{};
  qp.qp = &verbs.qps[0];
  qp.devIndex = 0;
  g_ncclIbResiliencyDataRqSizeGet = [](IbMsegResiliency*, uint, uint32_t* size) {
    *size = 3;
    return ncclSuccess;
  };

  EXPECT_EQ(IbMsegPostReceiveWorkRequestsOnQp(&comm, &qp), ncclSuccess);
  EXPECT_EQ(verbs.postRecvQps, std::vector<ibv_qp*>(3, &verbs.qps[0]));
}

TEST_F(NetIbMsegConnectMicrotest, ReceiverPrePostReceiveWorkRequests_VisitsEveryQp) {
  IbVerbsScene verbs("mlx5_0", 72);
  IbMsegRecvComm comm{};
  comm.base.nqps = 2;
  comm.base.qps[0].qp = &verbs.qps[0];
  comm.base.qps[1].qp = &verbs.qps[1];

  EXPECT_EQ(IbMsegReceiverPrePostReceiveWorkRequests(&comm), ncclSuccess);
  const auto& posts = verbs.postRecvQps;
  EXPECT_EQ(std::count(posts.begin(), posts.end(), &verbs.qps[0]), NET_IB_MAX_REQUESTS);
  EXPECT_EQ(std::count(posts.begin(), posts.end(), &verbs.qps[1]), NET_IB_MAX_REQUESTS);
  EXPECT_EQ(posts.size(), 2u * NET_IB_MAX_REQUESTS);
}

TEST_F(NetIbMsegConnectMicrotest, FreeGpuFlushMem_HipAllocationIsReleasedAndStateCleared) {
  IbMsegGpuFlush flush{};
  flush.gpuFlushGpuMem = static_cast<int*>(std::malloc(sizeof(int)));
  ASSERT_NE(flush.gpuFlushGpuMem, nullptr);
  void* const allocation = flush.gpuFlushGpuMem;
  flush.gpuFlushMemIsHipAlloc = true;
  ScopedHook release(g_hipFree, [allocation](void* ptr) {
    EXPECT_EQ(ptr, allocation);
    std::free(ptr);
    return hipSuccess;
  });

  EXPECT_EQ(IbMsegFreeGpuFlushMem(&flush), ncclSuccess);
  EXPECT_EQ(flush.gpuFlushGpuMem, nullptr);
  EXPECT_FALSE(flush.gpuFlushMemIsHipAlloc);
  EXPECT_EQ(release.calls, 1);
}

TEST_F(NetIbMsegConnectMicrotest, CloseFunctions_NullCommunicatorsAreNoOps) {
  EXPECT_EQ(IbMsegCloseSend(nullptr), ncclSuccess);
  EXPECT_EQ(IbMsegCloseRecv(nullptr), ncclSuccess);
  EXPECT_EQ(IbMsegCloseListen(nullptr), ncclSuccess);
}

// Discovery keeps the device's max_mr; multi-segment registration admits against it.
TEST_F(NetIbMsegConnectMicrotest, InitDevices_RecordsTheDeviceMaxMr) {
  IbVerbsWorld world;
  FakeHca& hca = world.AddHca("mlx5_0");
  hca.ports[0] = IbVerbsWorld::ActivePort(IBV_LINK_LAYER_INFINIBAND);
  hca.attr.max_mr = 4321;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_GID_INDEX") == 0 ? int64_t{0} : fallback;
  });
  ScopedHook gid(g_ibvQueryGid, [](ibv_context*, uint8_t, int, ibv_gid* out) {
    std::memset(out, 0, sizeof(*out));
    out->raw[15] = 0xaa;
    return ncclSuccess;
  });

  ASSERT_EQ(IbMsegInitDevices(nullptr, nullptr), ncclSuccess);

  ASSERT_EQ(IbMsegNDevs, 1);
  EXPECT_STREQ(IbMsegDevs[0].devName, "mlx5_0");
  EXPECT_EQ(IbMsegDevs[0].maxMr, 4321);
  EXPECT_EQ(IbMsegDevs[0].maxQp, 1024);
}

bool ReadWholeFile(const std::string& path, std::string* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::ostringstream text;
  text << in.rdbuf();
  *out = text.str();
  return true;
}

std::string WithoutWhitespace(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (!std::isspace(static_cast<unsigned char>(c))) out += c;
  }
  return out;
}

// Parenthesis depth change over one line, ignoring parentheses inside string literals.
int ParenDelta(const std::string& line) {
  int delta = 0;
  bool inString = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (inString) {
      if (c == '\\') ++i;
      else if (c == '"') inString = false;
    } else if (c == '"') {
      inString = true;
    } else {
      delta += (c == '(') - (c == ')');
    }
  }
  return delta;
}

// The hand-copied net_ib parameters in fakes/net_ib_mseg_params*.cc must not drift: every statement, whitespace
// aside, still appears in the net_ib file its "// src/transport/net_ib/<file>" header names.
TEST(NetIbMsegParamCopies, EveryCopiedStatementMatchesItsNetIbSource) {
  static const char* const kCopies[] = {"net_ib_mseg_params.cc", "net_ib_mseg_params_connect.cc",
                                        "net_ib_mseg_params_p2p.cc"};
  const std::string kHeader = "// src/transport/net_ib/";
  int checked = 0;
  for (const char* copy : kCopies) {
    SCOPED_TRACE(copy);
    std::string text;
    ASSERT_TRUE(ReadWholeFile(std::string(NET_IB_MSEG_PARAM_COPIES_DIR) + "/" + copy, &text));
    std::istringstream lines(text);
    std::string line, source, stmt;
    int depth = 0;
    while (std::getline(lines, line)) {
      if (stmt.empty()) {
        if (line.rfind(kHeader, 0) == 0) {
          const std::string file = line.substr(kHeader.size());
          ASSERT_TRUE(ReadWholeFile(std::string(NET_IB_SOURCE_DIR) + "/" + file, &source)) << file;
          source = WithoutWhitespace(source);
          continue;
        }
        if (line.find_first_not_of(" \t") == std::string::npos || line.rfind("//", 0) == 0 ||
            line.rfind("/*", 0) == 0 || line.rfind(" *", 0) == 0 || line.rfind("#include", 0) == 0) {
          continue;
        }
      }
      stmt += line + "\n";
      depth += ParenDelta(line);
      if (depth > 0) continue;
      EXPECT_FALSE(source.empty()) << "statement before any net_ib header:\n" << stmt;
      EXPECT_NE(source.find(WithoutWhitespace(stmt)), std::string::npos) << "drifted from net_ib:\n" << stmt;
      ++checked;
      stmt.clear();
      depth = 0;
    }
    EXPECT_TRUE(stmt.empty()) << "unterminated statement:\n" << stmt;
  }
  EXPECT_GT(checked, 0);
}

}  // namespace
