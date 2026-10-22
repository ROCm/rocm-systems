/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/init.cc (NET_IB_CAST_INIT_CC_PATH): HCA
// discovery, virtual-device merging, device properties and the AINIC defaults, over fake verbs.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <limits.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <new>
#include <string>

#include "../common/LogCapture.hpp"
#include "../common/ProcessIsolatedTestRunner.hpp"
#include "fakes/env_fakes.h"
#include "fakes/gdr_peermem_fakes.h"
#include "fakes/gdr_probe_fakes.h"
#include "fakes/net_ib_cast_p2p_fakes.h"
#include "fakes/net_ib_cast_qp_sharing_fakes.h"
#include "fakes/transport_stubs.h"
#include "net-ib-cast-test-common.h"
#include "fakes/param_redirect.h"

// No test host has /sys/class/infiniband/<hca>/device, so init.cc's realpath() of it is renamed
// onto this seam.
static std::function<char*(const char*, char*)> g_initRealpath = [](const char*, char*) -> char* { return nullptr; };
static char* InitTestRealpath(const char* path, char* resolved) { return g_initRealpath(path, resolved); }

#define realpath InitTestRealpath
#include NET_IB_CAST_INIT_CC_PATH
#undef realpath

void NetIbCastResetGlobals() {
  netRefCount = 0;
  for (struct ncclIbDev& dev : IbCastDevs) {
    free(dev.pciPath);
    free(dev.mrCache.slots);
    dev.~ncclIbDev();
    new (&dev) ncclIbDev();
  }
  memset(IbCastMergedDevs, 0, sizeof(IbCastMergedDevs));
  IbCastNDevs = -1;
  IbCastNMergedDevs = -1;
  memset(IbCastIfName, 0, sizeof(IbCastIfName));
  memset(&IbCastIfAddr, 0, sizeof(IbCastIfAddr));
  IbCastRelaxedOrderingEnabled = 0;
  IbCastProfilerFunction = nullptr;
  IbCastAinicRoce = IbCastMultiplaneEnable = IbCastOffloadEnabled = IbCastUseInline = IbCastAinicCtsInlineData = false;
  IbCastGdrFlushDisable = 0;
  IbCastArThreshold = 8192;
  IbCastQpSharingGlobalEnable = false;
  memset(&castGlobalQpSchedParms, 0, sizeof(castGlobalQpSchedParms));
}

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;
using RcclUnitTesting::ScopedDebugLogging;

class NetIbCastInitMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    world_.reset();
    NetIbCastResetGlobals();
    ResetEnvFakes();
    ResetIbvwrapFakes();
    ResetMlx5dvwrapFakes();
    ResetIonicdvwrapFakes();
    ResetSocketFakes();
    ResetNetIbCastConnectFakes();
    ResetNetIbCastP2pFakes();
    ResetNetIbCastQpSharingFakes();
    ResetNetIbCastSchedulerFakes();
    ResetNetIbCastP2pResiliencyRecoveryFakes();
    ResetGdrPeermemFakes();
    ResetGdrProbeFakes();
    ResetTransportStubs();
  }

  FakeIbWorld& World() {
    if (!world_) world_ = std::make_unique<FakeIbWorld>();
    return *world_;
  }
  // For a second discovery in the same test: the first one's async threads must be gone first.
  void ResetForAnotherInit() {
    World().Quiesce();
    NetIbCastResetGlobals();
  }

  // A physical device as IbCastInitDevices leaves it, for the tests that start past discovery.
  static struct ncclIbDev& Dev(int d, const char* name, uint8_t link = IBV_LINK_LAYER_ETHERNET) {
    if (IbCastNDevs <= d) IbCastNDevs = d + 1;
    struct ncclIbDev& dev = IbCastDevs[d];
    snprintf(dev.devName, sizeof(dev.devName), "%s", name);
    dev.link = link;
    dev.speed = 100000;
    dev.portNum = 1;
    dev.railId = NCCL_NET_ID_UNDEF;
    dev.planeId = NCCL_NET_ID_UNDEF;
    return dev;
  }
  static void SetPciPath(int d, const std::string& path) {
    free(IbCastDevs[d].pciPath);
    IbCastDevs[d].pciPath = strdup(path.c_str());
  }
  // Root complex 0000:00 with no numa_node behind it. Merging devices compares their roots, which
  // are left uninitialized for a NULL pciPath.
  static void OnRoot0(int d) { SetPciPath(d, "/nonexistent/pci0000:00/0000:0" + std::to_string(d + 1) + ":00.0"); }

  static ncclNetVDeviceProps_t Props(std::initializer_list<int> devs) {
    ncclNetVDeviceProps_t props{};
    for (int d : devs) props.devs[props.ndevs++] = d;
    return props;
  }

  // realpath() answers for /sys/class/infiniband/<hca>/device.
  void PciPaths(std::map<std::string, std::string> paths) {
    realpath_.emplace(g_initRealpath, [paths](const char* path, char*) -> char* {
      for (auto& [hca, pci] : paths) {
        if (std::string(path) == "/sys/class/infiniband/" + hca + "/device") return strdup(pci.c_str());
      }
      return nullptr;
    });
  }

  ParamScript params_;
  std::unique_ptr<FakeIbWorld> world_;
  std::optional<HookOf<decltype(g_initRealpath)>> realpath_;
};


TEST_F(NetIbCastInitMicrotest, Speed_EachRateBitMapsToItsLaneSpeed) {
  const int expected[] = {2500, 5000, 10000, 10000, 14000, 25000, 50000, 100000, 200000};
  for (int bit = 0; bit < (int)(sizeof(expected) / sizeof(expected[0])); bit++) {
    EXPECT_EQ(ncclIbCastTestSpeed(1 << bit), expected[bit]) << bit;
  }
  // The lowest set bit wins; no bit at all falls through to the last entry.
  EXPECT_EQ(ncclIbCastTestSpeed((1 << 1) | (1 << 5)), 5000);
  EXPECT_EQ(ncclIbCastTestSpeed(0), 200000);
}

TEST_F(NetIbCastInitMicrotest, Width_EachWidthBitMapsToItsLaneCount) {
  EXPECT_EQ(IbCastWidth(1), 1);
  EXPECT_EQ(IbCastWidth(2), 4);
  EXPECT_EQ(IbCastWidth(4), 8);
  EXPECT_EQ(IbCastWidth(8), 12);
  EXPECT_EQ(IbCastWidth(16), 2);
  EXPECT_EQ(IbCastWidth(0), 2);
}

TEST_F(NetIbCastInitMicrotest, MatchVfPath_MergeVfsIgnoresTheFunctionAndPortDigits) {
  char a[] = "/sys/devices/pci0000:00/0000:01:00.0";
  char b[] = "/sys/devices/pci0000:00/0000:01:10.1";
  char c[] = "/sys/devices/pci0000:00/0000:01:00.1";
  char d[] = "/sys/devices/pci0000:00/0000:02:00.0";
  EXPECT_TRUE(IbCastMatchVfPath(a, b));
  EXPECT_FALSE(IbCastMatchVfPath(a, d));
  params_.Set("IB_MERGE_VFS", 0);
  EXPECT_FALSE(IbCastMatchVfPath(a, b));
  EXPECT_TRUE(IbCastMatchVfPath(a, c));
}

TEST_F(NetIbCastInitMicrotest, CompareDevs_OrdersByPciAddressWithUnknownPathsLast) {
  struct ncclIbDev& lo = Dev(0, "lo");
  struct ncclIbDev& hi = Dev(1, "hi");
  struct ncclIbDev& none = Dev(2, "none");
  snprintf(lo.fullPciPath, PATH_MAX, "/sys/devices/pci0000:00/0000:01:00.0");
  snprintf(hi.fullPciPath, PATH_MAX, "/sys/devices/pci0000:80/0000:81:00.0");
  none.fullPciPath[0] = '\0';
  EXPECT_LT(IbCastCompareDevs(&lo, &hi), 0);
  EXPECT_GT(IbCastCompareDevs(&hi, &lo), 0);
  EXPECT_EQ(IbCastCompareDevs(&lo, &lo), 0);
  EXPECT_GT(IbCastCompareDevs(&none, &lo), 0);
  EXPECT_LT(IbCastCompareDevs(&lo, &none), 0);
}

TEST_F(NetIbCastInitMicrotest, GetPciPath_FoldsPortAndVfDigitsAndKeepsTheFullPath) {
  PciPaths({{"mlx5_0", "/sys/devices/pci0000:00/0000:01:02.3"}});
  char hca[] = "mlx5_0";
  char full[PATH_MAX];
  char* path = nullptr;
  ASSERT_EQ(IbCastGetPciPath(hca, &path, full), ncclSuccess);
  EXPECT_STREQ(full, "/sys/devices/pci0000:00/0000:01:02.3");
  EXPECT_STREQ(path, "/sys/devices/pci0000:00/0000:01:00.0");
  free(path);
  params_.Set("IB_MERGE_VFS", 0);
  ASSERT_EQ(IbCastGetPciPath(hca, &path, full), ncclSuccess);
  EXPECT_STREQ(path, "/sys/devices/pci0000:00/0000:01:02.0");
  free(path);
  ASSERT_EQ(IbCastGetPciPath(hca, nullptr, full), ncclSuccess);  // path not wanted: freed here
}

TEST_F(NetIbCastInitMicrotest, GetPciPath_NoSysfsEntry_LeavesAnEmptyFullPath) {
  char hca[] = "mlx5_9";
  char full[PATH_MAX] = "stale";
  char* path = reinterpret_cast<char*>(0x1);
  ASSERT_EQ(IbCastGetPciPath(hca, &path, full), ncclSuccess);
  EXPECT_EQ(path, nullptr);
  EXPECT_STREQ(full, "");
}

TEST_F(NetIbCastInitMicrotest, GetRealPort_CountsEarlierDevicesOnTheSamePhysicalNic) {
  char p0[] = "/sys/devices/pci0000:00/0000:01:00.0";
  Dev(0, "a").pciPath = strdup(p0);
  Dev(1, "b").pciPath = strdup("/sys/devices/pci0000:00/0000:02:00.0");
  Dev(2, "c").pciPath = strdup(p0);
  int realPort = -1;
  ASSERT_EQ(IbCastGetRealPort(p0, &realPort, 3), ncclSuccess);
  EXPECT_EQ(realPort, 2);
  ASSERT_EQ(IbCastGetRealPort(p0, &realPort, 0), ncclSuccess);
  EXPECT_EQ(realPort, 0);
  ASSERT_EQ(IbCastGetRealPort(nullptr, &realPort, 3), ncclSuccess);
  EXPECT_EQ(realPort, 0);
}

TEST_F(NetIbCastInitMicrotest, PciRootFromPath_TakesTheFirstDomainAndBus) {
  char root[8];
  ASSERT_EQ(IbCastGetPciRootFromPath("/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0", root, sizeof(root)),
            ncclSuccess);
  EXPECT_STREQ(root, "0000:80");
  // "pcie" and a short "pci1:2" are skipped on the way to the real root.
  ASSERT_EQ(IbCastGetPciRootFromPath("/sys/pcie/pci1:2/pci000a:1f/x", root, sizeof(root)), ncclSuccess);
  EXPECT_STREQ(root, "000a:1f");
  EXPECT_EQ(IbCastGetPciRootFromPath("/sys/devices/platform/x", root, sizeof(root)), ncclInvalidUsage);
  EXPECT_EQ(IbCastGetPciRootFromPath(nullptr, root, sizeof(root)), ncclInvalidUsage);
  EXPECT_EQ(IbCastGetPciRootFromPath("/pci0000:80", nullptr, sizeof(root)), ncclInvalidUsage);
  EXPECT_EQ(IbCastGetPciRootFromPath("/pci0000:80", root, 7), ncclInvalidUsage);
}

class NetIbCastInitNumaMicrotest : public NetIbCastInitMicrotest {
 protected:
  void SetUp() override {
    char tmpl[] = "/tmp/net-ib-cast-numa-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    dir_ = tmpl;
  }
  void TearDown() override {
    std::filesystem::remove_all(dir_);
    NetIbCastInitMicrotest::TearDown();
  }
  int NumaOf(const char* contents) {
    std::ofstream(dir_ + "/numa_node") << contents;
    return IbCastGetNumaNodeFromPath(dir_.c_str());
  }
  std::string dir_;
};

TEST_F(NetIbCastInitNumaMicrotest, NumaNode_ReadsTheSysfsAttribute) {
  EXPECT_EQ(NumaOf("1\n"), 1);
  EXPECT_EQ(NumaOf("-1\n"), -1);
}

TEST_F(NetIbCastInitNumaMicrotest, NumaNode_UnreadableOrGarbage_IsUnknown) {
  EXPECT_EQ(IbCastGetNumaNodeFromPath(nullptr), -1);
  EXPECT_EQ(IbCastGetNumaNodeFromPath(dir_.c_str()), -1);  // no numa_node file
  EXPECT_EQ(NumaOf(""), -1);
  EXPECT_EQ(NumaOf("node"), -1);
  EXPECT_EQ(NumaOf("-99999999999999999999"), -1);  // ERANGE
  std::string longPath(PATH_MAX, 'x');
  EXPECT_EQ(IbCastGetNumaNodeFromPath(longPath.c_str()), -1);
}

TEST_F(NetIbCastInitMicrotest, PlaneIndex_AssignsIndexesInFirstSeenOrder) {
  int16_t count = 1, planes[NCCL_IB_PLANE_MAX_INDEX] = {NCCL_NET_ID_UNDEF}, idx = -1;
  ASSERT_EQ(ncclIbCastTestGetPlaneIndex(NCCL_NET_ID_UNDEF, &count, planes, &idx), ncclSuccess);
  EXPECT_EQ(idx, 0);
  ASSERT_EQ(ncclIbCastTestGetPlaneIndex(5, &count, planes, &idx), ncclSuccess);
  EXPECT_EQ(idx, 1);
  ASSERT_EQ(ncclIbCastTestGetPlaneIndex(9, &count, planes, &idx), ncclSuccess);
  EXPECT_EQ(idx, 2);
  ASSERT_EQ(ncclIbCastTestGetPlaneIndex(5, &count, planes, &idx), ncclSuccess);
  EXPECT_EQ(idx, 1);
  EXPECT_EQ(count, 3);
  EXPECT_EQ(ncclIbCastTestGetPlaneIndex(5, nullptr, planes, &idx), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastTestGetPlaneIndex(5, &count, nullptr, &idx), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastTestGetPlaneIndex(5, &count, planes, nullptr), ncclInvalidArgument);
}

TEST_F(NetIbCastInitMicrotest, PlaneIndex_TableFullOrVirtualBitSet_IsRejected) {
  int16_t count = 1, planes[NCCL_IB_PLANE_MAX_INDEX] = {NCCL_NET_ID_UNDEF}, idx = -1;
  EXPECT_EQ(ncclIbCastTestGetPlaneIndex(NCCL_IB_PLANE_VIRT_BIT | 1, &count, planes, &idx), ncclInvalidUsage);
  for (int p = 1; count < NCCL_IB_PLANE_MAX_INDEX - 1; p++) {
    ASSERT_EQ(ncclIbCastTestGetPlaneIndex(p, &count, planes, &idx), ncclSuccess);
  }
  EXPECT_EQ(ncclIbCastTestGetPlaneIndex(100, &count, planes, &idx), ncclInvalidUsage);
  ASSERT_EQ(ncclIbCastTestGetPlaneIndex(1, &count, planes, &idx), ncclSuccess);  // known ones still resolve
}

TEST_F(NetIbCastInitMicrotest, RelaxedOrdering_NeedsTheModeAndTheIova2Api) {
  ncclResult_t iova2 = ncclSystemError;  // present: the NULL probe fails on its arguments
  ScopedHook probe(g_ibvRegMrIova2, [&](struct ibv_mr** ret, struct ibv_pd* pd, void*, size_t, uint64_t, int) {
    EXPECT_EQ(ret, nullptr);
    EXPECT_EQ(pd, nullptr);
    return iova2;
  });
  EXPECT_EQ(IbCastRelaxedOrderingCapable(), 1);  // default mode 2
  params_.Set("IB_PCI_RELAXED_ORDERING", 1);
  EXPECT_EQ(IbCastRelaxedOrderingCapable(), 1);
  iova2 = ncclInternalError;  // IBVERBS_1.8 missing
  EXPECT_EQ(IbCastRelaxedOrderingCapable(), 0);
  params_.Set("IB_PCI_RELAXED_ORDERING", 0);
  const int calls = probe.calls;
  EXPECT_EQ(IbCastRelaxedOrderingCapable(), 0);
  EXPECT_EQ(probe.calls, calls);
}

struct DmaBufProbe {
  const char* name;
  bool allocFails;
  int ibvErrno;
  int mlx5Errno;
  bool deallocFails;
  bool capable;
};

class NetIbCastInitDmaBufMicrotest : public NetIbCastInitMicrotest,
                                     public ::testing::WithParamInterface<DmaBufProbe> {};

TEST_P(NetIbCastInitDmaBufMicrotest, Mlx5DmaBufCapable_NeedsBothDummyRegistrationsToFailOnlyOnTheFd) {
  const DmaBufProbe c = GetParam();
  struct ibv_context ctx {};
  struct ibv_pd pd {};
  ScopedHook alloc(g_ibvAllocPd, [&](struct ibv_pd** ret, struct ibv_context*) {
    *ret = c.allocFails ? nullptr : &pd;
    return c.allocFails ? ncclSystemError : ncclSuccess;
  });
  ScopedHook ibvReg(g_ibvDirectRegDmabufMr, [&](struct ibv_pd*, uint64_t, size_t, uint64_t, int fd, int) {
    EXPECT_EQ(fd, -1);
    errno = c.ibvErrno;
    return (struct ibv_mr*)nullptr;
  });
  ScopedHook mlx5Reg(g_mlx5dvDirectRegDmabufMr, [&](struct ibv_pd*, uint64_t, size_t, uint64_t, int fd, int, int) {
    EXPECT_EQ(fd, -1);
    errno = c.mlx5Errno;
    return (struct ibv_mr*)nullptr;
  });
  ScopedHook dealloc(g_ibvDeallocPd, [&](struct ibv_pd*) { return c.deallocFails ? ncclSystemError : ncclSuccess; });
  EXPECT_EQ(ncclMlx5dvDmaBufCapable(&ctx), c.capable);
}

INSTANTIATE_TEST_SUITE_P(
    NetIbCastInit, NetIbCastInitDmaBufMicrotest,
    ::testing::Values(DmaBufProbe{"Capable", false, EBADF, EBADF, false, true},
                      DmaBufProbe{"PdAllocFails", true, EBADF, EBADF, false, false},
                      DmaBufProbe{"KernelLacksDmaBuf", false, EOPNOTSUPP, EBADF, false, false},
                      DmaBufProbe{"KernelProtocolUnsupported", false, EPROTONOSUPPORT, EBADF, false, false},
                      DmaBufProbe{"Mlx5LacksDmaBuf", false, EBADF, EPROTONOSUPPORT, false, false},
                      DmaBufProbe{"Mlx5OpUnsupported", false, EBADF, EOPNOTSUPP, false, false},
                      DmaBufProbe{"PdDeallocFails", false, EBADF, EBADF, true, false}),
    [](const ::testing::TestParamInfo<DmaBufProbe>& info) { return info.param.name; });

TEST_F(NetIbCastInitMicrotest, OooRqSize_ReportsTheDevicesOooReceiveDepthOnlyWhenEnabled) {
  struct ibv_context ctx {};
  uint32_t size = 99;
  int maxRc = 64;
  uint64_t mask = MLX5DV_CONTEXT_MASK_OOO_RECV_WRS;
  ScopedHook query(g_mlx5dvQueryDevice, [&](struct ibv_context*, struct mlx5dv_context* dv) {
    EXPECT_EQ(dv->comp_mask, (uint64_t)MLX5DV_CONTEXT_MASK_OOO_RECV_WRS);
    dv->comp_mask = mask;
    dv->ooo_recv_wrs_caps.max_rc = maxRc;
    return ncclSuccess;
  });
  ASSERT_EQ(IbCastQueryOooRqSize(&ctx, "mlx5_0", &size), ncclSuccess);
  EXPECT_EQ(size, 0u);  // NCCL_IB_OOO_RQ off: not asked
  EXPECT_EQ(query.calls, 0);
  params_.Set("IB_OOO_RQ", 1);
  ASSERT_EQ(IbCastQueryOooRqSize(&ctx, "mlx5_0", &size), ncclSuccess);
  EXPECT_EQ(size, 64u);
  maxRc = 0;
  ASSERT_EQ(IbCastQueryOooRqSize(&ctx, "mlx5_0", &size), ncclSuccess);
  EXPECT_EQ(size, 0u);
  maxRc = 64;
  mask = 0;  // the provider does not know the capability
  ASSERT_EQ(IbCastQueryOooRqSize(&ctx, "mlx5_0", &size), ncclSuccess);
  EXPECT_EQ(size, 0u);
  EXPECT_EQ(IbCastQueryOooRqSize(&ctx, "mlx5_0", nullptr), ncclInvalidArgument);
}

TEST_F(NetIbCastInitMicrotest, OooRqSize_QueryFails_IsAnInternalError) {
  struct ibv_context ctx {};
  uint32_t size = 0;
  params_.Set("IB_OOO_RQ", 1);
  ScopedHook query(g_mlx5dvQueryDevice, [](struct ibv_context*, struct mlx5dv_context*) { return ncclSystemError; });
  EXPECT_EQ(IbCastQueryOooRqSize(&ctx, "mlx5_0", &size), ncclInternalError);
}


class NetIbCastInitVDeviceMicrotest : public NetIbCastInitMicrotest {
 protected:
  void SetUp() override {
    IbCastNMergedDevs = 0;
    Dev(0, "mlx5_0");
    Dev(1, "mlx5_1");
    Dev(2, "mlx5_2");
    for (int d = 0; d < 3; d++) OnRoot0(d);
  }
  ncclResult_t Make(std::initializer_list<int> devs, int* d) {
    ncclNetVDeviceProps_t props = Props(devs);
    return IbCastMakeVDevice(d, &props);
  }
};

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_OneDevice_CopiesItsNameSpeedRailAndPlane) {
  IbCastDevs[1].railId = 4;
  IbCastDevs[1].planeId = 2;
  int d = -1;
  ASSERT_EQ(Make({1}, &d), ncclSuccess);
  EXPECT_EQ(d, 0);
  EXPECT_EQ(IbCastNMergedDevs, 1);
  const ncclIbMergedDev& m = IbCastMergedDevs[0];
  EXPECT_STREQ(m.devName, "mlx5_1");
  EXPECT_EQ(m.speed, 100000);
  EXPECT_EQ(m.railId, 4);
  EXPECT_EQ(m.planeId, 2);
  EXPECT_EQ(m.vProps.ndevs, 1);
  EXPECT_EQ(m.vProps.devs[0], 1);
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_SeveralDevices_FusesNamesSpeedsAndPlanes) {
  IbCastDevs[0].railId = IbCastDevs[2].railId = 3;
  IbCastDevs[0].planeIdx = 1;
  IbCastDevs[2].planeIdx = 2;
  IbCastOffloadEnabled = true;
  int d = -1;
  ASSERT_EQ(Make({0, 2, 0}, &d), ncclSuccess);
  const ncclIbMergedDev& m = IbCastMergedDevs[d];
  EXPECT_STREQ(m.devName, "mlx5_0+mlx5_2");  // the repeated device is merged once
  EXPECT_EQ(m.vProps.ndevs, 2);
  EXPECT_EQ(m.speed, 200000);
  EXPECT_EQ(m.railId, 3);
  EXPECT_EQ(m.planeId, NCCL_IB_PLANE_VIRT_BIT | (1 << 1) | (1 << 2));
  EXPECT_FALSE(IbCastOffloadEnabled);  // CTS offload does not support fused NICs
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_DifferentRails_LeavesTheRailUndefined) {
  IbCastDevs[0].railId = 1;
  IbCastDevs[1].railId = 2;
  int d = -1;
  ASSERT_EQ(Make({0, 1}, &d), ncclSuccess);
  EXPECT_EQ(IbCastMergedDevs[d].railId, NCCL_NET_ID_UNDEF);
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_MergeNicsOff_RefusesToFuse) {
  params_.Set("IB_MERGE_NICS", 0);
  int d = -1;
  EXPECT_EQ(Make({0, 1}, &d), ncclInvalidUsage);
  EXPECT_EQ(IbCastNMergedDevs, 0);
  ASSERT_EQ(Make({0}, &d), ncclSuccess);  // a single device is not a fusion
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_AinicWithMergeNicsNotOne_RefusesToFuse) {
  g_rcclUseAinicValue = true;
  params_.Set("IB_MERGE_NICS", 2);
  int d = -1;
  EXPECT_EQ(Make({0, 1}, &d), ncclInvalidUsage);
}

// AINIC is meant to fuse NICs only on an explicit NCCL_IB_MERGE_NICS=1, but 1 is the default.
// Pins AICOMRCCL-3088; a fix flips this pin.
TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_KnownBug_AinicFusesNicsByDefault) {
  g_rcclUseAinicValue = true;
  int d = -1;
  EXPECT_EQ(Make({0, 1}, &d), ncclSuccess);
  EXPECT_EQ(IbCastMergedDevs[d].vProps.ndevs, 2);
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_BadDeviceCounts_AreRejected) {
  int d = -1;
  ncclNetVDeviceProps_t props{};
  EXPECT_EQ(IbCastMakeVDevice(&d, &props), ncclInvalidUsage);
  props.ndevs = -1;
  EXPECT_EQ(IbCastMakeVDevice(&d, &props), ncclInvalidUsage);
  props.ndevs = NCCL_IB_MAX_DEVS_PER_NIC + 1;
  EXPECT_EQ(IbCastMakeVDevice(&d, &props), ncclInvalidUsage);
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_UnknownPhysicalDevice_IsRejected) {
  int d = -1;
  EXPECT_EQ(Make({0, 3}, &d), ncclInvalidUsage);
  // Not first: makeVDevice reads IbCastDevs[devs[0]] before it checks the range. The later checks
  // return the same code, so the warning is what shows this one fired.
  std::string log = CaptureLog([&] { EXPECT_EQ(Make({0, -1}, &d), ncclInvalidUsage); });
  EXPECT_TRUE(LogHas(log, "physical device -1")) << log;
  EXPECT_EQ(IbCastNMergedDevs, 0);
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_TableFull_IsRejected) {
  IbCastNMergedDevs = MAX_IB_VDEVS;
  int d = -1;
  EXPECT_EQ(Make({0}, &d), ncclInvalidUsage);
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_MixedLinkLayers_IsRejected) {
  IbCastDevs[1].link = IBV_LINK_LAYER_INFINIBAND;
  int d = -1;
  EXPECT_EQ(Make({0, 1}, &d), ncclInvalidUsage);
  EXPECT_EQ(IbCastNMergedDevs, 0);
}

TEST_F(NetIbCastInitVDeviceMicrotest, MakeVDevice_AcrossNumaNodesOrRootComplexes_WarnsButFuses) {
  char tmpl[] = "/tmp/net-ib-cast-vdev-XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::string base = tmpl;
  const std::string p0 = base + "/pci0000:00/a", p1 = base + "/pci0000:00/b", p2 = base + "/pci0000:80/c";
  for (const std::string& p : {p0, p1, p2}) std::filesystem::create_directories(p);
  std::ofstream(p0 + "/numa_node") << "0\n";
  std::ofstream(p1 + "/numa_node") << "1\n";
  std::ofstream(p2 + "/numa_node") << "0\n";
  SetPciPath(0, p0);
  SetPciPath(1, p1);
  SetPciPath(2, p2);
  int d = -1;
  std::string log = CaptureLog([&] { ASSERT_EQ(Make({0, 1}, &d), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "across NUMA nodes")) << log;
  log = CaptureLog([&] { ASSERT_EQ(Make({0, 2}, &d), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "PCIe Root Complexes")) << log;
  EXPECT_FALSE(LogHas(log, "NUMA")) << log;
  std::filesystem::remove_all(base);
}

TEST_F(NetIbCastInitVDeviceMicrotest, SetNetAttr_AcceptsAnything) {
  EXPECT_EQ(IbCastSetNetAttr(nullptr, nullptr), ncclSuccess);
}


class NetIbCastInitDevicesMicrotest : public NetIbCastInitMicrotest {
 protected:
  ncclResult_t Init() { return IbCastInitDevices(nullptr, nullptr); }
};

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_OneRoceHca_AddsOnePhysicalAndOneVirtualDevice) {
  World().AddHca("ionic_0");
  PciPaths({{"ionic_0", "/sys/devices/pci0000:00/0000:01:00.0"}});
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 1);
  ASSERT_EQ(IbCastNMergedDevs, 1);
  const struct ncclIbDev& dev = IbCastDevs[0];
  EXPECT_STREQ(dev.devName, "ionic_0");
  EXPECT_EQ(dev.portNum, 1);
  EXPECT_EQ(dev.link, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(dev.speed, 25000 * 4);
  EXPECT_FALSE(dev.speedFromQuery);
  EXPECT_EQ(dev.guid, World().hcas[0].attr.sys_image_guid);
  EXPECT_EQ(dev.maxQp, 1024);
  EXPECT_EQ(dev.maxCqe, 4096);
  EXPECT_EQ(dev.ar, 0);  // adaptive routing is an IB default only
  EXPECT_EQ(dev.udSupported, -1);
  EXPECT_EQ(dev.rdmaReadSupported, -1);
  EXPECT_EQ(dev.gidInfo.localGidIndex, 3);
  EXPECT_EQ(dev.context, &World().hcas[0].context);
  EXPECT_STREQ(dev.pciPath, "/sys/devices/pci0000:00/0000:01:00.0");
  EXPECT_EQ(dev.railId, -1);
  EXPECT_EQ(dev.planeId, -1);
  EXPECT_STREQ(IbCastMergedDevs[0].devName, "ionic_0");
  EXPECT_STREQ(IbCastIfName, "eth0");
  EXPECT_EQ(IbCastRelaxedOrderingEnabled, 0);
  EXPECT_EQ(World().threadNames, std::vector<std::string>{"NCCL IbAsync  0"});
  EXPECT_EQ(g_ibCastReportMatchingSchemeCalls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_SecondCall_OnlyTakesAReference) {
  World().AddHca("ionic_0");
  ASSERT_EQ(Init(), ncclSuccess);
  const int lists = World().getDeviceList.calls;
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(World().getDeviceList.calls, lists);
  EXPECT_EQ(IbCastNDevs, 1);
  // Both references are released before the pool check runs.
  params_.Set("RCCL_IB_QP_SHARING_VALIDATE_POOL", 1);
  ASSERT_EQ(IbCastFinalizeDevices(), ncclSuccess);
  EXPECT_EQ(g_ibCastValidateSharedQpPoolCalls, 0);
  ASSERT_EQ(IbCastFinalizeDevices(), ncclSuccess);
  EXPECT_EQ(g_ibCastValidateSharedQpPoolCalls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, FinalizeDevices_LastReferenceWithoutValidation_SkipsThePoolCheck) {
  World().AddHca("ionic_0");
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastFinalizeDevices(), ncclSuccess);
  EXPECT_EQ(g_ibCastValidateSharedQpPoolCalls, 0);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_IbDisabled_FailsBeforeLoadingVerbs) {
  FakeIbWorld& world = World();
  world.AddHca("ionic_0");
  params_.Set("IB_DISABLE", 1);
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(world.ibvSymbols.calls, 0);
}

// The reference is taken before the checks that can fail and is never dropped, so a retry
// returns at once as if the devices were up. Pins AICOMRCCL-3091; a fix flips this pin.
TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_KnownBug_FailedInitMakesTheNextOneASilentNoOp) {
  World().AddHca("ionic_0");
  params_.Set("IB_DISABLE", 1);
  ASSERT_EQ(Init(), ncclInternalError);
  params_.Set("IB_DISABLE", 0);
  EXPECT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(World().getDeviceList.calls, 0);
  EXPECT_EQ(IbCastNDevs, -1);
  int ndev = 0;
  ASSERT_EQ(IbCastDevices(&ndev), ncclSuccess);
  EXPECT_EQ(ndev, -1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_NoVerbsLibrary_Fails) {
  FakeIbWorld& world = World();
  world.AddHca("ionic_0");
  ScopedHook symbols(g_ibvSymbols, [] { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(world.ionicSymbols.calls, 0);
  EXPECT_EQ(IbCastNDevs, -1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_NoMlx5Library_CarriesOn) {
  World().AddHca("ionic_0");
  ScopedHook symbols(g_mlx5dvSymbols, [] { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastNDevs, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_NoIonicLibrary_DisablesTheTransport) {
  World().AddHca("ionic_0");
  ScopedHook symbols(g_ionicdvSymbols, [] { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(World().getDeviceList.calls, 0);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_NoIpInterface_Fails) {
  World().AddHca("ionic_0");
  ScopedHook none(g_ncclFindInterfaces, [](char*, union ncclSocketAddress*, int, int, int* n) {
    *n = 0;
    return ncclSuccess;
  });
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(World().freeDeviceList.calls, 0);  // never listed
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_InterfaceLookupErrors_ArePassedUp) {
  World();
  ScopedHook fails(g_ncclFindInterfaces, [](char*, union ncclSocketAddress*, int, int, int*) { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclSystemError);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_DeviceListFails_Fails) {
  World();
  ScopedHook list(g_ibvGetDeviceList, [](struct ibv_device***, int*) { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(World().freeDeviceList.calls, 0);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_NoHca_FindsNothingButSucceeds) {
  World();
  EXPECT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastNDevs, 0);
  EXPECT_EQ(IbCastNMergedDevs, 0);
  EXPECT_EQ(g_ibCastReportMatchingSchemeCalls, 0);
  EXPECT_EQ(World().freeDeviceList.calls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_UnopenableHca_IsSkipped) {
  World().AddHca("ionic_0").openFails = true;
  World().AddHca("ionic_1");
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 1);
  EXPECT_STREQ(IbCastDevs[0].devName, "ionic_1");
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_OpenSucceedsWithoutAContext_IsSkipped) {
  World().AddHca("ionic_0");
  ScopedHook open(g_ibvOpenDevice, [](struct ibv_context** ret, struct ibv_device*) {
    *ret = nullptr;
    return ncclSuccess;
  });
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(World().queryDevice.calls, 0);
  EXPECT_EQ(IbCastNDevs, 0);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_UnqueryableHca_IsClosedAndSkipped) {
  FakeHca& bad = World().AddHca("ionic_0");
  bad.queryDeviceFails = true;
  World().AddHca("ionic_1");
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(bad.closes, 1);
  EXPECT_EQ(IbCastNDevs, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_CloseAfterFailedQueryFails_FreesTheListAndFails) {
  World().AddHca("ionic_0").queryDeviceFails = true;
  ScopedHook close(g_ibvCloseDevice, [](struct ibv_context*) { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(World().freeDeviceList.calls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_PortsThatAreDownUnqueryableOrNotIbOrRoce_AreSkipped) {
  FakeHca& hca = World().AddHca("ionic_0");
  hca.ports[0].state = IBV_PORT_DOWN;
  hca.ports.push_back(FakeIbWorld::ActivePort(IBV_LINK_LAYER_UNSPECIFIED));
  hca.ports.push_back(FakeIbWorld::ActivePort(IBV_LINK_LAYER_INFINIBAND));
  hca.attr.phys_port_cnt = 4;  // port 4 answers no query
  ScopedHook query(g_ibvQueryDevice, [&](struct ibv_context*, struct ibv_device_attr* attr) {
    *attr = hca.attr;
    return ncclSuccess;
  });
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 1);
  EXPECT_EQ(IbCastDevs[0].portNum, 3);
  EXPECT_EQ(IbCastDevs[0].link, IBV_LINK_LAYER_INFINIBAND);
  EXPECT_EQ(IbCastDevs[0].ar, 1);  // IB defaults to adaptive routing
  EXPECT_EQ(hca.closes, 0);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_HcaWithNoUsablePort_IsClosed) {
  FakeHca& hca = World().AddHca("ionic_0");
  hca.ports[0].state = IBV_PORT_DOWN;
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(hca.closes, 1);
  EXPECT_EQ(IbCastNDevs, 0);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_ClosingAnUnusedHcaFails_Fails) {
  World().AddHca("ionic_0").ports[0].state = IBV_PORT_DOWN;
  ScopedHook close(g_ibvCloseDevice, [](struct ibv_context*) { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(World().freeDeviceList.calls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_FreeingTheListFails_Fails) {
  World().AddHca("ionic_0");
  ScopedHook freeList(g_ibvFreeDeviceList, [](struct ibv_device**) { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(freeList.calls, 2);  // the failure path tries once more
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_PortSpeedQuery_WinsOverThePortAttributes) {
  World().AddHca("ionic_0").portSpeed = 4000;  // 400 Gb/s
  World().AddHca("ionic_1").ports[0].active_speed_ex = 0x100;  // XDR, beyond active_speed
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 2);
  EXPECT_EQ(IbCastDevs[0].speed, 400000);
  int fromQuery = -1;
  ASSERT_EQ(ncclIbCastTestSpeedSource(0, &fromQuery), ncclSuccess);
  EXPECT_EQ(fromQuery, 1);
  EXPECT_EQ(IbCastDevs[1].speed, 200000 * 4);
  ASSERT_EQ(ncclIbCastTestSpeedSource(1, &fromQuery), ncclSuccess);
  EXPECT_EQ(fromQuery, 0);
  EXPECT_EQ(ncclIbCastTestGetNDevs(), 2);
  EXPECT_EQ(ncclIbCastTestSpeedSource(2, &fromQuery), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastTestSpeedSource(-1, &fromQuery), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastTestSpeedSource(0, nullptr), ncclInvalidArgument);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_IbHcaList_SelectsByPrefixExclusionAndExactName) {
  for (const char* name : {"ionic_0", "ionic_1", "ionic_10"}) World().AddHca(name);
  SetMicroEnv("NCCL_IB_HCA", "ionic_1");
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastNDevs, 2);  // a prefix: ionic_1 and ionic_10
  ResetForAnotherInit();
  SetMicroEnv("NCCL_IB_HCA", "=ionic_1");
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 1);
  EXPECT_STREQ(IbCastDevs[0].devName, "ionic_1");
  ResetForAnotherInit();
  SetMicroEnv("NCCL_IB_HCA", "^ionic_1");
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 1);
  EXPECT_STREQ(IbCastDevs[0].devName, "ionic_0");
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_IbHcaRailAndPlane_AreRecordedAndLogged) {
  World().AddHca("ionic_0");
  World().AddHca("ionic_1");
  World().AddHca("ionic_2");
  SetMicroEnv("NCCL_IB_HCA", "ionic_0:1:7,ionic_1::2:5,ionic_2:::6");
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 3);
  EXPECT_EQ(IbCastDevs[0].railId, 7);
  EXPECT_EQ(IbCastDevs[1].railId, 2);
  EXPECT_EQ(IbCastDevs[1].planeId, 5);
  EXPECT_EQ(IbCastDevs[2].planeId, 6);
  EXPECT_NE(IbCastDevs[1].planeIdx, IbCastDevs[2].planeIdx);
}

// The list is freed after enumeration but the pointer is kept, so a plane-index failure frees it
// again on the fail: path. Pins AICOMRCCL-TBD; a fix flips this pin.
TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_KnownBug_TooManyPlanesFreesTheDeviceListTwice) {
  std::string hca;
  for (int i = 0; i < NCCL_IB_PLANE_MAX_INDEX; i++) {
    const std::string name = "ionic_" + std::string(1, (char)('a' + i));
    World().AddHca(name.c_str());
    hca += (i ? "," : "") + name + ":::" + std::to_string(i + 1);
  }
  SetMicroEnv("NCCL_IB_HCA", hca.c_str());
  std::vector<struct ibv_device**> freed;
  ScopedHook freeList(g_ibvFreeDeviceList, [&freed](struct ibv_device** list) {
    freed.push_back(list);
    return ncclSuccess;
  });
  EXPECT_EQ(Init(), ncclInvalidUsage);
  ASSERT_EQ(freed.size(), 2u);
  EXPECT_EQ(freed[0], freed[1]);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_SortsByPciPathUnlessDisabled) {
  World().AddHca("ionic_0");
  World().AddHca("ionic_1");
  PciPaths({{"ionic_0", "/sys/devices/pci0000:80/0000:81:00.0"}, {"ionic_1", "/sys/devices/pci0000:00/0000:01:00.0"}});
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_STREQ(IbCastDevs[0].devName, "ionic_1");
  EXPECT_STREQ(IbCastMergedDevs[0].devName, "ionic_1");
  ResetForAnotherInit();
  params_.Set("IB_DEVICE_PCI_ORDER", 0);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_STREQ(IbCastDevs[0].devName, "ionic_0");
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_TwoPortsOfOneNic_GetDistinctRealPorts) {
  FakeHca& hca = World().AddHca("ionic_0");
  hca.ports.push_back(FakeIbWorld::ActivePort(IBV_LINK_LAYER_ETHERNET));
  PciPaths({{"ionic_0", "/sys/devices/pci0000:00/0000:01:00.1"}});
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 2);
  EXPECT_EQ(IbCastDevs[0].realPort, 0);
  EXPECT_EQ(IbCastDevs[1].realPort, 1);
  EXPECT_EQ(IbCastNMergedDevs, 2);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_AdaptiveRoutingParam_OverridesTheLinkDefault) {
  World().AddHca("ionic_0");
  params_.Set("IB_ADAPTIVE_ROUTING", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastDevs[0].ar, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_GidQueryFails_Fails) {
  World().AddHca("ionic_0");
  ScopedHook gid(g_ibCastGidInfoQuery, [](struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*) {
    return ncclSystemError;
  });
  EXPECT_EQ(Init(), ncclSystemError);
  EXPECT_EQ(World().freeDeviceList.calls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_RelaxedOrderingAvailable_IsEnabled) {
  World();
  ScopedHook iova2(g_ibvRegMrIova2, [](struct ibv_mr**, struct ibv_pd*, void*, size_t, uint64_t, int) {
    return ncclSystemError;
  });
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastRelaxedOrderingEnabled, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_ArThreshold_IsTakenUnlessOooRqIsOn) {
  World();
  params_.Set("IB_AR_THRESHOLD", 4096);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastArThreshold, 4096);
  ResetForAnotherInit();
  params_.Set("IB_OOO_RQ", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastArThreshold, 8192);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_Mlx5WithOooRq_RecordsTheOooDepth) {
  World().AddHca("mlx5_0").mlx5 = true;
  params_.Set("IB_OOO_RQ", 1);
  params_.Set("IB_DATA_DIRECT", 0);
  ScopedHook query(g_mlx5dvQueryDevice, [](struct ibv_context*, struct mlx5dv_context* dv) {
    dv->ooo_recv_wrs_caps.max_rc = 32;
    return ncclSuccess;
  });
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(IbCastDevs[0].ibProvider, IB_PROVIDER_MLX5);
  EXPECT_EQ(IbCastDevs[0].oooRqSize, 32u);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_Mlx5OooRqQueryFails_Fails) {
  World().AddHca("mlx5_0").mlx5 = true;
  params_.Set("IB_OOO_RQ", 1);
  ScopedHook query(g_mlx5dvQueryDevice, [](struct ibv_context*, struct mlx5dv_context*) { return ncclSystemError; });
  EXPECT_EQ(Init(), ncclInternalError);
}

class NetIbCastInitDataDirectMicrotest : public NetIbCastInitDevicesMicrotest {
 protected:
  void SetUp() override {
    World().AddHca("mlx5_0").mlx5 = true;
    PciPaths({{"mlx5_0", "/sys/devices/pci0000:00/0000:01:00.0"}});
  }
  // The DMA-BUF probe passes; the sysfs path lookup answers `result`.
  void DataDirect(ncclResult_t result) {
    allocPd_.emplace(g_ibvAllocPd, [this](struct ibv_pd** ret, struct ibv_context*) {
      *ret = &pd_;
      return ncclSuccess;
    });
    regDmabuf_.emplace(g_ibvDirectRegDmabufMr, [](struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
      errno = EBADF;
      return (struct ibv_mr*)nullptr;
    });
    mlx5RegDmabuf_.emplace(g_mlx5dvDirectRegDmabufMr, [](struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int) {
      errno = EBADF;
      return (struct ibv_mr*)nullptr;
    });
    sysfs_.emplace(g_mlx5dvGetDataDirectSysfsPath, [result](struct ibv_context*, char* buf, size_t len) {
      if (result == ncclSuccess) snprintf(buf, len, "/devices/dd0");
      return result;
    });
  }
  struct ibv_pd pd_ {};
  std::optional<HookOf<decltype(g_ibvAllocPd)>> allocPd_;
  std::optional<HookOf<decltype(g_ibvDirectRegDmabufMr)>> regDmabuf_;
  std::optional<HookOf<decltype(g_mlx5dvDirectRegDmabufMr)>> mlx5RegDmabuf_;
  std::optional<HookOf<decltype(g_mlx5dvGetDataDirectSysfsPath)>> sysfs_;
};

TEST_F(NetIbCastInitDataDirectMicrotest, InitDevices_DataDirectNic_ExposesOnlyTheDmaDevice) {
  DataDirect(ncclSuccess);
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 1);
  EXPECT_STREQ(IbCastDevs[0].devName, "mlx5_0_dma");
  EXPECT_STREQ(IbCastDevs[0].pciPath, "/sys/devices/dd0");
  EXPECT_EQ(IbCastDevs[0].capsProvider.mlx5.dataDirect, 1);
  EXPECT_STREQ(IbCastDevs[0].fullPciPath, "/sys/devices/pci0000:00/0000:01:00.0");
}

TEST_F(NetIbCastInitDataDirectMicrotest, InitDevices_DataDirectTwo_ExposesBothDevices) {
  DataDirect(ncclSuccess);
  params_.Set("IB_DATA_DIRECT", 2);
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 2);
  EXPECT_EQ(IbCastDevs[0].capsProvider.mlx5.dataDirect + IbCastDevs[1].capsProvider.mlx5.dataDirect, 1);
}

TEST_F(NetIbCastInitDataDirectMicrotest, InitDevices_NoDataDirectPath_KeepsThePlainDevice) {
  DataDirect(ncclInvalidArgument);
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(IbCastNDevs, 1);
  EXPECT_STREQ(IbCastDevs[0].devName, "mlx5_0");
  EXPECT_EQ(IbCastDevs[0].capsProvider.mlx5.dataDirect, 0);
}

// The error returns past the fail: label, so the device list and the open context are never released.
// Pins AICOMRCCL-3090; a fix flips this pin.
TEST_F(NetIbCastInitDataDirectMicrotest, InitDevices_KnownBug_DataDirectLookupErrorLeaksTheDeviceList) {
  DataDirect(ncclSystemError);
  EXPECT_EQ(Init(), ncclSystemError);
  EXPECT_EQ(World().freeDeviceList.calls, 0);
  EXPECT_EQ(World().hcas[0].closes, 0);
}

TEST_F(NetIbCastInitDataDirectMicrotest, InitDevices_DataDirectOff_SkipsTheProbe) {
  DataDirect(ncclSuccess);
  params_.Set("IB_DATA_DIRECT", 0);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(sysfs_->calls, 0);
  EXPECT_STREQ(IbCastDevs[0].devName, "mlx5_0");
}


class NetIbCastInitAinicMicrotest : public NetIbCastInitDevicesMicrotest {
 protected:
  void SetUp() override {
    g_rcclUseAinicValue = true;
    World().AddHca("ionic_0");
  }
};

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_Ainic_EnablesCtsOffloadAndInlineByDefault) {
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_TRUE(IbCastAinicRoce);
  EXPECT_TRUE(IbCastOffloadEnabled);
  EXPECT_TRUE(IbCastUseInline);
  EXPECT_TRUE(IbCastAinicCtsInlineData);
  EXPECT_EQ(IbCastGdrFlushDisable, 1);
  EXPECT_FALSE(IbCastMultiplaneEnable);
}

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_Ainic_StoresEachDevicesNormalizedUdmaCount) {
  World().AddHca("ionic_1");
  World().AddHca("ionic_2");
  // 0 is a driver without the query, which keeps the two legacy engines; an 8-bit mask caps it at 8.
  const std::map<std::string, uint8_t> raw = {{"ionic_0", 4}, {"ionic_1", 0}, {"ionic_2", 9}};
  ScopedHook count(g_ionicdvCtxGetUdmaCount, [&](ibv_context* ctx) -> uint8_t {
    FakeHca* hca = World().Find(ctx);
    return hca ? raw.at(hca->device.name) : 0;
  });
  ASSERT_EQ(Init(), ncclSuccess);
  std::map<std::string, int> got;
  for (int d = 0; d < IbCastNDevs; d++) got[IbCastDevs[d].devName] = IbCastDevs[d].capsProvider.ionic.udmaCount;
  EXPECT_EQ(got, (std::map<std::string, int>{{"ionic_0", 4}, {"ionic_1", 2}, {"ionic_2", 8}}));
}

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_AinicOffloadOff_KeepsInlineAsConfigured) {
  params_.Set("RCCL_CTS_OFFLOAD_ENABLED", 0);
  params_.Set("GDR_FLUSH_DISABLE", 0);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(IbCastOffloadEnabled);
  EXPECT_FALSE(IbCastUseInline);
  EXPECT_FALSE(IbCastAinicCtsInlineData);
  EXPECT_EQ(IbCastGdrFlushDisable, 0);
}

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_AinicInlineWithoutOffload_KeepsCtsInlineData) {
  params_.Set("RCCL_CTS_OFFLOAD_ENABLED", 0);
  params_.Set("IB_USE_INLINE", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(IbCastOffloadEnabled);
  EXPECT_TRUE(IbCastAinicCtsInlineData);
}

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_AinicWithQpScheduler_DropsOffloadAndInline) {
  ScopedHook sched(g_rcclUseIbCastQpSched, [] { return true; });
  params_.Set("IB_USE_INLINE", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(IbCastOffloadEnabled);
  EXPECT_FALSE(IbCastUseInline);
  EXPECT_FALSE(IbCastAinicCtsInlineData);
}

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_AinicWithQpSharing_DropsOffloadOnly) {
  params_.Set("RCCL_IB_QP_SHARING_ENABLE", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_TRUE(IbCastQpSharingGlobalEnable);
  EXPECT_FALSE(IbCastOffloadEnabled);
  EXPECT_TRUE(IbCastUseInline);
}

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_AinicWithMapFile_EnablesMultiplane) {
  SetMicroEnv("RCCL_MULTIPLANE_MAP_FILE", "/etc/multiplane.xml");
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_TRUE(IbCastMultiplaneEnable);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_MapFileWithoutAinicOrEmpty_LeavesMultiplaneOff) {
  World().AddHca("mlx5_0");
  SetMicroEnv("RCCL_MULTIPLANE_MAP_FILE", "/etc/multiplane.xml");
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(IbCastMultiplaneEnable);
  ResetForAnotherInit();
  g_rcclUseAinicValue = true;
  SetMicroEnv("RCCL_MULTIPLANE_MAP_FILE", "");
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(IbCastMultiplaneEnable);
}

// Init keeps no policy state: it only validates the param and reports what comms will use.
TEST_F(NetIbCastInitAinicMicrotest, InitDevices_UdmaPolicyOutOfRange_FallsBackToChannelAffinity) {
  for (int64_t policy : {(int64_t)ncclIbCastUDMAPolicy::First - 1, (int64_t)ncclIbCastUDMAPolicy::Last + 1}) {
    ResetForAnotherInit();
    params_.Set("RCCL_IB_UDMA_POLICY", policy);
    ScopedDebugLogging info(NCCL_LOG_INFO, NCCL_ALL);
    std::string log = CaptureLog([&] { ASSERT_EQ(Init(), ncclSuccess); });
    EXPECT_TRUE(LogHas(log, "is invalid")) << log;
    EXPECT_TRUE(LogHas(log, "uDMA policy: 0")) << log;
  }
  ResetForAnotherInit();
  params_.Set("RCCL_IB_UDMA_POLICY", (int64_t)ncclIbCastUDMAPolicy::Last);
  ScopedDebugLogging info(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = CaptureLog([&] { ASSERT_EQ(Init(), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "uDMA policy: 3")) << log;
}

TEST_F(NetIbCastInitAinicMicrotest, InitDevices_AinicWithResiliency_DropsOffloadAndInline) {
  params_.Set("IB_RESILIENCY_PORT_RECOVERY", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(IbCastOffloadEnabled);
  EXPECT_FALSE(IbCastUseInline);
  EXPECT_FALSE(IbCastAinicCtsInlineData);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_NotAinic_LeavesOffloadOffAndReadsInline) {
  World().AddHca("mlx5_0");
  params_.Set("IB_USE_INLINE", 1);
  ScopedHook count(g_ionicdvCtxGetUdmaCount, [](ibv_context*) -> uint8_t { return 4; });
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(count.calls, 0);
  EXPECT_FALSE(IbCastAinicRoce);
  EXPECT_FALSE(IbCastOffloadEnabled);
  EXPECT_TRUE(IbCastUseInline);
  EXPECT_FALSE(IbCastAinicCtsInlineData);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_SchedulerParamsFail_ReturnsTheError) {
  World();
  ScopedHook sched(g_ibCastQpSchedInitParms, [](struct ncclIbQpSchedParms*) { return ncclInvalidUsage; });
  EXPECT_EQ(Init(), ncclInvalidUsage);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_SchedulerWithResiliency_IsTurnedOff) {
  World();
  ScopedHook sched(g_ibCastQpSchedInitParms, [](struct ncclIbQpSchedParms* p) {
    p->enable = true;
    return ncclSuccess;
  });
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_TRUE(castGlobalQpSchedParms.enable);
  ResetForAnotherInit();
  params_.Set("IB_RESILIENCY_PORT_FAILOVER", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(castGlobalQpSchedParms.enable);
  ResetForAnotherInit();
  params_.Set("IB_RESILIENCY_PORT_FAILOVER", 0);
  params_.Set("IB_RESILIENCY_PORT_RECOVERY", 1);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(castGlobalQpSchedParms.enable);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_QpSharing_TurnsTheSchedulerOff) {
  World();
  params_.Set("RCCL_IB_QP_SHARING_ENABLE", 1);
  ScopedHook sched(g_ibCastQpSchedInitParms, [](struct ncclIbQpSchedParms* p) {
    p->enable = true;
    return ncclSuccess;
  });
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_TRUE(IbCastQpSharingGlobalEnable);
  EXPECT_FALSE(castGlobalQpSchedParms.enable);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_QpSharingWithNoGroups_TurnsSharingOff) {
  World();
  params_.Set("RCCL_IB_QP_SHARING_ENABLE", 1);
  params_.Set("RCCL_IB_COMM_NGROUPS", 0);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_FALSE(IbCastQpSharingGlobalEnable);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_ByOrderWithFailoverOrOooRq_WarnsAboutWhatItDisables) {
  World().AddHca("ionic_0");
  ScopedHook byOrder(g_ibCastByOrderRequested, [] { return true; });
  params_.Set("IB_RESILIENCY_PORT_FAILOVER", 1);
  params_.Set("IB_OOO_RQ", 1);
  std::string log = CaptureLog([&] { ASSERT_EQ(Init(), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "disabling resiliency")) << log;
  EXPECT_TRUE(LogHas(log, "disabling out-of-order RQ")) << log;
  EXPECT_EQ(g_ibCastReportMatchingSchemeCalls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, InitDevices_ByOrderAlone_OnlyReportsTheScheme) {
  World().AddHca("ionic_0");
  ScopedHook byOrder(g_ibCastByOrderRequested, [] { return true; });
  std::string log = CaptureLog([&] { ASSERT_EQ(Init(), ncclSuccess); });
  EXPECT_FALSE(LogHas(log, "BY_ORDER")) << log;
  EXPECT_EQ(g_ibCastReportMatchingSchemeCalls, 1);
}

class NetIbCastInitMicrotestIsolated : public NetIbCastInitMicrotest {};

TEST_F(NetIbCastInitMicrotestIsolated, InitDevices_IbHca_IsAnnouncedOncePerProcess) {
  RUN_ISOLATED_TEST("NetIbCastInit_IbHcaAnnouncedOnce", []() {
    FakeIbWorld world;
    SetMicroEnv("NCCL_IB_HCA", "ionic_0");
    ScopedDebugLogging info(NCCL_LOG_INFO, NCCL_ALL);
    std::string log = CaptureLog([&] { ASSERT_EQ(IbCastInitDevices(nullptr, nullptr), ncclSuccess); });
    EXPECT_TRUE(LogHas(log, "NCCL_IB_HCA set to ionic_0")) << log;
    world.Quiesce();
    NetIbCastResetGlobals();
    log = CaptureLog([&] { ASSERT_EQ(IbCastInitDevices(nullptr, nullptr), ncclSuccess); });
    EXPECT_FALSE(LogHas(log, "NCCL_IB_HCA set to")) << log;
  });
}

TEST_F(NetIbCastInitMicrotestIsolated, InitDevices_TelemetryCannotStart_CarriesOnWithoutIt) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "NetIbCastInit_TelemetryCannotStart",
      []() {
        FakeIbWorld world;
        world.AddHca("ionic_0");
        ScopedDebugLogging info(NCCL_LOG_INFO, NCCL_ALL);
        std::string log = CaptureLog([&] { EXPECT_EQ(IbCastInitDevices(nullptr, nullptr), ncclSuccess); });
        EXPECT_FALSE(rcclTelemetryOn());
        EXPECT_EQ(IbCastNDevs, 1);
        EXPECT_TRUE(LogHas(log, "telemetry was requested but could not start")) << log;
        EXPECT_FALSE(LogHas(log, "telemetry did not register")) << log;
      },
      {{"RCCL_TELEMETRY_ENABLE", "1"}, {"RCCL_TELEMETRY_OUTPUT_DIR", std::string(5000, 'd')}});
}

TEST_F(NetIbCastInitMicrotestIsolated, InitDevices_TelemetryOn_RegistersEachDeviceUnderItsSortedIndex) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "NetIbCastInit_TelemetryOn",
      []() {
        FakeIbWorld world;
        world.AddHca("ionic_0");
        world.AddHca("ionic_1");
        // ionic_1 sorts first, so a device registered under its listing index lands in the wrong slot.
        HookOf<decltype(g_initRealpath)> realpath{g_initRealpath, [](const char* path, char*) -> char* {
                                                    const std::string p(path);
                                                    if (p == "/sys/class/infiniband/ionic_0/device")
                                                      return strdup("/sys/devices/pci0000:80/0000:81:00.0");
                                                    if (p == "/sys/class/infiniband/ionic_1/device")
                                                      return strdup("/sys/devices/pci0000:00/0000:01:00.0");
                                                    return nullptr;
                                                  }};
        EXPECT_EQ(IbCastInitDevices(nullptr, nullptr), ncclSuccess);
        EXPECT_TRUE(rcclTelemetryOn());
        ASSERT_EQ(IbCastNDevs, 2);
        EXPECT_STREQ(IbCastDevs[0].devName, "ionic_1");
        EXPECT_EQ(rcclTelemetryNumDevs, 2);
        for (int d = 0; d < 2; d++) {
          EXPECT_EQ(rcclTelemetryDevs[d].device_id, d);
          EXPECT_STREQ(rcclTelemetryDevs[d].roce_device, IbCastDevs[d].devName) << d;
          EXPECT_STREQ(rcclTelemetryDevs[d].transport, "IB-CAST") << d;
        }
      },
      {{"RCCL_TELEMETRY_ENABLE", "1"}, {"RCCL_TELEMETRY_OUTPUT_DIR", "/tmp"}});
}


TEST_F(NetIbCastInitDevicesMicrotest, Init_Success_HandsBackAConfigWithTheTrafficClass) {
  World().AddHca("ionic_0");
  ncclNetCommConfig_t config{};
  config.trafficClass = 106;
  void* ctx = nullptr;
  ASSERT_EQ(IbCastInit(&ctx, 0, &config, nullptr, nullptr), ncclSuccess);
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(static_cast<ncclNetCommConfig_t*>(ctx)->trafficClass, 106);
  EXPECT_EQ(World().recoveryStart.calls, 1);
  int ndev = -1;
  ASSERT_EQ(IbCastDevices(&ndev), ncclSuccess);
  EXPECT_EQ(ndev, 1);
  ASSERT_EQ(IbCastFinalize(ctx), ncclSuccess);
  EXPECT_EQ(World().recoveryStop.calls, 1);
}

TEST_F(NetIbCastInitDevicesMicrotest, Init_DeviceInitFails_ReturnsWithoutStartingRecovery) {
  World();
  params_.Set("IB_DISABLE", 1);
  void* ctx = nullptr;
  ncclNetCommConfig_t config{};
  EXPECT_EQ(IbCastInit(&ctx, 0, &config, nullptr, nullptr), ncclInternalError);
  EXPECT_EQ(World().recoveryStart.calls, 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST_F(NetIbCastInitDevicesMicrotest, Init_RecoveryThreadFails_ReleasesTheDevicesReference) {
  World().AddHca("ionic_0");
  ScopedHook start(g_ibCastPortRecoveryThreadStart, [] { return ncclSystemError; });
  void* ctx = nullptr;
  ncclNetCommConfig_t config{};
  EXPECT_EQ(IbCastInit(&ctx, 0, &config, nullptr, nullptr), ncclSystemError);
  EXPECT_EQ(netRefCount, 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST_F(NetIbCastInitDevicesMicrotest, Finalize_RecoveryStopFails_KeepsTheReference) {
  World().AddHca("ionic_0");
  ncclNetCommConfig_t config{};
  void* ctx = nullptr;
  ASSERT_EQ(IbCastInit(&ctx, 0, &config, nullptr, nullptr), ncclSuccess);
  ScopedHook stop(g_ibCastPortRecoveryThreadStop, [] { return ncclSystemError; });
  EXPECT_EQ(IbCastFinalize(ctx), ncclSystemError);
  EXPECT_EQ(netRefCount, 1);
}

class NetIbCastInitPropertiesMicrotest : public NetIbCastInitMicrotest {
 protected:
  void SetUp() override {
    IbCastNMergedDevs = 0;
    struct ncclIbDev& dev = Dev(0, "mlx5_0");
    dev.guid = 0xabc;
    dev.maxQp = 77;
    dev.portNum = 1;
    dev.realPort = 1;
    dev.railId = 2;
    dev.planeId = 3;
    int d;
    ncclNetVDeviceProps_t props = Props({0});
    ASSERT_EQ(IbCastMakeVDevice(&d, &props), ncclSuccess);
  }
  // GDR is latched once per process; every test in this binary that reaches it sees peermem.
  ScopedHook<int()> peermem_{g_ncclIbScanDefaultPeerMemClients, [] { return 1; }};
};

TEST_F(NetIbCastInitPropertiesMicrotest, GetProperties_ReportsTheDeviceAndItsLimits) {
  ncclNetProperties_t props;
  memset(&props, 0, sizeof(props));
  ASSERT_EQ(IbCastGetProperties(0, &props), ncclSuccess);
  EXPECT_STREQ(props.name, "mlx5_0");
  EXPECT_EQ(props.speed, 100000);
  EXPECT_EQ(props.guid, 0xabcu);
  EXPECT_EQ(props.ptrSupport & (NCCL_PTR_HOST | NCCL_PTR_CUDA), NCCL_PTR_HOST | NCCL_PTR_CUDA);
  EXPECT_EQ(props.regIsGlobal, 1);
  EXPECT_EQ(props.forceFlush, 0);
  EXPECT_EQ(props.port, 2);  // ibv port 1 plus real port 1
  EXPECT_EQ(props.maxComms, 77);
  EXPECT_EQ(props.maxRecvs, NCCL_NET_IB_MAX_RECVS);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_HOST);
  EXPECT_EQ(props.netDeviceVersion, NCCL_NET_DEVICE_INVALID_VERSION);
  EXPECT_EQ(props.maxP2pBytes, NCCL_MAX_NET_SIZE_BYTES);
  EXPECT_EQ(props.maxCollBytes, (size_t)MAX_COLLNET_SIZE);
  EXPECT_EQ(props.maxMultiRequestSize, 1);
  EXPECT_EQ(props.railId, 2);
  EXPECT_EQ(props.planeId, 3);
  EXPECT_EQ(props.vProps.ndevs, 1);
}

TEST_F(NetIbCastInitPropertiesMicrotest, GetProperties_DataDirectAndCtsOffload_ChangeFlushAndRecvs) {
  IbCastDevs[0].ibProvider = IB_PROVIDER_MLX5;
  IbCastDevs[0].capsProvider.mlx5.dataDirect = 1;
  IbCastOffloadEnabled = true;
  params_.Set("RCCL_IB_P2P_DISABLE_CTS", 0);
  ncclNetProperties_t props;
  ASSERT_EQ(IbCastGetProperties(0, &props), ncclSuccess);
  EXPECT_EQ(props.forceFlush, 1);
  EXPECT_EQ(props.maxRecvs, 1);
  params_.Set("RCCL_IB_P2P_DISABLE_CTS", 1);  // offload without CTS: the full count again
  ASSERT_EQ(IbCastGetProperties(0, &props), ncclSuccess);
  EXPECT_EQ(props.maxRecvs, NCCL_NET_IB_MAX_RECVS);
}

// An ionic device's uDMA count shares capsProvider with mlx5.dataDirect.
TEST_F(NetIbCastInitPropertiesMicrotest, GetProperties_OnlyAnMlx5DataDirectDeviceForcesFlush) {
  IbCastDevs[0].capsProvider.ionic.udmaCount = 2;
  ncclNetProperties_t props;
  ASSERT_EQ(IbCastGetProperties(0, &props), ncclSuccess);
  EXPECT_EQ(props.forceFlush, 0);
  IbCastDevs[0].ibProvider = IB_PROVIDER_MLX5;
  IbCastDevs[0].capsProvider.mlx5.dataDirect = 0;
  ASSERT_EQ(IbCastGetProperties(0, &props), ncclSuccess);
  EXPECT_EQ(props.forceFlush, 0);
}

TEST_F(NetIbCastInitPropertiesMicrotest, GetProperties_MergedDevice_ReportsTheFusedNameSpeedAndIds) {
  Dev(1, "mlx5_1");
  OnRoot0(0);
  OnRoot0(1);
  int d;
  ncclNetVDeviceProps_t props = Props({0, 1});
  ASSERT_EQ(IbCastMakeVDevice(&d, &props), ncclSuccess);
  ncclNetProperties_t out;
  ASSERT_EQ(IbCastGetProperties(d, &out), ncclSuccess);
  EXPECT_STREQ(out.name, "mlx5_0+mlx5_1");
  EXPECT_EQ(out.speed, 200000);
  EXPECT_EQ(out.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(out.vProps.ndevs, 2);
}

TEST_F(NetIbCastInitPropertiesMicrotest, GetProperties_UnknownVirtualDevice_IsRejected) {
  ncclNetProperties_t props;
  EXPECT_EQ(IbCastGetProperties(1, &props), ncclInvalidUsage);
}

class NetIbCastInitPropertiesMicrotestIsolated : public NetIbCastInitMicrotest {};

TEST_F(NetIbCastInitPropertiesMicrotestIsolated, GetPhysProperties_GdrAndDmaBuf_AddTheirPointerTypes) {
  RUN_ISOLATED_TEST("NetIbCastInit_PhysPropertiesDmaBuf", []() {
    struct ibv_context ctx {};
    struct ibv_pd pd {};
    IbCastNDevs = 1;
    IbCastNMergedDevs = 1;
    IbCastDevs[0].context = &ctx;
    ScopedHook peermem(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [](struct ibv_context*, int) { return 0; });
    ScopedHook alloc(g_ibvAllocPd, [&](struct ibv_pd** ret, struct ibv_context*) {
      *ret = &pd;
      return ncclSuccess;
    });
    ScopedHook reg(g_ibvDirectRegDmabufMr, [](struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
      errno = EBADF;
      return (struct ibv_mr*)nullptr;
    });
    ncclNetProperties_t props;
    ASSERT_EQ(IbCastGetPhysProperties(0, &props), ncclSuccess);
    EXPECT_EQ(props.ptrSupport, NCCL_PTR_HOST | NCCL_PTR_DMABUF);
  });
}

}  // namespace
