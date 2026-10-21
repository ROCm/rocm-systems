/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/init.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>

#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/socket_fakes.h"
#include "fakes/transport_net_ib_fakes.h"
#include "fakes/transport_net_ib_p2p_fakes.h"
#include "fakes/transport_net_ib_recovery_fakes.h"
#include "transport/net_ib/common.h"

namespace {

static ncclResult_t DefaultPropertySupport() { return ncclSystemError; }
std::function<ncclResult_t()> g_testGdrSupport = DefaultPropertySupport;
std::function<ncclResult_t(int)> g_testDmaBufSupport = [](int) { return ncclSystemError; };
std::function<char*(const char*, char*)> g_testRealpath = [](const char*, char*) -> char* { return nullptr; };
std::function<int(const char*, int)> g_testOpen = [](const char*, int) { return -1; };
std::function<ssize_t(int, void*, size_t)> g_testRead = [](int, void*, size_t) { return ssize_t{-1}; };
std::function<int(int)> g_testClose = [](int) { return 0; };

ncclResult_t TestGdrSupport() { return g_testGdrSupport(); }
ncclResult_t TestDmaBufSupport(int dev) { return g_testDmaBufSupport(dev); }
char* TestRealpath(const char* path, char* resolved) { return g_testRealpath(path, resolved); }
int TestOpen(const char* path, int flags) { return g_testOpen(path, flags); }
ssize_t TestRead(int fd, void* buffer, size_t size) { return g_testRead(fd, buffer, size); }
int TestClose(int fd) { return g_testClose(fd); }
// Starting the real detached worker before device sorting races with qsort.
// Keep discovery deterministic here; ncclIbAsyncThreadMain itself is covered
// directly in common-test.cc. Production fix tracked by AICOMRCCL-3112.
void* TestIbAsyncThreadMain(void*) { return nullptr; }

}  // namespace

void ncclSetThreadName(std::thread&, const char*, ...) {}

#define ncclIbGdrSupport TestGdrSupport
#define ncclIbDmaBufSupport TestDmaBufSupport
#define realpath TestRealpath
#define open TestOpen
#define read TestRead
#define close TestClose
#define ncclIbAsyncThreadMain TestIbAsyncThreadMain
#include NET_IB_INIT_CC_PATH
#undef ncclIbAsyncThreadMain
#undef close
#undef read
#undef open
#undef realpath
#undef ncclIbDmaBufSupport
#undef ncclIbGdrSupport

namespace {

class ScopedInitDevicesBaseline {
 private:
  ibv_device** devices_;
  int count_;

 public:
  ScopedInitDevicesBaseline(ibv_device** devices, int count)
      : devices_(devices),
        count_(count),
        symbols(g_ibvSymbols, [] { return ncclSuccess; }),
        forkInit(g_ibvForkInit, [] { return ncclSuccess; }),
        findInterfaces(g_ncclFindInterfaces, [](char* names, ncclSocketAddress*, int nameSize, int, int* count) {
          std::snprintf(names, nameSize, "eth0");
          *count = 1;
          return ncclSuccess;
        }),
        getDevices(g_ibvGetDeviceList, [this](ibv_device*** out, int* count) {
          *out = devices_;
          *count = count_;
          return ncclSuccess;
        }),
        freeDevices(g_ibvFreeDeviceList, [](ibv_device**) { return ncclSuccess; }),
        relaxedProbe(g_ibvRegMrIova2, [](ibv_mr**, ibv_pd*, void*, size_t, uint64_t, int) {
          return ncclInternalError;
        }),
        socketToString(g_ncclSocketToString, [](const ncclSocketAddress*, char* text, size_t, int) {
          std::strcpy(text, "127.0.0.1");
          return text;
        }) {}

  ScopedHook<ncclResult_t()> symbols;
  ScopedHook<ncclResult_t()> forkInit;
  ScopedHook<ncclResult_t(char*, ncclSocketAddress*, int, int, int*)> findInterfaces;
  ScopedHook<ncclResult_t(ibv_device***, int*)> getDevices;
  ScopedHook<ncclResult_t(ibv_device**)> freeDevices;
  ScopedHook<ncclResult_t(ibv_mr**, ibv_pd*, void*, size_t, uint64_t, int)> relaxedProbe;
  ScopedHook<const char*(const ncclSocketAddress*, char*, size_t, int)> socketToString;
};

class NetIbInitMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    netRefCount = 0;
    ncclNIbDevs = -1;
    ncclNMergedIbDevs = -1;
    ncclIbRelaxedOrderingEnabled = 0;
    ResetTransportNetIbP2pFakes();
  }

  void TearDown() override {
    for (int dev = 0; dev < MAX_IB_DEVS; ++dev) {
      std::free(ncclIbDevs[dev].pciPath);
      ncclIbDevs[dev].pciPath = nullptr;
    }
    g_testGdrSupport = DefaultPropertySupport;
    g_testDmaBufSupport = [](int) { return ncclSystemError; };
    g_testRealpath = [](const char*, char*) -> char* { return nullptr; };
    g_testOpen = [](const char*, int) { return -1; };
    g_testRead = [](int, void*, size_t) { return ssize_t{-1}; };
    g_testClose = [](int) { return 0; };
    ResetTransportNetIbRecoveryFakes();
    ResetTransportNetIbP2pFakes();
    ResetTransportNetIbFakes();
    ResetSocketFakes();
    ResetMlx5dvwrapFakes();
    ResetIbvwrapFakes();
    ResetEnvFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbInitMicrotest, NormalizePciPath_MultiPortAndVfComponentsBecomeZero) {
  std::array<char, PATH_MAX> normalized{};

  ncclIbNormalizePciPath("/sys/devices/pci0000:00/0000:41:00.7", normalized.data(), normalized.size());

  EXPECT_STREQ(normalized.data(), "/sys/devices/pci0000:00/0000:41:00.0");
}

TEST_F(NetIbInitMicrotest, NormalizePciPath_InvalidOrShortInputIsSafe) {
  std::array<char, 8> output{{'x', '\0'}};

  ncclIbNormalizePciPath(nullptr, output.data(), output.size());
  EXPECT_STREQ(output.data(), "x");
  ncclIbNormalizePciPath("abc", output.data(), output.size());
  EXPECT_STREQ(output.data(), "abc");
  ncclIbNormalizePciPath("abcdefghi", output.data(), 5);
  EXPECT_STREQ(output.data(), "00c0");
}

TEST_F(NetIbInitMicrotest, GetPciRoot_ValidPathReturnsDomainAndBus) {
  char root[8] = {};

  EXPECT_EQ(ncclIbGetPciRootFromPath("/sys/devices/pci0000:3a/0000:3a:00.0", root, sizeof(root)), ncclSuccess);
  EXPECT_STREQ(root, "0000:3a");
}

TEST_F(NetIbInitMicrotest, GetPciRoot_InvalidArgumentsOrPathAreRejected) {
  char root[8] = {};

  EXPECT_EQ(ncclIbGetPciRootFromPath(nullptr, root, sizeof(root)), ncclInvalidUsage);
  EXPECT_EQ(ncclIbGetPciRootFromPath("not-a-pci-path", root, sizeof(root)), ncclInvalidUsage);
  EXPECT_EQ(ncclIbGetPciRootFromPath("pci0000:00", root, 7), ncclInvalidUsage);
}

TEST_F(NetIbInitMicrotest, GetNumaNode_ValidSysfsValueIsParsedAndClosed) {
  ScopedHook open(g_testOpen, [](const char* path, int flags) {
    EXPECT_STREQ(path, "/sys/device/numa_node");
    EXPECT_EQ(flags, O_RDONLY);
    return 17;
  });
  ScopedHook read(g_testRead, [](int fd, void* buffer, size_t) {
    EXPECT_EQ(fd, 17);
    std::memcpy(buffer, "3\n", 2);
    return ssize_t{2};
  });
  ScopedHook close(g_testClose, [](int fd) {
    EXPECT_EQ(fd, 17);
    return 0;
  });

  EXPECT_EQ(ncclIbGetNumaNodeFromPath("/sys/device"), 3);
  EXPECT_EQ(open.calls, 1);
  EXPECT_EQ(read.calls, 1);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbInitMicrotest, GetNumaNode_NullOpenReadOrParseFailureReturnsUnknown) {
  EXPECT_EQ(ncclIbGetNumaNodeFromPath(nullptr), -1);
  EXPECT_EQ(ncclIbGetNumaNodeFromPath("/sys/device"), -1);

  ScopedHook open(g_testOpen, [](const char*, int) { return 5; });
  ScopedHook emptyRead(g_testRead, [](int, void*, size_t) { return ssize_t{0}; });
  EXPECT_EQ(ncclIbGetNumaNodeFromPath("/sys/device"), -1);

  ScopedHook invalidRead(g_testRead, [](int, void* buffer, size_t) {
    std::memcpy(buffer, "invalid", 7);
    return ssize_t{7};
  });
  EXPECT_EQ(ncclIbGetNumaNodeFromPath("/sys/device"), -1);
}

TEST_F(NetIbInitMicrotest, GetPciPath_RealpathSuccessPublishesRawAndNormalizedPaths) {
  ScopedHook path(g_testRealpath, [](const char*, char*) {
    return strdup("/sys/devices/pci0000:00/0000:41:02.7");
  });
  char* normalized = nullptr;
  char full[PATH_MAX] = {};
  char name[] = "mlx5_0";

  ASSERT_EQ(ncclIbGetPciPath(name, &normalized, full), ncclSuccess);

  ASSERT_NE(normalized, nullptr);
  EXPECT_STREQ(full, "/sys/devices/pci0000:00/0000:41:02.7");
  EXPECT_STREQ(normalized, "/sys/devices/pci0000:00/0000:41:00.0");
  std::free(normalized);
}

TEST_F(NetIbInitMicrotest, GetPciPath_MissingRealpathPublishesEmptyPath) {
  char* normalized = reinterpret_cast<char*>(0x1);
  char full[PATH_MAX] = {'x', '\0'};
  char name[] = "missing";

  EXPECT_EQ(ncclIbGetPciPath(name, &normalized, full), ncclSuccess);
  EXPECT_EQ(normalized, nullptr);
  EXPECT_STREQ(full, "");
}

TEST_F(NetIbInitMicrotest, GetRealPort_CountsOnlyEarlierMatchingPhysicalPaths) {
  ncclNIbDevs = 3;
  ncclIbDevs[0].pciPath = strdup("/sys/0000:41:00.0");
  ncclIbDevs[1].pciPath = strdup("/sys/0000:41:00.0");
  int port = -1;

  EXPECT_EQ(ncclIbGetRealPort(nullptr, &port, 2), ncclSuccess);
  EXPECT_EQ(port, 0);
  EXPECT_EQ(ncclIbGetRealPort(ncclIbDevs[0].pciPath, &port, 2), ncclSuccess);
  EXPECT_EQ(port, 2);
}

TEST_F(NetIbInitMicrotest, CompareDevs_EmptyPathSortsLastAndBdfSortsAscending) {
  ncclIbDev first{};
  ncclIbDev second{};
  std::strcpy(first.fullPciPath, "/sys/0000:42:00.0");
  std::strcpy(second.fullPciPath, "/sys/0000:41:00.0");

  EXPECT_GT(ncclIbCompareDevs(&first, &second), 0);
  EXPECT_LT(ncclIbCompareDevs(&second, &first), 0);
  first.fullPciPath[0] = '\0';
  EXPECT_GT(ncclIbCompareDevs(&first, &second), 0);
  second.fullPciPath[0] = '\0';
  EXPECT_EQ(ncclIbCompareDevs(&first, &second), 0);
}

TEST_F(NetIbInitMicrotest, CompareVDevsByPlane_PlaneThenPciPathDefinesOrder) {
  ncclIbDevs[0].planeId = 2;
  ncclIbDevs[1].planeId = 1;
  std::strcpy(ncclIbDevs[0].fullPciPath, "/sys/0000:41:00.0");
  std::strcpy(ncclIbDevs[1].fullPciPath, "/sys/0000:42:00.0");
  int first = 0;
  int second = 1;

  EXPECT_GT(ncclIbCompareVDevsByPlane(&first, &second), 0);
  ncclIbDevs[0].planeId = ncclIbDevs[1].planeId = 1;
  EXPECT_LT(ncclIbCompareVDevsByPlane(&first, &second), 0);
}

TEST_F(NetIbInitMicrotest, IsCx9_RequiresMellanoxVendorAndCx9FirmwarePrefix) {
  ncclIbDev dev{};
  dev.vendorId = NCCL_IB_MLX_VENDOR_ID;
  std::strcpy(dev.fwVer, "82.12.3456");
  EXPECT_TRUE(ncclIbIsCx9(&dev));
  dev.vendorId = 0;
  EXPECT_FALSE(ncclIbIsCx9(&dev));
  dev.vendorId = NCCL_IB_MLX_VENDOR_ID;
  std::strcpy(dev.fwVer, "28.12.3456");
  EXPECT_FALSE(ncclIbIsCx9(&dev));
}

TEST_F(NetIbInitMicrotest, GetPciIndex_FindsMatchingNormalizedPathOrReturnsCount) {
  ncclIbDev dev{};
  dev.pciPath = strdup("/sys/0000:41:00.0");
  const char* paths[] = {"/sys/0000:42:00.0", "/sys/0000:41:00.7"};

  EXPECT_EQ(ncclIbGetPciIndex(2, paths, &dev), 1);
  std::free(dev.pciPath);
  dev.pciPath = strdup("/sys/0000:43:00.0");
  EXPECT_EQ(ncclIbGetPciIndex(2, paths, &dev), 2);
  std::free(dev.pciPath);
}

TEST_F(NetIbInitMicrotest, WidthAndSpeed_FirstAdvertisedCapabilityIsSelected) {
  EXPECT_EQ(ncclIbWidth(1), 1);
  EXPECT_EQ(ncclIbWidth(1 << 3), 12);
  EXPECT_EQ(ncclIbSpeed(1), 2500);
  EXPECT_EQ(ncclIbSpeed(1 << 7), 100000);
}

TEST_F(NetIbInitMicrotest, MatchVfPath_MergeVfsControlsComparedSuffix) {
  const char* first = "/sys/pci/0000:41:00.1";
  const char* samePf = "/sys/pci/0000:41:00.7";
  const char* adjacentVf = "/sys/pci/0000:41:01.1";
  const char* otherDevice = "/sys/pci/0000:42:00.1";
  ScopedHook mergeVfs(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_MERGE_VFS") == 0 ? int64_t{1} : fallback;
  });
  EXPECT_EQ(ncclIbMatchVfPath(first, samePf), 1);
  EXPECT_EQ(ncclIbMatchVfPath(first, adjacentVf), 1);
  EXPECT_EQ(ncclIbMatchVfPath(first, otherDevice), 0);

  ScopedHook keepVfs(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_MERGE_VFS") == 0 ? int64_t{0} : fallback;
  });
  EXPECT_EQ(ncclIbMatchVfPath(first, samePf), 1);
  EXPECT_EQ(ncclIbMatchVfPath(first, adjacentVf), 0);
  EXPECT_EQ(ncclIbMatchVfPath(first, otherDevice), 0);
}

TEST_F(NetIbInitMicrotest, GetPlaneIndex_ReusesExistingAndAddsNewPlanes) {
  int16_t count = 1;
  int16_t planes[NCCL_IB_PLANE_MAX_INDEX] = {NCCL_NET_ID_UNDEF};
  int16_t index = -1;

  ASSERT_EQ(ncclIbGetPlaneIndex(NCCL_NET_ID_UNDEF, &count, planes, &index), ncclSuccess);
  EXPECT_EQ(index, 0);
  EXPECT_EQ(count, 1);
  ASSERT_EQ(ncclIbGetPlaneIndex(7, &count, planes, &index), ncclSuccess);
  EXPECT_EQ(index, 1);
  EXPECT_EQ(count, 2);
  EXPECT_EQ(planes[1], 7);
}

TEST_F(NetIbInitMicrotest, GetPlaneIndex_ReservedVirtualOrCapacityLimitIsRejected) {
  int16_t count = 1;
  int16_t planes[NCCL_IB_PLANE_MAX_INDEX] = {NCCL_NET_ID_UNDEF};
  int16_t index = -1;

  EXPECT_EQ(ncclIbGetPlaneIndex(NCCL_IB_PLANE_VIRT_BIT, &count, planes, &index), ncclInvalidUsage);

  count = NCCL_IB_PLANE_MAX_INDEX - 1;
  for (int i = 0; i < count; ++i) planes[i] = i;
  EXPECT_EQ(ncclIbGetPlaneIndex(99, &count, planes, &index), ncclInvalidUsage);
}

TEST_F(NetIbInitMicrotest, RelaxedOrderingCapable_QueriesOnlyEnabledModes) {
  ScopedHook disabledParam(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_PCI_RELAXED_ORDERING") == 0 ? int64_t{0} : fallback;
  });
  ScopedHook probe(g_ibvRegMrIova2, [](ibv_mr**, ibv_pd*, void*, size_t, uint64_t, int) {
    return ncclSuccess;
  });
  EXPECT_EQ(ncclIbRelaxedOrderingCapable(), 0);
  EXPECT_EQ(probe.calls, 0);

  ScopedHook enabledParam(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_PCI_RELAXED_ORDERING") == 0 ? int64_t{2} : fallback;
  });
  EXPECT_EQ(ncclIbRelaxedOrderingCapable(), 1);
  EXPECT_EQ(probe.calls, 1);
}

TEST_F(NetIbInitMicrotest, QueryOooRqSize_DisabledSkipsProviderQuery) {
  uint32_t size = 99;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_OOO_RQ") == 0 ? int64_t{0} : fallback;
  });
  ScopedHook query(g_mlx5dvQueryDevice, [](ibv_context*, mlx5dv_context*) {
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbQueryOooRqSize(nullptr, "mlx5_0", &size), ncclSuccess);
  EXPECT_EQ(size, 0u);
  EXPECT_EQ(query.calls, 0);
  EXPECT_EQ(ncclIbQueryOooRqSize(nullptr, "mlx5_0", nullptr), ncclInvalidArgument);
}

TEST_F(NetIbInitMicrotest, QueryOooRqSize_ProviderCapabilityPublishesMaximum) {
  uint32_t size = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_OOO_RQ") == 0 ? int64_t{1} : fallback;
  });
  ScopedHook query(g_mlx5dvQueryDevice, [](ibv_context*, mlx5dv_context* attrs) {
    attrs->comp_mask = MLX5DV_CONTEXT_MASK_OOO_RECV_WRS;
    attrs->ooo_recv_wrs_caps.max_rc = 73;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbQueryOooRqSize(nullptr, "mlx5_0", &size), ncclSuccess);
  EXPECT_EQ(size, 73u);
  EXPECT_EQ(query.calls, 1);
}

TEST_F(NetIbInitMicrotest, QueryOooRqSize_ProviderQueryFailureReturnsInternalError) {
  uint32_t size = 99;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_OOO_RQ") == 0 ? int64_t{1} : fallback;
  });
  ScopedHook query(g_mlx5dvQueryDevice, [](ibv_context*, mlx5dv_context*) {
    return ncclRemoteError;
  });

  EXPECT_EQ(ncclIbQueryOooRqSize(nullptr, "mlx5_0", &size), ncclInternalError);
  EXPECT_EQ(size, 0u);
}

TEST_F(NetIbInitMicrotest, Mlx5DmaBufCapable_ProbeErrnoSelectsCapability) {
  ibv_context context{};
  ibv_pd pd{};
  ScopedDmaBufProbe dmaProbe(&pd, EBADF);
  ScopedHook mlxProbe(g_mlx5dvDirectRegDmaBufMr,
                      [](ibv_pd*, uint64_t, size_t, uint64_t, int, int, int) -> ibv_mr* {
                        errno = EBADF;
                        return nullptr;
                      });

  EXPECT_TRUE(ncclMlx5dvDmaBufCapable(&context));

  ScopedHook unsupported(g_mlx5dvDirectRegDmaBufMr,
                         [](ibv_pd*, uint64_t, size_t, uint64_t, int, int, int) -> ibv_mr* {
                           errno = EOPNOTSUPP;
                           return nullptr;
                         });
  EXPECT_FALSE(ncclMlx5dvDmaBufCapable(&context));
  EXPECT_EQ(dmaProbe.alloc.calls, 2);
  EXPECT_EQ(dmaProbe.probe.calls, 2);
  EXPECT_EQ(dmaProbe.dealloc.calls, 2);
}

TEST_F(NetIbInitMicrotest, AutoAssignRailPlane_NoRailPolicyLeavesExplicitIdsUnchanged) {
  ncclNIbDevs = 2;
  ncclIbDevs[0].railId = 3;
  ncclIbDevs[1].railId = 3;
  ncclIbDevs[0].planeId = NCCL_NET_ID_UNDEF;
  ncclIbDevs[1].planeId = NCCL_NET_ID_UNDEF;
  const char* paths[MAX_IB_DEVS] = {};
  int count = 0;

  ASSERT_EQ(ncclIbAutoAssignRailPlane(0, paths, &count), ncclSuccess);
  ASSERT_EQ(ncclIbAutoAssignRailPlane(1, paths, &count), ncclSuccess);

  EXPECT_EQ(ncclIbDevs[0].planeId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(ncclIbDevs[1].planeId, NCCL_NET_ID_UNDEF);
}

TEST_F(NetIbInitMicrotest, MakeVDevice_ZeroDevicesIsRejected) {
  ncclNetVDeviceProps_t props{};
  int dev = -1;

  EXPECT_EQ(ncclIbMakeVDeviceInternal(&dev, &props), ncclInvalidUsage);
  EXPECT_EQ(dev, -1);
}

TEST_F(NetIbInitMicrotest, MakeVDevice_InvalidCountIndexOrIncompatibleLinkIsRejected) {
  ncclNIbDevs = 2;
  ncclNMergedIbDevs = 0;
  ncclIbDevs[0].link = IBV_LINK_LAYER_ETHERNET;
  ncclIbDevs[1].link = IBV_LINK_LAYER_INFINIBAND;
  int output = -1;
  ncclNetVDeviceProps_t props{};
  props.ndevs = -1;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&output, &props), ncclInvalidUsage);
  props.ndevs = NCCL_IB_MAX_DEVS_PER_NIC + 1;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&output, &props), ncclInvalidUsage);
  props.ndevs = 1;
  props.devs[0] = 2;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&output, &props), ncclInvalidUsage);
  props.ndevs = 2;
  props.devs[0] = 0;
  props.devs[1] = 1;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&output, &props), ncclInvalidUsage);
  EXPECT_EQ(ncclNMergedIbDevs, 0);
}

TEST_F(NetIbInitMicrotest, MakeVDevice_CompatibleDevicesAggregateNameSpeedRailAndPlane) {
  ncclNIbDevs = 2;
  ncclNMergedIbDevs = 0;
  std::strcpy(ncclIbDevs[0].devName, "mlx5_0");
  std::strcpy(ncclIbDevs[1].devName, "mlx5_1");
  ncclIbDevs[0].speed = 100;
  ncclIbDevs[1].speed = 200;
  ncclIbDevs[0].link = ncclIbDevs[1].link = IBV_LINK_LAYER_ETHERNET;
  ncclIbDevs[0].railId = ncclIbDevs[1].railId = 4;
  ncclIbDevs[0].planeIdx = 1;
  ncclIbDevs[1].planeIdx = 3;
  ncclIbDevs[0].pciPath = strdup("/sys/devices/pci0000:00/0000:41:00.0");
  ncclIbDevs[1].pciPath = strdup("/sys/devices/pci0000:00/0000:42:00.0");
  ncclNetVDeviceProps_t props{};
  props.ndevs = 2;
  props.devs[0] = 0;
  props.devs[1] = 1;
  int output = -1;

  ASSERT_EQ(ncclIbMakeVDeviceInternal(&output, &props), ncclSuccess);

  EXPECT_EQ(output, 0);
  EXPECT_EQ(ncclNMergedIbDevs, 1);
  EXPECT_STREQ(ncclIbMergedDevs[0].devName, "mlx5_0+mlx5_1");
  EXPECT_EQ(ncclIbMergedDevs[0].speed, 300);
  EXPECT_EQ(ncclIbMergedDevs[0].railId, 4);
  EXPECT_NE(ncclIbMergedDevs[0].planeId & (1 << 1), 0);
  EXPECT_NE(ncclIbMergedDevs[0].planeId & (1 << 3), 0);
}

TEST_F(NetIbInitMicrotest, MakeVDevice_DuplicatePhysicalIndexIsIncludedOnce) {
  ncclNIbDevs = 1;
  ncclNMergedIbDevs = 0;
  std::strcpy(ncclIbDevs[0].devName, "mlx5_0");
  ncclIbDevs[0].speed = 100;
  ncclIbDevs[0].link = IBV_LINK_LAYER_ETHERNET;
  ncclIbDevs[0].railId = 1;
  ncclIbDevs[0].planeId = 2;
  ncclIbDevs[0].pciPath = strdup("/sys/devices/pci0000:00/0000:41:00.0");
  ncclNetVDeviceProps_t props{};
  props.ndevs = 2;
  props.devs[0] = 0;
  props.devs[1] = 0;
  int output = -1;

  ASSERT_EQ(ncclIbMakeVDeviceInternal(&output, &props), ncclSuccess);

  EXPECT_EQ(ncclIbMergedDevs[0].vProps.ndevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].speed, 100);
  EXPECT_STREQ(ncclIbMergedDevs[0].devName, "mlx5_0");
}

TEST_F(NetIbInitMicrotest, MakeVDevice_PublicApiSerializesAndPublishesDevice) {
  ncclNIbDevs = 1;
  ncclNMergedIbDevs = 0;
  std::strcpy(ncclIbDevs[0].devName, "mlx5_0");
  ncclIbDevs[0].speed = 100;
  ncclIbDevs[0].link = IBV_LINK_LAYER_ETHERNET;
  ncclIbDevs[0].railId = 1;
  ncclIbDevs[0].planeId = 2;
  ncclIbDevs[0].pciPath = strdup("/sys/devices/pci0000:00/0000:41:00.0");
  ncclNetVDeviceProps_t props{};
  props.ndevs = 1;
  props.devs[0] = 0;
  int output = -1;

  EXPECT_EQ(ncclIbMakeVDevice(&output, &props), ncclSuccess);
  EXPECT_EQ(output, 0);
  EXPECT_EQ(ncclNMergedIbDevs, 1);
}

TEST_F(NetIbInitMicrotest, Devices_ReturnsMergedDeviceCount) {
  ncclNMergedIbDevs = 3;
  int count = -1;

  EXPECT_EQ(ncclIbDevices(&count), ncclSuccess);
  EXPECT_EQ(count, 3);
}

TEST_F(NetIbInitMicrotest, GetPhysProperties_PublishesCapabilitiesAndDeviceMetadata) {
  ncclIbDevs[0].speed = 200000;
  ncclIbDevs[0].portNum = 2;
  ncclIbDevs[0].realPort = 3;
  ncclIbDevs[0].maxQp = 77;
  ncclIbDevs[0].guid = 0xabc;
  ncclIbDevs[0].railId = 4;
  ncclIbDevs[0].planeId = 5;
  std::strcpy(ncclIbDevs[0].devName, "mlx5_0");
  ncclIbDevs[0].pciPath = strdup("/sys/devices/0000:41:00.0");
  g_testGdrSupport = [] { return ncclSuccess; };
  g_testDmaBufSupport = [](int dev) {
    EXPECT_EQ(dev, 0);
    return ncclSuccess;
  };
  ncclNetProperties_t props{};

  ASSERT_EQ(ncclIbGetPhysProperties(0, &props), ncclSuccess);

  EXPECT_STREQ(props.name, "mlx5_0");
  EXPECT_EQ(props.speed, 200000);
  EXPECT_EQ(props.port, 5);
  EXPECT_EQ(props.maxComms, 77);
  EXPECT_EQ(props.guid, 0xabcu);
  EXPECT_EQ(props.ptrSupport, NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF);
  EXPECT_EQ(props.railId, 4);
  EXPECT_EQ(props.planeId, 5);
}

TEST_F(NetIbInitMicrotest, GetProperties_ValidMergedDeviceOverridesAggregateFields) {
  ncclNMergedIbDevs = 1;
  ncclIbMergedDevs[0] = {};
  ncclIbMergedDevs[0].vProps.ndevs = 1;
  ncclIbMergedDevs[0].vProps.devs[0] = 0;
  std::strcpy(ncclIbMergedDevs[0].devName, "mlx5_0+mlx5_1");
  ncclIbMergedDevs[0].speed = 300;
  ncclIbMergedDevs[0].railId = 6;
  ncclIbMergedDevs[0].planeId = 7;
  std::strcpy(ncclIbDevs[0].devName, "mlx5_0");
  g_testGdrSupport = [] { return ncclSystemError; };
  g_testDmaBufSupport = [](int) { return ncclSystemError; };
  ncclNetProperties_t props{};

  ASSERT_EQ(ncclIbGetProperties(0, &props), ncclSuccess);

  EXPECT_STREQ(props.name, "mlx5_0+mlx5_1");
  EXPECT_EQ(props.speed, 300);
  EXPECT_EQ(props.railId, 6);
  EXPECT_EQ(props.planeId, 7);
  EXPECT_EQ(props.ptrSupport, NCCL_PTR_HOST);
  EXPECT_EQ(props.vProps.ndevs, 1);
  EXPECT_EQ(ncclIbGetProperties(1, &props), ncclInvalidUsage);
}

TEST_F(NetIbInitMicrotest, InitDevices_OneActivePortPublishesPhysicalAndMergedDevice) {
  ibv_device device{};
  std::strcpy(device.name, "mlx5_0");
  std::strcpy(device.dev_name, "uverbs0");
  ibv_device* devices[] = {&device, nullptr};
  ibv_context context{};
  ScopedInitDevicesBaseline baseline(devices, 1);
  ScopedHook mlxSymbols(g_mlx5dvSymbols, [] { return ncclSuccess; });
  ScopedHook open(g_ibvOpenDevice, [&context, &device](ibv_context** out, ibv_device* value) {
    EXPECT_EQ(value, &device);
    *out = &context;
    return ncclSuccess;
  });
  ScopedHook queryDevice(g_ibvQueryDevice, [](ibv_context*, ibv_device_attr* attr) {
    *attr = ibv_device_attr{};
    attr->phys_port_cnt = 1;
    attr->sys_image_guid = 0xabc;
    attr->vendor_id = 0x2c9;
    attr->vendor_part_id = 123;
    attr->max_qp = 99;
    std::strcpy(attr->fw_ver, "28.1");
    return ncclSuccess;
  });
  ScopedHook queryPort(g_ibvQueryPort, [](ibv_context*, uint8_t port, ibv_port_attr* attr) {
    EXPECT_EQ(port, 1);
    *attr = ibv_port_attr{};
    attr->state = IBV_PORT_ACTIVE;
    attr->link_layer = IBV_LINK_LAYER_ETHERNET;
    attr->active_width = 1;
    attr->active_speed = 1;
    return ncclSuccess;
  });
  int speedQueries = 0;
  ScopedHook querySpeed(g_ibvQueryPortSpeed, [&speedQueries](ibv_context*, uint8_t, uint64_t* speed) {
    if (speedQueries++ == 0) {
      *speed = 1000;
      return ncclSuccess;
    }
    return ncclSystemError;
  });
  ScopedHook provider(g_mlx5dvIsSupported, [](ibv_device*) { return true; });
  ibv_pd pd{};
  ScopedDmaBufProbe dmaProbe(&pd, EBADF);
  ScopedHook mlxDmaProbe(g_mlx5dvDirectRegDmaBufMr,
                         [](ibv_pd*, uint64_t, size_t, uint64_t, int, int, int) -> ibv_mr* {
                           errno = EBADF;
                           return nullptr;
                         });
  ScopedHook dataDirectPath(g_mlx5dvGetDataDirectSysfsPath, [](ibv_context*, char* path, size_t size) {
    std::snprintf(path, size, "/class/infiniband/mlx5_0/device");
    return ncclSuccess;
  });
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_DATA_DIRECT") == 0) return int64_t{2};
    if (std::strcmp(name, "IB_ADAPTIVE_ROUTING") == 0) return int64_t{0};
    if (std::strcmp(name, "IB_AR_THRESHOLD") == 0) return int64_t{4096};
    return fallback;
  });
  int pathCalls = 0;
  ScopedHook path(g_testRealpath, [&pathCalls](const char*, char*) {
    // Data Direct passes a null output path; production leaks its normalized
    // allocation when realpath succeeds there (AICOMRCCL-3112).
    return pathCalls++ == 0 ? strdup("/sys/devices/pci0000:00/0000:41:00.0") : nullptr;
  });
  ScopedHook gid(g_ncclIbGidInfoQuery,
                 [](ibv_context*, uint8_t, ibv_port_attr* attr, ncclIbGidInfo* info) {
                   info->link_layer = attr->link_layer;
                   info->localGidIndex = 2;
                   info->localGid.raw[15] = 0xaa;
                   return ncclSuccess;
                 });
  ASSERT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclSuccess);

  ASSERT_EQ(ncclNIbDevs, 2);
  EXPECT_EQ(ncclNMergedIbDevs, 2);
  EXPECT_STREQ(ncclIbDevs[0].devName, "mlx5_0");
  EXPECT_STREQ(ncclIbDevs[1].devName, "mlx5_0_dma");
  EXPECT_EQ(ncclIbDevs[0].portNum, 1);
  EXPECT_EQ(ncclIbDevs[1].capsProvider.mlx5.dataDirect, 1);
  EXPECT_STREQ(ncclIbDevs[1].pciPath, "/sys/class/infiniband/mlx5_0/device");
  EXPECT_EQ(ncclIbDevs[0].link, IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(ncclIbDevs[0].speed, 100000);
  EXPECT_EQ(ncclIbDevs[1].speed, 2500);
  EXPECT_EQ(ncclIbDevs[0].currSpeed, 100000u);
  EXPECT_EQ(ncclIbDevs[0].ar, 0);
  EXPECT_EQ(ncclIbDevs[1].ar, 0);
  EXPECT_EQ(ncclIbArThreshold, 4096);
  EXPECT_EQ(ncclIbDevs[0].gidInfo.localGidIndex, 2);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.devs[0], 0);
  EXPECT_EQ(ncclIbMergedDevs[1].vProps.devs[0], 1);
  EXPECT_EQ(baseline.getDevices.calls, 1);
  EXPECT_EQ(baseline.freeDevices.calls, 1);
  EXPECT_EQ(dmaProbe.allocContext, &context);
  EXPECT_EQ(dmaProbe.probePd, &pd);
  EXPECT_EQ(dmaProbe.alloc.calls, 1);
  EXPECT_EQ(dmaProbe.probe.calls, 1);
  EXPECT_EQ(dmaProbe.dealloc.calls, 1);
}

TEST_F(NetIbInitMicrotest, InitDevices_DisabledOrMissingVerbsReturnsInternalError) {
  ScopedHook disabled(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_DISABLE") == 0 ? int64_t{1} : fallback;
  });
  ScopedHook symbols(g_ibvSymbols, [] { return ncclSuccess; });
  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclInternalError);
  EXPECT_EQ(symbols.calls, 0);

  netRefCount = 0;
  ScopedHook enabled(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_DISABLE") == 0 ? int64_t{0} : fallback;
  });
  ScopedHook missing(g_ibvSymbols, [] { return ncclSystemError; });
  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclInternalError);
  EXPECT_EQ(missing.calls, 1);
}

TEST_F(NetIbInitMicrotest, InitDevices_NoIpInterfaceReturnsInternalErrorBeforeEnumeratingIbDevices) {
  ScopedHook symbols(g_ibvSymbols, [] { return ncclSuccess; });
  ScopedHook forkInit(g_ibvForkInit, [] { return ncclSuccess; });
  ScopedHook interfaces(g_ncclFindInterfaces,
                        [](char*, ncclSocketAddress*, int, int, int* count) {
                          *count = 0;
                          return ncclSuccess;
                        });
  ScopedHook devices(g_ibvGetDeviceList, [](ibv_device***, int*) {
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclInternalError);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(devices.calls, 0);
}

TEST_F(NetIbInitMicrotest, InitDevices_DeviceEnumerationFailureIsPropagated) {
  ScopedHook symbols(g_ibvSymbols, [] { return ncclSuccess; });
  ScopedHook interfaces(g_ncclFindInterfaces,
                        [](char*, ncclSocketAddress*, int, int, int* count) {
                          *count = 1;
                          return ncclSuccess;
                        });
  ScopedHook devices(g_ibvGetDeviceList, [](ibv_device*** out, int* count) {
    *out = nullptr;
    *count = 0;
    return ncclRemoteError;
  });

  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclInternalError);
  EXPECT_EQ(devices.calls, 1);
}

TEST_F(NetIbInitMicrotest, InitDevices_DeviceQueryFailureClosesContextAndContinuesWithoutDevice) {
  ibv_device device{};
  std::strcpy(device.name, "mlx5_0");
  ibv_device* devices[] = {&device, nullptr};
  ibv_context context{};
  ScopedInitDevicesBaseline baseline(devices, 1);
  ScopedHook open(g_ibvOpenDevice, [&context](ibv_context** out, ibv_device*) {
    *out = &context;
    return ncclSuccess;
  });
  ScopedHook query(g_ibvQueryDevice, [](ibv_context*, ibv_device_attr*) {
    return ncclSystemError;
  });
  ScopedHook close(g_ibvCloseDevice, [&context](ibv_context* value) {
    EXPECT_EQ(value, &context);
    return ncclSuccess;
  });
  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbInitMicrotest, InitDevices_OpenFailureSkipsDeviceAndContinues) {
  ibv_device device{};
  std::strcpy(device.name, "mlx5_0");
  ibv_device* devices[] = {&device, nullptr};
  ScopedInitDevicesBaseline baseline(devices, 1);
  ScopedHook open(g_ibvOpenDevice, [](ibv_context** out, ibv_device*) {
    *out = nullptr;
    return ncclSystemError;
  });
  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(open.calls, 1);
}

TEST_F(NetIbInitMicrotest, InitDevices_UnusablePortsAreSkippedAndContextIsClosed) {
  ibv_device device{};
  std::strcpy(device.name, "mlx5_0");
  ibv_device* devices[] = {&device, nullptr};
  ibv_context context{};
  ScopedInitDevicesBaseline baseline(devices, 1);
  ScopedHook open(g_ibvOpenDevice, [&context](ibv_context** out, ibv_device*) {
    *out = &context;
    return ncclSuccess;
  });
  ScopedHook queryDevice(g_ibvQueryDevice, [](ibv_context*, ibv_device_attr* attr) {
    *attr = ibv_device_attr{};
    attr->phys_port_cnt = 3;
    return ncclSuccess;
  });
  ScopedHook queryPort(g_ibvQueryPort, [](ibv_context*, uint8_t port, ibv_port_attr* attr) {
    *attr = ibv_port_attr{};
    if (port == 1) return ncclSystemError;
    if (port == 2) {
      attr->state = IBV_PORT_DOWN;
      attr->link_layer = IBV_LINK_LAYER_ETHERNET;
    } else {
      attr->state = IBV_PORT_ACTIVE;
      attr->link_layer = IBV_LINK_LAYER_UNSPECIFIED;
    }
    return ncclSuccess;
  });
  ScopedHook close(g_ibvCloseDevice, [](ibv_context*) { return ncclSuccess; });
  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(ncclNIbDevs, 0);
  EXPECT_EQ(queryPort.calls, 3);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(NetIbInitMicrotest, InitDevices_AdditionalReferenceSkipsRediscovery) {
  netRefCount = 1;
  ScopedHook symbols(g_ibvSymbols, [] { return ncclSystemError; });

  EXPECT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(netRefCount, 2);
  EXPECT_EQ(symbols.calls, 0);
}

TEST_F(NetIbInitMicrotest, InitAndFinalize_ContextCarriesTrafficClassAndRecoveryLifecycle) {
  ncclNIbDevs = 0;
  ncclNMergedIbDevs = 0;
  ScopedHook symbols(g_ibvSymbols, [] { return ncclSuccess; });
  ScopedHook start(g_ncclIbPortRecoveryThreadStart, [] { return ncclSuccess; });
  ScopedHook stop(g_ncclIbPortRecoveryThreadStop, [] { return ncclSuccess; });
  ncclNetCommConfig_t config{};
  config.trafficClass = 37;
  void* context = nullptr;

  ASSERT_EQ(ncclIbInitDevices(nullptr, nullptr), ncclSuccess);
  netRefCount = 0;
  ASSERT_EQ(ncclIbInit(&context, 0, &config, nullptr, nullptr), ncclSuccess);
  auto* stored = static_cast<ncclNetCommConfig_t*>(context);
  ASSERT_NE(stored, nullptr);
  EXPECT_EQ(stored->trafficClass, 37);
  EXPECT_EQ(start.calls, 1);

  EXPECT_EQ(ncclIbFinalize(context), ncclSuccess);
  EXPECT_EQ(stop.calls, 1);
  EXPECT_EQ(netRefCount, 0);
}

TEST_F(NetIbInitMicrotest, SetNetAttr_IsNoOpSuccess) {
  EXPECT_EQ(ncclIbSetNetAttr(nullptr, nullptr), ncclSuccess);
}

}  // namespace
