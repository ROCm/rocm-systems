/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/gdr.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include "ScopedHook.h"
#include "fakes/gdr_peermem_fakes.h"
#include "fakes/gdr_probe_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/os_fakes.h"
#include "fakes/param_redirect.h"
#include "transport/net_ib/NetIbDmaBufProbe.h"
#include "transport/net_ib/NetIbGlobals.h"
#include "transport/net_ib/common.h"

namespace {
int g_accessResult = -1;
int g_accessCalls = 0;
int TestAccess(const char*, int) {
  ++g_accessCalls;
  return g_accessResult;
}
}  // namespace

#define access TestAccess
#include NET_IB_GDR_CC_PATH
#undef access

namespace {

constexpr const char* kHyperV = "Hyper-V UEFI Release v4.1";

class NetIbGdrMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    g_accessResult = -1;
    g_accessCalls = 0;
    NetIbResetGlobals();
    ResetOsFakes();
    ResetIbvwrapFakes();
    ResetGdrProbeFakes();
    ResetGdrPeermemFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbGdrMicrotest, GdrSupportInitOnce_ForceEnablePublishesModuleWithoutScanning) {
  ncclIbGdrModuleLoaded = 0;
  ncclIbGdrBlocklisted = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "RCCL_FORCE_ENABLE_GDRDMA") == 0 ? int64_t{1} : fallback;
  });
  ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 1; });

  ibGdrSupportInitOnce();

  EXPECT_EQ(ncclIbGdrModuleLoaded, 1);
  EXPECT_EQ(scan.calls, 0);
}

TEST_F(NetIbGdrMicrotest, GdrSupportInitOnce_DiscoveredPeerMemPublishesModule) {
  ncclIbGdrModuleLoaded = 0;
  ncclIbGdrBlocklisted = 0;
  ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 1; });

  ibGdrSupportInitOnce();

  EXPECT_EQ(ncclIbGdrModuleLoaded, 1);
  EXPECT_EQ(scan.calls, 1);
}

TEST_F(NetIbGdrMicrotest, GdrSupportInitOnce_HyperVBlocklistNeedsNumaBalancingAndStrictOrdering) {
  struct Case {
    const char* numaBalancing;
    int64_t relaxedOrdering;
    int loaded;
    int blocklisted;
  };
  for (const Case& c : {Case{"1", 0, 0, 1}, Case{"0", 0, 1, 0}, Case{"1", 1, 1, 0}}) {
    SCOPED_TRACE(::testing::Message() << "numa_balancing=" << c.numaBalancing
                                      << " IB_PCI_RELAXED_ORDERING=" << c.relaxedOrdering);
    ncclIbGdrModuleLoaded = 0;
    ncclIbGdrBlocklisted = 0;
    ScopedHook topo(g_ncclOsTopoGetStrFromSys, SysfsAnswers(kHyperV, c.numaBalancing));
    ScopedHook params(g_loadParam, [&c](const char* name, int64_t fallback) {
      return std::strcmp(name, "IB_PCI_RELAXED_ORDERING") == 0 ? c.relaxedOrdering : fallback;
    });
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 1; });

    ibGdrSupportInitOnce();

    EXPECT_EQ(ncclIbGdrModuleLoaded, c.loaded);
    EXPECT_EQ(ncclIbGdrBlocklisted, c.blocklisted);
    EXPECT_EQ(topo.calls, 2);
  }
}

TEST_F(NetIbGdrMicrotest, GdrSupport_PublicApiUsesRuntimeProbeFallback) {
  EXPECT_EXIT(
      {
        static ibv_context context{};
        ncclIbDevs[0].context = &context;
        ncclIbGdrModuleLoaded = 0;
        ncclIbGdrBlocklisted = 0;
        int scans = 0;
        int probes = 0;
        g_ncclIbScanDefaultPeerMemClients = [&] {
          ++scans;
          return 0;
        };
        g_ncclIbProbeGdrSupport = [&](ibv_context* value, int) {
          ++probes;
          return value == &context ? 1 : 0;
        };
        if (ncclIbGdrSupport() != ncclSuccess) std::exit(1);
        if (ncclIbGdrSupport() != ncclSuccess) std::exit(2);
        if (scans != 1) std::exit(3);
        if (probes != 1) std::exit(4);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbGdrMicrotest, GdrSupport_PublicApiCachesProbeFailure) {
  EXPECT_EXIT(
      {
        static ibv_context context{};
        ncclIbDevs[0].context = &context;
        ncclIbGdrModuleLoaded = 0;
        ncclIbGdrBlocklisted = 0;
        int scans = 0;
        int probes = 0;
        g_ncclIbScanDefaultPeerMemClients = [&] {
          ++scans;
          return 0;
        };
        g_ncclIbProbeGdrSupport = [&](ibv_context*, int) {
          ++probes;
          return 0;
        };
        if (ncclIbGdrSupport() != ncclSystemError) std::exit(1);
        if (ncclIbGdrSupport() != ncclSystemError) std::exit(2);
        if (scans != 1) std::exit(3);
        if (probes != 1) std::exit(4);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbGdrMicrotest, GdrSupport_PublicApiHyperVBlocklistSkipsRuntimeProbe) {
  EXPECT_EXIT(
      {
        static ibv_context context{};
        ncclIbDevs[0].context = &context;
        ncclIbGdrModuleLoaded = 0;
        ncclIbGdrBlocklisted = 0;
        int probes = 0;
        g_ncclIbScanDefaultPeerMemClients = [] { return 1; };
        g_ncclOsTopoGetStrFromSys = SysfsAnswers(kHyperV, "1");
        g_loadParam = [](const char* name, int64_t fallback) {
          return std::strcmp(name, "IB_PCI_RELAXED_ORDERING") == 0 ? int64_t{0} : fallback;
        };
        g_ncclIbProbeGdrSupport = [&](ibv_context*, int) {
          ++probes;
          return 1;
        };
        if (ncclIbGdrSupport() != ncclSystemError) std::exit(1);
        if (ncclIbGdrSupport() != ncclSystemError) std::exit(2);
        if (probes != 0) std::exit(3);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbGdrMicrotest, PeerMemSupport_PublicApiReportsLoadedModule) {
  EXPECT_EXIT(
      {
        ncclIbPeerMemModuleLoaded = 0;
        g_accessResult = 0;
        g_accessCalls = 0;
        if (ncclIbPeerMemSupport() != ncclSuccess) std::exit(1);
        if (ncclIbPeerMemSupport() != ncclSuccess) std::exit(2);
        if (g_accessCalls != 1) std::exit(3);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbGdrMicrotest, PeerMemSupport_PublicApiCachesMissingModule) {
  EXPECT_EXIT(
      {
        ncclIbPeerMemModuleLoaded = 0;
        g_accessResult = -1;
        g_accessCalls = 0;
        if (ncclIbPeerMemSupport() != ncclSystemError) std::exit(1);
        if (ncclIbPeerMemSupport() != ncclSystemError) std::exit(2);
        if (g_accessCalls != 1) std::exit(3);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbGdrMicrotest, DmaBufSupportInitOnce_ProbeOutcomeSetsPhysicalDeviceSupport) {
  struct Case {
    const char* name;
    int probeErrno;
    bool allocFails;
    bool deallocFails;
    int supported;
  };
  const Case cases[] = {
      {"capable", EBADF, false, false, 1},
      {"kernel lacks DMA-BUF", EOPNOTSUPP, false, false, -1},
      {"protocol unsupported", EPROTONOSUPPORT, false, false, -1},
      {"PD allocation fails", EBADF, true, false, -1},
      {"PD release fails", EBADF, false, true, -1},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ibv_context context{};
    ibv_pd pd{};
    ncclIbMergedDevs[0].vProps.ndevs = 1;
    ncclIbMergedDevs[0].vProps.devs[0] = 0;
    ncclIbDevs[0].context = &context;
    ncclIbDevs[0].dmaBufSupported = 0;
    ibDmaSupportInitDev = 0;
    ScopedDmaBufProbe dmaProbe(&pd, c.probeErrno);
    dmaProbe.allocFails = c.allocFails;
    dmaProbe.deallocFails = c.deallocFails;

    ibDmaBufSupportInitOnce();

    EXPECT_EQ(ncclIbDevs[0].dmaBufSupported, c.supported);
    EXPECT_EQ(dmaProbe.allocContext, &context);
    EXPECT_EQ(dmaProbe.alloc.calls, 1);
    EXPECT_EQ(dmaProbe.probe.calls, c.allocFails ? 0 : 1);
    EXPECT_EQ(dmaProbe.dealloc.calls, c.allocFails ? 0 : 1);
    if (!c.allocFails) {
      EXPECT_EQ(dmaProbe.probePd, &pd);
      EXPECT_EQ(dmaProbe.probeFd, -1);
      EXPECT_EQ(dmaProbe.deallocPd, &pd);
    }
  }
}

TEST_F(NetIbGdrMicrotest, DmaBufSupport_PublicApiReturnsCachedCapability) {
  EXPECT_EXIT(
      {
        constexpr int kMergedDev = 2;
        constexpr int kPhysicalDev = 2;
        static ibv_context context{};
        static ibv_pd pd{};
        ncclIbMergedDevs[kMergedDev].vProps.devs[0] = kPhysicalDev;
        ncclIbDevs[kPhysicalDev].context = &context;
        int allocs = 0;
        int probes = 0;
        int deallocs = 0;
        g_ibvAllocPd = [&](ibv_pd** out, ibv_context*) {
          ++allocs;
          *out = &pd;
          return ncclSuccess;
        };
        g_ibvDirectRegDmabufMr = [&](ibv_pd*, uint64_t, size_t, uint64_t, int, int) -> ibv_mr* {
          ++probes;
          errno = EBADF;
          return nullptr;
        };
        g_ibvDeallocPd = [&](ibv_pd*) {
          ++deallocs;
          return ncclSuccess;
        };
        if (ncclIbDmaBufSupport(kMergedDev) != ncclSuccess) std::exit(1);
        if (ncclIbDmaBufSupport(kMergedDev) != ncclSuccess) std::exit(2);
        if (allocs != 1) std::exit(3);
        if (probes != 1) std::exit(4);
        if (deallocs != 1) std::exit(5);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbGdrMicrotest, DmaBufSupport_PublicApiCachesUnsupportedCapability) {
  EXPECT_EXIT(
      {
        constexpr int kMergedDev = 3;
        constexpr int kPhysicalDev = 3;
        static ibv_context context{};
        static ibv_pd pd{};
        ncclIbMergedDevs[kMergedDev].vProps.devs[0] = kPhysicalDev;
        ncclIbDevs[kPhysicalDev].context = &context;
        int probes = 0;
        g_ibvAllocPd = [&](ibv_pd** out, ibv_context*) {
          *out = &pd;
          return ncclSuccess;
        };
        g_ibvDirectRegDmabufMr = [&](ibv_pd*, uint64_t, size_t, uint64_t, int, int) -> ibv_mr* {
          ++probes;
          errno = EOPNOTSUPP;
          return nullptr;
        };
        g_ibvDeallocPd = [](ibv_pd*) { return ncclSuccess; };
        if (ncclIbDmaBufSupport(kMergedDev) != ncclSystemError) std::exit(1);
        if (ncclIbDmaBufSupport(kMergedDev) != ncclSystemError) std::exit(2);
        if (probes != 1) std::exit(3);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

}  // namespace
