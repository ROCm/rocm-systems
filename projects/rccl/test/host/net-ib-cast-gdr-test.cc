/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/gdr.cc (NET_IB_CAST_GDR_CC_PATH): GPUDirect
// RDMA, peermem and DMA-BUF detection. Every answer is latched by a std::call_once, so each case
// runs in its own process.

#include <gtest/gtest.h>

#include <errno.h>
#include <unistd.h>

#include <cstring>
#include <functional>
#include <string>

#include "../common/LogCapture.hpp"
#include "../common/ProcessIsolatedTestRunner.hpp"
#include "fakes/gdr_peermem_fakes.h"
#include "fakes/gdr_probe_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/os_fakes.h"
#include "gdr_peermem.h"
#include "net-ib-cast-test-common.h"
#include "fakes/param_redirect.h"

#include "fakes/libc_seam.h"
#include NET_IB_CAST_GDR_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

// A g_ncclOsTopoGetStrFromSys that answers bios_version and numa_balancing; an absent answer is a failed read.
auto SysfsAnswers(const char* bios, const char* numaBalancing) {
  return [=](const char*, const char* file, char* value, int maxLen) {
    const char* answer = strcmp(file, "bios_version") == 0 ? bios : strcmp(file, "numa_balancing") == 0 ? numaBalancing
                                                                                                          : nullptr;
    if (answer == nullptr) return ncclSystemError;
    snprintf(value, maxLen, "%s", answer);
    return ncclSuccess;
  };
}

constexpr const char* kHyperV = "Hyper-V UEFI Release v4.1";

class NetIbCastGdrMicrotestIsolated : public ::testing::Test {};

TEST_F(NetIbCastGdrMicrotestIsolated, GdrSupport_ForceEnable_SucceedsWithoutScanningForPeermem) {
  RUN_ISOLATED_TEST("NetIbCastGdr_ForceEnable", []() {
    ParamScript params;
    params.Set("RCCL_FORCE_ENABLE_GDRDMA", 1);
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    EXPECT_EQ(IbCastGdrSupport(), ncclSuccess);
    EXPECT_EQ(scan.calls, 0);
  });
}

TEST_F(NetIbCastGdrMicrotestIsolated, GdrSupport_PeermemClientFound_SucceedsWithoutProbing) {
  RUN_ISOLATED_TEST("NetIbCastGdr_PeermemFound", []() {
    ScopedHook sysfs(g_ncclOsTopoGetStrFromSys, SysfsAnswers("Generic BIOS", "1"));
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 1; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [](struct ibv_context*, int) { return 0; });
    EXPECT_EQ(IbCastGdrSupport(), ncclSuccess);
    EXPECT_EQ(IbCastGdrSupport(), ncclSuccess);
    EXPECT_EQ(scan.calls, 1);
    EXPECT_EQ(probe.calls, 0);
  });
}

TEST_F(NetIbCastGdrMicrotestIsolated, GdrSupport_NoPeermem_ProbesDeviceZeroOnceAndCachesSuccess) {
  RUN_ISOLATED_TEST("NetIbCastGdr_ProbeSucceeds", []() {
    struct ibv_context ctx {};
    IbCastDevs[0].context = &ctx;
    IbCastRelaxedOrderingEnabled = 1;
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [&](struct ibv_context* c, int ro) {
      EXPECT_EQ(c, &ctx);
      EXPECT_EQ(ro, 1);
      return 1;
    });
    EXPECT_EQ(IbCastGdrSupport(), ncclSuccess);
    EXPECT_EQ(IbCastGdrSupport(), ncclSuccess);
    EXPECT_EQ(probe.calls, 1);
  });
}

TEST_F(NetIbCastGdrMicrotestIsolated, GdrSupport_ProbeSaysNo_FailsAndDoesNotProbeAgain) {
  RUN_ISOLATED_TEST("NetIbCastGdr_ProbeFails", []() {
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [](struct ibv_context*, int) { return 0; });
    EXPECT_EQ(IbCastGdrSupport(), ncclSystemError);
    EXPECT_EQ(IbCastGdrSupport(), ncclSystemError);
    EXPECT_EQ(probe.calls, 1);
  });
}

// gdr_peermem.h says a probe that could not run (-1) must not be cached, but call_once latches it.
// Pins AICOMRCCL-3089; a fix flips this pin.
TEST_F(NetIbCastGdrMicrotestIsolated, GdrSupport_KnownBug_ProbeThatCouldNotRunIsNeverRetried) {
  RUN_ISOLATED_TEST("NetIbCastGdr_ProbeCouldNotRun", []() {
    int answer = -1;
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 0; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [&](struct ibv_context*, int) { return answer; });
    EXPECT_EQ(IbCastGdrSupport(), ncclSystemError);
    answer = 1;  // the device could now be probed
    EXPECT_EQ(IbCastGdrSupport(), ncclSystemError);
    EXPECT_EQ(probe.calls, 1);
  });
}

TEST_F(NetIbCastGdrMicrotestIsolated, GdrSupport_HyperVWithNumaBalancingAndNoRelaxedOrdering_IsBlocklisted) {
  RUN_ISOLATED_TEST("NetIbCastGdr_HyperVBlocklisted", []() {
    ParamScript params;
    params.Set("IB_PCI_RELAXED_ORDERING", 0);
    ScopedHook sysfs(g_ncclOsTopoGetStrFromSys, SysfsAnswers(kHyperV, "1"));
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 1; });
    ScopedHook probe(g_ncclIbProbeGdrSupport, [](struct ibv_context*, int) { return 1; });
    EXPECT_EQ(IbCastGdrSupport(), ncclSystemError);
    EXPECT_EQ(probe.calls, 0);  // a deliberate override is not probed past
  });
}

struct HyperVCase {
  const char* name;
  const char* numaBalancing;  // nullptr: the read fails
  int64_t relaxedOrdering;
};

class NetIbCastGdrHyperVMicrotestIsolated : public ::testing::TestWithParam<HyperVCase> {};

TEST_P(NetIbCastGdrHyperVMicrotestIsolated, GdrSupport_HyperVWithoutTheOtherConditions_KeepsPeermem) {
  const HyperVCase c = GetParam();
  RUN_ISOLATED_TEST((std::string("NetIbCastGdr_HyperV_") + c.name).c_str(), [c]() {
    ParamScript params;
    params.Set("IB_PCI_RELAXED_ORDERING", c.relaxedOrdering);
    ScopedHook sysfs(g_ncclOsTopoGetStrFromSys, SysfsAnswers(kHyperV, c.numaBalancing));
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 1; });
    EXPECT_EQ(IbCastGdrSupport(), ncclSuccess);
  });
}

INSTANTIATE_TEST_SUITE_P(NetIbCastGdr, NetIbCastGdrHyperVMicrotestIsolated,
                         ::testing::Values(HyperVCase{"NumaBalancingOff", "0", 0},
                                           HyperVCase{"NumaBalancingUnreadable", nullptr, 0},
                                           HyperVCase{"RelaxedOrderingOn", "1", 1},
                                           HyperVCase{"RelaxedOrderingDefault", "1", 2}),
                         [](const ::testing::TestParamInfo<HyperVCase>& info) { return info.param.name; });

TEST_F(NetIbCastGdrMicrotestIsolated, GdrSupport_BiosUnreadable_SkipsTheHyperVCheck) {
  RUN_ISOLATED_TEST("NetIbCastGdr_BiosUnreadable", []() {
    ParamScript params;
    params.Set("IB_PCI_RELAXED_ORDERING", 0);
    ScopedHook sysfs(g_ncclOsTopoGetStrFromSys, SysfsAnswers(nullptr, "1"));
    ScopedHook scan(g_ncclIbScanDefaultPeerMemClients, [] { return 1; });
    EXPECT_EQ(IbCastGdrSupport(), ncclSuccess);
  });
}

TEST_F(NetIbCastGdrMicrotestIsolated, PeerMemSupport_NvidiaPeermemLoaded_IsCheckedOnce) {
  RUN_ISOLATED_TEST("NetIbCastGdr_PeerMemLoaded", []() {
    std::string probed;
    ScopedHook access(g_access, [&](const char* path, int mode) {
      probed = path;
      EXPECT_EQ(mode, F_OK);
      return 0;
    });
    EXPECT_EQ(IbCastPeerMemSupport(), ncclSuccess);
    EXPECT_EQ(IbCastPeerMemSupport(), ncclSuccess);
    EXPECT_EQ(access.calls, 1);
    EXPECT_EQ(probed, "/sys/module/nvidia_peermem/version");
  });
}

TEST_F(NetIbCastGdrMicrotestIsolated, PeerMemSupport_NoNvidiaPeermem_Fails) {
  RUN_ISOLATED_TEST("NetIbCastGdr_PeerMemAbsent", []() {
    ScopedHook access(g_access, [](const char*, int) { return -1; });
    EXPECT_EQ(IbCastPeerMemSupport(), ncclSystemError);
  });
}

// Physical device 1 backs merged device 0, so the probe has to look through the merge table.
void TwoDevices(struct ibv_context* ctx0, struct ibv_context* ctx1) {
  IbCastNDevs = 2;
  IbCastNMergedDevs = 2;
  IbCastDevs[0].context = ctx0;
  IbCastDevs[1].context = ctx1;
  IbCastMergedDevs[0].vProps.ndevs = 1;
  IbCastMergedDevs[0].vProps.devs[0] = 1;
  IbCastMergedDevs[1].vProps.ndevs = 1;
  IbCastMergedDevs[1].vProps.devs[0] = 0;
}

TEST_F(NetIbCastGdrMicrotestIsolated, DmaBufSupport_KernelRejectsTheDummyFd_IsSupportedAndCachedPerDevice) {
  RUN_ISOLATED_TEST("NetIbCastGdr_DmaBufSupported", []() {
    struct ibv_context ctx0 {}, ctx1 {};
    struct ibv_pd pd {};
    TwoDevices(&ctx0, &ctx1);
    ScopedHook alloc(g_ibvAllocPd, [&](struct ibv_pd** ret, struct ibv_context* ctx) {
      EXPECT_EQ(ctx, &ctx1);
      *ret = &pd;
      return ncclSuccess;
    });
    ScopedHook reg(g_ibvDirectRegDmabufMr, [](struct ibv_pd*, uint64_t, size_t, uint64_t, int fd, int) {
      EXPECT_EQ(fd, -1);
      errno = EBADF;  // a kernel that knows DMA-BUF rejects the bogus fd itself
      return (struct ibv_mr*)nullptr;
    });
    ScopedHook dealloc(g_ibvDeallocPd, [&](struct ibv_pd* p) {
      EXPECT_EQ(p, &pd);
      return ncclSuccess;
    });
    EXPECT_EQ(IbCastDmaBufSupport(0), ncclSuccess);
    EXPECT_EQ(IbCastDevs[1].dmaBufSupported, 1);
    EXPECT_EQ(IbCastDmaBufSupport(0), ncclSuccess);
    EXPECT_EQ(alloc.calls, 1);
    EXPECT_EQ(dealloc.calls, 1);
  });
}

// A missing mlx5dv_reg_dmabuf_mr used to leave EOPNOTSUPP. If the ibv dummy call
// does not replace it, that errno is what selects ibv_reg_mr_iova2.
TEST_F(NetIbCastGdrMicrotestIsolated, DmaBufSupport_IgnoresALeftoverMlx5Errno) {
  RUN_ISOLATED_TEST("NetIbCastGdr_DmaBufLeftoverErrno", []() {
    struct ibv_context ctx0 {}, ctx1 {};
    struct ibv_pd pd {};
    TwoDevices(&ctx0, &ctx1);
    errno = EOPNOTSUPP;
    ScopedHook alloc(g_ibvAllocPd, [&](struct ibv_pd** ret, struct ibv_context*) {
      *ret = &pd;
      return ncclSuccess;
    });
    ScopedHook reg(g_ibvDirectRegDmabufMr, [](struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
      return (struct ibv_mr*)nullptr;  // does not write errno
    });
    ScopedHook dealloc(g_ibvDeallocPd, [](struct ibv_pd*) { return ncclSuccess; });
    EXPECT_EQ(IbCastDmaBufSupport(0), ncclSuccess);
    EXPECT_EQ(IbCastDevs[1].dmaBufSupported, 1);
  });
}

struct DmaBufFailure {
  const char* name;
  bool allocFails;
  int regErrno;
  bool deallocFails;
};

class NetIbCastGdrDmaBufMicrotestIsolated : public ::testing::TestWithParam<DmaBufFailure> {};

TEST_P(NetIbCastGdrDmaBufMicrotestIsolated, DmaBufSupport_IsUnsupported) {
  const DmaBufFailure c = GetParam();
  RUN_ISOLATED_TEST((std::string("NetIbCastGdr_DmaBuf_") + c.name).c_str(), [c]() {
    struct ibv_context ctx0 {}, ctx1 {};
    struct ibv_pd pd {};
    TwoDevices(&ctx0, &ctx1);
    ScopedHook alloc(g_ibvAllocPd, [&](struct ibv_pd** ret, struct ibv_context*) {
      *ret = c.allocFails ? nullptr : &pd;
      return c.allocFails ? ncclSystemError : ncclSuccess;
    });
    ScopedHook reg(g_ibvDirectRegDmabufMr, [&](struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
      errno = c.regErrno;
      return (struct ibv_mr*)nullptr;
    });
    ScopedHook dealloc(g_ibvDeallocPd,
                       [&](struct ibv_pd*) { return c.deallocFails ? ncclSystemError : ncclSuccess; });
    EXPECT_EQ(IbCastDmaBufSupport(1), ncclSystemError);
    EXPECT_EQ(IbCastDevs[0].dmaBufSupported, -1);
    EXPECT_EQ(IbCastDmaBufSupport(1), ncclSystemError);
    EXPECT_EQ(alloc.calls, 1);
  });
}

INSTANTIATE_TEST_SUITE_P(NetIbCastGdr, NetIbCastGdrDmaBufMicrotestIsolated,
                         ::testing::Values(DmaBufFailure{"PdAllocFails", true, EBADF, false},
                                           DmaBufFailure{"KernelLacksDmaBuf", false, EOPNOTSUPP, false},
                                           DmaBufFailure{"ProtocolUnsupported", false, EPROTONOSUPPORT, false},
                                           DmaBufFailure{"PdDeallocFails", false, EBADF, true}),
                         [](const ::testing::TestParamInfo<DmaBufFailure>& info) { return info.param.name; });

}  // namespace
