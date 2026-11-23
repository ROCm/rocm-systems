/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/gin.cc: net_ib/gin-test.cc against the fork
 * (derived_transport_tests.cmake), then the fork's multi-segment tests.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <fcntl.h>
#include <unistd.h>

#include <string>

#include "fakes/hip_fakes.h"
#include "fakes/os_fakes.h"
#include "fakes/rccl_mseg_fakes.h"

#undef NET_IB_GIN_CC_PATH
#define NET_IB_GIN_CC_PATH "transport/net_ib_multiseg/shim/gin.cc"
// gin-test.cc's seam window defines these itself.
#undef ncclIbAcceptImpl
#undef ncclIbConnectImpl
#undef ncclIbCreateFlushQp
#undef ncclIbRegMrDmaBufInternal
#undef ncclIbSetTrafficClass
#define NetIbGinMicrotest NetIbMsegGinMicrotest
#include "transport/derived_transport_tests_begin.h"
#include "transport/net_ib/gin-test.cc"
#include "transport/derived_transport_tests_end.h"

namespace {

// The develop fixtures do not know the fork's fakes; put them back after every test.
class ResetRcclMsegFakesListener : public ::testing::EmptyTestEventListener {
  void OnTestEnd(const ::testing::TestInfo&) override { ResetRcclMsegFakes(); }
};
const bool g_resetRcclMsegFakes =
    (::testing::UnitTest::GetInstance()->listeners().Append(new ResetRcclMsegFakesListener), true);

TEST(NetIbMsegGinRmaSeqId, ReleaseSeqIdOnlyReturnsNewestId) {
  uint64_t posted = 5;
  rcclRmaReleaseSeqId(/*id=*/5, &posted);
  EXPECT_EQ(posted, uint64_t{4});

  posted = 7;
  rcclRmaReleaseSeqId(/*id=*/6, &posted);
  EXPECT_EQ(posted, uint64_t{7}) << "a later request already holds id 7";
}

// Switch on: connect allocates the per-rank registration records and close frees them.
TEST_F(NetIbMsegGinMicrotest, ProxyConnect_MultiSegAllocatesRegRecordsThatCloseFrees) {
  IbMsegListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  int recvToken = 2;
  ScopedHook multiSeg(g_multiSegOn, [](bool pluginRegistersPerSegment) { return pluginRegistersPerSegment; });
  ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  ScopedHook accept(g_ginAccept, [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  });
  ScopedHook closeRecv(g_ginNetCloseRecv, [](void*) { return ncclSuccess; });
  ScopedHook closeSend(g_ginNetCloseSend, [](void*) { return ncclSuccess; });
  void* coll = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll, nullptr), ncclSuccess);
  ASSERT_NE(coll, nullptr);
  auto* records = static_cast<IbMsegGinIbCollComm*>(coll)->regRecords;
  ASSERT_NE(records, nullptr);
  EXPECT_TRUE(g_hostAllocs.Covers(records, sizeof(rcclIbSegSymRecord)));

  ASSERT_EQ(IbMsegGinIbCloseColl(coll), ncclSuccess);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// Switch on: a failed record allocation closes both directions and hands back no coll comm.
TEST_F(NetIbMsegGinMicrotest, ProxyConnect_MultiSegRecordAllocFailureClosesCollComm) {
  IbMsegListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  int recvToken = 2;
  ScopedHook multiSeg(g_multiSegOn, [](bool pluginRegistersPerSegment) { return pluginRegistersPerSegment; });
  ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  ScopedHook accept(g_ginAccept, [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  });
  ScopedHook closeRecv(g_ginNetCloseRecv, [](void*) { return ncclSuccess; });
  ScopedHook closeSend(g_ginNetCloseSend, [](void*) { return ncclSuccess; });
  g_hostAllocs.failCall = 2; // the coll comm is call 1, the records call 2
  void* coll = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll, nullptr), ncclSystemError);
  EXPECT_EQ(coll, nullptr);
  EXPECT_EQ(closeRecv.calls, 1);
  EXPECT_EQ(closeSend.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

#if NCCL_CUMEM_DMABUF_EXPORT_GATE
// One rank with a two-segment device VMM window of kSegBytes per segment at kSegBase.
struct GinTwoSegmentScene {
  static constexpr uintptr_t kSegBase = 0x40000000;
  static constexpr size_t kSegBytes = 0x200000;
  IbMsegGinIbCollComm coll{};
  IbMsegRecvComm recv{};
  rcclIbSegSymRecord records[2] = {};

  GinTwoSegmentScene() {
    coll.rank = 0;
    coll.nranks = 1;
    coll.recvComm = &recv;
    coll.regRecords = records;
    coll.allGather = GinCollAllGather;
    recv.base.vProps.ndevs = 1;
  }
};

std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> TwoSegmentRanges() {
  return [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
    uintptr_t segBase = GinTwoSegmentScene::kSegBase;
    if (reinterpret_cast<uintptr_t>(ptr) >= segBase + GinTwoSegmentScene::kSegBytes) {
      segBase += GinTwoSegmentScene::kSegBytes;
    }
    *base = reinterpret_cast<hipDeviceptr_t>(segBase);
    *size = GinTwoSegmentScene::kSegBytes;
    return hipSuccess;
  };
}

// Each segment is kSegBytes long and aligned to it.
std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> AlignedSegmentRanges() {
  return [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
    *base = reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(ptr) & ~(GinTwoSegmentScene::kSegBytes - 1));
    *size = GinTwoSegmentScene::kSegBytes;
    return hipSuccess;
  };
}

// Every export hands back a fresh fd the walk owns.
std::function<hipError_t(void*, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long)> ExportsDevNull() {
  return [](void* handle, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long) {
    *static_cast<int*>(handle) = open("/dev/null", O_RDONLY);
    return hipSuccess;
  };
}

// A single-rank allGather.
ncclResult_t CopyGather(IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
  std::memcpy(dst, src, size);
  return ncclSuccess;
}

ncclResult_t RegSegments(GinTwoSegmentScene& scene, size_t bytes, void** handle) {
  return rcclRmaIbProxyRegMrSymDmaBuf(&scene.coll, reinterpret_cast<void*>(GinTwoSegmentScene::kSegBase), bytes,
                                      NCCL_PTR_CUDA, 0, -1, 0, handle);
}

// Segment locations: retain hands back the segment pointer as the handle, so isHost sees which segment it asks about.
struct GinSegLocations {
  explicit GinSegLocations(std::function<bool(uintptr_t)> isHost)
      : props(g_hipMemGetAllocationPropertiesFromHandle,
              [isHost](hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle) {
                *prop = hipMemAllocationProp{};
                prop->location.type = hipMemLocationTypeDevice;
                [[maybe_unused]] const bool host = isHost(reinterpret_cast<uintptr_t>(handle));
#if NCCL_CUMEM_HOST_GATE
                if (host) prop->location.type = hipMemLocationTypeHost;
#endif
                return hipSuccess;
              }) {}
  ScopedHook<hipError_t(hipMemGenericAllocationHandle_t*, void*)> retain{
      g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t* handle, void* ptr) {
        *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(ptr);
        return hipSuccess;
      }};
  ScopedHook<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)> props;
  ScopedHook<hipError_t(hipMemGenericAllocationHandle_t)> release{
      g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; }};
};

// Branch: nSeg > 1. A published owner fd and offset reach the second segment's DMA-BUF MR; the first is exported.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TwoSegments_UsesPublishedOwnerFd) {
  constexpr uint64_t kOwnerOffset = 0x3000;
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  int ownerFd = open("/dev/null", O_RDONLY);
  ASSERT_GE(ownerFd, 0);
  ScopedHook lookup(g_hostSegmentFdLookup, [ownerFd](uintptr_t base, size_t, uint64_t* offset) {
    if (base != GinTwoSegmentScene::kSegBase + GinTwoSegmentScene::kSegBytes) return -1;
    *offset = kOwnerOffset;
    return ownerFd;
  });
  struct Reg {
    uintptr_t data;
    size_t size;
    uint64_t offset;
    bool fdValid;
  };
  std::vector<Reg> regs;
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void* data, size_t size, int, uint64_t offset, int fd,
                                               uint64_t, void** handle) {
    size_t i = regs.size();
    regs.push_back({reinterpret_cast<uintptr_t>(data), size, offset, fd >= 0});
    ibHandles[i].mrs[0] = &mrs[i];
    *handle = &ibHandles[i];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return false; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->nSegments, 2);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->flushSeg, 0);
  EXPECT_EQ(locations.retain.calls, 1) << "the walk stops asking once it has a device segment";
  EXPECT_EQ(exportFd.calls, 1) << "only the first segment is exported";
  ASSERT_EQ(regs.size(), 2u);
  EXPECT_EQ(regs[0].offset, 0u);
  EXPECT_EQ(regs[1].data, GinTwoSegmentScene::kSegBase + GinTwoSegmentScene::kSegBytes);
  EXPECT_EQ(regs[1].size, GinTwoSegmentScene::kSegBytes);
  EXPECT_EQ(regs[1].offset, kOwnerOffset);
  EXPECT_TRUE(regs[1].fdValid);
  EXPECT_NE(fcntl(ownerFd, F_GETFD), -1) << "the published owner fd is borrowed, not closed";
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, 2);
  close(ownerFd);
}

// The builder has no segment cap: keys come from the window this transfer touches, bounded by maxWr.
TEST_F(NetIbMsegGinMicrotest, BuildSegmentedWrs_ResolvesKeysOverTheTouchedWindowOnly) {
  constexpr int kSeg = 24;
  constexpr size_t kBytes = 0x1000;
  std::vector<size_t> segOff(kSeg + 1);
  std::vector<uintptr_t> bases(kSeg);
  std::vector<uint64_t> rkeys((size_t)kSeg * NCCL_IB_MAX_DEVS_PER_NIC);
  std::vector<ibv_mr> mrs(kSeg);
  std::vector<IbMsegMrHandle> ibHandles(kSeg);
  std::vector<void*> mrHandle(kSeg);
  for (int s = 0; s <= kSeg; s++) segOff[s] = s * kBytes;
  for (int s = 0; s < kSeg; s++) {
    bases[s] = 0x10000000 + s * kBytes;
    rkeys[(size_t)s * NCCL_IB_MAX_DEVS_PER_NIC] = 0x500 + s;
    mrs[s].lkey = 0x900 + s;
    ibHandles[s].mrs[0] = &mrs[s];
    mrHandle[s] = &ibHandles[s];
  }
  rcclRmaIbProxySegMrHandle h{};
  h.nSegments = kSeg;
  h.segOff = segOff.data();
  h.mrHandle = mrHandle.data();
  h.sym.vas = bases.data();
  h.sym.keys = rkeys.data();
  IbMsegQp qp{};
  ibv_send_wr wr[RCCL_RMA_MAX_DATA_WRS] = {};
  ibv_sge sge[RCCL_RMA_MAX_DATA_WRS] = {};
  int nWr = 0;
  const uint64_t off = 20 * kBytes + kBytes / 2;

  ASSERT_EQ(rcclRmaBuildSegmentedWrs(wr, sge, RCCL_RMA_MAX_DATA_WRS, &nWr, IBV_WR_RDMA_WRITE, &qp, &h, 0, off, &h, 0,
                                     off, kBytes),
            ncclSuccess);
  ASSERT_EQ(nWr, 2);
  EXPECT_EQ(wr[0].wr.rdma.rkey, 0x514u);
  EXPECT_EQ(wr[1].wr.rdma.rkey, 0x515u);
  EXPECT_EQ(sge[0].lkey, 0x914u);
  EXPECT_EQ(sge[1].lkey, 0x915u);
  EXPECT_EQ(wr[1].wr.rdma.remote_addr, bases[0] + 21 * kBytes);
  EXPECT_EQ(sge[1].addr, bases[0] + 21 * kBytes);

  EXPECT_EQ(rcclRmaBuildSegmentedWrs(wr, sge, 1, &nWr, IBV_WR_RDMA_WRITE, &qp, &h, 0, off, &h, 0, off, kBytes),
            ncclInternalError)
      << "two touched segments do not fit one WR";
}

// Tables sized from the walked count: two ranks, two segments, so each rank's offsets sit nSeg + 1 apart.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TwoRanksTwoSegments_SizesTablesFromTheCount) {
  GinTwoSegmentScene scene;
  scene.coll.nranks = 2;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  int nReg = 0;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    mrs[nReg].rkey = 0x100 + nReg;
    ibHandles[nReg].mrs[0] = &mrs[nReg];
    *handle = &ibHandles[nReg++];
    return ncclSuccess;
  });
  // Both ranks contribute the same bytes.
  std::vector<size_t> gatherSizes;
  ScopedHook gather(g_ginCollAllGather, [&](IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
    gatherSizes.push_back(size);
    std::memcpy(dst, src, size);
    std::memcpy(static_cast<char*>(dst) + size, src, size);
    return ncclSuccess;
  });
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return false; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  auto* h = static_cast<rcclRmaIbProxySegMrHandle*>(handle);
  ASSERT_EQ(h->nSegments, 2);
  ASSERT_GE(gatherSizes.size(), 3u);
  EXPECT_EQ(gatherSizes[2], 3 * sizeof(size_t)) << "the offset gather carries nSeg + 1 entries";
  const size_t want[] = {0, GinTwoSegmentScene::kSegBytes, 2 * GinTwoSegmentScene::kSegBytes};
  for (int r = 0; r < 2; r++) {
    for (int i = 0; i < 3; i++) EXPECT_EQ(h->sym.rankOff[r * 3 + i], want[i]) << "rank " << r << " entry " << i;
  }
  EXPECT_EQ(rcclRmaRankOff(h, 1), h->sym.rankOff + 3);
  EXPECT_EQ(rcclRmaRemoteRkey(h, 1, 1, 0), 0x101u);
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, 2);
}

// Branch: nSeg > 1 with NCCL_DMABUF_ENABLE=0 fails through the status exchange before any MR is registered.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TwoSegmentsWithDmaBufOff_RejectsCollectively) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) -> int64_t {
    return std::string(name) == "DMABUF_ENABLE" ? 0 : fallback;
  });
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  void* handle = nullptr;

  EXPECT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclInvalidUsage);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_EQ(gather.calls, 1) << "the rejection travels in the first status exchange";
}

// A borrowed owner fd marks a host segment without asking the allocation; the first device segment is the flush one.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_OwnerFdSegmentFirst_FlushesTheExportedSegment) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  int ownerFd = open("/dev/null", O_RDONLY);
  ASSERT_GE(ownerFd, 0);
  ScopedHook lookup(g_hostSegmentFdLookup, [ownerFd](uintptr_t base, size_t, uint64_t* offset) {
    if (base != GinTwoSegmentScene::kSegBase) return -1;
    *offset = 0;
    return ownerFd;
  });
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  int nReg = 0;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    ibHandles[nReg].mrs[0] = &mrs[nReg];
    *handle = &ibHandles[nReg++];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  std::vector<uintptr_t> asked;
  GinSegLocations locations([&](uintptr_t ptr) {
    asked.push_back(ptr);
    return false;
  });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->flushSeg, 1);
  ASSERT_EQ(asked.size(), 1u) << "only the exported segment's location is asked";
  EXPECT_EQ(asked[0], GinTwoSegmentScene::kSegBase + GinTwoSegmentScene::kSegBytes);
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  close(ownerFd);
}

#if NCCL_CUMEM_HOST_GATE
// Every segment in host memory: no flush segment, so a flush of this window posts nothing.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_AllHostSegments_HasNoFlushSegment) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  int nReg = 0;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    ibHandles[nReg].mrs[0] = &mrs[nReg];
    *handle = &ibHandles[nReg++];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return true; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->flushSeg, -1);
  EXPECT_EQ(locations.retain.calls, 2);
  EXPECT_EQ(locations.release.calls, 2) << "every retained handle is released";
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
}
#endif

// A failed location query fails the walk: its fd is closed and no MR is registered.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_LocationQueryFails_FailsBeforeRegistering) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  int exported = -1;
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange,
                      [&](void* handle, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long) {
                        exported = open("/dev/null", O_RDONLY);
                        *static_cast<int*>(handle) = exported;
                        return hipSuccess;
                      });
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook retain(g_hipMemRetainAllocationHandle,
                    [](hipMemGenericAllocationHandle_t*, void*) { return hipErrorInvalidValue; });
  void* handle = nullptr;

  EXPECT_NE(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  ASSERT_GE(exported, 0);
  EXPECT_EQ(fcntl(exported, F_GETFD), -1) << "the exported fd is closed";
}

// A failed properties query after a successful retain still releases the retained handle.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_PropertiesQueryFails_ReleasesTheRetainedHandle) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook retain(g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t* handle, void* ptr) {
    *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(ptr);
    return hipSuccess;
  });
  ScopedHook props(g_hipMemGetAllocationPropertiesFromHandle,
                   [](hipMemAllocationProp*, hipMemGenericAllocationHandle_t) { return hipErrorInvalidValue; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  void* handle = nullptr;

  EXPECT_NE(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_GE(retain.calls, 1);
  EXPECT_EQ(release.calls, retain.calls) << "every retained handle is released";
}

// Past the old 16-segment limit: 17 segments register one MR each and agree on the count.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_SeventeenSegments_RegistersEachSegment) {
  constexpr int kSeg = 17;
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, AlignedSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  std::vector<ibv_mr> mrs(kSeg);
  std::vector<IbMsegMrHandle> ibHandles(kSeg);
  std::vector<uintptr_t> regData;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void* data, size_t, int, uint64_t, int, uint64_t, void** handle) {
    size_t i = regData.size();
    regData.push_back(reinterpret_cast<uintptr_t>(data));
    mrs[i].rkey = 0x100 + (uint32_t)i;
    ibHandles[i].mrs[0] = &mrs[i];
    *handle = &ibHandles[i];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return false; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, kSeg * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  auto* h = static_cast<rcclRmaIbProxySegMrHandle*>(handle);
  ASSERT_EQ(h->nSegments, kSeg);
  EXPECT_EQ(scene.records[0].nSegments, kSeg);
  ASSERT_EQ(regData.size(), (size_t)kSeg);
  for (int s = 0; s < kSeg; s++) {
    EXPECT_EQ(regData[s], GinTwoSegmentScene::kSegBase + s * GinTwoSegmentScene::kSegBytes) << s;
    EXPECT_EQ(rcclRmaRemoteRkey(h, 0, s, 0), 0x100u + s) << s;
  }
  EXPECT_EQ(rcclRmaRankOff(h, 0)[kSeg], kSeg * GinTwoSegmentScene::kSegBytes);
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, kSeg);
}

// Branch: ranks x segments past the table budget declines in the first exchange, before any MR or table.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TablePastTheBudget_DeclinesBeforeRegistering) {
  constexpr int kRanks = RCCL_RMA_SEG_TABLE_MAX_ENTRIES / RCCL_IB_REG_SEG_DEFAULT + 1;
  GinTwoSegmentScene scene;
  std::vector<rcclIbSegSymRecord> records(kRanks);
  scene.coll.nranks = kRanks;
  scene.coll.regRecords = records.data();
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, AlignedSegmentRanges());
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, [](IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
    for (int r = 0; r < kRanks; r++) std::memcpy(static_cast<char*>(dst) + r * size, src, size);
    return ncclSuccess;
  });
  void* handle = nullptr;

  EXPECT_EQ(RegSegments(scene, RCCL_IB_REG_SEG_DEFAULT * GinTwoSegmentScene::kSegBytes, &handle), ncclInvalidUsage);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_EQ(gather.calls, 1);
  EXPECT_EQ(records[kRanks - 1].nSegments, RCCL_IB_REG_SEG_DEFAULT);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// Branch: a walk past the supported count declines in the first exchange, before any MR or per-segment table.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_PastTheSupportedCount_DeclinesBeforeSizingTables) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, AlignedSegmentRanges());
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  void* handle = nullptr;

  EXPECT_EQ(RegSegments(scene, (RCCL_IB_REG_SEG_DEFAULT + 1) * GinTwoSegmentScene::kSegBytes, &handle), ncclInvalidUsage);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_EQ(gather.calls, 1);
  EXPECT_EQ(scene.records[0].nSegments, RCCL_IB_REG_SEG_DEFAULT + 1) << "every rank sees the walked count";
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// Branch: a zero-size window fails in the first exchange, before the walk or any MR.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_ZeroSize_FailsBeforeTheWalk) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, AlignedSegmentRanges());
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  void* handle = nullptr;

  EXPECT_EQ(RegSegments(scene, 0, &handle), ncclInternalError);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(ranges.calls, 0);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_EQ(gather.calls, 1) << "the failing rank still reaches the exchange";
  EXPECT_TRUE(g_hostAllocs.live.empty());
}
#endif

// Two ranks, n segments of kSegBytes: rank r's segment s at base + r * 0x100000 + s * kSegBytes, lkey lkey0 + s,
// rkey rkey0 + r * n + s in device slot 0.
struct GinSegHandle {
  static constexpr size_t kSegBytes = 0x100;
  std::vector<size_t> segOff;
  std::vector<uintptr_t> bases;
  std::vector<uint64_t> rkeys;
  std::vector<ibv_mr> mrs;
  std::vector<IbMsegMrHandle> ibHandles;
  std::vector<void*> slots;
  rcclRmaIbProxySegMrHandle h{};

  GinSegHandle(int n, uintptr_t base, uint32_t lkey0, uint32_t rkey0)
      : segOff(n + 1), bases(2 * n), rkeys((size_t)2 * n * NCCL_IB_MAX_DEVS_PER_NIC), mrs(n), ibHandles(n), slots(n) {
    for (int s = 0; s <= n; s++) segOff[s] = s * kSegBytes;
    for (int r = 0; r < 2; r++) {
      for (int s = 0; s < n; s++) {
        bases[r * n + s] = base + r * 0x100000 + s * kSegBytes;
        rkeys[(size_t)(r * n + s) * NCCL_IB_MAX_DEVS_PER_NIC] = rkey0 + r * n + s;
      }
    }
    for (int s = 0; s < n; s++) {
      mrs[s].lkey = lkey0 + s;
      ibHandles[s].mrs[0] = &mrs[s];
      slots[s] = &ibHandles[s];
    }
    h.tag = rcclRmaSegMrTag();
    h.nSegments = n;
    h.segOff = segOff.data();
    h.mrHandle = slots.data();
    h.sym.vas = bases.data();
    h.sym.keys = rkeys.data();
  }
};

// The data scene with the per-peer queues that multi-segment contexts allocate.
struct GinRoundScene : GinDataScene {
  std::array<rcclRmaRoundQueue, 2> sendQueues{};
  GinRoundScene() { context.pending.send = sendQueues.data(); }
  int Test(IbMsegRequest* req, ncclResult_t expect = ncclSuccess) {
    int done = 0;
    EXPECT_EQ(IbMsegRmaIbProxyTest(nullptr, req, &done), expect);
    return done;
  }
  int SignaledCount() const {
    int n = 0;
    for (const auto& p : verbs.posted) n += (p.sendFlags & IBV_SEND_SIGNALED) != 0;
    return n;
  }
};

// A partial round stops at maxWr and resolves keys over the segments it reaches.
TEST_F(NetIbMsegGinMicrotest, BuildSegmentedWrs_CoveredStopsAtMaxWrAndReportsBytes) {
  GinSegHandle seg(24, 0x10000000, 0x900, 0x500);
  IbMsegQp qp{};
  ibv_send_wr wr[4] = {};
  ibv_sge sge[4] = {};
  int nWr = 0;
  uint64_t covered = 0;
  const uint64_t off = 18 * GinSegHandle::kSegBytes;
  ASSERT_EQ(rcclRmaBuildSegmentedWrs(wr, sge, 4, &nWr, IBV_WR_RDMA_WRITE, &qp, &seg.h, 0, off, &seg.h, 1, off,
                                     6 * GinSegHandle::kSegBytes, &covered),
            ncclSuccess);
  ASSERT_EQ(nWr, 4);
  EXPECT_EQ(covered, 4 * GinSegHandle::kSegBytes);
  EXPECT_EQ(sge[3].lkey, 0x900u + 21);
  EXPECT_EQ(wr[3].wr.rdma.rkey, 0x500u + seg.h.nSegments + 21);
  EXPECT_EQ(wr[3].wr.rdma.remote_addr, seg.bases[seg.h.nSegments] + 21 * GinSegHandle::kSegBytes);
}

// A put wider than one round posts 64 WRs at a time. Each round's signaled tail starts the next round and does not
// complete the request; the last round's tail does.
TEST_F(NetIbMsegGinMicrotest, IPut_WiderThanOneRound_PostsTheNextRoundWhenEachTailCompletes) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  // A half-segment skew splits every segment in two: 138 WRs, posted as rounds of 64, 64 and 10.
  const size_t skew = GinSegHandle::kSegBytes / 2;
  const size_t size = 69 * GinSegHandle::kSegBytes;
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, skew, &src.h, size, 0, &dst.h, 1, 0, &request), ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  EXPECT_EQ(req->events[0], 1);
  const uint64_t wrId = req - scene.send.base.reqs;

  const size_t rounds[] = {64, 128, 138};
  for (int r = 0; r < 3; r++) {
    ASSERT_EQ(scene.verbs.posted.size(), rounds[r]) << "round " << r;
    EXPECT_EQ(scene.SignaledCount(), r + 1);
    EXPECT_NE(scene.verbs.posted.back().sendFlags & IBV_SEND_SIGNALED, 0);
    scene.QueueCompletion(req);
    EXPECT_EQ(scene.Test(req), r == 2 ? 1 : 0) << "round " << r;
    if (r < 2) EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  }
  EXPECT_EQ(scene.send.ginSeq.completed, req->id);
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
  for (size_t i = 0; i < scene.verbs.posted.size(); i++) {
    const auto& p = scene.verbs.posted[i];
    EXPECT_EQ(p.wrId, wrId) << i;
    EXPECT_EQ(p.sge.length, skew) << i;
    EXPECT_EQ(p.sge.addr, src.bases[0] + skew + i * skew) << i;
    EXPECT_EQ(p.remoteAddress, dst.bases[70] + i * skew) << i;
  }
  rcclIbSegStats stats{};
  ASSERT_EQ(rcclRmaIbProxySegStats(&scene.context, 0, 1, &stats), ncclSuccess);
  EXPECT_EQ(stats.posts, 3u) << "one count per posted round, none for the probe build";
  EXPECT_EQ(stats.wrs, 138u);
  EXPECT_EQ(stats.laterSegPosts, 3u);
  EXPECT_EQ(rcclRmaIbProxySegStats(&scene.context, 0, 2, &stats), ncclInvalidArgument);
}

// An op that arrives while a wider op is queued waits behind it, even one whose handle would take the single-MR
// path, so its completion cannot pass the earlier request.
TEST_F(NetIbMsegGinMicrotest, IPut_BehindAQueuedOp_PostsAfterItsLastRound) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  GinSegHandle one(1, 0x30000000, 0xe00, 0xf00);
  void* wide = nullptr;
  void* small = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &wide),
            ncclSuccess);
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, 0, &one.h, 8, 0, &one.h, 1, 0, &small), ncclSuccess);
  auto* wideReq = static_cast<IbMsegRequest*>(wide);
  auto* smallReq = static_cast<IbMsegRequest*>(small);
  ASSERT_NE(smallReq, nullptr);
  EXPECT_EQ(smallReq->id, wideReq->id + 1);
  EXPECT_EQ(scene.verbs.posted.size(), 64u);
  EXPECT_EQ(scene.Test(smallReq), 0);

  for (int r = 0; r < 2; r++) {
    scene.QueueCompletion(wideReq);
    EXPECT_EQ(scene.Test(smallReq), 0);
  }
  ASSERT_EQ(scene.verbs.posted.size(), 139u) << "the small put follows the wide put's last round";
  EXPECT_EQ(scene.verbs.posted.back().wrId, (uint64_t)(smallReq - scene.send.base.reqs));
  EXPECT_EQ(scene.verbs.posted.back().remoteAddress, one.bases[1]);
  EXPECT_EQ(scene.verbs.posted.back().sge.lkey, 0xe00u);
  scene.QueueCompletion(wideReq);
  scene.QueueCompletion(smallReq);
  EXPECT_EQ(scene.Test(smallReq), 1);
  EXPECT_EQ(scene.Test(wideReq), 1);
}

// The signal rides only the last round, after every data WR of the op.
TEST_F(NetIbMsegGinMicrotest, IPutSignal_WiderThanOneRound_SignalsOnlyInTheLastRound) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPutSignal(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h,
                                     69 * GinSegHandle::kSegBytes, 0, &dst.h, 1, 8, &dst.h, 5,
                                     NCCL_NET_SIGNAL_OP_ADD, false, 0, &request),
            ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  for (int r = 0; r < 2; r++) {
    scene.QueueCompletion(req);
    EXPECT_EQ(scene.Test(req), 0);
  }
  ASSERT_EQ(scene.verbs.posted.size(), 139u);
  for (size_t i = 0; i + 1 < scene.verbs.posted.size(); i++) {
    EXPECT_EQ(scene.verbs.posted[i].opcode, IBV_WR_RDMA_WRITE) << i;
  }
  const auto& signal = scene.verbs.posted.back();
  EXPECT_EQ(signal.opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(signal.compareAdd, 5u);
  EXPECT_EQ(signal.remoteAddress, dst.bases[70] + 8);
  EXPECT_EQ(signal.rkey, 0xd00u + 70);
  EXPECT_NE(signal.sendFlags & IBV_SEND_SIGNALED, 0);
}

// A round that fails to post marks its request FAILED, and its next test reports the error.
TEST_F(NetIbMsegGinMicrotest, IPut_LaterRoundPostFailure_FailsTheRequest) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &request),
            ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  scene.verbs.postSendFailOnCall = 2;
  scene.QueueCompletion(req);
  EXPECT_EQ(scene.Test(req), 0);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
  EXPECT_EQ(scene.Test(req, ncclRemoteError), 1);
}

// A chain whose prefix posted keeps its request: the accepted WRs complete into it, and test() reports the lost tail.
TEST_F(NetIbMsegGinMicrotest, IPut_PrefixPostOfADirectChain_KeepsTheRequestUntilTestFails) {
  GinRoundScene scene;
  scene.context.wrBatchsize = 8;
  scene.verbs.postSendFailOnCall = 1;
  scene.verbs.postSendFailAcceptsWrs = 5;
  GinSegHandle src(20, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(20, 0x20000000, 0xc00, 0xd00);
  void* request = nullptr;
  // A half-segment skew gives 38 WRs: one round, past the batch, so the chain posts directly.
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 19 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &request),
            ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  EXPECT_EQ(scene.verbs.posted.size(), 5u);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(scene.send.ginSeq.posted, req->id);
  EXPECT_EQ(scene.Test(req, ncclRemoteError), 1);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
}

// A later round whose prefix posted fails its request, which leaves the queue and reports the error on its next test.
TEST_F(NetIbMsegGinMicrotest, IPut_PrefixPostOfALaterRound_FailsTheRequest) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &request),
            ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 64u);
  scene.verbs.postSendFailOnCall = 2;
  scene.verbs.postSendFailAcceptsWrs = 10;
  scene.QueueCompletion(req);
  EXPECT_EQ(scene.Test(req), 0);
  EXPECT_EQ(scene.verbs.posted.size(), 74u);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
  EXPECT_EQ(scene.Test(req, ncclRemoteError), 1);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
}

// A wide op rings the aggregated batch ahead of it first. If that doorbell fails, nothing is published or queued.
TEST_F(NetIbMsegGinMicrotest, IPut_BatchDoorbellFailure_PublishesNoRequest) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* batched = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, 0, &src.h, 8, 0, &dst.h, 1, ncclRmaOptFlagsAggregateRequests,
                               &batched),
            ncclSuccess);
  ASSERT_EQ(scene.verbs.posted.size(), 0u);
  scene.verbs.postSendFailOnCall = 1;
  void* request = nullptr;
  EXPECT_NE(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &request),
            ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
  // The failed doorbell still empties the batch, so a later post cannot resend WRs the HCA accepted.
  EXPECT_EQ(scene.context.batch.nWrs, 0);
  EXPECT_EQ(scene.context.batch.qp, nullptr);
}

// The aggregated requests of a batch whose doorbell failed lost their signaled tail. They report the error, even
// after a later request on the same QP completes.
TEST_F(NetIbMsegGinMicrotest, IPut_BatchDoorbellFailure_FailsTheAggregatedRequests) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* first = nullptr;
  void* second = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, 0, &src.h, 8, 0, &dst.h, 1, ncclRmaOptFlagsAggregateRequests,
                               &first),
            ncclSuccess);
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, 8, &src.h, 8, 8, &dst.h, 1, ncclRmaOptFlagsAggregateRequests,
                               &second),
            ncclSuccess);
  ASSERT_EQ(scene.verbs.posted.size(), 0u);
  scene.verbs.postSendFailOnCall = 1;
  void* request = nullptr;
  EXPECT_NE(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &request),
            ncclSuccess);
  auto* firstReq = static_cast<IbMsegRequest*>(first);
  auto* secondReq = static_cast<IbMsegRequest*>(second);
  EXPECT_EQ(firstReq->type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(secondReq->type, NCCL_NET_IB_REQ_FAILED);

  void* later = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, 16, &src.h, 8, 16, &dst.h, 1, 0, &later), ncclSuccess);
  auto* laterReq = static_cast<IbMsegRequest*>(later);
  ASSERT_NE(laterReq, nullptr);
  scene.QueueCompletion(laterReq);
  EXPECT_EQ(scene.Test(laterReq), 1);
  EXPECT_EQ(scene.Test(firstReq, ncclRemoteError), 1);
  EXPECT_EQ(scene.Test(secondReq, ncclRemoteError), 1);
}

// A multi-segment flush is one signaled read of the flush segment, through that segment's local MR.
TEST_F(NetIbMsegGinMicrotest, IFlush_MultiSegment_ReadsOneByteOfTheFlushSegment) {
  GinRoundScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
  GinSegHandle win(40, 0x10000000, 0xa00, 0xb00);
  win.h.flushSeg = 5;
  win.mrs[5].rkey = 0x7705;
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIFlush(&scene.context, 0, &win.h, 1, &request), ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  const auto& flush = scene.verbs.posted[0];
  EXPECT_EQ(flush.qp, scene.flushQp);
  EXPECT_EQ(flush.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(flush.remoteAddress, win.bases[5]);
  EXPECT_EQ(flush.rkey, 0x7705u);
  EXPECT_EQ(flush.sge.addr, 0x9000u);
  EXPECT_NE(flush.sendFlags & IBV_SEND_SIGNALED, 0);
  scene.QueueCompletion(req);
  EXPECT_EQ(scene.Test(req), 1);
  EXPECT_EQ(scene.recv.ginSeq.completed, scene.recv.ginSeq.posted);
}

// A window with no device segment has nothing for a GPUDirect flush to order.
TEST_F(NetIbMsegGinMicrotest, IFlush_AllHostSegments_PostsNothing) {
  GinRoundScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
  GinSegHandle win(3, 0x10000000, 0xa00, 0xb00);
  win.h.flushSeg = -1;
  void* request = reinterpret_cast<void*>(0x1);
  ASSERT_EQ(rcclRmaIbProxyIFlush(&scene.context, 0, &win.h, 1, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
  EXPECT_EQ(createFlushQp.calls, 0);
}

// A flush does not wait behind a send round queued to the same peer.
TEST_F(NetIbMsegGinMicrotest, IFlush_BehindAQueuedSend_PostsAtOnce) {
  GinRoundScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* wide = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &wide),
            ncclSuccess);
  ASSERT_NE(scene.sendQueues[1].head, nullptr);
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIFlush(&scene.context, 0, &dst.h, 1, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 65u);
  EXPECT_EQ(scene.verbs.posted.back().qp, scene.flushQp);
  EXPECT_EQ(scene.verbs.posted.back().remoteAddress, dst.bases[0]);
  auto* wideReq = static_cast<IbMsegRequest*>(wide);
  for (int r = 0; r < 3; r++) {
    scene.QueueCompletion(wideReq);
    EXPECT_EQ(scene.Test(wideReq), r == 2 ? 1 : 0) << "round " << r;
  }
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
}

// The RMA proxy's GIN listen marks the handle and the listen comm, so its classic comms skip the multi-segment
// layout. The GDAKI listen marks them too, but it is not built on AMD.
TEST_F(NetIbMsegGinMicrotest, GinListens_MarkHandleAndListenCommAsGin) {
  ScopedHook init(g_ncclSocketInit, [](ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType,
                                       volatile uint32_t*, int, int) { return ncclSuccess; });
  ScopedHook socketListen(g_ncclSocketListen, [](ncclSocket*) { return ncclSuccess; });
  ScopedHook address(g_ncclSocketGetAddr, [](ncclSocket*, ncclSocketAddress* out) {
    std::memset(out, 0, sizeof(*out));
    out->sa.sa_family = AF_INET;
    return ncclSuccess;
  });
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  ScopedHook mark(g_ginMarkListen, rcclIbMarkGinListen);
  unsigned char classicHandle[NCCL_NET_HANDLE_MAXSIZE] = {};
  unsigned char ginHandle[NCCL_NET_HANDLE_MAXSIZE] = {};
  void* classic = nullptr;
  void* gin = nullptr;

  ASSERT_EQ(IbMsegListen(nullptr, 0, classicHandle, &classic), ncclSuccess);
  ASSERT_EQ(IbMsegRmaIbProxy.listen(nullptr, 0, ginHandle, &gin), ncclSuccess);
  EXPECT_FALSE(static_cast<IbMsegListenComm*>(classic)->isGin);
  EXPECT_TRUE(static_cast<IbMsegListenComm*>(gin)->isGin);
  EXPECT_NE(std::memcmp(classicHandle, ginHandle, sizeof(ginHandle)), 0);
  EXPECT_EQ(IbMsegCloseListen(classic), ncclSuccess);
  EXPECT_EQ(IbMsegCloseListen(gin), ncclSuccess);
}

TEST_F(NetIbMsegGinMicrotest, CreateContext_MarksTheRmaListenCommAsGin) {
  GinCreateContextScene scene(/*nranks=*/1, /*rank=*/0);
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  IbMsegRmaIbProxyCtx* contexts = nullptr;
  std::vector<void*> marked;
  ScopedHook mark(g_ginMarkListen, [&marked](void*, void* listenComm) { marked.push_back(listenComm); });

  ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  EXPECT_EQ(marked, std::vector<void*>{&scene.listenToken});
  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
  g_hostAllocs.Free(contexts);
}

// Branch: multi-segment on swaps in every MR and data entry; connection, test and teardown stay.
TEST_F(NetIbMsegGinMicrotest, RmaVtable_MultiSegmentInstallReplacesMrAndDataOps) {
  ncclRma_t rma = IbMsegRmaIbProxy;
  rcclRmaIbProxyInstallMultiSeg(&rma);
  EXPECT_EQ(rma.regMrSym, rcclRmaIbProxyRegMrSym);
  EXPECT_EQ(rma.regMrSymDmaBuf, rcclRmaIbProxyRegMrSymDmaBuf);
  EXPECT_EQ(rma.deregMrSym, rcclRmaIbProxyDeregMrSym);
  EXPECT_EQ(rma.iput, rcclRmaIbProxyIPut);
  EXPECT_EQ(rma.iputSignal, rcclRmaIbProxyIPutSignal);
  EXPECT_EQ(rma.iget, rcclRmaIbProxyIGet);
  EXPECT_EQ(rma.iflush, rcclRmaIbProxyIFlush);
  EXPECT_EQ(rma.init, IbMsegRmaIbProxy.init);
  EXPECT_EQ(rma.connect, IbMsegRmaIbProxy.connect);
  EXPECT_EQ(rma.test, IbMsegRmaIbProxy.test);
  EXPECT_EQ(rma.closeColl, IbMsegRmaIbProxy.closeColl);
  EXPECT_EQ(IbMsegRmaIbProxy.iput, IbMsegRmaIbProxyIPut) << "the shared table is untouched";
}

}  // namespace
