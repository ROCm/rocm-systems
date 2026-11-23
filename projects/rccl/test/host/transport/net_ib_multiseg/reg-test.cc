/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/reg.cc: net_ib/reg-test.cc against the fork
 * (derived_transport_tests.cmake), then the fork's multi-segment tests.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>

#include "fakes/hip_fakes.h"

#undef NET_IB_REG_CC_PATH
#define NET_IB_REG_CC_PATH NET_IB_MSEG_REG_CC_PATH
#define NetIbRegMicrotest NetIbMsegRegMicrotest
#include "transport/derived_transport_tests_begin.h"
#include "transport/net_ib/reg-test.cc"
#include "transport/derived_transport_tests_end.h"

namespace {

// Two segments, each in its own pages; fd 11 and 12 name them.
struct TwoSegmentArgs {
  explicit TwoSegmentArgs(size_t page) : storage(page * 4) {
    segAddrs[0] = storage.data();
    segAddrs[1] = storage.data() + 2 * page;
    segLens[0] = segLens[1] = page;
  }
  std::vector<unsigned char> storage;
  void* segAddrs[2];
  size_t segLens[2];
  uint64_t segOffsets[2] = {0, 0};
  int segFds[2] = {11, 12};
};

TEST_F(NetIbMsegRegMicrotest, RegMrDmaBufMultiSeg_LaterFailureUnwindsEarlierMrsAndLeavesHandleUntouched) {
  TwoSegmentArgs args(static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.devs[0].base = MakeDevBase(0);
  comm.devs[1].base = MakeDevBase(1);
  ibv_mr mrs[2][2]{};
  ScopedHook reg(g_ibvRegDmabufMr, [&mrs](ibv_mr** out, ibv_pd* pd, uint64_t, size_t, uint64_t, int fd, int) {
    const int dev = pd == reinterpret_cast<ibv_pd*>(0x1000) ? 0 : 1;
    const int seg = fd == 11 ? 0 : 1;
    if (seg == 1 && dev == 1) return ncclRemoteError;
    *out = &mrs[seg][dev];
    return ncclSuccess;
  });
  std::vector<ibv_mr*> deregs;
  ScopedHook dereg(g_ibvDeregMr, [&deregs](ibv_mr* mr) {
    deregs.push_back(mr);
    return ncclSuccess;
  });
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, 2, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclRemoteError);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(reg.calls, 4);
  ASSERT_EQ(deregs.size(), 3u);
  for (ibv_mr* mr : {&mrs[0][0], &mrs[0][1], &mrs[1][0]}) {
    EXPECT_EQ(std::count(deregs.begin(), deregs.end(), mr), 1);
  }
  EXPECT_EQ(IbMsegDevs[0].mrCache.population, 0);
  EXPECT_EQ(IbMsegDevs[1].mrCache.population, 0);
}

// A resilient comm replays whole slots, so a multi-segment buffer declines on the send side too.
TEST_F(NetIbMsegRegMicrotest, RegMrDmaBufMultiSeg_ResilientCommDeclinesBeforeRegistering) {
  TwoSegmentArgs args(static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base = MakeDevBase(0);
  comm.base.resiliency = reinterpret_cast<IbMsegResiliency*>(0x1);
  ScopedHook reg(g_ibvRegDmabufMr, [](ibv_mr**, ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
    return ncclSuccess;
  });
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, 2, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclInvalidUsage);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(reg.calls, 0);
}

// One MR per segment must fit under every rail's max_mr, counting MRs this process already cached; exactly at the
// limit still admits.
TEST_F(NetIbMsegRegMicrotest, RegMrDmaBufMultiSeg_PastTheDeviceMaxMrDeclinesAndAtTheLimitAdmits) {
  TwoSegmentArgs args(static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.devs[0].base = MakeDevBase(0);
  comm.devs[1].base = MakeDevBase(1);
  IbMsegDevs[0].maxMr = 2;
  IbMsegDevs[1].maxMr = 2;
  IbMsegDevs[1].mrCache.population = 1;
  ibv_mr mrs[2][2]{};
  ScopedHook reg(g_ibvRegDmabufMr, [&mrs](ibv_mr** out, ibv_pd* pd, uint64_t, size_t, uint64_t, int fd, int) {
    *out = &mrs[fd == 11 ? 0 : 1][pd == reinterpret_cast<ibv_pd*>(0x1000) ? 0 : 1];
    return ncclSuccess;
  });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, 2, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclInvalidUsage);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(reg.calls, 0);

  IbMsegDevs[1].mrCache.population = 0;
  ASSERT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, 2, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclSuccess);
  EXPECT_EQ(reg.calls, 4);
  EXPECT_EQ(IbMsegDeregMr(&comm, handle), ncclSuccess);
}

TEST_F(NetIbMsegRegMicrotest, DeregMr_MultiSegmentReturnsFirstErrorAndStillDeregistersLaterSegments) {
  TwoSegmentArgs args(static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base = MakeDevBase(0);
  ibv_mr mrs[2]{};
  ScopedHook reg(g_ibvRegDmabufMr, [&mrs](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int fd, int) {
    *out = &mrs[fd == 11 ? 0 : 1];
    return ncclSuccess;
  });
  std::vector<ibv_mr*> deregs;
  ScopedHook dereg(g_ibvDeregMr, [&deregs, &mrs](ibv_mr* mr) {
    deregs.push_back(mr);
    return mr == &mrs[0] ? ncclSystemError : ncclSuccess;
  });
  void* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, 2, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclSuccess);
  ASSERT_EQ(static_cast<IbMsegMrHandle*>(handle)->seg.nSegments, 2);

  EXPECT_EQ(IbMsegDeregMr(&comm, handle), ncclSystemError);
  EXPECT_EQ(deregs, (std::vector<ibv_mr*>{&mrs[0], &mrs[1]}));
  EXPECT_EQ(IbMsegDevs[0].mrCache.population, 0);
}

// Past the inline CTS limit, each segment in its own pages; fd 11 + s names segment s.
struct WideSegmentArgs {
  static constexpr int kSegs = RCCL_IB_CTS_INLINE_SEGS + 1;
  explicit WideSegmentArgs(size_t page) : storage(page * 2 * kSegs) {
    for (int s = 0; s < kSegs; s++) {
      segAddrs[s] = storage.data() + 2 * s * page;
      segLens[s] = page;
      segOffsets[s] = 0;
      segFds[s] = 11 + s;
    }
  }
  std::vector<unsigned char> storage;
  void* segAddrs[kSegs];
  size_t segLens[kSegs];
  uint64_t segOffsets[kSegs];
  int segFds[kSegs];
};

// A receive registration past the inline limit publishes {va0, offsets, rkeys} in a remote-readable MR.
TEST_F(NetIbMsegRegMicrotest, RegMrDmaBufMultiSeg_RecvPastTheInlineLimitPublishesItsLayout) {
  WideSegmentArgs args(static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  constexpr int kSegs = WideSegmentArgs::kSegs;
  IbMsegRecvComm comm{};
  comm.base.isSend = false;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base = MakeDevBase(0);
  comm.useMultiSeg = 1;
  ibv_mr segMrs[kSegs]{};
  ScopedHook regSeg(g_ibvRegDmabufMr, [&segMrs](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int fd, int) {
    segMrs[fd - 11].rkey = 0x70 + (fd - 11);
    *out = &segMrs[fd - 11];
    return ncclSuccess;
  });
  ibv_mr layoutMr{};
  layoutMr.rkey = 0x5a;
  void* published = nullptr;
  ScopedHook regLayout(g_ibvRegMr, [&](ibv_mr** out, ibv_pd* pd, void* addr, size_t length, int flags) {
    EXPECT_EQ(pd, reinterpret_cast<ibv_pd*>(0x1000));
    EXPECT_EQ(length, rcclIbLayoutBlobBytes(kSegs, 1));
    EXPECT_EQ(flags, IBV_ACCESS_REMOTE_READ);
    published = addr;
    *out = &layoutMr;
    return ncclSuccess;
  });
  std::vector<ibv_mr*> deregs;
  ScopedHook dereg(g_ibvDeregMr, [&deregs](ibv_mr* mr) {
    deregs.push_back(mr);
    return ncclSuccess;
  });
  void* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, kSegs, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclSuccess);
  auto* h = static_cast<IbMsegMrHandle*>(handle);
  ASSERT_NE(h->seg.layoutBlob, nullptr);
  EXPECT_EQ(h->seg.layoutBlob, published);
  EXPECT_EQ(h->seg.layoutMrs[0], &layoutMr);
  EXPECT_NE(h->seg.layoutGen, 0u);
  EXPECT_EQ(static_cast<uint64_t*>(h->seg.layoutBlob)[0], reinterpret_cast<uint64_t>(args.segAddrs[0]));
  for (int s = 0; s <= kSegs; s++) EXPECT_EQ(rcclIbLayoutBlobOff(h->seg.layoutBlob)[s], h->seg.segOff[s]);
  for (int s = 0; s < kSegs; s++) EXPECT_EQ(rcclIbLayoutBlobRkeys(h->seg.layoutBlob, kSegs, 0)[s], 0x70u + s);

  EXPECT_EQ(IbMsegDeregMr(&comm, handle), ncclSuccess);
  EXPECT_EQ(deregs.size(), static_cast<size_t>(kSegs + 1));
  EXPECT_EQ(std::count(deregs.begin(), deregs.end(), &layoutMr), 1);
}

TEST_F(NetIbMsegRegMicrotest, RegMrDmaBufMultiSeg_LayoutRegistrationFailureUnwindsEverySegment) {
  WideSegmentArgs args(static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  constexpr int kSegs = WideSegmentArgs::kSegs;
  IbMsegRecvComm comm{};
  comm.base.isSend = false;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base = MakeDevBase(0);
  comm.useMultiSeg = 1;
  ibv_mr segMrs[kSegs]{};
  ScopedHook regSeg(g_ibvRegDmabufMr, [&segMrs](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int fd, int) {
    *out = &segMrs[fd - 11];
    return ncclSuccess;
  });
  ScopedHook regLayout(g_ibvRegMr, [](ibv_mr**, ibv_pd*, void*, size_t, int) { return ncclRemoteError; });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, kSegs, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclRemoteError);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(dereg.calls, kSegs);
  EXPECT_EQ(IbMsegDevs[0].mrCache.population, 0);
}

// Within the inline limit nothing is published.
TEST_F(NetIbMsegRegMicrotest, RegMrDmaBufMultiSeg_RecvWithinTheInlineLimitPublishesNothing) {
  TwoSegmentArgs args(static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  IbMsegRecvComm comm{};
  comm.base.isSend = false;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base = MakeDevBase(0);
  comm.useMultiSeg = 1;
  ibv_mr mrs[2]{};
  ScopedHook reg(g_ibvRegDmabufMr, [&mrs](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int fd, int) {
    *out = &mrs[fd == 11 ? 0 : 1];
    return ncclSuccess;
  });
  ScopedHook regLayout(g_ibvRegMr, [](ibv_mr**, ibv_pd*, void*, size_t, int) { return ncclSuccess; });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  void* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufMultiSeg(&comm, 2, args.segAddrs, args.segLens, args.segOffsets, args.segFds,
                                      NCCL_PTR_CUDA, &handle),
            ncclSuccess);
  EXPECT_EQ(static_cast<IbMsegMrHandle*>(handle)->seg.layoutBlob, nullptr);
  EXPECT_EQ(regLayout.calls, 0);
  EXPECT_EQ(IbMsegDeregMr(&comm, handle), ncclSuccess);
}

#if NCCL_CUMEM_DMABUF_EXPORT_GATE
// A fake VMM range of 64 KiB segments; each export hands out a real /dev/null fd.
constexpr uintptr_t kRangeBase = 0x10000000;
constexpr size_t kRangeSegment = 0x10000;

class NetIbMsegRegRangeMicrotest : public NetIbMsegRegMicrotest {
 protected:
  void SetUp() override {
    NetIbMsegRegMicrotest::SetUp();
    comm_.base.isSend = true;
    comm_.base.vProps.ndevs = 1;
    comm_.devs[0].base = MakeDevBase(0);
  }

  // Every fd the walk exported must be closed again.
  void ExpectEveryFdClosed() {
    for (int fd : fds_) {
      const bool open = fcntl(fd, F_GETFD) != -1;
      EXPECT_FALSE(open) << "fd " << fd;
      if (open) close(fd);
    }
  }

  IbMsegSendComm comm_{};
  std::vector<int> fds_;
  ScopedHook<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> range_{
      g_hipMemGetAddressRange, [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
        *base = reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(ptr) & ~(kRangeSegment - 1));
        *size = kRangeSegment;
        return hipSuccess;
      }};
  ScopedHook<hipError_t(void*, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long)> export_{
      g_hipMemGetHandleForAddressRange,
      [this](void* handle, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long) {
        const int fd = open("/dev/null", O_RDONLY);
        *static_cast<int*>(handle) = fd;
        if (fd != -1) fds_.push_back(fd);
        return fd == -1 ? hipErrorInvalidValue : hipSuccess;
      }};
};

TEST_F(NetIbMsegRegRangeMicrotest, RegMrDmaBuf_RangeRegistrationFailureClosesEveryFdAndLeavesHandleUntouched) {
  ibv_mr firstMr{};
  ScopedHook reg(g_ibvRegDmabufMr, [this, &firstMr](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int fd, int) {
    if (fd != fds_[0]) return ncclRemoteError;
    *out = &firstMr;
    return ncclSuccess;
  });
  ScopedHook dereg(g_ibvDeregMr, [&firstMr](ibv_mr* mr) {
    EXPECT_EQ(mr, &firstMr);
    return ncclSuccess;
  });
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBuf(&comm_, reinterpret_cast<void*>(kRangeBase), 2 * kRangeSegment, NCCL_PTR_CUDA,
                              /*flags=*/0, /*fd=*/-1, &handle),
            ncclRemoteError);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(reg.calls, 2);
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(fds_.size(), 2u);
  ExpectEveryFdClosed();
}

// The handle's tables come from the walk: an unaligned three-segment range gives three starts and four offsets.
TEST_F(NetIbMsegRegRangeMicrotest, RegMrDmaBuf_ThreeSegmentRangeSizesTheHandleFromTheWalk) {
  std::array<ibv_mr, 3> mrs{};
  int nReg = 0;
  ScopedHook reg(g_ibvRegDmabufMr, [&](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
    *out = &mrs[nReg++];
    return ncclSuccess;
  });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  constexpr size_t kHead = 0x100;
  void* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBuf(&comm_, reinterpret_cast<void*>(kRangeBase + kHead), 3 * kRangeSegment - 2 * kHead,
                              NCCL_PTR_CUDA, /*flags=*/0, /*fd=*/-1, &handle),
            ncclSuccess);
  auto* h = static_cast<IbMsegMrHandle*>(handle);
  ASSERT_EQ(h->seg.nSegments, 3);
  const uint64_t want[] = {0, kRangeSegment - kHead, 2 * kRangeSegment - kHead, 3 * kRangeSegment - 2 * kHead};
  for (int i = 0; i < 4; i++) EXPECT_EQ(h->seg.segOff[i], want[i]) << "offset " << i;
  EXPECT_EQ(h->seg.segStart[2], kRangeBase + 2 * kRangeSegment);
  EXPECT_EQ(h->seg.segLen[2], kRangeSegment - kHead);
  EXPECT_EQ(rcclIbSegMrAt(&h->seg, 2, 0), &mrs[2]);
  EXPECT_EQ(rcclIbSegMrAt(&h->seg, 2, 1), nullptr) << "slots past ndevs stay empty";
  EXPECT_EQ(h->mrs[0], &mrs[0]);
  EXPECT_EQ(IbMsegDeregMr(&comm_, handle), ncclSuccess);
  EXPECT_EQ(dereg.calls, 3);
  ExpectEveryFdClosed();
}

// A host VMM range reports base 0, so each segment runs segSize bytes from where the walk stands.
TEST_F(NetIbMsegRegRangeMicrotest, RegMrDmaBuf_BaseZeroRangeStepsBySegmentSize) {
  ScopedHook<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> hostRange(
      g_hipMemGetAddressRange, [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t) {
        *base = hipDeviceptr_t{};
        *size = kRangeSegment;
        return hipSuccess;
      });
  std::array<ibv_mr, 2> mrs{};
  int nReg = 0;
  ScopedHook reg(g_ibvRegDmabufMr, [&](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
    *out = &mrs[nReg++];
    return ncclSuccess;
  });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  void* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBuf(&comm_, reinterpret_cast<void*>(kRangeBase + 0x100), 2 * kRangeSegment, NCCL_PTR_CUDA,
                              /*flags=*/0, /*fd=*/-1, &handle),
            ncclSuccess);
  auto* h = static_cast<IbMsegMrHandle*>(handle);
  ASSERT_EQ(h->seg.nSegments, 2);
  EXPECT_EQ(h->seg.segLen[0], kRangeSegment);
  EXPECT_EQ(h->seg.segLen[1], kRangeSegment);
  EXPECT_EQ(fds_.size(), 2u);
  EXPECT_EQ(IbMsegDeregMr(&comm_, handle), ncclSuccess);
  ExpectEveryFdClosed();
}

// The count stops one past the cap, and the decline comes before any export.
TEST_F(NetIbMsegRegRangeMicrotest, RegMrDmaBuf_RangePastTheSegmentCapDeclinesBeforeExport) {
  ScopedHook reg(g_ibvRegDmabufMr, [](ibv_mr**, ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
    return ncclSuccess;
  });
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBuf(&comm_, reinterpret_cast<void*>(kRangeBase),
                              (RCCL_IB_REG_SEG_DEFAULT + 4) * kRangeSegment, NCCL_PTR_CUDA, /*flags=*/0, /*fd=*/-1,
                              &handle),
            ncclInvalidUsage);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(range_.calls, RCCL_IB_REG_SEG_DEFAULT + 1);
  EXPECT_EQ(reg.calls, 0);
  EXPECT_TRUE(fds_.empty());
}

// The comm-level declines also come before any export.
TEST_F(NetIbMsegRegRangeMicrotest, RegMrDmaBuf_ResilientCommDeclinesARangeBeforeExport) {
  comm_.base.resiliency = reinterpret_cast<IbMsegResiliency*>(0x1);
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBuf(&comm_, reinterpret_cast<void*>(kRangeBase), 2 * kRangeSegment, NCCL_PTR_CUDA,
                              /*flags=*/0, /*fd=*/-1, &handle),
            ncclInvalidUsage);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_TRUE(fds_.empty());
  comm_.base.resiliency = nullptr;
}
#endif

}  // namespace
