/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/reg.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "transport/net_ib_multiseg/NetIbMsegGlobals.h"

#include NET_IB_MSEG_REG_CC_PATH

namespace {

class NetIbMsegRegMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    IbMsegRelaxedOrderingEnabled = 0;
    for (int dev = 0; dev < 2; ++dev) {
      IbMsegDevs[dev].mrCache = {};
      IbMsegDevs[dev].capsProvider.mlx5.dataDirect = 0;
      IbMsegDevs[dev].maxMr = 0;
    }
  }

  void TearDown() override {
    NetIbMsegResetGlobals();
    ResetMlx5dvwrapFakes();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }

  IbMsegNetCommDevBase MakeDevBase(int ibDev = 0) {
    IbMsegNetCommDevBase base{};
    base.ibDevN = ibDev;
    base.pd = reinterpret_cast<ibv_pd*>(0x1000 + ibDev);
    return base;
  }
};

TEST_F(NetIbMsegRegMicrotest, RegMrDmaBuf_ZeroSizeIsRejectedWithoutPublishingHandle) {
  IbMsegSendComm comm{};
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMrDmaBufInternal(&comm, nullptr, 0, NCCL_PTR_HOST, 0, -1, 0, &handle),
            ncclInternalError);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
}

TEST_F(NetIbMsegRegMicrotest, DeregMr_NullHandleIsNoOp) {
  EXPECT_EQ(IbMsegDeregMr(nullptr, nullptr), ncclSuccess);
}

TEST_F(NetIbMsegRegMicrotest, RegMrInternal_FirstHostRangeRegistersAlignedPagesAndCachesMr) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 2);
  auto base = MakeDevBase();
  ibv_mr mr{};
  mr.lkey = 17;
  mr.rkey = 18;
  ScopedHook reg(g_ibvRegMr, [&mr, page](ibv_mr** out, ibv_pd* pd, void* addr, size_t length, int flags) {
    EXPECT_EQ(pd, reinterpret_cast<ibv_pd*>(0x1000));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(addr) % page, 0u);
    EXPECT_EQ(length, 2 * page);
    EXPECT_EQ(flags, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC);
    *out = &mr;
    return ncclSuccess;
  });
  ibv_mr* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.data() + 3, page, NCCL_PTR_HOST, 0, -1, 0, &handle),
            ncclSuccess);

  EXPECT_EQ(handle, &mr);
  ASSERT_EQ(IbMsegDevs[0].mrCache.population, 1);
  EXPECT_EQ(IbMsegDevs[0].mrCache.slots[0].mr, &mr);
  EXPECT_EQ(IbMsegDevs[0].mrCache.slots[0].pages, 2u);
  EXPECT_EQ(IbMsegDevs[0].mrCache.slots[0].refs, 1);
  EXPECT_EQ(reg.calls, 1);
}

TEST_F(NetIbMsegRegMicrotest, RegMrInternal_RegistrationFailureLeavesCacheEmpty) {
  std::array<unsigned char, 16> data{};
  auto base = MakeDevBase();
  ScopedHook reg(g_ibvRegMr, [](ibv_mr**, ibv_pd*, void*, size_t, int) {
    return ncclRemoteError;
  });
  ibv_mr* handle = nullptr;

  EXPECT_EQ(IbMsegRegMrDmaBufInternal2(&base, data.data(), data.size(), NCCL_PTR_HOST, 0, -1, 0, &handle),
            ncclRemoteError);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(IbMsegDevs[0].mrCache.population, 0);
}

TEST_F(NetIbMsegRegMicrotest, RegMrInternal_RangeEndingOnCachedPageReusesMrAndOnePastRegistersAnother) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  // Page-aligned, so the first registration caches exactly one page.
  std::unique_ptr<unsigned char, decltype(&std::free)> storage(
      static_cast<unsigned char*>(std::aligned_alloc(page, 2 * page)), &std::free);
  ASSERT_NE(storage, nullptr);
  auto base = MakeDevBase();
  ibv_mr cachedMr{};
  ibv_mr nextMr{};
  int registrations = 0;
  ScopedHook reg(g_ibvRegMr, [&](ibv_mr** out, ibv_pd*, void*, size_t, int) {
    *out = registrations++ == 0 ? &cachedMr : &nextMr;
    return ncclSuccess;
  });
  ibv_mr* first = nullptr;
  ibv_mr* lastByte = nullptr;
  ibv_mr* pastEnd = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.get(), page, NCCL_PTR_HOST, 0, -1, 0, &first), ncclSuccess);
  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.get() + page - 32, 32, NCCL_PTR_HOST, 0, -1, 0, &lastByte),
            ncclSuccess);

  auto& cache = IbMsegDevs[0].mrCache;
  EXPECT_EQ(first, &cachedMr);
  EXPECT_EQ(lastByte, &cachedMr);
  ASSERT_EQ(cache.population, 1);
  EXPECT_EQ(cache.slots[0].pages, 1u);
  EXPECT_EQ(cache.slots[0].refs, 2);
  EXPECT_EQ(reg.calls, 1);

  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.get() + page - 32, 33, NCCL_PTR_HOST, 0, -1, 0, &pastEnd),
            ncclSuccess);
  EXPECT_EQ(pastEnd, &nextMr);
  EXPECT_EQ(cache.population, 2);
  EXPECT_EQ(cache.slots[0].refs, 2);
  EXPECT_EQ(reg.calls, 2);
}

TEST_F(NetIbMsegRegMicrotest, RegMrInternal_LowerAddressInsertsBeforeExistingCacheEntry) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 3);
  const uintptr_t lowAddr = reinterpret_cast<uintptr_t>(storage.data()) & -page;
  auto base = MakeDevBase();
  ibv_mr oldMr{};
  ibv_mr newMr{};
  auto& cache = IbMsegDevs[0].mrCache;
  cache.capacity = 32;
  cache.population = 1;
  cache.slots = static_cast<IbMsegMr*>(std::calloc(cache.capacity, sizeof(IbMsegMr)));
  ASSERT_NE(cache.slots, nullptr);
  cache.slots[0].addr = lowAddr + 2 * page;
  cache.slots[0].pages = 1;
  cache.slots[0].refs = 1;
  cache.slots[0].mr = &oldMr;
  ScopedHook reg(g_ibvRegMr, [&newMr](ibv_mr** out, ibv_pd*, void*, size_t, int) {
    *out = &newMr;
    return ncclSuccess;
  });
  ibv_mr* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, reinterpret_cast<void*>(lowAddr), 1, NCCL_PTR_HOST, 0, -1, 0,
                                       &handle),
            ncclSuccess);

  ASSERT_EQ(cache.population, 2);
  EXPECT_EQ(cache.slots[0].mr, &newMr);
  EXPECT_EQ(cache.slots[1].mr, &oldMr);
}

TEST_F(NetIbMsegRegMicrotest, RegMrInternal_RelaxedOrderingUsesIova2UnlessForceSoRequested) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 3);
  auto base = MakeDevBase();
  ibv_mr relaxedMr{};
  ibv_mr strongMr{};
  IbMsegRelaxedOrderingEnabled = 1;
  ScopedHook iova(g_ibvRegMrIova2,
                  [&relaxedMr](ibv_mr** out, ibv_pd*, void*, size_t, uint64_t, int flags) {
                    EXPECT_NE(flags & IBV_ACCESS_RELAXED_ORDERING, 0);
                    *out = &relaxedMr;
                    return ncclSuccess;
                  });
  ScopedHook regular(g_ibvRegMr, [&strongMr](ibv_mr** out, ibv_pd*, void*, size_t, int flags) {
    EXPECT_EQ(flags & IBV_ACCESS_RELAXED_ORDERING, 0);
    *out = &strongMr;
    return ncclSuccess;
  });
  ibv_mr* first = nullptr;
  ibv_mr* second = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.data(), 1, NCCL_PTR_HOST, 0, -1, 0, &first),
            ncclSuccess);
  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.data() + 2 * page, 1, NCCL_PTR_HOST, 0, -1,
                                       NCCL_NET_MR_FLAG_FORCE_SO, &second),
            ncclSuccess);

  EXPECT_EQ(first, &relaxedMr);
  EXPECT_EQ(second, &strongMr);
  EXPECT_EQ(iova.calls, 1);
  EXPECT_EQ(regular.calls, 1);
}

TEST_F(NetIbMsegRegMicrotest, RegMrInternal_DmaBufSelectsProviderSpecificRegistration) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 3);
  auto base = MakeDevBase();
  ibv_mr verbsMr{};
  ibv_mr directMr{};
  // reg.cc passes data-direct into wrap_reg_dmabuf_mr. The mlx5-then-ibv choice inside
  // that wrapper is covered by Mlx5dvwrapRegDmabufMicrotest.
  ScopedHook dmabuf(g_regDmabufMr, [&](ibv_mr** out, ibv_pd*, uint64_t offset, size_t, uint64_t, int fd, int,
                                       bool dataDirect) {
    if (dataDirect) {
      EXPECT_EQ(offset, 8u);
      EXPECT_EQ(fd, 12);
      *out = &directMr;
    } else {
      EXPECT_EQ(offset, 7u);
      EXPECT_EQ(fd, 11);
      *out = &verbsMr;
    }
    return ncclSuccess;
  });
  ibv_mr* first = nullptr;
  ibv_mr* second = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.data(), 1, NCCL_PTR_CUDA, 7, 11, 0, &first),
            ncclSuccess);
  IbMsegDevs[0].capsProvider.mlx5.dataDirect = 1;
  ASSERT_EQ(IbMsegRegMrDmaBufInternal2(&base, storage.data() + 2 * page, 1, NCCL_PTR_CUDA, 8, 12, 0, &second),
            ncclSuccess);

  EXPECT_EQ(first, &verbsMr);
  EXPECT_EQ(second, &directMr);
  EXPECT_EQ(dmabuf.calls, 2);
}

TEST_F(NetIbMsegRegMicrotest, RegAndDeregMr_MultiDeviceCommUsesEveryDevice) {
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.devs[0].base = MakeDevBase(0);
  comm.devs[1].base = MakeDevBase(1);
  ibv_mr mrs[2]{};
  ScopedHook reg(g_ibvRegMr, [&mrs](ibv_mr** out, ibv_pd* pd, void*, size_t, int) {
    const int index = pd == reinterpret_cast<ibv_pd*>(0x1000) ? 0 : 1;
    *out = &mrs[index];
    return ncclSuccess;
  });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  std::array<unsigned char, 64> data{};
  void* handle = nullptr;

  ASSERT_EQ(IbMsegRegMr(&comm, data.data(), data.size(), NCCL_PTR_HOST, &handle), ncclSuccess);
  auto* wrapper = static_cast<IbMsegMrHandle*>(handle);
  ASSERT_NE(wrapper, nullptr);
  EXPECT_EQ(wrapper->mrs[0], &mrs[0]);
  EXPECT_EQ(wrapper->mrs[1], &mrs[1]);
  EXPECT_EQ(reg.calls, 2);

  EXPECT_EQ(IbMsegDeregMr(&comm, handle), ncclSuccess);
  EXPECT_EQ(dereg.calls, 2);
}

TEST_F(NetIbMsegRegMicrotest, RegMr_MultiDeviceFailureReturnsErrorWithoutPublishingWrapper) {
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.devs[0].base = MakeDevBase(0);
  comm.devs[1].base = MakeDevBase(1);
  ibv_mr firstMr{};
  ScopedHook reg(g_ibvRegMr, [&firstMr](ibv_mr** out, ibv_pd* pd, void*, size_t, int) {
    if (pd == reinterpret_cast<ibv_pd*>(0x1000)) {
      *out = &firstMr;
      return ncclSuccess;
    }
    return ncclRemoteError;
  });
  std::array<unsigned char, 64> data{};
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMr(&comm, data.data(), data.size(), NCCL_PTR_HOST, &handle), ncclRemoteError);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(reg.calls, 2);
}

TEST_F(NetIbMsegRegMicrotest, DISABLED_RegMr_MultiDeviceFailureUnwindsEarlierRegistrations) {
  // Production leaves the first device's MR cached when the second device
  // registration fails. Enable with the fix tracked by AICOMRCCL-3112.
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.devs[0].base = MakeDevBase(0);
  comm.devs[1].base = MakeDevBase(1);
  ibv_mr firstMr{};
  ScopedHook reg(g_ibvRegMr, [&firstMr](ibv_mr** out, ibv_pd* pd, void*, size_t, int) {
    if (pd == reinterpret_cast<ibv_pd*>(0x1000)) {
      *out = &firstMr;
      return ncclSuccess;
    }
    return ncclRemoteError;
  });
  ScopedHook dereg(g_ibvDeregMr, [&firstMr](ibv_mr* value) {
    EXPECT_EQ(value, &firstMr);
    return ncclSuccess;
  });
  std::array<unsigned char, 64> data{};
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegRegMr(&comm, data.data(), data.size(), NCCL_PTR_HOST, &handle), ncclRemoteError);
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(IbMsegDevs[0].mrCache.population, 0);
  EXPECT_EQ(IbMsegDevs[0].mrCache.slots, nullptr);
}

TEST_F(NetIbMsegRegMicrotest, RegMrDmaBuf_PublicWrapperForwardsOffsetAndFd) {
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base = MakeDevBase(0);
  ibv_mr mr{};
  ScopedHook reg(g_ibvRegDmabufMr,
                 [&mr](ibv_mr** out, ibv_pd*, uint64_t offset, size_t, uint64_t, int fd, int) {
                   EXPECT_EQ(offset, 17u);
                   EXPECT_EQ(fd, 23);
                   *out = &mr;
                   return ncclSuccess;
                 });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  std::array<unsigned char, 64> data{};
  void* handle = nullptr;

  ASSERT_EQ(IbMsegRegMrDmaBuf(&comm, data.data(), data.size(), NCCL_PTR_CUDA, 17, 23, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(reg.calls, 1);
  EXPECT_EQ(IbMsegDeregMr(&comm, handle), ncclSuccess);
}

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

TEST_F(NetIbMsegRegMicrotest, DeregMrInternal_ReferenceThenFinalReleaseUpdatesCacheAndDeregisters) {
  auto base = MakeDevBase();
  ibv_mr mr{};
  auto& cache = IbMsegDevs[0].mrCache;
  cache.capacity = 32;
  cache.population = 1;
  cache.slots = static_cast<IbMsegMr*>(std::calloc(cache.capacity, sizeof(IbMsegMr)));
  ASSERT_NE(cache.slots, nullptr);
  cache.slots[0].mr = &mr;
  cache.slots[0].refs = 2;
  ScopedHook dereg(g_ibvDeregMr, [&mr](ibv_mr* value) {
    EXPECT_EQ(value, &mr);
    return ncclSuccess;
  });

  EXPECT_EQ(IbMsegDeregMrInternal(&base, &mr), ncclSuccess);
  EXPECT_EQ(cache.population, 1);
  EXPECT_EQ(cache.slots[0].refs, 1);
  EXPECT_EQ(dereg.calls, 0);
  EXPECT_EQ(IbMsegDeregMrInternal(&base, &mr), ncclSuccess);
  EXPECT_EQ(cache.population, 0);
  EXPECT_EQ(cache.slots, nullptr);
  EXPECT_EQ(cache.capacity, 0);
  EXPECT_EQ(dereg.calls, 1);
}

TEST_F(NetIbMsegRegMicrotest, DeregMrInternal_FinalReleaseOfMiddleEntryMovesLastEntryIntoItsSlot) {
  auto base = MakeDevBase();
  ibv_mr mrs[3]{};
  auto& cache = IbMsegDevs[0].mrCache;
  cache.capacity = 32;
  cache.population = 3;
  cache.slots = static_cast<IbMsegMr*>(std::calloc(cache.capacity, sizeof(IbMsegMr)));
  ASSERT_NE(cache.slots, nullptr);
  for (int i = 0; i < 3; ++i) {
    cache.slots[i].mr = &mrs[i];
    cache.slots[i].refs = 1;
  }
  ScopedHook dereg(g_ibvDeregMr, [&mrs](ibv_mr* value) {
    EXPECT_EQ(value, &mrs[1]);
    return ncclSuccess;
  });

  EXPECT_EQ(IbMsegDeregMrInternal(&base, &mrs[1]), ncclSuccess);

  ASSERT_EQ(cache.population, 2);
  EXPECT_EQ(cache.slots[0].mr, &mrs[0]);
  EXPECT_EQ(cache.slots[1].mr, &mrs[2]);
  EXPECT_EQ(dereg.calls, 1);
}

TEST_F(NetIbMsegRegMicrotest, DeregMrInternal_UnknownMrReturnsInternalError) {
  auto base = MakeDevBase();
  ibv_mr mr{};

  EXPECT_EQ(IbMsegDeregMrInternal(&base, &mr), ncclInternalError);
}

}  // namespace
