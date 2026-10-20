/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/reg.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "fakes/nccl_fakes.h"

#include NET_IB_REG_CC_PATH

namespace {

class NetIbRegMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclIbRelaxedOrderingEnabled = 0;
    for (int dev = 0; dev < 2; ++dev) {
      ncclIbDevs[dev].mrCache = {};
      ncclIbDevs[dev].capsProvider.mlx5.dataDirect = 0;
    }
  }

  void TearDown() override {
    for (int dev = 0; dev < 2; ++dev) {
      std::free(ncclIbDevs[dev].mrCache.slots);
      ncclIbDevs[dev].mrCache = {};
    }
    ncclIbRelaxedOrderingEnabled = 0;
    ResetMlx5dvwrapFakes();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }

  ncclIbNetCommDevBase MakeDevBase(int ibDev = 0) {
    ncclIbNetCommDevBase base{};
    base.ibDevN = ibDev;
    base.pd = reinterpret_cast<ibv_pd*>(0x1000 + ibDev);
    return base;
  }
};

TEST_F(NetIbRegMicrotest, RegMrDmaBuf_ZeroSizeIsRejectedWithoutPublishingHandle) {
  ncclIbSendComm comm{};
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclIbRegMrDmaBufInternal(&comm, nullptr, 0, NCCL_PTR_HOST, 0, -1, 0, &handle),
            ncclInternalError);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
}

TEST_F(NetIbRegMicrotest, DeregMr_NullHandleIsNoOp) {
  EXPECT_EQ(ncclIbDeregMr(nullptr, nullptr), ncclSuccess);
}

TEST_F(NetIbRegMicrotest, RegMrInternal_FirstHostRangeRegistersAlignedPagesAndCachesMr) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 2);
  auto base = MakeDevBase();
  ibv_mr mr{};
  mr.lkey = 17;
  mr.rkey = 18;
  ScopedHook reg(g_ibvRegMr, [&mr, page](ibv_mr** out, ibv_pd* pd, void* addr, size_t length, int flags) {
    EXPECT_EQ(pd, reinterpret_cast<ibv_pd*>(0x1000));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(addr) % page, 0u);
    EXPECT_GE(length, page);
    EXPECT_EQ(flags, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC);
    *out = &mr;
    return ncclSuccess;
  });
  ibv_mr* handle = nullptr;

  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, storage.data() + 3, page, NCCL_PTR_HOST, 0, -1, 0, &handle),
            ncclSuccess);

  EXPECT_EQ(handle, &mr);
  ASSERT_EQ(ncclIbDevs[0].mrCache.population, 1);
  EXPECT_EQ(ncclIbDevs[0].mrCache.slots[0].mr, &mr);
  EXPECT_EQ(ncclIbDevs[0].mrCache.slots[0].refs, 1);
  EXPECT_EQ(reg.calls, 1);
}

TEST_F(NetIbRegMicrotest, RegMrInternal_RegistrationFailureLeavesCacheEmpty) {
  std::array<unsigned char, 16> data{};
  auto base = MakeDevBase();
  ScopedHook reg(g_ibvRegMr, [](ibv_mr**, ibv_pd*, void*, size_t, int) {
    return ncclRemoteError;
  });
  ibv_mr* handle = nullptr;

  EXPECT_EQ(ncclIbRegMrDmaBufInternal2(&base, data.data(), data.size(), NCCL_PTR_HOST, 0, -1, 0, &handle),
            ncclRemoteError);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(ncclIbDevs[0].mrCache.population, 0);
}

TEST_F(NetIbRegMicrotest, RegMrInternal_ContainedRangeReusesCachedMrAndIncrementsRefs) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 2);
  auto base = MakeDevBase();
  ibv_mr mr{};
  ScopedHook reg(g_ibvRegMr, [&mr](ibv_mr** out, ibv_pd*, void*, size_t, int) {
    *out = &mr;
    return ncclSuccess;
  });
  ibv_mr* first = nullptr;
  ibv_mr* second = nullptr;

  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, storage.data(), page, NCCL_PTR_HOST, 0, -1, 0, &first),
            ncclSuccess);
  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, storage.data() + 16, 32, NCCL_PTR_HOST, 0, -1, 0, &second),
            ncclSuccess);

  EXPECT_EQ(first, &mr);
  EXPECT_EQ(second, &mr);
  EXPECT_EQ(ncclIbDevs[0].mrCache.population, 1);
  EXPECT_EQ(ncclIbDevs[0].mrCache.slots[0].refs, 2);
  EXPECT_EQ(reg.calls, 1);
}

TEST_F(NetIbRegMicrotest, RegMrInternal_LowerAddressInsertsBeforeExistingCacheEntry) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 3);
  const uintptr_t lowAddr = reinterpret_cast<uintptr_t>(storage.data()) & -page;
  auto base = MakeDevBase();
  ibv_mr oldMr{};
  ibv_mr newMr{};
  auto& cache = ncclIbDevs[0].mrCache;
  cache.capacity = 32;
  cache.population = 1;
  cache.slots = static_cast<ncclIbMr*>(std::calloc(cache.capacity, sizeof(ncclIbMr)));
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

  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, reinterpret_cast<void*>(lowAddr), 1, NCCL_PTR_HOST, 0, -1, 0,
                                       &handle),
            ncclSuccess);

  ASSERT_EQ(cache.population, 2);
  EXPECT_EQ(cache.slots[0].mr, &newMr);
  EXPECT_EQ(cache.slots[1].mr, &oldMr);
}

TEST_F(NetIbRegMicrotest, RegMrInternal_RelaxedOrderingUsesIova2UnlessForceSoRequested) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 3);
  auto base = MakeDevBase();
  ibv_mr relaxedMr{};
  ibv_mr strongMr{};
  ncclIbRelaxedOrderingEnabled = 1;
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

  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, storage.data(), 1, NCCL_PTR_HOST, 0, -1, 0, &first),
            ncclSuccess);
  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, storage.data() + 2 * page, 1, NCCL_PTR_HOST, 0, -1,
                                       NCCL_NET_MR_FLAG_FORCE_SO, &second),
            ncclSuccess);

  EXPECT_EQ(first, &relaxedMr);
  EXPECT_EQ(second, &strongMr);
  EXPECT_EQ(iova.calls, 1);
  EXPECT_EQ(regular.calls, 1);
}

TEST_F(NetIbRegMicrotest, RegMrInternal_DmaBufSelectsProviderSpecificRegistration) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> storage(page * 3);
  auto base = MakeDevBase();
  ibv_mr verbsMr{};
  ibv_mr mlx5Mr{};
  ScopedHook verbs(g_ibvRegDmaBufMr,
                   [&verbsMr](ibv_mr** out, ibv_pd*, uint64_t offset, size_t, uint64_t, int fd, int) {
                     EXPECT_EQ(offset, 7u);
                     EXPECT_EQ(fd, 11);
                     *out = &verbsMr;
                     return ncclSuccess;
                   });
  ScopedHook mlx5(g_mlx5dvRegDmaBufMr,
                  [&mlx5Mr](ibv_mr** out, ibv_pd*, uint64_t, size_t, uint64_t, int, int, int mlx5Access) {
                    EXPECT_EQ(mlx5Access, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
                    *out = &mlx5Mr;
                    return ncclSuccess;
                  });
  ibv_mr* first = nullptr;
  ibv_mr* second = nullptr;

  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, storage.data(), 1, NCCL_PTR_CUDA, 7, 11, 0, &first),
            ncclSuccess);
  ncclIbDevs[0].capsProvider.mlx5.dataDirect = 1;
  ASSERT_EQ(ncclIbRegMrDmaBufInternal2(&base, storage.data() + 2 * page, 1, NCCL_PTR_CUDA, 8, 12, 0, &second),
            ncclSuccess);

  EXPECT_EQ(first, &verbsMr);
  EXPECT_EQ(second, &mlx5Mr);
  EXPECT_EQ(verbs.calls, 1);
  EXPECT_EQ(mlx5.calls, 1);
}

TEST_F(NetIbRegMicrotest, RegAndDeregMr_MultiDeviceCommUsesEveryDevice) {
  ncclIbSendComm comm{};
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

  ASSERT_EQ(ncclIbRegMr(&comm, data.data(), data.size(), NCCL_PTR_HOST, &handle), ncclSuccess);
  auto* wrapper = static_cast<ncclIbMrHandle*>(handle);
  ASSERT_NE(wrapper, nullptr);
  EXPECT_EQ(wrapper->mrs[0], &mrs[0]);
  EXPECT_EQ(wrapper->mrs[1], &mrs[1]);
  EXPECT_EQ(reg.calls, 2);

  EXPECT_EQ(ncclIbDeregMr(&comm, handle), ncclSuccess);
  EXPECT_EQ(dereg.calls, 2);
}

TEST_F(NetIbRegMicrotest, RegMr_MultiDeviceFailureReturnsErrorWithoutPublishingWrapper) {
  ncclIbSendComm comm{};
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

  EXPECT_EQ(ncclIbRegMr(&comm, data.data(), data.size(), NCCL_PTR_HOST, &handle), ncclRemoteError);
  EXPECT_EQ(handle, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(reg.calls, 2);
}

TEST_F(NetIbRegMicrotest, DISABLED_RegMr_MultiDeviceFailureUnwindsEarlierRegistrations) {
  // Production leaves the first device's MR cached when the second device
  // registration fails. Enable with the fix tracked by AICOMRCCL-3112.
  ncclIbSendComm comm{};
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

  EXPECT_EQ(ncclIbRegMr(&comm, data.data(), data.size(), NCCL_PTR_HOST, &handle), ncclRemoteError);
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(ncclIbDevs[0].mrCache.population, 0);
  EXPECT_EQ(ncclIbDevs[0].mrCache.slots, nullptr);
}

TEST_F(NetIbRegMicrotest, RegMrDmaBuf_PublicWrapperForwardsOffsetAndFd) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base = MakeDevBase(0);
  ibv_mr mr{};
  ScopedHook reg(g_ibvRegDmaBufMr,
                 [&mr](ibv_mr** out, ibv_pd*, uint64_t offset, size_t, uint64_t, int fd, int) {
                   EXPECT_EQ(offset, 17u);
                   EXPECT_EQ(fd, 23);
                   *out = &mr;
                   return ncclSuccess;
                 });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });
  std::array<unsigned char, 64> data{};
  void* handle = nullptr;

  ASSERT_EQ(ncclIbRegMrDmaBuf(&comm, data.data(), data.size(), NCCL_PTR_CUDA, 17, 23, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(reg.calls, 1);
  EXPECT_EQ(ncclIbDeregMr(&comm, handle), ncclSuccess);
}

TEST_F(NetIbRegMicrotest, DeregMrInternal_ReferenceThenFinalReleaseUpdatesCacheAndDeregisters) {
  auto base = MakeDevBase();
  ibv_mr mr{};
  auto& cache = ncclIbDevs[0].mrCache;
  cache.capacity = 32;
  cache.population = 1;
  cache.slots = static_cast<ncclIbMr*>(std::calloc(cache.capacity, sizeof(ncclIbMr)));
  ASSERT_NE(cache.slots, nullptr);
  cache.slots[0].mr = &mr;
  cache.slots[0].refs = 2;
  ScopedHook dereg(g_ibvDeregMr, [&mr](ibv_mr* value) {
    EXPECT_EQ(value, &mr);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclIbDeregMrInternal(&base, &mr), ncclSuccess);
  EXPECT_EQ(cache.population, 1);
  EXPECT_EQ(cache.slots[0].refs, 1);
  EXPECT_EQ(dereg.calls, 0);
  EXPECT_EQ(ncclIbDeregMrInternal(&base, &mr), ncclSuccess);
  EXPECT_EQ(cache.population, 0);
  EXPECT_EQ(cache.slots, nullptr);
  EXPECT_EQ(cache.capacity, 0);
  EXPECT_EQ(dereg.calls, 1);
}

TEST_F(NetIbRegMicrotest, DeregMrInternal_UnknownMrReturnsInternalError) {
  auto base = MakeDevBase();
  ibv_mr mr{};

  EXPECT_EQ(ncclIbDeregMrInternal(&base, &mr), ncclInternalError);
}

}  // namespace
