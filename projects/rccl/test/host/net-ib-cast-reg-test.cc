/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/reg.cc (NET_IB_CAST_REG_CC_PATH): memory
// registration through the per-device MR cache, over the shared verbs fakes.

#include <gtest/gtest.h>

#include <unistd.h>

#include <vector>

#include "fakes/ibvwrap_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/mlx5dvwrap_fakes.h"
#include "net-ib-cast-test-common.h"

#include "fakes/libc_seam.h"
#include NET_IB_CAST_REG_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

constexpr unsigned int kBaseAccess =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

struct RegCall {
  void* addr;
  size_t length;
  uint64_t iova;
  uint64_t offset;
  int fd;
  int access;
  int mlx5Access;
};

class NetIbCastRegMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    pageSize_ = sysconf(_SC_PAGESIZE);
    buf_ = static_cast<char*>(aligned_alloc(pageSize_, 64 * pageSize_));
    send_->base.isSend = true;
    send_->base.vProps.ndevs = 1;
    send_->devs[0].base.ibDevN = 0;
    send_->devs[0].base.pd = &pd_;
  }
  void TearDown() override {
    NetIbCastResetGlobals();
    ResetIbvwrapFakes();
    ResetMlx5dvwrapFakes();
    ResetLibcFakes();
    free(buf_);
  }

  // Hands out a distinct MR per registration and records what was asked for.
  struct ibv_mr* NextMr(void* addr, size_t length, uint64_t iova, uint64_t offset, int fd, int access, int mlx5) {
    calls_.push_back({addr, length, iova, offset, fd, access, mlx5});
    mrs_.emplace_back(new ibv_mr{});
    mrs_.back()->lkey = 0x100 + (uint32_t)mrs_.size();
    mrs_.back()->rkey = 0x200 + (uint32_t)mrs_.size();
    return mrs_.back().get();
  }

  struct ncclIbMrCache& Cache(int dev = 0) { return IbCastDevs[dev].mrCache; }
  ibv_mr* MrOf(void* mhandle, int dev = 0) { return static_cast<ncclIbMrHandle*>(mhandle)->mrs[dev]; }

  size_t pageSize_ = 0;
  char* buf_ = nullptr;
  struct ibv_pd pd_ {};
  ZeroedPtr<ncclIbSendComm> send_;
  std::vector<RegCall> calls_;
  std::vector<std::unique_ptr<ibv_mr>> mrs_;
  ScopedHook<ncclResult_t(struct ibv_mr**, struct ibv_pd*, void*, size_t, int)> regMr_{
      g_ibvRegMr, [this](struct ibv_mr** ret, struct ibv_pd*, void* addr, size_t length, int access) {
        *ret = NextMr(addr, length, 0, 0, -1, access, 0);
        return ncclSuccess;
      }};
};

TEST_F(NetIbCastRegMicrotest, RegMr_RejectsNullCommNullHandleAndZeroSize) {
  void* mhandle = nullptr;
  EXPECT_EQ(IbCastRegMr(nullptr, buf_, 8, NCCL_PTR_HOST, &mhandle), ncclInvalidArgument);
  EXPECT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, nullptr), ncclInvalidArgument);
  EXPECT_EQ(IbCastRegMr(send_.get(), buf_, 0, NCCL_PTR_HOST, &mhandle), ncclInvalidArgument);
  EXPECT_EQ(mhandle, nullptr);
}

TEST_F(NetIbCastRegMicrotest, RegMr_HandleAllocationFails_ReturnsSystemError) {
  ScopedHook mallocFails(g_malloc, [](size_t) -> void* { return nullptr; });
  void* mhandle = nullptr;
  EXPECT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSystemError);
  EXPECT_EQ(mhandle, nullptr);
}

TEST_F(NetIbCastRegMicrotest, RegMr_Host_RegistersWholePagesWithTheBaseAccessFlags) {
  void* mhandle = nullptr;
  // Starts mid-page and spills into the next one.
  char* data = buf_ + pageSize_ + 16;
  ASSERT_EQ(IbCastRegMr(send_.get(), data, pageSize_, NCCL_PTR_HOST, &mhandle), ncclSuccess);
  ASSERT_EQ(calls_.size(), 1u);
  EXPECT_EQ(calls_[0].addr, buf_ + pageSize_);
  EXPECT_EQ(calls_[0].length, 2 * pageSize_);
  EXPECT_EQ(calls_[0].access, (int)kBaseAccess);
  EXPECT_EQ(MrOf(mhandle), mrs_[0].get());
  ASSERT_EQ(Cache().population, 1);
  EXPECT_EQ(Cache().slots[0].addr, (uintptr_t)(buf_ + pageSize_));
  EXPECT_EQ(Cache().slots[0].pages, 2u);
  EXPECT_EQ(Cache().slots[0].refs, 1);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMr_RelaxedOrdering_UsesIova2WithTheRelaxedFlag) {
  IbCastRelaxedOrderingEnabled = 1;
  ScopedHook iova2(g_ibvRegMrIova2,
                   [this](struct ibv_mr** ret, struct ibv_pd*, void* addr, size_t length, uint64_t iova, int access) {
                     *ret = NextMr(addr, length, iova, 0, -1, access, 0);
                     return ncclSuccess;
                   });
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSuccess);
  ASSERT_EQ(calls_.size(), 1u);
  EXPECT_EQ(calls_[0].access, (int)(kBaseAccess | IBV_ACCESS_RELAXED_ORDERING));
  EXPECT_EQ(calls_[0].iova, (uint64_t)(uintptr_t)buf_);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMr_ForceStrongOrdering_OverridesRelaxedOrdering) {
  IbCastRelaxedOrderingEnabled = 1;
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMrDmaBufInternal(send_.get(), buf_, 8, NCCL_PTR_HOST, 0, -1, NCCL_NET_MR_FLAG_FORCE_SO, &mhandle),
            ncclSuccess);
  ASSERT_EQ(calls_.size(), 1u);
  EXPECT_EQ(calls_[0].access, (int)kBaseAccess);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMrDmaBuf_PassesTheFdOffsetAndIova) {
  ScopedHook dmabuf(g_ibvRegDmabufMr, [this](struct ibv_mr** ret, struct ibv_pd*, uint64_t offset, size_t length,
                                             uint64_t iova, int fd, int access) {
    *ret = NextMr(nullptr, length, iova, offset, fd, access, 0);
    return ncclSuccess;
  });
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMrDmaBuf(send_.get(), buf_, 8, NCCL_PTR_CUDA, 4096, 17, &mhandle), ncclSuccess);
  ASSERT_EQ(calls_.size(), 1u);
  EXPECT_EQ(calls_[0].fd, 17);
  EXPECT_EQ(calls_[0].offset, 4096u);
  EXPECT_EQ(calls_[0].iova, (uint64_t)(uintptr_t)buf_);
  EXPECT_EQ(calls_[0].length, pageSize_);
  EXPECT_EQ(calls_[0].access, (int)kBaseAccess);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMrDmaBuf_DataDirectDevice_GoesThroughMlx5WithTheDataDirectFlag) {
  IbCastDevs[0].ibProvider = IB_PROVIDER_MLX5;
  IbCastDevs[0].capsProvider.mlx5.dataDirect = 1;
  ScopedHook dmabuf(g_mlx5dvRegDmabufMr, [this](struct ibv_mr** ret, struct ibv_pd*, uint64_t offset, size_t length,
                                                uint64_t iova, int fd, int access, int mlx5Access) {
    *ret = NextMr(nullptr, length, iova, offset, fd, access, mlx5Access);
    return ncclSuccess;
  });
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMrDmaBuf(send_.get(), buf_, 8, NCCL_PTR_CUDA, 0, 5, &mhandle), ncclSuccess);
  ASSERT_EQ(calls_.size(), 1u);
  EXPECT_EQ(calls_[0].mlx5Access, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
  EXPECT_EQ(calls_[0].fd, 5);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
}

// An ionic device's uDMA count shares capsProvider with mlx5.dataDirect.
TEST_F(NetIbCastRegMicrotest, RegMrDmaBuf_OnlyAnMlx5DataDirectDeviceGoesThroughMlx5) {
  IbCastDevs[0].capsProvider.ionic.udmaCount = 2;
  ScopedHook dmabuf(g_ibvRegDmabufMr, [this](struct ibv_mr** ret, struct ibv_pd*, uint64_t offset, size_t length,
                                             uint64_t iova, int fd, int access) {
    *ret = NextMr(nullptr, length, iova, offset, fd, access, 0);
    return ncclSuccess;
  });
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMrDmaBuf(send_.get(), buf_, 8, NCCL_PTR_CUDA, 0, 5, &mhandle), ncclSuccess);
  EXPECT_EQ(dmabuf.calls, 1);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
  IbCastDevs[0].ibProvider = IB_PROVIDER_MLX5;
  IbCastDevs[0].capsProvider.mlx5.dataDirect = 0;
  ASSERT_EQ(IbCastRegMrDmaBuf(send_.get(), buf_, 8, NCCL_PTR_CUDA, 0, 5, &mhandle), ncclSuccess);
  EXPECT_EQ(dmabuf.calls, 2);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMr_RangeInsideACachedOne_ReusesItsMr) {
  void *outer = nullptr, *inner = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, 4 * pageSize_, NCCL_PTR_HOST, &outer), ncclSuccess);
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_ + pageSize_, pageSize_, NCCL_PTR_HOST, &inner), ncclSuccess);
  EXPECT_EQ(calls_.size(), 1u);
  EXPECT_EQ(MrOf(inner), MrOf(outer));
  EXPECT_EQ(Cache().slots[0].refs, 2);
  // The MR survives until its last user lets go.
  int derefs = 0;
  ScopedHook dereg(g_ibvDeregMr, [&](struct ibv_mr*) {
    ++derefs;
    return ncclSuccess;
  });
  ASSERT_EQ(IbCastDeregMr(send_.get(), inner), ncclSuccess);
  EXPECT_EQ(derefs, 0);
  EXPECT_EQ(Cache().population, 1);
  ASSERT_EQ(IbCastDeregMr(send_.get(), outer), ncclSuccess);
  EXPECT_EQ(derefs, 1);
  EXPECT_EQ(Cache().population, 0);
  EXPECT_EQ(Cache().slots, nullptr);
  EXPECT_EQ(Cache().capacity, 0);
}

TEST_F(NetIbCastRegMicrotest, RegMr_RangeRunningPastACachedOne_RegistersAgain) {
  void *first = nullptr, *second = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, pageSize_, NCCL_PTR_HOST, &first), ncclSuccess);
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, 2 * pageSize_, NCCL_PTR_HOST, &second), ncclSuccess);
  EXPECT_EQ(calls_.size(), 2u);
  EXPECT_NE(MrOf(first), MrOf(second));
  EXPECT_EQ(Cache().population, 2);
  EXPECT_EQ(IbCastDeregMr(send_.get(), first), ncclSuccess);
  EXPECT_EQ(IbCastDeregMr(send_.get(), second), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMr_LowerAddress_IsInsertedInFrontToKeepTheCacheSorted) {
  void *high = nullptr, *low = nullptr, *again = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_ + 8 * pageSize_, pageSize_, NCCL_PTR_HOST, &high), ncclSuccess);
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, pageSize_, NCCL_PTR_HOST, &low), ncclSuccess);
  ASSERT_EQ(Cache().population, 2);
  EXPECT_EQ(Cache().slots[0].addr, (uintptr_t)buf_);
  EXPECT_EQ(Cache().slots[1].addr, (uintptr_t)(buf_ + 8 * pageSize_));
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_ + 8 * pageSize_, 8, NCCL_PTR_HOST, &again), ncclSuccess);
  EXPECT_EQ(MrOf(again), MrOf(high));
  EXPECT_EQ(calls_.size(), 2u);
  for (void* h : {high, low, again}) EXPECT_EQ(IbCastDeregMr(send_.get(), h), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMr_CacheFull_GrowsFrom32ToDouble) {
  std::vector<void*> handles(33);
  for (int i = 0; i < 33; i++) {
    ASSERT_EQ(IbCastRegMr(send_.get(), buf_ + i * pageSize_, 8, NCCL_PTR_HOST, &handles[i]), ncclSuccess);
    EXPECT_EQ(Cache().capacity, i < 32 ? 32 : 64) << i;
  }
  for (void* h : handles) EXPECT_EQ(IbCastDeregMr(send_.get(), h), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMr_RecvComm_UsesTheRecvDevicesBase) {
  ZeroedPtr<ncclIbRecvComm> recv;
  recv->base.isSend = false;
  recv->base.vProps.ndevs = 1;
  recv->devs[0].base.ibDevN = 1;
  recv->devs[0].base.pd = &pd_;
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMr(recv.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSuccess);
  EXPECT_EQ(Cache(1).population, 1);
  EXPECT_EQ(Cache(0).population, 0);
  EXPECT_EQ(IbCastDeregMr(recv.get(), mhandle), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, RegMr_TwoDevices_RegistersOnEachDevice) {
  send_->base.vProps.ndevs = 2;
  send_->devs[1].base.ibDevN = 3;
  send_->devs[1].base.pd = &pd_;
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSuccess);
  EXPECT_EQ(calls_.size(), 2u);
  EXPECT_EQ(MrOf(mhandle, 1), mrs_[1].get());
  EXPECT_EQ(Cache(3).population, 1);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
  EXPECT_EQ(Cache(3).population, 0);
}

TEST_F(NetIbCastRegMicrotest, RegMr_RegistrationFails_FreesTheHandleAndReturnsTheError) {
  ScopedHook regFails(g_ibvRegMr, [](struct ibv_mr**, struct ibv_pd*, void*, size_t, int) { return ncclSystemError; });
  void* wrapper = nullptr;
  ScopedHook alloc(g_malloc, [&](size_t size) { return wrapper = malloc(size); });
  void* mhandle = nullptr;
  EXPECT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSystemError);
  EXPECT_EQ(mhandle, nullptr);
  ASSERT_NE(wrapper, nullptr);
  EXPECT_EQ(g_freedPointers.back(), wrapper);
  EXPECT_EQ(Cache().population, 0);
}

// Deregistering moves the last slot into the hole, so the cache is no longer sorted and the
// lookup's early exit misses a cached range. Pins AICOMRCCL-3092; a fix flips this pin.
TEST_F(NetIbCastRegMicrotest, DeregMr_KnownBug_CacheLosesItsOrderAndRegistersTwice) {
  void *low = nullptr, *mid = nullptr, *high = nullptr, *again = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &low), ncclSuccess);
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_ + 2 * pageSize_, 8, NCCL_PTR_HOST, &mid), ncclSuccess);
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_ + 4 * pageSize_, 8, NCCL_PTR_HOST, &high), ncclSuccess);
  ASSERT_EQ(IbCastDeregMr(send_.get(), low), ncclSuccess);
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_ + 2 * pageSize_, 8, NCCL_PTR_HOST, &again), ncclSuccess);
  EXPECT_EQ(calls_.size(), 4u);
  EXPECT_NE(MrOf(again), MrOf(mid));
  for (void* h : {mid, high, again}) EXPECT_EQ(IbCastDeregMr(send_.get(), h), ncclSuccess);
}

// A failure on a later device returns without releasing the MRs the earlier devices took.
// Pins AICOMRCCL-3093; a fix flips this pin.
TEST_F(NetIbCastRegMicrotest, RegMr_KnownBug_FailureOnASecondDeviceKeepsTheFirstDevicesMr) {
  send_->base.vProps.ndevs = 2;
  send_->devs[1].base.ibDevN = 3;
  send_->devs[1].base.pd = &pd_;
  int regs = 0;
  ScopedHook secondFails(g_ibvRegMr, [&](struct ibv_mr** ret, struct ibv_pd*, void* addr, size_t length, int access) {
    if (++regs == 2) return ncclSystemError;
    *ret = NextMr(addr, length, 0, 0, -1, access, 0);
    return ncclSuccess;
  });
  void* mhandle = nullptr;
  EXPECT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSystemError);
  EXPECT_EQ(Cache(0).population, 1);
  EXPECT_EQ(Cache(0).slots[0].refs, 1);
}

TEST_F(NetIbCastRegMicrotest, DeregMr_NullHandle_IsANoOpAndNullCommIsRejected) {
  EXPECT_EQ(IbCastDeregMr(send_.get(), nullptr), ncclSuccess);
  ncclIbMrHandle handle{};
  EXPECT_EQ(IbCastDeregMr(nullptr, &handle), ncclInvalidArgument);
}

TEST_F(NetIbCastRegMicrotest, DeregMr_UnknownMr_IsAnInternalError) {
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSuccess);
  ibv_mr stranger{};
  ncclIbMrHandle* bogus = static_cast<ncclIbMrHandle*>(malloc(sizeof(ncclIbMrHandle)));
  bogus->mrs[0] = &stranger;
  EXPECT_EQ(IbCastDeregMr(send_.get(), bogus), ncclInternalError);
  free(bogus);
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSuccess);
}

TEST_F(NetIbCastRegMicrotest, DeregMr_VerbsDeregFails_ReturnsTheError) {
  void* mhandle = nullptr;
  ASSERT_EQ(IbCastRegMr(send_.get(), buf_, 8, NCCL_PTR_HOST, &mhandle), ncclSuccess);
  ScopedHook dereg(g_ibvDeregMr, [](struct ibv_mr*) { return ncclSystemError; });
  EXPECT_EQ(IbCastDeregMr(send_.get(), mhandle), ncclSystemError);
  EXPECT_EQ(Cache().population, 0);
  free(mhandle);  // the failed deregistration returns before releasing the handle
}

}  // namespace
