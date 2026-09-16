/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suite A: host IPC table unit tests (gin_anvil_ipc_table_host.cc).

#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

namespace RcclUnitTesting
{

class GinAnvilIpcTableHostTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_EQ(hipSetDevice(0), hipSuccess); }
};

// A1: invalid registration arguments.
TEST_F(GinAnvilIpcTableHostTest, RegisterVmm_InvalidArgs) {
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterVmm(nullptr, 64, 0, 2, 4096), -1);
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterVmm(reinterpret_cast<void*>(0x1000), 0, 0, 2, 4096), -1);
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterVmm(reinterpret_cast<void*>(0x1000), 64, 0, 0, 4096), -1);
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterVmm(reinterpret_cast<void*>(0x1000), 64, 0,
                                            NCCL_GIN_ANVIL_IPC_MAX_RANKS + 1, 4096),
            -1);
}

// A2: VMM stride fills remote_bases[pe].
TEST_F(GinAnvilIpcTableHostTest, RegisterVmm_StridePeerBases) {
  constexpr uintptr_t kBase = 0xA0000000ULL;
  constexpr ptrdiff_t kStride = 0x100000;
  constexpr int kMyRank = 2;
  constexpr int kNRanks = 4;
  void* local = reinterpret_cast<void*>(kBase);

  ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(local, 4096, kMyRank, kNRanks, kStride), 0);

  const ncclGinAnvilIpcBufEntry* table = nullptr;
  int count = 0;
  ncclGinAnvilIpcTableGetDevice(&table, &count);
  ASSERT_NE(table, nullptr);
  ASSERT_GE(count, 1);

  // Copy device table to host for inspection.
  std::vector<ncclGinAnvilIpcBufEntry> hostTable(static_cast<size_t>(count));
  ASSERT_EQ(hipMemcpy(hostTable.data(), table, count * sizeof(ncclGinAnvilIpcBufEntry),
                      hipMemcpyDeviceToHost),
            hipSuccess);

  const ncclGinAnvilIpcBufEntry& e = hostTable[0];
  EXPECT_EQ(e.local_base, kBase);
  EXPECT_EQ(e.length, 4096u);
  for (int pe = 0; pe < kNRanks; ++pe) {
  EXPECT_EQ(e.remote_bases[pe], kBase + static_cast<uintptr_t>((pe - kMyRank) * kStride))
      << "pe=" << pe;
  }

  ASSERT_EQ(ncclGinAnvilIpcTableUnregister(local), 0);
}

// A3: duplicate registration is idempotent.
TEST_F(GinAnvilIpcTableHostTest, RegisterVmm_DuplicateIsIdempotent) {
  void* local = reinterpret_cast<void*>(0xB0001000ULL);
  ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(local, 1024, 0, 2, 0x80000), 0);
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterVmm(local, 1024, 0, 2, 0x80000), 0);
  ASSERT_EQ(ncclGinAnvilIpcTableUnregister(local), 0);
}

// A4: explicit remote base array.
TEST_F(GinAnvilIpcTableHostTest, RegisterExplicit_RemoteBases) {
  void* local = reinterpret_cast<void*>(0xC0002000ULL);
  uintptr_t remotes[2] = {0xD0000000ULL, 0xD0010000ULL};
  ASSERT_EQ(ncclGinAnvilIpcTableRegisterExplicit(local, remotes, 2, 512), 0);

  const ncclGinAnvilIpcBufEntry* table = nullptr;
  int count = 0;
  ncclGinAnvilIpcTableGetDevice(&table, &count);
  std::vector<ncclGinAnvilIpcBufEntry> hostTable(static_cast<size_t>(count));
  ASSERT_EQ(hipMemcpy(hostTable.data(), table, count * sizeof(ncclGinAnvilIpcBufEntry),
                      hipMemcpyDeviceToHost),
            hipSuccess);

  bool found = false;
  for (const auto& e : hostTable) {
    if (e.local_base == reinterpret_cast<uintptr_t>(local)) {
      found = true;
      EXPECT_EQ(e.remote_bases[0], remotes[0]);
      EXPECT_EQ(e.remote_bases[1], remotes[1]);
    }
  }
  EXPECT_TRUE(found);
  ASSERT_EQ(ncclGinAnvilIpcTableUnregister(local), 0);
}

// A5: unregister miss and success (swap-remove middle entry).
TEST_F(GinAnvilIpcTableHostTest, Unregister_SuccessAndMiss) {
  void* a = reinterpret_cast<void*>(0xE0003000ULL);
  void* b = reinterpret_cast<void*>(0xE0004000ULL);
  ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(a, 256, 0, 2, 0x1000), 0);
  ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(b, 256, 0, 2, 0x1000), 0);
  EXPECT_EQ(ncclGinAnvilIpcTableUnregister(reinterpret_cast<void*>(0xDEADBEEFULL)), -1);
  ASSERT_EQ(ncclGinAnvilIpcTableUnregister(a), 0);
  ASSERT_EQ(ncclGinAnvilIpcTableUnregister(b), 0);
  EXPECT_EQ(ncclGinAnvilIpcTableUnregister(nullptr), -1);
}

// A6: GetDevice on empty table.
TEST_F(GinAnvilIpcTableHostTest, GetDevice_AfterFullUnregister) {
  void* local = reinterpret_cast<void*>(0xF0005000ULL);
  ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(local, 128, 0, 1, 0), 0);
  ASSERT_EQ(ncclGinAnvilIpcTableUnregister(local), 0);

  const ncclGinAnvilIpcBufEntry* table = nullptr;
  int count = -1;
  ncclGinAnvilIpcTableGetDevice(&table, &count);
  EXPECT_EQ(count, 0);
}

// A7–A9: track / untrack / refresh.
TEST_F(GinAnvilIpcTableHostTest, TrackContext_RefreshOnRegister) {
  ncclGinAnvilSdmaGPUContext hostCtx{};
  ncclGinAnvilSdmaGPUContext* devCtx = nullptr;
  ASSERT_EQ(hipMalloc(&devCtx, sizeof(ncclGinAnvilSdmaGPUContext)), hipSuccess);

  ncclGinAnvilIpcTableTrackContext(nullptr, devCtx);
  ncclGinAnvilIpcTableTrackContext(&hostCtx, nullptr);

  hostCtx.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  ncclGinAnvilIpcTableTrackContext(&hostCtx, devCtx);
  // Duplicate track is ignored.
  ncclGinAnvilIpcTableTrackContext(&hostCtx, devCtx);

  void* local = reinterpret_cast<void*>(0x11006000ULL);
  ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(local, 64, 0, 2, 0x2000), 0);
  EXPECT_NE(hostCtx.ipcTable, nullptr);
  EXPECT_GT(hostCtx.ipcTableCount, 0);

  ncclGinAnvilSdmaGPUContext devCopy{};
  ASSERT_EQ(hipMemcpy(&devCopy, devCtx, sizeof(devCopy), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(devCopy.ipcTableCount, hostCtx.ipcTableCount);

  ncclGinAnvilIpcTableUntrackContext(reinterpret_cast<ncclGinAnvilSdmaGPUContext*>(0x1));
  ncclGinAnvilIpcTableUntrackContext(&hostCtx);
  ASSERT_EQ(ncclGinAnvilIpcTableUnregister(local), 0);
  ASSERT_EQ(hipFree(devCtx), hipSuccess);
}

// A10: table capacity.
TEST_F(GinAnvilIpcTableHostTest, MaxBufs_ReturnsError) {
  std::vector<void*> locals;
  locals.reserve(NCCL_GIN_ANVIL_IPC_MAX_BUFS + 1);
  for (int i = 0; i < NCCL_GIN_ANVIL_IPC_MAX_BUFS; ++i) {
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(0x20000000ULL + i * 0x10000ULL));
    locals.push_back(p);
    ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(p, 64, 0, 2, 0x1000), 0) << "i=" << i;
  }
  void* overflow = reinterpret_cast<void*>(0x30000000ULL);
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterVmm(overflow, 64, 0, 2, 0x1000), -1);
  for (void* p : locals) {
    ASSERT_EQ(ncclGinAnvilIpcTableUnregister(p), 0);
  }
}

// A11: explicit register invalid args.
TEST_F(GinAnvilIpcTableHostTest, RegisterExplicit_InvalidArgs) {
  uintptr_t remotes[1] = {0};
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterExplicit(nullptr, remotes, 1, 1), -1);
  EXPECT_EQ(ncclGinAnvilIpcTableRegisterExplicit(reinterpret_cast<void*>(0x4000), nullptr, 1, 1),
            -1);
}

}  // namespace RcclUnitTesting
