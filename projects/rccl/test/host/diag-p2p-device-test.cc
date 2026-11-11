/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for the src/diagnostics/device/p2p.cu launchers, #include-d via DIAG_P2P_DEVICE_CU_PATH under
 * renamed symbols so they sit beside fakes/diagnostics_p2p_device_fakes.cc. Kernels stay device code: the HIP launch
 * entry points below record each launch instead of running it.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

#include "checks.h"
#include "diagnostics/p2p.h"

#define ncclDiagP2pInitSlots DiagP2pDeviceInitSlots
#define ncclDiagP2pRemoteWrite DiagP2pDeviceRemoteWrite
#define ncclDiagP2pVerifyWrites DiagP2pDeviceVerifyWrites
#define ncclDiagP2pRemoteRead DiagP2pDeviceRemoteRead

#include DIAG_P2P_DEVICE_CU_PATH

#undef ncclDiagP2pRemoteRead
#undef ncclDiagP2pVerifyWrites
#undef ncclDiagP2pRemoteWrite
#undef ncclDiagP2pInitSlots

namespace {

struct DiagP2pLaunch {
  const void* kernel;
  dim3 grid;
  dim3 block;
  size_t sharedMem;
  hipStream_t stream;
  std::vector<std::string> args;
};

DiagP2pLaunch g_pending;
std::vector<DiagP2pLaunch> g_launches;

}  // namespace

// This TU builds with -fno-hip-new-launch-api and -fuse-cuid=none (see CMakeLists.txt), so <<<>>> lowers to these
// rather than to hipLaunchKernel, which hip_fakes.cc pins to abort, and registers the unsuffixed __hip_fatbin. That
// is defined in asm: a C definition clashes with the compiler's own declaration of it.
asm(".pushsection .data\n.globl __hip_fatbin\n__hip_fatbin:\n.byte 0\n.popsection");
// Weak, so a strong copy in hip_fakes.cc wins.
extern "C" {
__attribute__((weak)) void** __hipRegisterFatBinary(const void*) {
  static void* module;
  return &module;
}
__attribute__((weak)) void __hipRegisterFunction(void**, const void*, char*, const char*, unsigned int, void*, void*,
                                                 void*, void*, int*) {}
__attribute__((weak)) void __hipUnregisterFatBinary(void**) {}
__attribute__((weak)) void __hipRegisterVar(void**, char*, char*, const char*, int, size_t, int, int) {}
// clang >= 23 registers per-kernel profile shadows under -fprofile-instr-generate.
__attribute__((weak)) void __llvm_profile_offload_register_shadow_variable(void*) {}
__attribute__((weak)) void __llvm_profile_offload_register_section_shadow_variable(void*) {}
}  // extern "C"

hipError_t hipConfigureCall(dim3 grid, dim3 block, size_t sharedMem, hipStream_t stream) {
  g_pending = {nullptr, grid, block, sharedMem, stream, {}};
  return hipSuccess;
}

hipError_t hipSetupArgument(const void* arg, size_t size, size_t) {
  g_pending.args.emplace_back(static_cast<const char*>(arg), size);
  return hipSuccess;
}

hipError_t hipLaunchByPtr(const void* kernel) {
  g_pending.kernel = kernel;
  g_launches.push_back(g_pending);
  return hipSuccess;
}

namespace {

const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x5717);

template <typename T>
std::string Bytes(const T& value) {
  return std::string(reinterpret_cast<const char*>(&value), sizeof(value));
}

class DiagP2pDeviceMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_launches.clear();
    g_hipGetLastError = [] { return hipSuccess; };
  }

  void TearDown() override {
    g_launches.clear();
    ResetHipFakes();
    ResetNcclFakes();
  }

  void ExpectLaunch(const void* kernel, unsigned int blocks, const std::vector<std::string>& args) {
    ASSERT_EQ(g_launches.size(), 1u);
    const DiagP2pLaunch& launch = g_launches[0];
    EXPECT_EQ(launch.kernel, kernel);
    EXPECT_EQ(launch.grid.x, blocks);
    EXPECT_EQ(launch.grid.y * launch.grid.z, 1u);
    EXPECT_EQ(launch.block.x, 128u);
    EXPECT_EQ(launch.block.y * launch.block.z, 1u);
    EXPECT_EQ(launch.sharedMem, 0u);
    EXPECT_EQ(launch.stream, kStream);
    EXPECT_EQ(launch.args, args);
    g_launches.clear();
  }

  ncclDiagP2pSlot slots_[2]{};
  ncclDiagP2pRemoteOp ops_[2]{};
  int slotRanks_[2] = {4, 6};
  uint64_t readback_[2]{};
};

TEST_F(DiagP2pDeviceMicrotest, InitSlots_LaunchesOneThreadPerSlotIn128ThreadBlocks) {
  const void* kernel = reinterpret_cast<const void*>(diagP2pInitSlotsKernel);
  for (auto [count, blocks] : {std::pair{1, 1u}, {128, 1u}, {129, 2u}, {385, 4u}}) {
    SCOPED_TRACE(::testing::Message() << "count " << count);
    ASSERT_EQ(DiagP2pDeviceInitSlots(slots_, slotRanks_, count, 9, kStream), ncclSuccess);
    ExpectLaunch(kernel, blocks, {Bytes(&slots_[0]), Bytes(&slotRanks_[0]), Bytes(count), Bytes(9)});
  }
}

TEST_F(DiagP2pDeviceMicrotest, RemoteWrite_LaunchesWriteKernelOverOps) {
  ASSERT_EQ(DiagP2pDeviceRemoteWrite(ops_, 2, kStream), ncclSuccess);
  ExpectLaunch(reinterpret_cast<const void*>(diagP2pRemoteWriteKernel), 1, {Bytes(&ops_[0]), Bytes(2)});
  ASSERT_EQ(DiagP2pDeviceRemoteWrite(ops_, 257, kStream), ncclSuccess);
  ExpectLaunch(reinterpret_cast<const void*>(diagP2pRemoteWriteKernel), 3, {Bytes(&ops_[0]), Bytes(257)});
}

TEST_F(DiagP2pDeviceMicrotest, VerifyWrites_LaunchesVerifyKernelOverSlots) {
  ASSERT_EQ(DiagP2pDeviceVerifyWrites(slots_, 130, kStream), ncclSuccess);
  ExpectLaunch(reinterpret_cast<const void*>(diagP2pVerifyWritesKernel), 2, {Bytes(&slots_[0]), Bytes(130)});
}

TEST_F(DiagP2pDeviceMicrotest, RemoteRead_LaunchesReadKernelIntoReadback) {
  ASSERT_EQ(DiagP2pDeviceRemoteRead(ops_, 2, readback_, kStream), ncclSuccess);
  ExpectLaunch(reinterpret_cast<const void*>(diagP2pRemoteReadKernel), 1,
               {Bytes(&ops_[0]), Bytes(2), Bytes(&readback_[0])});
}

TEST_F(DiagP2pDeviceMicrotest, EmptyOrNegativeCountLaunchesNothing) {
  ScopedHook lastError(g_hipGetLastError, [] { return hipErrorLaunchFailure; });
  for (int count : {0, -1}) {
    EXPECT_EQ(DiagP2pDeviceInitSlots(slots_, slotRanks_, count, 9, kStream), ncclSuccess);
    EXPECT_EQ(DiagP2pDeviceRemoteWrite(ops_, count, kStream), ncclSuccess);
    EXPECT_EQ(DiagP2pDeviceVerifyWrites(slots_, count, kStream), ncclSuccess);
    EXPECT_EQ(DiagP2pDeviceRemoteRead(ops_, count, readback_, kStream), ncclSuccess);
  }
  EXPECT_TRUE(g_launches.empty());
  EXPECT_EQ(lastError.calls, 0);
}

TEST_F(DiagP2pDeviceMicrotest, LaunchErrorIsReturnedAsUnhandledCudaError) {
  ScopedHook lastError(g_hipGetLastError, [] { return hipErrorLaunchFailure; });
  EXPECT_EQ(DiagP2pDeviceInitSlots(slots_, slotRanks_, 1, 9, kStream), ncclUnhandledCudaError);
  EXPECT_EQ(DiagP2pDeviceRemoteWrite(ops_, 1, kStream), ncclUnhandledCudaError);
  EXPECT_EQ(DiagP2pDeviceVerifyWrites(slots_, 1, kStream), ncclUnhandledCudaError);
  EXPECT_EQ(DiagP2pDeviceRemoteRead(ops_, 1, readback_, kStream), ncclUnhandledCudaError);
  EXPECT_EQ(g_launches.size(), 4u);
  EXPECT_EQ(lastError.calls, 4);
}

}  // namespace
