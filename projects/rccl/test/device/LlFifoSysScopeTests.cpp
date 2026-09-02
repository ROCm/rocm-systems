/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Guards the gfx1250 LL/LL128 comm-FIFO system-scope load+store pairing
// (RCCL_LL_FIFO_SYS_SCOPE). A cacheable FIFO between sibling DPX partitions
// hangs at slot reuse (NCCL_STEPS=8, first hang on the 9th op) unless both the
// flag poll and the FIFO store use system-scope b128.
//
//   Gfx1250EnablesSysScope     — device compile of the guard (1 GPU)
//   FifoLineSysScopeRoundtrip  — storeLL-shaped b128 store + load on one GPU
//   SiblingBroadcastSlotReuse  — 9 Ring/LL broadcasts on a sibling pair
//                                (the hang the store-side fix closed)

#include "DeviceTestBase.hpp"

#include "../common/ProcessIsolatedTestRunner.hpp"
#include "rccl_ptr.h"

#include <rccl/rccl.h>

#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace RcclUnitTesting
{

namespace
{

union TestLLLine {
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
};

__global__ void kernelSysScopeEnabled(int* out)
{
#if RCCL_LL_FIFO_SYS_SCOPE
  *out = 1;
#else
  *out = 0;
#endif
}

// Mirrors storeLL + loadLLLineB128: one 16-byte FIFO line, data+flag together.
__global__ void kernelFifoLineRoundtrip(TestLLLine* line, uint32_t data, uint32_t flag, int* ok)
{
#if RCCL_LL_FIFO_SYS_SCOPE
  union {
    v4u v;
    uint32_t w[4];
  } st, ld;
  st.w[0] = data;
  st.w[1] = flag;
  st.w[2] = data;
  st.w[3] = flag;
  __builtin_amdgcn_global_store_b128((v4u_gptr)line, st.v, RCCL_SYSTEM_SYNCSCOPE);
  ld.v = __builtin_amdgcn_global_load_b128((v4u_gptr)line, RCCL_SYSTEM_SYNCSCOPE);
  *ok = (ld.w[0] == data && ld.w[1] == flag && ld.w[2] == data && ld.w[3] == flag) ? 1 : 0;
#else
  line->v[0] = (uint64_t)data | ((uint64_t)flag << 32);
  line->v[1] = (uint64_t)data | ((uint64_t)flag << 32);
  *ok = (line->data1 == data && line->flag1 == flag && line->data2 == data && line->flag2 == flag)
          ? 1
          : 0;
#endif
}

bool isGfx1250(int device)
{
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
  return std::string(prop.gcnArchName).rfind("gfx1250", 0) == 0;
}

// HIP_VISIBLE_DEVICES lists physical indices; hipGetDeviceCount() is logical.
std::vector<int> hipVisiblePhysicalIds()
{
  const char* hvd = std::getenv("HIP_VISIBLE_DEVICES");
  std::vector<int> ids;
  if (hvd && *hvd) {
    std::stringstream ss(hvd);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) ids.push_back(std::stoi(tok));
    }
    return ids;
  }
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return {};
  ids.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = i;
  return ids;
}

// Sibling DPX partitions share PCI domain:bus (writeup / hang matrix).
// Returns physical device indices suitable for HIP_VISIBLE_DEVICES.
bool findSiblingPair(int* physA, int* physB)
{
  const std::vector<int> phys = hipVisiblePhysicalIds();
  const int n = static_cast<int>(phys.size());
  if (n < 2) return false;
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t pi{};
    if (hipGetDeviceProperties(&pi, i) != hipSuccess) continue;
    for (int j = i + 1; j < n; ++j) {
      hipDeviceProp_t pj{};
      if (hipGetDeviceProperties(&pj, j) != hipSuccess) continue;
      if (pi.pciDomainID == pj.pciDomainID && pi.pciBusID == pj.pciBusID &&
          std::string(pi.gcnArchName).rfind("gfx1250", 0) == 0) {
        *physA = phys[static_cast<size_t>(i)];
        *physB = phys[static_cast<size_t>(j)];
        return true;
      }
    }
  }
  return false;
}

void runSiblingBroadcastSlotReuse(int devA, int devB)
{
  const int devs[2] = {devA, devB};
  ncclComm_t comms[2] = {};
  ASSERT_EQ(ncclCommInitAll(comms, 2, devs), ncclSuccess);

  constexpr int kCount = 256;  // 1 KiB float broadcast; one LL step per coll
  constexpr int kIters = 9;    // first FIFO slot reuse (NCCL_STEPS == 8)
  float* send[2] = {};
  float* recv[2] = {};
  hipStream_t streams[2] = {};

  for (int r = 0; r < 2; ++r) {
    ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
    ASSERT_EQ(hipMalloc(&send[r], kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&recv[r], kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&streams[r]), hipSuccess);
    std::vector<float> h(kCount, r == 0 ? 42.0f : -1.0f);
    ASSERT_EQ(hipMemcpy(send[r], h.data(), kCount * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(recv[r], 0, kCount * sizeof(float)), hipSuccess);
  }

  for (int iter = 0; iter < kIters; ++iter) {
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int r = 0; r < 2; ++r) {
      ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
      ASSERT_EQ(ncclBroadcast(send[r], recv[r], kCount, ncclFloat, 0, comms[r], streams[r]),
                ncclSuccess);
    }
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  }

  for (int r = 0; r < 2; ++r) {
    ASSERT_EQ(hipSetDevice(devs[r]), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(streams[r]), hipSuccess);
    std::vector<float> h(kCount, 0);
    ASSERT_EQ(hipMemcpy(h.data(), recv[r], kCount * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    for (int i = 0; i < kCount; ++i) {
      ASSERT_EQ(h[i], 42.0f) << "rank " << r << " elem " << i;
    }
    ASSERT_EQ(hipStreamDestroy(streams[r]), hipSuccess);
    ASSERT_EQ(hipFree(send[r]), hipSuccess);
    ASSERT_EQ(hipFree(recv[r]), hipSuccess);
    ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
  }
}

} // namespace

TEST_F(DeviceTestBase, Gfx1250EnablesSysScope)
{
  DeviceBuffer<int> d_out(1);
  kernelSysScopeEnabled<<<1, 1>>>(d_out.ptr);
  syncAndCheck();
  const int enabled = d_out.download();
  if (isGfx1250(0)) {
    EXPECT_EQ(enabled, 1) << "RCCL_LL_FIFO_SYS_SCOPE must be 1 in gfx1250 device code";
  } else {
    EXPECT_EQ(enabled, 0) << "RCCL_LL_FIFO_SYS_SCOPE is gfx1250-only";
  }
}

TEST_F(DeviceTestBase, FifoLineSysScopeRoundtrip)
{
  DeviceBuffer<TestLLLine> d_line(1);
  DeviceBuffer<int> d_ok(1);
  d_line.zero();
  constexpr uint32_t kData = 0xA5A5A5A5u;
  constexpr uint32_t kFlag = 9;  // first reused-slot flag (step 8 -> flag 9)
  kernelFifoLineRoundtrip<<<1, 1>>>(d_line.ptr, kData, kFlag, d_ok.ptr);
  syncAndCheck();
  EXPECT_EQ(d_ok.download(), 1);
}

TEST(LlFifoSysScope, SiblingBroadcastSlotReuse)
{
  // Hang gate: skip unless gfx1250 DPX siblings; that is the only config that hung.
  int a = 0, b = 1;
  if (!findSiblingPair(&a, &b)) {
    GTEST_SKIP() << "needs two gfx1250 devices that share PCI domain:bus (DPX siblings)";
  }

  const std::string hvd = std::to_string(a) + "," + std::to_string(b);
  RUN_ISOLATED_TESTS(
    ProcessIsolatedTestRunner::TestConfig("LlFifoSysScope.SiblingBroadcastSlotReuse",
                                          []() { runSiblingBroadcastSlotReuse(0, 1); })
      .withEnvironment({
        {"HIP_VISIBLE_DEVICES", hvd},
        {"NCCL_PROTO", "LL"},
        {"NCCL_ALGO", "Ring"},
        {"RCCL_DDA_THRESHOLD", "0"},
        {"NCCL_IB_DISABLE", "1"},
        {"NCCL_SOCKET_IFNAME", "lo"},
      })
      .withTimeout(std::chrono::seconds(60))
      .withNumGpus(ProcessIsolatedTestRunner::TestConfig::kCpuOnly));
}

} // namespace RcclUnitTesting
