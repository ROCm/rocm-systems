/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/device/onerank.cu: ncclLaunchOneRank's copy-or-reduce choice, its kernel
// table and launch geometry, observed through hip_fakes.cc's launch and copy hooks.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "nccl.h"
#include "debug.h"
#include "device.h"

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/hip_fakes.h"

// device/common.h is the kernel-side runtime (shared memory, work batches) and does not compile host-only;
// onerank.cu only needs the reduceCopy and RedOp templates common_kernel.h layers on top of it.
#define NCCL_DEVICE_COMMON_H_
#include ONERANK_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

constexpr size_t kElts = 100;

struct Memcpy {
  void* dst;
  const void* src;
  size_t bytes;
  hipMemcpyKind kind;
  hipStream_t stream;
};
struct Launch {
  const void* fn;
  dim3 grid;
  dim3 block;
  size_t shmem;
  hipStream_t stream;
  void* dst;
  void* src;
  void* acc;
  size_t nElts;
  uint64_t scalarArg;
  bool scalarArgIsPtr;
};

ncclDevRedOpFull Op(ncclDevRedOp_t op, uint64_t scalarArg = 0, bool scalarArgIsPtr = false) {
  ncclDevRedOpFull full{};
  full.op = op;
  full.scalarArg = scalarArg;
  full.scalarArgIsPtr = scalarArgIsPtr;
  return full;
}

template <template <typename> class Func, typename T>
const void* Kernel() {
  return reinterpret_cast<const void*>(&oneRankReduce<Func, T>);
}

class OnerankMicrotest : public ::testing::Test {
 protected:
  ncclResult_t LaunchOneRank(ncclDevRedOpFull op, ncclDataType_t type, const void* acc = nullptr,
                             hipEvent_t event = nullptr, size_t nElts = kElts) {
    return ncclLaunchOneRank(dst_, src_, nElts, op, type, stream_, event, acc);
  }

  char dstStorage_[8] = {};
  char srcStorage_[8] = {};
  void* dst_ = dstStorage_;
  void* src_ = srcStorage_;
  hipStream_t stream_ = reinterpret_cast<hipStream_t>(0x5a);

  std::vector<std::string> trace_;
  std::vector<Memcpy> memcpys_;
  hipError_t memcpyResult_ = hipSuccess;
  std::vector<hipStreamCaptureMode> captureModes_;
  int exchangeFailAt_ = 0;
  std::vector<Launch> launches_;
  hipError_t launchResult_ = hipSuccess;
  std::vector<std::pair<hipEvent_t, hipStream_t>> eventRecords_;
  hipError_t eventRecordResult_ = hipSuccess;

  ScopedHook<hipError_t(hipStreamCaptureMode*)> exchange_{
      g_hipThreadExchangeStreamCaptureMode, [this](hipStreamCaptureMode* mode) {
        trace_.push_back("exchange");
        captureModes_.push_back(*mode);
        if (static_cast<int>(captureModes_.size()) == exchangeFailAt_) return hipErrorInvalidValue;
        *mode = captureModes_.size() % 2 ? hipStreamCaptureModeThreadLocal : hipStreamCaptureModeRelaxed;
        return hipSuccess;
      }};
  ScopedHook<hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)> memcpy_{
      g_hipMemcpyAsync, [this](void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
        trace_.push_back("memcpy");
        memcpys_.push_back({dst, src, bytes, kind, stream});
        return memcpyResult_;
      }};
  ScopedHook<hipError_t(hipEvent_t, hipStream_t)> record_{g_hipEventRecord, [this](hipEvent_t e, hipStream_t st) {
    trace_.push_back("record");
    eventRecords_.push_back({e, st});
    return eventRecordResult_;
  }};
  ScopedHook<hipError_t(const void*, dim3, dim3, void**, size_t, hipStream_t)> launch_{
      g_hipLaunchKernel, [this](const void* fn, dim3 grid, dim3 block, void** args, size_t shmem, hipStream_t st) {
        trace_.push_back("launch");
        launches_.push_back({fn, grid, block, shmem, st, *static_cast<void**>(args[0]), *static_cast<void**>(args[1]),
                             *static_cast<void**>(args[2]), *static_cast<size_t*>(args[3]),
                             *static_cast<uint64_t*>(args[4]), *static_cast<bool*>(args[5])});
        return launchResult_;
      }};
};

TEST_F(OnerankMicrotest, InPlaceWithoutBiasIsANoOp) {
  src_ = dst_;
  for (ncclDevRedOp_t op : {ncclDevSum, ncclDevProd, ncclDevMinMax, ncclDevSumPostDiv, ncclNumDevRedOps}) {
    EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(op), ncclFloat32)) << op;
  }
  EXPECT_TRUE(trace_.empty());
}

TEST_F(OnerankMicrotest, OutOfPlaceWithoutBiasIsOneRelaxedCopy) {
  EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevMinMax), ncclFloat64));
  EXPECT_EQ((std::vector<std::string>{"exchange", "memcpy", "exchange"}), trace_);
  ASSERT_EQ(1u, memcpys_.size());
  EXPECT_EQ(dst_, memcpys_[0].dst);
  EXPECT_EQ(src_, memcpys_[0].src);
  EXPECT_EQ(kElts * sizeof(double), memcpys_[0].bytes);
  EXPECT_EQ(hipMemcpyDefault, memcpys_[0].kind);
  EXPECT_EQ(stream_, memcpys_[0].stream);
  // Relaxed for the copy, then the caller's mode put back.
  EXPECT_EQ((std::vector<hipStreamCaptureMode>{hipStreamCaptureModeRelaxed, hipStreamCaptureModeThreadLocal}),
            captureModes_);
}

TEST_F(OnerankMicrotest, CopyFailuresAreHipErrors) {
  memcpyResult_ = hipErrorInvalidValue;
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclInt8));
  EXPECT_EQ(2u, captureModes_.size());  // the mode is restored on the failure path too
  memcpyResult_ = hipSuccess;
  captureModes_.clear();
  memcpys_.clear();
  exchangeFailAt_ = 1;
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclInt8));
  EXPECT_TRUE(memcpys_.empty());
  captureModes_.clear();
  exchangeFailAt_ = 2;  // restoring the caller's mode fails after the copy was issued
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclInt8));
  EXPECT_EQ(1u, memcpys_.size());
}

TEST_F(OnerankMicrotest, PreMulSumAlwaysReduces) {
  src_ = dst_;
  EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevPreMulSum, 0x3f800000u, true), ncclFloat32));
  EXPECT_TRUE(memcpys_.empty());
  ASSERT_EQ(1u, launches_.size());
  const Launch& l = launches_[0];
  EXPECT_EQ((Kernel<FuncPreMulSum, float>()), l.fn);
  EXPECT_EQ(dst_, l.dst);
  EXPECT_EQ(dst_, l.src);
  EXPECT_EQ(nullptr, l.acc);
  EXPECT_EQ(kElts, l.nElts);
  EXPECT_EQ(0x3f800000u, l.scalarArg);
  EXPECT_TRUE(l.scalarArgIsPtr);
  EXPECT_EQ(512u, l.block.x);
  EXPECT_EQ(1u, l.block.y);
  EXPECT_EQ(1u, l.grid.y);
  EXPECT_EQ(0u, l.shmem);
  EXPECT_EQ(stream_, l.stream);
}

TEST_F(OnerankMicrotest, BiasReducesWithEveryOp) {
  int acc = 0;
  EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevSum, 7), ncclInt32, &acc));
  ASSERT_EQ(1u, launches_.size());
  EXPECT_EQ((Kernel<FuncSum, int32_t>()), launches_[0].fn);
  EXPECT_EQ(&acc, launches_[0].acc);
  EXPECT_EQ(src_, launches_[0].src);
  EXPECT_EQ(7u, launches_[0].scalarArg);
  EXPECT_FALSE(launches_[0].scalarArgIsPtr);
  EXPECT_TRUE(memcpys_.empty());
}

// SumPostDiv divides by nRanks == 1, so it runs as Sum.
TEST_F(OnerankMicrotest, KernelTableCoversEveryOpAndType) {
  struct Case {
    ncclDataType_t type;
    std::vector<const void*> kernels;  // Sum, Prod, MinMax, PreMulSum, SumPostDiv
  };
#define KERNELS(T) {Kernel<FuncSum, T>(), Kernel<FuncProd, T>(), Kernel<FuncMinMax, T>(), \
                    Kernel<FuncPreMulSum, T>(), Kernel<FuncSum, T>()}
  const std::vector<Case> cases = {
      {ncclInt8, KERNELS(int8_t)},
      {ncclUint8, KERNELS(uint8_t)},
      {ncclInt32, KERNELS(int32_t)},
      {ncclUint32, KERNELS(uint32_t)},
      {ncclInt64, KERNELS(int64_t)},
      {ncclUint64, KERNELS(uint64_t)},
      {ncclFloat8e4m3, KERNELS(OneRankFloat8e4m3)},
      {ncclFloat8e5m2, KERNELS(OneRankFloat8e5m2)},
      {ncclFloat16, KERNELS(half)},
      {ncclBfloat16, KERNELS(hip_bfloat16)},
      {ncclFloat32, KERNELS(float)},
      {ncclFloat64, KERNELS(double)},
  };
#undef KERNELS
  const ncclDevRedOp_t ops[] = {ncclDevSum, ncclDevProd, ncclDevMinMax, ncclDevPreMulSum, ncclDevSumPostDiv};
  int acc = 0;
  for (const Case& tc : cases) {
    for (size_t i = 0; i < 5; i++) {
      SCOPED_TRACE(std::to_string(tc.type) + "/" + std::to_string(ops[i]));
      launches_.clear();
      ASSERT_EQ(ncclSuccess, LaunchOneRank(Op(ops[i]), tc.type, &acc));
      ASSERT_EQ(1u, launches_.size());
      EXPECT_EQ(tc.kernels[i], launches_[0].fn);
    }
  }
  // The table maps to distinct instantiations, not one kernel for everything.
  EXPECT_NE((Kernel<FuncSum, float>()), (Kernel<FuncProd, float>()));
  EXPECT_NE((Kernel<FuncSum, float>()), (Kernel<FuncSum, double>()));
}

TEST_F(OnerankMicrotest, UnknownOpOrTypeIsInvalid) {
  int acc = 0;
  EXPECT_EQ(ncclInvalidArgument, LaunchOneRank(Op(ncclNumDevRedOps), ncclFloat32, &acc));
  EXPECT_EQ(ncclInvalidArgument, LaunchOneRank(Op(ncclNumDevRedOps), ncclInt8, &acc));
  EXPECT_EQ(ncclInvalidArgument, LaunchOneRank(Op(ncclDevSum), ncclNumTypes, &acc));
  EXPECT_EQ(ncclInvalidArgument, LaunchOneRank(Op(ncclDevPreMulSum), ncclNumTypes));
  EXPECT_TRUE(launches_.empty());
}

// One block per 16 KiB, at most 32.
TEST_F(OnerankMicrotest, GridGrowsWithTheMessageUpTo32Blocks) {
  int acc = 0;
  const std::vector<std::pair<size_t, unsigned>> cases = {
      {1, 1}, {4096, 1}, {4097, 2}, {32 * 4096, 32}, {32 * 4096 + 1, 32}, {size_t(1) << 30, 32}};
  for (const auto& [nElts, blocks] : cases) {
    launches_.clear();
    ASSERT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc, nullptr, nElts));
    ASSERT_EQ(1u, launches_.size());
    EXPECT_EQ(blocks, launches_[0].grid.x) << nElts;
    EXPECT_EQ(nElts, launches_[0].nElts);
  }
}

// Without the CUDA 12.3 launch attribute the user event is recorded just before the launch.
TEST_F(OnerankMicrotest, CompletionEventIsRecordedBeforeTheLaunch) {
  int acc = 0;
  hipEvent_t event = reinterpret_cast<hipEvent_t>(0xe7);
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc, event)); });
  EXPECT_EQ((std::vector<std::string>{"record", "launch"}), trace_);
  ASSERT_EQ(1u, eventRecords_.size());
  EXPECT_EQ(event, eventRecords_[0].first);
  EXPECT_EQ(stream_, eventRecords_[0].second);
  EXPECT_TRUE(LogHas(log, "launch-completion events require CUDA 12.3"));

  trace_.clear();
  eventRecordResult_ = hipErrorInvalidHandle;
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc, event));
  EXPECT_EQ(std::vector<std::string>{"record"}, trace_);
}

TEST_F(OnerankMicrotest, NoCompletionEventNoRecord) {
  int acc = 0;
  RcclUnitTesting::ScopedDebugLogging all(-1, ~uint64_t(0));
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc)); });
  EXPECT_TRUE(eventRecords_.empty());
  EXPECT_TRUE(log.empty()) << log;
}

TEST_F(OnerankMicrotest, LaunchFailureIsAHipError) {
  int acc = 0;
  launchResult_ = hipErrorLaunchFailure;
  std::string log =
      CaptureLog([&] { EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc)); });
  EXPECT_TRUE(LogHas(log, "HIP failure"));
}

}  // namespace
