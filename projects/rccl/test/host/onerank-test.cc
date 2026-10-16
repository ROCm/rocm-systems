/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/device/onerank.cu: ncclLaunchOneRank's copy-or-reduce choice, its kernel
// table and launch geometry. The HIP launch and copy calls are seamed in this TU.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "nccl.h"
#include "debug.h"
#include "device.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {

struct LogLine {
  ncclDebugLogLevel level;
  std::string text;
};
std::vector<LogLine> g_log;
std::vector<std::string> g_trace;

struct Memcpy {
  void* dst;
  const void* src;
  size_t bytes;
  hipMemcpyKind kind;
  hipStream_t stream;
};
std::vector<Memcpy> g_memcpys;
hipError_t g_memcpyResult = hipSuccess;
std::vector<hipStreamCaptureMode> g_captureModes;
int g_exchangeFailAt = 0;

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
std::vector<Launch> g_launches;
hipError_t g_launchResult = hipSuccess;
std::vector<std::pair<hipEvent_t, hipStream_t>> g_eventRecords;
hipError_t g_eventRecordResult = hipSuccess;

}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_log.push_back({level, buf});
}

hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  g_trace.push_back("exchange");
  g_captureModes.push_back(*mode);
  if (static_cast<int>(g_captureModes.size()) == g_exchangeFailAt) return hipErrorInvalidValue;
  *mode = g_captureModes.size() % 2 ? hipStreamCaptureModeThreadLocal : hipStreamCaptureModeRelaxed;
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
  g_trace.push_back("memcpy");
  g_memcpys.push_back({dst, src, bytes, kind, stream});
  return g_memcpyResult;
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  g_trace.push_back("record");
  g_eventRecords.push_back({event, stream});
  return g_eventRecordResult;
}
hipError_t hipLaunchKernel(const void* fn, dim3 grid, dim3 block, void** args, size_t shmem, hipStream_t stream) {
  g_trace.push_back("launch");
  g_launches.push_back({fn, grid, block, shmem, stream, *static_cast<void**>(args[0]), *static_cast<void**>(args[1]),
                        *static_cast<void**>(args[2]), *static_cast<size_t*>(args[3]),
                        *static_cast<uint64_t*>(args[4]), *static_cast<bool*>(args[5])});
  return g_launchResult;
}
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }

// The UUT defines its kernels; the target embeds an empty device binary, so their registration lands here.
extern "C" void** __hipRegisterFatBinary(const void*) {
  static void* handle = nullptr;
  return &handle;
}
extern "C" void __hipRegisterFunction(void**, const void*, char*, const char*, unsigned int, uint3*, uint3*, dim3*,
                                      dim3*, int*) {}
extern "C" void __hipUnregisterFatBinary(void**) {}
// clang >= 23 registers per-kernel profile shadows under -fprofile-instr-generate; these keep the ROCm
// profile runtime out.
extern "C" void __hipRegisterVar(void**, char*, char*, const char*, int, size_t, int, int) {}
extern "C" __attribute__((weak)) void __llvm_profile_offload_register_shadow_variable(void*) {}
extern "C" __attribute__((weak)) void __llvm_profile_offload_register_section_shadow_variable(void*) {}
hipError_t __hipPopCallConfiguration(dim3*, dim3*, size_t*, hipStream_t*) { return hipErrorNotSupported; }

// device/common.h is the kernel-side runtime (shared memory, work batches) and does not compile host-only;
// onerank.cu only needs the reduceCopy and RedOp templates common_kernel.h layers on top of it.
#define NCCL_DEVICE_COMMON_H_
#include ONERANK_CC_PATH

namespace {

constexpr size_t kElts = 100;

bool Logged(ncclDebugLogLevel level, const std::string& needle) {
  for (const LogLine& line : g_log) {
    if (line.level == level && line.text.find(needle) != std::string::npos) return true;
  }
  return false;
}

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
  void SetUp() override {
    g_log.clear();
    g_trace.clear();
    g_memcpys.clear();
    g_memcpyResult = hipSuccess;
    g_captureModes.clear();
    g_exchangeFailAt = 0;
    g_launches.clear();
    g_launchResult = hipSuccess;
    g_eventRecords.clear();
    g_eventRecordResult = hipSuccess;
  }

  ncclResult_t LaunchOneRank(ncclDevRedOpFull op, ncclDataType_t type, const void* acc = nullptr,
                             hipEvent_t event = nullptr, size_t nElts = kElts) {
    return ncclLaunchOneRank(dst_, src_, nElts, op, type, stream_, event, acc);
  }

  char dstStorage_[8] = {};
  char srcStorage_[8] = {};
  void* dst_ = dstStorage_;
  void* src_ = srcStorage_;
  hipStream_t stream_ = reinterpret_cast<hipStream_t>(0x5a);
};

TEST_F(OnerankMicrotest, InPlaceWithoutBiasIsANoOp) {
  src_ = dst_;
  for (ncclDevRedOp_t op : {ncclDevSum, ncclDevProd, ncclDevMinMax, ncclDevSumPostDiv, ncclNumDevRedOps}) {
    EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(op), ncclFloat32)) << op;
  }
  EXPECT_TRUE(g_trace.empty());
}

TEST_F(OnerankMicrotest, OutOfPlaceWithoutBiasIsOneRelaxedCopy) {
  EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevMinMax), ncclFloat64));
  EXPECT_EQ((std::vector<std::string>{"exchange", "memcpy", "exchange"}), g_trace);
  ASSERT_EQ(1u, g_memcpys.size());
  EXPECT_EQ(dst_, g_memcpys[0].dst);
  EXPECT_EQ(src_, g_memcpys[0].src);
  EXPECT_EQ(kElts * sizeof(double), g_memcpys[0].bytes);
  EXPECT_EQ(hipMemcpyDefault, g_memcpys[0].kind);
  EXPECT_EQ(stream_, g_memcpys[0].stream);
  // Relaxed for the copy, then the caller's mode put back.
  EXPECT_EQ((std::vector<hipStreamCaptureMode>{hipStreamCaptureModeRelaxed, hipStreamCaptureModeThreadLocal}),
            g_captureModes);
}

TEST_F(OnerankMicrotest, CopyFailuresAreHipErrors) {
  g_memcpyResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclInt8));
  EXPECT_EQ(2u, g_captureModes.size());  // the mode is restored on the failure path too
  g_memcpyResult = hipSuccess;
  g_captureModes.clear();
  g_memcpys.clear();
  g_exchangeFailAt = 1;
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclInt8));
  EXPECT_TRUE(g_memcpys.empty());
  g_captureModes.clear();
  g_exchangeFailAt = 2;  // restoring the caller's mode fails after the copy was issued
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclInt8));
  EXPECT_EQ(1u, g_memcpys.size());
}

TEST_F(OnerankMicrotest, PreMulSumAlwaysReduces) {
  src_ = dst_;
  EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevPreMulSum, 0x3f800000u, true), ncclFloat32));
  EXPECT_TRUE(g_memcpys.empty());
  ASSERT_EQ(1u, g_launches.size());
  const Launch& l = g_launches[0];
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
  ASSERT_EQ(1u, g_launches.size());
  EXPECT_EQ((Kernel<FuncSum, int32_t>()), g_launches[0].fn);
  EXPECT_EQ(&acc, g_launches[0].acc);
  EXPECT_EQ(src_, g_launches[0].src);
  EXPECT_EQ(7u, g_launches[0].scalarArg);
  EXPECT_FALSE(g_launches[0].scalarArgIsPtr);
  EXPECT_TRUE(g_memcpys.empty());
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
      g_launches.clear();
      ASSERT_EQ(ncclSuccess, LaunchOneRank(Op(ops[i]), tc.type, &acc));
      ASSERT_EQ(1u, g_launches.size());
      EXPECT_EQ(tc.kernels[i], g_launches[0].fn);
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
  EXPECT_TRUE(g_launches.empty());
}

// One block per 16 KiB, at most 32.
TEST_F(OnerankMicrotest, GridGrowsWithTheMessageUpTo32Blocks) {
  int acc = 0;
  const std::vector<std::pair<size_t, unsigned>> cases = {
      {1, 1}, {4096, 1}, {4097, 2}, {32 * 4096, 32}, {32 * 4096 + 1, 32}, {size_t(1) << 30, 32}};
  for (const auto& [nElts, blocks] : cases) {
    g_launches.clear();
    ASSERT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc, nullptr, nElts));
    ASSERT_EQ(1u, g_launches.size());
    EXPECT_EQ(blocks, g_launches[0].grid.x) << nElts;
    EXPECT_EQ(nElts, g_launches[0].nElts);
  }
}

// Without the CUDA 12.3 launch attribute the user event is recorded just before the launch.
TEST_F(OnerankMicrotest, CompletionEventIsRecordedBeforeTheLaunch) {
  int acc = 0;
  hipEvent_t event = reinterpret_cast<hipEvent_t>(0xe7);
  EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc, event));
  EXPECT_EQ((std::vector<std::string>{"record", "launch"}), g_trace);
  ASSERT_EQ(1u, g_eventRecords.size());
  EXPECT_EQ(event, g_eventRecords[0].first);
  EXPECT_EQ(stream_, g_eventRecords[0].second);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "launch-completion events require CUDA 12.3"));

  g_trace.clear();
  g_eventRecordResult = hipErrorInvalidHandle;
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc, event));
  EXPECT_EQ(std::vector<std::string>{"record"}, g_trace);
}

TEST_F(OnerankMicrotest, NoCompletionEventNoRecord) {
  int acc = 0;
  EXPECT_EQ(ncclSuccess, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc));
  EXPECT_TRUE(g_eventRecords.empty());
  EXPECT_TRUE(g_log.empty());
}

TEST_F(OnerankMicrotest, LaunchFailureIsAHipError) {
  int acc = 0;
  g_launchResult = hipErrorLaunchFailure;
  EXPECT_EQ(ncclUnhandledCudaError, LaunchOneRank(Op(ncclDevSum), ncclFloat32, &acc));
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "HIP failure"));
}

}  // namespace
