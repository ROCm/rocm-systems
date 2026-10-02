/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Single-GPU device-leaf tests for the GIN proxy backend.

#include "DeviceTestBase.hpp"

// gin_proxy.h declares specializations of templates (ncclGinApi_Put, ...) and
// references ncclGinCtx / ncclGinDescriptorSmem; their primary declarations
// live in gin_device_common.h, which must be included first.
#include "nccl_device/coop.h"
#include "nccl_device/impl/core__funcs.h" 
#include "nccl_device/impl/gin__funcs.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/proxy/gin_proxy.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <vector>

namespace RcclUnitTesting
{

class GinDeviceTest : public DeviceTestBase {};

struct SignalActionResult {
  ncclGinSignalDescriptor descriptor;
  ncclGinSignalOp_t       op;
  uint64_t                arg;
};

template<typename Action>
__device__ void captureSignalAction(ncclGin const& gin, Action action,
                                    SignalActionResult* result) {
  result->descriptor = ncclGin_getSignalDescriptor(gin, action);
  result->op = ncclGin_getSignalOp(action);
  result->arg = ncclGin_getSignalOpArg(action);
}

__global__ void kernelExplicitSignalActions(
    ncclDevComm comm, ncclWindow_t signalWindow, SignalActionResult* results) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  ncclGin gin{comm, /*contextIndex=*/0};
  captureSignalAction(gin, ncclGin_StrongSignalInc{3}, &results[0]);
  captureSignalAction(gin, ncclGin_StrongSignalAdd{4, 0x1234}, &results[1]);
  captureSignalAction(gin, ncclGin_WeakSignalInc{5}, &results[2]);
  captureSignalAction(gin, ncclGin_WeakSignalAdd{6, 0x5678}, &results[3]);
  captureSignalAction(gin, ncclGin_StrongVASignalInc{signalWindow, 64}, &results[4]);
  captureSignalAction(gin, ncclGin_StrongVASignalAdd{signalWindow, 72, 0x9abc}, &results[5]);
  captureSignalAction(gin, ncclGin_WeakVASignalInc{signalWindow, 80}, &results[6]);
  captureSignalAction(gin, ncclGin_WeakVASignalAdd{signalWindow, 88, 0xdef0}, &results[7]);

  comm.ginStrongLegacySignals = false;
  ncclGin legacyWeak{comm, /*contextIndex=*/0};
  captureSignalAction(legacyWeak, ncclGin_SignalInc{7}, &results[8]);
  captureSignalAction(legacyWeak, ncclGin_VASignalAdd{signalWindow, 96, 0x1111}, &results[9]);

  comm.ginStrongLegacySignals = true;
  ncclGin legacyStrong{comm, /*contextIndex=*/0};
  captureSignalAction(legacyStrong, ncclGin_SignalAdd{8, 0x2222}, &results[10]);
  captureSignalAction(legacyStrong, ncclGin_VASignalInc{signalWindow, 104}, &results[11]);
  captureSignalAction(legacyStrong, ncclGin_WeakSignalAdd{9, 0x3333}, &results[12]);
}

TEST_F(GinDeviceTest, ExplicitSignalActions) {
  constexpr size_t kResultCount = 13;
  constexpr uint32_t kGinOffset4K = 3;
  constexpr uintptr_t kGinWindowToken = 0x12345000;

  DeviceBuffer<uint64_t> d_signalShadows(1);
  DeviceBuffer<ncclWindow_vidmem> d_signalWindow(1);
  DeviceBuffer<SignalActionResult> d_results(kResultCount);

  ncclWindow_vidmem hostWindow{};
  hostWindow.ginOffset4K = kGinOffset4K;
  hostWindow.ginWinsDefaultBackend[0] =
      reinterpret_cast<ncclGinWindow_t>(kGinWindowToken);
  hostWindow.numSegments = 1;
  d_signalWindow.upload(hostWindow);
  d_results.zero();

  ncclDevComm comm{};
  comm.ginConnectionCount = 1;
  comm.backendIndex = 0;
  comm.ginNetDeviceTypes[0] = NCCL_NET_DEVICE_GIN_PROXY;
  comm.ginSignalCount = 1;
  comm.ginSignalShadows = d_signalShadows.ptr;

  kernelExplicitSignalActions<<<1, 1>>>(
      comm, d_signalWindow.ptr, d_results.ptr);
  syncAndCheck();

  const std::vector<SignalActionResult> results = d_results.copyTo();

  struct ExpectedSignal {
    bool strong;
    ncclGinSignalOp_t op;
    uint64_t arg;
    ncclGinSignalType type;
    ncclGinSignal_t signalId;
    size_t vaOffset;
  };
  const ExpectedSignal cases[] = {
      {true,  ncclGinSignalInc, 1,      NCCL_GIN_SIGNAL_TYPE_INDEXED, 3, 0},
      {true,  ncclGinSignalAdd, 0x1234, NCCL_GIN_SIGNAL_TYPE_INDEXED, 4, 0},
      {false, ncclGinSignalInc, 1,      NCCL_GIN_SIGNAL_TYPE_INDEXED, 5, 0},
      {false, ncclGinSignalAdd, 0x5678, NCCL_GIN_SIGNAL_TYPE_INDEXED, 6, 0},
      {true,  ncclGinSignalInc, 1,      NCCL_GIN_SIGNAL_TYPE_VA,      0, 64},
      {true,  ncclGinSignalAdd, 0x9abc, NCCL_GIN_SIGNAL_TYPE_VA,      0, 72},
      {false, ncclGinSignalInc, 1,      NCCL_GIN_SIGNAL_TYPE_VA,      0, 80},
      {false, ncclGinSignalAdd, 0xdef0, NCCL_GIN_SIGNAL_TYPE_VA,      0, 88},
      {false, ncclGinSignalInc, 1,      NCCL_GIN_SIGNAL_TYPE_INDEXED, 7, 0},
      {false, ncclGinSignalAdd, 0x1111, NCCL_GIN_SIGNAL_TYPE_VA,      0, 96},
      {true,  ncclGinSignalAdd, 0x2222, NCCL_GIN_SIGNAL_TYPE_INDEXED, 8, 0},
      {true,  ncclGinSignalInc, 1,      NCCL_GIN_SIGNAL_TYPE_VA,      0, 104},
      {false, ncclGinSignalAdd, 0x3333, NCCL_GIN_SIGNAL_TYPE_INDEXED, 9, 0},
  };
  static_assert(sizeof(cases) / sizeof(cases[0]) == kResultCount,
                "one expected row per captured signal action");

  for (size_t i = 0; i < kResultCount; ++i) {
    const auto& expected = cases[i];
    const auto& descriptor = results[i].descriptor;
    EXPECT_EQ(descriptor.isStrong, expected.strong) << "case " << i;
    EXPECT_EQ(results[i].op, expected.op) << "case " << i;
    EXPECT_EQ(results[i].arg, expected.arg) << "case " << i;
    EXPECT_EQ(descriptor.type, expected.type) << "case " << i;
    if (expected.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
      EXPECT_EQ(descriptor.indexedSignal.signalId, expected.signalId) << "case " << i;
    } else {
      EXPECT_EQ(descriptor.vaSignal.signalWindow,
                reinterpret_cast<ncclGinWindow_t>(kGinWindowToken)) << "case " << i;
      EXPECT_EQ(descriptor.vaSignal.signalOffset,
                4096 * kGinOffset4K + expected.vaOffset) << "case " << i;
      EXPECT_EQ(descriptor.vaSignal.ncclWindow, d_signalWindow.ptr) << "case " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// ConstructProxyOp: pack (hasInline, hasSignal, signalOp, hasCounter) -> op byte
// ---------------------------------------------------------------------------

struct OpCase {
  bool     hasInline;
  bool     hasSignal;
  uint32_t signalOp;   // unused when hasSignal == false
  bool     hasCounter;
};

// Host-side reference encoding.
static uint8_t expectedOp(const OpCase& c) {
  uint8_t op = static_cast<uint8_t>(ncclGinProxyOpPut);
  if (c.hasInline)  op |= static_cast<uint8_t>(ncclGinProxyOpWithInline);
  if (c.hasCounter) op |= static_cast<uint8_t>(ncclGinProxyOpWithCounter);
  if (c.hasSignal) {
    if (c.signalOp == static_cast<uint32_t>(ncclGinSignalInc))
      op |= static_cast<uint8_t>(ncclGinProxyOpWithSignalInc);
    else
      op |= static_cast<uint8_t>(ncclGinProxyOpWithSignalAdd);
  }
  return op;
}

__global__ void kernelConstructProxyOp(const OpCase* __restrict__ in,
                                       uint8_t* __restrict__ out,
                                       int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  ncclGinProxyOp_t op;
  nccl::gin::proxy::constructProxyOp(
      op,
      /*isGet=*/false,
      /*isFlush=*/false,
      in[i].hasInline,
      in[i].hasSignal ? NCCL_GIN_SIGNAL_TYPE_INDEXED : NCCL_GIN_SIGNAL_TYPE_NONE,
      static_cast<ncclGinSignalOp_t>(in[i].signalOp),
      in[i].hasCounter);
  out[i] = static_cast<uint8_t>(op);
}

TEST_F(GinDeviceTest, ConstructProxyOp) {
  // 12 effective cases: 2 (hasInline) * 2 (hasCounter) * 3 (no-signal | Inc | Add).
  std::vector<OpCase> cases;
  cases.reserve(12);
  for (bool hasInline : {false, true}) {
    for (bool hasCounter : {false, true}) {
      cases.push_back({hasInline, /*hasSignal=*/false,
                       static_cast<uint32_t>(ncclGinSignalInc), hasCounter});
      cases.push_back({hasInline, /*hasSignal=*/true,
                       static_cast<uint32_t>(ncclGinSignalInc), hasCounter});
      cases.push_back({hasInline, /*hasSignal=*/true,
                       static_cast<uint32_t>(ncclGinSignalAdd), hasCounter});
    }
  }
  const int N = static_cast<int>(cases.size());
  ASSERT_EQ(N, 12);

  DeviceBuffer<OpCase>  d_in(N);
  DeviceBuffer<uint8_t> d_out(N);
  d_in.copyFrom(cases);
  d_out.zero();

  kernelConstructProxyOp<<<gridFor(N), kDefaultBlockSize>>>(d_in.ptr, d_out.ptr, N);
  syncAndCheck();

  std::vector<uint8_t> h_out = d_out.copyTo();
  for (int i = 0; i < N; i++) {
    const uint8_t got = h_out[i];
    const uint8_t exp = expectedOp(cases[i]);
    EXPECT_EQ(got, exp)
      << "case " << i
      << ": hasInline="  << cases[i].hasInline
      << ", hasSignal="  << cases[i].hasSignal
      << ", signalOp="   << cases[i].signalOp
      << ", hasCounter=" << cases[i].hasCounter
      << "; got=0x"      << std::hex << static_cast<int>(got)
      << ", expected=0x" << std::hex << static_cast<int>(exp);
  }

  // All-flags-off case must equal Put exactly (no stray bits OR'd in).
  for (int i = 0; i < N; i++) {
    if (!cases[i].hasInline && !cases[i].hasSignal && !cases[i].hasCounter) {
      EXPECT_EQ(h_out[i], static_cast<uint8_t>(ncclGinProxyOpPut))
        << "Base case (no flags) at index " << i << " is not ncclGinProxyOpPut";
    }
  }
}

// ---------------------------------------------------------------------------
// BuildGfd_PutOnly: pack a 128-byte GFD for a non-inline put (no signal, no counter)
// ---------------------------------------------------------------------------

__global__ void kernelBuildGfdPutOnly(ncclGinProxyGfd_t* gfd,
                                      uint64_t srcOff, uint64_t srcHandle,
                                      uint64_t dstOff, uint64_t dstHandle,
                                      uint64_t size) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  nccl::gin::proxy::buildGfd<uint64_t>(
      gfd,
      ncclGinProxyOpPut,
      /*srcVal=*/0ULL,                                   // unused (hasInline=false)
      /*hasInline=*/false,
      /*srcOff=*/srcOff,
      /*srcHandle=*/reinterpret_cast<ncclGinWindow_t>(srcHandle),
      /*dstOff=*/dstOff,
      /*dstHandle=*/reinterpret_cast<ncclGinWindow_t>(dstHandle),
      /*size=*/size,
      /*counterId=*/0,
      /*signalId=*/0,
      /*signalVal=*/0,
      /*signalWindow=*/nullptr,
      /*signalOff=*/0);
}

TEST_F(GinDeviceTest, BuildGfd_PutOnly) {
  static_assert(sizeof(ncclGinProxyGfd_t) == 128, "GFD must be 128 bytes");

  // Distinctive 48-bit values so bit-corruption in any qword is visible.
  // Each constant is a rotation of 0x0123456789ABCDEF truncated to 48 bits,
  // so every byte differs across the four constants.
  constexpr uint64_t kSrcOff    = 0x0000123456789ABCULL;
  constexpr uint64_t kSrcHandle = 0x000056789ABCDEF0ULL;
  constexpr uint64_t kDstOff    = 0x00009ABCDEF01234ULL;
  constexpr uint64_t kDstHandle = 0x0000DEF012345678ULL;
  constexpr uint64_t kSize      = 4096;

  DeviceBuffer<ncclGinProxyGfd_t> d_gfd(1);
  d_gfd.zero();   // start from zeros so we know exactly what buildGfd wrote

  kernelBuildGfdPutOnly<<<1, 1>>>(d_gfd.ptr, kSrcOff, kSrcHandle, kDstOff, kDstHandle, kSize);
  syncAndCheck();

  const ncclGinProxyGfd_t gfd = d_gfd.download();

  // Header qword: flag, size (op now lives in the headerExt qword).
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.version),
            static_cast<uint64_t>(NCCL_GIN_PROXY_GFD_VERSION));
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeaderExt].headerExt.op),
            static_cast<uint64_t>(ncclGinProxyOpPut));
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.size), kSize);

  // Source address qwords (only valid in the hasInline=false branch).
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcOff].srcOff.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcOff].srcOff.srcOff), kSrcOff);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcHandle].srcHandle.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle),
            kSrcHandle);

  // Destination address qwords.
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.dstOff), kDstOff);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle),
            kDstHandle);

  // Completion qword: flag set, ids and signal-value-low zero (no signal/counter).
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.flag), 1ULL);
  EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.counterId), 0u);
  EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.signalId), 0u);
  EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.signalValLow), 0u);

  // SignalVal qword: flag set, high halves of signalVal zero (no signal).
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.flag), 1ULL);
  EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2), 0u);
  EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh), 0u);

  // buildGfd flags every qword; the trailing qword is the end-of-GFD marker.
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdQwords - 1].flag.v), 1ULL);
}

// ---------------------------------------------------------------------------
// BuildGfd_Inline: pack a 128-byte GFD for an inline put (hasInline=true).
//   T=uint32_t -> only inlineValLow holds data; sizeof(T)>4/>6 branches stay quiet.
//   T=uint64_t -> value splits as 32 (Low) + 16 (Low2) + 16 (High) bits.
// ---------------------------------------------------------------------------

template<typename T>
__global__ void kernelBuildGfdInline(ncclGinProxyGfd_t* gfd, T srcVal,
                                     uint64_t dstOff, uint64_t dstHandle) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  nccl::gin::proxy::buildGfd<T>(
      gfd,
      ncclGinProxyOpPut,
      srcVal,
      /*hasInline=*/true,
      /*srcOff=*/0,                                        // unused (hasInline=true)
      /*srcHandle=*/reinterpret_cast<ncclGinWindow_t>(0),  // unused (hasInline=true)
      /*dstOff=*/dstOff,
      /*dstHandle=*/reinterpret_cast<ncclGinWindow_t>(dstHandle),
      /*size=*/sizeof(T),
      /*counterId=*/0,
      /*signalId=*/0,
      /*signalVal=*/0,
      /*signalWindow=*/nullptr,
      /*signalOff=*/0);
}

TEST_F(GinDeviceTest, BuildGfd_Inline) {
  static_assert(sizeof(ncclGinProxyGfd_t) == 128, "GFD must be 128 bytes");

  constexpr uint64_t kDstOff    = 0x00009ABCDEF01234ULL;
  constexpr uint64_t kDstHandle = 0x0000DEF012345678ULL;
  constexpr uint32_t kValU32    = 0xA5B6C7D8u;
  constexpr uint64_t kValU64    = 0x0123456789ABCDEFULL;

  // ---- uint32_t (4-byte): only inlineValLow set; Low2 and High stay zero ----
  {
    DeviceBuffer<ncclGinProxyGfd_t> d_gfd(1);
    d_gfd.zero();

    kernelBuildGfdInline<uint32_t><<<1, 1>>>(d_gfd.ptr, kValU32, kDstOff, kDstHandle);
    syncAndCheck();

    const ncclGinProxyGfd_t gfd = d_gfd.download();

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.flag), 1ULL) << "u32";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeaderExt].headerExt.op),
              static_cast<uint64_t>(ncclGinProxyOpPut)) << "u32";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.size),
              static_cast<uint64_t>(sizeof(uint32_t))) << "u32";

    // Inline qwords (slots 1+2): Low holds the whole 32-bit value; Low2 and High must stay 0.
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.flag),    1u) << "u32";
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow,    kValU32) << "u32";
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2,   0u) << "u32 must skip Low2 (sizeof<=4)";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdInlineHigh].inlineHigh.flag), 1u) << "u32";
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh, 0u) << "u32 must skip High (sizeof<=6)";

    // Destination address qwords.
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.flag),       1ULL) << "u32";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.dstOff),     kDstOff) << "u32";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.flag), 1ULL) << "u32";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle),
              kDstHandle) << "u32";

    // Completion / SignalVal: no signal, no counter -> id and value fields zero.
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.flag),         1ULL) << "u32";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.counterId),    0u)   << "u32";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.signalId),     0u)   << "u32";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.signalValLow), 0u)   << "u32";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.flag),           1ULL) << "u32";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2),  0u)   << "u32";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh),  0u)   << "u32";

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdQwords - 1].flag.v), 1ULL) << "u32";
  }

  // ---- uint64_t (8-byte): value splits across Low (32) + Low2 (16) + High (16) ----
  {
    DeviceBuffer<ncclGinProxyGfd_t> d_gfd(1);
    d_gfd.zero();

    kernelBuildGfdInline<uint64_t><<<1, 1>>>(d_gfd.ptr, kValU64, kDstOff, kDstHandle);
    syncAndCheck();

    const ncclGinProxyGfd_t gfd = d_gfd.download();

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.flag), 1ULL) << "u64";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeaderExt].headerExt.op),
              static_cast<uint64_t>(ncclGinProxyOpPut)) << "u64";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.size),
              static_cast<uint64_t>(sizeof(uint64_t))) << "u64";

    // Inline split: Low = bits[0:32), Low2 = bits[32:48), High = bits[48:64).
    constexpr uint32_t expectLow  = static_cast<uint32_t>(kValU64);              // 0x89ABCDEF
    constexpr uint16_t expectLow2 = static_cast<uint16_t>(kValU64 >> 32);        // 0x4567
    constexpr uint16_t expectHigh = static_cast<uint16_t>(kValU64 >> 48);        // 0x0123
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.flag),    1u) << "u64";
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow,    expectLow)  << "u64";
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2,   expectLow2) << "u64";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdInlineHigh].inlineHigh.flag), 1u) << "u64";
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh, expectHigh) << "u64";

    // Round-trip: reassemble the 64-bit value from the three pieces.
    const uint64_t roundtrip =
        static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow) |
        (static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2) << 32) |
        (static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh) << 48);
    EXPECT_EQ(roundtrip, kValU64) << "u64 inline value round-trip";

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.flag),       1ULL) << "u64";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.dstOff),     kDstOff) << "u64";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.flag), 1ULL) << "u64";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle),
              kDstHandle) << "u64";

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.flag),         1ULL) << "u64";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.counterId),    0u)   << "u64";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.signalId),     0u)   << "u64";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.signalValLow), 0u)   << "u64";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.flag),           1ULL) << "u64";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2),  0u)   << "u64";
    EXPECT_EQ(static_cast<uint32_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh),  0u)   << "u64";

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdQwords - 1].flag.v), 1ULL) << "u64";
  }
}

// ---------------------------------------------------------------------------
// BuildGfd_SignalAndCounter: signalId + counterId set; full 64-bit signalVal split as
//   completion.signalValLow  = bits[ 0:16)
//   signalVal.signalValLow2  = bits[16:32)
//   signalVal.signalValHigh  = bits[32:64)
// ---------------------------------------------------------------------------

__global__ void kernelBuildGfdSignalAndCounter(ncclGinProxyGfd_t* gfd,
                                               uint64_t srcOff, uint64_t srcHandle,
                                               uint64_t dstOff, uint64_t dstHandle,
                                               uint64_t size,
                                               uint32_t counterId, uint32_t signalId,
                                               uint64_t signalVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  nccl::gin::proxy::buildGfd<uint64_t>(
      gfd,
      static_cast<ncclGinProxyOp_t>(static_cast<uint32_t>(ncclGinProxyOpPut) |
                                    static_cast<uint32_t>(ncclGinProxyOpWithSignalAdd) |
                                    static_cast<uint32_t>(ncclGinProxyOpWithCounter)),
      /*srcVal=*/0ULL,                                   // unused (hasInline=false)
      /*hasInline=*/false,
      /*srcOff=*/srcOff,
      /*srcHandle=*/reinterpret_cast<ncclGinWindow_t>(srcHandle),
      /*dstOff=*/dstOff,
      /*dstHandle=*/reinterpret_cast<ncclGinWindow_t>(dstHandle),
      /*size=*/size,
      /*counterId=*/counterId,
      /*signalId=*/signalId,
      /*signalVal=*/signalVal,
      /*signalWindow=*/nullptr,
      /*signalOff=*/0);
}

TEST_F(GinDeviceTest, BuildGfd_SignalAndCounter) {
  constexpr uint64_t kSrcOff     = 0x0000123456789ABCULL;
  constexpr uint64_t kSrcHandle  = 0x000056789ABCDEF0ULL;
  constexpr uint64_t kDstOff     = 0x00009ABCDEF01234ULL;
  constexpr uint64_t kDstHandle  = 0x0000DEF012345678ULL;
  constexpr uint64_t kSize       = 4096;
  constexpr uint16_t kSignalId   = 0x1111;
  constexpr uint16_t kCounterId  = 0x2222;
  constexpr uint64_t kSignalVal  = 0x0123456789ABCDEFULL;
  constexpr uint16_t kSigValLow  = static_cast<uint16_t>(kSignalVal);          // 0xCDEF
  constexpr uint16_t kSigValLow2 = static_cast<uint16_t>(kSignalVal >> 16);    // 0x89AB
  constexpr uint32_t kSigValHigh = static_cast<uint32_t>(kSignalVal >> 32);    // 0x01234567

  DeviceBuffer<ncclGinProxyGfd_t> d_gfd(1);
  d_gfd.zero();

  kernelBuildGfdSignalAndCounter<<<1, 1>>>(d_gfd.ptr,
                                           kSrcOff, kSrcHandle,
                                           kDstOff, kDstHandle,
                                           kSize,
                                           kCounterId, kSignalId,
                                           kSignalVal);
  syncAndCheck();

  const ncclGinProxyGfd_t gfd = d_gfd.download();

  // Header: buildGfd stores the op byte unchanged (verify with the same OR we passed in).
  const uint64_t expectedOp = static_cast<uint64_t>(ncclGinProxyOpPut) |
                              static_cast<uint64_t>(ncclGinProxyOpWithSignalAdd) |
                              static_cast<uint64_t>(ncclGinProxyOpWithCounter);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeaderExt].headerExt.op),   expectedOp);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.size), kSize);

  // Source + destination address qwords (non-inline path).
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcOff].srcOff.flag),       1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcOff].srcOff.srcOff),     kSrcOff);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcHandle].srcHandle.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle),
            kSrcHandle);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.flag),       1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstOff].dstOff.dstOff),     kDstOff);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle),
            kDstHandle);

  // Completion qword: ids and low 16 bits of signalVal.
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.flag), 1ULL);
  EXPECT_EQ(gfd.qword[ncclGinProxyGfdCompletion].completion.counterId,    kCounterId);
  EXPECT_EQ(gfd.qword[ncclGinProxyGfdCompletion].completion.signalId,     kSignalId);
  EXPECT_EQ(gfd.qword[ncclGinProxyGfdCompletion].completion.signalValLow, kSigValLow);

  // SignalVal qword: bits[16:32) and bits[32:64) of signalVal.
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.flag), 1ULL);
  EXPECT_EQ(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2, kSigValLow2);
  EXPECT_EQ(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh, kSigValHigh);

  // Round-trip: reassemble the 64-bit signal value from the 16+16+32 pieces.
  const uint64_t signalValRoundtrip =
      static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdCompletion].completion.signalValLow) |
      (static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2) << 16) |
      (static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh) << 32);
  EXPECT_EQ(signalValRoundtrip, kSignalVal) << "signalVal 16+16+32 split round-trip";

  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdQwords - 1].flag.v), 1ULL);
}

__global__ void kernelBuildGfdSignalStrength(ncclGinProxyGfd_t* gfds, uint64_t signalVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int i = 0; i < 2; ++i) {
    nccl::gin::proxy::buildGfd<uint64_t>(
        &gfds[i],
        static_cast<ncclGinProxyOp_t>(
            static_cast<uint32_t>(ncclGinProxyOpPut) |
            static_cast<uint32_t>(ncclGinProxyOpWithSignalInc)),
        /*srcVal=*/0, /*hasInline=*/false,
        /*srcOff=*/0, /*srcHandle=*/nullptr,
        /*dstOff=*/0, /*dstHandle=*/nullptr,
        /*size=*/64, /*counterId=*/0, /*signalId=*/1,
        /*signalVal=*/signalVal, /*signalWindow=*/nullptr, /*signalOff=*/0,
        /*isStrongSignal=*/i == 1);
  }
}

TEST_F(GinDeviceTest, BuildGfd_SignalStrength) {
  // Exercise every field sharing the qword with isStrongSignal.
  constexpr uint64_t kSignalVal = 0x0123456789ABCDEFULL;
  constexpr uint16_t kSigValLow2 = static_cast<uint16_t>(kSignalVal >> 16);
  constexpr uint32_t kSigValHigh = static_cast<uint32_t>(kSignalVal >> 32);

  DeviceBuffer<ncclGinProxyGfd_t> d_gfds(2);
  d_gfds.zero();

  kernelBuildGfdSignalStrength<<<1, 1>>>(d_gfds.ptr, kSignalVal);
  syncAndCheck();

  const std::vector<ncclGinProxyGfd_t> gfds = d_gfds.copyTo();
  for (int i = 0; i < 2; ++i) {
    const auto& signalVal = gfds[i].qword[ncclGinProxyGfdSignalVal].signalVal;
    EXPECT_EQ(signalVal.flag, 1u) << i;
    EXPECT_EQ(signalVal.resv, 0u) << i;
    EXPECT_EQ(signalVal.isStrongSignal, static_cast<uint32_t>(i)) << i;
    EXPECT_EQ(signalVal.signalValLow2, kSigValLow2) << i;
    EXPECT_EQ(signalVal.signalValHigh, kSigValHigh) << i;
  }
}

// ---------------------------------------------------------------------------
// BuildGfd_SizeClasses: hasInline=true with T in {u8, u16, u32, u64}.
//   The `sizeof(T) > 4` and `sizeof(T) > 6` branches in buildGfd must stay quiet
//   for T <= 4 bytes (Low2 and High remain 0) and must fire for T = u64.
// ---------------------------------------------------------------------------

template<typename T>
__global__ void kernelBuildGfdSizeClass(ncclGinProxyGfd_t* gfd, T srcVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  nccl::gin::proxy::buildGfd<T>(
      gfd,
      ncclGinProxyOpPut,
      srcVal,
      /*hasInline=*/true,
      /*srcOff=*/0,                                        // unused
      /*srcHandle=*/reinterpret_cast<ncclGinWindow_t>(0),  // unused
      /*dstOff=*/0,
      /*dstHandle=*/reinterpret_cast<ncclGinWindow_t>(0),
      /*size=*/sizeof(T),
      /*counterId=*/0,
      /*signalId=*/0,
      /*signalVal=*/0,
      /*signalWindow=*/nullptr,
      /*signalOff=*/0);
}

TEST_F(GinDeviceTest, BuildGfd_SizeClasses) {
  // Each srcVal sets the upper bits non-zero so any spurious write to Low2/High
  // (which must stay 0 for T <= 4 bytes) would surface as a non-zero readback.
  // Expected Low2/High depend on sizeof(T): they must remain 0 unless the
  // sizeof(T)>4 / sizeof(T)>6 branches fire (i.e. only for uint64_t).
  auto run = [this](auto srcVal, const char* label) {
    using T = decltype(srcVal);

    DeviceBuffer<ncclGinProxyGfd_t> d_gfd(1);
    d_gfd.zero();
    kernelBuildGfdSizeClass<T><<<1, 1>>>(d_gfd.ptr, srcVal);
    syncAndCheck();
    const ncclGinProxyGfd_t gfd = d_gfd.download();

    const uint16_t expectLow2 = (sizeof(T) > 4)
        ? static_cast<uint16_t>(static_cast<uint64_t>(srcVal) >> 32) : uint16_t{0};
    const uint16_t expectHigh = (sizeof(T) > 6)
        ? static_cast<uint16_t>(static_cast<uint64_t>(srcVal) >> 48) : uint16_t{0};

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.size),
              static_cast<uint64_t>(sizeof(T)))                                       << label;
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow,
              static_cast<uint32_t>(srcVal))                                          << label;
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2,
              expectLow2)                                                             << label << " Low2";
    EXPECT_EQ(gfd.qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh,
              expectHigh)                                                             << label << " High";
  };

  run(uint8_t {0xA5u},                  "u8");
  run(uint16_t{0xA5B6u},                "u16");
  run(uint32_t{0xA5B6C7D8u},            "u32");
  run(uint64_t{0xA5B6C7D8E9FA0B1CULL},  "u64");
}

// ---------------------------------------------------------------------------
// PostGfd_OneSlot: a single postGfd writes the input GFD into slot 0 of peer 0's
//   ring, advances pis[peer] from 0 -> 1, and leaves cis untouched.
// ---------------------------------------------------------------------------

__global__ void kernelPostGfdOneSlot(ncclGinProxyGpuCtx_t* ctx,
                                     ncclGinProxyGfd_t* gfdIn,
                                     uint32_t peer) {
  nccl::gin::proxy::postGfd(ncclCoopCta{}, ctx, gfdIn, peer);
}

TEST_F(GinDeviceTest, PostGfd_OneSlot) {
  static_assert(sizeof(ncclGinProxyGfd_t) == 128, "GFD must be 128 bytes");

  constexpr uint32_t kNranks    = 2;
  constexpr uint32_t kQueueSize = 4;   // power of two: postGfd uses idx & (queueSize-1)
  constexpr uint32_t kPeer      = 0;

  // Device-side ring + PI/CI arrays + input GFD + the ctx struct itself.
  DeviceBuffer<ncclGinProxyGfd_t>     d_queues(kNranks * kQueueSize);
  DeviceBuffer<uint32_t>              d_pis(kNranks);
  DeviceBuffer<uint32_t>              d_cis(kNranks);
  DeviceBuffer<ncclGinProxyGfd_t>     d_gfdIn(1);
  DeviceBuffer<ncclGinProxyGpuCtx_t>  d_ctx(1);

  d_queues.zero();
  d_pis.zero();
  d_cis.zero();

  // Input GFD: byte ramp 0x10.. across the whole descriptor (unique non-zero
  // bytes). Any byte the kernel fails to write will read back as 0,
  // distinguishable from the input.
  ncclGinProxyGfd_t hostGfd{};
  uint8_t* gfdBytes = reinterpret_cast<uint8_t*>(&hostGfd);
  for (size_t i = 0; i < sizeof(ncclGinProxyGfd_t); i++) gfdBytes[i] = static_cast<uint8_t>(0x10 + i);
  d_gfdIn.upload(hostGfd);

  // Ctx struct: aliases the device buffers above. counters/signals unused by postGfd.
  ncclGinProxyGpuCtx_t hostCtx{};
  hostCtx.nranks    = static_cast<int>(kNranks);
  hostCtx.queueSize = kQueueSize;
  hostCtx.queues    = d_queues.ptr;
  hostCtx.pis       = d_pis.ptr;
  hostCtx.cis       = d_cis.ptr;
  hostCtx.counters  = nullptr;
  hostCtx.signals   = nullptr;
  d_ctx.upload(hostCtx);

  kernelPostGfdOneSlot<<<1, 1>>>(d_ctx.ptr, d_gfdIn.ptr, kPeer);
  syncAndCheck();

  std::vector<uint32_t>          pis    = d_pis.copyTo();
  std::vector<uint32_t>          cis    = d_cis.copyTo();
  std::vector<ncclGinProxyGfd_t> queues = d_queues.copyTo();

  // PI for peer 0 advanced exactly once; PI for peer 1 untouched.
  EXPECT_EQ(pis[0], 1u) << "peer 0 PI must advance 0 -> 1";
  EXPECT_EQ(pis[1], 0u) << "peer 1 PI must be untouched";

  // CIs are consumer-side; postGfd must not write them.
  EXPECT_EQ(cis[0], 0u);
  EXPECT_EQ(cis[1], 0u);

  // Slot 0 of peer 0's ring (queues[peer * queueSize + 0] = queues[0]) must hold
  // the input GFD byte-for-byte.
  const uint8_t* slotBytes = reinterpret_cast<const uint8_t*>(&queues[0]);
  for (size_t i = 0; i < sizeof(ncclGinProxyGfd_t); i++) {
    EXPECT_EQ(slotBytes[i], gfdBytes[i])
        << "slot 0 byte " << i << " mismatch";
  }

  // No other slot was written: peer 0 slots 1..3, plus all of peer 1.
  for (size_t s = 1; s < queues.size(); s++) {
    const uint8_t* zeroBytes = reinterpret_cast<const uint8_t*>(&queues[s]);
    for (size_t i = 0; i < sizeof(ncclGinProxyGfd_t); i++) {
      EXPECT_EQ(zeroBytes[i], 0u) << "slot " << s << " byte " << i << " was unexpectedly written";
    }
  }
}

// ---------------------------------------------------------------------------
// PostGfd_MultiPeer: post one GFD per peer; per-peer ring base addressing
//   (queues[pe * queueSize], pis[pe], cis[pe]) must keep peers fully isolated.
// ---------------------------------------------------------------------------

__global__ void kernelPostGfdMultiPeer(ncclGinProxyGpuCtx_t* ctx,
                                       ncclGinProxyGfd_t* gfdsIn,
                                       uint32_t nranks) {
  for (uint32_t p = 0; p < nranks; p++) {
    nccl::gin::proxy::postGfd(ncclCoopCta{}, ctx, &gfdsIn[p], p);
  }
}

TEST_F(GinDeviceTest, PostGfd_MultiPeer) {
  constexpr uint32_t kNranks    = 4;
  constexpr uint32_t kQueueSize = 4;

  DeviceBuffer<ncclGinProxyGfd_t>     d_queues(kNranks * kQueueSize);
  DeviceBuffer<uint32_t>              d_pis(kNranks);
  DeviceBuffer<uint32_t>              d_cis(kNranks);
  DeviceBuffer<ncclGinProxyGfd_t>     d_gfdsIn(kNranks);
  DeviceBuffer<ncclGinProxyGpuCtx_t>  d_ctx(1);

  d_queues.zero();
  d_pis.zero();
  d_cis.zero();

  // gfdsIn[p].header.size = p so each peer's slot is uniquely identifiable.
  std::vector<ncclGinProxyGfd_t> hostGfds(kNranks);
  std::memset(hostGfds.data(), 0, hostGfds.size() * sizeof(ncclGinProxyGfd_t));
  for (uint32_t p = 0; p < kNranks; p++) {
    hostGfds[p].qword[ncclGinProxyGfdHeader].header.size = p;
  }
  d_gfdsIn.copyFrom(hostGfds);

  ncclGinProxyGpuCtx_t hostCtx{};
  hostCtx.nranks    = static_cast<int>(kNranks);
  hostCtx.queueSize = kQueueSize;
  hostCtx.queues    = d_queues.ptr;
  hostCtx.pis       = d_pis.ptr;
  hostCtx.cis       = d_cis.ptr;
  hostCtx.counters  = nullptr;
  hostCtx.signals   = nullptr;
  d_ctx.upload(hostCtx);

  kernelPostGfdMultiPeer<<<1, 1>>>(d_ctx.ptr, d_gfdsIn.ptr, kNranks);
  syncAndCheck();

  std::vector<uint32_t>          pis    = d_pis.copyTo();
  std::vector<uint32_t>          cis    = d_cis.copyTo();
  std::vector<ncclGinProxyGfd_t> queues = d_queues.copyTo();

  // Each peer's PI advanced exactly once; each peer's CI untouched.
  for (uint32_t p = 0; p < kNranks; p++) {
    EXPECT_EQ(pis[p], 1u) << "pis[" << p << "] must advance 0 -> 1";
    EXPECT_EQ(cis[p], 0u) << "cis[" << p << "] must be untouched";
  }

  // Slot 0 of each peer's ring received its own GFD; other slots stay zero.
  // No cross-peer contamination -- peer p's GFD lands at queues[p * queueSize + 0].
  for (uint32_t p = 0; p < kNranks; p++) {
    const auto& slot0 = queues[p * kQueueSize + 0];
    EXPECT_EQ(static_cast<uint64_t>(slot0.qword[ncclGinProxyGfdHeader].header.size),
              static_cast<uint64_t>(p))
        << "peer " << p << " slot 0 header.size mismatch";
    for (uint32_t s = 1; s < kQueueSize; s++) {
      const auto& slotN = queues[p * kQueueSize + s];
      EXPECT_EQ(static_cast<uint64_t>(slotN.qword[ncclGinProxyGfdHeader].header.size), 0ULL)
          << "peer " << p << " slot " << s << " was unexpectedly written";
    }
  }
}

// ---------------------------------------------------------------------------
// Flush<GIN_PROXY>: the blocking overload flushes every peer via FlushAsync+Wait; a pe+=1 stride is equivalent.
// ---------------------------------------------------------------------------

// Expected-timeout bound for proxy waits that must fail fast; 2^20 cycles is well under a millisecond.
constexpr uint64_t kProxyShortTimeoutCycles = 1ULL << 20;
// Watchdog deadline for waits that only an abort poll should end; far longer than one testAbort poll interval.
constexpr uint64_t kProxyWatchdogCycles = 1ULL << 30;

// Watchdog for one block: until state[0] reaches nDone, after cycles it records a rescue in state[1] and keeps
// cis[pe] at pis[pe], so a wait that ignores abortFlag ends as a test failure instead of a hang.
static __device__ void drainAfterDeadline(ncclGinProxyGpuCtx_t* proxyCtx, int pe, uint32_t nDone, uint64_t cycles,
                                          uint32_t* state) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> done(state[0]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(proxyCtx->pis[pe]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(proxyCtx->cis[pe]);
  const uint64_t startCycle = clock64();
  while (done.load(cuda::memory_order_acquire) < nDone) {
    if (clock64() - startCycle >= cycles) {
      state[1] = 1u;
      ci.store(pi.load(cuda::memory_order_acquire), cuda::memory_order_release);
    }
  }
}

// True when every qword of a proxy GFD slot is zero, i.e. nothing was posted to it.
static bool gfdSlotIsZero(const ncclGinProxyGfd_t& slot) {
  for (int q = 0; q < ncclGinProxyGfdQwords; q++) {
    if (slot.qword[q].raw != 0) {
      return false;
    }
  }
  return true;
}

// Expects every slot of every queue in the buffer to be unwritten.
static void expectQueuesZero(const DeviceBuffer<ncclGinProxyGfd_t>& queues, const char* what) {
  const std::vector<ncclGinProxyGfd_t> slots = queues.copyTo();
  for (size_t s = 0; s < slots.size(); s++) {
    EXPECT_TRUE(gfdSlotIsZero(slots[s])) << what << " posted to queue slot " << s;
  }
}

// Uploads one proxy ctx per element of d_ctxs, each over its own contiguous nranks slice of the device arrays.
static void uploadProxyGpuCtxs(DeviceBuffer<ncclGinProxyGpuCtx_t>& d_ctxs, int nranks, uint32_t queueSize,
                               ncclGinProxyGfd_t* queues, uint32_t* pis, uint32_t* cis, uint32_t* lastIssuedGet,
                               uint32_t* lastVisibleGet) {
  std::vector<ncclGinProxyGpuCtx_t> hostCtxs(d_ctxs.count);
  for (size_t c = 0; c < hostCtxs.size(); c++) {
    const size_t base = c * static_cast<size_t>(nranks);
    hostCtxs[c].nranks = nranks;
    hostCtxs[c].queueSize = queueSize;
    hostCtxs[c].queues = queues + base * queueSize;
    hostCtxs[c].pis = pis + base;
    hostCtxs[c].cis = cis + base;
    hostCtxs[c].lastIssuedGet = lastIssuedGet + base;
    hostCtxs[c].lastVisibleGet = lastVisibleGet + base;
  }
  d_ctxs.copyFrom(hostCtxs);
}

// A GIN_PROXY ncclGinCtx whose handle is the d_ctxs array, selecting contextId.
static ncclGinCtx makeProxyGinCtx(const DeviceBuffer<ncclGinProxyGpuCtx_t>& d_ctxs, int rank, int nRanks,
                                  int contextId) {
  ncclGinCtx ctx{};
  ctx.backend = NCCL_NET_DEVICE_GIN_PROXY;
  ctx.rank = rank;
  ctx.nRanks = nRanks;
  ctx.handle = d_ctxs.ptr;
  ctx.contextId = contextId;
  return ctx;
}

// One proxy context with nranks drained queues, gets pending on every peer, a raised abort and pad spare peer slots.
struct GinApiFlushRig {
  static constexpr int kNranks = 5;
  static constexpr uint32_t kQueueSize = 16; // One flush GFD per peer fits; postGfd's credit wait has no abort check.
  static constexpr int kRank = 2;
  static constexpr uint32_t kThreads = 3; // Fewer threads than peers, so peers past blockDim.x need a second pass.
  static constexpr uint32_t kGetBase = 100;

  explicit GinApiFlushRig(int pad)
    : d_queues(kNranks * kQueueSize), d_pis(kNranks + pad), d_cis(kNranks + pad), d_lastIssuedGet(kNranks + pad),
      d_lastVisibleGet(kNranks + pad), d_abortFlag(1), d_proxyCtx(1), hostIssued(kNranks + pad, 0u) {
    d_queues.zero();
    d_pis.zero();
    d_cis.zero();
    d_lastVisibleGet.zero();
    for (int pe = 0; pe < kNranks; pe++) {
      hostIssued[pe] = kGetBase + pe;
    }
    // No consumer runs, so a raised abort ends every ci wait.
    d_abortFlag.upload(1u);
    uploadProxyGpuCtxs(d_proxyCtx, kNranks, kQueueSize, d_queues.ptr, d_pis.ptr, d_cis.ptr, d_lastIssuedGet.ptr,
                       d_lastVisibleGet.ptr);
  }

  // Uploads hostIssued, so callers can seed the padding slots first.
  ncclGinCtx UploadAndMakeCtx() {
    d_lastIssuedGet.copyFrom(hostIssued);
    return makeProxyGinCtx(d_proxyCtx, kRank, kNranks, /*contextId=*/0);
  }

  DeviceBuffer<ncclGinProxyGfd_t> d_queues;
  DeviceBuffer<uint32_t> d_pis;
  DeviceBuffer<uint32_t> d_cis;
  DeviceBuffer<uint32_t> d_lastIssuedGet;
  DeviceBuffer<uint32_t> d_lastVisibleGet;
  DeviceBuffer<uint32_t> d_abortFlag;
  DeviceBuffer<ncclGinProxyGpuCtx_t> d_proxyCtx;
  std::vector<uint32_t> hostIssued;
};

// Block 0 runs the blocking Flush; block 1 drains the local queue only if Flush outlives kProxyWatchdogCycles.
// state[0] counts block 0 threads that returned, state[1] records a rescue.
__global__ void kernelGinApiFlushBlocking(ncclGinCtx ctx, uint32_t* abortFlag, uint32_t* state) {
  if (blockIdx.x == 0) {
    ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, ncclCoopCta{}, /*hasDescriptor=*/false,
                                                      /*descriptor=*/nullptr, cuda::memory_order_acquire, abortFlag);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(state[0]).fetch_add(1u, cuda::memory_order_release);
  } else if (threadIdx.x == 0) {
    ncclGinProxyGpuCtx_t* proxyCtx = &static_cast<ncclGinProxyGpuCtx_t*>(ctx.handle)[ctx.contextId];
    drainAfterDeadline(proxyCtx, ctx.rank, blockDim.x, kProxyWatchdogCycles, state);
  }
}

TEST_F(GinDeviceTest, Flush_FlushesEveryPeerOnLocalQueue) {
  using Rig = GinApiFlushRig;
  constexpr uint32_t kSentinelGet = 999; // Index kNranks is one past the last peer: a loop overrun would flush it.
  Rig rig(/*pad=*/1);
  rig.hostIssued[Rig::kNranks] = kSentinelGet;

  DeviceBuffer<uint32_t> d_state(2);
  d_state.zero();

  kernelGinApiFlushBlocking<<<2, Rig::kThreads>>>(rig.UploadAndMakeCtx(), rig.d_abortFlag.ptr, d_state.ptr);
  syncAndCheck();
  EXPECT_EQ(d_state.copyTo(), (std::vector<uint32_t>{Rig::kThreads, 0u}))
    << "{returned, rescued}: rescued=1 means a local flush wait ignored abortFlag and the watchdog drained cis";

  std::vector<uint32_t> pis = rig.d_pis.copyTo();
  std::vector<uint32_t> cis = rig.d_cis.copyTo();
  std::vector<uint32_t> visible = rig.d_lastVisibleGet.copyTo();
  std::vector<ncclGinProxyGfd_t> queues = rig.d_queues.copyTo();

  // Pins a gap: an abort-ended flush still publishes lastVisibleGet; correct is unpublished (fix at gin_proxy.h:375).
  for (int pe = 0; pe < Rig::kNranks; pe++) {
    EXPECT_EQ(visible[pe], Rig::kGetBase + pe)
      << "lastVisibleGet[" << pe << "]: an aborted flush still publishes today; flips when gin_proxy.h:375 is fixed";
    EXPECT_EQ(cis[pe], 0u) << "cis[" << pe << "] is consumer-owned and must be untouched";
  }
  EXPECT_EQ(visible[Rig::kNranks], 0u) << "Flush visited a peer index past nRanks";

  // 1 to nranks flush GFDs, all on the local queue (today one per peer; gin_proxy.h notes that 1 would suffice).
  const uint32_t localPosts = pis[Rig::kRank];
  EXPECT_GE(localPosts, 1u) << "no local flush GFD";
  EXPECT_LE(localPosts, static_cast<uint32_t>(Rig::kNranks)) << "more flush GFDs than peers";
  for (int pe = 0; pe < Rig::kNranks; pe++) {
    for (uint32_t s = 0; s < Rig::kQueueSize; s++) {
      const ncclGinProxyGfd_t& slot = queues[pe * Rig::kQueueSize + s];
      if (pe == Rig::kRank && s < localPosts) {
        EXPECT_EQ(static_cast<uint64_t>(slot.qword[ncclGinProxyGfdHeader].header.flag), 1u)
          << "local slot " << s << " not written";
        EXPECT_EQ(static_cast<uint64_t>(slot.qword[ncclGinProxyGfdHeaderExt].headerExt.op),
                  static_cast<uint64_t>(ncclGinProxyOpFlush))
          << "local slot " << s << " is not a flush GFD";
      } else {
        EXPECT_TRUE(gfdSlotIsZero(slot)) << "queue " << pe << " slot " << s << " was unexpectedly written";
      }
    }
    if (pe != Rig::kRank) {
      EXPECT_EQ(pis[pe], 0u) << "flush GFD posted to peer queue " << pe;
    }
  }
}

// ---------------------------------------------------------------------------
// FlushTimeout_ReportsLaggingPeer: only the thread owning a peer with pi != ci in handle[contextId] times out.
// ---------------------------------------------------------------------------

__global__ void kernelGinApiFlushTimeout(ncclGinCtx ctx, uint64_t timeoutCycles, ncclResult_t* rets,
                                         uint32_t* abortFlag = nullptr) {
  rets[threadIdx.x] =
    ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, ncclCoopCta{}, /*hasDescriptor=*/false,
                                                      /*descriptor=*/nullptr, cuda::memory_order_acquire, abortFlag,
                                                      timeoutCycles);
}

TEST_F(GinDeviceTest, FlushTimeout_ReportsLaggingPeer) {
  constexpr uint32_t kNranks = 5;
  constexpr uint32_t kQueueSize = 16;
  constexpr uint32_t kThreads = 3; // Thread t owns peers t, t+3: thread 1 owns peers 1 and 4.
  constexpr uint32_t kLaggingPeer = 4;
  constexpr uint32_t kNumContexts = 2;
  constexpr uint32_t kContextId = 1; // ctx[0] is a decoy with every peer drained.
  constexpr uint32_t kIdxSize = kNumContexts * kNranks + 1; // Last index: lagging sentinel one past ctx[1]'s peers.
  constexpr uint32_t kLagPi = 7; // pi one ahead of ci: one GFD the absent consumer never takes.

  DeviceBuffer<ncclGinProxyGfd_t> d_queues(kNumContexts * kNranks * kQueueSize);
  DeviceBuffer<uint32_t> d_pis(kIdxSize);
  DeviceBuffer<uint32_t> d_cis(kIdxSize);
  DeviceBuffer<uint32_t> d_gets(kIdxSize); // Unread by the timeout overload; shared as lastIssuedGet/lastVisibleGet.
  DeviceBuffer<ncclResult_t> d_rets(kThreads);
  DeviceBuffer<ncclGinProxyGpuCtx_t> d_proxyCtx(kNumContexts);

  d_queues.zero();
  d_gets.zero();

  uploadProxyGpuCtxs(d_proxyCtx, static_cast<int>(kNranks), kQueueSize, d_queues.ptr, d_pis.ptr, d_cis.ptr, d_gets.ptr,
                     d_gets.ptr);
  const ncclGinCtx ctx =
    makeProxyGinCtx(d_proxyCtx, /*rank=*/0, static_cast<int>(kNranks), static_cast<int>(kContextId));

  // Case A: every peer of the selected context is drained; only the past-the-end sentinel lags, so all threads succeed.
  std::vector<uint32_t> hostPis(kIdxSize, 0u);
  std::vector<uint32_t> hostCis(kIdxSize, 0u);
  hostPis[kIdxSize - 1] = kLagPi;
  hostCis[kIdxSize - 1] = kLagPi - 1;
  d_pis.copyFrom(hostPis);
  d_cis.copyFrom(hostCis);
  d_rets.copyFrom(std::vector<ncclResult_t>(kThreads, ncclInternalError));
  kernelGinApiFlushTimeout<<<1, kThreads>>>(ctx, kProxyShortTimeoutCycles, d_rets.ptr);
  syncAndCheck();
  std::vector<ncclResult_t> retsA = d_rets.copyTo();
  for (uint32_t t = 0; t < kThreads; t++) {
    EXPECT_EQ(retsA[t], ncclSuccess) << "drained: thread " << t;
  }

  // Case B: the selected context's last peer has one unconsumed GFD; only its owning thread times out.
  hostPis[kContextId * kNranks + kLaggingPeer] = kLagPi;
  hostCis[kContextId * kNranks + kLaggingPeer] = kLagPi - 1;
  d_pis.copyFrom(hostPis);
  d_cis.copyFrom(hostCis);
  d_rets.copyFrom(std::vector<ncclResult_t>(kThreads, ncclInternalError));
  kernelGinApiFlushTimeout<<<1, kThreads>>>(ctx, kProxyShortTimeoutCycles, d_rets.ptr);
  syncAndCheck();
  std::vector<ncclResult_t> retsB = d_rets.copyTo();
  for (uint32_t t = 0; t < kThreads; t++) {
    const ncclResult_t expected = (t == kLaggingPeer % kThreads) ? ncclTimeout : ncclSuccess;
    EXPECT_EQ(retsB[t], expected) << "lagging peer " << kLaggingPeer << ": thread " << t;
  }

  EXPECT_EQ(d_pis.copyTo(), hostPis) << "timeout Flush moved a producer index";
  EXPECT_EQ(d_cis.copyTo(), hostCis) << "timeout Flush moved a consumer index";
}

// ---------------------------------------------------------------------------
// FlushTimeout_CurrentlyLeavesPendingGetsInvisible: pins a known gap; invert once the timeout overload flushes too.
// ---------------------------------------------------------------------------

TEST_F(GinDeviceTest, FlushTimeout_CurrentlyLeavesPendingGetsInvisible) {
  using Rig = GinApiFlushRig;
  constexpr uint64_t kNeverExpiresCycles = 1ULL << 62; // Never hit: queues are drained and no GFD is posted.
  Rig rig(/*pad=*/0);
  DeviceBuffer<ncclResult_t> d_rets(Rig::kThreads);
  d_rets.copyFrom(std::vector<ncclResult_t>(Rig::kThreads, ncclInternalError));

  kernelGinApiFlushTimeout<<<1, Rig::kThreads>>>(rig.UploadAndMakeCtx(), kNeverExpiresCycles, d_rets.ptr,
                                                 rig.d_abortFlag.ptr);
  syncAndCheck();

  std::vector<ncclResult_t> rets = d_rets.copyTo();
  for (uint32_t t = 0; t < Rig::kThreads; t++) {
    EXPECT_EQ(rets[t], ncclSuccess) << "thread " << t;
  }

  // Unlike the blocking overload, gets stay invisible and no local flush GFD is posted.
  const std::vector<uint32_t> zeros(Rig::kNranks, 0u);
  EXPECT_EQ(rig.d_lastVisibleGet.copyTo(), zeros) << "timeout Flush advanced lastVisibleGet";
  EXPECT_EQ(rig.d_pis.copyTo(), zeros) << "timeout Flush moved a producer index";
  EXPECT_EQ(rig.d_cis.copyTo(), zeros) << "timeout Flush moved a consumer index";
  expectQueuesZero(rig.d_queues, "timeout Flush");

  // Null abort, short timeout: every pi == ci, so only a Flush that waits on the pending gets can time out.
  // Skipped after a failure, since a blocking-overload dispatch could then spin forever without the abort.
  if (HasFailure()) {
    return;
  }
  d_rets.copyFrom(std::vector<ncclResult_t>(Rig::kThreads, ncclInternalError));
  kernelGinApiFlushTimeout<<<1, Rig::kThreads>>>(rig.UploadAndMakeCtx(), kProxyShortTimeoutCycles, d_rets.ptr);
  syncAndCheck();
  rets = d_rets.copyTo();
  for (uint32_t t = 0; t < Rig::kThreads; t++) {
    EXPECT_EQ(rets[t], ncclSuccess) << "no-abort launch, thread " << t << ": timeout Flush waited on pending gets";
  }
  EXPECT_EQ(rig.d_lastVisibleGet.copyTo(), zeros) << "no-abort timeout Flush advanced lastVisibleGet";
  expectQueuesZero(rig.d_queues, "no-abort timeout Flush");
}

// ---------------------------------------------------------------------------
// PostGfd_Wrap: pre-position pi=ci=queueSize-1 so 4 sequential posts straddle
//   the queueSize=4 boundary on iter 1 (idx=4 -> slot 0). Tests the
//   `idx & (queueSize-1)` wrap arithmetic at gin_proxy.h:57 deterministically.
//
//   Note: the credit-wait `while (queueSize <= idx - ci)` uses UNSIGNED
//   subtraction, so a single producer can advance idx past ci by at most
//   queueSize-1 before blocking. A more aggressive 8-post wrap test would
//   need a concurrent consumer to bump ci, which isn't worth the complexity.
//   Pre-positioning pi/ci near the wrap point still exercises the wrap.
// ---------------------------------------------------------------------------

__global__ void kernelPostGfdWrap(ncclGinProxyGpuCtx_t* ctx,
                                  ncclGinProxyGfd_t* gfdsIn,
                                  uint32_t n, uint32_t peer) {
  for (uint32_t i = 0; i < n; i++) {
    nccl::gin::proxy::postGfd(ncclCoopCta{}, ctx, &gfdsIn[i], peer);
  }
}

TEST_F(GinDeviceTest, PostGfd_Wrap) {
  constexpr uint32_t kNranks    = 1;
  constexpr uint32_t kQueueSize = 4;
  constexpr uint32_t kPeer      = 0;
  constexpr uint32_t kPiPreset  = 3;       // queueSize - 1: iter 1 hits idx=4 (wrap)
  constexpr uint32_t kCiPreset  = 3;       // = pi: idx-ci stays in [0, queueSize-1]
  constexpr uint32_t kNumPosts  = 4;

  DeviceBuffer<ncclGinProxyGfd_t>     d_queues(kNranks * kQueueSize);
  DeviceBuffer<uint32_t>              d_pis(kNranks);
  DeviceBuffer<uint32_t>              d_cis(kNranks);
  DeviceBuffer<ncclGinProxyGfd_t>     d_gfdsIn(kNumPosts);
  DeviceBuffer<ncclGinProxyGpuCtx_t>  d_ctx(1);

  d_queues.zero();
  d_pis.copyFrom(std::vector<uint32_t>{kPiPreset});
  d_cis.copyFrom(std::vector<uint32_t>{kCiPreset});

  std::vector<ncclGinProxyGfd_t> hostGfds(kNumPosts);
  std::memset(hostGfds.data(), 0, hostGfds.size() * sizeof(ncclGinProxyGfd_t));
  for (uint32_t i = 0; i < kNumPosts; i++) {
    hostGfds[i].qword[ncclGinProxyGfdHeader].header.size = i;
  }
  d_gfdsIn.copyFrom(hostGfds);

  ncclGinProxyGpuCtx_t hostCtx{};
  hostCtx.nranks    = static_cast<int>(kNranks);
  hostCtx.queueSize = kQueueSize;
  hostCtx.queues    = d_queues.ptr;
  hostCtx.pis       = d_pis.ptr;
  hostCtx.cis       = d_cis.ptr;
  hostCtx.counters  = nullptr;
  hostCtx.signals   = nullptr;
  d_ctx.upload(hostCtx);

  kernelPostGfdWrap<<<1, 1>>>(d_ctx.ptr, d_gfdsIn.ptr, kNumPosts, kPeer);
  syncAndCheck();

  std::vector<uint32_t>          pis    = d_pis.copyTo();
  std::vector<uint32_t>          cis    = d_cis.copyTo();
  std::vector<ncclGinProxyGfd_t> queues = d_queues.copyTo();

  // PI advanced by kNumPosts; CI unchanged.
  EXPECT_EQ(pis[0], kPiPreset + kNumPosts) << "PI must advance 3 -> 7";
  EXPECT_EQ(cis[0], kCiPreset)             << "CI must be untouched";

  // Slot mapping under `idx & (queueSize-1)`:
  //   iter 0: idx=3 -> slot 3 (gfd 0)
  //   iter 1: idx=4 -> slot 0 (gfd 1)   <-- WRAP
  //   iter 2: idx=5 -> slot 1 (gfd 2)
  //   iter 3: idx=6 -> slot 2 (gfd 3)
  EXPECT_EQ(static_cast<uint64_t>(queues[3].qword[ncclGinProxyGfdHeader].header.size), 0ULL) << "iter 0 -> slot 3";
  EXPECT_EQ(static_cast<uint64_t>(queues[0].qword[ncclGinProxyGfdHeader].header.size), 1ULL) << "iter 1 -> slot 0 (wrap)";
  EXPECT_EQ(static_cast<uint64_t>(queues[1].qword[ncclGinProxyGfdHeader].header.size), 2ULL) << "iter 2 -> slot 1";
  EXPECT_EQ(static_cast<uint64_t>(queues[2].qword[ncclGinProxyGfdHeader].header.size), 3ULL) << "iter 3 -> slot 2";
}

// ---------------------------------------------------------------------------
// PostGfd_PiCiOverflow: pre-position pi=ci=0xFFFFFFFE so 4 sequential posts
//   straddle the uint32 wrap. Verifies that:
//     1. `idx = pi.fetch_add(1)` correctly wraps the producer index past 2^32.
//     2. `idx & (queueSize-1)` produces the right slot post-wrap.
//     3. `queueSize <= idx - ci.load()` (unsigned subtraction) keeps the
//        credit-wait dormant when both pi and ci cross the boundary together.
// ---------------------------------------------------------------------------

__global__ void kernelPostGfdOverflow(ncclGinProxyGpuCtx_t* ctx,
                                      ncclGinProxyGfd_t* gfdsIn,
                                      uint32_t n, uint32_t peer) {
  for (uint32_t i = 0; i < n; i++) {
    nccl::gin::proxy::postGfd(ncclCoopCta{}, ctx, &gfdsIn[i], peer);
  }
}

TEST_F(GinDeviceTest, PostGfd_PiCiOverflow) {
  constexpr uint32_t kNranks    = 2;
  constexpr uint32_t kQueueSize = 4;
  constexpr uint32_t kPeer      = 0;
  constexpr uint32_t kPreset    = 0xFFFFFFFEu;   // 4 puts straddle the uint32 wrap
  constexpr uint32_t kNumPosts  = 4;

  DeviceBuffer<ncclGinProxyGfd_t>     d_queues(kNranks * kQueueSize);
  DeviceBuffer<uint32_t>              d_pis(kNranks);
  DeviceBuffer<uint32_t>              d_cis(kNranks);
  DeviceBuffer<ncclGinProxyGfd_t>     d_gfdsIn(kNumPosts);
  DeviceBuffer<ncclGinProxyGpuCtx_t>  d_ctx(1);

  d_queues.zero();
  // pis[0] = cis[0] = 0xFFFFFFFE; pis[1] = cis[1] = 0 (peer 1 acts as a sentinel).
  d_pis.copyFrom(std::vector<uint32_t>{kPreset, 0u});
  d_cis.copyFrom(std::vector<uint32_t>{kPreset, 0u});

  std::vector<ncclGinProxyGfd_t> hostGfds(kNumPosts);
  std::memset(hostGfds.data(), 0, hostGfds.size() * sizeof(ncclGinProxyGfd_t));
  for (uint32_t i = 0; i < kNumPosts; i++) {
    hostGfds[i].qword[ncclGinProxyGfdHeader].header.size = i;
  }
  d_gfdsIn.copyFrom(hostGfds);

  ncclGinProxyGpuCtx_t hostCtx{};
  hostCtx.nranks    = static_cast<int>(kNranks);
  hostCtx.queueSize = kQueueSize;
  hostCtx.queues    = d_queues.ptr;
  hostCtx.pis       = d_pis.ptr;
  hostCtx.cis       = d_cis.ptr;
  hostCtx.counters  = nullptr;
  hostCtx.signals   = nullptr;
  d_ctx.upload(hostCtx);

  kernelPostGfdOverflow<<<1, 1>>>(d_ctx.ptr, d_gfdsIn.ptr, kNumPosts, kPeer);
  syncAndCheck();

  std::vector<uint32_t>          pis    = d_pis.copyTo();
  std::vector<uint32_t>          cis    = d_cis.copyTo();
  std::vector<ncclGinProxyGfd_t> queues = d_queues.copyTo();

  // PI started at 0xFFFFFFFE; 4 fetch_adds wrap the uint32 to (0xFFFFFFFE + 4) mod 2^32 = 2.
  EXPECT_EQ(pis[0], static_cast<uint32_t>(kPreset + kNumPosts))
      << "PI must wrap through 0xFFFFFFFE -> 0xFFFFFFFF -> 0 -> 1 -> 2";
  EXPECT_EQ(pis[1], 0u)        << "peer 1 PI must be untouched";
  EXPECT_EQ(cis[0], kPreset)   << "CI must be untouched";
  EXPECT_EQ(cis[1], 0u)        << "peer 1 CI must be untouched";

  // Slot mapping (idx is the OLD pi value returned by fetch_add):
  //   iter 0: idx=0xFFFFFFFE -> slot 2 (gfd 0)
  //   iter 1: idx=0xFFFFFFFF -> slot 3 (gfd 1)
  //   iter 2: idx=0x00000000 -> slot 0 (gfd 2)
  //   iter 3: idx=0x00000001 -> slot 1 (gfd 3)
  EXPECT_EQ(static_cast<uint64_t>(queues[2].qword[ncclGinProxyGfdHeader].header.size), 0ULL) << "iter 0 -> slot 2";
  EXPECT_EQ(static_cast<uint64_t>(queues[3].qword[ncclGinProxyGfdHeader].header.size), 1ULL) << "iter 1 -> slot 3";
  EXPECT_EQ(static_cast<uint64_t>(queues[0].qword[ncclGinProxyGfdHeader].header.size), 2ULL) << "iter 2 -> slot 0 (wrap)";
  EXPECT_EQ(static_cast<uint64_t>(queues[1].qword[ncclGinProxyGfdHeader].header.size), 3ULL) << "iter 3 -> slot 1";

  // Peer 1's ring must remain entirely zero (no cross-peer contamination near the wrap).
  for (uint32_t s = 0; s < kQueueSize; s++) {
    EXPECT_EQ(static_cast<uint64_t>(queues[kQueueSize + s].qword[ncclGinProxyGfdHeader].header.size), 0ULL)
        << "peer 1 slot " << s << " was unexpectedly written";
  }
}

// ---------------------------------------------------------------------------
// FlushAsync<PROXY> snapshots pi and lastIssuedGet; Wait<PROXY> gates on it, flushes locally if needed, publishes.
// ---------------------------------------------------------------------------

static __device__ ncclGinProxyGpuCtx_t* proxyCtxOf(const ncclGinCtx& ctx) {
  return &static_cast<ncclGinProxyGpuCtx_t*>(ctx.handle)[ctx.contextId];
}

__global__ void kernelGinApiFlushAsync(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, peer, outRequest, /*hasDescriptor=*/false,
                                                         /*descriptor=*/nullptr, /*optFlags=*/0);
}

// Uses the timeout Wait overload; postExtra posts one GFD to the peer between FlushAsync and Wait.
__global__ void kernelGinApiFlushAsyncTimedWait(ncclGinCtx ctx, int peer, bool postExtra, uint64_t timeoutCycles,
                                                ncclResult_t* outResult) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  ncclGinRequest_t request;
  ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, peer, &request, /*hasDescriptor=*/false,
                                                         /*descriptor=*/nullptr, /*optFlags=*/0);
  if (postExtra) {
    ncclGinProxyGfd_t gfd{};
    nccl::gin::proxy::postGfd(ncclCoopThread(), proxyCtxOf(ctx), &gfd, peer);
  }
  *outResult = ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, request, /*hasDescriptor=*/false,
                                                                /*descriptor=*/nullptr, cuda::memory_order_acquire,
                                                                /*abortFlag=*/nullptr, timeoutCycles);
}

// Block 0 runs Wait; an optional block 1 plays the proxy and consumes one GFD on consumerPe by advancing cis, bounded.
__global__ void kernelGinApiWait(ncclGinCtx ctx, ncclGinRequest_t request, int consumerPe, uint64_t timeoutCycles,
                                 ncclResult_t* outResult) {
  if (threadIdx.x != 0) {
    return;
  }
  if (blockIdx.x == 0) {
    *outResult = ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, request, /*hasDescriptor=*/false,
                                                                  /*descriptor=*/nullptr, cuda::memory_order_acquire,
                                                                  /*abortFlag=*/nullptr, timeoutCycles);
    return;
  }
  ncclGinProxyGpuCtx_t* proxyCtx = proxyCtxOf(ctx);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(proxyCtx->pis[consumerPe]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(proxyCtx->cis[consumerPe]);
  const uint32_t consumed = ci.load(cuda::memory_order_relaxed);
  const uint64_t startCycle = clock64();
  while (pi.load(cuda::memory_order_acquire) == consumed) {
    if (clock64() - startCycle >= timeoutCycles) {
      return;
    }
  }
  ci.store(pi.load(cuda::memory_order_relaxed), cuda::memory_order_release);
}

// Block 0 runs one Wait overload under abortFlag; an optional block 1 drains cis[peer] if Wait outlives cycles.
// state[0] is set once Wait returns, state[1] once the watchdog rescued it; only the timed arm writes outResult.
__global__ void kernelGinApiWaitAbort(ncclGinCtx ctx, ncclGinRequest_t request, bool timed, uint32_t* abortFlag,
                                      uint64_t cycles, ncclResult_t* outResult, uint32_t* state) {
  if (threadIdx.x != 0) {
    return;
  }
  if (blockIdx.x == 0) {
    if (timed) {
      *outResult = ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, request, /*hasDescriptor=*/false,
                                                                    /*descriptor=*/nullptr, cuda::memory_order_acquire,
                                                                    abortFlag, cycles);
    } else {
      ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, request, /*hasDescriptor=*/false, /*descriptor=*/nullptr,
                                                       cuda::memory_order_acquire, abortFlag);
    }
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(state[0]).store(1u, cuda::memory_order_release);
    return;
  }
  const ncclGinCpuProxyRequest& req = reinterpret_cast<const ncclGinCpuProxyRequest&>(request);
  drainAfterDeadline(proxyCtxOf(ctx), req.peer, /*nDone=*/1u, cycles, state);
}

// Two proxy contexts (context 0 is a decoy) and ctx.rank != peer, so a wrong context or wrong flush queue shows up.
// Every queue starts drained (ci == pi) so postGfd never blocks, even on a wrong queue.
class GinProxyFlushWaitTest : public GinDeviceTest {
protected:
  static constexpr int kNranks = 3;
  static constexpr uint32_t kQueueSize = 4;
  static constexpr int kNumContexts = 2;
  static constexpr int kContextId = 1;
  static constexpr int kRank = 2;
  static constexpr int kPeer = 1;
  static constexpr uint64_t kConsumerCycles = 1ULL << 30; // Bounds the fake proxy block; never hit when Wait posts.

  GinProxyFlushWaitTest()
    : setDeviceErr_(hipSetDevice(0)), d_queues_(kNumContexts * kNranks * kQueueSize), d_pis_(kNumContexts * kNranks),
      d_cis_(kNumContexts * kNranks), d_lastIssuedGet_(kNumContexts * kNranks),
      d_lastVisibleGet_(kNumContexts * kNranks), d_proxyCtxs_(kNumContexts), pis_(kNumContexts * kNranks),
      cis_(kNumContexts * kNranks), lastIssuedGet_(kNumContexts * kNranks), lastVisibleGet_(kNumContexts * kNranks) {}

  // Distinct values per counter, context and peer; by default every GFD is consumed and every get is visible.
  void SetUp() override {
    ASSERT_EQ(setDeviceErr_, hipSuccess);
    GinDeviceTest::SetUp();
    for (int c = 0; c < kNumContexts; c++) {
      for (int p = 0; p < kNranks; p++) {
        pis_[Idx(c, p)] = 0x1000u + 0x100u * c + 0x11u * p; // + p so each queue starts at a different slot
        lastIssuedGet_[Idx(c, p)] = 0x7000u + 0x100u * c + 0x10u * p;
      }
    }
    cis_ = pis_;
    lastVisibleGet_ = lastIssuedGet_;
  }

  static int Idx(int contextId, int peer) {
    return contextId * kNranks + peer;
  }

  void Upload() {
    d_queues_.zero();
    d_pis_.copyFrom(pis_);
    d_cis_.copyFrom(cis_);
    d_lastIssuedGet_.copyFrom(lastIssuedGet_);
    d_lastVisibleGet_.copyFrom(lastVisibleGet_);
    uploadProxyGpuCtxs(d_proxyCtxs_, kNranks, kQueueSize, d_queues_.ptr, d_pis_.ptr, d_cis_.ptr, d_lastIssuedGet_.ptr,
                       d_lastVisibleGet_.ptr);
  }

  ncclGinCtx MakeCtx() const {
    return makeProxyGinCtx(d_proxyCtxs_, kRank, kNranks, kContextId);
  }

  static ncclGinRequest_t MakeWaitRequest(int peer, uint32_t nextGfdIdx, uint32_t lastIssuedGet) {
    ncclGinCpuProxyRequest req{peer, nextGfdIdx, lastIssuedGet};
    ncclGinRequest_t request{};
    std::memcpy(&request, &req, sizeof(req));
    return request;
  }

  // Expects exactly one local GFD; the queues start zeroed, so a complete flush GFD in its slot proves Wait posted it.
  void ExpectLocalFlushGfd(uint32_t localPi) {
    std::vector<uint32_t> expectedPis = pis_;
    expectedPis[Idx(kContextId, kRank)] = localPi + 1;
    EXPECT_EQ(d_pis_.copyTo(), expectedPis) << "exactly one flush GFD, posted to the local rank queue of ctx.contextId";
    std::vector<ncclGinProxyGfd_t> queues = d_queues_.copyTo();
    const ncclGinProxyGfd_t& gfd = queues[Idx(kContextId, kRank) * kQueueSize + (localPi & (kQueueSize - 1))];
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeaderExt].headerExt.op),
              static_cast<uint64_t>(ncclGinProxyOpFlush));
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.size), 0ULL);
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.version),
              static_cast<uint64_t>(NCCL_GIN_PROXY_GFD_VERSION));
    for (int i = 0; i < ncclGinProxyGfdQwords; i++) {
      EXPECT_EQ(static_cast<uint64_t>(gfd.qword[i].flag.v), 1ULL) << "qword " << i;
    }
  }

  hipError_t setDeviceErr_; // Must precede the DeviceBuffers; they allocate on the current device.
  DeviceBuffer<ncclGinProxyGfd_t> d_queues_;
  DeviceBuffer<uint32_t> d_pis_;
  DeviceBuffer<uint32_t> d_cis_;
  DeviceBuffer<uint32_t> d_lastIssuedGet_;
  DeviceBuffer<uint32_t> d_lastVisibleGet_;
  DeviceBuffer<ncclGinProxyGpuCtx_t> d_proxyCtxs_;
  std::vector<uint32_t> pis_;
  std::vector<uint32_t> cis_;
  std::vector<uint32_t> lastIssuedGet_;
  std::vector<uint32_t> lastVisibleGet_;
};

TEST_F(GinProxyFlushWaitTest, FlushAsyncSnapshotsPiAndLastIssuedGet) {
  Upload();
  DeviceBuffer<ncclGinRequest_t> d_request(1);
  ncclGinRequest_t poison;
  poison.opaque[0] = ~0ULL;
  poison.opaque[1] = ~0ULL;
  d_request.upload(poison);

  kernelGinApiFlushAsync<<<1, 1>>>(MakeCtx(), kPeer, d_request.ptr);
  syncAndCheck();

  const ncclGinRequest_t raw = d_request.download();
  ncclGinCpuProxyRequest req;
  std::memcpy(&req, &raw, sizeof(req));
  EXPECT_EQ(req.peer, kPeer);
  EXPECT_EQ(req.nextGfdIdx, pis_[Idx(kContextId, kPeer)]) << "nextGfdIdx must be pis[peer] of ctx.contextId";
  EXPECT_EQ(req.lastIssuedGet, lastIssuedGet_[Idx(kContextId, kPeer)])
    << "lastIssuedGet must be lastIssuedGet[peer] of ctx.contextId";
  EXPECT_EQ(d_pis_.copyTo(), pis_) << "FlushAsync must not post a GFD";
  EXPECT_EQ(d_lastIssuedGet_.copyTo(), lastIssuedGet_);
}

TEST_F(GinProxyFlushWaitTest, WaitCoversOnlyGfdsPostedBeforeFlushAsync) {
  // Snapshot 0, so case 2 seeds CI 0xFFFFFFFF and only a wrap-aware compare keeps Wait spinning.
  pis_[Idx(kContextId, kPeer)] = 0u;
  cis_ = pis_;
  const uint32_t snapshot = pis_[Idx(kContextId, kPeer)];
  DeviceBuffer<ncclResult_t> d_result(1);

  // CI equal to the snapshot: Wait must succeed even though the GFD posted after FlushAsync is still pending.
  Upload();
  d_result.upload(ncclInternalError);
  kernelGinApiFlushAsyncTimedWait<<<1, 1>>>(MakeCtx(), kPeer, /*postExtra=*/true, kProxyShortTimeoutCycles,
                                            d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess) << "Wait must wait on req.nextGfdIdx, not the live pi";
  std::vector<uint32_t> expectedPis = pis_;
  expectedPis[Idx(kContextId, kPeer)] = snapshot + 1;
  EXPECT_EQ(d_pis_.copyTo(), expectedPis) << "only the extra GFD is posted; no local flush when all gets are visible";
  EXPECT_EQ(d_lastVisibleGet_.copyTo(), lastVisibleGet_);

  // CI one short of the snapshot: a GFD posted before FlushAsync is unconsumed, so Wait must time out.
  cis_[Idx(kContextId, kPeer)] = snapshot - 1;
  Upload();
  d_result.upload(ncclInternalError);
  kernelGinApiFlushAsyncTimedWait<<<1, 1>>>(MakeCtx(), kPeer, /*postExtra=*/true, kProxyShortTimeoutCycles,
                                            d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout) << "Wait must not return before CI reaches req.nextGfdIdx";
}

TEST_F(GinProxyFlushWaitTest, TimedWaitFlushesLocalQueueWhenGetsPending) {
  const uint32_t localPi = pis_[Idx(kContextId, kRank)];
  lastVisibleGet_[Idx(kContextId, kPeer)] = lastIssuedGet_[Idx(kContextId, kPeer)] - 2;
  Upload();
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);

  kernelGinApiFlushAsyncTimedWait<<<1, 1>>>(MakeCtx(), kPeer, /*postExtra=*/false, kProxyShortTimeoutCycles,
                                            d_result.ptr);
  syncAndCheck();

  // Nothing consumes the local flush GFD, so the timed flush must spin on the local queue until it times out.
  EXPECT_EQ(d_result.download(), ncclTimeout) << "Wait must flush the local rank queue, not skip or flush the peer";
  ExpectLocalFlushGfd(localPi);
  EXPECT_EQ(d_lastVisibleGet_.copyTo(), lastVisibleGet_) << "a timed-out Wait must not publish lastVisibleGet";
}

TEST_F(GinProxyFlushWaitTest, WaitSkipsFlushWhenGetsAlreadyVisible) {
  struct Case {
    const char* name;
    uint32_t visible;
    uint32_t issued;
  };
  // Requests carry their own issued values, unlike the live 0x7110; the wrap case is ahead only under rolling order.
  const Case cases[] = {
    {"equal", 0x5010u, 0x5010u},
    {"visible ahead", 0x5011u, 0x5010u},
    {"visible ahead across wrap", 0x00000001u, 0xFFFFFFFFu},
  };
  // The gate is also satisfied across wrap: ci 0x1 is past nextGfdIdx 0xFFFFFFFE.
  constexpr uint32_t kNextGfdIdx = 0xFFFFFFFEu;
  DeviceBuffer<ncclResult_t> d_result(1);
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    cis_[Idx(kContextId, kPeer)] = 0x00000001u;
    pis_[Idx(kContextId, kPeer)] = cis_[Idx(kContextId, kPeer)];
    lastVisibleGet_[Idx(kContextId, kPeer)] = c.visible;
    Upload();
    d_result.upload(ncclInternalError);

    kernelGinApiWait<<<1, 1>>>(MakeCtx(), MakeWaitRequest(kPeer, kNextGfdIdx, c.issued), kRank,
                               kProxyShortTimeoutCycles, d_result.ptr);
    syncAndCheck();

    EXPECT_EQ(d_result.download(), ncclSuccess) << "no flush is needed, so nothing can time out";
    EXPECT_EQ(d_pis_.copyTo(), pis_) << "Wait must not post a flush GFD when all gets are visible";
    expectQueuesZero(d_queues_, "Wait");
    EXPECT_EQ(d_lastVisibleGet_.copyTo(), lastVisibleGet_) << "lastVisibleGet must not move backward";
  }
}

TEST_F(GinProxyFlushWaitTest, WaitGateHonorsAbortFlag) {
  // CI one short of req.nextGfdIdx and no consumer, so only the raised abort can end Wait's GFD gate.
  const uint32_t snapshot = pis_[Idx(kContextId, kPeer)];
  cis_[Idx(kContextId, kPeer)] = snapshot - 1;
  const ncclGinRequest_t request = MakeWaitRequest(kPeer, snapshot, lastVisibleGet_[Idx(kContextId, kPeer)]);
  DeviceBuffer<uint32_t> d_abortFlag(1);
  d_abortFlag.upload(1u);
  DeviceBuffer<ncclResult_t> d_result(1);
  DeviceBuffer<uint32_t> d_state(2);
  for (const bool timed : {true, false}) {
    SCOPED_TRACE(timed ? "timeout overload" : "blocking overload");
    Upload();
    d_result.upload(ncclInternalError);
    d_state.zero();

    // The timeout overload bounds itself; the blocking one gets the watchdog block so a dropped flag fails, not hangs.
    kernelGinApiWaitAbort<<<timed ? 1 : 2, 1>>>(MakeCtx(), request, timed, d_abortFlag.ptr, kConsumerCycles,
                                                d_result.ptr, d_state.ptr);
    syncAndCheck();

    if (timed) {
      EXPECT_EQ(d_result.download(), ncclSuccess) << "timed Wait must leave its GFD gate via abortFlag, not time out";
    }
    EXPECT_EQ(d_state.copyTo(), (std::vector<uint32_t>{1u, 0u}))
      << "{returned, rescued}: Wait must return; rescued=1 (blocking arm only) means the watchdog had to drain cis";
    EXPECT_EQ(d_cis_.copyTo(), cis_) << "nothing consumed the pending GFD";
    EXPECT_EQ(d_pis_.copyTo(), pis_) << "gets are visible, so Wait must not post a flush GFD";
    expectQueuesZero(d_queues_, "aborted Wait");
    EXPECT_EQ(d_lastVisibleGet_.copyTo(), lastVisibleGet_);
  }
}

TEST_F(GinProxyFlushWaitTest, WaitPublishesLastIssuedGetOnceFlushConsumed) {
  // kIssued must differ from the live lastIssuedGet (0x7110) and be rolling-ahead of visible.
  constexpr uint32_t kIssued = 0x00000002u;
  lastVisibleGet_[Idx(kContextId, kPeer)] = 0xFFFFFFFEu;
  Upload();
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);

  kernelGinApiWait<<<2, 1>>>(MakeCtx(), MakeWaitRequest(kPeer, cis_[Idx(kContextId, kPeer)], kIssued), kRank,
                             kConsumerCycles, d_result.ptr);
  syncAndCheck();

  EXPECT_EQ(d_result.download(), ncclSuccess) << "the fake proxy consumed the flush, so Wait must succeed";
  std::vector<uint32_t> expectedPis = pis_;
  expectedPis[Idx(kContextId, kRank)] += 1;
  EXPECT_EQ(d_pis_.copyTo(), expectedPis);
  EXPECT_EQ(d_cis_.copyTo(), expectedPis) << "the fake proxy must have drained the ctx.rank queue";
  std::vector<uint32_t> expectedVisible = lastVisibleGet_;
  expectedVisible[Idx(kContextId, kPeer)] = kIssued;
  EXPECT_EQ(d_lastVisibleGet_.copyTo(), expectedVisible) << "Wait must publish req.lastIssuedGet for req.peer only";
}

// ---------------------------------------------------------------------------
// ResetSignal: proxy signals use an offset/baseline model. The effective value
//   of an indexed signal is signals[id] measured against a baseline held in
//   signalOffsets[id] (see ncclGinApi_GetSignalPtr, which returns
//   {signals + id, signalOffsets[id]}). "Reset" therefore does NOT zero the raw
//   signals[] cell -- it snapshots the current signals[id] into signalOffsets[id]
//   so the effective (relative) value becomes 0. Verify that only the targeted
//   offset cell is updated, that signals[] is left completely untouched, and
//   that the rest of the offset pool stays intact.
// ---------------------------------------------------------------------------

__global__ void kernelResetSignal(ncclGinCtx ctx, ncclGinSignal_t signalId) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinSignalDescriptor signal{};
  signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  signal.indexedSignal.signalId = signalId;
  ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, signal);
}

TEST_F(GinDeviceTest, ResetSignal) {
  constexpr uint32_t        kNumSignals   = 8;
  constexpr ncclGinSignal_t kTargetId     = 3;
  constexpr uint64_t        kPattern      = 0xA000ULL;   // signals[i]       = 0xA000 + i
  constexpr uint64_t        kOffsetSeed   = 0xC000ULL;   // signalOffsets[i] = 0xC000 + i

  DeviceBuffer<uint64_t>             d_signals(kNumSignals);
  DeviceBuffer<uint64_t>             d_signalOffsets(kNumSignals);
  DeviceBuffer<ncclGinProxyGpuCtx_t> d_proxyCtx(1);

  // Pre-fill signals with one pattern and offsets with a distinct one so we can
  // tell exactly which array (and which cell) the reset touches.
  std::vector<uint64_t> hostSignals(kNumSignals);
  std::vector<uint64_t> hostOffsets(kNumSignals);
  for (uint32_t i = 0; i < kNumSignals; i++) {
    hostSignals[i] = kPattern + i;
    hostOffsets[i] = kOffsetSeed + i;
  }
  d_signals.copyFrom(hostSignals);
  d_signalOffsets.copyFrom(hostOffsets);

  // ResetSignal (indexed) reads proxyCtx->signals and writes proxyCtx->signalOffsets.
  ncclGinProxyGpuCtx_t hostProxyCtx{};
  hostProxyCtx.signals       = d_signals.ptr;
  hostProxyCtx.signalOffsets = d_signalOffsets.ptr;
  d_proxyCtx.upload(hostProxyCtx);

  // Wrap the proxy ctx in an ncclGinCtx; the leaf only dereferences ctx.handle.
  ncclGinCtx ctx{};
  ctx.backend = NCCL_NET_DEVICE_GIN_PROXY;
  ctx.handle  = d_proxyCtx.ptr;

  kernelResetSignal<<<1, 1>>>(ctx, kTargetId);
  syncAndCheck();

  std::vector<uint64_t> signals = d_signals.copyTo();
  std::vector<uint64_t> offsets = d_signalOffsets.copyTo();

  // The raw signals[] array must be left completely untouched.
  for (uint32_t i = 0; i < kNumSignals; i++) {
    EXPECT_EQ(signals[i], kPattern + i)
        << "signals[" << i << "] unexpectedly modified (expected 0x" << std::hex << (kPattern + i) << ")";
  }

  // The target offset cell is snapshotted to the current signal value (making the
  // effective value 0); every other offset cell keeps its pre-filled value.
  EXPECT_EQ(offsets[kTargetId], kPattern + kTargetId)
      << "signalOffsets[" << kTargetId << "] must snapshot signals[" << kTargetId << "]";
  for (uint32_t i = 0; i < kNumSignals; i++) {
    if (i == kTargetId) continue;
    EXPECT_EQ(offsets[i], kOffsetSeed + i)
        << "signalOffsets[" << i << "] unexpectedly modified (expected 0x" << std::hex << (kOffsetSeed + i) << ")";
  }
}

// ---------------------------------------------------------------------------
// ResetCounter: ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_PROXY>::call writes
//   0 to counters[counterId] via base + offset. Mirror of ResetSignal but on
//   the separate counter pool (different base pointer, same failure shape).
// ---------------------------------------------------------------------------

__global__ void kernelResetCounter(ncclGinCtx ctx, ncclGinCounter_t counterId) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, counterId);
}

TEST_F(GinDeviceTest, ResetCounter) {
  constexpr uint32_t         kNumCounters = 8;
  constexpr ncclGinCounter_t kTargetId    = 5;
  constexpr uint64_t         kPattern     = 0xB000ULL;   // counters[i] = 0xB000 + i

  DeviceBuffer<uint64_t>             d_counters(kNumCounters);
  DeviceBuffer<ncclGinProxyGpuCtx_t> d_proxyCtx(1);

  // Pre-fill with non-zero pattern so any spurious zeroing surfaces as a mismatch.
  std::vector<uint64_t> hostCounters(kNumCounters);
  for (uint32_t i = 0; i < kNumCounters; i++) {
    hostCounters[i] = kPattern + i;
  }
  d_counters.copyFrom(hostCounters);

  // ResetCounter only reads proxyCtx->counters; the rest of the struct is untouched.
  ncclGinProxyGpuCtx_t hostProxyCtx{};
  hostProxyCtx.counters = d_counters.ptr;
  d_proxyCtx.upload(hostProxyCtx);

  // Wrap the proxy ctx in an ncclGinCtx; the leaf only dereferences ctx.handle.
  ncclGinCtx ctx{};
  ctx.backend = NCCL_NET_DEVICE_GIN_PROXY;
  ctx.handle  = d_proxyCtx.ptr;

  kernelResetCounter<<<1, 1>>>(ctx, kTargetId);
  syncAndCheck();

  std::vector<uint64_t> result = d_counters.copyTo();

  // Target cell zeroed; all other cells keep their pre-filled value.
  EXPECT_EQ(result[kTargetId], 0ULL) << "target counter " << kTargetId << " must be zeroed";
  for (uint32_t i = 0; i < kNumCounters; i++) {
    if (i == kTargetId) continue;
    EXPECT_EQ(result[i], kPattern + i)
        << "counter " << i << " unexpectedly modified (expected 0x" << std::hex << (kPattern + i) << ")";
  }
}

// ---------------------------------------------------------------------------
// ProxyFlush_*: proxy::flush snapshots pis[pe] and spins until cis[pe] reaches it; only one overload times out.
// ---------------------------------------------------------------------------

__global__ void kernelProxyFlush(ncclGinProxyGpuCtx_t* ctx, uint32_t pe, uint32_t* abortFlag, uint64_t timeoutCycles,
                                 ncclResult_t* outResult) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  *outResult = nccl::gin::proxy::flush(ctx, pe, cuda::memory_order_acquire, abortFlag, clock64(), timeoutCycles);
}

// Far longer than testAbort's abort-flag poll interval (utility.h maxSteps), yet bounds a flush that ignores abort.
constexpr uint64_t kProxyFlushLongTimeoutCycles = 1ULL << 28;
// A pending flush runs one testAbort poll interval (10000 loop iterations, utility.h) against one for a ready flush.
constexpr uint64_t kProxyFlushMinPendingToReadyRatio = 100;

// Device copies of pis and cis behind a proxy ctx; flush reads only those, so the GFD queue ring is left unset.
struct ProxyFlushQueues {
  ProxyFlushQueues(const std::vector<uint32_t>& pis, const std::vector<uint32_t>& cis)
    : hostPis(pis), hostCis(cis), d_pis(pis.size()), d_cis(cis.size()), d_ctx(1) {
    d_pis.copyFrom(pis);
    d_cis.copyFrom(cis);
    ncclGinProxyGpuCtx_t hostCtx{};
    hostCtx.nranks = static_cast<int>(pis.size());
    hostCtx.pis = d_pis.ptr;
    hostCtx.cis = d_cis.ptr;
    d_ctx.upload(hostCtx);
  }

  void ExpectUnchanged() {
    EXPECT_EQ(d_pis.copyTo(), hostPis) << "flush must not post a GFD";
    EXPECT_EQ(d_cis.copyTo(), hostCis) << "flush must not consume a GFD";
  }

  std::vector<uint32_t> hostPis;
  std::vector<uint32_t> hostCis;
  DeviceBuffer<uint32_t> d_pis;
  DeviceBuffer<uint32_t> d_cis;
  DeviceBuffer<ncclGinProxyGpuCtx_t> d_ctx;
};

// Runs one flush on peer pe and checks it never moves pi or ci; ncclInProgress means the kernel did not run.
static ncclResult_t runProxyFlush(const std::vector<uint32_t>& pis, const std::vector<uint32_t>& cis, uint32_t pe,
                                  uint32_t abortValue, uint64_t timeoutCycles) {
  SCOPED_TRACE(testing::Message() << "pe=" << pe << " abort=" << abortValue << " timeout=" << timeoutCycles);
  ProxyFlushQueues queues(pis, cis);
  DeviceBuffer<uint32_t> d_abort(1);
  DeviceBuffer<ncclResult_t> d_result(1);
  d_abort.upload(abortValue);
  d_result.upload(ncclInProgress);

  kernelProxyFlush<<<1, 1>>>(queues.d_ctx.ptr, pe, d_abort.ptr, timeoutCycles, d_result.ptr);
  EXPECT_EQ(hipGetLastError(), hipSuccess);
  EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
  queues.ExpectUnchanged();
  return d_result.download();
}

TEST_F(GinDeviceTest, ProxyFlush_SucceedsOnlyWhenCiReachesPiOfThatPeer) {
  // Peer 0 is fully consumed (ci == pi); peer 1 still has one GFD in flight.
  const std::vector<uint32_t> pis = {7u, 9u};
  const std::vector<uint32_t> cis = {7u, 8u};
  EXPECT_EQ(runProxyFlush(pis, cis, /*pe=*/0u, /*abortValue=*/0u, kProxyShortTimeoutCycles), ncclSuccess)
    << "ci == pi must count as complete";
  EXPECT_EQ(runProxyFlush(pis, cis, /*pe=*/0u, /*abortValue=*/0u, /*timeoutCycles=*/0u), ncclSuccess)
    << "a complete peer must succeed even with no time left; success is checked before the timeout";
  EXPECT_EQ(runProxyFlush(pis, cis, /*pe=*/1u, /*abortValue=*/0u, kProxyShortTimeoutCycles), ncclTimeout)
    << "flush must wait while ci < pi";
}

TEST_F(GinDeviceTest, ProxyFlush_ReadsPiAndCiOfTheSamePeer) {
  // Only peer 1 is complete; reading either neighbour's pi or ci for peer 1 makes it time out.
  const std::vector<uint32_t> pis = {10u, 5u, 9u};
  const std::vector<uint32_t> cis = {4u, 5u, 0u};
  EXPECT_EQ(runProxyFlush(pis, cis, /*pe=*/0u, /*abortValue=*/0u, kProxyShortTimeoutCycles), ncclTimeout)
    << "peer 0 has six GFDs pending";
  EXPECT_EQ(runProxyFlush(pis, cis, /*pe=*/1u, /*abortValue=*/0u, kProxyShortTimeoutCycles), ncclSuccess)
    << "peer 1 is fully consumed";
}

TEST_F(GinDeviceTest, ProxyFlush_ComparesAcrossUint32Wrap) {
  // pi wrapped to 2 while ci is still at 0xFFFFFFFE: four GFDs pending, so flush must time out.
  EXPECT_EQ(runProxyFlush({2u, 0u}, {0xFFFFFFFEu, 0u}, /*pe=*/0u, /*abortValue=*/0u, kProxyShortTimeoutCycles),
            ncclTimeout)
    << "a plain unsigned compare reads 2 <= 0xFFFFFFFE as complete";
  // ci wrapped to 1, past a stale pi snapshot of 0xFFFFFFFF: complete, so flush must succeed.
  EXPECT_EQ(runProxyFlush({0xFFFFFFFFu, 0u}, {1u, 0u}, /*pe=*/0u, /*abortValue=*/0u, kProxyShortTimeoutCycles),
            ncclSuccess)
    << "a plain unsigned compare reads 0xFFFFFFFF <= 1 as pending";
  // Half-range edge: ci up to 2^31-1 ahead of pi is complete; exactly 2^31 reads as behind. Fails for any width < 32.
  EXPECT_EQ(runProxyFlush({0u, 0u}, {0x7FFFFFFFu, 0u}, /*pe=*/0u, /*abortValue=*/0u, kProxyShortTimeoutCycles),
            ncclSuccess);
  EXPECT_EQ(runProxyFlush({0u, 0u}, {0x80000000u, 0u}, /*pe=*/0u, /*abortValue=*/0u, kProxyShortTimeoutCycles),
            ncclTimeout);
}

TEST_F(GinDeviceTest, ProxyFlush_AbortFlagExitsWithSuccessBeforeTimeout) {
  EXPECT_EQ(runProxyFlush({5u, 5u}, {4u, 5u}, /*pe=*/0u, /*abortValue=*/1u, kProxyFlushLongTimeoutCycles), ncclSuccess)
    << "a set abortFlag must end the spin through testAbort, not through the timeout";
}

// Block 0 times the fastest of three ready flushes against a pending no-timeout flush that only the abort poll ends.
// Block 1 drains peer 1 only if block 0 outlives kProxyWatchdogCycles; state is {returned, rescued}.
__global__ void kernelProxyFlushNoTimeout(ncclGinProxyGpuCtx_t* ctx, uint32_t* abortFlag, uint64_t* outCycles,
                                          uint32_t* state) {
  if (threadIdx.x != 0) {
    return;
  }
  if (blockIdx.x == 1) {
    drainAfterDeadline(ctx, /*pe=*/1, /*nDone=*/1u, kProxyWatchdogCycles, state);
    return;
  }
  nccl::gin::proxy::flush(ctx, /*pe=*/0u, cuda::memory_order_acquire, abortFlag); // warms the ctx, pis and cis lines
  uint64_t readyCycles = ~uint64_t{0};
  for (int i = 0; i < 3; ++i) {
    const uint64_t t0 = clock64();
    nccl::gin::proxy::flush(ctx, /*pe=*/0u, cuda::memory_order_acquire, abortFlag);
    const uint64_t t1 = clock64();
    if (t1 - t0 < readyCycles) {
      readyCycles = t1 - t0;
    }
  }
  const uint64_t t0 = clock64();
  nccl::gin::proxy::flush(ctx, /*pe=*/1u, cuda::memory_order_acquire, abortFlag);
  const uint64_t t1 = clock64();
  outCycles[0] = readyCycles;
  outCycles[1] = t1 - t0;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(state[0]).store(1u, cuda::memory_order_release);
}

TEST_F(GinDeviceTest, ProxyFlush_NoTimeoutOverloadWaitsForAbortPoll) {
  // Peer 0 is complete; peer 1 lags, so only the abort poll ends its wait. A ratio is independent of clock64 units.
  ProxyFlushQueues queues({3u, 5u}, {3u, 4u});
  DeviceBuffer<uint32_t> d_abort(1);
  DeviceBuffer<uint64_t> d_cycles(2);
  d_abort.upload(1u);
  d_cycles.zero();
  DeviceBuffer<uint32_t> d_state(2);
  d_state.zero();

  kernelProxyFlushNoTimeout<<<2, 1>>>(queues.d_ctx.ptr, d_abort.ptr, d_cycles.ptr, d_state.ptr);
  syncAndCheck();
  EXPECT_EQ(d_state.copyTo(), (std::vector<uint32_t>{1u, 0u}))
    << "{returned, rescued}: rescued=1 means the no-timeout flush ignored abortFlag and the watchdog drained cis[1]";

  const std::vector<uint64_t> cycles = d_cycles.copyTo();
  const uint64_t readyCycles = std::max<uint64_t>(cycles[0], 1);
  EXPECT_GT(cycles[1], kProxyFlushMinPendingToReadyRatio * readyCycles)
    << "a no-timeout flush returned before the first abort poll (or testAbort's poll interval shrank); ready="
    << cycles[0] << " pending=" << cycles[1];
  queues.ExpectUnchanged();
}

} // namespace RcclUnitTesting
