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
#include "nccl_device/gin/gin_device_host_common.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/proxy/gin_proxy.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"

#include <iomanip>

namespace RcclUnitTesting
{

class GinDeviceTest : public DeviceTestBase {};

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
// BuildGfd_PutOnly: pack a 64-byte GFD for a non-inline put (no signal, no counter)
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
  static_assert(sizeof(ncclGinProxyGfd_t) == 64, "GFD must be 64 bytes");

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

  // Header qword: flag, op, size.
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.flag), 1ULL);
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.op),
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

  // Reserved qword: end-of-GFD marker.
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdReserved].flag.v), 1ULL);
}

// ---------------------------------------------------------------------------
// BuildGfd_Inline: pack a 64-byte GFD for an inline put (hasInline=true).
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
  static_assert(sizeof(ncclGinProxyGfd_t) == 64, "GFD must be 64 bytes");

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
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.op),
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

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdReserved].flag.v), 1ULL) << "u32";
  }

  // ---- uint64_t (8-byte): value splits across Low (32) + Low2 (16) + High (16) ----
  {
    DeviceBuffer<ncclGinProxyGfd_t> d_gfd(1);
    d_gfd.zero();

    kernelBuildGfdInline<uint64_t><<<1, 1>>>(d_gfd.ptr, kValU64, kDstOff, kDstHandle);
    syncAndCheck();

    const ncclGinProxyGfd_t gfd = d_gfd.download();

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.flag), 1ULL) << "u64";
    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.op),
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

    EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdReserved].flag.v), 1ULL) << "u64";
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
  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdHeader].header.op),   expectedOp);
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

  EXPECT_EQ(static_cast<uint64_t>(gfd.qword[ncclGinProxyGfdReserved].flag.v), 1ULL);
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
  static_assert(sizeof(ncclGinProxyGfd_t) == 64, "GFD must be 64 bytes");

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

  // Input GFD: byte ramp 0x10..0x4F (64 unique non-zero bytes). Any byte the
  // kernel fails to write will read back as 0, distinguishable from the input.
  ncclGinProxyGfd_t hostGfd{};
  uint8_t* gfdBytes = reinterpret_cast<uint8_t*>(&hostGfd);
  for (int i = 0; i < 64; i++) gfdBytes[i] = static_cast<uint8_t>(0x10 + i);
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
  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(slotBytes[i], gfdBytes[i])
        << "slot 0 byte " << i << " mismatch";
  }

  // No other slot was written: peer 0 slots 1..3, plus all of peer 1.
  for (size_t s = 1; s < queues.size(); s++) {
    const uint8_t* zeroBytes = reinterpret_cast<const uint8_t*>(&queues[s]);
    for (int i = 0; i < 64; i++) {
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
// ResetSignal: ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_PROXY>::call writes
//   0 to signals[signalId] via base + offset; verify only the targeted cell
//   is touched and the rest of the pool stays intact.
// ---------------------------------------------------------------------------

__global__ void kernelResetSignal(ncclGinCtx ctx, ncclGinSignal_t signalId) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinSignalDescriptor signal{};
  signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  signal.indexedSignal.signalId = signalId;
  ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, signal);
}

TEST_F(GinDeviceTest, ResetSignal) {
  constexpr uint32_t        kNumSignals = 8;
  constexpr ncclGinSignal_t kTargetId   = 3;
  constexpr uint64_t        kPattern    = 0xA000ULL;   // signals[i] = 0xA000 + i

  DeviceBuffer<uint64_t>             d_signals(kNumSignals);
  DeviceBuffer<ncclGinProxyGpuCtx_t> d_proxyCtx(1);

  // Pre-fill with non-zero pattern so any spurious zeroing surfaces as a mismatch.
  std::vector<uint64_t> hostSignals(kNumSignals);
  for (uint32_t i = 0; i < kNumSignals; i++) {
    hostSignals[i] = kPattern + i;
  }
  d_signals.copyFrom(hostSignals);

  // ResetSignal only reads proxyCtx->signals; the rest of the struct is untouched.
  ncclGinProxyGpuCtx_t hostProxyCtx{};
  hostProxyCtx.signals = d_signals.ptr;
  d_proxyCtx.upload(hostProxyCtx);

  // Wrap the proxy ctx in an ncclGinCtx; the leaf only dereferences ctx.handle.
  ncclGinCtx ctx{};
  ctx.backend = NCCL_NET_DEVICE_GIN_PROXY;
  ctx.handle  = d_proxyCtx.ptr;

  kernelResetSignal<<<1, 1>>>(ctx, kTargetId);
  syncAndCheck();

  std::vector<uint64_t> result = d_signals.copyTo();

  // Target cell zeroed; all other cells keep their pre-filled value.
  EXPECT_EQ(result[kTargetId], 0ULL) << "target signal " << kTargetId << " must be zeroed";
  for (uint32_t i = 0; i < kNumSignals; i++) {
    if (i == kTargetId) continue;
    EXPECT_EQ(result[i], kPattern + i)
        << "signal " << i << " unexpectedly modified (expected 0x" << std::hex << (kPattern + i) << ")";
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

} // namespace RcclUnitTesting
