/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Device-side tests for nccl::utility (nccl_device/utility.h): device-only helpers and the NCCL_DEVICE_ARCH branches.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "DeviceTestBase.hpp"

#include "nccl_device/utility.h"

namespace RcclUnitTesting {
namespace {

namespace nu = nccl::utility;

class NcclUtilityDeviceTest : public DeviceTestBase {};

uint64_t refMul64hi(uint64_t a, uint64_t b) {
  return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
}

// ---------------------------------------------------------------------------
// Integer math on device: mul32hi/mul64hi take the nccl_umulhi branch here, the __int128 branch on host.
// ---------------------------------------------------------------------------

struct MathOut {
  uint32_t mul32hi;
  uint64_t mul64hi;
  uint32_t quo32;
  uint32_t rem32;
  uint32_t idiv32;
  uint32_t imod32;
  uint64_t quo64;
  uint64_t rem64;
  uint32_t idiv64;
  uint64_t imod64;
  uint32_t imulRcp32;
  uint64_t imulRcp64;
};

struct MathIn {
  uint64_t a;
  uint64_t b;
};

__global__ void kernelMath(const MathIn* in, MathOut* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const uint64_t a = in[i].a;
  const uint64_t b = in[i].b;
  const uint32_t a32 = static_cast<uint32_t>(a);
  const uint32_t b32 = static_cast<uint32_t>(b) == 0 ? 1u : static_cast<uint32_t>(b);
  const uint64_t b64 = b == 0 ? 1 : b;
  MathOut o;
  o.mul32hi = nu::mul32hi(a32, static_cast<uint32_t>(b));
  o.mul64hi = nu::mul64hi(a, b);
  nu::idivmodFast32(&o.quo32, &o.rem32, a32, b32, nu::idivRcp32(b32));
  o.idiv32 = nu::idivFast32(a32, b32, nu::idivRcp32(b32));
  o.imod32 = nu::imodFast32(a32, b32, nu::idivRcp32(b32));
  nu::idivmodFast64(&o.quo64, &o.rem64, a, b64, nu::idivRcp64(b64));
  o.idiv64 = nu::idivFast64(a, b64, nu::idivRcp64(b64));
  o.imod64 = nu::imodFast64(a, b64, nu::idivRcp64(b64));
  const uint32_t x = (a32 & 0xFFFu) + 1;
  const uint32_t y = (static_cast<uint32_t>(b) & 0xFFFu) + 1;
  o.imulRcp32 = nu::imulRcp32(x, nu::idivRcp32(x), y, nu::idivRcp32(y));
  o.imulRcp64 = nu::imulRcp64(x, nu::idivRcp64(x), y, nu::idivRcp64(y));
  out[i] = o;
}

TEST_F(NcclUtilityDeviceTest, MathMatchesHostReference) {
  std::vector<MathIn> in = {
      {0, 1},
      {1, 1},
      {0xFFFFFFFFull, 0xFFFFFFFFull},
      {~uint64_t{0}, ~uint64_t{0}},
      {~uint64_t{0}, 1},
      {0x8000000000000000ull, 2},
      {0x123456789ABCDEFull, 0xFEDCBA987654321ull},
      {1000, 7},
      {0xFFFFFFFFull, 3},
      {0x7FFFFFFFull, 0x80000000ull},
      {(uint64_t{1} << 40) + 5, 1},
      {0, 0},
  };
  uint64_t s = 0x2545F4914F6CDD1Dull;
  for (int i = 0; i < 4096; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    const uint64_t a = s;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    in.push_back({a, s >> (s % 64)});
  }
  DeviceBuffer<MathIn> d_in(in.size());
  DeviceBuffer<MathOut> d_out(in.size());
  ASSERT_NO_FATAL_FAILURE(d_in.copyFrom(in));
  ASSERT_NO_FATAL_FAILURE(d_out.zero());
  kernelMath<<<gridFor(in.size()), kDefaultBlockSize>>>(d_in.ptr, d_out.ptr, static_cast<int>(in.size()));
  ASSERT_NO_FATAL_FAILURE(syncAndCheck());
  const std::vector<MathOut> out = d_out.copyTo();
  for (size_t i = 0; i < in.size(); ++i) {
    const uint64_t a = in[i].a;
    const uint64_t b = in[i].b;
    const uint32_t a32 = static_cast<uint32_t>(a);
    const uint32_t b32 = static_cast<uint32_t>(b) == 0 ? 1u : static_cast<uint32_t>(b);
    const uint64_t b64 = b == 0 ? 1 : b;
    SCOPED_TRACE(::testing::Message() << "i=" << i << " a=" << a << " b=" << b);
    const MathOut& o = out[i];
    EXPECT_EQ(o.mul32hi, static_cast<uint32_t>((uint64_t{a32} * static_cast<uint32_t>(b)) >> 32));
    EXPECT_EQ(o.mul32hi, nu::mul32hi(a32, static_cast<uint32_t>(b)));
    EXPECT_EQ(o.mul64hi, refMul64hi(a, b));
    EXPECT_EQ(o.mul64hi, nu::mul64hi(a, b));
    EXPECT_EQ(o.quo32, a32 / b32);
    EXPECT_EQ(o.rem32, a32 % b32);
    EXPECT_EQ(o.idiv32, a32 / b32);
    EXPECT_EQ(o.imod32, a32 % b32);
    EXPECT_EQ(o.quo64, a / b64);
    EXPECT_EQ(o.rem64, a % b64);
    EXPECT_EQ(o.idiv64, static_cast<uint32_t>(a / b64));  // idivFast64 narrows to uint32_t on device too.
    EXPECT_EQ(o.imod64, a % b64);
    const uint32_t x = (a32 & 0xFFFu) + 1;
    const uint32_t y = (static_cast<uint32_t>(b) & 0xFFFu) + 1;
    EXPECT_EQ(o.imulRcp32, nu::idivRcp32(x * y));
    EXPECT_EQ(o.imulRcp64, nu::idivRcp64(uint64_t{x} * y));
    if (::testing::Test::HasFailure()) {
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// idivRcp64_upto64 / idivRcp32_upto64: precomputed device tables for 1..64.
// ---------------------------------------------------------------------------

constexpr int kRcpTableSize = 65;

__global__ void kernelRcpTables(uint64_t* rcp64, uint32_t* rcp32, uint32_t* quotients) {
  const int x = threadIdx.x;
  if (x >= kRcpTableSize) {
    return;
  }
  rcp64[x] = nu::idivRcp64_upto64(x);
  rcp32[x] = nu::idivRcp32_upto64(x);
  if (x >= 1) {
    // Dividing through the table must give exact quotients for any 64-bit dividend whose quotient fits 32 bits.
    quotients[x] = nu::idivFast64(0xFFFFFFFFull * x + (x - 1), x, nu::idivRcp64_upto64(x));
  }
}

TEST_F(NcclUtilityDeviceTest, ReciprocalTablesUpTo64) {
  DeviceBuffer<uint64_t> d_rcp64(kRcpTableSize);
  DeviceBuffer<uint32_t> d_rcp32(kRcpTableSize);
  DeviceBuffer<uint32_t> d_quo(kRcpTableSize);
  ASSERT_NO_FATAL_FAILURE(d_rcp64.zero());
  ASSERT_NO_FATAL_FAILURE(d_rcp32.zero());
  ASSERT_NO_FATAL_FAILURE(d_quo.zero());
  kernelRcpTables<<<1, 128>>>(d_rcp64.ptr, d_rcp32.ptr, d_quo.ptr);
  ASSERT_NO_FATAL_FAILURE(syncAndCheck());
  const std::vector<uint64_t> rcp64 = d_rcp64.copyTo();
  const std::vector<uint32_t> rcp32 = d_rcp32.copyTo();
  const std::vector<uint32_t> quo = d_quo.copyTo();
  // Entry 0 duplicates entry 1 instead of dividing by zero.
  EXPECT_EQ(rcp64[0], nu::idivRcp64(1));
  EXPECT_EQ(rcp32[0], nu::idivRcp32(1));
  for (int x = 1; x < kRcpTableSize; ++x) {
    SCOPED_TRACE(::testing::Message() << "x=" << x);
    EXPECT_EQ(rcp64[x], nu::idivRcp64(static_cast<uint64_t>(x)));
    EXPECT_EQ(rcp32[x], nu::idivRcp32(static_cast<uint32_t>(x)));
    EXPECT_EQ(quo[x], 0xFFFFFFFFu);
  }
}

// ---------------------------------------------------------------------------
// testAbort (device overload: atomic_ref relaxed load of the flag every 10000th call)
// ---------------------------------------------------------------------------

constexpr uint32_t kMaxSteps = 10000;  // utility.h testAbort's poll interval.

struct AbortOut {
  uint32_t firstTrueCall;  // 1-based; 0 if never true within the bound.
  uint32_t stepsAfter;
  uint32_t falsesBefore;
};

// Bounded: at most 3 intervals of calls, so a broken counter fails the test instead of spinning.
__global__ void kernelTestAbort(uint32_t* flag, uint32_t startSteps, AbortOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  uint32_t steps = startSteps;
  AbortOut o{0, 0, 0};
  for (uint32_t call = 1; call <= 3 * kMaxSteps; ++call) {
    if (nu::testAbort(flag, steps)) {
      o.firstTrueCall = call;
      break;
    }
    ++o.falsesBefore;
  }
  o.stepsAfter = steps;
  *out = o;
}

void runTestAbort(uint32_t* d_flag, uint32_t startSteps, AbortOut* out) {
  DeviceBuffer<AbortOut> d_out(1);
  ASSERT_NO_FATAL_FAILURE(d_out.zero());
  kernelTestAbort<<<1, 1>>>(d_flag, startSteps, d_out.ptr);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  *out = d_out.download();
}

TEST_F(NcclUtilityDeviceTest, TestAbortPollsOncePerInterval) {
  DeviceBuffer<uint32_t> d_flag(1);
  ASSERT_NO_FATAL_FAILURE(d_flag.upload(1u));
  AbortOut o{};
  ASSERT_NO_FATAL_FAILURE(runTestAbort(d_flag.ptr, 0, &o));
  EXPECT_EQ(o.firstTrueCall, kMaxSteps);
  EXPECT_EQ(o.falsesBefore, kMaxSteps - 1);
  EXPECT_EQ(o.stepsAfter, 0u);
  ASSERT_NO_FATAL_FAILURE(runTestAbort(d_flag.ptr, kMaxSteps - 2, &o));
  EXPECT_EQ(o.firstTrueCall, 2u);
  // A count already past the interval polls on the next call.
  ASSERT_NO_FATAL_FAILURE(runTestAbort(d_flag.ptr, kMaxSteps + 7, &o));
  EXPECT_EQ(o.firstTrueCall, 1u);
  EXPECT_EQ(o.stepsAfter, 0u);
  ASSERT_NO_FATAL_FAILURE(d_flag.upload(0x80000000u));
  ASSERT_NO_FATAL_FAILURE(runTestAbort(d_flag.ptr, kMaxSteps - 1, &o));
  EXPECT_EQ(o.firstTrueCall, 1u);
}

TEST_F(NcclUtilityDeviceTest, TestAbortClearOrNullFlagNeverAborts) {
  DeviceBuffer<uint32_t> d_flag(1);
  ASSERT_NO_FATAL_FAILURE(d_flag.upload(0u));
  AbortOut o{};
  ASSERT_NO_FATAL_FAILURE(runTestAbort(d_flag.ptr, 0, &o));
  EXPECT_EQ(o.firstTrueCall, 0u);
  EXPECT_EQ(o.falsesBefore, 3 * kMaxSteps);
  EXPECT_EQ(o.stepsAfter, 0u);  // Reset at each of the three polls, then no further calls.
  ASSERT_NO_FATAL_FAILURE(runTestAbort(nullptr, 0, &o));
  EXPECT_EQ(o.firstTrueCall, 0u);
  EXPECT_EQ(o.stepsAfter, 0u);
  ASSERT_NO_FATAL_FAILURE(runTestAbort(d_flag.ptr, 5, &o));
  EXPECT_EQ(o.stepsAfter, 5u);  // Polls reset at calls 9995, 19995 and 29995; 5 calls remain.
}

// ---------------------------------------------------------------------------
// opaqueZero, lane, lanemask_lt
// ---------------------------------------------------------------------------

__global__ void kernelOpaqueZero(int* out, const int* table) {
  const int z = nu::opaqueZero();
  out[0] = z;
  out[1] = table[threadIdx.x + nu::opaqueZero()];
}

TEST_F(NcclUtilityDeviceTest, OpaqueZeroIsZero) {
  DeviceBuffer<int> d_out(2);
  DeviceBuffer<int> d_table(1);
  ASSERT_NO_FATAL_FAILURE(d_out.copyFrom(std::vector<int>{-1, -1}));
  ASSERT_NO_FATAL_FAILURE(d_table.upload(1234));
  kernelOpaqueZero<<<1, 1>>>(d_out.ptr, d_table.ptr);
  ASSERT_NO_FATAL_FAILURE(syncAndCheck());
  const std::vector<int> out = d_out.copyTo();
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 1234);
}

__global__ void kernelLanes(int* lanes, unsigned int* masks) {
  lanes[threadIdx.x] = nu::lane();
  masks[threadIdx.x] = nu::lanemask_lt();
}

TEST_F(NcclUtilityDeviceTest, LaneAndLanemaskLt) {
  int device = 0;
  ASSERT_EQ(hipGetDevice(&device), hipSuccess);
  int warpSize = 0;
  ASSERT_EQ(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, device), hipSuccess);
  ASSERT_TRUE(warpSize == 32 || warpSize == 64) << warpSize;
  const int nThreads = 4 * warpSize;
  DeviceBuffer<int> d_lanes(nThreads);
  DeviceBuffer<unsigned int> d_masks(nThreads);
  ASSERT_NO_FATAL_FAILURE(d_lanes.zero());
  ASSERT_NO_FATAL_FAILURE(d_masks.zero());
  kernelLanes<<<1, nThreads>>>(d_lanes.ptr, d_masks.ptr);
  ASSERT_NO_FATAL_FAILURE(syncAndCheck());
  const std::vector<int> lanes = d_lanes.copyTo();
  const std::vector<unsigned int> masks = d_masks.copyTo();
  for (int t = 0; t < nThreads; ++t) {
    const int lane = t % warpSize;
    // Only the low 32 lanes fit the 32-bit mask: wave64 lanes 32..63 report all of lanes 0..31 as lower.
    const unsigned int mask = lane >= 32 ? 0xFFFFFFFFu : (1u << lane) - 1;
    EXPECT_EQ(lanes[t], lane) << "t=" << t;
    EXPECT_EQ(masks[t], mask) << "t=" << t;
  }
}

// ---------------------------------------------------------------------------
// atomicLoad / atomicStore: every scope and the legal orders, for 2-, 4- and 8-byte types.
// ---------------------------------------------------------------------------

// Switches rather than constexpr tables: a host array indexed at run time is not addressable from device code.
__device__ cuda::thread_scope scopeAt(int i) {
  switch (i) {
    case 0:
      return cuda::thread_scope_thread;
    case 1:
      return cuda::thread_scope_block;
    case 2:
      return cuda::thread_scope_device;
    default:
      return cuda::thread_scope_system;
  }
}
__device__ cuda::memory_order storeOrderAt(int i) {
  return i == 0 ? cuda::memory_order_relaxed : i == 1 ? cuda::memory_order_release : cuda::memory_order_seq_cst;
}
__device__ cuda::memory_order loadOrderAt(int i) {
  return i == 0 ? cuda::memory_order_relaxed : i == 1 ? cuda::memory_order_acquire : cuda::memory_order_seq_cst;
}

constexpr int kNumScopes = 4;
constexpr int kNumOrders = 3;
constexpr int kNumSlots = kNumScopes * kNumOrders;

template <typename T>
__host__ __device__ T slotValue(int slot) {
  return static_cast<T>(0xA5A5A5A5C3C3C3C3ull + 0x0101010101010101ull * static_cast<uint64_t>(slot));
}

// Slot s: store with (scope s/3, store order s%3), then load back with (scope s/3, load order s%3).
template <typename T>
__global__ void kernelAtomicRoundTrip(T* mem, T* loaded, T* preloaded) {
  const int s = threadIdx.x;
  if (s >= kNumSlots) {
    return;
  }
  const cuda::thread_scope scope = scopeAt(s / kNumOrders);
  preloaded[s] = nu::atomicLoad(mem + s, loadOrderAt(s % kNumOrders), scope);
  nu::atomicStore(mem + s, slotValue<T>(s), storeOrderAt(s % kNumOrders), scope);
  loaded[s] = nu::atomicLoad(mem + s, loadOrderAt(s % kNumOrders), scope);
}

template <typename T>
void runAtomicRoundTrip() {
  SCOPED_TRACE(::testing::Message() << "sizeof(T)=" << sizeof(T));
  std::vector<T> init(kNumSlots);
  for (int s = 0; s < kNumSlots; ++s) {
    init[s] = static_cast<T>(~slotValue<T>(s));
  }
  DeviceBuffer<T> d_mem(kNumSlots);
  DeviceBuffer<T> d_loaded(kNumSlots);
  DeviceBuffer<T> d_pre(kNumSlots);
  ASSERT_NO_FATAL_FAILURE(d_mem.copyFrom(init));
  ASSERT_NO_FATAL_FAILURE(d_loaded.zero());
  ASSERT_NO_FATAL_FAILURE(d_pre.zero());
  kernelAtomicRoundTrip<T><<<1, 64>>>(d_mem.ptr, d_loaded.ptr, d_pre.ptr);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  const std::vector<T> mem = d_mem.copyTo();
  const std::vector<T> loaded = d_loaded.copyTo();
  const std::vector<T> pre = d_pre.copyTo();
  for (int s = 0; s < kNumSlots; ++s) {
    EXPECT_EQ(pre[s], init[s]) << "slot " << s;
    EXPECT_EQ(loaded[s], slotValue<T>(s)) << "slot " << s;
    EXPECT_EQ(mem[s], slotValue<T>(s)) << "slot " << s;
  }
}

TEST_F(NcclUtilityDeviceTest, AtomicLoadStoreRoundTripAllScopesAndOrders) {
  ASSERT_NO_FATAL_FAILURE(runAtomicRoundTrip<uint32_t>());
  ASSERT_NO_FATAL_FAILURE(runAtomicRoundTrip<uint64_t>());
  ASSERT_NO_FATAL_FAILURE(runAtomicRoundTrip<int32_t>());
  ASSERT_NO_FATAL_FAILURE(runAtomicRoundTrip<uint16_t>());  // sizeof 2 takes atomic_ref's unscoped __atomic fallback.
}

// ---------------------------------------------------------------------------
// fenceReleaseGpu / fenceAcquireGpu: message passing between two blocks, bounded so it fails instead of hanging.
// ---------------------------------------------------------------------------

constexpr int kMpWords = 256;
constexpr uint32_t kMpSpinBound = 1u << 26;

__global__ void kernelMessagePass(uint32_t* data, uint32_t* flag, uint32_t* seen, uint32_t* timedOut) {
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < kMpWords; i += blockDim.x) {
      data[i] = 0xC0DE0000u + i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      nu::fenceReleaseGpu();
      nu::atomicStore(flag, 1u, cuda::memory_order_relaxed, cuda::thread_scope_device);
    }
  } else {
    if (threadIdx.x == 0) {
      uint32_t spins = 0;
      while (nu::atomicLoad(flag, cuda::memory_order_relaxed, cuda::thread_scope_device) == 0) {
        if (++spins == kMpSpinBound) {
          *timedOut = 1;
          break;
        }
      }
      nu::fenceAcquireGpu();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kMpWords; i += blockDim.x) {
      seen[i] = data[i];
    }
  }
}

TEST_F(NcclUtilityDeviceTest, FenceReleaseAcquireMessagePassing) {
  DeviceBuffer<uint32_t> d_data(kMpWords);
  DeviceBuffer<uint32_t> d_flag(1);
  DeviceBuffer<uint32_t> d_seen(kMpWords);
  DeviceBuffer<uint32_t> d_timedOut(1);
  ASSERT_NO_FATAL_FAILURE(d_data.zero());
  ASSERT_NO_FATAL_FAILURE(d_flag.zero());
  ASSERT_NO_FATAL_FAILURE(d_seen.zero());
  ASSERT_NO_FATAL_FAILURE(d_timedOut.zero());
  kernelMessagePass<<<2, 64>>>(d_data.ptr, d_flag.ptr, d_seen.ptr, d_timedOut.ptr);
  ASSERT_NO_FATAL_FAILURE(syncAndCheck());
  ASSERT_EQ(d_timedOut.download(), 0u) << "consumer block never saw the producer's flag";
  const std::vector<uint32_t> seen = d_seen.copyTo();
  for (int i = 0; i < kMpWords; ++i) {
    EXPECT_EQ(seen[i], 0xC0DE0000u + i) << "word " << i;
  }
}

// ---------------------------------------------------------------------------
// loadConstLdg (every alignof branch) and loadConst (plain dereference while NCCL_DEVICE_LOADCONST_USE_LDG is 0)
// ---------------------------------------------------------------------------

static_assert(NCCL_DEVICE_LOADCONST_USE_LDG == 0, "loadConst is expected to be a plain *p load; update the tests");

struct Align1 {
  uint8_t b[5];
};
struct Align2 {
  uint16_t h[3];
};
struct Align4 {
  uint32_t w[3];
};
struct Align8 {
  uint64_t d[3];
};
struct alignas(16) Align16 {
  uint8_t b[48];
};
struct alignas(32) Align32 {
  uint8_t b[64];
};
static_assert(alignof(Align1) == 1 && sizeof(Align1) == 5, "alignof 1 branch, 5 byte loads");
static_assert(alignof(Align2) == 2 && sizeof(Align2) == 6, "alignof 2 branch, 3 halfword loads");
static_assert(alignof(Align4) == 4 && sizeof(Align4) == 12, "alignof 4 branch, 3 word loads");
static_assert(alignof(Align8) == 8 && sizeof(Align8) == 24, "alignof 8 branch, 3 dword loads");
static_assert(alignof(Align16) == 16 && sizeof(Align16) == 48, "alignof >= 16 branch, 3 ulonglong2 loads");
static_assert(alignof(Align32) == 32 && sizeof(Align32) == 64, "alignof >= 16 branch also takes alignof 32");

// Every byte of every element differs, so a wrong lane, a short loop or a swapped word shows up.
template <typename T>
std::vector<T> distinctBytePattern(int count, uint8_t salt) {
  std::vector<T> v(count);
  std::vector<uint8_t> raw(sizeof(T) * count);
  for (size_t k = 0; k < raw.size(); ++k) {
    raw[k] = static_cast<uint8_t>(k * 37 + salt);
  }
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

constexpr int kLoadElems = 3;

// Thread i loads element i, so non-zero element offsets are exercised too.
template <typename T>
__global__ void kernelLoadConst(const T* src, T* outLdg, T* outConst) {
  const int i = threadIdx.x;
  if (i >= kLoadElems) {
    return;
  }
  outLdg[i] = nu::loadConstLdg(src + i);
  outConst[i] = nu::loadConst(src + i);
}

template <typename T>
void runLoadConst(uint8_t salt) {
  SCOPED_TRACE(::testing::Message() << "alignof=" << alignof(T) << " sizeof=" << sizeof(T));
  const std::vector<T> host = distinctBytePattern<T>(kLoadElems, salt);
  DeviceBuffer<T> d_src(kLoadElems);
  DeviceBuffer<T> d_ldg(kLoadElems);
  DeviceBuffer<T> d_const(kLoadElems);
  ASSERT_NO_FATAL_FAILURE(d_src.copyFrom(host));
  ASSERT_NO_FATAL_FAILURE(d_ldg.zero());
  ASSERT_NO_FATAL_FAILURE(d_const.zero());
  kernelLoadConst<T><<<1, 64>>>(d_src.ptr, d_ldg.ptr, d_const.ptr);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  const std::vector<T> ldg = d_ldg.copyTo();
  const std::vector<T> plain = d_const.copyTo();
  for (int i = 0; i < kLoadElems; ++i) {
    EXPECT_EQ(std::memcmp(&ldg[i], &host[i], sizeof(T)), 0) << "loadConstLdg element " << i;
    EXPECT_EQ(std::memcmp(&plain[i], &host[i], sizeof(T)), 0) << "loadConst element " << i;
  }
}

TEST_F(NcclUtilityDeviceTest, LoadConstLdgEveryAlignmentBranch) {
  ASSERT_NO_FATAL_FAILURE(runLoadConst<Align1>(0x11));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<Align2>(0x22));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<Align4>(0x33));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<Align8>(0x44));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<Align16>(0x55));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<Align32>(0x66));
}

TEST_F(NcclUtilityDeviceTest, LoadConstLdgScalars) {
  ASSERT_NO_FATAL_FAILURE(runLoadConst<uint8_t>(0x71));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<int16_t>(0x72));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<uint32_t>(0x73));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<float>(0x74));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<double>(0x75));
  ASSERT_NO_FATAL_FAILURE(runLoadConst<uint64_t>(0x76));
}

}  // namespace
}  // namespace RcclUnitTesting
