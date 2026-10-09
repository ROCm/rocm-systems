/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The NaN-flag protocols (NCCL_PROTO=NaN) read an all-ones dword as "not yet
// written", so a sender must never put one on the wire. Real data can hold one
// (as a NaN), and the sender escapes it by clearing bit 31. These tests feed such
// data through both implementations and check that every collective completes
// and delivers the escaped value:
//
//   DdaNanSentinelEscape.*   — the DDA push kernels (CollCommon_nan.h) on one
//                              GPU. Rank 0's peer pointers are offset so that a
//                              push to rank p lands in rank 0's own slot p, so a
//                              lone launch sees all eight sources arrive.
//   PrimsNanSentinelEscape.* — the ring/tree primitives (prims_nan.h) through the
//                              public API on 2-4 GPUs.
//
// Without the escape these hang; the child-process timeout reports that as a
// failure.

#include "DeviceTestBase.hpp"

#include "../common/ProcessIsolatedTestRunner.hpp"
#include "algorithms/dda/device/CollCommon.h"

#include <rccl/rccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// librccl exports the same kernel instantiations, and HIP kernel handles always
// have default visibility, so the dynamic linker would merge them with these and
// the launches below would run librccl's device code rather than this header's.
// Wrapping the header in an anonymous namespace gives its kernels internal
// linkage instead.
namespace
{
namespace under_test
{
namespace dda
{
namespace common = ::dda::common;
}
#include "algorithms/dda/device/CollCommon_nan.h"
} // namespace under_test
} // namespace

namespace RcclUnitTesting
{

namespace
{

constexpr uint32_t kAllOnes = 0xFFFFFFFFu;

uint32_t escapedDword(uint32_t x) { return x == kAllOnes ? 0x7FFFFFFFu : x; }

// Copy payload: sentinel dwords among near misses that must pass untouched. The
// last dword is always the sentinel so the tail of a slice is covered.
uint32_t copyDword(size_t i, size_t n, uint32_t seed) {
  if (i + 1 == n) return kAllOnes;
  switch (i % 7) {
  case 0: return kAllOnes;
  case 1: return 0xFFFF3C00u; // one all-ones f16/bf16 half next to a 1.0
  case 2: return 0x7FFFFFFFu; // already the escaped form
  case 3: return 0xFFFFFFFEu;
  default: return 0x40000000u | ((seed * 131u + (uint32_t)i) & 0x3FFFFFu);
  }
}

std::vector<uint32_t> copyDwords(size_t n, uint32_t seed) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = copyDword(i, n, seed);
  return v;
}

bool isHalfNaN(uint16_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0; }

// ---------------------------------------------------------------------------
// DDA kernels
// ---------------------------------------------------------------------------

namespace ddan = under_test::dda::nan;
namespace ddan = under_test::dda::nan;
using ddan::kRanks;
constexpr int    kSelf  = 0;
constexpr size_t kUnits = 1000; // 16B units per source, not a multiple of the block
constexpr size_t kDw    = kUnits * 4;
constexpr int    kIters = 2; // one call on each bank

// Rank kSelf's scratch with peer p's scratch placed p strides above it, so the
// slot (or mailbox) kSelf writes in peer p is kSelf's own slot p. The scratch
// kernels index slots by the call's unit count and the mailboxes by kEpochCells.
struct DdaLoopback {
  explicit DdaLoopback(size_t strideUnits)
    : scratch(ddan::kScratchBytes / sizeof(v4u) + kRanks * strideUnits),
      epoch(ddan::kEpochCells),
      stride(strideUnits) {
    HIP_EXPECT(hipMemset(scratch.ptr, 0xFF, scratch.count * sizeof(v4u)));
    HIP_EXPECT(hipMemset(epoch.ptr, 0, epoch.count * sizeof(uint32_t)));
  }
  ddan::Peers peers() const {
    ddan::Peers p;
    for (int r = 0; r < kRanks; r++) p.p[r] = scratch.ptr + r * stride;
    return p;
  }
  DeviceBuffer<v4u>      scratch;
  DeviceBuffer<uint32_t> epoch;
  size_t                 stride;
};

const auto kDdaGeom = ddan::geometry(kUnits, 4, 256);

template <bool kAllToAll, bool kReg>
void runDdaCopy() {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  const size_t           sendDw = (kAllToAll ? kRanks : 1) * kDw;
  DdaLoopback            lb(kReg ? ddan::kEpochCells : kUnits);
  DeviceBuffer<uint32_t> send(sendDw), recv(kRanks * kDw);
  const v4u*             s = reinterpret_cast<const v4u*>(send.ptr);
  v4u*                   r = reinterpret_cast<v4u*>(recv.ptr);
  for (int iter = 0; iter < kIters; iter++) {
    const std::vector<uint32_t> in = copyDwords(sendDw, iter);
    send.copyFrom(in);
    if constexpr (kReg) {
      // The address kSelf posts to rank p comes back as its own dst[p], so its
      // push into "rank p's" slice kSelf lands in slice p of this buffer.
      ddan::RegAddrs mine;
      for (int p = 0; p < kRanks; p++) mine.p[p] = r + p * kUnits;
      if constexpr (kAllToAll) {
        hipLaunchKernelGGL(ddan::ddaNanAllToAllReg<uint32_t>, kDdaGeom.first, kDdaGeom.second, 0, 0,
                           lb.peers(), lb.epoch.ptr, s, r, mine, kUnits, kSelf);
      } else {
        hipLaunchKernelGGL(ddan::ddaNanAllGatherReg<uint32_t>, kDdaGeom.first, kDdaGeom.second, 0, 0,
                           lb.peers(), lb.epoch.ptr, s, r, mine, kUnits, kSelf);
      }
    } else if constexpr (kAllToAll) {
      hipLaunchKernelGGL(ddan::ddaNanAllToAll<uint32_t>, kDdaGeom.first, kDdaGeom.second, 0, 0, lb.peers(),
                         lb.epoch.ptr, s, r, kUnits, kSelf);
    } else {
      hipLaunchKernelGGL(ddan::ddaNanAllGather<uint32_t>, kDdaGeom.first, kDdaGeom.second, 0, 0, lb.peers(),
                         lb.epoch.ptr, s, r, kUnits, kSelf);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint32_t> out = recv.copyTo();
    for (int src = 0; src < kRanks; src++) {
      for (size_t i = 0; i < kDw; i++) {
        const uint32_t x = in[(kAllToAll ? src * kDw : 0) + i];
        ASSERT_EQ(out[src * kDw + i], escapedDword(x))
          << "iter " << iter << " source " << src << " dword " << i << " sent 0x" << std::hex << x;
      }
    }
  }
}

constexpr uint32_t kOneOne = 0x3C003C00u; // two f16 1.0s

// Reduction payload, dword j of source block b: the sentinel in one source, and
// two sources that each poison one half, so that the sum can be all-ones and
// needs escaping before it is pushed again.
uint32_t reduceDword(int b, size_t j) {
  switch (j % 5) {
  case 0: return b == 0 ? kAllOnes : kOneOne;
  case 1: return b == 0 ? 0x0000FFFFu : b == 1 ? 0xFFFF0000u : 0u;
  default: return kOneOne;
  }
}

std::vector<uint32_t> reduceBlocks() {
  std::vector<uint32_t> v(kRanks * kDw);
  for (int b = 0; b < kRanks; b++)
    for (size_t j = 0; j < kDw; j++) v[b * kDw + j] = reduceDword(b, j);
  return v;
}

// Dword j of the sum of all eight reduceBlocks() sources.
void checkBlockSum(uint32_t got, size_t j, const std::string& where) {
  if (j % 5 < 2) {
    ASSERT_TRUE(isHalfNaN(got & 0xFFFF) && isHalfNaN(got >> 16)) << where << " got 0x" << std::hex << got;
  } else {
    ASSERT_EQ(got, 0x48004800u) << where; // 8.0, 8.0
  }
}

std::string where(int iter, size_t j, int slice = -1) {
  std::ostringstream o;
  o << "iter " << iter;
  if (slice >= 0) o << " slice " << slice;
  o << " dword " << j;
  return o.str();
}

// Loopback ReduceScatter: rank kSelf's push to rank p lands in source slot p, so
// the output is the sum of the eight blocks of its own input.
void runDdaReduceScatter() {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  DdaLoopback            lb(kUnits);
  DeviceBuffer<uint32_t> send(kRanks * kDw), recv(kDw);
  send.copyFrom(reduceBlocks());
  for (int iter = 0; iter < kIters; iter++) {
    hipLaunchKernelGGL(ddan::ddaNanReduceScatter<half>, kDdaGeom.first, kDdaGeom.second, 0, 0, lb.peers(),
                       lb.epoch.ptr, reinterpret_cast<const v4u*>(send.ptr), reinterpret_cast<v4u*>(recv.ptr),
                       kUnits, kSelf);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint32_t> out = recv.copyTo();
    for (size_t j = 0; j < kDw; j++) checkBlockSum(out[j], j, where(iter, j));
  }
}

// Loopback one-shot AllReduce: all eight sources are this rank's input.
void runDdaAllReduceOneShot() {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  DdaLoopback            lb(kUnits);
  DeviceBuffer<uint32_t> send(kDw), recv(kDw);
  std::vector<uint32_t>  in(kDw);
  for (size_t j = 0; j < kDw; j++) in[j] = reduceDword(0, j);
  send.copyFrom(in);
  for (int iter = 0; iter < kIters; iter++) {
    hipLaunchKernelGGL(ddan::ddaNanAllReduceOneShot<half>, kDdaGeom.first, kDdaGeom.second, 0, 0, lb.peers(),
                       lb.epoch.ptr, reinterpret_cast<const v4u*>(send.ptr), reinterpret_cast<v4u*>(recv.ptr),
                       kUnits, kSelf);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint32_t> out = recv.copyTo();
    for (size_t j = 0; j < kDw; j++) {
      const uint32_t got = out[j];
      switch (j % 5) {
      case 0:
        ASSERT_TRUE(isHalfNaN(got & 0xFFFF) && isHalfNaN(got >> 16)) << where(iter, j);
        break;
      case 1:
        ASSERT_TRUE(isHalfNaN(got & 0xFFFF) && (got >> 16) == 0) << where(iter, j);
        break;
      default: ASSERT_EQ(got, 0x48004800u) << where(iter, j);
      }
    }
  }
}

// Loopback two-shot AllReduce: stage 0 reduces the eight blocks of this rank's
// input, stage 1 pushes that sum to every slice. The sum can be all-ones, so
// stage 1 only completes if it is escaped.
void runDdaAllReduceTwoShot() {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  DdaLoopback            lb(kUnits);
  DeviceBuffer<uint32_t> send(kRanks * kDw), recv(kRanks * kDw);
  send.copyFrom(reduceBlocks());
  for (int iter = 0; iter < kIters; iter++) {
    hipLaunchKernelGGL(ddan::ddaNanAllReduceTwoShot<half>, kDdaGeom.first, kDdaGeom.second, 0, 0, lb.peers(),
                       lb.epoch.ptr, reinterpret_cast<const v4u*>(send.ptr), reinterpret_cast<v4u*>(recv.ptr),
                       kUnits, kSelf);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint32_t> out = recv.copyTo();
    for (int s = 0; s < kRanks; s++) {
      for (size_t j = 0; j < kDw; j++) {
        const uint32_t got = out[s * kDw + j];
        ASSERT_NE(got, kAllOnes) << where(iter, j, s);
        checkBlockSum(got, j, where(iter, j, s));
      }
    }
  }
}

ProcessIsolatedTestRunner::TestConfig ddaConfig(const char* name, void (*body)()) {
  return ProcessIsolatedTestRunner::TestConfig(name, body)
    .withTimeout(std::chrono::seconds(30))
    .withNumGpus(1);
}

// ---------------------------------------------------------------------------
// Ring/tree primitives, through the public API
// ---------------------------------------------------------------------------

struct NanDtype {
  ncclDataType_t type;
  size_t         size;
  const char*    name;
};
constexpr NanDtype kNanDtypes[] = {
  {ncclFloat32, 4, "f32"}, {ncclFloat16, 2, "f16"}, {ncclBfloat16, 2, "bf16"}, {ncclFloat64, 8, "f64"}};

// Elements per rank (or per peer): one under a row, and one spanning many
// slices and FIFO laps with a partial last slice. Both even, so 16-bit payloads
// fill whole dwords.
constexpr size_t kCounts[] = {72, 40008};

uint32_t loadDword(const std::vector<uint8_t>& b, size_t off) {
  uint32_t d;
  memcpy(&d, b.data() + off, 4);
  return d;
}
void storeDword(std::vector<uint8_t>& b, size_t off, uint32_t d) { memcpy(b.data() + off, &d, 4); }

// What the wire carries for `b`: f16/bf16/f32 escape any all-ones dword, f64 only
// an all-ones high dword (a finite double can have an all-ones low dword).
std::vector<uint8_t> escaped(std::vector<uint8_t> b, size_t elemSize) {
  if (elemSize <= 4) {
    for (size_t o = 0; o + 4 <= b.size(); o += 4) storeDword(b, o, escapedDword(loadDword(b, o)));
  } else {
    for (size_t o = 0; o < b.size(); o += 8) storeDword(b, o + 4, escapedDword(loadDword(b, o + 4)));
  }
  return b;
}

uint64_t copyQword(size_t i, size_t n, uint32_t seed) {
  if (i + 1 == n) return ~0ull;
  switch (i % 5) {
  case 0: return ~0ull;
  case 1: return 0x3FF00000FFFFFFFFull; // finite, all-ones low dword: unchanged
  case 2: return 0xFFFFFFFF00000000ull; // NaN, all-ones high dword
  case 3: return 0x7FF8000000000000ull;
  default: return 0x4000000000000000ull | ((uint64_t)(seed * 131u + i) & 0xFFFFFFFFFull);
  }
}

std::vector<uint8_t> copyPayload(size_t elemSize, size_t count, uint32_t seed) {
  std::vector<uint8_t> b(count * elemSize);
  if (elemSize <= 4) {
    const size_t n = b.size() / 4;
    for (size_t i = 0; i < n; i++) storeDword(b, 4 * i, copyDword(i, n, seed));
  } else {
    for (size_t i = 0; i < count; i++) {
      const uint64_t q = copyQword(i, count, seed);
      memcpy(b.data() + 8 * i, &q, 8);
    }
  }
  return b;
}

// Whether element e of a reducePayload() reduction is NaN.
bool reducePoisoned(size_t elemSize, size_t e) {
  if (elemSize == 8) return e % 11 == 0 || e % 11 == 5;
  const size_t j = e * elemSize / 4;
  return j % 11 == 0 || (elemSize == 2 && j % 11 == 5);
}

// Rank r's input, of n ranks. Every element is 1.0 except the poisoned ones: each
// takes an all-ones element (or for f64 an all-ones high dword) from one rank,
// and for 16-bit types some dwords take one all-ones half from rank 0 and the
// other from rank 1, so a partial sum is all-ones and must be escaped on its next
// hop.
std::vector<uint8_t> reducePayload(const NanDtype& dt, int r, int n, size_t count) {
  std::vector<uint8_t> b(count * dt.size);
  if (dt.size == 8) {
    for (size_t e = 0; e < count; e++) {
      uint64_t q = 0x3FF0000000000000ull;
      if (e % 11 == 0 && (int)((e / 11) % n) == r) q = ~0ull;
      if (e % 11 == 5 && (int)((e / 11 + 1) % n) == r) q = 0xFFFFFFFF00000000ull;
      memcpy(b.data() + 8 * e, &q, 8);
    }
    return b;
  }
  const uint32_t one = dt.type == ncclFloat32 ? 0x3F800000u : dt.type == ncclFloat16 ? kOneOne : 0x3F803F80u;
  for (size_t j = 0; j < b.size() / 4; j++) {
    uint32_t d = one;
    if (j % 11 == 0 && (int)((j / 11) % n) == r) d = kAllOnes;
    if (dt.size == 2 && j % 11 == 5) {
      if (r == 0) d = (one & 0xFFFF0000u) | 0x0000FFFFu;
      if (r == 1) d = 0xFFFF0000u | (one & 0x0000FFFFu);
    }
    storeDword(b, 4 * j, d);
  }
  return b;
}

double halfToDouble(uint16_t h) {
  const int    e = (h >> 10) & 0x1F, m = h & 0x3FF;
  const double s = (h & 0x8000) ? -1.0 : 1.0;
  if (e == 0x1F) return m ? NAN : s * INFINITY;
  if (e == 0) return s * std::ldexp(m, -24);
  return s * std::ldexp(m + 1024, e - 25);
}

double decode(const NanDtype& dt, const uint8_t* p) {
  switch (dt.type) {
  case ncclFloat32: {
    float f;
    memcpy(&f, p, 4);
    return f;
  }
  case ncclFloat64: {
    double d;
    memcpy(&d, p, 8);
    return d;
  }
  case ncclBfloat16: {
    uint16_t h;
    memcpy(&h, p, 2);
    const uint32_t w = (uint32_t)h << 16;
    float          f;
    memcpy(&f, &w, 4);
    return f;
  }
  default: {
    uint16_t h;
    memcpy(&h, p, 2);
    return halfToDouble(h);
  }
  }
}

// Empty if `got` matches `want` byte for byte, else the first differing element.
std::string firstDiff(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, size_t elemSize) {
  if (got == want) return "";
  for (size_t o = 0; o < want.size(); o += elemSize) {
    if (memcmp(got.data() + o, want.data() + o, elemSize) == 0) continue;
    uint64_t g = 0, w = 0;
    memcpy(&g, got.data() + o, elemSize);
    memcpy(&w, want.data() + o, elemSize);
    std::ostringstream s;
    s << "element " << o / elemSize << ": got 0x" << std::hex << g << ", want 0x" << w;
    return s.str();
  }
  return "size mismatch";
}

// Empty if every element of `out` (elements [first, ...) of the reduction of n
// reducePayload() inputs) is NaN where poisoned and n elsewhere.
std::string reduceDiff(const NanDtype& dt, const std::vector<uint8_t>& out, size_t first, int n) {
  for (size_t i = 0; i < out.size() / dt.size; i++) {
    const double v      = decode(dt, out.data() + i * dt.size);
    const bool   poison = reducePoisoned(dt.size, first + i);
    if (poison ? std::isnan(v) : v == n) continue;
    std::ostringstream s;
    s << "element " << first + i << ": got " << v << ", want " << (poison ? "NaN" : std::to_string(n));
    return s.str();
  }
  return "";
}

struct Ranks {
  int                      n = 0;
  std::vector<ncclComm_t>  comms;
  std::vector<hipStream_t> streams;
};

void initRanks(Ranks& R) {
  HIP_CHECK(hipGetDeviceCount(&R.n));
  R.n = std::min(R.n, 4);
  ASSERT_GE(R.n, 2);
  R.comms.resize(R.n);
  R.streams.resize(R.n);
  ASSERT_EQ(ncclCommInitAll(R.comms.data(), R.n, nullptr), ncclSuccess);
  for (int r = 0; r < R.n; r++) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipStreamCreate(&R.streams[r]));
  }
}

void finiRanks(Ranks& R) {
  for (int r = 0; r < R.n; r++) {
    HIP_EXPECT(hipSetDevice(r));
    HIP_EXPECT(hipStreamDestroy(R.streams[r]));
    EXPECT_EQ(ncclCommDestroy(R.comms[r]), ncclSuccess);
  }
}

using Issue = std::function<ncclResult_t(int r, const void* send, void* recv, hipStream_t stream)>;

// Runs one collective with in[r] as rank r's send buffer and returns each rank's
// recv buffer of recvBytes in out.
void runColl(Ranks& R, const std::vector<std::vector<uint8_t>>& in, size_t recvBytes, const Issue& issue,
             std::vector<std::vector<uint8_t>>& out) {
  std::vector<void*> send(R.n), recv(R.n);
  for (int r = 0; r < R.n; r++) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMalloc(&send[r], in[r].size()));
    HIP_CHECK(hipMalloc(&recv[r], recvBytes));
    HIP_CHECK(hipMemcpy(send[r], in[r].data(), in[r].size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(recv[r], 0, recvBytes));
  }
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < R.n; r++) {
    HIP_CHECK(hipSetDevice(r));
    ASSERT_EQ(issue(r, send[r], recv[r], R.streams[r]), ncclSuccess);
  }
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  out.assign(R.n, std::vector<uint8_t>(recvBytes));
  for (int r = 0; r < R.n; r++) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipStreamSynchronize(R.streams[r]));
    HIP_CHECK(hipMemcpy(out[r].data(), recv[r], recvBytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(send[r]));
    HIP_CHECK(hipFree(recv[r]));
  }
}

void primsAllGather(Ranks& R) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : kCounts) {
      SCOPED_TRACE(std::string(dt.name) + " count " + std::to_string(count));
      std::vector<std::vector<uint8_t>> in(R.n), out;
      std::vector<uint8_t>              want;
      for (int r = 0; r < R.n; r++) {
        in[r]           = copyPayload(dt.size, count, r);
        const auto wire = escaped(in[r], dt.size);
        want.insert(want.end(), wire.begin(), wire.end());
      }
      ASSERT_NO_FATAL_FAILURE(runColl(R, in, want.size(), [&](int r, const void* s, void* d, hipStream_t st) {
        return ncclAllGather(s, d, count, dt.type, R.comms[r], st);
      }, out));
      for (int r = 0; r < R.n; r++) EXPECT_EQ(firstDiff(out[r], want, dt.size), "") << "rank " << r;
    }
  }
}

// AlltoAll takes the NaN protocol on its send/recv path.
void primsAllToAll(Ranks& R) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : kCounts) {
      SCOPED_TRACE(std::string(dt.name) + " count " + std::to_string(count));
      const size_t                      block = count * dt.size;
      std::vector<std::vector<uint8_t>> in(R.n), wire(R.n), out;
      for (int r = 0; r < R.n; r++) {
        in[r]   = copyPayload(dt.size, count * R.n, r);
        wire[r] = escaped(in[r], dt.size);
      }
      ASSERT_NO_FATAL_FAILURE(runColl(R, in, block * R.n, [&](int r, const void* s, void* d, hipStream_t st) {
        return ncclAlltoAll(s, d, count, dt.type, R.comms[r], st);
      }, out));
      for (int q = 0; q < R.n; q++) {
        // A rank's block to itself is a local copy that never takes the wire.
        std::vector<uint8_t> want;
        for (int r = 0; r < R.n; r++) {
          const auto& src = r == q ? in[r] : wire[r];
          want.insert(want.end(), src.begin() + q * block, src.begin() + (q + 1) * block);
        }
        EXPECT_EQ(firstDiff(out[q], want, dt.size), "") << "rank " << q;
      }
    }
  }
}

void primsAllReduce(Ranks& R) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : kCounts) {
      SCOPED_TRACE(std::string(dt.name) + " count " + std::to_string(count));
      std::vector<std::vector<uint8_t>> in(R.n), out;
      for (int r = 0; r < R.n; r++) in[r] = reducePayload(dt, r, R.n, count);
      ASSERT_NO_FATAL_FAILURE(runColl(R, in, count * dt.size, [&](int r, const void* s, void* d, hipStream_t st) {
        return ncclAllReduce(s, d, count, dt.type, ncclSum, R.comms[r], st);
      }, out));
      for (int r = 0; r < R.n; r++) {
        EXPECT_EQ(reduceDiff(dt, out[r], 0, R.n), "") << "rank " << r;
        // Whoever forwards a reduced value escapes its own copy too.
        EXPECT_EQ(firstDiff(out[r], out[0], dt.size), "") << "rank " << r << " vs rank 0";
      }
    }
  }
}

void primsReduceScatter(Ranks& R) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : kCounts) {
      SCOPED_TRACE(std::string(dt.name) + " count " + std::to_string(count));
      std::vector<std::vector<uint8_t>> in(R.n), out;
      for (int r = 0; r < R.n; r++) in[r] = reducePayload(dt, r, R.n, count * R.n);
      ASSERT_NO_FATAL_FAILURE(runColl(R, in, count * dt.size, [&](int r, const void* s, void* d, hipStream_t st) {
        return ncclReduceScatter(s, d, count, dt.type, ncclSum, R.comms[r], st);
      }, out));
      for (int r = 0; r < R.n; r++) EXPECT_EQ(reduceDiff(dt, out[r], r * count, R.n), "") << "rank " << r;
    }
  }
}

// Communicator setup (mostly loading librccl's kernels) dwarfs the collectives,
// so each child process runs all of its collectives on one set of communicators.
void runPrimsRing() {
  Ranks R;
  ASSERT_NO_FATAL_FAILURE(initRanks(R));
  {
    SCOPED_TRACE("AllGather");
    primsAllGather(R);
  }
  {
    SCOPED_TRACE("AlltoAll");
    primsAllToAll(R);
  }
  {
    SCOPED_TRACE("AllReduce");
    primsAllReduce(R);
  }
  {
    SCOPED_TRACE("ReduceScatter");
    primsReduceScatter(R);
  }
  finiRanks(R);
}

void runPrimsTree() {
  Ranks R;
  ASSERT_NO_FATAL_FAILURE(initRanks(R));
  {
    SCOPED_TRACE("AllReduce");
    primsAllReduce(R);
  }
  finiRanks(R);
}

ProcessIsolatedTestRunner::TestConfig primsConfig(const char* name, const char* algo, int gpus, void (*body)()) {
  return ProcessIsolatedTestRunner::TestConfig(name, body)
    .withEnvironment({
      {"NCCL_PROTO", "NaN"},
      {"NCCL_ALGO", algo},
      {"RCCL_DDA_ENABLE", "0"},
      {"NCCL_IB_DISABLE", "1"},
      {"NCCL_SOCKET_IFNAME", "lo"},
    })
    .withTimeout(std::chrono::seconds(300))
    .withNumGpus(gpus);
}

// GPUs for the primitives tests, or 0 to skip.
int primsGpus() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return 0;
  return n < 2 ? 0 : std::min(n, 4);
}

} // namespace

TEST(DdaNanSentinelEscape, AllGather)
{
  RUN_ISOLATED_TESTS(ddaConfig("DdaNanSentinelEscape.AllGather", runDdaCopy<false, false>));
}

TEST(DdaNanSentinelEscape, AllToAll)
{
  RUN_ISOLATED_TESTS(ddaConfig("DdaNanSentinelEscape.AllToAll", runDdaCopy<true, false>));
}

TEST(DdaNanSentinelEscape, AllGatherReg)
{
  RUN_ISOLATED_TESTS(ddaConfig("DdaNanSentinelEscape.AllGatherReg", runDdaCopy<false, true>));
}

TEST(DdaNanSentinelEscape, AllToAllReg)
{
  RUN_ISOLATED_TESTS(ddaConfig("DdaNanSentinelEscape.AllToAllReg", runDdaCopy<true, true>));
}

TEST(DdaNanSentinelEscape, ReduceScatter)
{
  RUN_ISOLATED_TESTS(ddaConfig("DdaNanSentinelEscape.ReduceScatter", runDdaReduceScatter));
}

TEST(DdaNanSentinelEscape, AllReduceOneShot)
{
  RUN_ISOLATED_TESTS(ddaConfig("DdaNanSentinelEscape.AllReduceOneShot", runDdaAllReduceOneShot));
}

TEST(DdaNanSentinelEscape, AllReduceTwoShot)
{
  RUN_ISOLATED_TESTS(ddaConfig("DdaNanSentinelEscape.AllReduceTwoShot", runDdaAllReduceTwoShot));
}

// AllGather, AlltoAll, AllReduce and ReduceScatter.
TEST(PrimsNanSentinelEscape, Ring)
{
  const int gpus = primsGpus();
  if (!gpus) GTEST_SKIP() << "needs at least 2 GPUs";
  RUN_ISOLATED_TESTS(primsConfig("PrimsNanSentinelEscape.Ring", "Ring", gpus, runPrimsRing));
}

// AllReduce, the only collective with a NaN tree kernel.
TEST(PrimsNanSentinelEscape, Tree)
{
  const int gpus = primsGpus();
  if (!gpus) GTEST_SKIP() << "needs at least 2 GPUs";
  RUN_ISOLATED_TESTS(primsConfig("PrimsNanSentinelEscape.Tree", "Tree", gpus, runPrimsTree));
}

} // namespace RcclUnitTesting
