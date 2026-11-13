/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The NaN-flag protocols (NCCL_PROTO=NaN) read an all-ones dword as "not yet
// written", so a sender must never put one on the wire. Real data can hold one
// (as a NaN), and the sender escapes it to another NaN by clearing bit 0. These
// tests feed such data through the ring/tree primitives (prims_nan.h) via the
// public API on 2-4 GPUs (8 for Direct AllGather) and check that every
// collective completes and delivers the escaped value.
//
// Without the escape these hang; the child-process timeout reports that as a
// failure.

#include "DeviceTestBase.hpp"

#include "../common/ProcessIsolatedTestRunner.hpp"

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

namespace RcclUnitTesting
{

namespace
{

constexpr uint32_t kAllOnes = 0xFFFFFFFFu;

uint32_t escapedDword(uint32_t x) { return x == kAllOnes ? 0xFFFFFFFEu : x; }

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

constexpr uint32_t kOneOne = 0x3C003C00u; // two f16 1.0s

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
const std::vector<size_t> kCounts = {72, 40008};
// The TDM runs add one spanning many windows per warp, again with a partial
// last row.
const std::vector<size_t> kTdmCounts = {72, 40008, 1000002};

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

// Empty if no dword the wire escapes (every dword, or each f64's high dword) is still all-ones. Other protocols
// carry an all-ones value through unchanged, so this holds only where a NaN kernel forwarded every value.
std::string sentinelLeft(size_t elemSize, const std::vector<uint8_t>& out) {
  const size_t first = elemSize == 8 ? 4 : 0, stride = elemSize == 8 ? 8 : 4;
  for (size_t o = first; o + 4 <= out.size(); o += stride) {
    if (loadDword(out, o) != kAllOnes) continue;
    std::ostringstream s;
    s << "all-ones dword at byte " << o;
    return s.str();
  }
  return "";
}

// Whether some dword the wire escapes reads 0xFFFFFFFE, which only the escape produces from these payloads.
bool anyEscaped(size_t elemSize, const std::vector<uint8_t>& out) {
  const size_t first = elemSize == 8 ? 4 : 0, stride = elemSize == 8 ? 8 : 4;
  for (size_t o = first; o + 4 <= out.size(); o += stride) {
    if (loadDword(out, o) == 0xFFFFFFFEu) return true;
  }
  return false;
}

struct Ranks {
  int                      n = 0;
  std::vector<ncclComm_t>  comms;
  std::vector<hipStream_t> streams;
};

void initRanks(Ranks& R, int maxRanks = 4) {
  HIP_CHECK(hipGetDeviceCount(&R.n));
  R.n = std::min(R.n, maxRanks);
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
// recv buffer of recvBytes in out. In place, rank r's send buffer is its slice r
// of the recv buffer, as for AllGather.
void runColl(Ranks& R, const std::vector<std::vector<uint8_t>>& in, size_t recvBytes, const Issue& issue,
             std::vector<std::vector<uint8_t>>& out, bool inPlace = false) {
  std::vector<void*> send(R.n), recv(R.n);
  for (int r = 0; r < R.n; r++) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMalloc(&recv[r], recvBytes));
    HIP_CHECK(hipMemset(recv[r], 0, recvBytes));
    if (inPlace) {
      send[r] = static_cast<char*>(recv[r]) + r * in[r].size();
    } else {
      HIP_CHECK(hipMalloc(&send[r], in[r].size()));
    }
    HIP_CHECK(hipMemcpy(send[r], in[r].data(), in[r].size(), hipMemcpyHostToDevice));
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
    if (!inPlace) HIP_CHECK(hipFree(send[r]));
    HIP_CHECK(hipFree(recv[r]));
  }
}

// Every rank, the owner included, holds each slice as the wire carries it.
void primsAllGather(Ranks& R, const std::vector<size_t>& counts, bool inPlace = false) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : counts) {
      SCOPED_TRACE(std::string(dt.name) + " count " + std::to_string(count) + (inPlace ? " in place" : ""));
      std::vector<std::vector<uint8_t>> in(R.n), out;
      std::vector<uint8_t>              want;
      for (int r = 0; r < R.n; r++) {
        in[r]           = copyPayload(dt.size, count, r);
        const auto wire = escaped(in[r], dt.size);
        want.insert(want.end(), wire.begin(), wire.end());
      }
      ASSERT_NO_FATAL_FAILURE(runColl(R, in, want.size(), [&](int r, const void* s, void* d, hipStream_t st) {
        return ncclAllGather(s, d, count, dt.type, R.comms[r], st);
      }, out, inPlace));
      for (int r = 0; r < R.n; r++) EXPECT_EQ(firstDiff(out[r], want, dt.size), "") << "rank " << r;
    }
  }
}

// AlltoAll takes the NaN protocol on its send/recv path over intra-node links. Over the network its halves stay on
// the usual protocols (addP2pToPlan), so the data arrives as sent.
void primsAllToAll(Ranks& R, const std::vector<size_t>& counts, bool overNet) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : counts) {
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
          const auto& src = r == q || overNet ? in[r] : wire[r];
          want.insert(want.end(), src.begin() + q * block, src.begin() + (q + 1) * block);
        }
        EXPECT_EQ(firstDiff(out[q], want, dt.size), "") << "rank " << q;
      }
    }
  }
}

void primsAllReduce(Ranks& R, const std::vector<size_t>& counts) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : counts) {
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
        EXPECT_EQ(sentinelLeft(dt.size, out[r]), "") << "rank " << r;
      }
    }
  }
}

void primsReduceScatter(Ranks& R, const std::vector<size_t>& counts) {
  for (const NanDtype& dt : kNanDtypes) {
    for (size_t count : counts) {
      SCOPED_TRACE(std::string(dt.name) + " count " + std::to_string(count));
      std::vector<std::vector<uint8_t>> in(R.n), out;
      for (int r = 0; r < R.n; r++) in[r] = reducePayload(dt, r, R.n, count * R.n);
      ASSERT_NO_FATAL_FAILURE(runColl(R, in, count * dt.size, [&](int r, const void* s, void* d, hipStream_t st) {
        return ncclReduceScatter(s, d, count, dt.type, ncclSum, R.comms[r], st);
      }, out));
      // The last hop reduces without forwarding, so the owner's own all-ones can stay; what came over the wire
      // arrives escaped.
      bool escaped = false;
      for (int r = 0; r < R.n; r++) {
        EXPECT_EQ(reduceDiff(dt, out[r], r * count, R.n), "") << "rank " << r;
        escaped |= anyEscaped(dt.size, out[r]);
      }
      EXPECT_TRUE(escaped) << "no output carries an escaped dword";
    }
  }
}

// Communicator setup (mostly loading librccl's kernels) dwarfs the collectives,
// so each child process runs all of its collectives on one set of communicators.
template <const std::vector<size_t>* Counts, bool OverNet = false>
void runPrimsRing() {
  Ranks R;
  ASSERT_NO_FATAL_FAILURE(initRanks(R));
  {
    SCOPED_TRACE("AllGather");
    primsAllGather(R, *Counts);
    primsAllGather(R, *Counts, /*inPlace=*/true);
  }
  {
    SCOPED_TRACE("AlltoAll");
    primsAllToAll(R, *Counts, OverNet);
  }
  {
    SCOPED_TRACE("AllReduce");
    primsAllReduce(R, *Counts);
  }
  {
    SCOPED_TRACE("ReduceScatter");
    primsReduceScatter(R, *Counts);
  }
  finiRanks(R);
}

void runPrimsTree() {
  Ranks R;
  ASSERT_NO_FATAL_FAILURE(initRanks(R));
  {
    SCOPED_TRACE("AllReduce");
    primsAllReduce(R, kCounts);
  }
  finiRanks(R);
}

// Direct AllGather takes a multiple of 8 ranks: per-peer send/recv on the NaN-flag protocol, and a copy to itself.
void runPrimsDirect() {
  Ranks R;
  ASSERT_NO_FATAL_FAILURE(initRanks(R, 8));
  {
    SCOPED_TRACE("AllGather");
    primsAllGather(R, kCounts);
    primsAllGather(R, kCounts, /*inPlace=*/true);
  }
  finiRanks(R);
}

// tdmMinBytes, when set, is RCCL_TDM_NAN_MIN_BYTES. Builds without ENABLE_TDM_NAN ignore it. Without algo,
// NCCL_ALGO stays unset. extraEnv is added last, e.g. to steer peers off P2P.
ProcessIsolatedTestRunner::TestConfig primsConfig(const char* name, const char* algo, int gpus, void (*body)(),
                                                  const char* tdmMinBytes = nullptr,
                                                  const std::unordered_map<std::string, std::string>& extraEnv = {}) {
  std::unordered_map<std::string, std::string> env = {
    {"NCCL_PROTO", "NaN"},
    {"RCCL_DDA_ENABLE", "0"},
    {"NCCL_IB_DISABLE", "1"},
    {"NCCL_SOCKET_IFNAME", "lo"},
    {"RCCL_TDM_NAN_ENABLE", tdmMinBytes ? "1" : "0"},
  };
  env.insert(extraEnv.begin(), extraEnv.end());
  if (algo) env["NCCL_ALGO"] = algo;
  if (tdmMinBytes) env["RCCL_TDM_NAN_MIN_BYTES"] = tdmMinBytes;
  return ProcessIsolatedTestRunner::TestConfig(name, body)
    .withEnvironment(env)
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

// AllGather, AlltoAll, AllReduce and ReduceScatter.
TEST(PrimsNanSentinelEscape, Ring)
{
  const int gpus = primsGpus();
  if (!gpus) GTEST_SKIP() << "needs at least 2 GPUs";
  RUN_ISOLATED_TESTS(primsConfig("PrimsNanSentinelEscape.Ring", "Ring", gpus, runPrimsRing<&kCounts>));
}

// The same collectives over the SHM transport, which seeds and restores the NaN FIFO in host memory.
TEST(PrimsNanSentinelEscape, RingShm)
{
  const int gpus = primsGpus();
  if (!gpus) GTEST_SKIP() << "needs at least 2 GPUs";
  RUN_ISOLATED_TESTS(primsConfig("PrimsNanSentinelEscape.RingShm", "Ring", gpus, runPrimsRing<&kCounts>,
                                 /*tdmMinBytes=*/nullptr, {{"NCCL_P2P_DISABLE", "1"}}));
}

// The same collectives over the net transport (sockets on lo), whose proxy sizes and maps the NaN FIFO.
TEST(PrimsNanSentinelEscape, RingNet)
{
  const int gpus = primsGpus();
  if (!gpus) GTEST_SKIP() << "needs at least 2 GPUs";
  RUN_ISOLATED_TESTS(primsConfig("PrimsNanSentinelEscape.RingNet", "Ring", gpus, runPrimsRing<&kCounts, true>,
                                 /*tdmMinBytes=*/nullptr, {{"NCCL_P2P_DISABLE", "1"}, {"NCCL_SHM_DISABLE", "1"}}));
}

// The same collectives with every primitive op on the TDM path (gfx1250 builds
// with -DENABLE_TDM_NAN=ON; elsewhere this repeats the register path).
TEST(PrimsNanSentinelEscape, RingTdm)
{
  const int gpus = primsGpus();
  if (!gpus) GTEST_SKIP() << "needs at least 2 GPUs";
  RUN_ISOLATED_TESTS(
    primsConfig("PrimsNanSentinelEscape.RingTdm", "Ring", gpus, runPrimsRing<&kTdmCounts>, /*tdmMinBytes=*/"0"));
}

// Direct AllGather, on gfx942 and gfx950 with 8 GPUs.
TEST(PrimsNanSentinelEscape, AllGatherDirect)
{
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < 8) GTEST_SKIP() << "needs 8 GPUs";
  RUN_ISOLATED_TESTS(primsConfig("PrimsNanSentinelEscape.AllGatherDirect", nullptr, 8, runPrimsDirect));
}

// AllReduce, the only collective with a NaN tree kernel.
TEST(PrimsNanSentinelEscape, Tree)
{
  const int gpus = primsGpus();
  if (!gpus) GTEST_SKIP() << "needs at least 2 GPUs";
  RUN_ISOLATED_TESTS(primsConfig("PrimsNanSentinelEscape.Tree", "Tree", gpus, runPrimsTree));
}

} // namespace RcclUnitTesting
