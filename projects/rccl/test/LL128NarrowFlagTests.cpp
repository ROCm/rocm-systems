/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Narrow LL128 flags (CMake RCCL_LL128_FLAG_BYTES = 4, 2 or 1). The tests run against the width librccl was built
// with, so one job per width covers each flag-lane layout; the default 8-byte build runs them as a baseline.
//
//   FlagEncoding, CleanSchedule  host only: the flag macros shared by device code and the net proxy.
//   Collectives{P2p,Net}         LL128 Ring collectives on 1-, 2-, 4- and 8-byte types at odd, power-of-two and
//                                misaligned sizes, over P2P and over the net proxy (Socket on lo, no GDR, so the
//                                proxy checks every line's flag before sending).
//   WrapAround{P2p,Net}          wraps the flag many times on one channel, with long runs of one-slice steps between
//                                steps that fill whole slots, so lines are left a whole flag period old. That breaks
//                                without the clean step or without forwarding the whole step over the net. Runs when
//                                the flag wraps within kMaxWrapPeriod steps: 1-byte flags, or a narrow width built
//                                with RCCL_LL128_FLAG_TEST=1 or 2.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "device.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace RcclUnitTesting
{

namespace
{

// Steps after which a narrow flag repeats; the 8-byte flag never wraps.
constexpr uint64_t kFlagPeriod = RCCL_LL128_NARROW_FLAG ? RCCL_LL128_FLAG_OF(~uint64_t(0)) + 1 : 0;
// Steps between two clean steps of the same slot.
constexpr uint64_t kCleanPeriod = uint64_t(RCCL_LL128_CLEAN_MASK) + NCCL_STEPS;
constexpr uint64_t kMaxWrapPeriod = 4096;

constexpr int kRanks = 2;

uint64_t mix(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

struct Bf16
{
  uint16_t bits;
};

// Element values: small integers for floating types, so sums over kRanks are exact even in bf16, and full-width
// patterns for integer types, so an 8-byte word assembled from the wrong halves shows up.
template <typename T> struct Elt;
template <> struct Elt<uint8_t>
{
  static constexpr ncclDataType_t kType = ncclUint8;
  static uint8_t make(uint64_t h) { return uint8_t(h); }
  static uint8_t add(uint8_t a, uint8_t b) { return uint8_t(a + b); }
};
template <> struct Elt<Bf16>
{
  static constexpr ncclDataType_t kType = ncclBfloat16;
  static float toFloat(Bf16 v)
  {
    uint32_t u = uint32_t(v.bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  }
  static Bf16 fromFloat(float f)
  {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return Bf16{uint16_t(u >> 16)};  // exact for the small integers used here
  }
  static Bf16 make(uint64_t h) { return fromFloat(float(h % 61)); }
  static Bf16 add(Bf16 a, Bf16 b) { return fromFloat(toFloat(a) + toFloat(b)); }
};
template <> struct Elt<float>
{
  static constexpr ncclDataType_t kType = ncclFloat32;
  static float make(uint64_t h) { return float(h % 61); }
  static float add(float a, float b) { return a + b; }
};
template <> struct Elt<uint64_t>
{
  static constexpr ncclDataType_t kType = ncclUint64;
  static uint64_t make(uint64_t h) { return h; }
  static uint64_t add(uint64_t a, uint64_t b) { return a + b; }
};

enum class Coll
{
  AllReduce,
  ReduceScatter,
  AllGather,
  Broadcast
};

const char* collName(Coll c)
{
  switch (c) {
  case Coll::AllReduce: return "AllReduce";
  case Coll::ReduceScatter: return "ReduceScatter";
  case Coll::AllGather: return "AllGather";
  default: return "Broadcast";
  }
}

struct Comms
{
  ncclComm_t comm[kRanks] = {};
  hipStream_t stream[kRanks] = {};

  void init()
  {
    int devs[kRanks];
    for (int r = 0; r < kRanks; r++) devs[r] = r;
    ASSERT_EQ(ncclCommInitAll(comm, kRanks, devs), ncclSuccess);
    for (int r = 0; r < kRanks; r++) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipStreamCreate(&stream[r]), hipSuccess);
    }
  }
  void sync()
  {
    for (int r = 0; r < kRanks; r++) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipStreamSynchronize(stream[r]), hipSuccess);
    }
  }
  ~Comms()
  {
    for (int r = 0; r < kRanks; r++) {
      if (stream[r] != nullptr) {
        (void)hipSetDevice(r);
        (void)hipStreamDestroy(stream[r]);
      }
      if (comm[r] != nullptr) (void)ncclCommDestroy(comm[r]);
    }
  }
};

// One collective on every rank. count is the elements each rank receives (per-rank block for ReduceScatter, per-rank
// contribution for AllGather). offset misaligns the user buffers by that many elements.
template <typename T> struct Run
{
  Coll coll;
  size_t count;
  size_t offset;
  size_t sendCount;
  size_t recvCount;
  std::vector<std::vector<T>> send;
  T* dSend[kRanks] = {};
  T* dRecv[kRanks] = {};

  Run(Coll c, size_t n, size_t off) : coll(c), count(n), offset(off), send(kRanks)
  {
    sendCount = coll == Coll::ReduceScatter ? count * kRanks : count;
    recvCount = coll == Coll::AllGather ? count * kRanks : count;
  }

  void alloc()
  {
    for (int r = 0; r < kRanks; r++) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipMalloc(&dSend[r], (sendCount + offset) * sizeof(T)), hipSuccess);
      ASSERT_EQ(hipMalloc(&dRecv[r], (recvCount + offset) * sizeof(T)), hipSuccess);
    }
  }

  void fill(uint64_t seed)
  {
    for (int r = 0; r < kRanks; r++) {
      send[r].resize(sendCount);
      for (size_t i = 0; i < sendCount; i++) send[r][i] = Elt<T>::make(mix(seed ^ mix(uint64_t(r) << 48 ^ i)));
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipMemcpy(dSend[r] + offset, send[r].data(), sendCount * sizeof(T), hipMemcpyHostToDevice),
                hipSuccess);
      ASSERT_EQ(hipMemset(dRecv[r] + offset, 0, recvCount * sizeof(T)), hipSuccess);
    }
  }

  void launch(Comms& c)
  {
    const ncclDataType_t type = Elt<T>::kType;
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int r = 0; r < kRanks; r++) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      T* s = dSend[r] + offset;
      T* d = dRecv[r] + offset;
      ncclResult_t res = ncclSuccess;
      switch (coll) {
      case Coll::AllReduce: res = ncclAllReduce(s, d, count, type, ncclSum, c.comm[r], c.stream[r]); break;
      case Coll::ReduceScatter: res = ncclReduceScatter(s, d, count, type, ncclSum, c.comm[r], c.stream[r]); break;
      case Coll::AllGather: res = ncclAllGather(s, d, count, type, c.comm[r], c.stream[r]); break;
      case Coll::Broadcast: res = ncclBroadcast(s, d, count, type, 0, c.comm[r], c.stream[r]); break;
      }
      ASSERT_EQ(res, ncclSuccess);
    }
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  }

  T expected(int r, size_t i) const
  {
    switch (coll) {
    case Coll::AllReduce:
    case Coll::ReduceScatter: {
      const size_t j = coll == Coll::ReduceScatter ? r * count + i : i;
      T v = send[0][j];
      for (int q = 1; q < kRanks; q++) v = Elt<T>::add(v, send[q][j]);
      return v;
    }
    case Coll::AllGather: return send[i / count][i % count];
    default: return send[0][i];
    }
  }

  void check(const char* what)
  {
    std::vector<T> got(recvCount);
    for (int r = 0; r < kRanks; r++) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipMemcpy(got.data(), dRecv[r] + offset, recvCount * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
      for (size_t i = 0; i < recvCount; i++) {
        const T want = expected(r, i);
        ASSERT_EQ(std::memcmp(&got[i], &want, sizeof(T)), 0)
          << what << ": " << collName(coll) << " " << sizeof(T) << "-byte type, count " << count << ", offset "
          << offset << ": rank " << r << " element " << i << " of " << recvCount << " is wrong";
      }
    }
  }

  ~Run()
  {
    for (int r = 0; r < kRanks; r++) {
      (void)hipSetDevice(r);
      if (dSend[r] != nullptr) (void)hipFree(dSend[r]);
      if (dRecv[r] != nullptr) (void)hipFree(dRecv[r]);
    }
  }
};

template <typename T> void runCollectivesOfType(Comms& c, uint64_t& seed)
{
  // Odd sizes end mid-line, mid-slice and on a lone word; power-of-two sizes end exactly at the buffer end, where a
  // flag-lane pair that starts on an odd word must not load past it.
  const size_t counts[] = {1, 5, 64, 1000, 4096, 12345, 65536, 262147};
  for (Coll coll : {Coll::AllReduce, Coll::ReduceScatter, Coll::AllGather, Coll::Broadcast}) {
    for (size_t count : counts) {
      for (size_t offset : {0, 1}) {
        Run<T> run(coll, count, offset);
        ASSERT_NO_FATAL_FAILURE(run.alloc());
        ASSERT_NO_FATAL_FAILURE(run.fill(++seed));
        ASSERT_NO_FATAL_FAILURE(run.launch(c));
        ASSERT_NO_FATAL_FAILURE(c.sync());
        ASSERT_NO_FATAL_FAILURE(run.check("collectives"));
      }
    }
  }
}

void runCollectives()
{
  Comms c;
  ASSERT_NO_FATAL_FAILURE(c.init());
  uint64_t seed = 0;
  ASSERT_NO_FATAL_FAILURE(runCollectivesOfType<uint8_t>(c, seed));
  ASSERT_NO_FATAL_FAILURE(runCollectivesOfType<Bf16>(c, seed));
  ASSERT_NO_FATAL_FAILURE(runCollectivesOfType<float>(c, seed));
  ASSERT_NO_FATAL_FAILURE(runCollectivesOfType<uint64_t>(c, seed));
}

// LL128 buffer for the wraparound tests: a few slices per step, so each collective spans many steps.
constexpr int kWrapBuffSize = 128 << 10;

void runWrapAround()
{
  Comms c;
  ASSERT_NO_FATAL_FAILURE(c.init());
  // A Ring allreduce takes 2 * (kRanks - 1) steps per chunk on each connection. The small one fits in one slice, so
  // it leaves the rest of each slot it uses untouched; the big one fills at least 8 whole steps per rank.
  constexpr uint64_t kSmallSteps = 2 * (kRanks - 1);
  constexpr size_t kStepBytes = kWrapBuffSize / NCCL_STEPS;
  Run<float> small(Coll::AllReduce, 64, 0);
  Run<uint64_t> big(Coll::AllReduce, kRanks * 8 * kStepBytes / sizeof(uint64_t), 0);
  ASSERT_NO_FATAL_FAILURE(small.alloc());
  ASSERT_NO_FATAL_FAILURE(big.alloc());
  ASSERT_NO_FATAL_FAILURE(small.fill(1));

  // Between two big allreduces, run enough small ones that the lines only the big one writes are about half, one or
  // one and a half flag periods old, over a spread of step alignments.
  uint64_t seed = 1;
  for (uint64_t target : {kFlagPeriod / 2, kFlagPeriod, 3 * kFlagPeriod / 2}) {
    for (int d = -NCCL_STEPS; d <= NCCL_STEPS; d++) {
      ASSERT_NO_FATAL_FAILURE(big.fill(++seed));
      ASSERT_NO_FATAL_FAILURE(big.launch(c));
      const int64_t nSmall = std::max<int64_t>(1, (int64_t(target) + d * int64_t(kSmallSteps)) / kSmallSteps);
      for (int64_t i = 0; i < nSmall; i++) ASSERT_NO_FATAL_FAILURE(small.launch(c));
      ASSERT_NO_FATAL_FAILURE(c.sync());
      ASSERT_NO_FATAL_FAILURE(big.check("wraparound"));
      ASSERT_NO_FATAL_FAILURE(small.check("wraparound"));
    }
  }
  ASSERT_NO_FATAL_FAILURE(big.fill(++seed));
  ASSERT_NO_FATAL_FAILURE(big.launch(c));
  ASSERT_NO_FATAL_FAILURE(c.sync());
  ASSERT_NO_FATAL_FAILURE(big.check("wraparound"));
}

std::unordered_map<std::string, std::string> ll128Env(bool net)
{
  std::unordered_map<std::string, std::string> env = {
    {"NCCL_PROTO", "LL128"},
    {"NCCL_ALGO", "Ring"},
    {"RCCL_LL128_FORCE_ENABLE", "1"},
    {"RCCL_DDA_ENABLE", "0"},
    {"RCCL_DIRECT_ALLGATHER_DISABLE", "1"},
  };
  if (net) {
    env["NCCL_P2P_DISABLE"] = "1";
    env["NCCL_SHM_DISABLE"] = "1";
    env["NCCL_IB_DISABLE"] = "1";
    env["NCCL_NET"] = "Socket";
    env["NCCL_SOCKET_IFNAME"] = "lo";
  }
  return env;
}

ProcessIsolatedTestRunner::TestConfig isolated(const char* name, std::function<void()> body,
                                               std::unordered_map<std::string, std::string> env)
{
  return ProcessIsolatedTestRunner::TestConfig(name, std::move(body))
    .withEnvironment(env)
    .withTimeout(std::chrono::seconds(600))
    .withNumGpus(kRanks);
}

std::unordered_map<std::string, std::string> wrapEnv(bool net)
{
  auto env = ll128Env(net);
  env["NCCL_MAX_NCHANNELS"] = "1";
  env["NCCL_LL128_BUFFSIZE"] = std::to_string(kWrapBuffSize);
  return env;
}

bool haveGpus()
{
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n >= kRanks;
}

} // namespace

TEST(LL128NarrowFlag, FlagEncoding)
{
  EXPECT_EQ(RCCL_LL128_FLAG_SHIFT, 64 - 8 * RCCL_LL128_FLAG_BYTES);
  const uint64_t dataMask = RCCL_LL128_FLAG_SHIFT ? (uint64_t(1) << RCCL_LL128_FLAG_SHIFT) - 1 : 0;
  for (uint64_t step1 : {0ull, 1ull, 31ull, 32ull, 255ull, 256ull, 65535ull, 65536ull, 0xffffffffull,
                         0x100000000ull, 0x123456789abcdefull}) {
    const uint64_t flag = RCCL_LL128_FLAG_OF(step1);
    EXPECT_EQ(flag, RCCL_LL128_NARROW_FLAG ? step1 % kFlagPeriod : step1) << "step1 " << step1;
    for (uint64_t data : {0ull, ~0ull, 0x0123456789abcdefull}) {
      // A flag lane's last word: data in the low bytes, the flag in the top FlagBytes. The device's flagMatches
      // and the net proxy's LL128 check both compare word >> RCCL_LL128_FLAG_SHIFT.
      const uint64_t word = (flag << RCCL_LL128_FLAG_SHIFT) | (data & dataMask);
      EXPECT_EQ(word >> RCCL_LL128_FLAG_SHIFT, flag) << "step1 " << step1;
      EXPECT_NE(word >> RCCL_LL128_FLAG_SHIFT, RCCL_LL128_FLAG_OF(step1 + 1)) << "step1 " << step1;
    }
  }
}

TEST(LL128NarrowFlag, CleanSchedule)
{
  if (!RCCL_LL128_NARROW_FLAG) {
    for (uint64_t step : {0ull, 7ull, 0x78ull, 0x7ff8ull, 0x7ffffff8ull, ~0ull}) {
      EXPECT_FALSE(RCCL_LL128_CLEAN_STEP(step)) << "the 8-byte flag never wraps, so it never cleans";
    }
    return;
  }
  // The clean steps are the last NCCL_STEPS steps of every kCleanPeriod, one per slot, and a slot is cleaned at
  // least twice per flag period, as in LL.
  ASSERT_EQ(kCleanPeriod & (kCleanPeriod - 1), 0u) << "RCCL_LL128_CLEAN_MASK must be contiguous bits";
  EXPECT_LE(2 * kCleanPeriod, kFlagPeriod);
  for (uint64_t base : {uint64_t(0), 3 * kCleanPeriod}) {
    EXPECT_FALSE(RCCL_LL128_CLEAN_STEP(base + kCleanPeriod - NCCL_STEPS - 1));
    for (int s = 0; s < NCCL_STEPS; s++) EXPECT_TRUE(RCCL_LL128_CLEAN_STEP(base + kCleanPeriod - NCCL_STEPS + s));
    EXPECT_FALSE(RCCL_LL128_CLEAN_STEP(base + kCleanPeriod));
  }
  if (kFlagPeriod > (1u << 16)) return;
  // Worst case: a line no data step reaches holds the flag of the last clean step of its slot (0 before the first).
  // It must never already hold the flag of the step about to be sent.
  uint64_t stale[NCCL_STEPS] = {};
  for (uint64_t step = 0; step < 4 * kFlagPeriod; step++) {
    const uint64_t flag = RCCL_LL128_FLAG_OF(step + 1);
    ASSERT_NE(stale[step % NCCL_STEPS], flag) << "step " << step;
    if (RCCL_LL128_CLEAN_STEP(step)) stale[step % NCCL_STEPS] = flag;
  }
}

TEST(LL128NarrowFlag, CollectivesP2p)
{
  if (!haveGpus()) GTEST_SKIP() << "needs " << kRanks << " GPUs";
  RUN_ISOLATED_TESTS(isolated("LL128NarrowFlag.CollectivesP2p", runCollectives, ll128Env(false)));
}

TEST(LL128NarrowFlag, CollectivesNet)
{
  if (!haveGpus()) GTEST_SKIP() << "needs " << kRanks << " GPUs";
  RUN_ISOLATED_TESTS(isolated("LL128NarrowFlag.CollectivesNet", runCollectives, ll128Env(true)));
}

TEST(LL128NarrowFlag, WrapAroundP2p)
{
  if (kFlagPeriod == 0 || kFlagPeriod > kMaxWrapPeriod) {
    GTEST_SKIP() << "the flag wraps every " << kFlagPeriod << " steps; build with RCCL_LL128_FLAG_BYTES=1 or "
                 << "RCCL_LL128_FLAG_TEST=1|2 to run";
  }
  if (!haveGpus()) GTEST_SKIP() << "needs " << kRanks << " GPUs";
  RUN_ISOLATED_TESTS(isolated("LL128NarrowFlag.WrapAroundP2p", runWrapAround, wrapEnv(false)));
}

TEST(LL128NarrowFlag, WrapAroundNet)
{
  if (kFlagPeriod == 0 || kFlagPeriod > kMaxWrapPeriod) {
    GTEST_SKIP() << "the flag wraps every " << kFlagPeriod << " steps; build with RCCL_LL128_FLAG_BYTES=1 or "
                 << "RCCL_LL128_FLAG_TEST=1|2 to run";
  }
  if (!haveGpus()) GTEST_SKIP() << "needs " << kRanks << " GPUs";
  RUN_ISOLATED_TESTS(isolated("LL128NarrowFlag.WrapAroundNet", runWrapAround, wrapEnv(true)));
}

} // namespace RcclUnitTesting
