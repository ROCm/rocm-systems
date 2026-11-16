/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for datamover/dataMover.h: every pair of endpoint spaces, the fan-out
// rule for multiple sources/destinations, and several backend chains. A chain
// whose DMA links are not compiled for the device under test still runs the
// test through its vector fallback; the DMA-only chains skip instead, so they
// are the ones that prove TDM / async-to-LDS really moved the bytes.
//
// Every destination sits between sentinel guards, so stray writes fail too.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "datamover/dataMover.h"

namespace RcclUnitTesting
{

namespace dm = rccl::dm;

constexpr size_t  kDmGuard    = 256;
constexpr uint8_t kDmSentinel = 0xAB;
constexpr int     kDmBlock    = 256;
constexpr int     kDmPacks    = 4;   // RegTile depth: 16 KiB per tile at 256 threads
constexpr int     kDmLdsPacks = 8;   // 32 KiB: holds the largest LDS case

using VecChain   = dm::DataMover<dm::VectorMover<>>;
using NtVecChain = dm::DataMover<dm::VectorMover<createCachePolicy(TemporalHint::NT, MemScope::DEV), 2>>;
using TdmChain   = dm::DataMover<dm::TdmMover<>, dm::VectorMover<>>;
using AsyncChain = dm::DataMover<dm::AsyncLdsMover<>, dm::VectorMover<>>;
using TdmOnly    = dm::DataMover<dm::TdmMover<>>;
using AsyncOnly  = dm::DataMover<dm::AsyncLdsMover<>>;

// ---------------------------------------------------------------------------
//  Building a chain on device: LDS DMA links get the staging window.
// ---------------------------------------------------------------------------

template<class B>
struct BackendInit {
  __device__ static B make(void*, uint32_t) { return B(); }
};
template<class E, CachePolicy CP>
struct BackendInit<dm::LdsDmaMover<E, CP>> {
  __device__ static dm::LdsDmaMover<E, CP> make(void* s, uint32_t n) { return {s, n}; }
};

template<class M>
struct ChainInit;
template<class... Bs>
struct ChainInit<dm::DataMover<Bs...>> {
  __device__ static dm::DataMover<Bs...> make(void* staging, uint32_t bytes) {
    return dm::DataMover<Bs...>(BackendInit<Bs>::make(staging, bytes)...);
  }
};

template<class M, dm::Space S, dm::Space D>
constexpr bool kCanMove = M::template anyEligible<S, D>();

// ---------------------------------------------------------------------------
//  Kernels
// ---------------------------------------------------------------------------

constexpr int kMaxSrcs = 2;
constexpr int kMaxDsts = 4;

struct MoveArgs {
  uint8_t* src[kMaxSrcs];
  uint8_t* dst[kMaxDsts];
  int      nSrcs;
  int      nDsts;
  size_t   bytes;
  uint32_t ldsBytes;   // dynamic LDS: staging for G->G, buffers for the LDS cases
  uint32_t ldsOff;     // byte skew of the LDS buffers
  uint32_t ldsStride;  // distance between the two LDS buffers
};

// Global -> Global with the runtime source/destination counts in `a`.
template<class M>
__global__ void kGlobalToGlobal(MoveArgs a, uint32_t firstWave, uint32_t nWaves) {
  extern __shared__ __align__(256) uint8_t lds[];
  if constexpr (kCanMove<M, dm::Space::Global, dm::Space::Global>) {
    const uint32_t w = threadIdx.x / dm::waveSize();
    if (w < firstWave || w >= firstWave + nWaves) return;
    const dm::Team team = dm::Team::waves(firstWave, nWaves);
    M mover = ChainInit<M>::make(lds, a.ldsBytes);
    mover.move(team, dm::memSet<dm::Space::Global, kMaxSrcs>(a.src, a.nSrcs),
               dm::memSet<dm::Space::Global, kMaxDsts>(a.dst, a.nDsts), a.bytes);
  }
}

// src0, src1 -> two LDS buffers (pairs); LDS0 -> dst0, dst1 (broadcast);
// LDS1 -> dst2. Both LDS legs of the second step are in flight together.
template<class M>
__global__ void kViaLds(MoveArgs a) {
  extern __shared__ __align__(256) uint8_t lds[];
  if constexpr (kCanMove<M, dm::Space::Global, dm::Space::Lds> &&
                kCanMove<M, dm::Space::Lds, dm::Space::Global>) {
    const dm::Team team = dm::Team::block();
    M mover = ChainInit<M>::make(nullptr, 0);
    uint8_t* l0 = lds + a.ldsOff;
    uint8_t* l1 = lds + a.ldsStride + a.ldsOff;
    mover.move(team, dm::global(a.src[0], a.src[1]), dm::lds(l0, l1), a.bytes);
    __syncthreads();
    mover.issue(team, dm::lds(l0), dm::global(a.dst[0], a.dst[1]), a.bytes);
    mover.issue(team, dm::lds(l1), dm::global(a.dst[2]), a.bytes);
    mover.wait(team);
  }
}

// src0, src1 -> two register tiles -> two more tiles -> dst0..dst3 with
// fan-out 2 (tile 0 -> dst0, dst1; tile 1 -> dst2, dst3).
template<class M>
__global__ void kViaRegs(MoveArgs a) {
  const dm::Team team = dm::Team::block();
  M mover = ChainInit<M>::make(nullptr, 0);
  dm::RegTile<kDmPacks, 2> t0;
  dm::RegTile<kDmPacks, 2> t1{};
  mover.move(team, dm::global(a.src[0], a.src[1]), t0, a.bytes);
  mover.move(team, t0, t1, a.bytes);
  mover.move(team, t1, dm::global(a.dst[0], a.dst[1], a.dst[2], a.dst[3]), a.bytes);
}

// src0 -> LDS0 -> register tile -> LDS1 -> dst0. The global<->LDS legs take the
// chain's DMA engine when it has one, the register legs always ds_*; the round
// trip only reproduces the source if both agree on the LDS byte layout.
template<class M>
__global__ void kLdsRegs(MoveArgs a) {
  extern __shared__ __align__(256) uint8_t lds[];
  const dm::Team team = dm::Team::block();
  M mover = ChainInit<M>::make(nullptr, 0);
  uint8_t* l0 = lds + a.ldsOff;
  uint8_t* l1 = lds + a.ldsStride + a.ldsOff;
  mover.move(team, dm::global(a.src[0]), dm::lds(l0), a.bytes);
  __syncthreads();
  dm::RegTile<kDmLdsPacks> tile;
  mover.move(team, dm::lds(l0), tile, a.bytes);
  mover.move(team, tile, dm::lds(l1), a.bytes);
  __syncthreads();
  mover.move(team, dm::lds(l1), dm::global(a.dst[0]), a.bytes);
}

// ---------------------------------------------------------------------------
//  Host side
// ---------------------------------------------------------------------------

struct DmCase {
  size_t      bytes;
  int         srcOff;   // applied to every source
  int         dstOff;   // applied to every destination (and LDS buffers)
  int         nSrcs;
  int         nDsts;
  std::string name;
};

class DataMoverBase : public DeviceTestBase {
protected:
  int waveSize_ = 32;

  std::vector<DeviceBuffer<uint8_t>> srcBufs_;
  std::vector<DeviceBuffer<uint8_t>> dstBufs_;
  std::vector<std::vector<uint8_t>>  srcHost_;

  void SetUp() override {
    DeviceTestBase::SetUp();
    hipDeviceProp_t p{};
    ASSERT_EQ(hipGetDeviceProperties(&p, 0), hipSuccess);
    waveSize_ = p.warpSize > 0 ? p.warpSize : 32;
  }

  static std::vector<uint8_t> makePattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      v[i] = static_cast<uint8_t>(s >> 24);
    }
    return v;
  }

  static uint32_t roundUp(size_t n, size_t a) { return static_cast<uint32_t>((n + a - 1) / a * a); }

  MoveArgs prepare(const DmCase& c, int nSrcs, int nDsts) {
    srcBufs_.clear();
    dstBufs_.clear();
    srcHost_.clear();
    MoveArgs a{};
    a.nSrcs = nSrcs;
    a.nDsts = nDsts;
    a.bytes = c.bytes;
    for (int s = 0; s < nSrcs; ++s) {
      srcHost_.push_back(makePattern(c.srcOff + c.bytes, static_cast<uint32_t>(c.bytes * 7 + s * 131 + c.srcOff)));
      srcBufs_.emplace_back(srcHost_.back().size());
      srcBufs_.back().copyFrom(srcHost_.back());
      a.src[s] = srcBufs_.back().ptr + c.srcOff;
    }
    const std::vector<uint8_t> init(2 * kDmGuard + c.dstOff + c.bytes, kDmSentinel);
    for (int d = 0; d < nDsts; ++d) {
      dstBufs_.emplace_back(init.size());
      dstBufs_.back().copyFrom(init);
      a.dst[d] = dstBufs_.back().ptr + kDmGuard + c.dstOff;
    }
    return a;
  }

  // Destination d must hold source srcOf[d] and nothing outside it.
  void verify(const DmCase& c, const std::vector<int>& srcOf, const std::string& tag) {
    for (size_t d = 0; d < srcOf.size(); ++d) {
      const auto out = dstBufs_[d].copyTo();
      const auto& src = srcHost_[srcOf[d]];
      const size_t start = kDmGuard + c.dstOff;
      for (size_t i = 0; i < out.size(); ++i) {
        const int want = (i >= start && i < start + c.bytes) ? src[c.srcOff + i - start] : kDmSentinel;
        ASSERT_EQ(static_cast<int>(out[i]), want)
            << tag << " " << c.name << ": dst " << d << " byte " << static_cast<long>(i - start)
            << (want == kDmSentinel && (i < start || i >= start + c.bytes) ? " (guard clobbered)" : "");
      }
    }
  }

  static std::vector<int> fanOutMap(int nSrcs, int nDsts) {
    std::vector<int> m(nDsts);
    for (int d = 0; d < nDsts; ++d) m[d] = d / (nDsts / nSrcs);
    return m;
  }

  template<class M>
  void runGlobalToGlobal(const DmCase& c, uint32_t stagingBytes, uint32_t firstWave = 0, uint32_t nWaves = 0) {
    MoveArgs a = prepare(c, c.nSrcs, c.nDsts);
    a.ldsBytes = stagingBytes;
    if (nWaves == 0) nWaves = kDmBlock / waveSize_ - firstWave;
    kGlobalToGlobal<M><<<1, kDmBlock, stagingBytes>>>(a, firstWave, nWaves);
    syncAndCheck();
    verify(c, fanOutMap(c.nSrcs, c.nDsts), "G->G");
  }

  void ldsLayout(const DmCase& c, MoveArgs& a) {
    a.ldsOff    = static_cast<uint32_t>(c.dstOff);
    a.ldsStride = roundUp(c.dstOff + c.bytes, 256);
    a.ldsBytes  = 2 * a.ldsStride;
  }

  template<class M>
  void runViaLds(const DmCase& c) {
    MoveArgs a = prepare(c, 2, 3);
    ldsLayout(c, a);
    kViaLds<M><<<1, kDmBlock, a.ldsBytes>>>(a);
    syncAndCheck();
    verify(c, {0, 0, 1}, "G->L->G");
  }

  template<class M>
  void runViaRegs(const DmCase& c) {
    MoveArgs a = prepare(c, 2, 4);
    kViaRegs<M><<<1, kDmBlock>>>(a);
    syncAndCheck();
    verify(c, {0, 0, 1, 1}, "G->R->R->G");
  }

  template<class M>
  void runLdsRegs(const DmCase& c) {
    MoveArgs a = prepare(c, 1, 1);
    ldsLayout(c, a);
    kLdsRegs<M><<<1, kDmBlock, a.ldsBytes>>>(a);
    syncAndCheck();
    verify(c, {0}, "G->L->R->L->G");
  }
};

// ---------------------------------------------------------------------------
//  Gamuts
// ---------------------------------------------------------------------------

inline const std::vector<DmCase>& globalGamut() {
  static const std::vector<DmCase> cases = {
    {1,           0, 0,  1, 1, "one_byte"},
    {15,          0, 0,  1, 1, "sub_pack"},
    {16,          0, 0,  1, 1, "one_pack"},
    {17,          0, 0,  1, 1, "pack_plus_one"},
    {255,         5, 5,  1, 1, "shared_misalign_small"},
    {4099,        5, 5,  1, 1, "shared_misalign"},
    {4099,        1, 3,  1, 1, "width2"},
    {4099,        8, 0,  1, 1, "width8"},
    {65613,       4, 12, 1, 1, "width8_offset"},
    {4099,        0, 7,  1, 1, "bytewise"},
    {65536,       0, 0,  1, 3, "broadcast3"},
    {65613,       3, 3,  2, 2, "pairs2"},
    {12345,       0, 0,  2, 4, "batched_broadcast"},
    {1u << 20,    0, 0,  1, 1, "one_mib"},
    {(1u << 20) + 77, 64, 64, 2, 2, "one_mib_pairs"},
  };
  return cases;
}

inline const std::vector<DmCase>& ldsGamut() {
  static const std::vector<DmCase> cases = {
    {1,     0,  0,  2, 3, "one_byte"},
    {17,    0,  0,  2, 3, "pack_plus_one"},
    {256,   0,  0,  2, 3, "one_row"},
    {4099,  5,  5,  2, 3, "shared_misalign"},
    {4099,  5,  0,  2, 3, "lds_aligned_global_not"},
    {16384, 0,  0,  2, 3, "sixteen_kib"},
    {16484, 64, 64, 2, 3, "sixteen_kib_ragged"},
  };
  return cases;
}

// Every size fits a RegTile<kDmPacks> filled by a 256-thread block.
inline const std::vector<DmCase>& regGamut() {
  static const std::vector<DmCase> cases = {
    {1,     0, 0, 2, 4, "one_byte"},
    {16,    0, 0, 2, 4, "one_pack"},
    {17,    0, 0, 2, 4, "pack_plus_one"},
    {4096,  0, 0, 2, 4, "one_pack_per_thread"},
    {4099,  0, 0, 2, 4, "ragged_pack"},
    {16384, 0, 0, 2, 4, "full_tile"},
    {4099,  3, 5, 2, 4, "misaligned_memory"},
  };
  return cases;
}

inline std::string dmCaseName(const ::testing::TestParamInfo<DmCase>& info) { return info.param.name; }

// ---------------------------------------------------------------------------
//  Chains with a vector fallback: every space pair, every target.
// ---------------------------------------------------------------------------

constexpr uint32_t kStagingBytes = 16384;

template<class M>
class DataMoverChain : public DataMoverBase, public ::testing::WithParamInterface<DmCase> {};

#define DM_CHAIN_SUITES(NAME, CHAIN)                                                                        \
  class NAME##GlobalTest : public DataMoverChain<CHAIN> {};                                                 \
  TEST_P(NAME##GlobalTest, NoStaging) { runGlobalToGlobal<CHAIN>(GetParam(), 0); }                          \
  TEST_P(NAME##GlobalTest, Staged) { runGlobalToGlobal<CHAIN>(GetParam(), kStagingBytes); }                 \
  INSTANTIATE_TEST_SUITE_P(Gamut, NAME##GlobalTest, ::testing::ValuesIn(globalGamut()), dmCaseName);        \
  class NAME##LdsTest : public DataMoverChain<CHAIN> {};                                                    \
  TEST_P(NAME##LdsTest, RoundTrip) { runViaLds<CHAIN>(GetParam()); }                                        \
  TEST_P(NAME##LdsTest, ThroughRegisters) { runLdsRegs<CHAIN>(GetParam()); }                                \
  INSTANTIATE_TEST_SUITE_P(Gamut, NAME##LdsTest, ::testing::ValuesIn(ldsGamut()), dmCaseName);              \
  class NAME##RegTest : public DataMoverChain<CHAIN> {};                                                    \
  TEST_P(NAME##RegTest, RoundTrip) { runViaRegs<CHAIN>(GetParam()); }                                       \
  INSTANTIATE_TEST_SUITE_P(Gamut, NAME##RegTest, ::testing::ValuesIn(regGamut()), dmCaseName)

DM_CHAIN_SUITES(DmVector, VecChain);
DM_CHAIN_SUITES(DmVectorNt, NtVecChain);
DM_CHAIN_SUITES(DmTdmChain, TdmChain);
DM_CHAIN_SUITES(DmAsyncChain, AsyncChain);
DM_CHAIN_SUITES(DmDefault, dm::DefaultMover);

// ---------------------------------------------------------------------------
//  DMA-only chains: no fallback, so a pass proves the engine moved the data.
// ---------------------------------------------------------------------------

#define DM_DMA_ONLY_SUITE(NAME, CHAIN, SUPPORTED)                                                           \
  class NAME##Test : public DataMoverChain<CHAIN> {                                                         \
  protected:                                                                                                \
    void SetUp() override {                                                                                 \
      DataMoverChain<CHAIN>::SetUp();                                                                       \
      if (!(SUPPORTED)) GTEST_SKIP() << #CHAIN " is not available on this device";                          \
    }                                                                                                       \
  };                                                                                                        \
  TEST_P(NAME##Test, Staged) { runGlobalToGlobal<CHAIN>(GetParam(), kStagingBytes); }                       \
  TEST_P(NAME##Test, SubTeam) { runGlobalToGlobal<CHAIN>(GetParam(), kStagingBytes / 2, 1, 2); }            \
  INSTANTIATE_TEST_SUITE_P(Gamut, NAME##Test, ::testing::ValuesIn(globalGamut()), dmCaseName);              \
  class NAME##LdsTest : public NAME##Test {};                                                               \
  TEST_P(NAME##LdsTest, RoundTrip) { runViaLds<CHAIN>(GetParam()); }                                        \
  INSTANTIATE_TEST_SUITE_P(Gamut, NAME##LdsTest, ::testing::ValuesIn(ldsGamut()), dmCaseName)

DM_DMA_ONLY_SUITE(DmTdmOnly, TdmOnly, tdm::IsTdmCopySupported(0));
DM_DMA_ONLY_SUITE(DmAsyncOnly, AsyncOnly, async::IsTdmCopySupported(0));

// A team of waves 1-2 of the block, with the other waves idle.
class DmSubTeamTest : public DataMoverBase {};

TEST_F(DmSubTeamTest, VectorWaves1to2) {
  runGlobalToGlobal<VecChain>(DmCase{70001, 3, 3, 2, 4, "sub_team"}, 0, 1, 2);
}

}  // namespace RcclUnitTesting
