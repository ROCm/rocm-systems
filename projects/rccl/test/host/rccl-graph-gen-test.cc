/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/rccl_graph_gen.cc.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

#include "nccl.h"
#include "debug.h"
#include "rccl_graph_gen.h"

#include "ScopedHook.h"
#include "fakes/libc_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

// The seam reaches greedyRingGen's out-of-memory arm; the .cc is included
// directly so the file-static constructions are reachable by name.
#include "fakes/libc_seam.h"
#include RCCL_GRAPH_GEN_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

constexpr int kCanary = -12345;
constexpr int kCanaryCount = 4;

// A ring is Hamiltonian over nNodes iff it visits every node exactly once.
::testing::AssertionResult IsHamiltonianRing(const int* ring, int nNodes) {
  std::vector<int> seen(ring, ring + nNodes);
  std::sort(seen.begin(), seen.end());
  for (int i = 0; i < nNodes; ++i) {
    if (seen[i] != i) {
      ::testing::AssertionResult fail = ::testing::AssertionFailure() << "ring [";
      for (int j = 0; j < nNodes; ++j) fail << (j ? ", " : "") << ring[j];
      return fail << "] is not a permutation of [0, " << nNodes << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

// A channel-major ring buffer of nChannels * nNodes ints, fenced by canaries at
// both ends so a write outside the caller's buffer is caught either way.
class RingBuffer {
 public:
  RingBuffer(int nChannels, int nNodes)
      : nChannels_(nChannels), nNodes_(nNodes),
        storage_(static_cast<size_t>(nChannels) * nNodes + 2 * kCanaryCount, kCanary) {}

  int* data() { return storage_.data() + kCanaryCount; }
  int* channel(int c) { return data() + static_cast<size_t>(c) * nNodes_; }
  const int* channel(int c) const {
    return storage_.data() + kCanaryCount + static_cast<size_t>(c) * nNodes_;
  }

  ::testing::AssertionResult CanaryIntact() const {
    for (size_t i = 0; i < storage_.size(); ++i) {
      const bool fenced = i < static_cast<size_t>(kCanaryCount) ||
                          i >= storage_.size() - kCanaryCount;
      if (fenced && storage_[i] != kCanary) {
        return ::testing::AssertionFailure()
               << "wrote outside the ring buffer: slot " << static_cast<long>(i) - kCanaryCount
               << " is " << storage_[i];
      }
    }
    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult AllChannelsAreHamiltonianRings() const {
    for (int c = 0; c < nChannels_; ++c) {
      ::testing::AssertionResult ring = IsHamiltonianRing(channel(c), nNodes_);
      if (!ring) return ::testing::AssertionFailure() << "channel " << c << ": " << ring.message();
    }
    return CanaryIntact();
  }

  std::vector<int> RingAt(int c) const { return std::vector<int>(channel(c), channel(c) + nNodes_); }

  size_t DistinctRingCount(int count) const {
    std::set<std::vector<int>> rings;
    for (int c = 0; c < count; ++c) rings.insert(RingAt(c));
    return rings.size();
  }

 private:
  int nChannels_;
  int nNodes_;
  std::vector<int> storage_;
};

using Edge = std::pair<int, int>;

// Directed edges, or undirected when `directed` is false (stored low node first).
std::set<Edge> RingEdges(const int* ring, int nNodes, bool directed) {
  std::set<Edge> edges;
  for (int i = 0; i < nNodes; ++i) {
    int u = ring[i], v = ring[(i + 1) % nNodes];
    if (!directed && u > v) std::swap(u, v);
    edges.insert({u, v});
  }
  return edges;
}

::testing::AssertionResult PairwiseEdgeDisjoint(const RingBuffer& rings, int count, int nNodes, bool directed) {
  for (int a = 0; a < count; ++a) {
    const std::set<Edge> ea = RingEdges(rings.channel(a), nNodes, directed);
    for (int b = a + 1; b < count; ++b) {
      for (const Edge& e : RingEdges(rings.channel(b), nNodes, directed)) {
        if (ea.count(e)) {
          return ::testing::AssertionFailure()
                 << "channels " << a << " and " << b << " share edge " << e.first << "->" << e.second;
        }
      }
    }
  }
  return ::testing::AssertionSuccess();
}

struct CutLoad {
  std::vector<int> exits;
  std::vector<int> entries;
};

// Range-checks every cut before indexing with it.
::testing::AssertionResult TallyCuts(const int* rings, int nChannels, int nNodes, const std::vector<int>& cuts,
                                     CutLoad* load) {
  load->exits.assign(nNodes, 0);
  load->entries.assign(nNodes, 0);
  for (int c = 0; c < nChannels; ++c) {
    if (cuts[c] < 0 || cuts[c] >= nNodes) {
      return ::testing::AssertionFailure() << "channel " << c << " cut index " << cuts[c] << " is out of range";
    }
    const int* ring = rings + static_cast<size_t>(c) * nNodes;
    load->exits[ring[cuts[c]]]++;
    load->entries[ring[(cuts[c] + 1) % nNodes]]++;
  }
  return ::testing::AssertionSuccess();
}

int MaxOf(const std::vector<int>& v) { return *std::max_element(v.begin(), v.end()); }
int SpreadOf(const std::vector<int>& v) { return MaxOf(v) - *std::min_element(v.begin(), v.end()); }

class GraphGenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetLibcFakes();
  }
  void TearDown() override {
    ResetLibcFakes();
    ResetNcclFakes();
  }
};

// ---------------------------------------------------------------------------
// permute_array_inplace
// ---------------------------------------------------------------------------

using GraphGenPermuteTest = GraphGenTest;

// One of each cycle shape: a fixed point, a two-cycle, and a three-cycle.
const std::vector<int> kMixedCyclePermutation = {0, 2, 1, 5, 3, 4};

TEST_F(GraphGenPermuteTest, PermuteArrayInplace_AnyPermutation_GathersEachElementFromItsSourceIndex) {
  std::vector<int> input = {10, 11, 12, 13, 14, 15};
  std::vector<int> permutation = kMixedCyclePermutation;
  const std::vector<int> original = input;

  permute_array_inplace(input.data(), static_cast<int>(input.size()), permutation.data());

  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(input[i], original[kMixedCyclePermutation[i]]) << "at index " << i;
  }
}

TEST_F(GraphGenPermuteTest, PermuteArrayInplace_AnyPermutation_RestoresThePermutationArray) {
  std::vector<int> input = {10, 11, 12, 13, 14, 15};
  std::vector<int> permutation = kMixedCyclePermutation;

  permute_array_inplace(input.data(), static_cast<int>(input.size()), permutation.data());

  EXPECT_EQ(permutation, kMixedCyclePermutation);
}

TEST_F(GraphGenPermuteTest, PermuteArrayInplace_IdentityPermutation_LeavesTheInputUnchanged) {
  std::vector<int> input = {10, 11, 12, 13};
  const std::vector<int> original = input;
  std::vector<int> permutation = {0, 1, 2, 3};

  permute_array_inplace(input.data(), static_cast<int>(input.size()), permutation.data());

  EXPECT_EQ(input, original);
}

TEST_F(GraphGenPermuteTest, PermuteArrayInplace_ZeroLength_TouchesNeitherArray) {
  std::vector<int> input = {kCanary, kCanary};
  std::vector<int> permutation = {kCanary, kCanary};

  permute_array_inplace(input.data(), 0, permutation.data());

  EXPECT_EQ(input, std::vector<int>({kCanary, kCanary}));
  EXPECT_EQ(permutation, std::vector<int>({kCanary, kCanary}));
}

// ---------------------------------------------------------------------------
// isPrime
// ---------------------------------------------------------------------------

using GraphGenIsPrimeTest = GraphGenTest;

TEST_F(GraphGenIsPrimeTest, IsPrime_BelowTwo_ReturnsFalse) {
  for (int n : {INT_MIN, -7, 0, 1}) EXPECT_FALSE(isPrime(n)) << "n = " << n;
}

TEST_F(GraphGenIsPrimeTest, IsPrime_TwoOrThree_ReturnsTrue) {
  EXPECT_TRUE(isPrime(2));
  EXPECT_TRUE(isPrime(3));
}

TEST_F(GraphGenIsPrimeTest, IsPrime_MultipleOfTwoOrThree_ReturnsFalse) {
  for (int n : {4, 6, 8, 9, 15, 21, 100}) EXPECT_FALSE(isPrime(n)) << "n = " << n;
}

// The 6k+/-1 walk tests n % i and n % (i + 2); 25 is caught by the first, 49 by the second.
TEST_F(GraphGenIsPrimeTest, IsPrime_CompositeCoprimeToSixThroughEitherStep_ReturnsFalse) {
  for (int n : {25, 35, 49, 77, 121, 143}) EXPECT_FALSE(isPrime(n)) << "n = " << n;
}

TEST_F(GraphGenIsPrimeTest, IsPrime_PrimeAboveThree_ReturnsTrue) {
  for (int n : {5, 7, 11, 13, 31, 97, 7919}) EXPECT_TRUE(isPrime(n)) << "n = " << n;
}

// ---------------------------------------------------------------------------
// generateWalecki
// ---------------------------------------------------------------------------

// An even node count rotates over nNodes - 1 nodes plus one fixed pivot.
int WaleckiRotationSize(int nNodes) { return (nNodes % 2) ? nNodes : nNodes - 1; }

class GraphGenWaleckiTest : public GraphGenTest, public ::testing::WithParamInterface<int> {};

TEST_P(GraphGenWaleckiTest, GenerateWalecki_AnyChannel_WritesAHamiltonianRing) {
  const int nNodes = GetParam();
  RingBuffer rings(1, nNodes);

  for (int channel = 0; channel < WaleckiRotationSize(nNodes); ++channel) {
    generateWalecki(nNodes, channel, rings.data());
    EXPECT_TRUE(IsHamiltonianRing(rings.channel(0), nNodes)) << "channel " << channel;
    EXPECT_TRUE(rings.CanaryIntact()) << "channel " << channel;
  }
}

// A construction that ignored the channel would still produce Hamiltonian rings.
TEST_P(GraphGenWaleckiTest, GenerateWalecki_ChannelsWithinOneRotation_ProduceDistinctRings) {
  const int nNodes = GetParam();
  const int rotation = WaleckiRotationSize(nNodes);
  RingBuffer rings(rotation, nNodes);

  for (int channel = 0; channel < rotation; ++channel) {
    generateWalecki(nNodes, channel, rings.channel(channel));
  }

  EXPECT_EQ(rings.DistinctRingCount(rotation), static_cast<size_t>(rotation));
}

INSTANTIATE_TEST_SUITE_P(NodeCounts, GraphGenWaleckiTest, ::testing::Values(3, 4, 5, 6, 8, 9, 12, 16),
                         [](const ::testing::TestParamInfo<int>& info) {
                           return "N" + std::to_string(info.param);
                         });

// For an even count the first (nNodes - 1) / 2 channels are a Walecki
// decomposition: no two rings share a link. Odd counts rotate every node and do not.
class GraphGenWaleckiEvenTest : public GraphGenTest, public ::testing::WithParamInterface<int> {};

TEST_P(GraphGenWaleckiEvenTest, GenerateWalecki_EvenNodeCount_FirstHalfRotationIsEdgeDisjoint) {
  const int nNodes = GetParam();
  const int count = WaleckiRotationSize(nNodes) / 2;
  RingBuffer rings(count, nNodes);

  for (int channel = 0; channel < count; ++channel) generateWalecki(nNodes, channel, rings.channel(channel));

  EXPECT_TRUE(PairwiseEdgeDisjoint(rings, count, nNodes, /*directed=*/false));
}

INSTANTIATE_TEST_SUITE_P(EvenNodeCounts, GraphGenWaleckiEvenTest, ::testing::Values(6, 8, 10, 12, 16),
                         [](const ::testing::TestParamInfo<int>& info) {
                           return "N" + std::to_string(info.param);
                         });

using GraphGenWaleckiGuardTest = GraphGenTest;

TEST_F(GraphGenWaleckiGuardTest, GenerateWalecki_EvenNodeCount_PinsTheLastSlotToTheHighestNode) {
  constexpr int kNodes = 8;
  RingBuffer rings(1, kNodes);

  for (int channel = 0; channel < kNodes; ++channel) {
    generateWalecki(kNodes, channel, rings.data());
    EXPECT_EQ(rings.channel(0)[kNodes - 1], kNodes - 1) << "channel " << channel;
  }
}

TEST_F(GraphGenWaleckiGuardTest, GenerateWalecki_NonPositiveNodeCount_LeavesTheOutputUntouched) {
  RingBuffer rings(1, 4);

  for (int nNodes : {0, -1, -8}) generateWalecki(nNodes, 0, rings.data());

  EXPECT_EQ(rings.RingAt(0), std::vector<int>(4, kCanary));
  EXPECT_TRUE(rings.CanaryIntact());
}

TEST_F(GraphGenWaleckiGuardTest, GenerateWalecki_NullOrder_ReturnsWithoutWriting) {
  generateWalecki(8, 0, nullptr);  // Fails by crashing.
}

// ---------------------------------------------------------------------------
// genRingsN_4 / genRingsN_6 / genRingsN_8 -- the lookup-table constructions
// ---------------------------------------------------------------------------

// The three table constructions share one contract, differing only in node
// count and table size.
struct FixedRingTableCase {
  const char* name;
  int (*generate)(int*, int);
  int nNodes;
  int tableSize;
};

class GraphGenFixedRingTableTest : public GraphGenTest,
                                   public ::testing::WithParamInterface<FixedRingTableCase> {};

TEST_P(GraphGenFixedRingTableTest, GenRingsFixed_FewerChannelsThanTableEntries_WritesOneRingPerChannel) {
  const FixedRingTableCase& c = GetParam();
  const int nChannels = c.tableSize - 1;
  RingBuffer rings(nChannels, c.nNodes);

  const int written = c.generate(rings.data(), nChannels);

  EXPECT_EQ(written, nChannels);
  EXPECT_TRUE(rings.AllChannelsAreHamiltonianRings());
}

// Duplicate entries would hand two channels the same ring, halving link diversity.
TEST_P(GraphGenFixedRingTableTest, GenRingsFixed_ExactlyTableEntries_WritesDistinctRings) {
  const FixedRingTableCase& c = GetParam();
  RingBuffer rings(c.tableSize, c.nNodes);

  const int written = c.generate(rings.data(), c.tableSize);

  EXPECT_EQ(written, c.tableSize);
  EXPECT_EQ(rings.DistinctRingCount(c.tableSize), static_cast<size_t>(c.tableSize));
}

TEST_P(GraphGenFixedRingTableTest, GenRingsFixed_MoreChannelsThanTableEntries_RepeatsTheTableFromTheStart) {
  const FixedRingTableCase& c = GetParam();
  const int nChannels = c.tableSize + 2;
  RingBuffer rings(nChannels, c.nNodes);

  const int written = c.generate(rings.data(), nChannels);

  EXPECT_EQ(written, nChannels);
  EXPECT_TRUE(rings.AllChannelsAreHamiltonianRings());
  for (int c2 = c.tableSize; c2 < nChannels; ++c2) {
    EXPECT_EQ(rings.RingAt(c2), rings.RingAt(c2 - c.tableSize)) << "channel " << c2;
  }
}

TEST_P(GraphGenFixedRingTableTest, GenRingsFixed_NonPositiveChannelCount_WritesNothingAndReportsZero) {
  const FixedRingTableCase& c = GetParam();
  RingBuffer rings(1, c.nNodes);

  for (int nChannels : {0, -1}) EXPECT_EQ(c.generate(rings.data(), nChannels), 0);

  EXPECT_EQ(rings.RingAt(0), std::vector<int>(c.nNodes, kCanary));
  EXPECT_TRUE(rings.CanaryIntact());
}

TEST_P(GraphGenFixedRingTableTest, GenRingsFixed_NullBuffer_ReportsZero) {
  EXPECT_EQ(GetParam().generate(nullptr, 1), 0);
}

INSTANTIATE_TEST_SUITE_P(Tables, GraphGenFixedRingTableTest,
                         ::testing::Values(FixedRingTableCase{"N4", genRingsN_4, 4, 6},
                                           FixedRingTableCase{"N6", genRingsN_6, 6, 15},
                                           FixedRingTableCase{"N8", genRingsN_8, 8, 14}),
                         [](const ::testing::TestParamInfo<FixedRingTableCase>& info) {
                           return info.param.name;
                         });

// ---------------------------------------------------------------------------
// genRingsN_prime
// ---------------------------------------------------------------------------

class GraphGenPrimeRingTest : public GraphGenTest, public ::testing::WithParamInterface<int> {};

TEST_P(GraphGenPrimeRingTest, GenRingsNPrime_FewerChannelsThanStrides_WritesOneRingPerChannel) {
  const int p = GetParam();
  const int nChannels = p - 2;
  RingBuffer rings(nChannels, p);

  const int written = genRingsN_prime(p, nChannels, rings.data());

  EXPECT_EQ(written, nChannels);
  EXPECT_TRUE(rings.AllChannelsAreHamiltonianRings());
}

TEST_P(GraphGenPrimeRingTest, GenRingsNPrime_AllAvailableStrides_UseEveryLinkOnce) {
  const int p = GetParam();
  RingBuffer rings(p - 1, p);

  genRingsN_prime(p, p - 1, rings.data());

  EXPECT_TRUE(PairwiseEdgeDisjoint(rings, p - 1, p, /*directed=*/true));
}

TEST_P(GraphGenPrimeRingTest, GenRingsNPrime_MoreChannelsThanStrides_RepeatsTheStridesFromTheStart) {
  const int p = GetParam();
  const int available = p - 1;
  const int nChannels = available + 2;
  RingBuffer rings(nChannels, p);

  const int written = genRingsN_prime(p, nChannels, rings.data());

  EXPECT_EQ(written, nChannels);
  EXPECT_TRUE(rings.AllChannelsAreHamiltonianRings());
  for (int c = available; c < nChannels; ++c) {
    EXPECT_EQ(rings.RingAt(c), rings.RingAt(c - available)) << "channel " << c;
  }
}

INSTANTIATE_TEST_SUITE_P(PrimeNodeCounts, GraphGenPrimeRingTest, ::testing::Values(3, 5, 7, 11, 13),
                         [](const ::testing::TestParamInfo<int>& info) {
                           return "P" + std::to_string(info.param);
                         });

using GraphGenPrimeRingGuardTest = GraphGenTest;

TEST_F(GraphGenPrimeRingGuardTest, GenRingsNPrime_NonPositiveArgs_WritesNothingAndReportsZero) {
  constexpr int kNodes = 5;
  RingBuffer rings(1, kNodes);

  EXPECT_EQ(genRingsN_prime(0, 1, rings.data()), 0);
  EXPECT_EQ(genRingsN_prime(-5, 1, rings.data()), 0);
  EXPECT_EQ(genRingsN_prime(kNodes, 0, rings.data()), 0);
  EXPECT_EQ(genRingsN_prime(kNodes, -1, rings.data()), 0);

  EXPECT_EQ(rings.RingAt(0), std::vector<int>(kNodes, kCanary));
  EXPECT_TRUE(rings.CanaryIntact());
}

TEST_F(GraphGenPrimeRingGuardTest, GenRingsNPrime_NullBuffer_ReportsZero) {
  EXPECT_EQ(genRingsN_prime(5, 1, nullptr), 0);
}

// ---------------------------------------------------------------------------
// greedyRingGen
// ---------------------------------------------------------------------------

using GraphGenGreedyTest = GraphGenTest;

// Composite, not tabulated, and big enough to leave real greedy work to do.
constexpr int kGreedyNodes = 10;

TEST_F(GraphGenGreedyTest, GreedyRingGen_MoreChannelsThanWaleckiProvides_WritesAHamiltonianRingPerChannel) {
  constexpr int kChannels = 3 * kGreedyNodes;
  RingBuffer rings(kChannels, kGreedyNodes);

  ASSERT_EQ(greedyRingGen(kGreedyNodes, kChannels, rings.data()), ncclSuccess);

  EXPECT_TRUE(rings.AllChannelsAreHamiltonianRings());
}

// The edge-usage cost is what spreads load over the links; without it a few
// links carry nearly every ring.
TEST_F(GraphGenGreedyTest, GreedyRingGen_ManyChannels_KeepEveryLinkWithinTwiceItsFairShare) {
  constexpr int kChannels = 3 * kGreedyNodes;
  RingBuffer rings(kChannels, kGreedyNodes);

  ASSERT_EQ(greedyRingGen(kGreedyNodes, kChannels, rings.data()), ncclSuccess);
  ASSERT_TRUE(rings.AllChannelsAreHamiltonianRings());

  std::vector<int> linkUse(kGreedyNodes * kGreedyNodes, 0);
  for (int c = 0; c < kChannels; ++c) {
    const int* ring = rings.channel(c);
    for (int i = 0; i < kGreedyNodes; ++i) linkUse[ring[i] * kGreedyNodes + ring[(i + 1) % kGreedyNodes]]++;
  }
  constexpr int kDirectedLinks = kGreedyNodes * (kGreedyNodes - 1);
  constexpr int kFairShare = (kChannels * kGreedyNodes + kDirectedLinks - 1) / kDirectedLinks;
  EXPECT_LE(MaxOf(linkUse), 2 * kFairShare);
}

TEST_F(GraphGenGreedyTest, GreedyRingGen_FirstHalfOfChannels_ComeFromTheWaleckiConstruction) {
  constexpr int kChannels = kGreedyNodes;
  RingBuffer rings(kChannels, kGreedyNodes);
  RingBuffer expected(1, kGreedyNodes);

  ASSERT_EQ(greedyRingGen(kGreedyNodes, kChannels, rings.data()), ncclSuccess);

  for (int c = 0; c < kGreedyNodes / 2; ++c) {
    generateWalecki(kGreedyNodes, c, expected.data());
    EXPECT_EQ(rings.RingAt(c), expected.RingAt(0)) << "channel " << c;
  }
}

// So no single node carries the start of every ring.
TEST_F(GraphGenGreedyTest, GreedyRingGen_GreedyChannels_StartFromConsecutiveNodes) {
  constexpr int kChannels = 2 * kGreedyNodes;
  RingBuffer rings(kChannels, kGreedyNodes);

  ASSERT_EQ(greedyRingGen(kGreedyNodes, kChannels, rings.data()), ncclSuccess);

  for (int c = kGreedyNodes / 2; c < kChannels; ++c) {
    const int expectedStart = (c - kGreedyNodes / 2 + 1) % kGreedyNodes;
    EXPECT_EQ(rings.channel(c)[0], expectedStart) << "channel " << c;
  }
}

// Order-insensitive, but a double free or a stray free still shows up.
std::vector<void*> FreedSet() {
  std::vector<void*> freed = g_freedPointers;
  std::sort(freed.begin(), freed.end());
  return freed;
}

TEST_F(GraphGenGreedyTest, GreedyRingGen_Success_FreesBothAllocationsExactlyOnce) {
  RingBuffer rings(kGreedyNodes, kGreedyNodes);
  void* edgeUsage = nullptr;
  void* visited = nullptr;
  ScopedHook callocHook(g_calloc, [&](size_t nmemb, size_t size) {
    EXPECT_EQ(nmemb, static_cast<size_t>(kGreedyNodes * kGreedyNodes));
    EXPECT_EQ(size, sizeof(uint32_t));
    edgeUsage = std::calloc(nmemb, size);
    return edgeUsage;
  });
  ScopedHook mallocHook(g_malloc, [&](size_t size) {
    EXPECT_EQ(size, kGreedyNodes * sizeof(uint32_t));
    visited = std::malloc(size);
    return visited;
  });

  ASSERT_EQ(greedyRingGen(kGreedyNodes, kGreedyNodes, rings.data()), ncclSuccess);

  std::vector<void*> expected = {edgeUsage, visited};
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(FreedSet(), expected);
}

TEST_F(GraphGenGreedyTest, GreedyRingGen_EdgeUsageAllocationFails_FreesTheVisitedSetAndReportsInternalError) {
  RingBuffer rings(kGreedyNodes, kGreedyNodes);
  void* visited = nullptr;
  ScopedHook callocHook(g_calloc, [](size_t, size_t) -> void* { return nullptr; });
  ScopedHook mallocHook(g_malloc, [&](size_t size) {
    visited = std::malloc(size);
    return visited;
  });

  EXPECT_EQ(greedyRingGen(kGreedyNodes, kGreedyNodes, rings.data()), ncclInternalError);

  ASSERT_NE(visited, nullptr) << "the visited-set allocation was meant to succeed";
  EXPECT_EQ(FreedSet(), std::vector<void*>{visited});
}

TEST_F(GraphGenGreedyTest, GreedyRingGen_VisitedSetAllocationFails_FreesTheEdgeUsageMatrixAndReportsInternalError) {
  RingBuffer rings(kGreedyNodes, kGreedyNodes);
  void* edgeUsage = nullptr;
  ScopedHook callocHook(g_calloc, [&](size_t nmemb, size_t size) {
    edgeUsage = std::calloc(nmemb, size);
    return edgeUsage;
  });
  ScopedHook mallocHook(g_malloc, [](size_t) -> void* { return nullptr; });

  EXPECT_EQ(greedyRingGen(kGreedyNodes, kGreedyNodes, rings.data()), ncclInternalError);

  ASSERT_NE(edgeUsage, nullptr) << "the edge-usage allocation was meant to succeed";
  EXPECT_EQ(FreedSet(), std::vector<void*>{edgeUsage});
}

TEST_F(GraphGenGreedyTest, GreedyRingGen_AllocationFails_LeavesTheRingBufferUntouched) {
  RingBuffer rings(kGreedyNodes, kGreedyNodes);
  ScopedHook callocHook(g_calloc, [](size_t, size_t) -> void* { return nullptr; });

  ASSERT_EQ(greedyRingGen(kGreedyNodes, kGreedyNodes, rings.data()), ncclInternalError);

  for (int c = 0; c < kGreedyNodes; ++c) {
    EXPECT_EQ(rings.RingAt(c), std::vector<int>(kGreedyNodes, kCanary)) << "channel " << c;
  }
}

// ---------------------------------------------------------------------------
// generateRings -- the entry point, and its dispatch across constructions
// ---------------------------------------------------------------------------

using GraphGenGenerateRingsTest = GraphGenTest;

TEST_F(GraphGenGenerateRingsTest, GenerateRings_NonPositiveNodeCount_ReturnsInvalidArgument) {
  RingBuffer rings(1, 8);

  for (int nNodes : {0, -1, -8}) {
    EXPECT_EQ(generateRings(nNodes, 1, rings.data()), ncclInvalidArgument) << "nNodes = " << nNodes;
  }
}

TEST_F(GraphGenGenerateRingsTest, GenerateRings_ZeroChannels_ReturnsInvalidArgument) {
  RingBuffer rings(1, 8);

  EXPECT_EQ(generateRings(8, 0, rings.data()), ncclInvalidArgument);
}

TEST_F(GraphGenGenerateRingsTest, GenerateRings_NullNodeOrder_ReturnsInvalidArgument) {
  EXPECT_EQ(generateRings(8, 1, nullptr), ncclInvalidArgument);
}

TEST_F(GraphGenGenerateRingsTest, GenerateRings_RejectedArguments_LeaveTheOutputUntouched) {
  RingBuffer rings(1, 8);

  ASSERT_EQ(generateRings(0, 1, rings.data()), ncclInvalidArgument);
  ASSERT_EQ(generateRings(8, 0, rings.data()), ncclInvalidArgument);

  EXPECT_EQ(rings.RingAt(0), std::vector<int>(8, kCanary));
  EXPECT_TRUE(rings.CanaryIntact());
}

TEST_F(GraphGenGenerateRingsTest, GenerateRings_FewerThanThreeNodes_GivesEveryChannelTheIdentityRing) {
  constexpr int kChannels = 4;

  for (int nNodes : {1, 2}) {
    RingBuffer rings(kChannels, nNodes);

    ASSERT_EQ(generateRings(nNodes, kChannels, rings.data()), ncclSuccess) << "nNodes = " << nNodes;

    std::vector<int> identity(nNodes);
    for (int n = 0; n < nNodes; ++n) identity[n] = n;
    for (int c = 0; c < kChannels; ++c) {
      EXPECT_EQ(rings.RingAt(c), identity) << "nNodes = " << nNodes << ", channel " << c;
    }
    EXPECT_TRUE(rings.CanaryIntact()) << "nNodes = " << nNodes;
  }
}

TEST_F(GraphGenGenerateRingsTest, GenerateRings_GreedyConstructionFailsToAllocate_PropagatesInternalError) {
  constexpr int kNodes = 10;
  constexpr uint32_t kChannels = 18;
  RingBuffer rings(static_cast<int>(kChannels), kNodes);
  ScopedHook callocHook(g_calloc, [](size_t, size_t) -> void* { return nullptr; });

  EXPECT_EQ(generateRings(kNodes, kChannels, rings.data()), ncclInternalError);
}

// Each row is checked against the construction it must select. Allocation is
// the only signal that tells WaleckiOnly from greedy, whose prefix is identical.
void ExpectTable4(int, uint32_t nChannels, int* out) { genRingsN_4(out, static_cast<int>(nChannels)); }
void ExpectTable6(int, uint32_t nChannels, int* out) { genRingsN_6(out, static_cast<int>(nChannels)); }
void ExpectTable8(int, uint32_t nChannels, int* out) { genRingsN_8(out, static_cast<int>(nChannels)); }
void ExpectPrime(int nNodes, uint32_t nChannels, int* out) {
  genRingsN_prime(nNodes, static_cast<int>(nChannels), out);
}
void ExpectWalecki(int nNodes, uint32_t nChannels, int* out) {
  for (uint32_t c = 0; c < nChannels; ++c) generateWalecki(nNodes, static_cast<int>(c), out + c * nNodes);
}
void ExpectGreedy(int nNodes, uint32_t nChannels, int* out) { greedyRingGen(nNodes, nChannels, out); }

struct DispatchCase {
  const char* name;
  int nNodes;
  uint32_t nChannels;
  void (*expected)(int, uint32_t, int*);
  bool allocates;
};

class GraphGenDispatchTest : public GraphGenTest, public ::testing::WithParamInterface<DispatchCase> {};

TEST_P(GraphGenDispatchTest, GenerateRings_AnyNodeCount_SelectsTheMatchingConstruction) {
  const DispatchCase& c = GetParam();
  const int nChannels = static_cast<int>(c.nChannels);
  RingBuffer expected(nChannels, c.nNodes);
  c.expected(c.nNodes, c.nChannels, expected.data());
  RingBuffer rings(nChannels, c.nNodes);
  ScopedHook callocHook(g_calloc, [](size_t nmemb, size_t size) { return std::calloc(nmemb, size); });

  ASSERT_EQ(generateRings(c.nNodes, c.nChannels, rings.data()), ncclSuccess);

  EXPECT_TRUE(rings.AllChannelsAreHamiltonianRings());
  for (int ch = 0; ch < nChannels; ++ch) EXPECT_EQ(rings.RingAt(ch), expected.RingAt(ch)) << "channel " << ch;
  EXPECT_EQ(callocHook.calls > 0, c.allocates);
}

INSTANTIATE_TEST_SUITE_P(
    Constructions, GraphGenDispatchTest,
    ::testing::Values(DispatchCase{"FourNodeTable", 4, 9, ExpectTable4, false},
                      DispatchCase{"SixNodeTable", 6, 20, ExpectTable6, false},
                      DispatchCase{"EightNodeTable", 8, 17, ExpectTable8, false},
                      DispatchCase{"PrimeStrides", 11, 16, ExpectPrime, false},
                      DispatchCase{"WaleckiOnly", 10, 5, ExpectWalecki, false},
                      DispatchCase{"WaleckiThenGreedy", 10, 18, ExpectGreedy, true}),
    [](const ::testing::TestParamInfo<DispatchCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// findRingCutIndices
// ---------------------------------------------------------------------------

using GraphGenCutIndicesTest = GraphGenTest;

std::vector<int> RepeatRing(const std::vector<int>& ring, int nChannels) {
  std::vector<int> flattened;
  flattened.reserve(ring.size() * nChannels);
  for (int c = 0; c < nChannels; ++c) flattened.insert(flattened.end(), ring.begin(), ring.end());
  return flattened;
}

// Cut indices live in a one-row RingBuffer so the output is fenced too.
TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_AnyRingSet_ChoosesACutIndexInsideEveryRing) {
  constexpr int kNodes = 7;
  constexpr int kChannels = 5;
  RingBuffer rings(kChannels, kNodes);
  ASSERT_EQ(generateRings(kNodes, kChannels, rings.data()), ncclSuccess);
  RingBuffer cuts(1, kChannels);

  findRingCutIndices(kChannels, kNodes, rings.data(), cuts.data());

  CutLoad load;
  EXPECT_TRUE(TallyCuts(rings.data(), kChannels, kNodes, cuts.RingAt(0), &load));
  EXPECT_TRUE(cuts.CanaryIntact());
}

// Identical rings offer every channel the same choice, so any spread in the
// chosen cuts comes from the running exit/entry counts alone.
TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_IdenticalRings_SpreadExitsAndEntriesEvenlyOverTheNodes) {
  constexpr int kNodes = 6;

  for (int passes : {1, 2, 3}) {
    const int nChannels = kNodes * passes;
    const std::vector<int> flattened = RepeatRing({3, 1, 4, 0, 5, 2}, nChannels);
    RingBuffer cuts(1, nChannels);

    findRingCutIndices(nChannels, kNodes, flattened.data(), cuts.data());

    CutLoad load;
    ASSERT_TRUE(TallyCuts(flattened.data(), nChannels, kNodes, cuts.RingAt(0), &load)) << "passes " << passes;
    EXPECT_EQ(load.exits, std::vector<int>(kNodes, passes)) << "passes " << passes;
    EXPECT_EQ(load.entries, std::vector<int>(kNodes, passes)) << "passes " << passes;
    EXPECT_TRUE(cuts.CanaryIntact()) << "passes " << passes;
  }
}

// Identical rings couple the two: a cut position fixes both the exit and the
// entry. A varied ring set -- the shape the production caller passes -- separates them.
TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_VariedRings_BalanceExitsAndEntriesIndependently) {
  constexpr int kNodes = 4;
  constexpr int kChannels = 8;
  RingBuffer rings(kChannels, kNodes);
  ASSERT_EQ(generateRings(kNodes, kChannels, rings.data()), ncclSuccess);
  RingBuffer cuts(1, kChannels);

  findRingCutIndices(kChannels, kNodes, rings.data(), cuts.data());

  CutLoad load;
  ASSERT_TRUE(TallyCuts(rings.data(), kChannels, kNodes, cuts.RingAt(0), &load));
  EXPECT_LE(SpreadOf(load.exits), 1) << "exits are not balanced across the nodes";
  EXPECT_LE(SpreadOf(load.entries), 1) << "entries are not balanced across the nodes";
  EXPECT_TRUE(cuts.CanaryIntact());
}

// The penalty is quadratic so two nodes at 2 beat one node at 3; a linear
// penalty scores those equal and lets node 3 take three exits here.
TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_CompetingLoads_NoNodeExceedsTheFairShare) {
  constexpr int kNodes = 4;
  constexpr int kChannels = 5;
  const std::vector<int> flattened = {0, 3, 2, 1,  3, 0, 1, 2,  1, 3, 2, 0,  3, 2, 0, 1,  3, 1, 0, 2};
  RingBuffer cuts(1, kChannels);

  findRingCutIndices(kChannels, kNodes, flattened.data(), cuts.data());

  CutLoad load;
  ASSERT_TRUE(TallyCuts(flattened.data(), kChannels, kNodes, cuts.RingAt(0), &load));
  constexpr int kFairShare = (kChannels + kNodes - 1) / kNodes;
  EXPECT_LE(MaxOf(load.exits), kFairShare);
  EXPECT_LE(MaxOf(load.entries), kFairShare);
  EXPECT_TRUE(cuts.CanaryIntact());
}

// Reaching the last position depends on the entry wrapping to the ring head.
TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_OneIdenticalRingPerNode_CutsEveryPositionOnce) {
  constexpr int kNodes = 4;
  const std::vector<int> flattened = RepeatRing({2, 0, 3, 1}, kNodes);
  RingBuffer cuts(1, kNodes);

  findRingCutIndices(kNodes, kNodes, flattened.data(), cuts.data());

  std::vector<int> cutIndices = cuts.RingAt(0);
  std::sort(cutIndices.begin(), cutIndices.end());
  EXPECT_EQ(cutIndices, std::vector<int>({0, 1, 2, 3}));
  EXPECT_TRUE(cuts.CanaryIntact());
}

TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_SingleNodeRings_CutAtTheOnlyPosition) {
  constexpr int kChannels = 3;
  const std::vector<int> flattened = {0, 0, 0};
  RingBuffer cuts(1, kChannels);

  findRingCutIndices(kChannels, 1, flattened.data(), cuts.data());

  EXPECT_EQ(cuts.RingAt(0), std::vector<int>(kChannels, 0));
  EXPECT_TRUE(cuts.CanaryIntact());
}

TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_NonPositiveCounts_LeaveTheOutputUntouched) {
  constexpr int kNodes = 4;
  constexpr int kChannels = 2;
  const std::vector<int> flattened = RepeatRing({0, 1, 2, 3}, kChannels);
  RingBuffer cuts(1, kChannels);

  findRingCutIndices(0, kNodes, flattened.data(), cuts.data());
  findRingCutIndices(-1, kNodes, flattened.data(), cuts.data());
  findRingCutIndices(kChannels, 0, flattened.data(), cuts.data());
  findRingCutIndices(kChannels, -1, flattened.data(), cuts.data());

  EXPECT_EQ(cuts.RingAt(0), std::vector<int>(kChannels, kCanary));
  EXPECT_TRUE(cuts.CanaryIntact());
}

TEST_F(GraphGenCutIndicesTest, FindRingCutIndices_NullArguments_LeaveTheOutputUntouched) {
  constexpr int kNodes = 4;
  constexpr int kChannels = 2;
  const std::vector<int> flattened = RepeatRing({0, 1, 2, 3}, kChannels);
  RingBuffer cuts(1, kChannels);

  findRingCutIndices(kChannels, kNodes, nullptr, cuts.data());
  findRingCutIndices(kChannels, kNodes, flattened.data(), nullptr);

  EXPECT_EQ(cuts.RingAt(0), std::vector<int>(kChannels, kCanary));
  EXPECT_TRUE(cuts.CanaryIntact());
}

}  // namespace
