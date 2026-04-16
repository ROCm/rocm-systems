//===-- InstrumentationReplayGTest.cpp - Multi-variant patch test *- C++ *-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// End-to-end test for `KernelPatcher::getOrPatchVariants` (Phase 2c of the
/// Instrumentation Replay 5a plan).
///
/// Each variant must:
///   - return a cached `PatchedKernel*` with its own `SiteMap`;
///   - avoid re-instrumenting any `OriginalPC` covered by a prior variant;
///   - together cover a superset of what a single-variant patch produces.
///
/// Uses the `mega_gather` gfx950 fixture: a real ~110 KB ELF with 2080 memory
/// sites — large enough to exercise the variant loop on realistic input
/// without a GPU.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/DispatchInterceptor.h"
#include "aegisbit/KernelPatcher.h"
#include "aegisbit/Types.h"

#include <fstream>
#include <gtest/gtest.h>
#include <unordered_set>

using namespace aegisbit;
using namespace llvm;

#ifndef MEGA_GATHER_FIXTURE_PATH
#error "MEGA_GATHER_FIXTURE_PATH must be defined at compile time"
#endif

namespace {

std::vector<uint8_t> loadFixture(const char *Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F)
    return {};
  auto Size = F.tellg();
  F.seekg(0);
  std::vector<uint8_t> Bytes(Size);
  F.read(reinterpret_cast<char *>(Bytes.data()), Size);
  return Bytes;
}

struct Fixture : ::testing::Test {
  void SetUp() override {
    FixtureBytes = loadFixture(MEGA_GATHER_FIXTURE_PATH);
    ASSERT_FALSE(FixtureBytes.empty())
        << "Cannot load fixture: " << MEGA_GATHER_FIXTURE_PATH;

    auto Handler = CodeObjectHandler::loadFromBytes(FixtureBytes);
    ASSERT_TRUE(!!Handler) << toString(Handler.takeError());
    auto Names = Handler->getKernelNames();
    ASSERT_FALSE(Names.empty());
    KName = Names[0];
    const KernelInfo *KI = Handler->getKernel(KName);
    ASSERT_NE(KI, nullptr);

    CodeObj.CodeObjectId = 0xABCD0001;
    CodeObj.Bytes = FixtureBytes;

    Symbol.KernelId = 0xFEED0001;
    Symbol.CodeObjectId = CodeObj.CodeObjectId;
    Symbol.KernelName = KName;
    Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
    Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
    Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
    Symbol.SGPRCount = KI->Descriptor.SGPRCount;
    Symbol.VGPRCount = KI->Descriptor.VGPRCount;
  }

  // Two distinct trace buffers so the trampoline's baked immediates don't
  // alias across variants. Values are just fake addresses for testing.
  static TraceConfig makeTrace(uint64_t Base) {
    TraceConfig T;
    T.BufferAddr   = Base;
    T.CounterAddr  = Base + 0x1000;
    T.BufferSize   = 32768;
    T.Strategy     = PayloadStrategy::OnGpuReduce;
    T.SupportsGPUAtomics = true;
    return T;
  }

  std::vector<uint8_t> FixtureBytes;
  std::string KName;
  CapturedCodeObject CodeObj;
  CapturedKernelSymbol Symbol;
};

// Collect OriginalPCs of a variant's SiteMap into an unordered_set.
std::unordered_set<uint64_t> pcsOf(const PatchedKernel &PK) {
  std::unordered_set<uint64_t> S;
  S.reserve(PK.SiteMap.size());
  for (const auto &SI : PK.SiteMap)
    S.insert(SI.OriginalPC);
  return S;
}

// Build a `TraceConfigProvider` that hands out the elements of `Traces` in
// order, returning nullptr once the variant index overshoots the array.
// Post-refactor `getOrPatchVariants` pulls these lazily instead of taking a
// pre-materialized `ArrayRef<TraceConfig>`.
KernelPatcher::TraceConfigProvider
providerFromArray(const TraceConfig *Traces, size_t N) {
  return KernelPatcher::TraceConfigProvider(
      [Traces, N](uint32_t V) -> const TraceConfig * {
        return V < N ? &Traces[V] : nullptr;
      });
}

} // namespace

//===----------------------------------------------------------------------===//
// IR-001: Single-variant call via the new API matches legacy getOrPatch
//===----------------------------------------------------------------------===//

TEST_F(Fixture, SingleVariantEquivalentToGetOrPatch) {
  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  TraceConfig T = makeTrace(0xDEAD000000000000ULL);
  auto Legacy = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                       InstrumentationMode::MEMORY_ONLY, &T);
  ASSERT_TRUE(!!Legacy) << toString(Legacy.takeError());
  const size_t LegacySites = (*Legacy)->SiteMap.size();
  (*Patcher)->clearCache();

  TraceConfig PerVariant[1] = {T};
  auto Vars = (*Patcher)->getOrPatchVariants(
      CodeObj, Symbol, InstrumentationMode::MEMORY_ONLY,
      /*MaxVariants=*/1u, providerFromArray(PerVariant, 1));
  ASSERT_TRUE(!!Vars) << toString(Vars.takeError());
  ASSERT_EQ(Vars->size(), 1u);
  EXPECT_EQ((*Vars)[0]->SiteMap.size(), LegacySites)
      << "single-variant getOrPatchVariants must match legacy patch count";
}

//===----------------------------------------------------------------------===//
// IR-002: Two variants — disjoint OriginalPCs and monotonic union
//===----------------------------------------------------------------------===//

TEST_F(Fixture, TwoVariantsAreDisjointAndMonotonic) {
  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  TraceConfig PerVariant[2] = {
      makeTrace(0xDEAD000000000000ULL),
      makeTrace(0xCAFE000000000000ULL),
  };
  auto Vars = (*Patcher)->getOrPatchVariants(
      CodeObj, Symbol, InstrumentationMode::MEMORY_ONLY,
      /*MaxVariants=*/2u, providerFromArray(PerVariant, 2));
  ASSERT_TRUE(!!Vars) << toString(Vars.takeError());
  ASSERT_GE(Vars->size(), 1u);

  auto V0PCs = pcsOf(*(*Vars)[0]);
  ASSERT_FALSE(V0PCs.empty())
      << "variant 0 must instrument at least one site on mega_gather";

  if (Vars->size() >= 2) {
    auto V1PCs = pcsOf(*(*Vars)[1]);
    // Disjointness: no PC appears in both variants.
    for (uint64_t PC : V1PCs)
      EXPECT_EQ(V0PCs.count(PC), 0u)
          << "PC 0x" << std::hex << PC << std::dec
          << " appears in both variant 0 and variant 1 SiteMaps";

    // Union count must match sum of individual counts when disjoint.
    std::unordered_set<uint64_t> Union = V0PCs;
    Union.insert(V1PCs.begin(), V1PCs.end());
    EXPECT_EQ(Union.size(), V0PCs.size() + V1PCs.size());
  }
}

//===----------------------------------------------------------------------===//
// IR-003: Cache reuse — second getOrPatchVariants call returns same pointers
//===----------------------------------------------------------------------===//

TEST_F(Fixture, CacheReusesPatchedKernelPointers) {
  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  TraceConfig PerVariant[2] = {
      makeTrace(0xDEAD000000000000ULL),
      makeTrace(0xCAFE000000000000ULL),
  };
  auto First = (*Patcher)->getOrPatchVariants(
      CodeObj, Symbol, InstrumentationMode::MEMORY_ONLY,
      /*MaxVariants=*/2u, providerFromArray(PerVariant, 2));
  ASSERT_TRUE(!!First) << toString(First.takeError());

  auto Second = (*Patcher)->getOrPatchVariants(
      CodeObj, Symbol, InstrumentationMode::MEMORY_ONLY,
      /*MaxVariants=*/2u, providerFromArray(PerVariant, 2));
  ASSERT_TRUE(!!Second) << toString(Second.takeError());

  ASSERT_EQ(First->size(), Second->size());
  for (size_t i = 0; i < First->size(); ++i)
    EXPECT_EQ((*First)[i], (*Second)[i])
        << "variant " << i << " pointer changed between calls";

  auto Stats = (*Patcher)->getCacheStats();
  EXPECT_GE(Stats.CacheHits, 1u) << "second call should hit the cache";
}
