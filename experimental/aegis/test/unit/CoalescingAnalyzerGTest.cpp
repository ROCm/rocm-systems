//===-- CoalescingAnalyzerGTest.cpp - Coalescing Analysis Tests --*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/CoalescingAnalyzer.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace aegisbit;

//===----------------------------------------------------------------------===//
// inferElemSize tests
//===----------------------------------------------------------------------===//

TEST(InferElemSize, DWORD) {
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_DWORD"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("GLOBAL_STORE_DWORD"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("BUFFER_LOAD_DWORD_OFFEN_gfx90a"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("BUFFER_STORE_DWORD_OFFEN_gfx90a"));
}

TEST(InferElemSize, DWORDX2) {
  EXPECT_EQ(8u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_DWORDX2"));
  EXPECT_EQ(8u, CoalescingAnalyzer::inferElemSize("GLOBAL_STORE_DWORDX2"));
  EXPECT_EQ(8u, CoalescingAnalyzer::inferElemSize("BUFFER_LOAD_DWORDX2_OFFEN_gfx90a"));
}

TEST(InferElemSize, DWORDX3) {
  EXPECT_EQ(12u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_DWORDX3"));
  EXPECT_EQ(12u, CoalescingAnalyzer::inferElemSize("BUFFER_STORE_DWORDX3_OFFEN"));
}

TEST(InferElemSize, DWORDX4) {
  EXPECT_EQ(16u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_DWORDX4"));
  EXPECT_EQ(16u, CoalescingAnalyzer::inferElemSize("BUFFER_LOAD_DWORDX4_OFFEN_gfx90a"));
  EXPECT_EQ(16u, CoalescingAnalyzer::inferElemSize("GLOBAL_STORE_DWORDX4"));
}

TEST(InferElemSize, SHORT) {
  EXPECT_EQ(2u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_USHORT"));
  EXPECT_EQ(2u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_SSHORT"));
  EXPECT_EQ(2u, CoalescingAnalyzer::inferElemSize("GLOBAL_STORE_SHORT"));
  EXPECT_EQ(2u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_SHORT_D16"));
  EXPECT_EQ(2u, CoalescingAnalyzer::inferElemSize("BUFFER_LOAD_SHORT_D16_OFFEN"));
}

TEST(InferElemSize, BYTE) {
  EXPECT_EQ(1u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_UBYTE"));
  EXPECT_EQ(1u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_SBYTE"));
  EXPECT_EQ(1u, CoalescingAnalyzer::inferElemSize("GLOBAL_STORE_BYTE"));
  EXPECT_EQ(1u, CoalescingAnalyzer::inferElemSize("BUFFER_LOAD_UBYTE_OFFEN_gfx90a"));
}

TEST(InferElemSize, D16Only) {
  EXPECT_EQ(2u, CoalescingAnalyzer::inferElemSize("GLOBAL_LOAD_D16_HI"));
}

TEST(InferElemSize, DSInstructions) {
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("DS_READ_B32"));
  EXPECT_EQ(8u, CoalescingAnalyzer::inferElemSize("DS_READ_B64"));
  EXPECT_EQ(16u, CoalescingAnalyzer::inferElemSize("DS_READ_B128"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("DS_WRITE_B32"));
  EXPECT_EQ(8u, CoalescingAnalyzer::inferElemSize("DS_WRITE_B64"));
  EXPECT_EQ(1u, CoalescingAnalyzer::inferElemSize("DS_READ_U8"));
  EXPECT_EQ(2u, CoalescingAnalyzer::inferElemSize("DS_READ_U16"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("DS_ADD_U32"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("DS_ADD_RTN_U32"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("DS_READ2_B32"));
  EXPECT_EQ(8u, CoalescingAnalyzer::inferElemSize("DS_READ2_B64"));
  EXPECT_EQ(12u, CoalescingAnalyzer::inferElemSize("DS_READ_B96"));
}

TEST(InferElemSize, UnknownDefaultsTo4) {
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize("SOMETHING_UNKNOWN"));
  EXPECT_EQ(4u, CoalescingAnalyzer::inferElemSize(""));
}

//===----------------------------------------------------------------------===//
// analyzeAccess tests — coalesced float32 (4B stride)
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, CoalescedDword) {
  uint64_t Addrs[64];
  uint64_t Base = 0x100000;
  for (int i = 0; i < 64; i++)
    Addrs[i] = Base + i * 4;

  auto M = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 4);

  EXPECT_EQ(64u, M.ActiveLanes);
  EXPECT_EQ(2u, M.CacheLines);       // 256B / 128B = 2
  EXPECT_EQ(8u, M.Sectors);          // 256B / 32B = 8
  EXPECT_EQ(256u, M.BytesRequested); // 64 × 4
  EXPECT_EQ(256u, M.BytesFetched);   // 2 × 128
  EXPECT_FLOAT_EQ(1.0f, M.Efficiency);
  EXPECT_EQ(AccessMetrics::Coalesced, M.AccessPattern);
}

//===----------------------------------------------------------------------===//
// analyzeAccess — coalesced float64 (8B stride, DWORDX2)
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, CoalescedDwordX2) {
  uint64_t Addrs[64];
  uint64_t Base = 0x100000;
  for (int i = 0; i < 64; i++)
    Addrs[i] = Base + i * 8;

  auto M = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 8);

  EXPECT_EQ(64u, M.ActiveLanes);
  // 64 × 8B = 512B. 512 / 128 = 4 cache lines.
  EXPECT_EQ(4u, M.CacheLines);
  // 64 × 8B = 512B. 512 / 32 = 16 sectors.
  EXPECT_EQ(16u, M.Sectors);
  EXPECT_EQ(512u, M.BytesRequested);
  EXPECT_EQ(512u, M.BytesFetched);   // 4 × 128
  EXPECT_FLOAT_EQ(1.0f, M.Efficiency);
  EXPECT_EQ(AccessMetrics::Coalesced, M.AccessPattern);
}

//===----------------------------------------------------------------------===//
// analyzeAccess — coalesced float16 (2B stride, USHORT)
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, CoalescedShort) {
  uint64_t Addrs[64];
  uint64_t Base = 0x100000;
  for (int i = 0; i < 64; i++)
    Addrs[i] = Base + i * 2;

  auto M = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 2);

  EXPECT_EQ(64u, M.ActiveLanes);
  // 64 × 2B = 128B = 1 cache line.
  EXPECT_EQ(1u, M.CacheLines);
  // 128B / 32B = 4 sectors.
  EXPECT_EQ(4u, M.Sectors);
  EXPECT_EQ(128u, M.BytesRequested);
  EXPECT_EQ(128u, M.BytesFetched);
  EXPECT_FLOAT_EQ(1.0f, M.Efficiency);
  EXPECT_EQ(AccessMetrics::Coalesced, M.AccessPattern);
}

//===----------------------------------------------------------------------===//
// analyzeAccess — coalesced DWORDX4 (16B stride)
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, CoalescedDwordX4) {
  uint64_t Addrs[64];
  uint64_t Base = 0x100000;
  for (int i = 0; i < 64; i++)
    Addrs[i] = Base + i * 16;

  auto M = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 16);

  EXPECT_EQ(64u, M.ActiveLanes);
  // 64 × 16B = 1024B. 1024 / 128 = 8 cache lines.
  EXPECT_EQ(8u, M.CacheLines);
  // 1024 / 32 = 32 sectors.
  EXPECT_EQ(32u, M.Sectors);
  EXPECT_EQ(1024u, M.BytesRequested);
  EXPECT_EQ(1024u, M.BytesFetched);
  EXPECT_FLOAT_EQ(1.0f, M.Efficiency);
  EXPECT_EQ(AccessMetrics::Coalesced, M.AccessPattern);
}

//===----------------------------------------------------------------------===//
// analyzeAccess — strided access with correct DWORDX2 accounting
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, StridedDwordX2) {
  uint64_t Addrs[64];
  uint64_t Base = 0x100000;
  // Stride of 64 bytes between elements, but each element is 8B.
  for (int i = 0; i < 64; i++)
    Addrs[i] = Base + i * 64;

  auto M = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 8);

  EXPECT_EQ(64u, M.ActiveLanes);
  // 64 elements × 64B stride = 4096B range. Each element is 8B.
  // This is heavily strided — many more cache lines than necessary.
  EXPECT_GT(M.CacheLines, 4u);  // Much more than the 4 for coalesced
  EXPECT_EQ(AccessMetrics::Strided, M.AccessPattern);
  EXPECT_LT(M.Efficiency, 0.5f);
}

//===----------------------------------------------------------------------===//
// analyzeAccess — DWORDX2 wrong elem size demonstrates the bug we're fixing
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, DwordX2WrongElemSizeGivesWrongEfficiency) {
  uint64_t Addrs[64];
  uint64_t Base = 0x100000;
  for (int i = 0; i < 64; i++)
    Addrs[i] = Base + i * 8;

  // Analyzing DWORDX2 with ElemSize=4 (the old bug) undercounts bytes requested.
  auto Wrong = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 4);
  auto Right = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 8);

  // 4 cache lines (512B range / 128). BytesFetched = 4 × 128 = 512.
  // Wrong ElemSize=4: BytesRequested=256, Eff = 256/512 = 0.5.
  // Right ElemSize=8: BytesRequested=512, Eff = 512/512 = 1.0.
  EXPECT_FLOAT_EQ(0.5f, Wrong.Efficiency);
  EXPECT_FLOAT_EQ(1.0f, Right.Efficiency);
}

//===----------------------------------------------------------------------===//
// analyzeAccess — BYTE (1B) elements
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, CoalescedByte) {
  uint64_t Addrs[64];
  uint64_t Base = 0x100000;
  for (int i = 0; i < 64; i++)
    Addrs[i] = Base + i * 1;

  auto M = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 1);

  EXPECT_EQ(64u, M.ActiveLanes);
  // 64 × 1B = 64B. Fits in 1 cache line.
  EXPECT_EQ(1u, M.CacheLines);
  // 64B / 32B = 2 sectors.
  EXPECT_EQ(2u, M.Sectors);
  EXPECT_EQ(64u, M.BytesRequested);
  EXPECT_EQ(128u, M.BytesFetched);   // 1 × 128 (full cache line)
  EXPECT_FLOAT_EQ(0.5f, M.Efficiency);
}

//===----------------------------------------------------------------------===//
// analyzeAccess — cache line spanning for large elements
//===----------------------------------------------------------------------===//

TEST(AnalyzeAccess, DwordX4SpanningCacheLineBoundary) {
  uint64_t Addrs[64] = {};
  // Place one 16B element at offset 120 within a 128-byte cache line.
  // Bytes 120-135 span cache line boundary at 128.
  Addrs[0] = 0x100078; // 0x100078 = 1048696, offset 120 within 128-byte line

  auto M = CoalescingAnalyzer::analyzeAccess(Addrs, 0, 16);

  EXPECT_EQ(1u, M.ActiveLanes);
  EXPECT_EQ(2u, M.CacheLines);  // Spans two cache lines (120-127, 128-135)
  // 0x100078/32 = sector 32771, (0x100078+15)/32 = sector 32772 → 2 sectors.
  EXPECT_EQ(2u, M.Sectors);
}

//===----------------------------------------------------------------------===//
// Per-site analyzeBuffer overload
//===----------------------------------------------------------------------===//

TEST(AnalyzeBuffer, PerSiteElemSize) {
  // Build 2 fake records: site 0 (DWORDX2=8B) and site 1 (DWORD=4B).
  struct RawRecord {
    uint32_t SiteID;
    uint32_t Padding;
    uint64_t Addresses[64];
  };
  static_assert(sizeof(RawRecord) == TraceConfig::RecordSize);

  std::vector<RawRecord> Records(2);
  std::memset(Records.data(), 0, sizeof(RawRecord) * 2);

  uint64_t Base = 0x200000;

  // Record 0: site 0, coalesced 8B stride.
  Records[0].SiteID = 0;
  for (int i = 0; i < 64; i++)
    Records[0].Addresses[i] = Base + i * 8;

  // Record 1: site 1, coalesced 4B stride.
  Records[1].SiteID = 1;
  for (int i = 0; i < 64; i++)
    Records[1].Addresses[i] = Base + i * 4;

  std::vector<SiteInfo> SiteMap(2);
  SiteMap[0].SiteID = 0;
  SiteMap[0].ElemSize = 8;
  SiteMap[0].InstrName = "GLOBAL_LOAD_DWORDX2";
  SiteMap[1].SiteID = 1;
  SiteMap[1].ElemSize = 4;
  SiteMap[1].InstrName = "GLOBAL_LOAD_DWORD";

  auto Results = CoalescingAnalyzer::analyzeBuffer(
      Records.data(), 2, SiteMap);

  ASSERT_EQ(2u, Results.size());

  // Site 0: 64 × 8B = 512B requested, 512B fetched → 100%
  EXPECT_EQ(0u, Results[0].SiteID);
  EXPECT_EQ(512u, Results[0].BytesRequested);
  EXPECT_FLOAT_EQ(1.0f, Results[0].Efficiency);

  // Site 1: 64 × 4B = 256B requested, 256B fetched → 100%
  EXPECT_EQ(1u, Results[1].SiteID);
  EXPECT_EQ(256u, Results[1].BytesRequested);
  EXPECT_FLOAT_EQ(1.0f, Results[1].Efficiency);
}

//===----------------------------------------------------------------------===//
// Uniform analyzeBuffer still works
//===----------------------------------------------------------------------===//

TEST(AnalyzeBuffer, UniformElemSize) {
  struct RawRecord {
    uint32_t SiteID;
    uint32_t Padding;
    uint64_t Addresses[64];
  };

  RawRecord Rec;
  std::memset(&Rec, 0, sizeof(Rec));
  Rec.SiteID = 0;
  uint64_t Base = 0x300000;
  for (int i = 0; i < 64; i++)
    Rec.Addresses[i] = Base + i * 4;

  auto Results = CoalescingAnalyzer::analyzeBuffer(&Rec, 1, 4);
  ASSERT_EQ(1u, Results.size());
  EXPECT_EQ(256u, Results[0].BytesRequested);
  EXPECT_FLOAT_EQ(1.0f, Results[0].Efficiency);
}

//===----------------------------------------------------------------------===//
// Summarize with per-site element sizes
//===----------------------------------------------------------------------===//

TEST(Summarize, MixedElemSizeEfficiency) {
  // Two sites: site 0 is DWORDX2 (coalesced at 8B stride),
  // site 1 is DWORD but with 8B stride (strided/waste).
  uint64_t Addrs0[64], Addrs1[64];
  uint64_t Base = 0x400000;
  for (int i = 0; i < 64; i++) {
    Addrs0[i] = Base + i * 8;       // DWORDX2, 8B stride → perfect
    Addrs1[i] = Base + i * 8;       // DWORD but 8B stride → 50% efficiency
  }

  std::vector<AccessMetrics> Metrics;
  Metrics.push_back(CoalescingAnalyzer::analyzeAccess(Addrs0, 0, 8));
  Metrics.push_back(CoalescingAnalyzer::analyzeAccess(Addrs1, 1, 4));

  auto Summary = CoalescingAnalyzer::summarize(Metrics);

  ASSERT_EQ(2u, Summary.PerSite.size());

  // Site 0: 100% efficiency
  EXPECT_FLOAT_EQ(1.0f, Summary.PerSite[0].AvgEfficiency);

  // Site 1: 50% efficiency (requests 256B but fetches 512B)
  EXPECT_FLOAT_EQ(0.5f, Summary.PerSite[1].AvgEfficiency);

  // Overall: (512 + 256) / (512 + 512) = 768/1024 = 0.75
  EXPECT_FLOAT_EQ(0.75f, Summary.OverallEfficiency);
}

//===----------------------------------------------------------------------===//
// LDS Bank Conflict Tests
//===----------------------------------------------------------------------===//

TEST(LDSBankConflict, NoConflict) {
  // 32 lanes accessing 32 consecutive dwords -> all different banks -> 1 cycle
  // Start from 128 to avoid address 0 (sentinel for inactive lane)
  uint64_t Addrs[64] = {};
  for (int i = 0; i < 32; i++)
    Addrs[i] = 128 + i * 4;  // bank = ((128+i*4) / 4) % 32 = (32+i) % 32 = i
  auto M = CoalescingAnalyzer::analyzeLDSAccess(Addrs, 0, 4);
  EXPECT_EQ(1u, M.ConflictCycles);
  EXPECT_EQ(32u, M.BanksUsed);
  EXPECT_EQ(32u, M.ActiveLanes);
}

TEST(LDSBankConflict, AllSameBank) {
  // 64 lanes all hitting bank 0 with different addresses
  // Start from 128 to avoid address 0 (sentinel for inactive lane)
  uint64_t Addrs[64];
  for (int i = 0; i < 64; i++)
    Addrs[i] = 128 + i * 128;  // bank = ((128+i*128) / 4) % 32 = (32+i*32) % 32 = 0
  auto M = CoalescingAnalyzer::analyzeLDSAccess(Addrs, 0, 4);
  EXPECT_EQ(64u, M.ConflictCycles);
  EXPECT_EQ(1u, M.BanksUsed);
}

TEST(LDSBankConflict, Broadcast) {
  // All 64 lanes access the same address -> broadcast, 1 cycle
  uint64_t Addrs[64];
  for (int i = 0; i < 64; i++)
    Addrs[i] = 256;
  auto M = CoalescingAnalyzer::analyzeLDSAccess(Addrs, 0, 4);
  EXPECT_EQ(1u, M.ConflictCycles);
  EXPECT_EQ(63u, M.BroadcastLanes);  // 64 active, 1 unique addr per bank
}

TEST(LDSBankConflict, TwoWayConflict) {
  // 64 lanes: first 32 get addrs in banks 0..31,
  // second 32 get addrs in banks 0..31 again (different addresses).
  // Each bank has 2 unique addresses -> 2 conflict cycles.
  // Offset by 256 to avoid address 0.
  uint64_t Addrs[64];
  for (int i = 0; i < 64; i++)
    Addrs[i] = 256 + (i / 32) * 128 + (i % 32) * 4;
  auto M = CoalescingAnalyzer::analyzeLDSAccess(Addrs, 0, 4);
  EXPECT_EQ(2u, M.ConflictCycles);
  EXPECT_EQ(32u, M.BanksUsed);
}

TEST(LDSBankConflict, WithDSOffset) {
  // 32 lanes accessing sequential dwords -> no conflicts
  // Offset by 128 to avoid address 0
  uint64_t Addrs[64] = {};
  for (int i = 0; i < 32; i++)
    Addrs[i] = 128 + i * 4;
  auto M1 = CoalescingAnalyzer::analyzeLDSAccess(Addrs, 0, 4, 0);
  EXPECT_EQ(1u, M1.ConflictCycles);

  // Additional DSOffset of 128 bytes is a multiple of 128 (bank-width * num-banks)
  // so bank assignment is unchanged
  auto M2 = CoalescingAnalyzer::analyzeLDSAccess(Addrs, 0, 4, 128);
  EXPECT_EQ(1u, M2.ConflictCycles);

  // DSOffset of 4 shifts every address by one bank position -> still no conflicts
  auto M3 = CoalescingAnalyzer::analyzeLDSAccess(Addrs, 0, 4, 4);
  EXPECT_EQ(1u, M3.ConflictCycles);
}

TEST(LDSBankConflict, SummarizeLDS) {
  // Two accesses to site 0: first conflict-free, second with 2-way conflict
  uint64_t Addrs1[64] = {};
  for (int i = 0; i < 32; i++)
    Addrs1[i] = 128 + i * 4;  // No conflicts

  uint64_t Addrs2[64] = {};
  for (int i = 0; i < 64; i++)
    Addrs2[i] = 256 + (i / 32) * 128 + (i % 32) * 4;  // 2-way conflict

  std::vector<LDSAccessMetrics> Metrics;
  Metrics.push_back(CoalescingAnalyzer::analyzeLDSAccess(Addrs1, 0, 4));
  Metrics.push_back(CoalescingAnalyzer::analyzeLDSAccess(Addrs2, 0, 4));

  std::vector<SiteInfo> SiteMap(1);
  SiteMap[0].SiteID = 0;
  SiteMap[0].IsLDS = true;
  SiteMap[0].InstrName = "DS_READ_B32";
  SiteMap[0].ElemSize = 4;

  auto Summary = CoalescingAnalyzer::summarizeLDS(Metrics, "test_kernel", SiteMap);
  EXPECT_EQ(2u, Summary.TotalRecords);
  EXPECT_EQ(1u, Summary.NumSites);
  ASSERT_EQ(1u, Summary.PerSite.size());
  EXPECT_EQ(1u, Summary.PerSite[0].ConflictFreeCount);
  EXPECT_EQ(1u, Summary.PerSite[0].ConflictCount);
  EXPECT_EQ(1u, Summary.PerSite[0].MinConflictCycles);
  EXPECT_EQ(2u, Summary.PerSite[0].MaxConflictCycles);
  EXPECT_FLOAT_EQ(1.5f, Summary.PerSite[0].AvgConflictCycles);
}

//===----------------------------------------------------------------------===//
// OnGpuReduce LDS Counter Tests (max-lanes-per-bank metric)
//===----------------------------------------------------------------------===//

TEST(OnGpuCountersLDS, ConflictFree) {
  // Site 0 is LDS with avg max-lanes-per-bank = 1.0 (no conflicts)
  // Counter layout: [total_max_lanes, total_samples] per site
  uint32_t Counters[2] = {100, 100};  // avg = 1.0

  std::vector<SiteInfo> SiteMap(1);
  SiteMap[0].SiteID = 0;
  SiteMap[0].IsLDS = true;
  SiteMap[0].InstrName = "DS_READ_B32";
  SiteMap[0].ElemSize = 4;

  auto S = CoalescingAnalyzer::analyzeOnGpuCountersLDS(
      Counters, SiteMap, "test_kernel");
  ASSERT_EQ(1u, S.PerSite.size());
  EXPECT_FLOAT_EQ(1.0f, S.PerSite[0].AvgConflictCycles);
  EXPECT_FALSE(S.PerSite[0].HasPerSampleBreakdown);
  EXPECT_FLOAT_EQ(1.0f, S.OverallAvgConflictCycles);
}

TEST(OnGpuCountersLDS, TwoWayConflict) {
  // avg max-lanes-per-bank = 2.0
  uint32_t Counters[2] = {200, 100};

  std::vector<SiteInfo> SiteMap(1);
  SiteMap[0].SiteID = 0;
  SiteMap[0].IsLDS = true;
  SiteMap[0].InstrName = "DS_READ_B64";
  SiteMap[0].ElemSize = 8;

  auto S = CoalescingAnalyzer::analyzeOnGpuCountersLDS(
      Counters, SiteMap, "test_kernel");
  ASSERT_EQ(1u, S.PerSite.size());
  EXPECT_FLOAT_EQ(2.0f, S.PerSite[0].AvgConflictCycles);
  EXPECT_FALSE(S.PerSite[0].HasPerSampleBreakdown);
}

TEST(OnGpuCountersLDS, SevereConflict) {
  // avg max-lanes-per-bank = 64 (worst case: all lanes in one bank)
  uint32_t Counters[2] = {6400, 100};

  std::vector<SiteInfo> SiteMap(1);
  SiteMap[0].SiteID = 0;
  SiteMap[0].IsLDS = true;
  SiteMap[0].InstrName = "DS_WRITE_B32";
  SiteMap[0].ElemSize = 4;

  auto S = CoalescingAnalyzer::analyzeOnGpuCountersLDS(
      Counters, SiteMap, "test_kernel");
  ASSERT_EQ(1u, S.PerSite.size());
  EXPECT_FLOAT_EQ(64.0f, S.PerSite[0].AvgConflictCycles);
  EXPECT_FALSE(S.PerSite[0].HasPerSampleBreakdown);
}

TEST(OnGpuCountersLDS, MixedSites) {
  // Two LDS sites + one VMEM site (should be skipped)
  // Site 0: LDS, conflict-free (avg=1.0, 50 samples)
  // Site 1: VMEM (should be skipped)
  // Site 2: LDS, 4-way conflict (avg=4.0, 200 samples)
  uint32_t Counters[6] = {
    50, 50,     // site 0: total_max_lanes=50, samples=50 → avg=1.0
    999, 999,   // site 1: VMEM, ignored
    800, 200,   // site 2: total_max_lanes=800, samples=200 → avg=4.0
  };

  std::vector<SiteInfo> SiteMap(3);
  SiteMap[0].SiteID = 0;
  SiteMap[0].IsLDS = true;
  SiteMap[0].InstrName = "DS_READ_B32";
  SiteMap[0].ElemSize = 4;
  SiteMap[1].SiteID = 1;
  SiteMap[1].IsLDS = false;
  SiteMap[1].InstrName = "GLOBAL_LOAD_DWORD";
  SiteMap[1].ElemSize = 4;
  SiteMap[2].SiteID = 2;
  SiteMap[2].IsLDS = true;
  SiteMap[2].InstrName = "DS_WRITE_B128";
  SiteMap[2].ElemSize = 16;

  auto S = CoalescingAnalyzer::analyzeOnGpuCountersLDS(
      Counters, SiteMap, "mixed_kernel");

  EXPECT_EQ(2u, S.NumSites);
  EXPECT_EQ(250u, S.TotalRecords);  // 50 + 200
  ASSERT_EQ(2u, S.PerSite.size());

  // Site 0: conflict-free (avg N-way = 1.0)
  EXPECT_EQ(0u, S.PerSite[0].SiteID);
  EXPECT_FLOAT_EQ(1.0f, S.PerSite[0].AvgConflictCycles);
  EXPECT_FALSE(S.PerSite[0].HasPerSampleBreakdown);

  // Site 2: 4-way conflict (avg N-way = 4.0)
  EXPECT_EQ(2u, S.PerSite[1].SiteID);
  EXPECT_FLOAT_EQ(4.0f, S.PerSite[1].AvgConflictCycles);
  EXPECT_FALSE(S.PerSite[1].HasPerSampleBreakdown);

  // Overall: (1.0*50 + 4.0*200) / 250 = 850/250 = 3.4
  EXPECT_FLOAT_EQ(3.4f, S.OverallAvgConflictCycles);
}

TEST(OnGpuCountersLDS, ZeroSamplesSkipped) {
  // Site with 0 samples should be excluded
  uint32_t Counters[2] = {0, 0};

  std::vector<SiteInfo> SiteMap(1);
  SiteMap[0].SiteID = 0;
  SiteMap[0].IsLDS = true;
  SiteMap[0].InstrName = "DS_READ_B32";
  SiteMap[0].ElemSize = 4;

  auto S = CoalescingAnalyzer::analyzeOnGpuCountersLDS(
      Counters, SiteMap, "test_kernel");
  EXPECT_EQ(0u, S.NumSites);
  EXPECT_EQ(0u, S.TotalRecords);
  EXPECT_TRUE(S.PerSite.empty());
}

TEST(LDSBankConflict, AnalyzeLDSBuffer) {
  struct RawRecord {
    uint32_t SiteID;
    uint32_t Padding;
    uint64_t Addresses[64];
  };
  static_assert(sizeof(RawRecord) == TraceConfig::RecordSize);

  std::vector<RawRecord> Records(2);
  std::memset(Records.data(), 0, sizeof(RawRecord) * 2);

  // Record 0: LDS site 0, no conflicts (offset by 128 to avoid address 0)
  Records[0].SiteID = 0;
  for (int i = 0; i < 32; i++)
    Records[0].Addresses[i] = 128 + i * 4;

  // Record 1: VMEM site 1 (should be skipped by analyzeLDSBuffer)
  Records[1].SiteID = 1;
  for (int i = 0; i < 64; i++)
    Records[1].Addresses[i] = 0x100000 + i * 4;

  std::vector<SiteInfo> SiteMap(2);
  SiteMap[0].SiteID = 0;
  SiteMap[0].IsLDS = true;
  SiteMap[0].ElemSize = 4;
  SiteMap[0].InstrName = "DS_READ_B32";
  SiteMap[1].SiteID = 1;
  SiteMap[1].IsLDS = false;
  SiteMap[1].ElemSize = 4;
  SiteMap[1].InstrName = "GLOBAL_LOAD_DWORD";

  auto Results = CoalescingAnalyzer::analyzeLDSBuffer(
      Records.data(), 2, SiteMap);

  // Only the LDS record should be processed
  ASSERT_EQ(1u, Results.size());
  EXPECT_EQ(0u, Results[0].SiteID);
  EXPECT_EQ(1u, Results[0].ConflictCycles);
}
