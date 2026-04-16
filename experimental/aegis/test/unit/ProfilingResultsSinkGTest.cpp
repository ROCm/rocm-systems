//===-- ProfilingResultsSinkGTest.cpp - Sink accumulation tests *- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for `ProfilingResultsSink` canonical-by-PC accumulation.  Verifies
/// that repeated and multi-variant `ingest` calls collapse onto the original
/// PC and accumulate `TotalCacheLines` / `TotalSamples` additively.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/ProfilingResultsSink.h"
#include "aegisbit/PersistentBufferCache.h"
#include "aegisbit/Types.h"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

using namespace aegisbit;

namespace {

/// Build a fake `PersistentTraceBuffer` backed by the caller-owned `Buffer`
/// vector.  Layout matches `OnGpuReduce`: per site, {cache_lines u32,
/// samples u32}.
PersistentTraceBuffer makeFakePB(std::vector<uint32_t> &Buffer,
                                 uint32_t NumSites) {
  Buffer.assign(NumSites * 2, 0);
  PersistentTraceBuffer PB;
  PB.Config.Strategy = PayloadStrategy::OnGpuReduce;
  PB.BufferPtr = Buffer.data();
  PB.BufferSize = Buffer.size() * sizeof(uint32_t);
  return PB;
}

SiteInfo vmemSite(uint32_t ID, uint64_t OriginalPC, uint32_t ElemSize = 4) {
  SiteInfo SI;
  SI.SiteID = ID;
  SI.PC = OriginalPC;
  SI.OriginalPC = OriginalPC;
  SI.InstrName = "GLOBAL_LOAD_DWORD";
  SI.IsLoad = true;
  SI.ElemSize = ElemSize;
  SI.IsLDS = false;
  return SI;
}

} // namespace

TEST(ProfilingResultsSink, IngestCanonicalizesByPC) {
  ProfilingResultsSink Sink;
  std::vector<uint32_t> Buf;
  auto PB = makeFakePB(Buf, /*NumSites=*/2);

  // Site 0 at PC 0x1000: 2 cache lines, 1 sample.
  Buf[0] = 2; Buf[1] = 1;
  // Site 1 at PC 0x2000: 4 cache lines, 2 samples.
  Buf[2] = 4; Buf[3] = 2;

  std::vector<SiteInfo> Sites{vmemSite(0, 0x1000), vmemSite(1, 0x2000)};
  Sink.ingest(PB, Sites, "kA", /*VariantID=*/0);

  const auto *Accum = Sink.findCanonical("kA");
  ASSERT_NE(Accum, nullptr);
  EXPECT_EQ(Accum->VMEMByPC.size(), 2u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x1000).TotalCacheLines, 2u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x1000).TotalSamples, 1u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x2000).TotalCacheLines, 4u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x2000).TotalSamples, 2u);
}

TEST(ProfilingResultsSink, OverlappingPCsCollapseAcrossCalls) {
  ProfilingResultsSink Sink;
  std::vector<uint32_t> Buf;
  auto PB = makeFakePB(Buf, 1);

  // First ingest: PC 0x1000 gets 4/1 (local site 0).
  Buf[0] = 4; Buf[1] = 1;
  Sink.ingest(PB, {vmemSite(0, 0x1000)}, "kB", /*VariantID=*/0);

  // Second ingest: PC 0x1000 again, but this time via local site 0 of
  // variant 1, contributing 6/3.
  Buf[0] = 6; Buf[1] = 3;
  Sink.ingest(PB, {vmemSite(0, 0x1000)}, "kB", /*VariantID=*/1);

  const auto *Accum = Sink.findCanonical("kB");
  ASSERT_NE(Accum, nullptr);
  ASSERT_EQ(Accum->VMEMByPC.size(), 1u);
  const auto &E = Accum->VMEMByPC.at(0x1000);
  EXPECT_EQ(E.TotalCacheLines, 10u);
  EXPECT_EQ(E.TotalSamples, 4u);
}

TEST(ProfilingResultsSink, DifferentVariantsMergeDistinctPCs) {
  ProfilingResultsSink Sink;
  std::vector<uint32_t> Buf;
  auto PB = makeFakePB(Buf, 2);

  // Variant 0 covers PC 0x1000 and 0x2000 (local sites 0, 1).
  Buf[0] = 1; Buf[1] = 1;
  Buf[2] = 2; Buf[3] = 2;
  Sink.ingest(PB, {vmemSite(0, 0x1000), vmemSite(1, 0x2000)}, "kC",
              /*VariantID=*/0);

  // Variant 1 covers PC 0x3000 and 0x4000 as its local sites 0, 1.
  Buf[0] = 3; Buf[1] = 3;
  Buf[2] = 4; Buf[3] = 4;
  Sink.ingest(PB, {vmemSite(0, 0x3000), vmemSite(1, 0x4000)}, "kC",
              /*VariantID=*/1);

  const auto *Accum = Sink.findCanonical("kC");
  ASSERT_NE(Accum, nullptr);
  EXPECT_EQ(Accum->VMEMByPC.size(), 4u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x1000).TotalSamples, 1u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x2000).TotalSamples, 2u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x3000).TotalSamples, 3u);
  EXPECT_EQ(Accum->VMEMByPC.at(0x4000).TotalSamples, 4u);
}

TEST(ProfilingResultsSink, ZeroSampleSitesAreSkipped) {
  ProfilingResultsSink Sink;
  std::vector<uint32_t> Buf;
  auto PB = makeFakePB(Buf, 2);
  // Only site 1 has non-zero samples; site 0 must be dropped from the
  // canonical table.
  Buf[0] = 99; Buf[1] = 0;
  Buf[2] = 1; Buf[3] = 1;
  Sink.ingest(PB, {vmemSite(0, 0x1000), vmemSite(1, 0x2000)}, "kD");

  const auto *Accum = Sink.findCanonical("kD");
  ASSERT_NE(Accum, nullptr);
  EXPECT_EQ(Accum->VMEMByPC.size(), 1u);
  EXPECT_TRUE(Accum->VMEMByPC.count(0x2000));
  EXPECT_FALSE(Accum->VMEMByPC.count(0x1000));
}

TEST(ProfilingResultsSink, ClearResetsCanonicalTable) {
  ProfilingResultsSink Sink;
  std::vector<uint32_t> Buf;
  auto PB = makeFakePB(Buf, 1);
  Buf[0] = 1; Buf[1] = 1;
  Sink.ingest(PB, {vmemSite(0, 0x1000)}, "kE");
  ASSERT_NE(Sink.findCanonical("kE"), nullptr);
  Sink.clear();
  EXPECT_EQ(Sink.findCanonical("kE"), nullptr);
}
