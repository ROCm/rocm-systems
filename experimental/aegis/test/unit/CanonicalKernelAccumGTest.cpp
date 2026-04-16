//===-- CanonicalKernelAccumGTest.cpp ---------------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for `CanonicalKernelAccum`: the pure per-kernel accumulator
/// keyed by original program counter that backs instrumentation replay.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/ProfilingResultsSink.h"
#include <gtest/gtest.h>

using namespace aegisbit;

namespace {

SiteInfo makeVMEMSite(uint32_t ID, uint64_t OriginalPC, uint32_t ElemSize = 4,
                     const std::string &Name = "GLOBAL_LOAD_DWORD") {
  SiteInfo SI;
  SI.SiteID = ID;
  SI.PC = OriginalPC;
  SI.OriginalPC = OriginalPC;
  SI.InstrName = Name;
  SI.IsLoad = true;
  SI.ElemSize = ElemSize;
  SI.IsLDS = false;
  return SI;
}

SiteInfo makeLDSSite(uint32_t ID, uint64_t OriginalPC, uint32_t ElemSize = 4,
                    const std::string &Name = "DS_READ_B32") {
  SiteInfo SI;
  SI.SiteID = ID;
  SI.PC = OriginalPC;
  SI.OriginalPC = OriginalPC;
  SI.InstrName = Name;
  SI.IsLoad = true;
  SI.ElemSize = ElemSize;
  SI.IsLDS = true;
  return SI;
}

} // namespace

TEST(CanonicalKernelAccum, InitiallyEmpty) {
  CanonicalKernelAccum A;
  EXPECT_TRUE(A.VMEMByPC.empty());
  EXPECT_TRUE(A.LDSByPC.empty());
}

TEST(CanonicalKernelAccum, SingleVMEMInsert) {
  CanonicalKernelAccum A;
  A.addVMEMSample(makeVMEMSite(7, 0x1000), /*CacheLines=*/4, /*Samples=*/1);
  ASSERT_EQ(A.VMEMByPC.size(), 1u);
  const auto &E = A.VMEMByPC.at(0x1000);
  EXPECT_EQ(E.OriginalPC, 0x1000u);
  EXPECT_EQ(E.FirstSeenSiteID, 7u);
  EXPECT_EQ(E.TotalCacheLines, 4u);
  EXPECT_EQ(E.TotalSamples, 1u);
  EXPECT_EQ(E.FirstSeenInfo.InstrName, "GLOBAL_LOAD_DWORD");
}

TEST(CanonicalKernelAccum, DuplicatePCAccumulates) {
  CanonicalKernelAccum A;
  A.addVMEMSample(makeVMEMSite(7, 0x1000), 4, 1);
  // Second variant hits the same PC with a different local site id.
  A.addVMEMSample(makeVMEMSite(42, 0x1000), 2, 3);

  ASSERT_EQ(A.VMEMByPC.size(), 1u);
  const auto &E = A.VMEMByPC.at(0x1000);
  EXPECT_EQ(E.TotalCacheLines, 6u);
  EXPECT_EQ(E.TotalSamples, 4u);
  // FirstSeenSiteID is stable (not overwritten by a later variant).
  EXPECT_EQ(E.FirstSeenSiteID, 7u);
}

TEST(CanonicalKernelAccum, DistinctPCsCoexist) {
  CanonicalKernelAccum A;
  A.addVMEMSample(makeVMEMSite(0, 0x1000), 2, 1);
  A.addVMEMSample(makeVMEMSite(1, 0x2000), 3, 2);
  A.addVMEMSample(makeVMEMSite(2, 0x3000), 4, 4);

  EXPECT_EQ(A.VMEMByPC.size(), 3u);
  EXPECT_EQ(A.VMEMByPC.at(0x1000).TotalCacheLines, 2u);
  EXPECT_EQ(A.VMEMByPC.at(0x2000).TotalCacheLines, 3u);
  EXPECT_EQ(A.VMEMByPC.at(0x3000).TotalCacheLines, 4u);
}

TEST(CanonicalKernelAccum, LDSDuplicatePCAccumulates) {
  CanonicalKernelAccum A;
  A.addLDSSample(makeLDSSite(5, 0x4000), /*UniqueBanks=*/8, /*Samples=*/1);
  A.addLDSSample(makeLDSSite(17, 0x4000), 4, 2);

  ASSERT_EQ(A.LDSByPC.size(), 1u);
  const auto &E = A.LDSByPC.at(0x4000);
  EXPECT_EQ(E.TotalUniqueBanks, 12u);
  EXPECT_EQ(E.TotalSamples, 3u);
  EXPECT_EQ(E.FirstSeenSiteID, 5u);
  EXPECT_TRUE(E.FirstSeenInfo.IsLDS);
}

TEST(CanonicalKernelAccum, VMEMAndLDSAreSeparateTables) {
  CanonicalKernelAccum A;
  // Same PC but one VMEM, one LDS: the site-type tables don't interfere.
  A.addVMEMSample(makeVMEMSite(0, 0x5000), 1, 1);
  A.addLDSSample(makeLDSSite(0, 0x5000), 2, 2);
  EXPECT_EQ(A.VMEMByPC.size(), 1u);
  EXPECT_EQ(A.LDSByPC.size(), 1u);
  EXPECT_EQ(A.VMEMByPC.at(0x5000).TotalCacheLines, 1u);
  EXPECT_EQ(A.LDSByPC.at(0x5000).TotalUniqueBanks, 2u);
}

TEST(CanonicalKernelAccum, FirstSeenSiteIDIsStableAcrossVariants) {
  CanonicalKernelAccum A;
  // Simulate three variants adding into the same canonical PC.
  A.addVMEMSample(makeVMEMSite(100, 0x8000), 1, 1);
  A.addVMEMSample(makeVMEMSite(7, 0x8000), 1, 1);
  A.addVMEMSample(makeVMEMSite(250, 0x8000), 1, 1);
  EXPECT_EQ(A.VMEMByPC.at(0x8000).FirstSeenSiteID, 100u);
  EXPECT_EQ(A.VMEMByPC.at(0x8000).TotalSamples, 3u);
}
