//===-- DescriptorUpdaterGTest.cpp - Descriptor Tests (GoogleTest) -*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for kernel descriptor parsing/modification using GoogleTest.
/// This is an example migrated test file showing the new test patterns.
///
/// Test IDs: K-001 through K-008 (Part A: Kernel descriptor tests)
///
//===----------------------------------------------------------------------===//

#include "fixtures/DescriptorFixture.h"
#include <gtest/gtest.h>

using namespace aegisbit;
using namespace aegisbit::test;

class DescriptorTest : public DescriptorFixture {};

//===----------------------------------------------------------------------===//
// Utility Tests
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, GranularityRoundUp) {
  // VGPR granularity is 4 for older architectures
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(1, 4), 4u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(4, 4), 4u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(5, 4), 8u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(128, 4), 128u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(129, 4), 132u);

  // VGPR granularity is 8 for gfx90a+
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(1, 8), 8u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(8, 8), 8u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(9, 8), 16u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(128, 8), 128u);
  EXPECT_EQ(DescriptorUpdater::roundUpVGPR(129, 8), 136u);

  // SGPR granularity is 8
  EXPECT_EQ(DescriptorUpdater::roundUpSGPR(1), 8u);
  EXPECT_EQ(DescriptorUpdater::roundUpSGPR(8), 8u);
  EXPECT_EQ(DescriptorUpdater::roundUpSGPR(9), 16u);
  EXPECT_EQ(DescriptorUpdater::roundUpSGPR(48), 48u);
  EXPECT_EQ(DescriptorUpdater::roundUpSGPR(49), 56u);
}

TEST_F(DescriptorTest, MaxRegisterCounts) {
  EXPECT_EQ(DescriptorUpdater::MAX_VGPRS_GRAN4, 256u);
  EXPECT_EQ(DescriptorUpdater::MAX_VGPRS_GRAN8, 512u);
  EXPECT_EQ(DescriptorUpdater::maxVGPRs(4), 256u);
  EXPECT_EQ(DescriptorUpdater::maxVGPRs(8), 512u);
  EXPECT_EQ(DescriptorUpdater::MAX_SGPRS, 104u);
}

//===----------------------------------------------------------------------===//
// K-001: VGPR count increased
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K001_VGPRCountIncreased) {
  auto Desc = makeDescriptor(/*VGPR=*/32, /*SGPR=*/16);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  EXPECT_EQ(KD.VGPRCount, 32u);

  // Increase VGPR count
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateVGPRCount(KD, 64));
  EXPECT_EQ(KD.VGPRCount, 64u);

  // Verify round-trip
  expectRoundTrip(KD);
}

//===----------------------------------------------------------------------===//
// K-002: SGPR count increased
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K002_SGPRCountIncreased) {
  auto Desc = makeDescriptor(/*VGPR=*/32, /*SGPR=*/16);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  EXPECT_EQ(KD.SGPRCount, 16u);

  // Increase SGPR count
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateSGPRCount(KD, 48));
  EXPECT_EQ(KD.SGPRCount, 48u);

  // Verify round-trip
  expectRoundTrip(KD);
}

//===----------------------------------------------------------------------===//
// K-003: LDS size unchanged when not modified
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K003_LDSSizeUnchanged) {
  auto Desc = makeDescriptor(32, 16, /*LDS=*/4096);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  EXPECT_EQ(KD.GroupSegmentFixedSize, 4096u);

  // Modify VGPRs but not LDS
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateVGPRCount(KD, 64));

  // LDS should be unchanged
  EXPECT_EQ(KD.GroupSegmentFixedSize, 4096u);

  // Verify in serialized form
  auto Bytes = serialize(KD);
  auto KD2OrErr = parse(Bytes);
  ASSERT_TRUE(static_cast<bool>(KD2OrErr));
  EXPECT_EQ(KD2OrErr->GroupSegmentFixedSize, 4096u);
}

//===----------------------------------------------------------------------===//
// K-004: Scratch size increased
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K004_ScratchSizeIncreased) {
  auto Desc = makeDescriptor(32, 16, 0, /*Scratch=*/0);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  EXPECT_EQ(KD.PrivateSegmentFixedSize, 0u);

  // Increase scratch size
  DescriptorUpdater::updateScratchSize(KD, 1024);
  EXPECT_EQ(KD.PrivateSegmentFixedSize, 1024u);

  // Verify round-trip
  expectRoundTrip(KD);
}

//===----------------------------------------------------------------------===//
// K-005: Kernarg size increased
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K005_KernargSizeIncreased) {
  auto Desc = makeDescriptor();  // Default kernarg = 32
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  EXPECT_EQ(KD.KernargSize, 32u);

  // Increase kernarg size
  DescriptorUpdater::updateKernargSize(KD, 128);
  EXPECT_EQ(KD.KernargSize, 128u);

  // Verify round-trip
  expectRoundTrip(KD);
}

//===----------------------------------------------------------------------===//
// K-006: Entry offset correct
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K006_EntryOffsetCorrect) {
  auto Desc = makeDescriptor();  // Default entry offset = 256
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  EXPECT_EQ(KDOrErr->KernelCodeEntryByteOffset, 256u);
}

//===----------------------------------------------------------------------===//
// K-007: Note metadata matches descriptor (placeholder)
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K007_NoteMetadataPlaceholder) {
  // This will be implemented when ELF handling is added.
  // For now, verify descriptor parsing works independently.
  auto Desc = makeDescriptor(128, 32);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  // Placeholder: just verify basic parsing
  EXPECT_EQ(KDOrErr->VGPRCount, 128u);
  EXPECT_EQ(KDOrErr->SGPRCount, 32u);
}

//===----------------------------------------------------------------------===//
// K-007b: ImplicitSGPRs set correctly based on GPU architecture
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K007b_ImplicitSGPRs_GFX950) {
  auto Desc = makeDescriptor(64, 24);
  auto KDOrErr = parse(Desc, "gfx950");
  ASSERT_TRUE(static_cast<bool>(KDOrErr));
  EXPECT_EQ(KDOrErr->ImplicitSGPRs, 6u)
      << "gfx950 has ArchitectedFlatScratch: VCC(2)+FLAT_SCRATCH(2)+XNACK_MASK(2)=6";
}

TEST_F(DescriptorTest, K007b_ImplicitSGPRs_GFX942) {
  auto Desc = makeDescriptor(64, 24);
  auto KDOrErr = parse(Desc, "gfx942");
  ASSERT_TRUE(static_cast<bool>(KDOrErr));
  EXPECT_EQ(KDOrErr->ImplicitSGPRs, 6u)
      << "gfx942 has ArchitectedFlatScratch: VCC(2)+FLAT_SCRATCH(2)+XNACK_MASK(2)=6";
}

TEST_F(DescriptorTest, K007b_ImplicitSGPRs_GFX90a) {
  auto Desc = makeDescriptor(64, 24);
  auto KDOrErr = parse(Desc, "gfx90a");
  ASSERT_TRUE(static_cast<bool>(KDOrErr));
  EXPECT_EQ(KDOrErr->ImplicitSGPRs, 4u)
      << "gfx90a does not have ArchitectedFlatScratch: VCC(2)+FLAT_SCRATCH(2)=4";
}

TEST_F(DescriptorTest, K007b_ImplicitSGPRs_DefaultNoArch) {
  auto Desc = makeDescriptor(64, 24);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));
  EXPECT_EQ(KDOrErr->ImplicitSGPRs, 4u)
      << "Unknown/empty arch should default to 4 implicit SGPRs";
}

//===----------------------------------------------------------------------===//
// K-008: Descriptor round-trip with properties preservation
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K008_DescriptorRoundTrip) {
  // Create descriptor with various non-default values
  // ENABLE_KERNARG_SEGMENT_PTR (bit 3) = 0x0008
  auto Desc = makeDescriptor(/*VGPR=*/96, /*SGPR=*/48, /*LDS=*/8192, /*Scratch=*/512,
                             /*KernelCodeProperties=*/0x0008);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;

  // Verify initial values
  EXPECT_EQ(KD.VGPRCount, 96u);
  EXPECT_EQ(KD.SGPRCount, 48u);
  EXPECT_EQ(KD.GroupSegmentFixedSize, 8192u);
  EXPECT_EQ(KD.PrivateSegmentFixedSize, 512u);
  EXPECT_EQ(KD.KernelCodeProperties, 0x0008u);
  EXPECT_EQ(KD.KernargPreload, 0u);

  // Modify some values
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateVGPRCount(KD, 128));
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateSGPRCount(KD, 64));
  DescriptorUpdater::updateLDSSize(KD, 16384);
  DescriptorUpdater::updateScratchSize(KD, 2048);

  // Verify round-trip preserves all modifications AND properties
  expectRoundTrip(KD);
}

//===----------------------------------------------------------------------===//
// K-009: Properties preserved when updating registers
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, K009_PropertiesPreservedWhenUpdatingRegisters) {
  // Create descriptor with multiple properties enabled
  // Bit 0: ENABLE_PRIVATE_SEGMENT_BUFFER
  // Bit 1: ENABLE_DISPATCH_PTR
  // Bit 2: ENABLE_QUEUE_PTR
  // Bit 3: ENABLE_KERNARG_SEGMENT_PTR
  const uint16_t Properties = 0x000F;
  auto Desc = makeDescriptor(/*VGPR=*/32, /*SGPR=*/16, /*LDS=*/0, /*Scratch=*/0,
                             /*KernelCodeProperties=*/Properties);
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  EXPECT_EQ(KD.KernelCodeProperties, Properties);

  // Update VGPR count (simulate instrumentation adding registers)
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateVGPRCount(KD, 64));
  EXPECT_EQ(KD.VGPRCount, 64u);

  // Update SGPR count
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateSGPRCount(KD, 32));
  EXPECT_EQ(KD.SGPRCount, 32u);

  // Serialize and verify properties are unchanged
  auto Bytes = serialize(KD);
  auto KD2OrErr = parse(Bytes);
  ASSERT_TRUE(static_cast<bool>(KD2OrErr));

  // Verify VGPR/SGPR counts changed
  EXPECT_EQ(KD2OrErr->VGPRCount, 64u);
  EXPECT_EQ(KD2OrErr->SGPRCount, 32u);

  // Verify properties unchanged
  EXPECT_EQ(KD2OrErr->KernelCodeProperties, Properties);
  EXPECT_EQ(KD2OrErr->KernargPreload, 0u);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST_F(DescriptorTest, VGPRCountExceedsMax_Gran4) {
  auto Desc = makeDescriptor();
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  // Default fixture uses granularity 4 → max 256
  AEGIS_EXPECT_ERROR(DescriptorUpdater::updateVGPRCount(KD, 300));
}

TEST_F(DescriptorTest, VGPRCountExceedsMax_Gran8) {
  auto Desc = makeDescriptor();
  auto KDOrErr = parse(Desc, "gfx950");
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  // gfx950 uses granularity 8 → max 512
  AEGIS_EXPECT_NO_ERROR(DescriptorUpdater::updateVGPRCount(KD, 360));
  EXPECT_EQ(KD.VGPRCount, 360u);
  AEGIS_EXPECT_ERROR(DescriptorUpdater::updateVGPRCount(KD, 520));
}

TEST_F(DescriptorTest, SGPRCountExceedsMax) {
  auto Desc = makeDescriptor();
  auto KDOrErr = parse(Desc);
  ASSERT_TRUE(static_cast<bool>(KDOrErr));

  auto& KD = *KDOrErr;
  AEGIS_EXPECT_ERROR(DescriptorUpdater::updateSGPRCount(KD, 120));  // Max is 104
}

TEST_F(DescriptorTest, DescriptorTooSmall) {
  std::vector<uint8_t> SmallDesc(32, 0);  // Only 32 bytes, need 64
  auto KDOrErr = parse(SmallDesc);
  EXPECT_FALSE(static_cast<bool>(KDOrErr)) << "Should fail for descriptor < 64 bytes";
  if (!KDOrErr) {
    llvm::consumeError(KDOrErr.takeError());
  }
}
