/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Standalone unit tests for doorbell type validation logic.
 * Can be compiled and run without ROCm/HSA runtime installed.
 *
 * When built as part of rocrtst, these TESTs are auto-registered with gtest
 * and run from the rocrtst64 binary's main(). No main() is defined here so
 * that this file can be linked into rocrtst64 without a duplicate-symbol
 * clash against suites/test_common/main.cc.
 *
 * For a standalone hardware-free build, link against the bundled gtest_main
 * (the `gtest-all.cpp` amalgamation in this tree is stale — it #includes .cc
 * files that don't exist here, so list the individual .cpp sources instead,
 * mirroring rocrtst/gtest/CMakeLists.txt).
 *
 * From projects/rocr-runtime/rocrtst/suites/functional/:
 *
 * Build:  g++ -std=c++17 -I../../gtest/include -I../../gtest \
 *           -o doorbell_type_test doorbell_type_test.cc \
 *           ../../gtest/src/gtest.cpp \
 *           ../../gtest/src/gtest-port.cpp \
 *           ../../gtest/src/gtest-printers.cpp \
 *           ../../gtest/src/gtest-filepath.cpp \
 *           ../../gtest/src/gtest-test-part.cpp \
 *           ../../gtest/src/gtest-typed-test.cpp \
 *           ../../gtest/src/gtest-death-test.cpp \
 *           ../../gtest/src/gtest_main.cpp \
 *           -lpthread
 *
 * Run:    ./doorbell_type_test
 */

#include <cstdint>
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"

// Doorbell type values from kfd_sysfs.h
static const unsigned int kDoorbellTypePre1_0 = 0;
static const unsigned int kDoorbellType1_0    = 1;
static const unsigned int kDoorbellType2_0    = 2;
static const unsigned int kDoorbellTypeReserved = 3;

// Extract DoorbellType (bits 12-13) from capability field.
// The field is currently 2 bits wide, so valid values are 0-3.
// If the field is widened in future kernels, this mask must be updated.
static unsigned int ExtractDoorbellType(uint32_t capability) {
  return (capability >> 12) & 0x3;
}

// Build a synthetic capability value with a given doorbell type.
// DoorbellType occupies bits 12-13 of the capability field.
// Values larger than 3 are masked to 2 bits (current field width).
static uint32_t MakeCapabilityWithDoorbell(unsigned int doorbell_type) {
  return (doorbell_type & 0x3) << 12;
}

// Check whether a doorbell type is supported by the HSA runtime.
// Only DoorbellType 2 (HSA_CAP_DOORBELL_TYPE_2_0, Vega+) is supported.
// This mirrors the logic in amd_gpu_agent.cpp — if the supported set changes
// there, it must change here too.
static bool IsDoorbellTypeSupported(unsigned int doorbell_type) {
  return doorbell_type == kDoorbellType2_0;
}

// --- Known doorbell types: verify correct classification ---

TEST(DoorbellTypeValidation, Supported_Type2_Vega_And_Newer) {
  EXPECT_TRUE(IsDoorbellTypeSupported(kDoorbellType2_0));
}

TEST(DoorbellTypeValidation, Deprecated_Type0_Pre1_Kaveri_Hawaii_Tonga) {
  EXPECT_FALSE(IsDoorbellTypeSupported(kDoorbellTypePre1_0));
}

TEST(DoorbellTypeValidation, Deprecated_Type1_Polaris_Fiji_Vegam) {
  // This is the exact case that triggered the original bug — a WX 2100
  // (gfx803/Polaris) with DoorbellType=1 killed HSA init for MI50 + W5700.
  EXPECT_FALSE(IsDoorbellTypeSupported(kDoorbellType1_0));
}

TEST(DoorbellTypeValidation, Reserved_Type3_Must_Be_Rejected) {
  EXPECT_FALSE(IsDoorbellTypeSupported(kDoorbellTypeReserved));
}

// --- Exhaustive coverage of all current field values ---

TEST(DoorbellTypeValidation, ExactlyOneTypeIsSupported_OutOfFourPossible) {
  int supported = 0;
  for (unsigned int dt = 0; dt <= 3; ++dt) {
    if (IsDoorbellTypeSupported(dt)) supported++;
  }
  EXPECT_EQ(supported, 1)
      << "Expected exactly 1 supported doorbell type out of 4 possible values. "
         "If a new type was added, update the switch in amd_gpu_agent.cpp and "
         "IsDoorbellTypeSupported in this test.";
}

// --- Bit extraction: ensure DoorbellType is correctly isolated ---

TEST(DoorbellTypeValidation, BitExtraction_EachDoorbellValue) {
  EXPECT_EQ(ExtractDoorbellType(0x00000000), 0u);  // bits 12-13 = 00
  EXPECT_EQ(ExtractDoorbellType(0x00001000), 1u);  // bits 12-13 = 01
  EXPECT_EQ(ExtractDoorbellType(0x00002000), 2u);  // bits 12-13 = 10
  EXPECT_EQ(ExtractDoorbellType(0x00003000), 3u);  // bits 12-13 = 11
}

TEST(DoorbellTypeValidation, BitExtraction_IgnoresAdjacentFields) {
  // Bits outside 12-13 should not affect the extracted doorbell type.
  EXPECT_EQ(ExtractDoorbellType(0xFFFF0FFF), 0u);
  EXPECT_EQ(ExtractDoorbellType(0xFFFF2FFF), 2u);
}

// --- Real hardware capability values (regression data) ---

TEST(DoorbellTypeValidation, RealHardware_MI50_gfx906_Supported) {
  // Instinct MI50 (gfx906, Vega20): capability=0xac73a280
  EXPECT_EQ(ExtractDoorbellType(0xac73a280), kDoorbellType2_0);
  EXPECT_TRUE(IsDoorbellTypeSupported(ExtractDoorbellType(0xac73a280)));
}

TEST(DoorbellTypeValidation, RealHardware_WX2100_gfx803_Deprecated) {
  // Radeon Pro WX 2100 (gfx803, Polaris12): capability=0x00001280
  // This is the card that triggered the original bug.
  EXPECT_EQ(ExtractDoorbellType(0x00001280), kDoorbellType1_0);
  EXPECT_FALSE(IsDoorbellTypeSupported(ExtractDoorbellType(0x00001280)));
}

TEST(DoorbellTypeValidation, RealHardware_W5700_gfx1010_Supported) {
  // Radeon Pro W5700 (gfx1010, Navi10): capability=0x2883a280
  EXPECT_EQ(ExtractDoorbellType(0x2883a280), kDoorbellType2_0);
  EXPECT_TRUE(IsDoorbellTypeSupported(ExtractDoorbellType(0x2883a280)));
}

// --- Future-proofing: what happens when the field width changes ---

TEST(DoorbellTypeValidation, FutureProofing_CurrentFieldIsTwoBits) {
  EXPECT_EQ(ExtractDoorbellType(0xFFFFFFFF), 3u)
      << "DoorbellType extracted a value > 3. Has the field been widened "
         "beyond 2 bits? Update ExtractDoorbellType mask and this test.";
}

TEST(DoorbellTypeValidation, FutureProofing_SyntheticValues_MaskedToFieldWidth) {
  // 666 & 3 = 2 (happens to map to supported type)
  uint32_t cap = MakeCapabilityWithDoorbell(666);
  unsigned int extracted = ExtractDoorbellType(cap);
  EXPECT_EQ(extracted, 666u & 0x3)
      << "Synthetic value 666 should be masked to " << (666u & 0x3)
      << " by the 2-bit field. If the field is now wider, update the mask.";

  // 0xDEAD & 3 = 1 (maps to deprecated Polaris type)
  cap = MakeCapabilityWithDoorbell(0xDEAD);
  extracted = ExtractDoorbellType(cap);
  EXPECT_EQ(extracted, 0xDEADu & 0x3);
  EXPECT_FALSE(IsDoorbellTypeSupported(extracted))
      << "0xDEAD masked to " << extracted << " should map to a deprecated type";

  // 0xFFFF & 3 = 3 (maps to reserved type)
  cap = MakeCapabilityWithDoorbell(0xFFFF);
  extracted = ExtractDoorbellType(cap);
  EXPECT_EQ(extracted, 3u);
  EXPECT_FALSE(IsDoorbellTypeSupported(extracted))
      << "0xFFFF masked to 3 should map to reserved/unsupported type";
}

TEST(DoorbellTypeValidation, FutureProofing_MakeAndExtract_RoundTripsForAllValidValues) {
  for (unsigned int dt = 0; dt <= 3; ++dt) {
    uint32_t cap = MakeCapabilityWithDoorbell(dt);
    EXPECT_EQ(ExtractDoorbellType(cap), dt)
        << "Round-trip failed for DoorbellType " << dt;
  }
}

