/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Standalone unit tests for doorbell type validation logic.
 * Can be compiled and run without ROCm/HSA runtime installed.
 *
 * The rocrtst CMake build compiles this file into a dedicated CPU-only target
 * and deliberately excludes it from the legacy rocrtst64 aggregate. No main()
 * is defined here because the target links the bundled gtest_main library.
 *
 * For a standalone hardware-free build, link against the bundled gtest_main
 * (the `gtest-all.cpp` amalgamation in this tree is stale — it #includes .cc
 * files that don't exist here, so list the individual .cpp sources instead,
 * mirroring rocrtst/gtest/CMakeLists.txt).
 *
 * From projects/rocr-runtime/rocrtst/suites/functional/:
 *
 * Build:  g++ -std=c++17 -I../../gtest/include -I../../gtest \
 *           -I../../../runtime/hsa-runtime \
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
#include "core/util/doorbell_type.h"

using rocr::AMD::ExtractDoorbellType;
using rocr::AMD::IsDoorbellTypeSupported;
using rocr::AMD::kDoorbellType1_0;
using rocr::AMD::kDoorbellType2_0;
using rocr::AMD::kDoorbellTypePre1_0;
using rocr::AMD::kDoorbellTypeReserved;
using rocr::AMD::MakeCapabilityWithDoorbell;

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

