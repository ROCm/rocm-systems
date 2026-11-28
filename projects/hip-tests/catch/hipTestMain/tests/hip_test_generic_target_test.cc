/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <catch2/catch_test_macros.hpp>
#include <hip_test_features.hh>

TEST_CASE("GenericTarget_RequiresABuiltCompatibleFamily") {
  const std::string built = "gfx11-generic,gfx9-4-generic:sramecc+:xnack-";
  REQUIRE(isGenericTargetSupported(built, "gfx1100"));
  REQUIRE(isGenericTargetSupported(built, "gfx1101:xnack-"));
  REQUIRE_FALSE(isGenericTargetSupported(built, "gfx1200"));
  REQUIRE_FALSE(isGenericTargetSupported(built, "gfx1201"));
  REQUIRE_FALSE(isGenericTargetSupported(built, "gfx1250"));
  REQUIRE_FALSE(isGenericTargetSupported(built, "unknown"));
  REQUIRE_FALSE(isGenericTargetSupported("", "gfx1100"));
  REQUIRE_FALSE(isGenericTargetSupported("gfx11-generic-extra", "gfx1100"));
  REQUIRE(isGenericTargetSupported("gfx12-generic", "gfx1201"));
}

TEST_CASE("GenericTarget_RequiresTheBuiltFeatureVariant") {
  for (const char* ecc : {"sramecc+", "sramecc-"}) {
    for (const char* xnack : {"xnack+", "xnack-"}) {
      const std::string built = std::string("gfx9-4-generic:") + ecc + ":" + xnack;
      INFO(built);
      for (const char* device_ecc : {"sramecc+", "sramecc-"}) {
        for (const char* device_xnack : {"xnack+", "xnack-"}) {
          const bool compatible =
              std::string(ecc) == device_ecc && std::string(xnack) == device_xnack;
          const std::string device = std::string("gfx942:") + device_ecc + ":" + device_xnack;
          INFO(device);
          REQUIRE(isGenericTargetSupported(built, device.c_str()) == compatible);
          const std::string reversed = std::string("gfx942:") + device_xnack + ":" + device_ecc;
          REQUIRE(isGenericTargetSupported(built, reversed.c_str()) == compatible);
        }
      }
      REQUIRE_FALSE(isGenericTargetSupported(built, "gfx942"));
      REQUIRE_FALSE(isGenericTargetSupported(built, (std::string("gfx942:") + ecc).c_str()));
      REQUIRE_FALSE(isGenericTargetSupported(built, (std::string("gfx942:") + xnack).c_str()));
    }
  }
}

TEST_CASE("GenericTarget_MatchesAnyCompatibleImageWithoutPrefixAliases") {
  const std::string built =
      "gfx11-generic,gfx9-4-generic:sramecc-:xnack+,gfx9-4-generic:sramecc+:xnack-";
  REQUIRE(isGenericTargetSupported(built, "gfx942:sramecc+:xnack-"));
  REQUIRE_FALSE(isGenericTargetSupported(built, "gfx942:sramecc+:xnack+"));
  REQUIRE_FALSE(isGenericTargetSupported(built, "gfx942:sramecc+:xnack-extra"));
  REQUIRE_FALSE(isGenericTargetSupported(built, "gfx942:xsramecc+:xnack-"));
  REQUIRE(isGenericTargetSupported("gfx9-4-generic:xnack-:sramecc+", "gfx942:sramecc+:xnack-"));
  REQUIRE(isGenericTargetSupported("gfx9-4-generic", "gfx942:sramecc+:xnack-"));
  REQUIRE(isGenericTargetSupported("gfx9-4-generic:xnack-", "gfx942:sramecc+:xnack-"));
}
