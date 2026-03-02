// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "isa_target_match.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

// Helper: collect all compatible targets into a vector.
std::vector<std::string> expand(const char* isa) {
  std::vector<std::string> result;
  kpack::for_each_compatible_target(isa, [&](const std::string& t) {
    result.push_back(t);
    return false;  // continue
  });
  return result;
}

using Vec = std::vector<std::string>;

// --- strip_target_prefix ---

TEST(StripTargetPrefix, StripsAmdgcnPrefix) {
  EXPECT_EQ(kpack::strip_target_prefix("amdgcn-amd-amdhsa--gfx942"),
            "gfx942");
}

TEST(StripTargetPrefix, StripsAmdgcnPrefixWithFeatures) {
  EXPECT_EQ(
      kpack::strip_target_prefix("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"),
      "gfx942:sramecc+:xnack-");
}

TEST(StripTargetPrefix, NoOpForBareArch) {
  EXPECT_EQ(kpack::strip_target_prefix("gfx1201"), "gfx1201");
}

TEST(StripTargetPrefix, NoOpForArchWithFeatures) {
  EXPECT_EQ(kpack::strip_target_prefix("gfx942:xnack+"), "gfx942:xnack+");
}

TEST(StripTargetPrefix, EmptyString) {
  EXPECT_EQ(kpack::strip_target_prefix(""), "");
}

TEST(StripTargetPrefix, Nullptr) {
  EXPECT_EQ(kpack::strip_target_prefix(nullptr), "");
}

// --- parse_target_id ---

TEST(ParseTargetId, BareProcessor) {
  auto p = kpack::parse_target_id("gfx942");
  EXPECT_EQ(p.processor, "gfx942");
  EXPECT_TRUE(p.features.empty());
}

TEST(ParseTargetId, TwoFeatures) {
  auto p = kpack::parse_target_id("gfx942:sramecc+:xnack-");
  EXPECT_EQ(p.processor, "gfx942");
  EXPECT_EQ(p.features, (Vec{"sramecc+", "xnack-"}));
}

TEST(ParseTargetId, SingleFeature) {
  auto p = kpack::parse_target_id("gfx942:xnack+");
  EXPECT_EQ(p.processor, "gfx942");
  EXPECT_EQ(p.features, (Vec{"xnack+"}));
}

TEST(ParseTargetId, GenericTarget) {
  auto p = kpack::parse_target_id("gfx9-4-generic");
  EXPECT_EQ(p.processor, "gfx9-4-generic");
  EXPECT_TRUE(p.features.empty());
}

TEST(ParseTargetId, EmptyString) {
  auto p = kpack::parse_target_id("");
  EXPECT_TRUE(p.processor.empty());
  EXPECT_TRUE(p.features.empty());
}

// --- for_each_compatible_target: consumer cards (no features) ---

TEST(IsaTargetMatch, ConsumerCardBare) {
  EXPECT_EQ(expand("gfx1201"), (Vec{"gfx1201"}));
}

TEST(IsaTargetMatch, ConsumerCardWithPrefix) {
  EXPECT_EQ(expand("amdgcn-amd-amdhsa--gfx1201"), (Vec{"gfx1201"}));
}

TEST(IsaTargetMatch, ConsumerCardGfx1100) {
  EXPECT_EQ(expand("gfx1100"), (Vec{"gfx1100"}));
}

// --- for_each_compatible_target: datacenter with two features ---

TEST(IsaTargetMatch, TwoFeaturesMostSpecificFirst) {
  EXPECT_EQ(expand("gfx942:sramecc+:xnack-"),
            (Vec{"gfx942:sramecc+:xnack-", "gfx942:sramecc+",
                 "gfx942:xnack-", "gfx942"}));
}

TEST(IsaTargetMatch, TwoFeaturesWithPrefix) {
  EXPECT_EQ(expand("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"),
            (Vec{"gfx942:sramecc+:xnack-", "gfx942:sramecc+",
                 "gfx942:xnack-", "gfx942"}));
}

TEST(IsaTargetMatch, AsanAgent) {
  // ASAN requires xnack+, hardware also has sramecc+
  EXPECT_EQ(expand("gfx942:sramecc+:xnack+"),
            (Vec{"gfx942:sramecc+:xnack+", "gfx942:sramecc+",
                 "gfx942:xnack+", "gfx942"}));
}

TEST(IsaTargetMatch, MixedFeatureValues) {
  EXPECT_EQ(expand("gfx942:sramecc-:xnack+"),
            (Vec{"gfx942:sramecc-:xnack+", "gfx942:sramecc-",
                 "gfx942:xnack+", "gfx942"}));
}

// --- for_each_compatible_target: single feature ---

TEST(IsaTargetMatch, SingleFeatureXnack) {
  EXPECT_EQ(expand("gfx942:xnack+"), (Vec{"gfx942:xnack+", "gfx942"}));
}

TEST(IsaTargetMatch, SingleFeatureSramecc) {
  EXPECT_EQ(expand("gfx942:sramecc+"), (Vec{"gfx942:sramecc+", "gfx942"}));
}

// --- for_each_compatible_target: generic ISA ---

TEST(IsaTargetMatch, GenericIsa) {
  EXPECT_EQ(expand("gfx9-4-generic"), (Vec{"gfx9-4-generic"}));
}

TEST(IsaTargetMatch, GenericIsaWithPrefix) {
  EXPECT_EQ(expand("amdgcn-amd-amdhsa--gfx9-4-generic"),
            (Vec{"gfx9-4-generic"}));
}

// --- for_each_compatible_target: gfx950 ---

TEST(IsaTargetMatch, Gfx950TwoFeatures) {
  EXPECT_EQ(expand("gfx950:sramecc+:xnack-"),
            (Vec{"gfx950:sramecc+:xnack-", "gfx950:sramecc+",
                 "gfx950:xnack-", "gfx950"}));
}

// --- for_each_compatible_target: edge cases ---

TEST(IsaTargetMatch, EmptyString) {
  EXPECT_EQ(expand(""), Vec{});
}

TEST(IsaTargetMatch, Nullptr) {
  EXPECT_EQ(expand(nullptr), Vec{});
}

// --- for_each_compatible_target: early termination ---

TEST(IsaTargetMatch, EarlyTerminationOnFirstMatch) {
  // Callback returns true on first call — should stop immediately
  std::vector<std::string> seen;
  bool found = kpack::for_each_compatible_target(
      "gfx942:sramecc+:xnack-", [&](const std::string& t) {
        seen.push_back(t);
        return true;  // match found — stop
      });

  EXPECT_TRUE(found);
  EXPECT_EQ(seen.size(), 1u);
  EXPECT_EQ(seen[0], "gfx942:sramecc+:xnack-");
}

TEST(IsaTargetMatch, EarlyTerminationOnThirdMatch) {
  // Simulate: archive exists for bare gfx942 (release build)
  // Callback should be called 4 times, returning true on the last
  std::vector<std::string> seen;
  bool found = kpack::for_each_compatible_target(
      "gfx942:sramecc+:xnack-", [&](const std::string& t) {
        seen.push_back(t);
        return t == "gfx942:xnack-";  // match on third candidate
      });

  EXPECT_TRUE(found);
  EXPECT_EQ(seen.size(), 3u);
  EXPECT_EQ(seen[0], "gfx942:sramecc+:xnack-");
  EXPECT_EQ(seen[1], "gfx942:sramecc+");
  EXPECT_EQ(seen[2], "gfx942:xnack-");
}

TEST(IsaTargetMatch, NoMatchReturnsAllCandidates) {
  // No match — all candidates should be visited
  std::vector<std::string> seen;
  bool found = kpack::for_each_compatible_target(
      "gfx942:sramecc+:xnack-", [&](const std::string& t) {
        seen.push_back(t);
        return false;  // no match
      });

  EXPECT_FALSE(found);
  EXPECT_EQ(seen.size(), 4u);
}

TEST(IsaTargetMatch, ReturnsFalseForEmptyInput) {
  bool found = kpack::for_each_compatible_target(
      "", [](const std::string&) { return true; });
  EXPECT_FALSE(found);
}

TEST(IsaTargetMatch, ConsumerCardEarlyTermination) {
  // Consumer card — single candidate, callback returns true
  bool found = kpack::for_each_compatible_target(
      "gfx1201", [](const std::string&) { return true; });
  EXPECT_TRUE(found);
}

}  // namespace
