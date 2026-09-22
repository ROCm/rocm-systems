/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// Unit tests for per-collective config validation: src/config/collconfig.cc (ncclParseCollConfig)
// and its eight call sites in src/collectives.cc (the nccl*Config entry points).
// One case pins a known unfixed defect: it asserts the parser still overruns the destination on an oversized config.
#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"

// The parser itself: only its own return code can separate an accepted config from a rejected one.
#include "config/collconfig.h"

namespace RcclUnitTesting {

namespace {

// Largest CTAPolicy the parser accepts, mirroring the bound collconfig.cc builds from the flags.
constexpr int kCtaPolicyMax = NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO;

// Padding behind the destination config; 0x5a is non-zero so a copy of zeroed source bytes still shows up.
constexpr size_t kGuardBytes = 256;
constexpr unsigned char kGuardFill = 0x5a;

// Destination with a tripwire behind it: in production this field lives in a stack ncclInfo, so an
// unclamped copy corrupts the caller's frame; here it only dirties guard[], which the test reads back.
struct GuardedConfig {
  ncclCollConfig_t config;
  unsigned char guard[kGuardBytes];
};

// Returns true if every guard byte still holds kGuardFill.
bool guardIntact(const GuardedConfig& dst) {
  for (size_t i = 0; i < kGuardBytes; ++i) {
    if (dst.guard[i] != kGuardFill) {
      return false;
    }
  }
  return true;
}

// Fresh destination with the tripwire armed.
GuardedConfig makeGuardedConfig() {
  GuardedConfig dst;
  std::memset(&dst, 0, sizeof(dst));
  std::memset(dst.guard, kGuardFill, kGuardBytes);
  return dst;
}

// A config whose header claims `claimedSize` bytes while the backing storage really is that large.
// Heap-backed on purpose: a stack source of the honest size would itself be the thing overrun.
std::vector<unsigned char> makeOversizedConfig(size_t claimedSize) {
  std::vector<unsigned char> storage(claimedSize, 0x00);
  ncclCollConfig_t seed = NCCL_COLLCONFIG_INITIALIZER;
  seed.size = claimedSize;
  std::memcpy(storage.data(), &seed, sizeof(seed));
  return storage;
}

// One named cast site instead of repeating the reinterpret_cast at every caller.
const ncclCollConfig_t* asConfig(const std::vector<unsigned char>& storage) {
  return reinterpret_cast<const ncclCollConfig_t*>(storage.data());
}

// One config row in the accept/reject tables driven against ncclParseCollConfig.
struct ConfigCase {
  const char* name;
  ncclCollConfig_t config;
};

// Each row starts from NCCL_COLLCONFIG_INITIALIZER and mutates one field, so tables are built at run time.
struct ConfigCaseBuilder {
  std::vector<ConfigCase> cases;

  void add(const char* name, void (*mutate)(ncclCollConfig_t*)) {
    ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
    mutate(&config);
    cases.push_back(ConfigCase{name, config});
  }
};

// Configs the parser must reject.
std::vector<ConfigCase> rejectedConfigs() {
  ConfigCaseBuilder builder;
  // The lower bound is the versioned v23100 tag, not ncclCollConfig_t; they differ once a field is appended.
  builder.add("size_one_below_minimum", [](ncclCollConfig_t* c) { c->size = sizeof(ncclCollConfig_v23100) - 1; });
  builder.add("size_zero", [](ncclCollConfig_t* c) { c->size = 0; });
  builder.add("bad_magic", [](ncclCollConfig_t* c) { c->magic = 0xdeadbeefu; });
  builder.add("force_alg_selection_two", [](ncclCollConfig_t* c) { c->forceAlgSelection = 2; });
  builder.add("force_alg_selection_negative", [](ncclCollConfig_t* c) { c->forceAlgSelection = -1; });
  builder.add("cta_policy_negative", [](ncclCollConfig_t* c) { c->CTAPolicy = -1; });
  builder.add("cta_policy_above_max", [](ncclCollConfig_t* c) { c->CTAPolicy = kCtaPolicyMax + 1; });
  return builder.cases;
}

// Configs the parser must accept.
std::vector<ConfigCase> acceptedConfigs() {
  ConfigCaseBuilder builder;
  // forceAlgSelection = 1 and CTAPolicy = NCCL_CONFIG_UNDEF_INT are already the initializer defaults,
  // so rows setting them would be byte-identical to pristine_initializer and are deliberately absent.
  builder.add("pristine_initializer", [](ncclCollConfig_t*) {});
  builder.add("force_alg_selection_zero", [](ncclCollConfig_t* c) { c->forceAlgSelection = 0; });
  builder.add("cta_policy_default", [](ncclCollConfig_t* c) { c->CTAPolicy = NCCL_CTA_POLICY_DEFAULT; });
  builder.add("cta_policy_efficiency", [](ncclCollConfig_t* c) { c->CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY; });
  builder.add("cta_policy_zero", [](ncclCollConfig_t* c) { c->CTAPolicy = NCCL_CTA_POLICY_ZERO; });
  builder.add("cta_policy_max", [](ncclCollConfig_t* c) { c->CTAPolicy = kCtaPolicyMax; });
  return builder.cases;
}

} // namespace

// NULL user config is accepted; size is zeroed to mark "no user config", which ncclCollConfigNeedAggIsolate
// later reads to skip aggregation isolation.
TEST(CollConfigTests, NullConfig_AcceptedAndMarkedAsNoUserConfig) {
  RUN_ISOLATED_TEST("NullConfig_AcceptedAndMarkedAsNoUserConfig", []() {
    ncclCollConfig_t internal = NCCL_COLLCONFIG_INITIALIZER;
    ASSERT_EQ(ncclParseCollConfig(nullptr, &internal), ncclSuccess);
    ASSERT_EQ(internal.size, 0u) << "NULL config must be marked as no-user-config";
    // The consumer-visible meaning of the zero-size marker: no user config means no aggregation isolation.
    EXPECT_FALSE(ncclCollConfigNeedAggIsolate(&internal));
  });
}

TEST(CollConfigTests, RejectedConfigs_ReturnInvalidArgument) {
  RUN_ISOLATED_TEST("RejectedConfigs_ReturnInvalidArgument", []() {
    for (const ConfigCase& c : rejectedConfigs()) {
      ncclCollConfig_t internal = NCCL_COLLCONFIG_INITIALIZER;
      EXPECT_EQ(ncclParseCollConfig(&c.config, &internal), ncclInvalidArgument) << "case: " << c.name;
    }
  });
}

TEST(CollConfigTests, AcceptedConfigs_ReturnSuccess) {
  RUN_ISOLATED_TEST("AcceptedConfigs_ReturnSuccess", []() {
    for (const ConfigCase& c : acceptedConfigs()) {
      ncclCollConfig_t internal = NCCL_COLLCONFIG_INITIALIZER;
      EXPECT_EQ(ncclParseCollConfig(&c.config, &internal), ncclSuccess) << "case: " << c.name;
    }
  });
}

// The clamp is not in ncclParseCollConfig yet, so an oversized config still writes past the destination.
// This pins that: the guard must be dirty today, and any fix, clamp or reject, turns the test red.
TEST(CollConfigTests, OversizedSize_OverrunsDestination_PinsKnownDefect) {
  RUN_ISOLATED_TEST("OversizedSize_OverrunsDestination", []() {
    // One byte past the end: the copy dirties guard[0] rather than crashing the child.
    std::vector<unsigned char> storage = makeOversizedConfig(sizeof(ncclCollConfig_t) + 1);
    GuardedConfig dst = makeGuardedConfig();
    ASSERT_EQ(ncclParseCollConfig(asConfig(storage), &dst.config), ncclSuccess);
    EXPECT_FALSE(guardIntact(dst))
        << "ncclParseCollConfig no longer overruns the destination; the clamp landed, so invert this pin";
  });
}

// The safe boundary: size == sizeof(ncclCollConfig_t) copies the whole struct and touches nothing beyond it.
TEST(CollConfigTests, ExactSize_CopiesWholeStructWithoutOverrun) {
  RUN_ISOLATED_TEST("ExactSize_CopiesWholeStructWithoutOverrun", []() {
    ncclCollConfig_t source = NCCL_COLLCONFIG_INITIALIZER;
    source.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    source.forceAlgSelection = 0;
    source.userProfilerTag = 0x0123456789abcdefULL;
    GuardedConfig dst = makeGuardedConfig();
    ASSERT_EQ(ncclParseCollConfig(&source, &dst.config), ncclSuccess);
    EXPECT_TRUE(guardIntact(dst)) << "exact-size copy must not write past the struct";
    // userProfilerTag is the last member, so seeing it proves the copy length reached the end.
    EXPECT_EQ(dst.config.userProfilerTag, source.userProfilerTag) << "trailing member must be copied";
    EXPECT_EQ(dst.config.CTAPolicy, NCCL_CTA_POLICY_ZERO);
    EXPECT_EQ(dst.config.forceAlgSelection, 0);
  });
}

// A newer version takes the else branch, which copies internal_config->size bytes instead of config->size.
// The source must be oversized for that to be observable: at the exact size both branches copy the same count.
TEST(CollConfigTests, NewerVersion_CopiesLibraryStructSizeWithoutOverrun) {
  RUN_ISOLATED_TEST("NewerVersion_CopiesLibraryStructSizeWithoutOverrun", []() {
    std::vector<unsigned char> storage = makeOversizedConfig(sizeof(ncclCollConfig_t) + kGuardBytes - 1);
    ncclCollConfig_t header;
    std::memcpy(&header, storage.data(), sizeof(header));
    header.version = NCCL_VERSION_CODE + 1;
    header.CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY;
    header.userProfilerTag = 0x0f1e2d3c4b5a6978ULL;
    std::memcpy(storage.data(), &header, sizeof(header));

    GuardedConfig dst = makeGuardedConfig();
    ASSERT_EQ(ncclParseCollConfig(asConfig(storage), &dst.config), ncclSuccess);
    EXPECT_TRUE(guardIntact(dst)) << "newer-version copy must stop at the library struct size";
    // userProfilerTag is the last member, so seeing it proves the copy length reached the end.
    EXPECT_EQ(dst.config.userProfilerTag, header.userProfilerTag) << "trailing member must be copied";
    EXPECT_EQ(dst.config.CTAPolicy, NCCL_CTA_POLICY_EFFICIENCY);
    EXPECT_EQ(dst.config.version, header.version);
  });
}

// Magic is validated before the copy, so a bad-magic config with an oversized size is never copied at all.
TEST(CollConfigTests, BadMagicWithOversizedSize_RejectedBeforeCopy) {
  RUN_ISOLATED_TEST("BadMagicWithOversizedSize_RejectedBeforeCopy", []() {
    std::vector<unsigned char> storage = makeOversizedConfig(sizeof(ncclCollConfig_t) + kGuardBytes - 1);
    ncclCollConfig_t header;
    std::memcpy(&header, storage.data(), sizeof(header));
    header.magic = 0xdeadbeefu;
    std::memcpy(storage.data(), &header, sizeof(header));

    GuardedConfig dst = makeGuardedConfig();
    EXPECT_EQ(ncclParseCollConfig(asConfig(storage), &dst.config), ncclInvalidArgument);
    EXPECT_TRUE(guardIntact(dst)) << "rejected config must not have been copied";
  });
}

// The eight nccl*Config entry points. Validation runs before ncclEnqueueCheck, so comm and buffers can be null.
// With a null comm every config, good or bad, yields ncclInvalidArgument from CommCheck, so the return code here
// cannot distinguish validation outcomes; that is pinned directly against ncclParseCollConfig above. These rows
// pin only that each overload links, reaches the config path, and errors out instead of dereferencing a null comm.

namespace {

// One row per entry point; the undefined primary template makes a missing specialization a compile error.
template <typename Tag>
struct CollApiTraits;

struct AllReduceTag {};
struct BroadcastTag {};
struct ReduceTag {};
struct AllGatherTag {};
struct ReduceScatterTag {};
struct AlltoAllTag {};
struct GatherTag {};
struct ScatterTag {};

// Comm, buffers and stream are null on purpose; count is non-zero so a short-circuit on count == 0 cannot hide a row.
constexpr size_t kCount = 16;

template <>
struct CollApiTraits<AllReduceTag> {
  static constexpr const char* kName = "ncclAllReduceConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclAllReduceConfig(nullptr, nullptr, kCount, ncclFloat, ncclSum, nullptr, nullptr, config);
  }
};

template <>
struct CollApiTraits<BroadcastTag> {
  static constexpr const char* kName = "ncclBroadcastConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclBroadcastConfig(nullptr, nullptr, kCount, ncclFloat, 0, nullptr, nullptr, config);
  }
};

template <>
struct CollApiTraits<ReduceTag> {
  static constexpr const char* kName = "ncclReduceConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclReduceConfig(nullptr, nullptr, kCount, ncclFloat, ncclSum, 0, nullptr, nullptr, config);
  }
};

template <>
struct CollApiTraits<AllGatherTag> {
  static constexpr const char* kName = "ncclAllGatherConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclAllGatherConfig(nullptr, nullptr, kCount, ncclFloat, nullptr, nullptr, config);
  }
};

template <>
struct CollApiTraits<ReduceScatterTag> {
  static constexpr const char* kName = "ncclReduceScatterConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclReduceScatterConfig(nullptr, nullptr, kCount, ncclFloat, ncclSum, nullptr, nullptr, config);
  }
};

template <>
struct CollApiTraits<AlltoAllTag> {
  static constexpr const char* kName = "ncclAlltoAllConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclAlltoAllConfig(nullptr, nullptr, kCount, ncclFloat, nullptr, nullptr, config);
  }
};

template <>
struct CollApiTraits<GatherTag> {
  static constexpr const char* kName = "ncclGatherConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclGatherConfig(nullptr, nullptr, kCount, ncclFloat, 0, nullptr, nullptr, config);
  }
};

template <>
struct CollApiTraits<ScatterTag> {
  static constexpr const char* kName = "ncclScatterConfig";
  static ncclResult_t Call(const ncclCollConfig_t* config) {
    return ncclScatterConfig(nullptr, nullptr, kCount, ncclFloat, 0, nullptr, nullptr, config);
  }
};

} // namespace

template <typename Tag>
class CollConfigTypedTests : public ::testing::Test {};

using CollConfigApis = ::testing::Types<AllReduceTag, BroadcastTag, ReduceTag, AllGatherTag,
                                        ReduceScatterTag, AlltoAllTag, GatherTag, ScatterTag>;
TYPED_TEST_SUITE(CollConfigTypedTests, CollConfigApis);

// One representative malformed config per API; the full rejection table is exercised against the parser directly.
TYPED_TEST(CollConfigTypedTests, MalformedConfig_ReturnsInvalidArgument) {
  using Traits = CollApiTraits<TypeParam>;
  RUN_ISOLATED_TEST(std::string("MalformedConfig_") + Traits::kName, []() {
    ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
    config.magic = 0xdeadbeefu;
    EXPECT_EQ(Traits::Call(&config), ncclInvalidArgument) << "api: " << Traits::kName;
  });
}

// A NULL config is documented as equivalent to the plain API, so it must pass validation and reach the comm check.
TYPED_TEST(CollConfigTypedTests, NullConfig_ReachesCommunicatorCheck) {
  using Traits = CollApiTraits<TypeParam>;
  RUN_ISOLATED_TEST(std::string("NullConfig_") + Traits::kName, []() {
    EXPECT_EQ(Traits::Call(nullptr), ncclInvalidArgument) << "api: " << Traits::kName;
  });
}

} // namespace RcclUnitTesting
