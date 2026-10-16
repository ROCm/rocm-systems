// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "embedded_schema.h"
#include "rj_arith/rj_arith.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"
#include "scoped_temp.h"
#include "test_paths.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace rocjitsu {
namespace {

std::string arithmetic_config(const std::string &policy) {
  auto json = config::read_config_file(test::config_path("gfx1201_r9700.json"));
  if (!policy.empty())
    json.insert(json.find('{') + 1, "\"arithmetic\": " + policy + ",");
  return json;
}

TEST(ArithConfigTest, DefaultsAndTopLevelPolicyReachExecution) {
  auto defaults = config::load_config_from_string(arithmetic_config(""), kEmbeddedSchema);
  auto *cu = defaults.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  EXPECT_EQ(cu->config().arithmetic.semantics, amdgpu::ArithmeticSemantics::Compatibility);
  EXPECT_EQ(cu->arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_COMPATIBILITY);
  auto loose = config::load_config_from_string(arithmetic_config(R"({"semantics":"loose"})"),
                                               kEmbeddedSchema);
  cu = loose.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  EXPECT_EQ(cu->arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_LOOSE);
}

TEST(ArithConfigTest, EveryModeAndEmptyObjectReachExecution) {
  for (const auto *name :
       {"loose", "bit_exact_empirical_allow_loose", "bit_exact_empirical_no_fp_status",
        "bit_exact_empirical", "compatibility"}) {
    const std::string policy = std::string("{\"semantics\":\"") + name + "\"}";
    auto loaded = config::load_config_from_string(arithmetic_config(policy), kEmbeddedSchema);
    auto *cu = loaded.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
    EXPECT_EQ(cu->arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK,
              (static_cast<uint64_t>(amdgpu::parse_arithmetic_policy(name).semantics) << 16));
  }
  auto loaded = config::load_config_from_string(arithmetic_config("{}"), kEmbeddedSchema);
  EXPECT_EQ(loaded.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags() &
                RJ_ARITH_SEMANTICS_MASK,
            RJ_ARITH_SEMANTICS_COMPATIBILITY);
}

TEST(ArithConfigTest, RejectsInvalidAndRemovedSemanticsNames) {
  for (const char *name :
       {"invalid", "exact_empirical", "bit_exact_empirical_fp_status", "4", "7"}) {
    const std::string policy = std::string("{\"semantics\":\"") + name + "\"}";
    EXPECT_THROW(config::load_config_from_string(arithmetic_config(policy), kEmbeddedSchema),
                 std::invalid_argument);
  }
}

TEST(ArithConfigTest, TopLevelOverridesPerCuButDoesNotHideInvalidPolicies) {
  auto json = arithmetic_config("");
  auto cu = json.find("\"type\": \"compute_unit\"");
  ASSERT_NE(cu, std::string::npos);
  auto entries = json.find('[', json.find("\"config\"", cu));
  ASSERT_NE(entries, std::string::npos);
  json.insert(entries + 1, R"({"key":"arithmetic_semantics","value":"loose"},)");
  auto per_cu = config::load_config_from_string(json, kEmbeddedSchema);
  EXPECT_EQ(per_cu.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags() &
                RJ_ARITH_SEMANTICS_MASK,
            RJ_ARITH_SEMANTICS_LOOSE);
  json.insert(json.find('{') + 1,
              R"("arithmetic":{"semantics":"bit_exact_empirical_no_fp_status"},)");
  auto overridden = config::load_config_from_string(json, kEmbeddedSchema);
  auto flags = overridden.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags();
  EXPECT_EQ(flags & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  auto value = json.find("\"value\":\"loose\"");
  ASSERT_NE(value, std::string::npos);
  json.replace(value, std::string("\"value\":\"loose\"").size(), "\"value\":\"invalid\"");
  EXPECT_THROW(config::load_config_from_string(json, kEmbeddedSchema), std::invalid_argument);
}

TEST(ArithConfigTest, RemovedPerCuControlsAreRejectedEvenWithTopLevelOverride) {
  for (auto key : {"arithmetic_fallback", "arithmetic_state_reporting"}) {
    auto json = arithmetic_config(R"({"semantics":"loose"})");
    const auto cu = json.find("\"type\": \"compute_unit\"");
    ASSERT_NE(cu, std::string::npos);
    const auto entries = json.find('[', json.find("\"config\"", cu));
    ASSERT_NE(entries, std::string::npos);
    json.insert(entries + 1, std::string("{\"key\":\"") + key + "\",\"value\":\"strict\"},");
    EXPECT_THROW(config::load_config_from_string(json, kEmbeddedSchema), std::invalid_argument);
  }
}

TEST(ArithConfigTest, CheckpointPreservesPerCuPolicyAndCachedFlags) {
  auto loaded = config::load_config_from_string(arithmetic_config(""), kEmbeddedSchema);
  auto *se = loaded.soc()->xcd(0)->shader_engine(0);
  ASSERT_GE(se->num_compute_units(), 2u);
  se->compute_unit(0)->restore_arithmetic_policy(
      amdgpu::parse_arithmetic_policy("bit_exact_empirical"));
  se->compute_unit(1)->restore_arithmetic_policy(amdgpu::parse_arithmetic_policy("loose"));
  test::ScopedTempFile file("rocjitsu-rj_arith-policy-");
  config::save_checkpoint(file.path(), *loaded.soc(), 0, loaded.engine_config,
                          loaded.cpu_dispatch_threads);
  auto restored = config::restore_checkpoint(file.path());
  auto *restored_se = restored.soc()->xcd(0)->shader_engine(0);
  for (unsigned i = 0; i < 2; ++i) {
    auto *before = se->compute_unit(i);
    auto *after = restored_se->compute_unit(i);
    EXPECT_EQ(after->config().arithmetic.semantics, before->config().arithmetic.semantics);
    EXPECT_EQ(after->arithmetic_flags(), before->arithmetic_flags());
  }
}

TEST(ArithConfigTest, MissingCheckpointPolicyUsesDefaultSemantics) {
  EXPECT_EQ(config::arithmetic_from_fb(nullptr).semantics,
            amdgpu::ArithmeticSemantics::Compatibility);
}

} // namespace
} // namespace rocjitsu
