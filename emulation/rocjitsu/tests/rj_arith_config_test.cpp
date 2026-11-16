// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "checkpoint_generated.h"
#include "config_test_util.h"
#include "embedded_schema.h"
#include "rj_arith/rj_arith.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/virtual_machine.h"
#include "scoped_temp.h"
#include "test_paths.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

std::string arithmetic_config(const std::string &policy) {
  util::StringDiagnostic diagnostic;
  auto result =
      config::read_config_file(test::config_path("gfx1201_r9700.json"), diagnostic.emitter());
  if (result.failed())
    throw std::runtime_error(diagnostic.message());
  auto json = std::move(result).value();
  if (!policy.empty())
    json.insert(json.find('{') + 1, "\"arithmetic\": " + policy + ",");
  return json;
}

TEST(ArithConfigTest, DefaultsAndTopLevelPolicyReachExecution) {
  config::LoadedConfig defaults;
  ASSERT_TRUE(test::load_valid_config_from_string(arithmetic_config(""), defaults));
  auto *cu = defaults.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  EXPECT_EQ(cu->config().arithmetic.semantics, amdgpu::ArithmeticSemantics::Compatibility);
  EXPECT_EQ(cu->arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_COMPATIBILITY);
  config::LoadedConfig loose;
  ASSERT_TRUE(
      test::load_valid_config_from_string(arithmetic_config(R"({"semantics":"loose"})"), loose));
  cu = loose.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  EXPECT_EQ(cu->arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_LOOSE);
}

TEST(ArithConfigTest, EveryModeAndEmptyObjectReachExecution) {
  for (const auto *name :
       {"loose", "bit_exact_empirical_allow_loose", "bit_exact_empirical_no_fp_status",
        "bit_exact_empirical", "compatibility"}) {
    const std::string policy = std::string("{\"semantics\":\"") + name + "\"}";
    config::LoadedConfig loaded;
    ASSERT_TRUE(test::load_valid_config_from_string(arithmetic_config(policy), loaded));
    auto *cu = loaded.soc()->xcd(0)->shader_engine(0)->compute_unit(0);
    EXPECT_EQ(cu->arithmetic_flags() & RJ_ARITH_SEMANTICS_MASK,
              (static_cast<uint64_t>(amdgpu::parse_arithmetic_policy(name).semantics) << 16));
  }
  config::LoadedConfig loaded;
  ASSERT_TRUE(test::load_valid_config_from_string(arithmetic_config("{}"), loaded));
  EXPECT_EQ(loaded.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags() &
                RJ_ARITH_SEMANTICS_MASK,
            RJ_ARITH_SEMANTICS_COMPATIBILITY);
}

TEST(ArithConfigTest, RejectsMisspelledFieldsWithoutRestrictingOtherObjects) {
  util::StringDiagnostic diagnostic;
  auto invalid =
      config::load_config_from_string(arithmetic_config(R"({"semantic":"bit_exact_empirical"})"),
                                      kEmbeddedSchema, diagnostic.emitter());
  EXPECT_TRUE(invalid.failed());
  EXPECT_NE(diagnostic.message().find("semantic"), std::string::npos);
  auto json = arithmetic_config("{}");
  json.insert(json.find('{') + 1, R"("future_unrelated_setting": true,)");
  config::LoadedConfig loaded;
  EXPECT_TRUE(test::load_valid_config_from_string(json, loaded));
  for (bool before_arithmetic : {false, true}) {
    auto empty_key = arithmetic_config("{}");
    auto at = before_arithmetic ? empty_key.find('{') + 1 : empty_key.rfind('}');
    empty_key.insert(at, before_arithmetic ? R"("": 42,)" : R"(,"": 42)");
    config::LoadedConfig with_empty_key;
    EXPECT_TRUE(test::load_valid_config_from_string(empty_key, with_empty_key));
  }
}

TEST(ArithConfigTest, ArithmeticPolicyAcceptsUtf8Bom) {
  config::LoadedConfig loaded;
  ASSERT_TRUE(test::load_valid_config_from_string(
      "\xef\xbb\xbf" + arithmetic_config(R"({"semantics":"loose"})"), loaded));
  EXPECT_EQ(loaded.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags() &
                RJ_ARITH_SEMANTICS_MASK,
            RJ_ARITH_SEMANTICS_LOOSE);
}

TEST(ArithConfigTest, RejectsInvalidAndRemovedSemanticsNames) {
  for (const char *name :
       {"invalid", "exact_empirical", "bit_exact_empirical_fp_status", "4", "7"}) {
    const std::string policy = std::string("{\"semantics\":\"") + name + "\"}";
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(config::load_config_from_string(arithmetic_config(policy), kEmbeddedSchema,
                                                diagnostic.emitter())
                    .failed());
    EXPECT_NE(diagnostic.message().find("arithmetic.semantics"), std::string::npos);
  }
}

TEST(ArithConfigTest, TopLevelOverridesPerCuButDoesNotHideInvalidPolicies) {
  auto json = arithmetic_config("");
  auto cu = json.find("\"type\": \"compute_unit\"");
  ASSERT_NE(cu, std::string::npos);
  auto entries = json.find('[', json.find("\"config\"", cu));
  ASSERT_NE(entries, std::string::npos);
  json.insert(entries + 1, R"({"key":"arithmetic_semantics","value":"loose"},)");
  config::LoadedConfig per_cu;
  ASSERT_TRUE(test::load_valid_config_from_string(json, per_cu));
  EXPECT_EQ(per_cu.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags() &
                RJ_ARITH_SEMANTICS_MASK,
            RJ_ARITH_SEMANTICS_LOOSE);
  auto empty_override = json;
  empty_override.insert(empty_override.find('{') + 1, R"("arithmetic":{},)");
  config::LoadedConfig empty;
  ASSERT_TRUE(test::load_valid_config_from_string(empty_override, empty));
  EXPECT_EQ(empty.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags() &
                RJ_ARITH_SEMANTICS_MASK,
            RJ_ARITH_SEMANTICS_COMPATIBILITY);
  json.insert(json.find('{') + 1,
              R"("arithmetic":{"semantics":"bit_exact_empirical_no_fp_status"},)");
  config::LoadedConfig overridden;
  ASSERT_TRUE(test::load_valid_config_from_string(json, overridden));
  auto flags = overridden.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags();
  EXPECT_EQ(flags & RJ_ARITH_SEMANTICS_MASK, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  auto value = json.find("\"value\":\"loose\"");
  ASSERT_NE(value, std::string::npos);
  json.replace(value, std::string("\"value\":\"loose\"").size(), "\"value\":\"invalid\"");
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(
      config::load_config_from_string(json, kEmbeddedSchema, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("arithmetic_semantics"), std::string::npos);
}

TEST(ArithConfigTest, InvalidPerCuEntriesCannotBeHidden) {
  for (auto entry :
       {R"({"key":"arithmetic_semantics","value":null})", R"({"key":"arithmetic_semantics"})",
        R"({"key":"arithmetic_semantics","value":"invalid"},
                        {"key":"arithmetic_semantics","value":"loose"})"}) {
    auto json = arithmetic_config(R"({"semantics":"loose"})");
    const auto cu = json.find("\"type\": \"compute_unit\"");
    const auto entries = json.find('[', json.find("\"config\"", cu));
    ASSERT_NE(entries, std::string::npos);
    json.insert(entries + 1, std::string(entry) + ',');
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(
        config::load_config_from_string(json, kEmbeddedSchema, diagnostic.emitter()).failed());
    EXPECT_NE(diagnostic.message().find("arithmetic_semantics"), std::string::npos);
  }
}

TEST(ArithConfigTest, UnknownPerCuArithmeticControlsAreRejected) {
  for (auto key : {"arithmetic_fallback", "arithmetic_state_reporting", "arithmetic_semantic"}) {
    for (auto policy : {"", R"({"semantics":"loose"})"}) {
      auto json = arithmetic_config(policy);
      const auto cu = json.find("\"type\": \"compute_unit\"");
      ASSERT_NE(cu, std::string::npos);
      const auto entries = json.find('[', json.find("\"config\"", cu));
      ASSERT_NE(entries, std::string::npos);
      json.insert(entries + 1, std::string("{\"key\":\"") + key + "\",\"value\":\"strict\"},");
      util::StringDiagnostic diagnostic;
      EXPECT_TRUE(
          config::load_config_from_string(json, kEmbeddedSchema, diagnostic.emitter()).failed());
      EXPECT_NE(diagnostic.message().find(key), std::string::npos);
    }
  }
}

TEST(ArithConfigTest, CheckpointPreservesPerCuPolicyAndCachedFlags) {
  config::LoadedConfig loaded;
  ASSERT_TRUE(test::load_valid_config_from_string(arithmetic_config(""), loaded));
  auto *se = loaded.soc()->xcd(0)->shader_engine(0);
  ASSERT_GE(se->num_compute_units(), 2u);
  se->compute_unit(0)->restore_arithmetic_policy(
      amdgpu::parse_arithmetic_policy("bit_exact_empirical"));
  se->compute_unit(1)->restore_arithmetic_policy(amdgpu::parse_arithmetic_policy("loose"));
  test::ScopedTempFile file;
  ASSERT_TRUE(file.initialize("rocjitsu-rj_arith-policy-"));
  util::StringDiagnostic diagnostic;
  ASSERT_TRUE(config::save_checkpoint(file.path(), *loaded.soc(), 0, loaded.engine_config,
                                      loaded.cpu_dispatch_threads, diagnostic.emitter())
                  .succeeded())
      << diagnostic.message();
  config::LoadedConfig restored;
  ASSERT_TRUE(test::restore_valid_checkpoint(file.path(), restored));
  auto *restored_se = restored.soc()->xcd(0)->shader_engine(0);
  for (unsigned i = 0; i < 2; ++i) {
    auto *before = se->compute_unit(i);
    auto *after = restored_se->compute_unit(i);
    EXPECT_EQ(after->config().arithmetic.semantics, before->config().arithmetic.semantics);
    EXPECT_EQ(after->arithmetic_flags(), before->arithmetic_flags());
  }
}

TEST(ArithConfigTest, CheckpointRejectsNonuniformCuMembership) {
  // Six CUs arranged 2/3/1 cannot round-trip through a uniform 2/2/2 template.
  const char *json = R"({
    "num_threads": 1, "vm": {"arch": "rdna4"},
    "topology": {"root": {"name": "soc", "type": "soc", "children": [
      {"name": "vram", "type": "gpu_memory"},
      {"name": "xcd0", "type": "xcd", "children": [
        {"name": "l2", "type": "l2_cache"},
        {"name": "cp", "type": "command_processor"},
        {"name": "se0", "type": "shader_engine", "children": [
          {"name": "cu[0:2]", "type": "compute_unit"}]},
        {"name": "se1", "type": "shader_engine", "children": [
          {"name": "cu[0:3]", "type": "compute_unit"}]},
        {"name": "se2", "type": "shader_engine", "children": [
          {"name": "cu0", "type": "compute_unit"}]}
      ]}
    ]}, "links": []}
  })";
  config::LoadedConfig loaded;
  ASSERT_TRUE(test::load_valid_config_from_string(json, loaded));
  loaded.soc()->xcd(0)->shader_engine(2)->compute_unit(0)->restore_arithmetic_policy(
      amdgpu::parse_arithmetic_policy("bit_exact_empirical"));
  test::ScopedTempFile file;
  ASSERT_TRUE(file.initialize("rocjitsu-rj_arith-topology-"));
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(config::save_checkpoint(file.path(), *loaded.soc(), 0, loaded.engine_config,
                                      loaded.cpu_dispatch_threads, diagnostic.emitter())
                  .failed());
  EXPECT_NE(diagnostic.message().find("nonuniform compute-unit topology"), std::string::npos);
}

TEST(ArithConfigTest, CheckpointRejectsUnserializedCus) {
  const std::string cu = R"({"name":"cu0","type":"compute_unit"})";
  const std::string cp = R"({"name":"cp","type":"command_processor"},)";
  const std::string xcd = R"({"name":"xcd0","type":"xcd","children":[
    {"name":"l2","type":"l2_cache"},)";
  for (const auto &children :
       {xcd + cp + cu + "]}", cp + cu,
        cp + cu + ',' + xcd + R"({"name":"se0","type":"shader_engine","children":[)" + cu +
            "]}]}"}) {
    auto json = R"({"num_threads":1,"vm":{"arch":"rdna4"},
      "arithmetic":{"semantics":"bit_exact_empirical"},
      "topology":{"root":{"name":"soc","type":"soc","children":[
        {"name":"vram","type":"gpu_memory"},)" +
                children + "]},\"links\":[]}}";
    config::LoadedConfig loaded;
    ASSERT_TRUE(test::load_valid_config_from_string(json, loaded));
    auto *soc = loaded.soc();
    std::unique_ptr<VirtualMachine> vm;
    for (bool wrapped : {false, true}) {
      SCOPED_TRACE(wrapped);
      if (wrapped) {
        loaded.take_root().release();
        vm = std::make_unique<VirtualMachine>(std::unique_ptr<SoC>(soc), false);
      }
      test::ScopedTempFile file;
      ASSERT_TRUE(file.initialize("rocjitsu-unserialized-cu-"));
      util::StringDiagnostic diagnostic;
      EXPECT_TRUE(config::save_checkpoint(file.path(), *soc, 0, loaded.engine_config,
                                          loaded.cpu_dispatch_threads, diagnostic.emitter())
                      .failed());
      EXPECT_NE(diagnostic.message().find("serialization does not cover every CU"),
                std::string::npos);
    }
  }
}

TEST(ArithConfigTest, CheckpointUsesPerGpuComponentRoot) {
  const char *json = R"({"num_threads":1,"vm":{"arch":"rdna4"},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"xcd0","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu0","type":"compute_unit"}]}]}]},"links":[]}})";
  config::LoadedConfig loaded;
  std::vector<std::unique_ptr<SoC>> socs;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(test::load_valid_config_from_string(json, loaded));
    auto *soc = loaded.soc();
    loaded.take_root().release();
    socs.emplace_back(soc);
  }
  VirtualMachine vm(std::move(socs), {0, 1}, false);
  for (int i = 0; i < 2; ++i) {
    SCOPED_TRACE(i);
    test::ScopedTempFile file;
    ASSERT_TRUE(file.initialize("rocjitsu-per-gpu-checkpoint-"));
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(config::save_checkpoint(file.path(), *vm.soc(i), 0, loaded.engine_config,
                                        loaded.cpu_dispatch_threads, diagnostic.emitter())
                    .succeeded())
        << diagnostic.message();
  }
}

TEST(ArithConfigTest, MissingCheckpointStatePolicyInheritsTemplate) {
  for (bool strict_template : {false, true}) {
    flatbuffers::FlatBufferBuilder builder;
    auto policy =
        strict_template
            ? fb::CreateArithmeticPolicyConfig(builder, builder.CreateString("bit_exact_empirical"))
            : flatbuffers::Offset<fb::ArithmeticPolicyConfig>{};
    fb::ComputeUnitConfigBuilder cu_builder(builder);
    cu_builder.add_num_wf_slots(1);
    cu_builder.add_sgprs_per_wf(104);
    cu_builder.add_vgprs_per_wf(256);
    cu_builder.add_lds_size_kb(64);
    cu_builder.add_arithmetic(policy);
    auto cu = cu_builder.Finish();
    auto se = fb::CreateShaderEngineConfig(builder, 1, cu);
    auto xcd = fb::CreateXcdConfig(builder, 1, se);
    auto gpu = fb::CreateAmdgpuConfig(builder, 1, 0, xcd);
    auto vm = fb::CreateVirtualMachineConfig(builder, builder.CreateString("rdna4"), gpu);
    auto config = fb::CreateSimulationConfig(builder, 0, 1, builder.CreateString("functional"), vm);
    auto state = fb::CreateComputeUnitState(builder);
    auto states = builder.CreateVector(&state, 1);
    builder.Finish(fb::CreateSimulationCheckpoint(builder, 0, config, states));
    test::ScopedTempFile file;
    ASSERT_TRUE(file.initialize("rocjitsu-missing-policy-"));
    ASSERT_TRUE(file.write(std::string_view(
        reinterpret_cast<const char *>(builder.GetBufferPointer()), builder.GetSize())));
    config::LoadedConfig restored;
    ASSERT_TRUE(test::restore_valid_checkpoint(file.path(), restored));
    EXPECT_EQ(restored.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->arithmetic_flags() &
                  RJ_ARITH_SEMANTICS_MASK,
              strict_template ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                              : RJ_ARITH_SEMANTICS_COMPATIBILITY);
  }
}

} // namespace
} // namespace rocjitsu
