//Copyright © Advanced Micro Devices, Inc., or its affiliates.
//SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "core/register_schema.hpp"

namespace aql_profile {
namespace {

TEST(RegisterSchemaTest, DefineAndGetRegister) {
  RegisterSchema schema;

  RegisterDef grbm_def(0x8010, 0, 0xFFFFFFFF, "GRBM GFX Index");
  schema.DefineRegister(RegisterId::GRBM_GFX_INDEX, grbm_def);

  const auto& retrieved = schema.GetRegister(RegisterId::GRBM_GFX_INDEX);
  EXPECT_EQ(retrieved.offset, 0x8010u);
  EXPECT_EQ(retrieved.default_value, 0u);
  EXPECT_EQ(retrieved.write_mask, 0xFFFFFFFFu);
  EXPECT_EQ(retrieved.description, "GRBM GFX Index");
}

TEST(RegisterSchemaTest, DefineRegisterSimple) {
  RegisterSchema schema;

  schema.DefineRegister(RegisterId::CP_PERFMON_CNTL, 0xC1F8);

  const auto& retrieved = schema.GetRegister(RegisterId::CP_PERFMON_CNTL);
  EXPECT_EQ(retrieved.offset, 0xC1F8u);
  EXPECT_EQ(retrieved.default_value, 0u);
  EXPECT_EQ(retrieved.write_mask, 0xFFFFFFFFu);
}

TEST(RegisterSchemaTest, GetOffsetDirectly) {
  RegisterSchema schema;
  schema.DefineRegister(RegisterId::GRBM_GFX_INDEX, 0x8010);

  EXPECT_EQ(schema.GetOffset(RegisterId::GRBM_GFX_INDEX), 0x8010u);
}

TEST(RegisterSchemaTest, HasRegister) {
  RegisterSchema schema;
  schema.DefineRegister(RegisterId::GRBM_GFX_INDEX, 0x8010);

  EXPECT_TRUE(schema.HasRegister(RegisterId::GRBM_GFX_INDEX));
  EXPECT_FALSE(schema.HasRegister(RegisterId::CP_PERFMON_CNTL));
}

TEST(RegisterSchemaTest, TryGetRegisterFound) {
  RegisterSchema schema;
  schema.DefineRegister(RegisterId::GRBM_GFX_INDEX, 0x8010);

  auto result = schema.TryGetRegister(RegisterId::GRBM_GFX_INDEX);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->offset, 0x8010u);
}

TEST(RegisterSchemaTest, TryGetRegisterNotFound) {
  RegisterSchema schema;

  auto result = schema.TryGetRegister(RegisterId::GRBM_GFX_INDEX);
  EXPECT_FALSE(result.has_value());
}

TEST(RegisterSchemaTest, GetRegisterThrowsIfNotFound) {
  RegisterSchema schema;

  EXPECT_THROW(schema.GetRegister(RegisterId::GRBM_GFX_INDEX), std::runtime_error);
}

TEST(RegisterSchemaTest, MergeSchemas) {
  RegisterSchema base_schema;
  base_schema.DefineRegister(RegisterId::GRBM_GFX_INDEX, 0x8010);
  base_schema.DefineRegister(RegisterId::CP_PERFMON_CNTL, 0xC1F8);

  RegisterSchema derived_schema;
  derived_schema.DefineRegister(RegisterId::CP_PERFMON_CNTL, 0xD000);  // Override
  derived_schema.DefineRegister(RegisterId::MC_CONFIG, 0x2000);        // New register

  derived_schema.Merge(base_schema);

  // Base register should be present
  EXPECT_TRUE(derived_schema.HasRegister(RegisterId::GRBM_GFX_INDEX));
  EXPECT_EQ(derived_schema.GetOffset(RegisterId::GRBM_GFX_INDEX), 0x8010u);

  // Overridden register should use base value (merge overwrites)
  EXPECT_EQ(derived_schema.GetOffset(RegisterId::CP_PERFMON_CNTL), 0xC1F8u);

  // New register should still be present
  EXPECT_TRUE(derived_schema.HasRegister(RegisterId::MC_CONFIG));
  EXPECT_EQ(derived_schema.GetOffset(RegisterId::MC_CONFIG), 0x2000u);
}

TEST(RegisterSchemaTest, GetAllRegisters) {
  RegisterSchema schema;
  schema.DefineRegister(RegisterId::GRBM_GFX_INDEX, 0x8010);
  schema.DefineRegister(RegisterId::CP_PERFMON_CNTL, 0xC1F8);

  const auto& all_regs = schema.GetAllRegisters();
  EXPECT_EQ(all_regs.size(), 2u);
  EXPECT_TRUE(all_regs.count(RegisterId::GRBM_GFX_INDEX) > 0);
  EXPECT_TRUE(all_regs.count(RegisterId::CP_PERFMON_CNTL) > 0);
}

TEST(RegisterSchemaTest, MakeGrbmBroadcastValue) {
  uint32_t value = MakeGrbmBroadcastValue(0x3FF, 0x3, 0x3);

  // SE mask in bits 16-25, SA mask in bits 8-9, instance mask in bits 0-1
  uint32_t expected = (0x3FF << 16) | (0x3 << 8) | 0x3;
  EXPECT_EQ(value, expected);
}

TEST(RegisterSchemaTest, MakeGrbmIndexValueSpecific) {
  // Target SE 2, SA 1, instance 0
  uint32_t value = MakeGrbmIndexValue(2, 1, 0, false, false, false);

  uint32_t expected = 2 | (1 << 8) | (0 << 16);
  EXPECT_EQ(value, expected);
}

TEST(RegisterSchemaTest, MakeGrbmIndexValueBroadcastSE) {
  // Broadcast SE, specific SA 1
  uint32_t value = MakeGrbmIndexValue(0, 1, 0, true, false, false);

  // Broadcast SE flag in bit 30
  uint32_t expected = (1 << 8) | (1 << 30);
  EXPECT_EQ(value, expected);
}

TEST(RegisterSchemaTest, MakeGrbmIndexValueBroadcastAll) {
  // Broadcast everything
  uint32_t value = MakeGrbmIndexValue(0, 0, 0, true, true, true);

  // All broadcast flags set
  uint32_t expected = (1 << 30) | (1 << 29) | (1 << 28);
  EXPECT_EQ(value, expected);
}

}  // namespace
}  // namespace aql_profile
