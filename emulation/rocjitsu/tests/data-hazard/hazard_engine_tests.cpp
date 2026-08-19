// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "data_hazard_engine.h"

#include <limits>
#include <set>
#include <stdexcept>

using namespace hazard_core;

namespace {

using hazard_core::EntityId;
using hazard_core::WaitCntType;

class FakeFormatter final : public SimulatorInstructionFormatter {
public:
  std::string format_instruction(const InstructionDescriptor &instruction) const override {
    return "fake_inst_" + std::to_string(instruction.instruction_id);
  }

  /// Overrides one combination only, so the tests below see the shared wording
  /// everywhere else.
  std::string format_wait_suggestion(WaitCntType kind, HazardAccessKind access,
                                     hazard_core::HazardResourceLabel resource) const override {
    if (kind == WaitCntType::VMEM && access == HazardAccessKind::Read &&
        resource == hazard_core::HazardResourceLabel::Register)
      return "fake wait vector-load";
    return "";
  }
};

InstructionDescriptor make_instruction(EntityId id, uint64_t pc) {
  InstructionDescriptor instruction;
  instruction.instruction_id = id;
  instruction.pc = pc;
  instruction.execution.dispatch_id = 1;
  instruction.execution.cluster_id = 0;
  instruction.execution.workgroup_id = 0;
  instruction.execution.wave_id = 0;
  return instruction;
}

DataHazardEngine &reset_generic_engine(FakeFormatter &formatter) {
  auto &engine = data_hazard_engine();
  engine.reset();
  engine.set_instruction_formatter(&formatter);
  engine.set_warning_sink(nullptr);
  engine.on_dispatch_begin(1);
  return engine;
}

} // namespace

TEST(GenericDataHazardEngineTest, DetectsRawHazardFromGenericEvents) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 4;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  engine.on_resource_access(load_write);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 4;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].suggestion, "fake wait vector-load");
}

TEST(GenericDataHazardEngineTest, PendingVgprWriteDoesNotHazardUnrelatedAccRead) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 0;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  engine.on_resource_access(load_write);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent acc_read;
  acc_read.instruction = read.instruction;
  acc_read.resource_kind = ResourceKind::AccumVectorRegister;
  acc_read.register_kind = RegisterKind::AccumVector;
  acc_read.resource_index = 0;
  acc_read.size_bytes = 4;
  acc_read.is_read = true;
  engine.on_resource_access(acc_read);

  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, DetectsVectorRawInsideMultiDwordReadRange) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 5;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  engine.on_resource_access(load_write);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 4;
  read_access.size_bytes = 8;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.resource_index, 5u);
  EXPECT_NE(warnings[0].message.find("RAW hazard: VGPR v5"), std::string::npos);
}

TEST(GenericDataHazardEngineTest, DoesNotInferVectorWriteWaitFromPriorMemoryRead) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load_without_semantics;
  load_without_semantics.instruction = make_instruction(1, 0x100);
  engine.on_instruction(load_without_semantics);

  ResourceAccessEvent memory_read;
  memory_read.instruction = load_without_semantics.instruction;
  memory_read.resource_kind = ResourceKind::GlobalMemory;
  memory_read.address = 0x2000;
  memory_read.size_bytes = 4;
  memory_read.is_read = true;
  engine.on_resource_access(memory_read);

  ResourceAccessEvent vector_write;
  vector_write.instruction = load_without_semantics.instruction;
  vector_write.resource_kind = ResourceKind::VectorRegister;
  vector_write.register_kind = RegisterKind::Vector;
  vector_write.resource_index = 6;
  vector_write.size_bytes = 4;
  vector_write.is_write = true;
  engine.on_resource_access(vector_write);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 6;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, ResourceLevelVectorWriteWaitDetectsRawHazard) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 6;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  load_write.hazards.write_wait = WaitCntType::VMEM;
  engine.on_resource_access(load_write);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 6;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);
}

TEST(GenericDataHazardEngineTest, ResourceEventsCanCarryInstructionContextWithoutInstructionEvent) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionDescriptor load = make_instruction(1, 0x100);
  load.raw_isa[0] = 0xABCD;

  ResourceAccessEvent load_write;
  load_write.instruction = load;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 6;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  load_write.hazards.write_wait = WaitCntType::VMEM;
  engine.on_resource_access(load_write);

  InstructionDescriptor read = make_instruction(2, 0x104);
  read.raw_isa[0] = 0x1234;

  ResourceAccessEvent read_access;
  read_access.instruction = read;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 6;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.instruction.pc, 0x104u);
  EXPECT_EQ(warnings[0].finding.instruction.raw_isa[0], 0x1234u);
  EXPECT_EQ(warnings[0].finding.source_instruction.pc, 0x100u);
  EXPECT_EQ(warnings[0].finding.source_instruction.raw_isa[0], 0xABCDu);
}

TEST(GenericDataHazardEngineTest, ResourceLevelVectorReadWaitDetectsWarHazard) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent store;
  store.instruction = make_instruction(1, 0x100);
  engine.on_instruction(store);

  ResourceAccessEvent source_read;
  source_read.instruction = store.instruction;
  source_read.resource_kind = ResourceKind::VectorRegister;
  source_read.register_kind = RegisterKind::Vector;
  source_read.resource_index = 7;
  source_read.size_bytes = 4;
  source_read.is_read = true;
  source_read.hazards.read_wait = WaitCntType::STORE;
  engine.on_resource_access(source_read);

  InstructionEvent overwrite;
  overwrite.instruction = make_instruction(2, 0x104);
  engine.on_instruction(overwrite);

  ResourceAccessEvent vector_write;
  vector_write.instruction = overwrite.instruction;
  vector_write.resource_kind = ResourceKind::VectorRegister;
  vector_write.register_kind = RegisterKind::Vector;
  vector_write.resource_index = 7;
  vector_write.size_bytes = 4;
  vector_write.is_write = true;
  engine.on_resource_access(vector_write);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::WAR);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::STORE);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);
}

TEST(GenericDataHazardEngineTest, ReportsVectorAndScalarWawForSameInstructionAndRegisterIndex) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent vector_load;
  vector_load.instruction = make_instruction(1, 0x100);
  vector_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(vector_load);

  ResourceAccessEvent pending_vgpr_write;
  pending_vgpr_write.instruction = vector_load.instruction;
  pending_vgpr_write.resource_kind = ResourceKind::VectorRegister;
  pending_vgpr_write.register_kind = RegisterKind::Vector;
  pending_vgpr_write.resource_index = 7;
  pending_vgpr_write.size_bytes = 4;
  pending_vgpr_write.is_write = true;
  engine.on_resource_access(pending_vgpr_write);

  InstructionEvent scalar_load;
  scalar_load.instruction = make_instruction(2, 0x104);
  scalar_load.hazards.scalar_write_wait = WaitCntType::SMEM;
  engine.on_instruction(scalar_load);

  ResourceAccessEvent pending_sgpr_write;
  pending_sgpr_write.instruction = scalar_load.instruction;
  pending_sgpr_write.resource_kind = ResourceKind::ScalarRegister;
  pending_sgpr_write.register_kind = RegisterKind::Scalar;
  pending_sgpr_write.resource_index = 7;
  pending_sgpr_write.size_bytes = 4;
  pending_sgpr_write.is_write = true;
  engine.on_resource_access(pending_sgpr_write);

  InstructionEvent overwrite;
  overwrite.instruction = make_instruction(3, 0x108);
  engine.on_instruction(overwrite);

  ResourceAccessEvent vgpr_overwrite;
  vgpr_overwrite.instruction = overwrite.instruction;
  vgpr_overwrite.resource_kind = ResourceKind::VectorRegister;
  vgpr_overwrite.register_kind = RegisterKind::Vector;
  vgpr_overwrite.resource_index = 7;
  vgpr_overwrite.size_bytes = 4;
  vgpr_overwrite.is_write = true;
  engine.on_resource_access(vgpr_overwrite);

  ResourceAccessEvent sgpr_overwrite;
  sgpr_overwrite.instruction = overwrite.instruction;
  sgpr_overwrite.resource_kind = ResourceKind::ScalarRegister;
  sgpr_overwrite.register_kind = RegisterKind::Scalar;
  sgpr_overwrite.resource_index = 7;
  sgpr_overwrite.size_bytes = 4;
  sgpr_overwrite.is_write = true;
  engine.on_resource_access(sgpr_overwrite);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);

  uint32_t vector_waw_count = 0;
  uint32_t scalar_waw_count = 0;
  for (const auto &warning : warnings) {
    if (warning.finding.kind != HazardKind::WAW || warning.finding.resource_index != 7u)
      continue;
    if (warning.finding.resource_kind == ResourceKind::VectorRegister)
      ++vector_waw_count;
    if (warning.finding.resource_kind == ResourceKind::ScalarRegister)
      ++scalar_waw_count;
  }

  EXPECT_EQ(vector_waw_count, 1u);
  EXPECT_EQ(scalar_waw_count, 1u);
}

TEST(GenericDataHazardEngineTest, MergesStagedInstructionSemanticsWithExistingIdentity) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent decoded_instruction;
  decoded_instruction.instruction = make_instruction(1, 0x100);
  decoded_instruction.instruction.raw_isa[0] = 0xDEADBEEF;
  engine.on_instruction(decoded_instruction);

  InstructionEvent memory_route_update;
  memory_route_update.instruction = make_instruction(1, 0);
  memory_route_update.hazards.vector_write_wait = WaitCntType::VMEM;
  memory_route_update.hazards.vector_write_also_waits_lds = true;
  engine.on_instruction(memory_route_update);

  ResourceAccessEvent vector_dest;
  vector_dest.instruction = memory_route_update.instruction;
  vector_dest.resource_kind = ResourceKind::VectorRegister;
  vector_dest.register_kind = RegisterKind::Vector;
  vector_dest.resource_index = 6;
  vector_dest.size_bytes = 4;
  vector_dest.is_write = true;
  engine.on_resource_access(vector_dest);

  const auto snapshot = engine.wave_snapshot(wave);
  ASSERT_TRUE(snapshot.has_value());
  ASSERT_EQ(snapshot->core.pending_vgpr_writes.count(6), 1u);
  EXPECT_EQ(snapshot->core.pending_vgpr_writes.at(6).pc, 0x100u);
  EXPECT_EQ(snapshot->core.pending_vgpr_writes_ds.count(6), 1u);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 6;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.source_instruction.pc, 0x100u);
  EXPECT_EQ(warnings[0].finding.source_instruction.raw_isa[0], 0xDEADBEEF);
}

TEST(GenericDataHazardEngineTest, SupportsOperandAccessesBeforeRouteSemanticUpdate) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent first_load;
  first_load.instruction = make_instruction(1, 0x100);
  first_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(first_load);

  ResourceAccessEvent first_dest;
  first_dest.instruction = first_load.instruction;
  first_dest.resource_kind = ResourceKind::VectorRegister;
  first_dest.register_kind = RegisterKind::Vector;
  first_dest.resource_index = 4;
  first_dest.size_bytes = 4;
  first_dest.is_write = true;
  engine.on_resource_access(first_dest);

  InstructionEvent routed_load_identity;
  routed_load_identity.instruction = make_instruction(2, 0x104);
  routed_load_identity.instruction.raw_isa[0] = 0xFEEDFACE;
  engine.on_instruction(routed_load_identity);

  ResourceAccessEvent early_source_read;
  early_source_read.instruction = routed_load_identity.instruction;
  early_source_read.resource_kind = ResourceKind::VectorRegister;
  early_source_read.register_kind = RegisterKind::Vector;
  early_source_read.resource_index = 4;
  early_source_read.size_bytes = 4;
  early_source_read.is_read = true;
  engine.on_resource_access(early_source_read);

  ResourceAccessEvent early_dest_write;
  early_dest_write.instruction = routed_load_identity.instruction;
  early_dest_write.resource_kind = ResourceKind::VectorRegister;
  early_dest_write.register_kind = RegisterKind::Vector;
  early_dest_write.resource_index = 6;
  early_dest_write.size_bytes = 4;
  early_dest_write.is_write = true;
  engine.on_resource_access(early_dest_write);

  auto snapshot = engine.wave_snapshot(wave);
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_EQ(snapshot->core.pending_vgpr_writes.count(6), 0u);

  InstructionEvent route_semantics;
  route_semantics.instruction = make_instruction(2, 0);
  route_semantics.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(route_semantics);

  ResourceAccessEvent routed_dest_write = early_dest_write;
  engine.on_resource_access(routed_dest_write);

  snapshot = engine.wave_snapshot(wave);
  ASSERT_TRUE(snapshot.has_value());
  ASSERT_EQ(snapshot->core.pending_vgpr_writes.count(6), 1u);
  EXPECT_EQ(snapshot->core.pending_vgpr_writes.at(6).pc, 0x104u);

  InstructionEvent later_read;
  later_read.instruction = make_instruction(3, 0x108);
  engine.on_instruction(later_read);

  ResourceAccessEvent later_read_access;
  later_read_access.instruction = later_read.instruction;
  later_read_access.resource_kind = ResourceKind::VectorRegister;
  later_read_access.register_kind = RegisterKind::Vector;
  later_read_access.resource_index = 6;
  later_read_access.size_bytes = 4;
  later_read_access.is_read = true;
  engine.on_resource_access(later_read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);
  EXPECT_EQ(warnings[1].finding.source_instruction.instruction_id, 2u);
  EXPECT_EQ(warnings[1].finding.source_instruction.pc, 0x104u);
  EXPECT_EQ(warnings[1].finding.source_instruction.raw_isa[0], 0xFEEDFACE);
}

TEST(GenericDataHazardEngineTest, RocjitsuStyleRouteReplayDoesNotDuplicateDestinationHazards) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent first_load;
  first_load.instruction = make_instruction(1, 0x100);
  first_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(first_load);

  ResourceAccessEvent first_dest;
  first_dest.instruction = first_load.instruction;
  first_dest.resource_kind = ResourceKind::VectorRegister;
  first_dest.register_kind = RegisterKind::Vector;
  first_dest.resource_index = 8;
  first_dest.size_bytes = 4;
  first_dest.is_write = true;
  engine.on_resource_access(first_dest);

  InstructionEvent routed_load_identity;
  routed_load_identity.instruction = make_instruction(2, 0x104);
  routed_load_identity.instruction.raw_isa[0] = 0xFEEDFACE;
  engine.on_instruction(routed_load_identity);

  ResourceAccessEvent early_dest_write;
  early_dest_write.instruction = routed_load_identity.instruction;
  early_dest_write.resource_kind = ResourceKind::VectorRegister;
  early_dest_write.register_kind = RegisterKind::Vector;
  early_dest_write.resource_index = 8;
  early_dest_write.size_bytes = 4;
  early_dest_write.is_write = true;
  engine.on_resource_access(early_dest_write);

  auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::WAW);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);

  InstructionEvent route_semantics;
  route_semantics.instruction = make_instruction(2, 0);
  route_semantics.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(route_semantics);

  ResourceAccessEvent routed_dest_write = early_dest_write;
  engine.on_resource_access(routed_dest_write);

  warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);

  engine.on_resource_access(routed_dest_write);

  warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);

  auto snapshot = engine.wave_snapshot(wave);
  ASSERT_TRUE(snapshot.has_value());
  ASSERT_EQ(snapshot->core.pending_vgpr_writes.count(8), 1u);
  EXPECT_EQ(snapshot->core.pending_vgpr_writes.at(8).instruction_id, 2u);
  EXPECT_EQ(snapshot->core.pending_vgpr_writes.at(8).pc, 0x104u);

  InstructionEvent later_read;
  later_read.instruction = make_instruction(3, 0x108);
  engine.on_instruction(later_read);

  ResourceAccessEvent later_read_access;
  later_read_access.instruction = later_read.instruction;
  later_read_access.resource_kind = ResourceKind::VectorRegister;
  later_read_access.register_kind = RegisterKind::Vector;
  later_read_access.resource_index = 8;
  later_read_access.size_bytes = 4;
  later_read_access.is_read = true;
  engine.on_resource_access(later_read_access);

  warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  EXPECT_EQ(warnings[1].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[1].finding.source_instruction.instruction_id, 2u);
  EXPECT_EQ(warnings[1].finding.source_instruction.pc, 0x104u);
  EXPECT_EQ(warnings[1].finding.source_instruction.raw_isa[0], 0xFEEDFACE);
}

TEST(GenericDataHazardEngineTest, SemanticUpdateDoesNotReplayExistingWaitAction) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 4;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  engine.on_resource_access(load_write);

  InstructionEvent wait;
  wait.instruction = make_instruction(2, 0x104);
  wait.wait_action.is_wait_instruction = true;
  wait.wait_action.counters.push_back({WaitCntType::VMEM, 0});
  engine.on_instruction(wait);

  InstructionEvent second_load;
  second_load.instruction = make_instruction(3, 0x108);
  second_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(second_load);

  ResourceAccessEvent second_write;
  second_write.instruction = second_load.instruction;
  second_write.resource_kind = ResourceKind::VectorRegister;
  second_write.register_kind = RegisterKind::Vector;
  second_write.resource_index = 4;
  second_write.size_bytes = 4;
  second_write.is_write = true;
  engine.on_resource_access(second_write);

  InstructionEvent semantic_only_wait_update;
  semantic_only_wait_update.instruction = make_instruction(2, 0);
  semantic_only_wait_update.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(semantic_only_wait_update);

  InstructionEvent read;
  read.instruction = make_instruction(4, 0x10c);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 4;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("RAW hazard: VGPR v4"), std::string::npos);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 3u);
}

TEST(GenericDataHazardEngineTest, WaitActionClearsPendingRawHazard) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 4;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  engine.on_resource_access(load_write);

  InstructionEvent wait;
  wait.instruction = make_instruction(2, 0x104);
  wait.wait_action.is_wait_instruction = true;
  wait.wait_action.counters.push_back({WaitCntType::VMEM, 0});
  engine.on_instruction(wait);

  InstructionEvent read;
  read.instruction = make_instruction(3, 0x108);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 4;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, CounterWaitClearsPendingEvenWithoutWaitInstructionFlag) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 4;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  engine.on_resource_access(load_write);

  InstructionEvent wait;
  wait.instruction = make_instruction(2, 0x104);
  wait.wait_action.counters.push_back({WaitCntType::VMEM, 0});
  engine.on_instruction(wait);

  InstructionEvent read;
  read.instruction = make_instruction(3, 0x108);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 4;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, IdleWaitClearsAllPendingDomainsWithoutExplicitCounters) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent vmem_load;
  vmem_load.instruction = make_instruction(1, 0x100);
  vmem_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(vmem_load);

  ResourceAccessEvent vector_write;
  vector_write.instruction = vmem_load.instruction;
  vector_write.resource_kind = ResourceKind::VectorRegister;
  vector_write.register_kind = RegisterKind::Vector;
  vector_write.resource_index = 4;
  vector_write.size_bytes = 4;
  vector_write.is_write = true;
  engine.on_resource_access(vector_write);

  InstructionEvent async_vector_read;
  async_vector_read.instruction = make_instruction(2, 0x104);
  async_vector_read.hazards.vector_read_wait = WaitCntType::STORE;
  engine.on_instruction(async_vector_read);

  ResourceAccessEvent vector_read;
  vector_read.instruction = async_vector_read.instruction;
  vector_read.resource_kind = ResourceKind::VectorRegister;
  vector_read.register_kind = RegisterKind::Vector;
  vector_read.resource_index = 7;
  vector_read.size_bytes = 4;
  vector_read.is_read = true;
  engine.on_resource_access(vector_read);

  InstructionEvent smem_load;
  smem_load.instruction = make_instruction(3, 0x108);
  smem_load.hazards.scalar_write_wait = WaitCntType::SMEM;
  engine.on_instruction(smem_load);

  ResourceAccessEvent scalar_write;
  scalar_write.instruction = smem_load.instruction;
  scalar_write.resource_kind = ResourceKind::ScalarRegister;
  scalar_write.register_kind = RegisterKind::Scalar;
  scalar_write.resource_index = 2;
  scalar_write.size_bytes = 4;
  scalar_write.is_write = true;
  engine.on_resource_access(scalar_write);

  InstructionEvent lds_write_inst;
  lds_write_inst.instruction = make_instruction(4, 0x10c);
  lds_write_inst.hazards.local_write_wait = WaitCntType::LDS;
  engine.on_instruction(lds_write_inst);

  ResourceAccessEvent lds_write;
  lds_write.instruction = lds_write_inst.instruction;
  lds_write.resource_kind = ResourceKind::LocalMemory;
  lds_write.address = 0x80;
  lds_write.size_bytes = 4;
  lds_write.is_write = true;
  engine.on_resource_access(lds_write);

  InstructionEvent tensor_write_inst;
  tensor_write_inst.instruction = make_instruction(5, 0x110);
  tensor_write_inst.hazards.local_write_wait = WaitCntType::TENSOR;
  engine.on_instruction(tensor_write_inst);

  ResourceAccessEvent tensor_write;
  tensor_write.instruction = tensor_write_inst.instruction;
  tensor_write.resource_kind = ResourceKind::LocalMemory;
  tensor_write.address = 0x100;
  tensor_write.size_bytes = 4;
  tensor_write.is_write = true;
  engine.on_resource_access(tensor_write);

  InstructionEvent idle_wait;
  idle_wait.instruction = make_instruction(6, 0x114);
  idle_wait.wait_action.waits_for_idle = true;
  engine.on_instruction(idle_wait);

  const auto snapshot = engine.wave_snapshot(wave);
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_TRUE(snapshot->core.pending_vgpr_writes.empty());
  EXPECT_TRUE(snapshot->core.pending_vgpr_reads.empty());
  EXPECT_TRUE(snapshot->core.pending_sgpr_writes.empty());
  EXPECT_TRUE(snapshot->core.lds_fifo.empty());
  EXPECT_TRUE(snapshot->core.tensor_lds_fifo.empty());
  EXPECT_TRUE(snapshot->core.vmem_load_fifo.empty());
  EXPECT_TRUE(snapshot->core.vmem_store_fifo.empty());
  EXPECT_TRUE(snapshot->core.smem_load_fifo.empty());
}

TEST(GenericDataHazardEngineTest, DetectsPureDsVectorDestinationWithDscntSuggestion) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent ds_read;
  ds_read.instruction = make_instruction(1, 0x100);
  ds_read.hazards.vector_write_wait = WaitCntType::LDS;
  engine.on_instruction(ds_read);

  ResourceAccessEvent ds_dest;
  ds_dest.instruction = ds_read.instruction;
  ds_dest.resource_kind = ResourceKind::VectorRegister;
  ds_dest.register_kind = RegisterKind::Vector;
  ds_dest.resource_index = 5;
  ds_dest.size_bytes = 4;
  ds_dest.is_write = true;
  engine.on_resource_access(ds_dest);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 5;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("RAW hazard: VGPR v5"), std::string::npos);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::LDS);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_dscnt 0 before reading this register");
}

TEST(GenericDataHazardEngineTest, TracksFlatVectorDestinationOnBothCounters) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent flat_load;
  flat_load.instruction = make_instruction(1, 0x100);
  flat_load.hazards.vector_write_wait = WaitCntType::VMEM;
  flat_load.hazards.vector_write_also_waits_lds = true;
  engine.on_instruction(flat_load);

  ResourceAccessEvent flat_dest;
  flat_dest.instruction = flat_load.instruction;
  flat_dest.resource_kind = ResourceKind::VectorRegister;
  flat_dest.register_kind = RegisterKind::Vector;
  flat_dest.resource_index = 10;
  flat_dest.size_bytes = 4;
  flat_dest.is_write = true;
  engine.on_resource_access(flat_dest);

  const auto snapshot = engine.wave_snapshot(wave);
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_EQ(snapshot->core.pending_vgpr_writes.count(10), 1u);
  EXPECT_EQ(snapshot->core.pending_vgpr_writes_ds.count(10), 1u);
  EXPECT_TRUE(snapshot->core.pending_vgpr_writes_ds.at(10).is_flat_ds);
}

TEST(GenericDataHazardEngineTest, ReportsFlatVectorRawWithCombinedWaitSuggestion) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent flat_load;
  flat_load.instruction = make_instruction(1, 0x100);
  flat_load.hazards.vector_write_wait = WaitCntType::VMEM;
  flat_load.hazards.vector_write_also_waits_lds = true;
  engine.on_instruction(flat_load);

  ResourceAccessEvent flat_dest;
  flat_dest.instruction = flat_load.instruction;
  flat_dest.resource_kind = ResourceKind::VectorRegister;
  flat_dest.register_kind = RegisterKind::Vector;
  flat_dest.resource_index = 10;
  flat_dest.size_bytes = 4;
  flat_dest.is_write = true;
  engine.on_resource_access(flat_dest);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 10;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("RAW hazard: VGPR v10"), std::string::npos);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_loadcnt_dscnt 0 before reading this register (flat "
                                    "load uses both LOADcnt and DScnt)");
}

TEST(GenericDataHazardEngineTest, FlatVectorDscntSideAloneDoesNotReportWaw) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent flat_load;
  flat_load.instruction = make_instruction(1, 0x100);
  flat_load.hazards.vector_write_wait = WaitCntType::VMEM;
  flat_load.hazards.vector_write_also_waits_lds = true;
  engine.on_instruction(flat_load);

  ResourceAccessEvent flat_dest;
  flat_dest.instruction = flat_load.instruction;
  flat_dest.resource_kind = ResourceKind::VectorRegister;
  flat_dest.register_kind = RegisterKind::Vector;
  flat_dest.resource_index = 10;
  flat_dest.size_bytes = 4;
  flat_dest.is_write = true;
  engine.on_resource_access(flat_dest);

  InstructionEvent loadcnt_wait;
  loadcnt_wait.instruction = make_instruction(2, 0x104);
  loadcnt_wait.wait_action.is_wait_instruction = true;
  loadcnt_wait.wait_action.counters.push_back({WaitCntType::VMEM, 0});
  engine.on_instruction(loadcnt_wait);

  InstructionEvent overwrite;
  overwrite.instruction = make_instruction(3, 0x108);
  engine.on_instruction(overwrite);

  ResourceAccessEvent overwrite_dest;
  overwrite_dest.instruction = overwrite.instruction;
  overwrite_dest.resource_kind = ResourceKind::VectorRegister;
  overwrite_dest.register_kind = RegisterKind::Vector;
  overwrite_dest.resource_index = 10;
  overwrite_dest.size_bytes = 4;
  overwrite_dest.is_write = true;
  engine.on_resource_access(overwrite_dest);

  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, DetectsVgprWawFromDsWriteAfterVmemWrite) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent vmem_load;
  vmem_load.instruction = make_instruction(1, 0x100);
  vmem_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(vmem_load);

  ResourceAccessEvent vmem_dest;
  vmem_dest.instruction = vmem_load.instruction;
  vmem_dest.resource_kind = ResourceKind::VectorRegister;
  vmem_dest.register_kind = RegisterKind::Vector;
  vmem_dest.resource_index = 12;
  vmem_dest.size_bytes = 4;
  vmem_dest.is_write = true;
  engine.on_resource_access(vmem_dest);

  InstructionEvent ds_read;
  ds_read.instruction = make_instruction(2, 0x104);
  ds_read.hazards.vector_write_wait = WaitCntType::LDS;
  engine.on_instruction(ds_read);

  ResourceAccessEvent ds_dest;
  ds_dest.instruction = ds_read.instruction;
  ds_dest.resource_kind = ResourceKind::VectorRegister;
  ds_dest.register_kind = RegisterKind::Vector;
  ds_dest.resource_index = 12;
  ds_dest.size_bytes = 4;
  ds_dest.is_write = true;
  engine.on_resource_access(ds_dest);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("WAW hazard: VGPR v12"), std::string::npos);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_loadcnt 0 before writing this register");
}

TEST(GenericDataHazardEngineTest, DetectsVgprWawBetweenPureDsVectorDestinations) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent first_ds_read;
  first_ds_read.instruction = make_instruction(1, 0x100);
  first_ds_read.hazards.vector_write_wait = WaitCntType::LDS;
  engine.on_instruction(first_ds_read);

  ResourceAccessEvent first_dest;
  first_dest.instruction = first_ds_read.instruction;
  first_dest.resource_kind = ResourceKind::VectorRegister;
  first_dest.register_kind = RegisterKind::Vector;
  first_dest.resource_index = 13;
  first_dest.size_bytes = 4;
  first_dest.is_write = true;
  engine.on_resource_access(first_dest);

  InstructionEvent second_ds_read;
  second_ds_read.instruction = make_instruction(2, 0x104);
  second_ds_read.hazards.vector_write_wait = WaitCntType::LDS;
  engine.on_instruction(second_ds_read);

  ResourceAccessEvent second_dest;
  second_dest.instruction = second_ds_read.instruction;
  second_dest.resource_kind = ResourceKind::VectorRegister;
  second_dest.register_kind = RegisterKind::Vector;
  second_dest.resource_index = 13;
  second_dest.size_bytes = 4;
  second_dest.is_write = true;
  engine.on_resource_access(second_dest);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("WAW hazard: VGPR v13"), std::string::npos);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::LDS);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_dscnt 0 before writing this register");
}

TEST(GenericDataHazardEngineTest, DetectsLdsRawHazardWithLdsSuggestion) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent write_inst;
  write_inst.instruction = make_instruction(1, 0x100);
  write_inst.hazards.local_write_wait = WaitCntType::LDS;
  engine.on_instruction(write_inst);

  ResourceAccessEvent write_access;
  write_access.instruction = write_inst.instruction;
  write_access.resource_kind = ResourceKind::LocalMemory;
  write_access.address = 0x80;
  write_access.size_bytes = 4;
  write_access.is_write = true;
  engine.on_resource_access(write_access);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(2, 0x104);
  read_inst.hazards.local_write_wait = WaitCntType::LDS;
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::LocalMemory;
  read_access.address = 0x80;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("RAW hazard: LDS address 0x80"), std::string::npos);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_dscnt 0 before reading this LDS address");
}

TEST(GenericDataHazardEngineTest, DetectsLdsWarHazardWithLdsAddressSuggestion) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(1, 0x100);
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::LocalMemory;
  read_access.address = 0x80;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  InstructionEvent write_inst;
  write_inst.instruction = make_instruction(2, 0x104);
  write_inst.hazards.local_write_wait = WaitCntType::LDS;
  engine.on_instruction(write_inst);

  ResourceAccessEvent write_access;
  write_access.instruction = write_inst.instruction;
  write_access.resource_kind = ResourceKind::LocalMemory;
  write_access.address = 0x80;
  write_access.size_bytes = 4;
  write_access.is_write = true;
  engine.on_resource_access(write_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("WAR hazard: LDS address 0x80"), std::string::npos);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_dscnt 0 before writing this LDS address");
}

TEST(GenericDataHazardEngineTest, DetectsVgprWawHazardWithWriteSuggestion) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent first_load;
  first_load.instruction = make_instruction(1, 0x100);
  first_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(first_load);

  ResourceAccessEvent first_write;
  first_write.instruction = first_load.instruction;
  first_write.resource_kind = ResourceKind::VectorRegister;
  first_write.register_kind = RegisterKind::Vector;
  first_write.resource_index = 7;
  first_write.size_bytes = 4;
  first_write.is_write = true;
  engine.on_resource_access(first_write);

  InstructionEvent second_load;
  second_load.instruction = make_instruction(2, 0x104);
  second_load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(second_load);

  ResourceAccessEvent second_write;
  second_write.instruction = second_load.instruction;
  second_write.resource_kind = ResourceKind::VectorRegister;
  second_write.register_kind = RegisterKind::Vector;
  second_write.resource_index = 7;
  second_write.size_bytes = 4;
  second_write.is_write = true;
  engine.on_resource_access(second_write);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("WAW hazard: VGPR v7"), std::string::npos);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_loadcnt 0 before writing this register");
}

TEST(GenericDataHazardEngineTest, DetectsSgprWawHazardWithWriteSuggestion) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent smem_load;
  smem_load.instruction = make_instruction(1, 0x100);
  smem_load.hazards.scalar_write_wait = WaitCntType::SMEM;
  engine.on_instruction(smem_load);

  ResourceAccessEvent first_write;
  first_write.instruction = smem_load.instruction;
  first_write.resource_kind = ResourceKind::ScalarRegister;
  first_write.register_kind = RegisterKind::Scalar;
  first_write.resource_index = 12;
  first_write.size_bytes = 4;
  first_write.is_write = true;
  engine.on_resource_access(first_write);

  InstructionEvent scalar_alu;
  scalar_alu.instruction = make_instruction(2, 0x104);
  engine.on_instruction(scalar_alu);

  ResourceAccessEvent second_write;
  second_write.instruction = scalar_alu.instruction;
  second_write.resource_kind = ResourceKind::ScalarRegister;
  second_write.register_kind = RegisterKind::Scalar;
  second_write.resource_index = 12;
  second_write.size_bytes = 4;
  second_write.is_write = true;
  engine.on_resource_access(second_write);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::WAW);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::ScalarRegister);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::SMEM);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_kmcnt 0 before writing this register");
}

namespace {

// s_load_b256 s[4:11] with the s_wait_kmcnt removed, then a narrow read of s10
// followed by a wide read of s[10:11]. Taken from the add_buffer mutant that
// exposed the two cases below.
DataHazardEngine &wide_scalar_load_pending(FakeFormatter &formatter) {
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);
  engine.on_wave_begin(ExecutionKey{1, 0, 0, 0, 0});

  InstructionEvent smem_load;
  smem_load.instruction = make_instruction(1, 0x100);
  smem_load.hazards.scalar_write_wait = WaitCntType::SMEM;
  engine.on_instruction(smem_load);

  ResourceAccessEvent load_dst;
  load_dst.instruction = smem_load.instruction;
  load_dst.resource_kind = ResourceKind::ScalarRegister;
  load_dst.register_kind = RegisterKind::Scalar;
  load_dst.resource_index = 4;
  load_dst.size_bytes = 32;
  load_dst.is_write = true;
  engine.on_resource_access(load_dst);
  return engine;
}

void read_scalars(DataHazardEngine &engine, EntityId instruction_id, uint64_t pc,
                  uint32_t first_sgpr, uint32_t size_bytes) {
  InstructionEvent consumer;
  consumer.instruction = make_instruction(instruction_id, pc);
  engine.on_instruction(consumer);

  ResourceAccessEvent read;
  read.instruction = consumer.instruction;
  read.resource_kind = ResourceKind::ScalarRegister;
  read.register_kind = RegisterKind::Scalar;
  read.resource_index = first_sgpr;
  read.size_bytes = size_bytes;
  read.is_read = true;
  engine.on_resource_access(read);
}

} // namespace

// A wide operand whose leading register was already reported for an earlier
// consumer must still report the registers that were not: only s10 has been
// named so far, so the read of s[10:11] still owes a warning for s11.
TEST(GenericDataHazardEngineTest, WideScalarReadReportsRegistersLeftOverFromAnEarlierConsumer) {
  FakeFormatter formatter;
  auto &engine = wide_scalar_load_pending(formatter);

  read_scalars(engine, 2, 0x104, /*first_sgpr=*/10, /*size_bytes=*/4);
  read_scalars(engine, 3, 0x108, /*first_sgpr=*/10, /*size_bytes=*/8);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  EXPECT_NE(warnings[0].message.find("SGPR s10"), std::string::npos);
  EXPECT_NE(warnings[1].message.find("SGPR s11"), std::string::npos);
}

// Frontends differ in whether they hand a 64-bit operand over as one access or
// as one per register. The reported hazards must not depend on that.
TEST(GenericDataHazardEngineTest, ScalarHazardsDoNotDependOnOperandDeliveryGranularity) {
  // The engine is a singleton, so each sequence has to be snapshotted before
  // the next one resets it.
  auto messages = [](const DataHazardEngine &engine) {
    std::vector<std::string> out;
    for (const auto &w : engine.warning_snapshot())
      out.push_back(w.message);
    return out;
  };

  FakeFormatter wide_formatter;
  auto &wide = wide_scalar_load_pending(wide_formatter);
  read_scalars(wide, 2, 0x104, 10, 4);
  read_scalars(wide, 3, 0x108, 10, 8);
  const std::vector<std::string> wide_messages = messages(wide);

  FakeFormatter split_formatter;
  auto &split = wide_scalar_load_pending(split_formatter);
  read_scalars(split, 2, 0x104, 10, 4);
  // Same instruction, delivered as two single-register accesses.
  read_scalars(split, 3, 0x108, 10, 4);
  read_scalars(split, 3, 0x108, 11, 4);
  const std::vector<std::string> split_messages = messages(split);

  EXPECT_EQ(wide_messages, split_messages);
  ASSERT_EQ(wide_messages.size(), 2u);
  EXPECT_NE(wide_messages[1].find("SGPR s11"), std::string::npos);
}

TEST(GenericDataHazardEngineTest, DetectsVgprWarOnlyForExplicitAsyncReadSemantics) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent async_read;
  async_read.instruction = make_instruction(1, 0x100);
  async_read.hazards.vector_read_wait = WaitCntType::STORE;
  engine.on_instruction(async_read);

  ResourceAccessEvent source_read;
  source_read.instruction = async_read.instruction;
  source_read.resource_kind = ResourceKind::VectorRegister;
  source_read.register_kind = RegisterKind::Vector;
  source_read.resource_index = 9;
  source_read.size_bytes = 4;
  source_read.is_read = true;
  engine.on_resource_access(source_read);

  InstructionEvent overwrite;
  overwrite.instruction = make_instruction(2, 0x104);
  engine.on_instruction(overwrite);

  ResourceAccessEvent vector_write;
  vector_write.instruction = overwrite.instruction;
  vector_write.resource_kind = ResourceKind::VectorRegister;
  vector_write.register_kind = RegisterKind::Vector;
  vector_write.resource_index = 9;
  vector_write.size_bytes = 4;
  vector_write.is_write = true;
  engine.on_resource_access(vector_write);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("WAR hazard: VGPR v9"), std::string::npos);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::STORE);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_storecnt 0 before writing this register");
}

TEST(GenericDataHazardEngineTest, UsesPendingWaitTypeForExplicitVectorReadWar) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent async_read;
  async_read.instruction = make_instruction(1, 0x100);
  async_read.hazards.vector_read_wait = WaitCntType::VMEM;
  engine.on_instruction(async_read);

  ResourceAccessEvent source_read;
  source_read.instruction = async_read.instruction;
  source_read.resource_kind = ResourceKind::VectorRegister;
  source_read.register_kind = RegisterKind::Vector;
  source_read.resource_index = 11;
  source_read.size_bytes = 4;
  source_read.is_read = true;
  engine.on_resource_access(source_read);

  InstructionEvent overwrite;
  overwrite.instruction = make_instruction(2, 0x104);
  engine.on_instruction(overwrite);

  ResourceAccessEvent vector_write;
  vector_write.instruction = overwrite.instruction;
  vector_write.resource_kind = ResourceKind::VectorRegister;
  vector_write.register_kind = RegisterKind::Vector;
  vector_write.resource_index = 11;
  vector_write.size_bytes = 4;
  vector_write.is_write = true;
  engine.on_resource_access(vector_write);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].message.find("WAR hazard: VGPR v11"), std::string::npos);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_loadcnt 0 before writing this register");
}

TEST(GenericDataHazardEngineTest, DetectsCrossWorkgroupGlobalRace) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);
  engine.on_workgroup_begin(1, 0, 1);

  ExecutionKey writer_wave{1, 0, 0, 0, 0};
  ExecutionKey reader_wave{1, 0, 1, 0, 0};
  engine.on_wave_begin(writer_wave);
  engine.on_wave_begin(reader_wave);

  InstructionEvent write_inst;
  write_inst.instruction = make_instruction(1, 0x100);
  write_inst.instruction.execution = writer_wave;
  write_inst.instruction.raw_isa[0] = 0xABCDEF01;
  engine.on_instruction(write_inst);

  ResourceAccessEvent write_access;
  write_access.instruction = write_inst.instruction;
  write_access.resource_kind = ResourceKind::GlobalMemory;
  write_access.address = 0x1000;
  write_access.size_bytes = 4;
  write_access.is_write = true;
  engine.on_resource_access(write_access);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(2, 0x200);
  read_inst.instruction.execution = reader_wave;
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::GlobalMemory;
  read_access.address = 0x1000;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
  EXPECT_NE(warnings[0].message.find("Global memory RAW data race at address 0x1000"),
            std::string::npos);
  EXPECT_EQ(warnings[0].suggestion,
            "Use atomic operations or add proper synchronization between workgroups");
  EXPECT_TRUE(warnings[0].has_source);
  EXPECT_EQ(warnings[0].finding.source_instruction.raw_isa[0], 0xABCDEF01);
  EXPECT_EQ(warnings[0].source_raw_isa[0], 0xABCDEF01);
}

// Later readers must not displace every earlier one: whichever workgroup writes
// next still races with a read from a different workgroup, including the
// workgroup that read most recently.
TEST(GenericDataHazardEngineTest, GlobalWarSurvivesReadsFromOtherWorkgroups) {
  constexpr EntityId kReadingWorkgroups = 3;

  for (EntityId writing_workgroup = 0; writing_workgroup < kReadingWorkgroups;
       ++writing_workgroup) {
    FakeFormatter formatter;
    auto &engine = reset_generic_engine(formatter);
    for (EntityId workgroup = 0; workgroup < kReadingWorkgroups; ++workgroup) {
      engine.on_workgroup_begin(1, 0, workgroup);
      engine.on_wave_begin(ExecutionKey{1, 0, workgroup, 0, 0});
    }

    auto access = [&](EntityId instruction_id, EntityId workgroup, bool is_write) {
      InstructionEvent instruction;
      instruction.instruction = make_instruction(instruction_id, 0x100 + instruction_id * 4);
      instruction.instruction.execution = ExecutionKey{1, 0, workgroup, 0, 0};
      engine.on_instruction(instruction);

      ResourceAccessEvent event;
      event.instruction = instruction.instruction;
      event.resource_kind = ResourceKind::GlobalMemory;
      event.address = 0x1000;
      event.size_bytes = 4;
      event.is_read = !is_write;
      event.is_write = is_write;
      engine.on_resource_access(event);
    };

    for (EntityId workgroup = 0; workgroup < kReadingWorkgroups; ++workgroup)
      access(1 + workgroup, workgroup, /*is_write=*/false);
    ASSERT_TRUE(engine.warning_snapshot().empty()) << "concurrent reads do not conflict";

    access(100, writing_workgroup, /*is_write=*/true);

    const auto warnings = engine.warning_snapshot();
    ASSERT_EQ(warnings.size(), 1u)
        << "a write by workgroup " << writing_workgroup << " races with the other reads";
    EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
    EXPECT_NE(warnings[0].message.find("Global memory WAR data race at address 0x1000"),
              std::string::npos);
    EXPECT_NE(warnings[0].finding.source_instruction.execution.workgroup_id, writing_workgroup)
        << "the conflicting read must come from another workgroup";
  }
}

TEST(GenericDataHazardEngineTest, GlobalRaceSourceInstructionUsesConflictingExecutionKey) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_dispatch_begin(2);
  engine.on_workgroup_begin(2, 3, 4);
  engine.on_workgroup_begin(2, 7, 8);

  ExecutionKey writer_wave{2, 3, 4, 5, 6};
  ExecutionKey reader_wave{2, 7, 8, 9, 10};
  engine.on_wave_begin(writer_wave);
  engine.on_wave_begin(reader_wave);

  InstructionEvent write_inst;
  write_inst.instruction = make_instruction(11, 0x110);
  write_inst.instruction.execution = writer_wave;
  write_inst.instruction.raw_isa[0] = 0x11111111;
  engine.on_instruction(write_inst);

  ResourceAccessEvent write_access;
  write_access.instruction = write_inst.instruction;
  write_access.resource_kind = ResourceKind::GlobalMemory;
  write_access.address = 0x2000;
  write_access.size_bytes = 4;
  write_access.is_write = true;
  engine.on_resource_access(write_access);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(12, 0x220);
  read_inst.instruction.execution = reader_wave;
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::GlobalMemory;
  read_access.address = 0x2000;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  const auto &source_execution = warnings[0].finding.source_instruction.execution;
  EXPECT_EQ(source_execution.dispatch_id, 2u);
  EXPECT_EQ(source_execution.cluster_id, 3u);
  EXPECT_EQ(source_execution.workgroup_id, 4u);
  EXPECT_EQ(source_execution.wavegroup_id, 0u);
  EXPECT_EQ(source_execution.wave_id, 6u);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 11u);
  EXPECT_EQ(warnings[0].finding.source_instruction.raw_isa[0], 0x11111111u);
}

TEST(GenericDataHazardEngineTest, GlobalShadowIsDispatchScoped) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_workgroup_begin(1, 0, 0);
  engine.on_workgroup_begin(1, 0, 1);

  ExecutionKey dispatch_one_writer{1, 0, 0, 0, 0};
  ExecutionKey dispatch_one_reader{1, 0, 1, 0, 1};
  engine.on_wave_begin(dispatch_one_writer);
  engine.on_wave_begin(dispatch_one_reader);

  InstructionEvent write_inst;
  write_inst.instruction = make_instruction(1, 0x100);
  write_inst.instruction.execution = dispatch_one_writer;
  engine.on_instruction(write_inst);

  ResourceAccessEvent write_access;
  write_access.instruction = write_inst.instruction;
  write_access.resource_kind = ResourceKind::GlobalMemory;
  write_access.address = 0x3000;
  write_access.size_bytes = 4;
  write_access.is_write = true;
  engine.on_resource_access(write_access);

  engine.on_dispatch_begin(2);
  engine.on_workgroup_begin(2, 0, 0);
  ExecutionKey dispatch_two_reader{2, 0, 0, 0, 0};
  engine.on_wave_begin(dispatch_two_reader);

  InstructionEvent dispatch_two_read_inst;
  dispatch_two_read_inst.instruction = make_instruction(2, 0x200);
  dispatch_two_read_inst.instruction.execution = dispatch_two_reader;
  engine.on_instruction(dispatch_two_read_inst);

  ResourceAccessEvent dispatch_two_read;
  dispatch_two_read.instruction = dispatch_two_read_inst.instruction;
  dispatch_two_read.resource_kind = ResourceKind::GlobalMemory;
  dispatch_two_read.address = 0x3000;
  dispatch_two_read.size_bytes = 4;
  dispatch_two_read.is_read = true;
  engine.on_resource_access(dispatch_two_read);

  ASSERT_TRUE(engine.warning_snapshot().empty())
      << "dispatch 2 must not race with dispatch 1 shadow state";

  InstructionEvent dispatch_one_read_inst;
  dispatch_one_read_inst.instruction = make_instruction(3, 0x300);
  dispatch_one_read_inst.instruction.execution = dispatch_one_reader;
  engine.on_instruction(dispatch_one_read_inst);

  ResourceAccessEvent dispatch_one_read;
  dispatch_one_read.instruction = dispatch_one_read_inst.instruction;
  dispatch_one_read.resource_kind = ResourceKind::GlobalMemory;
  dispatch_one_read.address = 0x3000;
  dispatch_one_read.size_bytes = 4;
  dispatch_one_read.is_read = true;
  engine.on_resource_access(dispatch_one_read);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
  EXPECT_EQ(warnings[0].finding.instruction.execution.dispatch_id, 1u);
  EXPECT_EQ(warnings[0].finding.source_instruction.execution.dispatch_id, 1u);
}

namespace {

/// Runs *first* then *second* as global accesses from two distinct workgroups
/// of one dispatch and returns the warnings raised.
struct SubDwordAccess {
  uint64_t address;
  uint32_t size_bytes;
  bool is_write;
};

std::vector<EngineWarning> run_cross_workgroup_global_accesses(DataHazardEngine &engine,
                                                               const SubDwordAccess &first,
                                                               const SubDwordAccess &second) {
  engine.on_workgroup_begin(1, 0, 0);
  engine.on_workgroup_begin(1, 0, 1);

  ExecutionKey first_wave{1, 0, 0, 0, 0};
  ExecutionKey second_wave{1, 0, 1, 0, 0};
  engine.on_wave_begin(first_wave);
  engine.on_wave_begin(second_wave);

  EntityId next_id = 1;
  for (const auto &[access, wave] :
       {std::pair{first, first_wave}, std::pair{second, second_wave}}) {
    InstructionEvent inst;
    inst.instruction = make_instruction(next_id, 0x100 * next_id);
    inst.instruction.execution = wave;
    engine.on_instruction(inst);

    ResourceAccessEvent event;
    event.instruction = inst.instruction;
    event.resource_kind = ResourceKind::GlobalMemory;
    event.address = access.address;
    event.size_bytes = access.size_bytes;
    event.is_write = access.is_write;
    event.is_read = !access.is_write;
    engine.on_resource_access(event);
    ++next_id;
  }
  return engine.warning_snapshot();
}

} // namespace

TEST(GenericDataHazardEngineTest, DisjointSubDwordAccessesInOneGranuleDoNotRace) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  // Both bytes live in the 4-byte granule at 0x3000, but the byte ranges are
  // disjoint, so the workgroups never touch the same memory.
  const auto warnings =
      run_cross_workgroup_global_accesses(engine, {0x3000, 1, true}, {0x3001, 1, true});

  EXPECT_TRUE(warnings.empty())
      << "byte 0x3000 and byte 0x3001 do not overlap, so this is not a race";
}

TEST(GenericDataHazardEngineTest, OverlappingSubDwordWritesRace) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  const auto warnings =
      run_cross_workgroup_global_accesses(engine, {0x3000, 1, true}, {0x3000, 1, true});

  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
}

TEST(GenericDataHazardEngineTest, PartiallyOverlappingSubDwordWritesRace) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  // Bytes 0-1 against bytes 1-2: they share byte 0x3001.
  const auto warnings =
      run_cross_workgroup_global_accesses(engine, {0x3000, 2, true}, {0x3001, 2, true});

  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
}

TEST(GenericDataHazardEngineTest, DisjointSubDwordWriteAfterReadDoesNotRace) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  const auto warnings =
      run_cross_workgroup_global_accesses(engine, {0x3002, 1, false}, {0x3003, 1, true});

  EXPECT_TRUE(warnings.empty()) << "a write only races a read that covers the same byte";
}

TEST(GenericDataHazardEngineTest, SubDwordAccessMasksApplyPerGranuleOfASpanningAccess) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  // 0x3002..0x3005 covers the top half of granule 0x3000 and the bottom half of
  // granule 0x3004, so each granule needs its own mask rather than the first.
  const auto disjoint =
      run_cross_workgroup_global_accesses(engine, {0x3002, 4, true}, {0x3001, 1, true});
  EXPECT_TRUE(disjoint.empty()) << "byte 0x3001 lies below the spanning write";

  auto &overlap_engine = reset_generic_engine(formatter);
  const auto overlap =
      run_cross_workgroup_global_accesses(overlap_engine, {0x3002, 4, true}, {0x3005, 1, true});
  ASSERT_EQ(overlap.size(), 1u);
  EXPECT_EQ(overlap[0].finding.kind, HazardKind::GlobalMemoryRace);
}

TEST(GenericDataHazardEngineTest, DisjointWriterDoesNotDisplaceTheWriterItDoesNotOverlap) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  // Workgroup 1 writes a byte that workgroup 0 never touched, so it takes a
  // slot of its own. Retaining workgroup 0 is what lets workgroup 2 still see
  // the byte it truly conflicts over.
  const std::array<SubDwordAccess, 3> writes{SubDwordAccess{0x3000, 1, true},
                                             SubDwordAccess{0x3001, 1, true},
                                             SubDwordAccess{0x3000, 1, true}};

  for (EntityId workgroup = 0; workgroup < writes.size(); ++workgroup) {
    engine.on_workgroup_begin(1, 0, workgroup);
    ExecutionKey wave{1, 0, workgroup, 0, 0};
    engine.on_wave_begin(wave);

    InstructionEvent inst;
    inst.instruction = make_instruction(workgroup + 1, 0x100 * (workgroup + 1));
    inst.instruction.execution = wave;
    engine.on_instruction(inst);

    ResourceAccessEvent event;
    event.instruction = inst.instruction;
    event.resource_kind = ResourceKind::GlobalMemory;
    event.address = writes[workgroup].address;
    event.size_bytes = writes[workgroup].size_bytes;
    event.is_write = true;
    engine.on_resource_access(event);

    if (workgroup == 1)
      EXPECT_TRUE(engine.warning_snapshot().empty()) << "byte 0x3001 overlaps nothing";
  }

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u) << "workgroup 2 overwrites the byte workgroup 0 wrote";
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
  EXPECT_EQ(warnings[0].finding.source_instruction.execution.workgroup_id, 0u);
}

TEST(GenericDataHazardEngineTest, RejectsInvalidGlobalAccessSizes) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent inst;
  inst.instruction = make_instruction(1, 0x100);
  inst.instruction.execution = wave;
  engine.on_instruction(inst);

  ResourceAccessEvent access;
  access.instruction = inst.instruction;
  access.resource_kind = ResourceKind::GlobalMemory;
  access.address = 0x1000;
  access.is_read = true;

  std::vector<std::string> diagnostics;
  engine.set_diagnostic_handler(
      [&](const std::string &message) { diagnostics.push_back(message); });

  access.size_bytes = 0;
  EXPECT_NO_THROW(engine.on_resource_access(access));

  access.size_bytes = hazard_core::MAX_ACCESS_SIZE_BYTES + 1;
  EXPECT_NO_THROW(engine.on_resource_access(access));

  EXPECT_EQ(engine.rejected_event_count(), 2u);
  EXPECT_EQ(diagnostics.size(), 2u);
  EXPECT_TRUE(engine.warning_snapshot().empty());
  engine.set_diagnostic_handler(nullptr);
}

TEST(GenericDataHazardEngineTest, RejectsOutOfRangeLocalMemoryAddress) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent inst;
  inst.instruction = make_instruction(1, 0x100);
  inst.hazards.local_write_wait = WaitCntType::LDS;
  inst.instruction.execution = wave;
  engine.on_instruction(inst);

  ResourceAccessEvent access;
  access.instruction = inst.instruction;
  access.resource_kind = ResourceKind::LocalMemory;
  access.address = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  access.size_bytes = 4;
  access.is_write = true;

  EXPECT_NO_THROW(engine.on_resource_access(access));
  EXPECT_EQ(engine.rejected_event_count(), 1u);
  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, HandlesGlobalAccessNearUint64MaxWithoutOverflowingLoop) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);
  engine.on_workgroup_begin(1, 0, 1);

  ExecutionKey writer_wave{1, 0, 0, 0, 0};
  ExecutionKey reader_wave{1, 0, 1, 0, 0};
  engine.on_wave_begin(writer_wave);
  engine.on_wave_begin(reader_wave);

  InstructionEvent write_inst;
  write_inst.instruction = make_instruction(1, 0x100);
  write_inst.instruction.execution = writer_wave;
  engine.on_instruction(write_inst);

  ResourceAccessEvent write_access;
  write_access.instruction = write_inst.instruction;
  write_access.resource_kind = ResourceKind::GlobalMemory;
  write_access.address = std::numeric_limits<uint64_t>::max() - 1;
  write_access.size_bytes = 4;
  write_access.is_write = true;
  engine.on_resource_access(write_access);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(2, 0x200);
  read_inst.instruction.execution = reader_wave;
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::GlobalMemory;
  read_access.address = std::numeric_limits<uint64_t>::max() - 1;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
  EXPECT_EQ(warnings[0].finding.address, std::numeric_limits<uint64_t>::max() - 3);
}

TEST(GenericDataHazardEngineTest, TracksVmemGuardedLocalWriteWithLoadcnt) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent direct_to_lds;
  direct_to_lds.instruction = make_instruction(1, 0x100);
  direct_to_lds.hazards.local_write_wait = WaitCntType::VMEM;
  engine.on_instruction(direct_to_lds);

  ResourceAccessEvent local_write;
  local_write.instruction = direct_to_lds.instruction;
  local_write.resource_kind = ResourceKind::LocalMemory;
  local_write.address = 0x180;
  local_write.size_bytes = 4;
  local_write.is_write = true;
  engine.on_resource_access(local_write);

  const auto snapshot = engine.wave_snapshot(wave);
  ASSERT_TRUE(snapshot.has_value());
  ASSERT_EQ(snapshot->core.lds_fifo.size(), 1u);
  EXPECT_EQ(snapshot->core.lds_fifo.front().second.wait_type, WaitCntType::VMEM);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(2, 0x104);
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::LocalMemory;
  read_access.address = 0x180;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_loadcnt 0 before reading this LDS address");
}

TEST(GenericDataHazardEngineTest, LoadcntClearsVmemGuardedLocalWrite) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent direct_to_lds;
  direct_to_lds.instruction = make_instruction(1, 0x100);
  direct_to_lds.hazards.local_write_wait = WaitCntType::VMEM;
  engine.on_instruction(direct_to_lds);

  ResourceAccessEvent local_write;
  local_write.instruction = direct_to_lds.instruction;
  local_write.resource_kind = ResourceKind::LocalMemory;
  local_write.address = 0x180;
  local_write.size_bytes = 4;
  local_write.is_write = true;
  engine.on_resource_access(local_write);

  InstructionEvent wait;
  wait.instruction = make_instruction(2, 0x104);
  wait.wait_action.is_wait_instruction = true;
  wait.wait_action.counters.push_back({WaitCntType::VMEM, 0});
  engine.on_instruction(wait);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(3, 0x108);
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::LocalMemory;
  read_access.address = 0x180;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, TracksTensorLocalWriteWithTensorWait) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);
  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(wave);

  InstructionEvent tensor_write_inst;
  tensor_write_inst.instruction = make_instruction(1, 0x100);
  tensor_write_inst.hazards.local_write_wait = WaitCntType::TENSOR;
  engine.on_instruction(tensor_write_inst);

  ResourceAccessEvent tensor_write;
  tensor_write.instruction = tensor_write_inst.instruction;
  tensor_write.resource_kind = ResourceKind::LocalMemory;
  tensor_write.address = 0x180;
  tensor_write.size_bytes = 4;
  tensor_write.is_write = true;
  engine.on_resource_access(tensor_write);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(2, 0x104);
  read_inst.hazards.local_write_wait = WaitCntType::LDS;
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::LocalMemory;
  read_access.address = 0x180;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::TENSOR);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_tensorcnt 0 before reading this LDS address");

  InstructionEvent wait;
  wait.instruction = make_instruction(3, 0x108);
  wait.wait_action.is_wait_instruction = true;
  wait.wait_action.counters.push_back({WaitCntType::TENSOR, 0});
  engine.on_instruction(wait);

  InstructionEvent second_read_inst;
  second_read_inst.instruction = make_instruction(4, 0x10c);
  second_read_inst.hazards.local_write_wait = WaitCntType::LDS;
  engine.on_instruction(second_read_inst);

  read_access.instruction = second_read_inst.instruction;
  engine.on_resource_access(read_access);

  warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
}

TEST(GenericDataHazardEngineTest, DoesNotMixLdsRaceEpochsAcrossDispatches) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_workgroup_begin(1, 0, 0);
  engine.on_workgroup_begin(2, 0, 0);

  ExecutionKey dispatch_one_wave{1, 0, 0, 0, 0};
  ExecutionKey dispatch_two_wave{2, 0, 0, 0, 1};
  engine.on_wave_begin(dispatch_one_wave);
  engine.on_wave_begin(dispatch_two_wave);

  InstructionEvent dispatch_one_write;
  dispatch_one_write.instruction = make_instruction(1, 0x100);
  dispatch_one_write.instruction.execution = dispatch_one_wave;
  engine.on_instruction(dispatch_one_write);

  ResourceAccessEvent lds_write;
  lds_write.instruction = dispatch_one_write.instruction;
  lds_write.resource_kind = ResourceKind::LocalMemory;
  lds_write.address = 0x80;
  lds_write.size_bytes = 4;
  lds_write.is_write = true;
  engine.on_resource_access(lds_write);

  InstructionEvent dispatch_two_read;
  dispatch_two_read.instruction = make_instruction(2, 0x200);
  dispatch_two_read.instruction.execution = dispatch_two_wave;
  engine.on_instruction(dispatch_two_read);

  ResourceAccessEvent lds_read;
  lds_read.instruction = dispatch_two_read.instruction;
  lds_read.resource_kind = ResourceKind::LocalMemory;
  lds_read.address = 0x80;
  lds_read.size_bytes = 4;
  lds_read.is_read = true;
  engine.on_resource_access(lds_read);

  engine.on_workgroup_end(1, 0, 0);
  engine.on_workgroup_end(2, 0, 0);

  EXPECT_TRUE(engine.warning_snapshot().empty());
}

TEST(GenericDataHazardEngineTest, WorkgroupBarrierFlushesOnlyMatchingDispatchEpoch) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_workgroup_begin(1, 0, 0);
  engine.on_workgroup_begin(2, 0, 0);

  ExecutionKey dispatch_one_wave_a{1, 0, 0, 0, 0};
  ExecutionKey dispatch_one_wave_b{1, 0, 0, 0, 1};
  ExecutionKey dispatch_two_wave_a{2, 0, 0, 0, 0};
  ExecutionKey dispatch_two_wave_b{2, 0, 0, 0, 1};
  engine.on_wave_begin(dispatch_one_wave_a);
  engine.on_wave_begin(dispatch_one_wave_b);
  engine.on_wave_begin(dispatch_two_wave_a);
  engine.on_wave_begin(dispatch_two_wave_b);

  auto record_lds_access = [&](EntityId instruction_id, const ExecutionKey &wave, bool is_write) {
    InstructionEvent instruction;
    instruction.instruction = make_instruction(instruction_id, 0x100 + instruction_id * 4);
    instruction.instruction.execution = wave;
    engine.on_instruction(instruction);

    ResourceAccessEvent access;
    access.instruction = instruction.instruction;
    access.resource_kind = ResourceKind::LocalMemory;
    access.address = 0x100;
    access.size_bytes = 4;
    access.is_read = !is_write;
    access.is_write = is_write;
    engine.on_resource_access(access);
  };

  record_lds_access(1, dispatch_one_wave_a, true);
  record_lds_access(2, dispatch_one_wave_b, false);
  record_lds_access(3, dispatch_two_wave_a, true);
  record_lds_access(4, dispatch_two_wave_b, false);

  BarrierEvent barrier;
  barrier.wave = dispatch_one_wave_a;
  barrier.kind = BarrierKind::Workgroup;
  engine.on_barrier(barrier);

  auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.instruction.execution.dispatch_id, 1u);

  engine.on_workgroup_end(2, 0, 0);

  warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  EXPECT_EQ(warnings[1].finding.instruction.execution.dispatch_id, 2u);
}

TEST(GenericDataHazardEngineTest, DispatchlessWorkgroupBarrierFlushesAllMatchingDispatchEpochs) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_workgroup_begin(1, 0, 0);
  engine.on_dispatch_begin(2);
  engine.on_workgroup_begin(2, 0, 0);

  ExecutionKey dispatch_one_wave_a{1, 0, 0, 0, 0};
  ExecutionKey dispatch_one_wave_b{1, 0, 0, 0, 1};
  ExecutionKey dispatch_two_wave_a{2, 0, 0, 0, 0};
  ExecutionKey dispatch_two_wave_b{2, 0, 0, 0, 1};
  engine.on_wave_begin(dispatch_one_wave_a);
  engine.on_wave_begin(dispatch_one_wave_b);
  engine.on_wave_begin(dispatch_two_wave_a);
  engine.on_wave_begin(dispatch_two_wave_b);

  auto record_lds_access = [&](EntityId instruction_id, const ExecutionKey &wave, bool is_write) {
    InstructionEvent instruction;
    instruction.instruction = make_instruction(instruction_id, 0x100 + instruction_id * 4);
    instruction.instruction.execution = wave;
    engine.on_instruction(instruction);

    ResourceAccessEvent access;
    access.instruction = instruction.instruction;
    access.resource_kind = ResourceKind::LocalMemory;
    access.address = 0x100;
    access.size_bytes = 4;
    access.is_read = !is_write;
    access.is_write = is_write;
    engine.on_resource_access(access);
  };

  record_lds_access(1, dispatch_one_wave_a, true);
  record_lds_access(2, dispatch_one_wave_b, false);
  record_lds_access(3, dispatch_two_wave_a, true);
  record_lds_access(4, dispatch_two_wave_b, false);

  BarrierEvent barrier;
  barrier.wave.cluster_id = 0;
  barrier.wave.workgroup_id = 0;
  barrier.wave.wave_id = 0;
  barrier.kind = BarrierKind::Workgroup;
  engine.on_barrier(barrier);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  std::set<EntityId> dispatch_ids;
  dispatch_ids.insert(warnings[0].finding.instruction.execution.dispatch_id);
  dispatch_ids.insert(warnings[1].finding.instruction.execution.dispatch_id);
  EXPECT_EQ(dispatch_ids, (std::set<EntityId>{1, 2}));
}

TEST(GenericDataHazardEngineTest, DispatchlessLocalMemoryAtomicBarrierClearsAllMatchingWaves) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_workgroup_begin(1, 0, 0);
  engine.on_dispatch_begin(2);
  engine.on_workgroup_begin(2, 0, 0);

  ExecutionKey dispatch_one_wave{1, 0, 0, 0, 0};
  ExecutionKey dispatch_two_wave{2, 0, 0, 0, 0};
  engine.on_wave_begin(dispatch_one_wave);
  engine.on_wave_begin(dispatch_two_wave);

  auto record_lds_write = [&](EntityId instruction_id, const ExecutionKey &wave) {
    InstructionEvent instruction;
    instruction.instruction = make_instruction(instruction_id, 0x100 + instruction_id * 4);
    instruction.instruction.execution = wave;
    instruction.hazards.local_write_wait = WaitCntType::LDS;
    engine.on_instruction(instruction);

    ResourceAccessEvent access;
    access.instruction = instruction.instruction;
    access.resource_kind = ResourceKind::LocalMemory;
    access.address = 0x100;
    access.size_bytes = 4;
    access.is_write = true;
    engine.on_resource_access(access);
  };

  record_lds_write(1, dispatch_one_wave);
  record_lds_write(2, dispatch_two_wave);

  ASSERT_EQ(engine.wave_snapshot(dispatch_one_wave)->core.lds_fifo.size(), 1u);
  ASSERT_EQ(engine.wave_snapshot(dispatch_two_wave)->core.lds_fifo.size(), 1u);

  BarrierEvent barrier;
  barrier.wave.cluster_id = 0;
  barrier.wave.workgroup_id = 0;
  barrier.wave.wave_id = 0;
  barrier.kind = BarrierKind::LocalMemoryAtomic;
  engine.on_barrier(barrier);

  EXPECT_EQ(engine.wave_snapshot(dispatch_one_wave)->core.lds_fifo.size(), 0u);
  EXPECT_EQ(engine.wave_snapshot(dispatch_two_wave)->core.lds_fifo.size(), 0u);
}

TEST(GenericDataHazardEngineTest, DispatchBeginDoesNotResetOtherDispatchState) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_workgroup_begin(1, 0, 0);
  ExecutionKey dispatch_one_wave{1, 0, 0, 0, 0};
  engine.on_wave_begin(dispatch_one_wave);

  InstructionEvent load;
  load.instruction = make_instruction(1, 0x100);
  load.instruction.execution = dispatch_one_wave;
  load.hazards.vector_write_wait = WaitCntType::VMEM;
  engine.on_instruction(load);

  ResourceAccessEvent load_write;
  load_write.instruction = load.instruction;
  load_write.resource_kind = ResourceKind::VectorRegister;
  load_write.register_kind = RegisterKind::Vector;
  load_write.resource_index = 4;
  load_write.size_bytes = 4;
  load_write.is_write = true;
  engine.on_resource_access(load_write);

  engine.on_dispatch_begin(2);

  InstructionEvent read;
  read.instruction = make_instruction(2, 0x104);
  read.instruction.execution = dispatch_one_wave;
  engine.on_instruction(read);

  ResourceAccessEvent read_access;
  read_access.instruction = read.instruction;
  read_access.resource_kind = ResourceKind::VectorRegister;
  read_access.register_kind = RegisterKind::Vector;
  read_access.resource_index = 4;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.instruction.execution.dispatch_id, 1u);
}

TEST(GenericDataHazardEngineTest, DispatchEndFlushesAndErasesOnlyMatchingDispatch) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_dispatch_begin(2);
  engine.on_workgroup_begin(1, 0, 0);
  engine.on_workgroup_begin(2, 0, 0);

  auto record_lds_access = [&](EntityId instruction_id, const ExecutionKey &wave, bool is_write) {
    engine.on_wave_begin(wave);

    InstructionEvent instruction;
    instruction.instruction = make_instruction(instruction_id, 0x100 + instruction_id * 4);
    instruction.instruction.execution = wave;
    engine.on_instruction(instruction);

    ResourceAccessEvent access;
    access.instruction = instruction.instruction;
    access.resource_kind = ResourceKind::LocalMemory;
    access.address = 0x100;
    access.size_bytes = 4;
    access.is_read = !is_write;
    access.is_write = is_write;
    engine.on_resource_access(access);
  };

  record_lds_access(1, ExecutionKey{1, 0, 0, 0, 0}, true);
  record_lds_access(2, ExecutionKey{1, 0, 0, 0, 1}, false);
  record_lds_access(3, ExecutionKey{2, 0, 0, 0, 0}, true);
  record_lds_access(4, ExecutionKey{2, 0, 0, 0, 1}, false);

  ASSERT_EQ(engine.wave_count(), 4u);

  engine.on_dispatch_end(1);

  auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.instruction.execution.dispatch_id, 1u);
  EXPECT_EQ(engine.wave_count(), 2u);

  engine.on_dispatch_end(2);

  warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  EXPECT_EQ(warnings[1].finding.instruction.execution.dispatch_id, 2u);
  EXPECT_EQ(engine.wave_count(), 0u);
}

TEST(GenericDataHazardEngineTest, ShutdownFlushesOutstandingWorkgroupEpochs) {
  FakeFormatter formatter;
  auto &engine = reset_generic_engine(formatter);

  engine.on_workgroup_begin(1, 0, 0);

  ExecutionKey writer_wave{1, 0, 0, 0, 0};
  ExecutionKey reader_wave{1, 0, 0, 0, 1};
  engine.on_wave_begin(writer_wave);
  engine.on_wave_begin(reader_wave);

  InstructionEvent write_inst;
  write_inst.instruction = make_instruction(1, 0x100);
  write_inst.instruction.execution = writer_wave;
  engine.on_instruction(write_inst);

  ResourceAccessEvent write_access;
  write_access.instruction = write_inst.instruction;
  write_access.resource_kind = ResourceKind::LocalMemory;
  write_access.address = 0x80;
  write_access.size_bytes = 4;
  write_access.is_write = true;
  engine.on_resource_access(write_access);

  InstructionEvent read_inst;
  read_inst.instruction = make_instruction(2, 0x104);
  read_inst.instruction.execution = reader_wave;
  engine.on_instruction(read_inst);

  ResourceAccessEvent read_access;
  read_access.instruction = read_inst.instruction;
  read_access.resource_kind = ResourceKind::LocalMemory;
  read_access.address = 0x80;
  read_access.size_bytes = 4;
  read_access.is_read = true;
  engine.on_resource_access(read_access);

  ASSERT_TRUE(engine.warning_snapshot().empty());

  engine.on_shutdown();

  const auto warnings = engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::LocalMemoryRace);
  EXPECT_EQ(warnings[0].finding.instruction.execution.dispatch_id, 1u);
}
