// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "data_hazard_engine.h"
#include "rocjitsu/vm/plugins/data_hazard/adapter.h"

#include <cstddef>
#include <string>
#include <vector>

using namespace hazard_core;

namespace dh = rocjitsu::plugins::data_hazard;

namespace {

using hazard_core::WaitCntType;

class FakeFormatter final : public SimulatorInstructionFormatter {
public:
  std::string format_instruction(const InstructionDescriptor &instruction) const override {
    return "rocjitsu_inst_" + std::to_string(instruction.instruction_id);
  }

  /// Overrides register wording only, so tests of LDS-address wording see the
  /// shared text.
  std::string format_wait_suggestion(WaitCntType kind, HazardAccessKind access,
                                     hazard_core::HazardResourceLabel resource) const override {
    (void)access;
    if (resource != hazard_core::HazardResourceLabel::Register)
      return "";
    switch (kind) {
    case WaitCntType::VMEM:
      return "wait loadcnt";
    case WaitCntType::STORE:
      return "wait storecnt";
    case WaitCntType::LDS:
      return "wait dscnt";
    default:
      return "";
    }
  }
};

struct AdapterHarness {
  FakeFormatter formatter;
  DataHazardEngine &engine = data_hazard_engine();
  dh::DataHazardAdapter adapter{engine};
  ExecutionKey wave{1, 0, 0, 0, 0};

  AdapterHarness() {
    engine.reset();
    engine.set_instruction_formatter(&formatter);
    engine.set_warning_sink(nullptr);
    adapter.reset();
    adapter.on_dispatch_begin(wave.dispatch_id);
    adapter.on_workgroup_begin(wave);
    adapter.on_wave_begin(wave);
  }

  dh::InstructionView instruction(hazard_core::EntityId id, uint64_t pc) const {
    dh::InstructionView inst;
    inst.execution = wave;
    inst.instruction_id = id;
    inst.pc = pc;
    inst.raw_isa[0] = static_cast<uint32_t>(0xCAFE0000u | id);
    return inst;
  }
};

class RecordingSimulatorApi final : public DataHazardSimulatorApi {
public:
  void on_dispatch_begin(hazard_core::EntityId dispatch_id) override {
    dispatch_begin_ids.push_back(dispatch_id);
  }

  void on_dispatch_end(hazard_core::EntityId dispatch_id) override {
    dispatch_end_ids.push_back(dispatch_id);
  }

  void on_workgroup_begin(hazard_core::EntityId dispatch_id, hazard_core::EntityId cluster_id,
                          hazard_core::EntityId workgroup_id) override {
    workgroup_begin_keys.push_back({dispatch_id, cluster_id, workgroup_id, 0, 0});
  }

  void on_workgroup_end(hazard_core::EntityId dispatch_id, hazard_core::EntityId cluster_id,
                        hazard_core::EntityId workgroup_id) override {
    workgroup_end_keys.push_back({dispatch_id, cluster_id, workgroup_id, 0, 0});
  }

  void on_wave_begin(const ExecutionKey &wave) override { wave_begin_keys.push_back(wave); }

  void on_wave_end(const ExecutionKey &wave) override { wave_end_keys.push_back(wave); }

  void on_instruction(const InstructionEvent &instruction) override {
    instructions.push_back(instruction);
  }

  void on_resource_access(const ResourceAccessEvent &access) override {
    resources.push_back(access);
  }

  void on_barrier(const BarrierEvent &barrier) override { barriers.push_back(barrier); }

  void on_shutdown() override { shutdown_count++; }

  std::vector<hazard_core::EntityId> dispatch_begin_ids;
  std::vector<hazard_core::EntityId> dispatch_end_ids;
  std::vector<ExecutionKey> workgroup_begin_keys;
  std::vector<ExecutionKey> workgroup_end_keys;
  std::vector<ExecutionKey> wave_begin_keys;
  std::vector<ExecutionKey> wave_end_keys;
  std::vector<InstructionEvent> instructions;
  std::vector<ResourceAccessEvent> resources;
  std::vector<BarrierEvent> barriers;
  uint32_t shutdown_count = 0;
};

const WaitCounterClear *find_counter(const WaitAction &action, WaitCntType type) {
  for (const auto &counter : action.counters) {
    if (counter.kind == type)
      return &counter;
  }
  return nullptr;
}

void expect_key(const ExecutionKey &key, hazard_core::EntityId dispatch_id,
                hazard_core::EntityId cluster_id, hazard_core::EntityId workgroup_id,
                hazard_core::EntityId wavegroup_id, hazard_core::EntityId wave_id) {
  EXPECT_EQ(key.dispatch_id, dispatch_id);
  EXPECT_EQ(key.cluster_id, cluster_id);
  EXPECT_EQ(key.workgroup_id, workgroup_id);
  EXPECT_EQ(key.wavegroup_id, wavegroup_id);
  EXPECT_EQ(key.wave_id, wave_id);
}

} // namespace

TEST(DataHazardAdapterTest, LifecycleCallbacksMapToGenericApi) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  ExecutionKey wave{11, 3, 5, 7, 13};

  adapter.on_dispatch_begin(wave.dispatch_id);
  adapter.on_workgroup_begin(wave);
  adapter.on_wave_begin(wave);
  adapter.on_workgroup_barrier(wave);
  adapter.on_local_memory_atomic_barrier(wave, false);
  adapter.on_local_memory_atomic_barrier(wave, true);
  adapter.on_wave_end(wave);
  adapter.on_workgroup_end(wave);
  adapter.on_dispatch_end(wave.dispatch_id);
  adapter.on_shutdown();

  ASSERT_EQ(api.dispatch_begin_ids.size(), 1u);
  EXPECT_EQ(api.dispatch_begin_ids[0], 11u);
  ASSERT_EQ(api.workgroup_begin_keys.size(), 1u);
  expect_key(api.workgroup_begin_keys[0], 11, 3, 5, 0, 0);
  ASSERT_EQ(api.wave_begin_keys.size(), 1u);
  expect_key(api.wave_begin_keys[0], 11, 3, 5, 7, 13);

  ASSERT_EQ(api.barriers.size(), 3u);
  EXPECT_EQ(api.barriers[0].kind, BarrierKind::Workgroup);
  EXPECT_EQ(api.barriers[1].kind, BarrierKind::LocalMemoryAtomic);
  EXPECT_EQ(api.barriers[2].kind, BarrierKind::LocalMemoryAtomicAsync);
  expect_key(api.barriers[0].wave, 11, 3, 5, 7, 13);

  ASSERT_EQ(api.wave_end_keys.size(), 1u);
  expect_key(api.wave_end_keys[0], 11, 3, 5, 7, 13);
  ASSERT_EQ(api.workgroup_end_keys.size(), 1u);
  expect_key(api.workgroup_end_keys[0], 11, 3, 5, 0, 0);
  ASSERT_EQ(api.dispatch_end_ids.size(), 1u);
  EXPECT_EQ(api.dispatch_end_ids[0], 11u);
  EXPECT_EQ(api.shutdown_count, 1u);
}

TEST(DataHazardAdapterTest, InstructionCallbackMapsContextAndWaitAction) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  ExecutionKey wave{4, 2, 6, 1, 9};

  dh::InstructionView instruction;
  instruction.execution = wave;
  instruction.instruction_id = 42;
  instruction.pc = 0x1234;
  instruction.raw_isa = {0xDEAD, 0xBEEF, 0xCAFE, 0xBABE};
  instruction.wait.kind = dh::WaitKind::WaitLoadcntDscnt;
  instruction.wait.immediate = (1u << 8) | 3u; // loadcnt(1) dscnt(3)
  adapter.on_instruction(instruction);

  ASSERT_EQ(api.instructions.size(), 1u);
  const auto &event = api.instructions[0];
  EXPECT_EQ(event.instruction.instruction_id, 42u);
  EXPECT_EQ(event.instruction.pc, 0x1234u);
  EXPECT_EQ(event.instruction.raw_isa[0], 0xDEADu);
  expect_key(event.instruction.execution, 4, 2, 6, 1, 9);
  ASSERT_TRUE(event.wait_action.is_wait_instruction);
  ASSERT_EQ(event.wait_action.counters.size(), 2u);
  ASSERT_NE(find_counter(event.wait_action, WaitCntType::VMEM), nullptr);
  ASSERT_NE(find_counter(event.wait_action, WaitCntType::LDS), nullptr);
  EXPECT_EQ(find_counter(event.wait_action, WaitCntType::VMEM)->keep_count, 1u);
  EXPECT_EQ(find_counter(event.wait_action, WaitCntType::LDS)->keep_count, 3u);
}

TEST(DataHazardAdapterTest, RegisterCallbacksMapScalarVectorAccumAndDropUnknown) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  ExecutionKey wave{1, 0, 0, 0, 0};

  dh::InstructionView instruction;
  instruction.execution = wave;
  instruction.instruction_id = 1;
  instruction.pc = 0x100;
  adapter.on_instruction(instruction);

  dh::RegisterAccessView scalar;
  scalar.instruction = instruction;
  scalar.register_class = dh::RegisterClass::Scalar;
  scalar.physical_reg = 17;
  scalar.size_bytes = 8;
  scalar.is_read = true;
  adapter.on_register_access(scalar);

  dh::RegisterAccessView vector;
  vector.instruction = instruction;
  vector.register_class = dh::RegisterClass::Vector;
  vector.physical_reg = 23;
  vector.size_bytes = 16;
  vector.is_write = true;
  adapter.on_register_access(vector);

  dh::RegisterAccessView accum;
  accum.instruction = instruction;
  accum.register_class = dh::RegisterClass::AccumVector;
  accum.physical_reg = 31;
  accum.size_bytes = 4;
  accum.is_read = true;
  accum.is_write = true;
  adapter.on_register_access(accum);

  dh::RegisterAccessView unknown;
  unknown.instruction = instruction;
  unknown.register_class = dh::RegisterClass::None;
  unknown.physical_reg = 99;
  unknown.is_read = true;
  adapter.on_register_access(unknown);

  ASSERT_EQ(api.resources.size(), 3u);
  EXPECT_EQ(api.resources[0].resource_kind, ResourceKind::ScalarRegister);
  EXPECT_EQ(api.resources[0].register_kind, RegisterKind::Scalar);
  EXPECT_EQ(api.resources[0].resource_index, 17u);
  EXPECT_TRUE(api.resources[0].is_read);
  EXPECT_FALSE(api.resources[0].is_write);

  EXPECT_EQ(api.resources[1].resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(api.resources[1].register_kind, RegisterKind::Vector);
  EXPECT_EQ(api.resources[1].resource_index, 23u);
  EXPECT_FALSE(api.resources[1].is_read);
  EXPECT_TRUE(api.resources[1].is_write);

  EXPECT_EQ(api.resources[2].resource_kind, ResourceKind::AccumVectorRegister);
  EXPECT_EQ(api.resources[2].register_kind, RegisterKind::AccumVector);
  EXPECT_EQ(api.resources[2].resource_index, 31u);
  EXPECT_TRUE(api.resources[2].is_read);
  EXPECT_TRUE(api.resources[2].is_write);
}

TEST(DataHazardAdapterTest, GlobalFlatLoadRouteEmitsGlobalReadAndDualCounterVectorDestination) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  ExecutionKey wave{1, 0, 0, 0, 0};

  dh::InstructionView instruction;
  instruction.execution = wave;
  instruction.instruction_id = 1;
  instruction.pc = 0x100;
  adapter.on_instruction(instruction);

  dh::MemoryRouteView route;
  route.instruction = instruction;
  route.resource_kind = ResourceKind::GlobalMemory;
  route.register_class = dh::RegisterClass::Vector;
  route.register_base = 12;
  route.address = 0x4000;
  route.size_bytes = 16;
  route.is_load = true;
  route.is_flat = true;
  route.exec_mask = 0x55;
  adapter.on_memory_route(route);

  ASSERT_EQ(api.resources.size(), 2u);
  EXPECT_EQ(api.resources[0].resource_kind, ResourceKind::GlobalMemory);
  EXPECT_EQ(api.resources[0].address, 0x4000u);
  EXPECT_EQ(api.resources[0].size_bytes, 16u);
  EXPECT_EQ(api.resources[0].exec_mask, 0x55u);
  EXPECT_TRUE(api.resources[0].is_read);
  EXPECT_FALSE(api.resources[0].is_write);

  EXPECT_EQ(api.resources[1].resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(api.resources[1].register_kind, RegisterKind::Vector);
  EXPECT_EQ(api.resources[1].resource_index, 12u);
  EXPECT_EQ(api.resources[1].hazards.write_wait, WaitCntType::VMEM);
  EXPECT_TRUE(api.resources[1].hazards.write_also_waits_lds);
}

TEST(DataHazardAdapterTest, GlobalLoadRouteTracksVectorDestinationForRawHazards) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView load;
  load.instruction = h.instruction(1, 0x100);
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 6;
  load.address = 0x2000;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView read;
  read.instruction = h.instruction(2, 0x104);
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 6;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);
  EXPECT_EQ(warnings[0].finding.source_instruction.raw_isa[0], 0xCAFE0001u);
}

TEST(DataHazardAdapterTest, GlobalStoreRouteDoesNotTrackVectorSourceForWarHazards) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView store;
  store.instruction = h.instruction(1, 0x100);
  store.resource_kind = ResourceKind::GlobalMemory;
  store.register_class = dh::RegisterClass::Vector;
  store.register_base = 5;
  store.address = 0x2000;
  store.size_bytes = 4;
  store.is_store = true;
  h.adapter.on_memory_route(store);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView write;
  write.instruction = h.instruction(2, 0x104);
  write.register_class = dh::RegisterClass::Vector;
  write.physical_reg = 5;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, ScratchLoadRouteEmitsScratchMemoryAndVectorDestinationEvents) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  dh::InstructionView instruction;
  dh::MemoryRouteView load;
  ExecutionKey const wave{1, 0, 0, 0, 0};

  instruction.execution = wave;
  instruction.instruction_id = 1;
  instruction.pc = 0x100;
  adapter.on_instruction(instruction);

  load.instruction = instruction;
  load.resource_kind = ResourceKind::ScratchMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 6;
  load.address = 0x7000;
  load.size_bytes = 16;
  load.is_load = true;
  load.exec_mask = 0x3;
  adapter.on_memory_route(load);

  ASSERT_EQ(api.resources.size(), 2u);
  EXPECT_EQ(api.resources[0].resource_kind, ResourceKind::ScratchMemory);
  EXPECT_EQ(api.resources[0].address, 0x7000u);
  EXPECT_EQ(api.resources[0].size_bytes, 16u);
  EXPECT_EQ(api.resources[0].exec_mask, 0x3u);
  EXPECT_TRUE(api.resources[0].is_read);
  EXPECT_FALSE(api.resources[0].is_write);

  EXPECT_EQ(api.resources[1].resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(api.resources[1].register_kind, RegisterKind::Vector);
  EXPECT_EQ(api.resources[1].resource_index, 6u);
  EXPECT_EQ(api.resources[1].size_bytes, 16u);
  EXPECT_TRUE(api.resources[1].is_write);
  EXPECT_EQ(api.resources[1].hazards.write_wait, WaitCntType::VMEM);
}

TEST(DataHazardAdapterTest, ScratchStorePerLaneRouteEmitsActiveScratchWritesOnly) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  dh::InstructionView instruction;
  dh::MemoryRouteView store;
  ExecutionKey const wave{1, 0, 0, 0, 0};

  instruction.execution = wave;
  instruction.instruction_id = 1;
  instruction.pc = 0x100;
  adapter.on_instruction(instruction);

  store.instruction = instruction;
  store.resource_kind = ResourceKind::ScratchMemory;
  store.register_class = dh::RegisterClass::Vector;
  store.register_base = 8;
  store.per_lane_addresses = {0x7100, 0x7200, 0x7300};
  store.size_bytes = 4;
  store.is_store = true;
  store.exec_mask = 0x5;
  adapter.on_memory_route(store);

  // One counter slot for the instruction, then the writes of the two active
  // lanes. The route did not name a counter, so the store takes a slot of the
  // separate store counter this adapter defaults to.
  ASSERT_EQ(api.resources.size(), 3u);
  EXPECT_EQ(api.resources[0].hazards.memory_op_wait, WaitCntType::STORE);
  EXPECT_FALSE(api.resources[0].is_write);
  EXPECT_FALSE(api.resources[0].is_read);
  EXPECT_EQ(api.resources[1].resource_kind, ResourceKind::ScratchMemory);
  EXPECT_EQ(api.resources[1].address, 0x7100u);
  EXPECT_TRUE(api.resources[1].is_write);
  EXPECT_FALSE(api.resources[1].is_read);
  EXPECT_EQ(api.resources[2].resource_kind, ResourceKind::ScratchMemory);
  EXPECT_EQ(api.resources[2].address, 0x7300u);
  EXPECT_TRUE(api.resources[2].is_write);
  EXPECT_FALSE(api.resources[2].is_read);
}

TEST(DataHazardAdapterTest, ReturningAtomicRouteEmitsMemoryRmwAndReturnEvents) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  dh::InstructionView instruction;
  dh::MemoryRouteView atomic;
  ExecutionKey const wave{1, 0, 0, 0, 0};

  instruction.execution = wave;
  instruction.instruction_id = 1;
  instruction.pc = 0x100;
  adapter.on_instruction(instruction);

  atomic.instruction = instruction;
  atomic.resource_kind = ResourceKind::GlobalMemory;
  atomic.register_class = dh::RegisterClass::Vector;
  atomic.register_base = 12;
  atomic.address = 0x8000;
  atomic.size_bytes = 4;
  atomic.is_atomic = true;
  atomic.is_load = true;
  adapter.on_memory_route(atomic);

  ASSERT_EQ(api.resources.size(), 2u);
  EXPECT_EQ(api.resources[0].resource_kind, ResourceKind::GlobalMemory);
  EXPECT_EQ(api.resources[0].address, 0x8000u);
  EXPECT_TRUE(api.resources[0].is_atomic);
  EXPECT_TRUE(api.resources[0].is_read);
  EXPECT_TRUE(api.resources[0].is_write);

  EXPECT_EQ(api.resources[1].resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(api.resources[1].resource_index, 12u);
  EXPECT_TRUE(api.resources[1].is_write);
  EXPECT_EQ(api.resources[1].hazards.write_wait, WaitCntType::VMEM);
}

TEST(DataHazardAdapterTest, AtomicGlobalRouteDoesNotTrackVectorSourceForWarHazards) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView atomic;
  atomic.instruction = h.instruction(1, 0x100);
  atomic.resource_kind = ResourceKind::GlobalMemory;
  atomic.register_class = dh::RegisterClass::Vector;
  atomic.register_base = 9;
  atomic.address = 0x4000;
  atomic.size_bytes = 4;
  atomic.is_atomic = true;
  h.adapter.on_memory_route(atomic);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView write;
  write.instruction = h.instruction(2, 0x104);
  write.register_class = dh::RegisterClass::Vector;
  write.physical_reg = 9;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, StorecntClearsAtomicSourceBeforeVectorWrite) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView atomic;
  atomic.instruction = h.instruction(1, 0x100);
  atomic.resource_kind = ResourceKind::GlobalMemory;
  atomic.register_class = dh::RegisterClass::Vector;
  atomic.register_base = 9;
  atomic.address = 0x4000;
  atomic.size_bytes = 4;
  atomic.is_atomic = true;
  h.adapter.on_memory_route(atomic);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitStorecnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::RegisterAccessView write;
  write.instruction = h.instruction(3, 0x108);
  write.register_class = dh::RegisterClass::Vector;
  write.physical_reg = 9;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, ReturningAtomicGlobalRouteTracksVectorDestinationForRawHazards) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView atomic;
  atomic.instruction = h.instruction(1, 0x100);
  atomic.resource_kind = ResourceKind::GlobalMemory;
  atomic.register_class = dh::RegisterClass::Vector;
  atomic.register_base = 10;
  atomic.address = 0x4000;
  atomic.size_bytes = 4;
  atomic.is_atomic = true;
  atomic.is_load = true;
  h.adapter.on_memory_route(atomic);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView read;
  read.instruction = h.instruction(2, 0x104);
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 10;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);
}

TEST(DataHazardAdapterTest, GlobalLoadRouteTracksVectorDestinationForWawHazards) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView load;
  load.instruction = h.instruction(1, 0x100);
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 6;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView write;
  write.instruction = h.instruction(2, 0x104);
  write.register_class = dh::RegisterClass::Vector;
  write.physical_reg = 6;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::WAW);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);
}

TEST(DataHazardAdapterTest, DestinationWriteBeforeRouteMatchesRocjitsuCallbackOrder) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView first_load;
  first_load.instruction = h.instruction(1, 0x100);
  first_load.resource_kind = ResourceKind::GlobalMemory;
  first_load.register_class = dh::RegisterClass::Vector;
  first_load.register_base = 8;
  first_load.size_bytes = 4;
  first_load.is_load = true;
  h.adapter.on_memory_route(first_load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView early_destination;
  early_destination.instruction = h.instruction(2, 0x104);
  early_destination.register_class = dh::RegisterClass::Vector;
  early_destination.physical_reg = 8;
  early_destination.size_bytes = 4;
  early_destination.is_write = true;
  h.adapter.on_register_access(early_destination);

  auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::WAW);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 1u);

  dh::MemoryRouteView second_load;
  second_load.instruction = h.instruction(2, 0x104);
  second_load.resource_kind = ResourceKind::GlobalMemory;
  second_load.register_class = dh::RegisterClass::Vector;
  second_load.register_base = 8;
  second_load.size_bytes = 4;
  second_load.is_load = true;
  h.adapter.on_memory_route(second_load);

  warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::RegisterAccessView later_read;
  later_read.instruction = h.instruction(3, 0x108);
  later_read.register_class = dh::RegisterClass::Vector;
  later_read.physical_reg = 8;
  later_read.size_bytes = 4;
  later_read.is_read = true;
  h.adapter.on_register_access(later_read);

  warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  EXPECT_EQ(warnings[1].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[1].finding.source_instruction.instruction_id, 2u);
  EXPECT_EQ(warnings[1].finding.source_instruction.pc, 0x104u);
}

TEST(DataHazardAdapterTest, StoreSourceReadBeforeRouteIsIgnoredAfterRouteReplay) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::RegisterAccessView early_source;
  early_source.instruction = h.instruction(1, 0x100);
  early_source.register_class = dh::RegisterClass::Vector;
  early_source.physical_reg = 5;
  early_source.size_bytes = 4;
  early_source.is_read = true;
  h.adapter.on_register_access(early_source);

  dh::MemoryRouteView store;
  store.instruction = h.instruction(1, 0x100);
  store.resource_kind = ResourceKind::GlobalMemory;
  store.register_class = dh::RegisterClass::Vector;
  store.register_base = 5;
  store.size_bytes = 4;
  store.is_store = true;
  h.adapter.on_memory_route(store);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView write;
  write.instruction = h.instruction(2, 0x104);
  write.register_class = dh::RegisterClass::Vector;
  write.physical_reg = 5;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, StorecntClearsGlobalStoreSourceBeforeVectorWrite) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView store;
  store.instruction = h.instruction(1, 0x100);
  store.resource_kind = ResourceKind::GlobalMemory;
  store.register_class = dh::RegisterClass::Vector;
  store.register_base = 5;
  store.size_bytes = 4;
  store.is_store = true;
  h.adapter.on_memory_route(store);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitStorecnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::RegisterAccessView write;
  write.instruction = h.instruction(3, 0x108);
  write.register_class = dh::RegisterClass::Vector;
  write.physical_reg = 5;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, LegacyWaitcntClearsBothVmemLoadsAndStores) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView load;
  load.instruction = h.instruction(1, 0x100);
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 4;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView store;
  store.instruction = h.instruction(2, 0x104);
  store.resource_kind = ResourceKind::GlobalMemory;
  store.register_class = dh::RegisterClass::Vector;
  store.register_base = 5;
  store.size_bytes = 4;
  store.is_store = true;
  h.adapter.on_memory_route(store);

  dh::InstructionView wait = h.instruction(3, 0x108);
  wait.wait.kind = dh::WaitKind::Waitcnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(4, 0x10c));

  dh::RegisterAccessView read;
  read.instruction = h.instruction(4, 0x10c);
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 4;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  dh::RegisterAccessView write;
  write.instruction = h.instruction(4, 0x10c);
  write.register_class = dh::RegisterClass::Vector;
  write.physical_reg = 5;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, PartialLegacyWaitcntKeepsNewestVmemLoad) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView first_load;
  first_load.instruction = h.instruction(1, 0x100);
  first_load.resource_kind = ResourceKind::GlobalMemory;
  first_load.register_class = dh::RegisterClass::Vector;
  first_load.register_base = 4;
  first_load.size_bytes = 4;
  first_load.is_load = true;
  h.adapter.on_memory_route(first_load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView second_load;
  second_load.instruction = h.instruction(2, 0x104);
  second_load.resource_kind = ResourceKind::GlobalMemory;
  second_load.register_class = dh::RegisterClass::Vector;
  second_load.register_base = 5;
  second_load.size_bytes = 4;
  second_load.is_load = true;
  h.adapter.on_memory_route(second_load);

  dh::InstructionView wait = h.instruction(3, 0x108);
  wait.wait.kind = dh::WaitKind::Waitcnt;
  wait.wait.immediate = 1;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(4, 0x10c));

  dh::RegisterAccessView read_first;
  read_first.instruction = h.instruction(4, 0x10c);
  read_first.register_class = dh::RegisterClass::Vector;
  read_first.physical_reg = 4;
  read_first.size_bytes = 4;
  read_first.is_read = true;
  h.adapter.on_register_access(read_first);

  dh::RegisterAccessView read_second = read_first;
  read_second.physical_reg = 5;
  h.adapter.on_register_access(read_second);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 2u);
}

TEST(DataHazardAdapterTest, SplitLoadcntDoesNotClearSmem) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView global_load;
  global_load.instruction = h.instruction(1, 0x100);
  global_load.resource_kind = ResourceKind::GlobalMemory;
  global_load.register_class = dh::RegisterClass::Vector;
  global_load.register_base = 4;
  global_load.size_bytes = 4;
  global_load.is_load = true;
  h.adapter.on_memory_route(global_load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView scalar_load;
  scalar_load.instruction = h.instruction(2, 0x104);
  scalar_load.resource_kind = ResourceKind::ScalarRegister;
  scalar_load.register_class = dh::RegisterClass::Scalar;
  scalar_load.register_base = 12;
  scalar_load.size_bytes = 4;
  scalar_load.is_load = true;
  h.adapter.on_memory_route(scalar_load);

  dh::InstructionView wait = h.instruction(3, 0x108);
  wait.wait.kind = dh::WaitKind::WaitLoadcnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(4, 0x10c));

  dh::RegisterAccessView read_vgpr;
  read_vgpr.instruction = h.instruction(4, 0x10c);
  read_vgpr.register_class = dh::RegisterClass::Vector;
  read_vgpr.physical_reg = 4;
  read_vgpr.size_bytes = 4;
  read_vgpr.is_read = true;
  h.adapter.on_register_access(read_vgpr);

  dh::RegisterAccessView read_sgpr;
  read_sgpr.instruction = h.instruction(4, 0x10c);
  read_sgpr.register_class = dh::RegisterClass::Scalar;
  read_sgpr.physical_reg = 12;
  read_sgpr.size_bytes = 4;
  read_sgpr.is_read = true;
  h.adapter.on_register_access(read_sgpr);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::ScalarRegister);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::SMEM);
}

TEST(DataHazardAdapterTest, SplitDscntClearsLdsVectorDestination) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView lds_read;
  lds_read.instruction = h.instruction(1, 0x100);
  lds_read.resource_kind = ResourceKind::LocalMemory;
  lds_read.register_class = dh::RegisterClass::Vector;
  lds_read.register_base = 8;
  lds_read.address = 0x100;
  lds_read.size_bytes = 4;
  lds_read.is_load = true;
  h.adapter.on_memory_route(lds_read);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitDscnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::RegisterAccessView read_vgpr;
  read_vgpr.instruction = h.instruction(3, 0x108);
  read_vgpr.register_class = dh::RegisterClass::Vector;
  read_vgpr.physical_reg = 8;
  read_vgpr.size_bytes = 4;
  read_vgpr.is_read = true;
  h.adapter.on_register_access(read_vgpr);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, SplitStorecntDscntClearsStoreSourceAndLocalWrite) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView store;
  store.instruction = h.instruction(1, 0x100);
  store.resource_kind = ResourceKind::GlobalMemory;
  store.register_class = dh::RegisterClass::Vector;
  store.register_base = 5;
  store.size_bytes = 4;
  store.is_store = true;
  h.adapter.on_memory_route(store);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView lds_write;
  lds_write.instruction = h.instruction(2, 0x104);
  lds_write.resource_kind = ResourceKind::LocalMemory;
  lds_write.address = 0x100;
  lds_write.size_bytes = 4;
  lds_write.is_store = true;
  h.adapter.on_memory_route(lds_write);

  dh::InstructionView wait = h.instruction(3, 0x108);
  wait.wait.kind = dh::WaitKind::WaitStorecntDscnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(4, 0x10c));

  dh::RegisterAccessView write_vgpr;
  write_vgpr.instruction = h.instruction(4, 0x10c);
  write_vgpr.register_class = dh::RegisterClass::Vector;
  write_vgpr.physical_reg = 5;
  write_vgpr.size_bytes = 4;
  write_vgpr.is_write = true;
  h.adapter.on_register_access(write_vgpr);

  dh::MemoryRouteView lds_read;
  lds_read.instruction = h.instruction(4, 0x10c);
  lds_read.resource_kind = ResourceKind::LocalMemory;
  lds_read.address = 0x100;
  lds_read.size_bytes = 4;
  lds_read.is_load = true;
  h.adapter.on_memory_route(lds_read);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, WaitIdleClearsAllRocjitsuDomains) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView global_load;
  global_load.instruction = h.instruction(1, 0x100);
  global_load.resource_kind = ResourceKind::GlobalMemory;
  global_load.register_class = dh::RegisterClass::Vector;
  global_load.register_base = 4;
  global_load.size_bytes = 4;
  global_load.is_load = true;
  h.adapter.on_memory_route(global_load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView global_store;
  global_store.instruction = h.instruction(2, 0x104);
  global_store.resource_kind = ResourceKind::GlobalMemory;
  global_store.register_class = dh::RegisterClass::Vector;
  global_store.register_base = 5;
  global_store.size_bytes = 4;
  global_store.is_store = true;
  h.adapter.on_memory_route(global_store);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::MemoryRouteView scalar_load;
  scalar_load.instruction = h.instruction(3, 0x108);
  scalar_load.resource_kind = ResourceKind::ScalarRegister;
  scalar_load.register_class = dh::RegisterClass::Scalar;
  scalar_load.register_base = 12;
  scalar_load.size_bytes = 4;
  scalar_load.is_load = true;
  h.adapter.on_memory_route(scalar_load);

  h.adapter.on_instruction(h.instruction(4, 0x10c));

  dh::MemoryRouteView lds_read;
  lds_read.instruction = h.instruction(4, 0x10c);
  lds_read.resource_kind = ResourceKind::LocalMemory;
  lds_read.register_class = dh::RegisterClass::Vector;
  lds_read.register_base = 8;
  lds_read.address = 0x100;
  lds_read.size_bytes = 4;
  lds_read.is_load = true;
  h.adapter.on_memory_route(lds_read);

  h.adapter.on_instruction(h.instruction(5, 0x110));

  dh::MemoryRouteView tensor;
  tensor.instruction = h.instruction(5, 0x110);
  tensor.resource_kind = ResourceKind::LocalMemory;
  tensor.address = 0x180;
  tensor.size_bytes = 64;
  tensor.is_tensor = true;
  h.adapter.on_memory_route(tensor);

  dh::InstructionView wait = h.instruction(6, 0x114);
  wait.wait.kind = dh::WaitKind::WaitIdle;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(7, 0x118));

  dh::RegisterAccessView read_vgpr;
  read_vgpr.instruction = h.instruction(7, 0x118);
  read_vgpr.register_class = dh::RegisterClass::Vector;
  read_vgpr.physical_reg = 4;
  read_vgpr.size_bytes = 4;
  read_vgpr.is_read = true;
  h.adapter.on_register_access(read_vgpr);

  dh::RegisterAccessView write_vgpr = read_vgpr;
  write_vgpr.physical_reg = 5;
  write_vgpr.is_read = false;
  write_vgpr.is_write = true;
  h.adapter.on_register_access(write_vgpr);

  dh::RegisterAccessView read_sgpr;
  read_sgpr.instruction = h.instruction(7, 0x118);
  read_sgpr.register_class = dh::RegisterClass::Scalar;
  read_sgpr.physical_reg = 12;
  read_sgpr.size_bytes = 4;
  read_sgpr.is_read = true;
  h.adapter.on_register_access(read_sgpr);

  dh::RegisterAccessView read_lds_vgpr = read_vgpr;
  read_lds_vgpr.physical_reg = 8;
  h.adapter.on_register_access(read_lds_vgpr);

  dh::MemoryRouteView lds_after_idle;
  lds_after_idle.instruction = h.instruction(7, 0x118);
  lds_after_idle.resource_kind = ResourceKind::LocalMemory;
  lds_after_idle.address = 0x180;
  lds_after_idle.size_bytes = 4;
  lds_after_idle.is_load = true;
  h.adapter.on_memory_route(lds_after_idle);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, LocalRouteEmitsLdsAccessAndVectorDestinationEvents) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView lds_write;
  lds_write.instruction = h.instruction(1, 0x100);
  lds_write.resource_kind = ResourceKind::LocalMemory;
  lds_write.address = 0x80;
  lds_write.size_bytes = 4;
  lds_write.is_store = true;
  h.adapter.on_memory_route(lds_write);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView lds_read;
  lds_read.instruction = h.instruction(2, 0x104);
  lds_read.resource_kind = ResourceKind::LocalMemory;
  lds_read.register_class = dh::RegisterClass::Vector;
  lds_read.register_base = 9;
  lds_read.address = 0x80;
  lds_read.size_bytes = 4;
  lds_read.is_load = true;
  h.adapter.on_memory_route(lds_read);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::RegisterAccessView read_vgpr;
  read_vgpr.instruction = h.instruction(3, 0x108);
  read_vgpr.register_class = dh::RegisterClass::Vector;
  read_vgpr.physical_reg = 9;
  read_vgpr.size_bytes = 4;
  read_vgpr.is_read = true;
  h.adapter.on_register_access(read_vgpr);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::LDS);
  EXPECT_EQ(warnings[1].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[1].finding.resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(warnings[1].finding.required_wait, WaitCntType::LDS);
}

TEST(DataHazardAdapterTest, LocalRouteWithDirectLocalFlagEmitsOneLocalMemoryEvent) {
  RecordingSimulatorApi api;
  dh::DataHazardAdapter adapter{api};
  dh::InstructionView instruction;
  dh::MemoryRouteView local_store;
  ExecutionKey const wave{1, 0, 0, 0, 0};

  instruction.execution = wave;
  instruction.instruction_id = 1;
  instruction.pc = 0x100;
  adapter.on_instruction(instruction);

  local_store.instruction = instruction;
  local_store.resource_kind = ResourceKind::LocalMemory;
  local_store.address = 0x80;
  local_store.local_address = 0x80;
  local_store.size_bytes = 4;
  local_store.local_size_bytes = 4;
  local_store.is_store = true;
  local_store.writes_local_memory = true;
  adapter.on_memory_route(local_store);

  ASSERT_EQ(api.resources.size(), 1u);
  EXPECT_EQ(api.resources[0].resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(api.resources[0].address, 0x80u);
  EXPECT_TRUE(api.resources[0].is_write);
  EXPECT_EQ(api.resources[0].hazards.write_wait, WaitCntType::LDS);
}

TEST(DataHazardAdapterTest, LocalPerLaneRouteFeedsSpecificLdsRegion) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView lds_write;
  lds_write.instruction = h.instruction(1, 0x100);
  lds_write.resource_kind = ResourceKind::LocalMemory;
  lds_write.size_bytes = 4;
  lds_write.is_store = true;
  lds_write.exec_mask = 0xFu;
  lds_write.per_lane_addresses = {0, 4, 8, 12};
  h.adapter.on_memory_route(lds_write);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView disjoint_read;
  disjoint_read.instruction = h.instruction(2, 0x104);
  disjoint_read.resource_kind = ResourceKind::LocalMemory;
  disjoint_read.address = 16;
  disjoint_read.size_bytes = 4;
  disjoint_read.is_load = true;
  h.adapter.on_memory_route(disjoint_read);

  dh::MemoryRouteView overlapping_read = disjoint_read;
  overlapping_read.address = 8;
  h.adapter.on_memory_route(overlapping_read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(warnings[0].finding.address, 8u);
}

TEST(DataHazardAdapterTest, GlobalToLdsRouteTracksVmemGuardedLocalWrite) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView direct_to_lds;
  direct_to_lds.instruction = h.instruction(1, 0x100);
  direct_to_lds.resource_kind = ResourceKind::GlobalMemory;
  direct_to_lds.address = 0x4000;
  direct_to_lds.local_address = 0x180;
  direct_to_lds.size_bytes = 4;
  direct_to_lds.local_size_bytes = 4;
  direct_to_lds.is_load = true;
  direct_to_lds.writes_local_memory = true;
  direct_to_lds.local_write_wait = WaitCntType::VMEM;
  h.adapter.on_memory_route(direct_to_lds);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView read;
  read.instruction = h.instruction(2, 0x104);
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x180;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(warnings[0].suggestion, "Add s_wait_loadcnt 0 before reading this LDS address");
}

TEST(DataHazardAdapterTest, LoadcntClearsGlobalToLdsRouteBeforeLocalRead) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView direct_to_lds;
  direct_to_lds.instruction = h.instruction(1, 0x100);
  direct_to_lds.resource_kind = ResourceKind::GlobalMemory;
  direct_to_lds.address = 0x4000;
  direct_to_lds.local_address = 0x180;
  direct_to_lds.size_bytes = 4;
  direct_to_lds.local_size_bytes = 4;
  direct_to_lds.is_load = true;
  direct_to_lds.writes_local_memory = true;
  direct_to_lds.local_write_wait = WaitCntType::VMEM;
  h.adapter.on_memory_route(direct_to_lds);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitLoadcnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::MemoryRouteView read;
  read.instruction = h.instruction(3, 0x108);
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x180;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, GlobalToLdsRouteFeedsCrossWaveLdsRaceDetection) {
  AdapterHarness h;

  ExecutionKey other_wave = h.wave;
  other_wave.wave_id = 1;
  h.adapter.on_wave_begin(other_wave);

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView direct_to_lds;
  direct_to_lds.instruction = h.instruction(1, 0x100);
  direct_to_lds.resource_kind = ResourceKind::GlobalMemory;
  direct_to_lds.address = 0x4000;
  direct_to_lds.local_address = 0x180;
  direct_to_lds.size_bytes = 4;
  direct_to_lds.local_size_bytes = 4;
  direct_to_lds.is_load = true;
  direct_to_lds.writes_local_memory = true;
  direct_to_lds.local_write_wait = WaitCntType::VMEM;
  h.adapter.on_memory_route(direct_to_lds);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitLoadcnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  dh::InstructionView read_inst;
  read_inst.execution = other_wave;
  read_inst.instruction_id = 3;
  read_inst.pc = 0x108;
  h.adapter.on_instruction(read_inst);

  dh::MemoryRouteView read;
  read.instruction = read_inst;
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x180;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  h.adapter.on_shutdown();

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::LocalMemoryRace);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(warnings[0].finding.address, 0x180u);
}

TEST(DataHazardAdapterTest, GlobalToLdsPerLaneRouteFeedsSpecificLdsRegion) {
  AdapterHarness h;

  ExecutionKey other_wave = h.wave;
  other_wave.wave_id = 1;
  h.adapter.on_wave_begin(other_wave);

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView direct_to_lds;
  direct_to_lds.instruction = h.instruction(1, 0x100);
  direct_to_lds.resource_kind = ResourceKind::GlobalMemory;
  direct_to_lds.address = 0x4000;
  direct_to_lds.size_bytes = 64;
  direct_to_lds.local_size_bytes = 16;
  direct_to_lds.is_load = true;
  direct_to_lds.writes_local_memory = true;
  direct_to_lds.local_write_wait = WaitCntType::VMEM;
  direct_to_lds.exec_mask = 0xFu;
  direct_to_lds.per_lane_local_addresses = {0, 16, 32, 48};
  h.adapter.on_memory_route(direct_to_lds);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitLoadcnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  dh::InstructionView read_inst;
  read_inst.execution = other_wave;
  read_inst.instruction_id = 3;
  read_inst.pc = 0x108;
  h.adapter.on_instruction(read_inst);

  dh::MemoryRouteView read;
  read.instruction = read_inst;
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 32;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  h.adapter.on_shutdown();

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::LocalMemoryRace);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(warnings[0].finding.address, 32u);
}

TEST(DataHazardAdapterTest, GlobalToLdsPerLaneRouteRespectsExecMask) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView direct_to_lds;
  direct_to_lds.instruction = h.instruction(1, 0x100);
  direct_to_lds.resource_kind = ResourceKind::GlobalMemory;
  direct_to_lds.address = 0x4000;
  direct_to_lds.size_bytes = 64;
  direct_to_lds.local_size_bytes = 16;
  direct_to_lds.is_load = true;
  direct_to_lds.writes_local_memory = true;
  direct_to_lds.local_write_wait = WaitCntType::VMEM;
  direct_to_lds.exec_mask = 0x3u;
  direct_to_lds.per_lane_local_addresses = {0, 16, 32, 48};
  h.adapter.on_memory_route(direct_to_lds);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView inactive_lane_read;
  inactive_lane_read.instruction = h.instruction(2, 0x104);
  inactive_lane_read.resource_kind = ResourceKind::LocalMemory;
  inactive_lane_read.address = 32;
  inactive_lane_read.size_bytes = 4;
  inactive_lane_read.is_load = true;
  h.adapter.on_memory_route(inactive_lane_read);

  dh::MemoryRouteView active_lane_read = inactive_lane_read;
  active_lane_read.address = 16;
  h.adapter.on_memory_route(active_lane_read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.address, 16u);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::VMEM);
}

TEST(DataHazardAdapterTest, SparseCallbackIdentityUsesCurrentInstructionContext) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(10, 0x100));

  dh::MemoryRouteView load;
  load.instruction.execution = h.wave;
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 7;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(11, 0x104));

  dh::RegisterAccessView read;
  read.instruction.execution = h.wave;
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 7;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.source_instruction.instruction_id, 10u);
  EXPECT_EQ(warnings[0].finding.instruction.instruction_id, 11u);
}

TEST(DataHazardAdapterTest, WaveEndClearsSparseCurrentInstructionContext) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(10, 0x100));
  h.adapter.on_wave_end(h.wave);
  h.adapter.on_wave_begin(h.wave);

  dh::MemoryRouteView load;
  load.instruction.execution = h.wave;
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 7;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(11, 0x104));

  dh::RegisterAccessView read;
  read.instruction = h.instruction(11, 0x104);
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 7;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].finding.source_instruction.instruction_id, 10u);
  EXPECT_EQ(warnings[0].finding.instruction.instruction_id, 11u);
}

TEST(DataHazardAdapterTest, DispatchEndClearsSparseCurrentInstructionContext) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(10, 0x100));
  h.adapter.on_dispatch_end(h.wave.dispatch_id);
  h.adapter.on_dispatch_begin(h.wave.dispatch_id);
  h.adapter.on_workgroup_begin(h.wave);
  h.adapter.on_wave_begin(h.wave);

  dh::MemoryRouteView load;
  load.instruction.execution = h.wave;
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 7;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(11, 0x104));

  dh::RegisterAccessView read;
  read.instruction = h.instruction(11, 0x104);
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 7;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].finding.source_instruction.instruction_id, 10u);
  EXPECT_EQ(warnings[0].finding.instruction.instruction_id, 11u);
}

TEST(DataHazardAdapterTest, PendingOpsAreIsolatedByWaveId) {
  AdapterHarness h;

  ExecutionKey other_wave = h.wave;
  other_wave.wave_id = 1;
  h.adapter.on_wave_begin(other_wave);

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView load;
  load.instruction = h.instruction(1, 0x100);
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 7;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  dh::InstructionView other_inst;
  other_inst.execution = other_wave;
  other_inst.instruction_id = 2;
  other_inst.pc = 0x104;
  h.adapter.on_instruction(other_inst);

  dh::RegisterAccessView read;
  read.instruction = other_inst;
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 7;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, ScalarMemoryRouteTracksSgprDestinationForRawHazards) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView smem_load;
  smem_load.instruction = h.instruction(1, 0x100);
  smem_load.resource_kind = ResourceKind::ScalarRegister;
  smem_load.register_class = dh::RegisterClass::Scalar;
  smem_load.register_base = 12;
  smem_load.size_bytes = 4;
  smem_load.is_load = true;
  h.adapter.on_memory_route(smem_load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView read;
  read.instruction = h.instruction(2, 0x104);
  read.register_class = dh::RegisterClass::Scalar;
  read.physical_reg = 12;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::ScalarRegister);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::SMEM);
}

TEST(DataHazardAdapterTest, ScalarMemoryRouteTracksSgprDestinationForWawHazards) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView smem_load;
  smem_load.instruction = h.instruction(1, 0x100);
  smem_load.resource_kind = ResourceKind::ScalarRegister;
  smem_load.register_class = dh::RegisterClass::Scalar;
  smem_load.register_base = 12;
  smem_load.size_bytes = 4;
  smem_load.is_load = true;
  h.adapter.on_memory_route(smem_load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView write;
  write.instruction = h.instruction(2, 0x104);
  write.register_class = dh::RegisterClass::Scalar;
  write.physical_reg = 12;
  write.size_bytes = 4;
  write.is_write = true;
  h.adapter.on_register_access(write);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::WAW);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::ScalarRegister);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::SMEM);
}

TEST(DataHazardAdapterTest, GlobalRoutesFeedCrossWorkgroupRaceDetection) {
  AdapterHarness h;

  ExecutionKey other_wave = h.wave;
  other_wave.workgroup_id = 1;
  other_wave.wave_id = 1;
  h.adapter.on_workgroup_begin(other_wave);
  h.adapter.on_wave_begin(other_wave);

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView write;
  write.instruction = h.instruction(1, 0x100);
  write.resource_kind = ResourceKind::GlobalMemory;
  write.address = 0x4000;
  write.size_bytes = 4;
  write.is_store = true;
  h.adapter.on_memory_route(write);

  dh::InstructionView read_inst;
  read_inst.execution = other_wave;
  read_inst.instruction_id = 2;
  read_inst.pc = 0x104;
  h.adapter.on_instruction(read_inst);

  dh::MemoryRouteView read;
  read.instruction = read_inst;
  read.resource_kind = ResourceKind::GlobalMemory;
  read.address = 0x4000;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::GlobalMemoryRace);
  EXPECT_EQ(warnings[0].finding.resource_kind, ResourceKind::GlobalMemory);
  EXPECT_EQ(warnings[0].finding.address, 0x4000u);
}

TEST(DataHazardAdapterTest, WorkgroupBarrierFlushesLdsEpochRaceDetection) {
  AdapterHarness h;

  ExecutionKey other_wave = h.wave;
  other_wave.wave_id = 1;
  h.adapter.on_wave_begin(other_wave);

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView write;
  write.instruction = h.instruction(1, 0x100);
  write.resource_kind = ResourceKind::LocalMemory;
  write.address = 0x100;
  write.size_bytes = 4;
  write.is_store = true;
  h.adapter.on_memory_route(write);

  dh::InstructionView read_inst;
  read_inst.execution = other_wave;
  read_inst.instruction_id = 2;
  read_inst.pc = 0x104;
  h.adapter.on_instruction(read_inst);

  dh::MemoryRouteView read;
  read.instruction = read_inst;
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x100;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  h.adapter.on_workgroup_barrier(h.wave);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_FALSE(warnings.empty());
  EXPECT_EQ(warnings.back().finding.kind, HazardKind::LocalMemoryRace);
  EXPECT_EQ(warnings.back().finding.resource_kind, ResourceKind::LocalMemory);
  EXPECT_EQ(warnings.back().finding.address, 0x100u);
}

// s_barrier_wait is seen before the wave stalls, so a wave that arrives early
// must not close the epoch for the waves still issuing pre-barrier accesses.
TEST(DataHazardAdapterTest, EarlyBarrierWaitKeepsLaterWaveInSameLdsEpoch) {
  AdapterHarness h;

  ExecutionKey late_wave = h.wave;
  late_wave.wave_id = 1;
  h.adapter.on_wave_begin(late_wave);

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView write;
  write.instruction = h.instruction(1, 0x100);
  write.resource_kind = ResourceKind::LocalMemory;
  write.address = 0x100;
  write.size_bytes = 4;
  write.is_store = true;
  h.adapter.on_memory_route(write);

  dh::InstructionView barrier_wait = h.instruction(2, 0x104);
  barrier_wait.wait.kind = dh::WaitKind::BarrierWait;
  h.adapter.on_instruction(barrier_wait);

  EXPECT_TRUE(h.engine.warning_snapshot().empty())
      << "reaching s_barrier_wait must not close the epoch on its own";

  dh::InstructionView read_inst;
  read_inst.execution = late_wave;
  read_inst.instruction_id = 3;
  read_inst.pc = 0x108;
  h.adapter.on_instruction(read_inst);

  dh::MemoryRouteView read;
  read.instruction = read_inst;
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x100;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  // onAmdgpuBarrierResolved reports every wave once they have all arrived.
  h.adapter.on_workgroup_barrier(h.wave);
  h.adapter.on_workgroup_barrier(late_wave);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_FALSE(warnings.empty());
  EXPECT_EQ(warnings.back().finding.kind, HazardKind::LocalMemoryRace);
  EXPECT_EQ(warnings.back().finding.address, 0x100u);
}

TEST(DataHazardAdapterTest, LocalAtomicBarrierClearsLdsPendingOps) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView write;
  write.instruction = h.instruction(1, 0x100);
  write.resource_kind = ResourceKind::LocalMemory;
  write.address = 0x100;
  write.size_bytes = 4;
  write.is_store = true;
  h.adapter.on_memory_route(write);

  h.adapter.on_local_memory_atomic_barrier(h.wave, false);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView read;
  read.instruction = h.instruction(2, 0x104);
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x100;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, TensorRouteTracksTensorLdsWrite) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView tensor;
  tensor.instruction = h.instruction(1, 0x100);
  tensor.resource_kind = ResourceKind::LocalMemory;
  tensor.address = 0x180;
  tensor.size_bytes = 64;
  tensor.is_tensor = true;
  h.adapter.on_memory_route(tensor);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::MemoryRouteView read;
  read.instruction = h.instruction(2, 0x104);
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x180;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].finding.kind, HazardKind::RAW);
  EXPECT_EQ(warnings[0].finding.required_wait, WaitCntType::TENSOR);
}

TEST(DataHazardAdapterTest, TensorcntClearsTensorLdsWriteBeforeLocalRead) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView tensor;
  tensor.instruction = h.instruction(1, 0x100);
  tensor.resource_kind = ResourceKind::LocalMemory;
  tensor.address = 0x180;
  tensor.size_bytes = 64;
  tensor.is_tensor = true;
  h.adapter.on_memory_route(tensor);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitTensorcnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::MemoryRouteView read;
  read.instruction = h.instruction(3, 0x108);
  read.resource_kind = ResourceKind::LocalMemory;
  read.address = 0x180;
  read.size_bytes = 4;
  read.is_load = true;
  h.adapter.on_memory_route(read);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, GeneratedInstructionIdsDoNotCollideWithExplicitIds) {
  AdapterHarness h;

  dh::InstructionView generated;
  generated.execution = h.wave;
  generated.pc = 0x100;
  generated.raw_isa[0] = 0x12345678;
  h.adapter.on_instruction(generated);

  dh::MemoryRouteView load;
  load.instruction.execution = h.wave;
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 3;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(1, 0x104));

  dh::RegisterAccessView read;
  read.instruction = h.instruction(1, 0x104);
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 3;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_NE(warnings[0].finding.source_instruction.instruction_id, 1u);
  EXPECT_EQ(warnings[0].finding.source_instruction.pc, 0x100u);
  EXPECT_EQ(warnings[0].finding.source_instruction.raw_isa[0], 0x12345678u);
}

TEST(DataHazardAdapterTest, GlobalLoadRawScenarioReportsDestinationReadHazard) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView load;
  load.instruction.execution = h.wave;
  load.resource_kind = ResourceKind::GlobalMemory;
  load.register_class = dh::RegisterClass::Vector;
  load.register_base = 6;
  load.address = 0x2000;
  load.size_bytes = 4;
  load.is_load = true;
  h.adapter.on_memory_route(load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView read;
  read.instruction.execution = h.wave;
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 6;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 1u);
  const auto &finding = warnings[0].finding;
  EXPECT_EQ(finding.kind, HazardKind::RAW);
  EXPECT_EQ(finding.resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(finding.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(finding.source_instruction.instruction_id, 1u);
  EXPECT_EQ(finding.instruction.instruction_id, 2u);
  EXPECT_EQ(finding.resource_index, 6u);
  EXPECT_EQ(finding.address, 0u);
}

TEST(DataHazardAdapterTest, CallbackOrderReplayReportsWawThenTracksLaterRaw) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView first_load;
  first_load.instruction.execution = h.wave;
  first_load.resource_kind = ResourceKind::GlobalMemory;
  first_load.register_class = dh::RegisterClass::Vector;
  first_load.register_base = 8;
  first_load.address = 0x2000;
  first_load.size_bytes = 4;
  first_load.is_load = true;
  h.adapter.on_memory_route(first_load);

  h.adapter.on_instruction(h.instruction(2, 0x104));

  dh::RegisterAccessView early_write;
  early_write.instruction.execution = h.wave;
  early_write.register_class = dh::RegisterClass::Vector;
  early_write.physical_reg = 8;
  early_write.size_bytes = 4;
  early_write.is_write = true;
  h.adapter.on_register_access(early_write);

  dh::MemoryRouteView second_load;
  second_load.instruction.execution = h.wave;
  second_load.resource_kind = ResourceKind::GlobalMemory;
  second_load.register_class = dh::RegisterClass::Vector;
  second_load.register_base = 8;
  second_load.address = 0x2010;
  second_load.size_bytes = 4;
  second_load.is_load = true;
  h.adapter.on_memory_route(second_load);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::RegisterAccessView read;
  read.instruction.execution = h.wave;
  read.register_class = dh::RegisterClass::Vector;
  read.physical_reg = 8;
  read.size_bytes = 4;
  read.is_read = true;
  h.adapter.on_register_access(read);

  const auto warnings = h.engine.warning_snapshot();
  ASSERT_EQ(warnings.size(), 2u);

  const auto &waw = warnings[0].finding;
  EXPECT_EQ(waw.kind, HazardKind::WAW);
  EXPECT_EQ(waw.resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(waw.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(waw.source_instruction.instruction_id, 1u);
  EXPECT_EQ(waw.instruction.instruction_id, 2u);
  EXPECT_EQ(waw.resource_index, 8u);
  EXPECT_EQ(waw.address, 0u);

  const auto &raw = warnings[1].finding;
  EXPECT_EQ(raw.kind, HazardKind::RAW);
  EXPECT_EQ(raw.resource_kind, ResourceKind::VectorRegister);
  EXPECT_EQ(raw.required_wait, WaitCntType::VMEM);
  EXPECT_EQ(raw.source_instruction.instruction_id, 2u);
  EXPECT_EQ(raw.instruction.instruction_id, 3u);
  EXPECT_EQ(raw.resource_index, 8u);
  EXPECT_EQ(raw.address, 0u);
}

TEST(DataHazardAdapterTest, DirectToLdsLoadcntClearsPendingWriteBeforeLocalRead) {
  AdapterHarness h;

  h.adapter.on_instruction(h.instruction(1, 0x100));

  dh::MemoryRouteView direct_to_lds;
  direct_to_lds.instruction.execution = h.wave;
  direct_to_lds.resource_kind = ResourceKind::GlobalMemory;
  direct_to_lds.register_class = dh::RegisterClass::None;
  direct_to_lds.address = 0x4000;
  direct_to_lds.size_bytes = 4;
  direct_to_lds.local_address = 0x180;
  direct_to_lds.local_size_bytes = 4;
  direct_to_lds.is_load = true;
  direct_to_lds.writes_local_memory = true;
  direct_to_lds.local_write_wait = WaitCntType::VMEM;
  h.adapter.on_memory_route(direct_to_lds);

  dh::InstructionView wait = h.instruction(2, 0x104);
  wait.wait.kind = dh::WaitKind::WaitLoadcnt;
  wait.wait.immediate = 0;
  h.adapter.on_instruction(wait);

  h.adapter.on_instruction(h.instruction(3, 0x108));

  dh::MemoryRouteView local_read;
  local_read.instruction.execution = h.wave;
  local_read.resource_kind = ResourceKind::LocalMemory;
  local_read.register_class = dh::RegisterClass::None;
  local_read.address = 0x180;
  local_read.size_bytes = 4;
  local_read.is_load = true;
  h.adapter.on_memory_route(local_read);

  EXPECT_TRUE(h.engine.warning_snapshot().empty());
}

TEST(DataHazardAdapterTest, MapsWaitMnemonicsToWaitKinds) {
  EXPECT_EQ(dh::make_wait_kind("s_waitcnt"), dh::WaitKind::Waitcnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_loadcnt"), dh::WaitKind::WaitLoadcnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_storecnt"), dh::WaitKind::WaitStorecnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_kmcnt"), dh::WaitKind::WaitKmcnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_dscnt"), dh::WaitKind::WaitDscnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_loadcnt_dscnt"), dh::WaitKind::WaitLoadcntDscnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_storecnt_dscnt"), dh::WaitKind::WaitStorecntDscnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_idle"), dh::WaitKind::WaitIdle);
  EXPECT_EQ(dh::make_wait_kind("s_waitcnt_vscnt"), dh::WaitKind::WaitVscnt);
  EXPECT_EQ(dh::make_wait_kind("s_wait_tensorcnt"), dh::WaitKind::WaitTensorcnt);
  EXPECT_EQ(dh::make_wait_kind("s_barrier_wait"), dh::WaitKind::BarrierWait);
  EXPECT_EQ(dh::make_wait_kind("s_sema_wait"), dh::WaitKind::SemaphoreWait);
  EXPECT_EQ(dh::make_wait_kind("s_wait_xcnt"), dh::WaitKind::AddressTranslation);
  EXPECT_EQ(dh::make_wait_kind("v_add_u32"), dh::WaitKind::None);
}

TEST(DataHazardAdapterTest, SemaphoreWaitClosesWavegroupEpochRatherThanDrainingACounter) {
  dh::WaitInfo wait;
  wait.kind = dh::WaitKind::SemaphoreWait;

  const WaitAction action = dh::make_wait_action(wait);
  EXPECT_TRUE(action.is_wait_instruction);
  EXPECT_TRUE(action.is_wavegroup_semaphore_wait);
  // A semaphore orders waves, so it neither drains this wave's counters nor
  // closes the workgroup-wide epoch a barrier closes.
  EXPECT_TRUE(action.counters.empty());
  EXPECT_FALSE(action.is_workgroup_barrier);
}

TEST(DataHazardAdapterTest, MapsWaitInfoToGenericWaitActions) {
  dh::WaitInfo wait;
  wait.kind = dh::WaitKind::Waitcnt;
  wait.immediate = 0x4000u | (5u << 8) | 7u;

  // s_waitcnt drains the vector memory counter, LDS and scalar memory. It
  // names no store counter of its own: a gfx9 store is outstanding on the
  // vector memory counter this already drains, and a gfx10 or gfx11 store
  // waits for the separate s_waitcnt_vscnt.
  WaitAction action = dh::make_wait_action(wait);
  ASSERT_TRUE(action.is_wait_instruction);
  ASSERT_EQ(action.counters.size(), 3u);
  ASSERT_NE(find_counter(action, WaitCntType::VMEM), nullptr);
  ASSERT_EQ(find_counter(action, WaitCntType::STORE), nullptr);
  ASSERT_NE(find_counter(action, WaitCntType::LDS), nullptr);
  ASSERT_NE(find_counter(action, WaitCntType::SMEM), nullptr);
  EXPECT_EQ(find_counter(action, WaitCntType::VMEM)->keep_count, 0x17u);
  EXPECT_EQ(find_counter(action, WaitCntType::LDS)->keep_count, 5u);
  EXPECT_EQ(find_counter(action, WaitCntType::SMEM)->keep_count, 5u);

  wait.kind = dh::WaitKind::WaitLoadcntDscnt;
  wait.immediate = (9u << 8) | 4u; // loadcnt(9) dscnt(4)
  action = dh::make_wait_action(wait);
  ASSERT_EQ(action.counters.size(), 2u);
  ASSERT_NE(find_counter(action, WaitCntType::VMEM), nullptr);
  ASSERT_NE(find_counter(action, WaitCntType::LDS), nullptr);
  EXPECT_EQ(find_counter(action, WaitCntType::VMEM)->keep_count, 9u);
  EXPECT_EQ(find_counter(action, WaitCntType::LDS)->keep_count, 4u);

  wait.kind = dh::WaitKind::WaitStorecntDscnt;
  wait.immediate = (11u << 8) | 6u; // storecnt(11) dscnt(6)
  action = dh::make_wait_action(wait);
  ASSERT_EQ(action.counters.size(), 2u);
  ASSERT_NE(find_counter(action, WaitCntType::STORE), nullptr);
  ASSERT_NE(find_counter(action, WaitCntType::LDS), nullptr);
  EXPECT_EQ(find_counter(action, WaitCntType::STORE)->keep_count, 11u);
  EXPECT_EQ(find_counter(action, WaitCntType::LDS)->keep_count, 6u);

  wait.kind = dh::WaitKind::WaitIdle;
  action = dh::make_wait_action(wait);
  EXPECT_TRUE(action.is_wait_instruction);
  EXPECT_TRUE(action.waits_for_idle);
  EXPECT_TRUE(action.counters.empty());

  // s_barrier_wait is classified before it stalls, so it must not claim the
  // barrier has completed; the epoch is closed on barrier resolution instead.
  wait.kind = dh::WaitKind::BarrierWait;
  action = dh::make_wait_action(wait);
  EXPECT_TRUE(action.is_wait_instruction);
  EXPECT_FALSE(action.is_workgroup_barrier);
  EXPECT_TRUE(action.counters.empty());

  wait.kind = dh::WaitKind::AddressTranslation;
  action = dh::make_wait_action(wait);
  EXPECT_TRUE(action.is_address_translation);
}
