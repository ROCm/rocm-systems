// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <sys/mman.h>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

struct CapturedTensorDmaAccess {
  std::string mnemonic;
  uint64_t pc = 0;
  uint32_t compute_unit_id = 0;
  uint32_t dispatch_id = 0;
  uint32_t queue_id = 0;
  uint32_t workgroup_id = 0;
  uint32_t wavefront_id = 0;
  uint32_t process_id = 0;
  uint32_t element_size_bytes = 0;
  uint32_t tile_dim0 = 0;
  uint32_t tile_dim1 = 0;
  uint32_t data_size = 0;
  int64_t tensor_dim0_stride = 0;
  int64_t tensor_dim1_stride = 0;
  bool is_load = true;
  std::vector<uint64_t> addresses;
};

class TensorDmaObservationPlugin final : public ExecutionPlugin {
public:
  explicit TensorDmaObservationPlugin(std::string name = "tensor_dma_observation",
                                      bool wants_hook = true)
      : ExecutionPlugin(std::move(name)), wants_hook_(wants_hook) {}

  bool observes_tensor_dma_memory_access() const override { return wants_hook_; }

  void
  onAmdgpuTensorDmaMemoryAccess(const amdgpu::TensorDmaMemoryAccessObservation &access) override {
    if (inspect)
      inspect(access);
    auto &captured = accesses.emplace_back(CapturedTensorDmaAccess{
        .mnemonic = std::string(access.mnemonic),
        .pc = access.pc,
        .compute_unit_id = access.compute_unit_id,
        .dispatch_id = access.dispatch_id,
        .queue_id = access.queue_id,
        .workgroup_id = access.workgroup_id,
        .wavefront_id = access.wavefront_id,
        .process_id = access.process_id,
        .element_size_bytes = access.element_size_bytes,
        .tile_dim0 = access.tile_dim0,
        .tile_dim1 = access.tile_dim1,
        .data_size = access.data_size,
        .tensor_dim0_stride = access.tensor_dim0_stride,
        .tensor_dim1_stride = access.tensor_dim1_stride,
        .is_load = access.is_load,
        .addresses = {},
    });
    captured.addresses.resize(access.addresses.size());
    EXPECT_TRUE(access.addresses.copy_to(captured.addresses));
  }

  std::function<void(const amdgpu::TensorDmaMemoryAccessObservation &)> inspect;
  std::vector<CapturedTensorDmaAccess> accesses;

private:
  bool wants_hook_ = true;
};

TensorDmaObservationPlugin &install_tensor_dma_observer(Gfx1250Sim &sim, bool wants_hook = true) {
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer =
      std::make_unique<TensorDmaObservationPlugin>("tensor_dma_observation", wants_hook);
  auto *result = observer.get();
  EXPECT_TRUE(group->add(std::move(observer)));
  sim.plugin_group = std::move(group);
  sim.soc->set_plugin_group(sim.plugin_group);
  return *result;
}

/// Records every tensor DMA issue observation, copying its LDS ranges.
class TensorDmaIssuePlugin final : public ExecutionPlugin {
public:
  explicit TensorDmaIssuePlugin(std::string name = "tensor_dma_issue", bool wants_hook = true)
      : ExecutionPlugin(std::move(name)), wants_hook_(wants_hook) {}

  bool observes_tensor_dma_issue() const override { return wants_hook_; }

  void onAmdgpuTensorDmaIssued(const amdgpu::TensorDmaIssueObservation &issue,
                               const amdgpu::Wavefront &wf) override {
    if (inspect)
      inspect(issue, wf);
    issues.push_back({.is_load = issue.is_load,
                      .lds_ranges = {issue.lds_ranges.begin(), issue.lds_ranges.end()}});
  }

  struct Captured {
    bool is_load = true;
    std::vector<amdgpu::TensorDmaLdsRange> lds_ranges;
  };

  std::function<void(const amdgpu::TensorDmaIssueObservation &, const amdgpu::Wavefront &)> inspect;
  std::vector<Captured> issues;

private:
  bool wants_hook_ = true;
};

TensorDmaIssuePlugin &install_tensor_dma_issue_observer(Gfx1250Sim &sim, bool wants_hook = true) {
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<TensorDmaIssuePlugin>("tensor_dma_issue", wants_hook);
  auto *result = observer.get();
  EXPECT_TRUE(group->add(std::move(observer)));
  sim.plugin_group = std::move(group);
  sim.soc->set_plugin_group(sim.plugin_group);
  return *result;
}

void write_tensor_dma_d0(amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wf, uint32_t reg,
                         uint64_t global_addr, uint32_t lds_base = 0) {
  write_wave_sgpr(cu, wf, reg + 0, 1);
  write_wave_sgpr(cu, wf, reg + 1, lds_base);
  write_wave_sgpr(cu, wf, reg + 2, static_cast<uint32_t>(global_addr));
  write_wave_sgpr(cu, wf, reg + 3,
                  static_cast<uint32_t>((global_addr >> 32) & 0x01ffffffu) | 0x80000000u);
}

TEST(TensorDmaMemoryAccessObservationTest, PreservesLegacyPositionalInitialization) {
  const std::array<uint64_t, 2> addresses{0x1000, 0x1004};
  const amdgpu::TensorDmaMemoryAccessObservation access{"tensor_load_to_lds",
                                                        0x2000,
                                                        1,
                                                        2,
                                                        3,
                                                        4,
                                                        5,
                                                        6,
                                                        4,
                                                        false,
                                                        std::span<const uint64_t>(addresses)};

  EXPECT_FALSE(access.is_load);
  std::array<uint64_t, 2> copied_addresses{};
  ASSERT_TRUE(access.addresses.copy_to(copied_addresses));
  EXPECT_EQ(copied_addresses, addresses);
  EXPECT_EQ(access.tile_dim0, 0u);
  EXPECT_EQ(access.tile_dim1, 0u);
  EXPECT_EQ(access.data_size, 0u);
  EXPECT_EQ(access.tensor_dim0_stride, 0);
  EXPECT_EQ(access.tensor_dim1_stride, 0);
}

TEST(Gfx1250ExecutionTest, TensorDmaDescriptorReadPropagatesFailure) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, 16, 32);
  ASSERT_NE(wf, nullptr);
  using amdgpu::tensor_dma_detail::read_sgpr_group;
  EXPECT_TRUE(read_sgpr_group<4>(*wf, 124, false).failed());
  EXPECT_TRUE(read_sgpr_group<4>(*wf, 104, false).failed());
  auto optional = read_sgpr_group<4>(*wf, 124, true);
  ASSERT_TRUE(optional.succeeded());
  EXPECT_EQ(optional.value(), (std::array<uint32_t, 4>{}));
  wf->halt();
  EXPECT_TRUE(read_sgpr_group<4>(*wf, 0, false).failed());
}

TEST(Gfx1250ExecutionTest, TensorDmaFailureSkipsCopyAndBarrierAndCanRecover) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_issue_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  constexpr uint64_t kGlobal = 0x1a0000;
  constexpr uint32_t kBarrierAddr = 64;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 2); // Unsupported active count.
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18));
  write_wave_sgpr(*cu, *wf, 13, (1u << 16) | (kBarrierAddr >> 3));
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  write_global_u32(*sim.memory, kGlobal, 0x12345678u);
  cu->lds().write32(wf->lds_base(), 0xabcdef00u);
  cu->lds().write64(wf->lds_base() + kBarrierAddr, 0);
  const std::array<uint32_t, 3> words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  cdna5::TensorLoadToLdsVimage inst(words.data());
  EXPECT_TRUE(amdgpu::execute_tensor_load_to_lds(inst, *wf).failed());
  EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0xabcdef00u);
  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierAddr), 0u);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_TRUE(cu->execute_instruction(&inst, *wf).failed());
  write_wave_sgpr(*cu, *wf, 0, 1);
  EXPECT_TRUE(cu->execute_instruction(&inst, *wf).succeeded());
  EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0x12345678u);
  EXPECT_NE(cu->lds().read64(wf->lds_base() + kBarrierAddr), 0u);
  EXPECT_TRUE(wf->wait_counters().empty());

  for (bool store : {false, true}) {
    SCOPED_TRACE(store);
    write_global_u32(*sim.memory, kGlobal, 0x12345678u);
    cu->lds().write32(wf->lds_base(), 0xabcdef00u);
    cu->lds().write64(wf->lds_base() + kBarrierAddr, 0);
    const std::array<uint32_t, 3> encoding = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                              0x7c7c0c00u};
    auto transfer =
        decode_gfx1250(encoding, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
    ASSERT_NE(transfer, nullptr);
    observer.issues.clear();
    observer.inspect = [](const amdgpu::TensorDmaIssueObservation &,
                          const amdgpu::Wavefront &wave) {
      wave.report_instruction_execution_error(amdgpu::InstructionExecutionError::PluginFailure,
                                              "tensor issue hook failed");
    };
    EXPECT_TRUE(cu->execute_instruction(transfer.get(), *wf).failed());
    EXPECT_EQ(wf->instruction_execution_error(), amdgpu::InstructionExecutionError::PluginFailure);
    ASSERT_EQ(observer.issues.size(), 1u);
    EXPECT_EQ(observer.issues[0].is_load, !store);
    EXPECT_EQ(transfer->data(), nullptr);
    EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0xabcdef00u);
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), 0x12345678u);
    EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierAddr), 0u);
    EXPECT_TRUE(wf->wait_counters().empty());

    observer.inspect = {};
    EXPECT_TRUE(cu->execute_instruction(transfer.get(), *wf).succeeded());
    EXPECT_FALSE(wf->instruction_execution_failed());
    EXPECT_EQ(cu->lds().read32(wf->lds_base()), store ? 0xabcdef00u : 0x12345678u);
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), store ? 0xabcdef00u : 0x12345678u);
    EXPECT_NE(cu->lds().read64(wf->lds_base() + kBarrierAddr), 0u);
    EXPECT_TRUE(wf->wait_counters().empty());
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaRejectsElementCountOverflow) {
  using namespace amdgpu::tensor_dma_detail;
  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 4;
  desc.tensor_rank = 5;
  desc.tensor_dims.fill(65535);
  desc.tile_dims = {65535, 65535, 65535, 65535, 1};
  const TensorDmaLayout layout(desc);
  EXPECT_TRUE(validate_supported_descriptor(desc, layout).succeeded());
  desc.tile_dims[4] = 2;
  EXPECT_TRUE(validate_supported_descriptor(desc, layout).failed());
  desc.tile_dims.fill(65535);
  EXPECT_TRUE(validate_supported_descriptor(desc, layout).failed());
}

TEST(Gfx1250ExecutionTest, TensorDmaDenseTransferWithUnusedInteriorDimensions) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  // Axes 0, 2 and 4 each contain two elements. Tile dimensions 1 and 3 are zero.
  constexpr uint64_t kGlobal = 0x135000;
  constexpr std::array<uint32_t, 16> groups{
      2u << 16, 2u << 16, 0,        2u << 16, 2u << 16, 99, 8u << 16, 0, // Group 1.
      2,        0,        99,       0,                                   // Group 2.
      24,       2u << 16, 2u << 16, 0,                                   // Group 3.
  };
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  for (uint32_t descriptor_word = 0; descriptor_word < groups.size(); ++descriptor_word)
    write_wave_sgpr(*cu, *wf, 12 + descriptor_word, groups[descriptor_word]);
  const std::array<uint32_t, 3> load_words{0xd0710001u, 0x7c000000u, 0x18140c00u};
  const std::array<uint32_t, 3> store_words{0xd0714001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(store, nullptr);

  constexpr std::array<uint32_t, 8> global_indices{0, 1, 8, 9, 24, 25, 32, 33};
  struct Case {
    uint32_t extent1, extent3, index_size;
    bool copies;
  };
  // In gfx1250 descriptor_fields.hip cases 2/3, LDS kept 0x80000000 + lane after the load.
  // Cases 6/7 kept global[index] = 1000 + index after the store. The probes have one
  // empty axis, numbered 1 and 2 respectively. Axis 3 here extends the same rule.
  constexpr Case cases[]{
      {0, 1, 0, false},
      {1, 0, 0, false},
      {1, 1, 0, true},
      {1, 1, 1u << 30, true},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "tensor_dim1=" << test_case.extent1 << " tensor_dim3=" << test_case.extent3
                 << " index_size=" << test_case.index_size);
    // The gather index-size bit has no effect on dense descriptors.
    write_wave_sgpr(*cu, *wf, 0, 1u | test_case.index_size);
    write_wave_sgpr(*cu, *wf, 14, test_case.extent1 << 16);
    write_wave_sgpr(*cu, *wf, 21, test_case.extent3);
    for (uint32_t tile_index = 0; tile_index < global_indices.size(); ++tile_index)
      cu->lds().write32(wf->lds_base() + 4 * tile_index, 3000 + tile_index);
    for (uint32_t global_index = 0; global_index < 34; ++global_index)
      write_global_u32(*sim.memory, kGlobal + 4 * global_index, 1000 + global_index);
    ASSERT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    for (uint32_t tile_index = 0; tile_index < global_indices.size(); ++tile_index) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * tile_index),
                test_case.copies ? 1000 + global_indices[tile_index] : 3000 + tile_index);
      cu->lds().write32(wf->lds_base() + 4 * tile_index, 2000 + global_indices[tile_index]);
    }
    ASSERT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());
    for (uint32_t global_index = 0; global_index < 34; ++global_index) {
      const bool copied = test_case.copies &&
                          std::ranges::find(global_indices, global_index) != global_indices.end();
      EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 4 * global_index),
                (copied ? 2000 : 1000) + global_index);
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaIterationsMayReuseResidentLds) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(16));
  wf->set_lds_size(16);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 4;
  desc.iterate = true;
  desc.iteration_count = 2;
  desc.global_increment = 4;
  desc.lds_increment = 0;
  desc.tensor_rank = 2;
  desc.tensor_dims = {8, 1, 0, 0, 0};
  desc.tile_dims = {4, 1, 0, 0, 0};
  desc.global_strides[0] = 8;
  TensorDmaState state(desc, false);
  EXPECT_TRUE(prepare_tensor(state, *wf).succeeded());
  ASSERT_EQ(state.elements.size(), 8u);
  EXPECT_EQ(state.elements[0].lds_address, state.elements[4].lds_address);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterationLdsIncrementPastAllocationIsNotRejected) {
  // Out-of-range LDS addresses do not abort a tensor operation.
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 1;
  desc.iterate = true;
  desc.iteration_count = 2;
  desc.lds_increment = 256;
  desc.tensor_rank = 2;
  desc.tensor_dims = {1, 1, 0, 0, 0};
  desc.tile_dims = {1, 1, 0, 0, 0};
  desc.global_strides[0] = 1;
  TensorDmaState state(desc, false);
  EXPECT_TRUE(prepare_tensor(state, *wf).succeeded());
  EXPECT_EQ(state.elements.size(), 2u);
}

TEST(Gfx1250ExecutionTest, TensorDmaZeroLdsAllocationIsNotRejected) {
  // Every LDS address is out of range, which does not abort the operation.
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 4;
  desc.tensor_rank = 1;
  desc.tensor_dims[0] = 1;
  desc.tile_dims[0] = 1;
  TensorDmaState state(desc, false);
  EXPECT_TRUE(prepare_tensor(state, *wf).succeeded());
  EXPECT_EQ(state.elements.size(), 1u);
}

/// A wave with one 2 KiB LDS allocation followed by a neighbouring allocation, and a
/// four-element i32 tile whose last two elements fall past the wave's LDS.
class TensorDmaLdsOutOfRangeTest : public ::testing::Test {
protected:
  static constexpr uint64_t kGlobal = 0x187000;
  static constexpr uint32_t kLdsBytes = 2048;
  static constexpr uint32_t kLdsOffset = kLdsBytes - 8;
  static constexpr uint32_t kSentinel = 0x5e5e5e5eu;

  Gfx1250Sim sim;
  amdgpu::ComputeUnitCore *cu = nullptr;
  amdgpu::Wavefront *wf = nullptr;
  TensorDmaObservationPlugin *observer = nullptr; ///< Completed global requests.
  TensorDmaIssuePlugin *issue_observer = nullptr; ///< LDS ranges at issue.
  uint32_t neighbour = 0;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto completion = std::make_unique<TensorDmaObservationPlugin>();
    auto issue = std::make_unique<TensorDmaIssuePlugin>();
    observer = completion.get();
    issue_observer = issue.get();
    ASSERT_TRUE(group->add(std::move(completion)));
    ASSERT_TRUE(group->add(std::move(issue)));
    sim.plugin_group = group;
    sim.soc->set_plugin_group(sim.plugin_group);
    cu = sim.cu();
    wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    // A preceding allocation distinguishes the wave-relative limit from its absolute end.
    const uint32_t preceding = cu->allocate_lds(kLdsBytes);
    wf->set_lds_base(cu->allocate_lds(kLdsBytes));
    wf->set_lds_size(kLdsBytes);
    ASSERT_EQ(wf->lds_base(), preceding + kLdsBytes);
    ASSERT_NE(wf->lds_base(), 0u);
    neighbour = cu->allocate_lds(kLdsBytes);
    ASSERT_EQ(neighbour, wf->lds_base() + wf->lds_size());
    cu->lds().write32(neighbour, kSentinel);
    cu->lds().write32(neighbour + 4, kSentinel);
    write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsOffset);
    write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32 elements.
    write_wave_sgpr(*cu, *wf, 13, 4u << 16); // tensor dim0.
    write_wave_sgpr(*cu, *wf, 14, 0);
    write_wave_sgpr(*cu, *wf, 15, 4u << 16); // tile dim0.
    for (uint32_t reg = 16; reg < 20; ++reg)
      write_wave_sgpr(*cu, *wf, reg, 0);
  }

  /// d2_d3 selects descriptor groups 2 and 3; the default leaves both NULL.
  bool execute(bool store, uint32_t d2_d3 = 0x7c7c0c00u) {
    const std::array<uint32_t, 3> words = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u, d2_d3};
    auto inst = decode_gfx1250(words, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
    return inst && cu->execute_instruction(inst.get(), *wf).succeeded();
  }
};

TEST_F(TensorDmaLdsOutOfRangeTest, LoadDropsWritesOutsideTheWavesLds) {
  for (uint32_t i = 0; i < 4; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x48000000u + i);
  ASSERT_TRUE(execute(/*store=*/false));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOffset), 0x48000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOffset + 4), 0x48000001u);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  EXPECT_EQ(cu->lds().read32(neighbour + 4), kSentinel);
  ASSERT_EQ(observer->accesses.size(), 1u);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(observer->accesses[0].addresses.size(), 4u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsOffset, .bytes = 8}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, StoreReadsZeroOutsideTheWavesLds) {
  // The store fills bytes past the wave's LDS with zeros. That is rocjitsu's
  // modeling choice: CDNA5 ISA 10.11.6 says only that the transfer continues
  // past the allocation, and the check for ordinary LDS reads (3.3.4.1)
  // depends on the alignment mode.
  cu->lds().write32(wf->lds_base() + kLdsOffset, 0x49000000u);
  cu->lds().write32(wf->lds_base() + kLdsOffset + 4, 0x49000001u);
  for (uint32_t i = 0; i < 4; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0xffffffffu);
  ASSERT_TRUE(execute(/*store=*/true));
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), 0x49000000u);
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 4), 0x49000001u);
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 8), 0u);
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 12), 0u);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsOffset, .bytes = 8}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, LoadI64StraddlingTheEndWritesOnlyItsFourInRangeBytes) {
  // One i64 element at offset 2044 has four bytes inside the wave's LDS.
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsBytes - 4);
  write_wave_sgpr(*cu, *wf, 12, 3u << 16); // i64 elements.
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  write_global_u32(*sim.memory, kGlobal, 0x55667788u);
  write_global_u32(*sim.memory, kGlobal + 4, 0x11223344u);
  ASSERT_TRUE(execute(/*store=*/false));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 4), 0x55667788u);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsBytes - 4, .bytes = 4}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, StoreI64StraddlingTheEndReadsZeroForItsFourOutOfRangeBytes) {
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsBytes - 4);
  write_wave_sgpr(*cu, *wf, 12, 3u << 16); // i64 elements.
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  cu->lds().write32(wf->lds_base() + kLdsBytes - 4, 0xaabbccddu);
  write_global_u32(*sim.memory, kGlobal, 0xffffffffu);
  write_global_u32(*sim.memory, kGlobal + 4, 0xffffffffu);
  ASSERT_TRUE(execute(/*store=*/true));
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), 0xaabbccddu);
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 4), 0u);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsBytes - 4, .bytes = 4}}));
}

// One i32 element at offset 2046 has two bytes inside the wave's LDS. Keeping a
// two-byte prefix distinguishes byte-prefix clipping from rounding the kept
// length to whole DWORDs, which would keep nothing.
TEST_F(TensorDmaLdsOutOfRangeTest, LoadI32StraddlingTheEndWritesOnlyItsTwoInRangeBytes) {
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsBytes - 2);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  cu->lds().write32(wf->lds_base() + kLdsBytes - 4, 0xa3a2a1a0u);
  write_global_u32(*sim.memory, kGlobal, 0x55667788u);
  ASSERT_TRUE(execute(/*store=*/false));
  // Bytes 2044 and 2045 are untouched; 2046 and 2047 take the element's low bytes.
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 4), 0x7788a1a0u);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsBytes - 2, .bytes = 2}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, StoreI32StraddlingTheEndReadsZeroForItsTwoOutOfRangeBytes) {
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsBytes - 2);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  cu->lds().write32(wf->lds_base() + kLdsBytes - 4, 0xddcc0000u);
  write_global_u32(*sim.memory, kGlobal, 0xffffffffu);
  ASSERT_TRUE(execute(/*store=*/true));
  // The two in-range bytes, then zero rather than the neighbour's sentinel.
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), 0x0000ddccu);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsBytes - 2, .bytes = 2}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, PaddingPushesTheLastElementPastTheEnd) {
  // One dword of padding after every two places the elements at 2032, 2036, 2044
  // and 2048; the last is past the wave's LDS.
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsBytes - 16);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 20)); // i32, pad enabled.
  for (uint32_t i = 0; i < 4; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x4b000000u + i);
  cu->lds().write32(wf->lds_base() + kLdsBytes - 8, kSentinel);
  ASSERT_TRUE(execute(/*store=*/false));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 16), 0x4b000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 12), 0x4b000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 8), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 4), 0x4b000002u);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsBytes - 16, .bytes = 8},
                {.address = wf->lds_base() + kLdsBytes - 4, .bytes = 4}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, IterationPastTheEndIsDropped) {
  // Two iterations of a 2-element row, 16 bytes apart in LDS starting at 2032.
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsBytes - 16);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                      // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 2u);                      // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 4);        // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 2);        // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.
  for (uint32_t i = 0; i < 4; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x4c000000u + i);
  ASSERT_TRUE(execute(/*store=*/false, 0x7c140c00u));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 16), 0x4c000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsBytes - 12), 0x4c000001u);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  EXPECT_EQ(cu->lds().read32(neighbour + 4), kSentinel);
  ASSERT_EQ(observer->accesses.size(), 1u);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(observer->accesses[0].addresses.size(), 4u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsBytes - 16, .bytes = 8}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, GatherRowPastTheEndIsDropped) {
  // Rows land 8 bytes apart in LDS starting at 2040; the second is past the end.
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsOffset);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30) | (1u << 31)); // 32-bit gather indices.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                    // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 3u << 16);                    // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);                    // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 2u);                          // valid indices.
  write_wave_sgpr(*cu, *wf, 17, 2u);                          // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 20, 2u);                          // row indices.
  write_wave_sgpr(*cu, *wf, 21, 0u);
  for (uint32_t reg = 22; reg < 28; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);
  for (uint32_t i = 0; i < 6; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x4d000000u + i);
  ASSERT_TRUE(execute(/*store=*/false, 0x18140c00u));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOffset), 0x4d000004u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOffset + 4), 0x4d000005u);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  EXPECT_EQ(cu->lds().read32(neighbour + 4), kSentinel);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsOffset, .bytes = 8}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, ZeroFillPastTheEndIsDropped) {
  // The last two elements are outside both the tensor and the wave's LDS.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);
  write_global_u32(*sim.memory, kGlobal, 0x4e000000u);
  write_global_u32(*sim.memory, kGlobal + 4, 0x4e000001u);
  ASSERT_TRUE(execute(/*store=*/false));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOffset), 0x4e000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOffset + 4), 0x4e000001u);
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  EXPECT_EQ(cu->lds().read32(neighbour + 4), kSentinel);
  ASSERT_EQ(observer->accesses.size(), 1u);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(observer->accesses[0].addresses.size(), 2u);
  EXPECT_EQ(issue_observer->issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{
                {.address = wf->lds_base() + kLdsOffset, .bytes = 8}}));
}

TEST_F(TensorDmaLdsOutOfRangeTest, TileWhollyOutsideTheWavesLdsStillCompletes) {
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal, kLdsBytes);
  write_global_u32(*sim.memory, kGlobal, 0x4a000000u);
  ASSERT_TRUE(execute(/*store=*/false));
  EXPECT_EQ(cu->lds().read32(neighbour), kSentinel);
  ASSERT_EQ(observer->accesses.size(), 1u);
  ASSERT_EQ(issue_observer->issues.size(), 1u);
  EXPECT_EQ(observer->accesses[0].addresses.size(), 4u);
  EXPECT_TRUE(issue_observer->issues[0].lds_ranges.empty());
}

TEST(Gfx1250ExecutionTest, TensorDmaRejectsWrappedWalkerArithmetic) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 1;
  desc.gather = true;
  desc.gather_indices_32bit = true;
  desc.valid_indices = 1;
  desc.gather_indices[0] = 65537;
  desc.tensor_rank = 2;
  desc.tensor_dims = {1, 65538, 0, 0, 0};
  desc.tile_dims = {1, 1, 0, 0, 0};
  desc.global_strides[0] = (uint64_t{1} << 48) - 1;
  TensorDmaState state(desc, false);
  EXPECT_TRUE(prepare_tensor(state, *wf).failed());
  EXPECT_TRUE(state.elements.empty());
}

TEST(Gfx1250ExecutionTest, TensorDmaRejectsWrappedDenseWalkerArithmetic) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 1;
  desc.iterate = true;
  desc.iteration_count = 65536;
  desc.global_increment = (uint64_t{1} << 48) - 1;
  desc.lds_increment = 0;
  desc.tensor_rank = 2;
  desc.tensor_dims = {1, 65538, 0, 0, 0};
  desc.tile_dims = {1, 3, 0, 0, 0};
  desc.global_strides[0] = (uint64_t{1} << 48) - 1;
  TensorDmaState state(desc, false);
  EXPECT_TRUE(prepare_tensor(state, *wf).failed());
  EXPECT_TRUE(state.elements.empty());
}

TEST(Gfx1250ExecutionTest, TensorDmaAllowsSafeTileInLargerUnaddressableTensor) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 1;
  desc.gather = true;
  desc.gather_indices_32bit = true;
  desc.valid_indices = 1;
  desc.gather_indices[0] = 0;
  desc.tensor_rank = 2;
  desc.tensor_dims = {1, 65538, 0, 0, 0};
  desc.tile_dims = {1, 1, 0, 0, 0};
  desc.global_strides[0] = (uint64_t{1} << 48) - 1;
  desc.global_base = 0x200000;
  TensorDmaState state(desc, false);
  ASSERT_TRUE(prepare_tensor(state, *wf).succeeded());
  ASSERT_EQ(state.elements.size(), 1u);
  EXPECT_TRUE(state.elements[0].in_bounds);
  EXPECT_EQ(state.elements[0].global_address, desc.global_base);
}

TEST(Gfx1250ExecutionTest, TensorDmaAllowsSafeDenseTileInLargerUnaddressableTensor) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 1;
  desc.tensor_rank = 2;
  desc.tensor_dims = {1, 65538, 0, 0, 0};
  desc.tile_dims = {1, 1, 0, 0, 0};
  desc.global_strides[0] = (uint64_t{1} << 48) - 1;
  desc.global_base = 0x200000;
  TensorDmaState state(desc, false);
  ASSERT_TRUE(prepare_tensor(state, *wf).succeeded());
  ASSERT_EQ(state.elements.size(), 1u);
  EXPECT_TRUE(state.elements[0].in_bounds);
  EXPECT_EQ(state.elements[0].global_address, desc.global_base);
}

TEST(Gfx1250ExecutionTest, TensorDmaMaskedGatherSkipsGlobalAddressArithmetic) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  cu->lds().write8(wf->lds_base(), 0xa5);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 1;
  desc.gather = true;
  desc.gather_indices_32bit = true;
  desc.valid_indices = 1;
  desc.gather_indices[0] = 65537;
  desc.tensor_rank = 2;
  desc.tensor_dims = {1, 1, 0, 0, 0};
  desc.tile_dims = {1, 1, 0, 0, 0};
  desc.global_strides[0] = (uint64_t{1} << 48) - 1;
  desc.global_base = 0x200000;
  TensorDmaState state(desc, false);
  ASSERT_TRUE(prepare_tensor(state, *wf).succeeded());
  ASSERT_EQ(state.elements.size(), 1u);
  EXPECT_FALSE(state.elements[0].in_bounds);
  EXPECT_EQ(resume_tensor_dma_state(state, *wf), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(cu->lds().read8(wf->lds_base()), 0u);
}

TEST(Gfx1250ExecutionTest, TensorDmaMaterializationLimitIsDiagnosable) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  constexpr uint32_t kResidentBytes = 64 * 64 * sizeof(uint32_t);
  wf->set_lds_base(cu->allocate_lds(kResidentBytes));
  wf->set_lds_size(kResidentBytes);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 4;
  desc.iterate = true;
  desc.iteration_count = 256;
  desc.lds_increment = 0;
  desc.tensor_rank = 2;
  desc.tensor_dims = {64, 64, 0, 0, 0};
  desc.tile_dims = {64, 64, 0, 0, 0};
  desc.global_strides[0] = 64;
  const auto boundary_count = tensor_element_count(desc, TensorDmaLayout(desc));
  ASSERT_TRUE(boundary_count);
  EXPECT_EQ(*boundary_count, kMaxMaterializedElements);
  TensorDmaState boundary(desc, false);
  EXPECT_TRUE(prepare_tensor(boundary, *wf).succeeded());
  EXPECT_EQ(boundary.elements.size(), kMaxMaterializedElements);

  desc.iteration_count = 257;
  TensorDmaState excessive(desc, false);
  testing::internal::CaptureStderr();
  EXPECT_TRUE(prepare_tensor(excessive, *wf).failed());
  const std::string output = testing::internal::GetCapturedStderr();
  EXPECT_NE(output.find("tensor DMA materialization limit exceeded"), std::string::npos) << output;
  EXPECT_TRUE(excessive.elements.empty());
}

TEST(Gfx1250ExecutionTest, TensorDmaRejectsWrappedGlobalTransferAddresses) {
  using namespace amdgpu::tensor_dma_detail;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.elem_size = 8;
  desc.tensor_rank = 1;
  desc.tensor_dims[0] = 1;
  desc.tile_dims[0] = 1;
  desc.global_base = UINT64_MAX;
  TensorDmaState state(desc, false);
  EXPECT_TRUE(prepare_tensor(state, *wf).failed());
  EXPECT_TRUE(state.elements.empty());
}

// rocjitsu forms LDS addresses in 64 bits, so a large descriptor LDS base lies
// past the wave's allocation rather than wrapping back into LDS. That width is a
// modeling choice: CDNA5 ISA 10.11.6 specifies only that an out-of-range address
// does not abort the transfer. The element is kept, all its bytes are dropped,
// and it touches no LDS. 0xffffff00 would wrap in 32 bits onto the first byte of
// the allocation just below the wave's; UINT32_MAX onto its last.
TEST(Gfx1250ExecutionTest, TensorDmaLdsAddressDoesNotWrapPastTheAllocation) {
  using namespace amdgpu::tensor_dma_detail;
  constexpr uint64_t kGlobal = 0x1a8000;
  constexpr uint64_t kGlobalValue = 0x0123456789abcdefu;
  constexpr uint8_t kLdsSentinel = 0x5e;
  for (const uint32_t lds_base : {0xfffff800u, UINT32_MAX}) {
    for (const bool store : {false, true}) {
      SCOPED_TRACE(testing::Message() << std::hex << lds_base << (store ? " store" : " load"));
      Gfx1250Sim sim;
      ASSERT_NO_FATAL_FAILURE(sim.initialize());
      auto *cu = sim.cu();
      auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
      ASSERT_NE(wf, nullptr);
      const uint32_t below = cu->allocate_lds(2048);
      wf->set_lds_base(cu->allocate_lds(2048));
      wf->set_lds_size(2048);
      ASSERT_EQ(wf->lds_base(), below + 2048);
      for (uint32_t byte = 0; byte < 4096; ++byte)
        cu->lds().write8(below + byte, kLdsSentinel);
      sim.memory->write64(kGlobal, kGlobalValue);

      TensorDmaDescriptor desc;
      desc.count = 1;
      desc.elem_size = 8;
      desc.tensor_rank = 1;
      desc.tensor_dims[0] = 1;
      desc.tile_dims[0] = 1;
      desc.global_base = kGlobal;
      desc.lds_base = lds_base;
      TensorDmaState state(desc, store);
      ASSERT_TRUE(prepare_tensor(state, *wf).succeeded());
      ASSERT_EQ(state.elements.size(), 1u);
      EXPECT_EQ(state.elements[0].lds_address, amdgpu::kInvalidLdsAddress);
      EXPECT_EQ(state.elements[0].lds_dropped_bytes, desc.elem_size);
      EXPECT_TRUE(tensor_dma_lds_ranges(state).empty());

      EXPECT_EQ(resume_tensor_dma_state(state, *wf), amdgpu::VmAccessOutcome::Complete);
      for (uint32_t byte = 0; byte < 4096; ++byte)
        EXPECT_EQ(cu->lds().read8(below + byte), kLdsSentinel) << byte;
      // A load's data is discarded. A store reads its dropped LDS bytes as zero,
      // which is the simulator's modeling choice rather than an ISA rule.
      EXPECT_EQ(sim.memory->read64(kGlobal), store ? 0u : kGlobalValue);
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaOverflowSkipsCopyAndBarrier) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  constexpr uint64_t kGlobal = 0x1a0000;
  constexpr uint32_t kBarrierAddr = 64;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18));
  write_wave_sgpr(*cu, *wf, 13, (1u << 16) | (kBarrierAddr >> 3));
  // The five encoded tile dimensions multiply to 2^64, which wrapped to zero.
  write_wave_sgpr(*cu, *wf, 15, 32768u << 16);
  write_wave_sgpr(*cu, *wf, 16, 32768u | (32768u << 16));
  write_wave_sgpr(*cu, *wf, 23, 32768u << 16);
  write_wave_sgpr(*cu, *wf, 26, 16u << 16);
  write_global_u32(*sim.memory, kGlobal, 0x12345678u);
  cu->lds().write32(wf->lds_base(), 0xabcdef00u);
  cu->lds().write64(wf->lds_base() + kBarrierAddr, 0);

  for (const auto &words : {std::array<uint32_t, 3>{0xd0710001u, 0x7c000000u, 0x18140c00u},
                            std::array<uint32_t, 3>{0xd0714001u, 0x7c000000u, 0x18140c00u}}) {
    auto inst = decode_gfx1250(words, words[0] == 0xd0714001u ? "tensor_store_from_lds"
                                                              : "tensor_load_to_lds");
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).failed());
    EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0xabcdef00u);
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), 0x12345678u);
    EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierAddr), 0u);
    EXPECT_TRUE(wf->wait_counters().empty());
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaBarrierMustFitOwnedLds) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  cu->allocate_lds(2048);
  wf->set_lds_base(cu->allocate_lds(2048));
  wf->set_lds_size(2048);
  const uint32_t neighbour = cu->allocate_lds(2048);
  ASSERT_EQ(neighbour, wf->lds_base() + wf->lds_size());
  constexpr uint64_t kGlobal = 0x1b0000;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18));
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  const std::array<uint32_t, 3> words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  for (bool store : {false, true}) {
    auto encoding = words;
    if (store)
      encoding[0] = 0xd0714001u;
    for (uint32_t barrier : {2048u, 0x7fff8u}) {
      SCOPED_TRACE(store);
      SCOPED_TRACE(barrier);
      write_wave_sgpr(*cu, *wf, 13, (1u << 16) | (barrier >> 3));
      write_global_u32(*sim.memory, kGlobal, 0x12345678u);
      cu->lds().write32(wf->lds_base(), 0xabcdef00u);
      cu->lds().write64(neighbour, 0);
      auto inst = decode_gfx1250(encoding, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
      ASSERT_NE(inst, nullptr);
      EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).failed());
      EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0xabcdef00u);
      EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), 0x12345678u);
      EXPECT_EQ(cu->lds().read64(neighbour), 0u);
      EXPECT_TRUE(wf->wait_counters().empty());
    }
    write_wave_sgpr(*cu, *wf, 13, (1u << 16) | (248u >> 3));
    cu->lds().write64(wf->lds_base() + 248, 0);
    auto valid = decode_gfx1250(encoding, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
    ASSERT_NE(valid, nullptr);
    EXPECT_TRUE(cu->execute_instruction(valid.get(), *wf).succeeded());
    EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(cu->lds().read64(wf->lds_base() + 248)));
  }

  using namespace amdgpu::tensor_dma_detail;
  TensorDmaDescriptor empty;
  empty.count = 1;
  empty.elem_size = 4;
  empty.atomic_barrier = true;
  empty.atomic_barrier_addr = 2048;
  TensorDmaState invalid_empty(empty, false);
  EXPECT_TRUE(prepare_tensor(invalid_empty, *wf).failed());
  EXPECT_TRUE(invalid_empty.elements.empty());
  empty.atomic_barrier_addr = 248;
  TensorDmaState valid_empty(empty, false);
  ASSERT_TRUE(prepare_tensor(valid_empty, *wf).succeeded());
  cu->lds().write64(wf->lds_base() + 248, 0);
  EXPECT_EQ(resume_tensor_dma_state(valid_empty, *wf), amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(cu->lds().read64(wf->lds_base() + 248)));
}

TEST(Gfx1250ExecutionTest, TensorDmaObservationReportsCompletedDenseLoadAndStore) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();

  constexpr uint32_t kWavefrontId = 5;
  constexpr uint32_t kWorkgroupId = 17;
  constexpr uint64_t kPc = 0x123450;
  constexpr uint32_t kDispatchId = 23;
  constexpr uint32_t kQueueId = 29;
  constexpr uint32_t kProcessId = 31;
  auto *wf = cu->dispatch_wf_at(kWavefrontId, kWorkgroupId, kPc, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(kDispatchId);
  wf->set_queue_id(kQueueId);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kLoadGlobal = 0x182000;
  constexpr uint64_t kStoreGlobal = 0x183000;
  constexpr uint32_t kTensorCols = 2;
  constexpr uint32_t kTensorRows = 2;
  constexpr uint32_t kTileCols = 3;
  constexpr uint32_t kTileRows = 3;
  constexpr uint32_t kGlobalRowStride = 4;
  constexpr uint32_t kGlobalPlaneStride = 7;
  constexpr uint32_t kBarrierLdsAddr = 128;

  KfdProcess process(kProcessId);
  std::array<uint32_t, kGlobalRowStride * kTensorRows> load_storage{};
  std::array<uint32_t, kGlobalRowStride * kTensorRows> store_storage{};
  process.map_pages(kLoadGlobal, load_storage.data(), sizeof(load_storage));
  process.map_pages(kStoreGlobal, store_storage.data(), sizeof(store_storage));
  amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
  const amdgpu::AddressSpaceHandle address_space =
      legacy_vm.register_address_space(kProcessId, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(address_space);
  wf->set_address_space(address_space);

  write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
  write_tensor_dma_d0(*cu, *wf, 8, kStoreGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18)); // i32, atomic barrier.
  write_wave_sgpr(*cu, *wf, 13, (kTensorCols << 16) | (kBarrierLdsAddr >> 3));
  write_wave_sgpr(*cu, *wf, 14, kTensorRows << 16);
  write_wave_sgpr(*cu, *wf, 15, kTileCols << 16);
  write_wave_sgpr(*cu, *wf, 16, kTileRows);
  write_wave_sgpr(*cu, *wf, 17, kGlobalRowStride);
  for (uint32_t reg = 18; reg < 28; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);
  // D1 word 6 contains stride-0[47:32] followed by stride-1[15:0].
  write_wave_sgpr(*cu, *wf, 18, kGlobalPlaneStride << 16);

  for (uint32_t row = 0; row < kTensorRows; ++row) {
    for (uint32_t col = 0; col < kTensorCols; ++col) {
      load_storage[row * kGlobalRowStride + col] = 0x81000000u + row * 0x100u + col;
    }
  }
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  bool load_was_complete_at_callback = false;
  observer.inspect = [&](const amdgpu::TensorDmaMemoryAccessObservation &access) {
    if (!access.is_load)
      return;
    const uint64_t barrier = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
    load_was_complete_at_callback =
        cu->lds().read32(wf->lds_base() + 0 * sizeof(uint32_t)) == 0x81000000u &&
        cu->lds().read32(wf->lds_base() + 1 * sizeof(uint32_t)) == 0x81000001u &&
        cu->lds().read32(wf->lds_base() + 3 * sizeof(uint32_t)) == 0x81000100u &&
        cu->lds().read32(wf->lds_base() + 4 * sizeof(uint32_t)) == 0x81000101u &&
        amdgpu::lds_barrier_cell_phase_parity(barrier);
  };

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());

  ASSERT_EQ(observer.accesses.size(), 1u);
  const auto &load_access = observer.accesses[0];
  EXPECT_EQ(load_access.mnemonic, "tensor_load_to_lds");
  EXPECT_EQ(load_access.pc, kPc);
  EXPECT_EQ(load_access.compute_unit_id, cu->id());
  EXPECT_EQ(load_access.dispatch_id, kDispatchId);
  EXPECT_EQ(load_access.queue_id, kQueueId);
  EXPECT_EQ(load_access.workgroup_id, kWorkgroupId);
  EXPECT_EQ(load_access.wavefront_id, kWavefrontId);
  EXPECT_EQ(load_access.process_id, kProcessId);
  EXPECT_EQ(load_access.element_size_bytes, sizeof(uint32_t));
  EXPECT_EQ(load_access.tile_dim0, kTileCols);
  EXPECT_EQ(load_access.tile_dim1, kTileRows);
  EXPECT_EQ(load_access.data_size, 2u);
  EXPECT_EQ(load_access.tensor_dim0_stride,
            static_cast<int64_t>(kGlobalRowStride * sizeof(uint32_t)));
  EXPECT_EQ(load_access.tensor_dim1_stride,
            static_cast<int64_t>(kGlobalPlaneStride * sizeof(uint32_t)));
  EXPECT_TRUE(load_access.is_load);
  EXPECT_EQ(load_access.addresses, (std::vector<uint64_t>{kLoadGlobal, kLoadGlobal + 4,
                                                          kLoadGlobal + 16, kLoadGlobal + 20}));
  EXPECT_TRUE(load_was_complete_at_callback);

  // The same partially out-of-bounds tile stores only its four valid elements.
  // Clear the barrier bit so the store-side callback probes only copy completion.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);
  for (uint32_t i = 0; i < kTileCols * kTileRows; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), 0x82000000u + i);

  bool store_was_complete_at_callback = false;
  observer.inspect = [&](const amdgpu::TensorDmaMemoryAccessObservation &access) {
    if (access.is_load)
      return;
    store_was_complete_at_callback =
        store_storage[0] == 0x82000000u && store_storage[1] == 0x82000001u &&
        store_storage[4] == 0x82000003u && store_storage[5] == 0x82000004u;
  };

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x18140c08u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());

  ASSERT_EQ(observer.accesses.size(), 2u);
  const auto &store_access = observer.accesses[1];
  EXPECT_EQ(store_access.mnemonic, "tensor_store_from_lds");
  EXPECT_EQ(store_access.element_size_bytes, sizeof(uint32_t));
  EXPECT_EQ(store_access.tile_dim0, kTileCols);
  EXPECT_EQ(store_access.tile_dim1, kTileRows);
  EXPECT_EQ(store_access.data_size, 2u);
  EXPECT_EQ(store_access.tensor_dim0_stride,
            static_cast<int64_t>(kGlobalRowStride * sizeof(uint32_t)));
  EXPECT_EQ(store_access.tensor_dim1_stride,
            static_cast<int64_t>(kGlobalPlaneStride * sizeof(uint32_t)));
  EXPECT_FALSE(store_access.is_load);
  EXPECT_EQ(store_access.addresses, (std::vector<uint64_t>{kStoreGlobal, kStoreGlobal + 4,
                                                           kStoreGlobal + 16, kStoreGlobal + 20}));
  EXPECT_TRUE(store_was_complete_at_callback);
  EXPECT_TRUE(legacy_vm.unregister_address_space(address_space));
}

TEST(Gfx1250ExecutionTest, TensorDmaObservationPreservesGatherOrderAndDuplicates) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x184000;
  constexpr uint32_t kCols = 2;
  constexpr uint32_t kRows = 3;
  constexpr std::array<uint32_t, 3> kIndices = {2, 0, 2};
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30) | (1u << 31));
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, static_cast<uint32_t>(kIndices.size()));
  write_wave_sgpr(*cu, *wf, 17, kCols);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, kIndices[0]);
  write_wave_sgpr(*cu, *wf, 21, kIndices[1]);
  write_wave_sgpr(*cu, *wf, 22, kIndices[2]);
  for (uint32_t reg = 23; reg < 28; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);

  for (uint32_t row = 0; row < kRows; ++row)
    for (uint32_t col = 0; col < kCols; ++col)
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * sizeof(uint32_t),
                       0x83000000u + row * 0x100u + col);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());

  ASSERT_EQ(observer.accesses.size(), 1u);
  EXPECT_EQ(observer.accesses[0].tile_dim0, kCols);
  EXPECT_EQ(observer.accesses[0].tile_dim1, kIndices.size());
  EXPECT_EQ(observer.accesses[0].data_size, 2u);
  EXPECT_EQ(observer.accesses[0].tensor_dim0_stride,
            static_cast<int64_t>(kCols * sizeof(uint32_t)));
  EXPECT_EQ(observer.accesses[0].tensor_dim1_stride, 0);
  EXPECT_EQ(observer.accesses[0].addresses,
            (std::vector<uint64_t>{kGlobal + 16, kGlobal + 20, kGlobal, kGlobal + 4, kGlobal + 16,
                                   kGlobal + 20}));
}

TEST(Gfx1250ExecutionTest, TensorDmaObservationIsOptInAndSuppressesEmptyTransfers) {
  ExecutionPluginGroup disabled_group(PluginSinkConfig{});
  EXPECT_FALSE(disabled_group.observes_tensor_dma_memory_access());
  ASSERT_TRUE(disabled_group.add(
      std::make_unique<TensorDmaObservationPlugin>("disabled_tensor_dma", false)));
  EXPECT_FALSE(disabled_group.observes_tensor_dma_memory_access());

  ExecutionPluginGroup enabled_group(PluginSinkConfig{});
  ASSERT_TRUE(
      enabled_group.add(std::make_unique<TensorDmaObservationPlugin>("enabled_tensor_dma", true)));
  EXPECT_TRUE(enabled_group.observes_tensor_dma_memory_access());

  ExecutionPluginGroup mixed_group(PluginSinkConfig{});
  auto quiet_member = std::make_unique<TensorDmaObservationPlugin>("quiet_tensor_dma", false);
  auto *quiet_member_ptr = quiet_member.get();
  auto enabled_member = std::make_unique<TensorDmaObservationPlugin>("observing_tensor_dma", true);
  auto *enabled_member_ptr = enabled_member.get();
  ASSERT_TRUE(mixed_group.add(std::move(quiet_member)));
  ASSERT_TRUE(mixed_group.add(std::move(enabled_member)));
  ASSERT_TRUE(mixed_group.observes_tensor_dma_memory_access());

  const std::array<uint64_t, 2> mixed_addresses{0x1000, 0x1004};
  rocjitsu::amdgpu::TensorDmaMemoryAccessObservation mixed_observation{};
  mixed_observation.element_size_bytes = 4;
  mixed_observation.addresses = std::span<const uint64_t>(mixed_addresses);
  mixed_group.onAmdgpuTensorDmaMemoryAccess(mixed_observation);
  EXPECT_TRUE(quiet_member_ptr->accesses.empty());
  ASSERT_EQ(enabled_member_ptr->accesses.size(), 1u);
  EXPECT_EQ(enabled_member_ptr->accesses[0].addresses, (std::vector<uint64_t>{0x1000, 0x1004}));

  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x185000;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 0); // Disabled descriptor.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  for (uint32_t reg = 16; reg < 20; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  EXPECT_TRUE(observer.accesses.empty());

  // An active descriptor whose tensor extent masks the whole tile also has no
  // actual global access to report.
  write_wave_sgpr(*cu, *wf, 0, 1);
  write_wave_sgpr(*cu, *wf, 13, 0);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  EXPECT_TRUE(observer.accesses.empty());

  // Failed validation cannot produce a completed-transfer observation.
  write_wave_sgpr(*cu, *wf, 0, 2);
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).failed());
  EXPECT_TRUE(observer.accesses.empty());
}

TEST(Gfx1250ExecutionTest, TensorDmaIssueIsOptIn) {
  ExecutionPluginGroup disabled_group(PluginSinkConfig{});
  EXPECT_FALSE(disabled_group.observes_tensor_dma_issue());
  ASSERT_TRUE(disabled_group.add(std::make_unique<TensorDmaIssuePlugin>("quiet", false)));
  EXPECT_FALSE(disabled_group.observes_tensor_dma_issue());
  // Opting in to the completion hook does not subscribe to issue.
  ASSERT_TRUE(disabled_group.add(std::make_unique<TensorDmaObservationPlugin>("completion", true)));
  EXPECT_FALSE(disabled_group.observes_tensor_dma_issue());

  ExecutionPluginGroup enabled_group(PluginSinkConfig{});
  ASSERT_TRUE(enabled_group.add(std::make_unique<TensorDmaIssuePlugin>("observing", true)));
  EXPECT_TRUE(enabled_group.observes_tensor_dma_issue());
  EXPECT_FALSE(enabled_group.observes_tensor_dma_memory_access());

  // A plugin that does not opt in is not called, even in a subscribed group.
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto quiet = std::make_unique<TensorDmaIssuePlugin>("quiet", false);
  auto *quiet_ptr = quiet.get();
  auto observing = std::make_unique<TensorDmaIssuePlugin>("observing", true);
  auto *observing_ptr = observing.get();
  ASSERT_TRUE(group->add(std::move(quiet)));
  ASSERT_TRUE(group->add(std::move(observing)));
  sim.plugin_group = group;
  sim.soc->set_plugin_group(sim.plugin_group);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  write_tensor_dma_d0(*cu, *wf, 0, 0x186000);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  for (uint32_t reg = 16; reg < 20; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);
  const std::array<uint32_t, 3> words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  EXPECT_TRUE(quiet_ptr->issues.empty());
  EXPECT_EQ(observing_ptr->issues.size(), 1u);
}

// Every tensor DMA instruction that issues is reported once, including one that
// moves no data and a NULL descriptor, which the emulator counts on TENSORcnt.
// One that fails validation does not issue.
TEST(Gfx1250ExecutionTest, TensorDmaIssueReportsEveryIssuedInstruction) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto issue_plugin = std::make_unique<TensorDmaIssuePlugin>();
  auto &observer = *issue_plugin;
  auto completion_plugin = std::make_unique<TensorDmaObservationPlugin>();
  auto &completions = *completion_plugin;
  ASSERT_TRUE(group->add(std::move(issue_plugin)));
  ASSERT_TRUE(group->add(std::move(completion_plugin)));
  sim.plugin_group = group;
  sim.soc->set_plugin_group(sim.plugin_group);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  write_tensor_dma_d0(*cu, *wf, 0, 0x187000);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 1u << 16); // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16); // tile dim0.
  for (uint32_t reg = 16; reg < 20; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);
  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(store, nullptr);
  const amdgpu::TensorDmaLdsRange first_element{.address = wf->lds_base(), .bytes = 4};

  struct Case {
    const char *name;
    uint32_t count;
    uint32_t tensor_dim0;
    bool store;
    std::vector<amdgpu::TensorDmaLdsRange> lds_ranges;
    uint32_t tile_dim0 = 1;
    uint32_t tile_dim1 = 0;
  };
  const Case cases[] = {
      {"in-bounds load", 1, 1, false, {first_element}},
      {"in-bounds store", 1, 1, true, {first_element}},
      // A load wholly outside the tensor still zero-fills its tile.
      {"load outside the tensor", 1, 0, false, {first_element}},
      {"store outside the tensor", 1, 0, true, {}},
      {"NULL load", 0, 1, false, {}},
      {"NULL store", 0, 1, true, {}},
      // An active NOP tile issues, but makes no global request to complete.
      {"active {0,1} NOP load", 1, 1, false, {}, /*tile_dim0=*/0, /*tile_dim1=*/1},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.name);
    observer.issues.clear();
    write_wave_sgpr(*cu, *wf, 0, c.count);
    write_wave_sgpr(*cu, *wf, 13, c.tensor_dim0 << 16);
    write_wave_sgpr(*cu, *wf, 15, c.tile_dim0 << 16);
    write_wave_sgpr(*cu, *wf, 16, c.tile_dim1);
    Instruction &inst = c.store ? *store : *load;
    EXPECT_TRUE(cu->execute_instruction(&inst, *wf).succeeded());
    ASSERT_EQ(observer.issues.size(), 1u);
    EXPECT_EQ(observer.issues[0].is_load, !c.store);
    EXPECT_EQ(observer.issues[0].lds_ranges, c.lds_ranges);
  }
  // Only the two in-bounds transfers made global requests to complete.
  EXPECT_EQ(completions.accesses.size(), 2u);

  // An unsupported count fails validation and does not issue.
  observer.issues.clear();
  write_wave_sgpr(*cu, *wf, 0, 2);
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  write_wave_sgpr(*cu, *wf, 16, 0);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).failed());
  EXPECT_TRUE(observer.issues.empty());
}

// A load covers every element of its tile, zero-filling those outside the
// tensor; a store covers only in-bounds elements. Here a 3x2 tile overhangs a
// 2x2 tensor by one column.
TEST(Gfx1250ExecutionTest, TensorDmaIssueCoversTheWholeLoadTileButOnlyInBoundsStoreElements) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_issue_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  write_tensor_dma_d0(*cu, *wf, 0, 0x188000);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16); // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16); // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 3u << 16); // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 2u);       // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 2u);       // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  for (const bool store : {false, true}) {
    const std::array<uint32_t, 3> words = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                           0x7c7c0c00u};
    auto inst = decode_gfx1250(words, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  }
  ASSERT_EQ(observer.issues.size(), 2u);
  EXPECT_EQ(observer.issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{{.address = wf->lds_base(), .bytes = 24}}));
  EXPECT_EQ(observer.issues[1].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{{.address = wf->lds_base(), .bytes = 8},
                                                    {.address = wf->lds_base() + 12, .bytes = 8}}));
}

TEST(Gfx1250ExecutionTest, TensorDmaIssueRangesSkipPadding) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_issue_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  write_tensor_dma_d0(*cu, *wf, 0, 0x120000);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 20)); // i32, one-dword pad every two.
  write_wave_sgpr(*cu, *wf, 13, 4u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 4u << 16); // tile dim0.
  for (uint32_t reg = 16; reg < 20; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);
  const std::array<uint32_t, 3> words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  ASSERT_EQ(observer.issues.size(), 1u);
  EXPECT_EQ(observer.issues[0].lds_ranges,
            (std::vector<amdgpu::TensorDmaLdsRange>{{.address = wf->lds_base(), .bytes = 8},
                                                    {.address = wf->lds_base() + 12, .bytes = 8}}));
}

// The issue hook fires before a load writes LDS. A transfer that waits for
// memory is covered in gpu_vm_pipeline_test.cpp.
TEST(Gfx1250ExecutionTest, TensorDmaIssueFiresBeforeALoadWritesLds) {
  constexpr uint64_t kGlobal = 0x189000;
  constexpr uint32_t kSentinel = 0x5e5e5e5eu;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_issue_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  cu->lds().write32(wf->lds_base(), kSentinel);
  write_global_u32(*sim.memory, kGlobal, 0x12345678u);
  bool lds_untouched_at_issue = false;
  observer.inspect = [&](const amdgpu::TensorDmaIssueObservation &, const amdgpu::Wavefront &) {
    lds_untouched_at_issue = cu->lds().read32(wf->lds_base()) == kSentinel;
  };
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  for (uint32_t reg = 16; reg < 20; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);
  const std::array<uint32_t, 3> words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
  EXPECT_TRUE(lds_untouched_at_issue);
  EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0x12345678u);
  EXPECT_EQ(observer.issues.size(), 1u);
}

// The issue hook fires after a store samples its LDS source and before it writes
// global memory, so overwriting the source in the callback does not change what
// the store writes.
TEST(Gfx1250ExecutionTest, TensorDmaIssueFiresAfterAStoreSamplesLds) {
  constexpr uint64_t kGlobal = 0x189000;
  constexpr uint32_t kSource = 0xaabbccddu;
  constexpr uint32_t kUnwritten = 0xffffffffu;
  constexpr uint32_t kOverwrite = 0x5e5e5e5eu;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_issue_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  cu->lds().write32(wf->lds_base(), kSource);
  write_global_u32(*sim.memory, kGlobal, kUnwritten);
  bool global_untouched_at_issue = false;
  observer.inspect = [&](const amdgpu::TensorDmaIssueObservation &, const amdgpu::Wavefront &) {
    global_untouched_at_issue = read_global_u32(*sim.memory, kGlobal) == kUnwritten;
    cu->lds().write32(wf->lds_base(), kOverwrite);
  };
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  for (uint32_t reg = 16; reg < 20; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);
  const std::array<uint32_t, 3> words = {0xd0714001u, 0x7c000000u, 0x7c7c0c00u};
  auto store = decode_gfx1250(words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());
  EXPECT_TRUE(global_untouched_at_issue);
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), kSource);
  EXPECT_EQ(observer.issues.size(), 1u);
}

TEST(Gfx1250ExecutionTest, TensorDmaZeroTileDim0IsANopWhateverTheHigherTileDims) {
  // A zero tile_dim0 makes the tensor a NOP (CDNA5 ISA 10.11.4, D# field
  // tile_dim0). Pipelines empty a prefetch this way and leave the higher tile
  // dimensions set.
  constexpr uint64_t kGlobal = 0x188000;
  // Distinct values, so a copy in either direction changes one of them.
  constexpr uint32_t kLdsSentinel = 0x5e5e5e5eu;
  constexpr uint32_t kGlobalSentinel = 0xa7a7a7a7u;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_issue_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  cu->lds().write32(wf->lds_base(), kLdsSentinel);
  write_global_u32(*sim.memory, kGlobal, kGlobalSentinel);

  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 4u << 16); // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 4u << 16); // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 0);        // tile dim0 = 0.
  write_wave_sgpr(*cu, *wf, 16, 4);        // tile dim1 = 4.
  write_wave_sgpr(*cu, *wf, 17, 4);        // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (const bool store : {false, true}) {
    SCOPED_TRACE(store ? "store" : "load");
    const std::array<uint32_t, 3> words = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                           0x7c7c0c00u};
    auto inst = decode_gfx1250(words, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  }
  // Each still issues, and touches no LDS.
  ASSERT_EQ(observer.issues.size(), 2u);
  for (const auto &issue : observer.issues)
    EXPECT_TRUE(issue.lds_ranges.empty());
  EXPECT_EQ(cu->lds().read32(wf->lds_base()), kLdsSentinel);
  EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), kGlobalSentinel);
}

// A zero-width tile can drain an iterating prefetch even when every dimension is zero.
TEST(Gfx1250ExecutionTest, TensorDmaIteratingNopTileSkipsRankValidation) {
  constexpr uint64_t kGlobal = 0x18c000;
  // Distinct values, so a copy in either direction changes one of them.
  constexpr uint32_t kLdsSentinel = 0x5e5e5e5eu;
  constexpr uint32_t kGlobalSentinel = 0xa7a7a7a7u;
  struct Case {
    const char *name;
    uint32_t tile_dim0;
    uint32_t tile_dim1;
    uint32_t tensor_dim0_stride;
    bool succeeds;
    bool barrier = false; ///< Request a completion arrival on a barrier cell.
  };
  constexpr Case kCases[] = {
      {.name = "zero-width tile, overlapping stride",
       .tile_dim0 = 0,
       .tile_dim1 = 4,
       .tensor_dim0_stride = 1,
       .succeeds = true},
      {.name = "zero-width tile, zero stride",
       .tile_dim0 = 0,
       .tile_dim1 = 4,
       .tensor_dim0_stride = 0,
       .succeeds = true},
      {.name = "zero-width tile, disjoint stride",
       .tile_dim0 = 0,
       .tile_dim1 = 4,
       .tensor_dim0_stride = 4,
       .succeeds = true},
      // Two iterations of a NOP tile still make one instruction, which arrives
      // once.
      {.name = "zero-width tile, disjoint stride, barrier",
       .tile_dim0 = 0,
       .tile_dim1 = 4,
       .tensor_dim0_stride = 4,
       .succeeds = true,
       .barrier = true},
      {.name = "all-zero tile",
       .tile_dim0 = 0,
       .tile_dim1 = 0,
       .tensor_dim0_stride = 4,
       .succeeds = true},
      {.name = "nonempty rank-one tile",
       .tile_dim0 = 1,
       .tile_dim1 = 0,
       .tensor_dim0_stride = 1,
       .succeeds = false},
  };
  for (const Case &c : kCases) {
    for (const bool store : {false, true}) {
      SCOPED_TRACE(testing::Message() << c.name << (store ? ", store" : ", load"));
      Gfx1250Sim sim;
      ASSERT_NO_FATAL_FAILURE(sim.initialize());
      auto &observer = install_tensor_dma_issue_observer(sim);
      auto *cu = sim.cu();
      auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_lds_base(cu->allocate_lds(256));
      wf->set_lds_size(256);
      for (uint32_t i = 0; i < 16; ++i) {
        cu->lds().write32(wf->lds_base() + i * 4, kLdsSentinel);
        write_global_u32(*sim.memory, kGlobal + i * 4, kGlobalSentinel);
      }
      // The barrier cell lies past the 16 sentinel dwords.
      constexpr uint32_t kBarrierLdsAddr = 128;
      const uint64_t kUnarrived = amdgpu::lds_barrier_cell_init_state(3);
      cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, kUnarrived);

      write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
      const uint32_t barrier_bit = c.barrier ? 1u << 18 : 0;
      const uint32_t barrier_addr = c.barrier ? kBarrierLdsAddr >> 3 : 0;
      write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19) | barrier_bit); // i32, iterate.
      write_wave_sgpr(*cu, *wf, 13, (4u << 16) | barrier_addr);             // tensor dim0, barrier.
      write_wave_sgpr(*cu, *wf, 14, 4u << 16);                              // tensor dim1.
      write_wave_sgpr(*cu, *wf, 15, c.tile_dim0 << 16);
      write_wave_sgpr(*cu, *wf, 16, c.tile_dim1);
      write_wave_sgpr(*cu, *wf, 17, c.tensor_dim0_stride);
      write_wave_sgpr(*cu, *wf, 18, 0);
      write_wave_sgpr(*cu, *wf, 19, 0);
      write_wave_sgpr(*cu, *wf, 20, 0);
      write_wave_sgpr(*cu, *wf, 21, 0);        // LDS increment.
      write_wave_sgpr(*cu, *wf, 22, 1);        // Global increment in elements.
      write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1: two iterations.

      const std::array<uint32_t, 3> words = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                             0x7c140c00u};
      auto inst = decode_gfx1250(words, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(cu->execute_instruction(inst.get(), *wf).succeeded(), c.succeeds);
      if (c.succeeds) {
        EXPECT_EQ(observer.issues.size(), 1u);
        for (const auto &issue : observer.issues)
          EXPECT_TRUE(issue.lds_ranges.empty());
        EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr),
                  c.barrier ? amdgpu::lds_barrier_cell_update_arrive(kUnarrived) : kUnarrived);
      } else {
        EXPECT_TRUE(observer.issues.empty());
      }
      for (uint32_t i = 0; i < 16; ++i) {
        EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * 4), kLdsSentinel) << i;
        EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + i * 4), kGlobalSentinel) << i;
      }
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherAndScatterAreNopsWithZeroWidthOrNoValidIndices) {
  // A gather may have zero valid indices (CDNA5 ISA 10.11.4, D# field
  // tile_dim1, which gather mode uses as the valid-index count), and a zero
  // tile_dim0 makes it a NOP like any other tile.
  constexpr uint64_t kGlobal = 0x189000;
  // Distinct values, so a copy in either direction changes one of them.
  constexpr uint32_t kLdsSentinel = 0x5e5e5e5eu;
  constexpr uint32_t kGlobalSentinel = 0xa7a7a7a7u;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_issue_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  cu->lds().write32(wf->lds_base(), kLdsSentinel);
  write_global_u32(*sim.memory, kGlobal, kGlobalSentinel);

  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30) | (1u << 31)); // 32-bit gather indices.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);                    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                    // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 3u << 16);                    // tensor dim1.
  write_wave_sgpr(*cu, *wf, 17, 2u);                          // tensor dim0 stride.
  for (uint32_t reg = 18; reg < 28; ++reg)
    write_wave_sgpr(*cu, *wf, reg, 0);

  struct EmptyGather {
    uint32_t tile_dim0;
    uint32_t valid_indices;
  };
  // A gather load and a scatter store, for each empty shape.
  for (const bool store : {false, true}) {
    const std::array<uint32_t, 3> words = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                           0x18140c00u};
    auto inst = decode_gfx1250(words, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
    ASSERT_NE(inst, nullptr);
    for (const EmptyGather gather : {EmptyGather{.tile_dim0 = 2, .valid_indices = 0},
                                     EmptyGather{.tile_dim0 = 0, .valid_indices = 2}}) {
      SCOPED_TRACE(testing::Message()
                   << (store ? "store" : "load") << ", tile_dim0 = " << gather.tile_dim0
                   << ", valid_indices = " << gather.valid_indices);
      observer.issues.clear();
      write_wave_sgpr(*cu, *wf, 15, gather.tile_dim0 << 16);
      write_wave_sgpr(*cu, *wf, 16, gather.valid_indices);
      EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      // Each issues once and touches neither LDS nor global memory.
      EXPECT_EQ(observer.issues.size(), 1u);
      if (observer.issues.size() == 1) {
        EXPECT_EQ(observer.issues[0].is_load, !store);
        EXPECT_TRUE(observer.issues[0].lds_ranges.empty());
      }
      EXPECT_EQ(cu->lds().read32(wf->lds_base()), kLdsSentinel);
      EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), kGlobalSentinel);
    }
  }
}

// A zero-width gather is a NOP, but its descriptor is still checked: with 32-bit
// indices it may name at most 8. Eight issues and arrives once; nine fails
// before issue and leaves the barrier untouched, so a NOP shortcut cannot
// bypass the index limit.
TEST(Gfx1250ExecutionTest, TensorDmaZeroWidthGatherStillChecksItsIndexLimit) {
  constexpr uint64_t kGlobal = 0x18e000;
  constexpr uint32_t kBarrierLdsAddr = 64;
  const uint64_t kUnarrived = amdgpu::lds_barrier_cell_init_state(3);
  const uint64_t kArrivedOnce = amdgpu::lds_barrier_cell_update_arrive(kUnarrived);
  for (const uint32_t indices : {8u, 9u}) {
    SCOPED_TRACE(testing::Message() << indices << " indices");
    const bool accepted = indices <= 8;
    Gfx1250Sim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    auto &observer = install_tensor_dma_issue_observer(sim);
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));
    wf->set_lds_size(256);
    cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, kUnarrived);

    write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
    write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30) | (1u << 31)); // 32-bit gather indices.
    write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18));     // i32, atomic barrier enabled.
    write_wave_sgpr(*cu, *wf, 13, (2u << 16) | (kBarrierLdsAddr >> 3)); // tensor dim0, barrier.
    write_wave_sgpr(*cu, *wf, 14, 16u << 16);                           // tensor dim1.
    write_wave_sgpr(*cu, *wf, 15, 0);                                   // tile dim0 = 0.
    write_wave_sgpr(*cu, *wf, 16, indices);                             // valid indices.
    write_wave_sgpr(*cu, *wf, 17, 2u);                                  // tensor dim0 stride.
    for (uint32_t reg = 18; reg < 28; ++reg)
      write_wave_sgpr(*cu, *wf, reg, 0);

    const std::array<uint32_t, 3> words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
    auto inst = decode_gfx1250(words, "tensor_load_to_lds");
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(cu->execute_instruction(inst.get(), *wf).succeeded(), accepted);
    EXPECT_EQ(observer.issues.size(), accepted ? 1u : 0u);
    EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr),
              accepted ? kArrivedOnce : kUnarrived);
  }
}

// With atomic_barrier set, a NOP tile or a gather with no valid indices moves
// no data but still sends exactly one barrier arrival; a NULL tensor sends
// none (see tensor_element_count). A barrier cell expecting three arrivals per
// phase tells zero, one and two arrivals apart.
TEST(Gfx1250ExecutionTest, TensorDmaNopTilesArriveAtTheirBarrierOnceAndNullTensorsDoNot) {
  constexpr uint64_t kGlobal = 0x18a000;
  constexpr uint32_t kBarrierLdsAddr = 64;
  // Distinct values, so a copy in either direction changes one of them.
  constexpr uint32_t kLdsSentinel = 0x5e5e5e5eu;
  constexpr uint32_t kGlobalSentinel = 0xa7a7a7a7u;
  const uint64_t kUnarrived = amdgpu::lds_barrier_cell_init_state(3);
  const uint64_t kArrivedOnce = amdgpu::lds_barrier_cell_update_arrive(kUnarrived);

  struct Case {
    const char *name;
    bool store = false;
    bool gather = false;
    uint32_t count = 1;
    uint32_t tile_dim0 = 0;
    uint32_t valid_indices = 0;
    bool arrives = true;
  };
  constexpr Case kCases[] = {
      {.name = "zero-width load"},
      {.name = "zero-width store", .store = true},
      {.name = "gather without valid indices", .gather = true, .tile_dim0 = 2},
      {.name = "zero-width gather", .gather = true, .valid_indices = 2},
      {.name = "NULL load", .count = 0, .tile_dim0 = 1, .arrives = false},
      {.name = "NULL store", .store = true, .count = 0, .tile_dim0 = 1, .arrives = false},
  };
  for (const Case &c : kCases) {
    SCOPED_TRACE(c.name);
    Gfx1250Sim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    auto &observer = install_tensor_dma_issue_observer(sim);
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));
    wf->set_lds_size(256);
    cu->lds().write32(wf->lds_base(), kLdsSentinel);
    write_global_u32(*sim.memory, kGlobal, kGlobalSentinel);
    cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, kUnarrived);

    write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
    const uint32_t gather_bits = c.gather ? (1u << 30) | (1u << 31) : 0; // 32-bit indices.
    write_wave_sgpr(*cu, *wf, 0, c.count | gather_bits);
    write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18)); // i32, atomic barrier enabled.
    write_wave_sgpr(*cu, *wf, 13, (2u << 16) | (kBarrierLdsAddr >> 3)); // tensor dim0, barrier.
    write_wave_sgpr(*cu, *wf, 14, 3u << 16);                            // tensor dim1.
    write_wave_sgpr(*cu, *wf, 15, c.tile_dim0 << 16);
    write_wave_sgpr(*cu, *wf, 16, c.gather ? c.valid_indices : 1u); // tile dim1 or indices.
    write_wave_sgpr(*cu, *wf, 17, 2u);                              // tensor dim0 stride.
    for (uint32_t reg = 18; reg < 28; ++reg)
      write_wave_sgpr(*cu, *wf, reg, 0);

    const std::array<uint32_t, 3> words = {c.store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                           c.gather ? 0x18140c00u : 0x7c7c0c00u};
    auto inst = decode_gfx1250(words, c.store ? "tensor_store_from_lds" : "tensor_load_to_lds");
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

    EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr),
              c.arrives ? kArrivedOnce : kUnarrived);
    ASSERT_EQ(observer.issues.size(), 1u);
    EXPECT_TRUE(observer.issues[0].lds_ranges.empty());
    EXPECT_EQ(cu->lds().read32(wf->lds_base()), kLdsSentinel);
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal), kGlobalSentinel);
  }
}

// An instruction with no elements makes no global request, so it runs on a
// compute unit without GPU memory, whatever its rank; one with elements still
// requires memory.
TEST(Gfx1250ExecutionTest, TensorDmaEmptyInstructionsNeedNoGpuMemory) {
  struct Case {
    const char *name;
    bool gather = false;
    uint32_t tile_dim0 = 0;
    uint32_t tile_dim1 = 0; ///< Or the valid-index count, for a gather.
    bool succeeds = true;
  };
  constexpr Case kCases[] = {
      {.name = "rank-zero tile"},
      {.name = "zero-width {0,4} tile", .tile_dim1 = 4},
      {.name = "gather without valid indices", .gather = true, .tile_dim0 = 2},
      {.name = "zero-width gather", .gather = true, .tile_dim1 = 2},
      {.name = "one-element tile", .tile_dim0 = 1, .tile_dim1 = 1, .succeeds = false},
  };
  constexpr uint32_t kProcessId = 1255;
  constexpr uint32_t kBarrierLdsAddr = 64;
  const uint64_t kUnarrived = amdgpu::lds_barrier_cell_init_state(3);
  const uint64_t kArrivedOnce = amdgpu::lds_barrier_cell_update_arrive(kUnarrived);
  for (const Case &c : kCases)
    for (const bool store : {false, true})
      // A wave with an address space whose VM binding is gone also needs none.
      for (const bool unregistered_binding : {false, true}) {
        SCOPED_TRACE(testing::Message() << c.name << (store ? ", store" : ", load")
                                        << (unregistered_binding ? ", unregistered binding" : ""));
        KfdProcess process(kProcessId);
        Gfx1250Sim sim;
        ASSERT_NO_FATAL_FAILURE(sim.initialize());
        auto &observer = install_tensor_dma_issue_observer(sim);
        auto *cu = sim.cu();
        auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_lds_base(cu->allocate_lds(256));
        wf->set_lds_size(256);
        cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, kUnarrived);
        if (unregistered_binding) {
          amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
          const amdgpu::AddressSpaceHandle address_space = legacy_vm.register_address_space(
              kProcessId, &process.page_table_, &process.page_table_mutex_,
              process.page_table_generation());
          ASSERT_TRUE(address_space);
          wf->set_process_id(kProcessId);
          wf->set_address_space(address_space);
          ASSERT_TRUE(legacy_vm.unregister_address_space(address_space));
        }
        cu->set_memory(nullptr);
        ASSERT_FALSE(wf->has_gpu_memory());

        write_tensor_dma_d0(*cu, *wf, 0, 0x18d000);
        const uint32_t gather_bits = c.gather ? (1u << 30) | (1u << 31) : 0; // 32-bit indices.
        write_wave_sgpr(*cu, *wf, 0, 1u | gather_bits);
        write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18)); // i32, atomic barrier enabled.
        write_wave_sgpr(*cu, *wf, 13, (4u << 16) | (kBarrierLdsAddr >> 3)); // tensor dim0, barrier.
        write_wave_sgpr(*cu, *wf, 14, 4u << 16);                            // tensor dim1.
        write_wave_sgpr(*cu, *wf, 15, c.tile_dim0 << 16);
        write_wave_sgpr(*cu, *wf, 16, c.tile_dim1);
        write_wave_sgpr(*cu, *wf, 17, 4u); // tensor dim0 stride.
        for (uint32_t reg = 18; reg < 28; ++reg)
          write_wave_sgpr(*cu, *wf, reg, 0);

        const std::array<uint32_t, 3> words = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                               c.gather ? 0x18140c00u : 0x7c7c0c00u};
        auto inst = decode_gfx1250(words, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
        ASSERT_NE(inst, nullptr);
        // Fatal: a failed instruction attaches no state for tensor_dma_outcome().
        ASSERT_EQ(cu->execute_instruction(inst.get(), *wf).succeeded(), c.succeeds);
        if (c.succeeds) {
          EXPECT_EQ(observer.issues.size(), 1u);
          if (observer.issues.size() == 1) {
            EXPECT_TRUE(observer.issues[0].lds_ranges.empty());
          }
          EXPECT_EQ(amdgpu::tensor_dma_outcome(*inst), amdgpu::VmAccessOutcome::Complete);
          EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr), kArrivedOnce);
        } else {
          // Rejected before issue: nothing is recorded and nothing arrives.
          EXPECT_TRUE(observer.issues.empty());
          EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr), kUnarrived);
        }
      }
}

// Because it still arrives, a NOP tile validates its barrier address: one past
// the allocation fails the instruction, while one inside it succeeds. Both NOP
// shapes are covered: a rank-zero tile, which was already accepted, and a {0,1}
// zero-width tile, which is accepted only as a NOP. A NULL tensor ignores the
// address.
TEST(Gfx1250ExecutionTest, TensorDmaNopTileValidatesItsBarrierAddressAndNullTensorDoesNot) {
  struct Case {
    const char *name;
    uint32_t count;
    uint32_t tile_dim0;
    uint32_t tile_dim1;
    bool validates_barrier;
  };
  constexpr Case kCases[] = {
      {.name = "rank-zero NOP",
       .count = 1,
       .tile_dim0 = 0,
       .tile_dim1 = 0,
       .validates_barrier = true},
      {.name = "zero-width NOP",
       .count = 1,
       .tile_dim0 = 0,
       .tile_dim1 = 1,
       .validates_barrier = true},
      {.name = "NULL", .count = 0, .tile_dim0 = 1, .tile_dim1 = 0, .validates_barrier = false},
  };
  for (const Case &c : kCases) {
    for (const bool store : {false, true}) {
      for (const uint32_t barrier : {248u, 2048u}) {
        SCOPED_TRACE(testing::Message()
                     << c.name << (store ? " store" : " load") << ", barrier at " << barrier);
        Gfx1250Sim sim;
        ASSERT_NO_FATAL_FAILURE(sim.initialize());
        auto *cu = sim.cu();
        auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_lds_base(cu->allocate_lds(2048));
        wf->set_lds_size(2048);
        const uint32_t neighbour = cu->allocate_lds(2048);
        ASSERT_EQ(neighbour, wf->lds_base() + wf->lds_size());
        cu->lds().write64(neighbour, 0);

        write_tensor_dma_d0(*cu, *wf, 0, 0x18b000);
        write_wave_sgpr(*cu, *wf, 0, c.count);
        write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18));     // i32, atomic barrier enabled.
        write_wave_sgpr(*cu, *wf, 13, (1u << 16) | (barrier >> 3)); // tensor dim0, barrier.
        write_wave_sgpr(*cu, *wf, 14, 0);
        write_wave_sgpr(*cu, *wf, 15, c.tile_dim0 << 16);
        write_wave_sgpr(*cu, *wf, 16, c.tile_dim1);
        for (uint32_t reg = 17; reg < 20; ++reg)
          write_wave_sgpr(*cu, *wf, reg, 0);

        const std::array<uint32_t, 3> words = {store ? 0xd0714001u : 0xd0710001u, 0x7c000000u,
                                               0x7c7c0c00u};
        auto inst = decode_gfx1250(words, store ? "tensor_store_from_lds" : "tensor_load_to_lds");
        ASSERT_NE(inst, nullptr);
        const bool past_the_end = barrier + 8 > 2048;
        EXPECT_EQ(cu->execute_instruction(inst.get(), *wf).succeeded(),
                  !(c.validates_barrier && past_the_end));
        EXPECT_EQ(cu->lds().read64(neighbour), 0u);
      }
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaUsesWaveProcessPageTable) {
  constexpr uint32_t kProcessId = 1250;
  constexpr uint32_t kElements = 4;
  constexpr uint64_t kLoadGlobal = 0x20000000;
  constexpr uint64_t kStoreGlobal = kLoadGlobal + KfdProcess::kPageSize;
  constexpr std::array<uint32_t, kElements> kLoadValues = {
      0x11000000u,
      0x22000000u,
      0x33000000u,
      0x44000000u,
  };
  constexpr std::array<uint32_t, kElements> kStoreValues = {
      0x55000000u,
      0x66000000u,
      0x77000000u,
      0x88000000u,
  };

  KfdProcess process(kProcessId);
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  std::array<uint32_t, kElements> load_storage = kLoadValues;
  std::array<uint32_t, kElements> store_storage{};
  process.map_pages(kLoadGlobal, load_storage.data(), sizeof(load_storage));
  process.map_pages(kStoreGlobal, store_storage.data(), sizeof(store_storage));
  amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
  const amdgpu::AddressSpaceHandle address_space =
      legacy_vm.register_address_space(kProcessId, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(address_space);
  wf->set_address_space(address_space);

  // The same GPU VA intentionally resolves to different storage for VMID zero
  // and for the dispatched process. Tensor DMA must use the wave's process ID.
  EXPECT_EQ(sim.memory->read32(kLoadGlobal), 0u);
  const auto access = sim.soc->gpu_vm().snapshot(address_space);
  ASSERT_TRUE(access);
  uint32_t translated_value = 0;
  ASSERT_EQ(access->read(kLoadGlobal, std::as_writable_bytes(std::span(&translated_value, 1))),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(translated_value, kLoadValues[0]);

  write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);        // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kElements << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, kElements << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);
  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), kLoadValues[i]);

  for (uint32_t i = 0; i < kElements; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kStoreValues[i]);

  write_tensor_dma_d0(*cu, *wf, 0, kStoreGlobal);
  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  store->execute(*store, wf);
  EXPECT_EQ(store_storage, kStoreValues);

  EXPECT_TRUE(legacy_vm.unregister_address_space(address_space));
}

TEST(Gfx1250ExecutionTest, TensorDmaD2CopiesGlobalAndLds) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint32_t kElements = 16;
  constexpr uint64_t kLoadGlobal = 0x100000;
  constexpr uint64_t kStoreGlobal = 0x110000;
  auto write_sgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + reg, value);
  };
  auto write_d0 = [&](uint32_t reg, uint64_t global_addr) {
    write_sgpr(reg + 0, 1);
    write_sgpr(reg + 1, 0);
    write_sgpr(reg + 2, static_cast<uint32_t>(global_addr));
    write_sgpr(reg + 3, static_cast<uint32_t>((global_addr >> 32) & 0x01ffffffu) | 0x80000000u);
  };

  write_d0(0, kLoadGlobal);
  write_d0(8, kStoreGlobal);
  write_sgpr(12, 2u << 16);        // i32 elements.
  write_sgpr(13, kElements << 16); // tensor dim0.
  write_sgpr(14, 0);
  write_sgpr(15, kElements << 16); // tile dim0.
  write_sgpr(16, 0);
  write_sgpr(17, 0);
  write_sgpr(18, 0);
  write_sgpr(19, 0);

  for (uint32_t i = 0; i < kElements; ++i) {
    const uint32_t value = 0x11000000u + i * 0x101u;
    for (uint32_t byte = 0; byte < 4; ++byte)
      sim.memory->write8(kLoadGlobal + i * 4 + byte, static_cast<uint8_t>(value >> (byte * 8)));
  }

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);
  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * 4), 0x11000000u + i * 0x101u);

  for (uint32_t i = 0; i < kElements; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, 0x22000000u + i * 0x303u);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c08u};
  cdna5::TensorStoreFromLdsVimage store_inst(store_words.data());
  store_inst.execute_impl(*wf);
  for (uint32_t i = 0; i < kElements; ++i) {
    const uint32_t actual = read_global_u32(*sim.memory, kStoreGlobal + i * 4);
    EXPECT_EQ(actual, 0x22000000u + i * 0x303u);
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaDecodeExecuteCoversLoadStoreD2D4Forms) {
  constexpr uint32_t kNullSgpr = 124;
  constexpr std::array<uint32_t, 3> kLoadD2 = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  constexpr std::array<uint32_t, 3> kStoreD2 = {0xd0714001u, 0x7c000000u, 0x7c7c0c08u};
  constexpr std::array<uint32_t, 3> kLoadD4 = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  constexpr std::array<uint32_t, 3> kStoreD4 = {0xd0714001u, 0x7c000000u, 0x18140c08u};

  {
    Gfx1250Sim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));
    wf->set_lds_size(256);

    constexpr uint32_t kElements = 4;
    constexpr uint64_t kLoadGlobal = 0x1a0000;
    constexpr uint64_t kStoreGlobal = 0x1b0000;

    write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
    write_tensor_dma_d0(*cu, *wf, 8, kStoreGlobal);
    write_wave_sgpr(*cu, *wf, 12, 2u << 16);        // i32 elements.
    write_wave_sgpr(*cu, *wf, 13, kElements << 16); // Tensor dim0.
    write_wave_sgpr(*cu, *wf, 14, 0);
    write_wave_sgpr(*cu, *wf, 15, kElements << 16); // Tile dim0.
    write_wave_sgpr(*cu, *wf, 16, 0);
    write_wave_sgpr(*cu, *wf, 17, 0);
    write_wave_sgpr(*cu, *wf, 18, 0);
    write_wave_sgpr(*cu, *wf, 19, 0);

    for (uint32_t i = 0; i < kElements; ++i)
      write_global_u32(*sim.memory, kLoadGlobal + i * 4, 0x71000000u + i);

    auto load = decode_gfx1250(kLoadD2, "tensor_load_to_lds");
    ASSERT_NE(load, nullptr);
    ASSERT_EQ(load->num_src_operands(), 4);
    EXPECT_EQ(load->src_operand(2)->encoding_value(), static_cast<int>(kNullSgpr));
    EXPECT_EQ(load->src_operand(3)->encoding_value(), static_cast<int>(kNullSgpr));
    load->execute(*load, wf);

    for (uint32_t i = 0; i < kElements; ++i)
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * 4), 0x71000000u + i);

    for (uint32_t i = 0; i < kElements; ++i)
      cu->lds().write32(wf->lds_base() + i * 4, 0x72000000u + i);

    auto store = decode_gfx1250(kStoreD2, "tensor_store_from_lds");
    ASSERT_NE(store, nullptr);
    ASSERT_EQ(store->num_src_operands(), 4);
    EXPECT_EQ(store->src_operand(2)->encoding_value(), static_cast<int>(kNullSgpr));
    EXPECT_EQ(store->src_operand(3)->encoding_value(), static_cast<int>(kNullSgpr));
    store->execute(*store, wf);

    for (uint32_t i = 0; i < kElements; ++i)
      EXPECT_EQ(read_global_u32(*sim.memory, kStoreGlobal + i * 4), 0x72000000u + i);
  }

  {
    Gfx1250Sim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));
    wf->set_lds_size(256);

    constexpr uint32_t kCols = 2;
    constexpr uint32_t kRows = 2;
    constexpr uint32_t kTileRows = 3;
    constexpr uint32_t kDepth = 2;
    constexpr uint32_t kPlaneStride = kTileRows * kCols;
    constexpr uint64_t kLoadGlobal = 0x1c0000;
    constexpr uint64_t kStoreGlobal = 0x1d0000;
    constexpr uint32_t kSentinel = 0x7e7e7e7eu;

    write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
    write_tensor_dma_d0(*cu, *wf, 8, kStoreGlobal);
    write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
    write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
    write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
    write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
    write_wave_sgpr(*cu, *wf, 16, kTileRows | (kDepth << 16));
    write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
    write_wave_sgpr(*cu, *wf, 18, kPlaneStride << 16);
    write_wave_sgpr(*cu, *wf, 19, 0);
    write_wave_sgpr(*cu, *wf, 20, kDepth); // Tensor dim2, from D2.
    write_wave_sgpr(*cu, *wf, 21, 0);
    write_wave_sgpr(*cu, *wf, 22, 0);
    write_wave_sgpr(*cu, *wf, 23, 0);
    write_wave_sgpr(*cu, *wf, 24, 0);
    write_wave_sgpr(*cu, *wf, 25, 0);
    write_wave_sgpr(*cu, *wf, 26, 0);
    write_wave_sgpr(*cu, *wf, 27, 0);

    for (uint32_t z = 0; z < kDepth; ++z) {
      for (uint32_t y = 0; y < kTileRows; ++y) {
        for (uint32_t x = 0; x < kCols; ++x) {
          const uint64_t offset = static_cast<uint64_t>(z * kPlaneStride + y * kCols + x) * 4;
          const uint32_t value = 0x73000000u + z * 0x100u + y * 0x10u + x;
          write_global_u32(*sim.memory, kLoadGlobal + offset, value);
          write_global_u32(*sim.memory, kStoreGlobal + offset, kSentinel);
        }
      }
    }

    auto load = decode_gfx1250(kLoadD4, "tensor_load_to_lds");
    ASSERT_NE(load, nullptr);
    ASSERT_EQ(load->num_src_operands(), 4);
    EXPECT_EQ(load->src_operand(2)->encoding_value(), 20);
    EXPECT_EQ(load->src_operand(3)->encoding_value(), 24);
    load->execute(*load, wf);

    for (uint32_t z = 0; z < kDepth; ++z) {
      for (uint32_t y = 0; y < kTileRows; ++y) {
        for (uint32_t x = 0; x < kCols; ++x) {
          const uint32_t lds_index = z * kTileRows * kCols + y * kCols + x;
          const uint32_t expected = (y < kRows) ? (0x73000000u + z * 0x100u + y * 0x10u + x) : 0u;
          EXPECT_EQ(cu->lds().read32(wf->lds_base() + lds_index * 4), expected);
        }
      }
    }

    for (uint32_t i = 0; i < kDepth * kTileRows * kCols; ++i)
      cu->lds().write32(wf->lds_base() + i * 4, 0x74000000u + i);

    auto store = decode_gfx1250(kStoreD4, "tensor_store_from_lds");
    ASSERT_NE(store, nullptr);
    ASSERT_EQ(store->num_src_operands(), 4);
    EXPECT_EQ(store->src_operand(2)->encoding_value(), 20);
    EXPECT_EQ(store->src_operand(3)->encoding_value(), 24);
    store->execute(*store, wf);

    for (uint32_t z = 0; z < kDepth; ++z) {
      for (uint32_t y = 0; y < kTileRows; ++y) {
        for (uint32_t x = 0; x < kCols; ++x) {
          const uint64_t offset = static_cast<uint64_t>(z * kPlaneStride + y * kCols + x) * 4;
          const uint32_t lds_index = z * kTileRows * kCols + y * kCols + x;
          const uint32_t expected = (y < kRows) ? (0x74000000u + lds_index) : kSentinel;
          EXPECT_EQ(read_global_u32(*sim.memory, kStoreGlobal + offset), expected);
        }
      }
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaLoadAppliesLdsPadding) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x120000;
  constexpr uint32_t kSentinel = 0xCAFECAFEu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 20)); // i32, pad enabled.
  write_wave_sgpr(*cu, *wf, 13, 4u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 4u << 16); // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < 4; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x33000000u + i);
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);
  }
  cu->lds().write32(wf->lds_base() + 4 * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x33000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x33000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 3 * 4), 0x33000002u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * 4), 0x33000003u);
}

// Each iterated tile starts lds_increment elements after the previous one
// (CDNA5 ISA 10.11.3.1), and padding applies within a tile (10.11.2). The ISA
// leaves the padding counter's state between tiles unspecified; rocjitsu
// restarts it with each tile (issue #14661). Three i32s per tile with one dword
// of padding per two, starting at 2012 of a 2048-byte allocation with a
// 5-element increment, land at 2012, 2016 and 2024, then 2032, 2036 and 2044,
// exactly where independent loads of each tile put them. The third word of each
// tile follows a padding gap, so dropping padding would put it in the gap, and
// carrying the counter into the second tile would move that tile's words too.
TEST(Gfx1250ExecutionTest, TensorDmaIteratedPaddingRestartsWithEachTile) {
  constexpr uint64_t kGlobal = 0x121000;
  constexpr uint32_t kLdsBytes = 2048;
  constexpr uint32_t kStart = 2012;
  constexpr uint32_t kTileElements = 3;
  constexpr uint32_t kIncrementElements = 5;
  constexpr uint32_t kSentinel = 0xCAFECAFEu;
  constexpr std::array<uint32_t, 2 * kTileElements> kValues = {
      0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u, 0x55000000u, 0x66000000u};
  struct Result {
    std::vector<uint32_t> lds;
    std::vector<amdgpu::TensorDmaLdsRange> ranges;
  };
  // Runs `loads` (LDS offset, global offset in elements, iterations) and returns
  // the wave's LDS from kStart to the end, relative to its base, and the issued
  // ranges.
  auto run = [&](std::initializer_list<std::array<uint32_t, 3>> loads) {
    Gfx1250Sim sim;
    EXPECT_NO_FATAL_FAILURE(sim.initialize());
    auto &observer = install_tensor_dma_issue_observer(sim);
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    EXPECT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(kLdsBytes));
    wf->set_lds_size(kLdsBytes);
    for (uint32_t i = 0; i < kValues.size(); ++i)
      write_global_u32(*sim.memory, kGlobal + i * 4, kValues[i]);
    for (uint32_t byte = kStart; byte < kLdsBytes; byte += 4)
      cu->lds().write32(wf->lds_base() + byte, kSentinel);
    for (const auto &[lds_offset, global_element, iterations] : loads) {
      write_tensor_dma_d0(*cu, *wf, 0, kGlobal + global_element * 4, lds_offset);
      const uint32_t iterate_bit = iterations > 1 ? 1u << 19 : 0;
      write_wave_sgpr(*cu, *wf, 12, (2u << 16) | iterate_bit | (1u << 20)); // i32, pad.
      write_wave_sgpr(*cu, *wf, 13, kTileElements << 16);                   // tensor dim0.
      write_wave_sgpr(*cu, *wf, 14, 2u << 16);                              // tensor dim1.
      write_wave_sgpr(*cu, *wf, 15, kTileElements << 16);                   // tile dim0.
      write_wave_sgpr(*cu, *wf, 16, 1);                                     // tile dim1.
      write_wave_sgpr(*cu, *wf, 17, kTileElements);                         // dim0 stride.
      for (uint32_t reg = 18; reg < 24; ++reg)
        write_wave_sgpr(*cu, *wf, reg, 0);
      if (iterations > 1) {
        write_wave_sgpr(*cu, *wf, 21, kIncrementElements);     // LDS increment.
        write_wave_sgpr(*cu, *wf, 22, kTileElements);          // Global increment.
        write_wave_sgpr(*cu, *wf, 23, (iterations - 1) << 16); // iteration_count - 1.
      }
      const std::array<uint32_t, 3> words = {0xd0710001u, 0x7c000000u,
                                             iterations > 1 ? 0x7c140c00u : 0x7c7c0c00u};
      auto load = decode_gfx1250(words, "tensor_load_to_lds");
      EXPECT_NE(load, nullptr);
      if (load) {
        EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
      }
    }
    Result result;
    for (uint32_t byte = kStart; byte < kLdsBytes; byte += 4)
      result.lds.push_back(cu->lds().read32(wf->lds_base() + byte));
    for (const auto &issue : observer.issues)
      for (const auto &range : issue.lds_ranges)
        result.ranges.push_back({.address = range.address - wf->lds_base(), .bytes = range.bytes});
    return result;
  };

  const Result iterated = run({{kStart, 0, 2}});
  const Result independent =
      run({{kStart, 0, 1}, {kStart + 4 * kIncrementElements, kTileElements, 1}});
  const std::vector<amdgpu::TensorDmaLdsRange> expected_ranges = {{.address = 2012, .bytes = 8},
                                                                  {.address = 2024, .bytes = 4},
                                                                  {.address = 2032, .bytes = 8},
                                                                  {.address = 2044, .bytes = 4}};
  // Bytes 2012..2047, one entry per dword; the padding gaps keep the sentinel.
  const std::vector<uint32_t> expected_lds = {kValues[0], kValues[1], kSentinel,
                                              kValues[2], kSentinel,  kValues[3],
                                              kValues[4], kSentinel,  kValues[5]};
  EXPECT_EQ(independent.lds, expected_lds);
  EXPECT_EQ(independent.ranges, expected_ranges);
  EXPECT_EQ(iterated.lds, independent.lds);
  EXPECT_EQ(iterated.ranges, independent.ranges);
}

TEST(Gfx1250ExecutionTest, TensorDmaByteLoadUsesDwordPaddingUnits) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));
  wf->set_lds_size(512);

  constexpr uint64_t kGlobal = 0x121000;
  constexpr uint8_t kSentinel = 0xA5u;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12,
                  (1u << 20) | (5u << 22) | (3u << 25)); // u8, pad 4 dwords per 64 dwords.
  write_wave_sgpr(*cu, *wf, 13, 260u << 16);             // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 260u << 16); // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < 260; ++i)
    sim.memory->write8(kGlobal + i, static_cast<uint8_t>(i));
  for (uint32_t i = 0; i < 300; ++i)
    cu->lds().write8(wf->lds_base() + i, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 63), 63u);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 64), 64u);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 255), 255u);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(cu->lds().read8(wf->lds_base() + 256 + i), kSentinel);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 272), 0u);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 275), 3u);
}

TEST(Gfx1250ExecutionTest, TensorDmaPaddedStoreIgnoresPadding) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x122000;
  constexpr uint32_t kSentinel = 0xAC1DAC1Du;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 20)); // i32, pad enabled.
  write_wave_sgpr(*cu, *wf, 13, 4u << 16);
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 4u << 16);
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < 4; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), kSentinel);
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), 0x34000000u + i);
  }

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c00u};
  cdna5::TensorStoreFromLdsVimage store_inst(store_words.data());
  store_inst.execute_impl(*wf);

  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t)), 0x34000000u + i);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateCopiesMultipleTiles) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x130000;
  constexpr uint32_t kSentinel = 0xFEEDFEEDu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                      // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 2u);                      // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 4);        // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 2);        // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  for (uint32_t i = 0; i < 4; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x44000000u + i);
  for (uint32_t i = 0; i < 8; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x44000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x44000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 3 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * 4), 0x44000002u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 5 * 4), 0x44000003u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 6 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 7 * 4), kSentinel);
  ASSERT_EQ(observer.accesses.size(), 1u);
  EXPECT_EQ(observer.accesses[0].addresses,
            (std::vector<uint64_t>{kGlobal, kGlobal + 4, kGlobal + 8, kGlobal + 12}));
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateAcceptsAliasedD3ForLoadsAndStores) {
  for (uint32_t rank : {2u, 3u}) {
    // D3 either aliases D2 at s20 or uses a separate zero group at s24.
    for (uint32_t d3 : {20u, 24u}) {
      SCOPED_TRACE("rank=" + std::to_string(rank));
      SCOPED_TRACE("d3=" + std::to_string(d3));
      Gfx1250Sim sim;
      ASSERT_NO_FATAL_FAILURE(sim.initialize());
      auto *cu = sim.cu();
      auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_lds_base(cu->allocate_lds(256));
      wf->set_lds_size(256);

      constexpr uint64_t kGlobal = 0x130000;
      constexpr uint32_t kRowStride = 65536;
      constexpr uint32_t kPlaneStride = 2 * kRowStride;
      constexpr uint32_t kLdsIncrement = 8;
      constexpr uint32_t kSentinel = 0xfeedfeedu;
      const uint32_t planes = rank == 3 ? 2 : 1;
      write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
      write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19));         // i32, iterate enabled.
      write_wave_sgpr(*cu, *wf, 13, 2u << 16);                        // Tensor dim0.
      write_wave_sgpr(*cu, *wf, 14, 2u << 16);                        // Tensor dim1.
      write_wave_sgpr(*cu, *wf, 15, 2u << 16);                        // Tile dim0.
      write_wave_sgpr(*cu, *wf, 16, 1u | (rank == 3 ? 2u << 16 : 0)); // Tile dim1/dim2.
      write_wave_sgpr(*cu, *wf, 17, kRowStride);                      // Tensor dim0 stride.
      write_wave_sgpr(*cu, *wf, 18, (kPlaneStride & 0xffffu) << 16);  // Dim1 stride, low 16.
      write_wave_sgpr(*cu, *wf, 19, kPlaneStride >> 16);              // Dim1 stride, upper 32.
      write_wave_sgpr(*cu, *wf, 20, rank == 3 ? 2 : 0);               // Tensor dim2.
      write_wave_sgpr(*cu, *wf, 21, kLdsIncrement);                   // LDS increment in elements.
      write_wave_sgpr(*cu, *wf, 22, kRowStride); // Global increment in elements.
      write_wave_sgpr(*cu, *wf, 23, 1u << 16);   // iteration_count - 1.
      for (uint32_t reg = 24; reg < 28; ++reg)
        write_wave_sgpr(*cu, *wf, reg, 0);

      for (uint32_t element = 0; element < 16; ++element)
        cu->lds().write32(wf->lds_base() + element * 4, kSentinel);
      for (uint32_t row = 0; row < 2; ++row)
        for (uint32_t plane = 0; plane < planes; ++plane)
          for (uint32_t column = 0; column < 2; ++column) {
            const uint32_t element = row * kLdsIncrement + plane * 2 + column;
            const uint64_t offset = row * kRowStride + plane * kPlaneStride + column;
            write_global_u32(*sim.memory, kGlobal + offset * 4, 0x44000000u + element);
          }

      const uint32_t operands = (d3 << 24) | 0x00140c00u;
      const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, operands};
      cdna5::TensorLoadToLdsVimage load(load_words.data());
      ASSERT_TRUE(cu->execute_instruction(&load, *wf).succeeded());
      for (uint32_t row = 0; row < 2; ++row)
        for (uint32_t element = 0; element < kLdsIncrement; ++element) {
          const uint32_t index = row * kLdsIncrement + element;
          EXPECT_EQ(cu->lds().read32(wf->lds_base() + index * 4),
                    element < planes * 2 ? 0x44000000u + index : kSentinel);
          cu->lds().write32(wf->lds_base() + index * 4, 0x55000000u + index);
        }

      const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, operands};
      cdna5::TensorStoreFromLdsVimage store(store_words.data());
      ASSERT_TRUE(cu->execute_instruction(&store, *wf).succeeded());
      for (uint32_t row = 0; row < 2; ++row)
        for (uint32_t plane = 0; plane < planes; ++plane)
          for (uint32_t column = 0; column < 2; ++column) {
            const uint32_t element = row * kLdsIncrement + plane * 2 + column;
            const uint64_t offset = row * kRowStride + plane * kPlaneStride + column;
            EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + offset * 4), 0x55000000u + element);
          }
      EXPECT_TRUE(wf->wait_counters().empty());
      if (d3 == 24) {
        // A distinct D3 still describes a fifth dimension. Iteration of that
        // rank is unsupported; only an operand alias makes D3 unused.
        write_wave_sgpr(*cu, *wf, 25, 1u << 16); // Tensor dim4.
        write_wave_sgpr(*cu, *wf, 26, 1u << 16); // Tile dim4.
        EXPECT_TRUE(cu->execute_instruction(&load, *wf).failed());
        EXPECT_TRUE(cu->execute_instruction(&store, *wf).failed());
        EXPECT_TRUE(wf->wait_counters().empty());
      }
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaNonIteratingAliasedD3DoesNotExtendTile) {
  for (uint32_t rank : {3u, 4u}) {
    SCOPED_TRACE("rank=" + std::to_string(rank));
    Gfx1250Sim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));
    wf->set_lds_size(256);

    constexpr uint64_t kGlobal = 0x130000;
    constexpr uint32_t kOuterStride = 65536;
    constexpr uint32_t kSentinel = 0xfeedfeedu;
    const uint32_t outer = rank == 4 ? 2 : 1;
    write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
    write_wave_sgpr(*cu, *wf, 12, 2u << 16);        // i32, no iteration.
    write_wave_sgpr(*cu, *wf, 13, 2u << 16);        // Tensor dim0.
    write_wave_sgpr(*cu, *wf, 14, 2u << 16);        // Tensor dim1.
    write_wave_sgpr(*cu, *wf, 15, 2u << 16);        // Tile dim0.
    write_wave_sgpr(*cu, *wf, 16, 2u | (2u << 16)); // Tile dim1/dim2.
    write_wave_sgpr(*cu, *wf, 17, 2u);              // Tensor dim0 stride.
    write_wave_sgpr(*cu, *wf, 18, 4u << 16);        // Tensor dim1 stride.
    write_wave_sgpr(*cu, *wf, 19, 0);
    write_wave_sgpr(*cu, *wf, 20, 2u);                        // Tensor dim2.
    write_wave_sgpr(*cu, *wf, 21, rank == 4 ? 2u : 0u);       // Tensor dim3.
    write_wave_sgpr(*cu, *wf, 22, kOuterStride);              // Aliases tile_dim4 in D3.
    write_wave_sgpr(*cu, *wf, 23, rank == 4 ? 2u << 16 : 0u); // Tile dim3.

    for (uint32_t i = 0; i < 64; ++i)
      cu->lds().write32(wf->lds_base() + i * 4, kSentinel);
    for (uint32_t outer_index = 0; outer_index < outer; ++outer_index)
      for (uint32_t i = 0; i < 8; ++i)
        write_global_u32(*sim.memory, kGlobal + (outer_index * kOuterStride + i) * 4,
                         0x44000000u + outer_index * 8 + i);

    // VADDR2 and VADDR3 both name s20.
    const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x14140c00u};
    cdna5::TensorLoadToLdsVimage load(load_words.data());
    ASSERT_TRUE(cu->execute_instruction(&load, *wf).succeeded());
    for (uint32_t i = 0; i < 64; ++i)
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * 4),
                i < outer * 8 ? 0x44000000u + i : kSentinel);

    for (uint32_t i = 0; i < outer * 8; ++i)
      cu->lds().write32(wf->lds_base() + i * 4, 0x55000000u + i);
    const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x14140c00u};
    cdna5::TensorStoreFromLdsVimage store(store_words.data());
    ASSERT_TRUE(cu->execute_instruction(&store, *wf).succeeded());
    for (uint32_t outer_index = 0; outer_index < outer; ++outer_index)
      for (uint32_t i = 0; i < 8; ++i)
        EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + (outer_index * kOuterStride + i) * 4),
                  0x55000000u + outer_index * 8 + i);
    EXPECT_TRUE(wf->wait_counters().empty());
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherAcceptsUnusedAliasedD3) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x130000;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30) | (1u << 31)); // i32 gather indices.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);                    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);                    // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 5u << 16);                    // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                    // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 4u);                          // Four indices from D2; D3 is unused.
  write_wave_sgpr(*cu, *wf, 17, 1u);                          // Row stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < 4; ++i)
    write_wave_sgpr(*cu, *wf, 20 + i, i + 1);

  for (uint32_t i = 0; i < 5; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x44000000u + i);

  // VADDR2 and VADDR3 both name s20.
  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x14140c00u};
  cdna5::TensorLoadToLdsVimage load(load_words.data());
  ASSERT_TRUE(cu->execute_instruction(&load, *wf).succeeded());
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * 4), 0x44000000u + i + 1);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250ExecutionTest, TensorDmaZeroCountDisablesLoadStoreAndBarrier) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kLoadGlobal = 0x133000;
  constexpr uint64_t kStoreGlobal = 0x134000;
  constexpr uint32_t kElements = 4;
  constexpr uint32_t kBarrierLdsAddr = 64;
  constexpr uint32_t kLdsSentinel = 0xC0DEC0DEu;
  constexpr uint32_t kGlobalSentinel = 0xA11CA11Cu;

  write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
  write_wave_sgpr(*cu, *wf, 0, 0);                        // Count zero disables this descriptor.
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18)); // i32, atomic barrier enabled.
  write_wave_sgpr(*cu, *wf, 13, (kElements << 16) | (kBarrierLdsAddr >> 3));
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, kElements << 16);
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < kElements; ++i) {
    write_global_u32(*sim.memory, kLoadGlobal + i * sizeof(uint32_t), 0x47000000u + i);
    write_global_u32(*sim.memory, kStoreGlobal + i * sizeof(uint32_t), kGlobalSentinel);
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kLdsSentinel);
  }
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), kLdsSentinel);
  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr), 0u);
  EXPECT_TRUE(wf->wait_counters().empty());

  for (uint32_t i = 0; i < kElements; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), 0x48000000u + i);
  write_tensor_dma_d0(*cu, *wf, 0, kStoreGlobal);
  write_wave_sgpr(*cu, *wf, 0, 0); // Keep the store descriptor disabled.

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  store->execute(*store, wf);

  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(read_global_u32(*sim.memory, kStoreGlobal + i * sizeof(uint32_t)), kGlobalSentinel);
  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr), 0u);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250ExecutionTest, TensorDmaUnsupportedCountEncodingsAreRejected) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  write_tensor_dma_d0(*cu, *wf, 0, 0x138000);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32.
  write_wave_sgpr(*cu, *wf, 13, 1u << 16);
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);

  for (uint32_t count : {2u, 3u}) {
    SCOPED_TRACE("count=" + std::to_string(count));
    write_wave_sgpr(*cu, *wf, 0, count);
    EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).failed());
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnsupportedOperandValue);
    EXPECT_TRUE(wf->wait_counters().empty());
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateLoadsAndStoresAtEachGlobalBase) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x131000;
  constexpr uint32_t kCols = 2;
  constexpr uint32_t kTileRows = 2;
  constexpr uint32_t kRowStride = 4;
  constexpr uint32_t kIterations = 2;
  constexpr uint32_t kTotalRows = kTileRows * kIterations;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16);             // tensor dim0.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16);             // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, kTileRows);               // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, kRowStride);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, kCols * kTileRows);      // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, kRowStride * kTileRows); // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, (kIterations - 1) << 16);

  const std::array<uint32_t, 3> load_words{0xd0710001u, 0x7c000000u, 0x7c140c00u};
  const std::array<uint32_t, 3> store_words{0xd0714001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(store, nullptr);

  struct Case {
    uint32_t tensor_rows;
    std::array<bool, kTotalRows> copied_rows;
  };
  constexpr Case cases[]{
      // gfx1250 iteration_bounds.hip case 1 loaded global words 12 and 13 into LDS words 6 and 7.
      // Case 5 stored 0x80000006 and 0x80000007 back to those global words.
      {3, {true, true, true, true}},
      // The model masks the second row in both tiles when the tensor has only one row.
      // The three-row captures above have no out-of-bounds row within either tile.
      {1, {true, false, true, false}},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "tensor_rows=" << test_case.tensor_rows);
    write_wave_sgpr(*cu, *wf, 14, test_case.tensor_rows << 16);
    for (uint32_t global_index = 0; global_index < kTotalRows * kRowStride; ++global_index)
      write_global_u32(*sim.memory, kGlobal + global_index * 4, 1000 + global_index);
    for (uint32_t lds_index = 0; lds_index < kTotalRows * kCols; ++lds_index)
      cu->lds().write32(wf->lds_base() + lds_index * 4, 0x80000000u + lds_index);

    ASSERT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    for (uint32_t lds_index = 0; lds_index < kTotalRows * kCols; ++lds_index) {
      const uint32_t row = lds_index / kCols, col = lds_index % kCols;
      const uint32_t expected = test_case.copied_rows[row] ? 1000 + row * kRowStride + col : 0;
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + lds_index * 4), expected)
          << "LDS word " << lds_index;
      cu->lds().write32(wf->lds_base() + lds_index * 4, 0x80000000u + lds_index);
    }

    ASSERT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());
    for (uint32_t global_index = 0; global_index < kTotalRows * kRowStride; ++global_index) {
      const uint32_t row = global_index / kRowStride, col = global_index % kRowStride;
      const bool copied = col < kCols && test_case.copied_rows[row];
      const uint32_t expected = copied ? 0x80000000u + row * kCols + col : 1000 + global_index;
      EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + global_index * 4), expected)
          << "global word " << global_index;
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaZeroLengthDimensionMasksEntireTile) {
  // A positive tile extent with zero tensor extent uses the ISA out-of-bounds rule.
  // Nonzero LDS distinguishes the modeled zero-fill from the measured both-zero no-op.
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x132000;
  constexpr uint32_t kCols = 2;
  constexpr uint32_t kTileRows = 2;
  constexpr uint32_t kSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);           // tensor dim1 has no valid rows.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, kTileRows);   // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, kCols);       // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t row = 0; row < kTileRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col)
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * 4,
                       0x46000000u + row * 0x100u + col);
  }
  for (uint32_t i = 0; i < kCols * kTileRows; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t i = 0; i < kCols * kTileRows; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), 0u) << "element " << i;
}

TEST(Gfx1250ExecutionTest, TensorDmaRankThreeNullD2MasksTransfersAndCompletes) {
  // Null D2 supplies a zero tensor extent on an active tile axis. This exercises
  // the modeled out-of-bounds load/store behavior, including barrier completion.
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x137000;
  constexpr uint32_t kElements = 2;
  constexpr uint32_t kBarrierLdsAddr = 64;
  constexpr uint32_t kGlobalSentinel = 0xAC1DAC1Du;
  constexpr uint32_t kLdsSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12,
                  (2u << 16) | (1u << 18)); // i32 elements, atomic barrier enabled.
  write_wave_sgpr(*cu, *wf, 13,
                  (2u << 16) | (kBarrierLdsAddr >> 3)); // tensor dim0 and barrier address.
  write_wave_sgpr(*cu, *wf, 14, 1u << 16);              // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kElements << 16);       // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u | (1u << 16));       // tile dim1 and dim2.
  write_wave_sgpr(*cu, *wf, 17, 2u);                    // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 2u << 16);              // tensor dim1 stride.
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < kElements; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x4C000000u + i);
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kLdsSentinel);
  }
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), 0u) << "element " << i;
  EXPECT_EQ(wf->wait_counters().tensorcnt, 0u);
  EXPECT_TRUE(wf->wait_counters().empty());
  uint64_t barrier_state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(barrier_state, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(barrier_state));

  for (uint32_t i = 0; i < kElements; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), kGlobalSentinel);
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), 0x4D000000u + i);
  }
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  store->execute(*store, wf);

  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t)), kGlobalSentinel)
        << "element " << i;
  EXPECT_EQ(wf->wait_counters().tensorcnt, 0u);
  EXPECT_TRUE(wf->wait_counters().empty());
  barrier_state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(barrier_state, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(barrier_state));
}

TEST(Gfx1250ExecutionTest, TensorDmaRankFiveAllowsZeroGlobalStrides) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x138000;
  constexpr uint32_t kValue = 0x4C000000u;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16); // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 0);
  write_wave_sgpr(*cu, *wf, 14, 1u);              // tensor dim0 = 65536.
  write_wave_sgpr(*cu, *wf, 15, 1u | (1u << 16)); // tensor dim1 = 65536, tile dim0 = 1.
  write_wave_sgpr(*cu, *wf, 16, 1u | (1u << 16)); // tile dim1 and dim2.
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 65536u); // tensor dim2.
  write_wave_sgpr(*cu, *wf, 21, 65536u); // tensor dim3.
  write_wave_sgpr(*cu, *wf, 22, 0);
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // tile dim3.
  write_wave_sgpr(*cu, *wf, 24, 0);
  write_wave_sgpr(*cu, *wf, 25, 1u << 16); // tensor dim4.
  write_wave_sgpr(*cu, *wf, 26, 1u << 16); // tile dim4.
  write_wave_sgpr(*cu, *wf, 27, 0);

  write_global_u32(*sim.memory, kGlobal, kValue);
  cu->lds().write32(wf->lds_base(), 0xDEADDEADu);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base()), kValue);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateAdvancesBySingleElements) {
  // Three one-element tiles start at consecutive global addresses.
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x133000;
  constexpr uint32_t kSentinel = 0xBAD0BAD0u;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 1u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                      // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 2u);                      // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 1u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 1u);       // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 2u << 16); // iteration_count - 1.

  for (uint32_t i = 0; i < 3; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x49000000u + i);
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kSentinel);
  }

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x49000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x49000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), 0x49000002u);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateMasksOutOfBoundsPlanesWithinEveryTile) {
  // Exercise the model's tile-local bounds in 3D. With two tile planes and tensor depth one,
  // it zero-fills both second planes. The observer reports only the first plane of each tile.
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x134000;
  constexpr uint32_t kSentinel = 0xFACEFACEu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u | (2u << 16));         // tile dim1 and dim2.
  write_wave_sgpr(*cu, *wf, 17, 2u);                      // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 4u << 16);                // tensor dim1 stride.
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 1u);       // tensor dim2.
  write_wave_sgpr(*cu, *wf, 21, 4u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 4u);       // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  for (uint32_t i = 0; i < 10; ++i)
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x4A000000u + i);
  for (uint32_t i = 0; i < 8; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x4A000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x4A000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), 0u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 3 * 4), 0u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * 4), 0x4A000004u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 5 * 4), 0x4A000005u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 6 * 4), 0u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 7 * 4), 0u);
  ASSERT_EQ(observer.accesses.size(), 1u);
  EXPECT_EQ(observer.accesses[0].addresses,
            (std::vector<uint64_t>{kGlobal, kGlobal + 4, kGlobal + 16, kGlobal + 20}));
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateCopiesOverlappingRows) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x135000;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 21, 2u);                      // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 1u);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c140c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);

  constexpr std::array<uint32_t, 6> global_indices{0, 1, 1, 2, 2, 3};
  for (const uint32_t row_dim : {1u, 2u}) {
    SCOPED_TRACE(row_dim);
    write_wave_sgpr(*cu, *wf, 23, 2u << 16); // Three overlapping tiles.
    // Use two active axes, also when the row resides in dimension 2.
    write_wave_sgpr(*cu, *wf, 14, (row_dim == 1 ? 2u : 1u) << 16); // tensor dim1.
    write_wave_sgpr(*cu, *wf, 16, row_dim == 1 ? 1u : 1u << 16);   // tile dim1 / dim2.
    write_wave_sgpr(*cu, *wf, 17, row_dim == 1 ? 1u : 99u);        // dim0 stride.
    write_wave_sgpr(*cu, *wf, 18, row_dim == 2 ? 1u << 16 : 0u);   // dim1 stride.
    write_wave_sgpr(*cu, *wf, 19, 0);
    write_wave_sgpr(*cu, *wf, 20, row_dim == 2 ? 2u : 0u); // tensor dim2.
    for (uint32_t global_index = 0; global_index < 5; ++global_index)
      write_global_u32(*sim.memory, kGlobal + 4 * global_index, 1000 + global_index);
    ASSERT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    for (uint32_t tile_index = 0; tile_index < global_indices.size(); ++tile_index) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * tile_index),
                1000 + global_indices[tile_index]);
      cu->lds().write32(wf->lds_base() + 4 * tile_index, 2000 + global_indices[tile_index]);
    }
    ASSERT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());
    for (uint32_t global_index = 0; global_index < 4; ++global_index)
      EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 4 * global_index), 2000 + global_index);
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 16), 1004u);
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateChecksAdvancingStepsAcrossUnusedDimension) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x135000;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // Tensor {2, 2, 3}.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);
  write_wave_sgpr(*cu, *wf, 15, 2u << 16); // Tile {2, tile_dim1, 1}.
  write_wave_sgpr(*cu, *wf, 17, 2);        // Strides {2, 4}.
  write_wave_sgpr(*cu, *wf, 18, 4u << 16);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 3);
  write_wave_sgpr(*cu, *wf, 21, 2);        // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // Two iterations.
  const std::array<uint32_t, 3> load_words{0xd0710001u, 0x7c000000u, 0x7c140c00u};
  const std::array<uint32_t, 3> store_words{0xd0714001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(store, nullptr);

  struct Case {
    uint32_t tile_dim1, global_increment;
  };
  constexpr Case cases[] = {
      {0, 2}, // Step along the unused axis.
      {1, 2}, // Same step with that axis active.
      {0, 4}, // Whole row along dimension 2.
      {0, 0}, // Repeated tile at the same origin.
  };
  for (const auto &[tile_dim1, increment] : cases) {
    SCOPED_TRACE(::testing::Message() << "tile_dim1=" << tile_dim1 << " increment=" << increment);
    write_wave_sgpr(*cu, *wf, 16, tile_dim1 | (1u << 16));
    write_wave_sgpr(*cu, *wf, 22, increment);
    std::array<uint32_t, 6> expected_global{};
    for (uint32_t global_index = 0; global_index < expected_global.size(); ++global_index) {
      expected_global[global_index] = 1000 + global_index;
      write_global_u32(*sim.memory, kGlobal + 4 * global_index, expected_global[global_index]);
    }
    constexpr uint32_t kSentinel = 0xdeadbeef;
    for (uint32_t tile_index = 0; tile_index < 4; ++tile_index)
      cu->lds().write32(wf->lds_base() + 4 * tile_index, kSentinel);

    ASSERT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    for (uint32_t tile_index = 0; tile_index < 4; ++tile_index) {
      const uint32_t global_element = (tile_index / 2) * increment + tile_index % 2;
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * tile_index), 1000 + global_element);
      cu->lds().write32(wf->lds_base() + 4 * tile_index, 2000 + global_element);
      expected_global[global_element] = 2000 + global_element;
    }
    ASSERT_TRUE(cu->execute_instruction(store.get(), *wf).succeeded());
    for (uint32_t global_index = 0; global_index < expected_global.size(); ++global_index)
      EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + 4 * global_index),
                expected_global[global_index]);
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateLoadsUnitTilesAcrossStrideLayouts) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x185000;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 21, 1u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // Two iterations.
  for (uint32_t global_index = 0; global_index < 8; ++global_index)
    write_global_u32(*sim.memory, kGlobal + global_index * 4, 1000 + global_index);
  const std::array<uint32_t, 3> load_words{0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);

  struct Case {
    const char *layout;
    uint32_t rows, planes, row_stride, plane_stride, global_increment;
  };
  constexpr Case cases[]{
      {"aliased unit extent", 1, 0, 1, 0, 1},
      {"permuted disjoint strides", 2, 2, 4, 2, 2},
      {"overlapping rows with repeated origin", 2, 0, 1, 0, 0},
      {"padded row and plane strides", 2, 2, 4, 6, 6},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.layout);
    write_wave_sgpr(*cu, *wf, 14, test_case.rows << 16);
    write_wave_sgpr(*cu, *wf, 16, test_case.planes ? (1u << 16) | 1u : 1u);
    write_wave_sgpr(*cu, *wf, 17, test_case.row_stride);
    write_wave_sgpr(*cu, *wf, 18, test_case.plane_stride << 16);
    write_wave_sgpr(*cu, *wf, 20, test_case.planes);
    write_wave_sgpr(*cu, *wf, 22, test_case.global_increment);
    cu->lds().write32(wf->lds_base(), 0xDEADDEADu);
    cu->lds().write32(wf->lds_base() + 4, 0xDEADDEADu);
    ASSERT_TRUE(cu->execute_instruction(load.get(), *wf).succeeded());
    EXPECT_EQ(cu->lds().read32(wf->lds_base()), 1000u);
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4), 1000u + test_case.global_increment);
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateZeroTensorExtentMasksEachTileAndCompletes) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x13B000;
  constexpr uint32_t kElements = 4;
  constexpr uint32_t kBarrierLdsAddr = 64;
  constexpr uint32_t kGlobalSentinel = 0xAC1DAC1Du;
  constexpr uint32_t kLdsSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12,
                  (2u << 16) | (1u << 18) | (1u << 19)); // i32, atomic barrier and iterate enabled.
  write_wave_sgpr(*cu, *wf, 13,
                  (2u << 16) | (kBarrierLdsAddr >> 3)); // tensor dim0 and barrier address.
  write_wave_sgpr(*cu, *wf, 14, 0);                     // tensor dim1 is empty.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);              // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                    // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 1u);                    // overlaps tensor dim0.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 2u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 1u);       // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  for (uint32_t i = 0; i < kElements; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kLdsSentinel);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), 0u) << "element " << i;
  EXPECT_EQ(wf->wait_counters().tensorcnt, 0u);
  EXPECT_TRUE(wf->wait_counters().empty());
  uint64_t barrier_state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(barrier_state, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(barrier_state));

  for (uint32_t i = 0; i < 3; ++i)
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), kGlobalSentinel);
  for (uint32_t i = 0; i < kElements; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), 0x4F000000u + i);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c140c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  store->execute(*store, wf);

  for (uint32_t i = 0; i < 3; ++i)
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t)), kGlobalSentinel)
        << "element " << i;
  EXPECT_EQ(wf->wait_counters().tensorcnt, 0u);
  EXPECT_TRUE(wf->wait_counters().empty());
  barrier_state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(barrier_state, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(barrier_state));
}

TEST(Gfx1250ExecutionTest, TensorDmaAtomicBarrierArrivesAfterCopy) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x140000;
  constexpr uint32_t kBarrierLdsAddr = 64;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18)); // i32, atomic barrier enabled.
  write_wave_sgpr(*cu, *wf, 13, (2u << 16) | (kBarrierLdsAddr >> 3));
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  write_global_u32(*sim.memory, kGlobal + 0 * 4, 0x55000000u);
  write_global_u32(*sim.memory, kGlobal + 1 * 4, 0x55000001u);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x55000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x55000001u);
  const uint64_t state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(state, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));
}

TEST(Gfx1250ExecutionTest, SBarrierWaitIsNoOpForSingleWaveWorkgroup) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->state(), amdgpu::WfState::RUNNING);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const std::array<uint32_t, 2> wait_words = {0xBF94FFFFu, 0u};
  std::unique_ptr<Instruction> wait_inst(decode_valid(*decoder, wait_words.data()));
  ASSERT_NE(wait_inst, nullptr);
  ASSERT_EQ(std::string_view(wait_inst->mnemonic()), "s_barrier_wait");

  EXPECT_TRUE(cu->execute_instruction(wait_inst.get(), *wf).succeeded());

  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
}

TEST(Gfx1250ExecutionTest, SBarrierWaitReleasesOnlyAfterSignalQuorum) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  constexpr uint64_t kEndPgmPc = 0x150000;
  const std::array<uint32_t, 1> endpgm_words = {S_ENDPGM_GFX12};
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(endpgm_words.data()), sizeof(uint32_t),
                         kEndPgmPc);

  auto *wf0 = cu->dispatch_wf(0, kEndPgmPc, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, kEndPgmPc, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const std::array<uint32_t, 2> wait_words = {0xBF94FFFFu, 0u};
  std::unique_ptr<Instruction> wait_inst(decode_valid(*decoder, wait_words.data()));
  ASSERT_NE(wait_inst, nullptr);
  ASSERT_EQ(std::string_view(wait_inst->mnemonic()), "s_barrier_wait");

  EXPECT_TRUE(cu->execute_instruction(wait_inst.get(), *wf0).succeeded());
  EXPECT_EQ(wf0->state(), amdgpu::WfState::BARRIER);
  EXPECT_TRUE(cu->execute_instruction(wait_inst.get(), *wf1).succeeded());
  ASSERT_EQ(wf1->state(), amdgpu::WfState::BARRIER);

  EXPECT_TRUE(wf0->barrier_signal(-1, 0));
  EXPECT_EQ(wf0->state(), amdgpu::WfState::BARRIER);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::BARRIER);
  EXPECT_FALSE(wf1->barrier_signal(-1, 0));
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  EXPECT_FALSE(cu->step());
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf1->is_halted());
}

TEST(Gfx1250ExecutionTest, ReleaseWaitCounterWakesWaitcntWhenTargetIsSatisfied) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);

  wf->wait_counters().increment(amdgpu::WaitCounterType::TENSORCNT);
  wf->set_wait_target_tensorcnt(0);
  ASSERT_EQ(wf->state(), amdgpu::WfState::WAITCNT);

  wf->release_wait_counter(amdgpu::WaitCounterType::TENSORCNT);

  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
}

TEST(Gfx1250ExecutionTest, ReleaseWaitCounterHaltsEndingWaveWhenLastCounterRetires) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);

  wf->wait_counters().increment(amdgpu::WaitCounterType::TENSORCNT);
  wf->end();
  ASSERT_EQ(wf->state(), amdgpu::WfState::ENDING);

  wf->release_wait_counter(amdgpu::WaitCounterType::TENSORCNT);

  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_TRUE(wf->is_halted());
}

TEST(Gfx1250ExecutionTest, DsAtomicAsyncBarrierArriveFlipsRawBarrierPhase) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint32_t kBarrierLdsAddr = 64;
  cu->write_vgpr(wf->vgpr_alloc().base + 0, 0, kBarrierLdsAddr);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, amdgpu::lds_barrier_cell_init_state(1));

  const std::array<uint32_t, 2> words = {0xd9580000u, 0x00000000u};
  auto *arrive_inst = new cdna5::DsAtomicAsyncBarrierArriveB64Vds(words.data());
  arrive_inst->execute_impl(*wf);
  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(arrive_inst, *wf);

  const uint64_t state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(state, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250ExecutionTest, DsAtomicBarrierArriveSubtractsLowDwordAndReturnsOldCell) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  struct Case {
    uint64_t before;
    uint64_t update;
    uint32_t next_state;
  };
  constexpr Case cases[] = {
      {0x0000000300000000ull, 2, 0xe0000003},           {0x0000000300000000ull, 6, 0xe0000003},
      {0x0000000320000002ull, 5, 0x00000003},           // Borrow from a nonzero phase.
      {0x0000000300000000ull, 0x20000000u, 0xe0000000}, // New pending sign bit is clear.
      {0x0000000300000000ull, 0x10000001u, 0xefffffff}, // Pending borrowed, bit 28 clear.
      {0x0000000310000000ull, 0, 0x00000003},           // Zero update, bit 28 already set.
      {0x0000000200000002ull, 0x100000001ull, 1},       {0x0000000200000002ull, 0x100000000ull, 2},
  };
  wf->set_exec((1u << std::size(cases)) - 1);
  const uint32_t base = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < std::size(cases); ++lane) {
    const Case &test_case = cases[lane];
    const uint32_t addr = lane * 8;
    cu->lds().write64(wf->lds_base() + addr, test_case.before);
    cu->write_vgpr(base, lane, addr);
    cu->write_vgpr(base + 2, lane, static_cast<uint32_t>(test_case.update));
    cu->write_vgpr(base + 3, lane, static_cast<uint32_t>(test_case.update >> 32));
  }

  const auto words =
      cdna5::build_vds(cdna5::kDsAtomicBarrierArriveRtnB64Vds, {.addr = 0, .data0 = 2, .vdst = 4});
  auto inst = std::make_unique<cdna5::DsAtomicBarrierArriveRtnB64Vds>(words.data());
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  amdgpu::LocalMemPipeline local_pipeline;
  ASSERT_EQ(local_pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
  for (uint32_t lane = 0; lane < std::size(cases); ++lane) {
    SCOPED_TRACE(lane);
    const Case &test_case = cases[lane];
    EXPECT_EQ(cu->lds().read64(wf->lds_base() + lane * 8),
              (test_case.before & ~amdgpu::kLdsBarrierCellBarStateMask) | test_case.next_state);
    EXPECT_EQ(cu->read_vgpr(base + 4, lane), static_cast<uint32_t>(test_case.before));
    EXPECT_EQ(cu->read_vgpr(base + 5, lane), static_cast<uint32_t>(test_case.before >> 32));
  }
}

TEST(Gfx1250ExecutionTest, LdsBarrierCellHandlesSingleArrivals) {
  uint64_t state = amdgpu::lds_barrier_cell_init_state(2);
  EXPECT_EQ(state, (1ull << 32) | 1ull);

  state = amdgpu::lds_barrier_cell_update_arrive(state);
  EXPECT_EQ(state, 1ull << 32);
  EXPECT_FALSE(amdgpu::lds_barrier_cell_phase_parity(state));

  state = amdgpu::lds_barrier_cell_update_arrive(state);
  EXPECT_EQ(state, (1ull << 32) |
                       (amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift) |
                       1ull);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));

  const uint64_t drained =
      amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(2));
  ASSERT_EQ(amdgpu::lds_barrier_cell_pending_count(drained), 0ull);
  EXPECT_EQ(amdgpu::lds_barrier_cell_update_arrive(drained, 0), drained);
  EXPECT_EQ(amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(3), 0),
            amdgpu::lds_barrier_cell_init_state(3));

  for (uint32_t arrivals_per_phase : {0u, 1u}) {
    state = amdgpu::lds_barrier_cell_init_state(arrivals_per_phase);
    EXPECT_EQ(amdgpu::lds_barrier_cell_pending_count(state), 0ull);
    state = amdgpu::lds_barrier_cell_update_arrive(state);
    EXPECT_EQ(amdgpu::lds_barrier_cell_pending_count(state), 0ull);
    EXPECT_EQ(amdgpu::lds_barrier_cell_phase(state), amdgpu::kLdsBarrierCellPhaseMask);
    EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));
  }

  const uint64_t reserved = 0xabcdull << 48;
  const uint64_t reserved_state =
      amdgpu::lds_barrier_cell_update_arrive(reserved | amdgpu::lds_barrier_cell_init_state(2));
  EXPECT_EQ(reserved_state & amdgpu::kLdsBarrierCellReservedMask, reserved);
  EXPECT_EQ(amdgpu::lds_barrier_cell_init_count(reserved_state), 1ull);
}

TEST(Gfx1250ExecutionTest, TensorDmaDenseDescriptorCopiesDenseRows) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));
  wf->set_lds_size(512);

  constexpr uint64_t kLoadGlobal = 0x150000;
  constexpr uint64_t kStoreGlobal = 0x160000;
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kCols = 4;
  constexpr uint32_t kTileDim1RowCount = 3;
  constexpr uint32_t kSentinel = 0xABCDABCDu;
  constexpr uint32_t kIgnoredD2 = 0xFFFFu;

  write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u);
  write_tensor_dma_d0(*cu, *wf, 8, kStoreGlobal);
  write_wave_sgpr(*cu, *wf, 8, 1u);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, kTileDim1RowCount);
  write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, kIgnoredD2);
  write_wave_sgpr(*cu, *wf, 21, kIgnoredD2);
  write_wave_sgpr(*cu, *wf, 22, kIgnoredD2);
  write_wave_sgpr(*cu, *wf, 23, 0);
  write_wave_sgpr(*cu, *wf, 24, 0);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col) {
      write_global_u32(*sim.memory, kLoadGlobal + (row * kCols + col) * 4,
                       0x66000000u + row * 0x100u + col);
      write_global_u32(*sim.memory, kStoreGlobal + (row * kCols + col) * 4, kSentinel);
    }
  }
  for (uint32_t i = 0; i < (kTileDim1RowCount + 1) * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  for (uint32_t row_idx = 0; row_idx < kTileDim1RowCount; ++row_idx) {
    const uint32_t src_row = row_idx;
    for (uint32_t col = 0; col < kCols; ++col) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + (row_idx * kCols + col) * 4),
                0x66000000u + src_row * 0x100u + col);
    }
  }
  for (uint32_t col = 0; col < kCols; ++col)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + (kTileDim1RowCount * kCols + col) * 4), kSentinel);

  for (uint32_t i = 0; i < kTileDim1RowCount * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, 0x77000000u + i);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x18140c08u};
  cdna5::TensorStoreFromLdsVimage store_inst(store_words.data());
  store_inst.execute_impl(*wf);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col) {
      const uint32_t actual = read_global_u32(*sim.memory, kStoreGlobal + (row * kCols + col) * 4);
      if (row < kTileDim1RowCount)
        EXPECT_EQ(actual, 0x77000000u + row * kCols + col);
      else
        EXPECT_EQ(actual, kSentinel);
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherSupportsI32Indices) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));
  wf->set_lds_size(512);

  constexpr uint64_t kGlobal = 0x170000;
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kCols = 2;
  constexpr std::array<uint32_t, 5> kIndices = {7, 0, 3, 6, 1};
  constexpr uint32_t kSentinel = 0xBCADBCADu;

  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30) | (1u << 31));
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, static_cast<uint32_t>(kIndices.size()));
  write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, kIndices[0]);
  write_wave_sgpr(*cu, *wf, 21, kIndices[1]);
  write_wave_sgpr(*cu, *wf, 22, kIndices[2]);
  write_wave_sgpr(*cu, *wf, 23, kIndices[3]);
  write_wave_sgpr(*cu, *wf, 24, kIndices[4]);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col)
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * 4,
                       0x68000000u + row * 0x100u + col);
  }
  for (uint32_t i = 0; i < kIndices.size() * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  for (uint32_t row_idx = 0; row_idx < kIndices.size(); ++row_idx) {
    const uint32_t src_row = kIndices[row_idx];
    for (uint32_t col = 0; col < kCols; ++col) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + (row_idx * kCols + col) * 4),
                0x68000000u + src_row * 0x100u + col);
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherSupportsI16Indices) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));
  wf->set_lds_size(512);

  constexpr uint64_t kGlobal = 0x180000;
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kCols = 2;
  constexpr std::array<uint32_t, 10> kIndices = {7, 0, 3, 6, 1, 4, 2, 5, 7, 3};
  constexpr uint32_t kSentinel = 0xCDAECDAEu;

  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 31));
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, static_cast<uint32_t>(kIndices.size()));
  write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, kIndices[0] | (kIndices[1] << 16));
  write_wave_sgpr(*cu, *wf, 21, kIndices[2] | (kIndices[3] << 16));
  write_wave_sgpr(*cu, *wf, 22, kIndices[4] | (kIndices[5] << 16));
  write_wave_sgpr(*cu, *wf, 23, kIndices[6] | (kIndices[7] << 16));
  write_wave_sgpr(*cu, *wf, 24, kIndices[8] | (kIndices[9] << 16));
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col)
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * 4,
                       0x69000000u + row * 0x100u + col);
  }
  for (uint32_t i = 0; i < kIndices.size() * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  cdna5::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  for (uint32_t row_idx = 0; row_idx < kIndices.size(); ++row_idx) {
    const uint32_t src_row = kIndices[row_idx];
    for (uint32_t col = 0; col < kCols; ++col) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + (row_idx * kCols + col) * 4),
                0x69000000u + src_row * 0x100u + col);
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherZeroStrideAliasesRows) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x180800;
  constexpr uint32_t kCols = 2;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 31)); // gather enabled.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);       // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16);    // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);       // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16);    // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 2u);             // two valid gather indices.
  write_wave_sgpr(*cu, *wf, 17, 0);              // literal zero row stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0u | (1u << 16)); // gather rows 0 and 1.
  write_wave_sgpr(*cu, *wf, 21, 0);
  write_wave_sgpr(*cu, *wf, 22, 0);
  write_wave_sgpr(*cu, *wf, 23, 0);
  write_wave_sgpr(*cu, *wf, 24, 0);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t col = 0; col < kCols; ++col)
    write_global_u32(*sim.memory, kGlobal + col * sizeof(uint32_t), 0x6B000000u + col);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t row = 0; row < 2; ++row) {
    for (uint32_t col = 0; col < kCols; ++col)
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + (row * kCols + col) * sizeof(uint32_t)),
                0x6B000000u + col);
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherZeroExtentMasksEntireTile) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x181000;
  constexpr uint32_t kTileElements = 2;
  constexpr uint32_t kSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 31)); // gather enabled.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);       // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 0);              // tensor dim0 is empty.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, kTileElements << 16);
  write_wave_sgpr(*cu, *wf, 16, 1u); // one valid gather index.
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 1u); // gather index.
  write_wave_sgpr(*cu, *wf, 21, 0);
  write_wave_sgpr(*cu, *wf, 22, 0);
  write_wave_sgpr(*cu, *wf, 23, 0);
  write_wave_sgpr(*cu, *wf, 24, 0);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t i = 0; i < 3; ++i)
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x6A000000u + i);
  for (uint32_t i = 0; i < kTileElements; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t i = 0; i < kTileElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), 0u) << "element " << i;
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherRankTwoZeroExtentMasksLoadAndStore) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x182000;
  constexpr uint32_t kRows = 4;
  constexpr uint32_t kTileElements = 2;
  constexpr uint32_t kGlobalSentinel = 0xAC1DAC1Du;
  constexpr uint32_t kLdsSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 31)); // gather enabled.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);       // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 0);              // tensor dim0 is empty.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16);    // tensor dim1 keeps rank two.
  write_wave_sgpr(*cu, *wf, 15, kTileElements << 16);
  write_wave_sgpr(*cu, *wf, 16, 1u); // one valid gather index.
  write_wave_sgpr(*cu, *wf, 17, kTileElements);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 1u); // gather index.
  write_wave_sgpr(*cu, *wf, 21, 0);
  write_wave_sgpr(*cu, *wf, 22, 0);
  write_wave_sgpr(*cu, *wf, 23, 0);
  write_wave_sgpr(*cu, *wf, 24, 0);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t i = 0; i < kRows * kTileElements; ++i)
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x50000000u + i);
  for (uint32_t i = 0; i < kTileElements; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kLdsSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t i = 0; i < kTileElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), 0u) << "element " << i;

  for (uint32_t i = 0; i < kRows * kTileElements; ++i)
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), kGlobalSentinel);
  for (uint32_t i = 0; i < kTileElements; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), 0x51000000u + i);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x18140c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  store->execute(*store, wf);

  for (uint32_t i = 0; i < kRows * kTileElements; ++i)
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t)), kGlobalSentinel)
        << "element " << i;
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherRankTwoZeroOuterExtentMasksLoadAndStore) {
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobal = 0x184000;
  constexpr uint32_t kTileElements = 2;
  constexpr uint32_t kGlobalSentinel = 0xAC1DAC1Du;
  constexpr uint32_t kLdsSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 31)); // gather enabled.
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);       // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);       // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);              // tensor dim1 is empty.
  write_wave_sgpr(*cu, *wf, 15, kTileElements << 16);
  write_wave_sgpr(*cu, *wf, 16, 1u); // one valid gather index.
  write_wave_sgpr(*cu, *wf, 17, kTileElements);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0); // gather index.
  write_wave_sgpr(*cu, *wf, 21, 0);
  write_wave_sgpr(*cu, *wf, 22, 0);
  write_wave_sgpr(*cu, *wf, 23, 0);
  write_wave_sgpr(*cu, *wf, 24, 0);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t i = 0; i < kTileElements; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x53000000u + i);
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kLdsSentinel);
  }

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t i = 0; i < kTileElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * sizeof(uint32_t)), 0u) << "element " << i;

  for (uint32_t i = 0; i < kTileElements; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), kGlobalSentinel);
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), 0x54000000u + i);
  }

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x18140c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  store->execute(*store, wf);

  for (uint32_t i = 0; i < kTileElements; ++i)
    EXPECT_EQ(read_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t)), kGlobalSentinel)
        << "element " << i;
}

TEST(Gfx1250ExecutionTest, TensorDmaCoalescingMatchesElementCopiesAtBoundaries) {
  using namespace amdgpu::tensor_dma_detail;
  constexpr uint32_t kProcessId = 81250;
  constexpr uint64_t kGlobal = 0x350000;
  constexpr uint32_t kLdsBytes = 16384;
  KfdProcess process(kProcessId);
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(kLdsBytes));
  wf->set_lds_size(kLdsBytes);
  std::vector<uint8_t> backing(kLdsBytes);
  std::vector<uint8_t> initial(backing.size());
  for (size_t i = 0; i < initial.size(); ++i)
    initial[i] = static_cast<uint8_t>(i * 37 + 11);
  process.map_pages(kGlobal, backing.data(), backing.size());
  amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
  const auto address_space =
      legacy_vm.register_address_space(kProcessId, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(address_space);
  wf->set_address_space(address_space);

  for (uint32_t element_size : {1u, 2u, 4u, 8u}) {
    for (uint32_t shape = 0; shape < 3; ++shape) {
      for (bool padded : {false, true}) {
        for (bool store : {false, true}) {
          SCOPED_TRACE(::testing::Message() << "element_size=" << element_size << " shape=" << shape
                                            << " padded=" << padded << " store=" << store);
          TensorDmaDescriptor desc;
          desc.count = 1;
          desc.tensor_rank = 2;
          desc.elem_size = element_size;
          desc.global_base = kGlobal + 4093; // An element can straddle a GPU page.
          desc.lds_base = 7;
          desc.tensor_dims = {129, 4};
          desc.tile_dims = {130, 3}; // Each row has a masked suffix.
          desc.global_strides[0] = 137;
          desc.pad = padded;
          desc.pad_interval = 8;
          desc.pad_amount = 1;
          desc.iterate = shape == 1;
          desc.iteration_count = 2;
          desc.global_increment = 137;
          desc.lds_increment = 395;
          desc.gather = shape == 2;
          desc.valid_indices = 3;
          desc.gather_indices = {2, 1, 2}; // Preserve duplicate store ordering.
          const TensorDmaLayout layout(desc);
          ASSERT_TRUE(validate_supported_descriptor(desc, layout).succeeded());
          auto reset = [&] {
            std::ranges::copy(initial, backing.begin());
            for (uint32_t i = 0; i < kLdsBytes; ++i)
              cu->lds().write8(wf->lds_base() + i, static_cast<uint8_t>(i * 19 + 3));
          };
          reset();
          TensorDmaState reference(desc, store);
          ASSERT_TRUE(prepare_tensor(reference, *wf).succeeded());
          ASSERT_EQ(resume_tensor_dma_state</*Coalesce=*/false>(reference, *wf),
                    amdgpu::VmAccessOutcome::Complete);
          const auto expected_backing = backing;
          std::vector<uint8_t> expected_lds(kLdsBytes);
          for (uint32_t i = 0; i < kLdsBytes; ++i)
            expected_lds[i] = cu->lds().read8(wf->lds_base() + i);
          reset();
          TensorDmaState batched(desc, store);
          ASSERT_TRUE(prepare_tensor(batched, *wf).succeeded());
          ASSERT_EQ(resume_tensor_dma_state(batched, *wf), amdgpu::VmAccessOutcome::Complete);
          EXPECT_EQ(batched.next_element, reference.next_element);
          EXPECT_EQ(backing, expected_backing);
          std::vector<uint8_t> actual_lds(kLdsBytes);
          for (uint32_t i = 0; i < kLdsBytes; ++i)
            actual_lds[i] = cu->lds().read8(wf->lds_base() + i);
          EXPECT_EQ(actual_lds, expected_lds);
        }
      }
    }
  }
  ASSERT_TRUE(legacy_vm.unregister_address_space(address_space));
}

TEST(Gfx1250ExecutionTest, TensorDmaCoalescingCopiesAFullBatch) {
  using namespace amdgpu::tensor_dma_detail;
  constexpr uint32_t kProcessId = 81252;
  constexpr uint64_t kGlobal = 0x370000;
  constexpr uint32_t kBytes = 256;
  alignas(4096) std::array<uint8_t, kBytes> backing;
  KfdProcess process(kProcessId);
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(kBytes));
  wf->set_lds_size(kBytes);
  process.map_pages(kGlobal, backing.data(), backing.size());
  amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
  const auto address_space =
      legacy_vm.register_address_space(kProcessId, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(address_space);
  wf->set_address_space(address_space);
  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.tensor_rank = 1;
  desc.elem_size = 4;
  desc.global_base = kGlobal;
  desc.tensor_dims[0] = kBytes / desc.elem_size;
  desc.tile_dims[0] = desc.tensor_dims[0];
  for (bool store : {false, true}) {
    for (uint32_t i = 0; i < kBytes; ++i) {
      backing[i] = static_cast<uint8_t>(i);
      wf->lds().write8(wf->lds_base() + i, static_cast<uint8_t>(i + 1));
    }
    TensorDmaState state(desc, store);
    ASSERT_TRUE(prepare_tensor(state, *wf).succeeded());
    const size_t copied = try_coalesced_transfer(state, *wf);
    ASSERT_EQ(copied, kBytes / desc.elem_size);
    state.next_element += copied;
    ASSERT_EQ(resume_tensor_dma_state(state, *wf), amdgpu::VmAccessOutcome::Complete);
    for (uint32_t i = 0; i < kBytes; ++i) {
      EXPECT_EQ(backing[i], static_cast<uint8_t>(store ? i + 1 : i));
      EXPECT_EQ(wf->lds().read8(wf->lds_base() + i), static_cast<uint8_t>(store ? i + 1 : i));
    }
  }
  ASSERT_TRUE(legacy_vm.unregister_address_space(address_space));
}

class TensorDmaFaultRecorder : public amdgpu::MemoryFaultReporter {
public:
  void report_memory_fault(uint32_t, uint64_t address, amdgpu::MemoryFaultCause) override {
    addresses.push_back(address);
  }
  std::vector<uint64_t> addresses;
};

TEST(Gfx1250ExecutionTest, TensorDmaCoalescingPreservesClippedAndRefusedExtents) {
  using namespace amdgpu::tensor_dma_detail;
  constexpr uint32_t kProcessId = 81251;
  constexpr uint64_t kGlobal = 0x360000;
  constexpr uint32_t kSpan = 64;
  constexpr uint32_t kBytes = 3 * kSpan;
  constexpr size_t kPage = KfdProcess::kPageSize;
  auto *pages = static_cast<uint8_t *>(
      mmap(nullptr, 3 * kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(pages, MAP_FAILED);
  auto unmap = [](uint8_t *ptr) { munmap(ptr, 3 * kPage); };
  const std::unique_ptr<uint8_t, decltype(unmap)> mapping(pages, unmap);
  ASSERT_EQ(mprotect(pages + kPage, kPage, PROT_NONE), 0);
  KfdProcess process(kProcessId);
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(kBytes));
  wf->set_lds_size(kBytes);
  amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
  const auto address_space =
      legacy_vm.register_address_space(kProcessId, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(address_space);
  wf->set_address_space(address_space);
  TensorDmaFaultRecorder reporter;
  ASSERT_TRUE(legacy_vm.set_fault_reporter(address_space, &reporter));
  TensorDmaDescriptor desc;
  desc.count = 1;
  desc.tensor_rank = 1;
  desc.elem_size = 4;
  desc.global_base = kGlobal;
  desc.tensor_dims[0] = kBytes / desc.elem_size;
  desc.tile_dims[0] = desc.tensor_dims[0];
  const TensorDmaLayout layout(desc);
  ASSERT_TRUE(validate_supported_descriptor(desc, layout).succeeded());
  for (bool refused : {false, true}) {
    process.unmap_pages(kGlobal, kPage);
    process.map_pages(kGlobal, pages, kSpan);
    process.map_pages(kGlobal + 2 * kSpan, pages + 2 * kPage, kSpan);
    if (refused)
      process.map_pages(kGlobal + kSpan, pages + kPage, kSpan);
    for (bool store : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "refused=" << refused << " store=" << store);
      auto reset = [&] {
        for (uint32_t i = 0; i < kSpan; ++i) {
          pages[i] = static_cast<uint8_t>(i + 1);
          pages[2 * kPage + i] = static_cast<uint8_t>(i + 71);
        }
        for (uint32_t i = 0; i < kBytes; ++i)
          wf->lds().write8(wf->lds_base() + i, static_cast<uint8_t>(i + 31));
      };
      auto snapshot = [&] {
        std::vector<uint8_t> bytes(pages, pages + kSpan);
        bytes.insert(bytes.end(), pages + 2 * kPage, pages + 2 * kPage + kSpan);
        for (uint32_t i = 0; i < kBytes; ++i)
          bytes.push_back(wf->lds().read8(wf->lds_base() + i));
        return bytes;
      };
      reset();
      TensorDmaState reference(desc, store);
      ASSERT_TRUE(prepare_tensor(reference, *wf).succeeded());
      const auto expected_outcome = resume_tensor_dma_state</*Coalesce=*/false>(reference, *wf);
      const auto expected = snapshot();
      const auto expected_faults = reporter.addresses;
      ASSERT_FALSE(expected_faults.empty());
      EXPECT_EQ(expected_faults.front(), kGlobal + kSpan);
      reporter.addresses.clear();
      reset();
      TensorDmaState batched(desc, store);
      ASSERT_TRUE(prepare_tensor(batched, *wf).succeeded());
      EXPECT_EQ(resume_tensor_dma_state(batched, *wf), expected_outcome);
      EXPECT_EQ(batched.next_element, reference.next_element);
      EXPECT_EQ(snapshot(), expected);
      EXPECT_EQ(reporter.addresses, expected_faults);
      reporter.addresses.clear();
    }
  }
  ASSERT_TRUE(legacy_vm.unregister_address_space(address_space));
}

} // namespace
