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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
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

TEST(Gfx1250ExecutionTest, TensorDmaOverflowSkipsCopyAndBarrier) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
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

TEST(Gfx1250ExecutionTest, TensorDmaObservationReportsCompletedDenseLoadAndStore) {
  Gfx1250Sim sim;
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
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));

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
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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

TEST(Gfx1250ExecutionTest, TensorDmaByteLoadUsesDwordPaddingUnits) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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

TEST(Gfx1250ExecutionTest, TensorDmaZeroCountDisablesLoadStoreAndBarrier) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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

TEST(Gfx1250ExecutionTest, TensorDmaIterateMasksRowsBeyondTensorDimension) {
  // Two iterations cover four rows, but the tensor has only three. The final
  // repeated row must be zero-filled instead of copied from global memory.
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x131000;
  constexpr uint32_t kCols = 2;
  constexpr uint32_t kRows = 3;
  constexpr uint32_t kTileRows = 2;
  constexpr uint32_t kIterations = 2;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16);             // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16);             // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16);             // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, kTileRows);               // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, kCols);                   // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, kCols * kTileRows); // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, kCols * kTileRows); // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, (kIterations - 1) << 16);

  for (uint32_t row = 0; row < kTileRows * kIterations; ++row) {
    for (uint32_t col = 0; col < kCols; ++col)
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * 4,
                       0x45000000u + row * 0x100u + col);
  }

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  for (uint32_t row = 0; row < kTileRows * kIterations; ++row) {
    for (uint32_t col = 0; col < kCols; ++col) {
      const uint32_t actual =
          cu->lds().read32(wf->lds_base() + (row * kCols + col) * sizeof(uint32_t));
      const uint32_t expected = row < kRows ? 0x45000000u + row * 0x100u + col : 0u;
      EXPECT_EQ(actual, expected) << "row " << row << ", col " << col;
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateStoreSkipsRowsBeyondTensorDimension) {
  // Two iterations source four LDS rows, but the tensor has only three. The
  // final repeated row must not overwrite global memory.
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x136000;
  constexpr uint32_t kCols = 2;
  constexpr uint32_t kRows = 3;
  constexpr uint32_t kTileRows = 2;
  constexpr uint32_t kIterations = 2;
  constexpr uint32_t kSentinel = 0xAC1DAC1Du;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16);             // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16);             // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16);             // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, kTileRows);               // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, kCols);                   // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, kCols * kTileRows); // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, kCols * kTileRows); // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, (kIterations - 1) << 16);

  for (uint32_t row = 0; row < kTileRows * kIterations; ++row) {
    for (uint32_t col = 0; col < kCols; ++col) {
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * 4, kSentinel);
      cu->lds().write32(wf->lds_base() + (row * kCols + col) * sizeof(uint32_t),
                        0x4B000000u + row * 0x100u + col);
    }
  }

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c140c00u};
  auto store = decode_gfx1250(store_words, "tensor_store_from_lds");
  ASSERT_NE(store, nullptr);
  store->execute(*store, wf);

  for (uint32_t row = 0; row < kTileRows * kIterations; ++row) {
    for (uint32_t col = 0; col < kCols; ++col) {
      const uint32_t actual =
          read_global_u32(*sim.memory, kGlobal + (row * kCols + col) * sizeof(uint32_t));
      const uint32_t expected = row < kRows ? 0x4B000000u + row * 0x100u + col : kSentinel;
      EXPECT_EQ(actual, expected) << "row " << row << ", col " << col;
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaZeroLengthDimensionMasksEntireTile) {
  // Global memory contains data and LDS starts nonzero. A zero active tensor
  // extent must actively zero-fill the tile rather than copy or do nothing.
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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

TEST(Gfx1250ExecutionTest, TensorDmaNonIteratingRankFiveAvoidsOriginDecomposition) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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

TEST(Gfx1250ExecutionTest, TensorDmaIterateMasksSubrowOriginBeyondTensor) {
  // Three one-element iterations walk a 2x1 tensor. The first two origins are
  // valid dim0 coordinates; the third carries into dim1 and must zero-fill.
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), 0u);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateMasksPlanesBeyondTensorDimension) {
  // The descriptor iterates across two planes of a tensor whose depth is one.
  // The second plane must be zero-filled rather than copied.
  Gfx1250Sim sim;
  auto &observer = install_tensor_dma_observer(sim);
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x134000;
  constexpr uint32_t kSentinel = 0xFACEFACEu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u | (1u << 16));         // tile dim1 and dim2.
  write_wave_sgpr(*cu, *wf, 17, 2u);                      // tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 4u << 16);                // tensor dim1 stride.
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 1u);       // tensor dim2.
  write_wave_sgpr(*cu, *wf, 21, 2u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 4u);       // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  for (uint32_t i = 0; i < 6; ++i)
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x4A000000u + i);
  for (uint32_t i = 0; i < 4; ++i)
    cu->lds().write32(wf->lds_base() + i * sizeof(uint32_t), kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x4A000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x4A000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), 0u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 3 * 4), 0u);
  ASSERT_EQ(observer.accesses.size(), 1u);
  EXPECT_EQ(observer.accesses[0].addresses,
            (std::vector<uint64_t>{kGlobal, kGlobal + sizeof(uint32_t)}));
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateRejectsOverlappingStrides) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  write_tensor_dma_d0(*cu, *wf, 0, 0x135000);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                      // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 1u);                      // Overlaps tensor dim0.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 1u);
  write_wave_sgpr(*cu, *wf, 22, 1u);
  write_wave_sgpr(*cu, *wf, 23, 1u << 16);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(cu->execute_instruction(load.get(), *wf).failed());
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnsupportedOperandValue);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateAllowsAliasedUnitExtent) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x185000;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 1u << 16);                // unit tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                      // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 1u); // dim1 aliases dim0 but has only one coordinate.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 1u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 1u);       // advance along tensor dim0.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  write_global_u32(*sim.memory, kGlobal, 0x55000000u);
  write_global_u32(*sim.memory, kGlobal + sizeof(uint32_t), 0x55000001u);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0x55000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + sizeof(uint32_t)), 0x55000001u);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateAllowsPermutedNonOverlappingStrides) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x183000;
  constexpr uint32_t kSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u | (1u << 16));         // tile dim1 and dim2.
  write_wave_sgpr(*cu, *wf, 17, 4u);                      // tensor dim1 stride.
  write_wave_sgpr(*cu, *wf, 18, 2u << 16);                // tensor dim2 stride.
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 2u);       // tensor dim2.
  write_wave_sgpr(*cu, *wf, 21, 1u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 2u);       // advance along tensor dim2.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  for (uint32_t i = 0; i < 8; ++i)
    write_global_u32(*sim.memory, kGlobal + i * sizeof(uint32_t), 0x52000000u + i);
  cu->lds().write32(wf->lds_base(), kSentinel);
  cu->lds().write32(wf->lds_base() + sizeof(uint32_t), kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0x52000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + sizeof(uint32_t)), 0x52000002u);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateAllowsRepeatedOriginWithOverlappingStrides) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x139000;
  constexpr uint32_t kValue = 0x4D000000u;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                      // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 1u);                      // overlapping tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 1u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 0);        // Global origin does not advance.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  write_global_u32(*sim.memory, kGlobal, kValue);
  cu->lds().write32(wf->lds_base() + 0 * sizeof(uint32_t), 0xDEADDEADu);
  cu->lds().write32(wf->lds_base() + 1 * sizeof(uint32_t), 0xDEADDEADu);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * sizeof(uint32_t)), kValue);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * sizeof(uint32_t)), kValue);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateAllowsPaddedRowPlaneStride) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x13A000;
  constexpr uint32_t kSentinel = 0xDEADDEADu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 1u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u | (1u << 16));         // tile dim1 and dim2.
  write_wave_sgpr(*cu, *wf, 17, 4u);                      // padded tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 6u << 16);                // occupied plane span.
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 2u);       // tensor dim2.
  write_wave_sgpr(*cu, *wf, 21, 1u);       // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 6u);       // Global plane increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  write_global_u32(*sim.memory, kGlobal, 0x4E000000u);
  write_global_u32(*sim.memory, kGlobal + 6 * sizeof(uint32_t), 0x4E000001u);
  cu->lds().write32(wf->lds_base(), kSentinel);
  cu->lds().write32(wf->lds_base() + sizeof(uint32_t), kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  auto load = decode_gfx1250(load_words, "tensor_load_to_lds");
  ASSERT_NE(load, nullptr);
  load->execute(*load, wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base()), 0x4E000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + sizeof(uint32_t)), 0x4E000001u);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateZeroExtentSkipsLayoutValidationAndCompletes) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  wf->set_lds_base(cu->allocate_lds(256));

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

TEST(Gfx1250ExecutionTest, LocalMemPipelineUsesInjectedBarrierDecrementPayload) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint32_t kBarrierLdsAddr = 64;
  cu->write_vgpr(wf->vgpr_alloc().base + 0, 0, kBarrierLdsAddr);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, amdgpu::lds_barrier_cell_init_state(2));

  const std::array<uint32_t, 2> words = {0xd9580000u, 0x00000000u};
  auto *arrive_inst = new cdna5::DsAtomicAsyncBarrierArriveB64Vds(words.data());
  arrive_inst->execute_impl(*wf);

  auto *state = arrive_inst->data_as<amdgpu::VectorMemState>();
  const uint64_t decrement = 2;
  state->store_data.resize(static_cast<size_t>(wf->wf_size()) * sizeof(decrement));
  std::memcpy(state->store_data.data(), &decrement, sizeof(decrement));

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(arrive_inst, *wf);

  const uint64_t expected =
      amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(2), decrement);
  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr), expected);
  EXPECT_EQ(expected, (1ull << 32) |
                          (amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift) |
                          1ull);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250ExecutionTest, LdsBarrierCellHandlesSingleAndBatchedArrivals) {
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

  uint64_t iterated = amdgpu::lds_barrier_cell_init_state(2);
  for (int i = 0; i < 5; ++i)
    iterated = amdgpu::lds_barrier_cell_update_arrive(iterated);
  const uint64_t batched =
      amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(2), 5);
  EXPECT_EQ(batched, iterated);
  EXPECT_EQ(batched, (1ull << 32) | (0x6ull << amdgpu::kLdsBarrierCellPhaseShift));
  EXPECT_FALSE(amdgpu::lds_barrier_cell_phase_parity(batched));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(kLdsBytes));
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
            std::copy(initial.begin(), initial.end(), backing.begin());
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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(kBytes));
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
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  wf->set_lds_base(cu->allocate_lds(kBytes));
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
