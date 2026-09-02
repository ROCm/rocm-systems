// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

class TestMemoryInstruction : public Instruction {
public:
  explicit TestMemoryInstruction(std::unique_ptr<DynamicInstState> state)
      : Instruction("test_mem", nullptr) {
    flags_ |= MEMORY_OP;
    set_data(std::move(state));
  }
};

class DeferredClusterLdsMulticastEngine : public amdgpu::ClusterLdsMulticastEngine {
public:
  amdgpu::ClusterLdsMulticastResult
  submit(amdgpu::ClusterLdsMulticastTransaction submitted,
         amdgpu::ClusterLdsMulticastCompletion submitted_completion) override {
    txn = std::move(submitted);
    completion = std::move(submitted_completion);
    return amdgpu::ClusterLdsMulticastResult::Deferred;
  }

  amdgpu::ClusterLdsMulticastTransaction txn;
  amdgpu::ClusterLdsMulticastCompletion completion;
};

std::string make_single_se_gfx1250_config(uint32_t num_cus) {
  std::string cu_range = "cu[0:" + std::to_string(num_cus) + "]";
  std::string links;
  for (uint32_t i = 0; i < num_cus; ++i) {
    if (i > 0)
      links += ",";
    links += R"({"src":"xcd0.cp.req_)" + std::to_string(i) + R"(","dst":"xcd0.se0.cu)" +
             std::to_string(i) + R"(.cpl","latency":1,"weight":2})";
    links += R"(,{"src":"xcd0.se0.cu)" + std::to_string(i) + R"(.req","dst":"xcd0.l2.cpl_)" +
             std::to_string(i) + R"(","latency":1,"weight":10})";
  }

  return R"({"max_ticks":10000,"num_threads":1,"vm":{"arch":"cdna5"},)"
         R"("topology":{"root":{"name":"soc","type":"soc","children":[)"
         R"({"name":"vram","type":"gpu_memory"},)"
         R"({"name":"xcd0","type":"xcd","children":[)"
         R"({"name":"l2","type":"l2_cache"},)"
         R"({"name":"cp","type":"command_processor"},)"
         R"({"name":"se0","type":"shader_engine","children":[)"
         R"({"name":")" +
         cu_range +
         R"(","type":"compute_unit","config":[)"
         R"({"key":"num_wf_slots","value":")" +
         std::to_string(kGfx1250WaveSlotsPerCu) +
         R"("},)"
         R"({"key":"sgprs_per_wf","value":"128"},)"
         R"({"key":"vgprs_per_wf","value":"1024"},)"
         R"({"key":"lds_size_kb","value":"160"})"
         R"(]}]}]}]},"links":[)" +
         links + R"(]}})";
}

TEST(Gfx1250ExecutionTest, ClusterLoadsDecodeAndPopulateVectorMemState) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);
  wf->set_m0(0xffff0003u);

  constexpr uint64_t kGlobalBase = 0x180000u;
  write_wave_sgpr(*cu, *wf, 0, static_cast<uint32_t>(kGlobalBase));
  write_wave_sgpr(*cu, *wf, 1, static_cast<uint32_t>(kGlobalBase >> 32));

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto write_vgpr_lane_offsets = [&](uint32_t reg, uint32_t stride, uint32_t base_offset = 0) {
    cu->write_vgpr(vgpr_base + reg, 0, base_offset);
    cu->write_vgpr(vgpr_base + reg, 1, base_offset + stride);
  };
  auto write_vgpr_lane_indices = [&](uint32_t reg) {
    cu->write_vgpr(vgpr_base + reg, 0, 0);
    cu->write_vgpr(vgpr_base + reg, 1, 1);
  };

  struct LoadCase {
    std::array<uint32_t, 3> words;
    std::string_view mnemonic;
    uint32_t num_elems;
    uint32_t dst_reg;
    bool scale_offset;
  };

  const LoadCase load_cases[] = {
      {{0xEE19C000u, 0x00000001u, 0x00000000u}, "cluster_load_b32", 1, 1, false},
      {{0xEE1A0000u, 0x00000000u, 0x00000002u}, "cluster_load_b64", 2, 0, false},
      {{0xEE1A4000u, 0x00010002u, 0x00000000u}, "cluster_load_b128", 4, 2, true},
  };

  for (const LoadCase &tc : load_cases) {
    if (tc.scale_offset)
      write_vgpr_lane_indices(0);
    else
      write_vgpr_lane_offsets(tc.words[2] & 0xffu, 16);

    auto inst = decode_gfx1250(tc.words, tc.mnemonic);
    ASSERT_NE(inst, nullptr);
    inst->execute(*inst, wf);
    auto *state = inst->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->tag(), amdgpu::GLOBAL_MEM);
    EXPECT_EQ(state->dst_reg_base, vgpr_base + tc.dst_reg);
    EXPECT_EQ(state->elem_size, 4u);
    EXPECT_EQ(state->num_elems, tc.num_elems);
    EXPECT_TRUE(state->is_load);
    EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::LOADCNT);
    EXPECT_FALSE(state->lds_dst);
    EXPECT_FALSE(state->cluster_multicast);
    EXPECT_TRUE(state->request_force_l1_bypass);
    EXPECT_EQ(state->lane_mask, 0x3u);
    EXPECT_EQ(state->per_lane_addr[0], kGlobalBase);
    EXPECT_EQ(state->per_lane_addr[1], kGlobalBase + 16);
  }

  struct AsyncCase {
    std::array<uint32_t, 3> words;
    std::string_view mnemonic;
    uint32_t elem_size;
    uint32_t num_elems;
    uint32_t lds_addr_reg;
    uint32_t global_stride;
    uint32_t lds_stride;
    uint32_t base_offset;
    int32_t instruction_offset;
    bool scale_offset;
    bool cluster_multicast;
  };

  const AsyncCase async_cases[] = {
      {.words = {0xEE1A8000u, 0x00000000u, 0x00000000u},
       .mnemonic = "cluster_load_async_to_lds_b8",
       .elem_size = 1,
       .num_elems = 1,
       .lds_addr_reg = 0,
       .global_stride = 1,
       .lds_stride = 1,
       .base_offset = 0,
       .instruction_offset = 0,
       .scale_offset = false,
       .cluster_multicast = true},
      {.words = {0xEE180000u, 0x00000000u, 0x00000800u},
       .mnemonic = "global_load_async_to_lds_b32",
       .elem_size = 4,
       .num_elems = 1,
       .lds_addr_reg = 0,
       .global_stride = 16,
       .lds_stride = 16,
       .base_offset = 0,
       .instruction_offset = 8,
       .scale_offset = false,
       .cluster_multicast = false},
      {.words = {0xEE1AC000u, 0x00000000u, 0x00000800u},
       .mnemonic = "cluster_load_async_to_lds_b32",
       .elem_size = 4,
       .num_elems = 1,
       .lds_addr_reg = 0,
       .global_stride = 16,
       .lds_stride = 16,
       .base_offset = 0,
       .instruction_offset = 8,
       .scale_offset = false,
       .cluster_multicast = true},
      {.words = {0xEE1AC000u, 0x00000000u, 0xFFFFC000u},
       .mnemonic = "cluster_load_async_to_lds_b32",
       .elem_size = 4,
       .num_elems = 1,
       .lds_addr_reg = 0,
       .global_stride = 16,
       .lds_stride = 16,
       .base_offset = 128,
       .instruction_offset = -64,
       .scale_offset = false,
       .cluster_multicast = true},
      {.words = {0xEE1B0000u, 0x00000004u, 0x00000004u},
       .mnemonic = "cluster_load_async_to_lds_b64",
       .elem_size = 4,
       .num_elems = 2,
       .lds_addr_reg = 4,
       .global_stride = 16,
       .lds_stride = 16,
       .base_offset = 0,
       .instruction_offset = 0,
       .scale_offset = false,
       .cluster_multicast = true},
      {.words = {0xEE1B4000u, 0x00010001u, 0x00000800u},
       .mnemonic = "cluster_load_async_to_lds_b128",
       .elem_size = 4,
       .num_elems = 4,
       .lds_addr_reg = 1,
       .global_stride = 16,
       .lds_stride = 16,
       .base_offset = 0,
       .instruction_offset = 8,
       .scale_offset = true,
       .cluster_multicast = true},
  };

  for (const AsyncCase &tc : async_cases) {
    if (tc.scale_offset) {
      write_vgpr_lane_indices(0);
      write_vgpr_lane_offsets(tc.lds_addr_reg, 16, tc.base_offset);
    } else {
      write_vgpr_lane_offsets(tc.lds_addr_reg, tc.global_stride, tc.base_offset);
    }

    auto inst = decode_gfx1250(tc.words, tc.mnemonic);
    ASSERT_NE(inst, nullptr);
    inst->execute(*inst, wf);
    auto *state = inst->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->tag(), amdgpu::GLOBAL_MEM);
    EXPECT_EQ(state->elem_size, tc.elem_size);
    EXPECT_EQ(state->num_elems, tc.num_elems);
    EXPECT_TRUE(state->is_load);
    EXPECT_TRUE(state->lds_dst);
    EXPECT_TRUE(state->lds_per_lane_addr);
    EXPECT_EQ(state->lds_base, wf->lds_base());
    EXPECT_EQ(state->cluster_multicast, tc.cluster_multicast);
    EXPECT_EQ(state->cluster_mcast_mask, tc.cluster_multicast ? 0x3u : 0u);
    EXPECT_EQ(state->request_force_l1_bypass, tc.cluster_multicast);
    EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
    EXPECT_EQ(state->lane_mask, 0x3u);
    uint64_t expected_global_base = static_cast<uint64_t>(
        static_cast<int64_t>(kGlobalBase + tc.base_offset) + tc.instruction_offset);
    uint32_t expected_lds_offset = tc.base_offset + static_cast<uint32_t>(tc.instruction_offset);
    EXPECT_EQ(state->per_lane_addr[0], expected_global_base);
    EXPECT_EQ(state->per_lane_addr[1], expected_global_base + tc.global_stride);
    EXPECT_EQ(state->per_lane_lds_addr[0], wf->lds_base() + expected_lds_offset);
    EXPECT_EQ(state->per_lane_lds_addr[1], wf->lds_base() + expected_lds_offset + tc.lds_stride);
  }
}

TEST(Gfx1250ExecutionTest, GlobalStoreAsyncFromLdsAppliesIoffsetToGlobalAndLdsSource) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobalBase = 0x190000u;
  constexpr uint32_t kLane0Value = 0x12345678u;
  constexpr uint32_t kLane1Value = 0xabcdef01u;
  constexpr uint32_t kVgprOffset = 128;
  constexpr int32_t kIoffset = -64;
  constexpr uint32_t kEffectiveOffset =
      static_cast<uint32_t>(static_cast<int32_t>(kVgprOffset) + kIoffset);
  write_wave_sgpr(*cu, *wf, 0, static_cast<uint32_t>(kGlobalBase));
  write_wave_sgpr(*cu, *wf, 1, static_cast<uint32_t>(kGlobalBase >> 32));

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base, 0, kVgprOffset);
  cu->write_vgpr(vgpr_base, 1, kVgprOffset + 16);
  cu->lds().write32(wf->lds_base() + kVgprOffset, ~kLane0Value);
  cu->lds().write32(wf->lds_base() + kVgprOffset + 16, ~kLane1Value);
  cu->lds().write32(wf->lds_base() + kEffectiveOffset, kLane0Value);
  cu->lds().write32(wf->lds_base() + kEffectiveOffset + 16, kLane1Value);

  auto inst =
      decode_gfx1250({0xEE190000u, 0x00000000u, 0xFFFFC000u}, "global_store_async_from_lds_b32");
  ASSERT_NE(inst, nullptr);
  inst->execute(*inst, wf);
  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->tag(), amdgpu::GLOBAL_MEM);
  EXPECT_FALSE(state->is_load);
  EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
  EXPECT_EQ(state->lane_mask, 0x3u);
  EXPECT_EQ(state->per_lane_addr[0], kGlobalBase + kEffectiveOffset);
  EXPECT_EQ(state->per_lane_addr[1], kGlobalBase + kEffectiveOffset + 16);
  ASSERT_GE(state->store_data.size(), 8u);

  uint32_t lane0_value = 0;
  uint32_t lane1_value = 0;
  std::memcpy(&lane0_value, &state->store_data[0], sizeof(lane0_value));
  std::memcpy(&lane1_value, &state->store_data[4], sizeof(lane1_value));
  EXPECT_EQ(lane0_value, kLane0Value);
  EXPECT_EQ(lane1_value, kLane1Value);
}

TEST(Gfx1250ExecutionTest, GlobalStoreAsyncFromOutOfRangeLdsSourceProducesZeroPayload) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1u);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobalBase = 0x1a0000u;
  write_wave_sgpr(*cu, *wf, 0, static_cast<uint32_t>(kGlobalBase));
  write_wave_sgpr(*cu, *wf, 1, static_cast<uint32_t>(kGlobalBase >> 32));
  cu->write_vgpr(wf->vgpr_alloc().base, 0, 256);

  auto inst =
      decode_gfx1250({0xEE190000u, 0x00000000u, 0x00000000u}, "global_store_async_from_lds_b32");
  ASSERT_NE(inst, nullptr);
  inst->execute(*inst, wf);
  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->lane_mask, 0x1u);
  ASSERT_GE(state->store_data.size(), sizeof(uint32_t));
  uint32_t value = UINT32_MAX;
  std::memcpy(&value, state->store_data.data(), sizeof(value));
  EXPECT_EQ(value, 0u);
}

TEST(Gfx1250ExecutionTest, AsyncLdsAddressRejectsAccessOutsideAllocation) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  cdna5::VglobalMachineInst inst{};
  inst.ioffset = 0xfffffcu;
  EXPECT_EQ(cdna5::signed_ioffset(inst.ioffset), -4);
  EXPECT_EQ(cdna5::async_lds_lane_address(inst, *wf, 0, 4), amdgpu::kInvalidLdsAddress);
  EXPECT_EQ(cdna5::async_lds_lane_address(inst, *wf, 8, 4), wf->lds_base() + 4);
  EXPECT_EQ(cdna5::async_lds_lane_address(inst, *wf, 256, 4), wf->lds_base() + 252);
  EXPECT_EQ(cdna5::async_lds_lane_address(inst, *wf, 256, 16), amdgpu::kInvalidLdsAddress);
  EXPECT_EQ(cdna5::async_lds_lane_address(inst, *wf, 260, 4), amdgpu::kInvalidLdsAddress);

  inst.ioffset = 64;
  EXPECT_EQ(cdna5::async_lds_lane_address(inst, *wf, UINT32_MAX - 63, 4), wf->lds_base());
}

TEST(Gfx1250ExecutionTest, DecodedAsyncLdsLoadsHonorEveryAccessWidthAtAllocationEnd) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1u);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_lds_size(256);

  constexpr uint64_t kGlobalBase = 0x1a0000u;
  write_wave_sgpr(*cu, *wf, 0, static_cast<uint32_t>(kGlobalBase));
  write_wave_sgpr(*cu, *wf, 1, static_cast<uint32_t>(kGlobalBase >> 32));
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base, 0, 0);

  struct TestCase {
    uint16_t opcode;
    std::string_view mnemonic;
    uint32_t access_size;
  };
  constexpr TestCase cases[] = {
      {cdna5::kGlobalLoadAsyncToLdsB8Vglobal, "global_load_async_to_lds_b8", 1},
      {cdna5::kGlobalLoadAsyncToLdsB32Vglobal, "global_load_async_to_lds_b32", 4},
      {cdna5::kGlobalLoadAsyncToLdsB64Vglobal, "global_load_async_to_lds_b64", 8},
      {cdna5::kGlobalLoadAsyncToLdsB128Vglobal, "global_load_async_to_lds_b128", 16},
  };

  for (const auto &tc : cases) {
    SCOPED_TRACE(tc.mnemonic);
    const auto words =
        cdna5::build_vglobal(tc.opcode, {.saddr = 0, .vdst = 1, .vaddr = 0, .ioffset = 0});
    auto execute_at = [&](uint32_t relative_addr) {
      cu->write_vgpr(vgpr_base + 1, 0, relative_addr);
      auto inst = decode_gfx1250(words, tc.mnemonic);
      EXPECT_NE(inst, nullptr);
      if (!inst)
        return amdgpu::kInvalidLdsAddress;
      inst->execute(*inst, wf);
      auto *state = inst->data_as<amdgpu::VectorMemState>();
      EXPECT_NE(state, nullptr);
      if (!state)
        return amdgpu::kInvalidLdsAddress;
      EXPECT_EQ(state->elem_size * state->num_elems, tc.access_size);
      return state->per_lane_lds_addr[0];
    };

    const uint32_t last_valid = wf->lds_size() - tc.access_size;
    EXPECT_EQ(execute_at(last_valid), wf->lds_base() + last_valid);
    EXPECT_EQ(execute_at(last_valid + 1), amdgpu::kInvalidLdsAddress);
  }
}

TEST(Gfx1250SimulationTest, GlobalLoadAsyncToLdsWrapsBeforeAddingAllocationBase) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2a00;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t expected = 0x12345678u;
  constexpr auto move_wrapping_lds =
      cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 255, .vdst = 1});
  constexpr auto async_load = cdna5::build_vglobal(
      cdna5::kGlobalLoadAsyncToLdsB32Vglobal, {.saddr = 0, .vdst = 1, .vaddr = 0, .ioffset = 64});
  const uint32_t code[] = {
      0xBE8000FFu,
      static_cast<uint32_t>(input_addr - 64), // s_mov_b32 s0, input_addr - 64
      0xBE810080u,                            // s_mov_b32 s1, 0
      0x7E000280u,                            // v_mov_b32_e32 v0, 0
      move_wrapping_lds[0],
      UINT32_MAX - 63, // v_mov_b32_e32 v1, 0xffffffc0
      async_load[0],
      async_load[1],
      async_load[2],
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  sim.memory->write32(input_addr, expected);

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 1;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 32;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.group_segment_size = lds_bytes_per_wg;
  pkt.kernel_object = kernel_object;
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  ASSERT_TRUE(sim.engine->step());

  auto *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(sim.cu()->allocate_lds(lds_bytes_per_wg));
  ASSERT_NE(wf->lds_base(), 0u);
  const uint32_t lds_base = wf->lds_base();
  sim.cu()->lds().write32(lds_base, 0xa5a5a5a5u);
  EXPECT_NO_THROW(sim.engine->run());
  EXPECT_EQ(sim.cu()->lds().read32(lds_base), expected);
}

TEST(Gfx1250SimulationTest, GlobalStoreAsyncFromLdsWrapsBeforeAddingAllocationBase) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t output_addr = 0x2c00;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t expected = 0x89abcdefu;
  constexpr auto move_wrapping_lds =
      cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 255, .vdst = 1});
  constexpr auto async_store =
      cdna5::build_vglobal(cdna5::kGlobalStoreAsyncFromLdsB32Vglobal,
                           {.saddr = 0, .vsrc = 1, .vaddr = 0, .ioffset = 64});
  const uint32_t code[] = {
      0xBE8000FFu,
      static_cast<uint32_t>(output_addr - 64), // s_mov_b32 s0, output_addr - 64
      0xBE810080u,                             // s_mov_b32 s1, 0
      0x7E000280u,                             // v_mov_b32_e32 v0, 0
      move_wrapping_lds[0],
      UINT32_MAX - 63, // v_mov_b32_e32 v1, 0xffffffc0
      async_store[0],
      async_store[1],
      async_store[2],
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  sim.memory->write32(output_addr, 0u);

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 1;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 32;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.group_segment_size = lds_bytes_per_wg;
  pkt.kernel_object = kernel_object;
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  ASSERT_TRUE(sim.engine->step());

  auto *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(sim.cu()->allocate_lds(lds_bytes_per_wg));
  ASSERT_NE(wf->lds_base(), 0u);
  const uint32_t lds_base = wf->lds_base();
  sim.cu()->lds().write32(lds_base, expected);
  EXPECT_NO_THROW(sim.engine->run());
  EXPECT_EQ(sim.memory->read32(output_addr), expected);
}

TEST(Gfx1250ExecutionTest, DispatchEntryClusterMathCoversMultiDimensionalShapes) {
  auto expect_rank_period = [](const amdgpu::DispatchEntry &candidate, uint64_t expected_period) {
    ASSERT_TRUE(candidate.cluster_grid_is_complete());
    EXPECT_EQ(candidate.cluster_rank_period(), expected_period);
    const uint32_t workgroup_count =
        candidate.grid_wgs_x * candidate.grid_wgs_y * candidate.grid_wgs_z;
    for (uint32_t flat_wg_id = 0; flat_wg_id < workgroup_count; ++flat_wg_id) {
      EXPECT_EQ(candidate.cluster_rank_for_flat_wg_id(flat_wg_id),
                candidate.cluster_rank_for_flat_wg_id(
                    flat_wg_id + static_cast<uint32_t>(candidate.cluster_rank_period())));
    }
  };

  amdgpu::DispatchEntry entry{};
  entry.grid_wgs_x = 4;
  entry.grid_wgs_y = 4;
  entry.grid_wgs_z = 4;
  entry.cluster_count_x = 2;
  entry.cluster_count_y = 2;
  entry.cluster_count_z = 2;
  entry.cluster_size_x = 2;
  entry.cluster_size_y = 2;
  entry.cluster_size_z = 2;

  EXPECT_TRUE(entry.cluster_grid_is_complete());
  EXPECT_EQ(entry.cluster_size(), 8u);
  expect_rank_period(entry, 32u);
  EXPECT_EQ(entry.cluster_rank_for_flat_wg_id(0), 0u);
  EXPECT_EQ(entry.cluster_rank_for_flat_wg_id(5), 3u);
  EXPECT_EQ(entry.cluster_rank_for_flat_wg_id(21), 7u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 0), 0u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 1), 1u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 2), 4u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 3), 5u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 4), 16u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 7), 21u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(42, 0), 42u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(42, 7), 63u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(0), 0u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(1), 2u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(2), 8u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(4), 32u);

  amdgpu::DispatchEntry one_dimensional{};
  one_dimensional.grid_wgs_x = 8;
  one_dimensional.grid_wgs_y = 1;
  one_dimensional.grid_wgs_z = 1;
  one_dimensional.cluster_count_x = 2;
  one_dimensional.cluster_count_y = 1;
  one_dimensional.cluster_count_z = 1;
  one_dimensional.cluster_size_x = 4;
  one_dimensional.cluster_size_y = 1;
  one_dimensional.cluster_size_z = 1;
  expect_rank_period(one_dimensional, 4u);

  amdgpu::DispatchEntry two_dimensional{};
  two_dimensional.grid_wgs_x = 8;
  two_dimensional.grid_wgs_y = 6;
  two_dimensional.grid_wgs_z = 1;
  two_dimensional.cluster_count_x = 2;
  two_dimensional.cluster_count_y = 2;
  two_dimensional.cluster_count_z = 1;
  two_dimensional.cluster_size_x = 4;
  two_dimensional.cluster_size_y = 3;
  two_dimensional.cluster_size_z = 1;
  expect_rank_period(two_dimensional, 24u);

  entry.grid_wgs_x = 3;
  entry.grid_wgs_y = 2;
  entry.grid_wgs_z = 1;
  entry.cluster_count_x = 2;
  entry.cluster_count_y = 2;
  entry.cluster_count_z = 1;
  entry.cluster_size_x = 2;
  entry.cluster_size_y = 1;
  entry.cluster_size_z = 1;
  EXPECT_FALSE(entry.cluster_grid_is_complete());
}

TEST(Gfx1250ExecutionTest, ClusterLdsMulticastTransactionCapturesRemapState) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(9, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(7);
  wf->set_cluster_info(/*rank=*/1, /*size=*/4);
  wf->set_lds_base(0x100);

  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.elem_size = 4;
  state.num_elems = 2;
  state.wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state.lds_base = wf->lds_base();
  state.lds_per_lane_addr = true;
  state.cluster_multicast = true;
  state.cluster_mcast_mask = 0xa;
  state.wf_size = 32;
  state.lane_mask = 0x3;
  state.per_lane_addr[0] = 0x8000;
  state.per_lane_addr[1] = 0x8020;
  state.per_lane_lds_addr[0] = state.lds_base + 0x10;
  state.per_lane_lds_addr[1] = state.lds_base + 0x24;
  state.response_data.resize(state.wf_size * state.num_elems * state.elem_size);

  std::vector<amdgpu::ClusterLdsTarget> targets = {{cu, /*wg_id=*/11, /*lds_base=*/0x400,
                                                    /*cluster_rank=*/3}};
  auto txn = amdgpu::make_cluster_lds_multicast_transaction(state, *wf, std::move(targets));

  EXPECT_EQ(txn.dispatch_id, 7u);
  EXPECT_EQ(txn.source_wg_id, 9u);
  EXPECT_EQ(txn.source_cluster_rank, 1u);
  EXPECT_EQ(txn.source_lds_base, 0x100u);
  EXPECT_EQ(txn.mcast_mask, 0xau);
  EXPECT_EQ(txn.wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
  EXPECT_EQ(txn.bytes_per_lane, 8u);
  // Retained for deferred/timing backends that model global request coalescing.
  EXPECT_EQ(txn.per_lane_global_addr[0], 0x8000u);
  EXPECT_EQ(txn.per_lane_global_addr[1], 0x8020u);
  ASSERT_EQ(txn.targets.size(), 1u);
  EXPECT_EQ(txn.targets[0].wg_id, 11u);
  EXPECT_EQ(amdgpu::cluster_lds_lane_addr(txn, 0, txn.targets[0].lds_base), 0x410u);
  EXPECT_EQ(amdgpu::cluster_lds_lane_addr(txn, 1, txn.targets[0].lds_base), 0x424u);
  EXPECT_THROW((void)amdgpu::remap_cluster_lds_addr(0x100, 0x400, 0xfc), std::runtime_error);
}

TEST(Gfx1250ExecutionTest, ClusterLdsSourceRankSelectionCoversDefaultAndMasks) {
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_cluster_rank = 2;

  txn.mcast_mask = 0;
  EXPECT_TRUE(amdgpu::cluster_lds_source_rank_selected(txn));

  txn.mcast_mask = amdgpu::cluster_multicast_rank_mask(2);
  EXPECT_TRUE(amdgpu::cluster_lds_source_rank_selected(txn));

  txn.mcast_mask = amdgpu::cluster_multicast_rank_mask(1);
  EXPECT_FALSE(amdgpu::cluster_lds_source_rank_selected(txn));
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineWritesOnlyIssuingParticipant) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  cu->clear_lds();

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = 1;
  txn.source_cluster_rank = 1;
  txn.source_lds_base = 0x300;
  txn.mcast_mask = 0x3;
  txn.bytes_per_lane = 4;
  txn.wf_size = 4;
  txn.lane_mask = 0x5;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = 0x304;
  txn.per_lane_lds_addr[2] = 0x30c;
  txn.payload.resize(txn.wf_size * txn.bytes_per_lane);
  const uint32_t lane0 = 0x11223344;
  const uint32_t lane2 = 0xaabbccdd;
  std::memcpy(&txn.payload[0 * txn.bytes_per_lane], &lane0, sizeof(lane0));
  std::memcpy(&txn.payload[2 * txn.bytes_per_lane], &lane2, sizeof(lane2));
  txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x200, /*cluster_rank=*/0},
                 {cu, /*wg_id=*/1, /*lds_base=*/0x300, /*cluster_rank=*/1}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  bool deferred_callback_called = false;
  EXPECT_EQ(engine.submit(std::move(txn), [&]() { deferred_callback_called = true; }),
            amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_FALSE(deferred_callback_called);
  EXPECT_EQ(cu->lds().read32(0x204), 0u);
  EXPECT_EQ(cu->lds().read32(0x20c), 0u);
  EXPECT_EQ(cu->lds().read32(0x304), lane0);
  EXPECT_EQ(cu->lds().read32(0x30c), lane2);
  EXPECT_EQ(cu->lds().read32(0x208), 0u);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineSkipsUnissuedSelectedPeer) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  cu->clear_lds();

  constexpr uint32_t kValue = 0x55667788;
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = 1;
  txn.source_cluster_rank = 1;
  txn.source_lds_base = 0x300;
  txn.mcast_mask = 0x1; // Selects rank 0, not the issuing rank 1.
  txn.bytes_per_lane = sizeof(kValue);
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = 0x310;
  txn.payload.resize(sizeof(kValue));
  std::memcpy(txn.payload.data(), &kValue, sizeof(kValue));
  txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x200, /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(std::move(txn), []() {}), amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(cu->lds().read32(0x210), 0u);
  EXPECT_EQ(cu->lds().read32(0x310), 0u);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineUsesRecipientOwnedDestinations) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  cu->clear_lds();

  constexpr uint32_t kWg0Value = 0x11112222;
  constexpr uint32_t kWg1Value = 0x33334444;

  auto make_txn = [&](uint32_t wg_id, uint32_t rank, uint32_t lds_base, uint32_t lds_offset,
                      uint32_t value) {
    amdgpu::ClusterLdsMulticastTransaction txn{};
    txn.source_wg_id = wg_id;
    txn.source_cluster_rank = rank;
    txn.source_lds_base = lds_base;
    txn.mcast_mask = 0x3;
    txn.bytes_per_lane = sizeof(value);
    txn.wf_size = 1;
    txn.lane_mask = 0x1;
    txn.per_lane_addr = true;
    txn.per_lane_lds_addr[0] = lds_base + lds_offset;
    txn.payload.resize(sizeof(value));
    std::memcpy(txn.payload.data(), &value, sizeof(value));
    txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x100, /*cluster_rank=*/0},
                   {cu, /*wg_id=*/1, /*lds_base=*/0x200, /*cluster_rank=*/1}};
    return txn;
  };

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(make_txn(/*wg_id=*/0, /*rank=*/0, /*lds_base=*/0x100,
                                   /*lds_offset=*/0x10, kWg0Value),
                          []() {}),
            amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(engine.submit(make_txn(/*wg_id=*/1, /*rank=*/1, /*lds_base=*/0x200,
                                   /*lds_offset=*/0x30, kWg1Value),
                          []() {}),
            amdgpu::ClusterLdsMulticastResult::Complete);

  EXPECT_EQ(cu->lds().read32(0x110), kWg0Value);
  EXPECT_EQ(cu->lds().read32(0x230), kWg1Value);
  EXPECT_EQ(cu->lds().read32(0x210), 0u);
  EXPECT_EQ(cu->lds().read32(0x130), 0u);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineRejectsUndersizedPayload) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.bytes_per_lane = 4;
  txn.wf_size = 2;
  txn.lane_mask = 0x3;
  txn.payload.resize(4);
  txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x200, /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_THROW((void)engine.submit(std::move(txn), []() {}), std::runtime_error);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineDropsOutOfRangeTarget) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_lds_base = 0;
  txn.bytes_per_lane = 4;
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = 0;
  txn.payload.resize(4);
  txn.targets = {{cu, /*wg_id=*/0, static_cast<uint32_t>(cu->lds().size_bytes()) - 2,
                  /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(std::move(txn), []() {}), amdgpu::ClusterLdsMulticastResult::Complete);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineRejectsStridedOutOfRangeTarget) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_lds_base = 0;
  txn.bytes_per_lane = 4;
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.payload.resize(4);
  txn.targets = {{cu, /*wg_id=*/0, static_cast<uint32_t>(cu->lds().size_bytes()) - 2,
                  /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_THROW((void)engine.submit(std::move(txn), []() {}), std::runtime_error);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineDropsWidenedRemapOverflow) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  constexpr uint32_t kSentinel = 0xa5a5a5a5u;
  cu->lds().write32(0, kSentinel);

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_lds_base = 0x100;
  txn.bytes_per_lane = 4;
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = 0xffffff00u;
  txn.payload.resize(4, 0x5a);
  txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x200, /*cluster_rank=*/0}};

  EXPECT_EQ(amdgpu::cluster_lds_lane_addr(txn, 0, txn.targets[0].lds_base), 0x100000000ULL);
  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(std::move(txn), []() {}), amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(cu->lds().read32(0), kSentinel);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineDropsInvalidSignedOffsetAddress) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  cu->clear_lds();

  constexpr uint32_t kLdsBase = 0x200;
  constexpr uint32_t kValue = 0x12345678;
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_lds_base = kLdsBase;
  txn.bytes_per_lane = sizeof(kValue);
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = amdgpu::kInvalidLdsAddress;
  txn.payload.resize(sizeof(kValue));
  std::memcpy(txn.payload.data(), &kValue, sizeof(kValue));
  txn.targets = {{cu, /*wg_id=*/0, kLdsBase, /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(std::move(txn), []() {}), amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(cu->lds().read32(kLdsBase), 0u);
}

TEST(Gfx1250ExecutionTest, ClusterLdsPinPreventsAllocatorReuseUntilClusterCompletes) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();

  EXPECT_EQ(cu->allocate_lds(257), 0u);
  cu->pin_lds_until_cluster_retired(7);
  // A cluster pin blocks LDS reclamation even when the CU has no resident waves.
  cu->maybe_reset_lds_alloc();

  EXPECT_EQ(cu->allocate_lds(257), 512u);
  cu->unpin_lds_for_cluster(7);
  // Once the pin is released and the CU is idle, the LDS allocator resets.
  cu->maybe_reset_lds_alloc();
  EXPECT_EQ(cu->allocate_lds(257), 0u);
}

TEST(Gfx1250ExecutionTest, OrdinaryLdsDstLoadWritesDirectlyAndCompletesAsyncCounter) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(3);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddr = 0x9000;
  constexpr uint32_t kLoadedValue = 0x12345678;
  for (uint32_t byte = 0; byte < sizeof(kLoadedValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte,
                       static_cast<uint8_t>((kLoadedValue >> (byte * 8)) & 0xffu));
  }

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state->lds_dst = true;
  state->lds_per_lane_addr = true;
  state->lds_base = wf->lds_base();
  state->wf_size = 32;
  state->lane_mask = 0x1;
  state->exec_mask = 0x1;
  state->per_lane_addr[0] = kGlobalAddr;
  state->per_lane_lds_addr[0] = wf->lds_base() + 0x20;

  DeferredClusterLdsMulticastEngine deferred_engine;
  cu->set_cluster_lds_multicast_engine(&deferred_engine);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(wf->wait_counters().asynccnt, 0u);
  EXPECT_FALSE(static_cast<bool>(deferred_engine.completion));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0x20), kLoadedValue);
  cu->set_cluster_lds_multicast_engine(nullptr);
}

TEST(Gfx1250ExecutionTest, NonClusterClusterLdsLoadDowngradesToOrdinaryAsyncToLds) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(5);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddr = 0x9080;
  constexpr uint32_t kLoadedValue = 0x78563412;
  for (uint32_t byte = 0; byte < sizeof(kLoadedValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte,
                       static_cast<uint8_t>((kLoadedValue >> (byte * 8)) & 0xffu));
  }

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state->lds_dst = true;
  state->lds_per_lane_addr = true;
  state->lds_base = wf->lds_base();
  state->wf_size = 32;
  state->lane_mask = 0x1;
  state->exec_mask = 0x1;
  state->cluster_multicast = true;
  state->cluster_mcast_mask = 0x2; // Excludes rank 0, but non-clustered loads downgrade.
  state->per_lane_addr[0] = kGlobalAddr;
  state->per_lane_lds_addr[0] = wf->lds_base() + 0x24;

  DeferredClusterLdsMulticastEngine deferred_engine;
  cu->set_cluster_lds_multicast_engine(&deferred_engine);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(wf->wait_counters().asynccnt, 0u);
  EXPECT_FALSE(static_cast<bool>(deferred_engine.completion));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0x24), kLoadedValue);
  cu->set_cluster_lds_multicast_engine(nullptr);
}

TEST(Gfx1250ExecutionTest, ClusterLoadRequestBypassesStaleL1VectorLine) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(6);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddr = 0xa000;
  constexpr uint32_t kOldValue = 0x11111111;
  constexpr uint32_t kNewValue = 0x22222222;
  for (uint32_t byte = 0; byte < sizeof(kOldValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte, static_cast<uint8_t>((kOldValue >> (byte * 8)) & 0xffu));
  }

  uint64_t addrs[64] = {};
  addrs[0] = kGlobalAddr;
  uint8_t l1_fill[64 * sizeof(kOldValue)] = {};
  cu->l1_vector().load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, l1_fill,
                       amdgpu::Mtype::RW, /*non_temporal=*/false,
                       /*request_l1_bypass=*/false, /*vmid=*/0);
  uint32_t filled_value = 0;
  std::memcpy(&filled_value, l1_fill, sizeof(filled_value));
  ASSERT_EQ(filled_value, kOldValue);

  cu->l2()->write(kGlobalAddr, reinterpret_cast<const uint8_t *>(&kNewValue), sizeof(kNewValue),
                  amdgpu::Mtype::RW, /*vmid=*/0);

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());

  auto ordinary = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  ordinary->elem_size = 4;
  ordinary->num_elems = 1;
  ordinary->is_load = true;
  ordinary->wait_counter_type = amdgpu::WaitCounterType::LOADCNT;
  ordinary->wf_size = 32;
  ordinary->lane_mask = 0x1;
  ordinary->exec_mask = 0x1;
  ordinary->dst_reg_base = wf->vgpr_alloc().base;
  ordinary->per_lane_addr[0] = kGlobalAddr;
  pipeline.issue(new TestMemoryInstruction(std::move(ordinary)), *wf);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), kOldValue);

  auto cluster = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  cluster->elem_size = 4;
  cluster->num_elems = 1;
  cluster->is_load = true;
  cluster->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  cluster->request_force_l1_bypass = true;
  cluster->lds_dst = true;
  cluster->lds_per_lane_addr = true;
  cluster->lds_base = wf->lds_base();
  cluster->wf_size = 32;
  cluster->lane_mask = 0x1;
  cluster->exec_mask = 0x1;
  cluster->cluster_multicast = true;
  cluster->cluster_mcast_mask = 0x1;
  cluster->per_lane_addr[0] = kGlobalAddr;
  cluster->per_lane_lds_addr[0] = wf->lds_base() + 0x40;
  pipeline.issue(new TestMemoryInstruction(std::move(cluster)), *wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0x40), kNewValue);
}

TEST(Gfx1250ExecutionTest, ClusterLdsFallbackSkipsSelfWhenMaskExcludesSource) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/3, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(17);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_cluster_info(/*rank=*/1, /*size=*/2);

  constexpr uint64_t kGlobalAddr = 0x9100;
  constexpr uint32_t kLoadedValue = 0xabcdef01;
  for (uint32_t byte = 0; byte < sizeof(kLoadedValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte,
                       static_cast<uint8_t>((kLoadedValue >> (byte * 8)) & 0xffu));
  }

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state->lds_dst = true;
  state->lds_per_lane_addr = true;
  state->lds_base = wf->lds_base();
  state->wf_size = 32;
  state->lane_mask = 0x1;
  state->exec_mask = 0x1;
  state->cluster_multicast = true;
  state->cluster_mcast_mask = 0x1; // Selects rank 0, not the source rank 1.
  state->per_lane_addr[0] = kGlobalAddr;
  state->per_lane_lds_addr[0] = wf->lds_base() + 0x20;

  DeferredClusterLdsMulticastEngine deferred_engine;
  cu->set_cluster_lds_multicast_engine(&deferred_engine);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(wf->wait_counters().asynccnt, 1u);
  ASSERT_TRUE(static_cast<bool>(deferred_engine.completion));
  EXPECT_TRUE(deferred_engine.txn.targets.empty());

  deferred_engine.completion();
  EXPECT_EQ(wf->wait_counters().asynccnt, 0u);
  cu->set_cluster_lds_multicast_engine(nullptr);
}

TEST(Gfx1250SimulationTest, ClusterLdsTargetsCoversMasksAndLifetime) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint32_t lds_bytes_per_wg = 256;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);

  ASSERT_TRUE(sim.engine->step());

  auto all = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x3);
  ASSERT_EQ(all.size(), 2u);
  EXPECT_EQ(all[0].wg_id, 0u);
  EXPECT_EQ(all[1].wg_id, 1u);

  auto self = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x1);
  ASSERT_EQ(self.size(), 1u);
  EXPECT_EQ(self[0].wg_id, 0u);

  auto peer = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x2);
  ASSERT_EQ(peer.size(), 1u);
  EXPECT_EQ(peer[0].wg_id, 1u);

  auto peer_from_rank1 =
      sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/1, /*mcast_mask=*/0x1);
  ASSERT_EQ(peer_from_rank1.size(), 1u);
  EXPECT_EQ(peer_from_rank1[0].wg_id, 0u);

  auto zero_mask = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0);
  ASSERT_EQ(zero_mask.size(), 1u);
  EXPECT_EQ(zero_mask[0].wg_id, 0u);

  auto out_of_range =
      sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x4);
  EXPECT_TRUE(out_of_range.empty());

  auto mixed_range =
      sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x5);
  ASSERT_EQ(mixed_range.size(), 1u);
  EXPECT_EQ(mixed_range[0].wg_id, 0u);

  EXPECT_TRUE(sim.cp()
                  ->cluster_lds_targets(/*dispatch_id=*/999, /*wg_id=*/0,
                                        /*mcast_mask=*/0x3)
                  .empty());

  sim.engine->run();
  EXPECT_TRUE(sim.cp()
                  ->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0,
                                        /*mcast_mask=*/0x3)
                  .empty());
}

TEST(Gfx1250SimulationTest, ClusterLdsTargetsUseMultiDimensionalClusterPlacement) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint32_t lds_bytes_per_wg = 256;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);

  amdgpu::AmdExtKernelDispatchPacket pkt{};
  pkt.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  pkt.amd_format = amdgpu::kHsaAmdPacketTypeExtKernelDispatch;
  pkt.setup = 2;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.cluster_count_x = 2;
  pkt.cluster_count_y = 2;
  pkt.cluster_count_z = 1;
  pkt.cluster_size_x = 2;
  pkt.cluster_size_y = 2;
  pkt.cluster_size_z = 1;
  pkt.group_segment_size = lds_bytes_per_wg;
  pkt.kernel_object = kernel_object;

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  ASSERT_TRUE(sim.engine->step());

  auto cluster0 = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0xf);
  ASSERT_EQ(cluster0.size(), 4u);
  EXPECT_EQ(cluster0[0].wg_id, 0u);
  EXPECT_EQ(cluster0[1].wg_id, 1u);
  EXPECT_EQ(cluster0[2].wg_id, 4u);
  EXPECT_EQ(cluster0[3].wg_id, 5u);

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/5, /*mcast_mask=*/0x5);
  ASSERT_EQ(targets.size(), 2u);
  EXPECT_EQ(targets[0].wg_id, 0u);
  EXPECT_EQ(targets[1].wg_id, 4u);

  auto source = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/5, /*mcast_mask=*/0x8);
  ASSERT_EQ(source.size(), 1u);
  EXPECT_EQ(source[0].wg_id, 5u);

  constexpr uint32_t kValue = 0xfeed1234;
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = source[0].wg_id;
  txn.source_cluster_rank = source[0].cluster_rank;
  txn.source_lds_base = source[0].lds_base;
  txn.mcast_mask = 0x8;
  txn.bytes_per_lane = sizeof(kValue);
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = source[0].lds_base + 0x10;
  txn.payload.resize(sizeof(kValue));
  std::memcpy(txn.payload.data(), &kValue, sizeof(kValue));
  txn.targets = source;

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  bool deferred_callback_called = false;
  EXPECT_EQ(engine.submit(std::move(txn), [&]() { deferred_callback_called = true; }),
            amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_FALSE(deferred_callback_called);
  EXPECT_EQ(targets[0].cu->lds().read32(targets[0].lds_base + 0x10), 0u);
  EXPECT_EQ(targets[1].cu->lds().read32(targets[1].lds_base + 0x10), 0u);
  EXPECT_EQ(source[0].cu->lds().read32(source[0].lds_base + 0x10), kValue);

  sim.engine->run();
}

TEST(Gfx1250SimulationTest, ClusterLdsDoesNotRemapIntoNonParticipatingPeerLdsBase) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t cluster_size = 8;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim(make_single_se_gfx1250_config(/*num_cus=*/4));
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, cluster_size,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  struct RemapCase {
    amdgpu::ClusterLdsTarget source;
    amdgpu::ClusterLdsTarget target;
  };
  std::optional<RemapCase> remap_case;
  for (uint32_t wg_id = 0; wg_id < cluster_size && !remap_case; ++wg_id) {
    auto source = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, wg_id, /*mcast_mask=*/0);
    ASSERT_EQ(source.size(), 1u);
    ASSERT_NE(source[0].cu, nullptr);

    auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, wg_id,
                                                 /*mcast_mask=*/(1u << cluster_size) - 1);
    for (const auto &target : targets) {
      if (target.wg_id == wg_id)
        continue;
      if (target.cu == source[0].cu && target.lds_base != source[0].lds_base) {
        remap_case = RemapCase{source[0], target};
        break;
      }
    }
  }
  ASSERT_TRUE(remap_case.has_value());
  const auto source = remap_case->source;
  const auto target = remap_case->target;
  ASSERT_NE(target.cu, nullptr);
  EXPECT_EQ(target.cu, source.cu);
  EXPECT_NE(target.lds_base, source.lds_base);

  constexpr uint32_t kValue = 0x13579bdf;
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = source.wg_id;
  txn.source_cluster_rank = source.cluster_rank;
  txn.source_lds_base = source.lds_base;
  txn.mcast_mask = (1u << cluster_size) - 1;
  txn.bytes_per_lane = sizeof(kValue);
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = source.lds_base + 0x20;
  txn.payload.resize(sizeof(kValue));
  std::memcpy(txn.payload.data(), &kValue, sizeof(kValue));
  txn.targets = {target};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(std::move(txn), []() {}), amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(target.cu->lds().read32(target.lds_base + 0x20), 0u);
  EXPECT_EQ(source.cu->lds().read32(source.lds_base + 0x20), 0u);

  sim.engine->run();
}

TEST(Gfx1250SimulationTest, RejectsUnsupportedClusterSize) {
  constexpr uint64_t kernel_addr = 0x10000;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1,
                           /*cluster_size_x=*/amdgpu::kClusterMulticastMaskBits + 1,
                           /*workgroup_size_x=*/32);

  EXPECT_THROW((void)sim.engine->step(), std::runtime_error);
}

TEST(Gfx1250SimulationTest, RejectsMisalignedClusteredWorkgroupIdOffset) {
  constexpr uint64_t kernel_addr = 0x10000;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  sim.cp()->set_workgroup_id_offset(1);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32);

  EXPECT_THROW((void)sim.engine->step(), std::runtime_error);
}

TEST(Gfx1250SimulationTest, RejectsIncompleteClusterGrid) {
  constexpr uint64_t kernel_addr = 0x10000;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/0);

  EXPECT_THROW((void)sim.engine->step(), std::runtime_error);
}

TEST(Gfx1250SimulationTest, ClusterLoadAsyncToLdsDoesNotWriteMaskExcludedParticipant) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2000;
  constexpr uint32_t lds_bytes_per_wg = 256;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr), // s_mov_b32 s0, input_addr
      0xBE810080u,                                       // s_mov_b32 s1, 0
      0xBEFD0081u,                                       // s_mov_b32 m0, 0x1
      0x30000082u,                                       // v_lshlrev_b32_e32 v0, 2, v0
      0xEE1AC000u,    0x00000000u,
      0x00000000u, // cluster_load_async_to_lds_b32 v0, v0, s[0:1]
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  for (uint32_t byte = 0; byte < 256; ++byte)
    sim.memory->write8(input_addr + byte, static_cast<uint8_t>(0x40u + byte));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x3);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_NE(targets[0].cu, nullptr);
  ASSERT_NE(targets[1].cu, nullptr);
  EXPECT_NE(targets[0].cu, targets[1].cu);
  sim.engine->run();
  EXPECT_TRUE(sim.cp()
                  ->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0,
                                        /*mcast_mask=*/0x3)
                  .empty());

  auto read_unaligned_input = [&](uint32_t byte_offset) {
    uint32_t value = 0;
    for (uint32_t byte = 0; byte < 4; ++byte)
      value |= static_cast<uint32_t>(sim.memory->read8(input_addr + byte_offset + byte))
               << (byte * 8);
    return value;
  };

  for (uint32_t lane = 0; lane < 32; ++lane) {
    uint32_t value0 = targets[0].cu->lds().read32(targets[0].lds_base + lane * 4);
    uint32_t value1 = targets[1].cu->lds().read32(targets[1].lds_base + lane * 4);
    uint32_t wg0_value = read_unaligned_input(lane * 4);
    EXPECT_EQ(value0, wg0_value) << "lane " << lane;
    EXPECT_EQ(value1, 0u) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, ClusterLoadAsyncToLdsWritesEachIssuingParticipant) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2400;
  constexpr uint32_t lds_bytes_per_wg = 256;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr), // s_mov_b32 s0, input_addr
      0xBE810080u,                                       // s_mov_b32 s1, 0
      0xBEFD0083u,                                       // s_mov_b32 m0, 0x3
      0x30000082u,                                       // v_lshlrev_b32_e32 v0, 2, v0
      0xEE1AC000u,    0x00000000u,
      0x00000000u, // cluster_load_async_to_lds_b32 v0, v0, s[0:1]
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  for (uint32_t byte = 0; byte < 256; ++byte)
    sim.memory->write8(input_addr + byte, static_cast<uint8_t>(0x80u + byte));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x3);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_NE(targets[0].cu, nullptr);
  ASSERT_NE(targets[1].cu, nullptr);
  EXPECT_NE(targets[0].cu, targets[1].cu);
  sim.engine->run();

  auto read_unaligned_input = [&](uint32_t byte_offset) {
    uint32_t value = 0;
    for (uint32_t byte = 0; byte < 4; ++byte)
      value |= static_cast<uint32_t>(sim.memory->read8(input_addr + byte_offset + byte))
               << (byte * 8);
    return value;
  };

  for (uint32_t lane = 0; lane < 32; ++lane) {
    uint32_t value0 = targets[0].cu->lds().read32(targets[0].lds_base + lane * 4);
    uint32_t value1 = targets[1].cu->lds().read32(targets[1].lds_base + lane * 4);
    uint32_t expected_value = read_unaligned_input(lane * 4);
    EXPECT_EQ(value0, expected_value) << "lane " << lane;
    EXPECT_EQ(value1, expected_value) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, GlobalLoadAsyncToLdsDropsNetNegativeDestination) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2600;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t kSentinel = 0xa5a5a5a5u;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr + 4), // s_mov_b32 s0, input_addr + 4
      0xBE810080u,                                           // s_mov_b32 s1, 0
      0x7E000280u,                                           // v_mov_b32_e32 v0, 0
      0xEE180000u,    0x00000000u,
      0xFFFFFC00u, // global_load_async_to_lds_b32 v0, v0, s[0:1] offset:-4
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  sim.memory->write32(input_addr, 0x12345678u);

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 1;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 32;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.group_segment_size = lds_bytes_per_wg;
  pkt.kernel_object = kernel_object;
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  ASSERT_TRUE(sim.engine->step());

  auto *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->lds_size(), lds_bytes_per_wg);
  sim.cu()->lds().write32(wf->lds_base(), kSentinel);
  EXPECT_NO_THROW(sim.engine->run());
  EXPECT_EQ(sim.cu()->lds().read32(wf->lds_base()), kSentinel);
}

TEST(Gfx1250SimulationTest, GlobalLoadAsyncToLdsDropsDestinationPastWorkgroupAllocation) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2800;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t kSentinel = 0xa5a5a5a5u;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr - lds_bytes_per_wg),
      0xBE810080u, // s_mov_b32 s1, 0
      0x7E000280u, // v_mov_b32_e32 v0, 0
      0xEE180000u,    0x00000000u,
      0x00010000u, // global_load_async_to_lds_b32 v0, v0, s[0:1] offset:256
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  sim.memory->write32(input_addr, 0x12345678u);

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 1;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 32;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.group_segment_size = lds_bytes_per_wg;
  pkt.kernel_object = kernel_object;
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  ASSERT_TRUE(sim.engine->step());

  auto *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->lds_size(), lds_bytes_per_wg);
  const uint32_t outside_allocation = wf->lds_base() + lds_bytes_per_wg;
  sim.cu()->lds().write32(outside_allocation, kSentinel);
  EXPECT_NO_THROW(sim.engine->run());
  EXPECT_EQ(sim.cu()->lds().read32(outside_allocation), kSentinel);
}

TEST(Gfx1250SimulationTest, ClusterLoadAsyncToLdsDropsNetNegativeDestination) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2700;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t kSentinel = 0xa5a5a5a5u;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr + 4), // s_mov_b32 s0, input_addr + 4
      0xBE810080u,                                           // s_mov_b32 s1, 0
      0xBEFD0083u,                                           // s_mov_b32 m0, 0x3
      0x7E000280u,                                           // v_mov_b32_e32 v0, 0
      0xEE1AC000u,    0x00000000u,
      0xFFFFFC00u, // cluster_load_async_to_lds_b32 v0, v0, s[0:1] offset:-4
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  sim.memory->write32(input_addr, 0x12345678u);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*mcast_mask=*/0x3);
  ASSERT_EQ(targets.size(), 2u);
  for (const auto &target : targets)
    target.cu->lds().write32(target.lds_base, kSentinel);
  EXPECT_NO_THROW(sim.engine->run());
  for (const auto &target : targets)
    EXPECT_EQ(target.cu->lds().read32(target.lds_base), kSentinel);
}

TEST(Gfx1250SimulationTest, ClusterLoadAsyncToLdsAppliesIoffsetToGlobalAndLds) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2800;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t instruction_offset = 4;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr), // s_mov_b32 s0, input_addr
      0xBE810080u,                                       // s_mov_b32 s1, 0
      0xBEFD0083u,                                       // s_mov_b32 m0, 0x3
      0x30000082u,                                       // v_lshlrev_b32_e32 v0, 2, v0
      0xEE1AC000u,    0x00000000u,
      0x00000400u, // cluster_load_async_to_lds_b32 v0, v0, s[0:1] offset:4
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  for (uint32_t byte = 0; byte < 256; ++byte)
    sim.memory->write8(input_addr + byte, static_cast<uint8_t>(0xc0u + byte));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*mcast_mask=*/0x3);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_NE(targets[0].cu, nullptr);
  ASSERT_NE(targets[1].cu, nullptr);
  EXPECT_NE(targets[0].cu, targets[1].cu);
  sim.engine->run();

  auto read_unaligned_input = [&](uint32_t byte_offset) {
    uint32_t value = 0;
    for (uint32_t byte = 0; byte < 4; ++byte)
      value |= static_cast<uint32_t>(sim.memory->read8(input_addr + byte_offset + byte))
               << (byte * 8);
    return value;
  };

  for (uint32_t lane = 0; lane < 32; ++lane) {
    const uint32_t lds_offset = lane * 4 + instruction_offset;
    const uint32_t expected_value = read_unaligned_input(lds_offset);
    EXPECT_EQ(targets[0].cu->lds().read32(targets[0].lds_base + lds_offset), expected_value)
        << "lane " << lane;
    EXPECT_EQ(targets[1].cu->lds().read32(targets[1].lds_base + lds_offset), expected_value)
        << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, Ttmp7ClusterGridYDoesNotBleedIntoZAt16BitBoundary) {
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code), 128);
  constexpr uint32_t kWorkgroupIdOffset = 0x10000;
  sim.cp()->set_workgroup_id_offset(kWorkgroupIdOffset);

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 2;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 32;
  pkt.grid_size_y = 0x10001;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  ASSERT_TRUE(sim.engine->step());

  amdgpu::Wavefront *target = nullptr;
  amdgpu::ComputeUnitCore *target_cu = nullptr;
  for (uint32_t se_idx = 0; se_idx < sim.xcd()->num_shader_engines(); ++se_idx) {
    auto *se = sim.xcd()->shader_engine(se_idx);
    for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx) {
      auto *cu = se->compute_unit(cu_idx);
      for (uint32_t wf_idx = 0; wf_idx < cu->num_wf_slots(); ++wf_idx) {
        auto *wf = cu->wf(wf_idx);
        if (wf && wf->sgpr_alloc().count > 0 && wf->wg_id() == kWorkgroupIdOffset) {
          target = wf;
          target_cu = cu;
          break;
        }
      }
      if (target)
        break;
    }
    if (target)
      break;
  }

  ASSERT_NE(target, nullptr);
  ASSERT_NE(target_cu, nullptr);
  EXPECT_EQ(target_cu->read_sgpr(target->sgpr_alloc().base + 115), 0u);
}

TEST(Gfx1250SimulationTest, IbSts2ClusterFieldsAreZeroForOrdinaryDispatch) {
  const uint32_t code[] = {
      0xB882199Cu, // s_getreg_b32 s2, hwreg(IB_STS2, 6, 4)
      0xB8831D5Cu, // s_getreg_b32 s3, hwreg(IB_STS2, 21, 4)
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, /*grid_size_x=*/32, /*workgroup_size_x=*/32);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  const auto &wf = sim.snapshot->snapshots().front();
  EXPECT_EQ(wf.sgpr(2), 0u);
  EXPECT_EQ(wf.sgpr(3), 0u);
}

TEST(Gfx1250SimulationTest, DynamicClusterLaunchStateMatchesCompilerAbiWithAlignedOffset) {
  // Scalar workgroup-ID reconstruction emitted by clang 23 when cluster
  // dimensions are not fixed by an amdgpu-cluster-dims attribute.
  const uint32_t code[] = {
      0x9302FF72u,    0x0004000Cu, // s_bfe_u32 s2, ttmp6, 0x4000c
      0x9305FF72u,    0x00040010u, // s_bfe_u32 s5, ttmp6, 0x40010
      0x9309FF72u,    0x00040014u, // s_bfe_u32 s9, ttmp6, 0x40014
      0x81038102u,                 // s_add_co_i32 s3, s2, 1
      0x8B06FF73u,    0x0000FFFFu, // s_and_b32 s6, ttmp7, 0xffff
      0x81078105u,                 // s_add_co_i32 s7, s5, 1
      0x850A9073u,                 // s_lshr_b32 s10, ttmp7, 16
      0x810B8109u,                 // s_add_co_i32 s11, s9, 1
      0x8B048F72u,                 // s_and_b32 s4, ttmp6, 15
      0x96030375u,                 // s_mul_i32 s3, ttmp9, s3
      0x96070706u,                 // s_mul_i32 s7, s6, s7
      0x9308FF72u,    0x00040004u, // s_bfe_u32 s8, ttmp6, 0x40004
      0x960B0B0Au,                 // s_mul_i32 s11, s10, s11
      0x930CFF72u,    0x00040008u, // s_bfe_u32 s12, ttmp6, 0x40008
      0xB88D199Cu,                 // s_getreg_b32 s13, hwreg(IB_STS2, 6, 4)
      0x81030304u,                 // s_add_co_i32 s3, s4, s3
      0x81070708u,                 // s_add_co_i32 s7, s8, s7
      0x810B0B0Cu,                 // s_add_co_i32 s11, s12, s11
      0xBF06800Du,                 // s_cmp_eq_u32 s13, 0
      0x980A0B0Au,                 // s_cselect_b32 s10, s10, s11
      0x98030375u,                 // s_cselect_b32 s3, ttmp9, s3
      0x98060706u,                 // s_cselect_b32 s6, s6, s7
      0xB8821D5Cu,                 // s_getreg_b32 s2, hwreg(IB_STS2, 21, 4)
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code), 128);
  constexpr uint32_t kWorkgroupIdOffset = 16;
  sim.cp()->set_workgroup_id_offset(kWorkgroupIdOffset);

  amdgpu::AmdExtKernelDispatchPacket pkt{};
  pkt.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  pkt.amd_format = amdgpu::kHsaAmdPacketTypeExtKernelDispatch;
  pkt.setup = 3;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.cluster_count_x = 2;
  pkt.cluster_count_y = 2;
  pkt.cluster_count_z = 2;
  pkt.cluster_size_x = 4;
  pkt.cluster_size_y = 2;
  pkt.cluster_size_z = 1;
  pkt.kernel_object = kernel_object;

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  step_until_xcd_halted(sim);

  struct LaunchState {
    uint32_t workgroup_id;
    uint32_t ttmp6;
    uint32_t ttmp7;
    uint32_t ttmp9;
    uint32_t cluster_rank;
    uint32_t global_x;
    uint32_t global_y;
    uint32_t global_z;
    uint32_t cluster_id;
  };
  std::vector<LaunchState> states;
  for (const auto &wf : sim.snapshot->snapshots())
    states.push_back({wf.wg_id, wf.ttmp(6), wf.ttmp(7), wf.ttmp(9), wf.sgpr(2), wf.sgpr(3),
                      wf.sgpr(6), wf.sgpr(10), wf.sgpr(13)});
  std::sort(states.begin(), states.end(), [](const LaunchState &lhs, const LaunchState &rhs) {
    return lhs.workgroup_id < rhs.workgroup_id;
  });

  ASSERT_EQ(states.size(), 64u);
  for (uint32_t local_workgroup_id = 0; local_workgroup_id < states.size(); ++local_workgroup_id) {
    const uint32_t workgroup_id = local_workgroup_id + kWorkgroupIdOffset;
    const uint32_t workgroup_x = workgroup_id % 8;
    const uint32_t workgroup_y = (workgroup_id / 8) % 4;
    const uint32_t workgroup_z = workgroup_id / 32;
    const uint32_t local_x = workgroup_x % 4;
    const uint32_t local_y = workgroup_y % 2;
    const uint32_t local_z = workgroup_z % 1;
    const auto &state = states[local_workgroup_id];
    EXPECT_EQ(state.workgroup_id, workgroup_id);
    EXPECT_EQ(state.ttmp6, 0x07013000u | local_x | (local_y << 4) | (local_z << 8));
    EXPECT_EQ(state.ttmp7, (workgroup_z << 16) | (workgroup_y / 2));
    EXPECT_EQ(state.ttmp9, workgroup_x / 4);
    EXPECT_EQ(state.cluster_rank, local_x + 4 * local_y);
    EXPECT_EQ(state.global_x, workgroup_x);
    EXPECT_EQ(state.global_y, workgroup_y);
    EXPECT_EQ(state.global_z, workgroup_z);
    EXPECT_EQ(state.cluster_id, 1u);
  }
}

} // namespace
