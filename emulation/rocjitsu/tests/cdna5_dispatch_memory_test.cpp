// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

namespace rocjitsu::amdgpu {

class WriterPreferredAccessGateTestAccess {
public:
  static bool has_waiting_writer(WriterPreferredAccessGate &gate) {
    std::lock_guard lock(gate.mutex_);
    return gate.waiting_writers_ != 0;
  }
};

} // namespace rocjitsu::amdgpu
namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

TEST(Gfx1250ExecutionTest, LdsAccessesRejectWrappedRanges) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto &lds = cu->lds();
  constexpr uint32_t kWrappedAddress = UINT32_MAX - 3;
  constexpr uint32_t kValue = 0x12345678u;

  lds.write32(kWrappedAddress, kValue);
  EXPECT_EQ(lds.read32(kWrappedAddress), 0u);

  std::array<uint8_t, sizeof(kValue)> bytes{};
  std::memcpy(bytes.data(), &kValue, sizeof(kValue));
  lds.write(kWrappedAddress, bytes.data(), bytes.size());
  bytes.fill(0xff);
  const auto &const_lds = lds;
  const_lds.read(kWrappedAddress, bytes.data(), bytes.size());
  EXPECT_EQ(bytes, (std::array<uint8_t, sizeof(kValue)>{}));
}

TEST(Gfx1250SimulationTest, DispatchesEndpgmThroughConfig) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  EXPECT_EQ(sim.cp()->dispatched_count(), 1u);
  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  EXPECT_EQ(sim.snapshot->snapshots().front().wf_size, 32u);
  EXPECT_EQ(sim.cu()->num_wfs(), 0u);
}

TEST(Gfx1250SimulationTest, DispatchedModeSetterControlsPseudoScalarRounding) {
  constexpr uint32_t kExpectedRoundTowardPositive = 0x3FB504F4u;
  const uint32_t code[] = {
      0xB9800801u, // s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 1
      0x00000001u,
      0xD6800004u, // v_s_exp_f32 s4, 0.5
      0x000000FFu, 0x3F000000u, S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *snapshot = dispatch_one_wave(sim, code, std::size(code));

  ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->sgpr(4), kExpectedRoundTowardPositive);
}

TEST(Gfx1250SimulationTest, MultiWaveDispatchHonorsPackedTidComponentCount) {
  const uint32_t code[] = {S_ENDPGM_GFX12};

  for (uint32_t component_count = 0; component_count <= 1; ++component_count) {
    SCOPED_TRACE("component_count=" + std::to_string(component_count));
    Gfx1250Sim sim;
    uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code), 104, 32, 2, false,
                                              false, false, 0, 0, 0, 0, component_count);

    test::AqlQueue queue(sim.memory, sim.cp());
    hsa_kernel_dispatch_packet_t pkt{};
    pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    pkt.setup = 2;
    pkt.workgroup_size_x = 32;
    pkt.workgroup_size_y = 4;
    pkt.workgroup_size_z = 1;
    pkt.grid_size_x = 32;
    pkt.grid_size_y = 4;
    pkt.grid_size_z = 1;
    pkt.kernel_object = kernel_object;
    queue.submit(pkt);
    step_until_xcd_halted(sim);

    std::vector<uint32_t> lane0_values;
    std::vector<uint32_t> lane31_values;
    for (const auto &wf : sim.snapshot->snapshots()) {
      lane0_values.push_back(wf.vgpr(0, 0));
      lane31_values.push_back(wf.vgpr(0, 31));
    }

    std::sort(lane0_values.begin(), lane0_values.end());
    std::sort(lane31_values.begin(), lane31_values.end());
    const uint32_t y_scale = component_count >= 1 ? 1u << 10 : 0;
    const std::vector<uint32_t> expected_lane0{0u, y_scale, 2 * y_scale, 3 * y_scale};
    const std::vector<uint32_t> expected_lane31{31u, 31u | y_scale, 31u | (2 * y_scale),
                                                31u | (3 * y_scale)};
    EXPECT_EQ(lane0_values, expected_lane0);
    EXPECT_EQ(lane31_values, expected_lane31);
  }
}

// Collect the EXEC mask each wavefront had at halt (captured before it freed).
std::vector<uint64_t> collect_active_exec_masks(Gfx1250Sim &sim) {
  std::vector<uint64_t> exec_masks;
  for (const auto &wf : sim.snapshot->snapshots())
    exec_masks.push_back(wf.exec);
  std::sort(exec_masks.begin(), exec_masks.end());
  return exec_masks;
}

TEST(Gfx1250SimulationTest, PartialWorkgroupMasksTailWaveExec) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 33, 33);
  step_until_xcd_halted(sim);

  // One 33-thread workgroup exercises the intra-workgroup tail wave.
  const std::vector<uint64_t> expected{1ULL, 0xFFFFFFFFULL};
  EXPECT_EQ(collect_active_exec_masks(sim), expected);
}

// Count the distinct workgroup ids across all wavefronts activated for the
// (single) dispatch. dispatched_count() counts dispatch packets, not
// workgroups, so it cannot observe the rounding; wavefront wg_ids can.
uint32_t count_dispatched_workgroups(Gfx1250Sim &sim) {
  std::set<uint32_t> wg_ids;
  for (const auto &wf : sim.snapshot->snapshots())
    wg_ids.insert(wf.wg_id);
  return static_cast<uint32_t>(wg_ids.size());
}

// The grid-to-workgroup count must round up: a partial final workgroup still
// counts. With grid_size_x=65 and workgroup_size_x=32, HW dispatches 3 WGs
// (ceil(65/32)); the old floor division dropped the tail WG and dispatched 2.
TEST(Gfx1250SimulationTest, PartialFinalWorkgroupRoundsUpDispatchCount) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, /*grid_size_x=*/65, /*workgroup_size_x=*/32);
  step_until_xcd_halted(sim);

  EXPECT_EQ(count_dispatched_workgroups(sim), 3u);
}

// Same rounding rule in 2D: grid 65x33 with workgroups 32x16 dispatches
// ceil(65/32) * ceil(33/16) = 3 * 3 = 9 workgroups, not floor's 2 * 2 = 4.
TEST(Gfx1250SimulationTest, PartialFinalWorkgroupRoundsUpDispatchCount2D) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 2; // 2D grid.
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 16;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 65;
  pkt.grid_size_y = 33;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  queue.submit(pkt);
  step_until_xcd_halted(sim);

  EXPECT_EQ(count_dispatched_workgroups(sim), 9u);
}

TEST(Gfx1250SimulationTest, PartialGridTailMasksFinalWorkgroupExec) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 33, 32);
  step_until_xcd_halted(sim);

  // Unlike PartialWorkgroupMasksTailWaveExec, this uses two full-size
  // workgroups and exercises the grid-bounds mask on the final workgroup.
  const std::vector<uint64_t> expected{1ULL, 0xFFFFFFFFULL};
  EXPECT_EQ(collect_active_exec_masks(sim), expected);
}

TEST(Gfx1250SimulationTest, Partial2DGridTailMasksNonContiguousExecLanes) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 2;
  pkt.workgroup_size_x = 8;
  pkt.workgroup_size_y = 2;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 13;
  pkt.grid_size_y = 2;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  queue.submit(pkt);
  step_until_xcd_halted(sim);

  const std::vector<uint64_t> expected{0x1F1FULL, 0xFFFFULL};
  EXPECT_EQ(collect_active_exec_masks(sim), expected);
}

TEST(Gfx1250SimulationTest, DispatchPreloadsKernargDwordsIntoUserSgprs) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint64_t kKernargAddr = 0x400000;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  struct Args {
    uint32_t skip;
    uint32_t first;
    uint32_t second;
  } args{0x11111111u, 0x22222222u, 0x33333333u};

  uint32_t kernel_code_properties = 0;
  AMDHSA_BITS_SET(kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);

  Gfx1250Sim sim;
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), kKernargAddr);
  uint64_t kernel_object =
      sim.write_kernel(kKernelAddr, code, std::size(code), 104, 32, 4, false, false, false,
                       kernel_code_properties, sizeof(args), 2, 1);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  const auto &wf = sim.snapshot->snapshots().front();
  EXPECT_EQ(wf.sgpr64(0), kKernargAddr);
  EXPECT_EQ(wf.sgpr(2), args.first);
  EXPECT_EQ(wf.sgpr(3), args.second);
}

TEST(Gfx1250SimulationTest, DispatchPreloadsKernargWhenDescriptorSizeIsUnknown) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint64_t kKernargAddr = 0x400000;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  const std::array<uint32_t, 3> args{0x11111111u, 0x22222222u, 0x33333333u};

  uint32_t kernel_code_properties = 0;
  AMDHSA_BITS_SET(kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);

  Gfx1250Sim sim;
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(args.data()),
                         args.size() * sizeof(args[0]), kKernargAddr);
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code, std::size(code), 104, 32, 4, false,
                                            false, false, kernel_code_properties, 0, 2, 1);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  const auto &wf = sim.snapshot->snapshots().front();
  EXPECT_EQ(wf.sgpr64(0), kKernargAddr);
  EXPECT_EQ(wf.sgpr(2), args[1]);
  EXPECT_EQ(wf.sgpr(3), args[2]);
}

TEST(Gfx1250SimulationTest, DispatchDecodesSixBitUserSgprCountForKernargPreload) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint64_t kKernargAddr = 0x400000;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  std::array<uint32_t, 30> args{};
  for (uint32_t i = 0; i < args.size(); ++i)
    args[i] = 0x1000u + i;

  uint32_t kernel_code_properties = 0;
  AMDHSA_BITS_SET(kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);

  Gfx1250Sim sim;
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(args.data()),
                         args.size() * sizeof(args[0]), kKernargAddr);
  uint64_t kernel_object =
      sim.write_kernel(kKernelAddr, code, std::size(code), 104, 32, 32, false, false, false,
                       kernel_code_properties, args.size() * sizeof(args[0]), args.size(), 0);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  const auto &wf = sim.snapshot->snapshots().front();
  EXPECT_EQ(wf.sgpr64(0), kKernargAddr);
  for (uint32_t i = 0; i < args.size(); ++i)
    EXPECT_EQ(wf.sgpr(i + 2), args[i]);
}

TEST(Gfx1250SimulationTest, SLoadB32DoesNotScaleImmediateOffset) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint64_t kKernargAddr = 0x400000;
  constexpr uint32_t kExpected = 0x12345678u;

  std::vector<uint32_t> code;
  // s_load_b32 s4, s[0:1], 0x4 scale_offset
  append_instruction(code, make_s_load_b32_scaled_imm(4, 0, 4));
  append_instruction(code, S_WAIT_KMCNT_0_GFX12);
  append_instruction(code, S_ENDPGM_GFX12);

  uint32_t kernel_code_properties = 0;
  AMDHSA_BITS_SET(kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);

  Gfx1250Sim sim;
  write_global_u32(*sim.memory, kKernargAddr + 4, kExpected);
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code.data(), code.size(), 104, 32, 2,
                                            false, false, false, kernel_code_properties, 16);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  EXPECT_EQ(sim.snapshot->snapshots().front().sgpr(4), kExpected);
}

TEST(Gfx1250SimulationTest, TtmpWorkgroupIdsUseGridCoordinatesFor2DDispatch) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object =
      sim.write_kernel(0x10000, code, std::size(code), 104, 32, 2, true, false, false);

  test::AqlQueue queue(sim.memory, sim.cp());
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 2;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 96;
  pkt.grid_size_y = 2;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  queue.submit(pkt);
  step_until_xcd_halted(sim);

  const auto *target = sim.snapshot->by_wg_id(4);
  ASSERT_NE(target, nullptr);
  // TTMP9 holds grid_wg_id_x; TTMP7 packs wg_id_y/z; TTMP8 bit 30 is grid_yz_valid.
  EXPECT_EQ(target->ttmp(9), 1u);
  EXPECT_EQ(target->ttmp(7), 1u);
  EXPECT_NE(target->ttmp(8) & (1u << 30), 0u);
}

TEST(Gfx1250SimulationTest, Ttmp8EncodesWaveIdWithinWorkgroup) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 64, 64);
  step_until_xcd_halted(sim);

  ASSERT_EQ(sim.snapshot->snapshots().size(), 2u);
  std::vector<uint32_t> ttmp8_values;
  for (const auto &wf : sim.snapshot->snapshots())
    ttmp8_values.push_back(wf.ttmp(8));
  std::sort(ttmp8_values.begin(), ttmp8_values.end());
  EXPECT_EQ(ttmp8_values, (std::vector<uint32_t>{0, 1u << 25}));
}

TEST(Gfx1250SimulationTest, Ttmp8EncodesQueuePacketId) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  queue.dispatch(kernel_object, 32, 32);
  step_until_xcd_halted(sim);

  ASSERT_EQ(sim.snapshot->snapshots().size(), 2u);
  // Dispatch ids increase per packet but are not dense: each command processor
  // mints them strided by the XCD count so ids from different XCDs cannot
  // collide. Order the observations by id instead of indexing with it.
  std::array<std::pair<uint32_t, uint32_t>, 2> by_dispatch{};
  size_t i = 0;
  for (const auto &wf : sim.snapshot->snapshots()) {
    ASSERT_GE(wf.dispatch_id, 1u);
    by_dispatch[i++] = {wf.dispatch_id, wf.ttmp(8) & 0x1FFFFFFu};
  }
  std::sort(by_dispatch.begin(), by_dispatch.end());
  ASSERT_LT(by_dispatch[0].first, by_dispatch[1].first);
  EXPECT_EQ(by_dispatch[0].second, 0u);
  EXPECT_EQ(by_dispatch[1].second, 1u);
}

TEST(Gfx1250SimulationTest, GlobalStoreWritesVisibleMemory) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t output_addr = 0x2000;

  const uint32_t code[] = {
      0xBE8400FFu,    static_cast<uint32_t>(output_addr), // s_mov_b32 s4, output_addr
      0xBE850080u,                                        // s_mov_b32 s5, 0
      0x30000082u,                                        // v_lshlrev_b32_e32 v0, 2, v0
      0x7E020281u,                                        // v_mov_b32_e32 v1, 1
      0xEE068004u,    0x00800000u,
      0x00000000u, // global_store_b32 v0, v1, s[4:5]
      0xBFC10000u, // s_wait_storecnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code));
  for (uint32_t lane = 0; lane < 32; ++lane)
    sim.memory->write32(output_addr + lane * sizeof(uint32_t), 0);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  sim.engine->run();
  sim.soc->flush_all();

  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(sim.memory->read32(output_addr + lane * sizeof(uint32_t)), 1u) << "lane " << lane;
}

/// @brief A GLOBAL saddr access sign-extends its VGPR offset on gfx1250.
/// @details LLVM gates this on hasSignedGVSOffset and hands a negative offset to
/// the saddr form whenever an access walks backwards from its base, so the case
/// arises from ordinary reversed iteration. Zero-extending -4 puts the access
/// 4 GiB above the allocation rather than one dword below it, which lands on
/// nothing and is dropped, making the symptom silently missing data rather than
/// a fault.
TEST(Gfx1250SimulationTest, GlobalStoreSignExtendsNegativeSaddrVgprOffset) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t base_addr = 0x2000;
  constexpr uint32_t marker = 0xA5A5A5A5u;

  const uint32_t code[] = {
      0xBE8400FFu,    static_cast<uint32_t>(base_addr), // s_mov_b32 s4, base_addr
      0xBE850080u,                                      // s_mov_b32 s5, 0
      0x7E0002FFu,    0xFFFFFFFCu,                      // v_mov_b32_e32 v0, -4
      0x7E0202FFu,    marker,                           // v_mov_b32_e32 v1, marker
      0xEE068004u,    0x00800000u,
      0x00000000u, // global_store_b32 v0, v1, s[4:5]
      0xBFC10000u, // s_wait_storecnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code));
  sim.memory->write32(base_addr - sizeof(uint32_t), 0);
  sim.memory->write32(base_addr, 0);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  sim.engine->run();
  sim.soc->flush_all();

  EXPECT_EQ(sim.memory->read32(base_addr - sizeof(uint32_t)), marker);
  EXPECT_EQ(sim.memory->read32(base_addr), 0u);
}

/// @brief The scaled form of a GLOBAL saddr offset scales it as a signed value.
/// @details `scale_offset` multiplies the VGPR offset by the access size, and
/// that multiply is a separate path from the unscaled one above. A -1 offset
/// scaled by the four bytes of a b32 access reaches one dword below the base;
/// scaling it unsigned reaches four bytes short of 16 GiB above it instead. The
/// distinction is invisible to the unscaled test, which never multiplies, and to
/// the existing scaled tests, whose offsets are all non-negative.
TEST(Gfx1250SimulationTest, GlobalStoreScalesANegativeSaddrVgprOffsetSigned) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t base_addr = 0x2000;
  constexpr uint32_t marker = 0x5A5A5A5Au;

  const uint32_t code[] = {
      0xBE8400FFu,
      static_cast<uint32_t>(base_addr), // s_mov_b32 s4, base_addr
      0xBE850080u,                      // s_mov_b32 s5, 0
      0x7E0002FFu,
      0xFFFFFFFFu, // v_mov_b32_e32 v0, -1
      0x7E0202FFu,
      marker, // v_mov_b32_e32 v1, marker
      // global_store_b32 v0, v1, s[4:5] scale_offset -- bit 48 of the VGLOBAL
      // encoding, which is bit 16 of the second dword, on top of the vsrc=1
      // already there.
      0xEE068004u,
      0x00810000u,
      0x00000000u,
      0xBFC10000u, // s_wait_storecnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code));
  sim.memory->write32(base_addr - sizeof(uint32_t), 0);
  sim.memory->write32(base_addr, 0);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  sim.engine->run();
  sim.soc->flush_all();

  EXPECT_EQ(sim.memory->read32(base_addr - sizeof(uint32_t)), marker)
      << "a -1 offset scaled by the b32 access size must land one dword below the base";
  EXPECT_EQ(sim.memory->read32(base_addr), 0u);
}

TEST(Gfx1250SimulationTest, BufferStoreUsesM0Soffset) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t output_addr = 0x2000;

  const uint32_t code[] = {
      0xBE8400FFu,    static_cast<uint32_t>(output_addr), // s_mov_b32 s4, output_addr
      0xBE850080u,                                        // s_mov_b32 s5, 0
      0xBE8600FFu,    0x1000u,                            // s_mov_b32 s6, num_records
      0xBEFD0090u,                                        // s_mov_b32 m0, 16
      0x300A0085u,                                        // v_lshlrev_b32_e32 v5, 5, v0
      0x7E000287u,                                        // v_mov_b32_e32 v0, 7
      0xC406807Du,    0x40800800u,
      0x00000005u, // buffer_store_b32 v0, v5, s[4:7], m0 offen
      0xBFC10000u, // s_wait_storecnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    sim.memory->write32(output_addr + lane * 32, 0);
    sim.memory->write32(output_addr + 16 + lane * 32, 0);
  }

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  sim.engine->run();
  sim.soc->flush_all();

  for (uint32_t lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(sim.memory->read32(output_addr + lane * 32), 0u) << "lane " << lane;
    EXPECT_EQ(sim.memory->read32(output_addr + 16 + lane * 32), 7u) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, MemorySideCacheFlushSerializesConcurrentAccess) {
  constexpr uint64_t output_addr = 0x2000;
  constexpr uint32_t iterations = 32;

  Gfx1250Sim sim;
  auto *msc = sim.soc->iod(0)->msc();
  std::barrier start(3);

  std::thread writer([&] {
    start.arrive_and_wait();
    for (uint32_t value = 1; value <= iterations; ++value) {
      msc->write(output_addr, reinterpret_cast<const uint8_t *>(&value), sizeof(value));
      std::this_thread::yield();
    }
  });
  std::thread reader([&] {
    start.arrive_and_wait();
    for (uint32_t i = 0; i < iterations; ++i) {
      uint32_t value = 0;
      msc->read(output_addr, reinterpret_cast<uint8_t *>(&value), sizeof(value));
      EXPECT_LE(value, iterations);
      std::this_thread::yield();
    }
  });
  std::thread flusher([&] {
    start.arrive_and_wait();
    for (uint32_t i = 0; i < iterations; ++i) {
      msc->flush_all();
      std::this_thread::yield();
    }
  });

  writer.join();
  reader.join();
  flusher.join();
  msc->flush_all();

  EXPECT_EQ(sim.memory->read32(output_addr), iterations);
}

TEST(WriterPreferredAccessGateTest, WaitingWriterBlocksLateSharedEntrant) {
  amdgpu::WriterPreferredAccessGate gate;
  std::shared_lock active_reader(gate);
  bool writer_acquired = false;
  std::thread writer([&] {
    std::unique_lock waiting_writer(gate);
    writer_acquired = true;
  });

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (!amdgpu::WriterPreferredAccessGateTestAccess::has_waiting_writer(gate) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();

  const bool writer_waiting = amdgpu::WriterPreferredAccessGateTestAccess::has_waiting_writer(gate);
  EXPECT_TRUE(writer_waiting);
  if (writer_waiting) {
    const bool late_reader_acquired = gate.try_lock_shared();
    EXPECT_FALSE(late_reader_acquired);
    if (late_reader_acquired)
      gate.unlock_shared();
  }

  active_reader.unlock();
  writer.join();
  EXPECT_TRUE(writer_acquired);
}

TEST(Gfx1250SimulationTest, MemorySideCacheFlushMakesProgressUnderContinuousReads) {
  constexpr uint64_t output_addr = 0x2000;
  constexpr uint32_t reader_count = 4;

  Gfx1250Sim sim;
  auto *msc = sim.soc->iod(0)->msc();
  std::barrier start(reader_count + 1);
  std::atomic<bool> stop_readers = false;
  std::atomic<uint32_t> readers_started = 0;
  std::vector<std::thread> readers;
  readers.reserve(reader_count);
  for (uint32_t i = 0; i < reader_count; ++i) {
    readers.emplace_back([&] {
      start.arrive_and_wait();
      bool first_read = true;
      while (!stop_readers.load(std::memory_order_acquire)) {
        uint32_t value = 0;
        msc->read(output_addr, reinterpret_cast<uint8_t *>(&value), sizeof(value));
        if (first_read) {
          readers_started.fetch_add(1, std::memory_order_release);
          first_read = false;
        }
      }
    });
  }

  start.arrive_and_wait();
  while (readers_started.load(std::memory_order_acquire) < reader_count)
    std::this_thread::yield();

  std::atomic<bool> flush_completed = false;
  std::thread flusher([&] {
    msc->flush_all();
    flush_completed.store(true, std::memory_order_release);
  });

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (!flush_completed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const bool completed_while_readers_active = flush_completed.load(std::memory_order_acquire);

  stop_readers.store(true, std::memory_order_release);
  for (auto &reader : readers)
    reader.join();
  flusher.join();

  EXPECT_TRUE(completed_while_readers_active);
}

} // namespace
