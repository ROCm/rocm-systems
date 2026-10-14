// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file ds_gws_test.cpp
/// @brief Tests for the DS Global Wave Sync (GWS) instructions and their
/// device-global store (GwsDevice). Covers three layers: decode/classification of
/// the six GWS opcodes across all seven GFX9/GFX10/GFX11 ISAs; wait-counter
/// accounting (LGKMCNT vs DSCNT) for GWS completion; and the stateful cross-CU
/// synchronization semantics -- barrier/semaphore rendezvous, late-admitted
/// participants, credit entitlement ordering, the deadlock-escape backstop, and
/// store lifetime/reconciliation on CU teardown, re-pointing, and process release.

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gws_device.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using namespace rocjitsu;

// GWS (Global Wave Sync) is defined by every GFX9 (CDNA1-3) ISA and by the
// GFX10/GFX11 (RDNA1-3.5) ISAs.
enum class GwsOp { kReleaseAll, kInit, kSemaV, kSemaBr, kSemaP, kBarrier };

struct GwsVariant {
  GwsOp op;
  std::string_view mnemonic;
};

constexpr std::array<GwsVariant, 6> kGwsVariants{{
    {GwsOp::kReleaseAll, "ds_gws_sema_release_all"},
    {GwsOp::kInit, "ds_gws_init"},
    {GwsOp::kSemaV, "ds_gws_sema_v"},
    {GwsOp::kSemaBr, "ds_gws_sema_br"},
    {GwsOp::kSemaP, "ds_gws_sema_p"},
    {GwsOp::kBarrier, "ds_gws_barrier"},
}};

// INIT/SEMA_BR/BARRIER take the resource count from the ADDR VGPR; the rest are
// address-less.
constexpr bool gws_has_addr(GwsOp op) {
  return op == GwsOp::kInit || op == GwsOp::kSemaBr || op == GwsOp::kBarrier;
}

constexpr bool is_gfx9(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3;
}

uint16_t gws_opcode(rj_code_arch_t arch, GwsOp op) {
  // GFX9 DS GWS opcodes start at 152 (SEMA_RELEASE_ALL); GFX10/11 start at 24.
  const uint16_t base = is_gfx9(arch) ? 152 : 24;
  switch (op) {
  case GwsOp::kReleaseAll:
    return base + 0;
  case GwsOp::kInit:
    return base + 1;
  case GwsOp::kSemaV:
    return base + 2;
  case GwsOp::kSemaBr:
    return base + 3;
  case GwsOp::kSemaP:
    return base + 4;
  case GwsOp::kBarrier:
    return base + 5;
  }
  return base;
}

// GFX11 routes GWS completion through DSCNT; the older ISAs use LGKMCNT.
amdgpu::WaitCounterType expected_wait_counter(rj_code_arch_t arch) {
  return (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
             ? amdgpu::WaitCounterType::DSCNT
             : amdgpu::WaitCounterType::LGKMCNT;
}

std::array<uint32_t, 2> build_gws(rj_code_arch_t arch, uint16_t op, uint8_t gds, uint8_t addr,
                                  uint8_t offset0 = 0) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_ds(op, {.offset0 = offset0, .gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_ds(op, {.offset0 = offset0, .gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_ds(op, {.offset0 = offset0, .gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_ds(op, {.offset0 = offset0, .gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_ds(op, {.offset0 = offset0, .gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_ds(op, {.offset0 = offset0, .gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_ds(op, {.offset0 = offset0, .gds = gds, .addr = addr});
  default:
    ADD_FAILURE() << "unsupported arch for GWS build: " << unsigned(arch);
    return {};
  }
}

// Build a plain workgroup s_barrier (SOPP) for the given arch. GFX9/GFX10 use
// opcode 10; GFX11 uses 61.
std::array<uint32_t, 1> build_s_barrier(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_sopp(cdna1::kSBarrierSopp);
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_sopp(cdna2::kSBarrierSopp);
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_sopp(cdna3::kSBarrierSopp);
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_sopp(rdna1::kSBarrierSopp);
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_sopp(rdna2::kSBarrierSopp);
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_sopp(rdna3::kSBarrierSopp);
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_sopp(rdna3_5::kSBarrierSopp);
  default:
    ADD_FAILURE() << "unsupported arch for s_barrier build: " << unsigned(arch);
    return {};
  }
}

// Build s_endpgm (SOPP) for the given arch. GFX9/GFX10 use opcode 1; GFX11
// (RDNA3/3.5) uses opcode 48 -- the same word on GFX11 would decode to s_setkill,
// so a parked wave released to this PC must use the arch-correct encoding to
// actually retire.
uint32_t gws_s_endpgm_word(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_sopp(cdna1::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_sopp(cdna2::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_sopp(cdna3::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_sopp(rdna1::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_sopp(rdna2::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_sopp(rdna3::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_sopp(rdna3_5::kSEndpgmSopp)[0];
  default:
    ADD_FAILURE() << "unsupported arch for s_endpgm build: " << unsigned(arch);
    return 0xBF810000u;
  }
}

// Build a GWS op carrying the DS acc bit (CDNA2/3 only, where the field exists).
std::array<uint32_t, 2> build_gws_acc(rj_code_arch_t arch, uint16_t op, uint8_t acc, uint8_t addr,
                                      uint8_t offset0 = 0) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_ds(op, {.offset0 = offset0, .gds = 1, .acc = acc, .addr = addr});
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_ds(op, {.offset0 = offset0, .gds = 1, .acc = acc, .addr = addr});
  default:
    ADD_FAILURE() << "DS acc bit only exists on CDNA2/3: " << unsigned(arch);
    return {};
  }
}

// ADDR VGPR that carries the resource count for INIT/SEMA_BR/BARRIER.
constexpr uint8_t kAddrVgpr = 4;

class DsGwsTest : public ::testing::TestWithParam<rj_code_arch_t> {};

INSTANTIATE_TEST_SUITE_P(GwsArchitectures, DsGwsTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_RDNA1,
                                           ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                                           ROCJITSU_CODE_ARCH_RDNA3_5));

// Each GWS operation must decode, execute without raising
// UnimplementedInstruction, and publish a zero-payload LOCAL_MEM state that
// participates in the arch's GWS wait counter (LGKMCNT on GFX9/GFX10, DSCNT on
// GFX11) against the GDS completion class.
TEST_P(DsGwsTest, StructuralModelExecutesAndAccounts) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_mem");
  amdgpu::L2Cache l2("ds_gws_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 102;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("ds_gws", cfg, &mem, &l2);
  auto decoder = Decoder::create(arch);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
  ASSERT_NE(wf, nullptr);
  // Decode and memory-pipeline accounting are EXEC-independent on every target;
  // an empty mask exercises that here. The stateful EXEC semantics (RDNA3/3.5
  // run the count ops via lane 0, older targets gate them out) are covered by
  // the dedicated EXEC-selection cases below (Rdna3CountOpsHonorLaneZero...,
  // OlderTargetsSkipCountOps..., CountOpsReadFirstActiveLaneNotLaneZero).
  wf->set_exec(0);
  wf->set_m0(0);

  const auto wait_counter = expected_wait_counter(arch);
  for (const auto &variant : kGwsVariants) {
    const auto words = build_gws(arch, gws_opcode(arch, variant.op), /*gds=*/1,
                                 gws_has_addr(variant.op) ? kAddrVgpr : 0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << variant.mnemonic << " failed to decode on arch " << unsigned(arch);
    EXPECT_EQ(inst->mnemonic(), variant.mnemonic);

    // Previously these threw UnimplementedInstruction; they must now succeed.
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded())
        << variant.mnemonic << " execution failed on arch " << unsigned(arch);

    // A zero-payload LOCAL_MEM store is published so the memory pipeline takes
    // the ordinary completion path.
    ASSERT_NE(inst->data(), nullptr) << variant.mnemonic;
    EXPECT_EQ(inst->data()->tag(), amdgpu::LOCAL_MEM);
    const auto *state = inst->data_as<amdgpu::VectorMemState>();
    EXPECT_EQ(state->num_elems, 0u) << variant.mnemonic;
    EXPECT_FALSE(state->is_load) << variant.mnemonic;
    EXPECT_EQ(state->wait_counter_type, wait_counter) << variant.mnemonic;

    // Wait-accounting metadata: MEMORY_OP with an EXEC-independent GDS
    // obligation on the arch's GWS counter (GWS legitimately targets GDS).
    EXPECT_TRUE(inst->is_memory_op()) << variant.mnemonic;
    const auto *info = inst->amdgpu_memory_issue_info();
    ASSERT_NE(info, nullptr) << variant.mnemonic;
    EXPECT_FALSE(info->exec_masked) << variant.mnemonic;
    bool has_gds_obligation = false;
    for (const auto &ob : info->counter_obligations()) {
      if (ob.wait_counter_type() == wait_counter &&
          ob.completion_class() == amdgpu::MemoryCompletionClass::GDS)
        has_gds_obligation = true;
    }
    EXPECT_TRUE(has_gds_obligation) << variant.mnemonic;

    // The op completes cleanly through the local-memory pipeline with no
    // register effect and no cross-workgroup blocking.
    amdgpu::LocalMemPipeline pipeline;
    pipeline.issue(inst.release(), *wf);
  }

  wf->halt();
}

// Without the GDS bit the model still executes (architecturally GWS always
// uses GDS, but the structural path must not throw): it decodes and completes
// as an LDS-classified op.
TEST_P(DsGwsTest, DecodesWithoutGdsBit) {
  const auto arch = GetParam();
  auto decoder = Decoder::create(arch);
  for (const auto &variant : kGwsVariants) {
    const auto words = build_gws(arch, gws_opcode(arch, variant.op), /*gds=*/0,
                                 gws_has_addr(variant.op) ? kAddrVgpr : 0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << variant.mnemonic << " failed to decode on arch " << unsigned(arch);
    EXPECT_EQ(inst->mnemonic(), variant.mnemonic);
    EXPECT_TRUE(inst->is_memory_op()) << variant.mnemonic;
  }
}

// ---------------------------------------------------------------------------
// Stateful GWS: device-global rendezvous. These exercise the shared counter
// table through execute_instruction(), which runs the generated execute body
// (decode rid, read count, call the CU hook). Resources are device-global and
// persist across dispatches, so a signal/arrival from any wave -- any workgroup,
// any CU, even a later dispatch -- releases a waiter parked by another.
// ---------------------------------------------------------------------------

// Decode one GWS op, publish @p count in the ADDR VGPR's first active lane, and
// run it on @p wf through the CU execute path. @p offset0 feeds the rid decode.
void run_gws(amdgpu::ComputeUnitCore &cu, Decoder &decoder, rj_code_arch_t arch, GwsOp op,
             amdgpu::Wavefront &wf, uint32_t count, uint8_t offset0 = 0) {
  const bool has_addr = gws_has_addr(op);
  if (has_addr)
    cu.write_vgpr(wf.vgpr_alloc().base + kAddrVgpr, /*lane=*/0, count);
  const auto words =
      build_gws(arch, gws_opcode(arch, op), /*gds=*/1, has_addr ? kAddrVgpr : 0, offset0);
  std::unique_ptr<Instruction> inst(decode_valid(decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu.execute_instruction(inst.get(), wf).succeeded());
}

std::unique_ptr<amdgpu::ComputeUnitCore> make_gws_cu(amdgpu::GpuMemory &mem, amdgpu::L2Cache &l2,
                                                     rj_code_arch_t arch, uint32_t wf_slots) {
  l2.set_backing_memory(&mem);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = wf_slots;
  cfg.sgprs_per_wf = 102;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  return amdgpu::ComputeUnitCore::create("ds_gws_cu", cfg, &mem, &l2);
}

// Like run_gws, but with an explicit EXEC mask and per-lane ADDR-VGPR counts so
// the EXEC=0 and nonzero-first-active-lane paths can be exercised. Each entry of
// @p lane_counts is (lane, value); @p exec is written verbatim onto the wave.
void run_gws_exec(amdgpu::ComputeUnitCore &cu, Decoder &decoder, rj_code_arch_t arch, GwsOp op,
                  amdgpu::Wavefront &wf, uint64_t exec,
                  std::initializer_list<std::pair<uint32_t, uint32_t>> lane_counts,
                  uint8_t offset0 = 0) {
  const bool has_addr = gws_has_addr(op);
  if (has_addr)
    for (const auto &[lane, value] : lane_counts)
      cu.write_vgpr(wf.vgpr_alloc().base + kAddrVgpr, lane, value);
  wf.set_exec(exec);
  const auto words =
      build_gws(arch, gws_opcode(arch, op), /*gds=*/1, has_addr ? kAddrVgpr : 0, offset0);
  std::unique_ptr<Instruction> inst(decode_valid(decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu.execute_instruction(inst.get(), wf).succeeded());
}

// RDNA3/RDNA3.5 (ISA "Global Wave Sync", section 13.4.2) run the count-carrying
// GWS operations (INIT, BARRIER, SEMA_BR) even when EXEC selects no lane, taking
// the count operand from lane 0. A barrier(count=1) issued with EXEC=0 must
// therefore still establish a real two-participant rendezvous: the first arrival
// parks and the co-resident arrival releases it. An EXEC-gated structural no-op
// (the pre-fix behavior) would instead leave the first arrival RUNNING. This is
// RDNA3/3.5-specific, so the case is skipped on the other parameterized targets.
TEST_P(DsGwsTest, Rdna3CountOpsHonorLaneZeroWhenExecEmpty) {
  const auto arch = GetParam();
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "EXEC=0 lane-0 count ops are specific to RDNA3/3.5";
  amdgpu::GpuMemory mem("ds_gws_exec0_mem");
  amdgpu::L2Cache l2("ds_gws_exec0_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1})
    wf->set_m0(0);

  // barrier(count=1) with EXEC=0 parks the first arrival -> the op took effect.
  // The first unarmed barrier self-seeds its counter from the lane-0 count, so no
  // separate init is needed here; INIT independence is witnessed on its own below.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  // The co-resident arrival (also EXEC=0) releases it: a genuine rendezvous.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// INIT must be witnessed independently of BARRIER. The barrier rendezvous above
// cannot stand in: an unarmed first barrier self-seeds its own counter, so an
// EXEC-gated INIT would leave every assertion there unchanged. Here an EXEC=0
// init(count=1) seeds a *semaphore credit* via lane 0; a later live-EXEC P
// consumes it and stays RUNNING. An EXEC-gated INIT would seed no credit, so the
// first P would park -- the RUNNING assertion is the actual INIT witness.
TEST_P(DsGwsTest, Rdna3InitSeedsCreditViaLaneZeroWhenExecEmpty) {
  const auto arch = GetParam();
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "EXEC=0 lane-0 count ops are specific to RDNA3/3.5";
  amdgpu::GpuMemory mem("ds_gws_init_exec0_mem");
  amdgpu::L2Cache l2("ds_gws_init_exec0_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1})
    wf->set_m0(0);

  // init(count=1) with EXEC=0 seeds one semaphore credit via lane 0.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kInit, *wf0, /*exec=*/0, {{0, 1}});
  // First P (live EXEC) consumes the seeded credit and stays RUNNING: the witness
  // that the EXEC=0 init took effect. An EXEC-gated init would leave no credit.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*exec=*/0x1, {});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  // Second P (live EXEC) finds no credit left and parks: only one was seeded.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kSemaP, *wf1, /*exec=*/0x1, {});
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  wf0->halt();
  wf1->halt();
}

// SEMA_BR is the third EXEC-independent count op: on RDNA3/3.5 a sema_br issued
// with EXEC=0 still releases parked waiters, taking the release count from lane 0.
// (sema_p itself stays EXEC-gated, so the waiter is parked with a live lane.)
TEST_P(DsGwsTest, Rdna3SemaBrReleasesViaLaneZeroWhenExecEmpty) {
  const auto arch = GetParam();
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "EXEC=0 lane-0 count ops are specific to RDNA3/3.5";
  amdgpu::GpuMemory mem("ds_gws_br_exec0_mem");
  amdgpu::L2Cache l2("ds_gws_br_exec0_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1})
    wf->set_m0(0);

  // sema_p parks wf0 (EXEC-gated op, so it uses a live lane).
  run_gws_exec(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*exec=*/0x1, {{0, 0}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  // sema_br(count=1) with EXEC=0 releases the waiter via the lane-0 count.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kSemaBr, *wf1, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// The empty-EXEC rule is RDNA3/3.5-specific and must NOT be extrapolated to the
// older GFX9/GFX10 targets: there a count op issued with EXEC=0 stays an
// EXEC-gated structural no-op (no park). The identical op with a live lane still
// parks normally, proving only the empty mask -- not the resource -- was gated.
TEST_P(DsGwsTest, OlderTargetsSkipCountOpsWhenExecEmpty) {
  const auto arch = GetParam();
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "RDNA3/3.5 run EXEC=0 count ops via lane 0 rather than gating";
  amdgpu::GpuMemory mem("ds_gws_exec0_old_mem");
  amdgpu::L2Cache l2("ds_gws_exec0_old_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1})
    wf->set_m0(0);

  // EXEC=0: the barrier is skipped, so the first arrival does not park.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  // With a live lane the identical op parks, then its co-resident releases it.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*exec=*/0x1, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws_exec(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*exec=*/0x1, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// The count operand comes from the first active lane (countr_zero(EXEC)), never a
// hardwired lane 0. With lane 0 masked out and lane 3 the first active lane, the
// count must be read from lane 3: lane 0 carries an oversized count (99 -> 100
// participants, which would take the residency fallback and not park), while
// lane 3 carries 1 (two participants, which parks then releases). EXEC here is
// non-empty, so first-active-lane selection is identical on all seven targets
// (no EXEC=0 gating distinction applies), and the case runs unskipped everywhere.
TEST_P(DsGwsTest, CountOpsReadFirstActiveLaneNotLaneZero) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_firstlane_mem");
  amdgpu::L2Cache l2("ds_gws_firstlane_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1})
    wf->set_m0(0);

  // EXEC bit 3 only: first active lane is 3. Reading lane 0 (99) would oversize
  // the barrier and keep wf0 RUNNING; reading lane 3 (1) parks it.
  run_gws_exec(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*exec=*/0x8, {{0, 99}, {3, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws_exec(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*exec=*/0x8, {{0, 99}, {3, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// A GWS barrier initialized for two resident participants parks the early
// arrival; the final arrival releases the co-resident set. GWS programs the
// barrier value as (participants - 1), so two participants use a count of 1.
TEST_P(DsGwsTest, BarrierParksUntilAllCoResidentWavesArrive) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_barrier_mem");
  amdgpu::L2Cache l2("ds_gws_barrier_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Explicitly initialize the resource for two participants (count = 1).
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *wf0, /*count=*/1);

  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// After a barrier releases, the counter re-arms so the same resource serves a
// second rendezvous phase.
TEST_P(DsGwsTest, BarrierReusesCounterAcrossPhases) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_reuse_mem");
  amdgpu::L2Cache l2("ds_gws_reuse_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Explicitly init the two-participant resource (count = 1) rather than relying
  // on first-arrival self-init; cooperative kernels are expected to init first
  // (the self-init default is covered on its own by BarrierSelfInitsWhenUninit).
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *wf0, /*count=*/1);

  for (int phase = 0; phase < 2; ++phase) {
    run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
    EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT) << "phase " << phase;
    run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
    EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING) << "phase " << phase;
    EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING) << "phase " << phase;
  }

  wf0->halt();
  wf1->halt();
}

// The releasing arrival reloads the counter from its own value, so consecutive
// phases may use different participant counts. init(2) sizes the first phase for
// three arrivals; the releasing barrier(1) then resizes the next phase to two.
// (A fixed reload would keep requiring three arrivals and hang the second phase.)
TEST_P(DsGwsTest, BarrierChangingPhaseSizeReusesArrivalCount) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_resize_mem");
  amdgpu::L2Cache l2("ds_gws_resize_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf2 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  ASSERT_NE(wf2, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/3);
  for (auto *wf : {wf0, wf1, wf2}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Phase 1: three participants (init value 2). Two park, the third releases.
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *wf0, /*count=*/2);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf2, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf2->state(), amdgpu::WfState::RUNNING);

  // Phase 2: the reloaded counter (1) now needs only two arrivals.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
  wf2->halt();
}

// When the participant set (count + 1) exceeds the resident set, the barrier
// cannot be proven deadlock-free, so it falls back to a non-blocking no-op.
TEST_P(DsGwsTest, BarrierFallsBackWhenCountExceedsResident) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_fallback_mem");
  amdgpu::L2Cache l2("ds_gws_fallback_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  wf0->set_exec(0x1);
  wf0->set_m0(0);

  // participants (count + 1 = 3) > resident (2): structural no-op, keep running.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/2);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// Boundary: a near-UINT32_MAX count must take the oversized-count fallback, not
// overflow (count + 1) to zero and park. The residency gate is written to avoid
// that wrap, so this huge count is a structural no-op with two resident waves.
TEST_P(DsGwsTest, BarrierHugeCountDoesNotOverflowResidencyGate) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_overflow_mem");
  amdgpu::L2Cache l2("ds_gws_overflow_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  wf0->set_exec(0x1);
  wf0->set_m0(0);

  // count = 0xffffffff on a fresh resource: outstanding + 1 would wrap to 0 under
  // a naive gate and wrongly park; the overflow-safe gate takes the fallback.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/0xffffffffu);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// ds_gws_sema_p parks when no credit is available; ds_gws_sema_v then releases a
// parked waiter within the dispatch.
TEST_P(DsGwsTest, SemaphorePParksAndVReleases) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_sema_mem");
  amdgpu::L2Cache l2("ds_gws_sema_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  run_gws(*cu, *decoder, arch, GwsOp::kSemaV, *wf1, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// ds_gws_sema_br releases up to @count parked waiters in one shot.
TEST_P(DsGwsTest, SemaphoreBulkReleaseWakesWaiters) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_br_mem");
  amdgpu::L2Cache l2("ds_gws_br_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf2 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  ASSERT_NE(wf2, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/3);
  for (auto *wf : {wf0, wf1, wf2}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf1, /*count=*/0);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  // One SEMA_BR releases both parked waiters.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaBr, *wf2, /*count=*/2);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  for (auto *wf : {wf0, wf1, wf2})
    wf->halt();
}

// ds_gws_sema_release_all wakes every parked waiter and clears the credits.
TEST_P(DsGwsTest, SemaphoreReleaseAllWakesWaiters) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_relall_mem");
  amdgpu::L2Cache l2("ds_gws_relall_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf2 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  ASSERT_NE(wf2, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/3);
  for (auto *wf : {wf0, wf1, wf2}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*count=*/0);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf1, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  run_gws(*cu, *decoder, arch, GwsOp::kReleaseAll, *wf2, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  for (auto *wf : {wf0, wf1, wf2})
    wf->halt();
}

// The rid is ((M0[21:16] + offset0) & 0x3f). A P on rid 0 (M0=0, offset0=0) is
// released by a V whose (M0[21:16]=63, offset0=1) sum wraps back to rid 0.
TEST_P(DsGwsTest, SemaphoreRidSumWrapsToSameResource) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_wrap_mem");
  amdgpu::L2Cache l2("ds_gws_wrap_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  wf0->set_exec(0x1);
  wf1->set_exec(0x1);

  wf0->set_m0(0); // rid = (0 + 0) & 0x3f = 0
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*count=*/0, /*offset0=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf1->set_m0(63u << 16); // rid = (63 + 1) & 0x3f = 0
  run_gws(*cu, *decoder, arch, GwsOp::kSemaV, *wf1, /*count=*/0, /*offset0=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// GWS resources are shared per process, so a P parked by one workgroup is
// released by a V issued from another workgroup sharing the resource -- the
// hardware rendezvous the workgroup-private model could not express.
TEST_P(DsGwsTest, CrossWorkgroupSemaphorePWakesOnCrossWgV) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xwg_mem");
  amdgpu::L2Cache l2("ds_gws_xwg_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  wf0->set_exec(0x1);
  wf0->set_m0(0);
  wf1->set_exec(0x1);
  wf1->set_m0(0);

  // wf0 (wg0) waits with no credit and parks on the shared resource.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 (wg1) signals the same shared process resource and wakes wf0.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaV, *wf1, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// INIT seeds a semaphore credit on the shared process resource: with a second
// resident wave, the first P consumes that credit and keeps running, while the
// second P finds no credit and parks. Guards the credit seed in GwsDevice::init.
TEST_P(DsGwsTest, InitSeedsCreditFirstPConsumesSecondParks) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_initcredit_mem");
  amdgpu::L2Cache l2("ds_gws_initcredit_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Seed exactly one credit.
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  // First P consumes the seeded credit and stays running.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  // Second P finds no credit and parks.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf1, /*count=*/0);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  wf0->halt();
  wf1->halt();
}

// Polling regression: a sibling that merely spins (stays RUNNING, never signals)
// must not prevent a cross-workgroup V from waking a parked P. The wake is
// event-driven and does not depend on the quiescence backstop, which a running
// sibling would otherwise block.
TEST_P(DsGwsTest, PollingSiblingDoesNotBlockCrossWgSemaphoreWake) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_poll_mem");
  amdgpu::L2Cache l2("ds_gws_poll_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *consumer = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *sibling = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *producer = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(sibling, nullptr);
  ASSERT_NE(producer, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  for (auto *wf : {consumer, sibling, producer}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Consumer (wg0) parks on P; its sibling stays RUNNING (a spin/poll loop).
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *consumer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(sibling->state(), amdgpu::WfState::RUNNING);

  // Producer (wg1) signals: the consumer wakes even though the sibling is still
  // running, so the running poll loop cannot deadlock the rendezvous.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(sibling->state(), amdgpu::WfState::RUNNING);

  consumer->halt();
  sibling->halt();
  producer->halt();
}

// GWS barriers share the process resource too: two single-wave workgroups that each
// arrive at a barrier programmed for two participants rendezvous across the
// workgroup boundary (the early arrival parks, the second releases it).
TEST_P(DsGwsTest, CrossWorkgroupBarrierReleasesBothParticipants) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xwgbar_mem");
  amdgpu::L2Cache l2("ds_gws_xwgbar_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  wf0->set_exec(0x1);
  wf0->set_m0(0);
  wf1->set_exec(0x1);
  wf1->set_m0(0);

  // Explicitly init the two-participant resource (count = 1) rather than relying
  // on first-arrival self-init (see BarrierSelfInitsWhenUninit for that policy).
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *wf0, /*count=*/1);

  // Two participants -> programmed value 1. wf0 (wg0) arrives first and parks.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 (wg1) is the releasing arrival and wakes the cross-workgroup peer.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// Safety net: if co-resident waves park at a GWS barrier but the remaining wave
// retires without ever arriving, update_wf_states() (driven by step()) releases
// the parked waves so the model never deadlocks. s_endpgm at the wave PC lets
// the released waves retire.
TEST_P(DsGwsTest, DeadlockEscapeReleasesParkedBarrierWaves) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_escape_mem");
  amdgpu::L2Cache l2("ds_gws_escape_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf2 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  ASSERT_NE(wf2, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/3);
  for (auto *wf : {wf0, wf1, wf2}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Three participants (count = 2); two arrive and park, the third retires.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/2);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/2);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);
  wf2->halt();

  // The escape scan must unblock the two parked waves, which then run their
  // (per-ISA) s_endpgm to completion: the dispatch actually finishes, with no
  // leftover GWS wait obligation on either wave.
  for (int i = 0; i < 8 && (!wf0->is_halted() || !wf1->is_halted()); ++i)
    cu->step();
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf1->is_halted());
  EXPECT_TRUE(wf0->wait_counters().empty());
  EXPECT_TRUE(wf1->wait_counters().empty());
}

// Mixed deadlock: one wave parks at a GWS barrier while its sibling stalls at an
// s_barrier. Neither can signal the other, so the escape scan (which treats an
// s_barrier stall as a blocked state) releases the parked GWS wave.
TEST_P(DsGwsTest, MixedBarrierSBarrierEscapeReleasesGwsWave) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_mixed_mem");
  amdgpu::L2Cache l2("ds_gws_mixed_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // wf0 parks at a two-participant GWS barrier.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 executes a plain workgroup s_barrier and stalls (wf0 never arrives).
  const auto words = build_s_barrier(arch);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf1).succeeded());
  ASSERT_EQ(wf1->state(), amdgpu::WfState::BARRIER);

  // The escape scan releases the GWS-parked wave even though its sibling is
  // stalled at an s_barrier rather than a GWS op; the released wave then runs its
  // (per-ISA) s_endpgm to completion with no leftover GWS wait obligation.
  for (int i = 0; i < 8 && !wf0->is_halted(); ++i)
    cu->step();
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf0->wait_counters().empty());
}

// The deadlock backstop must not leave a half-finished phase behind. When it
// abandons a barrier rendezvous it invalidates the resource (counter/credits
// cleared, disarmed), so a later dispatch reusing the same rid starts fresh.
// Regression: without the reset, the abandoned phase's zero counter would make
// the first of the next rendezvous's two arrivals release immediately instead of
// parking -- the escape silently handing its unfinished phase to a later grid.
TEST_P(DsGwsTest, EscapeInvalidatesAbandonedResourceForLaterDispatch) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_escinval_mem");
  amdgpu::L2Cache l2("ds_gws_escinval_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto decoder = Decoder::create(arch);

  // Dispatch 0: three participants (count = 2). Two arrive and park; the third
  // retires without arriving, so the backstop abandons the rendezvous with the
  // resource's counter stuck at 0.
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf2 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  ASSERT_NE(wf2, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/3);
  for (auto *wf : {wf0, wf1, wf2}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/2);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/2);
  ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);
  wf2->halt();
  for (int i = 0; i < 8 && (!wf0->is_halted() || !wf1->is_halted()); ++i)
    cu->step();
  ASSERT_TRUE(wf0->is_halted());
  ASSERT_TRUE(wf1->is_halted());

  // Dispatch 1: a fresh two-participant barrier(1) on the SAME rid. Because the
  // backstop invalidated the resource, the first arrival must park; the second
  // then releases it -- the normal two-phase rendezvous, not an instant release.
  auto *wf3 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf4 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf3, nullptr);
  ASSERT_NE(wf4, nullptr);
  wf3->set_dispatch_id(1);
  wf4->set_dispatch_id(1);
  cu->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf3, wf4}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf3, /*count=*/1);
  EXPECT_EQ(wf3->state(), amdgpu::WfState::GWS_WAIT)
      << "first arrival slipped through the abandoned phase's stale counter";
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf4, /*count=*/1);
  EXPECT_EQ(wf3->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf4->state(), amdgpu::WfState::RUNNING);

  wf3->halt();
  wf4->halt();
}

// GWS resources are device-global: the command processor scatters a dispatch's
// workgroups across CUs, so a P parked on one CU must be released by a V issued
// from a wave on a *different* CU. Because no CU may mutate a peer's wavefront
// state, the cross-CU wake is published to the shared store and applied by the
// waiter's own CU the next time it steps (update_wf_states runs the GWS poll).
// Both CUs share one GwsDevice (declared so it outlives them; CUs unregister on
// destruction).
TEST_P(DsGwsTest, CrossComputeUnitSemaphorePWakesOnCrossCuV) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xcu_mem");
  amdgpu::L2Cache l2("ds_gws_xcu_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto decoder = Decoder::create(arch);
  auto *consumer = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *producer = cu1->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(producer, nullptr);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  for (auto *wf : {consumer, producer}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Consumer parks on CU0's wave slot with no credit.
  run_gws(*cu0, *decoder, arch, GwsOp::kSemaP, *consumer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);

  // Producer signals from CU1: the credit lands in the shared store, and CU0
  // applies the cross-CU wake on its next step (consumer's PC is s_endpgm, so it
  // retires once released).
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);
  for (int i = 0; i < 4 && consumer->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(consumer->state(), amdgpu::WfState::GWS_WAIT);

  producer->halt();
}

// A GWS barrier programmed for two participants whose workgroups land on
// different CUs rendezvouses across the CU boundary: the early arrival on CU0
// parks and the releasing arrival on CU1 bumps the release generation. The peer
// parked on CU0 wakes when CU0 next steps (cross-CU wake is poll-applied). The
// residency bound is summed across both CUs (one resident wave each -> two).
TEST_P(DsGwsTest, CrossComputeUnitBarrierReleasesBothParticipants) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xcubar_mem");
  amdgpu::L2Cache l2("ds_gws_xcubar_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu1->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  wf0->set_exec(0x1);
  wf0->set_m0(0);
  wf1->set_exec(0x1);
  wf1->set_m0(0);

  // Explicitly init the two-participant resource (count = 1) rather than relying
  // on first-arrival self-init (see BarrierSelfInitsWhenUninit for that policy).
  run_gws(*cu0, *decoder, arch, GwsOp::kInit, *wf0, /*count=*/1);

  // Two participants -> programmed value 1. wf0 (CU0) arrives first and parks.
  run_gws(*cu0, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 (CU1) is the releasing arrival: it stays running immediately, and the
  // peer parked on CU0 wakes when CU0 next steps.
  run_gws(*cu1, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  for (int i = 0; i < 4 && wf0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf1->halt();
}

// Quiescence must not treat a peer CU's s_barrier stall as proof a wave cannot
// signal: scalar barriers resolve only on their own CU's next step, so a peer
// workgroup that has fully arrived is still marked BARRIER until then and can go
// on to signal. Here CU0 parks a P while CU1's two waves sit at a (satisfied)
// s_barrier before one of them issues the V. Stepping CU0 must NOT escape the P
// while CU1's barrier can still resolve; the P proceeds only once the V lands a
// credit. Exercises the exact step interleaving (CU0 park, CU1 step, CU0 steps).
TEST_P(DsGwsTest, CrossCuScalarBarrierBlocksPrematureEscape) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xcusbar_mem");
  amdgpu::L2Cache l2("ds_gws_xcusbar_l2");
  // Consumer retires at its own PC once released; CU1's two waves run real
  // streams: [s_barrier; ds_gws_sema_v; s_endpgm] and [s_barrier; s_endpgm].
  constexpr uint64_t kPcConsumer = 0x200000;
  constexpr uint64_t kPcA = 0x201000;
  constexpr uint64_t kPcB = 0x202000;
  const uint32_t endpgm = gws_s_endpgm_word(arch);
  const auto sbar = build_s_barrier(arch);
  const auto semav = build_gws(arch, gws_opcode(arch, GwsOp::kSemaV), /*gds=*/1, /*addr=*/0);
  mem.write32(kPcConsumer, endpgm);
  mem.write32(kPcA, sbar[0]);
  mem.write32(kPcA + 4, semav[0]); // GWS ops are two words (8 bytes).
  mem.write32(kPcA + 8, semav[1]);
  mem.write32(kPcA + 12, endpgm);
  mem.write32(kPcB, sbar[0]);
  mem.write32(kPcB + 4, endpgm);

  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto decoder = Decoder::create(arch);

  // CU0: a single consumer parks on a P (same process/dispatch as CU1's waves).
  auto *consumer = cu0->dispatch_wf(/*wg_id=*/0, kPcConsumer, 102, 16);
  ASSERT_NE(consumer, nullptr);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  consumer->set_exec(0x1);
  consumer->set_m0(0);

  // CU1: a two-wave workgroup that will rendezvous at an s_barrier, after which
  // wave A signals the semaphore.
  auto *wfa = cu1->dispatch_wf(/*wg_id=*/1, kPcA, 102, 16);
  auto *wfb = cu1->dispatch_wf(/*wg_id=*/1, kPcB, 102, 16);
  ASSERT_NE(wfa, nullptr);
  ASSERT_NE(wfb, nullptr);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/2);
  for (auto *wf : {wfa, wfb}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  run_gws(*cu0, *decoder, arch, GwsOp::kSemaP, *consumer, /*count=*/0);
  ASSERT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);

  // One CU1 step drives both waves into the s_barrier (fully arrived, but scalar-
  // barrier resolution is deferred to CU1's next step). CU1 publishes its
  // signalable snapshot here, before the waves stall.
  cu1->step();
  ASSERT_EQ(wfa->state(), amdgpu::WfState::BARRIER);
  ASSERT_EQ(wfb->state(), amdgpu::WfState::BARRIER);

  // Now step CU0 repeatedly. The P must stay parked: CU1's barrier is ready to
  // resolve and wave A can still signal, so this is not a genuine deadlock.
  for (int i = 0; i < 6; ++i)
    cu0->step();
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT)
      << "parked P was escaped while a peer CU's s_barrier could still resolve "
         "and signal";

  // Let CU1 resolve the barrier and run wave A's V; the credit then releases the
  // consumer on CU0's next step -- via a credit, never a premature escape.
  for (int i = 0; i < 4 && consumer->state() == amdgpu::WfState::GWS_WAIT; ++i) {
    cu1->step();
    cu0->step();
  }
  EXPECT_NE(consumer->state(), amdgpu::WfState::GWS_WAIT);

  if (!wfa->is_halted())
    wfa->halt();
  if (!wfb->is_halted())
    wfb->halt();
}

// Quiescence is process-scoped: a runnable producer in another dispatch of the
// same process must keep a parked P from being escape-released, even though its
// own dispatch looks idle on the consumer's CU. Only once the producer signals
// does the P proceed -- with a credit, never via a premature deadlock escape.
// (Regression for the escape backstop firing on dispatch, not process, scope.)
TEST_P(DsGwsTest, OverlappingDispatchQuiescenceIsProcessScoped) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_overlap_mem");
  amdgpu::L2Cache l2("ds_gws_overlap_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto decoder = Decoder::create(arch);

  // Dispatch 0's consumer parks on CU0.
  auto *consumer = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(consumer, nullptr);
  consumer->set_dispatch_id(0);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  consumer->set_exec(0x1);
  consumer->set_m0(0);

  // Dispatch 1's producer (same process) is resident and runnable on CU1 but has
  // not signaled yet; it is deliberately never stepped so it stays runnable.
  auto *producer = cu1->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(producer, nullptr);
  producer->set_dispatch_id(1);
  cu1->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/0, /*wf_count=*/1);
  producer->set_exec(0x1);
  producer->set_m0(0);

  run_gws(*cu0, *decoder, arch, GwsOp::kSemaP, *consumer, /*count=*/0);
  ASSERT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);

  // Stepping CU0 must NOT escape-release the consumer while the same-process
  // producer on CU1 is still runnable (it could still signal).
  for (int i = 0; i < 8; ++i)
    cu0->step();
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT)
      << "parked P was escaped while a same-process producer was still runnable";

  // The producer finally signals; CU0 then releases the consumer with a credit.
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);
  for (int i = 0; i < 4 && consumer->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(consumer->state(), amdgpu::WfState::GWS_WAIT);

  producer->halt();
}

// Snapshot-versioning regression (workgroup turnover / producer-alongside). A CU
// that publishes a zero-runnable quiescence summary while its only wave is parked
// must not have that stale summary trusted once new runnable work (a producer)
// becomes resident on it. The old park-epoch advanced only on a 0 -> positive
// parked transition, which workgroup turnover does not trigger, so the backstop
// would read CU1's pre-producer zero summary and escape CU0's P before the
// producer ever signaled -- retiring the P without consuming a credit. add_resident
// now advances the snapshot version, so CU1's summary is stale until it republishes
// with the producer counted, and the P proceeds only via a real V.
TEST_P(DsGwsTest, StaleQuiescenceSnapshotInvalidatedByNewResidentWork) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_snapver_mem");
  amdgpu::L2Cache l2("ds_gws_snapver_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto decoder = Decoder::create(arch);

  // Two consumers of the same process (dispatch 0) park on the same semaphore: C0
  // on CU0 and W1 on CU1. Each is the sole wave of its CU, so a step publishes a
  // zero-runnable summary for that CU.
  auto *c0 = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *w1 = cu1->dispatch_wf(/*wg_id=*/1, kPc, 102, 16);
  ASSERT_NE(c0, nullptr);
  ASSERT_NE(w1, nullptr);
  c0->set_dispatch_id(0);
  w1->set_dispatch_id(0);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  for (auto *wf : {c0, w1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  run_gws(*cu0, *decoder, arch, GwsOp::kSemaP, *c0, /*count=*/0);
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaP, *w1, /*count=*/0);
  ASSERT_EQ(c0->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(w1->state(), amdgpu::WfState::GWS_WAIT);

  // CU1 publishes its zero-runnable summary (its only wave, W1, is parked).
  cu1->step();

  // A runnable producer of the same process becomes resident on CU1 *after* that
  // summary was published. It is never stepped, so it stays RUNNING and able to
  // signal. begin_workgroup -> add_resident must advance the snapshot version here.
  auto *producer = cu1->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(producer, nullptr);
  producer->set_dispatch_id(1);
  cu1->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/0, /*wf_count=*/1);
  producer->set_exec(0x1);
  producer->set_m0(0);

  // Stepping CU0 must NOT escape C0: CU1's published summary predates the producer
  // and is now a stale version, so the process is not provably quiescent. CU1 is
  // never re-stepped here, so its summary stays stale and escape stays blocked.
  for (int i = 0; i < 8; ++i)
    cu0->step();
  EXPECT_EQ(c0->state(), amdgpu::WfState::GWS_WAIT)
      << "parked P was escaped via a stale pre-producer quiescence snapshot";

  // The producer signals twice: the first V is consumed by W1 on CU1 (same-CU
  // immediate poll), the second persists for C0 to consume on CU0's next step --
  // a real credit, never a premature escape.
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);
  for (int i = 0; i < 4 && c0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(c0->state(), amdgpu::WfState::GWS_WAIT);

  producer->halt();
  if (!w1->is_halted())
    w1->halt();
}

// Fast path: a workload that never issues a GWS op must not drive the shared
// store's locked bookkeeping each step. The lock-free parked-wave gate stays at
// zero, so a plain kernel runs to completion with the store reporting nothing
// parked at every step.
TEST_P(DsGwsTest, NoGwsWorkloadKeepsStoreIdle) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_nogws_mem");
  amdgpu::L2Cache l2("ds_gws_nogws_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  wf->set_exec(0x1);

  EXPECT_FALSE(cu->gws_device().any_parked());
  for (int i = 0; i < 8 && wf->state() != amdgpu::WfState::HALTED; ++i) {
    cu->step();
    EXPECT_FALSE(cu->gws_device().any_parked());
  }
  EXPECT_EQ(wf->state(), amdgpu::WfState::HALTED);
}

// Thread-safety: two CUs sharing one store, each stepped on its own thread while
// issuing GWS semaphore P/V, must not race (run under TSan) and must make
// forward progress. No CU ever touches a peer's wavefront state; signals cross
// via the store's credits/generation and are applied by the waiter's own CU.
TEST_P(DsGwsTest, ConcurrentCrossCuSteppingIsRaceFree) {
  const auto arch = GetParam();
  constexpr uint64_t kPc = 0x200000;
  amdgpu::GpuMemory mem0("ds_gws_tsan_mem0");
  amdgpu::GpuMemory mem1("ds_gws_tsan_mem1");
  amdgpu::L2Cache l20("ds_gws_tsan_l20");
  amdgpu::L2Cache l21("ds_gws_tsan_l21");
  mem0.write32(kPc, gws_s_endpgm_word(arch));
  mem1.write32(kPc, gws_s_endpgm_word(arch));
  // Each CU owns its memory/L2; the only shared mutable object is the store, so
  // any race TSan reports is a GWS store race.
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem0, l20, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem1, l21, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);

  constexpr int kIters = 48;
  std::atomic<int> consumed{0};
  std::atomic<int> issue_failures{0};

  // Thread-local, gtest-macro-free GWS issue (run_gws uses ASSERT_*, which is not
  // safe to call from multiple threads at once -- it records into shared gtest
  // state). Failures are counted and asserted on the main thread after join.
  auto issue_gws = [&](amdgpu::ComputeUnitCore &cu, Decoder &decoder, GwsOp op,
                       amdgpu::Wavefront &wf, uint32_t count) {
    const bool has_addr = gws_has_addr(op);
    if (has_addr)
      cu.write_vgpr(wf.vgpr_alloc().base + kAddrVgpr, /*lane=*/0, count);
    const auto words =
        build_gws(arch, gws_opcode(arch, op), /*gds=*/1, has_addr ? kAddrVgpr : 0, /*offset0=*/0);
    std::unique_ptr<Instruction> inst(decode_valid(decoder, words.data()));
    if (!inst || !cu.execute_instruction(inst.get(), wf).succeeded())
      issue_failures.fetch_add(1, std::memory_order_relaxed);
  };

  auto producer_fn = [&] {
    auto decoder = Decoder::create(arch);
    for (int i = 0; i < kIters; ++i) {
      auto *p = cu1->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
      if (!p) {
        std::this_thread::yield();
        --i;
        continue;
      }
      // Tag the producer with its dispatch id (matching begin_workgroup below) so
      // halt/retire reconciles dispatch-1 residency against the same key the store
      // was told to track, rather than leaving a phantom resident wave behind.
      p->set_dispatch_id(1);
      cu1->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/0, /*wf_count=*/1);
      p->set_exec(0x1);
      p->set_m0(0);
      issue_gws(*cu1, *decoder, GwsOp::kSemaV, *p, /*count=*/0);
      p->halt();
    }
  };
  auto consumer_fn = [&] {
    auto decoder = Decoder::create(arch);
    for (int i = 0; i < kIters; ++i) {
      auto *c = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
      if (!c) {
        std::this_thread::yield();
        --i;
        continue;
      }
      cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
      c->set_exec(0x1);
      c->set_m0(0);
      issue_gws(*cu0, *decoder, GwsOp::kSemaP, *c, /*count=*/0);
      for (int s = 0; s < 8 && c->state() == amdgpu::WfState::GWS_WAIT; ++s)
        cu0->step();
      if (c->state() != amdgpu::WfState::GWS_WAIT)
        consumed.fetch_add(1, std::memory_order_relaxed);
      if (!c->is_halted())
        c->halt();
    }
  };

  std::thread producer(producer_fn);
  std::thread consumer(consumer_fn);
  producer.join();
  consumer.join();

  EXPECT_EQ(issue_failures.load(), 0);
  // Primary purpose is race detection; also sanity-check forward progress (a
  // credit from a V or the quiescence backstop releases every parked consumer).
  EXPECT_GT(consumed.load(), 0);
}

// Production wiring (direct Xcd constructor): every CU of an XCD shares that
// XCD's single device-global GWS store, so cross-CU GWS works for production
// dispatches. Regression for CUs keeping their private default store.
TEST_P(DsGwsTest, XcdConstructorSharesGwsStoreAcrossCus) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xcd_mem");
  amdgpu::ComputeUnitCore::Config cu_cfg{};
  cu_cfg.arch = arch;
  cu_cfg.num_wf_slots = 4;
  cu_cfg.sgprs_per_wf = 102;
  cu_cfg.vgprs_per_wf = 16;
  cu_cfg.lds_size_kb = 64;
  amdgpu::ShaderEngine::Config se_cfg{};
  se_cfg.num_compute_units = 2;
  se_cfg.compute_unit = cu_cfg;
  se_cfg.cus_per_shader_array = 2;
  amdgpu::Xcd::Config xcd_cfg{};
  xcd_cfg.num_shader_engines = 1;
  xcd_cfg.shader_engine = se_cfg;
  amdgpu::Xcd xcd("ds_gws_xcd", xcd_cfg, arch, &mem);

  const amdgpu::GwsDevice *store = xcd.gws_device().get();
  ASSERT_NE(store, nullptr);
  uint32_t cu_count = 0;
  for (uint32_t s = 0; s < xcd.num_shader_engines(); ++s) {
    auto *se = xcd.shader_engine(s);
    for (uint32_t c = 0; c < se->num_compute_units(); ++c) {
      EXPECT_EQ(&se->compute_unit(c)->gws_device(), store)
          << "se" << s << " cu" << c << " did not share the XCD GWS store";
      ++cu_count;
    }
  }
  EXPECT_EQ(cu_count, 2u);
}

// Custom topologies can attach CUs directly below the XCD (the config loader
// registers them with the command processor but under no shader engine). Those
// CP-visible CUs must share the XCD's device-global GWS store too -- wire_gws_to_cus
// drives the command processor's CU list, not just shader_engines_. Regression: a
// shader_engines_-only walk left such a CU on its private default store, breaking
// cross-CU wakeups and the quiescence backstop for it.
TEST_P(DsGwsTest, DirectlyAttachedCpCuSharesGwsStore) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_directcu_mem");
  amdgpu::L2Cache l2("ds_gws_directcu_l2");
  amdgpu::ComputeUnitCore::Config cu_cfg{};
  cu_cfg.arch = arch;
  cu_cfg.num_wf_slots = 4;
  cu_cfg.sgprs_per_wf = 102;
  cu_cfg.vgprs_per_wf = 16;
  cu_cfg.lds_size_kb = 64;
  amdgpu::ShaderEngine::Config se_cfg{};
  se_cfg.num_compute_units = 1;
  se_cfg.compute_unit = cu_cfg;
  se_cfg.cus_per_shader_array = 1;
  amdgpu::Xcd::Config xcd_cfg{};
  xcd_cfg.num_shader_engines = 1;
  xcd_cfg.shader_engine = se_cfg;
  amdgpu::Xcd xcd("ds_gws_directcu_xcd", xcd_cfg, arch, &mem);

  const amdgpu::GwsDevice *store = xcd.gws_device().get();
  ASSERT_NE(store, nullptr);

  // A CU attached straight to the command processor, with no shader engine -- as a
  // custom topology would produce.
  auto direct_cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  ASSERT_NE(&direct_cu->gws_device(), store); // Starts on its private default store.
  xcd.command_processor()->add_compute_unit(direct_cu.get());
  xcd.wire_gws_to_cus();

  EXPECT_EQ(&direct_cu->gws_device(), store)
      << "a directly-attached CP-visible CU did not share the XCD GWS store";
}

// Device-global GWS state is not torn down when a workgroup retires: a V leaves a
// credit, the signaling workgroup fully retires, and a later workgroup's P still
// finds the persisted credit. The dispatch-keyed per-CU model lost it here.
TEST_P(DsGwsTest, WorkgroupTurnoverSemaphoreCreditPersists) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_turnover_mem");
  amdgpu::L2Cache l2("ds_gws_turnover_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *producer = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(producer, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  producer->set_exec(0x1);
  producer->set_m0(0);

  // Producer signals (no waiter), leaving one credit, then its workgroup retires.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);
  producer->halt();

  // A later workgroup's P consumes the credit that survived the retirement.
  auto *consumer = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(consumer, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  consumer->set_exec(0x1);
  consumer->set_m0(0);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *consumer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::RUNNING);

  consumer->halt();
}

// A separate init kernel (one dispatch) seeds a semaphore credit that a later
// dispatch's P consumes -- the CLR RunGwsInit / KFDGWSTest.Semaphore sequence.
// GWS resources are keyed by process and persist across dispatches, so the seed
// survives the init dispatch's retirement.
TEST_P(DsGwsTest, SeparateInitDispatchSeedsLaterDispatch) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_initdispatch_mem");
  amdgpu::L2Cache l2("ds_gws_initdispatch_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);

  // Init dispatch (dispatch_id 0) seeds one credit, then fully retires.
  auto *initer = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(initer, nullptr);
  initer->set_dispatch_id(0);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  initer->set_exec(0x1);
  initer->set_m0(0);
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *initer, /*count=*/1);
  initer->halt();

  // Main dispatch (dispatch_id 1) consumes the credit seeded by the init dispatch.
  auto *user = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(user, nullptr);
  user->set_dispatch_id(1);
  cu->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/0, /*wf_count=*/1);
  user->set_exec(0x1);
  user->set_m0(0);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *user, /*count=*/0);
  EXPECT_EQ(user->state(), amdgpu::WfState::RUNNING);

  user->halt();
}

// On CDNA2/CDNA3 the DS acc bit selects the AGPR bank for the count operand.
// Writing a parking count to the AGPR slot and a non-parking count to the VGPR
// slot proves acc routes the read to the AGPR. Other arches have no acc field.
TEST_P(DsGwsTest, AccBitSelectsAgprCountOperand) {
  const auto arch = GetParam();
  if (arch != ROCJITSU_CODE_ARCH_CDNA2 && arch != ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "DS acc bit only exists on CDNA2/3";
  amdgpu::GpuMemory mem("ds_gws_acc_mem");
  amdgpu::L2Cache l2("ds_gws_acc_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  const uint16_t op = gws_opcode(arch, GwsOp::kBarrier);
  // wf0: acc=1 reads the AGPR slot (count 1 -> participants 2 <= resident 2 ->
  // parks). The VGPR slot holds a non-parking value to prove it is ignored.
  cu->write_vgpr(wf0->vgpr_alloc().base + kAddrVgpr, /*lane=*/0, /*vgpr=*/5);
  cu->write_vgpr(wf0->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET + kAddrVgpr, /*lane=*/0,
                 /*agpr=*/1);
  const auto acc_words = build_gws_acc(arch, op, /*acc=*/1, kAddrVgpr);
  std::unique_ptr<Instruction> acc_inst(decode_valid(*decoder, acc_words.data()));
  ASSERT_NE(acc_inst, nullptr);
  // Decoded metadata must name the AGPR, not just the execute path: acc=1 makes
  // the ADDR source operand resolve to acc<kAddrVgpr> for disassembly and
  // register-analysis consumers.
  const Operand *acc_addr = acc_inst->src_operand(0);
  ASSERT_NE(acc_addr, nullptr);
  const auto acc_ref = acc_addr->to_register_ref();
  ASSERT_TRUE(acc_ref.has_value());
  EXPECT_EQ(*acc_ref, (RegisterRef{RegClass::ACC_VGPR, kAddrVgpr, 1}));
  EXPECT_EQ(acc_addr->name(), "acc" + std::to_string(kAddrVgpr));
  EXPECT_NE(acc_inst->disassemble().find("acc" + std::to_string(kAddrVgpr)), std::string::npos)
      << acc_inst->disassemble();
  ASSERT_TRUE(cu->execute_instruction(acc_inst.get(), *wf0).succeeded());
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1: acc=0 on a *fresh* resource (rid 1 via offset0=1), so its arrival is not
  // short-circuited by the already-released rid-0 counter. Both banks are seeded:
  // the VGPR with a non-parking count (5 -> participants 6 > resident 2 ->
  // structural no-op, keeps running) and the AGPR with a parking count (1). A
  // correct acc=0 read picks the VGPR and stays RUNNING; a mutation that read the
  // AGPR would see count 1 and park, so the RUNNING assertion detects wrong-bank
  // selection.
  cu->write_vgpr(wf1->vgpr_alloc().base + kAddrVgpr, /*lane=*/0, /*vgpr=*/5);
  cu->write_vgpr(wf1->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET + kAddrVgpr, /*lane=*/0,
                 /*agpr=*/1);
  const auto vgpr_words = build_gws_acc(arch, op, /*acc=*/0, kAddrVgpr, /*offset0=*/1);
  std::unique_ptr<Instruction> vgpr_inst(decode_valid(*decoder, vgpr_words.data()));
  ASSERT_NE(vgpr_inst, nullptr);
  // acc=0 keeps the ADDR source operand on the VGPR bank.
  const Operand *vgpr_addr = vgpr_inst->src_operand(0);
  ASSERT_NE(vgpr_addr, nullptr);
  const auto vgpr_ref = vgpr_addr->to_register_ref();
  ASSERT_TRUE(vgpr_ref.has_value());
  EXPECT_EQ(*vgpr_ref, (RegisterRef{RegClass::VGPR, kAddrVgpr, 1}));
  EXPECT_EQ(vgpr_addr->name(), "v" + std::to_string(kAddrVgpr));
  EXPECT_NE(vgpr_inst->disassemble().find("v" + std::to_string(kAddrVgpr)), std::string::npos)
      << vgpr_inst->disassemble();
  ASSERT_TRUE(cu->execute_instruction(vgpr_inst.get(), *wf1).succeeded());
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// Emulator policy (GwsDevice first-arrival self-init): a GWS barrier arriving at a
// resource never seeded by an explicit init seeds the counter from its own operand
// and still rendezvouses, rather than stalling. Hardware (RDNA3/3.5 Table 63)
// requires a prior init, so cooperative kernels are expected to init first (the
// rendezvous tests above do); this case deliberately omits the init to pin the
// self-init default that lets a test -- or a phase recovering after the deadlock
// backstop invalidated its resource -- proceed.
TEST_P(DsGwsTest, BarrierSelfInitsWhenUninitialized) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_selfinit_mem");
  amdgpu::L2Cache l2("ds_gws_selfinit_l2");
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // No init: the first arrival seeds the two-participant counter from its own
  // count (1) and parks; the co-resident second arrival releases it.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// A V leaves a credit for a P already parked (possibly on another CU), delivered
// by that waiter's own CU on its next poll. A fresh P must not steal that credit:
// the store reserves one credit per already-parked waiter (Resource::sema_waiters),
// so a second P issued on the signaling CU after the V but before the first waiter
// polls parks too, and the credit goes to the earlier waiter. Regression: without
// the reservation the later P would consume the V's credit and the earlier waiter
// could stall. Two CUs share one store; cross-CU wakes are poll-applied.
TEST_P(DsGwsTest, SemaphoreVReservedForEarlierParkedWaiter) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_reserve_mem");
  amdgpu::L2Cache l2("ds_gws_reserve_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto decoder = Decoder::create(arch);
  auto *waiter0 = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  // The producer sits at the s_endpgm PC so a single poll step retires it cleanly.
  auto *producer = cu1->dispatch_wf(/*wg_id=*/1, kPc, 102, 16);
  auto *waiter1 = cu1->dispatch_wf(/*wg_id=*/2, kPc, 102, 16);
  ASSERT_NE(waiter0, nullptr);
  ASSERT_NE(producer, nullptr);
  ASSERT_NE(waiter1, nullptr);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/2, /*wf_count=*/1);
  for (auto *wf : {waiter0, producer, waiter1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // waiter0 parks on CU0 with no credit -- the earlier waiter.
  run_gws(*cu0, *decoder, arch, GwsOp::kSemaP, *waiter0, /*count=*/0);
  ASSERT_EQ(waiter0->state(), amdgpu::WfState::GWS_WAIT);

  // producer V leaves one credit in the shared store (reserved for waiter0).
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);

  // waiter1's P (same CU as the V) is issued before waiter0 polls. It must NOT
  // steal the reserved credit; it parks instead.
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaP, *waiter1, /*count=*/0);
  EXPECT_EQ(waiter1->state(), amdgpu::WfState::GWS_WAIT)
      << "second P stole the credit owed to the earlier parked waiter";

  // Poll CU1 -- the later waiter's own CU, which also hosts the V -- BEFORE CU0.
  // poll_local_waiters releases a credit only to the oldest queued waiter, so
  // waiter1 must not consume the credit reserved for waiter0 even when its own CU
  // polls first. One step also retires the producer. The process stays non-quiescent
  // during that step (the producer is still runnable), so the deadlock backstop does
  // not fire. Regression: the pre-fix poll woke whichever parked P its CU reached
  // first, so polling CU1 here stole waiter0's credit.
  cu1->step();
  EXPECT_EQ(waiter1->state(), amdgpu::WfState::GWS_WAIT)
      << "later waiter consumed the credit when its own CU polled first";
  EXPECT_EQ(waiter0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_TRUE(producer->is_halted());

  // waiter0's own CU now polls and consumes the reserved credit -- it wakes; waiter1
  // stays parked until a later signal (CU1 is not stepped again, so its wave is not
  // subject to the per-CU backstop here).
  for (int i = 0; i < 4 && waiter0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(waiter0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(waiter1->state(), amdgpu::WfState::GWS_WAIT);
  // waiter1 left parked; cu1's unregister reconciles it on destruction.
}

// rocjitsu admits a grid's workgroups over time (and independently per XCD), so a
// barrier whose remaining participants are not yet resident must still rendezvous
// with them once admitted. When the whole-grid wave count is plumbed through
// begin_workgroup, the barrier bounds its participant set by that grid size rather
// than the currently-resident waves, so an early arrival parks instead of falling
// back to a no-op. Here CU0's lone resident wave arrives first (grid size 2) and
// parks; CU1's wave is admitted later and releases it. Regression: with the
// resident-only bound the first arrival would see participants (2) > resident (1)
// and no-op, never establishing the rendezvous.
TEST_P(DsGwsTest, BarrierRendezvousWithLateAdmittedParticipant) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_late_mem");
  amdgpu::L2Cache l2("ds_gws_late_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  // Only CU0's wave is resident so far, but the grid has two waves.
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1,
                       /*num_named_barriers=*/0, /*process_id=*/0,
                       /*dispatch_total_waves=*/2);
  wf0->set_exec(0x1);
  wf0->set_m0(0);

  // init(1) -> two participants. The lone resident arrival parks because the grid
  // size (2), not the resident count (1), bounds the participant set.
  run_gws(*cu0, *decoder, arch, GwsOp::kInit, *wf0, /*count=*/1);
  run_gws(*cu0, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT)
      << "early arrival fell back to a no-op instead of awaiting the late participant";

  // The second participant is admitted on CU1 and arrives: it releases the peer.
  auto *wf1 = cu1->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(wf1, nullptr);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1,
                       /*num_named_barriers=*/0, /*process_id=*/0,
                       /*dispatch_total_waves=*/2);
  wf1->set_exec(0x1);
  wf1->set_m0(0);
  run_gws(*cu1, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  for (int i = 0; i < 4 && wf0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf1->halt();
}

// Termination guard for the late-admission accounting: a single-CP GWS grid larger
// than the CU's wave-slot capacity must still terminate. The admitted waves park at
// a barrier bounded by the (larger) grid size and hold every slot, so the remaining
// participants can never be admitted. Placement lives in the command processor and
// is greedy -- a placeable producer would already be resident -- so with no fan-out
// shard still pending (nothing published via set_shard_pending), the quiescence
// backstop sees a provably quiescent process and escapes the parked waves instead of
// spinning forever. Regression: an unconditional expected>admitted escape block
// spins here indefinitely (a parked GWS wave counts as runnable), on all seven ISAs.
TEST_P(DsGwsTest, OversubscribedGridEscapesInsteadOfHanging) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_oversub_mem");
  amdgpu::L2Cache l2("ds_gws_oversub_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  // Only two wave slots, but the grid declares four participants.
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/2);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2,
                      /*num_named_barriers=*/0, /*process_id=*/0,
                      /*dispatch_total_waves=*/4);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // Both resident waves arrive at a four-participant barrier (count = 3) and park;
  // the grid-size bound keeps them off the oversized-count fallback. The other two
  // participants can never be admitted -- both slots are full of parked waves -- and
  // no fan-out shard is pending, so the backstop escapes.
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/3);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/3);
  ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  for (int i = 0; i < 16 && (!wf0->is_halted() || !wf1->is_halted()); ++i)
    cu->step();
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf1->is_halted());
  EXPECT_TRUE(wf0->wait_counters().empty());
  EXPECT_TRUE(wf1->wait_counters().empty());
}

// A fan-out shard still sitting in a command processor's inbox (its producer not yet
// placed) must defer the deadlock backstop: the command processor marks the shard
// pending (set_shard_pending true) at accept and clears it (false) once it runs its
// placement pass. Here two waves park at an oversized barrier; while a peer shard is
// pending the backstop must NOT escape them, and once that shard is marked attempted
// the backstop escapes. Regression (kuhar N4): a peer shard that contributed nothing
// yet let the first wave escape before its producer could arrive, on all seven ISAs.
TEST_P(DsGwsTest, PendingFanoutShardDefersThenAllowsEscape) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_pending_mem");
  amdgpu::L2Cache l2("ds_gws_pending_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/2);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2,
                      /*num_named_barriers=*/0, /*process_id=*/0,
                      /*dispatch_total_waves=*/4);
  // A peer command processor (opaque key) accepted a shard of this grid into its
  // inbox but has not yet placed it.
  int peer_cp_key = 0;
  cu->gws_device().set_shard_pending(/*dispatch_id=*/0, /*process_id=*/0,
                                     /*shard_key=*/&peer_cp_key, /*pending=*/true);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/3);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/3);
  ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  // While the peer shard is pending, the backstop must defer: its producer may still
  // arrive once that shard is drained.
  for (int i = 0; i < 16; ++i)
    cu->step();
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  // The peer command processor ran its placement pass (whatever it could place is
  // now resident; the rest is unplaceable): clear the pending mark. With no shard
  // pending the process is provably quiescent and the backstop escapes.
  cu->gws_device().set_shard_pending(/*dispatch_id=*/0, /*process_id=*/0,
                                     /*shard_key=*/&peer_cp_key, /*pending=*/false);
  for (int i = 0; i < 16 && (!wf0->is_halted() || !wf1->is_halted()); ++i)
    cu->step();
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf1->is_halted());
}

// Fan-out shards are tracked independently: a dispatch spread over two command
// processors defers escape until EVERY shard has attempted placement. A completed
// shard's spare capacity is irrelevant (it cannot place another shard's fixed
// workgroup share), so clearing one shard must not release the backstop while the
// other is still pending. Regression (kuhar N3): unioning the shards let a finished
// shard mask a peer shard that could never place its workgroup, on all seven ISAs.
TEST_P(DsGwsTest, FanoutShardsTrackedIndependentlyForEscape) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_twoshard_mem");
  amdgpu::L2Cache l2("ds_gws_twoshard_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/2);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2,
                      /*num_named_barriers=*/0, /*process_id=*/0,
                      /*dispatch_total_waves=*/4);
  int shard_a_key = 0;
  int shard_b_key = 0;
  cu->gws_device().set_shard_pending(/*dispatch_id=*/0, /*process_id=*/0, &shard_a_key,
                                     /*pending=*/true);
  cu->gws_device().set_shard_pending(/*dispatch_id=*/0, /*process_id=*/0, &shard_b_key,
                                     /*pending=*/true);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/3);
  run_gws(*cu, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/3);
  ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  // Shard A finished its placement pass; shard B is still pending, so the backstop
  // must keep deferring (B may yet place the producer).
  cu->gws_device().set_shard_pending(/*dispatch_id=*/0, /*process_id=*/0, &shard_a_key,
                                     /*pending=*/false);
  for (int i = 0; i < 16; ++i)
    cu->step();
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  // Shard B attempted too: no shard pending, so the backstop escapes.
  cu->gws_device().set_shard_pending(/*dispatch_id=*/0, /*process_id=*/0, &shard_b_key,
                                     /*pending=*/false);
  for (int i = 0; i < 16 && (!wf0->is_halted() || !wf1->is_halted()); ++i)
    cu->step();
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf1->is_halted());
}

// Free capacity on an idle peer CU that holds none of the dispatch's work must not
// defer escape. A full two-slot CU hosts the parked waves; a second CU shares the
// store (so it joins the quiescence scan) but is idle. Because the store no longer
// re-derives placement from CU wave slots (placement lives in the command
// processor), that idle CU's free capacity is irrelevant: with no fan-out shard
// pending the backstop escapes. Regression (kuhar N2/N3): the store must not consult
// a peer CU's allocator at all (it both raced that CU and misjudged placement).
TEST_P(DsGwsTest, IdlePeerCuDoesNotDeferEscape) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_idlepeer_mem");
  amdgpu::L2Cache l2("ds_gws_idlepeer_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/2);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws); // Shares the store (joins quiescence) but stays idle.
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *wf1 = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2,
                       /*num_named_barriers=*/0, /*process_id=*/0,
                       /*dispatch_total_waves=*/4);
  for (auto *wf : {wf0, wf1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }
  run_gws(*cu0, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/3);
  run_gws(*cu0, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/3);
  ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  for (int i = 0; i < 16 && (!wf0->is_halted() || !wf1->is_halted()); ++i)
    cu0->step();
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf1->is_halted());
}

// Fault/cancel paths must not leave a dropped dispatch's participant accounting
// behind. The centralized teardown (abort_dispatch, which the VM-fault handler and
// CommandProcessor::fail_pm4_queue both drive) notifies the store so the dropped
// dispatch's admission entry is cleared. Here a dispatch admits one of two expected
// waves and is then aborted; its admission entry must be gone afterwards, and a
// second, fully admitted single-wave dispatch of the same process must still escape.
TEST_P(DsGwsTest, FaultedDispatchClearsAdmissionAndDoesNotWedgeEscape) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_faultdrop_mem");
  amdgpu::L2Cache l2("ds_gws_faultdrop_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto decoder = Decoder::create(arch);

  // Dispatch 0: declares a two-wave grid but only one wave is ever admitted.
  auto *dropped = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  ASSERT_NE(dropped, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1,
                      /*num_named_barriers=*/0, /*process_id=*/0,
                      /*dispatch_total_waves=*/2);
  ASSERT_EQ(cu->gws_device().tracked_dispatch_count(), 1u);

  // The dispatch is aborted (its queue failed): the centralized teardown drops the
  // resident wave and clears the dropped dispatch's participant accounting.
  cu->abort_dispatch(/*dispatch_id=*/0);
  EXPECT_TRUE(dropped->is_halted());
  EXPECT_EQ(cu->gws_device().tracked_dispatch_count(), 0u)
      << "aborting a partially-admitted dispatch left its admission entry behind";

  // Dispatch 1: a single-wave grid of the same process. Its lone wave does a P with
  // no credit and parks; no fan-out shard is pending, so the backstop sees a
  // quiescent process and escapes it.
  auto *waiter = cu->dispatch_wf(/*wg_id=*/1, kPc, 102, 16);
  ASSERT_NE(waiter, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/1, /*wf_count=*/1,
                      /*num_named_barriers=*/0, /*process_id=*/0,
                      /*dispatch_total_waves=*/1);
  waiter->set_exec(0x1);
  waiter->set_m0(0);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *waiter, /*count=*/0);
  ASSERT_EQ(waiter->state(), amdgpu::WfState::GWS_WAIT);

  for (int i = 0; i < 16 && !waiter->is_halted(); ++i)
    cu->step();
  EXPECT_TRUE(waiter->is_halted());
}

// The store-level invariant CommandProcessor::fail_pm4_queue now relies on for its
// no-live-wave case: a dropped dispatch whose last wave already retired still holds
// an admitted<expected admission entry (remove_resident deliberately keeps it), so
// the fault path must notify the store directly. notify_dispatch_gone clears that
// entry even though no wave remains to drive the clear. Regression (kuhar N1): a
// fault path that discovered dispatches only from live waves left this entry behind.
TEST_P(DsGwsTest, NotifyDispatchGoneClearsRetiredWaveAdmission) {
  const auto arch = GetParam();
  (void)arch;
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  // A dispatch admits one of two expected waves, then that wave retires: residency
  // drops to zero but the admitted<expected admission entry is intentionally kept.
  gws->add_resident(/*process_id=*/0, /*dispatch_id=*/7, /*count=*/1,
                    /*expected_total_waves=*/2);
  gws->remove_resident(/*dispatch_id=*/7, /*count=*/1);
  EXPECT_EQ(gws->tracked_dispatch_count(), 1u)
      << "a partially-admitted dispatch must retain its admission after its wave retires";

  // The fault path clears it directly (no live wave remains to drive the clear).
  gws->notify_dispatch_gone(/*dispatch_id=*/7);
  EXPECT_EQ(gws->tracked_dispatch_count(), 0u);
}

// A remote release (a V/BR credit, or a barrier/release-all generation bump) is
// published as resource state and applied by the waiting peer's own CU on a later
// step. If the signaling CU then goes idle before that peer polls, a quiescence
// decision taken on a THIRD CU must not treat the peer's pre-release zero-runnable
// snapshot as proof that nothing can still signal -- the pending credit will wake
// the peer, which can then signal. Publishing such a release therefore invalidates
// the per-CU snapshots (advances the snapshot epoch). Here waiter_a parks on CU A;
// waiter_b parks on CU B (and, once woken, would signal A); a producer on CU C
// signals B's resource and halts. Stepping A before B must not escape waiter_a.
// Regression: without the epoch bump, A sees B's stale zero-runnable snapshot and
// escapes waiter_a prematurely.
TEST_P(DsGwsTest, RemoteReleaseDefersPeerQuiescenceDecision) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_remote_release_mem");
  amdgpu::L2Cache l2("ds_gws_remote_release_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu_a = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu_b = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu_c = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu_a->set_gws_device(gws);
  cu_b->set_gws_device(gws);
  cu_c->set_gws_device(gws);
  auto decoder = Decoder::create(arch);

  auto *waiter_a = cu_a->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *waiter_b = cu_b->dispatch_wf(/*wg_id=*/1, kPc, 102, 16);
  auto *producer_c = cu_c->dispatch_wf(/*wg_id=*/2, kPc, 102, 16);
  ASSERT_NE(waiter_a, nullptr);
  ASSERT_NE(waiter_b, nullptr);
  ASSERT_NE(producer_c, nullptr);
  cu_a->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu_b->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  cu_c->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/2, /*wf_count=*/1);
  for (auto *wf : {waiter_a, waiter_b, producer_c}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  // waiter_a parks on rid 0; waiter_b parks on rid 1 (offset0 selects the rid).
  run_gws(*cu_a, *decoder, arch, GwsOp::kSemaP, *waiter_a, /*count=*/0, /*offset0=*/0);
  run_gws(*cu_b, *decoder, arch, GwsOp::kSemaP, *waiter_b, /*count=*/0, /*offset0=*/1);
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  // Publish both waiters' current (zero-runnable) snapshots. The producer on CU C
  // is still runnable (and has published nothing), so the process is not quiescent
  // and the backstop does not fire during these refresh steps.
  cu_a->step();
  cu_b->step();
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  // CU C signals waiter_b's resource (leaving a credit B will consume on its next
  // poll) and then goes idle. waiter_b is still parked, so its published snapshot
  // is now stale: the pending credit will make it runnable.
  run_gws(*cu_c, *decoder, arch, GwsOp::kSemaV, *producer_c, /*count=*/0, /*offset0=*/1);
  producer_c->halt();

  // Step A before B. The release must have invalidated B's snapshot, so A cannot
  // conclude the process is quiescent: waiter_a must stay parked.
  cu_a->step();
  EXPECT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT)
      << "waiter A escaped before the peer a pending release will wake could signal";

  // B now polls, consumes the credit, and wakes -- confirming the deferred signaler
  // was real.
  for (int i = 0; i < 4 && waiter_b->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu_b->step();
  EXPECT_NE(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  waiter_a->halt();
}

// Companion to the V case (kuhar N5): a GWS init also publishes a consumable
// semaphore credit, so it too must invalidate a peer's stale quiescence snapshot.
// An init can run on a different dispatch/CU from the P it feeds (the CLR RunGwsInit
// pattern), so a peer-CU P already parked may become runnable once it consumes the
// seeded credit. Here waiter_a parks on rid 0; waiter_b parks on rid 1 (and, once
// woken, would signal A); a producer on CU C inits rid 1 and halts. Stepping A
// before B must not escape waiter_a. Regression: without init invalidating the
// snapshot, A escapes on B's pre-init zero-runnable summary, on all seven ISAs.
TEST_P(DsGwsTest, RemoteInitDefersPeerQuiescenceDecision) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_remote_init_mem");
  amdgpu::L2Cache l2("ds_gws_remote_init_l2");
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu_a = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu_b = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu_c = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu_a->set_gws_device(gws);
  cu_b->set_gws_device(gws);
  cu_c->set_gws_device(gws);
  auto decoder = Decoder::create(arch);

  auto *waiter_a = cu_a->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *waiter_b = cu_b->dispatch_wf(/*wg_id=*/1, kPc, 102, 16);
  auto *producer_c = cu_c->dispatch_wf(/*wg_id=*/2, kPc, 102, 16);
  ASSERT_NE(waiter_a, nullptr);
  ASSERT_NE(waiter_b, nullptr);
  ASSERT_NE(producer_c, nullptr);
  cu_a->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu_b->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  cu_c->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/2, /*wf_count=*/1);
  for (auto *wf : {waiter_a, waiter_b, producer_c}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }

  run_gws(*cu_a, *decoder, arch, GwsOp::kSemaP, *waiter_a, /*count=*/0, /*offset0=*/0);
  run_gws(*cu_b, *decoder, arch, GwsOp::kSemaP, *waiter_b, /*count=*/0, /*offset0=*/1);
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  cu_a->step();
  cu_b->step();
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  // CU C seeds a credit on waiter_b's resource via init and then goes idle.
  run_gws(*cu_c, *decoder, arch, GwsOp::kInit, *producer_c, /*count=*/1, /*offset0=*/1);
  producer_c->halt();

  // Step A before B. The init must have invalidated B's snapshot, so waiter_a stays
  // parked rather than escaping on B's stale zero-runnable summary.
  cu_a->step();
  EXPECT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT)
      << "waiter A escaped before the peer an init-seeded credit will wake could signal";

  // B now polls, consumes the seeded credit, and wakes.
  for (int i = 0; i < 4 && waiter_b->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu_b->step();
  EXPECT_NE(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  waiter_a->halt();
}

// A CU torn down while it still hosts a parked GWS wave must not leave the shared
// store's parked bookkeeping inflated (which would wedge any_parked() true and
// mislead the quiescence backstop). unregister_compute_unit (run from the CU's
// destructor) reconciles the departing CU's still-parked waves out of the store.
TEST_P(DsGwsTest, DestroyingCuReconcilesParkedWaveInSharedStore) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_destroy_mem");
  amdgpu::L2Cache l2("ds_gws_destroy_l2");
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  ASSERT_FALSE(gws->any_parked());
  {
    auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
    cu->set_gws_device(gws);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
    ASSERT_NE(wf, nullptr);
    cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
    wf->set_exec(0x1);
    wf->set_m0(0);
    // Park a semaphore P (no credit), then let the CU go out of scope unwoken.
    run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf, /*count=*/0);
    ASSERT_EQ(wf->state(), amdgpu::WfState::GWS_WAIT);
    ASSERT_TRUE(gws->any_parked());
  }
  // The CU's destructor unregistered it and reconciled the parked wave out.
  EXPECT_FALSE(gws->any_parked());
}

// Companion to the destruction case: re-pointing a CU at a different store via
// set_gws_device must reconcile the CU's parked contribution out of the store it
// is leaving AND into the store it joins. The destination here is already
// populated (another CU's wave is parked in store B), so the two stores' parked
// bookkeeping must stay independent and correct across the migration. Regression:
// if the migrated wave is not reconciled into B, B undercounts it, and the later
// notify_parked_wave_gone() for that wave decrements a count it never incremented
// -- wrongly cancelling B's unrelated, still-parked waiter.
TEST_P(DsGwsTest, ReplacingCuStoreReconcilesParkedWaveInOldStore) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_replace_mem");
  amdgpu::L2Cache l2("ds_gws_replace_l2");
  auto store_a = std::make_shared<amdgpu::GwsDevice>();
  auto store_b = std::make_shared<amdgpu::GwsDevice>();
  auto decoder = Decoder::create(arch);

  // A resident waiter already parked in store B on its own CU.
  auto cu_b = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu_b->set_gws_device(store_b);
  auto *wf_b = cu_b->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf_b, nullptr);
  cu_b->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/0, /*wf_count=*/1);
  wf_b->set_exec(0x1);
  wf_b->set_m0(0);
  run_gws(*cu_b, *decoder, arch, GwsOp::kSemaP, *wf_b, /*count=*/0);
  ASSERT_EQ(wf_b->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_TRUE(store_b->any_parked());

  // A second CU parks a waiter in store A.
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu->set_gws_device(store_a);
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  wf->set_exec(0x1);
  wf->set_m0(0);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf, /*count=*/0);
  ASSERT_EQ(wf->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_TRUE(store_a->any_parked());

  // Move the CU to store B: leaving A reconciles its parked wave out, joining B
  // reconciles it in (B now accounts both its own waiter and the migrated one).
  cu->set_gws_device(store_b);
  EXPECT_FALSE(store_a->any_parked());
  EXPECT_TRUE(store_b->any_parked());

  // Retire the migrated wave through the normal free path (notify_parked_wave_gone
  // on store B). B must still report its original waiter as parked.
  cu->abort_dispatch(/*dispatch_id=*/0);
  EXPECT_TRUE(wf->is_halted());
  EXPECT_TRUE(store_b->any_parked())
      << "retiring the migrated wave cancelled store B's unrelated parked waiter";

  // Halt cu_b's waiter before teardown so it is no longer GWS_WAIT.
  wf_b->halt();
}

// GWS resources persist across dispatches but belong to a process; on process
// teardown (SimulatedKfd::close -> GwsDevice::release_process) that process's table
// is dropped while other live processes keep theirs. Two processes each seed a
// semaphore credit; releasing process A drops only A's table (a later P on A then
// parks for lack of a credit), while B's seeded credit survives and is consumed.
TEST_P(DsGwsTest, ReleaseProcessDropsOnlyThatProcessTable) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_release_mem");
  amdgpu::L2Cache l2("ds_gws_release_l2");
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu = make_gws_cu(mem, l2, arch, /*wf_slots=*/8);
  cu->set_gws_device(gws);
  auto decoder = Decoder::create(arch);
  constexpr uint32_t kProcA = 11;
  constexpr uint32_t kProcB = 22;

  auto *init_a = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *init_b = cu->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(init_a, nullptr);
  ASSERT_NE(init_b, nullptr);
  init_a->set_process_id(kProcA);
  init_b->set_process_id(kProcB);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1,
                      /*num_named_barriers=*/0, kProcA, /*dispatch_total_waves=*/0);
  cu->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/1, /*wf_count=*/1,
                      /*num_named_barriers=*/0, kProcB, /*dispatch_total_waves=*/0);
  for (auto *wf : {init_a, init_b}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }
  // Both processes seed a credit; the store now tracks two process tables.
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *init_a, /*count=*/1);
  run_gws(*cu, *decoder, arch, GwsOp::kInit, *init_b, /*count=*/1);
  EXPECT_EQ(gws->tracked_process_count(), 2u);
  init_a->halt();
  init_b->halt();

  // Process A tears down: only its table is dropped.
  gws->release_process(kProcA);
  EXPECT_EQ(gws->tracked_process_count(), 1u);

  // A fresh P on process A finds no credit (its table was released) and parks.
  auto *p_a = cu->dispatch_wf(/*wg_id=*/2, /*pc=*/0, 102, 16);
  ASSERT_NE(p_a, nullptr);
  p_a->set_process_id(kProcA);
  cu->begin_workgroup(/*dispatch_id=*/2, /*wg_id=*/2, /*wf_count=*/1,
                      /*num_named_barriers=*/0, kProcA, /*dispatch_total_waves=*/0);
  p_a->set_exec(0x1);
  p_a->set_m0(0);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *p_a, /*count=*/0);
  EXPECT_EQ(p_a->state(), amdgpu::WfState::GWS_WAIT)
      << "released process A still had a lingering credit";

  // Process B was untouched: its seeded credit is still consumable.
  auto *p_b = cu->dispatch_wf(/*wg_id=*/3, /*pc=*/0, 102, 16);
  ASSERT_NE(p_b, nullptr);
  p_b->set_process_id(kProcB);
  cu->begin_workgroup(/*dispatch_id=*/3, /*wg_id=*/3, /*wf_count=*/1,
                      /*num_named_barriers=*/0, kProcB, /*dispatch_total_waves=*/0);
  p_b->set_exec(0x1);
  p_b->set_m0(0);
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *p_b, /*count=*/0);
  EXPECT_EQ(p_b->state(), amdgpu::WfState::RUNNING)
      << "process B lost its credit when A was released";

  p_b->halt();
  // p_a left parked; the CU's unregister reconciles it on destruction.
}

} // namespace
