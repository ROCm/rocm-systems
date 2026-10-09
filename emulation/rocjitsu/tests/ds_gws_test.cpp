// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

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
  // GWS executes independently of EXEC; use an empty mask to exercise that.
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
  constexpr uint32_t kSEndpgm = 0xBF810000u;
  mem.write32(kPc, kSEndpgm);
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

  // The escape scan must unblock the two parked waves.
  for (int i = 0; i < 4 && (wf0->state() == amdgpu::WfState::GWS_WAIT ||
                            wf1->state() == amdgpu::WfState::GWS_WAIT);
       ++i)
    cu->step();
  EXPECT_NE(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_NE(wf1->state(), amdgpu::WfState::GWS_WAIT);
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
  constexpr uint32_t kSEndpgm = 0xBF810000u;
  mem.write32(kPc, kSEndpgm);
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
  // stalled at an s_barrier rather than a GWS op.
  for (int i = 0; i < 4 && wf0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu->step();
  EXPECT_NE(wf0->state(), amdgpu::WfState::GWS_WAIT);
}

// GWS resources are device-global: the command processor scatters a dispatch's
// workgroups across CUs, so a P parked on one CU must be released by a V issued
// from a wave on a *different* CU. Both CUs share one GwsDevice (declared so it
// outlives them; CUs unregister on destruction).
TEST_P(DsGwsTest, CrossComputeUnitSemaphorePWakesOnCrossCuV) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xcu_mem");
  amdgpu::L2Cache l2("ds_gws_xcu_l2");
  amdgpu::GwsDevice gws;
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(&gws);
  cu1->set_gws_device(&gws);
  auto decoder = Decoder::create(arch);
  auto *consumer = cu0->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
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

  // Producer signals from CU1: the shared store wakes the waiter parked on CU0.
  run_gws(*cu1, *decoder, arch, GwsOp::kSemaV, *producer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::RUNNING);

  consumer->halt();
  producer->halt();
}

// A GWS barrier programmed for two participants whose workgroups land on
// different CUs rendezvouses across the CU boundary: the early arrival on CU0
// parks and the releasing arrival on CU1 wakes it. The residency bound is summed
// across both CUs (one resident wave each -> two participants).
TEST_P(DsGwsTest, CrossComputeUnitBarrierReleasesBothParticipants) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_xcubar_mem");
  amdgpu::L2Cache l2("ds_gws_xcubar_l2");
  amdgpu::GwsDevice gws;
  auto cu0 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  auto cu1 = make_gws_cu(mem, l2, arch, /*wf_slots=*/4);
  cu0->set_gws_device(&gws);
  cu1->set_gws_device(&gws);
  auto decoder = Decoder::create(arch);
  auto *wf0 = cu0->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu1->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  cu1->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/1, /*wf_count=*/1);
  wf0->set_exec(0x1);
  wf0->set_m0(0);
  wf1->set_exec(0x1);
  wf1->set_m0(0);

  // Two participants -> programmed value 1. wf0 (CU0) arrives first and parks.
  run_gws(*cu0, *decoder, arch, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 (CU1) is the releasing arrival and wakes the peer parked on CU0.
  run_gws(*cu1, *decoder, arch, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
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

} // namespace
