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
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
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
    const auto words =
        build_gws(arch, gws_opcode(arch, variant.op), /*gds=*/1, gws_has_addr(variant.op) ? 4 : 0);
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
    const auto words =
        build_gws(arch, gws_opcode(arch, variant.op), /*gds=*/0, gws_has_addr(variant.op) ? 4 : 0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << variant.mnemonic << " failed to decode on arch " << unsigned(arch);
    EXPECT_EQ(inst->mnemonic(), variant.mnemonic);
    EXPECT_TRUE(inst->is_memory_op()) << variant.mnemonic;
  }
}

// ---------------------------------------------------------------------------
// Stateful GWS: co-residency-gated parking. These exercise the CU-local
// counter table through execute_instruction(), which runs the generated
// execute body (decode rid, read count, call the CU hook). Resources are
// workgroup-private, so a rendezvous only ever blocks co-resident waves.
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

// ds_gws_sema_p parks when no credit is available; ds_gws_sema_v then releases a
// parked waiter within the co-resident scope.
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

// A P whose only possible producer lives in another workgroup cannot rendezvous
// in this CU-local, workgroup-private model, so it completes structurally
// instead of parking (otherwise it would hang forever).
TEST_P(DsGwsTest, CrossWorkgroupSemaphorePStaysStructural) {
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

  // wf1 is the only peer and lives in a different workgroup, so wf0's P has no
  // in-scope signaller and must not park.
  run_gws(*cu, *decoder, arch, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

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
TEST(DsGwsMixedTest, EscapeReleasesGwsWaveStalledAgainstSBarrier) {
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3}) {
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
    ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT) << "arch " << unsigned(arch);

    // wf1 executes a plain workgroup s_barrier and stalls (wf0 never arrives).
    const auto words =
        (arch == ROCJITSU_CODE_ARCH_CDNA1)
            ? cdna1::build_sopp(cdna1::kSBarrierSopp)
            : (arch == ROCJITSU_CODE_ARCH_CDNA2 ? cdna2::build_sopp(cdna2::kSBarrierSopp)
                                                : cdna3::build_sopp(cdna3::kSBarrierSopp));
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << "arch " << unsigned(arch);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf1).succeeded());
    ASSERT_EQ(wf1->state(), amdgpu::WfState::BARRIER) << "arch " << unsigned(arch);

    // The escape scan releases the GWS-parked wave even though its sibling is
    // stalled at an s_barrier rather than a GWS op.
    for (int i = 0; i < 4 && wf0->state() == amdgpu::WfState::GWS_WAIT; ++i)
      cu->step();
    EXPECT_NE(wf0->state(), amdgpu::WfState::GWS_WAIT) << "arch " << unsigned(arch);
  }
}

// On CDNA2/CDNA3 the DS acc bit selects the AGPR bank for the count operand.
// Writing a parking count to the AGPR slot and a non-parking count to the VGPR
// slot proves acc routes the read to the AGPR.
TEST(DsGwsAccTest, AccBitSelectsAgprCountOperand) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3}) {
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
    const auto acc_words = (arch == ROCJITSU_CODE_ARCH_CDNA2)
                               ? cdna2::build_ds(op, {.gds = 1, .acc = 1, .addr = kAddrVgpr})
                               : cdna3::build_ds(op, {.gds = 1, .acc = 1, .addr = kAddrVgpr});
    std::unique_ptr<Instruction> acc_inst(decode_valid(*decoder, acc_words.data()));
    ASSERT_NE(acc_inst, nullptr) << "arch " << unsigned(arch);
    ASSERT_TRUE(cu->execute_instruction(acc_inst.get(), *wf0).succeeded());
    EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT) << "arch " << unsigned(arch);

    // wf1: acc=0 reads the VGPR slot (count 5 -> participants 6 > resident 2 ->
    // structural no-op, keeps running) before touching the parked resource.
    cu->write_vgpr(wf1->vgpr_alloc().base + kAddrVgpr, /*lane=*/0, /*vgpr=*/5);
    const auto vgpr_words = (arch == ROCJITSU_CODE_ARCH_CDNA2)
                                ? cdna2::build_ds(op, {.gds = 1, .acc = 0, .addr = kAddrVgpr})
                                : cdna3::build_ds(op, {.gds = 1, .acc = 0, .addr = kAddrVgpr});
    std::unique_ptr<Instruction> vgpr_inst(decode_valid(*decoder, vgpr_words.data()));
    ASSERT_NE(vgpr_inst, nullptr) << "arch " << unsigned(arch);
    ASSERT_TRUE(cu->execute_instruction(vgpr_inst.get(), *wf1).succeeded());
    EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING) << "arch " << unsigned(arch);

    wf0->halt();
    wf1->halt();
  }
}

} // namespace
