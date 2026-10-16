// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "gws_test_encoding.h"
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
#include "rocjitsu/vm/plugins/instruction_family.h"
#include "util/except.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using namespace rocjitsu;

// GWS (Global Wave Sync) is defined by every GFX9 (CDNA1-3) ISA and by the
// GFX10/GFX11 (RDNA1-3.5) ISAs. The opcode map and the count-carrying ADDR choice
// are shared with the cross-XCD regressions via gws_test_encoding.h.
using test::gws_has_addr;
using test::gws_opcode;
using test::GwsOp;

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

struct WorkgroupConfig {
  uint32_t dispatch_id = 0;
  uint32_t wg_id = 0;
  uint32_t process_id = 0;
  uint64_t pc = 0;
};

// Ordinary instruction tests start with lane 0 active and resource 0 selected.
// Scheduling and lifetime tests can still dispatch and initialize waves directly.
template <size_t N>
std::array<amdgpu::Wavefront *, N> make_workgroup(amdgpu::ComputeUnitCore &cu,
                                                  WorkgroupConfig config = {}) {
  std::array<amdgpu::Wavefront *, N> waves{};
  for (auto &wave : waves) {
    wave = cu.dispatch_wf(config.wg_id, config.pc, 102, 16);
    // A setup failure must stop the test before it dereferences a missing wave.
    if (!wave)
      throw std::runtime_error("Failed to allocate a GWS test wave");
    wave->set_dispatch_id(config.dispatch_id);
    wave->set_process_id(config.process_id);
    wave->set_exec(0x1);
    wave->set_m0(0);
  }
  cu.begin_workgroup(config.dispatch_id, config.wg_id, N);
  return waves;
}

// Each context owns its memory, cache and decoder. Tests keep CUs local so
// teardown and store-lifetime scenarios remain explicit.
class GwsTestContext {
public:
  explicit GwsTestContext(rj_code_arch_t arch) : arch(arch), decoder(Decoder::create(arch)) {
    l2.set_backing_memory(&mem);
  }

  amdgpu::ComputeUnitCore::Config cu_config(uint32_t wf_slots = 4) const {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = wf_slots;
    cfg.sgprs_per_wf = 102;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    return cfg;
  }

  std::unique_ptr<amdgpu::ComputeUnitCore> make_cu(uint32_t wf_slots = 4) {
    return amdgpu::ComputeUnitCore::create("ds_gws_cu", cu_config(wf_slots), &mem, &l2);
  }

  const rj_code_arch_t arch;
  amdgpu::GpuMemory mem{"ds_gws_mem"};
  amdgpu::L2Cache l2{"ds_gws_l2"};
  std::unique_ptr<Decoder> decoder;

  // Decode one GWS op, publish @p count in lane 0 of the ADDR VGPR, and
  // run it on @p wf through the CU execute path. @p offset0 feeds the rid decode.
  void run_gws(amdgpu::ComputeUnitCore &cu, GwsOp op, amdgpu::Wavefront &wf, uint32_t count,
               uint8_t offset0 = 0) {
    const bool has_addr = gws_has_addr(op);
    if (has_addr)
      cu.write_vgpr(wf.vgpr_alloc().base + kAddrVgpr, /*lane=*/0, count);
    const auto words =
        build_gws(arch, gws_opcode(arch, op), /*gds=*/1, has_addr ? kAddrVgpr : 0, offset0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu.execute_instruction(inst.get(), wf).succeeded());
  }

  // Like run_gws, but with an explicit EXEC mask and per-lane ADDR-VGPR counts so
  // the EXEC=0 and nonzero-first-active-lane paths can be exercised. Each entry of
  // @p lane_counts is (lane, value); @p exec is written verbatim onto the wave.
  void run_gws_exec(amdgpu::ComputeUnitCore &cu, GwsOp op, amdgpu::Wavefront &wf, uint64_t exec,
                    std::initializer_list<std::pair<uint32_t, uint32_t>> lane_counts,
                    uint8_t offset0 = 0) {
    const bool has_addr = gws_has_addr(op);
    if (has_addr)
      for (const auto &[lane, value] : lane_counts)
        cu.write_vgpr(wf.vgpr_alloc().base + kAddrVgpr, lane, value);
    wf.set_exec(exec);
    const auto words =
        build_gws(arch, gws_opcode(arch, op), /*gds=*/1, has_addr ? kAddrVgpr : 0, offset0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu.execute_instruction(inst.get(), wf).succeeded());
  }
};

class DsGwsTest : public ::testing::TestWithParam<rj_code_arch_t>, public GwsTestContext {
public:
  DsGwsTest() : GwsTestContext(GetParam()) {}
};

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
  auto cfg = cu_config(1);
  auto cu = amdgpu::ComputeUnitCore::create("ds_gws", cfg, &mem, &l2);
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
    EXPECT_TRUE(inst->is_memory_op());
    EXPECT_EQ(plugins::classify_instruction(*inst), plugins::InstructionFamily::Control);

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
    EXPECT_EQ(plugins::classify_instruction(*inst), plugins::InstructionFamily::Control);

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

// GWS is architecturally defined only for gds=1. The gds=0 encoding still decodes
// to the same mnemonic (the GDS bit does not change decode), but executing it must
// be rejected rather than silently applying the stateful GWS semantics: the
// generated body would park a wave while the instruction's completion is classified
// as LDS and omits the GDS wait obligation, so the stateful behavior and the
// memory-wait accounting would disagree. The inverse GDS guard (ds_gws rejects
// gds=0, the mirror of the other DS classes rejecting gds=1) raises
// UnimplementedInst at execution instead.
TEST_P(DsGwsTest, RejectsExecutionWithoutGdsBit) {
  auto cfg = cu_config(1);
  auto cu = amdgpu::ComputeUnitCore::create("ds_gws_nogds", cfg, &mem, &l2);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1);
  wf->set_m0(0);
  for (const auto &variant : kGwsVariants) {
    const auto words = build_gws(arch, gws_opcode(arch, variant.op), /*gds=*/0,
                                 gws_has_addr(variant.op) ? kAddrVgpr : 0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << variant.mnemonic << " failed to decode on arch " << unsigned(arch);
    EXPECT_EQ(inst->mnemonic(), variant.mnemonic);
    // Decode is unaffected by the GDS bit, so the op still classifies as a memory op.
    EXPECT_TRUE(inst->is_memory_op()) << variant.mnemonic;
    // Execution must be rejected: no wave is parked and no stateful side effect runs.
    EXPECT_THROW((void)cu->execute_instruction(inst.get(), *wf), util::UnimplementedInst)
        << variant.mnemonic << " on arch " << unsigned(arch);
    EXPECT_NE(wf->state(), amdgpu::WfState::GWS_WAIT) << variant.mnemonic;
  }
  wf->halt();
}

// ---------------------------------------------------------------------------
// Stateful GWS: device-global rendezvous. These exercise the shared counter
// table through execute_instruction(), which runs the generated execute body
// (decode rid, read count, call the CU hook). Resources are device-global and
// persist across dispatches, so a signal/arrival from any wave -- any workgroup,
// any CU, even a later dispatch -- releases a waiter parked by another.
// ---------------------------------------------------------------------------

// RDNA3/RDNA3.5 (ISA "Global Wave Sync", section 13.4.2) run the count-carrying
// GWS operations (INIT, BARRIER, SEMA_BR) even when EXEC selects no lane, taking
// the count operand from lane 0. A barrier(count=1) issued with EXEC=0 must
// therefore still establish a real two-participant rendezvous: the first arrival
// parks and the co-resident arrival releases it. An EXEC-gated structural no-op
// (the pre-fix behavior) would instead leave the first arrival RUNNING. This is
// RDNA3/3.5-specific, so the case is skipped on the other parameterized targets.
TEST_P(DsGwsTest, Rdna3CountOpsHonorLaneZeroWhenExecEmpty) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "EXEC=0 lane-0 count ops are specific to RDNA3/3.5";
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  cu->gws_init(*wf0, 0, 1);
  run_gws_exec(*cu, GwsOp::kBarrier, *wf0, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  // The co-resident arrival (also EXEC=0) releases it: a genuine rendezvous.
  run_gws_exec(*cu, GwsOp::kBarrier, *wf1, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

TEST_P(DsGwsTest, Rdna3InitSeedsCreditViaLaneZeroWhenExecEmpty) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "EXEC=0 lane-0 count ops are specific to RDNA3/3.5";
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  // init(count=1) with EXEC=0 seeds one semaphore credit via lane 0.
  run_gws_exec(*cu, GwsOp::kInit, *wf0, /*exec=*/0, {{0, 1}});
  // First P (live EXEC) consumes the seeded credit and stays RUNNING: the witness
  // that the EXEC=0 init took effect. An EXEC-gated init would leave no credit.
  run_gws_exec(*cu, GwsOp::kSemaP, *wf0, /*exec=*/0x1, {});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  // Second P (live EXEC) finds no credit left and parks: only one was seeded.
  run_gws_exec(*cu, GwsOp::kSemaP, *wf1, /*exec=*/0x1, {});
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  wf0->halt();
  wf1->halt();
}

// SEMA_BR is the third EXEC-independent count op: on RDNA3/3.5 a sema_br issued
// with EXEC=0 still releases parked waiters, taking the release count from lane 0.
// (sema_p itself stays EXEC-gated, so the waiter is parked with a live lane.)
TEST_P(DsGwsTest, Rdna3SemaBrReleasesViaLaneZeroWhenExecEmpty) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "EXEC=0 lane-0 count ops are specific to RDNA3/3.5";
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  // sema_p parks wf0 (EXEC-gated op, so it uses a live lane).
  run_gws_exec(*cu, GwsOp::kSemaP, *wf0, /*exec=*/0x1, {{0, 0}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  // sema_br(count=1) with EXEC=0 releases the waiter via the lane-0 count.
  run_gws_exec(*cu, GwsOp::kSemaBr, *wf1, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// The empty-EXEC rule is RDNA3/3.5-specific and must NOT be extrapolated to the
// older GFX9/GFX10 targets: there a count op issued with EXEC=0 stays an
// EXEC-gated structural no-op (no park). The identical op with a live lane still
// parks normally, proving only the empty mask -- not the resource -- was gated.
TEST_P(DsGwsTest, OlderTargetsSkipCountOpsWhenExecEmpty) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "RDNA3/3.5 run EXEC=0 count ops via lane 0 rather than gating";
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  // EXEC=0: the barrier is skipped, so the first arrival does not park.
  cu->gws_init(*wf0, 0, 1);
  run_gws_exec(*cu, GwsOp::kBarrier, *wf0, /*exec=*/0, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  // With a live lane the identical op parks, then its co-resident releases it.
  run_gws_exec(*cu, GwsOp::kBarrier, *wf0, /*exec=*/0x1, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws_exec(*cu, GwsOp::kBarrier, *wf1, /*exec=*/0x1, {{0, 1}});
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

TEST_P(DsGwsTest, CountOpsReadFirstActiveLaneNotLaneZero) {
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  // INIT and both barrier reloads must read lane 3. A wrong lane-0 read
  // leaves the barrier waiting for 100 arrivals instead of two.
  run_gws_exec(*cu, GwsOp::kInit, *wf0, 0x8, {{0, 99}, {3, 1}});
  for (int phase = 0; phase < 2; ++phase) {
    run_gws_exec(*cu, GwsOp::kBarrier, *wf0, 0x8, {{0, 99}, {3, 1}});
    EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
    run_gws_exec(*cu, GwsOp::kBarrier, *wf1, 0x8, {{0, 99}, {3, 1}});
    EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
    EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  }
  wf0->halt();
  wf1->halt();
}

// A GWS barrier initialized for two resident participants parks the early
// arrival; the final arrival releases the co-resident set. GWS programs the
// barrier value as (participants - 1), so two participants use a count of 1.
TEST_P(DsGwsTest, BarrierParksUntilAllCoResidentWavesArrive) {
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  // Explicitly initialize the resource for two participants (count = 1).
  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/1);

  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  run_gws(*cu, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// After a barrier releases, the counter re-arms so the same resource serves a
// second rendezvous phase.
TEST_P(DsGwsTest, BarrierReusesCounterAcrossPhases) {
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/1);

  for (int phase = 0; phase < 2; ++phase) {
    run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/1);
    EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT) << "phase " << phase;
    run_gws(*cu, GwsOp::kBarrier, *wf1, /*count=*/1);
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
  auto cu = make_cu();
  auto [wf0, wf1, wf2] = make_workgroup<3>(*cu);

  // Phase 1: three participants (init value 2). Two park, the third releases.
  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/2);
  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/1);
  run_gws(*cu, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);
  run_gws(*cu, GwsOp::kBarrier, *wf2, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf2->state(), amdgpu::WfState::RUNNING);

  // Phase 2: the reloaded counter (1) now needs only two arrivals.
  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws(*cu, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
  wf2->halt();
}

TEST_P(DsGwsTest, BarrierWaitsWhenCountExceedsResident) {
  auto cu = make_cu();
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  wf0->set_exec(0x1);
  wf0->set_m0(0);

  // Three participants exceed the two resident waves: wait for a real arrival.
  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/2);
  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/2);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf0->halt();
  wf1->halt();
}

TEST_P(DsGwsTest, BarrierHugeCountStillWaits) {
  auto cu = make_cu();
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  wf0->set_exec(0x1);
  wf0->set_m0(0);

  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/0xffffffffu);
  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/0xffffffffu);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf0->halt();
  wf1->halt();
}

// ds_gws_sema_p parks when no credit is available; ds_gws_sema_v then releases a
// parked waiter within the dispatch.
TEST_P(DsGwsTest, SemaphorePParksAndVReleases) {
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  run_gws(*cu, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  run_gws(*cu, GwsOp::kSemaV, *wf1, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// ds_gws_sema_br releases up to @count parked waiters in one shot.
TEST_P(DsGwsTest, SemaphoreBulkReleaseWakesWaiters) {
  auto cu = make_cu();
  auto [wf0, wf1, wf2] = make_workgroup<3>(*cu);

  run_gws(*cu, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  run_gws(*cu, GwsOp::kSemaP, *wf1, /*count=*/0);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  // One SEMA_BR releases both parked waiters.
  run_gws(*cu, GwsOp::kSemaBr, *wf2, /*count=*/2);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  for (auto *wf : {wf0, wf1, wf2})
    wf->halt();
}

// ds_gws_sema_release_all wakes every parked waiter and clears the credits.
TEST_P(DsGwsTest, SemaphoreReleaseAllWakesWaiters) {
  auto cu = make_cu();
  auto [wf0, wf1, wf2] = make_workgroup<3>(*cu);

  run_gws(*cu, GwsOp::kSemaP, *wf0, /*count=*/0);
  run_gws(*cu, GwsOp::kSemaP, *wf1, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  run_gws(*cu, GwsOp::kReleaseAll, *wf2, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  for (auto *wf : {wf0, wf1, wf2})
    wf->halt();
}

// The rid is ((M0[21:16] + offset0) & 0x3f). A P on rid 0 (M0=0, offset0=0) is
// released by a V whose (M0[21:16]=63, offset0=1) sum wraps back to rid 0.
TEST_P(DsGwsTest, SemaphoreRidSumWrapsToSameResource) {
  auto cu = make_cu();
  auto *wf0 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  auto *wf1 = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  wf0->set_exec(0x1);
  wf1->set_exec(0x1);

  wf0->set_m0(0); // rid = (0 + 0) & 0x3f = 0
  run_gws(*cu, GwsOp::kSemaP, *wf0, /*count=*/0, /*offset0=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf1->set_m0(63u << 16); // rid = (63 + 1) & 0x3f = 0
  run_gws(*cu, GwsOp::kSemaV, *wf1, /*count=*/0, /*offset0=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// GWS resources are shared per process, so a P parked by one workgroup is
// released by a V issued from another workgroup sharing the resource -- the
// hardware rendezvous the workgroup-private model could not express.
TEST_P(DsGwsTest, CrossWorkgroupSemaphorePWakesOnCrossWgV) {
  auto cu = make_cu();
  auto [wf0] = make_workgroup<1>(*cu);
  auto [wf1] = make_workgroup<1>(*cu, {.wg_id = 1});

  // wf0 (wg0) waits with no credit and parks on the shared resource.
  run_gws(*cu, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 (wg1) signals the same shared process resource and wakes wf0.
  run_gws(*cu, GwsOp::kSemaV, *wf1, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

// INIT seeds a semaphore credit on the shared process resource: with a second
// resident wave, the first P consumes that credit and keeps running, while the
// second P finds no credit and parks. Guards the credit seed in GwsDevice::init.
TEST_P(DsGwsTest, InitSeedsCreditFirstPConsumesSecondParks) {
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  // Seed exactly one credit.
  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  // First P consumes the seeded credit and stays running.
  run_gws(*cu, GwsOp::kSemaP, *wf0, /*count=*/0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);

  // Second P finds no credit and parks.
  run_gws(*cu, GwsOp::kSemaP, *wf1, /*count=*/0);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  wf0->halt();
  wf1->halt();
}

TEST_P(DsGwsTest, PollingSiblingDoesNotBlockCrossWgSemaphoreWake) {
  auto cu = make_cu();
  auto [consumer, sibling] = make_workgroup<2>(*cu);
  auto [producer] = make_workgroup<1>(*cu, {.wg_id = 1});

  // Consumer (wg0) parks on P; its sibling stays RUNNING (a spin/poll loop).
  run_gws(*cu, GwsOp::kSemaP, *consumer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(sibling->state(), amdgpu::WfState::RUNNING);

  // Producer (wg1) signals: the consumer wakes even though the sibling is still
  // running, so the running poll loop cannot deadlock the rendezvous.
  run_gws(*cu, GwsOp::kSemaV, *producer, /*count=*/0);
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
  auto cu = make_cu();
  auto [wf0] = make_workgroup<1>(*cu);
  auto [wf1] = make_workgroup<1>(*cu, {.wg_id = 1});

  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/1);

  // Two participants -> programmed value 1. wf0 (wg0) arrives first and parks.
  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 (wg1) is the releasing arrival and wakes the cross-workgroup peer.
  run_gws(*cu, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  wf0->halt();
  wf1->halt();
}

TEST_P(DsGwsTest, UnsatisfiedBarrierFailsAtWaitLimit) {
  auto cu = make_cu();
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto [wf0, wf1, wf2] = make_workgroup<3>(*cu, {.pc = kPc});

  // Three participants (count = 2); two arrive and park, the third retires.
  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/2);
  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/2);
  run_gws(*cu, GwsOp::kBarrier, *wf1, /*count=*/2);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);
  wf2->halt();

  cu->set_gws_wait_limit(8);
  for (int i = 0; i < 7; ++i) {
    cu->step();
    EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  }
  EXPECT_THROW(cu->step(), std::runtime_error);
  EXPECT_FALSE(cu->has_active_wfs());
  EXPECT_FALSE(cu->gws_device().any_parked());
}

TEST_P(DsGwsTest, MixedBarrierWaitFailsAtWaitLimit) {
  auto cu = make_cu();
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto [wf0, wf1] = make_workgroup<2>(*cu, {.pc = kPc});

  // wf0 parks at a two-participant GWS barrier.
  run_gws(*cu, GwsOp::kInit, *wf0, /*count=*/1);
  run_gws(*cu, GwsOp::kBarrier, *wf0, /*count=*/1);
  ASSERT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 executes a plain workgroup s_barrier and stalls (wf0 never arrives).
  const auto words = build_s_barrier(arch);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf1).succeeded());
  ASSERT_EQ(wf1->state(), amdgpu::WfState::BARRIER);

  cu->set_gws_wait_limit(8);
  for (int i = 0; i < 7; ++i) {
    cu->step();
    EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);
  }
  EXPECT_THROW(cu->step(), std::runtime_error);
  EXPECT_FALSE(cu->has_active_wfs());
  EXPECT_FALSE(cu->gws_device().any_parked());
}

// GWS resources are device-global: the command processor scatters a dispatch's
// workgroups across CUs, so a P parked on one CU must be released by a V issued
// from a wave on a *different* CU. Because no CU may mutate a peer's wavefront
// state, the cross-CU wake is published to the shared store and applied by the
// waiter's own CU the next time it steps (update_wf_states runs the GWS poll).
// Both CUs share one GwsDevice (declared so it outlives them; CUs unregister on
// destruction).
TEST_P(DsGwsTest, CrossComputeUnitSemaphorePWakesOnCrossCuV) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto [consumer] = make_workgroup<1>(*cu0, {.pc = kPc});
  auto [producer] = make_workgroup<1>(*cu1, {.wg_id = 1});

  // Consumer parks on CU0's wave slot with no credit.
  run_gws(*cu0, GwsOp::kSemaP, *consumer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);

  // Producer signals from CU1: the credit lands in the shared store, and CU0
  // applies the cross-CU wake on its next step (consumer's PC is s_endpgm, so it
  // retires once released).
  run_gws(*cu1, GwsOp::kSemaV, *producer, /*count=*/0);
  for (int i = 0; i < 4 && consumer->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(consumer->state(), amdgpu::WfState::GWS_WAIT);

  producer->halt();
}

TEST_P(DsGwsTest, CrossComputeUnitBarrierReleasesBothParticipants) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto [wf0] = make_workgroup<1>(*cu0, {.pc = kPc});
  auto [wf1] = make_workgroup<1>(*cu1, {.wg_id = 1});

  run_gws(*cu0, GwsOp::kInit, *wf0, /*count=*/1);

  // Two participants -> programmed value 1. wf0 (CU0) arrives first and parks.
  run_gws(*cu0, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

  // wf1 (CU1) is the releasing arrival: it stays running immediately, and the
  // peer parked on CU0 wakes when CU0 next steps.
  run_gws(*cu1, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  for (int i = 0; i < 4 && wf0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf1->halt();
}

TEST_P(DsGwsTest, CrossCuSemaphoreWaitsForScalarBarrierProducer) {
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
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);

  // CU0: a single consumer parks on a P (same process/dispatch as CU1's waves).
  auto [consumer] = make_workgroup<1>(*cu0, {.pc = kPcConsumer});

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

  run_gws(*cu0, GwsOp::kSemaP, *consumer, /*count=*/0);
  ASSERT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);

  // One CU1 step drives both waves into the s_barrier (fully arrived, but scalar-
  // barrier resolution is deferred to CU1's next step).
  cu1->step();
  ASSERT_EQ(wfa->state(), amdgpu::WfState::BARRIER);
  ASSERT_EQ(wfb->state(), amdgpu::WfState::BARRIER);

  // Now step CU0 repeatedly. The P must stay parked: CU1's barrier is ready to
  // resolve and wave A can still signal, so this is not a genuine deadlock.
  for (int i = 0; i < 6; ++i)
    cu0->step();
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT)
      << "parked P was released before the peer scalar barrier producer signaled";

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

TEST_P(DsGwsTest, OverlappingDispatchesShareProcessResources) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);

  // Dispatch 0's consumer parks on CU0.
  auto [consumer] = make_workgroup<1>(*cu0, {.pc = kPc});

  // Dispatch 1's producer (same process) is resident and runnable on CU1 but has
  // not signaled yet; it is deliberately never stepped so it stays runnable.
  auto [producer] = make_workgroup<1>(*cu1, {.dispatch_id = 1});

  run_gws(*cu0, GwsOp::kSemaP, *consumer, /*count=*/0);
  ASSERT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT);

  for (int i = 0; i < 8; ++i)
    cu0->step();
  EXPECT_EQ(consumer->state(), amdgpu::WfState::GWS_WAIT)
      << "parked P was released while a same-process producer was still runnable";

  // The producer finally signals; CU0 then releases the consumer with a credit.
  run_gws(*cu1, GwsOp::kSemaV, *producer, /*count=*/0);
  for (int i = 0; i < 4 && consumer->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(consumer->state(), amdgpu::WfState::GWS_WAIT);

  producer->halt();
}

TEST_P(DsGwsTest, NewDispatchSignalsPreviouslyParkedWaiter) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);

  // Two consumers of the same process (dispatch 0) park on the same semaphore: C0
  // on CU0 and W1 on CU1. Each is the sole wave of its CU, so a step publishes a
  // zero-runnable summary for that CU.
  auto [c0] = make_workgroup<1>(*cu0, {.pc = kPc});
  auto [w1] = make_workgroup<1>(*cu1, {.wg_id = 1, .pc = kPc});

  run_gws(*cu0, GwsOp::kSemaP, *c0, /*count=*/0);
  run_gws(*cu1, GwsOp::kSemaP, *w1, /*count=*/0);
  ASSERT_EQ(c0->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(w1->state(), amdgpu::WfState::GWS_WAIT);

  // CU1 publishes its zero-runnable summary (its only wave, W1, is parked).
  cu1->step();

  auto [producer] = make_workgroup<1>(*cu1, {.dispatch_id = 1});

  for (int i = 0; i < 8; ++i)
    cu0->step();
  EXPECT_EQ(c0->state(), amdgpu::WfState::GWS_WAIT)
      << "parked P resumed before its producer signaled";

  run_gws(*cu1, GwsOp::kSemaV, *producer, /*count=*/0);
  run_gws(*cu1, GwsOp::kSemaV, *producer, /*count=*/0);
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
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto cu = make_cu();
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

TEST_P(DsGwsTest, ConcurrentCrossCuSteppingIsRaceFree) {
  constexpr uint64_t kPc = 0x200000;
  GwsTestContext consumer_context(arch), producer_context(arch);
  consumer_context.mem.write32(kPc, gws_s_endpgm_word(arch));
  producer_context.mem.write32(kPc, gws_s_endpgm_word(arch));
  // Each CU owns its memory/L2; the only shared mutable object is the store, so
  // any race TSan reports is a GWS store race.
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = consumer_context.make_cu();
  auto cu1 = producer_context.make_cu();
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

  {
    auto decoder = Decoder::create(arch);
    auto *c = cu0->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
    auto *p = cu1->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(p, nullptr);
    c->set_dispatch_id(0);
    p->set_dispatch_id(1);
    cu0->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
    cu1->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/0, /*wf_count=*/1);
    for (auto *wf : {c, p}) {
      wf->set_exec(0x1);
      wf->set_m0(0);
    }
    issue_gws(*cu0, *decoder, GwsOp::kSemaP, *c, /*count=*/0);
    ASSERT_EQ(c->state(), amdgpu::WfState::GWS_WAIT);
    for (int s = 0; s < 8; ++s)
      cu0->step();
    ASSERT_EQ(c->state(), amdgpu::WfState::GWS_WAIT)
        << "consumer released before the producer signalled";
    // The producer signals: the credit crosses the store and the consumer's own CU
    // applies it on its next step.
    issue_gws(*cu1, *decoder, GwsOp::kSemaV, *p, /*count=*/0);
    for (int s = 0; s < 8 && c->state() == amdgpu::WfState::GWS_WAIT; ++s)
      cu0->step();
    EXPECT_NE(c->state(), amdgpu::WfState::GWS_WAIT) << "consumer not released by the producer's V";
    c->halt();
    p->halt();
    ASSERT_EQ(issue_failures.load(), 0);
  }

  std::thread producer(producer_fn);
  std::thread consumer(consumer_fn);
  producer.join();
  consumer.join();

  EXPECT_EQ(issue_failures.load(), 0);
  // The stress loop is race detection only; forward progress is guaranteed by the
  // deterministic handshake above, so no scheduling-dependent release count is
  // asserted here. consumed is kept to surface gross regressions under -v.
  (void)consumed;
}

// Production wiring (direct Xcd constructor): every CU of an XCD shares that
// XCD's single device-global GWS store, so cross-CU GWS works for production
// dispatches. Regression for CUs keeping their private default store.
TEST_P(DsGwsTest, XcdConstructorSharesGwsStoreAcrossCus) {
  auto cu_cfg = cu_config(4);
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

TEST_P(DsGwsTest, DirectlyAttachedCpCuSharesGwsStore) {
  auto cu_cfg = cu_config(4);
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
  auto direct_cu = make_cu();
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
  auto cu = make_cu();
  auto [producer] = make_workgroup<1>(*cu);

  // Producer signals (no waiter), leaving one credit, then its workgroup retires.
  run_gws(*cu, GwsOp::kSemaV, *producer, /*count=*/0);
  producer->halt();

  // A later workgroup's P consumes the credit that survived the retirement.
  auto [consumer] = make_workgroup<1>(*cu, {.wg_id = 1});
  run_gws(*cu, GwsOp::kSemaP, *consumer, /*count=*/0);
  EXPECT_EQ(consumer->state(), amdgpu::WfState::RUNNING);

  consumer->halt();
}

// A separate init kernel (one dispatch) seeds a semaphore credit that a later
// dispatch's P consumes -- the CLR RunGwsInit / KFDGWSTest.Semaphore sequence.
// GWS resources are keyed by process and persist across dispatches, so the seed
// survives the init dispatch's retirement.
TEST_P(DsGwsTest, SeparateInitDispatchSeedsLaterDispatch) {
  auto cu = make_cu();

  // Init dispatch (dispatch_id 0) seeds one credit, then fully retires.
  auto [initer] = make_workgroup<1>(*cu);
  run_gws(*cu, GwsOp::kInit, *initer, /*count=*/1);
  initer->halt();

  // Main dispatch (dispatch_id 1) consumes the credit seeded by the init dispatch.
  auto [user] = make_workgroup<1>(*cu, {.dispatch_id = 1});
  run_gws(*cu, GwsOp::kSemaP, *user, /*count=*/0);
  EXPECT_EQ(user->state(), amdgpu::WfState::RUNNING);

  user->halt();
}

TEST_P(DsGwsTest, AccBitSelectsAgprCountOperand) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA2 && arch != ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "DS acc bit only exists on CDNA2/3";
  auto cu = make_cu();
  auto [wf0, wf1] = make_workgroup<2>(*cu);

  const uint16_t op = gws_opcode(arch, GwsOp::kInit);
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
  cu->gws_sema_p(*wf0, 0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  cu->gws_sema_p(*wf0, 0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT);

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
  for (int i = 0; i < 5; ++i) {
    cu->gws_sema_p(*wf1, 1);
    EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  }
  cu->gws_sema_p(*wf1, 1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::GWS_WAIT);

  wf0->halt();
  wf1->halt();
}

TEST_P(DsGwsTest, BarrierRequiresExplicitInitialization) {
  auto cu = make_cu();
  auto *wave = cu->dispatch_wf(0, 0, 102, 16);
  ASSERT_NE(wave, nullptr);
  cu->begin_workgroup(0, 0, 1);
  EXPECT_THROW(cu->gws_barrier_arrive(*wave, 0, 1), std::runtime_error);
  EXPECT_FALSE(cu->has_active_wfs());
}

// A V leaves a credit for a P already parked (possibly on another CU), delivered
// by that waiter's own CU on its next poll. A fresh P must not steal that credit:
// the store reserves one credit per already-parked waiter (Resource::sema_waiters),
// so a second P issued on the signaling CU after the V but before the first waiter
// polls parks too, and the credit goes to the earlier waiter. Regression: without
// the reservation the later P would consume the V's credit and the earlier waiter
// could stall. Two CUs share one store; cross-CU wakes are poll-applied.
TEST_P(DsGwsTest, SemaphoreVReservedForEarlierParkedWaiter) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  auto [waiter0] = make_workgroup<1>(*cu0, {.pc = kPc});
  auto [producer] = make_workgroup<1>(*cu1, {.wg_id = 1, .pc = kPc});
  auto [waiter1] = make_workgroup<1>(*cu1, {.wg_id = 2, .pc = kPc});

  // waiter0 parks on CU0 with no credit -- the earlier waiter.
  run_gws(*cu0, GwsOp::kSemaP, *waiter0, /*count=*/0);
  ASSERT_EQ(waiter0->state(), amdgpu::WfState::GWS_WAIT);

  // producer V leaves one credit in the shared store (reserved for waiter0).
  run_gws(*cu1, GwsOp::kSemaV, *producer, /*count=*/0);

  // waiter1's P (same CU as the V) is issued before waiter0 polls. It must NOT
  // steal the reserved credit; it parks instead.
  run_gws(*cu1, GwsOp::kSemaP, *waiter1, /*count=*/0);
  EXPECT_EQ(waiter1->state(), amdgpu::WfState::GWS_WAIT)
      << "second P stole the credit owed to the earlier parked waiter";

  cu1->step();
  EXPECT_EQ(waiter1->state(), amdgpu::WfState::GWS_WAIT)
      << "later waiter consumed the credit when its own CU polled first";
  EXPECT_EQ(waiter0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_TRUE(producer->is_halted());

  for (int i = 0; i < 4 && waiter0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(waiter0->state(), amdgpu::WfState::GWS_WAIT);
  EXPECT_EQ(waiter1->state(), amdgpu::WfState::GWS_WAIT);
  // waiter1 left parked; cu1's unregister reconciles it on destruction.
}

TEST_P(DsGwsTest, BarrierRendezvousWithLateAdmittedParticipant) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);
  // Only CU0's wave is resident so far, but the grid has two waves.
  auto [wf0] = make_workgroup<1>(*cu0, {.pc = kPc});

  run_gws(*cu0, GwsOp::kInit, *wf0, /*count=*/1);
  run_gws(*cu0, GwsOp::kBarrier, *wf0, /*count=*/1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::GWS_WAIT)
      << "early arrival fell back to a no-op instead of awaiting the late participant";

  // The second participant is admitted on CU1 and arrives: it releases the peer.
  auto [wf1] = make_workgroup<1>(*cu1, {.wg_id = 1});
  run_gws(*cu1, GwsOp::kBarrier, *wf1, /*count=*/1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  for (int i = 0; i < 4 && wf0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(wf0->state(), amdgpu::WfState::GWS_WAIT);

  wf1->halt();
}

TEST_P(DsGwsTest, RemoteReleaseDoesNotReleaseUnrelatedResource) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu_a = make_cu();
  auto cu_b = make_cu();
  auto cu_c = make_cu();
  cu_a->set_gws_device(gws);
  cu_b->set_gws_device(gws);
  cu_c->set_gws_device(gws);

  auto [waiter_a] = make_workgroup<1>(*cu_a, {.pc = kPc});
  auto [waiter_b] = make_workgroup<1>(*cu_b, {.wg_id = 1, .pc = kPc});
  auto [producer_c] = make_workgroup<1>(*cu_c, {.wg_id = 2, .pc = kPc});

  // waiter_a parks on rid 0; waiter_b parks on rid 1 (offset0 selects the rid).
  run_gws(*cu_a, GwsOp::kSemaP, *waiter_a, /*count=*/0, /*offset0=*/0);
  run_gws(*cu_b, GwsOp::kSemaP, *waiter_b, /*count=*/0, /*offset0=*/1);
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  cu_a->step();
  cu_b->step();
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  // CU C signals waiter_b's resource (leaving a credit B will consume on its next
  // poll) and then goes idle. The pending credit will make waiter_b runnable.
  run_gws(*cu_c, GwsOp::kSemaV, *producer_c, /*count=*/0, /*offset0=*/1);
  producer_c->halt();

  cu_a->step();
  EXPECT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT)
      << "waiter A was released before its peer could signal";

  // B now polls, consumes the credit, and wakes -- confirming the deferred signaler
  // was real.
  for (int i = 0; i < 4 && waiter_b->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu_b->step();
  EXPECT_NE(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  waiter_a->halt();
}

TEST_P(DsGwsTest, RemoteInitDoesNotReleaseUnrelatedResource) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu_a = make_cu();
  auto cu_b = make_cu();
  auto cu_c = make_cu();
  cu_a->set_gws_device(gws);
  cu_b->set_gws_device(gws);
  cu_c->set_gws_device(gws);

  auto [waiter_a] = make_workgroup<1>(*cu_a, {.pc = kPc});
  auto [waiter_b] = make_workgroup<1>(*cu_b, {.wg_id = 1, .pc = kPc});
  auto [producer_c] = make_workgroup<1>(*cu_c, {.wg_id = 2, .pc = kPc});

  run_gws(*cu_a, GwsOp::kSemaP, *waiter_a, /*count=*/0, /*offset0=*/0);
  run_gws(*cu_b, GwsOp::kSemaP, *waiter_b, /*count=*/0, /*offset0=*/1);
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  cu_a->step();
  cu_b->step();
  ASSERT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_EQ(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  // CU C seeds a credit on waiter_b's resource via init and then goes idle.
  run_gws(*cu_c, GwsOp::kInit, *producer_c, /*count=*/1, /*offset0=*/1);
  producer_c->halt();

  // Step A before B. The credit belongs to B, so A must remain parked.
  cu_a->step();
  EXPECT_EQ(waiter_a->state(), amdgpu::WfState::GWS_WAIT)
      << "waiter A was released before its peer could signal";

  // B now polls, consumes the seeded credit, and wakes.
  for (int i = 0; i < 4 && waiter_b->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu_b->step();
  EXPECT_NE(waiter_b->state(), amdgpu::WfState::GWS_WAIT);

  waiter_a->halt();
}

TEST_P(DsGwsTest, DestroyingCuReconcilesParkedWaveInSharedStore) {
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  ASSERT_FALSE(gws->any_parked());
  {
    auto cu = make_cu();
    cu->set_gws_device(gws);
    auto [wf] = make_workgroup<1>(*cu);
    // Park a semaphore P (no credit), then let the CU go out of scope unwoken.
    run_gws(*cu, GwsOp::kSemaP, *wf, /*count=*/0);
    ASSERT_EQ(wf->state(), amdgpu::WfState::GWS_WAIT);
    ASSERT_TRUE(gws->any_parked());
  }
  // The CU's destructor unregistered it and reconciled the parked wave out.
  EXPECT_FALSE(gws->any_parked());
}

// Re-pointing a CU to a different GWS store is only safe while the CU is drained: a
// GWS resource's phase (barrier generation / semaphore credits) lives in the store,
// so a live parked waiter cannot carry its rendezvous across stores -- migrating it
// would compare its stale generation against the destination's unrelated one and
// wake it without its peers arriving. set_gws_device must therefore REJECT a
// replacement requested while the CU still hosts a parked wave, keeping it on its
// current store, rather than performing an unsafe partial migration. The destination
// here is populated (another CU's wave is parked in store B), proving the rejected
// attempt leaves both stores' bookkeeping untouched. Once the CU is drained the
// re-point is accepted. Regression: migrating the live waiter corrupted the
// destination's parked accounting and could spuriously wake its unrelated waiter.
TEST_P(DsGwsTest, ReplacingLiveCuStoreIsRejectedUntilDrained) {
  auto store_a = std::make_shared<amdgpu::GwsDevice>();
  auto store_b = std::make_shared<amdgpu::GwsDevice>();

  // A resident waiter already parked in store B on its own CU (populated destination).
  auto cu_b = make_cu();
  ASSERT_TRUE(cu_b->set_gws_device(store_b)); // Drained CU: accepted.
  auto [wf_b] = make_workgroup<1>(*cu_b, {.dispatch_id = 1});
  run_gws(*cu_b, GwsOp::kSemaP, *wf_b, /*count=*/0);
  ASSERT_EQ(wf_b->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_TRUE(store_b->any_parked());

  // A second CU parks a waiter in store A.
  auto cu = make_cu();
  ASSERT_TRUE(cu->set_gws_device(store_a)); // Drained CU: accepted.
  auto [wf] = make_workgroup<1>(*cu);
  run_gws(*cu, GwsOp::kSemaP, *wf, /*count=*/0);
  ASSERT_EQ(wf->state(), amdgpu::WfState::GWS_WAIT);
  ASSERT_TRUE(store_a->any_parked());

  // Attempt to move the live CU to store B: rejected. The CU stays on store A, and
  // store B gains nothing (its own waiter is undisturbed).
  EXPECT_FALSE(cu->set_gws_device(store_b))
      << "re-pointing a CU with a parked GWS waiter must be rejected";
  EXPECT_EQ(&cu->gws_device(), store_a.get());
  EXPECT_TRUE(store_a->any_parked());

  // Drain the CU (its waiter retires on store A), then the re-point is accepted.
  cu->abort_dispatch(/*dispatch_id=*/0);
  EXPECT_TRUE(wf->is_halted());
  EXPECT_FALSE(store_a->any_parked());
  EXPECT_TRUE(store_b->any_parked())
      << "retiring store A's waiter disturbed store B's unrelated parked waiter";
  EXPECT_TRUE(cu->set_gws_device(store_b)) << "a drained CU must accept a re-point";
  EXPECT_EQ(&cu->gws_device(), store_b.get());

  // Halt cu_b's waiter before teardown so it is no longer GWS_WAIT.
  wf_b->halt();
}

// sema_release_all must retire the released waiters' credit reservations, not just
// bump the release generation. A release_all wakes every queued P-waiter through the
// generation (each on its own CU's next step), so their arrival stamps no longer
// entitle them to a credit; leaving the stamps queued makes a later V's credit be
// reserved for a wave that will never consume it, stranding a fresh P that should
// take it. Ordering matters: CU0 parks a P, then CU1 runs release_all, V, and a new
// P all before CU0 ever polls (so CU0's waiter has not yet been woken out of the
// queue). Regression: the new P parked behind the released waiter's stale
// reservation despite the available credit, on all seven ISAs.
TEST_P(DsGwsTest, ReleaseAllRetiresCreditReservationForDelayedPeerPoll) {
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(arch));
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu0 = make_cu();
  auto cu1 = make_cu();
  cu0->set_gws_device(gws);
  cu1->set_gws_device(gws);

  // CU0 parks a P waiter on the resource (no credit): it queues an arrival stamp.
  auto [p0] = make_workgroup<1>(*cu0, {.pc = kPc});
  run_gws(*cu0, GwsOp::kSemaP, *p0, /*count=*/0);
  ASSERT_EQ(p0->state(), amdgpu::WfState::GWS_WAIT);

  // CU1 runs release_all, then V, then a NEW P -- all before CU0 polls. release_all
  // wakes p0 through the generation (applied on CU0's own next step), so its credit
  // reservation must be retired now; the V credit then belongs to the new P.
  auto *rel = cu1->dispatch_wf(/*wg_id=*/1, /*pc=*/0, 102, 16);
  auto *p1 = cu1->dispatch_wf(/*wg_id=*/1, kPc, 102, 16);
  ASSERT_NE(rel, nullptr);
  ASSERT_NE(p1, nullptr);
  rel->set_dispatch_id(1);
  p1->set_dispatch_id(1);
  cu1->begin_workgroup(/*dispatch_id=*/1, /*wg_id=*/1, /*wf_count=*/2);
  for (auto *wf : {rel, p1}) {
    wf->set_exec(0x1);
    wf->set_m0(0);
  }
  run_gws(*cu1, GwsOp::kReleaseAll, *rel, /*count=*/0);
  run_gws(*cu1, GwsOp::kSemaV, *rel, /*count=*/0);
  run_gws(*cu1, GwsOp::kSemaP, *p1, /*count=*/0);

  // Core regression: the new P took the V credit instead of parking behind p0's
  // released-but-still-queued reservation.
  EXPECT_EQ(p1->state(), amdgpu::WfState::RUNNING)
      << "new P parked behind a released waiter's stale credit reservation";

  // p0 wakes on CU0's own next step via the release generation (not a credit) and
  // retires at its s_endpgm.
  for (int i = 0; i < 8 && p0->state() == amdgpu::WfState::GWS_WAIT; ++i)
    cu0->step();
  EXPECT_NE(p0->state(), amdgpu::WfState::GWS_WAIT);

  rel->halt();
  p1->halt();
}

// GWS resources persist across dispatches but belong to a process; on process
// teardown (SimulatedKfd::close -> GwsDevice::release_process) that process's table
// is dropped while other live processes keep theirs. Two processes each seed a
// semaphore credit; releasing process A drops only A's table (a later P on A then
// parks for lack of a credit), while B's seeded credit survives and is consumed.
TEST_P(DsGwsTest, ReleaseProcessDropsOnlyThatProcessTable) {
  auto gws = std::make_shared<amdgpu::GwsDevice>();
  auto cu = make_cu(8);
  cu->set_gws_device(gws);
  constexpr uint32_t kProcA = 11;
  constexpr uint32_t kProcB = 22;

  auto [init_a] = make_workgroup<1>(*cu, {.process_id = kProcA});
  auto [init_b] = make_workgroup<1>(*cu, {.dispatch_id = 1, .wg_id = 1, .process_id = kProcB});
  // Both processes seed a credit; the store now tracks two process tables.
  run_gws(*cu, GwsOp::kInit, *init_a, /*count=*/1);
  run_gws(*cu, GwsOp::kInit, *init_b, /*count=*/1);
  EXPECT_EQ(gws->tracked_process_count(), 2u);
  init_a->halt();
  init_b->halt();

  // Process A tears down: only its table is dropped.
  gws->release_process(kProcA);
  EXPECT_EQ(gws->tracked_process_count(), 1u);

  // A fresh P on process A finds no credit (its table was released) and parks.
  auto [p_a] = make_workgroup<1>(*cu, {.dispatch_id = 2, .wg_id = 2, .process_id = kProcA});
  run_gws(*cu, GwsOp::kSemaP, *p_a, /*count=*/0);
  EXPECT_EQ(p_a->state(), amdgpu::WfState::GWS_WAIT)
      << "released process A still had a lingering credit";

  // Process B was untouched: its seeded credit is still consumable.
  auto [p_b] = make_workgroup<1>(*cu, {.dispatch_id = 3, .wg_id = 3, .process_id = kProcB});
  run_gws(*cu, GwsOp::kSemaP, *p_b, /*count=*/0);
  EXPECT_EQ(p_b->state(), amdgpu::WfState::RUNNING)
      << "process B lost its credit when A was released";

  p_b->halt();
}

TEST_P(DsGwsTest, SemaphoreAndBarrierShareCounterAcrossTypeChanges) {
  auto cu = make_cu();
  auto [wave] = make_workgroup<1>(*cu);
  run_gws(*cu, GwsOp::kInit, *wave, /*count=*/1);
  run_gws(*cu, GwsOp::kSemaP, *wave, /*count=*/0);
  ASSERT_EQ(wave->state(), amdgpu::WfState::RUNNING);
  // P consumed the shared count: this arrival releases and reloads two credits.
  run_gws(*cu, GwsOp::kBarrier, *wave, /*count=*/2);
  ASSERT_EQ(wave->state(), amdgpu::WfState::RUNNING);
  for (int i = 0; i < 2; ++i) {
    run_gws(*cu, GwsOp::kSemaP, *wave, /*count=*/0);
    ASSERT_EQ(wave->state(), amdgpu::WfState::RUNNING);
  }
  run_gws(*cu, GwsOp::kSemaP, *wave, /*count=*/0);
  EXPECT_EQ(wave->state(), amdgpu::WfState::GWS_WAIT);
  wave->halt();
}

TEST_P(DsGwsTest, SemaphoreSignalsAreVisibleToBarrier) {
  for (auto op : {GwsOp::kSemaV, GwsOp::kSemaBr}) {
    SCOPED_TRACE(static_cast<int>(op));
    GwsTestContext context(arch);
    auto cu = context.make_cu();
    auto [wave] = make_workgroup<1>(*cu);
    context.run_gws(*cu, GwsOp::kInit, *wave, /*count=*/0);
    context.run_gws(*cu, op, *wave, /*count=*/1);
    context.run_gws(*cu, GwsOp::kBarrier, *wave, /*count=*/0);
    EXPECT_EQ(wave->state(), amdgpu::WfState::GWS_WAIT);
    wave->halt();
  }
}

TEST_P(DsGwsTest, ReleaseAllClearsCounterBeforeBarrier) {
  auto cu = make_cu();
  auto [wave] = make_workgroup<1>(*cu);
  run_gws(*cu, GwsOp::kInit, *wave, /*count=*/3);
  run_gws(*cu, GwsOp::kReleaseAll, *wave, /*count=*/0);
  run_gws(*cu, GwsOp::kBarrier, *wave, /*count=*/0);
  EXPECT_EQ(wave->state(), amdgpu::WfState::RUNNING);
  wave->halt();
}

TEST_P(DsGwsTest, UnlimitedQuantumYieldsWithGwsAndWorkgroupBarrierWaiters) {
  auto cu = make_cu();
  auto peer_cu = make_cu();
  auto store = std::make_shared<amdgpu::GwsDevice>();
  ASSERT_TRUE(cu->set_gws_device(store));
  ASSERT_TRUE(peer_cu->set_gws_device(store));
  cu->set_functional_quantum(0);
  cu->set_gws_wait_limit(8);
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, build_s_barrier(arch)[0]);
  mem.write32(kPc + 4, gws_s_endpgm_word(arch));
  auto *waiter = cu->dispatch_wf(/*wg_id=*/0, kPc, 102, 16);
  auto *sibling = cu->dispatch_wf(/*wg_id=*/0, kPc + 4, 102, 16);
  ASSERT_NE(waiter, nullptr);
  ASSERT_NE(sibling, nullptr);
  cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/2);
  cu->gws_init(*waiter, 0, 1);
  cu->gws_barrier_arrive(*waiter, 0, 1);
  sibling->set_state(amdgpu::WfState::BARRIER);
  // This is the second s_barrier in a multi-wave cooperative grid.sync().
  for (int i = 0; i < 3; ++i) {
    const auto result = cu->run_quantum();
    EXPECT_TRUE(result.yielded);
    EXPECT_EQ(result.iterations, 1u);
    EXPECT_EQ(waiter->state(), amdgpu::WfState::GWS_WAIT);
    EXPECT_EQ(sibling->state(), amdgpu::WfState::BARRIER);
  }
  auto *peer = peer_cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, 102, 16);
  ASSERT_NE(peer, nullptr);
  peer_cu->begin_workgroup(/*dispatch_id=*/0, /*wg_id=*/0, /*wf_count=*/1);
  peer_cu->gws_barrier_arrive(*peer, 0, 1);
  EXPECT_NO_THROW((void)cu->run_quantum());
  EXPECT_TRUE(waiter->is_halted());
  EXPECT_TRUE(sibling->is_halted());
  EXPECT_FALSE(store->any_parked());
  peer->halt();
}

TEST_P(DsGwsTest, UnlimitedQuantumYieldsToPeerBarrierArrival) {
  auto waiter_cu = make_cu();
  auto producer_cu = make_cu();
  auto store = std::make_shared<amdgpu::GwsDevice>();
  ASSERT_TRUE(waiter_cu->set_gws_device(store));
  ASSERT_TRUE(producer_cu->set_gws_device(store));
  waiter_cu->set_functional_quantum(0);
  waiter_cu->set_gws_wait_limit(8);
  constexpr uint64_t kPc = 0x200000;
  mem.write32(kPc, gws_s_endpgm_word(GetParam()));
  auto *waiter = waiter_cu->dispatch_wf(0, kPc, 102, 16);
  ASSERT_NE(waiter, nullptr);
  waiter_cu->begin_workgroup(0, 0, 1);
  waiter_cu->gws_init(*waiter, 0, 1);
  waiter_cu->gws_barrier_arrive(*waiter, 0, 1);
  for (int i = 0; i < 3; ++i) {
    amdgpu::FunctionalQuantumResult result{};
    ASSERT_NO_THROW(result = waiter_cu->run_quantum());
    EXPECT_TRUE(result.yielded);
    EXPECT_EQ(result.iterations, 1u);
    EXPECT_EQ(result.next_tick_delay(), amdgpu::ComputeUnitCore::kFunctionalQuantum);
    EXPECT_EQ(waiter->state(), amdgpu::WfState::GWS_WAIT);
  }
  auto *producer = producer_cu->dispatch_wf(0, 0, 102, 16);
  ASSERT_NE(producer, nullptr);
  producer_cu->begin_workgroup(0, 0, 1);
  producer_cu->gws_barrier_arrive(*producer, 0, 1);
  EXPECT_NO_THROW((void)waiter_cu->run_quantum());
  EXPECT_TRUE(waiter->is_halted());
  EXPECT_FALSE(store->any_parked());
  producer->halt();
}

TEST_P(DsGwsTest, ParkedWaiterDoesNotShortenResidentProducerQuantum) {
  auto cu = make_cu();
  cu->set_functional_quantum(16);
  cu->set_gws_wait_limit(4);
  constexpr uint64_t kPc = 0x200000;
  for (int i = 0; i < 32; ++i)
    mem.write32(kPc + 4 * i, 0xbf800000); // s_nop 0.
  auto *waiter = cu->dispatch_wf(0, kPc, 102, 16);
  auto *producer = cu->dispatch_wf(1, kPc, 102, 16);
  ASSERT_NE(waiter, nullptr);
  ASSERT_NE(producer, nullptr);
  cu->begin_workgroup(0, 0, 1);
  cu->begin_workgroup(0, 1, 1);
  cu->gws_sema_p(*waiter, 0);
  const auto result = cu->run_quantum();
  EXPECT_EQ(result.iterations, 16u);
  EXPECT_FALSE(result.yielded);
  EXPECT_EQ(producer->pc, kPc + 64);
  EXPECT_EQ(waiter->state(), amdgpu::WfState::GWS_WAIT);
  producer->halt();
  waiter->halt();
}

TEST_P(DsGwsTest, OnlySameProcessInstructionProgressExtendsWaitBudget) {
  for (bool same_process : {false, true}) {
    SCOPED_TRACE(same_process);
    GwsTestContext context(arch);
    auto waiter_cu = context.make_cu();
    auto producer_cu = context.make_cu();
    auto store = std::make_shared<amdgpu::GwsDevice>();
    ASSERT_TRUE(waiter_cu->set_gws_device(store));
    ASSERT_TRUE(producer_cu->set_gws_device(store));
    waiter_cu->set_gws_wait_limit(4);
    constexpr uint64_t kPc = 0x200000;
    // Real independent instructions continue long beyond the wait limit.
    for (int i = 0; i < 32; ++i)
      context.mem.write32(kPc + 4 * i, 0xbf800000); // s_nop 0.
    context.mem.write32(kPc + 0x100, gws_s_endpgm_word(GetParam()));
    auto *waiter = waiter_cu->dispatch_wf(0, kPc + 0x100, 102, 16);
    auto *producer = producer_cu->dispatch_wf(0, kPc, 102, 16);
    ASSERT_NE(waiter, nullptr);
    ASSERT_NE(producer, nullptr);
    waiter->set_process_id(same_process ? 0 : 1);
    producer->set_process_id(0);
    waiter_cu->begin_workgroup(0, 0, 1);
    producer_cu->begin_workgroup(0, 0, 1);
    waiter_cu->gws_sema_p(*waiter, 0);
    for (int i = 0; i < (same_process ? 16 : 3); ++i) {
      ASSERT_NO_THROW(producer_cu->step());
      ASSERT_EQ(producer->pc, kPc + 4 * (i + 1));
      ASSERT_NO_THROW(waiter_cu->step());
      EXPECT_EQ(waiter->state(), amdgpu::WfState::GWS_WAIT);
    }
    if (same_process) {
      producer_cu->gws_sema_v(*producer, 0);
      ASSERT_NO_THROW(waiter_cu->step());
      EXPECT_NE(waiter->state(), amdgpu::WfState::GWS_WAIT);
      waiter->halt();
    } else {
      ASSERT_NO_THROW(producer_cu->step());
      EXPECT_THROW(waiter_cu->step(), std::runtime_error);
      EXPECT_TRUE(waiter->is_halted());
    }
    producer->halt();
  }
}

TEST_P(DsGwsTest, WaitBudgetPausesAndCanBeDisabled) {
  auto cu = make_cu();
  auto *wave = cu->dispatch_wf(0, 0, 102, 16);
  ASSERT_NE(wave, nullptr);
  cu->begin_workgroup(0, 0, 1);
  cu->set_gws_wait_limit(2);
  cu->gws_sema_p(*wave, 0);
  wave->set_debug_suspended(true);
  for (int i = 0; i < 8; ++i)
    EXPECT_NO_THROW(cu->step());
  wave->set_debug_suspended(false);
  wave->set_runtime_suspended(true);
  for (int i = 0; i < 8; ++i)
    EXPECT_NO_THROW(cu->step());
  wave->set_runtime_suspended(false);
  EXPECT_NO_THROW(cu->step());
  EXPECT_EQ(wave->state(), amdgpu::WfState::GWS_WAIT);
  cu->set_gws_wait_limit(0);
  for (int i = 0; i < 8; ++i)
    EXPECT_NO_THROW(cu->step());
  EXPECT_EQ(wave->state(), amdgpu::WfState::GWS_WAIT);
  cu->gws_sema_v(*wave, 0);
  EXPECT_EQ(wave->state(), amdgpu::WfState::RUNNING);
  wave->halt();
}

TEST_P(DsGwsTest, XcdAndCpRejectStoreReplacementBeforeAnyMutation) {
  auto cfg = cu_config(4);
  amdgpu::ShaderEngine::Config se{};
  se.num_compute_units = 2;
  se.cus_per_shader_array = 2;
  se.compute_unit = cfg;
  amdgpu::Xcd::Config xc{};
  xc.num_shader_engines = 1;
  xc.shader_engine = se;
  amdgpu::Xcd xcd("xcd", xc, GetParam(), &mem);
  auto old = xcd.gws_device();
  auto replacement = std::make_shared<amdgpu::GwsDevice>();
  auto *cu0 = xcd.shader_engine(0)->compute_unit(0);
  auto *cu1 = xcd.shader_engine(0)->compute_unit(1);
  auto *wave = cu1->dispatch_wf(0, 0, 102, 16);
  ASSERT_NE(wave, nullptr);
  cu1->begin_workgroup(0, 0, 1);
  EXPECT_THROW(xcd.adopt_gws_store(replacement), std::logic_error);
  EXPECT_THROW(xcd.command_processor()->set_gws_device(replacement), std::logic_error);
  EXPECT_EQ(xcd.gws_device(), old);
  EXPECT_EQ(&cu0->gws_device(), old.get());
  EXPECT_EQ(&cu1->gws_device(), old.get());
  wave->halt();
  EXPECT_NO_THROW(xcd.adopt_gws_store(replacement));
  EXPECT_EQ(&cu0->gws_device(), replacement.get());
  EXPECT_EQ(&cu1->gws_device(), replacement.get());
}

} // namespace
