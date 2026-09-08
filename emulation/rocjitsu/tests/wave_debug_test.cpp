// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wave_debug_test.cpp
/// @brief Wave-level KFD debugger groundwork, exercised through the compute unit.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/spi.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rocjitsu/kmd/linux/cwsr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace {

using namespace rocjitsu;

constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 64;
constexpr uint32_t kKernelAddr = 0x1000;
constexpr uint32_t kTrapHandlerAddr = 0x2000;
constexpr uint32_t kSTrapBreakpoint = 0xBF920001u; // s_trap 1 (rocm-dbgapi breakpoint)
constexpr uint32_t kSTrapSeven = 0xBF92AB07u;      // s_trap 0xab07 (trap id is low 8 bits)
constexpr uint32_t kGfx12STrap = 0xBF900003u;      // s_trap 3 on gfx12+
constexpr uint32_t kSEndpgm = 0xBF810000u;         // s_endpgm

struct WaveDebugFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;

  explicit WaveDebugFixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4)
      : gpu_mem("wave_debug_mem"), l2("wave_debug_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_wave_debug", cfg, &gpu_mem, &l2);
  }

  amdgpu::Wavefront *dispatch(uint64_t pc) {
    wf = cu->dispatch_wf(0, pc, SGPRS_PER_WF, VGPRS_PER_WF);
    return wf;
  }
};

TEST(WaveDebugTest, STrapExecutesConfiguredTrapHandlerInstructions) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // Assembled for gfx950. The handler writes a proof value to TTMP4,
  // advances the saved PC, restores STATUS.HALT, emits MSG_INTERRUPT, and
  // returns through s_rfe_b64.
  const uint32_t handler[] = {
      0xBEF000FFu, 0x12345678u, // s_mov_b32 ttmp4, 0x12345678
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEF800FFu, 0x00002000u, // s_mov_b32 ttmp12, STATUS.HALT
      0xBF900001u,              // s_sendmsg sendmsg(MSG_INTERRUPT)
      0xB978F802u,              // s_setreg_b32 hwreg(HW_REG_STATUS), ttmp12
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x3000, true};
  });
  uint32_t interrupt_count = 0;
  fx.cu->set_sendmsg_handler([&](amdgpu::Wavefront &, uint32_t message) {
    if (message == 1)
      ++interrupt_count;
    return message == 1;
  });
  uint32_t completion_count = 0;
  fx.cu->set_trap_completion_handler([&](amdgpu::Wavefront &) { ++completion_count; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_EQ(wf->ttmp(0), kKernelAddr);
  EXPECT_EQ(wf->ttmp(1), 1u << 16);
  EXPECT_EQ(wf->ttmp(14), 0x3000u);

  for (int i = 0; i < 7; ++i)
    fx.cu->step();

  EXPECT_FALSE(wf->in_trap_handler());
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_EQ(wf->ttmp(4), 0x12345678u);
  EXPECT_EQ(interrupt_count, 1u);
  EXPECT_EQ(completion_count, 1u);
}

// The ROCr handler temporarily reuses EXEC_LO for MSG_GET_DOORBELL. The wave
// must be reported with the interrupted lane mask, not the handler's doorbell
// response, or resuming a breakpoint changes which lanes execute user code.
TEST(WaveDebugTest, TrapHandlerRestoresInterruptedExecBeforeReporting) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  const uint32_t handler[] = {
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEFE00FFu, 0x80000000u, // s_mov_b32 exec_lo, 0x80000000
      0xBF90000Au,              // s_sendmsg sendmsg(MSG_GET_DOORBELL)
      0xBEF800FFu, 0x00002000u, // s_mov_b32 ttmp12, STATUS.HALT
      0xBF900001u,              // s_sendmsg sendmsg(MSG_INTERRUPT)
      0xB978F802u,              // s_setreg_b32 hwreg(HW_REG_STATUS), ttmp12
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &wave, uint32_t message) {
    if (message == 10)
      wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | 7u);
    return message == 1 || message == 10;
  });

  constexpr uint64_t kInterruptedExec = 0x00FF00FF00FF00FFULL;
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kInterruptedExec);

  for (int i = 0; i < 9 && !wf->debug_halted(); ++i)
    fx.cu->step();

  ASSERT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->exec(), kInterruptedExec);
}

// Restoring the interrupted EXEC belongs to returning from the handler, not to
// stopping for a debugger. A handler that returns without stopping the wave
// used to leave its own mask installed, and nothing later put the
// application's back: the kernel ran on with every lane active, silently
// un-diverging a branch. gdb.rocm/lane-info.exp catches it as lanes that had
// converged out of a branch being reported active again.
TEST(WaveDebugTest, TrapHandlerRestoresInterruptedExecWhenReturningWithoutStopping) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // Same shape as the handler above with the halt removed, so the wave runs on
  // after s_rfe instead of stopping.
  const uint32_t handler[] = {
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEFE00FFu, 0x80000000u, // s_mov_b32 exec_lo, 0x80000000
      0xBF90000Au,              // s_sendmsg sendmsg(MSG_GET_DOORBELL)
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &wave, uint32_t message) {
    if (message == 10)
      wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | 7u);
    return message == 1 || message == 10;
  });

  // Lanes 0, 2 and 4 of a five-lane wave, the shape lane-info.exp produces.
  constexpr uint64_t kInterruptedExec = 0x15ULL;
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kInterruptedExec);

  for (int i = 0; i < 6 && wf->pc == kKernelAddr; ++i)
    fx.cu->step();
  // Every assertion below is satisfied by a wave that never entered the handler
  // at all -- EXEC is unmodified, nothing halts it for the debugger, and
  // s_endpgm ends it. Anchor on the handler actually running, or a regression
  // that stops dispatching into it passes this test.
  ASSERT_TRUE(wf->in_trap_handler());
  for (int i = 0; i < 6 && wf->in_trap_handler(); ++i)
    fx.cu->step();

  ASSERT_FALSE(wf->in_trap_handler());
  EXPECT_FALSE(wf->debug_halted());
  EXPECT_EQ(wf->exec(), kInterruptedExec);

  // Run the wave off the end. Leaving one resident past the fixture is not
  // inert: the CU pool-allocates decoded instructions and frees the pool with
  // the decoder.
  for (int i = 0; i < 4 && !wf->is_halted(); ++i)
    fx.cu->step();
  EXPECT_TRUE(wf->is_halted());
}

TEST(WaveDebugTest, UnmappedInstructionFetchReportsMemoryViolationAtBranchTarget) {
  WaveDebugFixture fx;
  constexpr uint32_t kProcessId = 7;
  constexpr uint64_t kBranchTarget = 0x100;
  std::vector<uint8_t> kernel_page(amdgpu::GpuMemory::PAGE_SIZE);
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xBE801D00u, kProcessId); // s_setpc_b64 s[0:1]
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kProcessId);
  fx.cu->write_sgpr(wf->sgpr_alloc().base, static_cast<uint32_t>(kBranchTarget));
  fx.cu->write_sgpr(wf->sgpr_alloc().base + 1, 0);

  uint64_t fault_address = 0;
  fx.cu->set_memory_violation_handler(
      [&](amdgpu::Wavefront &faulting_wave, uint64_t address, bool is_write) {
        EXPECT_EQ(&faulting_wave, wf);
        EXPECT_FALSE(is_write);
        fault_address = address;
        faulting_wave.debug_trap(0);
        return true;
      });

  fx.cu->step();
  EXPECT_EQ(wf->pc, kBranchTarget);
  EXPECT_FALSE(wf->debug_halted());

  fx.cu->step();
  EXPECT_EQ(fault_address, kBranchTarget);
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kBranchTarget);

  fx.gpu_mem.unregister_process(kProcessId);
}

TEST(WaveDebugTest, UnmappedScalarLoadReportsMemoryViolationAfterInstruction) {
  WaveDebugFixture fx;
  constexpr uint32_t kProcessId = 7;
  std::vector<uint8_t> kernel_page(amdgpu::GpuMemory::PAGE_SIZE);
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xC0020000u, kProcessId);     // s_load_dword s0, s[0:1]
  fx.gpu_mem.write32(kKernelAddr + 4, 0, kProcessId);           // immediate offset 0
  fx.gpu_mem.write32(kKernelAddr + 8, 0xBF800000u, kProcessId); // s_nop 0
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(kProcessId);
  fx.cu->set_debug_active(true);

  uint64_t fault_address = ~uint64_t{0};
  fx.cu->set_memory_violation_handler(
      [&](amdgpu::Wavefront &faulting_wave, uint64_t address, bool is_write) {
        EXPECT_EQ(&faulting_wave, wave);
        EXPECT_FALSE(is_write);
        fault_address = address;
        faulting_wave.debug_trap(0);
        return true;
      });

  fx.cu->step();
  EXPECT_EQ(fault_address, 0u);
  EXPECT_TRUE(wave->debug_halted());
  EXPECT_EQ(wave->pc, kKernelAddr + 8);

  fx.gpu_mem.unregister_process(kProcessId);
}

// Runtime suspension (queue_percentage 0) and debugger suspension overlap in
// time and are lifted by different actors. Representing both with one wave bit
// let a runtime resume clear a debugger pause, and a debugger or CWSR resume
// clear an active runtime pause -- in either case the wave started running
// while something still meant it to be stopped.
TEST(WaveDebugTest, RuntimeAndDebuggerSuspensionDoNotClobberEachOther) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSEndpgm);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  ASSERT_FALSE(wave->debug_paused());

  // Both reasons applied; releasing either one alone must not resume the wave.
  wave->set_runtime_suspended(true);
  wave->set_debug_suspended(true);
  EXPECT_TRUE(wave->debug_paused());

  wave->set_debug_suspended(false);
  EXPECT_TRUE(wave->runtime_suspended());
  EXPECT_TRUE(wave->debug_paused()) << "a debugger resume lifted the runtime pause";

  wave->set_runtime_suspended(false);
  EXPECT_FALSE(wave->debug_paused());

  // And the other way round.
  wave->set_runtime_suspended(true);
  wave->set_debug_suspended(true);
  wave->set_runtime_suspended(false);
  EXPECT_TRUE(wave->debug_suspended());
  EXPECT_TRUE(wave->debug_paused()) << "a runtime resume lifted the debugger pause";

  // A halt is a third, independent reason.
  wave->set_debug_suspended(false);
  wave->set_debug_halted(true);
  EXPECT_TRUE(wave->debug_paused());
}

// Trap entry and CWSR must agree on where a wave's dispatch identity lives, or
// rocm-dbgapi correlates the stopped wave to the wrong workgroup. On the gfx9
// layout that is TTMP8/9/10 = workgroup id x/y/z, which is exactly what
// cwsr.cpp serializes; trap entry used to write the flat wg_id() into TTMP8 and
// zero the other two.
TEST(WaveDebugTest, TrapEntryPublishesTheWorkgroupCoordinateCwsrSerializes) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);
  fx.gpu_mem.write32(kTrapHandlerAddr, kSEndpgm);
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_wg_coord(3, 5, 7);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());

  const auto &wg = wf->wg_coord();
  EXPECT_EQ(wf->ttmp(8), wg[0]);
  EXPECT_EQ(wf->ttmp(9), wg[1]);
  EXPECT_EQ(wf->ttmp(10), wg[2]);
  EXPECT_EQ(wf->ttmp(9), 5u) << "the y coordinate was zeroed on trap entry";
  EXPECT_EQ(wf->ttmp(10), 7u) << "the z coordinate was zeroed on trap entry";
}

// S_SENDMSGHALT halts the wave -- that is the instruction, not a debugger
// artefact. It has to publish the architectural STATUS.HALT and not only the
// scheduler's private flag, because s_rfe reads STATUS.HALT on the way out of
// the handler to decide whether the wave stays stopped. With only the private
// flag set, s_rfe saw 0 for a wave s_sendmsghalt had already halted and left
// single-step armed, so resuming took an extra step.
TEST(WaveDebugTest, SendmsghaltPublishesArchitecturalStatusHalt) {
  WaveDebugFixture fx;
  constexpr uint32_t kStatusHalt = 1u << 13;
  constexpr uint32_t kSSendmsghaltInterrupt = 0xBF910001u; // s_sendmsghalt sendmsg(MSG_INTERRUPT)

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  const uint32_t handler[] = {
      0x806C846Cu,            // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du,            // s_addc_u32 ttmp1, ttmp1, 0
      kSSendmsghaltInterrupt, // s_sendmsghalt sendmsg(MSG_INTERRUPT)
      0xBE801F6Cu,            // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &, uint32_t message) { return message == 1; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_debug_single_step(true);

  fx.cu->step(); // s_trap -> handler
  ASSERT_TRUE(wf->in_trap_handler());
  ASSERT_EQ(wf->status_raw() & kStatusHalt, 0u) << "STATUS.HALT set before s_sendmsghalt";

  for (int i = 0; i < 3 && !wf->debug_halted(); ++i)
    fx.cu->step();

  EXPECT_TRUE(wf->debug_halted());
  EXPECT_NE(wf->status_raw() & kStatusHalt, 0u)
      << "s_sendmsghalt halted the wave without publishing STATUS.HALT";
  // Which instruction raised the bit is the only thing that separates this stop
  // from a handler that raised HALT with s_setreg and then returned. The two
  // want opposite treatment on resume and are identical in the CWSR record.
  EXPECT_TRUE(wf->self_halted())
      << "s_sendmsghalt raised STATUS.HALT without recording that it did";
}

// The other origin of STATUS.HALT. The ROCr handler raises it through s_setreg
// on the way out, and there the bit means "keep this wave stopped" -- so a
// resume must not clear it. Nothing in the CWSR record distinguishes that from
// a wave halted at s_sendmsghalt, which wants exactly the opposite, so the
// provenance marker has to come out clear on this path.
TEST(WaveDebugTest, HandlerSetregHaltIsNotAttributedToSendmsghalt) {
  WaveDebugFixture fx;
  constexpr uint32_t kStatusHalt = 1u << 13;

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // SOPK s_setreg_imm32_b32 (op 20), hwreg id 2 = STATUS, offset 13, size 1 --
  // simm16 = 2 | (13 << 6) = 0x342. This is the ROCr handler's halt sequence.
  const uint32_t handler[] = {
      0x806C846Cu,              // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBA000342u, 0x00000001u, // s_setreg_imm32_b32 hwreg(STATUS, 13, 1), 1
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step(); // s_trap -> handler
  ASSERT_TRUE(wf->in_trap_handler());

  for (int i = 0; i < 4 && !wf->debug_halted(); ++i)
    fx.cu->step();

  ASSERT_NE(wf->status_raw() & kStatusHalt, 0u) << "the handler's s_setreg did not raise HALT";
  EXPECT_FALSE(wf->self_halted())
      << "a handler-raised HALT was attributed to s_sendmsghalt, so a resume would clear it "
         "and lose the breakpoint";
}

// Provenance is per stop, not per wave. A marker left behind by a resumed
// s_sendmsghalt would make the *next* stop -- one the handler halted with
// s_setreg -- look self-halted, and the resume after that would clear a HALT it
// must leave alone.
TEST(WaveDebugTest, SelfHaltedMarkerDoesNotSurviveTheStopThatSetIt) {
  WaveDebugFixture fx;
  constexpr uint32_t kSSendmsghaltInterrupt = 0xBF910001u;

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  const uint32_t handler[] = {
      0x806C846Cu,            // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du,            // s_addc_u32 ttmp1, ttmp1, 0
      kSSendmsghaltInterrupt, // s_sendmsghalt sendmsg(MSG_INTERRUPT)
      0xBE801F6Cu,            // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &, uint32_t message) { return message == 1; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step(); // s_trap -> handler
  for (int i = 0; i < 3 && !wf->self_halted(); ++i)
    fx.cu->step();
  ASSERT_TRUE(wf->self_halted());

  // What a debugger resume does: drop both halves of the stop, then let the
  // handler return.
  wf->set_status_halt(false);
  wf->set_self_halted(false);
  wf->set_debug_halted(false);
  for (int i = 0; i < 3 && wf->in_trap_handler(); ++i)
    fx.cu->step();
  ASSERT_FALSE(wf->in_trap_handler()) << "the handler never returned";
  EXPECT_FALSE(wf->self_halted());

  for (int i = 0; i < 4 && !wf->is_halted(); ++i)
    fx.cu->step();
  EXPECT_TRUE(wf->is_halted());
}

// S_SENDMSGHALT with nobody attached. The instruction halts the wave whether or
// not a debugger is there to resume it, so the flags it raises must be exactly
// the same ones a debugger stop raises -- and no more. What this pins is that
// an undebugged run does not diverge: the halt is a stop, not a corruption. The
// wave keeps the interrupted EXEC rather than running on under the handler's
// mask, its slot is not silently retired behind the halt, and clearing the two
// halves of the stop -- which is all a session release does -- puts it back on
// the scheduler and lets it retire.
TEST(WaveDebugTest, SendmsghaltWithoutADebugSessionHaltsAndStaysReleasable) {
  WaveDebugFixture fx;
  constexpr uint32_t kStatusHalt = 1u << 13;
  constexpr uint32_t kSSendmsghaltInterrupt = 0xBF910001u;
  constexpr uint64_t kApplicationExec = 0x5u;

  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  // The handler runs under its own mask, the way the real one does on its way
  // to MSG_INTERRUPT, so a lost EXEC restore is visible here.
  const uint32_t handler[] = {
      0x806C846Cu, // s_add_u32  ttmp0, ttmp0, 4
      0x826D806Du, // s_addc_u32 ttmp1, ttmp1, 0
      0xBEFE00FFu,
      0x80000000u,            // s_mov_b32 exec_lo, 0x80000000
      kSSendmsghaltInterrupt, // s_sendmsghalt sendmsg(MSG_INTERRUPT)
      0xBE801F6Cu,            // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  // debug_enabled clear: no debugger, so TTMP13 bit 23 stays 0 and nothing in
  // the driver is watching this wave.
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, false};
  });
  fx.cu->set_sendmsg_handler([](amdgpu::Wavefront &, uint32_t message) { return message == 1; });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kApplicationExec);

  fx.cu->step(); // s_trap -> handler
  ASSERT_TRUE(wf->in_trap_handler());

  for (int i = 0; i < 6 && !wf->debug_halted(); ++i)
    fx.cu->step();

  EXPECT_TRUE(wf->debug_halted()) << "s_sendmsghalt did not halt an undebugged wave";
  EXPECT_NE(wf->status_raw() & kStatusHalt, 0u);
  EXPECT_TRUE(wf->self_halted())
      << "an undebugged s_sendmsghalt must record provenance too, or the release below "
         "leaves STATUS.HALT set and the wave re-halts at s_rfe";
  EXPECT_FALSE(wf->is_halted()) << "the slot was retired behind a wave that is only stopped";

  // Stepping a stopped wave must not advance it, with or without a debugger.
  const uint64_t stopped_pc = wf->pc;
  fx.cu->step();
  EXPECT_EQ(wf->pc, stopped_pc);

  // What a session release does: drop both halves of the stop. Nothing else is
  // needed to make the wave runnable again.
  wf->set_status_halt(false);
  wf->set_self_halted(false);
  wf->set_debug_halted(false);
  for (int i = 0; i < 3 && wf->in_trap_handler(); ++i)
    fx.cu->step();
  ASSERT_FALSE(wf->in_trap_handler()) << "the handler never returned";
  EXPECT_EQ(wf->exec(), kApplicationExec)
      << "the application resumed under the handler's EXEC mask";

  for (int i = 0; i < 4 && !wf->is_halted(); ++i)
    fx.cu->step();
  EXPECT_TRUE(wf->is_halted()) << "the wave never retired, so its dispatch cannot complete";
}

// The CWSR codec reproduces the gfx9.4 and gfx12.5 record layouts, and the
// differences elsewhere are not confined to one field. Serializing a wave from an
// unmodelled architecture would hand rocm-dbgapi an image it decodes against a
// different layout, so the codec has to say plainly which ones it covers.
TEST(WaveDebugTest, CwsrLayoutIsModelledForGfx94AndGfx1250) {
  EXPECT_EQ(kmd::cwsr_layout_kind(ROCJITSU_CODE_ARCH_CDNA3), kmd::CwsrLayoutKind::Gfx9_4);
  EXPECT_EQ(kmd::cwsr_layout_kind(ROCJITSU_CODE_ARCH_CDNA4), kmd::CwsrLayoutKind::Gfx9_4);
  EXPECT_EQ(kmd::cwsr_layout_kind(ROCJITSU_CODE_ARCH_CDNA5), kmd::CwsrLayoutKind::Gfx12_5);
  EXPECT_TRUE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA3)); // gfx942
  EXPECT_TRUE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA4)); // gfx950
  EXPECT_TRUE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA5)); // gfx1250

  // gfx908 saves an ACC-VGPR block and gfx908/gfx90a keep the packet id in
  // TTMP6, so being gfx9 is not enough.
  EXPECT_FALSE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA1));
  EXPECT_FALSE(kmd::cwsr_layout_modelled(ROCJITSU_CODE_ARCH_CDNA2));

  // RDNA layouts remain outside this codec.
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4})
    EXPECT_EQ(kmd::cwsr_layout_kind(arch), kmd::CwsrLayoutKind::Unsupported)
        << "arch " << static_cast<int>(arch);
}

// TRAPSTS.EXCP is architecturally sticky, so "the bit changed" is not the same
// question as "this instruction raised the cause". Delivering on the rising
// edge went permanently quiet for any cause already latched -- including every
// cause raised before a debugger attached, since the handler declines then.
TEST(WaveDebugTest, EnabledExceptionIsDeliveredAfterAnEarlierDisabledOccurrence) {
  WaveDebugFixture fx;
  constexpr uint32_t kInvalidCause = 1u << 0;
  constexpr uint32_t kOverflowCause = 1u << 3;
  constexpr uint32_t kModeExcpEnShift = 12;
  // v_mul_f32 v0, v0, v0  (VOP2 op 5, src0 = v0 = 256, vsrc1 = v0, vdst = v0)
  constexpr uint32_t kVMulF32V0 = 0x0A000000u | 256u;
  constexpr float kHuge = 1e38f;

  fx.gpu_mem.write32(kKernelAddr, kVMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 4, kVMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 8, kSEndpgm);

  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(1ULL);
  const uint32_t vbase = wave->vgpr_alloc().base;

  uint32_t handler_calls = 0;
  fx.cu->set_alu_exception_handler([&](amdgpu::Wavefront &) {
    ++handler_calls;
    return false; // observe delivery without stopping the wave
  });

  // First occurrence with overflow *not* enabled. Some other cause is enabled
  // so the instruction takes the exception-aware path and latches TRAPSTS --
  // with no enable at all the generated body takes a SIMD fast path that skips
  // the sticky update, which would not exercise the sticky-bit problem.
  wave->set_mode_raw(kInvalidCause << kModeExcpEnShift);
  fx.cu->write_vgpr(vbase + 0, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 0u) << "overflow is not enabled yet";
  ASSERT_NE(wave->trapsts() & kOverflowCause, 0u) << "overflow was not latched";

  // Software now enables overflow and repeats the operation. Hardware traps on
  // this occurrence; a rising-edge test cannot, because the bit is already set.
  wave->set_mode_raw(kOverflowCause << kModeExcpEnShift);
  fx.cu->write_vgpr(vbase + 0, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 1u) << "second, enabled occurrence was not delivered";
}

TEST(WaveDebugTest, Gfx1250AluExceptionDeliveryUsesTrapCtrlNotModeVgprMsb) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  constexpr uint32_t kInvalidCause = 1u << 0;
  constexpr uint32_t kOverflowCause = 1u << 3;
  // Keep SRC2 in a nonzero bank. VOP2 does not use that role, but a regression
  // that reads gfx9's MODE[18:12] exception enables sees a nonzero mask here.
  constexpr uint32_t kSrc2VgprMsb = 1u << 18;
  // v_mul_f32 v0, v0, v0 assembled for gfx1250.
  constexpr uint32_t kGfx1250VMulF32V0 = 0x10000100u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr float kHuge = 1e38f;

  fx.gpu_mem.write32(kKernelAddr, kGfx1250VMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 4, kGfx1250VMulF32V0);
  fx.gpu_mem.write32(kKernelAddr + 8, kGfx1250SEndpgm);

  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(1ULL);
  wave->set_mode_raw(kSrc2VgprMsb);
  const uint32_t vbase = wave->vgpr_alloc().base;

  uint32_t handler_calls = 0;
  fx.cu->set_alu_exception_handler([&](amdgpu::Wavefront &) {
    ++handler_calls;
    return false;
  });

  // Keep one dedicated enable set so the first instruction takes the
  // exception-aware path and latches overflow without delivering it.
  wave->set_gfx12_trap_ctrl_raw(kInvalidCause);
  fx.cu->write_vgpr(vbase, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 0u) << "TRAP_CTRL does not enable overflow yet";
  ASSERT_NE(wave->trapsts() & kOverflowCause, 0u) << "overflow was not latched";
  EXPECT_EQ(wave->mode_raw(), kSrc2VgprMsb) << "exception delivery rewrote MODE.VGPR_MSB";

  wave->set_gfx12_trap_ctrl_raw(kOverflowCause);
  fx.cu->write_vgpr(vbase, 0, std::bit_cast<uint32_t>(kHuge));
  fx.cu->step();
  EXPECT_EQ(handler_calls, 1u) << "enabled TRAP_CTRL overflow was not delivered";
  EXPECT_EQ(wave->mode_raw(), kSrc2VgprMsb) << "TRAP_CTRL was folded into MODE.VGPR_MSB";
}

// debug_active_ is CU-wide, so the memory probe also runs for waves belonging to
// a process nobody is debugging. When the handler declines the stop for such a
// wave the access must still be issued -- discarding it silently lost stores and
// left loads' destination registers stale in an undebugged process.
TEST(WaveDebugTest, DeclinedMemoryViolationStillIssuesTheAccess) {
  WaveDebugFixture fx;
  constexpr uint32_t kProcessId = 11;
  constexpr uint64_t kPageSize = amdgpu::GpuMemory::PAGE_SIZE;
  constexpr uint64_t kUnmappedAddr = 0x900000;

  std::vector<uint8_t> kernel_page(kPageSize);
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xC0020000u, kProcessId);     // s_load_dword s0, s[0:1]
  fx.gpu_mem.write32(kKernelAddr + 4, 0, kProcessId);           // immediate offset 0
  fx.gpu_mem.write32(kKernelAddr + 8, 0xBF800000u, kProcessId); // s_nop 0
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(kProcessId);
  // Another process turned debugging on for this CU; this wave is not debugged.
  fx.cu->set_debug_active(true);

  // The declined access is expected to reach the pipeline, so the cache
  // hierarchy needs its functional writeback path wired up.
  fx.l2.set_backing_memory(&fx.gpu_mem);

  ASSERT_FALSE(fx.gpu_mem.is_mapped(kUnmappedAddr, kProcessId));
  const uint32_t sbase = wave->sgpr_alloc().base;
  fx.cu->write_sgpr(sbase + 0, static_cast<uint32_t>(kUnmappedAddr));
  fx.cu->write_sgpr(sbase + 1, static_cast<uint32_t>(kUnmappedAddr >> 32));
  // A recognisable value at the unmapped address, which falls through to sparse
  // backing, so a load that really issued can be told from one that did not.
  fx.gpu_mem.write32(kUnmappedAddr, 0xFEEDFACEu, kProcessId);

  bool handler_called = false;
  fx.cu->set_memory_violation_handler([&](amdgpu::Wavefront &, uint64_t, bool) {
    handler_called = true;
    return false; // no debug session for this wave's process
  });

  fx.cu->step();

  EXPECT_TRUE(handler_called);
  EXPECT_FALSE(wave->debug_halted());
  EXPECT_EQ(wave->pc, kKernelAddr + 8);
  // The load was issued rather than discarded, so s0 holds what memory held.
  EXPECT_EQ(fx.cu->read_sgpr(sbase + 0), 0xFEEDFACEu);

  fx.gpu_mem.unregister_process(kProcessId);
}

// A multi-byte access that starts on a mapped page can still run off the end of
// it. Checking only the first byte let such an access through with no
// EC_QUEUE_WAVE_MEMORY_VIOLATION, so the whole touched range is validated.
TEST(WaveDebugTest, ScalarLoadStraddlingIntoUnmappedPageReportsMemoryViolation) {
  WaveDebugFixture fx;
  constexpr uint32_t kProcessId = 9;
  constexpr uint64_t kPageSize = amdgpu::GpuMemory::PAGE_SIZE;
  constexpr uint64_t kDataAddr = 0x40000;

  std::vector<uint8_t> kernel_page(kPageSize);
  std::vector<uint8_t> data_page(kPageSize);
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kKernelAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {kernel_page.data(),
                                                              amdgpu::Mtype::RW};
  // Exactly one data page is mapped; the page above it deliberately is not.
  page_table[kDataAddr >> amdgpu::GpuMemory::PAGE_SHIFT] = {data_page.data(), amdgpu::Mtype::RW};
  fx.gpu_mem.register_process(kProcessId, &page_table, &page_table_mutex);

  fx.gpu_mem.write32(kKernelAddr, 0xC0060000u, kProcessId);     // s_load_dwordx2 s[0:1], s[0:1]
  fx.gpu_mem.write32(kKernelAddr + 4, 0, kProcessId);           // immediate offset 0
  fx.gpu_mem.write32(kKernelAddr + 8, 0xBF800000u, kProcessId); // s_nop 0
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(kProcessId);
  fx.cu->set_debug_active(true);

  // Wired up so that failing to detect the straddle lets the access proceed
  // through the pipeline, and the test then fails on the missing violation
  // rather than on unbacked cache plumbing.
  fx.l2.set_backing_memory(&fx.gpu_mem);

  // The last four bytes of the mapped page: the first dword is mapped, the
  // second runs into the unmapped page above.
  const uint64_t straddle = kDataAddr + kPageSize - 4;
  ASSERT_TRUE(fx.gpu_mem.is_mapped(straddle, kProcessId));
  ASSERT_FALSE(fx.gpu_mem.is_mapped(straddle + 4, kProcessId));
  const uint32_t sbase = wave->sgpr_alloc().base;
  fx.cu->write_sgpr(sbase + 0, static_cast<uint32_t>(straddle));
  fx.cu->write_sgpr(sbase + 1, static_cast<uint32_t>(straddle >> 32));

  uint64_t fault_address = ~uint64_t{0};
  fx.cu->set_memory_violation_handler(
      [&](amdgpu::Wavefront &faulting_wave, uint64_t address, bool is_write) {
        EXPECT_EQ(&faulting_wave, wave);
        EXPECT_FALSE(is_write);
        fault_address = address;
        faulting_wave.debug_trap(0);
        return true;
      });

  fx.cu->step();
  EXPECT_EQ(fault_address, straddle);
  EXPECT_TRUE(wave->debug_halted());

  fx.gpu_mem.unregister_process(kProcessId);
}

// Linked gfx950 code forms helper addresses as GETPC plus a signed rel32
// relocation. The 64-bit sum can have a 0x1ffff high dword even though it
// denotes an address relative to the loaded code object. SWAPPC must relocate
// that value to the dispatch load bias and preserve the architectural return
// address for SETPC.
TEST(WaveDebugTest, SwappcResolvesLinkedRel32AndSetpcReturns) {
  WaveDebugFixture fx;
  constexpr uint64_t kLoadBias = 0x7FFF00000000ULL;
  constexpr uint64_t kCallerPc = kLoadBias + 0xA3A8;
  constexpr uint64_t kCalleePc = kLoadBias + 0x8D60;

  fx.gpu_mem.write32(kCallerPc, 0xBE9E1E00u); // s_swappc_b64 s[30:31], s[0:1]
  fx.gpu_mem.write32(kCallerPc + 4, kSEndpgm);
  fx.gpu_mem.write32(kCalleePc, 0xBE801D1Eu); // s_setpc_b64 s[30:31]

  auto *wf = fx.dispatch(kCallerPc);
  ASSERT_NE(wf, nullptr);
  wf->set_code_load_bias(kLoadBias);
  // Linked gfx950 calls place GETPC, ADD_LO, ADDC_HI, then SWAPPC. At SWAPPC
  // the relocation base is therefore 16 bytes behind the current PC.
  const int32_t rel32 = static_cast<int32_t>(kCalleePc - (kCallerPc - 16));
  const uint64_t encoded_target = 0x0001FFFF00000000ULL | static_cast<uint32_t>(rel32);
  fx.cu->write_sgpr(wf->sgpr_alloc().base + 0, static_cast<uint32_t>(encoded_target));
  fx.cu->write_sgpr(wf->sgpr_alloc().base + 1, static_cast<uint32_t>(encoded_target >> 32));

  fx.cu->step();
  EXPECT_EQ(wf->pc, kCalleePc);
  EXPECT_EQ(wf->debug_read_sgpr(30), static_cast<uint32_t>(kCallerPc + 4));
  EXPECT_EQ(wf->debug_read_sgpr(31), static_cast<uint32_t>((kCallerPc + 4) >> 32));

  fx.cu->step();
  EXPECT_EQ(wf->pc, kCallerPc + 4);
}

// A single-stepped wave (rocm-dbgapi MODE.debug_en=1) executes exactly one
// instruction and is then handed to the single-step completion handler, which
// re-stops it. The engine must not run past that one instruction.
TEST(WaveDebugTest, SingleStepExecutesOneInstructionThenReports) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, 0xBF800000u);     // s_nop 0
  fx.gpu_mem.write32(kKernelAddr + 4, 0xBF800000u); // s_nop 0
  fx.gpu_mem.write32(kKernelAddr + 8, kSEndpgm);

  uint32_t step_count = 0;
  fx.cu->set_single_step_handler([&](amdgpu::Wavefront &w) {
    ++step_count;
    w.set_debug_halted(true); // re-stop after the step, as the driver does
    return true;
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_debug_single_step(true);

  // One engine step runs exactly one instruction, then the handler re-stops it.
  fx.cu->step();
  EXPECT_EQ(step_count, 1u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_TRUE(wf->debug_halted());

  // While halted, stepping the CU makes no further progress.
  fx.cu->step();
  EXPECT_EQ(step_count, 1u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4);

  // Resuming single-step runs the next instruction and re-stops again.
  wf->set_debug_halted(false);
  wf->set_debug_single_step(true);
  fx.cu->step();
  EXPECT_EQ(step_count, 2u);
  EXPECT_EQ(wf->pc, kKernelAddr + 8);
  EXPECT_TRUE(wf->debug_halted());
}

// Without a debugger the s_trap is a no-op: the wave advances past it and runs
// to completion. This keeps kernels that embed traps in unreached paths from
// aborting the emulator when no debugger is attached.
TEST(WaveDebugTest, STrapWithoutDebuggerIsNoOp) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  fx.cu->step(); // s_trap -> no handler -> no-op, PC advances
  EXPECT_FALSE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kKernelAddr + 4);

  fx.cu->step(); // s_endpgm -> wave halts
  EXPECT_TRUE(wf->is_halted());
}

TEST(WaveDebugTest, IllegalInstructionStopsWaveUnderDebugger) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, 0xFFFFFFFFu);
  uint32_t illegal_count = 0;
  fx.cu->set_illegal_inst_handler([&](amdgpu::Wavefront &wave) {
    ++illegal_count;
    wave.set_fatal_exception_pending(true);
    wave.set_debug_halted(true);
    return true;
  });
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  fx.cu->step();
  EXPECT_EQ(illegal_count, 1u);
  EXPECT_TRUE(wave->debug_halted());
  EXPECT_TRUE(wave->fatal_exception_pending());
  EXPECT_EQ(wave->pc, kKernelAddr);
  wave->reset();
  EXPECT_FALSE(wave->fatal_exception_pending());
}

TEST(WaveDebugTest, IllegalInstructionWithoutDebuggerHalts) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, 0xFFFFFFFFu);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  fx.cu->step();
  EXPECT_TRUE(wave->is_halted());
}

TEST(WaveDebugTest, STrapWithoutConfiguredHandlerAdvancesPcWithoutChangingWaveState) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapSeven);

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_status_raw(0x12345678u);
  wf->set_trapsts(0x87654321u);

  fx.cu->step();

  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_EQ(wf->status_raw(), 0x12345678u);
  EXPECT_EQ(wf->trapsts(), 0x87654321u);
  EXPECT_EQ(wf->trap_id(), 0u);
  EXPECT_FALSE(wf->debug_halted());
}

TEST(WaveDebugTest, Gfx1250TrapWithoutConfiguredHandlerAdvancesPc) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  fx.gpu_mem.write32(kKernelAddr, kGfx12STrap);

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  fx.cu->step();

  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_FALSE(wf->debug_halted());
}

TEST(WaveDebugTest, Rdna4TrapEntryAndReturnUseTheGfx12WaveStateLayout) {
  constexpr uint64_t kHighKernelAddr = 0x0000123400001000ULL;
  constexpr uint32_t kApplicationStatus = 1u | (0xAu << 1);
  constexpr uint32_t kSchedMode = 2u;
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  fx.gpu_mem.write32(kHighKernelAddr, kGfx12STrap);
  fx.gpu_mem.write32(kHighKernelAddr + 4, kSEndpgm);

  // GFX12.0 uses the common split STATE_PRIV/TRAP_CTRL ABI with a 48-bit PC,
  // preserves SCHED_MODE in TTMP1[27:26], and returns through s_rfe_b64.
  const uint32_t handler[] = {
      0x806C846Cu, 0x826D806Du, 0xB9800384u, 0x00000001u, 0xBE804A6Cu,
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0, true};
  });

  auto *wf = fx.dispatch(kHighKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_status_raw(kApplicationStatus);
  wf->set_wave_sched_mode_raw(kSchedMode);
  for (uint32_t i = 6; i <= 9; ++i)
    wf->set_ttmp(i, 0xB6000000u + i);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_TRUE(wf->uses_separate_trap_ctrl());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_EQ(wf->ttmp(0), static_cast<uint32_t>(kHighKernelAddr));
  EXPECT_EQ(wf->ttmp(1), 0x38001234u);
  for (uint32_t i = 6; i <= 9; ++i)
    EXPECT_EQ(wf->ttmp(i), 0xB6000000u + i) << "TTMP" << i << " was not preserved";
  EXPECT_EQ(wf->ttmp(11), 1u << 23);
  EXPECT_EQ(wf->ttmp(12), (1u << 9) | (0xAu << 10));
  EXPECT_NE(wf->status_raw() & (1u << 5), 0u) << "trap entry did not enter privileged mode";

  for (int i = 0; i < 4; ++i)
    fx.cu->step();

  EXPECT_FALSE(wf->in_trap_handler());
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kHighKernelAddr + 4);
  EXPECT_EQ(wf->status_raw(), kApplicationStatus | amdgpu::Wavefront::kStatusHaltMask);
}

// gfx12.5 changed both halves of the first-level trap ABI: PC is 57 bits with
// TrapID in TTMP1[31:28], and DebugEnabled/STATE_PRIV moved to TTMP11/12.
// Exercise a real gfx1250 s_rfe_i64 sequence as well, so a layout that merely
// looks right at entry cannot pass while truncating the return address.
TEST(WaveDebugTest, Gfx1250TrapEntryAndReturnFollowTheGfx12Abi) {
  constexpr uint64_t kHighKernelAddr = 0x0100000000001000ULL;
  constexpr uint32_t kApplicationStatus = 1u | (0xAu << 1);
  constexpr uint64_t kScratchBase = 0x0000001240008000ULL;
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  fx.gpu_mem.write32(kHighKernelAddr, kGfx12STrap);
  fx.gpu_mem.write32(kHighKernelAddr + 4, kSEndpgm);

  // Assembled for gfx1250:
  //   s_add_co_u32 ttmp0, ttmp0, 4
  //   s_add_co_ci_u32 ttmp1, ttmp1, 0
  //   s_setreg_imm32_b32 hwreg(WAVE_STATE_PRIV, 14, 1), 1
  //   s_rfe_i64 ttmp[0:1]
  const uint32_t handler[] = {
      0x806C846Cu, 0x826D806Du, 0xB9800384u, 0x00000001u, 0xBE804A6Cu,
  };
  for (uint32_t i = 0; i < std::size(handler); ++i)
    fx.gpu_mem.write32(kTrapHandlerAddr + i * 4, handler[i]);

  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{kTrapHandlerAddr, 0x123456789ABCuLL, true};
  });

  auto *wf = fx.dispatch(kHighKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_status_raw(kApplicationStatus);
  wf->set_scratch_base(kScratchBase);
  for (uint32_t i = 6; i <= 9; ++i)
    wf->set_ttmp(i, 0xA5000000u + i);

  fx.cu->step();
  ASSERT_TRUE(wf->in_trap_handler());
  EXPECT_EQ(wf->pc, kTrapHandlerAddr);
  EXPECT_EQ(wf->ttmp(0), static_cast<uint32_t>(kHighKernelAddr));
  EXPECT_EQ(wf->ttmp(1), 0x31000000u);
  for (uint32_t i = 6; i <= 9; ++i)
    EXPECT_EQ(wf->ttmp(i), 0xA5000000u + i) << "TTMP" << i << " was not preserved";
  EXPECT_EQ(wf->ttmp(11), 1u << 23);
  EXPECT_EQ(wf->ttmp(12), (1u << 9) | (0xAu << 10) | (1u << 18));
  EXPECT_EQ(wf->ttmp(14), 0x56789ABCu);
  EXPECT_EQ(wf->ttmp(15), 0x00001234u);
  EXPECT_NE(wf->status_raw() & (1u << 5), 0u) << "trap entry did not enter privileged mode";

  for (int i = 0; i < 4; ++i)
    fx.cu->step();

  EXPECT_FALSE(wf->in_trap_handler());
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->pc, kHighKernelAddr + 4);
  EXPECT_EQ(wf->status_raw(), kApplicationStatus | amdgpu::Wavefront::kStatusHaltMask);
  EXPECT_FALSE(wf->self_halted());
}

// The trap temporary registers and trap status register round-trip, and reset()
// (slot reuse) clears all debugger state.
TEST(WaveDebugTest, TrapRegistersRoundTripAndReset) {
  WaveDebugFixture fx;
  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);

  for (uint32_t i = 0; i < 16; ++i)
    wf->set_ttmp(i, 0x1000 + i);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(wf->ttmp(i), 0x1000 + i);
  EXPECT_EQ(wf->ttmp(16), 0u); // out of range reads zero

  wf->set_trapsts(0xDEAD);
  EXPECT_EQ(wf->trapsts(), 0xDEADu);
  wf->set_queue_id(42);
  EXPECT_EQ(wf->queue_id(), 42u);

  wf->debug_trap(7);
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->trap_id(), 7u);

  wf->reset();
  EXPECT_FALSE(wf->debug_halted());
  EXPECT_FALSE(wf->debug_single_step());
  EXPECT_EQ(wf->trap_id(), 0u);
  EXPECT_EQ(wf->trapsts(), 0u);
  EXPECT_EQ(wf->queue_id(), 0u);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(wf->ttmp(i), 0u);
}

// -----------------------------------------------------------------------------
// CWSR serialization round-trip: parse the serialized area back with the exact
// formulas rocm-dbgapi uses (projects/rocdbgapi/src/architecture.cpp gfx9/mi
// cwsr_record_t) and confirm the header invariants and every register offset.
// -----------------------------------------------------------------------------

struct ParsedWave {
  uint64_t pc = 0, exec = 0, vcc = 0, wave_id = 0;
  uint32_t status = 0, trapsts = 0, mode = 0, m0 = 0, ttmp6 = 0, ttmp11 = 0;
  bool first = false, last = false;
  uint32_t group[3] = {};
  std::vector<uint32_t> sgprs;
  std::vector<uint32_t> vgprs;
  std::vector<uint8_t> lds;
};

// Walk the CWSR area exactly as rocm-dbgapi's control_stack_iterate +
// register_address do, asserting the contiguity invariants dbgapi enforces.
std::vector<ParsedWave> parse_cwsr(const std::map<uint64_t, uint32_t> &mem, uint64_t base) {
  auto rd = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };
  auto rd64 = [&](uint64_t va) -> uint64_t {
    return rd(va) | (static_cast<uint64_t>(rd(va + 4)) << 32);
  };

  const uint32_t cs_off = rd(base + 0);
  const uint32_t cs_size = rd(base + 4);
  const uint32_t ws_off = rd(base + 8);
  const uint32_t ws_size = rd(base + 12);

  // dbgapi: control_stack_end must equal wave_area_begin.
  EXPECT_EQ(base + cs_off + cs_size, base + ws_off - ws_size);

  std::vector<ParsedWave> out;
  uint64_t last_wave_area = base + ws_off;
  uint32_t state = 0;
  const uint32_t words = cs_size / 4;
  for (uint32_t i = 2; i < words; ++i) {
    uint32_t rl = rd(base + cs_off + i * 4);
    if (rl & (1u << 30))
      continue; // event
    if (rl & (1u << 31)) {
      state = rl;
      continue;
    }
    const uint32_t vgprs_field = state & 0x3F;
    const uint32_t sgprs_field = (state >> 6) & 0x7;
    const uint32_t accum = (state >> 24) & 0x3F;
    const uint32_t vgpr_count = (accum + 1) * 4;
    const uint32_t acc = (vgprs_field + 1) * 8 - vgpr_count;
    const uint32_t sgpr_count = (sgprs_field + 1) * 16 - 16;
    const uint32_t lds_size = ((state >> 9) & 0xFF) * 1280;

    const uint64_t save = last_wave_area - 64;
    const uint64_t register_area_end = save - ((rl & (1u << 17)) ? lds_size : 0);
    const uint64_t hwregs = register_area_end - 32 * 4;
    const uint64_t ttmps = register_area_end - 16 * 4;
    const uint64_t sgprs_addr = hwregs - sgpr_count * 4;
    const uint64_t accv = sgprs_addr - acc * 256;
    const uint64_t vgprs_addr = accv - static_cast<uint64_t>(vgpr_count) * 256;

    ParsedWave pw;
    pw.last = rl & (1u << 16);
    pw.first = rl & (1u << 17);
    pw.m0 = rd(hwregs + 0 * 4);
    pw.pc = rd64(hwregs + 1 * 4);
    pw.exec = rd64(hwregs + 3 * 4);
    pw.status = rd(hwregs + 5 * 4);
    pw.trapsts = rd(hwregs + 6 * 4);
    pw.mode = rd(hwregs + 9 * 4);
    pw.wave_id = rd64(ttmps + 4 * 4);
    pw.ttmp6 = rd(ttmps + 6 * 4);
    pw.group[0] = rd(ttmps + 8 * 4);
    pw.group[1] = rd(ttmps + 9 * 4);
    pw.group[2] = rd(ttmps + 10 * 4);
    pw.ttmp11 = rd(ttmps + 11 * 4);
    const uint32_t vcc_lo_slot = std::min<uint32_t>(108, sgpr_count) - 2;
    pw.vcc = rd64(sgprs_addr + vcc_lo_slot * 4);
    pw.sgprs.resize(sgpr_count);
    for (uint32_t s = 0; s < sgpr_count; ++s)
      pw.sgprs[s] = rd(sgprs_addr + s * 4);
    pw.vgprs.resize(static_cast<size_t>(vgpr_count) * 64);
    for (uint32_t r = 0; r < vgpr_count; ++r)
      for (uint32_t l = 0; l < 64; ++l)
        pw.vgprs[r * 64 + l] = rd(vgprs_addr + r * 256 + l * 4);
    if (pw.first) {
      pw.lds.resize(lds_size);
      for (uint32_t byte = 0; byte < lds_size; ++byte)
        pw.lds[byte] =
            static_cast<uint8_t>(rd(register_area_end + (byte & ~3u)) >> ((byte & 3u) * 8));
    }
    out.push_back(std::move(pw));
    last_wave_area = vgprs_addr;
  }
  // dbgapi: the walk must bottom out exactly at wave_area_begin.
  EXPECT_EQ(last_wave_area, base + ws_off - ws_size);
  return out;
}

kmd::CwsrWaveState make_wave(uint64_t id, uint64_t pc) {
  kmd::CwsrWaveState w;
  w.pc = pc;
  w.exec = 0xF0F0F0F0ULL;
  w.vcc = 0xABCD1234ULL;
  w.status = (1u << 13); // HALT
  w.trapsts = 0;
  w.mode = 0;
  w.m0 = 0x55;
  w.wave_id = id;
  w.group_ids = {1, 2, 3};
  w.wave_in_group = 0;
  w.queue_packet_id = 7;
  w.trap_id = 1;
  w.wave_stopped = true;
  w.num_sgprs = 16;
  w.num_vgprs = 4;
  w.sgprs.resize(w.num_sgprs);
  for (uint32_t s = 0; s < w.num_sgprs; ++s)
    w.sgprs[s] = 0x1000 + s;
  w.vgprs.resize(static_cast<size_t>(w.num_vgprs) * 64);
  for (uint32_t r = 0; r < w.num_vgprs; ++r)
    for (uint32_t l = 0; l < 64; ++l)
      w.vgprs[r * 64 + l] = (r << 16) | l;
  return w;
}

TEST(WaveDebugTest, Gfx1250CwsrMatchesDbgapiWave32LayoutAndRoundTrips) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;

  auto wave = make_wave(0xAABBCCDD11223344ULL, 0x123456789ABCuLL);
  wave.exec = 0xFFFFFFFFu;
  wave.flat_scratch = 0x0000001240008000ULL;
  wave.state_priv = (1u << 9) | (1u << 14) | (1u << 18);
  wave.status = (1u << 6) | (1u << 16);
  wave.excp_flag_priv = (1u << 6) | (1u << 12);
  wave.excp_flag_user = 1u << 3;
  wave.trap_ctrl = (1u << 3) | (1u << 9);
  wave.xnack_state_priv = 0x00057F7Fu;
  wave.xnack_mask = 0x76543210u;
  wave.scratch_scoreboard_id = 1023;
  wave.shader_engine_id = 1;
  wave.group_ids = {9, 10, 11};
  wave.wave_in_group = 3;
  wave.queue_packet_id = 0x12345;
  wave.trap_id = 7;
  wave.saved_status_halt = true;
  wave.spi_ttmps_setup = true;
  wave.num_sgprs = 112;
  wave.sgprs.resize(wave.num_sgprs);
  for (uint32_t s = 0; s < wave.num_sgprs; ++s)
    wave.sgprs[s] = 0x51000000u + s;
  wave.lds = {0x11, 0x22, 0x33, 0x44};

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t address, uint32_t value) { mem[address] = value; };
  auto read32 = [&](uint64_t address) -> uint32_t {
    auto it = mem.find(address);
    return it == mem.end() ? 0u : it->second;
  };

  const auto layout = kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, kArch);
  ASSERT_TRUE(layout.ok);
  EXPECT_EQ(layout.control_stack_size, 5u * sizeof(uint32_t));
  const uint64_t control = kCtxBase + layout.control_stack_offset;
  EXPECT_EQ(read32(control + 8), (1u << 31) | (1u << 24) | (1u << 10));
  EXPECT_EQ(read32(control + 12), 0u);
  EXPECT_EQ(read32(control + 16), 1023u | (1u << 11) | (1u << 12) | (1u << 13) | (1u << 24));

  // gfx12.5 walks directly down from wave_state_offset: one 1-KiB LDS block,
  // 128 HWREG dwords, 128 SGPR dwords and sixteen 32-lane VGPRs.
  const uint64_t wave_end = kCtxBase + layout.wave_state_offset;
  const uint64_t hwregs = wave_end - 1024 - 128 * sizeof(uint32_t);
  const uint64_t sgprs = hwregs - 128 * sizeof(uint32_t);
  const uint64_t vgprs = sgprs - 16 * 32 * sizeof(uint32_t);
  const uint64_t ttmps = sgprs + 112 * sizeof(uint32_t);
  EXPECT_EQ(vgprs, kCtxBase + layout.wave_state_offset - layout.wave_state_size);
  EXPECT_EQ(read32(hwregs + 5 * 4), wave.state_priv);
  EXPECT_EQ(read32(hwregs + 6 * 4), wave.excp_flag_priv);
  EXPECT_EQ(read32(hwregs + 7 * 4), wave.xnack_mask);
  EXPECT_EQ(read32(hwregs + 9 * 4), static_cast<uint32_t>(wave.flat_scratch));
  EXPECT_EQ(read32(hwregs + 10 * 4), static_cast<uint32_t>(wave.flat_scratch >> 32));
  EXPECT_EQ(read32(hwregs + 11 * 4), wave.excp_flag_user);
  EXPECT_EQ(read32(hwregs + 12 * 4), wave.trap_ctrl);
  EXPECT_EQ(read32(hwregs + 13 * 4), wave.status);
  EXPECT_EQ(read32(ttmps + 4 * 4), static_cast<uint32_t>(wave.wave_id));
  EXPECT_EQ(read32(ttmps + 5 * 4), static_cast<uint32_t>(wave.wave_id >> 32));
  EXPECT_EQ(read32(ttmps + 6 * 4), (1u << 30) | (1u << 29));
  EXPECT_EQ(read32(ttmps + 7 * 4), 10u | (11u << 16));
  EXPECT_EQ(read32(ttmps + 8 * 4), wave.queue_packet_id | (3u << 25) | (1u << 30) | (1u << 31));
  EXPECT_EQ(read32(ttmps + 9 * 4), 9u);
  EXPECT_EQ(read32(ttmps + 11 * 4), wave.xnack_state_priv | (7u << 28));
  EXPECT_EQ(read32(vgprs + 3 * 32 * 4 + 31 * 4), wave.vgprs[3 * 64 + 31]);

  std::vector<kmd::CwsrWaveState> output(1);
  output[0].num_sgprs = wave.num_sgprs;
  output[0].num_vgprs = wave.num_vgprs;
  output[0].lds.resize(wave.lds.size());
  ASSERT_TRUE(kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, output, read32, kArch));
  const auto &restored = output[0];
  EXPECT_EQ(restored.pc, wave.pc);
  EXPECT_EQ(restored.exec, wave.exec);
  EXPECT_EQ(restored.vcc, wave.vcc);
  EXPECT_EQ(restored.flat_scratch, wave.flat_scratch);
  EXPECT_EQ(restored.state_priv, wave.state_priv);
  EXPECT_EQ(restored.status, wave.status);
  EXPECT_EQ(restored.excp_flag_priv, wave.excp_flag_priv);
  EXPECT_EQ(restored.excp_flag_user, wave.excp_flag_user);
  EXPECT_EQ(restored.trap_ctrl, wave.trap_ctrl);
  EXPECT_EQ(restored.xnack_state_priv, wave.xnack_state_priv);
  EXPECT_EQ(restored.xnack_mask, wave.xnack_mask);
  EXPECT_EQ(restored.wave_id, wave.wave_id);
  EXPECT_EQ(restored.group_ids, wave.group_ids);
  EXPECT_EQ(restored.wave_in_group, wave.wave_in_group);
  EXPECT_EQ(restored.queue_packet_id, wave.queue_packet_id);
  EXPECT_EQ(restored.trap_id, wave.trap_id);
  ASSERT_EQ(restored.lds.size(), 1024u);
  EXPECT_TRUE(std::equal(wave.lds.begin(), wave.lds.end(), restored.lds.begin()));
  for (uint32_t r = 0; r < wave.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(restored.vgprs[r * 64 + lane], wave.vgprs[r * 64 + lane]);
}

TEST(WaveDebugTest, Gfx1250CwsrAcceptsTheFullAddressableVgprFile) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  constexpr uint32_t kMaxVgprs = 1024;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;

  auto wave = make_wave(0xAA55, 0x4000);
  wave.num_vgprs = kMaxVgprs;
  wave.vgprs.resize(static_cast<size_t>(wave.num_vgprs) * 64);
  wave.vgprs[(kMaxVgprs - 1) * 64 + 31] = 0xA5A55A5Au;

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t address, uint32_t value) { mem[address] = value; };
  auto read32 = [&](uint64_t address) -> uint32_t {
    auto it = mem.find(address);
    return it == mem.end() ? 0u : it->second;
  };

  const auto layout = kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, kArch);
  ASSERT_TRUE(layout.ok);
  EXPECT_EQ(read32(kCtxBase + layout.control_stack_offset + 8) & 0x3Fu, 0x3Fu);

  std::vector<kmd::CwsrWaveState> output(1);
  output[0].num_sgprs = wave.num_sgprs;
  output[0].num_vgprs = wave.num_vgprs;
  ASSERT_TRUE(kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, output, read32, kArch));
  EXPECT_EQ(output[0].vgprs[(kMaxVgprs - 1) * 64 + 31], 0xA5A55A5Au);

  mem.clear();
  wave.num_vgprs = kMaxVgprs + 1;
  EXPECT_FALSE(kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {wave}, write32, kArch).ok);
  EXPECT_TRUE(mem.empty());
}

TEST(WaveDebugTest, CwsrSerializationRoundTripsThroughDbgapiLayout) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000; // 256 KiB

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };

  std::vector<kmd::CwsrWaveState> waves = {make_wave(0xAA01, 0x2000), make_wave(0xAA02, 0x2040),
                                           make_wave(0xAA03, 0x2080)};
  waves[0].wave_in_group = 0;
  waves[0].is_first_in_group = true;
  waves[0].is_last_in_group = false;
  waves[1].wave_in_group = 1;
  waves[1].is_first_in_group = false;
  waves[1].is_last_in_group = true;
  waves[2].group_ids[0] = 2;
  waves[2].is_first_in_group = true;
  waves[2].is_last_in_group = true;
  waves[0].lds = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  waves[2].lds = {0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18};

  kmd::CwsrLayout layout =
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(layout.ok);

  std::vector<ParsedWave> parsed = parse_cwsr(mem, kCtxBase);
  ASSERT_EQ(parsed.size(), waves.size());

  for (size_t i = 0; i < waves.size(); ++i) {
    const auto &in = waves[i];
    const auto &out = parsed[i];
    EXPECT_EQ(out.pc, in.pc) << "wave " << i;
    EXPECT_EQ(out.exec, in.exec);
    EXPECT_EQ(out.status, in.status);
    EXPECT_EQ(out.mode, in.mode);
    EXPECT_EQ(out.m0, in.m0);
    EXPECT_EQ(out.wave_id, in.wave_id);
    EXPECT_EQ(out.vcc, in.vcc);
    EXPECT_EQ(out.group[0], in.group_ids[0]);
    EXPECT_EQ(out.group[1], in.group_ids[1]);
    EXPECT_EQ(out.group[2], in.group_ids[2]);
    EXPECT_EQ(out.first, i == 0 || i == 2);
    EXPECT_EQ(out.last, i == 1 || i == 2);
    // TTMP6: wave_stopped (bit30) and trap id (bits 25:28).
    EXPECT_TRUE(out.ttmp6 & (1u << 30));
    EXPECT_EQ((out.ttmp6 >> 25) & 0xF, in.trap_id);
    // TTMP11: trap-handler-setup (bit31) and packet id (bits 6:30).
    EXPECT_TRUE(out.ttmp11 & (1u << 31));
    EXPECT_EQ((out.ttmp11 >> 6) & 0x1FFFFFF, in.queue_packet_id);
    // Meaningful scalars and vectors round-trip.
    for (uint32_t s = 0; s < in.num_sgprs; ++s)
      EXPECT_EQ(out.sgprs[s], in.sgprs[s]) << "sgpr " << s;
    for (uint32_t r = 0; r < in.num_vgprs; ++r)
      for (uint32_t l = 0; l < 64; ++l)
        EXPECT_EQ(out.vgprs[r * 64 + l], in.vgprs[r * 64 + l]) << "vgpr " << r << " lane " << l;
    if (in.is_first_in_group) {
      ASSERT_GE(out.lds.size(), in.lds.size());
      EXPECT_TRUE(std::equal(in.lds.begin(), in.lds.end(), out.lds.begin()));
    } else {
      EXPECT_TRUE(out.lds.empty());
    }
  }

  const auto original_mem = mem;
  auto invalid = make_wave(1, 0x3000);
  invalid.num_vgprs = 257;
  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {invalid}, write32, ROCJITSU_CODE_ARCH_CDNA4)
          .ok);
  EXPECT_EQ(mem, original_mem);

  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(kCtxBase + 1, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4)
          .ok);
  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(UINT64_MAX - 3, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4)
          .ok);
  EXPECT_FALSE(kmd::serialize_queue_cwsr(kCtxBase, layout.wave_state_offset - 1, waves, write32,
                                         ROCJITSU_CODE_ARCH_CDNA4)
                   .ok);
  EXPECT_EQ(mem, original_mem);
}

// rocm-dbgapi carves 32-byte instruction buffers out of a per-queue "debugger
// memory" region declared in the context-save header (debug_offset/size at byte
// 16/20). It aborts displaced stepping if that region is absent, so the
// serializer must reserve a non-zero, in-bounds, non-overlapping region.
TEST(WaveDebugTest, CwsrReservesDebuggerMemoryForDisplacedStepping) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000; // 256 KiB

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };
  auto rd = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };

  std::vector<kmd::CwsrWaveState> waves = {make_wave(0xAA01, 0x2000)};
  kmd::CwsrLayout layout =
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(layout.ok);

  // The header advertises the region dbgapi reads (kfd_context_save_area_header
  // debug_offset/size).
  EXPECT_EQ(rd(kCtxBase + 16), layout.debug_offset);
  EXPECT_EQ(rd(kCtxBase + 20), layout.debug_size);

  // Non-zero, or dbgapi aborts with "reserved memory is missing".
  EXPECT_NE(layout.debug_offset, 0u);
  EXPECT_NE(layout.debug_size, 0u);
  // 64-byte aligned (DEBUGGER_BYTES_ALIGN) so each 32-byte chunk is aligned.
  EXPECT_EQ(layout.debug_offset % 64u, 0u);
  EXPECT_EQ(layout.debug_size % 64u, 0u);
  // Sits above the wave area (no overlap) and inside the save area.
  EXPECT_GE(layout.debug_offset, layout.wave_state_offset);
  EXPECT_LE(layout.debug_offset + layout.debug_size, kAreaSize);
  // Holds enough 32-byte chunks for dbgapi's park + terminating buffers plus a
  // per-wave displaced-step buffer.
  EXPECT_GE(layout.debug_size / 32u, waves.size() + 2);
}

// FLAT_SCRATCH and VCC are written into aliased slots at the top of the saved
// SGPR block. num_sgprs is accepted up to 106, so those slots overlap real
// SGPR indices: serialization overwrote s102/s103 with FLAT_SCRATCH and
// deserialization read the alias bytes back as if they were registers, while
// flat_scratch itself was never restored at all. A round trip therefore
// corrupted live register state and lost the scratch base.
TEST(WaveDebugTest, CwsrRoundTripPreservesAliasedSgprsAndFlatScratch) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  // The full saved block. CDNA3/CDNA4 default to 112 SGPRs per wave
  // (config_loader default_sgprs_per_wf), so a wave that uses the whole
  // allocation is the ordinary case, not an extreme one.
  constexpr uint32_t kMaxAcceptedSgprs = kmd::kCwsrSavedSgprSlots;
  constexpr uint64_t kFlatScratch = 0x0000DEAD0000BEEFULL;

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };
  auto read32 = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };

  std::vector<kmd::CwsrWaveState> in = {make_wave(0xCC01, 0x3000)};
  in[0].num_sgprs = kMaxAcceptedSgprs;
  in[0].sgprs.resize(kMaxAcceptedSgprs);
  for (uint32_t s = 0; s < kMaxAcceptedSgprs; ++s)
    in[0].sgprs[s] = 0xA5A50000u + s;
  in[0].flat_scratch = kFlatScratch;
  in[0].vcc = 0x0123456789ABCDEFULL;

  ASSERT_TRUE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok);

  std::vector<kmd::CwsrWaveState> out(1);
  out[0].num_sgprs = in[0].num_sgprs;
  out[0].num_vgprs = in[0].num_vgprs;
  ASSERT_TRUE(
      kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, out, read32, ROCJITSU_CODE_ARCH_CDNA4));

  EXPECT_EQ(out[0].flat_scratch, kFlatScratch) << "flat_scratch was not restored";
  EXPECT_EQ(out[0].vcc, in[0].vcc);

  // FLAT_SCRATCH occupies s102/s103 and VCC s106/s107. Only the slots an alias
  // actually covers may be lost: clamping at the gfx9.4 architected count of
  // 102 instead would also drop s104 and s105, which sit *between* the two
  // pairs and are ordinary registers. That is why the predicate is an explicit
  // slot test and not a bound.
  for (uint32_t s = 0; s < kMaxAcceptedSgprs; ++s) {
    if (kmd::cwsr_sgpr_slot_is_aliased(s, ROCJITSU_CODE_ARCH_CDNA4))
      EXPECT_EQ(out[0].sgprs[s], 0u) << "aliased slot " << s << " decoded as an SGPR";
    else
      EXPECT_EQ(out[0].sgprs[s], in[0].sgprs[s]) << "sgpr " << s << " did not survive";
  }
  // Pin the two slots the "everything at or above 102 is an alias" shortcut
  // would silently drop, so a future simplification back to a bound fails here.
  EXPECT_EQ(out[0].sgprs[104], in[0].sgprs[104]);
  EXPECT_EQ(out[0].sgprs[105], in[0].sgprs[105]);
}

// The saved block is 112 slots, and CDNA3/CDNA4 default to exactly that many
// SGPRs per wave. When the codec accepted only 106 the geometry for a
// default-configured wave came back not-ok, so the whole queue's CWSR publish
// failed and the debugger saw no waves at all -- reachable from any CDNA4
// config that simply omits sgprs_per_wf.
TEST(WaveDebugTest, CwsrAcceptsTheFullDefaultSgprAllocation) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };

  std::vector<kmd::CwsrWaveState> in = {make_wave(0xCC02, 0x3000)};
  in[0].num_sgprs = kmd::kCwsrSavedSgprSlots;
  in[0].sgprs.assign(kmd::kCwsrSavedSgprSlots, 0u);
  EXPECT_TRUE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok)
      << "a wave using the default " << kmd::kCwsrSavedSgprSlots << "-SGPR allocation was rejected";

  // One past the block is genuinely out of range and must still be refused
  // rather than written past the end of the SGPR area.
  mem.clear();
  in[0].num_sgprs = kmd::kCwsrSavedSgprSlots + 1;
  in[0].sgprs.assign(in[0].num_sgprs, 0u);
  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok);
  EXPECT_TRUE(mem.empty()) << "a rejected image still wrote to the ctx-save area";
}

TEST(WaveDebugTest, CwsrDeserializeRecoversSerializedWaveState) {
  // serialize_queue_cwsr followed by deserialize_queue_cwsr must round-trip the
  // register state the resume path reads back (the debugger writes its edits
  // straight into this area, and resume reloads them onto the live wave).
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000; // 256 KiB

  std::map<uint64_t, uint32_t> mem;
  auto write32 = [&](uint64_t va, uint32_t val) { mem[va] = val; };
  auto read32 = [&](uint64_t va) -> uint32_t {
    auto it = mem.find(va);
    return it == mem.end() ? 0u : it->second;
  };

  std::vector<kmd::CwsrWaveState> in = {make_wave(0xBB01, 0x3000), make_wave(0xBB02, 0x3080)};
  // Give the second wave MODE.debug_en set so the round-trip preserves it (the
  // resume path uses this bit to select single-step).
  in[1].mode = (1u << 11);

  ASSERT_TRUE(
      kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32, ROCJITSU_CODE_ARCH_CDNA4).ok);

  // Reload: supply only the per-wave geometry (num_sgprs/num_vgprs) so the
  // decoder reproduces the exact layout, then read the values back.
  std::vector<kmd::CwsrWaveState> out;
  for (const auto &w : in) {
    kmd::CwsrWaveState g;
    g.num_sgprs = w.num_sgprs;
    g.num_vgprs = w.num_vgprs;
    out.push_back(g);
  }
  ASSERT_TRUE(
      kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, out, read32, ROCJITSU_CODE_ARCH_CDNA4));
  ASSERT_EQ(out.size(), in.size());

  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_EQ(out[i].pc, in[i].pc) << "wave " << i;
    EXPECT_EQ(out[i].exec, in[i].exec);
    EXPECT_EQ(out[i].vcc, in[i].vcc);
    EXPECT_EQ(out[i].status, in[i].status);
    EXPECT_EQ(out[i].mode, in[i].mode);
    EXPECT_EQ(out[i].m0, in[i].m0);
    EXPECT_EQ(out[i].wave_id, in[i].wave_id);
    EXPECT_EQ(out[i].wave_stopped, in[i].wave_stopped);
    EXPECT_EQ(out[i].spi_ttmps_setup, in[i].spi_ttmps_setup);
    EXPECT_EQ(out[i].group_ids, in[i].group_ids);
    EXPECT_EQ(out[i].wave_in_group, in[i].wave_in_group);
    EXPECT_EQ(out[i].queue_packet_id, in[i].queue_packet_id);
    for (uint32_t s = 0; s < in[i].num_sgprs; ++s)
      EXPECT_EQ(out[i].sgprs[s], in[i].sgprs[s]) << "wave " << i << " sgpr " << s;
    for (uint32_t r = 0; r < in[i].num_vgprs; ++r)
      for (uint32_t l = 0; l < 64; ++l)
        EXPECT_EQ(out[i].vgprs[r * 64 + l], in[i].vgprs[r * 64 + l])
            << "wave " << i << " vgpr " << r << " lane " << l;
  }
}

TEST(WaveDebugTest, CwsrBulkCodecMatchesDwordCodecAndPublishesHeaderLast) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  std::vector<kmd::CwsrWaveState> waves = {make_wave(0xCC01, 0x4000), make_wave(0xCC02, 0x4080)};

  std::map<uint64_t, uint32_t> dwords;
  const auto reference = kmd::serialize_queue_cwsr(
      kCtxBase, kAreaSize, waves,
      [&](uint64_t address, uint32_t value) { dwords[address] = value; }, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(reference.ok);

  std::vector<uint8_t> bulk_image(reference.wave_state_offset);
  std::vector<std::pair<uint64_t, size_t>> writes;
  const auto bulk = kmd::serialize_queue_cwsr_bulk(
      kCtxBase, kAreaSize, waves,
      [&](uint64_t address, std::span<const uint8_t> bytes) {
        writes.emplace_back(address, bytes.size());
        const size_t offset = static_cast<size_t>(address - kCtxBase);
        ASSERT_LE(offset + bytes.size(), bulk_image.size());
        std::memcpy(bulk_image.data() + offset, bytes.data(), bytes.size());
      },
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(bulk.ok);
  EXPECT_EQ(bulk.control_stack_offset, reference.control_stack_offset);
  EXPECT_EQ(bulk.control_stack_size, reference.control_stack_size);
  EXPECT_EQ(bulk.wave_state_offset, reference.wave_state_offset);
  EXPECT_EQ(bulk.wave_state_size, reference.wave_state_size);
  EXPECT_EQ(bulk.debug_offset, reference.debug_offset);
  EXPECT_EQ(bulk.debug_size, reference.debug_size);

  ASSERT_EQ(writes.size(), 2u);
  EXPECT_EQ(writes[0].first, kCtxBase + reference.control_stack_offset);
  EXPECT_EQ(writes[0].second, reference.wave_state_offset - reference.control_stack_offset);
  EXPECT_EQ(writes[1], (std::pair<uint64_t, size_t>{kCtxBase, 10 * sizeof(uint32_t)}));

  for (uint32_t offset = 0; offset < reference.wave_state_offset; offset += sizeof(uint32_t)) {
    uint32_t actual = 0;
    std::memcpy(&actual, bulk_image.data() + offset, sizeof(actual));
    auto expected = dwords.find(kCtxBase + offset);
    EXPECT_EQ(actual, expected == dwords.end() ? 0u : expected->second) << "offset " << offset;
  }
}

TEST(WaveDebugTest, CwsrBulkDeserializeRoundTripsAndRejectsCorruptHeader) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kAreaSize = 0x40000;
  std::vector<kmd::CwsrWaveState> input = {make_wave(0xDD01, 0x5000), make_wave(0xDD02, 0x5080)};
  input[1].mode = 1u << 11;

  std::vector<uint8_t> image(kAreaSize);
  ASSERT_TRUE(kmd::serialize_queue_cwsr_bulk(
                  kCtxBase, kAreaSize, input,
                  [&](uint64_t address, std::span<const uint8_t> bytes) {
                    std::memcpy(image.data() + (address - kCtxBase), bytes.data(), bytes.size());
                  },
                  ROCJITSU_CODE_ARCH_CDNA4)
                  .ok);

  auto geometry_only = [&] {
    std::vector<kmd::CwsrWaveState> states(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
      states[index].num_sgprs = input[index].num_sgprs;
      states[index].num_vgprs = input[index].num_vgprs;
    }
    return states;
  };
  auto read_block = [&](uint64_t address, std::span<uint8_t> bytes) {
    std::memcpy(bytes.data(), image.data() + (address - kCtxBase), bytes.size());
  };

  auto output = geometry_only();
  ASSERT_TRUE(kmd::deserialize_queue_cwsr_bulk(kCtxBase, kAreaSize, output, read_block,
                                               ROCJITSU_CODE_ARCH_CDNA4));
  ASSERT_EQ(output.size(), input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    EXPECT_EQ(output[index].pc, input[index].pc);
    EXPECT_EQ(output[index].exec, input[index].exec);
    EXPECT_EQ(output[index].vcc, input[index].vcc);
    EXPECT_EQ(output[index].mode, input[index].mode);
    EXPECT_EQ(output[index].wave_id, input[index].wave_id);
    EXPECT_EQ(output[index].group_ids, input[index].group_ids);
    EXPECT_EQ(output[index].sgprs, input[index].sgprs);
    EXPECT_EQ(output[index].vgprs, input[index].vgprs);
  }

  image[8] ^= 1;
  output = geometry_only();
  EXPECT_FALSE(kmd::deserialize_queue_cwsr_bulk(kCtxBase, kAreaSize, output, read_block,
                                                ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(WaveDebugTest, CwsrBulkCodecUsesConstantBlockOperationsAtFullScale) {
  constexpr uint64_t kCtxBase = 0x400000000ULL;
  constexpr uint32_t kWaveCount = 1024;
  constexpr uint32_t kAreaSize = 80 * 1024 * 1024;
  std::vector<kmd::CwsrWaveState> waves(kWaveCount);
  for (uint32_t index = 0; index < kWaveCount; ++index) {
    auto &wave = waves[index];
    wave.pc = 0x1000 + index * 4;
    wave.wave_id = index + 1;
    wave.group_ids[0] = index / 4;
    wave.wave_in_group = index % 4;
    wave.queue_packet_id = 1;
    wave.num_sgprs = 106;
    wave.num_vgprs = 256;
    wave.sgprs.resize(wave.num_sgprs, index);
    wave.vgprs.resize(static_cast<size_t>(wave.num_vgprs) * 64, index);
  }

  size_t write_calls = 0;
  size_t bytes_written = 0;
  const auto layout = kmd::serialize_queue_cwsr_bulk(
      kCtxBase, kAreaSize, waves,
      [&](uint64_t, std::span<const uint8_t> bytes) {
        ++write_calls;
        bytes_written += bytes.size();
      },
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(layout.ok);
  EXPECT_EQ(write_calls, 2u);
  EXPECT_EQ(bytes_written,
            layout.wave_state_offset - layout.control_stack_offset + 10 * sizeof(uint32_t));
  EXPECT_GT(layout.wave_state_size, 64u * 1024u * 1024u);
}

} // namespace
