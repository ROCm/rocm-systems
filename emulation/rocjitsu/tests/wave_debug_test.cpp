// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wave_debug_test.cpp
/// @brief Wave-level KFD debugger groundwork, exercised through the compute unit.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/spi.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 64;
constexpr uint32_t kKernelAddr = 0x1000;
constexpr uint32_t kSTrapBreakpoint = 0xBF920001u; // s_trap 1 (rocm-dbgapi breakpoint)
constexpr uint32_t kSTrapSeven = 0xBF92AB07u;      // s_trap 0xab07 (trap id is low 8 bits)
constexpr uint32_t kGfx1250STrap = 0xBF900003u;    // s_trap 3 on gfx12.5
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

// A wave that executes an s_trap breakpoint stops for the debugger: the trap
// handler is invoked with the trap id, the wave becomes debug-halted (not
// retired), and the PC points just past the s_trap. The CU then reports no
// runnable work (so the engine can quiesce) while the slot stays occupied.
TEST(WaveDebugTest, STrapBreakpointStopsWaveAndQuiescesCu) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapBreakpoint);
  fx.gpu_mem.write32(kKernelAddr + 4, kSEndpgm);

  uint32_t trap_seen = 0xFFFFFFFF;
  const amdgpu::Wavefront *trap_wf = nullptr;
  fx.cu->set_trap_handler([&](amdgpu::Wavefront &w, uint32_t id) {
    trap_seen = id;
    trap_wf = &w;
    return true; // debugger attached: stop the wave
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  EXPECT_FALSE(wf->debug_halted());
  EXPECT_TRUE(fx.cu->has_runnable_wfs());

  fx.cu->step();

  EXPECT_EQ(trap_seen, 1u);
  EXPECT_EQ(trap_wf, wf);
  EXPECT_TRUE(wf->debug_halted());
  EXPECT_EQ(wf->trap_id(), 1u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4); // saved PC is past the s_trap
  // The slot is still occupied (not retired) but nothing is runnable, so the
  // engine can go idle while the wave is stopped at the breakpoint.
  EXPECT_TRUE(fx.cu->has_active_wfs());
  EXPECT_FALSE(fx.cu->has_runnable_wfs());
  EXPECT_TRUE(fx.cu->is_idle());
  amdgpu::ShaderProcessorInput spi({fx.cu.get()});
  EXPECT_FALSE(spi.step());
  EXPECT_FALSE(spi.has_pending());
  spi.run_to_idle();

  // Stepping again makes no progress while the wave is debug-halted.
  uint64_t pc_before = wf->pc;
  fx.cu->step();
  EXPECT_EQ(wf->pc, pc_before);
  EXPECT_TRUE(wf->debug_halted());

  // Resuming clears the debug halt; the wave then runs to s_endpgm and retires.
  wf->set_debug_halted(false);
  EXPECT_TRUE(fx.cu->has_runnable_wfs());
  fx.cu->step();
  EXPECT_TRUE(wf->is_halted());
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

TEST(WaveDebugTest, DeclinedTrapAdvancesPcWithoutChangingWaveState) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, kSTrapSeven);

  uint32_t trap_seen = 0;
  fx.cu->set_trap_handler([&](amdgpu::Wavefront &, uint32_t id) {
    trap_seen = id;
    return false;
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  wf->set_status_raw(0x12345678u);
  wf->set_trapsts(0x87654321u);

  fx.cu->step();

  EXPECT_EQ(trap_seen, 7u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_EQ(wf->status_raw(), 0x12345678u);
  EXPECT_EQ(wf->trapsts(), 0x87654321u);
  EXPECT_EQ(wf->trap_id(), 0u);
  EXPECT_FALSE(wf->debug_halted());
}

TEST(WaveDebugTest, Gfx1250TrapUsesDecodedInstructionAndImmediate) {
  WaveDebugFixture fx(ROCJITSU_CODE_ARCH_GFX1250);
  fx.gpu_mem.write32(kKernelAddr, kGfx1250STrap);

  uint32_t trap_seen = 0;
  fx.cu->set_trap_handler([&](amdgpu::Wavefront &, uint32_t id) {
    trap_seen = id;
    return true;
  });

  auto *wf = fx.dispatch(kKernelAddr);
  ASSERT_NE(wf, nullptr);
  fx.cu->step();

  EXPECT_EQ(trap_seen, 3u);
  EXPECT_EQ(wf->pc, kKernelAddr + 4);
  EXPECT_TRUE(wf->debug_halted());
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

} // namespace
