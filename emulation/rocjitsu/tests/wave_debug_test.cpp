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
#include <map>
#include <memory>
#include <vector>

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
    wave.set_debug_halted(true);
    return true;
  });
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  fx.cu->step();
  EXPECT_EQ(illegal_count, 1u);
  EXPECT_TRUE(wave->debug_halted());
  EXPECT_EQ(wave->pc, kKernelAddr);
}

TEST(WaveDebugTest, IllegalInstructionWithoutDebuggerHalts) {
  WaveDebugFixture fx;
  fx.gpu_mem.write32(kKernelAddr, 0xFFFFFFFFu);
  auto *wave = fx.dispatch(kKernelAddr);
  ASSERT_NE(wave, nullptr);
  fx.cu->step();
  EXPECT_TRUE(wave->is_halted());
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

    const uint64_t save = last_wave_area - 64;
    const uint64_t hwregs = save - 32 * 4;
    const uint64_t ttmps = save - 16 * 4;
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

  kmd::CwsrLayout layout = kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32);
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
  }

  const auto original_mem = mem;
  auto invalid = make_wave(1, 0x3000);
  invalid.num_vgprs = 257;
  EXPECT_FALSE(kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, {invalid}, write32).ok);
  EXPECT_EQ(mem, original_mem);

  EXPECT_FALSE(kmd::serialize_queue_cwsr(kCtxBase + 1, kAreaSize, waves, write32).ok);
  EXPECT_FALSE(kmd::serialize_queue_cwsr(UINT64_MAX - 3, kAreaSize, waves, write32).ok);
  EXPECT_FALSE(
      kmd::serialize_queue_cwsr(kCtxBase, layout.wave_state_offset - 1, waves, write32).ok);
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
  kmd::CwsrLayout layout = kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, waves, write32);
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

  ASSERT_TRUE(kmd::serialize_queue_cwsr(kCtxBase, kAreaSize, in, write32).ok);

  // Reload: supply only the per-wave geometry (num_sgprs/num_vgprs) so the
  // decoder reproduces the exact layout, then read the values back.
  std::vector<kmd::CwsrWaveState> out;
  for (const auto &w : in) {
    kmd::CwsrWaveState g;
    g.num_sgprs = w.num_sgprs;
    g.num_vgprs = w.num_vgprs;
    out.push_back(g);
  }
  ASSERT_TRUE(kmd::deserialize_queue_cwsr(kCtxBase, kAreaSize, out, read32));
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
    for (uint32_t s = 0; s < in[i].num_sgprs; ++s)
      EXPECT_EQ(out[i].sgprs[s], in[i].sgprs[s]) << "wave " << i << " sgpr " << s;
    for (uint32_t r = 0; r < in[i].num_vgprs; ++r)
      for (uint32_t l = 0; l < 64; ++l)
        EXPECT_EQ(out[i].vgprs[r * 64 + l], in[i].vgprs[r * 64 + l])
            << "wave " << i << " vgpr " << r << " lane " << l;
  }
}

} // namespace
