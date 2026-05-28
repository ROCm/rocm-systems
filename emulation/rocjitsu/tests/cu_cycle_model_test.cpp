// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Engine-driven test: a CuCycleModel registered under a root composite, in a
/// clock domain, drains a wave's pending FIFO over real clock edges.

#include "plugins/cu_cycle_model.h"
#include "plugins/mem_sys_cycle_model.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

using namespace rocjitsu::amdgpu;

static cycle_model::UarchConfig tiny_cfg() {
  cycle_model::UarchConfig c;
  c.name = "t"; c.wave_size = 64; c.simds_per_cu = 4;
  c.wave_slots_per_simd = 8; c.front_end_issue_per_simd = 1;
  c.valu = {1, 4}; c.salu = {1, 1}; c.smem = {1, 20}; c.vmem = {1, 200}; c.lds_pipe = {1, 24};
  return c;
}

static cycle_model::PendingEvent valu_instr() {
  cycle_model::PendingEvent e;
  e.kind = cycle_model::PendingKind::Instruction; e.metadata_ready = true;
  e.instr.kind = cycle_model::InstrKind::VALU; e.instr.mnemonic = "v_add_f32";
  return e;
}

// A cache-configured uarch (mirrors cdna4's cache geometry) so a memory op actually
// exercises the L1->shared async path instead of the cacheless flat-latency shortcut.
static cycle_model::UarchConfig cache_cfg() {
  cycle_model::UarchConfig c = tiny_cfg();
  c.l1v = {/*size_kb=*/32, /*ways=*/4, /*line_bytes=*/128, /*hit_latency=*/6, /*miss_to_next_level=*/200};
  c.l1s = {16, 4, 64, 4, 200};
  c.l2  = {4096, 16, 128, 80, 400};
  c.mshrs_per_l1v = 16;
  c.hbm_channels = 8;
  c.l2_bytes_per_cycle = 1024;
  c.hbm_access_latency = 300;
  c.hbm_bytes_per_channel_per_cycle = 396;
  return c;
}

// A cold-miss vector load: misses L1 and L2 -> full hierarchy (l1.hit + l2.hit +
// hbm_access_latency = 6 + 80 + 300 = 386 cycles). Increments VMCNT; writes v0.
static cycle_model::PendingEvent cold_vmem_load() {
  cycle_model::PendingEvent e;
  e.kind = cycle_model::PendingKind::Instruction; e.metadata_ready = true;
  e.instr.kind = cycle_model::InstrKind::VMEM; e.instr.mnemonic = "global_load_dword";
  e.instr.wcnt = cycle_model::WaitCounter::VMCNT;
  e.instr.regs.vgprs_written.push_back(0);      // dst -> deferred RAW mark at completion
  e.instr.mem.lane_addr[0] = 0x1000; e.instr.mem.lane_mask = 1;
  e.instr.mem.elem_bytes = 4; e.instr.mem.mtype = 2 /*RW: cached*/;
  return e;
}

// Positional s_waitcnt vmcnt(0) gate: blocks the wave at FIFO head until the
// outstanding VMCNT returns to 0 (i.e. the cold miss has retired).
static cycle_model::PendingEvent waitcnt_vmcnt0() {
  cycle_model::PendingEvent e;
  e.kind = cycle_model::PendingKind::WaitcntGate; e.metadata_ready = true;
  e.waitcnt.target.t[cycle_model::idx(cycle_model::WaitCounter::VMCNT)] = 0;
  return e;
}

constexpr uint64_t VMEM_COLD = 6 + 80 + 300;   // 386

TEST(CuCycleModel, DrainsWaveOverClockEdges) {
  simdojo::SimulationEngine::Config cfg{};
  cfg.max_ticks = 1'000'000;     // bound the run
  cfg.num_threads = 1;
  simdojo::SimulationEngine engine(cfg);

  // 1 GHz CU clock (period = 1000 ticks). Frequency value is cosmetic here.
  auto *clk = engine.topology().add_clock_domain("cu_clk", /*frequency_hz=*/1'000'000'000ull);

  auto root = std::make_unique<simdojo::CompositeComponent>("root");
  auto *cm = static_cast<CuCycleModel *>(
      root->add_child(std::make_unique<CuCycleModel>("cu_cycle_model", *clk)));
  // The R2.2b in-process loopback is gone: the CuCycleModel sends L1-miss traffic
  // over its req_out port to a shared MemSysCycleModel and gets async completions on
  // cpl_in. Build + wire the shared component (both directions) before engine.build().
  auto *memsys = static_cast<MemSysCycleModel *>(
      root->add_child(std::make_unique<MemSysCycleModel>("mem_sys")));
  auto ports = memsys->add_cu_ports();
  engine.topology().add_link(cm->req_out(), ports.req_in, /*latency=*/1);
  engine.topology().add_link(ports.cpl_out, cm->cpl_in(), /*latency=*/1);
  // SharedMemModel holds a const UarchConfig&; bind to long-lived storage.
  static const cycle_model::UarchConfig memsys_cfg = tiny_cfg();
  memsys->configure(memsys_cfg);
  engine.topology().set_root(std::move(root));
  engine.build();

  // Configure + enqueue a 3-instruction wave AFTER build (mirrors first-dispatch
  // injection by the plugin). resume_clock arms the (idled) clock.
  cm->configure(tiny_cfg());
  auto *ws = new cycle_model::CycleWaveState();   // freed at process exit
  ws->simd_id = 0; ws->scoreboard.resize(16, 16);
  for (int i = 0; i < 3; ++i) ws->pending.push_back(valu_instr());
  cm->model().simd(0).waves.push_back(ws);
  cm->resume_clock(/*after=*/0);

  engine.run();

  EXPECT_TRUE(ws->pending.empty());               // drained by the clock
  EXPECT_GE(cm->model().cu().cu_cycle, 3u);        // at least 3 edges issued 3 instrs
}

// R2.2c async round-trip over ports: a cold-miss memory op no longer completes
// synchronously in attempt_issue — it emits a SharedReq into the ArchModel out-queue,
// which the CuCycleModel sends as a MemReqMsg to the shared MemSysCycleModel; the
// completion arrives async on cpl_in and calls on_mem_completion. The op must
// eventually retire: vmcnt returns to 0, the wave drains past its s_waitcnt gate,
// and cu_cycle reflects the cold-miss latency (~386 cycles).
TEST(CuCycleModel, ServicesAsyncMemoryOverPorts) {
  simdojo::SimulationEngine::Config cfg{};
  cfg.max_ticks = 1'000'000;
  cfg.num_threads = 1;
  simdojo::SimulationEngine engine(cfg);

  auto *clk = engine.topology().add_clock_domain("cu_clk", /*frequency_hz=*/1'000'000'000ull);

  // ArchModel + SharedMemModel hold a const UarchConfig& (never copy), so the config
  // must outlive the run — the real plugin keeps it in a long-lived map. Keep it here.
  // Both the CuCycleModel and the shared MemSysCycleModel use the SAME config.
  static const cycle_model::UarchConfig cfg_owned = cache_cfg();

  auto root = std::make_unique<simdojo::CompositeComponent>("root");
  auto *cm = static_cast<CuCycleModel *>(
      root->add_child(std::make_unique<CuCycleModel>("cu_cycle_model", *clk)));
  auto *memsys = static_cast<MemSysCycleModel *>(
      root->add_child(std::make_unique<MemSysCycleModel>("mem_sys")));
  auto ports = memsys->add_cu_ports();
  engine.topology().add_link(cm->req_out(), ports.req_in, /*latency=*/1);
  engine.topology().add_link(ports.cpl_out, cm->cpl_in(), /*latency=*/1);
  memsys->configure(cfg_owned);
  engine.topology().set_root(std::move(root));
  engine.build();

  cm->configure(cfg_owned);
  auto *ws = new cycle_model::CycleWaveState();   // freed at process exit
  ws->simd_id = 0; ws->scoreboard.resize(16, 16);
  // cold load -> s_waitcnt vmcnt(0) -> a dependent VALU that cannot issue until the
  // load retires (the gate clears only once outstanding VMCNT == 0).
  ws->pending.push_back(cold_vmem_load());
  ws->pending.push_back(waitcnt_vmcnt0());
  ws->pending.push_back(valu_instr());
  cm->model().simd(0).waves.push_back(ws);
  cm->resume_clock(/*after=*/0);

  engine.run();

  EXPECT_TRUE(ws->pending.empty());                       // wave fully drained past the gate
  EXPECT_TRUE(ws->outstanding.empty());                   // VMCNT decremented at retire -> 0
  EXPECT_TRUE(ws->in_flight.empty());                     // the in-flight memreq retired
  // Cold-miss latency dominates: the gate cannot clear before the load completes at
  // ~VMEM_COLD cycles, so the run must have taken at least that many CU edges.
  EXPECT_GE(cm->model().cu().cu_cycle, VMEM_COLD);
}
