// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Unit tests for the cycle-model scheduler core, driven by a synthetic clock
/// (no rocjitsu). Covers the scoreboard RAW/no-WAW contract, per-wave issue,
/// round-robin SIMD arbitration, and the tick_cycle driver.

#include "cycle_model/ArchModel.h"
#include "cycle_model/SharedMemModel.h"
#include "cycle_model/WaitCounters.h"
#include "cycle_model/WaveScoreboard.h"

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <unordered_map>

using namespace cycle_model;

// A memop that emits shared work leaves a request in mem_requests() that must be
// serviced or it never retires. Drive one CU edge then service the out-queue
// in-process (loopback). With a cache-less tiny_cfg the out-queue stays empty so this
// is a no-op; the helper keeps memory-driving cases correct.
static bool tick_with_loopback(ArchModel& m, SharedMemModel& shared) {
  bool more = m.tick_cycle();
  for (auto& req : m.mem_requests()) m.on_mem_completion(req.rid, shared.service(req.shared));
  m.mem_requests().clear();
  return more || !m.mem_requests().empty();
}

// --- lifetime / residency regressions (review HIGH UAFs) --------------------

// #1: MemorySystem holds a reference into ArchModel::cfg_, so ArchModel must be
// non-movable — otherwise moving it into a container (the adapter's per_cu_ map)
// leaves mem_.cfg_ dangling to the destroyed source. Lock it at compile time.
static_assert(!std::is_move_constructible_v<ArchModel>,
              "ArchModel must stay pinned: MemorySystem holds a cfg_ reference");
static_assert(!std::is_copy_constructible_v<ArchModel>,
              "ArchModel must stay pinned: MemorySystem holds a cfg_ reference");

// --- WaitCounters (M2 commit 0) ---------------------------------------------

TEST(WaitCounters, IncrementDecrementRoundTrips) {
  WaitCounters wc;
  EXPECT_TRUE(wc.empty());
  wc.increment(WaitCounter::VMCNT);
  wc.increment(WaitCounter::VMCNT);
  EXPECT_EQ(wc.c[idx(WaitCounter::VMCNT)], 2u);
  EXPECT_FALSE(wc.empty());
  wc.decrement(WaitCounter::VMCNT);
  EXPECT_EQ(wc.c[idx(WaitCounter::VMCNT)], 1u);
}

TEST(WaitCounters, DsKmAlsoBumpLgkm) {   // funcsim aggregate-subset quirk
  WaitCounters wc;
  wc.increment(WaitCounter::DSCNT);
  EXPECT_EQ(wc.c[idx(WaitCounter::DSCNT)], 1u);
  EXPECT_EQ(wc.c[idx(WaitCounter::LGKMCNT)], 1u);   // aggregate also bumped
  wc.increment(WaitCounter::KMCNT);
  EXPECT_EQ(wc.c[idx(WaitCounter::KMCNT)], 1u);
  EXPECT_EQ(wc.c[idx(WaitCounter::LGKMCNT)], 2u);   // both subsets fold in
  wc.decrement(WaitCounter::DSCNT);
  EXPECT_EQ(wc.c[idx(WaitCounter::DSCNT)], 0u);
  EXPECT_EQ(wc.c[idx(WaitCounter::LGKMCNT)], 1u);   // aggregate tracks back down
}

TEST(WaitCounters, DecrementAtZeroAndSaturationAreClamped) {
  WaitCounters wc;
  wc.decrement(WaitCounter::VMCNT);                 // no underflow
  EXPECT_EQ(wc.c[idx(WaitCounter::VMCNT)], 0u);
  for (int i = 0; i < 100; ++i) wc.increment(WaitCounter::VMCNT);
  EXPECT_EQ(wc.c[idx(WaitCounter::VMCNT)], WaitCounters::SAT);   // saturates
}

TEST(WaitTarget, SatisfiedComparesEverySlot) {
  WaitCounters wc;
  WaitTarget def;                                   // all 0xFF => wait on nothing
  EXPECT_TRUE(def.satisfied(wc));
  wc.increment(WaitCounter::VMCNT);
  EXPECT_TRUE(def.satisfied(wc));                   // still nothing waited on

  WaitTarget vm0; vm0.t[idx(WaitCounter::VMCNT)] = 0;
  EXPECT_FALSE(vm0.satisfied(wc));                  // 1 > 0 => blocked
  wc.decrement(WaitCounter::VMCNT);
  EXPECT_TRUE(vm0.satisfied(wc));                   // 0 <= 0 => satisfied

  WaitTarget vm1; vm1.t[idx(WaitCounter::VMCNT)] = 1;
  wc.increment(WaitCounter::VMCNT);
  EXPECT_TRUE(vm1.satisfied(wc));                   // 1 <= 1 boundary
}

TEST(Scoreboard, RawReadWaitsForProducerRetire) {
  Scoreboard sb;
  sb.resize(/*vgprs=*/8, /*sgprs=*/8);
  InstrRegSet producer; producer.vgprs_written = {3};
  sb.mark_writes(producer, /*retire_cyc=*/40);
  InstrRegSet consumer; consumer.vgprs_read = {3};
  EXPECT_EQ(sb.earliest_issue(consumer, /*now=*/10), 40u);   // must wait for v3
  InstrRegSet unrelated; unrelated.vgprs_read = {4};
  EXPECT_EQ(sb.earliest_issue(unrelated, /*now=*/10), 10u);  // v4 free
}

TEST(Scoreboard, NoWawInterlock) {
  Scoreboard sb; sb.resize(8, 8);
  InstrRegSet w1; w1.vgprs_written = {3}; sb.mark_writes(w1, 40);
  InstrRegSet w2; w2.vgprs_written = {3};                 // later write, same dest
  EXPECT_EQ(sb.earliest_issue(w2, /*now=*/10), 10u);      // writes are NOT scanned
}

// --- Scheduler core fixtures ------------------------------------------------

static UarchConfig tiny_cfg() {
  UarchConfig c; c.name = "t"; c.wave_size = 64; c.simds_per_cu = 4;
  c.wave_slots_per_simd = 8; c.front_end_issue_per_simd = 1;
  c.valu = {1, 4}; c.salu = {1, 1}; c.smem = {1, 20}; c.vmem = {1, 200}; c.lds_pipe = {1, 24};
  return c;
}
static CycleWaveState* make_wave(ArchModel& m, uint32_t simd) {
  auto* ws = new CycleWaveState();          // owned by the test process; freed at exit
  ws->simd_id = simd;
  ws->scoreboard.resize(16, 16);
  m.simd(simd).waves.push_back(ws);
  return ws;
}
static PendingEvent valu_instr() {
  PendingEvent e; e.kind = PendingKind::Instruction; e.metadata_ready = true;
  e.instr.kind = InstrKind::VALU; e.instr.mnemonic = "v_add_f32";
  return e;
}
static void run_cycles(ArchModel& m, uint64_t n) { for (uint64_t i = 0; i < n; ++i) m.tick_cycle(); }
// Memory-driving variant: service the async out-queue each edge (the loopback).
static void run_cycles_loopback(ArchModel& m, SharedMemModel& shared, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i) tick_with_loopback(m, shared);
}

TEST(Latency, FallbackIsPipeKindNotValu) {
  ArchModel m(tiny_cfg());          // vmem base_latency 200, salu 1, valu 4
  auto* ws = make_wave(m, 0);
  // VMEM opcode absent from opcode_latency must fall back to the VMEM pipe (200),
  // NOT VALU (4) — the whole point of M2's flat memory latency.
  InstrEvent vmem; vmem.kind = InstrKind::VMEM; vmem.mnemonic = "global_load_dword";
  vmem.regs.vgprs_written = {5};
  EXPECT_EQ(m.try_issue_inline(*ws, vmem, /*now=*/10), IssueResult::ISSUED);
  InstrRegSet rd_v; rd_v.vgprs_read = {5};
  EXPECT_EQ(ws->scoreboard.earliest_issue(rd_v, /*now=*/10), 210u);   // 10 + 200

  InstrEvent salu; salu.kind = InstrKind::SALU; salu.mnemonic = "s_and_b32";
  salu.regs.sgprs_written = {3};
  EXPECT_EQ(m.try_issue_inline(*ws, salu, /*now=*/20), IssueResult::ISSUED);
  InstrRegSet rd_s; rd_s.sgprs_read = {3};
  EXPECT_EQ(ws->scoreboard.earliest_issue(rd_s, /*now=*/20), 21u);    // 20 + 1
}

TEST(TryIssue, IssuesReadyValuAtNow) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  ws->pending.push_back(valu_instr());
  auto d = m.try_issue_head_at(*ws, /*now=*/5);
  EXPECT_EQ(d.result, IssueResult::ISSUED);
  EXPECT_TRUE(ws->pending.empty());
  EXPECT_EQ(m.simd(0).valu.next_free_cyc, 6u);   // now + issue_rate(1)
}

TEST(TryIssue, BlocksOnScoreboard) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  InstrRegSet w; w.vgprs_written = {2}; ws->scoreboard.mark_writes(w, 30);
  PendingEvent e = valu_instr(); e.instr.regs.vgprs_read = {2};
  ws->pending.push_back(std::move(e));
  auto d = m.try_issue_head_at(*ws, /*now=*/5);
  EXPECT_EQ(d.result, IssueResult::BLOCKED_SCOREBOARD);
  EXPECT_EQ(d.earliest_retry, 30u);
  EXPECT_FALSE(ws->pending.empty());             // not consumed
}

TEST(TryIssue, InlineFastPathIssuesWithoutQueue) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  InstrEvent ev; ev.kind = InstrKind::VALU; ev.mnemonic = "v_add_f32";
  EXPECT_EQ(m.try_issue_inline(*ws, ev, /*now=*/5), IssueResult::ISSUED);
  EXPECT_TRUE(ws->pending.empty());              // never queued
  EXPECT_EQ(m.simd(0).valu.next_free_cyc, 6u);
}

TEST(SchedulerStep, FourSimdValuConcurrent) {
  ArchModel m(tiny_cfg());
  for (uint32_t s = 0; s < 4; ++s) { auto* ws = make_wave(m, s); ws->pending.push_back(valu_instr()); }
  m.cu().cu_cycle = 0;
  auto r = m.scheduler_step();
  EXPECT_EQ(r.issues, 4u);             // per-SIMD VALU pipe + per-SIMD front-end => 4 IPC
}

TEST(SchedulerStep, TwoWavesSameSimdContend) {
  ArchModel m(tiny_cfg());
  auto* a = make_wave(m, 0); a->pending.push_back(valu_instr());
  auto* b = make_wave(m, 0); b->pending.push_back(valu_instr());
  m.cu().cu_cycle = 0;
  auto r = m.scheduler_step();
  EXPECT_EQ(r.issues, 1u);             // front-end is 1/SIMD/cycle
  EXPECT_EQ(r.earliest_retry, 1u);     // issued wave's next-cycle deadline
  (void)b;
}

TEST(TickLoop, DependentChainLetsOtherWaveFill) {
  ArchModel m(tiny_cfg());
  auto* a = make_wave(m, 0);                 // 3-instr dependent chain on v0
  for (int i = 0; i < 3; ++i) {
    auto e = valu_instr(); e.instr.regs.vgprs_read = {0}; e.instr.regs.vgprs_written = {0};
    a->pending.push_back(std::move(e));
  }
  auto* b = make_wave(m, 1);                 // 3 independent instrs on SIMD1
  for (int i = 0; i < 3; ++i) b->pending.push_back(valu_instr());
  run_cycles(m, 100);                        // 100 cycles
  EXPECT_TRUE(a->pending.empty());
  EXPECT_TRUE(b->pending.empty());
  EXPECT_GT(m.cu().cu_cycle, 0u);
}

TEST(TickLoop, IdleAdvanceIsConstantTime) {
  ArchModel m(tiny_cfg());
  run_cycles(m, 1000);
  EXPECT_EQ(m.cu().cu_cycle, 1000u);
}

TEST(Scheduler, SaluCrossSimdSerializes) {
  ArchModel m(tiny_cfg());
  auto salu = [] { auto e = valu_instr(); e.instr.kind = InstrKind::SALU; e.instr.mnemonic = "s_add_u32"; return e; };
  auto* a = make_wave(m, 0); a->pending.push_back(salu());
  auto* b = make_wave(m, 1); b->pending.push_back(salu());
  m.cu().cu_cycle = 0;
  auto r = m.scheduler_step();
  EXPECT_EQ(r.issues, 1u);             // shared CU-wide SALU pipe blocks the 2nd
  (void)a; (void)b;
}

TEST(Scheduler, LivelockGuardHolds) {
  ArchModel m(tiny_cfg());
  auto* a = make_wave(m, 0);
  auto e = valu_instr(); e.instr.regs.vgprs_read = {0}; e.instr.regs.vgprs_written = {0};
  a->pending.push_back(std::move(e));
  InstrRegSet w; w.vgprs_written = {0}; a->scoreboard.mark_writes(w, 5);   // ready at cyc 5
  run_cycles(m, 50);                   // 50 cycles; must terminate, not spin
  EXPECT_TRUE(a->pending.empty());
}

// --- Memory-request lifecycle (M2 commit 2) ---------------------------------

static InstrEvent vmem_load() {
  InstrEvent e; e.kind = InstrKind::VMEM; e.wcnt = WaitCounter::VMCNT;
  e.mnemonic = "global_load_dword";
  return e;
}
// Issue a VMEM load inline and return its allocated rid (newest in_flight key).
static MemReqId issue_mem(ArchModel& m, CycleWaveState& ws, uint64_t now) {
  MemReqId before = m.cu().next_mem_rid;
  EXPECT_EQ(m.try_issue_inline(ws, vmem_load(), now), IssueResult::ISSUED);
  return before;   // alloc_mem_req_id() returned the pre-increment value
}

TEST(MemLifecycle, CommitPopulatesInFlightAndOutstanding) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  MemReqId rid = issue_mem(m, *ws, /*now=*/0);
  EXPECT_EQ(ws->in_flight.size(), 1u);
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 1u);
  EXPECT_EQ(ws->in_flight.at(rid).complete_cyc, 200u);   // 0 + vmem base_latency
}

TEST(MemLifecycle, AdvanceRetiresAtCompleteCyc) {
  auto cfg = tiny_cfg();
  ArchModel m(cfg);
  SharedMemModel shared(cfg);
  auto* ws = make_wave(m, 0);
  issue_mem(m, *ws, /*now=*/0);
  run_cycles_loopback(m, shared, 150);                   // cyc 150 < 200: still in flight
  EXPECT_EQ(ws->in_flight.size(), 1u);
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 1u);
  run_cycles_loopback(m, shared, 100);                   // cyc 250 >= 200: retired
  EXPECT_TRUE(ws->in_flight.empty());
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 0u);
}

TEST(MemLifecycle, OutOfOrderCompletion) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  MemReqId a = issue_mem(m, *ws, /*now=*/0);
  MemReqId b = issue_mem(m, *ws, /*now=*/1);
  EXPECT_GT(b, a);                                       // monotonic
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 2u);
  EXPECT_TRUE(m.complete_mem_req(*ws, b));               // newer first
  EXPECT_TRUE(m.complete_mem_req(*ws, a));
  EXPECT_TRUE(ws->in_flight.empty());
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 0u);
  EXPECT_EQ(ws->mem_counters.dup_completions, 0u);
  EXPECT_EQ(ws->mem_counters.terminal_unknown, 0u);
}

TEST(MemLifecycle, DuplicateCompletion) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  MemReqId rid = issue_mem(m, *ws, /*now=*/0);
  EXPECT_TRUE(m.complete_mem_req(*ws, rid));
  EXPECT_FALSE(m.complete_mem_req(*ws, rid));            // already terminal
  EXPECT_EQ(ws->mem_counters.dup_completions, 1u);
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 0u);   // no underflow
}

TEST(MemLifecycle, UnknownTerminal) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  EXPECT_FALSE(m.complete_mem_req(*ws, /*rid=*/999));    // never issued
  EXPECT_EQ(ws->mem_counters.terminal_unknown, 1u);
}

TEST(MemLifecycle, WaveKilledMidFlight) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  MemReqId rid = issue_mem(m, *ws, /*now=*/0);
  issue_mem(m, *ws, /*now=*/1);
  m.drain_wave_at_halt(*ws);
  EXPECT_EQ(ws->mem_counters.leak_at_halt, 2u);
  EXPECT_TRUE(ws->in_flight.empty());
  EXPECT_TRUE(ws->outstanding.empty());
  EXPECT_TRUE(ws->halted_rids.contains(rid));
  // A late completion after halt is classified, not under-flowed.
  EXPECT_FALSE(m.complete_mem_req(*ws, rid));
  EXPECT_EQ(ws->mem_counters.late_completion, 1u);
}

TEST(MemLifecycle, MonotonicRidAcrossDispatches) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  MemReqId r0 = issue_mem(m, *ws, /*now=*/0);
  ws->reset_for_dispatch(/*cu_cycle=*/0);               // slot reuse: clears in_flight
  EXPECT_TRUE(ws->in_flight.empty());
  MemReqId r1 = issue_mem(m, *ws, /*now=*/10);           // later cyc: front-end slot free
  EXPECT_GT(r1, r0);                                     // allocator never resets
}

// --- Positional waitcnt / barrier gates (M2 commit 3) -----------------------

static PendingEvent vmem_pending() {
  PendingEvent e; e.kind = PendingKind::Instruction; e.metadata_ready = true;
  e.instr = vmem_load();
  return e;
}
static PendingEvent waitcnt_gate(uint8_t vmcnt) {
  PendingEvent e; e.kind = PendingKind::WaitcntGate; e.metadata_ready = true;
  e.waitcnt.target.t[idx(WaitCounter::VMCNT)] = vmcnt;   // other slots stay 0xFF
  return e;
}
static PendingEvent barrier_gate() {
  PendingEvent e; e.kind = PendingKind::BarrierGate; e.metadata_ready = true;
  return e;
}

TEST(Gates, WaitcntPositionalBlocksUntilCompletion) {
  auto cfg = tiny_cfg();                                 // vmem latency 200
  ArchModel m(cfg);
  SharedMemModel shared(cfg);
  auto* ws = make_wave(m, 0);
  ws->pending.push_back(vmem_pending());                 // outstanding vmcnt -> 1
  ws->pending.push_back(waitcnt_gate(0));                // wait vmcnt(0)
  ws->pending.push_back(valu_instr());                   // gated consumer
  run_cycles_loopback(m, shared, 300);                   // 300 cyc >> 200
  EXPECT_TRUE(ws->pending.empty());                      // all drained
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 0u);
  EXPECT_GE(m.cu().cu_cycle, 200u);                       // waited for completion
}

TEST(Gates, StaleWaitcntDoesNotReblock) {
  auto cfg = tiny_cfg();
  ArchModel m(cfg);
  SharedMemModel shared(cfg);
  auto* ws = make_wave(m, 0);
  // No outstanding: vmcnt(0) gate is already satisfied -> clears free. A later
  // memop must NOT be re-blocked by the (already popped, positional) gate.
  ws->pending.push_back(waitcnt_gate(0));
  ws->pending.push_back(vmem_pending());
  run_cycles_loopback(m, shared, 50);                    // 50 cyc < 200
  EXPECT_TRUE(ws->pending.empty());                      // gate cleared, memop issued
  EXPECT_EQ(ws->in_flight.size(), 1u);                   // memop still in flight (not gated)
  EXPECT_EQ(ws->outstanding.c[idx(WaitCounter::VMCNT)], 1u);
}

TEST(Gates, BarrierBlocksUntilSignaled) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  ws->pending.push_back(barrier_gate());
  ws->pending.push_back(valu_instr());
  run_cycles(m, 50);                                     // unsignaled: blocked, no spin
  EXPECT_EQ(ws->pending.size(), 2u);                     // still parked at barrier
  ws->barrier_signals++;                                 // adapter's onAmdgpuBarrierResolved
  m.tick_cycle();                                        // one edge to replay
  EXPECT_TRUE(ws->pending.empty());                      // barrier cleared + tail issued
  EXPECT_EQ(ws->barrier_signals, 0u);                    // signal consumed
}

// Passive LD_PRELOAD path: a wave with two s_barrier accumulates BOTH resolves on its
// signal counter before the FIFO is ever drained. A bool would collapse the 2nd resolve
// and park the wave at BarrierGate #2 forever, silently dropping the tail (undercount).
// The counter must let both gates clear and the tail issue.
TEST(Gates, MultipleBarrierResolvesDoNotCollapse) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  ws->pending.push_back(barrier_gate());                 // barrier #1
  ws->pending.push_back(valu_instr());                   // body
  ws->pending.push_back(barrier_gate());                 // barrier #2
  ws->pending.push_back(valu_instr());                   // tail (dropped under the bool bug)
  ws->barrier_signals += 2;                              // both resolved before any drain
  m.flush_to_quiescence();
  EXPECT_TRUE(ws->pending.empty());                      // both gates cleared, tail issued
  EXPECT_EQ(ws->barrier_signals, 0u);                    // both signals consumed
}

TEST(Gates, FlushTerminatesOnBarrierBlockedWave) {       // guard: no infinite +1 spin
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  ws->pending.push_back(barrier_gate());                 // never signaled
  m.flush_to_quiescence();                               // must return, not hang
  EXPECT_EQ(ws->pending.size(), 1u);                     // left parked (external event owed)
}

// Review #7: a wave with EMPTY pending but a still-outstanding async in_flight memop
// must be drained by flush_to_quiescence (advance cu_cycle to its completion and retire
// it). The old loop condition (any_pending only) exited immediately on such tails.
TEST(Gates, FlushDrainsInFlightTailWithEmptyPending) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  // Manually plant a single outstanding VMEM completing at cyc 500. Mirrors the state
  // attempt_issue + on_mem_completion leave when the last instruction is a shared memop.
  MemReqEntry e;
  e.kind = MemKind::VMEM_LOAD; e.wcnt = WaitCounter::VMCNT;
  e.complete_cyc = 500; e.issue_cyc = 0;
  ws->outstanding.increment(WaitCounter::VMCNT);
  ws->in_flight.emplace(/*rid=*/1, std::move(e));
  m.flush_to_quiescence();
  EXPECT_TRUE(ws->in_flight.empty());                    // retired by flush
  EXPECT_TRUE(ws->outstanding.empty());                  // vmcnt decremented at retire
}

// #1 runtime: an ArchModel stored node-stably in a map (as the adapter does via
// try_emplace) must read its own live cfg_ on a memory access — not a dangling
// reference. tiny_cfg has no cache geometry, so access() returns the flat per-kind
// pipe latency, which still dereferences cfg_ (vmem.base_latency == 200).
TEST(ArchModelLifetime, MapStoredModelReadsLiveConfig) {
  std::unordered_map<int, ArchModel> models;
  models.try_emplace(0, tiny_cfg());                     // in-place: no move, cfg_ self-ref valid
  ArchModel& m = models.at(0);
  auto* ws = make_wave(m, 0);
  InstrEvent e; e.kind = InstrKind::VMEM; e.mnemonic = "global_load_dword";
  uint64_t retry = UINT64_MAX;
  (void)retry;
  // Issue a VMEM at cyc 0; complete_cyc must reflect cfg_.vmem.base_latency (200).
  EXPECT_EQ(m.try_issue_inline(*ws, e, 0), IssueResult::ISSUED);
  ASSERT_EQ(ws->in_flight.size(), 1u);
  EXPECT_EQ(ws->in_flight.begin()->second.complete_cyc, 200u);
}

// #2: a wave that halts must be removed from its SIMD's resident-wave list, so the
// scheduler never dereferences a freed CycleWaveState after its plugin-state slot
// is reused on a later dispatch.
TEST(WaveResidency, HaltRemovesWaveFromSimd) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  ASSERT_EQ(m.simd(0).waves.size(), 1u);
  m.drain_wave_at_halt(*ws);
  EXPECT_EQ(m.simd(0).waves.size(), 0u);                 // pruned on halt (no dangling ptr)
}

TEST(TickCycle, HasWorkReflectsPendingAndInFlight) {
  ArchModel m(tiny_cfg());
  EXPECT_FALSE(m.has_work());                       // no waves => idle
  auto* ws = make_wave(m, 0);
  EXPECT_FALSE(m.has_work());                        // resident but empty => idle
  ws->pending.push_back(valu_instr());
  EXPECT_TRUE(m.has_work());                          // pending => work
}

TEST(TickCycle, IssuesOnePerCycleAndDrains) {
  ArchModel m(tiny_cfg());
  auto* ws = make_wave(m, 0);
  for (int i = 0; i < 3; ++i) ws->pending.push_back(valu_instr());  // 3 independent VALU
  EXPECT_EQ(m.cu().cu_cycle, 0u);

  EXPECT_TRUE(m.tick_cycle());          // cycle 0 issues one, work remains
  EXPECT_EQ(m.cu().cu_cycle, 1u);
  EXPECT_EQ(ws->pending.size(), 2u);

  EXPECT_TRUE(m.tick_cycle());          // cycle 1 issues one
  EXPECT_EQ(ws->pending.size(), 1u);

  bool more = m.tick_cycle();           // cycle 2 issues the last
  EXPECT_TRUE(ws->pending.empty());
  EXPECT_FALSE(more);                    // drained => idle
  EXPECT_EQ(m.cu().cu_cycle, 3u);
}
