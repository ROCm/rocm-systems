// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// M3 memory-submodel tests. Drives MemorySystem::access() with crafted MemAccess
/// address streams against the cdna4 sidecar config — no rocjitsu needed.

#include "cycle_model/Coalescer.h"
#include "cycle_model/CycleWaveState.h"
#include "cycle_model/MemorySystem.h"
#include "cycle_model/SharedMemModel.h"
#include "cycle_model/UarchConfig.h"

#include <gtest/gtest.h>

namespace {
using namespace cycle_model;

// Test helper: reproduce synchronous end-to-end latency by immediately servicing the
// shared lines and resolving the L1 fill / MSHR in the same call (loopback pattern).
uint64_t mem_latency(MemorySystem& mem, SharedMemModel& shared,
                     const InstrEvent& inst, uint64_t now, CycleWaveState& ws) {
  MemorySystem::AccessResult r = mem.access(inst, now, ws);
  uint64_t complete = r.local_ready_cyc;
  if (!r.shared.empty()) complete = std::max(complete, shared.service(r.shared));
  mem.resolve_fills(r.mshr_slots, r.l1_fill_lines, /*is_smem=*/inst.kind == InstrKind::SMEM, complete);
  return complete - now;
}

// Every memory-submodel test loads the cdna4 sidecar config and needs a scratch
// wave-state. The fixture owns both. MemorySystem is non-movable (it holds a
// const UarchConfig& into our cfg), so tests construct `MemorySystem mem(cfg);`
// in place from the fixture's cfg — and remain free to mutate cfg beforehand or
// build several independent instances (e.g. to decouple shared L2/HBM queues).
class MemSys : public testing::Test {
 protected:
  void SetUp() override { cfg = load_uarch_config(CYCLE_MODEL_CDNA4_JSON); }
  UarchConfig cfg;
  CycleWaveState ws;
};

InstrEvent mem_instr(InstrKind k, uint64_t addr = 0x1000, uint32_t elem_bytes = 4,
                     uint8_t mtype = 2 /*RW*/) {
  InstrEvent e;
  e.kind = k;
  e.mnemonic = "test_mem";
  e.mem = std::make_unique<MemAccess>();
  e.mem->lane_addr[0] = addr;
  e.mem->lane_mask = 1;
  e.mem->elem_bytes = elem_bytes;
  e.mem->mtype = mtype;
  return e;
}

// Constants from cdna4.json: l1v hit=6, l1s hit=4, l2 hit=80, hbm_access_latency=300,
// l2_bytes_per_cycle=1024, hbm 8 channels. Mtype RW=2 goes through L1; UC=0 / NT bypass.
// Cold (L1 miss + L2 miss -> HBM) = l1.hit + l2.hit + hbm_access_latency; the BW queues
// add 0 cycles for a single sub-cache-line-rate transaction (they only bite under load).
constexpr uint8_t MTYPE_RW = 2, MTYPE_UC = 0, MTYPE_NT = 4;
constexpr uint64_t VMEM_COLD = 6 + 80 + 300;   // 386
constexpr uint64_t SMEM_COLD = 4 + 80 + 300;   // 384
constexpr uint64_t L2_HIT_AFTER_L1_MISS = 6 + 80;   // 86

// A cold line misses L1 and L2 -> full hierarchy latency. Separate MemorySystem
// instances so the shared L2 bandwidth queue does not couple the two probes.
TEST_F(MemSys, ColdLineMissesL1AndL2) {
  MemorySystem mv(cfg); SharedMemModel shared_v(cfg);
  EXPECT_EQ(mem_latency(mv, shared_v, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 0, ws), VMEM_COLD);
  MemorySystem ms(cfg); SharedMemModel shared_s(cfg);
  EXPECT_EQ(mem_latency(ms, shared_s, mem_instr(InstrKind::SMEM, 0x80000, 4, MTYPE_RW), 0, ws), SMEM_COLD);
}

// A re-read of a resident, ready L1 line hits at L1 hit_latency.
TEST_F(MemSys, WarmL1LineHits) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 0, ws), VMEM_COLD);     // ready at 486
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 1000, ws), 6u);         // L1 hit
}

// Second access to the same line before its fill returns coalesces onto the
// in-flight miss (completes at ready_cyc), not a false fast hit (finding ①).
TEST_F(MemSys, PendingFillCoalesces) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 0, ws), VMEM_COLD);     // ready 486
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 50, ws), VMEM_COLD - 50);
}

// Findings #1/#2 (regression): the sibling-during-fill case the loopback helper above
// hides. Two raw access() calls to the same line WITHOUT a resolve_fills between them
// (the real async in-flight window): the second hits a line whose ready_cyc is still
// kUnresolved. It must coalesce via the shared path, NOT return the kUnresolved sentinel
// as a synchronous completion -- which attempt_issue would store as complete_cyc=MAX,
// hanging the wave forever. (Also covers #2: an orphaned pending slot left by a halted
// wave is the same unresolved-line re-access at this layer; lazy-healed via l1_fill_lines.)
TEST_F(MemSys, PendingHitCoalescesViaSharedNotSentinel) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  // First access: cold miss installs a pending L1 line (ready_cyc = kUnresolved).
  // Deliberately do NOT resolve_fills -- model the in-flight window.
  MemorySystem::AccessResult a1 = mem.access(mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 0, ws);
  ASSERT_FALSE(a1.shared.empty());                          // cold miss -> shared
  // Second access to the SAME line while the first fill is still in flight.
  MemorySystem::AccessResult a2 = mem.access(mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 50, ws);
  EXPECT_FALSE(a2.shared.empty());                          // coalesced onto the in-flight miss
  EXPECT_NE(a2.local_ready_cyc, MemorySystem::kUnresolved); // no sentinel escape into local
  EXPECT_FALSE(a2.l1_fill_lines.empty());                   // heals the pending slot at completion
  EXPECT_TRUE(a2.mshr_slots.empty());                       // coalesced -> claims no new MSHR
  uint64_t done = a2.shared.empty() ? a2.local_ready_cyc
                                    : std::max(a2.local_ready_cyc, shared.service(a2.shared));
  EXPECT_NE(done, MemorySystem::kUnresolved);               // resolves to a finite completion
}

// Finding #6: resolve_fills must route to ONLY the cache the access targeted. A VMEM
// (l1v) and SMEM (l1s) pending fill sharing the same numeric line_base must NOT
// cross-resolve at each other's completion.
TEST_F(MemSys, ResolveFillsRoutesPerCacheNoCrossResolve) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  // Cold VMEM at 0x1000 -> l1v pending slot at base 0x1000.
  auto v = mem.access(mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 0, ws);
  ASSERT_FALSE(v.shared.empty());
  // Cold SMEM at the same address -> l1s pending slot at base 0x1000 (same numeric base).
  auto s1 = mem.access(mem_instr(InstrKind::SMEM, 0x1000, 4, MTYPE_RW), 0, ws);
  ASSERT_FALSE(s1.shared.empty());
  // Resolve ONLY the VMEM fill. With the routing fix, the SMEM slot stays pending.
  mem.resolve_fills(v.mshr_slots, v.l1_fill_lines, /*is_smem=*/false, /*complete=*/500);
  // Re-touch SMEM line at now=200: if cross-resolve had happened (pre-fix), the SMEM
  // slot would be ready_cyc=500 (resolved-future) -> shared empty. With the fix the
  // slot is still kUnresolved -> #1 pending-hit path -> goes to shared again.
  auto s2 = mem.access(mem_instr(InstrKind::SMEM, 0x1000, 4, MTYPE_RW), 200, ws);
  EXPECT_FALSE(s2.shared.empty());
}

// L1 miss but L2 hit: an NT access installs the line in L2 (not L1); a later
// cacheable access L1-misses but L2-hits. NT is the L2-only bypass mode.
TEST_F(MemSys, L1MissL2Hit) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_NT), 0, ws);               // installs L2 only
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 1000, ws),
            L2_HIT_AFTER_L1_MISS);                                                                // 6 + 80
}

// Review #1: UC (mtype=0) bypasses BOTH L1 and L2 -> straight to HBM, installing no
// cache tag. So a later cacheable access to the same line is still a full cold miss
// (UC did NOT warm L2). The UC access itself pays only the HBM channel + DRAM latency
// (no L1/L2 probe latency). Only NT is the L2-only mode.
TEST_F(MemSys, UcBypassesAllCaches) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_UC), 0, ws),
            cfg.hbm_access_latency);                  // HBM-direct, no L1/L2 latency (300)
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 1000, ws),
            VMEM_COLD);                                // still cold: UC installed no L2 tag
}

// All MSHRs busy -> a new cold-miss access cannot admit; admit_probe returns a
// future retry cycle and mutates nothing (idempotent).
TEST_F(MemSys, MshrExhaustionBlocksIssue) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  // Saturate all 16 MSHRs with distinct cold-miss lines at now=0 (each fills at 486).
  for (uint32_t i = 0; i < cfg.mshrs_per_l1v; ++i)
    mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x10000 + uint64_t(i) * 0x1000, 4, MTYPE_RW), 0, ws);
  // A 17th distinct cold line at now=0 has no free MSHR -> blocked, retry at first free (486).
  InstrEvent newline = mem_instr(InstrKind::VMEM, 0x99000, 4, MTYPE_RW);
  uint64_t r1 = mem.admit_probe(newline, 0, ws);
  EXPECT_GT(r1, 0u);
  EXPECT_EQ(r1, mem.admit_probe(newline, 0, ws));       // read-only: no mutation between probes
  EXPECT_LE(mem.admit_probe(newline, 1000, ws), 1000u); // after the fills free, it admits
}

// A secondary miss to a line with an in-flight fill consumes NO new MSHR, so it
// does not contribute to MSHR exhaustion.
TEST_F(MemSys, SecondaryMissNeedsNoMshr) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  // Fill 15 of 16 MSHRs, then one more line (16th MSHR), then re-touch that line.
  for (uint32_t i = 0; i < cfg.mshrs_per_l1v - 1; ++i)
    mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x10000 + uint64_t(i) * 0x1000, 4, MTYPE_RW), 0, ws);
  InstrEvent x = mem_instr(InstrKind::VMEM, 0x99000, 4, MTYPE_RW);
  mem_latency(mem, shared, x, 0, ws);                  // 16th MSHR claimed
  EXPECT_LE(mem.admit_probe(x, 0, ws), 0u);             // re-touch x: pending, needs no MSHR -> admits
}

// Review #8: an unfilled MemAccess (mtype left at its default) must model a normal
// CACHED access, not a UC bypass. mtype==0 (UC) is the explicit uncached encoding and
// must be set only by the adapter from a real Mtype — never the default sentinel.
TEST_F(MemSys, DefaultMemAccessIsCachedNotBypass) {
  EXPECT_NE(MemAccess{}.mtype, 0);                  // default must not be the UC sentinel
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  InstrEvent e;
  e.kind = InstrKind::VMEM;
  e.mnemonic = "global_load_dword";
  e.mem = std::make_unique<MemAccess>();
  e.mem->lane_addr[0] = 0x5000; e.mem->lane_mask = 1; e.mem->elem_bytes = 4;  // mtype left default
  EXPECT_EQ(mem_latency(mem, shared, e, 0, ws), VMEM_COLD);       // cached cold (L1+L2 miss), not L1-bypass
}

// Review #6: an SMEM access whose 64B L1S sub-lines fall within fewer 128B L2 lines
// must book each physical L2 line's bandwidth ONCE, not once per L1S sub-line. With a
// tight L2 BW (one 128B line per cycle) the bookings become visible: a 256B SMEM access
// = 4 L1S lines over 2 L2 lines (both pre-warmed, so L1S-miss/L2-hit) must serialize as
// 2 L2 transactions (84 + 1 extra cycle = 86), not 4 (which would add more).
TEST_F(MemSys, SmemSubLinesBookEachL2LineOnce) {
  cfg.l2_bytes_per_cycle = 128;                         // one L2 line/cycle -> bookings visible
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  // Pre-warm the 2 covering L2 lines (128B) via vector NT (L2-only, kind-agnostic L2).
  mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x80000, 4, MTYPE_NT), 0, ws);
  mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x80080, 4, MTYPE_NT), 0, ws);
  // 256B scalar load: L1S-misses 4 sub-lines, both L2 lines hit. Deduped -> 2 L2 txns.
  InstrEvent smem = mem_instr(InstrKind::SMEM, 0x80000, /*elem_bytes=*/256, MTYPE_RW);
  EXPECT_EQ(mem_latency(mem, shared, smem, 10000, ws), 86u);
}

// Review #2: reset() clears cross-dispatch cache/MSHR/BW state so a new kernel starts
// with cold caches and a full MSHR pool (reset-per-dispatch: clean per-kernel numbers).
// cu_cycle is monotonic and lives on CUState, so it is intentionally NOT reset here.
TEST_F(MemSys, ResetClearsCrossDispatchState) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 0, ws);                  // warms L1+L2
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 1000, ws), 6u); // warm L1 hit
  // Saturate the MSHR pool with distinct cold-miss lines.
  for (uint32_t i = 0; i < cfg.mshrs_per_l1v; ++i)
    mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x20000 + uint64_t(i) * 0x1000, 4, MTYPE_RW), 1000, ws);

  mem.reset();
  shared.reset();   // reset both per-CU and shared hierarchy together (mirrors the plugin)

  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x1000, 4, MTYPE_RW), 2000, ws), VMEM_COLD); // cold again
  EXPECT_EQ(mem.admit_probe(mem_instr(InstrKind::VMEM, 0x99000, 4, MTYPE_RW), 0, ws), 0u);     // MSHRs free
}

// Review #4: a halted wave's in-flight MSHRs must be released, else they stay claimed
// until their (now-never-arriving) fill cycle and spuriously block live waves. MSHRs
// are tagged with their owning wave; release_wave_mshrs frees only that wave's entries.
TEST_F(MemSys, HaltReleasesOwningWaveMshrs) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  CycleWaveState other;
  // ws saturates the whole MSHR pool with distinct cold-miss lines (each fills at 486).
  for (uint32_t i = 0; i < cfg.mshrs_per_l1v; ++i)
    mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x10000 + uint64_t(i) * 0x1000, 4, MTYPE_RW), 0, ws);
  InstrEvent newline = mem_instr(InstrKind::VMEM, 0x99000, 4, MTYPE_RW);
  EXPECT_GT(mem.admit_probe(newline, 0, ws), 0u);     // pool full -> blocked

  mem.release_wave_mshrs(&other, 0);                   // a different wave halting frees nothing
  EXPECT_GT(mem.admit_probe(newline, 0, ws), 0u);     // still blocked (ownership respected)

  mem.release_wave_mshrs(&ws, 0);                      // ws halts at cycle 0
  EXPECT_EQ(mem.admit_probe(newline, 0, ws), 0u);     // its MSHRs freed -> admits now
}

// LDS stays flat until M3.4 (bank-conflict model).
TEST_F(MemSys, LdsFlatForNow) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, mem_instr(InstrKind::LDS, 0x40, 4, MTYPE_RW), 0, ws), 24u);
}

// ---- M3.3: L2 + HBM bandwidth (ordered per-transaction scheduler, finding ②) ----

// Build a vector access whose active lanes target distinct cache lines at `stride`.
InstrEvent strided_access(uint32_t n, uint64_t stride, uint64_t base = 0x100000) {
  InstrEvent e;
  e.kind = InstrKind::VMEM;
  e.mnemonic = "test_mem";
  e.mem = std::make_unique<MemAccess>();
  for (uint32_t i = 0; i < n; ++i) {
    e.mem->lane_addr[i] = base + uint64_t(i) * stride;
    e.mem->lane_mask |= (1ull << i);
  }
  e.mem->elem_bytes = 4;
  e.mem->mtype = MTYPE_RW;
  return e;
}

// HBM line->channel = (line/128) % 8. Stride 1024 (=8*128) -> all same channel;
// stride 128 -> 8 distinct channels. Same-channel transactions serialize on the one
// channel's bandwidth queue; distinct-channel transactions overlap -> lower latency.
TEST_F(MemSys, SameChannelSerializesMoreThanDistinct) {
  MemorySystem mem_same(cfg); SharedMemModel shared_same(cfg);
  uint64_t same = mem_latency(mem_same, shared_same, strided_access(8, 1024, 0x200000), 0, ws);

  MemorySystem mem_dist(cfg); SharedMemModel shared_dist(cfg);
  uint64_t dist = mem_latency(mem_dist, shared_dist, strided_access(8, 128, 0x200000), 0, ws);

  EXPECT_GT(same, dist);
}

// L2-bandwidth-only saturation: many L1-miss / L2-hit lines (pre-warmed in L2 via UC,
// no HBM) whose total bytes exceed l2_bytes_per_cycle push completion past a single
// L2 hit.
TEST_F(MemSys, L2BandwidthSaturationPushesCompletion) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  const uint32_t n = 16;                              // 16 * 128B = 2048B > 1024 B/cyc
  // Pre-warm L2 for each line via NT (L2-only) accesses (installs L2, not L1).
  for (uint32_t i = 0; i < n; ++i)
    mem_latency(mem, shared, mem_instr(InstrKind::VMEM, 0x300000 + uint64_t(i) * 128, 4, MTYPE_NT), 0, ws);
  // One cacheable access touching all n lines at t=10000 (L1 miss, L2 hit each).
  uint64_t many = mem_latency(mem, shared, strided_access(n, 128, 0x300000), 10000, ws);
  EXPECT_GT(many, L2_HIT_AFTER_L1_MISS);              // BW queue serialized the L2 hits
}

// ---- M3.4: LDS bank conflict ----------------------------------------------

// Build an LDS access with active lanes at explicit byte addresses.
InstrEvent lds_access(std::initializer_list<uint64_t> addrs) {
  InstrEvent e;
  e.kind = InstrKind::LDS;
  e.mnemonic = "ds_op";
  e.mem = std::make_unique<MemAccess>();
  uint32_t lane = 0;
  for (uint64_t a : addrs) { e.mem->lane_addr[lane] = a; e.mem->lane_mask |= (1ull << lane); ++lane; }
  e.mem->elem_bytes = 4;
  e.mem->mtype = MTYPE_RW;
  return e;
}
InstrEvent lds_strided(uint32_t n, uint64_t stride) {
  InstrEvent e;
  e.kind = InstrKind::LDS; e.mnemonic = "ds_op"; e.mem = std::make_unique<MemAccess>(); e.mem->elem_bytes = 4;
  for (uint32_t i = 0; i < n; ++i) { e.mem->lane_addr[i] = uint64_t(i) * stride; e.mem->lane_mask |= (1ull << i); }
  return e;
}

constexpr uint64_t LDS_BASE = 24;   // lds_pipe.base_latency

// All lanes the same address: a broadcast, no bank conflict -> base latency.
TEST_F(MemSys, BroadcastNoConflict) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, lds_access({0x40, 0x40, 0x40, 0x40}), 0, ws), LDS_BASE);
}

// Consecutive dwords map to consecutive banks: no conflict -> base latency.
TEST_F(MemSys, ConflictFreeStride) {                       // cdna4 has 32 banks
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, lds_strided(32, 4), 0, ws), LDS_BASE);   // stride 4B = bank i%32
}

// 32 lanes, distinct words all in one bank (stride = banks*4 = 128B): 32-way conflict
// -> base + 31 serialized cycles.
TEST_F(MemSys, ThirtyTwoWayConflict) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, lds_strided(32, 128), 0, ws), LDS_BASE + 31);
}

// Two distinct words in the same bank: a 2-way conflict -> base + 1.
TEST_F(MemSys, TwoWayConflict) {
  MemorySystem mem(cfg); SharedMemModel shared(cfg);
  EXPECT_EQ(mem_latency(mem, shared, lds_access({0x0, 0x80}), 0, ws), LDS_BASE + 1);   // both bank 0
}

// ---- M3.1: Coalescer (self-contained unit) --------------------------------

MemAccess vec_access(std::initializer_list<uint64_t> addrs, uint32_t elem_bytes = 4) {
  MemAccess a;
  uint32_t lane = 0;
  for (uint64_t ad : addrs) {
    a.lane_addr[lane] = ad;
    a.lane_mask |= (1ull << lane);
    ++lane;
  }
  a.elem_bytes = elem_bytes;
  return a;
}

TEST(Coalescer, ContiguousLanesShareLines) {
  MemAccess a;
  for (uint32_t i = 0; i < 64; ++i) { a.lane_addr[i] = 0x1000 + i * 4; a.lane_mask |= (1ull << i); }
  a.elem_bytes = 4;                              // 64*4 = 256B contiguous
  EXPECT_EQ(coalesce(a, 128).size(), 2u);        // 256B / 128B line = 2 transactions
}

TEST(Coalescer, ScatterGivesDistinctLines) {
  EXPECT_EQ(coalesce(vec_access({0x0, 0x1000, 0x2000, 0x3000}), 128).size(), 4u);
}

TEST(Coalescer, BroadcastOneLine) {
  MemAccess a;
  for (uint32_t i = 0; i < 64; ++i) { a.lane_addr[i] = 0x2000; a.lane_mask |= (1ull << i); }
  a.elem_bytes = 4;
  EXPECT_EQ(coalesce(a, 128).size(), 1u);
}

TEST(Coalescer, SizeSpanningStraddlesTwoLines) {
  // one lane, 64B access at [96,160) straddles the 128B line boundary at 128.
  EXPECT_EQ(coalesce(vec_access({96}, /*elem_bytes=*/64), 128).size(), 2u);
}

// Review #7: a misconfigured line size must not hang the coalescer (base += 0 loops
// forever) — it yields no transactions instead.
TEST(Coalescer, ZeroLineBytesYieldsNoTransactions) {
  EXPECT_TRUE(coalesce(vec_access({0x0, 0x40}), 0).empty());
}

// Review #7: strict config validation rejects a zero / non-power-of-two cache line.
TEST(UarchConfigValidate, RejectsBadCacheLineBytes) {
  auto good = load_uarch_config(CYCLE_MODEL_CDNA4_JSON);
  EXPECT_NO_THROW(validate_uarch_config(good));
  auto c = good; c.l1v.line_bytes = 96;                 // not a power of two
  EXPECT_THROW(validate_uarch_config(c), std::runtime_error);
  auto z = good; z.l2.line_bytes = 0;                   // zero line size
  EXPECT_THROW(validate_uarch_config(z), std::runtime_error);
}

TEST(Coalescer, InactiveLanesIgnored) {
  MemAccess a;
  a.lane_addr[0] = 0x0; a.lane_addr[2] = 0x4000;
  a.lane_addr[1] = 0xDEAD;            // lane 1 inactive — must be ignored
  a.lane_mask = 0b101;
  a.elem_bytes = 4;
  EXPECT_EQ(coalesce(a, 128).size(), 2u);
}

}  // namespace
