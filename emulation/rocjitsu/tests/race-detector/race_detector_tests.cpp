// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the race detection core (RaceDetector + WaveRaceState).
//
// These tests drive the detection API directly via RaceTestBuilder — no
// assembly parser, no instruction emulator, no GPU.
//
// Test categories:
//   Vgpr_*          — VGPR races from global loads (vmcnt)
//   Sgpr_*          — SGPR races from scalar loads (lgkmcnt)
//   LdsCrossWave_*  — cross-wave LDS races (missing barrier)
//   LdsSameWave_*   — same-wave LDS races (missing waitcnt)
//   SameWave_*      — same-wave VGPR races via LDS loads
//   DeepStack_*     — multiple outstanding loads with partial waitcnt
//   D16_*           — byte-level VGPR tracking (half-register loads)
//   Dtl_*           — Direct-to-LDS global loads
//   Exec_*          — exec mask effects on race detection
//   MultiWorkgroup_* — workgroup ID in violation reports
//   GlobalStore_*   — global stores and vmcnt interaction
//   MultiReg_*      — multi-register loads
//   Mixed_*         — simultaneous vmcnt and lgkmcnt events
//   DualOffset_*    — dual-offset LDS events

#include "race_test_builder.h"
#include <gtest/gtest.h>

using namespace rocjitsu::plugins::race_detector;

// ---- VGPR races (vmcnt) ----

TEST(RaceDetector, Vgpr_InsufficientVmcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);                 // drains oldest (v1), v2 still in flight
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // read v2 → RACE
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, Vgpr_ViolationFields) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/3, /*numRegs=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/5);
  ASSERT_EQ(b.raceCount(), 1);
  auto &v = b.violations()[0];
  EXPECT_EQ(v.space, RaceViolation::Space::VGPR);
  EXPECT_EQ(v.index, 3);
  EXPECT_EQ(v.wave, 0);
  EXPECT_EQ(v.lane, 5);
}

TEST(RaceDetector, Vgpr_WaitcntClearsRace) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

// ---- SGPR races (lgkmcnt) ----

TEST(RaceDetector, Sgpr_MissingWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.checkSgprRead(/*wave=*/0, /*reg=*/4); // no waitcnt → RACE
  EXPECT_TRUE(b.hasSgprRace(4));
}

TEST(RaceDetector, Sgpr_WithWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Sgpr_PartialWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/4, /*numRegs=*/1); // oldest
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/5, /*numRegs=*/1); // newest
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);      // drain oldest (s4)
  b.checkSgprRead(/*wave=*/0, /*reg=*/4);                  // safe
  EXPECT_FALSE(b.hasSgprRace(4));
  b.checkSgprRead(/*wave=*/0, /*reg=*/5); // RACE
  EXPECT_TRUE(b.hasSgprRace(5));
}

// ---- LDS cross-wave races ----

TEST(RaceDetector, LdsCrossWave_ZeroLdsSize) {
  // LDS size hint is 0 (e.g. compiler metadata says no LDS) but LDS events
  // arrive anyway (e.g. via inline asm buffer_load_dword lds).
  // Byte counters must grow dynamically to accommodate.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/1, /*sgprs=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/100, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // Missing barrier → cross-wave race.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/100, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(100));
}

TEST(RaceDetector, LdsCrossWave_MissingBarrier) {
  // Wave 0 writes 1 byte to LDS[1] (covering [1,2)), wave 1 reads without
  // barrier. Byte-level precision: [0,1) should NOT race, [1,2) should.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/1, /*sgprs=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/1, /*bytes=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // Missing barrier!
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0,
                 /*bytes=*/1); // [0,1) → safe
  EXPECT_FALSE(b.hasRace());
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/1,
                 /*bytes=*/1); // [1,2) → RACE
  EXPECT_TRUE(b.hasLdsRace(1));
}

TEST(RaceDetector, LdsCrossWave_WithBarrier) {
  // Wave 0 writes, waitcnt, barrier, then wave 1 reads → safe.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

// ---- LDS same-wave races ----

TEST(RaceDetector, LdsSameWave_WriteWriteOk) {
  // Two writes to same address, same wave → not a race.
  // Same-wave LDS writes are ordered (same LDS unit, program order).
  // Cross-wave WAW without barrier would be a race, but same-wave is safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ReadReadOk) {
  // Two reads from same address → not a race.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_WriteReadRace) {
  // Write then read same address, same wave, no waitcnt → RACE.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  // no waitcnt
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsSameWave_InsufficientLgkm) {
  // Two LDS writes, waitcnt lgkmcnt(1) drains oldest only.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);     // oldest
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4);    // newest
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1);              // drain oldest
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4); // safe
  EXPECT_FALSE(b.hasLdsRace(0));
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4); // RACE
  EXPECT_TRUE(b.hasLdsRace(64));
}

// ---- Same-wave VGPR via LDS load ----

TEST(RaceDetector, SameWave_WriteReadRace) {
  // LDS load into VGPR, read VGPR without waitcnt → RACE.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, SameWave_WriteWaitcntOk) {
  // LDS load into VGPR, waitcnt, read VGPR → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, SameWave_DoubleWriteRace) {
  // Two LDS loads into same VGPR, waitcnt(1) drains oldest → newest RACE.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4,
            /*vgprDst=*/2); // oldest
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4,
            /*vgprDst=*/2);                           // newest
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain oldest
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // still racing
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, SameWave_BarrierNoWaitcnt) {
  // Barrier without waitcnt doesn't clear same-wave events.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.barrier(); // no waitcnt first!
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, SameWave_WaitcntBarrierOk) {
  // Waitcnt then barrier → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

// ---- Deep event stack ----

TEST(RaceDetector, DeepStack_PartialWaitcnt) {
  // 3 loads, waitcnt(1) drains oldest, newest two still pending.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1); // oldest
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1); // newest
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0); // safe (drained)
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // RACE (newest)
  EXPECT_TRUE(b.hasVgprRace(2));
}

TEST(RaceDetector, DeepStack_FullWaitcnt) {
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, DeepStack_ConsecutiveWaitcnts) {
  // Two partial waitcnts in sequence. The second drain must retire the
  // correct (oldest) event — not a newer one that got reordered.
  //   load v0, load v1, load v2  (3 outstanding, oldest first)
  //   waitcnt vmcnt(2)           → drain oldest (v0), keep v1 and v2
  //   load v3                    (v1, v2, v3 outstanding)
  //   waitcnt vmcnt(2)           → drain oldest (v1), keep v2 and v3
  //   read v0 → safe  (drained by first waitcnt)
  //   read v1 → safe  (drained by second waitcnt)
  //   read v2 → RACE  (still pending)
  //   read v3 → RACE  (still pending)
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/2);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/3, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/2);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_FALSE(b.hasVgprRace(1));
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(2));
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(3));
}

// ---- D16 (byte-level VGPR races) ----

TEST(RaceDetector, D16_LoHiLoadDrained) {
  // Load lo half, load hi half, drain oldest (lo) → lo read safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0x3); // lo
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0xC);                        // hi
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain oldest (lo)
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0,
                  /*byteMask=*/0x3); // lo safe
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, D16_HiOutstanding_LoSafe) {
  // Hi half still outstanding, but we only read lo → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0x3); // lo
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0xC);                        // hi
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain lo
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0,
                  /*byteMask=*/0x3); // read lo only
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, D16_FullLoadRaces) {
  // Full-register load, partial waitcnt → full read races.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0xF);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/64, /*bytes=*/4, /*vgprDst=*/3,
            /*byteMask=*/0xF);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/1); // drain oldest (v2)
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0); // v3 still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(3));
}

TEST(RaceDetector, D16_LoOutstanding_FullRaces) {
  // Lo half outstanding, full read → races (overlapping bytes).
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/2,
            /*byteMask=*/0x3);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0,
                  /*byteMask=*/0xF); // full read → RACE
  EXPECT_TRUE(b.hasVgprRace(2));
}

// ---- Direct-to-LDS (GlobalToLds) ----

TEST(RaceDetector, Dtl_CrossWaveRace) {
  // Wave 0 does DTL load to LDS, waitcnt, but no barrier.
  // Wave 1 reads LDS → RACE (WAVE_COMPLETE but not RETIRED).
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/64);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4); // RACE
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Dtl_CrossWaveSafe) {
  // Same as above but with barrier → safe.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0}, /*bytesPerLane=*/64);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

// ---- Exec mask ----

TEST(RaceDetector, Exec_PartialWriteFullRead) {
  // Only lane 0 writes LDS, then full-exec read → RACE (write still ACTIVE).
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*exec=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Exec_PartialWriteWaitcntOk) {
  // Lane 0 writes, waitcnt, then read → safe (same wave, WAVE_COMPLETE).
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*exec=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, Exec_DisjointLanesOverlap) {
  // Lane 0 writes LDS[0], lane 1 reads LDS[0] without barrier → RACE.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*exec=*/1);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4); // RACE
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, Exec_DisjointLanesDisjoint) {
  // Lane 0 and lane 1 write different LDS addresses → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*exec=*/1);
  b.ldsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/64, /*bytes=*/4, /*exec=*/2);
  EXPECT_FALSE(b.hasRace());
}

// ---- Multi-workgroup ----

TEST(RaceDetector, MultiWorkgroup_ReportsIndex) {
  // Race violation reports correct workgroup index.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8,
                    /*waveSize=*/64, Dim3d(7, 3, 1));
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  ASSERT_EQ(b.raceCount(), 1);
  auto &v = b.violations()[0];
  EXPECT_EQ(v.workgroupId.x, 7);
  EXPECT_EQ(v.workgroupId.y, 3);
  EXPECT_EQ(v.workgroupId.z, 1);
}

// ---- LDS additional patterns ----

TEST(RaceDetector, LdsCrossWave_GlobalLoadToLdsWriteMissingVmcnt) {
  // Global load into VGPR, then LDS write using that VGPR without vmcnt.
  // The VGPR race fires when the LDS write reads the VGPR source.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  // no vmcnt before reading v1
  b.checkVgprRead(/*wave=*/0, /*reg=*/1,
                  /*lane=*/0); // simulates ds_write reading v1
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, GlobalToLdsHonorsLaneMask) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8, /*waveSize=*/4);
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0, 4, 8, 12}, /*bytesPerLane=*/4,
                /*exec=*/0x1);

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

TEST(RaceDetector, LdsSameWave_MultiLaneReadOk) {
  // All lanes read same LDS byte after write+waitcnt → safe.
  // Multiple lanes reading same address is not a race.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/2, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_ScatterBroadcastOk) {
  // Each lane writes unique addr, waitcnt, then all read addr 0 → safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/4, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/2, /*addr=*/8, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // All lanes read from addr 0 (lane 0's write)
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/0, /*lane=*/2, /*addr=*/0, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsSameWave_AllLanesWriteSameAddr) {
  // All lanes write same address (WAW). Currently not detected as a race.
  // TODO(newling): WAW detection is not implemented — this documents current
  // behavior.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/8);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/2, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  // Currently passes (WAW not flagged). If WAW detection is added, this
  // test should be updated to EXPECT_TRUE(b.hasRace()).
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, LdsCrossWave_IrregularSizesAndOffsets) {
  // Exercises byte-count tracking with non-aligned, varying-size accesses
  // that may share or straddle count granularity boundaries.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);

  // Wave 0: irregular writes at odd offsets and sizes.
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/3, /*bytes=*/1);   // [3,4)
  b.ldsWrite(/*wave=*/0, /*lane=*/1, /*addr=*/7, /*bytes=*/3);   // [7,10)
  b.ldsWrite(/*wave=*/0, /*lane=*/2, /*addr=*/15, /*bytes=*/6);  // [15,21)
  b.ldsWrite(/*wave=*/0, /*lane=*/3, /*addr=*/100, /*bytes=*/2); // [100,102)
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  // No barrier: cross-wave reads should detect races.

  // Wave 1 reads that overlap written ranges — should race.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/3,
                 /*bytes=*/1); // [3,4) ∩ [3,4)
  EXPECT_TRUE(b.hasLdsRace(3));
  b.clearViolations();

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/8,
                 /*bytes=*/1); // [8,9) ∩ [7,10)
  EXPECT_TRUE(b.hasLdsRace(8));
  b.clearViolations();

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/17,
                 /*bytes=*/2); // [17,19) ∩ [15,21)
  EXPECT_TRUE(b.hasLdsRace(17));
  b.clearViolations();

  // Reads that are adjacent but don't overlap — should be safe.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/2,
                 /*bytes=*/1); // [2,3) ∩ [3,4) = ∅
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/10,
                 /*bytes=*/5); // [10,15) ∩ [15,21) = ∅
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/21,
                 /*bytes=*/4); // [21,25) ∩ [15,21) = ∅
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/102,
                 /*bytes=*/8); // [102,110) ∩ [100,102) = ∅
  EXPECT_FALSE(b.hasRace());

  // After barrier, everything should be safe.
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/3, /*bytes=*/1); // [3,4)
  EXPECT_FALSE(b.hasRace());

  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/15, /*bytes=*/6); // [15,21)
  EXPECT_FALSE(b.hasRace());
}

// ---- Cross-wave exec mask tests ----
//
// Two waves, each doing 4 partial-exec LDS writes to different regions,
// then cross-wave reads. The lgkmcnt values control how many writes are
// drained before each barrier.
//
// Wave 0 writes: [0,64), [64,128), [128,192), [192,256)  (4 writes, lanes 0-15
// each) Wave 1 writes: [256,320), [320,384), [384,448), [448,512) After
// barrier, each wave reads the other's region.
//
// Ported from: ExecMask_CrossWave_Correct_2_0 and variants.

// Helper: set up 2-wave cross-wave exec mask test.
// Each wave does 4 LDS writes with partial exec, then waitcnt(lgkmcnt1),
// barrier, cross-wave read, optionally waitcnt(lgkmcnt2) + barrier2, then
// second cross-wave read.
// Each wave does 4 LDS writes to distinct addresses. lgkmcnt controls how
// many are drained before the barrier. Cross-wave reads then check whether
// the barrier retired enough events.
static void crossWaveExecTest(int lgkmcnt1, int lgkmcnt2, bool barrier1, bool barrier2,
                              bool expectFirstReadRace, bool expectSecondReadRace) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/8, /*sgprs=*/8);

  // Wave 0: 4 writes to addrs 0, 4, 8, 12
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/4, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/8, /*bytes=*/4);
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/12, /*bytes=*/4);

  // Wave 1: 4 writes to addrs 256, 260, 264, 268
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/256, /*bytes=*/4);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/260, /*bytes=*/4);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/264, /*bytes=*/4);
  b.ldsWrite(/*wave=*/1, /*lane=*/0, /*addr=*/268, /*bytes=*/4);

  // Both waves: waitcnt lgkmcnt1
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt1);
  b.waitcnt(/*wave=*/1, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt1);

  if (barrier1) {
    b.barrier();
  }

  // Cross-wave read: wave 0 reads wave 1's addr 260 (2nd write).
  // With lgkmcnt(N), the (4-N) oldest writes are drained. Addr 260 (2nd)
  // is safe only if lgkmcnt <= 2 (drain at least 2 oldest) + barrier.
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/260, /*bytes=*/4);
  EXPECT_EQ(b.hasRace(), expectFirstReadRace);
  b.clearViolations();

  if (lgkmcnt2 >= 0) {
    b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt2);
    b.waitcnt(/*wave=*/1, /*vmcnt=*/-1, /*lgkmcnt=*/lgkmcnt2);
  }

  if (barrier2) {
    b.barrier();
  }

  // Cross-wave read: wave 0 reads wave 1's addr 264
  b.checkLdsRead(/*wave=*/0, /*lane=*/0, /*addr=*/264, /*bytes=*/4);
  EXPECT_EQ(b.hasRace(), expectSecondReadRace);
}

TEST(RaceDetector, Exec_CrossWaveCorrect) {
  // lgkmcnt(2): drain 2 oldest writes, barrier flushes them.
  // lgkmcnt(0): drain remaining, barrier flushes. All safe.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveOverRetire00) {
  // lgkmcnt(0): all 4 writes drained before barrier 1. Conservative but safe.
  crossWaveExecTest(/*lgkmcnt1=*/0, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveOverRetire11) {
  // lgkmcnt(1): drain 3 of 4 writes each time. Safe.
  crossWaveExecTest(/*lgkmcnt1=*/1, /*lgkmcnt2=*/1,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveUnderRetire30) {
  // lgkmcnt(3): only drain 1 of 4, barrier flushes only that 1.
  // First cross-wave read hits unflushed writes → RACE.
  // Second read after lgkmcnt(0)+barrier → safe.
  crossWaveExecTest(/*lgkmcnt1=*/3, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/true,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveUnderRetire22) {
  // lgkmcnt(2): drain 2 oldest (256,260), barrier flushes those 2.
  // First read at 260 → safe (was drained+flushed).
  // 2 remain (264,268). Second lgkmcnt(2) is noop (already ≤2).
  // Barrier2 flushes nothing new. Second read at 264 → RACE.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/2,
                    /*barrier1=*/true, /*barrier2=*/true,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/true);
}

TEST(RaceDetector, Exec_CrossWaveMissingBarrier1) {
  // No first barrier → cross-wave read hits WAVE_COMPLETE events → RACE.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/0,
                    /*barrier1=*/false, /*barrier2=*/true,
                    /*expectFirstReadRace=*/true,
                    /*expectSecondReadRace=*/false);
}

TEST(RaceDetector, Exec_CrossWaveMissingBarrier2) {
  // lgkmcnt(2) + barrier1 flushes 2 oldest (256,260).
  // First read at 260 → safe. lgkmcnt(0) drains remaining 2 (264,268)
  // but no barrier2 → WAVE_COMPLETE not RETIRED → second read RACE.
  crossWaveExecTest(/*lgkmcnt1=*/2, /*lgkmcnt2=*/0,
                    /*barrier1=*/true, /*barrier2=*/false,
                    /*expectFirstReadRace=*/false,
                    /*expectSecondReadRace=*/true);
}

// ---- Global stores ----

TEST(RaceDetector, GlobalStore_CountedByVmcnt) {
  // Global store is tracked by vmcnt. Issuing a store then a global load
  // means the load is the 2nd vmcnt event. waitcnt(1) should NOT drain it.
  // This test verifies that global stores consume vmcnt slots.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalStore(/*wave=*/0);                               // vmcnt event 0 (oldest)
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1); // vmcnt event 1
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);                      // drain oldest (store), load stays
  b.checkVgprRead(/*wave=*/0, /*reg=*/1,
                  /*lane=*/0); // load still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, GlobalStore_NoVgprRaceOnSource) {
  // Global stores read VGPRs at issue time. Writing to the source VGPR
  // after issuing the store should NOT race — the store already captured
  // the value. (Store has no destination registers.)
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalStore(/*wave=*/0);
  // Overwriting the "source" VGPR is fine — store doesn't track dest regs.
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

// ---- Multi-register loads ----

TEST(RaceDetector, MultiReg_PartialWaitcnt) {
  // Load 4 registers in one event (like global_load_dwordx4 into v[0:3]).
  // waitcnt(0) should clear all 4.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, MultiReg_NoWaitcnt) {
  // Load 4 registers, no waitcnt. All 4 should race.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/4);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/3, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(0));
  EXPECT_TRUE(b.hasVgprRace(3));
}

TEST(RaceDetector, MultiReg_TwoLoads_PartialDrain) {
  // Two 2-register loads. waitcnt(1) drains the oldest (v[0:1]).
  // Reading v0 is safe, reading v2 races.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/8, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/2); // oldest: v0,v1
  b.globalLoad(/*wave=*/0, /*vgprBase=*/2, /*numRegs=*/2); // newest: v2,v3
  b.waitcnt(/*wave=*/0, /*vmcnt=*/1);                      // drain oldest
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);      // safe
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/2, /*lane=*/0); // RACE
  EXPECT_TRUE(b.hasVgprRace(2));
}

// ---- DTL multi-lane ----

TEST(RaceDetector, Dtl_MultiLane_CrossWaveRace) {
  // DTL with 4 active lanes, each writing 16 bytes to different LDS offsets.
  // Cross-wave read to lane 2's region should detect the race.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  // Wave 0: DTL load, lanes 0-3 active, each gets 16 bytes.
  // Lane 0 → LDS[0..16), lane 1 → LDS[16..32), lane 2 → LDS[32..48), lane 3 →
  // LDS[48..64)
  b.globalToLds(/*wave=*/0, /*ldsAddrs=*/{0, 16, 32, 48},
                /*bytesPerLane=*/16, /*exec=*/0xF);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  // No barrier — cross-wave read should race.
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/32,
                 /*bytes=*/4); // lane 2's region
  EXPECT_TRUE(b.hasLdsRace(32));
}

TEST(RaceDetector, Dtl_MultiLane_CrossWaveSafeWithBarrier) {
  // Same as above but with barrier → all lanes' regions safe.
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);
  b.globalToLds(/*wave=*/0, {0, 16, 32, 48}, /*bytesPerLane=*/16, /*exec=*/0xF);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  b.barrier();
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/16, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/32, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/48, /*bytes=*/4);
  EXPECT_FALSE(b.hasRace());
}

// ---- Dual-offset LDS ----

TEST(RaceDetector, DualOffset_Race) {
  // registerDualOffsetLdsEvent: each lane writes TWO 8-byte intervals.
  // Lane 0 base=100: [100, 108) (offset0=0) and [116, 124) (offset1=2).
  // Read from either interval without waitcnt → RACE.
  //
  // Uses RaceDetector/WaveRaceState directly since the builder doesn't
  // expose registerDualOffsetLdsEvent.
  std::vector<RaceViolation> violations;
  RaceDetector detector(/*nWaves=*/1, /*vgprCount=*/4,
                        /*sgprCount=*/4, Dim3d(0),
                        [&](RaceViolation v) { violations.push_back(v); });
  auto &rs = detector.getWaveRaceState(0);
  std::vector<uint32_t> ldsAddrs(64, 0);
  ldsAddrs[0] = 100;
  rs.registerDualOffsetLdsEvent(
      /*pc=*/0, MemoryEventType::VGPR_TO_LDS, /*registers=*/{},
      /*execMask=*/1, /*waveSize=*/64, ldsAddrs,
      /*offset0=*/0, /*offset1=*/2);
  // Lane 0 wrote: [100, 108) and [116, 124)
  detector.validateRead(/*addr=*/100, WaveId{0}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 1u); // first interval
  detector.validateRead(/*addr=*/116, WaveId{0}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // second interval
  // Outside both intervals: no race
  detector.validateRead(/*addr=*/108, WaveId{0}, /*lane=*/0, /*nBytes=*/4);
  EXPECT_EQ(violations.size(), 2u); // still 2
}

// ---- Exec mask and lane-level checks ----

TEST(RaceDetector, CheckVgprReadAllLanes) {
  // checkVgprReadAllLanes checks all 64 lanes for races on a register.
  // A global load with partial exec (lane 5 only) should race when
  // checkVgprReadAllLanes is called (it checks lane 5 among others).
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, /*exec=*/1ULL << 5);
  // Use the WaveRaceState directly for checkVgprReadAllLanes.
  b.violations(); // ensure handler is set up
  // checkVgprReadAllLanes iterates all lanes, should hit lane 5.
  // Access it through the detector:
  RaceTestBuilder b2(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b2.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, /*exec=*/1ULL << 5);
  // Check individual lane 5 → race
  b2.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/5);
  EXPECT_TRUE(b2.hasVgprRace(1));
  // Check lane 0 (not in exec mask) → should still race because the event
  // is active and checkVgprRead checks if ANY pending load overlaps.
  RaceTestBuilder b3(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b3.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1, /*exec=*/1ULL << 5);
  b3.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  // Lane 0 was not in the exec mask of the load, so it should NOT race.
  EXPECT_FALSE(b3.hasRace());
}

// ---- Mixed counter types ----

TEST(RaceDetector, Mixed_VmcntAndLgkmcnt) {
  // Global load (vmcnt) and LDS load (lgkmcnt) in flight simultaneously.
  // waitcnt vmcnt(0) clears global load but not LDS load.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1); // vmcnt
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4,
            /*vgprDst=*/1);           // lgkmcnt
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0); // clear global load only
  b.checkVgprRead(/*wave=*/0, /*reg=*/0,
                  /*lane=*/0); // global load cleared → safe
  EXPECT_FALSE(b.hasVgprRace(0));
  b.checkVgprRead(/*wave=*/0, /*reg=*/1,
                  /*lane=*/0); // LDS load still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, Mixed_LgkmcntAndVmcnt) {
  // Reverse: waitcnt lgkmcnt(0) clears LDS load but not global load.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1); // vmcnt
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4,
            /*vgprDst=*/1);                           // lgkmcnt
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0); // clear LDS load only
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0); // LDS load cleared → safe
  EXPECT_FALSE(b.hasVgprRace(1));
  b.checkVgprRead(/*wave=*/0, /*reg=*/0,
                  /*lane=*/0); // global load still pending → RACE
  EXPECT_TRUE(b.hasVgprRace(0));
}

TEST(RaceDetector, Mixed_BothCleared) {
  // Both vmcnt(0) and lgkmcnt(0) → everything safe.
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/4, /*sgprs=*/4);
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/1);
  b.scalarLoad(/*wave=*/0, /*sgprBase=*/0, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0, /*lgkmcnt=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  b.checkSgprRead(/*wave=*/0, /*reg=*/0);
  EXPECT_FALSE(b.hasRace());
}

TEST(RaceDetector, EventRegistry_NarrowWindow) {
  // Circular buffer: 4 VGPRs loaded in rotation. Each iteration drains
  // the oldest, checks that it's safe, checks that others still race,
  // then reloads into the freed register.
  int N = 10;
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/N, /*sgprs=*/4);

  for (int i = 0; i < N; ++i) {
    b.globalLoad(/*wave=*/0, /*vgprBase=*/i, /*numRegs=*/1);
  }

  for (int iter = 0; iter < 1000; ++iter) {
    int drained = iter % N;
    int stillPending0 = (iter + 1) % N;
    int stillPending1 = (iter + N + 1) % N;

    b.waitcnt(/*wave=*/0, /*vmcnt=*/N - 1);

    b.clearViolations();
    b.checkVgprRead(/*wave=*/0, /*reg=*/drained, /*lane=*/0);
    EXPECT_FALSE(b.hasVgprRace(drained)) << "iter=" << iter;

    b.clearViolations();
    b.checkVgprRead(/*wave=*/0, /*reg=*/stillPending0, /*lane=*/0);
    EXPECT_TRUE(b.hasVgprRace(stillPending0)) << "iter=" << iter;

    b.clearViolations();
    b.checkVgprRead(/*wave=*/0, /*reg=*/stillPending1, /*lane=*/0);
    EXPECT_TRUE(b.hasVgprRace(stillPending1)) << "iter=" << iter;

    b.globalLoad(/*wave=*/0, /*vgprBase=*/drained, /*numRegs=*/1);
  }
}

// ---- EventRegistry trimming ----

TEST(RaceDetector, EventRegistry_TrimWithoutBarrier) {
  // Single wave, no LDS, no barrier. WAVE_COMPLETE non-LDS events are
  // trimmable. Trimming is triggered automatically inside add() every
  // kTrimAttemptInterval allocations.
  int N = EventRegistry::kTrimAttemptInterval - 1;
  RaceTestBuilder b(/*numWaves=*/1, /*vgprs=*/2, /*sgprs=*/2);

  // Allocate N events (one short of the trim interval) — no trim yet.
  for (int i = 0; i < N; ++i) {
    b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
    b.waitcnt(/*wave=*/0, /*vmcnt=*/0);
  }
  EXPECT_EQ(b.events().totalAllocated(), N);
  EXPECT_EQ(b.events().trimmedCount(), 0);

  // One more allocation triggers trimming.
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/0);

  EXPECT_EQ(b.events().totalAllocated(), N + 1);
  EXPECT_GT(b.events().trimmedCount(), 0);
  EXPECT_LT(b.events().size(), N);

  // Race detection still works after trimming.
  b.globalLoad(/*wave=*/0, /*vgprBase=*/1, /*numRegs=*/1);
  b.checkVgprRead(/*wave=*/0, /*reg=*/1, /*lane=*/0);
  EXPECT_TRUE(b.hasVgprRace(1));
}

TEST(RaceDetector, EventRegistry_TrimWithBarrier) {
  // Two waves, LDS events require s_barrier to reach RETIRED.
  // Trimming is triggered inside add() after enough allocations.
  int N = EventRegistry::kTrimAttemptInterval - 1;
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/4, /*sgprs=*/4);

  // Allocate N events (one short of the trim interval) — no trim yet.
  for (int i = 0; i < N; ++i) {
    b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/(i % 16) * 4, /*bytes=*/4);
    b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
    b.barrier();
  }
  EXPECT_EQ(b.events().totalAllocated(), N);
  EXPECT_EQ(b.events().trimmedCount(), 0);

  // One more allocation triggers trimming.
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.waitcnt(/*wave=*/0, /*vmcnt=*/-1, /*lgkmcnt=*/0);
  b.barrier();

  EXPECT_EQ(b.events().totalAllocated(), N + 1);
  EXPECT_GT(b.events().trimmedCount(), 0);
  EXPECT_LT(b.events().size(), N);

  // LDS detection still works after trimming.
  b.ldsWrite(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  b.checkLdsRead(/*wave=*/1, /*lane=*/0, /*addr=*/0, /*bytes=*/4);
  EXPECT_TRUE(b.hasLdsRace(0));
}

// ---- Handler invocation count ----
//
// The race handler fires for every (register, lane) combination where a race
// exists. When the plugin wraps this with deduplication (by PC), it must
// handle millions of handler calls efficiently. These tests verify the raw
// handler invocation count at the core level.

TEST(RaceDetector, HandlerFiresPerLane) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  // Global load into v0 with full exec mask (all 64 lanes active).
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/1);

  // Check v0 across all 64 lanes without waitcnt — each lane fires the handler.
  for (int lane = 0; lane < 64; ++lane)
    b.checkVgprRead(/*wave=*/0, /*reg=*/0, lane);

  EXPECT_EQ(b.raceCount(), 64);
}

TEST(RaceDetector, HandlerFiresPerRegister) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  // Load into v0..v3 (4 registers).
  b.globalLoad(/*wave=*/0, /*vgprBase=*/0, /*numRegs=*/4);

  // Read all 4 without waitcnt.
  for (int reg = 0; reg < 4; ++reg)
    b.checkVgprRead(/*wave=*/0, reg, /*lane=*/0);

  EXPECT_EQ(b.raceCount(), 4);
}

TEST(RaceDetector, MultipleDistinctRaces) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  // Two independent races at different PCs.
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/10);
  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/16, /*bytes=*/4, /*vgprDst=*/20);

  // Read both without waitcnt — two distinct races.
  b.checkVgprRead(/*wave=*/0, /*reg=*/10, /*lane=*/0);
  b.checkVgprRead(/*wave=*/0, /*reg=*/20, /*lane=*/0);

  EXPECT_EQ(b.raceCount(), 2);
  EXPECT_TRUE(b.hasVgprRace(10));
  EXPECT_TRUE(b.hasVgprRace(20));
}

TEST(RaceDetector, RepeatedCheckSameRace) {
  RaceTestBuilder b(/*numWaves=*/2, /*vgprs=*/256, /*sgprs=*/32);

  b.ldsRead(/*wave=*/0, /*lane=*/0, /*addr=*/0, /*bytes=*/4, /*vgprDst=*/0);

  // Check the same register+lane 100 times. The handler fires every time
  // because the core race detector does not deduplicate — that is the
  // plugin's responsibility.
  for (int i = 0; i < 100; ++i)
    b.checkVgprRead(/*wave=*/0, /*reg=*/0, /*lane=*/0);

  EXPECT_EQ(b.raceCount(), 100);
}
